# Copyright (C) 2020 Arm Limited or its affiliates. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the License); you may
# not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an AS IS BASIS, WITHOUT
# WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# Description:
# Compresses and pads the weigths. It also calculates the scales and packs with the biases.
import math
from collections import namedtuple

import numpy as np
from ethosu import mlw_codec

from .architecture_features import Block
from .data_type import DataType
from .errors import UnsupportedFeatureError
from .nn_graph import SchedulingStrategy
from .numeric_util import round_up
from .operation import NpuBlockType
from .scaling import quantise_scale
from .scaling import reduced_quantise_scale
from .tensor import TensorBlockTraversal
from .tensor import TensorFormat
from .tensor import TensorPurpose
from .tensor import TensorSubPurpose


def encode(weight_stream):
    assert np.amin(weight_stream) >= -255
    assert np.amax(weight_stream) <= 255

    # Encode flattened signed weight stream
    compressed = mlw_codec.encode(weight_stream)

    # pad with 0xFF as needed so the length of the weight stream
    # is a multiple of 16

    while (len(compressed) % 16) != 0:
        compressed.append(0xFF)

    return compressed


def generate_brick(arch, brick_weights, ofm_block, block_traversal, ifm_bitdepth):
    is_depthwise = block_traversal == TensorBlockTraversal.DepthWise
    is_partkernel = block_traversal == TensorBlockTraversal.PartKernelFirst
    subkernel_max = arch.subkernel_max
    ofm_ublock = arch.ofm_ublock
    ifm_ublock = arch.ifm_ublock
    # Expect weights formatted HWIO
    ofm_depth = brick_weights.shape[-1]
    ifm_depth = brick_weights.shape[-2]
    kernel_width = brick_weights.shape[-3]
    kernel_height = brick_weights.shape[-4]
    # IFM block depth
    if is_partkernel or (ifm_bitdepth == 16):
        # IFM block depth is always 16 for part-kernel-first
        ifm_block_depth = 16
    elif ifm_bitdepth == 8:
        ifm_block_depth = 32
    else:
        assert False

    stream = []

    # Top level striping - OFM blocks in the entire brick's depth
    for ofm_block_z in range(0, ofm_depth, ofm_block.depth):
        clipped_ofm_block_depth = min(ofm_block.depth, ofm_depth - ofm_block_z)
        # IFM blocks required for the brick
        for ifm_block_z in range(0, (1 if is_depthwise else ifm_depth), ifm_block_depth):
            if is_depthwise:
                clipped_ifm_block_depth = ifm_ublock.depth
            else:
                clipped_ifm_block_depth = (
                    min(ifm_block_depth, ifm_depth - ifm_block_z) if is_partkernel else ifm_block_depth
                )
            # Weight decomposition
            # Subkernel Splitting  (H)
            for subkernel_y in range(0, kernel_height, subkernel_max.height):
                sub_height = min(kernel_height - subkernel_y, subkernel_max.height)
                # Subkernel splitting (W)
                for subkernel_x in range(0, kernel_width, subkernel_max.width):
                    sub_width = min(kernel_width - subkernel_x, subkernel_max.width)
                    subkernel_elements = sub_width * sub_height
                    # Part kernel first works across the kernel H/W and needs padding
                    if is_partkernel:
                        if ifm_bitdepth == 16 and subkernel_elements % 2 != 0:
                            subkernel_elements = int(math.ceil(subkernel_elements / 2) * 2)
                        elif ifm_bitdepth == 8 and subkernel_elements % 4 != 0:
                            subkernel_elements = int(math.ceil(subkernel_elements / 4) * 4)

                    # Depthwise Conv requires multiple of 4 kernel elements in its weight block
                    # this is different from normal conv which is considered "weights depth-first"
                    elif is_depthwise:
                        subkernel_elements = int(math.ceil(subkernel_elements / 4.0) * 4)

                    ifm_block_depth_outer = clipped_ifm_block_depth if is_partkernel else 1
                    ifm_block_depth_inner = 1 if is_partkernel else clipped_ifm_block_depth
                    # IFM Ublocks in IFM-block over depth for part-kernel-first mode
                    # For depth-first IFM Ublocks are traversed after subkernel elements so this loop is ignored.
                    for ifm_ublk_outer in range(0, ifm_block_depth_outer, ifm_ublock.depth):
                        # OFM Ublocks in OFM-block over depth
                        for ofm_ublk in range(0, clipped_ofm_block_depth, ofm_ublock.depth):
                            # HW Kernel element traversal - cannot be a H/W loop due to element
                            # padding requirement on depthwise/part-kernel configurations
                            for element in range(subkernel_elements):
                                kx = element % sub_width
                                ky = element // sub_width
                                # IFM Ublocks in IFM-block over depth (only 1 ublock if depthwise)
                                # In case of part-kernel-first IFM Ublock traversal have already been handled
                                # and this loop is ignored.
                                for ifm_ublk_inner in range(0, ifm_block_depth_inner, ifm_ublock.depth):
                                    # Feed OFM ublock elements
                                    for ofm_ublock_z in range(ofm_ublock.depth):
                                        # Source IFM ublock elements (only 1 element deep if depthwise)
                                        for ifm_ublock_z in range(1 if is_depthwise else ifm_ublock.depth):
                                            # Source position within the current subkernel
                                            wx = subkernel_x + kx
                                            wy = subkernel_y + ky
                                            # Source IFM/OFM slices
                                            ifm_ublk = ifm_ublk_inner + ifm_ublk_outer
                                            ifm_z = ifm_block_z + ifm_ublk + ifm_ublock_z
                                            ofm_z = ofm_block_z + ofm_ublk + ofm_ublock_z
                                            if (ifm_z >= ifm_depth) or (ofm_z >= ofm_depth) or (ky >= sub_height):
                                                stream.append(0)
                                            else:
                                                stream.append(brick_weights[wy][wx][ifm_z][ofm_z])
    return stream


# Compress the weights
def compress_weights(tens, arch, npu_block_type, ofm_block, ofm_depth_step, min_val=None, max_val=None):
    assert tens.purpose == TensorPurpose.Weights
    assert tens.format == TensorFormat.WeightsCompressed

    WeightCompressionConfig = namedtuple("WeightCompressionConfig", ["npu_block_type", "ofm_block", "ofm_depth_step"])

    # check if weights have already been compressed
    wcc = tens.weight_compression_config
    if wcc is not None:
        assert wcc.npu_block_type == npu_block_type, "Weights not used by the same operator type"

        if wcc.ofm_block == ofm_block and wcc.ofm_depth_step == ofm_depth_step:
            return

    assert tens.quantization is not None
    assert tens.quantization.scale_f32 is not None
    assert tens.quantization.zero_point is not None

    zero_point = tens.quantization.zero_point
    quant_buf = tens.quant_values.astype(np.int64)

    # Early zero-point correction
    weights = quant_buf - zero_point

    if len(weights.shape) == 2:
        weights = np.expand_dims(np.expand_dims(weights, axis=0), axis=0)
        weights_shape = (weights.shape[0], 1, 1, weights.shape[1])
    else:
        weights_shape = weights.shape

    compression_scales = []
    compressed_offsets = []
    encoded_streams = []
    offset = 0
    max_single_buffer_len = 0

    ifm_bitdepth = tens.consumer_list[0].inputs[0].dtype.size_in_bits()
    ifm_depth = weights.shape[-2]
    if npu_block_type == NpuBlockType.ConvolutionDepthWise:
        tens.block_traversal = TensorBlockTraversal.DepthWise
    if npu_block_type == NpuBlockType.ConvolutionMxN:
        # Determine which block traversal strategy has better DPU utilization
        kernel_size = weights_shape[0] * weights_shape[1]
        depth_utilization = weights_shape[2] / round_up(weights_shape[2], 32 if ifm_bitdepth == 8 else 16)
        part_kernel_utilization = (weights_shape[2] / round_up(weights_shape[2], 8)) * (
            kernel_size / round_up(kernel_size, 4 if ifm_bitdepth == 8 else 2)
        )
        if part_kernel_utilization >= depth_utilization or ifm_depth <= 8:
            # Part-kernel first is always better for ifm depths <= 8
            tens.block_traversal = TensorBlockTraversal.PartKernelFirst
        else:
            tens.block_traversal = TensorBlockTraversal.DepthFirst

    # Slice weight stream up depth-ways into bricks and compress
    full_ofm_depth = quant_buf.shape[-1]
    for idx in range(0, full_ofm_depth, ofm_depth_step):
        # Get the weights necessary for this brick
        count = min(full_ofm_depth - idx, ofm_depth_step)
        brick_weights = weights[:, :, :, idx : idx + count]

        # Encode all weights into one chunk
        raw_stream = generate_brick(arch, brick_weights, ofm_block, tens.block_traversal, ifm_bitdepth)
        encoded = encode(raw_stream)
        encoded_streams.append(encoded)

        # Remember maximum encoded length for DoubleBuffering
        if max_single_buffer_len < len(encoded):
            max_single_buffer_len = len(encoded)

        # Remember where we put it for linear addressing
        compressed_offsets.append(offset)
        offset += len(encoded)
        assert offset % 16 == 0

        # Compression scale tracking
        compression_scales.append(len(encoded) / len(raw_stream))

    # Also track complete length in the offsets array
    compressed_offsets.append(offset)

    if tens.sub_purpose == TensorSubPurpose.DoubleBuffer and len(encoded_streams) > 2:
        offset = 2 * max_single_buffer_len
        assert offset % 16 == 0

    tens.storage_shape = [1, 1, 1, offset]
    tens.weight_compression_scales = compression_scales
    tens.weight_compression_config = WeightCompressionConfig(npu_block_type, ofm_block, ofm_depth_step)
    tens.weight_compressed_offsets = compressed_offsets
    tens.compression_scale_for_worst_weight_stream = np.amax(compression_scales)
    tens.storage_compression_scale = tens.bandwidth_compression_scale = np.average(compression_scales)
    tens.compressed_values = encoded_streams
    tens.brick_size = (weights_shape[0], weights_shape[1], weights_shape[2], min(tens.shape[-1], ofm_depth_step))


def calc_scales_and_pack_biases(tens, arch, oc_quantum, rescale_for_faf=False):
    assert tens.purpose == TensorPurpose.FeatureMap
    assert tens.format == TensorFormat.NHWC
    # the connected operator should expect a bias input unless it is a FullyConnected
    assert "Bias" in tens.consumer_list[0].type or tens.consumer_list[0].type.startswith("FullyConnected")
    # the input bias tensor is the same as that connected to the operator
    assert tens is tens.consumer_list[0].inputs[2]
    # the operator should only have a single output
    assert len(tens.consumer_list[0].outputs) == 1

    def pack_bias_and_scale(bias, scale, shift):
        bias = np.int64(bias)
        assert -(1 << (40 - 1)) <= bias < (1 << (40 - 1))  # signed 40-bit range
        assert 0 <= scale < (1 << 32)  # unsigned 32-bit range
        assert 0 <= shift < (1 << 6)  # unsigned 6-bit range

        # pack the 80 bit value = [0(2-bits),shift(6-bits),scale(32-bits),bias(40-bits)]
        data = bytearray(10)
        data[0] = (bias >> (0 * 8)) & 0xFF
        data[1] = (bias >> (1 * 8)) & 0xFF
        data[2] = (bias >> (2 * 8)) & 0xFF
        data[3] = (bias >> (3 * 8)) & 0xFF
        data[4] = (bias >> (4 * 8)) & 0xFF
        data[5] = (scale >> (0 * 8)) & 0xFF
        data[6] = (scale >> (1 * 8)) & 0xFF
        data[7] = (scale >> (2 * 8)) & 0xFF
        data[8] = (scale >> (3 * 8)) & 0xFF
        data[9] = shift & 0x3F
        return data

    biases = tens.quant_values

    first_consumer_op = tens.consumer_list[0]
    ifm_dtype = first_consumer_op.inputs[0].dtype
    ifm_scale = first_consumer_op.inputs[0].quantization.scale_f32
    ofm_scale = first_consumer_op.outputs[0].quantization.scale_f32
    weight_scales = first_consumer_op.inputs[1].quantization.scale_f32

    # biases can have multiple consumers for rnn cells. if so, then check that they are all the same
    for op in tens.consumer_list[1:]:
        assert ifm_scale == op.inputs[0].quantization.scale_f32
        assert ofm_scale == op.outputs[0].quantization.scale_f32
        assert weight_scales == op.inputs[1].quantization.scale_f32

    if not hasattr(weight_scales, "__iter__"):
        # If weight_scales is not already an iterable make it into a list
        weight_scales = [weight_scales]

    # Convert scales to np.double (from np.float32) to conform to TensorFlow Lite which
    # uses double during scaling calculations
    # TensorFlow Lite casts the scales slightly differently for uint8 and int8
    if not rescale_for_faf:
        if ifm_dtype == DataType.uint8:
            scales = [np.double(ifm_scale * weight_scale) / np.double(ofm_scale) for weight_scale in weight_scales]
        elif ifm_dtype == DataType.int8 or ifm_dtype == DataType.int16:
            scales = [
                (np.double(ifm_scale) * np.double(weight_scale)) / np.double(ofm_scale)
                for weight_scale in weight_scales
            ]
        else:
            raise UnsupportedFeatureError(
                "Compression of {} is not implemented; tensor: {}".format(ifm_dtype, tens.name)
            )
    else:
        if ifm_dtype == DataType.uint8:
            scales = [np.double(ifm_scale * weight_scale * 0x3000) for weight_scale in weight_scales]
        elif ifm_dtype == DataType.int8 or ifm_dtype == DataType.int16:
            scales = [(np.double(ifm_scale * 0x3000) * np.double(weight_scale)) for weight_scale in weight_scales]
        else:
            raise UnsupportedFeatureError(
                "Compression of {} is not implemented; tensor: {}".format(ifm_dtype, tens.name)
            )

    # quantise all of the weight scales into (scale_factor, shift)
    if ifm_dtype == DataType.int16:
        quantised_scales = [reduced_quantise_scale(scale) for scale in scales]
    else:
        quantised_scales = [quantise_scale(scale) for scale in scales]

    for _, shift in quantised_scales:
        assert shift >= 16

    # pack the biases and scales
    tens.compressed_values = []
    if len(quantised_scales) == 1:
        # If only 1 quantised scale is used, repeat that value for the length of the biases
        quantised_scales = [quantised_scales[0]] * len(biases)

    assert len(quantised_scales) == len(biases)
    for i, bias in enumerate(biases):
        tens.compressed_values.append(pack_bias_and_scale(bias, *quantised_scales[i]))

    tens.element_size_bytes = 10

    # Figure out if we need padded storage (extra whole elements)
    padding = (len(tens.compressed_values) * tens.element_size_bytes) % 16
    if padding != 0:
        padding = 16 - padding

    # This adds enough padding to allow over-reads
    while padding > 0:
        tens.compressed_values.append(pack_bias_and_scale(0, 0, 0))
        padding = padding - tens.element_size_bytes

    tens.storage_shape = [len(tens.compressed_values)]


def update_pass_weight_and_scale_tensors(nng, arch):
    def find_npu_usage_of_tensor(tens):
        # TODO: This function is identical to the one in mark_tensors.py. A common version should be used.
        for op in tens.consumers():
            if op.type == "DMA":
                return find_npu_usage_of_tensor(op.outputs[0])
            if "npu_block_type" in op.attrs:
                return op.attrs["npu_block_type"]
            return NpuBlockType.Default

    for sg in nng.subgraphs:
        for ps in sg.passes:
            if ps.weight_tensor is not None:
                npu_usage_of_tensor = find_npu_usage_of_tensor(ps.weight_tensor)
                if npu_usage_of_tensor == NpuBlockType.ConvolutionDepthWise:
                    ps.weight_tensor.quant_values = np.transpose(ps.weight_tensor.quant_values, (0, 1, 3, 2))
                    ps.weight_tensor.shape = ps.weight_tensor.storage_shape = ps.weight_tensor.bandwidth_shape = list(
                        ps.weight_tensor.quant_values.shape
                    )
                    ps.weight_tensor.weight_transpose_depthwise = True

                needs_dma = len(ps.weight_tensor.ops) == 1 and ps.weight_tensor.ops[0].type == "DMA"
                if ps.cascade.strategy == SchedulingStrategy.WeightStream and needs_dma:
                    ofm_depth_step = ps.block_config[-1]
                else:
                    ofm_depth_step = ps.weight_tensor.shape[-1]

                compress_weights(
                    ps.weight_tensor,
                    arch,
                    npu_usage_of_tensor,
                    Block(ps.block_config[-3], ps.block_config[-4], ps.block_config[-1]),
                    ofm_depth_step,
                )
                # Update source tensor
                if len(ps.weight_tensor.ops) == 1 and ps.weight_tensor.ops[0].type == "DMA":
                    src_tens = ps.weight_tensor.ops[0].inputs[0]
                    src_tens.shape = ps.weight_tensor.shape
                    src_tens.weight_transpose_depthwise = ps.weight_tensor.weight_transpose_depthwise
                    src_tens.quant_values = ps.weight_tensor.quant_values
                    src_tens.compressed_values = ps.weight_tensor.compressed_values
                    src_tens.storage_shape = [1, 1, 1, ps.weight_tensor.weight_compressed_offsets[-1]]
                    src_tens.brick_size = ps.weight_tensor.brick_size
                    src_tens.weight_compression_scales = ps.weight_tensor.weight_compression_scales
                    src_tens.weight_compressed_offsets = ps.weight_tensor.weight_compressed_offsets

            if ps.scale_tensor is not None:
                rescale_for_faf = False
                activation_ops = set(("Sigmoid", "Tanh"))
                if (ps.ops[-1].type in activation_ops) and (ps.npu_block_type != NpuBlockType.ElementWise):
                    rescale_for_faf = True
                calc_scales_and_pack_biases(ps.scale_tensor, arch, ps.block_config[3], rescale_for_faf)
