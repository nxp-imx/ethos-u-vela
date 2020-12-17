# Copyright (C) 2020-2021 Arm Limited or its affiliates. All rights reserved.
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
from typing import Tuple

import numpy as np

from .api import NpuBlockTraversal
from .architecture_features import Accelerator
from .architecture_features import ArchitectureFeatures
from .data_type import DataType
from .errors import UnsupportedFeatureError
from .nn_graph import SchedulingStrategy
from .numeric_util import round_up
from .numeric_util import round_up_divide
from .operation import NpuBlockType
from .operation import Op
from .scaling import quantise_scale
from .scaling import reduced_quantise_scale
from .tensor import create_equivalence_id
from .tensor import TensorBlockTraversal
from .tensor import TensorFormat
from .tensor import TensorPurpose
from .tensor import TensorSubPurpose
from ethosu import mlw_codec


# Contains meta info for a weight compression. If two tensors have identical weight compression config,
# then they also will have identical compressed weights.
WeightCompressionConfig = namedtuple(
    "WeightCompressionConfig", ["npu_block_type", "ofm_block_depth", "ofm_depth_step", "dilation", "value_id"]
)


def encode_weights(
    accelerator: Accelerator,
    weights_volume: np.ndarray,
    dilation_xy: Tuple[int, int],
    ifm_bitdepth: int,
    ofm_block_depth: int,
    is_depthwise: bool,
    block_traversal: NpuBlockTraversal,
):
    """
    Internal implementation of the public facing API to use weight encoding.

    :param accelerator: architecture_features.Accelerator enum to pick the correct Ethos-U accelerator
    :param weights_volume: numpy.ndarray in OHWI layout with a shape of four
    :param dilation_xy: a two element tuple of dilation attributes in x,y dimension
    :param ifm_bitdepth: the bitdepth of input feature map
    :param ofm_block_depth: the depth of blocks for Ethos-U processing
    :param is_depthwise: a boolean indicating these weights are used for a depthwise traversal
    :param block_traversal: indicates how these weights are traversed on sub-kernel basis

    :return: a bytearray of compressed weights
    """
    # Check arg types
    assert isinstance(accelerator, Accelerator)
    assert isinstance(weights_volume, np.ndarray)
    assert isinstance(dilation_xy, tuple)
    assert isinstance(ifm_bitdepth, int)
    assert isinstance(ofm_block_depth, int)
    assert isinstance(is_depthwise, bool)
    assert isinstance(block_traversal, NpuBlockTraversal)

    # Checks for weight layout
    assert len(weights_volume.shape) == 4, "weights ndarray should have a shape of 4"

    # It cannot be both partkernel and depthwise
    assert not (
        is_depthwise and block_traversal == NpuBlockTraversal.PART_KERNEL_FIRST
    ), "encode_weights :: partkernel and depthwise are mutually exclusive"

    # Check valid values for dilation
    assert dilation_xy[0] in (1, 2), "encode_weights :: dilation x should be 1 or 2 not {}".format(dilation_xy[0])
    assert dilation_xy[1] in (1, 2), "encode_weights :: dilation y should be 1 or 2 not {}".format(dilation_xy[1])

    ifm_ublock = ArchitectureFeatures.accelerator_configs[accelerator].ifm_ublock
    ofm_ublock = ArchitectureFeatures.accelerator_configs[accelerator].ofm_ublock
    raw_stream = generate_brick(
        ifm_ublock=ifm_ublock,
        ofm_ublock=ofm_ublock,
        brick_weights=weights_volume,
        ofm_block_depth=ofm_block_depth,
        is_depthwise=is_depthwise,
        is_partkernel=block_traversal == NpuBlockTraversal.PART_KERNEL_FIRST,
        ifm_bitdepth=ifm_bitdepth,
        dilation=dilation_xy,
    )
    encoded_stream = encode(raw_stream)
    return encoded_stream


def encode_bias(bias: np.int64, scale: int, shift: int):
    """
    Internal implementation of public facing API to pack bias and scale values as required by the Ethos-U

    :param bias: 64bit signed number that includes 40bit signed bias
    :param scale: 32bit scale value
    :param shift: 6bit shift value
    :return: packed 80bit [0(2-bits),shift(6-bits),scale(32-bits),bias(40-bits)]
    """
    # Check arg types
    assert isinstance(bias, np.int64)
    assert isinstance(scale, int)
    assert isinstance(shift, int)

    assert -(1 << (40 - 1)) <= bias < (1 << (40 - 1))  # signed 40-bit range
    assert 0 <= scale < (1 << 32)  # unsigned 32-bit range
    assert 0 <= shift < (1 << 6)  # unsigned 6-bit range

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


def create_weight_compression_config(tens, npu_block_type, ofm_block_depth, ofm_depth_step, dilation):
    # Note: for an ofm block only its depth is used in weight compression.
    # And block depth > ofm depth gives same result as block depth == ofm depth
    block_depth = min(ofm_block_depth, tens.quant_values.shape[-1])
    return WeightCompressionConfig(npu_block_type, block_depth, ofm_depth_step, dilation, tens.value_id)


def set_storage_shape(tens):
    # Sets the storage shape depending on the tensor's sub purpose
    if tens.sub_purpose == TensorSubPurpose.DoubleBuffer and len(tens.compressed_values) > 2:
        offset = 2 * np.amax([len(x) for x in tens.compressed_values])
        assert offset % 16 == 0
    else:
        offset = tens.weight_compressed_offsets[-1]
    tens.storage_shape = [1, 1, 1, offset]


class CompressedWeightCache:
    # Contains weight compressions for all weight tensors in a graph
    def __init__(self):
        self.cache = {}  # maps from WeightCompressionConfig to a tensor clone containing compressed weights

    def get_tensor_with_same_compression(self, wcc):
        return self.cache.get(wcc)

    def add(self, tens):
        # Adds the compressed weights from the tensor to the cache
        wcc = tens.weight_compression_config
        # Clone the tensor to make sure that nothing related to the weight compression is modified
        tens_clone = tens.clone("_weights{}_{}".format(wcc.ofm_block_depth, wcc.ofm_depth_step))
        self.cache[wcc] = tens_clone


def encode(weight_stream):
    if len(weight_stream) == 0:
        return []
    assert np.amin(weight_stream) >= -255
    assert np.amax(weight_stream) <= 255

    # Encode flattened signed weight stream
    compressed = mlw_codec.encode(weight_stream)

    # pad with 0xFF as needed so the length of the weight stream
    # is a multiple of 16

    while (len(compressed) % 16) != 0:
        compressed.append(0xFF)

    return compressed


def generate_brick(
    ifm_ublock, ofm_ublock, brick_weights, ofm_block_depth, is_depthwise, is_partkernel, ifm_bitdepth, dilation
):

    decomp_h = ArchitectureFeatures.SubKernelMax.height // dilation[0]
    decomp_w = ArchitectureFeatures.SubKernelMax.width // dilation[1]
    # Expect weights formatted OHWI
    ofm_depth = brick_weights.shape[-4]
    ifm_depth = brick_weights.shape[-1]
    kernel_width = brick_weights.shape[-2]
    kernel_height = brick_weights.shape[-3]
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
    for ofm_block_z in range(0, ofm_depth, ofm_block_depth):
        clipped_ofm_block_depth = min(ofm_block_depth, ofm_depth - ofm_block_z)
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
            for subkernel_y in range(0, kernel_height, decomp_h):
                sub_height = min(kernel_height - subkernel_y, decomp_h)
                # Subkernel splitting (W)
                for subkernel_x in range(0, kernel_width, decomp_w):
                    sub_width = min(kernel_width - subkernel_x, decomp_w)
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
                                                stream.append(brick_weights[ofm_z][wy][wx][ifm_z])
    return stream


def core_deinterleave(hwio, core, ncores):
    # Put weights back into OHWI
    ohwi = np.transpose(hwio, (3, 0, 1, 2))
    return ohwi[core : ohwi.shape[0] : ncores]


# Compress the weights
def compress_weights(arch, nng, tens, npu_block_type, ofm_block_depth, ofm_depth_step, dilation):
    assert tens.purpose == TensorPurpose.Weights

    # Check the weight cache
    if nng.weight_cache is None:
        nng.weight_cache = CompressedWeightCache()
    wcc = create_weight_compression_config(tens, npu_block_type, ofm_block_depth, ofm_depth_step, dilation)
    tens.weight_compression_config = wcc
    # Reassign equivalence id such that tensors with same weight compression get identical equivalence ids,
    # but tensors with the same values but different compression get different equivalence ids
    tens.equivalence_id = create_equivalence_id(wcc)
    tens_cached = nng.weight_cache.get_tensor_with_same_compression(wcc)
    if tens_cached is not None:
        # Cache hit, copy weights from the cache
        tens.copy_compressed_weight_info(tens_cached)
        set_storage_shape(tens)
        return
    # No cache hit, perform the compression
    assert tens.quantization is not None
    assert tens.quantization.scale_f32 is not None
    assert tens.quantization.zero_point is not None

    zero_point = tens.quantization.zero_point
    quant_buf = tens.quant_values.astype(np.int64)

    # Early zero-point correction
    weights = quant_buf - zero_point

    if len(weights.shape) == 2:
        weights = np.expand_dims(np.expand_dims(weights, axis=0), axis=0)

    compression_scales = []
    compressed_offsets = []
    encoded_streams = []
    encoded_streams_substream_offsets = []
    offset = 0
    max_single_buffer_len = 0

    ifm_bitdepth = tens.consumer_list[0].inputs[0].dtype.size_in_bits()
    ifm_depth = weights.shape[-2]
    if npu_block_type == NpuBlockType.ConvolutionDepthWise:
        tens.block_traversal = TensorBlockTraversal.DepthWise
    if npu_block_type == NpuBlockType.ConvolutionMxN:
        # Determine which block traversal strategy has better DPU utilization
        kernel_size = weights.shape[0] * weights.shape[1]
        depth_utilization = weights.shape[2] / round_up(weights.shape[2], 32 if ifm_bitdepth == 8 else 16)
        part_kernel_utilization = (weights.shape[2] / round_up(weights.shape[2], 8)) * (
            kernel_size / round_up(kernel_size, 4 if ifm_bitdepth == 8 else 2)
        )
        if part_kernel_utilization >= depth_utilization or ifm_depth <= 8:
            # Part-kernel first is always better for ifm depths <= 8
            tens.block_traversal = TensorBlockTraversal.PartKernelFirst
        else:
            tens.block_traversal = TensorBlockTraversal.DepthFirst

    is_depthwise = tens.block_traversal == TensorBlockTraversal.DepthWise
    if tens.block_traversal == TensorBlockTraversal.PartKernelFirst:
        block_traversal = NpuBlockTraversal.PART_KERNEL_FIRST
    else:
        block_traversal = NpuBlockTraversal.DEPTH_FIRST

    if tens.consumer_list[0].type == Op.Conv2DBackpropInputSwitchedBias:
        # Transpose Convoluion, reverse weights in H and W axes
        weights = np.flip(weights, axis=(0, 1))

    # Calculate brick size
    brick_size = (weights.shape[0], weights.shape[1], weights.shape[2], min(tens.shape[-1], ofm_depth_step))
    elements_in_brick = np.prod(brick_size)

    # Slice weight stream up depth-ways into bricks and compress
    full_ofm_depth = quant_buf.shape[-1]
    for idx in range(0, full_ofm_depth, ofm_depth_step):
        # Get the weights necessary for this brick
        count = min(full_ofm_depth - idx, ofm_depth_step)
        brick_weights = weights[:, :, :, idx : idx + count]

        substream_offsets = [0]
        encoded_stream = []

        # For each core, deinterleave weights from the larger volume
        # and generate separate compressed streams.
        for core in range(0, min(arch.ncores, full_ofm_depth)):
            core_weights = core_deinterleave(brick_weights, core, arch.ncores)

            block_depth = (ofm_block_depth + arch.ncores - 1 - core) // arch.ncores
            encoded_substream = []
            if block_depth != 0:
                encoded_substream = encode_weights(
                    accelerator=arch.accelerator_config,
                    weights_volume=core_weights,
                    dilation_xy=dilation,
                    ifm_bitdepth=ifm_bitdepth,
                    ofm_block_depth=block_depth,
                    is_depthwise=is_depthwise,
                    block_traversal=block_traversal,
                )
            encoded_stream.extend(encoded_substream)
            substream_offsets.append(len(encoded_stream))

        encoded_streams.append(encoded_stream)
        encoded_streams_substream_offsets.append(substream_offsets)

        # Remember maximum encoded length for DoubleBuffering
        max_single_buffer_len = max(max_single_buffer_len, len(encoded_stream))

        # Remember where we put it for linear addressing
        compressed_offsets.append(offset)
        offset += len(encoded_stream)
        assert offset % 16 == 0

        # Compression scale tracking
        compression_scales.append(len(encoded_stream) / elements_in_brick)

    # Track total length as last element of the offsets array
    compressed_offsets.append(offset)

    tens.weight_compression_scales = compression_scales
    tens.weight_compressed_offsets = compressed_offsets
    tens.compression_scale_for_worst_weight_stream = np.amax(compression_scales)
    tens.storage_compression_scale = tens.bandwidth_compression_scale = np.average(compression_scales)
    tens.compressed_values = encoded_streams
    tens.compressed_values_substream_offsets = encoded_streams_substream_offsets
    tens.brick_size = brick_size
    set_storage_shape(tens)
    nng.weight_cache.add(tens)


def calc_scales_and_pack_biases(tens, arch, ofm_depth_step, rescale_for_faf=False):
    assert tens.purpose in [TensorPurpose.FeatureMap, TensorPurpose.FSBias]
    assert tens.format == TensorFormat.NHWC
    # the connected operator should expect a bias input unless it is a FullyConnected
    assert tens.consumer_list[0].type.needs_bias()
    # the input bias tensor is the same as that connected to the operator
    bias_tens = tens.consumer_list[0].bias
    assert tens is bias_tens

    # the operator should only have a single output
    assert len(tens.consumer_list[0].outputs) == 1
    biases = tens.quant_values

    first_consumer_op = tens.consumer_list[0]
    ifm_dtype = first_consumer_op.inputs[0].dtype
    ifm_scale = first_consumer_op.get_input_quantization().scale_f32
    ofm_scale = first_consumer_op.get_output_quantization().scale_f32
    weight_scales = first_consumer_op.inputs[1].quantization.scale_f32

    # biases can have multiple consumers for rnn cells. if so, then check that they are all the same
    for op in tens.consumer_list[1:]:
        assert ifm_scale == op.get_input_quantization().scale_f32
        assert ofm_scale == op.get_output_quantization().scale_f32
        assert weight_scales == op.inputs[1].quantization.scale_f32

    if not hasattr(weight_scales, "__iter__"):
        # If weight_scales is not already an iterable make it into a list
        weight_scales = [weight_scales]

    # Convert scales to np.double (from np.float32) to conform to TensorFlow Lite which
    # uses double during scaling calculations
    # TensorFlow Lite casts the scales slightly differently for uint8 and int8
    if not rescale_for_faf:
        if ifm_dtype == DataType.uint8:
            # for some cases of the Mean operator, the scale must be calculated differently to match reference
            if first_consumer_op.low_precision_scaling:
                scales = [
                    np.double(np.single(ifm_scale) / (np.single(weight_scale) * np.single(ofm_scale)))
                    for weight_scale in weight_scales
                ]
            else:
                scales = [np.double(ifm_scale * weight_scale) / np.double(ofm_scale) for weight_scale in weight_scales]
        elif ifm_dtype == DataType.int8 or ifm_dtype == DataType.int16:
            scales = [
                (np.double(ifm_scale) * np.double(weight_scale)) / np.double(ofm_scale)
                for weight_scale in weight_scales
            ]
        else:
            raise UnsupportedFeatureError(f"Compression of {ifm_dtype} is not implemented; Tensor: '{tens.name}'")
    else:
        if ifm_dtype == DataType.uint8:
            scales = [np.double(ifm_scale * weight_scale * 0x3000) for weight_scale in weight_scales]
        elif ifm_dtype == DataType.int8 or ifm_dtype == DataType.int16:
            scales = [(np.double(ifm_scale * 0x3000) * np.double(weight_scale)) for weight_scale in weight_scales]
        else:
            raise UnsupportedFeatureError(f"Compression of {ifm_dtype} is not implemented; Tensor: '{tens.name}'")

    # quantise all of the weight scales into (scale_factor, shift)
    if ifm_dtype == DataType.int16:
        quantised_scales = [reduced_quantise_scale(scale) for scale in scales]
    else:
        quantised_scales = [quantise_scale(scale) for scale in scales]

    # pack the biases and scales
    if len(quantised_scales) == 1:
        # If only 1 quantised scale is used, repeat that value for the length of the biases
        quantised_scales = [quantised_scales[0]] * len(biases)

    assert len(quantised_scales) == len(biases)
    tens.element_size_bytes = 10
    tens.compressed_values = []
    tens.compressed_values_substream_offsets = []

    total_elements = len(quantised_scales)
    alignment_bytes = 0
    for i in range(0, total_elements, ofm_depth_step):
        # Extract streams from brick to generate substreams for each core
        stream = bytearray()
        substream_offsets = [0]
        max_len = min(ofm_depth_step, total_elements - i)
        for core in range(0, min(arch.ncores, max_len)):
            core_scales = quantised_scales[i + core : i + core + max_len : arch.ncores]
            core_biases = biases[i + core : i + core + max_len : arch.ncores]
            for j, core_bias in enumerate(core_biases):
                stream.extend(encode_bias(np.int64(core_bias), *core_scales[j]))

            # Align to 16 for start for next substream
            remainder = (len(stream)) % 16
            if remainder > 0:
                stream.extend(bytearray(16 - remainder))
                alignment_bytes += 16 - remainder

            substream_offsets.append(len(stream))

        # Add to compressed values with their substream offset lists to the tensor
        tens.compressed_values.append(stream)
        tens.compressed_values_substream_offsets.append(substream_offsets)

    tens.storage_shape = [total_elements + round_up_divide(alignment_bytes, tens.element_size_bytes)]


def update_pass_weight_and_scale_tensors(nng, arch):
    for sg in nng.subgraphs:
        for ps in sg.passes:
            tens = ps.weight_tensor
            if tens is not None:
                op = tens.find_npu_op()
                if op is None:
                    continue
                needs_dma = tens.needs_dma()
                if ps.cascade.strategy == SchedulingStrategy.WeightStream and needs_dma:
                    ofm_depth_step = ps.block_config[-1]
                else:
                    ofm_depth_step = tens.shape[-1]
                compress_weights(
                    arch, nng, tens, op.type.npu_block_type, ps.block_config[-1], ofm_depth_step, op.get_dilation_h_w()
                )
                nng.total_compressed_weights += tens.weight_compressed_offsets[-1]
                nng.total_original_weights += tens.elements() * tens.element_size()

                # Update source tensor
                if needs_dma:
                    src_tens = tens.get_dma_src_tensor()
                    src_tens.shape = tens.shape
                    src_tens.quant_values = tens.quant_values
                    src_tens.copy_compressed_weight_info(tens)
                    set_storage_shape(src_tens)

            if ps.scale_tensor is not None:
                rescale_for_faf = False
                if (ps.ops[-1].type in (Op.Sigmoid, Op.Tanh)) and (ps.npu_block_type != NpuBlockType.ElementWise):
                    rescale_for_faf = True
                calc_scales_and_pack_biases(ps.scale_tensor, arch, ofm_depth_step, rescale_for_faf)
                if ps.scale_tensor.ops[0].type == Op.DMA:
                    src_tens = ps.scale_tensor.get_dma_src_tensor()
                    src_tens.shape = ps.scale_tensor.shape
                    src_tens.quant_values = ps.scale_tensor.quant_values
                    src_tens.element_size_bytes = ps.scale_tensor.element_size_bytes
                    src_tens.copy_compressed_weight_info(ps.scale_tensor)
