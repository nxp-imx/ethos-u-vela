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
from collections import namedtuple
from collections import OrderedDict
from typing import Tuple

import numpy as np

from .api import NpuBlockTraversal
from .architecture_features import Accelerator
from .architecture_features import ArchitectureFeatures
from .data_type import DataType
from .errors import UnsupportedFeatureError
from .numeric_util import round_up
from .operation import NpuBlockType
from .operation import Op
from .scaling import quantise_scale
from .scaling import reduced_quantise_scale
from .tensor import Tensor
from .tensor import TensorFormat
from .tensor import TensorPurpose
from ethosu import mlw_codec


# Contains meta info for a weight compression. If two tensors have identical weight compression config,
# then they also will have identical compressed weights.
WeightCompressionConfig = namedtuple(
    "WeightCompressionConfig",
    ["npu_block_type", "ofm_block_depth", "ofm_depth_step", "dilation", "weight_value_id", "scale_value_id"],
)

WeightKey = namedtuple("WeightKey", ["core", "depth"])


class WeightRange:
    def __init__(self):
        self.offset = 0
        self.scale_bytes = 0
        self.weight_offset = 0
        self.weight_bytes = 0
        self.index = 0

    @property
    def total_bytes(self):
        return self.scale_bytes + self.weight_bytes


class NpuWeightTensor(Tensor):
    def __init__(self, name):
        Tensor.__init__(self, None, None, name + "_npu_encoded_weights")
        self.buffer = []
        self.max_range_bytes = 0
        self.encoded_ranges = OrderedDict()
        self.hw_traversal = NpuBlockTraversal.DEPTH_FIRST
        self.dtype = DataType.uint8


class CompressedWeightCache:
    """Global tensor weight compression cache"""

    cache = {}

    @staticmethod
    def get_tensor_with_same_compression(wcc):
        return CompressedWeightCache.cache.get(wcc)

    @staticmethod
    def add(tens):
        # Adds the compressed weights from the tensor to the cache
        wcc = tens.weight_compression_config
        CompressedWeightCache.cache[wcc] = tens

    @staticmethod
    def has_tensor_with_same_compression(wcc):
        return wcc in CompressedWeightCache.cache

    @staticmethod
    def get_unencoded_size_with_same_compression(wcc):
        cache_obj = CompressedWeightCache.cache.get(wcc)
        return cache_obj[1] if cache_obj else None


def create_weight_compression_config(
    weight_tens, scale_tens, npu_block_type, ofm_block_depth, ofm_depth_step, dilation
):
    # Note: for an ofm block only its depth is used in weight compression.
    # And block depth > ofm depth gives same result as block depth == ofm depth
    block_depth = min(ofm_block_depth, weight_tens.quant_values.shape[-1])
    return WeightCompressionConfig(
        npu_block_type, block_depth, ofm_depth_step, dilation, weight_tens.value_id, scale_tens.value_id
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

    :return: a tuple with a bytearray of encoded weights and the size of the unencoded weights
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
    decomp_h = ArchitectureFeatures.SubKernelMax.height // dilation_xy[0]
    decomp_w = ArchitectureFeatures.SubKernelMax.width // dilation_xy[1]

    return mlw_codec.reorder_encode(
        ifm_ublock.depth,
        ofm_ublock.depth,
        weights_volume,
        ofm_block_depth,
        is_depthwise,
        block_traversal == NpuBlockTraversal.PART_KERNEL_FIRST,
        ifm_bitdepth,
        decomp_h,
        decomp_w,
    )


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


def core_deinterleave(hwio, core, ncores):
    # Put weights back into OHWI
    ohwi = np.transpose(hwio, (3, 0, 1, 2))
    return ohwi[core : ohwi.shape[0] : ncores]


def _prepare_scale_and_bias(arch, tens, rescale_for_faf):
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

    # If only 1 quantised scale is used, repeat that value for the length of the biases
    if len(quantised_scales) == 1:
        quantised_scales = [quantised_scales[0]] * len(biases)

    return quantised_scales, biases


def encode_weight_and_scale_tensor(
    arch, op, weight_tens, scale_tens, kernel, block_config, depth_offsets, rescale_for_faf=False
) -> NpuWeightTensor:
    npu_block_type = op.type.npu_block_type

    wcc = create_weight_compression_config(
        weight_tens, scale_tens, npu_block_type, block_config.ofm_block.depth, hash(str(depth_offsets)), kernel.dilation
    )

    tens_cached = CompressedWeightCache.get_tensor_with_same_compression(wcc)
    if tens_cached is not None:
        return tens_cached

    npu_tensor = NpuWeightTensor(weight_tens.name)
    npu_tensor.weight_compression_config = wcc

    # No cache hit, perform the compression
    assert weight_tens.quantization is not None
    assert weight_tens.quantization.scale_f32 is not None
    assert weight_tens.quantization.zero_point is not None

    zero_point = weight_tens.quantization.zero_point
    quant_buf = weight_tens.quant_values.astype(np.int64)

    # Early zero-point correction
    weights = quant_buf - zero_point

    if len(weights.shape) == 2:
        weights = np.expand_dims(np.expand_dims(weights, axis=0), axis=0)

    # Expect this (undilated) equivalence
    assert kernel.height == weights.shape[0]
    assert kernel.width == weights.shape[1]
    # Ensure depth offsets are terminated at end of OFM shape
    assert len(depth_offsets) > 1, "Require closed depth ranges"

    ifm_bitdepth = op.inputs[0].dtype.size_in_bits()
    ifm_depth = weights.shape[-2]

    # Default HW traversal
    npu_tensor.hw_traversal = NpuBlockTraversal.DEPTH_FIRST

    if npu_block_type == NpuBlockType.ConvolutionMxN:
        # Determine which block traversal strategy has better DPU utilization
        kernel_size = weights.shape[0] * weights.shape[1]
        depth_utilization = weights.shape[2] / round_up(weights.shape[2], 32 if ifm_bitdepth == 8 else 16)
        part_kernel_utilization = (weights.shape[2] / round_up(weights.shape[2], 8)) * (
            kernel_size / round_up(kernel_size, 4 if ifm_bitdepth == 8 else 2)
        )
        if part_kernel_utilization >= depth_utilization or ifm_depth <= 8:
            # Part-kernel first is always better for ifm depths <= 8
            npu_tensor.hw_traversal = NpuBlockTraversal.PART_KERNEL_FIRST

    if op.type == Op.Conv2DBackpropInputSwitchedBias:
        # Transpose Convoluion, reverse weights in H and W axes
        weights = np.flip(weights, axis=(0, 1))

    encoded_stream = bytearray()
    max_single_buffer_len = 0
    is_depthwise = npu_block_type == NpuBlockType.ConvolutionDepthWise

    # Bias & scale
    if scale_tens:
        quantised_scales, biases = _prepare_scale_and_bias(arch, scale_tens, rescale_for_faf)
        scale_tens.element_size_bytes = 10

    # Slice the weight stream up depth-ways into bricks and compress
    full_ofm_depth = quant_buf.shape[-1]
    ofm_block_depth = block_config.ofm_block.depth

    weight_range_index = 0
    for idx, depth_offset in enumerate(depth_offsets[:-1]):
        # Do not generate for offsets outside the OFM
        assert depth_offset >= 0 and depth_offset < full_ofm_depth
        depth_length = depth_offsets[idx + 1] - depth_offset

        # Get the weights necessary for this brick
        brick_weights = weights[:, :, :, depth_offset : depth_offset + depth_length]

        buffer_start_offset = len(encoded_stream)

        # For each core, deinterleave weights from the larger volume
        # and generate separate compressed streams.
        for core in range(0, min(arch.ncores, full_ofm_depth)):

            core_block_depth = int((ofm_block_depth + arch.ncores - 1 - core) // arch.ncores)

            if core_block_depth != 0:
                key = WeightKey(core, depth_offset)
                weight_range = WeightRange()
                weight_range.offset = len(encoded_stream)
                weight_range.index = weight_range_index
                weight_range_index += 1

                # Scales & biases
                if scale_tens:
                    scale_stream = []
                    core_scales = quantised_scales[
                        depth_offset + core : depth_offset + core + depth_length : arch.ncores
                    ]
                    core_biases = biases[depth_offset + core : depth_offset + core + depth_length : arch.ncores]
                    for j, core_bias in enumerate(core_biases):
                        scale_stream.extend(encode_bias(np.int64(core_bias), *core_scales[j]))

                    weight_range.scale_bytes = len(scale_stream)

                    encoded_stream.extend(scale_stream)

                    # Align to 16 for start of next substream
                    remainder = len(encoded_stream) % 16
                    if remainder > 0:
                        encoded_stream.extend(bytearray(16 - remainder))

                # Weights
                core_weights = core_deinterleave(brick_weights, core, arch.ncores)
                encoded_substream, _ = encode_weights(
                    accelerator=arch.accelerator_config,
                    weights_volume=core_weights,
                    dilation_xy=kernel.dilation,
                    ifm_bitdepth=ifm_bitdepth,
                    ofm_block_depth=core_block_depth,
                    is_depthwise=is_depthwise,
                    block_traversal=npu_tensor.hw_traversal,
                )

                weight_range.weight_offset = len(encoded_stream) - weight_range.offset
                weight_range.weight_bytes = len(encoded_substream)

                # Append encoded weights section
                encoded_stream.extend(encoded_substream)
                assert len(encoded_stream) % 16 == 0

                # Record encoded range in weights tensor
                npu_tensor.encoded_ranges[key] = weight_range

        # Remember maximum encoded length for DoubleBuffering
        max_single_buffer_len = max(max_single_buffer_len, len(encoded_stream) - buffer_start_offset)

    npu_tensor.buffer = encoded_stream
    npu_tensor.max_range_bytes = max_single_buffer_len
    npu_tensor.set_all_shapes([1, 1, 1, len(encoded_stream)])
    npu_tensor.format = TensorFormat.WeightsCompressed
    npu_tensor.purpose = TensorPurpose.Weights
    npu_tensor.mem_area = weight_tens.mem_area
    npu_tensor.mem_type = weight_tens.mem_type
    CompressedWeightCache.add(npu_tensor)
    return npu_tensor
