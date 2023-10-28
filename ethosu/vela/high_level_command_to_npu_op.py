# SPDX-FileCopyrightText: Copyright 2020-2023 Arm Limited and/or its affiliates <open-source-office@arm.com>
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
#
# Description:
# Conversion from high level command to NpuOperation
from enum import IntEnum
from typing import cast
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple

from .api import NpuActivation
from .api import NpuActivationOp
from .api import NpuAddressRange
from .api import NpuBlockOperation
from .api import NpuBlockTraversal
from .api import NpuConv2DOperation
from .api import NpuConvDepthWiseOperation
from .api import NpuDataType
from .api import NpuDmaOperation
from .api import NpuElementWiseOp
from .api import NpuElementWiseOperation
from .api import NpuFeatureMap
from .api import NpuLayout
from .api import NpuOperation
from .api import NpuOperationType
from .api import NpuPadding
from .api import NpuPoolingOp
from .api import NpuPoolingOperation
from .api import NpuQuantization
from .api import NpuResamplingMode
from .api import NpuRoundingMode
from .api import NpuShape3D
from .api import NpuTileBox
from .architecture_features import ArchitectureFeatures
from .data_type import DataType
from .debug_database import DebugDatabase
from .errors import UnsupportedFeatureError
from .ethos_u55_regs.ethos_u55_regs import resampling_mode
from .high_level_command_stream import Box
from .high_level_command_stream import Command
from .high_level_command_stream import DMA
from .high_level_command_stream import NOP
from .high_level_command_stream import NpuStripe
from .numeric_util import quantise_float32
from .numeric_util import round_up
from .operation import NpuBlockType
from .operation import Op
from .operation import Operation
from .operation import Padding
from .operation import RoundingMode
from .register_command_stream_generator import generate_command_stream
from .register_command_stream_util import BASE_PTR_INDEX_MEM2MEM
from .register_command_stream_util import to_npu_kernel
from .register_command_stream_util import UNARY_ELEMWISE_OPS
from .shape4d import Shape4D
from .tensor import MemType
from .tensor import Tensor
from .tensor import TensorFormat
from .tensor import TensorPurpose
from .weight_compressor import NpuWeightTensor
from .weight_compressor import WeightKey


class BasePointerIndex(IntEnum):
    WeightTensor = 0  # base address index for the Weight tensor
    ScratchTensor = 1  # base address index for the Scratch_tensor in the TensorArena
    ScratchFastTensor = 2  # base address for the Scratch_fast_tensor


dtype_map = {
    DataType.uint8: NpuDataType.UINT8,
    DataType.int8: NpuDataType.INT8,
    DataType.uint16: NpuDataType.UINT16,
    DataType.int16: NpuDataType.INT16,
    DataType.int32: NpuDataType.INT32,
}


# Maps an elementwise op type to an elementwise_mode enum value used by NPU_OP_ELEMENTWISE
elementwise_op_map = {
    Op.Mul: NpuElementWiseOp.MUL,
    Op.Add: NpuElementWiseOp.ADD,
    Op.Sub: NpuElementWiseOp.SUB,
    Op.Minimum: NpuElementWiseOp.MIN,
    Op.Maximum: NpuElementWiseOp.MAX,
    Op.LeakyRelu: NpuElementWiseOp.LRELU,
    Op.Abs: NpuElementWiseOp.ABS,
    Op.CLZ: NpuElementWiseOp.CLZ,
    Op.SHR: NpuElementWiseOp.SHR,
    Op.SHL: NpuElementWiseOp.SHL,
}


# inverse of the resampling_mode_map in the register command stream generator
resampling_mode_inv_map = {
    resampling_mode.NONE: NpuResamplingMode.NONE,
    resampling_mode.NEAREST: NpuResamplingMode.NEAREST,
    resampling_mode.TRANSPOSE: NpuResamplingMode.TRANSPOSE,
}


rounding_mode_map = {
    RoundingMode.TFLite: NpuRoundingMode.TFL,
    RoundingMode.ToZero: NpuRoundingMode.TRUNCATE,
    RoundingMode.HalfUp: NpuRoundingMode.NATURAL,
    RoundingMode.AwayZero: NpuRoundingMode.NATURAL,
}


def ifm_ifm2_correct_order(ifm_shape: Shape4D, ifm2_shape: Shape4D) -> bool:

    if ifm_shape is None:
        # Scalar needs to be in IFM2
        return False
    if ifm2_shape is None:
        return True

    for ifm, ifm2 in zip(ifm_shape.as_list(), ifm2_shape.as_list()):
        if ifm != ifm2 and ifm == 1:
            # Broadcasted FM needs to be in IFM2
            return False
    return True


def get_rounding_mode(op: Operation, fused_quantize: bool) -> NpuRoundingMode:
    """Specifies type of rounding to be used"""
    rounding_mode = NpuRoundingMode.TFL
    if op.type.is_resize_op():
        rounding_mode = NpuRoundingMode.NATURAL
    elif (
        op.original_type.npu_block_type in (NpuBlockType.ConvolutionMxN, NpuBlockType.ConvolutionDepthWise)
        and op.ifm.dtype == DataType.int16
    ):
        rounding_mode = NpuRoundingMode.NATURAL
    elif (
        not fused_quantize
        and op.type.is_avgpool_op()
        and op.memory_function == Op.ConcatSliceWrite
        and op.kernel.elements_wh() == 1
    ):
        rounding_mode = NpuRoundingMode.NATURAL
    if op.rounding_mode is not None:
        rounding_mode = rounding_mode_map[op.rounding_mode]
    return rounding_mode


def create_padding(cmd: NpuStripe, primary_op: Operation, npu_op: NpuBlockOperation) -> NpuPadding:
    if primary_op.type.npu_block_type == NpuBlockType.VectorProduct:
        return NpuPadding(top=0, left=0, bottom=0, right=0)
    top, left, bottom, right = primary_op.attrs["explicit_padding"]

    # Check if this is for horizontal ifm streaming
    if not (cmd.is_first_h_stripe and cmd.is_last_h_stripe):
        top = cmd.pad_top
        bottom = cmd.pad_bottom

    # the ifm box coordinate range depends upon whether the primary op was combined with a split slice read
    ifm_read_offset = primary_op.read_offsets[0]
    ifm_read_shape = primary_op.read_shapes[0]
    if ifm_read_offset is None or len(ifm_read_offset) < 2:
        box_start_coord_min = 0
        box_end_coord_max = cmd.ps.ifm_shapes[0].width
    else:
        box_start_coord_min = ifm_read_offset[-2]
        box_end_coord_max = ifm_read_shape[-2]

    # Indexing from end since a 1x1 Avgpool might have been added with non 4-dimensional input/output,
    # because of activation function needed to be fused.
    if len(cmd.ifm_box.start_coord) >= 2 and cmd.ifm_box.start_coord[-2] > box_start_coord_min:
        left = 0
    if len(cmd.ifm_box.end_coord) >= 2 and cmd.ifm_box.end_coord[-2] < box_end_coord_max:
        right = 0

    # If tile padding is selected, modify the tile base addresses and set NpuPadding to zero.
    if primary_op.attrs.get("padding", None) == Padding.TILE:
        assert cmd.ifm_tensor.format == TensorFormat.NHCWB16, "Tensor format NHCWB16 required to perform tile padding"
        assert npu_op.op_type == NpuOperationType.ConvDepthWise, "Tile padding only supported for depthwise convolution"
        assert npu_op.ifm is not None, "Feature map must be initialized to modify the tile addresses"
        npu_op.ifm.tiles = modify_tile_addresses_for_padding(
            npu_op.ifm.tiles,
            primary_op.attrs.get("explicit_padding", None),
            channels=cmd.ps.ifm_shapes[0].depth,
            dtype=cmd.ifm_tensor.dtype,
        )
        top, left, bottom, right = 0, 0, 0, 0

    return NpuPadding(top=top, left=left, bottom=bottom, right=right)


def modify_tile_addresses_for_padding(
    tile_box: NpuTileBox, padding_direction: List[int], channels: int, dtype: DataType
) -> NpuTileBox:
    # Addresses are 16-bytes aligned when using the NHCWB16 format, which is required to utilize tiling
    # Calculate the offset to top right, bottom left and bottom right element in the IFM (top left offset is 0)
    """
    Example: 4x4x1 IFM
    |  a b c d | <-- Offset to TR ('d') is (w0-1) = 3
    |  e f g h |
    |  i j k l |
    |  m n o p | <-- Offset to BL ('m') is (w0*(h0-1)) = 12 and to BR ('p') ((w0*h0)-1) = 15
    """
    h0, h1, w0, addresses = tile_box
    elem_size = 2 if dtype == DataType.int16 else 1
    tr_offset = (w0 - 1) * 16 * elem_size
    bl_offset = w0 * (h0 - 1) * 16 * (round_up(channels, 16) // 16) * elem_size
    br_offset = tr_offset + bl_offset

    # Explicit padding order: (Top, Left, Bottom, Right)
    if padding_direction == (1, 1, 0, 0):
        # Pad top left corner
        """
                     | a a b |
        |  a b  | -> | a a b |
        |  c d  |    | c c d |
        """
        addresses = [addresses[0]] * 4
        h0, h1, w0 = 1, 1, 1

    elif padding_direction == (1, 0, 0, 1):
        # Pad top right corner
        """
                     | a b b |
        |  a b  | -> | a b b |
        |  c d  |    | c d d |
        """
        addresses = [addresses[0], addresses[0] + tr_offset, addresses[0], addresses[0] + tr_offset]
        h0, h1, w0 = 1, 1, w0

    elif padding_direction == (0, 1, 1, 0):
        # Pad bottom left corner
        """
        |  a b  |    | a a b |
        |  c d  | -> | c c d |
                     | c c d |
        """
        addresses = [addresses[0], addresses[0], addresses[0] + bl_offset, addresses[0] + bl_offset]
        h0, h1, w0 = h0, h1, 1

    elif padding_direction == (0, 0, 1, 1):
        # Pad bottom right corner
        """
        |  a b  |    | a b b |
        |  c d  | -> | c d d |
                     | c d d |
        """
        addresses = [
            addresses[0],
            addresses[0] + tr_offset,
            addresses[0] + bl_offset,
            addresses[0] + br_offset,
        ]
        # h0, h1, w0 = h0, h1, w0
    else:
        assert 0, "Invalid padding direction for tile padding"

    return NpuTileBox(height_0=h0, height_1=h1, width_0=w0, addresses=[int(addr) for addr in addresses])


def get_region(mem_type: MemType, arch: ArchitectureFeatures) -> int:
    base_ptr_idx_map = {
        MemType.Permanent_NPU: BasePointerIndex.WeightTensor,
        MemType.Permanent_CPU: BasePointerIndex.WeightTensor,
        MemType.Scratch: BasePointerIndex.ScratchTensor,
    }

    if arch.is_spilling_enabled():
        base_ptr_idx_map[MemType.Scratch_fast] = BasePointerIndex.ScratchFastTensor
    else:
        base_ptr_idx_map[MemType.Scratch_fast] = BasePointerIndex.ScratchTensor

    return base_ptr_idx_map[mem_type].value


def get_mem_limits_for_regions(arch: ArchitectureFeatures) -> Dict[int, int]:
    """Returns map region -> max size of the region in bytes"""
    mem_limits = dict()
    for mem_type in MemType.all():
        mem_limits[get_region(mem_type, arch)] = arch.mem_type_size(mem_type)
    mem_limits[BASE_PTR_INDEX_MEM2MEM] = arch.shram_size_bytes
    return mem_limits


def get_ifm_depth(npu_block_type: NpuBlockType, ifm_box: Box, ofm_box: Box) -> int:
    if npu_block_type in (NpuBlockType.ConvolutionMxN, NpuBlockType.VectorProduct, NpuBlockType.ReduceSum):
        block = ifm_box.get_block()
    else:
        block = ofm_box.get_block()
    return block.depth


def use_zero_point_0(ps, tens: Tensor, is_ifm_tensor: bool) -> bool:
    """Checks if quantization should use 0 as zero point"""
    if tens.dtype == DataType.int32 and is_ifm_tensor:
        return True
    if ps.primary_op.rounding_mode == RoundingMode.AwayZero:
        if (
            ps.primary_op.original_type == Op.AvgPool
            and ps.primary_op.type == Op.Conv2DBias
            and ps.primary_op.attrs.get("padding", None) in (Padding.EXPLICIT, Padding.VALID)
        ):
            # Force zero point to 0 for AveragePool operators converted to a Conv2DBias with rounding away from
            # zero.
            return True
        if ps.primary_op.original_type == Op.ResizeBilinear and ps.primary_op.type == Op.DepthwiseConv2DBias:
            # Force zero point to 0 for ResizeBilinear operators converted to a DepthwiseConv with rounding away from
            # zero. This is because the reference kernel ignores the zero points.
            return True
        if (
            not is_ifm_tensor
            and ps.primary_op.original_type == Op.AvgPool
            and ps.primary_op.attrs.get("padding", None) == Padding.EXPLICIT
            and ps.primary_op.type == Op.DepthwiseConv2DBias
        ):
            # Force zero point to 0 for the OFM of AvgPool operators that have been combined with a previous PAD
            # operator and converted to a DepthwiseConv with rounding away from zero. This is because the zero point
            # will already have been applied in the Bias.
            return True
    if ps.primary_op.type not in (Op.AvgPool, Op.CLZ, Op.SHL) and not ps.primary_op.type.is_resize_op():
        return False
    if ps.primary_op.type == Op.AvgPool and ps.primary_op.explicit_scaling:
        return False
    fused_quantize = any(op.type == Op.Quantize or op.original_type == Op.Quantize for op in ps.ops)
    forced_ofm_quantization = ps.primary_op.forced_output_quantization
    use_0 = (
        (
            ps.primary_op.activation is None
            or forced_ofm_quantization is not None
            or (ps.primary_op.type.is_avgpool_op() and ps.primary_op.activation.op_type.is_relu_op())
        )
        and (ps.primary_op.memory_function != Op.ConcatSliceWrite)
        and not fused_quantize
    )
    return use_0


def get_ifm_or_ifm2_quantization(ps, tens: Tensor) -> Optional[NpuQuantization]:
    """Gets quantization for IFM/IFM2"""
    op = ps.primary_op
    ifm_quant = op.forced_input_quantization if op.forced_input_quantization is not None else tens.quantization
    if ifm_quant is None:
        return None
    if use_zero_point_0(ps, tens, True):
        zero_point = 0
    else:
        zero_point = int(ifm_quant.zero_point)
    return NpuQuantization(scale_f32=ifm_quant.scale_f32, zero_point=zero_point)


def get_ofm_quantization(ps, tens: Tensor) -> Optional[NpuQuantization]:
    """Gets quantization for OFM"""
    op = ps.primary_op
    # Check if operation's output quantization is should be used instead of the output tensor's quantization
    # (used in LUTs)
    ofm_quant = op.forced_output_quantization if op.forced_output_quantization is not None else tens.quantization
    if ofm_quant is None:
        return None
    if use_zero_point_0(ps, tens, False):
        zero_point = 0
    else:
        zero_point = int(ofm_quant.zero_point)
    return NpuQuantization(scale_f32=ofm_quant.scale_f32, zero_point=zero_point)


def create_feature_map(
    tens: Tensor,
    box: Box,
    arch: ArchitectureFeatures,
    op_shape4D: Shape4D,
    tile_base_offsets: List[int],
    stride_multiplier: Optional[List[int]] = None,
    is_ofm: bool = False,
) -> NpuFeatureMap:
    """Creates feature map with common fields populated"""
    fm = NpuFeatureMap()
    fm.region = get_region(tens.mem_type, arch)
    fm.data_type = dtype_map[tens.dtype]
    if tens.format == TensorFormat.NHWC:
        fm.layout = NpuLayout.NHWC
    elif tens.format == TensorFormat.NHCWB16:
        fm.layout = NpuLayout.NHCWB16
    else:
        assert 0, "Incorrect tensor format"

    if is_ofm and tens.ops[0] is not None and tens.ops[0].original_type == Op.Transpose:
        # op_shape4D has ifm shape, see fixup_transpose. Stride calculations needs to be
        # based on the correct ofm shape.
        op_shape4D_ofm_shape = Shape4D([op_shape4D.batch, op_shape4D.width, op_shape4D.height, op_shape4D.depth])
        strides = tens.get_strides(op_shape4D_ofm_shape)
        # Swap h and w strides which will cause the transpose to happen
        strides[-3], strides[-2] = strides[-2], strides[-3]
    else:
        strides = tens.get_strides(op_shape4D)

    assert strides is not None

    if stride_multiplier and stride_multiplier != [1, 1, 1]:
        assert (
            tens.format == TensorFormat.NHWC
        ), "Only default stride multiplier ([1, 1, 1]) supported for NHCWB16 format"
        # Multiply strides for C/H/W (in that order) with corresponding stride factor
        for i, stride_factor in enumerate(stride_multiplier, start=1):
            strides[i] *= stride_factor

    height_0, height_1, width_0, addresses = tens.addresses_for_rolling_buffer(
        box.start_coord, box.end_coord, strides, op_shape4D
    )

    for idx, offset in enumerate(tile_base_offsets):
        addresses[idx] += offset
    fm.tiles = NpuTileBox(
        height_0=height_0, height_1=height_1, width_0=width_0, addresses=[int(addr) for addr in addresses]
    )
    fm.strides = NpuShape3D(height=int(strides[2]), width=int(strides[3]), depth=int(strides[1]))
    fm.name = tens.name
    return fm


def create_weights(
    weight_tensor: NpuWeightTensor, weight_box: Box, scale_tensor: NpuWeightTensor, arch: ArchitectureFeatures
) -> Tuple[List[NpuAddressRange], List[NpuAddressRange]]:
    """Returns address ranges for weights and scales"""
    weights = []
    biases = []
    shared_region = get_region(weight_tensor.mem_type, arch)
    scale_region = get_region(scale_tensor.mem_type, arch) if scale_tensor else 0

    w_tensor_src = weight_tensor
    if weight_tensor.src_tensor:
        w_tensor_src = cast(NpuWeightTensor, weight_tensor.src_tensor)

    core_offset = 0
    for core in range(0, arch.ncores):
        # Get weight range per core
        key = WeightKey(core, weight_box.start_coord[-1])
        if key in w_tensor_src.encoded_ranges:
            weight_range = w_tensor_src.encoded_ranges[key]
            if weight_tensor == w_tensor_src:
                # Straight from source tensor
                address = weight_tensor.address + weight_range.offset
            else:
                # Weight buffered tensor
                address = weight_tensor.address + core_offset
                core_offset += round_up(weight_range.total_bytes, 16)

            # Location of weights in tensor
            addr_range = NpuAddressRange(
                shared_region, int(address + weight_range.weight_offset), round_up(int(weight_range.weight_bytes), 16)
            )
            weights.append(addr_range)

            # Location of standalone scales or combined weights tensor scales
            if scale_tensor:
                assert scale_tensor.src_tensor is None  # Must be standalone
                scale_range = scale_tensor.encoded_ranges[key]
                address = scale_tensor.address + scale_range.offset
                addr_range = NpuAddressRange(scale_region, int(address), round_up(int(scale_range.scale_bytes), 16))
            else:
                addr_range = NpuAddressRange(shared_region, int(address), round_up(int(weight_range.scale_bytes), 16))

            biases.append(addr_range)

    return weights, biases


def create_npu_activation(op: Operation) -> NpuActivation:
    """Creates fused activation function"""
    if op.activation is None:
        return NpuActivation(NpuActivationOp.NONE_OR_RELU)
    faf = op.activation.op_type
    act_op = NpuActivationOp.NONE_OR_RELU
    if faf == Op.Tanh:
        act_op = NpuActivationOp.TANH
    elif faf == Op.Sigmoid:
        act_op = NpuActivationOp.SIGMOID
    elif faf == Op.LUT:
        act_op = NpuActivationOp.TABLE_LOOKUP
    elif not faf.is_relu_op():
        raise UnsupportedFeatureError(f"Unsupported fused_activation_function: {faf.name}")

    act = NpuActivation(act_op)
    act.min = op.activation.min
    act.max = op.activation.max
    if act_op is NpuActivationOp.NONE_OR_RELU and op.type.is_avgpool_op() and not op.explicit_scaling:
        quant = op.ofm.quantization
        if quant and quant.zero_point:  # Zero point is not 0
            scale_f32 = 1 if quant.scale_f32 is None else quant.scale_f32
            zero_point = quant.zero_point
            if act.min is not None:
                act.min = scale_f32 * quantise_float32(act.min, scale_f32, zero_point)
            if act.max is not None:
                act.max = scale_f32 * quantise_float32(act.max, scale_f32, zero_point)
    act.lookup_table_index = op.activation.lut_index
    return act


def set_common_op_fields(npu_op: NpuBlockOperation, cmd: NpuStripe, arch: ArchitectureFeatures):
    """Sets common fields of the given operation"""
    ps = cmd.ps
    op = ps.primary_op

    ifm_height = cmd.ifm_box.get_block().height
    ifm_width = cmd.ifm_box.get_block().width
    ifm_depth = get_ifm_depth(op.type.npu_block_type, cmd.ifm_box, cmd.ofm_box)

    npu_op.ifm = create_feature_map(cmd.ifm_tensor, cmd.ifm_box, arch, ps.ifm_shapes[0], op.tile_base_offsets_ifm[0])
    npu_op.ifm.shape = NpuShape3D(height=ifm_height, width=ifm_width, depth=ifm_depth)
    npu_op.ifm.quantization = get_ifm_or_ifm2_quantization(ps, cmd.ifm_tensor)

    out_block = cmd.ofm_box.get_block()
    npu_op.ofm = create_feature_map(
        cmd.ofm_tensor,
        cmd.ofm_box,
        arch,
        ps.ofm_shapes[0],
        op.tile_base_offsets_ofm,
        op.ofm_stride_multiplier,
        is_ofm=True,
    )
    npu_op.ofm.shape = NpuShape3D(height=out_block.height, width=out_block.width, depth=out_block.depth)
    npu_op.ofm.quantization = get_ofm_quantization(ps, cmd.ofm_tensor)

    if cmd.weight_tensor is not None:
        npu_op.weights, npu_op.biases = create_weights(cmd.weight_tensor, cmd.weight_box, cmd.scale_tensor, arch)
    npu_op.activation = create_npu_activation(op)
    npu_op.fused_quantize = any(op.type == Op.Quantize or op.original_type == Op.Quantize for op in ps.ops)
    npu_op.rounding_mode = get_rounding_mode(op, npu_op.fused_quantize)
    npu_op.block_config = NpuShape3D(height=ps.block_config[0], width=ps.block_config[1], depth=ps.block_config[3])

    if not op.type.is_elementwise_op():
        npu_op.padding = create_padding(cmd, op, npu_op)
        npu_op.kernel = to_npu_kernel(op.kernel)
    npu_op.ifm_upscale = resampling_mode_inv_map[op.ifm_resampling_mode]
    return npu_op


def create_npu_conv2d_op(cmd: NpuStripe, arch: ArchitectureFeatures) -> NpuConv2DOperation:
    """Converts the command to NpuConv2DOperation"""
    npu_op = NpuConv2DOperation()
    set_common_op_fields(npu_op, cmd, arch)
    if cmd.ps.primary_op.type.npu_block_type == NpuBlockType.VectorProduct:
        npu_op.block_traversal = NpuBlockTraversal.DEPTH_FIRST
    else:
        if cmd.weight_tensor.src_tensor:
            npu_op.block_traversal = cmd.weight_tensor.src_tensor.hw_traversal
        else:
            npu_op.block_traversal = cmd.weight_tensor.hw_traversal
    return npu_op


def create_npu_conv_depthwise_op(cmd: NpuStripe, arch: ArchitectureFeatures) -> NpuConvDepthWiseOperation:
    """Converts the command to NpuConvDepthWiseOperation"""
    npu_op = NpuConvDepthWiseOperation()
    set_common_op_fields(npu_op, cmd, arch)
    return npu_op


def create_npu_pool_op(cmd: NpuStripe, arch: ArchitectureFeatures) -> NpuPoolingOperation:
    """Converts the command to NpuPoolingOperation"""
    ps = cmd.ps
    op = ps.primary_op
    if op.type.is_maxpool_op():
        pool_op = NpuPoolingOp.MAX
    elif op.type.is_avgpool_op() or op.type.is_resize_op():
        pool_op = NpuPoolingOp.AVERAGE
    elif op.type == Op.ReduceSum:
        pool_op = NpuPoolingOp.REDUCE_SUM
    else:
        assert 0, f"Unknown pool type {op.type}"
    npu_op = NpuPoolingOperation(pool_op)
    set_common_op_fields(npu_op, cmd, arch)
    # Pooling specific info
    if op.explicit_scaling:
        # Note: reuse of rescale for explicit scaling to not expose this in the external API
        npu_op.rescale = op.explicit_scaling
    return npu_op


def create_npu_elementwise_op(cmd: NpuStripe, arch: ArchitectureFeatures) -> NpuElementWiseOperation:
    """Converts the command to NpuElementWiseOperation"""
    ps = cmd.ps
    op = ps.primary_op
    assert op.type in elementwise_op_map, f"Unknown elementwise type {op.type}"
    elemwise_op = elementwise_op_map[op.type]
    npu_op = NpuElementWiseOperation(elemwise_op)

    if elemwise_op not in UNARY_ELEMWISE_OPS:
        ifm_shape = None if cmd.ifm_tensor.shape == [] else ps.ifm_shapes[0]
        ifm2_shape = None if cmd.ifm2_tensor.shape == [] else ps.ifm_shapes[1]
        if cmd.reversed_operands:
            assert ifm_ifm2_correct_order(ifm_shape, ifm2_shape)
            npu_op.reversed_operands = True
        elif not ifm_ifm2_correct_order(ifm_shape, ifm2_shape):
            # The scalar/broadcasted feature map has to be the ifm2 tensor so switch the ifms
            cmd.ifm_tensor, cmd.ifm2_tensor = cmd.ifm2_tensor, cmd.ifm_tensor
            cmd.ifm_box, cmd.ifm2_box = cmd.ifm2_box, cmd.ifm_box
            ps.ifm_shapes[0], ps.ifm_shapes[1] = ps.ifm_shapes[1], ps.ifm_shapes[0]
            npu_op.reversed_operands = True
        npu_op.ifm2 = create_feature_map(
            cmd.ifm2_tensor,
            cmd.ifm2_box,
            arch,
            ps.ifm_shapes[1],
            op.tile_base_offsets_ifm[1],
        )
        npu_op.ifm2.quantization = get_ifm_or_ifm2_quantization(ps, cmd.ifm2_tensor)
        if cmd.ifm2_tensor.shape == []:
            # scalar
            npu_op.ifm2_scalar = cmd.ifm2_tensor.get_scalar()
            npu_op.ifm2.shape = NpuShape3D(height=0, width=0, depth=0)
        else:
            ifm2_blk = cmd.ifm2_box.get_block()
            npu_op.ifm2.shape = NpuShape3D(height=ifm2_blk.height, width=ifm2_blk.width, depth=ifm2_blk.depth)
    set_common_op_fields(npu_op, cmd, arch)
    # Check if output scale needs to be overridden
    output_scale = None
    if op.explicit_scaling is not None:
        assert not op.explicit_scaling.per_channel
        assert op.type in (Op.Add, Op.Mul, Op.Sub)
        npu_op.rescale = (op.explicit_scaling.multiplier[0], op.explicit_scaling.shift[0])
    elif op.type == Op.Add and op.original_type.is_resize_op():
        # Force output scale same as the input scale for
        # resizebilinear/nearestneighbor 1x1 that is converted to add
        output_scale = npu_op.ifm2.quantization.scale_f32
    elif op.type == Op.Abs:
        output_scale = npu_op.ifm.quantization.scale_f32 / npu_op.ofm.quantization.scale_f32
    elif op.type == Op.LeakyRelu:
        output_scale = op.attrs["alpha"]
    elif op.type in (Op.Add, Op.Mul, Op.Sub):
        if op.activation is not None and op.activation.op_type in (Op.Sigmoid, Op.Tanh):
            output_scale = 1 / 0x3000
    if output_scale is not None:
        npu_op.ofm.quantization = NpuQuantization(scale_f32=output_scale, zero_point=npu_op.ofm.quantization.zero_point)
    return npu_op


def create_dma_op(cmd: DMA, arch: ArchitectureFeatures) -> NpuDmaOperation:
    """Converts the command to NpuDmaOperation"""
    src_region = get_region(cmd.in_tensor.mem_type, arch)
    if cmd.out_tensor.purpose == TensorPurpose.LUT:
        dest_region = BASE_PTR_INDEX_MEM2MEM
    else:
        dest_region = get_region(cmd.out_tensor.mem_type, arch)

    if cmd.in_tensor.purpose == TensorPurpose.Weights:
        # Get weight range per core
        sz = 0
        for core in range(0, arch.ncores):
            key = WeightKey(core, cmd.box.start_coord[-1])
            if key in cmd.in_tensor.encoded_ranges:
                weight_range = cmd.in_tensor.encoded_ranges[key]
                sz += round_up(weight_range.total_bytes, 16)

                if core == 0:
                    weight_range = cmd.in_tensor.encoded_ranges[key]
                    src_addr = cmd.in_tensor.address + weight_range.offset
                    dest_addr = cmd.out_tensor.address
    else:
        src_addr = cmd.in_tensor.address_for_coordinate(cmd.box.start_coord)
        dest_addr = cmd.out_tensor.address_for_coordinate(cmd.box.start_coord)
        # DMA must use 16 bytes alignment (tensors are always aligned but the sz calculation uses actual size)
        sz = round_up(cmd.in_tensor.address_for_coordinate(cmd.box.end_coord, is_top_box=True) - src_addr, 16)
    src = NpuAddressRange(src_region, int(src_addr), int(sz))
    dest = NpuAddressRange(dest_region, int(dest_addr), int(sz))
    return NpuDmaOperation(src, dest)


def convert_command_to_npu_op(cmd: Command, arch: ArchitectureFeatures) -> NpuOperation:
    """Converts the high level command to NpuOperation"""
    npu_op: NpuOperation
    if isinstance(cmd, DMA):
        npu_op = create_dma_op(cmd, arch)
        npu_op.name = cmd.out_tensor.name
    elif isinstance(cmd, NpuStripe):
        npu_block_type = cmd.ps.primary_op.type.npu_block_type
        if npu_block_type in (NpuBlockType.ConvolutionMxN, NpuBlockType.VectorProduct):
            npu_op = create_npu_conv2d_op(cmd, arch)
        elif npu_block_type == NpuBlockType.ConvolutionDepthWise:
            npu_op = create_npu_conv_depthwise_op(cmd, arch)
        elif npu_block_type in (NpuBlockType.Pooling, NpuBlockType.ReduceSum):
            npu_op = create_npu_pool_op(cmd, arch)
        elif npu_block_type == NpuBlockType.ElementWise:
            npu_op = create_npu_elementwise_op(cmd, arch)
        else:
            assert 0, f"Unknown command type {npu_block_type}"
        npu_op.name = cmd.ps.primary_op.name
    return npu_op


def generate_register_command_stream_for_sg(nng, sg, arch, verbose=False):
    """Generates command stream for the subgraph, adds it to sg.register_command_stream"""
    # Convert high level command stream to list of NpuOperation
    npu_op_list = []
    npu_op_to_cmd = dict()  # map from npu op to high level command
    for cmd in sg.high_level_command_stream:
        if isinstance(cmd, NpuStripe) and cmd.ps.npu_block_type == NpuBlockType.Default:
            print("Warning: Skipping register command stream generation for", cmd.ps)
        elif isinstance(cmd, NOP):
            # NOP should not generate anything
            continue
        else:
            npu_op = convert_command_to_npu_op(cmd, arch)
            npu_op_list.append(npu_op)
            npu_op_to_cmd[npu_op] = cmd
    mem_limits = get_mem_limits_for_regions(arch)
    # Generate register commands
    if len(sg.high_level_command_stream) > 0:
        stream_id = DebugDatabase.add_stream(sg)
        sg.generated_stream_id = stream_id

        def add_to_debug_db(npu_op: NpuOperation, offset: int):
            """Adds info to the debug database"""
            if not isinstance(npu_op, NpuDmaOperation):
                cmd = npu_op_to_cmd[npu_op]
                DebugDatabase.add_command(stream_id, offset, cmd.ps.primary_op)

        sg.register_command_stream = generate_command_stream(
            npu_op_list, arch, verbose, mem_limits, add_to_debug_db, npu_op_to_cmd
        )
