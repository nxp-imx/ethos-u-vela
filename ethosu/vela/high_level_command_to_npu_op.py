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
#
# Description:
# Conversion from high level command to NpuOperation
from enum import IntEnum
from typing import Dict
from typing import List
from typing import Optional

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
from .high_level_command_stream import Box
from .high_level_command_stream import Command
from .high_level_command_stream import DMA
from .high_level_command_stream import NpuStripe
from .numeric_util import quantise_float32
from .numeric_util import round_up
from .operation import NpuBlockType
from .operation import Op
from .operation import Operation
from .register_command_stream_generator import generate_command_stream
from .register_command_stream_util import BASE_PTR_INDEX_MEM2MEM
from .register_command_stream_util import to_npu_kernel
from .register_command_stream_util import UNARY_ELEMWISE_OPS
from .shape4d import Shape4D
from .tensor import MemType
from .tensor import Tensor
from .tensor import TensorFormat
from .tensor import TensorPurpose
from .tensor import TensorSubPurpose
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
    Op.RescaleMul: NpuElementWiseOp.MUL,
    Op.Add: NpuElementWiseOp.ADD,
    Op.RescaleAdd: NpuElementWiseOp.ADD,
    Op.Sub: NpuElementWiseOp.SUB,
    Op.Minimum: NpuElementWiseOp.MIN,
    Op.Maximum: NpuElementWiseOp.MAX,
    Op.LeakyRelu: NpuElementWiseOp.LRELU,
    Op.Abs: NpuElementWiseOp.ABS,
    Op.CLZ: NpuElementWiseOp.CLZ,
    Op.SHR: NpuElementWiseOp.SHR,
    Op.SHL: NpuElementWiseOp.SHL,
}


def ifm_ifm2_correct_order(ifm_shape: List[int], ifm2_shape: List[int]) -> bool:
    if ifm_shape == []:
        # Scalar needs to be in IFM2
        return False
    if ifm2_shape == []:
        return True

    for ifm, ifm2 in zip(ifm_shape, ifm2_shape):
        if ifm != ifm2 and ifm == 1:
            # Broadcasted FM needs to be in IFM2
            return False
    return True


def get_rounding_mode(op: Operation, fused_quantize: bool) -> NpuRoundingMode:
    """Specifies type of rounding to be used"""
    rounding_mode = NpuRoundingMode.TFL
    if op.type == Op.ResizeBilinear:
        rounding_mode = NpuRoundingMode.NATURAL
    elif (
        op.type.npu_block_type in (NpuBlockType.ConvolutionMxN, NpuBlockType.ConvolutionDepthWise)
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
        rounding_mode = op.rounding_mode
    return rounding_mode


def create_padding(cmd: NpuStripe, primary_op: Operation) -> NpuPadding:
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
    return NpuPadding(top=top, left=left, bottom=bottom, right=right)


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


def get_upscale(op: Operation) -> NpuResamplingMode:
    upscale = NpuResamplingMode.NONE
    if op.type == Op.ResizeBilinear:
        # perform nearest neighbor upscale
        upscale = NpuResamplingMode.NEAREST
    elif op.type == Op.Conv2DBackpropInputSwitchedBias:
        # perform insert zero upscale
        upscale = NpuResamplingMode.TRANSPOSE
    return upscale


def get_double_buffer_offset(arch: ArchitectureFeatures, range_index: int, core: int) -> int:
    """Returns 0 if the first half of a double buffer should be used, 1 if the second half should be used"""
    return ((range_index - core) // arch.ncores) % 2


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
    if ps.primary_op.type not in (Op.AvgPool, Op.ResizeBilinear, Op.CLZ, Op.SHL):
        return False
    if ps.primary_op.type == Op.AvgPool and ps.primary_op.explicit_scaling:
        return False
    fused_quantize = any(op.type == Op.Quantize for op in ps.ops)
    forced_ofm_quantization = ps.primary_op.forced_output_quantization
    use_0 = (
        (
            ps.primary_op.activation is None
            or forced_ofm_quantization is not None
            or (
                ps.primary_op.type.is_avgpool_op()
                and ps.primary_op.activation.op_type.is_relu_op()
                and not ps.primary_op.rescale
            )
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


def create_feature_map(tens: Tensor, box: Box, arch: ArchitectureFeatures, op_shape4D: Shape4D) -> NpuFeatureMap:
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
    height_0, height_1, width_0, addresses = tens.addresses_for_rolling_buffer(
        box.start_coord, box.end_coord, op_shape4D
    )
    for idx, addr in enumerate(addresses):
        if addr is None:
            addresses[idx] = 0
    fm.tiles = NpuTileBox(
        height_0=height_0, height_1=height_1, width_0=width_0, addresses=[int(addr) for addr in addresses]
    )
    strides = tens.get_strides(shape4D=op_shape4D)
    fm.strides = NpuShape3D(height=int(strides[2]), width=int(strides[3]), depth=int(strides[1]))
    return fm


def create_weights(
    weight_tensor: Tensor, weight_box: Box, scale_tensor: Tensor, arch: ArchitectureFeatures
) -> List[NpuAddressRange]:
    """Returns address ranges for weights and scales"""
    weights = []
    biases = []
    shared_region = get_region(weight_tensor.mem_type, arch)
    scale_region = scale_tensor and get_region(scale_tensor.mem_type, arch)

    w_tensor_src = weight_tensor
    if weight_tensor.src_tensor:
        w_tensor_src = weight_tensor.src_tensor

    core_offset = 0
    for core in range(0, arch.ncores):
        # Get weight range per core
        key = WeightKey(core, weight_box.start_coord[-1])
        if key in w_tensor_src.encoded_ranges:
            weight_range = w_tensor_src.encoded_ranges[key]
            if weight_tensor.sub_purpose == TensorSubPurpose.DoubleBuffer:
                assert weight_tensor != w_tensor_src
                # Double buffered inside weight_tensor
                address = weight_tensor.address + core_offset
                address += get_double_buffer_offset(arch, weight_range.index, core) * w_tensor_src.max_range_bytes
                core_offset += round_up(weight_range.total_bytes, 16)
            else:
                if weight_tensor == w_tensor_src:
                    # Straight from source tensor
                    address = weight_tensor.address + weight_range.offset
                else:
                    # Single buffered inside weight tensor
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
    if act_op is NpuActivationOp.NONE_OR_RELU and op.type.is_avgpool_op() and not op.rescale:
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
    ifm_width = cmd.ps.ifm_shapes[0].width
    ifm_depth = get_ifm_depth(op.type.npu_block_type, cmd.ifm_box, cmd.ofm_box)

    npu_op.ifm = create_feature_map(cmd.ifm_tensor, cmd.ifm_box, arch, ps.ifm_shapes[0])
    npu_op.ifm.shape = NpuShape3D(height=ifm_height, width=ifm_width, depth=ifm_depth)
    npu_op.ifm.quantization = get_ifm_or_ifm2_quantization(ps, cmd.ifm_tensor)

    out_block = cmd.ofm_box.get_block()
    npu_op.ofm = create_feature_map(cmd.ofm_tensor, cmd.ofm_box, arch, ps.ofm_shapes[0])
    npu_op.ofm.shape = NpuShape3D(height=out_block.height, width=out_block.width, depth=out_block.depth)
    npu_op.ofm.quantization = get_ofm_quantization(ps, cmd.ofm_tensor)

    if cmd.weight_tensor is not None:
        npu_op.weights, npu_op.biases = create_weights(cmd.weight_tensor, cmd.weight_box, cmd.scale_tensor, arch)
    npu_op.activation = create_npu_activation(op)
    npu_op.fused_quantize = any(op.type == Op.Quantize for op in ps.ops)
    npu_op.rounding_mode = get_rounding_mode(op, npu_op.fused_quantize)
    npu_op.block_config = NpuShape3D(height=ps.block_config[0], width=ps.block_config[1], depth=ps.block_config[3])

    if not op.type.is_elementwise_op():
        npu_op.padding = create_padding(cmd, op)
        npu_op.kernel = to_npu_kernel(op.kernel)
    npu_op.ifm_upscale = get_upscale(op)
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
    pool_op = NpuPoolingOp.AVERAGE
    if op.type.is_maxpool_op():
        pool_op = NpuPoolingOp.MAX
    elif op.type.is_avgpool_op() or op.type == Op.ResizeBilinear:
        pool_op = NpuPoolingOp.AVERAGE
    elif op.type == Op.ReduceSum:
        pool_op = NpuPoolingOp.REDUCE_SUM
    else:
        assert 0, f"Unknown pool type {op.type}"
    npu_op = NpuPoolingOperation(pool_op)
    set_common_op_fields(npu_op, cmd, arch)
    # Pooling specific info
    npu_op.rescale = op.rescale
    if op.explicit_scaling:
        # Note: reuse of rescale for explicit scaling to not expose this in the external API
        assert npu_op.rescale is None
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
        ifm_shape = [] if cmd.ifm_tensor.shape == [] else ps.ifm_shapes[0].as_list()
        ifm2_shape = [] if cmd.ifm2_tensor.shape == [] else ps.ifm_shapes[1].as_list()
        if not ifm_ifm2_correct_order(ifm_shape, ifm2_shape):
            # The scalar/broadcasted feature map has to be the ifm2 tensor so switch the ifms
            cmd.ifm_tensor, cmd.ifm2_tensor = cmd.ifm2_tensor, cmd.ifm_tensor
            cmd.ifm_box, cmd.ifm2_box = cmd.ifm2_box, cmd.ifm_box
            ps.ifm_shapes[0], ps.ifm_shapes[1] = ps.ifm_shapes[1], ps.ifm_shapes[0]
            npu_op.reversed_operands = True
        npu_op.ifm2 = create_feature_map(cmd.ifm2_tensor, cmd.ifm2_box, arch, ps.ifm_shapes[1])
        npu_op.ifm2.quantization = get_ifm_or_ifm2_quantization(ps, cmd.ifm2_tensor)
        if cmd.ifm2_tensor.shape == []:
            # scalar
            npu_op.ifm2_scalar = cmd.ifm2_tensor.get_scalar()
            npu_op.ifm2.shape = NpuShape3D(height=0, width=0, depth=0)
        else:
            ifm2_blk = cmd.ifm2_box.get_block()
            ifm2_width = ps.ifm_shapes[1].width
            npu_op.ifm2.shape = NpuShape3D(height=ifm2_blk.height, width=ifm2_width, depth=ifm2_blk.depth)
    set_common_op_fields(npu_op, cmd, arch)
    # Check if output scale needs to be overridden
    output_scale = None
    if op.type == Op.Add and "resizebilinear" in op.attrs:
        # Force output scale same as the input scale for
        # resizebilinear 1x1 that is converted to add
        output_scale = npu_op.ifm2.quantization.scale_f32
    if op.type == Op.Abs:
        output_scale = npu_op.ifm.quantization.scale_f32 / npu_op.ofm.quantization.scale_f32
    if op.type == Op.LeakyRelu:
        output_scale = op.attrs["alpha"]
    if op.type in (Op.RescaleAdd, Op.RescaleMul):
        assert op.rescale is not None, f"{op.type} must have rescale"
        npu_op.rescale = op.rescale
    if op.type in (Op.Add, Op.Mul, Op.Sub):
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

                    if cmd.out_tensor.sub_purpose == TensorSubPurpose.DoubleBuffer:
                        dest_addr = cmd.out_tensor.address + cmd.in_tensor.max_range_bytes * (
                            get_double_buffer_offset(arch, weight_range.index, core)
                        )
                    else:
                        dest_addr = cmd.out_tensor.address
    else:
        start_coord = cmd.box.start_coord
        src_addr = cmd.in_tensor.address_for_coordinate(start_coord)
        dest_addr = cmd.out_tensor.address_for_coordinate(start_coord)
        sz = cmd.in_tensor.address_for_coordinate(cmd.box.end_coord, is_top_box=True) - src_addr
    src = NpuAddressRange(src_region, int(src_addr), int(sz))
    dest = NpuAddressRange(dest_region, int(dest_addr), int(sz))
    return NpuDmaOperation(src, dest)


def convert_command_to_npu_op(cmd: Command, arch: ArchitectureFeatures) -> NpuOperation:
    """Converts the high level command to NpuOperation"""
    npu_op: NpuOperation
    if isinstance(cmd, DMA):
        npu_op = create_dma_op(cmd, arch)
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
    return npu_op


def generate_register_command_stream_for_sg(nng, sg, arch, verbose=False):
    """Generates command stream for the subgraph, adds it to sg.register_command_stream"""
    # Convert high level command stream to list of NpuOperation
    npu_op_list = []
    npu_op_to_cmd = dict()  # map from npu op to high level command
    for cmd in sg.high_level_command_stream:
        if isinstance(cmd, NpuStripe) and cmd.ps.npu_block_type == NpuBlockType.Default:
            print("Warning: Skipping register command stream generation for", cmd.ps)
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
