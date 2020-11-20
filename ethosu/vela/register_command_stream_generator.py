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
# Register level (low-level) command stream generation for Ethos-U. Takes a list of NPU operations and generates
# all the register settings. Calculates dependencies between commands and inserts wait operations. And generates a bit
# stream suitable for interpretation by the Ethos-U processor.
from collections import defaultdict
from collections import namedtuple
from enum import Enum
from enum import IntEnum
from typing import List
from typing import Optional
from typing import Tuple

import numpy as np

from . import numeric_util
from . import scaling
from .api import NpuAccelerator
from .api import NpuActivation
from .api import NpuActivationOp
from .api import NpuAddressRange
from .api import NpuBlockOperation
from .api import NpuBlockTraversal
from .api import NpuConv2DOperation
from .api import NpuDataType
from .api import NpuDmaOperation
from .api import NpuElementWiseOp
from .api import NpuElementWiseOperation
from .api import NpuFeatureMap
from .api import NpuKernel
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
from .architecture_features import Accelerator
from .architecture_features import ArchitectureFeatures
from .architecture_features import Block
from .architecture_features import create_default_arch
from .architecture_features import Rect
from .architecture_features import SharedBufferArea
from .architecture_features import SHRAMElements
from .debug_database import DebugDatabase
from .ethos_u55_regs.ethos_u55_regs import acc_format
from .ethos_u55_regs.ethos_u55_regs import activation
from .ethos_u55_regs.ethos_u55_regs import cmd0
from .ethos_u55_regs.ethos_u55_regs import cmd1
from .ethos_u55_regs.ethos_u55_regs import elementwise_mode
from .ethos_u55_regs.ethos_u55_regs import pooling_mode
from .ethos_u55_regs.ethos_u55_regs import resampling_mode
from .ethos_u55_regs.ethos_u55_regs import rounding
from .high_level_command_stream import CommandType
from .high_level_command_to_npu_op import convert_command_to_npu_op
from .high_level_command_to_npu_op import to_kernel
from .high_level_command_to_npu_op import unary_elementwise_ops
from .numeric_util import quantise_float32
from .numeric_util import round_away_zero
from .numeric_util import round_up_to_int
from .operation import NpuBlockType
from .range_set import AccessDirection
from .range_set import MemoryAccessSet
from .range_set import MemoryRangeSet
from .shared_buffer_allocation import find_suitable_block_configs
from .shared_buffer_allocation import shared_buffer_allocation_for_npu_op
from .shared_buffer_allocation import SharedBufferAllocation


class RegisterMachine:
    def __init__(self):
        self.n_banks = 1
        self.registers = [defaultdict(lambda: None) for _ in range(self.n_banks)]
        self.bank_idx = 0

    def set_register(self, reg, value):
        is_changed = self.registers[self.bank_idx][reg] != value
        self.registers[self.bank_idx][reg] = value
        # is_changed = True # force command
        return is_changed

    def switch_bank(self):
        self.bank_idx = (self.bank_idx + 1) % self.n_banks


class CmdMode(IntEnum):
    NoPayload = 0x0000
    Payload32 = 0x4000
    Mask = 0xC000
    CmdOpMask = 0x03FF


class CommandStreamEmitter:
    WORD_SIZE = 4

    def __init__(self):
        self.cmd_stream = []
        self.reg_machine = [RegisterMachine(), RegisterMachine()]
        self.last_absolute_wait = defaultdict(int)
        self.offset = 0

    def get_reg_machine(self, cmd):
        if "DMA" in cmd.name:
            return self.reg_machine[1]
        else:
            return self.reg_machine[0]

    def size_in_bytes(self):
        sz = 0
        for cmd in self.cmd_stream:
            sz += len(cmd) * CommandStreamEmitter.WORD_SIZE
        return sz

    def to_list(self) -> List[int]:
        return [elem for cmd in self.cmd_stream for elem in cmd]

    def print_cmds(self):
        print("Code:    Command:                       Param: Payload:")
        for words_for_one_command in self.cmd_stream:
            code = words_for_one_command[0] & 0x0000FFFF  # lower 16 bits
            param = words_for_one_command[0] >> 16  # higher 16 bits

            payload_mode = CmdMode(code & CmdMode.Mask)

            # code and command
            s = "  0x%04x " % code
            if payload_mode == CmdMode.NoPayload:
                s += str(cmd0(code & CmdMode.CmdOpMask))
            else:
                s += str(cmd1(code & CmdMode.CmdOpMask))

            s = s.ljust(40)
            s += "%5d" % param

            # payload
            if payload_mode == CmdMode.Payload32:
                s += "   0x%08x (%d)" % (words_for_one_command[1], words_for_one_command[1])
            else:
                s += "   -"

            print(s)

    def cmd0_with_param(self, cmd: cmd0, param):
        if isinstance(param, Enum):
            param = int(param.value)
        else:
            param = int(param)
        param = param & 0xFFFF
        command = cmd.value | (param << 16)
        if not self.get_reg_machine(cmd).set_register(cmd, (command, param)):
            return

        # This is not a redundant command, actually write it
        self.cmd_stream.append((command,))
        self.offset += CommandStreamEmitter.WORD_SIZE

    def cmd1_with_offset(self, cmd: cmd1, offset, param=0x0):
        offset = int(offset) & 0xFFFFFFFFF
        command = cmd.value | CmdMode.Payload32.value | (param << 16)

        if not self.get_reg_machine(cmd).set_register(cmd, (command, offset)):
            return

        # This is not a redundant command, actually write it
        self.cmd_stream.append((command, offset))
        self.offset += CommandStreamEmitter.WORD_SIZE * 2

    def cmd_wait(self, cmd: cmd0, channel: int, outstanding_count: int):
        param = (16 * channel) + outstanding_count
        command = ((param & 0xFFFF) << 16) | cmd.value
        self.cmd_stream.append((command,))
        self.offset += CommandStreamEmitter.WORD_SIZE

    def cmd_do_operation(self, cmd: cmd0, param=0):
        param = int(param)
        command = ((param & 0xFFFF) << 16) | cmd.value

        self.cmd_stream.append((command,))
        self.offset += CommandStreamEmitter.WORD_SIZE
        self.get_reg_machine(cmd).switch_bank()


# -------------------------------------------------------------------
# REGISTER GENERATION
# -------------------------------------------------------------------


class BasePointerIndex(IntEnum):
    WeightTensor = 0  # base address index for the Weight tensor
    ScratchTensor = 1  # base address index for the Scratch_tensor in the TensorArena
    ScratchFastTensor = 2  # base address for the Scratch_fast_tensor
    Mem2Mem = (1 << 8) | (3 << 0)  # base address slot for memory 2 memory transfer


# TODO: Replace with definitions from ethos_u55_regs
class IFM2Broadcast(IntEnum):
    BroadcastHdim = 1 << 0
    BroadcastWdim = 1 << 1
    BroadcastCdim = 1 << 2
    ReverseOperandOrder = 1 << 6
    UseIFM2Scalar = 1 << 7


pooling_op_map = {
    NpuPoolingOp.MAX: pooling_mode.MAX.value,
    NpuPoolingOp.AVERAGE: pooling_mode.AVERAGE.value,
    NpuPoolingOp.REDUCE_SUM: pooling_mode.REDUCE_SUM.value,
}

elementwise_op_map = {
    NpuElementWiseOp.MUL: elementwise_mode.MUL.value,
    NpuElementWiseOp.ADD: elementwise_mode.ADD.value,
    NpuElementWiseOp.SUB: elementwise_mode.SUB.value,
    NpuElementWiseOp.MIN: elementwise_mode.MIN.value,
    NpuElementWiseOp.MAX: elementwise_mode.MAX.value,
    NpuElementWiseOp.LRELU: elementwise_mode.LRELU.value,
    NpuElementWiseOp.ABS: elementwise_mode.ABS.value,
    NpuElementWiseOp.CLZ: elementwise_mode.CLZ.value,
    NpuElementWiseOp.SHR: elementwise_mode.SHR.value,
    NpuElementWiseOp.SHL: elementwise_mode.SHL.value,
}

activation_op_map = {
    NpuActivationOp.NONE_OR_RELU: activation.NONE,
    NpuActivationOp.TANH: activation.TANH,
    NpuActivationOp.SIGMOID: activation.SIGMOID,
}

# Maps an AccumulatorType enum to the corresponding acc_format value
acc_format_map = {
    SHRAMElements.Acc16: acc_format.FP_S5_10.value,
    SHRAMElements.Acc32: acc_format.INT_32BIT.value,
    SHRAMElements.Acc40: acc_format.INT_40BIT.value,
}

resampling_mode_map = {
    NpuResamplingMode.NONE: resampling_mode.NONE,
    NpuResamplingMode.NEAREST: resampling_mode.NEAREST,
    NpuResamplingMode.TRANSPOSE: resampling_mode.TRANSPOSE,
}

# Maps data type size in bits to activation precision
precision_map = {8: 0, 16: 1, 32: 2}

# Maps rounding mode to the corresponding value
rounding_mode_map = {
    NpuRoundingMode.TFL: rounding.TFL.value,
    NpuRoundingMode.TRUNCATE: rounding.TRUNCATE.value,
    NpuRoundingMode.NATURAL: rounding.NATURAL.value,
}


def quantise(value: float, quant: Optional[NpuQuantization]) -> int:
    """Quantizes the given value"""
    scale = 1 if quant is None or quant.scale_f32 is None else quant.scale_f32
    zp = 0 if quant is None else quant.zero_point
    return quantise_float32(value, scale, zp)


def has_ifm2(npu_op: NpuBlockOperation) -> bool:
    """Checks if op has non-scalar IFM2"""
    return npu_op.ifm2 is not None and npu_op.ifm2_scalar is None


def is_dma_op(npu_op: NpuOperation) -> bool:
    """Checks if op is a DMA operation"""
    return npu_op.op_type == NpuOperationType.Dma


def generate_padding(emit: CommandStreamEmitter, padding: NpuPadding):
    """Generates IFM_PAD registers"""
    emit.cmd0_with_param(cmd0.NPU_SET_IFM_PAD_TOP, padding.top)
    emit.cmd0_with_param(cmd0.NPU_SET_IFM_PAD_LEFT, padding.left)
    emit.cmd0_with_param(cmd0.NPU_SET_IFM_PAD_BOTTOM, padding.bottom)
    emit.cmd0_with_param(cmd0.NPU_SET_IFM_PAD_RIGHT, padding.right)


def generate_activation(emit: CommandStreamEmitter, activation: Optional[NpuActivation], ofm: NpuFeatureMap):
    """Generates ACTIVATION registers"""
    act = activation if activation is not None else NpuActivation(NpuActivationOp.NONE_OR_RELU)

    if act.min is None:
        quantized_min = ofm.data_type.min_value()
    else:
        quantized_min = quantise(act.min, ofm.quantization)
    if act.max is None:
        quantized_max = ofm.data_type.max_value()
    else:
        quantized_max = quantise(act.max, ofm.quantization)
    quantized_min = max(quantized_min, np.iinfo(np.int16).min, ofm.data_type.min_value())
    quantized_max = min(quantized_max, np.iinfo(np.int16).max, ofm.data_type.max_value())
    if act.op_type == NpuActivationOp.TABLE_LOOKUP:
        assert 0 <= act.lookup_table_index < 8
        activation_value = 16 + act.lookup_table_index
        if ofm.data_type == NpuDataType.INT32:
            activation_value |= 3 << 12  # Force I8 range
            quantized_min = max(-128, quantized_min)
            quantized_max = min(127, quantized_max)
    else:
        activation_value = activation_op_map[act.op_type]
    emit.cmd0_with_param(cmd0.NPU_SET_ACTIVATION, activation_value)
    emit.cmd0_with_param(cmd0.NPU_SET_ACTIVATION_MIN, quantized_min)
    emit.cmd0_with_param(cmd0.NPU_SET_ACTIVATION_MAX, quantized_max)


def generate_addresses(emit: CommandStreamEmitter, ptr_cmds: List[cmd1], addresses: List[int], layout: NpuLayout):
    """Generates xFM_BASE registers"""
    if layout == NpuLayout.NHCWB16:
        # Check that all BasePointer addresses are aligned to 16 bytes
        assert all((int(addr) % 16) == 0 for addr in addresses)
    emit.cmd1_with_offset(ptr_cmds[0], addresses[0])
    emit.cmd1_with_offset(ptr_cmds[1], addresses[1])
    emit.cmd1_with_offset(ptr_cmds[2], addresses[2])
    emit.cmd1_with_offset(ptr_cmds[3], addresses[3])


def generate_tiles(emit: CommandStreamEmitter, tile_cmds: List[cmd0], tiles: NpuTileBox):
    """Generates xFM_HEIGHT0/HEIGHT1/WIDTH0 registers"""
    emit.cmd0_with_param(tile_cmds[0], tiles.height_0 - 1)
    emit.cmd0_with_param(tile_cmds[1], tiles.height_1 - 1)
    emit.cmd0_with_param(tile_cmds[2], tiles.width_0 - 1)


def generate_strides(
    emit: CommandStreamEmitter, fm: NpuFeatureMap, stride_c_cmd: cmd1, stride_y_cmd: cmd1, stride_x_cmd: cmd1
):
    """Generates STRIDE_C/Y/X registers"""
    strides = get_strides(fm)
    emit.cmd1_with_offset(stride_c_cmd, strides.depth)  # stride between 16-byte channel blocks (C)
    emit.cmd1_with_offset(stride_y_cmd, strides.height)  # stride between vertical values (H)
    emit.cmd1_with_offset(stride_x_cmd, strides.width)  # stride between horisontal values (W)


def generate_ifm_precision(emit: CommandStreamEmitter, fm: NpuFeatureMap, op_to_scale: int, precision_cmd: cmd0):
    """Generates IFM/IFM2_PRECISION register"""
    dtype = fm.data_type
    prec = 1 if dtype.is_signed() else 0
    activation_precision = precision_map[dtype.size_in_bits()]
    prec += activation_precision << 2

    if fm.layout == NpuLayout.NHCWB16:
        prec |= 1 << 6

    prec |= op_to_scale << 8
    emit.cmd0_with_param(precision_cmd, prec)


def generate_ofm_precision(emit: CommandStreamEmitter, npu_op: NpuBlockOperation, use_global_scale: bool):
    """Generates OFM_PRECISION register"""
    dtype = npu_op.ofm.data_type
    prec = 1 if dtype.is_signed() else 0
    activation_precision = precision_map[dtype.size_in_bits()]
    prec += activation_precision << 1

    if use_global_scale:
        # Set global scale bit, as opposed to using per channel scale
        prec |= 1 << 8
    if npu_op.ofm.layout == NpuLayout.NHCWB16:
        prec |= 1 << 6
    prec |= rounding_mode_map[npu_op.rounding_mode] << 14
    emit.cmd0_with_param(cmd0.NPU_SET_OFM_PRECISION, prec)


def generate_ifm2_broadcast(emit: CommandStreamEmitter, npu_op: NpuElementWiseOperation):
    """Generates IFM2_BROADCAST register for binary elementwise operations"""
    ifm2_broadcast = 0
    ifm = npu_op.ifm
    ifm2 = npu_op.ifm2
    if npu_op.reversed_operands:
        ifm2_broadcast |= IFM2Broadcast.ReverseOperandOrder
    if npu_op.ifm2_scalar is not None:
        # IFM2 is a constant, set UseIFM2Scalar bit to IFM2_BROADCAST
        ifm2_broadcast |= IFM2Broadcast.UseIFM2Scalar
    else:
        if ifm.shape.height != ifm2.shape.height:
            # Broadcast in 'H' dimension
            assert ifm2.shape.height == 1
            ifm2_broadcast |= IFM2Broadcast.BroadcastHdim

        if ifm.shape.width != ifm2.shape.width:
            # Broadcast in 'W' dimension
            assert ifm2.shape.width == 1
            ifm2_broadcast |= IFM2Broadcast.BroadcastWdim

        if ifm.shape.depth != ifm2.shape.depth:
            # Broadcast in 'C' dimension
            assert ifm2.shape.depth == 1
            ifm2_broadcast |= IFM2Broadcast.BroadcastCdim

    emit.cmd0_with_param(cmd0.NPU_SET_IFM2_BROADCAST, ifm2_broadcast)


def generate_ifm(emit: CommandStreamEmitter, ifm: NpuFeatureMap):
    """Generates general IFM registers"""
    emit.cmd0_with_param(cmd0.NPU_SET_IFM_REGION, ifm.region)
    generate_addresses(
        emit,
        [cmd1.NPU_SET_IFM_BASE0, cmd1.NPU_SET_IFM_BASE1, cmd1.NPU_SET_IFM_BASE2, cmd1.NPU_SET_IFM_BASE3],
        ifm.tiles.addresses,
        ifm.layout,
    )
    generate_tiles(
        emit, [cmd0.NPU_SET_IFM_HEIGHT0_M1, cmd0.NPU_SET_IFM_HEIGHT1_M1, cmd0.NPU_SET_IFM_WIDTH0_M1], ifm.tiles
    )
    emit.cmd0_with_param(cmd0.NPU_SET_IFM_DEPTH_M1, ifm.shape.depth - 1)
    generate_strides(emit, ifm, cmd1.NPU_SET_IFM_STRIDE_C, cmd1.NPU_SET_IFM_STRIDE_Y, cmd1.NPU_SET_IFM_STRIDE_X)
    emit.cmd0_with_param(cmd0.NPU_SET_IFM_ZERO_POINT, int(ifm.quantization.zero_point))


def generate_ifm2(emit: CommandStreamEmitter, ifm2: NpuFeatureMap, has_scalar: bool):
    """Generates general IFM2 registers"""
    if not has_scalar:
        emit.cmd0_with_param(cmd0.NPU_SET_IFM2_REGION, ifm2.region)
        generate_addresses(
            emit,
            [cmd1.NPU_SET_IFM2_BASE0, cmd1.NPU_SET_IFM2_BASE1, cmd1.NPU_SET_IFM2_BASE2, cmd1.NPU_SET_IFM2_BASE3],
            ifm2.tiles.addresses,
            ifm2.layout,
        )
        generate_tiles(
            emit, [cmd0.NPU_SET_IFM2_HEIGHT0_M1, cmd0.NPU_SET_IFM2_HEIGHT1_M1, cmd0.NPU_SET_IFM2_WIDTH0_M1], ifm2.tiles
        )
        generate_strides(emit, ifm2, cmd1.NPU_SET_IFM2_STRIDE_C, cmd1.NPU_SET_IFM2_STRIDE_Y, cmd1.NPU_SET_IFM2_STRIDE_X)
    emit.cmd0_with_param(cmd0.NPU_SET_IFM2_ZERO_POINT, int(ifm2.quantization.zero_point))


def generate_ofm(emit: CommandStreamEmitter, ofm: NpuFeatureMap):
    """Generates general OFM registers"""
    emit.cmd0_with_param(cmd0.NPU_SET_OFM_REGION, ofm.region)
    generate_addresses(
        emit,
        [cmd1.NPU_SET_OFM_BASE0, cmd1.NPU_SET_OFM_BASE1, cmd1.NPU_SET_OFM_BASE2, cmd1.NPU_SET_OFM_BASE3],
        ofm.tiles.addresses,
        ofm.layout,
    )
    generate_tiles(
        emit, [cmd0.NPU_SET_OFM_HEIGHT0_M1, cmd0.NPU_SET_OFM_HEIGHT1_M1, cmd0.NPU_SET_OFM_WIDTH0_M1], ofm.tiles
    )
    emit.cmd0_with_param(cmd0.NPU_SET_OFM_HEIGHT_M1, ofm.shape.height - 1)
    emit.cmd0_with_param(cmd0.NPU_SET_OFM_WIDTH_M1, ofm.shape.width - 1)
    emit.cmd0_with_param(cmd0.NPU_SET_OFM_DEPTH_M1, ofm.shape.depth - 1)
    generate_strides(emit, ofm, cmd1.NPU_SET_OFM_STRIDE_C, cmd1.NPU_SET_OFM_STRIDE_Y, cmd1.NPU_SET_OFM_STRIDE_X)
    emit.cmd0_with_param(cmd0.NPU_SET_OFM_ZERO_POINT, int(ofm.quantization.zero_point))


def generate_kernel(emit: CommandStreamEmitter, kernel: NpuKernel, block_traversal: NpuBlockTraversal):
    """Generates KERNEL related registers"""
    emit.cmd0_with_param(cmd0.NPU_SET_KERNEL_HEIGHT_M1, kernel.dilation_y * (kernel.height - 1))
    emit.cmd0_with_param(cmd0.NPU_SET_KERNEL_WIDTH_M1, kernel.dilation_x * (kernel.width - 1))
    # set kernel x stride low bit
    stride = (kernel.stride_x - 1) & 1
    # set kernel y stride low bit
    stride |= (kernel.stride_y - 1 & 1) << 1
    # set kernel x stride extension bits
    stride |= (kernel.stride_x - 1 >> 1) << 6
    # set kernel y stride extension bits
    stride |= (kernel.stride_y - 1 >> 1) << 9
    stride |= (kernel.dilation_x - 1) << 3
    stride |= (kernel.dilation_y - 1) << 4
    if block_traversal == NpuBlockTraversal.PART_KERNEL_FIRST:
        stride |= 1 << 2
    emit.cmd0_with_param(cmd0.NPU_SET_KERNEL_STRIDE, stride)


def generate_weights(emit: CommandStreamEmitter, weights: List[NpuAddressRange], arch: ArchitectureFeatures):
    """Generates WEIGHT registers"""
    if len(weights) == 0:
        return
    emit.cmd0_with_param(cmd0.NPU_SET_WEIGHT_REGION, weights[0].region)
    # Set weights sources for active and present cores
    for core, (addr, length) in enumerate(
        [
            (cmd1.NPU_SET_WEIGHT_BASE, cmd1.NPU_SET_WEIGHT_LENGTH),
            (cmd1.NPU_SET_WEIGHT1_BASE, cmd1.NPU_SET_WEIGHT1_LENGTH),
        ]
    ):
        if core < len(weights):
            emit.cmd1_with_offset(addr, weights[core].address)
            emit.cmd1_with_offset(length, weights[core].length)
        elif core < arch.ncores:
            emit.cmd1_with_offset(addr, weights[0].address)
            emit.cmd1_with_offset(length, 0)


def generate_biases(emit: CommandStreamEmitter, biases: List[NpuAddressRange], arch: ArchitectureFeatures):
    """Generates SCALE registers"""
    if len(biases) == 0:
        return
    emit.cmd0_with_param(cmd0.NPU_SET_SCALE_REGION, biases[0].region)
    # Set weights sources for active and present cores
    for core, (addr, length) in enumerate(
        [(cmd1.NPU_SET_SCALE_BASE, cmd1.NPU_SET_SCALE_LENGTH), (cmd1.NPU_SET_SCALE1_BASE, cmd1.NPU_SET_SCALE1_LENGTH)]
    ):
        if core < len(biases):
            emit.cmd1_with_offset(addr, biases[core].address)
            emit.cmd1_with_offset(length, biases[core].length)
        elif core < arch.ncores:
            emit.cmd1_with_offset(addr, biases[0].address)
            emit.cmd1_with_offset(length, 0)


def generate_block_config(
    emit: CommandStreamEmitter,
    npu_op: NpuBlockOperation,
    arch: ArchitectureFeatures,
    shared_buffer: SharedBufferAllocation,
):
    """Generates OFM_BLK_HEIGHT/WIDTH/DEPTH registers"""
    block_config = npu_op.block_config
    assert block_config is not None, "block_config has not been set"
    alloc = shared_buffer.try_block(Block(block_config.width, block_config.height, block_config.depth))
    assert alloc is not None, f"Block config {block_config} does not fit, op: {npu_op.op_type}"
    emit.cmd0_with_param(cmd0.NPU_SET_OFM_BLK_HEIGHT_M1, block_config.height - 1)
    emit.cmd0_with_param(cmd0.NPU_SET_OFM_BLK_WIDTH_M1, block_config.width - 1)
    emit.cmd0_with_param(cmd0.NPU_SET_OFM_BLK_DEPTH_M1, block_config.depth - 1)


def generate_shram_registers_elementwise(
    emit: CommandStreamEmitter,
    npu_op: NpuElementWiseOperation,
    arch: ArchitectureFeatures,
    shared_buffer: SharedBufferAllocation,
):
    """Generates IB_END/IB_START/AB_START registers for elementwise operations"""
    # For elementwise set the required SHRAM to be equal to the total size of available SHRAM
    uses_lut = npu_op.activation is not None and npu_op.activation.op_type == NpuActivationOp.TABLE_LOOKUP
    shram_required = arch.available_shram_banks(uses_lut)

    # Acc buffers not needed so set AB_START to size of SHRAM
    emit.cmd0_with_param(cmd0.NPU_SET_IFM_IB_END, shram_required)
    emit.cmd0_with_param(cmd0.NPU_SET_AB_START, shram_required)
    if has_ifm2(npu_op):
        # Set IFM2_IB_START to the latter half of the IB space
        ifm_ib_start = shared_buffer.bank_locations[SharedBufferArea.IFM]
        emit.cmd0_with_param(
            cmd0.NPU_SET_IFM2_IB_START, (shram_required - ifm_ib_start) // shared_buffer.ifm_count + ifm_ib_start,
        )
    emit.cmd0_with_param(cmd0.NPU_SET_ACC_FORMAT, acc_format_map[shared_buffer.use_accumulator_element])


def generate_shram_registers_non_elementwise(emit: CommandStreamEmitter, shared_buffer: SharedBufferAllocation):
    """Generates IB_END/IB_START/AB_START registers for non-elementwise operations"""
    emit.cmd0_with_param(
        cmd0.NPU_SET_IFM_IB_END,
        shared_buffer.bank_locations[SharedBufferArea.IFM] + shared_buffer.banks_required[SharedBufferArea.IFM],
    )
    emit.cmd0_with_param(cmd0.NPU_SET_AB_START, shared_buffer.bank_locations[SharedBufferArea.Accumulators])
    emit.cmd0_with_param(cmd0.NPU_SET_ACC_FORMAT, acc_format_map[shared_buffer.use_accumulator_element])


def create_shared_buffer(npu_op: NpuBlockOperation, arch: ArchitectureFeatures) -> SharedBufferAllocation:
    """Creates shared buffer allocation for the given operation"""
    op_type = npu_op.op_type
    block_type = NpuBlockType.Default
    if op_type == NpuOperationType.Conv2D:
        block_type = NpuBlockType.ConvolutionMxN
    elif op_type == NpuOperationType.ConvDepthWise:
        block_type = NpuBlockType.ConvolutionDepthWise
    elif op_type == NpuOperationType.Pooling:
        block_type = NpuBlockType.ReduceSum if npu_op.sub_op_type == NpuPoolingOp.REDUCE_SUM else NpuBlockType.Pooling
    elif op_type == NpuOperationType.ElementWise:
        block_type = NpuBlockType.ElementWise
    else:
        assert 0, "Unsupported operation"
    ifm_resampling_mode = resampling_mode_map[npu_op.ifm_upscale]
    return shared_buffer_allocation_for_npu_op(arch, npu_op, block_type, ifm_resampling_mode)


def generate_common(
    emit: CommandStreamEmitter,
    npu_op: NpuBlockOperation,
    block_traversal: NpuBlockTraversal,
    arch: ArchitectureFeatures,
    use_global_scale: bool = False,
    op_to_scale: int = 0,
):
    """Generate registers that are common to most operations"""
    assert npu_op.ifm is not None and npu_op.ofm is not None
    generate_ifm(emit, npu_op.ifm)
    generate_ifm_precision(emit, npu_op.ifm, op_to_scale, cmd0.NPU_SET_IFM_PRECISION)
    emit.cmd0_with_param(cmd0.NPU_SET_IFM_UPSCALE, resampling_mode_map[npu_op.ifm_upscale])
    if npu_op.padding is not None:
        generate_padding(emit, npu_op.padding)
    generate_ofm(emit, npu_op.ofm)
    generate_ofm_precision(emit, npu_op, use_global_scale)
    if npu_op.op_type != NpuOperationType.ElementWise:
        assert npu_op.kernel is not None
        generate_kernel(emit, npu_op.kernel, block_traversal)
    generate_weights(emit, npu_op.weights, arch)
    generate_biases(emit, npu_op.biases, arch)
    generate_activation(emit, npu_op.activation, npu_op.ofm)
    shared_buffer = create_shared_buffer(npu_op, arch)
    generate_block_config(emit, npu_op, arch, shared_buffer)
    if npu_op.op_type == NpuOperationType.ElementWise:
        generate_shram_registers_elementwise(emit, npu_op, arch, shared_buffer)
    else:
        generate_shram_registers_non_elementwise(emit, shared_buffer)


# -------------------------------------------------------------------
# SCALING
# -------------------------------------------------------------------


def generate_ofm_scaling_for_pooling(emit: CommandStreamEmitter, pool_op: NpuPoolingOperation):
    """Generates OFM_SCALE register for pooling operations"""
    # For valid padding vela has to output scaling values
    kernel = pool_op.kernel
    ifm_quant = pool_op.ifm.quantization
    ofm_quant = pool_op.ofm.quantization
    if pool_op.activation is not None and pool_op.activation.op_type in (NpuActivationOp.SIGMOID, NpuActivationOp.TANH):
        assert ifm_quant.scale_f32 is not None
        rescale = 0x3000 * ifm_quant.scale_f32
        if pool_op.ifm.data_type == NpuDataType.INT16:
            # Calculate scale and shift for the output scale of 1/(3*4096)
            shift = 0
            max_rescale = np.iinfo(np.int16).max / 2
            while rescale <= max_rescale and shift <= 30:
                shift += 1
                rescale *= 2
            scale = int(rescale)
        else:
            rescale_bits = len(bin(round_up_to_int(rescale))) - 2 + 1
            scale, shift = scaling.quantise_pooling_scale(kernel.height * kernel.width, rescale_bits)
            scale = int(round_away_zero(scale * rescale))
    elif pool_op.fused_quantize:
        # Quantize op requires different scaling
        ifm_scale_f64 = np.double(ifm_quant.scale_f32)
        ofm_scale_f64 = np.double(ofm_quant.scale_f32)
        scale, shift = scaling.quantise_scale(ifm_scale_f64 / ofm_scale_f64)
    elif pool_op.rescale is not None:
        # for ResizeBilinear operations with "rescale" in primary_op.attrs
        rescale = pool_op.rescale
        rescale_bits = len(bin(round_up_to_int(rescale))) - 2 + 1
        scale, shift = scaling.quantise_pooling_scale(kernel.height * kernel.width, rescale_bits)
        scale = int(round_away_zero(scale * rescale))
    else:
        # In case avg pool fused with concat or other memory operation, rescaling might be needed.
        # kernel height == kernel width == 1 is always true in this case
        # Normally the scale is maximised, to get maximum precision, which means that
        # if rescale != 1, scale need to consider the number of bits needed for rescaling
        if ofm_quant.scale_f32 is not None and ifm_quant.scale_f32 is not None:
            rescale = ifm_quant.scale_f32 / ofm_quant.scale_f32
            rescale_bits = 0
            if kernel.height == kernel.width == 1:
                if rescale > 1:
                    rescale_bits = len(bin(round_up_to_int(rescale))) - 2 + 1
                elif rescale < 1:
                    rescale_bits = -(len(bin(round_up_to_int(1 / rescale))) - 2 - 1)
            scale, shift = scaling.quantise_pooling_scale(kernel.height * kernel.width, rescale_bits)
            scale = int(round_away_zero(scale * rescale))
        else:
            scale = 1
            shift = 0

    emit.cmd1_with_offset(cmd1.NPU_SET_OFM_SCALE, scale, shift)


def generate_scaling_for_elementwise(emit: CommandStreamEmitter, npu_op: NpuElementWiseOperation) -> int:
    """
    Generates OFM/OPA/OPB_SCALE registers for elementwise operators.
    Returns the operator to scale
    """
    op_to_scale = 0
    if npu_op.sub_op_type in (NpuElementWiseOp.ADD, NpuElementWiseOp.MUL, NpuElementWiseOp.SUB):
        input_scale = npu_op.ifm.quantization.scale_f32 if npu_op.ifm.quantization else None
        input2_scale = npu_op.ifm2.quantization.scale_f32 if npu_op.ifm2.quantization else None
        output_scale = npu_op.ofm.quantization.scale_f32 if npu_op.ofm.quantization else None

        if npu_op.activation is not None and npu_op.activation.op_type in (
            NpuActivationOp.SIGMOID,
            NpuActivationOp.TANH,
        ):
            output_scale = 1 / 0x3000

        if npu_op.sub_op_type == NpuElementWiseOp.MUL:
            if None in (input_scale, input2_scale, output_scale):
                ofm_scale = 1
                shift = 0
            else:
                ofm_scale, shift = scaling.elementwise_mul_scale(input_scale, input2_scale, output_scale)
            emit.cmd1_with_offset(cmd1.NPU_SET_OFM_SCALE, ofm_scale, shift)
        else:  # Add/Sub
            if None in (input_scale, input2_scale, output_scale):
                opa_scale = opb_scale = ofm_scale = 1
                opa_shift = shift = 0
                if npu_op.rescale is not None:
                    ofm_scale, shift = npu_op.rescale
            elif input_scale == input2_scale:
                opa_scale, opb_scale, ofm_scale, shift = scaling.simplified_elementwise_add_sub_scale(
                    input_scale, input2_scale, output_scale
                )
                opa_shift = 0  # Unused for this case
            else:
                # Use advanced implementation only when input scales differ
                bitdepth = npu_op.ifm.data_type.size_in_bits()
                (opa_scale, opa_shift, ofm_scale, shift, op_to_scale,) = scaling.advanced_elementwise_add_sub_scale(
                    input_scale, input2_scale, output_scale, bitdepth
                )
                opb_scale = 0  # Unused for this case
                if npu_op.reversed_operands:
                    # If the operand order is reversed we also have to swap which operand is scaled
                    if op_to_scale == scaling.OperandToScale.OPa:
                        op_to_scale = scaling.OperandToScale.OPb
                    else:
                        op_to_scale = scaling.OperandToScale.OPa
            emit.cmd1_with_offset(cmd1.NPU_SET_OPA_SCALE, opa_scale, opa_shift)
            emit.cmd1_with_offset(cmd1.NPU_SET_OPB_SCALE, opb_scale)
            emit.cmd1_with_offset(cmd1.NPU_SET_OFM_SCALE, ofm_scale, shift)
    elif npu_op.sub_op_type in (NpuElementWiseOp.LRELU, NpuElementWiseOp.ABS):
        output_scale = npu_op.ofm.quantization.scale_f32
        ofm_scale, shift = scaling.quantise_scale(output_scale)
        emit.cmd1_with_offset(cmd1.NPU_SET_OFM_SCALE, ofm_scale, shift)
    else:
        emit.cmd1_with_offset(cmd1.NPU_SET_OFM_SCALE, 1, 0)
    return op_to_scale


# -------------------------------------------------------------------
# ADDRESSING/STRIDES (helper functions)
# -------------------------------------------------------------------


def ranges_overlap(range1: NpuAddressRange, range2: NpuAddressRange) -> bool:
    """Checks if the ranges overlap"""
    return range1.region == range2.region and numeric_util.overlaps(
        range1.address, range1.address + range1.length, range2.address, range2.address + range2.length
    )


def range_lists_overlap(list1: List[Optional[NpuAddressRange]], list2: List[Optional[NpuAddressRange]]) -> bool:
    """Checks if there is any address overlap between list1 and list2"""
    for range1 in list1:
        if range1 is None:
            continue
        for range2 in list2:
            if range2 is not None and ranges_overlap(range1, range2):
                return True
    return False


def get_strides(fm: NpuFeatureMap) -> NpuShape3D:
    """Calculates STRIDE_C/Y/X"""
    if fm.strides is not None:
        return fm.strides
    elem_size = fm.data_type.size_in_bytes()
    if fm.layout == NpuLayout.NHWC:
        stride_c = elem_size
        stride_x = fm.shape.depth * stride_c
        stride_y = fm.shape.width * stride_x
    else:
        stride_x = 16 * elem_size
        stride_c = stride_x * fm.shape.width
        stride_y = elem_size * fm.shape.width * numeric_util.round_up(fm.shape.depth, 16)
    return NpuShape3D(depth=stride_c, height=stride_y, width=stride_x)


def get_address(fm: NpuFeatureMap, strides: NpuShape3D, y: int, x: int, c: int) -> int:
    """Returns address of given coordinate"""
    t = 0
    BRICK = 16
    stride_c = BRICK * fm.data_type.size_in_bytes() if fm.layout == NpuLayout.NHWC else strides.depth
    stride_x = BRICK * fm.data_type.size_in_bytes() if fm.layout == NpuLayout.NHCWB16 else strides.width
    if x >= fm.tiles.width_0:
        x -= fm.tiles.width_0
        t = 1
        if y >= fm.tiles.height_1:
            y -= fm.tiles.height_1
            t += 2
    elif y >= fm.tiles.height_0:
        y -= fm.tiles.height_0
        t += 2
    elem_size = fm.data_type.size_in_bytes()
    return (
        fm.tiles.addresses[t] + y * strides.height + x * stride_x + (c // BRICK) * stride_c + int(c % BRICK) * elem_size
    )


def get_address_range(
    fm: NpuFeatureMap, strides: NpuShape3D, y0: int, x0: int, c0: int, y1: int, x1: int, c1: int
) -> NpuAddressRange:
    """
    Gets address range for (y0, x0, c0) - (y1, x1, c1) (inclusive, so the second coordinate is within the fm).
    The begin and end coordinates must be within the same tile.
    """
    addr0 = get_address(fm, strides, y0, x0, c0)
    addr1 = get_address(fm, strides, y1, x1, c1)
    return NpuAddressRange(region=fm.region, address=addr0, length=addr1 - addr0 + fm.data_type.size_in_bytes())


def get_h_ranges(
    fm: NpuFeatureMap, strides: NpuShape3D, y0: int, x0: int, c0: int, y1: int, x1: int, c1: int
) -> List[NpuAddressRange]:
    """
    Gets address ranges for (y0, x0, c0) - (y1, x1, c1) (inclusive, so the second coordinate is within the fm);
    the begin and end coordinates must be within the same tile.
    Divides the area in horizontal "stripes" of height 1, and returns the address ranges for these "stripes".
    """
    return [get_address_range(fm, strides, y, x0, c0, y, x1, c1) for y in range(y0, y1 + 1)]


def get_address_ranges_for_area(
    fm: NpuFeatureMap, y0: int, x0: int, c0: int, y1: int, x1: int, c1: int
) -> List[NpuAddressRange]:
    """
    Returns a list of adddress ranges that covers the area (y0, x0, c0) - (y1, x1, c1) (inclusive).
    Divides the area in horizontal "stripes" of height 1, and returns the address ranges for these "stripes".

    For example, for the area marked with X (in a feature map with 4 tiles) as input, this function would return
    6 address ranges: the address ranges for 1-height areas [AAA, BBB, CC, DD, EEE, FF]

        .....|....           .....|....
     t0 ..XXX|XX.. t1     t0 ..AAA|CC.. t1
        ..XXX|XX..           ..BBB|DD..
        -----+----    -->    -----+----
     t2 ..XXX|XX.. t3     t2 ..EEE|FF.. t3
        .....|....           .....|....
    """
    strides = get_strides(fm)
    height_0, height_1, width_0 = fm.tiles.height_0, fm.tiles.height_1, fm.tiles.width_0
    h, w, c = fm.shape
    y2, x2, c2 = min(y1, h - 1), min(x1, w - 1), min(c1, c - 1)
    ranges = []
    if x0 < width_0 and y0 < height_0:
        # Horizontal ranges for tile 0
        ranges.extend(get_h_ranges(fm, strides, y0, x0, c0, min(y2, height_0 - 1), min(x2, width_0 - 1), c2))
    if x2 >= width_0 and y0 < height_1:
        # Horizontal ranges for tile 1
        ranges.extend(get_h_ranges(fm, strides, y0, max(x0, width_0), c0, min(y2, height_1 - 1), x2, c2))
    if x0 < width_0 and y2 >= height_0:
        # Horizontal ranges for tile 2
        ranges.extend(get_h_ranges(fm, strides, max(y0, height_0), x0, c0, y2, min(x2, width_0 - 1), c2))
    if x2 >= width_0 and y2 >= height_1:
        # Horizontal ranges for tile 3
        ranges.extend(get_h_ranges(fm, strides, max(y0, height_1), max(x0, width_0), c0, y2, x2, c2))
    return ranges


def get_address_ranges(fm: NpuFeatureMap) -> List[Optional[NpuAddressRange]]:
    """Returns 4 adddress ranges, one for every tile, None if the tile is not in use"""
    strides = get_strides(fm)
    height, width, depth = fm.shape.height, fm.shape.width, fm.shape.depth
    height_0, height_1, width_0 = fm.tiles.height_0, fm.tiles.height_1, fm.tiles.width_0
    t0 = get_address_range(fm, strides, 0, 0, 0, min(height, height_0) - 1, min(width, width_0) - 1, depth - 1,)
    if width > width_0:
        t1 = get_address_range(fm, strides, 0, width_0, 0, min(height, height_1) - 1, width - 1, depth - 1)
    else:
        t1 = None
    if height > height_0:
        t2 = get_address_range(fm, strides, height_0, 0, 0, height - 1, min(width, width_0) - 1, depth - 1)
    else:
        t2 = None
    if t1 is not None and t2 is not None:
        t3 = get_address_range(fm, strides, height_1, width_0, 0, height - 1, width - 1, depth - 1)
    else:
        t3 = None
    return [t0, t1, t2, t3]


# -------------------------------------------------------------------
# DMA_WAIT/KERNEL_WAIT
# -------------------------------------------------------------------


Watermark = namedtuple("Watermark", ["npu", "dma"])


def memory_range_set(range: NpuAddressRange) -> MemoryRangeSet:
    return MemoryRangeSet(range.region, range.address, range.address + range.length)


def get_dma_memory_accesses(dma_op: NpuDmaOperation) -> MemoryAccessSet:
    """Returns the address that are read and written by the given DMA operation"""
    res = MemoryAccessSet()
    res.add(memory_range_set(dma_op.src), AccessDirection.Read)
    res.add(memory_range_set(dma_op.dest), AccessDirection.Write)
    return res


def get_op_memory_accesses(npu_op: NpuBlockOperation, arch: ArchitectureFeatures) -> MemoryAccessSet:
    """Returns the addresses that are read and written by the given operation"""
    assert npu_op.ifm is not None and npu_op.ofm is not None
    # Read addresses
    read_ranges = get_address_ranges(npu_op.ifm)
    if has_ifm2(npu_op):
        assert npu_op.ifm2 is not None
        read_ranges.extend(get_address_ranges(npu_op.ifm2))
    read_ranges.extend(npu_op.weights)
    read_ranges.extend(npu_op.biases)
    if npu_op.activation is not None and npu_op.activation.op_type == NpuActivationOp.TABLE_LOOKUP:
        address = arch.available_shram_banks(True) * arch.shram_bank_size
        read_ranges.append(NpuAddressRange(region=BasePointerIndex.Mem2Mem, address=address, length=2048))
    # Written addresses
    write_ranges = get_address_ranges(npu_op.ofm)
    # Add write access to SHRAM, needed when LUTs can overwrite accumulator banks
    uses_lut = npu_op.activation is not None and npu_op.activation.op_type == NpuActivationOp.TABLE_LOOKUP
    written_shram_size = arch.available_shram_banks(uses_lut) * arch.shram_bank_size
    write_ranges.append(NpuAddressRange(region=BasePointerIndex.Mem2Mem, address=0, length=written_shram_size))

    res = MemoryAccessSet()
    for read_range in read_ranges:
        if read_range is not None:
            res.add(memory_range_set(read_range), AccessDirection.Read)
    for write_range in write_ranges:
        if write_range is not None:
            res.add(memory_range_set(write_range), AccessDirection.Write)
    return res


def get_wait_dependency(
    arch: ArchitectureFeatures, npu_op_list: List[NpuOperation], memory_accesses, op_index: int, watermark: Watermark
):
    """Used to calculate whether DMA wait or kernel wait operations are needed"""
    npu_op = npu_op_list[op_index]
    op_access = memory_accesses[npu_op]
    index = op_index - 1

    # NPU dependency tracking
    npu_outstanding = -1
    npu_ops = 0
    npu_index = watermark.npu

    # DMA dependency tracking
    dma_outstanding = -1
    dma_ops = 0
    dma_index = watermark.dma

    # Seek back in the command stream looking for NPU or DMA dependencies
    # but only as far as the first dependency or the watermarks (dependencies
    # before this point have been satisfied already).
    # The watermark moves to after the latest element we must wait for, not
    # the command that issues the wait.
    # NPU->NPU dependency is handled via blockdep.
    while (index >= npu_index) or (index >= dma_index):
        prev_op = npu_op_list[index]
        prev_access = memory_accesses[prev_op]

        # Check NPU consuming DMA output
        if is_dma_op(prev_op):
            if index >= dma_index:
                if not is_dma_op(npu_op):
                    if (dma_outstanding == -1) and prev_access.conflicts(op_access):
                        dma_outstanding = dma_ops
                dma_ops += 1  # Count DMA ops in the pipeline
                if dma_ops >= arch.max_outstanding_dma:
                    dma_index = max(index + 1, dma_index)
        # Check DMA consuming NPU output
        else:
            if index >= npu_index:
                if is_dma_op(npu_op) and npu_outstanding == -1 and prev_access.conflicts(op_access):
                    npu_outstanding = npu_ops
                npu_ops += 1  # Count NPU ops in the pipeline
                if npu_ops >= arch.max_outstanding_kernels:
                    npu_index = max(index + 1, npu_index)

        index -= 1

    # Update DMA watermark if we didn't see any and the NPU pipeline is full
    if (dma_ops == 0) and (npu_ops >= arch.max_outstanding_kernels):
        dma_index = op_index

    # Bring the search watermark forwards as we complete for those dependencies
    watermark = Watermark(npu_index, dma_index)
    outstanding = Watermark(npu_outstanding, dma_outstanding)

    return watermark, outstanding


def generate_cmd_waits(emit: CommandStreamEmitter, cmd_waits: Watermark):
    if cmd_waits.npu >= 0:
        emit.cmd_wait(cmd0.NPU_OP_KERNEL_WAIT, 0, cmd_waits.npu)

    if cmd_waits.dma >= 0:
        emit.cmd_wait(cmd0.NPU_OP_DMA_WAIT, 0, cmd_waits.dma)


# -------------------------------------------------------------------
# BLOCKDEP
# -------------------------------------------------------------------


def shape3d_size(shape: NpuShape3D) -> int:
    return shape.width * shape.height * shape.depth


def shape3d_to_rect(shape: NpuShape3D) -> Rect:
    return Rect(0, 0, 0, shape.width - 1, shape.height - 1, shape.depth - 1)


def get_ifm_ofm_block_depth(arch: ArchitectureFeatures, npu_op: NpuBlockOperation) -> int:
    # Note: NOT equivalent to the normal ifm block depth calculation since
    # it takes into account 'depthless' block operations by returning full
    # depth
    if npu_op.op_type == NpuOperationType.Conv2D:
        res = arch.calc_ifm_block_depth(npu_op.ifm.shape.depth, npu_op.ifm.data_type.size_in_bits())
        return res
    return npu_op.ofm.shape.depth


def calc_blockdep(arch: ArchitectureFeatures, prev_op: Optional[NpuBlockOperation], npu_op: NpuBlockOperation,) -> int:
    """Calculates the value of the BLOCKDEP register"""
    if prev_op is None:
        return 0
    assert npu_op.ifm is not None
    assert prev_op.ofm is not None
    # Check if IFM or IFM2 overlaps with prev op's OFM
    prev_ofm_ranges = get_address_ranges(prev_op.ofm)
    ifm_ranges = get_address_ranges(npu_op.ifm)
    ifm_overlaps = range_lists_overlap(prev_ofm_ranges, ifm_ranges)
    if has_ifm2(npu_op):
        assert npu_op.ifm2 is not None
        ifm2_ranges = get_address_ranges(npu_op.ifm2)
        ifm2_overlaps = range_lists_overlap(prev_ofm_ranges, ifm2_ranges)
    else:
        ifm2_overlaps = False
    if ifm_overlaps and ifm2_overlaps:
        # Both IFM and IFM2 overlap (should be rare)
        return 0
    if not ifm_overlaps and not ifm2_overlaps:
        # No overlap between prev OFM and IFM/IFM2
        return ArchitectureFeatures.MAX_BLOCKDEP
    if ifm2_overlaps and shape3d_size(npu_op.ifm2.shape) < shape3d_size(npu_op.ifm.shape):
        # Prev OFM produces IFM2 which is broadcasted (this should be rare)
        return 0
    prev_block_config = prev_op.block_config
    block_config = npu_op.block_config
    overlapping_fm = npu_op.ifm if ifm_overlaps else npu_op.ifm2
    assert overlapping_fm is not None

    def intersects(ifm_start_coord: Tuple, ifm_end_coord: Tuple, ofm_start_coord: Tuple, ofm_end_coord: Tuple) -> bool:
        """Checks if the given IFM area overlaps with the given OFM area"""
        if overlapping_fm.shape == prev_op.ofm.shape and overlapping_fm.tiles == prev_op.ofm.tiles:
            # Common case: prev_op.ofm == op.ifm; in this case it suffices to check
            # if the xyz coordinates overlap, which is quick and easy
            return ArchitectureFeatures.intersects(ifm_start_coord, ifm_end_coord, ofm_start_coord, ofm_end_coord)
        # The OFM produces a part of the IFM (e.g. a stripe), or the IFM consumes part of the OFM.
        # In this case address comparison is needed between the two areas
        x0, y0, c0 = ifm_start_coord
        x1, y1, c1 = ifm_end_coord
        ifm_ranges = get_address_ranges_for_area(overlapping_fm, y0, x0, c0, y1, x1, c1)
        x0, y0, c0 = ofm_start_coord
        x1, y1, c1 = ofm_end_coord
        prev_ofm_ranges = get_address_ranges_for_area(prev_op.ofm, y0, x0, c0, y1, x1, c1)
        return range_lists_overlap(ifm_ranges, prev_ofm_ranges)

    prev_ofm_block = Block(prev_block_config.width, prev_block_config.height, prev_block_config.depth)
    prev_ofm_rect = shape3d_to_rect(prev_op.ofm.shape)
    cur_ifm_block_depth = get_ifm_ofm_block_depth(arch, npu_op)
    cur_ofm_block = Block(block_config.width, block_config.height, block_config.depth)
    cur_ofm_rect = shape3d_to_rect(npu_op.ofm.shape)
    cur_ifm_rect = shape3d_to_rect(npu_op.ifm.shape)
    cur_padLT = (0, 0) if npu_op.padding is None else (npu_op.padding.left, npu_op.padding.top)
    return arch.calc_block_dep(
        prev_ofm_rect,
        prev_ofm_block,
        cur_ifm_rect,
        cur_ofm_rect,
        cur_ifm_block_depth,
        cur_ofm_block,
        to_kernel(npu_op.kernel),
        cur_padLT,
        intersects=intersects,
    )


# -------------------------------------------------------------------
# PRINT
# -------------------------------------------------------------------


def print_feature_map(fm: NpuFeatureMap, name: str):
    if fm is not None:
        q = (
            "no quantization"
            if fm.quantization is None
            else f"scale: {fm.quantization.scale_f32}, zero: {fm.quantization.zero_point}"
        )
        h, w, c = fm.shape
        sz = h * w * c * fm.data_type.size_in_bytes()
        print(f"      {name}: h={h},w={w},c={c}, region={fm.region}, {fm.layout}, {fm.data_type}, size={sz}, {q}")
        strides = get_strides(fm)
        stride_str = f"Stride y/x/c: {strides.height}/{strides.width}/{strides.depth}"
        t = fm.tiles
        addresses = [hex(addr) for addr in t.addresses]
        print(f"         {stride_str}, tiles: w0={t.width_0}, h0={t.height_0}, h1={t.height_1}, base={addresses}")


def print_operation(npu_op: NpuOperation, index: int = 0):
    pass_info = f", {npu_op.cmd}" if hasattr(npu_op, "cmd") else ""
    if is_dma_op(npu_op):
        print(f"{index} DMA_START src={npu_op.src}, dest={npu_op.dest}{pass_info}")
        return
    k = None if npu_op.kernel is None else to_kernel(npu_op.kernel)
    if npu_op.op_type in (NpuOperationType.Pooling, NpuOperationType.ElementWise):
        print(f"{index} {npu_op.sub_op_type.name} {npu_op.op_type.name}:{pass_info}")
    else:
        if (
            npu_op.op_type == NpuOperationType.Conv2D
            and k.elements_wh() * k.stride.x * k.stride.y * k.dilation.x * k.dilation.y == 1
        ):
            fc = "FullyConnected "
        else:
            fc = ""
        print(f"{index} {fc}{npu_op.op_type.name}{pass_info}")
    print_feature_map(npu_op.ifm, "IFM")
    if npu_op.ifm2_scalar is not None:
        quant_val = quantise(npu_op.ifm2_scalar, npu_op.ifm2.quantization)
        print(f"      IFM2: Scalar={npu_op.ifm2_scalar} (quantized: {quant_val}), {npu_op.ifm2.quantization}")
    else:
        print_feature_map(npu_op.ifm2, "IFM2")
    print_feature_map(npu_op.ofm, "OFM")
    if k is not None and npu_op.op_type != NpuOperationType.ElementWise:
        print(f"      Kernel: {k}")
    if npu_op.padding is not None:
        print(f"      {npu_op.padding}")
    for weights in npu_op.weights:
        print(f"      Weights: {weights}")
    for bias in npu_op.biases:
        print(f"      Scales: {bias}")
    if npu_op.activation is not None:
        act = npu_op.activation
        if act.op_type != NpuActivationOp.NONE_OR_RELU or act.min is not None or act.max is not None:
            lut = f", lut index={act.lookup_table_index}" if act.op_type == NpuActivationOp.TABLE_LOOKUP else ""
            print(f"      Activation: {act.op_type.name}, min={act.min}, max={act.max}{lut}")
    if npu_op.op_type == NpuOperationType.Conv2D:
        print(f"      {npu_op.block_traversal}")
    bh, bw, bc = npu_op.block_config
    rescale = f", rescale={npu_op.rescale}" if hasattr(npu_op, "rescale") else ""
    print(f"      Block config: h={bh},w={bw},c={bc}, {npu_op.ifm_upscale}, {npu_op.rounding_mode}{rescale}")


def print_operations(npu_op_list: List[NpuOperation]):
    for index, npu_op in enumerate(npu_op_list):
        print_operation(npu_op, index)


# -------------------------------------------------------------------
# OPERATIONS
# -------------------------------------------------------------------


def generate_operation_code(emit: CommandStreamEmitter, npu_op: NpuOperation):
    """Generates NPU_OP_* command"""
    op_type = npu_op.op_type
    if op_type == NpuOperationType.Dma:
        emit.cmd_do_operation(cmd0.NPU_OP_DMA_START, npu_op.channel * 16 + npu_op.mode)
    elif op_type == NpuOperationType.Conv2D:
        emit.cmd_do_operation(cmd0.NPU_OP_CONV)
    elif op_type == NpuOperationType.ConvDepthWise:
        emit.cmd_do_operation(cmd0.NPU_OP_DEPTHWISE)
    elif op_type == NpuOperationType.Pooling:
        emit.cmd_do_operation(cmd0.NPU_OP_POOL, param=pooling_op_map[npu_op.sub_op_type])
    elif op_type == NpuOperationType.ElementWise:
        emit.cmd_do_operation(cmd0.NPU_OP_ELEMENTWISE, param=elementwise_op_map[npu_op.sub_op_type])
    else:
        assert 0, "Unsupported operation"


def generate_conv2d_op(emit: CommandStreamEmitter, npu_op: NpuConv2DOperation, arch: ArchitectureFeatures):
    """Generates register commands for Conv2D operations"""
    generate_common(emit, npu_op, npu_op.block_traversal, arch)


def generate_conv_depthwise_op(emit: CommandStreamEmitter, npu_op: NpuPoolingOperation, arch: ArchitectureFeatures):
    """Generates register commands for depthwise convolution operations"""
    generate_common(emit, npu_op, NpuBlockTraversal.DEPTH_FIRST, arch)


def generate_pooling_op(emit: CommandStreamEmitter, npu_op: NpuPoolingOperation, arch: ArchitectureFeatures):
    """Generates register commands for pooling operations"""
    use_global_scale = (
        npu_op.sub_op_type in (NpuPoolingOp.AVERAGE, NpuPoolingOp.REDUCE_SUM) and sum(npu_op.padding) == 0
    )
    generate_common(emit, npu_op, NpuBlockTraversal.DEPTH_FIRST, arch, use_global_scale=use_global_scale)
    # Pooling op specific
    if use_global_scale:
        generate_ofm_scaling_for_pooling(emit, npu_op)


def generate_elementwise_op(emit: CommandStreamEmitter, npu_op: NpuElementWiseOperation, arch: ArchitectureFeatures):
    """Generates register commands for elementwise operations"""
    use_global_scale = npu_op.sub_op_type in (
        NpuElementWiseOp.ADD,
        NpuElementWiseOp.SUB,
        NpuElementWiseOp.MUL,
        NpuElementWiseOp.LRELU,
        NpuElementWiseOp.ABS,
    )
    op_to_scale = generate_scaling_for_elementwise(emit, npu_op)
    generate_common(
        emit, npu_op, NpuBlockTraversal.DEPTH_FIRST, arch, use_global_scale=use_global_scale, op_to_scale=op_to_scale
    )
    # Elementwise op specific
    if npu_op.sub_op_type not in unary_elementwise_ops:
        # Binary operation; generate IFM2 registers
        assert npu_op.ifm2 is not None
        has_scalar = npu_op.ifm2_scalar is not None
        generate_ifm2(emit, npu_op.ifm2, has_scalar)
        generate_ifm_precision(emit, npu_op.ifm2, 0, cmd0.NPU_SET_IFM2_PRECISION)
        generate_ifm2_broadcast(emit, npu_op)
        if has_scalar:
            quantized_scalar = quantise(npu_op.ifm2_scalar, npu_op.ifm2.quantization)
            assert npu_op.ifm2.data_type.min_value() <= quantized_scalar <= npu_op.ifm2.data_type.max_value()
            emit.cmd0_with_param(cmd0.NPU_SET_IFM2_SCALAR, quantized_scalar)


def generate_dma_op(emit: CommandStreamEmitter, dma_op: NpuDmaOperation):
    """Generates register commands for DMA operations"""
    emit.cmd0_with_param(cmd0.NPU_SET_DMA0_SRC_REGION, dma_op.src.region)
    emit.cmd1_with_offset(cmd1.NPU_SET_DMA0_SRC, dma_op.src.address)
    emit.cmd0_with_param(cmd0.NPU_SET_DMA0_DST_REGION, dma_op.dest.region)

    emit.cmd1_with_offset(cmd1.NPU_SET_DMA0_DST, dma_op.dest.address)
    emit.cmd1_with_offset(cmd1.NPU_SET_DMA0_LEN, dma_op.src.length)


def generate_registers_for_op(emit: CommandStreamEmitter, npu_op: NpuOperation, arch: ArchitectureFeatures):
    """
    Generates register commands for the given operation, but not the final NPU_OP_... command.
    Returns the selected block config
    """
    op_type = npu_op.op_type
    if op_type == NpuOperationType.Conv2D:
        generate_conv2d_op(emit, npu_op, arch)
    elif op_type == NpuOperationType.ConvDepthWise:
        generate_conv_depthwise_op(emit, npu_op, arch)
    elif op_type == NpuOperationType.Pooling:
        generate_pooling_op(emit, npu_op, arch)
    elif op_type == NpuOperationType.ElementWise:
        generate_elementwise_op(emit, npu_op, arch)
    elif op_type == NpuOperationType.Dma:
        generate_dma_op(emit, npu_op)
    else:
        assert 0, "Unsupported operation"


def generate_command_stream(
    emit: CommandStreamEmitter, npu_op_list: List[NpuOperation], arch: ArchitectureFeatures, add_to_debug_db=None
):
    """Generates register commands for the given list of NPU operations"""
    # Calculate memory accesses for every operation
    memory_accesses = {}
    for npu_op in npu_op_list:
        if is_dma_op(npu_op):
            memory_accesses[npu_op] = get_dma_memory_accesses(npu_op)
        else:
            memory_accesses[npu_op] = get_op_memory_accesses(npu_op, arch)
    if arch.is_ethos_u65_system:
        emit.cmd0_with_param(cmd0.NPU_SET_PARALLEL_MODE, arch.ncores - 1)
    dep_watermark = Watermark(0, 0)
    prev_op = None
    # Generate register commands for all operations
    for op_index, npu_op in enumerate(npu_op_list):
        dep_watermark, cmd_waits = get_wait_dependency(arch, npu_op_list, memory_accesses, op_index, dep_watermark)
        generate_registers_for_op(emit, npu_op, arch)
        if not is_dma_op(npu_op):
            # Generate BLOCKDEP
            blockdep = calc_blockdep(arch, prev_op, npu_op)
            blockdep = min(blockdep, arch.max_blockdep)
            emit.cmd0_with_param(cmd0.NPU_SET_BLOCKDEP, blockdep)
            prev_op = npu_op

        generate_cmd_waits(emit, cmd_waits)
        # Generate the actual NPU_OP command
        generate_operation_code(emit, npu_op)
        if add_to_debug_db is not None:
            add_to_debug_db(npu_op, emit.offset)
    # Fill in final part of command stream:
    emit.cmd_do_operation(cmd0.NPU_OP_STOP, param=0xFFFF)


def generate_register_command_stream_for_sg(nng, sg, arch, verbose=False):
    """Generates command stream for the subgraph, adds it to sg.register_command_stream"""
    # Convert high level command stream to list of NpuOperation
    npu_op_list = []
    npu_op_to_cmd = dict()  # map from npu op to high level command
    for cmd in sg.high_level_command_stream:
        if cmd.cmdtype == CommandType.NpuStripe and cmd.ps.npu_block_type == NpuBlockType.Default:
            print("Warning: Skipping register command stream generation for", cmd.ps)
        else:
            npu_op = convert_command_to_npu_op(cmd, arch)
            npu_op_list.append(npu_op)
            npu_op_to_cmd[npu_op] = cmd
    if verbose:
        print_operations(npu_op_list)
    # Generate register commands
    stream_id = DebugDatabase.add_stream(sg)
    DebugDatabase.set_stream_offset(sg, 0)  # Default to zero, can only set during file writing
    emit = CommandStreamEmitter()

    def add_to_debug_db(npu_op: NpuOperation, offset: int):
        """Adds info to the debug database"""
        if not is_dma_op(npu_op):
            cmd = npu_op_to_cmd[npu_op]
            DebugDatabase.add_command(stream_id, offset, cmd.ps.primary_op)

    generate_command_stream(emit, npu_op_list, arch, add_to_debug_db)
    sg.register_command_stream = emit.to_list()
    if verbose:
        emit.print_cmds()
        print("number of commands", len(emit.cmd_stream))
        print("command stream length in words", len(sg.register_command_stream))


def find_block_configs(npu_op: NpuOperation, npu_accelerator: NpuAccelerator) -> List[NpuShape3D]:
    """
    Internal implementation of the public facing API for finding block configs.
    """
    if is_dma_op(npu_op):
        return []
    arch = create_default_arch(Accelerator.from_npu_accelerator(npu_accelerator))
    shared_buffer = create_shared_buffer(npu_op, arch)
    blocks = find_suitable_block_configs(arch, shared_buffer)
    return [NpuShape3D(height=block[0], width=block[1], depth=block[3]) for block in blocks]


def generate_register_command_stream(npu_op_list: List[NpuOperation], npu_accelerator: NpuAccelerator) -> List[int]:
    """
    Internal implementation of the public facing API for generating an Ethos-U register command stream.
    Calculates dependencies between commands and inserts wait operations if needed.

    :param npu_op_list: List[NpuOperation] list of high level NPU operations
    :param accelerator: architecture_features.Accelerator enum to pick the correct Ethos-U accelerator
    :return Ethos-U instructions, as a list of 32-bit integers
    """
    accelerator = Accelerator.from_npu_accelerator(npu_accelerator)
    emit = CommandStreamEmitter()
    arch = create_default_arch(accelerator)
    generate_command_stream(emit, npu_op_list, arch)
    return emit.to_list()
