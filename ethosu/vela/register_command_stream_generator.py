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
# Register level (low-level) command stream generation for Ethos-U. Takes a list of NPU operations and generates
# all the register settings. Calculates dependencies between commands and inserts wait operations. And generates a bit
# stream suitable for interpretation by the Ethos-U processor.
import math
from collections import defaultdict
from enum import Enum
from enum import IntEnum
from typing import cast
from typing import Dict
from typing import List
from typing import Optional

import numpy as np

from . import scaling
from .api import NpuAccelerator
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
from .api import NpuKernel
from .api import NpuLayout
from .api import NpuOperation
from .api import NpuOperationType
from .api import NpuPadding
from .api import NpuPoolingOp
from .api import NpuPoolingOperation
from .api import NpuResamplingMode
from .api import NpuRoundingMode
from .api import NpuShape3D
from .api import NpuTileBox
from .architecture_allocator import ArchitectureBlockConfig
from .architecture_allocator import try_block_config
from .architecture_features import Accelerator
from .architecture_features import ArchitectureFeatures
from .architecture_features import create_default_arch
from .architecture_features import SHRAMElements
from .errors import ByteAlignmentError
from .errors import ByteSizeError
from .errors import VelaError
from .ethos_u55_regs.ethos_u55_regs import acc_format
from .ethos_u55_regs.ethos_u55_regs import activation
from .ethos_u55_regs.ethos_u55_regs import cmd0
from .ethos_u55_regs.ethos_u55_regs import cmd1
from .ethos_u55_regs.ethos_u55_regs import elementwise_mode
from .ethos_u55_regs.ethos_u55_regs import pooling_mode
from .ethos_u55_regs.ethos_u55_regs import resampling_mode
from .ethos_u55_regs.ethos_u55_regs import rounding
from .numeric_util import round_away_zero
from .numeric_util import round_up_to_int
from .operation import ExplicitScaling
from .operation import NpuBlockType
from .range_set import MemoryAccessSet
from .register_command_stream_util import BASE_PTR_INDEX_MEM2MEM
from .register_command_stream_util import calc_blockdep
from .register_command_stream_util import check_addresses
from .register_command_stream_util import check_alignment
from .register_command_stream_util import check_dma_op
from .register_command_stream_util import check_length
from .register_command_stream_util import check_strides
from .register_command_stream_util import get_dma_memory_accesses
from .register_command_stream_util import get_op_memory_accesses
from .register_command_stream_util import get_strides
from .register_command_stream_util import get_wait_dependency
from .register_command_stream_util import get_zero_point
from .register_command_stream_util import has_ifm2
from .register_command_stream_util import quantise
from .register_command_stream_util import shape3d_to_block
from .register_command_stream_util import to_kernel
from .register_command_stream_util import UNARY_ELEMWISE_OPS
from .register_command_stream_util import Watermark


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
        s = f"  {'Offset':6}:"
        s += f" {'Payload':8}"
        s += f"{'Param':4}"  # no leading space for alignment
        s += f" {'Code':4}"
        s += f" - {'Command':30}"
        s += f" {'Param':5}"
        print(s)

        offset = 0
        for words_for_one_command in self.cmd_stream:
            code = words_for_one_command[0] & 0x0000FFFF  # lower 16 bits
            param = words_for_one_command[0] >> 16  # higher 16 bits

            payload_mode = CmdMode(code & CmdMode.Mask)

            s = f"{offset:#08x}:"

            if payload_mode == CmdMode.NoPayload:
                s += f" {'':8}"
            else:
                assert payload_mode == CmdMode.Payload32
                s += f" {words_for_one_command[1]:08x}"

            s += f" {param:04x}"
            s += f" {code:04x}"

            if payload_mode == CmdMode.NoPayload:
                s += f" - {cmd0(code & CmdMode.CmdOpMask):30}"
                offset += 4
            else:
                s += f" - {cmd1(code & CmdMode.CmdOpMask):30}"
                offset += 8

            s += f" {param:5}"
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
        offset = int(offset) & 0xFFFFFFFF
        param = int(param) & 0xFFFF
        command = cmd.value | CmdMode.Payload32.value | (param << 16)

        if not self.get_reg_machine(cmd).set_register(cmd, (command, offset)):
            return

        # This is not a redundant command, actually write it
        self.cmd_stream.append((command, offset))
        self.offset += CommandStreamEmitter.WORD_SIZE * 2

    def cmd1_with_address(self, cmd: cmd1, offset):
        self.cmd1_with_offset(cmd, offset, offset >> 32)

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


def check_mem_limits(memory_accesses: MemoryAccessSet, mem_limits: Dict[int, int]):
    """Checks that an operation's memory accesses respect the boundaries imposed by mem_limits"""
    for mem_access in memory_accesses.accesses:
        for region, range_set in mem_access.regions.items():
            if region not in mem_limits:
                raise VelaError(f"Invalid region: {region}")
            max = mem_limits[region]
            for start, end in range_set.ranges:
                for offset in (start, end):
                    if offset < 0:
                        raise VelaError(f"Negative address offset: {offset}, region: {region}")
                    if offset > max:
                        raise VelaError(
                            f"Address offset out of range: {offset}, region: {region}, max: {max}. Perhaps try running"
                            f" with the HillClimb tensor allocator and/or increasing the maximum iteration of that"
                            f" allocator"
                        )


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
        activation_value = cast(int, activation_op_map[act.op_type])
    emit.cmd0_with_param(cmd0.NPU_SET_ACTIVATION, activation_value)
    emit.cmd0_with_param(cmd0.NPU_SET_ACTIVATION_MIN, quantized_min)
    emit.cmd0_with_param(cmd0.NPU_SET_ACTIVATION_MAX, quantized_max)


def generate_addresses(
    emit: CommandStreamEmitter,
    ptr_cmds: List[cmd1],
    addresses: List[int],
    layout: NpuLayout,
    element_size,
    arch: ArchitectureFeatures,
):
    """Generates xFM_BASE registers"""
    check_addresses(addresses, layout, element_size, arch)
    for i in range(4):
        emit.cmd1_with_address(ptr_cmds[i], addresses[i])


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
    check_strides(fm, strides)

    emit.cmd1_with_address(stride_c_cmd, strides.depth)  # stride between 16-byte channel blocks (C)
    emit.cmd1_with_address(stride_y_cmd, strides.height)  # stride between vertical values (H)
    emit.cmd1_with_address(stride_x_cmd, strides.width)  # stride between horisontal values (W)


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


def generate_ifm(emit: CommandStreamEmitter, ifm: NpuFeatureMap, arch: ArchitectureFeatures):
    """Generates general IFM registers"""
    emit.cmd0_with_param(cmd0.NPU_SET_IFM_REGION, ifm.region)
    generate_addresses(
        emit,
        [cmd1.NPU_SET_IFM_BASE0, cmd1.NPU_SET_IFM_BASE1, cmd1.NPU_SET_IFM_BASE2, cmd1.NPU_SET_IFM_BASE3],
        ifm.tiles.addresses,
        ifm.layout,
        ifm.data_type.size_in_bytes(),
        arch,
    )
    generate_tiles(
        emit, [cmd0.NPU_SET_IFM_HEIGHT0_M1, cmd0.NPU_SET_IFM_HEIGHT1_M1, cmd0.NPU_SET_IFM_WIDTH0_M1], ifm.tiles
    )
    emit.cmd0_with_param(cmd0.NPU_SET_IFM_DEPTH_M1, ifm.shape.depth - 1)
    generate_strides(emit, ifm, cmd1.NPU_SET_IFM_STRIDE_C, cmd1.NPU_SET_IFM_STRIDE_Y, cmd1.NPU_SET_IFM_STRIDE_X)
    emit.cmd0_with_param(cmd0.NPU_SET_IFM_ZERO_POINT, get_zero_point(ifm))


def generate_ifm2(emit: CommandStreamEmitter, ifm2: NpuFeatureMap, has_scalar: bool, arch: ArchitectureFeatures):
    """Generates general IFM2 registers"""
    if not has_scalar:
        emit.cmd0_with_param(cmd0.NPU_SET_IFM2_REGION, ifm2.region)
        generate_addresses(
            emit,
            [cmd1.NPU_SET_IFM2_BASE0, cmd1.NPU_SET_IFM2_BASE1, cmd1.NPU_SET_IFM2_BASE2, cmd1.NPU_SET_IFM2_BASE3],
            ifm2.tiles.addresses,
            ifm2.layout,
            ifm2.data_type.size_in_bytes(),
            arch,
        )
        generate_tiles(
            emit, [cmd0.NPU_SET_IFM2_HEIGHT0_M1, cmd0.NPU_SET_IFM2_HEIGHT1_M1, cmd0.NPU_SET_IFM2_WIDTH0_M1], ifm2.tiles
        )
        generate_strides(emit, ifm2, cmd1.NPU_SET_IFM2_STRIDE_C, cmd1.NPU_SET_IFM2_STRIDE_Y, cmd1.NPU_SET_IFM2_STRIDE_X)
    emit.cmd0_with_param(cmd0.NPU_SET_IFM2_ZERO_POINT, get_zero_point(ifm2))


def generate_ofm(emit: CommandStreamEmitter, ofm: NpuFeatureMap, arch: ArchitectureFeatures):
    """Generates general OFM registers"""
    emit.cmd0_with_param(cmd0.NPU_SET_OFM_REGION, ofm.region)
    generate_addresses(
        emit,
        [cmd1.NPU_SET_OFM_BASE0, cmd1.NPU_SET_OFM_BASE1, cmd1.NPU_SET_OFM_BASE2, cmd1.NPU_SET_OFM_BASE3],
        ofm.tiles.addresses,
        ofm.layout,
        ofm.data_type.size_in_bytes(),
        arch,
    )
    generate_tiles(
        emit, [cmd0.NPU_SET_OFM_HEIGHT0_M1, cmd0.NPU_SET_OFM_HEIGHT1_M1, cmd0.NPU_SET_OFM_WIDTH0_M1], ofm.tiles
    )
    emit.cmd0_with_param(cmd0.NPU_SET_OFM_HEIGHT_M1, ofm.shape.height - 1)
    emit.cmd0_with_param(cmd0.NPU_SET_OFM_WIDTH_M1, ofm.shape.width - 1)
    emit.cmd0_with_param(cmd0.NPU_SET_OFM_DEPTH_M1, ofm.shape.depth - 1)
    generate_strides(emit, ofm, cmd1.NPU_SET_OFM_STRIDE_C, cmd1.NPU_SET_OFM_STRIDE_Y, cmd1.NPU_SET_OFM_STRIDE_X)
    emit.cmd0_with_param(cmd0.NPU_SET_OFM_ZERO_POINT, get_zero_point(ofm))


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
            check_alignment(weights[core].address, 16)
            check_length(weights[core].length, 16)
            emit.cmd1_with_address(addr, weights[core].address)
            emit.cmd1_with_offset(length, weights[core].length)
        elif core < arch.ncores:
            check_alignment(weights[0].address, 16)
            emit.cmd1_with_address(addr, weights[0].address)
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
            emit.cmd1_with_address(addr, biases[core].address)
            check_length(biases[core].length, 16)
            emit.cmd1_with_offset(length, biases[core].length)
        elif core < arch.ncores:
            emit.cmd1_with_address(addr, biases[0].address)
            emit.cmd1_with_offset(length, 0)


def generate_block_config(
    emit: CommandStreamEmitter,
    block_config: NpuShape3D,
):
    """Generates OFM_BLK_HEIGHT/WIDTH/DEPTH registers"""
    emit.cmd0_with_param(cmd0.NPU_SET_OFM_BLK_HEIGHT_M1, block_config.height - 1)
    emit.cmd0_with_param(cmd0.NPU_SET_OFM_BLK_WIDTH_M1, block_config.width - 1)
    emit.cmd0_with_param(cmd0.NPU_SET_OFM_BLK_DEPTH_M1, block_config.depth - 1)


def generate_shram_registers(
    emit: CommandStreamEmitter,
    npu_op: NpuBlockOperation,
    arch_block_config: ArchitectureBlockConfig,
):
    """Generates IB_END/IB_START/AB_START/ACC_FORMAT registers"""
    emit.cmd0_with_param(cmd0.NPU_SET_IFM_IB_END, arch_block_config.layout.ib_end)
    emit.cmd0_with_param(cmd0.NPU_SET_AB_START, arch_block_config.layout.ab_start)
    if has_ifm2(npu_op):
        emit.cmd0_with_param(cmd0.NPU_SET_IFM2_IB_START, arch_block_config.layout.ib_start2)
    emit.cmd0_with_param(cmd0.NPU_SET_ACC_FORMAT, acc_format_map[arch_block_config.acc_type])


def get_block_config_for_npu_op(
    arch, npu_op: NpuBlockOperation, npu_block_type: NpuBlockType, is_partkernel: bool, ifm_resampling: resampling_mode
) -> Optional[ArchitectureBlockConfig]:
    """
    Given npu_op.block_config, returns a corresponding ArchitectureBlockConfig.
    Returns None if the block_config does not fit.
    """


def get_arch_block_config(
    npu_op: NpuBlockOperation, block_traversal: NpuBlockTraversal, arch: ArchitectureFeatures
) -> ArchitectureBlockConfig:
    """Creates shared buffer allocation for the given operation"""
    assert npu_op.block_config is not None, "block_config has not been set"
    block_type = NpuBlockType.Default
    if isinstance(npu_op, NpuConv2DOperation):
        block_type = NpuBlockType.ConvolutionMxN
    elif isinstance(npu_op, NpuConvDepthWiseOperation):
        block_type = NpuBlockType.ConvolutionDepthWise
    elif isinstance(npu_op, NpuPoolingOperation):
        block_type = NpuBlockType.ReduceSum if npu_op.sub_op_type == NpuPoolingOp.REDUCE_SUM else NpuBlockType.Pooling
    elif isinstance(npu_op, NpuElementWiseOperation):
        block_type = NpuBlockType.ElementWise
    else:
        assert 0, "Unsupported operation"
    ifm_resampling_mode = resampling_mode_map[npu_op.ifm_upscale]
    is_partkernel = block_traversal == NpuBlockTraversal.PART_KERNEL_FIRST
    uses_lut = npu_op.activation is not None and npu_op.activation.op_type == NpuActivationOp.TABLE_LOOKUP
    lut_banks = 2 if uses_lut else 0
    fms = [npu_op.ifm, npu_op.ofm]
    if npu_op.ifm2 is not None:
        fms.append(npu_op.ifm2)
    all_fms_have_quant = not any(fm.quantization is None or fm.quantization.scale_f32 is None for fm in fms)
    ifm_bits = npu_op.ifm.data_type.size_in_bits()
    ifm_shape = shape3d_to_block(npu_op.ifm.shape)
    if has_ifm2(npu_op):
        ifm2_shape = shape3d_to_block(npu_op.ifm2.shape)
    else:
        ifm2_shape = None
    uses_scalar = npu_op.ifm2_scalar is not None
    block_config = shape3d_to_block(npu_op.block_config)
    arch_block_config = try_block_config(
        block_config,
        arch,
        block_type,
        shape3d_to_block(npu_op.ofm.shape),
        ifm_shape,
        ifm2_shape,
        uses_scalar,
        ifm_bits,
        is_partkernel=is_partkernel,
        kernel=to_kernel(npu_op.kernel),
        lut_banks=lut_banks,
        scaled=all_fms_have_quant,
        ifm_resampling=ifm_resampling_mode,
    )
    assert arch_block_config is not None, f"block_config {npu_op.block_config} does not fit"
    return arch_block_config


def generate_cmd_waits(emit: CommandStreamEmitter, cmd_waits: Watermark):
    """Generates KERNEL_WAIT/DMA_WAIT"""
    if cmd_waits.npu >= 0:
        emit.cmd_wait(cmd0.NPU_OP_KERNEL_WAIT, 0, cmd_waits.npu)

    if cmd_waits.dma >= 0:
        emit.cmd_wait(cmd0.NPU_OP_DMA_WAIT, 0, cmd_waits.dma)


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
    generate_ifm(emit, npu_op.ifm, arch)
    generate_ifm_precision(emit, npu_op.ifm, op_to_scale, cmd0.NPU_SET_IFM_PRECISION)
    emit.cmd0_with_param(cmd0.NPU_SET_IFM_UPSCALE, resampling_mode_map[npu_op.ifm_upscale])
    if npu_op.padding is not None:
        generate_padding(emit, npu_op.padding)
    generate_ofm(emit, npu_op.ofm, arch)
    generate_ofm_precision(emit, npu_op, use_global_scale)
    if npu_op.op_type != NpuOperationType.ElementWise:
        assert npu_op.kernel is not None
        generate_kernel(emit, npu_op.kernel, block_traversal)
    generate_weights(emit, npu_op.weights, arch)
    generate_biases(emit, npu_op.biases, arch)
    generate_activation(emit, npu_op.activation, npu_op.ofm)
    arch_block_config = get_arch_block_config(npu_op, block_traversal, arch)
    generate_block_config(emit, npu_op.block_config)
    generate_shram_registers(emit, npu_op, arch_block_config)


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
            x_log2 = math.log2(ifm_quant.scale_f32)
            rounded_log2 = int(round(x_log2))
            is_power_of_two = abs(x_log2 - rounded_log2) < 0.001
            shift = rounded_log2 + 12
            if is_power_of_two and (
                (pool_op.activation.op_type == NpuActivationOp.TANH and shift in (0, 1))
                or (pool_op.activation.op_type == NpuActivationOp.SIGMOID and shift == 0)
            ):
                # Special handling if input scale is 1/2048 (tanh/sigmoid) or 1/4096 (tanh)
                scale = 3 << shift
                shift = 0
            else:
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
        if type(pool_op.rescale) == ExplicitScaling:
            # Note: reuse of rescale for explicit scaling to not expose this in the external API
            explicit_scaling = pool_op.rescale
            assert explicit_scaling.per_channel is False
            scale = explicit_scaling.multiplier[0]
            shift = explicit_scaling.shift[0]
        else:
            # for ResizeBilinear/NearestNeighbor operations with rescale
            # Note: this is not used, but part of the public API
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
            if npu_op.rescale:
                ofm_scale, shift = npu_op.rescale
            elif None in (input_scale, input2_scale, output_scale):
                ofm_scale = 1
                shift = 0
            else:
                ofm_scale, shift = scaling.elementwise_mul_scale(input_scale, input2_scale, output_scale)
        else:  # Add/Sub
            # Default operand scaling is no scaling
            opa_scale = opb_scale = 1
            opa_shift = 0
            bitdepth = npu_op.ifm.data_type.size_in_bits()
            use_advanced_scaling = False
            if npu_op.rescale is not None:
                # Explicit ofm scaling
                ofm_scale, shift = npu_op.rescale
            elif None in (input_scale, input2_scale, output_scale):
                # No ofm scaling
                ofm_scale = 1
                shift = 0
            elif input_scale == input2_scale and bitdepth == 16:
                # int16 same scaling
                opa_scale, opb_scale, ofm_scale, shift = scaling.simplified_elementwise_add_sub_scale(
                    input_scale, input2_scale, output_scale
                )
                # align the double rounding with that of advanced scaling
                opa_scale //= 2
                opb_scale //= 2
                shift -= 1
                opa_shift = 0  # Unused for this case
            elif input_scale == input2_scale:
                # Same scaling
                opa_scale, opb_scale, ofm_scale, shift = scaling.simplified_elementwise_add_sub_scale(
                    input_scale, input2_scale, output_scale
                )
                opa_shift = 0  # Unused for this case
                # For 8 bit we can't guarantee double rounding with simplified scaling will always be
                # the same as with advanced scaling due to different shifts. When the ofm scale fulfils
                # the following we know that double rounding will have no effect for advanced scaling
                # no matter the input, so we can safely use simplified scaling with double rounding disabled.
                use_advanced_scaling = int(ofm_scale) & 0xFFF != 0
            else:
                use_advanced_scaling = True
            if use_advanced_scaling:
                # Use advanced implementation only when input/output scales differ,
                # or when we can't guarantee the absence of rounding errors
                (
                    opa_scale,
                    opa_shift,
                    ofm_scale,
                    shift,
                    op_to_scale,
                ) = scaling.advanced_elementwise_add_sub_scale(input_scale, input2_scale, output_scale, bitdepth)
                opb_scale = 0  # Unused for this case
                if npu_op.reversed_operands:
                    # If the operand order is reversed we also have to swap which operand is scaled
                    if op_to_scale == scaling.OperandToScale.OPa:
                        op_to_scale = scaling.OperandToScale.OPb
                    else:
                        op_to_scale = scaling.OperandToScale.OPa
            emit.cmd1_with_offset(cmd1.NPU_SET_OPA_SCALE, opa_scale, opa_shift)
            emit.cmd1_with_offset(cmd1.NPU_SET_OPB_SCALE, opb_scale)
    elif npu_op.sub_op_type in (NpuElementWiseOp.LRELU, NpuElementWiseOp.ABS):
        output_scale = npu_op.ofm.quantization.scale_f32
        ofm_scale, shift = scaling.quantise_scale(output_scale)
    else:
        ofm_scale = 1
        shift = 0
    emit.cmd1_with_offset(cmd1.NPU_SET_OFM_SCALE, ofm_scale, shift)
    return op_to_scale


# -------------------------------------------------------------------
# PRINT
# -------------------------------------------------------------------


def print_feature_map(fm: Optional[NpuFeatureMap], name: str):
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
        print(f"         name={fm.name}")


def print_operation(npu_op: NpuOperation, index: int = 0, cmd=None):
    pass_info = f" {cmd}" if cmd else ""
    if isinstance(npu_op, NpuOperation) and not isinstance(npu_op, (NpuDmaOperation, NpuBlockOperation)):
        print(f"{index} {npu_op.op_type.name}  name={npu_op.name}:{pass_info}")
        return
    if isinstance(npu_op, NpuDmaOperation):
        print(f"{index} {npu_op.op_type.name} name={npu_op.name}, src={npu_op.src}, dest={npu_op.dest}:{pass_info}")
        return
    k = None if npu_op.kernel is None else to_kernel(npu_op.kernel)
    if isinstance(npu_op, (NpuPoolingOperation, NpuElementWiseOperation)):
        print(f"{index} {npu_op.sub_op_type.name} {npu_op.op_type.name} name={npu_op.name}:{pass_info}")
    else:
        if (
            isinstance(npu_op, NpuConv2DOperation)
            and k.elements_wh() * k.stride.x * k.stride.y * k.dilation.x * k.dilation.y == 1
        ):
            fc = "FullyConnected "
        else:
            fc = ""
        print(f"{index} {fc}{npu_op.op_type.name} name={npu_op.name}:{pass_info}")
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
    if isinstance(npu_op, NpuConv2DOperation):
        print(f"      {npu_op.block_traversal}")
    bh, bw, bc = npu_op.block_config
    rescale = (
        f", rescale={npu_op.rescale}" if isinstance(npu_op, (NpuPoolingOperation, NpuElementWiseOperation)) else ""
    )
    print(f"      Block config: h={bh},w={bw},c={bc}, {npu_op.ifm_upscale}, {npu_op.rounding_mode}{rescale}")


def print_operations(npu_op_list: List[NpuOperation], npu_op_to_cmd=None):
    npu_op_to_cmd = dict() if npu_op_to_cmd is None else npu_op_to_cmd
    for index, npu_op in enumerate(npu_op_list):
        print_operation(npu_op, index, npu_op_to_cmd.get(npu_op))


# -------------------------------------------------------------------
# OPERATIONS
# -------------------------------------------------------------------


def generate_operation_code(emit: CommandStreamEmitter, npu_op: NpuOperation):
    """Generates NPU_OP_* command"""
    if isinstance(npu_op, NpuDmaOperation):
        emit.cmd_do_operation(cmd0.NPU_OP_DMA_START, npu_op.channel * 16 + npu_op.mode)
    elif isinstance(npu_op, NpuConv2DOperation):
        emit.cmd_do_operation(cmd0.NPU_OP_CONV)
    elif isinstance(npu_op, NpuConvDepthWiseOperation):
        emit.cmd_do_operation(cmd0.NPU_OP_DEPTHWISE)
    elif isinstance(npu_op, NpuPoolingOperation):
        emit.cmd_do_operation(cmd0.NPU_OP_POOL, param=pooling_op_map[npu_op.sub_op_type])
    elif isinstance(npu_op, NpuElementWiseOperation):
        emit.cmd_do_operation(cmd0.NPU_OP_ELEMENTWISE, param=elementwise_op_map[npu_op.sub_op_type])
    else:
        assert 0, "Unsupported operation"


def generate_conv2d_op(emit: CommandStreamEmitter, npu_op: NpuConv2DOperation, arch: ArchitectureFeatures):
    """Generates register commands for Conv2D operations"""
    generate_common(emit, npu_op, npu_op.block_traversal, arch)


def generate_conv_depthwise_op(
    emit: CommandStreamEmitter, npu_op: NpuConvDepthWiseOperation, arch: ArchitectureFeatures
):
    """Generates register commands for depthwise convolution operations"""
    generate_common(emit, npu_op, NpuBlockTraversal.DEPTH_FIRST, arch)


def generate_pooling_op(emit: CommandStreamEmitter, npu_op: NpuPoolingOperation, arch: ArchitectureFeatures):
    """Generates register commands for pooling operations"""
    # check that reduce_sum input is NHWC
    if npu_op.sub_op_type == NpuPoolingOp.REDUCE_SUM and npu_op.ifm.layout != NpuLayout.NHWC:
        if npu_op.ifm.data_type == NpuDataType.INT32:
            raise VelaError(
                f"REDUCE_SUM ({npu_op.name}) with IFM data type of INT32 requires IFM layout to be NHWC"
                f" ({npu_op.ifm.name} == {npu_op.ifm.layout})"
            )
        elif arch.accelerator_config == Accelerator.Ethos_U65_512:
            raise VelaError(
                f"REDUCE_SUM ({npu_op.name}) with accelerator config of Ethos_U65_512 requires IFM layout to be NHWC"
                f" ({npu_op.ifm.name} == {npu_op.ifm.layout})"
            )

    use_global_scale = (
        npu_op.sub_op_type in (NpuPoolingOp.AVERAGE, NpuPoolingOp.REDUCE_SUM) and sum(npu_op.padding) == 0
    )
    # Note: reuse of rescale for explicit scaling to not expose this in the external API
    if npu_op.rescale is not None and type(npu_op.rescale) == ExplicitScaling:
        use_global_scale = not npu_op.rescale.per_channel
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
    if npu_op.sub_op_type not in UNARY_ELEMWISE_OPS:
        # Binary operation; generate IFM2 registers
        assert npu_op.ifm2 is not None
        has_scalar = npu_op.ifm2_scalar is not None
        generate_ifm2(emit, npu_op.ifm2, has_scalar, arch)
        generate_ifm_precision(emit, npu_op.ifm2, 0, cmd0.NPU_SET_IFM2_PRECISION)
        generate_ifm2_broadcast(emit, npu_op)
        if has_scalar:
            quantized_scalar = quantise(npu_op.ifm2_scalar, npu_op.ifm2.quantization)
            assert npu_op.ifm2.data_type.min_value() <= quantized_scalar <= npu_op.ifm2.data_type.max_value()
            emit.cmd0_with_param(cmd0.NPU_SET_IFM2_SCALAR, quantized_scalar)


def generate_dma_op(emit: CommandStreamEmitter, dma_op: NpuDmaOperation, arch: ArchitectureFeatures):
    """Generates register commands for DMA operations"""
    check_dma_op(dma_op, arch)

    emit.cmd0_with_param(cmd0.NPU_SET_DMA0_SRC_REGION, dma_op.src.region)
    emit.cmd1_with_address(cmd1.NPU_SET_DMA0_SRC, dma_op.src.address)
    emit.cmd0_with_param(cmd0.NPU_SET_DMA0_DST_REGION, dma_op.dest.region)

    emit.cmd1_with_address(cmd1.NPU_SET_DMA0_DST, dma_op.dest.address)
    emit.cmd1_with_address(cmd1.NPU_SET_DMA0_LEN, dma_op.src.length)


def generate_registers_for_op(emit: CommandStreamEmitter, npu_op: NpuOperation, arch: ArchitectureFeatures):
    """
    Generates register commands for the given operation, but not the final NPU_OP_... command.
    Returns the selected block config
    """
    if isinstance(npu_op, NpuConv2DOperation):
        generate_conv2d_op(emit, npu_op, arch)
    elif isinstance(npu_op, NpuConvDepthWiseOperation):
        generate_conv_depthwise_op(emit, npu_op, arch)
    elif isinstance(npu_op, NpuPoolingOperation):
        generate_pooling_op(emit, npu_op, arch)
    elif isinstance(npu_op, NpuElementWiseOperation):
        generate_elementwise_op(emit, npu_op, arch)
    elif isinstance(npu_op, NpuDmaOperation):
        generate_dma_op(emit, npu_op, arch)
    else:
        assert 0, "Unsupported operation"


def generate_command_stream(
    npu_op_list: List[NpuOperation],
    arch: ArchitectureFeatures,
    verbose: bool,
    mem_limits: Dict[int, int],
    add_to_debug_db=None,
    npu_op_to_cmd=None,
) -> List[int]:
    """
    Generates register commands for the given list of NPU operations.
    Returns Ethos-U instructions, as a list of 32-bit integers.
    """
    emit = CommandStreamEmitter()
    if verbose:
        print("Register-Level Command Stream: Input")
        print_operations(npu_op_list, npu_op_to_cmd)
    # Calculate memory accesses for every operation
    memory_accesses: Dict[NpuOperation, MemoryAccessSet] = {}
    for npu_op in npu_op_list:
        if isinstance(npu_op, NpuDmaOperation):
            memory_accesses[npu_op] = get_dma_memory_accesses(npu_op)
        elif isinstance(npu_op, NpuBlockOperation):
            memory_accesses[npu_op] = get_op_memory_accesses(npu_op, arch)
        else:
            assert 0, "Invalid operation type"

    if arch.is_ethos_u65_system:
        emit.cmd0_with_param(cmd0.NPU_SET_PARALLEL_MODE, arch.ncores - 1)
    prev_op = None
    # Generate register commands for all operations
    outstanding_dma_ops: List[NpuOperation] = list()
    outstanding_npu_ops: List[NpuOperation] = list()
    for op_index, npu_op in enumerate(npu_op_list):
        try:
            check_mem_limits(memory_accesses[npu_op], mem_limits)
            cmd_waits = get_wait_dependency(arch, npu_op, memory_accesses, outstanding_dma_ops, outstanding_npu_ops)
            generate_registers_for_op(emit, npu_op, arch)
        except ByteAlignmentError as e:
            # Enables testing for ByteAlignmentErrors specifically
            raise ByteAlignmentError(f"{e.error_msg}, in operation {op_index}:{npu_op.op_type.name}") from None
        except ByteSizeError as e:
            # Enables testing for ByteSizeErrors specifically
            raise ByteSizeError(f"{e.error_msg}, in operation {op_index}:{npu_op.op_type.name}") from None
        except VelaError as e:
            raise VelaError(f"{e.error_msg}, in operation {op_index}:{npu_op.op_type.name}") from None
        if not isinstance(npu_op, NpuDmaOperation) and isinstance(npu_op, NpuBlockOperation):
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
    res = emit.to_list()

    if emit.size_in_bytes() >= 1 << 24:
        raise VelaError(
            f"The command stream size exceeds the hardware limit of 16 MiB. "
            f"The current stream size is {emit.size_in_bytes()/2**20:.2F} MiB."
        )

    if verbose:
        print("Register-Level Command Stream: Output")
        emit.print_cmds()
        print(f"Number of commands = {len(emit.cmd_stream)}")
        print(f"Command stream length = {emit.size_in_bytes()} bytes")
    return res


def generate_register_command_stream(npu_op_list: List[NpuOperation], npu_accelerator: NpuAccelerator) -> List[int]:
    """
    Internal implementation of the public facing API for generating an Ethos-U register command stream.
    Calculates dependencies between commands and inserts wait operations if needed.

    :param npu_op_list: List[NpuOperation] list of high level NPU operations
    :param accelerator: architecture_features.Accelerator enum to pick the correct Ethos-U accelerator
    :return Ethos-U instructions, as a list of 32-bit integers
    """
    accelerator = Accelerator.from_npu_accelerator(npu_accelerator)
    arch = create_default_arch(accelerator)
    mem_limits = dict()
    for region in range(0, 8):
        mem_limits[region] = arch.max_address_offset
    mem_limits[BASE_PTR_INDEX_MEM2MEM] = arch.shram_size_bytes
    return generate_command_stream(npu_op_list, arch, verbose=False, mem_limits=mem_limits)
