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
# Register level (low-level) command stream generation for Ethos-U55. Takes a high-level command stream and generates
# all the register settings. Calculates dependencies between commands and inserts wait operations. And generates a bit
# stream suitable for interpretation by the Ethos-U55 processor.
from collections import defaultdict
from collections import namedtuple
from enum import Enum
from enum import IntEnum

import numpy as np

from . import scaling
from .architecture_features import ArchitectureFeatures
from .architecture_features import Block
from .architecture_features import Kernel
from .architecture_features import Rect
from .architecture_features import SharedBufferArea
from .architecture_features import SHRAMElements
from .data_type import BaseType
from .data_type import DataType
from .ethos_u55_regs.ethos_u55_regs import acc_format
from .ethos_u55_regs.ethos_u55_regs import activation
from .ethos_u55_regs.ethos_u55_regs import cmd0
from .ethos_u55_regs.ethos_u55_regs import cmd1
from .ethos_u55_regs.ethos_u55_regs import elementwise_mode
from .ethos_u55_regs.ethos_u55_regs import ifm_precision
from .ethos_u55_regs.ethos_u55_regs import pooling_mode
from .ethos_u55_regs.ethos_u55_regs import resampling_mode
from .ethos_u55_regs.ethos_u55_regs import rounding
from .high_level_command_stream import CommandType
from .numeric_util import clamp_sigmoid
from .numeric_util import clamp_tanh
from .numeric_util import full_shape
from .numeric_util import quantise_float32
from .numeric_util import round_away_zero
from .numeric_util import round_up_to_int
from .operation import NpuBlockType
from .tensor import MemType
from .tensor import TensorBlockTraversal
from .tensor import TensorFormat
from .tensor import TensorPurpose


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


class CommandStreamEmitter:
    def __init__(self):
        self.cmd_stream = []
        self.reg_machine = [RegisterMachine(), RegisterMachine()]
        self.last_absolute_wait = defaultdict(int)

    def get_reg_machine(self, cmd):
        if "DMA" in cmd.name:
            return self.reg_machine[1]
        else:
            return self.reg_machine[0]

    def size_in_bytes(self):
        sz = 0
        for cmd in self.cmd_stream:
            sz += len(cmd) * 4
        return sz

    def to_list(self):
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

    def cmd0_with_param(self, cmd, param):
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

    def cmd1_with_offset(self, cmd, offset, param=0x0):
        offset = int(offset) & 0xFFFFFFFFF
        command = cmd.value | CmdMode.Payload32.value | (param << 16)

        if not self.get_reg_machine(cmd).set_register(cmd, (command, offset)):
            return

        # This is not a redundant command, actually write it
        self.cmd_stream.append((command, offset))

    def cmd_wait(self, cmd, channel, outstanding_count):
        param = (16 * channel) + outstanding_count
        command = ((param & 0xFFFF) << 16) | cmd.value
        self.cmd_stream.append((command,))

    def cmd_do_operation(self, cmd, param=0):
        param = int(param)
        command = ((param & 0xFFFF) << 16) | cmd.value

        self.cmd_stream.append((command,))
        self.get_reg_machine(cmd).switch_bank()


Watermark = namedtuple("Watermark", ["npu", "dma"])


def get_cmd_wait_dependency(arch, cmd_stream, memory_accesses, cmd_index, watermark: Watermark):
    cmd = cmd_stream[cmd_index]
    cmd_access = memory_accesses[cmd]
    index = cmd_index - 1

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
        prev_cmd = cmd_stream[index]
        prev_access = memory_accesses[prev_cmd]

        # Check DMA consuming NPU output
        if prev_cmd.cmdtype == CommandType.NpuStripe:
            if index >= npu_index:
                if (cmd.cmdtype == CommandType.DMA) and (npu_outstanding == -1) and prev_access.conflicts(cmd_access):
                    npu_outstanding = npu_ops
                npu_ops = npu_ops + 1  # Count NPU ops in the pipeline
                if npu_ops >= arch.max_outstanding_kernels:
                    npu_index = max(index + 1, npu_index)

        # Check NPU consuming DMA output
        elif prev_cmd.cmdtype == CommandType.DMA:
            if index >= dma_index:
                if cmd.cmdtype == CommandType.NpuStripe:
                    if (dma_outstanding == -1) and prev_access.conflicts(cmd_access):
                        dma_outstanding = dma_ops
                dma_ops = dma_ops + 1  # Count DMA ops in the pipeline
                if dma_ops >= arch.max_outstanding_dma:
                    dma_index = max(index + 1, dma_index)

        index = index - 1

    # Update DMA watermark if we didn't see any and the NPU pipeline is full
    if (dma_ops == 0) and (npu_ops >= arch.max_outstanding_kernels):
        dma_index = cmd_index

    # Bring the search watermark forwards as we complete for those dependencies
    watermark = Watermark(npu_index, dma_index)
    outstanding = Watermark(npu_outstanding, dma_outstanding)

    return watermark, outstanding


def get_op_kernel(ps):
    if ps.primary_op is None:
        return None

    strides = ps.primary_op.attrs.get("strides", (1, 1, 1, 1))
    dilation = ps.primary_op.attrs.get("dilation", (1, 1, 1, 1))
    if ps.weight_tensor:
        if ps.npu_block_type in set((NpuBlockType.VectorProduct, NpuBlockType.ElementWise)):
            k_h = 1
            k_w = 1
        else:
            k_h = ps.weight_tensor.shape[0]
            k_w = ps.weight_tensor.shape[1]
    else:
        k_h = ps.primary_op.attrs.get("filter_height", 1)
        k_w = ps.primary_op.attrs.get("filter_width", 1)

    return Kernel(k_w, k_h, strides[2], strides[1], dilation[2], dilation[1])


def has_prev_op_dependency(prev_cmd, cmd):
    if prev_cmd is None:
        return False
    if (prev_cmd.cmdtype == cmd.cmdtype == CommandType.NpuStripe) and (prev_cmd.ps != cmd.ps):
        if prev_cmd.ofm_tensor.equivalent(cmd.ifm_tensor):
            return True
        elif cmd.ifm2_tensor is not None:
            return prev_cmd.ofm_tensor.equivalent(cmd.ifm2_tensor)
    return False


def get_op_ofm_rect(cmd):
    start = full_shape(4, cmd.ofm_box.start_coord, 0)
    end = full_shape(4, cmd.ofm_box.end_coord, 1)
    return Rect(start[-2], start[-3], start[-1], end[-2] - 1, end[-3] - 1, end[-1] - 1)


def get_op_ifm_rect(cmd):
    start = full_shape(4, cmd.ifm_box.start_coord, 0)
    end = full_shape(4, cmd.ifm_box.end_coord, 1)
    return Rect(start[-2], start[-3], start[-1], end[-2] - 1, end[-3] - 1, end[-1] - 1)


def get_op_ifmofm_block_depth(arch, cmd):
    # Note: NOT equivalent to the normal ifm block depth calculation since
    # it takes into account 'depthless' block operations by returning full
    # depth
    if cmd.ps.npu_block_type in (
        NpuBlockType.ConvolutionDepthWise,
        NpuBlockType.Pooling,
        NpuBlockType.ElementWise,
        NpuBlockType.ReduceSum,
    ):
        return cmd.ofm_box.get_size_shape()[-1]

    return arch.calc_ifm_block_depth(cmd.ifm_box.get_size_shape()[-1], cmd.ifm_tensor.dtype.bits)


def get_op_padding_lt(cmd):
    if cmd.ps.npu_block_type not in (
        NpuBlockType.ConvolutionDepthWise,
        NpuBlockType.Pooling,
        NpuBlockType.ConvolutionMxN,
        NpuBlockType.ReduceSum,
    ):
        return (0, 0)

    explicit_padding = list(cmd.ps.primary_op.attrs["explicit_padding"])  # (top, left, bottom, right)

    # Check if this is for horizontal ifm streaming
    if not (cmd.is_first_h_stripe and cmd.is_last_h_stripe):
        explicit_padding[0] = cmd.pad_top
        explicit_padding[2] = cmd.pad_bottom

    return (explicit_padding[1], explicit_padding[0])


def ifm_ifm2_correct_order(ifm_shape, ifm2_shape):
    if ifm_shape == []:
        # Scalar needs to be in IFM2
        return False
    elif ifm2_shape == []:
        return True

    for ifm, ifm2 in zip(ifm_shape, ifm2_shape):
        if ifm != ifm2 and ifm == 1:
            # Broadcasted FM needs to be in IFM2
            return False

    return True


def generate_register_command_stream(nng, sg, arch, verbose=False):
    emit = CommandStreamEmitter()

    if arch.feature_map_storage_mem_area == arch.fast_storage_mem_area:
        base_ptr_idx_map = {
            MemType.Permanent_NPU: BasePointerIndex.WeightTensor,
            MemType.Permanent_CPU: BasePointerIndex.WeightTensor,
            MemType.Scratch: BasePointerIndex.ScratchTensor,
            MemType.Scratch_fast: BasePointerIndex.ScratchTensor,
        }
    else:
        base_ptr_idx_map = {
            MemType.Permanent_NPU: BasePointerIndex.WeightTensor,
            MemType.Permanent_CPU: BasePointerIndex.WeightTensor,
            MemType.Scratch: BasePointerIndex.ScratchTensor,
            MemType.Scratch_fast: BasePointerIndex.ScratchFastTensor,
        }

    # Maps an AccumulatorType enum to the corresponding acc_format value
    acc_format_map = {
        SHRAMElements.Acc16: acc_format.FP_S5_10.value,
        SHRAMElements.Acc32: acc_format.INT_32BIT.value,
        SHRAMElements.Acc40: acc_format.INT_40BIT.value,
    }

    # Maps an elementwise op type to an elementwise_mode enum value used by NPU_OP_ELEMENTWISE
    elementwise_mode_map = {
        "MulAct": elementwise_mode.MUL.value,
        "AddAct": elementwise_mode.ADD.value,
        "SubAct": elementwise_mode.SUB.value,
        "Minimum": elementwise_mode.MIN.value,
        "Maximum": elementwise_mode.MAX.value,
        "LeakyRelu": elementwise_mode.LRELU.value,
        "Abs": elementwise_mode.ABS.value,
        "CLZ": elementwise_mode.CLZ.value,
        "SHR": elementwise_mode.SHR.value,
        "SHL": elementwise_mode.SHL.value,
    }

    cmd_stream = []
    memory_accesses = {}
    for cmd in sg.high_level_command_stream:
        if cmd.cmdtype == CommandType.NpuStripe and cmd.ps.npu_block_type == NpuBlockType.Default:
            print("Warning: Skipping register command stream generation for", cmd.ps)
        else:
            cmd_stream.append(cmd)
            memory_accesses[cmd] = cmd.get_memory_accesses()

    def emit_cmd_waits(cmd_waits):
        if cmd_waits.npu >= 0:
            emit.cmd_wait(cmd0.NPU_OP_KERNEL_WAIT, 0, cmd_waits.npu)

        if cmd_waits.dma >= 0:
            emit.cmd_wait(cmd0.NPU_OP_DMA_WAIT, 0, cmd_waits.dma)

    # Initialise operator dependency state
    prev_ifm_rect = cur_ifm_rect = None
    prev_ifm_block_depth = cur_ifm_block_depth = None
    prev_ofm_rect = cur_ofm_rect = None
    prev_ofm_block = cur_ofm_block = None
    prev_kernel = cur_kernel = None
    prev_cmd = None

    if arch.is_yoda_system:
        emit.cmd0_with_param(cmd0.NPU_SET_PARALLEL_MODE, arch.ncores - 1)

    dep_watermark = Watermark(0, 0)

    for cmd_index, cmd in enumerate(cmd_stream):
        dep_watermark, cmd_waits = get_cmd_wait_dependency(arch, cmd_stream, memory_accesses, cmd_index, dep_watermark)

        if cmd.cmdtype == CommandType.DMA:
            start_coord = cmd.box.start_coord

            src_addr = cmd.in_tensor.address_for_coordinate(start_coord)
            dst_addr = cmd.out_tensor.address_for_coordinate(start_coord)

            if cmd.in_tensor.compressed_values is not None:
                stream_index = cmd.in_tensor.compressed_stream_index_from_coord(start_coord)
                sz = cmd.in_tensor.size_of_compressed_stream(stream_index)
            else:
                sz = cmd.in_tensor.address_for_coordinate(cmd.box.end_coord, is_top_box=True) - src_addr

            emit.cmd0_with_param(cmd0.NPU_SET_DMA0_SRC_REGION, base_ptr_idx_map[cmd.in_tensor.mem_type])
            emit.cmd1_with_offset(cmd1.NPU_SET_DMA0_SRC, src_addr)
            if cmd.out_tensor.purpose == TensorPurpose.LUT:
                emit.cmd0_with_param(cmd0.NPU_SET_DMA0_DST_REGION, BasePointerIndex.Mem2Mem)
            else:
                emit.cmd0_with_param(cmd0.NPU_SET_DMA0_DST_REGION, base_ptr_idx_map[cmd.out_tensor.mem_type])

            emit.cmd1_with_offset(cmd1.NPU_SET_DMA0_DST, dst_addr)
            emit.cmd1_with_offset(cmd1.NPU_SET_DMA0_LEN, sz)
            dma_channel = 0
            mode = 0  # From external to external

            emit_cmd_waits(cmd_waits)
            emit.cmd_do_operation(cmd0.NPU_OP_DMA_START, dma_channel * 16 + mode)

        elif cmd.cmdtype == CommandType.NpuStripe:

            ps = cmd.ps
            primary_op = ps.primary_op
            npu_block_type = ps.npu_block_type
            # Specifies if global scale from the NPU_SET_OFM_SCALE register should be used instead of per-channel scale
            use_global_scale = False
            # Specifies type of rounding to be used.
            rounding_mode = (
                rounding.NATURAL if primary_op.attrs.get("rounding_mode", "") == b"NATURAL" else rounding.TFL
            )
            if primary_op.type == "ResizeBilinear":
                rounding_mode = rounding.TRUNCATE
            fmf = primary_op.attrs.get("fused_memory_function", None)
            faf = primary_op.attrs.get("fused_activation_function", None)
            fused_quantize = any(op.type == "Quantize" for op in ps.ops)
            # Force output scale, used in operations with fused LUT
            # Note: with current LUT support, forced_ofm_quantization is always equal to cmd.ofm_tensor.quantization
            # except when primary_op is AddAct + 0 (no-op) + LUT
            forced_ofm_quantization = primary_op.attrs.get("forced_output_quantization", None)
            ofm_quant = cmd.ofm_tensor.quantization
            if forced_ofm_quantization is not None:
                ofm_quant = forced_ofm_quantization

            # Specifies which operand to apply scaling to in bitexact elementwise ADD/SUB
            op_to_scale = 0

            # Update state history
            prev_ifm_rect = cur_ifm_rect
            prev_ifm_block_depth = cur_ifm_block_depth
            prev_ofm_rect = cur_ofm_rect
            prev_ofm_block = cur_ofm_block
            prev_kernel = cur_kernel
            cur_kernel = get_op_kernel(ps)

            block_config = ps.block_config
            emit.cmd0_with_param(cmd0.NPU_SET_OFM_BLK_HEIGHT_M1, block_config[0] - 1)
            emit.cmd0_with_param(cmd0.NPU_SET_OFM_BLK_WIDTH_M1, block_config[1] - 1)
            emit.cmd0_with_param(cmd0.NPU_SET_OFM_BLK_DEPTH_M1, block_config[3] - 1)

            shared_buffer = ps.shared_buffer

            if npu_block_type == NpuBlockType.ElementWise:
                ifm2_broadcast = 0

                if cmd.ifm2_tensor and not ifm_ifm2_correct_order(cmd.ifm_tensor.shape, cmd.ifm2_tensor.shape):
                    # The scalar has to be the ifm2 tensor so switch the ifms
                    cmd.ifm_tensor, cmd.ifm2_tensor = cmd.ifm2_tensor, cmd.ifm_tensor
                    cmd.ifm_box, cmd.ifm2_box = cmd.ifm2_box, cmd.ifm_box

                    # Set ReverseOperandOrder bit to IFM2_BROADCAST
                    ifm2_broadcast |= IFM2Broadcast.ReverseOperandOrder

                # Calculate scales needed for arithmetic elementwise operators
                if primary_op.type in set(("AddAct", "MulAct", "SubAct",)):
                    input_scale = cmd.ifm_tensor.quantization.scale_f32
                    input2_scale = cmd.ifm2_tensor.quantization.scale_f32
                    output_scale = ofm_quant.scale_f32
                    use_global_scale = True

                    if output_scale is not None and faf in ("Sigmoid", "Tanh"):
                        output_scale = 1 / 0x3000

                    if primary_op.type == "MulAct":
                        if None in (input_scale, input2_scale, output_scale):
                            ofm_scale = 1
                            shift = 0
                        else:
                            ofm_scale, shift = scaling.elementwise_mul_scale(input_scale, input2_scale, output_scale)
                        emit.cmd1_with_offset(cmd1.NPU_SET_OFM_SCALE, ofm_scale, shift)
                    else:  # AddAct/SubAct
                        # Force output scale same as the input scale for
                        # resizebilinear 1x1 that is converted to add
                        if "resizebilinear" in primary_op.attrs:
                            output_scale = input2_scale

                        if None in (input_scale, input2_scale, output_scale):
                            opa_scale = opb_scale = ofm_scale = 1
                            opa_shift = shift = 0
                            ofm_scale, shift = primary_op.attrs.get("rescale", [1, 0])
                        elif input_scale == input2_scale:
                            opa_scale, opb_scale, ofm_scale, shift = scaling.simplified_elementwise_add_sub_scale(
                                input_scale, input2_scale, output_scale
                            )
                            opa_shift = 0  # Unused for this case
                        else:
                            # Use advanced implementation only when input scales differ
                            bitdepth = cmd.ifm_tensor.dtype.bits
                            (
                                opa_scale,
                                opa_shift,
                                ofm_scale,
                                shift,
                                op_to_scale,
                            ) = scaling.advanced_elementwise_add_sub_scale(
                                input_scale, input2_scale, output_scale, bitdepth
                            )
                            opb_scale = 0  # Unused for this case
                            if ifm2_broadcast & IFM2Broadcast.ReverseOperandOrder:
                                # If the operand order is reversed we also have to swap which operand is scaled
                                if op_to_scale == scaling.OperandToScale.OPa:
                                    op_to_scale = scaling.OperandToScale.OPb
                                else:
                                    op_to_scale = scaling.OperandToScale.OPa

                        emit.cmd1_with_offset(cmd1.NPU_SET_OPA_SCALE, opa_scale, opa_shift)
                        emit.cmd1_with_offset(cmd1.NPU_SET_OPB_SCALE, opb_scale)
                        emit.cmd1_with_offset(cmd1.NPU_SET_OFM_SCALE, ofm_scale, shift)

                elif primary_op.type in set(("LeakyRelu", "Abs",)):
                    output_scale = ofm_quant.scale_f32
                    use_global_scale = True

                    if primary_op.type == "LeakyRelu":
                        output_scale = primary_op.attrs["alpha"]

                    ofm_scale, shift = scaling.quantise_scale(output_scale)
                    emit.cmd1_with_offset(cmd1.NPU_SET_OFM_SCALE, ofm_scale, shift)
                else:
                    emit.cmd1_with_offset(cmd1.NPU_SET_OFM_SCALE, 1, 0)

                # For elementwise set the required SHRAM to be equal to the total size of available SHRAM
                uses_lut = primary_op.activation_lut is not None
                shram_required = arch.available_shram_banks(uses_lut)
                emit.cmd0_with_param(cmd0.NPU_SET_IFM_IB_END, shram_required)

                # Acc buffers not needed so set AB_START to size of SHRAM
                emit.cmd0_with_param(cmd0.NPU_SET_AB_START, shram_required)

                # Is not a unary operator
                if cmd.ifm2_tensor is not None:
                    if cmd.ifm2_tensor.shape == []:
                        # IFM2 is a constant, set UseIFM2Scalar bit to IFM2_BROADCAST
                        ifm2_broadcast |= IFM2Broadcast.UseIFM2Scalar
                    else:
                        ifm_box_shape = cmd.ifm_box.get_size_shape()
                        ifm2_box_shape = cmd.ifm2_box.get_size_shape()

                        if len(cmd.ifm_tensor.shape) > 1 and ifm_box_shape[1] != ifm2_box_shape[1]:
                            # Broadcast in 'H' dimension
                            assert cmd.ifm2_tensor.shape[1] == 1
                            ifm2_broadcast |= IFM2Broadcast.BroadcastHdim

                        if len(cmd.ifm_tensor.shape) > 2 and ifm_box_shape[2] != ifm2_box_shape[2]:
                            # Broadcast in 'W' dimension
                            assert cmd.ifm2_tensor.shape[2] == 1
                            ifm2_broadcast |= IFM2Broadcast.BroadcastWdim

                        if len(cmd.ifm_tensor.shape) > 3 and ifm_box_shape[3] != ifm2_box_shape[3]:
                            # Broadcast in 'C' dimension
                            assert cmd.ifm2_tensor.shape[3] == 1
                            ifm2_broadcast |= IFM2Broadcast.BroadcastCdim

                        # Set IFM2_IB_START to the latter half of the IB space
                        ifm_ib_start = shared_buffer.bank_locations[SharedBufferArea.IFM]
                        emit.cmd0_with_param(
                            cmd0.NPU_SET_IFM2_IB_START, (shram_required - ifm_ib_start) / 2 + ifm_ib_start
                        )

                    emit.cmd0_with_param(cmd0.NPU_SET_IFM2_BROADCAST, ifm2_broadcast)

            else:
                emit.cmd0_with_param(
                    cmd0.NPU_SET_IFM_IB_END,
                    shared_buffer.bank_locations[SharedBufferArea.IFM]
                    + shared_buffer.banks_required[SharedBufferArea.IFM],
                )
                emit.cmd0_with_param(cmd0.NPU_SET_AB_START, shared_buffer.bank_locations[SharedBufferArea.Accumulators])

            emit.cmd0_with_param(cmd0.NPU_SET_ACC_FORMAT, acc_format_map[shared_buffer.use_accumulator_element])

            if primary_op.type == "ResizeBilinear":
                # perform nearest neighbor upscale
                emit.cmd0_with_param(cmd0.NPU_SET_IFM_UPSCALE, resampling_mode.NEAREST)
            elif primary_op.type == "Conv2DBackpropInputSwitchedBias":
                # perform insert zero upscale
                emit.cmd0_with_param(cmd0.NPU_SET_IFM_UPSCALE, resampling_mode.TRANSPOSE)
            else:
                emit.cmd0_with_param(cmd0.NPU_SET_IFM_UPSCALE, resampling_mode.NONE)

            if npu_block_type in set(
                (
                    NpuBlockType.ConvolutionMxN,
                    NpuBlockType.ConvolutionDepthWise,
                    NpuBlockType.Pooling,
                    NpuBlockType.ReduceSum,
                )
            ):
                # Set up padding
                explicit_padding = list(primary_op.attrs["explicit_padding"])  # (top, left, bottom, right)

                # Check if this is for horizontal ifm streaming
                if not (cmd.is_first_h_stripe and cmd.is_last_h_stripe):
                    explicit_padding[0] = cmd.pad_top
                    explicit_padding[2] = cmd.pad_bottom

                # Indexing from end since a 1x1 Avgpool might have been added with non 4-dimensional input/output,
                # because of activation function needed to be fused.
                if cmd.ifm_box.start_coord[-2] > 0:
                    explicit_padding[1] = 0
                if cmd.ifm_box.end_coord[-2] < cmd.ifm_tensor.shape[-2]:
                    explicit_padding[3] = 0
                emit.cmd0_with_param(cmd0.NPU_SET_IFM_PAD_TOP, explicit_padding[0])
                emit.cmd0_with_param(cmd0.NPU_SET_IFM_PAD_LEFT, explicit_padding[1])
                emit.cmd0_with_param(cmd0.NPU_SET_IFM_PAD_BOTTOM, explicit_padding[2])
                emit.cmd0_with_param(cmd0.NPU_SET_IFM_PAD_RIGHT, explicit_padding[3])

                # set kernel x stride low bit
                stride = primary_op.attrs["strides"][2] - 1 & 1
                # set kernel y stride low bit
                stride |= (primary_op.attrs["strides"][1] - 1 & 1) << 1
                # set kernel x stride extension bits
                stride |= (primary_op.attrs["strides"][2] - 1 >> 1) << 6
                # set kernel y stride extension bits
                stride |= (primary_op.attrs["strides"][1] - 1 >> 1) << 9

                if npu_block_type in set((NpuBlockType.Pooling, NpuBlockType.ReduceSum)):
                    k_height, k_width = primary_op.attrs["ksize"][1:3]
                    emit.cmd0_with_param(cmd0.NPU_SET_KERNEL_HEIGHT_M1, k_height - 1)
                    emit.cmd0_with_param(cmd0.NPU_SET_KERNEL_WIDTH_M1, k_width - 1)

                    valid_padding = sum(explicit_padding) == 0

                    if (
                        primary_op.type in set(("AvgPool", "AvgPoolAct", "ResizeBilinear", "ReduceSum"))
                        and valid_padding
                    ):
                        # For valid padding vela has to output scaling values
                        if faf == "Sigmoid" or faf == "Tanh":
                            rescale = 0x3000 * cmd.ifm_tensor.quantization.scale_f32
                            if cmd.ifm_tensor.dtype == DataType.int16:
                                # Calculate scale and shift for the output scale of 1/(3*4096)
                                shift = 0
                                max_rescale = np.iinfo(np.int16).max / 2
                                while rescale <= max_rescale and shift <= 30:
                                    shift += 1
                                    rescale *= 2
                                scale = int(rescale)
                            else:
                                rescale_bits = len(bin(round_up_to_int(rescale))) - 2 + 1
                                scale, shift = scaling.quantise_pooling_scale(k_height * k_width, rescale_bits)
                                scale = int(round_away_zero(scale * rescale))
                        elif fused_quantize:
                            # Quantize op requires different scaling
                            ifm_scale_f64 = np.double(cmd.ifm_tensor.quantization.scale_f32)
                            ofm_scale_f64 = np.double(ofm_quant.scale_f32)
                            scale, shift = scaling.quantise_scale(ifm_scale_f64 / ofm_scale_f64)
                        elif primary_op.type == "ResizeBilinear" and "rescale" in primary_op.attrs:
                            rescale = primary_op.attrs["rescale"]
                            rescale_bits = len(bin(round_up_to_int(rescale))) - 2 + 1
                            scale, shift = scaling.quantise_pooling_scale(k_height * k_width, rescale_bits)
                            scale = int(round_away_zero(scale * rescale))
                        else:
                            # In case avg pool fused with concat or other memory operation, rescaling might be needed.
                            # k_height == k_width == 1 is allways true in this case
                            # Normally the scale is maximised, to get maximum precision, which means that
                            # if rescale != 1, scale need to consider the number of bits needed for rescaling
                            if None not in (ofm_quant.scale_f32, cmd.ifm_tensor.quantization.scale_f32,):
                                rescale = cmd.ifm_tensor.quantization.scale_f32 / ofm_quant.scale_f32
                                rescale_bits = 0
                                if k_height == k_width == 1:
                                    if fmf == "ConcatSliceWrite":
                                        rounding_mode = rounding.NATURAL
                                    if rescale > 1:
                                        rescale_bits = len(bin(round_up_to_int(rescale))) - 2 + 1
                                    elif rescale < 1:
                                        rescale_bits = -(len(bin(round_up_to_int(1 / rescale))) - 2 - 1)
                                scale, shift = scaling.quantise_pooling_scale(k_height * k_width, rescale_bits)
                                scale = int(round_away_zero(scale * rescale))
                            else:
                                scale = 1
                                shift = 0

                        emit.cmd1_with_offset(cmd1.NPU_SET_OFM_SCALE, scale, shift)
                        # Valid-padded average pool should use the global scale from
                        # NPU_SET_OFM_SCALE register, which is set above.
                        use_global_scale = True

                else:  # Convolution
                    assert cmd.weight_tensor.block_traversal != TensorBlockTraversal.Default
                    # Reduced precision quantization and natural rounding used for int16
                    if cmd.ifm_tensor.dtype == DataType.int16:
                        rounding_mode = rounding.NATURAL
                    stride |= (cur_kernel.dilation.y - 1) << 4
                    stride |= (cur_kernel.dilation.x - 1) << 3
                    emit.cmd0_with_param(
                        cmd0.NPU_SET_KERNEL_HEIGHT_M1, cur_kernel.dilation.y * (cmd.weight_tensor.shape[0] - 1)
                    )
                    emit.cmd0_with_param(
                        cmd0.NPU_SET_KERNEL_WIDTH_M1, cur_kernel.dilation.x * (cmd.weight_tensor.shape[1] - 1)
                    )
                    if cmd.weight_tensor.block_traversal == TensorBlockTraversal.PartKernelFirst:
                        # Part-kernel-first weight ordering
                        assert npu_block_type == NpuBlockType.ConvolutionMxN
                        stride |= 1 << 2

                emit.cmd0_with_param(cmd0.NPU_SET_KERNEL_STRIDE, stride)

            elif npu_block_type in set((NpuBlockType.VectorProduct,)):
                # Vector product is implemented using a 1x1 convolution so need
                # to setup the appropriate padding and kernel info
                emit.cmd0_with_param(cmd0.NPU_SET_IFM_PAD_TOP, 0)
                emit.cmd0_with_param(cmd0.NPU_SET_IFM_PAD_LEFT, 0)
                emit.cmd0_with_param(cmd0.NPU_SET_IFM_PAD_BOTTOM, 0)
                emit.cmd0_with_param(cmd0.NPU_SET_IFM_PAD_RIGHT, 0)

                # kernel stride reg = 0 means stride(1,1) + depth first weight
                # order + dilation(0,0) + kernel_split_size=8
                emit.cmd0_with_param(cmd0.NPU_SET_KERNEL_STRIDE, 0)

                emit.cmd0_with_param(cmd0.NPU_SET_KERNEL_HEIGHT_M1, 0)
                emit.cmd0_with_param(cmd0.NPU_SET_KERNEL_WIDTH_M1, 0)

            if npu_block_type in set(
                (NpuBlockType.ConvolutionMxN, NpuBlockType.ConvolutionDepthWise, NpuBlockType.VectorProduct)
            ):
                # Emit Weight base address commands, only maps the area required for
                # this command's weights from the larger tensor.
                stream_index = cmd.weight_tensor.compressed_stream_index_from_coord(cmd.weight_box.start_coord)
                weight_substream_offsets = cmd.weight_tensor.compressed_values_substream_offsets[stream_index]
                substreams = len(weight_substream_offsets) - 1  # Offset list must terminate with full stream length

                # Extract weight substream offsets and calculate their lengths
                assert len(weight_substream_offsets) > 1 and (weight_substream_offsets[0] == 0)
                weight_addr = cmd.weight_tensor.address_for_coordinate(cmd.weight_box.start_coord)

                # Set weights sources for active and present cores
                for core, param in enumerate(
                    [
                        (cmd1.NPU_SET_WEIGHT_BASE, cmd1.NPU_SET_WEIGHT_LENGTH),
                        (cmd1.NPU_SET_WEIGHT1_BASE, cmd1.NPU_SET_WEIGHT1_LENGTH),
                    ]
                ):
                    if core < substreams:
                        emit.cmd1_with_offset(param[0], weight_addr + weight_substream_offsets[core])
                        emit.cmd1_with_offset(
                            param[1], weight_substream_offsets[core + 1] - weight_substream_offsets[core]
                        )
                    elif core < arch.ncores:
                        emit.cmd1_with_offset(param[0], weight_addr)
                        emit.cmd1_with_offset(param[1], 0)

                weight_region = base_ptr_idx_map[cmd.weight_tensor.mem_type]
                emit.cmd0_with_param(cmd0.NPU_SET_WEIGHT_REGION, weight_region)

                # Emit Scale & Bias base address commands, with length matching the amount required by
                # the weight tensors.
                if cmd.scale_tensor is not None:
                    scale_substream_offsets = cmd.scale_tensor.compressed_values_substream_offsets[stream_index]
                    substreams = len(scale_substream_offsets) - 1  # Offset list must terminate with full stream length

                    # Extract scale substream offsets and calculate their lengths
                    assert len(scale_substream_offsets) > 1 and (scale_substream_offsets[0] == 0)
                    scale_addr = cmd.scale_tensor.address_for_coordinate(cmd.weight_box.start_coord[-1:])

                    # Set scale sources for active and present cores
                    for core, param in enumerate(
                        [
                            (cmd1.NPU_SET_SCALE_BASE, cmd1.NPU_SET_SCALE_LENGTH),
                            (cmd1.NPU_SET_SCALE1_BASE, cmd1.NPU_SET_SCALE1_LENGTH),
                        ]
                    ):
                        if core < substreams:
                            emit.cmd1_with_offset(param[0], scale_addr + scale_substream_offsets[core])
                            emit.cmd1_with_offset(
                                param[1], scale_substream_offsets[core + 1] - scale_substream_offsets[core]
                            )
                        elif core < arch.ncores:
                            emit.cmd1_with_offset(param[0], scale_addr)
                            emit.cmd1_with_offset(param[1], 0)

                    # Emit base address for NPU to access scale & bias data
                    scale_region = base_ptr_idx_map[cmd.scale_tensor.mem_type]
                    emit.cmd0_with_param(cmd0.NPU_SET_SCALE_REGION, scale_region)

            ofm_quant_qmin = ofm_quant.quant_min
            ofm_quant_qmax = ofm_quant.quant_max
            ifm_min = cmd.ifm_tensor.quantization.min
            ifm_max = cmd.ifm_tensor.quantization.max

            # Emit commands for any fused activation function
            if faf is None:
                emit.cmd0_with_param(cmd0.NPU_SET_ACTIVATION, activation.NONE)
                # Even if no activation function, values need to be set to override previous values
                faf_min = ofm_quant_qmin
                faf_max = ofm_quant_qmax
            elif faf == "Relu":
                emit.cmd0_with_param(cmd0.NPU_SET_ACTIVATION, activation.NONE)
                faf_min = quantise_float32(0.0, ofm_quant.scale_f32, ofm_quant.zero_point)
                faf_max = ofm_quant_qmax
            elif faf == "Relu6":
                emit.cmd0_with_param(cmd0.NPU_SET_ACTIVATION, activation.NONE)
                faf_min = quantise_float32(0.0, ofm_quant.scale_f32, ofm_quant.zero_point)
                faf_max = quantise_float32(6.0, ofm_quant.scale_f32, ofm_quant.zero_point)
            elif faf == "ReluN1To1":
                emit.cmd0_with_param(cmd0.NPU_SET_ACTIVATION, activation.NONE)
                faf_min = quantise_float32(-1.0, ofm_quant.scale_f32, ofm_quant.zero_point)
                faf_max = quantise_float32(1.0, ofm_quant.scale_f32, ofm_quant.zero_point)
            elif faf == "Tanh":
                emit.cmd0_with_param(cmd0.NPU_SET_ACTIVATION, activation.TANH)
                if primary_op.type in set(("AvgPool", "AvgPoolAct", "ResizeBilinear")):
                    faf_min = quantise_float32(-1.0, ofm_quant.scale_f32, ofm_quant.zero_point)
                    faf_max = quantise_float32(1.0, ofm_quant.scale_f32, ofm_quant.zero_point)
                else:
                    faf_min = quantise_float32(clamp_tanh(ifm_min), ofm_quant.scale_f32, ofm_quant.zero_point)
                    faf_max = quantise_float32(clamp_tanh(ifm_max), ofm_quant.scale_f32, ofm_quant.zero_point)
            elif faf == "Sigmoid":
                emit.cmd0_with_param(cmd0.NPU_SET_ACTIVATION, activation.SIGMOID)
                if primary_op.type in set(("AvgPool", "AvgPoolAct", "ResizeBilinear")):
                    faf_min = quantise_float32(0, ofm_quant.scale_f32, ofm_quant.zero_point)
                    faf_max = quantise_float32(1.0, ofm_quant.scale_f32, ofm_quant.zero_point)
                else:
                    faf_min = quantise_float32(clamp_sigmoid(ifm_min), ofm_quant.scale_f32, ofm_quant.zero_point)
                    faf_max = quantise_float32(clamp_sigmoid(ifm_max), ofm_quant.scale_f32, ofm_quant.zero_point)
            elif faf == "LUT":
                lut_index = int(activation.LUT_START.value) + primary_op.attrs.get("lut_index", -1)
                assert activation.LUT_START.value <= lut_index <= activation.LUT_END.value, "LUT index out of range."
                if cmd.ofm_tensor.dtype == DataType.int32:
                    lut_index |= 3 << 12  # Force I8 range
                emit.cmd0_with_param(cmd0.NPU_SET_ACTIVATION, lut_index)
                faf_min = ofm_quant_qmin
                faf_max = ofm_quant_qmax
            else:
                raise Exception("Unsupported fused_activation_function = " + faf)

            # Activation range needs to be set based upon the quantisation range and the fused activation range
            emit.cmd0_with_param(cmd0.NPU_SET_ACTIVATION_MIN, max(ofm_quant_qmin, faf_min))
            emit.cmd0_with_param(cmd0.NPU_SET_ACTIVATION_MAX, min(ofm_quant_qmax, faf_max))

            out_shape = cmd.ofm_box.get_size_shape()
            if len(out_shape) >= 4:
                emit.cmd0_with_param(cmd0.NPU_SET_OFM_HEIGHT_M1, out_shape[-3] - 1)
            else:
                emit.cmd0_with_param(cmd0.NPU_SET_OFM_HEIGHT_M1, 0)
            if len(out_shape) >= 2:
                emit.cmd0_with_param(cmd0.NPU_SET_OFM_WIDTH_M1, out_shape[-2] - 1)
            else:
                emit.cmd0_with_param(cmd0.NPU_SET_OFM_WIDTH_M1, 0)
            emit.cmd0_with_param(cmd0.NPU_SET_OFM_DEPTH_M1, out_shape[-1] - 1)

            if npu_block_type in set((NpuBlockType.ConvolutionMxN, NpuBlockType.VectorProduct, NpuBlockType.ReduceSum)):
                in_shape = cmd.ifm_box.get_size_shape()
                emit.cmd0_with_param(cmd0.NPU_SET_IFM_DEPTH_M1, in_shape[-1] - 1)
            else:
                emit.cmd0_with_param(cmd0.NPU_SET_IFM_DEPTH_M1, out_shape[-1] - 1)

            for tens, box, region_op, ptr_ops, stride_ops, zero_point_op in (
                (
                    cmd.ifm_tensor,
                    cmd.ifm_box,
                    cmd0.NPU_SET_IFM_REGION,
                    (cmd1.NPU_SET_IFM_BASE0, cmd1.NPU_SET_IFM_BASE1, cmd1.NPU_SET_IFM_BASE2, cmd1.NPU_SET_IFM_BASE3),
                    (cmd1.NPU_SET_IFM_STRIDE_C, cmd1.NPU_SET_IFM_STRIDE_Y, cmd1.NPU_SET_IFM_STRIDE_X),
                    cmd0.NPU_SET_IFM_ZERO_POINT,
                ),
                (
                    cmd.ifm2_tensor,
                    cmd.ifm2_box,
                    cmd0.NPU_SET_IFM2_REGION,
                    (
                        cmd1.NPU_SET_IFM2_BASE0,
                        cmd1.NPU_SET_IFM2_BASE1,
                        cmd1.NPU_SET_IFM2_BASE2,
                        cmd1.NPU_SET_IFM2_BASE3,
                    ),
                    (cmd1.NPU_SET_IFM2_STRIDE_C, cmd1.NPU_SET_IFM2_STRIDE_Y, cmd1.NPU_SET_IFM2_STRIDE_X),
                    cmd0.NPU_SET_IFM2_ZERO_POINT,
                ),
                (
                    cmd.ofm_tensor,
                    cmd.ofm_box,
                    cmd0.NPU_SET_OFM_REGION,
                    (cmd1.NPU_SET_OFM_BASE0, cmd1.NPU_SET_OFM_BASE1, cmd1.NPU_SET_OFM_BASE2, cmd1.NPU_SET_OFM_BASE3),
                    (cmd1.NPU_SET_OFM_STRIDE_C, cmd1.NPU_SET_OFM_STRIDE_Y, cmd1.NPU_SET_OFM_STRIDE_X),
                    cmd0.NPU_SET_OFM_ZERO_POINT,
                ),
            ):

                if tens is None:
                    continue

                need_zero_point = (faf is not None) or (fmf == "ConcatSliceWrite") or fused_quantize
                if (
                    (
                        primary_op.type in set(("AvgPool", "AvgPoolAct", "ResizeBilinear", "CLZ", "SHL"))
                        and not need_zero_point
                    )
                    or (
                        tens.dtype == DataType.int32
                        and zero_point_op in (cmd0.NPU_SET_IFM_ZERO_POINT, cmd0.NPU_SET_IFM2_ZERO_POINT)
                    )
                    or tens.quantization is None
                ):
                    # Actual integer operation, just set scale to 1 and zero point to 0
                    emit.cmd0_with_param(zero_point_op, 0)
                else:
                    assert tens.quantization.zero_point is not None, "need an actual zero point set"
                    if cmd0.NPU_SET_OFM_ZERO_POINT == zero_point_op and forced_ofm_quantization is not None:
                        zero_point = forced_ofm_quantization.zero_point
                    elif (
                        "resizebilinear" in primary_op.attrs
                        and primary_op.type == "AddAct"
                        and cmd0.NPU_SET_OFM_ZERO_POINT == zero_point_op
                    ):
                        # Force output zero point same as the input zero point
                        # for resizebilinear 1x1 that is converted to add
                        zero_point = cmd.ifm2_tensor.quantization.zero_point
                    else:
                        zero_point = tens.quantization.zero_point
                    emit.cmd0_with_param(zero_point_op, int(zero_point))

                if tens.shape == []:
                    # Empty shape, elementwise constant
                    ifm2_scalar = tens.quant_values
                    assert ifm2_scalar.size == 1
                    emit.cmd0_with_param(cmd0.NPU_SET_IFM2_SCALAR, int(ifm2_scalar.item(0)))
                    continue

                height_0, height_1, width_0, addresses = tens.addresses_for_rolling_buffer(
                    box.start_coord, box.end_coord
                )
                if npu_block_type != NpuBlockType.VectorProduct:
                    if tens == cmd.ifm_tensor:
                        emit.cmd0_with_param(cmd0.NPU_SET_IFM_HEIGHT0_M1, height_0 - 1)
                        emit.cmd0_with_param(cmd0.NPU_SET_IFM_HEIGHT1_M1, height_1 - 1)
                        emit.cmd0_with_param(cmd0.NPU_SET_IFM_WIDTH0_M1, width_0 - 1)
                    elif tens == cmd.ofm_tensor:
                        emit.cmd0_with_param(cmd0.NPU_SET_OFM_HEIGHT0_M1, height_0 - 1)
                        emit.cmd0_with_param(cmd0.NPU_SET_OFM_HEIGHT1_M1, height_1 - 1)
                        emit.cmd0_with_param(cmd0.NPU_SET_OFM_WIDTH0_M1, width_0 - 1)
                    if tens == cmd.ifm2_tensor:
                        emit.cmd0_with_param(cmd0.NPU_SET_IFM2_HEIGHT0_M1, height_0 - 1)
                        emit.cmd0_with_param(cmd0.NPU_SET_IFM2_HEIGHT1_M1, height_1 - 1)
                        emit.cmd0_with_param(cmd0.NPU_SET_IFM2_WIDTH0_M1, width_0 - 1)
                else:
                    if len(out_shape) == 2:
                        # TODO: N is put in W-dimension for now
                        # Should be spread over H and W, but then block size selectetion,
                        # and stride calculation should be changed
                        if tens == cmd.ifm_tensor:
                            emit.cmd0_with_param(cmd0.NPU_SET_IFM_WIDTH0_M1, out_shape[-2] - 1)
                        elif tens == cmd.ofm_tensor:
                            emit.cmd0_with_param(cmd0.NPU_SET_OFM_WIDTH0_M1, out_shape[-2] - 1)
                    else:
                        assert False

                emit.cmd0_with_param(region_op, base_ptr_idx_map[tens.mem_type])

                for idx, addr in enumerate(addresses):
                    if addr is None:
                        addresses[idx] = 0

                emit.cmd1_with_offset(ptr_ops[0], addresses[0])
                emit.cmd1_with_offset(ptr_ops[1], addresses[1])
                emit.cmd1_with_offset(ptr_ops[2], addresses[2])
                emit.cmd1_with_offset(ptr_ops[3], addresses[3])

                strides = tens.get_strides()
                emit.cmd1_with_offset(stride_ops[0], strides[1])  # stride between 16-byte channel blocks (C)
                emit.cmd1_with_offset(stride_ops[2], strides[3])  # stride between horisontal values (W)
                emit.cmd1_with_offset(stride_ops[1], strides[2])  # stride between vertical values (H)

                if tens.format == TensorFormat.NHCWB16:
                    # Check that all BasePointer addresses are aligned to 16 bytes
                    assert (int(addresses[0]) % 16) == 0
                    assert (int(addresses[1]) % 16) == 0
                    assert (int(addresses[2]) % 16) == 0
                    assert (int(addresses[3]) % 16) == 0

            ofm_dtype = cmd.ofm_tensor.dtype
            assert ofm_dtype.type & BaseType.Int
            prec = 0
            if ofm_dtype.size_in_bits() == 8:
                prec = 0
            elif ofm_dtype.size_in_bits() == 16:
                prec = 2
            elif ofm_dtype.size_in_bits() == 32:
                prec = 4
            else:
                assert 0

            if ofm_dtype.type & BaseType.Signed:
                prec += 1

            if use_global_scale:
                # Set global scale bit, as opposed to using per channel scale
                prec |= 1 << 8

            if cmd.ofm_tensor.format == TensorFormat.NHCWB16:
                prec |= 1 << 6

            prec |= rounding_mode.value << 14

            emit.cmd0_with_param(cmd0.NPU_SET_OFM_PRECISION, prec)

            prec = None
            weight_bits = 8
            if cmd.weight_tensor is not None:
                weight_bits = cmd.weight_tensor.dtype.size_in_bits()

            ifm_dtype = cmd.ifm_tensor.dtype

            assert weight_bits == 8, "Unsupported weight bit depth"
            assert (
                ifm_dtype.size_in_bits() in {8, 16}
                or ifm_dtype.size_in_bits() == 32
                and npu_block_type in (NpuBlockType.ElementWise, NpuBlockType.ReduceSum)
            ), "Unsupported ifm bit depth"

            if ifm_dtype.size_in_bits() == 8:
                if ifm_dtype.type & BaseType.Signed:
                    prec = ifm_precision.S8
                else:
                    prec = ifm_precision.U8
            elif ifm_dtype.size_in_bits() == 16:
                if ifm_dtype.type & BaseType.Signed:
                    prec = ifm_precision.S16
                else:
                    prec = ifm_precision.U16
            elif ifm_dtype == DataType.int32:
                prec = ifm_precision.S32

            ifm_prec = prec.value
            ifm2_prec = ifm_prec

            if cmd.ifm_tensor.format == TensorFormat.NHCWB16:
                ifm_prec |= 1 << 6

            ifm_prec |= op_to_scale << 8

            emit.cmd0_with_param(cmd0.NPU_SET_IFM_PRECISION, ifm_prec)

            if cmd.ifm2_tensor is not None:
                if cmd.ifm2_tensor.format == TensorFormat.NHCWB16:
                    ifm2_prec |= 1 << 6
                emit.cmd0_with_param(cmd0.NPU_SET_IFM2_PRECISION, ifm2_prec)

            # Get op parameters
            cur_ifm_block_depth = get_op_ifmofm_block_depth(arch, cmd)
            cur_ofm_block = Block(ps.block_config[1], ps.block_config[0], ps.block_config[3])
            cur_ofm_rect = get_op_ofm_rect(cmd)
            cur_ifm_rect = get_op_ifm_rect(cmd)
            cur_padLT = get_op_padding_lt(cmd)
            if (prev_kernel is not None) and (cur_kernel is not None) and has_prev_op_dependency(prev_cmd, cmd):
                if cmd.ifm_tensor.shape == prev_cmd.ofm_tensor.shape:
                    blockdep = arch.calc_block_dep(
                        prev_ifm_rect,
                        prev_ofm_rect,
                        prev_ifm_block_depth,
                        prev_ofm_block,
                        prev_kernel,
                        cur_ifm_rect,
                        cur_ofm_rect,
                        cur_ifm_block_depth,
                        cur_ofm_block,
                        cur_kernel,
                        cur_padLT,
                    )
                else:
                    blockdep = 0
            else:
                blockdep = ArchitectureFeatures.MAX_BLOCKDEP

            # Set between every op (dependent or not)
            blockdep = min(blockdep, arch.max_blockdep)
            emit.cmd0_with_param(cmd0.NPU_SET_BLOCKDEP, blockdep)
            prev_cmd = cmd

            emit_cmd_waits(cmd_waits)

            if npu_block_type == NpuBlockType.ConvolutionMxN:
                emit.cmd_do_operation(cmd0.NPU_OP_CONV)
            elif npu_block_type == NpuBlockType.ConvolutionDepthWise:
                emit.cmd_do_operation(cmd0.NPU_OP_DEPTHWISE)
            elif npu_block_type == NpuBlockType.VectorProduct:
                # Vector product is implemented using a 1x1 convolution
                emit.cmd_do_operation(cmd0.NPU_OP_CONV)
            elif npu_block_type == NpuBlockType.Pooling:
                param = pooling_mode.MAX.value if "Max" in primary_op.type else pooling_mode.AVERAGE.value
                emit.cmd_do_operation(cmd0.NPU_OP_POOL, param=param)
            elif npu_block_type == NpuBlockType.ReduceSum:
                emit.cmd_do_operation(cmd0.NPU_OP_POOL, param=pooling_mode.REDUCE_SUM.value)
            elif npu_block_type == NpuBlockType.ElementWise:
                param = elementwise_mode_map[primary_op.type]
                emit.cmd_do_operation(cmd0.NPU_OP_ELEMENTWISE, param)
            else:
                print("Warning: Skipping register command stream generation for", ps)

    # Fill in final part of command stream:
    emit.cmd_do_operation(cmd0.NPU_OP_STOP, param=0xFFFF)

    sg.register_command_stream = emit.to_list()
    if verbose:
        emit.print_cmds()
        print("number of commands", len(emit.cmd_stream))
        print("command stream length in words", len(sg.register_command_stream))
