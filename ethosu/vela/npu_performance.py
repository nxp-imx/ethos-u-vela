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
# NPU performance estimation functions to estimate performance of a Pass and CascadedPass. Uses a model that takes the
# maximum of the 'cycles required for bandwidth' and 'cycles required for computing'.
#
# Called during scheduling to evaluate different proposals, as well as post-scheduling to provide a final performance
# estimate.
import copy
import csv
from enum import auto
from enum import IntEnum
from typing import Optional
from typing import Set
from uuid import UUID

import numpy as np

from . import numeric_util
from .architecture_allocator import ArchitectureBlockConfig
from .architecture_features import Accelerator
from .architecture_features import ArchitectureFeatures
from .architecture_features import NpuBlockType
from .architecture_features import SHRAMElements
from .architecture_features import TensorFormat
from .debug_database import DebugDatabase
from .nn_graph import Graph
from .nn_graph import NetworkType
from .nn_graph import PassPlacement
from .numeric_util import round_up
from .numeric_util import round_up_to_int
from .operation import Kernel
from .operation import Op
from .scheduler import Schedule
from .scheduler import SchedulerOperation
from .scheduler import SchedulerOpInfo
from .shape4d import Shape4D
from .tensor import BandwidthDirection
from .tensor import MemArea
from .tensor import TensorPurpose
from .tensor import TensorSubPurpose
from .tflite_mapping import optype_to_builtintype as tflite_optype_to_builtintype
from .tosa_mapping import optype_to_tosa_op_type as tosa_optype_to_tosa_op_type
from .weight_compressor import WeightKey


class PassCycles(IntEnum):
    Npu = 0
    SramAccess = auto()
    DramAccess = auto()
    OnChipFlashAccess = auto()
    OffChipFlashAccess = auto()
    Total = auto()
    Size = auto()

    def display_name(self):
        return (
            "NPU",
            "SRAM Access",
            "DRAM Access",
            "On-chip Flash Access",
            "Off-chip Flash Access",
            "Total",
            "Size",
        )[self.value]

    def identifier_name(self):
        return (
            "npu",
            "sram_access",
            "dram_access",
            "on_chip_flash_access",
            "off_chip_flash_access",
            "total",
            "size",
        )[self.value]

    @staticmethod
    def all():
        return (
            PassCycles.Npu,
            PassCycles.SramAccess,
            PassCycles.DramAccess,
            PassCycles.OnChipFlashAccess,
            PassCycles.OffChipFlashAccess,
            PassCycles.Total,
        )


class PerformanceQuery:
    def __init__(self, npu_block_type=0):
        self.npu_block_type = npu_block_type
        self.ifm_shape = Shape4D(0, 0, 0, 0)
        self.ifm_format = TensorFormat.NHWC
        self.ifm_memory_area = MemArea.Unknown
        self.ifm2_memory_area = MemArea.Unknown
        self.ifm_bits = 0
        self.ifm2_bits = 0
        self.ifm2_shape = None
        self.ifm2_format = TensorFormat.NHWC
        self.ofm_shape = Shape4D(0, 0, 0, 0)
        self.ofm_format = TensorFormat.NHWC
        self.ofm_memory_area = MemArea.Unknown
        self.ofm_bits = 0
        self.const_shape = Shape4D(0, 0, 0, 0)
        self.const_memory_area = MemArea.Unknown
        self.kernel = Kernel(1, 1)
        self.config = ArchitectureBlockConfig()


class CycleCost:
    def __init__(self):
        self.op_macs = 0
        self.op_cycles = 0

    def __mul__(self, scale):
        out = CycleCost()
        out.op_macs = self.op_macs * scale
        out.op_cycles = self.op_cycles * scale
        return out

    def __iadd__(self, rhs):
        self.op_macs += rhs.op_macs
        self.op_cycles += rhs.op_cycles
        return self

    def __str__(self):
        return "macs = {}, cycles = {}".format(self.op_macs, self.op_cycles)


class ElementAccess:
    def __init__(self):
        # List of ONLY element access counts, consumers
        # need to scale these values by the correct bitwidths
        # to calculated memory bandwidth
        self.ifm_read = [0, 0]  # ifm1, ifm2
        self.ofm_write = 0
        self.weights_refetch = 0
        self.const_read = [0, 0]  # weights, scales

    def __mul__(self, scale):
        out = ElementAccess()
        out.ifm_read[0] = self.ifm_read[0] * scale
        out.ifm_read[1] = self.ifm_read[1] * scale
        out.ofm_write = self.ofm_write * scale
        out.weights_refetch = self.weights_refetch * scale
        out.const_read[0] = self.const_read[0] * scale
        out.const_read[1] = self.const_read[1] * scale
        return out

    def __iadd__(self, rhs):
        self.ifm_read[0] += rhs.ifm_read[0]
        self.ifm_read[1] += rhs.ifm_read[1]
        self.ofm_write += rhs.ofm_write
        self.weights_refetch += rhs.weights_refetch
        self.const_read[0] += rhs.const_read[0]
        self.const_read[1] += rhs.const_read[1]
        return self

    def __str__(self):
        return "ifm read = {}, ofm write = {}, const read={}".format(self.ifm_read, self.ofm_write, self.const_read)


def _strides_for_shape(shape: Shape4D, format: TensorFormat, element_bits):
    if format == TensorFormat.NHWC:
        strides = [0, 0, 0, 0]
        strides[3] = element_bits / 8  # +Z
        strides[2] = (element_bits * shape.depth) // 8  # +X
        strides[1] = (element_bits * shape.depth * shape.width) // 8  # +Y
        strides[0] = (element_bits * shape.depth * shape.width * shape.height) // 8  # +N
    elif format == TensorFormat.NHCWB16:
        strides = [0, 0, 0, 0, 0]
        strides[4] = element_bits / 8  # +Z
        strides[3] = (element_bits * 16) / 8  # +X
        strides[2] = (element_bits * 16 * shape.width) / 8  # +C
        strides[1] = (element_bits * shape.width * shape.depth) / 8  # +Y
        strides[0] = (element_bits * shape.width * shape.depth) / 8  # +N

    return strides


def _estimate_memory_transfer_efficiency(
    arch, is_read, mem_area, format, element_bits, block_size, shape4D, to_transfer
):
    burst_len = 8

    strides = _strides_for_shape(shape4D, format, element_bits)

    if format == TensorFormat.NHCWB16:
        if strides[2] == block_size.depth:  # TODO is this check corrrect for non 8-bit
            burst_len = element_bits * block_size.depth * block_size.width
        elif is_read:
            burst_len = 16 * element_bits * block_size.width
        else:
            burst_len = 16 * element_bits * block_size.width * arch.ncores
    elif format == TensorFormat.NHWC:
        if is_read:
            if strides[3] == block_size.depth:
                burst_len = element_bits * block_size.depth * block_size.width
            else:
                burst_len = element_bits * block_size.depth
        else:
            if block_size.depth <= 16 and strides[3] == block_size.depth:
                burst_len = element_bits * block_size.depth * block_size.width
            else:
                burst_len = min(64 * 8, 16 * element_bits * arch.ncores, block_size.depth * element_bits)

    burst_len = burst_len // 8  # bits->bytes
    burst_len = min(arch.memory_burst_length[mem_area], burst_len)
    return to_transfer * (arch.memory_burst_length[mem_area] / burst_len)


def _estimate_minimum_memory_cycles(arch, query: PerformanceQuery):
    # Input block HW transfer (only for elements present)
    ifm_bytes = Shape4D.min(query.ifm_shape, query.config.ifm_block).elements()
    cycles_ifm_blk = arch.memory_latency[query.ifm_memory_area][BandwidthDirection.Read]
    cycles_ifm_blk = cycles_ifm_blk + (
        _estimate_memory_transfer_efficiency(
            arch,
            True,
            query.ifm_memory_area,
            query.ifm_format,
            query.ifm_bits,
            query.config.ifm_block,
            query.ifm_shape,
            ifm_bytes,
        )
        / arch.memory_bandwidths_per_cycle[query.ifm_memory_area]
    )
    # Output block HW transfer (only for elements present)
    ofm_bytes = Shape4D.min(query.ofm_shape, query.config.ofm_block).elements()
    cycles_ofm_blk = arch.memory_latency[query.ofm_memory_area][BandwidthDirection.Write]
    cycles_ofm_blk = cycles_ofm_blk + (
        _estimate_memory_transfer_efficiency(
            arch,
            False,
            query.ofm_memory_area,
            query.ofm_format,
            query.ofm_bits,
            query.config.ofm_block,
            query.ofm_shape,
            ofm_bytes,
        )
        / arch.memory_bandwidths_per_cycle[query.ofm_memory_area]
    )
    return cycles_ifm_blk, cycles_ofm_blk


def _estimate_output_cycles_per_element(arch, op_type: Op, faf_type: Op, query: PerformanceQuery):
    if query.npu_block_type == NpuBlockType.ElementWise and query.ifm_bits == 32:
        # Unary op else Binary op
        output_perf_index = 0 if query.ifm2_shape is not None else 1
    elif op_type == Op.Mul and query.ofm_bits == 32:
        output_perf_index = 2
    elif op_type == Op.Mul or (
        query.npu_block_type
        in (
            NpuBlockType.ConvolutionMxN,
            NpuBlockType.ConvolutionDepthWise,
            NpuBlockType.Pooling,
            NpuBlockType.ReduceSum,
            NpuBlockType.VectorProduct,
        )
        and query.config.acc_type == SHRAMElements.Acc40
    ):
        output_perf_index = 3
    elif op_type in (Op.Add, Op.Sub):
        if False:
            # Simple Add/Sub
            output_perf_index = 4
        else:
            # Advanced Add/Sub TODO: Add as perf selection as operator variant
            output_perf_index = 5
    elif op_type.is_maxpool_op():
        output_perf_index = 6
    else:
        output_perf_index = 7

    if faf_type in (Op.Sigmoid, Op.Tanh, Op.LUT):
        activation_perf_index = 0
    elif faf_type in (Op.Relu, Op.Relu6, Op.ReluN1To1):
        activation_perf_index = 1
    else:
        activation_perf_index = 2

    cycle_per_elem = max(
        arch.output_cycles_per_elem[output_perf_index], arch.activation_cycles_per_elem[activation_perf_index]
    )

    if op_type.is_elementwise_op():
        num_elems_blk = query.config.ofm_block.elements()
        ifm_blk_cycles, ofm_blk_cycles = _estimate_minimum_memory_cycles(arch, query)
        cycle_cmd = ifm_blk_cycles + ofm_blk_cycles
        cycle_cmd = (cycle_cmd + cycle_per_elem * num_elems_blk) / 4  # per DPU
        cycle_per_elem = max(cycle_per_elem, cycle_cmd / num_elems_blk)

    return cycle_per_elem


def _estimate_conv_cycles(arch, op_type: Op, faf_type: Op, query: PerformanceQuery):
    ifm_block = Shape4D.min(query.ifm_shape, query.config.ifm_block)
    ofm_block = Shape4D.min(query.ofm_shape, query.config.ofm_block)

    if (
        arch.config.ofm_ublock.height == 2
        and query.npu_block_type
        in (NpuBlockType.ConvolutionMxN, NpuBlockType.ConvolutionDepthWise, NpuBlockType.VectorProduct)
        and query.ofm_shape.height == 1
        # Optimisation only applies for even width tensors
        and query.ofm_shape.width % 2 == 0
        and query.kernel.height == 1
    ):
        ofm_ublock = Shape4D(1, 1, 4, arch.config.ofm_ublock.depth)
        ofm_block = ofm_block.with_height(1)
    else:
        ofm_ublock = Shape4D(arch.config.ofm_ublock.to_hwc())

    num_ublk_x = numeric_util.round_up_divide(ofm_block.width, ofm_ublock.width)
    num_ublk_y = numeric_util.round_up_divide(ofm_block.height, ofm_ublock.height)
    num_ublk_xy = num_ublk_x * num_ublk_y
    num_ublk_z = numeric_util.round_up_divide(ofm_block.depth, ofm_ublock.depth)
    use_acc_40bits = query.config.acc_type == SHRAMElements.Acc40

    sub_kernel_limits = arch.sub_kernel_limits[query.npu_block_type]
    n_sub_kernels_y = numeric_util.round_up_divide(query.kernel.height, sub_kernel_limits[0])
    n_sub_kernels_x = numeric_util.round_up_divide(query.kernel.width, sub_kernel_limits[1])
    sub_kernel_x = [
        min((query.kernel.width - i * sub_kernel_limits[1]), sub_kernel_limits[1]) for i in range(n_sub_kernels_x)
    ]
    sub_kernel_y = [
        min((query.kernel.height - i * sub_kernel_limits[0]), sub_kernel_limits[0]) for i in range(n_sub_kernels_y)
    ]
    sub_kernel_size = (x * y for y in sub_kernel_y for x in sub_kernel_x)

    cycles_dpu_blk = 0
    cycles_wb = 32 * ofm_ublock.depth // 8

    for num_kernel_elems in sub_kernel_size:
        if query.npu_block_type == NpuBlockType.Pooling:
            num_kernel_steps = 1
            cycles = max(4, num_kernel_elems) * num_ublk_xy * num_ublk_z
            if query.ifm_bits == 16 and arch.accelerator_config != Accelerator.Ethos_U55_32:
                cycles *= 2
        elif query.npu_block_type == NpuBlockType.ConvolutionDepthWise:
            cycles = 4 * num_ublk_xy
            if query.ifm_bits == 16:
                cycles *= 2
            num_kernel_steps = numeric_util.round_up_divide(num_kernel_elems, 4)
            cycles = max(cycles_wb, cycles) * num_kernel_steps * num_ublk_z
        elif (
            (query.npu_block_type == NpuBlockType.ConvolutionMxN and not query.config.is_partkernel)
            or query.npu_block_type == NpuBlockType.VectorProduct
            or query.npu_block_type == NpuBlockType.ReduceSum
        ):
            num_kernel_steps = num_kernel_elems
            cycles = max(cycles_wb, 4 * num_ublk_xy) * num_kernel_steps * num_ublk_z
        else:
            assert query.config.is_partkernel
            divider = 2 if query.ifm_bits == 16 else 4
            num_kernel_steps = numeric_util.round_up_divide(num_kernel_elems, divider)
            cycles = max(cycles_wb, 4 * num_ublk_xy) * (
                num_kernel_steps * numeric_util.round_up_divide(ifm_block.depth, 8) * num_ublk_z
            )

        delay_cycles = 0
        if arch.accelerator_config is Accelerator.Ethos_U55_32:
            delay = 7 if use_acc_40bits else 3
            if num_ublk_x == 1 and num_ublk_y == 1:
                if num_ublk_z == 1:
                    delay_cycles = delay * num_kernel_steps
                elif num_kernel_steps > 1:
                    delay_cycles = delay * (num_kernel_steps - 1) * num_ublk_z
            if (num_ublk_x == 1 or num_ublk_y == 1) and num_ublk_z > 1 and use_acc_40bits:
                delay_cycles += delay * num_ublk_z
        else:
            if use_acc_40bits and arch.accelerator_config in (Accelerator.Ethos_U55_64, Accelerator.Ethos_U55_128):
                delay = 3
            else:
                delay = 2

            if num_ublk_x == 1 and num_ublk_y == 1:
                if num_ublk_z == 1:
                    delay_cycles = delay * num_kernel_steps
                elif num_kernel_steps > 1:
                    delay_cycles = delay * (num_kernel_steps - 1) * num_ublk_z

        if query.npu_block_type == NpuBlockType.ConvolutionMxN and query.config.is_partkernel:
            delay_cycles *= numeric_util.round_up_divide(ifm_block.depth, 8)

        cycles_dpu_blk += cycles
        cycles_dpu_blk += delay_cycles

    if query.npu_block_type in (NpuBlockType.ConvolutionMxN, NpuBlockType.VectorProduct, NpuBlockType.ReduceSum):
        cycles_dpu_blk *= numeric_util.round_up_divide(query.ifm_shape.depth, ifm_block.depth)

    cycles_dpu_blk /= arch.ncores

    # Estimate output cycles
    num_ofm_blks = query.ofm_shape.div_round_up(ofm_block).elements()
    cycles_output_blk = round_up_to_int(
        _estimate_output_cycles_per_element(arch, op_type, faf_type, query) * ofm_block.elements()
    )

    # Scale and bias tensor
    if query.const_shape.depth > 0:
        cycles_bias_blk = (
            10 * ofm_block.depth * arch.memory_latency[query.const_memory_area][BandwidthDirection.Read] / 256
        )
        cycles_output_blk = max(cycles_output_blk, cycles_bias_blk)

    ifm_blk_cycles, ofm_blk_cycles = _estimate_minimum_memory_cycles(arch, query)
    cycles_cmd = ifm_blk_cycles + ofm_blk_cycles
    cycles_cmd = (cycles_cmd + cycles_output_blk + cycles_dpu_blk) / 4  # per DPU

    cycles_dpu_blk = max(cycles_dpu_blk, cycles_cmd)
    cycles_output_blk = max(cycles_output_blk, cycles_cmd)

    if cycles_dpu_blk > cycles_output_blk:
        total_cycles = cycles_dpu_blk * num_ofm_blks + cycles_output_blk
    else:
        total_cycles = cycles_output_blk * num_ofm_blks + cycles_dpu_blk

    return total_cycles


def measure_mem2mem_cycles(arch, from_mem_area, to_mem_area, to_transfer):
    from_cycles = to_transfer // arch.memory_bandwidths_per_cycle[from_mem_area]
    from_cycles += arch.memory_latency[from_mem_area][BandwidthDirection.Read]
    to_cycles = to_transfer // arch.memory_bandwidths_per_cycle[to_mem_area]
    return max(from_cycles, to_cycles)


def measure_cycle_cost(arch, op_type: Op, faf_type: Op, query: PerformanceQuery):
    cycles = CycleCost()

    # Convolution/Vector product cycle calculation
    if query.npu_block_type in (
        NpuBlockType.ConvolutionMxN,
        NpuBlockType.ConvolutionDepthWise,
        NpuBlockType.VectorProduct,
        NpuBlockType.Pooling,
        NpuBlockType.ReduceSum,
    ):
        # cycles.op_macs and cycles.op_cycles should both handle >32-bits
        if query.npu_block_type in (NpuBlockType.ConvolutionDepthWise, NpuBlockType.Pooling):
            cycles.op_macs = int(query.kernel.elements_wh()) * 1 * int(query.ofm_shape.elements())
        else:
            cycles.op_macs = (
                int(query.kernel.elements_wh()) * int(query.ifm_shape.depth) * int(query.ofm_shape.elements())
            )

        cycles.op_cycles = int(_estimate_conv_cycles(arch, op_type, faf_type, query))
    # Elementwise cycle calculation
    elif query.npu_block_type == NpuBlockType.ElementWise:
        cycles.op_macs = 0
        ofm_rounding = Shape4D(list(arch.storage_rounding_quantums[query.ofm_format]))
        cycles.op_cycles = round_up_to_int(
            _estimate_output_cycles_per_element(arch, op_type, faf_type, query)
            * Shape4D.round_up(query.ofm_shape, ofm_rounding).elements()
        )
    # DMA cycle calculation
    elif query.npu_block_type == NpuBlockType.Dma:
        # Return 0 since this is not an actual NPU op
        cycles.op_cycles = 0
    else:
        assert False

    return cycles


def measure_element_access(arch, query: PerformanceQuery):
    access = ElementAccess()

    ifm_block = Shape4D.min(query.ifm_shape, query.config.ifm_block)
    ofm_block = Shape4D.min(query.ofm_shape, query.config.ofm_block)
    ifm_rounding = Shape4D(list(arch.storage_rounding_quantums[query.ifm_format]))

    # Number of ofm blocks in the overall output shape
    ofm_blocks = query.ofm_shape.div_round_up(ofm_block)
    ofm_block_depth = ofm_block.depth
    if query.npu_block_type in (NpuBlockType.ConvolutionDepthWise, NpuBlockType.Pooling):
        ofm_blocks = ofm_blocks.with_depth(1)
        ofm_block_depth = query.ifm_shape.depth

    # Convolution & pooling
    if query.npu_block_type in (
        NpuBlockType.ConvolutionMxN,
        NpuBlockType.ConvolutionDepthWise,
        NpuBlockType.VectorProduct,
        NpuBlockType.Pooling,
        NpuBlockType.ReduceSum,
    ):
        # Number of sub kernels
        sub_kernel_limits = arch.sub_kernel_limits[query.npu_block_type]
        subkernels = numeric_util.round_up_divide(query.kernel.width, sub_kernel_limits[0])
        subkernels *= numeric_util.round_up_divide(query.kernel.height, sub_kernel_limits[1])

        ofm_block_count = ofm_blocks.elements()

        ifm_fetch = (
            Shape4D.round_up(ifm_block, ifm_rounding).elements_wh()
            * Shape4D.round_up(query.ifm_shape, ifm_rounding).depth
        )

        if query.npu_block_type in (NpuBlockType.ConvolutionDepthWise, NpuBlockType.Pooling):
            kernel_read = query.kernel.elements_wh() * 1  # force to no reread
        else:
            kernel_read = query.kernel.elements_wh() * query.ifm_shape.depth

        weight_fetch = kernel_read * ofm_block_depth * ofm_block_count

        access.ifm_read[0] = ifm_fetch * subkernels * ofm_block_count

        if query.npu_block_type not in (NpuBlockType.Pooling, NpuBlockType.ReduceSum):
            access.const_read[0] = weight_fetch
            access.const_read[1] = query.ofm_shape.depth  # Scales & biases
            access.weights_refetch = ofm_blocks.elements_wh()
    # Elementwise
    elif query.npu_block_type == NpuBlockType.ElementWise:
        if query.ifm_shape.elements() == 1:
            if query.ifm_bits > 8:
                # ifm is a non 8-bit scalar
                access.ifm_read[0] = Shape4D.round_up(query.ifm_shape, ifm_rounding).elements()
            if query.ifm2_shape:
                access.ifm_read[1] = Shape4D.round_up(query.ofm_shape, ifm_rounding).elements()
        else:
            access.ifm_read[0] = Shape4D.round_up(query.ofm_shape, ifm_rounding).elements()
            if query.ifm2_shape:
                if query.ifm2_shape.elements() > 1:
                    access.ifm_read[1] = Shape4D.round_up(query.ofm_shape, ifm_rounding).elements()
                elif query.ifm2_bits > 8:
                    # ifm2 is a non 8-bit scalar
                    access.ifm_read[1] = Shape4D.round_up(query.ifm2_shape, ifm_rounding).elements()
    # DMA
    elif query.npu_block_type == NpuBlockType.Dma:
        # Return empty access since this is not an actual NPU op
        return access
    # Unknown
    else:
        assert False

    ofm_rounding = Shape4D(list(arch.storage_rounding_quantums[query.ofm_format]))
    access.ofm_write = Shape4D.round_up(query.ofm_shape, ofm_rounding).elements()
    return access


def measure_performance_cost(
    arch, op_type: Op, faf_type: Op, query: PerformanceQuery, offset: Shape4D, sub_shape: Shape4D
):
    assert (query.ofm_bits > 0) and (query.ifm_bits > 0)
    assert query.ofm_shape.elements() != 0

    # Default to start if no offset provided
    if offset is None:
        offset = Shape4D(0, 0, 0, 0)

    # Default to entire area if no sub-shape provided
    if sub_shape is None:
        sub_shape = query.ofm_shape
    else:
        sub_shape = Shape4D.min(sub_shape, query.ofm_shape)

    sub_query = copy.deepcopy(query)
    sub_query.ofm_shape = query.ofm_shape.clip(offset, sub_shape)

    access = ElementAccess()
    cycles = CycleCost()

    cycle_tmp = measure_cycle_cost(arch, op_type, faf_type, sub_query)
    cycles += cycle_tmp
    access = measure_element_access(arch, sub_query)

    return access, cycles


def make_bandwidth_array():
    return np.zeros((MemArea.Size, TensorPurpose.Size, BandwidthDirection.Size))


def make_cycles_array():
    return np.zeros(PassCycles.Size)


def update_summary_cycles(arch, bws, cycles):
    cycles[PassCycles.SramAccess] = np.sum(bws[MemArea.Sram]) / arch.memory_bandwidths_per_cycle[MemArea.Sram]
    cycles[PassCycles.DramAccess] = np.sum(bws[MemArea.Dram]) / arch.memory_bandwidths_per_cycle[MemArea.Dram]
    cycles[PassCycles.OnChipFlashAccess] = (
        np.sum(bws[MemArea.OnChipFlash]) / arch.memory_bandwidths_per_cycle[MemArea.OnChipFlash]
    )
    cycles[PassCycles.OffChipFlashAccess] = (
        np.sum(bws[MemArea.OffChipFlash]) / arch.memory_bandwidths_per_cycle[MemArea.OffChipFlash]
    )

    cycles[PassCycles.Total] = np.max(cycles[: PassCycles.Total])
    return cycles


def estimate_full_op_performance(
    arch, schedule: Schedule, op: SchedulerOperation, prev_op: Optional[SchedulerOperation], block_config
):
    cycles_a = make_cycles_array()
    bws = make_bandwidth_array()
    scaled_bws = make_bandwidth_array()  # scaled bw with memory transfer efficiency
    macs = 0

    query = PerformanceQuery(op.op_type.npu_block_type)
    query.ifm_shape = op.ifm_read_shape
    query.ifm_format = op.ifm.format
    query.ifm_memory_area = op.ifm.connection.parent_tens.mem_area  # Mem Area is set directly on parent_tens
    query.ifm_bits = op.ifm.dtype.size_in_bits()
    query.ifm2_shape = op.ifm2_read_shape
    query.ifm2_format = op.ifm2 and op.ifm2.format
    query.ifm2_memory_area = op.ifm2 and op.ifm2.connection.parent_tens.mem_area
    query.ifm2_bits = op.ifm2 and op.ifm2.dtype.size_in_bits()
    query.ofm_shape = op.ofm_write_shape
    query.ofm_memory_area = op.ofm.connection.parent_tens.mem_area
    query.ofm_bits = op.ofm.dtype.size_in_bits()
    query.ofm_format = op.ofm.format
    query.kernel = op.kernel
    query.config = block_config

    cost = schedule.cost_map[op]
    prev_cost = schedule.cost_map[prev_op] if prev_op else None
    if op.parent_op.bias:
        query.const_shape = Shape4D(1, 1, 1, op.ofm.shape.depth)
        if cost.buffered_weight_tensors:
            query.const_memory_area = cost.buffered_weight_tensors[0].mem_area
        else:
            query.const_memory_area = cost.npu_weights_tensor.mem_area

    cycles = measure_cycle_cost(arch, op.op_type, op.parent_op.activation and op.parent_op.activation.op_type, query)
    cycles_a[PassCycles.Npu] = cycles.op_cycles
    macs = cycles.op_macs

    access = measure_element_access(arch, query)

    # How many NPU cycles are available under the previously executing
    # operator for performing buffered DMA transfers
    slack_cycles = prev_cost.slack_buffering_cycles if prev_cost else 0

    # LUT Transfer
    parent_op = op.parent_op
    dma_transfer_cycles = 0
    if parent_op.activation_lut:
        lut_tensor = [tens for tens in parent_op.inputs if tens.purpose == TensorPurpose.LUT][0]
        src_tensor = lut_tensor.src_tensor
        if src_tensor and lut_tensor.mem_area != src_tensor.mem_area:
            bw = src_tensor.storage_size()
            dma_transfer_cycles += measure_mem2mem_cycles(arch, src_tensor.mem_area, lut_tensor.mem_area, bw)

            bws[src_tensor.mem_area][lut_tensor.purpose][BandwidthDirection.Read] += bw
            # LUT read from SHRAM TODO remove?
            scaled_bws[lut_tensor.mem_area][lut_tensor.purpose][BandwidthDirection.Read] += bw

    # DMA Transfer
    if parent_op.type == Op.Memcpy:
        src_tensor = parent_op.ifm
        dst_tensor = parent_op.ofm
        if src_tensor.mem_area != dst_tensor.mem_area:
            bw = src_tensor.storage_size()
            dma_transfer_cycles += measure_mem2mem_cycles(arch, src_tensor.mem_area, dst_tensor.mem_area, bw)
            bws[src_tensor.mem_area][src_tensor.purpose][BandwidthDirection.Read] += bw
            bws[dst_tensor.mem_area][src_tensor.purpose][BandwidthDirection.Write] += bw

    if cost.npu_weights_tensor and cost.buffered_weight_tensors:
        # DMA Weight Transfer
        sz = 0
        # Get the size of the first DMA
        for core in range(0, arch.ncores):
            key = WeightKey(core, 0)
            if key in cost.npu_weights_tensor.encoded_ranges:
                weight_range = cost.npu_weights_tensor.encoded_ranges[key]
                sz += round_up(weight_range.total_bytes, 16)

        total_sz = len(cost.npu_weights_tensor.buffer)
        bws[cost.npu_weights_tensor.mem_area][TensorPurpose.Weights][BandwidthDirection.Read] += total_sz
        bws[cost.buffered_weight_tensors[0].mem_area][TensorPurpose.Weights][BandwidthDirection.Write] += total_sz

        ws_first_transfer_cycles = measure_mem2mem_cycles(
            arch, cost.npu_weights_tensor.mem_area, cost.buffered_weight_tensors[0].mem_area, sz
        )

        # Add cycles for Weight + Scale Transfer
        if cost.buffered_weight_tensors[0].sub_purpose == TensorSubPurpose.DoubleBuffer:
            # Double buffer - weights can be fetched in parallel
            cycles_a[PassCycles.Npu] = max(
                cost.full_weight_transfer_cycles - slack_cycles + cost.slack_buffering_cycles,
                cycles.op_cycles + max(ws_first_transfer_cycles - slack_cycles, 0),
            )
        else:
            # Standard buffer - weights can not be fetched in parallel so weight transfer
            # must be included in the result
            cycles_a[PassCycles.Npu] = (
                cycles.op_cycles + cost.full_weight_transfer_cycles - min(ws_first_transfer_cycles, slack_cycles)
            )

        # Add cycles for LUT + mempcy op Transfer
        cycles_a[PassCycles.Npu] += dma_transfer_cycles
    else:
        # Add cycles for LUT + mempcy op Transfer
        cycles_a[PassCycles.Npu] += max(dma_transfer_cycles - slack_cycles, 0)

    # OFM write
    ofm = op.ofm.connection.parent_tens
    bw = access.ofm_write * ofm.element_size()
    bws[query.ofm_memory_area][ofm.purpose][BandwidthDirection.Write] += bw
    scaled_bws[query.ofm_memory_area][ofm.purpose][BandwidthDirection.Write] += _estimate_memory_transfer_efficiency(
        arch,
        False,
        query.ofm_memory_area,
        query.ofm_format,
        query.ofm_bits,
        query.config.ofm_block,
        query.ofm_shape,
        bw,
    )

    # IFM read
    ifm = op.ifm.connection.parent_tens
    bw = access.ifm_read[0] * ifm.element_size()
    bws[query.ifm_memory_area][ifm.purpose][BandwidthDirection.Read] += bw
    scaled_bws[query.ifm_memory_area][ifm.purpose][BandwidthDirection.Read] += _estimate_memory_transfer_efficiency(
        arch, True, query.ifm_memory_area, query.ifm_format, query.ifm_bits, query.config.ifm_block, query.ifm_shape, bw
    )

    if query.ifm2_shape:
        ifm2 = op.ifm2.connection.parent_tens
        bw = access.ifm_read[1] * ifm2.element_size()
        bws[ifm2.mem_area][ifm2.purpose][BandwidthDirection.Read] += bw
        scaled_bws[ifm2.mem_area][ifm2.purpose][BandwidthDirection.Read] += _estimate_memory_transfer_efficiency(
            arch,
            True,
            query.ifm2_memory_area,
            query.ifm2_format,
            query.ifm2_bits,
            query.config.ifm_block,
            query.ifm2_shape,
            bw,
        )

    # Weight read
    if access.const_read[0] > 0:
        # alignment not accounted for in bandwidth_compression_scale_approx
        encoded_size_approx = (
            cost.npu_weights_tensor.elements() - access.const_read[1] * op.parent_op.bias.element_size()
        )
        orig_weight_size = parent_op.weights.elements()
        bandwidth_compression_scale_approx = encoded_size_approx / orig_weight_size
        bw = access.const_read[0] * bandwidth_compression_scale_approx
        bws[query.const_memory_area][TensorPurpose.Weights][BandwidthDirection.Read] += bw

        if not cost.buffered_weight_tensors:
            scaled_bws[query.const_memory_area][TensorPurpose.Weights][BandwidthDirection.Read] += bw

    if access.const_read[1] > 0:
        # Scales & biases
        bw = access.const_read[1] * op.parent_op.bias.element_size()
        bws[query.const_memory_area][TensorPurpose.FSBias][BandwidthDirection.Read] += bw

        if not cost.buffered_weight_tensors:
            scaled_bws[query.const_memory_area][TensorPurpose.FSBias][BandwidthDirection.Read] += bw

    update_summary_cycles(arch, scaled_bws, cycles_a)

    return bws, macs, cycles_a


def print_performance(
    nng: Graph,
    arch: ArchitectureFeatures,
    network_type: NetworkType,
    bws: dict,
    macs: dict,
    cycles: dict,
    mem_usage: dict,
    output_basename: str,
):
    def _percentage(part, whole):
        # desired behaviour is for division by zero to return 100%
        if whole == 0:
            return 100.0
        else:
            return part / whole * 100.0

    if network_type == NetworkType.TFLite:
        nng_optype_to_input_op_type = tflite_optype_to_builtintype
    else:
        nng_optype_to_input_op_type = tosa_optype_to_tosa_op_type

    suid_inv_map = {v: k for k, v in DebugDatabase._sourceUID.items()}

    # the header is a list (one entry per column) of tuples (column name, alignment, width, precision)
    header = [
        (f"{network_type.name}_operator", "<", 20, -1),
        ("NNG Operator", "<", 20, -1),
        ("SRAM Usage", ">", 10, 0.0),
        ("Peak%", ">", 6, 0.2),
        ("Op Cycles", ">", 10, 0.0),
        ("Network%", ">", 8, 0.2),
        ("NPU", ">", 10, 0.0),
        ("SRAM AC", ">", 10, 0.0),
        ("DRAM AC", ">", 10, 0.0),
        ("OnFlash AC", ">", 10, 0.0),
        ("OffFlash AC", ">", 11, 0.0),
        ("MAC Count", ">", 10, 0.0),
        ("Network%", ">", 8, 0.2),
        ("Util%", ">", 6, 0.2),
        ("Name", "<", 20, -1),
    ]

    # open the csv
    csv_file = open(output_basename + "_per-layer.csv", "w", encoding="UTF8")
    writer = csv.writer(csv_file)

    for sg in nng.subgraphs:

        if sg.placement != PassPlacement.Npu:
            continue

        sg_seperator_text = f"\n{str('#') * 80}\nPerformance for NPU Subgraph {sg.name}"

        # the data is a list (one entry per op) of lists (matching the header columns)
        data = []
        for sched_op in sg.sched_ops:
            # get source op name
            sched_op_src_uid = DebugDatabase._optimisedUID[sched_op.parent_op][1]
            if sched_op_src_uid == DebugDatabase.NULLREF:
                src_op_type = None
            else:
                src_op_type = suid_inv_map[sched_op_src_uid].original_type

            src_op_name = nng_optype_to_input_op_type(src_op_type)

            max_macs = cycles[sched_op][PassCycles.Total] * arch.num_macs_per_cycle * arch.ncores
            peak_sram = (
                _percentage(mem_usage[sched_op], nng.memory_used[MemArea.Sram])
                if MemArea.Sram in nng.memory_used
                else 0
            )

            data.append(
                [
                    src_op_name,
                    sched_op.op_type,
                    mem_usage[sched_op],
                    peak_sram,
                    cycles[sched_op][PassCycles.Total],
                    _percentage(cycles[sched_op][PassCycles.Total], nng.cycles[PassCycles.Total]),
                    cycles[sched_op][PassCycles.Npu],
                    cycles[sched_op][PassCycles.SramAccess],
                    cycles[sched_op][PassCycles.DramAccess],
                    cycles[sched_op][PassCycles.OnChipFlashAccess],
                    cycles[sched_op][PassCycles.OffChipFlashAccess],
                    macs[sched_op],
                    _percentage(macs[sched_op], nng.macs),
                    _percentage(macs[sched_op], max_macs),
                    sched_op.name,
                ]
            )

        # print to console
        print(sg_seperator_text)
        line = ""
        line2 = ""
        for col_name, align, width, _ in header:
            line_data = f"{col_name:{align}{width}}"
            line += line_data + " "
            line2 += "-" * len(line_data) + " "
        print(line)
        print(line2)

        for op_data in data:
            line = ""
            for idx, item in enumerate(op_data):
                _, align, width, precision = header[idx]
                if precision == -1:
                    w = str(width)
                else:
                    w = str(width + precision) + "f"
                line += f"{item:{align}{w}}" + " "
            print(line)

        # print to csv
        writer.writerow(col_name for col_name, _, _, _ in header)
        for op_data in data:
            writer.writerow(op_data)

    # close the csv
    csv_file.close()


def calc_new_performance_for_network(
    nng: Graph,
    arch,
    network_type: NetworkType,
    verbose_performance: bool,
    output_basename: str = "output/unnamed_network",
):
    total_bws = make_bandwidth_array()
    total_macs = 0
    total_cycles = np.zeros(PassCycles.Size)
    total_weight_size = 0
    total_encoded_weight_size = 0

    # Store unique instances of original/encoded weight tensor uuids to prevent double counting of weights
    original_weight_uuids: Set[UUID] = set()
    encoded_npu_weight_uuids: Set[UUID] = set()

    bws = {}
    macs = {}
    cycles = {}
    mem_usage = {}

    for sg in nng.subgraphs:
        prev_op = None
        for sched_op in sg.sched_ops:
            op_info: SchedulerOpInfo = sg.schedule.cost_map[sched_op]
            bws[sched_op], macs[sched_op], cycles[sched_op] = estimate_full_op_performance(
                arch, sg.schedule, sched_op, prev_op, op_info.block_config
            )

            # get op sram usage
            mem_usage[sched_op] = (
                sg.schedule.memory_snapshot[op_info.time_index]
                if op_info.time_index < len(sg.schedule.memory_snapshot)
                else 0
            )

            # Tensors for calculating weight sizes
            original_weight = sched_op.parent_op.weights
            encoded_npu_weight = op_info.npu_weights_tensor

            # Save UUIDs of original_weight so only unique instances of tensors are used to calculate weights
            if original_weight and (original_weight.equivalence_id not in original_weight_uuids):

                original_weight_uuids.add(original_weight.equivalence_id)
                total_weight_size += original_weight.values.itemsize * original_weight.values.size

            # Save UUIDs of encoded_npu_weight so only unique instances of tensors are used to calculate weights
            if encoded_npu_weight and (encoded_npu_weight.equivalence_id not in encoded_npu_weight_uuids):

                encoded_npu_weight_uuids.add(encoded_npu_weight.equivalence_id)
                total_encoded_weight_size += len(encoded_npu_weight.buffer)

            total_bws += bws[sched_op]
            total_macs += macs[sched_op]
            total_cycles += cycles[sched_op]
            prev_op = sched_op

    nng.bandwidths = total_bws
    nng.macs = total_macs
    nng.cycles = total_cycles
    nng.total_original_weights = total_weight_size
    nng.total_npu_encoded_weights = total_encoded_weight_size

    if verbose_performance:
        print_performance(nng, arch, network_type, bws, macs, cycles, mem_usage, output_basename)
