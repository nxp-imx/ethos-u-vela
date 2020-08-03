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
# NPU performance estimation functions to estimate performance of a Pass and CascadedPass. Uses a model that takes the
# maximum of the 'cycles required for bandwidth' and 'cycles required for computing'.
#
# Called during scheduling to evaluate different proposals, as well as post-scheduling to provide a final performance
# estimate.
import enum

import numpy as np

from . import numeric_util
from .architecture_features import Block
from .nn_graph import PassPlacement
from .nn_graph import SchedulerRewrite
from .operation import NpuBlockType
from .register_command_stream_generator import get_op_kernel
from .tensor import MemArea
from .tensor import shape_num_elements
from .tensor import TensorBlockTraversal
from .tensor import TensorPurpose


def rolling_buffer_dims_from_passes(arch, ps1, block_config_ps1, ps2, block_config_ps2):
    ofm_block = Block(block_config_ps2[-3], block_config_ps2[-4], block_config_ps2[-1])
    kernel = get_op_kernel(ps2)

    if ps2.npu_block_type in set((NpuBlockType.ConvolutionMxN, NpuBlockType.VectorProduct)):
        op = ps2.primary_op
        ifm_idx, _, _, _, _ = op.get_ifm_ifm2_weight_bias_ofm_indices()
        ifm_block_depth = arch.calc_ifm_block_depth(
            op.inputs[ifm_idx].shape[-1], op.inputs[ifm_idx].dtype.size_in_bits()
        )
    else:
        ifm_block_depth = block_config_ps2[-1]

    ifm_block = arch.get_ifm_block_size(ifm_block_depth, ofm_block, kernel, arch.ofm_block_max)

    # The performed height calculation is for worst case
    height = numeric_util.round_up(ifm_block.height + block_config_ps1[0], block_config_ps1[0])
    width = ifm_block.width
    return [height, width]


class PassCycles(enum.IntEnum):
    Dpu = 0
    ElementWise = 1
    Cpu = 2
    SramAccess = 3
    TotalPerPass = 4
    DramAccess = 5
    OnChipFlashAccess = 6
    OffChipFlashAccess = 7
    Total = 8
    Size = 9

    def display_name(self):
        return (
            "DPU",
            "Element wise",
            "CPU",
            "SRAM Access",
            "Total per Pass",
            "DRAM Access",
            "On-chip Flash Access",
            "Off-chip Flash Access",
            "Total",
            "Size",
        )[self.value]

    def identifier_name(self):
        return (
            "dpu",
            "element_wise",
            "cpu",
            "sram_access",
            "total_per_pass",
            "dram_access",
            "on_chip_flash_access",
            "off_chip_flash_access",
            "total",
            "size",
        )[self.value]

    @staticmethod
    def all():
        return (
            PassCycles.Dpu,
            PassCycles.ElementWise,
            PassCycles.Cpu,
            PassCycles.SramAccess,
            PassCycles.DramAccess,
            PassCycles.OnChipFlashAccess,
            PassCycles.OffChipFlashAccess,
            PassCycles.Total,
        )


class MacCount(enum.IntEnum):
    NeuralNetworkMacs = 0
    HardwareMacs = 1
    Size = 2

    def display_name(self):
        return ("Neural Network Macs", "Hardware Macs", "Size")[self.value]

    def identifier_name(self):
        return ("nn_macs", "hardware_macs", "size")[self.value]

    @staticmethod
    def all():
        return (MacCount.NeuralNetworkMacs, MacCount.HardwareMacs)


class BandwidthDirection(enum.IntEnum):
    Read = 0
    Write = 1
    Size = 2

    def display_name(self):
        return self.name

    def identifier_name(self):
        return self.name.lower()

    @staticmethod
    def all():
        return (BandwidthDirection.Read, BandwidthDirection.Write)


def make_bandwidth_array():
    return np.zeros((MemArea.Size, TensorPurpose.Size, BandwidthDirection.Size))


def make_macs_array():
    return np.zeros(MacCount.Size, np.int)


def make_cycles_array():
    return np.zeros(PassCycles.Size)


def make_metrics_arrays():
    return (make_bandwidth_array(), make_macs_array(), make_cycles_array())


def get_n_blocks_and_area(
    ifm_brick_size, ifm_height_width, orig_skirt, clamped_skirt, block_config, min_block_size, strides
):

    ifm_block_config = (block_config[0] * strides[1], block_config[1] * strides[2])

    n_normal_blocks = []
    remainder_size = []
    for i in range(2):
        non_skirt_dim = ifm_height_width[i] - orig_skirt[i] - orig_skirt[2 + i]
        n_blocks = non_skirt_dim // ifm_block_config[i]
        n_normal_blocks.append(n_blocks)
        remainder_dim = numeric_util.round_up(
            ((non_skirt_dim - n_blocks * ifm_block_config[i] - 1) // strides[i + 1]) + 1, min_block_size[i]
        )
        remainder_size.append(remainder_dim)

    # this will actually calculate reads into the edge padding.

    # there are four cases in total, handling the edges that will not fill a complete block.

    # 0000000001
    # 0000000001
    # 0000000001
    # 0000000001
    # 0000000001
    # 0000000001
    # 2222222223
    total_blocks = 0
    total_area = 0

    block_setup = (
        (n_normal_blocks[0] * n_normal_blocks[1], block_config),
        (1 * n_normal_blocks[1], (remainder_size[0], block_config[1])),
        (n_normal_blocks[0] * 1, (block_config[0], remainder_size[1])),
        (1 * 1, remainder_size),
    )

    for n_blocks, block_size in block_setup:
        if block_size[0] == 0 or block_size[1] == 0:
            continue
        read_dims = [0, 0]
        for i in range(2):
            read_dims[i] = (
                numeric_util.round_up(clamped_skirt[i], ifm_brick_size[i + 1])
                + block_size[i] * strides[i + 1]
                + numeric_util.round_up(clamped_skirt[2 + i], ifm_brick_size[i + 1])
            )
        assert n_blocks >= 0
        total_blocks += n_blocks
        total_area += n_blocks * read_dims[0] * read_dims[1]
    assert total_blocks >= 1
    return total_blocks, total_area, block_setup


def performance_metrics_for_pass(arch, ps, block_config=None, rewrite_list=[], force_outputs_to_fast_storage=False):
    if block_config is None:
        block_config = ps.block_config
    bws = make_bandwidth_array()
    macs = make_macs_array()
    cycles = make_cycles_array()
    blocks = 0
    ifm_read_multiple = 1
    weight_read_multiple = 0

    if ps.placement in set((PassPlacement.MemoryOnly, PassPlacement.StartupInit)):
        return bws, macs, cycles, blocks, ifm_read_multiple, weight_read_multiple  # nothing real happening in this pass

    min_block_size = arch.min_block_sizes[ps.npu_block_type]

    skirt = (0, 0, 0, 0)
    explicit_padding = (0, 0, 0, 0)
    primary_op = ps.primary_op
    replacement_read_bws = {}
    if ps.placement == PassPlacement.Cpu:
        cycles[PassCycles.Cpu] = arch.cpu_cycle_estimate(ps.ops[0])
    elif primary_op:
        skirt = primary_op.attrs.get("skirt", skirt)
        explicit_padding = primary_op.attrs.get("explicit_padding", explicit_padding)
        assert primary_op.attrs["npu_block_type"] == ps.npu_block_type
        npu_block_type = primary_op.attrs["npu_block_type"]

        ifm_tensor, _, weight_tensor, ofm_tensor = ps.get_primary_op_ifm_ifm2_weights_ofm()

        if npu_block_type in set(
            (NpuBlockType.ConvolutionMxN, NpuBlockType.ConvolutionDepthWise, NpuBlockType.Pooling)
        ):
            # extent the ifm to full dimension
            ifm_tensor_brick_size = tuple(numeric_util.full_shape(4, list(ifm_tensor.brick_size), 1))
            ifm_tensor_shape = numeric_util.full_shape(4, ifm_tensor.shape, 1)
            ifm_tensor_bandwidth_shape = numeric_util.full_shape(4, ifm_tensor.bandwidth_shape, 1)

            batch_size = ifm_tensor.shape[0]
            ifm_depth = ifm_tensor_bandwidth_shape[3]

            # add in padding
            ifm_tensor_shape[1] += explicit_padding[0] + explicit_padding[2]  # height += top and bottom
            ifm_tensor_shape[2] += explicit_padding[1] + explicit_padding[3]  # width  += left and right

            strides = primary_op.attrs["strides"]
            if npu_block_type != NpuBlockType.Pooling:
                weight_tensor_shape = weight_tensor.shape
                weight_tensor_bandwidth_shape = weight_tensor.bandwidth_shape
                weight_tensor_element_size = weight_tensor.element_size()
                weight_tensor_bandwidth_compression_scale = weight_tensor.bandwidth_compression_scale
                nn_ops = (
                    int(ofm_tensor.shape[0])
                    * int(ofm_tensor.shape[1])
                    * int(ofm_tensor.shape[2])
                    * int(weight_tensor_shape[0])
                    * int(weight_tensor_shape[1])
                    * int(weight_tensor_shape[2])
                    * int(weight_tensor_shape[3])
                )
            else:
                weight_tensor_shape = [
                    primary_op.attrs["ksize"][1],
                    primary_op.attrs["ksize"][2],
                    1,
                    ifm_tensor_shape[3],
                ]
                weight_tensor_bandwidth_shape = weight_tensor_shape
                weight_tensor_element_size = 0
                weight_tensor_bandwidth_compression_scale = 0.0
                nn_ops = 0  # pooling doesn't count as NN ops

            kernel_dims = weight_tensor_shape[:2]

            sub_kernel_limits = arch.sub_kernel_limits[npu_block_type]
            # count the sub kernels; the IFM block needs to be refetched for each of them
            n_sub_kernels_y = numeric_util.round_up_divide(kernel_dims[0], sub_kernel_limits[0])
            n_sub_kernels_x = numeric_util.round_up_divide(kernel_dims[1], sub_kernel_limits[1])
            n_sub_kernels = n_sub_kernels_y * n_sub_kernels_x

            clamped_skirt = list(skirt)
            clamped_skirt[2] = min(clamped_skirt[2], sub_kernel_limits[0] - 1 - clamped_skirt[0])
            clamped_skirt[3] = min(clamped_skirt[3], sub_kernel_limits[1] - 1 - clamped_skirt[1])
            n_blocks, area, block_setup = get_n_blocks_and_area(
                ifm_tensor_brick_size,
                ifm_tensor_shape[1:3],
                skirt,
                clamped_skirt,
                block_config,
                min_block_size,
                strides,
            )

            blocks = n_blocks * numeric_util.round_up_divide(weight_tensor_shape[3], block_config[3])

            n_weight_stages = numeric_util.round_up_divide(weight_tensor_bandwidth_shape[3], block_config[3])
            if npu_block_type == NpuBlockType.ConvolutionDepthWise or npu_block_type == NpuBlockType.Pooling:
                n_weight_stages = 1  # force to no reread

            ifm_tensor_bw = (
                n_sub_kernels
                * batch_size
                * area
                * ifm_depth
                * n_weight_stages
                * ifm_tensor.element_size()
                * ifm_tensor.bandwidth_compression_scale
            )
            replacement_read_bws[ifm_tensor] = ifm_tensor_bw
            ifm_read_multiple = n_weight_stages

            replacement_read_bws[weight_tensor] = (
                batch_size
                * shape_num_elements(weight_tensor_bandwidth_shape)
                * weight_tensor_element_size
                * weight_tensor_bandwidth_compression_scale
                * n_blocks
            )  # read once per block and batch
            weight_read_multiple = n_blocks

            n_kernel_xy = kernel_dims[0] * kernel_dims[1]
            n_input_channels_at_a_time = block_config[2]

            if npu_block_type == NpuBlockType.Pooling or weight_tensor.block_traversal in set(
                (TensorBlockTraversal.PartKernelFirst, TensorBlockTraversal.DepthWise)
            ):
                n_input_channels_at_a_time = numeric_util.round_up_divide(n_input_channels_at_a_time, 4)
                n_kernel_xy = max(
                    n_kernel_xy, 4
                )  # need at least 4, as this is the minimum duty cycle for secondary accumulator writes
                if weight_tensor is not None:
                    n_kernel_xy = numeric_util.round_up(n_kernel_xy, 4)  # weights need to be read in blocks of 4

            num_mac_ops = 0
            for n_blocks_for_size, block_size in block_setup:
                num_mac_ops += (
                    batch_size
                    * n_blocks_for_size
                    * block_size[0]
                    * block_size[1]
                    * numeric_util.round_up(weight_tensor_shape[2], n_input_channels_at_a_time)
                    * numeric_util.round_up(weight_tensor_shape[3], block_config[3])
                    * n_kernel_xy
                )

            if npu_block_type == NpuBlockType.Pooling:
                # TODO: improve pooling estimation
                cycles[PassCycles.Dpu] = num_mac_ops / arch.num_macs_per_cycle / 2
            else:
                cycles[PassCycles.Dpu] = num_mac_ops / arch.num_macs_per_cycle
            macs[MacCount.NeuralNetworkMacs] += nn_ops
            macs[MacCount.HardwareMacs] += num_mac_ops

        elif npu_block_type == NpuBlockType.VectorProduct:
            nn_macs = (
                ifm_tensor.shape[0]
                * numeric_util.round_up(weight_tensor.shape[-2], block_config[2])
                * numeric_util.round_up(weight_tensor.shape[-1], block_config[3])
            )
            num_mac_ops = nn_macs

            cycles[PassCycles.Dpu] = num_mac_ops / arch.num_macs_per_cycle
            macs[MacCount.NeuralNetworkMacs] += nn_macs
            macs[MacCount.HardwareMacs] += num_mac_ops

            blocks = 1 * numeric_util.round_up_divide(weight_tensor.shape[-1], block_config[3])

            non_zero_fraction = 1.0
            if ifm_tensor.values is not None:
                nz_vector = np.amax(ifm_tensor.values != 0, axis=0)  # max across batch axis
                non_zero_fraction = np.average(nz_vector)

            replacement_read_bws[ifm_tensor] = ifm_tensor.bandwidth()
            replacement_read_bws[weight_tensor] = weight_tensor.bandwidth() * non_zero_fraction
            ifm_read_multiple = 1
            weight_read_multiple = non_zero_fraction
    else:
        if ps.placement == PassPlacement.Npu and len(ps.outputs):
            # Assume element-wise operation going through the element pipelines.
            # Work out how many elements we have and calculate performance.
            out = ps.outputs[0]
            elms = out.elements()

            cycles[PassCycles.ElementWise] = numeric_util.round_up_divide(elms, arch.num_elem_wise_units)

    # apply the desired rewrites
    for rewrite_op, tens, _, _, _, ps_to_rewrite in rewrite_list:
        if ps != ps_to_rewrite:
            continue
        if rewrite_op == SchedulerRewrite.Nop:
            pass  # these are fine, no bandwidth changes
        elif rewrite_op in (SchedulerRewrite.ChangeTensorSubPurpose,):
            bws[arch.fast_storage_mem_area][tens.purpose][BandwidthDirection.Read] += replacement_read_bws[tens]
            replacement_read_bws[tens] = 0

    for tens in ps.outputs:
        if force_outputs_to_fast_storage:
            bws[arch.fast_storage_mem_area][tens.purpose][BandwidthDirection.Write] += tens.bandwidth()
        else:
            bws[tens.mem_area][tens.purpose][BandwidthDirection.Write] += tens.bandwidth()

    for tens in ps.intermediates:
        bws[tens.mem_area][tens.purpose][BandwidthDirection.Write] += tens.bandwidth()

        if tens in replacement_read_bws:
            bw = replacement_read_bws[tens]
        else:
            bw = tens.bandwidth()

        bws[tens.mem_area][tens.purpose][BandwidthDirection.Read] += bw

    for tens in ps.inputs:
        if tens in replacement_read_bws:
            bw = replacement_read_bws[tens]
        else:
            bw = tens.bandwidth()

        bws[tens.mem_area][tens.purpose][BandwidthDirection.Read] += bw

    cycles[PassCycles.SramAccess] = np.sum(bws[MemArea.Sram]) / arch.memory_bandwidths_per_cycle[MemArea.Sram]
    cycles[PassCycles.TotalPerPass] = np.max(cycles[: PassCycles.TotalPerPass])

    # quick build access counts for only current pass, even though these aren't the final numbers
    update_summary_cycles(arch, bws, macs, cycles)

    return bws, macs, cycles, blocks, ifm_read_multiple, weight_read_multiple


def update_summary_cycles(arch, bws, macs, cycles):
    cycles[PassCycles.DramAccess] = np.sum(bws[MemArea.Dram]) / arch.memory_bandwidths_per_cycle[MemArea.Dram]
    cycles[PassCycles.OnChipFlashAccess] = (
        np.sum(bws[MemArea.OnChipFlash]) / arch.memory_bandwidths_per_cycle[MemArea.OnChipFlash]
    )
    cycles[PassCycles.OffChipFlashAccess] = (
        np.sum(bws[MemArea.OffChipFlash]) / arch.memory_bandwidths_per_cycle[MemArea.OffChipFlash]
    )

    cycles[PassCycles.Total] = np.max(cycles[: PassCycles.Total])
    return cycles


def collate_stats_for_cascaded_pass(arch, bws, macs, cycles):
    return bws, macs, cycles


def performance_for_cascaded_pass(arch, cps):
    total_bws = make_bandwidth_array()
    total_macs = make_macs_array()
    total_cycles = make_cycles_array()

    for ps in cps.passes:
        bws, macs, cycles, blocks, _, _ = performance_metrics_for_pass(arch, ps)
        ps.bandwidths = bws
        ps.macs = macs
        ps.cycles = cycles
        ps.n_blocks = blocks
        total_bws += bws
        total_macs += macs
        total_cycles += cycles

    bws, macs, cycles = collate_stats_for_cascaded_pass(arch, total_bws, total_macs, total_cycles)
    cps.bandwidths = bws
    cps.macs = macs
    cps.cycles = cycles
    return bws, macs, cycles


def calc_performance_for_network(nng, arch):
    total_bws = make_bandwidth_array()
    total_macs = np.zeros(MacCount.Size)
    total_cycles = np.zeros(PassCycles.Size)

    for sg in nng.subgraphs:
        for cps in sg.cascaded_passes:
            bws, macs, cycles = performance_for_cascaded_pass(arch, cps)
            total_bws += bws
            total_macs += macs
            total_cycles += cycles

    nng.bandwidths = total_bws
    nng.macs = total_macs
    nng.cycles = total_cycles
