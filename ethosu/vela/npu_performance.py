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
# NPU performance estimation functions to estimate performance of a Pass and CascadedPass. Uses a model that takes the
# maximum of the 'cycles required for bandwidth' and 'cycles required for computing'.
#
# Called during scheduling to evaluate different proposals, as well as post-scheduling to provide a final performance
# estimate.
from enum import auto
from enum import IntEnum

import numpy as np

from . import numeric_util
from .architecture_features import Accelerator
from .architecture_features import Block
from .data_type import DataType
from .nn_graph import PassPlacement
from .nn_graph import SchedulerRewrite
from .operation import NpuBlockType
from .operation import Op
from .shared_buffer_allocation import is_acc_40bits_used
from .tensor import BandwidthDirection
from .tensor import MemArea
from .tensor import shape_num_elements
from .tensor import Tensor
from .tensor import TensorBlockTraversal
from .tensor import TensorFormat
from .tensor import TensorPurpose


def rolling_buffer_dims_from_passes(arch, ps1, block_config_ps1, ps2, block_config_ps2):
    ofm_block = Block(block_config_ps2[-3], block_config_ps2[-4], block_config_ps2[-1])
    kernel = ps2.primary_op.kernel

    if ps2.npu_block_type in (NpuBlockType.ConvolutionMxN, NpuBlockType.VectorProduct):
        op = ps2.primary_op
        ifm_block_depth = arch.calc_ifm_block_depth(op.ifm_shapes[0].depth, op.ifm.dtype.size_in_bits())
    else:
        ifm_block_depth = block_config_ps2[-1]

    ifm_block = arch.get_ifm_block_size(ifm_block_depth, ofm_block, kernel, arch.ofm_block_max)

    # The performed height calculation is for worst case
    height = numeric_util.round_up(ifm_block.height + block_config_ps1[0], block_config_ps1[0])
    width = ifm_block.width
    return [height, width]


class PassCycles(IntEnum):
    Npu = 0
    SramAccess = auto()
    DramAccess = auto()
    OnChipFlashAccess = auto()
    OffChipFlashAccess = auto()
    Total = auto()
    Size = auto()

    def display_name(self):
        return ("NPU", "SRAM Access", "DRAM Access", "On-chip Flash Access", "Off-chip Flash Access", "Total", "Size",)[
            self.value
        ]

    def identifier_name(self):
        return ("npu", "sram_access", "dram_access", "on_chip_flash_access", "off_chip_flash_access", "total", "size",)[
            self.value
        ]

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


def make_bandwidth_array():
    return np.zeros((MemArea.Size, TensorPurpose.Size, BandwidthDirection.Size))


def make_cycles_array():
    return np.zeros(PassCycles.Size)


def make_metrics_arrays():
    return (make_bandwidth_array(), 0, make_cycles_array())


def get_ifm_block_depth(npu_block_type, ifm_depth, ifm_elemwidth, block_traversal, ofm_blk_depth):
    ifm_blk_depth = ofm_blk_depth

    if npu_block_type in (NpuBlockType.ConvolutionMxN, NpuBlockType.VectorProduct, NpuBlockType.ReduceSum):
        if ifm_elemwidth == 16 or block_traversal == TensorBlockTraversal.PartKernelFirst:
            ifm_blk_depth = 16
        elif ifm_elemwidth == 8:
            ifm_blk_depth = 32
        else:
            ifm_blk_depth = 8

    return min(ifm_depth, ifm_blk_depth)


def get_minimal_cmd_cycles(
    arch, ifm_tensor, ofm_tensor, ifm_blk: Block, ofm_blk: Block, output_cycles, ifm_shape4D, ofm_shape4D, dpu_cycles=0
):
    ifm_tens_blk = Tensor((1, ifm_blk.height, ifm_blk.width, ifm_blk.depth), ifm_tensor.dtype, "ifm_blk")
    ofm_tens_blk = Tensor((1, ofm_blk.height, ofm_blk.width, ofm_blk.depth), ofm_tensor.dtype, "ofm_blk")
    cycles_ifm_blk = (
        estimate_memory_transfer_efficiency(
            arch, ifm_tensor.mem_area, BandwidthDirection.Read, ifm_tens_blk, ifm_blk, shape4D=ifm_shape4D
        )
        / arch.memory_bandwidths_per_cycle[ifm_tensor.mem_area]
    )
    cycles_ofm_blk = (
        estimate_memory_transfer_efficiency(
            arch, ofm_tensor.mem_area, BandwidthDirection.Write, ofm_tens_blk, ofm_blk, shape4D=ofm_shape4D
        )
        / arch.memory_bandwidths_per_cycle[ofm_tensor.mem_area]
    )
    return (
        arch.memory_latency[ifm_tensor.mem_area][BandwidthDirection.Read]
        + cycles_ifm_blk
        + dpu_cycles
        + output_cycles
        + arch.memory_latency[ofm_tensor.mem_area][BandwidthDirection.Write]
        + cycles_ofm_blk
    ) / 4


def estimate_output_cycles(
    arch,
    npu_block_type,
    primary_op,
    num_elems,
    ifm_tensor,
    ofm_tensor,
    use_acc_40bits=False,
    ifm2_tensor=None,
    block_config: Block = None,
):
    faf = None if primary_op.activation is None else primary_op.activation.op_type
    if npu_block_type == NpuBlockType.ElementWise and ifm_tensor.dtype == DataType.int32:
        if ifm2_tensor is None:
            # Unary op
            output_perf_index = 0
        else:
            # Binary op
            output_perf_index = 1
    elif primary_op.type == Op.Mul and ofm_tensor.dtype == DataType.int32:
        output_perf_index = 2
    elif primary_op.type == Op.Mul or (
        npu_block_type
        in (
            NpuBlockType.ConvolutionMxN,
            NpuBlockType.ConvolutionDepthWise,
            NpuBlockType.Pooling,
            NpuBlockType.ReduceSum,
            NpuBlockType.VectorProduct,
        )
        and use_acc_40bits
    ):
        output_perf_index = 3
    elif primary_op.type in (Op.Add, Op.Sub):
        input_scale = ifm_tensor.quantization.scale_f32
        input2_scale = ifm2_tensor.quantization.scale_f32
        output_scale = ofm_tensor.quantization.scale_f32

        if "resizebilinear" in primary_op.attrs:
            output_scale = input2_scale

        if None in (input_scale, input2_scale, output_scale) or input_scale == input2_scale:
            # Simple Add/Sub
            output_perf_index = 4
        else:
            # Advanced Add/Sub
            output_perf_index = 5
    elif primary_op.type.is_maxpool_op():
        output_perf_index = 6
    else:
        output_perf_index = 7

    if faf in (Op.Sigmoid, Op.Tanh, Op.LUT):
        activation_perf_index = 0
    elif faf in (Op.Relu, Op.Relu6, Op.ReluN1To1):
        activation_perf_index = 1
    else:
        activation_perf_index = 2

    cycle_per_elem = max(
        arch.output_cycles_per_elem[output_perf_index], arch.activation_cycles_per_elem[activation_perf_index]
    )

    if primary_op.type.is_elementwise_op() and block_config is not None:
        num_elems_blk = block_config.width * block_config.height * block_config.depth
        cycle_cmd = get_minimal_cmd_cycles(
            arch,
            ifm_tensor,
            ofm_tensor,
            block_config,
            block_config,
            num_elems_blk * cycle_per_elem,
            primary_op.ifm_shapes[0],
            primary_op.ofm_shapes[0],
        )
        cycle_per_elem = max(cycle_per_elem, cycle_cmd / num_elems_blk)

    return num_elems * cycle_per_elem


def estimate_conv_pooling_cycles(
    arch,
    npu_block_type,
    primary_op,
    ifm_block: Block,
    ofm_block: Block,
    block_traversal,
    kernel_dims,
    ifm_tensor,
    ofm_tensor,
    scale_tensor=None,
):
    ofm_ublock = Block(arch.config.ofm_ublock.width, arch.config.ofm_ublock.height, arch.config.ofm_ublock.depth)
    ifm_tens_shape = primary_op.ifm_shapes[0]
    ofm_tens_shape = primary_op.ofm_shapes[0]

    if (
        arch.config.ofm_ublock.height == 2
        and npu_block_type
        in (NpuBlockType.ConvolutionMxN, NpuBlockType.ConvolutionDepthWise, NpuBlockType.VectorProduct)
        and ofm_tens_shape.height == 1
        # Optimisation only applies for even width tensors
        and ofm_tens_shape.width % 2 == 0
        and kernel_dims[0] == 1
    ):
        ofm_ublock.width = 4
        ofm_ublock.height = 1
        ofm_block.height = 1

    num_ublk_x = numeric_util.round_up_divide(ofm_block.width, ofm_ublock.width)
    num_ublk_y = ofm_block.height // ofm_ublock.height
    num_ublk_xy = num_ublk_x * num_ublk_y
    num_ublk_z = ofm_block.depth // ofm_ublock.depth
    num_ofm_blk = 0
    total_cycles = 0
    num_elems_blk = ofm_block.width * ofm_block.height * ofm_block.depth
    use_acc_40bits = is_acc_40bits_used(npu_block_type, ifm_tensor, ofm_tensor)

    sub_kernel_limits = arch.sub_kernel_limits[npu_block_type]
    n_sub_kernels_y = numeric_util.round_up_divide(kernel_dims[0], sub_kernel_limits[0])
    n_sub_kernels_x = numeric_util.round_up_divide(kernel_dims[1], sub_kernel_limits[1])
    sub_kernel_x = [
        min((kernel_dims[1] - i * sub_kernel_limits[1]), sub_kernel_limits[1]) for i in range(n_sub_kernels_x)
    ]
    sub_kernel_y = [
        min((kernel_dims[0] - i * sub_kernel_limits[0]), sub_kernel_limits[0]) for i in range(n_sub_kernels_y)
    ]
    sub_kernel_size = (x * y for y in sub_kernel_y for x in sub_kernel_x)

    cycles_dpu_blk = 0
    cycles_wb = 32 * ofm_ublock.depth // 8

    for num_kernel_elems in sub_kernel_size:
        if npu_block_type == NpuBlockType.Pooling:
            num_kernel_steps = 1
            cycles = max(4, num_kernel_elems) * num_ublk_xy * num_ublk_z
            if ifm_tensor.dtype.size_in_bits() == 16 and arch.accelerator_config != Accelerator.Ethos_U55_32:
                cycles *= 2
        elif npu_block_type == NpuBlockType.ConvolutionDepthWise:
            cycles = 4 * num_ublk_xy
            if ifm_tensor.dtype.size_in_bits() == 16:
                cycles *= 2
            num_kernel_steps = numeric_util.round_up_divide(num_kernel_elems, 4)
            cycles = max(cycles_wb, cycles) * num_kernel_steps * num_ublk_z
        elif (
            (npu_block_type == NpuBlockType.ConvolutionMxN and block_traversal != TensorBlockTraversal.PartKernelFirst)
            or npu_block_type == NpuBlockType.VectorProduct
            or npu_block_type == NpuBlockType.ReduceSum
        ):
            num_kernel_steps = num_kernel_elems
            cycles = max(cycles_wb, 4 * num_ublk_xy) * num_kernel_steps * num_ublk_z
        else:
            assert block_traversal == TensorBlockTraversal.PartKernelFirst
            divider = 2 if ifm_tensor.dtype.size_in_bits() == 16 else 4
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
            delay = (
                3
                if use_acc_40bits and arch.accelerator_config in (Accelerator.Ethos_U55_64, Accelerator.Ethos_U55_128)
                else 2
            )
            if num_ublk_x == 1 and num_ublk_y == 1:
                if num_ublk_z == 1:
                    delay_cycles = delay * num_kernel_steps
                elif num_kernel_steps > 1:
                    delay_cycles = delay * (num_kernel_steps - 1) * num_ublk_z

        if npu_block_type == NpuBlockType.ConvolutionMxN and block_traversal == TensorBlockTraversal.PartKernelFirst:
            delay_cycles *= numeric_util.round_up_divide(ifm_block.depth, 8)

        cycles_dpu_blk += cycles
        cycles_dpu_blk += delay_cycles

    if npu_block_type in (NpuBlockType.ConvolutionMxN, NpuBlockType.VectorProduct, NpuBlockType.ReduceSum):
        cycles_dpu_blk *= numeric_util.round_up_divide(ifm_tens_shape.depth, ifm_block.depth)

    cycles_dpu_blk /= arch.ncores

    num_ofm_blk = (
        numeric_util.round_up_divide(ofm_tens_shape.height, ofm_block.height)
        * numeric_util.round_up_divide(ofm_tens_shape.width, ofm_block.width)
        * numeric_util.round_up_divide(ofm_tens_shape.depth, ofm_block.depth)
    )

    cycles_output_blk = estimate_output_cycles(
        arch, npu_block_type, primary_op, num_elems_blk, ifm_tensor, ofm_tensor, use_acc_40bits
    )

    if scale_tensor:
        cycles_bias_blk = (
            10
            * min(ofm_block.depth, ofm_tens_shape.depth)
            * arch.memory_latency[scale_tensor.mem_area][BandwidthDirection.Read]
            / 256
        )
        cycles_output_blk = max(cycles_output_blk, cycles_bias_blk)

    cycles_cmd = get_minimal_cmd_cycles(
        arch,
        ifm_tensor,
        ofm_tensor,
        ifm_block,
        ofm_block,
        cycles_dpu_blk,
        ifm_tens_shape,
        ofm_tens_shape,
        cycles_output_blk,
    )
    cycles_dpu_blk = max(cycles_dpu_blk, cycles_cmd)
    cycles_output_blk = max(cycles_output_blk, cycles_cmd)

    if cycles_dpu_blk > cycles_output_blk:
        total_cycles = cycles_dpu_blk * num_ofm_blk + cycles_output_blk
    else:
        total_cycles = cycles_output_blk * num_ofm_blk + cycles_dpu_blk

    return total_cycles


def estimate_memory_transfer_efficiency(
    arch, mem_area, direction, tensor, block_size: Block, replace_bw=None, shape4D=None
):
    if tensor.format not in (TensorFormat.NHWC, TensorFormat.NHCWB16):
        return tensor.bandwidth() if replace_bw is None else replace_bw

    # Estimate memory transfer efficiency by calculating the burst length
    # this is related to data format, block shape, and tensor shape, etc.
    burst_len = 0
    elem_size = tensor.dtype.size_in_bytes()
    is_ifm = direction == BandwidthDirection.Read
    tens = tensor.clone()
    if not tens.avoid_NHCWB16:
        tens.set_format(TensorFormat.NHCWB16, arch)
    strides = tens.get_strides(shape4D=shape4D)

    if tens.format == TensorFormat.NHCWB16:
        if strides[1] == block_size.depth:
            burst_len = elem_size * block_size.depth * block_size.width
        elif is_ifm:
            burst_len = 16 * elem_size * block_size.width
        else:
            burst_len = 16 * elem_size * block_size.width * arch.ncores
    else:
        assert tens.format == TensorFormat.NHWC
        if is_ifm:
            if strides[3] == block_size.depth:
                burst_len = elem_size * block_size.depth * block_size.width
            else:
                burst_len = elem_size * block_size.depth
        else:
            if block_size.depth <= 16 and strides[3] == block_size.depth:
                burst_len = elem_size * block_size.depth * block_size.width
            else:
                burst_len = min(64, 16 * elem_size * arch.ncores, block_size.depth * elem_size)

    burst_len = min(arch.memory_burst_length[mem_area], burst_len)
    bw = tens.bandwidth() if replace_bw is None else replace_bw

    return bw * (arch.memory_burst_length[mem_area] / burst_len)


def performance_metrics_for_pass(arch, ps, block_config=None, rewrite_list=None, force_outputs_to_fast_storage=False):
    if block_config is None:
        block_config = ps.block_config
    bws = make_bandwidth_array()
    scaled_bws = make_bandwidth_array()  # scaled bw with memory transfer efficiency
    macs = 0
    cycles = make_cycles_array()
    ifm_read_multiple = 1
    weight_read_multiple = 0

    if ps.placement in (PassPlacement.MemoryOnly, PassPlacement.StartupInit):
        return bws, macs, cycles, ifm_read_multiple, weight_read_multiple  # nothing real happening in this pass

    explicit_padding = (0, 0, 0, 0)
    primary_op = ps.primary_op
    replacement_read_bws = {}
    ofm_block = Block(block_config[1], block_config[0], block_config[3])
    ifm_block = Block(block_config[1], block_config[0], block_config[3])

    if ps.placement == PassPlacement.Npu and primary_op:
        explicit_padding = primary_op.attrs.get("explicit_padding", explicit_padding)
        assert primary_op.type.npu_block_type == ps.npu_block_type
        npu_block_type = primary_op.type.npu_block_type

        ifm_tensor, _, weight_tensor, ofm_tensor = ps.get_primary_op_ifm_ifm2_weights_ofm()
        ifm_tensor_shape = ps.primary_op.ifm_shapes[0]
        ofm_tensor_shape = ps.primary_op.ofm_shapes[0]
        ofm_block.width = min(ofm_block.width, ofm_tensor_shape.width)
        ofm_block.height = min(ofm_block.height, ofm_tensor_shape.height)
        ofm_block.depth = min(ofm_block.depth, ofm_tensor_shape.depth)

        if npu_block_type == NpuBlockType.ReduceSum:
            block_traversal = TensorBlockTraversal.DepthFirst
        elif npu_block_type in (
            NpuBlockType.ConvolutionMxN,
            NpuBlockType.ConvolutionDepthWise,
            NpuBlockType.VectorProduct,
        ):
            block_traversal = weight_tensor.block_traversal
        else:
            block_traversal = TensorBlockTraversal.Default
        ifm_block_depth = get_ifm_block_depth(
            npu_block_type, ifm_tensor_shape.depth, ifm_tensor.dtype.size_in_bits(), block_traversal, ofm_block.depth
        )
        ifm_block = arch.get_ifm_block_size(
            ifm_block_depth, ofm_block, primary_op.kernel, ifm_resampling_mode=ifm_tensor.resampling_mode
        )
        ifm_block.width = min(ifm_block.width, ifm_tensor_shape.width)
        ifm_block.height = min(ifm_block.height, ifm_tensor_shape.height)

        if npu_block_type in (
            NpuBlockType.ConvolutionMxN,
            NpuBlockType.ConvolutionDepthWise,
            NpuBlockType.VectorProduct,
            NpuBlockType.Pooling,
            NpuBlockType.ReduceSum,
        ):
            # extent the ifm to full dimension

            batch_size = ifm_tensor_shape.batch

            # add in padding, height += top and bottom, width  += left and right
            ifm_tensor_shape = ifm_tensor_shape.add(
                0, explicit_padding[0] + explicit_padding[2], explicit_padding[1] + explicit_padding[3], 0
            )

            if npu_block_type != NpuBlockType.Pooling:
                if npu_block_type == NpuBlockType.ReduceSum:
                    weight_tensor_shape = [1, 1, ifm_tensor.shape[3], ofm_tensor.shape[3]]
                    weight_tensor_bandwidth_shape = [0] * 4
                    weight_tensor_element_size = 0
                    weight_tensor_bandwidth_compression_scale = 0.0
                else:
                    # For Vector product, weight format of IO is extended to HWIO, with H=W=1
                    weight_tensor_shape = numeric_util.full_shape(4, weight_tensor.shape, 1)
                    weight_tensor_bandwidth_shape = numeric_util.full_shape(4, weight_tensor.bandwidth_shape, 1)
                    weight_tensor_element_size = weight_tensor.element_size()
                    weight_tensor_bandwidth_compression_scale = weight_tensor.bandwidth_compression_scale

                nn_ops = (
                    int(ofm_tensor_shape.batch)
                    * int(ofm_tensor_shape.height)
                    * int(ofm_tensor_shape.width)
                    * int(weight_tensor_shape[0])
                    * int(weight_tensor_shape[1])
                    * int(weight_tensor_shape[2])
                    * int(weight_tensor_shape[3])
                )
            else:
                weight_tensor_shape = [
                    *primary_op.get_kernel_size(),
                    1,
                    ifm_tensor_shape.depth,
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

            n_full_depth_stages = numeric_util.round_up_divide(weight_tensor_bandwidth_shape[3], ofm_block.depth)
            if npu_block_type in (NpuBlockType.ConvolutionDepthWise, NpuBlockType.Pooling):
                n_full_depth_stages = 1  # force to no reread

            ifm_read_multiple = n_sub_kernels * n_full_depth_stages
            replacement_read_bws[ifm_tensor] = ifm_tensor.bandwidth() * ifm_read_multiple

            weight_read_multiple = numeric_util.round_up_divide(
                ofm_tensor_shape.height, ofm_block.height
            ) * numeric_util.round_up_divide(ofm_tensor_shape.width, ofm_block.width)
            replacement_read_bws[weight_tensor] = (
                batch_size
                * shape_num_elements(weight_tensor_bandwidth_shape)
                * weight_tensor_element_size
                * weight_tensor_bandwidth_compression_scale
                * weight_read_multiple
            )

            macs += nn_ops
            cycles[PassCycles.Npu] = estimate_conv_pooling_cycles(
                arch,
                npu_block_type,
                primary_op,
                ifm_block,
                ofm_block,
                block_traversal,
                kernel_dims,
                ifm_tensor,
                ofm_tensor,
                ps.scale_tensor,
            )
        elif npu_block_type == NpuBlockType.ElementWise:
            # Work out how many elements we have and calculate performance.
            cycles[PassCycles.Npu] = estimate_output_cycles(
                arch,
                npu_block_type,
                primary_op,
                ofm_tensor.elements(),
                ps.ifm_tensor,
                ps.ofm_tensor,
                None,
                ps.ifm2_tensor,
                ofm_block,
            )

        prev_npu_pass = next((npu_ps for npu_ps in ps.dag_predecessors if npu_ps.placement is PassPlacement.Npu), None)
        if prev_npu_pass is None:
            # cycles for DMA ops in first pass
            dma_ops = (op for op in ps.ops if op.type == Op.DMA)
            for dma_op in dma_ops:
                mem_area = dma_op.attrs["source"]
                for tens in dma_op.inputs:
                    cycles[PassCycles.Npu] += tens.storage_size() / arch.memory_bandwidths_per_cycle[mem_area]

    if rewrite_list is not None:
        # apply the desired rewrites
        for rewrite_op, tens, _, _, _, ps_to_rewrite in rewrite_list:
            if ps != ps_to_rewrite:
                continue
            if rewrite_op == SchedulerRewrite.Nop:
                pass  # these are fine, no bandwidth changes
            elif rewrite_op in (SchedulerRewrite.ChangeTensorSubPurpose,):
                bws[arch.fast_storage_mem_area][tens.purpose][BandwidthDirection.Read] += replacement_read_bws[tens]
                if tens.purpose == TensorPurpose.FeatureMap:
                    scaled_bw = estimate_memory_transfer_efficiency(
                        arch,
                        arch.fast_storage_mem_area,
                        BandwidthDirection.Read,
                        tens,
                        ifm_block,
                        replacement_read_bws[tens],
                    )
                else:
                    scaled_bw = replacement_read_bws[tens]
                scaled_bws[arch.fast_storage_mem_area][tens.purpose][BandwidthDirection.Read] += scaled_bw
                replacement_read_bws[tens] = 0

    for tens in ps.outputs:
        if force_outputs_to_fast_storage:
            bws[arch.fast_storage_mem_area][tens.purpose][BandwidthDirection.Write] += tens.bandwidth()
            scaled_bws[arch.fast_storage_mem_area][tens.purpose][
                BandwidthDirection.Write
            ] += estimate_memory_transfer_efficiency(
                arch, arch.fast_storage_mem_area, BandwidthDirection.Write, tens, ofm_block, shape4D=ps.ofm_shapes[0],
            )
        else:
            bws[tens.mem_area][tens.purpose][BandwidthDirection.Write] += tens.bandwidth()
            scaled_bws[tens.mem_area][tens.purpose][BandwidthDirection.Write] += estimate_memory_transfer_efficiency(
                arch, tens.mem_area, BandwidthDirection.Write, tens, ofm_block, shape4D=ps.ofm_shapes[0]
            )

    for tens in ps.intermediates:
        bws[tens.mem_area][tens.purpose][BandwidthDirection.Write] += tens.bandwidth()
        scaled_bws[tens.mem_area][tens.purpose][BandwidthDirection.Write] += tens.bandwidth()

        if tens in replacement_read_bws:
            bw = replacement_read_bws[tens]
        else:
            bw = tens.bandwidth()

        bws[tens.mem_area][tens.purpose][BandwidthDirection.Read] += bw
        scaled_bws[tens.mem_area][tens.purpose][BandwidthDirection.Read] += bw

    for tens in ps.inputs:
        if tens in replacement_read_bws:
            bw = replacement_read_bws[tens]
        else:
            bw = tens.bandwidth()

        bws[tens.mem_area][tens.purpose][BandwidthDirection.Read] += bw

        op_shape = None
        if ps.placement == PassPlacement.Npu and primary_op:
            if tens == ps.ifm_tensor:
                op_shape = ps.ifm_shapes[0]
            elif tens == ps.ifm2_tensor:
                op_shape = ps.ifm_shapes[1]

        scaled_bws[tens.mem_area][tens.purpose][BandwidthDirection.Read] += estimate_memory_transfer_efficiency(
            arch, tens.mem_area, BandwidthDirection.Read, tens, ifm_block, bw, op_shape
        )

    # quick build access counts for only current pass, even though these aren't the final numbers
    update_summary_cycles(arch, scaled_bws, cycles)

    return bws, macs, cycles, ifm_read_multiple, weight_read_multiple


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


def collate_stats_for_cascaded_pass(arch, bws, macs, cycles):
    return bws, macs, cycles


def performance_for_cascaded_pass(arch, cps):
    total_bws = make_bandwidth_array()
    total_macs = 0
    total_cycles = make_cycles_array()

    for ps in cps.passes:
        bws, macs, cycles, _, _ = performance_metrics_for_pass(arch, ps)
        ps.bandwidths = bws
        ps.macs = macs
        ps.cycles = cycles
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
    total_macs = 0
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
