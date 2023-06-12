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
# Generate a high-level command stream from a schedule
from .architecture_allocator import is_nearest
from .high_level_command_stream import Box
from .high_level_command_stream import DMA
from .high_level_command_stream import NOP
from .high_level_command_stream import NpuStripe
from .numeric_util import round_up_divide
from .operation import create_activation_function
from .operation import NpuBlockType
from .operation import Op
from .shape4d import Shape4D
from .tensor import TensorPurpose


def dma_if_necessary(ps, box, tensor):
    src_tensor = tensor.src_tensor
    if src_tensor and tensor.mem_area != src_tensor.mem_area:
        yield DMA(ps, src_tensor, tensor, box)


def dma_feature_map_if_necessary(ps, src_tensor, dst_tensor):
    box = Box([0] * len(src_tensor.shape), list(src_tensor.shape))
    src_addr = src_tensor.address_for_coordinate(box.start_coord)
    dst_addr = dst_tensor.address_for_coordinate(box.start_coord)

    if src_addr != dst_addr or src_tensor.mem_area != dst_tensor.mem_area:
        yield DMA(ps, src_tensor, dst_tensor, box)
    else:
        # Source and destination is the same so no need for a DMA transaction
        # Create a NOP for visibility when printing the high_level_command_stream
        yield NOP(ps, src_tensor, dst_tensor)


def generate_high_level_command_stream_for_schedule(nng, sg, arch, verbose_high_level_command_stream):
    res = []
    # sg.sched_ops are ordered by execution
    processed_cascades = set()
    for sched_op in sg.sched_ops:
        op_info = sg.schedule.cost_map[sched_op]
        if op_info.cascade in processed_cascades:
            # This cascade has already been processed
            continue

        if op_info.cascade == 0:
            # Generate high-level commands for this Op in isolation
            res += list(generate_high_level_commands_for_sched_op(sched_op, sg.schedule))
        else:
            # Generate high-level commands for the whole cascade
            cascade_info = sg.schedule.cascades[op_info.cascade]
            # Start from the last Op in the cascade
            res += list(generate_high_level_commands_for_sched_op(sg.sched_ops[cascade_info.end], sg.schedule))
            processed_cascades.add(op_info.cascade)

    sg.high_level_command_stream = res
    if verbose_high_level_command_stream:
        sg.print_high_level_command_stream()


def generate_high_level_commands_for_sched_op(sched_op, schedule):
    op_info = schedule.cost_map[sched_op]
    cascade_info = schedule.cascades.get(op_info.cascade)
    npu_block_type = sched_op.parent_ps.npu_block_type
    block_config = op_info.block_config
    ps = sched_op.parent_ps
    parent_op = sched_op.parent_op
    ofm_tensor = ps.ofm_tensor

    # Get Tensors and Full Shapes
    (
        ifm_tensor,
        ifm2_tensor,
        uncomp_weight_tensor,
        _,
        _,
    ) = parent_op.get_ifm_ifm2_weights_biases_ofm()
    if sched_op.reversed_operands:
        ifm2_tensor, ifm_tensor = ifm_tensor, ifm2_tensor
    ifm = sched_op.ifm
    ifm2 = sched_op.ifm2
    ofm_shape = sched_op.ofm.shape

    # Get Kernel strides and upscaling factor
    kernel_stride = sched_op.kernel.stride
    strides = [1, kernel_stride.y, kernel_stride.x, 1]
    skirt = parent_op.attrs.get("skirt", None)
    upscaling = 1
    if sched_op.op_type == Op.Conv2DBackpropInputSwitchedBias:
        upscaling = ofm_shape.height // ifm.shape.height
    elif is_nearest(sched_op.resampling_mode):
        upscaling = round_up_divide(ofm_shape.height, ifm.shape.height)

    # Get kernel height and height dilation
    k_height = 1
    if npu_block_type in (NpuBlockType.Pooling, NpuBlockType.ReduceSum):
        if parent_op is not None:
            k_height = parent_op.attrs["ksize"][1]
    else:
        if uncomp_weight_tensor is not None:
            k_height = uncomp_weight_tensor.shape[0]

    k_height_dilation = parent_op.attrs.get("dilation", (_, 1, _, _))[-3]

    # Calculate dilated kernel height
    k_dilated_height = k_height_dilation * (k_height - 1) + 1

    # Define Start and End coordinates for the OFM
    ofm_start = Shape4D(0, 0, 0, op_info.ofm_depth_slices[0])
    ofm_end = ofm_shape

    ofm_depth_slices = op_info.ofm_depth_slices

    # Read/Write offsets
    read_offsets = list(parent_op.read_offsets)  # offset for [ifm, ifm2]
    read_shapes = list(parent_op.read_shapes)  # read shapes for [ifm, ifm2]
    write_offset = Shape4D(0, 0, 0, 0)
    if parent_op.write_offset is not None:
        write_offset = parent_op.write_offset
        ofm_start = write_offset
        ofm_end = parent_op.write_offset + parent_op.write_shape

    # Create activation function if needed
    for op in ps.ops:
        if op.type.is_relu_op() or op.type in (Op.Tanh, Op.Sigmoid):
            ps.primary_op.activation = create_activation_function(
                op.type, min=op.attrs.get("min", None), max=op.attrs.get("max", None)
            )

    # Generate commands for the Op that produces this Op's IFM, if applicable
    if cascade_info is None or cascade_info.start == sched_op.index:
        # Lone Op or First Op in cascade - all IFM data is present
        ifm_present = Box([0, 0, 0, 0], ifm.shape.as_list())
        producer_op = None
        prev_cmd_gen = []
    else:
        ifm_present = Box([0, 0, 0, 0], [0, 0, 0, 0])
        producer_op = sched_op.ifm.connection.producers[0]
        prev_cmd_gen = generate_high_level_commands_for_sched_op(producer_op, schedule)
    ofm_step = op_info.stripe
    for start_height in range(ofm_start.height, ofm_end.height, ofm_step.height):
        end_height = min(start_height + ofm_step.height, ofm_end.height)
        for start_width in range(ofm_start.width, ofm_end.width, ofm_step.width):
            end_width = min(start_width + ofm_step.width, ofm_end.width)

            lut_dma_done = False
            for depth_idx, start_channel in enumerate(ofm_depth_slices[:-1]):
                start_channel = max(start_channel, ofm_start.depth)
                end_channel = min(ofm_depth_slices[depth_idx + 1], ofm_end.depth)

                # Construct the OFM box for the current stripe
                ofm_box_start = Shape4D(ofm_start.batch, start_height, start_width, start_channel)
                ofm_box_end = Shape4D(ofm_end.batch, end_height, end_width, end_channel)
                ofm_box = Box(ofm_box_start.as_list(), ofm_box_end.as_list())
                ifm_box = Box([], [])
                ifm2_box = Box([], [])
                # Calculate IFM input box based on the OFM box
                if ifm:
                    ifm_box, pad_top, pad_bottom = ofm_box.transform_with_strides_and_skirt(
                        strides,
                        skirt,
                        ifm.shape,
                        npu_block_type,
                        write_offset.as_list(),
                        k_dilated_height,
                        read_offsets[0],
                        read_shapes[0],
                        upscaling,
                        sched_op.op_type,
                    )
                # Calculate IFM2 input box based on the OFM box
                if ifm2:
                    ifm2_box, pad_top, pad_bottom = ofm_box.transform_with_strides_and_skirt(
                        strides,
                        skirt,
                        ifm2.shape,
                        npu_block_type,
                        write_offset.as_list(),
                        k_dilated_height,
                        read_offsets[1],
                        read_shapes[1],
                        upscaling,
                        sched_op.op_type,
                    )

                ifm_required = ifm_box
                # Get the Op that produces this Op's IFM data - only applicable within cascades
                if producer_op:
                    assert op_info.cascade != 0
                    assert op_info.cascade == schedule.cost_map[producer_op].cascade
                    if not ifm_required.is_subbox_of(ifm_present):
                        for prev_cmd in prev_cmd_gen:
                            yield prev_cmd
                            if prev_cmd.is_npu_pass_command() and prev_cmd.ps == producer_op.parent_ps:
                                ifm_present.end_coord = prev_cmd.ofm_box.end_coord
                                if ifm_required.is_subbox_of(ifm_present):
                                    # There is enough IFM data - exit loop
                                    break

                # Information about the current stripe's location in the cascade
                is_first_h_stripe = ofm_box_start.height == ofm_start.height
                is_last_h_stripe = ofm_box_end.height >= ofm_end.height

                # Calculate the weight box - i.e. the subshape of weights needed for this NpuStripe command
                weight_tensor = op_info.npu_weights_tensor
                scale_tensor = op_info.npu_scales_tensor
                if op_info.npu_weights_tensor:
                    weight_box = Box([0, 0, 0, start_channel], [1, 1, 1, end_channel])

                    if op_info.buffered_weight_tensors:
                        idx = depth_idx % len(op_info.buffered_weight_tensors)
                        weight_tensor = op_info.buffered_weight_tensors[idx]
                        if is_first_h_stripe:
                            yield from dma_if_necessary(
                                sched_op.parent_ps, weight_box, op_info.buffered_weight_tensors[idx]
                            )
                else:
                    weight_box = None

                # Should only be done once per loop but not before weights above
                if parent_op.activation_lut and not lut_dma_done:
                    lut_tensor = [tens for tens in parent_op.inputs if tens.purpose == TensorPurpose.LUT][0]
                    lut_box = Box([0] * len(lut_tensor.shape), list(lut_tensor.shape))
                    lut_dma_done = True
                    yield from dma_if_necessary(sched_op.parent_ps, lut_box, lut_tensor)

                if parent_op.type == Op.Memcpy:
                    yield from dma_feature_map_if_necessary(sched_op.parent_ps, ifm_tensor, ofm_tensor)
                else:
                    yield NpuStripe(
                        sched_op.parent_ps,
                        block_config.old_style_representation(),
                        is_first_h_stripe,
                        is_last_h_stripe,
                        ifm_tensor,
                        ifm_box,
                        ofm_tensor,
                        ofm_box,
                        weight_tensor,
                        weight_box,
                        scale_tensor,
                        ifm2_tensor=ifm2_tensor,
                        ifm2_box=ifm2_box,
                        pad_top=pad_top,
                        pad_bottom=pad_bottom,
                        reversed_operands=sched_op.reversed_operands,
                    )
