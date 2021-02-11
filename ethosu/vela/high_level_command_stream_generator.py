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
# Generate a high-level command stream from a scheduled subgraph with CascadedPasses.
#
# Also used during scheduling to work out allowable IFM/OFM overlap, this functionality can be accessed using
# calc_allowed_ofm_ifm_overlap_for_cascaded_pass().
from .high_level_command_stream import Box
from .high_level_command_stream import DMA
from .high_level_command_stream import NpuStripe
from .nn_graph import PassPlacement
from .nn_graph import SchedulingStrategy
from .numeric_util import round_up_divide
from .operation import create_activation_function
from .operation import NpuBlockType
from .operation import Op
from .shape4d import Shape4D
from .tensor import TensorPurpose


def dma_if_necessary(ps, box, tensor):
    if tensor.needs_dma():
        dma_op = tensor.ops[0]
        in_tensor = dma_op.inputs[0]
        yield DMA(ps, in_tensor, tensor, box)


def generate_high_level_command_stream_for_pass(strat, passes, block_configs, idx):
    is_first = idx == 0
    is_last = idx == len(passes) - 1
    ps = passes[idx]
    block_config = block_configs[idx]
    npu_block_type = ps.npu_block_type
    split_offsets = list(ps.primary_op.read_offsets)  # offset for [ifm, ifm2]

    if ps.ifm_tensor is not None and ps.ifm2_tensor is not None and npu_block_type == NpuBlockType.ElementWise:
        # Ensure correct ifm and ifm2 order
        if ps.inputs[0] == ps.primary_op.inputs[1] and ps.inputs[1] == ps.primary_op.inputs[0]:
            ps.ifm_tensor, ps.ifm2_tensor = ps.ifm2_tensor, ps.ifm_tensor
            ps.ifm_shapes[0], ps.ifm_shapes[1] = ps.ifm_shapes[1], ps.ifm_shapes[0]

    ifm_tensor = ps.ifm_tensor
    ifm_shape = None
    if ifm_tensor.shape != []:
        ifm_shape = ps.ifm_shapes[0]
    ifm2_tensor = ps.ifm2_tensor
    ifm2_shape = None
    if ifm2_tensor is not None and ifm2_tensor.shape != []:
        ifm2_shape = ps.ifm_shapes[1]
    ofm_tensor = ps.ofm_tensor
    ofm_shape = ps.ofm_shapes[0]
    weight_tensor = ps.weight_tensor
    scale_tensor = ps.scale_tensor

    ofm_start = [0, 0, 0, 0]
    ofm_end = ofm_shape.as_list()

    strides = None
    skirt = None
    upscaling = 1
    if ps.primary_op is not None:
        strides = ps.primary_op.attrs.get("strides", None)
        skirt = ps.primary_op.attrs.get("skirt", None)
        if ps.primary_op.type == Op.Conv2DBackpropInputSwitchedBias:
            upscaling = ofm_shape.height // ifm_shape.height
        elif ps.primary_op.type == Op.ResizeBilinear:
            upscaling = round_up_divide(ofm_shape.height, ifm_shape.height)

    concat_axis = 0
    concat_offset = 0

    for op in ps.ops:
        if op.attrs.get("concat_axis", None) is not None:
            concat_axis = op.attrs["concat_axis"]
            concat_start = op.attrs["concat_start"]
            concat_end = op.attrs["concat_end"]

            ofm_start[concat_axis] = concat_start
            ofm_end[concat_axis] = concat_end
            concat_offset = concat_start
        elif op.type.is_relu_op() or op.type in (Op.Tanh, Op.Sigmoid):
            ps.primary_op.activation = create_activation_function(op.type)

    if strat == SchedulingStrategy.WeightStream:
        ofm_step = block_config[-1]
        ofm_stop = ofm_end[-1]
        if weight_tensor is None or not weight_tensor.needs_dma():
            ofm_step = ofm_stop
        for start in range(ofm_start[-1], ofm_stop, ofm_step):
            end = min(start + ofm_step, ofm_stop)
            ofm_start[-1] = start
            ofm_end[-1] = end
            ofm_box = Box(ofm_start, ofm_end)
            ifm_box = None
            ifm2_box = None

            if ifm_shape is not None:
                ifm_box, _, _ = ofm_box.transform_with_strides_and_skirt(
                    strides, skirt, ifm_shape, npu_block_type, concat_axis, concat_offset, split_offsets[0], upscaling,
                )
            else:
                ifm_box = Box([], [])
            if ifm2_shape is not None:
                ifm2_box, _, _ = ofm_box.transform_with_strides_and_skirt(
                    strides, skirt, ifm2_shape, npu_block_type, concat_axis, concat_offset, split_offsets[1], upscaling,
                )
            else:
                ifm2_box = Box([], [])

            for intermediate in ps.intermediates:
                if (
                    intermediate is not None
                    and intermediate.shape != []
                    and intermediate.purpose in (TensorPurpose.FeatureMap, TensorPurpose.LUT)
                ):
                    if intermediate.purpose is TensorPurpose.FeatureMap:
                        intermediate_box, _, _ = ofm_box.transform_with_strides_and_skirt(
                            strides,
                            skirt,
                            Shape4D(intermediate.shape),
                            npu_block_type,
                            concat_axis,
                            concat_offset,
                            split_offsets[0],
                            upscaling,
                        )
                    else:
                        intermediate_box = Box([0] * len(intermediate.shape), list(intermediate.shape))
                    yield from dma_if_necessary(ps, intermediate_box, intermediate)

            weight_box = None
            if weight_tensor is not None:
                weight_oc_start = start
                weight_oc_end = end
                if concat_axis - len(weight_tensor.shape) == -1:
                    weight_oc_start -= concat_offset
                    weight_oc_end -= concat_offset

                weight_box = Box.make_weight_box(
                    weight_tensor.shape,
                    npu_block_type,
                    weight_oc_start,
                    weight_oc_end,
                    weight_tensor.weight_transpose_depthwise,
                )
                yield from dma_if_necessary(ps, weight_box, weight_tensor)

            yield NpuStripe(
                ps,
                block_config,
                is_first,
                is_last,
                True,
                True,
                ifm_tensor,
                ifm_box,
                ofm_tensor,
                ofm_box,
                weight_tensor,
                weight_box,
                scale_tensor,
                concat_axis,
                concat_offset,
                ifm2_tensor=ifm2_tensor,
                ifm2_box=ifm2_box,
            )

    elif strat == SchedulingStrategy.IfmStream:
        assert ifm_shape is not None
        y_step = block_config[0]
        y_start = ofm_start[-3]
        y_dim = ofm_end[-3]

        if idx > 0:
            ifm_y_present = 0
            prev_pass = passes[idx - 1]
            prev_pass_gen = generate_high_level_command_stream_for_pass(strat, passes, block_configs, idx - 1)
        else:
            ifm_y_present = 1
            ifm_y_present = ifm_shape.height
            prev_pass_gen = []
            prev_pass = None

        if len(passes) == 1:
            # no cascading, can just issue one big stripe
            # but only if we've done allocation and OFM does not overlap IFM
            if ifm_tensor.address is not None and ofm_tensor.address is not None:
                if (
                    ifm_tensor.address + ifm_tensor.storage_size() <= ofm_tensor.address
                    or ofm_tensor.address + ofm_tensor.storage_size() <= ifm_tensor.address
                ):
                    y_step = y_dim

        weight_box = None
        scale_box = None

        for start in range(y_start, y_dim, y_step):
            end = min(start + y_step, y_dim)
            ofm_start[-3] = start
            ofm_end[-3] = end
            ofm_box = Box(ofm_start, ofm_end)

            k_height = 1
            if npu_block_type in (NpuBlockType.Pooling, NpuBlockType.ReduceSum):
                if ps.primary_op is not None:
                    k_height = ps.primary_op.attrs["ksize"][1]
            else:
                if weight_tensor is not None:
                    k_height = weight_tensor.shape[0]

            ifm_box, pad_top, pad_bottom = ofm_box.transform_with_strides_and_skirt(
                strides,
                skirt,
                ifm_shape,
                npu_block_type,
                concat_axis,
                concat_offset,
                split_offsets[0],
                k_height,
                upscaling,
            )

            ifm_y_needed = 1
            if len(ifm_box.end_coord) >= 3:
                ifm_y_needed = ifm_box.end_coord[-3]
            if ifm_y_present < ifm_y_needed:
                for prev_cmd in prev_pass_gen:
                    yield prev_cmd
                    rng = prev_cmd.get_ofm_y_range_for_pass(prev_pass)
                    if rng is not None:
                        ifm_y_present = max(ifm_y_present, rng[1])
                        if ifm_y_present >= ifm_y_needed:
                            break

            for intermediate in ps.intermediates:
                if (
                    intermediate is not None
                    and intermediate.shape != []
                    and intermediate.purpose in (TensorPurpose.FeatureMap, TensorPurpose.LUT)
                ):
                    if intermediate.purpose is TensorPurpose.FeatureMap:
                        intermediate_box, _, _ = ofm_box.transform_with_strides_and_skirt(
                            strides,
                            skirt,
                            Shape4D(intermediate.shape),
                            npu_block_type,
                            concat_axis,
                            concat_offset,
                            split_offsets[0],
                            upscaling,
                        )
                    else:
                        intermediate_box = Box([0] * len(intermediate.shape), list(intermediate.shape))
                    yield from dma_if_necessary(ps, intermediate_box, intermediate)

            if scale_tensor is not None and scale_tensor.purpose == TensorPurpose.FSBias and scale_box is None:
                scale_box = Box([0] * len(scale_tensor.shape), list(scale_tensor.shape))
                yield from dma_if_necessary(ps, scale_box, scale_tensor)

            if weight_tensor is not None and weight_box is None:
                weight_box = Box.make_weight_box(
                    weight_tensor.shape, npu_block_type, weights_transposed=weight_tensor.weight_transpose_depthwise
                )
                yield from dma_if_necessary(ps, weight_box, weight_tensor)

            # Check if first/last stripe in pass
            is_first_h_stripe = start == y_start
            is_last_h_stripe = (start + y_step) >= y_dim

            stripe = NpuStripe(
                ps,
                block_config,
                is_first,
                is_last,
                is_first_h_stripe,
                is_last_h_stripe,
                ifm_tensor,
                ifm_box,
                ofm_tensor,
                ofm_box,
                weight_tensor,
                weight_box,
                scale_tensor,
                concat_axis,
                concat_offset,
                None,
                None,
                pad_top,
                pad_bottom,
            )
            yield stripe
    else:
        assert 0, "unknown scheduling strategy"


def generate_high_level_command_stream_for_pass_list(strat, passes, block_configs):
    if strat == SchedulingStrategy.WeightStream:
        for idx in range(len(passes)):
            yield from generate_high_level_command_stream_for_pass(strat, passes, block_configs, idx)
    elif strat == SchedulingStrategy.IfmStream:
        yield from generate_high_level_command_stream_for_pass(strat, passes, block_configs, len(passes) - 1)
    else:
        assert 0, "Unknown streaming strategy"


def generate_high_level_command_stream_for_cascaded_pass(cps):
    yield from generate_high_level_command_stream_for_pass_list(
        cps.strategy, cps.passes, [ps.block_config for ps in cps.passes]
    )


def generate_high_level_command_stream(nng, sg, arch, verbose_high_level_command_stream):
    res = []
    for cps in sg.cascaded_passes:
        if cps.placement == PassPlacement.Npu:
            res += list(generate_high_level_command_stream_for_cascaded_pass(cps))

    sg.high_level_command_stream = res
    if verbose_high_level_command_stream:
        sg.print_high_level_command_stream()


def calc_allowed_ofm_ifm_overlap_for_pass_list(strat, passes, block_configs):
    highest_ofm_write = 0
    if not passes[0].ifm_tensor or not passes[-1].ofm_tensor:
        return 0

    ifm_read = passes[0].ifm_tensor.storage_size()
    min_overlap = 999999999999999999999
    ofm_size = passes[-1].ofm_tensor.storage_size()
    if strat == SchedulingStrategy.WeightStream:
        return 0
    for cmd in generate_high_level_command_stream_for_pass_list(strat, passes, block_configs):
        if cmd.is_npu_pass_command():
            if cmd.is_first:
                ifm_read = cmd.ifm_tensor.address_offset_for_coordinate(
                    cmd.ifm_box.start_coord, cmd.ps.ifm_shapes[0], is_top_box=False
                )
                if ifm_read is None:
                    return 0
            if cmd.is_last:
                write_offset = cmd.ofm_tensor.address_offset_for_coordinate(
                    cmd.ofm_box.end_coord, cmd.ps.ofm_shapes[0], is_top_box=True
                )
                if write_offset is None:
                    return 0
                highest_ofm_write = max(write_offset, highest_ofm_write)

            if cmd.is_first or cmd.is_last:
                overlap_required = max(highest_ofm_write - min(ifm_read, ofm_size), 0)
                can_overwrite = ofm_size - overlap_required
                min_overlap = min(min_overlap, can_overwrite)

            if cmd.is_first:
                ifm_read = cmd.ifm_tensor.address_offset_for_coordinate(
                    cmd.ifm_box.end_coord, cmd.ps.ifm_shapes[0], is_top_box=True
                )

    min_overlap = max(min_overlap, 0)
    return min_overlap
