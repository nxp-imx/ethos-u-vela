# Copyright (C) 2021 Arm Limited or its affiliates. All rights reserved.
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
# Early optimisation of the TOSA based network graph, using the rewrite_graph module to do the traversal of the graph.
import numpy as np

from . import rewrite_graph
from .api import NpuRoundingMode
from .data_type import DataType
from .debug_database import DebugDatabase
from .graph_optimiser_util import bypass_memory_only_ops
from .graph_optimiser_util import calc_explicit_padding
from .graph_optimiser_util import convert_depthwise_to_conv
from .graph_optimiser_util import move_splitsliceread_to_consumer
from .graph_optimiser_util import needed_total_padding
from .graph_optimiser_util import set_ifm_ofm_op_shapes
from .graph_optimiser_util import set_tensor_equivalence
from .operation import ExplicitScaling
from .operation import Op
from .operation_util import create_add_nop
from .operation_util import create_avgpool_nop
from .shape4d import Shape4D
from .tensor import create_const_tensor
from .tensor import create_equivalence_id


def replace_rescale_with_avg_pool(rescale_op):
    assert rescale_op.type == Op.Rescale

    avgpool_op = create_avgpool_nop(rescale_op.name + "_avgpool")
    rescale_op_clone = rescale_op.clone()
    op = rescale_op
    op.attrs = avgpool_op.attrs.copy()
    op.type = Op.AvgPool
    DebugDatabase.add_optimised(rescale_op_clone, op)

    return op


def calc_skirt(kernel, input_shape, explicit_padding):
    k_w, k_h = kernel.dilated_wh()
    s_x, s_y = kernel.stride
    ypad = needed_total_padding(int(input_shape.height), int(s_y), int(k_h))
    xpad = needed_total_padding(int(input_shape.width), int(s_x), int(k_w))

    top, left, bottom, right = explicit_padding
    top_pad, bottom_pad = calc_explicit_padding(int(input_shape.height), int(s_y), int(k_h), int(top), int(bottom))
    left_pad, right_pad = calc_explicit_padding(int(input_shape.width), int(s_x), int(k_w), int(left), int(right))

    padding = (top_pad, left_pad, bottom_pad, right_pad)
    skirt = (top_pad, left_pad, ypad - top_pad, xpad - left_pad)
    return padding, skirt


def add_padding_fields(op, arch, nng):
    if op.run_on_npu:
        if "explicit_padding" in op.attrs:
            input_shape = op.ifm_shapes[0]

            if op.type == Op.Conv2DBackpropInputSwitchedBias:
                # TODO not yet supported, but there will be need for separate handling
                assert False
            else:
                padding, skirt = calc_skirt(op.kernel, input_shape, op.attrs.get("explicit_padding"))

            op.attrs["explicit_padding"] = padding
            op.attrs["skirt"] = skirt

    return op


# Counts leading zeroes for a (int32)
def count_leading_zeros(a):
    lz = int(32)
    if a != 0:
        mask = 1 << (32 - 1)
        lz = 0
        while (mask & a) == 0:
            mask = mask >> 1
            lz = lz + 1
    return lz


def calc_scaling_avgpool(op, arch, nng):
    if op.type == Op.AvgPool:
        top, left, _, _ = op.attrs["explicit_padding"]
        # TODO Only support for when global scaling can be used.
        # That is when there is no padding
        assert top == 0 and left == 0
        assert op.explicit_scaling is None
        multiplier = []
        shift = []

        kernel_wh = op.kernel.elements_wh()
        k = 32 - count_leading_zeros(kernel_wh - 1)
        numerator = np.int64(((1 << 30) + 1) << k)
        multiplier.append(numerator // kernel_wh)
        shift.append(30 + k)

        op.rounding_mode = NpuRoundingMode.NATURAL
        op.explicit_scaling = ExplicitScaling(False, shift, multiplier)
    return op


def remove_const_transpose(op, arch, nng):
    if op.type == Op.Transpose:
        removed = False
        if len(op.ifm.ops) == 1:
            prev_op = op.ifm.ops[0]
            if prev_op.type == Op.Const:
                # Transpose the Tensor and data and remove Transpose
                # TODO move to Tensor?
                reorder = op.attrs["perms"]
                shape = op.ifm.shape.copy()
                tens = op.ifm

                tens.shape = [shape[idx] for idx in reorder]
                tens.bandwidth_shape = tens.shape
                tens.storage_shape = tens.shape

                if tens.values is not None:
                    tens.values = tens.values.transpose(reorder)

                op.ofm.values = tens.values
                # Bypass the Transpose op
                prev_op.set_output_tensor(op.ofm)
                DebugDatabase.add_optimised(op, prev_op)
                removed = True

        if not removed:
            print("Warning: Cannot remove Transpose, and handling of Transpose is not supported")
            assert False

    return op


# TODO can we change to add for both TFLite and TOSA?
def insert_add_copy_op_after_tens(tens):
    tens_cons_list_copy = tens.consumer_list.copy()
    copy_tens = tens.clone()

    name = tens.name + "_add"
    ifm2 = create_const_tensor(
        name + "_zero_scalar",
        [1],
        copy_tens.dtype,
        [0],
        copy_tens.dtype.as_numpy_type(),
        quantization=copy_tens.quantization,
    )
    copy_op = create_add_nop(name)
    copy_op.add_input_tensor(tens)
    copy_op.add_input_tensor(ifm2)
    copy_op.set_output_tensor(copy_tens)
    copy_op.set_ifm_ofm_shapes()
    copy_op.run_on_npu = True

    # Set copy_ifm consumers
    for tens_cons in tens_cons_list_copy:
        if tens_cons is not None:
            for ifm_idx, cons_inp in enumerate(tens_cons.inputs):
                if cons_inp == tens:
                    tens_cons.set_input_tensor(copy_tens, ifm_idx)

    DebugDatabase.add_optimised(tens.ops[0], copy_op)


def fix_sg_input_output_tosa(op, arch, nng):
    if not op.run_on_npu or op.type != Op.Reshape:
        return op

    # For the Reshape operators we want to remove, tensors are removed.
    # But in order to to do this, they cannot be outputs of the sg,
    # this need to be fixed prior to the removal.
    # Solution is to add a copy op, to maintain the original tensor.
    # This is also valid when reshape ifm/ofm is produced respectively
    # consumed by CPU

    # Check if operator ifm/ofm are sg ifm/ofm
    ifm_is_sg_ifm = op.ifm.ops[0].type in (Op.Placeholder, Op.SubgraphInput, Op.Const)
    ifm_is_sg_ofm = any(ifm_cons is None for ifm_cons in op.ifm.consumer_list)
    ofm_is_sg_ofm = any(ofm_cons is None for ofm_cons in op.ofm.consumer_list)
    # Check if ifm/ofm is produced repectivly consumed by CPU
    ifm_is_cpu_produced = any(ifm_prod is not None and not ifm_prod.run_on_npu for ifm_prod in op.ifm.ops)
    ofm_is_cpu_consumed = any(ofm_cons is not None and not ofm_cons.run_on_npu for ofm_cons in op.ofm.consumer_list)

    if (ifm_is_sg_ofm or ifm_is_sg_ifm or ifm_is_cpu_produced) and (ofm_is_sg_ofm or ofm_is_cpu_consumed):
        # Both ifm and ofm need to persist, but only ifm need a copy, in order to remove the Reshape
        insert_add_copy_op_after_tens(op.ifm)

    return op


def create_add_for_concat(concat_op, name, ifm, ifm_shape: Shape4D, write_offset: Shape4D):
    """Creates an add op for the given concat op/input feature map"""
    ofm = concat_op.ofm
    ifm2 = create_const_tensor(
        name + "_zero_scalar", [1], ofm.dtype, [0], ofm.dtype.as_numpy_type(), quantization=ofm.quantization
    )
    add_op = create_add_nop(name)

    add_op.inputs = [ifm, ifm2]
    add_op.outputs = [ofm]
    add_op.write_offset = write_offset
    add_op.write_shape = ifm_shape
    ofm.ops.append(add_op)
    DebugDatabase.add_optimised(concat_op, add_op)
    add_op.ifm_shapes.append(ifm_shape)
    add_op.ifm_shapes.append(Shape4D(ifm2.shape))
    add_op.ofm_shapes.append(concat_op.ofm_shapes[0])
    add_op.memory_function = Op.ConcatSliceWrite
    return add_op


# TODO Could be further optimized checking the type of the consumer,
# rather than just mimic the TFLite behaviour depending on type.
# TOSA bool_t not considered yet
def remove_splitsliceread(op, arch):

    if op.type == Op.SplitSliceRead:
        # Check if it is possible to put the SplitSliceRead on the tensor consumer, or if an avgpool need to be inserted
        if (
            len(op.ofm.consumer_list) == 1
            and op.ofm.consumer_list[0] is not None
            and op.ofm.consumer_list[0].run_on_npu
            and op.ofm.consumer_list[0].type != Op.Reshape
            and op.ofm_shapes[0] == Shape4D.from_list(op.ofm.shape)
            and op.ofm.dtype in (DataType.uint8, DataType.int8, DataType.int16)
        ):
            # SplitSliceRead can be performed by tensor consumer
            cons_op = op.ofm.consumer_list[0]
            move_splitsliceread_to_consumer(op, cons_op)
        else:
            name = op.name + "_add"
            ofm = op.ofm
            ifm2 = create_const_tensor(
                name + "_zero_scalar", [1], ofm.dtype, [0], ofm.dtype.as_numpy_type(), quantization=ofm.quantization
            )
            add_op = create_add_nop(name)
            add_op.inputs = [op.ifm, ifm2]
            add_op.outputs = [ofm]
            op.ofm.ops.remove(op)
            op.ofm.ops.append(add_op)
            add_op.ifm_shapes.append(op.ifm_shapes[0])
            add_op.ifm_shapes.append(Shape4D(ifm2.shape))
            add_op.ofm_shapes.append(op.ofm_shapes[0])
            add_op.read_offsets[0] = op.read_offsets[0]
            add_op.read_shapes[0] = op.read_shapes[0]

            op.ifm.consumer_list.remove(op)
            DebugDatabase.add_optimised(op, add_op)


def rewrite_concat_ops(op, arch):
    if not op.run_on_npu or not op.type == Op.Concat:
        return

    axis_4D = 0
    ofm = op.ofm
    ofm.ops = []
    offset = 0

    inputs = op.inputs
    axis = op.attrs["axis"]

    for idx, inp in enumerate(inputs):
        op.ifm_shapes[idx] = Shape4D(inp.shape)
        if axis >= 0:
            axis_4D = axis + (4 - len(inp.shape))
        else:
            axis_4D = axis
        write_offset = [0, 0, 0, 0]
        write_offset[axis_4D] = offset
        concat_end = offset + op.ifm_shapes[idx][axis_4D]
        create_add_for_concat(op, op.name + str(idx) + "_add", inp, op.ifm_shapes[idx], Shape4D.from_list(write_offset))
        offset = concat_end
    assert ofm.shape[axis] == offset

    return op


def remove_reshapes(op, arch):
    if op.run_on_npu and op.type == Op.Reshape:
        bypass_memory_only_ops(op)


def rewrite_activation(op, arch, nng):
    if op.type not in (Op.ReluN, Op.Clamp):
        return op

    ifm = op.ifm
    zp = ifm.quantization.zero_point if ifm.quantization.zero_point else 0
    if op.ofm.quantization.zero_point is None:
        op.ofm.quantization.zero_point = zp

    if op.type == Op.Clamp:
        op.attrs["min"] = op.attrs["min_int"] - zp
        op.attrs["max"] = op.attrs["max_int"] - zp
    elif op.type == Op.ReluN:
        op.attrs["max"] = op.attrs["max_int"] - zp

    return op


def rewrite_rescale(op, arch, nng):
    if op.type == Op.Rescale:
        ifm = op.ifm
        ofm = op.ofm

        # some error checking
        assert len(ifm.ops) == 1
        prev_op = ifm.ops[0]

        # TODO currently not supported
        assert len(ifm.consumer_list) == 1

        input_zp = op.attrs["input_zp"]
        output_zp = op.attrs["output_zp"]
        multiplier = op.attrs["multiplier"]
        shift = op.attrs["shift"]
        scale32 = op.attrs["scale32"]
        double_round = op.attrs["double_round"]
        per_channel = op.attrs["per_channel"]

        assert ifm.dtype in (DataType.uint8, DataType.int8, DataType.int32)
        assert ifm.dtype in (DataType.uint8, DataType.int8) or input_zp == 0
        assert ofm.dtype in (DataType.uint8, DataType.int8) or output_zp == 0
        assert (scale32 and ifm.dtype != DataType.int48) or (not scale32 and not double_round)

        # Check that input tensor has the same zp or no zp
        ifm_zp = ifm.quantization.zero_point
        if ifm_zp is not None and ifm_zp != input_zp:
            print("Error (fuse_rescale): zp of tensors producer/consumer differs unexpectedidly ")
            assert False
        ifm.quantization.zero_point = input_zp
        ofm.quantization.zero_point = output_zp
        for s, m in zip(shift, multiplier):
            # TODO these are the TOSA limitations
            assert m >= 0
            assert 2 <= s <= 62
            # TODO these are the HW limitations
            assert 0 <= s < (1 << 6)
        explicit_scaling = ExplicitScaling(per_channel, shift, multiplier)

        if double_round and scale32:
            rounding_mode = NpuRoundingMode.TFL
        else:
            rounding_mode = NpuRoundingMode.NATURAL

        if prev_op.type.is_depthwise_conv2d_op() or prev_op.type.is_conv2d_op() or prev_op.type == Op.FullyConnected:
            assert len(multiplier) == len(shift) == len(prev_op.bias.values)

            if ifm.dtype == DataType.int32 and per_channel:
                prev_op.explicit_scaling = explicit_scaling
                prev_op.rounding_mode = rounding_mode

                # Bypass op
                prev_op.set_output_tensor(ofm)
                DebugDatabase.add_optimised(op, prev_op)
                return op
            else:
                print("Warning, unsupported fusing of TOSA Rescale previous operator is of type:", prev_op.type)
                assert False
        # TODO which are the cases we need to and can do standalone Rescale?
        # TODO should we try to identify a conversion uint8<->int8 accomplished by 2 RESCALE ops?
        # origin might be TFLite op QUANTIZE, should we look to see if they can be translated to QUANTIZE?
        # limited to these at the moment:
        elif (
            (ifm.dtype == DataType.int8 and ofm.dtype == DataType.int8)
            or (ifm.dtype == DataType.uint8 and ofm.dtype == DataType.int8)
            or (ifm.dtype == DataType.int8 and ofm.dtype == DataType.uint8)
        ):
            # Create  NOP performing the RESCALE
            avgpool_op = replace_rescale_with_avg_pool(op)
            avgpool_op.rounding_mode = rounding_mode

            if per_channel:
                # TODO
                avgpool_op.explicit_scaling = explicit_scaling
                print("Warning, unsupported TOSA Rescale")
                assert False
            else:
                avgpool_op.explicit_scaling = explicit_scaling
        else:
            print("Warning, unsupported fusing of TOSA Rescale previous operator is of type:", prev_op.type)
            assert False
    return op


# TODO modified copy of TFLite, solution for TOSA PAD will change so reuse has not been considered
def convert_pad(op, arch, nng):
    """
    Rewrites PAD operator to an add that copies the IFM to the OFM
    + up to 4 add operators that fill the OFM with zeros at the borders.
    """

    if op.type != Op.Pad:
        return op

    # TODO assuming rank <= 4 and N = 1 for rank ==4
    # This is checked in tosa_supported_operators
    ifm = op.ifm
    assert ifm is not None
    ifm_shape = Shape4D(ifm.shape)
    ofm = op.ofm
    assert ofm is not None
    ofm.ops = []
    ofm_shape = op.ofm_shapes[0]

    rank = len(ifm.shape)
    padding = op.inputs[1].values
    pad_depth = padding[-1]
    if not (pad_depth == 0).all():
        print("Warning: For PAD, padding in depth not supported yet")
        assert False

    top, bottom = 0, 0
    left, right = 0, 0
    if rank > 1:
        left, right = padding[-2][0], padding[-2][1]
    if rank > 2:
        top, bottom = padding[-3][0], padding[-3][1]
    if rank == 4 and not (padding[-4] == 0).all():
        print("Warning: For PAD, padding not supported in first dimension when rank == 4 yet")
        assert False

    # Add op that copies IFM to the right place inside the OFM
    shp0 = Shape4D(0, 0, 0, 0)
    shp_top = shp0.with_height(top)
    add_op = create_add_for_concat(op, op.name + "_main", ifm, ifm_shape, shp_top.with_width(left))
    add_op.activation = op.activation

    quant = ofm.quantization
    pad_value = ifm.quantization.zero_point
    # Add operations that fill the borders of the OFM
    if top > 0:
        shape = Shape4D(1, top, ofm_shape.width, ofm_shape.depth)
        zero_tens = create_const_tensor(
            op.name + "_top",
            shape.as_list(),
            ofm.dtype,
            shape.elements() * [pad_value],
            np.uint8,
            quantization=quant,  # TODO
        )
        # If top/bottom or left/right are equal, the const tensors can be allocated to the same address
        zero_tens.equivalence_id = create_equivalence_id(tuple(zero_tens.values))
        create_add_for_concat(op, op.name + "_top", zero_tens, shape, shp0)
    if bottom > 0:
        shape = Shape4D(1, bottom, ofm_shape.width, ofm_shape.depth)
        zero_tens = create_const_tensor(
            op.name + "_bottom",
            shape.as_list(),
            ofm.dtype,
            shape.elements() * [pad_value],
            np.uint8,
            quantization=quant,
        )
        zero_tens.equivalence_id = create_equivalence_id(tuple(zero_tens.values))
        create_add_for_concat(op, op.name + "_bottom", zero_tens, shape, shp0.with_height(ofm_shape.height - bottom))
    if left > 0:
        shape = Shape4D(1, ifm_shape.height, left, ofm_shape.depth)
        zero_tens = create_const_tensor(
            op.name + "_left", shape.as_list(), ofm.dtype, shape.elements() * [pad_value], np.uint8, quantization=quant
        )
        zero_tens.equivalence_id = create_equivalence_id(tuple(zero_tens.values))
        create_add_for_concat(op, op.name + "_left", zero_tens, shape, shp_top)
    if right > 0:
        shape = Shape4D(1, ifm_shape.height, right, ofm_shape.depth)
        zero_tens = create_const_tensor(
            op.name + "_right", shape.as_list(), ofm.dtype, shape.elements() * [pad_value], np.uint8, quantization=quant
        )
        zero_tens.equivalence_id = create_equivalence_id(tuple(zero_tens.values))
        create_add_for_concat(op, op.name + "_right", zero_tens, shape, shp_top.with_width(ofm_shape.width - right))

    op.type = Op.ConcatTFLite
    return add_op


def fixup_quantization(op, arch, nng):
    if op.ifm and op.ifm.quantization.zero_point is None:
        op.ifm.quantization.zero_point = 0
    if op.ifm2 and op.ifm2.quantization.zero_point is None:
        op.ifm.quantization.zero_point = 0
    if op.ofm and op.ofm.quantization.zero_point is None:
        op.ofm.quantization.zero_point = 0
    return op


def supported_operator_check(op, arch, nng):
    op.run_on_npu = arch.tosa_supported_operators.is_operator_supported(op)
    assert op.run_on_npu or op.type in (Op.Placeholder, Op.SubgraphInput, Op.Const)
    return op


def tosa_optimise_graph(nng, arch):
    # Pre-processing step
    pre_process_list = [
        supported_operator_check,
        set_ifm_ofm_op_shapes,
    ]

    for idx, sg in enumerate(nng.subgraphs):
        nng.subgraphs[idx] = rewrite_graph.rewrite_graph_pre_order(
            nng, sg, arch, [], pre_process_list, rewrite_unsupported=False,
        )

    # Removal of Transpose
    for idx, sg in enumerate(nng.subgraphs):
        nng.subgraphs[idx] = rewrite_graph.rewrite_graph_pre_order(
            nng, sg, arch, [], [remove_const_transpose], rewrite_unsupported=False,
        )

    # Handle sg input output
    for idx, sg in enumerate(nng.subgraphs):
        nng.subgraphs[idx] = rewrite_graph.rewrite_graph_pre_order(
            nng, sg, arch, [], [fix_sg_input_output_tosa], rewrite_unsupported=False,
        )

    # Rewrite concat ops
    for idx, sg in enumerate(nng.subgraphs):
        rewrite_graph.visit_graph_post_order(sg.output_tensors, arch, [], [rewrite_concat_ops])
        sg.refresh_after_modification()

    # Removal of reshapes
    for sg in nng.subgraphs:
        rewrite_graph.visit_graph_post_order(sg.output_tensors, arch, [], [remove_reshapes])
        sg.refresh_after_modification()

    # TODO, when and where to best handle calc_scaling_avgpool
    for idx, sg in enumerate(nng.subgraphs):
        nng.subgraphs[idx] = rewrite_graph.rewrite_graph_pre_order(
            nng, sg, arch, [], [calc_scaling_avgpool], rewrite_unsupported=False,
        )

    # Rewite Operators step
    op_rewrite_list = [set_tensor_equivalence, rewrite_rescale, convert_depthwise_to_conv]

    for idx, sg in enumerate(nng.subgraphs):
        nng.subgraphs[idx] = rewrite_graph.rewrite_graph_pre_order(
            nng, sg, arch, [], op_rewrite_list, rewrite_unsupported=False,
        )

    # Post-processing step 1
    for idx, sg in enumerate(nng.subgraphs):
        nng.subgraphs[idx] = rewrite_graph.rewrite_graph_pre_order(
            nng, sg, arch, [], [rewrite_activation, convert_pad, add_padding_fields],
        )

    # Removal of Slice, need to be done after optimisation has been performed,
    # since ifm/ofm_shapes are of importance to this function
    for sg in nng.subgraphs:
        rewrite_graph.visit_graph_post_order(sg.output_tensors, arch, [], [remove_splitsliceread])
        sg.refresh_after_modification()

    # Post-processing step 2
    for idx, sg in enumerate(nng.subgraphs):
        nng.subgraphs[idx] = rewrite_graph.rewrite_graph_pre_order(nng, sg, arch, [], [fixup_quantization],)

    return nng
