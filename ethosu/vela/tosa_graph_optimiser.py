# SPDX-FileCopyrightText: Copyright 2021-2023 Arm Limited and/or its affiliates <open-source-office@arm.com>
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
# Early optimisation of the TOSA based network graph, using the rewrite_graph module to do the traversal of the graph.
import numpy as np

from . import rewrite_graph
from .data_type import DataType
from .debug_database import DebugDatabase
from .graph_optimiser_util import bypass_memory_only_ops
from .graph_optimiser_util import calc_explicit_padding
from .graph_optimiser_util import convert_depthwise_to_conv
from .graph_optimiser_util import move_splitsliceread_to_consumer
from .graph_optimiser_util import needed_total_padding
from .graph_optimiser_util import set_ifm_ofm_op_shapes
from .graph_optimiser_util import set_tensor_equivalence
from .lut import convert_to_lut
from .operation import ExplicitScaling
from .operation import Op
from .operation import RoundingMode
from .operation_util import create_add_nop
from .operation_util import create_avgpool_nop
from .operation_util import create_pad_nop
from .shape4d import Shape4D
from .tensor import create_const_tensor
from .tensor import create_equivalence_id
from .tensor import shape_num_elements
from .tensor import Tensor


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

        op.rounding_mode = RoundingMode.HalfUp
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


def insert_add_copy_for_const(op, ifm_ofm_shape):
    assert op.type == Op.Const
    ofm = op.ofm
    copy_tens = ofm.clone()
    op.set_output_tensor(copy_tens)

    name = ofm.name + "_add"
    ifm2 = create_const_tensor(
        name + "_zero_scalar",
        [1],
        copy_tens.dtype,
        [0],
        quantization=copy_tens.quantization,
    )
    copy_op = create_add_nop(name)
    copy_op.add_input_tensor(copy_tens)
    copy_op.add_input_tensor(ifm2)
    copy_op.set_output_tensor(ofm)
    copy_op.ifm_shapes.append(ifm_ofm_shape)
    copy_op.ifm_shapes.append(Shape4D(ifm2.shape))
    copy_op.ofm_shapes.append(ifm_ofm_shape)
    copy_op.run_on_npu = True

    DebugDatabase.add_optimised(op, copy_op)


# TODO can we change to add for both TFLite and TOSA?
def insert_add_copy_op_after_tens(tens, ifm_ofm_shape):
    tens_cons_list_copy = tens.consumer_list.copy()
    copy_tens = tens.clone()

    name = tens.name + "_add"
    ifm2 = create_const_tensor(
        name + "_zero_scalar",
        [1],
        copy_tens.dtype,
        [0],
        quantization=copy_tens.quantization,
    )
    copy_op = create_add_nop(name)
    copy_op.add_input_tensor(tens)
    copy_op.add_input_tensor(ifm2)
    copy_op.set_output_tensor(copy_tens)
    copy_op.ifm_shapes.append(ifm_ofm_shape)
    copy_op.ifm_shapes.append(Shape4D(ifm2.shape))
    copy_op.ofm_shapes.append(ifm_ofm_shape)
    copy_op.run_on_npu = True

    # Set copy_ifm consumers
    for tens_cons in tens_cons_list_copy:
        if tens_cons is not None:
            for ifm_idx, cons_inp in enumerate(tens_cons.inputs):
                if cons_inp == tens:
                    tens_cons.set_input_tensor(copy_tens, ifm_idx)

    DebugDatabase.add_optimised(tens.ops[0], copy_op)


def get_shape_for_copy_op(shape):
    # remove dimensions that are set to 1
    new_shape = []
    for dim in shape:
        if dim != 1:
            new_shape.append(dim)
    if not new_shape:
        new_shape = [1]

    rank = len(new_shape)
    if rank > 3:
        # Reshape so that batch becomes 1, by moving elements to H dimension
        n = rank - 2
        h = 1
        for i in range(n):
            h *= shape[i]
        new_shape = Shape4D(new_shape[n:]).with_height(h)
    else:
        new_shape = Shape4D(new_shape)
    return new_shape


def fix_sg_input_output_tosa(op, arch, nng):

    if op.type == Op.Const and any(ofm_cons is None for ofm_cons in op.ofm.consumer_list):
        # Const operator with sg output, insert copy op before the ofm
        new_shape = get_shape_for_copy_op(op.ofm.shape.copy())
        insert_add_copy_for_const(op, new_shape)
    elif op.run_on_npu and op.type in (Op.Reshape, Op.Identity):
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
            # Both ifm and ofm need to persist, but only ifm need a copy, in order to remove the Operator
            # Decide on ifm/ofm shapes for the copy op based on ifm
            new_shape = get_shape_for_copy_op(op.ifm.shape.copy())
            insert_add_copy_op_after_tens(op.ifm, new_shape)
    return op


def create_add_for_concat(concat_op, name, ifm, ifm_shape: Shape4D, write_offset: Shape4D):
    """Creates an add op for the given concat op/input feature map"""
    ofm = concat_op.ofm
    ifm2 = create_const_tensor(name + "_zero_scalar", [1], ofm.dtype, [0], quantization=ofm.quantization)
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
            ifm2 = create_const_tensor(name + "_zero_scalar", [1], ofm.dtype, [0], quantization=ofm.quantization)
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


def rewrite_concat(op):
    if not op.run_on_npu or not op.type == Op.Concat:
        return

    offset = 0
    inputs = op.inputs
    axis_4D = op.attrs["axis4D"]

    for idx, inp in enumerate(inputs):
        write_offset = [0, 0, 0, 0]
        write_offset[axis_4D] = offset
        concat_end = offset + op.ifm_shapes[idx][axis_4D]
        create_add_for_concat(op, op.name + str(idx) + "_add", inp, op.ifm_shapes[idx], Shape4D.from_list(write_offset))
        offset = concat_end
    assert op.ofm_shapes[0][axis_4D] == offset


def remove_memory_ops(op, arch):
    if op.run_on_npu and op.type in (Op.Reshape, Op.Identity):
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
            rounding_mode = RoundingMode.TFLite
        else:
            rounding_mode = RoundingMode.HalfUp

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


def convert_pad_in_width(op):
    """
    Rewrites PAD operator to an add that copies the IFM to the OFM
    + up to 4 add operators that fill the OFM with zeros at the borders.
    """
    assert op.type == Op.Pad
    assert op.ifm_shapes[0] is not None and op.ofm_shapes[0] is not None
    ifm = op.ifm
    ofm = op.ofm
    ifm_shape = op.ifm_shapes[0]
    ofm.ops = []
    ofm_shape = op.ofm_shapes[0]

    padding = op.inputs[1].values
    left, right = padding[-2]

    # Add op that copies IFM to the right place inside the OFM
    shp0 = Shape4D(0, 0, 0, 0)
    add_op = create_add_for_concat(op, op.name + "_main", ifm, ifm_shape, shp0.with_width(left))
    add_op.activation = op.activation

    quant = ofm.quantization
    pad_value = ifm.quantization.zero_point
    ifm.quantization.zero_point = 0
    if left > 0:
        shape = Shape4D(1, ifm_shape.height, left, ofm_shape.depth)
        zero_tens = create_const_tensor(
            op.name + "_left", shape.as_list(), ofm.dtype, shape.elements() * [pad_value], quantization=quant
        )
        zero_tens.equivalence_id = create_equivalence_id(tuple(zero_tens.values))
        create_add_for_concat(op, op.name + "_left", zero_tens, shape, shp0)
    if right > 0:
        shape = Shape4D(1, ifm_shape.height, right, ofm_shape.depth)
        zero_tens = create_const_tensor(
            op.name + "_right", shape.as_list(), ofm.dtype, shape.elements() * [pad_value], quantization=quant
        )
        zero_tens.equivalence_id = create_equivalence_id(tuple(zero_tens.values))
        create_add_for_concat(op, op.name + "_right", zero_tens, shape, shp0.with_width(ofm_shape.width - right))

    op.type = Op.ConcatTFLite
    return add_op


def convert_table_to_lut(op, arch, nng):
    # Converts table op to a no-op + LUT
    if op.type is not Op.Table:
        return op

    table = op.inputs[1]
    op.inputs.remove(table)
    op.set_ifm_ofm_shapes()

    return convert_to_lut(op, table.values, "table")


def decompose_elem_tensors_hwc(op):
    """
    Decomposes elementwise op if any of the ifm(s)/ofm are to large in any dimension to be handled by the NPU
    """
    max_t_size = 65535
    ofm_shape = op.write_shape if op.write_shape is not None else op.ofm_shapes[0]
    ifm_shape = op.read_shapes[0] if op.read_shapes[0] is not None else op.ifm_shapes[0]
    ifm2_shape = op.ifm_shapes[1] if op.ifm_shapes[1] else None
    ifm2_shape = op.read_shapes[1] if op.read_shapes[1] is not None else ifm2_shape
    limit_shape = Shape4D(1, max_t_size, max_t_size, max_t_size)

    if any(dim_size > max_t_size for dim_size in ofm_shape.as_list()):
        ofm_split = ofm_shape.floordiv_const(max_t_size).add(1, 1, 1, 1)

        for height in range(ofm_split.height):
            for width in range(ofm_split.width):
                for depth in range(ofm_split.depth):
                    ofm_offset = Shape4D(0, height * max_t_size, width * max_t_size, depth * max_t_size)
                    ofm_part_shape = ofm_shape.clip(ofm_offset, limit_shape)
                    ofm_cut = (ofm_offset, ofm_part_shape)

                    ifm_d = depth * max_t_size if ifm_shape.depth == ofm_shape.depth else 0
                    ifm_w = width * max_t_size if ifm_shape.width == ofm_shape.width else 0
                    ifm_h = height * max_t_size if ifm_shape.height == ofm_shape.height else 0
                    ifm_offset = Shape4D(0, ifm_h, ifm_w, ifm_d)
                    ifm_part_shape = ifm_shape.clip(ifm_offset, limit_shape)
                    ifm_cut = (ifm_offset, ifm_part_shape)

                    if ifm2_shape is not None:
                        ifm2_d = depth * max_t_size if ifm2_shape.depth == ofm_shape.depth else 0
                        ifm2_w = width * max_t_size if ifm2_shape.width == ofm_shape.width else 0
                        ifm2_h = height * max_t_size if ifm2_shape.height == ofm_shape.height else 0
                        ifm2_offset = Shape4D(0, ifm2_h, ifm2_w, ifm2_d)
                        ifm2_part_shape = ifm2_shape.clip(ifm2_offset, limit_shape)
                        ifm2_cut = (ifm2_offset, ifm2_part_shape)
                    else:
                        ifm2_cut = (None, None)

                    create_elem_part_op(op, ifm_cut, ifm2_cut, ofm_cut)
        op.ofm.ops.remove(op)
        op.ifm.consumer_list.remove(op)
        if op.ifm2 is not None:
            op.ifm2.consumer_list.remove(op)
    return


def create_elem_part_op(op, ifm_cut, ifm2_cut, ofm_cut):
    part_op = op.clone()
    ifm_read_offset = op.read_offsets[0] if op.read_offsets[0] is not None else Shape4D(0, 0, 0, 0)
    ofm_write_offset = op.write_offset if op.write_offset is not None else Shape4D(0, 0, 0, 0)
    ifm_offset, ifm_shape = ifm_cut
    ofm_offset, ofm_shape = ofm_cut

    part_op.read_offsets[0] = ifm_read_offset + ifm_offset
    part_op.read_shapes[0] = ifm_shape
    part_op.write_offset = ofm_write_offset + ofm_offset
    part_op.write_shape = ofm_shape
    part_op.ifm_shapes = op.ifm_shapes.copy()
    part_op.ofm_shapes = op.ofm_shapes.copy()
    part_op.ifm.consumer_list.append(part_op)
    op.ofm.ops.append(part_op)

    ifm2_offset, ifm2_shape = ifm2_cut
    if ifm2_offset:
        ifm2_read_offset = op.read_offsets[1] if op.read_offsets[1] is not None else Shape4D(0, 0, 0, 0)
        part_op.read_offsets[1] = ifm2_read_offset + ifm2_offset
        part_op.read_shapes[1] = ifm2_shape
        part_op.ifm2.consumer_list.append(part_op)

    return part_op


def get_nhwc_stride(shape):
    stride_x = shape.depth
    stride_y = shape.width * stride_x
    stride_n = shape.height * stride_y
    return Shape4D(stride_n, stride_y, stride_x, 1)


def pad_to_rank(shape, rank):
    """
    Pads a shape to the given rank
    """
    while len(shape) < rank:
        shape = [1] + shape

    return shape


def get_elem_shapes_removed_singles(op):
    """
    Returns the shapes of ifm(s)/ofms after removing all the dimensions that are 1 for all ifm(s)/ofm
    """
    binary = op.ifm2 is not None
    ofm_shape = op.ofm_shapes[0].as_list() if len(op.ofm_shapes) > 0 else op.ofm.shape
    ifm_shape = op.ifm_shapes[0].as_list() if len(op.ifm_shapes) > 0 else op.ifm.shape
    if binary:
        ifm2_shape = op.ifm_shapes[1].as_list() if len(op.ofm_shapes) else op.ifm2.shape

    rank = max(len(ofm_shape), len(ifm_shape), len(ifm2_shape) if binary else 0)
    ofm_shape = pad_to_rank(ofm_shape, rank)
    ifm_shape = pad_to_rank(ifm_shape, rank)
    if binary:
        ifm2_shape = pad_to_rank(ifm2_shape, rank)

    new_ofm_shape = []
    new_ifm_shape = []
    new_ifm2_shape = []
    for idx in range(rank):
        if ofm_shape[idx] != 1:
            new_ofm_shape.append(ofm_shape[idx])
            new_ifm_shape.append(ifm_shape[idx])
            if binary:
                new_ifm2_shape.append(ifm2_shape[idx])

    if new_ofm_shape == []:
        new_ofm_shape = [1]
        new_ifm_shape = [1]
        new_ifm2_shape = [1] if binary else None

    return new_ofm_shape, new_ifm_shape, new_ifm2_shape


def decomp_dims_elementwise(op):
    """
    Decompose elementwise ops with Rank > 3 (H,W,D).
    If Rank > 3, all the dimensions above H are viewed as the N dimension.
    the elementwise operation will be decomposed to N (of ofm) elementwise operations.
    By reading and writing with offsets from/to the ifm(s)/ofm.
    Note: Broadcast need to be handled for binary elementwise ops, and TOSA allowes for broadcast by both ifm and ifm2
    """

    ifm = op.ifm
    ifm2 = op.ifm2
    ofm = op.ofm
    binary = op.ifm2 is not None

    # Remove dimensions that are all 1
    new_ofm_shape, new_ifm_shape, new_ifm2_shape = get_elem_shapes_removed_singles(op)
    rank = len(new_ofm_shape)

    if rank > 3:
        n = rank - 3
        ofm_decomp_shape = Shape4D(new_ofm_shape[0:n])
        ofm_decomp_stride = get_nhwc_stride(ofm_decomp_shape)
        ofm_part_shape = Shape4D(new_ofm_shape[n:])
        op.ofm_shapes.append(Shape4D([ofm_decomp_shape.elements()] + new_ofm_shape[n:]))

        if binary:
            ifm_decomp_shape = Shape4D(new_ifm_shape[0:n])
            ifm2_decomp_shape = Shape4D(new_ifm2_shape[0:n])
            ifm_decomp_stride = get_nhwc_stride(ifm_decomp_shape)
            ifm2_decomp_stride = get_nhwc_stride(ifm2_decomp_shape)
            ifm_part_shape = Shape4D(new_ifm_shape[n:])
            ifm2_part_shape = Shape4D(new_ifm2_shape[n:])
            op.ifm_shapes.append(Shape4D([ifm_decomp_shape.elements()] + new_ifm_shape[n:]))
            op.ifm_shapes.append(Shape4D([ifm2_decomp_shape.elements()] + new_ifm2_shape[n:]))
        else:
            op.ifm_shapes.append(Shape4D([ofm_decomp_shape.elements()] + new_ofm_shape[n:]))

        op_list = []
        for height in range(ofm_decomp_shape.height):
            for width in range(ofm_decomp_shape.width):
                for depth in range(ofm_decomp_shape.depth):
                    ofm_offset = Shape4D(0, height, width, depth)
                    ofm_offset = Shape4D(ofm_offset.dot_prod(ofm_decomp_stride), 0, 0, 0)
                    ofm_cut = (ofm_offset, ofm_part_shape)

                    if binary:
                        ifm_d = depth if ifm_decomp_shape.depth == ofm_decomp_shape.depth else 0
                        ifm_w = width if ifm_decomp_shape.width == ofm_decomp_shape.width else 0
                        ifm_h = height if ifm_decomp_shape.height == ofm_decomp_shape.height else 0
                        ifm_offset = Shape4D(0, ifm_h, ifm_w, ifm_d)
                        ifm_offset = Shape4D(ifm_offset.dot_prod(ifm_decomp_stride), 0, 0, 0)
                        ifm_cut = (ifm_offset, ifm_part_shape)

                        ifm2_d = depth if ifm2_decomp_shape.depth == ofm_decomp_shape.depth else 0
                        ifm2_w = width if ifm2_decomp_shape.width == ofm_decomp_shape.width else 0
                        ifm2_h = height if ifm2_decomp_shape.height == ofm_decomp_shape.height else 0
                        ifm2_offset = Shape4D(0, ifm2_h, ifm2_w, ifm2_d)
                        ifm2_offset = Shape4D(ifm2_offset.dot_prod(ifm2_decomp_stride), 0, 0, 0)
                        ifm2_cut = (ifm2_offset, ifm2_part_shape)
                        op_list.append(create_elem_part_op(op, ifm_cut, ifm2_cut, ofm_cut))
                    else:
                        op_list.append(create_elem_part_op(op, ofm_cut, None, ofm_cut))

        ofm.ops.remove(op)
        ifm.consumer_list.remove(op)
        if binary:
            ifm2.consumer_list.remove(op)

        return op_list
    else:
        op.ofm_shapes.append(Shape4D(new_ofm_shape))
        op.ifm_shapes.append(Shape4D(new_ifm_shape))
        op.ifm_shapes.append(Shape4D(new_ifm2_shape))

    return [op]


def decomp_elementwise(tens, arch, nng):
    """
    Decompose elementwise ops with Rank > 3 (H,W,C).
    Decompose size of tensors exceeding NPU max size
    """
    tens_ops = tens.ops.copy()
    for op in tens_ops:
        if op.type.is_elementwise_op():
            decomp_list = decomp_dims_elementwise(op)
            for part_op in decomp_list:
                decompose_elem_tensors_hwc(part_op)
    return tens


def reshape_concat_shape(shape, rank, axis):
    new_h = 1
    for i in range(axis):
        new_h *= shape[i]
    new_c = 1
    for i in range(axis + 1, rank):
        new_c *= shape[i]
    if axis == (rank - 1):
        new_shape = [new_h, shape[axis], 1]
    else:
        new_shape = [new_h, shape[axis], new_c]
    return new_shape


def reshape_concat(op):
    """
    Reshapes concat ops with Rank > 3 (H,W,C).
    """
    ofm = op.ofm
    rank = len(ofm.shape)
    axis = op.attrs["axis"]
    if axis < 0:
        axis += rank

    if rank > 3:
        # Reshape so that axis in to be concatenated is the W dimension
        # Reshape inputs
        for inp in op.inputs:
            new_shape = reshape_concat_shape(inp.shape, rank, axis)
            op.ifm_shapes.append(Shape4D(new_shape))
        # Reshape output
        new_shape = reshape_concat_shape(ofm.shape, rank, axis)
        op.ofm_shapes.append(Shape4D(new_shape))
        op.attrs["axis4D"] = 2
    else:
        for inp in op.inputs:
            op.ifm_shapes.append(Shape4D(inp.shape))
        op.ofm_shapes.append(Shape4D(ofm.shape))
        op.attrs["axis4D"] = axis + (4 - rank)


def decomp_rewrite_concat(tens, arch, nng):
    """
    Decompose concat ops with Rank > 3 (H,W,C).
    Rewrite of concat to elementwise operations
    """
    if len(tens.ops) == 1 and tens.ops[0].type == Op.Concat:
        op = tens.ops[0]

        reshape_concat(op)
        rewrite_concat(op)

        op.ofm.ops.remove(op)
        for inp in op.inputs:
            inp.consumer_list.remove(op)

    return tens


def decomp_rewrite_pad(op, arch):
    """
    Decomposition of pad to elementwise operations:
    For each dimension that needs padding:
    -Create a new PAD operator for each dimension to be added
     Ifm/ofm are reshape so this is the width dimension is to be padded
     (rank for each is 3)
    -Rewrite the the new PAD operator so there is:
    -1 Add operator for copying the data
    -1 Add operator for each left/right to be padded
    """
    # TODO several things would be possible to optimize
    # For instance there are cases when it should be possible to pad 2
    # dimensions at the same time.
    if op.type == Op.Pad:
        ofm_elements = shape_num_elements(op.ofm.shape)
        padding = op.inputs[1].values

        rank = len(op.ifm.shape)
        next_ifm = op.ifm
        next_ifm_shape = next_ifm.shape.copy()

        first_pad_rewrite_op = None
        ifm_quant = op.ifm.quantization.clone()

        for dim in range(padding.shape[0]):
            # Check if padding is to be applied in this dimension
            dim_pad = padding[dim]
            if not (dim_pad == 0).all():
                # Reshape so that width dimension is to be padded
                new_ifm_shape = reshape_concat_shape(next_ifm_shape, rank, dim)
                new_pad_input = np.zeros((4, 2), dtype=np.int32)
                new_pad_input[2] = dim_pad

                pad_op = create_pad_nop(f"{op.name}_dim_{dim}")
                pad_op.add_input_tensor(next_ifm)
                new_pad_tens = op.inputs[1].clone("_dim_{dim}")

                name = op.inputs[1].name + f"_dim_{dim}"
                new_pad_tens = create_const_tensor(name, list(new_pad_input.shape), DataType.int32, new_pad_input)
                pad_op.add_input_tensor(new_pad_tens)

                new_ofm_shape = new_ifm_shape.copy()
                new_ofm_shape[-2] = new_ofm_shape[-2] + dim_pad.sum()
                next_ifm_shape[dim] = next_ifm_shape[dim] + dim_pad.sum()

                if Shape4D(new_ofm_shape).elements() == ofm_elements:
                    # Last one, use op.ofm
                    ofm = op.ofm
                else:
                    # add a new ofm Tensor
                    ofm = Tensor(new_ofm_shape, op.ofm.dtype, f"{pad_op.name}_tens")
                    ofm.quantization = ifm_quant.clone()

                pad_op.set_output_tensor(ofm)
                pad_op.ifm_shapes.append(Shape4D(new_ifm_shape))
                pad_op.ofm_shapes.append(Shape4D(new_ofm_shape))
                DebugDatabase.add_optimised(op, pad_op)
                next_ifm = ofm

                # Rewrite the pad op
                converted_pad_op = convert_pad_in_width(pad_op)
                first_pad_rewrite_op = converted_pad_op
            else:
                # Change to Identity operation (will be removed)
                op.type = Op.Identity

        if first_pad_rewrite_op:
            assert op.ofm.shape == next_ifm_shape
            for inp in op.inputs:
                inp.consumer_list.remove(op)
            return first_pad_rewrite_op

    return op


def fixup_quantization(op, arch, nng):
    if op.ifm and op.ifm.quantization.zero_point is None:
        op.ifm.quantization.zero_point = 0
    if op.ifm2 and op.ifm2.quantization.zero_point is None:
        op.ifm2.quantization.zero_point = 0
    if not op.forced_output_quantization:
        if op.ofm and op.ofm.quantization and op.ofm.quantization.zero_point is None:
            op.ofm.quantization.zero_point = 0
    return op


def supported_operator_check(op, arch, nng):
    op.run_on_npu = arch.tosa_supported_operators.is_operator_supported(op)
    assert op.run_on_npu or op.type in (Op.Placeholder, Op.SubgraphInput, Op.Const)
    return op


def tosa_optimise_graph(nng, arch):

    # TODO the supported operator checking need to be split in semantic and HW checks
    for idx, sg in enumerate(nng.subgraphs):
        nng.subgraphs[idx] = rewrite_graph.rewrite_graph_pre_order(
            nng,
            sg,
            arch,
            [],
            [supported_operator_check],
            rewrite_unsupported=False,
        )

    # Decomposing and rewrite of concat
    for idx, sg in enumerate(nng.subgraphs):
        nng.subgraphs[idx] = rewrite_graph.rewrite_graph_pre_order(
            nng, sg, arch, [decomp_rewrite_concat], [], rewrite_unsupported=False
        )

    # Decomposing of pad
    for idx, sg in enumerate(nng.subgraphs):
        rewrite_graph.visit_graph_post_order(sg.output_tensors, arch, [], [decomp_rewrite_pad])
        sg.refresh_after_modification()

    # Handle sg input output
    for idx, sg in enumerate(nng.subgraphs):
        nng.subgraphs[idx] = rewrite_graph.rewrite_graph_pre_order(
            nng,
            sg,
            arch,
            [],
            [fix_sg_input_output_tosa],
            rewrite_unsupported=True,
        )

    # Removal of reshapes
    for sg in nng.subgraphs:
        rewrite_graph.visit_graph_post_order(sg.output_tensors, arch, [], [remove_memory_ops])
        sg.refresh_after_modification()

    # Decomposing of elementwise
    for idx, sg in enumerate(nng.subgraphs):
        nng.subgraphs[idx] = rewrite_graph.rewrite_graph_pre_order(
            nng, sg, arch, [decomp_elementwise], [], rewrite_unsupported=False
        )

    for idx, sg in enumerate(nng.subgraphs):
        nng.subgraphs[idx] = rewrite_graph.rewrite_graph_pre_order(
            nng,
            sg,
            arch,
            [],
            [set_ifm_ofm_op_shapes],
            rewrite_unsupported=False,
        )

    # Removal of Transpose
    for idx, sg in enumerate(nng.subgraphs):
        nng.subgraphs[idx] = rewrite_graph.rewrite_graph_pre_order(
            nng,
            sg,
            arch,
            [],
            [remove_const_transpose],
            rewrite_unsupported=False,
        )

    # TODO, when and where to best handle calc_scaling_avgpool
    for idx, sg in enumerate(nng.subgraphs):
        nng.subgraphs[idx] = rewrite_graph.rewrite_graph_pre_order(
            nng,
            sg,
            arch,
            [],
            [calc_scaling_avgpool],
            rewrite_unsupported=False,
        )

    # Rewite Operators step
    op_rewrite_list = [set_tensor_equivalence, rewrite_rescale, convert_depthwise_to_conv, convert_table_to_lut]

    for idx, sg in enumerate(nng.subgraphs):
        nng.subgraphs[idx] = rewrite_graph.rewrite_graph_pre_order(
            nng,
            sg,
            arch,
            [],
            op_rewrite_list,
            rewrite_unsupported=False,
        )

    # Post-processing step 1
    for idx, sg in enumerate(nng.subgraphs):
        nng.subgraphs[idx] = rewrite_graph.rewrite_graph_pre_order(
            nng,
            sg,
            arch,
            [],
            [rewrite_activation, add_padding_fields],
        )

    # Removal of Slice, need to be done after optimisation has been performed,
    # since ifm/ofm_shapes are of importance to this function
    for sg in nng.subgraphs:
        rewrite_graph.visit_graph_post_order(sg.output_tensors, arch, [], [remove_splitsliceread])
        sg.refresh_after_modification()

    # Post-processing step 2
    for idx, sg in enumerate(nng.subgraphs):
        nng.subgraphs[idx] = rewrite_graph.rewrite_graph_pre_order(
            nng,
            sg,
            arch,
            [],
            [fixup_quantization],
        )

    return nng
