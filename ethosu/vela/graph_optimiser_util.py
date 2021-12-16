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
# Common functions and definitions used during the graph optimization.
from typing import Tuple

import numpy as np

from . import lut
from .data_type import DataType
from .debug_database import DebugDatabase
from .errors import UnsupportedFeatureError
from .errors import VelaError
from .operation import Op
from .operation_util import create_avgpool_nop
from .shape4d import Shape4D
from .tensor import create_const_tensor
from .tensor import QuantizationParameters

memory_only_ops = (
    Op.Reshape,
    Op.QuantizedReshape,
    Op.Squeeze,
    Op.ExpandDims,
    Op.Identity,
)


def _avoid_nhcwb16_for_concat(tens):
    # If axis corresponds to C-dimension, NHCWB16 can only be used in the output if all the concat_start's are a
    # multiple of 16. This as, it is only then the address offset for the ofm, for all operations, will be 16 byte
    # aligned. For other values of axis the address offsets will be 16 byte aligned, as they are all based on c = 0
    # and those addresses are always 16 byte aligned due to the NHCWB16 format.
    return any(op.write_offset.depth % 16 != 0 for op in tens.ops if op.write_offset is not None)


def _avoid_nhcwb16_for_split(tens):
    # If read offset is not a multiple of 16 in the C-dimension, NHCWB16 need to be avoided in the input

    # Return True if NHCWB16 needs to be avoided
    def offset_not_aligned(read_offset):
        return read_offset is not None and (read_offset.depth % 16) != 0

    for cons_op in tens.consumer_list:
        if cons_op.ifm == tens:
            if offset_not_aligned(cons_op.read_offsets[0]):
                return True
        if cons_op.ifm2 is not None and cons_op.ifm2 == tens:
            if offset_not_aligned(cons_op.read_offsets[1]):
                return True
    return False


def _avoid_nhcwb16_for_shapes(tens):
    # check all producers/consumers to see if any op shape is preventing NHCWB16
    for cons_op in tens.consumer_list:
        if cons_op.ifm == tens:
            cons_op_shape = cons_op.ifm_shapes[0]
        elif cons_op.type.is_binary_elementwise_op() and cons_op.ifm2 == tens:
            cons_op_shape = cons_op.ifm_shapes[1]
        else:
            assert False
        if Shape4D(tens.shape) != cons_op_shape:
            return True

    for prod_op in tens.ops:
        if Shape4D(tens.shape) != prod_op.ofm_shapes[0]:
            return True

    return False


# Check if non linear format can be used
def check_format_restrictions(tens, arch):
    if len(tens.ops) < 1:
        return
    if tens.ops[0].type in (Op.Placeholder, Op.SubgraphInput, Op.Const) or any(
        cons is None for cons in tens.consumer_list
    ):
        return

    # Check if any of the producers/consumers is run on CPU
    if not all(cons.run_on_npu for cons in tens.consumer_list):
        return
    if not all(prod.run_on_npu for prod in tens.ops):
        return

    # "Concat" ofm exception:
    if _avoid_nhcwb16_for_concat(tens):
        return

    # "Split" ifm exception:
    if _avoid_nhcwb16_for_split(tens):
        return

    # Shapes checking: check all producers/consumers are NHCWB16 compatible with tens.shape
    if _avoid_nhcwb16_for_shapes(tens):
        return

    for op in tens.consumer_list:
        if op.type == Op.ReduceSum and tens.dtype == DataType.int32:
            return
        if op.type == Op.Reshape:
            # Using NHCWB16 format for a no-op reshape is only an option if subsequent
            # consumers do not also need to perform a reshape or if the OFM is going to
            # be processed by CPU operations. No-op reshape consumers with empty lists
            # (those that have no consumers, or null-consumers used as list terminators)
            # must use normal NHWC output.

            def incompatible_consumers(oper):
                if oper and oper.type == Op.Reshape:
                    for consumer in oper.outputs[0].consumer_list:
                        yield from incompatible_consumers(consumer)
                yield not oper or not oper.run_on_npu

            if not any(incompatible_consumers(op)):

                def get_rewrites(oper):
                    if oper and oper.type == Op.Reshape:
                        for consumer in oper.outputs[0].consumer_list:
                            yield from get_rewrites(consumer)
                        yield oper

                # Detect no-op reshapes by comparing their full input and output tensor shapes.
                inshape = op.ifm_shapes[0]
                compatible_shape = [(inshape == oper.ofm_shapes[0]) for oper in get_rewrites(op)]
                if not (compatible_shape and all(compatible_shape)):
                    return
            else:
                return

    tens.needs_linear_format = False


def calc_explicit_padding(input_size, stride, filter_size, pad_before, pad_after) -> Tuple[int, int]:
    """
    Based on explicit padding provided in a PAD operation, returns the corresponding hardware padding
    that provides equivalent results.
    """
    total_padding = needed_total_padding(input_size, stride, filter_size)

    # The bottom/right padding might need downward adjustment depending on stride/input size
    total_minus_before = total_padding - pad_before
    output_pad_after = pad_after
    while output_pad_after > 0 and output_pad_after % stride != total_minus_before % stride:
        output_pad_after -= 1
    return pad_before, output_pad_after


def needed_total_padding(input_size, stride, filter_size):
    out_size = (input_size + stride - 1) // stride
    needed_input = (out_size - 1) * stride + filter_size
    total_padding = max(0, needed_input - input_size)
    return total_padding


# Set input/output tensor equivalence to the same id for memory operations
def set_tensor_equivalence(op, arch, nng):
    if op.type in memory_only_ops:
        eid = op.outputs[0].equivalence_id
        for inp in op.inputs:
            inp.equivalence_id = eid
    return op


def set_ifm_ofm_op_shapes(op, arch, nng):
    if op.run_on_npu and op.type.needs_shapes():
        if op.ifm_shapes or op.ofm_shapes:
            # Shapes already set
            return op
        op.set_ifm_ofm_shapes()
    return op


def bypass_memory_only_ops(op):
    assert op.type in memory_only_ops
    ofm = op.ofm
    ifm = op.ifm

    # Check if ifm/ofm are network ifm/ofm
    ifm_is_sg_ifm = ifm.ops[0].type in (Op.Placeholder, Op.SubgraphInput, Op.Const)
    ifm_is_sg_ofm = any(ifm_cons is None for ifm_cons in ifm.consumer_list)
    ofm_is_sg_ofm = any(ofm_cons is None for ofm_cons in ofm.consumer_list)
    # Check if ifm/ofm is produced respectively consumed by CPU
    ifm_is_cpu_produced = any(ifm_prod is not None and not ifm_prod.run_on_npu for ifm_prod in op.ifm.ops)
    ofm_is_cpu_consumed = any(ofm_cons is not None and not ofm_cons.run_on_npu for ofm_cons in op.ofm.consumer_list)

    # This case should be handled prior to this function
    assert not ((ifm_is_sg_ifm or ifm_is_sg_ofm or ifm_is_cpu_produced) and (ofm_is_sg_ofm or ofm_is_cpu_consumed))

    if ofm_is_sg_ofm or ofm_is_cpu_consumed:
        # Bypassed by replacing ifm with ofm
        ofm.ops = []
        for prev_op in ifm.ops:
            prev_op.outputs = [ofm]
            ofm.ops.append(prev_op)

        # All ifm consumers need to use ofm as input
        for ifm_cons in ifm.consumer_list:
            for ifm_idx, cons_ifm in enumerate(ifm_cons.inputs):
                if cons_ifm == ifm:
                    ifm_cons.set_input_tensor(ofm, ifm_idx)
    else:
        # Bypassed by replacing ofm with ifm
        for cons in ofm.consumer_list:
            for ifm_idx, cons_ifm in enumerate(cons.inputs):
                if cons_ifm == ofm:
                    cons.set_input_tensor(ifm, ifm_idx)


def move_splitsliceread_to_consumer(op, cons_op):
    assert op.type == Op.SplitSliceRead

    if cons_op.ifm == op.ofm:
        cons_op.read_offsets[0] = op.read_offsets[0]
        cons_op.read_shapes[0] = op.read_shapes[0]
        cons_op.set_input_tensor(op.ifm, cons_op.type.info.indices.ifms[0])
        cons_op.ifm_shapes[0] = op.ifm_shapes[0]
    elif cons_op.type.is_binary_elementwise_op() and cons_op.ifm2 == op.ofm:
        cons_op.read_offsets[1] = op.read_offsets[0]
        cons_op.read_shapes[1] = op.read_shapes[0]
        cons_op.set_input_tensor(op.ifm, cons_op.type.info.indices.ifms[1])
        cons_op.ifm_shapes[1] = op.ifm_shapes[0]

    op.ofm.consumer_list.remove(cons_op)
    op.ofm.ops = []
    op.ifm.consumer_list.remove(op)


def check_memory_only_removed(op, arch):
    if op.run_on_npu and op.type in memory_only_ops:
        # Memory only operators should have been removed
        raise VelaError(f"Memory only {op.type} op {op} expected to have been removed, still remains")


def record_optimised(op, arch):
    if op.type != Op.Const:
        DebugDatabase.add_optimised(op, op)


def insert_copy_op_after_tens(tens):
    tens_cons_list_copy = tens.consumer_list.copy()

    # Create a avg_pool nop op with ifm as input
    copy_tens = tens.clone()
    copy_op = create_avgpool_nop(tens.name + "_avgpool")
    copy_op.add_input_tensor(tens)
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


def fix_sg_input_output(op, arch, nng):
    if not op.run_on_npu or op.type not in memory_only_ops:
        return op

    # For the memory only operators we want to remove, tensors are removed.
    # But in order to to do this, they cannot be outputs of the sg,
    # this need to be fixed prior to the removal.
    # Solution is to add a avgpool NOP, to maintain the original tensor.
    # This is also valid when reshape ifm/ofm is produced respectively
    # consumed by CPU

    # Check if operator ifm/ofm are sg ifm/ofm
    ifm_is_sg_ifm = op.ifm.ops[0].type in (Op.Placeholder, Op.SubgraphInput, Op.Const)
    ifm_is_sg_ofm = any(ifm_cons is None for ifm_cons in op.ifm.consumer_list)
    ofm_is_sg_ofm = any(ofm_cons is None for ofm_cons in op.ofm.consumer_list)
    # Check if ifm/ofm is produced respectively consumed by CPU
    ifm_is_cpu_produced = any(ifm_prod is not None and not ifm_prod.run_on_npu for ifm_prod in op.ifm.ops)
    ofm_is_cpu_consumed = any(ofm_cons is not None and not ofm_cons.run_on_npu for ofm_cons in op.ofm.consumer_list)

    if (ifm_is_sg_ofm or ifm_is_sg_ifm or ifm_is_cpu_produced) and (ofm_is_sg_ofm or ofm_is_cpu_consumed):
        # Both ifm and ofm need to persist, but only ifm need a copy, in order to remove the memory only operator.
        insert_copy_op_after_tens(op.ifm)

    return op


def convert_depthwise_to_conv(op, arch, nng):
    # Depthwise is equivalent to a single conv2d if the ifm depth is 1 and
    # the ofm depth equals the depth multipler.
    # If those conditions are true, then we can perform a simple
    # switch of the operator type (and weight order)

    if op.type == Op.DepthwiseConv2DBias and (op.attrs["depth_multiplier"] != 1):
        ifm_shape = op.ifm_shapes[0]
        weight_tensor = op.inputs[1]
        ofm_shape = op.ofm_shapes[0]
        if (ifm_shape.depth == 1) and (ofm_shape.depth == op.attrs["depth_multiplier"]):
            # Change op type to Conv2d
            op.type = Op.Conv2DBias
            del op.attrs["channel_multiplier"]
            del op.attrs["depth_multiplier"]

            weight_tensor.values = np.transpose(weight_tensor.values, (0, 1, 3, 2))
            weight_tensor.set_all_shapes(list(weight_tensor.values.shape))
        else:
            raise UnsupportedFeatureError(
                f"Unsupported 'DEPTHWISE_CONV_2D' with depth_multiplier = {op.attrs['depth_multiplier']},",
                f" ifm channels = {ifm_shape.depth}, ofm channels = {ofm_shape.depth}",
            )
        DebugDatabase.add_optimised(op, op)
    return op


def convert_to_lut(op, lut_values, lut_name):
    # Rewrite the operation by Add with scalar 0 + LUT activation
    ifm = op.inputs[0]
    if ifm is None:
        return op
    assert ifm.dtype.size_in_bytes() == 1
    op.type = Op.Add
    op.name = op.name + "_lut_" + lut_name
    # Mark as no-op to enable potential fusing optimizations
    op.attrs["is_nop"] = True
    # Create an input tensor containing scalar zero
    quantization = QuantizationParameters(0.0, 255.0)
    quantization.scale_f32 = ifm.quantization.scale_f32
    quantization.zero_point = 0
    tens = create_const_tensor(op.inputs[0].name + "_scalar0", [], ifm.dtype, [0], np.uint8, quantization=quantization)
    op.add_input_tensor(tens)
    op.ifm_shapes.append(Shape4D(tens.shape))  # TODO no shape?

    # The LUT must be applied without any preceding rescaling (the LUT itself performs the rescale),
    # so even if the OFM has a different scale than the IFM, the generated OFM scale instructions
    # should be the same as the IFM
    op.forced_output_quantization = ifm.quantization
    lut_tensor = lut.create_lut_tensor(op.name + "_values", lut_values, DataType.int8)
    op.set_activation_lut(lut_tensor)
    op.set_ifm_ofm_shapes()
    return op
