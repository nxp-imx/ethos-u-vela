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
from .data_type import DataType
from .debug_database import DebugDatabase
from .errors import VelaError
from .operation import Op
from .shape4d import Shape4D
from .tensor import check_quantized_tens_scaling_equal


memory_only_ops = (
    Op.Reshape,
    Op.Squeeze,
)


def _avoid_nhcwb16_for_concat(tens):
    # If axis corresponds to C-dimension, NHCWB16 can only be used in the output if all the concat_start's are a
    # multiple of 16. This as, it is only then the address offset for the ofm, for all operations, will be 16 byte
    # aligned. For other values of axis the address offsets will be 16 byte aligned, as they are all based on c = 0
    # and those addresses are always 16 byte aligned due to the NHCWB16 format.
    return any(op.write_offset.depth % 16 != 0 for op in tens.ops if op.write_offset is not None)


def _avoid_nhcwb16_for_split(tens):
    # If read offset is not a multiple of 16 in the C-dimension, NHCWB16 need to be avoided in the input
    for cons_op in tens.consumer_list:
        if cons_op.ifm == tens:
            read_offset = cons_op.read_offsets[0]
        elif cons_op.type.is_binary_elementwise_op() and cons_op.ifm2 == tens:
            read_offset = cons_op.read_offsets[1]
        else:
            assert False
        if read_offset is not None and (read_offset[-1] % 16) != 0:
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


def check_reshapes(op, arch):
    if op.run_on_npu and op.type == Op.Reshape:
        ofm = op.ofm

        if check_quantized_tens_scaling_equal(op.ifm, ofm):
            # Reshape should have been removed
            raise VelaError(f"Reshape op {op} expected to have been removed, still remains")


def record_optimised(op, arch):
    if op.type != Op.Const:
        DebugDatabase.add_optimised(op, op)
