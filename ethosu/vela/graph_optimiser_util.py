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
# Common functions and definitions used during the graph optimization.
from typing import Tuple

import numpy as np

from .architecture_features import Accelerator
from .data_type import DataType
from .debug_database import DebugDatabase
from .errors import UnsupportedFeatureError
from .errors import VelaError
from .operation import Op
from .operation import Operation
from .operation_util import create_avgpool_nop
from .shape4d import Shape4D
from .tensor import Tensor

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


def _avoid_nhcwb16_for_memory_only(tens):
    # check all producers/consumers to see if any op is preventing NHCWB16
    return any(op.type == Op.Memcpy for op in (tens.consumer_list + tens.ops))


# Check if non linear format can be used
def check_format_restrictions(tens: Tensor, arch):
    if tens.force_linear_format:
        return
    if len(tens.ops) < 1:
        return
    if tens.ops[0].type in (Op.Placeholder, Op.SubgraphInput, Op.Const) or any(
        cons is None for cons in tens.consumer_list
    ):
        return

    # Writing to the buffer of a variable tensor needs to be linear format
    if tens.ops[0].memory_function == Op.VariableTensorWrite:
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

    # Memory only ifm/ofm exception: DMA ops must use NHCW
    if _avoid_nhcwb16_for_memory_only(tens):
        return

    # Resize bilinear half pixel center implementation requires OFM with linear format to
    # allow stride modification in H/W dimensions.
    for op in tens.ops:
        if op.original_type == Op.ResizeBilinear and op.type == Op.DepthwiseConv2DBias:
            return

    for op in tens.consumer_list:
        if op.type == Op.ReduceSum and (
            tens.dtype == DataType.int32 or arch.accelerator_config == Accelerator.Ethos_U65_512
        ):
            # ReduceSum requires NHWC input
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

    tens.force_linear_format = False


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
    """Compute hardware padding."""
    if input_size % stride == 0:
        return max(filter_size - stride, 0)

    return max(filter_size - (input_size % stride), 0)


def set_tensor_equivalence(op: Operation, arch, nng) -> Operation:
    """Set input/output tensor equivalence to the same id for memory operations."""
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
    if op in op.ifm.consumer_list:
        op.ifm.consumer_list.remove(op)


def check_memory_only_removed(op, arch):
    if op.run_on_npu and op.type in memory_only_ops:
        # Memory only operators should have been removed
        raise VelaError(f"Memory only {op.type} op {op} expected to have been removed, still remains")


def record_optimised(op, arch):
    if op.type not in (Op.Const, Op.Placeholder):
        DebugDatabase.add_optimised(op, op)


def bypass_memory_only_ops(op, arch, nng):
    if not op.run_on_npu or op.type not in memory_only_ops:
        return op

    # Memory only operators can be completely removed if there is a one to one
    # connection. The reshape OFM can be connected to the previous op.
    #
    #                Bypassed to
    #                    --->
    #       1x6x6x10             1x6x6x10
    #         ADD                  ADD
    #          |          ------->  |
    #       1x6x6x10      |      1x20x3x6
    #        RESHAPE      |        MEAN
    #          | ---------|
    #       1x20x3x10
    #         MEAN
    #
    # In the above the ADD OFM = RESHAPE IFM is removed and replaced by
    # the RESHAPE OFM.
    #
    # Then there are two cases when bypassing is not possible. One is when
    # the IFM is produced by the CPU. This tensor must be preserved. It
    # cannot be removed from the graph. The other case is when the IFM has
    # multiple consumers, then it is not possible to just bypass the op and
    # there is a need for a DMA (nop).
    #
    #                Converts to
    #                    --->
    #       1x6x6x10                1x6x6x10
    #    -----ADD-----           -----ADD-----
    #    |           |           |           |
    # 1x6x6x10    1x6x6x10   1x6x6x10    1x6x6x10
    #  RESHAPE      MEAN       DMA OP      MEAN
    #    |                       |
    # 1x20x3x6               1x20x3x6
    #   MEAN                   MEAN
    #
    # If the DMA IFM and DMA OFM ends up in the same memory area
    # the DMA op will be removed when the cmd stream is generated.

    ifm_has_multiple_cons = len(op.ifm.consumer_list) > 1
    ifm_is_cpu_produced = any(ifm_prod is not None and not ifm_prod.run_on_npu for ifm_prod in op.ifm.ops)

    if ifm_has_multiple_cons or ifm_is_cpu_produced:
        # Convert to a memcpy op
        op.type = Op.Memcpy
        DebugDatabase.add_optimised(op, op)
    else:
        # Bypass op
        ofm = op.ofm
        ifm = op.ifm
        ofm.ops = []
        for prev_op in ifm.ops:
            prev_op.outputs = [ofm]
            ofm.ops.append(prev_op)

    return op


def convert_depthwise_to_conv(op: Operation, arch, nng) -> Operation:
    """Convert DepthwiseConv2DBias to Conv2D to allow support for DepthwiseConv2DBias ops with 'depth multiplier' > 1,
    as long as IFM depth = 1 and OFM depth is equal to the depth multiplier.
    """
    if op.type == Op.DepthwiseConv2DBias and (op.attrs["depth_multiplier"] != 1):
        ifm_shape = op.ifm_shapes[0]
        weight_tensor = op.inputs[1]
        ofm_shape = op.ofm_shapes[0]
        # Depthwise is equivalent to a single conv2d if the ifm depth is 1 and
        # the ofm depth equals the depth multipler.
        if (ifm_shape.depth == 1) and (ofm_shape.depth == op.attrs["depth_multiplier"]):
            # Change op type to Conv2d
            op.type = Op.Conv2DBias
            del op.attrs["channel_multiplier"]
            del op.attrs["depth_multiplier"]

            weight_tensor.values = np.transpose(weight_tensor.values, (0, 1, 3, 2))
            weight_tensor.set_all_shapes(list(weight_tensor.values.shape))
            DebugDatabase.add_optimised(op, op)
        else:
            raise UnsupportedFeatureError(
                f"Unsupported 'DEPTHWISE_CONV_2D' with depth_multiplier = {op.attrs['depth_multiplier']},"
                f" ifm channels = {ifm_shape.depth}, ofm channels = {ofm_shape.depth}"
            )
    return op


def create_avg_pool_for_concat(concat_op, name, ifm, ifm_shape: Shape4D, write_offset: Shape4D):
    """Creates an average pool for the given concat op/input feature map"""
    ofm = concat_op.ofm
    avgpool_op = create_avgpool_nop(name)
    avgpool_op.inputs = [ifm]
    avgpool_op.outputs = [ofm]

    avgpool_op.write_offset = write_offset
    avgpool_op.write_shape = ifm_shape
    ofm.ops.append(avgpool_op)
    avgpool_op.ifm_shapes.append(ifm_shape)
    avgpool_op.ofm_shapes.append(concat_op.ofm_shapes[0])
    avgpool_op.memory_function = Op.ConcatSliceWrite
    DebugDatabase.add_optimised(concat_op, avgpool_op)
    return avgpool_op
