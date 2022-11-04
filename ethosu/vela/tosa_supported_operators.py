# SPDX-FileCopyrightText: Copyright 2021-2022 Arm Limited and/or its affiliates <open-source-office@arm.com>
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
# The TosaSupportedOperators class which is a collection of all supported operators and parameter checks.
from collections import defaultdict

from .data_type import DataType
from .operation import Op
from .supported_operators_util import docstring_format_args
from .supported_operators_util import list_formatter
from .tosa_mapping import optype_to_tosa_op_type


class TosaSupportedOperators:
    # TODO currently sparsely populated
    # Categorised lists of supported operators
    convolution_ops = set((Op.Conv2DBias,))
    depthwise_convolution_ops = set((Op.DepthwiseConv2DBias,))
    convolution_like_ops = convolution_ops | depthwise_convolution_ops

    # TODO depending on what will be committed
    max_pooling_ops = Op.op_set(Op.is_maxpool_op)
    avg_pooling_ops = Op.op_set(Op.is_avgpool_op)
    pooling_ops = max_pooling_ops | avg_pooling_ops
    fc_vector_products = set((Op.FullyConnected,))

    mac_main_ops = convolution_like_ops | pooling_ops | fc_vector_products
    memory_only_ops = set(
        (
            Op.Reshape,
            Op.Transpose,
            Op.Concat,
            Op.SplitSliceRead,
        )
    )
    binary_elem_wise_add_mul_sub = set(
        (
            Op.Add,
            Op.Mul,
            Op.Sub,
        )
    )
    elem_wise_ops = binary_elem_wise_add_mul_sub
    type_conversion_ops = set((Op.Rescale,))
    relu_ops = set(
        (
            Op.Clamp,
            Op.ReluN,
        )
    )
    activation_ops = relu_ops | set((Op.Table,))
    pad_ops = set((Op.Pad,))

    rank_unlimited_ops = set((Op.Concat, Op.Reshape, Op.Identity, Op.Pad))
    rank6_limited_ops = elem_wise_ops
    batch_enabled_ops = rank6_limited_ops | rank_unlimited_ops
    large_tens_dims_enabled_ops = batch_enabled_ops | set((Op.SplitSliceRead,))
    npu_post_ops = activation_ops

    supported_operators = (
        mac_main_ops
        | type_conversion_ops
        | npu_post_ops
        | memory_only_ops
        | elem_wise_ops
        | pad_ops
        | set((Op.Identity,))
    )

    # Supported data types
    # TODO will differ compared to TensorFlow Lite, currently set to the same
    supported_op_dtypes = set((DataType.uint8, DataType.int8, DataType.int16, DataType.int32))  # TODO add bool
    tens_dim_range = (1, 65535)  # TODO HW limitation, that is to be resolved in SW

    def __init__(self):
        # Setup the generic constraints. Note: the order matters
        self.generic_constraints = []
        self.generic_constraints.append(TosaSupportedOperators.constraint_tens_dtype)
        self.generic_constraints.append(TosaSupportedOperators.constraint_tens_dimension)  # TODO not supported yet
        self.generic_constraints.append(TosaSupportedOperators.constraint_rank)  # TODO not supported for all ops yet
        self.generic_constraints.append(TosaSupportedOperators.constraint_batch)  # TODO not supported for all ops yet

        # Setup generic constraint exceptions
        self.generic_constraints_exceptions = defaultdict(list)

        # Setup specific constraints. Note: the order matters
        self.specific_constraints = defaultdict(list)

        self.specific_constraints[Op.Transpose].append(TosaSupportedOperators.constraint_ifm_producer)
        self.specific_constraints[Op.Pad].append(TosaSupportedOperators.constraint_padding_producer)
        self.specific_constraints[Op.Table].append(TosaSupportedOperators.constraint_table_dtype)
        self.specific_constraints[Op.Table].append(TosaSupportedOperators.constraint_table_producer)

        # Depthwise Conv specific checks:
        for op_type in TosaSupportedOperators.depthwise_convolution_ops:
            self.specific_constraints[op_type].append(TosaSupportedOperators.constraint_depth_multiplier)

        # Avgpool specific checks
        for op_type in TosaSupportedOperators.avg_pooling_ops:
            self.specific_constraints[op_type].append(TosaSupportedOperators.constraint_padding)

    def is_operator_supported(self, op):
        ext_type = optype_to_tosa_op_type(op.type)
        if op.type not in TosaSupportedOperators.supported_operators:
            if op.type not in (Op.Placeholder, Op.SubgraphInput, Op.Const):
                print(f"Info: {ext_type} '{op.name}' is not a NPU op")
            return False

        for constraint in self.generic_constraints + self.specific_constraints[op.type]:
            valid, extra = constraint(op)
            if not valid:
                print(f"Warning: {ext_type} '{op.name}' is not supported on the NPU")
                print(f" - {constraint.__doc__}")
                if extra:
                    print(f"   {extra}")
                return False

        return True

    # TODO this function is the same for TensorFlow Lite, but input might differ
    @classmethod
    @docstring_format_args([list_formatter(supported_op_dtypes)])
    def constraint_tens_dtype(cls, op):
        "Tensors must be of type: {}"
        valid = True
        extra = []
        tensors = [tens for tens in op.get_ifm_ifm2_weights_ofm() if tens]
        if not tensors:
            tensors = [tens for tens in op.inputs if tens]
        for tens in tensors:
            if tens.dtype not in cls.supported_op_dtypes:
                valid = False
                extra.append(f"Tensor '{tens.name}' has data type: {tens.dtype}")
        return valid, ", ".join(extra)

    # TODO Duplicates check present for TFLite. But it is only temporarily added
    # This is for a HW limitation, that is to be resolved in SW later on
    @classmethod
    @docstring_format_args(tens_dim_range)
    def constraint_tens_dimension(self, op):
        "Tensor dimensions must be in the range [{}, {}]"
        tens_min, tens_max = self.tens_dim_range
        valid = True
        extra = []
        if op.type not in self.large_tens_dims_enabled_ops:
            tensors = [tens for tens in op.get_ifm_ifm2_weights_ofm() if tens]
            if not tensors:
                tensors = [tens for tens in op.inputs if tens]
            for tens in tensors:
                if not all(tens_min <= dim <= tens_max for dim in tens.shape):
                    valid = False
                    extra.append(f"Tensor '{tens.name}' has shape: {tens.shape}")
        return valid, ", ".join(extra)

    # TODO This is for a HW limitation, that is to be resolved in SW later on
    @classmethod
    def constraint_rank(self, op):
        "Tensor rank must be <= 6 or <= 4 depending on operator"
        valid = True
        extra = []
        if op.type not in self.rank_unlimited_ops:
            if op.type in self.rank6_limited_ops:
                rank_limit = 6
            else:
                rank_limit = 4
            tensors = [tens for tens in op.get_ifm_ifm2_weights_ofm() if tens]
            if not tensors:
                tensors = [tens for tens in op.inputs if tens]
            for tens in tensors:
                rank = len(tens.shape)
                if not rank <= rank_limit:
                    valid = False
                    extra.append(
                        f"Tensor '{tens.name}' has rank: {rank}, rank limit is currently {rank_limit}"
                        f" for op of type {op.type}"
                    )
        return valid, ", ".join(extra)

    # TODO This is for a HW limitation, that is to be resolved in SW later on
    @classmethod
    def constraint_batch(self, op):
        "If Tensor rank is 4 batch of ifms/ofm must be 1"
        valid = True
        extra = []
        if op.type not in self.batch_enabled_ops:
            tensors = [tens for tens in op.get_ifm_ifm2_ofm() if tens]
            if not tensors:
                tensors = [tens for tens in op.inputs if tens]
            for tens in tensors:
                rank = len(tens.shape)
                if rank == 4 and tens.shape[0] != 1:
                    valid = False
                    extra.append(f"Tensor '{tens.name}' has rank: 4 and N: {tens.shape[0]}")
        return valid, ", ".join(extra)

    @staticmethod
    def constraint_ifm_producer(cls, op):
        "Input must be constant data"
        valid = op.ifm.ops and op.ifm.ops[0].type == Op.Const
        return valid, "Op has ifm with non-constant data"

    @staticmethod
    def constraint_padding(op):
        # TODO Only support for when global scaling can be used.
        # That is when there is padding no padding
        "Avgpool only supported for no padding"
        top, left, _, _ = op.attrs["explicit_padding"]
        valid = top == 0 and left == 0

        return valid, "Avgpool with pad_top {top} and pad_left {left}"

    # TODO limit padding to be const data for now.
    # For TFLite it is assumed to be constant.
    @staticmethod
    def constraint_padding_producer(op):
        "Input must be constant data"
        valid = op.inputs[1].ops and op.inputs[1].ops[0].type == Op.Const
        return valid, "PAD Op with non-constant data padding"

    # TODO duplicates tflite_supported operators, but support for depth multiplier should be added at a later stage
    @staticmethod
    def constraint_depth_multiplier(op):
        "For depth multipliers > 1, IFM channels must be 1 and OFM channels must be equal to the depth multiplier"
        depth_multiplier = op.attrs.get("depth_multiplier", 1)
        if depth_multiplier > 1:
            ifm_channels = op.ifm.shape[3]
            ofm_channels = op.ofm.shape[3]
            valid = (ifm_channels == 1) and (ofm_channels == depth_multiplier)
            extra = (
                f"Op has ifm_channels={ifm_channels}, ofm_channels={ofm_channels}"
                f" and depth_multiplier={depth_multiplier}"
            )
            return valid, extra
        return True, "Op has depth_multiplier=1"

    # TODO Table operator support limited to int8 for now.
    # For TFLite it is assumed to be constant.
    @staticmethod
    def constraint_table_dtype(op):
        "Only supported is int8"
        valid = True
        tensors = [op.ifm, op.ofm, op.inputs[1]]
        for tens in tensors:
            if tens.dtype != DataType.int8:
                valid = False
        return valid, "Table operator with non int8 tensor"

    # TODO limit table to be constant data for now.
    # Can it be non-constant?
    @staticmethod
    def constraint_table_producer(op):
        "Input must be constant data"
        valid = op.inputs[1].ops and op.inputs[1].ops[0].type == Op.Const
        return valid, "Table Op with non-constant table input"
