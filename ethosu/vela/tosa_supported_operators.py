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
    memory_only_ops = set((Op.Reshape, Op.Transpose, Op.Concat, Op.SplitSliceRead,))

    binary_elem_wise_add_mul_sub = set((Op.Add, Op.Mul, Op.RescaleMul, Op.Sub,))

    type_conversion_ops = set((Op.Rescale,))
    relu_ops = set((Op.Clamp, Op.ReluN,))
    activation_ops = relu_ops

    npu_post_ops = activation_ops
    supported_operators = (
        mac_main_ops | type_conversion_ops | npu_post_ops | memory_only_ops | binary_elem_wise_add_mul_sub
    )

    # Supported data types
    # TODO will differ compared to TensorFlow Lite, currently set to the same
    supported_op_dtypes = set((DataType.uint8, DataType.int8, DataType.int16, DataType.int32))  # TODO add bool

    def __init__(self):
        # Setup the generic constraints. Note: the order matters
        self.generic_constraints = []
        self.generic_constraints.append(TosaSupportedOperators.constraint_tens_dtype)

        # Setup specific constraints. Note: the order matters
        self.specific_constraints = defaultdict(list)

        self.specific_constraints[Op.Transpose].append(TosaSupportedOperators.constraint_ifm_producer)

        # Depthwise Conv specific checks:
        for op_type in TosaSupportedOperators.depthwise_convolution_ops:
            self.specific_constraints[op_type].append(TosaSupportedOperators.constraint_depth_multiplier)

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

    @staticmethod
    def constraint_ifm_producer(cls, op):
        "Input must be constant data"
        valid = op.ifm.ops and op.ifm.ops[0].type == Op.Const
        return valid, "Op has ifm with non-constant data"

    # TODO duplicates TFLite_supported operators, but support for depth multiplier should be added at a later stage
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
