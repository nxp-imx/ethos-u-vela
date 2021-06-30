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
    convolution_like_ops = convolution_ops
    mac_main_ops = convolution_like_ops

    type_conversion_ops = set((Op.Rescale,))
    relu_ops = set((Op.Clamp, Op.ReluN,))
    activation_ops = relu_ops

    npu_post_ops = activation_ops
    supported_operators = mac_main_ops | type_conversion_ops | npu_post_ops

    # Supported data types
    # TODO will differ compared to TensorFlow Lite, currently set to the same
    supported_op_dtypes = set((DataType.uint8, DataType.int8, DataType.int16, DataType.int32))

    def __init__(self):
        # Setup the generic constraints. Note: the order matters
        self.generic_constraints = []
        self.generic_constraints.append(TosaSupportedOperators.constraint_tens_dtype)

        # Setup specific constraints. Note: the order matters
        self.specific_constraints = defaultdict(list)

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
