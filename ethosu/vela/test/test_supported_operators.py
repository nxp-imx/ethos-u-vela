# Copyright (C) 2020 Arm Limited or its affiliates. All rights reserved.
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
# Unit tests for support_operators
import numpy as np

from ethosu.vela.data_type import DataType
from ethosu.vela.supported_operators import SupportedOperators
from ethosu.vela.tensor import create_const_tensor
from ethosu.vela.tensor import QuantizationParameters
from ethosu.vela.tensor import Tensor
from ethosu.vela.test import testutil

support = SupportedOperators()


def create_strided_slice_op(in_shape, out_shape, start_offsets, end_offsets):
    in0 = Tensor(in_shape, DataType.uint8, "in")
    in1 = create_const_tensor("begin", [len(start_offsets)], DataType.uint8, start_offsets)
    in2 = create_const_tensor("end", [len(end_offsets)], DataType.uint8, end_offsets)
    in3 = create_const_tensor("strides", [len(end_offsets)], DataType.uint8, len(end_offsets) * [1])
    out = Tensor(out_shape, DataType.uint8, "out")
    attrs = {"ellipsis_mask": 0, "new_axis_mask": 0, "shrink_axis_mask": 0, "begin_mask": 0, "end_mask": 0}
    return testutil.create_op("StridedSlice", [in0, in1, in2, in3], out, attrs=attrs)


def create_strided_slice():
    # Creates a valid strided slice operator with some valid inputs/outputs
    op = create_strided_slice_op([1, 10, 10, 10], [1, 5, 5, 10], [127, 2, 2, 0], [0, 7, -3, 0])
    op.attrs["begin_mask"] = 1
    op.attrs["end_mask"] = 9
    assert support.is_operator_supported(op)
    return op


def test_strided_slice():
    # Tests support for StridedSlice operator
    op = create_strided_slice()
    # Setting one of new_axis_mask/shrink_axis_mask to non-zero is ok
    op.attrs["new_axis_mask"] = 2
    assert support.is_operator_supported(op)
    op = create_strided_slice()
    op.attrs["shrink_axis_mask"] = 3
    assert support.is_operator_supported(op)
    # But setting both to non-zero is not supported
    op.attrs["new_axis_mask"] = 2
    assert not support.is_operator_supported(op)
    # begin values must not be None
    op.inputs[1].values = None
    assert not support.is_operator_supported(op)
    # Unsupported strides
    op = create_strided_slice()
    op.inputs[3].values = [1, 1, 2, 1]
    assert not support.is_operator_supported(op)
    # Wrong number of input tensors
    op = create_strided_slice()
    op.add_input_tensor(op.inputs[0].clone())
    assert not support.is_operator_supported(op)
    # Unsupported ellipsis mask
    op = create_strided_slice()
    op.attrs["ellipsis_mask"] = 1
    assert not support.is_operator_supported(op)
    # Examples where end offset <= begin offset
    op = create_strided_slice()
    op.inputs[1].values = [0, 7, 2, 0]
    assert not support.is_operator_supported(op)
    op = create_strided_slice()
    op.inputs[2].values = [0, 7, 2, 0]
    assert not support.is_operator_supported(op)
    op = create_strided_slice()
    op.attrs["begin_mask"] = 0
    assert not support.is_operator_supported(op)
    op = create_strided_slice()
    op.attrs["end_mask"] = 0
    assert not support.is_operator_supported(op)


def test_constraint_tens_defined_shape():
    # Tensors cannot have None in them
    inp = Tensor([1, 8, None, 8], DataType.uint8, "in")
    out = Tensor([1, 8, 8, 8], DataType.uint8, "out")
    op = testutil.create_op("Relu", [inp], out)
    assert not support.is_operator_supported(op)


def test_constraint_tens_shapeless():
    # Shapeless input is allowed if its of a certain type:
    op = testutil.create_elemwise_op("Mul", "scalar_mul", [1, 8, 8, 8], [], [1, 8, 8, 8])
    assert support.is_operator_supported(op)
    # Shapeless output is not allowed at all:
    op = testutil.create_elemwise_op("Mul", "scalar_mul", [1, 8, 8, 8], [1, 8, 8, 8], [])
    assert not support.is_operator_supported(op)
    # Invalid shapeless input due to op type:
    inp = Tensor([], DataType.uint8, "in")
    out = Tensor([1, 8, 8, 8], DataType.uint8, "out")
    op = testutil.create_op("Relu", [inp], out)
    assert not support.is_operator_supported(op)


def test_constraint_tens_shape_size():
    # Tensors cannot be > 4D
    inp = Tensor([1, 1, 8, 8, 8], DataType.uint8, "in")
    out = Tensor([1, 1, 8, 8, 8], DataType.uint8, "out")
    op = testutil.create_op("Relu", [inp], out)
    assert not support.is_operator_supported(op)


def test_constraint_tens_dtype():
    # Tensors can only be of type uint8, int8, int16 (and int32)
    inp = Tensor([1, 8, 8, 8], DataType.float32, "in")
    out = Tensor([1, 8, 8, 8], DataType.float32, "out")
    op = testutil.create_op("Relu", [inp], out)
    assert not support.is_operator_supported(op)
    # For int32, only select op types are allowed:
    op = testutil.create_elemwise_op("Mul", "scalar_mul", [1, 8, 8, 8], [], [1, 8, 8, 8], DataType.int32)
    assert support.is_operator_supported(op)
    inp = Tensor([1, 8, 8, 8], DataType.int32, "in")
    out = Tensor([1, 8, 8, 8], DataType.int32, "out")
    op = testutil.create_op("Relu", [inp], out)
    assert not support.is_operator_supported(op)


def test_constraint_tens_dimension():
    # Tensors can only have values in the inclusive range of 1-65535
    inp = Tensor([1, 8, 8, 0], DataType.uint8, "in")
    out = Tensor([1, 8, 8, 0], DataType.uint8, "out")
    op = testutil.create_op("Relu", [inp], out)
    assert not support.is_operator_supported(op)
    inp = Tensor([1, 8, 8, 65536], DataType.uint8, "in")
    out = Tensor([1, 8, 8, 65536], DataType.uint8, "out")
    op = testutil.create_op("Relu", [inp], out)
    assert not support.is_operator_supported(op)


def test_constraint_faf():
    # Fused activation functions, if set, must be a valid op type
    inp = Tensor([1, 8, 8, 8], DataType.uint8, "in")
    out = Tensor([1, 8, 8, 8], DataType.uint8, "out")
    op = testutil.create_op("Relu", [inp], out, attrs={"fused_activation_function": "Conv2D"})
    assert not support.is_operator_supported(op)


def test_constraint_tens_quant_scale():
    # Quantization scale cannot be infinit
    op = testutil.create_elemwise_op("Mul", "scalar_mul", [1, 8, 8, 8], [], [1, 8, 8, 8])
    op.inputs[0].quantization = QuantizationParameters()
    op.inputs[0].quantization.scale_f32 = np.inf
    assert not support.is_operator_supported(op)
