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
from ethosu.vela.operation import Op
from ethosu.vela.supported_operators import SupportedOperators
from ethosu.vela.tensor import create_const_tensor
from ethosu.vela.tensor import QuantizationParameters
from ethosu.vela.tensor import Tensor
from ethosu.vela.test import testutil

support = SupportedOperators()


def create_strided_slice_op(in_shape, out_shape, start_offsets, end_offsets):
    qp = QuantizationParameters()
    in0 = Tensor(in_shape, DataType.uint8, "in")
    in0.quantization = qp
    in1 = create_const_tensor("begin", [len(start_offsets)], DataType.uint8, start_offsets, quantization=qp)
    in2 = create_const_tensor("end", [len(end_offsets)], DataType.uint8, end_offsets, quantization=qp)
    in3 = create_const_tensor("strides", [len(end_offsets)], DataType.uint8, len(end_offsets) * [1], quantization=qp)
    out = Tensor(out_shape, DataType.uint8, "out")
    out.quantization = qp
    attrs = {"ellipsis_mask": 0, "new_axis_mask": 0, "shrink_axis_mask": 0, "begin_mask": 0, "end_mask": 0}
    return testutil.create_op(Op.StridedSlice, [in0, in1, in2, in3], out, attrs=attrs)


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
    op = testutil.create_op_with_quant_tensors(Op.Relu, [1, 8, None, 8], [1, 8, 8, 8])
    assert not support.is_operator_supported(op)


def test_constraint_tens_output_shapeless():
    # Shapeless output is not allowed at all:
    op = testutil.create_elemwise_op(Op.Mul, "op", [1, 8, 8, 8], [1, 8, 8, 8], [])
    assert not support.is_operator_supported(op)


def test_constraint_tens_input_shapeless():
    # Shapeless input is allowed if its of a certain type:
    op = testutil.create_elemwise_op(Op.Mul, "op", [1, 8, 8, 8], [], [1, 8, 8, 8])
    assert support.is_operator_supported(op)
    # Invalid shapeless input due to op type:
    op = testutil.create_op_with_quant_tensors(Op.Relu, [], [1, 8, 8, 8])
    assert not support.is_operator_supported(op)


def test_constraint_tens_shape_size():
    # Tensors cannot be > 4D
    op = testutil.create_op_with_quant_tensors(Op.Relu, [1, 1, 8, 8, 8], [1, 1, 8, 8, 8])
    assert not support.is_operator_supported(op)


def test_constraint_tens_dtype():
    # Tensors can only be of type uint8, int8, int16 and int32
    op = testutil.create_op_with_quant_tensors(Op.Relu, [1, 8, 8, 8], [1, 8, 8, 8], datatype=DataType.float32)
    assert not support.is_operator_supported(op)


def test_constraint_tens_int32_ops():
    # For int32, only select op types are allowed:
    op = testutil.create_elemwise_op(Op.Mul, "op", [1, 8, 8, 8], [], [1, 8, 8, 8], datatype=DataType.int32)
    assert support.is_operator_supported(op)
    op = testutil.create_op_with_quant_tensors(Op.Relu, [1, 8, 8, 8], [1, 8, 8, 8], datatype=DataType.int32)
    assert not support.is_operator_supported(op)


def test_constraint_tens_dimension():
    # Tensors can only have values in the inclusive range of 1-65535
    op = testutil.create_op_with_quant_tensors(Op.Relu, [1, 8, 8, 0], [1, 8, 8, 65536])
    assert not support.is_operator_supported(op)


def test_constraint_tens_quant_none_check():
    # Tensors must have quantization parameters
    op = testutil.create_elemwise_op(Op.Mul, "op", [1, 8, 8, 8], [], [1, 8, 8, 8], ifm2_quant=None)
    assert not support.is_operator_supported(op)


def test_constraint_tens_quant_scale():
    # Quantization scale cannot be infinit
    qp = QuantizationParameters()
    qp.scale_f32 = np.inf
    op = testutil.create_elemwise_op(Op.Mul, "op", [1, 8, 8, 8], [], [1, 8, 8, 8], ifm_quant=qp)
    assert not support.is_operator_supported(op)


def test_constraint_faf():
    # Fused activation functions, if set, must be a valid op type
    op = testutil.create_op_with_quant_tensors(Op.Relu, [1, 8, 8, 8], [1, 8, 8, 8])
    op.activation = Op.Conv2D
    assert not support.is_operator_supported(op)


def test_constraint_conv_pass():
    # First test a simple conv passes
    op = testutil.create_op_with_quant_tensors(Op.Conv2D, [1, 1, 1, 1], [1, 1, 1, 1], weights_shape=[1, 1, 1, 1])
    op.attrs = {"stride_w": 1, "stride_h": 1}
    assert support.is_operator_supported(op)


def test_constraint_stride_type():
    # Stride width and height must be integer types
    op = testutil.create_op_with_quant_tensors(Op.Conv2D, [1, 8, 8, 8], [1, 8, 8, 8])
    op.attrs = {"stride_w": 1.5, "stride_h": "1"}
    assert not support.is_operator_supported(op)


def test_constraint_stride_range():
    # Stride width and height must lie within a certain range
    op = testutil.create_op_with_quant_tensors(Op.Conv2D, [1, 8, 8, 8], [1, 8, 8, 8])
    op.attrs = {"stride_w": 0, "stride_h": 20}
    assert not support.is_operator_supported(op)


def test_constraint_dilation_type():
    # Dilation width and height must be integer types
    op = testutil.create_op_with_quant_tensors(Op.Conv2D, [1, 8, 8, 8], [1, 8, 8, 8])
    op.attrs = {"stride_w": 1, "stride_h": 1, "dilation_w_factor": 1.5, "dilation_h_factor": "1"}
    assert not support.is_operator_supported(op)


def test_constraint_dilation_range():
    # Dilation width and height must lie within a certain range
    op = testutil.create_op_with_quant_tensors(Op.Conv2D, [1, 8, 8, 8], [1, 8, 8, 8])
    op.attrs = {"stride_w": 1, "stride_h": 1, "dilation_w_factor": 0, "dilation_h_factor": 20}
    assert not support.is_operator_supported(op)


def test_constraint_dilated_height_range():
    # Dilated kernel height must lie within a certain range
    op = testutil.create_op_with_quant_tensors(Op.Conv2D, [1, 8, 8, 8], [1, 8, 8, 8], weights_shape=[65, 64, 1, 1])
    op.attrs = {"stride_w": 1, "stride_h": 1}
    assert not support.is_operator_supported(op)


def test_constraint_dilated_product_range():
    # Dilated kernel width x height must lie within a certain range
    op = testutil.create_op_with_quant_tensors(Op.Conv2D, [1, 8, 8, 8], [1, 8, 8, 8], weights_shape=[64, 65, 1, 1])
    op.attrs = {"stride_w": 1, "stride_h": 1}
    assert not support.is_operator_supported(op)


def test_constraint_weights_type():
    # Weight tensor must be 8-bit
    op = testutil.create_op_with_quant_tensors(
        Op.Conv2D, [1, 8, 8, 8], [1, 8, 8, 8], weights_shape=[1, 1, 1, 1], datatype=DataType.int16
    )
    op.attrs = {"stride_w": 1, "stride_h": 1}
    assert not support.is_operator_supported(op)


def test_constraint_weights_nonconst():
    # Weight tensor cannot be non-const tensors
    op = testutil.create_op_with_quant_tensors(Op.Conv2D, [1, 8, 8, 8], [1, 8, 8, 8])
    op.attrs = {"stride_w": 1, "stride_h": 1}
    weights = Tensor([64, 64, 1, 1], DataType.uint8, "weights")
    weights.quantization = QuantizationParameters()
    op.add_input_tensor(weights)
    assert not support.is_operator_supported(op)


def test_constraint_weights_limit():
    # Sum of weights has a limit
    op = testutil.create_op_with_quant_tensors(Op.Conv2D, [1, 8, 8, 8], [1, 8, 8, 8], weights_shape=[1, 1, 1, 1])
    op.attrs = {"stride_w": 1, "stride_h": 1}
    op.weights.quantization.zero_point = np.array([[[[(127 * 65536) + 1]]]])
    assert not support.is_operator_supported(op)


def test_constraint_bias_type():
    # Bias must have a certain datatype
    op = testutil.create_op_with_quant_tensors(Op.Conv2DBias, [1, 8, 8, 8], [1, 8, 8, 8], weights_shape=[1, 1, 1, 1])
    op.attrs = {"stride_w": 1, "stride_h": 1}
    bias = Tensor([1, 8, 8, 8], DataType.uint8, "bias")
    op.add_input_tensor(bias)
    assert not support.is_operator_supported(op)


def test_constraint_bias_40bit():
    # Bias must not exceed 40-bit
    op = testutil.create_op_with_quant_tensors(Op.Conv2DBias, [1, 1, 1, 1], [1, 1, 1, 1], weights_shape=[1, 1, 1, 1])
    op.attrs = {"stride_w": 1, "stride_h": 1}
    bias = Tensor([1, 1, 1, 1], DataType.int64, "bias")
    bias.quant_values = np.array([0x1FF_FFFF_FFFF])
    op.add_input_tensor(bias)
    assert not support.is_operator_supported(op)


def test_constraint_batch_size():
    op = testutil.create_op_with_quant_tensors(Op.Conv2D, [2, 8, 8, 8], [1, 8, 8, 8], weights_shape=[1, 1, 1, 1])
    op.attrs = {"stride_w": 1, "stride_h": 1}
    assert not support.is_operator_supported(op)
