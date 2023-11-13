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
# Unit tests for tflite_model_semantic
import numpy as np

from ethosu.vela.data_type import DataType
from ethosu.vela.operation import Op
from ethosu.vela.operation import Padding
from ethosu.vela.tensor import create_const_tensor
from ethosu.vela.tensor import QuantizationParameters
from ethosu.vela.tensor import Tensor
from ethosu.vela.test import testutil
from ethosu.vela.tflite_model_semantic import TFLiteSemantic

semantic_checker = TFLiteSemantic()


def test_constraint_tens_no_dynamic():
    # Tensors cannot be dynamic (no shape, not a scalar)
    op = testutil.create_op_with_quant_tensors(Op.Relu, [1, 8, 8, 8], [])
    assert not semantic_checker.is_operator_semantic_valid(op)


def test_constraint_tens_defined_shape():
    # Tensors cannot have None in them
    op = testutil.create_op_with_quant_tensors(Op.Relu, [1, 8, None, 8], [1, 8, 8, 8])
    assert not semantic_checker.is_operator_semantic_valid(op)


def test_constraint_tens_output_scalar():
    # Scalar output is not allowed at all:
    op = testutil.create_elemwise_op(Op.Mul, "op", [1, 8, 8, 8], [1, 8, 8, 8], [])
    op.ofm.values = 0.5
    assert not semantic_checker.is_operator_semantic_valid(op)


def test_constraint_tens_input_scalar():
    # Shapeless input is allowed if its of a certain type:
    op = testutil.create_elemwise_op(Op.Mul, "op", [1, 8, 8, 8], [], [1, 8, 8, 8])
    assert semantic_checker.is_operator_semantic_valid(op)
    # Invalid shapeless input due to op type:
    op = testutil.create_op_with_quant_tensors(Op.Relu, [], [1, 8, 8, 8])
    op.ifm.values = 0.5
    assert not semantic_checker.is_operator_semantic_valid(op)


def test_constraint_tens_shape_size():
    # Tensors cannot be > 4D
    op = testutil.create_op_with_quant_tensors(Op.Relu, [1, 1, 8, 8, 8], [1, 1, 8, 8, 8], set_ifm_ofm_shapes=False)
    assert not semantic_checker.is_operator_semantic_valid(op)


def test_constraint_tens_quant_none_check():
    # Tensors must have quantization parameters
    op = testutil.create_elemwise_op(Op.Mul, "op", [1, 8, 8, 8], [], [1, 8, 8, 8], ifm2_quant=None)
    assert not semantic_checker.is_operator_semantic_valid(op)


def test_constraint_tens_quant_scale():
    # Quantization scale cannot be infinite
    qp = QuantizationParameters()
    qp.zero_point = 0
    qp.scale_f32 = np.inf
    op = testutil.create_elemwise_op(Op.Mul, "op", [1, 8, 8, 8], [], [1, 8, 8, 8], ifm_quant=qp)
    assert not semantic_checker.is_operator_semantic_valid(op)


def test_constraint_fc_output_2d_not_supp():
    op = testutil.create_op_with_quant_tensors(Op.FullyConnected, [7, 4, 6], [3, 2, 2, 8], weights_shape=[1, 9, 1])
    assert not semantic_checker.is_operator_semantic_valid(op)
    op = testutil.create_op_with_quant_tensors(Op.FullyConnected, [12, 1, 6, 1], [3, 7, 4], weights_shape=[1, 1, 7, 1])
    assert not semantic_checker.is_operator_semantic_valid(op)
    op = testutil.create_op_with_quant_tensors(Op.FullyConnected, [4, 1, 4, 7], [1, 9], weights_shape=[12, 3])
    assert not semantic_checker.is_operator_semantic_valid(op)
    op = testutil.create_op_with_quant_tensors(Op.FullyConnected, [4], [9], weights_shape=[3, 2])
    assert not semantic_checker.is_operator_semantic_valid(op)


def test_constraint_fc_output_2d_is_supp():
    op = testutil.create_op_with_quant_tensors(Op.FullyConnected, [4, 8, 8, 4], [32, 32], weights_shape=[4, 8, 8, 4])
    assert semantic_checker.is_operator_semantic_valid(op)
    op = testutil.create_op_with_quant_tensors(Op.FullyConnected, [1, 1024], [16, 64], weights_shape=[1, 1024])
    assert semantic_checker.is_operator_semantic_valid(op)
    op = testutil.create_op_with_quant_tensors(Op.FullyConnected, [12, 1], [3, 2, 1, 1], weights_shape=[12, 1, 1, 1])
    assert semantic_checker.is_operator_semantic_valid(op)
    op = testutil.create_op_with_quant_tensors(Op.FullyConnected, [12, 1], [3, 2, 1], weights_shape=[12, 1, 1, 1])
    assert semantic_checker.is_operator_semantic_valid(op)
    op = testutil.create_op_with_quant_tensors(Op.FullyConnected, [12, 1], [1, 1, 3, 2], weights_shape=[12, 1, 1, 1])
    assert semantic_checker.is_operator_semantic_valid(op)
    op = testutil.create_op_with_quant_tensors(Op.FullyConnected, [12, 1, 1, 1], [1, 1, 1], weights_shape=[12, 1, 1, 1])
    assert semantic_checker.is_operator_semantic_valid(op)
    op = testutil.create_op_with_quant_tensors(
        Op.FullyConnected, [12, 1, 1, 1], [1, 1, 24], weights_shape=[12, 1, 1, 1]
    )
    assert semantic_checker.is_operator_semantic_valid(op)
    op = testutil.create_op_with_quant_tensors(Op.FullyConnected, [1, 1, 1, 1], [1, 3, 4], weights_shape=[1, 1, 1, 1])
    assert semantic_checker.is_operator_semantic_valid(op)


def test_constraint_conv_pass():
    # First test a simple conv passes
    op = testutil.create_op_with_quant_tensors(Op.Conv2DBias, [1, 1, 1, 1], [1, 1, 1, 1], weights_shape=[1, 1, 1, 1])
    op.attrs = {"stride_w": 1, "stride_h": 1}
    assert semantic_checker.is_operator_semantic_valid(op)


def test_constraint_stride_type():
    # Stride width and height must be integer types
    op = testutil.create_op_with_quant_tensors(Op.Conv2DBias, [1, 8, 8, 8], [1, 8, 8, 8], weights_shape=[1, 1, 1, 1])
    op.attrs = {"stride_w": 1.5, "stride_h": "1"}
    assert not semantic_checker.is_operator_semantic_valid(op)


def test_constraint_conv_groups_ifm_depth():
    # Test IFM depth is a whole multiple of the filter kernel depth
    op = testutil.create_op_with_quant_tensors(Op.Conv2DBias, [1, 8, 8, 15], [1, 8, 8, 5], weights_shape=[1, 1, 3, 5])
    assert semantic_checker.is_operator_semantic_valid(op)

    op = testutil.create_op_with_quant_tensors(Op.Conv2DBias, [1, 8, 8, 15], [1, 8, 8, 5], weights_shape=[1, 1, 4, 5])
    assert not semantic_checker.is_operator_semantic_valid(op)


def test_constraint_conv_groups_num_filters():
    # Test number of filter kernels is equally divisible by the number of convolution groups
    op = testutil.create_op_with_quant_tensors(Op.Conv2DBias, [1, 8, 8, 15], [1, 8, 8, 20], weights_shape=[1, 1, 3, 20])
    assert semantic_checker.is_operator_semantic_valid(op)

    op = testutil.create_op_with_quant_tensors(Op.Conv2DBias, [1, 8, 8, 15], [1, 8, 8, 21], weights_shape=[1, 1, 3, 21])
    assert not semantic_checker.is_operator_semantic_valid(op)


def test_constraint_dilation_type():
    # Dilation width and height must be integer types
    op = testutil.create_op_with_quant_tensors(Op.Conv2DBias, [1, 8, 8, 8], [1, 8, 8, 8], weights_shape=[1, 1, 1, 1])
    op.attrs = {"stride_w": 1, "stride_h": 1, "dilation_w_factor": 1.5, "dilation_h_factor": "1"}
    assert not semantic_checker.is_operator_semantic_valid(op)


def test_constraint_quant_scale_inf():
    # Test handling IFM scale/OFM scale is infinite
    op = testutil.create_op_with_quant_tensors(Op.Relu, [1, 8, 8, 8], [1, 8, 8, 8])
    op.ifm.quantization.scale_f32 = np.float32(1e9)
    op.ofm.quantization.scale_f32 = np.float32(1e-35)
    # Temporarily ignore overflow in NumPy
    old_settings = np.seterr(over="ignore", under="ignore")
    valid = semantic_checker.is_operator_semantic_valid(op)
    # Reset NumPy settings
    np.seterr(**old_settings)

    assert not valid


def test_constraint_ofm_scale_too_small():
    # Tests handling of OFM scale < 1e-38
    shp = [1, 10, 20, 16]
    op = testutil.create_elemwise_op(
        Op.Mul,
        "mul",
        shp,
        shp,
        shp,
        ofm_quant=testutil.default_quant_params(),
    )
    assert semantic_checker.is_operator_semantic_valid(op)
    op.ofm.quantization.scale_f32 = 1e-43
    assert not semantic_checker.is_operator_semantic_valid(op)


def test_constraint_matching_in_out_types():
    # Valid
    op = testutil.create_op_with_quant_tensors(Op.AvgPool, [1, 8, 8, 8], [1, 8, 8, 8])
    op.attrs = {"stride_w": 2, "stride_h": 2, "filter_width": 2, "filter_height": 2, "padding": Padding.SAME}
    assert semantic_checker.is_operator_semantic_valid(op)
    # Invalid. datatypes for ifm and ofm must match (default uint8)
    op.ifm.dtype = DataType.int8
    assert not semantic_checker.is_operator_semantic_valid(op)


def test_constraint_filter_type():
    # Filter width/height must be integers
    op = testutil.create_op_with_quant_tensors(Op.AvgPool, [1, 8, 8, 8], [1, 8, 8, 8])
    op.attrs = {"stride_w": 2, "stride_h": 2, "filter_width": 2.5, "filter_height": "2", "padding": Padding.SAME}
    assert not semantic_checker.is_operator_semantic_valid(op)


def test_constraint_matching_shapes():
    # Softmax requires the ifm and ofm shapes to match
    op = testutil.create_op_with_quant_tensors(Op.Softmax, [1, 1, 1, 8], [1, 2, 2, 4])
    assert not semantic_checker.is_operator_semantic_valid(op)
    op = testutil.create_op_with_quant_tensors(Op.Softmax, [1, 1, 1, 8], [1, 1, 1, 8])
    assert semantic_checker.is_operator_semantic_valid(op)


def test_constraint_beta_value_range():
    # beta must be positive
    op = testutil.create_op_with_quant_tensors(Op.Softmax, [1, 1, 1, 8], [1, 1, 1, 8])
    op.attrs["beta"] = -1.0
    assert not semantic_checker.is_operator_semantic_valid(op)
    op.attrs["beta"] = 0.0
    assert semantic_checker.is_operator_semantic_valid(op)


def test_constraint_split_axis():
    # Axis value must be in the range [-<ifm_dimensions>, <ifm_dimensions>)"
    attrs = {"num_splits": 2}
    axis = create_const_tensor("axis", [1], DataType.int8, [3])
    ifm = Tensor([1, 1, 4], DataType.int8, "ifm")
    ifm.quantization = testutil.default_quant_params()
    ofm = Tensor([1, 1, 4], DataType.int8, "ofm")
    ofm.quantization = testutil.default_quant_params()
    op = testutil.create_op(Op.Split, [axis, ifm], ofm, attrs)
    # Check invalid axis value
    assert not semantic_checker.is_operator_semantic_valid(op)
    # Check valid axis value
    axis = create_const_tensor("axis", [1], DataType.int8, [-1])
    op = testutil.create_op(Op.Split, [axis, ifm], ofm, attrs)
    assert semantic_checker.is_operator_semantic_valid(op)


def test_constraint_split_num_splits():
    # Check that split number is valid"
    attrs = {"num_splits": 2}
    axis = create_const_tensor("axis", [1], DataType.int8, [-1])
    ifm = Tensor([1, 1, 3], DataType.int8, "ifm")
    ifm.quantization = testutil.default_quant_params()
    ofm = Tensor([1, 1, 3], DataType.int8, "ofm")
    ofm.quantization = testutil.default_quant_params()
    op = testutil.create_op(Op.Split, [axis, ifm], ofm, attrs)
    # Check invalid split number 2
    assert not semantic_checker.is_operator_semantic_valid(op)
    # Check valid split number 3
    attrs = {"num_splits": 3}
    op = testutil.create_op(Op.Split, [axis, ifm], ofm, attrs)
    assert semantic_checker.is_operator_semantic_valid(op)


def test_constraint_splitv_inferred():
    # SplitV requires a maximum of one inferred shape (-1)
    qp = testutil.default_quant_params()
    op = testutil.create_op_with_quant_tensors(Op.SplitV, [1, 1, 1, 8], [1, 1, 1, 8])
    sizes = create_const_tensor("sizes", [1, 1, 1, 4], DataType.int16, [[[[0, -1, 2, -1]]]], quantization=qp)
    op.add_input_tensor(sizes)
    assert not semantic_checker.is_operator_semantic_valid(op)
    op = testutil.create_op_with_quant_tensors(Op.SplitV, [1, 1, 1, 8], [1, 1, 1, 8])
    sizes = create_const_tensor("sizes", [1, 1, 1, 4], DataType.int16, [[[[0, 1, 2, -1]]]], quantization=qp)
    op.add_input_tensor(sizes)
    assert semantic_checker.is_operator_semantic_valid(op)


def test_constraint_concat_pass():
    # A working concat
    op = testutil.create_op_with_quant_tensors(Op.ConcatTFLite, [1, 1, 1, 4], [1, 1, 1, 8])
    ifm2 = Tensor([1, 1, 1, 4], DataType.uint8, "in2")
    ifm2.quantization = testutil.default_quant_params()
    op.add_input_tensor(ifm2)
    op.attrs["axis"] = 3
    assert semantic_checker.is_operator_semantic_valid(op)


def test_constraint_axis_exists():
    # Missing axis attribute
    op = testutil.create_op_with_quant_tensors(Op.ConcatTFLite, [1, 1, 1, 4], [1, 1, 1, 8])
    ifm2 = Tensor([1, 1, 1, 4], DataType.uint8, "in2")
    ifm2.quantization = testutil.default_quant_params()
    op.add_input_tensor(ifm2)
    assert not semantic_checker.is_operator_semantic_valid(op)


def test_constraint_axis_valid():
    # Invalid axis attribute
    op = testutil.create_op_with_quant_tensors(Op.ConcatTFLite, [1, 1, 1, 4], [1, 1, 1, 8])
    ifm2 = Tensor([1, 1, 1, 4], DataType.uint8, "in2")
    ifm2.quantization = testutil.default_quant_params()
    op.add_input_tensor(ifm2)
    op.attrs["axis"] = 7
    assert not semantic_checker.is_operator_semantic_valid(op)


def test_constraint_matching_dimensionality():
    # Mismatching dimensionality: 4D+2D=4D
    op = testutil.create_op_with_quant_tensors(Op.ConcatTFLite, [1, 1, 1, 4], [1, 1, 1, 8])
    ifm2 = Tensor([1, 4], DataType.uint8, "in2")
    ifm2.quantization = testutil.default_quant_params()
    op.add_input_tensor(ifm2)
    op.attrs["axis"] = 3
    assert not semantic_checker.is_operator_semantic_valid(op)


def test_constraint_valid_dimensions():
    # Mismatching dimension value:
    # ifm2 has w and h as 2, which is not the axis to concat and doesnt match ifm1 or ofm
    op = testutil.create_op_with_quant_tensors(Op.ConcatTFLite, [1, 1, 1, 4], [1, 1, 1, 8])
    ifm2 = Tensor([1, 2, 2, 4], DataType.uint8, "in2")
    ifm2.quantization = testutil.default_quant_params()
    op.add_input_tensor(ifm2)
    op.attrs["axis"] = 3
    assert not semantic_checker.is_operator_semantic_valid(op)


def create_strided_slice_op(in_shape, out_shape, start_offsets, end_offsets):
    qp = testutil.default_quant_params()
    in0 = Tensor(in_shape, DataType.uint8, "in")
    in0.quantization = qp
    in1 = create_const_tensor("begin", [len(start_offsets)], DataType.uint8, start_offsets, quantization=qp)
    in2 = create_const_tensor("end", [len(end_offsets)], DataType.uint8, end_offsets, quantization=qp)
    in3 = create_const_tensor("strides", [len(end_offsets)], DataType.uint8, len(end_offsets) * [1], quantization=qp)
    out = Tensor(out_shape, DataType.uint8, "out")
    out.quantization = qp
    attrs = {"ellipsis_mask": 0, "new_axis_mask": 0, "shrink_axis_mask": 0, "begin_mask": 0, "end_mask": 0}
    return testutil.create_op(Op.StridedSlice, [in0, in1, in2, in3], out, attrs=attrs)


def create_pad_op(
    in_shape,
    out_shape,
    padding,
    in_dtype=DataType.int8,
    out_dtype=DataType.int8,
    pad_dtype=DataType.int32,
):
    qp = testutil.default_quant_params()
    in0 = Tensor(in_shape, in_dtype, "in")
    in0.quantization = qp
    shape = [] if padding == [] else list(np.shape(padding))
    pad_tensor = create_const_tensor(name="pad", shape=shape, values=padding, dtype=pad_dtype)
    out = Tensor(out_shape, out_dtype, "out")
    out.quantization = qp.clone()
    op = testutil.create_op(Op.Pad, [in0, pad_tensor], out)
    return op


def test_constraint_pad_input_count():
    # Incorrect number of input tensors (2)
    op = create_pad_op(
        in_shape=[1, 1, 1, 1],
        out_shape=[1, 3, 3, 1],
        padding=[[0, 0], [1, 1], [1, 1], [0, 0]],
    )
    assert semantic_checker.is_operator_semantic_valid(op)
    op.add_input_tensor(op.inputs[0].clone())
    assert not semantic_checker.is_operator_semantic_valid(op)


def test_constraint_pad_output_shape():
    # Incorrect output tensor shape
    op = create_pad_op(
        in_shape=[1, 1, 1, 1],
        out_shape=[1, 3, 3, 1],
        padding=[[0, 0], [1, 1], [1, 1], [0, 0]],
    )
    assert semantic_checker.is_operator_semantic_valid(op)
    op.outputs[0].shape = [1, 1, 1, 1]
    assert not semantic_checker.is_operator_semantic_valid(op)


def create_strided_slice():
    # Creates a valid strided slice operator with some valid inputs/outputs
    op = create_strided_slice_op([1, 10, 10, 10], [1, 5, 5, 10], [127, 2, 2, 0], [0, 7, -3, 0])
    op.attrs["begin_mask"] = 1
    op.attrs["end_mask"] = 9
    assert semantic_checker.is_operator_semantic_valid(op)
    return op


def test_constraint_stridedslice_input_count():
    # Wrong number of input tensors
    op = create_strided_slice()
    op.add_input_tensor(op.inputs[0].clone())
    assert not semantic_checker.is_operator_semantic_valid(op)


def test_constraint_stridedslice_inputs_const():
    # begin, end, stride values must not be None
    op = create_strided_slice()
    op.inputs[1].values = None
    assert not semantic_checker.is_operator_semantic_valid(op)
    op = create_strided_slice()
    op.inputs[2].values = None
    assert not semantic_checker.is_operator_semantic_valid(op)
    op = create_strided_slice()
    op.inputs[3].values = None
    assert not semantic_checker.is_operator_semantic_valid(op)


def test_constraint_ellipsis_mask():
    # Unsemantic_checkered ellipsis mask
    op = create_strided_slice()
    op.attrs["ellipsis_mask"] = 1
    assert not semantic_checker.is_operator_semantic_valid(op)


def test_constraint_axis_masks():
    op = create_strided_slice()
    # Setting one of new_axis_mask/shrink_axis_mask to non-zero is ok
    op.attrs["new_axis_mask"] = 2
    assert semantic_checker.is_operator_semantic_valid(op)
    op = create_strided_slice()
    op.attrs["shrink_axis_mask"] = 3
    assert semantic_checker.is_operator_semantic_valid(op)
    # But setting both to non-zero is not semantic_checkered
    op.attrs["new_axis_mask"] = 2
    assert not semantic_checker.is_operator_semantic_valid(op)


def test_constraint_slice_ranges():
    # Examples where end offset <= begin offset
    op = create_strided_slice()
    op.inputs[1].values = [0, 7, 2, 0]
    assert not semantic_checker.is_operator_semantic_valid(op)
    op = create_strided_slice()
    op.inputs[2].values = [0, 7, 2, 0]
    assert not semantic_checker.is_operator_semantic_valid(op)
    op = create_strided_slice()
    op.attrs["begin_mask"] = 0
    assert not semantic_checker.is_operator_semantic_valid(op)
    op = create_strided_slice()
    op.attrs["end_mask"] = 0
    assert not semantic_checker.is_operator_semantic_valid(op)


def test_constraint_matching_inputs_types():
    # input data types must match (default is uint8)
    op = testutil.create_elemwise_op(Op.Mul, "op", [1, 8, 8, 8], [1, 8, 8, 8], [1, 8, 8, 8])
    op.ifm2.dtype = DataType.int8
    assert not semantic_checker.is_operator_semantic_valid(op)


def test_constraint_matching_signed():
    # signed inputs require output to also be signed
    op = testutil.create_elemwise_op(Op.Mul, "op", [1, 8, 8, 8], [1, 8, 8, 8], [1, 8, 8, 8], datatype=DataType.int8)
    op.ofm.dtype = DataType.uint8
    assert not semantic_checker.is_operator_semantic_valid(op)


def test_constraint_unsigned_valid():
    # unsigned inputs require output to be either:
    op = testutil.create_elemwise_op(Op.Mul, "op", [1, 8, 8, 8], [1, 8, 8, 8], [1, 8, 8, 8])
    # the same (default uint8)
    assert semantic_checker.is_operator_semantic_valid(op)
    op.ofm.dtype = DataType.int8
    assert not semantic_checker.is_operator_semantic_valid(op)
    op.ofm.dtype = DataType.int16
    assert not semantic_checker.is_operator_semantic_valid(op)
    # or int32
    op.ofm.dtype = DataType.int32
    assert semantic_checker.is_operator_semantic_valid(op)


def test_constraint_matching_either_shapes():
    # BINARY CASE
    # At least one ifm shape must match ofm's shape
    op = testutil.create_elemwise_op(Op.Add, "op", [1, 4], [4, 4], [4, 4])
    assert semantic_checker.is_operator_semantic_valid(op)
    op = testutil.create_elemwise_op(Op.Add, "op", [4, 4], [1, 4], [4, 4])
    assert semantic_checker.is_operator_semantic_valid(op)
    op = testutil.create_elemwise_op(Op.Add, "op", [4, 4], [4, 4], [2, 2])
    assert not semantic_checker.is_operator_semantic_valid(op)
    op = testutil.create_elemwise_op(Op.Add, "op", [1, 4, 1, 16], [1, 1, 4, 1], [1, 4, 4, 16])
    assert not semantic_checker.is_operator_semantic_valid(op)
    op = testutil.create_elemwise_op(Op.Add, "op", [1, 1, 4, 1], [1, 4, 1, 16], [1, 4, 4, 16])
    assert not semantic_checker.is_operator_semantic_valid(op)

    # UNARY CASE
    # No second input so this is treated the same as requiring ifm shape to match ofm shape
    op = testutil.create_elemwise_op(Op.CLZ, "op", [2, 2], None, [2, 2], datatype=DataType.int32)
    assert semantic_checker.is_operator_semantic_valid(op)
    op = testutil.create_elemwise_op(Op.CLZ, "op", [4, 4], None, [2, 2], datatype=DataType.int32)
    assert not semantic_checker.is_operator_semantic_valid(op)


def test_constraint_hardswish_dtype():
    # HardSwish operator dtype should be int8 or uint8, and input dtype must match output
    # UINT8
    op = testutil.create_op_with_quant_tensors(Op.HardSwish, [1, 8, 8, 8], [1, 8, 8, 8])
    assert semantic_checker.is_operator_semantic_valid(op)
    # INT8
    op = testutil.create_op_with_quant_tensors(Op.HardSwish, [1, 8, 8, 8], [1, 8, 8, 8], datatype=DataType.int8)
    assert semantic_checker.is_operator_semantic_valid(op)

    # Invalid
    op = testutil.create_op_with_quant_tensors(Op.HardSwish, [1, 8, 8, 8], [1, 8, 8, 8], datatype=DataType.int16)
    assert not semantic_checker.is_operator_semantic_valid(op)
    op = testutil.create_op_with_quant_tensors(Op.HardSwish, [1, 8, 8, 8], [1, 8, 8, 8], datatype=DataType.uint16)
    assert not semantic_checker.is_operator_semantic_valid(op)
    op = testutil.create_op_with_quant_tensors(Op.HardSwish, [1, 8, 8, 8], [1, 8, 8, 8], datatype=DataType.int32)
    assert not semantic_checker.is_operator_semantic_valid(op)

    in_tens = Tensor([1, 8, 8, 8], DataType.int8, "in")
    out_tens = Tensor([1, 8, 8, 8], DataType.uint8, "out")
    op = testutil.create_op(Op.HardSwish, [in_tens], out_tens)
    assert not semantic_checker.is_operator_semantic_valid(op)


def test_constraint_keep_dims_ifm_ofm():
    op = testutil.create_op_with_quant_tensors(Op.FullyConnected, [4, 8, 8, 4], [32, 32], weights_shape=[4, 8, 8, 4])
    op.attrs["keep_num_dims"] = True
    assert not semantic_checker.is_operator_semantic_valid(op)
    op.attrs["keep_num_dims"] = False
    assert semantic_checker.is_operator_semantic_valid(op)


def create_mean(input_shape, output_shape, axis, datatype, attrs):
    ifm = Tensor(input_shape, datatype, "in")
    ifm.quantization = testutil.default_quant_params()
    ofm = Tensor(output_shape, datatype, "out")
    ofm.quantization = testutil.default_quant_params()
    if type(axis) is list:
        indices = create_const_tensor("indices", [len(axis)], DataType.int32, axis)
    elif type(axis) is int:
        indices = create_const_tensor("indices", [], DataType.int32, axis)
    op = testutil.create_op(Op.Mean, [ifm, indices], ofm, attrs)
    return op


def test_mean_dtype():
    op = create_mean([1, 6, 6, 16], [1, 1, 1, 16], [1, 2], DataType.int8, {"keep_dims": True})
    assert semantic_checker.is_operator_semantic_valid(op)
    op.ifm.dtype = DataType.int16
    op.ofm.dtype = DataType.int16
    assert semantic_checker.is_operator_semantic_valid(op)


def test_mean_axis():
    op = create_mean([1, 6, 6, 16], [1, 1, 1, 16], [3], DataType.int8, {"keep_dims": True})
    assert not semantic_checker.is_operator_semantic_valid(op)
    op = create_mean([1, 6, 6, 1], [1, 1, 1, 1], [3], DataType.int8, {"keep_dims": True})
    assert semantic_checker.is_operator_semantic_valid(op)

    op = create_mean([2, 6, 6, 16], [2, 1, 1, 16], [0], DataType.int8, {"keep_dims": True})
    assert not semantic_checker.is_operator_semantic_valid(op)
    op = create_mean([1, 6, 6, 16], [1, 1, 1, 16], 0, DataType.int8, {"keep_dims": True})
    assert semantic_checker.is_operator_semantic_valid(op)

    op = create_mean([2, 6, 6, 16], [2, 1, 1, 16], [0, 1], DataType.int8, {"keep_dims": True})
    assert not semantic_checker.is_operator_semantic_valid(op)
    op = create_mean([1, 6, 6, 16], [1, 1, 1, 16], [0, 1], DataType.int8, {"keep_dims": True})
    assert semantic_checker.is_operator_semantic_valid(op)

    op = create_mean([1, 6, 6, 16], [1, 1, 1, 16], [1, 2], DataType.int8, {"keep_dims": True})
    assert semantic_checker.is_operator_semantic_valid(op)
    op = create_mean([1, 6, 6, 16], [1, 1, 1, 16], [1], DataType.int8, {"keep_dims": True})
    assert semantic_checker.is_operator_semantic_valid(op)
    op = create_mean([1, 6, 6, 16], [1, 1, 1, 16], 2, DataType.int8, {"keep_dims": True})
    assert semantic_checker.is_operator_semantic_valid(op)
    op = create_mean([1, 6, 6, 16], [1, 1, 1, 16], [2, 1], DataType.int8, {"keep_dims": True})
    assert semantic_checker.is_operator_semantic_valid(op)


def test_matching_in_out_quant():
    # quantisation parameters of ifm and ofm should match.
    quant = testutil.default_quant_params()
    # create reshape op
    ifm_shape = [64, 16]
    ofm_shape = [1, 4, 16, 16]
    ifm = create_const_tensor("reshape_in", ifm_shape, DataType.uint8, np.zeros(ifm_shape))
    ifm.quantization = quant
    ofm = create_const_tensor("reshape_out", ofm_shape, DataType.uint8, np.zeros(ofm_shape))
    ofm.quantization = quant.clone()
    shape_tens = create_const_tensor("shape", [1], DataType.int32, ofm_shape)
    op = testutil.create_op(Op.Reshape, [ifm, shape_tens], ofm, set_ifm_ofm_shapes=False)
    op.attrs["new_shape"] = ofm_shape

    # Matching quantisation parameters
    assert semantic_checker.is_operator_semantic_valid(op)

    # Different zp
    ofm.quantization.zero_point = 32
    assert not semantic_checker.is_operator_semantic_valid(op)

    # Different scale
    ofm.quantization.zero_point = 0
    ofm.quantization.scale_f32 = 0.9
    assert not semantic_checker.is_operator_semantic_valid(op)

    # Squeeze op diff quant
    # create squeeze op
    ifm_shape = [1, 1, 1, 1001]
    ofm_shape = [1, 1001]
    ifm = create_const_tensor("squeeze_in", ifm_shape, DataType.uint8, np.zeros(ifm_shape))
    ifm.quantization = quant
    ofm = create_const_tensor("squeeze_out", ofm_shape, DataType.uint8, np.zeros(ofm_shape))
    ofm.quantization = quant.clone()
    ofm.quantization.zero_point = 32
    op = testutil.create_op(Op.Squeeze, [ifm], ofm, set_ifm_ofm_shapes=False)
    op.attrs["squeeze_dims"] = [1, 2]
    assert not semantic_checker.is_operator_semantic_valid(op)

    # ExpandDims diff quant
    quant = testutil.default_quant_params()
    # create expand_dims op
    ifm_shape = [4, 16, 16]
    ofm_shape = [1, 4, 16, 16]
    ifm = create_const_tensor("expand_dims_in", ifm_shape, DataType.uint8, np.zeros(ifm_shape))
    ifm.quantization = quant
    ofm = create_const_tensor("expand_dims_out", ofm_shape, DataType.uint8, np.zeros(ofm_shape))
    ofm.quantization = quant.clone()
    ofm.quantization.zero_point = 32
    dim = create_const_tensor("expand_dims_dim", [], DataType.uint8, 0)
    op = testutil.create_op(Op.ExpandDims, [ifm, dim], ofm, set_ifm_ofm_shapes=False)
    assert not semantic_checker.is_operator_semantic_valid(op)


def test_lstm_semantics():
    # Test valid configurations
    op = testutil.create_lstm_op(3, 12, 24, 20, DataType.int8)
    assert semantic_checker.is_operator_semantic_valid(op)
    assert semantic_checker.is_operator_semantic_valid(testutil.create_lstm_op(3, 12, 24, 20, DataType.int16))
    # Test invalid datatype
    assert not semantic_checker.is_operator_semantic_valid(testutil.create_lstm_op(3, 12, 24, 20, DataType.uint8))
    # Test invalid shape
    ifm_shape = op.ifm.shape
    ofm_shape = op.ofm.shape
    op.ifm.shape = [12, 24]
    assert not semantic_checker.is_operator_semantic_valid(op)
    op.ifm.shape = ifm_shape
    op.ofm.shape = [12, 20]
    assert not semantic_checker.is_operator_semantic_valid(op)
    op.ofm.shape = ofm_shape
    # Test invalid number of intermediates
    intermediate = op.intermediates.pop()
    assert not semantic_checker.is_operator_semantic_valid(op)
    op.intermediates.append(intermediate)
    op.intermediates.append(intermediate)
    assert not semantic_checker.is_operator_semantic_valid(op)
    op.intermediates.pop()
    # Test invalid number of inputs
    input = op.inputs.pop()
    assert not semantic_checker.is_operator_semantic_valid(op)
    op.inputs.append(input)
    op.inputs.append(input)
    assert not semantic_checker.is_operator_semantic_valid(op)
    op.inputs.pop()
    # Test restored valid configuration
    assert semantic_checker.is_operator_semantic_valid(op)


def test_transpose_semantics():
    # Test valid op
    ifm = Tensor([2, 4], DataType.int8, "ifm")
    perm = create_const_tensor("perm", [2], DataType.int32, [1, 0])
    ofm = Tensor([4, 2], DataType.int8, "ofm")
    op = testutil.create_op(Op.Transpose, [ifm, perm], ofm)
    assert semantic_checker.is_operator_semantic_valid(op)
    # Test invalid permutation size
    perm = create_const_tensor("perm", [3], DataType.int32, [1, 0])
    op = testutil.create_op(Op.Transpose, [ifm, perm], ofm)
    assert not semantic_checker.is_operator_semantic_valid(op)
    # Test invalid permutation values
    perm = create_const_tensor("perm", [2], DataType.int32, [2, 0])
    op = testutil.create_op(Op.Transpose, [ifm, perm], ofm)
    assert not semantic_checker.is_operator_semantic_valid(op)
    # Test invalid permutation values
    perm = create_const_tensor("perm", [2], DataType.int32, [0, -1])
    op = testutil.create_op(Op.Transpose, [ifm, perm], ofm)
    assert not semantic_checker.is_operator_semantic_valid(op)
    # Test invalid permutation values
    perm = create_const_tensor("perm", [2], DataType.int32, [1, 0])
    perm.values = None
    op = testutil.create_op(Op.Transpose, [ifm, perm], ofm)
    assert not semantic_checker.is_operator_semantic_valid(op)
