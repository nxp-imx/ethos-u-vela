# Copyright (C) 2020-2021 Arm Limited or its affiliates. All rights reserved.
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
import pytest

from ethosu.vela.data_type import DataType
from ethosu.vela.operation import ActivationFunction
from ethosu.vela.operation import Op
from ethosu.vela.operation import Padding
from ethosu.vela.supported_operators import SupportedOperators
from ethosu.vela.tensor import create_const_tensor
from ethosu.vela.tensor import QuantizationParameters
from ethosu.vela.tensor import Tensor
from ethosu.vela.test import testutil

support = SupportedOperators()


def test_constraint_tens_no_dynamic():
    # Tensors cannot be dynamic (no shape, not a scalar)
    op = testutil.create_op_with_quant_tensors(Op.Relu, [1, 8, 8, 8], [])
    assert not support.is_operator_supported(op)


def test_constraint_tens_defined_shape():
    # Tensors cannot have None in them
    op = testutil.create_op_with_quant_tensors(Op.Relu, [1, 8, None, 8], [1, 8, 8, 8])
    assert not support.is_operator_supported(op)


def test_constraint_tens_output_scalar():
    # Scalar output is not allowed at all:
    op = testutil.create_elemwise_op(Op.Mul, "op", [1, 8, 8, 8], [1, 8, 8, 8], [])
    op.ofm.values = 0.5
    assert not support.is_operator_supported(op)


def test_constraint_tens_input_scalar():
    # Shapeless input is allowed if its of a certain type:
    op = testutil.create_elemwise_op(Op.Mul, "op", [1, 8, 8, 8], [], [1, 8, 8, 8])
    assert support.is_operator_supported(op)
    # Invalid shapeless input due to op type:
    op = testutil.create_op_with_quant_tensors(Op.Relu, [], [1, 8, 8, 8])
    op.ifm.values = 0.5
    assert not support.is_operator_supported(op)


def test_constraint_tens_shape_size():
    # Tensors cannot be > 4D
    op = testutil.create_op_with_quant_tensors(Op.Relu, [1, 1, 8, 8, 8], [1, 1, 8, 8, 8], set_ifm_ofm_shapes=False)
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
    # Quantization scale cannot be infinite
    qp = QuantizationParameters()
    qp.zero_point = 0
    qp.scale_f32 = np.inf
    op = testutil.create_elemwise_op(Op.Mul, "op", [1, 8, 8, 8], [], [1, 8, 8, 8], ifm_quant=qp)
    assert not support.is_operator_supported(op)


def test_constraint_tens_quant_per_axis_not_supp():
    # Quantization scale cannot be array-valued for elemwise ops
    qp = QuantizationParameters()
    qp.zero_point = np.zeros((1, 3))
    qp.scale_f32 = np.ones((1, 3))
    op = testutil.create_elemwise_op(Op.Mul, "op", [1, 8, 8, 8], [], [1, 8, 8, 8], ifm_quant=qp)
    assert not support.is_operator_supported(op)


def test_constraint_tens_quant_per_axis_is_supp():
    op = testutil.create_op_with_quant_tensors(
        Op.Conv2DBias, [1, 1, 1, 3], [1, 1, 1, 3], weights_shape=[1, 1, 1, 3], bias_shape=[1, 1, 1, 3]
    )
    op.attrs = {"stride_w": 1, "stride_h": 1}
    assert support.is_operator_supported(op)
    qp = QuantizationParameters()
    qp.zero_point = np.zeros((1, 3))
    qp.scale_f32 = np.ones((1, 3))
    op.bias.quantization = qp
    assert support.is_operator_supported(op)


def test_constraint_fc_output_2d_not_supp():
    op = testutil.create_op_with_quant_tensors(Op.FullyConnected, [12, 1], [3, 2, 2, 1], weights_shape=[12, 1, 1, 1])
    assert not support.is_operator_supported(op)
    op = testutil.create_op_with_quant_tensors(Op.FullyConnected, [12, 1, 1, 1], [1, 3, 4], weights_shape=[12, 1, 1, 1])
    assert not support.is_operator_supported(op)
    op = testutil.create_op_with_quant_tensors(Op.FullyConnected, [1, 1, 1, 1], [1], weights_shape=[1, 1, 1, 1])
    assert not support.is_operator_supported(op)


def test_constraint_fc_output_2d_is_supp():
    op = testutil.create_op_with_quant_tensors(Op.FullyConnected, [4, 8, 8, 4], [32, 32], weights_shape=[4, 8, 8, 4])
    assert support.is_operator_supported(op)
    op = testutil.create_op_with_quant_tensors(Op.FullyConnected, [1, 1024], [16, 64], weights_shape=[1, 1024])
    assert support.is_operator_supported(op)


def test_constraint_faf():
    # Fused activation functions, if set, must be a valid op type
    op = testutil.create_op_with_quant_tensors(Op.Relu, [1, 8, 8, 8], [1, 8, 8, 8])
    op.activation = ActivationFunction(Op.Conv2D)
    assert not support.is_operator_supported(op)


def test_constraint_faf_ofm_dtype():
    # If fused activation function is present, OFM must be 8 or 16 bit
    shp = [1, 8, 8, 8]
    for dtype in [DataType.int8, DataType.uint8, DataType.int16, DataType.int32]:
        op = testutil.create_elemwise_op(Op.Add, "op", shp, shp, shp, datatype=dtype)
        op.activation = ActivationFunction(Op.Relu)
        expected = dtype.size_in_bytes() <= 2
        assert support.is_operator_supported(op) == expected, f"Data type: {dtype}"


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


def test_constraint_weights_const():
    # Weight tensor cannot be non-const tensors
    op = testutil.create_op_with_quant_tensors(Op.Conv2D, [1, 8, 8, 8], [1, 8, 8, 8])
    op.attrs = {"stride_w": 1, "stride_h": 1}
    weights = Tensor([64, 64, 1, 1], DataType.uint8, "weights")
    weights.quantization = testutil.default_quant_params()
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
    bias.quant_values = np.array([0x01FF_FFFF_FFFF])
    op.add_input_tensor(bias)
    assert not support.is_operator_supported(op)


def test_constraint_batch_size():
    op = testutil.create_op_with_quant_tensors(Op.Conv2D, [2, 8, 8, 8], [1, 8, 8, 8], weights_shape=[1, 1, 1, 1])
    op.attrs = {"stride_w": 1, "stride_h": 1}
    assert not support.is_operator_supported(op)


def test_constraint_quant_scale_inf():
    # Test handling IFM scale/OFM scale is infinite
    op = testutil.create_op_with_quant_tensors(Op.Relu, [1, 8, 8, 8], [1, 8, 8, 8])
    op.ifm.quantization.scale_f32 = np.float32(1e9)
    op.ofm.quantization.scale_f32 = np.float32(1e-35)
    assert not support.is_operator_supported(op)


def test_constraint_ofm_scale_too_small():
    # Tests handling of OFM scale < 1e-38
    shp = [1, 10, 20, 16]
    op = testutil.create_elemwise_op(Op.Mul, "mul", shp, shp, shp, ofm_quant=testutil.default_quant_params(),)
    assert support.is_operator_supported(op)
    op.ofm.quantization.scale_f32 = 1e-43
    assert not support.is_operator_supported(op)


def test_constraint_depth_multiplier():
    # Valid. Depth multiplier is 1 so no further constraints
    op = testutil.create_op_with_quant_tensors(
        Op.DepthwiseConv2DBias, [1, 1, 1, 1], [1, 1, 1, 2], weights_shape=[1, 1, 1, 1]
    )
    op.attrs = {"stride_w": 1, "stride_h": 1, "depth_multiplier": 1}
    assert support.is_operator_supported(op)
    # Invalid. Depth multiplier doesnt equal ofm channel
    op = testutil.create_op_with_quant_tensors(
        Op.DepthwiseConv2DBias, [1, 1, 1, 1], [1, 1, 1, 1], weights_shape=[1, 1, 1, 1]
    )
    op.attrs = {"stride_w": 1, "stride_h": 1, "depth_multiplier": 2}
    assert not support.is_operator_supported(op)
    # Valid. Depth multiplier is equal to ofm channel
    op = testutil.create_op_with_quant_tensors(
        Op.DepthwiseConv2DBias, [1, 1, 1, 1], [1, 1, 1, 2], weights_shape=[1, 1, 1, 1]
    )
    op.attrs = {"stride_w": 1, "stride_h": 1, "depth_multiplier": 2}
    assert support.is_operator_supported(op)


def test_constraint_tconv_stride():
    # Strides must be 2
    op = testutil.create_op_with_quant_tensors(Op.Conv2DBackpropInput, [0], [1, 2, 2, 1], weights_shape=[1, 1, 1, 1])
    op.attrs = {"stride_w": 1, "stride_h": 1, "padding": Padding.SAME}
    ifm = Tensor([1, 1, 1, 1], DataType.uint8, "ifm")
    ifm.quantization = testutil.default_quant_params()
    op.add_input_tensor(ifm)
    assert not support.is_operator_supported(op)


def test_constraint_tconv_same():
    # Valid
    op = testutil.create_op_with_quant_tensors(Op.Conv2DBackpropInput, [0], [1, 2, 2, 1], weights_shape=[1, 1, 1, 1])
    op.attrs = {"stride_w": 2, "stride_h": 2, "padding": Padding.SAME}
    ifm = Tensor([1, 1, 1, 1], DataType.uint8, "ifm")
    ifm.quantization = testutil.default_quant_params()
    op.add_input_tensor(ifm)
    assert support.is_operator_supported(op)
    # Invalid
    op = testutil.create_op_with_quant_tensors(Op.Conv2DBackpropInput, [0], [1, 4, 4, 1], weights_shape=[1, 1, 1, 1])
    op.attrs = {"stride_w": 2, "stride_h": 2, "padding": Padding.SAME}
    ifm = Tensor([1, 1, 1, 1], DataType.uint8, "ifm")
    ifm.quantization = testutil.default_quant_params()
    op.add_input_tensor(ifm)
    assert not support.is_operator_supported(op)


def test_constraint_tconv_valid():
    # Valid
    op = testutil.create_op_with_quant_tensors(Op.Conv2DBackpropInput, [0], [1, 4, 4, 1], weights_shape=[4, 4, 1, 1])
    op.attrs = {"stride_w": 2, "stride_h": 2, "padding": Padding.VALID}
    ifm = Tensor([1, 1, 1, 1], DataType.uint8, "ifm")
    ifm.quantization = testutil.default_quant_params()
    op.add_input_tensor(ifm)
    assert support.is_operator_supported(op)
    # Invalid
    op = testutil.create_op_with_quant_tensors(Op.Conv2DBackpropInput, [0], [1, 4, 4, 1], weights_shape=[2, 2, 1, 1])
    op.attrs = {"stride_w": 2, "stride_h": 2, "padding": Padding.VALID}
    ifm = Tensor([1, 1, 1, 1], DataType.uint8, "ifm")
    ifm.quantization = testutil.default_quant_params()
    op.add_input_tensor(ifm)
    assert not support.is_operator_supported(op)


def test_constraint_matching_in_out_types():
    # Valid
    op = testutil.create_op_with_quant_tensors(Op.AvgPool, [1, 8, 8, 8], [1, 8, 8, 8])
    op.attrs = {"stride_w": 2, "stride_h": 2, "filter_width": 2, "filter_height": 2, "padding": Padding.SAME}
    assert support.is_operator_supported(op)
    # Invalid. datatypes for ifm and ofm must match (default uint8)
    op.ifm.dtype = DataType.int8
    assert not support.is_operator_supported(op)


def test_constraint_filter_type():
    # Filter width/height must be integers
    op = testutil.create_op_with_quant_tensors(Op.AvgPool, [1, 8, 8, 8], [1, 8, 8, 8])
    op.attrs = {"stride_w": 2, "stride_h": 2, "filter_width": 2.5, "filter_height": "2", "padding": Padding.SAME}
    assert not support.is_operator_supported(op)


def test_constraint_filter_range():
    # Avg pool restrictions are dependent on padding:
    # SAME padding restricts both W and H to max 8
    op = testutil.create_op_with_quant_tensors(Op.AvgPool, [1, 8, 8, 8], [1, 8, 8, 8])
    op.attrs = {"stride_w": 2, "stride_h": 2, "filter_width": 20, "filter_height": 20, "padding": Padding.SAME}
    assert not support.is_operator_supported(op)
    # VALID padding limits are much larger
    op.attrs["padding"] = Padding.VALID
    assert support.is_operator_supported(op)


def test_constraint_filter_height_range_valid_pad():
    # Avg pool restrictions are dependent on padding:
    op = testutil.create_op_with_quant_tensors(Op.AvgPool, [1, 8, 8, 8], [1, 8, 8, 8])
    op.attrs = {"stride_w": 2, "stride_h": 2, "filter_width": 2, "filter_height": 256, "padding": Padding.VALID}
    assert support.is_operator_supported(op)
    # VALID padding restricts to 256 in filter height
    op.attrs["filter_height"] = 257
    assert not support.is_operator_supported(op)


def test_constraint_filter_product_height_range_valid_pad():
    # Avg pool restrictions are dependent on padding:
    op = testutil.create_op_with_quant_tensors(Op.AvgPool, [1, 8, 8, 8], [1, 8, 8, 8])
    op.attrs = {"stride_w": 2, "stride_h": 2, "filter_width": 256, "filter_height": 256, "padding": Padding.VALID}
    assert support.is_operator_supported(op)
    # VALID padding restricts filter W x H to 256x256
    op.attrs["filter_width"] = 257
    assert not support.is_operator_supported(op)


def test_constraint_filter_height_range():
    # Max pool restrictions arent dependent on padding
    op = testutil.create_op_with_quant_tensors(Op.MaxPool, [1, 8, 8, 8], [1, 8, 8, 8])
    op.attrs = {"stride_w": 2, "stride_h": 2, "filter_width": 2, "filter_height": 256, "padding": Padding.SAME}
    assert support.is_operator_supported(op)
    # Restricts to 256 in filter height
    op.attrs["filter_height"] = 257
    assert not support.is_operator_supported(op)
    # Doesnt matter if SAME or VALID
    op.attrs["padding"] = Padding.VALID
    assert not support.is_operator_supported(op)


def test_constraint_filter_product_height_range():
    # Max pool restrictions arent dependent on padding
    op = testutil.create_op_with_quant_tensors(Op.MaxPool, [1, 8, 8, 8], [1, 8, 8, 8])
    op.attrs = {"stride_w": 2, "stride_h": 2, "filter_width": 256, "filter_height": 256, "padding": Padding.SAME}
    assert support.is_operator_supported(op)
    # Restricts filter W x H to 256x256
    op.attrs["filter_width"] = 257
    assert not support.is_operator_supported(op)
    # Doesnt matter if SAME or VALID
    op.attrs["padding"] = Padding.VALID
    assert not support.is_operator_supported(op)


def test_constraint_resize():
    # IFM W and H == 1
    op = testutil.create_op_with_quant_tensors(Op.ResizeBilinear, [1, 1, 1, 8], [1, 8, 8, 8])
    assert support.is_operator_supported(op)
    # IFM == OFM
    op = testutil.create_op_with_quant_tensors(Op.ResizeBilinear, [1, 8, 8, 8], [1, 8, 8, 8])
    assert support.is_operator_supported(op)
    # IFM x2 == OFM ; align_corners = False
    op = testutil.create_op_with_quant_tensors(Op.ResizeBilinear, [1, 4, 4, 8], [1, 8, 8, 8])
    assert support.is_operator_supported(op)
    # IFM x2 -1 == OFM ; align_corners = True
    op = testutil.create_op_with_quant_tensors(Op.ResizeBilinear, [1, 4, 4, 8], [1, 7, 7, 8])
    op.attrs["align_corners"] = True
    assert support.is_operator_supported(op)
    # Invalid cases
    op = testutil.create_op_with_quant_tensors(Op.ResizeBilinear, [1, 4, 4, 8], [1, 20, 20, 8])
    assert not support.is_operator_supported(op)
    op.attrs["align_corners"] = True
    assert not support.is_operator_supported(op)


def test_constraint_matching_shapes():
    # Softmax requires the ifm and ofm shapes to match
    op = testutil.create_op_with_quant_tensors(Op.Softmax, [1, 1, 1, 8], [1, 2, 2, 4])
    assert not support.is_operator_supported(op)
    op = testutil.create_op_with_quant_tensors(Op.Softmax, [1, 1, 1, 8], [1, 1, 1, 8])
    assert support.is_operator_supported(op)


def test_constraint_beta_value_range():
    # beta must be positive
    op = testutil.create_op_with_quant_tensors(Op.Softmax, [1, 1, 1, 8], [1, 1, 1, 8])
    op.attrs["beta"] = -1.0
    assert not support.is_operator_supported(op)
    op.attrs["beta"] = 0.0
    assert support.is_operator_supported(op)


def test_constraint_splitv_inferred():
    # SplitV requires a maximum of one inferred shape (-1)
    qp = testutil.default_quant_params()
    op = testutil.create_op_with_quant_tensors(Op.SplitV, [1, 1, 1, 8], [1, 1, 1, 8])
    sizes = create_const_tensor("sizes", [1, 1, 1, 4], DataType.int16, [[[[0, -1, 2, -1]]]], np.int16, quantization=qp)
    op.add_input_tensor(sizes)
    assert not support.is_operator_supported(op)
    op = testutil.create_op_with_quant_tensors(Op.SplitV, [1, 1, 1, 8], [1, 1, 1, 8])
    sizes = create_const_tensor("sizes", [1, 1, 1, 4], DataType.int16, [[[[0, 1, 2, -1]]]], np.int16, quantization=qp)
    op.add_input_tensor(sizes)
    assert support.is_operator_supported(op)


def test_constraint_concat_pass():
    # A working concat
    op = testutil.create_op_with_quant_tensors(Op.Concat, [1, 1, 1, 4], [1, 1, 1, 8])
    ifm2 = Tensor([1, 1, 1, 4], DataType.uint8, "in2")
    ifm2.quantization = testutil.default_quant_params()
    op.add_input_tensor(ifm2)
    op.attrs["axis"] = 3
    assert support.is_operator_supported(op)


def test_constraint_axis_exists():
    # Missing axis attribute
    op = testutil.create_op_with_quant_tensors(Op.Concat, [1, 1, 1, 4], [1, 1, 1, 8])
    ifm2 = Tensor([1, 1, 1, 4], DataType.uint8, "in2")
    ifm2.quantization = testutil.default_quant_params()
    op.add_input_tensor(ifm2)
    assert not support.is_operator_supported(op)


def test_constraint_axis_valid():
    # Invalid axis attribute
    op = testutil.create_op_with_quant_tensors(Op.Concat, [1, 1, 1, 4], [1, 1, 1, 8])
    ifm2 = Tensor([1, 1, 1, 4], DataType.uint8, "in2")
    ifm2.quantization = testutil.default_quant_params()
    op.add_input_tensor(ifm2)
    op.attrs["axis"] = 7
    assert not support.is_operator_supported(op)


def test_constraint_matching_dimensionality():
    # Mismatching dimensionality: 4D+2D=4D
    op = testutil.create_op_with_quant_tensors(Op.Concat, [1, 1, 1, 4], [1, 1, 1, 8])
    ifm2 = Tensor([1, 4], DataType.uint8, "in2")
    ifm2.quantization = testutil.default_quant_params()
    op.add_input_tensor(ifm2)
    op.attrs["axis"] = 3
    assert not support.is_operator_supported(op)


def test_constraint_valid_dimensions():
    # Mismatching dimension value:
    # ifm2 has w and h as 2, which is not the axis to concat and doesnt match ifm1 or ofm
    op = testutil.create_op_with_quant_tensors(Op.Concat, [1, 1, 1, 4], [1, 1, 1, 8])
    ifm2 = Tensor([1, 2, 2, 4], DataType.uint8, "in2")
    ifm2.quantization = testutil.default_quant_params()
    op.add_input_tensor(ifm2)
    op.attrs["axis"] = 3
    assert not support.is_operator_supported(op)


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
    pad_setting=Padding.VALID,
    kernel_size=3,
):
    qp = testutil.default_quant_params()
    in0 = Tensor(in_shape, in_dtype, "in")
    in0.quantization = qp
    pad_tensor = create_const_tensor(name="pad", shape=list(np.shape(padding)), values=padding, dtype=pad_dtype)
    out = Tensor(out_shape, out_dtype, "out")
    out.quantization = qp.clone()
    op = testutil.create_op(Op.Pad, [in0, pad_tensor], out)
    conv_out_tens = Tensor(in_shape, in_dtype, "output")
    conv_out_tens.quantization = qp.clone()
    weight_tens = Tensor([kernel_size, kernel_size, in_shape[-1], out_shape[-1]], in_dtype, "weights")
    weight_tens.values = np.zeros(weight_tens.shape)
    weight_tens.quant_values = np.zeros(weight_tens.shape, np.int8)
    weight_tens.quantization = qp.clone()
    bias_tens = Tensor(out_shape, pad_dtype, "biases")
    attrs = {"padding": pad_setting, "stride_w": 2, "stride_h": 2, "dilation_w_factor": 1, "dilation_h_factor": 1}
    attrs["strides"] = (1, attrs["stride_h"], attrs["stride_w"], 1)
    conv2d_op = testutil.create_op(Op.Conv2DBias, [out, weight_tens, bias_tens], conv_out_tens, attrs)
    conv2d_op.add_input_tensor(out)
    return op


def test_constraint_pad_input_count():
    # Incorrect number of input tensors (2)
    op = create_pad_op(in_shape=[1, 1, 1, 1], out_shape=[1, 3, 3, 1], padding=[[0, 0], [1, 1], [1, 1], [0, 0]],)
    assert support.is_operator_supported(op)
    op.add_input_tensor(op.inputs[0].clone())
    assert not support.is_operator_supported(op)


def test_constraint_padded_dimensions():
    # Incorrect padding dimensions, can only pad width and height
    op = create_pad_op(in_shape=[1, 1, 1, 1], out_shape=[1, 3, 3, 1], padding=[[1, 1], [1, 1], [1, 1], [0, 0]],)
    assert not support.is_operator_supported(op)


def test_constraint_pad_shape():
    # PAD operator must be of shape (4,2)
    op = create_pad_op(in_shape=[1, 1, 1, 1], out_shape=[1, 3, 3, 1], padding=[[0, 0], [1, 1], [1, 1], [0, 0], [0, 0]],)
    assert not support.is_operator_supported(op)


def test_constraint_pad_none():
    op = create_pad_op(in_shape=[1, 1, 1, 1], out_shape=[1, 3, 3, 1], padding=[],)
    assert not support.is_operator_supported(op)


def test_constraint_pad_dtype():
    # PAD operator dtype should be int32 or int64
    op = create_pad_op(
        in_shape=[1, 1, 1, 1],
        out_shape=[1, 3, 3, 1],
        padding=[[0, 0], [1, 1], [1, 1], [0, 0], [0, 0]],
        pad_dtype=DataType.int16,
    )
    assert not support.is_operator_supported(op)


def test_constraint_pad_consumer():
    # PAD operator must be followed by a valid consumer with Padding.VALID attribute
    op = create_pad_op(in_shape=[1, 1, 1, 1], out_shape=[1, 3, 3, 1], padding=[[0, 0], [1, 1], [1, 1], [0, 0]],)
    assert support.is_operator_supported(op)
    op = create_pad_op(
        in_shape=[1, 1, 1, 1],
        out_shape=[1, 3, 3, 1],
        padding=[[0, 0], [1, 1], [1, 1], [0, 0]],
        pad_setting=Padding.SAME,
    )
    assert not support.is_operator_supported(op)
    op_consumer = testutil.create_op_with_quant_tensors(Op.ConcatTFLite, [1, 1, 1, 4], [1, 1, 1, 8])
    op.ofm.consumer_list = [op_consumer]
    assert not support.is_operator_supported(op)
    op_consumer = testutil.create_elemwise_op(Op.Add, "op", [1, 3, 3, 1], [1, 3, 3, 1], [1, 3, 3, 1])
    op.ofm.consumer_list = [op_consumer]
    assert not support.is_operator_supported(op)


pad_invalid_size_test_data = [
    (2, 1, 1, 1),
    (1, 2, 1, 1),
    (1, 1, 2, 1),
    (1, 1, 1, 2),
]


@pytest.mark.parametrize("top, left, bottom, right", pad_invalid_size_test_data)
def test_constraint_pad_size(top, left, bottom, right):
    # Tests PAD operator with a padding that is too high to be handled by the NPU
    out_shape = [1, 11 + left + right, 11 + top + bottom, 1]
    padding = [[0, 0], [top, bottom], [left, right], [0, 0]]
    op = create_pad_op(in_shape=[1, 11, 11, 1], out_shape=out_shape, padding=padding,)
    assert not support.is_operator_supported(op)


leading_pad_test_data = [
    (2, 2, 11, True),
    (1, 2, 11, False),
    (2, 1, 11, False),
    (5, 2, 11, True),
]


@pytest.mark.parametrize("top, left, kernel_size, expected", leading_pad_test_data)
def test_constraint_leading_pad_size(top, left, kernel_size, expected):
    # Tests PAD operator with big kernel size; top and left pad must be multiple of stride
    out_shape = [1, 11 + left, 11 + top, 1]
    padding = [[0, 0], [top, 0], [left, 0], [0, 0]]
    op = create_pad_op(in_shape=[1, 11, 11, 1], out_shape=out_shape, padding=padding, kernel_size=kernel_size)
    assert support.is_operator_supported(op) == expected


pad_avg_pool_test_data = [
    ((3, 3), (1, 1, 1, 1), True),
    ((2, 4), (1, 2, 1, 2), True),
    ((5, 3), (2, 1, 2, 1), True),
    ((5, 3), (0, 1, 2, 1), True),
    ((5, 3), (2, 0, 2, 1), True),
    ((5, 3), (2, 1, 0, 1), True),
    ((5, 3), (2, 1, 0, 1), True),
    ((4, 4), (2, 2, 2, 2), True),
    ((4, 4), (1, 2, 2, 2), False),
    ((4, 4), (2, 1, 2, 2), False),
    ((4, 4), (2, 2, 1, 2), False),
    ((4, 4), (2, 2, 2, 1), False),
]


@pytest.mark.parametrize("k_size, padding, expected", pad_avg_pool_test_data)
def test_pad_followed_by_avg_pool(k_size, padding, expected):
    # Tests PAD followed by AvgPool
    k_w, k_h = k_size
    top, left, bottom, right = padding
    pad_values = [[0, 0], [top, bottom], [left, right], [0, 0]]
    dtype = DataType.int8
    qp = testutil.default_quant_params()
    in_shape = [1, 15, 17, 8]
    out_shape = [1, in_shape[1] + top + bottom, in_shape[2] + left + right, in_shape[3]]
    in0 = Tensor(in_shape, dtype, "in")
    in0.quantization = qp
    pad_tensor = create_const_tensor(
        name="pad", shape=list(np.shape(pad_values)), values=pad_values, dtype=DataType.int32
    )
    out = Tensor(out_shape, dtype, "out")
    out.quantization = qp.clone()
    op = testutil.create_op(Op.Pad, [in0, pad_tensor], out)
    pool_out_tens = Tensor(in_shape, dtype, "output")
    pool_out_tens.quantization = qp.clone()
    attrs = {
        "padding": Padding.VALID,
        "ksize": [1, k_w, k_h, 1],
        "stride_w": 1,
        "stride_h": 1,
        "dilation_w_factor": 1,
        "dilation_h_factor": 1,
    }
    pool_op = testutil.create_op(Op.AvgPool, [out], pool_out_tens, attrs)
    pool_op.add_input_tensor(out)
    assert support.is_operator_supported(op) == expected


def create_strided_slice():
    # Creates a valid strided slice operator with some valid inputs/outputs
    op = create_strided_slice_op([1, 10, 10, 10], [1, 5, 5, 10], [127, 2, 2, 0], [0, 7, -3, 0])
    op.attrs["begin_mask"] = 1
    op.attrs["end_mask"] = 9
    assert support.is_operator_supported(op)
    return op


def test_constraint_stridedslice_input_count():
    # Wrong number of input tensors
    op = create_strided_slice()
    op.add_input_tensor(op.inputs[0].clone())
    assert not support.is_operator_supported(op)


def test_constraint_stridedslice_inputs_const():
    # begin, end, stride values must not be None
    op = create_strided_slice()
    op.inputs[1].values = None
    assert not support.is_operator_supported(op)
    op = create_strided_slice()
    op.inputs[2].values = None
    assert not support.is_operator_supported(op)
    op = create_strided_slice()
    op.inputs[3].values = None
    assert not support.is_operator_supported(op)


def test_constraint_stridedslice_stride_values():
    # Unsupported strides
    op = create_strided_slice()
    op.inputs[3].values = [1, 1, 2, 1]
    assert not support.is_operator_supported(op)


def test_constraint_ellipsis_mask():
    # Unsupported ellipsis mask
    op = create_strided_slice()
    op.attrs["ellipsis_mask"] = 1
    assert not support.is_operator_supported(op)


def test_constraint_axis_masks():
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


def test_constraint_slice_ranges():
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


def test_constraint_matching_inputs_types():
    # input data types must match (default is uint8)
    op = testutil.create_elemwise_op(Op.Mul, "op", [1, 8, 8, 8], [1, 8, 8, 8], [1, 8, 8, 8])
    op.ifm2.dtype = DataType.int8
    assert not support.is_operator_supported(op)


def test_constraint_matching_signed():
    # signed inputs require output to also be signed
    op = testutil.create_elemwise_op(Op.Mul, "op", [1, 8, 8, 8], [1, 8, 8, 8], [1, 8, 8, 8], datatype=DataType.int8)
    op.ofm.dtype = DataType.uint8
    assert not support.is_operator_supported(op)


def test_constraint_unsigned_valid():
    # unsigned inputs require output to be either:
    op = testutil.create_elemwise_op(Op.Mul, "op", [1, 8, 8, 8], [1, 8, 8, 8], [1, 8, 8, 8])
    # the same (default uint8)
    assert support.is_operator_supported(op)
    op.ofm.dtype = DataType.int8
    assert not support.is_operator_supported(op)
    op.ofm.dtype = DataType.int16
    assert not support.is_operator_supported(op)
    # or int32
    op.ofm.dtype = DataType.int32
    assert support.is_operator_supported(op)


def test_constraint_inputs_int32():
    # both inputs must be type int32
    op = testutil.create_elemwise_op(Op.SHL, "op", [1, 8, 8, 8], [1, 8, 8, 8], [1, 8, 8, 8])
    assert not support.is_operator_supported(op)
    op = testutil.create_elemwise_op(Op.SHL, "op", [1, 8, 8, 8], [1, 8, 8, 8], [1, 8, 8, 8], datatype=DataType.int32)
    assert support.is_operator_supported(op)
    op.ifm2.dtype = DataType.int16
    assert not support.is_operator_supported(op)


def test_constraint_output_int32():
    # output must be type int32
    op = testutil.create_elemwise_op(Op.SHL, "op", [1, 8, 8, 8], [1, 8, 8, 8], [1, 8, 8, 8], datatype=DataType.int32)
    assert support.is_operator_supported(op)
    op.ofm.dtype = DataType.int16
    assert not support.is_operator_supported(op)


def test_constraint_matching_quantization_parameters():
    qp = QuantizationParameters()
    qp.scale_f32 = np.float32(1.5)
    qp.zero_point = 128
    # valid - all matching (uses default quant params)
    op = testutil.create_elemwise_op(Op.Minimum, "op", [1, 8, 8, 8], [1, 8, 8, 8], [1, 8, 8, 8])
    assert support.is_operator_supported(op)
    # invalid - ifm mismatch ofm
    op.ifm.quantization = qp
    assert not support.is_operator_supported(op)
    # invalid - ifm2 mismatch ofm
    op = testutil.create_elemwise_op(Op.Minimum, "op", [1, 8, 8, 8], [1, 8, 8, 8], [1, 8, 8, 8])
    op.ifm2.quantization = qp
    assert not support.is_operator_supported(op)
    # invalid - both ifm and ifm2 mismatch ofm
    op = testutil.create_elemwise_op(Op.Minimum, "op", [1, 8, 8, 8], [1, 8, 8, 8], [1, 8, 8, 8])
    op.ifm.quantization = qp
    op.ifm2.quantization = qp
    assert not support.is_operator_supported(op)
    # valid - all matching
    op.ofm.quantization = qp
    assert support.is_operator_supported(op)
    op = testutil.create_elemwise_op(Op.Minimum, "op", [1, 8, 8, 8], None, [1, 8, 8, 8])
    assert support.is_operator_supported(op)


def test_constraint_elemwise_batch_size():
    # BINARY CASE
    # Batch can be >1 if dims is <=2D
    op = testutil.create_elemwise_op(Op.Add, "op", [2, 2], [2, 2], [2, 2])
    assert support.is_operator_supported(op)
    # For dims >2D, batch must be 1
    op = testutil.create_elemwise_op(Op.Add, "op", [1, 2, 2], [1, 2, 2], [1, 2, 2])
    assert support.is_operator_supported(op)
    # invalid case
    op = testutil.create_elemwise_op(Op.Add, "op", [2, 2, 2], [2, 2, 2], [2, 2, 2])
    assert not support.is_operator_supported(op)

    # UNARY CASE
    # Batch can be >1 if dims is <=2D
    op = testutil.create_elemwise_op(Op.CLZ, "op", [2, 2], None, [2, 2], datatype=DataType.int32)
    assert support.is_operator_supported(op)
    # For dims >2D, batch must be 1
    op = testutil.create_elemwise_op(Op.CLZ, "op", [1, 2, 2], None, [1, 2, 2], datatype=DataType.int32)
    assert support.is_operator_supported(op)
    # invalid case
    op = testutil.create_elemwise_op(Op.CLZ, "op", [2, 2, 2], None, [2, 2, 2], datatype=DataType.int32)
    assert not support.is_operator_supported(op)


def test_constraint_matching_either_shapes():
    # BINARY CASE
    # At least one ifm shape must match ofm's shape
    op = testutil.create_elemwise_op(Op.Add, "op", [1, 4], [4, 4], [4, 4])
    assert support.is_operator_supported(op)
    op = testutil.create_elemwise_op(Op.Add, "op", [4, 4], [1, 4], [4, 4])
    assert support.is_operator_supported(op)
    op = testutil.create_elemwise_op(Op.Add, "op", [4, 4], [4, 4], [2, 2])
    assert not support.is_operator_supported(op)
    op = testutil.create_elemwise_op(Op.Add, "op", [1, 4, 1, 16], [1, 1, 4, 1], [1, 4, 4, 16])
    assert not support.is_operator_supported(op)
    op = testutil.create_elemwise_op(Op.Add, "op", [1, 1, 4, 1], [1, 4, 1, 16], [1, 4, 4, 16])
    assert not support.is_operator_supported(op)

    # UNARY CASE
    # No second input so this is treated the same as requiring ifm shape to match ofm shape
    op = testutil.create_elemwise_op(Op.CLZ, "op", [2, 2], None, [2, 2], datatype=DataType.int32)
    assert support.is_operator_supported(op)
    op = testutil.create_elemwise_op(Op.CLZ, "op", [4, 4], None, [2, 2], datatype=DataType.int32)
    assert not support.is_operator_supported(op)


def test_constraint_broadcast_shapes():
    # BINARY CASE
    # Only allow broadcast to 1 dim, for 1 rank index
    op = testutil.create_elemwise_op(Op.Add, "op", [1, 1, 4], [1, 2, 4], [1, 2, 4])
    assert support.is_operator_supported(op)
    op = testutil.create_elemwise_op(Op.Add, "op", [1, 2, 4], [1, 1, 4], [1, 2, 4])
    assert support.is_operator_supported(op)
    # Only allow broadcast to 1 dim, for 3 rank indexes
    op = testutil.create_elemwise_op(Op.Add, "op", [1, 1, 1, 1], [1, 4, 8, 16], [1, 4, 8, 16])
    assert support.is_operator_supported(op)
    op = testutil.create_elemwise_op(Op.Add, "op", [1, 4, 8, 16], [1, 1, 1, 1], [1, 4, 8, 16])
    assert support.is_operator_supported(op)
    # One broadcast dim not 1
    op = testutil.create_elemwise_op(Op.Add, "op", [1, 2, 4], [1, 4, 4], [1, 4, 4])
    assert not support.is_operator_supported(op)
    op = testutil.create_elemwise_op(Op.Add, "op", [1, 4, 4], [1, 2, 4], [1, 4, 4])
    assert not support.is_operator_supported(op)
    # OFM shape dim largest ifm/ifm2 shape dim
    op = testutil.create_elemwise_op(Op.Add, "op", [1, 4], [4, 4], [1, 4])
    assert not support.is_operator_supported(op)
    op = testutil.create_elemwise_op(Op.Add, "op", [1, 4], [4, 4], [1, 4])
    assert not support.is_operator_supported(op)
    op = testutil.create_elemwise_op(Op.Add, "op", [1, 4, 1, 16], [1, 1, 4, 1], [1, 4, 1, 16])
    assert not support.is_operator_supported(op)
    op = testutil.create_elemwise_op(Op.Add, "op", [1, 1, 4, 1], [1, 4, 1, 16], [1, 4, 1, 16])
    assert not support.is_operator_supported(op)


def test_constraint_alpha_valid():
    # Alpha cannot be negative
    op = testutil.create_elemwise_op(Op.LeakyRelu, "op", [2, 2], None, [2, 2])
    op.attrs["alpha"] = 0
    assert support.is_operator_supported(op)
    op.attrs["alpha"] = -1
    assert not support.is_operator_supported(op)


def test_constraint_hardswish_dtype():
    # HardSwish operator dtype should be int8 or uint8, and input dtype must match output
    # UINT8
    op = testutil.create_op_with_quant_tensors(Op.HardSwish, [1, 8, 8, 8], [1, 8, 8, 8])
    assert support.is_operator_supported(op)
    # INT8
    op = testutil.create_op_with_quant_tensors(Op.HardSwish, [1, 8, 8, 8], [1, 8, 8, 8], datatype=DataType.int8)
    assert support.is_operator_supported(op)

    # Invalid
    op = testutil.create_op_with_quant_tensors(Op.HardSwish, [1, 8, 8, 8], [1, 8, 8, 8], datatype=DataType.int16)
    assert not support.is_operator_supported(op)
    op = testutil.create_op_with_quant_tensors(Op.HardSwish, [1, 8, 8, 8], [1, 8, 8, 8], datatype=DataType.uint16)
    assert not support.is_operator_supported(op)
    op = testutil.create_op_with_quant_tensors(Op.HardSwish, [1, 8, 8, 8], [1, 8, 8, 8], datatype=DataType.int32)
    assert not support.is_operator_supported(op)

    in_tens = Tensor([1, 8, 8, 8], DataType.int8, "in")
    out_tens = Tensor([1, 8, 8, 8], DataType.uint8, "out")
    op = testutil.create_op(Op.HardSwish, [in_tens], out_tens)
    assert not support.is_operator_supported(op)


def test_constraint_keep_dims_ifm_ofm():
    op = testutil.create_op_with_quant_tensors(Op.FullyConnected, [4, 8, 8, 4], [32, 32], weights_shape=[4, 8, 8, 4])
    op.attrs["keep_num_dims"] = True
    assert not support.is_operator_supported(op)
    op.attrs["keep_num_dims"] = False
    assert support.is_operator_supported(op)


def create_mean(input_shape, output_shape, indices, datatype, attrs):
    ifm = Tensor(input_shape, datatype, "in")
    ifm.quantization = testutil.default_quant_params()
    indices = create_const_tensor("indices", [len(indices)], DataType.int32, indices, np.uint8)
    ofm = Tensor(output_shape, datatype, "out")
    ofm.quantization = testutil.default_quant_params()
    op = testutil.create_op(Op.Mean, [ifm, indices], ofm, attrs)
    return op


def test_mean_dtype():
    op = create_mean([1, 6, 6, 16], [1, 1, 1, 16], [1, 2], DataType.int8, {"keep_dims": True})
    assert support.is_operator_supported(op)
    op.ifm.dtype = DataType.int16
    op.ofm.dtype = DataType.int16
    assert not support.is_operator_supported(op)


def test_mean_properties():
    op = create_mean([1, 6, 6, 256], [1, 1, 256], [1, 2], DataType.uint8, {})
    assert support.is_operator_supported(op)
    op.ifm.quantization.zero_point = 55
    assert not support.is_operator_supported(op)


def test_mean_axis():
    op = create_mean([1, 6, 6, 16], [1, 1, 1, 16], [1], DataType.int8, {"keep_dims": True})
    assert not support.is_operator_supported(op)


def test_mean_hw_product():
    op = create_mean([1, 64, 64, 16], [1, 1, 16], [1, 2], DataType.uint8, {})
    assert support.is_operator_supported(op)
    op = create_mean([1, 65, 64, 16], [1, 1, 1, 16], [1, 2], DataType.int8, {"keep_dims": True})
    assert not support.is_operator_supported(op)


def test_mean_hw_product_int8():
    op = create_mean([1, 16, 16, 16], [1, 1, 1, 16], [1, 2], DataType.int8, {"keep_dims": True})
    assert support.is_operator_supported(op)
    op = create_mean([1, 16, 17, 16], [1, 1, 1, 16], [1, 2], DataType.int8, {"keep_dims": True})
    assert not support.is_operator_supported(op)
