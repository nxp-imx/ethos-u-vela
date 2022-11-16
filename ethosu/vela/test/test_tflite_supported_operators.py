# SPDX-FileCopyrightText: Copyright 2020-2022 Arm Limited and/or its affiliates <open-source-office@arm.com>
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
# Unit tests for tflite support_operators
import numpy as np

from ethosu.vela.data_type import DataType
from ethosu.vela.operation import ActivationFunction
from ethosu.vela.operation import Op
from ethosu.vela.operation import Padding
from ethosu.vela.tensor import create_const_tensor
from ethosu.vela.tensor import QuantizationParameters
from ethosu.vela.tensor import Tensor
from ethosu.vela.test import testutil
from ethosu.vela.tflite_supported_operators import TFLiteSupportedOperators

support = TFLiteSupportedOperators()


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
    op = testutil.create_op_with_quant_tensors(Op.Conv2DBias, [1, 1, 1, 1], [1, 1, 1, 1], weights_shape=[1, 1, 1, 1])
    op.attrs = {"stride_w": 1, "stride_h": 1}
    assert support.is_operator_supported(op)


def test_constraint_stride_range():
    # Stride width and height must lie within a certain range
    op = testutil.create_op_with_quant_tensors(Op.Conv2DBias, [1, 8, 8, 8], [1, 8, 8, 8])
    op.attrs = {"stride_w": 0, "stride_h": 20}
    assert not support.is_operator_supported(op)


def test_constraint_dilated_height_range():
    # Dilated kernel height must lie within a certain range
    op = testutil.create_op_with_quant_tensors(Op.Conv2DBias, [1, 8, 8, 8], [1, 8, 8, 8], weights_shape=[65, 64, 1, 1])
    op.attrs = {"stride_w": 1, "stride_h": 1}
    assert not support.is_operator_supported(op)


def test_constraint_dilated_product_range():
    # Dilated kernel width x height must lie within a certain range
    op = testutil.create_op_with_quant_tensors(Op.Conv2DBias, [1, 8, 8, 8], [1, 8, 8, 8], weights_shape=[64, 65, 1, 1])
    op.attrs = {"stride_w": 1, "stride_h": 1}
    assert not support.is_operator_supported(op)


def test_constraint_weights_type():
    # Weight tensor must be 8-bit
    op = testutil.create_op_with_quant_tensors(
        Op.Conv2DBias, [1, 8, 8, 8], [1, 8, 8, 8], weights_shape=[1, 1, 1, 1], datatype=DataType.int16
    )
    op.attrs = {"stride_w": 1, "stride_h": 1}
    assert not support.is_operator_supported(op)


def test_constraint_weights_const():
    # Weight tensor cannot be non-const tensors
    op = testutil.create_op_with_quant_tensors(Op.Conv2DBias, [1, 8, 8, 8], [1, 8, 8, 8])
    op.attrs = {"stride_w": 1, "stride_h": 1}
    weights = Tensor([64, 64, 1, 1], DataType.uint8, "weights")
    weights.quantization = testutil.default_quant_params()
    op.add_input_tensor(weights)
    assert not support.is_operator_supported(op)


def test_constraint_weights_limit():
    # Sum of weights has a limit
    op = testutil.create_op_with_quant_tensors(Op.Conv2DBias, [1, 8, 8, 8], [1, 8, 8, 8], weights_shape=[1, 1, 1, 1])
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
    bias.values = np.array([0x01FF_FFFF_FFFF])
    op.add_input_tensor(bias)
    assert not support.is_operator_supported(op)


def test_constraint_batch_size():
    op = testutil.create_op_with_quant_tensors(Op.Conv2DBias, [2, 8, 8, 8], [1, 8, 8, 8], weights_shape=[1, 1, 1, 1])
    op.attrs = {"stride_w": 1, "stride_h": 1}
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
    for resize_op in Op.op_set(Op.is_resize_op):
        # IFM W and H == 1
        op = testutil.create_op_with_quant_tensors(resize_op, [1, 1, 1, 8], [1, 8, 8, 8])
        op.add_input_tensor(create_const_tensor("size", [2], DataType.int32, [8, 8], np.int32))
        assert support.is_operator_supported(op)

        # IFM == OFM
        op = testutil.create_op_with_quant_tensors(resize_op, [1, 8, 8, 8], [1, 8, 8, 8])
        op.add_input_tensor(create_const_tensor("size", [2], DataType.int32, [8, 8], np.int32))
        assert support.is_operator_supported(op)

        # IFM x2 == OFM ; align_corners = False
        op = testutil.create_op_with_quant_tensors(resize_op, [1, 4, 4, 8], [1, 8, 8, 8])
        op.add_input_tensor(create_const_tensor("size", [2], DataType.int32, [8, 8], np.int32))
        assert support.is_operator_supported(op)

        # IFM x4 == OFM ; align_corners = False
        op = testutil.create_op_with_quant_tensors(resize_op, [1, 4, 4, 8], [1, 16, 16, 8])
        op.add_input_tensor(create_const_tensor("size", [2], DataType.int32, [16, 16], np.int32))
        assert support.is_operator_supported(op)

        # IFM x8 == OFM ; align_corners = False
        op = testutil.create_op_with_quant_tensors(resize_op, [1, 4, 4, 8], [1, 32, 32, 8])
        op.add_input_tensor(create_const_tensor("size", [2], DataType.int32, [32, 32], np.int32))
        assert support.is_operator_supported(op)

        # IFM -1 x2 == OFM -1 ; align_corners = True
        op = testutil.create_op_with_quant_tensors(resize_op, [1, 4, 4, 8], [1, 7, 7, 8])
        op.add_input_tensor(create_const_tensor("size", [2], DataType.int32, [7, 7], np.int32))
        op.attrs["align_corners"] = True
        assert support.is_operator_supported(op)

        # IFM -1 x4 == OFM -1 ; align_corners = True
        op = testutil.create_op_with_quant_tensors(resize_op, [1, 4, 4, 8], [1, 13, 13, 8])
        op.add_input_tensor(create_const_tensor("size", [2], DataType.int32, [13, 13], np.int32))
        op.attrs["align_corners"] = True
        assert support.is_operator_supported(op)

        # IFM -1 x8 == OFM -1 ; align_corners = True
        op = testutil.create_op_with_quant_tensors(resize_op, [1, 4, 4, 8], [1, 25, 25, 8])
        op.add_input_tensor(create_const_tensor("size", [2], DataType.int32, [25, 25], np.int32))
        op.attrs["align_corners"] = True
        assert support.is_operator_supported(op)

        # Invalid case - upscale size
        op = testutil.create_op_with_quant_tensors(resize_op, [1, 4, 4, 8], [1, 17, 17, 8])
        op.add_input_tensor(create_const_tensor("size", [2], DataType.int32, [17, 17], np.int32))
        assert not support.is_operator_supported(op)

        # Invalid case - upscale size with align corners
        op = testutil.create_op_with_quant_tensors(resize_op, [1, 4, 4, 8], [1, 15, 15, 8])
        op.add_input_tensor(create_const_tensor("size", [2], DataType.int32, [15, 15], np.int32))
        op.attrs["align_corners"] = True
        assert not support.is_operator_supported(op)


def test_constraint_resize_size():
    for resize_op in Op.op_set(Op.is_resize_op):
        # Invalid case - size != ofm size
        op = testutil.create_op_with_quant_tensors(resize_op, [1, 4, 4, 8], [1, 8, 8, 8])
        op.add_input_tensor(create_const_tensor("size", [2], DataType.int32, [7, 7], np.int32))
        assert not support.is_operator_supported(op)


def test_constraint_resize_attrs():
    for resize_op in Op.op_set(Op.is_resize_op):
        # Invalid case - both align corners and half-pixel centers
        op = testutil.create_op_with_quant_tensors(resize_op, [1, 4, 4, 8], [1, 8, 8, 8])
        op.add_input_tensor(create_const_tensor("size", [2], DataType.int32, [8, 8], np.int32))
        op.attrs["align_corners"] = True
        op.attrs["half_pixel_centers"] = True
        assert not support.is_operator_supported(op)


def test_constraint_concat_pass():
    # A working concat
    op = testutil.create_op_with_quant_tensors(Op.Concat, [1, 1, 1, 4], [1, 1, 1, 8])
    ifm2 = Tensor([1, 1, 1, 4], DataType.uint8, "in2")
    ifm2.quantization = testutil.default_quant_params()
    op.add_input_tensor(ifm2)
    op.attrs["axis"] = 3
    assert support.is_operator_supported(op)


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
    pad_tensor = create_const_tensor(name="pad", shape=list(np.shape(padding)), values=padding, dtype=pad_dtype)
    out = Tensor(out_shape, out_dtype, "out")
    out.quantization = qp.clone()
    op = testutil.create_op(Op.Pad, [in0, pad_tensor], out)
    return op


def test_constraint_padded_dimensions():
    # Incorrect padding dimensions, can only pad width and height
    op = create_pad_op(
        in_shape=[1, 1, 1, 1],
        out_shape=[1, 3, 3, 1],
        padding=[[1, 1], [1, 1], [1, 1], [0, 0]],
    )
    assert not support.is_operator_supported(op)
    op = create_pad_op(
        in_shape=[1, 1, 1, 1],
        out_shape=[1, 3, 3, 1],
        padding=[[1, 1], [1, 1], [0, 0]],
    )
    assert support.is_operator_supported(op)
    op = create_pad_op(
        in_shape=[1, 1, 1, 1],
        out_shape=[1, 3, 3, 1],
        padding=[[1, 1], [1, 1], [0, 1]],
    )
    assert not support.is_operator_supported(op)


def test_constraint_pad_shape():
    # PAD operator must be of shape (3,2) or (4,2)
    op = create_pad_op(in_shape=[1, 1, 1, 1], out_shape=[1, 3, 3, 1], padding=[[1, 1], [1, 1], [0, 0]])
    assert support.is_operator_supported(op)
    op = create_pad_op(
        in_shape=[1, 1, 1, 1],
        out_shape=[1, 3, 3, 1],
        padding=[[0, 0], [1, 1], [1, 1], [0, 0], [0, 0]],
    )
    assert not support.is_operator_supported(op)


def test_constraint_pad_none():
    op = create_pad_op(
        in_shape=[1, 1, 1, 1],
        out_shape=[1, 3, 3, 1],
        padding=[],
    )
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


def create_strided_slice():
    # Creates a valid strided slice operator with some valid inputs/outputs
    op = create_strided_slice_op([1, 10, 10, 10], [1, 5, 5, 10], [127, 2, 2, 0], [0, 7, -3, 0])
    op.attrs["begin_mask"] = 1
    op.attrs["end_mask"] = 9
    assert support.is_operator_supported(op)
    return op


def test_constraint_stridedslice_stride_values():
    # Unsupported strides
    op = create_strided_slice()
    op.inputs[3].values = [1, 1, 2, 1]
    assert not support.is_operator_supported(op)


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
    # Batch can be >1 if dims is <=3D
    op = testutil.create_elemwise_op(Op.Add, "op", [2, 2, 2], [2, 2, 2], [2, 2, 2])
    assert support.is_operator_supported(op)
    # For dims >3D, batch must be 1
    op = testutil.create_elemwise_op(Op.Add, "op", [1, 2, 2, 2], [1, 2, 2, 2], [1, 2, 2, 2])
    assert support.is_operator_supported(op)
    # invalid case
    op = testutil.create_elemwise_op(Op.Add, "op", [2, 2, 2, 2], [2, 2, 2, 2], [2, 2, 2, 2])
    assert not support.is_operator_supported(op)

    # UNARY CASE
    # Batch can be >1 if dims is <=3D
    op = testutil.create_elemwise_op(Op.CLZ, "op", [2, 2, 2], None, [2, 2, 2], datatype=DataType.int32)
    assert support.is_operator_supported(op)
    # For dims >3D, batch must be 1
    op = testutil.create_elemwise_op(Op.CLZ, "op", [1, 2, 2, 2], None, [1, 2, 2, 2], datatype=DataType.int32)
    assert support.is_operator_supported(op)
    # invalid case
    op = testutil.create_elemwise_op(Op.CLZ, "op", [2, 2, 2, 2], None, [2, 2, 2, 2], datatype=DataType.int32)
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


def create_mean(input_shape, output_shape, axis, datatype, attrs):
    ifm = Tensor(input_shape, datatype, "in")
    ifm.quantization = testutil.default_quant_params()
    ofm = Tensor(output_shape, datatype, "out")
    ofm.quantization = testutil.default_quant_params()
    if type(axis) is list:
        indices = create_const_tensor("indices", [len(axis)], DataType.int32, axis, np.uint8)
    elif type(axis) is int:
        indices = create_const_tensor("indices", [], DataType.int32, axis, np.uint8)
    op = testutil.create_op(Op.Mean, [ifm, indices], ofm, attrs)
    return op


def test_mean_hw_product():
    op = create_mean([1, 64, 64, 16], [1, 16], [1, 2], DataType.uint8, {})
    assert support.is_operator_supported(op)
    op = create_mean([1, 65, 64, 16], [1, 1, 1, 16], [1, 2], DataType.int8, {"keep_dims": True})
    assert not support.is_operator_supported(op)


def test_mean_hw_product_avgpool():
    op = create_mean([1, 200, 200, 16], [1, 16], [1, 2], DataType.uint8, {"keep_dims": False})
    assert support.is_operator_supported(op)
    op = create_mean([1, 200, 200, 16], [1, 1, 1, 16], [1, 2], DataType.int8, {"keep_dims": True})
    assert not support.is_operator_supported(op)
