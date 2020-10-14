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
# Description:
# The SupportedOperators class which is a collection of all supported operators and parameter checks.
from collections import defaultdict

import numpy as np

from .data_type import BaseType
from .data_type import DataType
from .numeric_util import is_integer
from .operation import get_slice_offsets
from .operation import Op


# Custom decorator function to allow formatting docstrings containing "{}"
def docstring_format_args(args):
    def docstring(func):
        func.__doc__ = func.__doc__.format(*args)
        return func

    return docstring


def warn_cpu(op, msg):
    print("Warning: {} {}, placing on CPU".format(op.type, msg))


class SupportedOperators:
    # Categorised lists of supported operators
    npu_pre_ops = set((Op.SplitSliceRead,))
    convolution_ops = set((Op.Conv2DBias, Op.Conv2D, Op.QuantizedConv2D,))
    depthwise_convolution_ops = set((Op.DepthwiseConv2DBias,))
    transpose_convolution_ops = set((Op.Conv2DBackpropInput,))
    convolution_like_ops = convolution_ops | depthwise_convolution_ops | transpose_convolution_ops
    max_pooling_ops = Op.op_set(Op.is_maxpool_op)
    avg_pooling_ops = Op.op_set(Op.is_avgpool_op)
    pooling_ops = set((Op.ReduceSum,)) | max_pooling_ops | avg_pooling_ops
    resizing_ops = set((Op.ResizeBilinear,))
    fc_vector_products = set((Op.QuantizedMatMul, Op.MatMul, Op.FullyConnected,))
    mac_main_ops = (
        # RNN/LSTM/GRU
        set((Op.BlockLSTM,))
        # conv/depthwiseconv/transposeconv
        | convolution_like_ops
        # pooling
        | pooling_ops
        # resizing/upscaling
        | resizing_ops
        # FC layers
        | fc_vector_products
    )
    unary_elem_wise_main_ops = Op.op_set(Op.is_unary_elementwise_op)
    binary_elem_wise_min_max_ops = set((Op.Minimum, Op.Maximum,))
    binary_elem_wise_shift_ops = set((Op.SHL, Op.SHR,))
    binary_elem_wise_add_mul_sub = set((Op.Add, Op.Mul, Op.Sub,))
    binary_elem_wise_main_ops = binary_elem_wise_min_max_ops | binary_elem_wise_add_mul_sub | binary_elem_wise_shift_ops
    elem_wise_main_ops = binary_elem_wise_main_ops | unary_elem_wise_main_ops
    supported_int32_tensor_ops = (
        set((Op.ReduceSum, Op.CLZ,)) | binary_elem_wise_add_mul_sub | binary_elem_wise_shift_ops
    )
    activation_ops = set((Op.Relu, Op.Relu6, Op.ReluN1To1, Op.Sigmoid, Op.Tanh, Op.Softmax,))
    npu_post_ops = (
        # activation functions
        activation_ops
        # concatenation write direction
        | set((Op.ConcatSliceWrite,))
        # Quantization
        | set((Op.Quantize,))
    )
    split_ops = set((Op.Split, Op.SplitV, Op.StridedSlice, Op.Slice, Op.UnpackReshaped, Op.Unpack,))
    concat_ops = set((Op.Concat, Op.ConcatTFLite, Op.PackReshaped, Op.Pack,))
    memory_only_ops = set((Op.Squeeze, Op.Reshape, Op.QuantizedReshape, Op.ExpandDims,)) | concat_ops | split_ops
    shapeless_input_ops = binary_elem_wise_main_ops | set((Op.Split, Op.SplitV,))
    supported_fused_activations = set((Op.Relu, Op.Relu6, Op.ReluN1To1, Op.Tanh, Op.Sigmoid, Op.LUT,))
    supported_operators = npu_pre_ops | mac_main_ops | elem_wise_main_ops | npu_post_ops | memory_only_ops
    # Supported data types
    supported_op_dtypes = set((DataType.uint8, DataType.int8, DataType.int16, DataType.int32))
    supported_bias_dtypes = set((DataType.int32, DataType.int64))
    # Defined ranges for allowed values:
    tens_dim_range = (1, 65535)
    stride_range = (1, 3)
    dilation_range = (1, 2)
    dilated_height_range = (1, 64)
    dilated_product_range = (1, 64 * 64)
    weights_limit = 127 * 65536

    def __init__(self):
        # Setup supported operator restriction checkers
        self.supported_operator_restrictions = {}
        self.supported_operator_restrictions.update(
            {op: self.check_depthwise_convolution_restrictions for op in SupportedOperators.depthwise_convolution_ops}
        )
        self.supported_operator_restrictions.update(
            {op: self.check_transpose_convolution_restrictions for op in SupportedOperators.transpose_convolution_ops}
        )
        self.supported_operator_restrictions.update(
            {op: self.check_pooling_restrictions for op in SupportedOperators.pooling_ops}
        )
        self.supported_operator_restrictions.update(
            {op: self.check_resize_restrictions for op in SupportedOperators.resizing_ops}
        )
        self.supported_operator_restrictions.update(
            {op: self.check_vector_product_restrictions for op in SupportedOperators.fc_vector_products}
        )
        self.supported_operator_restrictions.update(
            {op: self.check_element_wise_restrictions for op in SupportedOperators.elem_wise_main_ops}
        )
        self.supported_operator_restrictions.update(
            {op: self.check_memory_only_restrictions for op in SupportedOperators.memory_only_ops}
        )
        self.supported_operator_restrictions.update(
            {op: self.check_activation_ops for op in SupportedOperators.activation_ops}
        )
        # Setup the generic constraints. Note: the order matters
        self.generic_constraints = []
        self.generic_constraints.append(SupportedOperators.constraint_tens_defined_shape)
        self.generic_constraints.append(SupportedOperators.constraint_tens_output_shapeless)
        self.generic_constraints.append(SupportedOperators.constraint_tens_input_shapeless)
        self.generic_constraints.append(SupportedOperators.constraint_tens_shape_size)
        self.generic_constraints.append(SupportedOperators.constraint_tens_dtype)
        self.generic_constraints.append(SupportedOperators.constraint_tens_int32_ops)
        self.generic_constraints.append(SupportedOperators.constraint_tens_dimension)
        self.generic_constraints.append(SupportedOperators.constraint_tens_quant_none_check)
        self.generic_constraints.append(SupportedOperators.constraint_tens_quant_scale)
        self.generic_constraints.append(SupportedOperators.constraint_faf)
        # Setup specific constraints. The key in the dictionary must be a tuple of op types the constraints apply to
        self.specific_constraints = defaultdict(list)
        # Conv-like ops have the same checks applied to them:
        conv_like_ops = tuple(SupportedOperators.convolution_like_ops)
        self.specific_constraints[conv_like_ops].append(SupportedOperators.constraint_stride_type)
        self.specific_constraints[conv_like_ops].append(SupportedOperators.constraint_stride_range)
        self.specific_constraints[conv_like_ops].append(SupportedOperators.constraint_dilation_type)
        self.specific_constraints[conv_like_ops].append(SupportedOperators.constraint_dilation_range)
        self.specific_constraints[conv_like_ops].append(SupportedOperators.constraint_dilated_height_range)
        self.specific_constraints[conv_like_ops].append(SupportedOperators.constraint_dilated_product_range)
        self.specific_constraints[conv_like_ops].append(SupportedOperators.constraint_weights_type)
        self.specific_constraints[conv_like_ops].append(SupportedOperators.constraint_weights_nonconst)
        self.specific_constraints[conv_like_ops].append(SupportedOperators.constraint_weights_limit)
        self.specific_constraints[conv_like_ops].append(SupportedOperators.constraint_bias_type)
        self.specific_constraints[conv_like_ops].append(SupportedOperators.constraint_bias_40bit)
        self.specific_constraints[conv_like_ops].append(SupportedOperators.constraint_batch_size)

    def get_constraints_list(self, op_type):
        constraint_list = list(self.generic_constraints)
        for ops in self.specific_constraints:
            if op_type in ops:
                constraint_list.extend(self.specific_constraints[ops])
        return constraint_list

    def is_operator_supported(self, op):
        if op.type not in SupportedOperators.supported_operators:
            if op.type not in (Op.Placeholder, Op.SubgraphInput, Op.Const):
                print("Info: {} '{}' is not supported on the NPU. Placing on CPU instead".format(op.type, op.name))
            return False

        for constraint in self.get_constraints_list(op.type):
            valid, extra = constraint(op)
            if not valid:
                print("Warning: {} '{}' is not supported on the NPU. Placing on CPU instead".format(op.type, op.name))
                print(" - {}".format(constraint.__doc__))
                if extra:
                    print("   {}".format(extra))
                return False

        if op.type in self.supported_operator_restrictions:
            return self.supported_operator_restrictions[op.type](op)
        return True

    @staticmethod
    def constraint_tens_defined_shape(op):
        "Input(s) and Output Tensors must have a defined shape"
        valid = True
        extra = []
        tensors = [tens for tens in op.inputs + op.outputs if tens]
        for tens in tensors:
            if not tens.has_fully_defined_shape():
                valid = False
                extra.append("Tensor '{}' has shape: {}".format(tens.name, tens.shape))
        return valid, ", ".join(extra)

    @staticmethod
    def constraint_tens_output_shapeless(op):
        "Scalar or Broadcasting Tensors are only valid for Input Tensors"
        valid = True
        extra = []
        for tens in op.outputs:
            if tens.shape == []:
                valid = False
                extra.append("Output Tensor '{}' is shapeless".format(tens.name))
        return valid, ", ".join(extra)

    @classmethod
    @docstring_format_args([shapeless_input_ops])
    def constraint_tens_input_shapeless(cls, op):
        "Scalar or Broadcasting Input Tensors are only valid for op type: {}"
        valid = True
        extra = []
        tensors = [tens for tens in op.inputs if tens]
        for tens in tensors:
            if (tens.shape == []) and (op.type not in cls.shapeless_input_ops):
                valid = False
                extra.append(tens.name)
        extra = "Op has shapeless input tensor(s): {}".format(", ".join(extra))
        return valid, extra

    @staticmethod
    def constraint_tens_shape_size(op):
        "Input(s) and Output Tensors must not be greater than 4D"
        valid = True
        extra = []
        tensors = [tens for tens in op.inputs + op.outputs if tens]
        for tens in tensors:
            if len(tens.shape) > 4:
                valid = False
                extra.append("Tensor '{}' has shape: {}".format(tens.name, tens.shape))
        return valid, ", ".join(extra)

    @classmethod
    @docstring_format_args([supported_op_dtypes])
    def constraint_tens_dtype(cls, op):
        "Input(s), Output and Weight Tensors must be of type: {}"
        valid = True
        extra = []
        tensors = [tens for tens in op.get_ifm_ifm2_weights_ofm() if tens]
        tensors = tensors if tensors else op.inputs
        for tens in tensors:
            if tens.dtype not in cls.supported_op_dtypes:
                valid = False
                extra.append("Tensor '{}' has data type: {}".format(tens.name, tens.dtype))
        return valid, ", ".join(extra)

    @classmethod
    @docstring_format_args([supported_int32_tensor_ops])
    def constraint_tens_int32_ops(cls, op):
        "Tensors which are int32 are only valid when op type is: {}"
        valid = True
        extra = []
        tensors = [tens for tens in op.get_ifm_ifm2_weights_ofm() if tens]
        tensors = tensors if tensors else op.inputs
        for tens in tensors:
            if (tens.dtype == DataType.int32) and (op.type not in cls.supported_int32_tensor_ops):
                valid = False
                extra.append(tens.name)
        extra = "Op has int32 tensor(s): {}".format(", ".join(extra))
        return valid, extra

    @classmethod
    @docstring_format_args(tens_dim_range)
    def constraint_tens_dimension(cls, op):
        "Tensor dimensions must be in the range [{}, {}]"
        tens_min, tens_max = cls.tens_dim_range
        valid = True
        extra = []
        tensors = [tens for tens in op.get_ifm_ifm2_weights_ofm() if tens]
        tensors = tensors if tensors else op.inputs
        for tens in tensors:
            if not all(tens_min <= dim <= tens_max for dim in tens.shape):
                valid = False
                extra.append("Tensor '{}' has shape: {}".format(tens.name, tens.shape))
        return valid, ", ".join(extra)

    @staticmethod
    def constraint_tens_quant_none_check(op):
        "Tensors must have quantization parameters"
        valid = True
        extra = []
        tensors = [tens for tens in op.get_ifm_ifm2_weights_ofm() if tens]
        for tens in tensors:
            if tens.quantization is None:
                valid = False
                extra.append("Tensor '{}' has no quantization parameters".format(tens.name))
        return valid, ", ".join(extra)

    @staticmethod
    def constraint_tens_quant_scale(op):
        "Tensors with quantization scales must be finite"
        valid = True
        extra = []
        tensors = [tens for tens in op.get_ifm_ifm2_weights_ofm() if tens]
        for tens in tensors:
            if (tens.quantization.scale_f32 is not None) and np.isinf(tens.quantization.scale_f32).any():
                valid = False
                extra.append("Tensor '{}' has quantization scale: {}".format(tens.name, tens.quantization.scale_f32))
        return valid, ", ".join(extra)

    @classmethod
    @docstring_format_args([supported_fused_activations])
    def constraint_faf(cls, op):
        "The fused activation function (if present) must be one of type: {}"
        faf = op.activation
        valid = (faf is None) or (faf in cls.supported_fused_activations)
        extra = "Op has its fused activation function as: {}".format(faf)
        return valid, extra

    @staticmethod
    def constraint_stride_type(op):
        "Stride values for both width and height must be integer types"
        w = op.attrs["stride_w"]
        h = op.attrs["stride_h"]
        valid = is_integer(w) and is_integer(h)
        extra = "Op has stride WxH as: {}x{}".format(repr(w), repr(h))
        return valid, extra

    @classmethod
    @docstring_format_args(stride_range)
    def constraint_stride_range(cls, op):
        "Stride values for both width and height must be in the range [{}, {}]"
        w = op.attrs["stride_w"]
        h = op.attrs["stride_h"]
        stride_min, stride_max = cls.stride_range
        valid = (stride_min <= w <= stride_max) and (stride_min <= h <= stride_max)
        extra = "Op has stride WxH as: {}x{}".format(w, h)
        return valid, extra

    @staticmethod
    def constraint_dilation_type(op):
        "Dilation factor values for both width and height must be integer types"
        w = op.attrs.get("dilation_w_factor", 1)
        h = op.attrs.get("dilation_h_factor", 1)
        valid = is_integer(w) and is_integer(h)
        extra = "Op has dilation factor WxH as: {}x{}".format(repr(w), repr(h))
        return valid, extra

    @classmethod
    @docstring_format_args(dilation_range)
    def constraint_dilation_range(cls, op):
        "Dilation factor values for both width and height must be in the range [{}, {}]"
        w = op.attrs.get("dilation_w_factor", 1)
        h = op.attrs.get("dilation_h_factor", 1)
        dilation_min, dilation_max = cls.dilation_range
        valid = (dilation_min <= w <= dilation_max) and (dilation_min <= h <= dilation_max)
        extra = "Op has dilation factor WxH as: {}x{}".format(w, h)
        return valid, extra

    @classmethod
    @docstring_format_args(dilated_height_range)
    def constraint_dilated_height_range(cls, op):
        "Dilated kernel height must be in the range [{}, {}]"
        h = (op.weights.shape[0] - 1) * op.attrs.get("dilation_h_factor", 1) + 1
        dilated_height_min, dilated_height_max = cls.dilated_height_range
        valid = dilated_height_min <= h <= dilated_height_max
        extra = "Op has dilated kernel height as: {}".format(h)
        return valid, extra

    @classmethod
    @docstring_format_args(dilated_product_range)
    def constraint_dilated_product_range(cls, op):
        "Product of dilated kernel width and height must be in the range [{}, {}]"
        weights = op.weights
        w = (weights.shape[1] - 1) * op.attrs.get("dilation_w_factor", 1) + 1
        h = (weights.shape[0] - 1) * op.attrs.get("dilation_h_factor", 1) + 1
        product = w * h
        dilated_product_min, dilated_product_max = cls.dilated_product_range
        valid = dilated_product_min <= product <= dilated_product_max
        extra = "Op has product of dilated kernel width and height as: {}".format(product)
        return valid, extra

    @staticmethod
    def constraint_weights_type(op):
        "Weight Tensor must be 8-bit"
        weights = op.weights
        valid = weights.element_size() == 1
        extra = "Tensor '{}' is {}-bit".format(weights.name, int(weights.element_size() * 8))
        return valid, extra

    @staticmethod
    def constraint_weights_nonconst(op):
        "Weight tensor cannot be non-constant"
        weights = op.weights
        valid = weights.values is not None
        extra = "Tensor '{}' has non-constant values".format(weights.name)
        return valid, extra

    @classmethod
    @docstring_format_args([weights_limit])
    def constraint_weights_limit(cls, op):
        "The sum of the weights cannot exceed {}"
        weights = op.weights
        values = weights.quant_values.astype(np.int64) - weights.quantization.zero_point
        limit = np.amax(np.sum(np.absolute(values), axis=(0, 1, 2)))
        valid = limit <= cls.weights_limit
        extra = "Tensor '{}' has the sum of weights: {}".format(weights.name, limit)
        return valid, extra

    @classmethod
    @docstring_format_args([supported_bias_dtypes])
    def constraint_bias_type(cls, op):
        "Optional Bias Tensor must be of type: {}"
        valid = True
        extra = ""
        bias = op.bias
        if bias:
            valid = bias.dtype in cls.supported_bias_dtypes
            extra = "Tensor '{}' has data type: {}".format(bias.name, bias.dtype)
        return valid, extra

    @staticmethod
    def constraint_bias_40bit(op):
        "Optional Bias Tensor values must fit within 40-bits"
        valid = True
        extra = ""
        bias = op.bias
        if bias and bias.dtype == DataType.int64:
            valid = all(len(bin(quant_value)[2:]) <= 40 for quant_value in bias.quant_values)
            extra = "Tensor '{}' has values larger than 40-bits".format(bias.name)
        return valid, extra

    @staticmethod
    def constraint_batch_size(op):
        "IFM Tensor batch size must be 1"
        ifm = op.ifm
        valid = ifm.shape[0] == 1
        extra = "Tensor '{}' has batch size: {}".format(ifm.name, ifm.shape[0])
        return valid, extra

    @classmethod
    def check_depthwise_convolution_restrictions(cls, op):
        # check depth
        ifm_tensor, ofm_tensor = op.get_ifm_ofm()
        if op.attrs["depth_multiplier"] > 1 and not (
            (ifm_tensor.shape[3] == 1) and (ofm_tensor.shape[3] == op.attrs["depth_multiplier"])
        ):
            print(
                "Warning: for depth multipliers > 1,",
                "number of input channels must be 1 and number of output channels must be equal to depth multiplier.",
                "Placing on CPU",
            )
            return False
        return True

    @classmethod
    def check_transpose_convolution_restrictions(cls, op):
        # check stride
        stride_h, stride_w = op.attrs["stride_h"], op.attrs["stride_w"]
        if stride_h != 2 or stride_w != 2:
            print("Warning: stride must be equal to 2, placing on CPU")
            return False

        # check output dimensions
        ifm_tensor, weight_tensor, _, ofm_tensor = op.get_ifm_weights_biases_ofm()
        ifm_h, ifm_w = ifm_tensor.shape[1], ifm_tensor.shape[2]
        ofm_h, ofm_w = ofm_tensor.shape[1], ofm_tensor.shape[2]
        if op.attrs["padding"] == b"SAME":
            if (ofm_h != ifm_h * stride_h) or (ofm_w != ifm_w * stride_w):
                print(
                    "Warning: for",
                    op.type,
                    "using SAME padding, output dimensions must equal input dimensions multiplied by stride.",
                    "Placing on CPU",
                )
                return False
        elif op.attrs["padding"] == b"VALID":
            kernel_h, kernel_w = weight_tensor.shape[0], weight_tensor.shape[1]
            if (ofm_h != (ifm_h) * stride_h + max(kernel_h - stride_h, 0)) or (
                ofm_w != (ifm_w) * stride_w + max(kernel_w - stride_w, 0)
            ):
                print(
                    "Warning: for",
                    op.type,
                    "using VALID padding, output dimensions must equal input dimensions multiplied by stride,",
                    "minus difference between kernel size and stride. Placing on CPU",
                )
                return False
        return True

    @classmethod
    def check_pooling_restrictions(cls, op):
        # check stride
        stride_w, stride_h = op.attrs["stride_w"], op.attrs["stride_h"]
        if not is_integer(stride_w) or not is_integer(stride_h):
            print("Warning:", op.type, "has non-integer stride, placing on CPU")
            return False
        if not 1 <= stride_w <= 3 or not 1 <= stride_h <= 3:
            print(
                "Warning: {} has stride ({}, {}), only strides in range [1, 3] are allowed. Placing on CPU".format(
                    op.type, stride_w, stride_h
                )
            )
            return False

        # check data type
        ifm_tensor, ofm_tensor = op.get_ifm_ofm()
        if ifm_tensor.dtype != ofm_tensor.dtype:
            if op.type != Op.ReduceSum:
                print("Warning: input data type doesn't match output data type, placing on CPU")
                return False
            # TODO: else check ReduceSum restrictions.

        # check batch size
        if ifm_tensor.shape[0] != 1:
            print("Warning: input batch size must be 1, placing on CPU")
            return False

        # check kernel size
        kernel_w, kernel_h = op.attrs["filter_width"], op.attrs["filter_height"]
        if op.type in cls.avg_pooling_ops and op.attrs["padding"] == b"SAME":
            if not 1 <= kernel_w <= 8 or not 1 <= kernel_h <= 8:
                print(
                    "Warning:",
                    op.type,
                    "has kernel size ({}, {}), only kernel sizes in range [1, 8] are allowed. Placing on CPU".format(
                        kernel_w, kernel_h
                    ),
                )
                return False
        if op.type in cls.avg_pooling_ops and op.attrs["padding"] == b"VALID" or op.type in cls.max_pooling_ops:
            if not 1 <= kernel_w * kernel_h <= 256 * 256:
                print(
                    "Warning: product of kernel width and height must be >= 1 and not exceed 256 * 256 ({}),".format(
                        256 * 256
                    ),
                    "placing on CPU",
                )
                return False
            if not 1 <= kernel_h <= 256:
                print("Warning:", op.type, "has kernel height outside of range [1, 256], placing on CPU")
                return False

        return True

    @classmethod
    def check_resize_restrictions(cls, op):
        # check unsupported upscaling factor
        if op.type == Op.ResizeBilinear:
            if op.inputs[0].shape[1] == 1 and op.inputs[0].shape[2] == 1:
                return True
            if op.inputs[0].shape == op.outputs[0].shape:
                return True
            upscaled_shape = np.array(op.inputs[0].shape[1:3])
            out_shape = np.array(op.outputs[0].shape[1:3])
            while (upscaled_shape < out_shape).all():
                upscaled_shape *= 2
                if op.attrs["align_corners"]:
                    upscaled_shape -= 1
                if np.array_equal(out_shape, upscaled_shape):
                    return True
        return False

    @classmethod
    def check_vector_product_restrictions(cls, op):
        # check data type
        ifm_tensor, _, weight_tensor, bias_tensor, _ = op.get_ifm_ifm2_weights_biases_ofm()
        if weight_tensor.element_size() > 1:
            print("Warning: only 8-bit datatypes supported for {}, placing on CPU".format(op.type))
            return False

        if not cls.check_bias_restrictions(bias_tensor):
            return False

        # check non const weights
        if weight_tensor.values is None:
            print("Warning:", op.type, "has non-const weights, placing on CPU")
            return False

        return True

    @classmethod
    def check_element_wise_restrictions(cls, op):
        # check data type
        ifm_tensor, ifm2_tensor, _, ofm_tensor = op.get_ifm_ifm2_weights_ofm()
        # input and output datatype must match for these operators
        if (
            op.type in cls.binary_elem_wise_min_max_ops | cls.unary_elem_wise_main_ops
            and ifm_tensor.dtype != ofm_tensor.dtype
        ):
            print("Warning:", op.type, "must have same input and output datatype, placing on CPU")
            return False
        if op.type in cls.binary_elem_wise_add_mul_sub:
            # both inputs must have same type
            if ifm_tensor.dtype != ifm2_tensor.dtype:
                print("Warning:", op.type, "must have same datatype on both inputs, placing on CPU")
                return False
            # signed input check
            if ifm_tensor.dtype.type & BaseType.Signed:
                # output must be signed
                if ofm_tensor.dtype.type & BaseType.Unsigned:
                    print("Warning: only signed output types supported for {}, placing on CPU".format(op.type))
                    return False
                # and 8, 16 or 32-bit
                bit_lengths = {8, 16, 32}
                if ofm_tensor.element_size() * 8 not in bit_lengths:
                    print(
                        "Warning:", op.type, "is only supported for bit lengths {}, placing on CPU".format(bit_lengths)
                    )
                    return False
            # unsigned input check, output must be same type or int32
            if ifm_tensor.dtype.type & BaseType.Unsigned and not (
                ifm_tensor.dtype == ofm_tensor.dtype or ofm_tensor.dtype == DataType.int32
            ):
                print("Warning:", op.type, "has unsigned input but output is not unsigned or int32, placing on CPU")
                return False
        elif op.type in cls.binary_elem_wise_shift_ops:
            if ifm_tensor.dtype != DataType.int32 or ifm2_tensor.dtype != DataType.int32:
                print("Warning:", op.type, "input datatypes are not int32, placing on CPU")
                return False
            if op.type in (Op.CLZ, Op.SHL) and ofm_tensor.dtype != DataType.int32:
                print("Warning:", op.type, "output datatype is not int32, placing on CPU")
                return False

        # check batch size
        if len(ifm_tensor.shape) > 2 and ifm_tensor.shape[0] != 1:
            print(
                "Warning:",
                op.type,
                "only supports batch size 1 for tensors with more than 2 dimensions, placing on CPU",
            )
            return False
        if op.type in cls.binary_elem_wise_main_ops:  # if op type is unary, ifm2_tensor is None
            if len(ifm2_tensor.shape) > 2 and ifm2_tensor.shape[0] != 1:
                print(
                    "Warning:",
                    op.type,
                    "only supports batch size 1 for tensors with more than 2 dimensions, placing on CPU",
                )
                return False

        # negative alpha values are not supported
        if op.type == Op.LeakyRelu and op.attrs["alpha"] < 0:
            print("Warning:", op.type, "has negative alpha, placing on CPU")
            return False

        # check if ifm or ifm2 has ofm shape
        if ifm_tensor.shape != ofm_tensor.shape and ifm2_tensor.shape != ofm_tensor.shape:
            print("Warning:", op.type, "input shape(s) differ from output shape, placing on CPU")
            return False

        if op.type in cls.binary_elem_wise_min_max_ops and not cls.check_quantization_restrictions_binary_elem_wise(op):
            return False

        return True

    @classmethod
    def check_memory_only_restrictions(cls, op):
        if op.type == Op.StridedSlice:
            if len(op.inputs) != 4:
                warn_cpu(op, "has {} input tensors, only 4 inputs are supported".format(len(op.inputs)))
                return False
            input_tens, begin_tens, end_tens, strides_tens = op.inputs
            if begin_tens.values is None or end_tens.values is None or strides_tens.values is None:
                warn_cpu(op, "has a non-constant begin, end, or stride input tensor, which is not supported")
                return False
            if not (
                len(input_tens.shape)
                == len(op.outputs[0].shape)
                == len(begin_tens.values)
                == len(end_tens.values)
                == len(strides_tens.values)
            ):
                warn_cpu(op, "has input tensors with shapes that are not supported")
                return False
            # check stride size
            if any(stride != 1 for stride in strides_tens.values):
                warn_cpu(op, "has stride values {}, only stride 1 values are supported".format(strides_tens.values))
                return False
            # check ellipsis_mask
            if op.attrs["ellipsis_mask"] != 0:
                warn_cpu(op, "ellipsis_mask is {}, only 0 is supported".format(op.attrs["ellipsis_mask"]))
                return False
            # check if both new_axis_mask and shrink_axis_mask have bit set
            if op.attrs["new_axis_mask"] != 0 and op.attrs["shrink_axis_mask"] != 0:
                warn_cpu(op, "new_axis_mask and shrink_axis_mask are both non-zero, which is not supported")
                return False
            # Calculate offset start/end
            offset_start = get_slice_offsets(input_tens.shape, begin_tens, op.attrs["begin_mask"], is_begin=True)
            offset_end = get_slice_offsets(input_tens.shape, end_tens, op.attrs["end_mask"], is_begin=False)
            # check "end - begin" doesn't result in any zero or negative elements
            if any((end - begin) <= 0 for begin, end in zip(offset_start, offset_end)):
                warn_cpu(
                    op,
                    "has slice begin values {}, some of which are >= end values {}, which is illegal".format(
                        begin_tens.values, end_tens.values
                    ),
                )
                return False
        if op.type == Op.SplitV:
            # check that maximum one size is set to -1, indicating that size should be inferred
            sizes = op.inputs[1].values
            num_to_be_inferred = 0
            for size in sizes:
                if size == -1:
                    num_to_be_inferred += 1

            if num_to_be_inferred > 1:
                print("Warning:", op.type, "has more than one size to be inferred, which is illegal, placing on CPU")
                return False
        if op.type in set((Op.Concat, Op.ConcatTFLite,)):
            axis = op.attrs.get("axis", None)
            if axis is None:
                print("Warning:", op.type, "invalid or missing axis, placing on CPU")
                return False
            if axis < 0:
                axis += len(op.inputs[0].shape)
            if not 0 <= axis < len(op.inputs[0].shape):
                print("Warning:", op.type, "invalid axis", axis, ", placing on CPU")
                return False
            ofm = op.outputs[0]
            ofm_dims = len(ofm.shape)
            for ifm in op.inputs:
                if len(ifm.shape) != ofm_dims:
                    return False
                for i in range(ofm_dims):
                    if i != axis and ifm.shape[i] != ofm.shape[i]:
                        print(
                            "Warning:",
                            op.type,
                            "invalid ifm:",
                            ifm.name,
                            ifm.shape,
                            "mismatch in dimension",
                            i,
                            ", placing on CPU",
                        )
                        return False

        return True

    @classmethod
    def check_quantization_restrictions_binary_elem_wise(cls, op):
        # makes sure IFM1, IFM2 and OFM quantization are equal for binary ops
        assert len(op.inputs) >= 2 and len(op.outputs) == 1

        if (
            op.inputs[0].quantization is None
            or not op.inputs[0].is_scaling_equal(op.inputs[1])
            or not op.inputs[0].is_scaling_equal(op.outputs[0])
        ):
            print(
                "Warning: Input/output tensors with different quantization is unsupported for the", op.type, "operator"
            )
            return False

        return True

    @classmethod
    def check_activation_ops(cls, op):
        if op.type == Op.Softmax:
            ifm_tensor = op.inputs[0]
            ofm_tensor = op.outputs[0]

            # check data type
            if ifm_tensor.dtype != ofm_tensor.dtype:
                print("Warning:", op.type, "input type differs from output type, placing on CPU")
                return False

            if ifm_tensor.dtype not in (DataType.uint8, DataType.int8, DataType.int16):
                print(
                    "Warning: only datatypes supported for {} are uint8, int8 and int16; placing on CPU".format(op.type)
                )
                return False

            # check shape
            if ifm_tensor.shape != ofm_tensor.shape:
                print("Warning:", op.type, "input shape differs from output shape, placing on CPU")
                return False

        return True

    @classmethod
    def check_bias_restrictions(cls, bias_tensor):
        # check data type
        if bias_tensor is not None and bias_tensor.dtype not in (DataType.int32, DataType.int64):
            print("Warning: bias tensor datatype must be int32 or int64, placing on CPU")
            return False

        # check if values fits in 40-bit
        if bias_tensor is not None and bias_tensor.dtype == DataType.int64:
            for quant_value in bias_tensor.quant_values:
                if not (-(1 << 39) <= quant_value < (1 << 39)):
                    print("Warning: bias tensor values are larger than 40 bits, placing on CPU")
                    return False

        return True
