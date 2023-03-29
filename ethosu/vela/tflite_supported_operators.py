# SPDX-FileCopyrightText: Copyright 2020-2023 Arm Limited and/or its affiliates <open-source-office@arm.com>
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
# The TFLiteSupportedOperators class which is a collection of all TFLite supported operators and parameter checks.
from collections import defaultdict

import numpy as np

from .data_type import DataType
from .numeric_util import full_shape
from .operation import Op
from .operation import Padding
from .supported_operators_util import docstring_format_args
from .supported_operators_util import list_formatter
from .tensor import check_quantized_tens_scaling_equal
from .tflite_mapping import BUILTIN_OPERATOR_UNKNOWN
from .tflite_mapping import optype_to_builtintype
from .utils import calc_resize_factor


def _optype_formatter(op_list):
    # Convert internal op types to external names
    output = map(optype_to_builtintype, op_list)
    # Remove UNKNOWNs
    output = (x for x in output if x is not BUILTIN_OPERATOR_UNKNOWN)
    return list_formatter(output)


class TFLiteSupportedOperators:
    # Categorised lists of supported operators
    npu_pre_ops = set(
        (
            Op.SplitSliceRead,
            Op.Shape,
        )
    )
    convolution_ops = set(
        (
            Op.Conv2DBias,
            Op.Conv2D,
            Op.QuantizedConv2D,
        )
    )
    depthwise_convolution_ops = set((Op.DepthwiseConv2DBias,))
    transpose_convolution_ops = set((Op.Conv2DBackpropInput,))
    convolution_like_ops = convolution_ops | depthwise_convolution_ops | transpose_convolution_ops
    max_pooling_ops = Op.op_set(Op.is_maxpool_op)
    avg_pooling_ops = Op.op_set(Op.is_avgpool_op)
    pooling_ops = set((Op.ReduceSum,)) | max_pooling_ops | avg_pooling_ops
    resizing_ops = Op.op_set(Op.is_resize_op)
    fc_vector_products = set(
        (
            Op.QuantizedMatMul,
            Op.MatMul,
            Op.FullyConnected,
        )
    )
    mac_main_ops = (
        # LSTM
        set((Op.UnidirectionalSequenceLstm,))
        # conv/depthwiseconv/transposeconv
        | convolution_like_ops
        # pooling
        | pooling_ops
        # resizing/upscaling
        | resizing_ops
        # FC layers
        | fc_vector_products
        # Mean (converts to depthwise conv)
        | set((Op.Mean,))
        # ArgMax (converts to depthwise conv and maxpool)
        | set((Op.ArgMax,))
    )
    unary_elem_wise_main_ops = Op.op_set(Op.is_unary_elementwise_op)
    binary_elem_wise_min_max_ops = set(
        (
            Op.Minimum,
            Op.Maximum,
        )
    )
    binary_elem_wise_shift_ops = set(
        (
            Op.SHL,
            Op.SHR,
        )
    )
    binary_elem_wise_add_mul_sub = set(
        (
            Op.Add,
            Op.Mul,
            Op.Sub,
        )
    )
    binary_elem_wise_main_ops = binary_elem_wise_min_max_ops | binary_elem_wise_add_mul_sub | binary_elem_wise_shift_ops
    elem_wise_main_ops = binary_elem_wise_main_ops | unary_elem_wise_main_ops | set((Op.SquaredDifference,))
    pad_ops = set((Op.Pad,))
    supported_int32_tensor_ops = (
        set((Op.ReduceSum, Op.CLZ, Op.Shape, Op.ArgMax, Op.Transpose))
        | binary_elem_wise_add_mul_sub
        | binary_elem_wise_shift_ops
    )

    relu_ops = set(
        (
            Op.Relu,
            Op.Relu6,
            Op.ReluN1To1,
            Op.Clip,
        )
    )
    activation_ops = relu_ops | set(
        (
            Op.Tanh,
            Op.Sigmoid,
            Op.Softmax,
            Op.HardSwish,
            Op.LeakyRelu,
            Op.Prelu,
        )
    )
    npu_post_ops = (
        # activation functions
        activation_ops
        # concatenation write direction
        | set((Op.ConcatSliceWrite,))
        # Quantization
        | set((Op.Quantize,))
    )
    split_ops = set(
        (
            Op.Split,
            Op.SplitV,
            Op.StridedSlice,
            Op.Slice,
            Op.UnpackReshaped,
            Op.Unpack,
        )
    )
    concat_ops = set(
        (
            Op.Concat,
            Op.ConcatTFLite,
            Op.PackReshaped,
            Op.Pack,
        )
    )
    memory_only_ops = (
        set(
            (
                Op.Reshape,
                Op.QuantizedReshape,
                Op.Squeeze,
                Op.ExpandDims,
                Op.Transpose,
            )
        )
        | concat_ops
        | split_ops
    )
    per_axis_quant_ops = convolution_like_ops  # per-axis/channel quantization only currently supported for conv ops
    supported_fused_activations = relu_ops | set(
        (
            Op.Tanh,
            Op.Sigmoid,
            Op.LUT,
        )
    )
    supported_operators = npu_pre_ops | mac_main_ops | elem_wise_main_ops | pad_ops | npu_post_ops | memory_only_ops
    # Supported data types
    supported_op_dtypes = set((DataType.uint8, DataType.int8, DataType.int16, DataType.int32))
    supported_faf_dtypes = set((DataType.uint8, DataType.int8, DataType.int16))
    supported_bias_dtypes = set((DataType.int32, DataType.int64))
    supported_pad_dtypes = set((DataType.int32, DataType.int64))
    # Defined ranges for allowed values:
    tens_dim_range = (1, 65535)
    stride_range = (1, 3)
    dilated_height_range = (1, 64)
    dilated_product_range = (1, 64 * 64)
    weights_limit = 127 * 65536
    filter_range = (1, 8)
    filter_height_range = (1, 256)
    filter_product_range = (1, 256 * 256)
    mean_reduced_axis_max_size = 64 * 64
    mean_kernel_product_int8 = 2 ** (24)
    mean_kernel_product_uint8 = 2 ** (23)
    mean_kernel_product_int16 = 2 ** (16)

    def __init__(self):
        # Setup the generic constraints. Note: the order matters
        self.generic_constraints = []
        self.generic_constraints.append(TFLiteSupportedOperators.constraint_tens_dtype)
        self.generic_constraints.append(TFLiteSupportedOperators.constraint_tens_int32_ops)
        self.generic_constraints.append(TFLiteSupportedOperators.constraint_tens_dimension)
        self.generic_constraints.append(TFLiteSupportedOperators.constraint_tens_quant_per_axis)
        self.generic_constraints.append(TFLiteSupportedOperators.constraint_batch_size)
        self.generic_constraints.append(TFLiteSupportedOperators.constraint_faf)
        self.generic_constraints.append(TFLiteSupportedOperators.constraint_faf_type)

        # Setup generic constraint exceptions
        self.generic_constraints_exceptions = defaultdict(list)
        self.generic_constraints_exceptions[Op.ArgMax].append(TFLiteSupportedOperators.constraint_tens_dtype)
        self.generic_constraints_exceptions[Op.FullyConnected].append(TFLiteSupportedOperators.constraint_batch_size)
        self.generic_constraints_exceptions[Op.Softmax].append(TFLiteSupportedOperators.constraint_batch_size)
        self.generic_constraints_exceptions[Op.Reshape].append(TFLiteSupportedOperators.constraint_batch_size)
        self.generic_constraints_exceptions[Op.Shape].append(TFLiteSupportedOperators.constraint_batch_size)
        self.generic_constraints_exceptions[Op.Squeeze].append(TFLiteSupportedOperators.constraint_batch_size)
        for op_type in TFLiteSupportedOperators.split_ops - set((Op.UnpackReshaped,)):
            self.generic_constraints_exceptions[op_type].append(TFLiteSupportedOperators.constraint_batch_size)

        # Setup specific constraints. Note: the order matters
        self.specific_constraints = defaultdict(list)

        # Conv specific ops:
        for op_type in TFLiteSupportedOperators.convolution_ops:
            self.specific_constraints[op_type].append(TFLiteSupportedOperators.constraint_stride_width_no_upper_limit)

        # Conv-like checks:
        for op_type in TFLiteSupportedOperators.convolution_like_ops:
            self.specific_constraints[op_type].append(TFLiteSupportedOperators.constraint_dilated_height_range)
            self.specific_constraints[op_type].append(TFLiteSupportedOperators.constraint_dilated_product_range)
            self.specific_constraints[op_type].append(TFLiteSupportedOperators.constraint_weights_type)
            self.specific_constraints[op_type].append(TFLiteSupportedOperators.constraint_weights_const)
            self.specific_constraints[op_type].append(TFLiteSupportedOperators.constraint_weights_limit)
            self.specific_constraints[op_type].append(TFLiteSupportedOperators.constraint_bias_shape)
            self.specific_constraints[op_type].append(TFLiteSupportedOperators.constraint_bias_type)
            self.specific_constraints[op_type].append(TFLiteSupportedOperators.constraint_bias_40bit)

        # Transpose Conv specific checks:
        for op_type in TFLiteSupportedOperators.transpose_convolution_ops:
            self.specific_constraints[op_type].append(TFLiteSupportedOperators.constraint_tconv_stride)
            self.specific_constraints[op_type].append(TFLiteSupportedOperators.constraint_tconv_same)
            self.specific_constraints[op_type].append(TFLiteSupportedOperators.constraint_tconv_valid)
        # Depthwise Conv specific checks:
        for op_type in TFLiteSupportedOperators.depthwise_convolution_ops:
            self.specific_constraints[op_type].append(TFLiteSupportedOperators.constraint_depthwise_conv_stride)
            self.specific_constraints[op_type].append(TFLiteSupportedOperators.constraint_depth_multiplier)

        # Pooling checks:
        for op_type in TFLiteSupportedOperators.pooling_ops - TFLiteSupportedOperators.avg_pooling_ops:
            self.specific_constraints[op_type].append(TFLiteSupportedOperators.constraint_stride_range)
        # AVG pooling specific checks:
        for op_type in TFLiteSupportedOperators.avg_pooling_ops:
            self.specific_constraints[op_type].append(TFLiteSupportedOperators.constraint_stride_width_no_upper_limit)
            self.specific_constraints[op_type].append(TFLiteSupportedOperators.constraint_stride_range_no_padding)
            self.specific_constraints[op_type].append(TFLiteSupportedOperators.constraint_filter_range)
            self.specific_constraints[op_type].append(TFLiteSupportedOperators.constraint_filter_height_range_valid_pad)
            self.specific_constraints[op_type].append(
                TFLiteSupportedOperators.constraint_filter_product_range_valid_pad
            )
        # MAX pooling specific checks:
        for op_type in TFLiteSupportedOperators.max_pooling_ops:
            self.specific_constraints[op_type].append(TFLiteSupportedOperators.constraint_filter_height_range)
            self.specific_constraints[op_type].append(TFLiteSupportedOperators.constraint_filter_product_range)

        # Resizing specific checks:
        for op_type in TFLiteSupportedOperators.resizing_ops:
            self.specific_constraints[op_type].append(TFLiteSupportedOperators.constraint_resize)
            self.specific_constraints[op_type].append(TFLiteSupportedOperators.constraint_resize_size)
            self.specific_constraints[op_type].append(TFLiteSupportedOperators.constraint_resize_attrs)

        # Resize Bilinear specific checks:
        self.specific_constraints[Op.ResizeBilinear].append(
            TFLiteSupportedOperators.constraint_resizebi_half_pixel_centers_dims
        )

        # Vector Product specific checks:
        for op_type in TFLiteSupportedOperators.fc_vector_products:
            self.specific_constraints[op_type].append(TFLiteSupportedOperators.constraint_weights_type)
            self.specific_constraints[op_type].append(TFLiteSupportedOperators.constraint_weights_const)
            self.specific_constraints[op_type].append(TFLiteSupportedOperators.constraint_bias_shape)
            self.specific_constraints[op_type].append(TFLiteSupportedOperators.constraint_bias_type)
            self.specific_constraints[op_type].append(TFLiteSupportedOperators.constraint_bias_40bit)

        # Element-wise checks
        # Binary Min/Max specific checks:
        for op_type in TFLiteSupportedOperators.binary_elem_wise_min_max_ops:
            self.specific_constraints[op_type].append(
                TFLiteSupportedOperators.constraint_matching_quantization_parameters
            )
            self.specific_constraints[op_type].append(TFLiteSupportedOperators.constraint_broadcast_shapes)
        # Binary Add/Mul/Sub specific checks:
        for op_type in TFLiteSupportedOperators.binary_elem_wise_add_mul_sub:
            self.specific_constraints[op_type].append(TFLiteSupportedOperators.constraint_broadcast_shapes)
        # Binary Shift specific checks:
        for op_type in TFLiteSupportedOperators.binary_elem_wise_shift_ops:
            self.specific_constraints[op_type].append(TFLiteSupportedOperators.constraint_inputs_int32)
            self.specific_constraints[op_type].append(TFLiteSupportedOperators.constraint_broadcast_shapes)

        # SHL specific checks:
        self.specific_constraints[Op.SHL].append(TFLiteSupportedOperators.constraint_output_int32)

        # CLZ specific checks:
        self.specific_constraints[Op.CLZ].append(TFLiteSupportedOperators.constraint_output_int32)

        # StridedSlice specific checks:
        self.specific_constraints[Op.StridedSlice].append(
            TFLiteSupportedOperators.constraint_stridedslice_stride_values
        )
        self.specific_constraints[Op.StridedSlice].append(TFLiteSupportedOperators.constraint_stridedslice_offset_false)

        # Pad specific checks:
        self.specific_constraints[Op.Pad].append(TFLiteSupportedOperators.constraint_pad_shape)
        #self.specific_constraints[Op.Pad].append(TFLiteSupportedOperators.constraint_padding_dimensions)
        self.specific_constraints[Op.Pad].append(TFLiteSupportedOperators.constraint_pad_type)

        # Mean specific checks:
        self.specific_constraints[Op.Mean].append(TFLiteSupportedOperators.constraint_mean_height_width_product)
        self.specific_constraints[Op.Mean].append(TFLiteSupportedOperators.constraint_mean_width)
        self.specific_constraints[Op.Mean].append(TFLiteSupportedOperators.constraint_mean_depth)

        # Reshape specific checks:
        self.specific_constraints[Op.Reshape].append(TFLiteSupportedOperators.constraint_reshape_shape_constant)

        # ArgMax specific checks:
        self.specific_constraints[Op.ArgMax].append(TFLiteSupportedOperators.constraint_argmax_axis)
        self.specific_constraints[Op.ArgMax].append(TFLiteSupportedOperators.constraint_argmax_depth)

        # UnidirectionalSequenceLstm specific checks:
        op_type = Op.UnidirectionalSequenceLstm
        self.specific_constraints[op_type].append(TFLiteSupportedOperators.constraint_lstm_no_cifg)
        self.specific_constraints[op_type].append(TFLiteSupportedOperators.constraint_lstm_no_peep_hole)
        self.specific_constraints[op_type].append(TFLiteSupportedOperators.constraint_lstm_no_projection)
        self.specific_constraints[op_type].append(TFLiteSupportedOperators.constraint_lstm_no_normalisation)
        self.specific_constraints[op_type].append(TFLiteSupportedOperators.constraint_lstm_weights)
        self.specific_constraints[op_type].append(TFLiteSupportedOperators.constraint_lstm_weight_dimensions)

        # Rsqrt specific checks
        self.specific_constraints[Op.Rsqrt].append(TFLiteSupportedOperators.constraint_rsqrt_input_int8)

        # Slice specific checks:
        self.specific_constraints[Op.Slice].append(TFLiteSupportedOperators.constraint_slice_inputs_const)

        # Transpose specific checks:
        self.specific_constraints[Op.Transpose].append(TFLiteSupportedOperators.constraint_transpose)

    def is_operator_supported(self, op):
        ext_type = optype_to_builtintype(op.type)
        if op.type not in TFLiteSupportedOperators.supported_operators:
            if op.type not in (Op.Placeholder, Op.SubgraphInput, Op.Const):
                print(f"Info: {ext_type} '{op.name}' is a CPU only op")
            return False

        op_exceptions = self.generic_constraints_exceptions[op.type]
        generic_constraints = [constraint for constraint in self.generic_constraints if constraint not in op_exceptions]

        for constraint in generic_constraints + self.specific_constraints[op.type]:
            valid, extra = constraint(op)
            if not valid:
                print(f"Warning: {ext_type} '{op.name}' is not supported on the NPU. Placing on CPU instead")
                print(f" - {constraint.__doc__}")
                if extra:
                    print(f"   {extra}")
                return False

        return True

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

    @classmethod
    @docstring_format_args([_optype_formatter(supported_int32_tensor_ops)])
    def constraint_tens_int32_ops(cls, op):
        "Tensors which are int32 are only valid when op type is: {}"
        valid = True
        extra = []
        tensors = [tens for tens in op.get_ifm_ifm2_weights_ofm() if tens]
        if not tensors:
            tensors = [tens for tens in op.inputs if tens]
        for tens in tensors:
            if (tens.dtype == DataType.int32) and (op.type not in cls.supported_int32_tensor_ops):
                valid = False
                extra.append(tens.name)
        extra = ", ".join(extra)
        return valid, f"Op has int32 tensor(s): {extra}"

    @classmethod
    @docstring_format_args(tens_dim_range)
    def constraint_tens_dimension(cls, op):
        "Tensor dimensions must be in the range [{}, {}]"
        tens_min, tens_max = cls.tens_dim_range
        valid = True
        extra = []
        tensors = [tens for tens in op.get_ifm_ifm2_weights_ofm() if tens]
        if not tensors:
            tensors = [tens for tens in op.inputs if tens]
        for tens in tensors:
            if not all(tens_min <= dim <= tens_max for dim in tens.shape):
                valid = False
                extra.append(f"Tensor '{tens.name}' has shape: {tens.shape}")
        return valid, ", ".join(extra)

    @classmethod
    @docstring_format_args([_optype_formatter(per_axis_quant_ops)])
    def constraint_tens_quant_per_axis(cls, op):
        "Per-axis quantization is only supported for the following op types: {}"
        valid = True
        extra = []
        if op.type not in cls.per_axis_quant_ops:
            tensors = [tens for tens in op.get_ifm_ifm2_weights_ofm() if tens]
            for tens in tensors:
                if tens.quantization and tens.quantization.is_per_axis():
                    valid = False
                    extra.append(tens.name)
        return valid, "The following tensor(s) have per-axis quantization parameters: " + ", ".join(extra)

    @classmethod
    @docstring_format_args([_optype_formatter(supported_fused_activations)])
    def constraint_faf(cls, op):
        "The fused activation function (if present) must be one of type: {}"
        if op.activation is None:
            res = True, "Op has no fused activation function"
        else:
            faf = op.activation.op_type
            valid = faf in cls.supported_fused_activations
            res = valid, f"Op has its fused activation function as: {faf}"
        return res

    @classmethod
    @docstring_format_args([list_formatter(supported_faf_dtypes)])
    def constraint_faf_type(cls, op):
        "If a fused activation function is present, the Output tensor must be one of type: {}"
        if op.activation is None:
            res = True, "Op has no fused activation function"
        else:
            valid = op.ofm.dtype in cls.supported_faf_dtypes
            ext_type = optype_to_builtintype(op.activation.op_type)
            res = valid, f"Op has fused activation function {ext_type}, and Output tensor data type: {op.ofm.dtype}"
        return res

    @classmethod
    @docstring_format_args(stride_range)
    def constraint_stride_range(cls, op):
        "Stride values for both width and height must be in the range [{}, {}]"
        w, h = op.get_kernel_stride()
        stride_min, stride_max = cls.stride_range
        valid = (stride_min <= w <= stride_max) and (stride_min <= h <= stride_max)
        return valid, f"Op has stride WxH as: {w}x{h}"

    @classmethod
    @docstring_format_args(dilated_height_range)
    def constraint_dilated_height_range(cls, op):
        "Dilated kernel height must be in the range [{}, {}]"
        h = op.kernel.area_height()
        dilated_height_min, dilated_height_max = cls.dilated_height_range
        valid = dilated_height_min <= h <= dilated_height_max
        return valid, f"Op has dilated kernel height as: {h}"

    @classmethod
    @docstring_format_args(dilated_product_range)
    def constraint_dilated_product_range(cls, op):
        "Product of dilated kernel width and height must be in the range [{}, {}]"
        product = op.kernel.area_width() * op.kernel.area_height()
        dilated_product_min, dilated_product_max = cls.dilated_product_range
        valid = dilated_product_min <= product <= dilated_product_max
        return valid, f"Op has product of dilated kernel width and height as: {product}"

    @staticmethod
    def constraint_weights_type(op):
        "Weight tensor must be 8-bit"
        weights = op.weights
        valid = weights.element_size() == 1
        return valid, f"Tensor '{weights.name}' is {int(weights.element_size() * 8)}-bit"

    @staticmethod
    def constraint_weights_const(op):
        "Weight tensor must be constant"
        weights = op.weights
        valid = weights.values is not None
        return valid, f"Tensor '{weights.name}' has non-constant values"

    @classmethod
    @docstring_format_args([weights_limit])
    def constraint_weights_limit(cls, op):
        "The sum of the weights cannot exceed {}"
        weights = op.weights
        values = weights.values.astype(np.int64) - weights.quantization.zero_point
        limit = np.amax(np.sum(np.absolute(values), axis=(0, 1, 2)))
        valid = limit <= cls.weights_limit
        return valid, f"Tensor '{weights.name}' has the sum of weights: {limit}"

    @staticmethod
    def constraint_bias_shape(op):
        "Optional Bias tensor must be of shape: 1D"
        bias = op.bias
        if bias:
            valid = len(bias.shape) == 1
            return valid, f"Tensor '{bias.name}' has shape: {bias.shape}"
        return True, "Op has no bias tensor"

    @classmethod
    @docstring_format_args([list_formatter(supported_bias_dtypes)])
    def constraint_bias_type(cls, op):
        "Optional Bias tensor must be of type: {}"
        bias = op.bias
        if bias:
            valid = bias.dtype in cls.supported_bias_dtypes
            return valid, f"Tensor '{bias.name}' has data type: {bias.dtype}"
        return True, "Op has no bias tensor"

    @staticmethod
    def constraint_bias_40bit(op):
        "Optional Bias tensor values must fit within 40-bits"
        bias = op.bias
        if bias and bias.dtype == DataType.int64 and bias.values is not None:
            valid = all(len(bin(value)[2:]) <= 40 for value in bias.values)
            return valid, f"Tensor '{bias.name}' has values larger than 40-bits"
        return True, "Op has no bias tensor, or it fits in 40-bit"

    @staticmethod
    def constraint_batch_size(op):
        "IFM Tensor batch size must be 1"
        valid = True
        extra = []
        for tens in (op.ifm, op.ifm2):
            if tens is not None:
                batch_size = full_shape(4, tens.shape, 1)[0]
                if batch_size != 1:
                    valid = False
                    extra.append(f"Tensor '{tens.name}' has batch size: {batch_size}")
        extra = "\n   ".join(extra)
        return valid, extra

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

    @staticmethod
    def constraint_stride_width_no_upper_limit(op):
        """Strides must fulfil the following criteria:
        - Stride h must be between 1 and 3 when ofm height is greater than 1
        - Stride w must be between 1 and 3 when ofm height is greater than 1 or
          stride w must be divisible by 2 or 3 and ifm width must be divisible
          by stride_w/2 or stride_w/3"""

        stride_w, stride_h = op.get_kernel_stride()
        stride_min = 1
        stride_max_h = 3
        ifm_width = op.ifm.shape[2]
        ofm_height = op.ofm.shape[1]
        ofm_width = op.ofm.shape[2]

        stride_h_valid = ofm_height == 1 or stride_min <= stride_h <= stride_max_h

        _, optimized_stride = calc_resize_factor(ifm_width, stride_w) if stride_w > 1 else (1, stride_w)
        # Optimized stride indicates the final Conv2D stride width after all optimizations are performed
        can_optimize_stride_width_gt_3 = optimized_stride <= 3

        stride_w_valid = ofm_width == 1 or ((stride_min <= stride_w) and can_optimize_stride_width_gt_3)

        return (
            stride_h_valid and stride_w_valid,
            f"Op has stride WxH as: {stride_w}x{stride_h}, ifm shape as: {op.ifm.shape}, ofm shape as: {op.ofm.shape}",
        )

    @staticmethod
    def constraint_stride_range_no_padding(op):
        """Stride width must be greater than or equal to 1.
        For stride width greater than 3, valid padding needs to be used."""
        w, _ = op.get_kernel_stride()
        valid, message = TFLiteSupportedOperators.constraint_stride_width_no_upper_limit(op)
        padding = op.attrs.get("padding", None)
        is_optimized_with_valid_padding = padding in (None, Padding.VALID) or w <= 3
        valid = valid and is_optimized_with_valid_padding
        return valid, f"{message}, padding: {padding}"

    @staticmethod
    def constraint_depthwise_conv_stride(op):
        "Stride values for both width and height must be between 1 and 3"
        w, h = op.get_kernel_stride()
        stride_min, stride_max = 1, 3
        valid = (stride_min <= w <= stride_max) and (stride_min <= h <= stride_max)
        return valid, f"Op has stride WxH as: {w}x{h}"

    @staticmethod
    def constraint_tconv_stride(op):
        """Stride values for width and height must match one of the following criteria:
        Stride values WxH must be 1x1 or 2x2
        Stride WxH 2x1 supported if ifm height and kernel height = 1"""
        s_w = op.kernel.stride.x
        s_h = op.kernel.stride.y
        k_h = op.kernel.height
        i_h = op.ifm.shape[1]
        valid = False
        if s_w == 1 and s_h == 1:
            valid = True

        if s_w == 2 and s_h == 2:
            valid = True

        if s_w == 2 and s_h == 1 and i_h == 1 and k_h == 1:
            valid = True

        return valid, f"Op has ifm_height={i_h}, kernel_height={k_h} and stride WxH as {s_w}x{s_h}"

    @staticmethod
    def constraint_tconv_same(op):
        "SAME padding: OFM dimensions must equal IFM dimensions multiplied by stride"
        if op.attrs["padding"] == Padding.SAME:
            w = op.kernel.stride.x
            h = op.kernel.stride.y
            ifm_shape = op.ifm.shape
            ofm_shape = op.ofm.shape
            valid = (ofm_shape[1] == (ifm_shape[1] * h)) and (ofm_shape[2] == (ifm_shape[2] * w))
            return valid, f"Op has ifm_shape={ifm_shape}, ofm_shape={ofm_shape} and stride WxH as {w}x{h}"
        return True, "Op has padding=VALID"

    @staticmethod
    def constraint_tconv_valid(op):
        """VALID padding: OFM dimensions must equal IFM dimensions multiplied by stride,
        minus difference between kernel size and stride"""
        if op.attrs["padding"] == Padding.VALID:
            s_w = op.kernel.stride.x
            s_h = op.kernel.stride.y
            k_w = op.kernel.width
            k_h = op.kernel.height
            ifm_shape = op.ifm.shape
            ofm_shape = op.ofm.shape
            height_check = ofm_shape[1] == (ifm_shape[1] * s_h + max(k_h - s_h, 0))
            width_check = ofm_shape[2] == (ifm_shape[2] * s_w + max(k_w - s_w, 0))
            valid = height_check and width_check
            extra = (
                f"Op has ifm_shape={ifm_shape}, ofm_shape={ofm_shape},"
                f" stride WxH as {s_w}x{s_h} and kernel WxH as {k_w}x{k_h}"
            )
            return valid, extra
        return True, "Op has padding=SAME"

    @classmethod
    @docstring_format_args(filter_range)
    def constraint_filter_range(cls, op):
        "Kernel filter values for both width and height must be in the range [{}, {}]"
        if op.attrs["padding"] == Padding.SAME:
            sw, _ = op.get_kernel_stride()
            w = op.kernel.width
            h = op.kernel.height
            filter_min, filter_max = cls.filter_range
            valid = ((filter_min <= w <= filter_max) or sw == w) and (filter_min <= h <= filter_max)
            return valid, f"Op has kernel filter WxH as: {w}x{h}"
        return True, "Op has padding=VALID"

    @classmethod
    @docstring_format_args(filter_height_range)
    def constraint_filter_height_range(cls, op):
        "Kernel filter height must be in the range [{}, {}]"
        h = op.kernel.height
        filter_height_min, filter_height_max = cls.filter_height_range
        valid = filter_height_min <= h <= filter_height_max
        return valid, f"Op has kernel filter height as: {h}"

    @classmethod
    @docstring_format_args(filter_product_range)
    def constraint_filter_product_range(cls, op):
        "Product of kernel filter width and height must be in the range [{}, {}]"
        product = op.kernel.elements_wh()
        filter_product_min, filter_product_max = cls.filter_product_range
        valid = filter_product_min <= product <= filter_product_max
        return valid, f"Op has product of kernel filter width and height as: {product}"

    @staticmethod
    @docstring_format_args(filter_height_range)
    def constraint_filter_height_range_valid_pad(op):
        "VALID padding: Kernel filter height must be in the range [{}, {}]"
        if op.attrs["padding"] == Padding.VALID:
            return TFLiteSupportedOperators.constraint_filter_height_range(op)
        return True, "Op has padding=SAME"

    @staticmethod
    @docstring_format_args(filter_product_range)
    def constraint_filter_product_range_valid_pad(op):
        "VALID padding: Product of kernel filter width and height must be in the range [{}, {}]"
        if op.attrs["padding"] == Padding.VALID:
            return TFLiteSupportedOperators.constraint_filter_product_range(op)
        return True, "Op has padding=SAME"

    @staticmethod
    def constraint_resize(op):
        """The width and height of the IFM and OFM must match one of the following criteria:
        IFM W and H must both be 1
        IFM must match OFM
        W and H scaling must be equal and OFM W-1 and H-1 must be 2x/4x/8x IFM W-1 and H-1, if align_corners is True
        W and H scaling must be equal and OFM W and H must be 2x/4x/8x IFM W and H, if align_corners is False"""
        # Easier to start with False condition as very few cases result in a supported resize
        valid = False
        ifm_shape = op.ifm.shape
        ifm_shape_h = ifm_shape[1]
        ifm_shape_w = ifm_shape[2]
        ofm_shape = op.ofm.shape
        ofm_shape_h = ofm_shape[1]
        ofm_shape_w = ofm_shape[2]

        align_corners = op.attrs.get("align_corners", False)
        if len(ifm_shape) == 4:
            # Valid if IFM W and H are both 1, or IFM and OFM shape are the same
            if ((ifm_shape_h == 1) and (ifm_shape_w == 1)) or (ifm_shape == ofm_shape):
                valid = True
            else:
                # Valid if OFM is 2/4/8x IFM (-1 for align corners)
                if align_corners:
                    h_upscale_factor = (ofm_shape_h - 1) / (ifm_shape_h - 1)
                    w_upscale_factor = (ofm_shape_w - 1) / (ifm_shape_w - 1)
                else:
                    h_upscale_factor = ofm_shape_h / ifm_shape_h
                    w_upscale_factor = ofm_shape_w / ifm_shape_w

                # could use either height or width. save as int because it is more usable later in graph optimiser
                op.attrs["upscale_factor"] = int(h_upscale_factor)
                valid = h_upscale_factor == w_upscale_factor and h_upscale_factor in (2.0, 4.0, 8.0)

        return valid, f"Op has ifm_shape={ifm_shape}, ofm_shape={ofm_shape} and align_corners={align_corners}"

    @staticmethod
    def constraint_resize_size(op):
        "The size tensor must match the output tensor shape"
        valid = False
        ofm_shape = op.ofm.shape
        size_h, size_w = None, None
        # check that the size tensor (the second input) exists, is not none, and has the correct values
        if len(op.inputs) == 2 and op.inputs[1] is not None and len(op.inputs[1].values) == 2:
            size_h, size_w = op.inputs[1].values
            # check size and output size match
            if size_h == ofm_shape[1] and size_w == ofm_shape[2]:
                valid = True

        return valid, f"Op has size={size_h}x{size_w} and ofm_shape={ofm_shape}."

    @staticmethod
    def constraint_resize_attrs(op):
        "Both align_corners and half_pixel_centers can't be True"
        valid = True
        align_corners = op.attrs.get("align_corners", False)
        half_pixel_centers = op.attrs.get("half_pixel_centers", False)

        if align_corners and half_pixel_centers:
            valid = False
        return valid, "Op has both align_corners and half_pixel_centers set to True."

    @staticmethod
    def constraint_resizebi_half_pixel_centers_dims(op):
        """For half_pixel_centers the width and height of the IFM and OFM must match one of the following criteria:
        IFM W and H are both 1
        OFM W and H is 2x IFM W and H"""
        half_pixel_centers = op.attrs.get("half_pixel_centers", False)
        if not half_pixel_centers:
            valid = True
        elif len(op.ifm.shape) >= 3:
            ifm_h, ifm_w = op.ifm.shape[-3:-1]
            ofm_h, ofm_w = op.ofm.shape[-3:-1]
            if ifm_h == 1 and ifm_w == 1:
                valid = True
            else:
                valid = ofm_h / ifm_h == 2 and ofm_w / ifm_w == 2
        else:
            # Unexpected IFM shape
            valid = False
        return (
            valid,
            f"Op has ifm_shape={op.ifm.shape}, ofm_shape={op.ofm.shape} and half_pixel_centers={half_pixel_centers}",
        )

    @staticmethod
    def constraint_pad_shape(op):
        "The padding tensor must have the shape [3,2] or [4,2]"
        valid = op.inputs[1].shape in ([3, 2], [4, 2])
        return valid, f"The pad tensor has the shape: {op.inputs[1].shape}"

    @classmethod
    @docstring_format_args([list_formatter(supported_pad_dtypes)])
    def constraint_pad_type(cls, op):
        "Pad tensor must be of type: {}"
        pad_tensor = op.inputs[1]
        valid = pad_tensor.dtype in cls.supported_pad_dtypes
        return valid, f"Tensor '{pad_tensor.name}' has data type: {pad_tensor.dtype}"

    @staticmethod
    def constraint_padding_dimensions(op):
        "The pad tensor can only pad width and height"
        pad_tensor = op.inputs[1].values

        valid = sum(pad_tensor[-1, :]) == 0
        if valid and len(pad_tensor) > 3:
            valid = sum(pad_tensor[0, :]) == 0
        return valid, f"First dimension padding: {pad_tensor[0,:]}, last dimension padding: {pad_tensor[-1,:]}"

    @staticmethod
    def constraint_stridedslice_stride_values(op):
        "All Strides values must be 1"
        strides = op.inputs[3]
        valid = all(stride == 1 for stride in strides.values)
        return valid, f"Op has strides values {strides.values}"

    @staticmethod
    def constraint_stridedslice_offset_false(op):
        "Offset attribute must be False"
        offset = op.attrs.get("offset", False)
        valid = offset is False
        return valid, f"Op has offset={offset}"

    @staticmethod
    def constraint_inputs_int32(op):
        "Both Input data types must be int32"
        ifm_dtype = op.ifm.dtype
        ifm2_dtype = op.ifm2.dtype
        valid = (ifm_dtype == DataType.int32) and (ifm2_dtype == DataType.int32)
        return valid, f"Op has ifm_dtype={ifm_dtype} and ifm2_dtype={ifm2_dtype}"

    @staticmethod
    def constraint_output_int32(op):
        "OFM must be int32"
        ofm_dtype = op.ofm.dtype
        valid = ofm_dtype == DataType.int32
        return valid, f"Op has ofm_dtype={ofm_dtype}"

    @staticmethod
    def constraint_matching_quantization_parameters(op):
        "Both Input quantization parameters must match OFM quantization parameters"
        valid = True
        extra = []
        if not check_quantized_tens_scaling_equal(op.ofm, op.ifm):
            valid = False
            extra.append(op.ifm.name)
        if op.ifm2 is not None and not check_quantized_tens_scaling_equal(op.ofm, op.ifm2):
            valid = False
            extra.append(op.ifm2.name)
        extra = ", ".join(extra)
        return valid, f"Op has tensors with different quantization parameters to the OFM '{op.ofm.name}': {extra}"

    @staticmethod
    def constraint_broadcast_shapes(op):
        "Broadcasting is only allowed for rank indices with dimension 1, from either IFM1 or IFM2"
        ifm_shape = op.ifm.shape
        ifm2_shape = op.ifm2.shape if op.ifm2 else None
        ofm_shape = op.ofm.shape
        valid = True
        if ifm_shape is not None and ifm2_shape is not None:
            # align trailing dimensions
            size = min(len(ifm_shape), len(ifm2_shape))
            for i, i2, o in zip(ifm_shape[-size:], ifm2_shape[-size:], ofm_shape[-size:]):
                mi = max(i, i2)
                # Input dimensions should match or one should be of dimension 1
                # Output dimension should match the largest input dimension, together
                # with constraint_match_either_shapes ensures broadcast from only one input
                if not (i == i2 or i == 1 or i2 == 1) or o != mi:
                    valid = False
                    break

        return valid, f"Op has ifm_shape={ifm_shape} and ifm2_shape={ifm2_shape}"

    @classmethod
    @docstring_format_args([mean_kernel_product_int8, mean_kernel_product_uint8, mean_kernel_product_int16])
    def constraint_mean_height_width_product(cls, op):
        """Product of reduced axes must be no greater than:
        - {} for signed 8-bit inputs.
        - {} for unsigned 8-bit inputs.
        - {} for signed 16-bit inputs."""
        shape = op.inputs[0].shape
        if op.inputs[1].shape == []:
            axis = [int(op.inputs[1].values)]
        else:
            axis = list(op.inputs[1].values)

        # compute the product of the shape of all reduced axes
        axis_shapes = [shape[ax] for ax in axis]
        prod = np.prod(axis_shapes)

        if op.ifm.dtype == DataType.int16:
            max_prod = cls.mean_kernel_product_int16
            datatype = "int16"
        elif op.ifm.dtype == DataType.uint8:
            max_prod = cls.mean_kernel_product_uint8
            datatype = "uint8"
        else:
            max_prod = cls.mean_kernel_product_int8
            datatype = "int8"
        return prod <= max_prod, f"Datatype is {datatype}, product of axes is {prod}"

    @classmethod
    @docstring_format_args([mean_reduced_axis_max_size])
    def constraint_mean_width(cls, op):
        """If Width axis is reduced its shape must be no greater than {}."""
        shape = op.inputs[0].shape
        hi = 0 if len(shape) < 4 else 1
        h, w = shape[hi : hi + 2]
        max_width = cls.mean_reduced_axis_max_size
        return w <= max_width, f"Width is {w}"

    @classmethod
    @docstring_format_args([mean_reduced_axis_max_size])
    def constraint_mean_depth(cls, op):
        """If Depth axis is reduced its shape must be no greater than {}."""
        max_depth = cls.mean_reduced_axis_max_size
        shape = op.inputs[0].shape

        if op.inputs[1].shape == []:
            axis = [int(op.inputs[1].values)]
        else:
            axis = list(op.inputs[1].values)

        depth_idx = len(shape) - 1

        supported = True
        if depth_idx in axis and shape[-1] > max_depth:
            supported = False

        return supported, f"Depth is {shape[-1]}, shape is {shape}, axis is {axis}"

    @staticmethod
    def constraint_reshape_shape_constant(op):
        "Shape must be constant"
        valid = True
        extra = []

        # if a reshape tensor is specified then it must be constant
        if len(op.inputs) > 1:
            reshape_tens = op.inputs[1]
            if reshape_tens is not None:
                # constant inputs have either no driving operator or a const one
                # create a list of non-constant inputs
                if not (len(reshape_tens.ops) == 0 or reshape_tens.ops[0].type == Op.Const):
                    valid = False
                    extra.append(reshape_tens.name)
        extra = ", ".join(extra)

        return valid, f"Op has non-const input(s): {extra}"

    @staticmethod
    def constraint_argmax_axis(op):
        "Operation must be performed along the depth axis"
        inp_dims = len(op.inputs[0].shape)
        axis = op.inputs[1].values
        return (
            axis in (inp_dims - 1, -1),
            f"Axis is {axis} and number of input dimensions is {inp_dims}",
        )

    @staticmethod
    def constraint_argmax_depth(op):
        "IFM depth must be no greater than 127"
        ifm_depth = op.inputs[0].shape[-1]
        return ifm_depth <= 127, f"IFM depth is {ifm_depth}"

    @staticmethod
    def constraint_lstm_no_cifg(op):
        "Must not use CIFG"
        cifg = None not in op.inputs[2:5] + op.inputs[6:9]
        cifg = cifg and op.inputs[1] is None
        cifg = cifg and op.inputs[5] is None
        return not cifg, "Op uses CIFG"

    @staticmethod
    def constraint_lstm_no_peep_hole(op):
        "Must not use Peephole"
        valid = all([tens is None for tens in op.inputs[9:12]])
        return valid, "Op uses peephole"

    @staticmethod
    def constraint_lstm_no_projection(op):
        "Must not use Projection"
        valid = all([tens is None for tens in op.inputs[16:18]])
        return valid, "Op uses projection"

    @staticmethod
    def constraint_lstm_no_normalisation(op):
        "Must not use Normalisation"
        valid = all([tens is None for tens in op.inputs[20:24]])
        return valid, "Op uses normalisation"

    @staticmethod
    def constraint_lstm_weights(op):
        "All input and recurrent weights must be available"
        valid = None not in op.inputs[1:9]
        return valid, "Op has missing weights"

    @staticmethod
    def constraint_lstm_weight_dimensions(op):
        "All recurrent weights must be 2D"
        valid = all([len(input.shape) == 2 for input in op.inputs[5:9]])
        return valid, "Op recurrent weights are not 2D"

    @staticmethod
    def constraint_rsqrt_input_int8(op):
        "IFM must be int8"
        ifm_dtype = op.ifm.dtype
        valid = ifm_dtype == DataType.int8
        return valid, f"Op has ifm_dtype={ifm_dtype}"

    @staticmethod
    def constraint_slice_inputs_const(op):
        "Begin and Size Input tensors must be constant"
        valid = True
        extra = []
        _, begin, sizes = op.inputs
        if begin.values is None:
            valid = False
            extra.append(f"Begin tensor '{begin.name}'")
        if sizes.values is None:
            valid = False
            extra.append(f"Size tensor '{sizes.name}'")
        extra = ", ".join(extra)
        return valid, f"Op has non-constant tensors: {extra}"

    @staticmethod
    def constraint_transpose(op):
        """The following shape/permutations are supported for transpose:
        When ifm rank is 2: WxC -> CxW
        When ifm rank is 3: HxWxC -> WxHxC, 1xWxC -> 1xCxW, Hx1xC -> Cx1xH
        When ifm rank is 4: 1xHxWxC -> 1xWxHxC, 1x1xWxC -> 1x1xCxW, 1xHx1xC -> 1xCx1xW"""

        ifm_shape = op.inputs[0].shape
        perm = op.inputs[1]

        # WxC -> CxW
        valid = len(ifm_shape) == 2

        # HxWxC -> WxHxC
        if not valid and perm.shape == [3]:
            valid = perm.values[0] == 1 and perm.values[1] == 0

        # 1xWxC -> 1xCxW
        if not valid and perm.shape == [3] and ifm_shape[0] == 1:
            valid = perm.values[1] == 2 and perm.values[2] == 1

        # Hx1xC -> Cx1xH
        if not valid and perm.shape == [3] and ifm_shape[1] == 1:
            valid = perm.values[0] == 2 and perm.values[2] == 0

        # 1xHxWxC -> 1xWxHxC
        if not valid and perm.shape == [4]:
            valid = perm.values[0] == 0 and perm.values[1] == 2 and perm.values[2] == 1

        # 1x1xWxC -> 1x1xCxW
        if not valid and perm.shape == [4] and ifm_shape[1] == 1:
            valid = perm.values[0] == 0 and perm.values[2] == 3 and perm.values[3] == 2

        # 1xHx1xC -> 1xCx1xH
        if not valid and perm.shape == [4] and ifm_shape[2] == 1:
            valid = perm.values[0] == 0 and perm.values[1] == 3 and perm.values[3] == 1

        return valid, f"Op has ifm_shape: {ifm_shape} and permutation is: {perm.values}"
