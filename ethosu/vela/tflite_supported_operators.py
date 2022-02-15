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
# Description:
# The TFLiteSupportedOperators class which is a collection of all TFLite supported operators and parameter checks.
from collections import defaultdict

import numpy as np

from .data_type import DataType
from .operation import Op
from .operation import Padding
from .supported_operators_util import docstring_format_args
from .supported_operators_util import list_formatter
from .tensor import check_quantized_tens_scaling_equal
from .tflite_mapping import BUILTIN_OPERATOR_UNKNOWN
from .tflite_mapping import optype_to_builtintype


def _optype_formatter(op_list):
    # Convert internal op types to external names
    output = map(optype_to_builtintype, op_list)
    # Remove UNKNOWNs
    output = (x for x in output if x is not BUILTIN_OPERATOR_UNKNOWN)
    return list_formatter(output)


class TFLiteSupportedOperators:
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
        # Mean (converts to depthwise conv)
        | set((Op.Mean,))
    )
    unary_elem_wise_main_ops = Op.op_set(Op.is_unary_elementwise_op)
    binary_elem_wise_min_max_ops = set((Op.Minimum, Op.Maximum,))
    binary_elem_wise_shift_ops = set((Op.SHL, Op.SHR,))
    binary_elem_wise_add_mul_sub = set((Op.Add, Op.Mul, Op.Sub,))
    binary_elem_wise_main_ops = binary_elem_wise_min_max_ops | binary_elem_wise_add_mul_sub | binary_elem_wise_shift_ops
    elem_wise_main_ops = binary_elem_wise_main_ops | unary_elem_wise_main_ops
    pad_ops = set((Op.Pad,))
    supported_int32_tensor_ops = (
        set((Op.ReduceSum, Op.CLZ,)) | binary_elem_wise_add_mul_sub | binary_elem_wise_shift_ops
    )

    relu_ops = set((Op.Relu, Op.Relu6, Op.ReluN1To1, Op.Clip,))
    activation_ops = relu_ops | set((Op.Tanh, Op.Sigmoid, Op.Softmax, Op.HardSwish))
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
    memory_only_ops = set((Op.Reshape, Op.QuantizedReshape, Op.Squeeze, Op.ExpandDims,)) | concat_ops | split_ops
    per_axis_quant_ops = convolution_like_ops  # per-axis/channel quantization only currently supported for conv ops
    supported_fused_activations = relu_ops | set((Op.Tanh, Op.Sigmoid, Op.LUT,))
    supported_operators = npu_pre_ops | mac_main_ops | elem_wise_main_ops | pad_ops | npu_post_ops | memory_only_ops
    # Supported data types
    supported_op_dtypes = set((DataType.uint8, DataType.int8, DataType.int16, DataType.int32))
    supported_faf_dtypes = set((DataType.uint8, DataType.int8, DataType.int16))
    supported_bias_dtypes = set((DataType.int32, DataType.int64))
    supported_pad_dtypes = set((DataType.int32, DataType.int64))
    # Defined ranges for allowed values:
    tens_dim_range = (1, 65535)
    stride_range = (1, 3)
    dilation_range = (1, 2)
    dilated_height_range = (1, 64)
    dilated_product_range = (1, 64 * 64)
    weights_limit = 127 * 65536
    filter_range = (1, 8)
    filter_height_range = (1, 256)
    filter_product_range = (1, 256 * 256)
    mean_kernel_product = 64 * 64
    mean_kernel_product_int8 = 16 * 16
    mean_kernel_product_avgpool = 256 * 256

    def __init__(self):
        # Setup the generic constraints. Note: the order matters
        self.generic_constraints = []
        self.generic_constraints.append(TFLiteSupportedOperators.constraint_tens_dtype)
        self.generic_constraints.append(TFLiteSupportedOperators.constraint_tens_int32_ops)
        self.generic_constraints.append(TFLiteSupportedOperators.constraint_tens_dimension)
        self.generic_constraints.append(TFLiteSupportedOperators.constraint_tens_quant_per_axis)
        self.generic_constraints.append(TFLiteSupportedOperators.constraint_faf)
        self.generic_constraints.append(TFLiteSupportedOperators.constraint_faf_type)

        # Setup specific constraints. Note: the order matters
        self.specific_constraints = defaultdict(list)

        # Conv-like checks:
        for op_type in TFLiteSupportedOperators.convolution_like_ops:
            self.specific_constraints[op_type].append(TFLiteSupportedOperators.constraint_stride_range)
            self.specific_constraints[op_type].append(TFLiteSupportedOperators.constraint_dilation_range)
            self.specific_constraints[op_type].append(TFLiteSupportedOperators.constraint_dilated_height_range)
            self.specific_constraints[op_type].append(TFLiteSupportedOperators.constraint_dilated_product_range)
            self.specific_constraints[op_type].append(TFLiteSupportedOperators.constraint_weights_type)
            self.specific_constraints[op_type].append(TFLiteSupportedOperators.constraint_weights_const)
            self.specific_constraints[op_type].append(TFLiteSupportedOperators.constraint_weights_limit)
            self.specific_constraints[op_type].append(TFLiteSupportedOperators.constraint_bias_type)
            self.specific_constraints[op_type].append(TFLiteSupportedOperators.constraint_bias_40bit)
            self.specific_constraints[op_type].append(TFLiteSupportedOperators.constraint_batch_size)
        # Depthwise Conv specific checks:
        for op_type in TFLiteSupportedOperators.depthwise_convolution_ops:
            self.specific_constraints[op_type].append(TFLiteSupportedOperators.constraint_depth_multiplier)
        # Transpose Conv specific checks:
        for op_type in TFLiteSupportedOperators.transpose_convolution_ops:
            self.specific_constraints[op_type].append(TFLiteSupportedOperators.constraint_tconv_stride)
            self.specific_constraints[op_type].append(TFLiteSupportedOperators.constraint_tconv_same)
            self.specific_constraints[op_type].append(TFLiteSupportedOperators.constraint_tconv_valid)

        # Pooling checks:
        for op_type in TFLiteSupportedOperators.pooling_ops:
            self.specific_constraints[op_type].append(TFLiteSupportedOperators.constraint_batch_size)
            self.specific_constraints[op_type].append(TFLiteSupportedOperators.constraint_stride_range)
        # AVG pooling specific checks:
        for op_type in TFLiteSupportedOperators.avg_pooling_ops:
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
            self.specific_constraints[op_type].append(TFLiteSupportedOperators.constraint_bilinear_resize_attrs)

        # Vector Product specific checks:
        for op_type in TFLiteSupportedOperators.fc_vector_products:
            self.specific_constraints[op_type].append(TFLiteSupportedOperators.constraint_weights_type)
            self.specific_constraints[op_type].append(TFLiteSupportedOperators.constraint_weights_const)
            self.specific_constraints[op_type].append(TFLiteSupportedOperators.constraint_bias_type)
            self.specific_constraints[op_type].append(TFLiteSupportedOperators.constraint_bias_40bit)

        # Element-wise checks:
        for op_type in TFLiteSupportedOperators.elem_wise_main_ops:
            self.specific_constraints[op_type].append(TFLiteSupportedOperators.constraint_elemwise_batch_size)
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

        # Pad specific checks:
        self.specific_constraints[Op.Pad].append(TFLiteSupportedOperators.constraint_pad_shape)
        self.specific_constraints[Op.Pad].append(TFLiteSupportedOperators.constraint_padding_dimensions)
        self.specific_constraints[Op.Pad].append(TFLiteSupportedOperators.constraint_pad_type)

        # Mean specific checks:
        self.specific_constraints[Op.Mean].append(TFLiteSupportedOperators.constraint_batch_size)
        self.specific_constraints[Op.Mean].append(TFLiteSupportedOperators.constraint_mean_height_width_product_avgpool)
        self.specific_constraints[Op.Mean].append(TFLiteSupportedOperators.constraint_mean_height_width_product)
        self.specific_constraints[Op.Mean].append(TFLiteSupportedOperators.constraint_mean_height_width_product_int8)
        self.specific_constraints[Op.Mean].append(TFLiteSupportedOperators.constraint_mean_height_single_axis)

        # Reshape specific checks:
        self.specific_constraints[Op.Reshape].append(TFLiteSupportedOperators.constraint_reshape_shape_constant)

    def is_operator_supported(self, op):
        ext_type = optype_to_builtintype(op.type)
        if op.type not in TFLiteSupportedOperators.supported_operators:
            if op.type not in (Op.Placeholder, Op.SubgraphInput, Op.Const):
                print(f"Info: {ext_type} '{op.name}' is a CPU only op")
            return False

        for constraint in self.generic_constraints + self.specific_constraints[op.type]:
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
                if tens.quantization.is_per_axis():
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
    @docstring_format_args(dilation_range)
    def constraint_dilation_range(cls, op):
        "Dilation factor values for both width and height must be in the range [{}, {}]"
        w, h = op.get_kernel_dilation()
        dilation_min, dilation_max = cls.dilation_range
        valid = (dilation_min <= w <= dilation_max) and (dilation_min <= h <= dilation_max)
        return valid, f"Op has dilation factor WxH as: {w}x{h}"

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
        ifm = op.ifm
        valid = ifm.shape[0] == 1
        return valid, f"Tensor '{ifm.name}' has batch size: {ifm.shape[0]}"

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
    def constraint_tconv_stride(op):
        "Stride values for both width and height must be 2"
        w = op.kernel.stride.x
        h = op.kernel.stride.y
        valid = (w == 2) and (h == 2)
        return valid, f"Op has stride WxH as: {w}x{h}"

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
            w = op.kernel.width
            h = op.kernel.height
            filter_min, filter_max = cls.filter_range
            valid = (filter_min <= w <= filter_max) and (filter_min <= h <= filter_max)
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
        OFM W and H must be equal and 2/4/8x IFM -1, if align_corners is True
        OFM W and H must be equal and 2/4/8x IFM, if align_corners is False"""
        # Easier to start with False condition as very few cases result in a supported resize
        valid = False
        ifm_shape = op.ifm.shape
        ofm_shape = op.ofm.shape
        align_corners = op.attrs.get("align_corners", False)
        if len(ifm_shape) == 4:
            # Valid if IFM W and H are both 1, or IFM and OFM shape are the same
            if ((ifm_shape[1] == 1) and (ifm_shape[2] == 1)) or (ifm_shape == ofm_shape):
                valid = True
            else:
                # Valid if OFM is 2/4/8x IFM (-1 for align corners)
                w_upscale_factor = (ofm_shape[1] + 1) / ifm_shape[1] if align_corners else ofm_shape[1] / ifm_shape[1]
                h_upscale_factor = (ofm_shape[2] + 1) / ifm_shape[2] if align_corners else ofm_shape[2] / ifm_shape[2]

                valid = w_upscale_factor == h_upscale_factor and w_upscale_factor in [2, 4, 8]

        return valid, f"Op has ifm_shape={ifm_shape}, ofm_shape={ofm_shape} and align_corners={align_corners}"

    @staticmethod
    def constraint_bilinear_resize_attrs(op):
        "half_pixel_centers are not supported"
        valid = True
        if op.attrs.get("half_pixel_centers"):
            valid = False
        return valid, f"Op has half_pixel_centers set to {not valid}."

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
    def constraint_elemwise_batch_size(op):
        "Batch size must be 1 for Input tensors with more than 2 dimensions"
        valid = True
        extra = []
        for tens in (op.ifm, op.ifm2):
            # Unary ops have ifm2 as None
            if tens is not None:
                if (len(tens.shape) > 2) and (tens.shape[0] != 1):
                    valid = False
                    extra.append(tens.name)
        extra = ", ".join(extra)
        return valid, f"Op has invalid input tensors: {extra}"

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
    @docstring_format_args([mean_kernel_product_avgpool])
    def constraint_mean_height_width_product_avgpool(cls, op):
        """Product of height and width must be no greater than {}"""
        shape = op.inputs[0].shape
        hi = 0 if len(shape) < 4 else 1
        h, w = shape[hi : hi + 2]
        max_prod = cls.mean_kernel_product_avgpool
        return h * w <= max_prod, f"Product of height and width is {h * w}"

    @classmethod
    @docstring_format_args([mean_kernel_product])
    def constraint_mean_height_width_product(cls, op):
        """Product of height and width must be no greater than {} when:
        IFM and OFM have different scale or zero point; or
        'keep_dims' is True"""
        ifmq, ofmq = op.ifm.quantization, op.ofm.quantization
        keep_dims = op.attrs.get("keep_dims")
        # doesn't apply, size is checked by constraint_mean_height_width_product_avgpool
        if not keep_dims and ifmq.scale_f32 == ofmq.scale_f32 and ifmq.zero_point == ofmq.zero_point:
            return True, ""
        shape = op.inputs[0].shape
        hi = 0 if len(shape) < 4 else 1
        h, w = shape[hi : hi + 2]
        max_prod = cls.mean_kernel_product
        return h * w <= max_prod, f"Product of height and width is {h * w}"

    @classmethod
    @docstring_format_args([mean_kernel_product_int8])
    def constraint_mean_height_width_product_int8(cls, op):
        """Product of IFM height and width must be no greater than {} when:
        The IFM shape has 4 dimensions; and
        The axis indices specify reduction across 2 dimensions; and
        The axis indices correspond to the width and height dimensions of the IFM; and
        'keep_dims' is True; and
        IFM datatype is int8"""
        shape = op.ifm.shape
        axis = int(op.inputs[1].values) if op.inputs[1].shape == [] else list(op.inputs[1].values)
        # doesn't apply, size is checked by constraint_mean_height_width_product_avgpool
        # and constraint_mean_height_width_product
        if (
            len(shape) != 4
            or op.ifm.dtype != DataType.int8
            or not op.attrs.get("keep_dims")
            or axis not in ([1, 2], [2, 1])
        ):
            return True, ""
        h = shape[-3]
        w = shape[-2]
        max_prod = cls.mean_kernel_product_int8
        return h * w <= max_prod, f"Product of height and width is {h * w}"

    @classmethod
    @docstring_format_args([filter_height_range[1], dilated_height_range[1]])
    def constraint_mean_height_single_axis(cls, op):
        """For single axis averages across the height dimension:
        IFM height must be no greater than {} if the IFM and OFM scale and zero point match; otherwise
        IFM height must be no greater than {} if the IFM and OFM scale or zero point do not match"""
        inp, axis = op.inputs
        if axis.shape == [] or axis.shape[0] == 1:  # single axis
            axis = int(axis.values) if len(axis.shape) == 0 else int(axis.values[0])
        else:
            # Multiple axes
            return True, ""

        shape = inp.shape
        if len(shape) < 3:
            # No height dimension present in IFM
            return True, ""
        if axis != len(shape) - 3:
            # Not averaging across the height dimension
            return True, ""

        h = shape[axis]
        ifm, ofm = op.get_ifm_ofm()

        if check_quantized_tens_scaling_equal(ifm, ofm):
            return h <= cls.filter_height_range[1], f"Height is {h}, IFM and OFM quantizations match"
        else:
            return h <= cls.dilated_height_range[1], f"Height is {h}, IFM and OFM quantizations do not match"

    @staticmethod
    def constraint_reshape_shape_constant(op):
        "Shape must be constant"
        valid = True
        extra = []

        reshape_tens = op.inputs[1]
        if reshape_tens is not None:
            # constant inputs have either no driving operator or a const one
            # create a list of non-constant inputs
            if not (len(reshape_tens.ops) == 0 or reshape_tens.ops[0].type == Op.Const):
                valid = False
                extra.append(reshape_tens.name)
        extra = ", ".join(extra)

        return valid, f"Op has non-const input(s): {extra}"
