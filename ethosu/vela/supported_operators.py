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
import numpy as np

from .data_type import BaseType
from .data_type import DataType
from .operation import get_slice_offsets


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
    npu_pre_ops = set(("QuantizedResizeBilinear", "SplitSliceRead",))
    convolution_ops = set(("Conv2DBiasAct", "Conv2D", "QuantizedConv2D",))
    depthwise_convolution_ops = set(("DepthwiseConv2dBiasAct", "DepthwiseConv2dNative", "QuantizedDepthwiseConv2D",))
    transpose_convolution_ops = set(("Conv2DBackpropInput",))
    max_pooling_ops = set(("QuantizedMaxPool", "MaxPool", "MaxPoolAct",))
    avg_pooling_ops = set(("QuantizedAvgPool", "AvgPool", "AvgPoolAct",))
    pooling_ops = set(("ReduceSum",)) | max_pooling_ops | avg_pooling_ops
    resizing_ops = set(("ResizeBilinear",))
    fc_vector_products = set(("QuantizedMatMul", "MatMul", "FullyConnectedAct",))
    mac_main_ops = (
        # RNN/LSTM/GRU
        set(("BlockLSTM",))
        # convolutions
        | convolution_ops
        # depth-wise convolutions
        | depthwise_convolution_ops
        # transpose convolutions
        | transpose_convolution_ops
        # pooling
        | pooling_ops
        # resizing/upscaling
        | resizing_ops
        # FC layers
        | fc_vector_products
    )
    unary_elem_wise_main_ops = set(("LeakyRelu", "Abs", "CLZ",))
    binary_elem_wise_min_max_ops = set(("Minimum", "Maximum",))
    binary_elem_wise_shift_ops = set(("SHL", "SHR",))
    binary_elem_wise_add_mul_sub = set(
        ("AddAct", "MulAct", "SubAct", "QuantizedAdd", "QuantizedSub", "QuantizedMul", "Mul", "Add", "Sub",)
    )
    binary_elem_wise_main_ops = binary_elem_wise_min_max_ops | binary_elem_wise_add_mul_sub | binary_elem_wise_shift_ops
    elem_wise_main_ops = binary_elem_wise_main_ops | unary_elem_wise_main_ops
    supported_int32_tensor_ops = (
        set(("Requantize", "ReduceSum", "CLZ",)) | binary_elem_wise_add_mul_sub | binary_elem_wise_shift_ops
    )
    activation_ops = set(
        (
            "QuantizedRelu",
            "QuantizedRelu1",
            "QuantizedRelu6",
            "Relu",
            "Relu6",
            "ReluN1To1",
            "Sigmoid",
            "Tanh",
            "Softmax",
        )
    )
    npu_post_ops = (
        # concatenation write direction
        set(("ConcatSliceWrite",))
        # bias add and batch norm
        | set(("QuantizedBiasAdd", "Requantize", "QuantizedBatchNorm", "BiasAdd", "FusedBatchNorm",))
        # Quantization
        | set(("Quantize",))
        # activation functions
        | activation_ops
    )
    split_ops = set(("Split", "SplitV", "StridedSlice", "Slice", "UnpackReshaped", "Unpack",))
    concat_ops = set(("Concat", "ConcatV2", "QuantizedConcat", "ConcatTFLite", "PackReshaped", "Pack",))
    memory_only_ops = set(("Squeeze", "Reshape", "QuantizedReshape", "ExpandDims",)) | concat_ops | split_ops
    shapeless_input_ops = set(("Split", "SplitV",)) | binary_elem_wise_main_ops
    supported_fused_activations = set(("Relu", "Relu6", "ReluN1To1", "Tanh", "Sigmoid", "LUT",))
    supported_operators = npu_pre_ops | mac_main_ops | elem_wise_main_ops | npu_post_ops | memory_only_ops
    supported_dtypes = set((DataType.uint8, DataType.int8, DataType.int16, DataType.int32))
    # Defined ranges for allowed values:
    tens_dim_range = (1, 65535)

    def __init__(self):
        # Setup supported operator restriction checkers
        self.supported_operator_restrictions = {}
        self.supported_operator_restrictions.update(
            {op: self.check_convolution_restrictions for op in SupportedOperators.convolution_ops}
        )
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
        # Setup the generic constraints
        self.generic_constraints = []
        self.generic_constraints.append(SupportedOperators.constraint_tens_defined_shape)
        self.generic_constraints.append(SupportedOperators.constraint_tens_shapeless)
        self.generic_constraints.append(SupportedOperators.constraint_tens_shape_size)
        self.generic_constraints.append(SupportedOperators.constraint_tens_dtype)
        self.generic_constraints.append(SupportedOperators.constraint_tens_dimension)
        self.generic_constraints.append(SupportedOperators.constraint_faf)
        self.generic_constraints.append(SupportedOperators.constraint_tens_quant_scale)

    def is_operator_supported(self, op):
        if op.type not in SupportedOperators.supported_operators:
            return False
        for constraint in self.generic_constraints:
            valid, extra = constraint(op)
            if not valid:
                print('Warning: "{}" is not supported on the NPU. Placing on CPU instead'.format(op.type))
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
        for tens in op.inputs + op.outputs:
            if tens:
                valid &= tens.has_fully_defined_shape()
                extra.append("shape={}".format(tens.shape))
        return valid, " ".join(extra)

    @classmethod
    @docstring_format_args([shapeless_input_ops])
    def constraint_tens_shapeless(cls, op):
        "Scalar or Broadcasting Tensors are only valid for Input Tensors, and when op type is: {}"
        valid = True
        extra = []
        for tens in op.inputs:
            if tens and tens.shape == []:
                valid &= op.type in cls.shapeless_input_ops
                extra.append("shape={}".format(tens.shape))
        for tens in op.outputs:
            if tens.shape == []:
                valid = False
                extra.append("shape={}".format(tens.shape))
        return valid, " ".join(extra)

    @staticmethod
    def constraint_tens_shape_size(op):
        "Input(s) and Output Tensors must not be greater than 4D"
        valid = True
        extra = []
        for tens in op.inputs + op.outputs:
            if tens:
                valid &= len(tens.shape) <= 4
                extra.append("shape={}".format(tens.shape))
        return valid, " ".join(extra)

    @classmethod
    @docstring_format_args([supported_dtypes, supported_int32_tensor_ops])
    def constraint_tens_dtype(cls, op):
        "Tensors must be of type: {}. Tensors which are int32 are only valid when op type is: {}"
        valid = True
        extra = []
        tensors = [tens for tens in op.get_ifm_ifm2_weights_ofm() if tens]
        tensors = tensors if tensors else op.inputs
        for tens in tensors:
            if tens.dtype == DataType.int32:
                valid &= op.type in cls.supported_int32_tensor_ops
            else:
                valid &= tens.dtype in cls.supported_dtypes
            extra.append("dtype={}".format(tens.dtype))
        return valid, " ".join(extra)

    @classmethod
    @docstring_format_args(tens_dim_range)
    def constraint_tens_dimension(cls, op):
        "Tensor dimensions must be in the range {}-{} (inclusive)"
        tens_min, tens_max = cls.tens_dim_range
        valid = True
        extra = []
        tensors = [tens for tens in op.get_ifm_ifm2_weights_ofm() if tens]
        tensors = tensors if tensors else op.inputs
        for tens in tensors:
            valid &= all(tens_min <= dim <= tens_max for dim in tens.shape)
            extra.append("shape={}".format(tens.shape))
        return valid, " ".join(extra)

    @classmethod
    @docstring_format_args([supported_fused_activations])
    def constraint_faf(cls, op):
        "The fused activation function (if present) must be one of type: {}"
        faf = op.attrs.get("fused_activation_function")
        valid = (faf is None) or (faf in cls.supported_fused_activations)
        extra = "fused_activation_function={}".format(faf)
        return valid, extra

    @staticmethod
    def constraint_tens_quant_scale(op):
        "Tensors with quantization scales must be finite"
        valid = True
        extra = []
        tensors = [tens for tens in op.get_ifm_ifm2_weights_ofm() if tens]
        for tens in tensors:
            if tens.quantization is not None and tens.quantization.scale_f32 is not None:
                valid &= not np.isinf(tens.quantization.scale_f32).any()
                extra.append("quantization.scale_f32={}".format(tens.quantization.scale_f32))
        return valid, " ".join(extra)

    @classmethod
    def check_convolution_restrictions(cls, op):
        # check stride
        if op.attrs["stride_w"] > 3 or op.attrs["stride_h"] > 3:
            return False

        # check dilation
        dilation_w_factor = op.attrs.get("dilation_w_factor", 1)
        dilation_h_factor = op.attrs.get("dilation_h_factor", 1)
        if dilation_w_factor > 2 or dilation_h_factor > 2:
            return False

        # check data type
        ifm_tensor, _, weight_tensor, bias_tensor, _ = op.get_ifm_ifm2_weights_biases_ofm()
        if weight_tensor.element_size() > 1:
            return False

        if not cls.check_bias_restrictions(bias_tensor):
            return False

        # check kernel size [HWIO]
        dilated_weight_w = weight_tensor.shape[1] + (weight_tensor.shape[1] - 1) * (dilation_w_factor - 1)
        dilated_weight_h = weight_tensor.shape[0] + (weight_tensor.shape[0] - 1) * (dilation_h_factor - 1)

        if dilated_weight_w > 64 or dilated_weight_h > 64:
            return False

        # check non const weights
        if weight_tensor.values is None:
            print("Warning:", op.type, "has non-const weights, placing on CPU")
            return False

        # check weight sums over [HWI]
        zero_point = weight_tensor.quantization.zero_point
        quant_weights = weight_tensor.quant_values.astype(np.int64)
        weights = quant_weights - zero_point
        totals = np.sum(np.absolute(weights), axis=(0, 1, 2))

        if np.amax(totals) > 127 * 65536:
            return False

        # check batch size
        if ifm_tensor.shape[0] != 1:
            return False

        return True

    @classmethod
    def check_depthwise_convolution_restrictions(cls, op):
        # check depth
        ifm_tensor, _, _, ofm_tensor = op.get_ifm_ifm2_weights_ofm()
        if op.attrs["depth_multiplier"] > 1 and not (
            (ifm_tensor.shape[3] == 1) and (ofm_tensor.shape[3] == op.attrs["depth_multiplier"])
        ):
            return False
        return cls.check_convolution_restrictions(op)

    @classmethod
    def check_transpose_convolution_restrictions(cls, op):
        # check stride
        stride_h, stride_w = op.attrs["stride_h"], op.attrs["stride_w"]
        if stride_h != stride_w != 2:
            return False

        # check output dimensions
        ifm_tensor, weight_tensor, _, ofm_tensor = op.get_ifm_weights_biases_ofm()
        ifm_h, ifm_w = ifm_tensor.shape[1], ifm_tensor.shape[2]
        ofm_h, ofm_w = ofm_tensor.shape[1], ofm_tensor.shape[2]
        if op.attrs["padding"] == b"SAME":
            if (ofm_h != ifm_h * stride_h) or (ofm_w != ifm_w * stride_w):
                return False
        elif op.attrs["padding"] == b"VALID":
            kernel_h, kernel_w = weight_tensor.shape[0], weight_tensor.shape[1]
            if (ofm_h != (ifm_h) * stride_h + max(kernel_h - stride_h, 0)) or (
                ofm_w != (ifm_w) * stride_w + max(kernel_w - stride_w, 0)
            ):
                return False

        return cls.check_convolution_restrictions(op)

    @classmethod
    def check_pooling_restrictions(cls, op):
        # check stride
        if op.attrs["stride_w"] > 3 or op.attrs["stride_h"] > 3:
            return False

        # check data type
        ifm_tensor, _, _, ofm_tensor = op.get_ifm_ifm2_weights_ofm()
        if ifm_tensor.dtype != ofm_tensor.dtype:
            if op.type != "ReduceSum":
                return False
            # TODO: else check ReduceSum restrictions.

        # check batch size
        if ifm_tensor.shape[0] != 1:
            return False

        if op.type in cls.avg_pooling_ops:
            # check kernel size
            if op.attrs["padding"] == b"SAME" and (op.attrs["filter_width"] > 8 or op.attrs["filter_height"] > 8):
                return False
            if op.attrs["padding"] == b"VALID" and (
                op.attrs["filter_width"] * op.attrs["filter_height"] > 256 * 256 or op.attrs["filter_height"] > 256
            ):
                return False

        if op.type in cls.max_pooling_ops:
            # check kernel size (any padding)
            if op.attrs["filter_width"] * op.attrs["filter_height"] > 256 * 256 or op.attrs["filter_height"] > 256:
                return False
        return True

    @classmethod
    def check_resize_restrictions(cls, op):
        # check unsupported upscaling factor
        if op.type == "ResizeBilinear":
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
        _, _, weight_tensor, bias_tensor, _ = op.get_ifm_ifm2_weights_biases_ofm()
        if weight_tensor.element_size() > 1:
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
            return False
        if op.type in cls.binary_elem_wise_add_mul_sub:
            # both inputs must have same type
            if ifm_tensor.dtype != ifm2_tensor.dtype:
                return False
            # signed input check
            if ifm_tensor.dtype.type & BaseType.Signed:
                # output must be signed
                if ofm_tensor.dtype.type & BaseType.Unsigned:
                    return False
                # and 8, 16 or 32-bit
                if ofm_tensor.element_size() not in (1, 2, 4):
                    return False
            # unsigned input check, output must be same type or int32
            if ifm_tensor.dtype.type & BaseType.Unsigned and not (
                ifm_tensor.dtype == ofm_tensor.dtype or ofm_tensor.dtype == DataType.int32
            ):
                return False
        elif op.type in cls.binary_elem_wise_shift_ops | set(("CLZ")):
            if ifm_tensor.dtype != DataType.int32 or ifm2_tensor.dtype != DataType.int32:
                return False
            if op.type in ("CLZ", "SHL") and ofm_tensor.dtype != DataType.int32:
                return False

        # check batch size
        if len(ifm_tensor.shape) > 2 and ifm_tensor.shape[0] != 1:
            return False
        if op.type in cls.binary_elem_wise_main_ops:  # if op type is unary, ifm2_tensor is None
            if len(ifm2_tensor.shape) > 2 and ifm2_tensor.shape[0] != 1:
                return False

        # negative alpha values are not supported
        if op.type == "LeakyRelu" and op.attrs["alpha"] < 0:
            return False

        # check if ifm or ifm2 has ofm shape
        if ifm_tensor.shape != ofm_tensor.shape and ifm2_tensor.shape != ofm_tensor.shape:
            return False

        if op.type in cls.binary_elem_wise_min_max_ops and not cls.check_quantization_restrictions_binary_elem_wise(op):
            return False

        return True

    @classmethod
    def check_memory_only_restrictions(cls, op):
        if op.type == "StridedSlice":
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
        if op.type == "SplitV":
            # check that maximum one size is set to -1, indicating that size should be inferred
            sizes = op.inputs[1].values
            num_to_be_inferred = 0
            for size in sizes:
                if size == -1:
                    num_to_be_inferred += 1

            if num_to_be_inferred > 1:
                print("Warning:", op.type, "has more than one size to be inferred, which is illegal, placing on CPU")
                return False
        if op.type.find("Concat") != -1:
            axis = op.attrs.get("axis", None)
            if axis is None:
                print("Warning:", op.type, "invalid or missing axis, placing on CPU")
                return False
            if axis < 0:
                axis += len(op.inputs[0].shape)
            if not 0 < axis < len(op.inputs[0].shape):
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
        if op.type == "Softmax":
            ifm_tensor = op.inputs[0]
            ofm_tensor = op.outputs[0]

            # check data type
            if ifm_tensor.dtype != ofm_tensor.dtype:
                return False

            if ifm_tensor.dtype not in (DataType.uint8, DataType.int8, DataType.int16):
                return False

            # check shape
            if ifm_tensor.shape != ofm_tensor.shape:
                return False

        return True

    @classmethod
    def check_bias_restrictions(cls, bias_tensor):
        # check data type
        if bias_tensor is not None and bias_tensor.dtype not in (DataType.int32, DataType.int64):
            return False

        # check if values fits in 40-bit
        if bias_tensor is not None and bias_tensor.dtype == DataType.int64:
            for quant_value in bias_tensor.quant_values:
                if not (-(1 << 39) <= quant_value < (1 << 39)):
                    return False

        return True
