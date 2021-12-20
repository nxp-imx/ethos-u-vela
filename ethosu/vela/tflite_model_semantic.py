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
# The TFLiteSemantic class which is a collection of TensorFlow lite model semantic checks.
from collections import defaultdict

import numpy as np

from .data_type import BaseType
from .data_type import DataType
from .numeric_util import is_integer
from .operation import get_slice_offsets
from .operation import Op
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


class TFLiteSemantic:
    # Categorised lists of operators
    convolution_ops = set((Op.Conv2DBias, Op.Conv2D, Op.QuantizedConv2D,))
    depthwise_convolution_ops = set((Op.DepthwiseConv2DBias,))
    transpose_convolution_ops = set((Op.Conv2DBackpropInput,))
    convolution_like_ops = convolution_ops | depthwise_convolution_ops | transpose_convolution_ops
    max_pooling_ops = Op.op_set(Op.is_maxpool_op)
    avg_pooling_ops = Op.op_set(Op.is_avgpool_op)
    pooling_ops = set((Op.ReduceSum,)) | max_pooling_ops | avg_pooling_ops
    unary_elem_wise_main_ops = Op.op_set(Op.is_unary_elementwise_op)
    binary_elem_wise_min_max_ops = set((Op.Minimum, Op.Maximum,))
    binary_elem_wise_shift_ops = set((Op.SHL, Op.SHR,))
    binary_elem_wise_add_mul_sub = set((Op.Add, Op.Mul, Op.Sub,))
    binary_elem_wise_main_ops = binary_elem_wise_min_max_ops | binary_elem_wise_add_mul_sub | binary_elem_wise_shift_ops
    elem_wise_main_ops = binary_elem_wise_main_ops | unary_elem_wise_main_ops
    shapeless_input_ops = binary_elem_wise_main_ops | set((Op.Split, Op.SplitV, Op.Mean, Op.ExpandDims))
    reshape_ops = set((Op.Reshape, Op.QuantizedReshape, Op.Squeeze, Op.ExpandDims,))

    def __init__(self):
        # Setup the generic constraints. Note: the order matters
        self.generic_constraints = []
        self.generic_constraints.append(TFLiteSemantic.constraint_tens_no_dynamic)
        self.generic_constraints.append(TFLiteSemantic.constraint_tens_defined_shape)
        self.generic_constraints.append(TFLiteSemantic.constraint_tens_output_scalar)
        self.generic_constraints.append(TFLiteSemantic.constraint_tens_input_scalar)
        self.generic_constraints.append(TFLiteSemantic.constraint_tens_shape_size)

        self.generic_constraints.append(TFLiteSemantic.constraint_tens_quant_none_check)
        self.generic_constraints.append(TFLiteSemantic.constraint_tens_quant_scale)
        self.generic_constraints.append(TFLiteSemantic.constraint_quant_scale_inf)
        self.generic_constraints.append(TFLiteSemantic.constraint_none_const_tensors)

        # Setup specific constraints. Note: the order matters
        self.specific_constraints = defaultdict(list)

        # Conv-like checks:
        for op_type in TFLiteSemantic.convolution_like_ops:
            self.specific_constraints[op_type].append(TFLiteSemantic.constraint_stride_type)
            self.specific_constraints[op_type].append(TFLiteSemantic.constraint_dilation_type)

        # Pooling checks:
        for op_type in TFLiteSemantic.pooling_ops:
            self.specific_constraints[op_type].append(TFLiteSemantic.constraint_stride_type)
        # AVG pooling specific checks:
        for op_type in TFLiteSemantic.avg_pooling_ops:
            self.specific_constraints[op_type].append(TFLiteSemantic.constraint_matching_in_out_types)
            self.specific_constraints[op_type].append(TFLiteSemantic.constraint_filter_type)
        # MAX pooling specific checks:
        for op_type in TFLiteSemantic.max_pooling_ops:
            self.specific_constraints[op_type].append(TFLiteSemantic.constraint_matching_in_out_types)
            self.specific_constraints[op_type].append(TFLiteSemantic.constraint_filter_type)

        # Concat specific checks:
        for op_type in (Op.Concat, Op.ConcatTFLite):
            self.specific_constraints[op_type].append(TFLiteSemantic.constraint_axis_exists)
            self.specific_constraints[op_type].append(TFLiteSemantic.constraint_axis_valid)
            self.specific_constraints[op_type].append(TFLiteSemantic.constraint_matching_dimensionality)
            self.specific_constraints[op_type].append(TFLiteSemantic.constraint_valid_dimensions)

        # Element-wise checks:
        for op_type in TFLiteSemantic.elem_wise_main_ops:
            self.specific_constraints[op_type].append(TFLiteSemantic.constraint_matching_either_shapes)
        # Unary specific checks:
        for op_type in TFLiteSemantic.unary_elem_wise_main_ops:
            self.specific_constraints[op_type].append(TFLiteSemantic.constraint_matching_in_out_types)
        # Binary Min/Max specific checks:
        for op_type in TFLiteSemantic.binary_elem_wise_min_max_ops:
            self.specific_constraints[op_type].append(TFLiteSemantic.constraint_matching_in_out_types)
        # Binary Add/Mul/Sub specific checks:
        for op_type in TFLiteSemantic.binary_elem_wise_add_mul_sub:
            self.specific_constraints[op_type].append(TFLiteSemantic.constraint_matching_inputs_types)
            self.specific_constraints[op_type].append(TFLiteSemantic.constraint_matching_signed)
            self.specific_constraints[op_type].append(TFLiteSemantic.constraint_unsigned_valid)

        # Ops reshaping dimensions: Reshape, Squeeze and ExpandDims
        for op_type in TFLiteSemantic.reshape_ops:
            self.specific_constraints[op_type].append(TFLiteSemantic.constraint_matching_in_out_quant)

        # Softmax specific checks:
        self.specific_constraints[Op.Softmax].append(TFLiteSemantic.constraint_matching_shapes)
        self.specific_constraints[Op.Softmax].append(TFLiteSemantic.constraint_matching_in_out_types)
        self.specific_constraints[Op.Softmax].append(TFLiteSemantic.constraint_beta_value_range)

        # SplitV specific checks:
        self.specific_constraints[Op.SplitV].append(TFLiteSemantic.constraint_splitv_inferred)

        # StridedSlice specific checks:
        self.specific_constraints[Op.StridedSlice].append(TFLiteSemantic.constraint_stridedslice_input_count)
        self.specific_constraints[Op.StridedSlice].append(TFLiteSemantic.constraint_stridedslice_inputs_const)
        self.specific_constraints[Op.StridedSlice].append(TFLiteSemantic.constraint_ellipsis_mask)
        self.specific_constraints[Op.StridedSlice].append(TFLiteSemantic.constraint_axis_masks)
        self.specific_constraints[Op.StridedSlice].append(TFLiteSemantic.constraint_slice_ranges)

        # LeakyRelu specific checks:
        self.specific_constraints[Op.LeakyRelu].append(TFLiteSemantic.constraint_alpha_valid)

        # FullyConnected specific checks:
        self.specific_constraints[Op.FullyConnected].append(TFLiteSemantic.constraint_fc_output_2d)
        self.specific_constraints[Op.FullyConnected].append(TFLiteSemantic.constraint_keep_dim_ifm_ofm)

        # Pad specific checks:
        self.specific_constraints[Op.Pad].append(TFLiteSemantic.constraint_pad_input_count)
        self.specific_constraints[Op.Pad].append(TFLiteSemantic.constraint_pad_constant)

        # HardSwish specific checks:
        self.specific_constraints[Op.HardSwish].append(TFLiteSemantic.constraint_input_8bit)
        self.specific_constraints[Op.HardSwish].append(TFLiteSemantic.constraint_matching_in_out_types)
        # Mean specific checks:
        self.specific_constraints[Op.Mean].append(TFLiteSemantic.constraint_input_8bit)
        self.specific_constraints[Op.Mean].append(TFLiteSemantic.constraint_mean_input_dims)
        self.specific_constraints[Op.Mean].append(TFLiteSemantic.constraint_mean_axis)

    def is_operator_semantic_valid(self, op):
        ext_type = optype_to_builtintype(op.type)

        if op.type in (Op.Placeholder, Op.SubgraphInput, Op.Const):
            return True

        for constraint in self.generic_constraints + self.specific_constraints[op.type]:
            valid, extra = constraint(op)
            if not valid:
                print(
                    f"Warning: Unsupported TensorFlow Lite semantics for {ext_type} '{op.name}'. Placing on CPU instead"
                )
                print(f" - {constraint.__doc__}")
                if extra:
                    print(f"   {extra}")
                return False

        return True

    @staticmethod
    def constraint_none_const_tensors(op):
        "Constant tensors should not have NoneType-values"
        valid = True
        extra = ""
        for tens in filter(None, op.inputs):
            if len(tens.ops) > 0 and tens.ops[0].type == Op.Const and tens.values is None:
                valid = False
                extra = str(tens.name)
        return valid, f"Unexpected None value for constant tensor: {extra}"

    @staticmethod
    def constraint_tens_no_dynamic(op):
        "Input(s) and Output tensors must not be dynamic"
        valid = True
        extra = []
        tensors = [tens for tens in op.inputs + op.outputs if tens]
        for tens in tensors:
            if (tens.shape == []) and (tens.values is None):
                valid = False
                extra.append(tens.name)
        extra = ", ".join(extra)
        return valid, f"Op has dynamic tensor(s): {extra}"

    @staticmethod
    def constraint_tens_defined_shape(op):
        "Input(s) and Output tensors must have a defined shape"
        valid = True
        extra = []
        tensors = [tens for tens in op.inputs + op.outputs if tens]
        for tens in tensors:
            if not tens.has_fully_defined_shape():
                valid = False
                extra.append(f"Tensor '{tens.name}' has shape: {tens.shape}")
        return valid, ", ".join(extra)

    @staticmethod
    def constraint_tens_output_scalar(op):
        "Output tensors cannot be scalar"
        ofm = op.ofm
        valid = ofm.shape != []
        return valid, f"Output Tensor '{ofm.name}' is scalar"

    @classmethod
    @docstring_format_args([_optype_formatter(shapeless_input_ops)])
    def constraint_tens_input_scalar(cls, op):
        "Scalar Input tensors are only valid for op type: {}"
        valid = True
        extra = []
        tensors = [tens for tens in op.inputs if tens]
        for tens in tensors:
            if (tens.shape == []) and (op.type not in cls.shapeless_input_ops):
                valid = False
                extra.append(tens.name)
        extra = ", ".join(extra)
        return valid, f"Op has scalar input tensor(s): {extra}"

    @staticmethod
    def constraint_tens_shape_size(op):
        "Input(s) and Output tensors must not be greater than 4D"
        valid = True
        extra = []
        tensors = [tens for tens in op.inputs + op.outputs if tens]
        for tens in tensors:
            if len(tens.shape) > 4:
                valid = False
                extra.append(f"Tensor '{tens.name}' has shape: {tens.shape}")
        return valid, ", ".join(extra)

    @staticmethod
    def constraint_tens_quant_none_check(op):
        "Input(s), Output and Weight tensors must have quantization parameters"
        valid = True
        extra = []
        tensors = [tens for tens in op.get_ifm_ifm2_weights_ofm() if tens]
        for tens in tensors:
            if tens.quantization is None:
                valid = False
                extra.append(tens.name)
        extra = ", ".join(extra)
        return valid, f"Op has tensors with missing quantization parameters: {extra}"

    @staticmethod
    def constraint_tens_quant_scale(op):
        "Input(s), Output and Weight tensors with quantization scales must be finite"
        valid = True
        extra = []
        tensors = [tens for tens in op.get_ifm_ifm2_weights_ofm() if tens]
        for tens in tensors:
            if (tens.quantization.scale_f32 is not None) and np.isinf(tens.quantization.scale_f32).any():
                valid = False
                extra.append(f"Tensor '{tens.name}' has quantization scale: {tens.quantization.scale_f32}")
        return valid, ", ".join(extra)

    @staticmethod
    def constraint_fc_output_2d(op):
        "The output tensor(s) must have 2D shape"
        valid = True
        extra = []
        for tens in op.outputs:
            if len(tens.shape) != 2:
                valid = False
                extra.append(f"Tensor '{tens.name}' is {len(tens.shape)}D")
        return valid, ", ".join(extra)

    @staticmethod
    def constraint_stride_type(op):
        "Stride values for both width and height must be integer types"
        w, h = op.get_kernel_stride()
        valid = is_integer(w) and is_integer(h)
        return valid, f"Op has stride WxH as: {repr(w)}x{repr(h)}"

    @staticmethod
    def constraint_dilation_type(op):
        "Dilation factor values for both width and height must be integer types"
        w, h = op.get_kernel_dilation()
        valid = is_integer(w) and is_integer(h)
        return valid, f"Op has dilation factor WxH as: {repr(w)}x{repr(h)}"

    @staticmethod
    def constraint_quant_scale_inf(op):
        "Input and Output tensors must have quantization scales that fit within float32 precision"
        if op.ofm is not None and op.ofm.is_quantized():
            ofm_scale = op.ofm.quantization.scale_f32
            if np.any(ofm_scale < np.finfo(np.float32).tiny):
                return (
                    False,
                    f"The quantization scale of the output tensor is {ofm_scale}, "
                    + f"minimum supported is: {np.finfo(np.float32).tiny}",
                )
            if op.ifm is not None and op.ifm.is_quantized():
                ifm_scale = op.ifm.quantization.scale_f32
                if np.any(np.isinf(ifm_scale / ofm_scale)):
                    return (
                        False,
                        f"IFM scale divided by OFM scale is infinite, ifm_scale={ifm_scale} ofm_scale={ofm_scale}",
                    )
        return True, "Op's quantization is ok"

    @staticmethod
    def constraint_matching_in_out_types(op):
        "IFM and OFM data types must match"
        ifm_dtype = op.ifm.dtype
        ofm_dtype = op.ofm.dtype
        valid = ifm_dtype == ofm_dtype
        return valid, f"Op has ifm_dtype={ifm_dtype} and ofm_dtype={ofm_dtype}"

    @staticmethod
    def constraint_beta_value_range(op):
        "Beta value needs to be positive"
        beta = op.attrs.get("beta", 1.0)
        valid = beta >= 0
        return valid, f"Op has beta={beta}"

    @staticmethod
    def constraint_filter_type(op):
        "Kernel filter values for both width and height must be integer types"
        w = op.kernel.width
        h = op.kernel.height
        valid = is_integer(w) and is_integer(h)
        return valid, f"Op has kernel filter WxH as: {repr(w)}x{repr(h)}"

    @staticmethod
    def constraint_matching_shapes(op):
        "IFM and OFM shapes must match"
        ifm_shape = op.ifm.shape
        ofm_shape = op.ofm.shape
        valid = ifm_shape == ofm_shape
        return valid, f"Op has ifm_shape={ifm_shape} and ofm_shape={ofm_shape}"

    @staticmethod
    def constraint_splitv_inferred(op):
        "Only one size is allowed to be inferred"
        sizes = op.inputs[1].values
        valid = np.count_nonzero(sizes == -1) <= 1
        return valid, f"Op has multiple inferred sizes (-1): {sizes}"

    @staticmethod
    def constraint_axis_exists(op):
        "Axis attribute must exist"
        axis = op.attrs.get("axis")
        valid = axis is not None
        return valid, f"Op has axis={axis}"

    @staticmethod
    def constraint_axis_valid(op):
        "Axis attribute must be in the range [0, <ofm_dimensions>)"
        dims = len(op.ofm.shape)
        axis = op.attrs["axis"]
        axis += dims if axis < 0 else 0
        valid = 0 <= axis < dims
        return valid, f"Op has ofm_dimensions={dims} and axis attribute is: {axis}"

    @staticmethod
    def constraint_matching_dimensionality(op):
        "All Input dimensionalities must match OFM dimensionality"
        valid = True
        extra = []
        ofm_dim = len(op.ofm.shape)
        tensors = [tens for tens in op.inputs if tens]
        for tens in tensors:
            dim = len(tens.shape)
            if dim != ofm_dim:
                valid = False
                extra.append(f"Tensor '{tens.name}' has dimension: {dim}")
        extra = ", ".join(extra)
        return valid, f"Op has ofm_dimension={ofm_dim} and the list of mismatching inputs are: {extra}"

    @staticmethod
    def constraint_valid_dimensions(op):
        "All Input dimensions must match OFM dimension in all axes except the one defined by the axis attribute"
        valid = True
        extra = []
        ofm_shape = op.ofm.shape
        ofm_dim = len(ofm_shape)
        axis = op.attrs["axis"]
        axis += ofm_dim if axis < 0 else 0
        tensors = [tens for tens in op.inputs if tens]
        for tens in tensors:
            if any(tens.shape[dim] != ofm_shape[dim] for dim in range(ofm_dim) if dim != axis):
                valid = False
                extra.append(f"Tensor '{tens.name}' has shape: {tens.shape}")
        extra = ", ".join(extra)
        return valid, f"Op has axis={axis}, ofm_shape={ofm_shape} and the list of mismatching inputs are: {extra}"

    @staticmethod
    def constraint_stridedslice_input_count(op):
        "Exactly 4 Input tensors are required"
        inputs = len(op.inputs)
        valid = inputs == 4
        return valid, f"Op has {inputs} inputs"

    @staticmethod
    def constraint_pad_input_count(op):
        "Number of input tensors must be exactly 2"
        inputs = len(op.inputs)
        valid = inputs == 2
        return valid, f"Op has {inputs} inputs"

    @staticmethod
    def constraint_pad_constant(op):
        "The padding tensor must be constant"
        pad_tensor = op.inputs[1].values
        valid = pad_tensor is not None
        return valid, f"Op has non-constant padding tensor: {op.inputs[1].values}"

    @staticmethod
    def constraint_stridedslice_inputs_const(op):
        "Begin, End and Stride Input tensors must be constant"
        valid = True
        extra = []
        _, begin, end, strides = op.inputs
        if begin.values is None:
            valid = False
            extra.append(f"Begin tensor '{begin.name}'")
        if end.values is None:
            valid = False
            extra.append(f"End tensor '{end.name}'")
        if strides.values is None:
            valid = False
            extra.append(f"Stride tensor '{strides.name}'")
        extra = ", ".join(extra)
        return valid, f"Op has non-constant tensors: {extra}"

    @staticmethod
    def constraint_ellipsis_mask(op):
        "ellipsis_mask must be 0"
        ellipsis = op.attrs["ellipsis_mask"]
        valid = ellipsis == 0
        return valid, f"Op has ellipsis mask as: {ellipsis}"

    @staticmethod
    def constraint_axis_masks(op):
        "new_axis_mask and shrink_axis_mask cannot both be set"
        new_axis = op.attrs["new_axis_mask"]
        shrink_axis = op.attrs["shrink_axis_mask"]
        valid = (new_axis == 0) or (shrink_axis == 0)
        return valid, f"Op has new_axis_mask={new_axis} and shrink_axis_mask={shrink_axis}"

    @staticmethod
    def constraint_slice_ranges(op):
        "Slice 'end' values must be greater than 'begin' values"
        ifm, begin, end, _ = op.inputs
        # Calculate offset begin/end
        offset_begin = get_slice_offsets(ifm.shape, begin, op.attrs["begin_mask"], is_begin=True)
        offset_end = get_slice_offsets(ifm.shape, end, op.attrs["end_mask"], is_begin=False)
        # Check "end - begin" doesn't result in any zero or negative elements
        valid = all((e - b) > 0 for b, e in zip(offset_begin, offset_end))
        return valid, f"Op has begin_values={begin.values} and end_values={end.values}"

    @staticmethod
    def constraint_matching_inputs_types(op):
        "Both Input data types must match"
        ifm_dtype = op.ifm.dtype
        ifm2_dtype = op.ifm2.dtype
        valid = ifm_dtype == ifm2_dtype
        return valid, f"Op has ifm_dtype={ifm_dtype} and ifm2_dtype={ifm2_dtype}"

    @staticmethod
    def constraint_matching_signed(op):
        "For IFM that are signed, OFM must also be signed"
        valid = True
        ifm_dtype = op.ifm.dtype
        ofm_dtype = op.ofm.dtype
        if ifm_dtype.type & BaseType.Signed:
            valid = bool(ofm_dtype.type & BaseType.Signed)
        return valid, f"Op has ifm_dtype={ifm_dtype} and ofm_dtype={ofm_dtype}"

    @staticmethod
    def constraint_unsigned_valid(op):
        "For IFM that are unsigned, OFM must either be the same type or int32"
        valid = True
        ifm_dtype = op.ifm.dtype
        ofm_dtype = op.ofm.dtype
        if ifm_dtype.type & BaseType.Unsigned:
            valid = (ifm_dtype == ofm_dtype) or (ofm_dtype == DataType.int32)
        return valid, f"Op has ifm_dtype={ifm_dtype} and ofm_dtype={ofm_dtype}"

    @staticmethod
    def constraint_input_8bit(op):
        "IFM must be int8 or uint8"
        ifm_dtype = op.ifm.dtype
        valid = (ifm_dtype == DataType.int8) or (ifm_dtype == DataType.uint8)
        return valid, f"Op has ifm_dtype={ifm_dtype}"

    @staticmethod
    def constraint_matching_either_shapes(op):
        "At least one Input's shape must match the OFM's shape"
        ifm_shape = op.ifm.shape
        ifm2_shape = op.ifm2.shape if op.ifm2 else None
        ofm_shape = op.ofm.shape
        valid = (ifm_shape == ofm_shape) or (ifm2_shape == ofm_shape)
        return valid, f"Op has ifm_shape={ifm_shape}, ifm2_shape={ifm2_shape} and ofm_shape={ofm_shape}"

    @staticmethod
    def constraint_alpha_valid(op):
        "Alpha must not be negative"
        alpha = op.attrs["alpha"]
        valid = alpha >= 0
        return valid, f"Op has alpha={alpha}"

    @staticmethod
    def constraint_keep_dim_ifm_ofm(op):
        "The IFM and OFM must have the same number of dimensions if keep_num_dims is set to true"
        valid = True
        if op.attrs.get("keep_num_dims"):
            valid = len(op.ifm.shape) == len(op.ofm.shape)
        return valid, f"Op has ifm shape={op.ifm.shape} and ofm shape={op.ofm.shape}"

    @staticmethod
    def constraint_mean_input_dims(op):
        "Input tensor must be at least 2D"
        dims = len(op.inputs[0].shape)
        return 2 <= dims <= 4, f"Input is {dims}D"

    @staticmethod
    def constraint_mean_axis(op):
        "Axis indices must correspond to height and width axes"
        dims = len(op.inputs[0].shape)
        axis = int(op.inputs[1].values) if op.inputs[1].shape == [] else list(op.inputs[1].values)
        if dims == 2 or dims == 3:
            valid = axis in (0, 1, [0], [1], [0, 1], [1, 0])
        elif dims == 4:
            valid = axis in (1, 2, [1], [2], [1, 2], [2, 1])
        return valid, f"Axis is {axis}"

    @staticmethod
    def constraint_matching_in_out_quant(op):
        "Input and output quantisation must match."
        if not check_quantized_tens_scaling_equal(op.ifm, op.ofm):
            return False, "IFM and OFM quantisation parameters are not equal."
        return True, "IFM and OFM quantisation parameters matches."


def tflite_semantic_checker(nng):
    semantic_checker = TFLiteSemantic()
    for sg in nng.subgraphs:
        for op in sg.get_all_ops():
            op.run_on_npu = semantic_checker.is_operator_semantic_valid(op)
    return nng
