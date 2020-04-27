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

from .data_type import BaseType


class SupportedOperators:
    def __init__(self):
        # Categorised lists of supported operators
        self.npu_pre_ops = set(("QuantizedResizeBilinear", "SplitSliceRead"))
        self.convolution_ops = set(("Conv2DBiasAct", "Conv2D", "QuantizedConv2D", "Conv2DBackpropInputSwitched"))
        self.depthwise_convolution_ops = set(
            ("DepthwiseConv2dBiasAct", "DepthwiseConv2dNative", "QuantizedDepthwiseConv2D")
        )
        self.max_pooling_ops = set(("QuantizedMaxPool", "MaxPool", "MaxPoolAct"))
        self.avg_pooling_ops = set(("QuantizedAvgPool", "AvgPool", "AvgPoolAct"))
        self.pooling_ops = self.max_pooling_ops | self.avg_pooling_ops
        self.fc_vector_products = set(("QuantizedMatMul", "MatMul", "FullyConnectedAct"))
        self.mac_main_ops = (
            # convolutions
            self.convolution_ops
            # depth-wise convolutions
            | self.depthwise_convolution_ops
            # pooling
            | self.pooling_ops
            # FC layers
            | self.fc_vector_products
            # RNN/LSTM/GRU
            | set(("BlockLSTM"))
        )
        self.unary_elem_wise_main_ops = set(("LeakyRelu", "Abs"))
        self.binary_elem_wise_main_ops = set(
            (
                # binary element-wise
                "AddAct",
                "MulAct",
                "SubAct",
                "QuantizedAdd",
                "QuantizedSub",
                "QuantizedMul",
                "Mul",
                "Add",
                "Sub",
                "Minimum",
                "Maximum",
            )
        )
        self.elem_wise_main_ops = self.binary_elem_wise_main_ops | self.unary_elem_wise_main_ops
        self.activation_ops = set(
            ("QuantizedRelu", "QuantizedRelu1", "QuantizedRelu6", "Relu", "Relu6", "ReluN1To1", "Sigmoid", "Tanh")
        )
        self.npu_post_ops = (
            # activation functions
            self.activation_ops
            # concatenation write direction
            | set(("ConcatSliceWrite"))
            # bias add and batch norm
            | set(("QuantizedBiasAdd", "Requantize", "QuantizedBatchNorm", "BiasAdd", "FusedBatchNorm"))
        )
        self.split_ops = set(("Split", "StridedSlice", "Slice", "UnpackReshaped", "Unpack"))
        self.concat_ops = set(("Concat", "ConcatV2", "QuantizedConcat", "ConcatTFLite", "PackReshaped", "Pack"))
        self.memory_only_ops = (
            set(("Squeeze", "Reshape", "QuantizedReshape", "ExpandDims")) | self.concat_ops | self.split_ops
        )
        self.supported_fused_activations = set(("Relu", "Relu6", "ReluN1To1", "Tanh", "Sigmoid"))
        self.supported_operators = (
            self.npu_pre_ops | self.mac_main_ops | self.elem_wise_main_ops | self.npu_post_ops | self.memory_only_ops
        )
        # Setup supported operator restriction checkers
        self.supported_operator_restrictions = {}
        self.supported_operator_restrictions.update(
            {op: self.check_convolution_restrictions for op in self.convolution_ops}
        )
        self.supported_operator_restrictions.update(
            {op: self.check_depthwise_convolution_restrictions for op in self.depthwise_convolution_ops}
        )
        self.supported_operator_restrictions.update({op: self.check_pooling_restrictions for op in self.pooling_ops})
        self.supported_operator_restrictions.update(
            {op: self.check_vector_product_restrictions for op in self.fc_vector_products}
        )
        self.supported_operator_restrictions.update(
            {op: self.check_element_wise_restrictions for op in self.elem_wise_main_ops}
        )
        self.supported_operator_restrictions.update(
            {op: self.check_memory_only_restrictions for op in self.memory_only_ops}
        )

    def is_operator_supported(self, op):
        if op.type not in self.supported_operators:
            return False
        if not self.check_generic_restrictions(op):
            return False
        if op.type in self.supported_operator_restrictions:
            return self.supported_operator_restrictions[op.type](op)
        return True

    def check_generic_restrictions(self, op):
        # check fully defined shapes
        for t in op.inputs + op.outputs:
            if not t.has_fully_defined_shape():
                print("Warning:", op, "has inputs/outputs of undefined shape, placing on CPU")
                return False

        # check data type
        tensors = [t for t in op.get_ifm_ifm2_weights_ofm() if t is not None]
        if not tensors:
            tensors = op.inputs
        for t in tensors:
            if not (t.dtype.type & BaseType.Int):
                return False
            if t.element_size() > 2 and op.type != "Requantize":
                return False
            # check size
            if any(dim > 65536 for dim in t.shape):
                return False

        # check fused activations
        if (
            "fused_activation_function" in op.attrs
            and op.attrs["fused_activation_function"] is not None
            and op.attrs["fused_activation_function"] not in self.supported_fused_activations
        ):
            return False
        return True

    def check_convolution_restrictions(self, op):
        # check stride
        if op.attrs["stride_w"] > 2 or op.attrs["stride_h"] > 2:
            return False

        # check dilation
        dilation_w_factor = op.attrs.get("dilation_w_factor", 1)
        dilation_h_factor = op.attrs.get("dilation_h_factor", 1)
        if dilation_w_factor > 2 or dilation_h_factor > 2:
            return False

        # check data type
        ifm_tensor, _, weight_tensor, _ = op.get_ifm_ifm2_weights_ofm()
        if weight_tensor.element_size() > 1:
            return False

        # check kernel size
        dilated_weight_w = weight_tensor.shape[0] + (weight_tensor.shape[0] - 1) * (dilation_w_factor - 1)
        dilated_weight_h = weight_tensor.shape[1] + (weight_tensor.shape[1] - 1) * (dilation_h_factor - 1)
        if (
            dilated_weight_w > 64
            or dilated_weight_h > 64
            or dilated_weight_w * dilated_weight_h * weight_tensor.shape[2] > 127 * 65536
        ):
            return False

        # check batch size
        if ifm_tensor.shape[0] != 1:
            return False
        return True

    def check_depthwise_convolution_restrictions(self, op):
        # check depth
        ifm_tensor, _, _, ofm_tensor = op.get_ifm_ifm2_weights_ofm()
        if op.attrs["depth_multiplier"] > 1 and not (
            (ifm_tensor.shape[3] == 1) and (ofm_tensor.shape[3] == op.attrs["depth_multiplier"])
        ):
            return False
        return self.check_convolution_restrictions(op)

    def check_pooling_restrictions(self, op):
        # check stride
        if op.attrs["stride_w"] > 2 or op.attrs["stride_h"] > 2:
            return False

        # check data type
        ifm_tensor, _, _, ofm_tensor = op.get_ifm_ifm2_weights_ofm()
        if ifm_tensor.dtype != ofm_tensor.dtype:
            return False

        # check batch size
        if ifm_tensor.shape[0] != 1:
            return False

        if op.type in self.avg_pooling_ops:
            # check kernel size
            if op.attrs["padding"] == b"SAME" and (op.attrs["filter_width"] > 8 or op.attrs["filter_height"] > 8):
                return False
            if op.attrs["padding"] == b"VALID" and (op.attrs["filter_width"] > 256 or op.attrs["filter_height"] > 256):
                return False

        if op.type in self.max_pooling_ops:
            # check data type
            if not ifm_tensor.dtype == ofm_tensor.dtype:
                return False
            # check kernel size
            if op.attrs["filter_width"] > 256 or op.attrs["filter_height"] > 256:  # any padding
                return False
        return True

    def check_vector_product_restrictions(self, op):
        # check data type
        ifm_tensor, _, weight_tensor, _ = op.get_ifm_ifm2_weights_ofm()
        if weight_tensor.element_size() > 1:
            return False

        return True

    def check_element_wise_restrictions(self, op):
        # check data type
        ifm_tensor, ifm2_tensor, _, ofm_tensor = op.get_ifm_ifm2_weights_ofm()
        if op.type in ("Minimum", "Maximum") and ifm_tensor.dtype != ofm_tensor.dtype:
            return False

        # check batch size
        if len(ifm_tensor.shape) > 2 and ifm_tensor.shape[0] != 1:
                return False
        if op.type in self.binary_elem_wise_main_ops: # if op type is unary, ifm2_tensor is None
            if len(ifm2_tensor.shape) > 2 and ifm2_tensor.shape[0] != 1:
                return False

        # check scalar size
        if hasattr(ifm_tensor.values, "__len__") and len(ifm_tensor.values) > 1:
            return False
        if op.type in self.binary_elem_wise_main_ops: # same as above
            if hasattr(ifm2_tensor.values, "__len__") and len(ifm2_tensor.values) > 1:
                return False
        return True

    def check_memory_only_restrictions(self, op):
        # check stride size
        if op.type == "StridedSlice":
            if len(op.inputs) > 3 and any(stride != 1 for stride in op.inputs[3].values):
                return False
        return True
