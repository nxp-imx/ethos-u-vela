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
# Internal representation of a Neural Network Operation.
import enum


class NpuBlockType(enum.Enum):
    Default = 0
    ConvolutionMxN = 1
    VectorProduct = 2
    Pooling = 3
    ConvolutionDepthWise = 4
    ElementWise = 5
    ReduceSum = 6


class Operation:
    """Class representing a Neural Network operation. Has a name, a type,
input and output tensors, as well as an attribute dictionary."""

    __slots__ = (
        "type",
        "name",
        "op_index",
        "attrs",
        "inputs",
        "outputs",
        "flops",
        "scheduled_pass",
        "run_on_npu",
        "activation_lut",
    )

    def __init__(self, op_type, name):
        self.type = op_type
        self.name = name
        self.attrs = {}
        self.inputs = []
        self.outputs = []
        self.flops = 0
        self.run_on_npu = True
        self.scheduled_pass = None
        self.op_index = None  # input network operator index
        self.activation_lut = None

    def clone(self, suffix="_clone"):
        res = Operation(self.type, self.name + suffix)

        res.attrs = dict(self.attrs)
        res.inputs = list(self.inputs)
        res.outputs = list(self.outputs)
        res.flops = self.flops
        res.scheduled_pass = self.scheduled_pass
        res.op_index = None  # not relevant as not part of input network

        return res

    def __str__(self):
        return "<nng.Operation '%s' type=%s>" % (self.name, self.type)

    __repr__ = __str__

    def get_ifm_ifm2_weight_bias_ofm_indices(self):
        ifm_idx = -1
        ifm2_idx = -1
        weight_idx = -1
        bias_idx = -1
        ofm_idx = -1
        npu_block_type = self.attrs.get("npu_block_type", NpuBlockType.Default)
        if npu_block_type in (NpuBlockType.ConvolutionMxN, NpuBlockType.ConvolutionDepthWise):
            ifm_idx = 0
            weight_idx = 1
            ofm_idx = 0

            if self.type in ("Conv2DBiasAct", "DepthwiseConv2dBiasAct", "TransposeConvAct"):
                if len(self.inputs) >= 3:
                    bias_idx = 2

            elif self.type == "Conv2DBackpropInputSwitchedBias":
                bias_idx = 3

        elif npu_block_type in (NpuBlockType.Pooling, NpuBlockType.ReduceSum):
            ifm_idx = 0
            ofm_idx = 0
        elif npu_block_type == NpuBlockType.VectorProduct:
            ifm_idx = 0
            weight_idx = 1
            ofm_idx = 0

            if self.type == "FullyConnectedAct":
                if len(self.inputs) >= 3:
                    bias_idx = 2

            if self.type == "BlockLSTM":
                ifm_idx = 3
                weight_idx = 4
                ofm_idx = 6

        elif npu_block_type == NpuBlockType.ElementWise:
            ifm_idx = 0
            ifm2_idx = 1
            ofm_idx = 0

            # LeakyRelu, Abs and CLZ have a single IFM
            if self.type in ("LeakyRelu", "Abs", "CLZ"):
                ifm2_idx = -1

        elif self.type == "Conv2DBackpropInput":
            ifm_idx = 2
            weight_idx = 1
            ofm_idx = 0

        elif self.type in ("Squeeze", "Reshape", "QuantizedReshape", "ExpandDims"):
            ifm_idx = 0
            ofm_idx = 0

        elif self.is_split_op():
            ifm_idx = 0
            ofm_idx = 0
            if self.type == "Split":
                ifm_idx = 1

        elif self.is_concat_op():
            ifms, _ = self.get_concat_inputs_axis()
            ifm_idx = self.inputs.index(ifms[0])
            if len(ifms) > 1:
                ifm2_idx = self.inputs.index(ifms[1])
            ofm_idx = 0

        return ifm_idx, ifm2_idx, weight_idx, bias_idx, ofm_idx

    def get_ifm_ifm2_weights_ofm(self):
        ifm_tensor = None
        ifm2_tensor = None
        weight_tensor = None
        ofm_tensor = None

        ifm_idx, ifm2_idx, weight_idx, _, ofm_idx = self.get_ifm_ifm2_weight_bias_ofm_indices()
        if ifm_idx != -1:
            ifm_tensor = self.inputs[ifm_idx]
        if ifm2_idx != -1:
            ifm2_tensor = self.inputs[ifm2_idx]
        if weight_idx != -1:
            weight_tensor = self.inputs[weight_idx]
        if ofm_idx != -1:
            ofm_tensor = self.outputs[ofm_idx]

        return ifm_tensor, ifm2_tensor, weight_tensor, ofm_tensor

    def get_ifm_weights_biases_ofm(self):
        ifm_tensor = None
        weight_tensor = None
        bias_tensor = None
        ofm_tensor = None

        ifm_idx, _, weight_idx, bias_idx, ofm_idx = self.get_ifm_ifm2_weight_bias_ofm_indices()
        if ifm_idx != -1:
            ifm_tensor = self.inputs[ifm_idx]
        if weight_idx != -1:
            weight_tensor = self.inputs[weight_idx]
        if bias_idx != -1:
            bias_tensor = self.inputs[bias_idx]
        if ofm_idx != -1:
            ofm_tensor = self.outputs[ofm_idx]

        return ifm_tensor, weight_tensor, bias_tensor, ofm_tensor

    def get_ifm_ifm2_weights_biases_ofm(self):
        ifm_tensor = None
        ifm2_tensor = None
        weight_tensor = None
        bias_tensor = None
        ofm_tensor = None

        ifm_idx, ifm2_idx, weight_idx, bias_idx, ofm_idx = self.get_ifm_ifm2_weight_bias_ofm_indices()
        if ifm_idx != -1:
            ifm_tensor = self.inputs[ifm_idx]
        if ifm2_idx != -1:
            ifm2_tensor = self.inputs[ifm2_idx]
        if weight_idx != -1:
            weight_tensor = self.inputs[weight_idx]
        if bias_idx != -1:
            bias_tensor = self.inputs[bias_idx]
        if ofm_idx != -1:
            ofm_tensor = self.outputs[ofm_idx]

        return ifm_tensor, ifm2_tensor, weight_tensor, bias_tensor, ofm_tensor

    def is_concat_op(self):
        return self.type in ("Concat", "ConcatV2", "QuantizedConcat", "ConcatTFLite", "PackReshaped")

    def get_concat_inputs_axis(self):
        assert self.is_concat_op()

        if self.type == "ConcatV2":
            axis_tensor = self.inputs[-1]
            inputs = self.inputs[:-1]
        elif self.type == "Concat":
            axis_tensor = self.inputs[0]
            inputs = self.inputs[1:]
        elif self.type == "QuantizedConcat":
            axis_tensor = self.inputs[0]
            inputs = self.inputs[1:]
            inputs = inputs[: len(inputs) // 3]  # Skip min/max

        if self.type == "ConcatTFLite":
            inputs = self.inputs
            axis = self.attrs["axis"]
        elif self.type == "PackReshaped":
            # Requires fixup_pack_input to be called before this point
            inputs = self.inputs
            axis = self.attrs["axis"]
            assert len(self.inputs) == self.attrs["values_count"]
        else:
            assert len(axis_tensor.ops) == 1 and axis_tensor.ops[0].type == "Const"
            axis = int(axis_tensor.values)

        return inputs, axis

    def get_dilation_h_w(self):
        _, dilation_h, dilation_w, _ = self.attrs.get("dilation", (1, 1, 1, 1))
        return dilation_h, dilation_w

    def is_split_op(self):
        return self.type in ("Split", "SplitV", "StridedSlice", "Slice", "UnpackReshaped")

    def get_split_inputs_axis(self):
        assert self.is_split_op()

        offset_start = None
        offset_end = None
        axis = None
        if self.type == "Split":
            num_splits = self.attrs.get("num_splits")
            axis_tens = self.inputs[0]
            assert len(axis_tens.ops) == 1 and axis_tens.ops[0].type == "Const"
            axis = int(axis_tens.values)
            input_tens = self.inputs[1]
            outputs = self.outputs
            assert num_splits == len(outputs)

        elif self.type == "SplitV":
            num_splits = self.attrs.get("num_splits")
            input_tens = self.inputs[0]
            size_tens = self.inputs[1]
            assert len(size_tens.ops) == 1 and size_tens.ops[0].type == "Const"
            sizes = size_tens.values
            axis_tens = self.inputs[2]
            assert len(axis_tens.ops) == 1 and axis_tens.ops[0].type == "Const"
            axis = int(axis_tens.values)
            outputs = self.outputs
            assert num_splits == len(outputs)
            assert sum(sizes) == input_tens.shape[axis]

        elif self.type == "Slice":
            input_tens, begin_tens, size_tens = self.inputs
            outputs = self.outputs
            offset_start = [0] * len(input_tens.shape)
            offset_end = [0] * len(input_tens.shape)

            for idx in range(len(begin_tens.values)):
                # Check if the op should slice in dimension idx
                if size_tens.values[idx] != input_tens.shape[idx]:
                    offset_start[idx] = begin_tens.values[idx]
                    offset_end[idx] = size_tens.values[idx] + offset_start[idx]

        elif self.type == "StridedSlice":
            input_tens, begin_tens, end_tens, strides_tens = self.inputs
            outputs = self.outputs
            out_tens = outputs[0]
            offset_start = [0] * len(outputs[0].shape)
            offset_end = [0] * len(outputs[0].shape)

            # Extract masks
            begin_mask = self.attrs["begin_mask"]
            ellipsis_mask = self.attrs["ellipsis_mask"]
            end_mask = self.attrs["end_mask"]
            new_axis_mask = self.attrs["new_axis_mask"]
            shrink_axis_mask = self.attrs["shrink_axis_mask"]

            # shrink_axis_mask/new_axis_mask/ellipsis_mask is not supported by the Operation class but the operation
            # may have the attribute modified and handled in the graph optimization phase.
            assert shrink_axis_mask == new_axis_mask == ellipsis_mask == 0
            assert len(input_tens.shape) == len(out_tens.shape)

            for idx in range(len(input_tens.shape)):
                # Check if slicing is needed in this axis
                if end_tens.values[idx] != input_tens.shape[idx] or (
                    end_tens.values[idx] == input_tens.shape[idx] and begin_tens.values[idx] != 0
                ):
                    # If the i:th bit in begin_mask is set then the value on begin[i] should be ignored
                    if (begin_mask & (1 << idx)) == 0:
                        offset_start[idx] = begin_tens.values[idx]

                    # If the i:th bit in end_mask is set then the value on end[i] should be ignored
                    if (end_mask & (1 << idx)) == 0:
                        offset_end[idx] = end_tens.values[idx]

        elif self.type == "UnpackReshaped":
            # Requires fixup_unpack_output to be called before this point
            input_tens = self.inputs[0]
            outputs = self.outputs
            axis = self.attrs["axis"]
            num_splits = self.attrs["num"]
            # Number of outputs have to equal the value of the dimension to unpack
            assert num_splits == len(outputs) == input_tens.shape[axis]
        else:
            assert False

        return input_tens, outputs, axis, offset_start, offset_end

    def set_activation_lut(self, lut_tensor):
        self.attrs["fused_activation_function"] = "LUT"
        self.activation_lut = lut_tensor
        self.add_input_tensor(lut_tensor)

    def add_input_tensor(self, tens):
        self.inputs.append(tens)
        if self not in tens.consumer_list:
            tens.consumer_list.append(self)

    def set_input_tensor(self, tens, idx):
        tens_to_remove = self.inputs[idx]
        if tens_to_remove in tens.consumer_list:
            tens.consumer_list.remove(tens_to_remove)

        self.inputs[idx] = tens
        if self not in tens.consumer_list:
            tens.consumer_list.append(self)

    def set_output_tensor(self, tens):
        tens.ops = [self]
        self.outputs = [tens]

    def needs_bias(self):
        return self.type in (
            "Conv2DBiasAct",
            "DepthwiseConv2dBiasAct",
            "Conv2DBackpropInputSwitchedBias",
            "FullyConnectedAct",
        )
