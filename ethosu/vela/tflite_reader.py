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
# Functions used to read from a TensorFlow Lite format file.
import os.path

import numpy as np

from .errors import InputFileError
from .errors import TensorError
from .nn_graph import Graph
from .nn_graph import Subgraph
from .operation import Operation
from .tensor import QuantizationParameters
from .tensor import Tensor
from .tflite.BuiltinOperator import BuiltinOperator
from .tflite.Model import Model
from .tflite_mapping import builtin_operator_map
from .tflite_mapping import DataType
from .tflite_mapping import datatype_map
from .tflite_mapping import datatype_map_numpy


def decode_str(s):
    if s is None:
        return ""
    return s.decode("utf-8")


def clone_and_reshape_tensor(src_tens, reorder):

    tens = src_tens.clone("_reshape")
    tens.shape = [src_tens.shape[idx] for idx in reorder]
    tens.bandwidth_shape = tens.shape
    tens.storage_shape = tens.shape

    if tens.values is not None:
        tens.values = tens.values.transpose(reorder)

    if tens.quant_values is not None:
        tens.quant_values = tens.quant_values.transpose(reorder)

    op = Operation("Const", tens.name)
    op.set_output_tensor(tens)
    return tens


class TFLiteSubgraph:
    def __init__(self, graph, subgraph):
        self.graph = graph
        self.name = decode_str(subgraph.Name())

        self.tensors = []
        for idx in range(subgraph.TensorsLength()):
            self.tensors.append(self.parse_tensor(subgraph.Tensors(idx)))

        for idx in range(subgraph.OperatorsLength()):
            self.parse_operator(idx, subgraph.Operators(idx))

        self.outputs = self.get_tensors_from_indices_remove_duplicates(subgraph.OutputsAsNumpy(), "output")
        self.inputs = self.get_tensors_from_indices_remove_duplicates(subgraph.InputsAsNumpy(), "input")

        # Fix up tensors without operations. Generate either Placeholder or Constant ops
        for tens in self.inputs:
            if tens.ops != []:
                TensorError(tens, "This subgraph input tensor has unexpected driving operators.")

            op = Operation("Placeholder", tens.name)
            op.set_output_tensor(tens)

        for tens in self.tensors:
            if not tens.ops:
                op = Operation("Const", tens.name)
                op.set_output_tensor(tens)

    def get_tensors_from_indices_remove_duplicates(self, indices, warning_str):
        tensors = []
        for idx in indices:
            tensor = self.tensors[idx]
            if tensor not in tensors:
                tensors.append(tensor)
            else:
                print(
                    "Warning: Subgraph {0} tensor ({1}) with idx = {2} already seen. Removing the duplicate.".format(
                        warning_str, tensor, idx
                    )
                )

        return tensors

    def parse_tensor(self, tens_data):
        np_shape = tens_data.ShapeAsNumpy()
        shape = list(np_shape) if type(np_shape) is np.ndarray else []
        name = decode_str(tens_data.Name())
        dtype = datatype_map[tens_data.Type()]
        tens = Tensor(shape, dtype, name)
        quant = tens_data.Quantization()

        tens.quantization = QuantizationParameters()
        if quant is not None:
            tens.quantization.min = self.len1_array_to_scalar(quant.MinAsNumpy())
            tens.quantization.max = self.len1_array_to_scalar(quant.MaxAsNumpy())
            tens.quantization.scale_f32 = self.len1_array_to_scalar(quant.ScaleAsNumpy())
            tens.quantization.zero_point = self.len1_array_to_scalar(quant.ZeroPointAsNumpy())

        if dtype == DataType.uint8:
            tens.quantization.quant_min = 0
            tens.quantization.quant_max = (1 << dtype.bits) - 1
        elif dtype in set((DataType.int8, DataType.int16, DataType.int32, DataType.int64)):
            tens.quantization.quant_min = -(1 << (dtype.bits - 1))
            tens.quantization.quant_max = (1 << (dtype.bits - 1)) - 1

        if tens.quantization.scale_f32 is None and tens.quantization.zero_point is None:
            tens.quantization = None

        tens.values = None
        buf = self.graph.buffers[tens_data.Buffer()]
        if buf is not None:
            tens.values = np.array(buf.view(datatype_map_numpy[tens_data.Type()]).reshape(shape))
            if tens.quantization is not None:
                tens.quant_values = tens.values
                tens.values = tens.quantization.dequantize(tens.quant_values)
        return tens

    def parse_operator(self, op_index, op_data):
        op_type, opt_serializer = self.graph.operator_codes[op_data.OpcodeIndex()]
        inputs = [self.tensors[idx] if idx != -1 else None for idx in op_data.InputsAsNumpy()]
        outputs = [self.tensors[idx] if idx != -1 else None for idx in op_data.OutputsAsNumpy()]
        name = "unknown_op_name"
        if len(outputs):
            name = outputs[0].name
        op = Operation(op_type, name)
        op.op_index = op_index
        op.inputs = inputs
        op.outputs = outputs
        for out in op.outputs:
            out.ops = [op]

        activation_function_to_split_out = None

        if op_type.startswith("DepthwiseConv2d") or op_type.startswith("Conv2D"):
            inputs[1] = clone_and_reshape_tensor(inputs[1], (1, 2, 3, 0))
            if len(inputs) < 3 or (len(inputs) < 4 and "Backprop" in op_type):
                # No Bias tensor
                inputs.append(None)
            if inputs[-1]:
                inputs[-1] = clone_and_reshape_tensor(inputs[-1], (0,))

        if op_type.startswith("FullyConnected"):
            inputs[1] = clone_and_reshape_tensor(inputs[1], (1, 0))
            if len(inputs) < 3:
                # No Bias tensor
                inputs.append(None)
            if inputs[-1]:
                inputs[-1] = clone_and_reshape_tensor(inputs[-1], (0,))

        if opt_serializer is not None:
            op.attrs = opt_serializer.deserialize(op_data)

            if op_type == "Reshape" and "new_shape" not in op.attrs:
                # Reshape should have an attrib "new_shape" but if it is missing, add it based on the output shape
                op.attrs["new_shape"] = outputs[0].shape

            if "stride_w" in op.attrs:
                op.attrs["strides"] = (1, op.attrs["stride_h"], op.attrs["stride_w"], 1)
            if "filter_width" in op.attrs:
                op.attrs["ksize"] = (1, op.attrs["filter_height"], op.attrs["filter_width"], 1)
            if "dilation_w_factor" in op.attrs:
                op.attrs["dilation"] = (1, op.attrs["dilation_h_factor"], op.attrs["dilation_w_factor"], 1)
            if "depth_multiplier" in op.attrs:
                op.attrs["channel_multiplier"] = op.attrs["depth_multiplier"]

            if "fused_activation_function" in op.attrs:
                if op_type in set(("ConcatTFLite",)):
                    act = op.attrs["fused_activation_function"]
                    del op.attrs["fused_activation_function"]
                    if act is not None:
                        activation_function_to_split_out = act

        if activation_function_to_split_out is not None:
            act_op = Operation(activation_function_to_split_out, name + activation_function_to_split_out)
            out_tens = op.outputs[0]
            intermediate_tens = out_tens.clone("_act_intermediate")
            act_op.set_output_tensor(out_tens)
            intermediate_tens.ops = [op]
            op.outputs[0] = intermediate_tens
            act_op.inputs = [intermediate_tens]

    @staticmethod
    def len1_array_to_scalar(arr):
        # The following flatbuffer quantisation fields all return a scalar value of 0 if they are not definied in
        # the input buffer. This is represented in Vela by using None.
        # Otherwise, the fields returned are a single or multi-element array. In which case, single element arrays
        # are converted to scalars
        if isinstance(arr, int) and arr == 0:
            return None
        if len(arr) == 1:
            return arr[0]
        return arr


class TFLiteGraph:
    def __init__(
        self, filename, batch_size=1, feed_dict={}, output_node_names=[], initialisation_nodes=[],
    ):

        self.op_times = {}
        if batch_size is None:
            batch_size = 1
        self.batch_size = batch_size
        self.name = os.path.splitext(os.path.basename(filename))[0]
        self.initialisation_nodes = initialisation_nodes

        with open(filename, "rb") as f:
            buf = bytearray(f.read())

        model = Model.GetRootAsModel(buf, 0)

        self.buffers = []
        for idx in range(model.BuffersLength()):
            self.buffers.append(self.parse_buffer(model.Buffers(idx)))

        self.operator_codes = []
        for idx in range(model.OperatorCodesLength()):
            self.operator_codes.append(self.parse_operator_code(model.OperatorCodes(idx)))

        self.subgraphs = []
        for idx in range(model.SubgraphsLength()):
            self.subgraphs.append(TFLiteSubgraph(self, model.Subgraphs(idx)))

        self.nng = Graph(self.name, self.batch_size)
        for tflite_sg in self.subgraphs:
            sg = Subgraph(tflite_sg.name)
            sg.original_inputs = tflite_sg.inputs  # Preserve the original input order
            sg.output_tensors = tflite_sg.outputs
            self.nng.subgraphs.append(sg)

        # Preserve the original metadata
        for idx in range(model.MetadataLength()):
            meta = model.Metadata(idx)
            name = meta.Name()
            if name is not None:
                buf_data = self.buffers[meta.Buffer()]
                self.nng.metadata.append((name, buf_data))

    def parse_buffer(self, buf_data):
        if buf_data.DataLength() == 0:
            return None
        data = buf_data.DataAsNumpy()
        return data

    def parse_operator_code(self, code):
        c = code.BuiltinCode()
        if c not in builtin_operator_map:
            msg = "The input file contains operator code {} which is currently not supported".format(c)
            raise InputFileError(self.name, msg)
        op_type, ser = builtin_operator_map[c]
        if c == BuiltinOperator.CUSTOM:
            op_type += decode_str(code.CustomCode())
        return op_type, ser


def read_tflite(
    filename, batch_size=1, feed_dict={}, output_node_names=[], initialisation_nodes=[],
):
    tflite_graph = TFLiteGraph(filename, batch_size, feed_dict, output_node_names, initialisation_nodes)
    nng = tflite_graph.nng
    nng.refresh_after_modification()
    return nng
