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
# Functions used to read from a TensorFlow Lite format file.
import os.path
import struct
import sys

import numpy as np

from .errors import InputFileError
from .nn_graph import Graph
from .nn_graph import Subgraph
from .operation import create_activation_function
from .operation import Op
from .operation import Operation
from .reader_util import align_tensor_indices_to_nng
from .reader_util import clone_and_reshape_tensor
from .reader_util import decode_str
from .reader_util import fixup_tensors
from .tensor import QuantizationParameters
from .tensor import Tensor
from .tflite.BuiltinOperator import BuiltinOperator
from .tflite.Model import Model
from .tflite_mapping import builtin_operator_map
from .tflite_mapping import DataType
from .tflite_mapping import datatype_map
from .tflite_mapping import datatype_map_numpy


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
        fixup_tensors(self.inputs, self.tensors)

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
        tens_dtype = tens_data.Type()
        dtype = datatype_map[tens_dtype]
        tens = Tensor(shape, dtype, name)
        quant = tens_data.Quantization()
        tens.is_variable = tens_data.IsVariable()

        tens.quantization = QuantizationParameters()
        if quant is not None:
            tens.quantization.min = self.len1_array_to_scalar(quant.MinAsNumpy())
            tens.quantization.max = self.len1_array_to_scalar(quant.MaxAsNumpy())
            tens.quantization.scale_f32 = self.len1_array_to_scalar(quant.ScaleAsNumpy())
            tens.quantization.zero_point = self.len1_array_to_scalar(quant.ZeroPointAsNumpy())
            tens.quantization.quant_dim = quant.QuantizedDimension()

        if dtype == DataType.uint8:
            tens.quantization.quant_min = 0
            tens.quantization.quant_max = (1 << dtype.bits) - 1
        elif dtype in (DataType.int8, DataType.int16, DataType.int32, DataType.int64):
            tens.quantization.quant_min = -(1 << (dtype.bits - 1))
            tens.quantization.quant_max = (1 << (dtype.bits - 1)) - 1

        if tens.quantization.scale_f32 is None and tens.quantization.zero_point is None:
            tens.quantization = None

        tens.values = None
        buf = self.graph.buffers[tens_data.Buffer()]
        if buf is not None:
            np_dtype = datatype_map_numpy[tens_dtype]
            if dtype == DataType.string:
                tens.values = np.array(buf.view(np_dtype))
            else:
                tens.values = np.array(buf.view(np_dtype).reshape(shape))
        return tens

    def parse_operator(self, op_index, op_data):
        op_type, opt_serializer, custom_code, indices = self.graph.operator_codes[op_data.OpcodeIndex()]
        inputs = [self.tensors[idx] if idx != -1 else None for idx in op_data.InputsAsNumpy()]
        outputs = [self.tensors[idx] if idx != -1 else None for idx in op_data.OutputsAsNumpy()]
        intermediates = []
        if op_data.IntermediatesLength():
            intermediates = [self.tensors[idx] if idx != -1 else None for idx in op_data.IntermediatesAsNumpy()]

        name = "unknown_op_name"
        if len(outputs):
            name = outputs[0].name
        inputs = align_tensor_indices_to_nng(op_type, indices, inputs)
        op = Operation(op_type, name)
        op.op_index = op_index
        op.inputs = inputs
        op.outputs = outputs
        op.intermediates = intermediates
        for out in op.outputs:
            out.ops = [op]

        if op.type.is_depthwise_conv2d_op() or op.type.is_conv2d_op() or op.type == Op.FullyConnected:
            if inputs[1].values is not None:
                if op.type == Op.FullyConnected:
                    inputs[1] = clone_and_reshape_tensor(inputs[1], (1, 0), False)
                else:
                    inputs[1] = clone_and_reshape_tensor(inputs[1], (1, 2, 3, 0), False)
            if op.type.needs_bias() and len(inputs) <= op_type.info.indices.biases[0]:
                # No Bias tensor
                inputs.append(None)
            if inputs[-1] and inputs[-1].values is not None:
                # Since bias tensor is used for both bias and scale,
                # a clone with a unique equivalence_id is needed
                inputs[-1] = clone_and_reshape_tensor(inputs[-1], (0,), True)

        if opt_serializer is not None:
            op.attrs = opt_serializer.deserialize(op_data)

            if op_type == Op.Reshape and "new_shape" not in op.attrs:
                # Reshape should have an attrib "new_shape" but if it is missing, add it based on the output shape
                op.attrs["new_shape"] = outputs[0].shape

            if op_type == Op.Cast:
                # Cast op should have "in/out_data_type" attribs add if missing
                if "in_data_type" not in op.attrs:
                    op.attrs["in_data_type"] = inputs[0].dtype
                if "out_data_type" not in op.attrs:
                    op.attrs["out_data_type"] = outputs[0].dtype

            if "stride_w" in op.attrs:
                op.attrs["strides"] = (1, op.attrs["stride_h"], op.attrs["stride_w"], 1)
            if "filter_width" in op.attrs:
                op.attrs["ksize"] = (1, op.attrs["filter_height"], op.attrs["filter_width"], 1)
            if "dilation_w_factor" in op.attrs:
                op.attrs["dilation"] = (1, op.attrs["dilation_h_factor"], op.attrs["dilation_w_factor"], 1)
            if "depth_multiplier" in op.attrs:
                op.attrs["channel_multiplier"] = op.attrs["depth_multiplier"]

            if op_type == Op.DepthwiseConv2DBias and op.attrs["depth_multiplier"] == 0:
                # The depth multiplier is implicit and is calculated as weight channels / ifm channels
                # Note however that the weights have been reshaped above.
                # The original value is cached above in channel_multiplier
                op.attrs["depth_multiplier"] = op.weights.shape[2] // op.ifm.shape[-1]

            faf = op.attrs.pop("fused_activation_function", None)
            if faf is not None:
                op.activation = create_activation_function(faf)
            if custom_code is not None:
                op.attrs["custom_code"] = custom_code

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
    def __init__(self, filename, batch_size, feed_dict, output_node_names, initialisation_nodes):

        self.op_times = {}
        if batch_size is None:
            batch_size = 1
        self.batch_size = batch_size
        self.name = os.path.splitext(os.path.basename(filename))[0]
        self.initialisation_nodes = initialisation_nodes

        with open(filename, "rb") as f:
            buf = bytearray(f.read())

        try:
            parsing_step = "parsing root"
            model = Model.GetRootAsModel(buf, 0)

            parsing_step = "parsing buffers length"
            self.buffers = []
            for idx in range(model.BuffersLength()):
                parsing_step = f"parsing buffer {idx}"
                self.buffers.append(self.parse_buffer(model.Buffers(idx)))

            parsing_step = "parsing operator codes length"
            self.operator_codes = []
            for idx in range(model.OperatorCodesLength()):
                parsing_step = f"parsing operator code {idx}"
                self.operator_codes.append(self.parse_operator_code(model.OperatorCodes(idx)))

            parsing_step = "parsing subgraphs length"
            self.subgraphs = []
            for idx in range(model.SubgraphsLength()):
                parsing_step = f"parsing subgraph {idx}"
                self.subgraphs.append(TFLiteSubgraph(self, model.Subgraphs(idx)))

            self.nng = Graph(self.name, self.batch_size)
            for tflite_sg in self.subgraphs:
                sg = Subgraph(tflite_sg.name)
                sg.original_inputs = tflite_sg.inputs  # Preserve the original input order
                sg.output_tensors = tflite_sg.outputs
                self.nng.subgraphs.append(sg)

            parsing_step = "parsing metadata length"
            # Preserve the original metadata
            for idx in range(model.MetadataLength()):
                parsing_step = f"parsing metadata {idx}"
                meta = model.Metadata(idx)
                parsing_step = f"parsing metadata name of metadata {idx}"
                name = meta.Name()
                if name is not None:
                    parsing_step = f"parsing metadata {idx} ({name})"
                    buf_data = self.buffers[meta.Buffer()]
                    self.nng.metadata.append((name, buf_data))
        except (struct.error, TypeError, RuntimeError) as e:
            print(f'Error: Invalid tflite file. Got "{e}" while {parsing_step}.')
            sys.exit(1)

    def parse_buffer(self, buf_data):
        if buf_data.DataLength() == 0:
            return None
        data = buf_data.DataAsNumpy()
        return data

    def parse_operator_code(self, code):
        c = code.BuiltinCode()
        if c == 0:
            c = code.DeprecatedBuiltinCode()
        if c not in builtin_operator_map:
            raise InputFileError(
                self.name, f"The input file contains operator code '{c}' which is currently not supported"
            )
        op_type, ser, indices = builtin_operator_map[c]
        custom_code = None
        if c == BuiltinOperator.CUSTOM:
            custom_code = decode_str(code.CustomCode())
        return op_type, ser, custom_code, indices


def read_tflite(filename, batch_size, feed_dict, output_node_names, initialisation_nodes):
    tflite_graph = TFLiteGraph(filename, batch_size, feed_dict, output_node_names, initialisation_nodes)
    nng = tflite_graph.nng
    nng.refresh_after_modification()
    return nng
