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
from .tensor import create_virtual_tensor
from .tensor import QuantizationParameters
from .tensor import Tensor
from .tflite.BuiltinOperator import BuiltinOperator
from .tflite.Model import Model
from .tflite_mapping import builtin_operator_map
from .tflite_mapping import DataType
from .tflite_mapping import datatype_map
from .tflite_mapping import datatype_map_numpy
from .tflite_mapping import optype_to_builtintype


class TFLiteSubgraph:
    def __init__(self, graph, subgraph):
        self.graph = graph
        self.name = decode_str(subgraph.Name())

        self.tensors = []
        for idx in range(subgraph.TensorsLength()):
            self.tensors.append(self.parse_tensor(subgraph.Tensors(idx)))

        self.virtual_outputs = []
        for idx in range(subgraph.OperatorsLength()):
            self.parse_operator(idx, subgraph.Operators(idx))

        self.outputs = self.get_tensors_from_indices_remove_duplicates(subgraph.OutputsAsNumpy(), "output")
        self.inputs = self.get_tensors_from_indices_remove_duplicates(subgraph.InputsAsNumpy(), "input")
        fixup_tensors(self.inputs, self.tensors)

        self.outputs.extend(self.virtual_outputs)

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
        op_type, opt_serializer, custom_code, indices, version = self.graph.operator_codes[op_data.OpcodeIndex()]
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
        op.version = version
        op.inputs = inputs
        op.outputs = outputs
        op.intermediates = intermediates
        for out in op.outputs:
            out.ops = [op]

        if op_type in (Op.AssignVariable, Op.CallOnce):
            # All graph traversals are based on depth-first and the starting
            # points are the subgraph output tensors. Because of this, operators
            # like AssignVariable and CallOnce will not be visit when the
            # graph is traversed and the ops are never handled. In order to
            # fix that, the code base will have to be changed in several places.
            # Until then this workaround is applied. A virtual output is added
            # both to the operator and to the subgraph. By doing this the full
            # graph is traversed correctly. The tensor is not used for anything
            # else.
            op.name = f"{op_type}_{op_index}"
            tens = create_virtual_tensor(op.name)
            op.set_output_tensor(tens)
            self.virtual_outputs.append(tens)

        if op.type.is_depthwise_conv2d_op() or op.type.is_conv2d_op() or op.type == Op.FullyConnected:
            # Reshape and add bias for ops with constant weights
            # Do not modify ops with dynamic data since they will run on CPU
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
                    # a clone with a unique equivalence_id is needed.
                    inputs[-1] = clone_and_reshape_tensor(inputs[-1], None, True)

        if opt_serializer is not None:
            op.attrs = opt_serializer.deserialize(op_data)

            if op_type == Op.While:
                # Attach the actual nng subgraphs to the op
                cond_subgraph_index = op.attrs["cond_subgraph_index"]
                body_subgraph_index = op.attrs["body_subgraph_index"]
                op.attrs["subgraph"] = (
                    self.graph.nng.subgraphs[cond_subgraph_index],
                    self.graph.nng.subgraphs[body_subgraph_index],
                )
            if op_type == Op.CallOnce:
                # Attach the actual nng subgraphs to the op
                init_subgraph_index = op.attrs["init_subgraph_index"]
                op.attrs["subgraph"] = (self.graph.nng.subgraphs[init_subgraph_index],)

            if op_type == Op.Reshape:
                if "new_shape" in op.attrs["attribute_read_error"] and len(inputs) > 1:
                    # the "new_shape" attribute is optional if the new_shape tensor (inputs[1]) is specified. therefore,
                    # remove the attribute read error
                    op.attrs["attribute_read_error"].remove("new_shape")

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

            # The fused_activation_function attribute needs to be retained so that the
            # tflite_writer can correctly pass through operators that run on the CPU.
            # This is because the operator activation attribute is later converted to an
            # NpuActivation which treats None and ReLU the same, thereby making it difficult
            # for the tflite_writer to recover the original activation function.
            faf = op.attrs.get("fused_activation_function", None)
            if faf is not None:
                op.activation = create_activation_function(faf)
            if custom_code is not None:
                op.attrs["custom_code"] = custom_code

            # finally, report any missing attributes that could not be read during deserialize()
            attribute_read_error = op.attrs["attribute_read_error"]
            if len(attribute_read_error) != 0:
                print(
                    f"Warning: Could not read the following attributes from {optype_to_builtintype(op.type)}"
                    f" '{op.name}' {opt_serializer.name} field: {', '.join(attribute_read_error)}"
                )

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
    def __init__(self, data, batch_size, feed_dict, output_node_names, initialisation_nodes):

        self.op_times = {}
        if batch_size is None:
            batch_size = 1
        self.batch_size = batch_size
        self.initialisation_nodes = initialisation_nodes

        buf = None
        if type(data) == str:
            self.name = os.path.splitext(os.path.basename(data))[0]
            with open(data, "rb") as f:
                buf = bytearray(f.read())
        elif type(data) == bytearray:
            self.name = "delegate_model"
            buf = data
        elif type(data) == memoryview:
            self.name = "delegate_model"
            buf = data.tobytes()

        try:
            parsing_step = "parsing root"
            model = Model.GetRootAsModel(buf, 0)

            parsing_step = "parsing buffers length"
            self.buffers = []
            if not model.BuffersIsNone():
                for idx in range(model.BuffersLength()):
                    parsing_step = f"parsing buffer {idx}"
                    buffer = model.Buffers(idx)
                    buffer_data = self.parse_buffer(buffer)
                    # buffers can be either; empty, or contain no data (zero length), or contain data (non-zero length).
                    # when a buffer is None it means that it is either empty or zero length, and an empty buffer
                    # will have DataIsNone() equal to true.
                    # we should detect zero length buffers and report a warning because the TFLite semantics for these
                    # types of buffers changed in TensorFlow 2.11, whereby they could result in runtime errors
                    if buffer_data is None and not buffer.DataIsNone():
                        print(
                            f"Warning: Input TensorFlow Lite network contains a zero length buffer (index = {idx})"
                            f" which is semantically not empty. However, it will be treated as an empty buffer."
                        )

                    self.buffers.append(buffer_data)

            parsing_step = "parsing operator codes length"
            self.operator_codes = []
            for idx in range(model.OperatorCodesLength()):
                parsing_step = f"parsing operator code {idx}"
                self.operator_codes.append(self.parse_operator_code(model.OperatorCodes(idx)))

            parsing_step = "parsing subgraphs length"
            self.subgraphs = []

            # Pre-allocate nng subgraphs - needed when parsing an operator and the operator
            # has subgraph attributes.
            self.nng = Graph(self.name, self.batch_size)
            for idx in range(model.SubgraphsLength()):
                sg = Subgraph()
                self.nng.subgraphs.append(sg)

            for idx in range(model.SubgraphsLength()):
                parsing_step = f"parsing subgraph {idx}"
                self.subgraphs.append(TFLiteSubgraph(self, model.Subgraphs(idx)))

            for idx, tflite_sg in enumerate(self.subgraphs):
                sg = self.nng.subgraphs[idx]
                sg.name = tflite_sg.name
                sg.original_inputs = tflite_sg.inputs  # Preserve the original input order
                sg.output_tensors = tflite_sg.outputs
                sg.virtual_outputs = tflite_sg.virtual_outputs

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
        return op_type, ser, custom_code, indices, code.Version()


def read_tflite(filename, batch_size, feed_dict, output_node_names, initialisation_nodes):
    tflite_graph = TFLiteGraph(filename, batch_size, feed_dict, output_node_names, initialisation_nodes)
    nng = tflite_graph.nng
    nng.refresh_after_modification()
    return nng
