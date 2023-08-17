# SPDX-FileCopyrightText: Copyright 2021-2023 Arm Limited and/or its affiliates <open-source-office@arm.com>
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
# Functions used to read from a TOSA format file.
import os.path
import struct
import sys

import numpy as np

from .nn_graph import Graph
from .nn_graph import Subgraph
from .operation import ExplicitScaling
from .operation import Op
from .operation import Operation
from .reader_util import align_tensor_indices_to_nng
from .reader_util import clone_and_reshape_tensor
from .reader_util import decode_str
from .reader_util import fixup_tensors
from .shape4d import Shape4D
from .tensor import QuantizationParameters
from .tensor import shape_num_elements
from .tensor import Tensor
from .tflite_mapping import DataType
from .tosa.Op import Op as TosaOp
from .tosa.TosaGraph import TosaGraph as TG
from .tosa_mapping import datatype_map
from .tosa_mapping import datatype_map_numpy
from .tosa_mapping import tosa_operator_map
from .tosa_mapping import unsupported_tosa_operators


class TosaSubgraph:
    def __init__(self, graph, block):
        self.graph = graph
        self.name = decode_str(block.Name())

        self.tensors = []
        for idx in range(block.TensorsLength()):
            self.tensors.append(self.parse_tensor(block.Tensors(idx)))

        for idx in range(block.OperatorsLength()):
            self.parse_operator(idx, block.Operators(idx))

        # Get the subgraph inputs and outputs
        self.inputs = self.get_sg_inputs_remove_duplicates(block)
        self.outputs = self.get_sg_outputs_remove_duplicates(block)
        fixup_tensors(self.inputs, self.tensors)

    def get_sg_inputs_remove_duplicates(self, block):
        inputs = []
        for idx in range(block.InputsLength()):
            tens_data = block.Inputs(idx)
            self.add_not_duplicate(tens_data, inputs, "input")
        return inputs

    def get_sg_outputs_remove_duplicates(self, block):
        outputs = []
        for idx in range(block.OutputsLength()):
            tens_data = block.Outputs(idx)
            self.add_not_duplicate(tens_data, outputs, "output")
        return outputs

    def add_not_duplicate(self, tens_data, tensors, warning_str):
        name = decode_str(tens_data)
        tensor = self.get_tensor_by_name(name)
        if tensor not in tensors:
            tensors.append(tensor)
        else:
            print(f"Warning: Subgraph {warning_str} tensor ({tensor}) already seen. Removing the duplicate.")

    def get_tensor_by_name(self, name):
        for tens in self.tensors:
            if tens.name == name:
                return tens
        return None

    def parse_operator(self, op_index, op_data):
        op_code = op_data.Op()
        if op_code in unsupported_tosa_operators:
            print("Unsupported Operator", op_code)
            for opname in dir(TosaOp):
                if op_code == getattr(TosaOp, opname):
                    print(f"  {opname}")
            return

        op_type, attr_serializer, quant_serializer, indices = tosa_operator_map[op_code]
        inputs = []
        outputs = []
        for idx in range(op_data.InputsLength()):
            input = decode_str(op_data.Inputs(idx))
            input_tens = self.get_tensor_by_name(input)
            inputs.append(input_tens)
            if input_tens is None:
                print(f"could not find named input tensor {input}::{input_tens}")
            assert input_tens is not None

        for idx in range(op_data.OutputsLength()):
            output = decode_str(op_data.Outputs(idx))
            output_tens = self.get_tensor_by_name(output)
            outputs.append(output_tens)
            if output_tens is None:
                print(f"could not find named output tensor {output}::{output_tens}")
            assert output_tens is not None

        name = "unknown_op_name"
        if len(outputs):
            name = outputs[0].name
        inputs = align_tensor_indices_to_nng(op_type, indices, inputs)
        op = Operation(op_type, name)
        op.op_index = op_index
        op.inputs = inputs
        op.outputs = outputs

        for out in op.outputs:
            out.ops = [op]

        # TODO Transpose_conv and conv3d
        if op.type.is_depthwise_conv2d_op() or op.type.is_conv2d_op() or op.type == Op.FullyConnected:
            if inputs[1].values is not None:
                if op.type == Op.FullyConnected:
                    inputs[1] = clone_and_reshape_tensor(inputs[1], (1, 0), False)
                elif op.type.is_conv2d_op():
                    inputs[1] = clone_and_reshape_tensor(inputs[1], (1, 2, 3, 0), False)
                elif op.type.is_depthwise_conv2d_op():
                    inputs[1] = clone_and_reshape_tensor(inputs[1], (1, 2, 0, 3), False)
            if op.type.needs_bias() and len(inputs) <= op_type.info.indices.biases[0]:
                # No Bias tensor
                inputs.append(None)
            if inputs[-1] and inputs[-1].values is not None:
                # Since bias tensor is used for both bias and scale,
                # a clone with a unique equivalence_id is needed
                inputs[-1] = clone_and_reshape_tensor(inputs[-1], (0,), True)

        if attr_serializer is not None:
            op.attrs = attr_serializer.deserialize(op_data)

            if "padding" in op.attrs:
                padding = op.attrs["padding"]  # [top, bottom, left, right]
                op.attrs["explicit_padding"] = (
                    padding[0],
                    padding[2],
                    padding[1],
                    padding[3],
                )  # [top, left, bottom, right]
            if "stride" in op.attrs:
                stride = op.attrs["stride"]
                if len(stride) == 2:
                    op.attrs["strides"] = (1, stride[0], stride[1], 1)
                    del op.attrs["stride"]
                else:
                    # TODO CONV3D more to be done....
                    print("Unsupported kernel dimensions: ", len(stride))
                    assert False
            if "dilation" in op.attrs:
                dilation = op.attrs["dilation"]
                if len(dilation) == 2:
                    op.attrs["dilation"] = (1, dilation[0], dilation[1], 1)
                elif len(dilation) == 3:
                    # TODO CONV3D more to be done....
                    op.attrs["dilation"] = (dilation[0], dilation[1], dilation[2], 1)
            if "kernel" in op.attrs:
                kernel = op.attrs["kernel"]
                if len(kernel) == 2:
                    op.attrs["ksize"] = (1, kernel[0], kernel[1], 1)
                else:
                    # TODO CONV3D more to be done....
                    print("Unsupported kernel dimensions: ", len(kernel))
                    assert False
            if "shift" in op.attrs and op.type == Op.Mul:
                shift = op.attrs["shift"]
                if shift != 0:
                    op.explicit_scaling = ExplicitScaling(False, [shift], [1])
            if op.type.is_depthwise_conv2d_op():
                op.attrs["depth_multiplier"] = op.weights.shape[3]
            if op.type == Op.SplitSliceRead:
                op.read_offsets[0] = Shape4D.from_list(list(op.attrs["start"]), 0)
                op.read_shapes[0] = op.attrs["size"]

            # TODO tensor zero points currently set here
            # zero points part of Rescale operation, handled in tosa_graph_optimizer
            if "input_zp" in op.attrs:
                self.set_tensor_zp(op.ifm, op.attrs["input_zp"])
            if "weight_zp" in op.attrs:
                self.set_tensor_zp(op.weights, op.attrs["weight_zp"])
            if "output_zp" in op.attrs:
                self.set_tensor_zp(op.ofm, op.attrs["output_zp"])
            if "a_zp" in op.attrs:
                self.set_tensor_zp(op.ifm, op.attrs["a_zp"])
            if "b_zp" in op.attrs:
                self.set_tensor_zp(op.ifm2, op.attrs["b_zp"])

    def parse_tensor(self, tens_data):
        name = decode_str(tens_data.Name())
        np_shape = tens_data.ShapeAsNumpy()
        shape = list(np_shape) if type(np_shape) is np.ndarray else []
        tens_dtype = tens_data.Type()
        dtype = datatype_map[tens_dtype]

        tens = Tensor(shape, dtype, name)

        # Initialize quantization parameters
        tens.quantization = QuantizationParameters()

        if dtype == DataType.uint8:
            tens.quantization.quant_min = 0
            tens.quantization.quant_max = (1 << dtype.bits) - 1
        elif dtype in (DataType.int8, DataType.int16, DataType.int32, DataType.int48):
            tens.quantization.quant_min = -(1 << (dtype.bits - 1))
            tens.quantization.quant_max = (1 << (dtype.bits - 1)) - 1

        tens.values = None

        data_length = tens_data.DataLength()
        if data_length != 0:
            data_as_numpy = tens_data.DataAsNumpy()
            if tens_dtype in datatype_map_numpy:
                np_dtype = datatype_map_numpy[tens_dtype]

                # TOSA pads the tensor data
                shape_elements = shape_num_elements(shape)
                values = np.array(data_as_numpy.view(np_dtype))
                values = values[0:shape_elements]
                tens.values = values.reshape(shape)
            else:
                # int48 is only expected as an accumulated data/output format, int4 not supported
                print(f"Error: unsupported/unexpected Tensor type {dtype}, with data")
                assert False

        return tens

    def set_tensor_zp(self, tens, zp):
        if tens.quantization.zero_point is None:
            tens.quantization.zero_point = zp
        elif tens.quantization.zero_point != zp:
            print("Error: Setting tensor zp not possible, tensor already has different zero point")
            assert False


class TosaGraph:
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
            tosa_graph = TG.GetRootAsTosaGraph(buf, 0)

            parsing_step = "parsing version"
            self.check_version(tosa_graph)

            parsing_step = "parsing single main region"
            assert 1 == tosa_graph.RegionsLength()
            assert b"main" == tosa_graph.Regions(0).Name()

            parsing_step = "parsing blocks length"
            self.subgraphs = []
            for b_idx in range(tosa_graph.Regions(0).BlocksLength()):
                parsing_step = f"parsing block {b_idx}"
                self.subgraphs.append(TosaSubgraph(self, tosa_graph.Regions(0).Blocks(b_idx)))

            self.nng = Graph(self.name, self.batch_size)
            for tosa_sg in self.subgraphs:
                sg = Subgraph(tosa_sg.name)
                sg.original_inputs = tosa_sg.inputs  # Preserve the original input order
                sg.output_tensors = tosa_sg.outputs
                self.nng.subgraphs.append(sg)

        except (struct.error, TypeError, RuntimeError) as e:
            print(f'Error: Invalid .tosa file. Got "{e}" while {parsing_step}.')
            sys.exit(1)

    def check_version(self, tosa_graph):
        version = tosa_graph.Version()
        version_str = f"{version._Major()}.{version._Minor()}.{version._Patch()}"
        if version_str != "0.80.0":
            print(f"Unsupported TOSA version: {version_str}")
            assert False


def read_tosa(filename, batch_size, feed_dict, output_node_names, initialisation_nodes):
    tosa_graph = TosaGraph(filename, batch_size, feed_dict, output_node_names, initialisation_nodes)
    nng = tosa_graph.nng
    nng.refresh_after_modification()
    return nng
