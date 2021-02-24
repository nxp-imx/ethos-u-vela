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
# Utilities used in vela unit tests
import numpy as np

from ethosu.vela import architecture_features
from ethosu.vela.data_type import DataType
from ethosu.vela.nn_graph import Subgraph
from ethosu.vela.operation import Op
from ethosu.vela.operation import Operation
from ethosu.vela.tensor import create_const_tensor
from ethosu.vela.tensor import QuantizationParameters
from ethosu.vela.tensor import Tensor


def create_arch():
    return architecture_features.create_default_arch(architecture_features.Accelerator.Ethos_U55_128)


def default_quant_params():
    qp = QuantizationParameters()
    qp.scale_f32 = np.float32(1)
    qp.zero_point = 0
    return qp


def create_elemwise_op(
    op_type,
    name,
    ifm_shape,
    ifm2_shape,
    ofm_shape,
    datatype=DataType.uint8,
    ifm_quant=default_quant_params(),
    ifm2_quant=default_quant_params(),
    ofm_quant=default_quant_params(),
):
    # Creates elementwise operation with constant IFM/IFM2
    if datatype.size_in_bytes() == 1:
        np_type = np.uint8
    elif datatype.size_in_bytes() == 2:
        np_type = np.int16
    else:
        np_type = np.int32
    op = Operation(op_type, name)
    op.add_input_tensor(
        create_const_tensor(name + "_ifm", ifm_shape, datatype, np.zeros(ifm_shape), np_type, quantization=ifm_quant)
    )
    if ifm2_shape is not None:
        op.add_input_tensor(
            create_const_tensor(
                name + "_ifm2", ifm2_shape, datatype, np.zeros(ifm2_shape), np_type, quantization=ifm2_quant
            )
        )
    ofm = Tensor(ofm_shape, datatype, name + "_ofm")
    ofm.quantization = ofm_quant
    op.set_output_tensor(ofm)
    op.set_ifm_ofm_shapes()

    return op


def create_op_with_quant_tensors(
    op_type, ifm_shape, ofm_shape, weights_shape=None, bias_shape=None, datatype=DataType.uint8, set_ifm_ofm_shapes=True
):
    ifm = Tensor(ifm_shape, datatype, "in")
    ifm.quantization = default_quant_params()
    ofm = Tensor(ofm_shape, datatype, "out")
    ofm.quantization = default_quant_params()
    op = Operation(op_type, "op")
    op.add_input_tensor(ifm)
    op.set_output_tensor(ofm)
    # Optional weight tensor
    if weights_shape is not None:
        if datatype.size_in_bytes() == 1:
            np_type = np.uint8
        elif datatype.size_in_bytes() == 2:
            np_type = np.int16
        else:
            np_type = np.int32
        qp = default_quant_params()
        if op.type is not Op.FullyConnected:
            qp.zero_point = np.zeros(weights_shape)
        weights = create_const_tensor(
            "weights", weights_shape, datatype, np.zeros(weights_shape), np_type, quantization=qp
        )
        op.add_input_tensor(weights)
    # Optional bias tensor
    if bias_shape is not None:
        qp = default_quant_params()
        if op.type is not Op.FullyConnected:
            qp.zero_point = np.zeros(bias_shape)
        bias = create_const_tensor("bias", bias_shape, DataType.int32, np.zeros(bias_shape), np.int32, quantization=qp)
        op.add_input_tensor(bias)

    if set_ifm_ofm_shapes:
        op.set_ifm_ofm_shapes()

    return op


def create_op(op_type, inputs, output, attrs=None, set_ifm_ofm_shapes=True):
    op = Operation(op_type, output.name + "_op")
    for input in inputs:
        op.add_input_tensor(input)
    op.set_output_tensor(output)
    if attrs is not None:
        op.attrs = attrs
    if set_ifm_ofm_shapes:
        op.set_ifm_ofm_shapes()
    return op


def create_subgraph(op_list):
    # Creates subgraph using the given list of operations
    sg = Subgraph()
    all_inputs = set(tens for op in op_list for tens in op.inputs)
    # Reversing, so that the resulting subgraph has same order as op_list
    for op in op_list[::-1]:
        for tens in op.outputs:
            if tens not in all_inputs and tens not in sg.output_tensors:
                sg.output_tensors.append(tens)
    return sg
