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
#
# Description:
# Unit tests for graph_optimiser
import numpy as np

from ethosu.vela.data_type import DataType
from ethosu.vela.graph_optimiser import convert_batched_fc_shape
from ethosu.vela.graph_optimiser import optimise_pad
from ethosu.vela.nn_graph import Graph
from ethosu.vela.operation import Op
from ethosu.vela.operation import Padding
from ethosu.vela.tensor import create_const_tensor
from ethosu.vela.tensor import Shape4D
from ethosu.vela.tensor import Tensor
from ethosu.vela.test import testutil


def test_convert_batched_fc():
    """Tests shape conversion of batched fully connected"""
    shape = [4, 8]
    ifm = create_const_tensor("test_in", shape, np.uint8, np.zeros(shape))
    weights = create_const_tensor("weight_in", shape, np.uint8, np.zeros(shape))
    ofm = Tensor(ifm.shape, np.uint8, "test_out")
    op = testutil.create_op(Op.FullyConnected, [ifm, weights], ofm)

    ifm.consumer_list.append(op)

    op.ifm_shapes.append(Shape4D([4, 1, 1, 8]))
    op.ofm_shapes.append(Shape4D([4, 1, 1, 8]))

    prev_op = op.clone()
    prev_op.ifm_shapes = op.ifm_shapes
    prev_op.ofm_shapes = op.ofm_shapes

    conv_op = convert_batched_fc_shape(op, None, None)

    assert conv_op.ifm != prev_op.ifm
    assert conv_op.ofm != prev_op.ofm
    assert conv_op.type == Op.FullyConnected
    assert len(conv_op.ifm.shape) == 4
    assert conv_op.ifm.shape == conv_op.ofm.shape
    assert conv_op.ifm.ops[0].type == Op.Reshape

    shape = [1, 8]
    ifm.shape = shape
    weights.shape = shape
    ofm.shape = shape
    op = testutil.create_op(Op.FullyConnected, [ifm, weights], ofm)
    ifm.consumer_list.append(op)

    op.ifm_shapes.append([1, 1, 1, 8])
    op.ofm_shapes.append([1, 1, 1, 8])

    prev_op = op.clone()
    prev_op.ifm_shapes = op.ifm_shapes
    prev_op.ofm_shapes = op.ofm_shapes

    conv_op = convert_batched_fc_shape(op, None, None)

    assert conv_op.ifm == prev_op.ifm
    assert conv_op.ofm == prev_op.ofm
    assert conv_op.type == Op.FullyConnected
    assert len(conv_op.ifm.shape) == 2
    assert conv_op.ifm.shape == conv_op.ofm.shape


def test_optimise_pad():
    """
    Tests that the PAD operator is bypassed when followed by a convolution operator,
    and that the padding of the convolution operation is correctly updated
    """
    # Create Pad operation followed by Conv2D
    quant = testutil.default_quant_params()
    in_tens = Tensor([1, 76, 75, 64], DataType.uint8, "input")
    in_tens.quantization = quant
    pad_input = create_const_tensor("pad_input", [4, 2], DataType.int32, [[0, 0], [2, 1], [1, 1], [0, 0]])
    temp_tens = Tensor([1, 79, 77, 64], DataType.uint8, "pad_out")
    temp_tens.quantization = quant.clone()
    out_tens = Tensor([1, 76, 75, 64], DataType.uint8, "output")
    out_tens.quantization = quant.clone()
    weight_tens = Tensor([5, 3, 64, 64], DataType.uint8, "weights")
    weight_tens.values = np.zeros(weight_tens.shape)
    weight_tens.quant_values = np.zeros(weight_tens.shape, np.uint8)
    weight_tens.quantization = quant.clone()

    bias_tens = Tensor([64], DataType.int32, "biases")
    pad_op = testutil.create_op(Op.Pad, [in_tens, pad_input], temp_tens)
    attrs = {"padding": Padding.VALID, "stride_w": 2, "stride_h": 2, "dilation_w_factor": 1, "dilation_h_factor": 1}
    attrs["strides"] = (1, attrs["stride_h"], attrs["stride_w"], 1)
    pad_op.run_on_npu = True
    conv2d_op = testutil.create_op(Op.Conv2D, [temp_tens, weight_tens, bias_tens], out_tens, attrs)
    conv2d_op.run_on_npu = True
    nng = Graph()
    sg = testutil.create_subgraph([pad_op, conv2d_op])
    nng.subgraphs.append(sg)
    arch = testutil.create_arch()

    optimise_pad(conv2d_op, nng, arch)

    op = sg.output_tensors[0].ops[0]
    assert op.type == Op.Conv2D
    assert op.attrs["padding"] == Padding.EXPLICIT
    assert op.attrs["explicit_padding"] == (2, 1, 1, 1)
    assert op.ifm.shape == [1, 76, 75, 64]
    assert pad_op not in op.ifm.ops
