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

from ethosu.vela.graph_optimiser import convert_batched_fc_shape
from ethosu.vela.operation import Op
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
