# SPDX-FileCopyrightText: Copyright 2020-2021, 2023 Arm Limited and/or its affiliates <open-source-office@arm.com>
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
# Unit tests for LUT support
import random

from ethosu.vela import lut
from ethosu.vela import mark_tensors
from ethosu.vela import pass_packing
from ethosu.vela.data_type import DataType
from ethosu.vela.high_level_command_stream import DMA
from ethosu.vela.nn_graph import Graph
from ethosu.vela.operation import Op
from ethosu.vela.rewrite_graph import rewrite_graph_pre_order
from ethosu.vela.rewrite_graph import verify_graph_health
from ethosu.vela.tensor import create_const_tensor
from ethosu.vela.tensor import TensorPurpose
from ethosu.vela.test import testutil


def set_256_lut(op, key, arch):
    random.seed(key)
    values = random.choices(range(256), k=256)
    lut_tensor = create_const_tensor(op.name + "_lut", [1, 1, 1, 256], DataType.uint8, values, TensorPurpose.LUT)
    scratch_lut_tensor = lut_tensor.clone_into_shram(arch)
    op.set_activation_lut(scratch_lut_tensor)


def set_1K_lut(op, key, arch):
    random.seed(key)
    values = random.choices(range(256), k=256)
    lut_tensor = create_const_tensor(op.name + "_lut", [1, 1, 1, 256], DataType.int32, values, TensorPurpose.LUT)
    scratch_lut_tensor = lut_tensor.clone_into_shram(arch)
    op.set_activation_lut(scratch_lut_tensor)


def set_2K_lut(op, key, arch):
    random.seed(key)
    values = random.choices(range(512), k=512)
    lut_tensor = create_const_tensor(op.name + "_lut", [1, 1, 1, 512], DataType.int32, values, TensorPurpose.LUT)
    scratch_lut_tensor = lut_tensor.clone_into_shram(arch)
    op.set_activation_lut(scratch_lut_tensor)


def process(arch, op_list):
    # Returns subgraph with given operations
    nng = Graph()
    sg = testutil.create_subgraph(op_list)
    nng.subgraphs.append(sg)
    assert verify_graph_health(nng)
    nng = mark_tensors.mark_tensor_purpose(nng, arch, False)
    assert verify_graph_health(nng)
    rewrite_graph_pre_order(nng, sg, arch, [], [])
    pass_packing.pack_into_passes(nng, arch, False)
    assert verify_graph_health(nng)
    # Create a DMA instruction for every op
    cmd_list = []
    for ps in sg.passes:
        for input_tens in ps.inputs:
            if input_tens.src_tensor:
                cmd_list.append(DMA(ps, input_tens.src_tensor, input_tens, None))

    sg.high_level_command_stream = cmd_list
    return sg


def filter_lut_cmds(cmd_list):
    lut_cmd_list = []
    for cmd in cmd_list:
        if "lut" in cmd.in_tensor.name:
            lut_cmd_list.append(cmd)
    return lut_cmd_list


def test_optimize_high_level_cmd_stream_2K():
    # Tests lut.optimize_high_level_cmd_stream, blending 256 byte and 2K luts
    arch = testutil.create_arch()
    shape = [1, 1, 1, 1]
    # u8 LUT op, should lead to DMA
    op0 = testutil.create_elemwise_op(Op.Add, "op0", shape, shape, shape)
    set_256_lut(op0, "lut0", arch)
    # u8 LUT op, should lead to DMA
    op1 = testutil.create_elemwise_op(Op.Add, "op1", shape, shape, shape)
    set_256_lut(op1, "lut1", arch)
    # u8 LUT op with different LUT, should lead to DMA
    op2 = testutil.create_elemwise_op(Op.Add, "op2", shape, shape, shape)
    set_256_lut(op2, "lut2", arch)
    # u8 LUT op with same LUT as in op1, should not lead to DMA
    op3 = testutil.create_elemwise_op(Op.Add, "op3", shape, shape, shape)
    set_256_lut(op3, "lut1", arch)
    # u8 LUT op with same LUT as in op2, should not lead to DMA
    op4 = testutil.create_elemwise_op(Op.Add, "op4", shape, shape, shape)
    set_256_lut(op4, "lut2", arch)
    # 2K LUT op, should lead to DMA, and will overwrite all previous LUTs in SHRAM
    op5_2K = testutil.create_elemwise_op(Op.Add, "op5", shape, shape, shape)
    set_2K_lut(op5_2K, "lut5", arch)
    # Another 2K LUT op, should lead to DMA, and will overwrite the previous LUT in SHRAM
    op6_2K = testutil.create_elemwise_op(Op.Add, "op6", shape, shape, shape)
    set_2K_lut(op6_2K, "lut6", arch)
    # u8 LUT op with same LUT as in op1, should lead to DMA
    op7 = testutil.create_elemwise_op(Op.Add, "op7", shape, shape, shape)
    set_256_lut(op7, "lut1", arch)

    op_list = [op0, op1, op2, op3, op4, op5_2K, op6_2K, op7]
    sg = process(arch, op_list)
    orig_cmd_list = sg.high_level_command_stream
    sg.high_level_command_stream = orig_cmd_list
    lut.optimize_high_level_cmd_stream(sg, arch)
    cmd_list = sg.high_level_command_stream
    # Check that only the needed DMA commands are left
    expected_dma_ops = [op0, op1, op2, op5_2K, op6_2K, op7]

    cmd_list = filter_lut_cmds(cmd_list)
    orig_cmd_list = filter_lut_cmds(orig_cmd_list)

    for (cmd, op) in zip(cmd_list, expected_dma_ops):
        assert cmd.in_tensor == op.activation_lut.src_tensor
    # Check that lut0, lut1 and lut2 in op0, op1, op2 are stored on different addresses
    assert orig_cmd_list[0].out_tensor.address != orig_cmd_list[1].out_tensor.address
    assert orig_cmd_list[0].out_tensor.address != orig_cmd_list[2].out_tensor.address
    assert orig_cmd_list[1].out_tensor.address != orig_cmd_list[2].out_tensor.address
    # Check that lut1 in op1 and op3 have same address
    assert orig_cmd_list[1].out_tensor.address == orig_cmd_list[3].out_tensor.address
    # Check that lut2 in op2 and op4 have same address
    assert orig_cmd_list[2].out_tensor.address == orig_cmd_list[4].out_tensor.address
    # Check that lut-s for 16 bit (op5 and op6) are stored on same address
    assert orig_cmd_list[5].out_tensor.address == orig_cmd_list[6].out_tensor.address


def test_optimize_high_level_cmd_stream_1K():
    # Tests lut.optimize_high_level_cmd_stream, blending 256 and 1K luts
    arch = testutil.create_arch()
    shape = [1, 1, 1, 1]
    # u8 LUT op, should lead to DMA
    op0 = testutil.create_elemwise_op(Op.Add, "op0", shape, shape, shape)
    set_256_lut(op0, "lut0", arch)
    # u8 LUT op, should lead to DMA
    op1 = testutil.create_elemwise_op(Op.Add, "op1", shape, shape, shape)
    set_256_lut(op1, "lut1", arch)
    # 1K LUT op with different LUT, should lead to DMA
    op2_1K = testutil.create_elemwise_op(Op.Add, "op2", shape, shape, shape)
    set_1K_lut(op2_1K, "lut2", arch)
    # u8 LUT op with same LUT as in op1, should not lead to DMA
    op3 = testutil.create_elemwise_op(Op.Add, "op3", shape, shape, shape)
    set_256_lut(op3, "lut1", arch)
    # 1K LUT op with same LUT as in op2, should not lead to DMA
    op4_1K = testutil.create_elemwise_op(Op.Add, "op4", shape, shape, shape)
    set_1K_lut(op4_1K, "lut2", arch)
    # 1K LUT op, should lead to DMA, and will overwrite lut2
    op5_2K = testutil.create_elemwise_op(Op.Add, "op5", shape, shape, shape)
    set_1K_lut(op5_2K, "lut5", arch)
    # u8 LUT op, lut0 should still be present, should not lead to DMA
    op6 = testutil.create_elemwise_op(Op.Add, "op6", shape, shape, shape)
    set_256_lut(op6, "lut0", arch)
    # 1K LUT op with same LUT as in op2, should lead to DMA
    op7 = testutil.create_elemwise_op(Op.Add, "op7", shape, shape, shape)
    set_1K_lut(op7, "lut2", arch)

    op_list = [op0, op1, op2_1K, op3, op4_1K, op5_2K, op6, op7]
    sg = process(arch, op_list)
    orig_cmd_list = sg.high_level_command_stream
    sg.high_level_command_stream = orig_cmd_list
    lut.optimize_high_level_cmd_stream(sg, arch)
    cmd_list = sg.high_level_command_stream

    cmd_list = filter_lut_cmds(cmd_list)
    orig_cmd_list = filter_lut_cmds(orig_cmd_list)

    # Check that only the needed DMA commands are left
    expected_dma_ops = [op0, op1, op2_1K, op5_2K, op7]
    for (cmd, op) in zip(cmd_list, expected_dma_ops):
        assert cmd.in_tensor == op.activation_lut.src_tensor
    # Check that lut0, lut1 and lut2 in op0, op1, op2 are stored on different addresses
    assert orig_cmd_list[0].out_tensor.address != orig_cmd_list[1].out_tensor.address
    assert orig_cmd_list[0].out_tensor.address != orig_cmd_list[2].out_tensor.address
    assert orig_cmd_list[1].out_tensor.address != orig_cmd_list[2].out_tensor.address
    # Check that lut1 in op1 and op3 have same address
    assert orig_cmd_list[1].out_tensor.address == orig_cmd_list[3].out_tensor.address
    # Check that lut2 in op2 and op4 and op7 have same address
    assert orig_cmd_list[2].out_tensor.address == orig_cmd_list[4].out_tensor.address
    assert orig_cmd_list[2].out_tensor.address == orig_cmd_list[7].out_tensor.address
