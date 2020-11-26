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
# Contains unit tests for npu_generate_register_command_stream API for an external consumer
from ethosu.vela.api import npu_find_block_configs
from ethosu.vela.api import npu_generate_register_command_stream
from ethosu.vela.api import NpuAccelerator
from ethosu.vela.api import NpuActivation
from ethosu.vela.api import NpuActivationOp
from ethosu.vela.api import NpuAddressRange
from ethosu.vela.api import NpuBlockTraversal
from ethosu.vela.api import NpuConv2DOperation
from ethosu.vela.api import NpuConvDepthWiseOperation
from ethosu.vela.api import NpuDataType
from ethosu.vela.api import NpuDmaOperation
from ethosu.vela.api import NpuElementWiseOp
from ethosu.vela.api import NpuElementWiseOperation
from ethosu.vela.api import NpuFeatureMap
from ethosu.vela.api import NpuKernel
from ethosu.vela.api import NpuLayout
from ethosu.vela.api import NpuPadding
from ethosu.vela.api import NpuPoolingOp
from ethosu.vela.api import NpuPoolingOperation
from ethosu.vela.api import NpuQuantization
from ethosu.vela.api import NpuShape3D
from ethosu.vela.api import NpuTileBox
from ethosu.vela.ethos_u55_regs.ethos_u55_regs import cmd0
from ethosu.vela.ethos_u55_regs.ethos_u55_regs import cmd1
from ethosu.vela.register_command_stream_generator import CmdMode
from ethosu.vela.register_command_stream_util import get_address_ranges


def check_cmd0(cmd_stream, cmd, param):
    """Checks that the command stream contains the given command + parameter"""
    param = int(param) & 0xFFFF
    command = cmd.value | (param << 16)
    assert command in cmd_stream, f"Not in command stream: {cmd} {param}"


def check_cmd1(cmd_stream, cmd, offset, param=0x0):
    """Checks that the command stream contains the given command + parameter"""
    offset = int(offset) & 0xFFFFFFFFF
    command = cmd.value | CmdMode.Payload32.value | (param << 16)
    for i in range(len(cmd_stream) - 1):
        if cmd_stream[i] == command and cmd_stream[i + 1] == offset:
            return  # found
    assert False, f"Not in command stream: {cmd} {offset} {param}"


def find_cmd0(cmd_stream, cmd) -> int:
    """Returns parameter of the first command in the stream that matches the given command"""
    for command in cmd_stream:
        if (command & 0xFFFF) == cmd.value:
            return (command >> 16) & 0xFFFF
    assert False, f"Not in command stream: {cmd}"


def create_feature_map(
    shape: NpuShape3D,
    region: int,
    address: int,
    dtype: NpuDataType = NpuDataType.UINT8,
    layout: NpuLayout = NpuLayout.NHWC,
    quant=NpuQuantization(scale_f32=1, zero_point=0),
) -> NpuFeatureMap:
    """Creates feature map using 1 tile"""
    fm = NpuFeatureMap()
    fm.data_type = dtype
    fm.shape = shape
    fm.tiles = NpuTileBox(
        width_0=shape.width, height_0=shape.height, height_1=shape.height, addresses=[address, 0, 0, 0]
    )
    fm.region = region
    fm.layout = layout
    fm.quantization = quant
    return fm


def test_conv2d():
    """Tests command stream generation for a conv2d operation"""
    op = NpuConv2DOperation()
    op.ifm = create_feature_map(
        NpuShape3D(height=30, width=62, depth=46), 1, 512, quant=NpuQuantization(scale_f32=0.007843138, zero_point=128)
    )
    op.ofm = create_feature_map(
        NpuShape3D(height=30, width=31, depth=46),
        1,
        0x14E40,
        quant=NpuQuantization(scale_f32=0.20392157, zero_point=128),
    )
    op.kernel = NpuKernel(3, 2, 2, 1)
    op.weights = [NpuAddressRange(region=0, address=0, length=7696)]
    op.biases = [NpuAddressRange(region=0, address=32000, length=464)]
    op.padding = NpuPadding(top=0, left=0, right=1, bottom=1)
    op.block_traversal = NpuBlockTraversal.PART_KERNEL_FIRST
    op.block_config = NpuShape3D(height=16, width=4, depth=16)
    cmds = npu_generate_register_command_stream([op], NpuAccelerator.Ethos_U55_128)
    check_cmd0(cmds, cmd0.NPU_SET_IFM_REGION, 1)
    check_cmd1(cmds, cmd1.NPU_SET_IFM_BASE0, 512)
    check_cmd1(cmds, cmd1.NPU_SET_IFM_BASE1, 0)
    check_cmd1(cmds, cmd1.NPU_SET_IFM_BASE2, 0)
    check_cmd1(cmds, cmd1.NPU_SET_IFM_BASE3, 0)
    check_cmd0(cmds, cmd0.NPU_SET_IFM_HEIGHT0_M1, 29)
    check_cmd0(cmds, cmd0.NPU_SET_IFM_HEIGHT1_M1, 29)
    check_cmd0(cmds, cmd0.NPU_SET_IFM_WIDTH0_M1, 61)
    check_cmd0(cmds, cmd0.NPU_SET_IFM_DEPTH_M1, 45)
    check_cmd1(cmds, cmd1.NPU_SET_IFM_STRIDE_C, 1)
    check_cmd1(cmds, cmd1.NPU_SET_IFM_STRIDE_Y, 2852)
    check_cmd1(cmds, cmd1.NPU_SET_IFM_STRIDE_X, 46)
    check_cmd0(cmds, cmd0.NPU_SET_IFM_ZERO_POINT, 128)
    check_cmd0(cmds, cmd0.NPU_SET_IFM_PRECISION, 0)
    check_cmd0(cmds, cmd0.NPU_SET_IFM_UPSCALE, 0)
    check_cmd0(cmds, cmd0.NPU_SET_IFM_PAD_TOP, 0)
    check_cmd0(cmds, cmd0.NPU_SET_IFM_PAD_LEFT, 0)
    check_cmd0(cmds, cmd0.NPU_SET_IFM_PAD_BOTTOM, 1)
    check_cmd0(cmds, cmd0.NPU_SET_IFM_PAD_RIGHT, 1)
    check_cmd0(cmds, cmd0.NPU_SET_OFM_REGION, 1)
    check_cmd1(cmds, cmd1.NPU_SET_OFM_BASE0, 85568)
    check_cmd1(cmds, cmd1.NPU_SET_OFM_BASE1, 0)
    check_cmd1(cmds, cmd1.NPU_SET_OFM_BASE2, 0)
    check_cmd1(cmds, cmd1.NPU_SET_OFM_BASE3, 0)
    check_cmd0(cmds, cmd0.NPU_SET_OFM_HEIGHT0_M1, 29)
    check_cmd0(cmds, cmd0.NPU_SET_OFM_HEIGHT1_M1, 29)
    check_cmd0(cmds, cmd0.NPU_SET_OFM_WIDTH0_M1, 30)
    check_cmd0(cmds, cmd0.NPU_SET_OFM_HEIGHT_M1, 29)
    check_cmd0(cmds, cmd0.NPU_SET_OFM_WIDTH_M1, 30)
    check_cmd0(cmds, cmd0.NPU_SET_OFM_DEPTH_M1, 45)
    check_cmd1(cmds, cmd1.NPU_SET_OFM_STRIDE_C, 1)
    check_cmd1(cmds, cmd1.NPU_SET_OFM_STRIDE_Y, 1426)
    check_cmd1(cmds, cmd1.NPU_SET_OFM_STRIDE_X, 46)
    check_cmd0(cmds, cmd0.NPU_SET_OFM_ZERO_POINT, 128)
    check_cmd0(cmds, cmd0.NPU_SET_OFM_PRECISION, 0)
    check_cmd0(cmds, cmd0.NPU_SET_KERNEL_HEIGHT_M1, 1)
    check_cmd0(cmds, cmd0.NPU_SET_KERNEL_WIDTH_M1, 2)
    check_cmd0(cmds, cmd0.NPU_SET_KERNEL_STRIDE, 5)
    check_cmd0(cmds, cmd0.NPU_SET_WEIGHT_REGION, 0)
    check_cmd1(cmds, cmd1.NPU_SET_WEIGHT_BASE, 0)
    check_cmd1(cmds, cmd1.NPU_SET_WEIGHT_LENGTH, 7696)
    check_cmd0(cmds, cmd0.NPU_SET_SCALE_REGION, 0)
    check_cmd1(cmds, cmd1.NPU_SET_SCALE_BASE, 32000)
    check_cmd1(cmds, cmd1.NPU_SET_SCALE_LENGTH, 464)
    check_cmd0(cmds, cmd0.NPU_SET_ACTIVATION, 0)
    check_cmd0(cmds, cmd0.NPU_SET_ACTIVATION_MIN, 0)
    check_cmd0(cmds, cmd0.NPU_SET_ACTIVATION_MAX, 255)
    check_cmd0(cmds, cmd0.NPU_SET_OFM_BLK_HEIGHT_M1, 15)
    check_cmd0(cmds, cmd0.NPU_SET_OFM_BLK_WIDTH_M1, 3)
    check_cmd0(cmds, cmd0.NPU_SET_OFM_BLK_DEPTH_M1, 15)
    check_cmd0(cmds, cmd0.NPU_SET_IFM_IB_END, 14)
    check_cmd0(cmds, cmd0.NPU_SET_AB_START, 14)
    check_cmd0(cmds, cmd0.NPU_SET_ACC_FORMAT, 0)
    check_cmd0(cmds, cmd0.NPU_SET_BLOCKDEP, 0)
    check_cmd0(cmds, cmd0.NPU_OP_CONV, 0)


def create_fully_connected_op() -> NpuConv2DOperation:
    op = NpuConv2DOperation()
    op.ifm = create_feature_map(
        NpuShape3D(height=1, width=1, depth=114),
        1,
        0,
        quant=NpuQuantization(scale_f32=0.007843138, zero_point=128),
        layout=NpuLayout.NHCWB16,
    )
    op.ofm = create_feature_map(
        NpuShape3D(height=1, width=1, depth=96),
        1,
        0x6A0,
        quant=NpuQuantization(scale_f32=0.20392157, zero_point=128),
        layout=NpuLayout.NHCWB16,
    )
    op.kernel = NpuKernel(1, 1)
    op.weights = [NpuAddressRange(region=0, address=0x16880, length=13120)]
    op.biases = [NpuAddressRange(region=0, address=0x19BC0, length=960)]
    op.padding = NpuPadding(top=0, left=0, right=0, bottom=0)
    op.block_traversal = NpuBlockTraversal.DEPTH_FIRST
    op.block_config = NpuShape3D(height=2, width=4, depth=96)
    return op


def test_fully_connected():
    """Tests command stream generation for a fully connected operation"""
    op = create_fully_connected_op()
    cmds = npu_generate_register_command_stream([op], NpuAccelerator.Ethos_U55_128)
    check_cmd0(cmds, cmd0.NPU_OP_CONV, 0)
    assert len(cmds) > 20


def test_depthwise():
    """Test depthwise operation, preceeded by DMA operation"""
    weights_src = NpuAddressRange(region=0, address=0x40, length=96)
    weights_dest = NpuAddressRange(region=1, address=0x10000, length=96)
    dma_op = NpuDmaOperation(weights_src, weights_dest)
    op = NpuConvDepthWiseOperation()
    ifm_quant = NpuQuantization(scale_f32=0.007843138, zero_point=128)
    op.ifm = create_feature_map(NpuShape3D(height=64, width=64, depth=8), 1, 0x0, quant=ifm_quant)
    ofm_quant = NpuQuantization(scale_f32=0.062745101749897, zero_point=128)
    op.ofm = create_feature_map(NpuShape3D(height=64, width=64, depth=8), 1, 0x8000, quant=ofm_quant)
    op.kernel = NpuKernel(3, 3)
    op.padding = NpuPadding(top=1, left=1, right=1, bottom=1)
    op.weights = [weights_dest]
    op.biases = [NpuAddressRange(region=0, address=0, length=80)]
    op.block_config = NpuShape3D(height=8, width=12, depth=8)
    cmds = npu_generate_register_command_stream([dma_op, op], NpuAccelerator.Ethos_U55_128)
    check_cmd0(cmds, cmd0.NPU_SET_DMA0_SRC_REGION, 0)
    check_cmd1(cmds, cmd1.NPU_SET_DMA0_SRC, 0x40)
    check_cmd0(cmds, cmd0.NPU_SET_DMA0_DST_REGION, 1)
    check_cmd1(cmds, cmd1.NPU_SET_DMA0_DST, 0x10000)
    check_cmd1(cmds, cmd1.NPU_SET_DMA0_LEN, 96)
    check_cmd0(cmds, cmd0.NPU_OP_DMA_START, 0)
    # A DMA WAIT should have been inserted
    check_cmd0(cmds, cmd0.NPU_OP_DMA_WAIT, 0)
    check_cmd0(cmds, cmd0.NPU_OP_DEPTHWISE, 0)


def test_mul_with_broadcast_and_relu():
    """Test multiplication with broadcasted IFM2"""
    op = NpuElementWiseOperation(NpuElementWiseOp.MUL)
    op.ifm = create_feature_map(NpuShape3D(height=31, width=22, depth=31), 1, 0x20)
    op.ifm2 = create_feature_map(NpuShape3D(height=1, width=22, depth=1), 1, 0)
    op.ofm = create_feature_map(NpuShape3D(height=31, width=22, depth=31), 1, 0x52C0)
    op.activation = NpuActivation(NpuActivationOp.NONE_OR_RELU)
    op.activation.min = 0  # RELU
    accelerator = NpuAccelerator.Ethos_U55_32
    # Select a block config using npu_find_block_configs
    op.block_config = npu_find_block_configs(op, accelerator)[0]
    cmds = npu_generate_register_command_stream([op], accelerator)
    check_cmd1(cmds, cmd1.NPU_SET_OFM_SCALE, 1073741824, 30)
    check_cmd0(cmds, cmd0.NPU_SET_IFM_REGION, 1)
    check_cmd1(cmds, cmd1.NPU_SET_IFM_BASE0, 32)
    check_cmd1(cmds, cmd1.NPU_SET_IFM_BASE1, 0)
    check_cmd1(cmds, cmd1.NPU_SET_IFM_BASE2, 0)
    check_cmd1(cmds, cmd1.NPU_SET_IFM_BASE3, 0)
    check_cmd0(cmds, cmd0.NPU_SET_IFM_HEIGHT0_M1, 30)
    check_cmd0(cmds, cmd0.NPU_SET_IFM_HEIGHT1_M1, 30)
    check_cmd0(cmds, cmd0.NPU_SET_IFM_WIDTH0_M1, 21)
    check_cmd0(cmds, cmd0.NPU_SET_IFM_DEPTH_M1, 30)
    check_cmd1(cmds, cmd1.NPU_SET_IFM_STRIDE_C, 1)
    check_cmd1(cmds, cmd1.NPU_SET_IFM_STRIDE_Y, 682)
    check_cmd1(cmds, cmd1.NPU_SET_IFM_STRIDE_X, 31)
    check_cmd0(cmds, cmd0.NPU_SET_IFM_ZERO_POINT, 0)
    check_cmd0(cmds, cmd0.NPU_SET_IFM_PRECISION, 0)
    check_cmd0(cmds, cmd0.NPU_SET_IFM_UPSCALE, 0)
    check_cmd0(cmds, cmd0.NPU_SET_OFM_REGION, 1)
    check_cmd1(cmds, cmd1.NPU_SET_OFM_BASE0, 21184)
    check_cmd1(cmds, cmd1.NPU_SET_OFM_BASE1, 0)
    check_cmd1(cmds, cmd1.NPU_SET_OFM_BASE2, 0)
    check_cmd1(cmds, cmd1.NPU_SET_OFM_BASE3, 0)
    check_cmd0(cmds, cmd0.NPU_SET_OFM_HEIGHT0_M1, 30)
    check_cmd0(cmds, cmd0.NPU_SET_OFM_HEIGHT1_M1, 30)
    check_cmd0(cmds, cmd0.NPU_SET_OFM_WIDTH0_M1, 21)
    check_cmd0(cmds, cmd0.NPU_SET_OFM_HEIGHT_M1, 30)
    check_cmd0(cmds, cmd0.NPU_SET_OFM_WIDTH_M1, 21)
    check_cmd0(cmds, cmd0.NPU_SET_OFM_DEPTH_M1, 30)
    check_cmd1(cmds, cmd1.NPU_SET_OFM_STRIDE_C, 1)
    check_cmd1(cmds, cmd1.NPU_SET_OFM_STRIDE_Y, 682)
    check_cmd1(cmds, cmd1.NPU_SET_OFM_STRIDE_X, 31)
    check_cmd0(cmds, cmd0.NPU_SET_OFM_ZERO_POINT, 0)
    check_cmd0(cmds, cmd0.NPU_SET_OFM_PRECISION, 256)
    check_cmd0(cmds, cmd0.NPU_SET_ACTIVATION, 0)
    check_cmd0(cmds, cmd0.NPU_SET_ACTIVATION_MIN, 0)
    check_cmd0(cmds, cmd0.NPU_SET_ACTIVATION_MAX, 255)
    check_cmd0(cmds, cmd0.NPU_SET_IFM2_REGION, 1)
    check_cmd1(cmds, cmd1.NPU_SET_IFM2_BASE0, 0)
    check_cmd1(cmds, cmd1.NPU_SET_IFM2_BASE1, 0)
    check_cmd1(cmds, cmd1.NPU_SET_IFM2_BASE2, 0)
    check_cmd1(cmds, cmd1.NPU_SET_IFM2_BASE3, 0)
    check_cmd0(cmds, cmd0.NPU_SET_IFM2_HEIGHT0_M1, 0)
    check_cmd0(cmds, cmd0.NPU_SET_IFM2_HEIGHT1_M1, 0)
    check_cmd0(cmds, cmd0.NPU_SET_IFM2_WIDTH0_M1, 21)
    check_cmd1(cmds, cmd1.NPU_SET_IFM2_STRIDE_C, 1)
    check_cmd1(cmds, cmd1.NPU_SET_IFM2_STRIDE_Y, 22)
    check_cmd1(cmds, cmd1.NPU_SET_IFM2_STRIDE_X, 1)
    check_cmd0(cmds, cmd0.NPU_SET_IFM2_ZERO_POINT, 0)
    check_cmd0(cmds, cmd0.NPU_SET_IFM2_PRECISION, 0)
    check_cmd0(cmds, cmd0.NPU_SET_IFM2_BROADCAST, 5)
    check_cmd0(cmds, cmd0.NPU_SET_IFM_IB_END, 16)
    check_cmd0(cmds, cmd0.NPU_SET_AB_START, 16)
    check_cmd0(cmds, cmd0.NPU_SET_IFM2_IB_START, 9)
    check_cmd0(cmds, cmd0.NPU_SET_ACC_FORMAT, 0)
    check_cmd0(cmds, cmd0.NPU_SET_BLOCKDEP, 0)
    check_cmd0(cmds, cmd0.NPU_OP_ELEMENTWISE, 0)
    # Check that block width/height were generated that fit
    blk_height = find_cmd0(cmds, cmd0.NPU_SET_OFM_BLK_HEIGHT_M1)
    blk_width = find_cmd0(cmds, cmd0.NPU_SET_OFM_BLK_WIDTH_M1)
    blk_depth = find_cmd0(cmds, cmd0.NPU_SET_OFM_BLK_DEPTH_M1)
    assert blk_height >= 0
    assert blk_width >= 0
    assert blk_depth >= 0
    assert (blk_height + 1) * (blk_width + 1) + (blk_depth + 1) <= 3072


def create_avg_pool_op() -> NpuPoolingOperation:
    op = NpuPoolingOperation(NpuPoolingOp.AVERAGE)
    op.ifm = create_feature_map(
        NpuShape3D(height=29, width=30, depth=27), 2, 0, quant=NpuQuantization(scale_f32=0.007843138, zero_point=128)
    )
    op.ofm = create_feature_map(
        NpuShape3D(height=10, width=10, depth=27),
        2,
        0x5BD0,
        quant=NpuQuantization(scale_f32=0.20392157, zero_point=128),
    )
    op.kernel = NpuKernel(8, 2, 3, 3)
    op.padding = NpuPadding(top=0, left=2, right=3, bottom=0)
    # Select a block config
    op.block_config = NpuShape3D(height=4, width=4, depth=16)
    return op


def test_avg_pool():
    """Tests average pool operation"""
    op = create_avg_pool_op()
    cmds = npu_generate_register_command_stream([op], NpuAccelerator.Ethos_U55_128)
    check_cmd0(cmds, cmd0.NPU_OP_POOL, 1)
    assert len(cmds) > 10


def test_two_operations():
    """Tests code generation with 2 operations"""
    op1 = create_fully_connected_op()
    op2 = create_avg_pool_op()
    cmds = npu_generate_register_command_stream([op1, op2], NpuAccelerator.Ethos_U55_64)
    check_cmd0(cmds, cmd0.NPU_OP_POOL, 1)
    check_cmd0(cmds, cmd0.NPU_OP_CONV, 0)
    check_cmd0(cmds, cmd0.NPU_SET_BLOCKDEP, 0)
    # The operations are not dependent, so expect a blockdep 3
    check_cmd0(cmds, cmd0.NPU_SET_BLOCKDEP, 3)
    assert len(cmds) > 10


def test_dma_op():
    """Tests DMA operation followed by average pool. The DMA provides the contents of the average pool's IFM."""
    pool_op = create_avg_pool_op()
    assert pool_op.ofm is not None
    dest = get_address_ranges(pool_op.ofm)[0]
    assert dest is not None
    src = NpuAddressRange(0, 0x24000, dest.length)
    dma_op = NpuDmaOperation(src, dest)
    cmds = npu_generate_register_command_stream([dma_op, pool_op], NpuAccelerator.Ethos_U55_64)
    check_cmd0(cmds, cmd0.NPU_OP_DMA_START, 0)
    # A DMA WAIT should have been inserted
    check_cmd0(cmds, cmd0.NPU_OP_DMA_WAIT, 0)
    check_cmd0(cmds, cmd0.NPU_OP_POOL, 1)
