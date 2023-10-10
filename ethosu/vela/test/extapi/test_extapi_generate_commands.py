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
# Contains unit tests for npu_generate_register_command_stream API for an external consumer
import pytest

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
from ethosu.vela.architecture_features import Accelerator
from ethosu.vela.architecture_features import create_default_arch
from ethosu.vela.errors import ByteAlignmentError
from ethosu.vela.errors import ByteSizeError
from ethosu.vela.errors import VelaError
from ethosu.vela.ethos_u55_regs.ethos_u55_regs import cmd0
from ethosu.vela.ethos_u55_regs.ethos_u55_regs import cmd1
from ethosu.vela.high_level_command_to_npu_op import BasePointerIndex
from ethosu.vela.high_level_command_to_npu_op import get_mem_limits_for_regions
from ethosu.vela.register_command_stream_generator import CmdMode
from ethosu.vela.register_command_stream_generator import generate_command_stream
from ethosu.vela.register_command_stream_util import BASE_PTR_INDEX_MEM2MEM
from ethosu.vela.register_command_stream_util import get_address_ranges


def find_cmd0(cmd_stream, cmd, param, idx=0):
    """
    Searches the command stream from position idx
    Returns the position of cmd + param (if found) otherwise -1.
    """
    param = int(param) & 0xFFFF
    command = cmd.value | (param << 16)
    for i in range(idx, len(cmd_stream)):
        if cmd_stream[i] == command:
            return i
    return -1


def check_cmd0(cmd_stream, cmd, param, idx=0):
    """
    Checks that command + parameter exists in the command stream after position idx.
    Returns the position (if found) otherwise asserts.
    """
    pos = find_cmd0(cmd_stream, cmd, param, idx)
    assert pos >= 0, f"{cmd} {param} not found in the command stream (after position {idx})"
    return pos


def find_param_cmd0(cmd_stream, cmd) -> int:
    """Returns parameter of the first command in the stream that matches the given command"""
    for command in cmd_stream:
        if (command & 0xFFFF) == cmd.value:
            return (command >> 16) & 0xFFFF
    assert False, f"Not in command stream: {cmd}"


def find_cmd1(cmd_stream, cmd, offset, param=0x0, idx=0):
    """
    Searches the command stream from position idx
    Returns the position of the command (if found) otherwise -1.
    """
    offset = int(offset) & 0xFFFFFFFF
    command = cmd.value | CmdMode.Payload32.value | (param << 16)
    for i in range(idx, len(cmd_stream) - 1):
        if cmd_stream[i] == command and cmd_stream[i + 1] == offset:
            return i
    return -1


def check_cmd1(cmd_stream, cmd, offset, param=0x0, idx=0):
    """
    Checks that command + parameter exists in the command stream after position idx.
    Returns the position of the command (if found) otherwise asserts.
    """
    pos = find_cmd1(cmd_stream, cmd, offset, param, idx)
    assert pos >= 0, f"{cmd} {offset} {param} not found in the command stream (after position {idx})"
    return pos


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


def create_conv2d(
    ifm: NpuFeatureMap,
    ofm: NpuFeatureMap,
    kernel: NpuKernel,
    weights: NpuAddressRange,
    bias: NpuAddressRange,
    padding: NpuPadding,
    block_config: NpuShape3D,
):
    """Creates a Conv2D operation"""
    op = NpuConv2DOperation()
    op.ifm = ifm
    op.ofm = ofm
    op.kernel = kernel
    op.weights = [weights]
    if bias:
        op.biases = [bias]
    op.padding = padding
    op.block_traversal = NpuBlockTraversal.PART_KERNEL_FIRST
    op.block_config = block_config
    return op


def test_conv2d():
    """Tests command stream generation for a conv2d operation"""
    op = create_conv2d(
        ifm=create_feature_map(
            NpuShape3D(height=30, width=62, depth=46),
            1,
            512,
            quant=NpuQuantization(scale_f32=0.007843138, zero_point=128),
        ),
        ofm=create_feature_map(
            NpuShape3D(height=30, width=31, depth=46),
            1,
            0x14E40,
            quant=NpuQuantization(scale_f32=0.20392157, zero_point=128),
        ),
        kernel=NpuKernel(3, 2, 2, 1),
        weights=NpuAddressRange(region=0, address=0, length=7696),
        bias=NpuAddressRange(region=0, address=32000, length=464),
        padding=NpuPadding(top=0, left=0, right=1, bottom=1),
        block_config=NpuShape3D(height=16, width=4, depth=16),
    )

    cmds = npu_generate_register_command_stream([op], NpuAccelerator.Ethos_U55_128)
    set_cmds = list()
    set_cmds.append(check_cmd0(cmds, cmd0.NPU_SET_IFM_REGION, 1))
    set_cmds.append(check_cmd1(cmds, cmd1.NPU_SET_IFM_BASE0, 512))
    set_cmds.append(check_cmd1(cmds, cmd1.NPU_SET_IFM_BASE1, 0))
    set_cmds.append(check_cmd1(cmds, cmd1.NPU_SET_IFM_BASE2, 0))
    set_cmds.append(check_cmd1(cmds, cmd1.NPU_SET_IFM_BASE3, 0))
    set_cmds.append(check_cmd0(cmds, cmd0.NPU_SET_IFM_HEIGHT0_M1, 29))
    set_cmds.append(check_cmd0(cmds, cmd0.NPU_SET_IFM_HEIGHT1_M1, 29))
    set_cmds.append(check_cmd0(cmds, cmd0.NPU_SET_IFM_WIDTH0_M1, 61))
    set_cmds.append(check_cmd0(cmds, cmd0.NPU_SET_IFM_DEPTH_M1, 45))
    set_cmds.append(check_cmd1(cmds, cmd1.NPU_SET_IFM_STRIDE_C, 1))
    set_cmds.append(check_cmd1(cmds, cmd1.NPU_SET_IFM_STRIDE_Y, 2852))
    set_cmds.append(check_cmd1(cmds, cmd1.NPU_SET_IFM_STRIDE_X, 46))
    set_cmds.append(check_cmd0(cmds, cmd0.NPU_SET_IFM_ZERO_POINT, 128))
    set_cmds.append(check_cmd0(cmds, cmd0.NPU_SET_IFM_PRECISION, 0))
    set_cmds.append(check_cmd0(cmds, cmd0.NPU_SET_IFM_UPSCALE, 0))
    set_cmds.append(check_cmd0(cmds, cmd0.NPU_SET_IFM_PAD_TOP, 0))
    set_cmds.append(check_cmd0(cmds, cmd0.NPU_SET_IFM_PAD_LEFT, 0))
    set_cmds.append(check_cmd0(cmds, cmd0.NPU_SET_IFM_PAD_BOTTOM, 1))
    set_cmds.append(check_cmd0(cmds, cmd0.NPU_SET_IFM_PAD_RIGHT, 1))
    set_cmds.append(check_cmd0(cmds, cmd0.NPU_SET_OFM_REGION, 1))
    set_cmds.append(check_cmd1(cmds, cmd1.NPU_SET_OFM_BASE0, 85568))
    set_cmds.append(check_cmd1(cmds, cmd1.NPU_SET_OFM_BASE1, 0))
    set_cmds.append(check_cmd1(cmds, cmd1.NPU_SET_OFM_BASE2, 0))
    set_cmds.append(check_cmd1(cmds, cmd1.NPU_SET_OFM_BASE3, 0))
    set_cmds.append(check_cmd0(cmds, cmd0.NPU_SET_OFM_HEIGHT0_M1, 29))
    set_cmds.append(check_cmd0(cmds, cmd0.NPU_SET_OFM_HEIGHT1_M1, 29))
    set_cmds.append(check_cmd0(cmds, cmd0.NPU_SET_OFM_WIDTH0_M1, 30))
    set_cmds.append(check_cmd0(cmds, cmd0.NPU_SET_OFM_HEIGHT_M1, 29))
    set_cmds.append(check_cmd0(cmds, cmd0.NPU_SET_OFM_WIDTH_M1, 30))
    set_cmds.append(check_cmd0(cmds, cmd0.NPU_SET_OFM_DEPTH_M1, 45))
    set_cmds.append(check_cmd1(cmds, cmd1.NPU_SET_OFM_STRIDE_C, 1))
    set_cmds.append(check_cmd1(cmds, cmd1.NPU_SET_OFM_STRIDE_Y, 1426))
    set_cmds.append(check_cmd1(cmds, cmd1.NPU_SET_OFM_STRIDE_X, 46))
    set_cmds.append(check_cmd0(cmds, cmd0.NPU_SET_OFM_ZERO_POINT, 128))
    set_cmds.append(check_cmd0(cmds, cmd0.NPU_SET_OFM_PRECISION, 0))
    set_cmds.append(check_cmd0(cmds, cmd0.NPU_SET_KERNEL_HEIGHT_M1, 1))
    set_cmds.append(check_cmd0(cmds, cmd0.NPU_SET_KERNEL_WIDTH_M1, 2))
    set_cmds.append(check_cmd0(cmds, cmd0.NPU_SET_KERNEL_STRIDE, 5))
    set_cmds.append(check_cmd0(cmds, cmd0.NPU_SET_WEIGHT_REGION, 0))
    set_cmds.append(check_cmd1(cmds, cmd1.NPU_SET_WEIGHT_BASE, 0))
    set_cmds.append(check_cmd1(cmds, cmd1.NPU_SET_WEIGHT_LENGTH, 7696))
    set_cmds.append(check_cmd0(cmds, cmd0.NPU_SET_SCALE_REGION, 0))
    set_cmds.append(check_cmd1(cmds, cmd1.NPU_SET_SCALE_BASE, 32000))
    set_cmds.append(check_cmd1(cmds, cmd1.NPU_SET_SCALE_LENGTH, 464))
    set_cmds.append(check_cmd0(cmds, cmd0.NPU_SET_ACTIVATION, 0))
    set_cmds.append(check_cmd0(cmds, cmd0.NPU_SET_ACTIVATION_MIN, 0))
    set_cmds.append(check_cmd0(cmds, cmd0.NPU_SET_ACTIVATION_MAX, 255))
    set_cmds.append(check_cmd0(cmds, cmd0.NPU_SET_OFM_BLK_HEIGHT_M1, 15))
    set_cmds.append(check_cmd0(cmds, cmd0.NPU_SET_OFM_BLK_WIDTH_M1, 3))
    set_cmds.append(check_cmd0(cmds, cmd0.NPU_SET_OFM_BLK_DEPTH_M1, 15))
    set_cmds.append(check_cmd0(cmds, cmd0.NPU_SET_ACC_FORMAT, 0))
    set_cmds.append(check_cmd0(cmds, cmd0.NPU_SET_BLOCKDEP, 0))
    conv_idx = check_cmd0(cmds, cmd0.NPU_OP_CONV, 0)
    assert all([conv_idx > x for x in set_cmds]), "NPU_OP_CONV occured before the last SET operation."
    ib_end = find_param_cmd0(cmds, cmd0.NPU_SET_IFM_IB_END)
    ab_start = find_param_cmd0(cmds, cmd0.NPU_SET_AB_START)
    assert ib_end > 0
    assert ib_end <= ab_start


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
    set_cmds = list()
    set_cmds.append(check_cmd0(cmds, cmd0.NPU_SET_DMA0_SRC_REGION, 0))
    set_cmds.append(check_cmd1(cmds, cmd1.NPU_SET_DMA0_SRC, 0x40))
    set_cmds.append(check_cmd0(cmds, cmd0.NPU_SET_DMA0_DST_REGION, 1))
    set_cmds.append(check_cmd1(cmds, cmd1.NPU_SET_DMA0_DST, 0x10000))
    set_cmds.append(check_cmd1(cmds, cmd1.NPU_SET_DMA0_LEN, 96))
    dma_start_idx = check_cmd0(cmds, cmd0.NPU_OP_DMA_START, 0)
    assert all([dma_start_idx > x for x in set_cmds]), "DMA_START occured before the last SET_DMA operation"
    # A DMA WAIT should have been inserted
    dma_wait_idx = check_cmd0(cmds, cmd0.NPU_OP_DMA_WAIT, 0, dma_start_idx)
    check_cmd0(cmds, cmd0.NPU_OP_DEPTHWISE, 0, dma_wait_idx)


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
    set_cmds = list()
    set_cmds.append(check_cmd1(cmds, cmd1.NPU_SET_OFM_SCALE, 1073741824, 30))
    set_cmds.append(check_cmd0(cmds, cmd0.NPU_SET_IFM_REGION, 1))
    set_cmds.append(check_cmd1(cmds, cmd1.NPU_SET_IFM_BASE0, 32))
    set_cmds.append(check_cmd1(cmds, cmd1.NPU_SET_IFM_BASE1, 0))
    set_cmds.append(check_cmd1(cmds, cmd1.NPU_SET_IFM_BASE2, 0))
    set_cmds.append(check_cmd1(cmds, cmd1.NPU_SET_IFM_BASE3, 0))
    set_cmds.append(check_cmd0(cmds, cmd0.NPU_SET_IFM_HEIGHT0_M1, 30))
    set_cmds.append(check_cmd0(cmds, cmd0.NPU_SET_IFM_HEIGHT1_M1, 30))
    set_cmds.append(check_cmd0(cmds, cmd0.NPU_SET_IFM_WIDTH0_M1, 21))
    set_cmds.append(check_cmd0(cmds, cmd0.NPU_SET_IFM_DEPTH_M1, 30))
    set_cmds.append(check_cmd1(cmds, cmd1.NPU_SET_IFM_STRIDE_C, 1))
    set_cmds.append(check_cmd1(cmds, cmd1.NPU_SET_IFM_STRIDE_Y, 682))
    set_cmds.append(check_cmd1(cmds, cmd1.NPU_SET_IFM_STRIDE_X, 31))
    set_cmds.append(check_cmd0(cmds, cmd0.NPU_SET_IFM_ZERO_POINT, 0))
    set_cmds.append(check_cmd0(cmds, cmd0.NPU_SET_IFM_PRECISION, 0))
    set_cmds.append(check_cmd0(cmds, cmd0.NPU_SET_IFM_UPSCALE, 0))
    set_cmds.append(check_cmd0(cmds, cmd0.NPU_SET_OFM_REGION, 1))
    set_cmds.append(check_cmd1(cmds, cmd1.NPU_SET_OFM_BASE0, 21184))
    set_cmds.append(check_cmd1(cmds, cmd1.NPU_SET_OFM_BASE1, 0))
    set_cmds.append(check_cmd1(cmds, cmd1.NPU_SET_OFM_BASE2, 0))
    set_cmds.append(check_cmd1(cmds, cmd1.NPU_SET_OFM_BASE3, 0))
    set_cmds.append(check_cmd0(cmds, cmd0.NPU_SET_OFM_HEIGHT0_M1, 30))
    set_cmds.append(check_cmd0(cmds, cmd0.NPU_SET_OFM_HEIGHT1_M1, 30))
    set_cmds.append(check_cmd0(cmds, cmd0.NPU_SET_OFM_WIDTH0_M1, 21))
    set_cmds.append(check_cmd0(cmds, cmd0.NPU_SET_OFM_HEIGHT_M1, 30))
    set_cmds.append(check_cmd0(cmds, cmd0.NPU_SET_OFM_WIDTH_M1, 21))
    set_cmds.append(check_cmd0(cmds, cmd0.NPU_SET_OFM_DEPTH_M1, 30))
    set_cmds.append(check_cmd1(cmds, cmd1.NPU_SET_OFM_STRIDE_C, 1))
    set_cmds.append(check_cmd1(cmds, cmd1.NPU_SET_OFM_STRIDE_Y, 682))
    set_cmds.append(check_cmd1(cmds, cmd1.NPU_SET_OFM_STRIDE_X, 31))
    set_cmds.append(check_cmd0(cmds, cmd0.NPU_SET_OFM_ZERO_POINT, 0))
    set_cmds.append(check_cmd0(cmds, cmd0.NPU_SET_OFM_PRECISION, 256))
    set_cmds.append(check_cmd0(cmds, cmd0.NPU_SET_ACTIVATION, 0))
    set_cmds.append(check_cmd0(cmds, cmd0.NPU_SET_ACTIVATION_MIN, 0))
    set_cmds.append(check_cmd0(cmds, cmd0.NPU_SET_ACTIVATION_MAX, 255))
    set_cmds.append(check_cmd0(cmds, cmd0.NPU_SET_IFM2_REGION, 1))
    set_cmds.append(check_cmd1(cmds, cmd1.NPU_SET_IFM2_BASE0, 0))
    set_cmds.append(check_cmd1(cmds, cmd1.NPU_SET_IFM2_BASE1, 0))
    set_cmds.append(check_cmd1(cmds, cmd1.NPU_SET_IFM2_BASE2, 0))
    set_cmds.append(check_cmd1(cmds, cmd1.NPU_SET_IFM2_BASE3, 0))
    set_cmds.append(check_cmd0(cmds, cmd0.NPU_SET_IFM2_HEIGHT0_M1, 0))
    set_cmds.append(check_cmd0(cmds, cmd0.NPU_SET_IFM2_HEIGHT1_M1, 0))
    set_cmds.append(check_cmd0(cmds, cmd0.NPU_SET_IFM2_WIDTH0_M1, 21))
    set_cmds.append(check_cmd1(cmds, cmd1.NPU_SET_IFM2_STRIDE_C, 1))
    set_cmds.append(check_cmd1(cmds, cmd1.NPU_SET_IFM2_STRIDE_Y, 22))
    set_cmds.append(check_cmd1(cmds, cmd1.NPU_SET_IFM2_STRIDE_X, 1))
    set_cmds.append(check_cmd0(cmds, cmd0.NPU_SET_IFM2_ZERO_POINT, 0))
    set_cmds.append(check_cmd0(cmds, cmd0.NPU_SET_IFM2_PRECISION, 0))
    set_cmds.append(check_cmd0(cmds, cmd0.NPU_SET_IFM2_BROADCAST, 5))
    set_cmds.append(check_cmd0(cmds, cmd0.NPU_SET_IFM_IB_END, 16))
    set_cmds.append(check_cmd0(cmds, cmd0.NPU_SET_ACC_FORMAT, 0))
    set_cmds.append(check_cmd0(cmds, cmd0.NPU_SET_BLOCKDEP, 0))
    elementwise_idx = check_cmd0(cmds, cmd0.NPU_OP_ELEMENTWISE, 0)
    assert all([elementwise_idx > x for x in set_cmds]), "NPU_OP_ELEMENTWISE occured before the last SET cmd"
    ab_start = find_param_cmd0(cmds, cmd0.NPU_SET_AB_START)
    assert ab_start > 0
    ifm2_ib_start = find_param_cmd0(cmds, cmd0.NPU_SET_IFM2_IB_START)
    assert 0 < ifm2_ib_start < ab_start
    # Check that block width/height were generated that fit
    blk_height = find_param_cmd0(cmds, cmd0.NPU_SET_OFM_BLK_HEIGHT_M1)
    blk_width = find_param_cmd0(cmds, cmd0.NPU_SET_OFM_BLK_WIDTH_M1)
    blk_depth = find_param_cmd0(cmds, cmd0.NPU_SET_OFM_BLK_DEPTH_M1)
    assert blk_height >= 0
    assert blk_width >= 0
    assert blk_depth >= 0
    assert (blk_height + 1) * (blk_width + 1) + (blk_depth + 1) <= 3072


def create_avg_pool_op() -> NpuPoolingOperation:
    op = NpuPoolingOperation(NpuPoolingOp.AVERAGE)
    op.ifm = create_feature_map(
        NpuShape3D(height=32, width=30, depth=28), 2, 0, quant=NpuQuantization(scale_f32=0.007843138, zero_point=128)
    )
    op.ofm = create_feature_map(
        NpuShape3D(height=10, width=10, depth=28),
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
    assert pool_op.ifm is not None
    dest = get_address_ranges(pool_op.ifm)[0]
    assert dest is not None
    src = NpuAddressRange(0, 0x24000, dest.length)
    dma_op = NpuDmaOperation(src, dest)
    cmds = npu_generate_register_command_stream([dma_op, pool_op], NpuAccelerator.Ethos_U55_64)
    dma_start_idx = check_cmd0(cmds, cmd0.NPU_OP_DMA_START, 0)
    # A DMA WAIT should have been inserted after the dma start
    dma_wait_idx = check_cmd0(cmds, cmd0.NPU_OP_DMA_WAIT, 0, dma_start_idx)
    check_cmd0(cmds, cmd0.NPU_OP_POOL, 1, dma_wait_idx)


def setup_memory_barrier_tests():
    """
    Sets up 4 CONV operations and 4 DMA operations.
    Where dma_ops[i] provides the weights for conv[i]
    """
    ifm_addresses = [0x27100, 0x0, 0x27100, 0x0]
    ofm_addresses = [0x0, 0x27100, 0x0, 0x27100]
    weight_addr_r0 = [0x80, 0x150, 0x220, 0x2F0]
    weight_addr_r1 = [0x2E650, 0x4E220, 0x4E2F0, 0x4E3C0]
    weight_len = 208
    conv_ops = list()
    dma_ops = list()

    for i in range(4):
        weights_flash = NpuAddressRange(region=0, address=weight_addr_r0[i], length=weight_len)
        weights_sram = NpuAddressRange(region=1, address=weight_addr_r1[i], length=weight_len)
        dma_op = NpuDmaOperation(weights_flash, weights_sram)
        conv = create_conv2d(
            ifm=create_feature_map(
                shape=NpuShape3D(height=100, width=100, depth=3),
                region=1,
                address=ifm_addresses[i],
                quant=NpuQuantization(scale_f32=1.0, zero_point=0),
            ),
            ofm=create_feature_map(
                shape=NpuShape3D(height=100, width=100, depth=3),
                region=1,
                address=ofm_addresses[i],
                quant=NpuQuantization(scale_f32=1.0, zero_point=0),
            ),
            kernel=NpuKernel(3, 3, 1, 1),
            weights=weights_sram,
            bias=None,
            padding=NpuPadding(top=1, left=1, right=1, bottom=1),
            block_config=NpuShape3D(height=20, width=20, depth=8),
        )
        conv_ops.append(conv)
        dma_ops.append(dma_op)
    return conv_ops, dma_ops


def test_dma_wait_1():
    """
    Tests that DMA_WAIT barriers are properly inserted
    by the register command stream generator.
    high-level command stream:
        dma[0]
        conv[0]
        dma[1]
        conv[1]
        dma[2]
        conv[2]
    Where dma[i] provides the weights for conv[i]
    """
    conv_ops, dma_ops = setup_memory_barrier_tests()

    hlvl_cmds = [dma_ops[0], conv_ops[0], dma_ops[1], conv_ops[1], dma_ops[2], conv_ops[2]]

    # Ethos-U55
    cmds = npu_generate_register_command_stream(hlvl_cmds, NpuAccelerator.Ethos_U55_256)
    pos = 0
    pos = check_cmd0(cmds, cmd0.NPU_OP_DMA_START, 0, pos)
    pos = check_cmd0(cmds, cmd0.NPU_OP_DMA_WAIT, 0, pos)
    pos = check_cmd0(cmds, cmd0.NPU_OP_CONV, 0, pos)
    pos = check_cmd0(cmds, cmd0.NPU_OP_DMA_START, 0, pos)
    pos = check_cmd0(cmds, cmd0.NPU_OP_DMA_WAIT, 0, pos)
    pos = check_cmd0(cmds, cmd0.NPU_OP_CONV, 0, pos)
    pos = check_cmd0(cmds, cmd0.NPU_OP_DMA_START, 0, pos)
    pos = check_cmd0(cmds, cmd0.NPU_OP_DMA_WAIT, 0, pos)
    pos = check_cmd0(cmds, cmd0.NPU_OP_CONV, 0, pos)

    # Ethos-U65
    cmds = npu_generate_register_command_stream(hlvl_cmds, NpuAccelerator.Ethos_U65_256)
    pos = 0
    pos = check_cmd0(cmds, cmd0.NPU_OP_DMA_START, 0, pos)
    pos = check_cmd0(cmds, cmd0.NPU_OP_DMA_WAIT, 0, pos)
    pos = check_cmd0(cmds, cmd0.NPU_OP_CONV, 0, pos)
    pos = check_cmd0(cmds, cmd0.NPU_OP_DMA_START, 0, pos)
    pos = check_cmd0(cmds, cmd0.NPU_OP_DMA_WAIT, 0, pos)
    pos = check_cmd0(cmds, cmd0.NPU_OP_CONV, 0, pos)
    pos = check_cmd0(cmds, cmd0.NPU_OP_DMA_START, 0, pos)
    pos = check_cmd0(cmds, cmd0.NPU_OP_DMA_WAIT, 0, pos)
    pos = check_cmd0(cmds, cmd0.NPU_OP_CONV, 0, pos)


def test_dma_wait_2():
    """
    Tests that DMA_WAIT barriers are properly inserted
    by the register command stream generator.
    high-level command stream:
        dma[0]
        dma[1]
        conv[0]
        dma[2]
        conv[1]
        conv[2]
    Where dma[i] provides the weights for conv[i]
    """
    conv_ops, dma_ops = setup_memory_barrier_tests()

    hlvl_cmds = [dma_ops[0], dma_ops[1], conv_ops[0], dma_ops[2], conv_ops[1], conv_ops[2]]

    # Ethos-U55
    cmds = npu_generate_register_command_stream(hlvl_cmds, NpuAccelerator.Ethos_U55_256)
    pos = 0
    pos = check_cmd0(cmds, cmd0.NPU_OP_DMA_START, 0, pos)
    pos = check_cmd0(cmds, cmd0.NPU_OP_DMA_START, 0, pos + 1)
    pos = check_cmd0(cmds, cmd0.NPU_OP_CONV, 0, pos)
    pos = check_cmd0(cmds, cmd0.NPU_OP_DMA_START, 0, pos)
    pos = check_cmd0(cmds, cmd0.NPU_OP_CONV, 0, pos)
    pos = check_cmd0(cmds, cmd0.NPU_OP_DMA_WAIT, 0, pos)
    pos = check_cmd0(cmds, cmd0.NPU_OP_CONV, 0, pos)

    # Ethos-U65
    cmds = npu_generate_register_command_stream(hlvl_cmds, NpuAccelerator.Ethos_U65_256)
    pos = 0
    pos = check_cmd0(cmds, cmd0.NPU_OP_DMA_START, 0, pos)
    pos = check_cmd0(cmds, cmd0.NPU_OP_DMA_START, 0, pos + 1)
    pos = check_cmd0(cmds, cmd0.NPU_OP_DMA_WAIT, 1, pos)
    pos = check_cmd0(cmds, cmd0.NPU_OP_CONV, 0, pos)
    pos = check_cmd0(cmds, cmd0.NPU_OP_DMA_START, 0, pos)
    pos = check_cmd0(cmds, cmd0.NPU_OP_DMA_WAIT, 1, pos)
    pos = check_cmd0(cmds, cmd0.NPU_OP_CONV, 0, pos)
    pos = check_cmd0(cmds, cmd0.NPU_OP_DMA_WAIT, 0, pos)
    pos = check_cmd0(cmds, cmd0.NPU_OP_CONV, 0, pos)


def test_dma_wait_3():
    """
    Tests that DMA_WAIT barriers are properly inserted
    by the register command stream generator.
    high-level command stream:
        dma[0]
        dma[1]
        dma[2]
        conv[0]
        dma[3]
        conv[1]
        conv[2]
        conv[3]
    Where dma[i] provides the weights for conv[i]
    """
    conv_ops, dma_ops = setup_memory_barrier_tests()

    hlvl_cmds = [dma_ops[0], dma_ops[1], dma_ops[2], conv_ops[0], dma_ops[3], conv_ops[1], conv_ops[2], conv_ops[3]]

    # Ethos-U55
    cmds = npu_generate_register_command_stream(hlvl_cmds, NpuAccelerator.Ethos_U55_256)
    pos = 0
    pos = check_cmd0(cmds, cmd0.NPU_OP_DMA_START, 0, pos)
    pos = check_cmd0(cmds, cmd0.NPU_OP_DMA_START, 0, pos + 1)
    pos = check_cmd0(cmds, cmd0.NPU_OP_DMA_START, 0, pos + 1)
    pos = check_cmd0(cmds, cmd0.NPU_OP_CONV, 0, pos)
    pos = check_cmd0(cmds, cmd0.NPU_OP_DMA_START, 0, pos)
    pos = check_cmd0(cmds, cmd0.NPU_OP_CONV, 0, pos)
    pos = check_cmd0(cmds, cmd0.NPU_OP_CONV, 0, pos + 1)
    pos = check_cmd0(cmds, cmd0.NPU_OP_DMA_WAIT, 0, pos)
    pos = check_cmd0(cmds, cmd0.NPU_OP_CONV, 0, pos)

    # Ethos-U65
    cmds = npu_generate_register_command_stream(hlvl_cmds, NpuAccelerator.Ethos_U65_256)
    pos = 0
    pos = check_cmd0(cmds, cmd0.NPU_OP_DMA_START, 0, pos)
    pos = check_cmd0(cmds, cmd0.NPU_OP_DMA_START, 0, pos + 1)
    pos = check_cmd0(cmds, cmd0.NPU_OP_DMA_START, 0, pos + 1)
    pos = check_cmd0(cmds, cmd0.NPU_OP_CONV, 0, pos)
    pos = check_cmd0(cmds, cmd0.NPU_OP_DMA_START, 0, pos)
    pos = check_cmd0(cmds, cmd0.NPU_OP_CONV, 0, pos)
    pos = check_cmd0(cmds, cmd0.NPU_OP_DMA_WAIT, 1, pos)
    pos = check_cmd0(cmds, cmd0.NPU_OP_CONV, 0, pos)
    pos = check_cmd0(cmds, cmd0.NPU_OP_DMA_WAIT, 0, pos)
    pos = check_cmd0(cmds, cmd0.NPU_OP_CONV, 0, pos)


def test_dma_wait_4():
    """
    Tests that DMA_WAIT barriers are properly inserted
    by the register command stream generator.
    high-level command stream:
        dma[0]
        dma[1]
        dma[2]
        conv[0]
        conv[1]
        dma[3]
        conv[2]
        conv[3]
    Where dma[i] provides the weights for conv[i]
    """
    conv_ops, dma_ops = setup_memory_barrier_tests()
    hlvl_cmds = [dma_ops[0], dma_ops[1], dma_ops[2], conv_ops[0], conv_ops[1], dma_ops[3], conv_ops[2], conv_ops[3]]

    # Ethos-U55
    cmds = npu_generate_register_command_stream(hlvl_cmds, NpuAccelerator.Ethos_U55_256)
    pos = 0
    pos = check_cmd0(cmds, cmd0.NPU_OP_DMA_START, 0, pos)
    pos = check_cmd0(cmds, cmd0.NPU_OP_DMA_START, 0, pos + 1)
    pos = check_cmd0(cmds, cmd0.NPU_OP_DMA_START, 0, pos + 1)
    pos = check_cmd0(cmds, cmd0.NPU_OP_CONV, 0, pos)
    pos = check_cmd0(cmds, cmd0.NPU_OP_CONV, 0, pos + 1)
    pos = check_cmd0(cmds, cmd0.NPU_OP_DMA_START, 0, pos)
    pos = check_cmd0(cmds, cmd0.NPU_OP_CONV, 0, pos)
    pos = check_cmd0(cmds, cmd0.NPU_OP_DMA_WAIT, 0, pos)
    pos = check_cmd0(cmds, cmd0.NPU_OP_CONV, 0, pos)

    # Ethos-U65
    cmds = npu_generate_register_command_stream(hlvl_cmds, NpuAccelerator.Ethos_U65_256)
    pos = 0
    pos = check_cmd0(cmds, cmd0.NPU_OP_DMA_START, 0, pos)
    pos = check_cmd0(cmds, cmd0.NPU_OP_DMA_START, 0, pos + 1)
    pos = check_cmd0(cmds, cmd0.NPU_OP_DMA_START, 0, pos + 1)
    pos = check_cmd0(cmds, cmd0.NPU_OP_CONV, 0, pos)
    pos = check_cmd0(cmds, cmd0.NPU_OP_DMA_WAIT, 1, pos)
    pos = check_cmd0(cmds, cmd0.NPU_OP_CONV, 0, pos)
    pos = check_cmd0(cmds, cmd0.NPU_OP_DMA_START, 0, pos)
    pos = check_cmd0(cmds, cmd0.NPU_OP_DMA_WAIT, 1, pos)
    pos = check_cmd0(cmds, cmd0.NPU_OP_CONV, 0, pos)
    pos = check_cmd0(cmds, cmd0.NPU_OP_DMA_WAIT, 0, pos)
    pos = check_cmd0(cmds, cmd0.NPU_OP_CONV, 0, pos)


def test_dma_wait_5():
    """
    Tests that DMA_WAIT barriers are properly inserted
    by the register command stream generator.
    high-level command stream:
        dma[0]
        dma[1]
        dma[2]
        conv[0]
        conv[1]
        conv[2]
        dma[3]
        conv[3]
    Where dma[i] provides the weights for conv[i]
    """
    conv_ops, dma_ops = setup_memory_barrier_tests()
    hlvl_cmds = [dma_ops[0], dma_ops[1], dma_ops[2], conv_ops[0], conv_ops[1], conv_ops[2], dma_ops[3], conv_ops[3]]

    # Ethos-U55
    cmds = npu_generate_register_command_stream(hlvl_cmds, NpuAccelerator.Ethos_U55_256)
    pos = 0
    pos = check_cmd0(cmds, cmd0.NPU_OP_DMA_START, 0, pos)
    pos = check_cmd0(cmds, cmd0.NPU_OP_DMA_START, 0, pos + 1)
    pos = check_cmd0(cmds, cmd0.NPU_OP_DMA_START, 0, pos + 1)
    pos = check_cmd0(cmds, cmd0.NPU_OP_CONV, 0, pos)
    pos = check_cmd0(cmds, cmd0.NPU_OP_CONV, 0, pos + 1)
    pos = check_cmd0(cmds, cmd0.NPU_OP_DMA_WAIT, 0, pos)
    pos = check_cmd0(cmds, cmd0.NPU_OP_CONV, 0, pos)
    pos = check_cmd0(cmds, cmd0.NPU_OP_DMA_START, 0, pos)
    pos = check_cmd0(cmds, cmd0.NPU_OP_DMA_WAIT, 0, pos)
    pos = check_cmd0(cmds, cmd0.NPU_OP_CONV, 0, pos)

    # Ethos-U65
    cmds = npu_generate_register_command_stream(hlvl_cmds, NpuAccelerator.Ethos_U65_256)
    pos = 0
    pos = check_cmd0(cmds, cmd0.NPU_OP_DMA_START, 0, pos)
    pos = check_cmd0(cmds, cmd0.NPU_OP_DMA_START, 0, pos + 1)
    pos = check_cmd0(cmds, cmd0.NPU_OP_DMA_START, 0, pos + 1)
    pos = check_cmd0(cmds, cmd0.NPU_OP_CONV, 0, pos)
    pos = check_cmd0(cmds, cmd0.NPU_OP_DMA_WAIT, 1, pos)
    pos = check_cmd0(cmds, cmd0.NPU_OP_CONV, 0, pos)
    pos = check_cmd0(cmds, cmd0.NPU_OP_DMA_WAIT, 0, pos)
    pos = check_cmd0(cmds, cmd0.NPU_OP_CONV, 0, pos)
    pos = check_cmd0(cmds, cmd0.NPU_OP_DMA_START, 0, pos)
    pos = check_cmd0(cmds, cmd0.NPU_OP_DMA_WAIT, 0, pos)
    pos = check_cmd0(cmds, cmd0.NPU_OP_CONV, 0, pos)


def test_dma_wait_6():
    """
    Verify that DMA waits are not unnecessarily inserted
    between unrelated DMA and KERNEL commands
    """
    conv_ops, dma_ops = setup_memory_barrier_tests()
    cmds = npu_generate_register_command_stream([dma_ops[0], conv_ops[1]], NpuAccelerator.Ethos_U65_256)
    start_pos = check_cmd0(cmds, cmd0.NPU_OP_DMA_START, 0)
    check_cmd0(cmds, cmd0.NPU_OP_CONV, 0, start_pos)

    wait_pos = find_cmd0(cmds, cmd0.NPU_OP_DMA_WAIT, 0)
    assert (
        wait_pos == -1
    ), f"A DMA_WAIT command was unnecessarily inserted (pos {wait_pos}) between unrelated DMA and KERNEL commands"


def test_kernel_wait_0():
    """
    Verify that KERNEL_WAIT 0 is generated.
    dma_op[0] writes to the weight-address for conv[0]
    """
    conv_ops, dma_ops = setup_memory_barrier_tests()
    cmds = npu_generate_register_command_stream([conv_ops[0], dma_ops[0]], NpuAccelerator.Ethos_U65_256)
    pos = check_cmd0(cmds, cmd0.NPU_OP_CONV, 0)
    pos = check_cmd0(cmds, cmd0.NPU_OP_KERNEL_WAIT, 0, pos)
    pos = check_cmd0(cmds, cmd0.NPU_OP_DMA_START, 0, pos)


def test_kernel_wait_1():
    """
    Verify that KERNEL_WAIT 1 is generated.
    dma_op[0] writes to the weight-address for conv[0]
    """
    conv_ops, dma_ops = setup_memory_barrier_tests()
    cmds = npu_generate_register_command_stream([conv_ops[0], conv_ops[1], dma_ops[0]], NpuAccelerator.Ethos_U65_256)
    pos = check_cmd0(cmds, cmd0.NPU_OP_CONV, 0)
    pos = check_cmd0(cmds, cmd0.NPU_OP_CONV, 0, pos + 1)
    pos = check_cmd0(cmds, cmd0.NPU_OP_KERNEL_WAIT, 1, pos)
    pos = check_cmd0(cmds, cmd0.NPU_OP_DMA_START, 0, pos)


def test_check_mem_limits():
    # Tests that no code is generated with addresses out of bounds
    conv_op = create_fully_connected_op()
    # bias with end address out of range
    conv_op.biases = [NpuAddressRange(region=0, address=(1 << 32) - 16, length=1024)]
    with pytest.raises(VelaError):
        npu_generate_register_command_stream([conv_op], NpuAccelerator.Ethos_U55_64)
    # same test should pass with Ethos_U65_512
    npu_generate_register_command_stream([conv_op], NpuAccelerator.Ethos_U65_512)
    # weights with end address out of range
    conv_op = create_fully_connected_op()
    conv_op.weights = [NpuAddressRange(region=0, address=(1 << 40) - 960, length=1024)]
    with pytest.raises(VelaError):
        npu_generate_register_command_stream([conv_op], NpuAccelerator.Ethos_U65_256)
    # bias with high end address, but still within range
    addr = (1 << 40) - 1024
    conv_op = create_fully_connected_op()
    conv_op.biases = [NpuAddressRange(region=0, address=addr, length=1024)]
    cmds = npu_generate_register_command_stream([conv_op], NpuAccelerator.Ethos_U65_512)
    check_cmd1(cmds, cmd1.NPU_SET_SCALE_BASE, addr & ((1 << 32) - 1), (addr >> 32) & ((1 << 16) - 1))
    conv_op = create_fully_connected_op()
    # weights with negative address
    conv_op.weights = [NpuAddressRange(region=0, address=-16, length=1024)]
    with pytest.raises(VelaError):
        npu_generate_register_command_stream([conv_op], NpuAccelerator.Ethos_U55_32)
    op = create_avg_pool_op()
    # Tile 4's end address out of range
    op.ifm.tiles = NpuTileBox(width_0=1, height_0=1, height_1=1, addresses=[0, 800, 4000, (1 << 32) - 16])
    with pytest.raises(VelaError):
        npu_generate_register_command_stream([op], NpuAccelerator.Ethos_U55_256)
    op = create_avg_pool_op()
    # IFM region out of range
    op.ifm.region = 8
    with pytest.raises(VelaError):
        npu_generate_register_command_stream([op], NpuAccelerator.Ethos_U55_64)


def test_cmd1_payload_legality():
    # Tests payload legality

    # Test Bias and weight payload legality
    # Illegal bias length fails
    conv_op = create_fully_connected_op()
    conv_op.biases = [NpuAddressRange(region=0, address=111, length=24)]
    with pytest.raises(ByteSizeError):
        npu_generate_register_command_stream([conv_op], NpuAccelerator.Ethos_U55_64)
    # Legal bias length passes
    conv_op.biases = [NpuAddressRange(region=0, address=111, length=32)]
    npu_generate_register_command_stream([conv_op], NpuAccelerator.Ethos_U55_64)

    # Illegal weight length fails
    conv_op = create_fully_connected_op()
    conv_op.weights = [NpuAddressRange(region=0, address=128, length=24)]
    with pytest.raises(ByteSizeError):
        npu_generate_register_command_stream([conv_op], NpuAccelerator.Ethos_U55_64)
    # Legal weight length passes
    conv_op.weights = [NpuAddressRange(region=0, address=128, length=32)]
    npu_generate_register_command_stream([conv_op], NpuAccelerator.Ethos_U55_64)

    # Unaligned weight adress fails
    conv_op = create_fully_connected_op()
    conv_op.weights = [NpuAddressRange(region=0, address=120, length=32)]
    with pytest.raises(ByteAlignmentError):
        npu_generate_register_command_stream([conv_op], NpuAccelerator.Ethos_U55_64)
    # Aligned weight length already tested

    # Test DMA payload legality
    # Illegal dma length Ethos-U55 fails
    dest = NpuAddressRange(BASE_PTR_INDEX_MEM2MEM, 256, 120)
    src = NpuAddressRange(0, 512, 120)
    dma_op = NpuDmaOperation(src, dest)
    with pytest.raises(ByteSizeError):
        npu_generate_register_command_stream([dma_op], NpuAccelerator.Ethos_U55_64)

    # Legal dma length U55 passes
    dest = NpuAddressRange(BASE_PTR_INDEX_MEM2MEM, 256, 128)
    src = NpuAddressRange(0, 512, 128)
    dma_op = NpuDmaOperation(src, dest)
    npu_generate_register_command_stream([dma_op], NpuAccelerator.Ethos_U55_64)

    # Length not a multiple of 16, Ethos-U65, internal dma destination, fails
    dest = NpuAddressRange(BASE_PTR_INDEX_MEM2MEM, 256, 120)
    src = NpuAddressRange(0, 512, 120)
    dma_op = NpuDmaOperation(src, dest)
    with pytest.raises(ByteSizeError):
        npu_generate_register_command_stream([dma_op], NpuAccelerator.Ethos_U65_256)
    # Length not a multiple of 16, Ethos-U65, external dma destination passes
    dest = NpuAddressRange(2, 256, 120)
    src = NpuAddressRange(0, 512, 120)
    dma_op = NpuDmaOperation(src, dest)
    npu_generate_register_command_stream([dma_op], NpuAccelerator.Ethos_U65_256)

    # Test fm stride payload legality
    ifm_shape = NpuShape3D(height=30, width=62, depth=46)
    address = 512
    op = NpuConv2DOperation()
    op.ifm = create_feature_map(
        ifm_shape,
        1,
        address,
        quant=NpuQuantization(scale_f32=0.007843138, zero_point=128),
        dtype=NpuDataType.INT16,
    )
    op.ofm = create_feature_map(
        NpuShape3D(height=30, width=31, depth=46),
        1,
        0x14E40,
        quant=NpuQuantization(scale_f32=0.20392157, zero_point=128),
        dtype=NpuDataType.INT16,
    )
    op.kernel = NpuKernel(3, 2, 2, 1)
    op.weights = [NpuAddressRange(region=0, address=0, length=7696)]
    op.biases = [NpuAddressRange(region=0, address=32000, length=464)]
    op.padding = NpuPadding(top=0, left=0, right=1, bottom=1)
    op.block_traversal = NpuBlockTraversal.PART_KERNEL_FIRST
    op.block_config = NpuShape3D(height=16, width=4, depth=16)

    # NHWC height stride not a multiple of 16 passes
    op.ifm.strides = NpuShape3D(depth=16, height=2, width=16)
    npu_generate_register_command_stream([op], NpuAccelerator.Ethos_U65_256)

    # Same height stride fails for NHCWB16
    op.ifm = create_feature_map(
        ifm_shape,
        1,
        address,
        quant=NpuQuantization(scale_f32=0.007843138, zero_point=128),
        layout=NpuLayout.NHCWB16,
        dtype=NpuDataType.INT16,
    )
    op.ifm.strides = NpuShape3D(depth=16, height=2, width=16)
    with pytest.raises(ByteSizeError):
        npu_generate_register_command_stream([op], NpuAccelerator.Ethos_U65_256)

    # Test fm adress payload alignment

    # Unaligned adress fails
    op.ifm = create_feature_map(
        ifm_shape,
        1,
        address,
        quant=NpuQuantization(scale_f32=0.007843138, zero_point=128),
        layout=NpuLayout.NHCWB16,
        dtype=NpuDataType.INT16,
    )
    op.ifm.tiles = NpuTileBox(
        width_0=ifm_shape.width, height_0=ifm_shape.height, height_1=ifm_shape.height, addresses=[address, 16, 16, 24]
    )
    with pytest.raises(ByteAlignmentError):
        npu_generate_register_command_stream([op], NpuAccelerator.Ethos_U65_256)
    # Aligned address passes
    op.ifm.tiles = NpuTileBox(
        width_0=ifm_shape.width, height_0=ifm_shape.height, height_1=ifm_shape.height, addresses=[address, 16, 16, 16]
    )
    npu_generate_register_command_stream([op], NpuAccelerator.Ethos_U65_256)


def test_check_sram_limit_spilling():
    # Tests that no code is generated with addresses outside available sram spilling range
    arch = create_default_arch(Accelerator.Ethos_U65_512)
    assert arch.is_spilling_enabled()
    op = create_avg_pool_op()
    op.ifm.region = 0
    # OFM in scratch fast memory
    op.ofm.region = int(BasePointerIndex.ScratchFastTensor)
    w, h = op.ofm.shape.width, op.ofm.shape.height
    op.ofm.tiles = NpuTileBox(width_0=w, height_0=h, height_1=h, addresses=[32 * 1024, 0, 0, 0])
    # 384K for spilling should fit
    arch.arena_cache_size = 384 * 1024
    mem_limits = get_mem_limits_for_regions(arch)
    generate_command_stream([op], arch, verbose=False, mem_limits=mem_limits)
    # 32K for spilling does not fit, due to the OFM address
    arch.arena_cache_size = 32 * 1024
    mem_limits = get_mem_limits_for_regions(arch)
    with pytest.raises(VelaError):
        generate_command_stream([op], arch, verbose=False, mem_limits=mem_limits)
