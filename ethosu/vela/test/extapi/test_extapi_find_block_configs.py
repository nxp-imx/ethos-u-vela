# SPDX-FileCopyrightText: Copyright 2020-2021 Arm Limited and/or its affiliates <open-source-office@arm.com>
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
# Contains unit tests for npu_find_block_configs API for an external consumer
from ethosu.vela.api import npu_find_block_configs
from ethosu.vela.api import npu_generate_register_command_stream
from ethosu.vela.api import NpuAccelerator
from ethosu.vela.api import NpuAddressRange
from ethosu.vela.api import NpuBlockTraversal
from ethosu.vela.api import NpuConv2DOperation
from ethosu.vela.api import NpuKernel
from ethosu.vela.api import NpuPadding
from ethosu.vela.api import NpuQuantization
from ethosu.vela.api import NpuShape3D
from ethosu.vela.ethos_u55_regs.ethos_u55_regs import cmd0
from ethosu.vela.test.extapi.test_extapi_generate_commands import check_cmd0
from ethosu.vela.test.extapi.test_extapi_generate_commands import create_feature_map


def test_find_block_configs():
    """Tests npu_find_block_configs"""
    # Create a Conv2D operation
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
    op.biases = [NpuAddressRange(region=0, address=32000, length=464)]
    op.padding = NpuPadding(top=0, left=0, right=1, bottom=1)
    op.block_traversal = NpuBlockTraversal.PART_KERNEL_FIRST
    # Find valid block configs
    accelerator = NpuAccelerator.Ethos_U55_256
    block_configs = npu_find_block_configs(op, accelerator)
    # Select the last one
    op.block_config = block_configs[-1]
    # Note: the weights should be encoded with op.block_config.depth (not shown here)
    op.weights = [NpuAddressRange(region=0, address=0, length=7696)]
    # Check that generating register commands succeeds
    cmds = npu_generate_register_command_stream([op], accelerator)
    # Check that the selected block config was used
    check_cmd0(cmds, cmd0.NPU_SET_OFM_BLK_HEIGHT_M1, op.block_config.height - 1)
    check_cmd0(cmds, cmd0.NPU_SET_OFM_BLK_WIDTH_M1, op.block_config.width - 1)
    check_cmd0(cmds, cmd0.NPU_SET_OFM_BLK_DEPTH_M1, op.block_config.depth - 1)


def test_conv2d_block_height_1():
    """Test npu_find_block_configs returns valid config in the special case of reduced ublock height (H256)."""
    # Create a Conv2D operation
    op = NpuConv2DOperation()
    op.ifm = create_feature_map(
        NpuShape3D(height=1, width=1, depth=1024),
        1,
        512,
        quant=NpuQuantization(scale_f32=0.023528477177023888, zero_point=0),
    )
    op.ofm = create_feature_map(
        NpuShape3D(height=1, width=1, depth=1001),
        1,
        0x14E40,
        quant=NpuQuantization(scale_f32=0.16609922051429749, zero_point=66),
    )
    op.kernel = NpuKernel(1, 1, 1, 1, 1, 1)
    op.padding = NpuPadding(top=0, left=0, right=0, bottom=0)
    op.block_traversal = NpuBlockTraversal.PART_KERNEL_FIRST

    # Find valid block configs
    accelerator = NpuAccelerator.Ethos_U55_256
    block_configs = npu_find_block_configs(op, accelerator)
    # Select the last one
    op.block_config = block_configs[-1]
    # Note: the weights should be encoded with op.block_config.depth (not shown here)
    op.weights = [NpuAddressRange(region=0, address=0, length=7696)]

    # Check that generating register commands succeeds
    cmds = npu_generate_register_command_stream([op], accelerator)
    # Check that the selected block config was used
    check_cmd0(cmds, cmd0.NPU_SET_OFM_BLK_HEIGHT_M1, op.block_config.height - 1)
    check_cmd0(cmds, cmd0.NPU_SET_OFM_BLK_WIDTH_M1, op.block_config.width - 1)
    check_cmd0(cmds, cmd0.NPU_SET_OFM_BLK_DEPTH_M1, op.block_config.depth - 1)
