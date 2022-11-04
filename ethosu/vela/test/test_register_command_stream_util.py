# SPDX-FileCopyrightText: Copyright 2020, 2022 Arm Limited and/or its affiliates <open-source-office@arm.com>
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
# Contains unit tests for register command stream generator
from ethosu.vela.api import NpuAddressRange
from ethosu.vela.api import NpuBlockTraversal
from ethosu.vela.api import NpuConv2DOperation
from ethosu.vela.api import NpuConvDepthWiseOperation
from ethosu.vela.api import NpuDataType
from ethosu.vela.api import NpuElementWiseOp
from ethosu.vela.api import NpuElementWiseOperation
from ethosu.vela.api import NpuFeatureMap
from ethosu.vela.api import NpuKernel
from ethosu.vela.api import NpuLayout
from ethosu.vela.api import NpuPadding
from ethosu.vela.api import NpuShape3D
from ethosu.vela.api import NpuTileBox
from ethosu.vela.architecture_features import Accelerator
from ethosu.vela.architecture_features import create_default_arch
from ethosu.vela.register_command_stream_generator import calc_blockdep
from ethosu.vela.register_command_stream_generator import get_strides
from ethosu.vela.register_command_stream_util import get_address_ranges
from ethosu.vela.test.extapi.test_extapi_generate_commands import create_feature_map


def test_get_fm_strides():
    """Tests calculation of feature map strides"""
    fm = NpuFeatureMap()
    fm.layout = NpuLayout.NHCWB16
    fm.data_type = NpuDataType.INT16
    fm.shape = NpuShape3D(height=7, width=10, depth=24)
    assert get_strides(fm) == NpuShape3D(height=640, width=32, depth=320)
    fm.layout = NpuLayout.NHWC
    assert get_strides(fm) == NpuShape3D(height=480, width=48, depth=2)
    fm.data_type = NpuDataType.UINT8
    assert get_strides(fm) == NpuShape3D(height=240, width=24, depth=1)


# -------------------------------------------------------------------
# ADDRESS TESTS
# -------------------------------------------------------------------


def test_get_address_ranges_one_tile():
    """Tests calculation of feature map address ranges, with 1 tile used"""
    fm = NpuFeatureMap()
    fm.region = 4
    fm.layout = NpuLayout.NHWC
    fm.data_type = NpuDataType.INT16
    fm.shape = NpuShape3D(height=50, width=40, depth=3)
    fm.tiles = NpuTileBox(height_0=50, height_1=50, width_0=40, addresses=[8000, 0, 0, 0])
    ranges = get_address_ranges(fm)
    assert ranges == [NpuAddressRange(region=4, address=8000, length=12000), None, None, None]


def test_get_address_ranges_horizontal_tiles():
    """Tests calculation of feature map address ranges, with 2 horizontal tiles used"""
    fm = NpuFeatureMap()
    fm.region = 6
    fm.layout = NpuLayout.NHWC
    fm.data_type = NpuDataType.INT16
    fm.shape = NpuShape3D(height=50, width=10, depth=20)
    fm.tiles = NpuTileBox(height_0=20, height_1=30, width_0=10, addresses=[256, 0, 16000, 0])
    ranges = get_address_ranges(fm)
    assert ranges == [
        NpuAddressRange(region=6, address=256, length=8000),
        None,
        NpuAddressRange(region=6, address=16000, length=12000),
        None,
    ]


def test_get_address_ranges_vertical_tiles():
    """Tests calculation of feature map address ranges, with 2 vertical tiles used"""
    fm = NpuFeatureMap()
    fm.region = 6
    fm.layout = NpuLayout.NHWC
    fm.data_type = NpuDataType.INT8
    # Set strides explicitly
    fm.shape = NpuShape3D(height=50, width=10, depth=20)
    fm.strides = NpuShape3D(height=100, width=20, depth=1)
    fm.tiles = NpuTileBox(height_0=50, height_1=50, width_0=5, addresses=[16, 32000, 0, 0])
    ranges = get_address_ranges(fm)
    assert ranges == [
        NpuAddressRange(region=6, address=16, length=5000),
        NpuAddressRange(region=6, address=32000, length=5000),
        None,
        None,
    ]


def test_get_address_ranges_4_tiles():
    """Tests calculation of feature map address ranges, with 4 tiles used"""
    fm = NpuFeatureMap()
    fm.region = 6
    fm.layout = NpuLayout.NHCWB16
    fm.data_type = NpuDataType.INT16
    fm.shape = NpuShape3D(height=50, width=10, depth=20)
    fm.tiles = NpuTileBox(height_0=30, height_1=10, width_0=3, addresses=[16, 32000, 8000, 16000])
    ranges = get_address_ranges(fm)
    assert ranges == [
        NpuAddressRange(region=6, address=16, length=18952),
        NpuAddressRange(region=6, address=32000, length=6280),
        NpuAddressRange(region=6, address=8000, length=12552),
        NpuAddressRange(region=6, address=16000, length=25480),
    ]


# -------------------------------------------------------------------
# BLOCKDEP TESTS
# -------------------------------------------------------------------


def test_calc_blockdep0():
    """
    Tests blockdep calculation, op1 that produces op2's IFM2.
    op2 takes 1 block to complete, which results in blockdep 0
    """
    op1 = NpuElementWiseOperation(NpuElementWiseOp.CLZ)
    op1.ifm = create_feature_map(
        NpuShape3D(height=1, width=1, depth=1),
        1,
        0x60,
        layout=NpuLayout.NHCWB16,
    )
    intermediate_fm = create_feature_map(
        NpuShape3D(height=1, width=1, depth=1),
        1,
        0xA0,
        layout=NpuLayout.NHCWB16,
    )
    op1.ofm = intermediate_fm
    op1.block_config = NpuShape3D(height=1, width=1, depth=4)
    op2 = NpuElementWiseOperation(NpuElementWiseOp.SUB)
    op2.ifm = create_feature_map(
        NpuShape3D(height=1, width=1, depth=1),
        1,
        0x39AC0,
        layout=NpuLayout.NHCWB16,
    )
    op2.ifm2 = intermediate_fm
    op2.ofm = create_feature_map(
        NpuShape3D(height=1, width=1, depth=1),
        1,
        0xE0,
        layout=NpuLayout.NHCWB16,
    )
    op2.block_config = NpuShape3D(height=1, width=1, depth=4)
    arch = create_default_arch(Accelerator.Ethos_U55_128)
    block_dep = calc_blockdep(arch, op1, op2)
    assert block_dep == 0


def test_calc_blockdep2():
    """
    Tests blockdep calculation, op1 produces part of the input of op2,
    op1 and op2 have different sizes.
    op2 takes 3 blocks to complete, op1's last block collides with op2's last block
    which results in blockdep 2
    """
    op1 = NpuConv2DOperation()
    op1.ifm = create_feature_map(
        NpuShape3D(height=4, width=48, depth=8),
        1,
        0x4C80,
        layout=NpuLayout.NHCWB16,
    )
    op1.ofm = create_feature_map(
        NpuShape3D(height=4, width=48, depth=16),
        1,
        0x6480,
        layout=NpuLayout.NHCWB16,
    )
    op1.kernel = NpuKernel(1, 1)
    op1.weights = [NpuAddressRange(region=1, address=0x4AE0, length=208)]
    op1.biases = [NpuAddressRange(region=1, address=0x49A0, length=160)]
    op1.padding = NpuPadding(top=0, left=0, right=0, bottom=0)
    op1.block_traversal = NpuBlockTraversal.PART_KERNEL_FIRST
    op1.block_config = NpuShape3D(height=4, width=6, depth=16)
    op2 = NpuConvDepthWiseOperation()
    op2.ifm = create_feature_map(
        NpuShape3D(height=3, width=48, depth=16),
        1,
        0,
        layout=NpuLayout.NHCWB16,
    )
    # op2 has two tiles, the lower tile is produced by op1
    op2.ifm.tiles = NpuTileBox(height_0=2, height_1=2, width_0=48, addresses=[0x7680, 0, 0x6480, 0])
    op2.ofm = create_feature_map(
        NpuShape3D(height=1, width=24, depth=16),
        1,
        0x6480,
        layout=NpuLayout.NHCWB16,
    )
    op2.kernel = NpuKernel(3, 3, stride_x=2, stride_y=2)
    op2.weights = [NpuAddressRange(region=1, address=0x4BB0, length=208)]
    op2.biases = [NpuAddressRange(region=1, address=0x4A40, length=160)]
    op2.padding = NpuPadding(top=0, left=0, right=0, bottom=0)
    op2.block_config = NpuShape3D(height=1, width=8, depth=16)
    arch = create_default_arch(Accelerator.Ethos_U55_128)
    block_dep = calc_blockdep(arch, op1, op2)
    assert block_dep == 2


def test_calc_blockdep3():
    """
    Tests blockdep calculation, op2 consumes part of op1, op1 and op2 have different sizes.
    There is no overlap between the last blocks of op1 and the first jobs of op2,
    which results in blockdep 3
    """
    op1 = NpuConv2DOperation()
    op1.ifm = create_feature_map(
        NpuShape3D(height=13, width=96, depth=1),
        1,
        0,
        layout=NpuLayout.NHWC,
    )
    op1.ofm = create_feature_map(
        NpuShape3D(height=6, width=48, depth=8),
        1,
        0x7C80,
        layout=NpuLayout.NHCWB16,
    )
    op1.kernel = NpuKernel(3, 3, stride_x=2, stride_y=2)
    op1.weights = [NpuAddressRange(region=1, address=0x4AE0, length=144)]
    op1.biases = [NpuAddressRange(region=1, address=0x49A0, length=80)]
    op1.padding = NpuPadding(top=0, left=0, right=1, bottom=0)
    op1.block_traversal = NpuBlockTraversal.PART_KERNEL_FIRST
    op1.block_config = NpuShape3D(height=6, width=3, depth=8)
    op2 = NpuConvDepthWiseOperation()
    op2.ifm = create_feature_map(
        NpuShape3D(height=5, width=48, depth=8),
        1,
        0x7C80,
        layout=NpuLayout.NHCWB16,
    )
    op2.ofm = create_feature_map(
        NpuShape3D(height=4, width=48, depth=8),
        1,
        0x4C80,
        layout=NpuLayout.NHCWB16,
    )
    op2.kernel = NpuKernel(3, 3)
    op2.weights = [NpuAddressRange(region=1, address=0x4BB0, length=112)]
    op2.biases = [NpuAddressRange(region=1, address=0x4A40, length=80)]
    op2.padding = NpuPadding(top=0, left=0, right=0, bottom=0)
    op2.block_config = NpuShape3D(height=4, width=6, depth=8)
    arch = create_default_arch(Accelerator.Ethos_U55_128)
    block_dep = calc_blockdep(arch, op1, op2)
    assert block_dep == 3
