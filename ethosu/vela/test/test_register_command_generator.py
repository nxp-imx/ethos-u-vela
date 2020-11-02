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
# Contains unit tests for register command stream generator
from ethosu.vela.api import NpuAddressRange
from ethosu.vela.api import NpuDataType
from ethosu.vela.api import NpuFeatureMap
from ethosu.vela.api import NpuLayout
from ethosu.vela.api import NpuShape3D
from ethosu.vela.api import NpuTileBox
from ethosu.vela.register_command_stream_generator import get_address_ranges
from ethosu.vela.register_command_stream_generator import get_strides


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
        NpuAddressRange(region=6, address=28800, length=12680),
    ]
