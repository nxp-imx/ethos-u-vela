# SPDX-FileCopyrightText: Copyright 2020-2022 Arm Limited and/or its affiliates <open-source-office@arm.com>
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
# Unit tests for hillclimb_allocator.
import pytest

from ethosu.vela.hillclimb_allocation import allocate_live_ranges
from ethosu.vela.live_range import LiveRange


test_data = [
    ([(0, 100, 8000), (0, 1, 8016), (100, 110, 2000), (108, 110, 4000), (109, 110, 6000)], 16016),
    (
        [
            (0, 23, 131072),
            (4, 5, 65568),
            (4, 9, 8192),
            (8, 30, 15360),
            (10, 11, 65568),
            (10, 15, 4096),
            (16, 17, 65552),
            (16, 21, 2048),
            (22, 23, 32784),
            (22, 27, 1024),
        ],
        216096,
    ),
]


def live_range(start_time, end_time, size):
    lr = LiveRange(None, 1)
    lr.start_time = start_time
    lr.end_time = end_time
    lr.size = size
    return lr


@pytest.mark.parametrize("lrs, expected_size", test_data)
def test_allocate(lrs, expected_size):
    """Tests the search allocator"""
    lr_list = [live_range(start, end, size) for start, end, size in lrs]
    res = allocate_live_ranges(lr_list, None, 1 << 32)
    assert len(res) == len(lrs)
    assert max(addr + lr[2] for addr, lr in zip(res, lrs)) == expected_size


def test_allocate_empty_input():
    assert [] == allocate_live_ranges([], 0, 0)
