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
# Unit tests for tensor_allocator.
import pytest

from ethosu import tensor_allocator

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


@pytest.mark.parametrize("lrs, expected_size", test_data)
def test_allocate(lrs, expected_size):
    """Tests the search allocator"""
    input = [x for lr in lrs for x in lr]
    res = tensor_allocator.allocate(input, 0)
    assert len(res) == len(lrs)
    assert max(addr + lr[2] for addr, lr in zip(res, lrs)) == expected_size
