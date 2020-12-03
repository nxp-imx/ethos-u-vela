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
from ethosu import tensor_allocator


def test_allocate():
    """Tests the search allocator"""
    # Create an input that requires a search to produce a perfect allocation.
    # Should result in size 16016
    lrs = [(0, 100, 8000), (0, 1, 8016), (100, 110, 2000), (108, 110, 4000), (109, 110, 6000)]
    input = [x for lr in lrs for x in lr]
    res = tensor_allocator.allocate(input, 0)
    assert len(res) == len(lrs)
    assert max(addr + lr[2] for addr, lr in zip(res, lrs)) == 16016
