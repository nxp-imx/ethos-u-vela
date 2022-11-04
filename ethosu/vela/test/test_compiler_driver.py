# SPDX-FileCopyrightText: Copyright 2020 Arm Limited and/or its affiliates <open-source-office@arm.com>
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
# Unit tests for compiler driver
from ethosu.vela.compiler_driver import next_sram_factor


def test_next_sram_factor():
    lower = 0.7
    assert (1.0, False) == next_sram_factor([])
    assert (None, False) == next_sram_factor([True])
    assert (lower, True) == next_sram_factor([False])
    assert ((1 + lower) / 2, True) == next_sram_factor([False, True])
    assert (lower / 2, True) == next_sram_factor([False, False])
    # Tests next_sram_factor for a range of simulated allocator efficiencies
    for i in range(20):
        allocator_factor = i / 20.0  # The simulated allocator efficiency
        alloc_results = []
        bisected_factor = 0  # The end result of the bisect search
        while True:
            factor, dry_test = next_sram_factor(alloc_results)
            if factor is None:
                break
            alloc_result = factor < allocator_factor
            if alloc_result and not dry_test:
                bisected_factor = factor
            alloc_results.append(alloc_result)
            assert len(alloc_results) < 100
        assert bisected_factor <= allocator_factor
        assert abs(bisected_factor - allocator_factor) < 0.02
