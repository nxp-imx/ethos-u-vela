# SPDX-FileCopyrightText: Copyright 2022 Arm Limited and/or its affiliates <open-source-office@arm.com>
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
# Unit tests for scaling
from ethosu.vela.scaling import quantise_scale
from ethosu.vela.scaling import reduced_quantise_scale


def test_scaling():
    multiplier, shift = quantise_scale(1)
    assert multiplier == 1073741824 and shift == 30
    multiplier, shift = quantise_scale(0.5)
    assert multiplier == 1073741824 and shift == 31
    multiplier, shift = quantise_scale(0.001)
    assert multiplier == 1099511628 and shift == 40
    multiplier, shift = quantise_scale(0.008)
    assert multiplier == 1099511628 and shift == 37
    multiplier, shift = quantise_scale(0.00097652)
    assert multiplier == 2147390190 and shift == 41
    multiplier, shift = quantise_scale(0.0009765615959827986)
    assert multiplier == 2147481660 and shift == 41


def test_reduced_scaling():
    multiplier, shift = reduced_quantise_scale(1)
    assert multiplier == 16384 and shift == 14
    multiplier, shift = reduced_quantise_scale(0.5)
    assert multiplier == 16384 and shift == 15
    multiplier, shift = reduced_quantise_scale(0.001)
    assert multiplier == 16777 and shift == 24
    multiplier, shift = reduced_quantise_scale(0.008)
    assert multiplier == 16777 and shift == 21
    multiplier, shift = reduced_quantise_scale(0.00097652)
    assert multiplier == 32767 and shift == 25
    # multiplier saturated
    multiplier, shift = reduced_quantise_scale(0.0009765615959827986)
    assert multiplier == 32767 and shift == 25
