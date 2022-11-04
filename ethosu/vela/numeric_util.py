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
# Numerical utilities for various types of rounding etc.
import math

import numpy as np


def round_up(a, b):
    return ((a + b - 1) // b) * b


def round_down(a, b):
    return (a // b) * b


def round_up_divide(a, b):
    return (a + b - 1) // b


def round_up_to_int(v):
    return int(math.ceil(v))


def round_down_to_power_of_two(v):
    assert v > 0
    while v & (v - 1):
        v &= v - 1

    return v


def round_up_to_power_of_two(v):
    return round_down_to_power_of_two(2 * v - 1)


def round_down_log2(v):
    return int(math.floor(np.log2(v)))


def round_up_log2(v):
    return int(math.ceil(np.log2(v)))


def round_to_int(v):
    return np.rint(v).astype(np.int64)


# Performs rounding away from zero.
# n.b. This is identical to C++11 std::round()
def round_away_zero(f):
    r = -0.5 if (f < 0) else 0.5
    return np.trunc(f + r)


def quantise_float32(f, scale=1.0, zero_point=0):
    return zero_point + int(round_away_zero(np.float32(f) / np.float32(scale)))


def clamp_tanh(x):
    if x <= -4:
        y = -1.0
    elif x >= 4:
        y = 1.0
    else:
        y = math.tanh(x)
    return y


def clamp_sigmoid(x):
    if x <= -8:
        y = 0.0
    elif x >= 8:
        y = 1.0
    else:
        y = 1 / (1 + math.exp(-x))
    return y


def full_shape(dim, shape, fill):
    """Returns a shape of at least dim dimensions"""
    return ([fill] * (dim - len(shape))) + shape


def overlaps(start1, end1, start2, end2):
    return start1 < end2 and start2 < end1


def is_integer(num):
    if isinstance(num, (int, np.integer)):
        return True
    if type(num) is float and num.is_integer():
        return True
    if isinstance(num, np.inexact) and np.mod(num, 1) == 0:
        return True
    return False
