# Copyright (C) 2020 Arm Limited or its affiliates. All rights reserved.
#
# Copyright 2015 The Gemmlowp Authors. All Rights Reserved.
#
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Description:
# Contains various fixed point math functions based on the gemmlowp fixed
# point implementation.
import numpy as np


def saturating_rounding_mul(a, b):
    assert np.int32(a) == a
    assert np.int32(b) == b
    if a == b and a == np.iinfo(np.int32).min:
        return np.int32(np.iinfo(np.int32).max)
    ab = np.int64(a) * np.int64(b)
    nudge = (1 << 30) if ab >= 0 else (1 - (1 << 30))
    result = np.int32(np.right_shift(ab + nudge, 31))
    if result < 0:
        result += 1
    return result


def shift_left(a, offset):
    assert np.int32(a) == a
    assert offset >= 0
    a_info = np.iinfo(a)
    shifted = a * (1 << offset)
    if shifted < a_info.min:
        return np.int32(a_info.min)
    elif shifted > a_info.max:
        return np.int32(a_info.max)
    else:
        return np.int32(shifted)


def rounding_divide_by_pot(x, exponent):
    assert np.int32(x) == x
    assert np.int32(exponent) == exponent
    mask = (1 << exponent) - 1
    remainder = x & mask
    threshold = mask >> 1
    if x < 0:
        threshold += 1
    result = x >> exponent
    if remainder > threshold:
        result += 1
    return result


def saturating_rounding_multiply_by_pot(exponent, x):
    assert np.int32(x) == x
    assert np.int32(exponent) == exponent
    threshold = (1 << (np.iinfo(np.int32).bits - 1 - exponent)) - 1
    if x > threshold:
        return np.iinfo(np.int32).max
    elif x < -threshold:
        return np.iinfo(np.int32).min
    else:
        return shift_left(x, exponent)


def rescale(integer_bits_src, integer_bits_dst, x):
    assert np.int32(integer_bits_src) == integer_bits_src
    assert np.int32(integer_bits_dst) == integer_bits_dst
    assert np.int32(x) == x
    exponent = integer_bits_src - integer_bits_dst
    result = saturating_rounding_multiply_by_pot(exponent, x)
    return result


# Input Q0.31
def exp_on_interval_between_negative_one_quarter_and_0_excl(a):
    assert np.int32(a) == a
    assert -1 << (31 - 2) <= a < 0
    offset = 28
    constant_term = 1895147668
    constant_1_over_3 = 715827883
    x = a + (1 << offset)
    x2 = saturating_rounding_mul(x, x)
    x3 = saturating_rounding_mul(x2, x)
    x4 = saturating_rounding_mul(x2, x2)
    x4_over_4 = rounding_divide_by_pot(x4, 2)
    x4_over_24_plus_x3_over_6_plus_x2_over_2 = rounding_divide_by_pot(
        saturating_rounding_mul((x4_over_4 + x3), constant_1_over_3) + x2, 1
    )

    return np.int32(
        constant_term + saturating_rounding_mul(constant_term, x + x4_over_24_plus_x3_over_6_plus_x2_over_2)
    )


# Input Q5.26
def exp_on_negative_values(a):
    assert np.int32(a) == a
    assert a <= 0
    one_quarter = np.int32(16777216)
    mask = np.int32(16777215)
    a_mod_quarter_minus_one_quarter = np.int32((a & mask) - one_quarter)

    result = exp_on_interval_between_negative_one_quarter_and_0_excl(rescale(5, 0, a_mod_quarter_minus_one_quarter))
    remainder = np.int32(a_mod_quarter_minus_one_quarter - a)

    def exp_barrel_shifter(exponent, multiplier, result):
        fractional_bits = 26
        integer_bits = 5
        shift = fractional_bits + exponent if integer_bits > exponent else 0
        if remainder & (1 << shift):
            return saturating_rounding_mul(result, multiplier)
        else:
            return result

    result = exp_barrel_shifter(-2, 1672461947, result)
    result = exp_barrel_shifter(-1, 1302514674, result)
    result = exp_barrel_shifter(+0, 790015084, result)
    result = exp_barrel_shifter(+1, 290630308, result)
    result = exp_barrel_shifter(+2, 39332535, result)
    result = exp_barrel_shifter(+3, 720401, result)
    result = exp_barrel_shifter(+4, 242, result)

    if a == 0:
        return np.iinfo(np.int32).max
    else:
        return result


def multiply_by_quantized_multiplier(x, scale, shift):
    # Multiplies x (int32) by (scale, shift) which have obtained by a call to scaling.quantize_scale,
    # returns rounded result
    shift = 31 - shift
    left_shift = shift if shift > 0 else 0
    right_shift = -shift if shift < 0 else 0
    mul = saturating_rounding_mul(x * (1 << left_shift), scale)
    return rounding_divide_by_pot(mul, right_shift)
