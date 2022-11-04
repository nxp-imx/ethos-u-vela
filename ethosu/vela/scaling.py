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
# Contains various scaling calculations for weights, elementwise operations, pooling etc.
import math
from enum import IntEnum

from .numeric_util import round_away_zero


class OperandToScale(IntEnum):
    OPa = 1
    OPb = 2


# Quantise floating point scale value into 32-bit int scale and 6-bit shift
def quantise_scale(scale):
    significand, exponent = math.frexp(scale)
    significand_q31 = int(round_away_zero(significand * (1 << 31)))
    exponent_q31 = exponent - 31
    shift = exponent_q31 * -1

    if not (0 <= shift < (1 << 6)):
        # Shift outside of valid range, set scale to 0
        return 0, 16

    return significand_q31, shift


# Reduced precision quantization for int16
def reduced_quantise_scale(scale):
    multiplier, shift = quantise_scale(scale)
    reduced_multiplier = int((multiplier + (1 << 15)) >> 16) if multiplier < 32767 << 16 else 32767
    reduced_shift = shift - 16

    if not (0 <= shift < (1 << 6)):
        # Shift outside of valid range, set scale to 0
        return 0, 16

    return reduced_multiplier, reduced_shift


# Calculate global OFM scale for Average Pooling
def quantise_pooling_scale(nr_kernel_elements, rescale_bits=0):
    _, k = math.frexp(nr_kernel_elements - 1)
    N = 31 - rescale_bits
    scale = ((1 << (N + k)) + (1 << k)) // nr_kernel_elements
    shift = N + k

    assert shift < (1 << 6)

    return scale, shift


# Calculate elementwise Mul OFM scale+shift
def elementwise_mul_scale(input_scale, input2_scale, output_scale):
    output_rescale = (input_scale * input2_scale) / output_scale
    out_scale, out_shift = quantise_scale(output_rescale)
    return out_scale, out_shift


# Simplified version of calculating elementwise Add/Sub scales
def simplified_elementwise_add_sub_scale(input1_scale, input2_scale, output_scale, input_shift=16):
    max_input_scale = max(input1_scale, input2_scale)

    input1_rescale = input1_scale * (1 << input_shift) / (2 * max_input_scale)
    input2_rescale = input2_scale * (1 << input_shift) / (2 * max_input_scale)
    output_rescale = (2 * max_input_scale) / (output_scale * (1 << input_shift))

    out_scale, out_shift = quantise_scale(output_rescale)

    return input1_rescale, input2_rescale, out_scale, out_shift


# Advanced version of calculating elementwise Add/Sub scales
def advanced_elementwise_add_sub_scale(input1_scale, input2_scale, output_scale, bitdepth):
    # Always scale the smaller of the input scales
    max_input_scale = max(input1_scale, input2_scale)
    min_input_scale = min(input1_scale, input2_scale)
    input_shift = 20 if bitdepth == 8 else 15
    op_to_scale = OperandToScale.OPa if input1_scale < input2_scale else OperandToScale.OPb

    input1_rescale, _, out_scale, out_shift = simplified_elementwise_add_sub_scale(
        min_input_scale, max_input_scale, output_scale, input_shift
    )

    in_scale, in_shift = quantise_scale(input1_rescale)

    return in_scale, in_shift, out_scale, out_shift, op_to_scale
