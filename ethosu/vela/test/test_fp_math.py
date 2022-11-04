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
# Unit tests for fixed point math
import numpy as np
import pytest

from ethosu.vela import fp_math
from ethosu.vela import scaling
from ethosu.vela.softmax import SoftMax

# Turn off black formatting for EXP_LUT to keep it compact
# fmt: off

EXP_LUT = [
    0x000011c9, 0x000012b8, 0x000013b4, 0x000014bd, 0x000015d4, 0x000016fa, 0x0000182f, 0x00001975,
    0x00001acb, 0x00001c34, 0x00001daf, 0x00001f3f, 0x000020e3, 0x0000229e, 0x00002470, 0x0000265a,
    0x0000285e, 0x00002a7d, 0x00002cb9, 0x00002f13, 0x0000318c, 0x00003427, 0x000036e5, 0x000039c8,
    0x00003cd1, 0x00004004, 0x00004361, 0x000046ec, 0x00004aa6, 0x00004e93, 0x000052b4, 0x0000570d,
    0x00005ba1, 0x00006072, 0x00006583, 0x00006ada, 0x00007077, 0x00007661, 0x00007c9a, 0x00008327,
    0x00008a0c, 0x0000914d, 0x000098f1, 0x0000a0fb, 0x0000a971, 0x0000b259, 0x0000bbb9, 0x0000c597,
    0x0000cffa, 0x0000dae9, 0x0000e66b, 0x0000f288, 0x0000ff48, 0x00010cb3, 0x00011ad3, 0x000129b1,
    0x00013957, 0x000149d0, 0x00015b26, 0x00016d65, 0x0001809b, 0x000194d2, 0x0001aa1a, 0x0001c080,
    0x0001d814, 0x0001f0e4, 0x00020b03, 0x00022681, 0x00024371, 0x000261e7, 0x000281f7, 0x0002a3b5,
    0x0002c73b, 0x0002ec9e, 0x000313f8, 0x00033d64, 0x000368fd, 0x000396e1, 0x0003c72e, 0x0003fa05,
    0x00042f89, 0x000467dd, 0x0004a326, 0x0004e18e, 0x0005233d, 0x00056861, 0x0005b126, 0x0005fdbf,
    0x00064e5f, 0x0006a33c, 0x0006fc8e, 0x00075a93, 0x0007bd89, 0x000825b3, 0x00089356, 0x000906bd,
    0x00098035, 0x000a000f, 0x000a86a2, 0x000b1447, 0x000ba95f, 0x000c464e, 0x000ceb7c, 0x000d9959,
    0x000e505a, 0x000f10f9, 0x000fdbb9, 0x0010b120, 0x001191c0, 0x00127e2f, 0x0013770b, 0x00147cfc,
    0x001590b2, 0x0016b2e7, 0x0017e45d, 0x001925e1, 0x001a784c, 0x001bdc81, 0x001d536f, 0x001ede14,
    0x00207d77, 0x002232af, 0x0023fee4, 0x0025e349, 0x0027e125, 0x0029f9ce, 0x002c2ead, 0x002e813e,
    0x0030f30f, 0x003385c7, 0x00363b1f, 0x003914e9, 0x003c1510, 0x003f3d97, 0x004290a1, 0x00461066,
    0x0049bf41, 0x004d9fad, 0x0051b444, 0x0055ffc3, 0x005a850f, 0x005f4730, 0x0064495a, 0x00698eeb,
    0x006f1b6c, 0x0074f299, 0x007b185f, 0x008190de, 0x00886074, 0x008f8bae, 0x00971762, 0x009f08a2,
    0x00a764c2, 0x00b03164, 0x00b9746e, 0x00c3341b, 0x00cd76fa, 0x00d843ed, 0x00e3a23b, 0x00ef9983,
    0x00fc31d2, 0x010973a0, 0x011767d1, 0x012617cf, 0x01358d70, 0x0145d31c, 0x0156f3c1, 0x0168fadf,
    0x017bf4a0, 0x018fedb6, 0x01a4f394, 0x01bb145a, 0x01d25ee1, 0x01eae2e5, 0x0204b0c8, 0x021fd9ed,
    0x023c7091, 0x025a87f9, 0x027a343d, 0x029b8ac5, 0x02bea1ee, 0x02e3914d, 0x030a71c2, 0x03335d4e,
    0x035e6f8d, 0x038bc56a, 0x03bb7d57, 0x03edb77c, 0x04229573, 0x045a3ae4, 0x0494cd29, 0x04d2739e,
    0x051357c7, 0x0557a519, 0x059f8997, 0x05eb358d, 0x063adbcc, 0x068eb1ff, 0x06e6f049, 0x0743d21b,
    0x07a595d9, 0x080c7d29, 0x0878cd66, 0x08eacf1a, 0x0962cf07, 0x09e11dcc, 0x0a661032, 0x0af1ffea,
    0x0b854a9a, 0x0c20536f, 0x0cc3828e, 0x0d6f4584, 0x0e241040, 0x0ee25bb0, 0x0faaa7f2, 0x107d7b9e,
    0x115b64be, 0x1244f787, 0x133ad1c6, 0x143d9885, 0x154df999, 0x166cac7a, 0x179a70d5, 0x18d81262,
    0x1a266657, 0x1b864d4c, 0x1cf8b43e, 0x1e7e9316, 0x2018f0b9, 0x21c8e0b1, 0x238f851d, 0x256e1046,
    0x2765c287, 0x2977ef55, 0x2ba5fab4, 0x2df15b8a, 0x305b9d83, 0x32e65ea3, 0x35935539, 0x38644d75,
    0x3b5b2b74, 0x3e79eea7, 0x41c2addc, 0x45379f60, 0x48db159c, 0x4caf81fa, 0x50b7797f, 0x54f5af2b,
    0x596cfe46, 0x5e2066e8, 0x631310c8, 0x684852d8, 0x6dc3a909, 0x7388c43d, 0x799b84b7, 0x7fffffff,
]
# fmt: on


def test_saturating_rounding_mul():
    i32info = np.iinfo(np.int32)
    i16info = np.iinfo(np.int16)

    # Saturation
    assert fp_math.saturating_rounding_mul32(i32info.min, i32info.min) == i32info.max
    assert fp_math.saturating_rounding_mul32(i32info.min, i32info.max) == -i32info.max
    assert fp_math.saturating_rounding_mul32(i32info.max, i32info.min) == -i32info.max

    assert fp_math.saturating_rounding_mul16(i16info.min, i16info.min) == i16info.max
    assert fp_math.saturating_rounding_mul16(i16info.min, i16info.max) == -i16info.max
    assert fp_math.saturating_rounding_mul16(i16info.max, i16info.min) == -i16info.max

    # Multiply by zero
    assert fp_math.saturating_rounding_mul32(0, fp_math.from_float(1.0)) == 0
    assert fp_math.saturating_rounding_mul32(0, fp_math.from_float(-1.0)) == 0
    assert fp_math.saturating_rounding_mul32(fp_math.from_float(1.0), 0) == 0
    assert fp_math.saturating_rounding_mul32(fp_math.from_float(-1.0), 0) == 0

    assert fp_math.saturating_rounding_mul16(0, i16info.max) == 0
    assert fp_math.saturating_rounding_mul16(0, i16info.min) == 0
    assert fp_math.saturating_rounding_mul16(i16info.max, 0) == 0
    assert fp_math.saturating_rounding_mul16(i16info.min, 0) == 0

    # Multiply positive/negative
    assert fp_math.saturating_rounding_mul32(fp_math.from_float(1.0), fp_math.from_float(1.0)) == fp_math.from_float(
        1.0, 5 + 5
    )
    assert fp_math.saturating_rounding_mul32(fp_math.from_float(-1.0), fp_math.from_float(1.0)) == fp_math.from_float(
        -1.0, 5 + 5
    )
    assert fp_math.saturating_rounding_mul32(fp_math.from_float(1.0), fp_math.from_float(-1.0)) == fp_math.from_float(
        -1.0, 5 + 5
    )
    assert fp_math.saturating_rounding_mul32(fp_math.from_float(-1.0), fp_math.from_float(-1.0)) == fp_math.from_float(
        1.0, 5 + 5
    )

    # Rounding
    assert fp_math.saturating_rounding_mul32(fp_math.from_float(16.0), 1) == 1
    assert fp_math.saturating_rounding_mul32(fp_math.from_float(-16.0), 1) == 0
    assert fp_math.saturating_rounding_mul32(fp_math.from_float(16.0) - 1, 1) == 0
    assert fp_math.saturating_rounding_mul32(fp_math.from_float(-16.0) - 1, 1) == -1

    assert fp_math.saturating_rounding_mul16(fp_math.from_float(16.0, 21), 1) == 1
    assert fp_math.saturating_rounding_mul16(fp_math.from_float(-16.0, 21), 1) == 0
    assert fp_math.saturating_rounding_mul16(fp_math.from_float(16.0, 21) - 1, 1) == 0
    assert fp_math.saturating_rounding_mul16(fp_math.from_float(-16.0, 21) - 1, 1) == -1


def test_shift_left():
    i32info = np.iinfo(np.int32)
    i16info = np.iinfo(np.int16)
    assert fp_math.shift_left32(1, i32info.bits) == i32info.max
    assert fp_math.shift_left32(-1, i32info.bits) == i32info.min
    assert fp_math.shift_left32(1, i32info.bits - 2) == (i32info.max + 1) / 2
    assert fp_math.shift_left32(-1, i32info.bits - 2) == i32info.min // 2

    assert fp_math.shift_left16(1, i16info.bits) == i16info.max
    assert fp_math.shift_left16(-1, i16info.bits) == i16info.min
    assert fp_math.shift_left16(1, i16info.bits - 2) == (i16info.max + 1) / 2
    assert fp_math.shift_left16(-1, i16info.bits - 2) == i16info.min // 2

    assert fp_math.shift_left32(fp_math.from_float(1.0), 5) == i32info.max
    assert fp_math.shift_left32(fp_math.from_float(-1.0), 5) == i32info.min
    assert fp_math.shift_left32(fp_math.from_float(1.0), 4) == 16 * fp_math.from_float(1.0)
    assert fp_math.shift_left32(fp_math.from_float(-1.0), 4) == 16 * fp_math.from_float(-1.0)

    assert fp_math.shift_left16(fp_math.from_float(1.0, 21), 5) == i16info.max
    assert fp_math.shift_left16(fp_math.from_float(-1.0, 21), 5) == i16info.min
    assert fp_math.shift_left16(fp_math.from_float(1.0, 21), 4) == 16 * fp_math.from_float(1.0, 21)
    assert fp_math.shift_left16(fp_math.from_float(-1.0, 21), 4) == 16 * fp_math.from_float(-1.0, 21)

    with pytest.raises(AssertionError):
        fp_math.shift_left32(1, -1)
        fp_math.shift_left16(1, -1)


def test_rounding_divide_by_pot():
    # No remainder division
    assert fp_math.rounding_divide_by_pot(fp_math.from_float(1.0), 26) == 1
    assert fp_math.rounding_divide_by_pot(fp_math.from_float(-1.0), 26) == -1

    # Remainder rounding the result away from zero
    assert fp_math.rounding_divide_by_pot(fp_math.from_float(-1.0), 27) == -1
    assert fp_math.rounding_divide_by_pot(fp_math.from_float(1.0), 27) == 1

    # Remainder smaller than threshold to round the result away from zero
    # Positive and negative edge cases
    assert fp_math.rounding_divide_by_pot(fp_math.from_float(1.0) - 1, 27) == 0
    assert fp_math.rounding_divide_by_pot(fp_math.from_float(-1.0) + 1, 27) == 0
    # Far from the edge
    assert fp_math.rounding_divide_by_pot(fp_math.from_float(1.0), 28) == 0
    assert fp_math.rounding_divide_by_pot(fp_math.from_float(-1.0), 28) == 0

    # Regular division - no remainder
    assert fp_math.rounding_divide_by_pot(fp_math.from_float(1.0), 4) == fp_math.from_float(1.0 / 16)
    assert fp_math.rounding_divide_by_pot(fp_math.from_float(-1.0), 4) == fp_math.from_float(-1.0 / 16)

    # Rounding/no rounding edge cases
    assert fp_math.rounding_divide_by_pot(fp_math.from_float(1.0) + (1 << 3) - 1, 4) == fp_math.from_float(1.0 / 16)
    assert fp_math.rounding_divide_by_pot(fp_math.from_float(1.0) + (1 << 3), 4) == fp_math.from_float(1.0 / 16) + 1
    assert fp_math.rounding_divide_by_pot(fp_math.from_float(-1.0) - (1 << 3) + 1, 4) == fp_math.from_float(-1.0 / 16)
    assert fp_math.rounding_divide_by_pot(fp_math.from_float(-1.0) - (1 << 3), 4) == fp_math.from_float(-1.0 / 16) - 1


def test_saturating_rounding_multiply_by_pot():
    i32info = np.iinfo(np.int32)
    assert fp_math.saturating_rounding_multiply_by_pot(fp_math.from_float(1.0), 5) == i32info.max
    assert fp_math.saturating_rounding_multiply_by_pot(fp_math.from_float(-1.0), 5) == i32info.min
    assert fp_math.saturating_rounding_multiply_by_pot(fp_math.from_float(1.0) - 1, 5) == i32info.max - 32 + 1
    assert fp_math.saturating_rounding_multiply_by_pot(fp_math.from_float(-1.0) + 1, 5) == -i32info.max + 32 - 1
    assert fp_math.saturating_rounding_multiply_by_pot(fp_math.from_float(1.0), 4) == fp_math.from_float(1.0 * 16)
    assert fp_math.saturating_rounding_multiply_by_pot(fp_math.from_float(-1.0), 4) == fp_math.from_float(-1.0 * 16)


def test_rescale():
    assert fp_math.rescale(5, 0, fp_math.from_float(1.0)) == fp_math.from_float(1.0, 0)
    assert fp_math.rescale(5, 10, fp_math.from_float(1.0)) == fp_math.from_float(1.0, 10)
    assert fp_math.rescale(5, 0, fp_math.from_float(-1.0)) == fp_math.from_float(-1.0, 0)
    assert fp_math.rescale(5, 10, fp_math.from_float(-1.0)) == fp_math.from_float(-1.0, 10)

    assert fp_math.rescale(5, 4, fp_math.from_float(32.0)) == fp_math.from_float(32.0, 4)
    assert fp_math.rescale(5, 6, fp_math.from_float(32.0)) == fp_math.from_float(32.0, 6)
    assert fp_math.rescale(5, 4, fp_math.from_float(-32.0)) == fp_math.from_float(-32.0, 4)
    assert fp_math.rescale(5, 6, fp_math.from_float(-32.0)) == fp_math.from_float(-32.0, 6)

    assert fp_math.rescale(5, 4, fp_math.from_float(31.9)) == fp_math.from_float(31.9, 4)
    assert fp_math.rescale(5, 6, fp_math.from_float(31.9)) == fp_math.from_float(31.9, 6)
    assert fp_math.rescale(5, 4, fp_math.from_float(-31.9)) == fp_math.from_float(-31.9, 4)
    assert fp_math.rescale(5, 6, fp_math.from_float(-31.9)) == fp_math.from_float(-31.9, 6)


def test_exp():
    sm = SoftMax(None)
    for (expected, actual) in zip(EXP_LUT, sm.generate_exp_table(1.0, np.float32(0.05123165))):
        assert actual == expected


multiply_test_data = [
    (0, 0, 0),
    (0, 0.7, 0),
    (0, 55.8, 0),
    (6, 0.3, 2),
    (200, 0, 0),
    (1, 1, 1),
    (1, 0.1, 0),
    (1, 3.49, 3),
    (1, 3.51, 4),
    (27, 1, 27),
    (13, 0.9, 12),
    (3, 21.2, 64),
    (1000, 2000, 2000000),
    (32767, 32767, 32767 * 32767),  # extreme values
]


@pytest.mark.parametrize("x, factor, expected", multiply_test_data)
def test_multiply_by_quantized_multiplier(x, factor, expected):
    scale, shift = scaling.quantise_scale(factor)
    assert fp_math.multiply_by_quantized_multiplier(x, scale, shift) == expected
    assert fp_math.multiply_by_quantized_multiplier(-x, scale, shift) == -expected
    assert fp_math.multiply_by_quantized_multiplier(x, -scale, shift) == -expected
    assert fp_math.multiply_by_quantized_multiplier(-x, -scale, shift) == expected


def test_multiply_by_quantized_multiplier_int16_limits():
    # Tests min/max limits of foreseen practical usage of multiply_by_quantized_multiplier
    # for the purpose of calculating LUTs
    for x in [-32768, 32767]:
        for y in [-32768, 32767]:
            scale, shift = scaling.quantise_scale(y)
            assert fp_math.multiply_by_quantized_multiplier(x, scale, shift) == x * y
