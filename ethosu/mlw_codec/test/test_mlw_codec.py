#!/usr/bin/env python3
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
# Simple example of the usage of mlw_codec.
from typing import Any
from typing import List

import pytest

from ethosu import mlw_codec


class TestMLWCodec:
    """This class is responsible to test the mlw_codec library
    It mainly tests the two methods encode() and decode() with different inputs"""

    weights = [0, 2, 3, 0, -1, -2, -3, 0, 0, 0, 1, -250, 240] * 3
    compressed_weights = bytearray(
        b"\xb8\x00\\q^\x1f\xfc\x01\x03\x05\x08\x0c\x10\x908\x12\xd7\x99:\xd2\x99$\xae#\x9d\xa9#\x00\xf0\xff\xff\xff"
    )
    empty_decoded = bytearray(b"\xfe\xffC\x00\xf0\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff")

    # Generate parameters lists for the tests below
    encode_testdata = [
        (mlw_codec.encode, weights, compressed_weights),
        pytest.param(mlw_codec.encode, ["a"], empty_decoded, marks=pytest.mark.xfail),  # cannot accept strings
    ]

    decode_testdata = [(mlw_codec.decode, compressed_weights, weights)]

    codec_testdata = [
        (weights, weights),
        ([1] * 10, [1] * 10),
        pytest.param(["a"], ["a"], marks=pytest.mark.xfail),  # cannot accept strings
    ]

    @pytest.mark.parametrize("function_under_test,test_input,expected", encode_testdata)
    def test_mlw_codec(self, function_under_test, test_input, expected):
        self._call_mlw_codec_method(function_under_test, test_input, expected)

    @pytest.mark.parametrize("function_under_test,test_input,expected", decode_testdata)
    def test_mlw_decode(self, function_under_test, test_input, expected):
        self._call_mlw_codec_method(function_under_test, test_input, expected)

    @pytest.mark.parametrize("test_input,expected", codec_testdata)
    def test_mlw_encode_decode(self, test_input, expected):
        output = mlw_codec.decode(mlw_codec.encode(test_input))
        assert output == expected

    def _call_mlw_codec_method(self, method_name, test_input, expected):
        output = method_name(test_input)
        assert output == expected

    invalid_encode_test_data = [None, 3, [4, 5, None, 7], [0, 1, "two", 3], [1, 2, 256, 4], [2, 4, 8, -256]]

    @pytest.mark.parametrize("input", invalid_encode_test_data)
    def test_encode_invalid_input(self, input):
        with pytest.raises(Exception):
            mlw_codec.encode(input)

    invalid_decode_test_data: List[Any] = [None, 3, []]

    @pytest.mark.parametrize("input", invalid_decode_test_data)
    def test_decode_invalid_input(self, input):
        with pytest.raises(Exception):
            mlw_codec.decode(input)
