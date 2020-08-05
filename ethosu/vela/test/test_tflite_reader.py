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
# Description:
# Contains unit tests for tflite_reader
import pytest

from ethosu.vela.tflite_reader import TFLiteSubgraph


class TestTFLiteSubgraph:

    # Generate some data for testing len1_array_to_scalar
    len1_testdata = [
        (0, None),
        pytest.param(1, None, marks=pytest.mark.xfail),
        ([1, 2, 3], [1, 2, 3]),
        ([10], 10),
        ([], []),
    ]

    @pytest.mark.parametrize("test_input,expected", len1_testdata)
    def test_len1_array_to_scalar(self, test_input, expected):
        output = TFLiteSubgraph.len1_array_to_scalar(test_input)
        assert output == expected
