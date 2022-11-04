# SPDX-FileCopyrightText: Copyright 2021 Arm Limited and/or its affiliates <open-source-office@arm.com>
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
# Unit tests for the mapping of TFLite or TOSA to NNG
import pytest

from ethosu.vela.tflite_mapping import builtin_operator_map
from ethosu.vela.tosa_mapping import tosa_operator_map


class TestNNGMapping:
    """Ensure the mappings from TFLite to NNG are consistent."""

    @pytest.mark.parametrize(
        "operator_map",
        ((builtin_operator_map), (tosa_operator_map)),
        ids=("test_tflite_indices_match_nng", "test_tosa_indices_match_nng"),
    )
    def test_op_indices_match(self, operator_map):
        """Ensure TFLite/TOSA indices and NNG indices are consistent for each operator."""
        for map_op in operator_map.values():
            op_type = map_op[0]
            map_op_indices = map_op[-1]  # TFLite/TOSA indices in last element of tuple

            nng_indices = op_type.info.indices

            for idx in range(3):
                assert len(map_op_indices[idx]) == len(nng_indices[idx])
