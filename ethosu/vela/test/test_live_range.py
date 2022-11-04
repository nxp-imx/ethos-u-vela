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
# Contains unit tests for live ranges
from unittest.mock import MagicMock

import pytest

from ethosu.vela.live_range import LiveRange
from ethosu.vela.tensor import Tensor


class TestLiveRange:
    def test_instantiate_live_range_with_tensor(self):
        tens = MagicMock()
        tens.storage_size.return_value = 4
        tens.name = "test"

        live_range = LiveRange(tens, Tensor.AllocationQuantum)
        assert live_range.size == 4
        assert live_range.name == "test"
        assert live_range.tensors == [tens]

    def test_add_tensor_valid_size(self):
        tens = MagicMock()
        # When storage_size() is called twice, it returns 4 and then 3
        tens.storage_size.side_effect = [4, 3]
        tens.name = "test"

        live_range = LiveRange(tens, Tensor.AllocationQuantum)
        live_range.add_tensor(tens)

        assert live_range.size == 4
        assert live_range.name == "test"
        assert live_range.tensors == [tens, tens]

    def test_add_tensor_invalid_size(self):
        tens = MagicMock()
        # When storage_size() is called twice, it returns 4 and then 5
        tens.storage_size.side_effect = [4, 5]
        tens.name = "test"

        live_range = LiveRange(tens, Tensor.AllocationQuantum)
        # Expect an AssertionError with a message
        with pytest.raises(AssertionError, match=r".* to the same LiveRange .*"):
            live_range.add_tensor(tens)

        # Check that the interal status of the object didn't change
        assert live_range.size == 4
        assert live_range.name == "test"
        assert live_range.tensors == [tens]
