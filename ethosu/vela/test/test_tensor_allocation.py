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
# Contains unit tests for tensor allocation
import pytest

from ethosu.vela.data_type import DataType
from ethosu.vela.errors import AllocationError
from ethosu.vela.live_range import LiveRangeGraph
from ethosu.vela.tensor import Tensor
from ethosu.vela.tensor_allocation import verify_allocation


def test_verify_allocation():
    # Create live range graph with 2 live ranges with overlapping start/end time
    lr_graph = LiveRangeGraph()
    t1 = Tensor([1, 100, 10, 10], DataType.int8, "t1")
    lr1 = lr_graph.get_or_create_range(t1)
    lr1.mark_usage(4)
    lr1.mark_usage(8)
    t2 = Tensor([1, 10, 20, 10], DataType.int8, "t2")
    lr2 = lr_graph.get_or_create_range(t2)
    # Set overlapping addresses, should lead to verification failure
    lr1.set_address(16)
    lr2.set_address(32)
    lr2.mark_usage(7)
    lr2.mark_usage(12)
    with pytest.raises(AllocationError):
        verify_allocation(lr_graph, 16)
    # Set non-overlapping addresses, verification should now succeed
    lr2.set_address(None)
    lr2.set_address(160000)
    verify_allocation(lr_graph, 16)
