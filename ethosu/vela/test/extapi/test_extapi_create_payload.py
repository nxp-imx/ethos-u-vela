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
# Contains unit tests for npu_create_driver_payload API for an external consumer
import random

import pytest

from ethosu.vela.api import npu_create_driver_payload
from ethosu.vela.api import NpuAccelerator


@pytest.mark.parametrize("accelerator", list(NpuAccelerator))
def test_create_driver_payload(accelerator: NpuAccelerator):
    """Tests npu_create_driver_payload"""
    # Generate a random command stream with defined beginning and end
    random.seed(0)
    num_commands = 793
    register_command_stream = random.choices(range(1 << 32), k=num_commands)
    register_command_stream[0] = 0xFEDCBA98
    register_command_stream[-1] = 0xA0B1C2D3
    payload = npu_create_driver_payload(register_command_stream, accelerator)
    header_size = 32  # expected driver header size in bytes
    assert len(payload) == header_size + 4 * num_commands
    # Check that the first register command is located directly after the header
    assert list(payload[header_size : header_size + 4]) == [0x98, 0xBA, 0xDC, 0xFE]
    # Check that the last register command is present in the payload
    assert list(payload[-4:]) == [0xD3, 0xC2, 0xB1, 0xA0]
