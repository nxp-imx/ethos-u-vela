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
# Contains unit tests for get API version for an external consumer
from ethosu.vela.api import API_VERSION
from ethosu.vela.api import npu_get_api_version


def test_npu_get_api_version():
    int_version = npu_get_api_version()
    version_major = int_version >> 16
    version_minor = 0xFFFF & int_version
    assert API_VERSION == f"{version_major}.{version_minor}"
