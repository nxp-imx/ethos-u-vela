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
# Defines custom exceptions.


class VelaError(Exception):
    """Base class for vela exceptions"""

    def __init__(self, data):
        self.data = data

    def __str__(self):
        return repr(self.data)


class InputFileError(VelaError):
    """Raised when reading the input file results in errors"""

    def __init__(self, file_name, msg):
        self.data = "Error reading {}: {}".format(file_name, msg)


class UnsupportedFeatureError(VelaError):
    """Raised when the input file uses non-supported features that cannot be handled"""

    def __init__(self, data):
        self.data = "The input file uses a feature that is currently not supported: {}".format(data)


class OptionError(VelaError):
    """Raised when an incorrect command line option is used"""

    def __init__(self, option, option_value, msg):
        self.data = "Incorrect argument: {} {}: {}".format(option, option_value, msg)
