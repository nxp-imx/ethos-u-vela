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
import sys

from .operation import Operation
from .tensor import Tensor


class VelaError(Exception):
    """Base class for vela exceptions"""

    def __init__(self, data):
        self.data = data

    def __str__(self):
        return repr(self.data)


class InputFileError(VelaError):
    """Raised when reading the input file results in errors"""

    def __init__(self, file_name, msg):
        self.data = "Error reading input file {}: {}".format(file_name, msg)


class UnsupportedFeatureError(VelaError):
    """Raised when the input file uses non-supported features that cannot be handled"""

    def __init__(self, data):
        self.data = "The input file uses a feature that is currently not supported: {}".format(data)


class OptionError(VelaError):
    """Raised when an incorrect command line option is used"""

    def __init__(self, option, option_value, msg):
        self.data = "Incorrect argument to CLI option: {} {}: {}".format(option, option_value, msg)


def OperatorError(op, msg):
    """Called when parsing an operator results in errors"""

    assert isinstance(op, Operation)

    if op.op_index is None:
        data = "Invalid {} (name = {}) operator in the internal representation.".format(op.type, op.name)
    else:
        data = "Invalid {} (op_index = {}) operator in the input network.".format(op.type, op.op_index)

    data += " {}\n".format(msg)

    data += "   Input tensors:\n"
    for idx, tens in enumerate(op.inputs):
        if isinstance(tens, Tensor):
            tens_name = tens.name
        else:
            tens_name = "Not a Tensor"

        data += "      {} = {}\n".format(idx, tens_name)

    data += "   Output tensors:\n"
    for idx, tens in enumerate(op.outputs):
        if isinstance(tens, Tensor):
            tens_name = tens.name
        else:
            tens_name = "Not a Tensor"

        data += "      {} = {}\n".format(idx, tens_name)

    data = data[:-1]  # remove last newline

    print("Error: {}".format(data))
    sys.exit(1)


def TensorError(tens, msg):
    """Called when parsing a tensor results in errors"""

    assert isinstance(tens, Tensor)

    data = "Invalid {} tensor. {}\n".format(tens.name, msg)

    data += "   Driving operators:\n"
    for idx, op in enumerate(tens.ops):
        if isinstance(op, Operation):
            op_type = op.type
            op_id = op.op_index
        else:
            op_type = "Not an Operation"
            op_id = ""

        data += "      {} = {} ({})\n".format(idx, op_type, op_id)

    data += "   Consuming operators:\n"
    for idx, op in enumerate(tens.consumer_list):
        if isinstance(op, Operation):
            op_type = op.type
            op_id = op.op_index
        else:
            op_type = "Not an Operation"
            op_id = ""

        data += "      {} = {} ({})\n".format(idx, op_type, op_id)

    data = data[:-1]  # remove last newline

    print("Error: {}".format(data))
    sys.exit(1)
