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
from .operation import Operation
from .tensor import Tensor


class VelaError(Exception):
    """Base class for vela exceptions"""

    def __init__(self, data):
        self.data = "Error: " + data

    def __str__(self):
        return repr(self.data)


class InputFileError(VelaError):
    """Raised when reading an input file results in errors"""

    def __init__(self, file_name, msg):
        self.data = "Reading input file {}: {}".format(file_name, msg)


class UnsupportedFeatureError(VelaError):
    """Raised when the input network uses non-supported features that cannot be handled"""

    def __init__(self, data):
        self.data = "Input network uses a feature that is currently not supported: {}".format(data)


class CliOptionError(VelaError):
    """Raised for errors encountered with a command line option

    :param option: str object that contains the name of the command line option
    :param option_value: the command line option that resulted in the error
    :param msg: str object that contains a description of the specific error encountered
    """

    def __init__(self, option, option_value, msg):
        self.data = "Incorrect argument to CLI option: {} = {}: {}".format(option, option_value, msg)


class ConfigOptionError(VelaError):
    """Raised for errors encountered with a configuration option

    :param option: str object that contains the name of the configuration option
    :param option_value: the configuration option that resulted in the error
    :param option_valid_values (optional): str object that contains the valid configuration option values
    """

    def __init__(self, option, option_value, option_valid_values=None):
        self.data = "Invalid configuration of {} = {}".format(option, option_value)
        if option_valid_values is not None:
            self.data += " (must be {}).".format(option_valid_values)
        else:
            self.data += "."


class AllocationError(VelaError):
    """Raised when allocation fails"""

    def __init__(self, msg):
        self.data = msg


def OperatorError(op, msg):
    """
    Raises a VelaError exception for errors encountered when parsing an Operation

    :param op: Operation object that resulted in the error
    :param msg: str object that contains a description of the specific error encountered
    """

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

    raise VelaError(data)


def TensorError(tens, msg):
    """
    Raises a VelaError exception for errors encountered when parsing a Tensor

    :param tens: Tensor object that resulted in the error
    :param msg: str object that contains a description of the specific error encountered
    """

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

    raise VelaError(data)
