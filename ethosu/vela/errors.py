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
        self.data = f"Error! {data}"

    def __str__(self):
        return repr(self.data)


class InputFileError(VelaError):
    """Raised when reading an input file results in errors"""

    def __init__(self, file_name, msg):
        super().__init__(f"Reading input file '{file_name}': {msg}")


class UnsupportedFeatureError(VelaError):
    """Raised when the input network uses non-supported features that cannot be handled"""

    def __init__(self, data):
        super().__init__(f"Input network uses a feature that is currently not supported: {data}")


class CliOptionError(VelaError):
    """Raised for errors encountered with a command line option

    :param option: str object that contains the name of the command line option
    :param option_value: the command line option that resulted in the error
    :param msg: str object that contains a description of the specific error encountered
    """

    def __init__(self, option, option_value, msg):
        super().__init__(f"Incorrect argument to CLI option {option}={option_value}: {msg}")


class ConfigOptionError(VelaError):
    """Raised for errors encountered with a configuration option

    :param option: str object that contains the name of the configuration option
    :param option_value: the configuration option that resulted in the error
    :param option_valid_values (optional): str object that contains the valid configuration option values
    """

    def __init__(self, option, option_value, option_valid_values=None):
        data = f"Invalid configuration of {option}={option_value}"
        if option_valid_values is not None:
            data += f" (must be {option_valid_values})"
        super().__init__(data)


class AllocationError(VelaError):
    """Raised when allocation fails"""

    def __init__(self, msg):
        super().__init__(f"Allocation failed: {msg}")


def OperatorError(op, msg):
    """
    Raises a VelaError exception for errors encountered when parsing an Operation

    :param op: Operation object that resulted in the error
    :param msg: str object that contains a description of the specific error encountered
    """

    def _print_tensors(tensors):
        lines = []
        for idx, tens in enumerate(tensors):
            if isinstance(tens, Tensor):
                tens_name = tens.name
            else:
                tens_name = "Not a Tensor"
            lines.append(f"        {idx} = {tens_name}")
        return lines

    assert isinstance(op, Operation)

    if op.op_index is None:
        lines = [f"Invalid {op.type} (name = {op.name}) operator in the internal representation. {msg}"]
    else:
        lines = [f"Invalid {op.type} (op_index = {op.op_index}) operator in the input network. {msg}"]

    lines += ["    Input tensors:"]
    lines += _print_tensors(op.inputs)

    lines += ["    Output tensors:"]
    lines += _print_tensors(op.outputs)

    raise VelaError("\n".join(lines))


def TensorError(tens, msg):
    """
    Raises a VelaError exception for errors encountered when parsing a Tensor

    :param tens: Tensor object that resulted in the error
    :param msg: str object that contains a description of the specific error encountered
    """

    def _print_operators(ops):
        lines = []
        for idx, op in enumerate(ops):
            if isinstance(op, Operation):
                op_type = op.type
                op_id = f"({op.op_index})"
            else:
                op_type = "Not an Operation"
                op_id = ""
            lines.append(f"        {idx} = {op_type} {op_id}")
        return lines

    assert isinstance(tens, Tensor)

    lines = [f"Invalid {tens.name} tensor. {msg}"]

    lines += ["    Driving operators:"]
    lines += _print_operators(tens.ops)

    lines += ["    Consuming operators:"]
    lines += _print_operators(tens.consumer_list)

    raise VelaError("\n".join(lines))
