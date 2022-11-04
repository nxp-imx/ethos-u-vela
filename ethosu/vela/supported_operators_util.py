# SPDX-FileCopyrightText: Copyright 2020-2021 Arm Limited and/or its affiliates <open-source-office@arm.com>
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
# Utility functions used in supported operator checking


def list_formatter(arg):
    # Order and join into a string representation
    return ", ".join(sorted(map(str, arg)))


# Custom decorator function to allow formatting docstrings containing "{}"
def docstring_format_args(args):
    def docstring(func):
        func.__doc__ = func.__doc__.format(*args)
        return func

    return docstring
