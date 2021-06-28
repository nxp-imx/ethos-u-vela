# Copyright (C) 2021 Arm Limited or its affiliates. All rights reserved.
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
# Utlity function for reading .tosa and .tflite files
from .operation import Op
from .operation import Operation


def decode_str(s):
    if s is None:
        return ""
    return s.decode("utf-8")


def clone_and_reshape_tensor(src_tens, reorder, set_unique):
    tens = src_tens.clone("_reshape", set_unique)
    tens.shape = [src_tens.shape[idx] for idx in reorder]
    tens.bandwidth_shape = tens.shape
    tens.storage_shape = tens.shape

    if tens.values is not None:
        tens.values = tens.values.transpose(reorder)

    if tens.quant_values is not None:
        tens.quant_values = tens.quant_values.transpose(reorder)

    op = Operation(Op.Const, tens.name)
    op.set_output_tensor(tens)
    return tens


# Fix up tensors without operations. Generate either Placeholder or Constant ops
def fixup_tensors(input_tensors, tensors):
    for tens in input_tensors:
        if len(tens.ops) and tens.ops[0].type == Op.Const:
            break

        if tens.ops != []:
            tens.error("This subgraph input tensor has unexpected driving operators.")

        op = Operation(Op.Placeholder, tens.name)
        op.set_output_tensor(tens)

    for tens in tensors:
        if not tens.ops:
            op = Operation(Op.Const, tens.name)
            op.set_output_tensor(tens)
