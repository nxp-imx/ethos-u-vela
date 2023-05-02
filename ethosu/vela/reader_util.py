# SPDX-FileCopyrightText: Copyright 2021-2023 Arm Limited and/or its affiliates <open-source-office@arm.com>
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
# Utlity function for reading .tosa and .tflite files
from .operation import Op
from .operation import Operation


def decode_str(s):
    if s is None:
        return ""
    return s.decode("utf-8")


def clone_and_reshape_tensor(src_tens, reorder, set_unique):
    tens = src_tens.clone("_reshape", set_unique)
    if reorder is None:
        # reorder of None is a special case meaning 1D shape requested
        tens.as_1D()
    else:
        tens.transpose(reorder)

    tens.bandwidth_shape = tens.shape
    tens.storage_shape = tens.shape

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

        op = Operation(Op.Placeholder if tens.values is None else Op.Const, tens.name)
        op.set_output_tensor(tens)

    for tens in tensors:
        if not tens.ops:
            op = Operation(Op.Placeholder if tens.is_variable else Op.Const, tens.name)
            op.set_output_tensor(tens)


def align_inputs_indices(from_indices, to_indices, inputs):
    to_list = to_indices.ifms + to_indices.weights + to_indices.biases
    from_list = from_indices.ifms + from_indices.weights + from_indices.biases

    assert len(to_list) == len(from_list)
    if to_list != from_list:
        for idx, t_idx in enumerate(to_list):
            if t_idx >= len(inputs):
                # Biases are allowed to be left out
                assert t_idx in from_indices.biases and t_idx in to_indices.biases
                continue
            if to_list[idx] != from_list[idx]:
                # find t_idx in from list and swap.
                for jdx in from_list[idx:]:
                    if from_list[jdx] == t_idx:
                        inputs[idx], inputs[jdx] = inputs[jdx], inputs[idx]
                        from_list[idx], from_list[jdx] = from_list[jdx], from_list[idx]
                        break
    assert from_list == to_list
    return inputs


def align_tensor_indices_to_nng(op_type, indices, inputs):
    nng_op = Op(op_type)
    return align_inputs_indices(indices, nng_op.info.indices, inputs)
