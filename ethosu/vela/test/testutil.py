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
# Utilities used in vela unit tests
import numpy as np

from ethosu.vela import architecture_features
from ethosu.vela.data_type import DataType
from ethosu.vela.nn_graph import Subgraph
from ethosu.vela.operation import NpuBlockType
from ethosu.vela.operation import Operation
from ethosu.vela.tensor import create_const_tensor
from ethosu.vela.tensor import MemArea
from ethosu.vela.tensor import Tensor


def create_arch():
    return architecture_features.ArchitectureFeatures(
        vela_config=None,
        system_config=None,
        accelerator_config=architecture_features.Accelerator.Ethos_U55_128.value,
        permanent_storage=MemArea.OnChipFlash,
        override_block_config=None,
        block_config_limit=None,
        global_memory_clock_scale=1.0,
        max_blockdep=0,
        weight_estimation_scaling=1.0,
    )


def create_elemwise_op(type, name, ifm_shape, ifm2_shape, ofm_shape, datatype=DataType.uint8):
    # Creates elementwise operation with constant IFM/IFM2
    if datatype.size_in_bytes() == 1:
        np_type = np.uint8
    elif datatype.size_in_bytes() == 2:
        np_type = np.int16
    else:
        np_type = np.int32
    op = Operation(type, name)
    op.add_input_tensor(create_const_tensor(name + "_ifm", ifm_shape, datatype, np.zeros(ifm_shape), np_type))
    op.add_input_tensor(create_const_tensor(name + "_ifm2", ifm2_shape, datatype, np.zeros(ifm2_shape), np_type))
    ofm = Tensor(ofm_shape, datatype, name + "_ofm")
    op.set_output_tensor(ofm)
    op.attrs["npu_block_type"] = NpuBlockType.ElementWise
    return op


def create_subgraph(op_list):
    # Creates subgraph using the given list of operations
    sg = Subgraph()
    all_inputs = set(tens for op in op_list for tens in op.inputs)
    # Reversing, so that the resulting subgraph has same order as op_list
    for op in op_list[::-1]:
        for tens in op.outputs:
            if tens not in all_inputs and tens not in sg.output_tensors:
                sg.output_tensors.append(tens)
    return sg
