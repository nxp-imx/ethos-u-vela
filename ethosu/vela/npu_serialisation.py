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
# Serialises and packs an NPU subgraph into tensors.
import numpy as np

from . import driver_actions
from .data_type import DataType
from .nn_graph import PassPlacement
from .operation import Op
from .operation import Operation
from .tensor import MemArea
from .tensor import MemType
from .tensor import Tensor
from .tensor import TensorFormat
from .tensor import TensorPurpose


def make_memory_tensor(name, mem_area, mem_type, sz, want_values, arch):
    tens = Tensor([sz], DataType.uint8, name)
    tens.mem_area = mem_area
    tens.mem_type = mem_type
    tens.purpose = TensorPurpose.FeatureMap
    tens.set_format(TensorFormat.NHWC, arch)
    if want_values:
        tens.values = np.zeros(tens.shape, np.uint8)
    return tens


def copy_compressed_values_to_memory_tensor(memory_tensor, src_tensor):
    start_addr = src_tensor.address
    end_addr = src_tensor.address + src_tensor.storage_size()
    memory_tensor.values[start_addr:end_addr] = src_tensor.buffer.copy()


def copy_ifm_values_to_memory_tensor(memory_tensor, src_tensor):
    start_addr = src_tensor.address
    values = src_tensor.values.flatten()
    if src_tensor.dtype.size_in_bytes() > 1:
        values = np.frombuffer(values.tobytes(), dtype=np.uint8)
    end_addr = start_addr + values.size
    memory_tensor.values[start_addr:end_addr] = values


def serialise_npu_subgraph_into_tensors(sg, arch, scratch_tens, scratch_fast_tens, flash_tens):
    if sg.placement != PassPlacement.Npu:
        return scratch_tens, scratch_fast_tens, flash_tens

    flash_area = arch.permanent_storage_mem_area
    scratch_area = arch.feature_map_storage_mem_area
    scratch_fast_area = arch.fast_storage_mem_area

    flash_size = sg.memory_used.get(flash_area, 0)
    scratch_size = sg.memory_used.get(scratch_area, 0)

    payload_bytes = driver_actions.create_driver_payload(sg.register_command_stream, arch)

    command_stream_size_bytes = len(payload_bytes)

    if flash_tens == scratch_tens is None:
        # First Npu subgraph, create scratch and flash tensors
        sg.scratch_tensor = make_memory_tensor(
            sg.name + "_scratch", scratch_area, MemType.Scratch, scratch_size, False, arch
        )
        sg.scratch_tensor.purpose = TensorPurpose.Scratch
        sg.flash_tensor = make_memory_tensor(
            sg.name + "_flash", flash_area, MemType.Permanent_CPU, flash_size, True, arch
        )
        sg.scratch_fast_tensor = make_memory_tensor(
            sg.name + "_scratch_fast", scratch_fast_area, MemType.Scratch_fast, 0, False, arch
        )
        sg.scratch_fast_tensor.purpose = TensorPurpose.ScratchFast
    else:
        sg.scratch_tensor = scratch_tens
        sg.scratch_tensor.shape[0] += scratch_size
        sg.flash_tensor = flash_tens
        sg.flash_tensor.shape[0] += flash_size

        sg.scratch_fast_tensor = scratch_fast_tens
        sg.scratch_fast_tensor.shape[0] = 0

    for sched_op in sg.sched_ops:
        ifm_tensor, ifm2_tensor, _, _, _ = sched_op.parent_op.get_ifm_ifm2_weights_biases_ofm()

        op_info = sg.schedule.cost_map[sched_op]
        if op_info.npu_weights_tensor:
            copy_compressed_values_to_memory_tensor(sg.flash_tensor, op_info.npu_weights_tensor)
        if op_info.npu_scales_tensor:
            copy_compressed_values_to_memory_tensor(sg.flash_tensor, op_info.npu_scales_tensor)

        if ifm_tensor and ifm_tensor.mem_type not in (MemType.Scratch, MemType.Scratch_fast):
            copy_ifm_values_to_memory_tensor(sg.flash_tensor, ifm_tensor)
        if ifm2_tensor and (ifm2_tensor.mem_type not in (MemType.Scratch, MemType.Scratch_fast)):
            copy_ifm_values_to_memory_tensor(sg.flash_tensor, ifm2_tensor)

        if sched_op.parent_op.activation_lut:
            copy_ifm_values_to_memory_tensor(sg.flash_tensor, sched_op.parent_ps.lut_tensor)

    sg.command_stream_tensor = make_memory_tensor(
        sg.name + "_command_stream", flash_area, MemType.Permanent_CPU, command_stream_size_bytes, True, arch
    )
    sg.command_stream_tensor.values = np.frombuffer(payload_bytes, dtype=np.uint8)

    return sg.scratch_tensor, sg.scratch_fast_tensor, sg.flash_tensor


def add_const_tens_to_startup_cascaded_pass(startup_cps, tens):
    op = Operation(Op.Const, tens.name + "_const")
    op.set_output_tensor(tens)
    startup_cps.passes[0].ops.insert(0, op)
    startup_cps.passes[0].outputs.insert(0, tens)
    startup_cps.outputs.insert(0, tens)


def rewrite_npu_call_ops(sg, arch):
    if sg.placement != PassPlacement.Cpu:
        return

    startup_cps = sg.cascaded_passes[0]

    for idx, cps in enumerate(sg.cascaded_passes):
        for ps in cps.passes:
            for op in ps.ops:
                if op.type == Op.CustomNpuOp:
                    callee = op.attrs["subgraph"]

                    sz = 0
                    for tens in [
                        callee.scratch_fast_tensor,
                        callee.scratch_tensor,
                        callee.flash_tensor,
                        callee.command_stream_tensor,
                    ]:
                        op.inputs.insert(0, tens)
                        ps.inputs.insert(0, tens)
                        cps.inputs.insert(0, tens)
                        if tens != callee.scratch_tensor and tens != callee.scratch_fast_tensor:
                            add_const_tens_to_startup_cascaded_pass(startup_cps, tens)
                        sz += tens.storage_size()

                    for prev_cps in sg.cascaded_passes[: idx + 1]:
                        prev_cps.sram_used += sz

                    if callee.scratch_tensor is not None:
                        if callee.scratch_tensor.mem_area == MemArea.Sram:
                            cps.sram_used += callee.scratch_tensor.storage_size()

                    if callee.scratch_fast_tensor is not None:
                        if callee.scratch_fast_tensor.mem_area == MemArea.Sram:
                            cps.sram_used += callee.scratch_fast_tensor.storage_size()
