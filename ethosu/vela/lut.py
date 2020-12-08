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
# Functionality for lookup table support.
import uuid

import numpy as np

from . import numeric_util
from .high_level_command_stream import DMA
from .high_level_command_stream import NpuStripe
from .tensor import create_const_tensor
from .tensor import create_equivalence_id
from .tensor import TensorPurpose


class LUTState:
    # Tracks which LUT-s are located in SHRAM.
    def __init__(self):
        self.tensors = []

    def get_equivalent(self, lut_tens):
        # Returns existing lut with the same values, None if not found
        for t in self.tensors:
            if np.array_equal(t.values, lut_tens.values):
                return t
        return None

    def put(self, lut_tens):
        # Returns new LUT state containing given tensor + all tensors in this state
        # that do not overlap with the given tensor
        new_state = LUTState()
        new_state.tensors.append(lut_tens)
        start = lut_tens.address
        end = start + lut_tens.storage_size()
        for tens in self.tensors:
            start2 = tens.address
            end2 = start2 + tens.storage_size()
            if not numeric_util.overlaps(start, end, start2, end2):
                new_state.tensors.append(tens)

        return new_state

    def find_best_address(self, start, stop, step):
        # Finds the address in the given range that overlaps with the minimum number of
        # currently present LUT-s.
        # An improvement would be to also take future LUT usage into account
        best_addr = start
        best_nr_overlaps = stop
        for addr in range(start, stop, step):
            nr_overlaps = 0
            for tens in self.tensors:
                start2 = tens.address
                end2 = start2 + tens.storage_size()
                if numeric_util.overlaps(addr, addr + step, start2, end2):
                    nr_overlaps += 1
            if nr_overlaps < best_nr_overlaps:
                best_nr_overlaps = nr_overlaps
                best_addr = addr
        return best_addr


def get_lut_index(arch, lut_tensor):
    # Returns the index in SHRAM where the given LUT is stored, a value between 0 and 8
    slot = (lut_tensor.address - arch.shram_lut_address) // lut_tensor.storage_size()
    assert 0 <= slot < 8
    return slot


def create_lut_tensor(name, values, dtype):
    # Creates constant LUT tensor with the given values as lookup table.
    # The tensor's equivalence_id is based on these values, so if multiple
    # LUT tensors are created with identical values, they will get the same
    # address in constant memory, and unnecessary DMA operations can be avoided.
    sz = len(values)
    assert sz in (256, 512)
    ntype = np.uint8 if dtype.size_in_bytes() == 1 else np.uint32
    tens = create_const_tensor(name, [1, 1, 1, sz], dtype, values, ntype, TensorPurpose.LUT)
    tens.equivalence_id = create_equivalence_id(tuple(values))
    return tens


def optimize_high_level_cmd_stream(sg, arch):
    # - Allocates SHRAM address/lut index to LUT tensors
    # - Removes unnecessary DMA operations of LUT-s that are already present in SHRAM from sg's command stream
    cmd_stream = []  # will contain existing command stream minus unneeded DMA operations
    lut_state = LUTState()
    slot_size = 256
    lut_start = arch.shram_lut_address
    lut_end = lut_start + arch.shram_lut_size
    for cmd in sg.high_level_command_stream:
        if isinstance(cmd, NpuStripe) and cmd.ps.lut_tensor is None and arch.shram_reserved_unused_banks == 0:
            # The command overwrites the last 2 banks containing the LUT; next LUT operation will require DMA
            # TODO: check the command's SHRAM usage in more detail to determine if the LUT is overwritten or not
            lut_state = LUTState()
        if not isinstance(cmd, DMA) or cmd.out_tensor.purpose != TensorPurpose.LUT:
            # Non-LUT operation; leave untouched
            cmd_stream.append(cmd)
            continue
        # LUT DMA operation
        lut_tens = cmd.out_tensor
        existing_tens = lut_state.get_equivalent(lut_tens)
        if existing_tens is not None:
            # LUT is already in SHRAM, no need to perform DMA
            lut_tens.address = existing_tens.address
            cmd.ps.primary_op.activation.lut_index = get_lut_index(arch, existing_tens)
            continue
        # Place the LUT in the last 2 blocks of SHRAM
        # Alignment is always on the size of the LUT, 256 for 256-byte LUT, 1K for 1K LUT, etc
        address = lut_state.find_best_address(lut_start, lut_end, lut_tens.storage_size())
        lut_tens.equivalence_id = uuid.uuid4()
        lut_tens.address = address
        cmd.ps.primary_op.activation.lut_index = (address - lut_start) // slot_size
        lut_state = lut_state.put(lut_tens)
        cmd_stream.append(cmd)
    sg.high_level_command_stream = cmd_stream
