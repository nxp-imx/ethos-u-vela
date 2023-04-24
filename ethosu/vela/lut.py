# SPDX-FileCopyrightText: Copyright 2020-2021, 2023 Arm Limited and/or its affiliates <open-source-office@arm.com>
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
# Functionality for lookup table support.
import uuid

import numpy as np

from . import numeric_util
from .data_type import DataType
from .debug_database import DebugDatabase
from .high_level_command_stream import DMA
from .high_level_command_stream import NpuStripe
from .numeric_util import round_away_zero
from .operation import Op
from .tensor import create_const_tensor
from .tensor import create_equivalence_id
from .tensor import QuantizationParameters
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
    # int16 lut uses uint32 lut with base + slope
    dtype = DataType.uint32 if dtype == DataType.int16 else dtype
    tens = create_const_tensor(name, [1, 1, 1, sz], dtype, values, TensorPurpose.LUT)
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
            lut_tens.equivalence_id = existing_tens.equivalence_id
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


def convert_to_lut(op, lut_values, lut_name):
    # Rewrite the operation by Add with scalar 0 + LUT activation
    ifm = op.ifm
    ofm = op.ofm
    if ifm is None:
        return op
    assert ifm.dtype in (DataType.int8, DataType.uint8, DataType.int16)
    op.type = Op.Add
    op.name = f"{op.name}_lut_{lut_name}"
    # Mark as no-op to enable potential fusing optimizations
    op.attrs["is_nop"] = True
    # Create an input tensor containing scalar zero
    _max = 65536.0 if ifm.dtype == DataType.int16 else 255.0
    quantization = QuantizationParameters(0.0, _max)
    quantization.scale_f32 = ifm.quantization.scale_f32
    quantization.zero_point = 0
    tens = create_const_tensor(ifm.name + "_scalar0", [], ifm.dtype, [0], quantization=quantization)
    op.add_input_tensor(tens)

    # The LUT must be applied without any preceding rescaling (the LUT itself performs the rescale),
    # so even if the OFM has a different scale than the IFM, the generated OFM scale instructions
    # should be the same as the IFM
    op.forced_output_quantization = ifm.quantization

    # the lut tensor datatype needs to match both; the ofm datatype, because these are the values output; and the
    # datatype used to generate the lut values (which is probably the ifm datatype), because we want to avoid any
    # potential overflow errors in create_lut_tensor() caused by converting Python int (which could represent a uint)
    # to NumPy int. this can be guaranteed by checking that the ifm and ofm datatypes are the same
    assert ifm.dtype == ofm.dtype
    lut_tensor = create_lut_tensor(op.name + "_values", lut_values, ofm.dtype)
    op.set_activation_lut(lut_tensor)
    op.set_ifm_ofm_shapes()
    DebugDatabase.add_optimised(op, op)
    return op


def create_lut_8bit_op(op, lut_fn, fn_name):
    ifm_scale = op.ifm.quantization.scale_f32
    ofm_scale = op.ofm.quantization.scale_f32
    zp_in = op.ifm.quantization.zero_point
    zp_out = op.ofm.quantization.zero_point

    values = []
    ix = range(256) if op.ifm.dtype == DataType.uint8 else range(-128, 128)
    quantized_min = min(ix)
    quantized_max = max(ix)
    for x in ix:
        x_real = ifm_scale * (x - zp_in)
        y_real = lut_fn(x_real)
        lut_result = round_away_zero(y_real / ofm_scale) + zp_out
        lut_result = min(quantized_max, max(quantized_min, lut_result))
        values.append(lut_result)

    return convert_to_lut(op, values, fn_name)


def create_lut_int16_op(op, lut_fn, fn_name):
    ifm_scale = op.ifm.quantization.scale_f32
    ofm_scale = op.ofm.quantization.scale_f32
    zp_in = op.ifm.quantization.zero_point
    zp_out = op.ofm.quantization.zero_point

    input_min = ifm_scale * (np.iinfo(np.int16).min - zp_in)
    input_max = ifm_scale * (np.iinfo(np.int16).max - zp_in)
    output_min = ofm_scale * (np.iinfo(np.int16).min - zp_out)
    output_max = ofm_scale * (np.iinfo(np.int16).max - zp_out)

    # Create 16bit lut following the reference
    nbr_steps = 512
    step = (input_max - input_min) / nbr_steps
    half_step = step / 2
    output_scaling_inv = (np.iinfo(np.int16).max - np.iinfo(np.int16).min + 1) / (output_max - output_min)

    table_min = np.iinfo(np.int16).min
    table_max = np.iinfo(np.int16).max

    values = []
    for i in range(nbr_steps):
        val = lut_fn(input_min + i * step)
        val_midpoint = lut_fn(input_min + i * step + half_step)
        val_next = lut_fn(input_min + (i + 1) * step)

        sample_val = round_away_zero(val * output_scaling_inv)
        midpoint_interp_val = round_away_zero(
            (val_next * output_scaling_inv + round_away_zero(val * output_scaling_inv)) / 2
        )
        midpoint_val = round_away_zero(val_midpoint * output_scaling_inv)
        midpoint_err = midpoint_interp_val - midpoint_val
        bias = round_away_zero(midpoint_err / 2)

        lut_result = min(max(sample_val - bias, table_min), table_max)
        values.append(lut_result)

    val = round_away_zero(lut_fn(input_max) * output_scaling_inv)
    lut_result = min(max(val, table_min), table_max)
    values.append(lut_result)

    # Convert to hardware 16bit lut with base and slope
    lut = [0] * nbr_steps
    for i in range(nbr_steps):
        slope = (int(values[i + 1]) - int(values[i])) << 16
        base = int(values[i])
        lut[i] = slope + base

    return convert_to_lut(op, lut, fn_name)
