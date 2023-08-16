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

from . import fp_math
from . import numeric_util
from .data_type import DataType
from .debug_database import DebugDatabase
from .high_level_command_stream import DMA
from .high_level_command_stream import NpuStripe
from .numeric_util import round_away_zero
from .operation import Op
from .scaling import quantise_scale
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


def create_lut_rsqrt_int8_op(op):
    # Turn off black formatting for the LUT tables to keep them compact
    # fmt: off

    # RSQRT_LUT has been generated by printing the output from the reference.
    # These values are always the same but for some unknown reason it is not being
    # implemented as a LUT in the reference.
    # So based on the input range (-128, 127) the reference produces the following output:
    RSQRT_LUT = [
        0x00000000, 0x00100000, 0x000b504e, 0x00093cd4, 0x00080000, 0x000727c9, 0x0006882f, 0x00060c24,
        0x0005a827, 0x00055555, 0x00050f45, 0x0004d2fe, 0x00049e6a, 0x00047007, 0x000446b4, 0x00042195,
        0x00040000, 0x0003e16d, 0x0003c570, 0x0003abb0, 0x000393e5, 0x00037dd2, 0x00036945, 0x00035613,
        0x00034418, 0x00033333, 0x0003234b, 0x00031447, 0x00030612, 0x0002f89c, 0x0002ebd3, 0x0002dfaa,
        0x0002d414, 0x0002c906, 0x0002be75, 0x0002b45a, 0x0002aaab, 0x0002a161, 0x00029875, 0x00028fe3,
        0x000287a2, 0x00027fb0, 0x00027807, 0x000270a2, 0x0002697f, 0x00026298, 0x00025bec, 0x00025577,
        0x00024f35, 0x00024925, 0x00024343, 0x00023d8e, 0x00023803, 0x000232a1, 0x00022d65, 0x0002284e,
        0x0002235a, 0x00021e87, 0x000219d5, 0x00021541, 0x000210cb, 0x00020c70, 0x00020831, 0x0002040c,
        0x00020000, 0x0001fc0c, 0x0001f82f, 0x0001f468, 0x0001f0b7, 0x0001ed1a, 0x0001e991, 0x0001e61b,
        0x0001e2b8, 0x0001df67, 0x0001dc26, 0x0001d8f7, 0x0001d5d8, 0x0001d2c8, 0x0001cfc8, 0x0001ccd6,
        0x0001c9f2, 0x0001c71c, 0x0001c454, 0x0001c198, 0x0001bee9, 0x0001bc46, 0x0001b9af, 0x0001b723,
        0x0001b4a3, 0x0001b22d, 0x0001afc2, 0x0001ad61, 0x0001ab0a, 0x0001a8bc, 0x0001a678, 0x0001a43e,
        0x0001a20c, 0x00019fe3, 0x00019dc2, 0x00019baa, 0x0001999a, 0x00019791, 0x00019590, 0x00019397,
        0x000191a5, 0x00018fbb, 0x00018dd7, 0x00018bfa, 0x00018a23, 0x00018853, 0x0001868a, 0x000184c6,
        0x00018309, 0x00018152, 0x00017fa0, 0x00017df4, 0x00017c4e, 0x00017aad, 0x00017911, 0x0001777b,
        0x000175e9, 0x0001745d, 0x000172d6, 0x00017153, 0x00016fd5, 0x00016e5b, 0x00016ce7, 0x00016b76,
        0x00016a0a, 0x000168a2, 0x0001673e, 0x000165de, 0x00016483, 0x0001632b, 0x000161d7, 0x00016087,
        0x00015f3b, 0x00015df2, 0x00015cad, 0x00015b6b, 0x00015a2d, 0x000158f2, 0x000157bb, 0x00015686,
        0x00015555, 0x00015427, 0x000152fd, 0x000151d5, 0x000150b0, 0x00014f8f, 0x00014e70, 0x00014d54,
        0x00014c3b, 0x00014b24, 0x00014a11, 0x00014900, 0x000147f1, 0x000146e5, 0x000145dc, 0x000144d5,
        0x000143d1, 0x000142cf, 0x000141d0, 0x000140d3, 0x00013fd8, 0x00013ee0, 0x00013de9, 0x00013cf5,
        0x00013c03, 0x00013b14, 0x00013a26, 0x0001393b, 0x00013851, 0x0001376a, 0x00013684, 0x000135a1,
        0x000134bf, 0x000133e0, 0x00013302, 0x00013226, 0x0001314c, 0x00013074, 0x00012f9e, 0x00012ec9,
        0x00012df6, 0x00012d25, 0x00012c55, 0x00012b87, 0x00012abb, 0x000129f1, 0x00012928, 0x00012860,
        0x0001279a, 0x000126d6, 0x00012613, 0x00012552, 0x00012492, 0x000123d4, 0x00012317, 0x0001225c,
        0x000121a2, 0x000120e9, 0x00012032, 0x00011f7c, 0x00011ec7, 0x00011e14, 0x00011d62, 0x00011cb1,
        0x00011c02, 0x00011b54, 0x00011aa7, 0x000119fb, 0x00011950, 0x000118a7, 0x000117ff, 0x00011758,
        0x000116b3, 0x0001160e, 0x0001156b, 0x000114c8, 0x00011427, 0x00011387, 0x000112e8, 0x0001124a,
        0x000111ad, 0x00011111, 0x00011076, 0x00010fdc, 0x00010f44, 0x00010eac, 0x00010e15, 0x00010d7f,
        0x00010cea, 0x00010c56, 0x00010bc4, 0x00010b32, 0x00010aa0, 0x00010a10, 0x00010981, 0x000108f3,
        0x00010865, 0x000107d9, 0x0001074d, 0x000106c2, 0x00010638, 0x000105af, 0x00010527, 0x0001049f,
        0x00010419, 0x00010393, 0x0001030e, 0x0001028a, 0x00010206, 0x00010183, 0x00010102, 0x00010080
    ]

    # Transform the above LUT so it gets the correct quantization (following the reference)
    ifm_scale = op.ifm.quantization.scale_f32
    ofm_scale = op.ofm.quantization.scale_f32
    zp_in = op.ifm.quantization.zero_point
    zp_out = op.ofm.quantization.zero_point

    scale = np.double(1) / np.double(np.sqrt(ifm_scale) * ofm_scale)
    output_multiplier, output_shift = quantise_scale(scale)

    # Shift modification (value used in reference but Vela has opposite sign)
    kshift = -20

    ix = range(-128, 128)
    quantized_min = min(ix)
    quantized_max = max(ix)

    # Any value close to 0 (zero index in LUT) is mapped to the max output value
    values = [quantized_max]
    for x in ix:
        if x == -128:
            # Value already populated above
            continue
        # Rsqrt is only defined for positive values
        x_real = max(0, x - zp_in)
        val = RSQRT_LUT[x_real]
        val = fp_math.multiply_by_quantized_multiplier(val, output_multiplier, output_shift - kshift) + zp_out
        lut_result = min(quantized_max, max(quantized_min, val))
        values.append(lut_result)

    return convert_to_lut(op, values, "rsqrt")
