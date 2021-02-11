# Copyright (C) 2020-2021 Arm Limited or its affiliates. All rights reserved.
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
# Shared buffer allocation works out how to allocate the Ethos-U shared buffer for a given pass.
from typing import List
from typing import Tuple

import numpy as np

from .api import NpuActivationOp
from .api import NpuBlockOperation
from .architecture_features import ArchitectureFeatures
from .architecture_features import Block
from .architecture_features import SharedBufferArea
from .architecture_features import SHRAMElements
from .errors import AllocationError
from .ethos_u55_regs.ethos_u55_regs import resampling_mode
from .operation import Kernel
from .operation import NpuBlockType
from .range_set import MemoryRangeSet
from .register_command_stream_util import to_kernel
from .shape4d import Shape4D
from .tensor import MemArea


class SharedBufferAllocation:
    def __init__(
        self,
        arch,
        kernel,
        uses_lut,
        npu_block_type,
        all_fms_have_quant,
        ifm_resampling_mode,
        ifm_bits,
        ifm_depth,
        ifm_count,
        ofm_shape,
    ):
        self.arch = arch

        self.bank_locations = np.zeros(SharedBufferArea.Size)
        self.banks_required = np.zeros(SharedBufferArea.Size)

        self.kernel = Kernel(1, 1) if kernel is None else kernel
        self.is_elementwise = npu_block_type == NpuBlockType.ElementWise
        self.uses_lut = uses_lut
        self.ifm_count = ifm_count

        self.is_equal_depth_op = self.is_elementwise or npu_block_type in (
            NpuBlockType.ConvolutionDepthWise,
            NpuBlockType.Pooling,
        )

        self.use_accumulator_element = SHRAMElements.Acc32
        if self.is_elementwise:
            self.use_ifm_element = SHRAMElements.IFM8_Elementwise
        else:
            self.use_ifm_element = SHRAMElements.IFM8

        self.ifm_resampling_mode = ifm_resampling_mode
        self.ifm_bits = ifm_bits
        self.ifm_depth = ifm_depth
        self.ifm_count = ifm_count

        if self.ifm_bits == 16:
            if npu_block_type != NpuBlockType.Pooling and all_fms_have_quant:
                self.use_accumulator_element = SHRAMElements.Acc40
            self.use_ifm_element = self.use_ifm_element + 1
            assert (self.use_ifm_element == SHRAMElements.IFM16) or (
                self.use_ifm_element == SHRAMElements.IFM16_Elementwise
            )
        elif self.ifm_bits == 32:
            assert self.is_elementwise or npu_block_type == NpuBlockType.ReduceSum, "Unsupported 32-bit IFM operation"
            self.use_ifm_element = SHRAMElements.IFM32
        else:
            assert self.ifm_bits == 8, "Unexpected IFM bitdepth"

        self.ifm_block_depth = arch.calc_ifm_block_depth(self.ifm_depth, self.ifm_bits)
        self.ofm_shape = ofm_shape

        self.banks_required[SharedBufferArea.Weights] = arch.shram_reserved_weight_banks
        self.banks_required[SharedBufferArea.OFM] = arch.shram_reserved_output_banks

    def is_valid(self):
        # Assign zero-based bank starts (first element remains zero)
        self.bank_locations[1:] = np.cumsum(self.banks_required)[:-1]

        # Accumulator area is measured from the end of the buffer
        self.bank_locations[SharedBufferArea.Accumulators] = (
            self.arch.available_shram_banks(self.uses_lut) - self.banks_required[SharedBufferArea.Accumulators]
        )
        ifm_end = self.bank_locations[SharedBufferArea.IFM] + self.banks_required[SharedBufferArea.IFM]
        return ifm_end <= self.bank_locations[SharedBufferArea.Accumulators]

    def try_block(self, ofm_block: Block):
        # Get IFM block configuration
        ifm_block_depth = ofm_block.depth if self.is_equal_depth_op else self.ifm_block_depth
        ifm_block = self.arch.get_ifm_block_size(
            ifm_block_depth, ofm_block, self.kernel, ifm_resampling_mode=self.ifm_resampling_mode
        )
        ifm_config = self.arch.get_block_config(ifm_block.width, ifm_block.height, ifm_block.depth)
        if ifm_config is None:
            return None

        # Get OFM block configuration
        ofm_config = self.arch.get_block_config(ofm_block.width, ofm_block.height, ofm_block.depth)
        if ofm_config is None:
            return None

        acc_banks = ofm_config.banks[self.use_accumulator_element]

        # Update bank counts for IFM and Accumulator
        self.banks_required[SharedBufferArea.IFM] = ifm_config.banks[self.use_ifm_element] * self.ifm_count
        self.banks_required[SharedBufferArea.Accumulators] = 0 if self.is_elementwise else acc_banks

        # Validating calculates bank layout and returns validity
        if not self.is_valid():
            return None

        return (ofm_block.height, ofm_block.width, ifm_block.depth, ofm_block.depth)

    def generate_used_mask(self, active_set):
        res = np.zeros(self.arch.shram_total_banks, dtype=np.int64)
        for kind in active_set:
            start = int(self.bank_locations[kind])
            end = start + int(self.banks_required[kind])
            res[start:end] = 1
        return res

    def is_compatible(first, second):
        """See if the bank allocations of two convolutions are compatible,
        so that they can run back-to-back without a fence in between"""

        first_set = set((SharedBufferArea.OFM, SharedBufferArea.Accumulators))
        second_set = set((SharedBufferArea.IFM, SharedBufferArea.Weights))

        first_mask = first.generate_used_mask(first_set)
        second_mask = second.generate_used_mask(second_set)

        if np.sum(first_mask & second_mask):
            # overlap
            return False

        return True

    def get_shram_memory_access_range(self):
        # Returns the SHRAM memory access range used by this shared buffer,
        # excluding access to LUT
        return MemoryRangeSet(
            MemArea.Shram, 0, self.arch.available_shram_banks(self.uses_lut) * self.arch.shram_bank_size
        )


def _all_fms_have_quant(ifm_tensor, ofm_tensor, ifm2_tensor=None) -> bool:
    tensors = [t for t in (ifm_tensor, ifm2_tensor, ofm_tensor) if t is not None]
    scales = [t.quantization.scale_f32 for t in tensors if t.quantization is not None]
    return len(tensors) == len(scales) and None not in scales


def is_acc_40bits_used(npu_block_type, ifm_tensor, ofm_tensor, ifm2_tensor=None):
    return (
        ifm_tensor.dtype.size_in_bits() == 16
        and npu_block_type != NpuBlockType.Pooling
        and _all_fms_have_quant(ifm_tensor, ofm_tensor, ifm2_tensor)
    )


def shared_buffer_allocation_for_pass(arch, ps) -> SharedBufferAllocation:
    ifm_tensor, ifm2_tensor, _, ofm_tensor = ps.get_primary_op_ifm_ifm2_weights_ofm()
    all_fms_have_quant = _all_fms_have_quant(ifm_tensor, ifm2_tensor, ofm_tensor)

    kernel = Kernel(1, 1)
    is_elementwise = ps.npu_block_type == NpuBlockType.ElementWise
    uses_lut = False
    ifm_count = 1

    if ps.primary_op:
        kernel = ps.primary_op.kernel
        uses_lut = ps.primary_op.activation_lut is not None

    ifm_resampling_mode = resampling_mode.NONE
    ifm_bits = 0
    ifm_depth = 0
    if ifm_tensor:
        ifm_resampling_mode = ifm_tensor.resampling_mode
        ifm_bits = ifm_tensor.dtype.size_in_bits()
        ifm_shape = ps.primary_op.ifm_shapes[0]

        if ifm_tensor.shape != []:
            ifm_depth = ifm_shape.depth

        if is_elementwise:
            ifm_count = 2
            if ifm_tensor.shape == []:  # Scalar in ifm1
                assert ifm2_tensor
                ifm_depth = ps.primary_op.ifm_shapes[1].depth
                ifm_count = 1
            elif not ifm2_tensor or ifm2_tensor.shape == []:  # Scalar in ifm2
                ifm_count = 1
    return SharedBufferAllocation(
        arch,
        kernel,
        uses_lut,
        npu_block_type=ps.npu_block_type,
        all_fms_have_quant=all_fms_have_quant,
        ifm_resampling_mode=ifm_resampling_mode,
        ifm_bits=ifm_bits,
        ifm_depth=ifm_depth,
        ifm_count=ifm_count,
        ofm_shape=ps.primary_op.ofm_shapes[0],
    )


def shared_buffer_allocation_for_pass_and_block_config(arch, ps, block_config) -> SharedBufferAllocation:
    alloc = shared_buffer_allocation_for_pass(arch, ps)
    assert (alloc.ifm_block_depth == block_config[2]) or alloc.is_equal_depth_op
    if alloc.try_block(Block(block_config[1], block_config[0], block_config[3])):
        return alloc

    return None


def shared_buffer_allocation_for_npu_op(
    arch, npu_op: NpuBlockOperation, npu_block_type: NpuBlockType, ifm_resampling_mode
) -> SharedBufferAllocation:
    uses_lut = npu_op.activation is not None and npu_op.activation.op_type == NpuActivationOp.TABLE_LOOKUP
    fms = [npu_op.ifm, npu_op.ofm]
    if npu_op.ifm2 is not None:
        fms.append(npu_op.ifm2)
    all_fms_have_quant = not any(fm.quantization is None or fm.quantization.scale_f32 is None for fm in fms)
    ifm_bits = npu_op.ifm.data_type.size_in_bits()
    ifm_depth = npu_op.ifm.shape.depth
    ifm_count = 2 if npu_op.ifm2 is not None and npu_op.ifm2_scalar is None else 1
    ofm_shape = [1, npu_op.ofm.shape.height, npu_op.ofm.shape.width, npu_op.ofm.shape.depth]
    return SharedBufferAllocation(
        arch,
        to_kernel(npu_op.kernel),
        uses_lut,
        npu_block_type=npu_block_type,
        all_fms_have_quant=all_fms_have_quant,
        ifm_resampling_mode=ifm_resampling_mode,
        ifm_bits=ifm_bits,
        ifm_depth=ifm_depth,
        ifm_count=ifm_count,
        ofm_shape=Shape4D(ofm_shape),
    )


def find_suitable_block_configs(arch, alloc: SharedBufferAllocation) -> List[Tuple]:
    """Returns list of block configs that would fit with the given shared buffer allocation"""
    if arch.override_block_config:
        config = alloc.try_block(arch.override_block_config)
        if config is None:
            raise AllocationError(f"Block config override '{arch.override_block_config}' cannot be allocated")
        return [config]

    # Constrain the search space if the OFM is smaller than the max block size
    # - Add other block search constraints here if required
    max_block_width = alloc.ofm_shape.width
    max_block_height = alloc.ofm_shape.height
    max_block_depth = alloc.ofm_shape.depth

    # Constrain to valid ranges before search
    max_block_width = min(arch.ofm_block_max.width, max_block_width)
    max_block_height = min(arch.ofm_block_max.height, max_block_height)
    max_block_depth = min(arch.ofm_block_max.depth, max_block_depth)

    min_block_height = max(arch.ofm_ublock.height, 2 if alloc.ifm_resampling_mode != resampling_mode.NONE else 1)
    min_block_width = max(arch.ofm_ublock.width, 2 if alloc.ifm_resampling_mode != resampling_mode.NONE else 1)

    valid_block_configs = []
    # Try a range of block shapes against this pass
    for w in range(min_block_width, max_block_width + min_block_width, min_block_width):
        for h in range(min_block_height, max_block_height + min_block_height, min_block_height):
            # Try valid OFM block depths
            for c in range(arch.ofm_ublock.depth, max_block_depth + arch.ofm_ublock.depth, arch.ofm_ublock.depth):
                # OFM block depth has the constraint that if it causes the OFM to be
                # split, it must be a multiple of the OFM split size
                if (c >= max_block_depth) or (c < max_block_depth and (c % ArchitectureFeatures.OFMSplitDepth) == 0):
                    config = alloc.try_block(Block(w, h, c))
                    if config:
                        valid_block_configs.append(config)

    assert len(valid_block_configs) > 0
    return valid_block_configs


def find_block_configs_suitable_for_pass_and_shared_buffer(arch, ps) -> List[Tuple]:
    alloc = shared_buffer_allocation_for_pass(arch, ps)
    return find_suitable_block_configs(arch, alloc)
