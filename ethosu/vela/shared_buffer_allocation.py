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
# Shared buffer allocation works out how to allocate the Ethos-U55 shared buffer for a given pass.
import numpy as np

from .architecture_features import ArchitectureFeatures
from .architecture_features import Block
from .architecture_features import Kernel
from .architecture_features import SharedBufferArea
from .architecture_features import SHRAMElements
from .errors import VelaError
from .ethos_u55_regs.ethos_u55_regs import resampling_mode
from .operation import NpuBlockType
from .range_set import MemoryRangeSet
from .tensor import MemArea


class SharedBufferAllocation:
    def __init__(self, arch, ps):
        self.arch = arch

        self.bank_locations = np.zeros(SharedBufferArea.Size)
        self.banks_required = np.zeros(SharedBufferArea.Size)

        ifm_tensor, ifm2_tensor, weight_tensor, ofm_tensor = ps.get_primary_op_ifm_ifm2_weights_ofm()

        strides = (1, 1, 1, 1)
        dilation = (1, 1, 1, 1)
        self.kernel = Kernel(1, 1)
        is_elementwise = ps.npu_block_type == NpuBlockType.ElementWise
        self.uses_lut = False

        if ps.primary_op:
            strides = ps.primary_op.attrs.get("strides", strides)
            dilation = ps.primary_op.attrs.get("dilation", dilation)
            k_h = 1
            k_w = 1
            if weight_tensor:
                if ps.primary_op.type != "FullyConnectedAct":
                    k_h = weight_tensor.shape[0]
                    k_w = weight_tensor.shape[1]
            else:
                k_h = ps.primary_op.attrs.get("filter_height", 1)
                k_w = ps.primary_op.attrs.get("filter_width", 1)

            self.kernel = Kernel(k_w, k_h, strides[2], strides[1], dilation[2], dilation[1])
            self.uses_lut = ps.primary_op.activation_lut is not None

        self.is_equal_depth_op = is_elementwise or ps.npu_block_type in (
            NpuBlockType.ConvolutionDepthWise,
            NpuBlockType.Pooling,
        )
        self.strides = strides

        self.use_accumulator_element = SHRAMElements.Acc32
        if is_elementwise:
            self.use_ifm_element = SHRAMElements.IFM8_Elementwise
        else:
            self.use_ifm_element = SHRAMElements.IFM8

        self.ifm_resampling_mode = resampling_mode.NONE
        self.ifm_bits = 0
        self.ifm_depth = 0
        if ifm_tensor:
            self.ifm_resampling_mode = ifm_tensor.resampling_mode
            self.ifm_bits = ifm_tensor.dtype.size_in_bits()
            if ifm_tensor.shape == [] and is_elementwise:
                # Elementwise operator with scalar in ifm, use ifm2 depth
                self.ifm_depth = ifm2_tensor.shape[-1]
            else:
                self.ifm_depth = ifm_tensor.shape[-1]
            if self.ifm_bits == 16:
                if ps.npu_block_type != NpuBlockType.Pooling:
                    self.use_accumulator_element = SHRAMElements.Acc40
                self.use_ifm_element = self.use_ifm_element + 1
                assert (self.use_ifm_element == SHRAMElements.IFM16) or (
                    self.use_ifm_element == SHRAMElements.IFM16_Elementwise
                )
            elif is_elementwise or ps.npu_block_type == NpuBlockType.ReduceSum and self.ifm_bits == 32:
                self.use_ifm_element = SHRAMElements.IFM32
            else:
                assert self.ifm_bits == 8, "Unexpected IFM bitdepth"

        self.ifm_block_depth = arch.calc_ifm_block_depth(self.ifm_depth, self.ifm_bits)
        self.ofm_tensor = ofm_tensor

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

        # Update bank counts for IFM and Accumulator
        self.banks_required[SharedBufferArea.IFM] = ifm_config.banks[self.use_ifm_element]
        self.banks_required[SharedBufferArea.Accumulators] = ofm_config.banks[self.use_accumulator_element]

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


def shared_buffer_allocation_for_pass_and_block_config(arch, ps, block_config):
    alloc = SharedBufferAllocation(arch, ps)
    assert (alloc.ifm_block_depth == block_config[2]) or alloc.is_equal_depth_op
    if alloc.try_block(Block(block_config[1], block_config[0], block_config[3])):
        return alloc

    return None


def find_block_configs_suitable_for_pass_and_shared_buffer(arch, ps):
    alloc = SharedBufferAllocation(arch, ps)

    if arch.override_block_config:
        config = alloc.try_block(arch.override_block_config)
        if config is None:
            raise VelaError("Block config override '{0}' cannot be allocated".format(arch.override_block_config))
        return [config]

    # Constrain the search space if the OFM is smaller than the max block size
    # - Add other block search constraints here if required
    if len(alloc.ofm_tensor.shape) == 2:
        max_block_height = max_block_width = alloc.ofm_tensor.shape[0]
    else:
        max_block_width = alloc.ofm_tensor.shape[-2]
        max_block_height = alloc.ofm_tensor.shape[-3]

    # Common block depth
    max_block_depth = alloc.ofm_tensor.shape[-1]

    # Constrain to valid ranges before search
    max_block_width = min(arch.ofm_block_max.width, max_block_width)
    max_block_height = min(arch.ofm_block_max.height, max_block_height)
    max_block_depth = min(arch.ofm_block_max.depth, max_block_depth)

    valid_block_configs = []
    # Try a range of block shapes against this pass
    for w in range(arch.ofm_ublock.width, max_block_width + arch.ofm_ublock.width, arch.ofm_ublock.width):
        for h in range(arch.ofm_ublock.height, max_block_height + arch.ofm_ublock.height, arch.ofm_ublock.height):
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
