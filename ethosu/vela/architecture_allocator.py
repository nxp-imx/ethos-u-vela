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
#
# Description: Architecture SHRAM allocator
import enum
import math
from typing import Optional
from typing import Tuple

from .architecture_features import ArchitectureFeatures
from .architecture_features import Block
from .architecture_features import SHRAMConfig
from .architecture_features import SHRAMElements
from .ethos_u55_regs.ethos_u55_regs import resampling_mode
from .numeric_util import round_up
from .numeric_util import round_up_divide
from .operation import Kernel
from .operation import NpuBlockType
from .range_set import MemoryRangeSet
from .shape4d import Shape4D
from .tensor import MemArea


class SHRAMLayout:
    def __init__(self):
        self.ib_start = 0
        self.ib_end = 0
        self.ib_start2 = 0
        self.ab_start = 0
        self.lut_start = 0


class ArchitectureBlockConfig:
    def __init__(self):
        self.layout = SHRAMLayout()
        self.ifm_block = Shape4D()
        self.ofm_block = Shape4D()
        self.acc_type = SHRAMElements.Acc32
        self.is_partkernel = False
        self.bank_size = 0

    def get_shram_memory_access_range(self):
        # Returns the SHRAM memory access range used by this shared buffer,
        # excluding access to LUT
        return MemoryRangeSet(MemArea.Shram, 0, self.layout.lut_start * self.bank_size)

    def old_style_representation(self):
        return [self.ofm_block.height, self.ofm_block.width, self.ifm_block.depth, self.ofm_block.depth]

    def __str__(self):
        return str(self.old_style_representation())


_AccumulatorBits = {SHRAMElements.Acc16: 16, SHRAMElements.Acc32: 32, SHRAMElements.Acc40: 40}


class ElementwiseUsage(enum.IntEnum):
    No = 0
    Full = 1
    Scalar = 2


def _try_block_config(
    shram: SHRAMConfig,
    ew_usage: ElementwiseUsage,
    ofm_block: Block,
    ifm_block: Block,
    ifm_bits: int,
    ifm_granule: int,
    acc_bits: int,
    acc_granule: int,
    lut_banks: int,
) -> SHRAMLayout:
    assert (acc_bits > 0) and (acc_granule > 0)
    assert (ifm_bits >= 8) and ((ifm_bits % 8) == 0) and (ifm_granule > 0)

    # Aways need IFM space
    ifm_bytes = ifm_block.elements_wh() * round_up((ifm_block.depth * ifm_bits) / 8, 8)
    ifm_banks = round_up_divide(ifm_bytes, shram.bank_size_bytes) * 2
    ifm_banks = round_up(ifm_banks, ifm_granule)

    # Calculate SHRAM boundaries of the IFM and Accumulators
    lut_start = shram.total_banks - lut_banks
    ifm_end = shram.reserved_output_banks + ifm_banks
    ifm2_start = ifm_end
    acc_start = lut_start

    # If not elementwise then we need accumulator space
    if ew_usage == ElementwiseUsage.No:
        acc_bytes = (ofm_block.elements_wh() * round_up(ofm_block.depth, 8) * acc_bits) // 8
        acc_banks = round_up_divide(acc_bytes, shram.bank_size_bytes) * 2
        acc_banks = round_up(acc_banks, acc_granule)
        acc_start = acc_start - acc_banks
    else:
        ifm2_banks = ifm_banks if ew_usage == ElementwiseUsage.Full else 0
        if ifm2_start + ifm2_banks > acc_start:
            return None
        ifm_end = acc_start

    # IFM must still fit before accumulators
    if ifm_end > acc_start:
        return None

    # Should all fit, so return this layout
    layout = SHRAMLayout()
    layout.ib_start = shram.reserved_output_banks
    layout.ib_start2 = ifm2_start
    layout.ib_end = ifm_end
    layout.ab_start = acc_start
    layout.lut_start = lut_start
    return layout


def _choose_kernel_method(ifm_shape: Shape4D, ifm_bits: int, kernel: Kernel) -> bool:
    if ifm_shape.depth <= 8:
        return True

    # Compare part-kernel to depth-kernel and choose the one with best utilisation
    kernel_elements = kernel.elements_wh()
    depth_utilisation = ifm_shape.depth / round_up(ifm_shape.depth, 32 if ifm_bits == 8 else 16)
    part_utilisation = (
        ifm_shape.depth
        * kernel_elements
        / (round_up(ifm_shape.depth, 8) * round_up(kernel_elements, 4 if ifm_bits == 8 else 2))
    )

    return part_utilisation > depth_utilisation


def _ew_usage(npu_op_type: NpuBlockType, uses_scalar: bool) -> ElementwiseUsage:
    ew_usage = ElementwiseUsage.No
    if npu_op_type == NpuBlockType.ElementWise:
        ew_usage = ElementwiseUsage.Full
        if uses_scalar:
            ew_usage = ElementwiseUsage.Scalar
    return ew_usage


def _acc_type(npu_op_type: NpuBlockType, ifm_bits: int, scaled: bool) -> int:
    """Returns accumulator type"""
    acc_type = SHRAMElements.Acc32
    if (ifm_bits == 16) and npu_op_type != NpuBlockType.Pooling and scaled:
        acc_type = SHRAMElements.Acc40
    return acc_type


def to_upscale(ifm_resampling: resampling_mode) -> int:
    # Upscaling depending on resampling mode
    return 1 if ifm_resampling == resampling_mode.NONE else 2


def _ifm_blockdepth(arch, ifm_shape: Shape4D, ifm_bits: int, is_partkernel: bool):
    if ifm_bits == 16:
        ifm_blockdepth = round_up(min(ifm_shape.depth, 16), 4)
    else:
        ifm_blockdepth = round_up(min(ifm_shape.depth, 16 if is_partkernel else 32), arch.ifm_ublock.depth)
    return ifm_blockdepth


def _required_size(value: int, stride: int, border: int, upscale: int) -> int:
    return int(math.ceil(((value - 1) * stride + border) / upscale))


def get_ifm_area_required(ofm_shape: Shape4D, kernel: Kernel, upscale: int) -> Tuple[int, int]:
    h1 = _required_size(ofm_shape.height, kernel.stride.y, kernel.area_height(), upscale)
    w1 = _required_size(ofm_shape.width, kernel.stride.x, kernel.area_width(), upscale)
    return (w1, h1)


def _get_ifm_blocksize(
    ofm_block: Shape4D, kernel: Kernel, ublock: Block, subkernel_limit: Block, upscale: int
) -> Shape4D:
    # IFM block height
    h1 = _required_size(ofm_block.height, kernel.stride.y, min(kernel.area_height(), subkernel_limit.height), upscale)
    h2 = h1
    height = round_up(min(h1, h2), ublock.height)

    # IFM block width
    w1 = _required_size(ofm_block.width, kernel.stride.x, min(kernel.area_width(), subkernel_limit.width), upscale)
    w2 = w1
    width = round_up(min(w1, w2), ublock.width)

    return Shape4D(1, height, width, ofm_block.depth)


def find_block_config(
    arch: ArchitectureFeatures,
    npu_op_type: NpuBlockType,
    ofm_shape: Shape4D,
    ifm_shape: Shape4D,
    ifm2_shape: Shape4D,
    uses_scalar: bool,
    ifm_bits: int,
    kernel: Kernel,
    lut_banks: int,
    scaled: bool,
    ifm_resampling: resampling_mode,
) -> ArchitectureBlockConfig:
    SplitDepth = ArchitectureFeatures.OFMSplitDepth
    # Elementwise larger-volume correction
    if ifm2_shape is not None and ifm2_shape.elements() > ifm_shape.elements():
        ifm_shape = ifm2_shape

    # Figure out if SHRAM should be portioned for elementwise
    ew_usage = _ew_usage(npu_op_type, uses_scalar)

    # Operator typing help
    is_pooling = npu_op_type == NpuBlockType.Pooling
    is_depthwise = npu_op_type == NpuBlockType.ConvolutionDepthWise
    is_equal_depth_op = (ew_usage != ElementwiseUsage.No) or is_pooling or is_depthwise
    is_convolution = (npu_op_type == NpuBlockType.ConvolutionMxN) or is_depthwise

    # Block config to be returned
    config = ArchitectureBlockConfig()
    config.is_partkernel = is_convolution and _choose_kernel_method(ifm_shape, ifm_bits, kernel)

    # Accumulator & granule settings
    config.acc_type = _acc_type(npu_op_type, ifm_bits, scaled)

    # Memory rounding granules
    acc_granule = arch.accumulator_granules[config.acc_type]
    acc_bits = _AccumulatorBits[config.acc_type]
    if ew_usage != ElementwiseUsage.No:
        ifm_granule = arch.ifm_ew_bank_granules[ifm_bits]
    else:
        ifm_granule = arch.ifm_bank_granules[ifm_bits]
    lut_banks = max(lut_banks, arch.shram.reserved_end_banks)
    upscale = to_upscale(ifm_resampling)

    # Subkernel repeats of the IFM
    ifm_repeats = round_up_divide(kernel.area_width(), arch.SubKernelMax.width) * round_up_divide(
        kernel.area_height(), arch.SubKernelMax.height
    )
    ifm_blockdepth = _ifm_blockdepth(arch, ifm_shape, ifm_bits, config.is_partkernel)

    # Weights fetch (for operators that have them)
    weight_fetch_wh = (kernel.area_width() * kernel.area_height()) if is_convolution else 0

    search_space = Shape4D.min(ofm_shape, Shape4D(arch.ofm_block_max.to_hwc()))
    search_space = Shape4D.round_up(search_space, Shape4D(arch.ofm_ublock.to_hwc()))

    # Block WHC search, loops across the search space looking for best efficiency
    best_cost = math.inf
    depth = max(arch.ofm_ublock.depth, min(search_space.depth, SplitDepth))
    if depth < ofm_shape.depth:
        depth = round_up(depth, SplitDepth)

    while depth <= search_space.depth:
        wont_fit = {}
        for height in range(arch.ofm_ublock.height, search_space.height + 1, arch.ofm_ublock.height):
            for width in range(arch.ofm_ublock.width, search_space.width + 1, arch.ofm_ublock.width):
                # Avoid checking W/H transposed blocks that already didn't fit. i.e. if 8x4x16 didn't
                # fit, then 4x8x16 won't either.
                if wont_fit.get((height, width), False):
                    continue

                # Calculate the IFM block dimensions required to feed this OFM block
                ofm_block = Shape4D(1, height, width, depth)
                ifm_block = _get_ifm_blocksize(ofm_block, kernel, arch.ofm_ublock, arch.SubKernelMax, upscale)
                if not is_equal_depth_op:
                    ifm_block = ifm_block.with_depth(ifm_blockdepth)

                # Test if the IFM/OFM blocks fit into SHRAM
                layout = _try_block_config(
                    arch.shram, ew_usage, ofm_block, ifm_block, ifm_bits, ifm_granule, acc_bits, acc_granule, lut_banks
                )

                if layout:
                    # Calculate cost in terms of OFM pixels per IFM+Weights fetch
                    ifm_fetch = ifm_block.elements_wh() * ifm_shape.depth
                    weight_fetch = weight_fetch_wh * ifm_shape.depth * (1 if is_depthwise else ofm_block.depth)
                    relative_fetch = (ifm_fetch * ifm_repeats + weight_fetch) / ofm_block.elements()

                    # Bias by the number of blocks we'd need to fill the OFM area (fewer, larger, blocks are better)
                    block_bias = round_up_divide(ofm_shape.height, ofm_block.height)
                    block_bias *= round_up_divide(ofm_shape.width, ofm_block.width)
                    # Check waste on all axes (prefer depth, width then height)
                    waste_ratio = 1 + (1.2 * ((ofm_shape.depth % ofm_block.depth) / ofm_block.depth))
                    waste_ratio *= 1 + (1.1 * ((ofm_shape.width % ofm_block.width) / ofm_block.width))
                    waste_ratio *= 1 + (1.0 * ((ofm_shape.height % ofm_block.height) / ofm_block.height))

                    # Bias for larger area coverage (or volume if not depthwise)
                    area_bias = 1 / (ofm_block.height * ofm_block.width)
                    if not (is_depthwise or is_pooling):
                        area_bias = area_bias / ofm_block.depth

                    relative_cost = relative_fetch * block_bias * waste_ratio * area_bias

                    # If the entire IFM can be encompassed by both buffers, bias to prefer this configuration
                    if ifm_shape.elements() < ifm_block.elements() * 2:
                        relative_cost = relative_cost / 2

                    if relative_cost < best_cost:
                        best_cost = relative_cost
                        config.layout = layout
                        config.bank_size = arch.shram_bank_size
                        config.ifm_block = ifm_block
                        config.ofm_block = ofm_block
                else:
                    wont_fit[(width, height)] = True

        depth = depth + arch.ofm_ublock.depth
        if depth < ofm_shape.depth:
            depth = round_up(depth, SplitDepth)

    if best_cost != math.inf:
        return config

    return None


def try_block_config(
    block_config: Block,
    arch: ArchitectureFeatures,
    npu_op_type: NpuBlockType,
    ifm_shape: Block,
    ifm2_shape: Optional[Block],
    uses_scalar: bool,
    ifm_bits: int,
    is_partkernel: bool,
    kernel: Kernel,
    lut_banks: int,
    scaled: bool,
    ifm_resampling: resampling_mode,
) -> Optional[ArchitectureBlockConfig]:
    """
    Given a block_config, returns a corresponding ArchitectureBlockConfig.
    Returns None if the block_config does not fit or is invalid.
    """
    # Check block config validity
    if not all(
        blk > 0 and blk <= blk_max and blk % ublk == 0
        for blk, blk_max, ublk in zip(block_config.as_list(), arch.ofm_block_max.as_list(), arch.ofm_ublock.as_list())
    ):
        return None
    # Elementwise larger-volume correction
    if ifm2_shape is not None and ifm2_shape.elements() > ifm_shape.elements():
        ifm_shape = ifm2_shape

    ew_usage = _ew_usage(npu_op_type, uses_scalar)

    # Operator typing help
    is_pooling = npu_op_type == NpuBlockType.Pooling
    is_depthwise = npu_op_type == NpuBlockType.ConvolutionDepthWise
    is_equal_depth_op = (ew_usage != ElementwiseUsage.No) or is_pooling or is_depthwise

    # Block config to be returned
    config = ArchitectureBlockConfig()
    config.is_partkernel = is_partkernel

    # Accumulator & granule settings
    config.acc_type = _acc_type(npu_op_type, ifm_bits, scaled)

    # Memory rounding granules
    acc_granule = arch.accumulator_granules[config.acc_type]
    acc_bits = _AccumulatorBits[config.acc_type]
    if ew_usage != ElementwiseUsage.No:
        ifm_granule = arch.ifm_ew_bank_granules[ifm_bits]
    else:
        ifm_granule = arch.ifm_bank_granules[ifm_bits]
    lut_banks = max(lut_banks, arch.shram.reserved_end_banks)
    upscale = to_upscale(ifm_resampling)
    ifm_blockdepth = _ifm_blockdepth(arch, ifm_shape, ifm_bits, is_partkernel)
    ifm_block = _get_ifm_blocksize(block_config, kernel, arch.ofm_ublock, arch.SubKernelMax, upscale)
    if not is_equal_depth_op:
        ifm_block = ifm_block.with_depth(ifm_blockdepth)

    layout = _try_block_config(
        arch.shram, ew_usage, block_config, ifm_block, ifm_bits, ifm_granule, acc_bits, acc_granule, lut_banks
    )
    if layout is None:
        return None
    config.layout = layout
    config.bank_size = arch.shram_bank_size
    config.ifm_block = ifm_block
    return config
