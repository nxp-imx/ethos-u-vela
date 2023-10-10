# SPDX-FileCopyrightText: Copyright 2020-2023 Arm Limited and/or its affiliates <open-source-office@arm.com>
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
# Utility functions for code generation
from typing import List
from typing import NamedTuple
from typing import Optional

from . import numeric_util
from .api import NpuActivationOp
from .api import NpuAddressRange
from .api import NpuBlockOperation
from .api import NpuDmaOperation
from .api import NpuElementWiseOp
from .api import NpuFeatureMap
from .api import NpuKernel
from .api import NpuLayout
from .api import NpuOperation
from .api import NpuOperationType
from .api import NpuPadding
from .api import NpuQuantization
from .api import NpuShape3D
from .architecture_features import ArchitectureFeatures
from .architecture_features import Block
from .architecture_features import Rect
from .errors import ByteAlignmentError
from .errors import ByteSizeError
from .operation import Kernel
from .operation import PointXYZ
from .tensor import TensorFormat
from ethosu.vela.range_set import AccessDirection
from ethosu.vela.range_set import MemoryAccessSet
from ethosu.vela.range_set import MemoryRangeSet


# base address slot for memory to memory transfer
BASE_PTR_INDEX_MEM2MEM = int((1 << 8) | (3 << 0))


UNARY_ELEMWISE_OPS = (NpuElementWiseOp.ABS, NpuElementWiseOp.LRELU, NpuElementWiseOp.CLZ)


def check_alignment(payload, required_alignment):
    # assuming payload is defined in bytes
    if payload % required_alignment != 0:
        raise ByteAlignmentError(f"Cmd1 payload of size: {payload} Bytes is not {required_alignment}-byte aligned")


def check_size(payload, required_multiple, value_type):
    # assuming payload is defined in bytes
    if payload % required_multiple != 0:
        raise ByteSizeError(f"Cmd1 {value_type} of size: {payload} Bytes is not a multiple of {required_multiple}")


def check_stride(stride, required_multiple):
    check_size(stride, required_multiple, "stride")


def check_length(length, required_multiple):
    check_size(length, required_multiple, "length")


def to_npu_kernel(kernel: Kernel) -> NpuKernel:
    """Converts the given internally used kernel object to NpuKernel (of public API)"""
    return NpuKernel(
        kernel.width, kernel.height, kernel.stride.x, kernel.stride.y, kernel.dilation.x, kernel.dilation.y
    )


def to_kernel(kernel: Optional[NpuKernel]) -> Kernel:
    """Converts the given public API object to Kernel (used internally)"""
    if kernel is None:
        return Kernel(1, 1)
    return Kernel(kernel.width, kernel.height, kernel.stride_x, kernel.stride_y, kernel.dilation_x, kernel.dilation_y)


def has_ifm2(npu_op: NpuBlockOperation) -> bool:
    """Checks if op has non-scalar IFM2"""
    return npu_op.ifm2 is not None and npu_op.ifm2_scalar is None


def shape3d_size(shape: NpuShape3D) -> int:
    return shape.width * shape.height * shape.depth


def shape3d_to_rect(shape: NpuShape3D) -> Rect:
    return Rect(0, 0, 0, shape.width - 1, shape.height - 1, shape.depth - 1)


def shape3d_to_block(shape: NpuShape3D) -> Block:
    return Block(shape.width, shape.height, shape.depth)


def get_zero_point(fm: NpuFeatureMap):
    return int(fm.quantization.zero_point if fm.quantization else 0)


def quantise(value: float, quant: Optional[NpuQuantization]) -> int:
    """Quantizes the given value"""
    scale = 1 if quant is None or quant.scale_f32 is None else quant.scale_f32
    zp = 0 if quant is None else quant.zero_point
    return numeric_util.quantise_float32(value, scale, zp)


# -------------------------------------------------------------------
# ADDRESSING/STRIDES (helper functions)
# -------------------------------------------------------------------


def ranges_overlap(range1: NpuAddressRange, range2: NpuAddressRange) -> bool:
    """Checks if the ranges overlap"""
    return range1.region == range2.region and numeric_util.overlaps(
        range1.address, range1.address + range1.length, range2.address, range2.address + range2.length
    )


def range_lists_overlap(list1: List[Optional[NpuAddressRange]], list2: List[Optional[NpuAddressRange]]) -> bool:
    """Checks if there is any address overlap between list1 and list2"""
    for range1 in list1:
        if range1 is None:
            continue
        for range2 in list2:
            if range2 is not None and ranges_overlap(range1, range2):
                return True
    return False


def get_strides(fm: NpuFeatureMap) -> NpuShape3D:
    """Calculates STRIDE_C/Y/X"""
    if fm.strides is not None:
        return fm.strides
    elem_size = fm.data_type.size_in_bytes()
    if fm.layout == NpuLayout.NHWC:
        stride_c = elem_size
        stride_x = fm.shape.depth * stride_c
        stride_y = fm.shape.width * stride_x
    else:
        stride_x = 16 * elem_size
        stride_c = stride_x * fm.shape.width
        stride_y = elem_size * fm.shape.width * numeric_util.round_up(fm.shape.depth, 16)
    return NpuShape3D(depth=stride_c, height=stride_y, width=stride_x)


def get_address(fm: NpuFeatureMap, strides: NpuShape3D, y: int, x: int, c: int) -> int:
    """Returns address of given coordinate"""
    t = 0
    BRICK = 16
    stride_c = BRICK * fm.data_type.size_in_bytes() if fm.layout == NpuLayout.NHWC else strides.depth
    stride_x = BRICK * fm.data_type.size_in_bytes() if fm.layout == NpuLayout.NHCWB16 else strides.width
    if x >= fm.tiles.width_0:
        x -= fm.tiles.width_0
        t = 1
        if y >= fm.tiles.height_1:
            y -= fm.tiles.height_1
            t += 2
    elif y >= fm.tiles.height_0:
        y -= fm.tiles.height_0
        t += 2
    elem_size = fm.data_type.size_in_bytes()
    return (
        fm.tiles.addresses[t] + y * strides.height + x * stride_x + (c // BRICK) * stride_c + int(c % BRICK) * elem_size
    )


def get_address_range(
    fm: NpuFeatureMap, strides: NpuShape3D, y0: int, x0: int, c0: int, y1: int, x1: int, c1: int
) -> NpuAddressRange:
    """
    Gets address range for (y0, x0, c0) - (y1, x1, c1) (inclusive, so the second coordinate is within the fm).
    The begin and end coordinates must be within the same tile.
    """
    addr0 = get_address(fm, strides, y0, x0, c0)
    addr1 = get_address(fm, strides, y1, x1, c1)
    return NpuAddressRange(region=fm.region, address=addr0, length=addr1 - addr0 + fm.data_type.size_in_bytes())


def get_h_ranges(
    fm: NpuFeatureMap, strides: NpuShape3D, y0: int, x0: int, c0: int, y1: int, x1: int, c1: int
) -> List[NpuAddressRange]:
    """
    Gets address ranges for (y0, x0, c0) - (y1, x1, c1) (inclusive, so the second coordinate is within the fm);
    the begin and end coordinates must be within the same tile.
    Divides the area in horizontal "stripes" of height 1, and returns the address ranges for these "stripes".
    """
    return [get_address_range(fm, strides, y, x0, c0, y, x1, c1) for y in range(y0, y1 + 1)]


def get_address_ranges_for_area(fm: NpuFeatureMap, start: PointXYZ, end: PointXYZ) -> List[Optional[NpuAddressRange]]:
    """
    Returns a list of adddress ranges that covers the area start - end (inclusive).
    Divides the area in horizontal "stripes" of height 1, and returns the address ranges for these "stripes".

    For example, for the area marked with X (in a feature map with 4 tiles) as input, this function would return
    6 address ranges: the address ranges for 1-height areas [AAA, BBB, CC, DD, EEE, FF]

        .....|....           .....|....
     t0 ..XXX|XX.. t1     t0 ..AAA|CC.. t1
        ..XXX|XX..           ..BBB|DD..
        -----+----    -->    -----+----
     t2 ..XXX|XX.. t3     t2 ..EEE|FF.. t3
        .....|....           .....|....
    """
    strides = get_strides(fm)
    height_0, height_1, width_0 = fm.tiles.height_0, fm.tiles.height_1, fm.tiles.width_0
    h, w, c = fm.shape
    y0, x0, c0 = start.y, start.x, start.z
    y1, x1, c1 = min(end.y, h - 1), min(end.x, w - 1), min(end.z, c - 1)
    ranges: List[Optional[NpuAddressRange]] = []
    if x0 < width_0 and y0 < height_0:
        # Horizontal ranges for tile 0
        ranges.extend(get_h_ranges(fm, strides, y0, x0, c0, min(y1, height_0 - 1), min(x1, width_0 - 1), c1))
    if x1 >= width_0 and y0 < height_1:
        # Horizontal ranges for tile 1
        ranges.extend(get_h_ranges(fm, strides, y0, max(x0, width_0), c0, min(y1, height_1 - 1), x1, c1))
    if x0 < width_0 and y1 >= height_0:
        # Horizontal ranges for tile 2
        ranges.extend(get_h_ranges(fm, strides, max(y0, height_0), x0, c0, y1, min(x1, width_0 - 1), c1))
    if x1 >= width_0 and y1 >= height_1:
        # Horizontal ranges for tile 3
        ranges.extend(get_h_ranges(fm, strides, max(y0, height_1), max(x0, width_0), c0, y1, x1, c1))
    return ranges


def get_address_ranges(fm: NpuFeatureMap) -> List[Optional[NpuAddressRange]]:
    """Returns 4 adddress ranges, one for every tile, None if the tile is not in use"""
    strides = get_strides(fm)
    height, width, depth = fm.shape.height, fm.shape.width, fm.shape.depth
    height_0, height_1, width_0 = fm.tiles.height_0, fm.tiles.height_1, fm.tiles.width_0
    t0 = get_address_range(
        fm,
        strides,
        0,
        0,
        0,
        min(height, height_0) - 1,
        min(width, width_0) - 1,
        depth - 1,
    )
    if width > width_0:
        t1 = get_address_range(fm, strides, 0, width_0, 0, min(height, height_1) - 1, width - 1, depth - 1)
    else:
        t1 = None
    if height > height_0:
        t2 = get_address_range(fm, strides, height_0, 0, 0, height - 1, min(width, width_0) - 1, depth - 1)
    else:
        t2 = None
    if t1 is not None and t2 is not None:
        t3 = get_address_range(fm, strides, height_1, width_0, 0, height - 1, width - 1, depth - 1)
    else:
        t3 = None
    return [t0, t1, t2, t3]


def check_strides(fm: NpuFeatureMap, strides: NpuShape3D):

    element_size_in_bytes = fm.data_type.size_in_bytes()

    if fm.layout == NpuLayout.NHCWB16:
        strides_to_check = [strides.depth, strides.height]
        required_multiple = 16
    else:
        strides_to_check = [strides.height, strides.width]
        required_multiple = element_size_in_bytes
    for stride in strides_to_check:
        check_stride(stride, required_multiple)


def check_addresses(addresses: List[int], layout: NpuLayout, element_size, arch: ArchitectureFeatures):
    if layout == NpuLayout.NHCWB16:
        required_alignment = arch.storage_rounding_quantums[TensorFormat.NHCWB16][-1]
    else:
        required_alignment = element_size
    for addr in addresses:
        check_alignment(addr, required_alignment)


# -------------------------------------------------------------------
# DMA_WAIT/KERNEL_WAIT
# -------------------------------------------------------------------


class Watermark(NamedTuple):
    npu: int
    dma: int


def memory_range_set(range: NpuAddressRange) -> MemoryRangeSet:
    return MemoryRangeSet(range.region, range.address, range.address + range.length)


def get_dma_memory_accesses(dma_op: NpuDmaOperation) -> MemoryAccessSet:
    """Returns the address that are read and written by the given DMA operation"""
    res = MemoryAccessSet()
    res.add(memory_range_set(dma_op.src), AccessDirection.Read)
    res.add(memory_range_set(dma_op.dest), AccessDirection.Write)
    return res


def get_op_memory_accesses(npu_op: NpuBlockOperation, arch: ArchitectureFeatures) -> MemoryAccessSet:
    """Returns the addresses that are read and written by the given operation"""
    assert npu_op.ifm is not None and npu_op.ofm is not None
    # Read addresses
    read_ranges = get_address_ranges(npu_op.ifm)
    if has_ifm2(npu_op):
        assert npu_op.ifm2 is not None
        read_ranges.extend(get_address_ranges(npu_op.ifm2))
    read_ranges.extend(npu_op.weights)
    read_ranges.extend(npu_op.biases)
    if npu_op.activation is not None and npu_op.activation.op_type == NpuActivationOp.TABLE_LOOKUP:
        address = arch.available_shram_banks(True) * arch.shram_bank_size
        read_ranges.append(NpuAddressRange(region=BASE_PTR_INDEX_MEM2MEM, address=address, length=2048))
    # Written addresses
    write_ranges = get_address_ranges(npu_op.ofm)
    # Add write access to SHRAM, needed when LUTs can overwrite accumulator banks
    uses_lut = npu_op.activation is not None and npu_op.activation.op_type == NpuActivationOp.TABLE_LOOKUP
    written_shram_size = arch.available_shram_banks(uses_lut) * arch.shram_bank_size
    write_ranges.append(NpuAddressRange(region=BASE_PTR_INDEX_MEM2MEM, address=0, length=written_shram_size))

    res = MemoryAccessSet()
    for read_range in read_ranges:
        if read_range is not None:
            res.add(memory_range_set(read_range), AccessDirection.Read)
    for write_range in write_ranges:
        if write_range is not None:
            res.add(memory_range_set(write_range), AccessDirection.Write)
    return res


def get_wait_dependency(
    arch: ArchitectureFeatures,
    npu_op: NpuOperation,
    memory_accesses,
    outstanding_dma_ops: List[NpuOperation],
    outstanding_npu_ops: List[NpuOperation],
):
    """Used to calculate whether DMA wait or kernel wait operations are needed"""
    kern_wait = -1
    dma_wait = -1
    op_accesses = memory_accesses[npu_op]

    if isinstance(npu_op, NpuDmaOperation):
        outstanding_ops = outstanding_npu_ops
        outstanding_dma_ops.append(npu_op)
        if len(outstanding_dma_ops) > arch.max_outstanding_dma:
            outstanding_dma_ops.pop(0)
    else:
        outstanding_ops = outstanding_dma_ops
        outstanding_npu_ops.append(npu_op)
        if len(outstanding_npu_ops) > arch.max_outstanding_kernels:
            outstanding_npu_ops.pop(0)

    waits = -1
    for idx in range(len(outstanding_ops) - 1, -1, -1):
        waits += 1
        other_op = outstanding_ops[idx]
        other_accesses = memory_accesses[other_op]
        if other_accesses.conflicts(op_accesses):
            if isinstance(npu_op, NpuDmaOperation):
                kern_wait = waits
            else:
                dma_wait = waits
            # Current op needs to wait, and after it has waited,
            # outstanding_ops[0..idx] are not outstanding any longer
            for i in range(idx + 1):
                outstanding_ops.pop(0)
            break

    cmd_waits = Watermark(kern_wait, dma_wait)
    return cmd_waits


def check_dma_op(dma_op: NpuDmaOperation, arch: ArchitectureFeatures):

    # For Ethos-U65 only internal addresses have to be aligned, and if the internal address is the destination
    # then the length has to be aligned also.
    if arch.is_ethos_u65_system:
        if dma_op.src.region == BASE_PTR_INDEX_MEM2MEM:
            check_alignment(dma_op.src.address, 16)
        if dma_op.dest.region == BASE_PTR_INDEX_MEM2MEM:
            check_alignment(dma_op.dest.address, 16)
            check_length(dma_op.src.length, 16)
    else:
        check_alignment(dma_op.src.address, 16)
        check_alignment(dma_op.dest.address, 16)
        check_length(dma_op.src.length, 16)


# -------------------------------------------------------------------
# BLOCKDEP
# -------------------------------------------------------------------


def get_ifm_ofm_block_depth(arch: ArchitectureFeatures, npu_op: NpuBlockOperation) -> int:
    # Note: NOT equivalent to the normal ifm block depth calculation since
    # it takes into account 'depthless' block operations by returning full
    # depth
    if npu_op.op_type == NpuOperationType.Conv2D:
        res = arch.calc_ifm_block_depth(npu_op.ifm.shape.depth, npu_op.ifm.data_type.size_in_bits())
        return res
    return npu_op.ofm.shape.depth


def coords_intersect(start_a: PointXYZ, end_a: PointXYZ, start_b: PointXYZ, end_b: PointXYZ) -> bool:
    """Checks if the two areas overlap"""
    start_x = max(start_a.x, start_b.x)
    end_x = min(end_a.x, end_b.x)
    start_y = max(start_a.y, start_b.y)
    end_y = min(end_a.y, end_b.y)
    start_z = max(start_a.z, start_b.z)
    end_z = min(end_a.z, end_b.z)
    return ((end_x - start_x) > 0) and ((end_y - start_y) > 0) and ((end_z - start_z) > 0)


def intersects(
    ifm: NpuFeatureMap,
    ifm_start_coord: PointXYZ,
    ifm_end_coord: PointXYZ,
    prev_ofm: NpuFeatureMap,
    ofm_start_coord: PointXYZ,
    ofm_end_coord: PointXYZ,
) -> bool:
    """Checks if the given IFM area overlaps with the given OFM area"""
    if ifm.shape == prev_ofm.shape and ifm.tiles == prev_ofm.tiles:
        # Common case: prev_op.ofm == op.ifm; in this case it suffices to check
        # if the xyz coordinates overlap, which is quick and easy
        res = coords_intersect(ifm_start_coord, ifm_end_coord, ofm_start_coord, ofm_end_coord)
    else:
        # The OFM produces a part of the IFM (e.g. a stripe), or the IFM consumes part of the OFM.
        # In this case, address comparison between the two areas is needed
        ifm_ranges: List[Optional[NpuAddressRange]] = get_address_ranges_for_area(ifm, ifm_start_coord, ifm_end_coord)
        prev_ofm_ranges = get_address_ranges_for_area(prev_ofm, ofm_start_coord, ofm_end_coord)
        res = range_lists_overlap(ifm_ranges, prev_ofm_ranges)
    return res


# Block job dependency:
# Does the VOLUME of IFMs for block job B(0) overlap with VOLUME of OFMs block jobs A(8,9,10)
#
#  A                    | B
# ----------------------+------------------
# .... 3,4,5,6,7,8,9,10 | 0,1,2,3,4,5,6,8 10 < JOB NUMBER
#               |<------->| dependency offset
#


def get_offset_block_coords(area: Rect, block: Block, offset: int) -> Optional[PointXYZ]:
    """
    Get the coordinates of a block offset from either the end (negative)
    or the start (zero or positive) of the given 3D area
    """
    size = area.size()
    # Dimensions of the region, in blocks
    width_blocks = numeric_util.round_up_divide(size.width, block.width)
    height_blocks = numeric_util.round_up_divide(size.height, block.height)
    depth_blocks = numeric_util.round_up_divide(size.depth, block.depth)
    total_blocks = width_blocks * height_blocks * depth_blocks
    if offset < 0:
        index = total_blocks + offset
    else:
        index = offset

    if index >= total_blocks:
        return None

    # Coordinates of the indexed block
    coord_z = block.depth * (index % depth_blocks)
    coord_y = block.height * (index // (depth_blocks * width_blocks))
    coord_x = block.width * ((index // depth_blocks) % width_blocks)

    return PointXYZ(x=coord_x + area.x, y=coord_y + area.y, z=coord_z + area.z)


def get_first_job_input_volume(
    arch: ArchitectureFeatures,
    ifm: Rect,
    ofm: Rect,
    ifm_block_depth,
    ofm_block: Block,
    kernel: Kernel,
    padding: NpuPadding,
    block_offset: int,
):
    # Get ifm block size (jobs are invisibly decomposed into subkernels)
    ifm_block = arch.get_ifm_block_size(ifm_block_depth, ofm_block, kernel, arch.ofm_block_max)
    ifm_depth_blocks = numeric_util.round_up_divide(ifm.size().depth, ifm_block_depth)

    # Which OFM block are we calculating
    ofm_coord = get_offset_block_coords(ofm, ofm_block, block_offset // ifm_depth_blocks)
    if ofm_coord is None:
        return None

    # Coordinate of the source IFM block
    ifm_coord_x = max(0, ofm_coord[0] * kernel.stride.x - padding.left)
    ifm_coord_y = max(0, ofm_coord[1] * kernel.stride.y - padding.right)
    ifm_coord_z = ifm.z + (block_offset % ifm_depth_blocks) * ifm_block.depth

    # IFM block that will be sampled for the FIRST+block_offset job in the next operator's OFM
    start_coord = PointXYZ(x=ifm_coord_x, y=ifm_coord_y, z=ifm_coord_z)
    end_coord = PointXYZ(
        x=start_coord[0] + ifm_block.width,
        y=start_coord[1] + ifm_block.height,
        z=start_coord[2] + ifm_block.depth,
    )
    return (start_coord, end_coord, 1)  # start, end, total jobs


def get_prev_job_output_volume(ofm: Rect, ofm_block: Block, block_offset: int):
    assert block_offset >= 0

    # Get OFM block's volume coordinates
    start_coord = get_offset_block_coords(ofm, ofm_block, -1 - block_offset)
    if start_coord is None:
        return None
    end_coord = PointXYZ(
        x=start_coord.x + ofm_block.width,
        y=start_coord.y + ofm_block.height,
        z=start_coord.z + ofm_block.depth,
    )
    return (start_coord, end_coord, 1)  # start, end, total jobs for this OFM block


def calc_blockdep(
    arch: ArchitectureFeatures,
    prev_op: Optional[NpuBlockOperation],
    npu_op: NpuBlockOperation,
) -> int:
    """Calculates the value of the BLOCKDEP register"""
    if prev_op is None:
        return 0
    assert npu_op.ifm is not None
    assert prev_op.ofm is not None
    # Check if the reserved shram will be used in current/prev op
    prev_uses_lut = prev_op.activation is not None and prev_op.activation.op_type == NpuActivationOp.TABLE_LOOKUP
    curr_uses_lut = npu_op.activation is not None and npu_op.activation.op_type == NpuActivationOp.TABLE_LOOKUP
    if prev_uses_lut and arch.shram_reserved_unused_banks == 0 and not curr_uses_lut:
        return 0

    # Check if IFM or IFM2 overlaps with prev op's OFM
    prev_ofm_ranges = get_address_ranges(prev_op.ofm)
    ifm_ranges = get_address_ranges(npu_op.ifm)
    ifm_overlaps = range_lists_overlap(prev_ofm_ranges, ifm_ranges)
    if has_ifm2(npu_op):
        assert npu_op.ifm2 is not None
        ifm2_ranges = get_address_ranges(npu_op.ifm2)
        ifm2_overlaps = range_lists_overlap(prev_ofm_ranges, ifm2_ranges)
    else:
        ifm2_overlaps = False
    if ifm_overlaps and ifm2_overlaps:
        # Both IFM and IFM2 overlap (should be rare)
        return 0
    if not ifm_overlaps and not ifm2_overlaps:
        # No overlap between prev OFM and IFM/IFM2
        return ArchitectureFeatures.MAX_BLOCKDEP
    if ifm2_overlaps and shape3d_size(npu_op.ifm2.shape) < shape3d_size(npu_op.ifm.shape):
        # Prev OFM produces IFM2 which is broadcasted (this should be rare)
        return 0
    # Prev OFM overlaps with IFM or IFM2; calculate the blockdep
    prev_block_config = prev_op.block_config
    block_config = npu_op.block_config
    overlapping_fm = npu_op.ifm if ifm_overlaps else npu_op.ifm2
    assert overlapping_fm is not None

    cur_ifm_block_depth = get_ifm_ofm_block_depth(arch, npu_op)
    cur_ofm_block = Block(block_config.width, block_config.height, block_config.depth)
    cur_ofm_rect = shape3d_to_rect(npu_op.ofm.shape)
    cur_ifm_rect = shape3d_to_rect(npu_op.ifm.shape)
    padding = NpuPadding(0, 0, 0, 0) if npu_op.padding is None else npu_op.padding
    blockdep = ArchitectureFeatures.MAX_BLOCKDEP
    kernel = to_kernel(npu_op.kernel)

    prev_ofm_block = Block(prev_block_config.width, prev_block_config.height, prev_block_config.depth)
    prev_ofm_rect = shape3d_to_rect(prev_op.ofm.shape)
    # Iterate over the next BLOCKDEP inputs, checking to see if a sliding window
    # of IFM area overlaps with any previous OFM block generation.
    elapsed_jobs = 0
    for forward_offset in range(ArchitectureFeatures.MAX_BLOCKDEP):
        # This is the IFM block we want to sample from
        in_area = get_first_job_input_volume(
            arch, cur_ifm_rect, cur_ofm_rect, cur_ifm_block_depth, cur_ofm_block, kernel, padding, forward_offset
        )
        if in_area is None:
            break

        # Try several previous-OFM blocks in the past (they still might comprise multiple IFM jobs)
        outstanding_jobs = 0
        for block_offset in range(ArchitectureFeatures.MAX_BLOCKDEP):
            # This is the OFM block being generated by the previous op
            out_area = get_prev_job_output_volume(prev_ofm_rect, prev_ofm_block, block_offset)
            if out_area is None:
                break

            # Block dependency is the max number of allowed outstanding jobs
            # in the pipeline. Selected by determining how many jobs occur
            # in between two operators' overlapping OFM->IFM block volumes
            if intersects(overlapping_fm, in_area[0], in_area[1], prev_op.ofm, out_area[0], out_area[1]):
                break
            # Early exit if no intersections and we've seen enough jobs in the pipeline
            elif outstanding_jobs > ArchitectureFeatures.MAX_BLOCKDEP:
                break

            # This OFM had this many jobs (accumulate over multiple OFM blocks)
            outstanding_jobs += out_area[2]

        blockdep = min(blockdep, elapsed_jobs + outstanding_jobs)
        elapsed_jobs += in_area[2]
        # Early exit if no intersections and we've seen enough jobs in the pipeline
        if elapsed_jobs > ArchitectureFeatures.MAX_BLOCKDEP:
            break

    return blockdep
