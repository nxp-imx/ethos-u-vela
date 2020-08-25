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
# Holds a container for Ethos-U55/System architecture parameters.
import enum
from collections import namedtuple
from configparser import ConfigParser

import numpy as np

from .errors import OptionError
from .ethos_u55_regs.ethos_u55_regs import resampling_mode
from .numeric_util import round_up
from .numeric_util import round_up_divide
from .operation import NpuBlockType
from .supported_operators import SupportedOperators
from .tensor import MemArea
from .tensor import MemType
from .tensor import TensorFormat
from .tensor import TensorPurpose

PointXY = namedtuple("PointXY", "x y")
PointXYZ = namedtuple("PointXYZ", "x y z")


class Block:
    def __init__(self, w, h, d):
        self.width = w
        self.height = h
        self.depth = d

    def __eq__(self, other):
        if self.width == other.width and self.height == other.height and self.depth == other.depth:
            return True
        else:
            return False

    def __repr__(self):
        return "<Block: {0},{1},{2}>".format(self.width, self.height, self.depth)

    @classmethod
    def from_string(cls, s):
        w, h, c = (int(v) for v in s.split("x"))
        return cls(w, h, c)


class Rect:
    def __init__(self, x, y, z, x2, y2, z2):
        self.x = x
        self.y = y
        self.z = z
        self.x2 = x2
        self.y2 = y2
        self.z2 = z2

    def start(self):
        return PointXYZ(self.x, self.y, self.z)

    def end(self):
        return PointXYZ(self.x2, self.y2, self.z2)

    def size(self):
        return Block(self.x2 - self.x + 1, self.y2 - self.y + 1, self.z2 - self.z + 1)

    def __repr__(self):
        return "<Rect: ({0},{1},{2}) ({3},{4},{5})>".format(self.x, self.y, self.z, self.x2, self.y2, self.z2)


class Kernel:
    def __init__(self, w, h, sx=1, sy=1, dx=1, dy=1):
        assert sx > 0 and sy > 0
        assert dx > 0 and dy > 0
        self.width = w
        self.height = h
        self.stride = PointXY(sx, sy)
        self.dilation = PointXY(dx, dy)


class SHRAMElements:
    IFM8 = 0
    IFM16 = 1
    IFM8_Elementwise = 2
    IFM16_Elementwise = 3
    IFM32 = 4
    Acc16 = 5
    Acc32 = 6
    Acc40 = 7
    Last = Acc40
    BitSizes = np.array([8, 16, 8, 16, 32, 16, 32, 40], np.int32)
    ByteSizes = BitSizes // 8
    PostAlign = np.array([8, 8, 8, 8, 8, 1, 1, 1], np.int32)
    PreAlign = np.array([1, 1, 1, 1, 1, 8, 8, 8], np.int32)


class SHRAMBlockConfig:
    def __init__(self, sizes, banks):
        assert len(banks) == SHRAMElements.Last + 1
        self.sizes = sizes
        self.banks = banks


# Area indices must match Ethos-U55 SHRAM layout spec
class SharedBufferArea(enum.IntEnum):
    OFM = 0
    Weights = 1
    IFM = 2
    Accumulators = 3
    Size = Accumulators + 1


class Accelerator(enum.Enum):
    Ethos_U55_32 = "ethos-u55-32"
    Ethos_U55_64 = "ethos-u55-64"
    Ethos_U55_128 = "ethos-u55-128"
    Ethos_U55_256 = "ethos-u55-256"
    Yoda_256 = "yoda-256"
    Yoda_512 = "yoda-512"

    @classmethod
    def member_list(cls):
        return [e.value for e in cls]


class ArchitectureFeatures:
    """This class is a container for various parameters of the Ethos-U55 core
and system configuration that can be tuned, either by command line
parameters or by the Ethos-U55 architects. The class is often passed
around to passes that need to do architecture-dependent actions.

Note the difference between ArchitectureFeatures and CompilerOptions
- ArchitectureFeatures is for changing the Ethos-U55 and system architecture
- CompilerOptions is for changing the behaviour of the compiler

"""

    ArchitectureConfig = namedtuple(
        "ArchitectureConfig", "macs cores ofm_ublock ifm_ublock shram_banks shram_granules elem_units"
    )
    accelerator_configs = {
        Accelerator.Yoda_512: ArchitectureConfig(
            256, 2, Block(2, 2, 8), Block(2, 2, 8), 48, [8, 8, 8, 8, 16, 8, 16, 20], 8
        ),
        Accelerator.Yoda_256: ArchitectureConfig(
            256, 1, Block(2, 2, 8), Block(2, 2, 8), 48, [8, 8, 8, 8, 16, 8, 16, 20], 8
        ),
        Accelerator.Ethos_U55_256: ArchitectureConfig(
            256, 1, Block(2, 2, 8), Block(2, 2, 8), 48, [8, 8, 8, 8, 16, 8, 16, 20], 8
        ),
        Accelerator.Ethos_U55_128: ArchitectureConfig(
            128, 1, Block(2, 1, 8), Block(2, 2, 8), 24, [4, 4, 4, 4, 8, 4, 8, 12], 4
        ),
        Accelerator.Ethos_U55_64: ArchitectureConfig(
            64, 1, Block(1, 1, 8), Block(1, 1, 8), 16, [2, 2, 2, 2, 4, 4, 4, 8], 2
        ),
        Accelerator.Ethos_U55_32: ArchitectureConfig(
            32, 1, Block(1, 1, 4), Block(1, 1, 8), 16, [2, 2, 2, 2, 4, 4, 4, 4], 1
        ),
    }

    OFMSplitDepth = 16
    SubKernelMax = Block(8, 8, 65536)

    def __init__(
        self,
        vela_config: ConfigParser,
        accelerator_config,
        system_config,
        permanent_storage,
        override_block_config,
        block_config_limit,
        global_memory_clock_scale,
        max_blockdep,
        weight_estimation_scaling,
    ):
        accelerator_config = accelerator_config.lower()
        self.vela_config = vela_config
        if accelerator_config not in Accelerator.member_list():
            raise OptionError("--accelerator-config", self.accelerator_config, "Unknown accelerator configuration")
        self.accelerator_config = Accelerator(accelerator_config)
        accel_config = ArchitectureFeatures.accelerator_configs[self.accelerator_config]
        self.config = accel_config

        self.system_config = system_config
        self.is_yoda_system = self.accelerator_config in (Accelerator.Yoda_256, Accelerator.Yoda_512)

        self.max_outstanding_dma = 2 if self.is_yoda_system else 1
        self.max_outstanding_kernels = 3

        self.ncores = accel_config.cores
        self.ofm_ublock = accel_config.ofm_ublock
        self.ifm_ublock = accel_config.ifm_ublock
        self.ofm_block_max = Block(64, 32, 128)
        self.override_block_config = override_block_config
        self.block_config_limit = block_config_limit

        self.global_memory_clock_scale = global_memory_clock_scale
        if self.global_memory_clock_scale <= 0.0 or self.global_memory_clock_scale > 1.0:
            raise Exception(
                "Invalid global_memory_clock_scale = "
                + str(self.global_memory_clock_scale)
                + " (must be > 0.0 and <= 1.0)"
            )

        self.max_blockdep = max_blockdep
        self.weight_estimation_scaling = weight_estimation_scaling

        dpu_min_height = accel_config.ofm_ublock.height
        dpu_min_width = accel_config.ofm_ublock.width
        dpu_dot_product_width = 8
        dpu_min_ofm_channels = accel_config.ofm_ublock.depth

        self.num_elem_wise_units = accel_config.elem_units
        self.num_macs_per_cycle = dpu_min_height * dpu_min_width * dpu_dot_product_width * dpu_min_ofm_channels

        self.memory_clock_scales = np.zeros(MemArea.Size)
        self.memory_port_widths = np.zeros(MemArea.Size)

        # Get system configuration
        self.__read_sys_config(self.is_yoda_system)

        # apply the global memory clock scales to the individual ones from the system config
        for mem in MemArea.all():
            self.memory_clock_scales[mem] *= self.global_memory_clock_scale

        self.memory_clocks = self.memory_clock_scales * self.npu_clock
        self.memory_bandwidths_per_cycle = self.memory_port_widths * self.memory_clock_scales / 8

        self.memory_bandwidths_per_second = self.memory_bandwidths_per_cycle * self.npu_clock

        # sizes as N x H x W x C. we need to round up to these when allocating storage
        self.storage_rounding_quantums = {
            TensorFormat.Unknown: (1, 1, 1, 1),
            TensorFormat.WeightsCompressed: (1, 1, 1, 1),
            TensorFormat.NHWC: (1, 1, 1, 1),
            TensorFormat.NHCWB16: (1, 1, 1, 16),
        }

        # brick sizes as N x H x W x C. We have to fetch whole bricks at a time
        self.brick_sizes = {
            TensorFormat.Unknown: (1, 1, 1, 1),
            TensorFormat.WeightsCompressed: (1, 1, 1, 1),
            TensorFormat.NHWC: (1, 1, 1, 1),
            TensorFormat.NHCWB16: (1, 1, 1, 16),
        }

        self.default_weight_format = TensorFormat.WeightsCompressed
        self.default_feature_map_format = TensorFormat.NHWC

        # This is to ignore permanent_storage = On/OffChipflash for Yoda
        if not self.is_yoda_system and permanent_storage != MemArea.OffChipFlash:
            self.permanent_storage_mem_area = permanent_storage

        self.tensor_storage_mem_area = {
            # permanent mem_area
            TensorPurpose.Unknown: MemArea.Unknown,
            TensorPurpose.Weights: self.permanent_storage_mem_area,
            TensorPurpose.FeatureMap: self.feature_map_storage_mem_area,
            TensorPurpose.LUT: self.permanent_storage_mem_area,
        }

        self.tensor_storage_mem_type = {
            TensorPurpose.Unknown: MemType.Unknown,
            TensorPurpose.Weights: MemType.Permanent_NPU,
            TensorPurpose.FeatureMap: MemType.Scratch,
            TensorPurpose.LUT: MemType.Scratch,
        }

        self.min_block_sizes = {
            NpuBlockType.Default: (dpu_min_height, dpu_min_width),
            NpuBlockType.VectorProduct: (1, 1),
            NpuBlockType.ConvolutionMxN: (dpu_min_height, dpu_min_width),
            NpuBlockType.Pooling: (dpu_min_height, dpu_min_width),
            NpuBlockType.ConvolutionDepthWise: (dpu_min_height, dpu_min_width),
            NpuBlockType.ElementWise: (1, 1),
            NpuBlockType.ReduceSum: (dpu_min_height, dpu_min_width),
        }

        self.sub_kernel_limits = {
            NpuBlockType.Default: (8, 8),
            NpuBlockType.VectorProduct: (1, 1),
            NpuBlockType.ConvolutionMxN: (8, 8),
            NpuBlockType.Pooling: (8, 8),
            NpuBlockType.ConvolutionDepthWise: (8, 8),
            NpuBlockType.ElementWise: (1, 1),
            NpuBlockType.ReduceSum: (8, 8),
        }

        # weights for scheduler search
        from .npu_performance import make_bandwidth_array

        self.bandwidth_weights = make_bandwidth_array()
        self.bandwidth_weights[MemArea.Sram] = 1.0
        self.bandwidth_weights[MemArea.Dram] = 10.0
        self.bandwidth_weights[MemArea.OnChipFlash] = 2.0
        self.bandwidth_weights[MemArea.OffChipFlash] = 20.0
        self.cycles_weight = 40
        self.max_sram_used_weight = 1000

        if self.is_yoda_system and (self.fast_storage_mem_area != self.feature_map_storage_mem_area):
            self.max_sram_used_weight = 0

        # Shared Buffer Block allocations
        self.shram_bank_size = 1024  # bytes
        self.shram_size_bytes = accel_config.shram_banks * self.shram_bank_size
        self.shram_reserved_output_banks = 2
        self.shram_reserved_weight_banks = 0
        self.shram_reserved_unused_banks = 2 if accel_config.shram_banks > 16 else 0
        self.shram_total_banks = accel_config.shram_banks - self.shram_reserved_unused_banks
        self.shram_bank_granules = np.array(accel_config.shram_granules, np.int32)
        self.shram_lut_size = 2048
        # SHRAM base address of the activation lookup table
        self.shram_lut_address = self.shram_bank_size * self.available_shram_banks(True)

        # Build a map of acceptable IFM/OFM block configurations up to the maximum
        # IFM/OFM block size.
        ifm_block_max = self.get_ifm_block_size(32, self.ofm_block_max, Kernel(8, 8))
        self.block_config_map = dict()
        self.generate_block_config_map(Block(ifm_block_max.width, ifm_block_max.height, 128))

        # Setup supported operators and restriction checkers class
        self.supported_operators = SupportedOperators()

    # Returns available number of SHRAM banks depending on activation lookup table
    # being used or not
    def available_shram_banks(self, uses_activation_lut):
        banks = self.shram_total_banks
        if uses_activation_lut and self.shram_reserved_unused_banks == 0:
            banks -= 2
        return banks

    # Calculate block configuration for ALL known IFM operations and
    # accumulator sizes. Consumers will need to select their preferred
    # operation and bit-width at read-time.
    def generate_block_config(self, width, height, depth):
        # Number of bytes required for any SHRAM element for a FM of given dimensions.
        # For IFM: size = H*W*Align(D*BYTE_WIDTH, 8)
        # For ACC: size = H*W*Align(D,8)*BYTE_WIDTH
        d1 = round_up(depth, SHRAMElements.PreAlign)
        d2 = round_up(d1 * SHRAMElements.ByteSizes, SHRAMElements.PostAlign)
        size_bytes = (height * width) * d2

        # Convert byte size (rounded) to size in banks
        size_banks = round_up_divide(size_bytes, self.shram_bank_size)
        size_banks *= 2  # Double buffer the IFM/Acc (need twice as many banks)
        # Round bank requirement to bank granularity
        required_banks = round_up(size_banks, self.shram_bank_granules)
        return SHRAMBlockConfig(size_bytes, required_banks)

    @staticmethod
    def make_block_config_key(width, height, depth):
        return (int(height), int(width), int(depth))

    def get_block_config(self, width, height, depth):
        assert depth <= self.ofm_block_max.depth
        key = ArchitectureFeatures.make_block_config_key(width, height, depth)
        config = self.block_config_map.get(key, None)
        return config

    # Generate a key:value map of possible block configurations, where the
    # key is compounded from the block dimensions: 0x00HHWWCC
    def generate_block_config_map(self, block: Block):
        for h in range(1, block.height + 1):
            for w in range(1, block.width + 1):
                # All possible IFM/OFM depth values
                for c in [4, 8, 12, 16, 24, 32, 40, 48, 56, 64, 72, 80, 88, 96, 104, 112, 120, 128]:
                    key = ArchitectureFeatures.make_block_config_key(w, h, c)
                    self.block_config_map[key] = self.generate_block_config(w, h, c)

    def calc_ifm_block_depth(self, ifm_depth, ifm_bits):
        assert ifm_bits in (8, 16, 32)
        assert ifm_depth > 0
        ifm_depth = round_up(ifm_depth, self.ifm_ublock.depth)
        max_block_depth = 8 * 32 // ifm_bits
        return min(max_block_depth, ifm_depth)

    # Calculate the size of the IFM block given a depth, target OFM block and a kernel
    def get_ifm_block_size(
        self,
        ifm_block_depth,
        ofm_block: Block,
        kernel: Kernel,
        subkernel: Block = Block(8, 8, 65536),
        ifm_resampling_mode=resampling_mode.NONE,
    ):
        upscaling = 1 if ifm_resampling_mode == resampling_mode.NONE else 2
        # Height
        ifm_odd_2x_height_enable = 0
        dilated_kernel_height = ((kernel.height - 1) * kernel.dilation.y) + 1
        ifm_block_height = (
            (ofm_block.height - 1) * kernel.stride.y
            + min(subkernel.height, dilated_kernel_height)
            + ifm_odd_2x_height_enable
        ) // upscaling

        ifm_block_height = round_up(ifm_block_height, self.ofm_ublock.height)

        # Width
        ifm_odd_2x_width_enable = 0
        dilated_kernel_width = ((kernel.width - 1) * kernel.dilation.x) + 1
        ifm_block_width = (
            (ofm_block.width - 1) * kernel.stride.x
            + min(subkernel.width, dilated_kernel_width)
            + ifm_odd_2x_width_enable
        ) // upscaling

        ifm_block_width = round_up(ifm_block_width, self.ofm_ublock.width)

        return Block(ifm_block_width, ifm_block_height, ifm_block_depth)

    @staticmethod
    def intersects(start_a, end_a, start_b, end_b):
        start_x = max(start_a[0], start_b[0])
        end_x = min(end_a[0], end_b[0])
        start_y = max(start_a[1], start_b[1])
        end_y = min(end_a[1], end_b[1])
        start_z = max(start_a[2], start_b[2])
        end_z = min(end_a[2], end_b[2])
        return ((end_x - start_x) > 0) and ((end_y - start_y) > 0) and ((end_z - start_z) > 0)

    # Block job dependency:
    # Does the VOLUME of IFMs for block job B(0) overlap with VOLUME of OFMs block jobs A(8,9,10)
    #
    #  A                    | B
    # ----------------------+------------------
    # .... 3,4,5,6,7,8,9,10 | 0,1,2,3,4,5,6,8 10 < JOB NUMBER
    #               |<------->| dependency offset
    #
    MAX_BLOCKDEP = 3

    # Get the coordinates of a block offset from either the end (negative)
    # or the start (zero or positive) of the given 3d area
    def get_offset_block_coords(self, area: Rect, block: Block, offset):
        size = area.size()
        # Dimensions of the region, in blocks
        width_blocks = round_up_divide(size.width, block.width)
        height_blocks = round_up_divide(size.height, block.height)
        depth_blocks = round_up_divide(size.depth, block.depth)
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

        return (coord_x + area.x, coord_y + area.y, coord_z + area.z)

    def get_first_job_input_volume(
        self, ifm: Rect, ofm: Rect, ifm_block_depth, ofm_block: Block, kernel: Kernel, padLT, block_offset
    ):
        # Get ifm block size (jobs are invisibly decomposed into subkernels)
        ifm_block = self.get_ifm_block_size(ifm_block_depth, ofm_block, kernel, self.ofm_block_max)
        ifm_depth_blocks = round_up_divide(ifm.size().depth, ifm_block_depth)

        # Which OFM block are we calculating
        ofm_coord = self.get_offset_block_coords(ofm, ofm_block, block_offset // ifm_depth_blocks)
        if ofm_coord is None:
            return None

        # Coordinate of the source IFM block
        ifm_coord_x = max(0, ofm_coord[0] * kernel.stride.x - padLT[0])
        ifm_coord_y = max(0, ofm_coord[1] * kernel.stride.y - padLT[1])
        ifm_coord_z = ifm.z + (block_offset % ifm_depth_blocks) * ifm_block.depth

        # IFM block that will be sampled for the FIRST+block_offset job in the next operator's OFM
        start_coord = (ifm_coord_x, ifm_coord_y, ifm_coord_z)
        end_coord = (
            start_coord[0] + ifm_block.width,
            start_coord[1] + ifm_block.height,
            start_coord[2] + ifm_block.depth,
        )

        return (start_coord, end_coord, 1)  # start, end, total jobs

    def get_prev_job_output_volume(
        self, ifm: Block, ofm: Rect, ifm_block_depth, ofm_block: Block, kernel: Kernel, block_offset
    ):
        assert block_offset >= 0

        # Get OFM block's volume coordinates
        start_coord = self.get_offset_block_coords(ofm, ofm_block, -1 - block_offset)
        if start_coord is None:
            return None
        end_coord = (
            start_coord[0] + ofm_block.width,
            start_coord[1] + ofm_block.height,
            start_coord[2] + ofm_block.depth,
        )

        # Calculate how many IFM blocks this OFM block requires (i.e how many jobs)
        ifm_depth_blocks = round_up_divide(ifm.size().depth, ifm_block_depth)
        ifm_depth_blocks = 1  # Overwrite with 1 to force OFM block dependency, not IFM

        return (start_coord, end_coord, ifm_depth_blocks)  # start, end, total jobs for this OFM block

    def calc_block_dep(
        self,
        prev_ifm: Block,
        prev_ofm: Block,
        prev_ifm_block_depth,
        prev_ofm_block: Block,
        prev_kernel: Kernel,
        ifm: Block,
        ofm: Block,
        ifm_block_depth,
        ofm_block: Block,
        kernel: Kernel,
        padLT,
    ):

        blockdep = ArchitectureFeatures.MAX_BLOCKDEP

        # Iterate over the next BLOCKDEP inputs, checking to see if a sliding window
        # of IFM area overlaps with any previous OFM block generation.
        elapsed_jobs = 0
        for forward_offset in range(ArchitectureFeatures.MAX_BLOCKDEP):
            # This is the IFM block we want to sample from
            in_area = self.get_first_job_input_volume(
                ifm, ofm, ifm_block_depth, ofm_block, kernel, padLT, forward_offset
            )
            if in_area is None:
                break

            # Try several previous-OFM blocks in the past (they still might comprise multiple IFM jobs)
            outstanding_jobs = 0
            for block_offset in range(ArchitectureFeatures.MAX_BLOCKDEP):
                # This is the OFM block being generated by the previous op
                out_area = self.get_prev_job_output_volume(
                    prev_ifm, prev_ofm, prev_ifm_block_depth, prev_ofm_block, prev_kernel, block_offset
                )
                if out_area is None:
                    break

                # Block dependency is the max number of allowed outstanding jobs
                # in the pipeline. Selected by determining how many jobs occur
                # in between two operators' overlapping OFM->IFM block volumes
                if ArchitectureFeatures.intersects(in_area[0], in_area[1], out_area[0], out_area[1]):
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

    def cpu_cycle_estimate(self, op):
        """
        Gets estimated performance of a CPU operation, based on a linear model of intercept, slope,
        specified in the vela config file, in ConfigParser file format (.ini file).
        Example configuration snippet:
        [CpuPerformance.MyOperationType]
        Cortex-Mx.intercept=<some float value>
        Cortex-Mx.slope=<some float value>
        """
        section = "CpuPerformance." + op.type
        if self.vela_config is not None and section in self.vela_config:
            op_config = self.vela_config[section]
            try:
                intercept = float(op_config.get(self.cpu_config + ".intercept", op_config["default.intercept"]))
                slope = float(op_config.get(self.cpu_config + ".slope", op_config["default.slope"]))
                n_elements = op.inputs[0].elements()
                cycles = intercept + n_elements * slope
                return cycles
            except Exception:
                print("Error: Reading CPU cycle estimate in vela configuration file, section {}".format(section))
                raise

        print("Warning: No configured CPU performance estimate for", op.type)
        return 0

    def __read_sys_config(self, is_yoda_system):
        """
        Gets the system configuration with the given name from the vela configuration file
        Example configuration snippet:
        [SysConfig.MyConfigName]
        npu_freq=<some float value>
        cpu=Cortex-Mx
        ...
        """
        # Get system configuration from the vela configuration file
        if self.vela_config is None:
            print("Warning: Using default values for system configuration")
        else:
            section_key = "SysConfig." + self.system_config
            if section_key not in self.vela_config:
                raise OptionError("--system-config", self.system_config, "Unknown system configuration")

        try:
            self.npu_clock = float(self.__sys_config("npu_freq", "500e6"))
            self.cpu_config = self.__sys_config("cpu", "Cortex-M7")

            self.memory_clock_scales[MemArea.Sram] = float(self.__sys_config("Sram_clock_scale", "1"))
            self.memory_port_widths[MemArea.Sram] = int(self.__sys_config("Sram_port_width", "64"))

            self.memory_clock_scales[MemArea.OnChipFlash] = float(self.__sys_config("OnChipFlash_clock_scale", "1"))
            self.memory_port_widths[MemArea.OnChipFlash] = int(self.__sys_config("OnChipFlash_port_width", "64"))

            self.memory_clock_scales[MemArea.OffChipFlash] = float(
                self.__sys_config("OffChipFlash_clock_scale", "0.25")
            )
            self.memory_port_widths[MemArea.OffChipFlash] = int(self.__sys_config("OffChipFlash_port_width", "32"))

            self.memory_clock_scales[MemArea.Dram] = float(self.__sys_config("Dram_clock_scale", "1"))
            self.memory_port_widths[MemArea.Dram] = int(self.__sys_config("Dram_port_width", "32"))

            self.fast_storage_mem_area = MemArea[self.__sys_config("fast_storage_mem_area", "Sram")]
            self.feature_map_storage_mem_area = MemArea[self.__sys_config("feature_map_storage_mem_area", "Sram")]

            self.permanent_storage_mem_area = MemArea[self.__sys_config("permanent_storage_mem_area", "OffChipFlash")]
            if is_yoda_system:
                if self.permanent_storage_mem_area is not MemArea.Dram:
                    raise Exception(
                        "Invalid permanent_storage_mem_area = "
                        + str(self.permanent_storage_mem_area)
                        + " (must be 'DRAM' for Yoda)."
                    )
            else:
                if self.permanent_storage_mem_area not in set((MemArea.OnChipFlash, MemArea.OffChipFlash)):
                    raise Exception(
                        "Invalid permanent_storage_mem_area = "
                        + str(self.permanent_storage_mem_area)
                        + " (must be 'OnChipFlash' or 'OffChipFlash' for ethosu-55)."
                        " To store the weights and other constant data in SRAM on ethosu-55 select 'OnChipFlash'"
                    )

            self.sram_size = 1024 * int(self.__sys_config("sram_size_kb", "204800"))

        except Exception:
            print("Error: Reading System Configuration in vela configuration file, section {}".format(section_key))
            raise

    def __sys_config(self, key, default_value):
        """
        Gets the system configuration value with the given key from the vela config file.
        """
        if self.vela_config is None:
            return default_value
        section = "SysConfig." + self.system_config
        result = self.vela_config[section].get(key, None)
        if result is None:
            raise Exception("Error: System Configuration Missing key {} in section [{}] ".format(key, section))
        return result
