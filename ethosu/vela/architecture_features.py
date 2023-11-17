# SPDX-FileCopyrightText: Copyright 2020-2023 Arm Limited and/or its affiliates <open-source-office@arm.com>
# Copyright 2023 NXP
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
# Holds a container for Ethos-U and System architecture parameters.
import enum
from collections import namedtuple
from configparser import ConfigParser

import numpy as np

from .api import NpuAccelerator
from .errors import CliOptionError
from .errors import ConfigOptionError
from .ethos_u55_regs.ethos_u55_regs import resampling_mode
from .numeric_util import full_shape
from .numeric_util import round_up
from .numeric_util import round_up_divide
from .numeric_util import round_up_to_int
from .operation import Kernel
from .operation import NpuBlockType
from .operation import PointXYZ
from .tensor import BandwidthDirection
from .tensor import MemArea
from .tensor import MemType
from .tensor import TensorFormat
from .tensor import TensorPurpose
from .tflite_supported_operators import TFLiteSupportedOperators
from .tosa_supported_operators import TosaSupportedOperators


class Block:
    def __init__(self, w=0, h=0, d=0):
        self.width = w
        self.height = h
        self.depth = d

    def elements(self):
        return self.width * self.height * self.depth

    def elements_wh(self):
        return self.width * self.height

    def clone(self):
        return Block(self.width, self.height, self.depth)

    def as_list(self):
        return [self.height, self.width, self.depth]

    def __eq__(self, other):
        if self.width == other.width and self.height == other.height and self.depth == other.depth:
            return True
        else:
            return False

    def __repr__(self):
        return "<Block: {0},{1},{2}>".format(self.width, self.height, self.depth)

    def to_hwc(self):
        return [self.height, self.width, self.depth]

    @classmethod
    def from_string(cls, s):
        w, h, c = (int(v) for v in s.split("x"))
        return cls(w, h, c)

    @classmethod
    def from_shape(cls, shape) -> "Block":
        """Converts the shape to a Block"""
        shp = full_shape(3, shape, 1)
        # Note: index from end, as len(shp) may be > 3
        return Block(shp[-2], shp[-3], shp[-1])

    @classmethod
    def min(cls, a, b):
        return cls(min(a.width, b.width), min(a.height, b.height), min(a.depth, b.depth))

    @classmethod
    def max(cls, a, b):
        return cls(max(a.width, b.width), max(a.height, b.height), max(a.depth, b.depth))

    @classmethod
    def round(cls, a, b):
        return cls(round_up(a.width, b.width), round_up(a.height, b.height), round_up(a.depth, b.depth))

    @classmethod
    def div_round_up(cls, a, b):
        return cls(
            round_up_divide(a.width, b.width), round_up_divide(a.height, b.height), round_up_divide(a.depth, b.depth)
        )


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


# Area indices must match Ethos-U SHRAM layout spec
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
    Ethos_U65_256 = "ethos-u65-256"
    Ethos_U65_512 = "ethos-u65-512"

    @classmethod
    def member_list(cls):
        return [e.value for e in cls]

    @classmethod
    def from_npu_accelerator(cls, npu_accelerator: NpuAccelerator) -> "Accelerator":
        """Converts the given public API object to Accelerator (used internally)"""
        accelerator_map = {
            NpuAccelerator.Ethos_U55_32: cls.Ethos_U55_32,
            NpuAccelerator.Ethos_U55_64: cls.Ethos_U55_64,
            NpuAccelerator.Ethos_U55_128: cls.Ethos_U55_128,
            NpuAccelerator.Ethos_U55_256: cls.Ethos_U55_256,
            NpuAccelerator.Ethos_U65_256: cls.Ethos_U65_256,
            NpuAccelerator.Ethos_U65_512: cls.Ethos_U65_512,
        }
        assert npu_accelerator in accelerator_map, f"Unsupported accelerator {npu_accelerator}"
        return accelerator_map[npu_accelerator]


@enum.unique
class MemPort(enum.Enum):
    Axi0 = enum.auto()
    Axi1 = enum.auto()


SHRAMConfig = namedtuple(
    "SHRAMConfig", ["reserved_output_banks", "bank_size_bytes", "total_banks", "reserved_end_banks"]
)


class ArchitectureFeatures:
    """This class is a container for various parameters of the Ethos-U core
    and system configuration that can be tuned, either by command line
    parameters or by the Ethos-U architects. The class is often passed
    around to passes that need to do architecture-dependent actions.

    Note the difference between ArchitectureFeatures and CompilerOptions
    - ArchitectureFeatures is for changing the Ethos-U and system architecture
    - CompilerOptions is for changing the behaviour of the compiler
    """

    ArchitectureConfig = namedtuple(
        "ArchitectureConfig", "macs cores ofm_ublock ifm_ublock shram_banks shram_granules elem_units"
    )
    accelerator_configs = {
        Accelerator.Ethos_U65_512: ArchitectureConfig(
            256, 2, Block(2, 2, 8), Block(2, 2, 8), 48, [8, 8, 8, 8, 16, 8, 16, 20], 8
        ),
        Accelerator.Ethos_U65_256: ArchitectureConfig(
            256, 1, Block(2, 2, 8), Block(2, 2, 8), 48, [8, 8, 8, 8, 16, 8, 16, 20], 8
        ),
        Accelerator.Ethos_U55_256: ArchitectureConfig(
            256, 1, Block(2, 2, 8), Block(2, 2, 8), 48, [8, 8, 8, 8, 16, 8, 16, 20], 8
        ),
        Accelerator.Ethos_U55_128: ArchitectureConfig(
            128, 1, Block(2, 1, 8), Block(2, 1, 8), 24, [4, 4, 4, 4, 8, 4, 8, 12], 4
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

    DEFAULT_CONFIG = "internal-default"
    MAX_BLOCKDEP = 3

    def __init__(
        self,
        vela_config_files,
        accelerator_config,
        system_config,
        memory_mode,
        max_blockdep,
        verbose_config,
        arena_cache_size,
    ):
        accelerator_config = accelerator_config.lower()
        if accelerator_config not in Accelerator.member_list():
            raise CliOptionError("--accelerator-config", self.accelerator_config, "Unknown accelerator configuration")
        self.accelerator_config = Accelerator(accelerator_config)
        accel_config = ArchitectureFeatures.accelerator_configs[self.accelerator_config]
        self.config = accel_config

        self.accumulator_granules = {
            SHRAMElements.Acc16: accel_config.shram_granules[SHRAMElements.Acc16],
            SHRAMElements.Acc32: accel_config.shram_granules[SHRAMElements.Acc32],
            SHRAMElements.Acc40: accel_config.shram_granules[SHRAMElements.Acc40],
        }

        self.ifm_bank_granules = {
            8: accel_config.shram_granules[SHRAMElements.IFM8],
            16: accel_config.shram_granules[SHRAMElements.IFM16],
            32: accel_config.shram_granules[SHRAMElements.IFM32],
        }

        self.ifm_ew_bank_granules = {
            8: accel_config.shram_granules[SHRAMElements.IFM8_Elementwise],
            16: accel_config.shram_granules[SHRAMElements.IFM16_Elementwise],
            32: accel_config.shram_granules[SHRAMElements.IFM32],
        }

        self.shram = SHRAMConfig(2, 1024, accel_config.shram_banks, 2 if accel_config.shram_banks > 16 else 0)

        self.system_config = system_config
        self.memory_mode = memory_mode
        self.is_ethos_u65_system = self.accelerator_config in (Accelerator.Ethos_U65_256, Accelerator.Ethos_U65_512)

        if self.is_ethos_u65_system:
            self.max_outstanding_dma = 2
            axi_port_address_width = 40
            axi_port_data_width = 128
        else:
            self.max_outstanding_dma = 1
            axi_port_address_width = 32
            axi_port_data_width = 64

        self.max_outstanding_kernels = 2

        self.ncores = accel_config.cores
        self.ofm_ublock = accel_config.ofm_ublock
        self.ifm_ublock = accel_config.ifm_ublock
        self.ofm_block_max = Block(64, 32, 128)

        self.max_blockdep = max_blockdep

        dpu_min_height = accel_config.ofm_ublock.height
        dpu_min_width = accel_config.ofm_ublock.width
        dpu_dot_product_width = 8
        dpu_min_ofm_channels = accel_config.ofm_ublock.depth

        self.num_elem_wise_units = accel_config.elem_units
        self.num_macs_per_cycle = dpu_min_height * dpu_min_width * dpu_dot_product_width * dpu_min_ofm_channels
        assert self.num_macs_per_cycle == accel_config.macs, f"{self.num_macs_per_cycle} != {accel_config.macs}"
        # Max value in address offsets
        self.max_address_offset = 1 << axi_port_address_width

        # Get system configuration and memory mode
        self._get_vela_config(vela_config_files, verbose_config, arena_cache_size)

        self.memory_bandwidths_per_cycle = axi_port_data_width * self.memory_clock_scales / 8

        self.memory_bandwidths_per_second = self.memory_bandwidths_per_cycle * self.core_clock

        # Get output/activation performance numbers
        self._generate_output_perf_tables(self.accelerator_config)

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

        self.tensor_storage_mem_area = {
            # permanent mem_area
            TensorPurpose.Unknown: MemArea.Unknown,
            TensorPurpose.Weights: self.permanent_storage_mem_area,
            TensorPurpose.FeatureMap: self.feature_map_storage_mem_area,
            TensorPurpose.LUT: self.permanent_storage_mem_area,
            TensorPurpose.Scratch: self.feature_map_storage_mem_area,
            TensorPurpose.ScratchFast: self.fast_storage_mem_area,
        }

        self.tensor_storage_mem_type = {
            TensorPurpose.Unknown: MemType.Unknown,
            TensorPurpose.Weights: MemType.Permanent_NPU,
            TensorPurpose.FeatureMap: MemType.Scratch,
            TensorPurpose.LUT: MemType.Scratch,
            TensorPurpose.Scratch: MemType.Scratch,
            TensorPurpose.ScratchFast: MemType.Scratch_fast,
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

        if self.is_spilling_enabled():
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

        # Setup supported operators and restriction checkers class
        self.tflite_supported_operators = TFLiteSupportedOperators()
        self.tosa_supported_operators = TosaSupportedOperators()

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

    def _generate_output_perf_tables(self, accel_config):
        if accel_config == Accelerator.Ethos_U55_32:
            self.output_cycles_per_elem = (2.0, 3.0, 3.0, 3.0, 4.0, 6.0, 1.0, 2.0)
            self.activation_cycles_per_elem = (1.0, 1.0, 0.0)
        elif accel_config == Accelerator.Ethos_U55_64:
            self.output_cycles_per_elem = (1.0, 1.5, 1.5, 1.5, 2.0, 3.0, 0.5, 1.0)
            self.activation_cycles_per_elem = (1.0, 1.0, 0.0)
        elif accel_config == Accelerator.Ethos_U55_128:
            self.output_cycles_per_elem = (0.75, 1.25, 0.75, 0.75, 1.0, 1.5, 0.25, 0.5)
            self.activation_cycles_per_elem = (1.0, 0.5, 0.0)
        elif accel_config in (Accelerator.Ethos_U55_256, Accelerator.Ethos_U65_256):
            self.output_cycles_per_elem = (0.625, 1.125, 0.5, 0.375, 0.5, 0.75, 0.125, 0.25)
            self.activation_cycles_per_elem = (1.0, 0.25, 0.0)
        else:
            assert accel_config == Accelerator.Ethos_U65_512
            self.output_cycles_per_elem = (0.3125, 0.5625, 0.25, 0.1875, 0.25, 0.375, 0.0625, 0.125)
            self.activation_cycles_per_elem = (0.5, 0.125, 0.0)

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
        dilated_kernel_height = ((kernel.height - 1) * kernel.dilation.y) + 1
        ifm_block_height = round_up_to_int(
            ((ofm_block.height - 1) * kernel.stride.y + min(subkernel.height, dilated_kernel_height)) / upscaling
        )

        ifm_block_height = round_up(ifm_block_height, self.ifm_ublock.height)

        # Width
        dilated_kernel_width = ((kernel.width - 1) * kernel.dilation.x) + 1
        ifm_block_width = round_up_to_int(
            ((ofm_block.width - 1) * kernel.stride.x + min(subkernel.width, dilated_kernel_width)) / upscaling
        )

        ifm_block_width = round_up(ifm_block_width, self.ifm_ublock.width)

        return Block(ifm_block_width, ifm_block_height, ifm_block_depth)

    def is_spilling_enabled(self):
        """
        Spilling is a feature that allows the Ethos-U to use a dedicated SRAM as a cache for various types of data
        """
        return (
            self._mem_port_mapping(self.cache_mem_area) == MemArea.Sram and self.cache_mem_area != self.arena_mem_area
        )

    def mem_type_size(self, mem_type: MemType) -> int:
        """Returns size in bytes available for the given memory type. This is a hard limit."""
        if mem_type == MemType.Scratch_fast and self.is_spilling_enabled():
            # when accessing the scratch fast memory type with memory spilling enabled the arena_cache_size refers to
            # the cache memory area which is a hard limit
            return self.arena_cache_size
        else:
            # for all other memory types and modes the hard limit is the maximum possible address offset
            return self.max_address_offset

    def _mem_port_mapping(self, mem_port):
        mem_port_mapping = {MemPort.Axi0: self.axi0_port, MemPort.Axi1: self.axi1_port}
        return mem_port_mapping[mem_port]

    def _set_default_sys_config(self):
        # ArchitectureFeatures.DEFAULT_CONFIG values
        if self.is_ethos_u65_system:
            # Default Ethos-U65 system configuration
            # Ethos-U65 Client-Server: SRAM (16 GB/s) and DRAM (12 GB/s)
            self.core_clock = 1e9
            self.axi0_port = MemArea.Sram
            self.axi1_port = MemArea.Dram
            self.memory_clock_scales[MemArea.Sram] = 1.0
            self.memory_clock_scales[MemArea.Dram] = 0.75  # 3 / 4
            self.memory_burst_length[MemArea.Sram] = 32
            self.memory_burst_length[MemArea.Dram] = 128
            self.memory_latency[MemArea.Sram][BandwidthDirection.Read] = 32
            self.memory_latency[MemArea.Sram][BandwidthDirection.Write] = 32
            self.memory_latency[MemArea.Dram][BandwidthDirection.Read] = 500
            self.memory_latency[MemArea.Dram][BandwidthDirection.Write] = 250
        else:
            # Default Ethos-U55 system configuration
            # Ethos-U55 High-End Embedded: SRAM (4 GB/s) and Flash (0.5 GB/s)
            self.core_clock = 500e6
            self.axi0_port = MemArea.Sram
            self.axi1_port = MemArea.OffChipFlash
            self.memory_clock_scales[MemArea.Sram] = 1.0
            self.memory_clock_scales[MemArea.OffChipFlash] = 0.125  # 1 / 8
            self.memory_burst_length[MemArea.Sram] = 32
            self.memory_burst_length[MemArea.OffChipFlash] = 128
            self.memory_latency[MemArea.Sram][BandwidthDirection.Read] = 32
            self.memory_latency[MemArea.Sram][BandwidthDirection.Write] = 32
            self.memory_latency[MemArea.OffChipFlash][BandwidthDirection.Read] = 64
            self.memory_latency[MemArea.OffChipFlash][BandwidthDirection.Write] = 64

    def _set_default_mem_mode(self):
        # ArchitectureFeatures.DEFAULT_CONFIG values
        if self.is_ethos_u65_system:
            # Default Ethos-U65 memory mode
            # Dedicated SRAM: the SRAM is only for use by the Ethos-U
            # The non-SRAM memory is assumed to be read-writeable
            self.const_mem_area = MemPort.Axi1
            self.arena_mem_area = MemPort.Axi1
            self.cache_mem_area = MemPort.Axi0
            self.arena_cache_size = 384 * 1024
        else:
            # Default Ethos-U55 memory mode
            # Shared SRAM: the SRAM is shared between the Ethos-U and the Cortex-M software
            # The non-SRAM memory is assumed to be read-only
            self.const_mem_area = MemPort.Axi1
            self.arena_mem_area = MemPort.Axi0
            self.cache_mem_area = MemPort.Axi0
            self.arena_cache_size = self.max_address_offset

    def _get_vela_config(self, vela_config_files, verbose_config, arena_cache_size_from_cli):
        """
        Gets the system configuration and memory modes from one or more Vela configuration file(s) or uses some
        defaults.
        """

        # all properties are optional and are initialised to a value of 1 (or the equivalent)
        self.core_clock = 1
        self.axi0_port = MemArea(1)
        self.axi1_port = MemArea(1)
        self.memory_clock_scales = np.ones(MemArea.Size)
        self.memory_burst_length = np.ones(MemArea.Size, int)
        self.memory_latency = np.zeros((MemArea.Size, BandwidthDirection.Size), int)
        self.const_mem_area = MemPort(1)
        self.arena_mem_area = MemPort(1)
        self.cache_mem_area = MemPort(1)
        self.arena_cache_size = self.max_address_offset
        arena_cache_size_loc_text = "Default"

        # read configuration file(s)
        self.vela_config = None

        if vela_config_files is not None:
            self.vela_config = ConfigParser()
            self.vela_config.read(vela_config_files)

        # read system configuration
        sys_cfg_section = "System_Config." + self.system_config

        if self.vela_config is not None and self.vela_config.has_section(sys_cfg_section):
            self.core_clock = float(self._read_config(sys_cfg_section, "core_clock", self.core_clock))
            self.axi0_port = MemArea[self._read_config(sys_cfg_section, "axi0_port", self.axi0_port)]
            self.axi1_port = MemArea[self._read_config(sys_cfg_section, "axi1_port", self.axi1_port)]

            for mem_area in (self.axi0_port, self.axi1_port):
                self.memory_clock_scales[mem_area] = float(
                    self._read_config(
                        sys_cfg_section, mem_area.name + "_clock_scale", self.memory_clock_scales[mem_area]
                    )
                )
                self.memory_burst_length[mem_area] = int(
                    self._read_config(
                        sys_cfg_section, mem_area.name + "_burst_length", self.memory_burst_length[mem_area]
                    )
                )
                self.memory_latency[mem_area][BandwidthDirection.Read] = int(
                    self._read_config(
                        sys_cfg_section,
                        mem_area.name + "_read_latency",
                        self.memory_latency[mem_area][BandwidthDirection.Read],
                    )
                )
                self.memory_latency[mem_area][BandwidthDirection.Write] = int(
                    self._read_config(
                        sys_cfg_section,
                        mem_area.name + "_write_latency",
                        self.memory_latency[mem_area][BandwidthDirection.Write],
                    )
                )
        elif self.system_config == ArchitectureFeatures.DEFAULT_CONFIG:
            self._set_default_sys_config()

        elif vela_config_files is None:
            raise CliOptionError("--config", vela_config_files, "Vela config file not specified")

        else:
            raise CliOptionError(
                "--system-config",
                self.system_config,
                f"Section {sys_cfg_section} not found in Vela config file",
            )

        # read the memory mode
        mem_mode_section = "Memory_Mode." + self.memory_mode

        if self.vela_config is not None and self.vela_config.has_section(mem_mode_section):
            self.const_mem_area = MemPort[
                self._read_config(mem_mode_section, "const_mem_area", self.const_mem_area.name)
            ]
            self.arena_mem_area = MemPort[
                self._read_config(mem_mode_section, "arena_mem_area", self.arena_mem_area.name)
            ]
            self.cache_mem_area = MemPort[
                self._read_config(mem_mode_section, "cache_mem_area", self.cache_mem_area.name)
            ]
            found = []
            self.arena_cache_size = int(
                self._read_config(mem_mode_section, "arena_cache_size", self.arena_cache_size, found)
            )
            if found[-1]:
                arena_cache_size_loc_text = "Configuration file"

        elif self.memory_mode == ArchitectureFeatures.DEFAULT_CONFIG:
            self._set_default_mem_mode()

        elif vela_config_files is None:
            raise CliOptionError("--config", vela_config_files, "Vela config file not specified")

        else:
            raise CliOptionError(
                "--memory-mode",
                self.memory_mode,
                f"Section {mem_mode_section} not found in Vela config file",
            )

        # override sram to onchipflash
        if self._mem_port_mapping(self.const_mem_area) == MemArea.Sram:
            if self.const_mem_area == self.arena_mem_area == self.cache_mem_area:
                print(
                    "Info: Changing const_mem_area from Sram to OnChipFlash. This will use the same characteristics as"
                    " Sram."
                )
                if self.const_mem_area == MemPort.Axi0:
                    self.const_mem_area = MemPort.Axi1
                    self.axi1_port = MemArea.OnChipFlash
                else:
                    self.const_mem_area = MemPort.Axi0
                    self.axi0_port = MemArea.OnChipFlash
                self.memory_clock_scales[MemArea.OnChipFlash] = self.memory_clock_scales[MemArea.Sram]
                self.memory_burst_length[MemArea.OnChipFlash] = self.memory_burst_length[MemArea.Sram]
                self.memory_latency[MemArea.OnChipFlash] = self.memory_latency[MemArea.Sram]

        # override sram usage
        if arena_cache_size_from_cli is not None:
            self.arena_cache_size = arena_cache_size_from_cli
            arena_cache_size_loc_text = "CLI option"

        # check configuration
        if self._mem_port_mapping(self.const_mem_area) not in (
            MemArea.Dram,
            MemArea.OnChipFlash,
            MemArea.OffChipFlash,
        ):
            raise ConfigOptionError(
                "const_mem_area",
                self._mem_port_mapping(self.const_mem_area).name,
                "Dram or OnChipFlash or OffChipFlash",
            )

        if self._mem_port_mapping(self.arena_mem_area) not in (MemArea.Sram, MemArea.Dram):
            raise ConfigOptionError("arena_mem_area", self._mem_port_mapping(self.arena_mem_area).name, "Sram or Dram")

        if self._mem_port_mapping(self.cache_mem_area) != MemArea.Sram:
            raise ConfigOptionError("cache_mem_area", self._mem_port_mapping(self.cache_mem_area).name, "Sram")

        if self.arena_cache_size < 0:
            raise ConfigOptionError("arena_cache_size", self.arena_cache_size, ">= 0")
        if self.arena_cache_size > self.max_address_offset:
            raise ConfigOptionError(
                "arena_cache_size",
                f"{self.arena_cache_size}. Size is out of bounds, maximum is: {self.max_address_offset}",
            )

        # assign existing memory areas
        self.permanent_storage_mem_area = self._mem_port_mapping(self.const_mem_area)
        self.feature_map_storage_mem_area = self._mem_port_mapping(self.arena_mem_area)
        self.fast_storage_mem_area = self._mem_port_mapping(self.cache_mem_area)

        # display the system configuration and memory mode
        if verbose_config:
            print("Configuration files:")
            print(f"   {vela_config_files}")
            print(f"System Configuration ({self.system_config}):")
            print(f"   core_clock = {self.core_clock}")
            print(f"   axi0_port = {self.axi0_port.name}")
            print(f"   axi1_port = {self.axi1_port.name}")
            for mem in (MemArea.Sram, MemArea.Dram, MemArea.OnChipFlash, MemArea.OffChipFlash):
                print(f"   {mem.name}_clock_scales = {self.memory_clock_scales[mem]}")
                print(f"   {mem.name}_burst_length = {self.memory_burst_length[mem]}")
                print(f"   {mem.name}_read_latency = {self.memory_latency[mem][BandwidthDirection.Read]}")
                print(f"   {mem.name}_write_latency = {self.memory_latency[mem][BandwidthDirection.Write]}")

            print(f"Memory Mode ({self.memory_mode}):")
            print(f"   const_mem_area = {self.const_mem_area.name}")
            print(f"   arena_mem_area = {self.arena_mem_area.name}")
            print(f"   cache_mem_area = {self.cache_mem_area.name}")
            print(f"   arena_cache_size = {self.arena_cache_size} from {arena_cache_size_loc_text}")

            print("Architecture Settings:")
            print(f"   permanent_storage_mem_area = {self.permanent_storage_mem_area.name}")
            print(f"   feature_map_storage_mem_area = {self.feature_map_storage_mem_area.name}")
            print(f"   fast_storage_mem_area = {self.fast_storage_mem_area.name}")

    def _read_config(self, section, key, current_value, found=None):
        """
        Reads a given key from a particular section in the Vela config file. If the section contains the 'inherit'
        option then we recurse into the section specified. If inherited sections result in multiple keys for a
        particular option then the key from the parent section is used, regardless of the parsing order. if specified
        found should be an empty list that this function will append a True or False to the end of the list indicating
        whether the key was found or not.
        """
        if not self.vela_config.has_section(section):
            raise ConfigOptionError("section", f"{section}. The section was not found in the Vela config file(s)")

        result = str(current_value) if current_value is not None else None
        if found is not None:
            found.append(False)

        if self.vela_config.has_option(section, "inherit"):
            inheritance_section = self.vela_config.get(section, "inherit")
            # check for recursion loop
            if inheritance_section == section:
                raise ConfigOptionError(
                    "inherit",
                    f"{inheritance_section}. This references its own section and recursion is not allowed",
                )
            result = self._read_config(inheritance_section, key, result, found)

        if self.vela_config.has_option(section, key):
            result = self.vela_config.get(section, key)
            if found is not None:
                found.append(True)

        return result


# Cache for default arch instances, as these are expensive to create
default_arch_cache = dict()


def create_default_arch(accelerator: Accelerator) -> ArchitectureFeatures:
    """Creates architecture features object using default settings"""
    if accelerator not in default_arch_cache:
        default_arch_cache[accelerator] = ArchitectureFeatures(
            vela_config_files=None,
            accelerator_config=accelerator.value,
            system_config=ArchitectureFeatures.DEFAULT_CONFIG,
            memory_mode=ArchitectureFeatures.DEFAULT_CONFIG,
            max_blockdep=ArchitectureFeatures.MAX_BLOCKDEP,
            verbose_config=False,
            arena_cache_size=None,
        )
    return default_arch_cache[accelerator]
