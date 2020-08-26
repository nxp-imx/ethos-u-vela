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
# Internal representation of a Neural Network Tensor.
import enum
import uuid

import numpy as np

from . import numeric_util
from .data_type import DataType
from .ethos_u55_regs.ethos_u55_regs import resampling_mode
from .operation import Operation
from .range_set import MemoryRangeSet


class MemType(enum.IntFlag):
    Unknown = 0
    Permanent_NPU = 1
    Permanent_CPU = 2
    Scratch = 3
    Scratch_fast = 4
    Size = Scratch_fast + 1

    def display_name(self):
        return ("Unknown", "Permanent_NPU", "Permanent_CPU", "Scratch", "Scratch_fast", "Size")[self.value]

    def identifier_name(self):
        return ("unknown", "permanent_npu", "permanent_cpu", "scratch", "scratch_fast", "size")[self.value]

    def all():
        return (MemType.Permanent_NPU, MemType.Permanent_CPU, MemType.Scratch, MemType.Scratch_fast)

    def __str__(self):
        return self.name


class MemArea(enum.IntFlag):
    Unknown = 0
    Sram = 1
    Dram = 2
    OnChipFlash = 3
    OffChipFlash = 4
    Shram = 5  # for LUT
    Size = Shram + 1

    def display_name(self):
        return ("Unknown", "SRAM", "DRAM", "On-chip Flash", "Off-chip Flash", "SHRAM", "Size")[self.value]

    def identifier_name(self):
        return ("unknown", "sram", "dram", "on_chip_flash", "off_chip_flash", "shram", "size")[self.value]

    def all():
        return (MemArea.Sram, MemArea.Dram, MemArea.OnChipFlash, MemArea.OffChipFlash, MemArea.Shram)

    def __str__(self):
        return self.name


class TensorPurpose(enum.IntFlag):
    Unknown = 0
    Weights = 1
    FeatureMap = 2
    Scratch = 3
    LUT = 4
    Size = 5

    def display_name(self):
        return ("Unknown", "Weights", "FeatureMap", "Scratch", "LUT", "Size")[self.value]

    def identifier_name(self):
        return ("unknown", "weights", "feature_map", "scratch", "lut", "size")[self.value]

    def all():
        return (TensorPurpose.Weights, TensorPurpose.FeatureMap)


class TensorSubPurpose(enum.Enum):
    Standard = 0
    DoubleBuffer = 1
    RollingBufferX = 2
    RollingBufferY = 3
    RollingBufferXY = 4

    def display_name(self):
        return ("Standard", "Double Buffer", "Rolling Buffer X", "Rolling Buffer Y", "Rolling Buffer XY")[self.value]

    def identifier_name(self):
        return ("standard", "double_buffer", "rolling_buffer_x", "rolling_buffer_y", "rolling_buffer_xy")[self.value]

    def all():
        return (
            TensorSubPurpose.Standard,
            TensorSubPurpose.DoubleBuffer,
            TensorSubPurpose.RollingBufferX,
            TensorSubPurpose.RollingBufferY,
            TensorSubPurpose.RollingBufferXY,
        )


class TensorFormat(enum.Flag):
    Unknown = 0
    WeightsCompressed = 1
    NHWC = 2
    NHCWB16 = 3

    def __str__(self):
        return self.name


class TensorBlockTraversal(enum.Enum):
    Default = 0
    DepthWise = 1
    DepthFirst = 2
    PartKernelFirst = 3


def shape_num_elements(shp):
    elems = 1
    if shp is None:
        return None
    for d in shp:
        if d is None:
            return None
        elems *= d
    return elems


def shape_fully_defined(shp):
    if shp is None:
        return False
    for d in shp:
        if d is None:
            return False
    return True


def shape_round_to_quantum(shp, quantum):
    new_shp = list(shp)

    # Traverse backwards using length of shape since there may be more rounding quantums than shape elements
    for i in range(-1, -len(shp) - 1, -1):
        if new_shp[i] is not None:
            new_shp[i] = numeric_util.round_up(new_shp[i], quantum[i])
    return new_shp


class QuantizationParameters:
    __slots__ = "min", "max", "num_bits", "narrow_range", "scale_f32", "zero_point", "quant_min", "quant_max"

    def __init__(self, min=None, max=None, num_bits=None, narrow_range=None):
        self.min = min
        self.max = max

        self.num_bits = num_bits
        self.narrow_range = narrow_range

        self.scale_f32 = None
        self.zero_point = None
        self.quant_min = None
        self.quant_max = None

    def __str__(self):
        return "<nng.QuantizationParameters min=%s max=%s, num_bits=%s, scale=%s, zero_point=%s>" % (
            self.min,
            self.max,
            self.num_bits,
            self.scale_f32,
            self.zero_point,
        )

    __repr__ = __str__

    def clone(self):
        res = QuantizationParameters()
        res.min = self.min
        res.max = self.max

        res.num_bits = self.num_bits
        res.narrow_range = self.narrow_range

        res.scale_f32 = self.scale_f32
        res.zero_point = self.zero_point
        res.quant_min = self.quant_min
        res.quant_max = self.quant_max
        return res

    def dequantize(self, values):
        if self.zero_point.size == 1 and self.scale_f32.size == 1:
            # same scale is used for all values
            res = (values.astype(np.float64) - self.zero_point) * self.scale_f32
        else:
            # a different scale is used for different sets of values
            values_as_float = values.astype(np.float64)

            # this is not compatible with the format of depthwise weights,
            # where input is at index 3 (Output, Kh, Kw, Input)
            # return the quantized values
            return np.ndarray((values_as_float.shape))

            shape = values_as_float.shape[0]
            assert self.zero_point.size == self.scale_f32.size == shape
            res = np.ndarray(values_as_float.shape)
            for i in range(shape):
                res[i] = (values_as_float[i] - self.zero_point[i]) * self.scale_f32[i]

        return res

    def is_scaling_equal(self, other):
        if other is None or not isinstance(other, QuantizationParameters):
            return False

        return self.scale_f32 == other.scale_f32 and self.zero_point == other.zero_point


def create_const_tensor(name, shape, dtype, values, value_dtype=None, purpose=TensorPurpose.Unknown, quantization=None):
    # Tensor
    const_tensor = Tensor(shape, dtype, name + "_0")
    const_tensor.purpose = purpose
    const_tensor.quantization = quantization
    const_tensor.values = np.array(values, dtype=value_dtype)
    const_tensor.quant_values = np.frombuffer(const_tensor.values.tobytes(), dtype=np.uint8)
    # Operator
    const_op = Operation("Const", name)
    const_op.set_output_tensor(const_tensor)
    return const_tensor


def create_reshape_tensor(tens, shape, ifm_reshape=True):
    if shape == tens.shape:
        return tens
    # Tensors
    name = tens.name + "_reshape"
    reshape_ifm = tens
    reshape_ofm = tens.clone("_reshaped")
    reshape_ofm.set_all_shapes(shape)
    if not ifm_reshape:
        reshape_ifm, reshape_ofm = reshape_ofm, reshape_ifm
    # Operator
    reshape_op = Operation("Reshape", name)
    reshape_op.attrs["new_shape"] = shape
    reshape_op.add_input_tensor(reshape_ifm)
    reshape_op.add_input_tensor(create_const_tensor(name + "_shape", [1], DataType.int32, shape))
    reshape_op.set_output_tensor(reshape_ofm)
    return reshape_ofm if ifm_reshape else reshape_ifm


class Tensor:
    __slots__ = (
        "shape",
        "storage_shape",
        "bandwidth_shape",
        "dtype",
        "name",
        "ops",
        "consumer_list",
        "values",
        "quant_values",
        "compressed_values",
        "compressed_values_substream_offsets",
        "mem_area",
        "mem_type",
        "format",
        "purpose",
        "sub_purpose",
        "alignment",
        "weight_transpose_depthwise",
        "storage_compression_scale",
        "bandwidth_compression_scale",
        "compression_scale_for_worst_weight_stream",
        "weight_compression_scales",
        "weight_compression_config",
        "storage_rounding_quantum",
        "brick_size",
        "address",
        "quantization",
        "weight_compressed_offsets",
        "element_size_bytes",
        "block_traversal",
        "cpu_tensor",
        "npu_tensor",
        "equivalence_id",
        "resampling_mode",
        "avoid_NHCWB16",
    )
    AllocationQuantum = 16

    def __init__(self, shape, dtype, name):
        self.shape = shape
        self.storage_shape = shape
        self.bandwidth_shape = shape
        self.dtype = dtype
        self.name = name
        self.equivalence_id = uuid.uuid4()

        self.ops = []
        self.consumer_list = []
        # Below attributes are only set if a tensor has been cloned,
        # either from Cpu -> Npu or vice versa. Needed for offline allocation
        self.cpu_tensor = None  # reference to the corresponding Cpu tensor
        self.npu_tensor = None  # reference to the corresponding Npu tensor

        self.values = None
        self.quant_values = None
        self.compressed_values = None
        self.compressed_values_substream_offsets = None
        self.mem_area = MemArea.Unknown
        self.mem_type = MemType.Unknown
        self.format = TensorFormat.Unknown
        self.purpose = TensorPurpose.Unknown
        self.sub_purpose = TensorSubPurpose.Standard
        self.alignment = Tensor.AllocationQuantum
        self.weight_transpose_depthwise = False

        self.storage_compression_scale = 1.0
        self.bandwidth_compression_scale = 1.0
        self.compression_scale_for_worst_weight_stream = 1.0
        self.weight_compression_scales = None
        self.weight_compression_config = None
        self.weight_compressed_offsets = []
        self.storage_rounding_quantum = (1, 1, 1, 1)
        self.brick_size = (1, 1, 1, 1)
        self.address = None  # start address of tensor. will be filled in by tensor allocator
        self.element_size_bytes = 0

        # quantization parameters
        self.quantization = None
        self.block_traversal = TensorBlockTraversal.Default
        self.resampling_mode = resampling_mode.NONE

        self.avoid_NHCWB16 = False

    def element_size(self):
        if self.element_size_bytes == 0:
            return self.dtype.size_in_bits() / 8
        return self.element_size_bytes

    def clone(self, suffix="_clone"):
        res = Tensor(self.shape, self.dtype, self.name + suffix)
        res.storage_shape = list(self.storage_shape)
        res.bandwidth_shape = list(self.bandwidth_shape)

        res.ops = []
        res.consumer_list = []
        res.equivalence_id = self.equivalence_id

        res.values = self.values
        res.quant_values = self.quant_values
        res.mem_area = self.mem_area
        res.mem_type = self.mem_type
        res.format = self.format
        res.purpose = self.purpose
        res.sub_purpose = self.sub_purpose
        res.alignment = self.alignment
        res.bandwidth_compression_scale = self.bandwidth_compression_scale
        res.storage_rounding_quantum = self.storage_rounding_quantum
        res.address = None

        if self.quantization is not None:
            res.quantization = self.quantization.clone()
        else:
            res.quantization = None

        res.resampling_mode = self.resampling_mode

        res.copy_compressed_weight_info(self)
        res.avoid_NHCWB16 = self.avoid_NHCWB16
        return res

    def clone_into_fast_storage(self, arch):
        res = self.clone(suffix="_fast_storage")
        res.mem_area = arch.fast_storage_mem_area
        res.mem_type = MemType.Scratch_fast
        return res

    def copy_compressed_weight_info(self, src_tens):
        # Copies compressed values + all related weight compression info from the given tensor
        self.compressed_values = src_tens.compressed_values
        self.compressed_values_substream_offsets = src_tens.compressed_values_substream_offsets
        self.storage_shape = src_tens.storage_shape
        self.brick_size = src_tens.brick_size
        self.weight_compression_scales = src_tens.weight_compression_scales
        self.weight_compressed_offsets = src_tens.weight_compressed_offsets
        self.weight_transpose_depthwise = src_tens.weight_transpose_depthwise
        self.compression_scale_for_worst_weight_stream = src_tens.compression_scale_for_worst_weight_stream
        self.storage_compression_scale = src_tens.storage_compression_scale
        self.block_traversal = src_tens.block_traversal
        self.weight_compression_config = src_tens.weight_compression_config

    def set_format(self, fmt, arch):
        self.format = fmt
        shape_len = 0
        try:
            shape_len = len(self.shape)
        except TypeError:
            pass

        self.storage_rounding_quantum = arch.storage_rounding_quantums[self.format]
        self.storage_rounding_quantum = self.storage_rounding_quantum[-shape_len:]
        self.brick_size = arch.brick_sizes[self.format]
        self.brick_size = self.brick_size[-shape_len:]
        if self.shape is None:
            return

        self.bandwidth_shape = shape_round_to_quantum(self.shape, self.brick_size)
        self.storage_shape = shape_round_to_quantum(self.shape, self.storage_rounding_quantum)

        if fmt == TensorFormat.WeightsCompressed:
            compression_ratio = 5 / 8
            self.storage_compression_scale = compression_ratio
            self.bandwidth_compression_scale = compression_ratio
            self.compression_scale_for_worst_weight_stream = compression_ratio

    def storage_elements(self):
        elems = shape_num_elements(self.storage_shape)
        if elems is None:
            return 0
        return elems

    def elements(self):
        elems = shape_num_elements(self.shape)
        if elems is None:
            return 0
        return elems

    def has_fully_defined_shape(self):
        return shape_fully_defined(self.shape)

    def storage_size(self, scale=1.0):
        raw_size = self.storage_elements() * self.element_size() * scale
        if raw_size == 0:
            raw_size = 1  # force it to take up space
        rounded_size = numeric_util.round_up(numeric_util.round_up_to_int(raw_size), self.alignment)
        return rounded_size

    def storage_size_for_sub_purpose(self, arch, sub_purpose, param_a=None, param_b=None):
        alt_shape = self.storage_shape_for_sub_purpose(sub_purpose, param_a, param_b)
        elems = shape_num_elements(alt_shape)
        if elems is None:
            return 0
        if sub_purpose == TensorSubPurpose.DoubleBuffer:
            raw_size = (
                elems
                * self.element_size()
                * self.compression_scale_for_worst_weight_stream
                * arch.weight_estimation_scaling
            )
        else:
            # Rolling buffers are used for intermediate data in ifm streaming
            # These will all use the NHCWB16 format, and need to be aligned to 16 in the C-dimension
            if alt_shape[-1] % 16 != 0:
                nhcwb16_shape = alt_shape[0:-1] + [numeric_util.round_up(alt_shape[-1], 16)]
                elems = shape_num_elements(nhcwb16_shape)

            raw_size = elems * self.element_size() * self.storage_compression_scale
        rounded_size = numeric_util.round_up(numeric_util.round_up_to_int(raw_size), self.alignment)
        return rounded_size

    def storage_shape_for_sub_purpose(self, sub_purpose, param_a, param_b):
        if sub_purpose == TensorSubPurpose.DoubleBuffer:
            shp = list(self.shape)
            assert len(shp) >= 2
            shp[-1] = min(shp[-1], param_a * 2)
        else:
            shp = list(self.storage_shape)
            if sub_purpose == TensorSubPurpose.RollingBufferX:
                assert len(shp) == 4
                shp[0] = 1
                shp[2] = min(shp[2], param_a)
            elif sub_purpose == TensorSubPurpose.RollingBufferY:
                assert len(shp) == 4
                shp[0] = 1
                shp[1] = min(shp[1], param_a)
            elif sub_purpose == TensorSubPurpose.RollingBufferXY:
                assert len(shp) == 4
                shp[0] = 1
                shp[2] = min(shp[2], param_a)
                shp[1] = min(shp[1], param_b)
            elif sub_purpose == TensorSubPurpose.Standard:
                pass
            else:
                assert 0, "did not expect new sub purpose %s" % (sub_purpose,)

        return shp

    def set_new_sub_purpose(self, sub_purpose, param_a=None, param_b=None):
        self.storage_shape = self.storage_shape_for_sub_purpose(sub_purpose, param_a, param_b)
        self.sub_purpose = sub_purpose
        if sub_purpose == TensorSubPurpose.DoubleBuffer:
            self.storage_compression_scale = self.compression_scale_for_worst_weight_stream

    def bandwidth(self):
        elems = shape_num_elements(self.bandwidth_shape)
        if elems is None:
            return 0
        return elems * self.element_size() * self.bandwidth_compression_scale

    def consumers(self):
        return self.consumer_list

    def get_address_ranges_for_coordinates(self, start_coord, end_coord):
        if self.sub_purpose in set(
            (TensorSubPurpose.RollingBufferX, TensorSubPurpose.RollingBufferY, TensorSubPurpose.RollingBufferXY)
        ):
            # build dummy coordinates that cover the entire buffer
            start_coord = [0] * len(start_coord)
            end_coord = [min(self.storage_shape[i], self.shape[i]) for i in range(len(end_coord))]

        start = self.address_for_coordinate(start_coord, is_top_box=False)
        end = self.address_for_coordinate(end_coord, is_top_box=True)
        return MemoryRangeSet(self.mem_area, start, end)

    def addresses_for_rolling_buffer(self, start_coord, end_coord):
        # returns ( box_height0, box_height1, box_width, [address_tl, address_tr, address_bl, address_br] )

        if len(start_coord) < 4:
            box_height0 = 1
            box_width = 1

            if len(start_coord) >= 2:
                box_width = end_coord[-2] - start_coord[-2]

            return box_height0, box_height0, box_width, [self.address_for_coordinate(start_coord), None, None, None]

        crossing_y = numeric_util.round_up(start_coord[1] + 1, self.storage_shape[1])
        crossing_x = numeric_util.round_up(start_coord[2] + 1, self.storage_shape[2])

        crossing_y = min(crossing_y, end_coord[1])
        crossing_x = min(crossing_x, end_coord[2])

        box_height0 = crossing_y - start_coord[1]
        box_width = crossing_x - start_coord[2]

        addresses = [None] * 4
        addresses[0] = self.address_for_coordinate(start_coord)

        if end_coord[2] > crossing_x:
            addresses[1] = self.address_for_coordinate([start_coord[0], start_coord[1], crossing_x, start_coord[3]])
            raise Exception("Striping in vertical direction is not supported")
        if end_coord[1] > crossing_y:
            addresses[2] = self.address_for_coordinate([start_coord[0], crossing_y, start_coord[2], start_coord[3]])
        if end_coord[1] > crossing_y and end_coord[2] > crossing_x:
            addresses[3] = self.address_for_coordinate([start_coord[0], crossing_y, crossing_x, start_coord[3]])

        return box_height0, box_height0, box_width, addresses

    def address_for_coordinate(self, coord, is_top_box=False):
        return self.address + self.address_offset_for_coordinate(coord, is_top_box)

    def get_strides_and_coord(self, coord=None):
        if coord is None:
            coord = [0] * len(self.storage_shape)

        augmented_coord = coord
        augmented_shape = self.storage_shape
        while len(augmented_shape) < 4:
            augmented_shape = [1] + augmented_shape

        while len(augmented_coord) < 4:
            augmented_coord = [0] + augmented_coord

        assert len(augmented_coord) == len(augmented_shape)

        if self.format == TensorFormat.NHWC:
            augmented_shape = [augmented_shape[0], augmented_shape[3]] + augmented_shape[1:3] + [1]
            augmented_coord = [augmented_coord[0], augmented_coord[3]] + augmented_coord[1:3] + [0]
            stride_order = [4, 1, 3, 2, 0]

        elif self.format == TensorFormat.NHCWB16:
            channel_divisor = 16
            augmented_shape = augmented_shape[0:4] + [1]
            augmented_coord = (
                [augmented_coord[0], augmented_coord[3] // channel_divisor]
                + augmented_coord[1:3]
                + [augmented_coord[3] % channel_divisor]
            )

            if augmented_shape[1] == 0:
                augmented_shape[1] = 1

        else:
            assert self.format in set((TensorFormat.Unknown, TensorFormat.WeightsCompressed))
            return None, None

        strides = [0] * len(augmented_shape)
        stride = self.element_size() * self.storage_compression_scale

        if self.format != TensorFormat.NHCWB16:
            for i in stride_order:
                strides[i] = stride
                stride *= augmented_shape[i]
        else:
            assert len(strides) == 5
            strides[4] = stride
            strides[3] = 16 * stride  # STRIDE_X
            strides[1] = strides[3] * augmented_shape[2]  # STRIDE_C
            strides[2] = augmented_shape[2] * augmented_shape[3] * stride  # STRIDE_Y
            strides[0] = strides[2] * augmented_shape[1]  # STRIDE_N

        return strides, augmented_coord

    def get_strides(self):
        strides, _ = self.get_strides_and_coord()

        return strides

    def needs_dma(self):
        return len(self.ops) == 1 and self.ops[0].type == "DMA"

    def get_dma_src_tensor(self):
        # For weight tensors that need DMA: returns the source tensor in Flash, else None
        # Note: for DMA ops, Pass.weight_tensor is referring to the SRAM weight tensor
        return self.ops[0].inputs[0] if self.needs_dma() else None

    def find_npu_op(self):
        # Returns the NPU operator that uses this tensor, excluding DMA operators.
        for op in self.consumers():
            if op.type == "DMA":
                return op.outputs[0].find_npu_op()
            if op.run_on_npu:
                return op
            return None

    def compressed_stream_index_from_coord(self, coord):
        assert self.format == TensorFormat.WeightsCompressed
        assert len(self.compressed_values) > 0
        assert len(self.compressed_values) + 1 == len(self.weight_compressed_offsets)

        depth = coord[-1]
        brick_depth = self.brick_size[-1]
        # Clamp position at final element index
        if depth > self.shape[-1]:
            depth = self.shape[-1]

        # Always round up to next boundary
        index = numeric_util.round_up_divide(depth, brick_depth)

        # Check boundaries on all but last weight set (which may be shorter
        # than the brick we divided it up into)
        if index < len(self.weight_compressed_offsets) - 1:
            # There are no half-way points in the weights
            if (depth % brick_depth) != 0:
                raise Exception("Offset into weights must be aligned to a brick")

        return index

    def size_of_compressed_stream(self, index):
        assert 0 <= index < len(self.compressed_values)
        return len(self.compressed_values[index])

    def is_last_index_in_compressed_stream(self, index):
        assert 0 <= index < len(self.compressed_values)
        return index == len(self.compressed_values) - 1

    def address_offset_for_coordinate(self, orig_coord, is_top_box=False):
        address_offset = 0
        coord = orig_coord

        coord = coord[-len(self.storage_shape) :]

        if self.sub_purpose == TensorSubPurpose.Standard:
            for idx, c in enumerate(coord):
                if is_top_box:
                    assert c > 0 and c <= self.shape[idx]
                else:
                    assert c >= 0 and c < self.shape[idx]

        if self.format == TensorFormat.WeightsCompressed:
            if len(self.weight_compressed_offsets) == 0:
                return 0

            if self.needs_dma() and self.sub_purpose == TensorSubPurpose.DoubleBuffer:
                depth = orig_coord[-1]
                brick_depth = self.brick_size[-1]
                # Clamp position at final element index
                if depth > self.shape[-1]:
                    depth = self.shape[-1]

                # Always round up to next boundary
                index = numeric_util.round_up_divide(depth, brick_depth)
                index = index % 2

                if len(self.compressed_values) <= 2:
                    if is_top_box and index == 0:
                        for cv in self.compressed_values:
                            address_offset += len(cv)
                    else:
                        address_offset = index * len(self.compressed_values[0])
                else:
                    if is_top_box and index == 0:
                        address_offset = self.storage_shape[-1]
                    else:
                        address_offset = index * (self.storage_shape[-1] // 2)
            else:
                index = self.compressed_stream_index_from_coord(orig_coord)
                assert index < len(self.weight_compressed_offsets)
                address_offset = self.weight_compressed_offsets[index]
        else:
            if is_top_box:
                coord = [c - 1 for c in coord]

            # handle wraparound for partial buffers. make sure to do this after subtracting top box:
            coord = [c % self.storage_shape[idx] for idx, c in enumerate(coord)]

            strides, augmented_coord = self.get_strides_and_coord(coord)
            if strides is None:
                return None

            if is_top_box:
                address_offset += 1 * strides[-1]  # one element

            address_offset += np.dot(augmented_coord, strides)

        assert address_offset >= 0
        assert address_offset <= self.storage_size()
        return address_offset

    def is_allocated_in_tensor_arena(self, scratch_tensor_mem_area):
        if self.mem_area == scratch_tensor_mem_area and (self.mem_type in set((MemType.Scratch, MemType.Scratch_fast))):
            return True
        return False

    def is_scaling_equal(self, tens):
        return self.quantization.is_scaling_equal(tens.quantization)

    def equivalent(self, tens):
        return self.equivalence_id == tens.equivalence_id

    def set_all_shapes(self, shape):
        self.shape = shape
        self.storage_shape = shape
        self.bandwidth_shape = shape

    def get_full_shape(self):
        d = len(self.shape)
        if d in (1, 3):
            return numeric_util.full_shape(4, self.shape, 1)
        elif d == 2:
            return [self.shape[0], 1, 1, self.shape[1]]
        else:
            return self.shape

    def __str__(self):
        return "<nng.Tensor '%s' shape=%s dtype=%s>" % (self.name, self.shape, self.dtype)

    __repr__ = __str__
