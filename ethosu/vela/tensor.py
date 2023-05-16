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
# Internal representation of a Neural Network Tensor.
import copy
import enum
import uuid
from collections import defaultdict
from enum import auto
from functools import lru_cache
from functools import total_ordering
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple
from typing import Union
from uuid import UUID

import numpy as np

from . import numeric_util
from .data_type import BaseType
from .data_type import DataType
from .errors import UnsupportedFeatureError
from .errors import VelaError
from .numeric_util import full_shape
from .operation import Op
from .operation import Operation
from .shape4d import Shape4D

Shape = List


class MemType(enum.IntFlag):
    Unknown = 0
    Permanent_NPU = 1
    Permanent_CPU = 2
    Scratch = 3
    Scratch_fast = 4
    Size = Scratch_fast + 1

    def display_name(self) -> str:
        return ("Unknown", "Permanent_NPU", "Permanent_CPU", "Scratch", "Scratch_fast", "Size")[self.value]

    def identifier_name(self) -> str:
        return ("unknown", "permanent_npu", "permanent_cpu", "scratch", "scratch_fast", "size")[self.value]

    @staticmethod
    def all():
        return (MemType.Permanent_NPU, MemType.Permanent_CPU, MemType.Scratch, MemType.Scratch_fast)

    def __str__(self):
        return self.name


class BandwidthDirection(enum.IntEnum):
    Read = 0
    Write = auto()
    Size = auto()

    def display_name(self):
        return self.name

    def identifier_name(self):
        return self.name.lower()

    @staticmethod
    def all():
        return (BandwidthDirection.Read, BandwidthDirection.Write)


class MemArea(enum.IntFlag):
    Unknown = 0
    Sram = 1
    Dram = 2
    OnChipFlash = 3
    OffChipFlash = 4
    Shram = 5  # for LUT
    Size = Shram + 1

    def display_name(self) -> str:
        return ("Unknown", "SRAM", "DRAM", "On-chip Flash", "Off-chip Flash", "SHRAM", "Size")[self.value]

    def identifier_name(self) -> str:
        return ("unknown", "sram", "dram", "on_chip_flash", "off_chip_flash", "shram", "size")[self.value]

    @staticmethod
    def all():
        return (MemArea.Sram, MemArea.Dram, MemArea.OnChipFlash, MemArea.OffChipFlash, MemArea.Shram)

    def __str__(self):
        return self.name


class TensorPurpose(enum.IntFlag):
    Unknown = 0
    Weights = 1
    FeatureMap = 2
    Scratch = 3
    ScratchFast = 4
    LUT = 5
    FSBias = 6
    Virtual = 7
    Size = 8

    def display_name(self) -> str:
        return ("Unknown", "Weights", "FeatureMap", "Scratch", "ScratchFast", "LUT", "FastStorageBias", "Size")[
            self.value
        ]

    def identifier_name(self) -> str:
        return ("unknown", "weights", "feature_map", "scratch", "scratch_fast", "lut", "fast_storage_bias", "size")[
            self.value
        ]

    @staticmethod
    def all():
        return (TensorPurpose.Weights, TensorPurpose.FeatureMap, TensorPurpose.FSBias)


class TensorSubPurpose(enum.Enum):
    Standard = 0
    DoubleBuffer = 1
    RollingBufferX = 2
    RollingBufferY = 3
    RollingBufferXY = 4

    def display_name(self) -> str:
        return ("Standard", "Double Buffer", "Rolling Buffer X", "Rolling Buffer Y", "Rolling Buffer XY")[self.value]

    def identifier_name(self) -> str:
        return ("standard", "double_buffer", "rolling_buffer_x", "rolling_buffer_y", "rolling_buffer_xy")[self.value]

    @staticmethod
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


def shape_num_elements(shp: Shape) -> Optional[int]:
    elems = 1
    if shp is None:
        return None
    for d in shp:
        if d is None:
            return None
        elems *= d
    return elems


def shape_fully_defined(shp: Shape) -> bool:
    if shp is None:
        return False
    for d in shp:
        if d is None:
            return False
    return True


def shape_round_to_quantum(shp: Shape, quantum: Tuple) -> Shape:
    new_shp = list(shp)

    # Traverse backwards using length of shape since there may be more rounding quantums than shape elements
    for i in range(-1, -len(shp) - 1, -1):
        if new_shp[i] is not None:
            new_shp[i] = numeric_util.round_up(new_shp[i], quantum[i])
    return new_shp


@lru_cache(maxsize=None)
def create_equivalence_id(key) -> UUID:
    # Generates equivalence_id based on the given key.
    return uuid.uuid4()


class QuantizationParameters:
    __slots__ = (
        "min",
        "max",
        "num_bits",
        "narrow_range",
        "scale_f32",
        "zero_point",
        "quant_min",
        "quant_max",
        "quant_dim",
    )

    def __init__(
        self,
        min: Union[float, np.ndarray, None] = None,
        max: Union[float, np.ndarray, None] = None,
        num_bits=None,
        narrow_range=None,
        scale_f32: Union[float, np.ndarray, None] = None,
        zero_point: Union[int, np.ndarray, None] = None,
    ):
        self.min = min
        self.max = max

        self.num_bits = num_bits
        self.narrow_range = narrow_range

        self.scale_f32: Union[float, np.ndarray, None] = scale_f32
        self.zero_point: Union[int, np.ndarray, None] = zero_point
        self.quant_min: Optional[float] = None
        self.quant_max: Optional[float] = None
        self.quant_dim: Optional[int] = None

    def __str__(self):
        return (
            f"<nng.QuantizationParameters min={self.min}, max={self.max}, num_bits={self.num_bits}, "
            f"scale={self.scale_f32}, zero_point={self.zero_point}>"
        )

    __repr__ = __str__

    def clone(self) -> "QuantizationParameters":
        res = QuantizationParameters()
        res.min = self.min
        res.max = self.max

        res.num_bits = self.num_bits
        res.narrow_range = self.narrow_range

        res.scale_f32 = self.scale_f32
        res.zero_point = self.zero_point
        res.quant_min = self.quant_min
        res.quant_max = self.quant_max
        res.quant_dim = self.quant_dim
        return res

    def dequantize(self, values) -> np.ndarray:
        return np.subtract(values, self.zero_point) * self.scale_f32

    def is_scaling_equal(self, other: Optional["QuantizationParameters"]) -> bool:
        """
        Returns True if the scale and zero point of self and other are equal. If other is None then the scaling is
        not considered equal because the tensor is assumed to not be quantised and False will be returned
        """

        if not isinstance(other, QuantizationParameters):
            return False

        return self.scale_f32 == other.scale_f32 and self.zero_point == other.zero_point

    def is_valid(self) -> bool:
        """Return True if the quantisation parameters have a scale and zero point"""

        return self.scale_f32 is not None and self.zero_point is not None

    def is_per_axis(self) -> bool:
        """Returns True if either the scale, zero point, minimum or maximum values have more than one value"""

        for attr in ("scale_f32", "zero_point", "min", "max"):
            if np.size(getattr(self, attr)) > 1:
                return True
        return False


def create_virtual_tensor(
    name: str,
):
    virtual_tensor = Tensor([], DataType.int8, name)
    virtual_tensor.purpose = TensorPurpose.Virtual
    return virtual_tensor


def create_const_tensor(
    name: str,
    shape: Shape,
    dtype: DataType,  # datatype of the tensor
    values: Optional[Union[np.ndarray, list]],  # list-like data of some type, or scalar (skip mypy), or None
    purpose: TensorPurpose = TensorPurpose.Unknown,
    quantization: Optional[QuantizationParameters] = None,
):
    assert isinstance(dtype, DataType)

    # Tensor
    const_tensor = Tensor(shape, dtype, name + "_0")
    const_tensor.purpose = purpose
    const_tensor.quantization = quantization

    # if the tensor datatype does not match that of the values then np.array() will perform a cast operation. this can
    # result in undefined behaviour if casting from a numpy float to a numpy unsigned integer. therefore, we need to
    # avoid this undefined behaviour by converting the numpy floats to python floats as these give the desired behaviour
    # when casting to unsigned integers
    if (
        values is not None
        and shape != []  # values are not a scalar
        and isinstance(values[0], np.floating)
        and dtype.type == BaseType.Unsigned
    ):
        values = [float(v) for v in values]

    const_tensor.values = np.array(values).astype(dtype.as_numpy_type())
    # Operator
    const_op = Operation(Op.Const, name)
    const_op.set_output_tensor(const_tensor)
    const_op.set_ifm_ofm_shapes()
    return const_tensor


# class that keeps track of all tensor addresses in the different memory types
class TensorAddressMap:
    address_map: Dict = defaultdict(dict)  # dict (tens.equivalence_id -> dict (mem_type -> address))

    @classmethod
    def get_address_for_tens(cls, tens_id: UUID, mem_type: MemType) -> int:
        return cls.address_map[tens_id].get(mem_type)

    @classmethod
    def set_address_for_tens(cls, tens_id: UUID, mem_type: MemType, address: int):
        # Check previous address if there is one
        previous_address = cls.address_map[tens_id].get(mem_type)
        if address is not None and previous_address is not None:
            assert previous_address == address, "Two different addresses cannot be assigned to the same tensor."

        # Set tensor's address for memory type
        cls.address_map[tens_id][mem_type] = address


@total_ordering
class Tensor:
    __slots__ = (
        "shape",
        "_original_shape",
        "storage_shape",
        "bandwidth_shape",
        "dtype",
        "name",
        "is_variable",
        "pre_buffer",
        "ops",
        "consumer_list",
        "values",
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
        "value_id",
        "storage_rounding_quantum",
        "brick_size",
        "quantization",
        "weight_compressed_offsets",
        "element_size_bytes",
        "block_traversal",
        "equivalence_id",
        "src_tensor",
        "force_linear_format",
        "ifm_write_protected",
    )
    AllocationQuantum = 16

    def __init__(self, shape: Shape, dtype: DataType, name: str):
        self.shape = shape
        self._original_shape = shape
        self.storage_shape = shape
        self.bandwidth_shape = shape
        self.dtype = dtype
        self.name = name
        self.is_variable = False
        self.pre_buffer = False
        self.equivalence_id: UUID = uuid.uuid4()

        self.ops: List[Operation] = []
        self.consumer_list: List[Operation] = []

        self.values: Optional[np.ndarray] = None  # elements are of type self.dtype
        self.compressed_values: Optional[np.ndarray] = None
        self.compressed_values_substream_offsets: Optional[List] = None
        self.mem_area: MemArea = MemArea.Unknown
        self.mem_type: MemType = MemType.Unknown
        self.format: TensorFormat = TensorFormat.Unknown
        self.purpose: TensorPurpose = TensorPurpose.Unknown
        self.sub_purpose: TensorSubPurpose = TensorSubPurpose.Standard
        self.alignment: int = Tensor.AllocationQuantum
        self.weight_transpose_depthwise: bool = False

        self.storage_compression_scale: float = 1.0
        self.bandwidth_compression_scale: float = 1.0
        self.compression_scale_for_worst_weight_stream: float = 1.0
        self.weight_compression_scales: Optional[np.ndarray] = None
        # if two tensors have the same weight_compression_config, then they have the same compressed values
        self.weight_compression_config = None
        # if two tensors have the same value_id, then they have the same values
        self.value_id: UUID = uuid.uuid4()
        self.weight_compressed_offsets: List = []
        self.storage_rounding_quantum: Tuple = (1, 1, 1, 1)
        self.brick_size: Tuple = (1, 1, 1, 1)
        self.element_size_bytes: int = 0

        # quantization parameters
        self.quantization: Optional[QuantizationParameters] = None
        self.block_traversal: TensorBlockTraversal = TensorBlockTraversal.Default

        # Keep track of whether the linear format should be enforced
        self.force_linear_format: Optional[bool] = None
        self.ifm_write_protected = False

        # Reference to parent-tensor if this tensor is a clone
        self.src_tensor: Optional[Tensor] = None

    @property
    def use_linear_format(self) -> bool:
        """Return whether the tensor should use linear format or not."""
        return self.force_linear_format in (True, None)

    @property
    def original_shape(self):
        return self._original_shape

    @property
    def address(self) -> int:
        return TensorAddressMap.get_address_for_tens(self.equivalence_id, self.mem_type)

    @address.setter
    def address(self, address: int):
        TensorAddressMap.set_address_for_tens(self.equivalence_id, self.mem_type, address)

    @property
    def is_standard_fm(self) -> bool:
        return self.sub_purpose == TensorSubPurpose.Standard and self.purpose == TensorPurpose.FeatureMap

    @property
    def is_const(self) -> bool:
        return self.ops != [] and self.ops[0].type == Op.Const

    @property
    def is_scalar(self) -> bool:
        return self.shape == [] and self.elements() == 1

    def is_broadcast(self, ofm) -> bool:
        return self.shape != ofm.shape

    def element_size(self) -> int:
        if self.element_size_bytes == 0:
            return self.dtype.size_in_bits() // 8
        return self.element_size_bytes

    # Returns a copy, renamed to self.name + suffix
    # The references to Operators will be empty when returned
    # Depending on set_unique, the copy is shallow, or deep
    # For set_unique==True, a new equivalence_id will be set
    def clone(self, suffix="_clone", set_unique: bool = False) -> "Tensor":
        res = copy.copy(self)
        if set_unique:
            res.equivalence_id = uuid.uuid4()
        res.storage_shape = list(self.storage_shape)
        res.bandwidth_shape = list(self.bandwidth_shape)
        if self.quantization is not None:
            res.quantization = self.quantization.clone()

        res.name = res.name + suffix
        res.ops = []
        res.consumer_list = []
        res.src_tensor = self

        return res

    def clone_into_shram(self, arch) -> "Tensor":
        res = self.clone(suffix="_shram")
        res.mem_area = MemArea.Shram
        res.src_tensor = self
        return res

    def as_1D(self):
        self.shape = [np.prod(self.shape)]
        if self.values is not None:
            self.values = self.values.reshape(self.shape)

    def transpose(self, reorder):
        self.shape = [self.shape[idx] for idx in reorder]
        self._original_shape = [self._original_shape[idx] for idx in reorder]
        if self.values is not None:
            self.values = self.values.transpose(reorder)

    def copy_compressed_weight_info(self, src_tens: "Tensor"):
        # Copies compressed values + all related weight compression info from the given tensor
        self.equivalence_id = src_tens.equivalence_id
        self.compressed_values = src_tens.compressed_values
        self.compressed_values_substream_offsets = src_tens.compressed_values_substream_offsets
        self.storage_shape = src_tens.storage_shape
        self.brick_size = src_tens.brick_size
        self.weight_compression_scales = src_tens.weight_compression_scales
        self.weight_compressed_offsets = src_tens.weight_compressed_offsets
        self.weight_transpose_depthwise = src_tens.weight_transpose_depthwise
        self.compression_scale_for_worst_weight_stream = src_tens.compression_scale_for_worst_weight_stream
        self.storage_compression_scale = src_tens.storage_compression_scale
        self.bandwidth_compression_scale = src_tens.bandwidth_compression_scale
        self.block_traversal = src_tens.block_traversal
        self.weight_compression_config = src_tens.weight_compression_config
        self.value_id = src_tens.value_id

    def set_format(self, fmt: TensorFormat, arch):
        self.format = fmt
        shape_len = 0
        try:
            shape_len = len(self.shape)
        except TypeError:
            pass

        if shape_len > 4:
            return
        assert not (self.use_linear_format and fmt == TensorFormat.NHCWB16)
        self.storage_rounding_quantum = arch.storage_rounding_quantums[self.format]
        self.storage_rounding_quantum = tuple(self.storage_rounding_quantum[-shape_len:])
        self.brick_size = arch.brick_sizes[self.format]
        self.brick_size = tuple(self.brick_size[-shape_len:])
        if self.shape is None:
            return

        self.bandwidth_shape = shape_round_to_quantum(self.shape, self.brick_size)
        self.storage_shape = shape_round_to_quantum(self.shape, self.storage_rounding_quantum)

        if fmt == TensorFormat.WeightsCompressed:
            compression_ratio = 5 / 8
            self.storage_compression_scale = compression_ratio
            self.bandwidth_compression_scale = compression_ratio
            self.compression_scale_for_worst_weight_stream = compression_ratio

    def storage_elements(self) -> int:
        elems = shape_num_elements(self.storage_shape)
        if elems is None:
            return 0
        return elems

    def elements(self) -> int:
        elems = shape_num_elements(self.shape)
        if elems is None:
            return 0
        return elems

    def has_fully_defined_shape(self) -> bool:
        return shape_fully_defined(self.shape)

    def storage_size(self, scale: float = 1.0) -> int:
        raw_size = self.storage_elements() * self.element_size() * scale
        if raw_size == 0:
            raw_size = 1  # force it to take up space
        rounded_size = numeric_util.round_up(numeric_util.round_up_to_int(raw_size), self.alignment)
        return rounded_size

    def storage_size_for_shape(self, op_storage_shape: Shape) -> int:
        elems = shape_num_elements(op_storage_shape)
        elems = elems if elems else 0
        raw_size = elems * self.element_size()
        if raw_size == 0:
            raw_size = 1  # force it to take up space
        rounded_size = numeric_util.round_up(numeric_util.round_up_to_int(raw_size), self.alignment)
        return rounded_size

    def storage_shape_for_sub_purpose(
        self, sub_purpose: TensorSubPurpose, param_a: Optional[int], param_b: Optional[int]
    ) -> Shape:
        if sub_purpose == TensorSubPurpose.DoubleBuffer:
            shp = list(self.shape)
            assert len(shp) >= 2
            assert param_a is not None
            shp[-1] = min(shp[-1], param_a * 2)
        else:
            shp = full_shape(4, self.storage_shape, 1)
            if sub_purpose == TensorSubPurpose.RollingBufferX:
                assert len(shp) == 4
                assert param_a is not None
                shp[0] = 1
                shp[2] = min(shp[2], param_a)
            elif sub_purpose == TensorSubPurpose.RollingBufferY:
                assert len(shp) == 4
                assert param_a is not None
                shp[0] = 1
                shp[1] = min(shp[1], param_a)
            elif sub_purpose == TensorSubPurpose.RollingBufferXY:
                assert len(shp) == 4
                assert param_a is not None
                assert param_b is not None
                shp[0] = 1
                shp[2] = min(shp[2], param_a)
                shp[1] = min(shp[1], param_b)
            elif sub_purpose == TensorSubPurpose.Standard:
                pass
            else:
                assert 0, "did not expect new sub purpose %s" % (sub_purpose,)

        return shp

    def set_new_sub_purpose(self, sub_purpose: TensorSubPurpose, param_a=None, param_b=None):
        self.storage_shape = self.storage_shape_for_sub_purpose(sub_purpose, param_a, param_b)
        self.sub_purpose = sub_purpose
        if sub_purpose == TensorSubPurpose.DoubleBuffer:
            self.storage_compression_scale = self.compression_scale_for_worst_weight_stream

    def bandwidth(self) -> float:
        elems = shape_num_elements(self.bandwidth_shape)
        if elems is None:
            return 0
        return elems * self.element_size() * self.bandwidth_compression_scale

    def consumers(self) -> List[Operation]:
        return self.consumer_list

    def get_4D_storage_shape_for_shape(self, op_shape4D: Shape4D) -> Shape4D:
        rounding_quantum = full_shape(4, list(self.storage_rounding_quantum), 1)
        return Shape4D(shape_round_to_quantum(op_shape4D.as_list(), rounding_quantum))

    def addresses_for_rolling_buffer(
        self, start_coord: Shape, end_coord: Shape, strides: List[int], op_shape4D: Shape4D
    ) -> Tuple:
        # returns ( box_height0, box_height1, box_width, [address_tl, address_tr, address_bl, address_br] )

        if self.storage_shape == []:
            return (
                1,
                1,
                1,
                [self.address_for_coordinate(start_coord, strides, op_shape4D), 0, 0, 0],
            )

        if self.is_standard_fm:
            storage_shape_4D = self.get_4D_storage_shape_for_shape(op_shape4D)
        else:
            storage_shape_4D = Shape4D(self.storage_shape)

        crossing_y = numeric_util.round_up(start_coord[1] + 1, storage_shape_4D.height)
        crossing_x = numeric_util.round_up(start_coord[2] + 1, storage_shape_4D.width)

        crossing_y = min(crossing_y, end_coord[1])
        crossing_x = min(crossing_x, end_coord[2])

        box_height0 = crossing_y - start_coord[1]
        box_width = crossing_x - start_coord[2]

        addresses: List = [0] * 4
        addresses[0] = self.address_for_coordinate(start_coord, strides, op_shape4D)

        if end_coord[2] > crossing_x:
            addresses[1] = self.address_for_coordinate(
                [start_coord[0], start_coord[1], crossing_x, start_coord[3]], strides, op_shape4D
            )
            raise UnsupportedFeatureError("Striping in vertical direction is not supported")
        if end_coord[1] > crossing_y:
            addresses[2] = self.address_for_coordinate(
                [start_coord[0], crossing_y, start_coord[2], start_coord[3]], strides, op_shape4D
            )
        if end_coord[1] > crossing_y and end_coord[2] > crossing_x:
            addresses[3] = self.address_for_coordinate(
                [start_coord[0], crossing_y, crossing_x, start_coord[3]], strides, op_shape4D
            )

        return box_height0, box_height0, box_width, addresses

    def get_strides(self, shape4D: Optional[Shape4D]) -> List[int]:

        augmented_shape = self.get_augmented_shape(shape4D)
        assert len(augmented_shape) == 5
        strides: List = [0] * len(augmented_shape)
        stride = self.element_size() * self.storage_compression_scale

        if self.format != TensorFormat.NHCWB16:
            stride_order = [4, 1, 3, 2, 0]
            for i in stride_order:
                strides[i] = stride
                stride *= augmented_shape[i]
        else:
            strides[4] = stride
            strides[3] = 16 * stride  # STRIDE_X
            strides[1] = strides[3] * augmented_shape[2]  # STRIDE_C
            strides[2] = augmented_shape[2] * augmented_shape[3] * stride  # STRIDE_Y
            strides[0] = strides[2] * augmented_shape[1]  # STRIDE_N

        return strides

    def get_augmented_shape(self, shape4D: Optional[Shape4D] = None) -> Optional[Shape]:

        if shape4D and self.is_standard_fm:
            augmented_shape = self.get_4D_storage_shape_for_shape(shape4D).as_list()
        else:
            augmented_shape = full_shape(4, self.storage_shape, 1)

        if self.format == TensorFormat.NHWC:
            augmented_shape = [augmented_shape[0], augmented_shape[3]] + augmented_shape[1:3] + [1]

        elif self.format == TensorFormat.NHCWB16:
            augmented_shape = augmented_shape[0:4] + [1]

            if augmented_shape[1] == 0:
                augmented_shape[1] = 1

        else:
            assert self.format in (TensorFormat.Unknown, TensorFormat.WeightsCompressed)
            return None

        return augmented_shape

    def get_augmented_coord(self, coord: Optional[Shape] = None) -> Optional[Shape]:
        if coord is None:
            coord = [0] * min(len(self.storage_shape), 4)

        missing_len = 4 - len(coord)
        augmented_coord = ([0] * missing_len) + coord

        if self.format == TensorFormat.NHWC:
            augmented_coord = [augmented_coord[0], augmented_coord[3]] + augmented_coord[1:3] + [0]

        elif self.format == TensorFormat.NHCWB16:
            channel_divisor = 16
            augmented_coord = (
                [augmented_coord[0], augmented_coord[3] // channel_divisor]
                + augmented_coord[1:3]
                + [augmented_coord[3] % channel_divisor]
            )
        else:
            assert self.format in (TensorFormat.Unknown, TensorFormat.WeightsCompressed)
            return None

        return augmented_coord

    def find_npu_op(self) -> Optional[Operation]:
        # Returns the NPU operator that uses this tensor
        for op in self.consumers():
            if op.run_on_npu:
                return op
        return None

    def compressed_stream_index_from_coord(self, coord: Shape) -> int:
        assert self.format == TensorFormat.WeightsCompressed
        assert self.compressed_values is not None
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
                raise UnsupportedFeatureError("Offset into weights must be aligned to a brick")

        return index

    def size_of_compressed_stream(self, index: int) -> int:
        assert self.compressed_values is not None
        assert 0 <= index < len(self.compressed_values)
        return len(self.compressed_values[index])

    def is_last_index_in_compressed_stream(self, index: int) -> bool:
        assert self.compressed_values is not None
        assert 0 <= index < len(self.compressed_values)
        return index == len(self.compressed_values) - 1

    def address_for_coordinate(
        self,
        orig_coord: Shape,
        strides: Optional[List[int]] = None,
        op_shape4D: Optional[Shape4D] = None,
        is_top_box: bool = False,
    ) -> Optional[int]:

        address_offset = 0
        assert self.purpose != TensorPurpose.Weights

        # Strides may be passed as an argument, for example when creating feature maps as the strides may be modified
        # by the "ofm_stride_multiplier" operation attribute. If not, they are calculated here.
        if not strides:
            strides = self.get_strides(op_shape4D)

        coord = orig_coord
        if is_top_box:
            coord = [c - 1 for c in orig_coord]
            address_offset += 1 * strides[-1]  # one element

        if self.sub_purpose == TensorSubPurpose.Standard:
            shape = op_shape4D.as_list() if op_shape4D else self.shape
            for _coord, _shape in zip(coord, shape):
                assert _coord >= 0 and _coord < _shape

        if op_shape4D and self.is_standard_fm:
            storage_shape = self.get_4D_storage_shape_for_shape(op_shape4D).as_list()
            storage_size = self.storage_size_for_shape(storage_shape)
        else:
            storage_shape = self.storage_shape
            coord = coord[-len(storage_shape) :]
            storage_size = self.storage_size()

        # Handle wraparound for partial buffers. Make sure to do this after subtracting top box
        coord = [_coord % _shape for _coord, _shape in zip(coord, storage_shape)]

        augmented_coord = self.get_augmented_coord(coord)
        assert augmented_coord is not None

        address_offset += np.dot(augmented_coord, strides)

        assert address_offset >= 0 and address_offset <= storage_size
        return self.address + address_offset

    def is_allocated_in_tensor_arena(self, scratch_tensor_mem_area: MemArea) -> bool:
        return (self.mem_area == scratch_tensor_mem_area) and (self.mem_type in (MemType.Scratch, MemType.Scratch_fast))

    def equivalent(self, tens: "Tensor") -> bool:
        return self.equivalence_id == tens.equivalence_id

    def set_all_shapes(self, shape: Shape):
        self.shape = shape
        self.storage_shape = shape
        self.bandwidth_shape = shape

    def get_full_shape(self) -> Shape:
        d = len(self.shape)
        if d in (1, 3):
            return full_shape(4, self.shape, 1)
        elif d == 2:
            return [self.shape[0], 1, 1, self.shape[1]]
        else:
            return self.shape.copy()

    def is_quantized(self) -> bool:
        # a tensor is quantized if it has an integral type and it contains valid quantization params

        if not isinstance(self.quantization, QuantizationParameters):
            return False

        return (self.dtype.type & BaseType.Int) != 0 and self.quantization.is_valid()

    def get_scalar(self):
        """
        return: Unquantized or dequantized scalar value
        rtype: self.dtype (if unquantized) or float (if dequantized)
        """
        assert self.values.size == 1, "get_scalar called on non-scalar tensor"
        if self.is_quantized():
            return self.quantization.dequantize(self.values).item(0)
        else:
            return self.values.item(0)

    def get_shape_as_2d(self, dimension_2_size: int) -> Optional[Shape4D]:

        elms = self.elements()
        dimension_1_size = elms // dimension_2_size
        # Checks if the reduction works and shape is not 1D
        is_reducible = dimension_1_size * dimension_2_size == elms and not (len(self.shape) == 1)

        new_shape = None
        if is_reducible:
            new_shape = Shape4D([dimension_1_size, 1, 1, dimension_2_size])

        return new_shape

    def __lt__(self, other: "Tensor") -> bool:
        return self.equivalence_id < other.equivalence_id

    def __str__(self):
        return "<nng.Tensor '%s' shape=%s dtype=%s>" % (self.name, self.shape, self.dtype)

    __repr__ = __str__

    def error(self, msg):
        """
        Raises a VelaError exception for errors encountered when parsing a Tensor

        :param self: Tensor object that resulted in the error
        :param msg: str object that contains a description of the specific error encountered
        """

        def _print_operators(ops):
            lines = []
            for idx, op in enumerate(ops):
                op_type = getattr(op, "type", "Not an Operation")
                op_id = getattr(op, "op_index", "-")
                lines.append(f"        {idx} = {op_type} ({op_id})")
            return lines

        lines = [f"Invalid {self.name} tensor. {msg}"]

        lines += ["    Driving operators:"]
        lines += _print_operators(self.ops)

        lines += ["    Consuming operators:"]
        lines += _print_operators(self.consumer_list)

        raise VelaError("\n".join(lines))


def check_quantized_tens_scaling_equal(tens_a: Tensor, tens_b: Tensor) -> bool:
    # checks that the scaling of two quantized tensors are equal

    return tens_a.is_quantized() and tens_b.is_quantized() and tens_a.quantization.is_scaling_equal(tens_b.quantization)
