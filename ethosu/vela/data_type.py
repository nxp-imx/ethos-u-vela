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
# Defines the basic numeric type classes for tensors.
import enum
from typing import Any

import numpy as np

from .numeric_util import round_up_divide


class BaseType(enum.Flag):
    Signed = 1
    Unsigned = 2
    Asymmetric = 4
    Int = 8
    SignedInt = Int | Signed
    UnsignedInt = Int | Unsigned
    AsymmSInt = Int | Asymmetric | Signed
    AsymmUInt = Int | Asymmetric | Unsigned
    Float = 16
    BFloat = 32
    Bool = 64
    String = 128
    Resource = 256
    Variant = 512
    Complex = 1024


class DataType:
    """Defines a data type. Consists of a base type, and the number of bits used for this type"""

    __slots__ = "type", "bits"

    int4: Any
    int8: Any
    int16: Any
    int32: Any
    int48: Any
    int64: Any
    uint8: Any
    uint16: Any
    uint32: Any
    uint64: Any
    quint4: Any
    quint8: Any
    quint12: Any
    quint16: Any
    quint32: Any
    qint4: Any
    qint8: Any
    qint12: Any
    qint16: Any
    qint32: Any
    float16: Any
    float32: Any
    float64: Any
    string: Any
    bool: Any
    resource: Any
    variant: Any
    complex64: Any
    complex128: Any

    def __init__(self, type_, bits):
        self.type = type_
        self.bits = bits

    def __eq__(self, other):
        return self.type == other.type and self.bits == other.bits

    def __hash__(self):
        return hash((self.type, self.bits))

    def size_in_bytes(self):
        return round_up_divide(self.bits, 8)

    def size_in_bits(self):
        return self.bits

    def __str__(self):
        stem, needs_format = DataType.stem_name[self.type]
        if not needs_format:
            return stem
        else:
            return stem % (self.bits,)

    __repr__ = __str__

    def as_numpy_type(self):
        numpy_dtype_code = {
            BaseType.UnsignedInt: "u",
            BaseType.SignedInt: "i",
            BaseType.Float: "f",
            BaseType.Complex: "c",
        }
        assert self.type in numpy_dtype_code, f"Failed to interpret {self} as a numpy dtype"
        return np.dtype(numpy_dtype_code[self.type] + str(self.size_in_bytes())).type

    stem_name = {
        BaseType.UnsignedInt: ("uint%s", True),
        BaseType.SignedInt: ("int%s", True),
        BaseType.AsymmUInt: ("quint%s", True),
        BaseType.AsymmSInt: ("qint%s", True),
        BaseType.Float: ("float%s", True),
        BaseType.BFloat: ("bfloat%s", True),
        BaseType.Bool: ("bool", False),
        BaseType.String: ("string", False),
        BaseType.Resource: ("resource", False),
        BaseType.Variant: ("variant", False),
        BaseType.Complex: ("complex%s", True),
    }


# generate the standard set of data types
DataType.int4 = DataType(BaseType.SignedInt, 4)
DataType.int8 = DataType(BaseType.SignedInt, 8)
DataType.int16 = DataType(BaseType.SignedInt, 16)
DataType.int32 = DataType(BaseType.SignedInt, 32)
DataType.int48 = DataType(BaseType.SignedInt, 48)
DataType.int64 = DataType(BaseType.SignedInt, 64)

DataType.uint8 = DataType(BaseType.UnsignedInt, 8)
DataType.uint16 = DataType(BaseType.UnsignedInt, 16)
DataType.uint32 = DataType(BaseType.UnsignedInt, 32)
DataType.uint64 = DataType(BaseType.UnsignedInt, 64)

DataType.quint4 = DataType(BaseType.AsymmUInt, 4)
DataType.quint8 = DataType(BaseType.AsymmUInt, 8)
DataType.quint12 = DataType(BaseType.AsymmUInt, 12)
DataType.quint16 = DataType(BaseType.AsymmUInt, 16)
DataType.quint32 = DataType(BaseType.AsymmUInt, 32)

DataType.qint4 = DataType(BaseType.AsymmSInt, 4)
DataType.qint8 = DataType(BaseType.AsymmSInt, 8)
DataType.qint12 = DataType(BaseType.AsymmSInt, 12)
DataType.qint16 = DataType(BaseType.AsymmSInt, 16)
DataType.qint32 = DataType(BaseType.AsymmSInt, 32)

DataType.float16 = DataType(BaseType.Float, 16)
DataType.float32 = DataType(BaseType.Float, 32)
DataType.float64 = DataType(BaseType.Float, 64)

DataType.string = DataType(BaseType.String, 64)
DataType.bool = DataType(BaseType.Bool, 8)
DataType.resource = DataType(BaseType.Resource, 8)
DataType.variant = DataType(BaseType.Variant, 8)
DataType.complex64 = DataType(BaseType.Complex, 64)
DataType.complex128 = DataType(BaseType.Complex, 128)
