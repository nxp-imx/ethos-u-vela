# SPDX-FileCopyrightText: Copyright 2020-2021 Arm Limited and/or its affiliates <open-source-office@arm.com>
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
# Defines the class Shape4D.
from collections import namedtuple
from enum import Enum

from .numeric_util import full_shape
from .numeric_util import round_up
from .numeric_util import round_up_divide


class Shape4D(namedtuple("Shape4D", ["batch", "height", "width", "depth"])):
    """
    4D Shape (in NHWC format)
    """

    def __new__(cls, n=1, h=1, w=1, c=1):
        assert n is not None
        if isinstance(n, list):
            assert h == 1 and w == 1 and c == 1
            tmp = full_shape(4, n, 1)
            self = super(Shape4D, cls).__new__(cls, tmp[0], tmp[1], tmp[2], tmp[3])
        else:
            self = super(Shape4D, cls).__new__(cls, n, h, w, c)
        return self

    @classmethod
    def from_list(cls, shape, base=1):
        tmp = full_shape(4, shape, base)
        return cls(tmp[0], tmp[1], tmp[2], tmp[3])

    @classmethod
    def min(cls, lhs, rhs):
        return Shape4D(
            min(lhs.batch, rhs.batch), min(lhs.height, rhs.height), min(lhs.width, rhs.width), min(lhs.depth, rhs.depth)
        )

    @classmethod
    def max(cls, lhs, rhs):
        return Shape4D(
            max(lhs.batch, rhs.batch), max(lhs.height, rhs.height), max(lhs.width, rhs.width), max(lhs.depth, rhs.depth)
        )

    @classmethod
    def round_up(cls, lhs, rhs):
        return Shape4D(
            round_up(lhs.batch, rhs.batch),
            round_up(lhs.height, rhs.height),
            round_up(lhs.width, rhs.width),
            round_up(lhs.depth, rhs.depth),
        )

    @classmethod
    def from_hwc(cls, h, w, c):
        return cls(1, h, w, c)

    def with_batch(self, new_batch):
        return Shape4D(new_batch, self.height, self.width, self.depth)

    def with_height(self, new_height):
        return Shape4D(self.batch, new_height, self.width, self.depth)

    def with_width(self, new_width):
        return Shape4D(self.batch, self.height, new_width, self.depth)

    def with_hw(self, new_height, new_width):
        return Shape4D(self.batch, new_height, new_width, self.depth)

    def with_depth(self, new_depth):
        return Shape4D(self.batch, self.height, self.width, new_depth)

    def with_axis(self, axis, new_val):
        shape_as_list = self.as_list()
        shape_as_list[axis] = new_val
        return Shape4D.from_list(shape_as_list)

    @staticmethod
    def _clip_len(pos, length, size):
        if pos < 0:
            length = length + pos
            pos = 0
        return min(pos + length, size) - pos

    def clip(self, offset, sub_shape):
        n = Shape4D._clip_len(offset.batch, sub_shape.batch, self.batch)
        h = Shape4D._clip_len(offset.height, sub_shape.height, self.height)
        w = Shape4D._clip_len(offset.width, sub_shape.width, self.width)
        c = Shape4D._clip_len(offset.depth, sub_shape.depth, self.depth)
        return Shape4D(n, h, w, c)

    def add(self, n, h, w, c):
        return Shape4D(self.batch + n, self.height + h, self.width + w, self.depth + c)

    def __add__(self, rhs):
        return Shape4D(self.batch + rhs.batch, self.height + rhs.height, self.width + rhs.width, self.depth + rhs.depth)

    def __sub__(self, rhs):
        return Shape4D(self.batch - rhs.batch, self.height - rhs.height, self.width - rhs.width, self.depth - rhs.depth)

    def floordiv_const(self, const):
        return Shape4D(self.batch // const, self.height // const, self.width // const, self.depth // const)

    def __floordiv__(self, rhs):
        return Shape4D(
            self.batch // rhs.batch, self.height // rhs.height, self.width // rhs.width, self.depth // rhs.depth
        )

    def __truediv__(self, rhs):
        return Shape4D(self.batch / rhs.batch, self.height / rhs.height, self.width / rhs.width, self.depth / rhs.depth)

    def __mod__(self, rhs):
        return Shape4D(self.batch % rhs.batch, self.height % rhs.height, self.width % rhs.width, self.depth % rhs.depth)

    def __str__(self):
        return f"<Shape4D {list(self)}>"

    def div_round_up(self, rhs):
        return Shape4D(
            round_up_divide(self.batch, rhs.batch),
            round_up_divide(self.height, rhs.height),
            round_up_divide(self.width, rhs.width),
            round_up_divide(self.depth, rhs.depth),
        )

    def elements(self):
        return self.batch * self.width * self.height * self.depth

    def dot_prod(self, rhs):
        return self.batch * rhs.batch + self.width * rhs.width + self.height * rhs.height + self.depth * rhs.depth

    def elements_wh(self):
        return self.width * self.height

    def is_empty(self):
        return (self.batch + self.width + self.height + self.depth) == 0

    def as_list(self):
        return list(self)

    def get_hw_as_list(self):
        return list([self.height, self.width])


class VolumeIterator:
    """
    4D Volume iterator. Use to traverse 4D tensor volumes in smaller shapes.
    """

    class Direction(Enum):
        CWHN = 0

    def __init__(
        self,
        shape: Shape4D,
        sub_shape: Shape4D,
        start: Shape4D = Shape4D(0, 0, 0, 0),
        delta: Shape4D = None,
        dir=Direction.CWHN,
    ):
        self.b = start.batch
        self.y = start.height
        self.x = start.width
        self.z = start.depth
        self.shape = shape
        self.sub_shape = sub_shape
        self.delta = sub_shape if delta is None else delta
        assert self.delta.elements() > 0, "Iterator will not move"

    def __iter__(self):
        return self

    def __next__(self):
        if self.b >= self.shape.batch:
            raise StopIteration()

        offset = Shape4D(self.b, self.y, self.x, self.z)

        # CWHN
        self.z += self.delta.depth
        if self.z >= self.shape.depth:
            self.z = 0
            self.x += self.delta.width
            if self.x >= self.shape.width:
                self.x = 0
                self.y += self.delta.height
                if self.y >= self.shape.height:
                    self.y = 0
                    self.b += self.delta.batch

        return offset, self.shape.clip(offset, self.sub_shape)
