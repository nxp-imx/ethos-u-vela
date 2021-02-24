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
# Defines the class Shape4D.
from collections import namedtuple

from .numeric_util import full_shape
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

    def add(self, n, h, w, c):
        return Shape4D(self.batch + n, self.height + h, self.width + w, self.depth + c)

    def __add__(self, rhs):
        return Shape4D(self.batch + rhs.batch, self.height + rhs.height, self.width + rhs.width, self.depth + rhs.depth)

    def __sub__(self, rhs):
        return Shape4D(self.batch - rhs.batch, self.height - rhs.height, self.width - rhs.width, self.depth - rhs.depth)

    def __floordiv__(self, rhs):
        return Shape4D(
            self.batch // rhs.batch, self.height // rhs.height, self.width // rhs.width, self.depth // rhs.depth
        )

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

    def elements_wh(self):
        return self.width * self.height

    def is_empty(self):
        return (self.batch + self.width + self.height + self.depth) == 0

    def as_list(self):
        return list(self)

    def get_hw_as_list(self):
        return list([self.height, self.width])
