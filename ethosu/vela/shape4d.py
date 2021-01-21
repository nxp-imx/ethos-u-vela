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
# Defines the class Shape4D.
from .numeric_util import full_shape


class Shape4D:
    """
    4D Shape (in NHWC format)
    """

    def __init__(self, shape, base=1):
        assert shape is not None
        assert len(shape) <= 4
        self._shape4D = tuple(full_shape(4, shape, base))

    def __str__(self):
        return f"<Shape4D {self.as_list()}>"

    def __eq__(self, other):
        return self._shape4D == other._shape4D

    def clone(self):
        return Shape4D(self.as_list())

    @property
    def batch(self):
        return self._shape4D[0]

    @property
    def height(self):
        return self._shape4D[1]

    @property
    def width(self):
        return self._shape4D[2]

    @property
    def depth(self):
        return self._shape4D[3]

    @batch.setter
    def batch(self, new_batch):
        self._shape4D = (new_batch, self._shape4D[1], self._shape4D[2], self._shape4D[3])

    @height.setter
    def height(self, new_height):
        self._shape4D = (self._shape4D[0], new_height, self._shape4D[2], self._shape4D[3])

    @width.setter
    def width(self, new_width):
        self._shape4D = (self._shape4D[0], self._shape4D[1], new_width, self._shape4D[3])

    @depth.setter
    def depth(self, new_depth):
        self._shape4D = (self._shape4D[0], self._shape4D[1], self._shape4D[2], new_depth)

    def get_dim(self, dim):
        assert -4 <= dim < 4
        return self._shape4D[dim]

    def as_list(self):
        return list(self._shape4D)

    def get_hw_as_list(self):
        return list([self.height, self.width])
