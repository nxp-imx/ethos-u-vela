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
# Contains classes that hold commands for the high-level command stream (one command per DMA or NPU stripe).
from typing import List

import numpy as np

from .architecture_features import Block
from .numeric_util import round_up_divide
from .operation import NpuBlockType
from .shape4d import Shape4D


class Box:
    def __init__(self, start_coord, end_coord):
        self.start_coord = list(start_coord)
        self.end_coord = list(end_coord)
        assert len(self.start_coord) == len(end_coord)
        for i in range(len(self.start_coord)):
            assert self.start_coord[i] <= self.end_coord[i]

    def transform_with_strides_and_skirt(
        self,
        strides: List[int],
        skirt: List[int],
        ifm_shape: Shape4D,
        npu_block_type: NpuBlockType,
        concat_axis: int = 0,
        concat_offset: int = 0,
        split_offset: Shape4D = None,
        k_height: int = 1,
        upscaling_factor: int = 1,
    ):
        new_start_coord = list(self.start_coord)
        new_end_coord = list(self.end_coord)

        new_start_coord[concat_axis] -= concat_offset
        new_end_coord[concat_axis] -= concat_offset

        if split_offset is not None:
            for idx in range(len(split_offset)):
                new_start_coord[idx] += split_offset[idx]
                new_end_coord[idx] += split_offset[idx]

        if (split_offset is None) and (
            npu_block_type in (NpuBlockType.ConvolutionMxN, NpuBlockType.VectorProduct, NpuBlockType.ReduceSum)
        ):
            # these types of operations do a "dot product" or sum over the entire IFM
            new_start_coord[-1] = 0
            new_end_coord[-1] = ifm_shape.depth

        if npu_block_type == NpuBlockType.ElementWise and len(new_end_coord) >= 1:
            new_end_coord[-1] = min(new_end_coord[-1], ifm_shape.depth)
        if len(new_end_coord) >= 2:
            new_end_coord[-2] = min(new_end_coord[-2], ifm_shape.width * upscaling_factor)
        if len(new_end_coord) >= 3:
            original_end_coord = list(new_end_coord)
            new_end_coord[-3] = min(new_end_coord[-3], ifm_shape.height * upscaling_factor)

        pad_top = 0
        pad_bottom = 0
        if strides is not None and skirt is not None:
            if len(new_start_coord) >= 2:
                stride = strides[2]
                new_start_coord[-2] = max(new_start_coord[-2] * stride - skirt[1], 0)
                new_end_coord[-2] = min(new_end_coord[-2] * stride + skirt[3], ifm_shape.width)

            if len(new_start_coord) >= 3:
                stride = strides[1]
                skirt_top_remainder = skirt[0] % upscaling_factor

                total_stride = stride * (new_end_coord[-3] - new_start_coord[-3] - 1)
                new_start_coord[-3] = new_start_coord[-3] * stride - skirt[0] + skirt_top_remainder

                pad_top = max(0, 0 - new_start_coord[-3]) + skirt_top_remainder
                new_start_coord[-3] = max(new_start_coord[-3], 0)

                if (new_end_coord[-3] * stride + skirt[2]) > (ifm_shape.height * upscaling_factor):
                    # pad_bottom is calculated based the diff between the end position of the weight kernel,
                    # after last stride and the ifm height.
                    if upscaling_factor != 1 and original_end_coord[-3] > ifm_shape.height * upscaling_factor:
                        # Special case for Transpose Convolution with VALID padding.
                        pad_bottom = original_end_coord[-3] - (ifm_shape.height * upscaling_factor)
                    else:
                        k_start = new_start_coord[-3] - pad_top
                        pad_bottom = max(0, k_start + total_stride + k_height - (ifm_shape.height * upscaling_factor))

                # Adjust for upscaling
                new_start_coord[-3] = max(new_start_coord[-3] // upscaling_factor, 0)
                new_end_coord[-3] = new_end_coord[-3] * stride + skirt[2] + (skirt[2] % upscaling_factor)
                new_end_coord[-3] = max(min(new_end_coord[-3] // upscaling_factor, ifm_shape.height), 1)

        return Box(new_start_coord, new_end_coord), pad_top, pad_bottom

    def make_weight_box(weight_shape, npu_block_type, oc_range_start=None, oc_range_end=None, weights_transposed=False):
        start = [0] * len(weight_shape)
        end = list(weight_shape)
        if oc_range_start is not None and oc_range_end is not None:
            if npu_block_type == NpuBlockType.ConvolutionDepthWise:
                # input range is output range divided by channel multiplier
                if weights_transposed:
                    start[-1] = oc_range_start // weight_shape[-2]
                    end[-1] = oc_range_end // weight_shape[-2]
                else:
                    start[-2] = oc_range_start // weight_shape[-1]
                    end[-2] = oc_range_end // weight_shape[-1]
            else:
                start[-1] = oc_range_start
                end[-1] = oc_range_end
        for i in range(len(end)):
            assert 0 <= start[i] < weight_shape[i]
            assert 0 < end[i] <= weight_shape[i]

        return Box(start, end)

    def get_size_shape(self):
        return [int(self.end_coord[i] - self.start_coord[i]) for i in range(len(self.end_coord))]

    def get_size(self):
        return int(np.prod(self.get_size_shape()))

    def get_block(self) -> Block:
        return Block.from_shape(self.get_size_shape())

    def __str__(self):
        return "<Box %s - %s>" % (self.start_coord, self.end_coord)

    __repr__ = __str__


class Command:
    def get_ofm_y_range_for_pass(self, ps_requested):
        return None

    def is_npu_pass_command(self):
        return False

    def get_operation_count(self):
        # returns numpy array of (DPU blocks, dma_ops).
        return np.array((0, 0))


class NpuStripe(Command):
    def __init__(
        self,
        ps,
        block_config,
        is_first,
        is_last,
        is_first_h_stripe,
        is_last_h_stripe,
        ifm_tensor,
        ifm_box,
        ofm_tensor,
        ofm_box,
        weight_tensor=None,
        weight_box=None,
        scale_tensor=None,
        concat_axis=0,
        concat_offset=0,
        ifm2_tensor=None,
        ifm2_box=None,
        pad_top=0,
        pad_bottom=0,
    ):
        self.ps = ps
        self.block_config = block_config
        self.is_first = is_first
        self.is_last = is_last
        self.is_first_h_stripe = is_first_h_stripe
        self.is_last_h_stripe = is_last_h_stripe
        self.ifm_tensor = ifm_tensor
        self.ifm_box = ifm_box
        self.ifm2_tensor = ifm2_tensor
        self.ifm2_box = ifm2_box
        self.ofm_tensor = ofm_tensor
        self.ofm_box = ofm_box
        self.weight_tensor = weight_tensor
        self.scale_tensor = scale_tensor
        self.weight_box = weight_box
        self.concat_axis = concat_axis
        self.concat_offset = concat_offset
        self.pad_top = pad_top
        self.pad_bottom = pad_bottom
        for i in range(len(self.ofm_box.end_coord)):
            assert self.ofm_box.end_coord[i] <= ps.ofm_shapes[0][i]

    def is_npu_pass_command(self):
        return True

    def __str__(self):
        return "<NPUStripe: ps=%s, ifm_box=%s, ifm2_box=%s, ofm_box=%s, weight_box=%s, block_config=%s>" % (
            self.ps.name,
            self.ifm_box,
            self.ifm2_box,
            self.ofm_box,
            self.weight_box,
            self.block_config,
        )

    __repr__ = __str__

    def get_ofm_y_range_for_pass(self, ps_requested):
        if ps_requested != self.ps:
            return None
        if len(self.ofm_box.start_coord) >= 3:
            return (self.ofm_box.start_coord[-3], self.ofm_box.end_coord[-3])
        return None

    def get_block_dimensions(self):
        ofm_box = self.ofm_box
        block_config = self.block_config

        out_height = 1
        out_width = 1
        out_depth = ofm_box.end_coord[-1] - ofm_box.start_coord[-1]
        if len(ofm_box.end_coord) >= 4:
            out_width = ofm_box.end_coord[-2] - ofm_box.start_coord[-2]
            out_height = ofm_box.end_coord[-3] - ofm_box.start_coord[-3]

        assert out_height >= 0
        assert out_width >= 0
        assert out_depth >= 0
        return (
            round_up_divide(out_height, block_config[0]),
            round_up_divide(out_width, block_config[1]),
            round_up_divide(out_depth, block_config[3]),
        )

    def get_operation_count(self):
        # returns numpy array of (DPU blocks, dma_ops)
        return np.array((self.get_n_blocks(), 0))

    def get_n_blocks(self):
        h, w, d = self.get_block_dimensions()
        res = h * w * d
        assert res >= 0
        return res


class DMA(Command):
    def __init__(self, ps, in_tensor, out_tensor, box):
        self.ps = ps
        self.in_tensor = in_tensor
        self.out_tensor = out_tensor
        self.box = box

    def __str__(self):
        return "<DMA: in=%s, out=%s, box=%s>" % (self.in_tensor.name, self.out_tensor.name, self.box)

    __repr__ = __str__

    def get_operation_count(self):
        # returns numpy array of (DPU blocks, dma_ops)
        return np.array((0, 1))
