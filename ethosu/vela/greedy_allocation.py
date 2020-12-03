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
# Allocate tensor addresses using a greedy algorithm.
from . import numeric_util


class GreedyAllocator:
    def __init__(self, nng, arch, live_ranges, mem_area):
        self.nng = nng
        self.arch = arch
        self.mem_area = mem_area

        self.live_ranges = live_ranges
        self.memory_required = 0

        self.current_allocs = []

    def alloc(self, new_lr):
        size = new_lr.size
        current_top = 0
        if self.current_allocs:
            current_top = max(start_addr + lr.size for start_addr, lr in self.current_allocs)
        best_offset = numeric_util.round_up(current_top, new_lr.get_alignment())
        best_offset_fit = (1 << 64) - 1

        aligned_size = numeric_util.round_up(size, new_lr.get_alignment())
        current_offset = 0
        for start_addr, lr in self.current_allocs:
            aligned_current_offset = numeric_util.round_up(current_offset, new_lr.get_alignment())
            if aligned_current_offset + aligned_size <= start_addr and start_addr - current_offset < best_offset_fit:
                best_offset = current_offset
                best_offset_fit = start_addr - current_offset

            current_offset = start_addr + lr.size

        best_offset = new_lr.set_address(best_offset)
        self.memory_required = max(self.memory_required, best_offset + aligned_size)
        self.current_allocs.append((best_offset, new_lr))
        self.current_allocs = list(sorted(self.current_allocs))

    def dealloc(self, lr_to_dealloc):
        self.current_allocs = [(start_addr, lr) for start_addr, lr in self.current_allocs if lr != lr_to_dealloc]

    def allocate_live_ranges(self, verbose_allocation, alignment):
        lrs = set()
        for lr in self.live_ranges.ranges.values():
            lrs.add((lr.start_time, -lr.end_time, lr))

        lrs = sorted(lrs)

        for curr_time, _, new_lr in lrs:
            for _, lr in list(self.current_allocs):
                if lr.end_time < curr_time:
                    self.dealloc(lr)

            self.alloc(new_lr)

        return self.memory_required


def allocate_live_ranges(nng, arch, live_ranges, mem_area, alignment, verbose_allocation=False):
    g = GreedyAllocator(nng, arch, live_ranges, mem_area)
    return g.allocate_live_ranges(verbose_allocation, alignment)
