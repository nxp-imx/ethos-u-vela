# SPDX-FileCopyrightText: Copyright 2020, 2022 Arm Limited and/or its affiliates <open-source-office@arm.com>
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
# Helper classes to track memory accesses for calculating dependencies between Commands.
from enum import IntEnum
from functools import lru_cache


class RangeSet:
    """A Range set class to track ranges and whether they intersect.
    Intended for e.g. tracking sets of memory ranges and whether two commands use the same memory areas."""

    def __init__(self, start=None, end=None, ranges=None):
        if ranges is None:
            ranges = []

        self.ranges = ranges  # track a list of (start, end) tuples, always in ascending order sorted by start.

        if start is not None and start != end:
            assert start < end
            self.ranges.append((start, end))

    def __or__(self, other):
        combined_ranges = list(sorted(self.ranges + other.ranges))
        return RangeSet(ranges=combined_ranges)

    def __ior__(self, other):
        self.ranges = list(sorted(self.ranges + other.ranges))
        return self

    def intersects(self, other):
        a_ranges = self.ranges
        b_ranges = other.ranges

        a_idx = 0
        b_idx = 0

        while a_idx < len(a_ranges) and b_idx < len(b_ranges):
            ar = a_ranges[a_idx]
            br = b_ranges[b_idx]
            if max(ar[0], br[0]) < min(ar[1], br[1]):
                return True  # intersection

            # advance one of the two upwards
            if ar[0] < br[0]:
                a_idx += 1
            else:
                assert ar[0] != br[0]
                # note ar[0] == br[0] cannot happen, then we'd have an intersection
                b_idx += 1

        return False

    def __str__(self):
        return "<RangeSet %s>" % (["%#x:%#x" % (int(start), int(end)) for start, end in self.ranges],)

    __repr__ = __str__


class MemoryRangeSet:
    """Extended version of the RangeSet class that handles having different memory areas"""

    def __init__(self, mem_area=None, start=None, end=None, regions=None):

        if regions is None:
            regions = {}
        self.regions = regions

        if mem_area is not None:
            self.regions[mem_area] = RangeSet(start, end)

    def __or__(self, other):
        combined_regions = {
            mem_area: (self.regions.get(mem_area, RangeSet()) | other.regions.get(mem_area, RangeSet()))
            for mem_area in (self.regions.keys() | other.regions.keys())
        }
        return MemoryRangeSet(regions=combined_regions)

    def __ior__(self, other):
        self.regions = {
            mem_area: (self.regions.get(mem_area, RangeSet()) | other.regions.get(mem_area, RangeSet()))
            for mem_area in (self.regions.keys() | other.regions.keys())
        }
        return self

    def intersects(self, other):
        for mem_area in self.regions.keys() & other.regions.keys():
            if self.regions[mem_area].intersects(other.regions[mem_area]):
                return True
        return False

    def __str__(self):
        s = "<MemoryRangeSet>"
        for mem_area, rng in self.regions.items():
            s += "%s: %s\t" % (mem_area, rng)
        return s

    __repr__ = __str__


class AccessDirection(IntEnum):
    Read = 0
    Write = 1
    Size = 2


class MemoryAccessSet:
    """Tracks memory ranges, but also access patterns to know which accesses actually are in conflict"""

    def __init__(self):
        self.accesses = [MemoryRangeSet() for i in range(AccessDirection.Size)]

    def add(self, memory_range_set, access):
        self.accesses[access] |= memory_range_set

    @lru_cache(maxsize=None)
    def conflicts(self, other):

        # True dependencies, or write -> read
        if self.accesses[AccessDirection.Write].intersects(other.accesses[AccessDirection.Read]):
            return True

        # Anti-dependencies, or read -> write
        if self.accesses[AccessDirection.Read].intersects(other.accesses[AccessDirection.Write]):
            return True

        # Output dependencies, or write -> write
        if self.accesses[AccessDirection.Write].intersects(other.accesses[AccessDirection.Write]):
            return True

        # read -> read does not cause a conflict
        return False

    def __str__(self):
        return "Read: %s\nWrite: %s\n\n" % (self.accesses[AccessDirection.Read], self.accesses[AccessDirection.Write])

    __repr__ = __str__
