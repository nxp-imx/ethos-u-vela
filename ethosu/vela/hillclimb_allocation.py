# SPDX-FileCopyrightText: Copyright 2021-2022 Arm Limited and/or its affiliates <open-source-office@arm.com>
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
# Tensor allocator based on a hill-climb search
import random
from typing import List
from typing import Set

from . import numeric_util
from .live_range import LiveRange


class LiveRangeInfo:
    def __init__(self, id: int, start_time: int, end_time: int, size: int, min_alignment: int):
        # Index of this live range
        self.id = id
        # Start time (input to the allocator algorithm)
        self.start_time = start_time
        # End time, inclusive (input to the allocator algorithm)
        self.end_time = end_time
        # Size in bytes (input to the allocator algorithm)
        self.size = size
        # Allocated address (the main output from the allocator algorithm)
        self.address: int = 0
        # End address, exclusive
        self.end_address: int = 0
        # id of predecessor live range (predecessor's end address == this lr's address)
        self.predecessor: int = 0
        # Turn at which the live range was allocated
        self.turn: int = 0
        # Max value of size_at_time (only used in the heuristic allocation)
        self.urgency = 0
        self.neighbours: List["LiveRangeInfo"] = []
        self.min_alignment = min_alignment

    def overlaps(self, addr2: int, size2: int) -> int:
        return self.address < addr2 + size2 and addr2 < self.end_address

    def is_neighbour(self, lr: "LiveRangeInfo") -> bool:
        return self.start_time <= lr.end_time and lr.start_time <= self.end_time

    def __str__(self):
        return "<LiveRangeInfo: id={}, start_time={}, end_time={}, size={}, address={}>".format(
            self.id, self.start_time, self.end_time, self.size, self.address
        )

    def __lt__(self, other) -> bool:
        if self.urgency != other.urgency:
            return self.urgency > other.urgency
        duration1 = self.end_time - self.start_time
        duration2 = other.end_time - other.start_time
        if duration1 != duration2:
            return duration1 > duration2

        if self.start_time != other.start_time:
            return self.start_time < other.start_time

        if self.size != other.size:
            return self.size > other.size

        return self.id < other.id


class HillClimbAllocator:
    """
    Implements tensor allocator using a hill climb search.

    The basic algorithm is:

    Use a heuristic allocator to find an initial allocation
    while allocation is not optimal and iterations < MAX_ITERATIONS:
        find the "bottleneck": the live range with highest end address
        find all live ranges that affected the allocation of the bottleneck
        swap the order of any two affecting live ranges
        reallocate tensors using the reordered live ranges
        if the new allocation is better: keep it, else set allocation to previous allocation
    """

    # Maximum number of iterations of the algorithm (can be override from the command line)
    MAX_ITERATIONS = 99999
    NOT_ALLOCATED = -1
    # Used for live ranges allocated at address 0
    NO_PREDECESSOR = -1
    # Special handling if best solution has not improved during this many iterations
    MAX_ITERATIONS_STUCK = 50
    # Minimum number of iterations since the last improvement (unless an optimal solution is found)
    MIN_ITERATIONS_IMPROVE = 500

    def __init__(self, live_ranges: List[LiveRange], max_iterations: int, memory_limit: int):
        # Contains the live ranges
        self.lrs: List[LiveRangeInfo] = [
            LiveRangeInfo(id, lr.start_time, lr.end_time, lr.size, lr.get_alignment())
            for id, lr in enumerate(live_ranges)
        ]
        self.lrs_at_time: List[List[LiveRangeInfo]] = []
        # The available size (input to algorithm).
        self.available_size: int = 0
        # The algorithm stops once the target size has been achieved
        self.target_size: int = 0
        # The highest end address of the best found allocation
        self.best_size: int = 1 << 63
        # For each live range: max value of size_at_time (only used in the heuristic allocation)
        self.lr_urgency = len(self.lrs) * [0]
        nr_time_slots = 1 + max(lr.end_time for lr in self.lrs)
        # Contains active live ranges at each timestamp
        self.lrs_at_time = [[] for i in range(nr_time_slots)]
        for lr in self.lrs:
            for t in range(lr.start_time, lr.end_time + 1):
                self.lrs_at_time[t].append(lr)
        # At each timestamp: accumulated size of active live ranges
        size_at_time = [sum(lr.size for lr in self.lrs_at_time[t]) for t in range(nr_time_slots)]
        # The minimum possible size, assuming all live ranges can be perfectly allocated
        self.min_required_size: int = max(size_at_time)
        # Set the maximum number of iterations the search will iterate to find a solution
        if max_iterations is None:
            self.max_iterations = self.MAX_ITERATIONS
        else:
            self.max_iterations = max_iterations
        # Defines a memory limit that the allocation must meet
        self.memory_limit = memory_limit
        if self.memory_limit < self.min_required_size:
            print(
                f"Warning: Memory limit = {self.memory_limit} is less than the minimum possible allocation size ="
                f" {self.min_required_size}"
            )
        # Calculate all neighbours + the urgency of each live range
        for lr in self.lrs:
            lr.urgency = 0
            lr.neighbours = []
            neighbours = set()
            for t in range(lr.start_time, lr.end_time + 1):
                lr.urgency = max(size_at_time[t], lr.urgency)
                for lr2 in self.lrs_at_time[t]:
                    if lr2 not in neighbours and lr != lr2:
                        neighbours.add(lr2)
                        lr.neighbours.append(lr2)

    def allocate_lr(self, lr: LiveRangeInfo):
        """
        Allocates the given live range at the smallest possible address
        """
        address = 0
        predecessor = HillClimbAllocator.NO_PREDECESSOR
        fits = False
        while not fits:
            fits = True
            # Find neighbours that overlap with address
            for lr2 in lr.neighbours:
                if lr2.address == HillClimbAllocator.NOT_ALLOCATED or lr2.end_address <= address:
                    continue
                if lr2.overlaps(address, lr.size):
                    # Overlap found increase address
                    fits = False
                    address = numeric_util.round_up(lr2.end_address, lr.min_alignment)
                    predecessor = lr2.id
        lr.address = address
        lr.end_address = address + lr.size
        lr.predecessor = predecessor

    def allocate_indices(self, indices: List[int]):
        """
        Allocates the live ranges in the order indicated by the indices;
        allocates each live range at the lowest possible address.
        """
        for lr in self.lrs:
            lr.address = HillClimbAllocator.NOT_ALLOCATED
        size = 0
        for turn, index in enumerate(indices):
            lr = self.lrs[index]
            self.allocate_lr(lr)
            lr.turn = turn
            size = max(size, lr.end_address)
            if size > self.best_size:
                # This allocation is worse than the best known allocation;
                # no need to continue
                break
        return size

    def add_predecessor_turns(self, turn_set: Set[int], turn_list: List[int], lr: LiveRangeInfo):
        """
        Adds the given live range + predecessors to the turns vector.
        Note: the turn_set is for quick detection of duplicates,
        the turn_list is to get reproduceable results
        """
        if lr.turn not in turn_set:
            turn_set.add(lr.turn)
            turn_list.append(lr.turn)
        id = lr.id
        while self.lrs[id].predecessor != HillClimbAllocator.NO_PREDECESSOR:
            id = self.lrs[id].predecessor
            turn = self.lrs[id].turn
            if turn not in turn_set:
                turn_set.add(turn)
                turn_list.append(turn)

    def attempt_bottleneck_fix(self, indices: List[int], iterations_stuck):
        """
        Finds the "bottleneck", the live range with highest end address, and reorders the indices
        such that a next allocation might lower the memory usage.

                                  ---------
                                  |       |
                                  |   D   |
                                  |       |
         ----------------------------------
         |           B                 |
         -------------------------------
         | |
         |A|                      ---
         | |                      |C|
         | |                      | |
         ---------------------------------------

        In the above example, the allocation order was [A, B, C, D] and D is the resulting bottle-neck.
        The live ranges that affected the allocation of D are the direct neighbours of D (i.e. B and C),
        and all direct and indirect predecessors of D and its neighbours
        (i.e. A, which is the predecessor of B, and indirect predecessor of D).

        By permuting the order in which the affecting live ranges are allocated, the bottleneck might
        be lowered. In the above example, almost any permutation would lower the bottleneck.
        """
        # Find the bottleneck
        max_lr = self.lrs[0]
        for lr in self.lrs[1:]:
            if lr.end_address > max_lr.end_address:
                max_lr = lr

        # Find all live ranges that affected the placement of the bottleneck live range.
        # This consists of two types of live ranges:
        # - direct neighbours of the bottleneck live range
        # - direct and indirect predecessors of these neighbours + bottleneck
        # The turns at which these live ranges were allocated are put in the turns set.
        turn_set: Set[int] = set()
        turn_list: List[int] = list()
        self.add_predecessor_turns(turn_set, turn_list, max_lr)
        for lr2 in max_lr.neighbours:
            self.add_predecessor_turns(turn_set, turn_list, lr2)

        # Non-direct neighbours that interfere with the allocation of the bottleneck are the
        # immediate cause for gaps in the allocation, and are selected with higher probability.
        non_nb_turn_list = []
        for turn in turn_list:
            lr = self.lrs[indices[turn]]
            if not max_lr.is_neighbour(lr):
                non_nb_turn_list.append(turn)
        assert turn_list
        # Pick from non-neighbour list with 30% probability
        # (magic number based on tuning)
        if random.randint(0, 100) < 30 and non_nb_turn_list:
            # Pick a live range from the "non-neighbour list"
            ix1 = non_nb_turn_list[random.randint(0, len(non_nb_turn_list) - 1)]
        else:
            # Pick any affecting live range.
            ix1 = turn_list[random.randint(0, len(turn_list) - 1)]

        ix2 = turn_list[random.randint(0, len(turn_list) - 2)]
        if ix1 == ix2:
            ix2 = turn_list[-1]
        # Swap indices
        indices[ix1], indices[ix2] = indices[ix2], indices[ix1]
        if iterations_stuck > HillClimbAllocator.MAX_ITERATIONS_STUCK:
            # The best allocation has not improved for a while, maybe improvement is not possible
            # by single-swapping indices; add more neighbour live ranges and swap up to two more indices.
            # Adding more neighbours can sometimes resolve the situation where the current bottleneck
            # is resolved, but always results in a higher bottleneck at a nearby live range.
            # Magic number is based on tuning
            for turn in non_nb_turn_list:
                for lr in self.lrs[indices[turn]].neighbours:
                    if lr.turn not in turn_set:
                        turn_set.add(lr.turn)
                        turn_list.append(lr.turn)
            ix1 = turn_list[random.randint(0, len(turn_list) - 1)]
            ix2 = turn_list[random.randint(0, len(turn_list) - 1)]
            indices[ix1], indices[ix2] = indices[ix2], indices[ix1]

    def search(self, indices: List[int]):
        """
        Search for a solution, using the given indices as initial solution.
        """
        best_indices = indices[:]
        last_improvement_iteration = 0
        i = 0
        while (self.best_size > self.memory_limit and i < self.max_iterations) or (
            i - last_improvement_iteration < self.MIN_ITERATIONS_IMPROVE
        ):
            # Reorder the indices
            self.attempt_bottleneck_fix(indices, i - last_improvement_iteration)
            # Allocate the reordered indices and check if it gave an improvement
            new_size = self.allocate_indices(indices)
            if new_size <= self.best_size:
                # The new allocation produced a new best result remember it
                if new_size < self.best_size:
                    last_improvement_iteration = i
                self.best_size = new_size
                best_indices = indices[:]
                self.allocated_addresses = [lr.address for lr in self.lrs]
                if self.best_size <= self.min_required_size:
                    # Target reached; stop
                    return
            else:
                # The new allocation produced worse result undo the change
                indices = best_indices[:]
            i += 1

    def allocate(self) -> List[int]:
        """
        Runs the allocation algorithm. Finishes when an optimal solution has been
        found or when maximum iterations have been run.
        The allocated addresses are placed in the output vector, in the same
        order as the input vector.

        Implementation note: the algorithm produces reproduceable results by using
        a well-defined random number generator with well-defined default seed,
        and using a fixed number of iterations.
        """
        random.seed(1)
        # Sort indices on priority. Note: self.lrs must be left unchanged
        indices = [lr.id for lr in sorted(self.lrs)]
        # Allocate the initial solution
        self.best_size = self.allocate_indices(indices)
        self.allocated_addresses = [lr.address for lr in self.lrs]
        if self.best_size > self.min_required_size:
            # Try to improve the heuristic allocation
            self.search(indices)
        # else the heuristic allocation returned an optimal solution; no search needed
        return self.allocated_addresses


def allocate_live_ranges(lrs: List[LiveRange], max_iterations: int, memory_limit: int) -> List[int]:
    """
    Allocates live ranges using a search based allocator.
    Returns the list of allocated addresses (one for each live range)
    """
    if not lrs:
        return []
    allocator = HillClimbAllocator(lrs, max_iterations, memory_limit)
    return allocator.allocate()
