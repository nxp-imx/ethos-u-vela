/*
 * Copyright (c) 2020 Arm Limited. All rights reserved.
 *
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the License); you may
 * not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an AS IS BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 * Description:
 * Declaration of the search-based allocator.
 */

#ifndef __SEARCH_ALLOCATOR_H
#define __SEARCH_ALLOCATOR_H

#include <algorithm>
#include <cstdint>
#include <random>
#include <set>
#include <vector>

/**
 * Live range
 */
struct LiveRange {
    /** Start time (input to the allocator algorithm) */
    uint32_t start_time;
    /** End time, inclusive (input to the allocator algorithm) */
    uint32_t end_time;
    /** Size in bytes (input to the allocator algorithm) */
    uint32_t size;
    /** Index of this live range */
    int id;
    /** Allocated address (the main output from the allocator algorithm) */
    uint32_t address;
    /** End address, exclusive */
    uint32_t end_address;
    /** id of predecessor live range (predecessor's end address == this lr's address) */
    int predecessor;
    /** Turn at which the live range was allocated */
    size_t turn;

    bool overlaps(uint32_t addr2, uint32_t size2) const {
        return address < addr2 + size2 && addr2 < end_address;
    }
    bool is_neighbour(const LiveRange &lr) const {
        return start_time <= lr.end_time && lr.start_time <= end_time;
    }
};

/**
 * Implements tensor allocator using state space exploration.
 *
 * The basic algorithm is:
 *
 * Use a heuristic allocator to find an initial allocation
 * while allocation is not optimal and iterations < MAX_ITERATIONS {
 *     find the "bottleneck": the live range with highest end address
 *     find all live ranges that affected the allocation of the bottleneck
 *     swap the order of any two affecting live ranges
 *     reallocate tensors using the reordered live ranges
 *     if the new allocation is better: keep it, else set allocation to previous allocation
 * }
 */
class SearchAllocator {
private:
    static constexpr int MAX_ITERATIONS = 500;
    static constexpr uint32_t NOT_ALLOCATED = UINT32_MAX;
    /** Used for live ranges allocated at address 0 */
    static constexpr int NO_PREDECESSOR = -1;
    /** Contains the live ranges */
    std::vector<LiveRange> lrs;
    /** Contains active live ranges at each timestamp */
    std::vector<std::vector<LiveRange*>> lrs_at_time;
    /**
     * Contains neighbours of each live range (indexed by lr.id), i.e.
     * live ranges with overlapping start/end time.
     */
    std::vector<std::vector<LiveRange*>> neighbours;
    /**
     * At each timestamp: accumulated size of active live ranges
     */
    std::vector<uint32_t> size_at_time;
    /**
     * For each live range: max value of size_at_time (only used in the heuristic allocation)
     */
    std::vector<uint32_t> lr_urgency;
    /**
     * The minimum possible size, assuming all live ranges can be perfectly allocated
     */
    uint32_t min_required_size;
    /** The algorithm stops once the target size has been achieved */
    uint32_t target_size;
    /** The highest end address of the best found allocation */
    uint32_t best_size;
    /** Number of performed iterations */
    size_t nr_iterations = 0;
    /** Random number generator; use default seed (which is well-defined) */
    std::mt19937 rng;
public:
    SearchAllocator(const std::vector<LiveRange> &live_ranges, uint32_t size_limit);
    /**
     * Runs the allocation algorithm. Finishes when the target size has been
     * reached or when maximum iterations have been run.
     * The allocated addresses are placed in the output vector, in the same
     * order as the input vector.
     *
     * Implementation note: the algorithm produces reproduceable results by using
     * a well-defined random number generator with well-defined default seed,
     * and using a fixed number of iterations.
     */
    uint32_t allocate(std::vector<uint32_t> &output);
    uint32_t get_min_required_size() const {
        return min_required_size;
    }
    size_t get_nr_iterations() const {
        return nr_iterations;
    }
private:
    /**
     * Allocates the given live range at the smallest possible address
     */
    void allocate_lr(LiveRange &lr) const;
    /**
     * Allocates the live ranges in the order indicated by the indices;
     * allocates each live range at the lowest possible address.
     */
    uint32_t allocate_indices(const std::vector<size_t> &indices);
    /** Sorts live ranges based on heuristics, used for the initial allocation */
    void sort_indices_on_prio(std::vector<size_t> &indices) const;
    /** Adds the given live range + predecessors to the turns vector */
    void add_predecessor_turns(std::set<size_t> &turns, const LiveRange &lr) const;
    /**
     * Finds the "bottleneck", the live range with highest end address, and reorders the indices
     * such that a next allocation might lower the memory usage.
     *
     *                          ---------
     *                          |       |
     *                          |   D   |
     *                          |       |
     * ----------------------------------
     * |           B                 |
     * -------------------------------
     * | |
     * |A|                      ---
     * | |                      |C|
     * | |                      | |
     * ---------------------------------------
     *
     * In the above example, the allocation order was [A, B, C, D] and D is the resulting bottle-neck.
     * The live ranges that affected the allocation of D are the direct neighbours of D (i.e. B and C),
     * and all direct and indirect predecessors of D and its neighbours
     * (i.e. A, which is the predecessor of B, and indirect predecessor of D).
     *
     * By permuting the order in which the affecting live ranges are allocated, the bottleneck might
     * be lowered. In the above example, almost any permutation would lower the bottleneck.
     *
     * Note that there is room to improve the efficiency of the algorithm.
     * One way could be to first allocate all direct neighbours of the bottleneck
     * (i.e. B, C, D) and then the other affecting live ranges (i.e. A). The algorithm currently does
     * not actively try this, as it may lead to allocation loops (A could become the new bottle-neck);
     * it just uses a higher probability of selecting A.
     */
    void attempt_bottleneck_fix(std::vector<size_t> &indices);
    /** Search for a solution, using the given indices as initial solution. */
    void search(std::vector<size_t> &indices, uint32_t initial_size, int iterations);
};

/** Wrapper function to perform live range allocation */
uint32_t allocate(const std::vector<uint32_t> &input, int available_size, std::vector<uint32_t> &output);

#endif // __SEARCH_ALLOCATOR_H
