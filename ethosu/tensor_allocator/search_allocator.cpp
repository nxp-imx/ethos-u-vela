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
 * Implementation of the search-based allocator.
 */

#include <algorithm>
#include <cstdint>
#include <set>
#include <vector>

#include "search_allocator.h"

SearchAllocator::SearchAllocator(const std::vector<LiveRange> &live_ranges, uint32_t size_limit) {
    lrs = live_ranges;
    uint32_t max_end_time = 0;
    for (size_t i = 0; i < lrs.size(); ++i) {
        auto &lr = lrs[i];
        lr.id = static_cast<int>(i);
        max_end_time = std::max(max_end_time, lr.end_time);
    }
    lrs_at_time.resize(max_end_time + 1);
    size_at_time.resize(max_end_time + 1);
    neighbours.resize(lrs.size());
    // Calculate which live ranges are active at every timestamp
    for (size_t t = 0; t <= max_end_time; ++t) {
        lrs_at_time[t].clear();
    }
    for (auto &lr : lrs) {
        for (auto t = lr.start_time; t <= lr.end_time; ++t) {
            lrs_at_time[t].push_back(&lr);
        }
    }
    min_required_size = 0;
    for (size_t t = 0; t <= max_end_time; ++t) {
        // Calculate minimum needed size at each timestamp
        uint32_t size_at_t = 0;
        for (auto &lr : lrs_at_time[t]) {
            size_at_t += lr->size;
        }
        size_at_time[t] = size_at_t;
        min_required_size = std::max(size_at_t, min_required_size);
        // Calculate all neighbours
        for (size_t i = 0; i < lrs_at_time[t].size(); ++i) {
            auto lr1 = lrs_at_time[t][i];
            auto &nb1 = neighbours[lr1->id];
            for (size_t j = i + 1; j < lrs_at_time[t].size(); ++j) {
                auto lr2 = lrs_at_time[t][j];
                if (find(nb1.begin(), nb1.end(), lr2) == nb1.end()) {
                    nb1.push_back(lr2);
                    neighbours[lr2->id].push_back(lr1);
                }
            }
        }
    }
    target_size = std::max(min_required_size, size_limit);
    // Calculate the urgency of each live range
    lr_urgency.resize(lrs.size());
    for (size_t i = 0; i < lrs.size(); ++i) {
        auto &lr = lrs[i];
        uint32_t urgency = 0;
        for (size_t t = lr.start_time; t <= lr.end_time; ++t) {
            urgency = std::max(size_at_time[t], urgency);
        }
        lr_urgency[i] = urgency;
    }
    best_size = UINT32_MAX;
}

uint32_t SearchAllocator::allocate(std::vector<uint32_t> &output) {
    output.clear();
    nr_iterations = 0;
    std::vector<size_t> indices;
    // Initial solution, using a heuristic allocator
    for (size_t i = 0; i < lrs.size(); ++i) {
        indices.push_back(i);
    }
    sort_indices_on_prio(indices);
    // Allocate the initial solution
    best_size = UINT32_MAX;
    best_size = allocate_indices(indices);
    if (best_size <= target_size) {
        // The heuristic allocation returned an optimal solution.
        // No need to search.
    } else {
        // Try to improve the heuristic allocation
        search(indices, best_size, MAX_ITERATIONS);
    }
    output.clear();
    for (auto &lr : lrs) {
        output.push_back(lr.address);
    }
    return best_size;
}

void SearchAllocator::allocate_lr(LiveRange &lr) const {
    uint32_t address = 0;
    int predecessor = NO_PREDECESSOR;
    bool fits = false;
    while (!fits) {
        fits = true;
        // Find neighbours that overlap with address
        for (auto lr2_p : neighbours[lr.id]) {
            if (lr2_p->address == NOT_ALLOCATED || lr2_p->end_address <= address) {
                continue;
            }
            if (lr2_p->overlaps(address, lr.size)) {
                // Overlap found; increase address
                fits = false;
                address = lr2_p->end_address;
                predecessor = lr2_p->id;
            }
        }
    }
    lr.address = address;
    lr.end_address = address + lr.size;
    lr.predecessor = predecessor;
}

uint32_t SearchAllocator::allocate_indices(const std::vector<size_t> &indices) {
    ++nr_iterations;
    std::vector<size_t> count(indices.size());
    for (auto &lr : lrs) {
        lr.address = NOT_ALLOCATED;
    }
    uint32_t size = 0;
    for (size_t turn = 0; size <= best_size && turn < indices.size(); ++turn) {
        auto &lr = lrs[indices[turn]];
        allocate_lr(lr);
        lr.turn = turn;
        size = std::max(size, lr.end_address);
    }
    return size;
}

void SearchAllocator::sort_indices_on_prio(std::vector<size_t> &indices) const {
    std::sort(indices.begin(), indices.end(),
        [this] (size_t const& a, size_t const& b) {
            if (lr_urgency[a] != lr_urgency[b]) {
                return lr_urgency[a] > lr_urgency[b];
            }
            auto &lr1 = lrs[a];
            auto &lr2 = lrs[b];
            auto duration1 = lr1.end_time - lr1.start_time;
            auto duration2 = lr2.end_time - lr2.start_time;
            if (duration1 != duration2) {
                return duration1 > duration2;
            }
            if (lr1.start_time != lr2.start_time) {
                return lr1.start_time < lr2.start_time;
            }
            if (lr1.size != lr2.size) {
                return lr1.size > lr2.size;
            }
            return lr1.id < lr2.id;
        });
}

void SearchAllocator::add_predecessor_turns(std::set<size_t> &turns, const LiveRange &lr) const {
    turns.insert(lr.turn);
    int id = lr.id;
    while (lrs[id].predecessor != NO_PREDECESSOR) {
        id = lrs[id].predecessor;
        turns.insert(lrs[id].turn);
    }
}

void SearchAllocator::attempt_bottleneck_fix(std::vector<size_t> &indices) {
    // Find the bottleneck
    LiveRange *max_lr = &lrs[0];
    for (auto &lr : lrs) {
        if (lr.end_address > max_lr->end_address) {
            max_lr = &lr;
        }
    }
    // Find all live ranges that affected the placement of the bottleneck live range.
    // This consists of two types of live ranges:
    // - direct neighbours of the bottleneck live range
    // - direct and indirect predecessors of these neighbours + bottleneck
    // The turns at which these live ranges were allocated are put in the turns vector.
    std::set<size_t> turns;
    add_predecessor_turns(turns, *max_lr);
    for (auto lr_p : neighbours[max_lr->id]) {
        add_predecessor_turns(turns, *lr_p);
    }
    // Non-direct neighbours that interfere with the allocation of the bottleneck are the
    // immediate cause for gaps in the allocation, and are selected with higher probability.
    std::vector<size_t> turn_list;
    std::vector<size_t> non_nb_turn_list;
    for (auto turn : turns) {
        turn_list.push_back(turn);
        auto &lr = lrs[indices[turn]];
        if (!max_lr->is_neighbour(lr)) {
            non_nb_turn_list.push_back(turn);
        }
    }
    size_t ix1;
    if (rng() % 100 < 30 && !non_nb_turn_list.empty()) {
        // Pick a live range from the "non-neighbour list"
        ix1 = non_nb_turn_list[rng() % non_nb_turn_list.size()];
    } else {
        // Pick any affecting live range.
        ix1 = turn_list[rng() % turn_list.size()];
    }
    // Note: turn_list has always at least 2 elements for bottlenecks
    size_t ix2 = turn_list[rng() % (turn_list.size() - 1)];
    if (ix1 == ix2) {
        ix2 = turn_list[turn_list.size() - 1];
    }
    // Swap indices
    std::swap(indices[ix1], indices[ix2]);
}

void SearchAllocator::search(std::vector<size_t> &indices, uint32_t initial_size, int iterations) {
    std::vector<size_t> best_indices = indices;
    std::vector<LiveRange> best_lrs = lrs;
    for (int i = 0; i < iterations; ++i) {
        // Reorder the indices
        attempt_bottleneck_fix(indices);
        // Allocate the reordered indices and check if it gave an improvement
        auto new_size = allocate_indices(indices);
        if (new_size <= best_size) {
            // The new allocation produced a new best result; remember it
            best_size = new_size;
            best_indices = indices;
            best_lrs = lrs;
            if (best_size <= target_size) {
                // Target reached; stop
                return;
            }
        } else {
            // The new allocation produced worse result; undo the change
            indices = best_indices;
            lrs = best_lrs;
        }
    }
    lrs = best_lrs;
}

uint32_t allocate(const std::vector<uint32_t> &input, int available_size, std::vector<uint32_t> &output) {
    // Convert input to vector of live ranges
    std::vector<LiveRange> lrs;
    for (size_t i = 0; i < input.size(); i += 3) {
        LiveRange lr;
        lr.start_time = input[i];
        lr.end_time = input[i+1];
        lr.size = input[i+2];
        lrs.push_back(lr);
    }
    SearchAllocator allocator(lrs, available_size);
    return allocator.allocate(output);
}
