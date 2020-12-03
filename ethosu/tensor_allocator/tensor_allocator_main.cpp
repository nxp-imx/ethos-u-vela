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
 */

#include <cstdint>
#include <iostream>
#include <vector>

#include "search_allocator.h"

using namespace std;

/**
 * Reads live ranges from the input, and then performs allocation.
 * The input has format:

<number of live ranges>
<start_time> <end_time> <size>
...

 * e.g.:
4
0 20 4096
2 8 16000
4 10 800
6 20 1024
 */
int main() {
    int lr_count;
    cin >> lr_count;
    cin.ignore();
    vector<uint32_t> input;
    vector<uint32_t> output;
    for (int i = 0; i < lr_count; ++i) {
        LiveRange lr;
        cin >> lr.start_time >> lr.end_time >> lr.size;
        lr.id = i;
        cin.ignore();
        input.push_back(lr.start_time);
        input.push_back(lr.end_time);
        input.push_back(lr.size);
    }
    vector<LiveRange> lrs;
    for (size_t i = 0; i < input.size(); i += 3) {
        LiveRange lr;
        lr.start_time = input[i];
        lr.end_time = input[i+1];
        lr.size = input[i+2];
        lrs.push_back(lr);
    }
    SearchAllocator allocator(lrs, 0);
    uint32_t result = allocator.allocate(output);
    printf("Output:\n");
    for (int i = 0; i < lr_count; ++i) {
        printf("%4d: %6d %4d-%4d size %6d\n", i, output[i], input[3*i], input[3*i+1], input[3*i+2]);
    }
    uint32_t min_size = allocator.get_min_required_size();
    double overhead = 100.0 * (result - min_size)/(double)min_size;
    printf("Total size: %d (%1.1f K), minimum required size: %d, overhead: %1.2f%%\n",
        result, result/1024.0, min_size, overhead);
    printf("Search used %ld iterations\n", (long)allocator.get_nr_iterations());
    return 0;
}
