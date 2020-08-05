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
# Wrapping function to do tensor address allocation. That is, assigning addresses to tensors based on what has been
# worked out from the allowable overlaps that are calculated by the live range analysis.
import math

import numpy as np

from . import live_range
from . import numeric_util
from .greedy_allocation import allocate_live_ranges as greedy_allocate_live_ranges
from .nn_graph import TensorAllocator
from .tensor import MemArea
from .tensor import MemType
from .tensor import TensorPurpose


def linear_allocate_live_ranges(live_ranges, alloc_granularity=16):
    # Allocates using increasing addresses. Duplicate constant tensors will be allocated to the same address
    total_sz = 0
    allocated_tensors = []

    # just assign increasing addresses, except for duplicates
    for tens, lr in live_ranges.ranges.items():
        if tens in allocated_tensors:
            continue

        address = total_sz
        if tens.weight_compression_config is not None:
            for allocated_tens in allocated_tensors:
                if allocated_tens.weight_compression_config == tens.weight_compression_config:
                    address = allocated_tens.address
                    break
        if tens.purpose == TensorPurpose.LUT:
            for allocated_tens in allocated_tensors:
                if allocated_tens.equivalent(tens):
                    address = allocated_tens.address
                    break
        lr.set_address(address)
        allocated_tensors += lr.tensors
        if address == total_sz:
            total_sz += numeric_util.round_up(int(math.ceil(lr.size)), alloc_granularity)

    return total_sz


def mark_sram_used_for_cascaded_passes(sg, lrs):
    end_pos = max(ps.time for ps in sg.cascaded_passes) + 2
    mem_usage = np.zeros(end_pos, dtype=np.int64)

    for tens, rng in lrs.ranges.items():
        storage_size = tens.storage_size()
        mem_usage[rng.start_time : rng.end_time] += storage_size

    for cps in sg.cascaded_passes:
        sram_used = max(mem_usage[cps.time], mem_usage[cps.time + 1])
        cps.sram_used = sram_used
        for ps in cps.passes:
            ps.sram_used = sram_used


def print_allocation(lrs, mem_area, mem_type_set, sg, verbose_allocation, show_minimum_possible_allocation):
    if verbose_allocation:
        if mem_type_set == set((MemType.Permanent_NPU,)) or mem_type_set == set((MemType.Permanent_CPU,)):
            print("allocation for", mem_area, "- constant tensors in", sg.placement.name, "subgraph(s)")
        else:
            print("allocation for", mem_area, "- non-constant tensors in Cpu and Npu subgraphs")

        for start_time, start, end, name, end_time in sorted(
            (
                lr.start_time,
                tens.address,
                tens.address + int(math.ceil(tens.storage_size())),
                tens.name + " " + str(tens.purpose),
                lr.end_time,
            )
            for tens, lr in lrs.ranges.items()
        ):
            name = name.replace("\x00", "")
            print("%9d: %#12x - %#12x: %3d - %3d %s" % ((end - start), start, end, start_time, end_time, name))
        print()

    if show_minimum_possible_allocation and mem_area == MemArea.Sram:
        min_possible_allocation = max(cps.sram_used for cps in sg.cascaded_passes)
        print(
            "Min possible allocation %d bytes / %.1f KB / %.1f MB"
            % (min_possible_allocation, min_possible_allocation / 1024, min_possible_allocation / 1024 / 1024)
        )


def allocate_tensors(
    nng,
    sg,
    arch,
    mem_area,
    mem_type_set,
    use_ifm_ofm_overlap=True,
    tensor_allocator=TensorAllocator.Greedy,
    verbose_allocation=False,
    show_minimum_possible_allocation=False,
    lr_graph=None,
):
    ignore_subgraph_input_output_tensors = False
    lrs = live_range.extract_live_ranges_from_cascaded_passes(
        sg,
        mem_area,
        mem_type_set,
        mark_output_tensors_overlapping_with_input_tensors=False,
        use_ifm_ofm_overlap=use_ifm_ofm_overlap,
        ignore_subgraph_input_output_tensors=ignore_subgraph_input_output_tensors,
        lr_graph=lr_graph,
    )

    if lrs.ranges:
        tens_alloc = tensor_allocator
        if tens_alloc == TensorAllocator.Greedy:
            total_sz = greedy_allocate_live_ranges(sg, arch, lrs, mem_area, verbose_allocation)
        elif tens_alloc == TensorAllocator.LinearAlloc:
            total_sz = linear_allocate_live_ranges(lrs, 16)
        else:
            assert 0

        if sg.memory_used.get(mem_area, 0) == 0:
            sg.memory_used[mem_area] = total_sz
        else:
            sg.memory_used[mem_area] += total_sz

        # Keep track of how much should be used for scratch or permanent storage for NPU
        for mem_type in mem_type_set:
            if sg.memory_used_per_type.get(mem_type, 0) == 0:
                sg.memory_used_per_type[mem_type] = total_sz
            else:
                sg.memory_used_per_type[mem_type] += total_sz

        nng.total_size[mem_area] = nng.total_size.get(mem_area, 0) + sum(tens.storage_size() for tens in lrs.ranges)
        nng.total_elements[mem_area] = nng.total_elements.get(mem_area, 0) + sum(tens.elements() for tens in lrs.ranges)

        print_allocation(lrs, mem_area, mem_type_set, sg, verbose_allocation, show_minimum_possible_allocation)

        if mem_area == MemArea.Sram:
            # Mark Sram usage for all subgraphs
            for sg_ in nng.subgraphs:
                mark_sram_used_for_cascaded_passes(sg_, lrs)

    if sg == nng.get_root_subgraph():
        nng.memory_used = sg.memory_used
        for mem_area in nng.total_elements.keys():
            try:
                nng.bits_per_element[mem_area] = nng.total_size[mem_area] * 8 / nng.total_elements[mem_area]
            except ZeroDivisionError:
                nng.bits_per_element[mem_area] = 0.0
