# SPDX-FileCopyrightText: Copyright 2020-2023 Arm Limited and/or its affiliates <open-source-office@arm.com>
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
# Wrapping function to do tensor address allocation. That is, assigning addresses to tensors based on what has been
# worked out from the allowable overlaps that are calculated by the live range analysis.
import math
from typing import List

import numpy as np

from . import hillclimb_allocation
from . import live_range
from . import numeric_util
from .errors import AllocationError
from .greedy_allocation import allocate_live_ranges as greedy_allocate_live_ranges
from .live_range import LiveRange
from .live_range import LiveRangeGraph
from .nn_graph import TensorAllocator
from .tensor import MemArea
from .tensor import MemType
from .tensor import Tensor
from .tensor import TensorPurpose


def linear_allocate_live_ranges(live_ranges, alloc_granularity=Tensor.AllocationQuantum):
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
                    assert allocated_tens.scale_compression_config == tens.scale_compression_config
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

    verify_alignment(live_ranges, alloc_granularity)
    return total_sz


def hillclimb_allocate_live_ranges(
    live_ranges: LiveRangeGraph, alloc_granularity: int, max_iterations: int, mem_limit: int
) -> int:
    # Allocates using the hill climb allocator
    addresses = hillclimb_allocation.allocate_live_ranges(live_ranges.lrs, max_iterations, mem_limit)
    # The result is a list containing the allocated addresses
    total_sz = 0
    for lr, address in zip(live_ranges.lrs, addresses):
        total_sz = max(total_sz, address + lr.size)
        lr.set_address(address)
    verify_allocation(live_ranges, alloc_granularity)
    return total_sz


def verify_alignment(live_ranges: LiveRangeGraph, alignment: int):
    for lr in live_ranges.lrs:
        for tens in lr.tensors:
            if not all(op and op.run_on_npu for op in tens.ops + tens.consumer_list):
                # This is a CPU tensor, verify alignment
                if tens.address % alignment != 0:
                    raise AllocationError(f"Tensor '{tens.name}' not aligned to {alignment} bytes")


def verify_allocation(live_ranges: LiveRangeGraph, alignment: int):
    verify_alignment(live_ranges, alignment)
    nr_time_slots = 1 + max(lr.end_time for lr in live_ranges.lrs)
    # Contains active live ranges at each timestamp
    lrs_at_time: List[List[LiveRange]] = [[] for i in range(nr_time_slots)]
    for lr in live_ranges.lrs:
        for t in range(lr.start_time, lr.end_time + 1):
            lrs_at_time[t].append(lr)
    for t in range(nr_time_slots):
        lrs_new_items = [lr for lr in lrs_at_time[t] if t == 0 or lr not in lrs_at_time[t - 1]]
        for m in lrs_new_items:
            for n in lrs_at_time[t]:
                overlap, tens_n, tens_m = n.overlaps_address(m)
                if overlap and not (tens_n.equivalent(tens_m) and tens_n.address == tens_m.address):
                    raise AllocationError(
                        f"Overlapping buffers: {n.name}: {tens_n.address} -> {tens_n.address + n.size}"
                        f" and {m.name}: {tens_m.address} -> {tens_m.address + m.size}"
                    )


def mark_sram_used_for_cascaded_passes(sg, lrs):
    if len(sg.cascaded_passes) < 1:
        return
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


def print_allocation(lrs, mem_area, mem_type_set, tensor_allocator, sg, actual_mem_usage_for_alloc):
    print("\n" + "#" * 80)
    sg_placement = (
        sg.placement.name
        if mem_type_set.intersection(
            (
                MemType.Permanent_NPU,
                MemType.Permanent_CPU,
            )
        )
        else "Cpu and Npu"
    )
    print(
        f"Tensor Allocation for mem_area {mem_area.name}, of mem_type_set ("
        f'{", ".join(f"{mem_type.name}" for mem_type in mem_type_set)}'
        f"), using allocator {tensor_allocator}, in {sg_placement} subgraph:"
    )

    memory_hist = memory_usage_histogram(lrs.lrs)
    min_mem_usage_for_alloc = max(memory_hist)
    print(
        f"{'Start Time':>10s} - {'End Time':>10s}: {'Start Addr':>10s} - {'End Addr':>10s}: {'Tensor Size':>11s}:"
        f" {'Memory Usage':>12s}: {'Purpose':12s}: Name"
    )
    for start_time, end_time, size, start_addr, end_addr, purpose, name in sorted(
        (
            lr.start_time,
            lr.end_time,
            lr.size,
            tens.address,
            tens.address + lr.size,
            tens.purpose,
            tens.name,
        )
        for tens, lr in lrs.ranges.items()
    ):
        print(
            f"{start_time:10d} - {end_time:10d}: {start_addr:#10x} - {end_addr:#10x}: {size:11d}:"
            f" {memory_hist[start_time]:12d}: {purpose.display_name():12s}: {name:s}"
        )

    alloc_overhead_fraction = (actual_mem_usage_for_alloc - min_mem_usage_for_alloc) / min_mem_usage_for_alloc
    print(
        f"Allocation Peak Tensor Size:  {min_mem_usage_for_alloc:9d} ({min_mem_usage_for_alloc:#10x})"
        f" Bytes {min_mem_usage_for_alloc/1024.0:8.2f} KiB"
    )
    print(
        f"Allocation Peak Memory Usage: {actual_mem_usage_for_alloc:9d} ({actual_mem_usage_for_alloc:#10x})"
        f" Bytes {actual_mem_usage_for_alloc/1024.0:8.2f} KiB"
    )
    print(
        f"Allocation Overhead:          {actual_mem_usage_for_alloc-min_mem_usage_for_alloc:9d}"
        f" Bytes ({100*alloc_overhead_fraction:.2f} %)"
    )


def memory_usage_histogram(lrs: List[LiveRange]):
    histogram = [0] * (1 + max(lr.end_time for lr in lrs))
    for lr in lrs:
        for t in range(lr.start_time, lr.end_time + 1):
            histogram[t] += lr.size

    return histogram


def allocate(
    sg,
    arch,
    mem_area,
    mem_type_set,
    tensor_allocator=TensorAllocator.Greedy,
    lr_graph=None,
    cpu_tensor_alignment=Tensor.AllocationQuantum,
    hillclimb_max_iterations=None,
    verbose_progress=False,
):
    # Allocates addresses to tensors, returns False if tensors could not be fit within max_size
    lrs = live_range.extract_live_ranges_from_cascaded_passes(
        sg,
        mem_area,
        mem_type_set,
        lr_graph=lr_graph,
        cpu_tensor_alignment=cpu_tensor_alignment,
        verbose_progress=verbose_progress,
    )
    total_sz = 0
    if lrs.ranges:
        tens_alloc = tensor_allocator
        if tens_alloc == TensorAllocator.Greedy:
            total_sz = greedy_allocate_live_ranges(lrs, cpu_tensor_alignment)
            verify_allocation(lrs, cpu_tensor_alignment)
        elif tens_alloc == TensorAllocator.LinearAlloc:
            total_sz = linear_allocate_live_ranges(lrs, cpu_tensor_alignment)
        elif tens_alloc == TensorAllocator.HillClimb:
            mem_type = MemType.Scratch_fast if MemType.Scratch_fast in mem_type_set else list(mem_type_set)[0]
            mem_size = arch.mem_type_size(mem_type)
            total_sz = hillclimb_allocate_live_ranges(lrs, cpu_tensor_alignment, hillclimb_max_iterations, mem_size)
        else:
            assert 0
    return lrs, total_sz


def allocate_tensors(
    nng,
    sg,
    arch,
    mem_area,
    mem_type_set,
    tensor_allocator=TensorAllocator.Greedy,
    verbose_allocation=False,
    verbose_progress=False,
    lr_graph=None,
    cpu_tensor_alignment=Tensor.AllocationQuantum,
    hillclimb_max_iterations=None,
    max_size=None,
    dry_test=False,
):
    # Allocates addresses to tensors, returns False if tensors could not be fit within max_size
    lrs, total_sz = allocate(
        sg,
        arch,
        mem_area,
        mem_type_set,
        tensor_allocator=tensor_allocator,
        lr_graph=lr_graph,
        cpu_tensor_alignment=cpu_tensor_alignment,
        hillclimb_max_iterations=hillclimb_max_iterations,
        verbose_progress=verbose_progress,
    )

    if lrs.ranges:
        alloc_ok = max_size is None or total_sz <= max_size
        if dry_test or not alloc_ok:
            # Dry test or allocation failed; undo allocation
            for lr in lrs.ranges.values():
                lr.set_address(None)
            return alloc_ok

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

        if verbose_allocation:
            print_allocation(lrs, mem_area, mem_type_set, tensor_allocator, sg, total_sz)

        if mem_area == MemArea.Sram:
            # Mark Sram usage for all subgraphs
            for sg_ in nng.subgraphs:
                mark_sram_used_for_cascaded_passes(sg_, lrs)

    if sg == nng.get_root_subgraph():
        nng.memory_used = sg.memory_used
    return True
