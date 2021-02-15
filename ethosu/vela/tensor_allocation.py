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
# Wrapping function to do tensor address allocation. That is, assigning addresses to tensors based on what has been
# worked out from the allowable overlaps that are calculated by the live range analysis.
import math

import numpy as np

from . import hillclimb_allocation
from . import live_range
from . import numeric_util
from .errors import AllocationError
from .greedy_allocation import allocate_live_ranges as greedy_allocate_live_ranges
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


def hillclimb_allocate_live_ranges(live_ranges: LiveRangeGraph, alloc_granularity: int) -> int:
    # Allocates using the hill climb allocator
    lr_set = {(lr.start_time, lr.end_time, lr) for lr in live_ranges.ranges.values()}
    lr_list = [lr for _, _, lr in lr_set]
    lr_list.sort()

    addresses = hillclimb_allocation.allocate_live_ranges(lr_list)
    # The result is a list containing the allocated addresses
    total_sz = 0
    for lr, address in zip(lr_list, addresses):
        total_sz = max(total_sz, address + lr.size)
        lr.set_address(address)
    verify_allocation(live_ranges, alloc_granularity)
    return total_sz


def verify_alignment(live_ranges: LiveRangeGraph, alignment: int):
    for lr in live_ranges.ranges.values():
        for tens in lr.tensors:
            if not all(op and op.run_on_npu for op in tens.ops + tens.consumer_list):
                # This is a CPU tensor, verify alignment
                if tens.address % alignment != 0:
                    raise AllocationError(f"Tensor '{tens.name}' not aligned to {alignment} bytes")


def verify_allocation(live_ranges: LiveRangeGraph, alignment: int):
    lrs = list(live_ranges.ranges.values())
    for n in lrs:
        verify_alignment(live_ranges, alignment)

        for m in lrs:
            if n != m and n.overlaps_ranges(m):
                overlap, tens_n, tens_m = n.overlaps_address(m)
                if overlap and not (tens_n.equivalent(tens_m) and tens_n.address == tens_m.address):
                    raise AllocationError(
                        f"Overlapping buffers: {n.name}: {tens_n.address} -> {tens_n.address + n.size}"
                        f" and {m.name}: {tens_m.address} -> {tens_m.address + m.size}"
                    )


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


def print_allocation(lrs, mem_area, mem_type_set, sg, verbose_allocation):
    if verbose_allocation:
        if mem_type_set == set((MemType.Permanent_NPU,)) or mem_type_set == set((MemType.Permanent_CPU,)):
            print("allocation for", mem_area, "- constant tensors in", sg.placement.name, "subgraph(s)")
        else:
            print("allocation for", mem_area, "- non-constant tensors in Cpu and Npu subgraphs")
        mem_usage = 0
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
            mem_usage = max(mem_usage, end)
        print("Memory usage: {} ({:#x}) bytes / {:.1f} KB".format(mem_usage, mem_usage, mem_usage / 1024))
        print()


def allocate_tensors(
    nng,
    sg,
    arch,
    mem_area,
    mem_type_set,
    tensor_allocator=TensorAllocator.Greedy,
    verbose_allocation=False,
    lr_graph=None,
    cpu_tensor_alignment=Tensor.AllocationQuantum,
    max_size=None,
    dry_test=False,
):
    # Allocates addresses to tensors, returns False if tensors could not be fit within max_size
    ignore_subgraph_input_output_tensors = False
    lrs = live_range.extract_live_ranges_from_cascaded_passes(
        sg,
        mem_area,
        mem_type_set,
        ignore_subgraph_input_output_tensors=ignore_subgraph_input_output_tensors,
        lr_graph=lr_graph,
        cpu_tensor_alignment=cpu_tensor_alignment,
    )

    if lrs.ranges:
        tens_alloc = tensor_allocator
        if tens_alloc == TensorAllocator.Greedy:
            total_sz = greedy_allocate_live_ranges(sg, arch, lrs, mem_area, cpu_tensor_alignment, verbose_allocation)
            verify_allocation(lrs, cpu_tensor_alignment)
        elif tens_alloc == TensorAllocator.LinearAlloc:
            total_sz = linear_allocate_live_ranges(lrs, cpu_tensor_alignment)
        elif tens_alloc == TensorAllocator.HillClimb:
            total_sz = hillclimb_allocate_live_ranges(lrs, cpu_tensor_alignment)
        else:
            assert 0
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

        print_allocation(lrs, mem_area, mem_type_set, sg, verbose_allocation)

        if mem_area == MemArea.Sram:
            # Mark Sram usage for all subgraphs
            for sg_ in nng.subgraphs:
                mark_sram_used_for_cascaded_passes(sg_, lrs)

    if sg == nng.get_root_subgraph():
        nng.memory_used = sg.memory_used
        try:
            nng.weights_compression_ratio = nng.total_compressed_weights / nng.total_original_weights
        except ZeroDivisionError:
            nng.weights_compression_ratio = 0.0

    return True
