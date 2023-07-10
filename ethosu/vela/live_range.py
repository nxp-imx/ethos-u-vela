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
# Build a live range graph for tensors in one or more subgraphs. Used for tensor allocation as well as in the scheduler.
# Can work with either a pass packed subgraph or a scheduled subgraph.
from collections import namedtuple
from typing import List

import numpy as np

from .operation import Op
from .tensor import MemArea
from .tensor import MemType
from .tensor import Tensor
from .tensor import TensorPurpose
from .utils import progress_print


class LiveRange:
    def __init__(self, tens, alignment):
        self.tensors = []  # Tensors that are assigned to the same LiveRange will be allocated to the same address
        self.start_time = 99999999999
        self.end_time = -1
        self.size = 0
        self.name = ""
        self.alignment = alignment
        self.mem_area = tens.mem_area if tens else MemArea.Unknown

        if tens:
            self.add_tensor(tens)

    def __str__(self):
        return (
            f"<live_range.LiveRange: {self.start_time}-{self.end_time}, "
            f"size={self.size}, '{self.name}' #:{len(self.tensors)}>"
        )

    __repr__ = __str__

    def add_tensor(self, tens):
        if self.size == 0:
            self.size = tens.storage_size()
            self.name = tens.name  # LiveRange will be named after the first tensor added
        else:
            assert (
                self.size >= tens.storage_size()
            ), "Tensors assigned to the same LiveRange need to fit the size of the LiveRange."

        self.tensors.append(tens)

    def mark_usage(self, op_time, op_length=1):
        op_time_start = max(op_time, 0)
        op_time_end = op_time + op_length
        if op_time_end < op_time_start:
            return

        self.start_time = min(self.start_time, op_time_start)
        self.end_time = max(self.end_time, op_time_end)

    def set_buffer_size(self, buffer_size):
        self.size = buffer_size
        self.mem_area = MemArea.Sram

    def overlaps_ranges(self, other):
        return max(self.start_time, other.start_time) < min(self.end_time, other.end_time)

    def overlaps_address(self, other):
        # Returns the first pair of tensors in this LiveRange and 'other' which have
        # overlapping addresses
        for tens in self.tensors:
            for other_tens in other.tensors:
                if max(tens.address, other_tens.address) < min(
                    tens.address + self.size, other_tens.address + other.size
                ):
                    return True, tens, other_tens

        return False, None, None

    def __lt__(self, other):
        if self.start_time != other.start_time:
            return self.start_time < other.start_time
        if self.end_time != other.end_time:
            return self.end_time < other.end_time
        if self.size != other.size:
            return self.size < other.size
        return self.name < other.name

    def set_address(self, address):
        # Set address of all tensors in LiveRange
        for tens in self.tensors:
            tens.address = address

        return address

    def get_alignment(self):
        return self.alignment

    def set_alignment(self, alignment):
        self.alignment = max(self.alignment, alignment)


class LiveRangeGraph:
    def __init__(self):
        self.lrs: List[LiveRange] = []  # List of all created ranges
        self.ranges = {}  # tens -> range
        self.processed_subgraphs = set()
        self.current_time = 0
        self.end_time = None

    def get_or_create_range(self, tens, alignment=Tensor.AllocationQuantum):
        # Return the live range of the tensor (or any of its clones)
        for existing_tensor, rng in self.ranges.items():
            if tens.equivalent(existing_tensor):
                rng.set_alignment(alignment)
                return rng

        # No live range found for the tensor, create a new one
        rng = LiveRange(tens, alignment)
        self.ranges[tens] = rng
        self.lrs.append(rng)
        return rng

    def fuse_ranges(self, in_tens, out_tens):
        live_range = self.get_or_create_range(in_tens)
        assert out_tens not in self.ranges, out_tens
        live_range.add_tensor(out_tens)
        self.ranges[out_tens] = live_range
        return live_range

    def get_endtime(self):
        # op_length is 1 so max end time for lr is current + 1
        return self.current_time + 1

    def get_temporal_memory_usage(self, target_mem_area):
        usage = np.zeros(self.get_endtime() + 1, dtype=np.int32)
        for lr in self.lrs:
            if lr.mem_area == target_mem_area:
                # End time is inclusive
                assert lr.end_time <= self.get_endtime() + 1
                usage[lr.start_time : lr.end_time + 1] += lr.size

        return usage


def tensor_should_be_ignored(tens, target_mem_area, target_mem_type_set):
    if tens.purpose == TensorPurpose.Virtual:
        return True
    if target_mem_area is None or target_mem_type_set is None:
        return False
    if tens.mem_area != target_mem_area or tens.mem_type not in target_mem_type_set:
        return True
    return False


def _get_ifm_to_fuse(sched_op, target_mem_area=None, target_mem_type_set=None):
    ifm_tens = None
    elem_op = sched_op.parent_op
    if sched_op.op_type.is_elementwise_op() and elem_op.memory_function is not Op.VariableTensorWrite:
        # Check if possible to merge ifm/ofm live ranges of elementwise op
        if not tensor_should_be_ignored(elem_op.ofm, target_mem_area, target_mem_type_set):
            # Check if overwriting the inputs can be allowed
            OpShapeTens = namedtuple("OpShapeTens", ["op_shape", "tens"])
            outp = OpShapeTens(elem_op.ofm_shapes[0], elem_op.ofm)
            inps = []
            if elem_op.ifm is not None:
                inps.append(OpShapeTens(elem_op.ifm_shapes[0], elem_op.ifm))
            if elem_op.ifm2 is not None:
                inps.append(OpShapeTens(elem_op.ifm_shapes[1], elem_op.ifm2))
            # find an input tensor that can be overwritten by the output
            for inp in inps:
                if (
                    # check op input and output shapes allow overlapping
                    inp.op_shape == outp.op_shape
                    # check input tensor is valid
                    and inp.tens is not None
                    and inp.tens.shape != []
                    and not inp.tens.ifm_write_protected
                    and not tensor_should_be_ignored(inp.tens, target_mem_area, target_mem_type_set)
                    # check input and output tensors are compatible
                    and inp.tens.format == outp.tens.format
                    and inp.tens.dtype == outp.tens.dtype
                    # check input tensor only has one consumer
                    and len(inp.tens.consumer_list) == 1
                    # check output tensor only has one producer
                    and len(outp.tens.ops) == 1
                ):
                    ifm_tens = inp.tens
                    break
    elif sched_op.op_type == Op.Memcpy:
        # Check if possible to merge ifm/ofm live ranges of dma op
        dma_op = sched_op.parent_op
        ifm = dma_op.ifm
        ofm = dma_op.ofm
        if not (
            tensor_should_be_ignored(ifm, target_mem_area, target_mem_type_set)
            or tensor_should_be_ignored(ofm, target_mem_area, target_mem_type_set)
            # input tensor only allowed to have one consumer
            or len(ifm.consumer_list) > 1
        ):
            # Currently DMA only used when bypassing memory only ops so ok to reuse ifm
            # if ifm has only one consumer
            ifm_tens = ifm

    return ifm_tens


def ofm_can_reuse_ifm(sched_op, target_mem_area=None, target_mem_type_set=None):
    ifm = _get_ifm_to_fuse(sched_op, target_mem_area, target_mem_type_set)
    return ifm is not None


def merge_elementwise_op_ranges(sg, sched_op, lr_graph, target_mem_area, target_mem_type_set):
    ifm = _get_ifm_to_fuse(sched_op, target_mem_area, target_mem_type_set)
    if ifm:
        lr_graph.fuse_ranges(ifm, sched_op.parent_op.ofm)


def extract_live_ranges_from_cascaded_passes(
    sg,
    target_mem_area,
    target_mem_type_set,
    lr_graph=None,
    cpu_tensor_alignment=Tensor.AllocationQuantum,
    verbose_progress: bool = False,
):
    if lr_graph is None:
        lr_graph = LiveRangeGraph()

    if sg in lr_graph.processed_subgraphs:
        # if subgraph has been processed already, return the lr_graph as is
        return lr_graph

    for index, cps in enumerate(sg.cascaded_passes):
        progress_print(verbose_progress, "Processing cascaded pass", index, sg.cascaded_passes)
        cps.time = lr_graph.current_time

        time_for_pass = cps.time

        for tens in cps.inputs:
            if tensor_should_be_ignored(tens, target_mem_area, target_mem_type_set):
                continue
            rng = lr_graph.get_or_create_range(tens, cpu_tensor_alignment)
            rng.mark_usage(time_for_pass)

        op = cps.passes[0].ops[0] if cps.passes[0].ops else None
        op_subgraph = op.attrs.get("subgraph", None) if op else None

        if op_subgraph is not None and MemType.Permanent_CPU not in target_mem_type_set:
            if op.type == Op.CustomNpuOp:
                # If the primary-op is an NpuOp that means this is where an Npu subgraph
                # is called. Go into said subgraph and extract live ranges before continuing.
                # Use default allocation alignment of 16 for Npu tensors
                lr_graph = extract_live_ranges_from_schedule(
                    op_subgraph, target_mem_area, target_mem_type_set, lr_graph
                )
            else:
                # The op has one or more subgraphs in it (a typical op is the While op)
                # Go into all subgraphs and extract live ranges before continuing.
                for op_sg in op_subgraph:
                    lr_graph = extract_live_ranges_from_cascaded_passes(
                        op_sg, target_mem_area, target_mem_type_set, lr_graph, cpu_tensor_alignment
                    )
            # Set the new time after handling the Npu subgraph
            # current time is updated in subgraph path so do not tick the time
            time_for_pass = lr_graph.current_time
            cps.time = time_for_pass
        else:
            lr_graph.current_time += 2

        for tens in cps.intermediates + cps.outputs:
            if tensor_should_be_ignored(tens, target_mem_area, target_mem_type_set):
                continue
            rng = lr_graph.get_or_create_range(tens, cpu_tensor_alignment)
            rng.mark_usage(time_for_pass)

    time_to_set = lr_graph.current_time
    for tens in sg.output_tensors:
        if tensor_should_be_ignored(tens, target_mem_area, target_mem_type_set):
            continue
        rng = lr_graph.get_or_create_range(tens, cpu_tensor_alignment)
        rng.mark_usage(time_to_set)

    # Variable tensor live-range is for entire inference
    for tens, rng in lr_graph.ranges.items():
        if tens.is_variable:
            rng.mark_usage(0, time_to_set + 1)

    # Add subgraph to set of processed subgraphs
    lr_graph.processed_subgraphs.add(sg)
    return lr_graph


def create_linear_live_range_graph(sg, target_mem_area, target_mem_type_set, lr_graph):
    assert lr_graph is not None
    sg_time = lr_graph.current_time
    for ps in sg.passes:
        for tens in ps.inputs + ps.outputs + ps.intermediates:
            if tens.purpose == TensorPurpose.Weights or tensor_should_be_ignored(
                tens, target_mem_area, target_mem_type_set
            ):
                continue
            rng = lr_graph.get_or_create_range(tens)
            rng.mark_usage(sg_time)

    for _, op_info in sg.schedule.cost_map.items():
        for tensor in [op_info.npu_weights_tensor, op_info.npu_scales_tensor]:
            if tensor and not (tensor_should_be_ignored(tensor, target_mem_area, target_mem_type_set)):
                rng = lr_graph.get_or_create_range(tensor)
                rng.mark_usage(sg_time)

    lr_graph.current_time += 1
    return lr_graph


def extract_live_ranges_from_schedule(sg, target_mem_area, target_mem_type_set, lr_graph, verbose_progress=False):
    time_for_cascade = {}
    for index, sched_op in enumerate(sg.sched_ops):
        progress_print(verbose_progress, "Processing SchedulerOp", index, sg.sched_ops)
        op_info = sg.schedule.cost_map[sched_op]
        cascade = op_info.cascade
        cascade_info = sg.schedule.cascades.get(cascade, None)

        if cascade_info is None:
            # Op is not part of a cascade, check if the ifm can be overwritten by the ofm
            merge_elementwise_op_ranges(sg, sched_op, lr_graph, target_mem_area, target_mem_type_set)

        time_to_set = time_for_cascade.get(cascade, lr_graph.current_time)

        op_info.time_index = time_to_set

        # Mark usage for all tensors related to this Pass
        ps = sched_op.parent_ps
        for tens in ps.inputs + ps.outputs + ps.intermediates:
            if (
                target_mem_area == MemArea.Sram
                and cascade_info
                and tens == ps.ifm_tensor
                and sched_op in cascade_info.buffers
            ):
                # This tensor is a rolling buffer in a cascade and the size of the LiveRange needs to be modified
                # for enabling temporal memory snapshots without modifying the original Tensor
                rng = lr_graph.get_or_create_range(tens)
                rng.set_buffer_size(cascade_info.buffers[sched_op].elements() * sched_op.ifm.dtype.size_in_bytes())
            elif (
                tens.purpose == TensorPurpose.Weights
                or tens.purpose == TensorPurpose.FSBias
                or tens.mem_type not in target_mem_type_set
                or tens.mem_area != target_mem_area
            ):
                continue

            else:
                rng = lr_graph.get_or_create_range(tens)

            rng.mark_usage(time_to_set)

        for idx, weight_tens in enumerate(op_info.buffered_weight_tensors):
            if weight_tens.mem_type in target_mem_type_set and weight_tens.mem_area == target_mem_area:
                rng = lr_graph.get_or_create_range(weight_tens)
                start_time = time_to_set
                length = 1
                if weight_tens.pre_buffer:
                    start_time -= 1
                    length += 1
                if len(op_info.buffered_weight_tensors) > 1:
                    last_idx = len(op_info.ofm_depth_slices) % len(op_info.buffered_weight_tensors)
                    # Double buffering: reduce end time of the buffer that is not used last
                    if last_idx != idx:
                        length -= 1
                rng.mark_usage(start_time, length)

        if time_to_set == lr_graph.current_time:
            lr_graph.current_time += 2

        if cascade != 0:
            time_for_cascade[cascade] = time_to_set

    time_to_set = lr_graph.current_time
    for tens in sg.output_tensors:
        if tens.mem_type not in target_mem_type_set or tens.mem_area != target_mem_area:
            continue
        rng = lr_graph.get_or_create_range(tens)
        rng.mark_usage(time_to_set)

    return lr_graph
