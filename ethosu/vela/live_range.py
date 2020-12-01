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
# Build a live range graph for tensors in one or more subgraphs. Used for tensor allocation as well as in the scheduler.
# Can work with either a pass packed subgraph or a scheduled subgraph.
from .nn_graph import PassPlacement
from .operation import Op
from .tensor import MemType
from .tensor import Tensor


class LiveRange:
    def __init__(self, tens, alignment):
        self.tensors = []  # Tensors that are assigned to the same LiveRange will be allocated to the same address
        self.start_time = 99999999999
        self.end_time = -1
        self.size = 0
        self.name = ""
        self.alignment = alignment

        if tens:
            self.add_tensor(tens)

    def __str__(self):
        return "<live_range.LiveRange: '%s' start_time=%s, end_time=%s>" % (self.name, self.start_time, self.end_time)

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

    def mark_usage(self, op_time):
        if op_time == -1:
            return
        op_time_start = op_time
        op_time_end = op_time + 1

        self.start_time = min(self.start_time, op_time_start)
        self.end_time = max(self.end_time, op_time_end)

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
        self.ranges = {}  # tens -> range
        self.ignore_tensors = set()
        self.processed_subgraphs = set()
        self.current_time = 0

    def get_or_create_range(self, tens, alignment=Tensor.AllocationQuantum):
        # Return the live range of the tensor (or any of its clones)
        for existing_tensor, rng in self.ranges.items():
            if tens.equivalent(existing_tensor):
                rng.set_alignment(alignment)
                return rng

        # No live range found for the tensor, create a new one
        rng = LiveRange(tens, alignment)
        self.ranges[tens] = rng
        return rng

    def fuse_ranges(self, in_tens, out_tens):
        live_range = self.get_or_create_range(in_tens)
        assert out_tens not in self.ranges, out_tens
        live_range.add_tensor(out_tens)
        self.ranges[out_tens] = live_range
        return live_range


def tensor_should_be_ignored(lr_graph, tens, target_mem_area, target_mem_type_set):
    if tens.mem_area != target_mem_area or tens.mem_type not in target_mem_type_set:
        return True
    if tens in lr_graph.ignore_tensors:
        return True
    if tens.name.endswith("reshape_shape_npu"):
        # Reshape tensor, no need to allocate
        lr_graph.ignore_tensors.add(tens)
        return True
    return False


# Tries merging of ifm/ofm live ranges for memory only ops and elementwise ops
def merge_op_ranges(sg, lr_graph, target_mem_area, target_mem_type_set):
    for ps in sg.passes:
        if ps.placement == PassPlacement.MemoryOnly:
            # For memory only passes, e.g. Reshape. Add input and output tensor to the same LiveRange
            input_tensor = ps.inputs[0]
            output_tensor = ps.outputs[0]
            if not tensor_should_be_ignored(lr_graph, input_tensor, target_mem_area, target_mem_type_set) and not (
                tensor_should_be_ignored(lr_graph, output_tensor, target_mem_area, target_mem_type_set)
            ):
                lr_graph.fuse_ranges(input_tensor, output_tensor)
        elif ps.is_element_wise:
            merge_elementwise_op_ranges(ps, lr_graph, target_mem_area, target_mem_type_set)


# Tries to merge ifm/ofm live of elementwise op
def merge_elementwise_op_ranges(ps, lr_graph, target_mem_area, target_mem_type_set):
    elem_op = None
    for op in ps.ops:
        if op.type.is_elementwise_op():
            assert elem_op is None
            elem_op = op

    if elem_op is not None and not tensor_should_be_ignored(
        lr_graph, elem_op.ofm, target_mem_area, target_mem_type_set
    ):
        # Check if overwriting the inputs can be allowed
        if elem_op.type not in (Op.SHL, Op.SHR):
            inps = []
            if (
                elem_op.ifm is not None
                and elem_op.ifm.shape != []
                and elem_op.ifm.mem_area == target_mem_area
                and elem_op.ifm.mem_type in target_mem_type_set
            ):
                inps.append(elem_op.ifm)
            if (
                elem_op.ifm2 is not None
                and elem_op.ifm2.shape != []
                and elem_op.ifm2.mem_area == target_mem_area
                and elem_op.ifm.mem_type in target_mem_type_set
            ):
                inps.append(elem_op.ifm2)

            if len(inps) > 0:
                for i, inp in enumerate(inps):
                    # check input format, dtype, broadcasting or if there are more input consumers
                    if (
                        inp.format == elem_op.ofm.format
                        and inp.dtype == elem_op.ofm.dtype
                        and elem_op.ifm_shapes[i] == elem_op.ofm_shapes[0]
                        and (len(inp.consumer_list) == 1 and len(inp.ops) == 1)
                    ):
                        lr_graph.fuse_ranges(inp, elem_op.ofm)
                        break


def extract_live_ranges_from_passes(
    sg, target_mem_area, target_mem_type_set=None, ignore_subgraph_input_output_tensors=False,
):
    lr_graph = LiveRangeGraph()

    if ignore_subgraph_input_output_tensors:
        lr_graph.ignore_tensors.update(sg.input_tensors)
        lr_graph.ignore_tensors.update(sg.output_tensors)

    if target_mem_type_set is None:
        target_mem_type_set = set((MemType.Scratch, MemType.Scratch_fast))

    # Try to merge live ranges of operations in the NPU subgraphs
    if sg.placement == PassPlacement.Npu:
        merge_op_ranges(sg, lr_graph, target_mem_area, target_mem_type_set)

    for idx, ps in enumerate(sg.passes):
        ps.time = 2 * idx

        time_for_pass = ps.time

        for tens in ps.inputs + ps.intermediates + ps.outputs:
            if tensor_should_be_ignored(lr_graph, tens, target_mem_area, target_mem_type_set):
                continue
            rng = lr_graph.get_or_create_range(tens)
            rng.mark_usage(time_for_pass)

    end_time = len(sg.passes) * 2
    for tens in sg.output_tensors:
        if tensor_should_be_ignored(lr_graph, tens, target_mem_area, target_mem_type_set):
            continue
        rng = lr_graph.get_or_create_range(tens)
        rng.mark_usage(end_time)

    return lr_graph


def extract_live_ranges_from_cascaded_passes(
    sg,
    target_mem_area,
    target_mem_type_set,
    ignore_subgraph_input_output_tensors=False,
    lr_graph=None,
    cpu_tensor_alignment=Tensor.AllocationQuantum,
):
    if lr_graph is None:
        lr_graph = LiveRangeGraph()

    if sg in lr_graph.processed_subgraphs:
        # if subgraph has been processed already, return the lr_graph as is
        return lr_graph

    if ignore_subgraph_input_output_tensors:
        lr_graph.ignore_tensors.update(sg.input_tensors)
        lr_graph.ignore_tensors.update(sg.output_tensors)

    # Try to merge live ranges of operations in the NPU subgraphs
    if sg.placement == PassPlacement.Npu:
        merge_op_ranges(sg, lr_graph, target_mem_area, target_mem_type_set)

    for cps in sg.cascaded_passes:
        cps.time = lr_graph.current_time

        time_for_pass = cps.time

        for tens in cps.inputs:
            if tensor_should_be_ignored(lr_graph, tens, target_mem_area, target_mem_type_set):
                continue
            rng = lr_graph.get_or_create_range(tens, cpu_tensor_alignment)
            rng.mark_usage(time_for_pass)

        cps_primary_op = cps.passes[0].primary_op

        if (
            cps_primary_op
            and cps_primary_op.type == Op.CustomNpuOp
            and MemType.Permanent_CPU not in target_mem_type_set
        ):
            # If the primary-op is an NpuOp that means this is where an Npu subgraph
            # is called. Go into said subgraph and extract live ranges before continuing.
            # Use default allocation alignment of 16 for Npu tensors
            npu_sg = cps_primary_op.attrs["subgraph"]
            lr_graph = extract_live_ranges_from_cascaded_passes(
                npu_sg, target_mem_area, target_mem_type_set, False, lr_graph,
            )
            # Set the new time after handling the Npu subgraph
            time_for_pass = lr_graph.current_time
            cps.time = time_for_pass

        for tens in cps.intermediates + cps.outputs:
            if tensor_should_be_ignored(lr_graph, tens, target_mem_area, target_mem_type_set):
                continue
            rng = lr_graph.get_or_create_range(tens, cpu_tensor_alignment)
            rng.mark_usage(time_for_pass)

        lr_graph.current_time += 2

    end_time = 0
    for rng in lr_graph.ranges.values():
        # Find the maximum end time of all live-ranges in the graph
        end_time = max(end_time, rng.end_time)

    for tens in sg.output_tensors:
        if tensor_should_be_ignored(lr_graph, tens, target_mem_area, target_mem_type_set):
            continue
        rng = lr_graph.get_or_create_range(tens, cpu_tensor_alignment)
        rng.mark_usage(end_time)

    # Add subgraph to set of processed subgraphs
    lr_graph.processed_subgraphs.add(sg)
    return lr_graph
