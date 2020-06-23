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
from .high_level_command_stream_generator import calc_allowed_ofm_ifm_overlap_for_cascaded_pass
from .nn_graph import PassPlacement
from .tensor import MemType
from .tensor import Tensor


class LiveRange:
    def __init__(self, tens):
        self.tensors = []  # Tensors that are assigned to the same LiveRange will be allocated to the same address
        self.start_time = 99999999999
        self.end_time = -1
        self.size = 0
        self.name = ""

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
        # Set address of all unaddressed tensors in LiveRange
        for tens in self.tensors:
            if tens.address is None:
                addr = address
            else:
                # Limit to single tensor for the lr if the tensor address already assigned
                assert len(self.tensors) == 1
                addr = tens.address
            tens.address = addr
            # Also need to set the address to the tensor's cpu/npu clones
            if tens.cpu_tensor is not None:
                tens.cpu_tensor.address = addr
            if tens.npu_tensor is not None:
                tens.npu_tensor.address = addr
        return addr

    def get_alignment(self):
        # Get max alignment of LiveRange's tensors
        if self.tensors:
            alignment = 0
            for tens in self.tensors:
                alignment = max(alignment, tens.alignment)

            return alignment

        return Tensor.AllocationQuantum


def merge_memory_op_ranges(sg, lr_graph, tensor_should_be_ignored, target_mem_area):
    for ps in sg.passes:
        if ps.placement == PassPlacement.MemoryOnly:
            # For memory only passes, e.g. Reshape. Add input and output tensor to the same LiveRange
            input_tensor = ps.inputs[0]
            output_tensor = ps.outputs[0]
            # If the input or output tensor is tied to a Cpu tensor, i.e. a subgraph input
            # or output, fuse the live-range with the Cpu tensors' live-range instead.
            input_tensor = input_tensor.cpu_tensor if input_tensor.cpu_tensor is not None else input_tensor
            output_tensor = output_tensor.cpu_tensor if output_tensor.cpu_tensor is not None else output_tensor
            if not tensor_should_be_ignored(input_tensor, target_mem_area) and not tensor_should_be_ignored(
                output_tensor, target_mem_area
            ):
                lr_graph.fuse_ranges(input_tensor, output_tensor)


class LiveRangeGraph:
    def __init__(self):
        self.ranges = {}  # tens -> range
        self.allowed_overlaps = {}  # (tens,tens) -> overlap_int
        self.ignore_tensors = set()
        self.processed_subgraphs = set()
        self.current_time = 0

    def get_or_create_range(self, tens):
        for rng in self.ranges.values():
            # Return the live range of the tensor (or it's cpu/npu clone)
            if any(tensor in rng.tensors for tensor in [tens, tens.npu_tensor, tens.cpu_tensor]):
                return rng

        # No live range found for the tensor, create a new one
        rng = LiveRange(tens)
        self.ranges[tens] = rng
        return rng

    def fuse_ranges(self, in_tens, out_tens):
        live_range = self.get_or_create_range(in_tens)
        assert out_tens not in self.ranges, out_tens
        live_range.add_tensor(out_tens)
        self.ranges[out_tens] = live_range
        return live_range


def extract_live_ranges_from_passes(
    sg,
    target_mem_area,
    mark_output_tensors_overlapping_with_input_tensors=False,
    ignore_subgraph_input_output_tensors=False,
):
    lr_graph = LiveRangeGraph()

    if ignore_subgraph_input_output_tensors:
        lr_graph.ignore_tensors.update(sg.input_tensors)
        lr_graph.ignore_tensors.update(sg.output_tensors)

    def tensor_should_be_ignored(tens, target_mem_area):
        if tens.mem_area != target_mem_area:
            return True
        if tens in lr_graph.ignore_tensors:
            return True
        if tens.name.endswith("reshape_shape_npu"):
            # Reshape tensor, no need to allocate
            lr_graph.ignore_tensors.add(tens)
            return True
        return False

    # Merge only memory operations in the NPU subgraphs
    if sg.placement == PassPlacement.Npu:
        merge_memory_op_ranges(sg, lr_graph, tensor_should_be_ignored, target_mem_area)

    for idx, ps in enumerate(sg.passes):
        ps.time = 2 * idx

        time_for_pass = ps.time

        for tens in ps.inputs:
            if tensor_should_be_ignored(tens, target_mem_area):
                continue
            rng = lr_graph.get_or_create_range(tens)
            rng.mark_usage(time_for_pass)

        for tens in ps.intermediates:
            if tensor_should_be_ignored(tens, target_mem_area):
                continue
            rng = lr_graph.get_or_create_range(tens)
            rng.mark_usage(time_for_pass)

        for tens in ps.outputs:
            if tensor_should_be_ignored(tens, target_mem_area):
                continue
            rng = lr_graph.get_or_create_range(tens)
            output_time = time_for_pass
            if not mark_output_tensors_overlapping_with_input_tensors and ps.is_element_wise:
                output_time += 1
            rng.mark_usage(output_time)

    end_time = len(sg.passes) * 2
    for tens in sg.output_tensors:
        if tensor_should_be_ignored(tens, target_mem_area):
            continue
        rng = lr_graph.get_or_create_range(tens)
        rng.mark_usage(end_time)

    return lr_graph


def extract_live_ranges_from_cascaded_passes(
    sg,
    target_mem_area,
    target_mem_type_set,
    mark_output_tensors_overlapping_with_input_tensors=False,
    use_ifm_ofm_overlap=True,
    ignore_subgraph_input_output_tensors=False,
    lr_graph=None,
):
    if lr_graph is None:
        lr_graph = LiveRangeGraph()

    if sg in lr_graph.processed_subgraphs:
        # if subgraph has been processed already, return the lr_graph as is
        return lr_graph

    if ignore_subgraph_input_output_tensors:
        lr_graph.ignore_tensors.update(sg.input_tensors)
        lr_graph.ignore_tensors.update(sg.output_tensors)

    def tensor_should_be_ignored(tens, target_mem_area, target_mem_type_set):
        if tens.mem_area != target_mem_area or tens.mem_type not in target_mem_type_set:
            return True
        if tens in lr_graph.ignore_tensors:
            return True
        if tens.name.endswith("reshape_shape_npu"):
            # Reshape tensor, no need to allocate
            lr_graph.ignore_tensors.add(tens)
            return True
        return False

    def merge_memory_op_ranges(sg, lr_graph, tensor_should_be_ignored, target_mem_area, target_mem_type_set):
        for ps in sg.passes:
            if ps.placement == PassPlacement.MemoryOnly:
                # For memory only passes, e.g. Reshape. Add input and output tensor to the same LiveRange
                input_tensor = ps.inputs[0]
                output_tensor = ps.outputs[0]
                # If the input or output tensor is tied to a Cpu tensor, i.e. a subgraph input
                # or output, fuse the live-range with the Cpu tensors' live-range instead.
                input_tensor = input_tensor.cpu_tensor if input_tensor.cpu_tensor is not None else input_tensor
                output_tensor = output_tensor.cpu_tensor if output_tensor.cpu_tensor is not None else output_tensor
                if not tensor_should_be_ignored(input_tensor, target_mem_area, target_mem_type_set) and not (
                    tensor_should_be_ignored(output_tensor, target_mem_area, target_mem_type_set)
                ):
                    lr_graph.fuse_ranges(input_tensor, output_tensor)

    # Merge only memory operations in the NPU subgraphs
    if sg.placement == PassPlacement.Npu:
        merge_memory_op_ranges(sg, lr_graph, tensor_should_be_ignored, target_mem_area, target_mem_type_set)

    for cps in sg.cascaded_passes:
        cps.time = lr_graph.current_time

        time_for_pass = cps.time

        is_element_wise = cps.is_element_wise

        for tens in cps.inputs:
            if tensor_should_be_ignored(tens, target_mem_area, target_mem_type_set):
                continue
            rng = lr_graph.get_or_create_range(tens)
            rng.mark_usage(time_for_pass)

        cps_primary_op = cps.passes[0].primary_op

        if cps_primary_op and cps_primary_op.type == "NpuOp" and MemType.Permanent_CPU not in target_mem_type_set:
            # If the primary-op is an NpuOp that means this is where an Npu subgraph
            # is called. Go into said subgraph and extract live ranges before continuing.
            npu_sg = cps_primary_op.attrs["subgraph"]
            lr_graph = extract_live_ranges_from_cascaded_passes(
                npu_sg,
                target_mem_area,
                target_mem_type_set,
                mark_output_tensors_overlapping_with_input_tensors,
                use_ifm_ofm_overlap,
                False,
                lr_graph,
            )
            # Set the new time after handling the Npu subgraph
            time_for_pass = lr_graph.current_time
            cps.time = time_for_pass

        for tens in cps.intermediates:
            if tensor_should_be_ignored(tens, target_mem_area, target_mem_type_set):
                continue
            rng = lr_graph.get_or_create_range(tens)
            rng.mark_usage(time_for_pass)

        for tens in cps.outputs:
            if tensor_should_be_ignored(tens, target_mem_area, target_mem_type_set):
                continue
            rng = lr_graph.get_or_create_range(tens)
            output_time = time_for_pass
            if not mark_output_tensors_overlapping_with_input_tensors and is_element_wise:
                output_time += 1
            rng.mark_usage(output_time)

        if use_ifm_ofm_overlap:
            # fill allowed overlap for ifm and ofm tensor
            ifm_tensor = cps.passes[0].ifm_tensor
            ofm_tensor = cps.passes[-1].ofm_tensor
            if (
                ifm_tensor is not None
                and ofm_tensor is not None
                and not tensor_should_be_ignored(ifm_tensor, target_mem_area, target_mem_type_set)
                and not tensor_should_be_ignored(ofm_tensor, target_mem_area, target_mem_type_set)
            ):
                lr_graph.allowed_overlaps[(ifm_tensor, ofm_tensor)] = calc_allowed_ofm_ifm_overlap_for_cascaded_pass(
                    cps
                )

        lr_graph.current_time += 2

    end_time = 0
    for rng in lr_graph.ranges.values():
        # Find the maximum end time of all live-ranges in the graph
        end_time = max(end_time, rng.end_time)

    for tens in sg.output_tensors:
        if tensor_should_be_ignored(tens, target_mem_area, target_mem_type_set):
            continue
        rng = lr_graph.get_or_create_range(tens)
        rng.mark_usage(end_time)

    # Add subgraph to set of processed subgraphs
    lr_graph.processed_subgraphs.add(sg)
    return lr_graph
