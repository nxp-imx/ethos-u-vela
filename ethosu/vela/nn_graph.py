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
# Neural network graph classes and enums.
# Pass - A packed pass containing one or more Operations.
# CascadedPass - A scheduled pass containing one or more Passes, as well as a scheduling strategy and block
#                configurations.
# Subgraph - Holds a neural network subgraph, pointing at Tensors, Operations, Passes, and CascadedPasses.
# Graph - A full neural network graph with one or more Subgraphs.
import enum
from typing import List
import csv

from .operation import Op
from .shape4d import Shape4D


class PassPlacement(enum.Enum):
    Unknown = 0
    Cpu = 1
    Npu = 2
    MemoryOnly = 3
    StartupInit = 4


class TensorAllocator(enum.Enum):
    LinearAlloc = 1
    Greedy = 2
    HillClimb = 3

    def __str__(self):
        return self.name


class NetworkType(enum.Enum):
    TFLite = 1
    TOSA = 2


class Pass:
    def __init__(self, name, placement, is_element_wise, npu_block_type):
        self.inputs = []
        self.intermediates = []
        self.outputs = []
        self.ops = []
        self.primary_op = None
        self.ifm_tensor = None
        self.ifm2_tensor = None
        self.ofm_tensor = None
        self.weight_tensor = None
        self.scale_tensor = None
        self.lut_tensor = None
        self.name = name
        self.cascade = None
        self.placement = placement
        self.ifm_shapes: List[Shape4D] = []
        self.ofm_shapes: List[Shape4D] = []

        # TODO: rename is_element_wise because it is not the same as an ElementWise operator. It is used by the tensor
        # allocation and requires that the OFM and IFM has the exact same address. Essentially complete overlap.
        self.is_element_wise = is_element_wise
        self.npu_block_type = npu_block_type
        self.block_config = None  # will be filled in by scheduler
        self.shared_buffer = None  # will be filled in by scheduler
        self.scheduling_info = None  # will be filled in by scheduler

        self.predecessors = []
        self.successors = []

    def __str__(self):
        return "<nng.Pass '%s', %s, ops=%s>" % (self.name, self.placement, [op.type for op in self.ops])

    __repr__ = __str__

    def get_primary_op_ifm_weights(self):
        if not self.primary_op:
            return None, None
        return self.primary_op.get_ifm_ifm2_weights_ofm()[::2]

    def get_primary_op_ifm_ifm2_weights_ofm(self):
        if not self.primary_op:
            return None, None, None, None
        return self.primary_op.get_ifm_ifm2_weights_ofm()

    def get_primary_op_ifm_weights_biases_ofm(self):
        if not self.primary_op:
            return None, None, None, None
        return self.primary_op.get_ifm_weights_biases_ofm()

    def get_primary_op_lut(self):
        if not self.primary_op:
            return None
        return self.primary_op.activation_lut


class SchedulingStrategy(enum.Enum):
    Unknown = -1
    IfmStream = 0
    WeightStream = 1


class SchedulerRewrite(enum.Enum):
    Nop = 0
    ChangeTensorSubPurpose = 1


class CascadedPass:
    def __init__(self, name, strat, inputs, intermediates, outputs, passes, placement, is_element_wise):
        self.name = name
        self.strategy = strat
        self.inputs = inputs
        self.intermediates = intermediates
        self.outputs = outputs
        self.passes = passes
        self.placement = placement
        self.is_element_wise = is_element_wise

        self.predecessors = []
        self.successors = []
        self.sram_used = 0
        self.time = 0

    def __str__(self):
        return "<nng.CascadedPass strategy=%s x %s '%s',  passes=%s, block_configs=%s>" % (
            self.strategy,
            len(self.passes),
            self.name,
            [ps.name for ps in self.passes],
            [ps.block_config for ps in self.passes],
        )

    __repr__ = __str__


class Subgraph:
    def __init__(self, name="<unnamed>", placement=PassPlacement.Cpu):
        self.output_tensors = []
        self.input_tensors = []
        # Preserve the original input order
        self.original_inputs = []
        # Attach virtual outputs to resource variables op
        # in order to be able to traverse the graph correctly
        self.virtual_outputs = []
        self.passes = []
        self.cascaded_passes = []
        self.name = name
        self.high_level_command_stream = []
        self.placement = placement
        self.command_stream_tensor = None
        self.flash_tensor = None
        # Scratch information locally used in the scheduler
        self.schedule = None
        self.sched_ops = []

        self.generated_stream_id = None

        self.memory_used = {}
        self.memory_used_per_type = {}

    def __str__(self):
        return "<nng.Subgraph '%s',  n_passes=%d, n_cascaded_passes=%d>" % (
            self.name,
            len(self.passes),
            len(self.cascaded_passes),
        )

    __repr__ = __str__

    def update_consumers(self):
        visit_op_set = set()
        visit_tensor_set = set()
        self.input_tensors = []

        print_visit = False

        def visit_op(op):
            if op in visit_op_set:
                return

            visit_op_set.add(op)
            for inp in op.inputs:
                if not inp:
                    continue
                if print_visit:
                    print(inp, "adding consumer", op)
                visit_tensor(inp)
                inp.consumer_list.append(op)

            if op.type in (Op.Placeholder, Op.SubgraphInput):
                assert len(op.outputs) == 1
                if not op.outputs[0].is_variable:
                    self.input_tensors.append(op.outputs[0])

            for out in op.outputs:
                if out not in visit_tensor_set:
                    out.consumer_list = []  # reset unvisited output, just in case

        def visit_tensor(tens):
            if tens in visit_tensor_set:
                return
            visit_tensor_set.add(tens)
            tens.consumer_list = []
            for op in tens.ops:
                visit_op(op)

        for ps in self.passes:
            for tens in ps.outputs + ps.inputs:
                if not tens:
                    continue
                tens.consumer_list = []  # reset unvisited tensors to start with

        for tens in self.output_tensors:
            visit_tensor(tens)
            tens.consumer_list.append(None)  # special op to indicate that the graph consumes the result

        print_visit = True
        for ps in self.passes:
            for op in ps.ops:
                visit_op(op)
            for tens in ps.inputs:
                visit_tensor(tens)

    def build_pass_links(self):
        for idx, ps in enumerate(self.passes):
            ps.time = 2 * idx
            ps.predecessors = []
            ps.successors = []

        for ps in self.passes:
            for tens in ps.inputs:
                for op in tens.ops:
                    pred_pass = op.scheduled_pass
                    assert pred_pass.time < ps.time
                    if ps not in pred_pass.successors:
                        pred_pass.successors.append(ps)

                    if pred_pass not in ps.predecessors:
                        ps.predecessors.append(pred_pass)

                    assert tens in pred_pass.outputs

    def build_pass_dag_predecessors(self):
        for ps in self.passes:
            ps.dag_predecessors = []

        class State(enum.Enum):
            NotVisited = 0
            BeingVisited = 1
            Visited = 2

        pass_visit_dict = {}

        def visit_pass(ps):
            state = pass_visit_dict.get(ps, State.NotVisited)
            if state == State.Visited:
                return True
            elif state == State.BeingVisited:
                return False  # this is a loop, need to remove this link
            elif state == State.NotVisited:
                pass_visit_dict[ps] = State.BeingVisited

                ps.dag_predecessors = []
                for pred in ps.predecessors:
                    if visit_pass(pred):
                        ps.dag_predecessors.append(pred)

                pass_visit_dict[ps] = State.Visited
                return True

        for ps in self.passes:
            if not ps.successors:
                visit_pass(ps)

    def build_cascaded_pass_links(self):
        for cps in self.cascaded_passes:
            cps.predecessors = []
            cps.successors = []

        for cps in self.cascaded_passes:
            for tens in cps.inputs:
                for op in tens.ops:
                    pred_cpass = op.scheduled_pass.cascade
                    if cps not in pred_cpass.successors:
                        pred_cpass.successors.append(cps)

                    if pred_cpass not in cps.predecessors:
                        cps.predecessors.append(pred_cpass)

                    assert tens in pred_cpass.outputs

    def refresh_after_modification(self):
        try:
            self.update_consumers()
        except RecursionError as e:
            raise RecursionError(
                "Compilation failed due to exceeding the default maximum recursion depth.\n"
                'Try increasing the maximum recursion depth with the "--recursion-limit" option.'
            ) from e

    def prune_startup_init_pass(self):
        assert len(self.passes) >= 1
        ps = self.passes[0]
        assert ps.placement == PassPlacement.StartupInit

        ps.outputs = [out_tens for out_tens in ps.outputs if len(out_tens.consumers()) > 0]
        ps.ops = [op for op in ps.ops if op.outputs[0] in ps.outputs]

    def get_all_ops(self):
        all_ops = []
        visit_op_set = set()
        visit_tensor_set = set()

        def visit_op(op):
            if op in visit_op_set:
                return
            visit_op_set.add(op)
            for inp in op.inputs:
                visit_tensor(inp)

            all_ops.append(op)

        def visit_tensor(tens):
            if tens is None or tens in visit_tensor_set:
                return
            visit_tensor_set.add(tens)
            for op in tens.ops:
                visit_op(op)

        for tens in self.output_tensors:
            visit_tensor(tens)

        return all_ops

    def print_operators(self, ignore_placeholder_const=True, show_attributes=True):
        print(f"Operators of Subgraph {self.name}")

        ignore_ops = (Op.Const, Op.Identity, Op.Placeholder) if ignore_placeholder_const else ()
        all_ops = [op for op in self.get_all_ops() if op.type not in ignore_ops]

        if len(all_ops) > 0:
            max_op_type_len = max([len(op.type.name) for op in all_ops])

            for idx, op in enumerate(all_ops):
                attrs_str = f" - {op.attrs}" if show_attributes else ""
                print(f"{idx:3}: {op.type:{max_op_type_len}}{attrs_str} - {op.name}")

        else:
            print("No Operators")

    def print_graph(self, label=None):
        if label:
            print(f"\n[ {label} ]")
        print("print_graph()", self.name)
        all_ops = self.get_all_ops()
        for idx, op in enumerate(all_ops):
            print(idx, op.type, op.name)

    def write_tensor_value_to_file(self,file, value):
        if value is not None:
            print(value.shape)
            print(value)
            shape = value.shape
            num_of_values = 1
            for i in shape:
                num_of_values *= i
            value = value.reshape((num_of_values, 1))
            for i in range(num_of_values):
                file.write(str(value[i][0]) + " ")
            file.write("\n")

    def print_npu_graph(self, output_basename, label=None):
        all_ops = self.get_all_ops()
        f = open(output_basename+"_ethos_u_subgraph.csv", "w")
        writer = csv.writer(f, delimiter = '|')
        writer.writerow(['name', 'type', 'inputs', 'outputs', 'attributes'])
        v = open(output_basename + "_ethos_u_subgraph_values.txt", "w")
        for idx, op in enumerate(all_ops):
            for input in op.inputs:
                self.write_tensor_value_to_file(v,input.values)
            if op.run_on_npu:
                writer.writerow([op.name,op.type,op.inputs,op.outputs,op.attrs])
        v.close()
        f.close()

    def print_graph_with_tensors(self):
        print("print_graph_with_tensors()", self.name)
        all_ops = self.get_all_ops()
        for idx, op in enumerate(all_ops):
            print(idx, op.type, op.name)
            for idx, tens in enumerate(op.inputs):
                if tens:
                    print(
                        f"    Input  {idx:02d}"
                        f" {tens.purpose.name:>20} {tens.mem_area.name:>20} {tens.mem_type.name:>20} {tens}"
                    )
                else:
                    print(f"    Input  {idx:02d} {'-':>20} {'-':>20} {'-':>20} {tens}")
            for idx, tens in enumerate(op.outputs):
                print(
                    f"    Output {idx:02d}"
                    f" {tens.purpose.name:>20} {tens.mem_area.name:>20} {tens.mem_type.name:>20} {tens}"
                )
            print()

    def print_graph_with_tensor_quantization(self):
        print("print_graph_with_tensor_quantization()", self.name)
        all_ops = self.get_all_ops()
        for idx, op in enumerate(all_ops):
            print(idx, op.type, op.name)
            for idx, tens in enumerate(op.inputs):
                if tens:
                    q = tens.quantization
                    if q is None:
                        print(f"    Input  {idx:02d} {tens.dtype!s:>10} NO QUANTIZATION INFO {tens.name}")
                    else:
                        print(
                            f"    Input  {idx:02d} {tens.dtype!s:>10}"
                            f" min={q.min} max={q.max} scale={q.scale_f32!s} zero_point={q.zero_point} {tens.name}"
                        )
                else:
                    print(f"    Input  {idx:02d} {'-':>10} {tens}")
            for idx, tens in enumerate(op.outputs):
                q = tens.quantization
                if q is None:
                    print(f"    Output {idx:02d} {tens.dtype!s:>10} NO QUANTIZATION INFO {tens.name}")
                else:
                    print(
                        f"    Output {idx:02d} {tens.dtype!s:>10}"
                        f" min={q.min} max={q.max} scale={q.scale_f32!s} zero_point={q.zero_point} {tens.name}"
                    )
            print()

    def print_passes(self):
        print("print_passes()", self.name)
        for idx, ps in enumerate(self.passes):
            print("%03d %s" % (idx * 2, ps))

    def print_passes_with_tensors(self):
        print("print_passes_with_tensors()", self.name)
        for idx, ps in enumerate(self.passes):
            print("%3d %s" % (idx * 2, ps))
            for idx, tens in enumerate(ps.inputs):
                print(
                    "    Input        %2d %-15s %-15s %-15s     %s"
                    % (idx, tens.purpose.name, tens.mem_area.name, tens.format.name, tens.name)
                )
            for idx, tens in enumerate(ps.intermediates):
                print(
                    "    Intermediate %2d %-15s %-15s %-15s     %s"
                    % (idx, tens.purpose.name, tens.mem_area.name, tens.format.name, tens.name)
                )
            for idx, tens in enumerate(ps.outputs):
                print(
                    "    Output       %2d %-15s %-15s %-15s     %s"
                    % (idx, tens.purpose.name, tens.mem_area.name, tens.format.name, tens.name)
                )
            print()

    def print_cascaded_passes(self):
        print("print_cascaded_passes()", self.name)
        for idx, ps in enumerate(self.cascaded_passes):
            print("%3d %s SRAM used %.1f KB" % (idx * 2, ps, ps.sram_used / 1024))

    def print_cascaded_passes_with_tensors(self):
        print("print_cascaded_passes_with_tensors()", self.name)
        for idx, ps in enumerate(self.cascaded_passes):
            print("%3d %s SRAM used %.1f KB" % (idx * 2, ps, ps.sram_used / 1024))
            for idx, tens in enumerate(ps.inputs):
                print(
                    "    Input        %2d %-15s %-15s %-15s     %s"
                    % (idx, tens.purpose.name, tens.mem_area.name, tens.format.name, tens.name)
                )
            for idx, tens in enumerate(ps.intermediates):
                print(
                    "    Intermediate %2d %-15s %-15s %-15s     %s"
                    % (idx, tens.purpose.name, tens.mem_area.name, tens.format.name, tens.name)
                )
            for idx, tens in enumerate(ps.outputs):
                print(
                    "    Output       %2d %-15s %-15s %-15s     %s"
                    % (idx, tens.purpose.name, tens.mem_area.name, tens.format.name, tens.name)
                )
            print()

    def print_cascaded_passes_with_tensor_sizes(self):
        print("print_cascaded_passes_with_tensor_sizes()", self.name)
        for idx, ps in enumerate(self.cascaded_passes):
            print("%3d %s SRAM used %.1f KB" % (idx * 2, ps, ps.sram_used / 1024))
            for idx, tens in enumerate(ps.inputs):
                print(
                    "    Input        %2d %7.1f KB %-24s %-15s %-15s %-20s  %s"
                    % (
                        idx,
                        tens.storage_size() / 1024,
                        tens.storage_shape,
                        tens.mem_area.name,
                        tens.purpose.name,
                        tens.format.name,
                        tens.name,
                    )
                )
            for idx, tens in enumerate(ps.intermediates):
                print(
                    "    Intermediate %2d %7.1f KB %-24s %-15s %-15s %-20s  %s"
                    % (
                        idx,
                        tens.storage_size() / 1024,
                        tens.storage_shape,
                        tens.mem_area.name,
                        tens.purpose.name,
                        tens.format.name,
                        tens.name,
                    )
                )
            for idx, tens in enumerate(ps.outputs):
                print(
                    "    Output       %2d %7.1f KB %-24s %-15s %-15s %-20s  %s"
                    % (
                        idx,
                        tens.storage_size() / 1024,
                        tens.storage_shape,
                        tens.mem_area.name,
                        tens.purpose.name,
                        tens.format.name,
                        tens.name,
                    )
                )
            print()

    def print_high_level_command_stream(self):
        print("print_high_level_command_stream()", self.name)
        for idx, cmd in enumerate(self.high_level_command_stream):
            print("%3d %s" % (idx, cmd))


class Graph:
    def __init__(self, name="<unnamed>", batch_size=1):
        self.name = name
        self.batch_size = batch_size
        self.subgraphs = []
        self.metadata = []
        self.memory_used = {}
        self.total_original_weights = 0
        self.total_npu_encoded_weights = 0
        self.weight_cache = None  # See CompressedWeightCache
        self.bandwidths = 0
        self.macs = 0
        self.cycles = 0

    def get_root_subgraph(self):
        return self.subgraphs[0]

    def prune_startup_init_pass(self):
        for sg in self.subgraphs:
            sg.prune_startup_init_pass()

    def update_consumers(self):
        for sg in self.subgraphs:
            sg.update_consumers()

    def refresh_after_modification(self):
        for sg in self.subgraphs:
            sg.refresh_after_modification()

    def print_operators(self, ignore_placeholder_const=True, show_attributes=True):
        for sg in self.subgraphs:
            sg.print_operators(ignore_placeholder_const, show_attributes)

    def print_graph(self, label=None):
        for sg in self.subgraphs:
            sg.print_graph(label)

    def print_graph_with_tensors(self):
        for sg in self.subgraphs:
            sg.print_graph_with_tensors()

    def print_graph_with_tensor_quantization(self):
        for sg in self.subgraphs:
            sg.print_graph_with_tensor_quantization()

    def print_passes(self):
        for sg in self.subgraphs:
            sg.print_passes()

    def print_passes_with_tensors(self):
        for sg in self.subgraphs:
            sg.print_passes_with_tensors()

    def print_cascaded_passes(self):
        for sg in self.subgraphs:
            sg.print_cascaded_passes()

    def print_cascaded_passes_with_tensors(self):
        for sg in self.subgraphs:
            sg.print_cascaded_passes_with_tensors()

    def print_cascaded_passes_with_tensor_sizes(self):
        for sg in self.subgraphs:
            sg.print_cascaded_passes_with_tensor_sizes()

    def print_high_level_command_stream(self):
        for sg in self.subgraphs:
            sg.print_high_level_command_stream()
