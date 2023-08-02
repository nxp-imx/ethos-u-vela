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
# Vela separates CPU operations and NPU operations into separate internal subgraphs. The CPU operations are left
# untouched in the final output.
#
# Vela does this by identifying NPU passes and pulling them out from the main CPU graph into separate subgraphs, invoked
# by NpuOp operations. Later, Vela generates command streams and compressed weight streams for the NPU subgraphs and
# attaches them to the NpuOp. This encapsulates everything the NPU subgraph is supposed to do.
import numpy as np

from .nn_graph import Pass
from .nn_graph import PassPlacement
from .nn_graph import Subgraph
from .operation import CustomType
from .operation import NpuBlockType
from .operation import Op
from .operation import Operation


def make_npu_call_op_pass(npu_subgraph):
    op = Operation(Op.CustomNpuOp, "call_" + npu_subgraph.name)
    op.attrs["subgraph"] = npu_subgraph
    op.attrs["custom_type"] = CustomType.NpuOp
    ps = Pass(op.name, PassPlacement.MemoryOnly, False, NpuBlockType.Default)
    ps.ops = [op]
    ps.primary_op = op
    op.scheduled_pass = ps

    # Inputs and outputs filled in later as we cut the graphs
    return ps


def switch_tensor_for_op(op, orig_tens, new_tens):

    op.inputs = [new_tens if tens == orig_tens else tens for tens in op.inputs]
    op.outputs = [new_tens if tens == orig_tens else tens for tens in op.outputs]

    ps = op.scheduled_pass
    if ps is None:
        return

    ps.inputs = [new_tens if tens == orig_tens else tens for tens in ps.inputs]
    ps.outputs = [new_tens if tens == orig_tens else tens for tens in ps.outputs]

    if ps.ifm_tensor == orig_tens:
        ps.ifm_tensor = new_tens
    if ps.ifm2_tensor == orig_tens:
        ps.ifm2_tensor = new_tens
    if ps.ofm_tensor == orig_tens:
        ps.ofm_tensor = new_tens
    if ps.weight_tensor == orig_tens:
        ps.weight_tensor = new_tens
    if ps.scale_tensor == orig_tens:
        ps.scale_tensor = new_tens


def rewrite_tensor_cpu_producer_npu_consumers(
    orig_tens, call_ps, startup_init_ps, npu_subgraph, cpu_subgraph, subgraph_for_pass
):
    is_const = orig_tens.ops[0].type == Op.Const
    new_tens = orig_tens.clone("_npu")

    op_type = Op.SubgraphInput
    if is_const:
        op_type = Op.Const
    op = Operation(op_type, orig_tens.name + "_input")
    op.scheduled_pass = startup_init_ps
    op.set_output_tensor(new_tens)
    startup_init_ps.ops.append(op)
    startup_init_ps.outputs.append(new_tens)

    if not is_const:
        call_ps.inputs.append(orig_tens)
        call_ps.primary_op.inputs.append(orig_tens)

    # Elementwise op can not overwrite ifm if input is used by many consumers
    if orig_tens in cpu_subgraph.input_tensors and len(orig_tens.consumers()) > 1:
        new_tens.ifm_write_protected = True

    # Elementwise op can not overwrite ifm if tensor is used as output from sub graph
    if orig_tens in cpu_subgraph.output_tensors:
        new_tens.ifm_write_protected = True

    for op in list(orig_tens.consumers()):
        if op is None:
            continue  # Subgraph consumers handled separately.
        ps = op.scheduled_pass
        if subgraph_for_pass[ps] == npu_subgraph:
            switch_tensor_for_op(op, orig_tens, new_tens)
            orig_tens.consumer_list.remove(op)
            new_tens.consumer_list.append(op)

    # Deal with output tensors for the NPU graph. These are special.
    npu_subgraph.output_tensors = [new_tens if tens == orig_tens else tens for tens in npu_subgraph.output_tensors]


def rewrite_tensor_npu_producer_cpu_consumers(
    orig_tens, call_ps, npu_subgraph, cpu_subgraph, subgraph_for_pass, multiple_npu_sg_have_same_cpu_out_tens
):
    if multiple_npu_sg_have_same_cpu_out_tens:
        new_tens = orig_tens
        orig_tens = orig_tens.src_tensor
    else:
        new_tens = orig_tens.clone("")
        orig_tens.name = orig_tens.name + "_cpu"
        new_tens.ops = []

    npu_subgraph.output_tensors.append(orig_tens)

    call_ps.outputs.append(new_tens)
    call_ps.primary_op.outputs.append(new_tens)
    new_tens.ops.append(call_ps.primary_op)
    # Elementwise op can not overwrite ifm if input is used by many consumers
    if orig_tens in npu_subgraph.input_tensors and len(orig_tens.consumers()) > 1:
        new_tens.ifm_write_protected = True

    # Elementwise op can not overwrite ifm if tensor is used as output from sub graph
    if orig_tens in npu_subgraph.output_tensors:
        new_tens.ifm_write_protected = True

    for op in list(orig_tens.consumers()):
        if op is None:
            continue  # Subgraph consumers handled separately.
        ps = op.scheduled_pass
        if subgraph_for_pass[ps] != npu_subgraph:
            switch_tensor_for_op(op, orig_tens, new_tens)
            orig_tens.consumer_list.remove(op)
            new_tens.consumer_list.append(op)

    # Deal with output tensors for the CPU graph. These are special.
    cpu_subgraph.output_tensors = [new_tens if tens == orig_tens else tens for tens in cpu_subgraph.output_tensors]


def extract_subgraph(nng, orig_sg, arch):
    assert orig_sg.placement == PassPlacement.Cpu

    passes = list(orig_sg.passes)
    place_vec = np.array([ps.placement for ps in passes])
    place_vec[
        place_vec == PassPlacement.StartupInit
    ] = PassPlacement.Cpu  # Keep the startup init pass on the CPU, we'll make new ones to move onto NPU.

    # MemoryOnly passes that are either squeezed between NPU passes or on the boundary of NPU and CPU
    # passes should be assigned to the NPU, unless they are assigned to run on CPU explicitly.

    # Forward, then backwards
    for is_reversed in range(2):
        last_place = PassPlacement.Cpu
        seq = enumerate(place_vec)
        if is_reversed:
            seq = reversed(list(seq))
        for idx, place in seq:
            if place == PassPlacement.MemoryOnly and passes[idx].ops[0].run_on_npu:
                if last_place == PassPlacement.Npu:
                    place = PassPlacement.Npu
                    place_vec[idx] = place

            if place != PassPlacement.MemoryOnly:
                last_place = place

    # Anything left, assign to the CPU.
    place_vec[place_vec == PassPlacement.MemoryOnly] = PassPlacement.Cpu

    if np.all(place_vec == PassPlacement.Cpu):
        return []  # Nothing to do

    # Create the subgraphs and split passes between them

    new_subgraphs = []
    split_count = 0
    subgraph_for_pass = {}
    orig_sg.passes = []
    call_pass = {}
    startup_init_passes = {}

    last_place = PassPlacement.Cpu
    curr_sg = orig_sg

    for idx, place in enumerate(place_vec):
        if place != last_place:
            if place == PassPlacement.Npu:
                split_count += 1
                curr_sg = Subgraph("%s_split_%d" % (orig_sg.name, split_count), PassPlacement.Npu)
                new_subgraphs.append(curr_sg)
                call_ps = make_npu_call_op_pass(curr_sg)
                subgraph_for_pass[call_ps] = orig_sg
                orig_sg.passes.append(call_ps)
                call_pass[curr_sg] = call_ps

                startup_init_ps = Pass(
                    curr_sg.name + "_startup_init", PassPlacement.StartupInit, False, NpuBlockType.Default
                )
                curr_sg.passes.append(startup_init_ps)
                startup_init_passes[curr_sg] = startup_init_ps
                subgraph_for_pass[startup_init_ps] = curr_sg

            else:
                curr_sg = orig_sg
            last_place = place
        ps = passes[idx]
        subgraph_for_pass[ps] = curr_sg
        curr_sg.passes.append(ps)

    # Rewrite tensors to fix up graphs.

    for curr_sg in new_subgraphs:
        for ps in curr_sg.passes:
            for tens in ps.inputs:
                source_sgs = [subgraph_for_pass[op.scheduled_pass] for op in tens.ops]
                assert len(source_sgs) >= 0
                producer_sg = source_sgs[0]
                for sg in source_sgs:
                    assert sg == producer_sg  # All need to be the same.

                if producer_sg != curr_sg:
                    assert (
                        producer_sg == orig_sg
                    )  # Because we go in-order, all the producers must be the original graph.
                    rewrite_tensor_cpu_producer_npu_consumers(
                        tens, call_pass[curr_sg], startup_init_passes[curr_sg], curr_sg, orig_sg, subgraph_for_pass
                    )

            for tens in ps.outputs:

                dest_sgs = [subgraph_for_pass[op.scheduled_pass] for op in tens.consumers() if op is not None]
                need_rewrite = False
                multiple_npu_sg_have_same_cpu_out_tens = False
                output_tensor = tens
                for sg in dest_sgs:
                    if sg != curr_sg:
                        need_rewrite = True
                        break
                for orig_out_tens in orig_sg.output_tensors:
                    if tens not in curr_sg.output_tensors:
                        if tens == orig_out_tens:
                            need_rewrite = True
                        elif tens.equivalence_id == orig_out_tens.equivalence_id:
                            need_rewrite = True
                            multiple_npu_sg_have_same_cpu_out_tens = True
                            output_tensor = orig_out_tens

                if need_rewrite:
                    rewrite_tensor_npu_producer_cpu_consumers(
                        output_tensor,
                        call_pass[curr_sg],
                        curr_sg,
                        orig_sg,
                        subgraph_for_pass,
                        multiple_npu_sg_have_same_cpu_out_tens,
                    )

        for tens in curr_sg.output_tensors:
            # ofm can depend on multiple ops. These ops can be divided into different NPU
            # nodes due to CPU nodes. If that is the case the ofm must be NHWC.
            tens.force_linear_format = True

    return new_subgraphs


def extract_npu_subgraphs(nng, arch):

    nng.refresh_after_modification()

    for sg in list(nng.subgraphs):
        if sg.placement == PassPlacement.Cpu:
            new_subgraphs = extract_subgraph(nng, sg, arch)
            nng.subgraphs += new_subgraphs

    nng.refresh_after_modification()
    nng.prune_startup_init_pass()

    for sg in nng.subgraphs:
        sg.build_pass_links()
