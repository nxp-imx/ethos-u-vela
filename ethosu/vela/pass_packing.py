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
# Packs a subgraph with Neural Network Operations into Passes. Each Pass has one or more Operations.
import collections
import enum

from .debug_database import DebugDatabase
from .nn_graph import Pass
from .nn_graph import PassPlacement
from .operation import NpuBlockType
from .operation import Op
from .operation_util import create_avgpool_nop
from .tensor import TensorPurpose


class PassFlags(enum.Flag):
    Empty = 0
    Main = 1
    Post = 2
    Mac = 4
    ElementWise = 8
    Npu = 16
    Cpu = 32
    StartupInit = 64
    MemoryOnly = 128
    PostFusingLimited = 256
    Memcpy = 512


mac_main_ops = set(
    (
        # convolutions
        Op.Conv2DBias,
        Op.Conv2D,
        Op.QuantizedConv2D,
        Op.Conv2DBackpropInputSwitchedBias,
        # depth-wise convolutions
        Op.DepthwiseConv2DBias,
        # FC layers
        Op.QuantizedMatMul,
        Op.MatMul,
        Op.FullyConnected,
        # pooling
        Op.QuantizedMaxPool,
        Op.QuantizedAvgPool,
        Op.AvgPool,
        Op.MaxPool,
        Op.ReduceSum,
    )
    # resize ops use pooling operations unless explicitly converted to other operations prior to pass packing
) | Op.op_set(Op.is_resize_op)

binary_elem_wise_main_ops = Op.op_set(Op.is_binary_elementwise_op)

unary_elem_wise_main_ops = Op.op_set(Op.is_unary_elementwise_op)

elem_wise_main_ops = binary_elem_wise_main_ops | unary_elem_wise_main_ops

activation_ops = Op.op_set(Op.is_relu_op)
npu_post_ops = activation_ops

npu_post_fuse_limited_ops = set(
    # Set of post operators that should not be fused with main/elementwise ops
    (Op.Sigmoid, Op.Tanh, Op.Quantize)
)

elem_wise_ops = elem_wise_main_ops | activation_ops | set((Op.Sigmoid, Op.Tanh))


quantization_ops = set((Op.Dequantize, Op.Max, Op.Min))
cpu_ops = set((Op.Softmax, Op.LRN, Op.Shape, Op.Pad, Op.AddN)) | quantization_ops

startup_init_ops = set((Op.Const, Op.Placeholder, Op.SubgraphInput))
memory_only_ops = set(
    (
        Op.Squeeze,
        Op.Reshape,
        Op.QuantizedReshape,
        Op.ExpandDims,
    )
)
memcpy_ops = set((Op.Memcpy,))


test_sequence = [
    (
        # ops_set
        npu_post_ops,
        # incompatible_pack_flags
        PassFlags.Cpu | PassFlags.MemoryOnly | PassFlags.Main,
        # flags_to_set
        PassFlags.Npu | PassFlags.Post,
        # flags_to_clear
        PassFlags.Empty,
    ),
    (
        # ops_set
        npu_post_fuse_limited_ops,
        # incompatible_pack_flags
        PassFlags.Cpu | PassFlags.MemoryOnly | PassFlags.Main | PassFlags.PostFusingLimited,
        # flags_to_set
        PassFlags.Npu | PassFlags.PostFusingLimited,
        # flags_to_clear
        PassFlags.Empty,
    ),
    (
        # ops_set
        mac_main_ops,
        # incompatible_pack_flags
        PassFlags.Cpu | PassFlags.MemoryOnly | PassFlags.ElementWise | PassFlags.Main | PassFlags.PostFusingLimited,
        # flags_to_set
        PassFlags.Npu | PassFlags.Mac | PassFlags.Main,
        # flags_to_clear
        PassFlags.Empty,
    ),
    (
        # ops_set
        elem_wise_main_ops,
        # incompatible_pack_flags
        PassFlags.Cpu | PassFlags.MemoryOnly | PassFlags.Mac | PassFlags.Main | PassFlags.PostFusingLimited,
        # flags_to_set
        PassFlags.Npu | PassFlags.ElementWise | PassFlags.Main,
        # flags_to_clear
        PassFlags.Empty,
    ),
    (
        # ops_set
        startup_init_ops,
        # incompatible_pack_flags
        PassFlags.Npu | PassFlags.Cpu | PassFlags.MemoryOnly,
        # flags_to_set
        PassFlags.StartupInit | PassFlags.Main,
        # flags_to_clear
        PassFlags.Empty,
    ),
    (
        # ops_set
        memory_only_ops,
        # incompatible_pack_flags
        PassFlags.Npu | PassFlags.Cpu,
        # flags_to_set
        PassFlags.MemoryOnly | PassFlags.Main,
        # flags_to_clear
        PassFlags.Empty,
    ),
    (
        # ops_set
        memcpy_ops,
        # incompatible_pack_flags
        PassFlags.Cpu | PassFlags.MemoryOnly | PassFlags.Mac | PassFlags.Main | PassFlags.PostFusingLimited,
        # flags_to_set
        PassFlags.Npu | PassFlags.Memcpy | PassFlags.Main,
        # flags_to_clear
        PassFlags.Empty,
    ),
    (
        # ops_set
        cpu_ops,
        # incompatible_pack_flags
        PassFlags.Npu | PassFlags.MemoryOnly | PassFlags.Main,
        # flags_to_set
        PassFlags.Cpu | PassFlags.Main,
        # flags_to_clear
        PassFlags.Empty,
    ),
    (  # This last one is a fallback for unrecognised operations
        # ops_set
        None,
        # incompatible_pack_flags
        PassFlags.Npu | PassFlags.MemoryOnly | PassFlags.Main,
        # flags_to_set
        PassFlags.Cpu | PassFlags.Main,
        # flags_to_clear
        PassFlags.Empty,
    ),
]

# Some sanity checking
for (operation_set, incompatible_pack_flags, flags_to_set, flags_to_clear) in test_sequence:
    assert not flags_to_clear & flags_to_set


def pack_into_passes(nng, arch, verbose_packing=False):
    def visit_op(op, ignored):
        visit_op_refcount[op] += 1

        if visit_op_refcount[op] == 1:  # First-time visit, go and fix up unused output tensors
            for tens in op.outputs:
                if len(tens.consumers()) == 0:
                    visit_op_refcount[op] += 1

        assert visit_op_refcount[op] <= len(op.outputs)
        if visit_op_refcount[op] == len(op.outputs):

            if op.type in startup_init_ops:
                startup_list.append(op)
            else:
                ofm_tensor = op.ofm
                if ofm_tensor is None:
                    ofm_tensor = op.outputs[0]
                ofm_shape = op.ofm_shapes[0] if op.run_on_npu else None

                build_pass((op,), ofm_tensor, ofm_shape)

    def build_pass(start_ops_to_process, ofm_tensor=None, ofm_shape=None):
        reverse_ops_list = []
        curr_flags = PassFlags.Empty
        npu_block_type = NpuBlockType.Default

        reverse_intermediates = []
        input_set = set()
        ifm_tensor = None
        primary_op = None
        ifm_shapes = None

        to_process = collections.deque()
        for start_op in start_ops_to_process:
            to_process.append((start_op, None))

        while to_process:
            curr_op, tens = to_process.popleft()

            if curr_op in reverse_ops_list:
                continue

            for operation_set, incompatible_pack_flags, flags_to_set, flags_to_clear in test_sequence:
                if operation_set is None or curr_op.type in operation_set:
                    if not (curr_flags & incompatible_pack_flags):
                        if flags_to_set & PassFlags.Npu:
                            if not curr_op.run_on_npu:
                                continue

                        reverse_ops_list.append(curr_op)
                        new_block_type = curr_op.type.npu_block_type
                        if new_block_type != NpuBlockType.Default:
                            assert npu_block_type == NpuBlockType.Default
                            npu_block_type = new_block_type  # Only one major block type per pass
                            assert primary_op is None
                            primary_op = curr_op

                        curr_flags &= ~flags_to_clear
                        curr_flags |= flags_to_set

                        if flags_to_set & PassFlags.Npu:
                            if flags_to_set & (
                                PassFlags.Mac
                                | PassFlags.ElementWise
                                | PassFlags.Post
                                | PassFlags.PostFusingLimited
                                | PassFlags.Memcpy
                            ):
                                assert len(curr_op.inputs) >= 1
                                ifm_tensor = curr_op.ifm
                                ifm_shapes = curr_op.ifm_shapes.copy()
                                assert ifm_tensor is not None, "IFM missing in {}".format(curr_op)
                                assert ifm_tensor.purpose == TensorPurpose.FeatureMap

                        if operation_set is None:
                            assert not curr_op.run_on_npu  # operator should have been placed on the CPU

                        for inp in reversed(curr_op.inputs):
                            if inp is None:
                                continue
                            if can_pack(inp, curr_op):
                                to_process.append((inp.ops[0], inp))
                            else:
                                input_set.add(inp)

                        break

            else:
                # This operation is not compatible with already packed operations, just register the tensor as an input
                assert tens is not None
                input_set.add(tens)

        if curr_flags & PassFlags.Npu and not curr_flags & (PassFlags.ElementWise | PassFlags.Mac):
            # Make the choice that if we don't have a mac operation, the ambidextrous operations go on the
            # element wise unit
            curr_flags |= PassFlags.ElementWise

        is_element_wise = True
        for op in reverse_ops_list:
            if op.type not in elem_wise_ops and op.type:
                is_element_wise = False
                break

        placement = PassPlacement.Unknown
        if curr_flags & PassFlags.Npu:
            assert placement == PassPlacement.Unknown
            placement = PassPlacement.Npu
        if curr_flags & PassFlags.Cpu:
            assert placement == PassPlacement.Unknown
            placement = PassPlacement.Cpu
        if curr_flags & PassFlags.MemoryOnly:
            assert placement == PassPlacement.Unknown
            placement = PassPlacement.MemoryOnly
        if curr_flags & PassFlags.StartupInit:
            assert placement == PassPlacement.Unknown
            placement = PassPlacement.StartupInit
        assert placement != PassPlacement.Unknown

        ops_list = list(reversed(reverse_ops_list))
        intermediates = list(reversed(reverse_intermediates))

        if primary_op is None:
            primary_op = create_primary_op(ops_list)
            if primary_op is not None:
                visit_tensor_refcount[primary_op.inputs[0]] += 1
                npu_block_type = primary_op.type.npu_block_type
                for input_tens in primary_op.inputs:
                    if input_tens not in input_set:
                        input_set.add(input_tens)

        ordered_input_list = []
        # Keep LUT-s in a separate list and add as inputs at the end
        # to avoid that they would accidentally be assigned as ifm or ifm2
        lut_list = []
        input_refcounts = collections.defaultdict(int)
        input_ops_list = ops_list.copy()

        # Check primary_op first
        if primary_op is not None:
            for inp in primary_op.inputs:
                if inp is None:
                    continue
                add_input_list(inp, input_set, input_refcounts, lut_list, ordered_input_list)
            input_ops_list.remove(primary_op)

        # Check rest of the list
        for op in input_ops_list:
            for inp in op.inputs:
                add_input_list(inp, input_set, input_refcounts, lut_list, ordered_input_list)

        name = ops_list[0].name
        ps = Pass(name, placement, is_element_wise, npu_block_type)
        ps.ops = ops_list
        ps.primary_op = primary_op
        ps.inputs = ordered_input_list
        ps.intermediates = intermediates
        ps.outputs = list(ops_list[-1].outputs)

        # ElementWise operation, 2 IFMs
        if ps.primary_op and ps.primary_op.type in binary_elem_wise_main_ops:
            ps.ifm_tensor = ps.inputs[0]
            ps.ifm2_tensor = ps.inputs[-1]

            if len(ps.inputs) > 2:
                ps.ifm_tensor = ps.inputs[-2]

            # Get the corresponding ifm_shapes
            for op in input_ops_list + [primary_op]:
                if op.run_on_npu:
                    if ps.ifm_tensor == op.ifm:
                        ps.ifm_shapes.append(op.ifm_shapes[0])
                    elif ps.ifm_tensor == op.ifm2:
                        ps.ifm_shapes.append(op.ifm_shapes[1])

                    if ps.ifm2_tensor == op.ifm:
                        ps.ifm_shapes.append(op.ifm_shapes[0])
                    elif ps.ifm2_tensor == op.ifm2:
                        ps.ifm_shapes.append(op.ifm_shapes[1])
        else:
            ps.ifm_tensor = ifm_tensor
            ps.ifm2_tensor = None
            if ps.primary_op is not None and ps.primary_op.run_on_npu:
                ps.ifm_shapes.append(ifm_shapes[0])

        ps.ofm_tensor = ofm_tensor
        ps.ofm_shapes.append(ofm_shape)

        assert ps.placement != PassPlacement.Npu or ps.ofm_tensor is not None
        ps.weight_tensor = ps.get_primary_op_ifm_weights()[1]
        ps.scale_tensor = ps.get_primary_op_ifm_weights_biases_ofm()[2]
        ps.lut_tensor = ps.get_primary_op_lut()
        ps.inputs.extend(lut_list)

        for op in ps.ops:
            op.scheduled_pass = ps

        reverse_pass_list.append(ps)

        for inp, refcount in input_refcounts.items():
            for _ in range(refcount):
                visit_tensor(inp)

        return ps

    def visit_tensor(tens):
        visit_tensor_refcount[tens] += 1
        assert visit_tensor_refcount[tens] <= len(tens.consumers())
        if visit_tensor_refcount[tens] == len(tens.consumers()):
            for op in reversed(tens.ops):
                visit_op(op, tens)

    def create_primary_op(op_list):
        if any(op.type in (npu_post_ops | npu_post_fuse_limited_ops) and op.run_on_npu for op in op_list):
            # Configure a 1x1 AvgPool and attach the op onto it
            op = op_list[0]
            inp = op.inputs[0]
            avgpool_op = create_avgpool_nop(op.name + "_avgpool")
            avgpool_op.add_input_tensor(inp)
            avgpool_out = inp.clone("_avgpooled")
            avgpool_out.consumer_list.append(op)
            avgpool_op.set_output_tensor(avgpool_out)
            avgpool_op.ifm_shapes = op.ifm_shapes.copy()
            avgpool_op.ofm_shapes = op.ofm_shapes.copy()
            avgpool_op.read_offsets = op.read_offsets.copy()
            avgpool_op.read_shapes = op.read_shapes.copy()

            op.inputs[0] = avgpool_out
            op_list.insert(0, avgpool_op)

            DebugDatabase.add_optimised(op, avgpool_op)
            return avgpool_op

        return None

    def can_pack(inp, curr_op):
        if len(inp.ops) == 1:
            next_op = inp.ops[0]
            for outp in next_op.outputs:
                consumers = outp.consumers()
                if len(consumers) > 1 or (len(consumers) == 1 and consumers[0] != curr_op):
                    return False

            # There cannot be any reshaping between next_op ofm and corresponding curr_op ifm
            if len(curr_op.ifm_shapes) != 0 and len(next_op.ofm_shapes) != 0:
                if inp == curr_op.ifm and next_op.ofm_shapes[0] != curr_op.ifm_shapes[0]:
                    return False
                elif (
                    curr_op.ifm2 is not None and inp == curr_op.ifm2 and next_op.ofm_shapes[0] != curr_op.ifm_shapes[1]
                ):
                    return False
        else:
            return False

        return True

    def add_input_list(inp_to_add, inp_set, inp_refcnts, lut_list, ordered_inp_list):
        if inp_to_add in inp_set:
            if inp_refcnts[inp_to_add] == 0:
                if inp_to_add.purpose == TensorPurpose.LUT:
                    lut_list.append(inp_to_add)
                else:
                    ordered_inp_list.append(inp_to_add)
            inp_refcnts[inp_to_add] += 1

    for sg in nng.subgraphs:
        reverse_pass_list = []
        visit_op_refcount = collections.defaultdict(int)
        visit_tensor_refcount = collections.defaultdict(int)

        startup_list = []

        for tens in sg.output_tensors:
            visit_tensor(tens)

        if startup_list:
            startup_ps = build_pass(startup_list)
            startup_ps.outputs = [op.outputs[0] for op in startup_list]  # Need to fixup the outputs
            startup_ps.name = "startup_weight_initialisation"

        # Graphs with both CPU and NPU ops might not have an optimal order in
        # the pass list due to how the graph is traversed (depth first search).
        # This can result in more context switching between CPU and NPU.
        # Try to optmize this by moving/grouping CPU ops where that is possible.
        # Criteria for CPU pass to be moved:
        #
        # 1) CPU passes that only depends on sg.input_tensors can be
        #    moved to the top of the list.
        #    ResourceVariables ops like VarHandle, ReadVariable, CallOnce
        #    can also be moved to the top of list.
        #
        # 2) A CPU pass X is allowed to be grouped together with CPU pass Y
        #    if there is no NPU pass between pass X and pass Y that depends
        #    on output from pass X or a MemoryOnly pass.
        #
        # Criteria 2 will try to move as many CPU passes towards the bottom of
        # the list.

        pass_list_top = []
        pass_list = []

        # Filter out early passes from the rest
        for ps in list(reversed(reverse_pass_list)):
            if startup_ps == ps:
                # startup pass belongs in the top
                pass_list_top.insert(0, ps)
                continue

            ifm2 = ps.ops[0].ifm2
            if ifm2 is None:
                # Dynamic weights must be treated as ifm's.
                if ps.ops[0].type == Op.FullyConnected and ps.ops[0].weights.purpose == TensorPurpose.FeatureMap:
                    # Op has dynamic weights, include this in the check below
                    ifm2 = ps.ops[0].weights

            if ps.placement == PassPlacement.Cpu and (
                ps.ops[0].ifm in sg.input_tensors
                and (ifm2 in sg.input_tensors or ifm2 is None)
                or (ps.ops[0].type in (Op.VarHandle, Op.ReadVariable, Op.CallOnce))
            ):
                # This CPU pass only depends on sg.input_tensors or resource variable
                pass_list_top.append(ps)
            else:
                # Add pass to the list that will be sorted in the next step
                pass_list.append(ps)

        # Sort ops by op_index (same call order as in the original graph)
        pass_list_top = sorted(pass_list_top, key=lambda ps: -1 if ps.ops[0].op_index is None else ps.ops[0].op_index)

        # Sort the rest of the list based on critera 2.
        # Search from bottom of list and when a CPU pass is found
        # search forward in the list and see if it is possible to join another CPU pass.
        last_idx = len(pass_list) - 1
        for cpu_ps in reversed(pass_list):
            if cpu_ps.placement != PassPlacement.Cpu:
                continue
            # CPU pass found, search forward and move pass if possible
            idx = pass_list.index(cpu_ps)
            for next_ps in pass_list[idx + 1 :]:
                if next_ps.placement == PassPlacement.Cpu:
                    # It is possible to move the CPU pass
                    pass_list.remove(cpu_ps)
                    insert_index = pass_list.index(next_ps)
                    pass_list.insert(insert_index, cpu_ps)
                    break

                # Check all outputs from the cpu pass
                if (
                    any(ofm in [next_ps.ops[0].ifm, next_ps.ops[0].ifm2] for ofm in cpu_ps.ops[0].outputs)
                    or next_ps.placement == PassPlacement.MemoryOnly
                ):
                    # Not possible to move since next pass depends on the output from the cpu pass
                    break

                if pass_list.index(next_ps) == last_idx:
                    # Last element, ok to move the CPU pass
                    pass_list.remove(cpu_ps)
                    pass_list.append(cpu_ps)
                    break

        pass_list_top.extend(pass_list)

        sg.passes = pass_list_top
        sg.build_pass_links()

    if verbose_packing:
        nng.print_passes()

    return nng
