# Copyright (C) 2021 Arm Limited or its affiliates. All rights reserved.
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
# Early optimisation of the TOSA based network graph, using the rewrite_graph module to do the traversal of the graph.
from . import rewrite_graph
from .api import NpuRoundingMode
from .data_type import DataType
from .debug_database import DebugDatabase
from .graph_optimiser_util import calc_explicit_padding
from .graph_optimiser_util import needed_total_padding
from .graph_optimiser_util import set_ifm_ofm_op_shapes
from .graph_optimiser_util import set_tensor_equivalence
from .operation import ExplicitScaling
from .operation import NpuBlockType
from .operation import Op
from .operation_util import create_avgpool_nop


def replace_rescale_with_avg_pool(rescale_op):
    assert rescale_op.type == Op.Rescale

    avgpool_op = create_avgpool_nop(rescale_op.name + "_avgpool")
    rescale_op_clone = rescale_op.clone()
    op = rescale_op
    op.attrs = avgpool_op.attrs.copy()
    op.type = Op.AvgPool
    DebugDatabase.add_optimised(rescale_op_clone, op)

    return op


def calc_skirt(kernel, input_shape, explicit_padding):
    k_w, k_h = kernel.dilated_wh()
    s_x, s_y = kernel.stride
    ypad = needed_total_padding(int(input_shape.height), int(s_y), int(k_h))
    xpad = needed_total_padding(int(input_shape.width), int(s_x), int(k_w))

    top, left, bottom, right = explicit_padding
    top_pad, bottom_pad = calc_explicit_padding(int(input_shape.height), int(s_y), int(k_h), int(top), int(bottom))
    left_pad, right_pad = calc_explicit_padding(int(input_shape.width), int(s_x), int(k_w), int(left), int(right))

    padding = (top_pad, left_pad, bottom_pad, right_pad)
    skirt = (top_pad, left_pad, ypad - top_pad, xpad - left_pad)
    return padding, skirt


def add_padding_fields(op, arch, nng):
    if op.run_on_npu:
        if "explicit_padding" in op.attrs:
            input_shape = op.ifm_shapes[0]

            if op.type == Op.Conv2DBackpropInputSwitchedBias:
                # TODO not yet supported, but there will be need for separate handling
                assert False
            else:
                padding, skirt = calc_skirt(op.kernel, input_shape, op.attrs.get("explicit_padding"))

            op.attrs["explicit_padding"] = padding
            op.attrs["skirt"] = skirt

    return op


def rewrite_activation(op, arch, nng):
    if op.type not in (Op.ReluN, Op.Clamp):
        return op

    ifm = op.ifm
    prev_op = ifm.ops[0]

    # Note: the below checks on prev_op require that a first optimize pass on the full graph has been performed
    fuseable = (
        prev_op.run_on_npu
        and prev_op.type.npu_block_type != NpuBlockType.Default
        and len(ifm.ops) == 1
        and len(prev_op.outputs[0].consumers()) == 1
        and prev_op.activation is None
    )
    if not fuseable:
        print("Warning: relu like op will not be possible to fuse, currently not supported")
        assert False

    zp = ifm.quantization.zero_point if ifm.quantization.zero_point else 0
    if op.ofm.quantization.zero_point is None:
        op.ofm.quantization.zero_point = zp

    if op.type == Op.Clamp:
        op.attrs["min"] = op.attrs["min_int"] - zp
        op.attrs["max"] = op.attrs["max_int"] - zp
    elif op.type == Op.ReluN:
        op.attrs["max"] = op.attrs["max_int"] - zp
    else:
        print("Warning: Unknown TOSA activation Op")
        assert False

    return op


def rewrite_rescale(op, arch, nng):
    if op.type == Op.Rescale:
        ifm = op.ifm
        ofm = op.ofm

        # some error checking
        assert len(ifm.ops) == 1
        prev_op = ifm.ops[0]

        # TODO currently not supported
        assert len(ifm.consumer_list) == 1

        input_zp = op.attrs["input_zp"]
        output_zp = op.attrs["output_zp"]
        multiplier = op.attrs["multiplier"]
        shift = op.attrs["shift"]
        scale32 = op.attrs["scale32"]
        double_round = op.attrs["double_round"]
        per_channel = op.attrs["per_channel"]

        assert ifm.dtype in (DataType.uint8, DataType.int8, DataType.int32)
        assert ifm.dtype in (DataType.uint8, DataType.int8) or input_zp == 0
        assert ofm.dtype in (DataType.uint8, DataType.int8) or output_zp == 0
        assert (scale32 and ifm.dtype != DataType.int48) or (not scale32 and not double_round)

        # Check that input tensor has the same zp or no zp
        ifm_zp = ifm.quantization.zero_point
        if ifm_zp is not None and ifm_zp != input_zp:
            print("Error (fuse_rescale): zp of tensors producer/consumer differs unexpectedidly ")
            assert False
        ifm.quantization.zero_point = input_zp
        ofm.quantization.zero_point = output_zp
        for s, m in zip(shift, multiplier):
            # TODO these are the TOSA limitations
            assert m >= 0
            assert 2 <= s <= 62
            # TODO these are the HW limitations
            assert 0 <= s < (1 << 6)
        explicit_scaling = ExplicitScaling(per_channel, shift, multiplier)

        if double_round and scale32:
            rounding_mode = NpuRoundingMode.TFL
        else:
            rounding_mode = NpuRoundingMode.NATURAL

        if prev_op.type.is_depthwise_conv2d_op() or prev_op.type.is_conv2d_op() or prev_op.type == Op.FullyConnected:
            assert len(multiplier) == len(shift) == len(prev_op.bias.values)

            if ifm.dtype == DataType.int32 and per_channel:
                prev_op.explicit_scaling = explicit_scaling
                prev_op.rounding_mode = rounding_mode

                # Bypass op
                prev_op.set_output_tensor(ofm)
                DebugDatabase.add_optimised(op, prev_op)
                return op
            else:
                print("Warning, unsupported fusing of TOSA Rescale previous operator is of type:", prev_op.type)
                assert False
        # TODO which are the cases we need to and can do standalone Rescale?
        # TODO should we try to identify a conversion uint8<->int8 accomplished by 2 RESCALE ops?
        # origin might be TFLite op QUANTIZE, should we look to see if they can be translated to QUANTIZE?
        # limited to these at the moment:
        elif (
            (ifm.dtype == DataType.int8 and ofm.dtype == DataType.int8)
            or (ifm.dtype == DataType.uint8 and ofm.dtype == DataType.int8)
            or (ifm.dtype == DataType.int8 and ofm.dtype == DataType.uint8)
        ):
            # Create  NOP performing the RESCALE
            avgpool_op = replace_rescale_with_avg_pool(op)
            avgpool_op.rounding_mode = rounding_mode

            if per_channel:
                # TODO
                avgpool_op.explicit_scaling = explicit_scaling
                print("Warning, unsupported TOSA Rescale")
                assert False
            else:
                avgpool_op.explicit_scaling = explicit_scaling
        else:
            print("Warning, unsupported fusing of TOSA Rescale previous operator is of type:", prev_op.type)
            assert False
    return op


def fixup_quantization(op, arch, nng):
    if op.ifm and op.ifm.quantization.zero_point is None:
        op.ifm.quantization.zero_point = 0
    if op.ifm2 and op.ifm2.quantization.zero_point is None:
        op.ifm.quantization.zero_point = 0
    if op.ofm and op.ofm.quantization.zero_point is None:
        op.ofm.quantization.zero_point = 0
    return op


def supported_operator_check(op, arch, nng):
    op.run_on_npu = arch.tosa_supported_operators.is_operator_supported(op)
    return op


def tosa_optimise_graph(nng, arch):
    # Pre-processing step
    pre_process_list = [
        supported_operator_check,
        set_ifm_ofm_op_shapes,
    ]

    for idx, sg in enumerate(nng.subgraphs):
        nng.subgraphs[idx] = rewrite_graph.rewrite_graph_pre_order(
            nng, sg, arch, [], pre_process_list, rewrite_unsupported=False,
        )

    # Rewite Operators step
    op_rewrite_list = [set_tensor_equivalence, rewrite_rescale]

    for idx, sg in enumerate(nng.subgraphs):
        nng.subgraphs[idx] = rewrite_graph.rewrite_graph_pre_order(
            nng, sg, arch, [], op_rewrite_list, rewrite_unsupported=False,
        )

    # Post-processing step 1
    for idx, sg in enumerate(nng.subgraphs):
        nng.subgraphs[idx] = rewrite_graph.rewrite_graph_pre_order(
            nng, sg, arch, [], [rewrite_activation, add_padding_fields],
        )

    # Post-processing step 2
    for idx, sg in enumerate(nng.subgraphs):
        nng.subgraphs[idx] = rewrite_graph.rewrite_graph_pre_order(nng, sg, arch, [], [fixup_quantization],)

    return nng
