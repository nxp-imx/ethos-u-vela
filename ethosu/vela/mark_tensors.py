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
# Mark purpose and select formats for Tensors. Also compresses the weights.
from . import rewrite_graph
from . import weight_compressor
from .errors import OperatorError
from .operation import CustomType
from .operation import Op
from .tensor import MemType
from .tensor import TensorFormat
from .tensor import TensorPurpose


def purpose_from_list(lst):
    def purpose(op, idx):
        return lst[idx]

    return purpose


def all_fm(op, idx):
    return TensorPurpose.FeatureMap


def all_parameter(op, idx):
    return TensorPurpose.FeatureMap


def input0_from_output_rest_parameter(op, idx):
    if idx == 0:
        res = op.outputs[0].purpose
        if res == TensorPurpose.Unknown:
            print("Warning: Propagating unknown tensor purpose", op)
        return res
    return TensorPurpose.FeatureMap


def inputs_from_output(op, idx):
    res = op.outputs[0].purpose
    if res == TensorPurpose.Unknown:
        print("Warning: Propagating unknown tensor purpose", op)
    return res


tensor_purposes = [  # ops, input_purpose
    (
        set(
            (
                Op.Relu,
                Op.Relu6,
                Op.Rsqrt,
                Op.Abs,
                Op.Cast,
                Op.Exp,
                Op.Floor,
                Op.FloorDiv,
                Op.FloorMod,
                Op.SquaredDifference,
                Op.AddN,
                Op.Maximum,
                Op.Minimum,
                Op.Sigmoid,
                Op.Tanh,
                Op.AvgPool,
                Op.MaxPool,
                Op.Squeeze,
                Op.Softmax,
                Op.LRN,
                Op.BatchMatMul,
                Op.ZerosLike,
                Op.Mul,
                Op.Add,
                Op.Sub,
                Op.Div,
                Op.LeakyRelu,
                Op.CLZ,
                Op.SHL,
                Op.SHR,
                Op.ReduceSum,
            )
        ),
        all_fm,
    ),
    (
        set((Op.Conv2D, Op.MatMul, Op.Conv2DBias, Op.DepthwiseConv2DBias, Op.FullyConnected,)),
        purpose_from_list([TensorPurpose.FeatureMap, TensorPurpose.Weights, TensorPurpose.FeatureMap]),
    ),
    (
        set((Op.Conv2DBackpropInputSwitchedBias,)),
        purpose_from_list(
            [TensorPurpose.FeatureMap, TensorPurpose.Weights, TensorPurpose.FeatureMap, TensorPurpose.FeatureMap]
        ),
    ),
    (
        set((Op.QuantizedConv2D, Op.QuantizedMatMul)),
        purpose_from_list(
            [
                TensorPurpose.FeatureMap,
                TensorPurpose.Weights,
                TensorPurpose.FeatureMap,
                TensorPurpose.FeatureMap,
                TensorPurpose.FeatureMap,
                TensorPurpose.FeatureMap,
            ]
        ),
    ),
    (
        set(
            (
                Op.Reshape,
                Op.Min,
                Op.Max,
                Op.Mean,
                Op.Pad,
                Op.MirrorPad,
                Op.ArgMax,
                Op.ArgMin,
                Op.ExpandDims,
                Op.ResizeNearestNeighbor,
                Op.ResizeBilinear,
                Op.Tile,
                Op.Transpose,
            )
        ),
        purpose_from_list([TensorPurpose.FeatureMap, TensorPurpose.FeatureMap]),
    ),
    (
        set((Op.QuantizedReshape,)),
        purpose_from_list(
            [TensorPurpose.FeatureMap, TensorPurpose.FeatureMap, TensorPurpose.FeatureMap, TensorPurpose.FeatureMap]
        ),
    ),
    (
        set((Op.Dequantize, Op.Quantize, Op.QuantizedAvgPool, Op.QuantizedMaxPool, Op.Slice, Op.SplitV,)),
        purpose_from_list([TensorPurpose.FeatureMap, TensorPurpose.FeatureMap, TensorPurpose.FeatureMap]),
    ),
    (
        set((Op.BatchToSpaceND, Op.SpaceToBatchND, Op.DepthToSpace, Op.SpaceToDepth)),
        purpose_from_list([TensorPurpose.FeatureMap, TensorPurpose.FeatureMap, TensorPurpose.FeatureMap]),
    ),
    (
        set((Op.BlockLSTM,)),
        purpose_from_list(
            [
                TensorPurpose.FeatureMap,
                TensorPurpose.FeatureMap,
                TensorPurpose.FeatureMap,
                TensorPurpose.FeatureMap,
                TensorPurpose.Weights,
                TensorPurpose.FeatureMap,
                TensorPurpose.FeatureMap,
                TensorPurpose.FeatureMap,
                TensorPurpose.FeatureMap,
            ]
        ),
    ),
    (set((Op.SplitSliceRead,)), purpose_from_list([TensorPurpose.FeatureMap, TensorPurpose.FeatureMap])),
    (set((Op.Shape, Op.ConcatSliceWrite)), purpose_from_list([TensorPurpose.FeatureMap])),
    (
        set((Op.StridedSlice,)),
        purpose_from_list(
            [TensorPurpose.FeatureMap, TensorPurpose.FeatureMap, TensorPurpose.FeatureMap, TensorPurpose.FeatureMap]
        ),
    ),
    (set((Op.Fill, Op.Pack, Op.Range)), all_parameter),
    (set((Op.Placeholder, Op.SubgraphInput, Op.Const,)), purpose_from_list([])),
    (set((Op.FakeQuantWithMinMaxArgs,)), input0_from_output_rest_parameter),
    (set((Op.Square, Op.Sqrt, Op.Log, Op.Less, Op.Identity,)), inputs_from_output,),
    (None, all_fm),
]


for ops, input_purpose in tensor_purposes:
    if ops is None:
        continue


def mark_tensor_purpose(nng, arch, verbose_tensor_purpose=False):
    def mark_tensor_helper(tens, purpose):
        if tens.purpose == TensorPurpose.Unknown or tens.purpose == purpose:
            tens.purpose = purpose
        elif tens.purpose != TensorPurpose.LUT:
            assert 0, "Cannot resolve tensor purpose %s and %s for tensor %s" % (tens.purpose, purpose, tens)
        tens.mem_area = arch.tensor_storage_mem_area[tens.purpose]
        tens.mem_type = arch.tensor_storage_mem_type[tens.purpose]

        if len(tens.ops) == 1 and tens.ops[0].type == Op.Const:
            tens.mem_area = (
                arch.permanent_storage_mem_area
            )  # special case constants, as they must be in permanent storage
            tens.mem_type = MemType.Permanent_NPU

    def rewrite_mark_tensor_purpose(op, arch, nng):
        # find disconnected outputs and mark as parameters
        for tens in op.outputs:
            if not tens.consumers():
                mark_tensor_helper(tens, TensorPurpose.FeatureMap)

        for ops, input_purpose in tensor_purposes:
            if ops is None or op.type in ops:
                if ops is None:
                    print(
                        "Warning: Don't know how to mark up purpose for",
                        op.type,
                        op.inputs,
                        "triggering all feature map fallback",
                    )

                for idx, tens in enumerate(op.inputs):
                    if tens is None:
                        continue
                    purpose = input_purpose(op, idx) if tens.purpose == TensorPurpose.Unknown else tens.purpose
                    mark_tensor_helper(tens, purpose)

                if op.type == Op.Reshape:
                    # Reshape's input and output point to same data
                    op.outputs[0].mem_area = op.inputs[0].mem_area

                if op.type == Op.Custom and op.attrs.get("custom_type") == CustomType.ExistingNpuOp:
                    scratch_tensor = None

                    if len(op.inputs) >= 3:
                        scratch_tensor = op.inputs[2]  # should be existing scratch tensor
                        if scratch_tensor.name.endswith("_scratch"):
                            scratch_tensor.purpose = TensorPurpose.Scratch

                    if scratch_tensor is None:
                        OperatorError(op, "Scratch tensor not found.")

                break

        return op

    for sg in nng.subgraphs:
        sg = rewrite_graph.rewrite_graph_pre_order(nng, sg, arch, [], [rewrite_mark_tensor_purpose])
        for tens in sg.output_tensors:
            mark_tensor_helper(tens, TensorPurpose.FeatureMap)

    if verbose_tensor_purpose:
        nng.print_graph_with_tensors()

    return nng


def mark_tensor_format(nng, arch, verbose_tensor_format=False):
    formats_for_tensor = {}

    def init_tens(tens):
        if tens.purpose in (TensorPurpose.FeatureMap, TensorPurpose.LUT):
            fmt = arch.default_feature_map_format
        elif tens.purpose == TensorPurpose.Weights:
            fmt = arch.default_weight_format
        elif tens.purpose == TensorPurpose.Scratch:
            fmt = arch.default_feature_map_format
        elif tens.purpose == TensorPurpose.Unknown:
            fmt = TensorFormat.Unknown
        else:
            assert 0, "unknown tensor purpose %s" % (tens.purpose,)
        return fmt

    def visit_tens(tens, ps):
        if tens not in formats_for_tensor:
            fmt = init_tens(tens)
        else:
            fmt = formats_for_tensor[tens]

        formats_for_tensor[tens] = fmt

    for sg in nng.subgraphs:
        for ps in sg.passes:
            for tens in ps.outputs:
                visit_tens(tens, ps)
            for tens in ps.intermediates:
                visit_tens(tens, ps)
            for tens in ps.inputs:
                visit_tens(tens, ps)

    for tens, fmt in formats_for_tensor.items():
        if len(tens.shape) > 4:
            continue
        tens.set_format(fmt, arch)
        if fmt == TensorFormat.WeightsCompressed and tens.values is not None:
            src_tens = tens.get_dma_src_tensor()
            if src_tens is not None:
                op = tens.find_npu_op()
                if op is not None:
                    weight_compressor.compress_weights(
                        arch, nng, tens, op.type.npu_block_type, 16, 16, op.get_dilation_h_w()
                    )
                    # Alias compressed weights back into source tensor
                    src_tens.copy_compressed_weight_info(tens)

    if verbose_tensor_format:
        nng.print_passes_with_tensors()
