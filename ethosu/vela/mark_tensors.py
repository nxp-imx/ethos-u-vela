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
from .tensor import MemType
from .tensor import TensorFormat
from .tensor import TensorPurpose
from .tflite_mapping import custom_prefix


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
                "Relu",
                "Relu6",
                "Mul",
                "Add",
                "Sub",
                "Rsqrt",
                "Abs",
                "Cast",
                "Exp",
                "Floor",
                "FloorDiv",
                "FloorMod",
                "SquaredDifference",
                "AddN",
                "BiasAdd",
                "RealDiv",
                "Maximum",
                "Minimum",
                "Sigmoid",
                "Tanh",
                "FusedBatchNorm",
                "AvgPool",
                "MaxPool",
                "Squeeze",
                "Softmax",
                "LRN",
                "Assign",
                "BatchMatMul",
                "ZerosLike",
                "ExtractImagePatches",
                "MulAct",
                "AddAct",
                "SubAct",
                "DivAct",
                "AvgPoolAct",
                "MaxPoolAct",
                "LeakyRelu",
                "CLZ",
                "SHL",
                "SHR",
                "ReduceSum",
            )
        ),
        all_fm,
    ),
    (
        set(
            (
                "Conv2D",
                "DepthwiseConv2dNative",
                "MatMul",
                "Conv2DBiasAct",
                "DepthwiseConv2dBiasAct",
                "FullyConnectedAct",
            )
        ),
        purpose_from_list([TensorPurpose.FeatureMap, TensorPurpose.Weights, TensorPurpose.FeatureMap]),
    ),
    (
        set(("Conv2DBackpropInputSwitchedBias",)),
        purpose_from_list(
            [TensorPurpose.FeatureMap, TensorPurpose.Weights, TensorPurpose.FeatureMap, TensorPurpose.FeatureMap]
        ),
    ),
    (
        set(("QuantizedConv2D", "QuantizedMatMul")),
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
                "Reshape",
                "Min",
                "Max",
                "Mean",
                "Pad",
                "MirrorPad",
                "ArgMax",
                "ArgMin",
                "ExpandDims",
                "ResizeNearestNeighbor",
                "ResizeBilinear",
                "Tile",
                "Transpose",
                "Mfcc",
            )
        ),
        purpose_from_list([TensorPurpose.FeatureMap, TensorPurpose.FeatureMap]),
    ),
    (
        set(("QuantizedReshape", "QuantizedResizeBilinear")),
        purpose_from_list(
            [TensorPurpose.FeatureMap, TensorPurpose.FeatureMap, TensorPurpose.FeatureMap, TensorPurpose.FeatureMap]
        ),
    ),
    (
        set(("QuantizedBiasAdd", "QuantizedAdd", "QuantizedMul")),
        purpose_from_list(
            [
                TensorPurpose.FeatureMap,
                TensorPurpose.FeatureMap,
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
                "Dequantize",
                "Quantize",
                "QuantizeV2",
                "QuantizedRelu",
                "QuantizedRelu1",
                "QuantizedRelu6",
                "QuantizedAvgPool",
                "QuantizedMaxPool",
                "Slice",
                "SplitV",
            )
        ),
        purpose_from_list([TensorPurpose.FeatureMap, TensorPurpose.FeatureMap, TensorPurpose.FeatureMap]),
    ),
    (
        set(("BatchToSpaceND", "SpaceToBatchND", "DepthToSpaceND", "SpaceToDepthND")),
        purpose_from_list([TensorPurpose.FeatureMap, TensorPurpose.FeatureMap, TensorPurpose.FeatureMap]),
    ),
    (
        set(("BlockLSTM",)),
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
    (set(("SplitSliceRead",)), purpose_from_list([TensorPurpose.FeatureMap, TensorPurpose.FeatureMap])),
    (set(("Shape", "ConcatSliceWrite", "AudioSpectrogram")), purpose_from_list([TensorPurpose.FeatureMap])),
    (
        set(("StridedSlice",)),
        purpose_from_list(
            [TensorPurpose.FeatureMap, TensorPurpose.FeatureMap, TensorPurpose.FeatureMap, TensorPurpose.FeatureMap]
        ),
    ),
    (set(("Fill", "Pack", "Range")), all_parameter),
    (
        set(("Requantize",)),
        purpose_from_list(
            [
                TensorPurpose.FeatureMap,
                TensorPurpose.FeatureMap,
                TensorPurpose.FeatureMap,
                TensorPurpose.FeatureMap,
                TensorPurpose.FeatureMap,
            ]
        ),
    ),
    (set(("Placeholder", "SubgraphInput", "Const", "VariableV2")), purpose_from_list([])),
    (set(("FakeQuantWithMinMaxArgs", "FakeQuantWithMinMaxVars")), input0_from_output_rest_parameter),
    (
        set(("Square", "Sqrt", "Log", "Less", "Enter", "Exit", "Identity", "StopGradient", "Merge", "Switch")),
        inputs_from_output,
    ),
    (None, all_fm),
]


for ops, input_purpose in tensor_purposes:
    if ops is None:
        continue
    for op in ops:
        assert len(op) > 1, "string literal has been decomposed"


def mark_tensor_purpose(nng, arch, verbose_tensor_purpose=False):
    def mark_tensor_helper(tens, purpose):

        if tens.purpose == TensorPurpose.Unknown or tens.purpose == purpose:
            tens.purpose = purpose
        elif tens.purpose != TensorPurpose.LUT:
            assert 0, "Cannot resolve tensor purpose %s and %s for tensor %s" % (tens.purpose, purpose, tens)
        tens.mem_area = arch.tensor_storage_mem_area[tens.purpose]
        tens.mem_type = arch.tensor_storage_mem_type[tens.purpose]

        if len(tens.ops) == 1 and tens.ops[0].type == "Const":
            tens.mem_area = (
                arch.permanent_storage_mem_area
            )  # special case constants, as they must be in permanent storage
            tens.mem_type = MemType.Permanent_NPU

    def rewrite_mark_tensor_purpose(op, arch):
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
                    purpose = input_purpose(op, idx) if tens.purpose == TensorPurpose.Unknown else tens.purpose
                    mark_tensor_helper(tens, purpose)

                if op.type == "Reshape":
                    # Reshape's input and output point to same data
                    op.outputs[0].mem_area = op.inputs[0].mem_area

                if op.type.startswith(custom_prefix) and op.attrs.get("custom_type", "") == "ExistingNpuOp":
                    scratch_tensor = None

                    if len(op.inputs) >= 3:
                        scratch_tensor = op.inputs[2]  # should be existing scratch tensor
                        if scratch_tensor.name.endswith("_scratch"):
                            scratch_tensor.purpose = TensorPurpose.Scratch

                    if scratch_tensor is None:
                        raise OperatorError(op, "Scratch tensor not found.")

                break

        return op

    for sg in nng.subgraphs:
        sg = rewrite_graph.rewrite_graph_pre_order(sg, arch, [], [rewrite_mark_tensor_purpose])
        for tens in sg.output_tensors:
            mark_tensor_helper(tens, TensorPurpose.FeatureMap)

    if verbose_tensor_purpose:
        nng.print_graph_with_tensors()

    return nng


reshape_operations = set(
    (
        "Reshape",
        "QuantizedReshape",
        "ExpandDims",
        "Squeeze",
        "BatchToSpaceND",
        "SpaceToBatchND",
        "DepthToSpaceND",
        "SpaceToDepthND",
        "Placeholder",
    )
)


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
        tens.set_format(fmt, arch)
        if fmt == TensorFormat.WeightsCompressed and tens.values is not None:
            src_tens = tens.get_dma_src_tensor()
            if src_tens is not None:
                op = tens.find_npu_op()
                if op is not None:
                    npu_block_type = op.attrs["npu_block_type"]
                    weight_compressor.compress_weights(arch, nng, tens, npu_block_type, 16, 16, op.get_dilation_h_w())
                    # Alias compressed weights back into source tensor
                    src_tens.copy_compressed_weight_info(tens)

    if verbose_tensor_format:
        nng.print_passes_with_tensors()
