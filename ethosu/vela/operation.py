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
# Internal representation of a Neural Network Operation.
import copy
from collections import namedtuple
from enum import Enum
from typing import Any
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple
from typing import TYPE_CHECKING

from .api import NpuRoundingMode
from .errors import VelaError
from .numeric_util import full_shape
from .shape4d import Shape4D


if TYPE_CHECKING:
    from .tensor import Tensor

PointXY = namedtuple("PointXY", "x y")
PointXYZ = namedtuple("PointXYZ", "x y z")


class NpuBlockType(Enum):
    Default = 0
    ConvolutionMxN = 1
    VectorProduct = 2
    Pooling = 3
    ConvolutionDepthWise = 4
    ElementWise = 5
    ReduceSum = 6


class Kernel:
    """
    Kernel information for NPU operations
    """

    def __init__(self, w: int, h: int, stride_x: int = 1, stride_y: int = 1, dilation_x: int = 1, dilation_y: int = 1):
        assert stride_x > 0 and stride_y > 0
        assert dilation_x > 0 and dilation_y > 0
        self.width = w
        self.height = h
        self.stride = PointXY(stride_x, stride_y)
        self.dilation = PointXY(dilation_x, dilation_y)

    def elements_wh(self) -> int:
        return self.width * self.height

    def area_width(self) -> int:
        return (self.width - 1) * self.dilation.x + 1

    def area_height(self) -> int:
        return (self.height - 1) * self.dilation.y + 1

    def dilated_wh(self) -> Tuple[int, int]:
        """Returns the dilated kernel width/height"""
        return self.dilation.x * (self.width - 1) + 1, self.dilation.y * (self.height - 1) + 1

    def __str__(self):
        return f"w={self.width}, h={self.height}, stride={tuple(self.stride)}, dilation={tuple(self.dilation)}"


# Classifies operators of type Custom
class CustomType(Enum):
    ThirdPartyOp = 0  # Third party custom op
    NpuOp = 1  # NPU op
    ExistingNpuOp = 2  # NPU op that was part of the input network


TensorIndices = namedtuple("TensorIndices", ["ifms", "weights", "biases"])

NO_INDICES = TensorIndices([], [], [])
IFM_INDICES = TensorIndices([0], [], [])
IFM_WEIGHTS_INDICES = TensorIndices([0], [1], [])
IFM_WEIGHTS_BIAS_INDICES = TensorIndices([0], [1], [2])
IFM_IFM2_INDICES = TensorIndices([0, 1], [], [])
CONV2D_BACKPROP_INDICES = TensorIndices([2], [1], [3])
TRANSPOSE_CONV_INDICES = TensorIndices([0], [1], [3])
CONCAT_INDICES = TensorIndices([1, 2], [], [])
SPLIT_IFM_INDICES = TensorIndices([1], [], [])
BLOCK_LSTM_INDICES = TensorIndices([3], [4], [])


# Static information related to operation codes
class OperatorInfo:
    __slots__ = ("id", "block_type", "indices", "is_unary")
    _id = 0

    def __init__(self, block_type=NpuBlockType.Default, indices=NO_INDICES, is_unary=False):
        OperatorInfo._id += 1
        self.id = OperatorInfo._id
        self.block_type = block_type
        self.indices = indices  # Indices of the different tensor purposes
        self.is_unary = is_unary  # Classifies elementwise operators


# Internally used operation codes
class Op(Enum):
    Abs = OperatorInfo(block_type=NpuBlockType.ElementWise, indices=IFM_INDICES, is_unary=True)
    Add = OperatorInfo(block_type=NpuBlockType.ElementWise, indices=IFM_IFM2_INDICES)
    AddN = OperatorInfo()
    Any = OperatorInfo()
    ArgMax = OperatorInfo()
    ArgMin = OperatorInfo()
    AvgPool = OperatorInfo(block_type=NpuBlockType.Pooling, indices=IFM_INDICES)
    BatchMatMul = OperatorInfo()
    BatchToSpaceND = OperatorInfo()
    BidirectionalSequenceLstm = OperatorInfo(block_type=NpuBlockType.VectorProduct, indices=IFM_WEIGHTS_INDICES)
    BidirectionalSequenceRnn = OperatorInfo(block_type=NpuBlockType.VectorProduct, indices=IFM_WEIGHTS_INDICES)
    BlockLSTM = OperatorInfo(block_type=NpuBlockType.VectorProduct, indices=BLOCK_LSTM_INDICES)

    CLZ = OperatorInfo(
        block_type=NpuBlockType.ElementWise, indices=IFM_INDICES, is_unary=True
    )  # NPU specific operation
    Call = OperatorInfo()
    Cast = OperatorInfo()
    Ceil = OperatorInfo()
    Clip = OperatorInfo()  # NPU specific fused activation function for clipping between activation.min/max
    Concat = OperatorInfo(indices=CONCAT_INDICES)
    ConcatEmbeddings = OperatorInfo()
    ConcatSliceWrite = OperatorInfo(indices=IFM_INDICES)
    ConcatTFLite = OperatorInfo(indices=CONCAT_INDICES)
    Const = OperatorInfo()  # Constant tensor, only used in CPU subgraphs
    Conv2D = OperatorInfo(block_type=NpuBlockType.ConvolutionMxN, indices=IFM_WEIGHTS_INDICES)
    Conv2DBackpropInput = OperatorInfo(block_type=NpuBlockType.ConvolutionMxN, indices=CONV2D_BACKPROP_INDICES)
    Conv2DBackpropInputSwitchedBias = OperatorInfo(
        block_type=NpuBlockType.ConvolutionMxN, indices=TRANSPOSE_CONV_INDICES
    )
    Conv2DBias = OperatorInfo(block_type=NpuBlockType.ConvolutionMxN, indices=IFM_WEIGHTS_BIAS_INDICES)
    Cos = OperatorInfo()
    Cumsum = OperatorInfo()
    Custom = OperatorInfo()  # Custom 3rd party operator, only used in CPU subgraphs
    CustomNpuOp = OperatorInfo()  # NPU custom operator, only used in CPU subgraphs
    DMA = OperatorInfo()
    Delegate = OperatorInfo()
    Densify = OperatorInfo()
    DepthToSpace = OperatorInfo()
    DepthwiseConv2DBias = OperatorInfo(block_type=NpuBlockType.ConvolutionDepthWise, indices=IFM_WEIGHTS_BIAS_INDICES)
    Dequantize = OperatorInfo(indices=IFM_INDICES)
    Div = OperatorInfo()
    Elu = OperatorInfo()
    EmbeddingLookup = OperatorInfo()
    EmbeddingLookupSparse = OperatorInfo()
    Equal = OperatorInfo()
    Exp = OperatorInfo()
    ExpandDims = OperatorInfo(indices=IFM_INDICES)
    FakeQuantWithMinMaxArgs = OperatorInfo()
    Fill = OperatorInfo()
    Floor = OperatorInfo()
    FloorDiv = OperatorInfo()
    FloorMod = OperatorInfo()
    FullyConnected = OperatorInfo(block_type=NpuBlockType.VectorProduct, indices=IFM_WEIGHTS_BIAS_INDICES)
    GatherNd = OperatorInfo()
    GatherV2 = OperatorInfo()
    Greater = OperatorInfo()
    GreaterEqual = OperatorInfo()
    HardSwish = OperatorInfo(indices=IFM_INDICES)
    HashtableLookup = OperatorInfo()
    Identity = OperatorInfo()
    If = OperatorInfo()
    L2Norm = OperatorInfo()
    L2Pool2D = OperatorInfo()
    LRN = OperatorInfo()
    LSHProjection = OperatorInfo()
    LeakyRelu = OperatorInfo(block_type=NpuBlockType.ElementWise, indices=IFM_INDICES, is_unary=True)
    Less = OperatorInfo()
    LessEqual = OperatorInfo()
    Log = OperatorInfo()
    LogSoftmax = OperatorInfo()
    LogicalAnd = OperatorInfo()
    LogicalNot = OperatorInfo()
    LogicalOr = OperatorInfo()
    Lstm = OperatorInfo(block_type=NpuBlockType.VectorProduct, indices=IFM_WEIGHTS_INDICES)
    LUT = OperatorInfo()  # NPU specific, operator has LUT, only used in fused activation functions
    MatMul = OperatorInfo(block_type=NpuBlockType.VectorProduct, indices=IFM_WEIGHTS_INDICES)
    MatrixDiag = OperatorInfo()
    MatrixSetDiag = OperatorInfo()
    Max = OperatorInfo()
    MaxPool = OperatorInfo(block_type=NpuBlockType.Pooling, indices=IFM_INDICES)
    Maximum = OperatorInfo(block_type=NpuBlockType.ElementWise, indices=IFM_IFM2_INDICES)
    Mean = OperatorInfo(indices=IFM_INDICES)
    Min = OperatorInfo()
    Minimum = OperatorInfo(block_type=NpuBlockType.ElementWise, indices=IFM_IFM2_INDICES)
    MirrorPad = OperatorInfo()
    Mul = OperatorInfo(block_type=NpuBlockType.ElementWise, indices=IFM_IFM2_INDICES)
    Neg = OperatorInfo()
    NonMaxSuppressionV4 = OperatorInfo()
    NonMaxSuppressionV5 = OperatorInfo()
    NotEqual = OperatorInfo()
    OneHot = OperatorInfo()
    Pack = OperatorInfo(indices=IFM_INDICES)
    PackReshaped = OperatorInfo(indices=IFM_INDICES)
    Pad = OperatorInfo(indices=IFM_INDICES)
    PadV2 = OperatorInfo()
    Placeholder = OperatorInfo()  # Only used in CPU subgraphs
    Pow = OperatorInfo()
    Prelu = OperatorInfo()
    Prod = OperatorInfo()
    Quantize = OperatorInfo(indices=IFM_INDICES)
    QuantizedAvgPool = OperatorInfo(block_type=NpuBlockType.Pooling, indices=IFM_INDICES)
    QuantizedConv2D = OperatorInfo(block_type=NpuBlockType.ConvolutionMxN, indices=IFM_WEIGHTS_INDICES)
    QuantizedMatMul = OperatorInfo(block_type=NpuBlockType.VectorProduct, indices=IFM_WEIGHTS_INDICES)
    QuantizedMaxPool = OperatorInfo(block_type=NpuBlockType.Pooling, indices=IFM_INDICES)
    QuantizedReshape = OperatorInfo(indices=IFM_INDICES)
    Range = OperatorInfo()
    Rank = OperatorInfo()
    ReduceSum = OperatorInfo(block_type=NpuBlockType.ReduceSum, indices=IFM_INDICES)
    Relu = OperatorInfo(indices=IFM_INDICES)
    Relu6 = OperatorInfo(indices=IFM_INDICES)
    ReluN1To1 = OperatorInfo(indices=IFM_INDICES)
    RescaleAdd = OperatorInfo(block_type=NpuBlockType.ElementWise, indices=IFM_IFM2_INDICES)
    Reshape = OperatorInfo(indices=IFM_INDICES)
    ResizeBilinear = OperatorInfo(block_type=NpuBlockType.Pooling, indices=IFM_INDICES)
    ResizeNearestNeighbor = OperatorInfo()
    ReverseSequence = OperatorInfo()
    ReverseV2 = OperatorInfo()
    Rnn = OperatorInfo(block_type=NpuBlockType.VectorProduct, indices=IFM_WEIGHTS_INDICES)
    Round = OperatorInfo()
    Rsqrt = OperatorInfo()
    SHL = OperatorInfo(block_type=NpuBlockType.ElementWise, indices=IFM_IFM2_INDICES)  # NPU specific operation
    SHR = OperatorInfo(block_type=NpuBlockType.ElementWise, indices=IFM_IFM2_INDICES)  # NPU specific operation
    ScatterNd = OperatorInfo()
    SegmentSum = OperatorInfo()
    Select = OperatorInfo()
    SelectV2 = OperatorInfo()
    Shape = OperatorInfo()
    Sigmoid = OperatorInfo(indices=IFM_INDICES)
    SignBit = OperatorInfo()
    Sin = OperatorInfo()
    SkipGram = OperatorInfo()
    Slice = OperatorInfo(indices=IFM_INDICES)
    Softmax = OperatorInfo(indices=IFM_INDICES)
    SpaceToBatchND = OperatorInfo()
    SpaceToDepth = OperatorInfo()
    SparseToDense = OperatorInfo()
    Split = OperatorInfo(indices=SPLIT_IFM_INDICES)
    SplitSliceRead = OperatorInfo(indices=IFM_INDICES)
    SplitV = OperatorInfo(indices=IFM_INDICES)
    Sqrt = OperatorInfo()
    Square = OperatorInfo()
    SquaredDifference = OperatorInfo()
    Squeeze = OperatorInfo(indices=IFM_INDICES)
    StridedSlice = OperatorInfo(indices=IFM_INDICES)
    Sub = OperatorInfo(block_type=NpuBlockType.ElementWise, indices=IFM_IFM2_INDICES)
    SubgraphInput = OperatorInfo()  # Only used in CPU subgraphs
    Sum = OperatorInfo()
    Svdf = OperatorInfo()
    Tanh = OperatorInfo(indices=IFM_INDICES)
    Tile = OperatorInfo()
    TopKV2 = OperatorInfo()
    Transpose = OperatorInfo()
    UnidirectionalSequenceLstm = OperatorInfo(block_type=NpuBlockType.VectorProduct, indices=IFM_WEIGHTS_INDICES)
    UnidirectionalSequenceRnn = OperatorInfo(block_type=NpuBlockType.VectorProduct, indices=IFM_WEIGHTS_INDICES)
    Unique = OperatorInfo()
    Unpack = OperatorInfo(indices=IFM_INDICES)
    UnpackReshaped = OperatorInfo(indices=IFM_INDICES)
    Where = OperatorInfo()
    While = OperatorInfo()
    ZerosLike = OperatorInfo()

    @property
    def info(self):
        return self.value

    @property
    def npu_block_type(self):
        return self.info.block_type

    def is_conv2d_op(self):
        return self.info.block_type == NpuBlockType.ConvolutionMxN

    def is_depthwise_conv2d_op(self):
        return self.info.block_type == NpuBlockType.ConvolutionDepthWise

    def is_pool_op(self):
        return self.info.block_type == NpuBlockType.Pooling

    def is_maxpool_op(self):
        return self in (Op.MaxPool, Op.QuantizedMaxPool)

    def is_avgpool_op(self):
        return self in (Op.QuantizedAvgPool, Op.AvgPool)

    def is_elementwise_op(self):
        return self.info.block_type == NpuBlockType.ElementWise

    def is_unary_elementwise_op(self):
        return self.info.block_type == NpuBlockType.ElementWise and self.info.is_unary

    def is_binary_elementwise_op(self):
        return self.info.block_type == NpuBlockType.ElementWise and not self.info.is_unary

    def is_relu_op(self):
        return self in (Op.Relu, Op.Relu6, Op.ReluN1To1, Op.Clip)

    def is_activation_op(self):
        return self.is_relu_op() or self in (Op.Tanh, Op.Sigmoid, Op.Softmax, Op.LUT, Op.HardSwish)

    def is_split_op(self):
        return self in (Op.Split, Op.SplitV, Op.StridedSlice, Op.Slice, Op.UnpackReshaped, Op.Unpack)

    def is_concat_op(self):
        return self in (Op.Concat, Op.ConcatTFLite, Op.PackReshaped, Op.Pack)

    def needs_bias(self):
        return bool(self.info.indices.biases)

    def needs_shapes(self):
        return bool(self.info.indices.ifms)

    @classmethod
    def op_set(cls, predicate):
        # Returns the set of all operator codes that fulfill the given predicate
        return {op_type for op_type in Op if predicate(op_type)}

    def __str__(self):
        return self.name

    __repr__ = __str__

    def __lt__(self, other):
        return self.value.id < other.value.id


class Padding(Enum):
    SAME = 0
    VALID = 1
    EXPLICIT = 2  # Padding is specified in a PAD operation (only used for NPU operations)


class ActivationFunction:
    """Fused activation function"""

    def __init__(self, op_type: Op):
        self.op_type = op_type  # The activation operation to be performed
        # min/max are optional; if present they are non-quantized values
        self.min: Optional[float] = None
        self.max: Optional[float] = None
        # Table lookup index, only applicable for Op.LUT activation, 0-7
        self.lut_index: int = 0

    def clone(self):
        res = copy.copy(self)
        return res


def create_activation_function(op_type: Op) -> ActivationFunction:
    """Creates activation function with min/max depending on op_type"""
    act = ActivationFunction(op_type)
    if op_type == Op.Relu:
        act.min = 0.0
    elif op_type == Op.Relu6:
        act.min = 0.0
        act.max = 6.0
    elif op_type == Op.ReluN1To1:
        act.min = -1.0
        act.max = 1.0
    elif op_type == Op.Tanh:
        act.min = -1.0
        act.max = 1.0
    elif op_type == Op.Sigmoid:
        act.min = 0.0
        act.max = 1.0
    elif op_type == Op.HardSwish:
        act.min = 0.0
    return act


def get_slice_offsets(input_shape: List[int], offset_tens: int, offset_mask: int, is_begin: bool = True):
    # For strided slice operator: get start or end offsets
    offsets = len(input_shape) * [0] if is_begin else input_shape[:]
    for idx in range(len(input_shape)):
        # If the i:th bit in the mask is set then the value on offset_tens[i] should be ignored
        if (offset_mask & (1 << idx)) == 0:
            offsets[idx] = offset_tens.values[idx]
            if offsets[idx] < 0:
                # Convert offset to positive value
                offsets[idx] += input_shape[idx]
    return offsets


class Operation:
    """Class representing a Neural Network operation. Has a name, a type,
    input and output tensors, as well as an attribute dictionary."""

    __slots__ = (
        "type",
        "name",
        "op_index",
        "attrs",
        "inputs",
        "outputs",
        "intermediates",
        "flops",
        "scheduled_pass",
        "run_on_npu",
        "activation",
        "memory_function",
        "forced_input_quantization",
        "forced_output_quantization",
        "activation_lut",
        "_kernel",
        "ifm_shapes",
        "ofm_shapes",
        "rescale",
        "read_offsets",
        "rounding_mode",
        "low_precision_scaling",
    )

    def __init__(self, op_type: Op, name: str):
        self.type = op_type
        self.name = name
        self.attrs: Dict[str, Any] = {}
        self.inputs: List[Tensor] = []
        self.outputs: List[Tensor] = []
        self.intermediates: List[Tensor] = []
        self.flops = 0
        self.run_on_npu = True
        # Fused activation function. If not none: operator code.
        self.activation: Optional[ActivationFunction] = None
        # Fused memory function, if not None: operator code
        self.memory_function = None
        # If not none: contains QuantizationParameters to be used as output quantization
        # (which overrides the ofm tensor's quantization), used in LUT
        self.forced_input_quantization = None
        self.forced_output_quantization = None
        self.scheduled_pass = None
        self.op_index = None  # input network operator index
        self.activation_lut = None
        self._kernel = None
        self.ifm_shapes: List[Shape4D] = []
        self.ofm_shapes: List[Shape4D] = []
        # If not none: contains rescale to be used as output scaling
        # (which overrides the ofm tensor's scale)
        self.rescale = None
        self.read_offsets: List[Shape4D] = [None, None]  # offset for [ifm, ifm2]
        self.rounding_mode: Optional[NpuRoundingMode] = None
        # The Mean operator (implemented as a depthwise convolution) requires scaling
        # to be calculated differently in one case. In that case, this is set to True.
        self.low_precision_scaling = False

    def clone(self, suffix="_clone"):
        res = Operation(self.type, self.name + suffix)

        res.attrs = dict(self.attrs)
        res.inputs = list(self.inputs)
        res.outputs = list(self.outputs)
        res.intermediates = list(self.intermediates)
        res.flops = self.flops
        res.run_on_npu = self.run_on_npu
        res.activation = None if self.activation is None else self.activation.clone()
        res.memory_function = self.memory_function
        res.forced_input_quantization = self.forced_input_quantization
        res.forced_output_quantization = self.forced_output_quantization
        res.scheduled_pass = self.scheduled_pass
        res.op_index = None  # not relevant as not part of input network
        res.read_offsets = list(self.read_offsets)
        res.rounding_mode = self.rounding_mode
        res.low_precision_scaling = self.low_precision_scaling

        return res

    def __str__(self):
        return "<nng.Operation '{}' type={}>".format(self.name, self.type)

    __repr__ = __str__

    def get_kernel_size(self):
        weights = self.weights
        if weights and self.type.npu_block_type in (NpuBlockType.ConvolutionDepthWise, NpuBlockType.ConvolutionMxN):
            weight_shape = full_shape(4, weights.shape, 1)
            h = weight_shape[-4]
            w = weight_shape[-3]
        elif self.type.npu_block_type in (NpuBlockType.Pooling, NpuBlockType.ReduceSum) and "ksize" in self.attrs:
            h, w = self.attrs["ksize"][1:3]
        else:
            h = self.attrs.get("filter_height", 1)
            w = self.attrs.get("filter_width", 1)
        return w, h

    def get_kernel_stride(self):
        if "strides" in self.attrs:
            _, h, w, _ = self.attrs["strides"]
        else:
            h = self.attrs.get("stride_h", 1)
            w = self.attrs.get("stride_w", 1)
        return w, h

    def get_kernel_dilation(self):
        if "dilation" in self.attrs:
            _, h, w, _ = self.attrs["dilation"]
        else:
            h = self.attrs.get("dilation_h_factor", 1)
            w = self.attrs.get("dilation_w_factor", 1)
        return w, h

    @property
    def kernel(self):
        k_w, k_h = self.get_kernel_size()
        s_w, s_h = self.get_kernel_stride()
        d_w, d_h = self.get_kernel_dilation()
        self._kernel = Kernel(k_w, k_h, s_w, s_h, d_w, d_h)
        return self._kernel

    def get_ifm_ifm2_weights_ofm(self):
        return self.ifm, self.ifm2, self.weights, self.ofm

    def get_ifm_weights_biases_ofm(self):
        return self.ifm, self.weights, self.bias, self.ofm

    def get_ifm_ifm2_weights_biases_ofm(self):
        return self.ifm, self.ifm2, self.weights, self.bias, self.ofm

    def get_ifm_ofm(self):
        return self.ifm, self.ofm

    @property
    def ifm(self):
        # Gets the IFM tensor, or None if not applicable
        return self.get_input(self.type.info.indices.ifms, 0)

    @property
    def ifm2(self):
        # Gets the IFM2 tensor, or None if not applicable
        return self.get_input(self.type.info.indices.ifms, 1)

    @property
    def bias(self):
        # Gets the bias tensor, or None if not applicable
        return self.get_input(self.type.info.indices.biases, 0)

    @property
    def weights(self):
        # Gets the weight tensor, or None if not applicable
        return self.get_input(self.type.info.indices.weights, 0)

    def get_ifm_tensors(self):
        # Gets the IFM tensors, or empty list if not applicable
        return self._index_list_to_tensors(self.type.info.indices.ifms)

    def get_weight_tensors(self):
        # Gets the weight tensors, or empty list if not applicable
        return self._index_list_to_tensors(self.type.info.indices.weights)

    def get_bias_tensors(self):
        # Gets the bias tensors, or empty list if not applicable
        return self._index_list_to_tensors(self.type.info.indices.biases)

    def _index_list_to_tensors(self, index_list):
        return [self.inputs[ix] for ix in index_list if ix < len(self.inputs)]

    def get_input(self, index_list, ix):
        if ix >= len(index_list):
            return None
        if index_list[ix] >= len(self.inputs):
            return None
        return self.inputs[index_list[ix]]

    @property
    def ofm(self):
        # Gets the OFM tensor, or None if not applicable
        return self.outputs[0] if self.outputs else None

    def get_concat_inputs_axis(self):
        assert self.type.is_concat_op()

        if self.type == Op.Concat:
            axis_tensor = self.inputs[0]
            inputs = self.inputs[1:]
        elif self.type == Op.ConcatTFLite:
            inputs = self.inputs
            axis = self.attrs["axis"]
        elif self.type == Op.PackReshaped:
            # Requires fixup_pack_input to be called before this point
            inputs = self.inputs
            axis = self.attrs["axis"]
            assert len(self.inputs) == self.attrs["values_count"]
        else:
            assert len(axis_tensor.ops) == 1 and axis_tensor.ops[0].type == Op.Const
            axis = int(axis_tensor.values)

        return inputs, axis

    def get_dilation_h_w(self):
        _, dilation_h, dilation_w, _ = self.attrs.get("dilation", (1, 1, 1, 1))
        return dilation_h, dilation_w

    def get_split_inputs_axis(self):
        assert self.type.is_split_op()

        offset_start = None
        offset_end = None
        axis = None
        if self.type == Op.Split:
            num_splits = self.attrs.get("num_splits")
            axis_tens = self.inputs[0]
            assert len(axis_tens.ops) == 1 and axis_tens.ops[0].type == Op.Const
            axis = int(axis_tens.values)
            input_tens = self.inputs[1]
            outputs = self.outputs
            assert num_splits == len(outputs)

        elif self.type == Op.SplitV:
            num_splits = self.attrs.get("num_splits")
            input_tens = self.inputs[0]
            size_tens = self.inputs[1]
            assert len(size_tens.ops) == 1 and size_tens.ops[0].type == Op.Const
            sizes = size_tens.values

            axis_tens = self.inputs[2]
            assert len(axis_tens.ops) == 1 and axis_tens.ops[0].type == Op.Const
            axis = int(axis_tens.values)

            for idx, size in enumerate(sizes):
                # One but only one size might be set to -1, indicating that size should be inferred
                if size == -1:
                    sizes[idx] = input_tens.shape[axis] - (sum(sizes) + 1)
                    break

            outputs = self.outputs
            assert num_splits == len(outputs)
            assert sum(sizes) == input_tens.shape[axis]

        elif self.type == Op.Slice:
            input_tens, begin_tens, size_tens = self.inputs
            outputs = self.outputs
            offset_start = [0] * len(input_tens.shape)
            offset_end = [0] * len(input_tens.shape)

            for idx in range(len(begin_tens.values)):
                # Check if the op should slice in dimension idx
                if size_tens.values[idx] != input_tens.shape[idx]:
                    offset_start[idx] = begin_tens.values[idx]
                    offset_end[idx] = size_tens.values[idx] + offset_start[idx]

        elif self.type == Op.StridedSlice:
            input_tens, begin_tens, end_tens, strides_tens = self.inputs
            outputs = self.outputs

            # Extract masks
            begin_mask = self.attrs["begin_mask"]
            ellipsis_mask = self.attrs["ellipsis_mask"]
            end_mask = self.attrs["end_mask"]
            new_axis_mask = self.attrs["new_axis_mask"]
            shrink_axis_mask = self.attrs["shrink_axis_mask"]

            # shrink_axis_mask/new_axis_mask/ellipsis_mask is not supported by the Operation class but the operation
            # may have the attribute modified and handled in the graph optimization phase.
            assert shrink_axis_mask == new_axis_mask == ellipsis_mask == 0
            offset_start = get_slice_offsets(input_tens.shape, begin_tens, begin_mask, is_begin=True)
            offset_end = get_slice_offsets(input_tens.shape, end_tens, end_mask, is_begin=False)
        elif self.type == Op.UnpackReshaped:
            # Requires fixup_unpack_output to be called before this point
            input_tens = self.inputs[0]
            outputs = self.outputs
            axis = self.attrs["axis"]
            num_splits = self.attrs["num"]
            # Number of outputs have to equal the value of the dimension to unpack
            assert num_splits == len(outputs) == input_tens.shape[axis]
        else:
            assert False

        return input_tens, outputs, axis, offset_start, offset_end

    def set_activation_lut(self, lut_tensor):
        self.activation = ActivationFunction(Op.LUT)
        self.activation_lut = lut_tensor
        self.add_input_tensor(lut_tensor)

    def add_input_tensor(self, tens):
        self.inputs.append(tens)
        if self not in tens.consumer_list:
            tens.consumer_list.append(self)

    def set_input_tensor(self, tens, idx):
        tens_to_remove = self.inputs[idx]
        if tens_to_remove in tens.consumer_list:
            tens.consumer_list.remove(tens_to_remove)

        self.inputs[idx] = tens
        if self not in tens.consumer_list:
            tens.consumer_list.append(self)

    def get_input_quantization(self):
        if self.forced_input_quantization is not None:
            return self.forced_input_quantization
        return self.ifm.quantization

    def set_output_tensor(self, tens):
        tens.ops = [self]
        self.outputs = [tens]

    def get_output_quantization(self):
        if self.forced_output_quantization is not None:
            return self.forced_output_quantization
        return self.ofm.quantization

    def error(self, msg):
        """
        Raises a VelaError exception for errors encountered when parsing an Operation

        :param self: Operation object that resulted in the error
        :param msg: str object that contains a description of the specific error encountered
        """

        def _print_tensors(tensors):
            lines = []
            for idx, tens in enumerate(tensors):
                tens_name = getattr(tens, "name", "Not a Tensor")
                lines.append(f"        {idx} = {tens_name}")
            return lines

        if self.op_index is None:
            lines = [f"Invalid {self.type} (name = {self.name}) operator in the internal representation. {msg}"]
        else:
            lines = [f"Invalid {self.type} (op_index = {self.op_index}) operator in the input network. {msg}"]

        lines += ["    Input tensors:"]
        lines += _print_tensors(self.inputs)

        lines += ["    Output tensors:"]
        lines += _print_tensors(self.outputs)

        raise VelaError("\n".join(lines))

    def set_ifm_ofm_shapes(self):
        self.ifm_shapes = []
        self.ofm_shapes = []

        ifm_tensor, ifm2_tensor, weight_tensor, ofm_tensor = self.get_ifm_ifm2_weights_ofm()

        # set all shapes to op, as 4D
        if self.type == Op.FullyConnected:
            if len(self.ifm.shape) == 2:
                self.ifm_shapes.append(Shape4D([self.ifm.shape[0], 1, 1, self.ifm.shape[1]]))
            else:
                # Special case, handled in graph optimization
                self.ifm_shapes.append(Shape4D(ifm_tensor.get_full_shape()))
            if len(self.ofm.shape) == 2:
                self.ofm_shapes.append(Shape4D([self.ofm.shape[0], 1, 1, self.ofm.shape[1]]))
            else:
                self.ofm_shapes.append(Shape4D(ofm_tensor.get_full_shape()))
        if self.type == Op.Softmax:
            self.ifm_shapes.append(Shape4D(ifm_tensor.get_full_shape()))
            self.ofm_shapes.append(Shape4D(ofm_tensor.get_full_shape()))
        elif self.type.is_split_op() or self.type.is_concat_op():
            for inp in self.inputs:
                if inp is not None:
                    self.ifm_shapes.append(Shape4D(full_shape(4, inp.shape, 1)))
                else:
                    self.ifm_shapes.append(None)
            for out in self.outputs:
                if out is not None:
                    self.ofm_shapes.append(Shape4D(full_shape(4, out.shape, 1)))
                else:
                    self.ofm_shapes.append(None)
        else:
            if ifm_tensor is not None:
                self.ifm_shapes.append(Shape4D(full_shape(4, ifm_tensor.shape, 1)))
            if ifm2_tensor is not None:
                self.ifm_shapes.append(Shape4D(full_shape(4, ifm2_tensor.shape, 1)))
            if ofm_tensor is not None:
                self.ofm_shapes.append(Shape4D(full_shape(4, ofm_tensor.shape, 1)))
