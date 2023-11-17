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
# Internal representation of a Neural Network Operation.
# For Class name forward references for the type annotations. (see PEP 563).
from __future__ import annotations

import copy
from collections import namedtuple
from enum import auto
from enum import Enum
from typing import Any
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple
from typing import TYPE_CHECKING

from .errors import VelaError
from .ethos_u55_regs.ethos_u55_regs import resampling_mode
from .numeric_util import full_shape
from .shape4d import Shape4D

# Import needed for Type annotations. Only import for Type checking to avoid run-time errors due to cyclic import.
if TYPE_CHECKING:
    from .tensor import QuantizationParameters
    from .tensor import Tensor

PointXY = namedtuple("PointXY", "x y")
PointXYZ = namedtuple("PointXYZ", "x y z")


class RoundingMode(Enum):
    TFLite = auto()  # Round like TensorFlow Lite
    ToZero = auto()  # Round towards zero (truncate)
    HalfUp = auto()  # Round to nearest with x.5 rounded up towards positive infinity (natural)
    AwayZero = auto()  # Round away from zero (towards infinity)


class NpuBlockType(Enum):
    Default = 0
    ConvolutionMxN = 1
    VectorProduct = 2
    Pooling = 3
    ConvolutionDepthWise = 4
    ElementWise = 5
    ReduceSum = 6
    Dma = 7


class Kernel:
    """
    Kernel information for NPU operations
    """

    def __init__(
        self,
        w: int,
        h: int,
        stride_x: int = 1,
        stride_y: int = 1,
        dilation_x: int = 1,
        dilation_y: int = 1,
        valid_padding=False,
    ):
        assert stride_x > 0 and stride_y > 0
        assert dilation_x > 0 and dilation_y > 0
        self.width = w
        self.height = h
        self.stride = PointXY(stride_x, stride_y)
        self.dilation = PointXY(dilation_x, dilation_y)
        self.valid_padding = valid_padding

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

NNG_NO_INDICES = TensorIndices([], [], [])
NNG_IFM_INDICES = TensorIndices([0], [], [])
NNG_IFM_WEIGHTS_INDICES = TensorIndices([0], [1], [])
NNG_IFM_WEIGHTS_BIAS_INDICES = TensorIndices([0], [1], [2])
NNG_IFM_IFM2_INDICES = TensorIndices([0, 1], [], [])
NNG_CONV2D_BACKPROP_INDICES = TensorIndices([2], [1], [3])
NNG_TRANSPOSE_CONV_INDICES = TensorIndices([0], [1], [3])
NNG_CONCAT_INDICES = TensorIndices([1, 2], [], [])
NNG_SPLIT_IFM_INDICES = TensorIndices([1], [], [])
NNG_BLOCK_LSTM_INDICES = TensorIndices([3], [4], [])


# Static information related to operation codes
class OperatorInfo:
    __slots__ = ("id", "block_type", "indices", "is_unary")
    _id = 0

    def __init__(self, block_type=NpuBlockType.Default, indices=NNG_NO_INDICES, is_unary=False):
        OperatorInfo._id += 1
        self.id = OperatorInfo._id
        self.block_type = block_type
        self.indices = indices  # Indices of the different tensor purposes
        self.is_unary = is_unary  # Classifies elementwise operators


# Internally used operation codes
class Op(Enum):
    Abs = OperatorInfo(block_type=NpuBlockType.ElementWise, indices=NNG_IFM_INDICES, is_unary=True)
    Add = OperatorInfo(block_type=NpuBlockType.ElementWise, indices=NNG_IFM_IFM2_INDICES)
    AddN = OperatorInfo()
    Any = OperatorInfo()
    ArgMax = OperatorInfo(indices=NNG_IFM_INDICES)
    ArgMin = OperatorInfo()
    AvgPool = OperatorInfo(block_type=NpuBlockType.Pooling, indices=NNG_IFM_INDICES)
    Atan2 = OperatorInfo(indices=NNG_IFM_IFM2_INDICES)
    BatchMatMul = OperatorInfo()
    BatchToSpaceND = OperatorInfo()
    BidirectionalSequenceLstm = OperatorInfo(indices=NNG_IFM_WEIGHTS_INDICES)
    BidirectionalSequenceRnn = OperatorInfo(indices=NNG_IFM_WEIGHTS_INDICES)
    CLZ = OperatorInfo(
        block_type=NpuBlockType.ElementWise, indices=NNG_IFM_INDICES, is_unary=True
    )  # NPU specific operation
    Call = OperatorInfo()
    Cast = OperatorInfo()
    Ceil = OperatorInfo()
    Clamp = OperatorInfo(indices=NNG_IFM_INDICES)  # TOSA specific
    Clip = OperatorInfo()  # NPU specific fused activation function for clipping between activation.min/max
    Concat = OperatorInfo(indices=NNG_CONCAT_INDICES)
    ConcatEmbeddings = OperatorInfo()
    ConcatSliceWrite = OperatorInfo(indices=NNG_IFM_INDICES)
    ConcatTFLite = OperatorInfo(indices=NNG_CONCAT_INDICES)
    Const = OperatorInfo()  # Constant tensor, only used in CPU subgraphs
    Conv2D = OperatorInfo(block_type=NpuBlockType.ConvolutionMxN, indices=NNG_IFM_WEIGHTS_INDICES)
    Conv2DBackpropInput = OperatorInfo(block_type=NpuBlockType.ConvolutionMxN, indices=NNG_CONV2D_BACKPROP_INDICES)
    Conv2DBackpropInputSwitchedBias = OperatorInfo(
        block_type=NpuBlockType.ConvolutionMxN, indices=NNG_TRANSPOSE_CONV_INDICES
    )
    Conv2DBias = OperatorInfo(block_type=NpuBlockType.ConvolutionMxN, indices=NNG_IFM_WEIGHTS_BIAS_INDICES)
    Cos = OperatorInfo()
    Cumsum = OperatorInfo()
    Custom = OperatorInfo()  # Custom 3rd party operator, only used in CPU subgraphs
    CustomNpuOp = OperatorInfo()  # NPU custom operator, only used in CPU subgraphs
    Delegate = OperatorInfo()
    Densify = OperatorInfo()
    DepthToSpace = OperatorInfo()
    DepthwiseConv2DBias = OperatorInfo(
        block_type=NpuBlockType.ConvolutionDepthWise, indices=NNG_IFM_WEIGHTS_BIAS_INDICES
    )
    Dequantize = OperatorInfo(indices=NNG_IFM_INDICES)
    Div = OperatorInfo()
    Memcpy = OperatorInfo(block_type=NpuBlockType.Dma, indices=NNG_IFM_INDICES)
    Elu = OperatorInfo()
    EmbeddingLookup = OperatorInfo()
    EmbeddingLookupSparse = OperatorInfo()
    Equal = OperatorInfo()
    Exp = OperatorInfo(block_type=NpuBlockType.ElementWise, indices=NNG_IFM_INDICES, is_unary=True)
    ExpandDims = OperatorInfo(indices=NNG_IFM_INDICES)
    FakeQuantWithMinMaxArgs = OperatorInfo()
    Fill = OperatorInfo()
    Floor = OperatorInfo()
    FloorDiv = OperatorInfo()
    FloorMod = OperatorInfo()
    FullyConnected = OperatorInfo(block_type=NpuBlockType.VectorProduct, indices=NNG_IFM_WEIGHTS_BIAS_INDICES)
    GatherNd = OperatorInfo()
    GatherV2 = OperatorInfo()
    Greater = OperatorInfo()
    GreaterEqual = OperatorInfo()
    HardSwish = OperatorInfo(indices=NNG_IFM_INDICES)
    HashtableLookup = OperatorInfo()
    Identity = OperatorInfo(indices=NNG_IFM_INDICES)
    If = OperatorInfo()
    L2Norm = OperatorInfo()
    L2Pool2D = OperatorInfo()
    LRN = OperatorInfo()
    LSHProjection = OperatorInfo()
    LeakyRelu = OperatorInfo(block_type=NpuBlockType.ElementWise, indices=NNG_IFM_INDICES, is_unary=True)
    Less = OperatorInfo()
    LessEqual = OperatorInfo()
    Log = OperatorInfo(block_type=NpuBlockType.ElementWise, indices=NNG_IFM_INDICES, is_unary=True)
    LogSoftmax = OperatorInfo()
    LogicalAnd = OperatorInfo()
    LogicalNot = OperatorInfo()
    LogicalOr = OperatorInfo()
    Lstm = OperatorInfo(indices=NNG_IFM_WEIGHTS_INDICES)
    LUT = OperatorInfo()  # NPU specific, operator has LUT, only used in fused activation functions
    MatMul = OperatorInfo(indices=NNG_IFM_WEIGHTS_INDICES)
    MatrixDiag = OperatorInfo()
    MatrixSetDiag = OperatorInfo()
    Max = OperatorInfo()
    MaxPool = OperatorInfo(block_type=NpuBlockType.Pooling, indices=NNG_IFM_INDICES)
    Maximum = OperatorInfo(block_type=NpuBlockType.ElementWise, indices=NNG_IFM_IFM2_INDICES)
    Mean = OperatorInfo(indices=NNG_IFM_INDICES)
    Min = OperatorInfo()
    Minimum = OperatorInfo(block_type=NpuBlockType.ElementWise, indices=NNG_IFM_IFM2_INDICES)
    MirrorPad = OperatorInfo()
    Mul = OperatorInfo(block_type=NpuBlockType.ElementWise, indices=NNG_IFM_IFM2_INDICES)
    Neg = OperatorInfo()
    NonMaxSuppressionV4 = OperatorInfo()
    NonMaxSuppressionV5 = OperatorInfo()
    NotEqual = OperatorInfo()
    OneHot = OperatorInfo()
    Pack = OperatorInfo(indices=NNG_IFM_INDICES)
    PackReshaped = OperatorInfo(indices=NNG_IFM_INDICES)
    Pad = OperatorInfo(indices=NNG_IFM_INDICES)
    PadV2 = OperatorInfo()
    Placeholder = OperatorInfo()  # Only used in CPU subgraphs
    Pow = OperatorInfo()
    Prelu = OperatorInfo(indices=NNG_IFM_IFM2_INDICES)
    Prod = OperatorInfo()
    Quantize = OperatorInfo(indices=NNG_IFM_INDICES)
    QuantizedAvgPool = OperatorInfo(block_type=NpuBlockType.Pooling, indices=NNG_IFM_INDICES)
    QuantizedConv2D = OperatorInfo(block_type=NpuBlockType.ConvolutionMxN, indices=NNG_IFM_WEIGHTS_INDICES)
    QuantizedMatMul = OperatorInfo(block_type=NpuBlockType.VectorProduct, indices=NNG_IFM_WEIGHTS_INDICES)
    QuantizedMaxPool = OperatorInfo(block_type=NpuBlockType.Pooling, indices=NNG_IFM_INDICES)
    QuantizedReshape = OperatorInfo(indices=NNG_IFM_INDICES)
    Range = OperatorInfo()
    Rank = OperatorInfo()
    ReduceSum = OperatorInfo(block_type=NpuBlockType.ReduceSum, indices=NNG_IFM_INDICES)
    Relu = OperatorInfo(indices=NNG_IFM_INDICES)
    Relu0To1 = OperatorInfo(indices=NNG_IFM_INDICES)
    Relu6 = OperatorInfo(indices=NNG_IFM_INDICES)
    ReluN1To1 = OperatorInfo(indices=NNG_IFM_INDICES)
    ReluN = OperatorInfo(indices=NNG_IFM_INDICES)  # TOSA specific
    Rescale = OperatorInfo(indices=NNG_IFM_INDICES)  # TOSA specific
    Reshape = OperatorInfo(indices=NNG_IFM_INDICES)
    # resize ops map to pooling operations unless explicitly converted to other operations in the graph optimiser
    ResizeBilinear = OperatorInfo(block_type=NpuBlockType.Pooling, indices=NNG_IFM_INDICES)
    ResizeNearestNeighbor = OperatorInfo(block_type=NpuBlockType.Pooling, indices=NNG_IFM_INDICES)
    ReverseSequence = OperatorInfo()
    ReverseV2 = OperatorInfo()
    Rnn = OperatorInfo(indices=NNG_IFM_WEIGHTS_INDICES)
    Round = OperatorInfo()
    Rsqrt = OperatorInfo(block_type=NpuBlockType.ElementWise, indices=NNG_IFM_INDICES, is_unary=True)
    SHL = OperatorInfo(block_type=NpuBlockType.ElementWise, indices=NNG_IFM_IFM2_INDICES)  # NPU specific operation
    SHR = OperatorInfo(block_type=NpuBlockType.ElementWise, indices=NNG_IFM_IFM2_INDICES)  # NPU specific operation
    ScatterNd = OperatorInfo()
    SegmentSum = OperatorInfo()
    Select = OperatorInfo()
    SelectV2 = OperatorInfo()
    Shape = OperatorInfo(indices=NNG_IFM_INDICES)
    Sigmoid = OperatorInfo(indices=NNG_IFM_INDICES)
    Sign = OperatorInfo(indices=NNG_IFM_INDICES)
    SignBit = OperatorInfo()
    Sin = OperatorInfo()
    SkipGram = OperatorInfo()
    Slice = OperatorInfo(indices=NNG_IFM_INDICES)
    Softmax = OperatorInfo(indices=NNG_IFM_INDICES)
    SpaceToBatchND = OperatorInfo()
    SpaceToDepth = OperatorInfo()
    SparseToDense = OperatorInfo()
    Split = OperatorInfo(indices=NNG_SPLIT_IFM_INDICES)
    SplitSliceRead = OperatorInfo(indices=NNG_IFM_INDICES)
    SplitV = OperatorInfo(indices=NNG_IFM_INDICES)
    Sqrt = OperatorInfo(block_type=NpuBlockType.ElementWise, indices=NNG_IFM_INDICES, is_unary=True)
    Square = OperatorInfo()
    SquaredDifference = OperatorInfo(block_type=NpuBlockType.ElementWise, indices=NNG_IFM_IFM2_INDICES)
    Squeeze = OperatorInfo(indices=NNG_IFM_INDICES)
    StridedSlice = OperatorInfo(indices=NNG_IFM_INDICES)
    Sub = OperatorInfo(block_type=NpuBlockType.ElementWise, indices=NNG_IFM_IFM2_INDICES)
    SubgraphInput = OperatorInfo()  # Only used in CPU subgraphs
    Sum = OperatorInfo()
    Svdf = OperatorInfo()
    Table = OperatorInfo(indices=NNG_IFM_INDICES)
    Tanh = OperatorInfo(indices=NNG_IFM_INDICES)
    Tile = OperatorInfo()
    TopKV2 = OperatorInfo()
    Transpose = OperatorInfo(indices=NNG_IFM_IFM2_INDICES)
    UnidirectionalSequenceLstm = OperatorInfo(indices=NNG_IFM_WEIGHTS_INDICES)
    UnidirectionalSequenceRnn = OperatorInfo(indices=NNG_IFM_WEIGHTS_INDICES)
    Unique = OperatorInfo()
    Unpack = OperatorInfo(indices=NNG_IFM_INDICES)
    UnpackReshaped = OperatorInfo(indices=NNG_IFM_INDICES)
    VariableTensorWrite = OperatorInfo()
    Where = OperatorInfo()
    While = OperatorInfo()
    ZerosLike = OperatorInfo()
    CallOnce = OperatorInfo()
    BroadcastTo = OperatorInfo()
    Rfft2D = OperatorInfo()
    Conv3D = OperatorInfo()
    Imag = OperatorInfo()
    Real = OperatorInfo()
    ComplexAbs = OperatorInfo()
    Hashtable = OperatorInfo()
    HashtableFind = OperatorInfo()
    HashtableImport = OperatorInfo()
    HashtableSize = OperatorInfo()
    ReduceAll = OperatorInfo()
    Conv3DTranspose = OperatorInfo()
    VarHandle = OperatorInfo()
    ReadVariable = OperatorInfo()
    AssignVariable = OperatorInfo()
    BroadcastArgs = OperatorInfo()
    RandomStandardNormal = OperatorInfo()
    Bucketize = OperatorInfo()
    RandomUniform = OperatorInfo()
    Multinomial = OperatorInfo()
    Gelu = OperatorInfo(block_type=NpuBlockType.ElementWise, indices=NNG_IFM_INDICES, is_unary=True)
    DynamicUpdateSlice = OperatorInfo()
    UnsortedSegmentProd = OperatorInfo()
    UnsortedSegmentMax = OperatorInfo()
    UnsortedSegmentMin = OperatorInfo()
    UnsortedSegmentSum = OperatorInfo()
    Bitcast = OperatorInfo()
    BitwiseXor = OperatorInfo()
    RightShift = OperatorInfo()

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
        return self in (Op.Relu, Op.Relu6, Op.ReluN1To1, Op.ReluN, Op.Clip, Op.Clamp)

    def is_activation_op(self):
        return self.is_relu_op() or self in (Op.Tanh, Op.Sigmoid, Op.Softmax, Op.LUT, Op.HardSwish)

    def is_split_op(self):
        return self in (Op.Split, Op.SplitV, Op.StridedSlice, Op.Slice, Op.UnpackReshaped, Op.Unpack)

    def is_concat_op(self):
        return self in (Op.Concat, Op.ConcatTFLite, Op.PackReshaped, Op.Pack)

    def is_resize_op(self):
        return self in (Op.ResizeBilinear, Op.ResizeNearestNeighbor)

    def is_memcpy_op(self):
        return self.info.block_type == NpuBlockType.Dma

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
    TILE = 3  # Uses hardware tiles to pad by 1 with edge values on two sides of the IFM specified in explicit_padding


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


class ExplicitScaling:
    """Explicit scaling parameters"""

    def __init__(self, per_channel, shift, multiplier):
        self.per_channel = per_channel
        self.shift = shift
        self.multiplier = multiplier

    def clone(self):
        res = copy.copy(self)
        return res


def create_activation_function(op_type: Op, min=None, max=None) -> ActivationFunction:
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
    elif op_type == Op.Clamp:
        assert min is not None and max is not None
        act.min = min
        act.max = max
    elif op_type == Op.ReluN:
        assert max is not None
        act.min = 0.0
        act.max = max

    return act


class Operation:
    """Class representing a Neural Network operation. Has a name, a type,
    input and output tensors, as well as an attribute dictionary."""

    __slots__ = (
        "type",
        "_original_type",
        "name",
        "version",
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
        "read_shapes",
        "_rounding_mode",
        "explicit_scaling",
        "write_offset",
        "write_shape",
        "ifm_resampling_mode",
        "tile_base_offsets_ifm",
        "tile_base_offsets_ofm",
        "ofm_stride_multiplier",
    )

    def __init__(self, op_type: Op, name: str):
        self.type = op_type
        self._original_type = op_type  # the original type of the operation. once set this shouldn't be changed
        self.name = name
        self.version = 1  # Used to track original operator version.
        self.attrs: Dict[str, Any] = {}
        self.inputs: List[Optional[Tensor]] = []
        self.outputs: List[Tensor] = []
        self.intermediates: List[Tensor] = []
        self.flops = 0
        self.run_on_npu = True
        # Fused activation function. If not none: operator code.
        self.activation: Optional[ActivationFunction] = None
        # Fused memory function, if not None: operator code
        self.memory_function: Optional[Op] = None
        # If not none: contains QuantizationParameters to be used as output quantization
        # (which overrides the ofm tensor's quantization), used in LUT
        self.forced_input_quantization: Optional[QuantizationParameters] = None
        self.forced_output_quantization: Optional[QuantizationParameters] = None
        self.scheduled_pass = None
        self.op_index = None  # input network operator index
        self.activation_lut = None
        self._kernel = None
        self.ifm_shapes: List[Shape4D] = []
        self.ofm_shapes: List[Shape4D] = []
        self.read_offsets: List[Optional[Shape4D]] = [None, None]  # offset for [ifm, ifm2]
        self.read_shapes: List[Optional[Shape4D]] = [None, None]  # read shape for [ifm, ifm2]
        self._rounding_mode: Optional[RoundingMode] = None
        # Rescale op in TOSA supplies explicit multiplier and shift values
        self.explicit_scaling: Optional[ExplicitScaling] = None
        # Write offset, for operations that only produce a part of the OFM
        self.write_offset: Optional[Shape4D] = None
        # The amount of OFM that is produced by the operation (only if write_offset is not None).
        # E.g. an operation that only fills the bottom row of an OFM of size 1x10x8x1 would have
        # write_offset 0,9,0,0, write_shape 1,1,8,1
        self.write_shape: Optional[Shape4D] = None
        self.ifm_resampling_mode: resampling_mode = resampling_mode.NONE
        # ifm (nhwc), ifm2 (nhwc)
        self.tile_base_offsets_ifm: List[List[int]] = [[0, 0, 0, 0], [0, 0, 0, 0]]
        # ofm (nhwc)
        self.tile_base_offsets_ofm: List[int] = [0, 0, 0, 0]
        # For interleaved/sparse outputs - stride is multiplied with the stride factor of the corresponding axis
        # Order is [C, H, W] - default is no multiplication
        self.ofm_stride_multiplier: List[int] = [1, 1, 1]

    def clone(self, suffix="_clone"):
        res = Operation(self.type, self.name + suffix)

        # maintain the original type, in cases where the type was changed to something different
        res._original_type = self._original_type
        res.version = self.version
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
        res.read_shapes = list(self.read_shapes)
        res.write_offset = Shape4D(*self.write_offset) if self.write_offset else None
        res.write_shape = Shape4D(*self.write_shape) if self.write_shape else None
        res.rounding_mode = self.rounding_mode
        res.explicit_scaling = self.explicit_scaling
        res.ifm_resampling_mode = self.ifm_resampling_mode
        res.tile_base_offsets_ifm = [_ifm.copy() for _ifm in self.tile_base_offsets_ifm]
        res.tile_base_offsets_ofm = self.tile_base_offsets_ofm.copy()
        res.ofm_stride_multiplier = self.ofm_stride_multiplier.copy()

        return res

    def __str__(self):
        return "<nng.Operation '{}' type={}>".format(self.name, self.type)

    __repr__ = __str__

    @property
    def original_type(self):
        return self._original_type

    @property
    def rounding_mode(self):
        return self._rounding_mode

    @rounding_mode.setter
    def rounding_mode(self, mode: RoundingMode):
        # All rounding modes are supported by all operators with the exception of rounding away from zero (see comment
        # below)
        is_supported = True
        if mode == RoundingMode.AwayZero:
            # Rounding away from zero does not have direct hardware support and so the compiler implements it indirectly
            # in different ways. The exact process depends upon the operator type and not all operators are supported.
            # Basically, rounding away from zero works by adjusting the accumulated value by a "small" amount before
            # rounding up with the addition of a half (natural rounding). This "small" amount should be big enough to
            # cause x.5 to be rounded correctly but small enough that smaller values are not incorrectly rounded. This
            # is done by slightly adjusting the scale and shift on the ofm tensor using the scale and bias tensor,
            # it has no affect on global scaling (i.e. the ofm_scale register). In addition, the zero points of the
            # input and/or output tensors may also require forcing to zero but the exact behaviour also depends upon the
            # corresponding optimisation performed in graph_optimisation.py where the rounding mode is set
            is_supported = False
            if self.original_type == Op.ResizeBilinear and self.type == Op.DepthwiseConv2DBias:
                is_supported = True
            if self.original_type == Op.AvgPool and self.type in (Op.DepthwiseConv2DBias, Op.Conv2DBias):
                is_supported = True

        if is_supported:
            self._rounding_mode = mode
        else:
            assert (
                False
            ), f"Setting rounding mode = {mode} on {self.original_type} operator '{self.name}' is not supported."

    @property
    def type_changed(self):
        return self.type != self.original_type

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

    def get_ifm_ifm2_ofm(self):
        return self.ifm, self.ifm2, self.ofm

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
                offset_start[idx] = begin_tens.values[idx]
                offset_end[idx] = size_tens.values[idx] + offset_start[idx]

        elif self.type == Op.StridedSlice:
            input_tens, begin_tens, end_tens, strides_tens = self.inputs
            outputs = self.outputs

            # Extract masks
            ellipsis_mask = self.attrs["ellipsis_mask"]
            new_axis_mask = self.attrs["new_axis_mask"]
            shrink_axis_mask = self.attrs["shrink_axis_mask"]

            # shrink_axis_mask/new_axis_mask/ellipsis_mask is not supported by the Operation class but the operation
            # may have the attribute modified and handled in the graph optimization phase.
            assert shrink_axis_mask == new_axis_mask == ellipsis_mask == 0
            # use the begin and end values that were calculated in the model semantic check. this is because the end
            # values can be affected (ignored) by the shrink_axis_mask and this mask may have been changed in the graph
            # optimizer (see assert above)
            offset_start = self.attrs["offset_begin"]
            offset_end = self.attrs["offset_end"]
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

    def add_output_tensor(self, tens):
        self.outputs.append(tens)
        if self not in tens.ops:
            tens.ops.append(self)

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

        ifm_tensor, ifm2_tensor, ofm_tensor = self.get_ifm_ifm2_ofm()

        # set all shapes to op, as 4D
        if self.type == Op.FullyConnected:
            if len(self.ifm.shape) == 2:
                self.ifm_shapes.append(Shape4D([self.ifm.shape[0], 1, 1, self.ifm.shape[1]]))
            else:
                # Special case, handled in graph optimization
                self.ifm_shapes.append(Shape4D(ifm_tensor.get_full_shape()))
            self.ofm_shapes.append(Shape4D(ofm_tensor.get_full_shape()))

        elif self.type == Op.Softmax:
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

    def has_scaling(self):
        scaled = True
        for tensor in [self.ifm, self.ifm2, self.ofm]:
            if tensor is not None:
                if tensor.quantization is None:
                    scaled = False
                    break

        return scaled
