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
# Early optimisation of a TensorFlow Lite based network graph, using the rewrite_graph module
# to do the traversal of the graph.
from __future__ import annotations

import math
import uuid
import sys

import numpy as np

from . import fp_math
from . import rewrite_graph
from . import scaling
from .data_type import BaseType
from .data_type import DataType
from .debug_database import DebugDatabase
from .errors import UnsupportedFeatureError
from .ethos_u55_regs.ethos_u55_regs import resampling_mode
from .graph_optimiser_util import bypass_memory_only_ops
from .graph_optimiser_util import calc_explicit_padding
from .graph_optimiser_util import convert_depthwise_to_conv
from .graph_optimiser_util import create_avg_pool_for_concat
from .graph_optimiser_util import memory_only_ops
from .graph_optimiser_util import move_splitsliceread_to_consumer
from .graph_optimiser_util import needed_total_padding
from .graph_optimiser_util import set_ifm_ofm_op_shapes
from .graph_optimiser_util import set_tensor_equivalence
from .lstm import Lstm
from .lut import convert_to_lut
from .lut import create_lut_8bit_op
from .lut import create_lut_int16_op
from .lut import create_lut_rsqrt_int8_op
from .numeric_util import clamp_sigmoid
from .numeric_util import full_shape
from .numeric_util import round_away_zero
from .numeric_util import round_down_log2
from .operation import create_activation_function
from .operation import ExplicitScaling
from .operation import NpuBlockType
from .operation import Op
from .operation import Operation
from .operation import Padding
from .operation import RoundingMode
from .operation_util import create_add
from .operation_util import create_add_nop
from .operation_util import create_avgpool_nop
from .operation_util import create_cast_op
from .operation_util import create_depthwise_maxpool
from .operation_util import create_memcpy
from .operation_util import get_pad_values_from_input
from .scaling import quantise_scale
from .shape4d import Shape4D
from .softmax import SoftMax
from .tensor import check_quantized_tens_scaling_equal
from .tensor import create_const_tensor
from .tensor import create_equivalence_id
from .tensor import QuantizationParameters
from .tensor import Tensor
from .tensor import TensorPurpose
from .tflite_mapping import optype_to_builtintype
from .utils import calc_resize_factor

passthrough_nodes = (Op.Identity,)


def remove_passthrough_tensor(tens, arch, nng):
    if len(tens.ops) == 1 and tens.ops[0].type in passthrough_nodes:
        assert len(tens.ops[0].inputs) == 1
        tens = tens.ops[0].inputs[0]
    return tens

def add_add_op_after_concat(op, arch):
    if not op.run_on_npu or not op.type.is_concat_op():
        return op

    # Add scaled and alpha multiplied values (without scaling)
    out = op.outputs[0]
    out_shape = out.shape.copy()
    in1 = Tensor(out_shape, out.dtype, f"{op.outputs[0].name}_sub")
    in1.quantization = out.quantization

    quantization = QuantizationParameters(0.0, 255.0)
    quantization.scale_f32 = in1.quantization.scale_f32
    quantization.zero_point = 0
    in2 = create_const_tensor(
           f"{op.name}_sub", [1], out.dtype, [0], quantization=quantization)

    add_op = Operation(Op.Add, op.name + "_add")
    add_op.explicit_scaling = ExplicitScaling(False, shift=[0], multiplier=[1])  # No scaling
    add_op.add_input_tensor(in1)
    add_op.add_input_tensor(in2)
    add_op.set_output_tensor(out)
    add_op.set_ifm_ofm_shapes()
    add_op.attrs["pot_scale_int16"] = False

    op.set_output_tensor(in1)
    op.set_ifm_ofm_shapes()

    return add_op

def rewrite_concat_ops(op, arch):
    if not op.run_on_npu or not op.type.is_concat_op():
        return

    axis_4D = 0
    ofm = op.ofm
    ofm.ops = []
    offset = 0

    unfuse_activation_function(op)

    if op.type == Op.Pack:
        # Pack is also referred to as Stack
        axis = int(op.attrs["axis"])
        if axis < 0:  # Convert to positive axis
            axis = len(op.inputs[0].shape) + 1 + axis

        desired_shape = op.inputs[0].shape[:axis] + [1] + op.inputs[0].shape[axis:]

        axis_4D = axis + (4 - len(desired_shape))

        for idx, inp in enumerate(op.inputs):
            op.ifm_shapes[idx] = Shape4D(desired_shape)
        op.type = Op.PackReshaped

    inputs, axis = op.get_concat_inputs_axis()
    for idx, inp in enumerate(inputs):
        if op.type != Op.PackReshaped:
            op.ifm_shapes[idx] = Shape4D(inp.shape)
            if axis >= 0:
                axis_4D = axis + (4 - len(inp.shape))
            else:
                axis_4D = axis
        write_offset = [0, 0, 0, 0]
        write_offset[axis_4D] = offset
        concat_end = offset + op.ifm_shapes[idx][axis_4D]
        create_avg_pool_for_concat(
            op, op.name + str(idx) + "_avgpool", inp, op.ifm_shapes[idx], Shape4D.from_list(write_offset)
        )
        offset = concat_end
    assert ofm.shape[axis] == offset

    return op


def rewrite_split_ops(tens, arch, nng):

    if len(tens.ops) == 1 and tens.ops[0].type.is_split_op() and tens.ops[0].type != Op.Unpack:
        split_op = tens.ops[0]

        # Not supported so leave it and run on CPU
        if not split_op.run_on_npu:
            return tens

        inp, outputs, axis, offset_start, offset_end = split_op.get_split_inputs_axis()

        tens.ops = []
        new_op = Operation(Op.SplitSliceRead, split_op.name)
        new_op.inputs = [inp]
        ofm_shape_idx = 0
        if None in (offset_end, offset_start):
            read_shape = None
        else:
            # the read shape is relative to each start offset
            read_shape = Shape4D([oe - os for oe, os in zip(offset_end, offset_start)])

        # For Split the offset cannot be extracted from the tensor so it has to
        # be calculated from the index of the output tensor
        if axis is not None:
            # Get the start and end of the split
            offset_start = [0] * 4
            axis_4D_list = split_op.attrs.get("split_axis_4D", None)  # Present for UnpackReshaped and some StridedSlice
            for idx, out in enumerate(outputs):
                if axis_4D_list is not None:
                    axis_4D = axis_4D_list[idx]
                else:
                    split_op.ofm_shapes[idx] = Shape4D(out.shape)
                    if axis >= 0:
                        axis_4D = axis + (4 - len(out.shape))
                    else:
                        axis_4D = axis

                if out == tens:
                    ofm_shape_idx = idx
                    read_shape = split_op.ofm_shapes[idx]
                    break

                offset_start[axis_4D] += split_op.ofm_shapes[idx][axis_4D]

        new_op.read_offsets[0] = Shape4D.from_list(offset_start, 0)
        new_op.read_shapes[0] = read_shape
        new_op.run_on_npu = True
        new_op.set_output_tensor(tens)
        new_op.ifm_shapes.append(Shape4D(inp.shape))
        new_op.ofm_shapes.append(split_op.ofm_shapes[ofm_shape_idx])
        DebugDatabase.add_optimised(split_op, new_op)

    return tens


def remove_SplitSliceRead(op, arch):

    if op.type == Op.SplitSliceRead:
        # Check if it is possible to put the SplitSliceRead on the tensor consumer(s),
        # or if an avgpool need to be inserted
        # Not possible to do if consumer is a Transpose op since ifm shape has been reshaped and can not be changed
        if op.ofm_shapes[0] == Shape4D.from_list(op.ofm.shape) and all(
            consumer is not None
            and consumer.run_on_npu
            and consumer.type not in memory_only_ops
            and consumer.type != Op.Mul
            and consumer.original_type != Op.Transpose
            for consumer in op.ofm.consumer_list
        ):
            # SplitSliceRead can be performed by tensor consumer(s)
            for cons_op in list(op.ofm.consumer_list):
                move_splitsliceread_to_consumer(op, cons_op)
        else:
            avgpool_op = create_avgpool_nop(op.name + "_avgpool")
            avgpool_op.add_input_tensor(op.ifm)
            avgpool_op.outputs = [op.ofm]
            op.ofm.ops.remove(op)
            op.ofm.ops.append(avgpool_op)
            avgpool_op.ifm_shapes.append(op.ifm_shapes[0])
            avgpool_op.ofm_shapes.append(op.ofm_shapes[0])
            avgpool_op.read_offsets[0] = op.read_offsets[0]
            avgpool_op.read_shapes[0] = op.read_shapes[0]

            op.ifm.consumer_list.remove(op)
            DebugDatabase.add_optimised(op, avgpool_op)


def calc_padding_and_skirt(padding_type, kernel, input_shape, explicit_padding):
    k_w, k_h = kernel.dilated_wh()
    s_x, s_y = kernel.stride
    ypad = needed_total_padding(int(input_shape.height), int(s_y), int(k_h))
    xpad = needed_total_padding(int(input_shape.width), int(s_x), int(k_w))
    if padding_type == Padding.SAME:
        left_pad = (xpad + 0) // 2
        right_pad = (xpad + 1) // 2
        top_pad = (ypad + 0) // 2
        bottom_pad = (ypad + 1) // 2
    elif padding_type == Padding.VALID:
        left_pad = 0
        right_pad = 0
        top_pad = 0
        bottom_pad = 0
    elif padding_type == Padding.EXPLICIT:
        # Padding is specified in a PAD operator which has been bypassed.
        top, left, bottom, right = explicit_padding
        top_pad, bottom_pad = calc_explicit_padding(int(input_shape.height), int(s_y), int(k_h), int(top), int(bottom))
        left_pad, right_pad = calc_explicit_padding(int(input_shape.width), int(s_x), int(k_w), int(left), int(right))
    elif padding_type == Padding.TILE:
        # The values in the explicit padding only represent the "direction" in which to pad
        top_pad, left_pad, bottom_pad, right_pad = explicit_padding
    else:
        raise UnsupportedFeatureError(f"Unsupported padding = {padding_type} for padding calculation")
    padding = (top_pad, left_pad, bottom_pad, right_pad)
    skirt = (top_pad, left_pad, ypad - top_pad, xpad - left_pad)
    return padding, skirt


def calc_upscaled_padding_and_skirt(
    padding_type, kernel_size, stride, input_shape, upscaling_factor_y, upscaling_factor_x
):
    kernel_height, kernel_width = kernel_size[0], kernel_size[1]
    if padding_type == Padding.SAME:
        ypad = needed_total_padding(int(input_shape.height) * upscaling_factor_y, int(stride[1]), int(kernel_height))
        xpad = needed_total_padding(int(input_shape.width) * upscaling_factor_x, int(stride[2]), int(kernel_width))
        right_pad = max(((xpad + 1) // upscaling_factor_x) - 1, 0)
        bottom_pad = max(((ypad + 1) // upscaling_factor_y) - 1, 0)
        left_pad = max(kernel_width - 1 - right_pad, 0)
        top_pad = max(kernel_height - 1 - bottom_pad, 0)
    elif padding_type == Padding.VALID:
        right_pad = max(kernel_width - 2, 0)
        bottom_pad = max(kernel_height - 2, 0)
        left_pad = kernel_width - 1
        top_pad = kernel_height - 1
    else:
        raise UnsupportedFeatureError(f"Unsupported padding = {padding_type} for up-scaled padding calculation")
    padding = (top_pad, left_pad, bottom_pad, right_pad)
    skirt = padding
    return padding, skirt


def fixup_conv2d_backprop(op: Operation, arch, nng) -> Operation:
    if op.type == Op.Conv2DBackpropInput:
        # flip the inputs
        op.inputs[0], op.inputs[2] = op.inputs[2], op.inputs[0]
        op.type = Op.Conv2DBackpropInputSwitchedBias
        stride_w = op.kernel.stride.x
        stride_h = op.kernel.stride.y
        if stride_w > 1 or stride_h > 1:
            # Transpose conv2d with upscaling
            op.ifm_resampling_mode = resampling_mode.TRANSPOSE

        # Update strides
        op.attrs.update({"stride_w": 1, "stride_h": 1, "strides": (1, 1, 1, 1)})
        DebugDatabase.add_optimised(op, op)

    return op


# Convert the op to an elementwise add
def convert_resize_1x1_to_add(op):
    op.type = Op.Add  # original_type will stay as Op.ResizeBilinear or Op.ResizeNearestNeighbor
    op.name = op.name + "_add"
    # Create an input tensor filled with zeros
    name = op.inputs[1].name + "_add"
    dtype = op.inputs[0].dtype
    shape = op.ofm_shapes[0].as_list()
    values = np.zeros(shape, dtype.as_numpy_type())
    quantization = QuantizationParameters(0.0, 255.0)
    quantization.scale_f32 = 1.0
    quantization.zero_point = 0
    op.inputs[1] = op.inputs[0]
    op.set_input_tensor(create_const_tensor(name, shape, dtype, values, quantization=quantization), 0)
    op.set_ifm_ofm_shapes()
    DebugDatabase.add_optimised(op, op)

    return op


# Convert ResizeNearestNeighbor with align corners to a depthwise convolution. The IFM will already have been upscaled
# apart from the final x2 scaling which will be done as part of this operation. The kernel contains a single coefficient
# to select the appropriate nearest neighbor value
def convert_resizenn_ac_to_depthwise_conv(op, upscale_factor):
    ifm = op.ifm
    ofm = op.ofm
    output_depth = ofm.shape[-1]
    dw_op_attrs = {
        "padding": Padding.VALID,
        "stride_h": 1,
        "stride_w": 1,
        "strides": (1, 1, 1, 1),
        "depth_multiplier": 1,
        "channel_multiplier": 1,
        "dilation_h_factor": 1,
        "dilation_w_factor": 1,
        "dilation": (1, 1, 1, 1),
    }

    # change ResizeNearestNeighbor to Depthwise
    op.type = Op.DepthwiseConv2DBias
    op.attrs.update(dw_op_attrs)
    op.set_input_tensor(ifm, 0)  # ifm tensor index
    op.activation = None

    # add input resample to resize by x2
    op.ifm_resampling_mode = resampling_mode.NEAREST

    # don't care about the rounding mode as it is nearest neighbor

    # setup weight tensor
    weight_quant = QuantizationParameters()
    weight_quant.scale_f32 = 1.0  # no scaling as only a single non-zero coeff to select the desired value
    weight_quant.zero_point = 0
    weight_quant.quant_dim = 0
    ofm_dtype = ofm.dtype
    if ofm_dtype.type == BaseType.UnsignedInt:
        weight_quant.quant_min = 0
        weight_quant.quant_max = (1 << ofm_dtype.bits) - 1
    else:
        weight_quant.quant_min = -(1 << (ofm_dtype.bits - 1))
        weight_quant.quant_max = (1 << (ofm_dtype.bits - 1)) - 1

    weight_shape = [upscale_factor, upscale_factor, output_depth, output_depth]  # HWIO

    # the single non-zero coefficient used to select the desired value needs to be placed in the 'centre value', which
    # is calculated by finding the 'centre position' ('*' in the diagram below) and then choosing the 'value' that is
    # below-and-right (i.e. next) to it (D).
    # 0---1---2
    # | A | B |
    # 1---*---+
    # | C | D |
    # 2---+---+
    weight_values = [0] * (upscale_factor * upscale_factor)
    centre_coeff = (upscale_factor // 2) * upscale_factor + (upscale_factor // 2)
    weight_values[centre_coeff] = 1

    # add weight tensor, this will discard the size tensor of the resize op
    op.set_input_tensor(
        create_const_tensor(
            "weights",
            weight_shape,
            ofm_dtype,
            np.array(weight_values).reshape(weight_shape),
            quantization=weight_quant,
        ),
        1,  # inputs tensor weight index
    )

    # setup bias tensor by assign None and then call the fix-up function to create a suitable tensor.
    # need to append the bias tensor as resize ops only have 2 inputs
    assert len(op.inputs) == 2
    op.inputs.append(None)
    fixup_bias_tensors(op, None, None, DataType.int32)

    # finally update the shape incase we've change the tensor shapes or connections
    op.set_ifm_ofm_shapes()
    DebugDatabase.add_optimised(op, op)

    return op


# Convert ResizeBilinear/NearestNeighbor to a number of 1x1 average pools with nearest neighbor x2 upscaling and one
# final average pool with a kernel size that depends upon the resize ops upscaling factor (x2, x4 or x8). The maximum
# upscale factor is limited to x8 because of the limit 8x8 kernel size limit for average pool with padding.
def convert_resize_to_upscale_and_average_pool(op):
    pre_op = op
    outputs = op.outputs
    dtype = op.ifm.dtype

    op.attrs.update({"strides": (1, 1, 1, 1), "ksize": (1, 1, 1, 1)})
    op.attrs["padding"] = Padding.SAME  # doesn't really matter as the kernel is 1x1
    op.ifm_resampling_mode = resampling_mode.NEAREST

    upscaled_shape = np.array(op.ifm_shapes[0].get_hw_as_list())

    # Get upscale factor that was calculated in the supported operators check
    upscale_factor = op.attrs["upscale_factor"]

    # Calculate how many times 2x2 upscaling needs to be performed
    # Force the result of round to be an integer. This is because the behaviour of rounding numpy.float64 values changed
    # between different versions of numpy. This consistency ensures that the kernel dimensions are kept integral
    n = int(np.log2(upscale_factor))

    # Perform x2 upscaling n-1 times
    scaled_op = pre_op
    for count in range(n - 1):
        if count > 0:
            scaled_op = op.clone(f"_{count}")
            scaled_op.inputs[0] = pre_op.outputs[0]

        # Nearest neighbor x2 upscaling
        upscaled_shape = upscaled_shape * 2
        shape = op.ofm_shapes[0].as_list()
        shape[1:3] = upscaled_shape
        out_tens = Tensor(shape, dtype, f"{op.outputs[0].name}_{count}")
        out_tens.quantization = op.outputs[0].quantization.clone()
        scaled_op.set_output_tensor(out_tens)
        pre_op = scaled_op

        scaled_op.set_ifm_ofm_shapes()
        DebugDatabase.add_optimised(op, scaled_op)

    # Last x2 upscaling
    if n > 1:
        scaled_op = op.clone(f"_{n-1}")
        scaled_op.inputs[0] = pre_op.outputs[0]

    if scaled_op.original_type == Op.ResizeBilinear:
        if scaled_op.attrs["align_corners"]:
            # no padding
            scaled_op.attrs["padding"] = Padding.VALID
        else:
            # padding to the right and bottom (limits average pool to 8x8 kernel)
            scaled_op.attrs["padding"] = Padding.EXPLICIT
            scaled_op.attrs["explicit_padding"] = [0, 0, upscale_factor - 1, upscale_factor - 1]

        # kernal size dependent on the upscaling factor
        scaled_op.attrs.update({"ksize": (1, upscale_factor, upscale_factor, 1)})
    else:  # Op.ResizeNearestNeighbor
        if scaled_op.attrs["align_corners"]:
            # use depthwise conv to select the correct value
            scaled_op = convert_resizenn_ac_to_depthwise_conv(scaled_op, upscale_factor)
        else:
            # Keep 1x1 kernel and average pool, this applies both when
            # half-pixel-centers is True and False. Calculations are the
            # same in the reference.
            pass

    scaled_op.outputs = outputs
    scaled_op.outputs[0].ops = [scaled_op]
    scaled_op.set_ifm_ofm_shapes()
    DebugDatabase.add_optimised(op, scaled_op)

    return op


def convert_argmax_to_depthwise_conv_and_max_pool(op: Operation, arch, nng) -> Operation:
    """
    Convert ArgMax to DWConv2D->MaxPool->DWConv2D, see details below.

    Example:
    arr = [4,   [00000100,
           6, =  00000110,  # <-- This is the largest value, so we're expecting argmax(arr) = 1
           5]    00000101]

    Use 16-bit precision and shift all values 7 bits to the left:
    Shifted_arr = [0000001000000000,
                   0000001100000000,
                   0000001010000000]

    Add "c - index of channel" to each channel:
    Shifted_arr_plus_reverse_idx = [0000001000000010, (+2)
                                    0000001100000001, (+1)
                                    0000001010000000] (+0)

    The index is reversed since ArgMax selects the lowest index if maximum value is found at two index. The index will
    act as a tie-breaker between channels with equal values and since we want the smallest channel index to be chosen
    we reverse the index before the maxpool and then subtract the index from the number of channel after the maxpool to
    get the correct index.

    Find the maximum value in the array:
    val = max(shifted_arr_plus_reverse_idx) = 0000001100000001

    Subtract the value from the number of channels:
    shifted_arr_plus_idx = (c-1) - val = 2 - 1 = 1

    Extract the 7 lowest bits using a LUT to cut off the 9 most significant bits:
    idx = LUT(val) = 0000000000000001 = 1
    """

    if op.type == Op.ArgMax:
        ifm, ofm = op.inputs[0], op.outputs[0]
        identity_quant = QuantizationParameters()
        identity_quant.zero_point = 0
        identity_quant.scale_f32 = 1.0
        # Add last dimension to ofm shape
        ofm.shape += [1]
        ofm.ops = []

        # Create 1x1 Depthwise convolution with 2**7 weights for each channel to convert precision to 16 bit and shift
        # all values 7 bits to the left
        # Set necessary depthwise attributes
        dw_op_attrs = {
            "padding": Padding.VALID,
            "stride_h": 1,
            "stride_w": 1,
            "strides": (1, 1, 1, 1),
            "depth_multiplier": 1,
            "channel_multiplier": 1,
            "dilation_h_factor": 1,
            "dilation_w_factor": 1,
            "dilation": (1, 1, 1, 1),
            "explicit_padding": None,
        }
        orig_name = op.name
        op.name = f"{orig_name}_depthwise_conv_SHL_7"
        op.type = Op.DepthwiseConv2DBias
        op.attrs.update(dw_op_attrs)
        n, h, w, c = full_shape(4, ifm.shape, 1)
        shape = [1, 1, 1, c]
        kernel = np.dstack([2**7] * c)
        op.inputs = []
        op.add_input_tensor(ifm)
        op.add_input_tensor(
            create_const_tensor(
                "weights",
                shape,
                DataType.uint8,
                np.array(kernel).reshape(shape),
                quantization=identity_quant,
            ),
        )
        # Let the bias for each channel be the "reverse" index of the channel it is in, ie c - channel_idx
        reverse_idxs = list(reversed(range(c)))
        bias_tensor = create_const_tensor(op.name + "_bias", [c], DataType.int64, reverse_idxs)
        op.add_input_tensor(bias_tensor)

        intermediate_tens = Tensor([n, h, w, c], DataType.int16, "int16_and_shifted_7_bits_left")
        intermediate_tens.quantization = ifm.quantization
        op.set_output_tensor(intermediate_tens)
        op.set_ifm_ofm_shapes()
        orig_ifm_shape = op.ifm_shapes[0]
        DebugDatabase.add_optimised(op, op)

        # To extract 7 least significant bits and swap reverse index back to real index using a LUT activation, we set
        # the base value to c-1 and slope to -128. The 16-bit LUT uses a table of 32-bit values where the top 16 bits
        # represent the slope and bottom 16 bits the base which are used to interpolate the activation value.
        slope = (-128 & 0xFFFF) << 16  # Top 16 bits of 32 bit LUT table value
        base = c - 1  # Bottom 16 bits of the LUT table value
        lut_tensor = create_const_tensor(
            "maxpool_LUT_extract_7_LSB",
            [1, 1, 1, 512],
            DataType.uint32,
            [slope + base] * 512,
            TensorPurpose.LUT,
        )

        # Split large feature maps into smaller chunks since the Depthwise Maxpool height dimension can overflow due to
        # flattening the ifm to (H*W)xCx1
        max_height = 2**16 // orig_ifm_shape.width
        num_full_height_ops = orig_ifm_shape.height // max_height
        last_op_height = orig_ifm_shape.height - max_height * num_full_height_ops
        op_heights = [max_height] * num_full_height_ops
        if last_op_height > 0:
            op_heights.append(last_op_height)

        # Create maxpool output tensor which is reshaped to 1x(H*W)x1x1. The product H*W might be larger than the
        # maximum allowed height, but that's handled by reading and writing the data in chunks
        maxpool_ofm = Tensor([1, orig_ifm_shape.height * orig_ifm_shape.width, 1, 1], DataType.int16, "argmax_maxpool")
        maxpool_ofm.quantization = identity_quant

        for op_idx, op_height in enumerate(op_heights):
            maxpool_op = create_depthwise_maxpool(
                f"dw_maxpool_{op_idx}", intermediate_tens, orig_ifm_shape, identity_quant
            )
            maxpool_op.outputs = [maxpool_ofm]
            maxpool_ofm.ops.append(maxpool_op)
            maxpool_op.ofm_shapes = [Shape4D(maxpool_ofm.shape)]
            maxpool_op.set_activation_lut(lut_tensor)

            # Set read and write shapes/offsets to read/write chunks of the IFM/OFM
            maxpool_op.read_shapes[0] = Shape4D([1, op_height * orig_ifm_shape.width, orig_ifm_shape.depth, 1])
            maxpool_op.read_offsets[0] = Shape4D([0, sum(op_heights[:op_idx]) * orig_ifm_shape.width, 0, 0])
            maxpool_op.write_shape = Shape4D([1, op_height * orig_ifm_shape.width, 1, 1])
            maxpool_op.write_offset = Shape4D([0, sum(op_heights[:op_idx]) * orig_ifm_shape.width, 0, 0])
            DebugDatabase.add_optimised(op, maxpool_op)

        # Set final shape
        maxpool_ofm.set_all_shapes([1, h, w, 1])

        # Convert 16bit to 32bit or 64bit
        if ofm.dtype == DataType.int64:
            # If OFM dtype is int64 the result is converted by two cast ops (16bit to 32bit)
            #
            #   A     -> B         -> C          -> D (OFM)
            #   |0001|   |00010000|   |0001|0000|   |00010000|00000000|
            #    i16      i32          i16  i16      i32      i32
            #                                       <-------i64------->
            #
            #   Memcpy is used to copy the content from B to C and from D to OFM
            #   Memcpy will be turned into a nop or an DMA transer if memory regions differs.
            intermediate_32bit = Tensor([1, h, w, 1], DataType.int32, f"{orig_name}_32bit")
        else:
            intermediate_32bit = ofm

        op_cast = create_cast_op(f"{orig_name}_cast_to_32bit_1", maxpool_ofm, intermediate_32bit)
        DebugDatabase.add_optimised(op, op_cast)

        if ofm.dtype == DataType.int64:
            # Create int16 tensor with double shape to cover the intermediate_32bit result from the first cast
            intermediate_16bit_2x_size = Tensor([1, h, w, 2], DataType.int16, f"{orig_name}_16bit_2x_size")
            memcpy_op = create_memcpy(f"{orig_name}_memcpy_1", intermediate_32bit, intermediate_16bit_2x_size)
            DebugDatabase.add_optimised(op, memcpy_op)

            # Create int32 tensor with double ofm shape to be able to store a "int64" result
            intermediate_32bit_2x_size = Tensor([1, h, w, 2], DataType.int32, f"{orig_name}_32bit_2x_size")

            op_cast = create_cast_op(
                f"{orig_name}_cast_to_32bit_2", intermediate_16bit_2x_size, intermediate_32bit_2x_size
            )
            DebugDatabase.add_optimised(op, op_cast)

            memcpy_op = create_memcpy("f{orig_name}_memcpy_2", intermediate_32bit_2x_size, ofm)
            DebugDatabase.add_optimised(op, memcpy_op)

    return op


def convert_resizebilinear_to_depthwise_convolutions(op, half_pixel_centers=True):
    def _compute_interpolation_values(index, input_size, output_size):
        scale = input_size / output_size
        scaled_value = (index + 0.5 * half_pixel_centers) * scale - 0.5 * half_pixel_centers
        lower_bound = max(np.floor(scaled_value), 0)

        return scaled_value, lower_bound

    def _compute_kernels(input_height, input_width, output_height, output_width):
        kernels = []
        for y in (1, 2):
            for x in (1, 2):
                sv_h, lb_h = _compute_interpolation_values(y, input_height, output_height)
                sv_w, lb_w = _compute_interpolation_values(x, input_width, output_width)

                # Interpolation values calculated for (x, y) = ([1, 2], [1, 2]) will always generalize to the whole
                # input for upscale = 2 and input sizes >= 2x2 and be in the correct order for going left-to-right,
                # top-to-bottom - same as the depthwise convolution strides across each tile
                kernel = np.zeros((2, 2))
                kernel[1, 1] = (1 - (sv_h - lb_h)) * (1 - (sv_w - lb_w))
                kernel[0, 1] = (sv_h - lb_h) * (1 - (sv_w - lb_w))
                kernel[1, 0] = (1 - (sv_h - lb_h)) * (sv_w - lb_w)
                kernel[0, 0] = (sv_h - lb_h) * (sv_w - lb_w)
                kernel *= 16
                kernels.append(kernel)

        return kernels

    def _build_convolutions(op, kernels):
        dw_op_attrs = {
            "padding": Padding.TILE,
            "stride_h": 1,
            "stride_w": 1,
            "strides": (1, 1, 1, 1),
            "depth_multiplier": 1,
            "channel_multiplier": 1,
            "dilation_h_factor": 1,
            "dilation_w_factor": 1,
            "dilation": (1, 1, 1, 1),
        }
        ifm = op.ifm
        ofm = op.ofm
        ofm.ops = []
        elem_size = 2 if ofm.dtype == DataType.int16 else 1

        n, h, w, c = ifm.shape
        _, _, ow, _ = ofm.shape

        intermediate_tens = Tensor(ifm.shape, ifm.dtype, "intermediate_tens")
        intermediate_tens.quantization = op.outputs[0].quantization.clone()
        avgpool_op = op
        avgpool_op.name = "rb_init_avgpool"
        avgpool_op.type = Op.AvgPool
        avgpool_op.attrs["padding"] = Padding.VALID
        avgpool_op.attrs["stride_w"] = 1
        avgpool_op.attrs["stride_h"] = 1
        avgpool_op.attrs["filter_width"] = 1
        avgpool_op.attrs["filter_height"] = 1
        avgpool_op.attrs["strides"] = [1, 1, 1, 1]
        avgpool_op.attrs["ksize"] = [1, 1, 1, 1]

        avgpool_op.add_input_tensor(ifm)
        avgpool_op.set_output_tensor(intermediate_tens)
        avgpool_op.set_ifm_ofm_shapes()
        DebugDatabase.add_optimised(op, op)

        dw_conv = Operation(Op.DepthwiseConv2DBias, "depthwise_conv")
        dw_conv._original_type = Op.ResizeBilinear
        dw_conv.write_shape = Shape4D(n, h, w, c)
        dw_conv.write_offset = Shape4D(0, 0, 0, 0)

        # Resize bilinear requires rounding away from zero
        dw_conv.rounding_mode = RoundingMode.AwayZero

        # Double height and width stride to write the output of each of the four depthwise convolutions below
        # interleaved with each other when combined with OFM tile base offsets.
        dw_conv.ofm_stride_multiplier = [1, 2, 2]  # C/H/W

        # Choose tile padding direction - pad by 1 with edge values in two direction.
        # For example, TL (top left) will pad top and left in H/W-plane in all channels.
        directions = [[1, 1, 0, 0], [1, 0, 0, 1], [0, 1, 1, 0], [0, 0, 1, 1]]  # TL, TR, BL, BR
        for i in (0, 1):
            for j in (0, 1):
                index = i * 2 + j
                dw_conv.name = f"depthwise_conv_{index}"
                dw_op_attrs["explicit_padding"] = directions[index]
                dw_conv.attrs.update(dw_op_attrs)

                # This will offset the start of the write by modifying the Tile 0 base address
                dw_conv.tile_base_offsets_ofm[0] = (i * ow + j) * c * elem_size

                ofm.ops.append(dw_conv)
                dw_conv.outputs = [ofm]

                kernel = kernels[index]
                shape = [2, 2, 1, c]
                kernel = np.dstack([kernel] * c)

                quant = QuantizationParameters()
                quant.zero_point = 0
                quant.scale_f32 = 1.0 / 16

                dw_conv.inputs = []
                dw_conv.add_input_tensor(intermediate_tens)
                dw_conv.add_input_tensor(
                    create_const_tensor(
                        "weights",
                        shape,
                        intermediate_tens.dtype,
                        np.array(kernel).reshape(shape),
                        quantization=quant,
                    ),
                )

                # setup bias tensor by assign None and then call the fix-up function to create a suitable tensor.
                # need to append the bias tensor as resize ops only have 2 inputs
                assert len(dw_conv.inputs) == 2
                dw_conv.inputs.append(None)
                fixup_bias_tensors(dw_conv, None, None, dtype=DataType.int32)

                dw_conv.set_ifm_ofm_shapes()
                DebugDatabase.add_optimised(op, dw_conv)

                dw_conv = dw_conv.clone(f"_{index}")
        return op

    _, input_height, input_width, _ = op.ifm.shape
    _, output_height, output_width, _ = op.ofm.shape

    kernels = _compute_kernels(input_height, input_width, output_height, output_width)
    op = _build_convolutions(op, kernels)

    return op


def fixup_resize(op: Operation, arch, nng) -> Operation:
    """Fixup resize ops to increase support for ResizeNearestNeighbor cases."""
    if op.type.is_resize_op() and op.run_on_npu:
        if op.ifm_shapes[0] == op.ofm_shapes[0]:
            # Bypass the resize op which is essentially a NOP
            op.inputs = op.inputs[:1]
            op.type = Op.Identity
        elif op.ifm_shapes[0].height == 1 and op.ifm_shapes[0].width == 1:
            convert_resize_1x1_to_add(op)
        elif op.type == Op.ResizeBilinear and op.attrs.get("half_pixel_centers", False):
            convert_resizebilinear_to_depthwise_convolutions(op)
        else:
            convert_resize_to_upscale_and_average_pool(op)

    return op


def convert_nop_split_to_identity(op, arch, nng):
    if op.type == Op.Split and op.attrs.get("num_splits") == 1:
        # the list comprehension should return a list with a single tensor
        # if it shouldn't, remove_passthrough_tensor will fail appropriately
        op.inputs = [i for i in op.inputs if i.shape == op.outputs[0].shape]
        op.type = Op.Identity
    return op


def rewrite_fully_connected_input(op: Operation, arch, nng) -> Operation:
    """Rewrite FullyConnected shape as 2D to allow it to run on NPU."""
    # If the operation already have a read shape do not modify
    # the ifm shape, since that will already be correct
    if op.type == Op.FullyConnected and not op.read_shapes[0]:
        new_shape = op.ifm.get_shape_as_2d(op.weights.shape[-2])
        assert new_shape is not None, "Tensor can not be reshaped to 2D"
        op.ifm_shapes[0] = new_shape

        if op.ifm_shapes[0].batch > 1 and op.ofm_shapes[0].batch == 1:
            # If IFM is batching then also make sure OFM is batching
            h, w = op.ofm_shapes[0].height, op.ofm_shapes[0].width
            op.ofm_shapes[0] = Shape4D([h * w, 1, 1, op.ofm_shapes[0].depth])

    return op


def convert_batched_fc_shape(op: Operation, arch, nng) -> Operation:
    """Convert batched FullyConnected op shape to allow for support on NPU."""
    if op.type == Op.FullyConnected:
        # Check if the first dimension indicates batching
        if op.ifm_shapes[0].batch > 1:
            batching_split = {4: (2, 2), 8: (2, 4), 16: (4, 4)}
            n = op.ifm_shapes[0].batch
            h, w = batching_split.get(n, (1, n))
            op.ifm_shapes[0] = Shape4D([1, h, w, op.ifm_shapes[0].depth])

            # Reshape Weights to be 4D. IO becomes HWIO
            weight_tensor = op.inputs[1]
            weight_tensor.values = np.expand_dims(np.expand_dims(weight_tensor.values, axis=0), axis=0)
            weight_tensor.set_all_shapes(list(weight_tensor.values.shape))

            n = op.ofm_shapes[0].batch
            h, w = batching_split.get(n, (1, n))
            op.ofm_shapes[0] = Shape4D([1, h, w, op.ofm_shapes[0].depth])
    return op


def unfuse_activation_function(op):
    if op.type == Op.ConcatTFLite and op.run_on_npu and op.activation is not None:
        act_op = Operation(op.activation.op_type, op.name + op.activation.op_type.name)
        op.activation = None
        out_tens = op.outputs[0]
        intermediate_tens = out_tens.clone("_act_intermediate")
        act_op.set_output_tensor(out_tens)
        act_op.add_input_tensor(intermediate_tens)
        op.set_output_tensor(intermediate_tens)
        act_op.set_ifm_ofm_shapes()
        DebugDatabase.add_optimised(op, act_op)


def rewrite_stridedslice_output(op, arch, nng):
    if not op.run_on_npu or op.type != Op.StridedSlice:
        return op

    new_axis_mask = op.attrs["new_axis_mask"]
    shrink_axis_mask = op.attrs["shrink_axis_mask"]

    if shrink_axis_mask == 0 and new_axis_mask == 0:
        return op

    axis_4D = [0] * len(op.outputs)
    for idx, out_tens in enumerate(op.outputs):
        output_shape = list(out_tens.shape)

        if shrink_axis_mask != 0:
            n = 0
            axis = 0
            while shrink_axis_mask:
                prev_mask = shrink_axis_mask
                n += 1
                shrink_axis_mask &= shrink_axis_mask - 1
                axis = int(math.log2(prev_mask - shrink_axis_mask))
                output_shape = output_shape[:axis] + [1] + output_shape[axis:]

            assert len(out_tens.shape) == (len(op.inputs[0].shape) - n)
            op.attrs["shrink_axis_mask"] = 0
            if axis >= 0:
                axis_4D[idx] = axis + (4 - len(output_shape))
            else:
                axis_4D[idx] = axis
            op.ofm_shapes[idx] = Shape4D(output_shape)

        elif new_axis_mask != 0:
            n = 0
            axis = 0
            while new_axis_mask:
                prev_mask = new_axis_mask
                n += 1
                new_axis_mask &= new_axis_mask - 1
                axis = int(math.log2(prev_mask - new_axis_mask))
                output_shape = output_shape[:axis] + output_shape[(axis + 1) :]
                new_axis_mask >>= 1

            assert len(out_tens.shape) == (len(op.inputs[0].shape) + n)
            op.attrs["new_axis_mask"] = 0
            if axis >= 0:
                axis_4D[idx] = axis + (4 - len(output_shape))
            else:
                axis_4D[idx] = axis
            op.ofm_shapes[idx] = Shape4D(output_shape)

    op.attrs["split_axis_4D"] = axis_4D
    return op


def rewrite_unpack_output(op, arch, nng):
    tens = op.outputs[0]
    if op.run_on_npu and op.type == Op.Unpack:
        # Unpack is also referred to as Unstack
        axis = int(op.attrs["axis"])
        if axis < 0:  # Convert to positive axis
            axis = len(op.inputs[0].shape) + 1 + axis
        op.type = Op.UnpackReshaped
        desired_output_shape = tens.shape[:axis] + [1] + tens.shape[axis:]

        axis_4D = axis + (4 - len(desired_output_shape))
        op.attrs["split_axis_4D"] = [axis_4D] * len(op.outputs)

        for idx, out_tens in enumerate(op.outputs):
            op.ofm_shapes[idx] = Shape4D(desired_output_shape)
    return op


def add_padding_fields(op, arch, nng):
    if op.run_on_npu:
        if "padding" in op.attrs:
            input_shape = op.ifm_shapes[0]
            output_shape = op.ofm_shapes[0]
            if op.type.is_conv2d_op() or op.type.is_depthwise_conv2d_op():
                kernel_size = op.inputs[1].shape[:2]
            elif op.type.is_pool_op() or op.type.npu_block_type == NpuBlockType.ReduceSum:
                kernel_size = op.attrs["ksize"][1:3]
            else:
                raise UnsupportedFeatureError(f"Unknown operation that uses padding: {optype_to_builtintype(op.type)}")

            if op.type == Op.Conv2DBackpropInputSwitchedBias and op.ifm_resampling_mode == resampling_mode.TRANSPOSE:
                # Transpose with upscale
                padding, skirt = calc_upscaled_padding_and_skirt(
                    op.attrs["padding"],
                    kernel_size,
                    op.attrs["strides"],
                    input_shape,
                    output_shape.height // input_shape.height,
                    output_shape.width // input_shape.width,
                )
            else:
                padding, skirt = calc_padding_and_skirt(
                    op.attrs["padding"],
                    op.kernel,
                    input_shape,
                    op.attrs.get("explicit_padding"),
                )

            op.attrs["explicit_padding"] = padding
            op.attrs["skirt"] = skirt

    return op


def reorder_depthwise_weights(op: Operation, arch, nng) -> Operation:
    if op.type.is_depthwise_conv2d_op():
        weight_tensor = op.inputs[1]
        if not weight_tensor.weight_transpose_depthwise:
            weight_tensor.values = np.transpose(weight_tensor.values, (0, 1, 3, 2))
            weight_tensor.set_all_shapes(list(weight_tensor.values.shape))
            weight_tensor.weight_transpose_depthwise = True

    return op


def convert_avg_pool_to_conv2d(op: Operation, arch, nng) -> Operation:
    """Convert strided Average Pools with stride >= 4 to Conv2D."""
    if op.type != Op.AvgPool:
        return op

    stride_x, stride_y = op.get_kernel_stride()
    # For strides <= 3 no optimization is needed
    if stride_x <= 3:
        return op
    h, w = op.attrs["filter_height"], op.attrs["filter_width"]
    inputs = op.inputs[0]
    shape = inputs.shape

    # Set necessary conv2d attributes
    op.attrs.update(
        {
            "stride_h": stride_y,
            "stride_w": stride_x,
            "dilation_h_factor": 1,
            "dilation_w_factor": 1,
            "strides": (1, stride_y, stride_x, 1),
            "dilation": (1, 1, 1, 1),
        }
    )

    # Change op type
    op.type = Op.Conv2DBias
    op.name += "_conv2d"

    op.rounding_mode = RoundingMode.AwayZero
    shape = [h, w, 1, op.ofm.shape[-1]]
    weights = np.full(shape, 1)
    quant = QuantizationParameters(scale_f32=1 / (h * w), zero_point=0)
    # Add unit weight tensor
    op.add_input_tensor(
        create_const_tensor(
            "weights",
            shape,
            inputs.dtype,
            weights,
            quantization=quant,
        ),
    )
    op.weights.values = np.reshape(op.inputs[1].values, shape)

    # Set IFM/OFM shapes after changing op type
    op.set_ifm_ofm_shapes()
    return op


def fixup_strided_conv(op: Operation, arch, nng):
    """Optimize or fixup strided Conv2DBias
    Optimization:
        Reduce, when possible, the Conv2DBias stride from N with 1 > N > 4 to 1
        by re-shaping both IFM and filter.

    Fixup:
        Introduce software support for Conv2DBias with stride_width > 4 by
        reducing it to 1, 2 or 3 (HW supported strides) when possible by
        re-shaping both IFM and filter.
    """
    if op.type != Op.Conv2DBias:
        return op
    stride_x, stride_y = op.get_kernel_stride()
    weight_tensor = op.weights
    ifm_shape = op.ifm_shapes[0]

    # Do not optimize if op is not the first in the network and stride is
    # supported by the hardware
    if op.op_index != 0 and stride_x < 4:
        return op

    resize_factor, final_stride = calc_resize_factor(ifm_shape.width, stride_x)

    def calc_filter_padding(
        ifm_padding_type: Padding | None,
        ifm_current_padding_x: int,
        post_op_stride: int,
        opt_resize_factor: int,
        filter_width: int,
        ifm_width: int,
    ) -> tuple[int, int, int, int]:
        """Calculate zero padding to be added to the filter.

        Parameters
        ----------
        ifm_padding_type : Padding or None
            The padding type that is applied to the IFM.
        ifm_current_padding_x : int
            Padding amount that is added to the IFM before optimization.
        post_op_stride : int
            The final stride once optimization is performed.
        opt_resize_factor : int
            The factor by which the stride will be reduced.
            E.g. opt_resize_factor = 2 on a stride of 4 will produce
            a stride of 2 after the optimization
        filter_width : int
            Width of the filter before optimization.
        ifm_width : int
            Width of the IFM before optimization

        Returns
        -------
        padding : tuple[int, int, int, int]
            A tuple with the ammount of padding on each side (top, left, bottom, right)
        """
        padding_size = 0
        padding = (0, 0, 0, 0)
        if ifm_padding_type and ifm_padding_type != Padding.VALID:
            # Compute padding size for the filter that guarantees that HW padding added to IFM matches
            # before and after the optimization is performed
            expected_filter_size = 0
            pre_opt_stride = post_op_stride * opt_resize_factor
            post_opt_ifm_width = ifm_width // opt_resize_factor
            # Compute the total expected filter size post optimization that ensures that the same HW padding
            # is added to IFM.
            # There are two ways of calculating required filter size depending on whether IFM width is divisible
            # by stride width or not. These approaches match the cases used to calculate HW padding in
            # needed_total_padding method.
            if ifm_width % pre_opt_stride == 0:
                expected_filter_size = ifm_current_padding_x + post_op_stride
            else:
                expected_filter_size = ifm_current_padding_x + (post_opt_ifm_width % post_op_stride)
            # Compute padding size from expected filter size
            padding_size = expected_filter_size * opt_resize_factor - filter_width

            if ifm_current_padding_x == 0:
                # If no HW padding is added to IFM, divide filter padding between left and right following
                # the same strategy as the reference.
                padding_left = padding_size // 2
            else:
                # If HW padding is added to IFM, split padding for the filter so that left padding and right padding
                # are proportional to left and right HW padding.
                left_hw_padding = ifm_current_padding_x // 2
                # Compute filter padding
                padding_left = padding_size // ifm_current_padding_x * left_hw_padding
            padding = (0, padding_left, 0, padding_size - padding_left)

        # Check if filter width is divisible by the stride width (required for optimization)
        # If filter width is not divisible by stride width and no HW padding is added to IFM, compute
        # filter padding required for the filter width to be divisible by the stride width and apply it as right
        # padding.
        if filter_width % opt_resize_factor != 0 and (padding_size == 0 or ifm_current_padding_x == 0):
            padding_size = opt_resize_factor - (filter_width % opt_resize_factor)
            # Add padding zeros to the right
            padding = (0, 0, 0, padding_size)

        return padding

    # Compute the depth of the IFM once the strided Conv2D is optimised
    post_opt_ifm_depth = ifm_shape.depth * resize_factor

    if stride_x > 1 and (post_opt_ifm_depth <= 8 or stride_x > 3) and resize_factor != 1 and weight_tensor is not None:
        k_w, _ = op.get_kernel_size()
        weight_shape = weight_tensor.shape

        padding_type = op.attrs.get("padding", None)
        if padding_type in (None, Padding.EXPLICIT, Padding.TILE):
            return op
        # Compute current padding as if IFM padding is SAME
        curr_padding_x = needed_total_padding(ifm_shape.width, stride_x, k_w)
        # Compute the padding needed on the filter for the optimisation
        _, left_filter_padding, _, right_filter_padding = calc_filter_padding(
            padding_type, curr_padding_x, final_stride, resize_factor, k_w, ifm_shape.width
        )
        total_horizontal_padding = left_filter_padding + right_filter_padding
        # If IFM padding is enabled, check if pre-opt and post-opt padding is
        # the same while taking into consideration the extra filter padding.
        if padding_type == Padding.SAME:
            optimised_padding_x = needed_total_padding(
                ifm_shape.width // resize_factor, final_stride, (k_w + 1 + total_horizontal_padding) // resize_factor
            )
            if curr_padding_x != optimised_padding_x:
                # Horizontal padding would become different after optimisation; this would not work
                return op

        # Resize IFM
        op.ifm_shapes[0] = Shape4D(
            [ifm_shape.batch, ifm_shape.height, ifm_shape.width // resize_factor, ifm_shape.depth * resize_factor]
        )

        # Compute list of 0 padding for each dimensions of the filter
        filter_dimension_padding = [(0, 0) for _ in weight_tensor.shape]
        # Update padding for filter width with computed padding
        filter_dimension_padding[1] = (left_filter_padding, right_filter_padding)
        # Add padding to the filter
        zero_point = weight_tensor.quantization.zero_point
        padding_constant = zero_point if np.isscalar(zero_point) else 0
        padded_filter_tensor = np.pad(weight_tensor.values, filter_dimension_padding, constant_values=padding_constant)
        weight_shape[1] = padded_filter_tensor.shape[1]
        weight_tensor.values = padded_filter_tensor
        # Change weight shape based on stride_x
        weight_shape[1] //= resize_factor
        weight_shape[2] *= resize_factor

        weight_tensor.values = np.reshape(weight_tensor.values, weight_shape)
        weight_tensor.set_all_shapes(weight_shape)
        # If multiple copies of the weights are used, we could avoid
        # them having the same address by changing the value_id
        weight_tensor.value_id = uuid.uuid4()

        # Strides
        stride_x = final_stride
        op.attrs.update({"stride_w": stride_x, "stride_h": stride_y, "strides": (1, stride_y, stride_x, 1)})

    ofm_shape = op.ofm_shapes[0]
    if ofm_shape.height == 1 or ofm_shape.width == 1:
        # If height or width is 1 no stride is done in y or x direction and stride value can be set to 1
        # Before forcing kernel stride to 1 make sure to calculate the correct padding since it is
        # based on the original kernel stride
        padding, _ = calc_padding_and_skirt(
            op.attrs["padding"],
            op.kernel,
            ifm_shape,
            op.attrs.get("explicit_padding"),
        )
        # Use explicit padding so it is not recalculated later with the wrong kernel stride
        op.attrs["padding"] = Padding.EXPLICIT
        op.attrs["explicit_padding"] = padding

        stride_y = 1 if ofm_shape.height == 1 else stride_y
        stride_x = 1 if ofm_shape.width == 1 else stride_x

        op.attrs.update({"stride_w": stride_x, "stride_h": stride_y, "strides": (1, stride_y, stride_x, 1)})

    return op


def convert_conv_to_fc(op: Operation, arch, nng) -> Operation:
    """Convert 1x1 Conv2D that behave like FullyConnected to FullyConnected, since they don't need any weight
    buffering.
    """
    # Conv 1x1 can be equivalent to Fully Connected.
    # (Weights dont need to be reloaded for convs when IFM H and W are 1)
    if op.type == Op.Conv2DBias:
        h = op.ifm_shapes[0].height
        w = op.ifm_shapes[0].width
        kh, kw, _, _ = op.inputs[1].shape
        if h == 1 and w == 1 and kh == 1 and kw == 1:
            # Overwrite this op as a Fully Connected Op
            op.name += "_fc"
            op.type = Op.FullyConnected
            op.attrs = {
                "weights_format": 0,
            }
            # Reshape Weights to be 2D. HWIO becomes just IO (as H and W are 1, they can just be dropped)
            weight_tensor = op.inputs[1]
            weight_tensor.values = weight_tensor.values.squeeze(axis=(0, 1))
            weight_tensor.set_all_shapes(list(weight_tensor.values.shape))

            DebugDatabase.add_optimised(op, op)
    return op


def fixup_relus_with_differing_ifm_ofm_scaling(op: Operation, arch, nng) -> Operation:
    """Fixup Relu with different IFM and OFM to allow fusing by adding its own primary op."""
    if op.run_on_npu and op.type.is_relu_op():
        ifm = op.inputs[0]
        ofm = op.outputs[0]
        # Relu with differing IFM and OFM scaling cannot be fused with another primary op
        # and requires its own to be inserted
        if not check_quantized_tens_scaling_equal(ifm, ofm):
            # Override this op with its own primary op (avgpool)
            relu_fused_op = create_avgpool_nop(op.name + "_avgpool")
            # And fuse the original activation function to it
            relu_fused_op.activation = create_activation_function(op.type)
            # Add explicit rescaling
            rescale = ifm.quantization.scale_f32 / ofm.quantization.scale_f32
            multiplier, shift = scaling.quantise_scale(rescale)
            relu_fused_op.explicit_scaling = ExplicitScaling(False, [shift], [multiplier])
            # Tidy up and assign the ifm and ofm to the new op
            ifm.consumer_list.remove(op)

            relu_fused_op.add_input_tensor(ifm)
            relu_fused_op.set_output_tensor(ofm)
            relu_fused_op.set_ifm_ofm_shapes()
            op = relu_fused_op
    return op


def convert_lstm(op: Operation, arch, nng) -> Operation:
    """Convert LSTM op into its basic opearations to allow for support on NPU."""
    if op.type == Op.UnidirectionalSequenceLstm:
        lstm = Lstm(op)
        op = lstm.get_graph()
    return op


def convert_softmax(op: Operation, arch, nng) -> Operation:
    """Convert Softmax op into its basic operations to allow for support on NPU."""
    if op.type == Op.Softmax and op.run_on_npu:
        softmax = SoftMax(op)
        op = softmax.get_graph()
    return op


def convert_prelu(op: Operation, arch, nng) -> Operation:
    """Convert PReLU op to other ops based on alpha values to allow for support on NPU."""
    if op.type == Op.Prelu:
        ifm, alpha, ofm = op.get_ifm_ifm2_ofm()
        if None in (ifm, alpha, ofm):
            return op

        if alpha.values is not None:
            # If const alpha check for possible optimisations
            alpha_zp = alpha.quantization.zero_point
            alpha_scale = alpha.quantization.scale_f32
            # If all alpha values are the same the PReLU can be converted to LeakyRelu
            alpha_min = (alpha.values.min().astype(int) - alpha_zp) * alpha_scale
            alpha_max = (alpha.values.max().astype(int) - alpha_zp) * alpha_scale
            if alpha_min == alpha_max:
                # or even a Relu
                if alpha_min == 0:
                    new_op = Op.Relu
                else:
                    new_op = Op.LeakyRelu
                    op.attrs["alpha"] = alpha_min
                    # setup alpha_scaling for bit exact result
                    ifm_scale = ifm.quantization.scale_f32
                    ofm_scale = ofm.quantization.scale_f32
                    alpha_scale, alpha_shift = scaling.elementwise_mul_scale(ifm_scale, alpha_scale, ofm_scale)
                    op.attrs["alpha_scaling"] = (alpha.values.min() - alpha_zp, alpha_scale, alpha_shift)
                # Change op type
                op.type = new_op
                op.name = op.name.replace("Prelu", new_op.name)
                del op.inputs[1]  # Remove alpha tensor
                return op
            elif alpha_max < 1:
                # If alpha_max is less than 1 convert PReLU to Max(alpha * IFM, identity * IFM)
                # Multiply with alpha tensor
                mul_alpha = Operation(Op.Mul, op.name + "_mul_alpha")
                mul_alpha.add_input_tensor(ifm)
                mul_alpha.add_input_tensor(alpha)
                fm_alpha = ofm.clone(op.name + "_alpha", set_unique=True)
                mul_alpha.set_output_tensor(fm_alpha)
                mul_alpha.set_ifm_ofm_shapes()
                DebugDatabase.add_optimised(op, mul_alpha)
                if check_quantized_tens_scaling_equal(ifm, ofm):
                    # No scaling is needed
                    fm_id = ifm
                else:
                    # Add multiplication with identity
                    mul_identity = Operation(Op.Mul, op.name + "_mul_identity")
                    mul_identity.add_input_tensor(ifm)
                    # Create const tensor containing identity as scalar
                    quantization = ifm.quantization.clone()
                    quantization.scale_f32 = np.float32(1)
                    quantization.zero_point = 0
                    one = create_const_tensor("one_const", [], ifm.dtype, [1], quantization=quantization)
                    mul_identity.add_input_tensor(one)
                    # Make sure that fm_id is allocated to a different address than fm_alpha
                    fm_id = ofm.clone(op.name + "_id", set_unique=True)
                    mul_identity.set_output_tensor(fm_id)
                    mul_identity.set_ifm_ofm_shapes()
                    DebugDatabase.add_optimised(op, mul_identity)

                # Combine scaled and alpha multiplied values
                max_op = Operation(Op.Maximum, op.name + "_max")
                max_op.add_input_tensor(fm_alpha)
                max_op.add_input_tensor(fm_id)
                max_op.set_output_tensor(ofm)
                max_op.set_ifm_ofm_shapes()

                DebugDatabase.add_optimised(op, max_op)
                ifm.consumer_list.remove(op)
                return max_op

        # Catch all PReLU conversion for the cases that could not be optimised above
        no_scale_quant = ifm.quantization.clone()
        no_scale_quant.scale_f32 = None
        no_scale_quant.zero_point = 0
        zero = create_const_tensor("zero_const", [], ifm.dtype, [0], quantization=no_scale_quant)

        # Select values < 0
        min_op = Operation(Op.Minimum, op.name + "_min")
        min_op.add_input_tensor(ifm)
        min_op.add_input_tensor(zero)
        fm_negative = ifm.clone(op.name + "_negative", set_unique=True)
        min_op.set_output_tensor(fm_negative)
        min_op.set_ifm_ofm_shapes()
        DebugDatabase.add_optimised(op, min_op)

        # and multiply with alpha tensor
        mul_alpha = Operation(Op.Mul, op.name + "_mul_alpha")
        mul_alpha.add_input_tensor(fm_negative)
        mul_alpha.add_input_tensor(alpha)
        fm_alpha = ofm.clone(op.name + "_negative_alpha", set_unique=True)
        mul_alpha.set_output_tensor(fm_alpha)
        mul_alpha.set_ifm_ofm_shapes()
        DebugDatabase.add_optimised(op, mul_alpha)

        # Select (and scale) values > 0
        relu_op = Operation(Op.Relu, op.name + "_relu")
        relu_op.add_input_tensor(ifm)
        fm_scaled = ofm.clone(op.name + "_positive_scaled", set_unique=True)
        relu_op.set_output_tensor(fm_scaled)
        relu_op.set_ifm_ofm_shapes()
        DebugDatabase.add_optimised(op, relu_op)

        # Add scaled and alpha multiplied values (without scaling)
        add_op = Operation(Op.Add, op.name + "_add")
        add_op.explicit_scaling = ExplicitScaling(False, shift=[0], multiplier=[1])  # No scaling
        add_op.add_input_tensor(fm_alpha)
        add_op.add_input_tensor(fm_scaled)
        add_op.set_output_tensor(ofm)
        add_op.set_ifm_ofm_shapes()

        DebugDatabase.add_optimised(op, add_op)
        ifm.consumer_list.remove(op)
        op = add_op

    return op


def convert_mul_max_to_abs_or_lrelu(op: Operation, arch, nng) -> Operation:
    r"""Whenever there is a subgraph with this topology:

    Input    X   For X = -1 or X > 0
    |   \   /    This subgraph can be replaced with either
    |    Mul     an Abs (if X = -1) or a LeakyReLU (if X > 0)
    |   /
    Max
    """

    if op.type == Op.Maximum:
        # finds the Mul input(s) to the Max
        muls = [i for i in op.inputs if i.ops[0].type == Op.Mul]
        if len(muls) == 1:
            mul = muls[0].ops[0]
        elif len(muls) == 2:
            # In the case both inputs are Muls, find the one with the same input as the Max
            mul_ifms = [m for m in muls if len(set(op.inputs + m.ops[0].inputs)) == 1]
            if len(mul_ifms):
                mul = mul_ifms[0].ops[0]
            else:
                # Not using same input
                return op
        else:
            # No Mul inputs
            return op

        # make sure the Mul doesn't have any other consumers
        mul_ofm = mul.outputs[0]
        if len(mul_ofm.consumers()) != 1:
            return op
        # make sure the Mul doesn't have a fused activation function
        if mul.activation:
            return op
        ifm, ofm = op.get_ifm_ofm()
        if ifm is None or ofm is None:
            return op

        if ifm.dtype not in (DataType.uint8, DataType.int8) or ifm.dtype != ofm.dtype:
            return op
        if not check_quantized_tens_scaling_equal(ifm, ofm) or not check_quantized_tens_scaling_equal(ifm, mul_ofm):
            # rewrite to LeakyRelu currently only makes sense if the quantization is identical
            return op

        # finds the branched input that goes to both the Max and the Mul
        shared = set(op.inputs) & set(mul.inputs)
        if len(shared) == 1:
            shared_in = shared.pop()
            # find the constant scalar input to the Mul
            const_tens = (set(mul.inputs) - {shared_in}).pop()
            # check that it is a scalar
            if const_tens.shape != []:
                return op
            const = const_tens.ops[0]
            # check that it is a constant
            if const.type != Op.Const:
                return op
            # Remove the Mul from the shared input's consumers
            shared_in.consumer_list.remove(mul)
        else:
            return op

        val = const.outputs[0].values
        if val >= 0:
            new_op = Op.LeakyRelu
            op.attrs["alpha"] = val
            # to produce bit exact results, the alpha is not enough;
            # save additional scaling info in attr "alpha_scale", to be used as input
            # to the LUT construction
            alpha_scalar = const_tens.values - const_tens.quantization.zero_point
            mul_ifm_scale = np.double(ifm.quantization.scale_f32)
            mul_ifm2_scale = np.double(const_tens.quantization.scale_f32)
            mul_ofm_scale = np.double(mul_ofm.quantization.scale_f32)
            alpha_scale, alpha_shift = scaling.elementwise_mul_scale(mul_ifm_scale, mul_ifm2_scale, mul_ofm_scale)
            op.attrs["alpha_scaling"] = (alpha_scalar, alpha_scale, alpha_shift)
        elif val == -1:
            new_op = Op.Abs
        else:
            return op

        op.type = new_op
        op.name = op.name.replace("Maximum", new_op.name)
        op.outputs[0].name = op.outputs[0].name.replace("Maximum", new_op.name)
        op.inputs = [shared_in]
        op.set_ifm_ofm_shapes()

        # Record optimisation in debug database
        DebugDatabase.add_optimised(op, op)

    return op


def convert_hardswish_to_lut(op: Operation, arch, nng) -> Operation:
    """Convert HardSwish to LUT to allow for support on NPU."""
    if op.type == Op.HardSwish:
        ifm, ofm = op.get_ifm_ofm()
        # Generate the LUT
        ifm_scale = np.double(ifm.quantization.scale_f32)
        ofm_scale = np.double(ofm.quantization.scale_f32)
        zp_in = ifm.quantization.zero_point
        zp_out = ofm.quantization.zero_point
        ifm_scale_hires = (1 / 128) * ifm_scale
        relu_multiplier = np.double(3 / 32768)
        out_scale, out_shift = scaling.quantise_scale(ifm_scale_hires / ofm_scale)
        relu_scale, relu_shift = scaling.quantise_scale(ifm_scale_hires / relu_multiplier)
        # Use 16bit scale
        out_scale_16 = fp_math.downscale_multiplier_int32_to_int16(out_scale)
        relu_scale_16 = fp_math.downscale_multiplier_int32_to_int16(relu_scale)

        values = []
        ix = range(256) if ifm.dtype == DataType.uint8 else range(-128, 128)
        quantized_min = min(ix)
        quantized_max = max(ix)
        for x in ix:
            input_value = x - zp_in
            input_value_hires = input_value * 128
            # Compute the input value on essentially the output scale, not shifted yet
            input_value_preshift = fp_math.saturating_rounding_mul16(input_value_hires, out_scale_16)
            # Compute the "relu-ish multiplier". This matches the code in TensorFlow Lite Micro kernel
            relu_value = np.int16(input_value_hires)
            if relu_shift < 31:
                relu_value = fp_math.shift_left16(relu_value, 30 - relu_shift)

            relu_value = fp_math.saturating_rounding_mul16(relu_value, relu_scale_16)

            if relu_shift < 31:
                relu_value = fp_math.shift_left16(relu_value, 1)

            if relu_shift > 31:
                relu_value = fp_math.rounding_divide_by_pot(relu_value, relu_shift - 31)

            # Rescaled the value into a 16bit fixedpoint relu_value in [-1, 1]
            # Now convert that to a 16bit fixedpoint value in [0, 1]
            relu_value = (relu_value + (1 << 15)) >> 1
            lut_result = fp_math.saturating_mul16(relu_value, input_value_preshift)
            shift = 31 - out_shift
            shift = -shift if shift < 0 else 0
            # Finally apply the output shift
            lut_result = fp_math.rounding_divide_by_pot(lut_result, shift) + zp_out
            lut_result = min(quantized_max, max(quantized_min, lut_result))
            values.append(lut_result)
        return convert_to_lut(op, values, "hardswish")
    return op


def convert_lrelu_to_mul_max(op, arch):
    # Converts LeakyRelu to Max(alpha * IFM, identity * IFM)
    # (the opposite of convert_mul_max_to_abs_or_lrelu)
    ifm, ofm = op.get_ifm_ofm()
    if ifm is None or ofm is None:
        return op

    alpha = np.float32(op.attrs["alpha"])
    use_mul_max = 0 < alpha < 1
    is_converted_prelu = "alpha_scaling" in op.attrs
    if use_mul_max:
        mul_ifm = ifm
        new_op = Op.Maximum
    else:
        # Need to use a different approach for alpha < 0 or alpha > 1
        no_scale_quant = ifm.quantization.clone()
        no_scale_quant.scale_f32 = None
        no_scale_quant.zero_point = 0
        zero = create_const_tensor("zero_const", [], ifm.dtype, [0], quantization=no_scale_quant)

        # Select values < 0
        min_op = Operation(Op.Minimum, op.name + "_min")
        min_op.add_input_tensor(ifm)
        min_op.add_input_tensor(zero)
        mul_ifm = ifm.clone(op.name + "_negative", set_unique=True)
        if alpha < 0 and not is_converted_prelu:
            # For negative alpha that is not from a converted PReLU we need to use
            # int32 Mul below to perform the (negative) alpha scaling
            mul_ifm.dtype = DataType.int32
        min_op.set_output_tensor(mul_ifm)
        min_op.set_ifm_ofm_shapes()
        new_op = Op.Add
        op.explicit_scaling = ExplicitScaling(False, shift=[0], multiplier=[1])  # No scaling
        DebugDatabase.add_optimised(op, min_op)

    # Add multiplication with alpha
    mul_alpha = Operation(Op.Mul, op.name + "_mul_alpha")
    mul_alpha.add_input_tensor(mul_ifm)
    # Create const tensor containing alpha as scalar
    quantization = ifm.quantization.clone()
    quantization.min = 0
    quantization.max = alpha * (quantization.quant_max - quantization.quant_min)
    quantization.zero_point = 0
    alpha_dtype = mul_ifm.dtype
    if is_converted_prelu:
        # The LeakyRelu was the result from convert_prelu and the scaling is provided
        scalar, alpha_scale, alpha_shift = op.attrs["alpha_scaling"]
        mul_alpha.explicit_scaling = ExplicitScaling(False, [alpha_shift], [alpha_scale])
    elif alpha == 0 or np.isinf(1 / alpha):
        # Handling of alpha near or at zero
        quantization.scale_f32 = np.float32(1)
        scalar = 0
    else:
        quantization.scale_f32 = alpha
        if alpha_dtype == DataType.int32:
            # When the datatype is int32 (alpha negative) we need to do the scaling with the multiplication
            scalar, _ = scaling.elementwise_mul_scale(ifm.quantization.scale_f32, alpha, ofm.quantization.scale_f32)
        else:
            scalar = 1
    alpha_tens = create_const_tensor(op.name + "_alpha_scalar", [1], alpha_dtype, [scalar], quantization=quantization)
    mul_alpha.add_input_tensor(alpha_tens)
    fm_alpha = ofm.clone(op.name + "_alpha", set_unique=True)
    mul_alpha.set_output_tensor(fm_alpha)
    mul_alpha.set_ifm_ofm_shapes()
    DebugDatabase.add_optimised(op, mul_alpha)

    if not use_mul_max:
        relu_op = Operation(Op.Relu, op.name + "_relu")
        relu_op.add_input_tensor(ifm)
        fm_id = ofm.clone(op.name + "_positive_scaled", set_unique=True)
        relu_op.set_output_tensor(fm_id)
        relu_op.set_ifm_ofm_shapes()
        DebugDatabase.add_optimised(op, relu_op)
    elif check_quantized_tens_scaling_equal(ifm, ofm):
        # No identity multiplication is needed
        fm_id = ifm
    else:
        # Add multiplication with identity
        mul_identity = Operation(Op.Mul, op.name + "_mul_identity")
        mul_identity.add_input_tensor(ifm)
        # Create const tensor containing identity as scalar
        quantization = ifm.quantization.clone()
        quantization.min = 0
        quantization.max = quantization.quant_max - quantization.quant_min
        quantization.scale_f32 = np.float32(1)
        quantization.zero_point = 0
        identity_tens = create_const_tensor(op.name + "_id_scalar", [], ifm.dtype, [1], quantization=quantization)
        mul_identity.add_input_tensor(identity_tens)
        # Make sure that fm_id is allocated to a different address than fm_alpha
        fm_id = ofm.clone(op.name + "_id", set_unique=True)
        mul_identity.set_output_tensor(fm_id)
        mul_identity.set_ifm_ofm_shapes()
        DebugDatabase.add_optimised(op, mul_identity)

    # Convert LeakyRelu to Max, add the results of the multiplication(s) as inputs
    op.type = new_op
    op.name = op.name.replace("LeakyRelu", new_op.name)
    op.inputs = []
    ifm.consumer_list.remove(op)
    op.add_input_tensor(fm_alpha)
    op.add_input_tensor(fm_id)
    op.set_ifm_ofm_shapes()

    DebugDatabase.add_optimised(op, op)
    return op


def convert_to_lut8(op, fn, fn_name):
    # Converts op to a no-op + int8/uint8 LUT which is generated with the given function.
    # fn is a function(real) -> real
    ifm, ofm = op.get_ifm_ofm()
    if ifm.dtype not in (DataType.uint8, DataType.int8) or ifm.dtype != ofm.dtype:
        return op
    # Generate the LUT
    ifm_scale = np.double(ifm.quantization.scale_f32)
    ofm_scale = np.double(ofm.quantization.scale_f32)
    zp_in = ifm.quantization.zero_point
    zp_out = ofm.quantization.zero_point
    values = []
    ix = range(256) if ifm.dtype == DataType.uint8 else range(-128, 128)
    quantized_min = min(ix)
    quantized_max = max(ix)
    for x in ix:
        x_real = ifm_scale * (x - zp_in)
        y_real = fn(x_real)
        lut_result = round_away_zero(zp_out + y_real / ofm_scale)
        lut_result = min(quantized_max, max(quantized_min, lut_result))
        values.append(lut_result)
    return convert_to_lut(op, values, fn_name)


def convert_lrelu_to_lut(op, arch):
    ifm, ofm = op.get_ifm_ofm()
    # Generate the LUT
    alpha = op.attrs["alpha"]
    ifm_scale = np.double(ifm.quantization.scale_f32)
    ofm_scale = np.double(ofm.quantization.scale_f32)
    zp_in = ifm.quantization.zero_point
    zp_out = ofm.quantization.zero_point
    identity_scale, identity_shift = scaling.elementwise_mul_scale(ifm_scale, 1, ofm_scale)
    alpha_scalar = 1
    alpha_scale, alpha_shift = scaling.elementwise_mul_scale(ifm_scale, alpha, ofm_scale)
    if "alpha_scaling" in op.attrs:
        # The LeakyRelu was the result from convert_mul_max_to_abs_or_lrelu
        alpha_scalar, alpha_scale, alpha_shift = op.attrs["alpha_scaling"]
    values = []
    ix = range(256) if ifm.dtype == DataType.uint8 else range(-128, 128)
    quantized_min = min(ix)
    quantized_max = max(ix)
    for x in ix:
        if x < zp_in:
            lut_result = zp_out + fp_math.multiply_by_quantized_multiplier(
                alpha_scalar * (x - zp_in), alpha_scale, alpha_shift
            )
        else:
            lut_result = zp_out + fp_math.multiply_by_quantized_multiplier(x - zp_in, identity_scale, identity_shift)
        lut_result = min(quantized_max, max(quantized_min, lut_result))
        values.append(lut_result)
    return convert_to_lut(op, values, "lrelu")


def convert_lrelu(op: Operation, arch, nng) -> Operation:
    """Convert LeakyRelu to a LUT based solution if possible, otherwise a mul + max."""
    if op.type != Op.LeakyRelu:
        return op
    ifm, ofm = op.get_ifm_ofm()
    if ifm is None or ofm is None:
        return op
    alpha = op.attrs["alpha"]
    if alpha == 0:
        # When alpha is 0 the opertion can be converted to a ReLU
        op.type = Op.Relu
        op.name = op.name.replace("LeakyRelu", op.type.name)
        return op
    if ifm.dtype in (DataType.uint8, DataType.int8) and ifm.dtype == ofm.dtype:
        # use LUT for int8/uint8
        return convert_lrelu_to_lut(op, arch)
    if check_quantized_tens_scaling_equal(ifm, ofm) and ifm.dtype == ofm.dtype == DataType.int16 and alpha > 0:
        # use LeakyRelu unmodified for int16 with equal input/output scaling and positive alpha
        return op
    return convert_lrelu_to_mul_max(op, arch)


def convert_tanh_sigmoid_to_lut(op: Operation, arch, nng) -> Operation:
    """Convert int8/uint8 Sigmoid and Tanh to a LUT based solution."""
    if op.type == Op.Sigmoid:
        return convert_to_lut8(op, clamp_sigmoid, "sigmoid")
    elif op.type == Op.Tanh:
        return convert_to_lut8(op, math.tanh, "tanh")
    return op


def convert_quantize(op: Operation, arch, nng) -> Operation:
    """Convert Quantize to Avgpool. This conversion only works for int-to-int re-quantization and
    not to/from floats. Therefor, this rewrite should only run after the supported ops check to
    avoid rewriting ops that will run on CPU."""
    if op.type == Op.Quantize:
        # Create a new AvgPool op and steal its attrs, then reuse the original op with different type
        avgpool_op = create_avgpool_nop(op.name + "_avgpool")
        op.type = Op.AvgPool
        op.attrs = avgpool_op.attrs.copy()

        DebugDatabase.add_optimised(op, op)

    return op


def fuse_activation_function_with_prev(op, arch, nng):
    # if op is a no-op: attempts to move the activation function to the preceding op
    if not op.attrs.get("is_nop", False) or op.activation is None:
        return op
    ifm, ofm = op.get_ifm_ofm()
    if ifm is None or ofm is None:
        return op
    # finds the input(s) to the operation
    prev_op = ifm.ops[0]
    # Note: the below checks on prev_op require that a first optimize pass on the full graph has been performed
    fuse = (
        prev_op.run_on_npu
        and prev_op.type != Op.Memcpy
        and prev_op.type.npu_block_type != NpuBlockType.Default
        and len(ifm.ops) == 1
        and len(prev_op.outputs[0].consumers()) == 1
        and prev_op.activation is None
    )
    if op.activation_lut is not None and arch.shram_reserved_unused_banks == 0:
        # TODO: if SHRAM LUT space is shared with SHRAM ACC (32, 64 MAC),
        # LUT currently only works correctly for elementwise ops
        fuse = False
    if not fuse:
        return op
    # Move the fused activation function + corresponding info to prev_op
    prev_op.activation = op.activation
    prev_op.forced_output_quantization = op.forced_output_quantization
    if op.activation_lut is not None:
        prev_op.set_activation_lut(op.activation_lut)
    # Bypass op
    prev_op.set_output_tensor(ofm)
    DebugDatabase.add_optimised(prev_op, prev_op)
    return op


def _leading_pad_ok(leading_pad, stride, kernel_size):
    # If kernel size // 2 > stride, then (left, top) padding must be a multiple of stride,
    # otherwise replacing PAD by hardware padding would iterate the wrong IFM rows/columns
    max_size = kernel_size // 2
    return leading_pad == max_size or max_size <= stride or leading_pad % stride == 0


def replace_pad_by_hw_pad(op: Operation, arch, nng) -> Operation:
    """
    Tries to completely remove a PAD operator by using hardware padding.
    E.g. a PAD operation that pads 1, followed by a CONV with VALID padding and kernel size 3
    is rewritten such that the PAD is removed, and the CONV uses SAME padding.
    Converts tens1 -> PAD -> tens2 -> CONV to tens1 -> CONV
    if both operations can be run on the NPU.
    This is the most efficient way to implement PAD, but cannot be done for all pad sizes.
    """
    if (
        (op.type.is_conv2d_op() or op.type.is_depthwise_conv2d_op() or op.type.is_avgpool_op())
        and op.type not in (Op.Conv2DBackpropInput, Op.Conv2DBackpropInputSwitchedBias)
        and op.run_on_npu
        and op.attrs["padding"] == Padding.VALID
    ):
        pad_op = op.ifm.ops[0]
        if pad_op.type != Op.Pad or not pad_op.run_on_npu:
            return op
        if pad_op.ifm.dtype != pad_op.ofm.dtype or not check_quantized_tens_scaling_equal(pad_op.ofm, pad_op.ifm):
            return op
        top, left, bottom, right = get_pad_values_from_input(pad_op.inputs[1].values)
        k = op.kernel
        k_w, k_h = k.dilated_wh()

        # Check if the PAD operator can be replaced by hardware padding
        if left > k_w // 2 or right > k_w // 2 or top > k_h // 2 or bottom > k_h // 2:
            # Too much padding, it would require hardware padding to actually insert zeros
            return op
        if not _leading_pad_ok(top, k.stride.y, k_h) or not _leading_pad_ok(left, k.stride.x, k_w):
            return op

        if op.type.is_avgpool_op():
            # For average pool, hardware padding can only be used if padding is 0 or kernel size / 2
            for pad, k_size in (
                (left, k_w),
                (right, k_w),
                (top, k_h),
                (bottom, k_h),
            ):
                if pad not in (0, k_size // 2):
                    return op
            # Average pool is converted to depthwise, because NPU average pool + same padding
            # has a special implementation that is different from PAD followed by average pool with
            # valid padding.
            k_w, k_h = op.kernel.width, op.kernel.height
            ifm = op.ifm
            # Remember other inputs
            other_inputs = op.inputs[1:]
            # Create a weight tensor, all weights are set to 1/(kernel width * kernel height)
            quantization = QuantizationParameters(0.0, 255.0)
            quantization.scale_f32 = 1.0 / (k_w * k_h)
            quantization.zero_point = 0
            shape = [k_h, k_w, 1, op.ofm.shape[-1]]
            weights = np.full(shape, 1)

            weight_tens = create_const_tensor(
                op.name + "_weights",
                shape,
                op.ifm.dtype,
                weights,
                purpose=TensorPurpose.Weights,
                quantization=quantization,
            )
            weight_tens.values = weights
            op.type = Op.DepthwiseConv2DBias
            op.inputs = []
            op.add_input_tensor(ifm)
            op.add_input_tensor(weight_tens)

            if op.ifm.dtype == DataType.uint8:
                op.rounding_mode = RoundingMode.HalfUp

                # Add bias tensor, all biases set to 0
                op.inputs.append(None)
                fixup_bias_tensors(op, arch, nng, DataType.int32)

            else:
                op.rounding_mode = RoundingMode.AwayZero

                # The DepthwiseConv needs to be performed with the IFM zero point set appropriately so that the correct
                # pad values are used. However, in order to use the rounding away from zero mode the zero point needs to
                # have been removed so that the zero point is at zero. This is done by adding a kernel sized amount of
                # the zero point as a bias. The datatype of the bias needs to be set to int32, even for an int16 IFM,
                # because this will cause full precision scaling to be used (see weight compression). Finally, the OFM
                # zero point will need forcing to zero (as it has already been removed)
                nr_biases = op.inputs[1].shape[-1]
                bias_values = [op.ifm.quantization.zero_point * k_h * k_w] * nr_biases
                bias_tensor = create_const_tensor(op.name + "_bias", [nr_biases], DataType.int32, bias_values)
                op.add_input_tensor(bias_tensor)

            # Add other inputs
            op.inputs.extend(other_inputs)

        # Bypass the PAD operator
        op.set_input_tensor(pad_op.ifm, 0)
        # Adjust the padding attributes of the convolution operator
        op.attrs["padding"] = Padding.EXPLICIT
        op.attrs["explicit_padding"] = (top, left, bottom, right)
        op.set_ifm_ofm_shapes()
        DebugDatabase.add_optimised(op, op)

    return op

# If the op pad channel and batch at the same time, we will split to two pad ops
# one for channel and the other for batch, and then will convert pad to concat
def split_pad_to_sub_pad(op, arch, nng):
    if not op.type == Op.Pad or not op.run_on_npu:
        return op

    inp, pad_tensor = op.inputs
    if len(pad_tensor.values) == 3 or sum(pad_tensor.values[-1, :]) == 0 or sum(pad_tensor.values[0, :]) == 0:
        return op

    pad_sub = op.clone("_sub")

    dtype = op.outputs[0].dtype
    out_shape = op.outputs[0].shape.copy()
    out_shape[0] -= sum(pad_tensor.values[0])
    pad_sub_out = Tensor(out_shape, dtype, f"{op.outputs[0].name}_sub")
    pad_sub_out.quantization = op.outputs[0].quantization

    pad_value = pad_tensor.values.copy()
    pad_shape = list(pad_tensor.shape)
    pad_dtype = pad_tensor.dtype
    quantization = pad_tensor.quantization
    pad_tensor2 = create_const_tensor(
            f"{pad_tensor.name}_sub", pad_shape, pad_dtype, pad_value, quantization=quantization)

    pad_tensor.values[3] = [0, 0]
    pad_tensor2.values[0] = [0, 0]

    op.set_input_tensor(pad_sub_out, 0)
    pad_sub.set_output_tensor(pad_sub_out)
    pad_sub.set_input_tensor(pad_tensor2, 1)

    op.set_ifm_ofm_shapes()
    pad_sub.set_ifm_ofm_shapes()

    return op

# The pad tensor can only pad width and height, we use concat to replace
def convert_pad_to_concat(op, arch, nng):
    if not op.type == Op.Pad or not op.run_on_npu:
        return op

    inp, pad_tensor = op.inputs
    if sum(pad_tensor.values[-1, :]) != 0:
        axis = -1
    elif sum(pad_tensor.values[0, :]) != 0:
        axis = 0
    else:
        return op

    outputs = op.outputs
    dtype = op.ifm.dtype
    quantization = inp.quantization
    left_size, right_size = pad_tensor.values[axis, :]

    input_tensors = [inp]
    if left_size != 0:
        shape = inp.shape.copy()
        shape[axis] = left_size
        pad_value = np.prod(shape) * [quantization.zero_point]
        left_tens = create_const_tensor(
                f"{op.name}_left", shape, dtype, pad_value, quantization=quantization)
        input_tensors.insert(0, left_tens)

    if right_size !=0:
        shape = inp.shape.copy()
        shape[axis] = right_size
        pad_value = np.prod(shape) * [quantization.zero_point]
        right_tens = create_const_tensor(
                f"{op.name}_right", shape, dtype, pad_value, quantization=quantization)
        input_tensors.append(right_tens)

    op.type = Op.ConcatTFLite
    op.name = f"{op.name}_concat"
    op.attrs["axis"] = axis
    op.inputs = input_tensors
    op.set_ifm_ofm_shapes()

    return op

def convert_pad(op: Operation, arch, nng):
    """
    Rewrites PAD operator to an average pool that copies the IFM to the OFM
    + up to 4 average pool operators that fill the OFM with zeros at the borders.
    This is done as fall-back for the PAD operators that remain after replace_pad_by_hw_pad
    """
    if op.type != Op.Pad or not op.run_on_npu:
        return op
    top, left, bottom, right = get_pad_values_from_input(op.inputs[1].values)

    ifm = op.ifm
    assert ifm is not None
    ifm_shape = op.ifm_shapes[0]
    ofm = op.ofm
    assert ofm is not None
    ofm.ops = []
    ofm_shape = op.ofm_shapes[0]

    # Average pool op that copies IFM to the right place inside the OFM
    shp0 = Shape4D(0, 0, 0, 0)
    shp_top = shp0.with_height(top)
    avgpool_op = create_avg_pool_for_concat(op, op.name + "_main", ifm, ifm_shape, shp_top.with_width(left))
    avgpool_op.activation = op.activation
    quant = ofm.quantization
    pad_value = quant.zero_point
    # Add operations that fill the borders of the OFM
    if top > 0:
        shape = Shape4D(1, top, ofm_shape.width, ofm_shape.depth)
        zero_tens = create_const_tensor(
            op.name + "_top", shape.as_list(), ofm.dtype, shape.elements() * [pad_value], quantization=quant
        )
        # If top/bottom or left/right are equal, the const tensors can be allocated to the same address
        zero_tens.equivalence_id = create_equivalence_id(tuple(zero_tens.values))
        create_avg_pool_for_concat(op, op.name + "_top", zero_tens, shape, shp0)
    if bottom > 0:
        shape = Shape4D(1, bottom, ofm_shape.width, ofm_shape.depth)
        zero_tens = create_const_tensor(
            op.name + "_bottom",
            shape.as_list(),
            ofm.dtype,
            shape.elements() * [pad_value],
            quantization=quant,
        )
        zero_tens.equivalence_id = create_equivalence_id(tuple(zero_tens.values))
        create_avg_pool_for_concat(
            op, op.name + "_bottom", zero_tens, shape, shp0.with_height(ofm_shape.height - bottom)
        )
    if left > 0:
        shape = Shape4D(1, ifm_shape.height, left, ofm_shape.depth)
        zero_tens = create_const_tensor(
            op.name + "_left", shape.as_list(), ofm.dtype, shape.elements() * [pad_value], quantization=quant
        )
        zero_tens.equivalence_id = create_equivalence_id(tuple(zero_tens.values))
        create_avg_pool_for_concat(op, op.name + "_left", zero_tens, shape, shp_top)
    if right > 0:
        shape = Shape4D(1, ifm_shape.height, right, ofm_shape.depth)
        zero_tens = create_const_tensor(
            op.name + "_right", shape.as_list(), ofm.dtype, shape.elements() * [pad_value], quantization=quant
        )
        zero_tens.equivalence_id = create_equivalence_id(tuple(zero_tens.values))
        create_avg_pool_for_concat(
            op, op.name + "_right", zero_tens, shape, shp_top.with_width(ofm_shape.width - right)
        )

    op.type = Op.ConcatTFLite
    return avgpool_op


def fixup_bias_tensors(op: Operation, arch, nng, dtype=None) -> Operation:
    """Fixup ops that require a bias and don't have one by adding a bias tensor filled with zeros."""
    if op.type.needs_bias() and op.bias is None:
        # Op has no bias, add bias tensor filled with zeros
        nr_biases = op.inputs[1].shape[-1]
        bias_values = [0] * nr_biases
        # The DataType of the bias tensor can be explicitly provided or deduced from the ifm
        # DataType. Default is int32 bias for 8-bit ifms and int64 for int16 ifms.
        # For int16 the selected bias DataType will have an impact on the scaling
        # used when encoding the scales and biases later. The default mode will match the
        # refence with reduced scaling for int64 bias.
        # This means that in cases (in the graph optimiser) where DepthwiseConv2DBias
        # is used to emulate average pool int32 bias should be selected for full precision
        # int16 scaling.
        if dtype is None:
            dtype = DataType.int64 if op.ifm.dtype == DataType.int16 else DataType.int32
        bias_tensor = create_const_tensor(op.name + "_bias", [nr_biases], dtype, bias_values)
        bias_index = op.type.info.indices.biases[0]
        if bias_index < len(op.inputs):
            op.set_input_tensor(bias_tensor, bias_index)
        else:
            op.add_input_tensor(bias_tensor)

    return op


def detect_asymmetric_weights(op):
    # Check all ops (cpu and npu)
    if op.type.is_conv2d_op() or op.type.is_depthwise_conv2d_op():
        if op.ifm.dtype in (DataType.int8, DataType.int16):
            if not np.all(op.weights.quantization.zero_point == 0):
                print(f"Warning: Op {op.type} '{op.name}' has asymmetric weights.", end=" ")
                return True
    return False


def fixup_asymmetric_weights(op: Operation, arch, nng) -> Operation:
    if detect_asymmetric_weights(op):
        if op.run_on_npu:
            print("Zero points have been adjusted.")
            op.weights.quantization.zero_point *= 0
    return op


def check_asymmetric_weights(op, arch, nng):
    # This function can modify the run_on_npu flag which causes an operator to be placed on the CPU. It is usually only
    # set by the supported operator checks. Therefore, it should be run immediately after those checks to avoid the
    # possibility of other graph optimiser functions modify the operator (that is later run on the CPU)
    if detect_asymmetric_weights(op):
        if op.run_on_npu:
            print("To run the operator on Ethos-U use the option --force-symmetric-int-weights")
            op.run_on_npu = False
    return op


def fixup_or_check_asymmetric_weights(force_symmetric_int_weights):
    if force_symmetric_int_weights:
        return fixup_asymmetric_weights
    else:
        return check_asymmetric_weights


def convert_squared_difference(op, arch, nng):
    if op.type == Op.SquaredDifference and op.run_on_npu:
        ifm, ifm2, ofm = op.get_ifm_ifm2_ofm()

        identity_quant = QuantizationParameters(scale_f32=1.0, zero_point=0)

        # All the calculations/parameters same as reference kernel
        twice_max_input_scale = np.double(2.0 * max(ifm.quantization.scale_f32, ifm2.quantization.scale_f32))
        real_input1_multiplier = np.double(ifm.quantization.scale_f32) / twice_max_input_scale
        real_input2_multiplier = np.double(ifm2.quantization.scale_f32) / twice_max_input_scale

        left_shift = 0 if op.ifm.dtype == DataType.int16 else 7

        real_output_multiplier = (twice_max_input_scale * twice_max_input_scale) / (
            np.double((1 << (left_shift * 2)) * ofm.quantization.scale_f32)
        )

        input1_multiplier, input1_shift = quantise_scale(real_input1_multiplier)
        input2_multiplier, input2_shift = quantise_scale(real_input2_multiplier)
        output_multiplier, output_shift = quantise_scale(real_output_multiplier)

        input1_multiplier_const = create_const_tensor(
            op.name + "_input1_multiplier", [1], DataType.int32, [input1_multiplier], quantization=identity_quant
        )
        input2_multiplier_const = create_const_tensor(
            op.name + "_input2_multiplier", [1], DataType.int32, [input2_multiplier], quantization=identity_quant
        )
        output_multiplier_const = create_const_tensor(
            op.name + "_output_multiplier", [1], DataType.int32, [output_multiplier], quantization=identity_quant
        )

        # Convert ifm to 32 bit
        ifm_32bit_shifted = ifm.clone(suffix="_ifm_32bit_shifted", set_unique=True)
        ifm_32bit_shifted.dtype = DataType.int32
        ifm_32bit_shifted.quantization = identity_quant
        cast_op = create_cast_op(op.name + "_ifm_32bit_shifted", ifm, ifm_32bit_shifted)
        # Use explicit scaling (multiplier) for the left shift
        cast_op.explicit_scaling = ExplicitScaling(False, [0], [1 << left_shift])
        DebugDatabase.add_optimised(op, cast_op)

        # 32 bit Mul op do not scale the value so the input has to be multiplied with the "multiplier" calculated above
        ifm_scaled = ifm.clone(suffix="_scaled", set_unique=True)
        ifm_scaled.dtype = DataType.int32
        ifm_scaled.quantization = identity_quant
        mul_op = Operation(Op.Mul, op.name + "_scaled_input1")
        mul_op.add_input_tensor(ifm_32bit_shifted)
        mul_op.add_input_tensor(input1_multiplier_const)
        mul_op.set_output_tensor(ifm_scaled)
        # Use explicit scaling for the shift (multiplier not actually used for int32, but value can not be empty)
        mul_op.explicit_scaling = ExplicitScaling(False, [input1_shift], [input1_multiplier])
        mul_op.set_ifm_ofm_shapes()
        DebugDatabase.add_optimised(op, mul_op)

        # Convert ifm2 to 32 bit
        ifm2_32bit_shifted = ifm2.clone(suffix="_ifm2_32bit_shifted", set_unique=True)
        ifm2_32bit_shifted.dtype = DataType.int32
        ifm2_32bit_shifted.quantization = identity_quant
        cast_op = create_cast_op(op.name + "_ifm2_32bit_shifted", ifm2, ifm2_32bit_shifted)
        # Use explicit scaling (multiplier) for the left shift
        cast_op.explicit_scaling = ExplicitScaling(False, [0], [1 << left_shift])
        DebugDatabase.add_optimised(op, cast_op)

        # 32 bit Mul op do not scale the value so input has to be multiplied with the "multiplier" calculated above
        ifm2_scaled = ifm2.clone(suffix="_scaled", set_unique=True)
        ifm2_scaled.dtype = DataType.int32
        ifm2_scaled.quantization = identity_quant
        mul_op = Operation(Op.Mul, op.name + "_scaled_input2")
        mul_op.add_input_tensor(ifm2_32bit_shifted)
        mul_op.add_input_tensor(input2_multiplier_const)
        mul_op.set_output_tensor(ifm2_scaled)
        # Use explicit scaling for the shift (multiplier not actually used for int32, but value can not be empty)
        mul_op.explicit_scaling = ExplicitScaling(False, [input2_shift], [input2_multiplier])
        mul_op.set_ifm_ofm_shapes()
        DebugDatabase.add_optimised(op, mul_op)

        # Calculate the raw diff
        raw_diff = ifm.clone(suffix="_raw_diff", set_unique=True)
        raw_diff.dtype = DataType.int32
        raw_diff.quantization = None
        sub_op = Operation(Op.Sub, op.name + "_raw_diff")
        sub_op.add_input_tensor(ifm_scaled)
        sub_op.add_input_tensor(ifm2_scaled)
        sub_op.set_output_tensor(raw_diff)
        sub_op.set_ifm_ofm_shapes()
        DebugDatabase.add_optimised(op, sub_op)

        # Calculate the squared diff
        squared_raw = ifm.clone(suffix="_squared_raw", set_unique=True)
        squared_raw.dtype = DataType.int32
        squared_raw.quantization = None
        mul_op = Operation(Op.Mul, op.name + "_squared_raw")
        mul_op.add_input_tensor(raw_diff)
        mul_op.add_input_tensor(raw_diff)
        mul_op.set_output_tensor(squared_raw)
        mul_op.set_ifm_ofm_shapes()
        DebugDatabase.add_optimised(op, mul_op)

        # 32 bit Mul op do not scale the value so output has to be multiplied with "multiplier" calculated above
        op.set_input_tensor(squared_raw, 0)
        op.set_input_tensor(output_multiplier_const, 1)
        op.type = Op.Mul
        # Use explicit scaling for the shift (multiplier not actually used for int32, but value can not be empty)
        op.explicit_scaling = ExplicitScaling(False, [output_shift], [output_multiplier])
        op.set_ifm_ofm_shapes()
        DebugDatabase.add_optimised(op, op)

    return op


def convert_mean_to_depthwise_conv(op, arch, nng):
    """
    When h x w <= 4096     When h x w > 4096 there is a need to split into several ops.
                           Do this by splitting up h and change the read_offset/shape.
                           Below is an example where ifm is 1x190x64x1
           MEAN                                           MEAN
             |                      |-----------------------|----------------------|
    DepthwiseConv2DBias    1_DepthwiseConv2DBias   2_DepthwiseConv2DBias   3_DepthwiseConv2DBias
             |                      |                       |                     |
            MUL                     |---------ADD-----------|                     |
                                               |                                  |
                                               |----------------ADD---------------|
                                                                 |
                                                                MUL
               1_DepthwiseConv2DBias: read_offset [0, 0, 0, 0]> read_shape [1,  64, 64, 1]>
               2_DepthwiseConv2DBias: read_offset [0, 64, 0, 0]> read_shape [1,  64, 64, 1]>
               3_DepthwiseConv2DBias: read_offset [0, 128, 0, 0]> read_shape [1,  62, 64, 1]>
    """
    if op.type == Op.Mean and op.run_on_npu:
        max_kernel_size = 4096
        max_height = 64
        inp, axis = op.inputs
        dims = len(inp.shape)
        dims_ofm = len(op.ofm.shape)
        ofmq = op.ofm.quantization
        ifmq = op.ifm.quantization

        # reduce_axis[i] is true if axis i should be reduced
        if axis.shape == []:
            reduce_axis = [True if i == axis.values else False for i in range(dims)]
        else:
            reduce_axis = [True if i in axis.values else False for i in range(dims)]

        ifm_shape = inp.shape.copy()
        intermediate_shape = op.ofm.shape.copy()

        # Fix intermediate_shape when keep_dims is false
        # e.g. IFM=1xHxWxC axis=2 OFM=1xHxC, the intermediate_shape should be 1xHx1xC
        if dims_ofm < dims:
            for i in range(dims):
                if reduce_axis[i]:
                    intermediate_shape.insert(i, 1)

        # Reshape to 4D
        reduce_axis = full_shape(4, reduce_axis, False)
        ifm_shape = full_shape(4, ifm_shape, 1)
        intermediate_shape = full_shape(4, intermediate_shape, 1)

        # If all dimensions to reduce have shape 1, the operation is essentially a memcpy.
        # We can then remove the whole op by propagating ofm to previous ops
        if not any([reduce_axis[i] and ifm_shape[i] > 1 for i in range(4)]):
            op.type = Op.Memcpy
            op = bypass_memory_only_ops(op, arch, nng)
            return op

        # Support mean over depth-axis by left-shifting the C channel
        # From semantics checks we can assume that one of H,W,C has shape 1
        if reduce_axis[3] and ifm_shape[3] > 1:
            assert 1 in ifm_shape[1:], "Mean reduction over depth channel, but none of H,W,C has shape 1"
            # If W=1 reshape NxHx1xC -> NxHxCx1, else reshape Nx1xWxC -> NxWxCx1
            idx_to_del = 2 if ifm_shape[2] == 1 else 1

            # Delete axis with size 1
            del reduce_axis[idx_to_del]
            del ifm_shape[idx_to_del]
            del intermediate_shape[idx_to_del]

            # Add another element to set channel-axis to one
            reduce_axis.append(False)
            ifm_shape.append(1)
            intermediate_shape.append(1)

        # Compute kernel sizes for our convolutions
        # Batch axis is implicit as it is only supported if batch size is 1.
        h = ifm_shape[1] if reduce_axis[1] else 1
        w = ifm_shape[2] if reduce_axis[2] else 1

        num_elements_in_axis = h * w

        # If one convolution is enough, but height is greater than max kernel height
        # reshape from HxW to 1x(HxW)
        # This can only be done if the mean is computed over both H and W
        if h > max_height and num_elements_in_axis <= max_kernel_size and reduce_axis[1] and reduce_axis[2]:
            ifm_shape = [ifm_shape[0], 1, h * w, ifm_shape[3]]
            w = h * w
            h = 1

        intermediate_op = None
        height_per_conv = min(max_kernel_size // w, h)
        height_per_conv = min(height_per_conv, max_height)
        num_convs = math.ceil(h / height_per_conv)
        convs = list()

        for i in range(num_convs):
            is_last_op = i == (num_convs - 1)

            intermediate_op = op.clone(f"{op.name}_conv_{i}")

            intermediate_op.type = Op.DepthwiseConv2DBias

            # Set necessary depthwise attributes
            intermediate_op.attrs.update(
                {
                    "padding": Padding.VALID,
                    "stride_h": 1,
                    "stride_w": 1,
                    "strides": (1, 1, 1, 1),
                    "depth_multiplier": 1,
                    "channel_multiplier": 1,
                    "dilation_h_factor": 1,
                    "dilation_w_factor": 1,
                    "dilation": (1, 1, 1, 1),
                }
            )

            b, _, _, c = ifm_shape

            intermediate_tensor = op.ofm.clone(suffix=f"_conv_sum_{i}", set_unique=True)
            intermediate_tensor.dtype = DataType.int32
            intermediate_tensor.shape = intermediate_shape
            intermediate_op.set_output_tensor(intermediate_tensor)

            # as we have several convs, scaling/rounding must be done after the sum has been calculated
            intermediate_op.explicit_scaling = ExplicitScaling(False, shift=[0], multiplier=[1])

            # compute height for the kernel
            if is_last_op and h % height_per_conv != 0:
                weight_h = h % height_per_conv
            else:
                weight_h = height_per_conv

            # compute ifm read offset and shape for the convolution
            read_shape_h = weight_h if reduce_axis[1] else ifm_shape[1]
            read_shape_w = w if reduce_axis[2] else ifm_shape[2]

            intermediate_op.read_offsets[0] = Shape4D([0, i * height_per_conv, 0, 0])
            intermediate_op.read_shapes[0] = Shape4D(ifm_shape).with_hw(read_shape_h, read_shape_w)

            weight_quant = QuantizationParameters(0, 255, scale_f32=1.0, zero_point=0)
            weight_shape = [weight_h, w, c, b]
            weight_tensor = create_const_tensor(
                f"{intermediate_op.name}_weights",
                weight_shape,
                DataType.uint8,
                np.ones(weight_shape),
                TensorPurpose.Weights,
                quantization=weight_quant,
            )

            weights_1D = np.ones(np.prod(weight_shape))
            weight_tensor.equivalence_id = create_equivalence_id(tuple(weights_1D))
            weight_tensor.value_id = weight_tensor.equivalence_id

            intermediate_op.set_input_tensor(weight_tensor, 1)

            dtype = DataType.int64 if intermediate_op.ifm.dtype == DataType.int16 else DataType.int32
            bias_values = [0] * c
            bias = create_const_tensor(f"{intermediate_op.name}_bias", [c], dtype, bias_values)
            bias.equivalence_id = create_equivalence_id(tuple(bias_values))
            bias.value_id = bias.equivalence_id
            intermediate_op.inputs.append(bias)
            intermediate_op.set_ifm_ofm_shapes()

            # We want to avoid reshaping the ifm tensor directly, to not affect other ops
            # so we update the shape explicitly for this operation
            intermediate_op.ifm_shapes[0] = Shape4D(ifm_shape)

            convs.append(intermediate_op)
            DebugDatabase.add_optimised(op, intermediate_op)

        # If we have more than one convolution
        # We use add operations to accumulate the intermediate tensors
        if len(convs) > 1:
            prev_add_op = None
            idx = 0

            while len(convs):
                intermediate_tensor = op.ofm.clone(suffix=f"_add_sum_{idx}", set_unique=True)
                intermediate_tensor.dtype = DataType.int32
                intermediate_tensor.shape = intermediate_shape

                one_scale_quant = QuantizationParameters(scale_f32=1.0, zero_point=0)

                ifm = convs.pop().ofm
                if not prev_add_op:
                    ifm2 = convs.pop().ofm
                else:
                    ifm2 = prev_add_op.ofm
                intermediate_op = create_add(f"{op.name}_add_{idx}", ifm, ifm2, one_scale_quant)
                intermediate_op.explicit_scaling = ExplicitScaling(False, shift=[0], multiplier=[1])
                intermediate_op.set_output_tensor(intermediate_tensor)
                intermediate_op.set_ifm_ofm_shapes()

                prev_add_op = intermediate_op
                idx += 1

                DebugDatabase.add_optimised(op, intermediate_op)

        # Convert the original mean op to our final Mul operation
        # Which scales and divides by num_elements_in_axis
        op.type = Op.Mul
        op.name = f"{op.name}_mul"
        op.attrs = {}
        op.set_input_tensor(intermediate_op.ofm, 0)

        # The multiplier is calculated in the same way as in the reference,
        # clamping the shift value at the price of some precision loss.
        output_multiplier, output_shift_vela = quantise_scale(np.double(ifmq.scale_f32) / np.double(ofmq.scale_f32))

        # Convert to reference representation shift value
        output_shift = 31 - output_shift_vela

        # Reference calculation
        # round_down_log2 same as 63 - CountLeadingZeros(num_elements_in_axis)
        shift = round_down_log2(num_elements_in_axis)
        shift = min(shift, 32)
        shift = min(shift, 31 + output_shift)
        output_multiplier = (output_multiplier << shift) // num_elements_in_axis
        output_shift = output_shift - shift

        # Convert to vela representation shift
        output_shift_vela = 31 - output_shift

        # For int32 scaling is not supported so instead multiply with the scale
        # intermediate * scale -> round and shift.
        identity_quant = QuantizationParameters(scale_f32=1.0, zero_point=0)
        scalar = create_const_tensor(
            op.name + "_scalar", [1, 1, 1, 1], DataType.int32, [output_multiplier], quantization=identity_quant
        )
        op.set_input_tensor(scalar, 1)
        op.set_ifm_ofm_shapes()
        op.ofm_shapes[0] = Shape4D(intermediate_shape)

        # Reference using TFL rounding for the multiply
        op.rounding_mode = RoundingMode.TFLite

        # Need to use explicit scaling to get the wanted shift
        op.explicit_scaling = ExplicitScaling(False, [output_shift_vela], [1])
        DebugDatabase.add_optimised(op, op)
    return op


def convert_ops_to_lut(op: Operation, arch, nng) -> Operation:
    if op.type == Op.Rsqrt:
        return create_lut_rsqrt_int8_op(op)

    """Convert Exp to 8bit or 16bit LUT to allow for support on NPU."""
    if op.type == Op.Exp:
        func = math.exp
        name = "exp"
    elif op.type == Op.Log:
        def log(value):
            if (value == 0):
                value = sys.float_info.min
            return math.log(value)
        func = log
        name = "log"
    elif op.type == Op.Sqrt:
        func = math.sqrt
        name = "sqrt"
    elif op.type == Op.Gelu:
        def gelu(x):
            if (op.attrs["approximate"]):
                ret = 0.5 * x * (1 + math.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * math.pow(x, 3))))
            else:
                ret = 0.5 * x * (1 + math.erf(x / math.sqrt(2)))
            return ret
        func = gelu
        name = "gelu"
    else:
        return op

    if op.ifm.dtype == DataType.int8:
        return create_lut_8bit_op(op, func, name)
    elif op.ifm.dtype == DataType.int16:
        return create_lut_int16_op(op, func, name)
    else:
        # Should already be catched in tflite supported ops
        assert False, f"Unsupported data type {op.ifm.dtype} for {op.type}"

def optimise_quantize(op: Operation, arch, nng):

    if op.type == Op.Quantize and op.run_on_npu:

        ifm, ofm = op.get_ifm_ofm()
        input_values = ifm.values

        # Guard clause - input not const or no values to quantize
        if ifm.ops[0].type != Op.Const or input_values is None:
            return op

        # Singular val in numpy array, convert to indexable array
        if input_values.ndim == 0:
            input_values = np.array([input_values])

        # requantized int8 to int8 or int16 to int16
        if ifm.dtype == ofm.dtype == DataType.int8 or ifm.dtype == ofm.dtype == DataType.int16:

            # scale needs to use double precision to match TFLite reference kernel
            effective_scale = np.float64(ifm.quantization.scale_f32) / np.float64(ofm.quantization.scale_f32)
            effective_multiplier, effective_shift = quantise_scale(effective_scale)

            requantized_vals = []
            for val in input_values.flatten():
                input_val = val - ifm.quantization.zero_point

                ofm_val = fp_math.multiply_by_quantized_multiplier(input_val, effective_multiplier, effective_shift)
                ofm_val += ofm.quantization.zero_point

                clamped_ofm_value = max(min(ofm_val, ofm.quantization.quant_max), ofm.quantization.quant_min)
                requantized_vals.append(clamped_ofm_value)

            ofm.values = np.array(requantized_vals, ofm.dtype.as_numpy_type())
            ofm.values.shape = input_values.shape

        # Case: Float input - quantize to int
        elif ifm.dtype.type == BaseType.Float:

            quantized_vals = []
            for val in input_values:

                # Derive quantized value
                quant_val = (val / ofm.quantization.scale_f32) + ofm.quantization.zero_point
                clamped_quantized_val = np.clip(quant_val, ofm.quantization.quant_min, ofm.quantization.quant_max)
                quantized_vals.append(clamped_quantized_val)

            # Pass the statically calculated quant val to output tensor
            ofm.values = np.array(quantized_vals, ofm.dtype.as_numpy_type())

        # Unsupported data type
        else:
            return op

        # Make quantize op const and disconnect from parent node

        # Remove reference of the current quant op from the parent tensor's consumer list
        ifm.consumer_list = [consumer for consumer in ifm.consumer_list if consumer.op_index != op.op_index]

        # Clear any references to parent node
        op.inputs = []

        # Convert this quantize op to const
        op.type = Op.Const

    return op


def convert_shape_op_to_constant_tensor(op: Operation, arch, nng):
    """Static optimisation for SHAPE operator output value known at compile time"""

    # Disconnect SHAPE operator from its parent and transform SHAPE OP into constant

    if op.type == Op.Shape and op.run_on_npu:

        ifm, ofm = op.get_ifm_ofm()

        if len(ifm.shape) != ofm.shape[0]:
            return op

        # Remove reference of the current shape op from the parent tensor's consumer list
        ifm.consumer_list = [consumer for consumer in ifm.consumer_list if consumer is None or consumer.op_index != op.op_index]

        # Clear any references to parent node
        op.inputs = []

        # Convert this SHAPE op to const
        op.type = Op.Const

        # Add size calculation to shape output tensors
        ofm.values = np.array(ifm.shape)

    return op


def fixup_pool_strides(op: Operation, arch, nng):
    """Fixup Pool strides when the kernel size, IFM shape and stride are equal. Then stride can be changed
    to (1, 1) and padding can be changed to VALID, so the strides are within the limits for the NPU."""
    if op.type in (Op.AvgPool, Op.MaxPool, Op.QuantizedAvgPool, Op.QuantizedMaxPool):
        ifm, _ = op.get_ifm_ofm()
        kernel_w, kernel_h = op.get_kernel_size()
        stride_w, stride_h = op.get_kernel_stride()
        if kernel_w == stride_w == ifm.shape[2] and kernel_h == stride_h == ifm.shape[1]:
            if "strides" in op.attrs:
                stride_n, _, _, stride_c = op.attrs["strides"]
                op.attrs["strides"] = (stride_n, 1, 1, stride_c)
            op.attrs["stride_w"] = 1
            op.attrs["stride_h"] = 1
            op.attrs["padding"] = Padding.VALID

    return op


def fixup_dilation_gt2(op: Operation, arch, nng) -> Operation:
    """Fixup Conv2DBias and DepthwiseConv2DBias to allow dilation greater than 2."""
    assert op.run_on_npu
    if op.type == Op.Conv2DBias or op.type == Op.DepthwiseConv2DBias:
        dilation_w, dilation_h = op.get_kernel_dilation()

        # if dilation in either axis is greater than that supported by the hardware then we must manually dilate the
        # kernel
        if dilation_w > 2 or dilation_h > 2:
            kernel_w, kernel_h = op.get_kernel_size()
            kernel_ic = op.weights.shape[-2]
            kernel_oc = op.weights.shape[-1]

            # if the dilation is a multiple of 2 then the hardware dialtion can be enabled to provide that multiple
            # of 2. this allows the kernel size to be reduced (via the scaled dilation) by half in that dimension.
            # odd = 1, even = 2
            hw_dilation_h = 1 if (dilation_h & 1) else 2
            hw_dilation_w = 1 if (dilation_w & 1) else 2

            scale_dilation_h = dilation_h // hw_dilation_h
            scale_dilation_w = dilation_w // hw_dilation_w

            # create new empty kernel (HWIO format)
            new_kernel_h = (kernel_h - 1) * scale_dilation_h + 1
            new_kernel_w = (kernel_w - 1) * scale_dilation_w + 1

            new_kernel_shape = [new_kernel_h, new_kernel_w, kernel_ic, kernel_oc]
            new_kernel_values = np.zeros(new_kernel_shape, dtype=op.weights.values.dtype)

            # copy the original kernel values into the new sparse kernel
            for h in range(0, kernel_h):
                for w in range(0, kernel_w):
                    new_h = h * scale_dilation_h
                    new_w = w * scale_dilation_w
                    new_kernel_values[new_h, new_w, :, :] = op.weights.values[h, w, :, :]

            # update the weight tensor with the new dilated kernel
            op.weights.shape = new_kernel_shape
            op.weights.values = new_kernel_values

            # enable(=2) / disable(=1) hardware dilation
            op.attrs["dilation"] = (1, hw_dilation_h, hw_dilation_w, 1)  # nhwc format
            op.attrs["dilation_h_factor"] = hw_dilation_h
            op.attrs["dilation_w_factor"] = hw_dilation_w

    return op


def fixup_transpose(op, arch, nng):
    """
    Convert Transpose to AvgPool where the strides for height and width is swapped on the OFM
    in order to achieve the transpose. It is only possible to swap height and width on the op.

    Shape (2,3)            transposed to Shape (3,2)
    |0|1|2| ifm_stride_w = 1             |0|3| ofm_stride_w = 1
    |4|5|6| ifm_stride_h = 3             |1|4| ofm_stride_h = 2
                                         |2|5|

    To achieve the above with the AvgPool, the ofm_shape must be set equal to the ifm_shape.
    The reason is that AvgPool uses the ofm shape when looping over the memory. So if the
    ofm shape is not equal to the ifm shape the full ifm will not be read.
    When looping over the values the following formula is used:

    IFM [h_pos, w_pos] = h_pos * ifm_stride_h + w_pos * ifm_stride_w
    OFM [h_pos, w_pos] = h_pos * ofm_stride_w + w_pos * ofm_stride_h (stride has been swapped)

    Below code changes op to an AvgPool and sets the correct shapes. The actual stride swap
    is done when creating the ofm featuremap. As seen there are several corner cases
    when it is possible to transpose the depth channel.
    """
    if op.type == Op.Transpose:
        op.name = f"{op.name}_avgpool"
        op.type = Op.AvgPool
        op.attrs["padding"] = Padding.VALID
        op.attrs["stride_w"] = 1
        op.attrs["stride_h"] = 1
        op.attrs["filter_width"] = 1
        op.attrs["filter_height"] = 1
        op.attrs["strides"] = [1, 1, 1, 1]
        op.attrs["ksize"] = [1, 1, 1, 1]
        # Swapping strides only works in linear format (ofm)
        op.ofm.force_linear_format = True

        # Convert IFM to correct 4D shape
        perm = op.inputs[1]
        ifm_shape = op.ifm.shape

        # IFM rank 2 case
        if len(ifm_shape) == 2:
            # IFM shape: WxC -> 1xWxCx1
            op.ifm_shapes[0] = Shape4D([1, ifm_shape[0], ifm_shape[1], 1])

        # IFM rank 3 cases
        elif len(ifm_shape) == 3:
            # Check if HxWxC -> WxHxC
            if perm.values[0] == 1 and perm.values[1] == 0:
                # IFM shape: HxWxC -> 1xHxWxC
                op.ifm_shapes[0] = Shape4D([1, ifm_shape[0], ifm_shape[1], ifm_shape[2]])

            # Check if 1xWxC -> 1xCxW
            elif ifm_shape[0] == 1 and perm.values[1] == 2 and perm.values[2] == 1:
                # IFM shape: 1xWxC -> 1xWxCx1
                op.ifm_shapes[0] = Shape4D([1, ifm_shape[1], ifm_shape[2], 1])

            # Check if Hx1xC -> Cx1xH
            elif ifm_shape[1] == 1 and perm.values[0] == 2 and perm.values[2] == 0:
                # IFM shape: Hx1xC -> 1xHxCx1
                op.ifm_shapes[0] = Shape4D([1, ifm_shape[0], ifm_shape[2], 1])

        # IFM rank 4 cases
        elif len(ifm_shape) == 4:
            # Check if 1xHxWxC -> 1xWxHxC
            if perm.values[1] == 2 and perm.values[2] == 1:
                # IFM shape is correct
                pass

            # Check if 1x1xWxC -> 1x1xCxW
            elif ifm_shape[1] == 1 and perm.values[2] == 3 and perm.values[3] == 2:
                # IFM shape: 1x1xWxC -> 1xWxCx1
                op.ifm_shapes[0] = Shape4D([1, ifm_shape[2], ifm_shape[3], 1])

            # Check if 1xHx1xC -> 1xCx1xH
            elif ifm_shape[2] == 1 and perm.values[1] == 3 and perm.values[3] == 1:
                # IFM shape: 1xHx1xC -> 1xHxCx1
                op.ifm_shapes[0] = Shape4D([1, ifm_shape[1], ifm_shape[3], 1])

        # OFM shape must use IFM shape
        op.ofm_shapes[0] = op.ifm_shapes[0]

        DebugDatabase.add_optimised(op, op)

    return op


def fixup_reshape(op, arch, nng):
    def _get_explicit_shape(implicit_shape, total_size):
        # the explicit shape is a copy of the implicit shape but with the special -1 (remaining size) value converted to
        # the appropriate value
        if implicit_shape is None:
            return None

        explicit_shape = list(implicit_shape)
        if -1 in explicit_shape:
            explicit_shape[explicit_shape.index(-1)] = int(total_size / abs(np.prod(implicit_shape)))

        return explicit_shape

    if op.type == Op.Reshape:
        ifm_tensor, _, ofm_tensor = op.get_ifm_ifm2_ofm()
        ifm_size = ifm_tensor.elements()
        ofm_shape = ofm_tensor.shape

        new_shape_tensor_shape = op.inputs[1].values.flatten() if len(op.inputs) > 1 else None
        new_shape_tensor_shape = _get_explicit_shape(new_shape_tensor_shape, ifm_size)

        new_shape_attribute = op.attrs.get("new_shape", None)
        new_shape_attribute = _get_explicit_shape(new_shape_attribute, ifm_size)

        # if present the new shape tensor overrides the new_shape attribute
        if new_shape_tensor_shape is not None:
            # check tensor
            if not np.array_equal(new_shape_tensor_shape, ofm_shape):
                print(
                    f"Warning: {optype_to_builtintype(op.type)} '{op.name}' has new shape tensor"
                    f" ({new_shape_tensor_shape}) that does not match output tensor shape {ofm_shape}. Will use output"
                    f" tensor shape."
                )
        elif new_shape_attribute is not None:
            # check attribute
            if not np.array_equal(new_shape_attribute, ofm_shape):
                print(
                    f"Warning: {optype_to_builtintype(op.type)} '{op.name}' has new_shape attribute"
                    f" ({new_shape_attribute}) that does not match output tensor shape {ofm_shape}. Will use output"
                    f" tensor shape."
                )
        else:
            print(
                f"Warning: {optype_to_builtintype(op.type)} '{op.name}' does not have a new shape tensor or a new_shape"
                f" attribute. Will use output tensor shape {ofm_shape}."
            )

        # force new shape tensor to output shape
        new_shape_tensor = create_const_tensor(
            op.name + "_new_shape", [len(ofm_shape)], DataType.int32, np.array(ofm_shape, np.int32)
        )
        if len(op.inputs) > 1:
            op.set_input_tensor(new_shape_tensor, 1)
        else:
            op.add_input_tensor(new_shape_tensor)

        # force new_shape attribute to output shape
        op.attrs["new_shape"] = ofm_shape

    return op


def convert_conv_groups(op: Operation, arch, nng):
    """
    Convert convolution groups to a split followed by separate convolutions and then a concat.
    This needs to run before the concat and split handling functions"""
    if not op.type.is_conv2d_op():
        return op

    num_conv_groups = op.attrs.get("num_conv_groups", 0)
    if num_conv_groups > 1:
        # convolution groups params
        ifm_depth_cg = op.ifm.shape[-1] // num_conv_groups
        num_filters_cg = op.weights.shape[-1] // num_conv_groups

        # create split
        split_op = Operation(Op.Split, f"{op.name}_split")
        split_op.attrs.update(
            {
                "num_splits": num_conv_groups,
            }
        )
        # first input is the split axis
        split_op.add_input_tensor(
            # split along the depth axis
            create_const_tensor(f"{split_op.name}_axis", [0], DataType.int32, [-1])
        )
        # second input is the ifm
        split_op.add_input_tensor(op.ifm)
        # calculate shape of each ofm part
        split_op_ofm_shape = op.ifm.shape[:-1] + [ifm_depth_cg]

        # create concat. do this prior to each conv group so that the for-loop can reference the concat as it iterates
        concat_op = Operation(Op.ConcatTFLite, f"{op.name}_concat")
        concat_op.attrs.update(
            {
                "axis": -1,
                "fused_activation_function": None,
            }
        )
        # calculate shape of each ifm part
        concat_op_ifm_shape = op.ofm.shape[:-1] + [num_filters_cg]
        # output is the concatenated tensor
        concat_op.set_output_tensor(op.ofm)  # will disconnect ofm from op

        # for each conv group
        for i in range(num_conv_groups):
            # cg params
            cg_oc_start = i * num_filters_cg
            cg_oc_end = (i + 1) * num_filters_cg

            # split has multiple outputs
            split_op_ofm_part = Tensor(split_op_ofm_shape, op.ifm.dtype, f"{split_op.name}_out{i}")
            split_op_ofm_part.quantization = op.ifm.quantization.clone()
            split_op.add_output_tensor(split_op_ofm_part)

            # concat has multiple inputs
            concat_op_ifm_part = Tensor(concat_op_ifm_shape, op.ifm.dtype, f"{concat_op.name}_in{i}")
            concat_op_ifm_part.quantization = op.ofm.quantization.clone()
            concat_op.add_input_tensor(concat_op_ifm_part)

            # create convolution group operator
            conv_group_op = Operation(op.type, f"{op.name}_cg{i}")
            conv_group_op.attrs = op.attrs.copy()
            conv_group_op.attrs["num_conv_groups"] = 1
            # first input is the ifm
            conv_group_op.add_input_tensor(split_op_ofm_part)
            # second input is weights. the number of filters (i.e. the output channels) need to be split equally
            # across all of the convolution groups
            conv_group_op_weights_shape = op.weights.shape[:-1] + [num_filters_cg]
            conv_group_op_weights_quant = op.weights.quantization.clone()
            conv_group_op_weights_quant.scale_f32 = op.weights.quantization.scale_f32[..., cg_oc_start:cg_oc_end]
            conv_group_op_weights_quant.zero_point = op.weights.quantization.zero_point[..., cg_oc_start:cg_oc_end]
            conv_group_op.add_input_tensor(
                create_const_tensor(
                    f"{op.weights.name}_cg{i}",
                    conv_group_op_weights_shape,
                    op.weights.dtype,
                    op.weights.values[..., cg_oc_start:cg_oc_end],
                    op.weights.purpose,
                    conv_group_op_weights_quant,
                )
            )
            # third input is bias. like the weights, the bias needs to be split equally across all of the convolution
            # groups
            if op.bias is None:
                conv_group_op.add_input_tensor(None)
            else:
                conv_group_op_bias_shape = op.bias.shape[:-1] + [num_filters_cg]
                conv_group_op_bias_quant = op.bias.quantization.clone()
                conv_group_op_bias_quant.scale_f32 = op.bias.quantization.scale_f32[..., cg_oc_start:cg_oc_end]
                conv_group_op_bias_quant.zero_point = op.bias.quantization.zero_point[..., cg_oc_start:cg_oc_end]
                conv_group_op.add_input_tensor(
                    create_const_tensor(
                        f"{op.bias.name}_cg{i}",
                        conv_group_op_bias_shape,
                        op.bias.dtype,
                        op.bias.values[..., cg_oc_start:cg_oc_end],
                        op.bias.purpose,
                        op.bias.quantization,
                    )
                )
            # output goes to the concat
            conv_group_op.set_output_tensor(concat_op_ifm_part)
            # update the cg op shapes and debug db
            conv_group_op.set_ifm_ofm_shapes()
            DebugDatabase.add_optimised(op, conv_group_op)

        # update the split/concat op shapes/debug db
        split_op.set_ifm_ofm_shapes()
        DebugDatabase.add_optimised(op, split_op)
        concat_op.set_ifm_ofm_shapes()
        DebugDatabase.add_optimised(op, concat_op)

        # disconnect the original convolution operator.
        # the ofm has already been disconnected by concat_op.set_output_tensor()
        op.ifm.consumer_list.remove(op)
        op.inputs = []
        op.outputs = []

        # return last op so that other graph optimiser functions can process the new operators
        op = concat_op

    return op

def replace_dilated_convolution(op, arch, nng=None):
    if op.type != Op.BatchToSpaceND:
        return op

    post_op = op
    op = op.inputs[0].ops[0]
    if not (op.type.is_conv2d_op() or op.type.is_depthwise_conv2d_op()):
        return post_op

    pre_op = op.inputs[0].ops[0]
    if pre_op.type != Op.SpaceToBatchND:
        return post_op

    pre_block = pre_op.inputs[1].values
    post_block = pre_op.inputs[1].values
    assert (pre_block == post_block).all
    assert len(np.array(pre_block).shape) == 1
    assert np.array(pre_block).shape[0] == 2

    op.attrs.update({'padding': Padding.SAME, "dilation": (1, pre_block[0], pre_block[1], 1)})
    op.set_output_tensor(post_op.outputs[0])
    ppre_op = pre_op.inputs[0].ops[0]
    op.set_input_tensor(ppre_op.outputs[0], 0)
    op.set_ifm_ofm_shapes()
    op.run_on_npu = arch.tflite_supported_operators.is_operator_supported(op)

    return op

def merge_dequant_lut_quant(op, arch, nng=None):
    if op.type != Op.Quantize:
        return op
    post_op = op

    lut_op = post_op.inputs[0].ops[0]
    if lut_op.type != Op.Exp and lut_op.type != Op.Log:
        return op

    pre_op = lut_op.inputs[0].ops[0]
    if pre_op.type != Op.Dequantize:
        return op

    lut_op.set_input_tensor(pre_op.inputs[0], 0)
    lut_op.set_output_tensor(post_op.outputs[0])

    lut_op.set_ifm_ofm_shapes()

    ifm, ofm = lut_op.get_ifm_ofm()
    lut_op.run_on_npu = arch.tflite_supported_operators.is_operator_supported(lut_op)

    return lut_op

def supported_operator_check(op, arch, nng):
    op.run_on_npu = arch.tflite_supported_operators.is_operator_supported(op)
    return op


def tflite_optimise_graph(nng, arch, force_symmetric_int_weights, output_basename=None, subgraph_output = False):
    if output_basename != None and subgraph_output:
        for idx, sg in enumerate(nng.subgraphs):
            sg.print_npu_graph(output_basename, "ethos-u custom OP subgraph")

    # Compile time static optimisations
    optimisation_list = [
        optimise_quantize,
        convert_shape_op_to_constant_tensor,
        fixup_or_check_asymmetric_weights(force_symmetric_int_weights),
        fixup_pool_strides,
    ]

    for idx, sg in enumerate(nng.subgraphs):
        nng.subgraphs[idx] = rewrite_graph.rewrite_graph_pre_order(
            nng,
            sg,
            arch,
            [],
            optimisation_list,
            rewrite_unsupported=False,
        )

    # Pre-processing step
    pre_process_list = [supported_operator_check, set_ifm_ofm_op_shapes, fixup_reshape, convert_conv_groups]

    for idx, sg in enumerate(nng.subgraphs):
        nng.subgraphs[idx] = rewrite_graph.rewrite_graph_pre_order(
            nng,
            sg,
            arch,
            [],
            pre_process_list,
            rewrite_unsupported=False,
        )

    for idx, sg in enumerate(nng.subgraphs):
        nng.subgraphs[idx] = rewrite_graph.rewrite_graph_pre_order(
            nng, sg, arch, [], [merge_dequant_lut_quant, replace_dilated_convolution], rewrite_unsupported=True,
        )

    # Handle Pad ops
    for idx, sg in enumerate(nng.subgraphs):
        nng.subgraphs[idx] = rewrite_graph.rewrite_graph_pre_order(
            nng, sg, arch, [], [split_pad_to_sub_pad, convert_pad_to_concat], rewrite_unsupported=False,
        )

    # Handle Concat Ops
    for idx, sg in enumerate(nng.subgraphs):
        rewrite_graph.visit_graph_post_order(sg.output_tensors, arch, [], [add_add_op_after_concat, rewrite_concat_ops])
        sg.refresh_after_modification()

    # Handle Split Ops
    for idx, sg in enumerate(nng.subgraphs):
        nng.subgraphs[idx] = rewrite_graph.rewrite_graph_pre_order(
            nng,
            sg,
            arch,
            [],
            [rewrite_unpack_output, rewrite_stridedslice_output, convert_nop_split_to_identity],
            rewrite_unsupported=False,
        )

    for idx, sg in enumerate(nng.subgraphs):
        nng.subgraphs[idx] = rewrite_graph.rewrite_graph_pre_order(
            nng,
            sg,
            arch,
            [rewrite_split_ops],
            [],
            rewrite_unsupported=False,
        )

    # Bypass or rewrite memory only operators
    for idx, sg in enumerate(nng.subgraphs):
        nng.subgraphs[idx] = rewrite_graph.rewrite_graph_pre_order(
            nng,
            sg,
            arch,
            [],
            [bypass_memory_only_ops],
            rewrite_unsupported=False,
        )

    # Rewrite of operators
    op_rewrite_list = [
        set_tensor_equivalence,
        convert_ops_to_lut,
        convert_squared_difference,
        convert_mean_to_depthwise_conv,
        convert_depthwise_to_conv,
        convert_conv_to_fc,
        convert_lstm,
        convert_softmax,
        convert_prelu,
        convert_mul_max_to_abs_or_lrelu,
        convert_lrelu,
        convert_avg_pool_to_conv2d,
        fixup_strided_conv,
        convert_hardswish_to_lut,
        rewrite_fully_connected_input,
        convert_batched_fc_shape,
        fixup_conv2d_backprop,
        fixup_relus_with_differing_ifm_ofm_scaling,
        reorder_depthwise_weights,
        convert_argmax_to_depthwise_conv_and_max_pool,
        fixup_resize,
        fixup_bias_tensors,
        fixup_asymmetric_weights,
        convert_tanh_sigmoid_to_lut,
        convert_quantize,
        replace_pad_by_hw_pad,
        fixup_dilation_gt2,
        fixup_transpose,
    ]

    for idx, sg in enumerate(nng.subgraphs):
        nng.subgraphs[idx] = rewrite_graph.rewrite_graph_pre_order(
            nng,
            sg,
            arch,
            [],
            op_rewrite_list,
            rewrite_unsupported=False,
        )

    for idx, sg in enumerate(nng.subgraphs):
        # remove passthrough tensors and attempt further optimizations
        nng.subgraphs[idx] = rewrite_graph.rewrite_graph_pre_order(
            nng,
            sg,
            arch,
            [remove_passthrough_tensor],
            [fuse_activation_function_with_prev, convert_pad, add_padding_fields],
        )

    # Removal of SplitSliceRead, need to be done after optimisation has been performed,
    # since ifm/ofm_shapes are of importance to this function
    for sg in nng.subgraphs:
        rewrite_graph.visit_graph_post_order(sg.output_tensors, arch, [], [remove_SplitSliceRead])
        sg.refresh_after_modification()

    # Make sure that const optimisations on subgraph outputs are handled correctly
    for sg in nng.subgraphs:
        for ofm in sg.output_tensors:
            if ofm.is_const and ofm.ops[0].type_changed:
                # Subgraph output cannot be const - insert a memory copy
                op = ofm.ops[0]
                ofm_clone = ofm.clone()
                ofm_clone.values = ofm.values
                ofm.values = None
                zero = create_const_tensor("zero", [1], ofm.dtype, [0], quantization=ofm.quantization)
                memcpy = create_add_nop(f"{ofm.name}_copy")
                memcpy.add_input_tensor(ofm_clone)
                memcpy.add_input_tensor(zero)
                memcpy.set_output_tensor(ofm)
                memcpy.set_ifm_ofm_shapes()
                op.set_output_tensor(ofm_clone)
                DebugDatabase.add_optimised(op, memcpy)

    return nng
