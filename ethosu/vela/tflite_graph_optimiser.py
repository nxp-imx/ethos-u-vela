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
# Early optimisation of a TensorFlow Lite based network graph, using the rewrite_graph module
# to do the traversal of the graph.
import math
import uuid

import numpy as np

from . import fp_math
from . import rewrite_graph
from . import scaling
from .api import NpuRoundingMode
from .data_type import DataType
from .debug_database import DebugDatabase
from .errors import UnsupportedFeatureError
from .ethos_u55_regs.ethos_u55_regs import resampling_mode
from .graph_optimiser_util import bypass_memory_only_ops
from .graph_optimiser_util import calc_explicit_padding
from .graph_optimiser_util import convert_depthwise_to_conv
from .graph_optimiser_util import convert_to_lut
from .graph_optimiser_util import fix_sg_input_output
from .graph_optimiser_util import memory_only_ops
from .graph_optimiser_util import move_splitsliceread_to_consumer
from .graph_optimiser_util import needed_total_padding
from .graph_optimiser_util import set_ifm_ofm_op_shapes
from .graph_optimiser_util import set_tensor_equivalence
from .numeric_util import clamp_sigmoid
from .numeric_util import round_away_zero
from .operation import create_activation_function
from .operation import ExplicitScaling
from .operation import NpuBlockType
from .operation import Op
from .operation import Operation
from .operation import Padding
from .operation_util import create_avgpool_nop
from .operation_util import get_pad_values_from_input
from .shape4d import Shape4D
from .softmax import SoftMax
from .tensor import check_quantized_tens_scaling_equal
from .tensor import create_const_tensor
from .tensor import create_equivalence_id
from .tensor import QuantizationParameters
from .tensor import Tensor
from .tensor import TensorPurpose
from .tflite_mapping import optype_to_builtintype

passthrough_nodes = (Op.Identity,)


def create_avg_pool_for_concat(concat_op, name, ifm, ifm_shape: Shape4D, write_offset: Shape4D):
    """Creates an average pool for the given concat op/input feature map"""
    ofm = concat_op.ofm
    avgpool_op = create_avgpool_nop(name)
    avgpool_op.inputs = [ifm]
    avgpool_op.outputs = [ofm]

    avgpool_op.write_offset = write_offset
    avgpool_op.write_shape = ifm_shape
    ofm.ops.append(avgpool_op)
    DebugDatabase.add_optimised(concat_op, avgpool_op)
    avgpool_op.ifm_shapes.append(ifm_shape)
    avgpool_op.ofm_shapes.append(concat_op.ofm_shapes[0])
    avgpool_op.memory_function = Op.ConcatSliceWrite
    return avgpool_op


def remove_passthrough_tensor(tens, arch, nng):
    if len(tens.ops) == 1 and tens.ops[0].type in passthrough_nodes:
        assert len(tens.ops[0].inputs) == 1
        tens = tens.ops[0].inputs[0]
    return tens


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
            read_shape = [oe - os for oe, os in zip(offset_end, offset_start)]

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
        # Check if it is possible to put the SplitSliceRead on the tensor consumer, or if an avgpool need to be inserted
        if (
            len(op.ofm.consumer_list) == 1
            and op.ofm.consumer_list[0] is not None
            and op.ofm.consumer_list[0].run_on_npu
            and op.ofm.consumer_list[0].type not in memory_only_ops
            and op.ofm_shapes[0] == Shape4D.from_list(op.ofm.shape)
        ):
            # SplitSliceRead can be performed by tensor consumer
            cons_op = op.ofm.consumer_list[0]
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
    else:
        raise UnsupportedFeatureError(f"Unsupported padding = {padding_type} for padding calculation")
    padding = (top_pad, left_pad, bottom_pad, right_pad)
    skirt = (top_pad, left_pad, ypad - top_pad, xpad - left_pad)
    return padding, skirt


def calc_upscaled_padding_and_skirt(padding_type, kernel_size, stride, input_shape, upscaling_factor):
    kernel_height, kernel_width = kernel_size[0], kernel_size[1]
    if padding_type == Padding.SAME:
        ypad = needed_total_padding(int(input_shape.height) * upscaling_factor, int(stride[1]), int(kernel_height))
        xpad = needed_total_padding(int(input_shape.width) * upscaling_factor, int(stride[2]), int(kernel_width))
        right_pad = max(((xpad + 1) // upscaling_factor) - 1, 0)
        bottom_pad = max(((ypad + 1) // upscaling_factor) - 1, 0)
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


def fixup_conv2d_backprop(op, arch, nng):
    if op.type == Op.Conv2DBackpropInput:
        # flip the inputs
        op.inputs[0], op.inputs[2] = op.inputs[2], op.inputs[0]
        op.type = Op.Conv2DBackpropInputSwitchedBias
        op.ifm.resampling_mode = resampling_mode.TRANSPOSE

        # Update strides
        op.attrs.update({"stride_w": 1, "stride_h": 1, "strides": (1, 1, 1, 1)})

    return op


# Convert the op to an elementwise add
def convert_resizebilinear_1x1_to_add(op):
    op.type = Op.Add
    op.name = op.name + "_add"
    op.attrs["resizebilinear"] = True
    # Create an input tensor filled with zeros
    shape = op.ofm_shapes[0].as_list()
    tens = Tensor(shape, op.inputs[0].dtype, op.inputs[1].name + "_add")
    tens.values = np.zeros(shape, tens.dtype.as_numpy_type())
    tens.quantization = QuantizationParameters(0.0, 255.0)
    tens.quantization.scale_f32 = 1.0
    tens.quantization.zero_point = 0
    tens.consumer_list = [op]
    tens_op = op.inputs[1].ops[0]
    tens_op.set_output_tensor(tens)
    # Set the add inputs
    op.inputs[1] = op.inputs[0]
    op.inputs[0] = tens
    op.set_ifm_ofm_shapes()

    return op


# Convert ResizeBilinear to a number of 2x2 nearest neighbor upscaling and one avgpool op with kernel size dependent
# on the upscaling factor. Avgpool kernel limit of 8x8 when padding is applied limits upscaling to 8x8.
def convert_resizebilinear_to_nearest_neighbor_upscaling_and_pool(op):
    pre_op = op
    outputs = op.outputs
    dtype = op.ifm.dtype
    op.attrs.update({"strides": (1, 1, 1, 1), "ksize": (1, 1, 1, 1)})
    if op.attrs["align_corners"]:
        shape_modifier = 1
        op.attrs["padding"] = Padding.VALID
    else:
        shape_modifier = 0
        op.attrs["padding"] = Padding.SAME
    op.inputs[0].resampling_mode = resampling_mode.NEAREST

    upscaled_shape = np.array(op.ifm_shapes[0].get_hw_as_list())
    out_shape = np.array(op.ofm_shapes[0].get_hw_as_list())

    # Calculate how many times 2x2 upscaling needs to be performed
    upscale_factor = round(out_shape[1] / upscaled_shape[1])
    n = int(np.log2(upscale_factor))

    # Perform 2x2 upscaling n-1 times
    scaled_op = pre_op
    for count in range(n - 1):
        if count > 0:
            scaled_op = op.clone(f"_{count}")
            scaled_op.inputs[0] = pre_op.outputs[0]

        # Nearest neighbor 2x2 upscaling
        upscaled_shape = upscaled_shape * 2 - shape_modifier
        shape = op.ofm_shapes[0].as_list()
        shape[1:3] = upscaled_shape
        out_tens = Tensor(shape, dtype, f"{op.outputs[0].name}_{count}")
        out_tens.quantization = op.outputs[0].quantization.clone()
        scaled_op.set_output_tensor(out_tens)
        pre_op = scaled_op

        scaled_op.set_ifm_ofm_shapes()

    # Last 2x2 upscaling also applies avgpool with kernel size dependent on the upscaling factor and adds
    # padding to the right and bottom.
    if n > 1:
        scaled_op = op.clone(f"_{n-1}")
        scaled_op.inputs[0] = pre_op.outputs[0]
    scaled_op.attrs["padding"] = Padding.EXPLICIT
    scaled_op.attrs["explicit_padding"] = [0, 0, upscale_factor - 1, upscale_factor - 1]
    scaled_op.attrs.update({"ksize": (1, upscale_factor, upscale_factor, 1)})
    scaled_op.outputs = outputs
    scaled_op.outputs[0].ops = [scaled_op]
    scaled_op.set_ifm_ofm_shapes()

    return op


def fixup_resizebilinear(op, arch, nng):
    if op.type == Op.ResizeBilinear and op.run_on_npu:
        if op.ifm_shapes[0] == op.ofm_shapes[0]:
            # Bypass nop resizebilinear
            op.inputs = op.inputs[:1]
            op.type = Op.Identity
        elif op.ifm_shapes[0].height == 1 and op.ifm_shapes[0].width == 1:
            convert_resizebilinear_1x1_to_add(op)
        else:
            convert_resizebilinear_to_nearest_neighbor_upscaling_and_pool(op)

    return op


def convert_nop_split_to_identity(op, arch, nng):
    if op.type == Op.Split and op.attrs.get("num_splits") == 1:
        # the list comprehension should return a list with a single tensor
        # if it shouldn't, remove_passthrough_tensor will fail appropriately
        op.inputs = [i for i in op.inputs if i.shape == op.outputs[0].shape]
        op.type = Op.Identity
    return op


def rewrite_fully_connected_input(op, arch, nng):
    if op.type == Op.FullyConnected:
        n_in_elems = op.weights.shape[-2]
        elms = op.ifm.elements()
        batch_size = elms // n_in_elems
        assert batch_size * n_in_elems == elms

        op.ifm_shapes[0] = Shape4D([batch_size, 1, 1, n_in_elems])
    return op


def convert_batched_fc_shape(op, arch, nng):
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

            if op.type == Op.Conv2DBackpropInputSwitchedBias:
                upscaling_factor = output_shape.height // input_shape.height
                padding, skirt = calc_upscaled_padding_and_skirt(
                    op.attrs["padding"], kernel_size, op.attrs["strides"], input_shape, upscaling_factor
                )
            else:
                padding, skirt = calc_padding_and_skirt(
                    op.attrs["padding"], op.kernel, input_shape, op.attrs.get("explicit_padding"),
                )

            op.attrs["explicit_padding"] = padding
            op.attrs["skirt"] = skirt

    return op


def reorder_depthwise_weights(op, arch, nng):
    if op.type.is_depthwise_conv2d_op():
        weight_tensor = op.inputs[1]
        weight_tensor.values = np.transpose(weight_tensor.values, (0, 1, 3, 2))
        weight_tensor.set_all_shapes(list(weight_tensor.values.shape))
        weight_tensor.weight_transpose_depthwise = True

    return op


def optimise_strided_conv(op, arch, nng):
    stride_x, stride_y = op.get_kernel_stride()
    ifm_tensor, _, weight_tensor, _ = op.get_ifm_ifm2_weights_ofm()

    if (
        op.type == Op.Conv2DBias
        and op.op_index == 0
        and stride_x == 2
        and op.ifm_shapes[0].depth <= 4
        and op.ifm_shapes[0].width % 2 == 0
        and weight_tensor is not None
        and weight_tensor.shape[1] >= 2
    ):
        ifm_shape = op.ifm_shapes[0]
        # IFM
        op.ifm_shapes[0] = Shape4D([ifm_shape.batch, ifm_shape.height, ifm_shape.width // 2, ifm_shape.depth * 2])

        # Weights
        weight_shape = weight_tensor.shape
        if weight_shape[1] % 2 != 0:
            weight_shape[1] = weight_shape[1] + 1
            padded_array = np.zeros(weight_shape)
            for i in range(weight_shape[0]):
                padded_array[i] = np.vstack(
                    [
                        weight_tensor.values[i],
                        np.full((1, weight_shape[2], weight_shape[3]), weight_tensor.quantization.zero_point),
                    ]
                )
            weight_tensor.values = padded_array
        weight_shape[1] //= 2
        weight_shape[2] *= 2
        weight_tensor.values = np.reshape(weight_tensor.values, weight_shape)
        weight_tensor.set_all_shapes(weight_shape)
        # If multiple copies of the weights are used, we could avoid
        # them having the same address by changing the value_id
        weight_tensor.value_id = uuid.uuid4()

        # Strides
        stride_x = 1
        op.attrs.update({"stride_w": stride_x, "stride_h": stride_y, "strides": (1, stride_y, stride_x, 1)})

    return op


def convert_conv_to_fc(op, arch, nng):
    # Conv 1x1 can be equivalent to Fully Connected.
    # By representing certain convs as fully connected layers, Vela can better determine wether or not to use
    # caching/double buffering for the weights.
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


def fixup_relus_with_differing_ifm_ofm_scaling(op, arch, nng):
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
            relu_fused_op.rescale = ExplicitScaling(False, [shift], [multiplier])
            # Tidy up and assign the ifm and ofm to the new op
            ifm.consumer_list.remove(op)

            relu_fused_op.add_input_tensor(ifm)
            relu_fused_op.set_output_tensor(ofm)
            relu_fused_op.set_ifm_ofm_shapes()
            op = relu_fused_op
    return op


def convert_softmax(op, arch, nng):
    if op.type == Op.Softmax and op.run_on_npu:
        softmax = SoftMax(op)
        op = softmax.get_graph()
    return op


def convert_mul_max_to_abs_or_lrelu(op, arch, nng):
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
            mul = [m for m in muls if len(set(op.inputs + m.ops[0].inputs)) == 1][0].ops[0]
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


def convert_hardswish_to_lut(op, arch, nng):
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

    # Add multiplication with alpha
    mul_alpha = Operation(Op.Mul, op.name + "_mul_alpha")
    mul_alpha.add_input_tensor(ifm)
    # Create const tensor containing alpha as scalar
    alpha = np.float32(op.attrs["alpha"])
    quantization = ifm.quantization.clone()
    quantization.min = 0
    quantization.max = alpha * (quantization.quant_max - quantization.quant_min)
    quantization.zero_point = 0
    if np.isinf(1 / alpha):
        # Handling of alpha near zero
        quantization.scale_f32 = np.float32(1)
        scalar = 0
    else:
        quantization.scale_f32 = alpha
        scalar = alpha
    alpha_tens = create_const_tensor(
        op.name + "_alpha_scalar", [], ifm.dtype, [scalar], np.float32, quantization=quantization
    )
    alpha_tens.values = np.array([1])
    mul_alpha.add_input_tensor(alpha_tens)
    fm_alpha = ofm.clone(op.name + "_alpha", set_unique=True)
    mul_alpha.set_output_tensor(fm_alpha)
    mul_alpha.set_ifm_ofm_shapes()
    DebugDatabase.add_optimised(op, mul_alpha)

    if check_quantized_tens_scaling_equal(ifm, ofm):
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
        identity_tens = create_const_tensor(
            op.name + "_id_scalar", [], ifm.dtype, [1], np.uint8, quantization=quantization
        )
        mul_identity.add_input_tensor(identity_tens)
        # Make sure that fm_id is allocated to a different address than fm_alpha
        fm_id = ofm.clone(op.name + "_id", set_unique=True)
        mul_identity.set_output_tensor(fm_id)
        mul_identity.set_ifm_ofm_shapes()
        DebugDatabase.add_optimised(op, mul_identity)

    # Convert LeakyRelu to Max, add the results of the multiplication(s) as inputs
    op.type = Op.Maximum
    op.name = op.name.replace("LeakyRelu", "Maximum")
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


def convert_lrelu(op, arch, nng):
    # Converts LeakyRelu to a LUT based solution if possible, otherwise a mul + max
    if op.type != Op.LeakyRelu:
        return op
    ifm, ofm = op.get_ifm_ofm()
    if ifm is None or ofm is None:
        return op
    if ifm.dtype in (DataType.uint8, DataType.int8) and ifm.dtype == ofm.dtype:
        # use LUT for int8/uint8
        return convert_lrelu_to_lut(op, arch)
    if check_quantized_tens_scaling_equal(ifm, ofm) and ifm.dtype == ofm.dtype == DataType.int16:
        # use LeakyRelu unmodified for int16 with equal input/output scaling
        return op
    return convert_lrelu_to_mul_max(op, arch)


def convert_tanh_sigmoid_to_lut(op, arch, nng):
    # Converts int8/uint8 Sigmoid and Tanh to a LUT based solution
    if op.type == Op.Sigmoid:
        return convert_to_lut8(op, clamp_sigmoid, "sigmoid")
    elif op.type == Op.Tanh:
        return convert_to_lut8(op, math.tanh, "tanh")
    return op


def remove_memory_only_ops(op, arch):
    if op.run_on_npu and op.type in memory_only_ops:
        bypass_memory_only_ops(op)


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
    DebugDatabase.add_optimised(op, prev_op)
    return op


def _leading_pad_ok(leading_pad, stride, kernel_size):
    # If kernel size // 2 > stride, then (left, top) padding must be a multiple of stride,
    # otherwise replacing PAD by hardware padding would iterate the wrong IFM rows/columns
    max_size = kernel_size // 2
    return leading_pad == max_size or max_size <= stride or leading_pad % stride == 0


def replace_pad_by_hw_pad(op: Operation, arch, nng):
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
                np.uint8,
                purpose=TensorPurpose.Weights,
                quantization=quantization,
            )
            weight_tens.values = weights
            op.type = Op.DepthwiseConv2DBias
            op.inputs = []
            op.add_input_tensor(ifm)
            op.add_input_tensor(weight_tens)
            # Add bias tensor, all biases set to 0
            op.inputs.append(None)
            fixup_bias_tensors(op, arch, nng)
            # Add other inputs
            op.inputs.extend(other_inputs)
            op.rounding_mode = NpuRoundingMode.NATURAL

        # Bypass the PAD operator
        op.set_input_tensor(pad_op.ifm, 0)
        # Adjust the padding attributes of the convolution operator
        op.attrs["padding"] = Padding.EXPLICIT
        op.attrs["explicit_padding"] = (top, left, bottom, right)
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
            op.name + "_top", shape.as_list(), ofm.dtype, shape.elements() * [pad_value], np.uint8, quantization=quant
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
            np.uint8,
            quantization=quant,
        )
        zero_tens.equivalence_id = create_equivalence_id(tuple(zero_tens.values))
        create_avg_pool_for_concat(
            op, op.name + "_bottom", zero_tens, shape, shp0.with_height(ofm_shape.height - bottom)
        )
    if left > 0:
        shape = Shape4D(1, ifm_shape.height, left, ofm_shape.depth)
        zero_tens = create_const_tensor(
            op.name + "_left", shape.as_list(), ofm.dtype, shape.elements() * [pad_value], np.uint8, quantization=quant
        )
        zero_tens.equivalence_id = create_equivalence_id(tuple(zero_tens.values))
        create_avg_pool_for_concat(op, op.name + "_left", zero_tens, shape, shp_top)
    if right > 0:
        shape = Shape4D(1, ifm_shape.height, right, ofm_shape.depth)
        zero_tens = create_const_tensor(
            op.name + "_right", shape.as_list(), ofm.dtype, shape.elements() * [pad_value], np.uint8, quantization=quant
        )
        zero_tens.equivalence_id = create_equivalence_id(tuple(zero_tens.values))
        create_avg_pool_for_concat(
            op, op.name + "_right", zero_tens, shape, shp_top.with_width(ofm_shape.width - right)
        )

    op.type = Op.ConcatTFLite
    return avgpool_op


def add_attrs_to_resizebilinear(op, arch, nng):
    if op.type == Op.ResizeBilinear and op.run_on_npu:
        input_tensor = op.inputs[0]
        input_shape = op.ifm_shapes[0]
        upscaled_height = input_shape.height * 2
        upscaled_width = input_shape.width * 2
        out_shape = op.ofm_shapes[0]
        if not op.attrs["align_corners"] and out_shape.height == upscaled_height and out_shape.width == upscaled_width:
            # this means the output is supposed to be a x2 upscale,
            # so we need to do SAME padding
            op.attrs["padding"] = Padding.SAME
        elif (
            op.attrs["align_corners"]
            and out_shape.height == (upscaled_height - 1)
            and out_shape.width == (upscaled_width - 1)
        ):
            # here we can just run the avg pool without padding and
            # produce a (M * 2 - 1, N * 2 - 1) sized output
            op.attrs["padding"] = Padding.VALID
        else:
            return op
        input_tensor.resampling_mode = resampling_mode.NEAREST
        op.attrs.update({"strides": (1, 1, 1, 1), "ksize": (1, 2, 2, 1)})
    return op


def fixup_bias_tensors(op, arch, nng):
    if op.type.needs_bias() and op.bias is None:
        # Op has no bias, add bias tensor filled with zeros
        nr_biases = op.inputs[1].shape[-1]
        bias_values = [0] * nr_biases
        bias_tensor = create_const_tensor(op.name + "_bias", [nr_biases], DataType.int32, bias_values)
        op.set_input_tensor(bias_tensor, op.type.info.indices.biases[0])

    return op


def fixup_asymmetric_weights(op, arch, nng):
    if op.run_on_npu and (op.type.is_conv2d_op() or op.type.is_depthwise_conv2d_op()):
        if op.ifm.dtype == DataType.int8:
            if not np.all(op.weights.quantization.zero_point == 0):
                print(f"Warning: {op.type} '{op.name}' has asymmetric weights, zero points have been adjusted.")
                op.weights.quantization.zero_point *= 0

    return op


def convert_mean_to_depthwise_conv_or_avgpool(op, arch, nng):
    if op.type == Op.Mean and op.run_on_npu:
        keep_dims = op.attrs.get("keep_dims", False)
        inp, axis = op.inputs
        shape = inp.shape
        dims = len(shape)

        # Height and width axes have different index depending on dimensions
        if axis.shape == [] or axis.shape[0] == 1:  # single axis
            axis = int(axis.values) if len(axis.shape) == 0 else int(axis.values[0])
            if dims in (2, 3):
                if axis == 0:
                    h, w = shape[axis], 1
                else:
                    h, w = 1, shape[axis]
            else:
                if axis == 1:
                    h, w = shape[axis], 1
                else:
                    h, w = 1, shape[axis]
        else:  # multiple axes
            axis = sorted(axis.values)
            h, w = [shape[i] for i in axis]

        # Set necessary depthwise attributes
        op.attrs.update(
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
        # Change op type
        op.type = Op.DepthwiseConv2DBias
        # Set IFM/OFM shapes after changing op type
        op.set_ifm_ofm_shapes()

        weight_scale, bias = 1, None
        ofmq, ifmq = op.ofm.quantization, inp.quantization
        # Set rounding mode, scaling and zero point based on which reference implementation to match
        if len(shape) == 4 and axis == [1, 2] and keep_dims:
            if inp.dtype == DataType.uint8:
                # This attribute means a different scaling calculation is used in order to match reference
                op.low_precision_scaling = True
                weight_scale = h * w
                # Set zero points to 0 as they will be adjusted for with bias term
                foq = ofmq.clone()
                foq.zero_point = 0
                fiq = ifmq.clone()
                fiq.zero_point = 0
                op.forced_input_quantization = fiq
                bias_term = ofmq.zero_point - int(ifmq.zero_point * ifmq.scale_f32 / ofmq.scale_f32)
                # If the bias term is outside uint8 range, we need an Add op to apply it.
                if bias_term < 0 or bias_term > 255:
                    intermediate = op.ofm.clone(suffix="_intermediate", set_unique=True)
                    # Bias term has higher bitness (i32) than input/output (u8).
                    # 16 bits is enough since the bias is added/subtracted from a u8 value,
                    # the bias can only effectively assume values in the range [-255, 255].
                    intermediate.dtype = DataType.int16
                    intermediate.quantization.zero_point = 0
                    add_op = Operation(Op.Add, op.name + "_bias")
                    add_op.forced_output_quantization = foq
                    add_op.add_input_tensor(intermediate)
                    quant = QuantizationParameters()
                    quant.zero_point = 0
                    bias_term_tens = create_const_tensor(
                        op.name + "_bias", [1, 1, 1, 1], DataType.int16, [bias_term], np.int16, quantization=quant,
                    )
                    add_op.add_input_tensor(bias_term_tens)
                    add_op.set_output_tensor(op.ofm)
                    add_op.set_ifm_ofm_shapes()
                    add_op.activation = op.activation
                    op.activation = None
                    op.set_output_tensor(intermediate)
                    op.set_ifm_ofm_shapes()
                # If not, we can just do it with the OFM zero point.
                else:
                    foq.zero_point = bias_term
                    op.forced_output_quantization = foq
            else:
                assert inp.dtype == DataType.int8
                # Use a depthwise to calculate the sum,
                # followed by a multiplication with 1/N to get the MEAN
                weight_scale = 1
                intermediate = op.ofm.clone(suffix="_intermediate", set_unique=True)
                intermediate.dtype = DataType.int16
                mul_op = Operation(Op.Mul, op.name + "_mul")
                mul_op.add_input_tensor(intermediate)
                # Create scalar containing 1/N
                quant = QuantizationParameters()
                quant.zero_point = 0
                # The reference rounds negative numbers downwards, e.g. -1.5 is rounded to -2,
                # while rounding mode NATURAL would round this to -1.
                # This can only occur if N is even, and can be emulated by
                # multiplying with a number that is slightly smaller than 1/N.
                # It must be so small that other roundings are not affected;
                # the calculated value is based on worst case,
                # which is sum 256 * N (the maximum sum that can occur with int8)
                n = int(h * w)
                eps = 1 / (256 * (n + 1)) if n % 2 == 0 else 0
                quant.scale_f32 = 1 / (n - eps)
                scalar = create_const_tensor(
                    op.name + "_scalar", [1, 1, 1, 1], DataType.uint8, [1], np.uint8, quantization=quant
                )
                mul_op.add_input_tensor(scalar)
                mul_op.set_output_tensor(op.ofm)
                mul_op.set_ifm_ofm_shapes()
                mul_op.rounding_mode = NpuRoundingMode.NATURAL
                mul_op.activation = op.activation
                op.activation = None
                op.set_output_tensor(intermediate)
                op.set_ifm_ofm_shapes()
        elif ifmq.zero_point == ofmq.zero_point and ifmq.scale_f32 == ofmq.scale_f32:
            # Here we can just use a simple AvgPool with truncating rounding,
            # as we're emulating simple integer division.
            op.rounding_mode = NpuRoundingMode.TRUNCATE
            op.type = Op.AvgPool
            op.attrs.update({"ksize": (1, h, w, 1), "filter_height": h, "filter_width": w})
        else:
            op.rounding_mode = NpuRoundingMode.NATURAL
            weight_scale = 1 / (h * w)
            # Input zero point is adjusted after mean calculation, so we emulate that with a bias
            bias = -ifmq.zero_point * h * w
            fiq = ifmq.clone()
            fiq.zero_point = 0
            op.forced_input_quantization = fiq

        # Change dimensions to 4
        if dims < 4:
            shape = [1] + shape
            if dims == 2:
                shape += [1]

        # If height is greater than max kernel height, reshape from HxW to 1x(HxW)
        if (h > 64 and op.type == Op.DepthwiseConv2DBias) or (h > 256 and op.type == Op.AvgPool):
            shape = [shape[0], 1, h * w, shape[3]]
            op.ifm_shapes[0] = Shape4D(shape)
            if h > 256 and op.type == Op.AvgPool:
                op.attrs.update({"ksize": (1, 1, h * w, 1), "filter_height": 1, "filter_width": h * w})

        # If the AvgPool version is used, we don't need to do anything else
        if op.type == Op.AvgPool:
            return op

        # Make unit weight tensor quantization
        weight_quant = ifmq.clone()
        weight_quant.min = 0
        weight_quant.max = 255
        weight_quant.scale_f32 = weight_scale
        weight_quant.zero_point = 0

        # Set weight shape to [H,W,C,B]
        weight_shape = shape[1:4] + [shape[0]]
        # Add unit weight tensor
        op.set_input_tensor(
            create_const_tensor(
                "weights",
                weight_shape,
                inp.dtype,
                np.ones(weight_shape),
                value_dtype=np.uint8,
                quantization=weight_quant,
            ),
            1,
        )
        op.weights.values = np.reshape(op.inputs[1].values, weight_shape)

        # Add None bias tensor
        op.inputs.append(None)
        # Add bias tensor
        if bias:
            bias_shape = [shape[-1]]
            op.set_input_tensor(
                create_const_tensor(
                    "bias", bias_shape, inp.dtype, np.ones(bias_shape) * bias, value_dtype=np.int32, quantization=None,
                ),
                2,
            )

    return op


def supported_operator_check(op, arch, nng):
    op.run_on_npu = arch.tflite_supported_operators.is_operator_supported(op)
    return op


def tflite_optimise_graph(nng, arch):
    # Pre-processing step
    pre_process_list = [
        supported_operator_check,
        set_ifm_ofm_op_shapes,
    ]

    for idx, sg in enumerate(nng.subgraphs):
        nng.subgraphs[idx] = rewrite_graph.rewrite_graph_pre_order(
            nng, sg, arch, [], pre_process_list, rewrite_unsupported=False,
        )

    # Handle Concat Ops
    for idx, sg in enumerate(nng.subgraphs):
        rewrite_graph.visit_graph_post_order(sg.output_tensors, arch, [], [rewrite_concat_ops])
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
            nng, sg, arch, [rewrite_split_ops], [], rewrite_unsupported=False,
        )

    # Handle sg input output
    for idx, sg in enumerate(nng.subgraphs):
        nng.subgraphs[idx] = rewrite_graph.rewrite_graph_pre_order(
            nng, sg, arch, [], [fix_sg_input_output], rewrite_unsupported=False,
        )

    # Removal of memory only operators
    for sg in nng.subgraphs:
        rewrite_graph.visit_graph_post_order(sg.output_tensors, arch, [], [remove_memory_only_ops])
        sg.refresh_after_modification()

    # Rewrite of operators
    op_rewrite_list = [
        set_tensor_equivalence,
        convert_mean_to_depthwise_conv_or_avgpool,
        convert_depthwise_to_conv,
        convert_conv_to_fc,
        convert_softmax,
        optimise_strided_conv,
        convert_hardswish_to_lut,
        rewrite_fully_connected_input,
        convert_batched_fc_shape,
        fixup_conv2d_backprop,
        fixup_relus_with_differing_ifm_ofm_scaling,
        reorder_depthwise_weights,
        fixup_resizebilinear,
        fixup_bias_tensors,
        fixup_asymmetric_weights,
        convert_mul_max_to_abs_or_lrelu,
        convert_lrelu,
        convert_tanh_sigmoid_to_lut,
        replace_pad_by_hw_pad,
    ]

    for idx, sg in enumerate(nng.subgraphs):
        nng.subgraphs[idx] = rewrite_graph.rewrite_graph_pre_order(
            nng, sg, arch, [], op_rewrite_list, rewrite_unsupported=False,
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

    return nng
