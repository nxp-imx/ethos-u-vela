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
# Early optimisation of the network graph, using the rewrite_graph module to do the traversal of the graph. These are
# split into two parts optimise_graph_a and optimise_graph_b.
import math

import numpy as np

from . import rewrite_graph
from .data_type import DataType
from .errors import UnsupportedFeatureError
from .ethos_u55_regs.ethos_u55_regs import resampling_mode
from .numeric_util import full_shape
from .operation import NpuBlockType
from .operation import Operation
from .tensor import Tensor

passthrough_nodes = set(("Identity",))


def remove_passthrough_tensor(tens, arch):
    if len(tens.ops) == 1 and tens.ops[0].type in passthrough_nodes:
        assert len(tens.ops[0].inputs) == 1
        tens = tens.ops[0].inputs[0]
    return tens


def rewrite_concat(tens, arch):
    if len(tens.ops) == 1 and tens.ops[0].is_concat_op():
        concat_op = tens.ops[0]
        if tens != concat_op.outputs[0]:
            return tens  # don't attempt to rewrite the min/max outputs of QuantizedConcat

        # Not supported so leave it and run on CPU
        if not concat_op.run_on_npu:
            return tens

        inputs, axis = concat_op.get_concat_inputs_axis()

        tens.ops = []
        offset = 0
        for idx, inp in enumerate(inputs):
            new_op = Operation("ConcatSliceWrite", concat_op.name + str(idx))
            new_op.inputs = [inp]
            new_op.outputs = [tens]
            new_op.attrs["concat_axis"] = axis
            new_op.attrs["concat_start"] = offset
            offset += inp.shape[axis]
            new_op.attrs["concat_end"] = offset
            new_op.run_on_npu = True
            tens.ops.append(new_op)
        assert tens.shape[axis] == offset

    return tens


def rewrite_split(tens, arch):

    if len(tens.ops) == 1 and tens.ops[0].is_split_op():
        split_op = tens.ops[0]

        # Not supported so leave it and run on CPU
        if not split_op.run_on_npu:
            return tens

        inp, outputs, axis, offset_start, offset_end = split_op.get_split_inputs_axis()

        tens.ops = []
        new_op = Operation("SplitSliceRead", split_op.name)
        new_op.inputs = [inp]
        new_op.outputs = [tens]

        # For Split the offset cannot be extracted from the tensor so it has to
        # be calculated from the index of the output tensor
        if axis is not None:
            # Get the start and end of the split
            offset_start = [0] * len(tens.shape)
            offset_end = [0] * len(tens.shape)
            for out in outputs:
                if out == tens:
                    break
                offset_start[axis] += out.shape[axis]

            offset_end[axis] = offset_start[axis] + tens.shape[axis]

        new_op.attrs["split_start"] = offset_start
        new_op.attrs["split_end"] = offset_end
        new_op.run_on_npu = True
        tens.ops.append(new_op)

    return tens


def needed_total_padding(input_size, stride, filter_size):
    out_size = (input_size + stride - 1) // stride
    needed_input = (out_size - 1) * stride + filter_size
    total_padding = max(0, needed_input - input_size)
    return total_padding


def calc_padding_and_skirt(padding_type, kernel_size, stride, input_dims):
    ypad = needed_total_padding(int(input_dims[1]), int(stride[1]), int(kernel_size[0]))
    xpad = needed_total_padding(int(input_dims[2]), int(stride[2]), int(kernel_size[1]))
    if padding_type == b"SAME":
        left_pad = (xpad + 0) // 2
        right_pad = (xpad + 1) // 2
        top_pad = (ypad + 0) // 2
        bottom_pad = (ypad + 1) // 2
    elif padding_type == b"VALID":
        left_pad = 0
        right_pad = 0
        top_pad = 0
        bottom_pad = 0
    else:
        raise UnsupportedFeatureError("Unknown padding {}".format(str(padding_type)))
    padding = (top_pad, left_pad, bottom_pad, right_pad)
    skirt = (top_pad, left_pad, ypad - top_pad, xpad - left_pad)
    return padding, skirt


def calc_upscaled_padding_and_skirt(padding_type, kernel_size, stride, input_dims, upscaling_factor):
    kernel_height, kernel_width = kernel_size[0], kernel_size[1]
    if padding_type == b"SAME":
        ypad = needed_total_padding(int(input_dims[1]) * upscaling_factor, int(stride[1]), int(kernel_height))
        xpad = needed_total_padding(int(input_dims[2]) * upscaling_factor, int(stride[2]), int(kernel_width))

        right_pad = ((xpad + 1) // upscaling_factor) - 1
        bottom_pad = ((ypad + 1) // upscaling_factor) - 1
        left_pad = max(kernel_width - 1 - right_pad, 0)
        top_pad = max(kernel_height - 1 - bottom_pad, 0)

    elif padding_type == b"VALID":
        right_pad = max(kernel_width - 2, 0)
        bottom_pad = max(kernel_height - 2, 0)
        left_pad = kernel_width - 1
        top_pad = kernel_height - 1
    else:
        assert 0, "Unknown padding"

    padding = (top_pad, left_pad, bottom_pad, right_pad)
    skirt = padding
    return padding, skirt


def fixup_conv2d_backprop(op, arch):
    if op.type == "Conv2DBackpropInput":
        # flip the inputs
        op.inputs[0], op.inputs[2] = op.inputs[2], op.inputs[0]
        op.type = "Conv2DBackpropInputSwitchedBias"
        weight_shape = op.inputs[1].shape
        weight_sets = weight_shape[3]

        if len(op.inputs) < 4:
            # Add bias/scale tensor filled with zeros
            scale_op = Operation("Const", op.name + "_bias")
            scale_tens = Tensor([weight_sets], DataType.int32, op.name + "_bias_tens")
            scale_tens.values = [0] * weight_sets
            scale_tens.quant_values = [0] * weight_sets
            scale_tens.ops = [scale_op]
            scale_op.outputs = [scale_tens]
            scale_tens.consumer_list = [op]
            op.inputs.append(scale_tens)

        # Update strides
        op.attrs.update({"stride_w": 1, "stride_h": 1, "strides": (1, 1, 1, 1)})

    return op


def fixup_fully_connected_input(op, arch):
    if op.type == "FullyConnectedAct":
        inp = op.inputs[0]
        weights = op.inputs[1]

        n_in_elems = weights.shape[-2]
        elms = inp.elements()
        batch_size = elms // n_in_elems
        assert batch_size * n_in_elems == elms

        desired_shape = [batch_size, n_in_elems]
        if inp.shape != desired_shape:
            # mismatch, insert a reshape to fix this.
            reshape_name = op.name + "_reshape"
            new_shape_tens = Tensor([1], DataType.int32, reshape_name + "_shape")
            new_shape_tens.values = np.array(desired_shape)
            new_shape_tens_const = Operation("Const", new_shape_tens.name + "_const")
            new_shape_tens.ops = [new_shape_tens_const]
            new_shape_tens_const.outputs = [new_shape_tens]

            reshape_op = Operation("Reshape", reshape_name)
            reshape_op.inputs = [inp, new_shape_tens]
            reshape_op.attrs["new_shape"] = desired_shape
            reshape_out = inp.clone("_reshaped")
            reshape_out.shape = reshape_out.storage_shape = reshape_out.bandwidth_shape = desired_shape
            reshape_out.ops = [reshape_op]
            reshape_op.outputs = [reshape_out]

            op.inputs[0] = reshape_out

    return op


def fixup_pack_input(op, arch):
    if op.type == "Pack":
        # Pack is also referred to as Stack
        # Requires the rewrite_concat function to be called on the op afterwards
        axis = int(op.attrs["axis"])
        desired_shape = op.inputs[0].shape[:axis] + [1] + op.inputs[0].shape[axis:]

        # Construct 1 shape tensor to be used by all inserted reshape ops
        new_shape_name = op.name + "_reshape_shape"
        new_shape_tens = Tensor([1], DataType.int32, new_shape_name)
        new_shape_tens.values = np.array(desired_shape)
        new_shape_tens_const = Operation("Const", new_shape_tens.name + "_const")
        new_shape_tens.ops = [new_shape_tens_const]
        new_shape_tens_const.outputs = [new_shape_tens]

        for idx, inp in enumerate(op.inputs):
            reshape_name = op.name + str(idx) + "_reshape"
            reshape_op = Operation("Reshape", reshape_name)
            reshape_op.inputs = [inp, new_shape_tens]
            reshape_op.attrs["new_shape"] = desired_shape
            reshape_out = inp.clone("_reshaped")
            reshape_out.shape = reshape_out.storage_shape = reshape_out.bandwidth_shape = desired_shape
            reshape_out.ops = [reshape_op]
            reshape_op.outputs = [reshape_out]

            op.inputs[idx] = reshape_out

        op.type = "PackReshaped"

    return op


def fixup_unpack_output(tens, arch):
    op = tens.ops[0]
    if op.type in set(("Unpack", "StridedSlice")):
        # Unpack is also referred to as Unstack
        # Requires the rewrite_split function to be called on the op afterwards

        reshape_input_shape = tens.shape
        if op.type == "StridedSlice":
            new_axis_mask = op.attrs["new_axis_mask"]
            shrink_axis_mask = op.attrs["shrink_axis_mask"]
            ellipsis_mask = op.attrs["ellipsis_mask"]

            if (new_axis_mask != 0 and shrink_axis_mask != 0) or ellipsis_mask != 0:
                # Not supported, will be put on CPU
                return tens
            if shrink_axis_mask == 0 and new_axis_mask == 0:
                # Equal Rank StridedSlice, no need to insert reshape
                return tens
            elif shrink_axis_mask != 0:
                n = 0
                axis = 0
                while shrink_axis_mask:
                    prev_mask = shrink_axis_mask
                    n += 1
                    shrink_axis_mask &= shrink_axis_mask - 1
                    axis = int(math.log2(prev_mask - shrink_axis_mask))
                    reshape_input_shape = reshape_input_shape[:axis] + [1] + reshape_input_shape[axis:]

                assert len(tens.shape) == (len(op.inputs[0].shape) - n)
                op.attrs["shrink_axis_mask"] = 0

            elif new_axis_mask != 0:
                n = 0
                axis = 0
                while new_axis_mask:
                    prev_mask = new_axis_mask
                    n += 1
                    new_axis_mask &= new_axis_mask - 1
                    axis = int(math.log2(prev_mask - new_axis_mask))
                    reshape_input_shape = reshape_input_shape[:axis] + reshape_input_shape[(axis + 1) :]
                    new_axis_mask >>= 1

                assert len(tens.shape) == (len(op.inputs[0].shape) + n)
                op.attrs["new_axis_mask"] = 0
        else:
            axis = int(op.attrs["axis"])
            op.type = "UnpackReshaped"
            reshape_input_shape = tens.shape[:axis] + [1] + tens.shape[axis:]

        # Construct 1 shape tensor to be used by all inserted reshape ops
        new_shape_name = op.name + "_reshape_shape"
        new_shape_tens = Tensor([1], DataType.int32, new_shape_name)
        new_shape_tens.values = np.array(tens.shape)
        new_shape_tens_const = Operation("Const", new_shape_tens.name + "_const")
        new_shape_tens.ops = [new_shape_tens_const]
        new_shape_tens_const.outputs = [new_shape_tens]

        for idx, out_tens in enumerate(op.outputs):
            reshape_name = op.name + str(idx) + "_reshape"
            reshape_op = Operation("Reshape", reshape_name)
            reshape_op.outputs = [out_tens]
            reshape_in = out_tens.clone("_reshaped")
            reshape_in.shape = reshape_in.storage_shape = reshape_in.bandwidth_shape = reshape_input_shape
            reshape_in.ops = [op]
            out_tens.ops = [reshape_op]
            reshape_op.inputs = [reshape_in, new_shape_tens]

            op.outputs[idx] = reshape_in

    return tens


def add_padding_fields(op, arch):
    if "padding" in op.attrs:
        if "Conv" in op.type:
            kernel_size = op.inputs[1].shape[:2]
            input_shape = op.inputs[0].shape
        elif "Pool" in op.type or "ResizeBilinear" == op.type:
            kernel_size = op.attrs["ksize"][1:3]
            input_shape = op.inputs[0].shape
        elif op.type == "ExtractImagePatches":
            kernel_size = op.attrs["ksizes"][1:3]
            input_shape = op.inputs[0].shape
        else:
            raise UnsupportedFeatureError("Unknown operation that uses padding: {}".format(op.type))

        if op.type == "Conv2DBackpropInputSwitchedBias":
            upscaling_factor = op.outputs[0].shape[1] // input_shape[1]
            padding, skirt = calc_upscaled_padding_and_skirt(
                op.attrs["padding"], kernel_size, op.attrs["strides"], input_shape, upscaling_factor
            )
        else:
            dilation_h, dilation_w = op.get_dilation_h_w()
            dilated_kernel_size = [dilation_h * (kernel_size[0] - 1) + 1, dilation_w * (kernel_size[1] - 1) + 1]
            padding, skirt = calc_padding_and_skirt(
                op.attrs["padding"], dilated_kernel_size, op.attrs["strides"], input_shape
            )

        op.attrs["explicit_padding"] = padding
        op.attrs["skirt"] = skirt

    return op


conv_op = set(("Conv2D", "QuantizedConv2D", "Conv2DBackpropInputSwitchedBias", "Conv2DBiasAct"))
fc_op = set(
    (
        "MatMul",
        "QuantizedMatMul",
        "BlockLSTM",
        "RnnAct",
        "UnidirectionalSequenceRnnAct",
        "BidirectionalSequenceRnnAct",
        "LstmAct",
        "UnidirectionalSequenceLstmAct",
        "BidirectionalSequenceLstmAct",
        "FullyConnectedAct",
    )
)
depthwise_op = set(("DepthwiseConv2dNative", "DepthwiseConv2dBiasAct",))
pool_op = set(
    ("AvgPool", "MaxPool", "QuantizedAvgPool", "QuantizedMaxPool", "AvgPoolAct", "MaxPoolAct", "ResizeBilinear",)
)
elementwise_op = set(("AddAct", "MulAct", "SubAct", "Maximum", "Minimum", "LeakyRelu", "Abs"))
binary_elementwise_op = set(("AddAct", "MulAct", "SubAct", "Maximum", "Minimum"))
activation_ops = set(("Relu", "Relu6", "ReluN1To1", "Sigmoid", "Tanh"))
memory_only_ops = set(("Reshape",))


# Check if the op can be reordered
def get_prepend_op(op):
    inp = op.inputs[0]
    # The op should be reordered between prev_op and prep_op
    prev_op = inp.ops[-1]
    prep_op = None
    while prev_op.type in memory_only_ops and len(prev_op.outputs) == 1 and len(prev_op.outputs[0].consumers()) == 1:
        prep_op = prev_op
        inp = prev_op.inputs[0]
        prev_op = inp.ops[-1]
    if prev_op is not None and len(prev_op.outputs) == 1 and len(prev_op.outputs[0].consumers()) == 1:
        return prep_op

    return None


def mark_npu_block_type(op, arch):
    npu_block_type = NpuBlockType.Default
    if op.type in conv_op:
        npu_block_type = NpuBlockType.ConvolutionMxN
    elif op.type in fc_op:
        npu_block_type = NpuBlockType.VectorProduct
    elif op.type in depthwise_op:
        npu_block_type = NpuBlockType.ConvolutionDepthWise
    elif op.type in pool_op:
        npu_block_type = NpuBlockType.Pooling
    elif op.type in elementwise_op:
        npu_block_type = NpuBlockType.ElementWise

    op.attrs["npu_block_type"] = npu_block_type
    return op


def convert_depthwise_to_conv(op, arch):
    # Depthwise is equivalent to a single conv2d if the ifm depth is 1 and
    # the ofm depth equals the depth multipler.
    # If those conditions are true, then we can perform a simple
    # switch of the operator type (and weight order)

    if ("DepthwiseConv2d" in op.type) and (op.attrs["depth_multiplier"] != 1):
        ifm_tensor = op.inputs[0]
        weight_tensor = op.inputs[1]
        ofm_tensor = op.outputs[0]
        if (ifm_tensor.shape[3] == 1) and (ofm_tensor.shape[3] == op.attrs["depth_multiplier"]):
            # Change op type to Conv2d
            op.type = op.type.replace("DepthwiseConv2d", "Conv2D")
            del op.attrs["channel_multiplier"]
            del op.attrs["depth_multiplier"]

            weight_tensor.quant_values = np.transpose(weight_tensor.quant_values, (0, 1, 3, 2))
            weight_tensor.shape = weight_tensor.storage_shape = weight_tensor.bandwidth_shape = list(
                weight_tensor.quant_values.shape
            )
        else:
            raise UnsupportedFeatureError(
                "Unsupported DepthwiseConv2d with depth_multiplier = {}, ifm channels = {}, ofm channels = {}".format(
                    op.attrs["depth_multiplier"], ifm_tensor.shape[3], ofm_tensor.shape[3]
                )
            )
    return op


def reorder_depthwise_weights(op, arch):
    if "DepthwiseConv2d" in op.type:
        weight_tensor = op.inputs[1]
        weight_tensor.quant_values = np.transpose(weight_tensor.quant_values, (0, 1, 3, 2))
        weight_tensor.shape = weight_tensor.storage_shape = weight_tensor.bandwidth_shape = list(
            weight_tensor.quant_values.shape
        )
        weight_tensor.weight_transpose_depthwise = True

    return op


# Reorder activation op if it's after the memory only operations
def fixup_act_reorder(op, arch):
    if op.type in activation_ops:
        prep_op = get_prepend_op(op)
        if prep_op is not None:
            act_op = op.clone("_reordered")
            act_op.inputs = [prep_op.inputs[0]]
            act_op_out = act_op.inputs[0].clone("_acted")
            act_op_out.quantization = op.outputs[0].quantization.clone()
            act_op_out.ops = [act_op]
            act_op.outputs = [act_op_out]
            prep_op.inputs[0] = act_op_out
            prep_op.outputs[0].quantization = act_op_out.quantization.clone()

            # Mark the op so that it will be removed as passthrough later on
            op.type = "Identity"
    return op


def fixup_elementwise_with_scalars(op, arch):
    if op.type in binary_elementwise_op:
        ifm_tensor, ifm2_tensor, _, _ = op.get_ifm_ifm2_weights_ofm()
        if ifm2_tensor.shape != [] and ifm_tensor.shape != []:
            diff = len(ifm_tensor.shape) - len(ifm2_tensor.shape)
            if diff > 0:
                ifm2_tensor.shape = full_shape(len(ifm_tensor.shape), ifm2_tensor.shape, 1)
            elif diff < 0:
                ifm_tensor.shape = full_shape(len(ifm2_tensor.shape), ifm_tensor.shape, 1)
        elif ifm_tensor.shape == [] and ifm_tensor.quant_values is None:
            # IFM is marked as a scalar, but is a result of an operation; change it to a shape of size 1
            ifm_tensor.shape = len(ifm2_tensor.shape) * [1]
            ifm_tensor.storage_shape = ifm_tensor.shape
        elif ifm2_tensor.shape == [] and ifm2_tensor.quant_values is None:
            # IFM2 is marked as a scalar, but is a result of an operation; change it to a shape of size 1
            ifm2_tensor.shape = len(ifm_tensor.shape) * [1]
            ifm2_tensor.storage_shape = ifm2_tensor.shape
    return op


# Set input/output tensor equivalence to the same id for memory operations
def set_tensor_equivalence(op, arch):
    if op.type == "Reshape":
        eid = op.outputs[0].equivalence_id
        for inp in op.inputs:
            inp.equivalence_id = eid
    return op


def convert_mul_max_to_abs_or_lrelu(op, arch):
    r"""Whenever there is a subgraph with this topology:

       Input    X   For X = -1 or X > 0
       |   \   /    This subgraph can be replaced with either
       |    Mul     an Abs (if X = -1) or a LeakyReLU (if X > 0)
       |   /
       Max
    """

    if op.type == "Maximum":
        # finds the Mul input(s) to the Max
        muls = [i for i in op.inputs if i.ops[0].type == "MulAct"]
        if len(muls) == 1:
            mul = muls[0].ops[0]
        elif len(muls) == 2:
            # In the case both inputs are Muls, find the one with the same input as the Max
            mul = [m for m in muls if len(set(op.inputs + m.ops[0].inputs)) == 1][0].ops[0]
        else:
            # No Mul inputs
            return op

        # make sure the Mul doesn't have any other consumers
        if len(mul.outputs[0].consumers()) != 1:
            return op
        # make sure the Mul doesn't have a faf
        if mul.attrs["fused_activation_function"]:
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
            if const.type != "Const":
                return op
        else:
            return op

        val = const.outputs[0].values
        if val >= 0:
            new_op = "LeakyRelu"
            op.attrs["alpha"] = val
        elif val == -1:
            new_op = "Abs"
        else:
            return op

        op.type = op.type.replace("Maximum", new_op)
        op.name = op.name.replace("Maximum", new_op)
        op.outputs[0].name = op.outputs[0].name.replace("Maximum", new_op)
        op.inputs = [shared_in]
    return op


def add_attrs_to_resizebilinear(op, arch):
    if op.type == "ResizeBilinear" and op.run_on_npu:
        input_tensor = op.inputs[0]
        upscaled_shape = [input_tensor.shape[1] * 2, input_tensor.shape[2] * 2]
        out_shape = op.outputs[0].shape[1:3]
        if not op.attrs["align_corners"] and out_shape == upscaled_shape:
            # this means the output is supposed to be a x2 upscale,
            # so we need to do SAME padding
            op.attrs["padding"] = b"SAME"
        elif op.attrs["align_corners"] and out_shape == [upscaled_shape[0] - 1, upscaled_shape[1] - 1]:
            # here we can just run the avg pool without padding and
            # produce a (M * 2 - 1, N * 2 - 1) sized output
            op.attrs["padding"] = b"VALID"
        else:
            # If this exception is raised, something is wrong with the supported op check
            raise UnsupportedFeatureError("Unsupported upscaling factor")
        input_tensor.resampling_mode = resampling_mode.NEAREST
        op.attrs.update({"strides": (1, 1, 1, 1), "ksize": (1, 2, 2, 1)})
    return op


def supported_operator_check(op, arch):
    op.run_on_npu = arch.supported_operators.is_operator_supported(op)
    return op


def optimise_graph_a(nng, arch, verbose_graph=False):
    if verbose_graph:
        nng.print_graph()

    op_rewrite_list = [
        # mark block type and check if the operations are supported
        mark_npu_block_type,
        set_tensor_equivalence,
        supported_operator_check,
        # then do any rewrites of supported operators
        convert_depthwise_to_conv,
        fixup_fully_connected_input,
        fixup_pack_input,
        fixup_conv2d_backprop,
        fixup_act_reorder,
        add_attrs_to_resizebilinear,
        add_padding_fields,
        mark_npu_block_type,
        fixup_elementwise_with_scalars,
        reorder_depthwise_weights,
        # convert_mul_max_to_abs_or_lrelu # TODO: enable optimisation once quantisation issues are resolved
    ]

    for idx, sg in enumerate(nng.subgraphs):
        # rewrite graph pass
        nng.subgraphs[idx] = rewrite_graph.rewrite_graph_pre_order(
            sg, arch, [fixup_unpack_output], op_rewrite_list, rewrite_unsupported=False
        )

    for idx, sg in enumerate(nng.subgraphs):
        # remove passthrough tensors
        nng.subgraphs[idx] = rewrite_graph.rewrite_graph_pre_order(sg, arch, [remove_passthrough_tensor], [])

    if verbose_graph:
        nng.print_graph()
    return nng


def optimise_graph_b(nng, arch, verbose_graph=False):
    if verbose_graph:
        nng.print_graph()

    for idx, sg in enumerate(nng.subgraphs):
        # combined rewrite graph pass
        nng.subgraphs[idx] = rewrite_graph.rewrite_graph_pre_order(sg, arch, [rewrite_concat, rewrite_split], [])

    if verbose_graph:
        nng.print_graph()
    return nng
