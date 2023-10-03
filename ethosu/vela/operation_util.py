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
# Utility functions for creating Network Operations.
from typing import Optional
from typing import Tuple

import numpy as np

from .data_type import DataType
from .high_level_command_to_npu_op import ifm_ifm2_correct_order
from .operation import ActivationFunction
from .operation import Op
from .operation import Operation
from .operation import Padding
from .reader_util import clone_and_reshape_tensor
from .shape4d import Shape4D
from .tensor import create_const_tensor
from .tensor import create_equivalence_id
from .tensor import QuantizationParameters
from .tensor import Tensor


def create_avgpool_nop(name: str) -> Operation:
    op = Operation(Op.AvgPool, name)
    op.attrs["padding"] = Padding.VALID
    op.attrs["stride_w"] = 1
    op.attrs["stride_h"] = 1
    op.attrs["filter_width"] = 1
    op.attrs["filter_height"] = 1
    op.attrs["strides"] = [1, 1, 1, 1]
    op.attrs["ksize"] = [1, 1, 1, 1]
    op.attrs["skirt"] = [0, 0, 0, 0]
    op.attrs["explicit_padding"] = [0, 0, 0, 0]  # [top, left, bottom, right]
    op.run_on_npu = True
    return op


def create_add_nop(name: str) -> Operation:
    op = Operation(Op.Add, name)
    op.run_on_npu = True
    return op


def create_memcpy(
    name: str,
    ifm: Tensor,
    ofm: Tensor,
) -> Operation:
    op = Operation(Op.Memcpy, name)
    op.run_on_npu = True
    op.add_input_tensor(ifm)
    op.set_output_tensor(ofm)
    op.set_ifm_ofm_shapes()
    return op


def create_pad_nop(name: str) -> Operation:
    op = Operation(Op.Pad, name)
    op.run_on_npu = True
    return op


def create_cast_op(
    name: str,
    ifm: Tensor,
    ofm: Tensor,
) -> Operation:
    op = Operation(Op.DepthwiseConv2DBias, name)
    op_attrs = {
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
    op.attrs.update(op_attrs)
    op.add_input_tensor(ifm)

    c = ifm.shape[-1]

    # Weigth shape is in format [h, w, b, c] for DepthwiseConv2D
    shape = [1, 1, 1, c]
    kernel = np.dstack([1] * c)
    identity_quant = QuantizationParameters(scale_f32=1.0, zero_point=0)
    op.add_input_tensor(
        create_const_tensor(
            op.name + "_weights",
            shape,
            DataType.uint8,
            np.array(kernel).reshape(shape),
            quantization=identity_quant,
        ),
    )
    # Set flag to indicate that weights are already in correct order
    # and prevent that they are transposed in reorder_depthwise_weights
    op.inputs[1].weight_transpose_depthwise = True
    bias_values = [0] * c
    dtype = DataType.int64 if op.ifm.dtype == DataType.int16 else DataType.int32
    op.add_input_tensor(create_const_tensor(op.name + "_bias", [c], dtype, bias_values))
    op.set_output_tensor(ofm)
    op.set_ifm_ofm_shapes()

    return op


def create_fused_activation(op_type: Op, name: str, ifm: Tensor, quantization: QuantizationParameters) -> Operation:
    assert op_type.is_activation_op()
    op = create_avgpool_nop(name)
    op.activation = ActivationFunction(op_type)
    ofm = Tensor(ifm.shape, ifm.dtype, f"{op.name}_tens0")
    ofm.quantization = quantization
    op.add_input_tensor(ifm)
    op.set_output_tensor(ofm)
    op.set_ifm_ofm_shapes()
    return op


def create_fullyconnected(
    name: str,
    ifm: Tensor,
    weights: Tensor,
    bias: Optional[Tensor],
    quantization: QuantizationParameters,
    vela_weight_order: bool = True,
) -> Operation:
    # Reshape weights if needed
    if not vela_weight_order:
        weights = clone_and_reshape_tensor(weights, (1, 0), False)

    n_ofm = weights.shape[-1]

    # Setup bias if needed
    if not bias:
        bias_values = [0] * n_ofm
        dtype = DataType.int64 if ifm.dtype == DataType.int16 else DataType.int32
        bias = create_const_tensor(f"{name}_bias", [n_ofm], dtype, bias_values)
        # Set equivalence_id based on values to avoid placing duplicate data in flash
        bias.equivalence_id = create_equivalence_id(tuple(bias_values))
        bias.value_id = bias.equivalence_id

    # Setup ofm
    ofm = Tensor([ifm.shape[0], n_ofm], ifm.dtype, f"{name}_tens0")
    ofm.quantization = quantization

    # Create op and add tensors
    op = Operation(Op.FullyConnected, name)
    op.add_input_tensor(ifm)
    op.add_input_tensor(weights)
    op.add_input_tensor(bias)
    op.set_output_tensor(ofm)
    op.set_ifm_ofm_shapes()
    return op


def create_depthwise_maxpool(
    name: str,
    ifm: Tensor,
    inp_shape: Shape4D,
    quantization: QuantizationParameters,
    activation: Optional[ActivationFunction] = None,
) -> Operation:
    op = Operation(Op.MaxPool, name)
    height = inp_shape.height * inp_shape.width
    width = inp_shape.depth
    ifm_shape = Shape4D([1, height, width, 1])

    op.attrs["padding"] = Padding.VALID
    op.attrs["stride_w"] = 1
    op.attrs["stride_h"] = 1
    op.attrs["filter_width"] = width
    op.attrs["filter_height"] = 1
    op.attrs["strides"] = [1, op.attrs["stride_h"], op.attrs["stride_w"], 1]
    op.attrs["ksize"] = [1, op.attrs["filter_height"], op.attrs["filter_width"], 1]
    op.activation = activation
    op.inputs = [ifm]
    ofm = Tensor([1, height, 1, 1], ifm.dtype, op.name + "_tens0")
    ofm.quantization = quantization
    op.set_output_tensor(ofm)
    op.ifm_shapes.append(ifm_shape)
    op.ofm_shapes.append(Shape4D(ofm.shape))
    return op


def create_reduce_sum(
    name: str, ifm: Tensor, quantization: QuantizationParameters, activation: Optional[ActivationFunction] = None
) -> Operation:
    op = Operation(Op.ReduceSum, name)
    op.attrs["padding"] = Padding.VALID
    op.attrs["stride_w"] = 1
    op.attrs["stride_h"] = 1
    op.attrs["filter_width"] = 1
    op.attrs["filter_height"] = 1
    op.attrs["strides"] = [1, op.attrs["stride_h"], op.attrs["stride_w"], 1]
    op.attrs["ksize"] = [1, op.attrs["filter_height"], op.attrs["filter_width"], 1]
    op.add_input_tensor(ifm)
    op.activation = activation
    ofm_shape = [1, ifm.shape[1], ifm.shape[2], 1]
    sum_of_exp = Tensor(ofm_shape, DataType.int32, op.name + "_tens0")
    sum_of_exp.quantization = quantization
    op.set_output_tensor(sum_of_exp)
    op.set_ifm_ofm_shapes()
    return op


def create_add(
    name: str,
    ifm: Tensor,
    ifm2: Tensor,
    quantization: QuantizationParameters,
    activation: Optional[ActivationFunction] = None,
    dtype: Optional[DataType] = None,
    attrs: Optional[dict] = None,
    ifm_shape: Optional[Shape4D] = None,
    ifm2_shape: Optional[Shape4D] = None,
) -> Operation:
    return create_binary_elementwise(
        Op.Add, name, ifm, ifm2, quantization, activation, dtype, attrs, ifm_shape, ifm2_shape
    )


def create_clz(
    name: str,
    ifm: Tensor,
    quantization: QuantizationParameters,
    activation: Optional[ActivationFunction] = None,
    dtype: Optional[DataType] = None,
    attrs: Optional[dict] = None,
    ifm_shape: Optional[Shape4D] = None,
) -> Operation:
    return create_unary_elementwise(Op.CLZ, name, ifm, quantization, activation, dtype, attrs, ifm_shape)


def create_mul(
    name: str,
    ifm: Tensor,
    ifm2: Tensor,
    quantization: QuantizationParameters,
    activation: Optional[ActivationFunction] = None,
    dtype: Optional[DataType] = None,
    attrs: Optional[dict] = None,
    ifm_shape: Optional[Shape4D] = None,
    ifm2_shape: Optional[Shape4D] = None,
) -> Operation:
    return create_binary_elementwise(
        Op.Mul, name, ifm, ifm2, quantization, activation, dtype, attrs, ifm_shape, ifm2_shape
    )


def create_shl(
    name: str,
    ifm: Tensor,
    ifm2: Tensor,
    quantization: QuantizationParameters,
    activation: Optional[ActivationFunction] = None,
    dtype: Optional[DataType] = None,
    attrs: Optional[dict] = None,
    ifm_shape: Optional[Shape4D] = None,
    ifm2_shape: Optional[Shape4D] = None,
) -> Operation:
    return create_binary_elementwise(
        Op.SHL, name, ifm, ifm2, quantization, activation, dtype, attrs, ifm_shape, ifm2_shape
    )


def create_shr(
    name: str,
    ifm: Tensor,
    ifm2: Tensor,
    quantization: QuantizationParameters,
    activation: Optional[ActivationFunction] = None,
    dtype: Optional[DataType] = None,
    attrs: Optional[dict] = None,
    ifm_shape: Optional[Shape4D] = None,
    ifm2_shape: Optional[Shape4D] = None,
) -> Operation:
    return create_binary_elementwise(
        Op.SHR, name, ifm, ifm2, quantization, activation, dtype, attrs, ifm_shape, ifm2_shape
    )


def create_sub(
    name: str,
    ifm: Tensor,
    ifm2: Tensor,
    quantization: QuantizationParameters,
    activation: Optional[ActivationFunction] = None,
    dtype: Optional[DataType] = None,
    attrs: Optional[dict] = None,
    ifm_shape: Optional[Shape4D] = None,
    ifm2_shape: Optional[Shape4D] = None,
) -> Operation:
    return create_binary_elementwise(
        Op.Sub, name, ifm, ifm2, quantization, activation, dtype, attrs, ifm_shape, ifm2_shape
    )


def create_unary_elementwise(
    op_type: Op,
    name: str,
    ifm: Tensor,
    quantization: QuantizationParameters,
    activation: Optional[ActivationFunction] = None,
    dtype: Optional[DataType] = None,
    attrs: Optional[dict] = None,
    ifm_shape: Optional[Shape4D] = None,
) -> Operation:
    return create_binary_elementwise(op_type, name, ifm, None, quantization, activation, dtype, attrs, ifm_shape, None)


def create_binary_elementwise(
    op_type: Op,
    name: str,
    ifm: Tensor,
    ifm2: Optional[Tensor],
    quantization: QuantizationParameters,
    activation: Optional[ActivationFunction] = None,
    dtype: Optional[DataType] = None,
    attrs: Optional[dict] = None,
    ifm_shape: Optional[Shape4D] = None,
    ifm2_shape: Optional[Shape4D] = None,
) -> Operation:
    if ifm_shape is None:
        ifm_shape = Shape4D(ifm.shape)
    op = Operation(op_type, name)
    op.add_input_tensor(ifm)
    op.ifm_shapes.append(ifm_shape)
    if ifm2:
        op.add_input_tensor(ifm2)
        if ifm2_shape is None:
            ifm2_shape = Shape4D(ifm2.shape)
        op.ifm_shapes.append(ifm2_shape)
    op.activation = activation
    if not dtype:
        dtype = ifm.dtype
    if attrs:
        op.attrs.update(attrs)

    if ifm2 is None:
        ofm_shape = ifm_shape
    else:
        in_shape = None if ifm.shape == [] else ifm_shape
        in2_shape = None if ifm2.shape == [] else ifm2_shape
        ofm_shape = ifm_shape if ifm_ifm2_correct_order(in_shape, in2_shape) else ifm2_shape

    ofm = Tensor(ofm_shape.as_list(), dtype, f"{op.name}_tens0")
    ofm.quantization = quantization
    op.set_output_tensor(ofm)
    op.ofm_shapes.append(ofm_shape)
    return op


def get_pad_values_from_input(padding) -> Tuple:
    """Returns top, left, bottom, right padding from input values in a Pad input tensor"""
    return (padding[-3][0], padding[-2][0], padding[-3][1], padding[-2][1])
