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
# Utility functions for creating Network Operations.
from typing import Optional
from typing import Tuple

from .data_type import DataType
from .high_level_command_to_npu_op import ifm_ifm2_correct_order
from .operation import ActivationFunction
from .operation import Op
from .operation import Operation
from .operation import Padding
from .shape4d import Shape4D
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
    op.attrs["explicit_padding"] = [0, 0, 0, 0]
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
    op.ifm.avoid_NHCWB16 = True
    op.ofm.avoid_NHCWB16 = True
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


def create_rescale_add(
    name: str,
    ifm: Tensor,
    ifm2: Tensor,
    rescale: Tuple[int, int],
    quantization: QuantizationParameters,
    activation: Optional[ActivationFunction] = None,
    dtype: Optional[DataType] = None,
    attrs: Optional[dict] = None,
    ifm_shape: Optional[Shape4D] = None,
    ifm2_shape: Optional[Shape4D] = None,
) -> Operation:
    op = create_binary_elementwise(
        Op.RescaleAdd, name, ifm, ifm2, quantization, activation, dtype, attrs, ifm_shape, ifm2_shape
    )
    op.rescale = rescale
    return op


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
    ifm2: Tensor,
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
        in_shape = [] if ifm.shape == [] else ifm_shape.as_list()
        in2_shape = [] if ifm2.shape == [] else ifm2_shape.as_list()
        ofm_shape = ifm_shape if ifm_ifm2_correct_order(in_shape, in2_shape) else ifm2_shape

    ofm = Tensor(ofm_shape.as_list(), dtype, f"{op.name}_tens0")
    ofm.quantization = quantization
    op.set_output_tensor(ofm)
    op.ofm_shapes.append(ofm_shape)
    return op
