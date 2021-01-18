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
# Utility functions for creating Network Operations.
from typing import Optional
from typing import Tuple

from .data_type import DataType
from .high_level_command_to_npu_op import ifm_ifm2_correct_order
from .operation import ActivationFunction
from .operation import Op
from .operation import Operation
from .operation import Padding
from .tensor import create_reshape_tensor
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
    name: str, ifm: Tensor, quantization: QuantizationParameters, activation: Optional[ActivationFunction] = None
) -> Operation:
    op = Operation(Op.MaxPool, name)
    height = ifm.shape[1] * ifm.shape[2]
    width = ifm.shape[3]
    ifm_shape = [1, height, width, 1]
    op.attrs["padding"] = Padding.VALID
    op.attrs["stride_w"] = 1
    op.attrs["stride_h"] = 1
    op.attrs["filter_width"] = width
    op.attrs["filter_height"] = 1
    op.attrs["strides"] = [1, op.attrs["stride_h"], op.attrs["stride_w"], 1]
    op.attrs["ksize"] = [1, op.attrs["filter_height"], op.attrs["filter_width"], 1]
    op.activation = activation
    op.inputs = [create_reshape_tensor(ifm, ifm_shape)]
    ofm = Tensor([1, height, 1, 1], ifm.dtype, op.name + "_tens0")
    ofm.quantization = quantization
    op.set_output_tensor(ofm)
    op.set_ifm_ofm_shapes()
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
) -> Operation:
    return create_binary_elementwise(Op.Add, name, ifm, ifm2, quantization, activation, dtype, attrs)


def create_rescale_add(
    name: str,
    ifm: Tensor,
    ifm2: Tensor,
    rescale: Tuple[int, int],
    quantization: QuantizationParameters,
    activation: Optional[ActivationFunction] = None,
    dtype: Optional[DataType] = None,
    attrs: Optional[dict] = None,
) -> Operation:
    op = create_binary_elementwise(Op.RescaleAdd, name, ifm, ifm2, quantization, activation, dtype, attrs)
    op.rescale = rescale
    return op


def create_clz(
    name: str,
    ifm: Tensor,
    quantization: QuantizationParameters,
    activation: Optional[ActivationFunction] = None,
    dtype: Optional[DataType] = None,
    attrs: Optional[dict] = None,
) -> Operation:
    return create_unary_elementwise(Op.CLZ, name, ifm, quantization, activation, dtype, attrs)


def create_mul(
    name: str,
    ifm: Tensor,
    ifm2: Tensor,
    quantization: QuantizationParameters,
    activation: Optional[ActivationFunction] = None,
    dtype: Optional[DataType] = None,
    attrs: Optional[dict] = None,
) -> Operation:
    return create_binary_elementwise(Op.Mul, name, ifm, ifm2, quantization, activation, dtype, attrs)


def create_shl(
    name: str,
    ifm: Tensor,
    ifm2: Tensor,
    quantization: QuantizationParameters,
    activation: Optional[ActivationFunction] = None,
    dtype: Optional[DataType] = None,
    attrs: Optional[dict] = None,
) -> Operation:
    return create_binary_elementwise(Op.SHL, name, ifm, ifm2, quantization, activation, dtype, attrs)


def create_shr(
    name: str,
    ifm: Tensor,
    ifm2: Tensor,
    quantization: QuantizationParameters,
    activation: Optional[ActivationFunction] = None,
    dtype: Optional[DataType] = None,
    attrs: Optional[dict] = None,
) -> Operation:
    return create_binary_elementwise(Op.SHR, name, ifm, ifm2, quantization, activation, dtype, attrs)


def create_sub(
    name: str,
    ifm: Tensor,
    ifm2: Tensor,
    quantization: QuantizationParameters,
    activation: Optional[ActivationFunction] = None,
    dtype: Optional[DataType] = None,
    attrs: Optional[dict] = None,
) -> Operation:
    return create_binary_elementwise(Op.Sub, name, ifm, ifm2, quantization, activation, dtype, attrs)


def create_unary_elementwise(
    op_type: Op,
    name: str,
    ifm: Tensor,
    quantization: QuantizationParameters,
    activation: Optional[ActivationFunction] = None,
    dtype: Optional[DataType] = None,
    attrs: Optional[dict] = None,
) -> Operation:
    return create_binary_elementwise(op_type, name, ifm, None, quantization, activation, dtype, attrs)


def create_binary_elementwise(
    op_type: Op,
    name: str,
    ifm: Tensor,
    ifm2: Tensor,
    quantization: QuantizationParameters,
    activation: Optional[ActivationFunction] = None,
    dtype: Optional[DataType] = None,
    attrs: Optional[dict] = None,
) -> Operation:
    op = Operation(op_type, name)
    op.add_input_tensor(ifm)
    if ifm2:
        op.add_input_tensor(ifm2)
    op.activation = activation
    if not dtype:
        dtype = ifm.dtype
    if attrs:
        op.attrs.update(attrs)
    ofm_shape = ifm.shape if ifm2 is None or ifm_ifm2_correct_order(ifm.shape, ifm2.shape) else ifm2.shape
    ofm = Tensor(ofm_shape, dtype, f"{op.name}_tens0")
    ofm.quantization = quantization
    op.set_output_tensor(ofm)
    op.set_ifm_ofm_shapes()
    return op
