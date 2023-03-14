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
# Mark purpose and select formats for Tensors.
from .graph_optimiser_util import memory_only_ops
from .operation import CustomType
from .operation import Op
from .rewrite_graph import visit_graph_post_order
from .tensor import MemType
from .tensor import TensorFormat
from .tensor import TensorPurpose


def get_format(purpose, arch):
    if purpose in (TensorPurpose.FeatureMap, TensorPurpose.LUT, TensorPurpose.Scratch, TensorPurpose.ScratchFast):
        fmt = arch.default_feature_map_format
    elif purpose == TensorPurpose.Weights:
        fmt = arch.default_weight_format
    elif purpose == TensorPurpose.Unknown:
        fmt = TensorFormat.Unknown
    else:
        assert 0, "unknown tensor purpose {}".format(purpose)
    return fmt


def mark_purpose(tens, arch, purpose):
    # Sets tensor's purpose, format, mem_area and mem_type
    if tens.purpose == TensorPurpose.Unknown:
        tens.purpose = purpose
    elif tens.purpose == TensorPurpose.Virtual:
        return
    elif tens.purpose not in (purpose, TensorPurpose.LUT):
        assert 0, "Cannot resolve tensor purpose {} and {} for tensor {}".format(tens.purpose, purpose, tens)

    fmt = get_format(purpose, arch)
    tens.set_format(fmt, arch)
    tens.mem_area = arch.tensor_storage_mem_area[tens.purpose]
    tens.mem_type = arch.tensor_storage_mem_type[tens.purpose]

    if (
        len(tens.ops) == 1
        and tens.ops[0].type == Op.Const
        and purpose not in (TensorPurpose.Scratch, TensorPurpose.ScratchFast)
    ):
        tens.mem_area = arch.permanent_storage_mem_area  # special case constants, as they must be in permanent storage
        tens.mem_type = MemType.Permanent_NPU


def rewrite_mark_tensor_purpose(op, arch):
    # find disconnected outputs and mark as feature maps
    for tens in op.outputs:
        if not tens.consumers():
            mark_purpose(tens, arch, TensorPurpose.FeatureMap)
    weight_tensors = op.get_weight_tensors()
    for tens in op.inputs:
        if tens is None:
            continue
        if tens.purpose != TensorPurpose.Unknown:
            purpose = tens.purpose
        elif tens in weight_tensors:
            purpose = TensorPurpose.Weights
        else:
            purpose = TensorPurpose.FeatureMap
        # Treat Dynamic Weights as FeatureMap to avoid issues during scheduling caused by
        # having non constant OPs that produce tensors used as weights.
        if any(op.type != Op.Const and tens == op.ofm and purpose == TensorPurpose.Weights for op in tens.ops):
            purpose = TensorPurpose.FeatureMap
        mark_purpose(tens, arch, purpose)
    if op.type in memory_only_ops:
        # Memory only operator input and output point to same data
        op.ofm.mem_area = op.ifm.mem_area

    if op.type == Op.Custom and op.attrs.get("custom_type") == CustomType.ExistingNpuOp:
        scratch_tensor = None

        if len(op.inputs) >= 3:
            scratch_tensor = op.inputs[2]  # should be existing scratch tensor
            if scratch_tensor.name.endswith("_scratch"):
                scratch_tensor.purpose = TensorPurpose.Scratch

        if len(op.inputs) >= 4:
            scratch_fast_tensor = op.inputs[3]  # should be existing scratch fast tensor
            if scratch_fast_tensor.name.endswith("_scratch_fast"):
                scratch_fast_tensor.purpose = TensorPurpose.ScratchFast

        if scratch_tensor is None:
            op.error("Scratch tensor not found.")


def mark_tensor_purpose(nng, arch, verbose_tensor_purpose=False):
    # Sets purpose, format, mem_area and mem_type for all tensors in the graph
    for sg in nng.subgraphs:
        visit_graph_post_order(sg.output_tensors, arch, [], [rewrite_mark_tensor_purpose])
        for tens in sg.output_tensors:
            mark_purpose(tens, arch, TensorPurpose.FeatureMap)

    if verbose_tensor_purpose:
        nng.print_graph_with_tensors()

    return nng
