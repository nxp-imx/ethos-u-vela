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
# Insert DMA operations into the graph for transfering weights.
from . import rewrite_graph
from .operation import NpuBlockType
from .operation import Operation
from .tensor import MemArea
from .tensor import MemType
from .tensor import TensorPurpose


binary_elementwise_op = set(("AddAct", "MulAct", "SubAct", "Maximum", "Minimum"))


def insert_dma_cmd(op, arch):
    if op.type == "DMA":
        return op  # Already rewritten
    for idx, tens in enumerate(op.inputs):

        if tens.mem_type not in (MemType.Scratch, MemType.Scratch_fast):
            # Tensor is in permanent storage
            # Only when permanent storage differs from fast storage, there is a point moving the data
            if tens.mem_area in (MemArea.Dram, MemArea.OffChipFlash) and (
                arch.permanent_storage_mem_area != arch.fast_storage_mem_area
            ):
                if tens.purpose == TensorPurpose.Weights or (
                    tens.purpose == TensorPurpose.FeatureMap and op.type in binary_elementwise_op and tens.shape != []
                ):
                    only_vector_product_consumers = True
                    for oper in tens.consumers():
                        if oper is None or oper.attrs.get("npu_block_type") != NpuBlockType.VectorProduct:
                            only_vector_product_consumers = False
                            break

                    # Tensor products has no need for DMA, tensors are only read once and can be in flash.
                    # Other operations re-reads tensors, this is better done from SRAM.
                    if not only_vector_product_consumers:
                        # Insert a DMA command here, as well as a new tensor situated in SRAM of the same size.
                        new_tens = tens.clone_into_fast_storage(arch)
                        dma_cmd = Operation("DMA", tens.ops[0].name + "_dma")
                        dma_cmd.inputs = [tens]
                        dma_cmd.outputs = [new_tens]
                        dma_cmd.attrs["source"] = tens.mem_area
                        dma_cmd.attrs["destination"] = new_tens.mem_area
                        dma_cmd.run_on_npu = True
                        new_tens.ops = [dma_cmd]
                        op.inputs[idx] = new_tens
    return op


def insert_dma_commands(nng, arch, verbose_graph=False):

    for idx, sg in enumerate(nng.subgraphs):
        nng.subgraphs[idx] = rewrite_graph.rewrite_graph_pre_order(sg, arch, [], [insert_dma_cmd])
    if verbose_graph:
        nng.print_graph()
    return nng
