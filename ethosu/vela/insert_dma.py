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
from .operation import Op
from .operation import Operation
from .tensor import MemArea
from .tensor import MemType
from .tensor import TensorPurpose
from .weight_compressor import compress_weights


def weights_fit_sram(arch, op, tens, nng):
    # Compresses weights and checks if they fit in SRAM
    if tens.purpose != TensorPurpose.Weights:
        return True

    min_weight_size = 0
    if len(tens.shape) == 4:
        min_weight_size = tens.shape[0] * tens.shape[1] * tens.shape[2] * arch.OFMSplitDepth
    elif len(tens.shape) == 2:
        min_weight_size = tens.shape[0] * arch.OFMSplitDepth

    compress_weights(arch, nng, tens, op.type.npu_block_type, 16, 16, op.get_dilation_h_w())

    # Need to be fit into Sram, as a double buffer
    worst_buffer_size = tens.compression_scale_for_worst_weight_stream * min_weight_size * 2
    if worst_buffer_size > arch.sram_size:
        print(
            "Weights, {}, are too big to be DMAed to SRAM, estimated minimum size is {} bytes".format(
                tens.name, worst_buffer_size
            )
        )
        return False
    return True


def insert_dma_cmd(op, arch, nng):
    if op.type == Op.DMA or not op.run_on_npu:
        return op

    is_lut_used = any(inp.purpose == TensorPurpose.LUT for inp in op.inputs)
    max_ifm_shram_avail = (
        (arch.available_shram_banks(is_lut_used) - arch.shram_reserved_output_banks) * arch.shram_bank_size // 2
    )

    for idx, tens in enumerate(op.inputs):

        if tens.mem_type not in (MemType.Scratch, MemType.Scratch_fast):
            # Tensor is in permanent storage
            # Only when permanent storage differs from fast storage, there is a point moving the data
            if (
                tens.mem_area in (MemArea.Dram, MemArea.OffChipFlash)
                and arch.permanent_storage_mem_area != arch.fast_storage_mem_area
            ) or tens.purpose == TensorPurpose.LUT:
                if tens.purpose in (TensorPurpose.Weights, TensorPurpose.LUT) or (
                    tens.purpose == TensorPurpose.FeatureMap
                    and op.type.is_binary_elementwise_op()
                    and tens.shape != []
                    and op.ifm_shapes[0] != op.ofm_shapes[0]
                    and tens.storage_size() > max_ifm_shram_avail
                ):
                    only_vector_product_consumers = True
                    for oper in tens.consumers():
                        if oper is None or oper.type.npu_block_type != NpuBlockType.VectorProduct:
                            only_vector_product_consumers = False
                            break

                    # Tensor products has no need for DMA, tensors are only read once and can be in flash.
                    # Other operations re-reads tensors, this is better done from SRAM.
                    # LUTs must be placed in the last 2 blocks of SHRAM.
                    if (
                        not only_vector_product_consumers and weights_fit_sram(arch, op, tens, nng)
                    ) or tens.purpose == TensorPurpose.LUT:
                        # Insert a DMA command here, as well as a new tensor situated in SRAM of the same size.
                        new_tens = tens.clone_into_fast_storage(arch)
                        dma_cmd = Operation(Op.DMA, tens.ops[0].name + "_dma")
                        dma_cmd.inputs = [tens]
                        dma_cmd.set_output_tensor(new_tens)
                        dma_cmd.attrs["source"] = tens.mem_area
                        dma_cmd.attrs["destination"] = new_tens.mem_area
                        dma_cmd.run_on_npu = True
                        if tens.purpose == TensorPurpose.LUT:
                            new_tens.mem_area = MemArea.Shram
                        op.inputs[idx] = new_tens
    return op


def insert_dma_commands(nng, arch, verbose_graph=False):

    for idx, sg in enumerate(nng.subgraphs):
        nng.subgraphs[idx] = rewrite_graph.rewrite_graph_pre_order(nng, sg, arch, [], [insert_dma_cmd])
    if verbose_graph:
        nng.print_graph()
    return nng
