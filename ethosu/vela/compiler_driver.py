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
# Contains the main sequencing of the compiler.

from . import graph_optimiser
from . import mark_tensors
from . import insert_dma
from . import pass_packing
from . import scheduler
from . import tensor_allocation
from . import npu_performance
import time

from . import high_level_command_stream
from . import high_level_command_stream_generator
from . import register_command_stream_generator
from . import extract_npu_subgraphs
from . import npu_serialisation
from . import weight_compressor
from . import live_range
from .tensor import MemArea
from .nn_graph import TensorAllocator, PassPlacement
from .rewrite_graph import verify_graph_health, verify_subgraph_health


class CompilerOptions:
    """Set of options to change compiler behaviour - verbosity, targets, turning off passes.

Note the difference between ArchitectureFeatures and CompilerOptions
- ArchitectureFeatures is for changing the Ethos-U55 and system architecture
- CompilerOptions is for changing the behaviour of the compiler
"""

    def __init__(
        self,
        verbose_graph=False,
        verbose_quantization=False,
        verbose_packing=False,
        verbose_tensor_purpose=False,
        verbose_tensor_format=False,
        verbose_allocation=False,
        verbose_high_level_command_stream=False,
        verbose_register_command_stream=False,
        verbose_operators=False,
        show_minimum_possible_allocation=False,
        show_cpu_operations=False,
        tensor_allocator=TensorAllocator.Greedy,
        timing=False,
        output_dir="outputs",
    ):

        self.verbose_graph = verbose_graph
        self.verbose_quantization = verbose_quantization
        self.verbose_packing = verbose_packing
        self.verbose_tensor_purpose = verbose_tensor_purpose
        self.verbose_tensor_format = verbose_tensor_format
        self.verbose_allocation = verbose_allocation
        self.verbose_high_level_command_stream = verbose_high_level_command_stream
        self.verbose_register_command_stream = verbose_register_command_stream
        self.verbose_operators = verbose_operators
        self.show_minimum_possible_allocation = show_minimum_possible_allocation
        self.show_cpu_operations = show_cpu_operations
        self.tensor_allocator = tensor_allocator
        self.timing = timing
        self.output_dir = output_dir

    def __str__(self):
        return type(self).__name__ + ": " + str(self.__dict__)

    __repr__ = __str__


def compiler_driver(nng, arch, options, scheduler_options):
    assert verify_graph_health(nng)
    nng = graph_optimiser.optimise_graph_a(nng, arch, options.verbose_graph)
    assert verify_graph_health(nng)

    if options.verbose_quantization:
        nng.print_graph_with_tensor_quantization()

    nng = graph_optimiser.optimise_graph_b(nng, arch, options.verbose_graph)
    assert verify_graph_health(nng)

    nng = mark_tensors.mark_tensor_purpose(nng, arch, options.verbose_tensor_purpose)
    assert verify_graph_health(nng)
    nng = insert_dma.insert_dma_commands(nng, arch, options.verbose_graph)
    assert verify_graph_health(nng)
    pass_packing.pack_into_passes(nng, arch, options.verbose_packing)
    assert verify_graph_health(nng)

    extract_npu_subgraphs.extract_npu_subgraphs(nng, arch)

    mark_tensors.mark_tensor_format(nng, arch, options.verbose_tensor_format)
    assert verify_graph_health(nng)
    if options.timing:
        start = time.time()

    # Run the scheduler
    scheduler.schedule_passes(nng, arch, scheduler_options)

    if options.timing:
        stop = time.time()
        print("Scheduling took %f s" % (stop - start))
        start = time.time()

    # Update the compressed weights now that we have determined the
    # block config, and calc and pack the scales and biases
    weight_compressor.update_pass_weight_and_scale_tensors(nng, arch)

    # Memory area for all non-constant tensors (Cpu and Npu)
    non_const_mem_area = MemArea.Sram

    # LiveRanges for constant tensors for all Npu subgraphs
    permanent_storage = arch.permanent_storage_mem_area
    lr_graph_flash = live_range.LiveRangeGraph()

    # Placeholders for scratch and flash tensors that are common for all Npu subgraphs
    scratch_tens = None
    flash_tens = None

    # Calculate live ranges for all constant Npu tensors, in permanent storage
    for sg in nng.subgraphs:
        if sg.placement == PassPlacement.Npu:
            lr_graph_flash = live_range.extract_live_ranges_from_cascaded_passes(
                sg, permanent_storage, ignore_subgraph_input_output_tensors=True, lr_graph=lr_graph_flash
            )

    # Allocate all Npu constant tensors to the first Npu subgraph since it is
    # processed first during serialization into tensors
    first_npu_sg = nng.subgraphs[1]
    assert first_npu_sg.placement == PassPlacement.Npu
    tensor_allocation.allocate_tensors(
        nng,
        first_npu_sg,
        arch,
        permanent_storage,
        scheduler_options.use_ifm_ofm_overlap,
        options.tensor_allocator,
        options.verbose_allocation,
        options.show_minimum_possible_allocation,
        lr_graph_flash,
    )

    # Allocate all non-constant tensors to the root, i.e. Cpu, subgraph. This step
    # will start at the root subgraph's input and traverse from top to bottom. When
    # it comes across an Npu-op it will extract live ranges for it's corresponding
    # Npu subgraph and add them to the root's live range graph. Finally, all of the
    # non-constant tensors are allocated together
    root_sg = nng.get_root_subgraph()
    tensor_allocation.allocate_tensors(
        nng,
        root_sg,
        arch,
        non_const_mem_area,
        scheduler_options.use_ifm_ofm_overlap,
        options.tensor_allocator,
        options.verbose_allocation,
        options.show_minimum_possible_allocation,
    )

    # Generate command streams and serialise Npu-ops into tensors
    for sg in nng.subgraphs:
        high_level_command_stream_generator.generate_high_level_command_stream(
            nng, sg, arch, options.verbose_high_level_command_stream
        )
        register_command_stream_generator.generate_register_command_stream(
            nng, sg, arch, options.verbose_register_command_stream
        )
        scratch_tens, flash_tens = npu_serialisation.serialise_npu_subgraph_into_tensors(
            nng, sg, arch, scratch_tens, flash_tens
        )

    npu_serialisation.rewrite_npu_call_ops(nng, root_sg, arch)

    # Allocate all Cpu constant tensors, this is done last because the Npu-ops
    # have to be serialized into flash and scratch tensors first
    tensor_allocation.allocate_tensors(
        nng,
        root_sg,
        arch,
        permanent_storage,
        scheduler_options.use_ifm_ofm_overlap,
        options.tensor_allocator,
        options.verbose_allocation,
        options.show_minimum_possible_allocation,
    )

    npu_performance.calc_performance_for_network(nng, arch)
