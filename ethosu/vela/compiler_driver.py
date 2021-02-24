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
# Contains the main sequencing of the compiler.
import time

from . import extract_npu_subgraphs
from . import graph_optimiser
from . import high_level_command_stream_generator
from . import high_level_command_to_npu_op
from . import insert_dma
from . import live_range
from . import lut
from . import mark_tensors
from . import npu_performance
from . import npu_serialisation
from . import pass_packing
from . import scheduler
from . import tensor_allocation
from . import weight_compressor
from .debug_database import DebugDatabase
from .errors import VelaError
from .nn_graph import PassPlacement
from .nn_graph import TensorAllocator
from .operation import Op
from .rewrite_graph import verify_graph_health
from .rewrite_graph import visit_graph_post_order
from .tensor import MemType
from .tensor import Tensor


class CompilerOptions:
    """Set of options to change compiler behaviour - verbosity, targets, turning off passes.

Note the difference between ArchitectureFeatures and CompilerOptions
- ArchitectureFeatures is for changing the Ethos-U and system architecture
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
        show_cpu_operations=False,
        tensor_allocator=TensorAllocator.Greedy,
        timing=False,
        output_dir="outputs",
        cpu_tensor_alignment=Tensor.AllocationQuantum,
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
        self.show_cpu_operations = show_cpu_operations
        self.tensor_allocator = tensor_allocator
        self.timing = timing
        self.output_dir = output_dir
        self.cpu_tensor_alignment = cpu_tensor_alignment

    def __str__(self):
        return type(self).__name__ + ": " + str(self.__dict__)

    __repr__ = __str__


def next_sram_factor(alloc_results):
    # Bisects to find the max SRAM usage that successfully can be fitted with the tensor allocator.
    # Returns tuple (factor, dry_test), with factor is None (stop) or 0 <= factor <= 1 (next SRAM factor to try),
    # dry_test is True while still bisecting.
    upper = 1.0
    lower = 0.7
    MAX_ITERATIONS = 8
    if len(alloc_results) == 0:
        # First iteration, try max SRAM, keep the result if it succeeds
        return (upper, False)
    elif len(alloc_results) == 1:
        if alloc_results[0]:
            # The allocator succeeded at first try; stop
            return (None, False)
        else:
            # Start bisecting, try lowerbound SRAM
            return (lower, True)
    elif len(alloc_results) > MAX_ITERATIONS:
        # Stop
        return (None, False)
    if not alloc_results[1]:
        # Allocation at lower failed; search interval 0 - lower
        upper = lower
        lower = 0
    best = lower
    for success in alloc_results[2:]:
        middle = (lower + upper) / 2
        if success:
            best = max(best, middle)
            lower = middle
        else:
            upper = middle
    if len(alloc_results) == MAX_ITERATIONS:
        # Done bisecting; repeat the best match, but not as dry test
        return (best, False)
    # Next try; run only as dry test
    return ((lower + upper) / 2, True)


def _record_operator(op, arch):
    if op.type != Op.Const:
        DebugDatabase.add_source(op)


def compiler_driver(nng, arch, options, scheduler_options):
    assert verify_graph_health(nng)

    # Pre-optimisation operator tracking
    for sg in nng.subgraphs:
        visit_graph_post_order(sg.output_tensors, arch, [], [_record_operator])

    nng = graph_optimiser.optimise_graph_a(nng, arch, options.verbose_graph)
    assert verify_graph_health(nng)

    if options.verbose_quantization:
        nng.print_graph_with_tensor_quantization()

    nng = mark_tensors.mark_tensor_purpose(nng, arch, options.verbose_tensor_purpose)
    assert verify_graph_health(nng)
    nng = insert_dma.insert_dma_commands(nng, arch, options.verbose_graph)
    assert verify_graph_health(nng)
    pass_packing.pack_into_passes(nng, arch, options.verbose_packing)
    assert verify_graph_health(nng)

    extract_npu_subgraphs.extract_npu_subgraphs(nng, arch)

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

    if scheduler_options.cache_bias_scale_tensor:
        scheduler.move_scales_to_fast_storage(nng, arch)

    # LiveRanges for constant tensors for all Npu subgraphs
    permanent_storage = arch.permanent_storage_mem_area
    lr_graph_flash = live_range.LiveRangeGraph()

    # Placeholders for scratch and flash tensors that are common for all Npu subgraphs
    scratch_tens = None
    scratch_fast_tens = None
    flash_tens = None

    # Calculate live ranges for all constant Npu tensors, in permanent storage
    for sg in nng.subgraphs:
        if sg.placement == PassPlacement.Npu:
            lr_graph_flash = live_range.extract_live_ranges_from_cascaded_passes(
                sg,
                permanent_storage,
                MemType.Permanent_NPU,
                ignore_subgraph_input_output_tensors=True,
                lr_graph=lr_graph_flash,
            )

    if len(nng.subgraphs) > 1:
        # Allocate all Npu constant tensors to the first Npu subgraph since it is
        # processed first during serialization into tensors
        first_npu_sg = nng.subgraphs[1]
        assert first_npu_sg.placement == PassPlacement.Npu
        tensor_allocation.allocate_tensors(
            nng,
            first_npu_sg,
            arch,
            permanent_storage,
            set((MemType.Permanent_NPU,)),
            tensor_allocator=TensorAllocator.LinearAlloc,
            verbose_allocation=options.verbose_allocation,
            lr_graph=lr_graph_flash,
        )

    # Allocate all non-constant tensors to the root, i.e. Cpu, subgraph. This step
    # will start at the root subgraph's input and traverse from top to bottom. When
    # it comes across an Npu-op it will extract live ranges for it's corresponding
    # Npu subgraph and add them to the root's live range graph.
    # The non-constant tensors are stored either in arch.feature_map_storage_mem_area or
    # arch.fast_storage_mem_area.
    # When these memory areas are the same, all non-constant tensors are allocated together.
    # Otherwise they are allocated separately.

    root_sg = nng.get_root_subgraph()

    alloc_list = []
    if arch.is_spilling_enabled():
        mem_alloc_scratch_fast = (arch.fast_storage_mem_area, set((MemType.Scratch_fast,)))
        mem_alloc_scratch = (arch.feature_map_storage_mem_area, set((MemType.Scratch,)))
        # Order is important
        alloc_list.append(mem_alloc_scratch_fast)
        alloc_list.append(mem_alloc_scratch)
    else:
        mem_alloc_scratch = (arch.feature_map_storage_mem_area, set((MemType.Scratch, MemType.Scratch_fast)))
        alloc_list.append(mem_alloc_scratch)

    for mem_area, mem_type_set in alloc_list:
        if arch.is_spilling_enabled() and mem_area == arch.fast_storage_mem_area:
            # For the case where scratch_fast != scratch: attempt to place feature maps used between
            # cascaded passes in fast storage. Bisection is used to find the max possible usage of SRAM.
            alloc_results = []
            while True:
                assert len(alloc_results) < 10, "Infinite allocator loop"
                sram_factor, dry_test = next_sram_factor(alloc_results)
                if sram_factor is None:
                    break
                # Try to move as many feature maps as possible to SRAM before allocating
                sram_limit = sram_factor * arch.sram_size
                for sg in nng.subgraphs:
                    scheduler.use_fast_storage_for_feature_maps(sg, sram_limit, arch)
                alloc_success = tensor_allocation.allocate_tensors(
                    nng,
                    root_sg,
                    arch,
                    mem_area,
                    mem_type_set,
                    max_size=arch.sram_size,
                    dry_test=dry_test,
                    tensor_allocator=options.tensor_allocator,
                    verbose_allocation=options.verbose_allocation,
                    cpu_tensor_alignment=options.cpu_tensor_alignment,
                )
                if dry_test or not alloc_success:
                    for sg in nng.subgraphs:
                        scheduler.undo_use_fast_storage(sg, arch)
                alloc_results.append(alloc_success)
            if not alloc_results[-1]:
                raise VelaError(
                    f"Sram limit {arch.sram_size} bytes, has been exceeded by the scratch fast tensor. "
                    "Increasing the value of --weight-estimation-scaling may help to resolve the issue. "
                    "See OPTIONS.md for more information"
                )
        else:
            tensor_allocation.allocate_tensors(
                nng,
                root_sg,
                arch,
                mem_area,
                mem_type_set,
                tensor_allocator=options.tensor_allocator,
                verbose_allocation=options.verbose_allocation,
                cpu_tensor_alignment=options.cpu_tensor_alignment,
            )

    # Generate command streams and serialise Npu-ops into tensors
    for sg in nng.subgraphs:
        high_level_command_stream_generator.generate_high_level_command_stream(
            nng, sg, arch, options.verbose_high_level_command_stream
        )
        lut.optimize_high_level_cmd_stream(sg, arch)
        high_level_command_to_npu_op.generate_register_command_stream_for_sg(
            nng, sg, arch, options.verbose_register_command_stream
        )
        scratch_tens, scratch_fast_tens, flash_tens = npu_serialisation.serialise_npu_subgraph_into_tensors(
            nng, sg, arch, scratch_tens, scratch_fast_tens, flash_tens
        )

    npu_serialisation.rewrite_npu_call_ops(nng, root_sg, arch)

    # Set Scratch and Fast_scratch Tensor size
    if scratch_tens is not None:
        scratch_tens.set_all_shapes([root_sg.memory_used_per_type.get(MemType.Scratch, 0)])
    if scratch_fast_tens is not None:
        scratch_fast_tens.set_all_shapes([root_sg.memory_used_per_type.get(MemType.Scratch_fast, 0)])

    # Allocate all Cpu constant tensors, this is done last because the Npu-ops
    # have to be serialized into flash and scratch tensors first
    tensor_allocation.allocate_tensors(
        nng,
        root_sg,
        arch,
        permanent_storage,
        set((MemType.Permanent_CPU,)),
        tensor_allocator=TensorAllocator.LinearAlloc,
        verbose_allocation=options.verbose_allocation,
        cpu_tensor_alignment=options.cpu_tensor_alignment,
    )

    npu_performance.calc_performance_for_network(nng, arch)
