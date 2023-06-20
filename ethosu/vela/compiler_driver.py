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
# Contains the main sequencing of the compiler.
import time

from . import extract_npu_subgraphs
from . import graph_optimiser
from . import high_level_command_stream_generator
from . import high_level_command_to_npu_op
from . import live_range
from . import lut
from . import mark_tensors
from . import npu_performance
from . import npu_serialisation
from . import pass_packing
from . import scheduler
from . import tensor_allocation
from .debug_database import DebugDatabase
from .nn_graph import PassPlacement
from .nn_graph import TensorAllocator
from .operation import Op
from .rewrite_graph import verify_graph_health
from .rewrite_graph import visit_graph_post_order
from .scheduler import OptimizationStrategy
from .tensor import MemArea
from .tensor import MemType
from .tensor import Tensor
from .utils import progress_print


class CompilerOptions:
    """Set of options to change compiler behaviour - verbosity, targets, turning off passes.

    Note the difference between ArchitectureFeatures and CompilerOptions
    - ArchitectureFeatures is for changing the Ethos-U and system architecture
    - CompilerOptions is for changing the behaviour of the compiler"""

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
        verbose_weights=False,
        verbose_performance=False,
        verbose_progress=False,
        show_cpu_operations=False,
        tensor_allocator=TensorAllocator.Greedy,
        timing=False,
        force_symmetric_int_weights=False,
        output_dir="outputs",
        cpu_tensor_alignment=Tensor.AllocationQuantum,
        hillclimb_max_iterations=None,
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
        self.verbose_weights = verbose_weights
        self.verbose_performance = verbose_performance
        self.verbose_progress = verbose_progress
        self.show_cpu_operations = show_cpu_operations
        self.tensor_allocator = tensor_allocator
        self.timing = timing
        self.force_symmetric_int_weights = force_symmetric_int_weights
        self.output_dir = output_dir
        self.cpu_tensor_alignment = cpu_tensor_alignment
        self.hillclimb_max_iterations = hillclimb_max_iterations

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
    if op.type not in (Op.Const, Op.Placeholder):
        DebugDatabase.add_source(op)


def _check_schedule(nng, arch, scheduler_options):
    # check sram usage for optimisation strategy
    sram_usage = nng.get_root_subgraph().memory_used.get(MemArea.Sram)
    if sram_usage is not None and scheduler_options.optimization_strategy == OptimizationStrategy.Performance:
        if sram_usage > scheduler_options.optimization_sram_limit:
            print(
                f"Warning: SRAM target for arena memory area exceeded."
                f" Target = {scheduler_options.optimization_sram_limit} Bytes,"
                f" Actual = {sram_usage} Bytes"
            )


def compiler_driver(nng, arch, options, scheduler_options, network_type, output_basename, subgraph_output = False):
    assert verify_graph_health(nng)
    verbose_progress = scheduler_options.verbose_progress

    # Pre-optimisation operator tracking
    for sg in nng.subgraphs:
        visit_graph_post_order(sg.output_tensors, arch, [], [_record_operator])

    progress_print(verbose_progress, "Performing graph optimisation")
    nng = graph_optimiser.optimise_graph(
        nng, arch, network_type, options.verbose_graph, options.force_symmetric_int_weights, output_basename, subgraph_output
    )
    assert verify_graph_health(nng)

    if options.verbose_quantization:
        nng.print_graph_with_tensor_quantization()

    progress_print(verbose_progress, "Defining tensor purpose")
    nng = mark_tensors.mark_tensor_purpose(nng, arch, options.verbose_tensor_purpose)
    assert verify_graph_health(nng)

    progress_print(verbose_progress, "Performing pass packing")
    pass_packing.pack_into_passes(nng, arch, options.verbose_packing)
    assert verify_graph_health(nng)

    progress_print(verbose_progress, "Extracting npu subgraphs")
    extract_npu_subgraphs.extract_npu_subgraphs(nng, arch)

    assert verify_graph_health(nng)
    if options.timing:
        start = time.time()

    progress_print(verbose_progress, "Scheduling passes")
    # Run the scheduler
    scheduler.schedule_passes(nng, arch, options, scheduler_options)
    _check_schedule(nng, arch, scheduler_options)

    if options.timing:
        stop = time.time()
        print("Scheduling took %f s" % (stop - start))
        start = time.time()

    # LiveRanges for constant tensors for all Npu subgraphs
    permanent_storage = arch.permanent_storage_mem_area
    lr_graph_flash = live_range.LiveRangeGraph()

    # Placeholders for scratch and flash tensors that are common for all Npu subgraphs
    scratch_tens = None
    scratch_fast_tens = None
    flash_tens = None

    # Create list of NPU subgraphs with same order as the list of all subgraphs
    npu_subgraphs = [sg for sg in nng.subgraphs if sg.placement == PassPlacement.Npu]

    progress_print(verbose_progress, "Calculating live ranges for constant NPU tensors")
    # Calculate live ranges for all constant Npu tensors, in permanent storage
    for sg in npu_subgraphs:
        lr_graph_flash = live_range.create_linear_live_range_graph(
            sg,
            permanent_storage,
            MemType.Permanent_NPU,
            lr_graph=lr_graph_flash,
        )

    if npu_subgraphs:
        progress_print(verbose_progress, "Allocating NPU constant tensors to the first NPU subgraph")
        # Allocate all Npu constant tensors to the first Npu subgraph since it is
        # processed first during serialization into tensors
        first_npu_sg = npu_subgraphs[0]
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

    root_sg = nng.get_root_subgraph()

    progress_print(verbose_progress, "Generating command stream")
    # Generate command streams and serialise Npu-ops into tensors
    for sg in npu_subgraphs:
        high_level_command_stream_generator.generate_high_level_command_stream_for_schedule(
            nng, sg, arch, options.verbose_high_level_command_stream
        )
        lut.optimize_high_level_cmd_stream(sg, arch)
        high_level_command_to_npu_op.generate_register_command_stream_for_sg(
            nng, sg, arch, options.verbose_register_command_stream
        )
        scratch_tens, scratch_fast_tens, flash_tens = npu_serialisation.serialise_npu_subgraph_into_tensors(
            sg, arch, scratch_tens, scratch_fast_tens, flash_tens
        )

    # Create list of CPU subgraphs with same order as the list of all subgraphs
    cpu_subgraphs = [sg for sg in nng.subgraphs if sg.placement == PassPlacement.Cpu]
    for sg in cpu_subgraphs:
        npu_serialisation.rewrite_npu_call_ops(sg, arch)

    # Set Scratch and Fast_scratch Tensor size
    if scratch_tens is not None:
        scratch_tens.set_all_shapes([root_sg.memory_used_per_type.get(MemType.Scratch, 0)])
    if scratch_fast_tens is not None:
        scratch_fast_tens.set_all_shapes([root_sg.memory_used_per_type.get(MemType.Scratch_fast, 0)])

    progress_print(verbose_progress, "Allocating CPU constant tensors")
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
    progress_print(verbose_progress, "Calculating new performance for the network")
    npu_performance.calc_new_performance_for_network(
        nng, arch, network_type, options.verbose_performance, output_basename
    )
