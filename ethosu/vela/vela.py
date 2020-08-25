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
# Main entry point for the Vela compiler.
#
# Provides command line interface, options parsing, and network loading. Before calling the compiler driver.
import argparse
import ast
import configparser
import os.path
import sys
import time

from . import architecture_features
from . import compiler_driver
from . import model_reader
from . import scheduler
from . import stats_writer
from . import tflite_writer
from ._version import __version__
from .errors import InputFileError
from .nn_graph import PassPlacement
from .nn_graph import TensorAllocator
from .scheduler import ParetoMetric
from .tensor import MemArea


def process(fname, arch, model_reader_options, compiler_options, scheduler_options):
    if compiler_options.timing:
        start = time.time()

    nng = model_reader.read_model(fname, model_reader_options)

    if not nng:
        raise InputFileError(fname, "input file could not be read")

    if compiler_options.verbose_operators:
        nng.print_operators()

    if compiler_options.timing:
        stop = time.time()
        print("Model reading took %f s" % (stop - start))
        start = time.time()

    compiler_driver.compiler_driver(nng, arch, compiler_options, scheduler_options)

    passes_csv_file = "%s/%s_pass-breakdown_%s.csv" % (compiler_options.output_dir, nng.name, arch.system_config)
    stats_writer.write_pass_metrics_csv(nng, passes_csv_file)

    summary_csv_file = "%s/%s_summary_%s.csv" % (compiler_options.output_dir, nng.name, arch.system_config)
    stats_writer.write_summary_metrics_csv(nng, summary_csv_file, arch)

    stats_writer.print_performance_metrics(nng, show_cpu_operations=compiler_options.show_cpu_operations, arch=arch)

    if fname.endswith(".tflite"):
        tflite_writer.write_tflite(nng, "%s/%s_vela.tflite" % (compiler_options.output_dir, nng.name))

    if compiler_options.timing:
        stop = time.time()
        print("Compiler driver took %f s" % (stop - start))

    return nng


def print_subgraph_io_summary(nng):
    """Print a summary of all the input and output tensor sizes for all subgraphs.
    Also displays the total tensor size and the memory used area for sram.
    """

    print("Subgraph IO Summary")
    print("-------------------")
    print("NNG: {0}".format(nng.name))
    max_sg_size = 0
    for sg in reversed(nng.subgraphs):
        print("   Subgraph: {0} = {1}".format(sg.name, sg.placement))
        sg_size = 0

        if sg.placement == PassPlacement.Npu:
            for tens in sg.input_tensors + [sg.scratch_tensor] + sg.output_tensors:
                if tens in sg.input_tensors:
                    tens_dir = "In"
                elif tens in sg.output_tensors:
                    tens_dir = "Out"
                else:
                    tens_dir = "In/Out"

                size = tens.elements() * tens.element_size() / 1024.0
                sg_size = sg_size + size
                print("         Tensor [{0}]: {1} = {2} KiB".format(tens_dir, tens.name, size))

        print("      Total Size = {0} KiB".format(sg_size))
        print("      SRAM Memory Used = {0} KiB".format(sg.memory_used.get(MemArea.Sram, 0) / 1024.0))
        max_sg_size = max(sg_size, max_sg_size)

    print("   Maximum Subgraph Size = {0} KiB".format(max_sg_size))


def main(args=None):
    if args is None:
        args = sys.argv[1:]

    parser = argparse.ArgumentParser(prog="vela", description="Neural network model compiler for Ethos-U55")

    parser.add_argument(
        "network", metavar="NETWORK", type=str, default=None, nargs=None, help="Filename of network to process"
    )

    parser.add_argument("--version", action="version", version=__version__)
    parser.add_argument(
        "--output-dir", type=str, default="output", help="Output directory to write files to (default: %(default)s)"
    )
    parser.add_argument("--config", type=str, help="Location of vela configuration file")

    parser.add_argument("--verbose-graph", action="store_true", help="Verbose graph rewriter")
    parser.add_argument("--verbose-quantization", action="store_true", help="Verbose quantization")
    parser.add_argument("--verbose-packing", action="store_true", help="Verbose pass packing")
    parser.add_argument("--verbose-tensor-purpose", action="store_true", help="Verbose tensor purpose")
    parser.add_argument("--verbose-tensor-format", action="store_true", help="Verbose tensor format")
    parser.add_argument("--verbose-schedule", action="store_true", help="Verbose schedule")
    parser.add_argument(
        "--verbose-pareto-frontier-schedules",
        action="store_true",
        help="Show all schedules along the pareto frontier of optimisation criteria",
    )
    parser.add_argument("--verbose-allocation", action="store_true", help="Verbose tensor allocation")
    parser.add_argument(
        "--verbose-high-level-command-stream", action="store_true", help="Verbose high level command stream"
    )
    parser.add_argument(
        "--verbose-register-command-stream", action="store_true", help="Verbose register command stream"
    )
    parser.add_argument("--verbose-operators", action="store_true", help="Verbose operator list")

    parser.add_argument(
        "--show-minimum-possible-allocation", action="store_true", help="Show the minimum possible allocation"
    )
    parser.add_argument(
        "--show-cpu-operations", action="store_true", help="Show the operations that fall back to the CPU"
    )
    parser.add_argument(
        "--cascading",
        type=ast.literal_eval,
        default=True,
        choices=[True, False],
        help="Controls the packing of multiple passes into a cascade (default: %(default)s)",
    )
    parser.add_argument(
        "--ifm-ofm-overlap",
        type=ast.literal_eval,
        default=True,
        choices=[True, False],
        help="Controls the overlapping of IFM and OFM buffers (default: %(default)s)",
    )
    parser.add_argument("--force-block-config", type=str, default="", help="Force a specific block configuration HxWxC")
    parser.add_argument("--timing", action="store_true", help="Time the compiler doing operations")
    parser.add_argument(
        "--accelerator-config",
        type=str,
        default="ethos-u55-256",
        choices=list(architecture_features.Accelerator.member_list()),
        help="Accelerator configuration to use (default: %(default)s)",
    )
    parser.add_argument(
        "--system-config",
        type=str,
        default="internal-default",
        help="System configuration to use (default: %(default)s)",
    )
    parser.add_argument(
        "--permanent-storage",
        default=MemArea.OffChipFlash,
        type=lambda s: MemArea[s],
        choices=list(MemArea)[3:5],
        help=(
            "Memory area for permanent storage, only valid for Ethos-U55. "
            "To store the weights and other constant data in SRAM, select 'OnChipFlash'. (default: %(default)s)"
        ),
    )
    parser.add_argument(
        "--tensor-allocator",
        default=TensorAllocator.Greedy,
        type=lambda s: TensorAllocator[s],
        choices=list(TensorAllocator),
        help="Tensor Allocator algorithm (default: %(default)s)",
    )
    parser.add_argument(
        "--show-subgraph-io-summary",
        action="store_true",
        help="Shows a summary of all the subgraphs and their inputs and outputs",
    )
    parser.add_argument(
        "--ifm-streaming",
        type=ast.literal_eval,
        default=True,
        choices=[True, False],
        help="Controls scheduler IFM streaming search (default: %(default)s)",
    )
    parser.add_argument(
        "--block-config-limit",
        type=int,
        default=16,
        help="Limit block config search space, use zero for unlimited (default: %(default)s)",
    )
    parser.add_argument(
        "--global-memory-clock-scale",
        type=float,
        default=1.0,
        help=(
            "Performs an additional scaling of the individual memory clock scales specified by the system config "
            "(default: %(default)s)"
        ),
    )
    parser.add_argument(
        "--pareto-metric",
        default=ParetoMetric.BwCycMem,
        type=lambda s: ParetoMetric[s],
        choices=list(ParetoMetric),
        help="Controls the calculation of the pareto metric (default: %(default)s)",
    )
    parser.add_argument(
        "--recursion-limit",
        type=int,
        default=10000,
        help="Set the recursion depth limit, may result in RecursionError if too low (default: %(default)s)",
    )
    parser.add_argument(
        "--max-block-dependency",
        type=int,
        default=architecture_features.ArchitectureFeatures.MAX_BLOCKDEP,
        choices=range(0, architecture_features.ArchitectureFeatures.MAX_BLOCKDEP + 1),
        help=(
            "Set the maximum value that can be used for the block dependency between npu kernel operations "
            "(default: %(default)s)"
        ),
    )
    parser.add_argument(
        "--nhcwb16-between-cascaded-passes",
        type=ast.literal_eval,
        default=True,
        choices=[True, False],
        help="Control if NHCWB16 or NHWC should be used in between cascaded passes (default: %(default)s)",
    )
    parser.add_argument(
        "--weight-estimation-scaling",
        type=float,
        default=1.0,
        help=("Performs an additional scaling of weight compression scale estimate (default: %(default)s)"),
    )

    args = parser.parse_args(args=args)

    # Read configuration file
    config_file = args.config
    config = None
    if config_file is not None:
        with open(config_file) as f:
            config = configparser.ConfigParser()
            config.read_file(f)

    if args.network is None:
        parser.error("the following argument is required: NETWORK")

    sys.setrecursionlimit(args.recursion_limit)

    if args.force_block_config:
        force_block_config = architecture_features.Block.from_string(args.force_block_config)
    else:
        force_block_config = None

    arch = architecture_features.ArchitectureFeatures(
        vela_config=config,
        system_config=args.system_config,
        accelerator_config=args.accelerator_config,
        permanent_storage=args.permanent_storage,
        override_block_config=force_block_config,
        block_config_limit=args.block_config_limit,
        global_memory_clock_scale=args.global_memory_clock_scale,
        max_blockdep=args.max_block_dependency,
        weight_estimation_scaling=args.weight_estimation_scaling,
    )

    compiler_options = compiler_driver.CompilerOptions(
        verbose_graph=args.verbose_graph,
        verbose_quantization=args.verbose_quantization,
        verbose_packing=args.verbose_packing,
        verbose_tensor_purpose=args.verbose_tensor_purpose,
        verbose_tensor_format=args.verbose_tensor_format,
        verbose_allocation=args.verbose_allocation,
        verbose_high_level_command_stream=args.verbose_high_level_command_stream,
        verbose_register_command_stream=args.verbose_register_command_stream,
        verbose_operators=args.verbose_operators,
        show_minimum_possible_allocation=args.show_minimum_possible_allocation,
        show_cpu_operations=args.show_cpu_operations,
        tensor_allocator=args.tensor_allocator,
        timing=args.timing,
        output_dir=args.output_dir,
    )

    scheduler_options = scheduler.SchedulerOptions(
        use_cascading=args.cascading,
        use_ifm_ofm_overlap=args.ifm_ofm_overlap,
        verbose_schedule=args.verbose_schedule,
        verbose_pareto_frontier_schedules=args.verbose_pareto_frontier_schedules,
        use_ifm_streaming=args.ifm_streaming,
        pareto_metric=args.pareto_metric,
        use_nhcwb16_between_cascaded_passes=args.nhcwb16_between_cascaded_passes,
    )

    model_reader_options = model_reader.ModelReaderOptions()

    os.makedirs(args.output_dir, exist_ok=True)

    nng = process(args.network, arch, model_reader_options, compiler_options, scheduler_options)

    if args.show_subgraph_io_summary:
        print_subgraph_io_summary(nng)

    return 0
