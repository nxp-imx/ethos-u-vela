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
# Main entry point for the Vela compiler.
#
# Provides command line interface, options parsing, and network loading. Before calling the compiler driver.
import argparse
import ast
import os
import sys
import time

import flatbuffers

from . import architecture_features
from . import compiler_driver
from . import model_reader
from . import scheduler
from . import stats_writer
from . import tflite_writer
from ._version import __version__
from .api import API_VERSION
from .debug_database import DebugDatabase
from .errors import InputFileError
from .nn_graph import PassPlacement
from .nn_graph import TensorAllocator
from .scheduler import ParetoMetric
from .supported_operators import SupportedOperators
from .tensor import MemArea
from .tensor import Tensor
from .tflite.Model import Model
from .tflite_mapping import builtin_operator_map
from .tflite_mapping import builtin_type_name
from ethosu.vela.architecture_features import ArchitectureFeatures


def process(input_name, enable_debug_db, arch, model_reader_options, compiler_options, scheduler_options):
    if compiler_options.timing:
        start = time.time()

    os.makedirs(compiler_options.output_dir, exist_ok=True)
    output_basename = os.path.join(compiler_options.output_dir, os.path.splitext(os.path.basename(input_name))[0])
    DebugDatabase.show_warnings = enable_debug_db

    nng = model_reader.read_model(input_name, model_reader_options)

    if not nng:
        raise InputFileError(input_name, "Input file could not be read")

    if compiler_options.verbose_operators:
        nng.print_operators()

    if compiler_options.timing:
        stop = time.time()
        print("Model reading took %f s" % (stop - start))
        start = time.time()

    compiler_driver.compiler_driver(nng, arch, compiler_options, scheduler_options)

    passes_csv_file = "{0}_pass-breakdown_{1}.csv".format(output_basename, arch.system_config)
    stats_writer.write_pass_metrics_csv(nng, passes_csv_file)

    summary_csv_file = "{0}_summary_{1}.csv".format(output_basename, arch.system_config)
    stats_writer.write_summary_metrics_csv(nng, summary_csv_file, arch)

    stats_writer.print_performance_metrics(nng, show_cpu_operations=compiler_options.show_cpu_operations, arch=arch)

    output_filename = output_basename + "_vela.tflite"
    if input_name.endswith(".tflite"):
        tflite_writer.write_tflite(nng, output_filename)

    if enable_debug_db:
        file_offsets = calculate_operator_file_offsets(output_filename)
        for idx, offset in enumerate(sorted(file_offsets)):
            sg = find_subgraph_with_command_stream_order(nng, idx)
            if sg is not None:
                DebugDatabase.set_stream_offset(sg, offset)
        debug_filename = output_basename + "_debug.xml"
        DebugDatabase.write(debug_filename, input_name, output_filename)

    if compiler_options.timing:
        stop = time.time()
        print("Compiler driver took %f s" % (stop - start))

    return nng


def find_subgraph_with_command_stream_order(nng, idx):
    for sg in nng.subgraphs:
        if sg.generated_stream_id == idx:
            return sg
    return None


def calculate_operator_file_offsets(name: str):
    # Read the vela optimized tflite file
    with open(name, "rb") as f:
        buf = bytearray(f.read())
    # Calculate the file offsets for each custom operator
    file_offsets = []
    model = Model.GetRootAsModel(buf, 0)
    for idx in range(model.SubgraphsLength()):  # However only one subgraph is supported as of now
        sg = model.Subgraphs(idx)
        for idx in range(sg.OperatorsLength()):
            operator = sg.Operators(idx)
            if model.OperatorCodes(operator.OpcodeIndex()).CustomCode() is not None:
                tensor_idx = operator.Inputs(0)
                tensor = sg.Tensors(tensor_idx)
                buffer = model.Buffers(tensor.Buffer())
                offset = flatbuffers.number_types.UOffsetTFlags.py_type(buffer._tab.Offset(4))
                file_offsets.append(buffer._tab.Vector(offset))
    return file_offsets


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


def generate_supported_ops():
    lines = [
        "# Supported Ops",
        "",
        "This file was automatically generated by Vela using the `--supported-ops-report` parameter.  ",
        f"Vela version: `{__version__}`",
        "",
        "This file complies with",
        "[**Gitiles Markdown syntax**](https://github.com/google/gitiles/blob/master/Documentation/markdown.md)",
        "",
        "## Summary Table",
        "",
        "The table below contains TFLite operators that can be placed on the Ethos-U NPU.  ",
        "If the constraints are not met, then that operator will be scheduled on the CPU instead.  ",
        "For any other TFLite operator not listed, will be left untouched and scheduled on the CPU.  ",
        "Please check the supported operator list for your chosen runtime for further information.",
        "",
        "| Operator | Constraints |",
        "| --- | --- |",
    ]
    supported = SupportedOperators()
    op_constraint_links = []
    op_list = sorted(((op, builtin_type_name(op)) for op in builtin_operator_map), key=lambda x: x[1])
    for op, name in op_list:
        internal_op = builtin_operator_map[op][0]
        if internal_op in SupportedOperators.supported_operators:
            links = "[Generic](#generic-constraints)"
            if internal_op in supported.specific_constraints:
                links += f", [Specific](#{name.lower()}-constraints)"
                op_constraint_links.append((internal_op, name))
            lines.append(f"| {name} | {links} |")
    lines += [
        "",
        "## Generic Constraints",
        "",
        "This is a list of constraints that all NPU operators must satisfy in order to be scheduled on the NPU.",
        "",
    ]
    for constraint in supported.generic_constraints:
        # Markdown needs two spaces at the end of a line to render it as a separate line
        reason = constraint.__doc__.replace("\n", "  \n")
        lines.append(f"- {reason}")
    for op, name in op_constraint_links:
        lines += [
            "",
            f"## {name} Constraints",
            "",
            f"This is a list of constraints that the {name} operator must satisfy in order to be scheduled on the NPU.",
            "",
        ]
        for constraint in supported.specific_constraints[op]:
            # Markdown needs two spaces at the end of a line to render it as a separate line
            reason = constraint.__doc__.replace("\n", "  \n")
            lines.append(f"- {reason}")

    # Note. this will generate the file in the CWD
    filepath = os.path.join(os.getcwd(), "SUPPORTED_OPS.md")
    with open(filepath, "wt") as md:
        md.writelines(line + "\n" for line in lines)
        print(f"Report file: {filepath}")


def main(args=None):
    if args is None:
        args = sys.argv[1:]

    parser = argparse.ArgumentParser(prog="vela", description="Neural network model compiler for Arm Ethos-U NPUs")
    parser.add_argument("--version", action="version", version=__version__)
    parser.add_argument(
        "--api-version", action="version", version=API_VERSION, help="Displays the version of the external API."
    )
    parser.add_argument(
        "--supported-ops-report",
        action="store_true",
        help="Generate the SUPPORTED_OPS.md file in the current working directory and exit",
    )

    # set network nargs to be optional to allow the support-ops-report CLI option to be used standalone
    parser.add_argument(
        "network",
        metavar="NETWORK",
        type=str,
        default=None,
        nargs="?",
        help="Filename of the input TensorFlow Lite for Microcontrollers network",
    )
    parser.add_argument(
        "--output-dir", type=str, default="output", help="Output directory to write files to (default: %(default)s)"
    )
    parser.add_argument(
        "--enable-debug-db",
        action="store_true",
        default=None,
        help="Enables the calculation and writing of a network debug database to output directory",
    )
    parser.add_argument(
        "--config", type=str, action="append", help="Vela configuration file(s) in Python ConfigParser .ini file format"
    )
    parser.add_argument("--verbose-all", action="store_true", help="Enable all verbose options")
    parser.add_argument("--verbose-config", action="store_true", help="Verbose system configuration and memory mode")
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
        "--show-cpu-operations", action="store_true", help="Show the operations that fall back to the CPU"
    )
    parser.add_argument(
        "--cache-bias-scale-tensor",
        type=ast.literal_eval,
        default=True,
        choices=[True, False],
        help="Controls the caching of the bias & scale tensors in SRAM (default: %(default)s)",
    )
    parser.add_argument(
        "--cascading",
        type=ast.literal_eval,
        default=True,
        choices=[True, False],
        help="Controls the packing of multiple passes into a cascade (default: %(default)s)",
    )
    parser.add_argument("--force-block-config", type=str, default="", help="Force a specific block configuration WxHxC")
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
        default=architecture_features.ArchitectureFeatures.DEFAULT_CONFIG,
        help="System configuration to select from the Vela configuration file (default: %(default)s)",
    )
    parser.add_argument(
        "--memory-mode",
        type=str,
        default=architecture_features.ArchitectureFeatures.DEFAULT_CONFIG,
        help="Memory mode to select from the Vela configuration file (default: %(default)s)",
    )
    parser.add_argument(
        "--tensor-allocator",
        default=TensorAllocator.HillClimb,
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
            "Set the maximum value that can be used for the block dependency between npu kernel operations"
            " (default: %(default)s)"
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
    parser.add_argument(
        "--cpu-tensor-alignment",
        type=int,
        default=Tensor.AllocationQuantum,
        help=(
            "Controls the allocation byte alignment of cpu tensors including Ethos-U Custom operator inputs and outputs"
            " (default: %(default)s)"
        ),
    )
    args = parser.parse_args(args=args)

    # Generate the supported ops report and exit
    if args.supported_ops_report:
        generate_supported_ops()
        return 0

    if args.network is None:
        parser.error("the following argument is required: NETWORK")

    # check all config files exist because they will be read as a group
    if args.config is not None:
        for filename in args.config:
            if not os.access(filename, os.R_OK):
                raise InputFileError(filename, "File not found or is not readable")

    sys.setrecursionlimit(args.recursion_limit)

    if args.force_block_config:
        force_block_config = architecture_features.Block.from_string(args.force_block_config)
    else:
        force_block_config = None

    if args.cpu_tensor_alignment < 16 or args.cpu_tensor_alignment & (args.cpu_tensor_alignment - 1) != 0:
        parser.error(
            "Invalid argument to --cpu-tensor-alignment = {} (must be greater than or equal to 16 and a power of 2)"
            "".format(args.cpu_tensor_alignment)
        )

    if args.system_config == ArchitectureFeatures.DEFAULT_CONFIG:
        print(f"Warning: Using {ArchitectureFeatures.DEFAULT_CONFIG} values for system configuration")

    if args.memory_mode == ArchitectureFeatures.DEFAULT_CONFIG:
        print(f"Warning: Using {ArchitectureFeatures.DEFAULT_CONFIG} values for memory mode")

    if args.verbose_all:
        for v in vars(args):
            if v.startswith("verbose") and v != "verbose_all":
                setattr(args, v, True)

    arch = architecture_features.ArchitectureFeatures(
        vela_config_files=args.config,
        system_config=args.system_config,
        memory_mode=args.memory_mode,
        accelerator_config=args.accelerator_config,
        override_block_config=force_block_config,
        block_config_limit=args.block_config_limit,
        max_blockdep=args.max_block_dependency,
        weight_estimation_scaling=args.weight_estimation_scaling,
        verbose_config=args.verbose_config,
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
        show_cpu_operations=args.show_cpu_operations,
        tensor_allocator=args.tensor_allocator,
        timing=args.timing,
        output_dir=args.output_dir,
        cpu_tensor_alignment=args.cpu_tensor_alignment,
    )

    scheduler_options = scheduler.SchedulerOptions(
        use_cascading=args.cascading,
        verbose_schedule=args.verbose_schedule,
        verbose_pareto_frontier_schedules=args.verbose_pareto_frontier_schedules,
        use_ifm_streaming=args.ifm_streaming,
        pareto_metric=args.pareto_metric,
        use_nhcwb16_between_cascaded_passes=args.nhcwb16_between_cascaded_passes,
        cache_bias_scale_tensor=args.cache_bias_scale_tensor,
    )

    model_reader_options = model_reader.ModelReaderOptions()

    nng = process(args.network, args.enable_debug_db, arch, model_reader_options, compiler_options, scheduler_options)

    if args.show_subgraph_io_summary:
        print_subgraph_io_summary(nng)

    return 0
