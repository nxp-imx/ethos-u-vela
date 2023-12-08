# SPDX-FileCopyrightText: Copyright 2020-2023 Arm Limited and/or its affiliates <open-source-office@arm.com>
# Copyright 2022-2023 NXP
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
# Main entry point for the Vela compiler.
#
# Provides command line interface, options parsing, and network loading. Before calling the compiler driver.
import argparse
import glob
import os
import sys
import time

import flatbuffers

from . import architecture_features
from . import compiler_driver
from . import model_reader
from . import rawdata_writer
from . import scheduler
from . import stats_writer
from . import tflite_writer
from ._version import __version__
from .api import API_VERSION
from .debug_database import DebugDatabase
from .errors import InputFileError
from .errors import VelaError
from .hillclimb_allocation import HillClimbAllocator
from .nn_graph import NetworkType
from .nn_graph import TensorAllocator
from .tensor import MemArea
from .tensor import Tensor
from .tensor import TensorAddressMap
from .tflite.Model import Model
from .tflite_mapping import builtin_operator_map
from .tflite_mapping import builtin_operator_name_map
from .tflite_mapping import optype_to_builtintype
from .tflite_model_semantic import TFLiteSemantic
from .tflite_supported_operators import TFLiteSupportedOperators
from .tosa_model_semantic import TosaSemantic
from .tosa_supported_operators import TosaSupportedOperators
from ethosu.vela.architecture_features import ArchitectureFeatures

CONFIG_FILES_PATH = os.path.normpath(os.path.join(__file__, "..", "..", "config_files"))


def process(input_name, enable_debug_db, arch, model_reader_options, compiler_options, scheduler_options, subgraph_output):
    if compiler_options.timing:
        start = time.time()

    os.makedirs(compiler_options.output_dir, exist_ok=True)
    output_basename = os.path.join(compiler_options.output_dir, os.path.splitext(os.path.basename(input_name))[0])
    DebugDatabase.show_warnings = enable_debug_db

    nng, network_type = model_reader.read_model(input_name, model_reader_options)

    if not nng:
        raise InputFileError(input_name, "Input file could not be read")

    if compiler_options.verbose_operators:
        nng.print_operators()

    if compiler_options.timing:
        stop = time.time()
        print("Model reading took %f s" % (stop - start))
        start = time.time()

    compiler_driver.compiler_driver(nng, arch, compiler_options, scheduler_options, network_type, output_basename, subgraph_output)

    summary_csv_file = "{0}_summary_{1}.csv".format(output_basename, arch.system_config)
    stats_writer.write_summary_metrics_csv(nng, summary_csv_file, arch)

    stats_writer.print_performance_metrics(
        nng,
        show_cpu_operations=compiler_options.show_cpu_operations,
        verbose_weights=compiler_options.verbose_weights,
        arch=arch,
    )

    output_tfl_filename = output_basename + "_vela.tflite"
    if input_name.endswith(".tflite"):
        tflite_writer.write_tflite(nng, output_tfl_filename)
    if input_name.endswith(".tosa"):
        rawdata_writer.write_rawdata_output(nng, arch, output_basename)

    if enable_debug_db:
        file_offsets = calculate_operator_file_offsets(output_tfl_filename)
        for idx, offset in enumerate(sorted(file_offsets)):
            sg = find_subgraph_with_command_stream_order(nng, idx)
            if sg is not None:
                DebugDatabase.set_stream_offset(sg, offset)
        debug_filename = output_basename + "_debug.xml"
        DebugDatabase.write(debug_filename, input_name, output_tfl_filename)

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
    print(f"NNG: {nng.name}")
    max_sg_size = 0
    for sg in reversed(nng.subgraphs):
        print(f"  NNG Subgraph: {sg.name} = {sg.placement}")
        sg_size = 0

        if hasattr(sg, "scratch_tensor") and sg.scratch_tensor is not None:
            sg_tensors = sg.input_tensors + [sg.scratch_tensor] + sg.output_tensors
        else:
            sg_tensors = sg.input_tensors + sg.output_tensors

        for tens in sg_tensors:
            if tens in sg.input_tensors:
                tens_dir = "In"
            elif tens in sg.output_tensors:
                tens_dir = "Out"
            else:
                tens_dir = "In/Out"

            size = tens.elements() * tens.element_size() / 1024.0
            sg_size = sg_size + size
            print(f"         Tensor [{tens_dir}]: {tens.name} = {size} KiB")

        print(f"      Total Size = {sg_size} KiB")
        print(f"      SRAM Memory Used = {sg.memory_used.get(MemArea.Sram, 0) / 1024.0} KiB")
        max_sg_size = max(sg_size, max_sg_size)

    print(f"   Maximum NNG Subgraph Size = {max_sg_size} KiB")


def generate_license():
    lines = [
        "<!--",
        "SPDX-FileCopyrightText: Copyright 2020-2023 Arm Limited and/or its affiliates <open-source-office@arm.com>",
        "",
        "SPDX-License-Identifier: Apache-2.0",
        "",
        "Licensed under the Apache License, Version 2.0 (the License); you may",
        "not use this file except in compliance with the License.",
        "You may obtain a copy of the License at",
        "",
        "www.apache.org/licenses/LICENSE-2.0",
        "",
        "Unless required by applicable law or agreed to in writing, software",
        "distributed under the License is distributed on an AS IS BASIS, WITHOUT",
        "WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.",
        "See the License for the specific language governing permissions and",
        "limitations under the License.",
        "-->",
        "",
    ]
    return lines


def generate_supported_ops():
    # Exclude network type from generation by adding value to exclude list.
    # To easily exclude NetworkType from generated documentation.
    exclude_generation_network_type_value = [NetworkType.TOSA.value]

    def _exclude_list_names(constraint, exclude_list):
        constraints_excluded_names = [
            optype_to_builtintype(op) for op, exclude_constraint in exclude_list if constraint in exclude_constraint
        ]
        return f" - [{', '.join(sorted(constraints_excluded_names))}]" if constraints_excluded_names else ""

    lines = generate_license()
    lines += [
        "# Supported Ops",
        "",
        "This file was automatically generated by Vela using the `--supported-ops-report` parameter.  ",
        f"Vela version: `{__version__}`",
        "",
        "This file complies with",
        "[**Gitiles Markdown syntax**](https://gerrit.googlesource.com/gitiles/+/HEAD/Documentation/markdown.md)",
        "",
        "Summary table of constraints for:",
    ]

    for network_type in NetworkType:
        if network_type.value in exclude_generation_network_type_value:
            continue

        lines += [
            f"- [{network_type.name}](#{network_type.name.lower()}-summary-table)",
        ]

    for network_type in NetworkType:
        if network_type.value in exclude_generation_network_type_value:
            continue

        lines += [
            "",
            f"## {network_type.name} Summary Table",
            "",
        ]
        if network_type == NetworkType.TFLite:
            lines += [
                "The table below contains TFLite operators that can be placed on the Ethos-U NPU.  ",
                "If the constraints are not met, then that operator will be scheduled on the CPU instead.  ",
                "For any other TFLite operator not listed, will be left untouched and scheduled on the CPU.  ",
                "Please check the supported operator list for your chosen runtime for further information.",
                "",
                "| Operator | TFLite Constraints |",
                "| --- | --- |",
            ]
            semantic_checker = TFLiteSemantic()
            supported = TFLiteSupportedOperators()
        elif network_type == NetworkType.TOSA:
            lines += [
                "The table below contains TOSA operators that can be placed on the Ethos-U NPU.  ",
                "Note: There is limited support for compiling a TOSA neural network (EXPERIMENTAL).  ",
                "The related constraints have not yet been populated in the list.",
                "",
                "| Operator | TOSA Constraints |",
                "| --- | --- |",
            ]
            semantic_checker = TosaSemantic()
            supported = TosaSupportedOperators()
        else:
            raise ValueError

        op_constraint_links = []
        op_list = sorted(((op, builtin_operator_name_map[op]) for op in builtin_operator_map), key=lambda x: x[1])
        for op, name in op_list:
            internal_op = builtin_operator_map[op][0]
            if internal_op in TFLiteSupportedOperators.supported_operators:
                links = f"[Generic](#{network_type.name.lower()}-generic-constraints)"
                if (
                    internal_op in supported.specific_constraints
                    or internal_op in semantic_checker.specific_constraints
                ):
                    links += f", [Specific](#{network_type.name.lower()}-{name.lower()}-constraints)"
                    op_constraint_links.append((internal_op, name))
                lines.append(f"| {name} | {links} |")
        lines += [
            "",
            f"### {network_type.name} Generic Constraints",
            "",
            "This is a list of constraints most NPU operators must satisfy in order to be scheduled on the NPU.",
            "(Operators excluded from certain constraints are shown in brackets [ ] )\n" "",
        ]
        for constraint in semantic_checker.generic_constraints:
            # Markdown needs two spaces at the end of a line to render it as a separate line
            reason = constraint.__doc__.replace("\n", "  \n")
            exclude_list = TFLiteSemantic.get_generic_constraint_exclude_list().items()
            lines.append(f"- {reason}{_exclude_list_names(constraint, exclude_list)}")
        for constraint in supported.generic_constraints:
            # Markdown needs two spaces at the end of a line to render it as a separate line
            reason = constraint.__doc__.replace("\n", "  \n")
            exclude_list = supported.generic_constraints_exceptions.items()
            lines.append(f"- {reason}{_exclude_list_names(constraint, exclude_list)}")
        for op, name in op_constraint_links:
            lines += [
                "",
                f"### {network_type.name} {name} Constraints",
                "",
                f"This is a list of constraints that the {name} operator must satisfy in order to be scheduled on the"
                " NPU.",
                "",
            ]
            for constraint in semantic_checker.specific_constraints[op]:
                # Markdown needs two spaces at the end of a line to render it as a separate line
                reason = constraint.__doc__.replace("\n", "  \n")
                lines.append(f"- {reason}")
            for constraint in supported.specific_constraints[op]:
                # Markdown needs two spaces at the end of a line to render it as a separate line
                reason = constraint.__doc__.replace("\n", "  \n")
                lines.append(f"- {reason}")

    # Note. this will generate the file in the CWD
    filepath = os.path.join(os.getcwd(), "SUPPORTED_OPS.md")
    with open(filepath, "wt") as md:
        md.writelines(line + "\n" for line in lines)
        print(f"Report file: {filepath}")


def list_config_files():
    print("Available config files:")
    path_length = len(CONFIG_FILES_PATH + os.path.sep)
    for config in glob.glob(os.path.join(CONFIG_FILES_PATH, "*", "*.ini")):
        print(config[path_length:])


class Imx93ArchitectureFeatures(architecture_features.ArchitectureFeatures):

    def _set_default_sys_config(self):
        # Default Ethos-U65 system configuration
        # Ethos-U65 High-End: SRAM (16 GB/s) and DRAM (3.75 GB/s)
        from .tensor import BandwidthDirection
        self.core_clock = 1e9
        self.axi0_port = MemArea.Sram
        self.axi1_port = MemArea.Dram
        self.memory_clock_scales[MemArea.Sram] = 1.0
        self.memory_clock_scales[MemArea.Dram] = 0.234375
        self.memory_burst_length[MemArea.Sram] = 32
        self.memory_burst_length[MemArea.Dram] = 128
        self.memory_latency[MemArea.Sram][BandwidthDirection.Read] = 32
        self.memory_latency[MemArea.Sram][BandwidthDirection.Write] = 32
        self.memory_latency[MemArea.Dram][BandwidthDirection.Read] = 500
        self.memory_latency[MemArea.Dram][BandwidthDirection.Write] = 250

def convert(input_model_name):
    sys.setrecursionlimit(4000)

    if not os.path.exists(input_model_name):
        raise InputFileError(input_model_name, "No such file")

    arch = Imx93ArchitectureFeatures(
        vela_config_files=None,
        system_config=ArchitectureFeatures.DEFAULT_CONFIG,
        memory_mode=ArchitectureFeatures.DEFAULT_CONFIG,
        accelerator_config='ethos-u65-256',
        max_blockdep=ArchitectureFeatures.MAX_BLOCKDEP,
        verbose_config=False,
        arena_cache_size=384 * 1024,
    )

    compiler_options = compiler_driver.CompilerOptions(
        tensor_allocator = TensorAllocator.HillClimb,
        output_dir="output",
    )

    scheduler_options = scheduler.SchedulerOptions(
        optimization_strategy=scheduler.OptimizationStrategy.Performance,
        sram_target=arch.arena_cache_size,
        verbose_schedule=False,
    )

    model_reader_options = model_reader.ModelReaderOptions()

    os.makedirs(compiler_options.output_dir, exist_ok=True)
    output_basename = os.path.join(compiler_options.output_dir,
                                   os.path.splitext(os.path.basename(input_model_name))[0])

    nng, network_type = model_reader.read_model(input_model_name, model_reader_options)
    if not nng:
        raise InputFileError(input_model_name, "Input file could not be read")

    compiler_driver.compiler_driver(nng, arch, compiler_options, scheduler_options, network_type, output_basename)

    output_tfl_filename = output_basename + "_vela.tflite"
    tflite_writer.write_tflite(nng, output_tfl_filename)
    DebugDatabase.clean_db()

    return output_tfl_filename

def convert_bytes(data):
    sys.setrecursionlimit(4000)

    arch = Imx93ArchitectureFeatures(
        vela_config_files=None,
        system_config=ArchitectureFeatures.DEFAULT_CONFIG,
        memory_mode=ArchitectureFeatures.DEFAULT_CONFIG,
        accelerator_config='ethos-u65-256',
        max_blockdep=ArchitectureFeatures.MAX_BLOCKDEP,
        verbose_config=False,
        arena_cache_size=384 * 1024,
    )

    compiler_options = compiler_driver.CompilerOptions(
        tensor_allocator = TensorAllocator.HillClimb,
        output_dir="output",
    )

    scheduler_options = scheduler.SchedulerOptions(
        optimization_strategy=scheduler.OptimizationStrategy.Performance,
        sram_target=arch.arena_cache_size,
        verbose_schedule=False,
    )

    model_reader_options = model_reader.ModelReaderOptions()

    nng, network_type = model_reader.read_tflite_model(data, model_reader_options)
    if not nng:
        raise InputFileError(input_model_name, "Invalid data")

    compiler_driver.compiler_driver(nng, arch, compiler_options, scheduler_options, network_type, "data_model")
    buf = tflite_writer.write_tflite_buffer(nng)
    DebugDatabase.clean_db()
    TensorAddressMap.clear_address_map()

    return memoryview(buf)

def main(args=None):
    try:
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

        parser.add_argument(
            "--list-config-files",
            action="store_true",
            help=(
                "Display all available configurations in the `config_files` folder and exit. To select config file, "
                "use the --config argument with one of the listed config files (For example: --config Arm/vela.ini )"
            ),
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
            "--config",
            type=str,
            action="append",
            help="Vela configuration file(s) in Python ConfigParser .ini file format",
        )
        parser.add_argument("--verbose-all", action="store_true", help="Enable all verbose options")
        parser.add_argument(
            "--verbose-config", action="store_true", help="Verbose system configuration and memory mode"
        )
        parser.add_argument("--verbose-graph", action="store_true", help="Verbose graph rewriter")
        parser.add_argument("--verbose-quantization", action="store_true", help="Verbose quantization")
        parser.add_argument("--verbose-packing", action="store_true", help="Verbose pass packing")
        parser.add_argument("--verbose-tensor-purpose", action="store_true", help="Verbose tensor purpose")
        parser.add_argument("--verbose-tensor-format", action="store_true", help="Verbose tensor format")
        parser.add_argument("--verbose-schedule", action="store_true", help="Verbose schedule")
        parser.add_argument("--verbose-allocation", action="store_true", help="Verbose tensor allocation")
        parser.add_argument(
            "--verbose-high-level-command-stream", action="store_true", help="Verbose high level command stream"
        )
        parser.add_argument(
            "--verbose-register-command-stream", action="store_true", help="Verbose register command stream"
        )
        parser.add_argument("--verbose-operators", action="store_true", help="Verbose operator list")
        parser.add_argument("--verbose-weights", action="store_true", help="Verbose weights information")
        parser.add_argument("--verbose-performance", action="store_true", help="Verbose performance information")
        parser.add_argument("--verbose-progress", action="store_true", help="Verbose progress information")
        parser.add_argument(
            "--show-cpu-operations", action="store_true", help="Show the operations that fall back to the CPU"
        )
        parser.add_argument("--timing", action="store_true", help="Time the compiler doing operations")
        parser.add_argument("--subgraph-output", action="store_true", help="Generate files to reconstruct converted subgraph. Used by Model Tool to display it.")
        parser.add_argument(
            "--force-symmetric-int-weights",
            action="store_true",
            help="Forces all zero points to 0 for signed integer weights",
        )
        parser.add_argument(
            "--accelerator-config",
            type=str,
            default="ethos-u65-256",
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
            "--optimise",
            type=lambda s: scheduler.OptimizationStrategy[s],
            default=scheduler.OptimizationStrategy.Performance,
            choices=list(scheduler.OptimizationStrategy),
            help=(
                "Set the optimisation strategy. The Size strategy results in minimal SRAM usage (does not use"
                " arena-cache-size). The Performance strategy results in maximal performance (uses the arena-cache-size"
                " if specified) (default: %(default)s)"
            ),
        )
        parser.add_argument(
            "--arena-cache-size",
            type=int,
            default=384 * 1024,
            help=(
                "Set the size of the arena cache memory area, in bytes. If specified, this option overrides the memory"
                " mode attribute with the same name in a Vela configuration file (default: %(default)s)"
            ),
        )
        parser.add_argument(
            "--cpu-tensor-alignment",
            type=int,
            default=Tensor.AllocationQuantum,
            help=(
                "Controls the allocation byte alignment of cpu tensors including Ethos-U Custom"
                " operator inputs and outputs (default: %(default)s Bytes)"
            ),
        )
        parser.add_argument(
            "--recursion-limit",
            type=int,
            default=4000,
            help="Set the recursion depth limit, may result in RecursionError if too low (default: %(default)s)",
        )
        parser.add_argument(
            "--hillclimb-max-iterations",
            type=int,
            default=HillClimbAllocator.MAX_ITERATIONS,
            help=(
                "Set the maximum number of iterations the Hill Climb tensor allocator will run (default: %(default)s)"
            ),
        )
        args = parser.parse_args(args=args)

        # Generate the supported ops report and exit
        if args.supported_ops_report:
            generate_supported_ops()
            return 0

        if args.list_config_files:
            list_config_files()
            return 0

        if args.network is None:
            parser.error("the following argument is required: NETWORK")

        def _parse_config(config):
            # Make sure the correct separator is used depending on OS
            config = os.path.normpath(config)

            if not config.endswith(".ini"):
                raise InputFileError(config, "Configuration files must use the .ini extension")

            if (
                len(config.split(os.path.sep)) == 2
                and not config.startswith(os.path.sep)
                and not config.startswith(".")
                and not config.startswith("~")
            ):
                config_path = os.path.join(CONFIG_FILES_PATH, config)
            else:
                # Check if the configuration file is correctly placed inside the config_files directory
                if os.access(os.path.join(CONFIG_FILES_PATH, *config.split(os.path.sep)[-2:]), os.R_OK):
                    rel_path = os.path.join(*config.split(os.path.sep)[-2:])
                    print(
                        f"Warning: Consider accessing the configuration by --config {rel_path} since it is located "
                        "inside the config_files directory."
                    )
                config_path = config

            if not os.access(config_path, os.R_OK):
                raise InputFileError(
                    config_path,
                    "File not found or is not readable. The configuration file is either not located in a folder "
                    "directly under the `config_files` directory or its path has not been provided correctly.",
                )

            return config_path

        # check all config files exist because they will be read as a group
        config_files = [_parse_config(cfg) for cfg in args.config] if args.config else None

        if args.cpu_tensor_alignment < 16 or args.cpu_tensor_alignment & (args.cpu_tensor_alignment - 1) != 0:
            parser.error(
                "Invalid argument to --cpu-tensor-alignment = {} (must be greater than or equal to 16 and a power of 2)"
                "".format(args.cpu_tensor_alignment)
            )
        if args.verbose_all:
            for v in vars(args):
                if v.startswith("verbose") and v != "verbose_all":
                    setattr(args, v, True)

        sys.setrecursionlimit(args.recursion_limit)

        # Use Imx93 Architecture by default(args.config is None)
        if args.config is None and args.system_config == args.memory_mode == ArchitectureFeatures.DEFAULT_CONFIG:
             arch = Imx93ArchitectureFeatures(
                vela_config_files=args.config,
                system_config=ArchitectureFeatures.DEFAULT_CONFIG,
                memory_mode=ArchitectureFeatures.DEFAULT_CONFIG,
                accelerator_config=args.accelerator_config,
                max_blockdep=args.max_block_dependency,
                verbose_config=args.verbose_config,
                arena_cache_size=args.arena_cache_size,
            )
        else:
            if args.system_config == ArchitectureFeatures.DEFAULT_CONFIG:
                print(f"Warning: Using {ArchitectureFeatures.DEFAULT_CONFIG} values for system configuration")

            if args.memory_mode == ArchitectureFeatures.DEFAULT_CONFIG:
                print(f"Warning: Using {ArchitectureFeatures.DEFAULT_CONFIG} values for memory mode")

            arch = architecture_features.ArchitectureFeatures(
                vela_config_files=args.config,
                system_config=args.system_config,
                memory_mode=args.memory_mode,
                accelerator_config=args.accelerator_config,
                max_blockdep=args.max_block_dependency,
                verbose_config=args.verbose_config,
                arena_cache_size=args.arena_cache_size,
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
            verbose_weights=args.verbose_weights,
            verbose_performance=args.verbose_performance,
            verbose_progress=args.verbose_progress,
            show_cpu_operations=args.show_cpu_operations,
            tensor_allocator=args.tensor_allocator,
            timing=args.timing,
            force_symmetric_int_weights=args.force_symmetric_int_weights,
            output_dir=args.output_dir,
            cpu_tensor_alignment=args.cpu_tensor_alignment,
            hillclimb_max_iterations=args.hillclimb_max_iterations,
        )

        scheduler_options = scheduler.SchedulerOptions(
            optimization_strategy=args.optimise,
            sram_target=arch.arena_cache_size,
            verbose_schedule=args.verbose_schedule,
            verbose_progress=args.verbose_progress,
        )

        model_reader_options = model_reader.ModelReaderOptions()

        nng = process(
            args.network, args.enable_debug_db, arch, model_reader_options, compiler_options, scheduler_options, args.subgraph_output
        )

        if args.show_subgraph_io_summary:
            print_subgraph_io_summary(nng)

        return 0
    except VelaError as e:
        print(e.data)
        return 1
