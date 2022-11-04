# SPDX-FileCopyrightText: Copyright 2020-2022 Arm Limited and/or its affiliates <open-source-office@arm.com>
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
# Writes out per-pass and summary performance statistics to CSV files.
import csv
import sys

import numpy as np

from .nn_graph import PassPlacement
from .npu_performance import BandwidthDirection
from .npu_performance import PassCycles
from .numeric_util import round_up_to_int
from .operation import Op
from .tensor import MemArea
from .tensor import TensorPurpose


def mem_areas_to_report():
    # Exclude SHRAM, as the SHRAM performance numbers only cover LUT usage
    return [area for area in MemArea.all() if area != MemArea.Shram]


def write_summary_metrics_csv(nng, summary_filename, arch):
    with open(summary_filename, "w") as f:
        writer = csv.writer(f)
        mem_areas = mem_areas_to_report()

        labels = [
            "experiment",
            "network",
        ]

        labels += (
            ["accelerator_configuration", "system_config", "memory_mode", "core_clock", "arena_cache_size"]
            + [area.identifier_name() + "_bandwidth" for area in mem_areas]
            + ["weights_storage_area", "feature_map_storage_area"]
        )

        labels += [
            "inferences_per_second",
            "batch_size",
            "inference_time",
            "passes_before_fusing",
            "passes_after_fusing",
        ]
        labels += [area.identifier_name() + "_memory_used" for area in mem_areas]
        labels += ["total_original_weights"]
        labels += ["total_npu_encoded_weights"]

        for mem_area in mem_areas:
            labels += [
                mem_area.identifier_name() + "_feature_map_read_bytes",
                mem_area.identifier_name() + "_feature_map_write_bytes",
                mem_area.identifier_name() + "_weight_read_bytes",
                mem_area.identifier_name() + "_weight_write_bytes",
                mem_area.identifier_name() + "_total_bytes",
            ]

        labels += ["nn_macs", "nn_tops"]

        labels += ["cycles_" + kind.identifier_name() for kind in PassCycles.all()]

        writer.writerow(labels)

        data_items = [
            "default",
            nng.name,
        ]

        if arch:
            data_items += (
                [
                    arch.accelerator_config.name,
                    arch.system_config,
                    arch.memory_mode,
                    arch.core_clock,
                    arch.arena_cache_size / 1024,
                ]
                + [arch.memory_bandwidths_per_second[mem_area] / 1000.0 / 1000 / 1000 for mem_area in mem_areas]
                + [
                    arch.tensor_storage_mem_area[TensorPurpose.Weights].display_name(),
                    arch.tensor_storage_mem_area[TensorPurpose.FeatureMap].display_name(),
                ]
            )

        midpoint_inference_time = nng.cycles[PassCycles.Total] / arch.core_clock
        if midpoint_inference_time > 0:
            midpoint_fps = 1 / midpoint_inference_time
        else:
            midpoint_fps = np.nan

        n_passes = sum(len(sg.passes) for sg in nng.subgraphs)
        n_cascaded_passes = sum(len(sg.cascaded_passes) for sg in nng.subgraphs)

        data_items += [midpoint_fps, nng.batch_size, midpoint_inference_time, n_passes, n_cascaded_passes]
        data_items += [nng.memory_used.get(mem_area, 0) / 1024.0 for mem_area in mem_areas]
        data_items += [nng.total_original_weights]
        data_items += [nng.total_npu_encoded_weights]

        for mem_area in mem_areas:
            bws = nng.bandwidths[mem_area]
            total_bw = np.sum(bws)
            weight_bws = bws[TensorPurpose.Weights]
            fm_bws = bws[TensorPurpose.FeatureMap]
            data_items += [
                fm_bws[BandwidthDirection.Read],
                fm_bws[BandwidthDirection.Write],
                weight_bws[BandwidthDirection.Read],
                weight_bws[BandwidthDirection.Write],
                total_bw,
            ]

        data_items += [
            nng.macs,
            nng.macs * 2 * midpoint_fps / 1e12,
        ]

        data_items += [nng.cycles[kind] for kind in PassCycles.all()]

        writer.writerow(data_items)


def write_pass_metrics_csv(nng, pass_filename):

    with open(pass_filename, "w") as f:
        writer = csv.writer(f)

        purpose_list = (
            ("total", (TensorPurpose.Weights, TensorPurpose.FeatureMap)),
            ("weights", (TensorPurpose.Weights,)),
            ("feature_map", (TensorPurpose.FeatureMap,)),
        )

        direction_list = (
            ("total", (BandwidthDirection.Read, BandwidthDirection.Write)),
            ("read", (BandwidthDirection.Read,)),
            ("write", (BandwidthDirection.Write,)),
        )
        bandwidth_names = []
        bandwidth_indices = []
        for mem_area in mem_areas_to_report():
            for purpose, purpose_candidates in purpose_list:
                for direction, direction_candidates in direction_list:
                    label = "bytes_{}_{}_{}".format(mem_area.identifier_name(), purpose, direction)
                    bandwidth_names.append(label)
                    bandwidth_indices.append((mem_area, purpose_candidates, direction_candidates))

        all_cycles = (
            PassCycles.Total,
            PassCycles.Npu,
            PassCycles.SramAccess,
            PassCycles.DramAccess,
            PassCycles.OnChipFlashAccess,
            PassCycles.OffChipFlashAccess,
        )
        writer.writerow(
            [
                "name",
                "operators",
                "placement",
                "streaming_strategy",
                "block_config_height",
                "block_config_width",
                "block_config_input_channels",
                "block_config_output_channels",
            ]
            + ["cycles_" + v.identifier_name() for v in all_cycles]
            + ["nn_macs"]
            + bandwidth_names
            + ["sram_used"]
        )

        def write_subgraph(sg):
            for cps in sg.cascaded_passes:
                if cps.placement == PassPlacement.StartupInit:
                    continue  # skip the dummy init pass

                for ps in cps.passes:
                    if len(ps.ops) == 1 and ps.ops[0].type == Op.CustomNpuOp:
                        # just treat this as a call, unroll it
                        write_subgraph(ps.ops[0].attrs["subgraph"])
                        continue
                    stats = [ps.name, " ".join(op.type.name for op in ps.ops)]
                    stats += [ps.placement.name]
                    stats += [cps.strategy.name]
                    stats += list(ps.block_config)
                    stats += [round_up_to_int(ps.cycles[v]) for v in all_cycles]
                    stats += [round_up_to_int(ps.macs)]
                    for indices in bandwidth_indices:
                        res = 0
                        i = indices[0]
                        for j in indices[1]:
                            for k in indices[2]:
                                res += round_up_to_int(ps.bandwidths[i, j, k])
                        stats.append(res)
                    try:
                        stats += [ps.sram_used]
                    except AttributeError:
                        stats += [0]

                    writer.writerow(stats)

        write_subgraph(nng.get_root_subgraph())


def print_performance_metrics_for_strat(
    arch,
    name,
    cycles,
    macs,
    bandwidths,
    batch_size,
    memory_used,
    cpu_operations=None,
    npu_operations=None,
    show_cpu_operations=False,
    weights_data=None,
    f=sys.stdout,
):

    orig_mem_areas_labels = [(v, v.display_name()) for v in mem_areas_to_report()]

    midpoint_inference_time = cycles[PassCycles.Total] / arch.core_clock
    if midpoint_inference_time > 0:
        midpoint_fps = 1 / midpoint_inference_time
    else:
        midpoint_fps = np.nan

    mem_area_labels = [
        (mem_area, label) for mem_area, label in orig_mem_areas_labels if np.sum(bandwidths[mem_area]) > 0
    ]

    if name:
        print("", file=f)
        print(f"Network summary for {name}", file=f)
    print(f"Accelerator configuration        {arch.accelerator_config.name:>20}", file=f)
    print(f"System configuration             {arch.system_config:>20}", file=f)
    print(f"Memory mode                      {arch.memory_mode:>20}", file=f)
    print(f"Accelerator clock                        {int(arch.core_clock / 1e6):12d} MHz", file=f)
    for mem_area, label in mem_area_labels:
        label += " bandwidth"
        bandwidth = arch.memory_bandwidths_per_second[mem_area] / 1000.0 / 1000 / 1000
        print(
            f"Design peak {label:25}    {bandwidth:12.2f} GB/s",
            file=f,
        )
    print(file=f)
    for mem_area, label in mem_area_labels:
        if mem_area not in memory_used:
            continue

        aug_label = label + " used"

        print(f"Total {aug_label:25}          {memory_used[mem_area] / 1024.0:12.2f} KiB", file=f)

    print(file=f)

    if cpu_operations is None:
        cpu_operations = []
    if npu_operations is None:
        npu_operations = []

    n_cpu_operations = len(cpu_operations)
    n_npu_operations = len(npu_operations)
    n_total_operations = max(n_cpu_operations + n_npu_operations, 1)  # avoid potential divide by zero

    def format_tens_list(lst):
        return " ".join(str(list(tens.shape)) for tens in lst)

    for str_ops_type, n_ops, ops in (
        ("CPU", n_cpu_operations, cpu_operations),
        ("NPU", n_npu_operations, npu_operations),
    ):
        print(f"{str_ops_type} operators = {n_ops:d} ({n_ops / n_total_operations:4.1%})", file=f)
        if show_cpu_operations:
            for op in ops:
                print(
                    f"   {str_ops_type}: {op.type} = {op.name}"
                    f" (inputs {format_tens_list(op.inputs)}, outputs {format_tens_list(op.outputs)})"
                )

    print("", file=f)

    for mem_area, label in mem_area_labels:
        bws = bandwidths[mem_area]
        total_bw = np.sum(bws)
        weight_bws = bws[TensorPurpose.Weights]
        fm_bws = bws[TensorPurpose.FeatureMap]
        aug_label = label + " bandwidth"
        print(
            f"Average {aug_label:25}        {total_bw * midpoint_fps / 1000.0 / 1000.0 / 1000.0:12.2f} GB/s",
            file=f,
        )
        print(
            f"Input   {aug_label:25}        {np.sum(fm_bws[BandwidthDirection.Read]) / 1000.0 / 1000.0:12.2f} MB/batch",
            file=f,
        )
        print(f"Weight  {aug_label:25}        {np.sum(weight_bws) / 1000.0 / 1000.0:12.2f} MB/batch", file=f)
        print(
            f"Output  {aug_label:25}        "
            f"{np.sum(fm_bws[BandwidthDirection.Write]) / 1000.0 / 1000.0:12.2f} MB/batch",
            file=f,
        )
        print(f"Total   {aug_label:25}        {total_bw / 1000.0 / 1000.0:12.2f} MB/batch", file=f)
        print(
            f"Total   {aug_label:25} per input "
            f"{total_bw / 1000.0 / 1000.0 / batch_size:9.2f} MB/inference (batch size {batch_size:d})",
            file=f,
        )
        print(file=f)

    if weights_data:
        print(f"Original Weights Size                    {weights_data['original'] / 1024.0:12.2f} KiB", file=f)
        print(f"NPU Encoded Weights Size                 {weights_data['npu_encoded'] / 1024.0:12.2f} KiB", file=f)
        print(file=f)

    print(
        f"Neural network macs                      {int(macs):12d} MACs/batch",
        file=f,
    )
    print(
        f"Network Tops/s                           {macs * 2 * midpoint_fps / 1e12:12.2f} Tops/s",
        file=f,
    )
    print(file=f)

    for kind in PassCycles.all():
        aug_label = kind.display_name() + " cycles"
        cyc = cycles[kind]
        print(f"{aug_label:30}           {int(cyc):12d} cycles/batch", file=f)
    print(file=f)

    print(
        f"Batch Inference time              {midpoint_inference_time * 1000:7.2f} ms,"
        f" {midpoint_fps:7.2f} inferences/s (batch size {batch_size:d})",
        file=f,
    )
    print(file=f)


def print_performance_metrics(nng, arch, show_cpu_operations=False, verbose_weights=False, f=sys.stdout):
    cpu_operations = []
    npu_operations = []
    ir_only_ops = (
        Op.Const,
        Op.Placeholder,
        Op.CustomNpuOp,
        Op.SubgraphInput,
    )

    for sg in nng.subgraphs:
        if sg.placement == PassPlacement.Cpu:
            for op in sg.get_all_ops():
                if op.type not in ir_only_ops:
                    cpu_operations.append(op)
        elif sg.placement == PassPlacement.Npu:
            for op in sg.get_all_ops():
                if op.type not in ir_only_ops:
                    npu_operations.append(op)

    weights_data = (
        {"original": nng.total_original_weights, "npu_encoded": nng.total_npu_encoded_weights}
        if verbose_weights
        else None
    )
    return print_performance_metrics_for_strat(
        arch,
        nng.name,
        nng.cycles,
        nng.macs,
        nng.bandwidths,
        nng.batch_size,
        nng.memory_used,
        cpu_operations,
        npu_operations,
        show_cpu_operations,
        weights_data,
        f,
    )
