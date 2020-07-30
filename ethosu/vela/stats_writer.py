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
# Writes out per-pass and summary performance statistics to CSV files.
import csv
import sys

import numpy as np

from .nn_graph import PassPlacement
from .npu_performance import BandwidthDirection
from .npu_performance import MacCount
from .npu_performance import PassCycles
from .numeric_util import round_up_to_int
from .tensor import MemArea
from .tensor import TensorPurpose


def write_summary_metrics_csv(nng, summary_filename, arch):
    with open(summary_filename, "w") as f:
        writer = csv.writer(f)

        labels = [
            "experiment",
            "network",
        ]

        labels += (
            ["accelerator_configuration", "system_config", "npu_clock", "sram_size"]
            + [area.identifier_name() + "_bandwidth" for area in MemArea.all()]
            + ["weights_storage_area", "feature_map_storage_area"]
        )

        labels += [
            "inferences_per_second",
            "batch_size",
            "inference_time",
            "passes_before_fusing",
            "passes_after_fusing",
        ]
        labels += [area.identifier_name() + "_memory_used" for area in MemArea.all()]
        labels += ["on_chip_flash_bits_per_element", "off_chip_flash_bits_per_element"]

        for mem_area in MemArea.all():
            labels += [
                mem_area.identifier_name() + "_feature_map_read_bytes",
                mem_area.identifier_name() + "_feature_map_write_bytes",
                mem_area.identifier_name() + "_weight_read_bytes",
                mem_area.identifier_name() + "_weight_write_bytes",
                mem_area.identifier_name() + "_total_bytes",
            ]

        labels += ["nn_macs", "hardware_macs", "nn_tops", "hardware_tops"]

        labels += ["cycles_" + kind.identifier_name() for kind in PassCycles.all()]

        writer.writerow(labels)

        data_items = [
            "default",
            nng.name,
        ]

        if arch:
            data_items += (
                [arch.accelerator_config, arch.system_config, arch.npu_clock, arch.sram_size / 1024]
                + [arch.memory_bandwidths_per_second[mem_area] / 1000.0 / 1000 / 1000 for mem_area in MemArea.all()]
                + [
                    arch.tensor_storage_mem_area[TensorPurpose.Weights].display_name(),
                    arch.tensor_storage_mem_area[TensorPurpose.FeatureMap].display_name(),
                ]
            )

        midpoint_inference_time = nng.cycles[PassCycles.Total] / arch.npu_clock
        if midpoint_inference_time > 0:
            midpoint_fps = 1 / midpoint_inference_time
        else:
            midpoint_fps = np.nan

        n_passes = sum(len(sg.passes) for sg in nng.subgraphs)
        n_cascaded_passes = sum(len(sg.cascaded_passes) for sg in nng.subgraphs)

        data_items += [midpoint_fps, nng.batch_size, midpoint_inference_time, n_passes, n_cascaded_passes]
        data_items += [nng.memory_used.get(mem_area, 0) / 1024.0 for mem_area in MemArea.all()]

        data_items += [
            nng.bits_per_element.get(MemArea.OnChipFlash, 0.0),
            nng.bits_per_element.get(MemArea.OffChipFlash, 0.0),
        ]

        for mem_area in MemArea.all():
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
            nng.macs[MacCount.NeuralNetworkMacs],
            nng.macs[MacCount.HardwareMacs],
            nng.macs[MacCount.NeuralNetworkMacs] * 2 * midpoint_fps / 1e12,
            nng.macs[MacCount.HardwareMacs] * 2 * midpoint_fps / 1e12,
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
        for mem_area in MemArea.all():
            for purpose, purpose_candidates in purpose_list:
                for direction, direction_candidates in direction_list:
                    label = "bytes_%s_%s_%s" % (mem_area.identifier_name(), purpose, direction)
                    bandwidth_names.append(label)
                    bandwidth_indices.append((mem_area, purpose_candidates, direction_candidates))

        all_macs = MacCount.all()
        all_cycles = (
            PassCycles.Total,
            PassCycles.Dpu,
            PassCycles.ElementWise,
            PassCycles.Cpu,
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
                "n_blocks_in_pass",
            ]
            + ["cycles_" + v.identifier_name() for v in all_cycles]
            + [v.identifier_name() for v in all_macs]
            + bandwidth_names
            + ["sram_used"]
        )

        def write_subgraph(sg):
            for cps in sg.cascaded_passes:
                if cps.placement == PassPlacement.StartupInit:
                    continue  # skip the dummy init pass

                for ps in cps.passes:
                    if len(ps.ops) == 1 and ps.ops[0].type == "NpuOp":
                        # just treat this as a call, unroll it
                        write_subgraph(ps.ops[0].attrs["subgraph"])
                        continue
                    stats = [ps.name, " ".join(op.type for op in ps.ops)]
                    stats += [ps.placement.name]
                    stats += [cps.strategy.name]
                    stats += list(ps.block_config)
                    stats += [ps.n_blocks]
                    stats += [round_up_to_int(ps.cycles[v]) for v in all_cycles]
                    stats += [round_up_to_int(ps.macs[v]) for v in all_macs]
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
    num_passes,
    num_cascaded_passes,
    n_operations=0,
    cpu_operations=[],
    bits_per_element=None,
    show_cpu_operations=False,
    f=sys.stdout,
):

    orig_mem_areas_labels = [(v, v.display_name()) for v in MemArea.all()]

    midpoint_inference_time = cycles[PassCycles.Total] / arch.npu_clock
    if midpoint_inference_time > 0:
        midpoint_fps = 1 / midpoint_inference_time
    else:
        midpoint_fps = np.nan

    mem_area_labels = [
        (mem_area, label) for mem_area, label in orig_mem_areas_labels if np.sum(bandwidths[mem_area]) > 0
    ]

    if name:
        print("", file=f)
        print("Network summary for", name, file=f)
    print("Accelerator configuration        %20s" % (arch.accelerator_config,), file=f)
    print("System configuration             %20s" % (arch.system_config,), file=f)
    print("Accelerator clock                        %12d MHz" % (arch.npu_clock / 1e6,), file=f)
    for mem_area, label in mem_area_labels:
        print(
            "Design peak %-25s    %12.2f GB/s"
            % (label + " bandwidth", arch.memory_bandwidths_per_second[mem_area] / 1000.0 / 1000 / 1000,),
            file=f,
        )

    print(file=f)
    for mem_area, label in mem_area_labels:
        if mem_area not in memory_used:
            continue

        aug_label = label + " used"

        extra = ""
        if (mem_area == MemArea.OnChipFlash or mem_area == MemArea.OffChipFlash) and bits_per_element is not None:
            extra = " (%.2f bits per element)" % (bits_per_element[mem_area],)

        print("Total %-25s          %12.2f KiB%s" % (aug_label, memory_used[mem_area] / 1024.0, extra), file=f)

    print(file=f)
    print("%d passes fused into %d" % (num_passes, num_cascaded_passes), file=f)

    n_cpu_operations = len(cpu_operations)
    if n_operations > 0:
        print(
            "%d/%d (%4.1f %%) operations falling back to the CPU"
            % (n_cpu_operations, n_operations, n_cpu_operations / n_operations * 100),
            file=f,
        )

    if show_cpu_operations:
        for op in cpu_operations:

            def format_tens_list(lst):
                return " ".join(str(list(tens.shape)) for tens in lst)

            print(
                "CPU operation: %s, inputs %s, outputs %s"
                % (op.type, format_tens_list(op.inputs), format_tens_list(op.outputs)),
                file=f,
            )

        print("", file=f)

    for mem_area, label in mem_area_labels:
        bws = bandwidths[mem_area]
        total_bw = np.sum(bws)
        weight_bws = bws[TensorPurpose.Weights]
        fm_bws = bws[TensorPurpose.FeatureMap]
        aug_label = label + " bandwidth"
        print(
            "Average %-25s        %12.2f GB/s" % (aug_label, total_bw * midpoint_fps / 1000.0 / 1000.0 / 1000.0,),
            file=f,
        )
        print(
            "Input   %-25s        %12.2f MB/batch"
            % (aug_label, np.sum(fm_bws[BandwidthDirection.Read]) / 1000.0 / 1000.0,),
            file=f,
        )
        print("Weight  %-25s        %12.2f MB/batch" % (aug_label, np.sum(weight_bws) / 1000.0 / 1000.0,), file=f)
        print(
            "Output  %-25s        %12.2f MB/batch"
            % (aug_label, np.sum(fm_bws[BandwidthDirection.Write]) / 1000.0 / 1000.0,),
            file=f,
        )
        print("Total   %-25s        %12.2f MB/batch" % (aug_label, total_bw / 1000.0 / 1000.0,), file=f)
        print(
            "Total   %-25s per input %9.2f MB/inference (batch size %d)"
            % (aug_label, total_bw / 1000.0 / 1000.0 / batch_size, batch_size),
            file=f,
        )
        print(file=f)

    print("Neural network macs                      %12d MACs/batch" % (macs[MacCount.NeuralNetworkMacs],), file=f)
    print("Hardware macs                            %12d MACs/batch" % (macs[MacCount.HardwareMacs],), file=f)
    print(
        "Network Tops/s                           %12.2f Tops/s"
        % (macs[MacCount.NeuralNetworkMacs] * 2 * midpoint_fps / 1e12),
        file=f,
    )
    print(
        "Hardware Tops/s                          %12.2f Tops/s"
        % (macs[MacCount.HardwareMacs] * 2 * midpoint_fps / 1e12),
        file=f,
    )
    print(file=f)

    for kind in PassCycles.all():
        aug_label = kind.display_name() + " cycles"
        cyc = cycles[kind]
        print("%-30s           %12d cycles/batch" % (aug_label, cyc,), file=f)
    print(file=f)

    print(
        "Batch Inference time              %7.2f ms, %7.2f inferences/s (batch size %d)"
        % (midpoint_inference_time * 1000, midpoint_fps, batch_size),
        file=f,
    )
    print(file=f)


def print_performance_metrics(nng, arch, show_cpu_operations=False, f=sys.stdout):
    n_passes = sum(len(sg.passes) for sg in nng.subgraphs)
    n_cascaded_passes = sum(len(sg.cascaded_passes) for sg in nng.subgraphs)
    n_operations = sum(len(ps.ops) for sg in nng.subgraphs for ps in sg.passes)
    cpu_operations = sum((ps.ops for sg in nng.subgraphs for ps in sg.passes if ps.placement == PassPlacement.Cpu), [])
    return print_performance_metrics_for_strat(
        arch,
        nng.name,
        nng.cycles,
        nng.macs,
        nng.bandwidths,
        nng.batch_size,
        nng.memory_used,
        n_passes,
        n_cascaded_passes,
        n_operations,
        cpu_operations,
        nng.bits_per_element,
        show_cpu_operations,
        f,
    )


def write_human_friendly_metrics(nng, arch, filename):
    f = open(filename, "w")
    print_performance_metrics(nng, arch, f=f)
