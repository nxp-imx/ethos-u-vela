<!--
SPDX-FileCopyrightText: Copyright 2021-2023 Arm Limited and/or its affiliates <open-source-office@arm.com>

SPDX-License-Identifier: Apache-2.0

Licensed under the Apache License, Version 2.0 (the License); you may
not use this file except in compliance with the License.
You may obtain a copy of the License at

www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an AS IS BASIS, WITHOUT
WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
-->
# Vela Performance Estimation Summary

This is a description of the performance estimation summary that Vela prints
after each compilation.  This summary is also printed to a csv in the output
directory.

The following is an example of the output.
```
$ vela my_network.tflite

Network summary for my_network
Accelerator configuration               Ethos_U55_256
System configuration                 internal-default
Memory mode                          internal-default
Accelerator clock                                 500 MHz
Design peak SRAM bandwidth                       4.00 GB/s
Design peak Off-chip Flash bandwidth             0.50 GB/s

Total SRAM used                                  0.95 KiB
Total Off-chip Flash used                      106.98 KiB

CPU operators = 0 (0.0%)
NPU operators = 44 (100.0%)

Average SRAM bandwidth                           0.04 GB/s
Input   SRAM bandwidth                           0.01 MB/batch
Weight  SRAM bandwidth                           0.00 MB/batch
Output  SRAM bandwidth                           0.00 MB/batch
Total   SRAM bandwidth                           0.01 MB/batch
Total   SRAM bandwidth            per input      0.01 MB/inference (batch size 1)

Average Off-chip Flash bandwidth                 0.46 GB/s
Input   Off-chip Flash bandwidth                 0.01 MB/batch
Weight  Off-chip Flash bandwidth                 0.09 MB/batch
Output  Off-chip Flash bandwidth                 0.00 MB/batch
Total   Off-chip Flash bandwidth                 0.10 MB/batch
Total   Off-chip Flash bandwidth  per input      0.10 MB/inference (batch size 1)

Neural network macs                             86952 MACs/batch
Network Tops/s                                   0.00 Tops/s

NPU cycles                                      21298 cycles/batch
SRAM Access cycles                               2261 cycles/batch
DRAM Access cycles                                  0 cycles/batch
On-chip Flash Access cycles                         0 cycles/batch
Off-chip Flash Access cycles                   112755 cycles/batch
Total cycles                                   114098 cycles/batch

Batch Inference time                 0.23 ms, 4382.18 inferences/s (batch size 1)
```

## Configuration

The first section of the summary shows the configuration used for
optimizing the network.

```
Accelerator configuration               Ethos_U55_256
System configuration                 internal-default
Memory mode                          internal-default
Accelerator clock                                 500 MHz
Design peak SRAM bandwidth                       4.00 GB/s
Design peak Off-chip Flash bandwidth             0.50 GB/s
```

### Accelerator configuration

This shows the selected accelerator configuration.  It identifies the Embedded
NPU that the compiler is targeting.  **NOTE: It is crucial to select
the correct device, otherwise a run-time check in the driver will fail.**
To select a different accelerator configuration use the CLI option
`--accelerator-config`, see [OPTIONS.md](OPTIONS.md#Accelerator-Configuration).

### System configuration

The selected system configuration from the provided configuration file or
`internal-default`.  **NOTE: It is very important to select a system
configuration that correctly describes the target embedded system.  ** This is
because the compiler makes some of its optimization decisions based upon this
information.  Failing to select the correct configuration could result in
run-time errors, bit-inexact operation, or suboptimal operation of the Embedded
NPU.  To select a different system configuration use the CLI option
`--system-config`, see [OPTIONS.md](OPTIONS.md#System-Config).

### Memory mode

The selected memory mode from the provided configuration file or
internal-default.  **NOTE: It is very important to select a memory
mode that correctly describes the target embedded system.  ** This is
because the compiler makes some of its optimization decisions based upon this
information.  To select a different memory mode use the CLI option
`--memory-mode`, see [OPTIONS.md](OPTIONS.md#Memory-Mode).

### Accelerator clock

The accelerator clock for the given the system configuration.

### Design peak memory bandwidth

The design peak memory bandwidth for the given system configuration.
It gives the theoretical maximum bandwidth of the memory based upon the
[System Configuration](OPTIONS.md#Configuration-File) parameters specified
and the AXI port width of the Ethos-U NPU.

## Memory Usage

The next section of the summary shows the memory usage for the
the various memory types in the system.

```
Total SRAM used                                  0.95 KiB
Total Off-chip Flash used                      106.98 KiB
```

The contents of this section and the meaning of it changes depending upon the
system config and memory mode.

## Operator information

Information about the number of operators that will run on the CPU and NPU.

```
CPU operators = 0 (0.0%)
NPU operators = 44 (100.0%)
```

## Estimated memory bandwidth

The next section shows the estimated memory bandwidth for each memory type.
Data is provided for average, batch and per data type.

```
Average SRAM bandwidth                           0.04 GB/s
Input   SRAM bandwidth                           0.01 MB/batch
Weight  SRAM bandwidth                           0.00 MB/batch
Output  SRAM bandwidth                           0.00 MB/batch
Total   SRAM bandwidth                           0.01 MB/batch
Total   SRAM bandwidth            per input      0.01 MB/inference (batch size 1)

Average Off-chip Flash bandwidth                 0.46 GB/s
Input   Off-chip Flash bandwidth                 0.01 MB/batch
Weight  Off-chip Flash bandwidth                 0.09 MB/batch
Output  Off-chip Flash bandwidth                 0.00 MB/batch
Total   Off-chip Flash bandwidth                 0.10 MB/batch
Total   Off-chip Flash bandwidth  per input      0.10 MB/inference (batch size 1)
```

### Average bandwidth

This shows the average memory bandwidth usage for the memory type.

### Input bandwidth

This shows the memory bandwidth usage for reading feature maps for the memory
type per batch.

### Weight bandwidth

This shows the memory bandwidth usage for reading and writing weights for the
memory type per batch.

### Output bandwidth

This shows the memory bandwidth usage for writing feature maps for the memory
type per batch.

### Total bandwidth

This shows the total memory bandwidth usage the memory
type per batch and per inference.

## Weights data

This section is only visible if the CLI option `--verbose-weights` is provided.
```
Original Weights Size                           84.91 KiB
NPU Weights Size                                94.00 KiB
NPU Encoded Weights Size                        89.30 KiB
```

### Original Weights Size

This is the total size of all weights in the network before optimization.

### NPU Weights Size

This is the total size of the weights rearranged and padded to fit the NPUs
block based processing.

### NPU Encoded Weights Size

This is the total size of the [NPU Weights](#NPU-Weights-Size) after being
encoded for the NPU.

## Estimated performance

The final sections show the estimated required compute power and performance
for the network.

```
Neural network macs                             86952 MACs/batch
Network Tops/s                                   0.00 Tops/s

NPU cycles                                      21298 cycles/batch
SRAM Access cycles                               2261 cycles/batch
DRAM Access cycles                                  0 cycles/batch
On-chip Flash Access cycles                         0 cycles/batch
Off-chip Flash Access cycles                   112755 cycles/batch
Total cycles                                   114098 cycles/batch

Batch Inference time                 0.23 ms, 4382.18 inferences/s (batch size 1)
```

### Neural network MACs

This shows the estimated number of MACs in the network per batch.  This number
includes MACs from convolutions, vector products and pooling operations.
It does not include MACs from elementwise or any other type of operation.

### Network Tops/s

This shows the estimated TOPs/s for the network, which is an alternative
representation of [Neural network MACs](#Neural-network-MACs)

### Cycles

This shows the estimated number of cycles per batch for NPU, memory accesses
and in total. The total is the sum of the single action that consumes the most
cycles per pass, i.e. if memory access consumes the most cycles for a pass
only that will account for the pass cycles in the total.  
To clarify: for each type of cycle counts, the number of cycles per batch is the
sum of cycle counts for each layer, where each layer's cycle count is based on
the maximal processing path.  
A layer consists of a feature map and an operator. For example, if the DMA
transfer for a feature map requires less cycles than the cycles for the
operation, then the DMA cycles will not contribute to the layer cycle count.
As a result, it will not be part of the summed SRAM or DRAM access cycles.  
Looking at the example above in [Estimated performance](#Estimated-performance),
the zero cycle count for DRAM Access cycles means that either there was no DRAM
access or, like in our previously described example, the DMA cycles were fewer
than for the operation for every layer that required a DMA transfer.

### Batch Inference time

This shows the estimated inference time and inferences per second per batch.
**NOTE: This is just an estimate, for more accurate numbers we recomend to run
the compiled network in the software model.**

# Vela Performance Estimation Per-Layer

This section describes the per-layer performance output that is printed when the
--verbose-performance option is used. This is also printed to a csv file in the
output directory.

The following is an example of the output:

```
################################################################################
Performance for NPU Subgraph _split_1
TFLite_operator      NNG Operator         SRAM Usage  Peak%  Op Cycles Network%        NPU    SRAM AC    DRAM AC OnFlash AC OffFlash AC  MAC Count Network%  Util% Name
-------------------- -------------------- ---------- ------ ---------- -------- ---------- ---------- ---------- ---------- ----------- ---------- -------- ------ --------------------
CONV_2D              Conv2DBias               629616  86.18    1889913    46.80    1889913      21504          0          0           0   99090432    49.99  20.48 ResNet18/activation_32/Relu;ResNet18/batch_normalization_32/FusedBatchNormV3;ResNet18/conv2d_38/BiasAdd/ReadVariableOp/resource;ResNet18/conv2d_38/BiasAdd;ResNet18/conv2d_39/Conv2D;ResNet18/conv2d_38/Conv2D1
CONV_2D              Conv2DBias               730624 100.00    2127584    52.69    2127584      21504          0          0           0   99090432    49.99  18.19 ResNet18/batch_normalization_33/FusedBatchNormV3;ResNet18/conv2d_39/BiasAdd/ReadVariableOp/resource;ResNet18/conv2d_39/BiasAdd;ResNet18/conv2d_39/Conv2D
ADD                  Add                       43008   5.89      16128     0.40      16128       8064          0          0           0          0     0.00   0.00 ResNet18/activation_33/Relu;ResNet18/add_15/add
AVERAGE_POOL_2D      AvgPool                   27648   3.78       4224     0.10       2200       4224          0          0           0      24576     0.01   2.27 ResNet18/average_pooling2d_1/AvgPool
```

The columns in the above output have the following meaning:

## TFLite Operator

Shows the original type of the operator that the scheduled operator corresponds
to.  This column may not contain all of the operators that are in the input
network because some compiler optimisations may end up removing some operators.

## NNG Operator

Shows the operator used by Vela's internal representation at the layer-level.
There is a direct mapping between type of operator in Vela's internal
representation and the type of those that are run on the hardware.
However, there may be a multiple number of operators that are run
on the hardware for every one of Vela's internal representation.

## SRAM Usage

Shows the SRAM usage in terms of bytes and as a fraction (%) of peak usage,
where peak usage is the usage of the op with the largest usage.

## Op Cycles

Shows the total cycle estimation for the operator in terms of cycles and as a
fraction (%) of the estimated total cycles of the entire network.

The cycle counts are then broken down into NPU, SRAM AC, DRAM AC, OnFlash AC
and OffFlashAC:

### NPU

The estimated number of total cycles for the entire NPU.

### SRAM AC, DRAM AC, OnFlash AC, OffFlash AC

Estimated number of Access cycles for respective memory

## Mac Count

Shows the total MAC count in terms of actual count and as a fraction of the
total MACs.  Note that this is not an estimation.

### MAC Util

Shows the estimated Macs/cycle as a fraction of the theoretical maximum
MACs/cycle.

## Name

Shows the name of the operator in Vela.