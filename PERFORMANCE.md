# Vela Performance Estimation Summary

This is a description of the performance estimation summary that Vela prints
after each compilation.

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

51 passes fused into 51
0/106 (0.0%) operations falling back to the CPU
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

This shows the selected accelerator configuration. It identifies the Embedded
NPU that the compiler is targeting. **NOTE: It is extremely important to select
the correct device, otherwise a run-time check in the driver will fail.**
To select a different accelerator configuration use the CLI option
`--accelerator-config`, see [OPTIONS.md](OPTIONS.md#Accelerator-Configuration).

### System configuration

The selected system configuration from the provided configuration file or
`internal-default`. **NOTE: It is very important to select a system
configuration that correctly describes the target embedded system.** This is
because the compiler makes some of its optimization decisions based upon this
information. Failing to select the correct configuration could result in
run-time errors, bit-inexact operation, or suboptimal operation of the Embedded
NPU. To select a different system configuration use the CLI option
`--system-config`, see [OPTIONS.md](OPTIONS.md#System-Config).

### Memory mode

The selected memory mode from the provided configuration file or
internal-default. **NOTE: It is very important to select a memory
mode that correctly describes the target embedded system.** This is
because the compiler makes some of its optimization decisions based upon this
information. To select a different memory mode use the CLI option
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

Information about cascading and operators.
The first line shows the number of passes (i.e. operations) and how many NPU
passes they have been fused or combined into.
The second line shows how many operators in the network are falling back to
the CPU (i.e. not supported by the NPU).

```
51 passes fused into 51
0/106 (0.0%) operations falling back to the CPU
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

The final sections show the estimated required compute power and performance for
the network.

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

### Neural network macs

This shows the estimated number of MACs in the network per batch. This number
includes MACs from convolutions and vector products, not from operations such as
elementwise and pooling operations.

### Network Tops/s

This shows the estimated TOPs/s for the network, which is an alternative
representation of [Neural network macs](#Neural-network-macs)

### Cycles

This shows the estimated number of cycles per batch for NPU, memory accesses and
in total. The total is the sum of the single action that consumes the most
cycles per pass, i.e. if memory access consumes the most cycles for a pass only
that will account for the pass cycles in the total.

### Batch Inference time

This shows the estimated inference time and inferences per second per batch.
**NOTE: This is just an estimate, for more accurate numbers we recomend to run
the compiled network in the software model.**
