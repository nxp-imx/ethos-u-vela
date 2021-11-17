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

### Neural network MACs

This shows the estimated number of MACs in the network per batch. This number
includes MACs from convolutions, vector products and pooling operations.
It does not include MACs from elementwise or any other type of operation.

### Network Tops/s

This shows the estimated TOPs/s for the network, which is an alternative
representation of [Neural network MACs](#Neural-network-MACs)

### Cycles

This shows the estimated number of cycles per batch for NPU, memory accesses and
in total. The total is the sum of the single action that consumes the most
cycles per pass, i.e. if memory access consumes the most cycles for a pass only
that will account for the pass cycles in the total.  
To clarify: for each type of cycle counts, the number of cycles per batch is the
sum of cycle counts for each layer, where each layer's cycle count is based on
the maximal processing path. A layer consists of a feature map and an operator.  
For example, if the DMA transfer for a feature map requires less cycles than the
cycles for the operation, then the DMA cycles will not contribute to the layer
cycle count. As a result, it will not be part of the summed SRAM or DRAM access
cycles.  
Looking at the example above in [Estimated performance](#Estimated-performance),
the zero cycle count for DRAM Access cycles means that either there was no DRAM
access or, like in our previously described example, the DMA cycles were fewer
than for the operation for every layer that required a DMA transfer.

### Batch Inference time

This shows the estimated inference time and inferences per second per batch.
**NOTE: This is just an estimate, for more accurate numbers we recomend to run
the compiled network in the software model.**
