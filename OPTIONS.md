# Vela Options

This file contains a more verbose and detailed description of the Vela
Compiler's CLI options than the built-in help strings.  It also defines and
describes the Vela system configuration file format.

## Command Line Interface

### Network (required)

Filename of the network model to compile. The file has to be a `.tflite` file.  
**Type: POSIX path**  
**Default: N/A**  

```bash
vela path/to/network.tflite
```

### Help

Displays the help strings of all CLI options. Can be used without the required
Network argument.  
**Type: N/A**  
**Default: N/A**  

```bash
vela --help
```

### Version

Displays the version of the installed Vela Compiler. Can be used without the
required Network argument.  
**Type: N/A**  
**Default: N/A**  

```bash
vela --version
```

### Output Directory

Specifies the output directory of the optimised network model as well as the
`.csv` files containing performance estimations.  
**Type: POSIX path**  
**Default: ./output**  

```bash
vela network.tflite --output-dir ./custom_directory
```

### Config

Specifies the path to the config file. The file has to be a `.ini` file. The
format is described further in a the Config section below.  
**Type: POSIX path**  
**Default: use default configuration**  

```bash
vela network.tflite --config custom_config.ini
```

### Cascading

Controls the packing of multiple passes into cascades. This allows for lower
memory usage. If the network's intermediate feature maps are too large for the
system's SRAM this optimisation is required.  
**Type: Boolean**  
**Default: True**  

```bash
vela network.tflite --cascading False
```

### Keep scale placement

Prevents scheduler from placing scale tensors for IFM streamed passes in SRAM
and keeps these in flash.  

```bash
vela network.tflite --keep-scale-placement
```

### Force Block Config

Force a specific block configuration in the format HxWxC, where H, W, and C are
positive integers specifying height, width, and channels (depth), respectively.
The default behaviour is Vela searching for an optimal block configuration. An
exception will be raised if the chosen block configuration is incompatible.  
**Type: String**  
**Default: N/A**  

```bash
vela network.tflite --force-block-config 2x2x8
```

### Timing

Measure time taken for different compiler steps, e.g. model reading and
scheduling. Prints the results to standard out.  
**Type: Set True**  
**Default: False**  

```bash
vela network.tflite --timing
```

### Accelerator Configuration

Choose which hardware accelerator configuration to compile for. Format is
accelerator name followed by a hyphen, followed by the number of MACs in the
configuration.
**Type: String**  
**Default: ethos-u55-256**  
**Choices: [ethos-u55-32, ethos-u55-64, ethos-u55-128, ethos-u55-256]**  

```bash
vela network.tflite --accelerator-config ethos-u55-64
```

### System Config

Selects the system configuration to use as specified in the System Configuration
File (see section below).  
**Type: String**  
**Default: Use internal default config**  

```bash
vela network.tflite --system-config MySysConfig
```

### Tensor Allocator

Specify which allocator algorithm to use for non-constant NPU and CPU tensor
allocation.  
**Type: String**  
**Default: Greedy**  
**Choices: [Greedy, LinearAlloc]**  

```bash
vela network.tflite --tensor-allocator=LinearAlloc
```

### Ifm Streaming

Controls scheduler IFM streaming search. Vela's scheduler will choose between
IFM Streaming and Weight Streaming for optimal memory usage. Disabling this will
cause Vela to always choose Weight Streaming.  
**Type: Boolean**  
**Default: True**  

```bash
vela network.tflite --ifm-streaming False
```

### Block Config Limit

Limit the block config search space. This will result in faster compilation
times but may impact the performance of the output network. Use 0 for unlimited
search.  
**Type: Integer**  
**Choices: >= 0**  
**Default: 16**  

```bash
vela network.tflite --block-config-limit 0
```

### Global Memory Clock Scale

Performs an additional scaling of the individual memory clock scales specified
by the system configuration. Used to globally adjust the bandwidth of the
various memories  
**Type: Float**  
**Default: 1.0**  

```bash
vela network.tflite --global-memory-clock-scale 1.5
```

### Pareto Metric

Controls the calculation of the pareto metric. Use 'BwCycMemBlkH' to consider
Block Height in addition to Bandwidth, Cycle count and Memory. This can reduce
SRAM usage in some circumstances.  
**Type: String**  
**Default: BwCycMem**  
**Choices: [BwCycMem, BwCycMemBlkH]**  

```bash
vela network.tflite --pareto-metric BwCycMemBlkH
```

### Recursion Limit

Some of Vela's algorithms use recursion and the required depth can be network
dependant. This option allows the limit to be increased if needed. The maximum
limit is platform dependent. If limit is set too low then compilation will raise
a RecursionError exception.  
**Type: Integer**  
**Default: 10000**  

```bash
vela network.tflite --recursion-limit 50000
```

### Enable Debug DB

The neural network debug database allows tracking of optimisations from the
input network graph to the output command stream.  Set this option to enable the
calculation and writing of an XML file that contains the network debug database
tables to the output directory.  
**Type: Boolean**  
**Default: Disabled**  

```bash
vela network.tflite --enable-debug-db
```

### Max Block Dependency

Set the maximum value that can be used for the block dependency delay between
NPU kernel operations. A lower value may result in longer execution time.  
**Type: Integer**  
**Default: 3**  
**Choices: [0, 1, 2, 3]**  

```bash
vela network.tflite --max-block-dependency 0
```

### Tensor Format Between Cascaded Passes

Controls if NHCWB16 or NHWC Tensor format should be used in between cascaded passes. NHWCB16 means FeatureMaps are laid
out in 1x1x16B bricks in row-major order. This enables more efficient FeatureMap reading from external memory.  
**Type: Boolean**  
**Default: True**  
**Choices: [True, False]**  

```bash
vela network.tflite --nhcwb16-between-cascaded-passes
```

### Scaling of weight estimates

Performs an additional scaling of weight compression estimate used by Vela to estimate SRAM usage.
Increasing this scaling factor will make the estimates more conservative (lower) and this can result
in optimisations that use less SRAM, albeit at the cost of performance (inference speed).  
**Type: Float**  
**Default: 1.0**  

```bash
vela network.tflite --weight-estimation-scaling=1.2
```

### Allocation alignment

Controls the allocation byte alignment. Only affects CPU tensors, NPU tensors will remain 16-byte
aligned independent of this option. Alignment has to be a power of two and greater or equal to 16.  
**Type: Integer**  
**Default: 16**  

```bash
vela network.tflite --allocation-alignment 128
```

## Verbose Print Options

All of the options below are disabled by default and enabling them will add
prints to standard out without any functional changes.  

### Show Subgraph IO Summary

Prints a summary of all the subgraphs and their inputs and outputs.  

```bash
vela network.tflite --show-subgraph-io-summary
```

### Show Minimum Possible Allocation

Prints the minimum possible allocation.  

```bash
vela network.tflite --show-minimum-possible-allocation
```

### Show Cpu Operations

Show the operations that fall back to the CPU.  

```bash
vela network.tflite --show-cpu-operations
```

### Verbose Graph

Verbose graph rewriter.  

```bash
vela network.tflite --verbose-graph
```

### Verbose Quantization

Verbose quantization.  

```bash
vela network.tflite --verbose-quantization
```

### Verbose Packing

Verbose pass packing.  

```bash
vela network.tflite --verbose-packing
```

### Verbose Tensor Purpose

Verbose tensor purpose.  

```bash
vela network.tflite --verbose-tensor-purpose
```

### Verbose Tensor Format

Verbose tensor format.  

```bash
vela network.tflite --verbose-tensor-format
```

### Verbose Schedule

Verbose schedule.  

```bash
vela network.tflite --verbose-schedule
```

### Verbose Pareto Frontier Schedules

Show all schedules along the pareto frontier of optimisation criteria.  

```bash
vela network.tflite --verbose-pareto-frontier-schedules
```

### Verbose Allocation

Verbose tensor allocation.  

```bash
vela network.tflite --verbose-allocation
```

### Verbose High Level Command Stream

Verbose high level command stream.  

```bash
vela network.tflite --verbose-high-level-command-stream
```

### Verbose Register Command Stream

Verbose register command stream.  

```bash
vela network.tflite --verbose-register-command-stream
```

### Verbose Operators

Verbose operator list.  

```bash
vela network.tflite --verbose-operators
```

## System Configuration File

This is used to describe various properties of the embedded system that the
network will run in. The configuration file is selected with the `--config` CLI
option. The system config is selected by Name (defined in the
`[SysConfig.Name]` field) with the CLI option `--system-config`. The `cpu=X`
attribute in the `[SysConfig.Name]` is used to cross-reference and select CPU
operator attributes in the `[CpuPerformance.OpName]` section.  
Example usage based on the file below:  

```bash
vela network.tflite --config sys_cfg_vela.ini --system-config MySysConfig
```

Example of a Vela system configuration file.  

```ini
; File: sys_cfg_vela.ini
; The file contains two parts; a system config part and a CPU operator
; performance part.

; System config
; Specifies properties such as the core clock speed, the size and speed of the
; four potential memory areas, and for various types of data which memory area
; is used to store them. The cpu property is used to link with the CPU operator
; performance.
; The four potential memory areas are: Sram, Dram, OnChipFlash, OffChipFlash.

[SysConfig.MySysConfig]
npu_freq=500e6
cpu=MyCpu
Sram_clock_scale=1
Sram_port_width=64
Dram_clock_scale=1
Dram_port_width=64
OnChipFlash_clock_scale=1
OnChipFlash_port_width=64
OffChipFlash_clock_scale=0.25
OffChipFlash_port_width=32
permanent_storage_mem_area=OffChipFlash
feature_map_storage_mem_area=Sram
fast_storage_mem_area=Sram

; CPU operator performance
; Specifies properties that are used by a linear model to estimate the
; performance for any operations that will be run on the CPU (such as those not
; supported by the NPU). Setting the intercept and slope to 0 will result in
; the operator being excluded from the performance estimation. This is the same
; as not specifying the operator. If an explicit cpu is specified rather than
; using the default then the cpu name must match the cpu specified in the
; SysConfig.<system config name> section.

[CpuPerformance.MyCpuOperator]
default.intercept=0.0
default.slope=1.0

MyCpu.intercept=0.0
MyCpu.slope=1.0
```
