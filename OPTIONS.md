# Vela Options

This file contains a more verbose and detailed description of the Vela
Compiler's CLI options than the built-in help strings.  It also defines and
describes Vela's configuration file format.

## Command Line Interface

### Network (required)

Filename of the network model to compile.  The file has to be a `.tflite` file.  
**Type: POSIX path**  
**Default: N/A**  

```bash
vela path/to/network.tflite
```

### Help

Displays the help strings of all CLI options.  Can be used without the required
Network argument.  
**Type: N/A**  
**Default: N/A**  

```bash
vela --help
```

### Version

Displays the version of the installed Vela Compiler.  Can be used without the
required Network argument.  
**Type: N/A**  
**Default: N/A**  

```bash
vela --version
```

### API version

Displays the version of the external API.  Can be used without the
required Network argument.  
**Type: N/A**  
**Default: N/A**  

```bash
vela --api-version
```

### Supported Operator Report

Generate the SUPPORTED_OPS.md file in the current working directory.
Contains a summary table of all TFLite operators that can be placed on the NPU,
and what the constraints are for that operator to be scheduled on the NPU.
If the constraints are not met, then it will be scheduled on the CPU instead.
Can be used without the required Network argument.  
**Type: N/A**  
**Default: N/A**  

```bash
vela --supported-ops-report
```

### Output Directory

Specifies the output directory of the optimised network model as well as the
`.csv` files containing performance estimations.  
**Type: POSIX path**  
**Default: ./output**  

```bash
vela network.tflite --output-dir ./custom_directory
```

### Enable Debug Database

The neural network debug database allows tracking of optimisations from the
input network graph to the output command stream.  Set this option to enable the
calculation and writing of an XML file that contains the network debug database
tables to the output directory.  

```bash
vela network.tflite --enable-debug-db
```

### Config

Specifies the path to the Vela configuration file.  The format of the file is a
Python ConfigParser `.ini` file.  This option can be specified multiple times to
allow multiple files to be searched for the required system config and memory
mode.  More details can be found in the Configuration File section below.  
**Type: POSIX path**  
**Default: use default configuration**  

```bash
vela network.tflite --config my_vela_cfg1.ini --config my_vela_cfg2.ini --system-config My_Sys_Cfg --memory-mode My_Mem_Mode
```

### Cache Bias Scale Tensor

Controls whether the scheduler caches the bias & scale tensors in SRAM or if it
leaves them in Flash.  This only affects IFM streamed passes.  
**Type: Boolean**  
**Default: True**  

```bash
vela network.tflite --cache-bias-scale-tensor False
```

### Cascading

Controls the packing of multiple passes into cascades.  This allows for lower
memory usage.  If the network's intermediate feature maps are too large for the
system's SRAM this optimisation is required.  
**Type: Boolean**  
**Default: True**  

```bash
vela network.tflite --cascading False
```

### Force Block Config

Force a specific block configuration in the format HxWxC, where H, W, and C are
positive integers specifying height, width, and channels (depth), respectively.
The default behaviour is Vela searching for an optimal block configuration.  An
exception will be raised if the chosen block configuration is incompatible.  
**Type: String**  
**Default: N/A**  

```bash
vela network.tflite --force-block-config 2x2x8
```

### Timing

Measure time taken for different compiler steps, e.g. model reading and
scheduling.  Prints the results to standard out.

```bash
vela network.tflite --timing
```

### Accelerator Configuration

Choose which hardware accelerator configuration to compile for.  Format is
accelerator name followed by a hyphen, followed by the number of MACs in the
configuration.  
**Type: String**  
**Default: ethos-u55-256**  
**Choices: [ethos-u55-32, ethos-u55-64, ethos-u55-128, ethos-u55-256, ethos-u65-256, ethos-u65-512]**  

```bash
vela network.tflite --accelerator-config ethos-u55-64
```

### System Config

Selects the system configuration to use as specified in the Vela configuration
file (see section below).  
**Type: String**  
**Default: Use internal default config**  

```bash
vela network.tflite --config my_vela_cfg.ini --system-config My_Sys_Cfg
```

### Memory Mode

Selects the memory mode to use as specified in the Vela configuration file (see
section below).  
**Type: String**  
**Default: Use internal default config**  

```bash
vela network.tflite --config my_vela_cfg.ini --memory-mode My_Mem_Mode
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

Controls scheduler IFM streaming search.  Vela's scheduler will choose between
IFM Streaming and Weight Streaming for optimal memory usage.  Disabling this
will cause Vela to always choose Weight Streaming.  
**Type: Boolean**  
**Default: True**  

```bash
vela network.tflite --ifm-streaming False
```

### Block Config Limit

Limit the block config search space.  This will result in faster compilation
times but may impact the performance of the output network.  Use 0 for unlimited
search.  
**Type: Integer**  
**Default: 16**  
**Choices: >= 0**  

```bash
vela network.tflite --block-config-limit 0
```

### Pareto Metric

Controls the calculation of the pareto metric.  Use 'BwCycMemBlkH' to consider
Block Height in addition to Bandwidth, Cycle count and Memory.  This can reduce
SRAM usage in some circumstances.  
**Type: String**  
**Default: BwCycMem**  
**Choices: [BwCycMem, BwCycMemBlkH]**  

```bash
vela network.tflite --pareto-metric BwCycMemBlkH
```

### Recursion Limit

Some of Vela's algorithms use recursion and the required depth can be network
dependant.  This option allows the limit to be increased if needed.  The maximum
limit is platform dependent.  If limit is set too low then compilation will
raise a RecursionError exception.  
**Type: Integer**  
**Default: 10000**  

```bash
vela network.tflite --recursion-limit 50000
```

### Max Block Dependency

Set the maximum value that can be used for the block dependency delay between
NPU kernel operations.  A lower value may result in longer execution time.  
**Type: Integer**  
**Default: 3**  
**Choices: [0, 1, 2, 3]**  

```bash
vela network.tflite --max-block-dependency 0
```

### Tensor Format Between Cascaded Passes

Controls if NHCWB16 or NHWC Tensor format should be used in between cascaded
passes.  NHWCB16 means FeatureMaps are laid out in 1x1x16B bricks in row-major
order.  This enables more efficient FeatureMap reading from external memory.  
**Type: Boolean**  
**Default: True**  

```bash
vela network.tflite --nhcwb16-between-cascaded-passes False
```

### Scaling of weight estimates

Performs an additional scaling of weight compression estimate used by Vela to
estimate SRAM usage.  Increasing this scaling factor will make the estimates
more conservative (lower) and this can result in optimisations that use less
SRAM, albeit at the cost of performance (inference speed).  
**Type: Float**  
**Default: 1.0**  

```bash
vela network.tflite --weight-estimation-scaling=1.2
```

### CPU Tensor Alignment

Controls the allocation byte alignment.  This affects all CPU tensors including
Ethos-U Custom operator inputs and outputs.  In this instance a CPU tensor is
defined as any tensor that is explicitly listed in the resulting `.tflite` file.
The Ethos-U NPU internal tensors will remain 16-byte aligned independent of this
option, these tensors are contained within the command stream.  Alignment has to
be a power of two and greater or equal to 16.  
**Type: Integer**  
**Default: 16**  

```bash
vela network.tflite --allocation-alignment 128
```

## Verbose Print Options

All of the options below are disabled by default and enabling them will add
prints to standard out without any functional changes.  

### Show Cpu Operations

Show the operations that fall back to the CPU.  

```bash
vela network.tflite --show-cpu-operations
```

### Show Subgraph IO Summary

Prints a summary of all the subgraphs and their inputs and outputs.  

```bash
vela network.tflite --show-subgraph-io-summary
```

### Verbose Config

Verbose system configuration and memory mode.  If no `--system-config` or
`--memory-mode` CLI options are specified then the `internal-default` values
will be displayed.  

```bash
vela network.tflite --verbose-config
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

## Configuration File

This is used to describe various properties of the Ethos-U embedded system.  The
configuration file is selected using the `--config` CLI option along with a file
that describes the properties.  The format of the file is a Python ConfigParser
`.ini` file format consists of sections used to identify a configuration, and
key/value pair options used to specify the properties.  All sections and
key/value pairs are case-sensitive.

There are two types of section, system configuration `[System_Config.*]`
sections and memory mode `[Memory_Mode.*]` sections.  A complete Ethos-U
embedded system should define at least one entry in each section, where an entry
is identified using the format `[Part.Name]` (Part = {System_Config or
Memory_Mode}, Name = {a string with no spaces}.).  A configuration file may
contain multiple entries per section, with the entries `.Name` being used to
select it using the `--system-config` and `--memory-mode` CLI options.  If the
CLI options are not specified then the sections named `internal-default` are
used.  These are special sections which are defined internally and contain
default values.

Each section contains a number of options which are described in more detail
below.  All options are optional.  If they are not specified, then they will be
assigned a value of 1 (or the equivalent).  They will not be assigned the value
of `internal-default`.

One special option is the `inherit` option.  This can be used in any section and
its value is the name of another section to inherit options from.  The only
restriction on this option is that recursion is not allowed and so it cannot
reference its own section.

To see the configuration values being used by Vela use the `--verbose_config`
CLI option.  This can also be used to display the internal-default values and to
see a full list of all the available options.

An example Vela configuration file, called `vela_cfg.ini`, is included in the
directory containing this file.  Example usage based on this file is:  

```bash
vela network.tflite --accelerator-config ethos-u55-256 --config vela_cfg.ini --system-config Ethos_U55_High_End_Embedded --memory-mode Shared_Sram
```

The following is an in-line explanation of the Vela configuration file format:

```ini
; file: my_vela_cfg.ini
; -----------------------------------------------------------------------------
; Vela configuration file

; -----------------------------------------------------------------------------
; System Configuration

; My_Sys_Cfg
[System_Config.My_Sys_Cfg]
core_clock=???               ---> Clock frequency of the Ethos-U.  ??? = {float in Hz} 
axi0_port=???                ---> Memory type connected to AXI0.  ??? = {Sram, Dram, OnChipFlash or OffChipFlash}
axi1_port=???                ---> Memory type connected to AXI1.  ??? = {Sram, Dram, OnChipFlash or OffChipFlash}
Sram_clock_scale=???         ---> Scaling of core_clock to specify the Sram bandwidth.  Only required if selected by an AXI port.  ??? = {float 0.0 to 1.0}
Dram_clock_scale=???         ---> Scaling of core_clock to specify the Dram bandwidth.  Only required if selected by an AXI port.  ??? = {float 0.0 to 1.0}
OnChipFlash_clock_scale=???  ---> Scaling of core_clock to specify the OnChipFlash bandwidth.  Only required if selected by an AXI port.  ??? = {float 0.0 to 1.0}
OffChipFlash_clock_scale=??? ---> Scaling of core_clock to specify the OffChipFlash bandwidth.  Only required if selected by an AXI port.  ??? = {float 0.0 to 1.0}

; -----------------------------------------------------------------------------
; Memory Mode

; My_Mem_Mode_Parent
[Memory_Mode.My_Mem_Mode_Parent]
const_mem_area=???          ---> AXI port used by the read-only data (e.g. weight tensors, scale & bias tensors).  ??? = {Axi0, Axi1}
arena_mem_area=???          ---> AXI port used by the read-write data (e.g. feature map tensors, internal buffers).  ??? = {Axi0, Axi1}
cache_mem_area=???          ---> AXI port used by the dedicated SRAM read-write (e.g. feature map part-tensors, internal buffers).  ??? = {Axi0, Axi1}
cache_sram_size=???         ---> Size of the dedicated cache SRAM.  Only required when cache_mem_area != arena_mem_area.  ??? = {int in Bytes}

; My_Mem_Mode_Child
[Memory_Mode.My_Mem_Mode_Child]
inherit=???                 ---> Parent section to inherit from.  An option in the child overwrites an identical option in the parent.  ??? = {[Part.Name]}
cache_sram_size=???         ---> Size of the dedicated cache SRAM.  Only required when cache_mem_area != arena_mem_area.  ??? = {int in Bytes}
```
