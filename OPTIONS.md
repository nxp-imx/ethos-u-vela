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

Generate the SUPPORTED_OPS.md file in the current working directory. Contains
a summary table for each supported network model format (TFLite/TOSA). The
tables shows all the operators that can be placed on the NPU, and what the
constraints are for that operator to be scheduled on the NPU. If the constraints
are not met for a TFLite operator, then it will be scheduled on the CPU instead.
For TOSA operators there are no fallback to the CPU. Note: There is limited
support for compiling a TOSA neural network (EXPERIMENTAL). Can be used without
the required Network argument.  
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
**Default: Use `internal-default` config.  This maps to the following configs from the example `vela.ini` file**  

- **Ethos-U65** - System configuration Ethos-U65 Client-Server: SRAM (16 GB/s)
  and DRAM (12 GB/s)  
- **Ethos-U55** - System configuration Ethos-U55 High-End Embedded: SRAM
  (4 GB/s) and Flash (0.5 GB/s)  

```bash
vela network.tflite --config my_vela_cfg.ini --system-config My_Sys_Cfg
```

### Memory Mode

Selects the memory mode to use as specified in the Vela configuration file (see
section below).  
**Type: String**  
**Default: Use `internal-default` config.  This maps to the following configs from the example `vela.ini` file**  

- **Ethos-U65** - Memory mode Dedicated SRAM: the SRAM is only for use by the
  Ethos-U.  The non-SRAM memory is assumed to be read-writeable  
- **Ethos-U55** - Memory mode Shared SRAM: the SRAM is shared between the
  Ethos-U and the Cortex-M software.  The non-SRAM memory is assumed to be
  read-only  

```bash
vela network.tflite --config my_vela_cfg.ini --memory-mode My_Mem_Mode
```

### Tensor Allocator

Specify which allocator algorithm to use for non-constant NPU and CPU tensor
allocation.  
**Type: String**  
**Default: HillClimb**  
**Choices: [Greedy, LinearAlloc, HillClimb]**  

```bash
vela network.tflite --tensor-allocator=LinearAlloc
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

### Optimise

Set the optimisation strategy. The Size strategy results in minimal SRAM usage
(it does not use arena cache memory area size).  The Performance strategy
results in maximal performance (it uses the arena cache memory area size if
specified either via the CLI option of Vela configuration file).
**Type: String**  
**Default: Performance**  
**Choices: [Size, Performance]**  

```bash
vela network.tflite --optimise Size
```

### Arena Cache Size

Set the size of the arena cache memory area, in bytes.  If specified, this
option overrides the memory mode attribute with the same name in a Vela
configuration file.  If neither this nor the memory mode attribute are specified
then a size equal to the maximum address supported by the Ethos-U is used.  This
option is intended to be used with the `--optimise Performance` option.  
**Type: Integer**  
**Choices: [ >= 0]**  

```bash
vela network.tflite --optimise Performance --arena-cache-size 2097152
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

### Recursion Limit

Sets the Python internal limit to depth of recursion. It may be
necessary to increase this from the default for very large networks
due to the recursive nature of the graph traversal algorithm.
If Vela fails with a `RecursionError`, try increasing the limit using
this option to see if it resolves the issue.  
Please note that this option may not work as intended on Microsoft Windows
systems, as there is a hard limit on thread stack size.  
**Type: Integer**  
**Default: 1000**

```bash
vela network.tflite --recursion-limit 2000
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

### Verbose All

Enable all `--verbose-*` options.  

```bash
vela network.tflite --verbose-all
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

### Verbose Weights

Verbose weights information.  

```bash
vela network.tflite --verbose-weights
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
core_clock=???                 ---> Clock frequency of the Ethos-U.  ??? = {float in Hz}
axi0_port=???                  ---> Memory type connected to AXI0.  ??? = {Sram, Dram, OnChipFlash or OffChipFlash}
axi1_port=???                  ---> Memory type connected to AXI1.  ??? = {Sram, Dram, OnChipFlash or OffChipFlash}
Sram_clock_scale=???           ---> Scaling of core_clock to specify the Sram bandwidth.  Only required if selected by an AXI port.  ??? = {float 0.0 to 1.0}
Sram_burst_length=???          ---> Minimum efficient burst length in Sram. Only required if selected by an AXI port. ??? = {int in Bytes}
Sram_read_latency=???          ---> Read latency in Sram. Only required if selected by an AXI port. ??? = {int in Cycles}
Sram_write_latency=???         ---> Write latency in Sram. Only required if selected by an AXI port. ??? = {int in Cycles}
Dram_clock_scale=???           ---> Scaling of core_clock to specify the Dram bandwidth.  Only required if selected by an AXI port.  ??? = {float 0.0 to 1.0}
Dram_burst_length=???          ---> Minimum efficient burst length in Dram. Only required if selected by an AXI port. ??? = {int in Bytes}
Dram_read_latency=???          ---> Read latency in Dram. Only required if selected by an AXI port. ??? = {int in Cycles}
Dram_write_latency=???         ---> Write latency in Dram. Only required if selected by an AXI port. ??? = {int in Cycles}
OnChipFlash_clock_scale=???    ---> Scaling of core_clock to specify the OnChipFlash bandwidth.  Only required if selected by an AXI port.  ??? = {float 0.0 to 1.0}
OffChipFlash_clock_scale=???   ---> Scaling of core_clock to specify the OffChipFlash bandwidth.  Only required if selected by an AXI port.  ??? = {float 0.0 to 1.0}
OffChipFlash_burst_length=???  ---> Minimum efficient burst length in OffChipFlash. Only required if selected by an AXI port. ??? = {int in Bytes}
OffChipFlash_read_latency=???  ---> Read latency in OffChipFlash. Only required if selected by an AXI port. ??? = {int in Cycles}
OffChipFlash_write_latency=??? ---> Write latency in OffChipFlash. Only required if selected by an AXI port. ??? = {int in Cycles}

; -----------------------------------------------------------------------------
; Memory Mode

; My_Mem_Mode_Parent
[Memory_Mode.My_Mem_Mode_Parent]
const_mem_area=???     ---> AXI port used by the read-only data (e.g. weight tensors, scale & bias tensors).  ??? = {Axi0, Axi1}
arena_mem_area=???     ---> AXI port used by the read-write data (e.g. feature map tensors, internal buffers).  ??? = {Axi0, Axi1}
cache_mem_area=???     ---> AXI port used by the dedicated SRAM read-write (e.g. feature map part-tensors, internal buffers).  ??? = {Axi0, Axi1}
arena_cache_size=???   ---> Size of the arena/cache memory area.  ??? = {int in Bytes}

; My_Mem_Mode_Child
[Memory_Mode.My_Mem_Mode_Child]
inherit=???            ---> Parent section to inherit from.  An option in the child overwrites an identical option in the parent.  ??? = {[Part.Name]}
arena_cache_size=???   ---> Size of the arena/cache memory area.  ??? = {int in Bytes}
```

## Memory Modes

The Vela configuration file defines three potential memory modes although other configurations are possible.  Each
memory mode is defined with respect to four attributes.  If any of those attributes are not specified then an internal
default value will be used.  Note that this value may not be valid for the target embedded system.  Therefore, the user
is recommended to explicitly specify all settings.  
The three memory area attributes are each assigned to a virtual AXI port.  This assignment is used by the compiler to
map a memory area to a specific memory type (as defined in the System Configuration section).  It allows the System
Configuration sections to be reused with different Memory Mode sections.  It does not control the mapping of the
physical AXI ports of the hardware, which are pre-determined in the compiler and driver.

1. `const_mem_area` this is the memory area in which the compiler will store all constant data such as weights,
scales & biases, and constant value tensors.
1. `arena_mem_area` this is the memory area in which the compiler will look to access the TensorFlow Lite for
Microcontrollers Tensor Arena.
1. `cache_mem_area` this is the memory area in which the compiler uses as a cache memory if required by the selected
memory mode
1. `arena_cache_size` this is the size of the memory area available to the compiler for use by either the arena or cache
depending upon the memory mode

Please note that all of the above attributes must have values that correspond to the settings used by the Ethos-U Driver
and the TensorFlow Lite for Microcontrollers Application.  This is because the compiler does not have any direct control
over these other components.

### Sram Only Mode

In this mode, the Embedded NPU only has access to SRAM memory.  The compiler will make use of two regions in the SRAM,
which may be separate or contiguous.  One region is used for the `const_mem_area` and the other region is used for the
`arena_mem_area`.  It is assumed that SRAM outside of these regions will be used by other software in the system (e.g.
TensorFlow Lite for Microcontrollers or an RTOS running on the Cortex-M CPU).  The `cache_mem_area` is not used.  The
`arena_cache_size` refers to the size of the `arena_mem_area`. The TensorFlow Lite for Microcontrollers Tensor Arena
will contain all of the network input, output, and intermediate tensors, including the Ethos-U scratch tensor which
contains the NPU's internal working buffers.

### Shared Sram Mode

In this mode, the Embedded NPU has access to SRAM which is used for the `arena_mem_area`.  It also has access to some
other type of memory (e.g. Flash or DRAM) that is used for the `const_mem_area`.  The `cache_mem_area` is not used.  The
`arena_cache_size` refers to the size of the `arena_mem_area`.  It is assumed that SRAM outside of the `arena_mem_area`
will be used by other software in the system (e.g. TensorFlow Lite for Microcontrollers or an RTOS running on the
Cortex-M CPU).  The TensorFlow Lite for Microcontrollers Tensor Arena will contain all of the network input, output, and
intermediate tensors, including the Ethos-U scratch tensor which contains the NPU's internal working buffers.

### Dedicated Sram Mode

In this mode, the Embedded NPU has access to SRAM which is used for the `cache_mem_area`.  It is assumed that use of
this memory is entirely dedicated to the Embedded NPU, as no support is provided for allocating parts of this at
run-time.  It also has access to some other type of memory (e.g. DRAM).  The compiler will make use of two regions in
this other type of memory, which may be separate or contiguous.  One region is used for the `const_mem_area` and
the other region is used for the `arena_mem_area`.  The `arena_cache_size` refers to the size of the `cache_mem_area`.
It is assumed that memory outside of those regions will be used by other software in the system (e.g. TensorFlow Lite
for Microcontrollers or an RTOS running on the Cortex-M CPU).  The TensorFlow Lite for Microcontrollers Tensor Arena
will contain all of the network input, output, and intermediate tensors, including the Ethos-U scratch tensor which
contains the NPU's internal working buffers.