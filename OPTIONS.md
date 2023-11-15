<!--
SPDX-FileCopyrightText: Copyright 2020-2023 Arm Limited and/or its affiliates <open-source-office@arm.com>

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

### List Configuration Files

Displays the configuration files in the `ethosu/config_files` directory. All
configuration files must have the .ini extension and be placed in an
appropriately named directory under `ethosu/config_files`. Note that the file
depth from `ethosu/config_files` must be exactly 2 for the file to be
discovered (e.g. `config_files/directory_name/my_config_file.ini`). Can be
used without the required Network argument.

```bash
vela --list-config-files
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
mode.  Custom configuration files can be used by adding a .ini file in an
appropriate directory under the `ethosu/config_files` directory or by providing
its absolute path. More details can be found in the Configuration File and List
Configuration Files sections.
**Type: POSIX path**  
**Default: use default configuration**  

```bash
vela network.tflite --config DirectoryName/my_vela_cfg1.ini --config absolute/path/to/my_vela_cfg2.ini --system-config My_Sys_Cfg --memory-mode My_Mem_Mode
```

### Timing

Measure time taken for different compiler steps, e.g. model reading and
scheduling.  Prints the results to standard out.

```bash
vela network.tflite --timing
```

### Force Symmetric Int Weights

Forces symmetric quantization for signed integer weights. This means that all zero points are set
to 0. This might lead to unintended behaviour.

```bash
vela network.tflite --force-symmetric-int-weights
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
vela network.tflite --cpu-tensor-alignment 128
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

### HillClimb Max Iterations

Sets the maximum number of iterations the Hill Climb tensor allocator will run.
This is a hard limit on the total number of iterations of the algorithm.
Reducing this value is unlikely to reduce the compilation time of a working
solution, and it may cause the algorithm to terminate before finding a workable
solution.  
**Type: Integer**  
**Default: 99999**

```bash
vela network.tflite --hillclimb-max-iterations 1000
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

Displays two lists of operators. The first lists all of the operators that exist
in Vela's internal representation (Graph IR) of the Neural Network Graph (NNG)
before the graph optimisation process has run.  The second lists all of the
operators after that process.  The lists do not show the connectivity
information of the NNG and are unordered, therefore the execution order may
differ.  Each line in the list is of the format:  
`<num> <op_type> <op_name>`, where;  
num = an increasing operator count  
op_type = the Graph IR Operator Type  
op_name = the Graph IR Operator Name (this may have been derived from the
corresponding TFLite operator name)  

```bash
vela network.tflite --verbose-graph
```

### Verbose Quantization

Displays quantization information of all *weight*, *bias*, *input* and *output*
tensors for each operator in the Neural Network Graph (NNG).  The quantization
approximates floating point values as:
`approx_float_value = (integer_value - zero_point) * scale`
The information of each tensor is displayed in the format:
`<idx> <data_type> <min> <max> <scale> <zero_point> <name>`, where;  
idx = the tensor index on each operator  
min = the minimum floating point value before quantization  
max = the maximum floating point value before quantization  
scale = the quantization scaling, zero_point = the quantization zero point  
name = the name of the tensor  

```bash
vela network.tflite --verbose-quantization
```

### Verbose Packing

Displays a list of passes where a pass represents one or more Graph IR operators
that are run together as one hardware operation e.g. a pass could be a
convolution operator fused with a hardswish activation.  Each line of the list
has the format:  
`<id> <pass>`, where;  
id = an increasing pass count  
pass = name of the pass (usually derived from the first operator in the pass)  

```bash
vela network.tflite --verbose-packing
```

### Verbose Performance

Verbose per-layer Performance. Please see
[Vela Performance Estimation Summary](PERFORMANCE.md)
for a detailed explanation.  

```bash
vela network.tflite --verbose-performance
```

### Verbose Tensor Purpose

Displays a list of all operators and the tensors that are connected to them.
Additional information is shown about the tensors. The format is:
`<num> <op_type> <op_name> <direction> <idx> <purpose> <mem_area> <mem_type>
<tens>`, where;  
num = an increasing operator count  
op_type = the Graph IR Operator Type  
op_name = the Graph IR Operator Name (this may have been derived from the
corresponding TFLite operator name)  
direction = either *Input* or *Output* and indicates the connection direction of
the tensor with respect 
idx = the index position where on each operator  
purpose = purpose of the tensor (weight, bias, feature map, etc.)  
mem_area = assigned memory area (for example SRAM or Flash)  
mem_type = memory type (i.e. Scratch or Permanent NPU)  
tens = string representation of the tensor containing its name, shape and data
type  

```bash
vela network.tflite --verbose-tensor-purpose
```

### Verbose Schedule

Display all schedule operations which contain information about the operator
type, block config, stripe sizes, size of encoded weights, size of weight
buffers, depth slices, cascade assignment and SRAM usage. The purpose of the
scheduler is to come up with an execution plan for the network. It will make
decisions on how to split an operator execution into stripes, group operators
together in cascades to either reduce SRAM footprint or, in a multi-level
memory system, better utilize the SRAM. The scheduler will also decide in what
memory to put tensors as well as how to buffer data from a slower memory like
Flash/DRAM to SRAM.

Feature maps can be split up into horizontal subsections called stripes that
allow us to apply operators independently to smaller sections of feature maps.
The output stripes that are produced can fit into a smaller buffer than the
output of a full feature map would, which combined with cascading can reduce
memory usage.

A cascade is a group of operators that will be computed interleaved in stripes.
Instead of storing the full output of an operator applied on a whole feature
map, we calculate the smallest possible buffer that allows storing intermediate
results of enough output stripes of one operator to allow the consecutive
operator to calculate one output stripe. Then, the consumed parts of the buffer
that is no longer needed by the consecutive operator in the cascade can be
overwritten by a new output stripe of the first operator, allowing us to reuse
and reduce the memory usage.  

```bash
vela network.tflite --verbose-schedule
```

### Verbose Allocation

This option displays tensor allocation information in separate tables for each
type of memory area. Each table contains information about each tensor's start
and end time, address, size and purpose as well as the memory usage during the
each tensors live range. The start- and end time denotes the time steps during
when the tensor needs to be allocated in the memory. After the end time, the
addresses are allowed to be overwritten by other tensors. The reported memory
usage is the peak usage at any time step of the tensors live range, which means
that the maximum memory usage value of all tensors will be the minimum required
size to fit the proposed allocation.  

```bash
vela network.tflite --verbose-allocation
```

### Verbose High Level Command Stream

Display an enumerated list of High-Level (HL) commands in execution
order.  There are three types of command and each one displays individual
information:

* NPU Stripe = `<name> <ifm_box> <ifm2_box> <ofm_box> <weight_box>
<block_config>`, represents a data processing operation that maps directly to
a single Ethos-U operation where;  
name = name of the pass that corresponds to this HL command (not unique)  
ifm_box = part of the IFM in NHWC format  
ifm2_box = part of the IFM2 in NHWC format (is empty [] when not present)  
ofm_box = part of the OFM in NHWC format  
weight_box = part of the filter kernel in NHWC format  
block_config = block processing size in HWIO format

* DMA = `<in> <out> <box>`, represents a memory copy operation from source to
destination where;  
name = name of the pass that corresponds to this HL command (not unique)  
in = name of the source tensor  
out = name of the destination tensor  
box = part of the source tensor in NHWC format

* NOP = `<in> <out>`, represents a memory copy operation that has source equal
to destination and therefore does nothing, where;  
name = name of the pass that corresponds to this HL command (not unique)  
in = name of the input tensor  
out = name of the output tensor

```bash
vela network.tflite --verbose-high-level-command-stream
```

### Verbose Register Command Stream

Display two groups of information.  The first group is the input to the register
command stream generator.  The second group is the output of the register
command stream generator:

* Input = an enumerated list of the High-Level commands that are the input to
the generator.  Each command details all of its attributes.

* Output = a disassembly of the Ethos-U command stream (referred to as the
register command stream).  More information about the commands listed in the
register command stream can be found in the Arm Ethos-U NPU Technical Reference
Manuals that are available from the Arm Developer website (see
[README - Resources](README.md#resources)).

```bash
vela network.tflite --verbose-register-command-stream
```

### Verbose Operators

Display a list of all operators in the neural network graph along with their
attributes before any optimization is made by Vela.  

```bash
vela network.tflite --verbose-operators
```

### Verbose Weights

Displays the size of the *Original* and *Ethos-U NPU Encoded* weights as part of
the final summary information.  The *original* weights size refers to the size
of the weights as read from the input `.tflite` file.  The *NPU Encoded* weights
size refers to the total size of all of the weight tensors after they have been
reordered, padded and encoded for the operators that run on the Ethos-U.  

```bash
vela network.tflite --verbose-weights
```

### Verbose Progress

This option displays progress information of the most time consuming parts of
the compiler driver and scheduler.  

```bash
vela network.tflite --verbose-progress
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

An example Vela configuration file, called `vela.ini`, is included in the
`ethosu/config_files/Arm` directory. Example usage based on this file is:

```bash
vela network.tflite --accelerator-config ethos-u55-256 --config Arm/vela.ini --system-config Ethos_U55_High_End_Embedded --memory-mode Shared_Sram
```

Hardware vendors and/or users may wish to contribute their own configuration
files for various SoC platforms by adding a .ini file in an appropriate
directory under the ethosu/config_files directory.  This can be done by
following the process outlined in CONTRIBUTIONS.md. These can then be accessed
with `--config <DirectoryName>/config.ini` as in the example above.

To use configuration files located outside the config_files directory, provide
its absolute path to `--config`. The `--list-config-files` option can be used to
view all available configuration files:

```bash
vela --list-config-files
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