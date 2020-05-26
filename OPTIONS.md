# Vela Options

This document contains a description of all of the Command Line Interface (CLI) options.  It also contains a definition and description of the System Configuration file format.

## CLI Options

### Compulsory (Positional) Arguments

|Option Name|Value|Default Value|Description|
|-|-|-|-|
|NETWORK|POSIX pathname|-|Filename of the .tflite model to compile.|

### Optional Arguments

|Option Name|Value|Default Value|Description|
|-|-|-|-|
|-h, --help|-|-|Show the help message and exit.|
|--version|-|-|Show program's version number and exit.|
|--output-dir \<Value>|POSIX pathname|./output|Output directory to write the output files to. The output files are an optimised version of the input model with a `_vela.tflite` file prefix, along with the performance estimate (EXPERIMENTAL) CSV files).|
|--accelerator-config \<Value>|ethos-u55-256, ethos-u55-128, <br/>ethos-u55-64, or ethos-u55-32|ethos-u55-256|Accelerator configuration to use.|
|--block-config-limit \<Value>|Positive integer|16|Limits the block config search space, use zero for unlimited.|
|--cascading \<Value>|True or False|True|Controls the packing of multiple passes into a cascade.|
|--config \<Value>|POSIX pathname|-|Specify a [Vela system configuration file](#system-configuration-file) to read.|
|--dram-bandwidth \<Value>|Floating point|0.0|DRAM memory bandwidth in GB/s, use zero to select the value from system config.|
|--force-block-config \<Value>|HxWxC where H,W and C are integers|-|Force a specific block configuration.|
|--global-memory-clock-scale \<Value>|Floating point|1.0|Performs an additional scaling of the individual memory clock scales specified by the system config.|
|--ifm-ofm-overlap \<Value>|True or False|True|Controls the overlapping of IFM and OFM buffers.|
|--ifm-streaming \<Value>|True or False|True|Controls scheduler IFM streaming search.|
|--inter-pass-cycle-delay \<Value>|Integer|0|Artificial delay between passes, measured in NPU cycles.|
|--max-block-dependency \<Value>|0 to 3 inclusive|3|Set the maximum value that can be used for the block dependency between npu kernel operations.|
|--pareto-metric \<Value>|BwCycMem or BwCycMemBlkH|BwCycMem|Controls the calculation of the pareto metric.|
|--permanent-storage \<Value>|OnChipFlash or OffChipFlash|OffChipFlash|Memory area for permanent storage. To store the weights and other constant data in SRAM select 'OnChipFlash'.|
|--recursion-limit \<Value>|Positive integer|10000|Set the recursion depth limit, may result in RecursionError if too low.|
|--system-config \<Value>|String|internal-default|System configuration to use.|
|--tensor-allocator \<Value>|LinearAlloc or Greedy|Greedy|Tensor Allocator algorithm.|
|--timing|-|Disabled|Enable the timing of the compiler's operations|
|--verbose-allocation|-|-|Verbose tensor allocation|
|--verbose-graph|-|-|Verbose graph rewriter|
|--verbose-high-level-command-stream|-|-|Verbose high level command stream|
|--verbose-operators|-|-|Verbose operator list|
|--verbose-packing|-|-|Verbose pass packing|
|--verbose-pareto-frontier-schedules|-|-|Show all schedules along the pareto frontier of optimisation criteria|
|--verbose-quantization|-|-|Verbose quantization|
|--verbose-register-command-stream|-|-|Verbose register command stream|
|--verbose-schedule|-|-|Verbose schedule|
|--verbose-tensor-format|-|-|Verbose tensor format|
|--verbose-tensor-purpose|-|-|Verbose tensor purpose|
|--show-cpu-operations|-|-|Show the operations that fall back to the CPU|
|--show-minimum-possible-allocation|-|-|Show the minimum possible allocation|
|--show-subgraph-io-summary|-|-|Shows a summary of all the subgraphs and their inputs and outputs|

## System Configuration File

This is used to describe various properties of the embedded system that the network will run in.

Example of a Vela system configuration file.

```
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
