# Vela
This tool is used to compile a [TensorFlow Lite for Microcontrollers](https://www.tensorflow.org/lite/microcontrollers) neural network model into an optimised version that can run on an embedded system containing an [Ethos-U55 NPU](https://www.arm.com/products/silicon-ip-cpu/ethos/ethos-u55).

The optimised model will contain TensorFlow Lite Custom operators for those parts of the model that can be accelerated by the Ethos-U55. Parts of the model that cannot be accelerated are left unchanged and will instead run on the Cortex-M series CPU using an appropriate kernel (such as the [Arm](https://www.arm.com) optimised [CMSIS-NN](https://github.com/ARM-software/CMSIS_5/tree/develop/CMSIS/NN) kernels).

After compilation the optimised model can only be run on an Ethos-U55 NPU embedded system.

The tool will also generate performance estimates (EXPERIMENTAL) for the compiled model.

## Environment
Vela runs on the Linux operating system.

## Prerequisites
The following should be installed prior to the installation of Vela:
 - Python >= 3.6
 - GNU toolchain (GCC, Binutils and libraries) or alternative C compiler/linker toolchain

## Installation
Before running, the Vela package must be installed along with all its dependencies. To do this, first change to the directory that contains this README.md file. Then use the command:
```
pip3 install -U setuptools>=40.1.0
pip3 install .
```

Or, if you use the `pipenv` virtual environment tool:
```
pipenv install .
```

## Running
Vela is run with an input `.tflite` file passed on the command line. This file contains the neural network to be compiled. The tool then outputs an optimised version with a `_vela.tflite` file prefix, along with the performance estimate (EXPERIMENTAL) CSV files, all to the output directory.

If you use the `pipenv` virtual environment tool then first start by spawning a shell in the virtual environment.:
```
pipenv shell
```
After which running Vela is the same regardless of whether you are in a virtual environment or not.

Example usage:
1) Compile the network `my_model.tflite`. The optimised version will be output to `./output/my_network_vela.tflite`.
```
vela my_model.tflite
```
2) Compile the network `/path/to/my_model.tflite` and specify the output to go in the directory `./results_dir/`.
```
vela --output-dir ./results_dir /path/to/my_model.tflite
```
3) To get a list of all available options:
```
vela --help
```
4) To specifiy information about the embedded system's configuration use Vela's system configuration file. The following command selects the `MySysConfig` settings that are described in the `sys_cfg_vela.ini` system configuration file. More details can be found in the next section.
```
vela --config sys_cfg_vela.ini --system-config MySysConfig my_model.tflite
```

### Vela's System Configuration file
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

## Contribution Guidlines and Pull Requests
Contributions are accepted under [Apache License 2.0](LICENSE.txt). Only submit contributions where you have authored all of the code.

## Resources
* [Ethos-U55](https://www.arm.com/products/silicon-ip-cpu/ethos/ethos-u55)

## License
Vela is licensed under [Apache License 2.0](LICENSE.txt)
## Contributions and Pull Requests

Contributions are accepted under Apache-2.0. Only submit contributions where you have authored all of the code.

### Sanity checks

The Python codebase is PEP8 compliant with the exception of 120 characters line length.
We run black and flake8 against the code base excluding "ethosu/vela/tflite/" and "ethosu/vela/ethos\_u55\_regs" directories because they are auto-generated by third party tools.
Those tools are run using [pre-commit framework](https://pre-commit.com/). The configuration file is .pre-commit-config.yaml

#### Install tools

To install pre-commit, run the following:

```
pipenv install -e . --dev
```

After the installation, pre-commit is available in the virtual environment.

#### Install the pre-commit hook

To ease the development, we can run those sanity checks before committing the code.
To install the git hook, run:

```
$ pre-commit install
pre-commit installed at .git/hooks/pre-commit
```

The checks will be run before the commit: if one of them fails, you need to fix the code to make the checks pass.

#### Run the sanity checks

Those checks can be run manually. This can be achievied running the following
```
$ pre-commit run flake8 --all-files
...
$ pre-commit run black --all-files
```

If you don't specify anything after run, it will execute all the checks.
