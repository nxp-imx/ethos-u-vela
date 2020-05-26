# Vela

This tool is used to compile a
[TensorFlow Lite for Microcontrollers](https://www.tensorflow.org/lite/microcontrollers)
neural network model into an optimised version that can run on an embedded
system containing an
[Ethos-U55 NPU](https://www.arm.com/products/silicon-ip-cpu/ethos/ethos-u55).

The optimised model will contain TensorFlow Lite Custom operators for those
parts of the model that can be accelerated by the Ethos-U55.  Parts of the model
that cannot be accelerated are left unchanged and will instead run on the
Cortex-M series CPU using an appropriate kernel (such as the
[Arm](https://www.arm.com) optimised
[CMSIS-NN](https://github.com/ARM-software/CMSIS_5/tree/develop/CMSIS/NN)
kernels).

After compilation the optimised model can only be run on an Ethos-U55 NPU
embedded system.

The tool will also generate performance estimates (EXPERIMENTAL) for the
compiled model.

## TensorFlow Support

Vela supports TensorFlow 2.1.0.

## Environment

Vela runs on the Linux operating system.

## Prerequisites

The following should be installed prior to the installation of Vela:

* Python >= 3.6
* Pip3
* GNU toolchain (GCC, Binutils and libraries) or alternative C compiler/linker
toolchain

And optionally:

* Pipenv virtual environment tool

## Installation

Before running, the Vela package must be installed along with all its
dependencies.  To do this, first change to the directory that contains this
README.md file. Then use the command:

```bash
pip3 install -U setuptools>=40.1.0
pip3 install .
```

Or, if you use `pipenv`:

```bash
pipenv install .
```

## Running

Vela is run with an input `.tflite` file passed on the command line.  This file
contains the neural network to be compiled. The tool then outputs an optimised
version with a `_vela.tflite` file prefix, along with the performance estimate
(EXPERIMENTAL) CSV files, all to the output directory.

If you use the `pipenv` virtual environment tool then first start by spawning a
shell in the virtual environment.:

```bash
pipenv shell
```

After which running Vela is the same regardless of whether you are in a virtual
environment or not.

Example usage:

1) Compile the network `my_model.tflite`. The optimised version will be output
to `./output/my_network_vela.tflite`.

```bash
vela my_model.tflite
```

1) Compile the network `/path/to/my_model.tflite` and specify the output to go
in the directory `./results_dir/`.

```bash
vela --output-dir ./results_dir /path/to/my_model.tflite
```

1) To specify information about the embedded system's configuration use Vela's
system configuration file. The following command selects the `MySysConfig`
settings that are described in the `sys_cfg_vela.ini` system configuration file.
More details can be found in the next section.

```bash
vela --config sys_cfg_vela.ini --system-config MySysConfig my_model.tflite
```

1) To get a list of all available options:

```bash
vela --help
```

Information about all of Vela's CLI options as well as the system configuration
file format can be found in [Vela Options](OPTIONS.md)

## Testing

Please see [Vela Testing](TESTING.md)

## Contributions

Please see [Vela Contributions](CONTRIBUTIONS.md).

## Security

Please see [Vela Security](SECURITY.md).

## Releases

Please see [Vela Releases](RELEASES.md).

## Resources

Additional useful information:

* [Arm Products: Ethos-U55](https://www.arm.com/products/silicon-ip-cpu/ethos/ethos-u55)
* [Arm Developer: Ethos-U55](https://developer.arm.com/ip-products/processors/machine-learning/ethos-u55)

## License

Vela is licensed under [Apache License 2.0](LICENSE.txt)
