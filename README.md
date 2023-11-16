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
# Vela

This tool is used to compile a
[TensorFlow Lite for Microcontrollers](https://www.tensorflow.org/lite/microcontrollers)
neural network model into an optimised version that can run on an embedded
system containing an
[Arm Ethos-U NPU](https://www.arm.com/products/silicon-ip-cpu).

In order to be accelerated by the Ethos-U NPU the network operators must be
quantised to either 8-bit (unsigned or signed) or 16-bit (signed).

The optimised model will contain TensorFlow Lite Custom operators for those
parts of the model that can be accelerated by the Ethos-U NPU.  Parts of the
model that cannot be accelerated are left unchanged and will instead run on the
Cortex-M series CPU using an appropriate kernel (such as the
[Arm](https://www.arm.com) optimised
[CMSIS-NN](https://github.com/ARM-software/CMSIS_5/tree/develop/CMSIS/NN)
kernels).

After compilation the optimised model can only be run on an Ethos-U NPU
embedded system.

The tool will also generate performance estimates (EXPERIMENTAL) for the
compiled model.

The tool has limited functionality for compiling a
[TOSA](https://git.mlplatform.org/tosa/specification.git/) neural network
(EXPERIMENTAL).

## TensorFlow Support
Vela is tested by comparing the bit exact numerical behaviour of the Ethos-U
optimised operators against that of the corresponding TensorFlow Lite reference
kernels (or TensorFlow Lite for Microcontrollers reference kernels in the case
of the UNIDIRECTIONAL_SEQUENCE_LSTM operator).  The following list indicates
which version is used for comparison:

* Vela 3.10.0 to current supports TensorFlow 2.14
* Vela 3.9.0 supports TensorFlow 2.12
* Vela 3.8.0 supports TensorFlow 2.11
* Vela 3.6.0 to 3.7.0 supports TensorFlow 2.10
* Vela 3.5.0 supports TensorFlow 2.9
* Vela 3.4.0 supports TensorFlow 2.8
* Vela 3.3.0 supports TensorFlow 2.7
* Vela 3.1.0 to 3.2.0 supports TensorFlow 2.5
* Vela 2.1.0 to 3.0.0 supports TensorFlow 2.4
* Vela 2.0.0 to 2.0.1 supports TensorFlow 2.3
* Vela 0.1.0 to 1.2.0 supports TensorFlow 2.1

## Python Version Support
The majority of Vela's testing is done using a single version of Python, as
indicated by the first version in the list below.  However, some additional
testing is also performed across a range of newer versions starting at the
minimum version (pyproject.toml:project.requires-python) indicated in the
brackets:

* Vela 3.10.0 to current supports Python 3.10 (3.9)
* Vela 3.9.0 supports Python 3.10 (3.8)
* Vela 3.8.0 supports Python 3.9 (3.8)
* Vela 3.4.0 to 3.7.0 supports Python 3.7 (3.8)
* Vela 3.3.0 supports Python 3.8 (3.7)
* Vela 0.1.0 to 3.2.0 supports Python 3.6 (3.7)

## Environment

Vela runs on Linux and Microsoft Windows 10 operating systems.

## Prerequisites

The following should be installed prior to the installation of Vela:

* Python 3.10 or compatible
   - Development version containing the Python/C API header files
   - e.g. `apt install python3.10-dev` or `yum install python310-devel`
* Pip3
* C99 capable compiler and associated toolchain
    - For Linux operating systems, a GNU toolchain is recommended.
    - For Microsoft Windows 10, the Microsoft Visual C++ 14.2 Build Tools are recommended.
      See <https://wiki.python.org/moin/WindowsCompilers>

## Installation

Vela is available to install as a package from
[PyPi](https://pypi.org/project/ethos-u-vela/), or as
source code from
[ML Platform](https://review.mlplatform.org/plugins/gitiles/ml/ethos-u/ethos-u-vela).
Both methods will automatically install all the required dependencies.

### PyPi

Install Vela from PyPi using the following command:

```bash
pip3 install ethos-u-vela
```

### ML Platform

First obtain the source code by either downloading the desired TGZ file from:  
<https://review.mlplatform.org/plugins/gitiles/ml/ethos-u/ethos-u-vela>

Or by cloning the git repository:

```bash
git clone https://review.mlplatform.org/ml/ethos-u/ethos-u-vela.git
```

Once you have the source code, Vela can be installed using the following
command from the root directory of the repository:

```bash
pip3 install .
```

#### Advanced Installation for Developers

If you plan to modify the Vela codebase then it is recommended to install Vela
as an editable package to avoid the need to re-install after every modification.
This is done by adding the `-e` option to the install command like so:

```bash
pip3 install -e .[dev]
```

If you plan to contribute to the Vela project (highly encouraged!) then it is
recommended to install Vela with the development dependencies (see
[Vela Testing](TESTING.md) for more details).

## Running

Vela is run with an input `.tflite` or `.tosa` (EXPERIMENTAL) file passed on the
command line. This file contains the neural network to be compiled. The tool then
outputs an optimised `.tflite` file with a `_vela` suffix in the file name, along
with performance estimate (EXPERIMENTAL) CSV files, all to the output directory.
It also prints a performance estimation summary back to the console, see
[Vela Performance Estimation Summary](PERFORMANCE.md).

Example usage:

1) Compile the network `my_model.tflite`.  The optimised version will be output
to `./output/my_network_vela.tflite`.

```bash
vela my_model.tflite
```

2) Compile the network `/path/to/my_model.tflite` and specify the output to go
in the directory `./results_dir/`.

```bash
vela --output-dir ./results_dir /path/to/my_model.tflite
```

3) Compile a network targeting a particular Ethos-U NPU.  The following command
selects an Ethos-U65 NPU accelerator configured with 512 MAC units.

```bash
vela --accelerator-config ethos-u65-512 my_model.tflite
```
4) Compile a network while minimizing peak SRAM usage, prioritising lower SRAM
usage over runtime performance.

```bash
vela --optimise Size my_model.tflite
```

5) Compile a network to have maximum performance, i.e. the fastest inference time.
This prioritises a higher runtime performance over a lower peak SRAM usage.

```bash
vela --optimise Performance my_model.tflite
```

6) Compile a network while optimising for the fastest inference time possible,
with an upper bound for the SRAM usage. The memory limit is set in bytes, i.e.
run the following example if one requires a limit of 300KB.

```bash
vela --optimise Performance --arena-cache-size 300000 my_model.tflite
```

7) Compile a network using a particular embedded system configuration defined in
Vela's configuration file.  The following command selects the `My_Sys_Config`
system configuration along with the `My_Mem_Mode` memory mode from the `vela.ini`
configuration file located in the config_files directory.

```bash
vela --config Arm/vela.ini --system-config My_Sys_Config --memory-mode My_Mem_Mode my_model.tflite
```

8) To get a list of all available configuration files in the config_files directory:

```bash
vela --list-config-files
```

9) To get a list of all available options (see CLI Options section below):

```bash
vela --help
```

## Warnings

When running the Vela compiler it may report a number of warning messages to the
console. These should all be thoroughly reviewed as they will indicate decisions
that the compiler has made in order to create the optimised network.

## Example Networks

Some example networks that contain quantised operators which can be compiled by
Vela to run on the Ethos-U NPU can be found at:
<https://tfhub.dev/s?deployment-format=lite&q=quantized>

## Known Issues

### 1. NumPy C API version change

Once ethos-u-vela is installed, the user might want to install a different NumPy
version that is still within the dependency constraints defined in pyproject.toml.

In some scenarios, doing so might prevent ethos-u-vela from functioning as
expected due to incompatibilities between the installed NumPy C headers used in
the mlw_codec and the current version of NumPy.

**Example scenario:**

In the ethos-u-vela source directory, run:

```bash
virtualenv -p 3.10 venv
. venv/bin/activate
pip install ethos-u-vela
```

Next, install a different NumPy version (e.g. 1.21.3)

```bash
pip install numpy==1.21.3 --force
```

Finally, run ethos-u-vela. You might get an error similar to this:

```
ImportError: NumPy C API version mismatch
(Build-time version: 0x10, Run-time version: 0xe)
This is a known issue most likely caused by a change in the API version in
NumPy after installing ethos-u-vela.
```

#### Solution

In order for ethos-u-vela to work with an older version of NumPy that uses
different C APIs, you will need to install the desired NumPy version first, and
then build ethos-u-vela with that specific NumPy version:

1) Uninstall ethos-u-vela and install the desired version of NumPy
   ```
   pip uninstall ethos-u-vela
   pip install numpy==1.21.3 --force
   ```

2) Install required build dependencies
   ```
   pip install "setuptools_scm[toml]<6" wheel
   ```

3) Install ethos-u-vela without build isolation. Not using build isolation
   ensures that the correct version of NumPy is used when copying the C headers
   in mlw_codec during the build process.
   ```
   pip install ethos-u-vela --no-build-isolation --no-cache-dir
   ```

## APIs

Please see [Vela External APIs](API.md).

## Bug Reporting

Please see [Vela Community Bug Reporting](BUGS.md) for a description of how to
report bugs.

## Contributions

Please see [Vela Contributions](CONTRIBUTIONS.md).

## Debug Database

Please see [Vela Debug Database](DEBUG_DB.md).

## Inclusive language commitment

This product conforms to Arm’s inclusive language policy and, to the best of
our knowledge, does not contain any non-inclusive language. If you find
something that concerns you, email terms@arm.com.

## Options

Please see [Vela CLI Options](OPTIONS.md).  This includes a description of the
system configuration file format.

## Performance

Please see [Vela Performance Estimation Summary](PERFORMANCE.md).

## Releases

Please see [Vela Releases](RELEASES.md).

## Resources

Additional useful information:

* [Arm Products: Ethos-U55 NPU](https://www.arm.com/products/silicon-ip-cpu/ethos/ethos-u55)
* [Arm Products: Ethos-U65 NPU](https://www.arm.com/products/silicon-ip-cpu/ethos/ethos-u65)
* [Arm Developer: Ethos-U55 NPU](https://developer.arm.com/ip-products/processors/machine-learning/arm-ethos-u/ethos-u55)
* [Arm Developer: Ethos-U65 NPU](https://developer.arm.com/ip-products/processors/machine-learning/arm-ethos-u/ethos-u65)

## Security

Please see [Vela Security](SECURITY.md).

## Supported Operators

Please see [Vela Supported Operators](SUPPORTED_OPS.md) for the list of
operators supported in this release.

## Testing

Please see [Vela Testing](TESTING.md).

## License

Vela is licensed under [Apache License 2.0](LICENSE.txt).