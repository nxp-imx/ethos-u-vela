<!--
SPDX-FileCopyrightText: Copyright 2020, 2023 Arm Limited and/or its affiliates <open-source-office@arm.com>

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
# Vela External APIs

Vela provides a low-level external API to enable Ethos-U code generation from
other tools.

The external APIs facilitate other tools that require backend compiler
functionality. From herein this functionality is referred to as "the compiler".
The compiler takes as input a network model, and uses the APIs to convert the
model to instructions that can be run on an Ethos-U NPU.

This document contains an overview of the available APIs and the steps that are
needed to use them.

## Overview

All data types and functions to facilitate code generation are located in module
`ethosu.vela.api`. All API function prototypes are fully documented in the
module using docstrings.

### Data types

Class `NpuOperation` is the base class for all operations. It contains a low
level abstraction of an operation that can be run on an Ethos-U NPU. It has the
following sub-classes:

* `NpuDmaOperation`, to perform memory to memory DMA operations, e.g. for moving
  a chunk of memory from DRAM to SRAM
* `NpuConv2DOperation`, for convolution operations like 2-D convolutions,
  transpose convolutions, and also for fully connected operations
* `NpuConvDepthWiseOperation`, for depthwise convolutions
* `NpuPoolingOperation`, for max pooling/average pooling operations
* `NpuElementWiseOperation`, for unary and binary elementwise operations like
  add, subtract, abs, etc.

Class `NpuActivation` is used to represent activation functions which are fused
with the NPU operation, for instance relu or sigmoid.

It is up to the compiler to convert operations of the input model to a list of
these basic NPU operations. Note that the compiler is responsible for all
address planning, i.e. it needs to supply addresses of all input and output
tensors, weights, and biases.

### Finding block configs

For all NPU operations, a block config must be set, which is the unit of work in
which the NPU generates the output. There are restrictions to the size of block
configs. Function `npu_find_block_configs` can be used to find valid block
configs for an operation.

### Encoding of weights and biases

All weights that are used in the NPU operations must be encoded using
function `npu_encode_weights`, and all biases using function `npu_encode_bias`.

### Generating a register command stream

The instructions that are executed by Ethos-U NPUs are called *register
commands*. When the compiler has compressed all weights and biases, converted
all network operations to NPU operations, and allocated all addresses, the
register command stream can be generated using function
`register_command_stream_generator`. This returns a list of 32-bit integers.

In addition to transforming NPU operations to register commands, Vela also:

* selects a suitable block configuration for each instruction (optional)
* adds kernel/DMA wait commands if necessary
* selects the most efficient "block dependency" that controls the NPU pipeline.

### Creating a Driver Payload for the Ethos-U driver

If an Ethos-U driver is used to trigger the execution of the register command
stream, a Driver Payload byte-array must be provided to the driver that
contains:

* a header with driver actions
* the register command stream

This byte array can be generated using function `npu_create_driver_payload`.

### API version

Function `npu_get_api_version` returns the version of the Vela External APIs,
which is maintained separately from Vela's overall version.

## Unit tests

For examples of how to use these APIs, please see the unit tests that are
bundled with Vela's source code, in module `ethosu.vela.test.extapi`.
