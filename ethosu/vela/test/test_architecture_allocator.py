# SPDX-FileCopyrightText: Copyright 2021 Arm Limited and/or its affiliates <open-source-office@arm.com>
#
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the License); you may
# not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an AS IS BASIS, WITHOUT
# WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Description:
# Unit tests for architecture_allocator.py
import pytest

from ethosu.vela.architecture_allocator import find_block_config
from ethosu.vela.architecture_allocator import try_block_config
from ethosu.vela.architecture_features import Accelerator
from ethosu.vela.architecture_features import Block
from ethosu.vela.architecture_features import create_default_arch
from ethosu.vela.ethos_u55_regs.ethos_u55_regs import resampling_mode
from ethosu.vela.operation import Kernel
from ethosu.vela.operation import NpuBlockType
from ethosu.vela.shape4d import Shape4D

test_data = [
    {
        "block_type": NpuBlockType.ConvolutionDepthWise,
        "kernel": Kernel(25, 5, 2, 2, 1, 1),
        "ofm_shape": Shape4D(2, 11, 22),
        "ifm_shape": Shape4D(27, 25, 22),
    },
    {
        "block_type": NpuBlockType.Pooling,
        "kernel": Kernel(2, 2),
        "ofm_shape": Shape4D(53, 49, 22),
        "ifm_shape": Shape4D(27, 25, 22),
        "ifm_resampling": resampling_mode.NEAREST,
    },
    {
        "block_type": NpuBlockType.ConvolutionMxN,
        "accelerator": Accelerator.Ethos_U55_32,
        "kernel": Kernel(2, 5),
        "ofm_shape": Shape4D(48, 1, 17),
        "ifm_shape": Shape4D(24, 5, 18),
        "ifm_resampling": resampling_mode.TRANSPOSE,
    },
    {
        "block_type": NpuBlockType.ElementWise,
        "ofm_shape": Shape4D(27, 2, 22),
        "ifm_shape": Shape4D(27, 2, 1),
        "ifm2_shape": Shape4D(27, 25, 22),
    },
    {
        "block_type": NpuBlockType.ElementWise,
        "accelerator": Accelerator.Ethos_U55_32,
        "ofm_shape": Shape4D(48, 37, 17),
        "ifm_shape": Shape4D(48, 37, 17),
        "uses_scalar": True,
        "lut_banks": 2,
    },
    {
        "block_type": NpuBlockType.ElementWise,
        "ofm_shape": Shape4D(27, 2, 22),
        "ifm_shape": Shape4D(27, 2, 22),
        "ifm_bits": 16,
    },
]


@pytest.mark.parametrize("test_data", test_data)
def test_allocate(test_data):
    """Tests that find_block_config and try_block_config produce consistent SHRAM layouts"""
    accelerator = test_data.get("accelerator", Accelerator.Ethos_U55_128)
    arch = create_default_arch(accelerator)
    kernel = test_data.get("kernel", Kernel(1, 1))
    block_type = test_data["block_type"]
    ofm_shape = test_data["ofm_shape"]
    ifm_shape = test_data["ifm_shape"]
    ifm2_shape = test_data.get("ifm2_shape")
    uses_scalar = test_data.get("uses_scalar", False)
    ifm_bits = test_data.get("ifm_bits", 8)
    ifm_resampling = test_data.get("ifm_resampling", resampling_mode.NONE)
    scaled = test_data.get("scaled", True)
    lut_banks = test_data.get("lut_banks", 0)
    config = find_block_config(
        arch,
        block_type,
        ofm_shape,
        ifm_shape,
        ifm2_shape,
        uses_scalar=uses_scalar,
        ifm_bits=ifm_bits,
        kernel=kernel,
        lut_banks=lut_banks,
        scaled=scaled,
        ifm_resampling=ifm_resampling,
    )
    assert config is not None
    config2 = try_block_config(
        Block.from_shape(config.ofm_block.as_list()),
        arch,
        block_type,
        ofm_shape,
        ifm_shape,
        ifm2_shape,
        is_partkernel=config.is_partkernel,
        uses_scalar=uses_scalar,
        ifm_bits=ifm_bits,
        kernel=kernel,
        lut_banks=lut_banks,
        scaled=scaled,
        ifm_resampling=ifm_resampling,
    )
    assert config2 is not None
    assert config.layout.ib_end == config2.layout.ib_end
    assert config.layout.ab_start == config2.layout.ab_start
    assert config.layout.ib_start2 == config2.layout.ib_start2
    assert config.acc_type == config2.acc_type
