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
# Contains unit tests for new performance estimation code
from ethosu.vela import architecture_allocator
from ethosu.vela import architecture_features
from ethosu.vela import npu_performance
from ethosu.vela import operation
from ethosu.vela.architecture_features import resampling_mode
from ethosu.vela.shape4d import Shape4D
from ethosu.vela.shape4d import VolumeIterator
from ethosu.vela.tensor import MemArea


def test_new_performance():
    arch = architecture_features.create_default_arch(architecture_features.Accelerator.Ethos_U55_128)

    query = npu_performance.PerformanceQuery(architecture_features.NpuBlockType.ConvolutionMxN)
    query.ifm_shape = Shape4D(1, 16, 16, 16)
    query.ifm2_shape = Shape4D()
    query.ifm_memory_area = MemArea.Sram
    query.ifm_bits = 8
    query.ofm_shape = Shape4D(1, 16, 16, 1)
    query.ofm_memory_area = MemArea.Sram
    query.ofm_bits = 8
    query.const_shape = Shape4D(1, 1, 1, query.ofm_shape.depth)
    query.const_memory_area = MemArea.OffChipFlash
    query.kernel = operation.Kernel(1, 1, 1, 1, 1, 1, valid_padding=False)
    query.config = architecture_allocator.find_block_config(
        arch,
        architecture_features.NpuBlockType.ConvolutionMxN,
        Shape4D(1, 16, 16, 1),
        query.ifm_shape,
        None,
        False,
        8,
        query.kernel,
        0,
        False,
        resampling_mode.NONE,
    )

    print("For block Config = {}".format(query.config))

    # -s to display output
    for sub_shape in [Shape4D(1, 4, 8, 16), Shape4D(1, 8, 8, 16), Shape4D(1, 8, 16, 16), query.ofm_shape]:
        print("\n-- Subshape = {}".format(sub_shape))
        iterator = VolumeIterator(query.ofm_shape, sub_shape)
        a = npu_performance.ElementAccess()
        c = npu_performance.CycleCost()
        for pos, shape in iterator:
            print("\tpos = {} shape = {}".format(pos, shape))
            ta, tc = npu_performance.measure_performance_cost(
                arch, operation.Op.Conv2D, operation.Op.Relu, query, pos, shape
            )
            a += ta
            c += tc
            print("\t\taccess: {}".format(ta))
            print("\t\tcycles: {}".format(tc))
        print("\tAccess: {}".format(a))
        print("\tCycles: {}".format(c))
        assert c.op_macs == 4096

    assert True  # Any successful result is okay
