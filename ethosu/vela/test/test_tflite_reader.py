# SPDX-FileCopyrightText: Copyright 2020-2021, 2023 Arm Limited and/or its affiliates <open-source-office@arm.com>
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
# Contains unit tests for tflite_reader
from unittest.mock import MagicMock
from unittest.mock import patch

import numpy as np
import pytest

from ethosu.vela.operation import Op
from ethosu.vela.tflite.TensorType import TensorType
from ethosu.vela.tflite_mapping import TFLITE_CONV2D_BACKPROP_INDICES
from ethosu.vela.tflite_mapping import TFLITE_IFM_WEIGHTS_BIAS_INDICES
from ethosu.vela.tflite_reader import TFLiteSubgraph


class TestTFLiteSubgraph:

    # Generate some data for testing len1_array_to_scalar
    len1_testdata = [
        (0, None),
        pytest.param(1, None, marks=pytest.mark.xfail),
        ([1, 2, 3], [1, 2, 3]),
        ([10], 10),
        ([], []),
    ]

    @pytest.mark.parametrize("test_input,expected", len1_testdata)
    def test_len1_array_to_scalar(self, test_input, expected):
        output = TFLiteSubgraph.len1_array_to_scalar(test_input)
        assert output == expected

    parse_op_testdata = [
        # op_type, opt_serializer, indices, version, inputs, output, expected
        (Op.FullyConnected, None, TFLITE_IFM_WEIGHTS_BIAS_INDICES, 1, [0, 1, 2], 3, 3),  # FC
        (Op.FullyConnected, None, TFLITE_IFM_WEIGHTS_BIAS_INDICES, 1, [0, 1, -1], 3, 3),  # FC disabled Bias
        (Op.FullyConnected, None, TFLITE_IFM_WEIGHTS_BIAS_INDICES, 5, [0, 1], 3, 3),  # FC no Bias
        (Op.Conv2DBias, None, TFLITE_IFM_WEIGHTS_BIAS_INDICES, 5, [2, 1, 3], 0, 3),  # Conv2D
        (Op.Conv2DBackpropInput, None, TFLITE_CONV2D_BACKPROP_INDICES, 5, [0, 1, 2, 3], 4, 4),  # TransposeConv
        (Op.Conv2DBackpropInput, None, TFLITE_CONV2D_BACKPROP_INDICES, 5, [0, 1, 2], 4, 4),  # TransposeConv no Bias
        pytest.param(
            Op.Conv2DBias, None, TFLITE_IFM_WEIGHTS_BIAS_INDICES, 5, [0, -1, 1], 3, 3, marks=pytest.mark.xfail
        ),  # Conv2D no Weights
    ]

    @pytest.mark.parametrize("op_type, opt_serializer, indices, version, inputs, output, expected", parse_op_testdata)
    def test_parse_operator(self, op_type, opt_serializer, indices, version, inputs, output, expected):
        with patch.object(TFLiteSubgraph, "__init__", lambda self, graph, subraph: None):
            # Mock a TFLiteSubGraph
            sg = TFLiteSubgraph(None, None)
            sg.graph = MagicMock()
            sg.graph.operator_codes = [(op_type, opt_serializer, "", indices, version)]

            # Mock a couple of tensors
            sg.tensors = [MagicMock() for _ in range(5)]
            for i, tens in enumerate(sg.tensors):
                tens.name = "tensor_{}".format(i)
                tens.ops = []

            # Mock op data
            op_data = MagicMock()
            op_data.OpcodeIndex.return_value = 0
            op_data.InputsAsNumpy.return_value = inputs
            op_data.OutputsAsNumpy.return_value = [output]

            sg.parse_operator(0, op_data)

            # Verify the created Operation
            created_op = sg.tensors[output].ops[0]
            assert created_op.type == op_type
            assert len(created_op.inputs) == expected
            assert created_op.outputs[0].name == "tensor_{}".format(output)
            assert inputs[-1] != -1 or not created_op.inputs[-1]

    string_buffer_testdata = [
        (np.array([np.random.randint(256) for _ in range(100)], dtype=np.uint8), [3, 5]),
        (np.array([np.random.randint(256) for _ in range(100)], dtype=np.uint8), [10, 10]),
        (np.array([np.random.randint(256) for _ in range(100)], dtype=np.uint8), []),
        (np.array([], dtype=np.uint8), [30]),
    ]

    @pytest.mark.parametrize("buffer, tens_shape", string_buffer_testdata)
    def test_parse_tensor_with_string_buffer(self, buffer, tens_shape):
        tens_data = MagicMock()
        tens_data.ShapeAsNumpy = MagicMock(return_value=np.array(tens_shape), dtype=np.int32)
        tens_data.Name = MagicMock(return_value=b"test_data")
        tens_data.Type = MagicMock(return_value=TensorType.STRING)
        tens_data.Quantization = MagicMock(return_value=None)
        tens_data.Buffer = MagicMock(return_value=0)

        tfl_sg = MagicMock()
        tfl_sg.Name = MagicMock(return_value=b"test_sg")
        tfl_sg.TensorsLength = MagicMock(return_value=0)
        tfl_sg.OperatorsLength = MagicMock(return_value=0)
        tfl_sg.OutputsAsNumpy = MagicMock(return_value=[])
        tfl_sg.InputsAsNumpy = MagicMock(return_value=[])

        graph = MagicMock()
        graph.buffers = [buffer]

        subgraph = TFLiteSubgraph(graph, tfl_sg)

        tens = subgraph.parse_tensor(tens_data)
        assert np.array_equal(tens.values, buffer)
