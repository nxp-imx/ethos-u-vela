# SPDX-FileCopyrightText: Copyright 2023 Arm Limited and/or its affiliates <open-source-office@arm.com>
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
# Contains implementation of UnidirectionalSequenceLstm graph optimisation.
from enum import Enum
from typing import Tuple

import numpy as np

from .data_type import DataType
from .debug_database import DebugDatabase
from .graph_optimiser_util import create_avg_pool_for_concat
from .operation import ActivationFunction
from .operation import ExplicitScaling
from .operation import Op
from .operation import Operation
from .operation_util import create_add
from .operation_util import create_fullyconnected
from .operation_util import create_fused_activation
from .operation_util import create_mul
from .scaling import elementwise_mul_scale
from .shape4d import Shape4D
from .tensor import QuantizationParameters
from .tensor import Tensor

Q0_15_SCALE = np.float32(2**-15)
"""Q0.15 scale like the reference defines it"""


class Lstm:
    """Lstm graph optimisation.

    Unrolls a UNIDIRECTIONAL_SEQUENCE_LSTM operation into its basic operations.

    Usage:

    unrolled_op = Lstm(op).get_graph()
    """

    class State(Enum):
        """States (variable tensors)"""

        OUTPUT = 18  # Value = tensor index
        CELL = 19  # Value = tensor index

    def __init__(self, op):
        self.op = op

    def get_graph(self) -> Operation:
        """Return the generated graph implementation"""
        self.op.ofm.ops = []
        if self.time_major:
            output_state = self.get_initial_state(Lstm.State.OUTPUT)
            cell_state = self.get_initial_state(Lstm.State.CELL)
            for time in range(self.n_time):
                feature = self.get_feature(time)
                output_state, cell_state = self.lstm_step(feature, output_state, cell_state, time)
                op = self.put_ofm(output_state, time)
        else:
            for batch in range(self.n_batch):
                output_state = self.get_initial_state(Lstm.State.OUTPUT, batch)
                cell_state = self.get_initial_state(Lstm.State.CELL, batch)
                for time in range(self.n_time):
                    feature = self.get_feature(time, batch)
                    output_state, cell_state = self.lstm_step(feature, output_state, cell_state, time, batch)
                    op = self.put_ofm(output_state, time, batch)
        return op

    def get_feature(self, time: int, batch: int = 0) -> Tensor:
        """Get input feature for provided time and batch"""
        feature = self.op.ifm.clone(f"_feature#{batch}.{time}")
        feature.set_all_shapes([self.n_batch if self.time_major else 1, self.n_feature])
        op = Operation(Op.SplitSliceRead, feature.name)
        op.add_input_tensor(self.op.ifm)
        op.set_output_tensor(feature)
        op.set_ifm_ofm_shapes()
        offset = [time, 0, 0] if self.time_major else [batch, time, 0]
        op.read_offsets[0] = Shape4D.from_list(offset, 0)
        op.read_shapes[0] = op.ofm_shapes[0]
        DebugDatabase.add_optimised(self.op, op)
        return feature

    def get_initial_state(self, state_type: State, batch: int = 0) -> Tensor:
        """Get state tensor for provided state type and batch"""
        state = self.state(state_type)
        if self.time_major:
            # For time major just return the 2D state, since all batches
            # are calculated at the same time
            return state
        else:
            # For non time major return one batch of the 2D state
            # by setting the read offset to the provided batch

            # The cloned state tensor will share equivalence id and buffer
            # with the variable state tensor
            n_state = state.shape[-1]
            state_ofm = state.clone(f"_state#{batch}")
            # Set shape to be one batch
            state_ofm.set_all_shapes([1, n_state])
            # Create the op for reading one batch of the state
            # (will be optimised away at a later stage)
            op = Operation(Op.SplitSliceRead, state_ofm.name)
            op.add_input_tensor(state)
            op.set_output_tensor(state_ofm)
            op.set_ifm_ofm_shapes()
            # Set the read offset to the provided batch
            op.read_offsets[0] = Shape4D.from_list([batch, 0], 0)
            # Set the read shape to one batch, see above
            op.read_shapes[0] = op.ofm_shapes[0]
            DebugDatabase.add_optimised(self.op, op)
            return state_ofm

    def get_state(self, op: Operation, batch: int = 0) -> Operation:
        """Setup the correct read offset for reading the state from
        a variable tensor state"""
        if not self.time_major and self.n_batch > 1:
            op.read_offsets[0] = Shape4D.from_list([batch, 0], 0)
            op.read_shapes[0] = Shape4D(op.ifm.shape)
            op.ifm_shapes[0] = Shape4D([self.n_batch, op.ifm.shape[-1]])
        return op

    def put_state(self, op: Operation, state_type: State, batch: int = 0) -> Operation:
        """Save the state for the provided batch by pointing the operations
        ofm to the variable state tensor"""
        # The create op functions always return 4D shape, however the state
        # should have 2D shape for correct operation
        op.ofm.shape = op.ofm.shape[-2:]
        # Get state from type
        state = self.state(state_type)
        # By using the same equivalence_id the backing buffer for the ofm
        # tensor will be the state variable tensor buffer
        op.ofm.equivalence_id = state.equivalence_id
        # Set memory function which will make the tensor be in linear format
        # just as the state variable tensor
        op.memory_function = Op.VariableTensorWrite
        # Set the batch write offset into the state tensor buffer unless
        # time_major mode when all batches are written at once
        if not self.time_major:
            op.write_offset = Shape4D.from_list([batch, 0], 0)
            op.write_shape = Shape4D(op.ofm.shape)
            op.ofm_shapes = [Shape4D(state.shape)]
        DebugDatabase.add_optimised(self.op, op)
        return op

    def put_ofm(self, state: Tensor, time: int, batch: int = 0) -> Operation:
        """Save the output state for the provided batch and time to OFM"""
        name = f"{self.op.ofm.name}#{batch}.{time}"
        offset = Shape4D.from_list([time, 0, 0] if self.time_major else [batch, time, 0], 0)
        op = create_avg_pool_for_concat(self.op, name, state, Shape4D(state.shape), offset)
        # The provided state tensor use the output state tensors buffer, so unless
        # time_major mode we need to set the correct batch read offset
        if not self.time_major:
            op.read_offsets[0] = Shape4D.from_list([batch, 0], 0)
            op.read_shapes[0] = Shape4D(state.shape)
            op.ifm_shapes[0] = Shape4D(self.output_state.shape)
        return op

    def lstm_step(
        self, feature: Tensor, output_state: Tensor, cell_state: Tensor, time: int, batch: int = 0
    ) -> Tuple[Tensor, Tensor]:
        """Generate one step of the LSTM implementation for the provided feature, batch and time"""
        input_gate = self.calculate_gate(
            f"input_gate#{batch}.{time}",
            feature,
            output_state,
            self.input_to_input_weights,
            self.input_bias,
            self.recurrent_to_input_weights,
            None,
            Op.Sigmoid,
            batch,
        )
        forget_gate = self.calculate_gate(
            f"forget_gate#{batch}.{time}",
            feature,
            output_state,
            self.input_to_forget_weights,
            self.forget_bias,
            self.recurrent_to_forget_weights,
            None,
            Op.Sigmoid,
            batch,
        )
        cell_gate = self.calculate_gate(
            f"cell_gate#{batch}.{time}",
            feature,
            output_state,
            self.input_to_cell_weights,
            self.cell_bias,
            self.recurrent_to_cell_weights,
            None,
            Op.Tanh,
            batch,
        )
        cell_state = self.calculate_cell_state(cell_state, input_gate, forget_gate, cell_gate, time, batch)
        output_gate = self.calculate_gate(
            f"output_gate#{batch}.{time}",
            feature,
            output_state,
            self.input_to_output_weights,
            self.output_bias,
            self.recurrent_to_output_weights,
            None,
            Op.Sigmoid,
            batch,
        )
        output_state = self.calculate_output_state(output_gate, cell_state, time, batch)
        return (output_state, cell_state)

    def calculate_gate(
        self,
        name: str,
        input: Tensor,
        state: Tensor,
        input_weights: Tensor,
        input_bias: Tensor,
        recurrent_weights: Tensor,
        recurrent_bias: Tensor,
        activation: Op,
        batch: int = 0,
    ):
        """Generate a gate for the provided input and weights"""
        # Activation( Add( FC(input), FC(output state) ) )
        # Setup fullyconnected quantization
        q_fc = QuantizationParameters()
        q_fc.scale_f32 = np.float32(2**-12)
        q_fc.zero_point = 0
        # Create fullyconnected
        in_fc = create_fullyconnected(f"{name}:{input.name}_fc", input, input_weights, input_bias, q_fc, False)
        re_fc = create_fullyconnected(f"{name}:{state.name}_fc", state, recurrent_weights, recurrent_bias, q_fc, False)
        self.get_state(re_fc, batch)
        # Change fullyconnected ofm data type
        in_fc.ofm.dtype = DataType.int16
        re_fc.ofm.dtype = DataType.int16
        # Setup add quantization
        q_add = q_fc.clone()
        q_add.scale_f32 = Q0_15_SCALE
        # Create add + activation
        add = create_add(f"{name}_add", in_fc.ofm, re_fc.ofm, q_add, ActivationFunction(activation))
        if activation is Op.Sigmoid:
            # For Sigmoid we need to set the activation min/max values to match the possible range
            # in the reference. The values below are the quantized min/max values that the reference
            # can achive for the LUT based Sigmoid/Logistic. (The NPU does however have a larger range
            # due to intermediate higher precision.)
            # The quantized min/max values are divided by the effective output scale 0x3000 (3<<12) used for
            # elementwise operations with fused Tanh/Sigmoid activations (to get correct scaling before the
            # fused activation function). This will yield the dequantized min/max values which are later
            # quantized again by the command stream generator.
            add.activation.max = 32757 / 0x3000
            add.activation.min = 11 / 0x3000
        # Add to debug database
        DebugDatabase.add_optimised(self.op, in_fc)
        DebugDatabase.add_optimised(self.op, re_fc)
        DebugDatabase.add_optimised(self.op, add)
        return add.ofm

    def calculate_cell_state(
        self, cell_state: Tensor, input_gate: Tensor, forget_gate: Tensor, cell_gate: Tensor, time: int, batch: int = 0
    ):
        """Update the cell state from the provided gate output"""
        # Clip( Add( Mul(cell state, forget gate), Mul(input gate, cell gate) ) )
        base_name = f"cell_state#{batch}.{time}"
        # Cell scale
        cell_scale = cell_state.quantization.scale_f32
        # Create mul(cell_state, forget_gate)
        mul_cf = create_mul(f"{base_name}_cf_mul", cell_state, forget_gate, cell_state.quantization)
        self.get_state(mul_cf, batch)
        # Calculate explicit scales to match reference
        multiplier, shift = elementwise_mul_scale(np.double(cell_scale), np.double(Q0_15_SCALE), np.double(cell_scale))
        mul_cf.explicit_scaling = ExplicitScaling(False, [shift], [multiplier])
        # Create mul(cell_gate, input_gate)
        mul_ci = create_mul(f"{base_name}_ci_mul", cell_gate, input_gate, cell_state.quantization)
        # Calculate explicit scales to match reference
        multiplier, shift = elementwise_mul_scale(np.double(Q0_15_SCALE), np.double(Q0_15_SCALE), np.double(cell_scale))
        mul_ci.explicit_scaling = ExplicitScaling(False, [shift], [multiplier])
        # Setup cell clip
        activation = None if self.cell_clip == 0 else ActivationFunction(Op.Clip)
        if activation:
            activation.max = self.cell_clip
            activation.min = -self.cell_clip
        # Create add + activation
        add = create_add(f"{base_name}_add", mul_cf.ofm, mul_ci.ofm, cell_state.quantization, activation)
        add.explicit_scaling = ExplicitScaling(False, shift=[0], multiplier=[1])
        # Save new state
        self.put_state(add, Lstm.State.CELL, batch)
        # Add to debug database
        DebugDatabase.add_optimised(self.op, mul_cf)
        DebugDatabase.add_optimised(self.op, mul_ci)
        DebugDatabase.add_optimised(self.op, add)
        return add.ofm

    def calculate_output_state(self, output_gate: Tensor, cell_state: Tensor, time: int, batch: int):
        """Generate the output state from the provided gate output"""
        # Mul( Tanh(cell state), output gate )
        base_name = f"output_state#{batch}.{time}"
        # Setup tanh quantization
        q_out_tanh = QuantizationParameters()
        q_out_tanh.scale_f32 = Q0_15_SCALE
        q_out_tanh.zero_point = 0
        # Create tanh(cell state)
        tanh = create_fused_activation(Op.Tanh, f"{base_name}_tanh", cell_state, q_out_tanh)
        self.get_state(tanh, batch)
        # Create Mul( Tanh(cell state), output gate )
        q_mul = self.output_state.quantization
        mul = create_mul(f"{base_name}_mul", tanh.ofm, output_gate, q_mul, dtype=self.op.ifm.dtype)
        # Use explicit scaling to match reference, the following line would have been the preferred way
        # mul.forced_output_quantization = self.hidden_quantization
        out_scale = self.hidden_quantization.scale_f32
        multiplier, shift = elementwise_mul_scale(np.double(Q0_15_SCALE), np.double(Q0_15_SCALE), np.double(out_scale))
        mul.explicit_scaling = ExplicitScaling(False, [shift], [multiplier])
        # Save new state
        self.put_state(mul, Lstm.State.OUTPUT, batch)
        # Add to debug database
        DebugDatabase.add_optimised(self.op, tanh)
        DebugDatabase.add_optimised(self.op, mul)
        return mul.ofm

    def state(self, state_type: State) -> Tensor:
        """Get state tensor from type"""
        return self.output_state if state_type == Lstm.State.OUTPUT else self.cell_state

    # Dimensions
    @property
    def n_feature(self) -> int:
        return self.op.ifm.shape[-1]

    @property
    def n_time(self) -> int:
        return self.op.ifm.shape[0 if self.time_major else 1]

    @property
    def n_batch(self) -> int:
        return self.op.ifm.shape[1 if self.time_major else 0]

    # Attributes
    @property
    def cell_clip(self) -> int:
        return self.op.attrs.get("cell_clip", 0)

    @property
    def projection_clip(self) -> int:
        return self.op.attrs.get("proj_clip", 0)

    @property
    def time_major(self) -> bool:
        return self.op.attrs.get("time_major", False)

    # Hidden (intermediate)
    @property
    def hidden_quantization(self) -> QuantizationParameters:
        return self.op.intermediates[4].quantization

    # Input weights
    @property
    def input_to_input_weights(self) -> Tensor:
        return self.op.inputs[1]

    @property
    def input_to_forget_weights(self) -> Tensor:
        return self.op.inputs[2]

    @property
    def input_to_cell_weights(self) -> Tensor:
        return self.op.inputs[3]

    @property
    def input_to_output_weights(self) -> Tensor:
        return self.op.inputs[4]

    # Recurrent weights
    @property
    def recurrent_to_input_weights(self) -> Tensor:
        return self.op.inputs[5]

    @property
    def recurrent_to_forget_weights(self) -> Tensor:
        return self.op.inputs[6]

    @property
    def recurrent_to_cell_weights(self) -> Tensor:
        return self.op.inputs[7]

    @property
    def recurrent_to_output_weights(self) -> Tensor:
        return self.op.inputs[8]

    # Peephole weights
    @property
    def cell_to_input_weights(self) -> Tensor:
        return self.op.inputs[9]

    @property
    def cell_to_forget_weights(self) -> Tensor:
        return self.op.inputs[10]

    @property
    def cell_to_output_weights(self) -> Tensor:
        return self.op.inputs[11]

    # Bias tensors
    @property
    def input_bias(self) -> Tensor:
        return self.op.inputs[12]

    @property
    def forget_bias(self) -> Tensor:
        return self.op.inputs[13]

    @property
    def cell_bias(self) -> Tensor:
        return self.op.inputs[14]

    @property
    def output_bias(self) -> Tensor:
        return self.op.inputs[15]

    # Projection tensors
    @property
    def projection_weights(self) -> Tensor:
        return self.op.inputs[16]

    @property
    def projection_bias(self) -> Tensor:
        return self.op.inputs[17]

    # State tensors (variable)
    @property
    def output_state(self) -> Tensor:
        return self.op.inputs[Lstm.State.OUTPUT.value]

    @property
    def cell_state(self) -> Tensor:
        return self.op.inputs[Lstm.State.CELL.value]
