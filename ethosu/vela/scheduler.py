# SPDX-FileCopyrightText: Copyright 2020-2023 Arm Limited and/or its affiliates <open-source-office@arm.com>
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
# The scheduler creates and searches for an optimal plan for the network, selecting block configurations and
# subdivisions for the Operators
# For Class name forward references for the type annotations. (see PEP 563).
from __future__ import annotations

import copy
from collections import namedtuple
from enum import auto
from enum import IntEnum
from typing import Any
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple
from typing import TYPE_CHECKING

from .utils import progress_print

# Import needed for Type annotations. Only import for Type checking to avoid run-time errors due to cyclic import.
if TYPE_CHECKING:
    from .npu_performance import CycleCost

import numpy as np

from . import live_range
from . import npu_performance
from . import tensor_allocation
from . import weight_compressor
from .architecture_allocator import ArchitectureBlockConfig
from .architecture_allocator import find_block_config
from .architecture_allocator import get_ifm_area_required
from .architecture_allocator import to_upscale
from .architecture_allocator import is_nearest
from .architecture_features import ArchitectureFeatures
from .architecture_features import Block
from .cascade_builder import CascadeBuilder
from .cascade_builder import CascadeInfo
from .data_type import DataType
from .nn_graph import CascadedPass
from .nn_graph import Graph
from .nn_graph import Pass
from .nn_graph import PassPlacement
from .nn_graph import SchedulingStrategy
from .nn_graph import Subgraph
from .live_range import ofm_can_reuse_ifm
from .numeric_util import round_down
from .numeric_util import round_up
from .operation import Op
from .shape4d import Shape4D
from .tensor import MemArea
from .tensor import MemType
from .tensor import Tensor
from .tensor import TensorFormat
from .tensor import TensorPurpose
from .tensor import TensorSubPurpose
from .weight_compressor import NpuWeightTensor


def shape_for_format(shape: Shape4D, tensor_format: TensorFormat) -> Shape4D:
    if tensor_format == TensorFormat.NHCWB16:
        return shape.with_depth(round_up(shape.depth, 16))

    return shape


class OptimizationStrategy(IntEnum):
    """Enum defining the different optimization strategies for the Scheduler"""

    Size = auto()
    Performance = auto()

    def __str__(self):
        return self.name


class SchedulerOpInfo:
    """Contains metadata about a SchedulerOperation that is unique to one Schedule"""

    def __init__(
        self,
        block_config: ArchitectureBlockConfig,
        weights_size: int,
        stripe_input: Shape4D,
        stripe_input2: Optional[Shape4D],
        stripe: Shape4D,
    ):
        self.block_config = block_config
        self.weights_size = weights_size
        self.stripe_input = stripe_input
        self.stripe_input2 = stripe_input2
        self.stripe = stripe
        self.cascade = 0  # Assigned by CascadeBuilder. 0 means not part of a cascade
        self.time_index = None  # Set by update_op_memory_snapshot
        self.ofm_depth_slices: List[int] = [0, stripe.depth]
        self.npu_weights_tensor: Optional[NpuWeightTensor] = None
        self.npu_scales_tensor: Optional[NpuWeightTensor] = None
        self.buffered_weight_tensors: List[Tensor] = []
        self.cycles: Optional[CycleCost] = None
        self.slack_buffering_cycles = 0
        self.slack_buffering_memory = 0
        self.full_weight_transfer_cycles = 0

    def copy(self):
        res = SchedulerOpInfo(
            self.block_config,
            self.weights_size,
            self.stripe_input,
            self.stripe_input2,
            self.stripe,
        )
        res.cascade = self.cascade
        return res

    def __str__(self):
        res = f"\t\tBlock Config = {self.block_config}\n"
        res += f"\t\tOFM Block = {self.block_config.ofm_block}\n"
        res += f"\t\tIFM Stripe   = {self.stripe_input}\n"
        res += f"\t\tIFM2 Stripe  = {self.stripe_input2}\n"
        res += f"\t\tOFM Stripe   = {self.stripe}\n"
        res += f"\t\tEncoded Weights = {self.npu_weights_tensor and len(self.npu_weights_tensor.buffer)} bytes\n"
        for idx, tens in enumerate(self.buffered_weight_tensors):
            res += f"\t\tWeight buffer{idx + 1} = {tens.storage_size()} bytes\n"
        res += f"\t\tDepth slices = {self.ofm_depth_slices}\n"
        res += f"\t\tAssigned Cascade = {self.cascade}"
        return res


class SchedulerOptions:
    """Contains options for the Scheduler"""

    def __init__(
        self,
        optimization_strategy,
        sram_target,
        verbose_schedule,
        verbose_progress=False,
    ):
        self.optimization_strategy = optimization_strategy
        self.optimization_sram_limit = sram_target
        self.verbose_schedule = verbose_schedule
        self.verbose_progress = verbose_progress

    def __str__(self) -> str:
        return f"{type(self).__name__}: {str(self.__dict__)}"

    __repr__ = __str__


class SchedulerTensor:
    def __init__(self, shape, dt, mem_area, _format):
        self.dtype = dt
        self.mem_area = mem_area
        self.shape = shape
        self.format = _format
        self.connection = None


class SchedulerOperation:
    """Scheduler internal representation of 'Operation'
    This class can be seen as a node within the Scheduler Graph representation
    """

    def __init__(self, ps: Pass, arch: ArchitectureFeatures, nng: Graph):
        self.arch = arch
        self.parent_ps = ps
        self.parent_op = ps.primary_op
        self.name = ps.primary_op.name
        self.op_type = ps.primary_op.type
        self.activation = ps.primary_op.activation
        self.kernel = ps.primary_op.kernel
        self.resampling_mode = ps.primary_op.ifm_resampling_mode
        self.reversed_operands = False
        self.uses_scalar = ps.primary_op.ifm2 is not None and (
            ps.primary_op.ifm.shape == [] or ps.primary_op.ifm2.shape == []
        )

        self.ifm_ublock = arch.ifm_ublock

        self.ifm = SchedulerTensor(
            ps.ifm_shapes[0],
            ps.ifm_tensor.dtype,
            ps.ifm_tensor.mem_area,
            ps.ifm_tensor.format,
        )

        self.ifm2 = None
        if ps.ifm2_tensor:
            self.ifm2 = SchedulerTensor(
                ps.ifm_shapes[1],
                ps.ifm2_tensor.dtype,
                ps.ifm2_tensor.mem_area,
                ps.ifm2_tensor.format,
            )

        self.ofm = SchedulerTensor(
            ps.ofm_shapes[0],
            ps.ofm_tensor.dtype,
            ps.ofm_tensor.mem_area,
            ps.ofm_tensor.format,
        )

        # LUT must be placed in shram area. The copy is done by DMA
        # generated by the high level command stream generator.
        for idx, tens in enumerate(self.parent_op.inputs):
            if tens.purpose == TensorPurpose.LUT:
                new_tens = tens.clone_into_shram(self.arch)
                new_tens.consumer_list.append(self.parent_op)
                self.parent_op.inputs[idx] = new_tens

        # Input volume width and height required to produce the smallest possible stripe
        self.min_stripe_input_w, self.min_stripe_input_h = self._calculate_min_stripe_input()

        # Flags that marks whether this SchedulerOperation requires full IFM/OFM
        self.requires_full_ifm = False
        self.requires_full_ifm2 = False
        self.requires_full_ofm = False

        self.evicted_fms_size = 0

        self.index = 0

        # Perform an IFM swap for certain binary elementwise operators
        # in order to enable cascading, if the SchedOp conforms to
        # Elementwise cascading rules.
        # The non-constant/non-scalar/non-broadcast IFM should be the primary input
        if self.op_type.is_binary_elementwise_op():
            ifm = self.parent_op.ifm
            ifm2 = self.parent_op.ifm2
            ofm = self.parent_op.ofm

            ifm_can_swap = ifm.is_const or ifm.is_scalar
            ifm2_can_be_primary = not (ifm2.is_const or ifm2.is_scalar or ifm2.is_broadcast(ofm))

            if ifm_can_swap and ifm2_can_be_primary:
                # IFM2 is the primary input
                self.reversed_operands = True
                self.ifm, self.ifm2 = self.ifm2, self.ifm

                self.parent_ps.ifm_shapes = self.parent_ps.ifm_shapes[::-1]
                self.parent_ps.inputs = self.parent_ps.inputs[::-1]
                self.parent_ps.ifm_tensor, self.parent_ps.ifm2_tensor = (
                    self.parent_ps.ifm2_tensor,
                    self.parent_ps.ifm_tensor,
                )

    @property
    def ofm_write_shape(self):
        if self.ofm:
            ofm_write_shape = self.parent_op.write_shape
            return ofm_write_shape if ofm_write_shape else self.ofm.shape
        return None

    @property
    def ifm_read_shape(self):
        if self.ifm:
            ifm_read_shape = self.parent_op.read_shapes[1] if self.reversed_operands else self.parent_op.read_shapes[0]
            return ifm_read_shape if ifm_read_shape else self.ifm.shape
        return None

    @property
    def ifm2_read_shape(self):
        if self.ifm2:
            ifm2_read_shape = self.parent_op.read_shapes[0] if self.reversed_operands else self.parent_op.read_shapes[1]
            return ifm2_read_shape if ifm2_read_shape else self.ifm2.shape
        return None

    def add_ifm_connection(self, conn: "Connection"):
        """Add input connection to another SchedulerOperation or Subgraph Input"""
        conn.consumers.append(self)
        self.ifm.connection = conn

    def add_ifm2_connection(self, conn: "Connection"):
        """Add input connection to another SchedulerOperation or Subgraph Input"""
        if self.ifm2:
            conn.consumers.append(self)
            self.ifm2.connection = conn
        else:
            assert False, f"Trying to set an IFM2 Connection to {self} which has no IFM2"

    def add_ofm_connection(self, conn: "Connection"):
        """Add output connection to another SchedulerOperation or Subgraph Output"""
        conn.producers.append(self)
        self.ofm.connection = conn

    def get_dependants(self):
        """Returns a list of the Ops that depend on this Operation's OFM"""
        return self.ofm.connection.consumers

    def ifm_size_in_bytes(self) -> int:
        """Returns size of the IFM in bytes"""
        ifm_storage_shape = shape_for_format(self.ifm.shape, self.ifm.format)
        return round_up(ifm_storage_shape.elements() * self.ifm.dtype.size_in_bytes(), Tensor.AllocationQuantum)

    def ifm2_size_in_bytes(self) -> int:
        """Returns size of the IFM2 in bytes"""
        if self.ifm2:
            ifm2_storage_shape = shape_for_format(self.ifm2.shape, self.ifm2.format)
            return round_up(ifm2_storage_shape.elements() * self.ifm2.dtype.size_in_bytes(), Tensor.AllocationQuantum)

        return 0

    def ofm_size_in_bytes(self) -> int:
        """Returns size of the OFM in bytes"""
        ofm_storage_shape = shape_for_format(self.ofm.shape, self.ofm.format)
        return round_up(ofm_storage_shape.elements() * self.ofm.dtype.size_in_bytes(), Tensor.AllocationQuantum)

    def create_scheduler_info(self, nng: Graph, stripe: Shape4D) -> SchedulerOpInfo:
        """Returns schedule info about this SchedulerOperation based on how many ofm elements it should produce"""
        ifm_shape = self.ifm.shape
        ifm2_shape = self.ifm2.shape if self.ifm2 is not None else None
        ofm_shape = stripe

        if ofm_shape != self.ofm.shape:
            # Striped Op - Need to calculate stripe input volume
            stripe_input_w, stripe_input_h = self._get_stripe_input_requirement(stripe)
            # Ensure stripe input volume is within the full IFM volume
            stripe_input_h = min(stripe_input_h, self.ifm.shape.height)
            stripe_input_w = min(stripe_input_w, self.ifm.shape.width)
            ifm_shape = ifm_shape.with_hw(stripe_input_h, stripe_input_w)

            if self.ifm2:
                stripe_input2_h = min(stripe_input_h, self.ifm2.shape.height)
                stripe_input2_w = min(stripe_input_w, self.ifm2.shape.width)
                ifm2_shape = ifm2_shape.with_hw(stripe_input2_h, stripe_input2_w)

        block_config = self._get_block_config(ifm_shape, ifm2_shape, self.uses_scalar, ofm_shape)

        scheduler_op_info = SchedulerOpInfo(block_config, 0, ifm_shape, ifm2_shape, ofm_shape)
        if self.parent_op.weights:
            # Default full-depth weight encoding with no buffering
            (
                scheduler_op_info.npu_weights_tensor,
                scheduler_op_info.npu_scales_tensor,
            ) = weight_compressor.encode_weight_and_scale_tensor(
                self.arch,
                self.parent_op,
                self.parent_op.weights,
                self.parent_op.bias,
                self.kernel,
                block_config,
                [0, self.ofm.shape.depth],
            )

        self.parent_ps.block_config = block_config.old_style_representation()
        return scheduler_op_info

    def _get_stripe_input_requirement(self, stripe_shape: Shape4D) -> Tuple[int, int]:
        """Returns the amount of IFM required to produce the stripe with shape:'stripe_shape'"""
        ofm_shape_to_produce = Block.from_shape(stripe_shape.as_list())

        return get_ifm_area_required(ofm_shape_to_produce, self.kernel, self.resampling_mode)

    def _calculate_min_stripe_input(self) -> Tuple[int, int]:
        # Calculate the input volume required height and width for the smallest possible stripe (h,w = 1,1)
        min_stripe = self.ofm.shape.with_hw(1, 1)
        return self._get_stripe_input_requirement(min_stripe)

    def _get_block_config(
        self, ifm_shape: Shape4D, ifm2_shape: Optional[Shape4D], uses_scalar: bool, ofm_shape: Shape4D
    ) -> Optional[ArchitectureBlockConfig]:
        # Returns a block config and SHRAM layout
        lut_banks = 2 if self.parent_op.activation_lut else 0
        return find_block_config(
            self.arch,
            self.op_type.npu_block_type,
            ofm_shape,
            ifm_shape,
            ifm2_shape,
            uses_scalar,
            self.ifm.dtype.size_in_bits(),
            self.kernel,
            lut_banks,
            self.parent_op.has_scaling(),
            self.resampling_mode,
        )


class Connection:
    """Scheduler internal representation of a Tensor that connects two SchedulerOperations
    This class can be seen as an edge within the Scheduler Graph representation
    """

    def __init__(self, tensor: Tensor):
        self.parent_tens = tensor

        # SchedulerOperation relationships
        self.producers: List[SchedulerOperation] = []
        self.consumers: List[SchedulerOperation] = []

    def __str__(self):
        return f"<Connection {self.parent_tens.name}>"

    __repr__ = __str__


class Schedule:
    """Class that contains a solution of how to schedule an NPU subgraph and its cost"""

    def __init__(self, sg: Subgraph, label: str):
        self.sg = sg
        self.label = label
        self.cost_map: Dict[SchedulerOperation, SchedulerOpInfo] = {}
        self.cascades: Dict[int, CascadeInfo] = {}
        self.fast_storage_peak_usage = 0
        self.memory_snapshot: Optional[List[int]] = None

    @property
    def name(self):
        return f"{self.sg.name}_{self.label}"


class Scheduler:
    """Main class of the Vela Scheduling"""

    def __init__(self, nng: Graph, sg: Subgraph, arch: ArchitectureFeatures, options: SchedulerOptions):
        self.nng = nng
        self.sg = sg
        self.arch = arch
        self.sched_ops: List[SchedulerOperation] = []
        self.max_schedule: Optional[Schedule] = None
        self.scheduler_options = options

        # sram limit can be changed when scheduling for Size
        self.sram_limit = options.optimization_sram_limit

        self.scratched_fms: Dict[Tensor, Any] = {}
        self.evicted_fms: List[live_range.LiveRange] = []

    def avoid_nhcwb16_for_ofm(self, tens, ps, arch):
        """For elementwise ops when ifm is in nhwc format and not brick format aligned (16),
        then if the ofm can overwrite the ifm it is better to enforce ofm format to nhwc in
        order to reduce memory transactions"""

        op = ps.primary_op
        if not op.type.is_elementwise_op():
            return False

        depth = op.ofm_shapes[0][-1]
        if (depth % 16) == 0:
            return False

        # Check if overwriting the inputs can be allowed
        OpShapeTens = namedtuple("OpShapeTens", ["op_shape", "tens"])
        outp = OpShapeTens(op.ofm_shapes[0], op.ofm)
        inps = []
        if op.ifm is not None:
            inps.append(OpShapeTens(op.ifm_shapes[0], op.ifm))
        if op.ifm2 is not None:
            inps.append(OpShapeTens(op.ifm_shapes[1], op.ifm2))

        # Find an input tensor that can be overwritten by the output
        for inp in inps:
            if (
                # check op input and output shapes allow overlapping
                inp.op_shape == outp.op_shape
                # check input tensor is valid
                and inp.tens is not None
                and inp.tens.shape != []
                # check input and output tensors are compatible
                and inp.tens.format == outp.tens.format
                and inp.tens.dtype == outp.tens.dtype
            ):
                if inp.tens.format == TensorFormat.NHWC:
                    return True

        return False

    def create_scheduler_representation(self, arch: ArchitectureFeatures):
        """Creates a Scheduler Graph representation"""
        # Temporary dict for creating connections between the Operations
        connections: Dict[Tensor, Connection] = {}
        # Memory required for the largest FeatureMap that has to be full
        min_memory_req = 0
        for ps in self.sg.passes:
            if ps.primary_op:
                # Set tensor format to NHCWB16 for output FeatureMaps, if possible
                for output in ps.outputs:
                    if output in self.sg.output_tensors or output.purpose != TensorPurpose.FeatureMap:
                        continue

                    if output.use_linear_format:
                        continue

                    if self.avoid_nhcwb16_for_ofm(output, ps, arch):
                        output.force_linear_format = True
                        continue

                    output.set_format(TensorFormat.NHCWB16, arch)

                # Create SchedulerOperations
                op = SchedulerOperation(ps, arch, self.nng)
                op.index = len(self.sched_ops)

                # Make connections
                if ps.ifm_tensor not in connections:
                    connections[ps.ifm_tensor] = Connection(ps.ifm_tensor)
                if ps.ifm2_tensor and ps.ifm2_tensor not in connections:
                    connections[ps.ifm2_tensor] = Connection(ps.ifm2_tensor)
                if ps.ofm_tensor not in connections:
                    connections[ps.ofm_tensor] = Connection(ps.ofm_tensor)

                op.add_ifm_connection(connections[ps.ifm_tensor])
                if ps.ifm2_tensor:
                    op.add_ifm2_connection(connections[ps.ifm2_tensor])
                op.add_ofm_connection(connections[ps.ofm_tensor])

                # Set requirements on the ifm/ofm buffers
                self.sched_ops.append(op)
                if ps.ifm_tensor in self.sg.input_tensors:
                    # This Op consumes a subgraph input
                    op.requires_full_ifm = True
                if ps.ifm2_tensor and ps.ifm2_tensor in self.sg.input_tensors:
                    # This Op consumes a subgraph input
                    op.requires_full_ifm2 = True
                if ps.ofm_tensor in self.sg.output_tensors:
                    # This Op produces a subgraph output
                    op.requires_full_ofm = True
                if ps.ifm_tensor.use_linear_format:
                    op.requires_full_ifm = True
                if ps.ifm2_tensor and ps.ifm2_tensor.use_linear_format:
                    op.requires_full_ifm2 = True
                if ps.ofm_tensor.use_linear_format or ps.primary_op.memory_function == Op.ConcatSliceWrite:
                    op.requires_full_ofm = True
                if len(ps.primary_op.outputs) > 1 or len(ps.primary_op.outputs[0].consumer_list) > 1:
                    # Op has multiple outputs or consumers - requires full OFM
                    op.requires_full_ofm = True

                # Check memory requirements if this Op requires any full FeatureMaps
                op_memory_req = 0
                if op.requires_full_ifm:
                    op_memory_req += op.ifm_size_in_bytes()
                if op.requires_full_ifm2:
                    op_memory_req += op.ifm2_size_in_bytes()
                if op.requires_full_ofm:
                    op_memory_req += op.ofm_size_in_bytes()

                min_memory_req = max(op_memory_req, min_memory_req)

        # Theoretical minimum required memory - used to guide the cascade building
        self.min_memory_req = min_memory_req

    def create_initial_schedule(self) -> Schedule:
        """Creates an initial schedule with no cascading or buffering of any kind"""
        schedule = Schedule(self.sg, "MAX")
        verbose_progress = self.scheduler_options.verbose_progress
        for index, op in enumerate(self.sched_ops):
            progress_print(verbose_progress, "Processing SchedulerOp", index, self.sched_ops)
            cost = op.create_scheduler_info(self.nng, op.ofm.shape)
            cost.cycles = self.estimate_op_performance(op, cost.block_config, op.ofm.shape.depth)
            schedule.cost_map[op] = cost

        return schedule

    def update_op_memory_snapshot(self, schedule: Schedule):
        memories_list = [(self.arch.fast_storage_mem_area, set((MemType.Scratch, MemType.Scratch_fast)))]
        verbose_progress = self.scheduler_options.verbose_progress
        progress_print(verbose_progress, "")
        # Collect live ranges from tensors
        lr_graph = live_range.LiveRangeGraph()
        for mem_area, mem_type_set in memories_list:
            live_range.extract_live_ranges_from_schedule(self.sg, mem_area, mem_type_set, lr_graph, verbose_progress)

        # Populate time-array with memory used by live ranges
        temporal_usage = lr_graph.get_temporal_memory_usage(self.arch.fast_storage_mem_area)
        schedule.memory_snapshot = temporal_usage

        # Set the peak memory usage
        schedule.fast_storage_peak_usage = max(temporal_usage, default=0)

    def estimate_op_performance(self, op: SchedulerOperation, block_config, ofm_depth):
        query = npu_performance.PerformanceQuery(op.op_type.npu_block_type)
        query.ifm_shape = op.ifm_read_shape
        query.ifm_memory_area = op.ifm.connection.parent_tens.mem_area
        query.ifm_bits = op.ifm.dtype.size_in_bits()
        query.ifm_format = op.ifm.format
        query.ifm2_shape = op.ifm2_read_shape
        query.ifm2_memory_area = op.ifm2 and op.ifm2.connection.parent_tens.mem_area
        query.ifm2_bits = op.ifm2 and op.ifm2.dtype.size_in_bits()
        query.ifm2_format = op.ifm2 and op.ifm2.format
        query.ofm_shape = op.ofm_write_shape.with_depth(ofm_depth)
        query.ofm_memory_area = op.ofm.connection.parent_tens.mem_area
        query.ofm_bits = op.ofm.dtype.size_in_bits()
        query.ofm_format = op.ofm.format
        if op.parent_op.bias:
            query.const_shape = Shape4D(1, 1, 1, op.ofm.shape.depth)
            query.const_memory_area = self.arch.fast_storage_mem_area

        query.kernel = op.kernel
        query.config = block_config

        return npu_performance.measure_cycle_cost(self.arch, op.op_type, op.activation and op.activation.op_type, query)

    def estimate_element_access(self, op: SchedulerOperation, block_config, ofm_depth):
        query = npu_performance.PerformanceQuery(op.op_type.npu_block_type)
        query.ifm_shape = op.ifm_read_shape
        query.ifm_memory_area = op.ifm.connection.parent_tens.mem_area
        query.ifm_bits = op.ifm.dtype.size_in_bits()
        query.ifm_format = op.ifm.format
        query.ifm2_shape = op.ifm2_read_shape
        query.ifm2_memory_area = op.ifm2 and op.ifm2.connection.parent_tens.mem_area
        query.ifm2_bits = op.ifm2 and op.ifm2.dtype.size_in_bits()
        query.ifm2_format = op.ifm2 and op.ifm2.format
        query.ofm_shape = op.ofm_write_shape.with_depth(ofm_depth)
        query.ofm_memory_area = op.ofm.connection.parent_tens.mem_area
        query.ofm_bits = op.ofm.dtype.size_in_bits()
        query.ofm_format = op.ofm.format
        if op.parent_op.bias:
            query.const_shape = Shape4D(1, 1, 1, op.ofm.shape.depth)
            query.const_memory_area = self.arch.fast_storage_mem_area

        query.kernel = op.kernel
        query.config = block_config

        return npu_performance.measure_element_access(self.arch, query)

    def propose_schedule_buffering(self, ref_schedule: Schedule, staging_limit_bytes):
        """Create a buffered schedule"""
        buffered_schedule = Schedule(self.sg, f"{ref_schedule.label}_BUFFERED")
        verbose_progress = self.scheduler_options.verbose_progress
        prev_op = None
        for index, sched_op in enumerate(self.sched_ops):
            progress_print(verbose_progress, "Processing SchedulerOp", index, self.sched_ops)
            if sched_op not in ref_schedule.cost_map:
                # sched_op is not part of this sub-schedule - skip
                continue

            self.propose_operator_buffering(sched_op, prev_op, buffered_schedule, ref_schedule, staging_limit_bytes)
            prev_op = sched_op

        return buffered_schedule

    def propose_operator_buffering(
        self,
        sched_op: SchedulerOperation,
        prev_op: Optional[SchedulerOperation],
        buffered_schedule: Schedule,
        ref_schedule: Schedule,
        staging_limit_bytes,
    ):
        # Mild recursion might mean this Op has already been seen
        if sched_op in buffered_schedule.cost_map:
            return

        # Take the reference schedule as default costings for this schedule
        ref_cost = ref_schedule.cost_map[sched_op]
        cost = copy.copy(ref_cost)
        cost.slack_buffering_cycles = ref_cost.cycles.op_cycles
        memory_snapshot = ref_schedule.memory_snapshot
        ref_memory_usage = memory_snapshot[ref_cost.time_index] if ref_cost.time_index < len(memory_snapshot) else 0
        cost.slack_buffering_memory = staging_limit_bytes - ref_memory_usage
        buffered_schedule.cost_map[sched_op] = cost

        # Attempt weight buffering on anything with a weights tensor
        if sched_op.parent_op.weights:
            buffer_limit_bytes = cost.slack_buffering_memory

            # If applicable apply size limitation, but keep it within reason (ratio 1.5).
            # Size limitation is used when use_fast_storage_for_feature_maps have
            # detected that there are fms that do not fit in fast storage.
            if sched_op.evicted_fms_size and ((buffer_limit_bytes / sched_op.evicted_fms_size) >= 1.5):
                buffer_limit_bytes -= sched_op.evicted_fms_size

            self.propose_weight_buffering(
                sched_op.parent_op.weights,
                sched_op.parent_op.bias,
                sched_op,
                prev_op,
                buffered_schedule,
                ref_schedule,
                buffer_limit_bytes,
            )

        return cost

    def weights_needs_dma(self, weight_tensor):
        if weight_tensor and weight_tensor.mem_type not in (MemType.Scratch, MemType.Scratch_fast):
            # Weights are in permanent storage
            # Only when permanent storage differs from feature map storage, there is a point moving the data
            if (
                weight_tensor.mem_area in (MemArea.Dram, MemArea.OffChipFlash)
                and self.arch.permanent_storage_mem_area != self.arch.fast_storage_mem_area
            ):
                return True
        return False

    def propose_weight_buffering(
        self,
        weight_tensor,
        scale_tensor,
        sched_op: SchedulerOperation,
        prev_op: SchedulerOperation,
        buffered_schedule: Schedule,
        ref_schedule: Schedule,
        buffer_limit_bytes,
    ):
        cost = buffered_schedule.cost_map[sched_op]
        prev_cost = buffered_schedule.cost_map.get(prev_op)
        ref_cost = ref_schedule.cost_map[sched_op]
        assert cost and ref_cost

        needs_dma = self.weights_needs_dma(weight_tensor)

        ofm_full_depth_slices = [0, ref_cost.stripe.depth]

        # Encode weights for the full depth
        full_weights, full_scales = weight_compressor.encode_weight_and_scale_tensor(
            self.arch,
            sched_op.parent_op,
            weight_tensor,
            scale_tensor,
            sched_op.kernel,
            cost.block_config,
            ofm_full_depth_slices,
        )
        full_weights_bytes = len(full_weights.buffer)
        cost.ofm_depth_slices = ofm_full_depth_slices

        # No buffering required - take all the weights from permanent storage
        if sched_op.op_type == Op.FullyConnected or not needs_dma:
            cost.npu_weights_tensor = full_weights
            cost.npu_scales_tensor = full_scales
            return

        encoded_weights: Optional[NpuWeightTensor] = full_weights
        encoded_scales = full_scales

        # How many NPU cycles are available under the previously executing
        # operator and SRAM unused for performing buffered DMA transfers
        slack_cycles = prev_cost.slack_buffering_cycles if prev_cost else 0
        slack_memory = prev_cost.slack_buffering_memory if prev_cost else 0

        # Force full depth for cascaded Ops
        if ref_cost.cascade != 0:
            weight_tensor_purpose = TensorSubPurpose.Standard
            weight_buffer_size = full_weights_bytes
            # Update the memory snapshot to reflect the added size of the weights
            ref_schedule.memory_snapshot[ref_cost.time_index] += weight_buffer_size
        else:
            # Estimate the buffering cycle time for the full set of weights
            full_transfer_cycles = npu_performance.measure_mem2mem_cycles(
                self.arch, weight_tensor.mem_area, self.arch.fast_storage_mem_area, full_weights_bytes
            )
            cost.full_weight_transfer_cycles = full_transfer_cycles

            # Calculate the amount of prebuffering necessary (or what is possible with limited
            # double buffer buffer size)
            half_buffer_limit = buffer_limit_bytes // 2
            if full_transfer_cycles > slack_cycles:
                prebuffer_ratio = slack_cycles / full_transfer_cycles
                prebuffer_bytes = min(prebuffer_ratio * full_weights_bytes, half_buffer_limit)
            else:
                prebuffer_bytes = min(full_weights_bytes, half_buffer_limit)

            prebuffer_ratio = prebuffer_bytes / full_weights_bytes

            # Have to split the weights if the initial buffering can't store
            # all of the compressed weights
            if prebuffer_bytes < full_weights_bytes:
                block_depth = cost.block_config.ofm_block.depth

                # Choose initial prebuffering depth (already buffer clamped)
                prebuffer_depth = ref_cost.stripe.depth * prebuffer_ratio
                prebuffer_depth = int(max(16, round_down(prebuffer_depth, ArchitectureFeatures.OFMSplitDepth)))

                # Calculate cycles executed during the prebuffer
                pre_op_cycles = self.estimate_op_performance(sched_op, cost.block_config, prebuffer_depth)
                buffering_depth = ref_cost.stripe.depth * (pre_op_cycles.op_cycles / full_transfer_cycles)

                # Choose initial buffering depth and clamp to the double buffering limit
                buffering_depth = round_up(buffering_depth, block_depth)
                buffering_bytes = (buffering_depth / ref_cost.stripe.depth) * full_weights_bytes
                if buffering_bytes > half_buffer_limit:
                    buffering_depth = (half_buffer_limit / full_weights_bytes) * ref_cost.stripe.depth

                while True:
                    # Attempt to buffer whole blocks
                    if buffering_depth > block_depth:
                        buffering_depth = round_down(buffering_depth, block_depth)
                    else:
                        buffering_depth = round_down(buffering_depth, ArchitectureFeatures.OFMSplitDepth)
                    buffering_depth = int(max(buffering_depth, ArchitectureFeatures.OFMSplitDepth))

                    # Create list of depth slices
                    depth_slices = [0]
                    if prebuffer_depth < ref_cost.stripe.depth:
                        depth_slices += list(range(prebuffer_depth, ref_cost.stripe.depth, buffering_depth))
                    depth_slices.append(ref_cost.stripe.depth)

                    # Encode weights based depth slices
                    cost.ofm_depth_slices = depth_slices
                    encoded_weights, encoded_scales = weight_compressor.encode_weight_and_scale_tensor(
                        self.arch,
                        sched_op.parent_op,
                        weight_tensor,
                        scale_tensor,
                        sched_op.kernel,
                        cost.block_config,
                        cost.ofm_depth_slices,
                    )
                    assert encoded_weights is not None
                    # Chosen buffering might not fit at all, iterate until it does
                    # or until the minimum usable slice size is reached
                    if (
                        encoded_weights.double_buffer_size() <= buffer_limit_bytes
                        or prebuffer_depth == ArchitectureFeatures.OFMSplitDepth
                    ):
                        break

                    if buffering_depth > prebuffer_depth:
                        buffering_depth = round_up(buffering_depth // 2, ArchitectureFeatures.OFMSplitDepth)
                    else:
                        prebuffer_depth = round_up(prebuffer_depth // 2, ArchitectureFeatures.OFMSplitDepth)

                # Calculate cycles required to run the last op for use as future slack
                tail_cycles = self.estimate_op_performance(
                    sched_op, cost.block_config, depth_slices[-1] - depth_slices[-2]
                )
                cost.slack_buffering_cycles = tail_cycles.op_cycles

        # Determine whether the weights need to be double buffered
        weight_buffer_size = min(len(encoded_weights.buffer), encoded_weights.max_range_bytes())

        # Only buffer weights if there's still space left for the buffer
        if weight_buffer_size <= buffer_limit_bytes:
            assert weight_buffer_size % 16 == 0
            # Determine whether to double buffer or single buffer
            double_buffer_size = encoded_weights.double_buffer_size()
            if (double_buffer_size <= buffer_limit_bytes) and (weight_buffer_size < len(encoded_weights.buffer)):
                weight_tensor_purpose = TensorSubPurpose.DoubleBuffer
            else:
                weight_tensor_purpose = TensorSubPurpose.Standard

            cost.buffered_weight_tensors = [
                self.buffer_tensor(
                    encoded_weights,
                    weight_tensor_purpose,
                    encoded_weights.double_buffer_sizes[0],
                    weight_tensor.name + "_buffer",
                )
            ]
            if weight_tensor_purpose == TensorSubPurpose.DoubleBuffer:
                buf2 = self.buffer_tensor(
                    encoded_weights,
                    weight_tensor_purpose,
                    encoded_weights.double_buffer_sizes[1],
                    weight_tensor.name + "_buffer2",
                )
                cost.buffered_weight_tensors.append(buf2)

            # Note! OFM depth slices define slices as [0, s1, ... sn]. For example, [0, 70, 140] describes two slices
            # (0-70 and 70-140) but has a length of 3, which would result in idx = 3 % 2 = 1 if two buffers were used.
            last_used_buffer_idx = len(cost.ofm_depth_slices) % len(cost.buffered_weight_tensors)
            weight_buffer_size = encoded_weights.double_buffer_sizes[last_used_buffer_idx]

            if ref_cost.cascade == 0:
                # Determine if the lifetime can be extended and pre-buffer the first weight buffer
                # under the previous operation
                cost.buffered_weight_tensors[0].pre_buffer = encoded_weights.double_buffer_size() < slack_memory

            cost.slack_buffering_memory -= weight_buffer_size
        else:
            # Don't slice or buffer - use the whole depth from persistent storage
            cost.ofm_depth_slices = ofm_full_depth_slices
            encoded_weights = full_weights
            encoded_scales = full_scales

        cost.npu_weights_tensor = encoded_weights
        cost.npu_scales_tensor = encoded_scales

    def buffer_tensor(self, src_tensor: Tensor, sub_purpose: TensorSubPurpose, buffer_size: int, name: str) -> Tensor:
        buffered_weight_tensor = Tensor([1, 1, 1, buffer_size], DataType.uint8, name)
        buffered_weight_tensor.src_tensor = src_tensor
        buffered_weight_tensor.mem_area = self.arch.fast_storage_mem_area
        buffered_weight_tensor.mem_type = MemType.Scratch_fast
        buffered_weight_tensor.purpose = TensorPurpose.Weights
        buffered_weight_tensor.sub_purpose = sub_purpose
        return buffered_weight_tensor

    def propose_minimal_schedule(self) -> Schedule:
        """Proposes scheduling parameters where every operator is subdivided into the smallest stripe that satisfies the
        next operators stride"""
        min_schedule = Schedule(self.sg, "MIN")
        cost_map = min_schedule.cost_map
        verbose_progress = self.scheduler_options.verbose_progress
        # Keep track of the previous Op - which consumes the current Op's OFM
        prev_op: Optional[SchedulerOperation] = None
        for index, sched_op in enumerate(reversed(self.sched_ops)):
            progress_print(verbose_progress, "Processing SchedulerOp", index, self.sched_ops)
            min_stripe_height = prev_op.kernel.stride.y if prev_op else 1
            if is_nearest(sched_op.resampling_mode):
                # is nearest requires even stripes
                min_stripe_height += min_stripe_height % 2
            min_stripe = sched_op.ofm.shape.with_height(min_stripe_height)

            cost = sched_op.create_scheduler_info(self.nng, min_stripe)
            cost.cycles = self.estimate_op_performance(sched_op, cost.block_config, sched_op.ofm.shape.depth)
            cost_map[sched_op] = cost

            prev_op = sched_op

        return min_schedule

    def propose_schedule_striping(self, final_stripe: Shape4D, label: str, ref_schedule: Schedule) -> Schedule:
        """Proposes new striping for a schedule. The stripe is derived from the ifm requirements of the next Op down"""
        ref_cost = ref_schedule.cost_map

        striped_schedule = Schedule(self.sg, label)
        stripe = final_stripe
        for sched_op in reversed(self.sched_ops):
            if sched_op not in ref_cost:
                # sched_op is not part of the sub-schedule - skip
                continue

            # Create a cost entry with the new stripe
            cost = sched_op.create_scheduler_info(self.nng, stripe)

            weight_tensor = cost.npu_weights_tensor
            for idx, buffered_tens in enumerate(ref_cost[sched_op].buffered_weight_tensors):
                # If the weights are buffered in the reference schedule they should be in the new proposal
                cost.buffered_weight_tensors.append(
                    self.buffer_tensor(
                        weight_tensor,
                        buffered_tens.sub_purpose,
                        weight_tensor.double_buffer_sizes[idx],
                        buffered_tens.name,
                    )
                )

            # Estimate performance
            cost.cycles = self.estimate_op_performance(sched_op, cost.block_config, sched_op.ofm.shape.depth)
            striped_schedule.cost_map[sched_op] = cost

            # Calculate the preceeding Op's stripe.

            # In certain cases where an upscaling Op is cascaded,
            # it may get instructed to produce an odd stripe height.
            # Thus we need to force it back to even heights.
            force_even_stripe_heights = False
            for op in self.sched_ops:
                # Check if the cascade has a Nearest Neighbor-op.
                # If that is the case, force the stripes to be even.
                if (
                    ref_cost.get(op, None)
                    and ref_cost.get(sched_op, None)
                    and ref_cost[op].cascade == ref_cost[sched_op].cascade
                    and is_nearest(op.resampling_mode)
                ):
                    force_even_stripe_heights = True
                    break
            upscaling_remainder = stripe.height % to_upscale(sched_op.resampling_mode)
            height = stripe.height + (stripe.height % 2 if force_even_stripe_heights else upscaling_remainder)
            stripe = sched_op.ifm.shape.with_height(height * sched_op.kernel.stride.y)

        return striped_schedule

    def estimate_schedule_memory_usage(self, schedule: Schedule, non_local_mem_usage: dict):
        """Estimates the memory usage of a schedule"""
        cost = schedule.cost_map
        cascades = schedule.cascades
        peak_mem_usage = 0
        for sched_op in self.sched_ops:
            if sched_op not in cost:
                # sched_op is not part of the sub-schedule - skip
                continue

            if cost[sched_op].cascade:
                # This Op is part of a cascade - use the cascade's memory usage
                cascade_info = cascades[cost[sched_op].cascade]
                op_mem_usage = cascade_info.mem_usage + non_local_mem_usage.get(sched_op, 0)
            else:
                # This Op is not part of a cascade - calculate the memory usage
                op_weight_buffer = sum(tens.storage_size() for tens in cost[sched_op].buffered_weight_tensors)

                op_mem_usage = (
                    sched_op.ifm_size_in_bytes()
                    + sched_op.ofm_size_in_bytes()
                    + op_weight_buffer
                    + non_local_mem_usage.get(sched_op, 0)
                )
            peak_mem_usage = max(op_mem_usage, peak_mem_usage)

        return peak_mem_usage

    def build_cascades_for_min_schedule(self, min_schedule: Schedule, max_template: Schedule, memory_limit: int):
        verbose_progress = self.scheduler_options.verbose_progress
        # Update memory snapshot
        self.sg.schedule = min_schedule
        self.update_op_memory_snapshot(min_schedule)

        # Calculate residual memory for Min schedule
        non_local_mem_usage = {}
        for index, sched_op in enumerate(self.sched_ops):
            progress_print(verbose_progress, "Processing SchedulerOp", index, self.sched_ops)
            time_index = min_schedule.cost_map[sched_op].time_index

            if self.arch.is_spilling_enabled():
                # For Dedicated SRAM only the intermediate buffers are in SRAM, hence op_mem_usage is 0
                op_mem_usage = 0
            else:
                # Min schedule only have ifm and ofm in SRAM (no buffered weigth tensors)
                # Only include IFM's that are in the scratch area
                ifm = sched_op.ifm.connection.parent_tens
                ifm_size = (
                    0 if ifm.mem_type not in (MemType.Scratch, MemType.Scratch_fast) else sched_op.ifm_size_in_bytes()
                )
                ofm_size = 0 if ofm_can_reuse_ifm(sched_op) else sched_op.ofm_size_in_bytes()
                op_mem_usage = ifm_size + ofm_size

            non_local_mem_usage[sched_op] = min_schedule.memory_snapshot[time_index] - op_mem_usage
            assert non_local_mem_usage[sched_op] >= 0

        # Create cascades for Min schedule
        cascade_builder = CascadeBuilder(self.sched_ops, self.arch.is_spilling_enabled(), non_local_mem_usage)
        cascade_builder.build_cascades(min_schedule, max_template, memory_limit)

    def optimize_sub_schedule(
        self, cascade_info: CascadeInfo, ref_schedule: Schedule, max_template: Schedule, memory_limit: int
    ) -> Schedule:
        """Extracts the Ops covered by the given cascade and creates a sub-schedule. The sub-schedule is optimized by
        proposing weight buffering and then continously proposing new stripe sizes"""
        ref_cost = ref_schedule.cost_map
        # Extract the ops that are part of this sub-schedule
        start = cascade_info.start
        end = cascade_info.end
        sub_schedule_ops = self.sched_ops[start : end + 1]
        # Create a sub-schedule that contains only the costs for the Ops that are part of the sub-schedule
        sub_schedule = Schedule(self.sg, f"SUB_{start}_{end}")
        for sched_op in sub_schedule_ops:
            sub_schedule.cost_map[sched_op] = ref_cost[sched_op]

        sub_schedule.cascades[end] = cascade_info
        # Use the memory snapshot from the reference schedule
        sub_schedule.memory_snapshot = ref_schedule.memory_snapshot

        # Calculate memory usage that is live during the sub-schedule but not part of it
        time_for_cascade = ref_cost[sub_schedule_ops[0]].time_index
        mem_usage_parallel_to_sub_schedule = ref_schedule.memory_snapshot[time_for_cascade] - cascade_info.mem_usage
        # If the first Op's IFM has other consumers it has to live throughout the whole sub-schedule whether it's
        # included in a cascade or not. Not valid in Dedicated SRAM mode (spilling enabled).
        persistent_initial_ifm = (
            sub_schedule_ops[0].ifm_size_in_bytes()
            if not self.arch.is_spilling_enabled() and len(sub_schedule_ops[0].ifm.connection.consumers) > 1
            else 0
        )
        # Calculate non-local-mem-usage per Operator
        non_local_mem_usage = {}
        for idx, sched_op in enumerate(sub_schedule_ops):
            non_local_mem_usage[sched_op] = mem_usage_parallel_to_sub_schedule
            if idx != 0:
                non_local_mem_usage[sched_op] += persistent_initial_ifm

        cascade_builder = CascadeBuilder(sub_schedule_ops, self.arch.is_spilling_enabled(), non_local_mem_usage)

        # Start by adding buffering
        buffered_sub_schedule = self.propose_schedule_buffering(sub_schedule, self.sram_limit)
        # Copy the cascades over from the unbuffered-schedule
        buffered_sub_schedule.cascades = sub_schedule.cascades

        # Generate the possible stripings for the final Op in the sub-schedule
        final_ofm_shape = sub_schedule_ops[-1].ofm.shape

        # Skip testing the min stripe used in the MIN schedule since that will be used
        # anyway if no new cascades are created below
        last_op = sub_schedule_ops[-1]
        min_stripe_h = sub_schedule.cost_map[last_op].stripe.height + 1

        possible_stripes = [
            final_ofm_shape.with_height(stripe_h) for stripe_h in range(min_stripe_h, final_ofm_shape.height // 2 + 1)
        ]
        # Propose different striping
        best_schedule = None
        max_nbr_of_cascades = 0
        for iteration, proposed_stripe in enumerate(possible_stripes):
            proposed_schedule = self.propose_schedule_striping(
                proposed_stripe, f"OPTIMIZED_{iteration}", buffered_sub_schedule
            )

            cascade_builder.build_cascades(proposed_schedule, max_template, memory_limit)
            nbr_of_cascades = len(proposed_schedule.cascades)
            if iteration == 0:
                # First iteration - used as limit to prevent splitting up the cascades
                # Long cascades are better in order to reduce IFM/IFM dram bandwidth
                max_nbr_of_cascades = nbr_of_cascades

            # Check if proposal fits
            proposed_schedule_mem_usage = self.estimate_schedule_memory_usage(proposed_schedule, non_local_mem_usage)
            if (proposed_schedule_mem_usage) <= memory_limit and nbr_of_cascades <= max_nbr_of_cascades:
                best_schedule = proposed_schedule

                if not proposed_schedule.cascades:
                    # No cascading required - early exit
                    break
            else:
                break

        return best_schedule

    def optimize_schedule(
        self,
        schedule: Schedule,
        max_sched: Schedule,
        max_template: Schedule,
    ) -> Schedule:
        """Extracts sub-schedules based on the cascades and optimizes them and applies them to the final schedule"""
        verbose_progress = self.scheduler_options.verbose_progress
        if max_sched.fast_storage_peak_usage < self.sram_limit and not self.arch.is_spilling_enabled():
            # Maximum performance schedule fits within the SRAM target
            return max_sched

        # Iterate over a copy of the cascades since they may change during the loop
        cascades = list(schedule.cascades.values())
        for index, cascade_info in enumerate(cascades):
            progress_print(verbose_progress, "Processing cascade", index, cascades)
            # Optimize the sub-schedule in this cascade
            opt_sub_schedule = self.optimize_sub_schedule(cascade_info, schedule, max_template, self.sram_limit)
            if opt_sub_schedule:
                # Remove the existing cascade
                del schedule.cascades[cascade_info.end]
                # Update the sub-schedule Op and cascade costs to the full schedule
                schedule.cost_map.update(opt_sub_schedule.cost_map)
                schedule.cascades.update(opt_sub_schedule.cascades)

        # Update memory snapshot
        self.sg.schedule = schedule
        self.update_op_memory_snapshot(schedule)
        # Propose schedule buffering to the optimized schedule
        optimized_sched = self.propose_schedule_buffering(schedule, self.sram_limit)
        # Copy the cascade's metadata from the unbuffered schedule
        optimized_sched.cascades = schedule.cascades
        return optimized_sched

    def optimize_weight_buffering_size(
        self,
        min_schedule: Schedule,
    ):
        verbose_progress = self.scheduler_options.verbose_progress
        default_schedule = self.sg.schedule
        npu_performance.calc_new_performance_for_network(self.nng, self.arch, None, False)
        default_tot_cycles = self.nng.cycles[npu_performance.PassCycles.Total]
        default_dram_cycles = self.nng.cycles[npu_performance.PassCycles.DramAccess]

        # Restore mem/type for scratched_fms
        for tens in self.scratched_fms:
            tens.mem_area = self.scratched_fms[tens][0]
            tens.mem_type = self.scratched_fms[tens][1]

        self.update_op_memory_snapshot(self.sg.schedule)

        # Collect live ranges from tensors
        memories_list = [(self.arch.fast_storage_mem_area, set((MemType.Scratch, MemType.Scratch_fast)))]
        lr_graph = live_range.LiveRangeGraph()
        for mem_area, mem_type_set in memories_list:
            live_range.extract_live_ranges_from_schedule(self.sg, mem_area, mem_type_set, lr_graph, verbose_progress)

        # Find the relation between the sched_op and the buffering tensor
        weight_ops = {}
        for sched_op in self.sched_ops:
            cost = self.sg.schedule.cost_map[sched_op]
            for tens in cost.buffered_weight_tensors:
                weight_ops[tens] = sched_op

        # Filter out weight buffer live ranges
        weight_lrs = []
        for lr in lr_graph.lrs:
            for tens in lr.tensors:
                if weight_ops.get(tens):
                    weight_lrs.append(lr)
                    break

        # See if any evicted fm overlaps with a weight buffering op.
        # If this is the case add a size limitation to the buffering op
        for lr in self.evicted_fms:
            for weight_lr in weight_lrs:
                if lr.overlaps_ranges(weight_lr):
                    for tens in weight_lr.tensors:
                        sched_op = weight_ops.get(tens)
                        if sched_op:
                            # Add size reduction to the op
                            sched_op.evicted_fms_size += lr.size
                            break

        self.sg.schedule = min_schedule
        self.update_op_memory_snapshot(self.sg.schedule)

        # Run schedule buffering - with weight buffer size reduction
        schedule = self.propose_schedule_buffering(self.sg.schedule, self.sram_limit)
        schedule.cascades = self.sg.schedule.cascades
        self.sg.schedule = schedule

        # Apply new buffer schdule and calc new performance
        self.update_op_memory_snapshot(self.sg.schedule)
        self.apply_schedule(self.sg.schedule)
        self.use_fast_storage_for_feature_maps(self.sg.schedule, self.sram_limit)

        npu_performance.calc_new_performance_for_network(self.nng, self.arch, None, False)
        new_tot_cycles = self.nng.cycles[npu_performance.PassCycles.Total]
        new_dram_cycles = self.nng.cycles[npu_performance.PassCycles.DramAccess]

        improvement_tot = (
            round((default_tot_cycles - new_tot_cycles) / default_tot_cycles, 2) if default_tot_cycles != 0 else 0
        )
        improvement_dram = (
            round((default_dram_cycles - new_dram_cycles) / default_dram_cycles, 2) if default_dram_cycles != 0 else 0
        )

        # Compare both total and dram improvement
        if not (improvement_tot >= 0 and improvement_dram > 0):
            # No improvement, restore the default schedule
            for sched_op in self.sched_ops:
                sched_op.evicted_fms_size = 0

            for tens in self.scratched_fms:
                tens.mem_area = self.scratched_fms[tens][0]
                tens.mem_type = self.scratched_fms[tens][1]

            self.sg.schedule = default_schedule
            self.update_op_memory_snapshot(self.sg.schedule)
            self.apply_schedule(self.sg.schedule)
            self.use_fast_storage_for_feature_maps(self.sg.schedule, self.sram_limit)

    def apply_schedule(self, sched: Schedule):
        """Applies the given schedule as a final solution"""
        for sched_op in self.sched_ops:
            op_info = sched.cost_map[sched_op]
            cascade_info = sched.cascades.get(op_info.cascade, None)
            if cascade_info and sched_op in cascade_info.buffers:
                buffer_tens = sched_op.ifm.connection.parent_tens
                # Apply memory area and type
                buffer_tens.mem_area = self.arch.fast_storage_mem_area
                buffer_tens.mem_type = MemType.Scratch_fast
                # Apply Rolling buffer
                buffer_tens.set_format(TensorFormat.NHCWB16, self.arch)
                buffer_tens.set_new_sub_purpose(TensorSubPurpose.RollingBufferY, cascade_info.buffers[sched_op].height)

            sched_op.parent_ps.block_config = op_info.block_config.old_style_representation()

            # Ensure that the src_tensor reference is set correctly
            for tens in op_info.buffered_weight_tensors:
                tens.src_tensor = op_info.npu_weights_tensor

    def use_fast_storage_for_feature_maps(self, schedule, staging_limit):
        """Finds the set of feature maps that fits within the staging limit which combined has the largest amount of
        access cycles and moves those feature map into fast storage"""
        max_mem_usage = []
        base_mem_usage = []
        fast_storage_type = MemType.Scratch_fast
        fast_storage_mem_area = self.arch.fast_storage_mem_area
        self.evicted_fms = []

        # Force all OFMs to fast-storage
        for sched_op in self.sched_ops:
            cost = schedule.cost_map[sched_op]
            if cost.cascade == 0 and sched_op.get_dependants():
                ofm_tens = sched_op.ofm.connection.parent_tens
                # Do not move subgraph outputs or Variable Tensor Writes
                if (
                    not any(cons is None for cons in ofm_tens.consumer_list)
                    and sched_op.parent_op.memory_function is not Op.VariableTensorWrite
                ):
                    if ofm_tens not in self.scratched_fms:
                        # Remember default mem area and mem type, only done once
                        self.scratched_fms[ofm_tens] = (ofm_tens.mem_area, ofm_tens.mem_type)

                    ofm_tens.mem_area = fast_storage_mem_area
                    ofm_tens.mem_type = fast_storage_type

        # Collect live ranges from tensors
        memories_list = [(fast_storage_mem_area, set((MemType.Scratch, MemType.Scratch_fast)))]
        lr_graph = live_range.LiveRangeGraph()
        for mem_area, mem_type_set in memories_list:
            live_range.extract_live_ranges_from_schedule(
                self.sg,
                mem_area,
                mem_type_set,
                lr_graph,
            )

        max_mem_usage = lr_graph.get_temporal_memory_usage(fast_storage_mem_area)

        # If max_mem_usage does not exceed staging limit at any point all lrs fit and can stay in fast storage
        if max(max_mem_usage) <= staging_limit:
            return

        # Build up the base memory usage by removing the mem_usage of the lrs we previously moved to fast-storage
        base_mem_usage = np.array(max_mem_usage)
        curr_lrs = []
        for lr in lr_graph.lrs:
            for tens in lr.tensors:
                if self.scratched_fms.get(tens):
                    curr_lrs.append(lr)
                    base_mem_usage[lr.start_time : lr.end_time + 1] -= lr.size
                    break
        competing_lrs = []
        competing_tens_access = {}

        # Evict live ranges that will never fit
        for lr in curr_lrs.copy():
            base_usage = max(base_mem_usage[lr.start_time : lr.end_time + 1])
            if base_usage + lr.size > staging_limit:
                # Lr will never fit and may thus be evicted
                self.evicted_fms.append(lr)
                FastStorageComponentAllocator.evict(lr, max_mem_usage, self.scratched_fms)
                curr_lrs.remove(lr)

        # Keep live ranges that will always fit in fast storage and let the remaining ones compete
        for lr in curr_lrs:
            # Since max_mem_usage is the memory usage with all FMs still in fast-storage,
            # the memory limit cannot be exceeded if max_mem_usage does not.
            # Thus, the affected lrs can remain in fast-storage if the following is true
            if max(max_mem_usage[lr.start_time : lr.end_time + 1]) <= staging_limit:
                FastStorageComponentAllocator.keep(lr, base_mem_usage)
            else:
                competing_lrs.append(lr)
                for tens in lr.tensors:
                    competing_tens_access[tens] = 0

        # All lrs and their tensors have been handled if competing_lrs_sz is zero, we may thus return
        if len(competing_lrs) == 0:
            return

        # Estimate element access for all tensors that are competing for a place in fast-storage.
        # This number is used when deciding which tensor that stays in fast-storage
        for sched_op in self.sched_ops:
            cost = schedule.cost_map[sched_op]

            if competing_tens_access.get(sched_op.ifm.connection.parent_tens) is not None:
                tens = sched_op.ifm.connection.parent_tens
                access = self.estimate_element_access(sched_op, cost.block_config, sched_op.ofm.shape.depth)
                competing_tens_access[tens] += access.ifm_read[0]

            if sched_op.ifm2 and competing_tens_access.get(sched_op.ifm2.connection.parent_tens) is not None:
                tens = sched_op.ifm2.connection.parent_tens
                access = self.estimate_element_access(sched_op, cost.block_config, sched_op.ofm.shape.depth)
                competing_tens_access[tens] += access.ifm_read[1]

            if competing_tens_access.get(sched_op.ofm.connection.parent_tens) is not None:
                tens = sched_op.ofm.connection.parent_tens
                access = self.estimate_element_access(sched_op, cost.block_config, sched_op.ofm.shape.depth)
                competing_tens_access[tens] += access.ofm_write

        # Sort live ranges "from left to right" on the time axis to simplify checking overlapping ranges
        competing_lrs = sorted(competing_lrs, key=lambda lr: (lr.start_time, lr.end_time + 1, lr.size))

        # Remove lrs that have a live range that is too long compared to others.
        # They are causing problems for the HillClimb Allocator when it has to
        # change the allocation indices, in order to fit all the allocations into SRAM.
        # This problem only occur in larger networks with complex graphs.
        #
        # Limit the number of items for allocate_component to work with max MAX_EXHAUSTIVE_ITEMS
        # at the time. Too many will give too long compilation time
        #
        # Too long is currently decided to be (based on experience, analyzing many networks):
        # Compare lr at postion i with lr at position i + MAX_EXHAUSTIVE_ITEMS.
        # If end time differs by at least MAX_EXHAUSTIVE_LIFE_RANGE then do not include lr at position i.
        if len(competing_lrs) > FastStorageComponentAllocator.MAX_EXHAUSTIVE_ITEMS:
            # Create a copy of the original list to iterate over because the original version is modified in-loop
            competing_lrs_copy = competing_lrs.copy()
            for i, lr in enumerate(competing_lrs_copy):
                lr_time = lr.end_time - lr.start_time
                # Only check live ranges longer than MAX_EXHAUSTIVE_LIFE_RANGE
                if lr_time >= FastStorageComponentAllocator.MAX_EXHAUSTIVE_LIFE_RANGE:
                    # Compare current lr with lr at position lr + MAX_EXHAUSTIVE_ITEMS
                    cmp_pos = min(i + FastStorageComponentAllocator.MAX_EXHAUSTIVE_ITEMS, len(competing_lrs) - 1)

                    # Compare end times + plus a margin by MAX_EXHAUSTIVE_LIFE_RANGE
                    if (
                        lr.end_time
                        > competing_lrs_copy[cmp_pos].end_time + FastStorageComponentAllocator.MAX_EXHAUSTIVE_LIFE_RANGE
                    ):
                        # Current lr live time stands out, remove it. No use adding it to the
                        # evicted_fms list since the lr should not be included in the fast storage allocation
                        FastStorageComponentAllocator.evict(lr, max_mem_usage, self.scratched_fms)
                        competing_lrs.remove(lr)

        # Split competing live ranges into components by finding disconnected groups of live ranges or components of
        # max size MAX_EXHAUSTIVE_ITEMS
        start = 0
        end_time = competing_lrs[0].end_time
        component_allocator = FastStorageComponentAllocator(base_mem_usage, max_mem_usage, staging_limit)
        component_ranges = []
        for i, lr in enumerate(competing_lrs):
            nbr_items = i - start
            if lr.start_time <= end_time and (nbr_items < FastStorageComponentAllocator.MAX_EXHAUSTIVE_ITEMS):
                end_time = max(end_time, lr.end_time)
            else:
                # Number items reached max items or current lr's start time
                # does not overlap with previous lr's end time
                component_ranges.append((start, i))
                start = i
                end_time = lr.end_time
        component_ranges.append((start, len(competing_lrs)))

        # Allocate each component separately
        for start, end in component_ranges:
            component_allocator.allocate_component(
                competing_lrs[start:end],
                max_mem_usage,
                base_mem_usage,
                self.scratched_fms,
                competing_tens_access,
                self.evicted_fms,
            )
        assert max(max_mem_usage) <= staging_limit, "Allocation exceeds staging limit"

    def print_schedule(self, schedule: Schedule):
        print(f"Schedule: '{schedule.name}'")
        for sched_op in self.sched_ops:
            if sched_op not in schedule.cost_map:
                # Sub-schedule printing
                continue

            op_info = schedule.cost_map[sched_op]
            print(f"\t{sched_op.index}: Operation {sched_op.name}  - OFM {sched_op.ofm.shape}")
            print(f"\t\tType: {sched_op.op_type}")
            print(f"\t\tKernel: {sched_op.kernel}")
            print(f"{op_info}")
            mem_usage = (
                schedule.memory_snapshot[op_info.time_index]
                if op_info.time_index < len(schedule.memory_snapshot)
                else 0
            )
            print(f"\t\tSRAM Used: {mem_usage} bytes")

        print("\tCascades:")
        for i, cascade in enumerate(schedule.cascades.values()):
            print(f"\t\t{i}: {cascade.start} -> {cascade.end}, size: {cascade.mem_usage}")


def _update_memory_snapshot_for_all_npu_graphs(
    nng: Graph, arch: ArchitectureFeatures, schedulers, verbose_progress: bool = False
):
    mem_area = arch.fast_storage_mem_area
    mem_type_set = set((MemType.Scratch, MemType.Scratch_fast))

    # Collect live ranges for the full graph
    # extract_live_ranges_from_cascaded_passes will start from the root sg and
    # all sub graphs/cascaded passes will be visited and the correct time_index
    # will be set for all the tensors.
    lr_graph = live_range.LiveRangeGraph()
    live_range.extract_live_ranges_from_cascaded_passes(
        nng.get_root_subgraph(), mem_area, mem_type_set, lr_graph, Tensor.AllocationQuantum, verbose_progress
    )
    # Populate time-array with memory used by live ranges
    temporal_usage = lr_graph.get_temporal_memory_usage(arch.fast_storage_mem_area)

    # Update snapshot for all the npu sub graphs
    # Not needed for the scheduler any longer but npu_performance
    # is using this information so it must have the correct state
    for sg in schedulers:
        sg.schedule.memory_snapshot = temporal_usage
        sg.schedule.fast_storage_peak_usage = max(temporal_usage, default=0)


def _update_tensor_allocation(nng: Graph, arch: ArchitectureFeatures, options):
    """
    Creates live ranges and runs tensor allocator for the current schedule
    (i.e. sg.schedule for all subgraphs), returns the maximum memory usage
    and updates SchedulerOpInfo.mem_usage for all operations in the schedule.
    """
    root_sg = nng.get_root_subgraph()

    alloc_list = []
    if arch.is_spilling_enabled():
        mem_alloc_scratch_fast = (arch.fast_storage_mem_area, set((MemType.Scratch_fast,)))
        mem_alloc_scratch = (arch.feature_map_storage_mem_area, set((MemType.Scratch,)))
        # Order is important
        alloc_list.append(mem_alloc_scratch_fast)
        alloc_list.append(mem_alloc_scratch)
    else:
        mem_alloc_scratch = (arch.feature_map_storage_mem_area, set((MemType.Scratch, MemType.Scratch_fast)))
        alloc_list.append(mem_alloc_scratch)

    for mem_area, mem_type_set in alloc_list:
        tensor_allocation.allocate_tensors(
            nng,
            root_sg,
            arch,
            mem_area,
            mem_type_set,
            tensor_allocator=options.tensor_allocator,
            verbose_allocation=options.verbose_allocation,
            verbose_progress=options.verbose_progress,
            cpu_tensor_alignment=options.cpu_tensor_alignment,
            hillclimb_max_iterations=options.hillclimb_max_iterations,
        )


class FastStorageComponentAllocator:
    MAX_EXHAUSTIVE_LIFE_RANGE = 20
    MAX_EXHAUSTIVE_ITEMS = 20

    def __init__(self, base_mem_usage, max_mem_usage, staging_limit):
        self.base_mem_usage = base_mem_usage
        self.max_mem_usage = max_mem_usage
        self.staging_limit = staging_limit
        self.lrs = []
        self.evicted = []
        self.curr_evicted = []
        self.remaining_total_size = []
        self.best_score = 0
        self.competing_tens_access = {}

    def allocate_exhaustive(self, ix, score):
        # Favour tensors with highest element access (score)
        if ix >= self.num_lrs:
            if score > self.best_score:
                self.best_score = score
                self.evicted = self.curr_evicted.copy()
            return

        lr = self.lrs[ix]

        # If adding the tensor size to the base mem usage doesn't exceed the staging limit anywhere on the lr time
        # range, it can fit and the case where the tensor is included needs to be checked
        can_fit = max(self.base_mem_usage[lr.start_time : lr.end_time + 1]) + lr.size <= self.staging_limit
        if can_fit:
            # Tensor can fit, add tensor element access to the score and check case where tensor is included
            self.curr_evicted[ix] = False
            self.base_mem_usage = self.update_mem_usage(self.base_mem_usage, lr, True)
            self.allocate_exhaustive(ix + 1, score + self.competing_tens_access[lr.tensors[0]])
            self.base_mem_usage = self.update_mem_usage(self.base_mem_usage, lr, False)

        # If the max mem usage doesn't exceed the staging limit anywhere on the lr time range, it always fits and the
        # case where the tensor is not included can be skipped
        always_fits = max(self.max_mem_usage[lr.start_time : lr.end_time + 1]) <= self.staging_limit
        if not always_fits:
            # Tensor doesn't always fit, check case when tensor is not included
            self.curr_evicted[ix] = True
            self.max_mem_usage = self.update_mem_usage(self.max_mem_usage, lr, False)
            self.allocate_exhaustive(ix + 1, score)
            self.max_mem_usage = self.update_mem_usage(self.max_mem_usage, lr, True)

    @staticmethod
    def update_mem_usage(mem_usage, lr, increase):
        size = lr.size if increase else -lr.size
        for t in range(lr.start_time, lr.end_time + 1):
            mem_usage[t] += size
        return mem_usage

    @staticmethod
    def evict(lr, max_mem_usage, scratched_fms):
        for t in range(lr.start_time, lr.end_time + 1):
            max_mem_usage[t] -= lr.size
        for tens in lr.tensors:
            if tens in scratched_fms:
                tens.mem_area = scratched_fms[tens][0]
                tens.mem_type = scratched_fms[tens][1]

    @staticmethod
    def keep(lr, base_mem_usage):
        for t in range(lr.start_time, lr.end_time + 1):
            base_mem_usage[t] += lr.size

    def allocate_component(self, lrs, max_mem, min_mem, scratched_fms, competing_tens_access, evicted_fms):
        self.lrs = lrs
        self.num_lrs = len(lrs)
        self.evicted = [0] * self.num_lrs
        self.curr_evicted = [0] * self.num_lrs
        self.best_score = -1
        self.competing_tens_access = competing_tens_access
        # Recursively evaluate all permutations of allocations of the lrs found in the component.
        # For every permutation that fits within the staging_limit there is a score calculated.
        # The permutation with the highest score will then be chosen. The score is calculated
        # as the sum of the actual element access (ifm read and ofm write) for all the
        # including tensors. So it is not necessary the tensor with the biggest size that ends up
        # being included in the result.
        self.allocate_exhaustive(0, 0)
        # Optimal allocation has been found, move lrs accordingly
        for i, lr in enumerate(self.lrs):
            if self.evicted[i]:
                self.evict(lr, max_mem, scratched_fms)
                if lr not in evicted_fms:
                    evicted_fms.append(lr)
            else:
                self.keep(lr, min_mem)
                if lr in evicted_fms:
                    evicted_fms.remove(lr)


def schedule_passes(nng: Graph, arch: ArchitectureFeatures, options, scheduler_options: SchedulerOptions):
    """Entry point for the Scheduler"""
    verbose_progress = scheduler_options.verbose_progress
    # Initialize CPU subgraphs
    schedulers = dict()
    # Initialize schedulers with max schedule. Only schedule NPU subgraphs
    for sg_idx, sg in enumerate(nng.subgraphs):
        progress_print(verbose_progress, "Processing subgraph", sg_idx, nng.subgraphs)
        if sg.placement != PassPlacement.Npu:
            # Create cascaded passes for CPU Ops
            cascaded_passes = []
            for pass_idx, ps in enumerate(sg.passes):
                progress_print(verbose_progress, "Creating cascaded passes for CPU op", pass_idx, sg.passes)
                cps = CascadedPass(
                    ps.name,
                    SchedulingStrategy.WeightStream,
                    ps.inputs,
                    [],
                    ps.outputs,
                    [ps],
                    ps.placement,
                    False,
                )

                cps.time = pass_idx
                ps.cascade = cps
                cascaded_passes.append(cps)

            sg.cascaded_passes = cascaded_passes
        else:
            # Npu subgraph - create schedule
            scheduler = Scheduler(nng, sg, arch, scheduler_options)
            schedulers[sg] = scheduler

            progress_print(verbose_progress, "Creating scheduler representation")
            scheduler.create_scheduler_representation(arch)
            sg.sched_ops = scheduler.sched_ops

            # Create the Max schedule template
            max_schedule_template = scheduler.create_initial_schedule()
            scheduler.max_schedule = max_schedule_template

            progress_print(verbose_progress, "Creating optimised max schedule")
            # Create the optimimised Max schedule
            sg.schedule = max_schedule_template
            scheduler.update_op_memory_snapshot(max_schedule_template)
            opt_max_schedule = scheduler.propose_schedule_buffering(max_schedule_template, 1 << 32)
            sg.schedule = opt_max_schedule
            scheduler.update_op_memory_snapshot(opt_max_schedule)

            progress_print(verbose_progress, "Creating minimal schedule")
            # Create Min schedule
            min_schedule = scheduler.propose_minimal_schedule()
            initial_sram_limit = scheduler.sram_limit
            if scheduler_options.optimization_strategy == OptimizationStrategy.Size:
                initial_sram_limit = scheduler.min_memory_req

            # Build cascades for Min schedule
            progress_print(verbose_progress, "Building cascades for minimal schedule")
            scheduler.build_cascades_for_min_schedule(min_schedule, max_schedule_template, initial_sram_limit)
            sg.schedule = min_schedule
            scheduler.update_op_memory_snapshot(min_schedule)

            if scheduler_options.optimization_strategy == OptimizationStrategy.Size:
                progress_print(verbose_progress, "Creating schedule optimized for performance")
                # Update sram limit to peak usage from the minimum scheduler when optimizing for Size.
                # Then optimize schedule can be called for both OptimizationStrategy Performance and Size
                # as long the max sram usage is <= scheduler.sram_limit
                scheduler.sram_limit = min_schedule.fast_storage_peak_usage

            # Create an optimized schedule
            sg.schedule = scheduler.optimize_schedule(min_schedule, opt_max_schedule, max_schedule_template)
            scheduler.update_op_memory_snapshot(sg.schedule)

            scheduler.apply_schedule(sg.schedule)
            scheduler.use_fast_storage_for_feature_maps(sg.schedule, scheduler.sram_limit)

            if scheduler_options.optimization_strategy == OptimizationStrategy.Performance and scheduler.evicted_fms:
                progress_print(verbose_progress, "Optimizing weight buffering size")
                # It might be possible to gain performance by reducing
                # weight buffer size and instead fit fms in fast storage
                scheduler.optimize_weight_buffering_size(min_schedule)

            if scheduler_options.verbose_schedule:
                scheduler.print_schedule(sg.schedule)

    progress_print(verbose_progress, "Update memory snapshot for all NPU graphs")
    # Make a full live range calculation starting from the root sg
    _update_memory_snapshot_for_all_npu_graphs(nng, arch, schedulers, verbose_progress)

    progress_print(verbose_progress, "Update tensor allocation")
    # Evaluate schedule
    _update_tensor_allocation(nng, arch, options)
