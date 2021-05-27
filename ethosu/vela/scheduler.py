# Copyright (C) 2021 Arm Limited or its affiliates. All rights reserved.
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
import copy
from enum import auto
from enum import IntEnum
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple

from . import live_range
from . import npu_performance
from . import tensor_allocation
from . import weight_compressor
from .architecture_allocator import ArchitectureBlockConfig
from .architecture_allocator import find_block_config
from .architecture_allocator import get_ifm_area_required
from .architecture_allocator import to_upscale
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
from .numeric_util import round_down
from .numeric_util import round_up
from .operation import NpuBlockType
from .operation import Op
from .shape4d import Shape4D
from .tensor import MemArea
from .tensor import MemType
from .tensor import Tensor
from .tensor import TensorFormat
from .tensor import TensorPurpose
from .tensor import TensorSubPurpose


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
        self.npu_weights_tensor = None
        self.buffered_weight_tensor = None
        self.cycles = None
        self.slack_buffering_cycles = 0
        self.slack_buffering_memory = 0
        self.full_weight_transfer_cycles = 0

    def copy(self):
        res = SchedulerOpInfo(self.block_config, self.weights_size, self.stripe_input, self.stripe_input2, self.stripe,)
        res.cascade = self.cascade
        return res

    def __str__(self):
        res = f"\t\tBlock Config = {self.block_config}\n"
        res += f"\t\tOFM Block = {self.block_config.ofm_block}\n"
        res += f"\t\tIFM Stripe   = {self.stripe_input}\n"
        res += f"\t\tIFM2 Stripe  = {self.stripe_input2}\n"
        res += f"\t\tOFM Stripe   = {self.stripe}\n"
        res += f"\t\tEncoded Weights = {self.npu_weights_tensor and len(self.npu_weights_tensor.buffer)} bytes\n"
        res += (
            f"\t\tWeight buffer = {self.buffered_weight_tensor and self.buffered_weight_tensor.storage_size()} bytes\n"
        )
        res += f"\t\tDepth slices = {self.ofm_depth_slices}\n"
        res += f"\t\tAssigned Cascade = {self.cascade}"
        return res


class SchedulerOptions:
    """Contains options for the Scheduler"""

    def __init__(
        self, optimization_strategy, sram_target, verbose_schedule,
    ):
        self.optimization_strategy = optimization_strategy
        self.optimization_sram_limit = sram_target
        self.verbose_schedule = verbose_schedule

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
        self.resampling_mode = ps.primary_op.ifm.resampling_mode
        self.uses_scalar = ps.primary_op.ifm2 is not None and (
            ps.primary_op.ifm.shape == [] or ps.primary_op.ifm2.shape == []
        )
        self.ifm_ublock = arch.ifm_ublock

        self.ifm = SchedulerTensor(ps.ifm_shapes[0], ps.ifm_tensor.dtype, ps.ifm_tensor.mem_area, ps.ifm_tensor.format,)

        self.ifm2 = None
        if ps.ifm2_tensor:
            self.ifm2 = SchedulerTensor(
                ps.ifm_shapes[1], ps.ifm2_tensor.dtype, ps.ifm2_tensor.mem_area, ps.ifm2_tensor.format,
            )

        self.ofm = SchedulerTensor(ps.ofm_shapes[0], ps.ofm_tensor.dtype, ps.ofm_tensor.mem_area, ps.ofm_tensor.format,)

        # Input volume width and height required to produce the smallest possible stripe
        self.min_stripe_input_w, self.min_stripe_input_h = self._calculate_min_stripe_input()

        # Flags that marks whether this SchedulerOperation requires full IFM/OFM
        self.requires_full_ifm = False
        self.requires_full_ifm2 = False
        self.requires_full_ofm = False

        self.index = 0

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
        ifm2_shape = self.ifm2 and self.ifm2.shape
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
            scheduler_op_info.npu_weights_tensor = weight_compressor.encode_weight_and_scale_tensor(
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

        return get_ifm_area_required(ofm_shape_to_produce, self.kernel, to_upscale(self.resampling_mode))

    def _calculate_min_stripe_input(self) -> Shape4D:
        # Calculate the input volume required height and width for the smallest possible stripe (h,w = 1,1)
        min_stripe = self.ofm.shape.with_hw(1, 1)
        return self._get_stripe_input_requirement(min_stripe)

    def _get_block_config(
        self, ifm_shape: Shape4D, ifm2_shape: Optional[Shape4D], uses_scalar: bool, ofm_shape: Shape4D
    ) -> ArchitectureBlockConfig:
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
        self.memory_snapshot = None

    @property
    def name(self):
        return f"{self.sg.name}_{self.label}"


class Scheduler:
    """Main class of the Vela Scheduling"""

    def __init__(self, nng: Graph, sg: Subgraph, arch: ArchitectureFeatures, options: SchedulerOptions):
        self.nng = nng
        self.sg = sg
        self.arch = arch
        self.sched_ops: List(SchedulerOperation) = []
        self.max_schedule = None
        self.scheduler_options = options

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
                    if output.purpose != TensorPurpose.FeatureMap:
                        continue
                    if not output.needs_linear_format:
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
                if ps.ifm_tensor.needs_linear_format:
                    op.requires_full_ifm = True
                if ps.ifm2_tensor and ps.ifm2_tensor.needs_linear_format:
                    op.requires_full_ifm2 = True
                if ps.ofm_tensor.needs_linear_format or ps.primary_op.memory_function == Op.ConcatSliceWrite:
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

        for op in self.sched_ops:
            cost = op.create_scheduler_info(self.nng, op.ofm.shape)
            cost.cycles = self.estimate_op_performance(op, cost.block_config, op.ofm.shape.depth)
            schedule.cost_map[op] = cost

        return schedule

    def update_op_memory_snapshot(self, schedule: Schedule):
        memories_list = [(self.arch.fast_storage_mem_area, set((MemType.Scratch, MemType.Scratch_fast)))]

        # Collect live ranges from tensors
        lr_graph = live_range.LiveRangeGraph()
        for mem_area, mem_type_set in memories_list:
            live_range.extract_live_ranges_from_cascaded_passes(
                self.nng.get_root_subgraph(), mem_area, mem_type_set, False, lr_graph, Tensor.AllocationQuantum,
            )

        # Populate time-array with memory used by live ranges
        temporal_usage = lr_graph.get_temporal_memory_usage(self.arch.fast_storage_mem_area)
        schedule.memory_snapshot = temporal_usage

        # Set the peak memory usage
        schedule.fast_storage_peak_usage = max(temporal_usage, default=0)

    def estimate_op_performance(self, op: SchedulerOperation, block_config, ofm_depth):
        query = npu_performance.PerformanceQuery(op.op_type.npu_block_type)
        query.ifm_shape = op.ifm.shape
        query.ifm_memory_area = op.ifm.mem_area
        query.ifm_bits = op.ifm.dtype.size_in_bits()
        query.ifm_format = op.ifm.format
        query.ifm2_shape = op.ifm2 and op.ifm2.shape
        query.ifm2_memory_area = op.ifm2 and op.ifm2.mem_area
        query.ifm2_bits = op.ifm2 and op.ifm2.dtype.size_in_bits()
        query.ifm2_format = op.ifm2 and op.ifm2.format
        query.ofm_shape = op.ofm.shape.with_depth(ofm_depth)
        query.ofm_memory_area = op.ofm.mem_area
        query.ofm_bits = op.ofm.dtype.size_in_bits()
        query.ofm_format = op.ofm.format
        if op.parent_op.bias:
            query.const_shape = Shape4D(1, 1, 1, op.ofm.shape.depth)
            query.const_memory_area = self.arch.fast_storage_mem_area

        query.kernel = op.kernel
        query.config = block_config

        return npu_performance.measure_cycle_cost(self.arch, op.op_type, op.activation and op.activation.op_type, query)

    def propose_schedule_buffering(self, ref_schedule: Schedule):
        """Create a buffered schedule"""
        buffered_schedule = Schedule(self.sg, f"{ref_schedule.label}_BUFFERED")
        staging_limit_bytes = self.scheduler_options.optimization_sram_limit

        prev_op = None
        for sched_op in self.sched_ops:
            if sched_op not in ref_schedule.cost_map:
                # sched_op is not part of this sub-schedule - skip
                continue

            self.propose_operator_buffering(sched_op, prev_op, buffered_schedule, ref_schedule, staging_limit_bytes)
            prev_op = sched_op

        return buffered_schedule

    def propose_operator_buffering(
        self,
        sched_op: SchedulerOperation,
        prev_op: SchedulerOperation,
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
            self.propose_weight_buffering(
                sched_op.parent_op.weights,
                sched_op.parent_op.bias,
                sched_op,
                prev_op,
                buffered_schedule,
                ref_schedule,
                cost.slack_buffering_memory,
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
        full_weights = weight_compressor.encode_weight_and_scale_tensor(
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
            return

        encoded_weights = full_weights

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
                prebuffer_depth = int(ref_cost.stripe.depth * prebuffer_ratio)

                # Round prebuffering down to nearest valid split depth
                prebuffer_depth = int(max(16, round_down(prebuffer_depth, ArchitectureFeatures.OFMSplitDepth)))

                while True:
                    buffering_depth = max(cost.block_config.ofm_block.depth, prebuffer_depth)

                    # Clamp buffering to the double buffering limit
                    buffering_bytes = (buffering_depth / ref_cost.stripe.depth) * full_weights_bytes
                    if buffering_bytes > half_buffer_limit:
                        buffering_depth = (half_buffer_limit / full_weights_bytes) * ref_cost.stripe.depth
                        buffering_depth = int(max(16, round_down(prebuffer_depth, ArchitectureFeatures.OFMSplitDepth)))

                    # Create list of depth slices
                    depth_slices = [0]
                    if prebuffer_depth < ref_cost.stripe.depth:
                        depth_slices += list(range(prebuffer_depth, ref_cost.stripe.depth, buffering_depth))
                    depth_slices.append(ref_cost.stripe.depth)

                    # Encode weights based depth slices
                    cost.ofm_depth_slices = depth_slices
                    encoded_weights = weight_compressor.encode_weight_and_scale_tensor(
                        self.arch,
                        sched_op.parent_op,
                        weight_tensor,
                        scale_tensor,
                        sched_op.kernel,
                        cost.block_config,
                        cost.ofm_depth_slices,
                    )

                    # Chosen buffering might not fit at all, iterate until it does
                    # or until the minimum usable slice size is reached
                    if (
                        encoded_weights.max_range_bytes <= half_buffer_limit
                        or prebuffer_depth == ArchitectureFeatures.OFMSplitDepth
                    ):
                        break

                    prebuffer_depth = round_up(prebuffer_depth // 2, ArchitectureFeatures.OFMSplitDepth)

                # Calculate cycles required to run the last op for use as future slack
                tail_cycles = self.estimate_op_performance(
                    sched_op, cost.block_config, depth_slices[-1] - depth_slices[-2]
                )
                cost.slack_buffering_cycles = tail_cycles.op_cycles

        # Determine whether the weights need to be double buffered
        weight_buffer_size = min(len(encoded_weights.buffer), encoded_weights.max_range_bytes)

        # Only buffer weights if there's still space left for the buffer
        if weight_buffer_size <= buffer_limit_bytes:
            assert weight_buffer_size % 16 == 0
            # Determine whether to double buffer or single buffer
            if (weight_buffer_size * 2 <= buffer_limit_bytes) and (weight_buffer_size < len(encoded_weights.buffer)):
                weight_buffer_size = weight_buffer_size * 2
                weight_tensor_purpose = TensorSubPurpose.DoubleBuffer
            else:
                weight_tensor_purpose = TensorSubPurpose.Standard

            cost.buffered_weight_tensor = Tensor(
                [1, 1, 1, weight_buffer_size], DataType.uint8, weight_tensor.name + "_buffer"
            )
            cost.buffered_weight_tensor.src_tensor = encoded_weights
            cost.buffered_weight_tensor.mem_area = self.arch.fast_storage_mem_area
            cost.buffered_weight_tensor.mem_type = MemType.Scratch_fast
            cost.buffered_weight_tensor.purpose = TensorPurpose.Weights
            cost.buffered_weight_tensor.sub_purpose = weight_tensor_purpose
            if ref_cost.cascade == 0:
                # Determine if the lifetime can be extended and pre-buffer weights under the previous operation
                cost.buffered_weight_tensor.pre_buffer = weight_buffer_size < slack_memory

            cost.slack_buffering_memory -= weight_buffer_size
        else:
            # Don't slice or buffer - use the whole depth from persistent storage
            cost.ofm_depth_slices = ofm_full_depth_slices
            encoded_weights = full_weights

        cost.npu_weights_tensor = encoded_weights

    def propose_minimal_schedule(self) -> Schedule:
        """Proposes scheduling parameters where every operator is subdivided into the smallest stripe that satisfies the
        next operators stride"""
        min_schedule = Schedule(self.sg, "MIN")
        cost_map = min_schedule.cost_map

        # Keep track of the previous Op - which consumes the current Op's OFM
        prev_op = None
        for sched_op in reversed(self.sched_ops):
            min_stripe_height = prev_op.kernel.stride.y if prev_op else 1
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

            # Copy the weight buffering from the reference schedule
            cost.buffered_weight_tensor = ref_cost[sched_op].buffered_weight_tensor

            # Estimate performance
            cost.cycles = self.estimate_op_performance(sched_op, cost.block_config, sched_op.ofm.shape.depth)
            striped_schedule.cost_map[sched_op] = cost

            # Calculate the preceeding Op's stripe
            stripe = sched_op.ifm.shape.with_height(stripe.height * sched_op.kernel.stride.y)

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
                # Non-local memory usage is already included in the cascade_info
                peak_mem_usage = max(cascade_info.mem_usage, peak_mem_usage)
            else:
                # This Op is not part of a cascade - calculate the memory usage
                op_weight_buffer = 0
                if cost[sched_op].buffered_weight_tensor:
                    op_weight_buffer = cost[sched_op].buffered_weight_tensor.storage_size()

                op_mem_usage = (
                    sched_op.ifm_size_in_bytes()
                    + sched_op.ofm_size_in_bytes()
                    + op_weight_buffer
                    + non_local_mem_usage.get(sched_op, 0)
                )
                peak_mem_usage = max(op_mem_usage, peak_mem_usage)

        return peak_mem_usage

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
        # included in a cascade or not
        persistent_initial_ifm = (
            sub_schedule_ops[0].ifm_size_in_bytes() if len(sub_schedule_ops[0].ifm.connection.consumers) > 1 else 0
        )
        # Calculate non-local-mem-usage per Operator
        non_local_mem_usage = {}
        for idx, sched_op in enumerate(sub_schedule_ops):
            non_local_mem_usage[sched_op] = mem_usage_parallel_to_sub_schedule
            if idx != 0:
                non_local_mem_usage[sched_op] += persistent_initial_ifm

        cascade_builder = CascadeBuilder(sub_schedule_ops, self.arch.is_spilling_enabled(), non_local_mem_usage)

        # Start by adding buffering
        buffered_sub_schedule = self.propose_schedule_buffering(sub_schedule)
        # Copy the cascades over from the unbuffered-schedule
        buffered_sub_schedule.cascades = sub_schedule.cascades

        # Generate the possible stripings for the final Op in the sub-schedule
        final_ofm_shape = sub_schedule_ops[-1].ofm.shape
        possible_stripes = [
            final_ofm_shape.with_height(stripe_h) for stripe_h in range(1, final_ofm_shape.height // 2 + 1)
        ]

        # Propose different striping - the possible stripes are proposed similarly to a binary search
        best_schedule = buffered_sub_schedule
        iteration = 0
        while len(possible_stripes) > 1:
            proposed_stripe = possible_stripes[len(possible_stripes) // 2]
            proposed_schedule = self.propose_schedule_striping(
                proposed_stripe, f"OPTIMIZED_{iteration}", buffered_sub_schedule
            )

            cascade_builder.build_cascades(proposed_schedule, max_template, memory_limit)

            # Check if proposal fits
            proposed_schedule_mem_usage = self.estimate_schedule_memory_usage(proposed_schedule, non_local_mem_usage)
            if (proposed_schedule_mem_usage) <= memory_limit:
                # Remove all possible stripes smaller than this
                possible_stripes = possible_stripes[len(possible_stripes) // 2 :]
                best_schedule = proposed_schedule
                if not proposed_schedule.cascades:
                    # No cascading required - early exit
                    break
            else:
                # Proposal doesn't fit within the limit - remove all possible stripes larger than this
                possible_stripes = possible_stripes[: len(possible_stripes) // 2]

            iteration += 1

        return best_schedule

    def optimize_schedule(
        self, schedule: Schedule, max_sched: Schedule, max_template: Schedule, options: SchedulerOptions,
    ) -> Schedule:
        """Extracts sub-schedules based on the cascades and optimizes them and applies them to the final schedule"""
        sram_limit = options.optimization_sram_limit
        if max_sched.fast_storage_peak_usage < sram_limit and not self.arch.is_spilling_enabled():
            # Maximum performance schedule fits within the SRAM target
            return max_sched

        # Extract the cascades
        cascades = [cascade for cascade in schedule.cascades.values()]
        for cascade_info in cascades:
            # Remove existing cascade from schedule
            del schedule.cascades[cascade_info.end]
            # Optimize the sub-schedule in this cascade
            opt_sub_schedule = self.optimize_sub_schedule(cascade_info, schedule, max_template, sram_limit)
            # Update the sub-schedule Op and cascade costs to the full schedule
            schedule.cost_map.update(opt_sub_schedule.cost_map)
            schedule.cascades.update(opt_sub_schedule.cascades)

        # Update memory snapshot
        self.sg.schedule = schedule
        self.update_op_memory_snapshot(schedule)
        # Propose schedule buffering to the optimized schedule
        optimized_sched = self.propose_schedule_buffering(schedule)
        # Copy the cascade's metadata from the unbuffered schedule
        optimized_sched.cascades = schedule.cascades
        return optimized_sched

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
            if op_info.buffered_weight_tensor:
                op_info.buffered_weight_tensor.src_tensor = op_info.npu_weights_tensor

    def use_fast_storage_for_feature_maps(self, schedule: Schedule, memory_limit: int):
        if self.arch.fast_storage_mem_area == self.arch.feature_map_storage_mem_area:
            return

        # Force all OFMs to fast-storage
        for sched_op in self.sched_ops:
            cost = schedule.cost_map[sched_op]
            if cost.cascade == 0:
                if sched_op.get_dependants():
                    ofm_tens = sched_op.ofm.connection.parent_tens
                    if not any(cons is None for cons in ofm_tens.consumer_list):
                        ofm_tens.mem_area = self.arch.fast_storage_mem_area
                        ofm_tens.mem_type = MemType.Scratch_fast

        # Collect live ranges from tensors
        memories_list = [(self.arch.fast_storage_mem_area, set((MemType.Scratch, MemType.Scratch_fast)))]
        lr_graph = live_range.LiveRangeGraph()
        for mem_area, mem_type_set in memories_list:
            live_range.extract_live_ranges_from_cascaded_passes(
                self.nng.get_root_subgraph(), mem_area, mem_type_set, False, lr_graph, Tensor.AllocationQuantum,
            )

        # Iterate over live ranges and evict tensors that doesn't fit
        fast_storage_snapshot = lr_graph.get_temporal_memory_usage(self.arch.fast_storage_mem_area)
        for lr in lr_graph.lrs:
            if (
                lr.mem_area == self.arch.fast_storage_mem_area
                and max(fast_storage_snapshot[lr.start_time : lr.end_time + 1]) > memory_limit
            ):
                # Evict tensor to DRAM
                for tens in lr.tensors:
                    if tens.purpose == TensorPurpose.FeatureMap and tens.sub_purpose == TensorSubPurpose.Standard:
                        # Can only evict unbuffered FeatureMaps
                        tens.mem_area = self.arch.feature_map_storage_mem_area
                        tens.mem_type = MemType.Scratch
                        # Adjust the snapshot
                        fast_storage_snapshot[lr.start_time : lr.end_time + 1] -= lr.size

    def move_constant_data(self):
        """Determine if  data, can be moved from permanent storage to another memory area. A move
        will generate a DMA command in the high-level command stream"""
        for sched_op in self.sched_ops:
            parent_op = sched_op.parent_op
            is_lut_used = any(inp.purpose == TensorPurpose.LUT for inp in parent_op.inputs)
            max_ifm_shram_avail = (
                (self.arch.available_shram_banks(is_lut_used) - self.arch.shram_reserved_output_banks)
                * self.arch.shram_bank_size
                // 2
            )

            for idx, tens in enumerate(parent_op.inputs):
                if tens.mem_type not in (MemType.Scratch, MemType.Scratch_fast):
                    # Tensor is in permanent storage
                    # Only when permanent storage differs from feature map storage, there is a point moving the data
                    if (
                        tens.mem_area in self.arch.permanent_storage_mem_area
                        and self.arch.permanent_storage_mem_area != self.arch.feature_map_storage_mem_area
                    ) or tens.purpose == TensorPurpose.LUT:
                        if tens.purpose == TensorPurpose.LUT or (
                            tens.purpose == TensorPurpose.FeatureMap
                            and sched_op.op_type.is_binary_elementwise_op()
                            and tens.shape != []
                            and sched_op.ifm.shape != sched_op.ofm.shape
                            and tens.storage_size() > max_ifm_shram_avail
                        ):
                            only_vector_product_consumers = all(
                                oper and oper.type.npu_block_type == NpuBlockType.VectorProduct
                                for oper in tens.consumers()
                            )

                            if (not only_vector_product_consumers) or tens.purpose == TensorPurpose.LUT:
                                new_tens = tens.clone_into_fast_storage(self.arch)
                                if tens.purpose == TensorPurpose.LUT:
                                    new_tens.mem_area = MemArea.Shram

                                new_tens.consumer_list.append(parent_op)
                                parent_op.inputs[idx] = new_tens
                                sched_op.parent_ps.inputs[idx] = new_tens

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

        print(f"\tCascades:")
        for i, cascade in enumerate(schedule.cascades.values()):
            print(f"\t\t{i}: {cascade.start} -> {cascade.end}, size: {cascade.mem_usage}")


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
            cpu_tensor_alignment=options.cpu_tensor_alignment,
        )


def schedule_passes(nng: Graph, arch: ArchitectureFeatures, options, scheduler_options: SchedulerOptions):
    """Entry point for the Scheduler"""
    # Initialize CPU subgraphs
    schedulers = dict()
    # Initialize schedulers with max schedule. Only schedule NPU subgraphs
    for sg in nng.subgraphs:
        if sg.placement != PassPlacement.Npu:
            # Create cascaded passes for CPU Ops
            cascaded_passes = []
            for idx, ps in enumerate(sg.passes):
                cps = CascadedPass(
                    ps.name, SchedulingStrategy.WeightStream, ps.inputs, [], ps.outputs, [ps], ps.placement, False,
                )

                cps.time = idx
                ps.cascade = cps
                cascaded_passes.append(cps)

            sg.cascaded_passes = cascaded_passes
        else:
            # Npu subgraph - create schedule
            scheduler = Scheduler(nng, sg, arch, scheduler_options)
            schedulers[sg] = scheduler

            scheduler.create_scheduler_representation(arch)
            sg.sched_ops = scheduler.sched_ops
            scheduler.move_constant_data()

            # Create the Max schedule template
            max_schedule_template = scheduler.create_initial_schedule()
            scheduler.max_schedule = max_schedule_template

            # Create the optimimised Max schedule
            sg.schedule = max_schedule_template
            scheduler.update_op_memory_snapshot(max_schedule_template)
            opt_max_schedule = scheduler.propose_schedule_buffering(max_schedule_template)
            sg.schedule = opt_max_schedule
            scheduler.update_op_memory_snapshot(opt_max_schedule)

            # Create Min schedule
            min_schedule = scheduler.propose_minimal_schedule()
            initial_sram_limit = scheduler_options.optimization_sram_limit
            if scheduler_options.optimization_strategy == OptimizationStrategy.Size:
                initial_sram_limit = scheduler.min_memory_req

            cascade_builder = CascadeBuilder(scheduler.sched_ops, arch.is_spilling_enabled())
            cascade_builder.build_cascades(min_schedule, max_schedule_template, initial_sram_limit)
            sg.schedule = min_schedule
            scheduler.update_op_memory_snapshot(min_schedule)

            if scheduler_options.optimization_strategy == OptimizationStrategy.Performance:
                # Create an optimized schedule
                sg.schedule = scheduler.optimize_schedule(
                    min_schedule, opt_max_schedule, max_schedule_template, scheduler_options
                )
                scheduler.update_op_memory_snapshot(sg.schedule)

            scheduler.apply_schedule(sg.schedule)
            scheduler.use_fast_storage_for_feature_maps(sg.schedule, scheduler_options.optimization_sram_limit)

            if scheduler_options.verbose_schedule:
                scheduler.print_schedule(sg.schedule)

    # Evaluate schedule
    _update_tensor_allocation(nng, arch, options)
