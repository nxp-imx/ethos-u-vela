# SPDX-FileCopyrightText: Copyright 2021-2023 Arm Limited and/or its affiliates <open-source-office@arm.com>
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
# Groups Operators in a schedule together to form Cascades.
from .live_range import ofm_can_reuse_ifm
from .numeric_util import round_up
from .operation import NpuBlockType
from .operation import Op
from .operation import Padding
from .shape4d import Shape4D

non_cascadable_blocks = (
    NpuBlockType.Default,
    NpuBlockType.VectorProduct,
    NpuBlockType.ReduceSum,
)


class CascadeInfo:
    """Contains metadata about a cascade"""

    def __init__(self, start, end, buffers, mem_usage: int):
        self.start = start
        self.end = end
        self.buffers = buffers
        self.mem_usage = mem_usage


class BufferMap:
    """Caches the buffers seen"""

    def __init__(self):
        self.buffer_map = {}

    def get_buffer(self, producer, consumer, cost):
        assert producer or consumer
        key = (producer, consumer)
        if key not in self.buffer_map:
            # No cached buffer between these two SchedulerOperations
            if consumer is None:
                # There are either no consumers or multiple consumers - FeatureMap needs to be stored in full
                buffer_shape = producer.ofm.shape
                buffer_size = producer.ofm_size_in_bytes()
            elif producer is None:
                # First Op in subgraph or cascade - FeatureMap needs to be stored in full
                buffer_shape = consumer.ifm.shape
                buffer_size = consumer.ifm_size_in_bytes()
            elif producer.requires_full_ofm or consumer.requires_full_ifm:
                # FeatureMap needs to be stored in full
                buffer_shape = max(producer.ofm.shape, consumer.ifm.shape)
                buffer_size = max(producer.ofm_size_in_bytes(), consumer.ifm_size_in_bytes())
            else:
                # Use a rolling buffer
                buffer_shape = rolling_buffer_shape(cost[producer].stripe, cost[consumer].stripe_input)
                buffer_size = buffer_shape.elements() * producer.ofm.dtype.size_in_bytes()

            self.buffer_map[key] = (buffer_shape, buffer_size)

        return self.buffer_map[key]


def rolling_buffer_shape(producer_stripe: Shape4D, consumer_stripe_input: Shape4D) -> Shape4D:
    """Calculates the storage shape of the rolling buffer between two SchedulerOperations in a Cascade"""
    buffer_height = round_up(producer_stripe.height + consumer_stripe_input.height, consumer_stripe_input.height)
    # Striding on the consumer op can result in IFM widths that are narrower than the OFM width of the producer.
    # Therefore, the maximum of the two needs to be used.
    buffer_width = max(producer_stripe.width, consumer_stripe_input.width)
    # Rolling buffers have to conform to NHCWB16 format
    return Shape4D([1, buffer_height, buffer_width, round_up(producer_stripe.depth, 16)])


class CascadeBuilder:
    """Class for grouping SchedulerOperations into cascades"""

    def __init__(self, sched_ops, spilling, non_local_mem_usage=None):
        self.sched_ops = sched_ops
        self.no_cascade = 0
        self.non_local_mem_usage = non_local_mem_usage if non_local_mem_usage else {}
        self.spilling = spilling

    def _is_cascadable(self, sched_op, cost) -> bool:
        """Checks if 'sched_op' can be cascaded"""

        return (
            sched_op.op_type.npu_block_type not in non_cascadable_blocks
            and cost.stripe.height < sched_op.ofm.shape.height
            and sched_op.parent_op.read_offsets[0] is None
            and sched_op.parent_op.read_offsets[1] is None
            and self.elementwise_cascadable(sched_op)
            and not sched_op.parent_op.type == Op.Conv2DBackpropInputSwitchedBias
            and sched_op.parent_op.attrs.get("padding", None) != Padding.TILE
        )

    def _estimate_sram_usage(self, sched_op, cost) -> int:
        """Estimate the SRAM required for the Op if all FeatureMaps are in SRAM"""
        if sched_op.parent_op.type.is_binary_elementwise_op():
            # ifm2 is scalar or constant and will always persist in permanent memory
            ifm2_size = 0
        else:
            ifm2_size = sched_op.ifm2_size_in_bytes()
        if sched_op.requires_full_ifm:
            ifm_size = sched_op.ifm_size_in_bytes()
        else:
            ifm_size = (
                cost.stripe_input.with_depth(round_up(cost.stripe_input.depth, 16)).elements()
                * sched_op.ifm.dtype.size_in_bytes()
            )
        if ofm_can_reuse_ifm(sched_op):
            # ofm will use the ifm buffer to reduce SRAM usage, hence ofm_size = 0
            ofm_size = 0
        elif sched_op.requires_full_ofm:
            ofm_size = sched_op.ofm_size_in_bytes()
        else:
            ofm_size = (
                cost.stripe.with_depth(round_up(cost.stripe.depth, 16)).elements() * sched_op.ofm.dtype.size_in_bytes()
            )

        return ifm_size + ifm2_size + ofm_size + self.non_local_mem_usage.get(sched_op, 0)

    @staticmethod
    def elementwise_cascadable(sched_op):
        """Check if the elementwise can be cascaded."""

        if sched_op.parent_op.type.is_binary_elementwise_op():
            ifm = sched_op.parent_op.ifm
            ifm2 = sched_op.parent_op.ifm2
            ofm = sched_op.parent_op.ofm

            # IFM must be non-constant/non-scalar/non-broadcast
            ifm_cascadable = not (ifm.is_const or ifm.is_scalar or ifm.is_broadcast(ofm))

            # IFM2 must be constant or scalar
            ifm2_cascadable = ifm2.is_const or ifm2.is_scalar

            return ifm_cascadable and ifm2_cascadable
        else:
            return True

    def build_cascades(self, ref_schedule, fallback_schedule, guiding_mem_limit):
        ref_cost = ref_schedule.cost_map
        fallback_cost = fallback_schedule.cost_map
        cost = {}
        cascade_map = {}
        buffers = BufferMap()

        # Peak memory usage so far - updated continously, unless dedicated SRAM where this is a hard limit
        peak_sram_usage = guiding_mem_limit

        idx = 0
        while idx < len(self.sched_ops):
            op = self.sched_ops[idx]
            if op in cost:
                # Already processed this Op
                idx += 1
                continue

            if not self._is_cascadable(op, ref_cost[op]):
                # Op is not a candidate for cascading - assign fallback cost
                cost[op] = fallback_cost[op]
                if not self.spilling:
                    peak_sram_usage = max(self._estimate_sram_usage(op, fallback_cost[op]), peak_sram_usage)
                idx += 1
                continue

            # Propose a cascade starting with this Op
            cascade_start = op.index
            # Keep track of which Ops are in the proposed cascade as well as the best cascade so far
            ops_in_cascade = [op]
            ops_in_best_cascade = [op]
            # Get the size of the weight buffer(s)
            weight_buffer = sum(tens.storage_size() for tens in ref_cost[op].buffered_weight_tensors)

            # The first IFM needs to be stored in full
            cascade_ifm_size = op.ifm_size_in_bytes() if not self.spilling else 0

            # Sum of all intermediate cascade buffers (including weight buffers)
            cascade_buffers = weight_buffer
            # Best cascade size - Initially it's the fallback cost of the first Op in the cascade
            best_cascade_size = self._estimate_sram_usage(op, fallback_cost[op])

            # Op is the producer of the OFM consumed by the next Op to consider
            producer = op
            while True:
                dependants = producer.get_dependants()
                if len(dependants) != 1:
                    # producer is either the last Op in the schedule or the start of a branch
                    break

                current_op = dependants[0]
                if (
                    current_op in cost
                    or current_op not in ref_cost
                    or not self._is_cascadable(current_op, ref_cost[current_op])
                    or producer.ofm.shape != current_op.ifm.shape
                    or current_op.requires_full_ifm
                    or producer.requires_full_ofm
                ):
                    # Current op has already been processed or cannot be cascaded
                    break

                if producer.index + 1 != current_op.index:
                    # Cascading is possible, but requires reordering of operations in the schedule,
                    # this is currently not supported
                    break

                # Get the size of the FeatureMap buffers between current and neighbouring Ops
                op_full_ofm = current_op.ofm_size_in_bytes()
                _, op_ifm_buffer = buffers.get_buffer(producer, current_op, ref_cost)

                # Get the size of the weight buffer(s)
                op_weight_buffer = sum(tens.storage_size() for tens in ref_cost[current_op].buffered_weight_tensors)

                # Calculate the uncascaded memory requirement for current Op
                uncascaded_sram_usage = self._estimate_sram_usage(current_op, fallback_cost[current_op])

                # Add current Op to cascade
                ops_in_cascade.append(current_op)

                # Increase the accumulated intermediate buffers in the cascade
                cascade_buffers += op_ifm_buffer + op_weight_buffer

                if self.spilling:
                    # For Dedicated SRAM only the intermediate buffers are in SRAM
                    if uncascaded_sram_usage < peak_sram_usage or cascade_buffers > peak_sram_usage:
                        # Cascade until an Op fits in its entirety or the accumulated buffers no longer fit
                        break
                    else:
                        # Any addition to the cascade that fits is the new best cascade for Dedicated SRAM
                        ops_in_best_cascade = [op for op in ops_in_cascade]
                        best_cascade_size = cascade_buffers

                else:
                    # Calculate the total size of the current cascade including non local mem usage
                    cascade_size = (
                        cascade_ifm_size + cascade_buffers + op_full_ofm + self.non_local_mem_usage.get(op, 0)
                    )

                    # Determine if cascading search should stop
                    if (
                        uncascaded_sram_usage < peak_sram_usage
                        and best_cascade_size < peak_sram_usage
                        or (cascade_ifm_size + cascade_buffers) > best_cascade_size
                    ):
                        # Both the existing cascade and current Op fits or
                        # not possible to reduce cascade size any further
                        break

                    """
                    One of two conditions will update the best cascade:

                    - cascade_size < best_cascade_size or
                    - cascade_size < uncascaded_sram_usage

                    The last condition is illustrated below, showing an example where it is
                    better to choose a larger cascade_size (with more OPs) because it will
                    use less total SRAM usage.

                    For simplicity, all featuremaps have same size.

                    Cascade OP1-OP2, OP3 is standalone

                                ->  |OP1| -> roll buffer -> |OP2| -> FM -> |OP3| -> FM
                               /
                    |OP0| -> FM
                               \
                                ->  ....


                    best_cascade_size    : FM + roll buffer + FM
                    uncascaded_sram_usage: FM + FM + FM

                    compared with:

                    Cascade OP1-OP3

                                ->  |OP1| -> roll buffer -> |OP2| -> roll buffer -> |OP3| -> FM
                               /
                    |OP0| -> FM
                               \
                                ->  ....


                    cascade_size         : FM + roll buffer + roll buffer + FM


                    So, for this use case the comparison will be

                    (FM + roll buffer + roll buffer + FM) <  (FM + roll buffer + FM) or
                    (FM + roll buffer + roll buffer + FM) <  (FM + FM + FM)

                    hence, better to choose Cascade OP1-OP3 in this case.
                    """
                    if cascade_size < best_cascade_size or cascade_size < uncascaded_sram_usage:
                        best_cascade_size = cascade_size
                        ops_in_best_cascade = [op for op in ops_in_cascade]

                producer = current_op

            if len(ops_in_best_cascade) > 1:
                # A cascade was created - assign cascade and ref_cost to all of the Ops
                cascade_end = cascade_start + (len(ops_in_best_cascade) - 1)
                buffers_in_cascade = {}
                prev_op = None
                for cascaded_op in ops_in_best_cascade:
                    assert cascade_start <= cascaded_op.index <= cascade_end
                    cost[cascaded_op] = ref_cost[cascaded_op]
                    cost[cascaded_op].cascade = cascade_end
                    if prev_op:
                        rolling_buffer_shape, _ = buffers.get_buffer(prev_op, cascaded_op, ref_cost)
                        buffers_in_cascade[cascaded_op] = rolling_buffer_shape

                    prev_op = cascaded_op

                # Create a CascadeInfo for the cascade, only store the actual size used by
                # the cascade so non local usage is removed. This is done in order to be
                # able to calculate the correct non local usage in the scheduler when
                # optimizing the sub schedules.
                cascade_map[cascade_end] = CascadeInfo(
                    cascade_start,
                    cascade_end,
                    buffers_in_cascade,
                    best_cascade_size - self.non_local_mem_usage.get(op, 0),
                )
                if not self.spilling:
                    # Update peak memory usage
                    peak_sram_usage = max(best_cascade_size, peak_sram_usage)
            else:
                # Assign fallback cost to the initial Op
                cost[op] = fallback_cost[op]
                if not self.spilling:
                    peak_sram_usage = max(self._estimate_sram_usage(op, fallback_cost[op]), peak_sram_usage)

        # Update costing and cascade information for the ref_schedule
        ref_schedule.cost_map = cost
        ref_schedule.cascades = cascade_map
        return ref_schedule
