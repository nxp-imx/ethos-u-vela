# Copyright (C) 2020-2021 Arm Limited or its affiliates. All rights reserved.
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
# Description:
# The scheduler costs various strategies for scheduling the network in order to select the block configuration.
import copy
import enum
from functools import lru_cache

import numpy as np

from . import live_range
from . import npu_performance
from . import stats_writer
from .data_type import DataType
from .high_level_command_stream_generator import calc_allowed_ofm_ifm_overlap_for_pass_list
from .nn_graph import CascadedPass
from .nn_graph import PassPlacement
from .nn_graph import SchedulerRewrite
from .nn_graph import SchedulingStrategy
from .npu_performance import make_bandwidth_array
from .npu_performance import make_cycles_array
from .npu_performance import make_metrics_arrays
from .npu_performance import PassCycles
from .operation import NpuBlockType
from .operation import Op
from .operation import Operation
from .shared_buffer_allocation import find_block_configs_suitable_for_pass_and_shared_buffer
from .shared_buffer_allocation import shared_buffer_allocation_for_pass_and_block_config
from .tensor import MemArea
from .tensor import MemType
from .tensor import TensorFormat
from .tensor import TensorPurpose
from .tensor import TensorSubPurpose


class ParetoMetric(enum.Enum):
    BwCycMem = 1
    BwCycMemBlkH = 2

    def __str__(self):
        return self.name


class SchedulerOptions:
    def __init__(
        self,
        use_cascading=True,
        verbose_schedule=False,
        verbose_pareto_frontier_schedules=False,
        use_ifm_streaming=True,
        pareto_metric=ParetoMetric.BwCycMem,
        use_nhcwb16_between_cascaded_passes=True,
        cache_bias_scale_tensor=True,
    ):
        self.use_cascading = use_cascading
        self.verbose_schedule = verbose_schedule
        self.verbose_pareto_frontier_schedules = verbose_pareto_frontier_schedules
        self.use_ifm_streaming = use_ifm_streaming
        self.pareto_metric = pareto_metric
        self.use_nhcwb16_between_cascaded_passes = use_nhcwb16_between_cascaded_passes
        self.cache_bias_scale_tensor = cache_bias_scale_tensor

    def __str__(self):
        return type(self).__name__ + ": " + str(self.__dict__)

    __repr__ = __str__


class Strategy:
    __slots__ = "strat", "param", "passes", "block_configs", "rewrite_list", "bws", "macs", "cycles", "sram_used"

    def __init__(self, strat, param, passes, block_configs, rewrite_list, bws, macs, cycles, sram_used):
        self.strat = strat
        self.param = param
        self.passes = passes
        self.block_configs = block_configs
        self.rewrite_list = (
            rewrite_list  # list of (SchedulerRewrite, Tensor, new sub purpose, purpose param a, purpose param b, pass)
        )
        self.bws = bws
        self.macs = macs
        self.cycles = cycles
        self.sram_used = sram_used

    def __eq__(self, other):
        if self.strat != other.strat:
            return False
        if self.param != other.param:
            return False
        if self.block_configs != other.block_configs:
            return False
        if self.passes != other.passes:
            return False
        if (self.bws != other.bws).any():
            return False
        if self.macs != other.macs:
            return False
        if (self.cycles != other.cycles).any():
            return False
        if self.sram_used != other.sram_used:
            return False
        return True

    def empty(self):
        return not self.passes

    def key(self):
        return self.passes[-1]

    def clone(self):
        return Strategy(
            self.strat,
            self.param,
            self.passes,
            self.block_configs,
            self.rewrite_list,
            self.bws,
            self.macs,
            self.cycles,
            self.sram_used,
        )

    def __str__(self):
        return "<scheduler.Strategy: %s %s %s %s %s %s %s>" % (
            self.strat,
            self.passes,
            self.rewrite_list,
            self.bws,
            self.macs,
            self.cycles,
            self.sram_used,
        )

    __repr__ = __str__


class StrategySet:
    __slots__ = "strats", "bws", "macs", "cycles", "max_sram_used", "total_sram_used"

    def __init__(self, strats=None):
        if strats is None:
            strats = dict()
        self.strats = strats  # final pass in packed pass -> Strategy
        self.bws, self.macs, self.cycles = make_metrics_arrays()
        self.max_sram_used = 0
        self.total_sram_used = 0

    def update_statistics(self):
        self.bws = make_bandwidth_array()
        self.max_sram_used = 0
        for ps, strat in self.strats.items():
            self.bws += strat.bws
            self.macs += strat.macs
            self.cycles += strat.cycles
            self.max_sram_used = max(self.max_sram_used, strat.sram_used)
            self.total_sram_used += strat.sram_used

    def clone_add_strategy(self, new_strat):
        key = new_strat.key()
        if key in self.strats:
            assert new_strat == self.strats[key]
            return self
        else:
            new_strats = dict(self.strats)
            new_strats[key] = new_strat
            new_set = StrategySet(new_strats)
            new_set.bws = self.bws + new_strat.bws
            new_set.macs = self.macs + new_strat.macs
            new_set.cycles = self.cycles + new_strat.cycles
            new_set.max_sram_used = max(self.max_sram_used, new_strat.sram_used)
            new_set.total_sram_used = self.total_sram_used + new_strat.sram_used
            return new_set

    def __eq__(self, other):
        if (self.bws != other.bws).any():
            return False
        if self.macs != other.macs:
            return False
        if (self.cycles != other.cycles).any():
            return False
        if self.max_sram_used != other.max_sram_used:
            return False
        if self.total_sram_used != other.total_sram_used:
            return False
        if self.strats != other.strats:
            return False
        return True

    def __str__(self):
        return "<scheduler.StrategySet: max_sram_used=%s passes_covered=%s>" % (
            self.max_sram_used,
            list(ps.name for ps in self.strats),
        )

    __repr__ = __str__


empty_strategy = Strategy(
    SchedulingStrategy.Unknown, None, [], [], [], make_bandwidth_array(), 0, make_cycles_array(), 0
)
INFINITY = 1e30

ABORT_SEARCH = []


def flatten_list_of_lists(lstlst):
    lst = []
    for v in lstlst:
        lst.extend(v)
    return lst


class DynamicProgrammingScheduler:
    def __init__(self, nng, sg, arch, sram_limit, options: SchedulerOptions):
        self.nng = nng
        self.sg = sg
        self.arch = arch
        self.sram_limit = sram_limit
        self.options = copy.copy(options)
        self.use_cascading = options.use_cascading

        if self.arch.feature_map_storage_mem_area != MemArea.Sram:
            self.use_ifm_ofm_overlap = False  # force off IFM/OFM overlap if IFMs and OFMs are not in the SRAM
        else:
            self.use_ifm_ofm_overlap = True

        self.verbose_schedule = options.verbose_schedule
        self.verbose_pareto_frontier_schedules = options.verbose_pareto_frontier_schedules
        self.mem_area = MemArea.Sram

        self.bandwidth_weights = arch.bandwidth_weights
        self.cycles_weight = arch.cycles_weight
        self.max_sram_used_weight = arch.max_sram_used_weight

        self.n_combinations_searched = 0

        self.pareto_max_candidates = 16

        self.ifm_stream_npu_blocks = set(
            (NpuBlockType.ConvolutionMxN, NpuBlockType.ConvolutionDepthWise, NpuBlockType.Pooling,)
        )

    num_pareto_metrics = 4
    view_values = ",".join(["d"] * num_pareto_metrics)
    order_values = ["f%d" % (idx,) for idx in range(num_pareto_metrics)]

    def pareto_metric(self, candidate):
        strat, strat_set = candidate
        total_cycles = strat.cycles[PassCycles.Total] + strat_set.cycles[PassCycles.Total]
        bws = strat.bws + strat_set.bws
        last_block_height = 0
        if self.options.pareto_metric == ParetoMetric.BwCycMemBlkH and len(strat.block_configs) > 0:
            last_block_height = strat.block_configs[-1][0]

        return (
            np.tensordot(bws, self.bandwidth_weights, axes=3) + total_cycles * self.cycles_weight,
            strat_set.max_sram_used,
            strat.sram_used,
            last_block_height,
        )

    def filter_pareto_frontier(self, candidates, remove_equally_good_candidates):

        candidates = [cand for cand in candidates if max(cand[0].sram_used, cand[1].max_sram_used) <= self.sram_limit]

        if len(candidates) <= 1:
            return candidates
        assert remove_equally_good_candidates
        pareto_vals = np.zeros((len(candidates), DynamicProgrammingScheduler.num_pareto_metrics))
        ids = np.arange(len(candidates), dtype=np.int32)
        for idx, cand in enumerate(candidates):
            pareto_vals[idx] = self.pareto_metric(cand)

        sort_order = np.argsort(
            pareto_vals.view(DynamicProgrammingScheduler.view_values),
            order=DynamicProgrammingScheduler.order_values,
            axis=0,
            kind="stable",
        ).flatten()
        pareto_vals = pareto_vals[sort_order]
        ids = ids[sort_order]

        pareto_frontier = []
        while len(ids) > 0:
            pareto_frontier.append(candidates[ids[0]])
            not_dominated_by_first = (pareto_vals < pareto_vals[0]).any(axis=1)
            ids = ids[not_dominated_by_first]
            pareto_vals = pareto_vals[not_dominated_by_first]

        if len(pareto_frontier) > self.pareto_max_candidates:
            pareto_frontier = self.sort_by_candidate_metric(pareto_frontier)
            pareto_frontier = pareto_frontier[: self.pareto_max_candidates]

        return pareto_frontier

    def candidate_metric(self, candidate):
        strat, strat_set = candidate
        max_sram_used = max(strat_set.max_sram_used, strat.sram_used)
        bws = strat.bws + strat_set.bws
        total_cycles = strat.cycles[PassCycles.Total] + strat_set.cycles[PassCycles.Total]

        return (
            max_sram_used * self.max_sram_used_weight
            + np.tensordot(bws, self.bandwidth_weights, axes=3)
            + total_cycles * self.cycles_weight
        )

    def sort_by_candidate_metric(self, candidate_list):
        sorted_list = list(sorted(candidate_list, key=self.candidate_metric))
        return sorted_list

    def best_candidate(self, candidate_list):
        if len(candidate_list) == 0:
            return ABORT_SEARCH
        if len(candidate_list) == 1:
            return candidate_list[0]
        sorted_list = self.sort_by_candidate_metric(candidate_list)
        return sorted_list[0]

    def graduate_strat(self, strat_type, sram_used, old_strat_data):
        res = []
        for old_strat, old_strat_set in old_strat_data:
            if old_strat.sram_used + sram_used > self.sram_limit:
                continue  # This strategy is bad, drop it
            if old_strat_set.max_sram_used > self.sram_limit:
                continue  # This strategy is bad, drop it
            assert old_strat.strat == SchedulingStrategy.Unknown

            new_strat = old_strat.clone()
            new_strat.strat = strat_type
            new_strat.sram_used = old_strat.sram_used + sram_used

            if self.use_ifm_ofm_overlap:
                overlap = calc_allowed_ofm_ifm_overlap_for_pass_list(
                    new_strat.strat, new_strat.passes, new_strat.block_configs
                )
                new_strat.sram_used -= overlap

            new_strat_set = old_strat_set.clone_add_strategy(new_strat)
            res.append((empty_strategy, new_strat_set))
        return self.filter_pareto_frontier(res, remove_equally_good_candidates=True)

    def append_sram(self, sram_used, old_strat_data):
        res = []
        for old_strat, strat_set in old_strat_data:
            assert old_strat.strat == SchedulingStrategy.Unknown
            assert old_strat.sram_used == 0
            new_strat = old_strat.clone()
            new_strat.sram_used = old_strat.sram_used + sram_used

            res.append((new_strat, strat_set))
        return res

    def append_sram_block_config_performance_metrics(self, sram_used, block_config, metrics, old_strat_data):
        res = []
        for old_strat, strat_set in old_strat_data:
            assert old_strat.strat == SchedulingStrategy.Unknown
            new_strat = old_strat.clone()
            bws, macs, cycles = metrics[:3]

            new_strat.sram_used = old_strat.sram_used + sram_used
            new_strat.block_configs = old_strat.block_configs + [block_config]
            new_strat.bws = old_strat.bws + bws
            new_strat.macs = old_strat.macs + macs
            new_strat.cycles = old_strat.cycles + cycles
            new_strat.bws, new_strat.macs, new_strat.cycles = npu_performance.collate_stats_for_cascaded_pass(
                self.arch, new_strat.bws, new_strat.macs, new_strat.cycles
            )

            res.append((new_strat, strat_set))
        return res

    def append_sram_pass_block_config_performance_metrics_rewrite_list(
        self, sram_used, new_pass, block_config, metrics, rewrite_list, old_strat_data
    ):
        res = []
        for old_strat, strat_set in old_strat_data:
            assert old_strat.strat == SchedulingStrategy.Unknown
            new_strat = old_strat.clone()
            bws, macs, cycles = metrics[:3]
            new_strat.sram_used = old_strat.sram_used + sram_used
            new_strat.block_configs = old_strat.block_configs + [block_config]
            new_strat.bws = old_strat.bws + bws
            new_strat.macs = old_strat.macs + macs
            new_strat.cycles = old_strat.cycles + cycles
            new_strat.passes = old_strat.passes + [new_pass]
            new_strat.bws, new_strat.macs, new_strat.cycles = npu_performance.collate_stats_for_cascaded_pass(
                self.arch, new_strat.bws, new_strat.macs, new_strat.cycles
            )
            new_strat.rewrite_list = old_strat.rewrite_list + rewrite_list
            res.append((new_strat, strat_set))
        return res

    def append_sram_rewrite_list(self, sram_used, rewrite_list, old_strat_data):
        res = []
        for old_strat, strat_set in old_strat_data:
            assert old_strat.strat == SchedulingStrategy.Unknown
            new_strat = old_strat.clone()
            new_strat.sram_used = old_strat.sram_used + sram_used
            new_strat.rewrite_list = old_strat.rewrite_list + rewrite_list
            res.append((new_strat, strat_set))
        return res

    def pass_to_strat(self, strat_data):
        res = {}
        for strat in strat_data[1].strats.values():
            for ps in strat.passes:
                res[ps] = strat
        return res

    def compatible_strats(self, a, b):
        intersection = a.keys() & b.keys()
        for k in intersection:
            if a[k] != b[k]:
                return False
        return True

    def collate_strats_for_passes(self, all_passes):
        if len(all_passes) == 0:
            return [(empty_strategy, StrategySet(dict()))]
        if len(all_passes) == 1:
            return all_passes[0]  # save some space in the common case
        all_strands = [[self.pass_to_strat(strat_data) for strat_data in strand] for strand in all_passes]
        prev_combos = [dict()]
        for j, strand in enumerate(all_strands):
            new_combos = []
            for i, alt in enumerate(strand):
                for prev in prev_combos:
                    if self.compatible_strats(prev, alt):
                        cmb = dict(prev)
                        cmb.update(all_passes[j][i][1].strats)
                        new_combos.append(cmb)
            prev_combos = new_combos

        res = []
        for d in prev_combos:
            s = StrategySet(d)
            s.update_statistics()
            res.append((empty_strategy, s))
        return res

    def search_all_but_one_predecessor(self, ps, pred_pass, pred_pass_data):
        # get the rest of the predecessors
        other_predecessors = [pred for pred in ps.dag_predecessors if pred != pred_pass]
        other_predecessor_data = self.search_pass_list(other_predecessors)

        # pred strat data has an incomplete strategy, which we need
        # to continue on, whereas the other ones have completed strategies.
        # we need to merge these, but keep the incomplete strategy too.

        res = []
        for pred_pass_strat, pred_pass_strat_set in pred_pass_data:
            all_strats = [
                [(empty_strategy, pred_pass_strat_set)],  # pred strat data but with a dummy empty strategy
                other_predecessor_data,  # this one is fine to use as-is
            ]
            collated_strat_data = self.collate_strats_for_passes(all_strats)
            strat_data = [(pred_pass_strat, strat_set) for _, strat_set in collated_strat_data]
            res.extend(strat_data)
        return res

    def calc_non_local_mem_usage(self):
        ignore_subgraph_input_output_tensors = self.sg.placement == PassPlacement.Cpu
        range_set = live_range.extract_live_ranges_from_passes(
            self.sg, self.mem_area, ignore_subgraph_input_output_tensors=ignore_subgraph_input_output_tensors,
        )
        range_dict = range_set.ranges

        # find which ranges overlap passes but aren't input/outputs of the passes.
        # these won't be counted by the dynamic programming search and must be counted in manually.
        end_pos = max(ps.time for ps in self.sg.passes) + 2
        mem_usage = np.zeros(end_pos) + self.sg.base_sram_used
        non_local_mem_usage = np.zeros(end_pos, dtype=np.int64)

        for tens, rng in range_dict.items():
            storage_size = tens.storage_size()
            assert tens.mem_area == self.mem_area
            mem_usage[rng.start_time : rng.end_time] += storage_size

        for ps in self.sg.passes:
            local_mem_usage = 0
            for tens in ps.inputs + ps.outputs + ps.intermediates:
                if tens.mem_area != self.mem_area:
                    continue

                local_mem_usage += tens.storage_size()

            non_local_mem_usage[ps.time] = mem_usage[ps.time] - local_mem_usage

        self.non_local_mem_usage = non_local_mem_usage

    def search(self):
        self.calc_non_local_mem_usage()
        starting_passes = [ps for ps in self.sg.passes if not ps.successors]
        strat_data = self.search_pass_list(starting_passes)

        _, best_set = self.best_candidate(strat_data)

        if self.verbose_pareto_frontier_schedules:
            print(
                "Scheduler searched %d combinations and found %d candidate schedules along the pareto frontier"
                % (self.n_combinations_searched, len(strat_data))
            )
            for idx, (_, strat_set) in enumerate(strat_data):
                extra = ""
                if strat_set == best_set:
                    extra = "(Best candidate)"
                print("Candidate", idx, extra)
                memory_used = {MemArea.Sram: strat_set.max_sram_used}
                stats_writer.print_performance_metrics_for_strat(
                    self.arch,
                    "",
                    strat_set.cycles,
                    strat_set.macs,
                    strat_set.bws,
                    self.nng.batch_size,
                    memory_used,
                    len(self.sg.passes),
                    len(strat_set.strats),
                )

        return best_set

    def search_pass_list(self, pass_list):
        all_strats = []
        for ps in pass_list:
            strat = self.search_output(ps)
            all_strats.append(strat)
        strat_data = self.collate_strats_for_passes(all_strats)
        for strd in strat_data:
            for ps in pass_list:
                assert ps in strd[1].strats  # should have strategies for everything we asked to search
        return strat_data

    def search_predecessors(self, ps):

        # protect against graphs with loops. collate_strats_for_passes will sort this out later so that
        # we have strats for all passes

        pass_list = ps.dag_predecessors
        strat_data = self.search_pass_list(pass_list)

        return strat_data

    @lru_cache(maxsize=None)
    def search_output(self, ps):

        assert ps in self.sg.passes
        candidate_list = []

        candidate_list.extend(self.search_weight_streaming_output(ps))

        if self.options.use_ifm_streaming:
            candidate_list.extend(self.search_ifm_streaming_output(ps))

        best = self.filter_pareto_frontier(candidate_list, remove_equally_good_candidates=True)

        if not best:
            print(
                "Warning: Dynamic search programming algorithm failed for pass %s, invoking fallback strategy"
                % (ps.name,)
            )
            return self.search_predecessors(ps)

        return best

    def search_ifm_streaming_output(self, ps):
        if ps.placement != PassPlacement.Npu:
            return ABORT_SEARCH
        if ps.npu_block_type not in self.ifm_stream_npu_blocks:
            return ABORT_SEARCH
        strat_data = self.search_ifm_streaming_body(ps, False)

        sram_used = self.non_local_mem_usage[ps.time]
        for tens in ps.outputs:
            if tens.mem_area == self.mem_area:
                sram_used += tens.storage_size()

        return self.graduate_strat(SchedulingStrategy.IfmStream, sram_used, strat_data)

    @lru_cache(maxsize=None)
    def search_ifm_streaming_body(self, ps, force_outputs_to_fast_storage):
        if ps.placement != PassPlacement.Npu:
            return ABORT_SEARCH
        if ps.npu_block_type not in self.ifm_stream_npu_blocks:
            return ABORT_SEARCH
        ifm_input_search_resuls = self.search_ifm_streaming_input(ps)
        res = []

        base_sram_used = 0
        for tens in ps.intermediates:
            if tens.mem_area == self.mem_area:
                if tens.purpose == TensorPurpose.Weights:
                    base_sram_used = tens.storage_size(self.arch.weight_estimation_scaling)
                else:
                    base_sram_used += tens.storage_size()

        all_block_configs = self.get_block_configs(ps)
        for block_config in all_block_configs:
            all_strats = []

            if self.use_cascading:
                all_strats.extend(self.search_ifm_streaming_partial(ps, block_config))

            all_strats.extend(ifm_input_search_resuls)

            rewrite_list = []
            sram_used = base_sram_used

            metrics = npu_performance.performance_metrics_for_pass(
                self.arch,
                ps,
                block_config,
                rewrite_list=rewrite_list,
                force_outputs_to_fast_storage=force_outputs_to_fast_storage,
            )

            res.extend(
                self.append_sram_pass_block_config_performance_metrics_rewrite_list(
                    sram_used, ps, block_config, metrics, rewrite_list, all_strats
                )
            )

        self.n_combinations_searched += len(res)
        res = self.filter_pareto_frontier(res, remove_equally_good_candidates=True)
        return res

    def avoid_for_cascading(self, pred_candidate):
        for op in pred_candidate.ops:
            if (
                op.memory_function == Op.ConcatSliceWrite
                and self.arch.feature_map_storage_mem_area != self.arch.fast_storage_mem_area
            ):
                # For SRAM spilling, concat op is avoided as predecessor
                return True
            if len(op.outputs) > 1 or len(op.outputs[0].consumer_list) > 1:
                # The op has consumers in other subgraphs
                return True
        return False

    def search_ifm_streaming_partial(self, ps, block_config):
        if ps.placement != PassPlacement.Npu:
            return ABORT_SEARCH

        if len(ps.inputs) < 1:
            return ABORT_SEARCH

        ifm_tensor = ps.ifm_tensor

        if ifm_tensor is None:
            return ABORT_SEARCH
        if ifm_tensor.purpose != TensorPurpose.FeatureMap:
            return ABORT_SEARCH
        if not ifm_tensor.storage_shape or len(ifm_tensor.storage_shape) != 4:
            return ABORT_SEARCH

        pred_pass_list = []
        for pred_candidate in ps.dag_predecessors:
            if len(pred_candidate.outputs) == 1 and pred_candidate.outputs[0] == ifm_tensor:
                # we found a predecessor that produces this IFM tensor
                if not ifm_tensor.avoid_NHCWB16:
                    # and NHCWB16 format is not to be avoided
                    if len(pred_candidate.successors) == 1 and pred_candidate.successors[0] == ps:
                        # and it only has one successor, namely us
                        if pred_candidate.placement == PassPlacement.Npu:
                            if pred_candidate.npu_block_type in self.ifm_stream_npu_blocks:
                                # and it is on the Npu
                                if not self.avoid_for_cascading(pred_candidate):
                                    # and fusable - it's a candidate
                                    pred_pass_list.append(pred_candidate)

        if not pred_pass_list:
            return ABORT_SEARCH

        all_candidates = []
        for pred_pass in pred_pass_list:
            # recurse into the next pass
            ifm_strat_data = self.search_ifm_streaming_body(pred_pass, self.arch.is_spilling_enabled())

            strat_data = self.search_all_but_one_predecessor(ps, pred_pass, ifm_strat_data)
            for strat_opt in strat_data:

                pred_pass_block_config = strat_opt[0].block_configs[-1]
                rolling_buffer_dims = npu_performance.rolling_buffer_dims_from_passes(
                    self.arch, pred_pass, pred_pass_block_config, ps, block_config
                )
                if rolling_buffer_dims is None:
                    continue  # this does not pack properly, skip it.

                sram_used = 0
                for tens in ps.inputs:
                    if tens != ifm_tensor:
                        if tens.mem_area == self.mem_area:
                            sram_used += tens.storage_size()

                rolling_buffer_y, rolling_buffer_x = rolling_buffer_dims

                rewrite_list = [
                    (
                        SchedulerRewrite.ChangeTensorSubPurpose,
                        ifm_tensor,
                        TensorSubPurpose.RollingBufferY,
                        rolling_buffer_y,
                        None,
                        ps,
                    )
                ]
                sram_used += ifm_tensor.storage_size_for_sub_purpose(
                    self.arch, TensorSubPurpose.RollingBufferY, rolling_buffer_y, None
                )

                all_candidates.extend(self.append_sram_rewrite_list(sram_used, rewrite_list, [strat_opt]))

        self.n_combinations_searched += len(all_candidates)
        return all_candidates

    def get_block_configs(self, ps):
        if ps.placement != PassPlacement.Npu:
            return [(1, 1, 1, 1)]  # default

        block_configs = find_block_configs_suitable_for_pass_and_shared_buffer(self.arch, ps)

        # Take a limited number of the largest blocks
        if self.arch.block_config_limit > 0:
            # Sort by block area, followed by depth
            block_configs.sort(key=lambda cfg: (cfg[0] * cfg[1]) << 8 | cfg[3], reverse=True)
            bound = min(len(block_configs), self.arch.block_config_limit)
            # We take 'n' from the fat end of the list, and 'n' from the thin end of the list.
            tmp = block_configs[:bound]
            tmp.extend(block_configs[max(bound, len(block_configs) - bound) :])
            block_configs = tmp

        return block_configs

    def search_ifm_streaming_input(self, ps):
        sram_used = 0
        for tens in ps.inputs:
            if tens.mem_area == self.mem_area:
                sram_used += tens.storage_size()

        return self.append_sram(sram_used, self.search_predecessors(ps))

    def search_weight_streaming_output(self, ps):
        strat_data = self.search_weight_streaming_body(ps)

        sram_used = self.non_local_mem_usage[ps.time]
        for tens in ps.outputs:
            if tens.mem_area == self.mem_area:
                sram_used += tens.storage_size()

        return self.graduate_strat(SchedulingStrategy.WeightStream, sram_used, strat_data)

    @lru_cache(maxsize=None)
    def search_weight_streaming_body(self, ps):

        strat_data = self.search_weight_streaming_input(ps)

        res = []

        all_block_configs = self.get_block_configs(ps)

        for block_config in all_block_configs:

            sram_used = 0
            rewrite_list = []

            for tens in ps.intermediates:
                if tens.mem_area == self.mem_area:
                    if tens.purpose == TensorPurpose.Weights:
                        sram_used += tens.storage_size_for_sub_purpose(
                            self.arch, TensorSubPurpose.DoubleBuffer, block_config[3]
                        )
                        rewrite_list.append(
                            (
                                SchedulerRewrite.ChangeTensorSubPurpose,
                                tens,
                                TensorSubPurpose.DoubleBuffer,
                                block_config[3],
                                None,
                                ps,
                            )
                        )
                    else:
                        sram_used += tens.storage_size()

            metrics = npu_performance.performance_metrics_for_pass(
                self.arch, ps, block_config, rewrite_list=rewrite_list
            )

            res.extend(
                self.append_sram_pass_block_config_performance_metrics_rewrite_list(
                    sram_used, ps, block_config, metrics, rewrite_list, strat_data
                )
            )

        self.n_combinations_searched += len(res)
        res = self.filter_pareto_frontier(res, remove_equally_good_candidates=True)
        return res

    def search_weight_streaming_input(self, ps):
        sram_used = 0
        for tens in ps.inputs:
            if tens.mem_area == self.mem_area:
                sram_used += tens.storage_size()

        return self.append_sram(sram_used, self.search_predecessors(ps))

    def apply_result(self, strat_set, arch):
        pass_to_cascaded_pass = dict()
        for _, strat in strat_set.strats.items():
            # rewrite the tensors that need this first. e.g. make rolling buffers
            inputs = []
            intermediates = []
            outputs = []

            for ps in strat.passes:
                inputs += ps.inputs
                intermediates += ps.intermediates
                outputs += ps.outputs

            for tens in set(inputs) & set(outputs):
                # tensors that are in both sets are intermediates

                # find pass with input/output tensor, and check if they are both placed on NPU
                input_placement = None
                output_placement = None
                for ps in strat.passes:
                    if tens in ps.inputs:
                        input_placement = ps.placement
                    if tens in ps.outputs:
                        output_placement = ps.placement
                if input_placement == output_placement == PassPlacement.Npu:
                    tens.set_format(TensorFormat.NHCWB16, arch)

                intermediates.append(tens)
                inputs.remove(tens)
                outputs.remove(tens)

            for rewrite_op, tens, sub_purpose, param_a, param_b, ps in strat.rewrite_list:
                if rewrite_op == SchedulerRewrite.ChangeTensorSubPurpose:
                    tens.mem_area = self.arch.fast_storage_mem_area
                    tens.mem_type = MemType.Scratch_fast
                    tens.set_new_sub_purpose(sub_purpose, param_a, param_b)
                else:
                    assert 0, "unknown rewrite_op " + str(rewrite_op)

            is_element_wise = True
            for ps in strat.passes:
                assert ps.placement == strat.passes[0].placement
                if not ps.is_element_wise:
                    is_element_wise = False
                    break

            cascaded_pass = CascadedPass(
                strat.passes[0].name,
                strat.strat,
                inputs,
                intermediates,
                outputs,
                strat.passes,
                strat.passes[0].placement,
                is_element_wise,
            )
            assert strat.sram_used >= 0
            cascaded_pass.sram_used = strat.sram_used

            for idx, ps in enumerate(strat.passes):
                assert ps not in pass_to_cascaded_pass
                pass_to_cascaded_pass[ps] = cascaded_pass
                ps.cascade = cascaded_pass
                ps.block_config = strat.block_configs[idx]

                if ps.placement == PassPlacement.Npu:
                    ps.shared_buffer = shared_buffer_allocation_for_pass_and_block_config(
                        self.arch, ps, ps.block_config
                    )
                    assert ps.shared_buffer is not None

                sram_used = max(self.non_local_mem_usage[ps.time], 0)
                for op in ps.ops:
                    subgraph = op.attrs.get("subgraph")
                    if subgraph:
                        subgraph.base_sram_used = sram_used

        # all passes should have a cascaded pass now
        if len(pass_to_cascaded_pass) != len(self.sg.passes):
            print(
                "mismatch: we have %d passes, but only %d have cascaded passes associated"
                % (len(self.sg.passes), len(pass_to_cascaded_pass))
            )
            for ps in self.sg.passes:
                if ps not in pass_to_cascaded_pass:
                    print("%3d pass missing cascaded pass %s" % (ps.time, ps))

            assert len(pass_to_cascaded_pass) == len(self.sg.passes)

        cascaded_passes = []
        if self.sg.placement == PassPlacement.Cpu:
            # Retain the pass order for CPU subgraph
            cascaded_passes = [ps.cascade for ps in self.sg.passes]
        else:
            # we have all the passes, but we need to put them in order and build predecessor/successor links.
            visit_pass_set = set()

            def visit_pass(ps):
                if ps in visit_pass_set:
                    return
                visit_pass_set.add(ps)

                cps = ps.cascade
                dont_traverse = set(cps.passes)

                for ps in cps.passes:
                    for pred in ps.predecessors:
                        if pred in dont_traverse:
                            continue
                        visit_pass(pred)

                cascaded_passes.append(cps)

            starting_passes = [ps for ps in self.sg.passes if not ps.successors]
            for ps in starting_passes:
                visit_pass(ps)

        # reorder so startup init cascaded passes come first
        def is_startup_cascaded_pass(cps):
            if not cps.passes:
                return False
            return cps.placement == PassPlacement.StartupInit

        cascaded_passes = [cps for cps in cascaded_passes if is_startup_cascaded_pass(cps)] + [
            cps for cps in cascaded_passes if not is_startup_cascaded_pass(cps)
        ]

        self.sg.cascaded_passes = cascaded_passes
        self.sg.build_cascaded_pass_links()

        # Check if NHCWB16 and/or fast storage can be used in between cascaded passes
        # (NHCWB16 within cascaded passes has been handled earlier in this function)
        if self.sg.placement == PassPlacement.Npu:
            # Dictionary tensor -> list of ops, containing feature maps that can be attempted
            # to be moved to fast storage
            fast_storage_tensor_rewrites = {}
            last_op_in_subgraph = self.sg.cascaded_passes[-1].passes[-1].primary_op
            # Memory only passes have no primary_op, so use the last op in ops
            if last_op_in_subgraph is None:
                last_op_in_subgraph = self.sg.cascaded_passes[-1].passes[-1].ops[-1]
            for ps in self.sg.cascaded_passes:
                if ps.placement != PassPlacement.Npu:
                    continue
                for output in ps.outputs:
                    if output.purpose != TensorPurpose.FeatureMap:
                        continue

                    use_NHCWB16 = not output.avoid_NHCWB16
                    use_fast_storage = True
                    rewrites = []
                    for op in output.consumer_list:
                        if op is None:
                            use_NHCWB16 = False
                            use_fast_storage = False
                            continue
                        if op.type == Op.ReduceSum and output.dtype == DataType.int32:
                            use_NHCWB16 = False
                        elif op.type == Op.Reshape:
                            # Using NHCWB16 format for a no-op reshape is only an option if subsequent
                            # consumers do not also need to perform a reshape or if the OFM is going to
                            # be processed by CPU operations. No-op reshape consumers with empty lists
                            # (those that have no consumers, or null-consumers used as list terminators)
                            # must use normal NHWC output.
                            def incompatible_consumers(oper):
                                if oper and oper.type == Op.Reshape:
                                    for consumer in oper.outputs[0].consumer_list:
                                        yield from incompatible_consumers(consumer)
                                yield not oper or not oper.run_on_npu or oper is last_op_in_subgraph

                            if not any(incompatible_consumers(op)):

                                def get_rewrites(oper):
                                    if oper and oper.type == Op.Reshape:
                                        for consumer in oper.outputs[0].consumer_list:
                                            yield from get_rewrites(consumer)
                                        yield oper

                                rewrites.extend(get_rewrites(op))
                                # Detect no-op reshapes by comparing their full input and output tensor shapes.
                                inshape = op.ifm_shapes[0]
                                compatible_shape = [(inshape == oper.ofm_shapes[0]) for oper in get_rewrites(op)]
                                use_NHCWB16 &= compatible_shape and all(compatible_shape)
                            else:
                                use_NHCWB16 = False
                                use_fast_storage = False
                        use_NHCWB16 &= op.run_on_npu
                        use_fast_storage &= op.run_on_npu

                    if use_fast_storage:
                        fast_storage_tensor_rewrites[output] = rewrites
                    if use_NHCWB16 and self.options.use_nhcwb16_between_cascaded_passes:
                        output.set_format(TensorFormat.NHCWB16, arch)
                        for rewrite_op in rewrites:
                            rewrite_op.outputs[0].set_format(TensorFormat.NHCWB16, arch)
            if arch.is_spilling_enabled():
                # Remember feature maps that can be moved to fast storage for later use
                # in use_fast_storage_for_feature_maps
                self.sg.scheduling_info["feature_map_rewrites"] = fast_storage_tensor_rewrites


def move_scales_to_fast_storage(nng, arch):
    for sg in nng.subgraphs:
        # IFM streamed ops reads bias tensors several times, move these to fast storage
        for cp in sg.cascaded_passes:
            if cp.strategy == SchedulingStrategy.IfmStream:
                # Calculate SRAM usage
                new_size = 0
                all_tens = []
                for ps in cp.passes:
                    pass_tens = np.array([ps.ifm_tensor, ps.ifm2_tensor, ps.ofm_tensor, ps.weight_tensor])
                    pass_tens = np.append(pass_tens, ps.intermediates)
                    for tens in pass_tens:
                        if tens and tens.mem_area == MemArea.Sram and tens not in all_tens:
                            all_tens.append(tens)
                            new_size += tens.storage_size()

                cp.sram_used = new_size

                for ps in cp.passes:
                    if ps.scale_tensor:
                        tens = ps.scale_tensor

                        # Find op using scale tensor
                        op = next((op for op in ps.ops if tens in op.inputs), None)
                        assert op

                        # Create fast storage tensor
                        new_tens = tens.clone_into_fast_storage(arch)
                        new_tens.consumer_list = tens.consumer_list.copy()
                        new_tens.purpose = TensorPurpose.FSBias
                        new_tens_size = new_tens.storage_size()

                        if (cp.sram_used + new_tens_size) <= arch.sram_size:
                            # Create DMA cmd
                            dma_cmd = Operation(Op.DMA, tens.ops[0].name + "_dma")
                            dma_cmd.inputs = [tens]
                            dma_cmd.set_output_tensor(new_tens)
                            dma_cmd.attrs["source"] = tens.mem_area
                            dma_cmd.attrs["destination"] = new_tens.mem_area
                            dma_cmd.run_on_npu = True

                            tens.consumer_list.clear()
                            tens.consumer_list.append(dma_cmd)

                            # Replace tensor and op
                            idx = op.inputs.index(tens)
                            op.inputs[idx] = new_tens

                            ps.ops.insert(0, dma_cmd)
                            ps.scale_tensor = new_tens
                            ps.intermediates.append(new_tens)
                            ps.cascade.intermediates.append(new_tens)

                            cp.sram_used += new_tens_size


def schedule_passes(nng, arch, options: SchedulerOptions):

    for sg in nng.subgraphs:
        sg.base_sram_used = 0

    for sg in nng.subgraphs:
        # re-entering the same nodes from different contexts requires us to
        # build a simplified directed acyclic (DAG) version of the graph to
        # use for traversal, rather than using a visit dictionary. this avoids
        # recursing infinitely due to loops.
        sg.build_pass_dag_predecessors()

        dps = DynamicProgrammingScheduler(nng, sg, arch, arch.sram_size, options)

        strat_set = dps.search()

        dps.apply_result(strat_set, arch)

        if options.verbose_schedule:
            sg.print_cascaded_passes()


def _calc_tens_to_cps(sg, tensor_rewrites):
    # Determines for each tensor the list of affected cascaded passes, in terms of SRAM consumption.
    # Returns dictionary tensor -> list of cascaded passes
    # Note: if cascaded passes are A, B, C, D, and a tensor is output
    # of A and input to D, then it also consumes SRAM in passes B and C.
    if "tens_to_cps" in sg.scheduling_info:
        return sg.scheduling_info["tens_to_cps"]
    # Determine life-time of tensors
    min_index = {}
    max_index = {}
    index = 0
    cps_list = [cps for cps in sg.cascaded_passes if cps.placement == PassPlacement.Npu]
    for cps in cps_list:
        for tens in cps.inputs + cps.outputs:
            if tens in tensor_rewrites:
                min_index[tens] = min(index, min_index.get(tens, len(cps_list)))
                max_index[tens] = index
        index += 1
    # Convert to affected cps-es
    tens_to_cps = {}
    for tens in min_index:
        tens_to_cps[tens] = cps_list[min_index[tens] : max_index[tens] + 1]
    sg.scheduling_info["tens_to_cps"] = tens_to_cps
    return tens_to_cps


def use_fast_storage_for_feature_maps(sg, sram_limit, arch):
    # Attempts to use as much fast storage as possible for feature maps shared between cascaded passes.
    tensor_rewrites = sg.scheduling_info.get("feature_map_rewrites", {})
    tens_to_cps = _calc_tens_to_cps(sg, tensor_rewrites)
    # Sort tensors first on life-time (smallest first), then on size (biggest first)
    tens_list = sorted([(len(tens_to_cps[tens]), -tens.storage_size(), tens.name, tens) for tens in tens_to_cps])
    for _, _, _, tens in tens_list:
        cps_list = tens_to_cps[tens]
        if len(cps_list) < 1:
            continue
        sz = tens.storage_size()
        fits_in_fast_storage = all([cps.sram_used + sz <= sram_limit for cps in cps_list])
        if fits_in_fast_storage:
            tens.mem_area = arch.fast_storage_mem_area
            tens.mem_type = MemType.Scratch_fast
            tens.set_new_sub_purpose(TensorSubPurpose.Standard, None, None)
            assert tens in tensor_rewrites
            # Also rewrite reshapes
            for rewrite_op in tensor_rewrites[tens]:
                tens2 = rewrite_op.outputs[0]
                tens2.mem_area = arch.fast_storage_mem_area
                tens2.mem_type = MemType.Scratch_fast
                tens2.set_new_sub_purpose(TensorSubPurpose.Standard, None, None)
            for cps in cps_list:
                cps.sram_used += sz


def undo_use_fast_storage(sg, arch):
    # Undoes the effects of a previous call to use_fast_storage_for_feature_maps
    tensor_rewrites = sg.scheduling_info.get("feature_map_rewrites", {})
    tens_to_cps = _calc_tens_to_cps(sg, tensor_rewrites)
    mem_area = arch.tensor_storage_mem_area[TensorPurpose.FeatureMap]
    for tens, cps_list in tens_to_cps.items():
        if tens.mem_type == MemType.Scratch_fast:
            sz = tens.storage_size()
            tens.mem_area = mem_area
            tens.mem_type = MemType.Scratch
            # Also undo reshapes
            for rewrite_op in tensor_rewrites[tens]:
                tens2 = rewrite_op.outputs[0]
                tens2.mem_area = mem_area
                tens2.mem_type = MemType.Scratch
            for cps in cps_list:
                cps.sram_used -= sz
