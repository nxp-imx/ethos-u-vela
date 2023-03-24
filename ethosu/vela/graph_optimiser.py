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
# Early optimisation of the network graph, using the rewrite_graph module to do the traversal of the graph.
from . import rewrite_graph
from .graph_optimiser_util import check_format_restrictions
from .graph_optimiser_util import check_memory_only_removed
from .graph_optimiser_util import record_optimised
from .nn_graph import NetworkType
from .tflite_graph_optimiser import tflite_optimise_graph
from .tosa_graph_optimiser import tosa_optimise_graph


def optimise_graph(nng, arch, network_type, verbose_graph=False, force_symmetric_int_weights=False, output_basename=None, subgraph_output=False):
    if verbose_graph:
        nng.print_graph("Before Graph Optimization")

    if network_type == NetworkType.TFLite:
        # TensorFlow Lite graph optimization
        nng = tflite_optimise_graph(nng, arch, force_symmetric_int_weights, output_basename, subgraph_output)
    else:
        # TOSA graph optimization
        nng = tosa_optimise_graph(nng, arch)

    # Post-optimisation operator debug tracing, and checking that no undesired reshapes are left in the graph
    for sg in nng.subgraphs:
        rewrite_graph.visit_graph_post_order(
            sg.output_tensors, arch, [check_format_restrictions], [check_memory_only_removed, record_optimised]
        )

    if verbose_graph:
        nng.print_graph("After Graph Optimization")
    return nng
