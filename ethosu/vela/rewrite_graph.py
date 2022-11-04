# SPDX-FileCopyrightText: Copyright 2020 Arm Limited and/or its affiliates <open-source-office@arm.com>
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
# Functions for abstracting out the traversal and rewriting of graphs so that the optimisation passes can focus on the
# correct operation.
#
# Requires two lists, one of functions that rewrite Tensors, and one of functions that rewrite Operations.
#
# Pre-order traversal, this supports rewrites. Therefore, functions can return something other than the original value.
#
# Post-order traversal, this does not support rewrites. Therefore, functions must return the original value.


def rewrite_graph_pre_order(nng, sg, arch, tensor_rewrite_list, op_rewrite_list, rewrite_unsupported=True):

    op_visit_dict = dict()
    tens_visit_dict = dict()

    def visit_op(op):
        if op in op_visit_dict:
            return op_visit_dict[op]
        res = op
        prev_res = None
        while prev_res != res:
            prev_res = res
            for rewrite in op_rewrite_list:
                if res.run_on_npu or rewrite_unsupported:
                    res = rewrite(res, arch, nng)

        op_visit_dict[op] = res
        op_visit_dict[res] = res

        inputs = res.inputs
        res.inputs = []
        for tens in inputs:
            res.inputs.append(visit_tens(tens))

        outputs = res.outputs
        res.outputs = []
        for tens in outputs:
            res.outputs.append(visit_tens(tens))

        return res

    def visit_tens(tens):
        if tens in tens_visit_dict:
            return tens_visit_dict[tens]

        res = tens
        prev_res = None
        while prev_res != res:
            prev_res = res
            for rewrite in tensor_rewrite_list:
                res = rewrite(res, arch, nng)

        tens_visit_dict[tens] = res
        tens_visit_dict[res] = res

        if res:
            ops = res.ops
            res.ops = []
            for op in ops:
                res.ops.append(visit_op(op))
        return res

    sg.output_tensors = [visit_tens(tens) for tens in sg.output_tensors]
    sg.refresh_after_modification()

    return sg


def visit_graph_post_order(start_tensors, arch, tensor_visit_list, op_visit_list):
    # Depth-first graph traversal, starting from the given list of tensors
    # (typically a subgraph's output_tensors).
    # Visits ops and tensors in input to output order.
    op_visit_dict = dict()
    tens_visit_dict = dict()

    def visit_op(op):
        if op in op_visit_dict:
            return
        op_visit_dict[op] = op

        for tens in op.inputs:
            visit_tens(tens)

        for visit in op_visit_list:
            visit(op, arch)

        for tens in op.outputs:
            visit_tens(tens)

    def visit_tens(tens):
        if tens is None or tens in tens_visit_dict:
            return

        tens_visit_dict[tens] = tens

        for op in tens.ops:
            visit_op(op)

        for visit in tensor_visit_list:
            visit(tens, arch)

    for tens in start_tensors:
        visit_tens(tens)


def verify_graph_health(nng):

    for sg in nng.subgraphs:
        verify_subgraph_health(sg)

    return True


def verify_subgraph_health(sg):
    op_visit_dict = dict()
    tens_visit_dict = dict()

    def visit_op(op):
        if op in op_visit_dict:
            return op_visit_dict[op]
        op_visit_dict[op] = op

        for tens in op.inputs:
            if not tens:
                continue
            assert op in tens.consumers()
            visit_tens(tens)

        for tens in op.outputs:
            assert op in tens.ops
            visit_tens(tens)

        return op

    def visit_tens(tens):
        if tens in tens_visit_dict:
            return tens_visit_dict[tens]

        tens_visit_dict[tens] = tens

        for op in tens.ops:
            assert tens in op.outputs
            visit_op(op)

        return tens

    for tens in sg.output_tensors:
        visit_tens(tens)

    return True
