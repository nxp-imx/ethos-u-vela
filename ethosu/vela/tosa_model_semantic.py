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
# The TosaSemantic class which is a collection of TOSA model semantic checks.
from collections import defaultdict

from .operation import Op
from .tosa_mapping import optype_to_tosa_op_type


class TosaSemantic:
    # TODO populate this

    def __init__(self):
        # Setup the generic constraints. Note: the order matters
        self.generic_constraints = []

        # Setup specific constraints. Note: the order matters
        self.specific_constraints = defaultdict(list)

    def is_operator_semantic_valid(self, op):
        ext_type = optype_to_tosa_op_type(op.type)

        if op.type in (Op.Placeholder, Op.SubgraphInput, Op.Const):
            return True

        for constraint in self.generic_constraints + self.specific_constraints[op.type]:
            valid, extra = constraint(op)
            if not valid:
                print(f"Warning: unsupported TOSA semantics for {ext_type} '{op.name}'.")
                print(f" - {constraint.__doc__}")
                if extra:
                    print(f"   {extra}")
                return False

        return True


def tosa_semantic_checker(nng):
    semantic_checker = TosaSemantic()
    for sg in nng.subgraphs:
        for op in sg.get_all_ops():
            op.run_on_npu = semantic_checker.is_operator_semantic_valid(op)
    return nng
