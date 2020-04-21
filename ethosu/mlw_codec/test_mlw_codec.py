#!/usr/bin/env python3
# Copyright (C) 2020 Arm Limited or its affiliates. All rights reserved.
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
# Simple example of the usage of mlw_codec.
import sys

from ethosu import mlw_codec

# Simple example
if __name__ == "__main__":
    weights = [0, 2, 3, 0, -1, -2, -3, 0, 0, 0, 1, -250, 240] * 3
    print("Original weights    :", weights)

    compressed_weights = mlw_codec.encode(weights)
    print("Compressed weights  :", len(compressed_weights), compressed_weights)

    uncompressed_weights = mlw_codec.decode(compressed_weights)
    print("Uncompressed weights:", uncompressed_weights)

    if weights != uncompressed_weights:
        print("TEST FAILED")
        sys.exit(1)
    else:
        print("TEST PASSED")
        sys.exit(0)
