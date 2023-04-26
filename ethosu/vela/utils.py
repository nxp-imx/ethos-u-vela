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
# Contains various utility functions used across the codebase.
from __future__ import annotations

import collections
import inspect


def progress_print(
    enabled: bool,
    message: str,
    progress_counter: int = -1,
    progress_total: int | collections.Sized = 0,
    progress_granularity: float = 0.20,
):
    """Print progress information.

    :param enabled: boolean indicating whether message should be printed.
    :param message: message to be printed
    :param progress_counter: the value of the incremental counter that indicates the progress
    :param progress_total: integer value or sized data structure to use to extract the total number of elements that
                           progress is measured against
    :param progress_granularity: floating point percentage indicating how often progress information should be printed
    :param enable_context: boolean used to indicate whether context information should be printed with the message

    Example
    -------
    def example_function(verbose_progress: bool = True):
        a_list = [x for x in range(101)]
        for index, value in a:
            progress_print(verbose_progress,
                            message="Processing",
                            progress_counter=index,
                            progress_total=a_list,
                            progress_granulrity=0.25,
                            enable_context=True)

    **Output**
    Processing 0/100
    Processing 25/100
    Processing 50/100
    Processing 75/100
    Processing 100/100
    """
    if not enabled:
        return

    context_str = ""
    # Get calling function name
    context_str = inspect.stack()[1].function
    context_str += ": " if message else ""
    display_total = progress_total
    # If a sized collection is provided, extract its size to use as progress total
    if isinstance(progress_total, collections.Sized):
        progress_total = len(progress_total)
        display_total = progress_total - 1

    # Print progress information with "counter/total" information
    if progress_counter > -1 and progress_total > 0 and 0 < progress_granularity < 1:
        # Extract progress frequency and ensure it is not equal to 0 (avoid zero division)
        progress_frequency = int(progress_total * progress_granularity)
        progress_frequency = progress_frequency if progress_frequency else 1
        # Check whether information should be printed based on computed progress frequency
        if (
            progress_counter % progress_frequency == 0 and progress_counter <= progress_total - progress_frequency
        ) or progress_counter == display_total:
            print(f"{context_str}{message} {progress_counter}/{display_total}")
        return

    print(f"{context_str}{message}")
