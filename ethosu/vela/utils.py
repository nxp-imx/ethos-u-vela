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
    progress_total: int | collections.abc.Sized = 0,
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
    if isinstance(progress_total, collections.abc.Sized):
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


def calc_resize_factor(ifm_width: int, stride_x: int) -> tuple[int, int]:
    """Compute resize factor for strided Conv2D optimization."""
    # Define strides that are supported by HW
    hw_supported_strides = (2, 3)
    resize_factor = stride_x

    if ifm_width % resize_factor != 0:
        # In case it is not divisible, check if the resize factor is
        # divisible by any of the hw_supported_strides. If it is, re-compute
        # the resize factor to be the value that leads us to
        # reach a hw supported stride. The IFM width needs to be divisible by the new resize factor.
        # E.g.: IFM width = 133, stride = 14, filter width = 7 can be
        #       optimised to IFM width = 19, stride = 2, filter width = 7 using
        #       a resize factor of 7. The final stride is 2 which is
        #       supported by the hardware.

        # Filter strides that can be obtained from current stride
        divisible_strides = (x for x in hw_supported_strides if resize_factor % x == 0)
        # Remove strides that are not IFM width divisors
        divisor_strides = (x for x in divisible_strides if ifm_width % (stride_x // x) == 0)
        # Compute new resize factor based on chosen stride
        new_resize_factor = resize_factor // next(divisor_strides, 1)
        resize_factor = new_resize_factor if resize_factor != new_resize_factor else 1

    optimised_stride = stride_x // resize_factor

    return resize_factor, optimised_stride
