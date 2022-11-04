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
# Unit tests for model_reader.
import pytest

from ethosu.vela import model_reader
from ethosu.vela.errors import InputFileError


def test_read_model_incorrect_extension(tmpdir):
    # Tests read_model with a file name that does not end with .tflite
    with pytest.raises(InputFileError):
        model_reader.read_model("no_tflite_file.txt", model_reader.ModelReaderOptions())


def test_read_model_file_not_found(tmpdir):
    # Tests read_model with a .tflite file that does not exist
    with pytest.raises(FileNotFoundError):
        model_reader.read_model("non_existing.tflite", model_reader.ModelReaderOptions())
