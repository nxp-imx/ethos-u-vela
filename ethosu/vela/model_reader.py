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
# Description:
# Dispatcher for reading a neural network model.
from . import tflite_reader
from .errors import InputFileError


class ModelReaderOptions:
    def __init__(self, batch_size=1):
        self.batch_size = batch_size

    def __str__(self):
        return type(self).__name__ + ": " + str(self.__dict__)

    __repr__ = __str__


def read_model(fname, options, feed_dict=None, output_node_names=None, initialisation_nodes=None):
    if fname.endswith(".tflite"):
        if feed_dict is None:
            feed_dict = {}
        if output_node_names is None:
            output_node_names = []
        if initialisation_nodes is None:
            initialisation_nodes = []
        return tflite_reader.read_tflite(
            fname,
            options.batch_size,
            feed_dict=feed_dict,
            output_node_names=output_node_names,
            initialisation_nodes=initialisation_nodes,
        )
    else:
        raise InputFileError(fname, "Unsupported file extension. Only .tflite files are supported")
