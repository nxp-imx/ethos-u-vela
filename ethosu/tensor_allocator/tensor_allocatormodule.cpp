/*
 * Copyright (c) 2020 Arm Limited. All rights reserved.
 *
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the License); you may
 * not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an AS IS BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#define PY_SSIZE_T_CLEAN
#include <Python.h>

#include <vector>
#include "search_allocator.h"



/**
 * C++ extension wrapper for allocate
 *
 * This method is exposed directly in python with the arguments with a
 * prototype of the form:
 *
 * output = tensor_allocator.allocate(input, available_size=0)
 *
 * input: [int]
 * available_size: int
 * output: [int]
 */
static PyObject *method_allocate (PyObject *self, PyObject *args)
{
    /* Object to hold the input integer list. */
    PyObject *input_list_object;

    /* Object to hold the available size */
    int available_size = 0;

    /* Arguments to the method are delivered as a tuple, unpack the
      * tuple to get the individual arguments, note the second is
      * optional.
      */
    if (!PyArg_ParseTuple(args, "O|i", &input_list_object, &available_size)) {
        return NULL;
    }

    /* Unpack the length of the input integer list. */
    int input_length = static_cast<int>(PyObject_Length (input_list_object));
    if (input_length < 0) {
        input_length = 0;
    }
    std::vector<uint32_t> input;
    std::vector<uint32_t> output;
    for (int i = 0; i < input_length; ++i) {
        PyObject *obj = PyList_GetItem(input_list_object, i);
        uint32_t value = (uint32_t)PyLong_AsLong(obj);
        input.push_back(value);
    }
    allocate(input, available_size, output);
    PyObject *output_list = PyList_New(output.size());
    for (size_t i = 0; i < output.size(); ++i) {
        PyList_SetItem(output_list, i, PyLong_FromLong(output[i]));
    }
    return output_list;
}

/** tensor_allocator method descriptors. */
static PyMethodDef tensor_allocator_methods[] = {
    {"allocate", method_allocate, METH_VARARGS, "Python interface for allocate"},
    {NULL, NULL, 0, NULL}
};

/** tensor_allocator module descriptor. */
static struct PyModuleDef tensor_allocatormodule = {
    PyModuleDef_HEAD_INIT,
    "tensor_allocator",
    "Python interface for tensor_allocator",
    -1,
    tensor_allocator_methods
};

PyMODINIT_FUNC PyInit_tensor_allocator(void) {
    return PyModule_Create(&tensor_allocatormodule);
}
