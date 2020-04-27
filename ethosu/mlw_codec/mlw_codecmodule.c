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

#include "mlw_decode.h"
#include "mlw_encode.h"

/* C extension wrapper for mlw_encode
 *
 * This method is exposed directly in python with the arguments with a
 * prototype of the form:
 *
 * output = mlw_codec.encode(input, verbose=0)
 *
 * input: [int]
 * verbose: int
 * output: bytearray
 */

static PyObject *
method_encode (PyObject *self, PyObject *args)
{
  /* Object to hold the input integer list. */
  PyObject *input_list_object;

  /* Object to hold the input verbosity integer, the verbose argument
   * is optional so defaulted to 0.
   */
  int verbose = 0;

  /* Arguments to the method are delivered as a tuple, unpack the
   * tuple to get the individual arguments, note the second is
   * optional.
   */
  if (!PyArg_ParseTuple(args, "O|i", &input_list_object, &verbose))
    return NULL;

  /* Unpack the length of the input integer list.  */
  int input_length = PyObject_Length (input_list_object);
  if (input_length < 0)
    input_length = 0;

  /* We need to marshall the integer list into an input buffer
   * suitable for mlw_encode, use a temporary heap allocated buffer
   * for that purpose.
   */
  int16_t *input_buffer = (int16_t *) malloc(sizeof(int16_t *) * input_length);
  if (input_buffer == NULL)
    return PyErr_NoMemory();

  /* Unpack the input integer list into the temporary buffer.
   */
  for (int i = 0; i < input_length; i++)
    {
      PyObject *item;
      item = PyList_GetItem(input_list_object, i);
      if (!PyLong_Check(item))
        input_buffer[i] = 0;
      input_buffer[i] = PyLong_AsLong(item);
    }

  /* We don't know the output length required, we guess worst case,
   * the mlw_encode call will do a resize (downwards) anyway.
   */
  uint8_t *output_buffer = malloc(input_length);
  if (output_buffer == NULL)
    return PyErr_NoMemory();

  int output_length = mlw_encode(input_buffer, input_length, &output_buffer, verbose);

  PyObject *output_byte_array = PyByteArray_FromStringAndSize ((char *) output_buffer, output_length);

  /* Discard the temporary input and output buffers.  */
  free (input_buffer);
  free (output_buffer);

  return output_byte_array;
}

/* C extension wrapper for mlw_decode
 *
 * This method is exposed directly in python with the arguments with a
 * prototype of the form:
 *
 * output = mlw_codec.decode(input, verbose=0)
 *
 * input: bytearray
 * verbose: int
 * output: [int]
 */

static PyObject *
method_decode(PyObject *self, PyObject *args)
{
  /* Object to hold the input bytearray. */
  PyObject *input_bytearray_object;

  /* Object to hold the input verbosity integer, the verbose argument
   * is optional so defaulted to 0.
   */
  int verbose = 0;

  /* Arguments to the method are delivered as a tuple, unpack the
   * tuple to get the individual arguments, note the second is
   * optional.
   */
  if (!PyArg_ParseTuple(args, "Y|i", &input_bytearray_object, &verbose))
    return NULL;

  /* Unpack the input buffer and length from the bytearray object.  */
  uint8_t *input_buffer = (uint8_t *) PyByteArray_AsString(input_bytearray_object);
  int input_length = PyByteArray_Size(input_bytearray_object);

  /* We don't know the output length required, we guess, but the guess
   * will be too small, the mlw_decode call will do a resize (upwards)
   * anyway.
   */
  int16_t *output_buffer = malloc (input_length);
  if (output_buffer == NULL)
    return PyErr_NoMemory();

  int output_length = mlw_decode (input_buffer, input_length, &output_buffer, verbose);

  /* Construct a new integer list and marshall the output buffer
   * contents into the list.  */
  PyObject *output_list = PyList_New(output_length);
  for (int i = 0; i <output_length; i++)
    PyList_SetItem (output_list, i, PyLong_FromLong (output_buffer[i]));

  free (output_buffer);

  return output_list;
}

/* mlw_codec method descriptors.
 */

static PyMethodDef mlw_methods[] = {
    {"decode", method_decode, METH_VARARGS, "Python interface for decode"},
    {"encode", method_encode, METH_VARARGS, "Python interface for encode"},
    {NULL, NULL, 0, NULL}
};

/* mlw_codec module descriptor.
 */

static struct PyModuleDef mlw_codecmodule = {
    PyModuleDef_HEAD_INIT,
    "mlw_codec",
    "Python interface for the mlw encoder",
    -1,
    mlw_methods
};

PyMODINIT_FUNC PyInit_mlw_codec(void) {
    return PyModule_Create(&mlw_codecmodule);
}
