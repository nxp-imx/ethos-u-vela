/*
 * SPDX-FileCopyrightText: Copyright 2020-2021, 2023 Arm Limited and/or its affiliates <open-source-office@arm.com>
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
#include <numpy/ndarrayobject.h>

#include "mlw_decode.h"
#include "mlw_encode.h"

/* C extension wrapper for mlw_reorder_encode
 *
 * This method is exposed directly in python with the arguments with a
 * prototype of the form:
 *
 * output = mlw_codec.reorder_encode(
 *  ifm_ublock_depth,
 *  ofm_ublock_depth,
 *  input,
 *  ofm_block_depth,
 *  is_depthwise,
 *  is_partkernel,
 *  ifm_bitdepth,
 *  decomp_h,
 *  decomp_w,
 *  verbose=0)
 *
 * output: (bytearray, int)
 */

static PyObject *
method_reorder_encode (PyObject *self, PyObject *args)
{
    /* Object to hold the input integer list. */
    int ifm_ublock_depth;
    int ofm_ublock_depth;
    PyObject *input_object;
    int ofm_block_depth;
    int is_depthwise;
    int is_partkernel;
    int ifm_bitdepth;
    int decomp_h;
    int decomp_w;

    /* Object to hold the input verbosity integer, the verbose argument
     * is optional so defaulted to 0.
     */
    int verbose = 0;

    /* Arguments to the method are delivered as a tuple, unpack the
     * tuple to get the individual arguments, note the second is
     * optional.
     */
    if (!PyArg_ParseTuple(args, "iiOiiiiii|i",
        &ifm_ublock_depth,
        &ofm_ublock_depth,
        &input_object,
        &ofm_block_depth,
        &is_depthwise,
        &is_partkernel,
        &ifm_bitdepth,
        &decomp_h,
        &decomp_w,
        &verbose))
        return NULL;

    PyArrayObject* input_ndarray_object = (PyArrayObject*)PyArray_FROM_OTF(
        input_object,
        NPY_INT16,
        NPY_ARRAY_ALIGNED);
    if (input_ndarray_object == NULL)
    {
        return NULL;
    }

    if ((int)PyArray_NDIM(input_ndarray_object) < 4)
    {
        PyErr_SetString(PyExc_ValueError, "Invalid input shape");
        return NULL;
    }

    int ofm_depth = (int)PyArray_DIM(input_ndarray_object, 0);
    int kernel_height = (int)PyArray_DIM(input_ndarray_object, 1);
    int kernel_width = (int)PyArray_DIM(input_ndarray_object, 2);
    int ifm_depth = (int)PyArray_DIM(input_ndarray_object, 3);

    int16_t* brick_weights = (int16_t*)PyArray_DATA(input_ndarray_object);
    int brick_strides[4];
    for (int i = 0; i < 4; i++)
    {
        int stride = (int)PyArray_STRIDE(input_ndarray_object, i);
        if (stride % sizeof(int16_t))
        {
            PyErr_SetString(PyExc_ValueError, "Invalid stride");
            return NULL;
        }
        brick_strides[i] = stride / sizeof(int16_t);
    }
    if ((unsigned)PyArray_ITEMSIZE(input_ndarray_object) != sizeof(int16_t))
    {
        PyErr_SetString(PyExc_ValueError, "Invalid input type");
        return NULL;
    }
    uint8_t* output_buffer = NULL;
    int64_t padded_length;

    int output_length = mlw_reorder_encode(
        ifm_ublock_depth,
        ofm_ublock_depth,
        ofm_depth,
        kernel_height,
        kernel_width,
        ifm_depth,
        brick_strides,
        brick_weights,
        ofm_block_depth,
        is_depthwise,
        is_partkernel,
        ifm_bitdepth,
        decomp_h,
        decomp_w,
        &output_buffer,
        &padded_length,
        verbose);

    PyObject *output_byte_array = PyByteArray_FromStringAndSize((char*)output_buffer, output_length);
    PyObject *padded_length_obj = Py_BuildValue("i", padded_length);

    /* Discard the output buffer */
    mlw_free_outbuf(output_buffer);

    PyObject* ret = PyTuple_Pack(2, output_byte_array, padded_length_obj);

    Py_DECREF(input_ndarray_object);
    Py_DECREF(output_byte_array);
    Py_DECREF(padded_length_obj);
    return ret;
}

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
  Py_ssize_t input_length = PyObject_Length (input_list_object);
  if (input_length < 0 || input_length > INT32_MAX) {
    return NULL;
  }

  /* We need to marshall the integer list into an input buffer
   * suitable for mlw_encode, use a temporary heap allocated buffer
   * for that purpose.
   */
  int16_t *input_buffer = (int16_t *) malloc(sizeof(int16_t *) * input_length);
  uint8_t *output_buffer = NULL;
  if (input_buffer == NULL)
    return PyErr_NoMemory();

  /* Unpack the input integer list into the temporary buffer.
   */
  for (int i = 0; i < input_length; i++)
    {
      PyObject *item;
      item = PyList_GetItem(input_list_object, i);
      long value = PyLong_AsLong(item);
      if (value < -255 || value > 255) {
        PyErr_SetString(PyExc_ValueError, "Input value out of bounds");
        return NULL;
      }
      input_buffer[i] = (int16_t)value;
    }
  if (PyErr_Occurred() != NULL) {
    PyErr_SetString(PyExc_ValueError, "Invalid input");
    return NULL;
  }

  int output_length = mlw_encode(input_buffer, (int)input_length, &output_buffer, verbose);

  PyObject *output_byte_array = PyByteArray_FromStringAndSize ((char *) output_buffer, output_length);

  /* Discard the temporary input and output buffers.  */
  free (input_buffer);
  mlw_free_outbuf(output_buffer);

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
  Py_ssize_t input_length = PyByteArray_Size(input_bytearray_object);
  if (input_length < 0 || input_length > INT32_MAX) {
    return NULL;
  }

  /* We don't know the output length required, we guess, but the guess
   * will be too small, the mlw_decode call will do a resize (upwards)
   * anyway.
   */
  int16_t *output_buffer = (int16_t *) malloc (input_length);
  if (output_buffer == NULL)
    return PyErr_NoMemory();

  int output_length = mlw_decode (input_buffer, (int)input_length, &output_buffer, verbose);

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
    {"reorder_encode", method_reorder_encode, METH_VARARGS, "Python interface for reorder and encode"},
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

PyMODINIT_FUNC PyInit_mlw_codec(void)
{
    PyObject *ptype, *pvalue, *ptraceback;
    PyObject* ret = PyModule_Create(&mlw_codecmodule);
    if (_import_array() < 0)
    {
      // Fetch currently set error
      PyErr_Fetch(&ptype, &pvalue, &ptraceback);
      // Extract the error message
      const char *pStrErrorMessage = PyUnicode_AsUTF8(pvalue);
      // Re-format error message to start with "mlw_codec Error: " so it is
      // clearer it comes from mlw_codec.
      PyErr_Format(PyExc_RuntimeError, "mlw_codec error: %s", pStrErrorMessage);
      return NULL;
    }

    return ret;
}
