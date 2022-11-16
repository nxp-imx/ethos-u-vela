/*
 * SPDX-FileCopyrightText: Copyright 2020 Arm Limited and/or its affiliates <open-source-office@arm.com>
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

#include <stdint.h>

#ifndef __MLW_DECODE_H__
#define __MLW_DECODE_H__

#ifdef _MSC_VER
  #define EXPORTED __declspec(dllexport)
#else
  #define EXPORTED __attribute__((visibility("default")))
#endif

#if __cplusplus
extern "C"
{
#endif

EXPORTED
int mlw_decode(uint8_t *inbuf, int inbuf_size, int16_t **outbuf, int verbose);

#if __cplusplus
}
#endif

#endif
