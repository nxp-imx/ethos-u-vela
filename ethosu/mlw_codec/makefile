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
# Makefile to build mlw_codec

UNAME=$(shell uname -o)

CFLAGS=-Wall -Werror -Wno-unused-variable
CFLAGS+=-fwrapv -fstack-protector-strong -flto -fuse-linker-plugin -ffat-lto-objects

ifeq ($(DEBUG),1)
    CFLAGS+=-g -O0 -DDEBUG
else
    CFLAGS+=-O3
endif

LIBSRCS=mlw_encode.c mlw_decode.c
LIBHDRS=mlw_encode.h mlw_decode.h mlw_common.h

ifeq ($(UNAME),Cygwin)
    MLWEXE=mlw_codec.exe
else
    MLWEXE=mlw_codec
endif

all: mlwexe

.PHONY: mlwexe
mlwexe: $(MLWEXE)

clean:
	rm -f $(MLWEXE)

$(MLWEXE): mlw_main.c $(LIBSRCS) $(LIBHDRS) makefile
	gcc $(CFLAGS) mlw_main.c $(LIBSRCS) -o $(MLWEXE) -lm
