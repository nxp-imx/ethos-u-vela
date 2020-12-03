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
# Makefile to build tensor_allocator_main

UNAME=$(shell uname -o)

CXXFLAGS=--std=c++11 -pedantic-errors -Wall -Werror -Wdate-time
CXXFLAGS+=-fwrapv -fstack-protector-strong -flto -fuse-linker-plugin -ffat-lto-objects -fPIC

ifeq ($(DEBUG),1)
    CXXFLAGS+=-g -O0 -DDEBUG
else
    CXXFLAGS+=-O2
endif

LIBSRCS=tensor_allocator_main.cpp search_allocator.cpp
LIBHDRS=search_allocator.h

ifeq ($(UNAME),Cygwin)
    TENSOR_ALLOCATOR_EXE=tensor_allocator_main.exe
else
    TENSOR_ALLOCATOR_EXE=tensor_allocator_main
endif

all: tensor_allocator_exe

.PHONY: tensor_allocator_exe
tensor_allocator_exe: $(TENSOR_ALLOCATOR_EXE)

clean:
	rm -f $(TENSOR_ALLOCATOR_EXE)

$(TENSOR_ALLOCATOR_EXE): $(LIBSRCS) $(LIBHDRS) makefile
	g++ $(CXXFLAGS) $(LIBSRCS) -o $(TENSOR_ALLOCATOR_EXE)
