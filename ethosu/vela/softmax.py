# SPDX-FileCopyrightText: Copyright 2020-2023 Arm Limited and/or its affiliates <open-source-office@arm.com>
#
# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
#
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Description:
# Contains SoftMax
import math

import numpy as np

from . import fp_math
from . import scaling
from .data_type import DataType
from .debug_database import DebugDatabase
from .operation import ActivationFunction
from .operation import ExplicitScaling
from .operation import Op
from .operation import Operation
from .operation import RoundingMode
from .operation_util import create_add
from .operation_util import create_clz
from .operation_util import create_depthwise_maxpool
from .operation_util import create_mul
from .operation_util import create_reduce_sum
from .operation_util import create_shl
from .operation_util import create_shr
from .operation_util import create_sub
from .shape4d import Shape4D
from .tensor import create_const_tensor
from .tensor import TensorPurpose


class SoftMax:
    # Turn off black formatting for the LUT tables to keep them compact
    # fmt: off

    EXP_LUT = [
        0x00000002, 0x00000002, 0x00000002, 0x00000002, 0x00000002, 0x00000002, 0x00000002, 0x00000002,
        0x00000002, 0x00000002, 0x00000002, 0x00000002, 0x00000002, 0x00000002, 0x00000002, 0x00000002,
        0x00000002, 0x00000002, 0x00000002, 0x00000002, 0x00000002, 0x00000002, 0x00000002, 0x00000002,
        0x00000002, 0x00000002, 0x00010002, 0x00000003, 0x00000003, 0x00000003, 0x00000003, 0x00000003,
        0x00000003, 0x00000003, 0x00000003, 0x00000003, 0x00000003, 0x00000003, 0x00000003, 0x00000003,
        0x00000003, 0x00000003, 0x00000003, 0x00010003, 0x00000004, 0x00000004, 0x00000004, 0x00000004,
        0x00000004, 0x00000004, 0x00000004, 0x00000004, 0x00000004, 0x00000004, 0x00000004, 0x00000004,
        0x00010004, 0x00000005, 0x00000005, 0x00000005, 0x00000005, 0x00000005, 0x00000005, 0x00000005,
        0x00000005, 0x00000005, 0x00010005, 0x00000006, 0x00000006, 0x00000006, 0x00000006, 0x00000006,
        0x00000006, 0x00000006, 0x00010006, 0x00000007, 0x00000007, 0x00000007, 0x00000007, 0x00000007,
        0x00000007, 0x00000007, 0x00010007, 0x00000008, 0x00000008, 0x00000008, 0x00000008, 0x00000008,
        0x00010008, 0x00000009, 0x00000009, 0x00000009, 0x00000009, 0x00000009, 0x00010009, 0x0000000a,
        0x0000000a, 0x0000000a, 0x0000000a, 0x0001000a, 0x0000000b, 0x0000000b, 0x0000000b, 0x0000000b,
        0x0001000b, 0x0000000c, 0x0000000c, 0x0000000c, 0x0001000c, 0x0000000d, 0x0000000d, 0x0000000d,
        0x0001000d, 0x0000000e, 0x0000000e, 0x0000000e, 0x0001000e, 0x0000000f, 0x0000000f, 0x0001000f,
        0x00000010, 0x00000010, 0x00010010, 0x00000011, 0x00000011, 0x00010011, 0x00000012, 0x00000012,
        0x00010012, 0x00000013, 0x00000013, 0x00010013, 0x00000014, 0x00010014, 0x00000015, 0x00000015,
        0x00010015, 0x00000016, 0x00010016, 0x00000017, 0x00010017, 0x00000018, 0x00010018, 0x00000019,
        0x00010019, 0x0000001a, 0x0001001a, 0x0000001b, 0x0001001b, 0x0000001c, 0x0001001c, 0x0000001d,
        0x0001001d, 0x0000001e, 0x0001001e, 0x0001001f, 0x00000020, 0x00010020, 0x00010021, 0x00000022,
        0x00010022, 0x00010023, 0x00000024, 0x00010024, 0x00000025, 0x00010025, 0x00010026, 0x00010027,
        0x00000028, 0x00020028, 0x0000002a, 0x0001002a, 0x0001002b, 0x0001002c, 0x0000002d, 0x0001002d,
        0x0001002e, 0x0001002f, 0x00010030, 0x00010031, 0x00010032, 0x00010033, 0x00010034, 0x00010035,
        0x00010036, 0x00010037, 0x00010038, 0x00020039, 0x0001003b, 0x0000003c, 0x0002003c, 0x0001003e,
        0x0002003f, 0x00000041, 0x00020041, 0x00010043, 0x00010044, 0x00020045, 0x00020047, 0x00010049,
        0x0001004a, 0x0002004b, 0x0001004d, 0x0002004e, 0x00010050, 0x00020051, 0x00020053, 0x00010055,
        0x00020056, 0x00020058, 0x0002005a, 0x0001005c, 0x0002005d, 0x0002005f, 0x00020061, 0x00020063,
        0x00020065, 0x00020067, 0x00020069, 0x0002006b, 0x0003006d, 0x00020070, 0x00020072, 0x00020074,
        0x00030076, 0x00020079, 0x0003007b, 0x0002007e, 0x00030080, 0x00020083, 0x00020085, 0x00040087,
        0x0002008b, 0x0003008d, 0x00030090, 0x00020093, 0x00030095, 0x00030098, 0x0003009b, 0x0004009e,
        0x000300a2, 0x000300a5, 0x000300a8, 0x000300ab, 0x000400ae, 0x000300b2, 0x000400b5, 0x000400b9,
        0x000300bd, 0x000400c0, 0x000400c4, 0x000400c8, 0x000400cc, 0x000400d0, 0x000500d4, 0x000400d9,
        0x000400dd, 0x000500e1, 0x000400e6, 0x000500ea, 0x000400ef, 0x000500f3, 0x000500f8, 0x000500fd,
        0x00050102, 0x00050107, 0x0005010c, 0x00060111, 0x00050117, 0x0006011c, 0x00060122, 0x00060128,
        0x0006012e, 0x00060134, 0x0006013a, 0x00070140, 0x00060147, 0x0007014d, 0x00060154, 0x0007015a,
        0x00070161, 0x00060168, 0x0008016e, 0x00070176, 0x0008017d, 0x00080185, 0x0007018d, 0x00090194,
        0x0008019d, 0x000801a5, 0x000801ad, 0x000901b5, 0x000901be, 0x000901c7, 0x000901d0, 0x000901d9,
        0x000a01e2, 0x000901ec, 0x000a01f5, 0x000b01ff, 0x000a020a, 0x000b0214, 0x000a021f, 0x000b0229,
        0x000b0234, 0x000b023f, 0x000c024a, 0x000c0256, 0x000c0262, 0x000c026e, 0x000c027a, 0x000d0286,
        0x000d0293, 0x000d02a0, 0x000e02ad, 0x000e02bb, 0x000e02c9, 0x000e02d7, 0x000f02e5, 0x000f02f4,
        0x000f0303, 0x000f0312, 0x00100321, 0x00100331, 0x00110341, 0x00100352, 0x00120362, 0x00110374,
        0x00120385, 0x00120397, 0x001203a9, 0x001303bb, 0x001303ce, 0x001403e1, 0x001403f5, 0x00140409,
        0x0015041d, 0x00150432, 0x00160447, 0x0016045d, 0x00160473, 0x00170489, 0x001704a0, 0x001904b7,
        0x001804d0, 0x001904e8, 0x00190501, 0x001a051a, 0x001a0534, 0x001b054e, 0x001b0569, 0x001c0584,
        0x001c05a0, 0x001d05bc, 0x001e05d9, 0x001e05f7, 0x001e0615, 0x00200633, 0x00200653, 0x00200673,
        0x00210693, 0x002206b4, 0x002306d6, 0x002306f9, 0x0024071c, 0x00240740, 0x00260764, 0x0026078a,
        0x002607b0, 0x002807d6, 0x002907fe, 0x00290827, 0x002a0850, 0x002a087a, 0x002c08a4, 0x002c08d0,
        0x002e08fc, 0x002e092a, 0x002f0958, 0x00310987, 0x003109b8, 0x003209e9, 0x00330a1b, 0x00340a4e,
        0x00350a82, 0x00350ab7, 0x00380aec, 0x00380b24, 0x003a0b5c, 0x003a0b96, 0x003c0bd0, 0x003d0c0c,
        0x003e0c49, 0x003f0c87, 0x00400cc6, 0x00420d06, 0x00430d48, 0x00440d8b, 0x00460dcf, 0x00480e15,
        0x00480e5d, 0x00490ea5, 0x004c0eee, 0x004d0f3a, 0x004e0f87, 0x00500fd5, 0x00511025, 0x00531076,
        0x005610c9, 0x0056111f, 0x00581175, 0x005a11cd, 0x005c1227, 0x005e1283, 0x005e12e1, 0x0061133f,
        0x006413a0, 0x00651404, 0x00671469, 0x006914d0, 0x006c1539, 0x006c15a5, 0x00701611, 0x00721681,
        0x007416f3, 0x00761767, 0x007917dd, 0x007a1856, 0x007d18d0, 0x0080194d, 0x008319cd, 0x00841a50,
        0x00881ad4, 0x00891b5c, 0x008d1be5, 0x00911c72, 0x00911d03, 0x00961d94, 0x00981e2a, 0x009c1ec2,
        0x009e1f5e, 0x00a21ffc, 0x00a4209e, 0x00a92142, 0x00ab21eb, 0x00ae2296, 0x00b22344, 0x00b523f6,
        0x00b924ab, 0x00be2564, 0x00c02622, 0x00c526e2, 0x00c827a7, 0x00cc286f, 0x00d0293b, 0x00d52a0b,
        0x00d72ae0, 0x00dd2bb7, 0x00e12c94, 0x00e62d75, 0x00eb2e5b, 0x00ef2f46, 0x00f23035, 0x00f83127,
        0x00fe321f, 0x0101331d, 0x0108341e, 0x010c3526, 0x01123632, 0x01173744, 0x011c385b, 0x01233977,
        0x01273a9a, 0x012e3bc1, 0x01343cef, 0x013a3e23, 0x01403f5d, 0x0146409d, 0x014c41e3, 0x0154432f,
        0x01594483, 0x016145dc, 0x0168473d, 0x016f48a5, 0x01764a14, 0x017d4b8a, 0x01854d07, 0x018d4e8c,
        0x01945019, 0x019d51ad, 0x01a4534a, 0x01ad54ee, 0x01b5569b, 0x01be5850, 0x01c75a0e, 0x01d05bd5,
        0x01d85da5, 0x01e35f7d, 0x01eb6160, 0x01f6634b, 0x01ff6541, 0x02096740, 0x02146949, 0x021e6b5d,
        0x02296d7b, 0x02336fa4, 0x023f71d7, 0x024a7416, 0x02567660, 0x026278b6, 0x026d7b18, 0x027a7d85,
    ]

    ONE_OVER_ONE_PLUS_X_LUT = [
        0xffc17fff, 0xffc07fc0, 0xffc27f80, 0xffc07f42, 0xffc17f02, 0xffc17ec3, 0xffc27e84, 0xffc27e46,
        0xffc27e08, 0xffc37dca, 0xffc27d8d, 0xffc37d4f, 0xffc37d12, 0xffc37cd5, 0xffc37c98, 0xffc47c5b,
        0xffc47c1f, 0xffc47be3, 0xffc57ba7, 0xffc57b6c, 0xffc37b31, 0xffc67af4, 0xffc57aba, 0xffc67a7f,
        0xffc57a45, 0xffc67a0a, 0xffc779d0, 0xffc67997, 0xffc6795d, 0xffc77923, 0xffc778ea, 0xffc778b1,
        0xffc87878, 0xffc77840, 0xffc87807, 0xffc877cf, 0xffc97797, 0xffc87760, 0xffc97728, 0xffc976f1,
        0xffc976ba, 0xffc87683, 0xffca764b, 0xffca7615, 0xffca75df, 0xffca75a9, 0xffca7573, 0xffcb753d,
        0xffca7508, 0xffcb74d2, 0xffcb749d, 0xffca7468, 0xffcc7432, 0xffcc73fe, 0xffcb73ca, 0xffcc7395,
        0xffcd7361, 0xffcc732e, 0xffcc72fa, 0xffcd72c6, 0xffcd7293, 0xffcd7260, 0xffcc722d, 0xffce71f9,
        0xffcd71c7, 0xffce7194, 0xffce7162, 0xffce7130, 0xffcf70fe, 0xffce70cd, 0xffce709b, 0xffcf7069,
        0xffcf7038, 0xffcf7007, 0xffcf6fd6, 0xffcf6fa5, 0xffd06f74, 0xffd06f44, 0xffd06f14, 0xffd06ee4,
        0xffd06eb4, 0xffd06e84, 0xffd16e54, 0xffd16e25, 0xffd16df6, 0xffd16dc7, 0xffd06d98, 0xffd26d68,
        0xffd16d3a, 0xffd26d0b, 0xffd26cdd, 0xffd26caf, 0xffd26c81, 0xffd26c53, 0xffd36c25, 0xffd26bf8,
        0xffd36bca, 0xffd36b9d, 0xffd36b70, 0xffd26b43, 0xffd46b15, 0xffd36ae9, 0xffd46abc, 0xffd46a90,
        0xffd46a64, 0xffd46a38, 0xffd46a0c, 0xffd469e0, 0xffd469b4, 0xffd56988, 0xffd5695d, 0xffd56932,
        0xffd56907, 0xffd568dc, 0xffd568b1, 0xffd56886, 0xffd6685b, 0xffd56831, 0xffd66806, 0xffd667dc,
        0xffd667b2, 0xffd76788, 0xffd6675f, 0xffd76735, 0xffd6670c, 0xffd766e2, 0xffd666b9, 0xffd7668f,
        0xffd86666, 0xffd6663e, 0xffd86614, 0xffd765ec, 0xffd865c3, 0xffd8659b, 0xffd86573, 0xffd8654b,
        0xffd86523, 0xffd864fb, 0xffd964d3, 0xffd864ac, 0xffd96484, 0xffd8645d, 0xffd96435, 0xffd9640e,
        0xffd963e7, 0xffd963c0, 0xffd96399, 0xffda6372, 0xffd9634c, 0xffda6325, 0xffda62ff, 0xffda62d9,
        0xffda62b3, 0xffda628d, 0xffda6267, 0xffdb6241, 0xffda621c, 0xffdb61f6, 0xffda61d1, 0xffdc61ab,
        0xffd96187, 0xffdc6160, 0xffdb613c, 0xffdb6117, 0xffdb60f2, 0xffdc60cd, 0xffdc60a9, 0xffdb6085,
        0xffdc6060, 0xffdc603c, 0xffdc6018, 0xffdc5ff4, 0xffdc5fd0, 0xffdd5fac, 0xffdc5f89, 0xffdc5f65,
        0xffdd5f41, 0xffdd5f1e, 0xffdd5efb, 0xffdd5ed8, 0xffdd5eb5, 0xffdd5e92, 0xffdd5e6f, 0xffdd5e4c,
        0xffdd5e29, 0xffde5e06, 0xffde5de4, 0xffdd5dc2, 0xffde5d9f, 0xffde5d7d, 0xffde5d5b, 0xffde5d39,
        0xffdf5d17, 0xffde5cf6, 0xffde5cd4, 0xffdf5cb2, 0xffdf5c91, 0xffde5c70, 0xffdf5c4e, 0xffdf5c2d,
        0xffde5c0c, 0xffe05bea, 0xffdf5bca, 0xffdf5ba9, 0xffdf5b88, 0xffdf5b67, 0xffe05b46, 0xffe05b26,
        0xffdf5b06, 0xffe05ae5, 0xffe05ac5, 0xffe05aa5, 0xffe05a85, 0xffe05a65, 0xffe05a45, 0xffe15a25,
        0xffe05a06, 0xffe059e6, 0xffe159c6, 0xffe159a7, 0xffe05988, 0xffe15968, 0xffe15949, 0xffe1592a,
        0xffe1590b, 0xffe158ec, 0xffe258cd, 0xffe158af, 0xffe15890, 0xffe25871, 0xffe15853, 0xffe25834,
        0xffe25816, 0xffe257f8, 0xffe157da, 0xffe257bb, 0xffe3579d, 0xffe25780, 0xffe25762, 0xffe25744,
        0xffe35726, 0xffe25709, 0xffe256eb, 0xffe356cd, 0xffe356b0, 0xffe35693, 0xffe25676, 0xffe35658,
        0xffe3563b, 0xffe3561e, 0xffe35601, 0xffe355e4, 0xffe455c7, 0xffe355ab, 0xffe4558e, 0xffe35572,
        0xffe45555, 0xffe35539, 0xffe4551c, 0xffe45500, 0xffe454e4, 0xffe454c8, 0xffe454ac, 0xffe45490,
        0xffe45474, 0xffe55458, 0xffe4543d, 0xffe45421, 0xffe55405, 0xffe553ea, 0xffe453cf, 0xffe553b3,
        0xffe45398, 0xffe5537c, 0xffe55361, 0xffe55346, 0xffe5532b, 0xffe55310, 0xffe552f5, 0xffe552da,
        0xffe652bf, 0xffe552a5, 0xffe5528a, 0xffe6526f, 0xffe55255, 0xffe6523a, 0xffe65220, 0xffe55206,
        0xffe651eb, 0xffe651d1, 0xffe651b7, 0xffe6519d, 0xffe65183, 0xffe65169, 0xffe7514f, 0xffe65136,
        0xffe6511c, 0xffe75102, 0xffe650e9, 0xffe750cf, 0xffe650b6, 0xffe7509c, 0xffe75083, 0xffe6506a,
        0xffe75050, 0xffe75037, 0xffe7501e, 0xffe75005, 0xffe74fec, 0xffe74fd3, 0xffe74fba, 0xffe74fa1,
        0xffe84f88, 0xffe74f70, 0xffe84f57, 0xffe74f3f, 0xffe84f26, 0xffe74f0e, 0xffe84ef5, 0xffe84edd,
        0xffe84ec5, 0xffe84ead, 0xffe74e95, 0xffe84e7c, 0xffe84e64, 0xffe94e4c, 0xffe84e35, 0xffe84e1d,
        0xffe84e05, 0xffe94ded, 0xffe84dd6, 0xffe84dbe, 0xffe94da6, 0xffe94d8f, 0xffe84d78, 0xffe84d60,
        0xffea4d48, 0xffe84d32, 0xffe94d1a, 0xffe94d03, 0xffe84cec, 0xffe94cd4, 0xffe94cbd, 0xffea4ca6,
        0xffe94c90, 0xffe84c79, 0xffea4c61, 0xffe94c4b, 0xffe94c34, 0xffea4c1d, 0xffe94c07, 0xffea4bf0,
        0xffe94bda, 0xffea4bc3, 0xffea4bad, 0xffe94b97, 0xffea4b80, 0xffea4b6a, 0xffea4b54, 0xffea4b3e,
        0xffea4b28, 0xffea4b12, 0xffea4afc, 0xffea4ae6, 0xffea4ad0, 0xffeb4aba, 0xffea4aa5, 0xffea4a8f,
        0xffeb4a79, 0xffea4a64, 0xffea4a4e, 0xffeb4a38, 0xffeb4a23, 0xffea4a0e, 0xffeb49f8, 0xffea49e3,
        0xffeb49cd, 0xffeb49b8, 0xffeb49a3, 0xffeb498e, 0xffea4979, 0xffeb4963, 0xffeb494e, 0xffec4939,
        0xffeb4925, 0xffea4910, 0xffec48fa, 0xffeb48e6, 0xffeb48d1, 0xffec48bc, 0xffeb48a8, 0xffec4893,
        0xffeb487f, 0xffec486a, 0xffeb4856, 0xffec4841, 0xffec482d, 0xffeb4819, 0xffec4804, 0xffec47f0,
        0xffec47dc, 0xffec47c8, 0xffec47b4, 0xffec47a0, 0xffec478c, 0xffec4778, 0xffec4764, 0xffec4750,
        0xffec473c, 0xffed4728, 0xffec4715, 0xffec4701, 0xffed46ed, 0xffec46da, 0xffed46c6, 0xffec46b3,
        0xffec469f, 0xffed468b, 0xffed4678, 0xffec4665, 0xffed4651, 0xffed463e, 0xffed462b, 0xffec4618,
        0xffed4604, 0xffed45f1, 0xffed45de, 0xffed45cb, 0xffed45b8, 0xffed45a5, 0xffed4592, 0xffed457f,
        0xffee456c, 0xffed455a, 0xffed4547, 0xffed4534, 0xffee4521, 0xffed450f, 0xffed44fc, 0xffee44e9,
        0xffed44d7, 0xffee44c4, 0xffee44b2, 0xffed44a0, 0xffee448d, 0xffee447b, 0xffed4469, 0xffee4456,
        0xffee4444, 0xffee4432, 0xffee4420, 0xffee440e, 0xffee43fc, 0xffee43ea, 0xffee43d8, 0xffee43c6,
        0xffee43b4, 0xffee43a2, 0xffee4390, 0xffef437e, 0xffee436d, 0xffee435b, 0xffef4349, 0xffee4338,
        0xffee4326, 0xffef4314, 0xffee4303, 0xffef42f1, 0xffee42e0, 0xffef42ce, 0xffee42bd, 0xffef42ab,
        0xffef429a, 0xffee4289, 0xfff04277, 0xffee4267, 0xffef4255, 0xffef4244, 0xffef4233, 0xffef4222,
        0xffee4211, 0xffef41ff, 0xfff041ee, 0xffef41de, 0xffef41cd, 0xffee41bc, 0xfff041aa, 0xffef419a,
        0xffef4189, 0xffef4178, 0xfff04167, 0xffef4157, 0xffef4146, 0xfff04135, 0xffef4125, 0xfff04114,
        0xffef4104, 0xfff040f3, 0xffef40e3, 0xfff040d2, 0xfff040c2, 0xffef40b2, 0xfff040a1, 0xfff04091,
        0xfff04081, 0xffef4071, 0xfff04060, 0xfff04050, 0xfff04040, 0xfff04030, 0xfff04020, 0xfff04010
    ]
    # fmt: on

    def __init__(self, op):
        self.op = op

    def generate_exp_table(self, beta, input_scale):
        integer_bits = 5
        total_signed_bits = 31
        # Calculate scaling
        real_beta = min(
            np.double(beta) * np.double(input_scale) * (1 << (31 - integer_bits)), np.double((1 << 31) - 1.0)
        )
        scale, shift = scaling.quantise_scale(real_beta)
        shift = 31 - shift
        diff_min = -1.0 * math.floor(
            1.0 * ((1 << integer_bits) - 1) * (1 << (total_signed_bits - integer_bits)) / (1 << shift)
        )
        # Generate the exp LUT
        lut = []
        for x in range(256):
            input_diff = x - 255
            if input_diff >= diff_min:
                rescale = fp_math.saturating_rounding_mul32(input_diff * (1 << shift), scale)
                lut.append(fp_math.exp_on_negative_values(rescale))
            else:
                lut.append(0)
        return lut

    def get_graph(self):
        ifm = self.op.inputs[0]
        ofm = self.op.outputs[0]

        # Reshape ifm/ofm (if needed)
        ifm_shape = self.op.ifm_shapes[0]
        if ifm_shape.batch > 1:
            self.op.ifm_shapes[0] = ifm_shape.with_height(ifm_shape.batch * ifm_shape.height).with_batch(1)
            self.op.ofm_shapes[0] = self.op.ifm_shapes[0]

        if ifm.dtype in (DataType.uint8, DataType.int8) and ofm.dtype == ifm.dtype:
            return self.get_graph_8bit(ifm, ofm)
        elif ifm.dtype == DataType.int16 and ofm.dtype == DataType.int16:
            return self.get_graph_int16(ifm, ofm)
        else:
            self.op.run_on_npu = False
            return self.op

    def get_graph_8bit(self, ifm, ofm):
        exp_lut = self.generate_exp_table(self.op.attrs.get("beta", 1.0), ifm.quantization.scale_f32)
        no_scale_quant = ifm.quantization.clone()
        no_scale_quant.scale_f32 = None
        activation = ActivationFunction(Op.Clip)
        activation.min = ifm.quantization.quant_min
        activation.max = ifm.quantization.quant_max
        activation2 = activation.clone()
        activation2.min = 2 * ifm.quantization.quant_min
        activation2.max = 2 * ifm.quantization.quant_max
        one_scale_quant = ifm.quantization.clone()
        one_scale_quant.scale_f32 = 1.0
        one_scale_quant.zero_point = 0
        two_scale_quant = one_scale_quant.clone()
        two_scale_quant.scale_f32 = 2.0
        pass_number = 0

        def add_op_get_ofm(op):
            DebugDatabase.add_optimised(self.op, op)
            nonlocal pass_number
            pass_number += 1
            return op.ofm

        # PASS 0 - Depthwise Maxpool
        ifm_shape = self.op.ifm_shapes[0]
        ifm_max = add_op_get_ofm(
            create_depthwise_maxpool(f"{self.op.name}_maxpool{pass_number}", ifm, ifm_shape, no_scale_quant)
        )

        # PASS 1 - Sub+LUT(exp)
        sub_op_quantization = one_scale_quant.clone()
        sub_op_quantization.zero_point = 127
        ifm_max_shape = Shape4D([1, ifm_shape.height, ifm_shape.width, 1])
        sub_op = create_sub(
            f"{self.op.name}_sub{pass_number}",
            ifm,
            ifm_max,
            sub_op_quantization,
            dtype=DataType.int32,
            ifm_shape=ifm_shape,
            ifm2_shape=ifm_max_shape,
        )
        sub_op.set_activation_lut(
            create_const_tensor(f"{sub_op.name}_exp_lut", [1, 1, 1, 256], DataType.uint32, exp_lut, TensorPurpose.LUT)
        )
        ifm_exp = add_op_get_ofm(sub_op)
        # Note: activation.min/max are non-quantized values
        sub_op.activation.min = -128 - ifm_exp.quantization.zero_point
        sub_op.activation.max = 127 - ifm_exp.quantization.zero_point

        # PASS 2 - SHR
        name = f"{self.op.name}_shr{pass_number}"
        shift = create_const_tensor(f"{name}_const", [1, 1, 1, 1], DataType.int32, [12], quantization=no_scale_quant)
        shr_op = create_shr(name, ifm_exp, shift, no_scale_quant, activation)
        shr_op.rounding_mode = RoundingMode.HalfUp
        rescaled_exp = add_op_get_ofm(shr_op)

        # PASS 3 - Reduce sum
        sum_of_exp = add_op_get_ofm(
            create_reduce_sum(f"{self.op.name}_reduce_sum{pass_number}", rescaled_exp, no_scale_quant, activation)
        )

        # PASS 4 - CLZ
        headroom_plus_one = add_op_get_ofm(
            create_clz(f"{self.op.name}_clz{pass_number}", sum_of_exp, no_scale_quant, activation)
        )

        # PASS 5 - Sub
        headroom_offset = create_const_tensor(
            "headroom_offset_const",
            [1, 1, 1, 1],
            DataType.int32,
            [12 + 31 - 8],
            quantization=no_scale_quant,
        )
        right_shift = add_op_get_ofm(
            create_sub(
                f"{self.op.name}_sub{pass_number}",
                headroom_offset,
                headroom_plus_one,
                no_scale_quant,
                activation,
            )
        )

        # PASS 6 - Sub
        one = create_const_tensor("one_const", [1, 1, 1, 1], DataType.int32, [1], quantization=no_scale_quant)
        headroom = add_op_get_ofm(
            create_sub(f"{self.op.name}_sub{pass_number}", headroom_plus_one, one, no_scale_quant, activation)
        )

        # PASS 7 - SHL
        shifted_sum = add_op_get_ofm(
            create_shl(f"{self.op.name}_shl{pass_number}", sum_of_exp, headroom, no_scale_quant, activation)
        )

        # PASS 8 - Sub
        shifted_one = create_const_tensor(
            "shifted_one_const", [1, 1, 1, 1], DataType.int32, [1 << 30], quantization=no_scale_quant
        )
        shifted_sum_minus_one = add_op_get_ofm(
            create_sub(f"{self.op.name}_sub{pass_number}", shifted_sum, shifted_one, no_scale_quant, activation)
        )

        # PASS 9 - SHL
        shifted_sum_minus_one = add_op_get_ofm(
            create_shl(
                f"{self.op.name}_shl{pass_number}",
                shifted_sum_minus_one,
                one,
                no_scale_quant,
                activation,
            )
        )

        # PASS 10 - Add
        f0_one_const = create_const_tensor(
            "F0_one_const", [1, 1, 1, 1], DataType.int32, [(1 << 31) - 1], quantization=no_scale_quant
        )
        add_op = create_add(
            f"{self.op.name}_add{pass_number}",
            shifted_sum_minus_one,
            f0_one_const,
            one_scale_quant,
            activation,
        )
        add_op.explicit_scaling = ExplicitScaling(False, shift=[1], multiplier=[1])  # Custom rescale
        half_denominator = add_op_get_ofm(add_op)

        # PASS 11 - Multiply
        neg_32_over_17 = create_const_tensor(
            "neg_32_over_17_const", [1, 1, 1, 1], DataType.int32, [-1010580540], quantization=one_scale_quant
        )
        rescaled = add_op_get_ofm(
            create_mul(
                f"{self.op.name}_mul{pass_number}",
                half_denominator,
                neg_32_over_17,
                two_scale_quant,
                activation2,
            )
        )

        # PASS 12 - Add
        const_48_over_17 = create_const_tensor(
            "48_over_17_const", [1, 1, 1, 1], DataType.int32, [1515870810], quantization=no_scale_quant
        )
        rescale_w_offset = add_op_get_ofm(
            create_add(
                f"{self.op.name}_add{pass_number}",
                rescaled,
                const_48_over_17,
                one_scale_quant,
                activation,
            )
        )

        # PASS 13 - 27
        nr_x = rescale_w_offset
        F2_one = create_const_tensor(
            "F2_one_const", [1, 1, 1, 1], DataType.int32, [(1 << 29)], quantization=no_scale_quant
        )
        four = create_const_tensor("four_const", [1, 1, 1, 1], DataType.int32, [4], quantization=no_scale_quant)
        for _ in range(3):
            # PASS 13, 18, 23 - MUL
            half_denominator_times_x = add_op_get_ofm(
                create_mul(
                    f"{self.op.name}_mul{pass_number}",
                    nr_x,
                    half_denominator,
                    two_scale_quant,
                    activation2,
                )
            )
            # PASS 14, 19, 24 - SUB
            one_minus_half_denominator_times_x = add_op_get_ofm(
                create_sub(
                    f"{self.op.name}_sub{pass_number}",
                    F2_one,
                    half_denominator_times_x,
                    one_scale_quant,
                    activation,
                )
            )
            # PASS 15, 20, 25 - MUL
            to_rescale = add_op_get_ofm(
                create_mul(
                    f"{self.op.name}_mul{pass_number}",
                    nr_x,
                    one_minus_half_denominator_times_x,
                    two_scale_quant,
                    activation2,
                )
            )
            # PASS 16, 21, 26 - MUL
            to_add = add_op_get_ofm(
                create_mul(f"{self.op.name}_mul{pass_number}", to_rescale, four, no_scale_quant, activation)
            )
            # PASS 17, 22, 27 - ADD
            nr_x = add_op_get_ofm(
                create_add(f"{self.op.name}_add{pass_number}", nr_x, to_add, one_scale_quant, activation)
            )

        # PASS 28 - Multiply
        two = create_const_tensor("two_const", [1, 1, 1, 1], DataType.int32, [2], quantization=no_scale_quant)
        scale_factor = add_op_get_ofm(
            create_mul(f"{self.op.name}_mul{pass_number}", nr_x, two, one_scale_quant, activation)
        )

        # PASS 29 - Multiply
        scaled_exp = add_op_get_ofm(
            create_mul(f"{self.op.name}_mul{pass_number}", ifm_exp, scale_factor, two_scale_quant, activation2)
        )

        # PASS 30 - SHR
        shr30_op = Operation(Op.SHR, f"{self.op.name}_shr{pass_number}")
        shr30_op.rounding_mode = RoundingMode.HalfUp
        shr30_op.add_input_tensor(scaled_exp)
        shr30_op.add_input_tensor(right_shift)
        shr30_op.set_output_tensor(ofm)
        shr30_op.ifm_shapes.append(Shape4D(scaled_exp.shape))
        shr30_op.ifm_shapes.append(Shape4D(right_shift.shape))
        shr30_op.ofm_shapes.append(Shape4D(scaled_exp.shape))
        DebugDatabase.add_optimised(self.op, shr30_op)

        return shr30_op

    def get_graph_int16(self, ifm, ofm):
        no_scale_quant = ifm.quantization.clone()
        no_scale_quant.scale_f32 = None
        pass_number = 0

        def add_op_get_ofm(op):
            DebugDatabase.add_optimised(self.op, op)
            nonlocal pass_number
            pass_number += 1
            return op.ofm

        # PASS 0 - Depthwise Maxpool
        ifm_shape = self.op.ifm_shapes[0]
        ifm_max = add_op_get_ofm(
            create_depthwise_maxpool(f"{self.op.name}_maxpool{pass_number}", ifm, ifm_shape, no_scale_quant)
        )

        # PASS 1 - Sub
        ifm_max_shape = Shape4D([1, ifm_shape.height, ifm_shape.width, 1])
        sub1_ofm = add_op_get_ofm(
            create_sub(
                f"{self.op.name}_sub{pass_number}",
                ifm,
                ifm_max,
                ifm.quantization.clone(),
                dtype=DataType.int32,
                ifm_shape=ifm_shape,
                ifm2_shape=ifm_max_shape,
            )
        )

        # PASS 2 - Mul
        name = f"{self.op.name}_mul{pass_number}"
        beta = self.op.attrs.get("beta", 1.0)
        mul2_out_range = 10.0 / 65535.0
        mul2_scale, _ = scaling.elementwise_mul_scale(sub1_ofm.quantization.scale_f32, beta, mul2_out_range)
        scale_quant = ifm.quantization.clone()
        scale_quant.scale_f32 = beta
        mul2_quant = ofm.quantization.clone()
        mul2_quant.scale_f32 = mul2_out_range
        scale = create_const_tensor(
            f"{name}_scale_const", [1, 1, 1, 1], DataType.int32, [mul2_scale], quantization=scale_quant
        )
        mul2_ofm = add_op_get_ofm(create_mul(name, sub1_ofm, scale, mul2_quant))

        # PASS 3 - Add+LUT(exp)
        name = f"{self.op.name}_add{pass_number}"
        const_add = create_const_tensor(
            f"{name}_const", [1, 1, 1, 1], DataType.int32, [32767], quantization=no_scale_quant
        )
        add_op = create_add(name, mul2_ofm, const_add, mul2_ofm.quantization.clone(), dtype=DataType.int16)
        # lut activation values are int32 type however they are defined as Python ints. If these are converted to
        # numpy.int32 it could result in an overflow error. Therefore, they are forced to uint32 to avoid this
        add_op.set_activation_lut(
            create_const_tensor(f"{name}_exp_lut", [1, 1, 1, 512], DataType.uint32, self.EXP_LUT, TensorPurpose.LUT)
        )
        ifm_exp = add_op_get_ofm(add_op)

        # PASS 4 - Reduce sum
        sum_of_exp = add_op_get_ofm(
            create_reduce_sum(f"{self.op.name}_reduce_sum{pass_number}", ifm_exp, no_scale_quant)
        )

        # PASS 5 - CLZ
        headroom_plus_one = add_op_get_ofm(create_clz(f"{self.op.name}_clz{pass_number}", sum_of_exp, no_scale_quant))

        # PASS 6 - Sub
        name = f"{self.op.name}_sub{pass_number}"
        const_31 = create_const_tensor(f"{name}_const", [1, 1, 1, 1], DataType.int32, [31], quantization=no_scale_quant)
        reciprocal_right_shift = add_op_get_ofm(create_sub(name, const_31, headroom_plus_one, no_scale_quant))

        # PASS 7 - SHL
        one = create_const_tensor("one_const", [1, 1, 1, 1], DataType.int32, [1], quantization=no_scale_quant)
        constant_one = add_op_get_ofm(
            create_shl(f"{self.op.name}_shl{pass_number}", one, reciprocal_right_shift, no_scale_quant)
        )

        # PASS 8 - Sub
        sum_of_exps_minus_one = add_op_get_ofm(
            create_sub(f"{self.op.name}_sub{pass_number}", sum_of_exp, constant_one, no_scale_quant)
        )

        # PASS 9 - SHL
        shifted_sum_minus_one = add_op_get_ofm(
            create_shl(f"{self.op.name}_shl{pass_number}", sum_of_exps_minus_one, headroom_plus_one, no_scale_quant)
        )

        # PASS 10 - SHR
        name = f"{self.op.name}_shr{pass_number}"
        shift = create_const_tensor(f"{name}_const", [1, 1, 1, 1], DataType.int32, [15], quantization=no_scale_quant)
        shifted_sum_minus_one_16 = add_op_get_ofm(create_shr(name, shifted_sum_minus_one, shift, no_scale_quant))

        # PASS 11 - Sub+LUT(one over one plus x)
        name = f"{self.op.name}_sub{pass_number}"
        sub11_const = create_const_tensor(
            f"{name}_const", [1, 1, 1, 1], DataType.int32, [32768], quantization=no_scale_quant
        )
        sub11_op = create_sub(name, shifted_sum_minus_one_16, sub11_const, no_scale_quant, dtype=DataType.int16)
        # lut activation values are int32 type however they are defined as Python ints. If these are converted to
        # numpy.int32 it could result in an overflow error. Therefore, they are forced to uint32 to avoid this
        sub11_op.set_activation_lut(
            create_const_tensor(
                f"{name}_one_over_one_plus_x_lut",
                [1, 1, 1, 512],
                DataType.uint32,
                self.ONE_OVER_ONE_PLUS_X_LUT,
                TensorPurpose.LUT,
            )
        )
        reciprocal_scale = add_op_get_ofm(sub11_op)

        # PASS 12 - Multiply
        mul_ofm = add_op_get_ofm(
            create_mul(
                f"{self.op.name}_mul{pass_number}", ifm_exp, reciprocal_scale, no_scale_quant, dtype=DataType.int32
            )
        )

        # PASS 13 - SHR
        shr13_op = Operation(Op.SHR, f"{self.op.name}_shr{pass_number}")
        shr13_op.add_input_tensor(mul_ofm)
        shr13_op.add_input_tensor(reciprocal_right_shift)
        shr13_op.set_output_tensor(ofm)
        shr13_op.ifm_shapes.append(Shape4D(mul_ofm.shape))
        shr13_op.ifm_shapes.append(Shape4D(reciprocal_right_shift.shape))
        shr13_op.ofm_shapes.append(Shape4D(mul_ofm.shape))
        DebugDatabase.add_optimised(self.op, shr13_op)

        return shr13_op
