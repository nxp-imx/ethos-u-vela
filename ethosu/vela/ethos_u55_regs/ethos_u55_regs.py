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



from ctypes import *
from enum import Enum

ARCH_VER = '1.0.6'


class BASE(Enum):
    ID = 0x0000
    STATUS = 0x0004
    CMD = 0x0008
    RESET = 0x000C
    QBASE0 = 0x0010
    QBASE1 = 0x0014
    QREAD = 0x0018
    QCONFIG = 0x001C
    QSIZE = 0x0020
    PROT = 0x0024
    CONFIG = 0x0028
    LOCK = 0x002C
    REGIONCFG = 0x003C
    AXI_LIMIT0 = 0x0040
    AXI_LIMIT1 = 0x0044
    AXI_LIMIT2 = 0x0048
    AXI_LIMIT3 = 0x004C
    SIZE = 0x0050

class BASE_POINTERS(Enum):
    BASEP0 = 0x0080
    BASEP1 = 0x0084
    BASEP2 = 0x0088
    BASEP3 = 0x008C
    BASEP4 = 0x0090
    BASEP5 = 0x0094
    BASEP6 = 0x0098
    BASEP7 = 0x009C
    BASEP8 = 0x00A0
    BASEP9 = 0x00A4
    BASEP10 = 0x00A8
    BASEP11 = 0x00AC
    BASEP12 = 0x00B0
    BASEP13 = 0x00B4
    BASEP14 = 0x00B8
    BASEP15 = 0x00BC
    SIZE = 0x00C0

class DEBUG(Enum):
    WD_STATUS = 0x0100
    MAC_STATUS = 0x0104
    AO_STATUS = 0x0108
    DMA_STATUS0 = 0x0110
    DMA_STATUS1 = 0x0114
    CLKFORCE = 0x0140
    DEBUG_ADDRESS = 0x0144
    DEBUG_MISC = 0x0148
    DEBUGCORE = 0x014C
    DEBUG_BLOCK = 0x0150
    SIZE = 0x0154

class ID(Enum):
    REVISION = 0x0FC0
    PID4 = 0x0FD0
    PID5 = 0x0FD4
    PID6 = 0x0FD8
    PID7 = 0x0FDC
    PID0 = 0x0FE0
    PID1 = 0x0FE4
    PID2 = 0x0FE8
    PID3 = 0x0FEC
    CID0 = 0x0FF0
    CID1 = 0x0FF4
    CID2 = 0x0FF8
    CID3 = 0x0FFC
    SIZE = 0x1000

class PMU(Enum):
    PMCR = 0x0180
    PMCNTENSET = 0x0184
    PMCNTENCLR = 0x0188
    PMOVSSET = 0x018C
    PMOVSCLR = 0x0190
    PMINTSET = 0x0194
    PMINTCLR = 0x0198
    PMCCNTR_LO = 0x01A0
    PMCCNTR_HI = 0x01A4
    PMCCNTR_CFG = 0x01A8
    PMCAXI_CHAN = 0x01AC
    PMEVCNTR0 = 0x0300
    PMEVCNTR1 = 0x0304
    PMEVCNTR2 = 0x0308
    PMEVCNTR3 = 0x030C
    PMEVTYPER0 = 0x0380
    PMEVTYPER1 = 0x0384
    PMEVTYPER2 = 0x0388
    PMEVTYPER3 = 0x038C
    SIZE = 0x0390

class SHARED_BUFFER(Enum):
    SHARED_BUFFER0 = 0x0400
    SHARED_BUFFER1 = 0x0404
    SHARED_BUFFER2 = 0x0408
    SHARED_BUFFER3 = 0x040C
    SHARED_BUFFER4 = 0x0410
    SHARED_BUFFER5 = 0x0414
    SHARED_BUFFER6 = 0x0418
    SHARED_BUFFER7 = 0x041C
    SHARED_BUFFER8 = 0x0420
    SHARED_BUFFER9 = 0x0424
    SHARED_BUFFER10 = 0x0428
    SHARED_BUFFER11 = 0x042C
    SHARED_BUFFER12 = 0x0430
    SHARED_BUFFER13 = 0x0434
    SHARED_BUFFER14 = 0x0438
    SHARED_BUFFER15 = 0x043C
    SHARED_BUFFER16 = 0x0440
    SHARED_BUFFER17 = 0x0444
    SHARED_BUFFER18 = 0x0448
    SHARED_BUFFER19 = 0x044C
    SHARED_BUFFER20 = 0x0450
    SHARED_BUFFER21 = 0x0454
    SHARED_BUFFER22 = 0x0458
    SHARED_BUFFER23 = 0x045C
    SHARED_BUFFER24 = 0x0460
    SHARED_BUFFER25 = 0x0464
    SHARED_BUFFER26 = 0x0468
    SHARED_BUFFER27 = 0x046C
    SHARED_BUFFER28 = 0x0470
    SHARED_BUFFER29 = 0x0474
    SHARED_BUFFER30 = 0x0478
    SHARED_BUFFER31 = 0x047C
    SHARED_BUFFER32 = 0x0480
    SHARED_BUFFER33 = 0x0484
    SHARED_BUFFER34 = 0x0488
    SHARED_BUFFER35 = 0x048C
    SHARED_BUFFER36 = 0x0490
    SHARED_BUFFER37 = 0x0494
    SHARED_BUFFER38 = 0x0498
    SHARED_BUFFER39 = 0x049C
    SHARED_BUFFER40 = 0x04A0
    SHARED_BUFFER41 = 0x04A4
    SHARED_BUFFER42 = 0x04A8
    SHARED_BUFFER43 = 0x04AC
    SHARED_BUFFER44 = 0x04B0
    SHARED_BUFFER45 = 0x04B4
    SHARED_BUFFER46 = 0x04B8
    SHARED_BUFFER47 = 0x04BC
    SHARED_BUFFER48 = 0x04C0
    SHARED_BUFFER49 = 0x04C4
    SHARED_BUFFER50 = 0x04C8
    SHARED_BUFFER51 = 0x04CC
    SHARED_BUFFER52 = 0x04D0
    SHARED_BUFFER53 = 0x04D4
    SHARED_BUFFER54 = 0x04D8
    SHARED_BUFFER55 = 0x04DC
    SHARED_BUFFER56 = 0x04E0
    SHARED_BUFFER57 = 0x04E4
    SHARED_BUFFER58 = 0x04E8
    SHARED_BUFFER59 = 0x04EC
    SHARED_BUFFER60 = 0x04F0
    SHARED_BUFFER61 = 0x04F4
    SHARED_BUFFER62 = 0x04F8
    SHARED_BUFFER63 = 0x04FC
    SHARED_BUFFER64 = 0x0500
    SHARED_BUFFER65 = 0x0504
    SHARED_BUFFER66 = 0x0508
    SHARED_BUFFER67 = 0x050C
    SHARED_BUFFER68 = 0x0510
    SHARED_BUFFER69 = 0x0514
    SHARED_BUFFER70 = 0x0518
    SHARED_BUFFER71 = 0x051C
    SHARED_BUFFER72 = 0x0520
    SHARED_BUFFER73 = 0x0524
    SHARED_BUFFER74 = 0x0528
    SHARED_BUFFER75 = 0x052C
    SHARED_BUFFER76 = 0x0530
    SHARED_BUFFER77 = 0x0534
    SHARED_BUFFER78 = 0x0538
    SHARED_BUFFER79 = 0x053C
    SHARED_BUFFER80 = 0x0540
    SHARED_BUFFER81 = 0x0544
    SHARED_BUFFER82 = 0x0548
    SHARED_BUFFER83 = 0x054C
    SHARED_BUFFER84 = 0x0550
    SHARED_BUFFER85 = 0x0554
    SHARED_BUFFER86 = 0x0558
    SHARED_BUFFER87 = 0x055C
    SHARED_BUFFER88 = 0x0560
    SHARED_BUFFER89 = 0x0564
    SHARED_BUFFER90 = 0x0568
    SHARED_BUFFER91 = 0x056C
    SHARED_BUFFER92 = 0x0570
    SHARED_BUFFER93 = 0x0574
    SHARED_BUFFER94 = 0x0578
    SHARED_BUFFER95 = 0x057C
    SHARED_BUFFER96 = 0x0580
    SHARED_BUFFER97 = 0x0584
    SHARED_BUFFER98 = 0x0588
    SHARED_BUFFER99 = 0x058C
    SHARED_BUFFER100 = 0x0590
    SHARED_BUFFER101 = 0x0594
    SHARED_BUFFER102 = 0x0598
    SHARED_BUFFER103 = 0x059C
    SHARED_BUFFER104 = 0x05A0
    SHARED_BUFFER105 = 0x05A4
    SHARED_BUFFER106 = 0x05A8
    SHARED_BUFFER107 = 0x05AC
    SHARED_BUFFER108 = 0x05B0
    SHARED_BUFFER109 = 0x05B4
    SHARED_BUFFER110 = 0x05B8
    SHARED_BUFFER111 = 0x05BC
    SHARED_BUFFER112 = 0x05C0
    SHARED_BUFFER113 = 0x05C4
    SHARED_BUFFER114 = 0x05C8
    SHARED_BUFFER115 = 0x05CC
    SHARED_BUFFER116 = 0x05D0
    SHARED_BUFFER117 = 0x05D4
    SHARED_BUFFER118 = 0x05D8
    SHARED_BUFFER119 = 0x05DC
    SHARED_BUFFER120 = 0x05E0
    SHARED_BUFFER121 = 0x05E4
    SHARED_BUFFER122 = 0x05E8
    SHARED_BUFFER123 = 0x05EC
    SHARED_BUFFER124 = 0x05F0
    SHARED_BUFFER125 = 0x05F4
    SHARED_BUFFER126 = 0x05F8
    SHARED_BUFFER127 = 0x05FC
    SHARED_BUFFER128 = 0x0600
    SHARED_BUFFER129 = 0x0604
    SHARED_BUFFER130 = 0x0608
    SHARED_BUFFER131 = 0x060C
    SHARED_BUFFER132 = 0x0610
    SHARED_BUFFER133 = 0x0614
    SHARED_BUFFER134 = 0x0618
    SHARED_BUFFER135 = 0x061C
    SHARED_BUFFER136 = 0x0620
    SHARED_BUFFER137 = 0x0624
    SHARED_BUFFER138 = 0x0628
    SHARED_BUFFER139 = 0x062C
    SHARED_BUFFER140 = 0x0630
    SHARED_BUFFER141 = 0x0634
    SHARED_BUFFER142 = 0x0638
    SHARED_BUFFER143 = 0x063C
    SHARED_BUFFER144 = 0x0640
    SHARED_BUFFER145 = 0x0644
    SHARED_BUFFER146 = 0x0648
    SHARED_BUFFER147 = 0x064C
    SHARED_BUFFER148 = 0x0650
    SHARED_BUFFER149 = 0x0654
    SHARED_BUFFER150 = 0x0658
    SHARED_BUFFER151 = 0x065C
    SHARED_BUFFER152 = 0x0660
    SHARED_BUFFER153 = 0x0664
    SHARED_BUFFER154 = 0x0668
    SHARED_BUFFER155 = 0x066C
    SHARED_BUFFER156 = 0x0670
    SHARED_BUFFER157 = 0x0674
    SHARED_BUFFER158 = 0x0678
    SHARED_BUFFER159 = 0x067C
    SHARED_BUFFER160 = 0x0680
    SHARED_BUFFER161 = 0x0684
    SHARED_BUFFER162 = 0x0688
    SHARED_BUFFER163 = 0x068C
    SHARED_BUFFER164 = 0x0690
    SHARED_BUFFER165 = 0x0694
    SHARED_BUFFER166 = 0x0698
    SHARED_BUFFER167 = 0x069C
    SHARED_BUFFER168 = 0x06A0
    SHARED_BUFFER169 = 0x06A4
    SHARED_BUFFER170 = 0x06A8
    SHARED_BUFFER171 = 0x06AC
    SHARED_BUFFER172 = 0x06B0
    SHARED_BUFFER173 = 0x06B4
    SHARED_BUFFER174 = 0x06B8
    SHARED_BUFFER175 = 0x06BC
    SHARED_BUFFER176 = 0x06C0
    SHARED_BUFFER177 = 0x06C4
    SHARED_BUFFER178 = 0x06C8
    SHARED_BUFFER179 = 0x06CC
    SHARED_BUFFER180 = 0x06D0
    SHARED_BUFFER181 = 0x06D4
    SHARED_BUFFER182 = 0x06D8
    SHARED_BUFFER183 = 0x06DC
    SHARED_BUFFER184 = 0x06E0
    SHARED_BUFFER185 = 0x06E4
    SHARED_BUFFER186 = 0x06E8
    SHARED_BUFFER187 = 0x06EC
    SHARED_BUFFER188 = 0x06F0
    SHARED_BUFFER189 = 0x06F4
    SHARED_BUFFER190 = 0x06F8
    SHARED_BUFFER191 = 0x06FC
    SHARED_BUFFER192 = 0x0700
    SHARED_BUFFER193 = 0x0704
    SHARED_BUFFER194 = 0x0708
    SHARED_BUFFER195 = 0x070C
    SHARED_BUFFER196 = 0x0710
    SHARED_BUFFER197 = 0x0714
    SHARED_BUFFER198 = 0x0718
    SHARED_BUFFER199 = 0x071C
    SHARED_BUFFER200 = 0x0720
    SHARED_BUFFER201 = 0x0724
    SHARED_BUFFER202 = 0x0728
    SHARED_BUFFER203 = 0x072C
    SHARED_BUFFER204 = 0x0730
    SHARED_BUFFER205 = 0x0734
    SHARED_BUFFER206 = 0x0738
    SHARED_BUFFER207 = 0x073C
    SHARED_BUFFER208 = 0x0740
    SHARED_BUFFER209 = 0x0744
    SHARED_BUFFER210 = 0x0748
    SHARED_BUFFER211 = 0x074C
    SHARED_BUFFER212 = 0x0750
    SHARED_BUFFER213 = 0x0754
    SHARED_BUFFER214 = 0x0758
    SHARED_BUFFER215 = 0x075C
    SHARED_BUFFER216 = 0x0760
    SHARED_BUFFER217 = 0x0764
    SHARED_BUFFER218 = 0x0768
    SHARED_BUFFER219 = 0x076C
    SHARED_BUFFER220 = 0x0770
    SHARED_BUFFER221 = 0x0774
    SHARED_BUFFER222 = 0x0778
    SHARED_BUFFER223 = 0x077C
    SHARED_BUFFER224 = 0x0780
    SHARED_BUFFER225 = 0x0784
    SHARED_BUFFER226 = 0x0788
    SHARED_BUFFER227 = 0x078C
    SHARED_BUFFER228 = 0x0790
    SHARED_BUFFER229 = 0x0794
    SHARED_BUFFER230 = 0x0798
    SHARED_BUFFER231 = 0x079C
    SHARED_BUFFER232 = 0x07A0
    SHARED_BUFFER233 = 0x07A4
    SHARED_BUFFER234 = 0x07A8
    SHARED_BUFFER235 = 0x07AC
    SHARED_BUFFER236 = 0x07B0
    SHARED_BUFFER237 = 0x07B4
    SHARED_BUFFER238 = 0x07B8
    SHARED_BUFFER239 = 0x07BC
    SHARED_BUFFER240 = 0x07C0
    SHARED_BUFFER241 = 0x07C4
    SHARED_BUFFER242 = 0x07C8
    SHARED_BUFFER243 = 0x07CC
    SHARED_BUFFER244 = 0x07D0
    SHARED_BUFFER245 = 0x07D4
    SHARED_BUFFER246 = 0x07D8
    SHARED_BUFFER247 = 0x07DC
    SHARED_BUFFER248 = 0x07E0
    SHARED_BUFFER249 = 0x07E4
    SHARED_BUFFER250 = 0x07E8
    SHARED_BUFFER251 = 0x07EC
    SHARED_BUFFER252 = 0x07F0
    SHARED_BUFFER253 = 0x07F4
    SHARED_BUFFER254 = 0x07F8
    SHARED_BUFFER255 = 0x07FC
    SIZE = 0x0800

class TSU(Enum):
    IFM_PAD_TOP = 0x0800
    IFM_PAD_LEFT = 0x0804
    IFM_PAD_RIGHT = 0x0808
    IFM_PAD_BOTTOM = 0x080C
    IFM_DEPTH_M1 = 0x0810
    IFM_PRECISION = 0x0814
    IFM_UPSCALE = 0x081C
    IFM_ZERO_POINT = 0x0824
    IFM_WIDTH0_M1 = 0x0828
    IFM_HEIGHT0_M1 = 0x082C
    IFM_HEIGHT1_M1 = 0x0830
    IFM_IB_END = 0x0834
    IFM_REGION = 0x083C
    OFM_WIDTH_M1 = 0x0844
    OFM_HEIGHT_M1 = 0x0848
    OFM_DEPTH_M1 = 0x084C
    OFM_PRECISION = 0x0850
    OFM_BLK_WIDTH_M1 = 0x0854
    OFM_BLK_HEIGHT_M1 = 0x0858
    OFM_BLK_DEPTH_M1 = 0x085C
    OFM_ZERO_POINT = 0x0860
    OFM_WIDTH0_M1 = 0x0868
    OFM_HEIGHT0_M1 = 0x086C
    OFM_HEIGHT1_M1 = 0x0870
    OFM_REGION = 0x087C
    KERNEL_WIDTH_M1 = 0x0880
    KERNEL_HEIGHT_M1 = 0x0884
    KERNEL_STRIDE = 0x0888
    PARALLEL_MODE = 0x088C
    ACC_FORMAT = 0x0890
    ACTIVATION = 0x0894
    ACTIVATION_MIN = 0x0898
    ACTIVATION_MAX = 0x089C
    WEIGHT_REGION = 0x08A0
    SCALE_REGION = 0x08A4
    AB_START = 0x08B4
    BLOCKDEP = 0x08BC
    DMA0_SRC_REGION = 0x08C0
    DMA0_DST_REGION = 0x08C4
    DMA0_SIZE0 = 0x08C8
    DMA0_SIZE1 = 0x08CC
    IFM2_BROADCAST = 0x0900
    IFM2_SCALAR = 0x0904
    IFM2_PRECISION = 0x0914
    IFM2_ZERO_POINT = 0x0924
    IFM2_WIDTH0_M1 = 0x0928
    IFM2_HEIGHT0_M1 = 0x092C
    IFM2_HEIGHT1_M1 = 0x0930
    IFM2_IB_START = 0x0934
    IFM2_REGION = 0x093C
    IFM_BASE0 = 0x0A00
    IFM_BASE0_HI = 0x0A04
    IFM_BASE1 = 0x0A08
    IFM_BASE1_HI = 0x0A0C
    IFM_BASE2 = 0x0A10
    IFM_BASE2_HI = 0x0A14
    IFM_BASE3 = 0x0A18
    IFM_BASE3_HI = 0x0A1C
    IFM_STRIDE_X = 0x0A20
    IFM_STRIDE_X_HI = 0x0A24
    IFM_STRIDE_Y = 0x0A28
    IFM_STRIDE_Y_HI = 0x0A2C
    IFM_STRIDE_C = 0x0A30
    IFM_STRIDE_C_HI = 0x0A34
    OFM_BASE0 = 0x0A40
    OFM_BASE0_HI = 0x0A44
    OFM_BASE1 = 0x0A48
    OFM_BASE1_HI = 0x0A4C
    OFM_BASE2 = 0x0A50
    OFM_BASE2_HI = 0x0A54
    OFM_BASE3 = 0x0A58
    OFM_BASE3_HI = 0x0A5C
    OFM_STRIDE_X = 0x0A60
    OFM_STRIDE_X_HI = 0x0A64
    OFM_STRIDE_Y = 0x0A68
    OFM_STRIDE_Y_HI = 0x0A6C
    OFM_STRIDE_C = 0x0A70
    OFM_STRIDE_C_HI = 0x0A74
    WEIGHT_BASE = 0x0A80
    WEIGHT_BASE_HI = 0x0A84
    WEIGHT_LENGTH = 0x0A88
    SCALE_BASE = 0x0A90
    SCALE_BASE_HI = 0x0A94
    SCALE_LENGTH = 0x0A98
    OFM_SCALE = 0x0AA0
    OFM_SCALE_SHIFT = 0x0AA4
    OPA_SCALE = 0x0AA8
    OPA_SCALE_SHIFT = 0x0AAC
    OPB_SCALE = 0x0AB0
    DMA0_SRC = 0x0AC0
    DMA0_SRC_HI = 0x0AC4
    DMA0_DST = 0x0AC8
    DMA0_DST_HI = 0x0ACC
    DMA0_LEN = 0x0AD0
    DMA0_LEN_HI = 0x0AD4
    DMA0_SKIP0 = 0x0AD8
    DMA0_SKIP0_HI = 0x0ADC
    DMA0_SKIP1 = 0x0AE0
    DMA0_SKIP1_HI = 0x0AE4
    IFM2_BASE0 = 0x0B00
    IFM2_BASE0_HI = 0x0B04
    IFM2_BASE1 = 0x0B08
    IFM2_BASE1_HI = 0x0B0C
    IFM2_BASE2 = 0x0B10
    IFM2_BASE2_HI = 0x0B14
    IFM2_BASE3 = 0x0B18
    IFM2_BASE3_HI = 0x0B1C
    IFM2_STRIDE_X = 0x0B20
    IFM2_STRIDE_X_HI = 0x0B24
    IFM2_STRIDE_Y = 0x0B28
    IFM2_STRIDE_Y_HI = 0x0B2C
    IFM2_STRIDE_C = 0x0B30
    IFM2_STRIDE_C_HI = 0x0B34
    WEIGHT1_BASE = 0x0B40
    WEIGHT1_BASE_HI = 0x0B44
    WEIGHT1_LENGTH = 0x0B48
    SCALE1_BASE = 0x0B50
    SCALE1_BASE_HI = 0x0B54
    SCALE1_LENGTH = 0x0B58
    SIZE = 0x0B5C

class TSU_DEBUG(Enum):
    KERNEL_X = 0x0200
    KERNEL_Y = 0x0204
    KERNEL_W_M1 = 0x0208
    KERNEL_H_M1 = 0x020C
    OFM_CBLK_WIDTH_M1 = 0x0210
    OFM_CBLK_HEIGHT_M1 = 0x0214
    OFM_CBLK_DEPTH_M1 = 0x0218
    IFM_CBLK_DEPTH_M1 = 0x021C
    OFM_X = 0x0220
    OFM_Y = 0x0224
    OFM_Z = 0x0228
    IFM_Z = 0x022C
    PAD_TOP = 0x0230
    PAD_LEFT = 0x0234
    IFM_CBLK_WIDTH = 0x0238
    IFM_CBLK_HEIGHT = 0x023C
    DMA_IFM_SRC = 0x0240
    DMA_IFM_SRC_HI = 0x0244
    DMA_IFM_DST = 0x0248
    DMA_OFM_SRC = 0x024C
    DMA_OFM_DST = 0x0250
    DMA_OFM_DST_HI = 0x0254
    DMA_WEIGHT_SRC = 0x0258
    DMA_WEIGHT_SRC_HI = 0x025C
    DMA_CMD_SRC = 0x0260
    DMA_CMD_SRC_HI = 0x0264
    DMA_CMD_SIZE = 0x0268
    DMA_M2M_SRC = 0x026C
    DMA_M2M_SRC_HI = 0x0270
    DMA_M2M_DST = 0x0274
    DMA_M2M_DST_HI = 0x0278
    CURRENT_QREAD = 0x027C
    DMA_SCALE_SRC = 0x0280
    DMA_SCALE_SRC_HI = 0x0284
    CURRENT_BLOCK = 0x02B4
    CURRENT_OP = 0x02B8
    CURRENT_CMD = 0x02BC
    SIZE = 0x02C0



class acc_format(Enum):
    INT_32BIT = 0
    INT_40BIT = 1
    FP_S5_10 = 2

class activation(Enum):
    NONE = 0
    TANH = 3
    SIGMOID = 4
    LUT_START = 16
    LUT_END = 23

class axi_mem_encoding_type(Enum):
    DEVICE_NON_BUFFERABLE = 0x0
    DEVICE_BUFFERABLE = 0x1
    NORMAL_NON_CACHEABLE_NON_BUFFERABLE = 0x2
    NORMAL_NON_CACHEABLE_BUFFERABLE = 0x3
    WRITE_THROUGH_NO_ALLOCATE = 0x4
    WRITE_THROUGH_READ_ALLOCATE = 0x5
    WRITE_THROUGH_WRITE_ALLOCATE = 0x6
    WRITE_THROUGH_READ_AND_WRITE_ALLOCATE = 0x7
    WRITE_BACK_NO_ALLOCATE = 0x8
    WRITE_BACK_READ_ALLOCATE = 0x9
    WRITE_BACK_WRITE_ALLOCATE = 0xA
    WRITE_BACK_READ_AND_WRITE_ALLOCATE = 0xB
    RESERVED_12 = 0xC
    RESERVED_13 = 0xD
    RESERVED_14 = 0xE
    RESERVED_15 = 0xF

class clip_range(Enum):
    OFM_PRECISION = 0
    FORCE_UINT8 = 2
    FORCE_INT8 = 3
    FORCE_INT16 = 5

class cmd0(Enum):
    NPU_OP_STOP = 0x000
    NPU_OP_IRQ = 0x001
    NPU_OP_CONV = 0x002
    NPU_OP_DEPTHWISE = 0x003
    NPU_OP_POOL = 0x005
    NPU_OP_ELEMENTWISE = 0x006
    NPU_OP_DMA_START = 0x010
    NPU_OP_DMA_WAIT = 0x011
    NPU_OP_KERNEL_WAIT = 0x012
    NPU_OP_PMU_MASK = 0x013
    NPU_SET_IFM_PAD_TOP = 0x100
    NPU_SET_IFM_PAD_LEFT = 0x101
    NPU_SET_IFM_PAD_RIGHT = 0x102
    NPU_SET_IFM_PAD_BOTTOM = 0x103
    NPU_SET_IFM_DEPTH_M1 = 0x104
    NPU_SET_IFM_PRECISION = 0x105
    NPU_SET_IFM_UPSCALE = 0x107
    NPU_SET_IFM_ZERO_POINT = 0x109
    NPU_SET_IFM_WIDTH0_M1 = 0x10A
    NPU_SET_IFM_HEIGHT0_M1 = 0x10B
    NPU_SET_IFM_HEIGHT1_M1 = 0x10C
    NPU_SET_IFM_IB_END = 0x10D
    NPU_SET_IFM_REGION = 0x10F
    NPU_SET_OFM_WIDTH_M1 = 0x111
    NPU_SET_OFM_HEIGHT_M1 = 0x112
    NPU_SET_OFM_DEPTH_M1 = 0x113
    NPU_SET_OFM_PRECISION = 0x114
    NPU_SET_OFM_BLK_WIDTH_M1 = 0x115
    NPU_SET_OFM_BLK_HEIGHT_M1 = 0x116
    NPU_SET_OFM_BLK_DEPTH_M1 = 0x117
    NPU_SET_OFM_ZERO_POINT = 0x118
    NPU_SET_OFM_WIDTH0_M1 = 0x11A
    NPU_SET_OFM_HEIGHT0_M1 = 0x11B
    NPU_SET_OFM_HEIGHT1_M1 = 0x11C
    NPU_SET_OFM_REGION = 0x11F
    NPU_SET_KERNEL_WIDTH_M1 = 0x120
    NPU_SET_KERNEL_HEIGHT_M1 = 0x121
    NPU_SET_KERNEL_STRIDE = 0x122
    NPU_SET_PARALLEL_MODE = 0x123
    NPU_SET_ACC_FORMAT = 0x124
    NPU_SET_ACTIVATION = 0x125
    NPU_SET_ACTIVATION_MIN = 0x126
    NPU_SET_ACTIVATION_MAX = 0x127
    NPU_SET_WEIGHT_REGION = 0x128
    NPU_SET_SCALE_REGION = 0x129
    NPU_SET_AB_START = 0x12D
    NPU_SET_BLOCKDEP = 0x12F
    NPU_SET_DMA0_SRC_REGION = 0x130
    NPU_SET_DMA0_DST_REGION = 0x131
    NPU_SET_DMA0_SIZE0 = 0x132
    NPU_SET_DMA0_SIZE1 = 0x133
    NPU_SET_IFM2_BROADCAST = 0x180
    NPU_SET_IFM2_SCALAR = 0x181
    NPU_SET_IFM2_PRECISION = 0x185
    NPU_SET_IFM2_ZERO_POINT = 0x189
    NPU_SET_IFM2_WIDTH0_M1 = 0x18A
    NPU_SET_IFM2_HEIGHT0_M1 = 0x18B
    NPU_SET_IFM2_HEIGHT1_M1 = 0x18C
    NPU_SET_IFM2_IB_START = 0x18D
    NPU_SET_IFM2_REGION = 0x18F

class cmd1(Enum):
    NPU_SET_IFM_BASE0 = 0x000
    NPU_SET_IFM_BASE1 = 0x001
    NPU_SET_IFM_BASE2 = 0x002
    NPU_SET_IFM_BASE3 = 0x003
    NPU_SET_IFM_STRIDE_X = 0x004
    NPU_SET_IFM_STRIDE_Y = 0x005
    NPU_SET_IFM_STRIDE_C = 0x006
    NPU_SET_OFM_BASE0 = 0x010
    NPU_SET_OFM_BASE1 = 0x011
    NPU_SET_OFM_BASE2 = 0x012
    NPU_SET_OFM_BASE3 = 0x013
    NPU_SET_OFM_STRIDE_X = 0x014
    NPU_SET_OFM_STRIDE_Y = 0x015
    NPU_SET_OFM_STRIDE_C = 0x016
    NPU_SET_WEIGHT_BASE = 0x020
    NPU_SET_WEIGHT_LENGTH = 0x021
    NPU_SET_SCALE_BASE = 0x022
    NPU_SET_SCALE_LENGTH = 0x023
    NPU_SET_OFM_SCALE = 0x024
    NPU_SET_OPA_SCALE = 0x025
    NPU_SET_OPB_SCALE = 0x026
    NPU_SET_DMA0_SRC = 0x030
    NPU_SET_DMA0_DST = 0x031
    NPU_SET_DMA0_LEN = 0x032
    NPU_SET_DMA0_SKIP0 = 0x033
    NPU_SET_DMA0_SKIP1 = 0x034
    NPU_SET_IFM2_BASE0 = 0x080
    NPU_SET_IFM2_BASE1 = 0x081
    NPU_SET_IFM2_BASE2 = 0x082
    NPU_SET_IFM2_BASE3 = 0x083
    NPU_SET_IFM2_STRIDE_X = 0x084
    NPU_SET_IFM2_STRIDE_Y = 0x085
    NPU_SET_IFM2_STRIDE_C = 0x086
    NPU_SET_WEIGHT1_BASE = 0x090
    NPU_SET_WEIGHT1_LENGTH = 0x091
    NPU_SET_SCALE1_BASE = 0x092
    NPU_SET_SCALE1_LENGTH = 0x093

class data_format(Enum):
    NHWC = 0
    NHCWB16 = 1

class elementwise_mode(Enum):
    MUL = 0
    ADD = 1
    SUB = 2
    MIN = 3
    MAX = 4
    LRELU = 5
    ABS = 6
    CLZ = 7
    SHR = 8
    SHL = 9

class ifm_precision(Enum):
    U8 = 0
    S8 = 1
    U16 = 4
    S16 = 5
    S32 = 9

class ifm_scale_mode(Enum):
    SCALE_16BIT = 0
    SCALE_OPA_32BIT = 1
    SCALE_OPB_32BIT = 2

class macs_per_cc(Enum):
    MACS_PER_CC_IS_5 = 0x5
    MACS_PER_CC_IS_6 = 0x6
    MACS_PER_CC_IS_7 = 0x7
    MACS_PER_CC_IS_8 = 0x8

class memory_type(Enum):
    AXI0_OUTSTANDING_COUNTER0 = 0
    AXI0_OUTSTANDING_COUNTER1 = 1
    AXI1_OUTSTANDING_COUNTER2 = 2
    AXI1_OUTSTANDING_COUNTER3 = 3

class ofm_precision(Enum):
    U8 = 0
    S8 = 1
    U16 = 2
    S16 = 3
    S32 = 5

class pmu_event_type(Enum):
    NO_EVENT = 0x00
    CYCLE = 0x11
    NPU_IDLE = 0x20
    CC_STALLED_ON_BLOCKDEP = 0x21
    CC_STALLED_ON_SHRAM_RECONFIG = 0x22
    NPU_ACTIVE = 0x23
    MAC_ACTIVE = 0x30
    MAC_ACTIVE_8BIT = 0x31
    MAC_ACTIVE_16BIT = 0x32
    MAC_DPU_ACTIVE = 0x33
    MAC_STALLED_BY_WD_ACC = 0x34
    MAC_STALLED_BY_WD = 0x35
    MAC_STALLED_BY_ACC = 0x36
    MAC_STALLED_BY_IB = 0x37
    MAC_ACTIVE_32BIT = 0x38
    MAC_STALLED_BY_INT_W = 0x39
    MAC_STALLED_BY_INT_ACC = 0x3A
    AO_ACTIVE = 0x40
    AO_ACTIVE_8BIT = 0x41
    AO_ACTIVE_16BIT = 0x42
    AO_STALLED_BY_OFMP_OB = 0x43
    AO_STALLED_BY_OFMP = 0x44
    AO_STALLED_BY_OB = 0x45
    AO_STALLED_BY_ACC_IB = 0x46
    AO_STALLED_BY_ACC = 0x47
    AO_STALLED_BY_IB = 0x48
    WD_ACTIVE = 0x50
    WD_STALLED = 0x51
    WD_STALLED_BY_WS = 0x52
    WD_STALLED_BY_WD_BUF = 0x53
    WD_PARSE_ACTIVE = 0x54
    WD_PARSE_STALLED = 0x55
    WD_PARSE_STALLED_IN = 0x56
    WD_PARSE_STALLED_OUT = 0x57
    WD_TRANS_WS = 0x58
    WD_TRANS_WB = 0x59
    WD_TRANS_DW0 = 0x5a
    WD_TRANS_DW1 = 0x5b
    AXI0_RD_TRANS_ACCEPTED = 0x80
    AXI0_RD_TRANS_COMPLETED = 0x81
    AXI0_RD_DATA_BEAT_RECEIVED = 0x82
    AXI0_RD_TRAN_REQ_STALLED = 0x83
    AXI0_WR_TRANS_ACCEPTED = 0x84
    AXI0_WR_TRANS_COMPLETED_M = 0x85
    AXI0_WR_TRANS_COMPLETED_S = 0x86
    AXI0_WR_DATA_BEAT_WRITTEN = 0x87
    AXI0_WR_TRAN_REQ_STALLED = 0x88
    AXI0_WR_DATA_BEAT_STALLED = 0x89
    AXI0_ENABLED_CYCLES = 0x8c
    AXI0_RD_STALL_LIMIT = 0x8e
    AXI0_WR_STALL_LIMIT = 0x8f
    AXI1_RD_TRANS_ACCEPTED = 0x180
    AXI1_RD_TRANS_COMPLETED = 0x181
    AXI1_RD_DATA_BEAT_RECEIVED = 0x182
    AXI1_RD_TRAN_REQ_STALLED = 0x183
    AXI1_WR_TRANS_ACCEPTED = 0x184
    AXI1_WR_TRANS_COMPLETED_M = 0x185
    AXI1_WR_TRANS_COMPLETED_S = 0x186
    AXI1_WR_DATA_BEAT_WRITTEN = 0x187
    AXI1_WR_TRAN_REQ_STALLED = 0x188
    AXI1_WR_DATA_BEAT_STALLED = 0x189
    AXI1_ENABLED_CYCLES = 0x18c
    AXI1_RD_STALL_LIMIT = 0x18e
    AXI1_WR_STALL_LIMIT = 0x18f
    AXI_LATENCY_ANY = 0xa0
    AXI_LATENCY_32 = 0xa1
    AXI_LATENCY_64 = 0xa2
    AXI_LATENCY_128 = 0xa3
    AXI_LATENCY_256 = 0xa4
    AXI_LATENCY_512 = 0xa5
    AXI_LATENCY_1024 = 0xa6
    ECC_DMA = 0xb0
    ECC_SB0 = 0xb1
    ECC_SB1 = 0x1b1

class pooling_mode(Enum):
    MAX = 0
    AVERAGE = 1
    REDUCE_SUM = 2

class privilege_level(Enum):
    USER = 0
    PRIVILEGED = 1

class resampling_mode(Enum):
    NONE = 0
    NEAREST = 1
    TRANSPOSE = 2

class rounding(Enum):
    TFL = 0
    TRUNCATE = 1
    NATURAL = 2

class security_level(Enum):
    SECURE = 0
    NON_SECURE = 1

class shram_size(Enum):
    SHRAM_96KB = 0x60
    SHRAM_48KB = 0x30
    SHRAM_24KB = 0x18
    SHRAM_16KB = 0x10

class state(Enum):
    STOPPED = 0
    RUNNING = 1

class stride_mode(Enum):
    STRIDE_MODE_1D = 0
    STRIDE_MODE_2D = 1
    STRIDE_MODE_3D = 2


class id_r(Union):
    class _bitfield(Structure):
        _fields_ = [
            ("version_status", c_uint32, 4),
            ("version_minor", c_uint32, 4),
            ("version_major", c_uint32, 4),
            ("product_major", c_uint32, 4),
            ("arch_patch_rev", c_uint32, 4),
            ("arch_minor_rev", c_uint32, 8),
            ("arch_major_rev", c_uint32, 4),
        ]
    _fields_ = [("bits", _bitfield),
                ("word", c_uint32)]
    def set_version_status(self, value): self.bits.version_status = value
    def get_version_status(self): value = self.bits.version_status; return value
    def set_version_minor(self, value): self.bits.version_minor = value
    def get_version_minor(self): value = self.bits.version_minor; return value
    def set_version_major(self, value): self.bits.version_major = value
    def get_version_major(self): value = self.bits.version_major; return value
    def set_product_major(self, value): self.bits.product_major = value
    def get_product_major(self): value = self.bits.product_major; return value
    def set_arch_patch_rev(self, value): self.bits.arch_patch_rev = value
    def get_arch_patch_rev(self): value = self.bits.arch_patch_rev; return value
    def set_arch_minor_rev(self, value): self.bits.arch_minor_rev = value
    def get_arch_minor_rev(self): value = self.bits.arch_minor_rev; return value
    def set_arch_major_rev(self, value): self.bits.arch_major_rev = value
    def get_arch_major_rev(self): value = self.bits.arch_major_rev; return value


class status_r(Union):
    class _bitfield(Structure):
        _fields_ = [
            ("state", c_uint32, 1),
            ("irq_raised", c_uint32, 1),
            ("bus_status", c_uint32, 1),
            ("reset_status", c_uint32, 1),
            ("cmd_parse_error", c_uint32, 1),
            ("cmd_end_reached", c_uint32, 1),
            ("pmu_irq_raised", c_uint32, 1),
            ("wd_fault", c_uint32, 1),
            ("ecc_fault", c_uint32, 1),
            ("reserved0", c_uint32, 2),
            ("faulting_interface", c_uint32, 1),
            ("faulting_channel", c_uint32, 4),
            ("irq_history_mask", c_uint32, 16),
        ]
    _fields_ = [("bits", _bitfield),
                ("word", c_uint32)]
    def set_state(self, value): self.bits.state = value
    def get_state(self): value = self.bits.state; return value
    def set_irq_raised(self, value): self.bits.irq_raised = value
    def get_irq_raised(self): value = self.bits.irq_raised; return value
    def set_bus_status(self, value): self.bits.bus_status = value
    def get_bus_status(self): value = self.bits.bus_status; return value
    def set_reset_status(self, value): self.bits.reset_status = value
    def get_reset_status(self): value = self.bits.reset_status; return value
    def set_cmd_parse_error(self, value): self.bits.cmd_parse_error = value
    def get_cmd_parse_error(self): value = self.bits.cmd_parse_error; return value
    def set_cmd_end_reached(self, value): self.bits.cmd_end_reached = value
    def get_cmd_end_reached(self): value = self.bits.cmd_end_reached; return value
    def set_pmu_irq_raised(self, value): self.bits.pmu_irq_raised = value
    def get_pmu_irq_raised(self): value = self.bits.pmu_irq_raised; return value
    def set_wd_fault(self, value): self.bits.wd_fault = value
    def get_wd_fault(self): value = self.bits.wd_fault; return value
    def set_ecc_fault(self, value): self.bits.ecc_fault = value
    def get_ecc_fault(self): value = self.bits.ecc_fault; return value
    def set_faulting_interface(self, value): self.bits.faulting_interface = value
    def get_faulting_interface(self): value = self.bits.faulting_interface; return value
    def set_faulting_channel(self, value): self.bits.faulting_channel = value
    def get_faulting_channel(self): value = self.bits.faulting_channel; return value
    def set_irq_history_mask(self, value): self.bits.irq_history_mask = value
    def get_irq_history_mask(self): value = self.bits.irq_history_mask; return value


class cmd_r(Union):
    class _bitfield(Structure):
        _fields_ = [
            ("transition_to_running_state", c_uint32, 1),
            ("clear_irq", c_uint32, 1),
            ("clock_q_enable", c_uint32, 1),
            ("power_q_enable", c_uint32, 1),
            ("stop_request", c_uint32, 1),
            ("reserved0", c_uint32, 11),
            ("clear_irq_history", c_uint32, 16),
        ]
    _fields_ = [("bits", _bitfield),
                ("word", c_uint32)]
    def set_transition_to_running_state(self, value): self.bits.transition_to_running_state = value
    def get_transition_to_running_state(self): value = self.bits.transition_to_running_state; return value
    def set_clear_irq(self, value): self.bits.clear_irq = value
    def get_clear_irq(self): value = self.bits.clear_irq; return value
    def set_clock_q_enable(self, value): self.bits.clock_q_enable = value
    def get_clock_q_enable(self): value = self.bits.clock_q_enable; return value
    def set_power_q_enable(self, value): self.bits.power_q_enable = value
    def get_power_q_enable(self): value = self.bits.power_q_enable; return value
    def set_stop_request(self, value): self.bits.stop_request = value
    def get_stop_request(self): value = self.bits.stop_request; return value
    def set_clear_irq_history(self, value): self.bits.clear_irq_history = value
    def get_clear_irq_history(self): value = self.bits.clear_irq_history; return value


class reset_r(Union):
    class _bitfield(Structure):
        _fields_ = [
            ("pending_cpl", c_uint32, 1),
            ("pending_csl", c_uint32, 1),
            ("reserved0", c_uint32, 30),
        ]
    _fields_ = [("bits", _bitfield),
                ("word", c_uint32)]
    def set_pending_cpl(self, value): self.bits.pending_cpl = value
    def get_pending_cpl(self): value = self.bits.pending_cpl; return value
    def set_pending_csl(self, value): self.bits.pending_csl = value
    def get_pending_csl(self): value = self.bits.pending_csl; return value


class qbase0_r(Union):
    class _bitfield(Structure):
        _fields_ = [
            ("qbase0", c_uint32, 32),
        ]
    _fields_ = [("bits", _bitfield),
                ("word", c_uint32)]
    def set_qbase0(self, value): self.bits.qbase0 = value
    def get_qbase0(self): value = self.bits.qbase0; return value


class qbase1_r(Union):
    class _bitfield(Structure):
        _fields_ = [
            ("qbase1", c_uint32, 32),
        ]
    _fields_ = [("bits", _bitfield),
                ("word", c_uint32)]
    def set_qbase1(self, value): self.bits.qbase1 = value
    def get_qbase1(self): value = self.bits.qbase1; return value


class qread_r(Union):
    class _bitfield(Structure):
        _fields_ = [
            ("qread", c_uint32, 32),
        ]
    _fields_ = [("bits", _bitfield),
                ("word", c_uint32)]
    def set_qread(self, value): self.bits.qread = value
    def get_qread(self): value = self.bits.qread; return value


class qconfig_r(Union):
    class _bitfield(Structure):
        _fields_ = [
            ("qconfig", c_uint32, 32),
        ]
    _fields_ = [("bits", _bitfield),
                ("word", c_uint32)]
    def set_qconfig(self, value): self.bits.qconfig = value
    def get_qconfig(self): value = self.bits.qconfig; return value


class qsize_r(Union):
    class _bitfield(Structure):
        _fields_ = [
            ("qsize", c_uint32, 32),
        ]
    _fields_ = [("bits", _bitfield),
                ("word", c_uint32)]
    def set_qsize(self, value): self.bits.qsize = value
    def get_qsize(self): value = self.bits.qsize; return value


class prot_r(Union):
    class _bitfield(Structure):
        _fields_ = [
            ("active_cpl", c_uint32, 1),
            ("active_csl", c_uint32, 1),
            ("reserved0", c_uint32, 30),
        ]
    _fields_ = [("bits", _bitfield),
                ("word", c_uint32)]
    def set_active_cpl(self, value): self.bits.active_cpl = value
    def get_active_cpl(self): value = self.bits.active_cpl; return value
    def set_active_csl(self, value): self.bits.active_csl = value
    def get_active_csl(self): value = self.bits.active_csl; return value


class config_r(Union):
    class _bitfield(Structure):
        _fields_ = [
            ("macs_per_cc", c_uint32, 4),
            ("cmd_stream_version", c_uint32, 4),
            ("shram_size", c_uint32, 8),
            ("reserved0", c_uint32, 12),
            ("product", c_uint32, 4),
        ]
    _fields_ = [("bits", _bitfield),
                ("word", c_uint32)]
    def set_macs_per_cc(self, value): self.bits.macs_per_cc = value
    def get_macs_per_cc(self): value = self.bits.macs_per_cc; return value
    def set_cmd_stream_version(self, value): self.bits.cmd_stream_version = value
    def get_cmd_stream_version(self): value = self.bits.cmd_stream_version; return value
    def set_shram_size(self, value): self.bits.shram_size = value
    def get_shram_size(self): value = self.bits.shram_size; return value
    def set_product(self, value): self.bits.product = value
    def get_product(self): value = self.bits.product; return value


class lock_r(Union):
    class _bitfield(Structure):
        _fields_ = [
            ("lock", c_uint32, 32),
        ]
    _fields_ = [("bits", _bitfield),
                ("word", c_uint32)]
    def set_lock(self, value): self.bits.lock = value
    def get_lock(self): value = self.bits.lock; return value


class regioncfg_r(Union):
    class _bitfield(Structure):
        _fields_ = [
            ("region0", c_uint32, 2),
            ("region1", c_uint32, 2),
            ("region2", c_uint32, 2),
            ("region3", c_uint32, 2),
            ("region4", c_uint32, 2),
            ("region5", c_uint32, 2),
            ("region6", c_uint32, 2),
            ("region7", c_uint32, 2),
            ("reserved0", c_uint32, 16),
        ]
    _fields_ = [("bits", _bitfield),
                ("word", c_uint32)]
    def set_region0(self, value): self.bits.region0 = value
    def get_region0(self): value = self.bits.region0; return value
    def set_region1(self, value): self.bits.region1 = value
    def get_region1(self): value = self.bits.region1; return value
    def set_region2(self, value): self.bits.region2 = value
    def get_region2(self): value = self.bits.region2; return value
    def set_region3(self, value): self.bits.region3 = value
    def get_region3(self): value = self.bits.region3; return value
    def set_region4(self, value): self.bits.region4 = value
    def get_region4(self): value = self.bits.region4; return value
    def set_region5(self, value): self.bits.region5 = value
    def get_region5(self): value = self.bits.region5; return value
    def set_region6(self, value): self.bits.region6 = value
    def get_region6(self): value = self.bits.region6; return value
    def set_region7(self, value): self.bits.region7 = value
    def get_region7(self): value = self.bits.region7; return value


class axi_limit0_r(Union):
    class _bitfield(Structure):
        _fields_ = [
            ("max_beats", c_uint32, 2),
            ("reserved0", c_uint32, 2),
            ("memtype", c_uint32, 4),
            ("reserved1", c_uint32, 8),
            ("max_outstanding_read_m1", c_uint32, 8),
            ("max_outstanding_write_m1", c_uint32, 8),
        ]
    _fields_ = [("bits", _bitfield),
                ("word", c_uint32)]
    def set_max_beats(self, value): self.bits.max_beats = value
    def get_max_beats(self): value = self.bits.max_beats; return value
    def set_memtype(self, value): self.bits.memtype = value
    def get_memtype(self): value = self.bits.memtype; return value
    def set_max_outstanding_read_m1(self, value): self.bits.max_outstanding_read_m1 = value
    def get_max_outstanding_read_m1(self): value = self.bits.max_outstanding_read_m1; return value
    def set_max_outstanding_write_m1(self, value): self.bits.max_outstanding_write_m1 = value
    def get_max_outstanding_write_m1(self): value = self.bits.max_outstanding_write_m1; return value


class axi_limit1_r(Union):
    class _bitfield(Structure):
        _fields_ = [
            ("max_beats", c_uint32, 2),
            ("reserved0", c_uint32, 2),
            ("memtype", c_uint32, 4),
            ("reserved1", c_uint32, 8),
            ("max_outstanding_read_m1", c_uint32, 8),
            ("max_outstanding_write_m1", c_uint32, 8),
        ]
    _fields_ = [("bits", _bitfield),
                ("word", c_uint32)]
    def set_max_beats(self, value): self.bits.max_beats = value
    def get_max_beats(self): value = self.bits.max_beats; return value
    def set_memtype(self, value): self.bits.memtype = value
    def get_memtype(self): value = self.bits.memtype; return value
    def set_max_outstanding_read_m1(self, value): self.bits.max_outstanding_read_m1 = value
    def get_max_outstanding_read_m1(self): value = self.bits.max_outstanding_read_m1; return value
    def set_max_outstanding_write_m1(self, value): self.bits.max_outstanding_write_m1 = value
    def get_max_outstanding_write_m1(self): value = self.bits.max_outstanding_write_m1; return value


class axi_limit2_r(Union):
    class _bitfield(Structure):
        _fields_ = [
            ("max_beats", c_uint32, 2),
            ("reserved0", c_uint32, 2),
            ("memtype", c_uint32, 4),
            ("reserved1", c_uint32, 8),
            ("max_outstanding_read_m1", c_uint32, 8),
            ("max_outstanding_write_m1", c_uint32, 8),
        ]
    _fields_ = [("bits", _bitfield),
                ("word", c_uint32)]
    def set_max_beats(self, value): self.bits.max_beats = value
    def get_max_beats(self): value = self.bits.max_beats; return value
    def set_memtype(self, value): self.bits.memtype = value
    def get_memtype(self): value = self.bits.memtype; return value
    def set_max_outstanding_read_m1(self, value): self.bits.max_outstanding_read_m1 = value
    def get_max_outstanding_read_m1(self): value = self.bits.max_outstanding_read_m1; return value
    def set_max_outstanding_write_m1(self, value): self.bits.max_outstanding_write_m1 = value
    def get_max_outstanding_write_m1(self): value = self.bits.max_outstanding_write_m1; return value


class axi_limit3_r(Union):
    class _bitfield(Structure):
        _fields_ = [
            ("max_beats", c_uint32, 2),
            ("reserved0", c_uint32, 2),
            ("memtype", c_uint32, 4),
            ("reserved1", c_uint32, 8),
            ("max_outstanding_read_m1", c_uint32, 8),
            ("max_outstanding_write_m1", c_uint32, 8),
        ]
    _fields_ = [("bits", _bitfield),
                ("word", c_uint32)]
    def set_max_beats(self, value): self.bits.max_beats = value
    def get_max_beats(self): value = self.bits.max_beats; return value
    def set_memtype(self, value): self.bits.memtype = value
    def get_memtype(self): value = self.bits.memtype; return value
    def set_max_outstanding_read_m1(self, value): self.bits.max_outstanding_read_m1 = value
    def get_max_outstanding_read_m1(self): value = self.bits.max_outstanding_read_m1; return value
    def set_max_outstanding_write_m1(self, value): self.bits.max_outstanding_write_m1 = value
    def get_max_outstanding_write_m1(self): value = self.bits.max_outstanding_write_m1; return value


class basep0_r(Union):
    class _bitfield(Structure):
        _fields_ = [
            ("addr_word", c_uint32, 32),
        ]
    _fields_ = [("bits", _bitfield),
                ("word", c_uint32)]
    def set_addr_word(self, value): self.bits.addr_word = value
    def get_addr_word(self): value = self.bits.addr_word; return value


class basep1_r(Union):
    class _bitfield(Structure):
        _fields_ = [
            ("addr_word", c_uint32, 32),
        ]
    _fields_ = [("bits", _bitfield),
                ("word", c_uint32)]
    def set_addr_word(self, value): self.bits.addr_word = value
    def get_addr_word(self): value = self.bits.addr_word; return value


class basep2_r(Union):
    class _bitfield(Structure):
        _fields_ = [
            ("addr_word", c_uint32, 32),
        ]
    _fields_ = [("bits", _bitfield),
                ("word", c_uint32)]
    def set_addr_word(self, value): self.bits.addr_word = value
    def get_addr_word(self): value = self.bits.addr_word; return value


class basep3_r(Union):
    class _bitfield(Structure):
        _fields_ = [
            ("addr_word", c_uint32, 32),
        ]
    _fields_ = [("bits", _bitfield),
                ("word", c_uint32)]
    def set_addr_word(self, value): self.bits.addr_word = value
    def get_addr_word(self): value = self.bits.addr_word; return value


class basep4_r(Union):
    class _bitfield(Structure):
        _fields_ = [
            ("addr_word", c_uint32, 32),
        ]
    _fields_ = [("bits", _bitfield),
                ("word", c_uint32)]
    def set_addr_word(self, value): self.bits.addr_word = value
    def get_addr_word(self): value = self.bits.addr_word; return value


class basep5_r(Union):
    class _bitfield(Structure):
        _fields_ = [
            ("addr_word", c_uint32, 32),
        ]
    _fields_ = [("bits", _bitfield),
                ("word", c_uint32)]
    def set_addr_word(self, value): self.bits.addr_word = value
    def get_addr_word(self): value = self.bits.addr_word; return value


class basep6_r(Union):
    class _bitfield(Structure):
        _fields_ = [
            ("addr_word", c_uint32, 32),
        ]
    _fields_ = [("bits", _bitfield),
                ("word", c_uint32)]
    def set_addr_word(self, value): self.bits.addr_word = value
    def get_addr_word(self): value = self.bits.addr_word; return value


class basep7_r(Union):
    class _bitfield(Structure):
        _fields_ = [
            ("addr_word", c_uint32, 32),
        ]
    _fields_ = [("bits", _bitfield),
                ("word", c_uint32)]
    def set_addr_word(self, value): self.bits.addr_word = value
    def get_addr_word(self): value = self.bits.addr_word; return value


class basep8_r(Union):
    class _bitfield(Structure):
        _fields_ = [
            ("addr_word", c_uint32, 32),
        ]
    _fields_ = [("bits", _bitfield),
                ("word", c_uint32)]
    def set_addr_word(self, value): self.bits.addr_word = value
    def get_addr_word(self): value = self.bits.addr_word; return value


class basep9_r(Union):
    class _bitfield(Structure):
        _fields_ = [
            ("addr_word", c_uint32, 32),
        ]
    _fields_ = [("bits", _bitfield),
                ("word", c_uint32)]
    def set_addr_word(self, value): self.bits.addr_word = value
    def get_addr_word(self): value = self.bits.addr_word; return value


class basep10_r(Union):
    class _bitfield(Structure):
        _fields_ = [
            ("addr_word", c_uint32, 32),
        ]
    _fields_ = [("bits", _bitfield),
                ("word", c_uint32)]
    def set_addr_word(self, value): self.bits.addr_word = value
    def get_addr_word(self): value = self.bits.addr_word; return value


class basep11_r(Union):
    class _bitfield(Structure):
        _fields_ = [
            ("addr_word", c_uint32, 32),
        ]
    _fields_ = [("bits", _bitfield),
                ("word", c_uint32)]
    def set_addr_word(self, value): self.bits.addr_word = value
    def get_addr_word(self): value = self.bits.addr_word; return value


class basep12_r(Union):
    class _bitfield(Structure):
        _fields_ = [
            ("addr_word", c_uint32, 32),
        ]
    _fields_ = [("bits", _bitfield),
                ("word", c_uint32)]
    def set_addr_word(self, value): self.bits.addr_word = value
    def get_addr_word(self): value = self.bits.addr_word; return value


class basep13_r(Union):
    class _bitfield(Structure):
        _fields_ = [
            ("addr_word", c_uint32, 32),
        ]
    _fields_ = [("bits", _bitfield),
                ("word", c_uint32)]
    def set_addr_word(self, value): self.bits.addr_word = value
    def get_addr_word(self): value = self.bits.addr_word; return value


class basep14_r(Union):
    class _bitfield(Structure):
        _fields_ = [
            ("addr_word", c_uint32, 32),
        ]
    _fields_ = [("bits", _bitfield),
                ("word", c_uint32)]
    def set_addr_word(self, value): self.bits.addr_word = value
    def get_addr_word(self): value = self.bits.addr_word; return value


class basep15_r(Union):
    class _bitfield(Structure):
        _fields_ = [
            ("addr_word", c_uint32, 32),
        ]
    _fields_ = [("bits", _bitfield),
                ("word", c_uint32)]
    def set_addr_word(self, value): self.bits.addr_word = value
    def get_addr_word(self): value = self.bits.addr_word; return value


class wd_status_r(Union):
    class _bitfield(Structure):
        _fields_ = [
            ("core_slice_state", c_uint32, 2),
            ("core_idle", c_uint32, 1),
            ("ctrl_state", c_uint32, 2),
            ("ctrl_idle", c_uint32, 1),
            ("write_buf_index0", c_uint32, 3),
            ("write_buf_valid0", c_uint32, 1),
            ("write_buf_idle0", c_uint32, 1),
            ("write_buf_index1", c_uint32, 3),
            ("write_buf_valid1", c_uint32, 1),
            ("write_buf_idle1", c_uint32, 1),
            ("events", c_uint32, 12),
            ("reserved0", c_uint32, 4),
        ]
    _fields_ = [("bits", _bitfield),
                ("word", c_uint32)]
    def set_core_slice_state(self, value): self.bits.core_slice_state = value
    def get_core_slice_state(self): value = self.bits.core_slice_state; return value
    def set_core_idle(self, value): self.bits.core_idle = value
    def get_core_idle(self): value = self.bits.core_idle; return value
    def set_ctrl_state(self, value): self.bits.ctrl_state = value
    def get_ctrl_state(self): value = self.bits.ctrl_state; return value
    def set_ctrl_idle(self, value): self.bits.ctrl_idle = value
    def get_ctrl_idle(self): value = self.bits.ctrl_idle; return value
    def set_write_buf_index0(self, value): self.bits.write_buf_index0 = value
    def get_write_buf_index0(self): value = self.bits.write_buf_index0; return value
    def set_write_buf_valid0(self, value): self.bits.write_buf_valid0 = value
    def get_write_buf_valid0(self): value = self.bits.write_buf_valid0; return value
    def set_write_buf_idle0(self, value): self.bits.write_buf_idle0 = value
    def get_write_buf_idle0(self): value = self.bits.write_buf_idle0; return value
    def set_write_buf_index1(self, value): self.bits.write_buf_index1 = value
    def get_write_buf_index1(self): value = self.bits.write_buf_index1; return value
    def set_write_buf_valid1(self, value): self.bits.write_buf_valid1 = value
    def get_write_buf_valid1(self): value = self.bits.write_buf_valid1; return value
    def set_write_buf_idle1(self, value): self.bits.write_buf_idle1 = value
    def get_write_buf_idle1(self): value = self.bits.write_buf_idle1; return value
    def set_events(self, value): self.bits.events = value
    def get_events(self): value = self.bits.events; return value


class mac_status_r(Union):
    class _bitfield(Structure):
        _fields_ = [
            ("block_cfg_valid", c_uint32, 1),
            ("trav_en", c_uint32, 1),
            ("wait_for_ib", c_uint32, 1),
            ("wait_for_acc_buf", c_uint32, 1),
            ("wait_for_weights", c_uint32, 1),
            ("stall_stripe", c_uint32, 1),
            ("dw_sel", c_uint32, 1),
            ("wait_for_dw0_ready", c_uint32, 1),
            ("wait_for_dw1_ready", c_uint32, 1),
            ("acc_buf_sel_ai", c_uint32, 1),
            ("wait_for_acc0_ready", c_uint32, 1),
            ("wait_for_acc1_ready", c_uint32, 1),
            ("acc_buf_sel_aa", c_uint32, 1),
            ("acc0_valid", c_uint32, 1),
            ("acc1_valid", c_uint32, 1),
            ("reserved0", c_uint32, 1),
            ("events", c_uint32, 11),
            ("reserved1", c_uint32, 5),
        ]
    _fields_ = [("bits", _bitfield),
                ("word", c_uint32)]
    def set_block_cfg_valid(self, value): self.bits.block_cfg_valid = value
    def get_block_cfg_valid(self): value = self.bits.block_cfg_valid; return value
    def set_trav_en(self, value): self.bits.trav_en = value
    def get_trav_en(self): value = self.bits.trav_en; return value
    def set_wait_for_ib(self, value): self.bits.wait_for_ib = value
    def get_wait_for_ib(self): value = self.bits.wait_for_ib; return value
    def set_wait_for_acc_buf(self, value): self.bits.wait_for_acc_buf = value
    def get_wait_for_acc_buf(self): value = self.bits.wait_for_acc_buf; return value
    def set_wait_for_weights(self, value): self.bits.wait_for_weights = value
    def get_wait_for_weights(self): value = self.bits.wait_for_weights; return value
    def set_stall_stripe(self, value): self.bits.stall_stripe = value
    def get_stall_stripe(self): value = self.bits.stall_stripe; return value
    def set_dw_sel(self, value): self.bits.dw_sel = value
    def get_dw_sel(self): value = self.bits.dw_sel; return value
    def set_wait_for_dw0_ready(self, value): self.bits.wait_for_dw0_ready = value
    def get_wait_for_dw0_ready(self): value = self.bits.wait_for_dw0_ready; return value
    def set_wait_for_dw1_ready(self, value): self.bits.wait_for_dw1_ready = value
    def get_wait_for_dw1_ready(self): value = self.bits.wait_for_dw1_ready; return value
    def set_acc_buf_sel_ai(self, value): self.bits.acc_buf_sel_ai = value
    def get_acc_buf_sel_ai(self): value = self.bits.acc_buf_sel_ai; return value
    def set_wait_for_acc0_ready(self, value): self.bits.wait_for_acc0_ready = value
    def get_wait_for_acc0_ready(self): value = self.bits.wait_for_acc0_ready; return value
    def set_wait_for_acc1_ready(self, value): self.bits.wait_for_acc1_ready = value
    def get_wait_for_acc1_ready(self): value = self.bits.wait_for_acc1_ready; return value
    def set_acc_buf_sel_aa(self, value): self.bits.acc_buf_sel_aa = value
    def get_acc_buf_sel_aa(self): value = self.bits.acc_buf_sel_aa; return value
    def set_acc0_valid(self, value): self.bits.acc0_valid = value
    def get_acc0_valid(self): value = self.bits.acc0_valid; return value
    def set_acc1_valid(self, value): self.bits.acc1_valid = value
    def get_acc1_valid(self): value = self.bits.acc1_valid; return value
    def set_events(self, value): self.bits.events = value
    def get_events(self): value = self.bits.events; return value


class ao_status_r(Union):
    class _bitfield(Structure):
        _fields_ = [
            ("cmd_sbw_valid", c_uint32, 1),
            ("cmd_act_valid", c_uint32, 1),
            ("cmd_ctl_valid", c_uint32, 1),
            ("cmd_scl_valid", c_uint32, 1),
            ("cmd_sbr_valid", c_uint32, 1),
            ("cmd_ofm_valid", c_uint32, 1),
            ("blk_cmd_ready", c_uint32, 1),
            ("blk_cmd_valid", c_uint32, 1),
            ("reserved0", c_uint32, 8),
            ("events", c_uint32, 8),
            ("reserved1", c_uint32, 8),
        ]
    _fields_ = [("bits", _bitfield),
                ("word", c_uint32)]
    def set_cmd_sbw_valid(self, value): self.bits.cmd_sbw_valid = value
    def get_cmd_sbw_valid(self): value = self.bits.cmd_sbw_valid; return value
    def set_cmd_act_valid(self, value): self.bits.cmd_act_valid = value
    def get_cmd_act_valid(self): value = self.bits.cmd_act_valid; return value
    def set_cmd_ctl_valid(self, value): self.bits.cmd_ctl_valid = value
    def get_cmd_ctl_valid(self): value = self.bits.cmd_ctl_valid; return value
    def set_cmd_scl_valid(self, value): self.bits.cmd_scl_valid = value
    def get_cmd_scl_valid(self): value = self.bits.cmd_scl_valid; return value
    def set_cmd_sbr_valid(self, value): self.bits.cmd_sbr_valid = value
    def get_cmd_sbr_valid(self): value = self.bits.cmd_sbr_valid; return value
    def set_cmd_ofm_valid(self, value): self.bits.cmd_ofm_valid = value
    def get_cmd_ofm_valid(self): value = self.bits.cmd_ofm_valid; return value
    def set_blk_cmd_ready(self, value): self.bits.blk_cmd_ready = value
    def get_blk_cmd_ready(self): value = self.bits.blk_cmd_ready; return value
    def set_blk_cmd_valid(self, value): self.bits.blk_cmd_valid = value
    def get_blk_cmd_valid(self): value = self.bits.blk_cmd_valid; return value
    def set_events(self, value): self.bits.events = value
    def get_events(self): value = self.bits.events; return value


class dma_status0_r(Union):
    class _bitfield(Structure):
        _fields_ = [
            ("cmd_idle", c_uint32, 1),
            ("ifm_idle", c_uint32, 1),
            ("wgt_idle_c0", c_uint32, 1),
            ("bas_idle_c0", c_uint32, 1),
            ("m2m_idle", c_uint32, 1),
            ("ofm_idle", c_uint32, 1),
            ("halt_req", c_uint32, 1),
            ("halt_ack", c_uint32, 1),
            ("pause_req", c_uint32, 1),
            ("pause_ack", c_uint32, 1),
            ("ib0_ai_valid_c0", c_uint32, 1),
            ("ib0_ai_ready_c0", c_uint32, 1),
            ("ib1_ai_valid_c0", c_uint32, 1),
            ("ib1_ai_ready_c0", c_uint32, 1),
            ("ib0_ao_valid_c0", c_uint32, 1),
            ("ib0_ao_ready_c0", c_uint32, 1),
            ("ib1_ao_valid_c0", c_uint32, 1),
            ("ib1_ao_ready_c0", c_uint32, 1),
            ("ob0_valid_c0", c_uint32, 1),
            ("ob0_ready_c0", c_uint32, 1),
            ("ob1_valid_c0", c_uint32, 1),
            ("ob1_ready_c0", c_uint32, 1),
            ("cmd_valid", c_uint32, 1),
            ("cmd_ready", c_uint32, 1),
            ("wd_bitstream_valid_c0", c_uint32, 1),
            ("wd_bitstream_ready_c0", c_uint32, 1),
            ("bs_bitstream_valid_c0", c_uint32, 1),
            ("bs_bitstream_ready_c0", c_uint32, 1),
            ("axi0_ar_stalled", c_uint32, 1),
            ("axi0_rd_limit_stall", c_uint32, 1),
            ("axi0_aw_stalled", c_uint32, 1),
            ("axi0_w_stalled", c_uint32, 1),
        ]
    _fields_ = [("bits", _bitfield),
                ("word", c_uint32)]
    def set_cmd_idle(self, value): self.bits.cmd_idle = value
    def get_cmd_idle(self): value = self.bits.cmd_idle; return value
    def set_ifm_idle(self, value): self.bits.ifm_idle = value
    def get_ifm_idle(self): value = self.bits.ifm_idle; return value
    def set_wgt_idle_c0(self, value): self.bits.wgt_idle_c0 = value
    def get_wgt_idle_c0(self): value = self.bits.wgt_idle_c0; return value
    def set_bas_idle_c0(self, value): self.bits.bas_idle_c0 = value
    def get_bas_idle_c0(self): value = self.bits.bas_idle_c0; return value
    def set_m2m_idle(self, value): self.bits.m2m_idle = value
    def get_m2m_idle(self): value = self.bits.m2m_idle; return value
    def set_ofm_idle(self, value): self.bits.ofm_idle = value
    def get_ofm_idle(self): value = self.bits.ofm_idle; return value
    def set_halt_req(self, value): self.bits.halt_req = value
    def get_halt_req(self): value = self.bits.halt_req; return value
    def set_halt_ack(self, value): self.bits.halt_ack = value
    def get_halt_ack(self): value = self.bits.halt_ack; return value
    def set_pause_req(self, value): self.bits.pause_req = value
    def get_pause_req(self): value = self.bits.pause_req; return value
    def set_pause_ack(self, value): self.bits.pause_ack = value
    def get_pause_ack(self): value = self.bits.pause_ack; return value
    def set_ib0_ai_valid_c0(self, value): self.bits.ib0_ai_valid_c0 = value
    def get_ib0_ai_valid_c0(self): value = self.bits.ib0_ai_valid_c0; return value
    def set_ib0_ai_ready_c0(self, value): self.bits.ib0_ai_ready_c0 = value
    def get_ib0_ai_ready_c0(self): value = self.bits.ib0_ai_ready_c0; return value
    def set_ib1_ai_valid_c0(self, value): self.bits.ib1_ai_valid_c0 = value
    def get_ib1_ai_valid_c0(self): value = self.bits.ib1_ai_valid_c0; return value
    def set_ib1_ai_ready_c0(self, value): self.bits.ib1_ai_ready_c0 = value
    def get_ib1_ai_ready_c0(self): value = self.bits.ib1_ai_ready_c0; return value
    def set_ib0_ao_valid_c0(self, value): self.bits.ib0_ao_valid_c0 = value
    def get_ib0_ao_valid_c0(self): value = self.bits.ib0_ao_valid_c0; return value
    def set_ib0_ao_ready_c0(self, value): self.bits.ib0_ao_ready_c0 = value
    def get_ib0_ao_ready_c0(self): value = self.bits.ib0_ao_ready_c0; return value
    def set_ib1_ao_valid_c0(self, value): self.bits.ib1_ao_valid_c0 = value
    def get_ib1_ao_valid_c0(self): value = self.bits.ib1_ao_valid_c0; return value
    def set_ib1_ao_ready_c0(self, value): self.bits.ib1_ao_ready_c0 = value
    def get_ib1_ao_ready_c0(self): value = self.bits.ib1_ao_ready_c0; return value
    def set_ob0_valid_c0(self, value): self.bits.ob0_valid_c0 = value
    def get_ob0_valid_c0(self): value = self.bits.ob0_valid_c0; return value
    def set_ob0_ready_c0(self, value): self.bits.ob0_ready_c0 = value
    def get_ob0_ready_c0(self): value = self.bits.ob0_ready_c0; return value
    def set_ob1_valid_c0(self, value): self.bits.ob1_valid_c0 = value
    def get_ob1_valid_c0(self): value = self.bits.ob1_valid_c0; return value
    def set_ob1_ready_c0(self, value): self.bits.ob1_ready_c0 = value
    def get_ob1_ready_c0(self): value = self.bits.ob1_ready_c0; return value
    def set_cmd_valid(self, value): self.bits.cmd_valid = value
    def get_cmd_valid(self): value = self.bits.cmd_valid; return value
    def set_cmd_ready(self, value): self.bits.cmd_ready = value
    def get_cmd_ready(self): value = self.bits.cmd_ready; return value
    def set_wd_bitstream_valid_c0(self, value): self.bits.wd_bitstream_valid_c0 = value
    def get_wd_bitstream_valid_c0(self): value = self.bits.wd_bitstream_valid_c0; return value
    def set_wd_bitstream_ready_c0(self, value): self.bits.wd_bitstream_ready_c0 = value
    def get_wd_bitstream_ready_c0(self): value = self.bits.wd_bitstream_ready_c0; return value
    def set_bs_bitstream_valid_c0(self, value): self.bits.bs_bitstream_valid_c0 = value
    def get_bs_bitstream_valid_c0(self): value = self.bits.bs_bitstream_valid_c0; return value
    def set_bs_bitstream_ready_c0(self, value): self.bits.bs_bitstream_ready_c0 = value
    def get_bs_bitstream_ready_c0(self): value = self.bits.bs_bitstream_ready_c0; return value
    def set_axi0_ar_stalled(self, value): self.bits.axi0_ar_stalled = value
    def get_axi0_ar_stalled(self): value = self.bits.axi0_ar_stalled; return value
    def set_axi0_rd_limit_stall(self, value): self.bits.axi0_rd_limit_stall = value
    def get_axi0_rd_limit_stall(self): value = self.bits.axi0_rd_limit_stall; return value
    def set_axi0_aw_stalled(self, value): self.bits.axi0_aw_stalled = value
    def get_axi0_aw_stalled(self): value = self.bits.axi0_aw_stalled; return value
    def set_axi0_w_stalled(self, value): self.bits.axi0_w_stalled = value
    def get_axi0_w_stalled(self): value = self.bits.axi0_w_stalled; return value


class dma_status1_r(Union):
    class _bitfield(Structure):
        _fields_ = [
            ("axi0_wr_limit_stall", c_uint32, 1),
            ("axi1_ar_stalled", c_uint32, 1),
            ("axi1_rd_limit_stall", c_uint32, 1),
            ("axi1_wr_stalled", c_uint32, 1),
            ("axi1_w_stalled", c_uint32, 1),
            ("axi1_wr_limit_stall", c_uint32, 1),
            ("wgt_idle_c1", c_uint32, 1),
            ("bas_idle_c1", c_uint32, 1),
            ("ib0_ai_valid_c1", c_uint32, 1),
            ("ib0_ai_ready_c1", c_uint32, 1),
            ("ib1_ai_valid_c1", c_uint32, 1),
            ("ib1_ai_ready_c1", c_uint32, 1),
            ("ib0_ao_valid_c1", c_uint32, 1),
            ("ib0_ao_ready_c1", c_uint32, 1),
            ("ib1_ao_valid_c1", c_uint32, 1),
            ("ib1_ao_ready_c1", c_uint32, 1),
            ("ob0_valid_c1", c_uint32, 1),
            ("ob0_ready_c1", c_uint32, 1),
            ("ob1_valid_c1", c_uint32, 1),
            ("ob1_ready_c1", c_uint32, 1),
            ("wd_bitstream_valid_c1", c_uint32, 1),
            ("wd_bitstream_ready_c1", c_uint32, 1),
            ("bs_bitstream_valid_c1", c_uint32, 1),
            ("bs_bitstream_ready_c1", c_uint32, 1),
            ("reserved0", c_uint32, 8),
        ]
    _fields_ = [("bits", _bitfield),
                ("word", c_uint32)]
    def set_axi0_wr_limit_stall(self, value): self.bits.axi0_wr_limit_stall = value
    def get_axi0_wr_limit_stall(self): value = self.bits.axi0_wr_limit_stall; return value
    def set_axi1_ar_stalled(self, value): self.bits.axi1_ar_stalled = value
    def get_axi1_ar_stalled(self): value = self.bits.axi1_ar_stalled; return value
    def set_axi1_rd_limit_stall(self, value): self.bits.axi1_rd_limit_stall = value
    def get_axi1_rd_limit_stall(self): value = self.bits.axi1_rd_limit_stall; return value
    def set_axi1_wr_stalled(self, value): self.bits.axi1_wr_stalled = value
    def get_axi1_wr_stalled(self): value = self.bits.axi1_wr_stalled; return value
    def set_axi1_w_stalled(self, value): self.bits.axi1_w_stalled = value
    def get_axi1_w_stalled(self): value = self.bits.axi1_w_stalled; return value
    def set_axi1_wr_limit_stall(self, value): self.bits.axi1_wr_limit_stall = value
    def get_axi1_wr_limit_stall(self): value = self.bits.axi1_wr_limit_stall; return value
    def set_wgt_idle_c1(self, value): self.bits.wgt_idle_c1 = value
    def get_wgt_idle_c1(self): value = self.bits.wgt_idle_c1; return value
    def set_bas_idle_c1(self, value): self.bits.bas_idle_c1 = value
    def get_bas_idle_c1(self): value = self.bits.bas_idle_c1; return value
    def set_ib0_ai_valid_c1(self, value): self.bits.ib0_ai_valid_c1 = value
    def get_ib0_ai_valid_c1(self): value = self.bits.ib0_ai_valid_c1; return value
    def set_ib0_ai_ready_c1(self, value): self.bits.ib0_ai_ready_c1 = value
    def get_ib0_ai_ready_c1(self): value = self.bits.ib0_ai_ready_c1; return value
    def set_ib1_ai_valid_c1(self, value): self.bits.ib1_ai_valid_c1 = value
    def get_ib1_ai_valid_c1(self): value = self.bits.ib1_ai_valid_c1; return value
    def set_ib1_ai_ready_c1(self, value): self.bits.ib1_ai_ready_c1 = value
    def get_ib1_ai_ready_c1(self): value = self.bits.ib1_ai_ready_c1; return value
    def set_ib0_ao_valid_c1(self, value): self.bits.ib0_ao_valid_c1 = value
    def get_ib0_ao_valid_c1(self): value = self.bits.ib0_ao_valid_c1; return value
    def set_ib0_ao_ready_c1(self, value): self.bits.ib0_ao_ready_c1 = value
    def get_ib0_ao_ready_c1(self): value = self.bits.ib0_ao_ready_c1; return value
    def set_ib1_ao_valid_c1(self, value): self.bits.ib1_ao_valid_c1 = value
    def get_ib1_ao_valid_c1(self): value = self.bits.ib1_ao_valid_c1; return value
    def set_ib1_ao_ready_c1(self, value): self.bits.ib1_ao_ready_c1 = value
    def get_ib1_ao_ready_c1(self): value = self.bits.ib1_ao_ready_c1; return value
    def set_ob0_valid_c1(self, value): self.bits.ob0_valid_c1 = value
    def get_ob0_valid_c1(self): value = self.bits.ob0_valid_c1; return value
    def set_ob0_ready_c1(self, value): self.bits.ob0_ready_c1 = value
    def get_ob0_ready_c1(self): value = self.bits.ob0_ready_c1; return value
    def set_ob1_valid_c1(self, value): self.bits.ob1_valid_c1 = value
    def get_ob1_valid_c1(self): value = self.bits.ob1_valid_c1; return value
    def set_ob1_ready_c1(self, value): self.bits.ob1_ready_c1 = value
    def get_ob1_ready_c1(self): value = self.bits.ob1_ready_c1; return value
    def set_wd_bitstream_valid_c1(self, value): self.bits.wd_bitstream_valid_c1 = value
    def get_wd_bitstream_valid_c1(self): value = self.bits.wd_bitstream_valid_c1; return value
    def set_wd_bitstream_ready_c1(self, value): self.bits.wd_bitstream_ready_c1 = value
    def get_wd_bitstream_ready_c1(self): value = self.bits.wd_bitstream_ready_c1; return value
    def set_bs_bitstream_valid_c1(self, value): self.bits.bs_bitstream_valid_c1 = value
    def get_bs_bitstream_valid_c1(self): value = self.bits.bs_bitstream_valid_c1; return value
    def set_bs_bitstream_ready_c1(self, value): self.bits.bs_bitstream_ready_c1 = value
    def get_bs_bitstream_ready_c1(self): value = self.bits.bs_bitstream_ready_c1; return value


class clkforce_r(Union):
    class _bitfield(Structure):
        _fields_ = [
            ("top_level_clk", c_uint32, 1),
            ("cc_clk", c_uint32, 1),
            ("dma_clk", c_uint32, 1),
            ("mac_clk", c_uint32, 1),
            ("ao_clk", c_uint32, 1),
            ("wd_clk", c_uint32, 1),
            ("reserved0", c_uint32, 26),
        ]
    _fields_ = [("bits", _bitfield),
                ("word", c_uint32)]
    def set_top_level_clk(self, value): self.bits.top_level_clk = value
    def get_top_level_clk(self): value = self.bits.top_level_clk; return value
    def set_cc_clk(self, value): self.bits.cc_clk = value
    def get_cc_clk(self): value = self.bits.cc_clk; return value
    def set_dma_clk(self, value): self.bits.dma_clk = value
    def get_dma_clk(self): value = self.bits.dma_clk; return value
    def set_mac_clk(self, value): self.bits.mac_clk = value
    def get_mac_clk(self): value = self.bits.mac_clk; return value
    def set_ao_clk(self, value): self.bits.ao_clk = value
    def get_ao_clk(self): value = self.bits.ao_clk; return value
    def set_wd_clk(self, value): self.bits.wd_clk = value
    def get_wd_clk(self): value = self.bits.wd_clk; return value


class pid4_r(Union):
    class _bitfield(Structure):
        _fields_ = [
            ("pid4", c_uint32, 32),
        ]
    _fields_ = [("bits", _bitfield),
                ("word", c_uint32)]
    def set_pid4(self, value): self.bits.pid4 = value
    def get_pid4(self): value = self.bits.pid4; return value


class pid5_r(Union):
    class _bitfield(Structure):
        _fields_ = [
            ("pid5", c_uint32, 32),
        ]
    _fields_ = [("bits", _bitfield),
                ("word", c_uint32)]
    def set_pid5(self, value): self.bits.pid5 = value
    def get_pid5(self): value = self.bits.pid5; return value


class pid6_r(Union):
    class _bitfield(Structure):
        _fields_ = [
            ("pid6", c_uint32, 32),
        ]
    _fields_ = [("bits", _bitfield),
                ("word", c_uint32)]
    def set_pid6(self, value): self.bits.pid6 = value
    def get_pid6(self): value = self.bits.pid6; return value


class pid7_r(Union):
    class _bitfield(Structure):
        _fields_ = [
            ("pid7", c_uint32, 32),
        ]
    _fields_ = [("bits", _bitfield),
                ("word", c_uint32)]
    def set_pid7(self, value): self.bits.pid7 = value
    def get_pid7(self): value = self.bits.pid7; return value


class pid0_r(Union):
    class _bitfield(Structure):
        _fields_ = [
            ("pid0", c_uint32, 32),
        ]
    _fields_ = [("bits", _bitfield),
                ("word", c_uint32)]
    def set_pid0(self, value): self.bits.pid0 = value
    def get_pid0(self): value = self.bits.pid0; return value


class pid1_r(Union):
    class _bitfield(Structure):
        _fields_ = [
            ("pid1", c_uint32, 32),
        ]
    _fields_ = [("bits", _bitfield),
                ("word", c_uint32)]
    def set_pid1(self, value): self.bits.pid1 = value
    def get_pid1(self): value = self.bits.pid1; return value


class pid2_r(Union):
    class _bitfield(Structure):
        _fields_ = [
            ("pid2", c_uint32, 32),
        ]
    _fields_ = [("bits", _bitfield),
                ("word", c_uint32)]
    def set_pid2(self, value): self.bits.pid2 = value
    def get_pid2(self): value = self.bits.pid2; return value


class pid3_r(Union):
    class _bitfield(Structure):
        _fields_ = [
            ("pid3", c_uint32, 32),
        ]
    _fields_ = [("bits", _bitfield),
                ("word", c_uint32)]
    def set_pid3(self, value): self.bits.pid3 = value
    def get_pid3(self): value = self.bits.pid3; return value


class cid0_r(Union):
    class _bitfield(Structure):
        _fields_ = [
            ("cid0", c_uint32, 32),
        ]
    _fields_ = [("bits", _bitfield),
                ("word", c_uint32)]
    def set_cid0(self, value): self.bits.cid0 = value
    def get_cid0(self): value = self.bits.cid0; return value


class cid1_r(Union):
    class _bitfield(Structure):
        _fields_ = [
            ("cid1", c_uint32, 32),
        ]
    _fields_ = [("bits", _bitfield),
                ("word", c_uint32)]
    def set_cid1(self, value): self.bits.cid1 = value
    def get_cid1(self): value = self.bits.cid1; return value


class cid2_r(Union):
    class _bitfield(Structure):
        _fields_ = [
            ("cid2", c_uint32, 32),
        ]
    _fields_ = [("bits", _bitfield),
                ("word", c_uint32)]
    def set_cid2(self, value): self.bits.cid2 = value
    def get_cid2(self): value = self.bits.cid2; return value


class cid3_r(Union):
    class _bitfield(Structure):
        _fields_ = [
            ("cid3", c_uint32, 32),
        ]
    _fields_ = [("bits", _bitfield),
                ("word", c_uint32)]
    def set_cid3(self, value): self.bits.cid3 = value
    def get_cid3(self): value = self.bits.cid3; return value


class pmcr_r(Union):
    class _bitfield(Structure):
        _fields_ = [
            ("cnt_en", c_uint32, 1),
            ("event_cnt_rst", c_uint32, 1),
            ("cycle_cnt_rst", c_uint32, 1),
            ("mask_en", c_uint32, 1),
            ("reserved0", c_uint32, 7),
            ("num_event_cnt", c_uint32, 5),
            ("reserved1", c_uint32, 16),
        ]
    _fields_ = [("bits", _bitfield),
                ("word", c_uint32)]
    def set_cnt_en(self, value): self.bits.cnt_en = value
    def get_cnt_en(self): value = self.bits.cnt_en; return value
    def set_event_cnt_rst(self, value): self.bits.event_cnt_rst = value
    def get_event_cnt_rst(self): value = self.bits.event_cnt_rst; return value
    def set_cycle_cnt_rst(self, value): self.bits.cycle_cnt_rst = value
    def get_cycle_cnt_rst(self): value = self.bits.cycle_cnt_rst; return value
    def set_mask_en(self, value): self.bits.mask_en = value
    def get_mask_en(self): value = self.bits.mask_en; return value
    def set_num_event_cnt(self, value): self.bits.num_event_cnt = value
    def get_num_event_cnt(self): value = self.bits.num_event_cnt; return value


class pmcntenset_r(Union):
    class _bitfield(Structure):
        _fields_ = [
            ("event_cnt_0", c_uint32, 1),
            ("event_cnt_1", c_uint32, 1),
            ("event_cnt_2", c_uint32, 1),
            ("event_cnt_3", c_uint32, 1),
            ("reserved0", c_uint32, 27),
            ("cycle_cnt", c_uint32, 1),
        ]
    _fields_ = [("bits", _bitfield),
                ("word", c_uint32)]
    def set_event_cnt_0(self, value): self.bits.event_cnt_0 = value
    def get_event_cnt_0(self): value = self.bits.event_cnt_0; return value
    def set_event_cnt_1(self, value): self.bits.event_cnt_1 = value
    def get_event_cnt_1(self): value = self.bits.event_cnt_1; return value
    def set_event_cnt_2(self, value): self.bits.event_cnt_2 = value
    def get_event_cnt_2(self): value = self.bits.event_cnt_2; return value
    def set_event_cnt_3(self, value): self.bits.event_cnt_3 = value
    def get_event_cnt_3(self): value = self.bits.event_cnt_3; return value
    def set_cycle_cnt(self, value): self.bits.cycle_cnt = value
    def get_cycle_cnt(self): value = self.bits.cycle_cnt; return value


class pmcntenclr_r(Union):
    class _bitfield(Structure):
        _fields_ = [
            ("event_cnt_0", c_uint32, 1),
            ("event_cnt_1", c_uint32, 1),
            ("event_cnt_2", c_uint32, 1),
            ("event_cnt_3", c_uint32, 1),
            ("reserved0", c_uint32, 27),
            ("cycle_cnt", c_uint32, 1),
        ]
    _fields_ = [("bits", _bitfield),
                ("word", c_uint32)]
    def set_event_cnt_0(self, value): self.bits.event_cnt_0 = value
    def get_event_cnt_0(self): value = self.bits.event_cnt_0; return value
    def set_event_cnt_1(self, value): self.bits.event_cnt_1 = value
    def get_event_cnt_1(self): value = self.bits.event_cnt_1; return value
    def set_event_cnt_2(self, value): self.bits.event_cnt_2 = value
    def get_event_cnt_2(self): value = self.bits.event_cnt_2; return value
    def set_event_cnt_3(self, value): self.bits.event_cnt_3 = value
    def get_event_cnt_3(self): value = self.bits.event_cnt_3; return value
    def set_cycle_cnt(self, value): self.bits.cycle_cnt = value
    def get_cycle_cnt(self): value = self.bits.cycle_cnt; return value


class pmovsset_r(Union):
    class _bitfield(Structure):
        _fields_ = [
            ("event_cnt_0_ovf", c_uint32, 1),
            ("event_cnt_1_ovf", c_uint32, 1),
            ("event_cnt_2_ovf", c_uint32, 1),
            ("event_cnt_3_ovf", c_uint32, 1),
            ("reserved0", c_uint32, 27),
            ("cycle_cnt_ovf", c_uint32, 1),
        ]
    _fields_ = [("bits", _bitfield),
                ("word", c_uint32)]
    def set_event_cnt_0_ovf(self, value): self.bits.event_cnt_0_ovf = value
    def get_event_cnt_0_ovf(self): value = self.bits.event_cnt_0_ovf; return value
    def set_event_cnt_1_ovf(self, value): self.bits.event_cnt_1_ovf = value
    def get_event_cnt_1_ovf(self): value = self.bits.event_cnt_1_ovf; return value
    def set_event_cnt_2_ovf(self, value): self.bits.event_cnt_2_ovf = value
    def get_event_cnt_2_ovf(self): value = self.bits.event_cnt_2_ovf; return value
    def set_event_cnt_3_ovf(self, value): self.bits.event_cnt_3_ovf = value
    def get_event_cnt_3_ovf(self): value = self.bits.event_cnt_3_ovf; return value
    def set_cycle_cnt_ovf(self, value): self.bits.cycle_cnt_ovf = value
    def get_cycle_cnt_ovf(self): value = self.bits.cycle_cnt_ovf; return value


class pmovsclr_r(Union):
    class _bitfield(Structure):
        _fields_ = [
            ("event_cnt_0_ovf", c_uint32, 1),
            ("event_cnt_1_ovf", c_uint32, 1),
            ("event_cnt_2_ovf", c_uint32, 1),
            ("event_cnt_3_ovf", c_uint32, 1),
            ("reserved0", c_uint32, 27),
            ("cycle_cnt_ovf", c_uint32, 1),
        ]
    _fields_ = [("bits", _bitfield),
                ("word", c_uint32)]
    def set_event_cnt_0_ovf(self, value): self.bits.event_cnt_0_ovf = value
    def get_event_cnt_0_ovf(self): value = self.bits.event_cnt_0_ovf; return value
    def set_event_cnt_1_ovf(self, value): self.bits.event_cnt_1_ovf = value
    def get_event_cnt_1_ovf(self): value = self.bits.event_cnt_1_ovf; return value
    def set_event_cnt_2_ovf(self, value): self.bits.event_cnt_2_ovf = value
    def get_event_cnt_2_ovf(self): value = self.bits.event_cnt_2_ovf; return value
    def set_event_cnt_3_ovf(self, value): self.bits.event_cnt_3_ovf = value
    def get_event_cnt_3_ovf(self): value = self.bits.event_cnt_3_ovf; return value
    def set_cycle_cnt_ovf(self, value): self.bits.cycle_cnt_ovf = value
    def get_cycle_cnt_ovf(self): value = self.bits.cycle_cnt_ovf; return value


class pmintset_r(Union):
    class _bitfield(Structure):
        _fields_ = [
            ("event_cnt_0_int", c_uint32, 1),
            ("event_cnt_1_int", c_uint32, 1),
            ("event_cnt_2_int", c_uint32, 1),
            ("event_cnt_3_int", c_uint32, 1),
            ("reserved0", c_uint32, 27),
            ("cycle_cnt_int", c_uint32, 1),
        ]
    _fields_ = [("bits", _bitfield),
                ("word", c_uint32)]
    def set_event_cnt_0_int(self, value): self.bits.event_cnt_0_int = value
    def get_event_cnt_0_int(self): value = self.bits.event_cnt_0_int; return value
    def set_event_cnt_1_int(self, value): self.bits.event_cnt_1_int = value
    def get_event_cnt_1_int(self): value = self.bits.event_cnt_1_int; return value
    def set_event_cnt_2_int(self, value): self.bits.event_cnt_2_int = value
    def get_event_cnt_2_int(self): value = self.bits.event_cnt_2_int; return value
    def set_event_cnt_3_int(self, value): self.bits.event_cnt_3_int = value
    def get_event_cnt_3_int(self): value = self.bits.event_cnt_3_int; return value
    def set_cycle_cnt_int(self, value): self.bits.cycle_cnt_int = value
    def get_cycle_cnt_int(self): value = self.bits.cycle_cnt_int; return value


class pmintclr_r(Union):
    class _bitfield(Structure):
        _fields_ = [
            ("event_cnt_0_int", c_uint32, 1),
            ("event_cnt_1_int", c_uint32, 1),
            ("event_cnt_2_int", c_uint32, 1),
            ("event_cnt_3_int", c_uint32, 1),
            ("reserved0", c_uint32, 27),
            ("cycle_cnt_int", c_uint32, 1),
        ]
    _fields_ = [("bits", _bitfield),
                ("word", c_uint32)]
    def set_event_cnt_0_int(self, value): self.bits.event_cnt_0_int = value
    def get_event_cnt_0_int(self): value = self.bits.event_cnt_0_int; return value
    def set_event_cnt_1_int(self, value): self.bits.event_cnt_1_int = value
    def get_event_cnt_1_int(self): value = self.bits.event_cnt_1_int; return value
    def set_event_cnt_2_int(self, value): self.bits.event_cnt_2_int = value
    def get_event_cnt_2_int(self): value = self.bits.event_cnt_2_int; return value
    def set_event_cnt_3_int(self, value): self.bits.event_cnt_3_int = value
    def get_event_cnt_3_int(self): value = self.bits.event_cnt_3_int; return value
    def set_cycle_cnt_int(self, value): self.bits.cycle_cnt_int = value
    def get_cycle_cnt_int(self): value = self.bits.cycle_cnt_int; return value


class pmccntr_lo_r(Union):
    class _bitfield(Structure):
        _fields_ = [
            ("cycle_cnt_lo", c_uint32, 32),
        ]
    _fields_ = [("bits", _bitfield),
                ("word", c_uint32)]
    def set_cycle_cnt_lo(self, value): self.bits.cycle_cnt_lo = value
    def get_cycle_cnt_lo(self): value = self.bits.cycle_cnt_lo; return value


class pmccntr_hi_r(Union):
    class _bitfield(Structure):
        _fields_ = [
            ("cycle_cnt_hi", c_uint32, 16),
            ("reserved0", c_uint32, 16),
        ]
    _fields_ = [("bits", _bitfield),
                ("word", c_uint32)]
    def set_cycle_cnt_hi(self, value): self.bits.cycle_cnt_hi = value
    def get_cycle_cnt_hi(self): value = self.bits.cycle_cnt_hi; return value


class pmccntr_cfg_r(Union):
    class _bitfield(Structure):
        _fields_ = [
            ("cycle_cnt_cfg_start", c_uint32, 10),
            ("reserved0", c_uint32, 6),
            ("cycle_cnt_cfg_stop", c_uint32, 10),
            ("reserved1", c_uint32, 6),
        ]
    _fields_ = [("bits", _bitfield),
                ("word", c_uint32)]
    def set_cycle_cnt_cfg_start(self, value): self.bits.cycle_cnt_cfg_start = value
    def get_cycle_cnt_cfg_start(self): value = self.bits.cycle_cnt_cfg_start; return value
    def set_cycle_cnt_cfg_stop(self, value): self.bits.cycle_cnt_cfg_stop = value
    def get_cycle_cnt_cfg_stop(self): value = self.bits.cycle_cnt_cfg_stop; return value


class pmcaxi_chan_r(Union):
    class _bitfield(Structure):
        _fields_ = [
            ("ch_sel", c_uint32, 4),
            ("reserved0", c_uint32, 4),
            ("axi_cnt_sel", c_uint32, 2),
            ("bw_ch_sel_en", c_uint32, 1),
            ("reserved1", c_uint32, 21),
        ]
    _fields_ = [("bits", _bitfield),
                ("word", c_uint32)]
    def set_ch_sel(self, value): self.bits.ch_sel = value
    def get_ch_sel(self): value = self.bits.ch_sel; return value
    def set_axi_cnt_sel(self, value): self.bits.axi_cnt_sel = value
    def get_axi_cnt_sel(self): value = self.bits.axi_cnt_sel; return value
    def set_bw_ch_sel_en(self, value): self.bits.bw_ch_sel_en = value
    def get_bw_ch_sel_en(self): value = self.bits.bw_ch_sel_en; return value


class pmevtyper0_r(Union):
    class _bitfield(Structure):
        _fields_ = [
            ("ev_type", c_uint32, 10),
            ("reserved0", c_uint32, 22),
        ]
    _fields_ = [("bits", _bitfield),
                ("word", c_uint32)]
    def set_ev_type(self, value): self.bits.ev_type = value
    def get_ev_type(self): value = self.bits.ev_type; return value


class pmevtyper1_r(Union):
    class _bitfield(Structure):
        _fields_ = [
            ("ev_type", c_uint32, 10),
            ("reserved0", c_uint32, 22),
        ]
    _fields_ = [("bits", _bitfield),
                ("word", c_uint32)]
    def set_ev_type(self, value): self.bits.ev_type = value
    def get_ev_type(self): value = self.bits.ev_type; return value


class pmevtyper2_r(Union):
    class _bitfield(Structure):
        _fields_ = [
            ("ev_type", c_uint32, 10),
            ("reserved0", c_uint32, 22),
        ]
    _fields_ = [("bits", _bitfield),
                ("word", c_uint32)]
    def set_ev_type(self, value): self.bits.ev_type = value
    def get_ev_type(self): value = self.bits.ev_type; return value


class pmevtyper3_r(Union):
    class _bitfield(Structure):
        _fields_ = [
            ("ev_type", c_uint32, 10),
            ("reserved0", c_uint32, 22),
        ]
    _fields_ = [("bits", _bitfield),
                ("word", c_uint32)]
    def set_ev_type(self, value): self.bits.ev_type = value
    def get_ev_type(self): value = self.bits.ev_type; return value

class command_no_payload_t(Structure):
    _fields_ = [
        ("cmd_code", c_uint32, 10),
        ("must_be_zero0", c_uint32, 6),
        ("param", c_uint32, 16),
    ]
    def valid(self): return must_be_zero0==0;
    def get_cmd_code(self): return cmd_code
    def set_cmd_code(self, value): cmd_code = value
    def get_param(self): return param
    def set_param(self, value): param = value

class command_with_payload_t(Structure):
    _fields_ = [
        ("cmd_code", c_uint32, 10),
        ("must_be_zero", c_uint32, 4),
        ("payload_size", c_uint32, 2),
        ("param", c_uint32, 16),
        ("data", c_uint32, 32),
    ]
    def valid(self): return must_be_zero==0 and payload_size>=1 and payload_size<=2;
    def get_cmd_code(self): return cmd_code
    def set_cmd_code(self, value): cmd_code = value
    def get_data(self): return data
    def set_data(self, value): data = value
    def get_param(self): return param
    def set_param(self, value): param = value
    def get_payload_size(self): return payload_size
    def set_payload_size(self, value): payload_size = value

class npu_op_stop_t(Structure):
    _fields_ = [
        ("cmd_code", c_uint32, 10),
        ("must_be_zero0", c_uint32, 6),
        ("mask", c_uint32, 16),
    ]
    def valid(self): return cmd_code==cmd0.NPU_OP_STOP and must_be_zero0==0;
    def get_cmd_code(self): return cmd_code
    def set_cmd_code(self, value): cmd_code = value
    def get_mask(self): return mask
    def set_mask(self, value): mask = value

class npu_op_irq_t(Structure):
    _fields_ = [
        ("cmd_code", c_uint32, 10),
        ("must_be_zero0", c_uint32, 6),
        ("mask", c_uint32, 16),
    ]
    def valid(self): return cmd_code==cmd0.NPU_OP_IRQ and must_be_zero0==0;
    def get_cmd_code(self): return cmd_code
    def set_cmd_code(self, value): cmd_code = value
    def get_mask(self): return mask
    def set_mask(self, value): mask = value

class npu_op_conv_t(Structure):
    _fields_ = [
        ("cmd_code", c_uint32, 10),
        ("must_be_zero0", c_uint32, 6),
        ("reserved0", c_uint32, 16),
    ]
    def valid(self): return cmd_code==cmd0.NPU_OP_CONV and must_be_zero0==0;
    def get_cmd_code(self): return cmd_code
    def set_cmd_code(self, value): cmd_code = value

class npu_op_depthwise_t(Structure):
    _fields_ = [
        ("cmd_code", c_uint32, 10),
        ("must_be_zero0", c_uint32, 6),
        ("reserved0", c_uint32, 16),
    ]
    def valid(self): return cmd_code==cmd0.NPU_OP_DEPTHWISE and must_be_zero0==0;
    def get_cmd_code(self): return cmd_code
    def set_cmd_code(self, value): cmd_code = value

class npu_op_pool_t(Structure):
    _fields_ = [
        ("cmd_code", c_uint32, 10),
        ("must_be_zero0", c_uint32, 6),
        ("mode", c_uint32, 16),
    ]
    def valid(self): return cmd_code==cmd0.NPU_OP_POOL and must_be_zero0==0;
    def get_cmd_code(self): return cmd_code
    def set_cmd_code(self, value): cmd_code = value
    def get_mode(self): return mode
    def set_mode(self, value): mode = value

class npu_op_elementwise_t(Structure):
    _fields_ = [
        ("cmd_code", c_uint32, 10),
        ("must_be_zero0", c_uint32, 6),
        ("mode", c_uint32, 16),
    ]
    def valid(self): return cmd_code==cmd0.NPU_OP_ELEMENTWISE and must_be_zero0==0;
    def get_cmd_code(self): return cmd_code
    def set_cmd_code(self, value): cmd_code = value
    def get_mode(self): return mode
    def set_mode(self, value): mode = value

class npu_op_dma_start_t(Structure):
    _fields_ = [
        ("cmd_code", c_uint32, 10),
        ("must_be_zero0", c_uint32, 6),
        ("channel_mode", c_uint32, 16),
    ]
    def valid(self): return cmd_code==cmd0.NPU_OP_DMA_START and must_be_zero0==0;
    def get_channel_mode(self): return channel_mode
    def set_channel_mode(self, value): channel_mode = value
    def get_cmd_code(self): return cmd_code
    def set_cmd_code(self, value): cmd_code = value

class npu_op_dma_wait_t(Structure):
    _fields_ = [
        ("cmd_code", c_uint32, 10),
        ("must_be_zero0", c_uint32, 6),
        ("reserved0", c_uint32, 16),
    ]
    def valid(self): return cmd_code==cmd0.NPU_OP_DMA_WAIT and must_be_zero0==0;
    def get_cmd_code(self): return cmd_code
    def set_cmd_code(self, value): cmd_code = value

class npu_op_kernel_wait_t(Structure):
    _fields_ = [
        ("cmd_code", c_uint32, 10),
        ("must_be_zero0", c_uint32, 6),
        ("param", c_uint32, 16),
    ]
    def valid(self): return cmd_code==cmd0.NPU_OP_KERNEL_WAIT and must_be_zero0==0;
    def get_cmd_code(self): return cmd_code
    def set_cmd_code(self, value): cmd_code = value
    def get_param(self): return param
    def set_param(self, value): param = value

class npu_op_pmu_mask_t(Structure):
    _fields_ = [
        ("cmd_code", c_uint32, 10),
        ("must_be_zero0", c_uint32, 6),
        ("param", c_uint32, 16),
    ]
    def valid(self): return cmd_code==cmd0.NPU_OP_PMU_MASK and must_be_zero0==0;
    def get_cmd_code(self): return cmd_code
    def set_cmd_code(self, value): cmd_code = value
    def get_param(self): return param
    def set_param(self, value): param = value

class npu_set_ifm_pad_top_t(Structure):
    _fields_ = [
        ("cmd_code", c_uint32, 10),
        ("must_be_zero0", c_uint32, 6),
        ("param", c_uint32, 16),
    ]
    def valid(self): return cmd_code==cmd0.NPU_SET_IFM_PAD_TOP and must_be_zero0==0;
    def get_cmd_code(self): return cmd_code
    def set_cmd_code(self, value): cmd_code = value
    def get_param(self): return param
    def set_param(self, value): param = value

class npu_set_ifm_pad_left_t(Structure):
    _fields_ = [
        ("cmd_code", c_uint32, 10),
        ("must_be_zero0", c_uint32, 6),
        ("param", c_uint32, 16),
    ]
    def valid(self): return cmd_code==cmd0.NPU_SET_IFM_PAD_LEFT and must_be_zero0==0;
    def get_cmd_code(self): return cmd_code
    def set_cmd_code(self, value): cmd_code = value
    def get_param(self): return param
    def set_param(self, value): param = value

class npu_set_ifm_pad_right_t(Structure):
    _fields_ = [
        ("cmd_code", c_uint32, 10),
        ("must_be_zero0", c_uint32, 6),
        ("param", c_uint32, 16),
    ]
    def valid(self): return cmd_code==cmd0.NPU_SET_IFM_PAD_RIGHT and must_be_zero0==0;
    def get_cmd_code(self): return cmd_code
    def set_cmd_code(self, value): cmd_code = value
    def get_param(self): return param
    def set_param(self, value): param = value

class npu_set_ifm_pad_bottom_t(Structure):
    _fields_ = [
        ("cmd_code", c_uint32, 10),
        ("must_be_zero0", c_uint32, 6),
        ("param", c_uint32, 16),
    ]
    def valid(self): return cmd_code==cmd0.NPU_SET_IFM_PAD_BOTTOM and must_be_zero0==0;
    def get_cmd_code(self): return cmd_code
    def set_cmd_code(self, value): cmd_code = value
    def get_param(self): return param
    def set_param(self, value): param = value

class npu_set_ifm_depth_m1_t(Structure):
    _fields_ = [
        ("cmd_code", c_uint32, 10),
        ("must_be_zero0", c_uint32, 6),
        ("param", c_uint32, 16),
    ]
    def valid(self): return cmd_code==cmd0.NPU_SET_IFM_DEPTH_M1 and must_be_zero0==0;
    def get_cmd_code(self): return cmd_code
    def set_cmd_code(self, value): cmd_code = value
    def get_param(self): return param
    def set_param(self, value): param = value

class npu_set_ifm_precision_t(Structure):
    _fields_ = [
        ("cmd_code", c_uint32, 10),
        ("must_be_zero0", c_uint32, 6),
        ("precision", c_uint32, 4),
        ("reserved0", c_uint32, 2),
        ("format", c_uint32, 2),
        ("scale_mode", c_uint32, 2),
        ("reserved1", c_uint32, 4),
        ("round_mode", c_uint32, 2),
    ]
    def valid(self): return cmd_code==cmd0.NPU_SET_IFM_PRECISION and must_be_zero0==0;
    def get_cmd_code(self): return cmd_code
    def set_cmd_code(self, value): cmd_code = value
    def get_format(self): return format
    def set_format(self, value): format = value
    def get_precision(self): return precision
    def set_precision(self, value): precision = value
    def get_round_mode(self): return round_mode
    def set_round_mode(self, value): round_mode = value
    def get_scale_mode(self): return scale_mode
    def set_scale_mode(self, value): scale_mode = value

class npu_set_ifm_upscale_t(Structure):
    _fields_ = [
        ("cmd_code", c_uint32, 10),
        ("must_be_zero0", c_uint32, 6),
        ("mode", c_uint32, 2),
        ("reserved0", c_uint32, 14),
    ]
    def valid(self): return cmd_code==cmd0.NPU_SET_IFM_UPSCALE and must_be_zero0==0;
    def get_cmd_code(self): return cmd_code
    def set_cmd_code(self, value): cmd_code = value
    def get_mode(self): return mode
    def set_mode(self, value): mode = value

class npu_set_ifm_zero_point_t(Structure):
    _fields_ = [
        ("cmd_code", c_uint32, 10),
        ("must_be_zero0", c_uint32, 6),
        ("param", c_uint32, 16),
    ]
    def valid(self): return cmd_code==cmd0.NPU_SET_IFM_ZERO_POINT and must_be_zero0==0;
    def get_cmd_code(self): return cmd_code
    def set_cmd_code(self, value): cmd_code = value
    def get_param(self): return param
    def set_param(self, value): param = value

class npu_set_ifm_width0_m1_t(Structure):
    _fields_ = [
        ("cmd_code", c_uint32, 10),
        ("must_be_zero0", c_uint32, 6),
        ("param", c_uint32, 16),
    ]
    def valid(self): return cmd_code==cmd0.NPU_SET_IFM_WIDTH0_M1 and must_be_zero0==0;
    def get_cmd_code(self): return cmd_code
    def set_cmd_code(self, value): cmd_code = value
    def get_param(self): return param
    def set_param(self, value): param = value

class npu_set_ifm_height0_m1_t(Structure):
    _fields_ = [
        ("cmd_code", c_uint32, 10),
        ("must_be_zero0", c_uint32, 6),
        ("param", c_uint32, 16),
    ]
    def valid(self): return cmd_code==cmd0.NPU_SET_IFM_HEIGHT0_M1 and must_be_zero0==0;
    def get_cmd_code(self): return cmd_code
    def set_cmd_code(self, value): cmd_code = value
    def get_param(self): return param
    def set_param(self, value): param = value

class npu_set_ifm_height1_m1_t(Structure):
    _fields_ = [
        ("cmd_code", c_uint32, 10),
        ("must_be_zero0", c_uint32, 6),
        ("param", c_uint32, 16),
    ]
    def valid(self): return cmd_code==cmd0.NPU_SET_IFM_HEIGHT1_M1 and must_be_zero0==0;
    def get_cmd_code(self): return cmd_code
    def set_cmd_code(self, value): cmd_code = value
    def get_param(self): return param
    def set_param(self, value): param = value

class npu_set_ifm_ib_end_t(Structure):
    _fields_ = [
        ("cmd_code", c_uint32, 10),
        ("must_be_zero0", c_uint32, 6),
        ("param", c_uint32, 16),
    ]
    def valid(self): return cmd_code==cmd0.NPU_SET_IFM_IB_END and must_be_zero0==0;
    def get_cmd_code(self): return cmd_code
    def set_cmd_code(self, value): cmd_code = value
    def get_param(self): return param
    def set_param(self, value): param = value

class npu_set_ifm_region_t(Structure):
    _fields_ = [
        ("cmd_code", c_uint32, 10),
        ("must_be_zero0", c_uint32, 6),
        ("param", c_uint32, 16),
    ]
    def valid(self): return cmd_code==cmd0.NPU_SET_IFM_REGION and must_be_zero0==0;
    def get_cmd_code(self): return cmd_code
    def set_cmd_code(self, value): cmd_code = value
    def get_param(self): return param
    def set_param(self, value): param = value

class npu_set_ofm_width_m1_t(Structure):
    _fields_ = [
        ("cmd_code", c_uint32, 10),
        ("must_be_zero0", c_uint32, 6),
        ("param", c_uint32, 16),
    ]
    def valid(self): return cmd_code==cmd0.NPU_SET_OFM_WIDTH_M1 and must_be_zero0==0;
    def get_cmd_code(self): return cmd_code
    def set_cmd_code(self, value): cmd_code = value
    def get_param(self): return param
    def set_param(self, value): param = value

class npu_set_ofm_height_m1_t(Structure):
    _fields_ = [
        ("cmd_code", c_uint32, 10),
        ("must_be_zero0", c_uint32, 6),
        ("param", c_uint32, 16),
    ]
    def valid(self): return cmd_code==cmd0.NPU_SET_OFM_HEIGHT_M1 and must_be_zero0==0;
    def get_cmd_code(self): return cmd_code
    def set_cmd_code(self, value): cmd_code = value
    def get_param(self): return param
    def set_param(self, value): param = value

class npu_set_ofm_depth_m1_t(Structure):
    _fields_ = [
        ("cmd_code", c_uint32, 10),
        ("must_be_zero0", c_uint32, 6),
        ("param", c_uint32, 16),
    ]
    def valid(self): return cmd_code==cmd0.NPU_SET_OFM_DEPTH_M1 and must_be_zero0==0;
    def get_cmd_code(self): return cmd_code
    def set_cmd_code(self, value): cmd_code = value
    def get_param(self): return param
    def set_param(self, value): param = value

class npu_set_ofm_precision_t(Structure):
    _fields_ = [
        ("cmd_code", c_uint32, 10),
        ("must_be_zero0", c_uint32, 6),
        ("precision", c_uint32, 3),
        ("reserved0", c_uint32, 3),
        ("format", c_uint32, 2),
        ("scaling", c_uint32, 1),
        ("reserved1", c_uint32, 5),
        ("rounding", c_uint32, 2),
    ]
    def valid(self): return cmd_code==cmd0.NPU_SET_OFM_PRECISION and must_be_zero0==0;
    def get_cmd_code(self): return cmd_code
    def set_cmd_code(self, value): cmd_code = value
    def get_format(self): return format
    def set_format(self, value): format = value
    def get_precision(self): return precision
    def set_precision(self, value): precision = value
    def get_rounding(self): return rounding
    def set_rounding(self, value): rounding = value
    def get_scaling(self): return scaling
    def set_scaling(self, value): scaling = value

class npu_set_ofm_blk_width_m1_t(Structure):
    _fields_ = [
        ("cmd_code", c_uint32, 10),
        ("must_be_zero0", c_uint32, 6),
        ("param", c_uint32, 16),
    ]
    def valid(self): return cmd_code==cmd0.NPU_SET_OFM_BLK_WIDTH_M1 and must_be_zero0==0;
    def get_cmd_code(self): return cmd_code
    def set_cmd_code(self, value): cmd_code = value
    def get_param(self): return param
    def set_param(self, value): param = value

class npu_set_ofm_blk_height_m1_t(Structure):
    _fields_ = [
        ("cmd_code", c_uint32, 10),
        ("must_be_zero0", c_uint32, 6),
        ("param", c_uint32, 16),
    ]
    def valid(self): return cmd_code==cmd0.NPU_SET_OFM_BLK_HEIGHT_M1 and must_be_zero0==0;
    def get_cmd_code(self): return cmd_code
    def set_cmd_code(self, value): cmd_code = value
    def get_param(self): return param
    def set_param(self, value): param = value

class npu_set_ofm_blk_depth_m1_t(Structure):
    _fields_ = [
        ("cmd_code", c_uint32, 10),
        ("must_be_zero0", c_uint32, 6),
        ("param", c_uint32, 16),
    ]
    def valid(self): return cmd_code==cmd0.NPU_SET_OFM_BLK_DEPTH_M1 and must_be_zero0==0;
    def get_cmd_code(self): return cmd_code
    def set_cmd_code(self, value): cmd_code = value
    def get_param(self): return param
    def set_param(self, value): param = value

class npu_set_ofm_zero_point_t(Structure):
    _fields_ = [
        ("cmd_code", c_uint32, 10),
        ("must_be_zero0", c_uint32, 6),
        ("param", c_uint32, 16),
    ]
    def valid(self): return cmd_code==cmd0.NPU_SET_OFM_ZERO_POINT and must_be_zero0==0;
    def get_cmd_code(self): return cmd_code
    def set_cmd_code(self, value): cmd_code = value
    def get_param(self): return param
    def set_param(self, value): param = value

class npu_set_ofm_width0_m1_t(Structure):
    _fields_ = [
        ("cmd_code", c_uint32, 10),
        ("must_be_zero0", c_uint32, 6),
        ("param", c_uint32, 16),
    ]
    def valid(self): return cmd_code==cmd0.NPU_SET_OFM_WIDTH0_M1 and must_be_zero0==0;
    def get_cmd_code(self): return cmd_code
    def set_cmd_code(self, value): cmd_code = value
    def get_param(self): return param
    def set_param(self, value): param = value

class npu_set_ofm_height0_m1_t(Structure):
    _fields_ = [
        ("cmd_code", c_uint32, 10),
        ("must_be_zero0", c_uint32, 6),
        ("param", c_uint32, 16),
    ]
    def valid(self): return cmd_code==cmd0.NPU_SET_OFM_HEIGHT0_M1 and must_be_zero0==0;
    def get_cmd_code(self): return cmd_code
    def set_cmd_code(self, value): cmd_code = value
    def get_param(self): return param
    def set_param(self, value): param = value

class npu_set_ofm_height1_m1_t(Structure):
    _fields_ = [
        ("cmd_code", c_uint32, 10),
        ("must_be_zero0", c_uint32, 6),
        ("param", c_uint32, 16),
    ]
    def valid(self): return cmd_code==cmd0.NPU_SET_OFM_HEIGHT1_M1 and must_be_zero0==0;
    def get_cmd_code(self): return cmd_code
    def set_cmd_code(self, value): cmd_code = value
    def get_param(self): return param
    def set_param(self, value): param = value

class npu_set_ofm_region_t(Structure):
    _fields_ = [
        ("cmd_code", c_uint32, 10),
        ("must_be_zero0", c_uint32, 6),
        ("param", c_uint32, 16),
    ]
    def valid(self): return cmd_code==cmd0.NPU_SET_OFM_REGION and must_be_zero0==0;
    def get_cmd_code(self): return cmd_code
    def set_cmd_code(self, value): cmd_code = value
    def get_param(self): return param
    def set_param(self, value): param = value

class npu_set_kernel_width_m1_t(Structure):
    _fields_ = [
        ("cmd_code", c_uint32, 10),
        ("must_be_zero0", c_uint32, 6),
        ("param", c_uint32, 16),
    ]
    def valid(self): return cmd_code==cmd0.NPU_SET_KERNEL_WIDTH_M1 and must_be_zero0==0;
    def get_cmd_code(self): return cmd_code
    def set_cmd_code(self, value): cmd_code = value
    def get_param(self): return param
    def set_param(self, value): param = value

class npu_set_kernel_height_m1_t(Structure):
    _fields_ = [
        ("cmd_code", c_uint32, 10),
        ("must_be_zero0", c_uint32, 6),
        ("param", c_uint32, 16),
    ]
    def valid(self): return cmd_code==cmd0.NPU_SET_KERNEL_HEIGHT_M1 and must_be_zero0==0;
    def get_cmd_code(self): return cmd_code
    def set_cmd_code(self, value): cmd_code = value
    def get_param(self): return param
    def set_param(self, value): param = value

class npu_set_kernel_stride_t(Structure):
    _fields_ = [
        ("cmd_code", c_uint32, 10),
        ("must_be_zero0", c_uint32, 6),
        ("param", c_uint32, 16),
    ]
    def valid(self): return cmd_code==cmd0.NPU_SET_KERNEL_STRIDE and must_be_zero0==0;
    def get_cmd_code(self): return cmd_code
    def set_cmd_code(self, value): cmd_code = value
    def get_param(self): return param
    def set_param(self, value): param = value

class npu_set_parallel_mode_t(Structure):
    _fields_ = [
        ("cmd_code", c_uint32, 10),
        ("must_be_zero0", c_uint32, 6),
        ("param", c_uint32, 16),
    ]
    def valid(self): return cmd_code==cmd0.NPU_SET_PARALLEL_MODE and must_be_zero0==0;
    def get_cmd_code(self): return cmd_code
    def set_cmd_code(self, value): cmd_code = value
    def get_param(self): return param
    def set_param(self, value): param = value

class npu_set_acc_format_t(Structure):
    _fields_ = [
        ("cmd_code", c_uint32, 10),
        ("must_be_zero0", c_uint32, 6),
        ("param", c_uint32, 16),
    ]
    def valid(self): return cmd_code==cmd0.NPU_SET_ACC_FORMAT and must_be_zero0==0;
    def get_cmd_code(self): return cmd_code
    def set_cmd_code(self, value): cmd_code = value
    def get_param(self): return param
    def set_param(self, value): param = value

class npu_set_activation_t(Structure):
    _fields_ = [
        ("cmd_code", c_uint32, 10),
        ("must_be_zero0", c_uint32, 6),
        ("type", c_uint32, 12),
        ("act_clip_range", c_uint32, 4),
    ]
    def valid(self): return cmd_code==cmd0.NPU_SET_ACTIVATION and must_be_zero0==0;
    def get_act_clip_range(self): return act_clip_range
    def set_act_clip_range(self, value): act_clip_range = value
    def get_cmd_code(self): return cmd_code
    def set_cmd_code(self, value): cmd_code = value
    def get_type(self): return type
    def set_type(self, value): type = value

class npu_set_activation_min_t(Structure):
    _fields_ = [
        ("cmd_code", c_uint32, 10),
        ("must_be_zero0", c_uint32, 6),
        ("param", c_uint32, 16),
    ]
    def valid(self): return cmd_code==cmd0.NPU_SET_ACTIVATION_MIN and must_be_zero0==0;
    def get_cmd_code(self): return cmd_code
    def set_cmd_code(self, value): cmd_code = value
    def get_param(self): return param
    def set_param(self, value): param = value

class npu_set_activation_max_t(Structure):
    _fields_ = [
        ("cmd_code", c_uint32, 10),
        ("must_be_zero0", c_uint32, 6),
        ("param", c_uint32, 16),
    ]
    def valid(self): return cmd_code==cmd0.NPU_SET_ACTIVATION_MAX and must_be_zero0==0;
    def get_cmd_code(self): return cmd_code
    def set_cmd_code(self, value): cmd_code = value
    def get_param(self): return param
    def set_param(self, value): param = value

class npu_set_weight_region_t(Structure):
    _fields_ = [
        ("cmd_code", c_uint32, 10),
        ("must_be_zero0", c_uint32, 6),
        ("param", c_uint32, 16),
    ]
    def valid(self): return cmd_code==cmd0.NPU_SET_WEIGHT_REGION and must_be_zero0==0;
    def get_cmd_code(self): return cmd_code
    def set_cmd_code(self, value): cmd_code = value
    def get_param(self): return param
    def set_param(self, value): param = value

class npu_set_scale_region_t(Structure):
    _fields_ = [
        ("cmd_code", c_uint32, 10),
        ("must_be_zero0", c_uint32, 6),
        ("param", c_uint32, 16),
    ]
    def valid(self): return cmd_code==cmd0.NPU_SET_SCALE_REGION and must_be_zero0==0;
    def get_cmd_code(self): return cmd_code
    def set_cmd_code(self, value): cmd_code = value
    def get_param(self): return param
    def set_param(self, value): param = value

class npu_set_ab_start_t(Structure):
    _fields_ = [
        ("cmd_code", c_uint32, 10),
        ("must_be_zero0", c_uint32, 6),
        ("param", c_uint32, 16),
    ]
    def valid(self): return cmd_code==cmd0.NPU_SET_AB_START and must_be_zero0==0;
    def get_cmd_code(self): return cmd_code
    def set_cmd_code(self, value): cmd_code = value
    def get_param(self): return param
    def set_param(self, value): param = value

class npu_set_blockdep_t(Structure):
    _fields_ = [
        ("cmd_code", c_uint32, 10),
        ("must_be_zero0", c_uint32, 6),
        ("param", c_uint32, 16),
    ]
    def valid(self): return cmd_code==cmd0.NPU_SET_BLOCKDEP and must_be_zero0==0;
    def get_cmd_code(self): return cmd_code
    def set_cmd_code(self, value): cmd_code = value
    def get_param(self): return param
    def set_param(self, value): param = value

class npu_set_dma0_src_region_t(Structure):
    _fields_ = [
        ("cmd_code", c_uint32, 10),
        ("must_be_zero0", c_uint32, 6),
        ("region", c_uint32, 8),
        ("internal", c_uint32, 1),
        ("stride_mode", c_uint32, 2),
        ("reserved0", c_uint32, 5),
    ]
    def valid(self): return cmd_code==cmd0.NPU_SET_DMA0_SRC_REGION and must_be_zero0==0;
    def get_cmd_code(self): return cmd_code
    def set_cmd_code(self, value): cmd_code = value
    def get_internal(self): return internal
    def set_internal(self, value): internal = value
    def get_region(self): return region
    def set_region(self, value): region = value
    def get_stride_mode(self): return stride_mode
    def set_stride_mode(self, value): stride_mode = value

class npu_set_dma0_dst_region_t(Structure):
    _fields_ = [
        ("cmd_code", c_uint32, 10),
        ("must_be_zero0", c_uint32, 6),
        ("region", c_uint32, 8),
        ("internal", c_uint32, 1),
        ("stride_mode", c_uint32, 2),
        ("reserved0", c_uint32, 5),
    ]
    def valid(self): return cmd_code==cmd0.NPU_SET_DMA0_DST_REGION and must_be_zero0==0;
    def get_cmd_code(self): return cmd_code
    def set_cmd_code(self, value): cmd_code = value
    def get_internal(self): return internal
    def set_internal(self, value): internal = value
    def get_region(self): return region
    def set_region(self, value): region = value
    def get_stride_mode(self): return stride_mode
    def set_stride_mode(self, value): stride_mode = value

class npu_set_dma0_size0_t(Structure):
    _fields_ = [
        ("cmd_code", c_uint32, 10),
        ("must_be_zero0", c_uint32, 6),
        ("param", c_uint32, 16),
    ]
    def valid(self): return cmd_code==cmd0.NPU_SET_DMA0_SIZE0 and must_be_zero0==0;
    def get_cmd_code(self): return cmd_code
    def set_cmd_code(self, value): cmd_code = value
    def get_param(self): return param
    def set_param(self, value): param = value

class npu_set_dma0_size1_t(Structure):
    _fields_ = [
        ("cmd_code", c_uint32, 10),
        ("must_be_zero0", c_uint32, 6),
        ("param", c_uint32, 16),
    ]
    def valid(self): return cmd_code==cmd0.NPU_SET_DMA0_SIZE1 and must_be_zero0==0;
    def get_cmd_code(self): return cmd_code
    def set_cmd_code(self, value): cmd_code = value
    def get_param(self): return param
    def set_param(self, value): param = value

class npu_set_ifm2_broadcast_t(Structure):
    _fields_ = [
        ("cmd_code", c_uint32, 10),
        ("must_be_zero0", c_uint32, 6),
        ("broadcast_height", c_uint32, 1),
        ("broadcast_width", c_uint32, 1),
        ("broadcast_depth", c_uint32, 1),
        ("reserved0", c_uint32, 3),
        ("operand_order", c_uint32, 1),
        ("broadcast_scalar", c_uint32, 1),
        ("reserved1", c_uint32, 8),
    ]
    def valid(self): return cmd_code==cmd0.NPU_SET_IFM2_BROADCAST and must_be_zero0==0;
    def get_broadcast_depth(self): return broadcast_depth
    def set_broadcast_depth(self, value): broadcast_depth = value
    def get_broadcast_height(self): return broadcast_height
    def set_broadcast_height(self, value): broadcast_height = value
    def get_broadcast_scalar(self): return broadcast_scalar
    def set_broadcast_scalar(self, value): broadcast_scalar = value
    def get_broadcast_width(self): return broadcast_width
    def set_broadcast_width(self, value): broadcast_width = value
    def get_cmd_code(self): return cmd_code
    def set_cmd_code(self, value): cmd_code = value
    def get_operand_order(self): return operand_order
    def set_operand_order(self, value): operand_order = value

class npu_set_ifm2_scalar_t(Structure):
    _fields_ = [
        ("cmd_code", c_uint32, 10),
        ("must_be_zero0", c_uint32, 6),
        ("param", c_uint32, 16),
    ]
    def valid(self): return cmd_code==cmd0.NPU_SET_IFM2_SCALAR and must_be_zero0==0;
    def get_cmd_code(self): return cmd_code
    def set_cmd_code(self, value): cmd_code = value
    def get_param(self): return param
    def set_param(self, value): param = value

class npu_set_ifm2_precision_t(Structure):
    _fields_ = [
        ("cmd_code", c_uint32, 10),
        ("must_be_zero0", c_uint32, 6),
        ("precision", c_uint32, 4),
        ("reserved0", c_uint32, 2),
        ("format", c_uint32, 2),
        ("reserved1", c_uint32, 8),
    ]
    def valid(self): return cmd_code==cmd0.NPU_SET_IFM2_PRECISION and must_be_zero0==0;
    def get_cmd_code(self): return cmd_code
    def set_cmd_code(self, value): cmd_code = value
    def get_format(self): return format
    def set_format(self, value): format = value
    def get_precision(self): return precision
    def set_precision(self, value): precision = value

class npu_set_ifm2_zero_point_t(Structure):
    _fields_ = [
        ("cmd_code", c_uint32, 10),
        ("must_be_zero0", c_uint32, 6),
        ("param", c_uint32, 16),
    ]
    def valid(self): return cmd_code==cmd0.NPU_SET_IFM2_ZERO_POINT and must_be_zero0==0;
    def get_cmd_code(self): return cmd_code
    def set_cmd_code(self, value): cmd_code = value
    def get_param(self): return param
    def set_param(self, value): param = value

class npu_set_ifm2_width0_m1_t(Structure):
    _fields_ = [
        ("cmd_code", c_uint32, 10),
        ("must_be_zero0", c_uint32, 6),
        ("param", c_uint32, 16),
    ]
    def valid(self): return cmd_code==cmd0.NPU_SET_IFM2_WIDTH0_M1 and must_be_zero0==0;
    def get_cmd_code(self): return cmd_code
    def set_cmd_code(self, value): cmd_code = value
    def get_param(self): return param
    def set_param(self, value): param = value

class npu_set_ifm2_height0_m1_t(Structure):
    _fields_ = [
        ("cmd_code", c_uint32, 10),
        ("must_be_zero0", c_uint32, 6),
        ("param", c_uint32, 16),
    ]
    def valid(self): return cmd_code==cmd0.NPU_SET_IFM2_HEIGHT0_M1 and must_be_zero0==0;
    def get_cmd_code(self): return cmd_code
    def set_cmd_code(self, value): cmd_code = value
    def get_param(self): return param
    def set_param(self, value): param = value

class npu_set_ifm2_height1_m1_t(Structure):
    _fields_ = [
        ("cmd_code", c_uint32, 10),
        ("must_be_zero0", c_uint32, 6),
        ("param", c_uint32, 16),
    ]
    def valid(self): return cmd_code==cmd0.NPU_SET_IFM2_HEIGHT1_M1 and must_be_zero0==0;
    def get_cmd_code(self): return cmd_code
    def set_cmd_code(self, value): cmd_code = value
    def get_param(self): return param
    def set_param(self, value): param = value

class npu_set_ifm2_ib_start_t(Structure):
    _fields_ = [
        ("cmd_code", c_uint32, 10),
        ("must_be_zero0", c_uint32, 6),
        ("param", c_uint32, 16),
    ]
    def valid(self): return cmd_code==cmd0.NPU_SET_IFM2_IB_START and must_be_zero0==0;
    def get_cmd_code(self): return cmd_code
    def set_cmd_code(self, value): cmd_code = value
    def get_param(self): return param
    def set_param(self, value): param = value

class npu_set_ifm2_region_t(Structure):
    _fields_ = [
        ("cmd_code", c_uint32, 10),
        ("must_be_zero0", c_uint32, 6),
        ("param", c_uint32, 16),
    ]
    def valid(self): return cmd_code==cmd0.NPU_SET_IFM2_REGION and must_be_zero0==0;
    def get_cmd_code(self): return cmd_code
    def set_cmd_code(self, value): cmd_code = value
    def get_param(self): return param
    def set_param(self, value): param = value

class npu_set_ifm_base0_t(Structure):
    _fields_ = [
        ("cmd_code", c_uint32, 10),
        ("must_be_zero", c_uint32, 4),
        ("payload_size", c_uint32, 2),
        ("reserved0", c_uint32, 16),
        ("data", c_uint32, 32),
    ]
    def valid(self): return cmd_code==cmd1.NPU_SET_IFM_BASE0 and must_be_zero==0 and payload_size>=1 and payload_size<=2;
    def get_cmd_code(self): return cmd_code
    def set_cmd_code(self, value): cmd_code = value
    def get_data(self): return data
    def set_data(self, value): data = value
    def get_payload_size(self): return payload_size
    def set_payload_size(self, value): payload_size = value

class npu_set_ifm_base1_t(Structure):
    _fields_ = [
        ("cmd_code", c_uint32, 10),
        ("must_be_zero", c_uint32, 4),
        ("payload_size", c_uint32, 2),
        ("reserved0", c_uint32, 16),
        ("data", c_uint32, 32),
    ]
    def valid(self): return cmd_code==cmd1.NPU_SET_IFM_BASE1 and must_be_zero==0 and payload_size>=1 and payload_size<=2;
    def get_cmd_code(self): return cmd_code
    def set_cmd_code(self, value): cmd_code = value
    def get_data(self): return data
    def set_data(self, value): data = value
    def get_payload_size(self): return payload_size
    def set_payload_size(self, value): payload_size = value

class npu_set_ifm_base2_t(Structure):
    _fields_ = [
        ("cmd_code", c_uint32, 10),
        ("must_be_zero", c_uint32, 4),
        ("payload_size", c_uint32, 2),
        ("reserved0", c_uint32, 16),
        ("data", c_uint32, 32),
    ]
    def valid(self): return cmd_code==cmd1.NPU_SET_IFM_BASE2 and must_be_zero==0 and payload_size>=1 and payload_size<=2;
    def get_cmd_code(self): return cmd_code
    def set_cmd_code(self, value): cmd_code = value
    def get_data(self): return data
    def set_data(self, value): data = value
    def get_payload_size(self): return payload_size
    def set_payload_size(self, value): payload_size = value

class npu_set_ifm_base3_t(Structure):
    _fields_ = [
        ("cmd_code", c_uint32, 10),
        ("must_be_zero", c_uint32, 4),
        ("payload_size", c_uint32, 2),
        ("reserved0", c_uint32, 16),
        ("data", c_uint32, 32),
    ]
    def valid(self): return cmd_code==cmd1.NPU_SET_IFM_BASE3 and must_be_zero==0 and payload_size>=1 and payload_size<=2;
    def get_cmd_code(self): return cmd_code
    def set_cmd_code(self, value): cmd_code = value
    def get_data(self): return data
    def set_data(self, value): data = value
    def get_payload_size(self): return payload_size
    def set_payload_size(self, value): payload_size = value

class npu_set_ifm_stride_x_t(Structure):
    _fields_ = [
        ("cmd_code", c_uint32, 10),
        ("must_be_zero", c_uint32, 4),
        ("payload_size", c_uint32, 2),
        ("reserved0", c_uint32, 16),
        ("data", c_uint32, 32),
    ]
    def valid(self): return cmd_code==cmd1.NPU_SET_IFM_STRIDE_X and must_be_zero==0 and payload_size>=1 and payload_size<=2;
    def get_cmd_code(self): return cmd_code
    def set_cmd_code(self, value): cmd_code = value
    def get_data(self): return data
    def set_data(self, value): data = value
    def get_payload_size(self): return payload_size
    def set_payload_size(self, value): payload_size = value

class npu_set_ifm_stride_y_t(Structure):
    _fields_ = [
        ("cmd_code", c_uint32, 10),
        ("must_be_zero", c_uint32, 4),
        ("payload_size", c_uint32, 2),
        ("reserved0", c_uint32, 16),
        ("data", c_uint32, 32),
    ]
    def valid(self): return cmd_code==cmd1.NPU_SET_IFM_STRIDE_Y and must_be_zero==0 and payload_size>=1 and payload_size<=2;
    def get_cmd_code(self): return cmd_code
    def set_cmd_code(self, value): cmd_code = value
    def get_data(self): return data
    def set_data(self, value): data = value
    def get_payload_size(self): return payload_size
    def set_payload_size(self, value): payload_size = value

class npu_set_ifm_stride_c_t(Structure):
    _fields_ = [
        ("cmd_code", c_uint32, 10),
        ("must_be_zero", c_uint32, 4),
        ("payload_size", c_uint32, 2),
        ("reserved0", c_uint32, 16),
        ("data", c_uint32, 32),
    ]
    def valid(self): return cmd_code==cmd1.NPU_SET_IFM_STRIDE_C and must_be_zero==0 and payload_size>=1 and payload_size<=2;
    def get_cmd_code(self): return cmd_code
    def set_cmd_code(self, value): cmd_code = value
    def get_data(self): return data
    def set_data(self, value): data = value
    def get_payload_size(self): return payload_size
    def set_payload_size(self, value): payload_size = value

class npu_set_ofm_base0_t(Structure):
    _fields_ = [
        ("cmd_code", c_uint32, 10),
        ("must_be_zero", c_uint32, 4),
        ("payload_size", c_uint32, 2),
        ("reserved0", c_uint32, 16),
        ("data", c_uint32, 32),
    ]
    def valid(self): return cmd_code==cmd1.NPU_SET_OFM_BASE0 and must_be_zero==0 and payload_size>=1 and payload_size<=2;
    def get_cmd_code(self): return cmd_code
    def set_cmd_code(self, value): cmd_code = value
    def get_data(self): return data
    def set_data(self, value): data = value
    def get_payload_size(self): return payload_size
    def set_payload_size(self, value): payload_size = value

class npu_set_ofm_base1_t(Structure):
    _fields_ = [
        ("cmd_code", c_uint32, 10),
        ("must_be_zero", c_uint32, 4),
        ("payload_size", c_uint32, 2),
        ("reserved0", c_uint32, 16),
        ("data", c_uint32, 32),
    ]
    def valid(self): return cmd_code==cmd1.NPU_SET_OFM_BASE1 and must_be_zero==0 and payload_size>=1 and payload_size<=2;
    def get_cmd_code(self): return cmd_code
    def set_cmd_code(self, value): cmd_code = value
    def get_data(self): return data
    def set_data(self, value): data = value
    def get_payload_size(self): return payload_size
    def set_payload_size(self, value): payload_size = value

class npu_set_ofm_base2_t(Structure):
    _fields_ = [
        ("cmd_code", c_uint32, 10),
        ("must_be_zero", c_uint32, 4),
        ("payload_size", c_uint32, 2),
        ("reserved0", c_uint32, 16),
        ("data", c_uint32, 32),
    ]
    def valid(self): return cmd_code==cmd1.NPU_SET_OFM_BASE2 and must_be_zero==0 and payload_size>=1 and payload_size<=2;
    def get_cmd_code(self): return cmd_code
    def set_cmd_code(self, value): cmd_code = value
    def get_data(self): return data
    def set_data(self, value): data = value
    def get_payload_size(self): return payload_size
    def set_payload_size(self, value): payload_size = value

class npu_set_ofm_base3_t(Structure):
    _fields_ = [
        ("cmd_code", c_uint32, 10),
        ("must_be_zero", c_uint32, 4),
        ("payload_size", c_uint32, 2),
        ("reserved0", c_uint32, 16),
        ("data", c_uint32, 32),
    ]
    def valid(self): return cmd_code==cmd1.NPU_SET_OFM_BASE3 and must_be_zero==0 and payload_size>=1 and payload_size<=2;
    def get_cmd_code(self): return cmd_code
    def set_cmd_code(self, value): cmd_code = value
    def get_data(self): return data
    def set_data(self, value): data = value
    def get_payload_size(self): return payload_size
    def set_payload_size(self, value): payload_size = value

class npu_set_ofm_stride_x_t(Structure):
    _fields_ = [
        ("cmd_code", c_uint32, 10),
        ("must_be_zero", c_uint32, 4),
        ("payload_size", c_uint32, 2),
        ("reserved0", c_uint32, 16),
        ("data", c_uint32, 32),
    ]
    def valid(self): return cmd_code==cmd1.NPU_SET_OFM_STRIDE_X and must_be_zero==0 and payload_size>=1 and payload_size<=2;
    def get_cmd_code(self): return cmd_code
    def set_cmd_code(self, value): cmd_code = value
    def get_data(self): return data
    def set_data(self, value): data = value
    def get_payload_size(self): return payload_size
    def set_payload_size(self, value): payload_size = value

class npu_set_ofm_stride_y_t(Structure):
    _fields_ = [
        ("cmd_code", c_uint32, 10),
        ("must_be_zero", c_uint32, 4),
        ("payload_size", c_uint32, 2),
        ("reserved0", c_uint32, 16),
        ("data", c_uint32, 32),
    ]
    def valid(self): return cmd_code==cmd1.NPU_SET_OFM_STRIDE_Y and must_be_zero==0 and payload_size>=1 and payload_size<=2;
    def get_cmd_code(self): return cmd_code
    def set_cmd_code(self, value): cmd_code = value
    def get_data(self): return data
    def set_data(self, value): data = value
    def get_payload_size(self): return payload_size
    def set_payload_size(self, value): payload_size = value

class npu_set_ofm_stride_c_t(Structure):
    _fields_ = [
        ("cmd_code", c_uint32, 10),
        ("must_be_zero", c_uint32, 4),
        ("payload_size", c_uint32, 2),
        ("reserved0", c_uint32, 16),
        ("data", c_uint32, 32),
    ]
    def valid(self): return cmd_code==cmd1.NPU_SET_OFM_STRIDE_C and must_be_zero==0 and payload_size>=1 and payload_size<=2;
    def get_cmd_code(self): return cmd_code
    def set_cmd_code(self, value): cmd_code = value
    def get_data(self): return data
    def set_data(self, value): data = value
    def get_payload_size(self): return payload_size
    def set_payload_size(self, value): payload_size = value

class npu_set_weight_base_t(Structure):
    _fields_ = [
        ("cmd_code", c_uint32, 10),
        ("must_be_zero", c_uint32, 4),
        ("payload_size", c_uint32, 2),
        ("reserved0", c_uint32, 16),
        ("data", c_uint32, 32),
    ]
    def valid(self): return cmd_code==cmd1.NPU_SET_WEIGHT_BASE and must_be_zero==0 and payload_size>=1 and payload_size<=2;
    def get_cmd_code(self): return cmd_code
    def set_cmd_code(self, value): cmd_code = value
    def get_data(self): return data
    def set_data(self, value): data = value
    def get_payload_size(self): return payload_size
    def set_payload_size(self, value): payload_size = value

class npu_set_weight_length_t(Structure):
    _fields_ = [
        ("cmd_code", c_uint32, 10),
        ("must_be_zero", c_uint32, 4),
        ("payload_size", c_uint32, 2),
        ("reserved0", c_uint32, 16),
        ("data", c_uint32, 32),
    ]
    def valid(self): return cmd_code==cmd1.NPU_SET_WEIGHT_LENGTH and must_be_zero==0 and payload_size>=1 and payload_size<=2;
    def get_cmd_code(self): return cmd_code
    def set_cmd_code(self, value): cmd_code = value
    def get_data(self): return data
    def set_data(self, value): data = value
    def get_payload_size(self): return payload_size
    def set_payload_size(self, value): payload_size = value

class npu_set_scale_base_t(Structure):
    _fields_ = [
        ("cmd_code", c_uint32, 10),
        ("must_be_zero", c_uint32, 4),
        ("payload_size", c_uint32, 2),
        ("reserved0", c_uint32, 16),
        ("data", c_uint32, 32),
    ]
    def valid(self): return cmd_code==cmd1.NPU_SET_SCALE_BASE and must_be_zero==0 and payload_size>=1 and payload_size<=2;
    def get_cmd_code(self): return cmd_code
    def set_cmd_code(self, value): cmd_code = value
    def get_data(self): return data
    def set_data(self, value): data = value
    def get_payload_size(self): return payload_size
    def set_payload_size(self, value): payload_size = value

class npu_set_scale_length_t(Structure):
    _fields_ = [
        ("cmd_code", c_uint32, 10),
        ("must_be_zero", c_uint32, 4),
        ("payload_size", c_uint32, 2),
        ("reserved0", c_uint32, 16),
        ("data", c_uint32, 32),
    ]
    def valid(self): return cmd_code==cmd1.NPU_SET_SCALE_LENGTH and must_be_zero==0 and payload_size>=1 and payload_size<=2;
    def get_cmd_code(self): return cmd_code
    def set_cmd_code(self, value): cmd_code = value
    def get_data(self): return data
    def set_data(self, value): data = value
    def get_payload_size(self): return payload_size
    def set_payload_size(self, value): payload_size = value

class npu_set_ofm_scale_t(Structure):
    _fields_ = [
        ("cmd_code", c_uint32, 10),
        ("must_be_zero", c_uint32, 4),
        ("payload_size", c_uint32, 2),
        ("shift", c_uint32, 16),
        ("data", c_uint32, 32),
    ]
    def valid(self): return cmd_code==cmd1.NPU_SET_OFM_SCALE and must_be_zero==0 and payload_size>=1 and payload_size<=2;
    def get_cmd_code(self): return cmd_code
    def set_cmd_code(self, value): cmd_code = value
    def get_data(self): return data
    def set_data(self, value): data = value
    def get_payload_size(self): return payload_size
    def set_payload_size(self, value): payload_size = value
    def get_shift(self): return shift
    def set_shift(self, value): shift = value

class npu_set_opa_scale_t(Structure):
    _fields_ = [
        ("cmd_code", c_uint32, 10),
        ("must_be_zero", c_uint32, 4),
        ("payload_size", c_uint32, 2),
        ("shift", c_uint32, 16),
        ("data", c_uint32, 32),
    ]
    def valid(self): return cmd_code==cmd1.NPU_SET_OPA_SCALE and must_be_zero==0 and payload_size>=1 and payload_size<=2;
    def get_cmd_code(self): return cmd_code
    def set_cmd_code(self, value): cmd_code = value
    def get_data(self): return data
    def set_data(self, value): data = value
    def get_payload_size(self): return payload_size
    def set_payload_size(self, value): payload_size = value
    def get_shift(self): return shift
    def set_shift(self, value): shift = value

class npu_set_opb_scale_t(Structure):
    _fields_ = [
        ("cmd_code", c_uint32, 10),
        ("must_be_zero", c_uint32, 4),
        ("payload_size", c_uint32, 2),
        ("reserved0", c_uint32, 16),
        ("data", c_uint32, 32),
    ]
    def valid(self): return cmd_code==cmd1.NPU_SET_OPB_SCALE and must_be_zero==0 and payload_size>=1 and payload_size<=2;
    def get_cmd_code(self): return cmd_code
    def set_cmd_code(self, value): cmd_code = value
    def get_data(self): return data
    def set_data(self, value): data = value
    def get_payload_size(self): return payload_size
    def set_payload_size(self, value): payload_size = value

class npu_set_dma0_src_t(Structure):
    _fields_ = [
        ("cmd_code", c_uint32, 10),
        ("must_be_zero", c_uint32, 4),
        ("payload_size", c_uint32, 2),
        ("reserved0", c_uint32, 16),
        ("data", c_uint32, 32),
    ]
    def valid(self): return cmd_code==cmd1.NPU_SET_DMA0_SRC and must_be_zero==0 and payload_size>=1 and payload_size<=2;
    def get_cmd_code(self): return cmd_code
    def set_cmd_code(self, value): cmd_code = value
    def get_data(self): return data
    def set_data(self, value): data = value
    def get_payload_size(self): return payload_size
    def set_payload_size(self, value): payload_size = value

class npu_set_dma0_dst_t(Structure):
    _fields_ = [
        ("cmd_code", c_uint32, 10),
        ("must_be_zero", c_uint32, 4),
        ("payload_size", c_uint32, 2),
        ("reserved0", c_uint32, 16),
        ("data", c_uint32, 32),
    ]
    def valid(self): return cmd_code==cmd1.NPU_SET_DMA0_DST and must_be_zero==0 and payload_size>=1 and payload_size<=2;
    def get_cmd_code(self): return cmd_code
    def set_cmd_code(self, value): cmd_code = value
    def get_data(self): return data
    def set_data(self, value): data = value
    def get_payload_size(self): return payload_size
    def set_payload_size(self, value): payload_size = value

class npu_set_dma0_len_t(Structure):
    _fields_ = [
        ("cmd_code", c_uint32, 10),
        ("must_be_zero", c_uint32, 4),
        ("payload_size", c_uint32, 2),
        ("reserved0", c_uint32, 16),
        ("data", c_uint32, 32),
    ]
    def valid(self): return cmd_code==cmd1.NPU_SET_DMA0_LEN and must_be_zero==0 and payload_size>=1 and payload_size<=2;
    def get_cmd_code(self): return cmd_code
    def set_cmd_code(self, value): cmd_code = value
    def get_data(self): return data
    def set_data(self, value): data = value
    def get_payload_size(self): return payload_size
    def set_payload_size(self, value): payload_size = value

class npu_set_dma0_skip0_t(Structure):
    _fields_ = [
        ("cmd_code", c_uint32, 10),
        ("must_be_zero", c_uint32, 4),
        ("payload_size", c_uint32, 2),
        ("param", c_uint32, 16),
        ("data", c_uint32, 32),
    ]
    def valid(self): return cmd_code==cmd1.NPU_SET_DMA0_SKIP0 and must_be_zero==0 and payload_size>=1 and payload_size<=2;
    def get_cmd_code(self): return cmd_code
    def set_cmd_code(self, value): cmd_code = value
    def get_data(self): return data
    def set_data(self, value): data = value
    def get_param(self): return param
    def set_param(self, value): param = value
    def get_payload_size(self): return payload_size
    def set_payload_size(self, value): payload_size = value

class npu_set_dma0_skip1_t(Structure):
    _fields_ = [
        ("cmd_code", c_uint32, 10),
        ("must_be_zero", c_uint32, 4),
        ("payload_size", c_uint32, 2),
        ("param", c_uint32, 16),
        ("data", c_uint32, 32),
    ]
    def valid(self): return cmd_code==cmd1.NPU_SET_DMA0_SKIP1 and must_be_zero==0 and payload_size>=1 and payload_size<=2;
    def get_cmd_code(self): return cmd_code
    def set_cmd_code(self, value): cmd_code = value
    def get_data(self): return data
    def set_data(self, value): data = value
    def get_param(self): return param
    def set_param(self, value): param = value
    def get_payload_size(self): return payload_size
    def set_payload_size(self, value): payload_size = value

class npu_set_ifm2_base0_t(Structure):
    _fields_ = [
        ("cmd_code", c_uint32, 10),
        ("must_be_zero", c_uint32, 4),
        ("payload_size", c_uint32, 2),
        ("reserved0", c_uint32, 16),
        ("data", c_uint32, 32),
    ]
    def valid(self): return cmd_code==cmd1.NPU_SET_IFM2_BASE0 and must_be_zero==0 and payload_size>=1 and payload_size<=2;
    def get_cmd_code(self): return cmd_code
    def set_cmd_code(self, value): cmd_code = value
    def get_data(self): return data
    def set_data(self, value): data = value
    def get_payload_size(self): return payload_size
    def set_payload_size(self, value): payload_size = value

class npu_set_ifm2_base1_t(Structure):
    _fields_ = [
        ("cmd_code", c_uint32, 10),
        ("must_be_zero", c_uint32, 4),
        ("payload_size", c_uint32, 2),
        ("reserved0", c_uint32, 16),
        ("data", c_uint32, 32),
    ]
    def valid(self): return cmd_code==cmd1.NPU_SET_IFM2_BASE1 and must_be_zero==0 and payload_size>=1 and payload_size<=2;
    def get_cmd_code(self): return cmd_code
    def set_cmd_code(self, value): cmd_code = value
    def get_data(self): return data
    def set_data(self, value): data = value
    def get_payload_size(self): return payload_size
    def set_payload_size(self, value): payload_size = value

class npu_set_ifm2_base2_t(Structure):
    _fields_ = [
        ("cmd_code", c_uint32, 10),
        ("must_be_zero", c_uint32, 4),
        ("payload_size", c_uint32, 2),
        ("reserved0", c_uint32, 16),
        ("data", c_uint32, 32),
    ]
    def valid(self): return cmd_code==cmd1.NPU_SET_IFM2_BASE2 and must_be_zero==0 and payload_size>=1 and payload_size<=2;
    def get_cmd_code(self): return cmd_code
    def set_cmd_code(self, value): cmd_code = value
    def get_data(self): return data
    def set_data(self, value): data = value
    def get_payload_size(self): return payload_size
    def set_payload_size(self, value): payload_size = value

class npu_set_ifm2_base3_t(Structure):
    _fields_ = [
        ("cmd_code", c_uint32, 10),
        ("must_be_zero", c_uint32, 4),
        ("payload_size", c_uint32, 2),
        ("reserved0", c_uint32, 16),
        ("data", c_uint32, 32),
    ]
    def valid(self): return cmd_code==cmd1.NPU_SET_IFM2_BASE3 and must_be_zero==0 and payload_size>=1 and payload_size<=2;
    def get_cmd_code(self): return cmd_code
    def set_cmd_code(self, value): cmd_code = value
    def get_data(self): return data
    def set_data(self, value): data = value
    def get_payload_size(self): return payload_size
    def set_payload_size(self, value): payload_size = value

class npu_set_ifm2_stride_x_t(Structure):
    _fields_ = [
        ("cmd_code", c_uint32, 10),
        ("must_be_zero", c_uint32, 4),
        ("payload_size", c_uint32, 2),
        ("reserved0", c_uint32, 16),
        ("data", c_uint32, 32),
    ]
    def valid(self): return cmd_code==cmd1.NPU_SET_IFM2_STRIDE_X and must_be_zero==0 and payload_size>=1 and payload_size<=2;
    def get_cmd_code(self): return cmd_code
    def set_cmd_code(self, value): cmd_code = value
    def get_data(self): return data
    def set_data(self, value): data = value
    def get_payload_size(self): return payload_size
    def set_payload_size(self, value): payload_size = value

class npu_set_ifm2_stride_y_t(Structure):
    _fields_ = [
        ("cmd_code", c_uint32, 10),
        ("must_be_zero", c_uint32, 4),
        ("payload_size", c_uint32, 2),
        ("reserved0", c_uint32, 16),
        ("data", c_uint32, 32),
    ]
    def valid(self): return cmd_code==cmd1.NPU_SET_IFM2_STRIDE_Y and must_be_zero==0 and payload_size>=1 and payload_size<=2;
    def get_cmd_code(self): return cmd_code
    def set_cmd_code(self, value): cmd_code = value
    def get_data(self): return data
    def set_data(self, value): data = value
    def get_payload_size(self): return payload_size
    def set_payload_size(self, value): payload_size = value

class npu_set_ifm2_stride_c_t(Structure):
    _fields_ = [
        ("cmd_code", c_uint32, 10),
        ("must_be_zero", c_uint32, 4),
        ("payload_size", c_uint32, 2),
        ("reserved0", c_uint32, 16),
        ("data", c_uint32, 32),
    ]
    def valid(self): return cmd_code==cmd1.NPU_SET_IFM2_STRIDE_C and must_be_zero==0 and payload_size>=1 and payload_size<=2;
    def get_cmd_code(self): return cmd_code
    def set_cmd_code(self, value): cmd_code = value
    def get_data(self): return data
    def set_data(self, value): data = value
    def get_payload_size(self): return payload_size
    def set_payload_size(self, value): payload_size = value

class npu_set_weight1_base_t(Structure):
    _fields_ = [
        ("cmd_code", c_uint32, 10),
        ("must_be_zero", c_uint32, 4),
        ("payload_size", c_uint32, 2),
        ("param", c_uint32, 16),
        ("data", c_uint32, 32),
    ]
    def valid(self): return cmd_code==cmd1.NPU_SET_WEIGHT1_BASE and must_be_zero==0 and payload_size>=1 and payload_size<=2;
    def get_cmd_code(self): return cmd_code
    def set_cmd_code(self, value): cmd_code = value
    def get_data(self): return data
    def set_data(self, value): data = value
    def get_param(self): return param
    def set_param(self, value): param = value
    def get_payload_size(self): return payload_size
    def set_payload_size(self, value): payload_size = value

class npu_set_weight1_length_t(Structure):
    _fields_ = [
        ("cmd_code", c_uint32, 10),
        ("must_be_zero", c_uint32, 4),
        ("payload_size", c_uint32, 2),
        ("reserved0", c_uint32, 16),
        ("data", c_uint32, 32),
    ]
    def valid(self): return cmd_code==cmd1.NPU_SET_WEIGHT1_LENGTH and must_be_zero==0 and payload_size>=1 and payload_size<=2;
    def get_cmd_code(self): return cmd_code
    def set_cmd_code(self, value): cmd_code = value
    def get_data(self): return data
    def set_data(self, value): data = value
    def get_payload_size(self): return payload_size
    def set_payload_size(self, value): payload_size = value

class npu_set_scale1_base_t(Structure):
    _fields_ = [
        ("cmd_code", c_uint32, 10),
        ("must_be_zero", c_uint32, 4),
        ("payload_size", c_uint32, 2),
        ("param", c_uint32, 16),
        ("data", c_uint32, 32),
    ]
    def valid(self): return cmd_code==cmd1.NPU_SET_SCALE1_BASE and must_be_zero==0 and payload_size>=1 and payload_size<=2;
    def get_cmd_code(self): return cmd_code
    def set_cmd_code(self, value): cmd_code = value
    def get_data(self): return data
    def set_data(self, value): data = value
    def get_param(self): return param
    def set_param(self, value): param = value
    def get_payload_size(self): return payload_size
    def set_payload_size(self, value): payload_size = value

class npu_set_scale1_length_t(Structure):
    _fields_ = [
        ("cmd_code", c_uint32, 10),
        ("must_be_zero", c_uint32, 4),
        ("payload_size", c_uint32, 2),
        ("reserved0", c_uint32, 16),
        ("data", c_uint32, 32),
    ]
    def valid(self): return cmd_code==cmd1.NPU_SET_SCALE1_LENGTH and must_be_zero==0 and payload_size>=1 and payload_size<=2;
    def get_cmd_code(self): return cmd_code
    def set_cmd_code(self, value): cmd_code = value
    def get_data(self): return data
    def set_data(self, value): data = value
    def get_payload_size(self): return payload_size
    def set_payload_size(self, value): payload_size = value

