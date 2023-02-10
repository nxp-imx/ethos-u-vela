/*
 * SPDX-FileCopyrightText: Copyright 2020, 2022 Arm Limited and/or its affiliates <open-source-office@arm.com>
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

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <stdbool.h>
#include <string.h>
#include <assert.h>
#include <math.h>
#include <stdarg.h>
#include <math.h>
#include "mlw_common.h"
#include "mlw_decode.h"

#define CHECKED_MALLOC(var, size) { if ( !(var = malloc(size)) ) break; }

/////////////////////////////// Read from bitstream

typedef struct bitbuf {
    uint8_t *buf;
    int buf_size;               // in bytes
    int pos;                    // bit pos of next bit
    int log_symbols;
} bitbuf_t;


// size in byte
static void bitbuf_init( bitbuf_t *bb, uint8_t *buf, int size, int log_symbols) {
    bb->buf  = buf;
    bb->pos  = 0;
    bb->buf_size = size;
    bb->log_symbols = log_symbols;
}

static int bitbuf_getbit( bitbuf_t *bb) {
    int byte_pos = bb->pos>>3;
    int bit_pos = bb->pos&7;
    if ( byte_pos < 0 || byte_pos >= bb->buf_size ) {
        printf("bitbuf_getbit: underrun, bit_pos %3d byte_pos %3d buf_size %3d\n", bit_pos, byte_pos, bb->buf_size);
        exit(1);
    }
    int bit = bb->buf[ byte_pos ] & (1<<bit_pos) ? 1 : 0;
    bb->pos++;
    return bit;
}

static int bitbuf_get( bitbuf_t *bb, const char *name, int len) {
    int i, data=0, save_pos=bb->pos;
    if (len>0) {
        for(i=0; i<len; i++) {
            data |= bitbuf_getbit(bb)<<i;
        }
        if (bb->log_symbols)
            printf("bitbuf: pos %3d %7s len %d data %x\n", save_pos, name, len, data);
    }
    return data;
}

// Decode the given weight stream
//      inbuf       compressed bitstream
//      inbuf_size  size of compressed bitstream in bytes
//      outbuf      uncompressed 9bit signed weights, buffer malloced
//      verbose     if non-zero, printf log
// Return value is the number of uncompressed weights
int mlw_decode( uint8_t *inbuf, int inbuf_size, int16_t **outbuf, int verbose) {
    int nvalues;
    int w_grc_div;
    int w_grc_trunc;
    int w_uncompressed;
    int z_grc_div, z_prev_grc_div=0;
    int new_palette;
    int palsize=0, palbits=0;
    int direct_offset=0;
    int16_t palette[512];
    int first=1;
    int use_zero_run, i, j;
    int outbuf_size=0;
    int nchunks=0;

    *outbuf=0;

    bitbuf_t bitbuf_s, *bb=&bitbuf_s;
    bitbuf_init( bb, inbuf, inbuf_size, (verbose&2)?1:0 );

    int *w_value = NULL;
    int *z_value = NULL;
    // Loop over all slices
    do {
        // Decode slice header
        z_grc_div = bitbuf_get( bb, "ZDIV", 3 );
        while(z_grc_div==ZDIV_EOS) {                    // TODO: change to ZDIV_PAD
            // End of stream
            // Byte align
            bitbuf_get( bb, "BYTEALIGN", (8-(bb->pos&7))&7 );
            first=1;
            if ( (bb->pos/8) == inbuf_size) {
                // Quit if we actually reached end of input stream
                break;
            }
            z_grc_div = bitbuf_get( bb, "ZDIV", 3 );
        }
        if ( (bb->pos/8) == inbuf_size) {
            break;  // reached end of input stream
        }
        assert(z_grc_div<4 || z_grc_div==ZDIV_DISABLE);
        use_zero_run = z_grc_div!=ZDIV_DISABLE;    // alternating grc
        nvalues = bitbuf_get( bb, "SLICELEN", 15 )+1;
        w_grc_div = bitbuf_get( bb, "WDIV", 3 );
        w_grc_trunc = bitbuf_get( bb, "WTRUNC", 1 );
        new_palette = bitbuf_get( bb, "NEWPAL", 1 );
        if (first) {
            // the first slice must have a palette/direct mode setup
            assert(new_palette);
            first=0;
        }
        if (!new_palette) {
            // At the moment it is not supported to change between alternating
            // and non-alternating without redefining the palette (this is because
            // the zero is not included in the palette in case of alternating)
            int prev_use_zero_run = z_prev_grc_div!=ZDIV_DISABLE;
            (void)(prev_use_zero_run);
            assert( use_zero_run == prev_use_zero_run);
        }
        z_prev_grc_div = z_grc_div;
        if (new_palette) {
            direct_offset = bitbuf_get( bb, "DIROFS", 5 );
            palsize = bitbuf_get( bb, "PALSIZE", 5 );
            if (palsize>0)
                palsize++;
            palbits = bitbuf_get( bb, "PALBITS", 3 )+2;
            for(i=0; i<palsize; i++) {
                palette[i] = bitbuf_get( bb, "PALETTE", palbits );
            }
        }

        if (w_grc_div==WDIV_UNCOMPRESSED) {
            // Uncompressed mode
            w_uncompressed = 1;
            int uncompressed_bits;
            if (palsize>0) {
                // Uncompressed bits is given by palette size.
                uncompressed_bits=0;
                while( (1<<uncompressed_bits) < palsize )
                    uncompressed_bits++;
            } else {
                // No palette. PALBITS is used to specify uncompressed bits.
                uncompressed_bits=palbits;
            }
            // In uncompressed mode there's only a remainder part (no unary)
            // This is achieved by setting w_grc_div to index bit width
            w_grc_div = uncompressed_bits;
        } else {
            w_uncompressed = 0;
            assert(w_grc_div<6);
        }

        // Decode the slice
        int z_nvalues = nvalues + (new_palette?1:0);
        CHECKED_MALLOC( w_value, nvalues*sizeof(int) );
        CHECKED_MALLOC( z_value, z_nvalues*sizeof(int) );
        z_value[0] = 0;
        int w_pos=0, z_pos=0;
        int w_prev_pos=0, z_prev_pos=0;
        int w_unary0=0, w_unary1=0, w_unary1_len=0, w_q[12]={0}, w_carry=0;
        int z_unary=0, z_q[12]={0}, z_carry=0;
        int w_nsymbols=0;
        int w_prev_enable=0, w_prev_nsymbols=0, w_prev_q[12]={0};
        int z_nsymbols=0;
        int z_prev_enable=0, z_prev_nsymbols=0, z_prev_q[12]={0};
        int total_zcnt=0;
        int z_unary_len = z_grc_div<3 ? 12 : 8;

        // Loop over all chunks in the slice
        do {
            // Flow control to possibly throttle either the weights or zero-runs
            int balance = use_zero_run ? w_pos - z_pos : 0;
            int w_enable = (balance<8 || !use_zero_run) && w_pos<nvalues;
            int z_enable = balance>=0 && use_zero_run && z_pos<z_nvalues;
            if (w_enable) {
                if (!w_uncompressed)
                    w_unary0 = bitbuf_get( bb, "WUNARY0", 12 );
                else
                    w_unary0 = 0;
            }
            if (z_enable) {
                z_unary = bitbuf_get( bb, "ZUNARY", z_unary_len );
                z_nsymbols=0;
                int cnt = z_carry;
                for(i=0; i<z_unary_len; i++) {
                    if (z_unary & (1<<i)) {
                        cnt++;
                    } else {
                        z_q[z_nsymbols++] = cnt;
                        cnt=0;
                    }
                }
                z_carry = cnt;
                z_pos += z_nsymbols;
            }
            if (w_enable) {
                w_unary1_len=0;
                int max_symbols = w_uncompressed && w_grc_div>5 ? 8 : 12;
                for(i=0; i<max_symbols; i++) {
                    if (w_unary0&(1<<i))
                        w_unary1_len++;
                }
                w_unary1 = bitbuf_get( bb, "WUNARY1", w_unary1_len );
                w_nsymbols=0;
                int cnt = w_carry;
                for(i=0; i<max_symbols; i++) {
                    int code=0;
                    if (w_unary0 & (1<<i)) {
                        code++;
                        if (w_unary1&1) {
                            code++;
                        }
                        w_unary1 = w_unary1>>1;
                    }
                    cnt += code;
                    if (code<2 || w_grc_trunc) {
                        w_q[w_nsymbols++] = cnt;
                        cnt=0;
                    }
                }
                w_carry = cnt;
                w_pos += w_nsymbols;
            }
            if (w_prev_enable) {
                for(i=0; i<w_prev_nsymbols && w_prev_pos<nvalues; i++, w_prev_pos++) {
                    int remain = bitbuf_get( bb, "WREMAIN", w_grc_div );
                    w_value[w_prev_pos] = (w_prev_q[i]<<w_grc_div) + remain;
                }
            }
            if (z_prev_enable) {
                for(i=0; i<z_prev_nsymbols && z_prev_pos<z_nvalues; i++, z_prev_pos++) {
                    int remain = bitbuf_get( bb, "ZREMAIN", z_grc_div );
                    z_value[z_prev_pos] = (z_prev_q[i]<<z_grc_div) + remain;
                    total_zcnt += z_value[z_prev_pos];
                }
            }
            w_prev_enable = w_enable;
            w_prev_nsymbols = w_nsymbols;
            memcpy( w_prev_q, w_q, sizeof(w_prev_q));
            z_prev_enable = z_enable;
            z_prev_nsymbols = z_nsymbols;
            memcpy( z_prev_q, z_q, sizeof(z_prev_q));
            nchunks++;
        } while( w_prev_enable || z_prev_enable );

        // Interleave non-zero and zeros into the outbut buffer
        // Increase the outbuffer to fit the new slice
        *outbuf = realloc( *outbuf, (outbuf_size + nvalues + total_zcnt)*sizeof(int16_t));
        if (*outbuf)
        {
            int k=outbuf_size;

            // Insert initial zeros
            // (slices redefining the palette may start with zeros)
            if (new_palette && use_zero_run) {
                for(j=0; j<z_value[0]; j++) {
                    (*outbuf)[k++] = 0;
                }
            }

            // Loop over all weights and insert zeros in-between
            for(i=0; i<nvalues; i++) {
                int val;
                assert(w_value[i]<512); // HW supports 9bit
                if (w_value[i]<palsize) {
                    val = palette[w_value[i]];
                } else {
                    val = w_value[i]-palsize+direct_offset;
                }
                int sign = val&1;
                int mag  = val>>1;
                (*outbuf)[k++] = sign ? -mag : mag;
                if (use_zero_run) {
                    for(j=0; j<z_value[i+(new_palette?1:0)]; j++) {
                        (*outbuf)[k++] = 0;
                    }
                }
            }

            outbuf_size = k;
        } else {
            outbuf_size = 0;
        }
        free(w_value);
        free(z_value);
        w_value = NULL;
        z_value = NULL;
    } while(*outbuf);

    free(w_value);
    free(z_value);

    return outbuf_size;
}
