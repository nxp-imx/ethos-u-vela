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
#include <getopt.h>
#include <stdarg.h>
#include "mlw_encode.h"
#include "mlw_decode.h"

#define UNCHECKED(call) (void)call

static void fatal_error(const char *format, ...) {
  va_list ap;
  va_start (ap, format);
  vfprintf(stderr, format, ap);
  va_end(ap);
  exit(1);
}

static void print_usage(void) {
    printf("Usage:\n");
    printf("    Encode: ./mlw_codec [<options>] [-o <outfile.mlw>] infiles.bin\n");
    printf("    Decode: ./mlw_codec [<options>] -d [-o <outfile.bin>] infiles.mlw\n");
    printf("\n");
    printf("Options:\n");
    printf("    -w      The uncompressed weight file is an int16_t (word) stream.\n");
    printf("            This is to support 9bit signed weights. Little endian is assuemd.\n");
    printf("            The default format is int8_t (byte) stream (if -w is not specified)\n");
    printf("\n");
}

// Read file into allocated buffer. Return length in bytes.
static size_t read_file( FILE *f, uint8_t **buf) {
    size_t rsize = 0;
    UNCHECKED(fseek(f, 0, SEEK_END));
    long size = ftell(f);
    size = size < 0 ? 0 : size;
    UNCHECKED(fseek(f, 0, SEEK_SET));
    *buf = malloc(size);
    if (*buf)
    {
        rsize = fread(*buf, 1, size, f);
        assert(rsize==size);
    }
    UNCHECKED(fclose(f));
    return rsize;
}


#define MAX_INFILES 1000

int main(int argc, char *argv[])
{
    int c, decode=0, inbuf_size, outbuf_size;
    char *infile_name[MAX_INFILES], *outfile_name=0;
    uint8_t *inbuf=0, *outbuf=0;
    FILE *infile, *outfile=0;
    int verbose=0, infile_idx=0;
    int int16_format=0;

    if (argc==1) {
        print_usage();
        exit(1);
    }

    // Parse command line options
    while( optind < argc) {
        // Parse options
        while ((c = getopt (argc, argv, "di:o:v:w?")) != -1) {
            switch (c) {
            case 'd':
                decode=1;
                break;
            case 'i':
                assert(infile_idx<MAX_INFILES);
                infile_name[infile_idx++]=optarg;
                break;
            case 'o':
                outfile_name=optarg;
                break;
            case 'v':
                verbose=atoi(optarg);
                break;
            case 'w':
                int16_format=1;
                break;
            case '?':
                print_usage();
                exit(0);
            }
        }

        if (optind<argc) {
            assert(infile_idx<MAX_INFILES);
            infile_name[infile_idx++]=argv[optind];
            optind++;

        }
    }

    if (outfile_name) {
        outfile=fopen(outfile_name, "wb");
        if (!outfile)
            fatal_error("ERROR: cannot open outfile %s\n", outfile_name);
    }

    // Loop over input files
    int nbr_of_infiles=infile_idx;
    for(infile_idx=0; infile_idx<nbr_of_infiles; infile_idx++) {
        infile=fopen(infile_name[infile_idx], "rb");
        if (!infile)
            fatal_error("ERROR: cannot open infile %s\n", infile_name[infile_idx]);

        // Read infile to buffer
        inbuf_size = read_file(infile, &inbuf);

        if (!decode) {
            // Encode
            int i, n = int16_format ? inbuf_size/(int)sizeof(int16_t) : inbuf_size;
            int16_t *weights = malloc( n * sizeof(int16_t) );
            for(i=0; i<n; i++) {
                weights[i] = int16_format ? ((int16_t*)inbuf)[i] : ((int8_t*)inbuf)[i];
            }
            outbuf_size = mlw_encode( weights, n, &outbuf, verbose);
            free(weights);
            printf("Input size %d output size %d bpw %4.2f\n", n, outbuf_size, outbuf_size*8.0/n);
        } else {
            // Decode
            int i, n;
            int16_t *weights;
            n = mlw_decode( inbuf, inbuf_size, &weights, verbose);
            outbuf_size = int16_format ? n*(int)sizeof(int16_t) : n;
            outbuf = malloc( outbuf_size );
            assert(outbuf);
            for(i=0; i<n; i++) {
                if (int16_format)
                    ((int16_t*)outbuf)[i] = weights[i];
                else
                    outbuf[i] = weights[i];
            }
            free(weights);
            printf("Input size %d output size %d bpw %4.2f\n", inbuf_size, n, n ? inbuf_size*8.0/n : 0);

        }

        if (outfile) {
            UNCHECKED(fwrite(outbuf, 1, outbuf_size, outfile));
        }

        if (inbuf)
            free(inbuf);
        if (outbuf)
            free(outbuf);
    }

    if (outfile) {
        UNCHECKED(fclose(outfile));
    }

    return 0;
}
