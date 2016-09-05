#include <stdio.h>
#include <string.h>
#include <assert.h>

/* float16 data type definition */

typedef struct {
  float s0;
  float s1;
  float s2;
  float s3;
  float s4;
  float s5;
  float s6;
  float s7;
  float s8;
  float s9;
  float sa;
  float sb;
  float sc;
  float sd;
  float se;
  float sf;
} float16;

/* Kernel used for computing Winograd (F(3x3, 2x2)) based convolution. This 
 * kernel assumes that the weights have been pre-transformed. 
 * input:         flattened input array containing image data
 * weights:       pre-transformed 3x3 filters
 * bias:          flattened bias array
 * output:        output of the convolution, padded to be divisible by 16 on 
 *                the x dimension
 * group:         group index, leave as 0 if not using group convolution
 * inchannels:    number of input channels
 * outchannels:   number of output channels
 * burstchannels: number of input channels to be handled at once
 * rpo:           number of reads required to cover all input channels
 * ydim:          size in the  y dimension
 * xdim:          size in the x dimension
 * ytile:         number of rows of tiles
 * xtile:         number of columns of tiles
 * ytile_pad:     padded number of rows of tiles (not used)
 * xtile_pad:     padded number of columns of tiles
 * rburst:        number of input rows to read
 * dataoff:       image offset
 * numgroups:     number of groups
 */ 

void direct_conv(float *input, float16 *weights, float *bias, float16 *output, 
      int group, int inchannels, int outchannels, int burstchannels, int rpo,
      int ydim, int xdim, int ytile, int xtile, int ytile_pad, int xtile_pad, 
      int rburst, int dataoff, int numgroups) {

/* Ports */

#pragma HLS data_pack variable=weights
#pragma HLS data_pack variable=output
#pragma HLS INTERFACE m_axi port=input offset=slave bundle=gmem1
#pragma HLS INTERFACE m_axi port=output offset=slave bundle=gmem2
#pragma HLS INTERFACE m_axi port=weights offset=slave bundle=gmem3
#pragma HLS INTERFACE m_axi port=bias offset=slave bundle=gmem4
#pragma HLS INTERFACE s_axilite port=input bundle=control
#pragma HLS INTERFACE s_axilite port=output bundle=control
#pragma HLS INTERFACE s_axilite port=weights bundle=control
#pragma HLS INTERFACE s_axilite port=bias bundle=control

#pragma HLS INTERFACE s_axilite port=group bundle=control
#pragma HLS INTERFACE s_axilite port=inchannels bundle=control
#pragma HLS INTERFACE s_axilite port=outchannels bundle=control
#pragma HLS INTERFACE s_axilite port=burstchannels bundle=control
#pragma HLS INTERFACE s_axilite port=rpo bundle=control
#pragma HLS INTERFACE s_axilite port=rburst bundle=control
#pragma HLS INTERFACE s_axilite port=dataoff bundle=control
#pragma HLS INTERFACE s_axilite port=numgroups bundle=control

#pragma HLS INTERFACE s_axilite port=ydim bundle=control
#pragma HLS INTERFACE s_axilite port=xdim bundle=control
#pragma HLS INTERFACE s_axilite port=xtile bundle=control
#pragma HLS INTERFACE s_axilite port=ytile bundle=control
#pragma HLS INTERFACE s_axilite port=xtile_pad bundle=control
#pragma HLS INTERFACE s_axilite port=ytile_pad bundle=control
#pragma HLS INTERFACE s_axilite port=rburst bundle=control
#pragma HLS INTERFACE s_axilite port=return bundle=control

  // Input tile buffer
  float inbuf[256 * 256][3]; 
//#pragma HLS ARRAY_PARTITION variable=inbuf complete dim=3
#pragma HLS ARRAY_PARTITION variable=inbuf complete dim=2
#pragma HLS ARRAY_PARTITION variable=inbuf cyclic factor=16 dim=1

  float sidebuf[4096][3];
#pragma HLS ARRAY_PARTITION variable=sidebuf complete dim=2
 

  // Output buffer used for writing
  float16 outbuf[256 * 16];

  // Buffer used for reading lines from the input
  float line[260] = {0};
#pragma HLS ARRAY_PARTITION variable=line cyclic factor=2

  // Weight buffer
  float16 wbuf[256];

  // Bias buffer
  float biasbuf[1024];

  float inrow[3];
#pragma HLS ARRAY_PARTITION variable=inrow complete

  float it[16][3 * 3];
#pragma HLS ARRAY_PARTITION variable=it complete dim=1
#pragma HLS ARRAY_PARTITION variable=it complete dim=2

  // Temporary output tile registers
  float ott[16][10];
#pragma HLS ARRAY_PARTITION variable=ott complete dim=1
#pragma HLS ARRAY_PARTITION variable=ott complete dim=2

  float ot[16];
#pragma HLS ARRAY_PARTITION variable=ot complete dim=1

  int bank_off;

  assert(rburst >= 1);
  assert(rburst <= 4096);
  assert(inchannels >= 1);
  assert(inchannels <= 1024);
  assert(outchannels >= 1);
  assert(outchannels <= 1024);

  assert(burstchannels >= 1);
  assert(burstchannels <= 256);

  assert(xdim >= 12);
  assert(xdim <= 256);
  assert(ydim >= 12);
  assert(ydim <= 256);
  assert(xtile >= 12);
  assert(xtile <= 256);
  assert(ytile >= 12);
  assert(ytile <= 256);

  assert(group >= 0);
  assert(group <= 1);

  assert(numgroups <= 2);
  assert(numgroups >= 1);

  assert(xtile_pad >= 16);
  assert(xtile_pad <= 256);
  assert(ytile_pad >= 16);
  assert(ytile_pad <= 256);

  assert(rpo >= 1);
  assert(rpo <= 64);

  int i;

  unsigned int n, y, x, p, q, j, o;

  unsigned int sb_off = 0;
  unsigned int w_off = 0;
  unsigned int tile_cnt = 0;
  unsigned int xt_off = 0;
  unsigned int yt_off = 0;
  unsigned int fact = 0;
  int out_offset = 0;  
  unsigned int y_off = 0;
  unsigned int p_off = 1;
  unsigned int i_off = 0;

  unsigned int offset = 0;
  unsigned int offset2 = 0;

  unsigned int offtemp1 = 0;
  unsigned int offtemp2 = 0;

  unsigned int off;

  fact = ((xtile_pad) >> 4);

  /* Read bias data into buffer */
  memcpy(biasbuf, bias + outchannels * group, sizeof(float) * outchannels);

  for (n = 0; n < rpo; ++n) {
    y_off = 0;
    p_off = 1;
    i_off = 0;

    /* Read the input line by line and tile it into the tile buffer */
    IREADLOOP: for (i = 0; i < rburst; ++i) {
      memcpy(line + 1, input + dataoff * numgroups * inchannels * ydim * xdim + 
          i * xdim + group * inchannels * ydim * xdim + 
          n * burstchannels * ydim * xdim, sizeof(float) * xdim);

      if (y_off == ydim - 2) {
        i_off++;
        y_off = 0;
        p_off = 1;
      }

      if (p_off == 3) {
        p_off = 2;
        y_off++;
      }

      int id_in = (i_off * ytile + y_off) * xtile_pad;

      XTILELOOP: for (x = 0; x < xtile_pad; ++x) {
#pragma HLS pipeline

        for (p = 0; p < 1; ++p) {
          inrow[p] = line[x + p];
        }

        for (p = 0; p < 1; ++p) {
          float temp = inrow[p];
          int id_inx = id_in + x;

          inbuf[id_inx][p_off] = temp;
          if (y_off == ydim - 2) {
            inbuf[id_inx + xtile_pad][2] = 0;
          }

          if (p_off >= 1) {
            id_inx = id_inx + xtile_pad;
            inbuf[id_inx][p_off - 1] = temp;
          }
          if (p_off >= 2 && y_off != ydim - 2) {
            id_inx = id_inx + xtile_pad;
            inbuf[id_inx][p_off - 2] = temp;
          } 
        }
      }
      sidebuf[i_off * ytile + y_off][p_off] = line[xtile_pad];
      if (p_off >= 1) {
        sidebuf[i_off * ytile + y_off + 1][p_off - 1] = line[xtile_pad];
      }
      if (p_off >= 2 && y_off != ydim - 2)
        sidebuf[i_off * ytile + y_off + 2][p_off - 2] = line[xtile_pad];
      if (y_off == ydim - 2) {
        sidebuf[i_off * ytile + y_off + 1][2] = 0;
      }
      p_off++;
    }

    for (o = 0; o < outchannels; ++o) {
      memcpy(wbuf, weights + (o + outchannels * group) * inchannels + 
          n * burstchannels, sizeof(float16) * burstchannels);
      out_offset = dataoff * numgroups * outchannels * ydim * fact + 
                    ((o + outchannels * group) * ydim) * fact;

      if (n == 0) {
//      Set the output buffers to contain the biases 
        for (y = 0; y < ydim; ++y) {
          for (x = 0; x < fact; ++x) {
#pragma HLS pipeline
            offtemp1 = y * fact + x;
            outbuf[offtemp1].s0 = biasbuf[o];
            outbuf[offtemp1].s1 = biasbuf[o];
            outbuf[offtemp1].s2 = biasbuf[o];
            outbuf[offtemp1].s3 = biasbuf[o];
            outbuf[offtemp1].s4 = biasbuf[o];
            outbuf[offtemp1].s5 = biasbuf[o];
            outbuf[offtemp1].s6 = biasbuf[o];
            outbuf[offtemp1].s7 = biasbuf[o];
            outbuf[offtemp1].s8 = biasbuf[o];
            outbuf[offtemp1].s9 = biasbuf[o];
            outbuf[offtemp1].sa = biasbuf[o];
            outbuf[offtemp1].sb = biasbuf[o];
            outbuf[offtemp1].sc = biasbuf[o];
            outbuf[offtemp1].sd = biasbuf[o];
            outbuf[offtemp1].se = biasbuf[o];
            outbuf[offtemp1].sf = biasbuf[o];
          }
        } 
      } else {
        memcpy(outbuf, output + out_offset, sizeof(float16) * fact * ydim);
      }

      w_off = 0;
      tile_cnt = 0;
      xt_off = 0;
      yt_off = 0;
      sb_off = 0;
      MULTACCSTAGE: for (i = 0; i < (burstchannels * ydim * xtile_pad) >> 4; 
                        ++i, ++tile_cnt, ++xt_off) {
#pragma HLS DEPENDENCE variable=outbuf inter true distance=12
#pragma HLS DEPENDENCE variable=inbuf intra false
#pragma HLS pipeline 
        if (tile_cnt * 16 == ydim * xtile_pad) {
          tile_cnt = 0;
          yt_off = 0;
          xt_off = 0;
          w_off++;
          sb_off++;
        } else if (xt_off * 16 == xtile_pad) {
          yt_off++;
          xt_off = 0;
          sb_off++;
        }

        for (j = 0; j < 16; ++j) {
          for (p = 0; p < 3; ++p) {
            it[j][p * 3 + 0] = inbuf[i * 16 + j][p];
            it[j][p * 3 + 1] = inbuf[i * 16 + j + 1][p];
            it[j][p * 3 + 2] = inbuf[i * 16 + j + 2][p];

            if (j == 14 && (xt_off * 16 + j + 1 == xtile_pad - 1)) {
              it[j][p * 3 + 2] = sidebuf[sb_off][p];
            }

            if (j == 15 && (xt_off * 16 + j == xtile_pad - 1)) {
              it[j][p * 3 + 1] = sidebuf[sb_off][p];
              it[j][p * 3 + 2] = 0;
            }
          }
        }

        for (j = 0; j < 16; ++j) {
          ott[j][0] = it[j][0] * wbuf[w_off].s0;
          ott[j][1] = it[j][1] * wbuf[w_off].s1;
          ott[j][2] = it[j][2] * wbuf[w_off].s2;

          ott[j][3] = it[j][3] * wbuf[w_off].s3;
          ott[j][4] = it[j][4] * wbuf[w_off].s4;
          ott[j][5] = it[j][5] * wbuf[w_off].s5;
      
          ott[j][6] = it[j][6] * wbuf[w_off].s6;
          ott[j][7] = it[j][7] * wbuf[w_off].s7;
          ott[j][8] = it[j][8] * wbuf[w_off].s8;
        }

        offset = yt_off * fact + xt_off;

        for (j = 0; j < 16; ++j) {
          ot[j] = ott[j][0];
          for (p = 1; p < 9; ++p) {
            ot[j] += ott[j][p];
          }
        }
        outbuf[offset].s0 += ot[0];
        outbuf[offset].s1 += ot[1];
        outbuf[offset].s2 += ot[2];
        outbuf[offset].s3 += ot[3];
        outbuf[offset].s4 += ot[4];
        outbuf[offset].s5 += ot[5];
        outbuf[offset].s6 += ot[6];
        outbuf[offset].s7 += ot[7];
        outbuf[offset].s8 += ot[8];
        outbuf[offset].s9 += ot[9];
        outbuf[offset].sa += ot[10];
        outbuf[offset].sb += ot[11];
        outbuf[offset].sc += ot[12];
        outbuf[offset].sd += ot[13];
        outbuf[offset].se += ot[14];
        outbuf[offset].sf += ot[15]; 
      }

      memcpy(output + out_offset, outbuf, sizeof(float16) * fact * ydim);
    }
  }
}
