#include <stdio.h>
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

/* Partial transform for the input tile */

void p_transform(float input[4], float output[4]) {
#pragma HLS INLINE off
  float A0 = input[0];
  float A1 = input[1];
  float A2 = input[2];
  float A3 = input[3];

  output[0] = A0 - A2;
  output[1] = A1 + A2;
  output[2] = A2 - A1;
  output[3] = A1 - A3;
}

float out_trans_p(float in0, float in1, float in2) {
#pragma HLS INLINE off
  return in0 + in1 + in2;
}

float out_trans_m(float in0, float in1, float in2) {
#pragma HLS INLINE off
  return in0 - in1 - in2;
}

/* Output accumulation function, accumulates the output tiles into the 
 * appropriate output buffer based off of the bank. Note this function needs
 * inlining to be disabled or it is synthesized incorrectly */

void acc_out(float16 *outbuf, float ot[4][2], int bank_off, int offset) {
#pragma HLS INLINE off
  if (bank_off == 0) {
    outbuf[offset].s0 += ot[0][0];
    outbuf[offset].s1 += ot[0][1];
    outbuf[offset].s2 += ot[1][0];
    outbuf[offset].s3 += ot[1][1];
    outbuf[offset].s4 += ot[2][0];
    outbuf[offset].s5 += ot[2][1];
    outbuf[offset].s6 += ot[3][0];
    outbuf[offset].s7 += ot[3][1];
  } else {
    outbuf[offset].s8 += ot[0][0];
    outbuf[offset].s9 += ot[0][1];
    outbuf[offset].sa += ot[1][0];
    outbuf[offset].sb += ot[1][1];
    outbuf[offset].sc += ot[2][0];
    outbuf[offset].sd += ot[2][1];
    outbuf[offset].se += ot[3][0];
    outbuf[offset].sf += ot[3][1];
  }
}

/* Bias initialization function, sets the output to the given bias value */

void bias_init(float16 *outbuf, float bias, int offset) {
#pragma HLS INLINE 
  outbuf[offset].s0 = bias;
  outbuf[offset].s1 = bias;
  outbuf[offset].s2 = bias;
  outbuf[offset].s3 = bias;
  outbuf[offset].s4 = bias;
  outbuf[offset].s5 = bias;
  outbuf[offset].s6 = bias;
  outbuf[offset].s7 = bias;
  outbuf[offset].s8 = bias;
  outbuf[offset].s9 = bias;
  outbuf[offset].sa = bias;
  outbuf[offset].sb = bias;
  outbuf[offset].sc = bias;
  outbuf[offset].sd = bias;
  outbuf[offset].se = bias;
  outbuf[offset].sf = bias;
}

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

void winograd_pe(float *input, float16 *weights, float *bias, float16 *output, 
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
  float inbuf[128 * 128][4][2]; 
#pragma HLS ARRAY_PARTITION variable=inbuf complete dim=3
#pragma HLS ARRAY_PARTITION variable=inbuf complete dim=2
#pragma HLS ARRAY_PARTITION variable=inbuf cyclic factor=4 dim=1

  // Buffer for holding extra data in an edge case where xtile==xtile_pad
  float sidebuf[2048][4];
#pragma HLS ARRAY_PARTITION variable=sidebuf complete dim=2

  // Registers used for loading data from sidebuf
  float sidebuf_val[4][2];
#pragma HLS ARRAY_PARTITION variable=sidebuf_val complete dim=2
#pragma HLS ARRAY_PARTITION variable=sidebuf_val complete dim=1

  // Output buffer used for writing
  float16 outbuf[256 * 16];
  // Temporary output buffer to improve throughput during mult-acc stage
  float16 tempout[128 * 16];
  float16 tempout2[128 * 16];
  // Buffer used for reading lines from the input
  float line[260] = {0};
#pragma HLS ARRAY_PARTITION variable=line cyclic factor=2

  // Weight buffer
  float16 wbuf[256];

  // Bias buffer
  float biasbuf[1024];

  // Input tile column register
  float itcol[4];
#pragma HLS ARRAY_PARTITION variable=itcol complete
  // Column register post p_transform
  float otcol[4];
#pragma HLS ARRAY_PARTITION variable=otcol complete

  // Temporary input tile registers
  float itt[4][4][4];
#pragma HLS ARRAY_PARTITION variable=itt complete dim=1
#pragma HLS ARRAY_PARTITION variable=itt complete dim=2
#pragma HLS ARRAY_PARTITION variable=itt complete dim=3

  // Input tile registers post transform
  float it[4][4][4];
#pragma HLS ARRAY_PARTITION variable=it complete dim=1
#pragma HLS ARRAY_PARTITION variable=it complete dim=2
#pragma HLS ARRAY_PARTITION variable=it complete dim=3

  // Temporary output tile registers
  float ott[4][4][4];
#pragma HLS ARRAY_PARTITION variable=ott complete dim=1
#pragma HLS ARRAY_PARTITION variable=ott complete dim=2
#pragma HLS ARRAY_PARTITION variable=ott complete dim=3

  // Ouput tile transform stage 1 output
  float ot_s1[4][8];
#pragma HLS ARRAY_PARTITION variable=ot_s1 complete dim=1
#pragma HLS ARRAY_PARTITION variable=ot_s1 complete dim=2

  // Output tile transform stage 2 output
  float ot_l[4][2];
#pragma HLS ARRAY_PARTITION variable=ot_l complete dim=1
#pragma HLS ARRAY_PARTITION variable=ot_l complete dim=2 

  float ot_u[4][2];
#pragma HLS ARRAY_PARTITION variable=ot_u complete dim=1
#pragma HLS ARRAY_PARTITION variable=ot_u complete dim=2 

  float colbuf_in[5][2][4];
#pragma HLS ARRAY_PARTITION variable=colbuf_in complete dim=1
#pragma HLS ARRAY_PARTITION variable=colbuf_in complete dim=2
#pragma HLS ARRAY_PARTITION variable=colbuf_in complete dim=3

  float colbuf_out[5][2][4];
#pragma HLS ARRAY_PARTITION variable=colbuf_out complete dim=1
#pragma HLS ARRAY_PARTITION variable=colbuf_out complete dim=2
#pragma HLS ARRAY_PARTITION variable=colbuf_out complete dim=3

  int bank_off;

  assert(rburst >= 12);
  assert(rburst <= 4096);
  assert(inchannels >= 1);
  assert(inchannels <= 1024);
  assert(outchannels >= 1);
  assert(outchannels <= 1024);

  assert(burstchannels >= 1);
  assert(burstchannels <= 256);

  assert(xdim >= 7);
  assert(xdim <= 256);
  assert(ydim >= 7);
  assert(ydim <= 256);
  assert(xtile >= 4);
  assert(xtile <= 128);
  assert(ytile >= 4);
  assert(ytile <= 128);

  assert(group >= 0);
  assert(group <= 1);

  assert(numgroups <= 2);
  assert(numgroups >= 1);

  assert(xtile_pad >= 8);
  assert(xtile_pad <= 128);

  assert(rpo >= 1);
  assert(rpo <= 64);

  int i;

  int n, y, x, p, q, j, o;

  unsigned short sb_off = 0;
  unsigned short w_off = 0;
  int tile_cnt = 0;
  unsigned short xt_off = 0;
  unsigned short yt_off = 0;
  int fact = xtile_pad >> 3;
  int out_offset = 0;  
  unsigned short y_off = 0;
  unsigned short p_off = 1;
  unsigned short i_off = 0;

  int offset1 = 0;
  int offset2 = 0;

  int offtemp1 = 0;
  int offtemp2 = 0;

  float lineval;

  int id_iny, id_in, id_inx;

  /* Read bias data into buffer */
  memcpy(biasbuf, bias + outchannels * group, sizeof(float) * outchannels);

  for (n = 0; n < rpo; ++n) {
    y_off = 0;
    p_off = 1;
    i_off = 0;

    /* Read the input line by line and tile it into the tile buffer */

    IREADLOOP: for (i = 0; i < rburst; ++i) {
      memcpy(line + 1, input + ((((dataoff * numgroups + group) * inchannels +
          n * burstchannels) * ydim + i) * xdim), sizeof(float) * xdim);

      if (y_off * 2 + (p_off - 1) == ydim) {
        i_off++;
        y_off = 0;
        p_off = 1;
      }

      if (p_off == 4) {
        p_off = 2;
        y_off++;
      }
      id_iny = i_off * ytile + y_off;
      id_in = (id_iny) * xtile_pad;

      XTILELOOP: for (x = 0; x < xtile_pad; ++x) {
#pragma HLS pipeline
        for (p = 0; p < 2; ++p) {
          lineval = line[x * 2 + p];
          id_inx = id_in + x;
          inbuf[id_inx][p_off][p] = lineval;
          if (p_off >= 2 && (y_off + 1) < ytile) {
            inbuf[id_inx + xtile_pad][p_off - 2][p] = lineval;
          }
        } 
      }
      lineval = line[xtile_pad * 2];
      sidebuf[id_iny][p_off] = lineval;
      if (p_off >= 2 && (y_off + 1) < ytile) {
        sidebuf[id_iny + 1][p_off - 2] = lineval;
      } 
      p_off++;
    }
    
    for (o = 0; o < outchannels; ++o) {
      memcpy(wbuf, weights + (o + outchannels * group) * inchannels + 
          n * burstchannels, sizeof(float16) * burstchannels);
      out_offset = dataoff * numgroups * outchannels * ydim * fact + 
                    ((o + outchannels * group) * ydim) * fact;

      if (n == 0) {
        /* Set the output buffers to contain the biases */
        for (y = 0; y < ytile; ++y) {
          for (x = 0; x < fact; ++x) {
#pragma HLS pipeline
            offtemp1 = y * 2 * fact + x;
            offtemp2 = y * fact + x;
            bias_init(outbuf, biasbuf[o], offtemp1);
            bias_init(tempout, biasbuf[o], offtemp2);
          }
        } 
      } else {
        memcpy(outbuf, output + out_offset, sizeof(float16) * fact * ydim);
        for (y = 0; y < ytile; ++y) {
          for (x = 0; x < fact; ++x) {
#pragma HLS pipeline
            offtemp1 = (y * 2 + 1) * fact + x;
            offtemp2 = y * fact + x;
            tempout[offtemp2] = outbuf[offtemp1];
          }
        }
      }

      w_off = 0;
      tile_cnt = 0;
      xt_off = 0;
      yt_off = 0;
      bank_off = 0;
      sb_off = 0;
      MULTACCSTAGE: for (i = 0; i < (burstchannels * ytile * xtile_pad) >> 2; 
                        ++i, ++tile_cnt, ++xt_off) {
#pragma HLS DEPENDENCE variable=outbuf inter distance=12 true
#pragma HLS DEPENDENCE variable=tempout inter distance=12 true
#pragma HLS pipeline
        if (tile_cnt * 4 == ytile * xtile_pad) {
          tile_cnt = 0;
          yt_off = 0;
          xt_off = 0;
          w_off++;
          sb_off++;
        } 
        if (xt_off * 4 == xtile_pad) {
          sb_off++;
          yt_off++;
          xt_off = 0;
        }

        for (p = 0; p < 4; ++p) {
          sidebuf_val[p][0] = sidebuf[sb_off][p];
          sidebuf_val[p][1] = 0;
        }
        for (j = 0; j < 5; ++j) {
          for (q = 0; q < 2; ++q) {
            for (p = 0; p < 4; ++p) {
              if (j == 4 && (xt_off * 4 + j == xtile_pad)) {
                colbuf_in[j][q][p] = sidebuf_val[p][q];
              }
              else {
                colbuf_in[j][q][p] = inbuf[i * 4 + j][p][q];
              }
            }
            p_transform(colbuf_in[j][q], colbuf_out[j][q]);
          }
        }
        /* Complete the remaining transform steps */
        for (j = 0; j < 4; ++j) {
          for (p = 0; p < 4; ++p) {
            for (q = 0; q < 2; ++q) {
              itt[j][p][q] = colbuf_out[j][q][p];
              itt[j][p][q + 2] = colbuf_out[j + 1][q][p];
            }
          }
        }
        for (j = 0; j < 4; ++j) {
          for (p = 0; p < 4; ++p) {
            p_transform(itt[j][p], it[j][p]);
          }
         
          /* Compute the element-wise multiplication between weight and input
           * tile */

          ott[j][0][0] = it[j][0][0] * wbuf[w_off].s0;
          ott[j][0][1] = it[j][0][1] * wbuf[w_off].s1;
          ott[j][0][2] = it[j][0][2] * wbuf[w_off].s2;
          ott[j][0][3] = it[j][0][3] * wbuf[w_off].s3;

          ott[j][1][0] = it[j][1][0] * wbuf[w_off].s4;
          ott[j][1][1] = it[j][1][1] * wbuf[w_off].s5;
          ott[j][1][2] = it[j][1][2] * wbuf[w_off].s6;
          ott[j][1][3] = it[j][1][3] * wbuf[w_off].s7;

          ott[j][2][0] = it[j][2][0] * wbuf[w_off].s8;
          ott[j][2][1] = it[j][2][1] * wbuf[w_off].s9;
          ott[j][2][2] = it[j][2][2] * wbuf[w_off].sa;
          ott[j][2][3] = it[j][2][3] * wbuf[w_off].sb;

          ott[j][3][0] = it[j][3][0] * wbuf[w_off].sc;
          ott[j][3][1] = it[j][3][1] * wbuf[w_off].sd;
          ott[j][3][2] = it[j][3][2] * wbuf[w_off].se;
          ott[j][3][3] = it[j][3][3] * wbuf[w_off].sf;

           /* Transform the output */
         
          for (p = 0; p < 4; ++p) {
            ot_s1[j][p * 2] = out_trans_p(ott[j][p][0], ott[j][p][1], 
                ott[j][p][2]);
            ot_s1[j][p * 2 + 1] = out_trans_m(ott[j][p][1], 
                ott[j][p][2], ott[j][p][3]);
          }

          for (p = 0; p < 2; ++p) {
            ot_l[j][p] = out_trans_p(ot_s1[j][p], ot_s1[j][p + 2], 
                ot_s1[j][p + 4]);
            ot_u[j][p] = out_trans_m(ot_s1[j][p + 2], ot_s1[j][p + 4], 
                ot_s1[j][p + 6]);
          }
        }

        offset1 = yt_off * 2 * fact + (xt_off >> 1);
        offset2 = yt_off * fact + (xt_off >> 1);
       
        acc_out(outbuf, ot_l, bank_off, offset1);
        acc_out(tempout, ot_u, bank_off, offset2);
        
        bank_off = ~bank_off;
      }
      for (y = 0; y < ytile; ++y) {
        for (x = 0; x < fact; ++x) {
#pragma HLS pipeline
          offtemp1 = (y * 2 + 1) * fact + x;
          offtemp2 = y * fact + x;
          outbuf[offtemp1] = tempout[offtemp2];
        }
      }
      memcpy(output + out_offset, outbuf, sizeof(float16) * fact * ydim);
    }
  }
}
