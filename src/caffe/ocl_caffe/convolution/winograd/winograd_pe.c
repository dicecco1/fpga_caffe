#include <stdio.h>
#include <string.h>
#include <assert.h>

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

/* First stage of the output tile transform */

void transform_outtile_s1(float output_t[16], float output[8]) {
#pragma HLS INLINE off
  output[0] = output_t[0] + output_t[1] + output_t[2];
  output[1] = output_t[1] - output_t[2] - output_t[3];
  output[2] = output_t[4] + output_t[5] + output_t[6];
  output[3] = output_t[5] - output_t[6] - output_t[7];
  output[4] = output_t[8] + output_t[9] + output_t[10];
  output[5] = output_t[9] - output_t[10] - output_t[11];
  output[6] = output_t[12] + output_t[13] + output_t[14];
  output[7] = output_t[13] - output_t[14] - output_t[15];
}

/* Second stage of the output tile transform */

void transform_outtile_s2(float input[8], float output[4]) {
#pragma HLS INLINE off
  output[0] = input[0] + input[2] + input[4];
  output[1] = input[1] + input[3] + input[5];
  output[2] = input[2] - input[4] - input[6];
  output[3] = input[3] - input[5] - input[7];
}

typedef struct {
  float val0;
  float val1;
  float val2;
  float val3;
  float val4;
  float val5;
  float val6;
  float val7;
  float val8;
  float val9;
  float val10;
  float val11;
  float val12;
  float val13;
  float val14;
  float val15;
} floatv16;

/* Output accumulation function, accumulates the output tiles into the 
 * appropriate output buffer based off of the bank. Note this function needs
 * inlining to be disabled or it is synthesized incorrectly */

void acc_out(floatv16 *outbuf, floatv16 *tempout, float ot[4][4], int bank_off, 
    int offset, int offset2) {
#pragma HLS INLINE off
  if (bank_off == 0) {
    outbuf[offset].val0 += ot[0][0];
    outbuf[offset].val1 += ot[0][1];
    outbuf[offset].val2 += ot[1][0];
    outbuf[offset].val3 += ot[1][1];
    outbuf[offset].val4 += ot[2][0];
    outbuf[offset].val5 += ot[2][1];
    outbuf[offset].val6 += ot[3][0];
    outbuf[offset].val7 += ot[3][1];
    tempout[offset2].val0 += ot[0][2];
    tempout[offset2].val1 += ot[0][3];
    tempout[offset2].val2 += ot[1][2];
    tempout[offset2].val3 += ot[1][3];
    tempout[offset2].val4 += ot[2][2];
    tempout[offset2].val5 += ot[2][3];
    tempout[offset2].val6 += ot[3][2];
    tempout[offset2].val7 += ot[3][3];   
  } else {
    outbuf[offset].val8 += ot[0][0];
    outbuf[offset].val9 += ot[0][1];
    outbuf[offset].val10 += ot[1][0];
    outbuf[offset].val11 += ot[1][1];
    outbuf[offset].val12 += ot[2][0];
    outbuf[offset].val13 += ot[2][1];
    outbuf[offset].val14 += ot[3][0];
    outbuf[offset].val15 += ot[3][1];
    tempout[offset2].val8 += ot[0][2];
    tempout[offset2].val9 += ot[0][3];
    tempout[offset2].val10 += ot[1][2];
    tempout[offset2].val11 += ot[1][3];
    tempout[offset2].val12 += ot[2][2];
    tempout[offset2].val13 += ot[2][3];
    tempout[offset2].val14 += ot[3][2];
    tempout[offset2].val15 += ot[3][3]; 
  }
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

void winograd_pe(float *input, floatv16 *weights, float *bias, floatv16 *output, 
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
#pragma HLS ARRAY_PARTITION variable=inbuf cyclic factor=2 dim=3
#pragma HLS ARRAY_PARTITION variable=inbuf cyclic factor=4 dim=2
#pragma HLS ARRAY_PARTITION variable=inbuf cyclic factor=4 dim=1

  // Buffer for holding extra data in an edge case where xtile==xtile_pad
  float sidebuf[2048][4][2];
#pragma HLS ARRAY_PARTITION variable=sidebuf cyclic factor=2 dim=3
#pragma HLS ARRAY_PARTITION variable=sidebuf cyclic factor=4 dim=2

  // Registers used for loading data from sidebuf
  float sidebuf_val[4][2];
#pragma HLS ARRAY_PARTITION variable=sidebuf_val cyclic factor=2 dim=2
#pragma HLS ARRAY_PARTITION variable=sidebuf_val cyclic factor=4 dim=1

  // Output buffer used for writing
  floatv16 outbuf[256 * 16];
  // Temporary output buffer to improve throughput during mult-acc stage
  floatv16 tempout[128 * 16];

  // Buffer used for reading lines from the input
  float line[260] = {0};
#pragma HLS ARRAY_PARTITION variable=line cyclic factor=2

  // Weight buffer
  floatv16 wbuf[256];

  // Bias buffer
  float biasbuf[1024];


  float inrow[2];
#pragma HLS ARRAY_PARTITION variable=inrow complete

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
  float ott[4][4 * 4];
#pragma HLS ARRAY_PARTITION variable=ott complete dim=1
#pragma HLS ARRAY_PARTITION variable=ott complete dim=2

  // Ouput tile transform stage 1 output
  float ot_s1[4][8];
#pragma HLS ARRAY_PARTITION variable=ot_s1 complete dim=1
#pragma HLS ARRAY_PARTITION variable=ot_s1 complete dim=2

  // Output tile transform stage 2 output
  float ot[4][4];
#pragma HLS ARRAY_PARTITION variable=ot complete dim=1
#pragma HLS ARRAY_PARTITION variable=ot complete dim=2 

  int bank_off;

  assert(rburst >= 13);
  assert(rburst <= 4096);
  assert(inchannels >= 1);
  assert(inchannels <= 1024);
  assert(outchannels >= 1);
  assert(outchannels <= 1024);

  assert(burstchannels >= 1);
  assert(burstchannels <= 256);

  assert(xdim >= 1);
  assert(xdim <= 256);
  assert(ydim >= 1);
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
  assert(ytile_pad >= 8);
  assert(ytile_pad <= 128);

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

  fact = ((xtile_pad) >> 3);

  /* Read bias data into buffer */
  memcpy(biasbuf, bias + outchannels * group, sizeof(float) * outchannels);

  for (n = 0; n < rpo; ++n) {
    y_off = 0;
    p_off = 1;
    i_off = 0;

    /* Clear tile buffers that have been modified after column transform */
    RESETLOOP:for (i = 0; i < burstchannels; ++i) {
      for (x = 0; x < xtile_pad; ++x) {
#pragma HLS pipeline
        for (p = 0; p < 2; ++p) {
          inbuf[i * ytile * xtile_pad + x][0][p] = 0;
          inbuf[i * ytile * xtile_pad + x + (ytile - 1) * xtile_pad][0][p] = 0;
          inbuf[i * ytile * xtile_pad + x + (ytile - 1) * xtile_pad][1][p] = 0;
          inbuf[i * ytile * xtile_pad + x + (ytile - 1) * xtile_pad][2][p] = 0;
          inbuf[i * ytile * xtile_pad + x + (ytile - 1) * xtile_pad][3][p] = 0;
          sidebuf[i * ytile + x][0][p] = 0;
          sidebuf[i * ytile + x][1][p] = 0;
          sidebuf[i * ytile + x][2][p] = 0;
          sidebuf[i * ytile + x][3][p] = 0;
        }
      }
    }

    /* Read the input line by line and tile it into the tile buffer */
    IREADLOOP: for (i = 0; i < rburst; ++i) {
      memcpy(line + 1, input + dataoff * numgroups * inchannels * ydim * xdim + 
          i * xdim + group * inchannels * ydim * xdim + 
          n * burstchannels * ydim * xdim, sizeof(float) * xdim);

      if (y_off * 2 + (p_off - 1) == ydim) {
        i_off++;
        y_off = 0;
        p_off = 1;
      }

      if (p_off == 4) {
        p_off = 2;
        y_off++;
      }

      int id_in = (i_off * ytile + y_off) * xtile_pad;

      XTILELOOP: for (x = 0; x < xtile_pad; ++x) {
#pragma HLS DEPENDENCE variable=inbuf inter false
#pragma HLS pipeline
        int start_x = x * 2;

        for (p = 0; p < 2; ++p) {
          int in_x = start_x + p;
          inrow[p] = line[in_x];
        }
        
        for (p = 0; p < 2; ++p) {
          float temp = inrow[p];
          int id_inx = id_in + x;
          inbuf[id_inx][p_off][p] = temp;
          if (p_off >= 2 && (y_off + 1) < ytile) {
            id_inx = id_inx + xtile_pad;
            inbuf[id_inx][p_off - 2][p] = temp;
          }
        } 
      }
      if (xtile == xtile_pad) {
        for (p = 0; p < 2; ++p) {
          float temp = line[xtile_pad * 2 + p];
          int idx = i_off * ytile + y_off;
          sidebuf[idx][p_off][p] = temp;
          if (p_off >= 2 && (y_off + 1) < ytile) {
            idx = idx + 1;
            sidebuf[idx][p_off - 2][p] = temp;
          }
        }
      }
      p_off++;
    }

    /* Apply a partial transform to all of the columns of the tile buffer */
    ITRANSLOOP: for (i = 0; i < (burstchannels * ytile * xtile_pad) >> 2; ++i) {
#pragma HLS pipeline
      for (off = 0; off < 4; ++off) {
        for (q = 0; q < 2; ++q) {
          for (p = 0; p < 4; ++p) {
            itcol[p] = inbuf[i * 4 + off][p][q];
          }
          p_transform(itcol, otcol);
          for (p = 0; p < 4; ++p) {
            inbuf[i * 4 + off][p][q] = otcol[p];
          }
        }
      }
    }

    if (xtile == xtile_pad) {
      for (i = 0; i < burstchannels * ytile; ++i) {
#pragma HLS pipeline
        for (q = 0; q < 2; ++ q) {
          for (p = 0; p < 4; ++p) {
            itcol[p] = sidebuf[i][p][q];
          }
          p_transform(itcol, otcol);
          for (p = 0; p < 4; ++p) {
            sidebuf[i][p][q] = otcol[p];
          }
        }
      }
    }
    
    for (o = 0; o < outchannels; ++o) {
      memcpy(wbuf, weights + (o + outchannels * group) * inchannels + 
          n * burstchannels, sizeof(floatv16) * burstchannels);
      out_offset = dataoff * numgroups * outchannels * ydim * fact + 
                    ((o + outchannels * group) * ydim) * fact;

      if (n == 0) {
        /* Set the output buffers to contain the biases */
        for (y = 0; y < ytile; ++y) {
          for (x = 0; x < fact; ++x) {
#pragma HLS pipeline
            offtemp1 = (y * 2 + 0) * fact + x;
            offtemp2 = y * fact + x;
            tempout[offtemp2].val0 = biasbuf[o];
            tempout[offtemp2].val1 = biasbuf[o];
            tempout[offtemp2].val2 = biasbuf[o];
            tempout[offtemp2].val3 = biasbuf[o];
            tempout[offtemp2].val4 = biasbuf[o];
            tempout[offtemp2].val5 = biasbuf[o];
            tempout[offtemp2].val6 = biasbuf[o];
            tempout[offtemp2].val7 = biasbuf[o];
            tempout[offtemp2].val8 = biasbuf[o];
            tempout[offtemp2].val9 = biasbuf[o];
            tempout[offtemp2].val10 = biasbuf[o];
            tempout[offtemp2].val11 = biasbuf[o];
            tempout[offtemp2].val12 = biasbuf[o];
            tempout[offtemp2].val13 = biasbuf[o];
            tempout[offtemp2].val14 = biasbuf[o];
            tempout[offtemp2].val15 = biasbuf[o];
            outbuf[offtemp1].val0 = biasbuf[o];
            outbuf[offtemp1].val1 = biasbuf[o];
            outbuf[offtemp1].val2 = biasbuf[o];
            outbuf[offtemp1].val3 = biasbuf[o];
            outbuf[offtemp1].val4 = biasbuf[o];
            outbuf[offtemp1].val5 = biasbuf[o];
            outbuf[offtemp1].val6 = biasbuf[o];
            outbuf[offtemp1].val7 = biasbuf[o];
            outbuf[offtemp1].val8 = biasbuf[o];
            outbuf[offtemp1].val9 = biasbuf[o];
            outbuf[offtemp1].val10 = biasbuf[o];
            outbuf[offtemp1].val11 = biasbuf[o];
            outbuf[offtemp1].val12 = biasbuf[o];
            outbuf[offtemp1].val13 = biasbuf[o];
            outbuf[offtemp1].val14 = biasbuf[o];
            outbuf[offtemp1].val15 = biasbuf[o];
          }
        } 
      } else {
        memcpy(outbuf, output + out_offset, sizeof(floatv16) * fact * ydim);
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
#pragma HLS DEPENDENCE variable=outbuf inter distance=14 true
#pragma HLS DEPENDENCE variable=tempout inter distance=14 true
#pragma HLS DEPENDENCE variable=ot inter false
#pragma HLS DEPENDENCE variable=ot intra false
#pragma HLS pipeline 
        if (tile_cnt * 4 == ytile * xtile_pad) {
          tile_cnt = 0;
          yt_off = 0;
          xt_off = 0;
          w_off++;
          sb_off++;
        } else if (xt_off * 4 == xtile_pad) {
          sb_off++;
          yt_off++;
          xt_off = 0;
        }

        sidebuf_val[0][0] = sidebuf[sb_off][0][0];
        sidebuf_val[0][1] = sidebuf[sb_off][0][1];
        sidebuf_val[1][0] = sidebuf[sb_off][1][0];
        sidebuf_val[1][1] = sidebuf[sb_off][1][1];
        sidebuf_val[2][0] = sidebuf[sb_off][2][0];
        sidebuf_val[2][1] = sidebuf[sb_off][2][1];
        sidebuf_val[3][0] = sidebuf[sb_off][3][0];
        sidebuf_val[3][1] = sidebuf[sb_off][3][1];

        for (j = 0; j < 4; ++j) {
          itt[j][0][0] = inbuf[i * 4 + j][0][0];
          itt[j][0][1] = inbuf[i * 4 + j][0][1];
          itt[j][1][0] = inbuf[i * 4 + j][1][0];
          itt[j][1][1] = inbuf[i * 4 + j][1][1];
          itt[j][2][0] = inbuf[i * 4 + j][2][0];
          itt[j][2][1] = inbuf[i * 4 + j][2][1];
          itt[j][3][0] = inbuf[i * 4 + j][3][0];
          itt[j][3][1] = inbuf[i * 4 + j][3][1];
          if (xtile == xtile_pad && (xt_off * 4 + j == xtile_pad - 1)) {
            itt[j][0][2] = sidebuf_val[0][0];
            itt[j][0][3] = sidebuf_val[0][1];
            itt[j][1][2] = sidebuf_val[1][0];
            itt[j][1][3] = sidebuf_val[1][1];
            itt[j][2][2] = sidebuf_val[2][0];
            itt[j][2][3] = sidebuf_val[2][1];
            itt[j][3][2] = sidebuf_val[3][0];
            itt[j][3][3] = sidebuf_val[3][1];
          } else {
            itt[j][0][2] = inbuf[i * 4 + j + 1][0][0];
            itt[j][0][3] = inbuf[i * 4 + j + 1][0][1];
            itt[j][1][2] = inbuf[i * 4 + j + 1][1][0];
            itt[j][1][3] = inbuf[i * 4 + j + 1][1][1]; 
            itt[j][2][2] = inbuf[i * 4 + j + 1][2][0];
            itt[j][2][3] = inbuf[i * 4 + j + 1][2][1];
            itt[j][3][2] = inbuf[i * 4 + j + 1][3][0];
            itt[j][3][3] = inbuf[i * 4 + j + 1][3][1]; 
          }

          /* Complete the remaining transform steps */

          p_transform(itt[j][0], it[j][0]);
          p_transform(itt[j][1], it[j][1]);
          p_transform(itt[j][2], it[j][2]);
          p_transform(itt[j][3], it[j][3]);

          /* Compute the element-wise multiplication between weight and input
           * tile */

          ott[j][0] = it[j][0][0] * wbuf[w_off].val0;
          ott[j][1] = it[j][0][1] * wbuf[w_off].val1;
          ott[j][2] = it[j][0][2] * wbuf[w_off].val2;
          ott[j][3] = it[j][0][3] * wbuf[w_off].val3;

          ott[j][4] = it[j][1][0] * wbuf[w_off].val4;
          ott[j][5] = it[j][1][1] * wbuf[w_off].val5;
          ott[j][6] = it[j][1][2] * wbuf[w_off].val6;
          ott[j][7] = it[j][1][3] * wbuf[w_off].val7;

          ott[j][8] = it[j][2][0] * wbuf[w_off].val8;
          ott[j][9] = it[j][2][1] * wbuf[w_off].val9;
          ott[j][10] = it[j][2][2] * wbuf[w_off].val10;
          ott[j][11] = it[j][2][3] * wbuf[w_off].val11;

          ott[j][12] = it[j][3][0] * wbuf[w_off].val12;
          ott[j][13] = it[j][3][1] * wbuf[w_off].val13;
          ott[j][14] = it[j][3][2] * wbuf[w_off].val14;
          ott[j][15] = it[j][3][3] * wbuf[w_off].val15;

          /* Transform the output */
          transform_outtile_s1(ott[j], ot_s1[j]);
          transform_outtile_s2(ot_s1[j], ot[j]);
        }
        
        offset = yt_off * 2 * fact + (xt_off >> 1);
        offset2 = yt_off * fact + (xt_off >> 1);
        acc_out(outbuf, tempout, ot, bank_off, offset, offset2);
        if (bank_off == 0)
          bank_off = 1;
        else
          bank_off = 0;
      }
      for (y = 0; y < ytile; ++y) {
        for (x = 0; x < fact; ++x) {
#pragma HLS pipeline
          offtemp1 = (y * 2 + 1) * fact + x;
          offtemp2 = y * fact + x;
          outbuf[offtemp1] = tempout[offtemp2];
        }
      }
      memcpy(output + out_offset, outbuf, sizeof(floatv16) * fact * ydim);
    }
  }
}
