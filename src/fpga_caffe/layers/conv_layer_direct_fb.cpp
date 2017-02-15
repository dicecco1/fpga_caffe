#include <stdio.h>
#include <assert.h>
#include <string.h>
#include <stdbool.h>
#include "hls_half.h"
#include "fpga_caffe/layers/conv_layer.hpp"

#define OCFACT 4 
#define OCDIV 2
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

void input_stage(float16 inbuf[256 * 32], unsigned short ksize,
    unsigned short xt_off, unsigned short xtile_pad, unsigned short yt_off, 
    unsigned short row_off, unsigned short ydim, unsigned short xdim, 
    unsigned short w_off, unsigned short burstchannels, float it[16][3]) {
  unsigned short p, q, j, toff;
  float itt[8][4];
#pragma HLS ARRAY_PARTITION variable=itt complete dim=1
#pragma HLS ARRAY_PARTITION variable=itt complete dim=2
  float tempbuf[21];
#pragma HLS ARRAY_PARTITION variable=tempbuf complete 

  short crow_off;
  unsigned short c_off = (ksize == 5) ? w_off >> 1 : w_off;
  unsigned short flag = w_off & 0x1;

  crow_off = (ksize >> 1) - row_off;
 
  unsigned short pad = 2;

  int in_idx = (((c_off * ydim + (yt_off - crow_off)) * xtile_pad * 2) >> 4)
      + xt_off;
  if (yt_off - crow_off >= 0 && yt_off - crow_off < ydim && 
      c_off < burstchannels) {
    tempbuf[2] = inbuf[in_idx].s0;
    tempbuf[3] = inbuf[in_idx].s1;
    tempbuf[4] = inbuf[in_idx].s2;
    tempbuf[5] = inbuf[in_idx].s3;
    tempbuf[6] = inbuf[in_idx].s4;
    tempbuf[7] = inbuf[in_idx].s5;
    tempbuf[8] = inbuf[in_idx].s6;
    tempbuf[9] = inbuf[in_idx].s7;
    tempbuf[10] = inbuf[in_idx].s8;
    tempbuf[11] = inbuf[in_idx].s9;
    tempbuf[12] = inbuf[in_idx].sa;
    tempbuf[13] = inbuf[in_idx].sb;
    tempbuf[14] = inbuf[in_idx].sc;
    tempbuf[15] = inbuf[in_idx].sd;
    tempbuf[16] = inbuf[in_idx].se;
    tempbuf[17] = inbuf[in_idx].sf; 

    if (xt_off == 0) {
      tempbuf[0] = 0;
      tempbuf[1] = 0;
    } else {
      tempbuf[0] = inbuf[in_idx - 1].se;
      tempbuf[1] = inbuf[in_idx - 1].sf;
    }

    if (xt_off * 8 + 8 == xtile_pad) {
      tempbuf[18] = 0;
      tempbuf[19] = 0;
    } else {
      tempbuf[18] = inbuf[in_idx + 1].s0;
      tempbuf[19] = inbuf[in_idx + 1].s1;
    }    
  } else {
    for (p = 0; p < 20; ++p) 
      tempbuf[p] = 0;
  }
  tempbuf[20] = 0;

  if (ksize != 5)
    toff = 1;
  else if (flag == 0)
    toff = 0;
  else
    toff = 3;

  for (p = 0; p < 3; ++p) {
    for (q = 0; q < 16; ++q) {
      if (p + q + toff + xt_off * 16 < xdim + pad)
        it[q][p] = tempbuf[p + q + toff];
      else
        it[q][p] = 0;
    }
  }
}

void wt_set_backward(float16 outbuf[OCFACT][256 * 16], int yt_off, 
    int xt_off, int fact, float wt[OCFACT][16][3]) { 
  int off = yt_off * fact + xt_off;

  for (int k = 0; k < OCFACT; ++k) {
    for (int p = 0; p < 3; ++p) {
      wt[k][0][p] = outbuf[k][off].s0;
      wt[k][1][p] = outbuf[k][off].s1;
      wt[k][2][p] = outbuf[k][off].s2;
      wt[k][3][p] = outbuf[k][off].s3;
      wt[k][4][p] = outbuf[k][off].s4;
      wt[k][5][p] = outbuf[k][off].s5;
      wt[k][6][p] = outbuf[k][off].s6;
      wt[k][7][p] = outbuf[k][off].s7;
      wt[k][8][p] = outbuf[k][off].s8;
      wt[k][9][p] = outbuf[k][off].s9;
      wt[k][10][p] = outbuf[k][off].sa;
      wt[k][11][p] = outbuf[k][off].sb;
      wt[k][12][p] = outbuf[k][off].sc;
      wt[k][13][p] = outbuf[k][off].sd;
      wt[k][14][p] = outbuf[k][off].se;
      wt[k][15][p] = outbuf[k][off].sf;
    }
  }
}

void wt_set(float16 wbuf[OCFACT][512], float wt[OCFACT][16][3], 
    unsigned short w_off, unsigned short row_off, unsigned short ksize) { 
  float wvals[16]; 
#pragma HLS ARRAY_PARTITION variable=wvals complete

  float it[3];
  float ot[3];

  for (int k = 0; k < OCFACT; ++k) {
    wvals[0] = wbuf[k][w_off].s0;
    wvals[1] = wbuf[k][w_off].s1;
    wvals[2] = wbuf[k][w_off].s2;
    wvals[3] = wbuf[k][w_off].s3;
    wvals[4] = wbuf[k][w_off].s4;
    wvals[5] = wbuf[k][w_off].s5;
    wvals[6] = wbuf[k][w_off].s6;
    wvals[7] = wbuf[k][w_off].s7;
    wvals[8] = wbuf[k][w_off].s8;
    wvals[9] = wbuf[k][w_off].s9;
    wvals[10] = wbuf[k][w_off].sa;
    wvals[11] = wbuf[k][w_off].sb;
    wvals[12] = wbuf[k][w_off].sc;
    wvals[13] = wbuf[k][w_off].sd;
    wvals[14] = wbuf[k][w_off].se;
    wvals[15] = wbuf[k][w_off].sf;

    if (ksize != 1) {
      it[0] = wvals[row_off * 3 + 0];
      it[1] = wvals[row_off * 3 + 1];
      it[2] = wvals[row_off * 3 + 2];
    } else {
      it[0] = 0;
      it[1] = wvals[0];
      it[2] = 0;
    }

    for (int p = 0; p < 16; ++p) {
      for (int q = 0; q < 3; ++q) {
        wt[k][p][q] = it[q];
      }
    } 
  }
}

/* Kernel used for computing direct convolution forward/backward. 
 * input:         flattened input array containing image data, padded to be
 *                divisible by 16 on the x dimension
 * weights:       pre-transformed 3x3 filters
 * bias:          flattened bias array
 * output:        output of the convolution, padded to be divisible by 16 on 
 *                the x dimension
 * group_idx:     group_idx index, leave as 0 if not using group_idx
 *                convolution
 * image_idx:     image offset
 */ 

extern "C" {

void conv_layer_direct_fb(float16 *input, float16 *weights, float *bias,
    float16 *output, kernel_params *params, int group_idx, int image_idx) { 
/* Ports */
#pragma HLS data_pack variable=weights
#pragma HLS data_pack variable=output
#pragma HLS data_pack variable=input
#pragma HLS INTERFACE m_axi port=input offset=slave bundle=gmem1
#pragma HLS INTERFACE m_axi port=output offset=slave bundle=gmem2
#pragma HLS INTERFACE m_axi port=weights offset=slave bundle=gmem3
#pragma HLS INTERFACE m_axi port=bias offset=slave bundle=gmem4
#pragma HLS INTERFACE m_axi port=params offset=slave bundle=gmem5
#pragma HLS INTERFACE s_axilite port=input bundle=control
#pragma HLS INTERFACE s_axilite port=output bundle=control
#pragma HLS INTERFACE s_axilite port=weights bundle=control
#pragma HLS INTERFACE s_axilite port=bias bundle=control
#pragma HLS INTERFACE s_axilite port=params bundle=control

#pragma HLS INTERFACE s_axilite port=group_idx bundle=control
#pragma HLS INTERFACE s_axilite port=image_idx bundle=control
#pragma HLS INTERFACE s_axilite port=return bundle=control

  // Input tile buffer
  float16 inbuf[256 * 16]; 

  // Output buffer used for writing
  float16 outbuf[OCFACT][256 * 16];
#pragma HLS ARRAY_PARTITION variable=outbuf complete dim=1

  // Weight buffer
  float16 wbuf[OCFACT][512];
#pragma HLS ARRAY_PARTITION variable=wbuf complete dim=1

  // Bias buffer
  float biasbuf[1024];
#pragma HLS ARRAY_PARTITION variable=biasbuf cyclic factor=4

  // Input tile registers post transform
  float it[16][3];
#pragma HLS ARRAY_PARTITION variable=it complete dim=1
#pragma HLS ARRAY_PARTITION variable=it complete dim=2

  // Temporary output tile registers
  float ot[OCFACT][16][3];
#pragma HLS ARRAY_PARTITION variable=ot complete dim=1
#pragma HLS ARRAY_PARTITION variable=ot complete dim=2
#pragma HLS ARRAY_PARTITION variable=ot complete dim=3

  float wt[OCFACT][16][3];
#pragma HLS ARRAY_PARTITION variable=wt complete dim=1
#pragma HLS ARRAY_PARTITION variable=wt complete dim=2
#pragma HLS ARRAY_PARTITION variable=wt complete dim=3

  // Ouput tile transform stage 1 output
  float ot_s1[24];
#pragma HLS ARRAY_PARTITION variable=ot_s1 complete dim=1

  float ot_s2[3][4];
#pragma HLS ARRAY_PARTITION variable=ot_s2 complete dim=1
#pragma HLS ARRAY_PARTITION variable=ot_s2 complete dim=2

  float ot_s3[3][2];
#pragma HLS ARRAY_PARTITION variable=ot_s3 complete dim=1
#pragma HLS ARRAY_PARTITION variable=ot_s3 complete dim=2

  float ot_s4[3];
#pragma HLS ARRAY_PARTITION variable=ot_s4 complete dim=1
  
  float otf[16];
#pragma HLS ARRAY_PARTITION variable=otf complete dim=1
  
  int inchannels = params->inchannels;
  int outchannels = params->outchannels;
  int burstchannels = params->burstchannels;
  int xdim = params->xdim;
  int ydim = params->ydim;
  int xtile_pad = params->xtile_pad;
  int ksize = params->ksize;
  int rpo = params->rpo;
  int numgroups = params->numgroups;
  int backward = params->backward;

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

  assert(group_idx >= 0);
  assert(group_idx <= 1);

  assert(numgroups <= 2);
  assert(numgroups >= 1);

  assert(xtile_pad >= 8);
  assert(xtile_pad <= 128);

  assert(rpo >= 1);
  assert(rpo <= 64);

  assert(ksize == 1 || ksize == 3 || ksize == 5);

  assert(backward == 0 || backward == 1);

  int i, n, y, x, p, q, j, o, k;

  unsigned short w_off;
  unsigned short row_off, xt_off, yt_off;
  unsigned short fact = xtile_pad >> 3;
  unsigned short out_size, weight_size;
  int out_offset;  
  int weight_offset;
  unsigned short offset;
  int in_off;
  unsigned short iter, ofm_iters;
  bool backward_flag = backward;
  unsigned short mod_channel;
  
  if (backward_flag) {
    if (ksize == 5) {
      mod_channel = (burstchannels < 5) ? 10 : burstchannels * 2;
    } else {
      mod_channel = (burstchannels < 10) ? 10 : burstchannels;
    }
  } else {
    mod_channel = (ksize == 5) ? burstchannels * 2 : burstchannels;
  }

  unsigned short mod_ydim = ((ydim < 10) && !(backward_flag)) ? 10 : ydim; 
  // Read bias data into buffer 
  memcpy(biasbuf, bias + (outchannels * group_idx), sizeof(float) *
      outchannels);

  int mac_iterations = mod_channel * mod_ydim * fact;

  if (ksize == 3)
    mac_iterations *= 3;
  else if (ksize == 5) 
    mac_iterations *= 5;
  
  for (n = 0; n < rpo; ++n) {
    // Read the input 
    in_off = (((image_idx * numgroups + group_idx) * inchannels) * ydim *
        xtile_pad * 2 + n * burstchannels * ydim * xtile_pad * 2) >> 4;

    memcpy(inbuf, input + in_off, sizeof(float16) * ((burstchannels * ydim * 
            xtile_pad * 2) >> 4)); 

    ofm_iters = (outchannels & (OCFACT - 1)) ? (outchannels >> OCDIV) + 1 : 
      (outchannels >> OCDIV);
    for (o = 0; o < ofm_iters; ++o) {
      if (n == 0 && !backward_flag) {
        // Set the output buffers to contain the biases 
        for (i = 0; i < ydim * fact; ++i) {
#pragma HLS pipeline
          for (k = 0; k < OCFACT; ++k) {
            outbuf[k][i].s0 = biasbuf[o * OCFACT + k];
            outbuf[k][i].s1 = biasbuf[o * OCFACT + k];
            outbuf[k][i].s2 = biasbuf[o * OCFACT + k];
            outbuf[k][i].s3 = biasbuf[o * OCFACT + k];
            outbuf[k][i].s4 = biasbuf[o * OCFACT + k];
            outbuf[k][i].s5 = biasbuf[o * OCFACT + k];
            outbuf[k][i].s6 = biasbuf[o * OCFACT + k];
            outbuf[k][i].s7 = biasbuf[o * OCFACT + k];
            outbuf[k][i].s8 = biasbuf[o * OCFACT + k];
            outbuf[k][i].s9 = biasbuf[o * OCFACT + k];
            outbuf[k][i].sa = biasbuf[o * OCFACT + k];
            outbuf[k][i].sb = biasbuf[o * OCFACT + k];
            outbuf[k][i].sc = biasbuf[o * OCFACT + k];
            outbuf[k][i].sd = biasbuf[o * OCFACT + k];
            outbuf[k][i].se = biasbuf[o * OCFACT + k];
            outbuf[k][i].sf = biasbuf[o * OCFACT + k];
          }
        } 
      } else {
        for (k = 0; k < OCFACT; ++k) {
          out_offset = image_idx * numgroups * outchannels * ydim * fact + 
          ((o * OCFACT + k + outchannels * group_idx) * ydim) * fact;
          out_size = fact * ydim;
          memcpy(outbuf[k], output + out_offset, sizeof(float16) * fact * 
              ydim);
        }
      } 
      
      for (k = 0; k < OCFACT; ++k) {
          weight_offset = (o * OCFACT + k + outchannels * group_idx) *
            inchannels + n * burstchannels;
          weight_size = burstchannels;

          if (ksize == 5) {
            weight_offset = weight_offset << 1;
            weight_size = weight_size << 1;
          }
          memcpy(wbuf[k], weights + weight_offset, sizeof(float16) *
              weight_size);
      }

      w_off = 0;
      xt_off = 0;
      yt_off = 0;
      row_off = 0;
      iter = 0;
      MULTACCSTAGE: for (i = 0; i < mac_iterations; ++i, ++iter) {
#pragma HLS DEPENDENCE variable=outbuf inter false
#pragma HLS DEPENDENCE variable=wbuf inter false
#pragma HLS DEPENDENCE variable=outbuf intra false
#pragma HLS DEPENDENCE variable=wbuf intra false
#pragma HLS pipeline        
        if (backward_flag) {
          if (iter == mod_channel) {
            iter = 0;
            if (row_off + 1 == ksize) {
              row_off = 0;
              if (xt_off + 1 == fact) {
                xt_off = 0;
                yt_off++;
              } else {
                xt_off++;
              }
            } else {
              row_off++;
            }
          }
          w_off = iter;
        } else {
          if (iter == fact) {
            if (yt_off + 1 == mod_ydim) {
              yt_off = 0;
              if (row_off + 1 == ksize) {
                row_off = 0;
                w_off++;
              } else {
                row_off++;
              }
            } else {
              yt_off++;
            }
            iter = 0;
          }
          xt_off = iter;
        } 

        offset = yt_off * fact + xt_off;

        input_stage(inbuf, ksize, xt_off, xtile_pad, yt_off, 
            row_off, ydim, xdim, w_off, burstchannels, it);
        if (backward_flag)
          wt_set_backward(outbuf, yt_off, xt_off, fact, wt);
        else
          wt_set(wbuf, wt, w_off, row_off, ksize); 

        for (k = 0; k < OCFACT; ++k) {
          for (p = 0; p < 16; ++p) {
            for (q = 0; q < 3; ++q) {
              ot[k][p][q] = it[p][q] * wt[k][p][q];
            }
          }
          for (p = 0; p < 8; ++p) 
            ot_s1[p + 2 * 8] = ot[k][p * 2][2] + ot[k][p * 2 + 1][2];

          if (backward_flag) {
            for (q = 0; q < 2; ++q) {
              for (p = 0; p < 8; ++p) 
                ot_s1[p + q * 8] = ot[k][p * 2][q] + ot[k][p * 2 + 1][q];
            }
            for (q = 0; q < 3; ++q) {
              for (p = 0; p < 4; ++p)
                ot_s2[q][p] = ot_s1[p * 2 + q * 8] + ot_s1[p * 2 + 1 + q * 8];
            }
          } else {
            for (p = 0; p < 16; ++p) {
              ot_s1[p] = ot[k][p][0] + ot[k][p][1] + ot[k][p][2];
            }
          }
          for (q = 0; q < 3; ++q) {
            for (p = 0; p < 2; ++p)
              ot_s3[q][p] = ot_s2[q][p * 2] + ot_s2[q][p * 2 + 1];
            ot_s4[q] = ot_s3[q][0] + ot_s3[q][1];
          }
 
          for (p = 0; p < 16; ++p)
            otf[p] = ot_s1[p];

          if (backward_flag) {
            for (p = 0; p < 16; ++p)
              otf[p] = 0;
            for (p = 0; p < 3; ++p)
              otf[row_off * 3 + p] = ot_s4[p];
          }
          if (backward_flag) {
            wbuf[k][w_off].s0 += otf[0];
            wbuf[k][w_off].s1 += otf[1];
            wbuf[k][w_off].s2 += otf[2];
            wbuf[k][w_off].s3 += otf[3];
            wbuf[k][w_off].s4 += otf[4];
            wbuf[k][w_off].s5 += otf[5];
            wbuf[k][w_off].s6 += otf[6];
            wbuf[k][w_off].s7 += otf[7];
            wbuf[k][w_off].s8 += otf[8];
            wbuf[k][w_off].s9 += otf[9];
            wbuf[k][w_off].sa += otf[10];
            wbuf[k][w_off].sb += otf[11];
            wbuf[k][w_off].sc += otf[12];
            wbuf[k][w_off].sd += otf[13];
            wbuf[k][w_off].se += otf[14];
            wbuf[k][w_off].sf += otf[15]; 
          } else {
            outbuf[k][offset].s0 += otf[0];
            outbuf[k][offset].s1 += otf[1];
            outbuf[k][offset].s2 += otf[2];
            outbuf[k][offset].s3 += otf[3];
            outbuf[k][offset].s4 += otf[4];
            outbuf[k][offset].s5 += otf[5];
            outbuf[k][offset].s6 += otf[6];
            outbuf[k][offset].s7 += otf[7];
            outbuf[k][offset].s8 += otf[8];
            outbuf[k][offset].s9 += otf[9];
            outbuf[k][offset].sa += otf[10];
            outbuf[k][offset].sb += otf[11];
            outbuf[k][offset].sc += otf[12];
            outbuf[k][offset].sd += otf[13];
            outbuf[k][offset].se += otf[14];
            outbuf[k][offset].sf += otf[15]; 
          }
        }
      }
      for (k = 0; k < OCFACT; ++k) {
        if (backward_flag) {
          out_offset = (o * OCFACT + k + outchannels * group_idx) * inchannels
            + n * burstchannels;
          out_size = burstchannels;

          if (ksize == 5) {
            out_offset = out_offset << 1;
            out_size = out_size << 1;
          }
          if (o * OCFACT + k < outchannels) {
            memcpy(weights + out_offset, wbuf[k], sizeof(float16) * out_size);
          }
        } else {
          out_offset = image_idx * numgroups * outchannels * ydim * fact +
            ((o * OCFACT + k + outchannels * group_idx) * ydim) * fact;
          out_size = fact * ydim;
          if (o * OCFACT + k < outchannels) {
            memcpy(output + out_offset, outbuf[k], sizeof(float16) * out_size);
          }
        }
      }
    }
  }
}

}
