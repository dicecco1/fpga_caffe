#include <stdio.h>
#include <assert.h>
#include <string.h>
#include <stdbool.h>

#include "../../../include/fpga_caffe/layers/conv_layer.hpp"

#define OCFACT 8 
#define OCDIV 3
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

void input_stage(float16 inbuf[256 * 16], unsigned short ksize,
    unsigned short xt_off, unsigned short xtile_pad, unsigned short yt_off, 
    unsigned short row_off, unsigned short ydim, unsigned short xdim, 
    unsigned short w_off, unsigned short burstchannels, int image_off, float it[16][3]) {
  unsigned short p, q, j, toff;

  float tempbuf[21];
#pragma HLS ARRAY_PARTITION variable=tempbuf complete 

  short crow_off;
  unsigned short c_off = (ksize == 5) ? w_off >> 1 : w_off;
  unsigned short flag = w_off & 0x1;

  crow_off = (ksize >> 1) - row_off;
 
  unsigned short pad = 2;

  int in_idx = ((((image_off * burstchannels + c_off) * ydim + (yt_off - crow_off)) * xtile_pad * 2) >> 4)
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
    for (p = 0; p < 21; ++p) 
      tempbuf[p] = 0;
  }

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

void wt_set(float16 wbuf[OCFACT][512], float wt[OCFACT][16][3], 
    unsigned short w_off, unsigned short row_off, unsigned short ksize,
    unsigned short xt_off, unsigned short xdim, bool backward_flag) { 
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
        if (backward_flag) {
          if (p + xt_off * 16 < xdim)
            wt[k][p][q] = wvals[p];
          else
            wt[k][p][q] = 0;
        } else {
          wt[k][p][q] = it[q];
        }
      }
    } 
  }
}
/* Kernel used for computing direct convolution forward and backward. 
 * input:         flattened input array containing image data, padded to be
 *                divisible by 16 on the x dimension
 * weights:       3x3 filters, padded to be size 16
 * bias:          flattened bias array
 * output:        output of the convolution, padded to be divisible by 16 on 
 *                the x dimension
 * group_idx:     group_idx index, leave as 0 if not using group convolution
 * image_idx:     image offset
 */ 

extern "C" {

void conv_layer_direct_fb(float16 *input, float16 *weights, float *bias,
    float16 *output, int *params, int group_idx, int image_idx) { 
// Ports 
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
  float16 inbuf[4 * 256 * 16]; 

  // Output buffer used for writing
  float16 outbuf[OCFACT][512];
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
  
  float otf[OCFACT][16];
#pragma HLS ARRAY_PARTITION variable=otf complete dim=1
#pragma HLS ARRAY_PARTITION variable=otf complete dim=2
 
  int inchannels = params[0];
  int outchannels = params[1];
  int burstchannels = params[2];
  int rpo = params[3];
  int rpofm = params[4];
  int burstydim = params[5];
  int ydim = params[6];
  int xdim = params[7];
  int xtile_pad = params[8];
  int ksize = params[9];
  int numgroups = params[10];
  int numimages = params[11];
  int backward = params[13];

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

  assert(numgroups <= 2);
  assert(numgroups >= 1);

  assert(xtile_pad >= 8);
  assert(xtile_pad <= 128);

  assert(rpo >= 1);
  assert(rpo <= 64);

  assert(rpofm <= 32);
  assert(rpofm >= 1);

  assert(burstydim <= 64);
  assert(burstydim >= 1);

  assert(ksize == 1 || ksize == 3 || ksize == 5);

  assert(numimages >= 1);

  assert(backward == 0 || backward == 1);

  unsigned short w_off;
  unsigned short row_off, xt_off, yt_off;
  unsigned short fact = xtile_pad >> 3;
  unsigned short out_size, weight_size;
  int out_offset;  
  int weight_offset;
  unsigned short offset;
  int in_off, inbuf_off;
  unsigned short iter, ofm_iters;
  bool backward_flag = backward;
  unsigned short mod_channel;
  
  if (backward_flag) {
    if (ksize == 5) {
      mod_channel = (burstchannels * 2 * ksize < FADD_LATENCY) ? FADD_LATENCY :
        burstchannels * 2;
    } else {
      mod_channel = (burstchannels * ksize < FADD_LATENCY) ? FADD_LATENCY :
        burstchannels;
    }
  } else {
    mod_channel = (ksize == 5) ? burstchannels * 2 : burstchannels;
  }

  unsigned short mod_ydim;
  mod_ydim = ((burstydim < FADD_LATENCY) && !(backward_flag)) ? FADD_LATENCY :
    burstydim; 
  // Read bias data into buffer 
  memcpy(biasbuf, bias + (outchannels * group_idx), sizeof(float) *
      outchannels);

  int mac_iterations = mod_channel * mod_ydim * fact * ksize;
    ofm_iters = (outchannels & (OCFACT - 1)) ? (outchannels >> OCDIV) + 1 : 
      (outchannels >> OCDIV);

  for (int n = 0; n < rpo; ++n) {
    // Read the input
    for (int image_off = 0; image_off < numimages; ++image_off) {
      in_off = (((image_idx * numimages + image_off) * numgroups + group_idx)
        * inchannels) * ydim * fact + n * burstchannels * ydim * fact;

      inbuf_off = image_off * burstchannels * ydim * fact;
      memcpy(inbuf + inbuf_off, input + in_off, sizeof(float16) *
        burstchannels * ydim * fact); 
    }
    for (int o = 0; o < ofm_iters; ++o) {
      for (int image_off = 0; image_off < numimages; ++image_off) {
        for (int offy = 0; offy < rpofm; ++offy) {
          if (n == 0 && !backward_flag) {
            // Set the output buffers to contain the biases 
            for (int i = 0; i < burstydim * fact; ++i) {
#pragma HLS pipeline
              for (int k = 0; k < OCFACT; ++k) {
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
          } else if (!backward_flag) {
            for (int k = 0; k < OCFACT; ++k) {
              out_offset = (image_idx * numimages + image_off) * numgroups *
                outchannels * ydim * fact + ((o * OCFACT + k + outchannels *
                group_idx) * ydim + offy * burstydim) * fact;
              out_size = fact * burstydim;
              memcpy(outbuf[k], output + out_offset, sizeof(float16) *
                  out_size);
            }
          } else if (offy == 0 && image_off == 0) {
            for (int k = 0; k < OCFACT; ++k) {
              out_offset = (o * OCFACT + k + outchannels * group_idx) *
                inchannels + n * burstchannels;
              out_size = burstchannels;

              if (ksize == 5) {
                out_offset = out_offset << 1;
                out_size = out_size << 1;
              }
              memcpy(outbuf[k], output + out_offset, sizeof(float16) *
                  out_size);
            } 
          }
          for (int k = 0; k < OCFACT; ++k) {
            if (backward_flag) {
              weight_offset = (image_idx * numimages + image_off) * numgroups
                * outchannels * ydim * fact + ((o * OCFACT + k + outchannels
                * group_idx) * ydim + offy * burstydim) * fact;
              weight_size = fact * burstydim;
            } else {
              weight_offset = (o * OCFACT + k + outchannels * group_idx) *
                inchannels + n * burstchannels;
              weight_size = burstchannels;

              if (ksize == 5) {
                weight_offset = weight_offset << 1;
                weight_size = weight_size << 1;
              }
            } 
            if ((!backward_flag && (offy == 0) && image_off == 0) ||
              backward_flag)
              memcpy(wbuf[k], weights + weight_offset, sizeof(float16) *
                  weight_size);
          }
          
          w_off = 0;
          xt_off = 0;
          yt_off = 0;
          row_off = 0;
          iter = 0;
          MULTACCSTAGE: for (int i = 0; i < mac_iterations; ++i, ++iter) {
#pragma HLS DEPENDENCE variable=outbuf inter false
#pragma HLS DEPENDENCE variable=wbuf inter false
#pragma HLS DEPENDENCE variable=otf inter false
#pragma HLS pipeline        
            if (backward_flag) {
              if (iter == ksize) {
                iter = 0;
                if (w_off == mod_channel - 1) {
                  w_off = 0;
                  if (xt_off + 1 == fact) {
                    xt_off = 0;
                    yt_off++;
                  } else {
                    xt_off++;
                  }
                } else {
                  w_off++;
                }
              }
              row_off = iter;
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
            unsigned short w_idx = (backward_flag) ? offset : w_off;

            input_stage(inbuf, ksize, xt_off, xtile_pad, yt_off + offy *
              burstydim, row_off, ydim, xdim, w_off, burstchannels,
              image_off, it);
            
            wt_set(wbuf, wt, w_idx, row_off, ksize, xt_off, xdim,
                backward_flag); 

            for (int k = 0; k < OCFACT; ++k) {
              for (int p = 0; p < 16; ++p) {
                for (int q = 0; q < 3; ++q) {
                  ot[k][p][q] = it[p][q] * wt[k][p][q];
                }
              }
              for (int p = 0; p < 8; ++p) 
                ot_s1[p + 2 * 8] = ot[k][p * 2][2] + ot[k][p * 2 + 1][2];

              if (backward_flag) {
                for (int q = 0; q < 2; ++q) {
                  for (int p = 0; p < 8; ++p) 
                    ot_s1[p + q * 8] = ot[k][p * 2][q] + ot[k][p * 2 + 1][q];
                }
                for (int q = 0; q < 3; ++q) {
                  for (int p = 0; p < 4; ++p)
                    ot_s2[q][p] = ot_s1[p * 2 + q * 8] +
                      ot_s1[p * 2 + 1 + q * 8];
                }
              } else {
                for (int p = 0; p < 16; ++p) {
                  ot_s1[p] = ot[k][p][0] + ot[k][p][1] + ot[k][p][2];
                }
              }

              for (int q = 0; q < 3; ++q) {
                for (int p = 0; p < 2; ++p)
                  ot_s3[q][p] = ot_s2[q][p * 2] + ot_s2[q][p * 2 + 1];
                ot_s4[q] = ot_s3[q][0] + ot_s3[q][1];
              }
     
              if (backward_flag) {
                for (int p = 0; p < 3; ++p)
                  otf[k][row_off * 3 + p] = ot_s4[p];
              } else {
                for (int p = 0; p < 16; ++p)
                  otf[k][p] = ot_s1[p];
              }

              unsigned short o_idx = (backward_flag) ? w_off : offset;
             
              bool acc_enable = ((backward_flag && (row_off == ksize - 1)) ||
                  !backward_flag);

              if (acc_enable) {
                outbuf[k][o_idx].s0 += otf[k][0];
                outbuf[k][o_idx].s1 += otf[k][1];
                outbuf[k][o_idx].s2 += otf[k][2];
                outbuf[k][o_idx].s3 += otf[k][3];
                outbuf[k][o_idx].s4 += otf[k][4];
                outbuf[k][o_idx].s5 += otf[k][5];
                outbuf[k][o_idx].s6 += otf[k][6];
                outbuf[k][o_idx].s7 += otf[k][7];
                outbuf[k][o_idx].s8 += otf[k][8];
                outbuf[k][o_idx].s9 += otf[k][9];
                outbuf[k][o_idx].sa += otf[k][10];
                outbuf[k][o_idx].sb += otf[k][11];
                outbuf[k][o_idx].sc += otf[k][12];
                outbuf[k][o_idx].sd += otf[k][13];
                outbuf[k][o_idx].se += otf[k][14];
                outbuf[k][o_idx].sf += otf[k][15];
              }
            }
          }
          for (int k = 0; k < OCFACT; ++k) {
            if (backward_flag) {
              out_offset = (o * OCFACT + k + outchannels * group_idx) *
                inchannels + n * burstchannels;
              out_size = burstchannels;

              if (ksize == 5) {
                out_offset = out_offset << 1;
                out_size = out_size << 1;
              }
            } else {
              out_offset = (image_idx * numimages + image_off) * numgroups *
                outchannels * ydim * fact + ((o * OCFACT + k + outchannels *
                group_idx) * ydim + offy * burstydim) * fact;
              out_size = fact * burstydim;
            }

            if (o * OCFACT + k < outchannels && 
              ((backward_flag && (offy == rpofm - 1) && 
              (image_off == numimages - 1)) || !backward_flag)) {
              memcpy(output + out_offset, outbuf[k], sizeof(float16) *
                out_size);
            }
          }
        }
      }
    }
  }
}

}
