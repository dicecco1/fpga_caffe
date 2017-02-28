#include <stdio.h>
#include <assert.h>
#include <string.h>

#define OCFACT 16 
#define OCDIV 4

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
    unsigned short yrow_off, unsigned short xrow_off, unsigned short ydim,
    unsigned short w_off, float it[16]) {

  unsigned short p, q, j, toff;

  float tempbuf[21];
#pragma HLS ARRAY_PARTITION variable=tempbuf complete 

  short crow_off;
  unsigned short c_off = (ksize == 5) ? w_off >> 1 : w_off;
  unsigned short flag = w_off & 0x1;
 
  crow_off = (ksize >> 1) - yrow_off;
 
  int in_idx = (((c_off * ydim + (yt_off - crow_off)) * xtile_pad * 2) >> 4)
      + xt_off;
  if (yt_off - crow_off >= 0 && yt_off - crow_off < ydim) {
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

  if (ksize == 1)
    toff = 2;
  else if (ksize == 3)
    toff = 1;
  else if (flag == 0)
    toff = 0;
  else
    toff = 3;

  for (q = 0; q < 16; ++q) {
    it[q] = tempbuf[xrow_off + q + toff];
  }
}

void wt_set(float16 wbuf[OCFACT][512], float wt[OCFACT][16], 
    unsigned short w_off, unsigned short yrow_off, unsigned short xrow_off,
    unsigned short ksize) { 
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
      it[0] = wvals[yrow_off * 3 + 0];
      it[1] = wvals[yrow_off * 3 + 1];
      it[2] = wvals[yrow_off * 3 + 2];
    } else {
      for (int q = 0; q < 3; ++q) 
        it[q] = wvals[0];
    }
    for (int p = 0; p < 16; ++p) {
      wt[k][p] = it[xrow_off];
    }  
  }
}

/* Kernel used for computing direct convolution. 
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

void conv_layer_direct(float16 *input, float16 *weights, float *bias,
    float16 *output, int *params, int group_idx, int image_idx) {

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
  float16 outbuf[OCFACT][512];
#pragma HLS ARRAY_PARTITION variable=outbuf complete dim=1

  // Weight buffer
  float16 wbuf[OCFACT][512];
#pragma HLS ARRAY_PARTITION variable=wbuf complete dim=1

  // Bias buffer
  float biasbuf[1024];
#pragma HLS ARRAY_PARTITION variable=biasbuf cyclic factor=4

  // Input tile registers post transform
  float it[16];
#pragma HLS ARRAY_PARTITION variable=it complete dim=1

  // Temporary output tile registers
  float ot[OCFACT][16];
#pragma HLS ARRAY_PARTITION variable=ot complete dim=1
#pragma HLS ARRAY_PARTITION variable=ot complete dim=2

  float wt[OCFACT][16];
#pragma HLS ARRAY_PARTITION variable=wt complete dim=1
#pragma HLS ARRAY_PARTITION variable=wt complete dim=2

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

  assert(rpofm <= 32);
  assert(rpofm >= 1);

  assert(burstydim <= 64);
  assert(burstydim >= 1);

  assert(ksize == 1 || ksize == 3 || ksize == 5);

  unsigned short w_off = 0;
  unsigned short yrow_off = 0;
  unsigned short xrow_off = 0;
  unsigned short xt_off = 0;
  unsigned short yt_off = 0;
  int fact = xtile_pad >> 3;
  int out_offset = 0;  
  int weight_offset = 0;
  int weight_size = 0;
  int offset = 0;

  float lineval;
  int in_off;
  /* Read bias data into buffer */
  memcpy(biasbuf, bias + (outchannels * group_idx), sizeof(float) *
      outchannels);

  int mac_iterations = burstchannels * burstydim * fact;
  
  if (ksize == 3)
    mac_iterations *= (3 * 3);
  else if (ksize == 5)
    mac_iterations *= (5 * 2 * 3);

  for (int n = 0; n < rpo; ++n) {
    /* Read the input line by line and tile it into the tile buffer */
    in_off = (((image_idx * numgroups + group_idx) * inchannels) * ydim *
        xtile_pad * 2 + n * burstchannels * ydim * xtile_pad * 2) >> 4;

    memcpy(inbuf, input + in_off, sizeof(float16) * ((burstchannels * ydim * 
            xtile_pad * 2) >> 4)); 

    unsigned short ofm_iters = (outchannels & (OCFACT - 1)) ? 
      (outchannels >> OCDIV) + 1 : (outchannels >> OCDIV);
    for (int o = 0; o < ofm_iters; ++o) {
      for (int k = 0; k < OCFACT; ++k) {
        weight_offset = (o * OCFACT + k + outchannels * group_idx) * inchannels
          + n * burstchannels;
        weight_size = burstchannels;

        if (ksize == 5) {
          weight_offset = weight_offset << 1;
          weight_size = weight_size << 1;
        }

        memcpy(wbuf[k], weights + weight_offset, 
            sizeof(float16) * weight_size);
      }

      for (int offy = 0; offy < rpofm; ++offy) {
        if (n == 0) {
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
        } else {
          for (int k = 0; k < OCFACT; ++k) {
            out_offset = image_idx * numgroups * outchannels * ydim * fact +
              ((o * OCFACT + k + outchannels * group_idx) * ydim + offy *
              burstydim) * fact;
            memcpy(outbuf[k], output + out_offset, sizeof(float16) * fact *
                burstydim);
          }
        }

        w_off = 0;
        xt_off = 0;
        yt_off = 0;
        yrow_off = 0;
        xrow_off = 0;
        MULTACCSTAGE: for (int i = 0; i < mac_iterations; ++i, ++xt_off) {
#pragma HLS DEPENDENCE variable=outbuf inter false 
#pragma HLS pipeline        
          if (xt_off * 8 == xtile_pad) {
            if (yt_off + 1 == burstydim) {
              yt_off = 0;
              if (yrow_off == ksize - 1) {
                yrow_off = 0;
                if ((xrow_off + 1 == 3) || ksize == 1) {
                  xrow_off = 0;
                  w_off++;
                } else {
                  xrow_off++;
                }
              } else {
                yrow_off++;
              }
            } else {
              yt_off++;
            }
            xt_off = 0;
          }

          offset = yt_off * fact + xt_off;        
          input_stage(inbuf, ksize, xt_off, xtile_pad, yt_off + offy *
              burstydim, yrow_off, xrow_off, ydim, w_off, it);
          // Compute the element-wise multiplication
          wt_set(wbuf, wt, w_off, yrow_off, xrow_off, ksize); 
          for (int k = 0; k < OCFACT; ++k) {
            for (int p = 0; p < 16; ++p) {
              ot[k][p] = it[p] * wt[k][p];
            }
                        
            outbuf[k][offset].s0 += ot[k][0];
            outbuf[k][offset].s1 += ot[k][1];
            outbuf[k][offset].s2 += ot[k][2];
            outbuf[k][offset].s3 += ot[k][3];
            outbuf[k][offset].s4 += ot[k][4];
            outbuf[k][offset].s5 += ot[k][5];
            outbuf[k][offset].s6 += ot[k][6];
            outbuf[k][offset].s7 += ot[k][7];
            outbuf[k][offset].s8 += ot[k][8];
            outbuf[k][offset].s9 += ot[k][9];
            outbuf[k][offset].sa += ot[k][10];
            outbuf[k][offset].sb += ot[k][11];
            outbuf[k][offset].sc += ot[k][12];
            outbuf[k][offset].sd += ot[k][13];
            outbuf[k][offset].se += ot[k][14];
            outbuf[k][offset].sf += ot[k][15];
          }
        }
        int image_off = image_idx * numgroups * outchannels * ydim * fact;
        for (int k = 0; k < OCFACT; ++k) {
          out_offset = image_off +
            ((o * OCFACT + k + outchannels * group_idx) * ydim + offy *
             burstydim) * fact;
          if (o * OCFACT + k < outchannels) {
            memcpy(output + out_offset, outbuf[k], sizeof(float16) * fact *
                burstydim);
          }
        }
      }   
    }
  }
}

}
