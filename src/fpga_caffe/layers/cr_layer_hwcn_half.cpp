#include <stdio.h>
#include <assert.h>
#include <string.h>
#include <stdbool.h>

#include "../../../include/fpga_caffe/layer.hpp"
#include "../../../include/fpga_caffe/half.hpp"
#include "../../../include/fpga_caffe/vector_types.hpp"

#define OCFACT 1 

/* Kernel used for computing direct convolution forward and backward. 
 * input:         flattened input array containing image data
 * weights:       convolution filters
 * bias:          flattened bias array
 * output:        output of the convolution
 */ 

extern "C" {

void cr_layer_hwcn_half(chalf16 *input, chalf16 *weights, chalf *bias,
    chalf16 *output, int *params) { 
// Ports 
#pragma HLS data_pack variable=weights
#pragma HLS data_pack variable=output
#pragma HLS data_pack variable=input
#pragma HLS INTERFACE m_axi port=input offset=slave bundle=gmem1
#pragma HLS INTERFACE m_axi port=output offset=slave bundle=gmem2
#pragma HLS INTERFACE m_axi port=weights offset=slave bundle=gmem3
#pragma HLS INTERFACE m_axi port=bias offset=slave bundle=gmem4
#pragma HLS INTERFACE m_axi port=params offset=slave bundle=gmem6
#pragma HLS INTERFACE s_axilite port=input bundle=control
#pragma HLS INTERFACE s_axilite port=output bundle=control
#pragma HLS INTERFACE s_axilite port=weights bundle=control
#pragma HLS INTERFACE s_axilite port=bias bundle=control
#pragma HLS INTERFACE s_axilite port=params bundle=control
#pragma HLS INTERFACE s_axilite port=return bundle=control

  // Input tile buffer
  chalf16 inbuf[8 * 256 * 16];

  // Output buffer used for writing
  chalf16 outbuf[OCFACT][256];
#pragma HLS ARRAY_PARTITION variable=outbuf complete dim=1

  // Weight buffer
  chalf16 wbuf[OCFACT][256];
#pragma HLS ARRAY_PARTITION variable=wbuf complete dim=1

  // Bias buffer
  chalf biasbuf[1024];
DO_PRAGMA(HLS ARRAY_PARTITION variable=biasbuf cyclic factor=OCFACT)

  chalf multres[OCFACT][2][16];
#pragma HLS ARRAY_PARTITION variable=multres complete dim=1
#pragma HLS ARRAY_PARTITION variable=multres complete dim=2
#pragma HLS ARRAY_PARTITION variable=multres complete dim=3

  chalf weight_fw[16];
#pragma HLS ARRAY_PARTITION variable=weight_fw complete

  chalf weight_val[2][16];
#pragma HLS ARRAY_PARTITION variable=weight_val complete dim=1
#pragma HLS ARRAY_PARTITION variable=weight_val complete dim=2

  chalf in_val[2][16];
#pragma HLS ARRAY_PARTITION variable=in_val complete dim=1
#pragma HLS ARRAY_PARTITION variable=in_val complete dim=2

  chalf addres_s1[OCFACT][16];
#pragma HLS ARRAY_PARTITION variable=addres_s1 complete dim=1
#pragma HLS ARRAY_PARTITION variable=addres_s1 complete dim=2

  chalf addres_s2[OCFACT][4];
#pragma HLS ARRAY_PARTITION variable=addres_s2 complete dim=1
#pragma HLS ARRAY_PARTITION variable=addres_s2 complete dim=2

  chalf addres_s3[OCFACT][2];
#pragma HLS ARRAY_PARTITION variable=addres_s3 complete dim=1
#pragma HLS ARRAY_PARTITION variable=addres_s3 complete dim=2

  chalf finalOut[OCFACT][16];
#pragma HLS ARRAY_PARTITION variable=finalOut complete dim=1
#pragma HLS ARRAY_PARTITION variable=finalOut complete dim=2

  chalf addres_s4[OCFACT][2];
#pragma HLS ARRAY_PARTITION variable=addres_s4 complete dim=1
#pragma HLS ARRAY_PARTITION variable=addres_s4 complete dim=2

  chalf addres_f[OCFACT][16];
#pragma HLS ARRAY_PARTITION variable=addres_f complete dim=1
#pragma HLS ARRAY_PARTITION variable=addres_f complete dim=2

  chalf wUpdate[OCFACT][16];
#pragma HLS ARRAY_PARTITION variable=wUpdate complete dim=1
#pragma HLS ARRAY_PARTITION variable=wUpdate complete dim=2

  short inchannels = params[0];
  short outchannels = params[1];
  short burstchannels = params[2];
  short rpo = params[3];
  short rpofm = params[4];
  short burstydim = params[5];
  short ydim = params[6];
  short xdim = params[7];
  short xtile_pad = params[8];
  ap_uint<5> ksize = params[9];
  short numgroups = params[10];
  short numimages = params[11];
  short fc = params[12];
  short relu = params[13];
  short backward = params[14];
  short stride = params[15];
  short pad = params[16];
  bool mode = backward;

  short xdim_out = ((xdim - ksize + 2 * pad) / stride) + 1;
  short ydim_out = xdim_out;

  short fact = numimages >> 4;

  memcpy(biasbuf, bias, sizeof(chalf) * outchannels);
  short out_div = outchannels / OCFACT;
  short ofm_iters = (outchannels % OCFACT == 0) ? out_div : out_div + 1;
  short mac_iterations = ksize * ksize * fact * (burstchannels >> 1);

  for (int n = 0; n < rpo; ++n) {
    for (int y = 0; y < ydim_out; ++y) {
      for (int x = 0; x < xdim_out; ++x) {
        for (int p = 0; p < ksize; ++p) {
          for (int q = 0; q < ksize; ++q) {
            short in_y = y * stride - pad + p;
            short in_x = x * stride - pad + q;
            int in_idx = ((in_y * xdim + in_x) * inchannels +
                n * burstchannels) * fact;
            int inbuf_idx = (p * ksize + q) * burstchannels * fact;
            short in_size = burstchannels * fact;
            if (in_y >= 0 && in_y < ydim && in_x >= 0 && in_x < xdim) {
              if ((x != 0) && (stride == 1) && (q != ksize - 1)) {
                short q_off = burstchannels * fact;
                SHIFT_LOOP: for (int i = 0; i < in_size; ++i) {
#pragma HLS pipeline
#pragma HLS dependence variable=inbuf inter false
                  inbuf[i + inbuf_idx] = inbuf[i + inbuf_idx + q_off];
                }
              } else {
                memcpy(inbuf + inbuf_idx, input + in_idx, sizeof(chalf16) *
                    in_size);
              }
            } else {
              for (int i = 0; i < in_size; ++i) {
#pragma HLS pipeline
                inbuf[i + inbuf_idx] = chalf(0);
              }
            }
          }
        }
        for (int o = 0; o < ofm_iters; ++o) {
          if (n == 0 && !mode) {
            for (int i = 0; i < fact; ++i) {
#pragma HLS pipeline
              for (int k = 0; k < OCFACT; ++k) {
                outbuf[k][i] = biasbuf[o * OCFACT + k];
              }
            }
          } else {
            int out_idx;
            int out_size;
            for (int k = 0; k < OCFACT; ++k) {
              if (mode) {
                out_idx = (o * OCFACT + k) * ksize * ksize *
                  (inchannels >> 4) + n * ksize * ksize * (burstchannels >> 4);
                out_size = ksize * ksize * (burstchannels >> 4);
              } else {
                out_idx = ((y * xdim + x) * outchannels +
                    (o * OCFACT) + k) * (fact);
                out_size = fact;
              }
              memcpy(outbuf[k], output + out_idx, sizeof(chalf16) * out_size);
            }
          } 
          
          for (int k = 0; k < OCFACT; ++k) {
            int w_idx_f, w_idx_b, w_size_f, w_size_b, w_idx, w_size;
            w_idx_b = ((y * xdim_out + x) * outchannels +
                (o * OCFACT + k)) * (fact);
            w_size_b = fact;
            w_idx_f = (o * OCFACT + k) * ksize * ksize * (inchannels >> 4)
                + n * ksize * ksize * (burstchannels >> 4);
            w_size_f = ksize * ksize * (burstchannels >> 4);

            if (mode) {
              w_idx = w_idx_b;
              w_size = w_size_b;
            } else {
              w_idx = w_idx_f;
              w_size = w_size_f;
            }
            memcpy(wbuf[k], weights + w_idx, sizeof(chalf16) * w_size);
          }

          short w_off = 0;
          short img_off = 0;
          short iter = 0;
          short kdim_off = 0;
          short counter = 0;
          short counter_fw = 0;
          MAC_LOOP: for (int i = 0; i < mac_iterations; ++i, ++iter,
            ++counter) {
#pragma HLS pipeline
#pragma HLS DEPENDENCE variable outbuf inter false
#pragma HLS DEPENDENCE variable finalOut inter false
#pragma HLS DEPENDENCE variable wUpdate inter false
            if (counter == 8)
              counter = 0;
            if (!mode) {
              if (iter == (fact)) {
                if (counter_fw == 7)
                  counter_fw = 0;
                else
                  counter_fw++;
                if (w_off == (burstchannels >> 1) - 1) {
                  w_off = 0;
                  kdim_off++;       
                } else {
                  w_off++;
                }
                iter = 0;
              }
              img_off = iter;
            } else {
              if (iter == (burstchannels >> 1)) {
                if (kdim_off == ksize * ksize - 1) {
                  kdim_off = 0;
                  img_off++;
                } else {
                  kdim_off++;
                }
                iter = 0;
              }
              w_off = iter;
            }
            short w_idx_f = kdim_off * (burstchannels >> 4) + (w_off >> 3);
            short w_idx_b = img_off;
            short w_idx = (mode) ? w_idx_b : w_idx_f;
            short fout_idx = counter * 2;

            for (int k = 0; k < OCFACT; ++k) {
              weight_fw[0] = wbuf[k][w_idx].s0;
              weight_fw[1] = wbuf[k][w_idx].s1;
              weight_fw[2] = wbuf[k][w_idx].s2;
              weight_fw[3] = wbuf[k][w_idx].s3;   
              weight_fw[4] = wbuf[k][w_idx].s4;
              weight_fw[5] = wbuf[k][w_idx].s5;
              weight_fw[6] = wbuf[k][w_idx].s6;
              weight_fw[7] = wbuf[k][w_idx].s7;
              weight_fw[8] = wbuf[k][w_idx].s8;
              weight_fw[9] = wbuf[k][w_idx].s9;
              weight_fw[10] = wbuf[k][w_idx].sa;
              weight_fw[11] = wbuf[k][w_idx].sb;
              weight_fw[12] = wbuf[k][w_idx].sc;
              weight_fw[13] = wbuf[k][w_idx].sd;
              weight_fw[14] = wbuf[k][w_idx].se;
              weight_fw[15] = wbuf[k][w_idx].sf;
              for (int m = 0; m < 2; ++m) {
                if (mode) {
                  for (int j = 0; j < 16; ++j)
                    weight_val[m][j] = weight_fw[j];
                } else {
                  for (int j = 0; j < 16; ++j)
                    weight_val[m][j] = weight_fw[counter_fw * 2 + m];
                }

                short in_idx = (kdim_off * burstchannels + w_off * 2 + m) *
                  (fact) + img_off;
                in_val[m][0] = inbuf[in_idx].s0;
                in_val[m][1] = inbuf[in_idx].s1;
                in_val[m][2] = inbuf[in_idx].s2;
                in_val[m][3] = inbuf[in_idx].s3;
                in_val[m][4] = inbuf[in_idx].s4;
                in_val[m][5] = inbuf[in_idx].s5;
                in_val[m][6] = inbuf[in_idx].s6;
                in_val[m][7] = inbuf[in_idx].s7;
                in_val[m][8] = inbuf[in_idx].s8;
                in_val[m][9] = inbuf[in_idx].s9;
                in_val[m][10] = inbuf[in_idx].sa;
                in_val[m][11] = inbuf[in_idx].sb;
                in_val[m][12] = inbuf[in_idx].sc;
                in_val[m][13] = inbuf[in_idx].sd;
                in_val[m][14] = inbuf[in_idx].se;
                in_val[m][15] = inbuf[in_idx].sf;

                for (int j = 0; j < 16; ++j)
                  multres[k][m][j] = in_val[m][j] * weight_val[m][j];
              }

              for (int m = 0; m < 2; ++m) {
                for (int j = 0; j < 8; ++j) {
                  chalf temp1, temp2;
                  if (mode) {
                    temp1 = multres[k][m][j * 2];
                    temp2 = multres[k][m][j * 2 + 1];
                  } else {
                    temp1 = multres[k][0][m * 8 + j];
                    temp2 = multres[k][1][m * 8 + j];
                  }
                  addres_s1[k][m * 8 + j] = temp1 + temp2;
                }
              }

              for (int m = 0; m < 2; ++m) {
                for (int j = 0; j < 4; ++j)
                  addres_s2[k][j] = addres_s1[k][m * 8 + j * 2] +
                    addres_s1[k][m * 8 + j * 2 + 1];
                for (int j = 0; j < 2; ++j)
                  addres_s3[k][j] = addres_s2[k][j * 2] +
                    addres_s2[k][j * 2 + 1];
                addres_s4[k][m] = addres_s3[k][0] + addres_s3[k][1];
              }

              for (int m = 0; m < 2; ++m)
                wUpdate[k][fout_idx + m] = addres_s4[k][m];

              for (int j = 0; j < 16; ++j) {
                if (mode)
                  finalOut[k][j] = wUpdate[k][j];
                else
                  finalOut[k][j] = addres_s1[k][j];
              }

              short out_idx_f = img_off;
              short out_idx_b = kdim_off * (burstchannels >> 4) + (w_off >> 3);
              short out_idx = (mode) ? out_idx_b : out_idx_f;

              bool acc_enable = (mode) ? (counter == 7) : true;
              
              if (acc_enable) {
                outbuf[k][out_idx].s0 += finalOut[k][0];
                outbuf[k][out_idx].s1 += finalOut[k][1];
                outbuf[k][out_idx].s2 += finalOut[k][2];
                outbuf[k][out_idx].s3 += finalOut[k][3];
                outbuf[k][out_idx].s4 += finalOut[k][4];
                outbuf[k][out_idx].s5 += finalOut[k][5];
                outbuf[k][out_idx].s6 += finalOut[k][6];
                outbuf[k][out_idx].s7 += finalOut[k][7];
                outbuf[k][out_idx].s8 += finalOut[k][8];
                outbuf[k][out_idx].s9 += finalOut[k][9];
                outbuf[k][out_idx].sa += finalOut[k][10];
                outbuf[k][out_idx].sb += finalOut[k][11];
                outbuf[k][out_idx].sc += finalOut[k][12];
                outbuf[k][out_idx].sd += finalOut[k][13];
                outbuf[k][out_idx].se += finalOut[k][14];
                outbuf[k][out_idx].sf += finalOut[k][15];
              } 
            }
          }
          for (int k = 0; k < OCFACT; ++k) {
            int out_idx;
            int out_size;
            if (mode) {
              out_idx = (o * OCFACT + k) * ksize * ksize *
                (inchannels >> 4) + n * ksize * ksize * (burstchannels >> 4);
              out_size = ksize * ksize * (burstchannels >> 4);
            } else {
              out_idx = (((y * xdim_out) + x) * outchannels +
                (o * OCFACT) + k) * (fact);
              out_size = fact;
            }
            if (o * OCFACT + k < outchannels)
              memcpy(output + out_idx, outbuf[k], sizeof(chalf16) * out_size);
          }
        }
      }
    }
  }
}

}
