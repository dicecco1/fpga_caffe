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

chalf relu_bw(chalf input, bool enable) {
#pragma HLS INLINE off
  chalf res = (enable) ? input : chalf(0);
  return res;
}
void relu_fw(chalf16 outbuf[256], short outbuf_relu[256], chalf acc[16],
  int out_idx, bool reluf) {
#pragma HLS inline
  chalf16 val;
  val.s0 = max(acc[0], reluf);
  val.s1 = max(acc[1], reluf);
  val.s2 = max(acc[2], reluf);
  val.s3 = max(acc[3], reluf);
  val.s4 = max(acc[4], reluf);
  val.s5 = max(acc[5], reluf);
  val.s6 = max(acc[6], reluf);
  val.s7 = max(acc[7], reluf);
  val.s8 = max(acc[8], reluf);
  val.s9 = max(acc[9], reluf);
  val.sa = max(acc[10], reluf);
  val.sb = max(acc[11], reluf);
  val.sc = max(acc[12], reluf);
  val.sd = max(acc[13], reluf);
  val.se = max(acc[14], reluf);
  val.sf = max(acc[15], reluf);

  outbuf[out_idx] = val;

  outbuf_relu[out_idx] = 0;
  outbuf_relu[out_idx] |= (val.s0 != chalf(0)) ? 1 << 0 : 0;
  outbuf_relu[out_idx] |= (val.s1 != chalf(0)) ? 1 << 1 : 0;
  outbuf_relu[out_idx] |= (val.s2 != chalf(0)) ? 1 << 2 : 0;
  outbuf_relu[out_idx] |= (val.s3 != chalf(0)) ? 1 << 3 : 0;
  outbuf_relu[out_idx] |= (val.s4 != chalf(0)) ? 1 << 4 : 0;
  outbuf_relu[out_idx] |= (val.s5 != chalf(0)) ? 1 << 5 : 0;
  outbuf_relu[out_idx] |= (val.s6 != chalf(0)) ? 1 << 6 : 0;
  outbuf_relu[out_idx] |= (val.s7 != chalf(0)) ? 1 << 7 : 0;
  outbuf_relu[out_idx] |= (val.s8 != chalf(0)) ? 1 << 8 : 0;
  outbuf_relu[out_idx] |= (val.s9 != chalf(0)) ? 1 << 9 : 0;
  outbuf_relu[out_idx] |= (val.sa != chalf(0)) ? 1 << 10 : 0;
  outbuf_relu[out_idx] |= (val.sb != chalf(0)) ? 1 << 11 : 0;
  outbuf_relu[out_idx] |= (val.sc != chalf(0)) ? 1 << 12 : 0;
  outbuf_relu[out_idx] |= (val.sd != chalf(0)) ? 1 << 13 : 0;
  outbuf_relu[out_idx] |= (val.se != chalf(0)) ? 1 << 14 : 0;
  outbuf_relu[out_idx] |= (val.sf != chalf(0)) ? 1 << 15 : 0;
}

extern "C" {

void cr_layer_hwcn_half(chalf16 *input, chalf16 *weights, chalf *bias,
    chalf16 *output, short *track_relu, int *params, int group_idx) { 
// Ports 
#pragma HLS data_pack variable=weights
#pragma HLS data_pack variable=output
#pragma HLS data_pack variable=input
#pragma HLS INTERFACE m_axi port=input offset=slave bundle=gmem1
#pragma HLS INTERFACE m_axi port=output offset=slave bundle=gmem2
#pragma HLS INTERFACE m_axi port=weights offset=slave bundle=gmem3
#pragma HLS INTERFACE m_axi port=bias offset=slave bundle=gmem4
#pragma HLS INTERFACE m_axi port=track_relu offset=slave bundle=gmem5
#pragma HLS INTERFACE m_axi port=params offset=slave bundle=gmem6
#pragma HLS INTERFACE s_axilite port=input bundle=control
#pragma HLS INTERFACE s_axilite port=output bundle=control
#pragma HLS INTERFACE s_axilite port=weights bundle=control
#pragma HLS INTERFACE s_axilite port=bias bundle=control
#pragma HLS INTERFACE s_axilite port=track_relu bundle=control
#pragma HLS INTERFACE s_axilite port=params bundle=control
#pragma HLS INTERFACE s_axilite port=group_idx bundle=control
#pragma HLS INTERFACE s_axilite port=return bundle=control

  // Input tile buffer
  chalf16 inbuf[4][2 * 256 * 16];
#pragma HLS ARRAY_PARTITION variable=inbuf complete dim=1

  short inbuf_relu[4][2 * 256 * 16];
#pragma HLS ARRAY_PARTITION variable=inbuf_relu complete dim=1

  short outbuf_relu[OCFACT][256];
#pragma HLS ARRAY_PARTITION variable=outbuf_relu complete dim=1

  // Output buffer used for writing
  chalf16 outbuf[OCFACT][256];
#pragma HLS ARRAY_PARTITION variable=outbuf complete dim=1

  // Weight buffer
  chalf16 wbuf[2][OCFACT][256];
#pragma HLS ARRAY_PARTITION variable=wbuf complete dim=1
#pragma HLS ARRAY_PARTITION variable=wbuf complete dim=2

  // Bias buffer
  chalf biasbuf[1024];
DO_PRAGMA(HLS ARRAY_PARTITION variable=biasbuf cyclic factor=OCFACT)

  chalf multres[OCFACT][4][16];
#pragma HLS ARRAY_PARTITION variable=multres complete dim=1
#pragma HLS ARRAY_PARTITION variable=multres complete dim=2
#pragma HLS ARRAY_PARTITION variable=multres complete dim=3

  chalf weight_fw[16];
#pragma HLS ARRAY_PARTITION variable=weight_fw complete

  chalf weight_val[4][16];
#pragma HLS ARRAY_PARTITION variable=weight_val complete dim=1
#pragma HLS ARRAY_PARTITION variable=weight_val complete dim=2

  chalf in_val[4][16];
#pragma HLS ARRAY_PARTITION variable=in_val complete dim=1
#pragma HLS ARRAY_PARTITION variable=in_val complete dim=2

  chalf addres_s1[OCFACT][32];
#pragma HLS ARRAY_PARTITION variable=addres_s1 complete dim=1
#pragma HLS ARRAY_PARTITION variable=addres_s1 complete dim=2

  chalf addres_s2[OCFACT][16];
#pragma HLS ARRAY_PARTITION variable=addres_s2 complete dim=1
#pragma HLS ARRAY_PARTITION variable=addres_s2 complete dim=2

  chalf addres_s3[OCFACT][4][2];
#pragma HLS ARRAY_PARTITION variable=addres_s3 complete dim=1
#pragma HLS ARRAY_PARTITION variable=addres_s3 complete dim=2
#pragma HLS ARRAY_PARTITION variable=addres_s3 complete dim=3

  chalf finalOut[OCFACT][16];
#pragma HLS ARRAY_PARTITION variable=finalOut complete dim=1
#pragma HLS ARRAY_PARTITION variable=finalOut complete dim=2

  chalf addres_s4[OCFACT][4];
#pragma HLS ARRAY_PARTITION variable=addres_s4 complete dim=1
#pragma HLS ARRAY_PARTITION variable=addres_s4 complete dim=2

  chalf addres_f[OCFACT][16];
#pragma HLS ARRAY_PARTITION variable=addres_f complete dim=1
#pragma HLS ARRAY_PARTITION variable=addres_f complete dim=2

  chalf wUpdate[OCFACT][16];
#pragma HLS ARRAY_PARTITION variable=wUpdate complete dim=1
#pragma HLS ARRAY_PARTITION variable=wUpdate complete dim=2

  chalf acc[OCFACT][16];
#pragma HLS ARRAY_PARTITION variable=acc complete dim=1
#pragma HLS ARRAY_PARTITION variable=acc complete dim=2

  short rfw[16];
#pragma HLS ARRAY_PARTITION variable=rfw complete dim=1

  short inchannels = params[0];
  short outchannels = params[1];
  short burstchannels = params[2];
  short rpo = params[3];
  short rpofm = params[4];
  ap_uint<10> burstydim = params[5];
  ap_uint<10> ydim = params[6];
  ap_uint<10> xdim = params[7];
  ap_uint<10> xtile_pad = params[8];
  ap_uint<5> ksize = params[9];
  short numgroups = params[10];
  ap_uint<10> numimages = params[11];
  short fc = params[12];
  short relu = params[13];
  short backward = params[14];
  ap_uint<4> stride = params[15];
  ap_uint<4> pad = params[16];
  bool mode = backward;

  ap_uint<10> xdim_out = ((xdim - ksize + 2 * pad) / stride) + 1;
  ap_uint<10> ydim_out = xdim_out;

  ap_uint<8> img_fact = numimages >> 4;
  short burst_fact = burstchannels >> 2;
  short ic_fact = (inchannels % 16 == 0) ? (inchannels >> 4) :
    (inchannels >> 4) + 1;
  short wc_fact = (burstchannels % 16 == 0) ? (burstchannels >> 4) :
    (burstchannels >> 4) + 1;
  int bias_offset = outchannels * group_idx;
  memcpy(biasbuf, bias + bias_offset, sizeof(chalf) * outchannels);
  short out_div = outchannels / OCFACT;
  short ofm_iters = (outchannels % OCFACT == 0) ? out_div : out_div + 1;
  
  for (int n = 0; n < rpo; ++n) {
    for (int y = 0; y < ydim_out; ++y) {
      for (int x = 0; x < xdim_out; ++x) {
        ap_uint<8> yk_off = 0;
        ap_uint<8> xk_off = 0;
        ap_uint<8> yksize = 0;
        ap_uint<8> xksize = 0;
        bool xkset = false;
        bool ykset = false;
        for (int p = 0; p < ksize; ++p) {
          for (int q = 0; q < ksize; ++q) {
            short in_y = y * stride - pad + p;
            short in_x = x * stride - pad + q;
            int in_idx = (((in_y * xdim + in_x) * numgroups + group_idx) *
                inchannels + n * burstchannels) * img_fact;
            int inbuf_idx = (p * ksize + q) * burst_fact * img_fact;
            short in_size = burst_fact * img_fact;

            if (in_y >= 0 && in_y < ydim) {
              if (q == 0)
                yksize++;
              if (yk_off == 0 && !ykset) {
                yk_off = p;
                ykset = true;
              }
            }
            if (in_x >= 0 && in_x < xdim) {
              if (p == 0)
                xksize++;
              if (xk_off == 0 && !xkset) {
                xk_off = q;
                xkset = true;
              }
            }

            if (in_y >= 0 && in_y < ydim && in_x >= 0 && in_x < xdim) {
              if ((x != 0) && (stride == 1) && (q != ksize - 1)) {
                short q_off = burst_fact * img_fact;
                SHIFT_LOOP: for (int i = 0; i < in_size; ++i) {
#pragma HLS pipeline
#pragma HLS dependence variable=inbuf inter false
#pragma HLS dependence variable=inbuf_relu inter false
                  for (int j = 0; j < 4; ++j) {
                    inbuf[j][i + inbuf_idx] = inbuf[j][i + inbuf_idx + q_off];
                    if (backward != 0)
                      inbuf_relu[j][i + inbuf_idx] =
                        inbuf_relu[j][i + inbuf_idx + q_off];
                  }
                }
              } else {
                for (int j = 0; j < 4; ++j) {
                  int f_in_idx = in_idx + j * burst_fact * img_fact;
                  memcpy(inbuf[j] + inbuf_idx, input + f_in_idx,
                      sizeof(chalf16) * in_size);
                  if (backward != 0)
                    memcpy(inbuf_relu[j] + inbuf_idx, track_relu + f_in_idx,
                        sizeof(short) * in_size);
                }
              }
            }
          }
        }

        for (int k = 0; k < OCFACT; ++k) {
          int o = 0;
          int w_idx_f, w_idx_b, w_idx;
          short w_size_f, w_size_b, w_size;
          w_idx_b = (((y * xdim_out + x) * numgroups + group_idx) *
              outchannels + (o * OCFACT + k)) * img_fact;
          w_size_b = img_fact;
          w_idx_f = (o * OCFACT + k + outchannels * group_idx) * ksize *
            ksize * ic_fact + n * ksize * ksize * wc_fact;
          w_size_f = ksize * ksize * wc_fact;

          if (mode) {
            w_idx = w_idx_b;
            w_size = w_size_b;
          } else {
            w_idx = w_idx_f;
            w_size = w_size_f;
          }
          if (o * OCFACT + k < outchannels)
            memcpy(wbuf[0][k], weights + w_idx, sizeof(chalf16) * w_size);
        }

        for (int o = 0; o < ofm_iters; ++o) {
          if (n == 0 && !mode) {
            for (int i = 0; i < img_fact; ++i) {
#pragma HLS pipeline
              for (int k = 0; k < OCFACT; ++k) {
                outbuf[k][i] = biasbuf[o * OCFACT + k];
              }
            }
          } else {
            for (int k = 0; k < OCFACT; ++k) {
              int out_idx, out_idx_f, out_idx_b;
              short out_size, out_size_f, out_size_b;
              out_idx_b = (o * OCFACT + k + outchannels * group_idx) * ksize *
                ksize * ic_fact + n * ksize * ksize * wc_fact;
              out_size_b = ksize * ksize * wc_fact;
              out_idx_f = (((y * xdim_out + x) * numgroups + group_idx) *
                outchannels + (o * OCFACT) + k) * img_fact;
              out_size_f = img_fact;

              if (mode) {
                out_idx = out_idx_b;
                out_size = out_size_b;
              } else {
                out_idx = out_idx_f;
                out_size = out_size_f;
              }
              if (o * OCFACT + k < outchannels)
                memcpy(outbuf[k], output + out_idx, sizeof(chalf16) *
                    out_size);
            }
          }           

          ap_uint<8> w_off = 0;
          ap_uint<5> img_off = 0;
          ap_uint<8> iter = 0;
          ap_uint<8> xdim_off = 0;
          ap_uint<8> ydim_off = 0;
          ap_uint<2> counter_bw = 0;
          ap_uint<2> counter_fw = 0;
          short mac_iterations = yksize * xksize * img_fact * burst_fact;
          MAC_LOOP: for (int i = 0; i < mac_iterations; ++i, ++iter,
            ++counter_bw) {
#pragma HLS pipeline
#pragma HLS DEPENDENCE variable outbuf inter false
#pragma HLS DEPENDENCE variable outbuf_relu inter false
#pragma HLS DEPENDENCE variable finalOut inter false
#pragma HLS DEPENDENCE variable wUpdate inter false
            if (!mode) {
              if (iter == img_fact) {
                counter_fw++;
                if (w_off == burst_fact - 1) {
                  w_off = 0;
                  if (xdim_off == xksize - 1) {
                    xdim_off = 0;
                    ydim_off++;
                  } else {
                    xdim_off++;
                  }
                } else {
                  w_off++;
                }
                iter = 0;
              }
              img_off = iter;
            } else {
              if (iter == burst_fact) {
                if (xdim_off == xksize - 1) {
                  xdim_off = 0;
                  if (ydim_off == yksize - 1) {
                    ydim_off = 0;
                    img_off++;
                  } else {
                    ydim_off++;
                  }
                } else {
                  xdim_off++;
                }
                iter = 0;
              }
              w_off = iter;
            }

            short filt_off = (yk_off + ydim_off) * ksize + xk_off + xdim_off;
            short w_idx_f = filt_off * wc_fact + (w_off >> 2);
            short w_idx_b = img_off;
            short w_idx = (mode) ? w_idx_b : w_idx_f;
            short fout_idx = counter_bw * 4;
            short in_idx = (filt_off * burst_fact + w_off) * img_fact
              + img_off;
            short out_idx_f = img_off;
            short out_idx_b = filt_off * wc_fact + (w_off >> 2);
            short out_idx = (mode) ? out_idx_b : out_idx_f;
            bool acc_enable = (mode) ? (counter_bw == 3) : true;
            bool reluf = (!mode) && relu && (n == rpo - 1) &&
                (xdim_off == xksize - 1) && (ydim_off == yksize - 1) &&
                (w_off == burst_fact - 1);
            bool relub = (relu && ((backward == 1) ||
                (backward == 2) || (backward == 3)));

            for (int k = 0; k < OCFACT; ++k) {
              weight_fw[0] = wbuf[o & 0x1][k][w_idx].s0;
              weight_fw[1] = wbuf[o & 0x1][k][w_idx].s1;
              weight_fw[2] = wbuf[o & 0x1][k][w_idx].s2;
              weight_fw[3] = wbuf[o & 0x1][k][w_idx].s3;   
              weight_fw[4] = wbuf[o & 0x1][k][w_idx].s4;
              weight_fw[5] = wbuf[o & 0x1][k][w_idx].s5;
              weight_fw[6] = wbuf[o & 0x1][k][w_idx].s6;
              weight_fw[7] = wbuf[o & 0x1][k][w_idx].s7;
              weight_fw[8] = wbuf[o & 0x1][k][w_idx].s8;
              weight_fw[9] = wbuf[o & 0x1][k][w_idx].s9;
              weight_fw[10] = wbuf[o & 0x1][k][w_idx].sa;
              weight_fw[11] = wbuf[o & 0x1][k][w_idx].sb;
              weight_fw[12] = wbuf[o & 0x1][k][w_idx].sc;
              weight_fw[13] = wbuf[o & 0x1][k][w_idx].sd;
              weight_fw[14] = wbuf[o & 0x1][k][w_idx].se;
              weight_fw[15] = wbuf[o & 0x1][k][w_idx].sf;
              for (int m = 0; m < 4; ++m) {
                for (int j = 0; j < 16; ++j) {
                  if (mode)
                    weight_val[m][j] = weight_fw[j];
                  else
                    weight_val[m][j] = weight_fw[counter_fw * 4 + m];
                }

                short relu_val = inbuf_relu[m][in_idx];
                bool fw_mode = (backward == 0);
                bool relu0 = (relub && ((relu_val >> 0) & 0x1)) || fw_mode;
                bool relu1 = (relub && ((relu_val >> 1) & 0x1)) || fw_mode;
                bool relu2 = (relub && ((relu_val >> 2) & 0x1)) || fw_mode;
                bool relu3 = (relub && ((relu_val >> 3) & 0x1)) || fw_mode;
                bool relu4 = (relub && ((relu_val >> 4) & 0x1)) || fw_mode;
                bool relu5 = (relub && ((relu_val >> 5) & 0x1)) || fw_mode;
                bool relu6 = (relub && ((relu_val >> 6) & 0x1)) || fw_mode;
                bool relu7 = (relub && ((relu_val >> 7) & 0x1)) || fw_mode;
                bool relu8 = (relub && ((relu_val >> 8) & 0x1)) || fw_mode;
                bool relu9 = (relub && ((relu_val >> 9) & 0x1)) || fw_mode;
                bool relu10 = (relub && ((relu_val >> 10) & 0x1)) || fw_mode;
                bool relu11 = (relub && ((relu_val >> 11) & 0x1)) || fw_mode;
                bool relu12 = (relub && ((relu_val >> 12) & 0x1)) || fw_mode;
                bool relu13 = (relub && ((relu_val >> 13) & 0x1)) || fw_mode;
                bool relu14 = (relub && ((relu_val >> 14) & 0x1)) || fw_mode;
                bool relu15 = (relub && ((relu_val >> 15) & 0x1)) || fw_mode; 

                in_val[m][0] = relu_bw(inbuf[m][in_idx].s0, relu0);
                in_val[m][1] = relu_bw(inbuf[m][in_idx].s1, relu1);
                in_val[m][2] = relu_bw(inbuf[m][in_idx].s2, relu2);
                in_val[m][3] = relu_bw(inbuf[m][in_idx].s3, relu3);
                in_val[m][4] = relu_bw(inbuf[m][in_idx].s4, relu4);
                in_val[m][5] = relu_bw(inbuf[m][in_idx].s5, relu5);
                in_val[m][6] = relu_bw(inbuf[m][in_idx].s6, relu6);
                in_val[m][7] = relu_bw(inbuf[m][in_idx].s7, relu7);
                in_val[m][8] = relu_bw(inbuf[m][in_idx].s8, relu8);
                in_val[m][9] = relu_bw(inbuf[m][in_idx].s9, relu9);
                in_val[m][10] = relu_bw(inbuf[m][in_idx].sa, relu10);
                in_val[m][11] = relu_bw(inbuf[m][in_idx].sb, relu11);
                in_val[m][12] = relu_bw(inbuf[m][in_idx].sc, relu12);
                in_val[m][13] = relu_bw(inbuf[m][in_idx].sd, relu13);
                in_val[m][14] = relu_bw(inbuf[m][in_idx].se, relu14);
                in_val[m][15] = relu_bw(inbuf[m][in_idx].sf, relu15);

                for (int j = 0; j < 16; ++j) 
                  multres[k][m][j] = in_val[m][j] * weight_val[m][j];
              }

              for (int off = 0; off < 2; ++off) {
                for (int m = 0; m < 2; ++m) {
                  for (int j = 0; j < 8; ++j) {
                    chalf temp1, temp2;
                    if (mode) {
                      temp1 = multres[k][off * 2 + m][j * 2];
                      temp2 = multres[k][off * 2 + m][j * 2 + 1];
                    } else {
                      temp1 = multres[k][off * 2 + 0][m * 8 + j];
                      temp2 = multres[k][off * 2 + 1][m * 8 + j];
                    }
                    addres_s1[k][(off * 2 + m) * 8 + j] = temp1 + temp2;
                  }
                }
              }
              for (int off = 0; off < 2; ++off) {
                for (int m = 0; m < 2; ++m) {
                  for (int j = 0; j < 4; ++j) {
                    chalf temp1, temp2;
                    if (mode) {
                      temp1 = addres_s1[k][(off * 2 + m) * 8 + j * 2];
                      temp2 = addres_s1[k][(off * 2 + m) * 8 + j * 2 + 1];
                    } else {
                      temp1 = addres_s1[k][(off * 2 + m) * 4 + j];
                      temp2 = addres_s1[k][(off * 2 + m) * 4 + j + 16];
                    }
                    addres_s2[k][(off * 2 + m) * 4 + j] = temp1 + temp2;
                  }
                }
              }

              for (int m = 0; m < 4; ++m) {
                for (int j = 0; j < 2; ++j)
                  addres_s3[k][m][j] = addres_s2[k][m * 4 + j * 2] +
                    addres_s2[k][m * 4 + j * 2 + 1];
                addres_s4[k][m] = addres_s3[k][m][0] + addres_s3[k][m][1];
              }

              for (int m = 0; m < 4; ++m)
                wUpdate[k][fout_idx + m] = addres_s4[k][m];

              for (int j = 0; j < 16; ++j) {
                if (mode)
                  finalOut[k][j] = wUpdate[k][j];
                else
                  finalOut[k][j] = addres_s2[k][j];
              }
              
              if (acc_enable) {
                short relu_out = 0;
                acc[k][0] = outbuf[k][out_idx].s0 + finalOut[k][0];
                acc[k][1] = outbuf[k][out_idx].s1 + finalOut[k][1];
                acc[k][2] = outbuf[k][out_idx].s2 + finalOut[k][2];
                acc[k][3] = outbuf[k][out_idx].s3 + finalOut[k][3];
                acc[k][4] = outbuf[k][out_idx].s4 + finalOut[k][4];
                acc[k][5] = outbuf[k][out_idx].s5 + finalOut[k][5];
                acc[k][6] = outbuf[k][out_idx].s6 + finalOut[k][6];
                acc[k][7] = outbuf[k][out_idx].s7 + finalOut[k][7];
                acc[k][8] = outbuf[k][out_idx].s8 + finalOut[k][8];
                acc[k][9] = outbuf[k][out_idx].s9 + finalOut[k][9];
                acc[k][10] = outbuf[k][out_idx].sa + finalOut[k][10];
                acc[k][11] = outbuf[k][out_idx].sb + finalOut[k][11];
                acc[k][12] = outbuf[k][out_idx].sc + finalOut[k][12];
                acc[k][13] = outbuf[k][out_idx].sd + finalOut[k][13];
                acc[k][14] = outbuf[k][out_idx].se + finalOut[k][14];
                acc[k][15] = outbuf[k][out_idx].sf + finalOut[k][15];
                relu_fw(outbuf[k], outbuf_relu[k], acc[k], out_idx, reluf);
              } 
            }
          }
          for (int k = 0; k < OCFACT; ++k) {
            int w_idx_f, w_idx_b, w_idx;
            short w_size_f, w_size_b, w_size;
            w_idx_b = (((y * xdim_out + x) * numgroups + group_idx) *
                outchannels + ((o + 1) * OCFACT + k)) * img_fact;
            w_size_b = img_fact;
            w_idx_f = ((o + 1) * OCFACT + k + outchannels * group_idx) * ksize
              * ksize * ic_fact + n * ksize * ksize * wc_fact;
            w_size_f = ksize * ksize * wc_fact;

            if (mode) {
              w_idx = w_idx_b;
              w_size = w_size_b;
            } else {
              w_idx = w_idx_f;
              w_size = w_size_f;
            }
            if ((o + 1) * OCFACT + k < outchannels && (o + 1 < ofm_iters))
              memcpy(wbuf[((o + 1) & 0x1)][k], weights + w_idx,
                sizeof(chalf16) * w_size);
          }
          for (int k = 0; k < OCFACT; ++k) {
            int out_idx, out_idx_f, out_idx_b;
            short out_size, out_size_f, out_size_b;
            out_idx_b = (o * OCFACT + k + outchannels * group_idx) * ksize *
              ksize * ic_fact + n * ksize * ksize * wc_fact;
            out_size_b = ksize * ksize * wc_fact;
            out_idx_f = (((y * xdim_out + x) * numgroups + group_idx) *
                outchannels + (o * OCFACT) + k) * img_fact;
            out_size_f = img_fact;

            if (mode) {
              out_idx = out_idx_b;
              out_size = out_size_b;
            } else {
              out_idx = out_idx_f;
              out_size = out_size_f;
            }

            if (relu && (o * OCFACT + k < outchannels) && (!mode) &&
                (n == rpo - 1)) {
              memcpy(track_relu + out_idx, outbuf_relu[k], sizeof(short) *
                  out_size);
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
