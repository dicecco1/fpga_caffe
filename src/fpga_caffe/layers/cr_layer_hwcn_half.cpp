#include <stdio.h>
#include <assert.h>
#include <string.h>
#include <stdbool.h>

#include "../../../include/fpga_caffe/layer.hpp"
#include "../../../include/fpga_caffe/half.hpp"

#define HADD_LATENCY 12 
#define OCFACT 1 
/* chalf16 data type definition */

typedef struct {
  char s0;
  char s1;
  char s2;
  char s3;
  char s4;
  char s5;
  char s6;
  char s7;
  char s8;
  char s9;
  char sa;
  char sb;
  char sc;
  char sd;
  char se;
  char sf;
} char16;

typedef struct {
  chalf s0;
  chalf s1;
  chalf s2;
  chalf s3;
  chalf s4;
  chalf s5;
  chalf s6;
  chalf s7;
  chalf s8;
  chalf s9;
  chalf sa;
  chalf sb;
  chalf sc;
  chalf sd;
  chalf se;
  chalf sf;
} chalf16;

void cfp_convert(bool mode, chalf ot[16][3], ap_uint<EXP_SIZE> exp_out_f[16],
    ap_uint<EXP_SIZE> exp_out_b[3], ap_int<SHIFT_SIZE> out_vals[16][3]) {
#pragma HLS pipeline
	ap_uint<EXP_SIZE> exp_array[16][3];
#pragma HLS ARRAY_PARTITION variable=exp_array complete dim=1
#pragma HLS ARRAY_PARTITION variable=exp_array complete dim=2
	ap_uint<1> sign_array[16][3];
#pragma HLS ARRAY_PARTITION variable=sign_array complete dim=1
#pragma HLS ARRAY_PARTITION variable=sign_array complete dim=2
	ap_int<SHIFT_SIZE> mant_array[16][3];
#pragma HLS ARRAY_PARTITION variable=mant_array complete dim=1
#pragma HLS ARRAY_PARTITION variable=mant_array complete dim=2

	for (int i = 0; i < 16; ++i) {
		for (int j = 0; j < 3; ++j) {
			exp_array[i][j] = ot[i][j].getdata() >> EXP_SHIFT;
			if (exp_array[i][j] != 0)
				mant_array[i][j] = (ot[i][j].getdata() & MANT_MASK) | MANT_NORM;
			else
				mant_array[i][j] = 0;
			sign_array[i][j] = ot[i][j].getdata() >> SIGN_SHIFT;
		}
    ap_uint<EXP_SIZE> temp_exp;
		if (exp_array[i][0] > exp_array[i][1])
			temp_exp = exp_array[i][0];
		else
			temp_exp = exp_array[i][1];
		if (exp_array[i][2] > temp_exp)
			temp_exp = exp_array[i][2];
		exp_out_f[i] = temp_exp;
	}

	ap_uint<EXP_SIZE> exp_r1[8];
#pragma HLS ARRAY_PARTITION variable=exp_r1 complete dim=1
	ap_uint<EXP_SIZE> exp_r2[4];
#pragma HLS ARRAY_PARTITION variable=exp_r2 complete dim=1
	ap_uint<EXP_SIZE> exp_r3[2];
#pragma HLS ARRAY_PARTITION variable=exp_r3 complete dim=1

	for (int i = 0; i < 3; ++i) {
		for (int j = 0; j < 8; ++j) {
			if (exp_array[j * 2 + 0][i] > exp_array[j * 2 + 1][i])
				exp_r1[j] = exp_array[j * 2 + 0][i];
			else
				exp_r1[j] = exp_array[j * 2 + 1][i];
		}
		for (int j = 0; j < 4; ++j) {
			if (exp_r1[j * 2 + 0] > exp_r1[j * 2 + 1])
				exp_r2[j] = exp_r1[j * 2 + 0];
			else
				exp_r2[j] = exp_r1[j * 2 + 1];
		}
		for (int j = 0; j < 2; ++j) {
			if (exp_r2[j * 2 + 0] > exp_r2[j * 2 + 1])
				exp_r3[j] = exp_r2[j * 2 + 0];
			else
				exp_r3[j] = exp_r2[j * 2 + 1];
		}
		if (exp_r3[0] > exp_r3[1])
			exp_out_b[i] = exp_r3[0];
		else
			exp_out_b[i] = exp_r3[1];
	}

	for (int i = 0; i < 16; ++i) {
		for (int j = 0; j < 3; ++j) {
			ap_uint<6> diff;
			if (mode) {
// backward mode
				diff = exp_out_b[j] - exp_array[i][j];
			} else {
// forward mode
				diff = exp_out_f[i] - exp_array[i][j];
			}
			if (sign_array[i][j])
				out_vals[i][j] = ((mant_array[i][j] << (MANT_SIZE + 1)) >> diff) * -1;
			else
				out_vals[i][j] = ((mant_array[i][j] << (MANT_SIZE + 1)) >> diff);
		}
	}
}
void adder_tree(bool mode, ap_int<SHIFT_SIZE> ot[16][3],
    ap_int<SHIFT_SIZE> otForward[16],
    ap_int<SHIFT_SIZE> otBackward[3]) {
  
  ap_int<SHIFT_SIZE> ot_s1[8][3];
#pragma HLS ARRAY_PARTITION variable=ot_s1 complete dim=1
#pragma HLS ARRAY_PARTITION variable=ot_s1 complete dim=2

  ap_int<SHIFT_SIZE> ot_s2[4][3];
#pragma HLS ARRAY_PARTITION variable=ot_s2 complete dim=1
#pragma HLS ARRAY_PARTITION variable=ot_s2 complete dim=2
  ap_int<SHIFT_SIZE> ot_s3[2][3];
#pragma HLS ARRAY_PARTITION variable=ot_s3 complete dim=1
#pragma HLS ARRAY_PARTITION variable=ot_s3 complete dim=2
 
#pragma HLS pipeline
  ap_int<SHIFT_SIZE> temp, temp1, temp2;
  for (int q = 0; q < 2; ++q) {
    for (int p = 0; p < 8; ++p) {
      if (mode) {
        temp1 = ot[p][q];
        temp2 = ot[p + 8][q];
      } else {
        temp1 = ot[p + q * 8][0];
        temp2 = ot[p + q * 8][1] + ot[p + q * 8][2];
#pragma HLS RESOURCE variable=temp2 core=AddSub_DSP
      }
      temp = temp1 + temp2;
#pragma HLS RESOURCE variable=temp core=AddSub_DSP
      ot_s1[p][q] = temp;
      otForward[p + q * 8] = temp;
    }
  }
  for (int p = 0; p < 8; ++p) {
    temp = ot[p][2] + ot[p + 8][2];
    ot_s1[p][2] = temp;
#pragma HLS RESOURCE variable=temp core=AddSub_DSP
  }

  for (int q = 0; q < 3; ++q) {
    for (int p = 0; p < 4; ++p) {
      temp = ot_s1[p][q] + ot_s1[p + 4][q];
      ot_s2[p][q] = temp;
#pragma HLS RESOURCE variable=temp core=AddSub_DSP
    }
  }

  for (int q = 0; q < 3; ++q) {
    for (int p = 0; p < 2; ++p) {
      temp = ot_s2[p][q] + ot_s2[p + 2][q];
      ot_s3[p][q] = temp;
#pragma HLS RESOURCE variable=temp core=AddSub_DSP
    }
  }

  for (int q = 0; q < 3; ++q) {
    temp = ot_s3[0][q] + ot_s3[1][q];
    otBackward[q] = temp;
#pragma HLS RESOURCE variable=temp core=AddSub_DSP
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
#pragma HLS ARRAY_PARTITION variable=inbuf cyclic dim=1 factor=2
  // Output buffer used for writing
  chalf16 outbuf[OCFACT][512];
#pragma HLS ARRAY_PARTITION variable=outbuf complete dim=1

  // Weight buffer
  chalf16 wbuf[OCFACT][512];
#pragma HLS ARRAY_PARTITION variable=wbuf complete dim=1

  // Bias buffer
  chalf biasbuf[1024];
#pragma HLS ARRAY_PARTITION variable=biasbuf cyclic factor=8

  chalf multres[2][OCFACT][16];
#pragma HLS ARRAY_PARTITION variable=multres complete dim=1
#pragma HLS ARRAY_PARTITION variable=multres complete dim=2
#pragma HLS ARRAY_PARTITION variable=multres complete dim=3

  chalf weight_fw[16];
#pragma HLS ARRAY_PARTITION variable=weight_fw complete

  chalf weight_val[16];
#pragma HLS ARRAY_PARTITION variable=weight_val complete

  chalf addres_s1[8];
#pragma HLS ARRAY_PARTITION variable=addres_s1 complete

  chalf addres_s2[4];
#pragma HLS ARRAY_PARTITION variable=addres_s2 complete

  chalf addres_s3[2];
#pragma HLS ARRAY_PARTITION variable=addres_s3 complete

  chalf finalOut[OCFACT][16];
#pragma HLS ARRAY_PARTITION variable=finalOut complete dim=1
#pragma HLS ARRAY_PARTITION variable=finalOut complete dim=2

  chalf addres_s4[2];
#pragma HLS ARRAY_PARTITION variable=addres_s4 complete dim=1

  chalf addres_f[OCFACT][16];
#pragma HLS ARRAY_PARTITION variable=addres_f complete dim=1
#pragma HLS ARRAY_PARTITION variable=addres_f complete dim=2

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
  int fc = params[12];
  int relu = params[13];
  int backward = params[14];
  int stride = params[15];
  int pad = params[16];
  bool mode = (backward);

  int xdim_out = ((xdim - ksize + 2 * pad) / stride) + 1;
  int ydim_out = xdim_out;

  memcpy(biasbuf, bias, sizeof(chalf) * outchannels);
  int ofm_iters = (outchannels % OCFACT == 0) ? outchannels / OCFACT :
    (outchannels / OCFACT) + 1;

  for (int y = 0; y < ydim_out; ++y) {
    for (int x = 0; x < xdim_out; ++x) {
      for (int n = 0; n < rpo; ++n) {
        for (int p = 0; p < ksize; ++p) {
          for (int q = 0; q < ksize; ++q) {
            int in_y = y * stride - pad + p;
            int in_x = x * stride - pad + q;
            int in_idx = ((in_y * xdim + in_x) * inchannels +
                n * burstchannels) * (numimages >> 4);
            int inbuf_idx = (p * ksize + q) * burstchannels * (numimages >> 4);
            int in_size = burstchannels * (numimages >> 4);
            if (in_y >= 0 && in_y < ydim && in_x >= 0 && in_x < xdim) {
              memcpy(inbuf + inbuf_idx, input + in_idx, sizeof(chalf16) *
                  in_size);
            } else {
              for (int i = 0; i < in_size; ++i) {
#pragma HLS pipeline
                inbuf[i + inbuf_idx].s0 = 0;
                inbuf[i + inbuf_idx].s1 = 0;
                inbuf[i + inbuf_idx].s2 = 0;
                inbuf[i + inbuf_idx].s3 = 0;
                inbuf[i + inbuf_idx].s4 = 0;
                inbuf[i + inbuf_idx].s5 = 0;
                inbuf[i + inbuf_idx].s6 = 0;
                inbuf[i + inbuf_idx].s7 = 0;
                inbuf[i + inbuf_idx].s8 = 0;
                inbuf[i + inbuf_idx].s9 = 0;
                inbuf[i + inbuf_idx].sa = 0;
                inbuf[i + inbuf_idx].sb = 0;
                inbuf[i + inbuf_idx].sc = 0;
                inbuf[i + inbuf_idx].sd = 0;
                inbuf[i + inbuf_idx].se = 0;
                inbuf[i + inbuf_idx].sf = 0;
              }
            }
          }
        }

        for (int o = 0; o < ofm_iters; ++o) {
          if (n == 0 && !mode) {
            for (int i = 0; i < (numimages >> 4); ++i) {
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
            int out_idx;
            int out_size;
            for (int k = 0; k < OCFACT; ++k) {
              if (mode) {
                out_idx = (o * OCFACT + k) * ksize * ksize *
                  (inchannels >> 4) + n * (burstchannels >> 4);
                out_size = ksize * ksize * (burstchannels >> 4);
              } else {
                out_idx = ((y * xdim + x) * outchannels +
                    (o * OCFACT) + k) * (numimages >> 4);
                out_size = numimages >> 4;
              }
              memcpy(outbuf[k], output + out_idx,
                  sizeof(chalf16) * out_size);
            }
          } 
          
          for (int k = 0; k < OCFACT; ++k) {
            int w_idx, w_size;
            if (mode) {
              w_idx = ((y * xdim + x) * outchannels +
                (o * OCFACT + k)) * (numimages >> 4);
              w_size = numimages >> 4;
            } else {
              w_idx = (o * OCFACT + k) * ksize * ksize *
                (inchannels >> 4) + n * (burstchannels >> 4);
              w_size = ksize * ksize * (burstchannels >> 4);
            }
            memcpy(wbuf[k], weights + w_idx, sizeof(chalf16) * w_size);
          }

          short w_off = 0;
          short img_off = 0;
          short iter = 0;
          short kdim_off = 0;
          int mac_iterations = ksize * ksize * (numimages >> 4) *
            (burstchannels >> 1);
          for (int i = 0; i < mac_iterations; ++i, ++iter) {
#pragma HLS pipeline
#pragma HLS DEPENDENCE variable outbuf inter false
#pragma HLS DEPENDENCE variable finalOut inter false
            if (!mode) {
              if (iter == (numimages >> 4)) {
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
            
            for (int k = 0; k < OCFACT; ++k) {
              for (int m = 0; m < 2; ++m) {
                short w_idx = (mode) ? img_off : kdim_off *
                  (burstchannels >> 4) + ((w_off * 2 + m) >> 4);
                if (!mode) {
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
                  for (int j = 0; j < 16; ++j)
                    weight_val[j] = weight_fw[(w_off & 0x7) * 2 + m];
                } else {
                  weight_val[0] = wbuf[k][w_idx].s0;
                  weight_val[1] = wbuf[k][w_idx].s1;
                  weight_val[2] = wbuf[k][w_idx].s2;
                  weight_val[3] = wbuf[k][w_idx].s3;
                  weight_val[4] = wbuf[k][w_idx].s4;
                  weight_val[5] = wbuf[k][w_idx].s5;
                  weight_val[6] = wbuf[k][w_idx].s6;
                  weight_val[7] = wbuf[k][w_idx].s7;
                  weight_val[8] = wbuf[k][w_idx].s8;
                  weight_val[9] = wbuf[k][w_idx].s9;
                  weight_val[10] = wbuf[k][w_idx].sa;
                  weight_val[11] = wbuf[k][w_idx].sb;
                  weight_val[12] = wbuf[k][w_idx].sc;
                  weight_val[13] = wbuf[k][w_idx].sd;
                  weight_val[14] = wbuf[k][w_idx].se;
                  weight_val[15] = wbuf[k][w_idx].sf;
                }
                short in_idx = (kdim_off * burstchannels + w_off * 2 + m) *
                  (numimages >> 4) + img_off;
                multres[m][k][0] = inbuf[in_idx].s0 * weight_val[0];
                multres[m][k][1] = inbuf[in_idx].s1 * weight_val[1];
                multres[m][k][2] = inbuf[in_idx].s2 * weight_val[2];
                multres[m][k][3] = inbuf[in_idx].s3 * weight_val[3];
                multres[m][k][4] = inbuf[in_idx].s4 * weight_val[4];
                multres[m][k][5] = inbuf[in_idx].s5 * weight_val[5];
                multres[m][k][6] = inbuf[in_idx].s6 * weight_val[6];
                multres[m][k][7] = inbuf[in_idx].s7 * weight_val[7];
                multres[m][k][8] = inbuf[in_idx].s8 * weight_val[8];
                multres[m][k][9] = inbuf[in_idx].s9 * weight_val[9];
                multres[m][k][10] = inbuf[in_idx].sa * weight_val[10];
                multres[m][k][11] = inbuf[in_idx].sb * weight_val[11];
                multres[m][k][12] = inbuf[in_idx].sc * weight_val[12];
                multres[m][k][13] = inbuf[in_idx].sd * weight_val[13];
                multres[m][k][14] = inbuf[in_idx].se * weight_val[14];
                multres[m][k][15] = inbuf[in_idx].sf * weight_val[15];
              
                for (int j = 0; j < 8; ++j)
                  addres_s1[j] = multres[m][k][j * 2] +
                    multres[m][k][j * 2 + 1];
                for (int j = 0; j < 4; ++j)
                  addres_s2[j] = addres_s1[j * 2] + addres_s1[j * 2 + 1];
                for (int j = 0; j < 2; ++j)
                  addres_s3[j] = addres_s2[j * 2] + addres_s2[j * 2 + 1];
                addres_s4[m] = addres_s3[0] + addres_s3[1];
              }

              for (int j = 0; j < 16; ++j) {
                addres_f[k][j] = multres[0][k][j] + multres[1][k][j];
              }

              if (mode) {
                for (int m = 0; m < 2; ++m) {
                  finalOut[k][(w_off & 0x7) * 2 + m] = addres_s4[m];
                }
              } else {
                for (int j = 0; j < 16; ++j)
                  finalOut[k][j] = addres_f[k][j];
              }

              short out_idx = (mode) ? kdim_off * (burstchannels >> 4) +
                ((w_off * 2) >> 4) : img_off;

              bool acc_enable = (mode) ? ((w_off & 0x7) == 7) : true;
              
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
                (inchannels >> 4) + n * (burstchannels >> 4);
              out_size = ksize * ksize * (burstchannels >> 4);
            } else {
              out_idx = (((y * xdim) + x) * outchannels +
                (o * OCFACT) + k) * (numimages >> 4);
              out_size = numimages >> 4;
            }
            if (o * OCFACT + k < outchannels)
              memcpy(output + out_idx, outbuf[k],
                  sizeof(chalf16) * out_size);
          }
        }
      }
    }
  }
}

}
