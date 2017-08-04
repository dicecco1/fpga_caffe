#include <stdio.h>
#include <assert.h>
#include <string.h>
#include <stdbool.h>

#include "../../../include/fpga_caffe/layer.hpp"
#include "../../../include/fpga_caffe/cpfp.hpp"
#include "../../../include/fpga_caffe/vector_types.hpp"

#define OCFACT 1 

/* Kernel used for computing direct convolution forward and backward. 
 * input:         flattened input array containing image data
 * weights:       convolution filters
 * bias:          flattened bias array
 * output:        output of the convolution
 */ 

cpfp16 max9(cpfp16 poolInBuf[9][16 * 16], int n, short16 *outMask) {
#pragma HLS INLINE
  cpfp16 reduce_s1[4];
#pragma HLS ARRAY_PARTITION variable=reduce_s1 complete
  cpfp16 reduce_s2[2];
#pragma HLS ARRAY_PARTITION variable=reduce_s2 complete
  cpfp16 reduce_s3;
  cpfp16 reduce_s4;

  short16 mask_s0[9];
#pragma HLS ARRAY_PARTITION variable=mask_s0 complete
  short16 mask_s1[4];
#pragma HLS ARRAY_PARTITION variable=mask_s1 complete
  short16 mask_s2[2];
#pragma HLS ARRAY_PARTITION variable=mask_s2 complete
  short16 mask_s3;

  short16 mask_s4;

  for (int i = 0; i < 9; ++i)
    mask_s0[i] = i;

  for (int i = 0; i < 4; ++i)
    reduce_s1[i] = max(poolInBuf[i * 2][n], poolInBuf[i * 2 + 1][n],
        mask_s0[i * 2], mask_s0[i * 2 + 1], &mask_s1[i]);

  for (int i = 0; i < 2; ++i)
    reduce_s2[i] = max(reduce_s1[i * 2], reduce_s1[i * 2 + 1],
        mask_s1[i * 2], mask_s1[i * 2 + 1], &mask_s2[i]);

  reduce_s3 = max(reduce_s2[0], reduce_s2[1], mask_s2[0], mask_s2[1],
      &mask_s3);

  reduce_s4 = max(reduce_s3, poolInBuf[8][n], mask_s3, mask_s0[8],
      &mask_s4);

  *outMask = mask_s4;

  return reduce_s4;
}

cpfp16 relu_fw(cpfp16 outVal, short *outBufRelu, bool enable) {
  cpfp16 val = max(outVal);
  short reluOut = 0;
  reluOut |= (val.s0 != cpfp(0)) ? 1 << 0 : 0;
  reluOut |= (val.s1 != cpfp(0)) ? 1 << 1 : 0;
  reluOut |= (val.s2 != cpfp(0)) ? 1 << 2 : 0;
  reluOut |= (val.s3 != cpfp(0)) ? 1 << 3 : 0;
  reluOut |= (val.s4 != cpfp(0)) ? 1 << 4 : 0;
  reluOut |= (val.s5 != cpfp(0)) ? 1 << 5 : 0;
  reluOut |= (val.s6 != cpfp(0)) ? 1 << 6 : 0;
  reluOut |= (val.s7 != cpfp(0)) ? 1 << 7 : 0;
  reluOut |= (val.s8 != cpfp(0)) ? 1 << 8 : 0;
  reluOut |= (val.s9 != cpfp(0)) ? 1 << 9 : 0;
  reluOut |= (val.sa != cpfp(0)) ? 1 << 10 : 0;
  reluOut |= (val.sb != cpfp(0)) ? 1 << 11 : 0;
  reluOut |= (val.sc != cpfp(0)) ? 1 << 12 : 0;
  reluOut |= (val.sd != cpfp(0)) ? 1 << 13 : 0;
  reluOut |= (val.se != cpfp(0)) ? 1 << 14 : 0;
  reluOut |= (val.sf != cpfp(0)) ? 1 << 15 : 0;

  if (enable) {
    *outBufRelu = reluOut;
    return val;
  } else {
    *outBufRelu = reluOut;
    return outVal;
  }
}

extern "C" {

void cr_layer_hwcn_cpfp_fw(cpfp16 *input, cpfp16 *weights, cpfp *bias,
    cpfp16 *output, short *tagVals, int *params, int group_idx) { 
// Ports 
#pragma HLS data_pack variable=weights
#pragma HLS data_pack variable=output
#pragma HLS data_pack variable=input
#pragma HLS INTERFACE m_axi port=input offset=slave bundle=gmem1
#pragma HLS INTERFACE m_axi port=output offset=slave bundle=gmem2
#pragma HLS INTERFACE m_axi port=weights offset=slave bundle=gmem3
#pragma HLS INTERFACE m_axi port=bias offset=slave bundle=gmem4
#pragma HLS INTERFACE m_axi port=tagVals offset=slave bundle=gmem5
#pragma HLS INTERFACE m_axi port=params offset=slave bundle=gmem6
#pragma HLS INTERFACE s_axilite port=input bundle=control
#pragma HLS INTERFACE s_axilite port=output bundle=control
#pragma HLS INTERFACE s_axilite port=weights bundle=control
#pragma HLS INTERFACE s_axilite port=bias bundle=control
#pragma HLS INTERFACE s_axilite port=tagVals bundle=control
#pragma HLS INTERFACE s_axilite port=params bundle=control
#pragma HLS INTERFACE s_axilite port=group_idx bundle=control
#pragma HLS INTERFACE s_axilite port=return bundle=control

  // Input tile buffer
  cpfp16 inBuf[4][2 * 256 * 16];
#pragma HLS ARRAY_PARTITION variable=inBuf complete dim=1

  short outBufRelu[OCFACT][256];
#pragma HLS ARRAY_PARTITION variable=outBufRelu complete dim=1

  // Output buffer used for writing
  cpfp16 outBuf[OCFACT][256];
#pragma HLS ARRAY_PARTITION variable=outBuf complete dim=1

  // Weight buffer
  cpfp16 wBuf[OCFACT][8 * 256];
#pragma HLS ARRAY_PARTITION variable=wBuf complete dim=1

  // Bias buffer
  cpfp biasBuf[OCFACT][(6144 / OCFACT)];
#pragma HLS ARRAY_PARTITION variable=biasBuf complete dim=1

  cpfp16 poolInBuf[9][16 * 16];
#pragma HLS ARRAY_PARTITION variable=poolInBuf complete dim=1

  cpfp16 poolOutBuf[16 * 16];

  short outMask[16 * 256];
#pragma HLS ARRAY_PARTITION variable=outMask cyclic factor=16 dim=1

  cpfp multRes[OCFACT][4][16];
#pragma HLS ARRAY_PARTITION variable=multRes complete dim=1
#pragma HLS ARRAY_PARTITION variable=multRes complete dim=2
#pragma HLS ARRAY_PARTITION variable=multRes complete dim=3

  cpfp weightFW[16];
#pragma HLS ARRAY_PARTITION variable=weightFW complete

  cpfp weightVal[4][16];
#pragma HLS ARRAY_PARTITION variable=weightVal complete dim=1
#pragma HLS ARRAY_PARTITION variable=weightVal complete dim=2

  cpfp inVal[4][16];
#pragma HLS ARRAY_PARTITION variable=inVal complete dim=1
#pragma HLS ARRAY_PARTITION variable=inVal complete dim=2

  cpfp addTreeS1[OCFACT][32];
#pragma HLS ARRAY_PARTITION variable=addTreeS1 complete dim=1
#pragma HLS ARRAY_PARTITION variable=addTreeS1 complete dim=2

  cpfp addTreeS2[OCFACT][16];
#pragma HLS ARRAY_PARTITION variable=addTreeS2 complete dim=1
#pragma HLS ARRAY_PARTITION variable=addTreeS2 complete dim=2

  cpfp finalOut[OCFACT][16];
#pragma HLS ARRAY_PARTITION variable=finalOut complete dim=1
#pragma HLS ARRAY_PARTITION variable=finalOut complete dim=2

  short inChannels = params[0];
  short outChannels = params[1];
  short burstChannels = params[2];
  short rpo = params[3];
  short ocrdfact = params[4];
  ap_uint<9> burstoc = params[5];
  ap_uint<10> ydim = params[6];
  ap_uint<10> xdim = params[7];
  ap_uint<5> ksize = params[9];
  short numgroups = params[10];
  ap_uint<10> numImages = params[11];
  short reluWeights = params[12];
  short relu = params[13];
  short backward = params[14];
  ap_uint<4> stride = params[15];
  ap_uint<4> pad = params[16];
  bool mode = (backward == 1);
  short pool = params[17];
  ap_uint<3> pksize = params[18];

  assert((pksize == 2) || (pksize == 3));
  assert(ksize <= 11);
  assert(ksize >= 1);
  assert(burstChannels <= 2048);
  assert(burstChannels >= 4);
  assert(numImages <= 256);

  ap_uint<10> xdim_out = ((xdim - ksize + 2 * pad) / stride) + 1;
  ap_uint<10> ydim_out = xdim_out;

  ap_uint<8> imgFact = numImages >> 4;
  short burstFact = burstChannels >> 2;
  short icFact = (inChannels % 16 == 0) ? (inChannels >> 4) :
    (inChannels >> 4) + 1;
  short wcFact = (burstChannels % 16 == 0) ? (burstChannels >> 4) :
    (burstChannels >> 4) + 1;
  short out_div = ocrdfact / OCFACT;
  short ofm_iters = (ocrdfact % OCFACT == 0) ? out_div : out_div + 1;
  
  if (pool == 0) {
    for (int o = 0; o < ofm_iters; ++o) {
      for (int k = 0; k < OCFACT; ++k) {
        int biasOffset = (o * OCFACT + k) * burstoc + outChannels
          * group_idx;
        int biasSize = burstoc;
        if ((o * OCFACT + k) * burstoc + burstoc > outChannels) {
          short newBurst = outChannels - (o * OCFACT + k) * burstoc;
          biasSize = newBurst;
        }
        bool writeEnable = ((o * OCFACT + k) * burstoc < outChannels);
        if (writeEnable) {
          memcpy(biasBuf[k] + o * burstoc, bias + biasOffset,
            sizeof(cpfp) * biasSize);
        }
      }
    }
    for (int n = 0; n < rpo; ++n) {
      for (int o = 0; o < ofm_iters; ++o) {
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
                int inIdx = (((in_y * xdim + in_x) * numgroups + group_idx) *
                    inChannels + n * burstChannels) * imgFact;
                int inBufIdx = (p * ksize + q) * burstFact * imgFact;
                short inSize = burstFact * imgFact;

                if (in_y >= 0 && in_y < ydim) {
                  if (q == 0)
                    yksize++;
                  if (!ykset) {
                    yk_off = p;
                    ykset = true;
                  }
                }
                if (in_x >= 0 && in_x < xdim) {
                  if (p == 0)
                    xksize++;
                  if (!xkset) {
                    xk_off = q;
                    xkset = true;
                  }
                }

                if (in_y >= 0 && in_y < ydim && in_x >= 0 && in_x < xdim) {
                  if ((x != 0) && (stride == 1) && (q != ksize - 1)) {
                    short q_off = burstFact * imgFact;
                    SHIFT_LOOP: for (int i = 0; i < inSize; ++i) {
#pragma HLS pipeline
#pragma HLS dependence variable=inBuf inter false
                      for (int j = 0; j < 4; ++j) {
                        inBuf[j][i + inBufIdx] = inBuf[j][i + inBufIdx
                          + q_off];
                      }
                    }
                  } else {
                    for (int j = 0; j < 4; ++j) {
                      int f_inIdx = inIdx + j * burstFact * imgFact;
                      memcpy(inBuf[j] + inBufIdx, input + f_inIdx,
                          sizeof(cpfp16) * inSize);
                    }
                  }
                }
              }
            }

            if ((n == 0)) {
              for (int b = 0; b < burstoc; ++b) {
                for (int i = 0; i < imgFact; ++i) {
#pragma HLS pipeline
                  for (int k = 0; k < OCFACT; ++k) {
                    outBuf[k][b * imgFact + i] =
                      biasBuf[k][o * burstoc + b];
                  }
                }
              }
            }  else {
              for (int k = 0; k < OCFACT; ++k) {
                int outIdx, outIdxFW;
                short outSize, outSizeFW;
                outIdxFW = (((y * xdim_out + x) * numgroups + group_idx) *
                  outChannels + (o * OCFACT + k) * burstoc) * imgFact; 
                outSizeFW = burstoc * imgFact;
                if ((o * OCFACT + k) * burstoc + burstoc > outChannels) {
                  short newBurst = outChannels - (o * OCFACT + k) * burstoc;
                  outSizeFW =  newBurst * imgFact;
                }

                outIdx = outIdxFW;
                outSize = outSizeFW;
                bool readEnable = ((o * OCFACT + k) * burstoc < outChannels);

                if (readEnable)
                  memcpy(outBuf[k], output + outIdx, sizeof(cpfp16) *
                      outSize);
              }
            }  
            for (int k = 0; k < OCFACT; ++k) {
              int wIdxFW, wIdx;
              short wSizeFW, wSize;
              wIdxFW = ((o * OCFACT + k) * burstoc + outChannels *
                group_idx) * ksize * ksize * icFact + n * burstoc * ksize *
                ksize * wcFact;
              wSizeFW = burstoc * ksize * ksize * wcFact;
              
              if ((o * OCFACT + k) * burstoc + burstoc > outChannels) {
                short newBurst = outChannels - (o * OCFACT + k) * burstoc;
                wSizeFW = newBurst * ksize * ksize * wcFact;
              }

              wIdx = wIdxFW;
              wSize = wSizeFW;

              bool readEnable = ((o * OCFACT + k) * burstoc <
                  outChannels) && ((mode) || ((x == 0) && (y == 0)));
              if (readEnable) {
                memcpy(wBuf[k], weights + wIdx, sizeof(cpfp16) * wSize);
              }
            }

            ap_uint<8> w_off = 0;
            ap_uint<5> img_off = 0;
            ap_uint<8> iter = 0;
            ap_uint<8> xdim_off = 0;
            ap_uint<8> ydim_off = 0;
            ap_uint<2> counter_fw = 0;
            ap_uint<5> b_off = 0;
            int mac_iterations = burstoc * yksize * xksize * imgFact
              * burstFact;
            MAC_LOOP: for (int i = 0; i < mac_iterations; ++i, ++iter) {
#pragma HLS pipeline
#pragma HLS DEPENDENCE variable outBuf inter false
#pragma HLS DEPENDENCE variable outBufRelu inter false
#pragma HLS DEPENDENCE variable finalOut inter false
              if (iter == imgFact) {
                if (b_off == burstoc - 1) {
                  b_off = 0;
                  if (w_off == burstFact - 1) {
                    counter_fw = 0;
                    w_off = 0;
                    if (xdim_off == xksize - 1) {
                      xdim_off = 0;
                      ydim_off++;
                    } else {
                      xdim_off++;
                    }
                  } else {
                    counter_fw++;
                    w_off++;
                  }
                } else {
                  b_off++;
                }
                iter = 0;
              }
              img_off = iter;

              short filt_off = (yk_off + ydim_off) * ksize + xk_off +
                xdim_off;
              short wIdxFW = (b_off * ksize * ksize + filt_off) * wcFact +
                (w_off >> 2);
              short wIdx = wIdxFW;
              short inIdx = (filt_off * burstFact + w_off) * imgFact
                + img_off;
              short outIdxFW = b_off * imgFact + img_off;
              short outIdx = outIdxFW;

              for (int k = 0; k < OCFACT; ++k) {
                weightFW[0] = wBuf[k][wIdx].s0;
                weightFW[1] = wBuf[k][wIdx].s1;
                weightFW[2] = wBuf[k][wIdx].s2;
                weightFW[3] = wBuf[k][wIdx].s3;   
                weightFW[4] = wBuf[k][wIdx].s4;
                weightFW[5] = wBuf[k][wIdx].s5;
                weightFW[6] = wBuf[k][wIdx].s6;
                weightFW[7] = wBuf[k][wIdx].s7;
                weightFW[8] = wBuf[k][wIdx].s8;
                weightFW[9] = wBuf[k][wIdx].s9;
                weightFW[10] = wBuf[k][wIdx].sa;
                weightFW[11] = wBuf[k][wIdx].sb;
                weightFW[12] = wBuf[k][wIdx].sc;
                weightFW[13] = wBuf[k][wIdx].sd;
                weightFW[14] = wBuf[k][wIdx].se;
                weightFW[15] = wBuf[k][wIdx].sf;
                for (int m = 0; m < 4; ++m) {
                  for (int j = 0; j < 16; ++j) {
                    weightVal[m][j] = weightFW[counter_fw * 4 + m];
                  }

                  inVal[m][0] = inBuf[m][inIdx].s0;
                  inVal[m][1] = inBuf[m][inIdx].s1;
                  inVal[m][2] = inBuf[m][inIdx].s2;
                  inVal[m][3] = inBuf[m][inIdx].s3;
                  inVal[m][4] = inBuf[m][inIdx].s4;
                  inVal[m][5] = inBuf[m][inIdx].s5;
                  inVal[m][6] = inBuf[m][inIdx].s6;
                  inVal[m][7] = inBuf[m][inIdx].s7;
                  inVal[m][8] = inBuf[m][inIdx].s8;
                  inVal[m][9] = inBuf[m][inIdx].s9;
                  inVal[m][10] = inBuf[m][inIdx].sa;
                  inVal[m][11] = inBuf[m][inIdx].sb;
                  inVal[m][12] = inBuf[m][inIdx].sc;
                  inVal[m][13] = inBuf[m][inIdx].sd;
                  inVal[m][14] = inBuf[m][inIdx].se;
                  inVal[m][15] = inBuf[m][inIdx].sf;

                  for (int j = 0; j < 16; ++j) 
                    multRes[k][m][j] = inVal[m][j] * weightVal[m][j];
                }

                for (int off = 0; off < 2; ++off) {
                  for (int m = 0; m < 2; ++m) {
                    for (int j = 0; j < 8; ++j) {
                      cpfp temp1, temp2;
                      temp1 = multRes[k][off * 2 + 0][m * 8 + j];
                      temp2 = multRes[k][off * 2 + 1][m * 8 + j];
                      addTreeS1[k][(off * 2 + m) * 8 + j] = temp1 + temp2;
                    }
                  }
                }
                for (int off = 0; off < 2; ++off) {
                  for (int m = 0; m < 2; ++m) {
                    for (int j = 0; j < 4; ++j) {
                      cpfp temp1, temp2;
                      temp1 = addTreeS1[k][(off * 2 + m) * 4 + j];
                      temp2 = addTreeS1[k][(off * 2 + m) * 4 + j + 16];
                      addTreeS2[k][(off * 2 + m) * 4 + j] = temp1 + temp2;
                    }
                  }
                }

                for (int j = 0; j < 16; ++j) {
                  finalOut[k][j] = addTreeS2[k][j];
                }
                bool reluFWEnable = relu && (n == rpo - 1)
                  && (w_off == burstFact - 1) && (xdim_off == xksize - 1) &&
                  (ydim_off == yksize - 1);
                outBuf[k][outIdx] = relu_fw(outBuf[k][outIdx] + finalOut[k],
                    &(outBufRelu[k][outIdx]), reluFWEnable);
              }
            }

            for (int k = 0; k < OCFACT; ++k) {
              int outIdx, outIdxFW;
              short outSize, outSizeFW;
              outIdxFW = (((y * xdim_out + x) * numgroups + group_idx) *
                outChannels + (o * OCFACT + k) * burstoc) * imgFact;
              outSizeFW = burstoc * imgFact;
              if ((o * OCFACT + k) * burstoc + burstoc > outChannels) { 
                short newBurst = outChannels - (o * OCFACT + k) * burstoc;
                outSizeFW =  newBurst * imgFact;
              }

              outIdx = outIdxFW;
              outSize = outSizeFW;

              bool writeEnable = ((o * OCFACT + k) * burstoc < outChannels);

              if (relu && (writeEnable) && (n == rpo - 1)) {
                memcpy(tagVals + outIdx, outBufRelu[k], sizeof(short) *
                    outSize);
              }

              if (writeEnable)
                memcpy(output + outIdx, outBuf[k], sizeof(cpfp16) * outSize);
            }
          }
        }
      }
    }
  } else {
    short pooled_height = ydim - pksize;
    if ((pooled_height & 0x1) == 1)
      pooled_height = (pooled_height >> 1) + 2;
    else
      pooled_height = (pooled_height >> 1) + 1;

    short pooled_width = pooled_height;

    for (int ph = 0; ph < pooled_height; ++ph) {
      for (int pw = 0; pw < pooled_width; ++pw) {
        int hstart = ph * 2;
        int wstart = pw * 2;
        for (int c = 0; c < rpo; ++c) {
          for (int h = 0; h < 3; ++h) {
            for (int w = 0; w < 3; ++w) {
              int inIdx = (((hstart + h) * xdim + (wstart + w)) *
                  inChannels + c * burstChannels) * imgFact;
              if ((hstart + h < ydim) && (wstart + w < xdim) &&
                  (h < pksize) && (w < pksize))
                memcpy(poolInBuf[h * 3 + w], input + inIdx,
                    sizeof(cpfp16) * imgFact * burstChannels);
              else
                for (int n = 0; n < imgFact * burstChannels; ++n)
#pragma HLS pipeline
                  poolInBuf[h * 3 + w][n] = cpfp(CPFP_MIN_VAL);
            }
          }
          POOL_LOOP: for (int n = 0; n < (imgFact * burstChannels) >> 1;
            ++n) {
#pragma HLS pipeline
            for (int j = 0; j < 2; ++j) {
              short16 mask;
              poolOutBuf[n * 2 + j] = max9(poolInBuf, n * 2 + j, &mask);
              outMask[(n * 2 + j) * 16 + 0] = mask.s0;
              outMask[(n * 2 + j) * 16 + 1] = mask.s1;
              outMask[(n * 2 + j) * 16 + 2] = mask.s2;
              outMask[(n * 2 + j) * 16 + 3] = mask.s3;
              outMask[(n * 2 + j) * 16 + 4] = mask.s4;
              outMask[(n * 2 + j) * 16 + 5] = mask.s5;
              outMask[(n * 2 + j) * 16 + 6] = mask.s6;
              outMask[(n * 2 + j) * 16 + 7] = mask.s7;
              outMask[(n * 2 + j) * 16 + 8] = mask.s8;
              outMask[(n * 2 + j) * 16 + 9] = mask.s9;
              outMask[(n * 2 + j) * 16 + 10] = mask.sa;
              outMask[(n * 2 + j) * 16 + 11] = mask.sb;
              outMask[(n * 2 + j) * 16 + 12] = mask.sc;
              outMask[(n * 2 + j) * 16 + 13] = mask.sd;
              outMask[(n * 2 + j) * 16 + 14] = mask.se;
              outMask[(n * 2 + j) * 16 + 15] = mask.sf;
            }
          }
          int outIdx = ((ph * pooled_width + pw) * inChannels +
              c * burstChannels) * imgFact;
          memcpy(output + outIdx, poolOutBuf, sizeof(cpfp16) *
              imgFact * burstChannels);
          memcpy(tagVals + outIdx * 16, outMask,
              sizeof(short) * numImages * burstChannels);
        }
      }
    }
  }
}

}
