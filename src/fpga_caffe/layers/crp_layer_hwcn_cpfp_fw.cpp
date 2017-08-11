#include <stdio.h>
#include <assert.h>
#include <string.h>
#include <stdbool.h>

#include "../../../include/fpga_caffe/layer.hpp"
#include "../../../include/fpga_caffe/cpfp.hpp"
#include "../../../include/fpga_caffe/vector_types.hpp"

#define OCFACT 1 

/* Computes the maximum value of a 3x3 window via a reduction tree,
 * also saves the window index at each stage to determine the index of the
 * maximum value in the window */

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

/* ReLU forward pass implementation, processes 16 input values in parallel */

cpfp16 relu_fw_nobuf(cpfp16 outVal, bool enable) {
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
    return val;
  } else {
    return outVal;
  }
}


extern "C" {
/* Kernel used for computing direct convolution, ReLU, max pooling, and inner
 * product forward only. 
 * input:         Flattened input array containing image data in HWCN format
 * weights:       Convolution filters in forward pass, output diff in backward
 *                pass
 * bias:          Flattened bias array, used only in forward pass
 * output:        Output of the convolution in the forward pass, weight diffs
 *                in the backward pass
 * tagVals:       Tags for indicating if ReLU activation was non-zero for
 *                conv-relu modes, and tag indicating the max value index for
 *                max pooling mode
 * params:        Engine specific parameters used for controlling the output
 *                and compute modes
 * group_idx:     Group index used for forward convolution only currently
 */ 

void crp_layer_hwcn_cpfp_fw(cpfp16 *input, cpfp16 *weights, cpfp *bias,
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

  // Output buffer used for writing
  cpfp16 outBuf[OCFACT][256];
#pragma HLS ARRAY_PARTITION variable=outBuf complete dim=1

  // Weight buffer
  cpfp16 wBuf[OCFACT][8 * 256];
#pragma HLS ARRAY_PARTITION variable=wBuf complete dim=1

  // Bias buffer
  cpfp biasBuf[OCFACT][(6144 / OCFACT)];
#pragma HLS ARRAY_PARTITION variable=biasBuf complete dim=1

  // Pooling input buffer, used for reading in pooling window data
  cpfp16 poolInBuf[9][16 * 16];
#pragma HLS ARRAY_PARTITION variable=poolInBuf complete dim=1

  // Pooling output buffer, used for outputting max pooling value
  cpfp16 poolOutBuf[16 * 16];

  cpfp multRes[OCFACT][4][16];
#pragma HLS ARRAY_PARTITION variable=multRes complete dim=1
#pragma HLS ARRAY_PARTITION variable=multRes complete dim=2
#pragma HLS ARRAY_PARTITION variable=multRes complete dim=3

  cpfp weightIn[16];
#pragma HLS ARRAY_PARTITION variable=weightIn complete

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

  // Input parameters

  // Input image channels
  short inChannels = params[0];
  // Output images channels
  short outChannels = params[1];
  // Input image channels to be burst read, same as inChannels unless the full
  // set of image channels + window size can fit in on-chip memory
  short burstChannels = params[2];
  // Number of input reads per output, this will be larger than one if 
  // not all input channels can fit in on-chip memory. rpo * burstChannels = 
  // inChannels
  short rpo = params[3];
  // Number of output channel reads required to process all output channels
  short ocrdfact = params[4];
  // Output image channels to be burst read per processing element group,
  // ocrdfact * burstoc >= outChannels
  ap_uint<9> burstoc = params[5];
  // Input image y dimension size
  ap_uint<10> ydim = params[6];
  // Input image x dimension size
  ap_uint<10> xdim = params[7];
  // Kernel size, only square kernels support currently
  ap_uint<5> ksize = params[9];
  // Number of groups for group convolution, currently only supported in
  // forward path
  short numgroups = params[10];
  // Number of input/output images, this should be a multiple of 16 and
  // burstoc * numImages / 16 should be >= 12
  ap_uint<10> numImages = params[11];
  // This controls whether ReLU is applied to the inputs or the weights in the
  // backward passes
  short reluWeights = params[12];
  // This controls whether ReLU is applied anywhere in the engine
  short relu = params[13];
  // This is the flag for various modes of operation:
  // backward == 0: forward pass
  // backward == 1: backward wrt weights
  // backward == 2: backward wrt data
  short backward = params[14];
  // Convolution stride: stride is the same in x and y dimensions currently
  ap_uint<4> stride = params[15];
  // Convolution padding: symmetric padding in x and y dimensions
  ap_uint<4> pad = params[16];
  // Max pooling Enable
  short pool = params[17];
  // Pooling size, 2 or 3 supported currently
  ap_uint<3> pksize = params[18];

  assert((pksize == 2) || (pksize == 3));
  assert(ksize <= 11);
  assert(ksize >= 1);
  assert(burstChannels <= 2048);
  assert(burstChannels >= 4);
  assert(numImages <= 256);

  bool bwMode = (backward == 1);
  bool fwMode = (backward == 0);
  bool poolMode = (pool == 1);

  ap_uint<10> xdim_out = ((xdim - ksize + 2 * pad) / stride) + 1;
  ap_uint<10> ydim_out = xdim_out;

  ap_uint<8> imgFact = numImages >> 4;
  short burstFact = burstChannels >> 2;
  short icFact = (inChannels % 16 == 0) ? (inChannels >> 4) :
    (inChannels >> 4) + 1;
  short wcFact = (burstChannels % 16 == 0) ? (burstChannels >> 4) :
    (burstChannels >> 4) + 1;

  // Split output channel computations across PE groups
  short out_div = ocrdfact / OCFACT;
  // Reduced amount of ouput feature map iterations 
  short ofm_iters = (ocrdfact % OCFACT == 0) ? out_div : out_div + 1;
  
  if (!poolMode) {
    // Read in bias data 
    for (int o = 0; o < ofm_iters; ++o) {
      for (int k = 0; k < OCFACT; ++k) {
        int biasOffset = (o * OCFACT + k) * burstoc + outChannels
          * group_idx;
        int biasSize = burstoc;
        if ((o * OCFACT + k) * burstoc + burstoc > outChannels) {
          short newBurst = outChannels - (o * OCFACT + k) * burstoc;
          biasSize = newBurst;
        }
        bool readEnable = ((o * OCFACT + k) * burstoc < outChannels);
        if (readEnable) {
          memcpy(biasBuf[k] + o * burstoc, bias + biasOffset,
            sizeof(cpfp) * biasSize);
        }
      }
    }
    // Read in the input data
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
            // Iterate over each window position
            for (int p = 0; p < ksize; ++p) {
              for (int q = 0; q < ksize; ++q) {
                short in_y = y * stride - pad + p;
                short in_x = x * stride - pad + q;
                int inIdx = (((in_y * xdim + in_x) * numgroups + group_idx) *
                    inChannels + n * burstChannels) * imgFact;
                int inBufIdx = (p * ksize + q) * burstFact * imgFact;
                short inSize = burstFact * imgFact;

                // Determine the begining of none-zero data for non-zero
                // padding
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
                    // Shift input to the left rather than doing a memory
                    // transfer for each window (stride of one only)
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
                    // If we can't shift the data then we need to transfer
                    // from on-board memory
                    for (int j = 0; j < 4; ++j) {
                      int f_inIdx = inIdx + j * burstFact * imgFact;
                      memcpy(inBuf[j] + inBufIdx, input + f_inIdx,
                          sizeof(cpfp16) * inSize);
                    }
                  }
                }
              }
            }

            if (n == 0) {
              // Set the output to be the bias
              for (int b = 0; b < burstoc; ++b) {
                for (int i = 0; i < imgFact; ++i) {
#pragma HLS pipeline
                  for (int k = 0; k < OCFACT; ++k) {
                    outBuf[k][b * imgFact + i] =
                      biasBuf[k][o * burstoc + b];
                  }
                }
              }
            } else {
              // Read the output from on-board memory in the case where not all
              // of the input could fit on device
              for (int k = 0; k < OCFACT; ++k) {
                int outIdx, outIdxFW;
                short outSize, outSizeFW;
                outIdxFW = (((y * xdim_out + x) * numgroups + group_idx) *
                  outChannels + (o * OCFACT + k) * burstoc) * imgFact; 
                outSizeFW = burstoc * imgFact;

                // Handles the edge case where the burst transfer exceeds the
                // data size by reducing the burst transfer
                if ((o * OCFACT + k) * burstoc + burstoc > outChannels) {
                  short newBurst = outChannels - (o * OCFACT + k) * burstoc;
                  outSizeFW =  newBurst * imgFact;
                }

                outIdx = outIdxFW;
                outSize = outSizeFW;
                bool readEnable = ((o * OCFACT + k) * burstoc < outChannels);

                if (readEnable)
                  memcpy(outBuf[k], output + outIdx, sizeof(cpfp16) * outSize);
              }
            }

            // Read the weights from main memory for the forward pass, or read
            // the output diff for the backward pass
            for (int k = 0; k < OCFACT; ++k) {
              int wIdxFW, wIdx;
              short wSizeFW, wSize;
              wIdxFW = ((o * OCFACT + k) * burstoc + outChannels *
                group_idx) * ksize * ksize * icFact + n * burstoc * ksize *
                ksize * wcFact;
              wSizeFW = burstoc * ksize * ksize * wcFact;
              
              // Handles the edge case where the burst transfer exceeds the
              // data size by reducing the burst transfer

              if ((o * OCFACT + k) * burstoc + burstoc > outChannels) {
                short newBurst = outChannels - (o * OCFACT + k) * burstoc;
                wSizeFW = newBurst * ksize * ksize * wcFact;
              }

              wIdx = wIdxFW;
              wSize = wSizeFW;

              bool readEnable = ((o * OCFACT + k) * burstoc < outChannels) &&
                ((x == 0) && (y == 0));
              if (readEnable) {
                memcpy(wBuf[k], weights + wIdx, sizeof(cpfp16) * wSize);
              }
            }

            ap_uint<10> w_off = 0;
            ap_uint<6> img_off = 0;
            ap_uint<10> iter = 0;
            ap_uint<8> xdim_off = 0;
            ap_uint<8> ydim_off = 0;
            ap_uint<2> counter_bw = 0;
            ap_uint<2> counter_fw = 0;
            ap_uint<5> b_off = 0;
            int mac_iterations = burstoc * yksize * xksize * imgFact
              * burstFact;
            MAC_LOOP: for (int i = 0; i < mac_iterations; ++i, ++iter,
              ++counter_bw) {
#pragma HLS pipeline
#pragma HLS DEPENDENCE variable outBuf inter false
#pragma HLS DEPENDENCE variable finalOut inter false
              // FW index calculation
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
              bool accEnable = true;

              for (int k = 0; k < OCFACT; ++k) {
                weightIn[0] = wBuf[k][wIdx].s0;
                weightIn[1] = wBuf[k][wIdx].s1;
                weightIn[2] = wBuf[k][wIdx].s2;
                weightIn[3] = wBuf[k][wIdx].s3;   
                weightIn[4] = wBuf[k][wIdx].s4;
                weightIn[5] = wBuf[k][wIdx].s5;
                weightIn[6] = wBuf[k][wIdx].s6;
                weightIn[7] = wBuf[k][wIdx].s7;
                weightIn[8] = wBuf[k][wIdx].s8;
                weightIn[9] = wBuf[k][wIdx].s9;
                weightIn[10] = wBuf[k][wIdx].sa;
                weightIn[11] = wBuf[k][wIdx].sb;
                weightIn[12] = wBuf[k][wIdx].sc;
                weightIn[13] = wBuf[k][wIdx].sd;
                weightIn[14] = wBuf[k][wIdx].se;
                weightIn[15] = wBuf[k][wIdx].sf;
                for (int m = 0; m < 4; ++m) {
                  for (int j = 0; j < 16; ++j) {
                    weightVal[m][j] = weightIn[counter_fw * 4 + m];
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

                  // 4x16xOCFACT multiplications
                  for (int j = 0; j < 16; ++j) 
                    multRes[k][m][j] = inVal[m][j] * weightVal[m][j];
                }

                // Adder tree, forward: OCFACTx16x4 to OCFACTx16 reduction
                
                // Stage 1
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

                // Stage 2
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
                bool reluFWEnable = relu && (n == rpo - 1) &&
                  (w_off == burstFact - 1) && (xdim_off == xksize - 1) &&
                  (ydim_off == yksize - 1);
                // 16 Accumulations, forward accumulate every cycle, backward
                // accumulate every four cycles. In the forward path ReLU is
                // applied when all accumulations for an output are computed.
                if (accEnable) {
                  outBuf[k][outIdx] = relu_fw_nobuf(outBuf[k][outIdx] +
                    finalOut[k], reluFWEnable);
                }               
              }
            }

            // Write the outputs back to board memory
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

              if (writeEnable)
                memcpy(output + outIdx, outBuf[k], sizeof(cpfp16) * outSize);
            }
          }
        }
      }
    }
  } else {
    // Max Pooling, 2x2 or 3x3
    short pooled_height = ydim - pksize;
    if ((pooled_height & 0x1) == 1)
      pooled_height = (pooled_height >> 1) + 2;
    else
      pooled_height = (pooled_height >> 1) + 1;

    short pooled_width = pooled_height;

    // Forward path
    for (int ph = 0; ph < pooled_height; ++ph) {
      for (int pw = 0; pw < pooled_width; ++pw) {
        int hstart = ph * 2;
        int wstart = pw * 2;
        for (int c = 0; c < rpo; ++c) {
          // Read in a burst of input windows
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
              // Compute 3x3 max window
              poolOutBuf[n * 2 + j] = max9(poolInBuf, n * 2 + j, &mask);
            }
          }
          // Write the output and tags to on-board memory
          int outIdx = ((ph * pooled_width + pw) * inChannels +
              c * burstChannels) * imgFact;
          memcpy(output + outIdx, poolOutBuf, sizeof(cpfp16) *
              imgFact * burstChannels);
        }
      }
    }
  }
}

}
