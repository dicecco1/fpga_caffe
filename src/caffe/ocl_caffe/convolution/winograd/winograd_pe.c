#include <stdio.h>
#include <string.h>
#include <assert.h>

#include <stdbool.h>

void transform_incol(float *input) {
#pragma HLS INLINE off
  float A0 = input[0] - input[2];
  float A1 = input[1] + input[2];
  float A2 = input[2] - input[1];
  float A3 = input[1] - input[3];
  input[0] = A0;
  input[1] = A1;
  input[2] = A2;
  input[3] = A3;
}

void transform_intilecol(float input[4 * 4]) {
#pragma HLS INLINE off 
  float A0 = input[0];
  float A1 = input[1];
  float A2 = input[2];
  float A3 = input[3];
  float B0 = input[4];
  float B1 = input[5];
  float B2 = input[6];
  float B3 = input[7];
  float C0 = input[8];
  float C1 = input[9];
  float C2 = input[10];
  float C3 = input[11];
  float D0 = input[12];
  float D1 = input[13];
  float D2 = input[14];
  float D3 = input[15];

  input[0] = A0 - A2;
  input[1] = A1 + A2;
  input[2] = A2 - A1;
  input[3] = A1 - A3;
  input[4] = B0 - B2;
  input[5] = B1 + B2;
  input[6] = B2 - B1;
  input[7] = B1 - B3;
  input[8] = C0 - C2;
  input[9] = C1 + C2;
  input[10] = C2 - C1;
  input[11] = C1 - C3;
  input[12] = D0 - D2;
  input[13] = D1 + D2;
  input[14] = D2 - D1;
  input[15] = D1 - D3; 
}

void transform_outtile(float *output_t, float output[4][4], int j) {
#pragma HLS INLINE
  float t00 = output_t[0] + output_t[1] + output_t[2];
  float t01 = output_t[1] - output_t[2] - output_t[3];
  float t10 = output_t[4] + output_t[5] + output_t[6];
  float t11 = output_t[5] - output_t[6] - output_t[7];
  float t20 = output_t[8] + output_t[9] + output_t[10];
  float t21 = output_t[9] - output_t[10] - output_t[11];
  float t30 = output_t[12] + output_t[13] + output_t[14];
  float t31 = output_t[13] - output_t[14] - output_t[15];

  output[j][0] = t00 + t10 + t20;
  output[j][1] = t01 + t11 + t21;
  output[j][2] = t10 - t20 - t30;
  output[j][3] = t11 - t21 - t31;
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

void acc_out(floatv16 *outbuf, floatv16 *tempout, float ot[4][4], int bank_off, int offset, int offset2) {
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

void winograd_pe(float *input, floatv16 *weights, float *bias, floatv16 *output, 
      int group, int inchannels, int outchannels, int burstchannels, int rpo,
      int ydim, int xdim, int ytile, int xtile, int ytile_pad, int xtile_pad, 
      int rburst, int dataoff, int numgroups) {
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

  float inbuf[128 * 128][4][2];
#pragma HLS ARRAY_PARTITION variable=inbuf cyclic factor=2 dim=3
#pragma HLS ARRAY_PARTITION variable=inbuf cyclic factor=4 dim=2
#pragma HLS ARRAY_PARTITION variable=inbuf cyclic factor=4 dim=1

  float sidebuf[2048][4][2];
#pragma HLS ARRAY_PARTITION variable=sidebuf cyclic factor=2 dim=3
#pragma HLS ARRAY_PARTITION variable=sidebuf cyclic factor=4 dim=2

  floatv16 outbuf[256 * 16];
  floatv16 tempout[128 * 16];

  float line[256] = {0};
#pragma HLS ARRAY_PARTITION variable=line cyclic factor=2

  floatv16 wbuf[256];

  float biasbuf[512];

  float inrow[2];
#pragma HLS ARRAY_PARTITION variable=inrow complete

  float itcol[4];
#pragma HLS ARRAY_PARTITION variable=itcol complete
  float itt[4 * 4];
#pragma HLS ARRAY_PARTITION variable=itt complete
  float it[4 * 4];
#pragma HLS ARRAY_PARTITION variable=it complete
  float wt[4 * 4];
#pragma HLS ARRAY_PARTITION variable=wt complete
  float ot[4][4];
#pragma HLS ARRAY_PARTITION variable=ot complete dim=1
#pragma HLS ARRAY_PARTITION variable=ot complete dim=2 

  int bank_off;

  assert(rburst >= 13);
  assert(rburst <= 4096);
  assert(inchannels >= 1);
  assert(inchannels <= 512);
  assert(outchannels >= 1);
  assert(outchannels <= 512);

  assert(burstchannels >= 1);
  assert(burstchannels <= 256);

  assert(xdim >= 13);
  assert(xdim <= 256);
  assert(ydim >= 13);
  assert(ydim <= 256);
  assert(xtile >= 7);
  assert(xtile <= 128);
  assert(ytile >= 7);
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

  int n, i, y, x, p, q, j, o;

  int sb_off = 0;
  int w_off = 0;
  int tile_cnt = 0;
  int xt_off = 0;
  int yt_off = 0;
  int fact = 0;
  int out_offset = 0;  
  int y_off = 0;
  int p_off = 1;
  int i_off = 0;

  int in_y = 0;
  int offset = 0;
  int in_y2 = 0;
  int offset2 = 0;

  int offtemp1 = 0;
  int offtemp2 = 0;

  int init_sel = 0;

  fact = ((xtile_pad) >> 3);

  memcpy(biasbuf, bias + outchannels * group, sizeof(float) * outchannels);

  for (n = 0; n < rpo; ++n) {
    y_off = 0;
    p_off = 1;
    i_off = 0;
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
//#pragma HLS DEPENDENCE variable=inbuf intra false
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
    ITRANSLOOP: for (i = 0; i < (burstchannels * ytile * xtile_pad) >> 2; ++i) {
#pragma HLS pipeline
      for (int off = 0; off < 4; ++off) {
        for (q = 0; q < 2; ++q) {
          for (p = 0; p < 4; ++p) {
            itcol[p] = inbuf[i * 4 + off][p][q];
          }
          transform_incol(itcol);
          for (p = 0; p < 4; ++p) {
            inbuf[i * 4 + off][p][q] = itcol[p];
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
          transform_incol(itcol);
          for (p = 0; p < 4; ++p) {
            sidebuf[i][p][q] = itcol[p];
          }
        }
      }
    }
    
    for (o = 0; o < outchannels; ++o) {
      memcpy(wbuf, weights + (o + outchannels * group) * inchannels + n * burstchannels, sizeof(floatv16) * burstchannels);
      out_offset = dataoff * numgroups * outchannels * ydim * fact + 
                    ((o + outchannels * group) * ydim) * fact;

      if (n == 0) {
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
      MULTACCSTAGE: for (i = 0; i < (burstchannels * ytile * xtile_pad) >> 2; ++i, ++tile_cnt, ++xt_off) {
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
        for (j = 0; j < 4; ++j) {
          itt[0] = inbuf[i * 4 + j][0][0];
          itt[1] = inbuf[i * 4 + j][0][1];
          if (xtile == xtile_pad && xt_off * 4 + j == xtile_pad - 1) {
            itt[2] = sidebuf[sb_off][0][0];
            itt[3] = sidebuf[sb_off][0][1];
          } else {
            itt[2] = inbuf[i * 4 + j + 1][0][0];
            itt[3] = inbuf[i * 4 + j + 1][0][1];
          }

          itt[4] = inbuf[i * 4 + j][1][0];
          itt[5] = inbuf[i * 4 + j][1][1];
          if (xtile == xtile_pad && xt_off * 4 + j == xtile_pad - 1) {
            itt[6] = sidebuf[sb_off][1][0];
            itt[7] = sidebuf[sb_off][1][1];
          } else { 
            itt[6] = inbuf[i * 4 + j + 1][1][0];
            itt[7] = inbuf[i * 4 + j + 1][1][1];
          }

          itt[8] = inbuf[i * 4 + j][2][0];
          itt[9] = inbuf[i * 4 + j][2][1];
          if (xtile == xtile_pad && xt_off * 4 + j == xtile_pad - 1) {
            itt[10] = sidebuf[sb_off][2][0];
            itt[11] = sidebuf[sb_off][2][1];
          } else {
            itt[10] = inbuf[i * 4 + j + 1][2][0];
            itt[11] = inbuf[i * 4 + j + 1][2][1];
          }

          itt[12] = inbuf[i * 4 + j][3][0];
          itt[13] = inbuf[i * 4 + j][3][1];
          if (xtile == xtile_pad && xt_off * 4 + j == xtile_pad - 1) {
            itt[14] = sidebuf[sb_off][3][0];
            itt[15] = sidebuf[sb_off][3][1];
          } else {
            itt[14] = inbuf[i * 4 + j + 1][3][0];
            itt[15] = inbuf[i * 4 + j + 1][3][1];
          }
          transform_intilecol(itt);

          it[0] = itt[0] * wbuf[w_off].val0;
          it[1] = itt[1] * wbuf[w_off].val1;
          it[2] = itt[2] * wbuf[w_off].val2;
          it[3] = itt[3] * wbuf[w_off].val3;

          it[4] = itt[4] * wbuf[w_off].val4;
          it[5] = itt[5] * wbuf[w_off].val5;
          it[6] = itt[6] * wbuf[w_off].val6;
          it[7] = itt[7] * wbuf[w_off].val7;

          it[8] = itt[8] * wbuf[w_off].val8;
          it[9] = itt[9] * wbuf[w_off].val9;
          it[10] = itt[10] * wbuf[w_off].val10;
          it[11] = itt[11] * wbuf[w_off].val11;

          it[12] = itt[12] * wbuf[w_off].val12;
          it[13] = itt[13] * wbuf[w_off].val13;
          it[14] = itt[14] * wbuf[w_off].val14;
          it[15] = itt[15] * wbuf[w_off].val15;

          transform_outtile(it, ot, j);
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
          outbuf[offtemp1].val0 = tempout[offtemp2].val0;
          outbuf[offtemp1].val1 = tempout[offtemp2].val1;
          outbuf[offtemp1].val2 = tempout[offtemp2].val2;
          outbuf[offtemp1].val3 = tempout[offtemp2].val3;
          outbuf[offtemp1].val4 = tempout[offtemp2].val4;
          outbuf[offtemp1].val5 = tempout[offtemp2].val5;
          outbuf[offtemp1].val6 = tempout[offtemp2].val6;
          outbuf[offtemp1].val7 = tempout[offtemp2].val7;
          outbuf[offtemp1].val8 = tempout[offtemp2].val8;
          outbuf[offtemp1].val9 = tempout[offtemp2].val9;
          outbuf[offtemp1].val10 = tempout[offtemp2].val10;
          outbuf[offtemp1].val11 = tempout[offtemp2].val11;
          outbuf[offtemp1].val12 = tempout[offtemp2].val12;
          outbuf[offtemp1].val13 = tempout[offtemp2].val13;
          outbuf[offtemp1].val14 = tempout[offtemp2].val14;
          outbuf[offtemp1].val15 = tempout[offtemp2].val15;
        }
      }
      memcpy(output + out_offset, outbuf, sizeof(floatv16) * fact * ydim);
    }
  }
}
