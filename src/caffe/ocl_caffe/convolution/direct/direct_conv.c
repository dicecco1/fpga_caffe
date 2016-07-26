#include <stdio.h>
#include <string.h>
#include <assert.h>

#define NUM_DATA_ROWS (13)
#define NUM_DATA_COLS (13)
#define PAD           (1)
#define NUM_MASK_ROWS (3)
#define NUM_MASK_COLS (3)
#define STRIDE (1)
#define IN_CHANNEL 384
#define OUT_CHANNEL 256
#define K_CHANNEL 192
#define GROUPS 2
#define TOP_NUM 1
#define K_NUM 256
#define O_G 128
#define K_G 192
#define BURST 192

#define MAX_BURST 256
#define MAX_O_G 384
#define MAX_GROUPS 2

#define NUM_OUT_COLS (((NUM_DATA_COLS - NUM_MASK_COLS + (2*PAD) )/ (STRIDE)) + 1)
#define NUM_OUT_ROWS (((NUM_DATA_ROWS - NUM_MASK_ROWS + (2*PAD) )/ (STRIDE)) + 1)

#define DATA_SIZE_KERN      TOP_NUM * IN_CHANNEL * (NUM_DATA_ROWS) * (NUM_DATA_COLS)
#define FILTER_SIZE_KERN    K_NUM * K_CHANNEL * NUM_MASK_ROWS * NUM_MASK_COLS
#define OUTPUT_SIZE_KERN    TOP_NUM * OUT_CHANNEL * NUM_OUT_ROWS * NUM_OUT_COLS

void get_inbuf(float *input, float *inbuf, int g, int burst) {
#pragma HLS INLINE
  int k_head = burst * g;
  memcpy(inbuf, input + k_head * NUM_DATA_ROWS * NUM_DATA_COLS, sizeof(float) * burst * NUM_DATA_ROWS * NUM_DATA_COLS);
}

void conv5_omap(float *inbuf, float *weights, float *output, int win_off, int out_off, int burst)
{
  float weightbuf[MAX_BURST * NUM_MASK_ROWS * NUM_MASK_COLS];
#pragma HLS ARRAY_PARTITION variable=weightbuf cyclic factor=9

  float bufout[NUM_OUT_ROWS * NUM_OUT_COLS];
#pragma HLS ARRAY_PARTITION variable=bufout cyclic factor=13

  assert(burst >= 192);
  assert(burst <= MAX_BURST);

  int in_y = 0;
  int in_x = 0;
  int y, x, p, q, i;

  float temp[NUM_MASK_COLS];
#pragma HLS ARRAY_PARTITION variable=temp complete

  float temp_out;

  memcpy(weightbuf, weights + (win_off), sizeof(float) * burst * NUM_MASK_ROWS * NUM_MASK_COLS);
  
  LOOPI:for (i = 0; i < burst; ++i) {
    LOOPY:for (y = 0; y < NUM_OUT_ROWS; ++y) {
#pragma HLS pipeline
#pragma HLS DEPENDENCE variable=bufout inter false
      LOOPX:for (x = 0; x < NUM_OUT_COLS; ++x) {
#pragma HLS DEPENDENCE variable=bufout inter false
        temp_out = 0;
        LOOPP:for (p = 0; p < NUM_MASK_ROWS; ++p) {
          LOOPQ:for(q = 0; q < NUM_MASK_COLS; ++q) {
            in_y = y * STRIDE - PAD + p;
            in_x = x * STRIDE - PAD + q;
            temp[q] = 0;
            if(in_x >= 0 && in_x < NUM_DATA_COLS 
                && in_y >= 0 && in_y < NUM_DATA_ROWS) {
              temp[q] = inbuf[i * NUM_DATA_ROWS * NUM_DATA_COLS + in_y * NUM_DATA_COLS + in_x] 
                * weightbuf[i * NUM_MASK_ROWS * NUM_MASK_COLS + p * NUM_MASK_COLS + q];
            }
          }
          LOOPQ2:for (q = 1; q < NUM_MASK_COLS; ++q) {
            temp[0] += temp[q];
          }
          temp_out += temp[0];
        }
        if (i == 0)
          bufout[y * NUM_OUT_COLS + x] = temp_out;
        else
          bufout[y * NUM_OUT_COLS + x] += temp_out;
      }
    }
  }
  memcpy(output + (out_off), bufout, sizeof(float) * NUM_OUT_ROWS * NUM_OUT_COLS);
  return;
}

void direct_conv(float *input, float *weights, float *output, 
    int groups, int o_g, int burst) {
#pragma HLS INTERFACE m_axi port=input offset=slave bundle=gmem
#pragma HLS INTERFACE m_axi port=weights offset=slave bundle=gmem
#pragma HLS INTERFACE m_axi port=output offset=slave bundle=gmem
#pragma HLS INTERFACE s_axilite port=groups bundle=control
#pragma HLS INTERFACE s_axilite port=o_g bundle=control
#pragma HLS INTERFACE s_axilite port=burst bundle=control
#pragma HLS INTERFACE s_axilite port=input bundle=control
#pragma HLS INTERFACE s_axilite port=weights bundle=control
#pragma HLS INTERFACE s_axilite port=output bundle=control
#pragma HLS INTERFACE s_axilite port=return bundle=control

  int g, o, o_head, idx_o;

  float inbuf[MAX_BURST * NUM_DATA_ROWS * NUM_DATA_COLS];
#pragma HLS ARRAY_PARTITION variable=inbuf cyclic factor=13

  assert(groups >= 1);
  assert(o_g >= 128);
  assert(burst >= 192);
  assert(burst <= MAX_BURST);
  assert(groups <= MAX_GROUPS);
  assert(o_g <= MAX_O_G);

  LOOPG:for (g = 0; g < groups; ++g) {
    get_inbuf(input, inbuf, g, burst);
    LOOPO:for (o = 0; o < o_g; ++o) {
      int win_off = (o + o_g * g) * burst * NUM_MASK_ROWS * NUM_MASK_COLS;
      int out_off = (o + o_g * g) * NUM_OUT_ROWS * NUM_OUT_COLS;
      conv5_omap(inbuf, weights, output, win_off, out_off, burst);
    }
  }
}

