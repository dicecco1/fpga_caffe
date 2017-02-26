#include <stdio.h>
#include <assert.h>
#include <string.h>

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


/* Kernel used for computing direct convolution. 
 * input:         flattened input array containing image data, padded to be
 *                divisible by 16 on the x dimension
 * output:        output of the convolution, padded to be divisible by 16 on 
 */ 

extern "C" {

void bandwidth_test_16(float16 *input, float16 *output, int burst, int rpo) {

/* Ports */
#pragma HLS data_pack variable=output
#pragma HLS data_pack variable=input
#pragma HLS INTERFACE m_axi port=input offset=slave bundle=gmem1
#pragma HLS INTERFACE m_axi port=output offset=slave bundle=gmem2
#pragma HLS INTERFACE s_axilite port=input bundle=control
#pragma HLS INTERFACE s_axilite port=output bundle=control
#pragma HLS INTERFACE s_axilite port=burst bundle=control
#pragma HLS INTERFACE s_axilite port=rpo bundle=control
#pragma HLS INTERFACE s_axilite port=return bundle=control

  // Input tile buffer
  float16 inbuf[256 * 16];

  assert(rpo >= 1);
  assert(rpo <= 256 * 16);
  assert(burst >= 16);
  assert(burst <= 256 * 256);

  for (int i = 0; i < 100; ++i) {
    for (int j = 0; j < rpo; ++j)
      memcpy(inbuf + j * (burst >> 4), input + j * (burst >> 4),
          sizeof(float16) * (burst >> 4));
    for (int j = 0; j < rpo; ++j)
      memcpy(output + j * (burst >> 4), inbuf + j * (burst >> 4),
          sizeof(float16) * (burst >> 4));
  }

}

}
