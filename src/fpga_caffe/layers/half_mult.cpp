#include <cstdio>
#include <cstring>
#include "half.hpp"

extern "C" {

void half_mult(float *input, float *weights, float *output,
    int * params) {

/* Ports */
#pragma HLS INTERFACE m_axi port=input offset=slave bundle=gmem1
#pragma HLS INTERFACE m_axi port=output offset=slave bundle=gmem2
#pragma HLS INTERFACE m_axi port=weights offset=slave bundle=gmem3
#pragma HLS INTERFACE m_axi port=params offset=slave bundle=gmem4
#pragma HLS INTERFACE s_axilite port=input bundle=control
#pragma HLS INTERFACE s_axilite port=output bundle=control
#pragma HLS INTERFACE s_axilite port=weights bundle=control
#pragma HLS INTERFACE s_axilite port=params bundle=control
#pragma HLS INTERFACE s_axilite port=return bundle=control

  int inchannels = params[1];
  float inbuf[1024];
#pragma HLS ARRAY_PARTITION variable=inbuf cyclic factor=16
  float wbuf[1024];
#pragma HLS ARRAY_PARTITION variable=wbuf cyclic factor=16
  float outbuf[1024];
#pragma HLS ARRAY_PARTITION variable=outbuf cyclic factor=16

  memcpy(inbuf, input, sizeof(float) * inchannels);
  memcpy(wbuf, weights, sizeof(float) * inchannels);

  for (int i = 0; i < inchannels >> 4; ++i) {
#pragma HLS pipeline
    for (int j = 0; j < 16; ++j) 
      outbuf[i * 16 + j] = float(half(inbuf[i * 16 + j]) * half(wbuf[i * 16 + j]));
  }

  memcpy(output, outbuf, sizeof(float) * inchannels);
}

}
