#include <cstdio>
#include <cstring>
#include <stdint.h>
#include "../../../include/fpga_caffe/cpfp.hpp"

extern "C" {

void cpfp_ops(float *input, float *weights, float *output,
    int * params, int op) {

/* Ports */
#pragma HLS data_pack variable=input
#pragma HLS data_pack variable=weights
#pragma HLS data_pack variable=output
#pragma HLS INTERFACE m_axi port=input offset=slave bundle=gmem1
#pragma HLS INTERFACE m_axi port=output offset=slave bundle=gmem2
#pragma HLS INTERFACE m_axi port=weights offset=slave bundle=gmem3
#pragma HLS INTERFACE m_axi port=params offset=slave bundle=gmem4
#pragma HLS INTERFACE s_axilite port=input bundle=control
#pragma HLS INTERFACE s_axilite port=output bundle=control
#pragma HLS INTERFACE s_axilite port=weights bundle=control
#pragma HLS INTERFACE s_axilite port=params bundle=control
#pragma HLS INTERFACE s_axilite port=op bundle=control
#pragma HLS INTERFACE s_axilite port=return bundle=control

  int inchannels = params[1];  

  for (int i = 0; i < inchannels >> 4; ++i) {
    for (int j = 0; j < 16; ++j) {
      if (op == 0)
        output[i * 16 + j] = float(cpfp(input[i * 16 + j]) * cpfp(weights[i * 16 + j]));
      else if (op == 1)
        output[i * 16 + j] = float(cpfp(input[i * 16 + j]) + cpfp(weights[i * 16 + j]));
      else if (op == 2) {
        cpfp temp1, temp2;
        mult2_1(cpfp(input[i * 16 + j]), cpfp(input[i * 16 + j]), cpfp(weights[i * 16 + j]), &temp1, &temp2);
        if (temp1 != temp2)
          output[i * 16 + j] = 0;
        else
          output[i * 16 + j] = float(temp1);
      }
    }
  }
}

}
