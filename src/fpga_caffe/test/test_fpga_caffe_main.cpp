#include "fpga_caffe/test/test_fpga_caffe_main.hpp"

OCLUtil::OCLUtil(std::string xclbin, std::string xclkernel) {
  xcl_name = xclbin;
  kernel = xclkernel;
}

void OCLUtil::Setup_Platform() {
  oclPlatform.resize(1);
  clGetPlatformIDs(0, NULL, &oclNumPlatforms);
  clGetPlatformIDs(1, &(oclPlatform[0]), NULL);
  clGetDeviceIDs(oclPlatform[0], CL_DEVICE_TYPE_ACCELERATOR, 1, &oclDevices,
      NULL);
  oclContext = clCreateContext(NULL, 1, &oclDevices, NULL, NULL, NULL);
  oclCommandQueue = clCreateCommandQueue(oclContext, oclDevices,
      /*CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE*/ 0, NULL);
}

void OCLUtil::Setup() {
  std::string path(".build_release/opencl/src/caffe/layers/");

  const char *filename = (path + xcl_name).c_str();

  std::ifstream file_stream(filename);
  std::string source( (std::istreambuf_iterator<char>(file_stream)),
      (std::istreambuf_iterator<char>()));
  size_t sourceSize = source.length();
 
  const char *sourceStr = source.c_str();

  oclPlatform.resize(1);
  clGetPlatformIDs(0, NULL, &oclNumPlatforms);
  clGetPlatformIDs(1, &(oclPlatform[0]), NULL);
  clGetDeviceIDs(oclPlatform[0], CL_DEVICE_TYPE_ACCELERATOR, 1, &oclDevices,
      NULL);
  oclContext = clCreateContext(NULL, 1, &oclDevices, NULL, NULL, NULL);
  oclCommandQueue = clCreateCommandQueue(oclContext, oclDevices,
      /*CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE*/ 0, NULL);
  cl_program oclProgram = clCreateProgramWithBinary(oclContext, 1,
      &oclDevices, &sourceSize, (const unsigned char **)(&sourceStr), NULL,
      NULL);
  clBuildProgram(oclProgram, 0, NULL, NULL, NULL, NULL);
  oclKernel = clCreateKernel(oclProgram, kernel.c_str(), NULL);
}

void fillVector(std::vector<float>& input, float beg, float end) {
  static boost::random::mt19937 engine(time(0));
  boost::random::uniform_real_distribution<float> dis(beg, end);
  std::vector<float>::iterator it;
  for (it = input.begin(); it < input.end(); ++it) {
    *it = dis(engine);
  }
}

void fillVectorCPFP(std::vector<float>& input, float beg, float end) {
  static boost::random::mt19937 engine(time(0));
  boost::random::uniform_real_distribution<float> dis(beg, end);
  std::vector<float>::iterator it;
  for (it = input.begin(); it < input.end(); ++it) {
    *it = float(cpfp(dis(engine)));
  }
}

void copyVector(std::vector<float> input, std::vector<float>& output,
    int xsize, int xsize_pad) {
  for (int i = 0; i < output.size() / xsize_pad; ++i) {
    for (int j = 0; j < xsize_pad; ++j) {
      if (j < xsize) 
        output[i * xsize_pad + j] = input[i * xsize + j];
      else
        output[i * xsize_pad + j] = 0;
    }
  }
}

void copyWeights(std::vector<float> w_input, std::vector<float>& w_output,
    int ksize, int padsize, kernel_params params) {
  int oc = params.outchannels * params.numgroups;
  int ic = params.inchannels;
  int bc = params.burstchannels;
  for (int i = 0; i < oc; ++i) {
    for (int n = 0; n < ic / bc; ++n) {
      for (int m = 0; m < bc; ++m) {
        int out_idx = (i * ic + n * bc + m) * padsize;
        int in_idx = (i * ic + n * bc + m) * ksize * ksize;
        if (ksize == 1) {
          int bc_mod = (bc % 16 == 0) ? bc : (bc / 16 + 1) * 16;
          out_idx = (i * bc_mod * (ic / bc) + n * bc_mod + m) * padsize;
          in_idx = (i * ic + n * bc + m) * ksize * ksize;
          w_output[out_idx] = w_input[in_idx];
        } else if (ksize == 3) {
          for (int j = 0; j < ksize * ksize; ++j) {
            w_output[out_idx + j] = w_input[in_idx + j];
          }
        } else if (ksize == 5) {
          for (int j = 0; j < 5; ++j) {
            for (int k = 0; k < 3; ++k) {
              w_output[out_idx + j * 3 + k] = w_input[in_idx + j * 5 + k];
              if (k < 2) 
                w_output[out_idx + 16 + j * 3 + k] =
                  w_input[in_idx + j * 5 + 3 + k];
              else
                w_output[out_idx + 16 + j * 3 + k] = 0;
            }
          } 
        }
      }
    }
  }
}

void toCPFP(std::vector<float> input, std::vector<cpfp>& output) {
  for (int i = 0; i < input.size(); ++i) {
    output[i] = cpfp(input[i]);
  } 
}

void toFloat(std::vector<cpfp> input, std::vector<float>& output) {
  for (int i = 0; i < input.size(); ++i) {
    output[i] = float(input[i]);
  }
}

void ref_fc_layer(std::vector<float> input, std::vector<float> weights,
    std::vector<float> bias, std::vector<float>& output,
    kernel_params params) {
  int inchannels = params.xtile_pad * 2;
  int outchannels = params.outchannels;

  for (int n = 0; n < params.numimages; ++n) {
    for (int j = 0; j < outchannels; ++j)
      output[n * outchannels + j] = bias[j];

    for (int i = 0; i < inchannels; ++i) {
      for (int j = 0; j < outchannels; ++j) {
        output[n * outchannels + j] += input[n * inchannels + i] *
          weights[j * inchannels + i];
      }
    }
  }
}

void ref_backward_fc_layer(std::vector<float> input,
    std::vector<float> weights, std::vector<float>& output,
    kernel_params params) {
  int inchannels = params.xtile_pad * 2;
  int outchannels = params.outchannels;

  for (int n = 0; n < params.numimages; ++n) {
    for (int i = 0; i < inchannels; ++i) {
      for (int j = 0; j < outchannels; ++j) {
        output[j * inchannels + i] += input[i * params.numimages + n] *
          weights[j * params.numimages + n];
      }
    }
  }
}

void ref_relu_layer(std::vector<float>& output) {
  for (int i = 0; i < output.size(); ++i)
    output[i] = std::max((float)0.0, output[i]);
}

void ref_conv_layer(std::vector<float> input, std::vector<float> weights,
    std::vector<float> bias, std::vector<float>& output,
    kernel_params params) {
  int o_head, k_head;
  int out_idx, in_idx, k_idx;

  int numgroups = params.numgroups;
  int inchannels = params.inchannels * numgroups;
  int outchannels = params.outchannels * numgroups;
  int ydim = params.ydim;
  int xdim = params.xdim;
  int numimages = params.numimages;
  int ksize = params.ksize;

  int pad = 1;
  int stride = 1;
  if (ksize == 5) 
    pad = 2;
  else if(ksize == 3)
    pad = 1;
  else if (ksize == 1)
    pad = 0;

  // Convolution
  for (int n = 0; n < numimages; n++) {
    for (int g = 0; g < numgroups; g++) {
      o_head = (outchannels / numgroups) * g;
      k_head = (inchannels / numgroups) * g;
      int o_g = outchannels / numgroups;
      int k_g = inchannels / numgroups;
      for (int o = 0; o < o_g; o++) {
        for (int k = 0; k < k_g; k++) {
          for (int y = 0; y < ydim; y++) {
            for (int x = 0; x < xdim; x++) {
              for (int p = 0; p < ksize; p++) {
                for (int q = 0; q < ksize; q++) {
                  int in_y = y * stride - pad + p;
                  int in_x = x * stride - pad + q;
                  if (in_y >= 0 && in_y < ydim && in_x >= 0 && in_x < xdim) {
                    out_idx = (((n * outchannels) + o + o_head) * ydim + y) 
                      * xdim + x;
                    in_idx = (((n * inchannels) + k + k_head) * ydim + in_y) 
                      * xdim + in_x;
                    k_idx = (((o + o_head) * (k_g) + k) * ksize + p) 
                      * ksize + q;
                    output[out_idx] += input[in_idx] * weights[k_idx];
                  }
                }
              }
            }
          }
        }
      }
    }
  }
  for (int n = 0; n < numimages; n++) {
    for (int o = 0; o < outchannels; ++o) {
      for (int y = 0; y < ydim; ++y) {
        for (int x = 0; x < xdim; ++x) {
          out_idx = (((n * outchannels) + o) * ydim + y) * xdim + x;
          output[out_idx] += bias[o];
        }
      }
    }
  }
}

void ref_backward_conv_layer(std::vector<float> input,
    std::vector<float> weights, std::vector<float>& output,
    kernel_params params) {
  int o_head, k_head;
  int out_idx, in_idx, k_idx;

  int numgroups = params.numgroups;
  int inchannels = params.inchannels * numgroups;
  int outchannels = params.outchannels * numgroups;
  int ydim = params.ydim;
  int xdim = params.xdim;
  int numimages = params.numimages;
  int ksize = params.ksize;

  int pad = 1;
  int stride = 1;
  if (ksize == 5) 
    pad = 2;
  else if(ksize == 3)
    pad = 1;
  else if (ksize == 1)
    pad = 0;

  for (int n = 0; n < numimages; n++) {
    for (int g = 0; g < numgroups; g++) {
      o_head = (outchannels / numgroups) * g;
      k_head = (inchannels / numgroups) * g;
      int o_g = outchannels / numgroups;
      int k_g = inchannels / numgroups;
      for (int o = 0; o < o_g; o++) {
        for (int k = 0; k < k_g; k++) {
          for (int p = 0; p < ydim; p++) {
            for (int q = 0; q < xdim; q++) {
              for (int y = 0; y < ksize; y++) {
                for (int x = 0; x < ksize; x++) {
                  int in_y = y * stride - pad + p;
                  int in_x = x * stride - pad + q;
                  if (in_y >= 0 && in_y < ydim
                    && in_x >= 0 && in_x < xdim) {
                    out_idx = (((n * outchannels) + o + o_head) * ydim + p) 
                      * xdim + q;
                    in_idx = (((n * inchannels) + k + k_head) * ydim + in_y) 
                      * xdim + in_x;
                    k_idx = (((o + o_head) * (k_g) + k) * ksize + y) 
                      * ksize + x;
                    output[k_idx] += input[in_idx] * weights[out_idx];
                  }
                }
              }
            }
          }
        }
      }
    }
  }
}

void ref_conv_layer_hwcn(std::vector<float> input, std::vector<float> weights,
    std::vector<float> bias, std::vector<float>& output, kernel_params params,
    bool wino) {
  int o_head, k_head;
  int out_idx, in_idx, k_idx;

  int numgroups = params.numgroups;
  int inchannels = params.inchannels * numgroups;
  int outchannels = params.outchannels * numgroups;
  int ydim = params.ydim;
  int xdim = params.xdim;
  int numimages = params.numimages;
  int ksize = params.ksize;

  int pad = params.pad;
  int stride = params.stride;

  // Convolution
  for (int n = 0; n < numimages; n++) {
    for (int g = 0; g < numgroups; g++) {
      o_head = (outchannels / numgroups) * g;
      k_head = (inchannels / numgroups) * g;
      int o_g = outchannels / numgroups;
      int k_g = inchannels / numgroups;
      for (int o = 0; o < o_g; o++) {
        for (int k = 0; k < k_g / 4; k++) {
          for (int m = 0; m < 4; ++m) {
            for (int y = 0; y < ydim; y++) {
              for (int x = 0; x < xdim; x++) {
                for (int p = 0; p < ksize; p++) {
                  for (int q = 0; q < ksize; q++) {
                    int in_y = y * stride - pad + p;
                    int in_x = x * stride - pad + q;
                    if (in_y >= 0 && in_y < ydim && in_x >= 0 && in_x < xdim) {
                      out_idx = ((y * xdim + x) * outchannels + o + o_head)
                        * numimages + n;
                      in_idx = ((in_y * xdim + in_x) * inchannels + m *
                        (k_g / 4) + k + k_head) * numimages + n;
                      if (wino)
                        k_idx = ((q * o_g + (o + o_head)) * ksize + p) * k_g +
                          k * 4 + m;
                      else 
                        k_idx = (((o + o_head) * ksize + p) * ksize + q) *
                          k_g + k * 4 + m;
                      output[out_idx] += input[in_idx] * weights[k_idx];
                    }
                  }
                }
              }
            }
          }
        }
      }
    }
  }
  for (int n = 0; n < numimages; n++) {
    for (int o = 0; o < outchannels; ++o) {
      for (int y = 0; y < ydim; ++y) {
        for (int x = 0; x < xdim; ++x) {
          out_idx = ((y * xdim + x) * outchannels + o) * numimages + n;
          output[out_idx] += bias[o];
        }
      }
    }
  }
}

void ref_backward_conv_layer_hwcn(std::vector<float> input,
    std::vector<float> weights, std::vector<float>& output,
    kernel_params params) {
  int o_head, k_head;
  int out_idx, in_idx, k_idx;

  int numgroups = params.numgroups;
  int inchannels = params.inchannels * numgroups;
  int outchannels = params.outchannels * numgroups;
  int ydim = params.ydim;
  int xdim = params.xdim;
  int numimages = params.numimages;
  int ksize = params.ksize;

  int pad = params.pad;
  int stride = params.stride;

  for (int n = 0; n < numimages; n++) {
    for (int g = 0; g < numgroups; g++) {
      o_head = (outchannels / numgroups) * g;
      k_head = (inchannels / numgroups) * g;
      int o_g = outchannels / numgroups;
      int k_g = inchannels / numgroups;
      for (int o = 0; o < o_g; o++) {
        for (int k = 0; k < k_g / 4; k++) {
          for (int m = 0; m < 4; ++m) {
            for (int p = 0; p < ydim; p++) {
              for (int q = 0; q < xdim; q++) {
                for (int y = 0; y < ksize; y++) {
                  for (int x = 0; x < ksize; x++) {
                    int in_y = y * stride - pad + p;
                    int in_x = x * stride - pad + q;
                    if (in_y >= 0 && in_y < ydim
                      && in_x >= 0 && in_x < xdim) {
                      out_idx = ((p * xdim + q) * outchannels + o + o_head)
                        * numimages + n;
                      in_idx = ((in_y * xdim + in_x) * inchannels + m *
                        (k_g / 4) + k + k_head) * numimages + n;
                      k_idx = (((o + o_head) * ksize + y) * ksize + x) *
                        k_g + k * 4 + m;
                      output[k_idx] += input[in_idx] * weights[out_idx];
                    }
                  }
                }
              }
            }
          }
        }
      }
    }
  }
}

bool checkEQ(float expected, float result, float epsilon, float absError) {
  float absExpected = fabs(expected);
  float absResult = fabs(result);
  float diff = fabs(expected - result);


  if ((isnan(expected) == 1) && isnan(expected) == isnan(result))
    return true;

  if (expected == result) 
    return true;
  
  if (diff <= absError)
    return true;

  float largest = (absExpected > absResult) ? absExpected : absResult;

  if (diff <= largest * epsilon)
    return true;
  std::cout << "Expected: " << expected << " Got: " << result << std::endl;
  return false;
}

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);

  // invoke the test.
  return RUN_ALL_TESTS();
}
