#include <vector>
#include <string>
#include "gtest/gtest.h"

#include "fpga_caffe/test/test_fpga_caffe_main.hpp"

void ref_conv_layer_hwcn(std::vector<float> input, std::vector<float> weights,
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
        for (int k = 0; k < k_g; k++) {
          for (int y = 0; y < ydim; y++) {
            for (int x = 0; x < xdim; x++) {
              for (int p = 0; p < ksize; p++) {
                for (int q = 0; q < ksize; q++) {
                  int in_y = y * stride - pad + p;
                  int in_x = x * stride - pad + q;
                  if (in_y >= 0 && in_y < ydim && in_x >= 0 && in_x < xdim) {
                    out_idx = ((y * xdim + x) * outchannels + o + o_head)
                        * numimages + n;
                    in_idx = ((in_y * xdim + in_x) * inchannels + k + k_head) *
                      numimages + n;
                    k_idx = (((o + o_head) * ksize + p) * ksize + q) * k_g + k;
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
        for (int k = 0; k < k_g; k++) {
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
                    in_idx = ((in_y * xdim + in_x) * inchannels + k + k_head)
                      * numimages + n;
                    k_idx = (((o + o_head) * ksize + y) * ksize + x) * k_g + k;
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
template <typename TypeParam>
class CRLayerHWCNHalfTest : public OCLDeviceTest<TypeParam> {
  typedef typename TypeParam::Dtype Dtype;

 protected:
  CRLayerHWCNHalfTest()
    : ocl("cr_layer_hwcn_half.xclbin", "cr_layer_hwcn_half") 
  {}
  virtual void SetUp() {
    params.resize(1);
    batch_size.resize(1);
    params[0].numgroups = 1;
    params[0].inchannels = 16;
    params[0].outchannels = 1;
    params[0].burstchannels = 16;
    params[0].rpo = 1;
    params[0].rpofm = 1;
    params[0].burstydim = 8;
    params[0].ydim = 8;
    params[0].xdim = 8;
    params[0].xtile_pad = 8;
    params[0].numimages = 256;
    batch_size[0] = 256;
  }

  virtual ~CRLayerHWCNHalfTest() {}
  
  OCLUtil ocl;
  std::vector<Dtype> input;
  std::vector<Dtype> input_pad;
  std::vector<chalf> input_pad_half;
  std::vector<Dtype> weights;
  std::vector<Dtype> weights_pad;
  std::vector<chalf> weights_pad_half;
  std::vector<Dtype> bias;
  std::vector<chalf> bias_half;
  std::vector<Dtype> hw_results;
  std::vector<chalf> hw_results_half;
  std::vector<Dtype> sw_results;
  std::vector<kernel_params> params;
  std::vector<char> relu_vals;
  std::vector<int> batch_size;
  cl_mem ocl_input;
  cl_mem ocl_weights;
  cl_mem ocl_output;
  cl_mem ocl_bias;
  cl_mem ocl_relu_vals;
  cl_mem ocl_params;
};

TYPED_TEST_CASE(CRLayerHWCNHalfTest, TestOCLDtypesAndDevices);

TYPED_TEST(CRLayerHWCNHalfTest, TestCR1x1F_HALF) {
  typedef typename TypeParam::Dtype Dtype;
  this->ocl.Setup();
  int ksize = 3;
  int ksize_pad = 1;
  std::vector<kernel_params> params = this->params;
  std::vector<cl_event> events;

  for (int i = 0; i < params.size(); ++i) {
    // Set sizes
    params[i].ksize = ksize;
    params[i].backward = 0;
    params[i].relu = 0;
    params[i].stride = 1;
    params[i].pad = 1;
    int insize = params[i].numimages * params[i].inchannels * params[i].ydim *
      params[i].xdim * params[i].numgroups;
    int wsize = params[i].outchannels * params[i].numgroups *
      params[i].inchannels * ksize * ksize;
    int outsize = params[i].numimages * params[i].outchannels * params[i].ydim
      * params[i].xdim * params[i].numgroups;
    int bsize = params[i].outchannels * params[i].numgroups;
    int events_size = this->batch_size[i] / params[i].numimages *
      params[i].numgroups;
    // Clear input vectors
    this->input.clear();
    this->input_pad_half.clear();
    this->weights.clear();
    this->weights_pad_half.clear();
    this->bias.clear();
    this->bias_half.clear();
    this->hw_results.clear();
    this->hw_results_half.clear();
    this->sw_results.clear();
    events.clear();
    // Resize vectors
    this->input.resize(insize, 0);
    this->input_pad_half.resize(insize, chalf(0));
    this->weights.resize(wsize, 0);
    this->weights_pad_half.resize(wsize, chalf(0));
    this->bias.resize(bsize, 0);
    this->bias_half.resize(bsize, chalf(0));
    this->sw_results.resize(outsize, 0);
    this->hw_results.resize(outsize, 0);
    this->hw_results_half.resize(outsize, chalf(0));
    events.resize(events_size);
    // Populate vectors
    fillVectorHalf(this->input, 0.0, 1.0);
    fillVectorHalf(this->weights, -1.0, 1.0);
    fillVectorHalf(this->bias, -1.0, 1.0);
   
    toHalf(this->input, this->input_pad_half);
    toHalf(this->weights, this->weights_pad_half);
    toHalf(this->bias, this->bias_half);

    // Create buffers
    this->ocl_input = clCreateBuffer(this->ocl.oclContext, CL_MEM_READ_ONLY,
        sizeof(chalf) * insize, NULL, NULL);
    this->ocl_weights = clCreateBuffer(this->ocl.oclContext, CL_MEM_READ_ONLY,
        sizeof(chalf) * wsize, NULL, NULL);
    this->ocl_output = clCreateBuffer(this->ocl.oclContext, CL_MEM_READ_WRITE,
        sizeof(chalf) * outsize, NULL, NULL);
    this->ocl_bias = clCreateBuffer(this->ocl.oclContext, CL_MEM_READ_ONLY,
        sizeof(chalf) * bsize, NULL, NULL);
    this->ocl_params = clCreateBuffer(this->ocl.oclContext, CL_MEM_READ_ONLY,
        sizeof(kernel_params), NULL, NULL);

    clEnqueueWriteBuffer(this->ocl.oclCommandQueue, this->ocl_input, CL_TRUE,
        0, sizeof(chalf) * insize, this->input_pad_half.data(), 0, NULL,
        NULL);
    clEnqueueWriteBuffer(this->ocl.oclCommandQueue, this->ocl_weights, CL_TRUE,
        0, sizeof(chalf) * wsize, this->weights_pad_half.data(), 0, NULL,
        NULL);
    clEnqueueWriteBuffer(this->ocl.oclCommandQueue, this->ocl_bias, CL_TRUE, 0,
        sizeof(chalf) * bsize, this->bias_half.data(), 0, NULL, 
        NULL);
    clEnqueueWriteBuffer(this->ocl.oclCommandQueue, this->ocl_output, CL_TRUE,
        0, sizeof(chalf) * outsize, this->hw_results_half.data(), 0, NULL, 
        NULL);
    clEnqueueWriteBuffer(this->ocl.oclCommandQueue, this->ocl_params, CL_TRUE,
        0, sizeof(kernel_params), &params[i], 0, NULL, NULL);

    for (int n = 0; n < this->batch_size[i] / params[i].numimages; ++n) {
      for (int g = 0; g < params[i].numgroups; ++g) {
        clSetKernelArg(this->ocl.oclKernel, 0, sizeof(cl_mem),
            &this->ocl_input);
        clSetKernelArg(this->ocl.oclKernel, 1, sizeof(cl_mem), 
            &this->ocl_weights);
        clSetKernelArg(this->ocl.oclKernel, 2, sizeof(cl_mem),
            &this->ocl_bias);
        clSetKernelArg(this->ocl.oclKernel, 3, sizeof(cl_mem),
            &this->ocl_output);
        clSetKernelArg(this->ocl.oclKernel, 4, sizeof(cl_mem), 
            &this->ocl_params);
        clEnqueueTask(this->ocl.oclCommandQueue, this->ocl.oclKernel, 0, NULL,
            &(events[n * params[i].numgroups + g]));
      }
    }

    clWaitForEvents(events_size, events.data());

    clEnqueueReadBuffer(this->ocl.oclCommandQueue, this->ocl_output, CL_TRUE,
        0, sizeof(chalf) * outsize, this->hw_results_half.data(), 0, NULL,
        NULL);

    toFloat(this->hw_results_half, this->hw_results);

    ref_conv_layer_hwcn(this->input, this->weights, this->bias,
        this->sw_results, params[i]);
    //ref_relu_layer(this->sw_results);
    int size = params[i].numimages * params[i].outchannels *
      params[i].numgroups * params[i].ydim * params[i].xdim;
    for (int j = 0; j < size; ++j) {
      std::cout<<this->sw_results[j]<<" "<<this->hw_results[j]<<std::endl;
      EXPECT_TRUE(checkEQ(this->sw_results[j],
            this->hw_results[j], 1e-1, 1e-1));
    }
    clReleaseMemObject(this->ocl_input);
    clReleaseMemObject(this->ocl_weights);
    clReleaseMemObject(this->ocl_output);
    clReleaseMemObject(this->ocl_bias);
    clReleaseMemObject(this->ocl_params);
  }
}

TYPED_TEST(CRLayerHWCNHalfTest, TestCR1x1B_HALF) {
  typedef typename TypeParam::Dtype Dtype;
  this->ocl.Setup();
  int ksize = 3;
  std::vector<kernel_params> params = this->params;
  std::vector<cl_event> events;

  for (int i = 0; i < params.size(); ++i) {
    // Set sizes
    params[i].ksize = ksize;
    params[i].relu = 0;
    params[i].backward = 1;
    params[i].stride = 1;
    params[i].pad = 1;
    int insize = params[i].numimages * params[i].inchannels * params[i].ydim *
      params[i].xdim * params[i].numgroups;
    int wsize = params[i].numimages * params[i].outchannels * params[i].ydim *
      params[i].xdim * params[i].numgroups;
    int outsize = params[i].outchannels * params[i].numgroups *
      params[i].inchannels * ksize * ksize;
    int bsize = params[i].outchannels * params[i].numgroups;
    int events_size = this->batch_size[i] / params[i].numimages *
       params[i].numgroups;
    // Clear input vectors
    this->input.clear();
    this->input_pad_half.clear();
    this->weights.clear();
    this->weights_pad_half.clear();
    this->bias.clear();
    this->bias_half.clear();
    this->hw_results.clear();
    this->hw_results_half.clear();
    this->sw_results.clear();
    events.clear();
    // Resize vectors
    this->input.resize(insize, 0);
    this->input_pad_half.resize(insize, chalf(0));
    this->weights.resize(wsize, 0);
    this->weights_pad_half.resize(wsize, chalf(0));
    this->bias.resize(bsize, 0);
    this->bias_half.resize(bsize, chalf(0));
    this->sw_results.resize(outsize, 0);
    this->hw_results.resize(outsize, 0);
    this->hw_results_half.resize(outsize, chalf(0));
    events.resize(events_size);
    // Populate vectors
    fillVectorHalf(this->input, -1.0, 1.0);
    fillVectorHalf(this->weights, 0.0, 1.0);
    fillVectorHalf(this->bias, -1.0, 1.0);
  
    toHalf(this->input, this->input_pad_half);
    toHalf(this->weights, this->weights_pad_half);
    toHalf(this->bias, this->bias_half);

    // Create buffers
    this->ocl_input = clCreateBuffer(this->ocl.oclContext, CL_MEM_READ_ONLY,
        sizeof(chalf) * insize, NULL, NULL);
    this->ocl_weights = clCreateBuffer(this->ocl.oclContext, CL_MEM_READ_ONLY,
        sizeof(chalf) * wsize, NULL, NULL);
    this->ocl_output = clCreateBuffer(this->ocl.oclContext, CL_MEM_READ_WRITE,
        sizeof(chalf) * outsize, NULL, NULL);
    this->ocl_bias = clCreateBuffer(this->ocl.oclContext, CL_MEM_READ_ONLY,
        sizeof(chalf) * bsize, NULL, NULL);
    this->ocl_params = clCreateBuffer(this->ocl.oclContext, CL_MEM_READ_ONLY,
        sizeof(kernel_params), NULL, NULL);

    clEnqueueWriteBuffer(this->ocl.oclCommandQueue, this->ocl_input, CL_TRUE,
        0, sizeof(chalf) * insize, this->input_pad_half.data(), 0, NULL,
        NULL);
    clEnqueueWriteBuffer(this->ocl.oclCommandQueue, this->ocl_weights, CL_TRUE,
        0, sizeof(chalf) * wsize, this->weights_pad_half.data(), 0, NULL,
        NULL);
    clEnqueueWriteBuffer(this->ocl.oclCommandQueue, this->ocl_bias, CL_TRUE, 0,
        sizeof(chalf) * bsize, this->bias_half.data(), 0, NULL, 
        NULL);
    clEnqueueWriteBuffer(this->ocl.oclCommandQueue, this->ocl_output, CL_TRUE,
        0, sizeof(chalf) * outsize, this->hw_results_half.data(), 0, NULL, 
        NULL);
    clEnqueueWriteBuffer(this->ocl.oclCommandQueue, this->ocl_params, CL_TRUE,
        0, sizeof(kernel_params), &params[i], 0, NULL, NULL);

    for (int n = 0; n < this->batch_size[i] / params[i].numimages; ++n) {
      for (int g = 0; g < params[i].numgroups; ++g) {
        clSetKernelArg(this->ocl.oclKernel, 0, sizeof(cl_mem),
            &this->ocl_input);
        clSetKernelArg(this->ocl.oclKernel, 1, sizeof(cl_mem), 
            &this->ocl_weights);
        clSetKernelArg(this->ocl.oclKernel, 2, sizeof(cl_mem),
            &this->ocl_bias);
        clSetKernelArg(this->ocl.oclKernel, 3, sizeof(cl_mem),
            &this->ocl_output);
        clSetKernelArg(this->ocl.oclKernel, 4, sizeof(cl_mem), 
            &this->ocl_params);
        clEnqueueTask(this->ocl.oclCommandQueue, this->ocl.oclKernel, 0, NULL,
            &(events[n * params[i].numgroups + g]));
      }
    }

    clWaitForEvents(events_size, events.data());

    clEnqueueReadBuffer(this->ocl.oclCommandQueue, this->ocl_output, CL_TRUE,
        0, sizeof(chalf) * outsize, this->hw_results_half.data(), 0, NULL,
        NULL);

    toFloat(this->hw_results_half, this->hw_results);

    ref_backward_conv_layer_hwcn(this->input, this->weights,
        this->sw_results, params[i]);
    int size = params[i].outchannels * params[i].numgroups *
      params[i].inchannels * ksize * ksize;
    for (int j = 0; j < size; ++j) {
      std::cout<<this->sw_results[j]<<" "<<this->hw_results[j]<<std::endl;
      EXPECT_TRUE(checkEQ(this->sw_results[j], this->hw_results[j], 1e-1,
            1e-1));
    }
    clReleaseMemObject(this->ocl_input);
    clReleaseMemObject(this->ocl_weights);
    clReleaseMemObject(this->ocl_output);
    clReleaseMemObject(this->ocl_bias);
    clReleaseMemObject(this->ocl_params);
  }
}
