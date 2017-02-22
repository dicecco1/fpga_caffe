#include <vector>
#include <string>
#include "gtest/gtest.h"

#include "fpga_caffe/test/test_fpga_caffe_main.hpp"

template <typename TypeParam>
class ConvLayerDirectFCTest : public OCLDeviceTest<TypeParam> {
  typedef typename TypeParam::Dtype Dtype;

 protected:
  ConvLayerDirectFCTest()
    : ocl("conv_layer_direct_fc.xclbin", "conv_layer_direct_fb") 
  {}
  virtual void SetUp() {
    params.resize(3);
    params[0].numgroups = 1;
    params[0].inchannels = 9216;
    params[0].outchannels = 4096;
    params[0].burstchannels = 1;
    params[0].rpo = 1;
    params[0].ydim = 1;
    params[0].xdim = 1;
    params[0].xtile_pad = 1;
    params[0].numimages = 1;
    params[1].numgroups = 1;
    params[1].inchannels = 4096;
    params[1].outchannels = 4096;
    params[1].burstchannels = 1;
    params[1].rpo = 1;
    params[1].ydim = 1;
    params[1].xdim = 1;
    params[1].xtile_pad = 1;
    params[1].numimages = 1;
    params[2].numgroups = 1;
    params[2].inchannels = 4096;
    params[2].outchannels = 1024;
    params[2].burstchannels = 1;
    params[2].rpo = 1;
    params[2].ydim = 1;
    params[2].xdim = 1;
    params[2].xtile_pad = 1;
    params[2].numimages = 1;
  }

  virtual ~ConvLayerDirectFCTest() {}
  
  OCLUtil ocl;
  std::vector<Dtype> input;
  std::vector<Dtype> input_pad;
  std::vector<Dtype> weights;
  std::vector<Dtype> weights_pad;
  std::vector<Dtype> bias;
  std::vector<Dtype> hw_results;
  std::vector<Dtype> sw_results;
  std::vector<kernel_params> params;
  cl_mem ocl_input;
  cl_mem ocl_weights;
  cl_mem ocl_output;
  cl_mem ocl_bias;
  cl_mem ocl_params; 
};

TYPED_TEST_CASE(ConvLayerDirectFCTest, TestOCLDtypesAndDevices);

TYPED_TEST(ConvLayerDirectFCTest, TestDirectFCF) {
  typedef typename TypeParam::Dtype Dtype;
  this->ocl.Setup();
  int ksize = 1;
  std::vector<kernel_params> params = this->params;
  std::vector<cl_event> events;

  for (int i = 0; i < params.size(); ++i) {
    // Set sizes
    params[i].ksize = ksize;
    params[i].backward = 0;
    params[i].fc = 1;
    int insize = params[i].numimages * params[i].inchannels;
    int insize_pad = insize;
    int wsize = params[i].outchannels * params[i].inchannels; 
    int wsize_pad = wsize;
    int outsize = params[i].numimages * params[i].outchannels;
    int outsize_pad = outsize;
    int bsize = 1;
    // Clear input vectors
    this->input.clear();
    this->input_pad.clear();
    this->weights.clear();
    this->weights_pad.clear();
    this->bias.clear();
    this->hw_results.clear();
    this->sw_results.clear();
    events.clear();
    // Resize vectors
    this->input.resize(insize, 0);
    this->input_pad.resize(insize_pad, 0);
    this->weights.resize(wsize, 0);
    this->weights_pad.resize(wsize_pad, 0);
    this->bias.resize(bsize, 0);
    this->sw_results.resize(outsize, 0);
    this->hw_results.resize(outsize_pad, 0);
    events.resize(params[i].numimages * params[i].numgroups);
    // Populate vectors
    fillVector(this->input, 0.0, 1.0);
    fillVector(this->weights, -1.0, 1.0);
    fillVector(this->bias, -1.0, 1.0);
    copyVector(this->input, this->input_pad, 1, 1);
    copyVector(this->weights, this->weights_pad, 1, 1);
  
    // Create buffers
    this->ocl_input = clCreateBuffer(this->ocl.oclContext, CL_MEM_READ_ONLY,
        sizeof(Dtype) * insize_pad, NULL, NULL);
    this->ocl_weights = clCreateBuffer(this->ocl.oclContext, CL_MEM_READ_ONLY,
        sizeof(Dtype) * wsize_pad, NULL, NULL);
    this->ocl_output = clCreateBuffer(this->ocl.oclContext, CL_MEM_READ_WRITE,
        sizeof(Dtype) * outsize_pad, NULL, NULL);
    this->ocl_bias = clCreateBuffer(this->ocl.oclContext, CL_MEM_READ_ONLY,
        sizeof(Dtype) * bsize, NULL, NULL);
    this->ocl_params = clCreateBuffer(this->ocl.oclContext, CL_MEM_READ_ONLY,
        sizeof(kernel_params), NULL, NULL);

    clEnqueueWriteBuffer(this->ocl.oclCommandQueue, this->ocl_input, CL_TRUE,
        0, sizeof(Dtype) * insize_pad, this->input_pad.data(), 0, NULL, NULL);
    clEnqueueWriteBuffer(this->ocl.oclCommandQueue, this->ocl_weights, CL_TRUE,
        0, sizeof(Dtype) * wsize_pad, this->weights_pad.data(), 0, NULL, NULL);
    clEnqueueWriteBuffer(this->ocl.oclCommandQueue, this->ocl_bias, CL_TRUE, 0,
        sizeof(Dtype) * bsize, this->bias.data(), 0, NULL, 
        NULL);
    clEnqueueWriteBuffer(this->ocl.oclCommandQueue, this->ocl_output, CL_TRUE,
        0, sizeof(Dtype) * outsize_pad, this->hw_results.data(), 0, NULL, 
        NULL);
    clEnqueueWriteBuffer(this->ocl.oclCommandQueue, this->ocl_params, CL_TRUE,
        0, sizeof(kernel_params), &params[i], 0, NULL, NULL);

    for (int n = 0; n < params[i].numimages; ++n) {
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
        clSetKernelArg(this->ocl.oclKernel, 5, sizeof(cl_int), &g);
        clSetKernelArg(this->ocl.oclKernel, 6, sizeof(cl_int), &n);
        clEnqueueTask(this->ocl.oclCommandQueue, this->ocl.oclKernel, 0, NULL,
            &(events[n * params[i].numgroups + g]));
      }
    }

    clWaitForEvents(params[i].numimages * params[i].numgroups, events.data());

    clEnqueueReadBuffer(this->ocl.oclCommandQueue, this->ocl_output, CL_TRUE,
        0, sizeof(Dtype) * outsize_pad, this->hw_results.data(), 0, NULL,
        NULL);
    ref_fc_layer(this->input, this->weights, this->bias, this->sw_results,
        params[i]);
    int size = params[i].numimages * params[i].outchannels;
    for (int j = 0; j < size; ++j) {
      EXPECT_TRUE(checkEQ(this->sw_results[j], this->hw_results[j], 1e-3,
            1e-3)); 
    }
    clReleaseMemObject(this->ocl_input);
    clReleaseMemObject(this->ocl_weights);
    clReleaseMemObject(this->ocl_output);
    clReleaseMemObject(this->ocl_bias);
  }
}
