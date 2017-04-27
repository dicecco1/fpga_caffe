#include <vector>
#include <string>
#include <limits>
#include "gtest/gtest.h"

#include "fpga_caffe/test/test_fpga_caffe_main.hpp"

template <typename TypeParam>
class HalfTest : public OCLDeviceTest<TypeParam> {
  typedef typename TypeParam::Dtype Dtype;

 protected:
  HalfTest()
    : ocl("half_ops.xclbin", "half_ops") 
  {}
  virtual void SetUp() {
    params.resize(1);
    params[0].numgroups = 1;
    params[0].inchannels = 16384;
    params[0].outchannels = 16384;
    params[0].burstchannels = 16384;
    params[0].rpo = 1;
    params[0].ydim = 1;
    params[0].xdim = 1;
    params[0].xtile_pad = 1;
    params[0].numimages = 1;
  }

  virtual ~HalfTest() {}
  
  OCLUtil ocl;
  std::vector<Dtype> input;
  std::vector<Dtype> weights;
  std::vector<Dtype> hw_results;
  std::vector<kernel_params> params;
  cl_mem ocl_input;
  cl_mem ocl_weights;
  cl_mem ocl_output;
  cl_mem ocl_params; 
};

TYPED_TEST_CASE(HalfTest, TestOCLDtypesAndDevices);

TYPED_TEST(HalfTest, TestMult) {
  typedef typename TypeParam::Dtype Dtype;
  this->ocl.Setup();
  std::vector<kernel_params> params = this->params;
  std::vector<cl_event> events;

  int op = 0; // multiply

  for (int i = 0; i < params.size(); ++i) {
    // Set sizes
    int insize = params[i].inchannels;
    int wsize = params[i].inchannels; 
    int outsize = params[i].inchannels;
    // Clear input vectors
    this->input.clear();
    this->weights.clear();
    this->hw_results.clear();
    events.clear();
    // Resize vectors
    this->input.resize(insize, 0);
    this->weights.resize(wsize, 0);
    this->hw_results.resize(outsize, 0);
    events.resize(1);
    // Populate vectors
    fillVector(this->input, -1.0, 1.0);
    fillVector(this->weights, -1.0, 1.0);
 
    this->input[6] = 0;
    this->weights[7] = 0;
    this->input[8] = 0.769412;
    this->weights[8] = -0.769567;
    this->input[9] = -0.51234;
    this->weights[9] = 0.51234;

    // Create buffers
    this->ocl_input = clCreateBuffer(this->ocl.oclContext, CL_MEM_READ_ONLY,
        sizeof(Dtype) * insize, NULL, NULL);
    this->ocl_weights = clCreateBuffer(this->ocl.oclContext, CL_MEM_READ_ONLY,
        sizeof(Dtype) * wsize, NULL, NULL);
    this->ocl_output = clCreateBuffer(this->ocl.oclContext, CL_MEM_READ_WRITE,
        sizeof(Dtype) * outsize, NULL, NULL);
    this->ocl_params = clCreateBuffer(this->ocl.oclContext, CL_MEM_READ_ONLY,
        sizeof(kernel_params), NULL, NULL);

    clEnqueueWriteBuffer(this->ocl.oclCommandQueue, this->ocl_input, CL_TRUE,
        0, sizeof(Dtype) * insize, this->input.data(), 0, NULL, NULL);
    clEnqueueWriteBuffer(this->ocl.oclCommandQueue, this->ocl_weights, CL_TRUE,
        0, sizeof(Dtype) * wsize, this->weights.data(), 0, NULL, NULL);
    clEnqueueWriteBuffer(this->ocl.oclCommandQueue, this->ocl_output, CL_TRUE,
        0, sizeof(Dtype) * outsize, this->hw_results.data(), 0, NULL, 
        NULL);
    clEnqueueWriteBuffer(this->ocl.oclCommandQueue, this->ocl_params, CL_TRUE,
        0, sizeof(kernel_params), &params[i], 0, NULL, NULL);

    clSetKernelArg(this->ocl.oclKernel, 0, sizeof(cl_mem),
        &this->ocl_input);
    clSetKernelArg(this->ocl.oclKernel, 1, sizeof(cl_mem), 
        &this->ocl_weights);
    clSetKernelArg(this->ocl.oclKernel, 2, sizeof(cl_mem),
        &this->ocl_output);
    clSetKernelArg(this->ocl.oclKernel, 3, sizeof(cl_mem), 
        &this->ocl_params);
    clSetKernelArg(this->ocl.oclKernel, 4, sizeof(cl_int), 
        &op);

    clEnqueueTask(this->ocl.oclCommandQueue, this->ocl.oclKernel, 0, NULL,
        &events[0]);
 
    clWaitForEvents(1, events.data());

    clEnqueueReadBuffer(this->ocl.oclCommandQueue, this->ocl_output, CL_TRUE,
        0, sizeof(Dtype) * outsize, this->hw_results.data(), 0, NULL, NULL);

    int size = params[i].inchannels;
    for (int j = 0; j < size; ++j) {
      EXPECT_TRUE(checkEQ(this->input[j] * this->weights[j],
            this->hw_results[j], 1e-3, 1e-3));
    }
    clReleaseMemObject(this->ocl_input);
    clReleaseMemObject(this->ocl_weights);
    clReleaseMemObject(this->ocl_output);
  }
}

TYPED_TEST(HalfTest, TestAdd) {
  typedef typename TypeParam::Dtype Dtype;
  this->ocl.Setup();
  std::vector<kernel_params> params = this->params;
  std::vector<cl_event> events;

  int op = 1; // add

  for (int i = 0; i < params.size(); ++i) {
    // Set sizes
    int insize = params[i].inchannels;
    int wsize = params[i].inchannels; 
    int outsize = params[i].inchannels;
    // Clear input vectors
    this->input.clear();
    this->weights.clear();
    this->hw_results.clear();
    events.clear();
    // Resize vectors
    this->input.resize(insize, 0);
    this->weights.resize(wsize, 0);
    this->hw_results.resize(outsize, 0);
    events.resize(1);
    // Populate vectors
    fillVector(this->input, -1.0, 1.0);
    fillVector(this->weights, -1.0, 1.0);
 
    this->input[6] = 0;
    this->weights[7] = 0;
    this->input[8] = 0.769412;
    this->weights[8] = -0.769567;
    this->input[9] = -0.51234;
    this->weights[9] = 0.51234;
    this->input[10] = -0.263659;
    this->weights[10] = 0.0643187;

    // Create buffers
    this->ocl_input = clCreateBuffer(this->ocl.oclContext, CL_MEM_READ_ONLY,
        sizeof(Dtype) * insize, NULL, NULL);
    this->ocl_weights = clCreateBuffer(this->ocl.oclContext, CL_MEM_READ_ONLY,
        sizeof(Dtype) * wsize, NULL, NULL);
    this->ocl_output = clCreateBuffer(this->ocl.oclContext, CL_MEM_READ_WRITE,
        sizeof(Dtype) * outsize, NULL, NULL);
    this->ocl_params = clCreateBuffer(this->ocl.oclContext, CL_MEM_READ_ONLY,
        sizeof(kernel_params), NULL, NULL);

    clEnqueueWriteBuffer(this->ocl.oclCommandQueue, this->ocl_input, CL_TRUE,
        0, sizeof(Dtype) * insize, this->input.data(), 0, NULL, NULL);
    clEnqueueWriteBuffer(this->ocl.oclCommandQueue, this->ocl_weights, CL_TRUE,
        0, sizeof(Dtype) * wsize, this->weights.data(), 0, NULL, NULL);
    clEnqueueWriteBuffer(this->ocl.oclCommandQueue, this->ocl_output, CL_TRUE,
        0, sizeof(Dtype) * outsize, this->hw_results.data(), 0, NULL, 
        NULL);
    clEnqueueWriteBuffer(this->ocl.oclCommandQueue, this->ocl_params, CL_TRUE,
        0, sizeof(kernel_params), &params[i], 0, NULL, NULL);

    clSetKernelArg(this->ocl.oclKernel, 0, sizeof(cl_mem),
        &this->ocl_input);
    clSetKernelArg(this->ocl.oclKernel, 1, sizeof(cl_mem), 
        &this->ocl_weights);
    clSetKernelArg(this->ocl.oclKernel, 2, sizeof(cl_mem),
        &this->ocl_output);
    clSetKernelArg(this->ocl.oclKernel, 3, sizeof(cl_mem), 
        &this->ocl_params);
    clSetKernelArg(this->ocl.oclKernel, 4, sizeof(cl_int), 
        &op);
    clEnqueueTask(this->ocl.oclCommandQueue, this->ocl.oclKernel, 0, NULL,
        &events[0]);
 
    clWaitForEvents(1, events.data());

    clEnqueueReadBuffer(this->ocl.oclCommandQueue, this->ocl_output, CL_TRUE,
        0, sizeof(Dtype) * outsize, this->hw_results.data(), 0, NULL, NULL);

    int size = params[i].inchannels;
    for (int j = 0; j < size; ++j) {
      EXPECT_TRUE(checkEQ(this->input[j] + this->weights[j],
            this->hw_results[j], 1e-3, 1e-3));
    }
    clReleaseMemObject(this->ocl_input);
    clReleaseMemObject(this->ocl_weights);
    clReleaseMemObject(this->ocl_output);
  }
}
