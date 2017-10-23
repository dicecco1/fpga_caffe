#include <vector>
#include <string>
#include <limits>
#include "gtest/gtest.h"

#include "fpga_caffe/test/test_fpga_caffe_main.hpp"

template <typename TypeParam>
class DropoutTest : public OCLDeviceTest<TypeParam> {
  typedef typename TypeParam::Dtype Dtype;

 protected:
  DropoutTest()
    : ocl("crpd_layer_hwcn_cpfp.xclbin", "crpd_layer_hwcn_cpfp") 
  {}
  virtual void SetUp() {
    params.resize(1);
    params[0].numgroups = 1;
    params[0].inchannels = 256;
    params[0].outchannels = 256;
    params[0].burstchannels = 16;
    params[0].rpo = 16;
    params[0].ydim = 7;
    params[0].xdim = 5;
    params[0].xtile_pad = 0;
    params[0].numimages = 256;
    params[0].rpofm = 8;
    params[0].burstydim = 2;
    params[0].numimages = 16;
    params[0].ksize = 5;
    params[0].backward = 0;
    params[0].relu = 0;
    params[0].stride = 2;
    params[0].fc = 1;
    params[0].pad = 0;
    params[0].pool = 2;
    params[0].pksize = 2;
  }

  virtual ~DropoutTest() {}
  OCLUtil ocl;
  std::vector<Dtype> input;
  std::vector<Dtype> input_pad;
  std::vector<cpfp> input_pad_cpfp;
  std::vector<Dtype> weights;
  std::vector<Dtype> weights_pad;
  std::vector<cpfp> weights_pad_cpfp;
  std::vector<Dtype> bias;
  std::vector<cpfp> bias_cpfp;
  std::vector<Dtype> hw_results;
  std::vector<cpfp> hw_results_cpfp;
  std::vector<Dtype> sw_results;
  std::vector<kernel_params> params;
  std::vector<short> relu_vals;
  std::vector<short> sw_relu_vals;
  cl_mem ocl_input;
  cl_mem ocl_weights;
  cl_mem ocl_output;
  cl_mem ocl_bias;
  cl_mem ocl_relu_vals;
  cl_mem ocl_params;

};

TYPED_TEST_CASE(DropoutTest, TestOCLDtypesAndDevices);

TYPED_TEST(DropoutTest, TestMultRandom) {
  typedef typename TypeParam::Dtype Dtype;
  this->ocl.Setup();
  std::vector<kernel_params> params = this->params;
  std::vector<cl_event> events;

  for (int i = 0; i < params.size(); ++i) {
    // Set sizes
    int insize = params[i].inchannels * params[i].xdim * params[i].ydim *
      params[i].numimages;
    int events_size = 1;
    // Resize vectors
    events.resize(1);
    // Populate vectors
    int bsize = 1;
    this->input.resize(insize, 0);
    this->input_pad_cpfp.resize(insize, cpfp(0));
    this->weights.resize(insize, 0.5);
    this->weights_pad_cpfp.resize(insize, cpfp(0));
    this->bias.resize(bsize, 0);
    this->bias_cpfp.resize(bsize, cpfp(0));
    this->sw_results.resize(insize, 0);
    this->hw_results.resize(insize, 0);
    this->hw_results_cpfp.resize(insize, cpfp(0));
    this->relu_vals.resize(insize / 16, 65535);
    events.resize(events_size);
    // Populate vectors
    fillVectorCPFP(this->input, 0.0, 1.0);
   
    toCPFP(this->input, this->input_pad_cpfp);
    toCPFP(this->weights, this->weights_pad_cpfp);
    toCPFP(this->bias, this->bias_cpfp);

    // Create buffers
    this->ocl_input = clCreateBuffer(this->ocl.oclContext, CL_MEM_READ_ONLY,
        sizeof(cpfp) * insize, NULL, NULL);
    this->ocl_weights = clCreateBuffer(this->ocl.oclContext, CL_MEM_READ_ONLY,
        sizeof(cpfp) * insize, NULL, NULL);
    this->ocl_output = clCreateBuffer(this->ocl.oclContext, CL_MEM_READ_WRITE,
        sizeof(cpfp) * insize, NULL, NULL);
    this->ocl_bias = clCreateBuffer(this->ocl.oclContext, CL_MEM_READ_ONLY,
        sizeof(cpfp) * bsize, NULL, NULL);
    this->ocl_relu_vals = clCreateBuffer(this->ocl.oclContext,
        CL_MEM_READ_WRITE, sizeof(short) * insize / 16, NULL, NULL);
    this->ocl_params = clCreateBuffer(this->ocl.oclContext, CL_MEM_READ_ONLY,
        sizeof(kernel_params), NULL, NULL);

    clEnqueueWriteBuffer(this->ocl.oclCommandQueue, this->ocl_input, CL_TRUE,
        0, sizeof(cpfp) * insize, this->input_pad_cpfp.data(), 0, NULL,
        NULL);
    clEnqueueWriteBuffer(this->ocl.oclCommandQueue, this->ocl_weights, CL_TRUE,
        0, sizeof(cpfp) * insize, this->weights_pad_cpfp.data(), 0, NULL,
        NULL);
    clEnqueueWriteBuffer(this->ocl.oclCommandQueue, this->ocl_bias, CL_TRUE, 0,
        sizeof(cpfp) * bsize, this->bias_cpfp.data(), 0, NULL, 
        NULL);
    clEnqueueWriteBuffer(this->ocl.oclCommandQueue, this->ocl_relu_vals,
        CL_TRUE, 0, sizeof(short) * insize / 16, this->relu_vals.data(), 0,
        NULL, NULL);
    clEnqueueWriteBuffer(this->ocl.oclCommandQueue, this->ocl_params, CL_TRUE,
        0, sizeof(kernel_params), &params[i], 0, NULL, NULL);

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
          &this->ocl_relu_vals);
      clSetKernelArg(this->ocl.oclKernel, 5, sizeof(cl_mem), 
          &this->ocl_params);
      clSetKernelArg(this->ocl.oclKernel, 6, sizeof(cl_int), 
          &g);
      clEnqueueTask(this->ocl.oclCommandQueue, this->ocl.oclKernel, 0, NULL,
          &(events[g]));
    }

    clWaitForEvents(events_size, events.data());

    clEnqueueReadBuffer(this->ocl.oclCommandQueue, this->ocl_output, CL_TRUE,
        0, sizeof(cpfp) * insize, this->hw_results_cpfp.data(), 0, NULL,
        NULL);
    clEnqueueReadBuffer(this->ocl.oclCommandQueue, this->ocl_relu_vals,
        CL_TRUE, 0, sizeof(short) * insize / 16, this->relu_vals.data(), 0,
        NULL, NULL);

    toFloat(this->hw_results_cpfp, this->hw_results);

    for (int j = 0; j < insize; ++j) {
      EXPECT_TRUE(checkEQ(float(this->input_pad_cpfp[j]) * float(this->weights_pad_cpfp[j]),
            this->hw_results[j], 1e-3, 1e-3));
      std::cout<<this->input[j] * this->weights[j]<<" "<<this->hw_results[j]<<std::endl;
    }
    clReleaseMemObject(this->ocl_input);
    clReleaseMemObject(this->ocl_weights);
    clReleaseMemObject(this->ocl_output);
    clReleaseMemObject(this->ocl_bias);
    clReleaseMemObject(this->ocl_relu_vals);
    clReleaseMemObject(this->ocl_params);
  }
}
