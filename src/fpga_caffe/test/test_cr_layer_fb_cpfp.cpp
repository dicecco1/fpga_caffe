#include <vector>
#include <string>
#include "gtest/gtest.h"

#include "fpga_caffe/test/test_fpga_caffe_main.hpp"

template <typename TypeParam>
class CRLayerFBCPFPTest : public OCLDeviceTest<TypeParam> {
  typedef typename TypeParam::Dtype Dtype;

 protected:
  CRLayerFBCPFPTest()
    : ocl("cr_layer_fb_cpfp.xclbin", "cr_layer_fb_cpfp") 
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
    params[0].xtile_pad = 16;
    params[0].numimages = 1;
    batch_size[0] = 1;
  }

  virtual ~CRLayerFBCPFPTest() {}
  
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
  std::vector<char> relu_vals;
  std::vector<int> batch_size;
  cl_mem ocl_input;
  cl_mem ocl_weights;
  cl_mem ocl_output;
  cl_mem ocl_bias;
  cl_mem ocl_relu_vals;
  cl_mem ocl_params;
};

TYPED_TEST_CASE(CRLayerFBCPFPTest, TestOCLDtypesAndDevices);

TYPED_TEST(CRLayerFBCPFPTest, TestCR1x1F_HALF) {
  typedef typename TypeParam::Dtype Dtype;
  this->ocl.Setup();
  int ksize = 1;
  int ksize_pad = 1;
  std::vector<kernel_params> params = this->params;
  std::vector<cl_event> events;

  for (int i = 0; i < params.size(); ++i) {
    // Set sizes
    params[i].ksize = ksize;
    params[i].backward = 0;
    params[i].relu = 0;
    int insize = params[i].numimages * params[i].inchannels * params[i].ydim *
      params[i].xdim * params[i].numgroups;
    int insize_pad = (insize / params[i].xdim) * params[i].xtile_pad * 2;
    int wsize = params[i].outchannels * params[i].numgroups *
      params[i].inchannels * ksize * ksize;

    int bc_mod = (params[i].burstchannels % 16 == 0) ? params[i].burstchannels
      : (params[i].burstchannels / 16 + 1) * 16;

    int wsize_pad = params[i].outchannels * params[i].numgroups *
      params[i].rpo * bc_mod;

    int outsize = params[i].numimages * params[i].outchannels * params[i].ydim
      * params[i].xdim * params[i].numgroups;
    int outsize_pad = (outsize / params[i].xdim) * params[i].xtile_pad * 2;
    int bsize = params[i].outchannels * params[i].numgroups;
    int events_size = this->batch_size[i] / params[i].numimages *
      params[i].numgroups;
    // Clear input vectors
    this->input.clear();
    this->input_pad.clear();
    this->input_pad_cpfp.clear();
    this->weights.clear();
    this->weights_pad.clear();
    this->weights_pad_cpfp.clear();
    this->bias.clear();
    this->bias_cpfp.clear();
    this->hw_results.clear();
    this->hw_results_cpfp.clear();
    this->sw_results.clear();
    this->relu_vals.clear();
    events.clear();
    // Resize vectors
    this->input.resize(insize, 0);
    this->input_pad.resize(insize_pad, 0);
    this->input_pad_cpfp.resize(insize_pad, cpfp(0));
    this->weights.resize(wsize, 0);
    this->weights_pad.resize(wsize_pad, 0);
    this->weights_pad_cpfp.resize(wsize_pad, cpfp(0));
    this->bias.resize(bsize, 0);
    this->bias_cpfp.resize(bsize, cpfp(0));
    this->sw_results.resize(outsize, 0);
    this->hw_results.resize(outsize_pad, 0);
    this->hw_results_cpfp.resize(outsize_pad, cpfp(0));
    this->relu_vals.resize(outsize_pad, 0);
    events.resize(events_size);
    // Populate vectors
    fillVectorCPFP(this->input, 0.0, 1.0);
    fillVectorCPFP(this->weights, -1.0, 1.0);
    fillVectorCPFP(this->bias, -1.0, 1.0);
    copyVector(this->input, this->input_pad, params[i].xdim, 
        params[i].xtile_pad * 2);
    copyWeights(this->weights, this->weights_pad, ksize, ksize_pad, params[i]);
   
    toCPFP(this->input_pad, this->input_pad_cpfp);
    toCPFP(this->weights_pad, this->weights_pad_cpfp);
    toCPFP(this->bias, this->bias_cpfp);

    // Create buffers
    this->ocl_input = clCreateBuffer(this->ocl.oclContext, CL_MEM_READ_ONLY,
        sizeof(cpfp) * insize_pad, NULL, NULL);
    this->ocl_weights = clCreateBuffer(this->ocl.oclContext, CL_MEM_READ_ONLY,
        sizeof(cpfp) * wsize_pad, NULL, NULL);
    this->ocl_output = clCreateBuffer(this->ocl.oclContext, CL_MEM_READ_WRITE,
        sizeof(cpfp) * outsize_pad, NULL, NULL);
    this->ocl_bias = clCreateBuffer(this->ocl.oclContext, CL_MEM_READ_ONLY,
        sizeof(cpfp) * bsize, NULL, NULL);
    this->ocl_relu_vals = clCreateBuffer(this->ocl.oclContext,
        CL_MEM_READ_WRITE, sizeof(char) * outsize_pad, NULL, NULL);
    this->ocl_params = clCreateBuffer(this->ocl.oclContext, CL_MEM_READ_ONLY,
        sizeof(kernel_params), NULL, NULL);

    clEnqueueWriteBuffer(this->ocl.oclCommandQueue, this->ocl_input, CL_TRUE,
        0, sizeof(cpfp) * insize_pad, this->input_pad_cpfp.data(), 0, NULL,
        NULL);
    clEnqueueWriteBuffer(this->ocl.oclCommandQueue, this->ocl_weights, CL_TRUE,
        0, sizeof(cpfp) * wsize_pad, this->weights_pad_cpfp.data(), 0, NULL,
        NULL);
    clEnqueueWriteBuffer(this->ocl.oclCommandQueue, this->ocl_bias, CL_TRUE, 0,
        sizeof(cpfp) * bsize, this->bias_cpfp.data(), 0, NULL, 
        NULL);
    clEnqueueWriteBuffer(this->ocl.oclCommandQueue, this->ocl_output, CL_TRUE,
        0, sizeof(cpfp) * outsize_pad, this->hw_results_cpfp.data(), 0, NULL, 
        NULL);
    clEnqueueWriteBuffer(this->ocl.oclCommandQueue, this->ocl_relu_vals,
        CL_TRUE, 0, sizeof(char) * outsize_pad, this->relu_vals.data(), 0,
        NULL, NULL);
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
            &this->ocl_relu_vals);
        clSetKernelArg(this->ocl.oclKernel, 5, sizeof(cl_mem), 
            &this->ocl_params);
        clSetKernelArg(this->ocl.oclKernel, 6, sizeof(cl_int), &g);
        clSetKernelArg(this->ocl.oclKernel, 7, sizeof(cl_int), &n);
        clEnqueueTask(this->ocl.oclCommandQueue, this->ocl.oclKernel, 0, NULL,
            &(events[n * params[i].numgroups + g]));
      }
    }

    clWaitForEvents(events_size, events.data());

    clEnqueueReadBuffer(this->ocl.oclCommandQueue, this->ocl_output, CL_TRUE,
        0, sizeof(cpfp) * outsize_pad, this->hw_results_cpfp.data(), 0, NULL,
        NULL);
    clEnqueueReadBuffer(this->ocl.oclCommandQueue, this->ocl_relu_vals,
        CL_TRUE, 0, sizeof(char) * outsize_pad, this->relu_vals.data(), 0,
        NULL, NULL);

    toFloat(this->hw_results_cpfp, this->hw_results);

    ref_conv_layer(this->input, this->weights, this->bias, this->sw_results,
        params[i]);
    //ref_relu_layer(this->sw_results);
    int size = params[i].numimages * params[i].outchannels *
      params[i].numgroups * params[i].ydim;
    for (int j = 0; j < size; ++j) {
      for (int x = 0; x < params[i].xtile_pad * 2; ++x) {
        if (x < params[i].xdim) {
          std::cout<<this->sw_results[j * params[i].xdim + x]<<" "<<this->hw_results[j * params[i].xtile_pad * 2 + x]<<std::endl;
          EXPECT_TRUE(checkEQ(this->sw_results[j * params[i].xdim + x],
              this->hw_results[j * params[i].xtile_pad * 2 + x], 1e-1, 1e-1));
        }
      }
    }
    clReleaseMemObject(this->ocl_input);
    clReleaseMemObject(this->ocl_weights);
    clReleaseMemObject(this->ocl_output);
    clReleaseMemObject(this->ocl_bias);
    clReleaseMemObject(this->ocl_relu_vals);
    clReleaseMemObject(this->ocl_params);
  }
}

TYPED_TEST(CRLayerFBCPFPTest, TestCR3x3F_HALF) {
  typedef typename TypeParam::Dtype Dtype;
  this->ocl.Setup();
  int ksize = 3;
  int ksize_pad = 16;
  std::vector<kernel_params> params = this->params;
  std::vector<cl_event> events;

  for (int i = 0; i < params.size(); ++i) {
    // Set sizes
    params[i].ksize = ksize;
    params[i].backward = 0;
    params[i].relu = 0;
    int insize = params[i].numimages * params[i].inchannels * params[i].ydim *
      params[i].xdim * params[i].numgroups;
    int insize_pad = (insize / params[i].xdim) * params[i].xtile_pad * 2;
    int wsize = params[i].outchannels * params[i].numgroups *
      params[i].inchannels * ksize * ksize; 
    int wsize_pad = wsize / (ksize * ksize) * ksize_pad;
    int outsize = params[i].numimages * params[i].outchannels * params[i].ydim
      * params[i].xdim * params[i].numgroups;
    int outsize_pad = (outsize / params[i].xdim) * params[i].xtile_pad * 2;
    int bsize = params[i].outchannels * params[i].numgroups;
    int events_size = this->batch_size[i] / params[i].numimages *
      params[i].numgroups;
    // Clear input vectors
    this->input.clear();
    this->input_pad.clear();
    this->input_pad_cpfp.clear();
    this->weights.clear();
    this->weights_pad.clear();
    this->weights_pad_cpfp.clear();
    this->bias.clear();
    this->bias_cpfp.clear();
    this->hw_results.clear();
    this->hw_results_cpfp.clear();
    this->sw_results.clear();
    this->relu_vals.clear();
    events.clear();
    // Resize vectors
    this->input.resize(insize, 0);
    this->input_pad.resize(insize_pad, 0);
    this->input_pad_cpfp.resize(insize_pad, cpfp(0));
    this->weights.resize(wsize, 0);
    this->weights_pad.resize(wsize_pad, 0);
    this->weights_pad_cpfp.resize(wsize_pad, cpfp(0));
    this->bias.resize(bsize, 0);
    this->bias_cpfp.resize(bsize, cpfp(0));
    this->sw_results.resize(outsize, 0);
    this->hw_results.resize(outsize_pad, 0);
    this->hw_results_cpfp.resize(outsize_pad, cpfp(0));
    this->relu_vals.resize(outsize_pad, 0);
    events.resize(events_size);
    // Populate vectors
    fillVectorCPFP(this->input, -1.0, 1.0);
    fillVectorCPFP(this->weights, -1.0, 1.0);
    fillVectorCPFP(this->bias, -1.0, 1.0);
    copyVector(this->input, this->input_pad, params[i].xdim, 
        params[i].xtile_pad * 2);
    copyWeights(this->weights, this->weights_pad, ksize, ksize_pad,
        params[i]);
   
    toCPFP(this->input_pad, this->input_pad_cpfp);
    toCPFP(this->weights_pad, this->weights_pad_cpfp);
    toCPFP(this->bias, this->bias_cpfp);

    // Create buffers
    this->ocl_input = clCreateBuffer(this->ocl.oclContext, CL_MEM_READ_ONLY,
        sizeof(cpfp) * insize_pad, NULL, NULL);
    this->ocl_weights = clCreateBuffer(this->ocl.oclContext, CL_MEM_READ_ONLY,
        sizeof(cpfp) * wsize_pad, NULL, NULL);
    this->ocl_output = clCreateBuffer(this->ocl.oclContext, CL_MEM_READ_WRITE,
        sizeof(cpfp) * outsize_pad, NULL, NULL);
    this->ocl_bias = clCreateBuffer(this->ocl.oclContext, CL_MEM_READ_ONLY,
        sizeof(cpfp) * bsize, NULL, NULL);
    this->ocl_relu_vals = clCreateBuffer(this->ocl.oclContext,
        CL_MEM_READ_WRITE, sizeof(char) * outsize_pad, NULL, NULL);
    this->ocl_params = clCreateBuffer(this->ocl.oclContext, CL_MEM_READ_ONLY,
        sizeof(kernel_params), NULL, NULL);

    clEnqueueWriteBuffer(this->ocl.oclCommandQueue, this->ocl_input, CL_TRUE,
        0, sizeof(cpfp) * insize_pad, this->input_pad_cpfp.data(), 0, NULL,
        NULL);
    clEnqueueWriteBuffer(this->ocl.oclCommandQueue, this->ocl_weights, CL_TRUE,
        0, sizeof(cpfp) * wsize_pad, this->weights_pad_cpfp.data(), 0, NULL,
        NULL);
    clEnqueueWriteBuffer(this->ocl.oclCommandQueue, this->ocl_bias, CL_TRUE, 0,
        sizeof(cpfp) * bsize, this->bias_cpfp.data(), 0, NULL, 
        NULL);
    clEnqueueWriteBuffer(this->ocl.oclCommandQueue, this->ocl_output, CL_TRUE,
        0, sizeof(cpfp) * outsize_pad, this->hw_results_cpfp.data(), 0, NULL, 
        NULL);
    clEnqueueWriteBuffer(this->ocl.oclCommandQueue, this->ocl_relu_vals,
        CL_TRUE, 0, sizeof(char) * outsize_pad, this->relu_vals.data(), 0,
        NULL, NULL);
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
            &this->ocl_relu_vals);
        clSetKernelArg(this->ocl.oclKernel, 5, sizeof(cl_mem), 
            &this->ocl_params);
        clSetKernelArg(this->ocl.oclKernel, 6, sizeof(cl_int), &g);
        clSetKernelArg(this->ocl.oclKernel, 7, sizeof(cl_int), &n);
        clEnqueueTask(this->ocl.oclCommandQueue, this->ocl.oclKernel, 0, NULL,
            &(events[n * params[i].numgroups + g]));
      }
    }

    clWaitForEvents(events_size, events.data());

    clEnqueueReadBuffer(this->ocl.oclCommandQueue, this->ocl_output, CL_TRUE,
        0, sizeof(cpfp) * outsize_pad, this->hw_results_cpfp.data(), 0, NULL,
        NULL);
    clEnqueueReadBuffer(this->ocl.oclCommandQueue, this->ocl_relu_vals,
        CL_TRUE, 0, sizeof(char) * outsize_pad, this->relu_vals.data(), 0,
        NULL, NULL);

    toFloat(this->hw_results_cpfp, this->hw_results);

    ref_conv_layer(this->input, this->weights, this->bias, this->sw_results,
        params[i]);
    //ref_relu_layer(this->sw_results);
    int size = params[i].numimages * params[i].outchannels *
      params[i].numgroups * params[i].ydim;
    for (int j = 0; j < size; ++j) {
      for (int x = 0; x < params[i].xtile_pad * 2; ++x) {
        if (x < params[i].xdim) {
          EXPECT_TRUE(checkEQ(this->sw_results[j * params[i].xdim + x],
              this->hw_results[j * params[i].xtile_pad * 2 + x], 1e-2, 1e-2));
        }
      }
    }
    clReleaseMemObject(this->ocl_input);
    clReleaseMemObject(this->ocl_weights);
    clReleaseMemObject(this->ocl_output);
    clReleaseMemObject(this->ocl_bias);
    clReleaseMemObject(this->ocl_relu_vals);
    clReleaseMemObject(this->ocl_params);
  }
}

TYPED_TEST(CRLayerFBCPFPTest, TestCR5x5F_HALF) {
  typedef typename TypeParam::Dtype Dtype;
  this->ocl.Setup();
  int ksize = 5;
  int ksize_pad = 32;
  std::vector<kernel_params> params = this->params;
  std::vector<cl_event> events;

  for (int i = 0; i < params.size(); ++i) {
    // Set sizes
    params[i].ksize = ksize;
    params[i].backward = 0;
    params[i].relu = 1;
    int insize = params[i].numimages * params[i].inchannels * params[i].ydim *
      params[i].xdim * params[i].numgroups;
    int insize_pad = (insize / params[i].xdim) * params[i].xtile_pad * 2;
    int wsize = params[i].outchannels * params[i].numgroups *
      params[i].inchannels * ksize * ksize; 
    int wsize_pad = wsize / (ksize * ksize) * ksize_pad;
    int outsize = params[i].numimages * params[i].outchannels * params[i].ydim
      * params[i].xdim * params[i].numgroups;
    int outsize_pad = (outsize / params[i].xdim) * params[i].xtile_pad * 2;
    int bsize = params[i].outchannels * params[i].numgroups;
    int events_size = this->batch_size[i] / params[i].numimages *
      params[i].numgroups;
    // Clear input vectors
    this->input.clear();
    this->input_pad.clear();
    this->input_pad_cpfp.clear();
    this->weights.clear();
    this->weights_pad.clear();
    this->weights_pad_cpfp.clear();
    this->bias.clear();
    this->bias_cpfp.clear();
    this->hw_results.clear();
    this->hw_results_cpfp.clear();
    this->sw_results.clear();
    this->relu_vals.clear();
    events.clear();
    // Resize vectors
    this->input.resize(insize, 0);
    this->input_pad.resize(insize_pad, 0);
    this->input_pad_cpfp.resize(insize_pad, cpfp(0));
    this->weights.resize(wsize, 0);
    this->weights_pad.resize(wsize_pad, 0);
    this->weights_pad_cpfp.resize(wsize_pad, cpfp(0));
    this->bias.resize(bsize, 0);
    this->bias_cpfp.resize(bsize, cpfp(0));
    this->sw_results.resize(outsize, 0);
    this->hw_results.resize(outsize_pad, 0);
    this->hw_results_cpfp.resize(outsize_pad, cpfp(0));
    this->relu_vals.resize(outsize_pad, 0);
    events.resize(events_size);
    // Populate vectors
    fillVectorCPFP(this->input, -1.0, 1.0);
    fillVectorCPFP(this->weights, -1.0, 1.0);
    fillVectorCPFP(this->bias, -1.0, 1.0);
    copyVector(this->input, this->input_pad, params[i].xdim, 
        params[i].xtile_pad * 2);
    copyWeights(this->weights, this->weights_pad, ksize, ksize_pad,
        params[i]);
   
    toCPFP(this->input_pad, this->input_pad_cpfp);
    toCPFP(this->weights_pad, this->weights_pad_cpfp);
    toCPFP(this->bias, this->bias_cpfp);

    // Create buffers
    this->ocl_input = clCreateBuffer(this->ocl.oclContext, CL_MEM_READ_ONLY,
        sizeof(cpfp) * insize_pad, NULL, NULL);
    this->ocl_weights = clCreateBuffer(this->ocl.oclContext, CL_MEM_READ_ONLY,
        sizeof(cpfp) * wsize_pad, NULL, NULL);
    this->ocl_output = clCreateBuffer(this->ocl.oclContext, CL_MEM_READ_WRITE,
        sizeof(cpfp) * outsize_pad, NULL, NULL);
    this->ocl_bias = clCreateBuffer(this->ocl.oclContext, CL_MEM_READ_ONLY,
        sizeof(cpfp) * bsize, NULL, NULL);
    this->ocl_relu_vals = clCreateBuffer(this->ocl.oclContext,
        CL_MEM_READ_WRITE, sizeof(char) * outsize_pad, NULL, NULL);
    this->ocl_params = clCreateBuffer(this->ocl.oclContext, CL_MEM_READ_ONLY,
        sizeof(kernel_params), NULL, NULL);

    clEnqueueWriteBuffer(this->ocl.oclCommandQueue, this->ocl_input, CL_TRUE,
        0, sizeof(cpfp) * insize_pad, this->input_pad_cpfp.data(), 0, NULL,
        NULL);
    clEnqueueWriteBuffer(this->ocl.oclCommandQueue, this->ocl_weights, CL_TRUE,
        0, sizeof(cpfp) * wsize_pad, this->weights_pad_cpfp.data(), 0, NULL,
        NULL);
    clEnqueueWriteBuffer(this->ocl.oclCommandQueue, this->ocl_bias, CL_TRUE, 0,
        sizeof(cpfp) * bsize, this->bias_cpfp.data(), 0, NULL, 
        NULL);
    clEnqueueWriteBuffer(this->ocl.oclCommandQueue, this->ocl_output, CL_TRUE,
        0, sizeof(cpfp) * outsize_pad, this->hw_results_cpfp.data(), 0, NULL, 
        NULL);
    clEnqueueWriteBuffer(this->ocl.oclCommandQueue, this->ocl_relu_vals,
        CL_TRUE, 0, sizeof(char) * outsize_pad, this->relu_vals.data(), 0,
        NULL, NULL);
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
            &this->ocl_relu_vals);
        clSetKernelArg(this->ocl.oclKernel, 5, sizeof(cl_mem), 
            &this->ocl_params);
        clSetKernelArg(this->ocl.oclKernel, 6, sizeof(cl_int), &g);
        clSetKernelArg(this->ocl.oclKernel, 7, sizeof(cl_int), &n);
        clEnqueueTask(this->ocl.oclCommandQueue, this->ocl.oclKernel, 0, NULL,
            &(events[n * params[i].numgroups + g]));
      }
    }

    clWaitForEvents(events_size, events.data());

    clEnqueueReadBuffer(this->ocl.oclCommandQueue, this->ocl_output, CL_TRUE,
        0, sizeof(cpfp) * outsize_pad, this->hw_results_cpfp.data(), 0, NULL,
        NULL);
    clEnqueueReadBuffer(this->ocl.oclCommandQueue, this->ocl_relu_vals,
        CL_TRUE, 0, sizeof(char) * outsize_pad, this->relu_vals.data(), 0,
        NULL, NULL);

    toFloat(this->hw_results_cpfp, this->hw_results);

    ref_conv_layer(this->input, this->weights, this->bias, this->sw_results,
        params[i]);
    ref_relu_layer(this->sw_results);
    int size = params[i].numimages * params[i].outchannels *
      params[i].numgroups * params[i].ydim;
    for (int j = 0; j < size; ++j) {
      for (int x = 0; x < params[i].xtile_pad * 2; ++x) {
        if (x < params[i].xdim) {
          std::cout<<this->sw_results[j * params[i].xdim + x]<<" "<<this->hw_results[j * params[i].xtile_pad * 2 + x]<<std::endl;
          EXPECT_TRUE(checkEQ(this->sw_results[j * params[i].xdim + x],
              this->hw_results[j * params[i].xtile_pad * 2 + x], 1e-1, 1e-1));
        }
      }
    }
    clReleaseMemObject(this->ocl_input);
    clReleaseMemObject(this->ocl_weights);
    clReleaseMemObject(this->ocl_output);
    clReleaseMemObject(this->ocl_bias);
    clReleaseMemObject(this->ocl_relu_vals);
    clReleaseMemObject(this->ocl_params);
  }
}

TYPED_TEST(CRLayerFBCPFPTest, TestCR1x1B_HALF) {
  typedef typename TypeParam::Dtype Dtype;
  this->ocl.Setup();
  int ksize = 1;
  int ksize_pad = 16;
  std::vector<kernel_params> params = this->params;
  std::vector<cl_event> events;

  for (int i = 0; i < params.size(); ++i) {
    // Set sizes
    params[i].ksize = ksize;
    params[i].relu = 1;
    params[i].backward = 1;
    int insize = params[i].numimages * params[i].inchannels * params[i].ydim *
      params[i].xdim * params[i].numgroups;
    int insize_pad = (insize / params[i].xdim) * params[i].xtile_pad * 2;
    int wsize = params[i].numimages * params[i].outchannels * params[i].ydim *
      params[i].xdim * params[i].numgroups;
    int wsize_pad = (wsize / (params[i].xdim)) * params[i].xtile_pad * 2;
    int outsize = params[i].outchannels * params[i].numgroups *
      params[i].inchannels * ksize * ksize;
    int outsize_pad = outsize / (ksize * ksize) * ksize_pad; 
    int bsize = params[i].outchannels * params[i].numgroups;
    int events_size = this->batch_size[i] / params[i].numimages *
       params[i].numgroups;
    // Clear input vectors
    this->input.clear();
    this->input_pad.clear();
    this->input_pad_cpfp.clear();
    this->weights.clear();
    this->weights_pad.clear();
    this->weights_pad_cpfp.clear();
    this->bias.clear();
    this->bias_cpfp.clear();
    this->hw_results.clear();
    this->hw_results_cpfp.clear();
    this->sw_results.clear();
    this->relu_vals.clear();
    events.clear();
    // Resize vectors
    this->input.resize(insize, 0);
    this->input_pad.resize(insize_pad, 0);
    this->input_pad_cpfp.resize(insize_pad, cpfp(0));
    this->weights.resize(wsize, 0);
    this->weights_pad.resize(wsize_pad, 0);
    this->weights_pad_cpfp.resize(wsize_pad, cpfp(0));
    this->bias.resize(bsize, 0);
    this->bias_cpfp.resize(bsize, cpfp(0));
    this->sw_results.resize(outsize, 0);
    this->hw_results.resize(outsize_pad, 0);
    this->hw_results_cpfp.resize(outsize_pad, cpfp(0));
    this->relu_vals.resize(insize_pad, 1);
    events.resize(events_size);
    // Populate vectors
    fillVectorCPFP(this->input, -1.0, 1.0);
    fillVectorCPFP(this->weights, 0.0, 1.0);
    fillVectorCPFP(this->bias, -1.0, 1.0);
    copyVector(this->input, this->input_pad, params[i].xdim, 
        params[i].xtile_pad * 2);
    copyVector(this->weights, this->weights_pad, params[i].xdim,
        params[i].xtile_pad * 2);
  
    toCPFP(this->input_pad, this->input_pad_cpfp);
    toCPFP(this->weights_pad, this->weights_pad_cpfp);
    toCPFP(this->bias, this->bias_cpfp);

    // Create buffers
    this->ocl_input = clCreateBuffer(this->ocl.oclContext, CL_MEM_READ_ONLY,
        sizeof(cpfp) * insize_pad, NULL, NULL);
    this->ocl_weights = clCreateBuffer(this->ocl.oclContext, CL_MEM_READ_ONLY,
        sizeof(cpfp) * wsize_pad, NULL, NULL);
    this->ocl_output = clCreateBuffer(this->ocl.oclContext, CL_MEM_READ_WRITE,
        sizeof(cpfp) * outsize_pad, NULL, NULL);
    this->ocl_bias = clCreateBuffer(this->ocl.oclContext, CL_MEM_READ_ONLY,
        sizeof(cpfp) * bsize, NULL, NULL);
    this->ocl_relu_vals = clCreateBuffer(this->ocl.oclContext,
        CL_MEM_READ_WRITE, sizeof(char) * insize_pad, NULL, NULL);
    this->ocl_params = clCreateBuffer(this->ocl.oclContext, CL_MEM_READ_ONLY,
        sizeof(kernel_params), NULL, NULL);

    clEnqueueWriteBuffer(this->ocl.oclCommandQueue, this->ocl_input, CL_TRUE,
        0, sizeof(cpfp) * insize_pad, this->input_pad_cpfp.data(), 0, NULL,
        NULL);
    clEnqueueWriteBuffer(this->ocl.oclCommandQueue, this->ocl_weights, CL_TRUE,
        0, sizeof(cpfp) * wsize_pad, this->weights_pad_cpfp.data(), 0, NULL,
        NULL);
    clEnqueueWriteBuffer(this->ocl.oclCommandQueue, this->ocl_bias, CL_TRUE, 0,
        sizeof(cpfp) * bsize, this->bias_cpfp.data(), 0, NULL, 
        NULL);
    clEnqueueWriteBuffer(this->ocl.oclCommandQueue, this->ocl_output, CL_TRUE,
        0, sizeof(cpfp) * outsize_pad, this->hw_results_cpfp.data(), 0, NULL, 
        NULL);
    clEnqueueWriteBuffer(this->ocl.oclCommandQueue, this->ocl_relu_vals,
        CL_TRUE, 0, sizeof(char) * insize_pad, this->relu_vals.data(), 0,
        NULL, NULL);
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
            &this->ocl_relu_vals);
        clSetKernelArg(this->ocl.oclKernel, 5, sizeof(cl_mem), 
            &this->ocl_params);
        clSetKernelArg(this->ocl.oclKernel, 6, sizeof(cl_int), &g);
        clSetKernelArg(this->ocl.oclKernel, 7, sizeof(cl_int), &n);
        clEnqueueTask(this->ocl.oclCommandQueue, this->ocl.oclKernel, 0, NULL,
            &(events[n * params[i].numgroups + g]));
      }
    }

    clWaitForEvents(events_size, events.data());

    clEnqueueReadBuffer(this->ocl.oclCommandQueue, this->ocl_output, CL_TRUE,
        0, sizeof(cpfp) * outsize_pad, this->hw_results_cpfp.data(), 0, NULL,
        NULL);

    toFloat(this->hw_results_cpfp, this->hw_results);

    ref_backward_conv_layer(this->input, this->weights,
        this->sw_results, params[i]);
    int size = params[i].outchannels * params[i].numgroups *
      params[i].inchannels;
    for (int j = 0; j < size / 2; ++j) {
      std::cout<<this->sw_results[j * 2]<<" "<<this->hw_results[j * ksize_pad]<<std::endl;
      EXPECT_TRUE(checkEQ(this->sw_results[j * 2], 
            this->hw_results[j * ksize_pad], 1e-1, 1e-1));
      std::cout<<this->sw_results[j * 2 + 1]<<" "<<this->hw_results[j * ksize_pad + 1]<<std::endl;
      EXPECT_TRUE(checkEQ(this->sw_results[j * 2 + 1], 
            this->hw_results[j * ksize_pad + 1], 1e-1, 1e-1));

    }
    clReleaseMemObject(this->ocl_input);
    clReleaseMemObject(this->ocl_weights);
    clReleaseMemObject(this->ocl_output);
    clReleaseMemObject(this->ocl_bias);
    clReleaseMemObject(this->ocl_relu_vals);
    clReleaseMemObject(this->ocl_params);
  }
}

TYPED_TEST(CRLayerFBCPFPTest, TestCR3x3B_HALF) {
  typedef typename TypeParam::Dtype Dtype;
  this->ocl.Setup();
  int ksize = 3;
  int ksize_pad = 16;
  std::vector<kernel_params> params = this->params;
  std::vector<cl_event> events;

  for (int i = 0; i < params.size(); ++i) {
    // Set sizes
    params[i].ksize = ksize;
    params[i].relu = 1;
    params[i].backward = 1;
    int insize = params[i].numimages * params[i].inchannels * params[i].ydim *
      params[i].xdim * params[i].numgroups;
    int insize_pad = (insize / params[i].xdim) * params[i].xtile_pad * 2;
    int wsize = params[i].numimages * params[i].outchannels * params[i].ydim *
      params[i].xdim * params[i].numgroups;
    int wsize_pad = (wsize / (params[i].xdim)) * params[i].xtile_pad * 2;
    int outsize = params[i].outchannels * params[i].numgroups *
      params[i].inchannels * ksize * ksize;
    int outsize_pad = outsize / (ksize * ksize) * ksize_pad; 
    int bsize = params[i].outchannels * params[i].numgroups;
    int events_size = this->batch_size[i] / params[i].numimages *
      params[i].numgroups;
    // Clear input vectors
    this->input.clear();
    this->input_pad.clear();
    this->input_pad_cpfp.clear();
    this->weights.clear();
    this->weights_pad.clear();
    this->weights_pad_cpfp.clear();
    this->bias.clear();
    this->bias_cpfp.clear();
    this->hw_results.clear();
    this->hw_results_cpfp.clear();
    this->sw_results.clear();
    this->relu_vals.clear();
    events.clear();
    // Resize vectors
    this->input.resize(insize, 0);
    this->input_pad.resize(insize_pad, 0);
    this->input_pad_cpfp.resize(insize_pad, cpfp(0));
    this->weights.resize(wsize, 0);
    this->weights_pad.resize(wsize_pad, 0);
    this->weights_pad_cpfp.resize(wsize_pad, cpfp(0));
    this->bias.resize(bsize, 0);
    this->bias_cpfp.resize(bsize, cpfp(0));
    this->sw_results.resize(outsize, 0);
    this->hw_results.resize(outsize_pad, 0);
    this->hw_results_cpfp.resize(outsize_pad, cpfp(0));
    this->relu_vals.resize(insize_pad, 1);
    events.resize(events_size);
    // Populate vectors
    fillVectorCPFP(this->input, -1.0, 1.0);
    fillVectorCPFP(this->weights, -1.0, 1.0);
    fillVectorCPFP(this->bias, -1.0, 1.0);
    copyVector(this->input, this->input_pad, params[i].xdim, 
        params[i].xtile_pad * 2);
    copyVector(this->weights, this->weights_pad, params[i].xdim,
        params[i].xtile_pad * 2);
  
    toCPFP(this->input_pad, this->input_pad_cpfp);
    toCPFP(this->weights_pad, this->weights_pad_cpfp);
    toCPFP(this->bias, this->bias_cpfp);

    // Create buffers
    this->ocl_input = clCreateBuffer(this->ocl.oclContext, CL_MEM_READ_ONLY,
        sizeof(cpfp) * insize_pad, NULL, NULL);
    this->ocl_weights = clCreateBuffer(this->ocl.oclContext, CL_MEM_READ_ONLY,
        sizeof(cpfp) * wsize_pad, NULL, NULL);
    this->ocl_output = clCreateBuffer(this->ocl.oclContext, CL_MEM_READ_WRITE,
        sizeof(cpfp) * outsize_pad, NULL, NULL);
    this->ocl_bias = clCreateBuffer(this->ocl.oclContext, CL_MEM_READ_ONLY,
        sizeof(cpfp) * bsize, NULL, NULL);
    this->ocl_relu_vals = clCreateBuffer(this->ocl.oclContext,
        CL_MEM_READ_WRITE, sizeof(char) * insize_pad, NULL, NULL);
    this->ocl_params = clCreateBuffer(this->ocl.oclContext, CL_MEM_READ_ONLY,
        sizeof(kernel_params), NULL, NULL);

    clEnqueueWriteBuffer(this->ocl.oclCommandQueue, this->ocl_input, CL_TRUE,
        0, sizeof(cpfp) * insize_pad, this->input_pad_cpfp.data(), 0, NULL,
        NULL);
    clEnqueueWriteBuffer(this->ocl.oclCommandQueue, this->ocl_weights, CL_TRUE,
        0, sizeof(cpfp) * wsize_pad, this->weights_pad_cpfp.data(), 0, NULL,
        NULL);
    clEnqueueWriteBuffer(this->ocl.oclCommandQueue, this->ocl_bias, CL_TRUE, 0,
        sizeof(cpfp) * bsize, this->bias_cpfp.data(), 0, NULL, 
        NULL);
    clEnqueueWriteBuffer(this->ocl.oclCommandQueue, this->ocl_output, CL_TRUE,
        0, sizeof(cpfp) * outsize_pad, this->hw_results_cpfp.data(), 0, NULL, 
        NULL);
    clEnqueueWriteBuffer(this->ocl.oclCommandQueue, this->ocl_relu_vals,
        CL_TRUE, 0, sizeof(char) * insize_pad, this->relu_vals.data(), 0,
        NULL, NULL);
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
            &this->ocl_relu_vals);
        clSetKernelArg(this->ocl.oclKernel, 5, sizeof(cl_mem), 
            &this->ocl_params);
        clSetKernelArg(this->ocl.oclKernel, 6, sizeof(cl_int), &g);
        clSetKernelArg(this->ocl.oclKernel, 7, sizeof(cl_int), &n);
        clEnqueueTask(this->ocl.oclCommandQueue, this->ocl.oclKernel, 0, NULL,
            &(events[n * params[i].numgroups + g]));
      }
    }

    clWaitForEvents(events_size, events.data());

    clEnqueueReadBuffer(this->ocl.oclCommandQueue, this->ocl_output, CL_TRUE,
        0, sizeof(cpfp) * outsize_pad, this->hw_results_cpfp.data(), 0, NULL,
        NULL);

    toFloat(this->hw_results_cpfp, this->hw_results);

    ref_backward_conv_layer(this->input, this->weights,
        this->sw_results, params[i]);
    int size = params[i].outchannels * params[i].numgroups *
      params[i].inchannels;
    for (int j = 0; j < size; ++j) {
      for (int k = 0; k < ksize * ksize; ++k) {
        EXPECT_TRUE(checkEQ(this->sw_results[j * ksize * ksize + k], 
              this->hw_results[j * ksize_pad + k], 1e-2, 1e-2));
      }
    }
    clReleaseMemObject(this->ocl_input);
    clReleaseMemObject(this->ocl_weights);
    clReleaseMemObject(this->ocl_output);
    clReleaseMemObject(this->ocl_bias);
    clReleaseMemObject(this->ocl_relu_vals);
    clReleaseMemObject(this->ocl_params);
  }
}

TYPED_TEST(CRLayerFBCPFPTest, TestCR5x5B_HALF) {
  typedef typename TypeParam::Dtype Dtype;
  this->ocl.Setup();
  int ksize = 5;
  int ksize_pad = 32;
  std::vector<kernel_params> params = this->params;
  std::vector<cl_event> events;

  for (int i = 0; i < params.size(); ++i) {
    // Set sizes
    params[i].ksize = ksize;
    params[i].relu = 0;
    params[i].backward = 1;
    int insize = params[i].numimages * params[i].inchannels * params[i].ydim *
      params[i].xdim * params[i].numgroups;
    int insize_pad = (insize / params[i].xdim) * params[i].xtile_pad * 2;
    int wsize = params[i].numimages * params[i].outchannels * params[i].ydim *
      params[i].xdim * params[i].numgroups;
    int wsize_pad = (wsize / (params[i].xdim)) * params[i].xtile_pad * 2;
    int outsize = params[i].outchannels * params[i].numgroups *
      params[i].inchannels * ksize * ksize;
    int outsize_pad = outsize / (ksize * ksize) * ksize_pad; 
    int bsize = params[i].outchannels * params[i].numgroups;
    int events_size = this->batch_size[i] / params[i].numimages *
       params[i].numgroups;
    // Clear input vectors
    this->input.clear();
    this->input_pad.clear();
    this->input_pad_cpfp.clear();
    this->weights.clear();
    this->weights_pad.clear();
    this->weights_pad_cpfp.clear();
    this->bias.clear();
    this->bias_cpfp.clear();
    this->hw_results.clear();
    this->hw_results_cpfp.clear();
    this->sw_results.clear();
    this->relu_vals.clear();
    events.clear();
    // Resize vectors
    this->input.resize(insize, 0);
    this->input_pad.resize(insize_pad, 0);
    this->input_pad_cpfp.resize(insize_pad, cpfp(0));
    this->weights.resize(wsize, 0);
    this->weights_pad.resize(wsize_pad, 0);
    this->weights_pad_cpfp.resize(wsize_pad, cpfp(0));
    this->bias.resize(bsize, 0);
    this->bias_cpfp.resize(bsize, cpfp(0));
    this->sw_results.resize(outsize, 0);
    this->hw_results.resize(outsize_pad, 0);
    this->hw_results_cpfp.resize(outsize_pad, cpfp(0));
    this->relu_vals.resize(insize_pad, 0);
    events.resize(events_size);
    // Populate vectors
    fillVectorCPFP(this->input, -1.0, 1.0);
    fillVectorCPFP(this->weights, -1.0, 1.0);
    //fillVectorCPFP(this->bias, -1.0, 1.0);
    copyVector(this->input, this->input_pad, params[i].xdim, 
        params[i].xtile_pad * 2);
    copyVector(this->weights, this->weights_pad, params[i].xdim,
        params[i].xtile_pad * 2);
  
    toCPFP(this->input_pad, this->input_pad_cpfp);
    toCPFP(this->weights_pad, this->weights_pad_cpfp);
    toCPFP(this->bias, this->bias_cpfp);

    // Create buffers
    this->ocl_input = clCreateBuffer(this->ocl.oclContext, CL_MEM_READ_ONLY,
        sizeof(cpfp) * insize_pad, NULL, NULL);
    this->ocl_weights = clCreateBuffer(this->ocl.oclContext, CL_MEM_READ_ONLY,
        sizeof(cpfp) * wsize_pad, NULL, NULL);
    this->ocl_output = clCreateBuffer(this->ocl.oclContext, CL_MEM_READ_WRITE,
        sizeof(cpfp) * outsize_pad, NULL, NULL);
    this->ocl_bias = clCreateBuffer(this->ocl.oclContext, CL_MEM_READ_ONLY,
        sizeof(cpfp) * bsize, NULL, NULL);
    this->ocl_relu_vals = clCreateBuffer(this->ocl.oclContext,
        CL_MEM_READ_WRITE, sizeof(char) * insize_pad, NULL, NULL);
    this->ocl_params = clCreateBuffer(this->ocl.oclContext, CL_MEM_READ_ONLY,
        sizeof(kernel_params), NULL, NULL);

    clEnqueueWriteBuffer(this->ocl.oclCommandQueue, this->ocl_input, CL_TRUE,
        0, sizeof(cpfp) * insize_pad, this->input_pad_cpfp.data(), 0, NULL,
        NULL);
    clEnqueueWriteBuffer(this->ocl.oclCommandQueue, this->ocl_weights, CL_TRUE,
        0, sizeof(cpfp) * wsize_pad, this->weights_pad_cpfp.data(), 0, NULL,
        NULL);
    clEnqueueWriteBuffer(this->ocl.oclCommandQueue, this->ocl_bias, CL_TRUE, 0,
        sizeof(cpfp) * bsize, this->bias_cpfp.data(), 0, NULL, 
        NULL);
    clEnqueueWriteBuffer(this->ocl.oclCommandQueue, this->ocl_output, CL_TRUE,
        0, sizeof(cpfp) * outsize_pad, this->hw_results_cpfp.data(), 0, NULL, 
        NULL);
    clEnqueueWriteBuffer(this->ocl.oclCommandQueue, this->ocl_relu_vals,
        CL_TRUE, 0, sizeof(char) * insize_pad, this->relu_vals.data(), 0, NULL,
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
            &this->ocl_relu_vals);
        clSetKernelArg(this->ocl.oclKernel, 5, sizeof(cl_mem), 
            &this->ocl_params);
        clSetKernelArg(this->ocl.oclKernel, 6, sizeof(cl_int), &g);
        clSetKernelArg(this->ocl.oclKernel, 7, sizeof(cl_int), &n);
        clEnqueueTask(this->ocl.oclCommandQueue, this->ocl.oclKernel, 0, NULL,
            &(events[n * params[i].numgroups + g]));
      }
    }

    clWaitForEvents(events_size, events.data());

    clEnqueueReadBuffer(this->ocl.oclCommandQueue, this->ocl_output, CL_TRUE,
        0, sizeof(cpfp) * outsize_pad, this->hw_results_cpfp.data(), 0, NULL,
        NULL);

    toFloat(this->hw_results_cpfp, this->hw_results);

    ref_backward_conv_layer(this->input, this->weights,
        this->sw_results, params[i]);
    int size = params[i].outchannels * params[i].numgroups *
      params[i].inchannels;
    for (int j = 0; j < size; ++j) {
      int woff = j * ksize * ksize;
      int wtoff = j * ksize_pad;
      for (int k = 0; k < ksize; ++k) {
        for (int l = 0; l < 3; ++l) {
          EXPECT_TRUE(checkEQ(this->sw_results[woff + k * 5 + l], 
                this->hw_results[wtoff + k * 3 + l], 1e-1, 1e-1));
          if (l < 2) {
            EXPECT_TRUE(checkEQ(this->sw_results[woff + k * 5 + l + 3], 
                  this->hw_results[wtoff + k * 3 + l + 16], 1e-1, 1e-1));
          }
        }
      }
    }
    clReleaseMemObject(this->ocl_input);
    clReleaseMemObject(this->ocl_weights);
    clReleaseMemObject(this->ocl_output);
    clReleaseMemObject(this->ocl_bias);
    clReleaseMemObject(this->ocl_relu_vals);
    clReleaseMemObject(this->ocl_params);
  }
}
/*
TYPED_TEST(CRLayerFBCPFPTest, TestFCF_HALF) {
  typedef typename TypeParam::Dtype Dtype;
  this->ocl.Setup();
  int ksize = 1;
  std::vector<kernel_params> params = this->params;
  std::vector<cl_event> events;

  int i = 0;

  // Set sizes
  params[i].ksize = ksize;
  params[i].backward = 0;
  params[i].relu = 1;
  params[i].fc = 1;
  params[i].numgroups = 1;
  params[i].inchannels = 1;
  params[i].outchannels = 1024;
  params[i].burstchannels = 1;
  params[i].rpo = 1;
  params[i].ydim = 1;
  params[i].xdim = 4096;
  params[i].rpofm = 1;
  params[i].burstydim = 1;
  params[i].xtile_pad = 2048;
  params[i].numimages = 64;
  int batch = 64;

  int insize = params[i].numimages * params[i].xtile_pad * 2;
  int insize_pad = insize;
  int wsize = params[i].outchannels * params[i].xtile_pad * 2; 
  int wsize_pad = wsize;
  int outsize = params[i].numimages * params[i].outchannels;
  int outsize_pad = outsize;
  int bsize = params[i].outchannels;
  int events_size = batch / params[i].numimages;
  std::vector<cpfp> temp;
  // Clear input vectors
  this->input.clear();
  this->input_pad.clear();
  this->input_pad_cpfp.clear();
  this->weights.clear();
  this->weights_pad.clear();
  this->weights_pad_cpfp.clear();
  this->bias.clear();
  this->bias_cpfp.clear();
  this->hw_results.clear();
  this->hw_results_cpfp.clear();
  temp.clear();
  this->sw_results.clear();
  this->relu_vals.clear();
  events.clear();
  // Resize vectors
  this->input.resize(insize, 1.25);
  this->input_pad.resize(insize_pad, 0);
  this->input_pad_cpfp.resize(insize_pad, cpfp(0));
  this->weights.resize(wsize, 0.25);
  this->weights_pad.resize(wsize_pad, 0);
  this->weights_pad_cpfp.resize(wsize_pad, cpfp(0));
  this->bias.resize(bsize, 0.5);
  this->bias_cpfp.resize(bsize, cpfp(0));
  this->sw_results.resize(outsize, 0);
  this->hw_results.resize(outsize_pad, 0);
  temp.resize(outsize_pad, 0);
  this->hw_results_cpfp.resize(outsize_pad, cpfp(0));
  this->relu_vals.resize(outsize_pad, 0);
  events.resize(events_size);
  // Populate vectors
  //fillVectorCPFP(this->input, 0.0, 1.0);
  //fillVectorCPFP(this->weights, -1.0, 1.0);
  //fillVectorCPFP(this->bias, 0.0, 1.0);
  copyVector(this->input, this->input_pad, 1, 1);
  copyVector(this->weights, this->weights_pad, 1, 1);

  toCPFP(this->input_pad, this->input_pad_cpfp);
  toCPFP(this->weights_pad, this->weights_pad_cpfp);
  toCPFP(this->bias, this->bias_cpfp);

  // Create buffers
  this->ocl_input = clCreateBuffer(this->ocl.oclContext, CL_MEM_READ_ONLY,
      sizeof(cpfp) * insize_pad, NULL, NULL);
  this->ocl_weights = clCreateBuffer(this->ocl.oclContext, CL_MEM_READ_ONLY,
      sizeof(cpfp) * wsize_pad, NULL, NULL);
  this->ocl_output = clCreateBuffer(this->ocl.oclContext, CL_MEM_READ_WRITE,
      sizeof(cpfp) * outsize_pad, NULL, NULL);
  this->ocl_bias = clCreateBuffer(this->ocl.oclContext, CL_MEM_READ_ONLY,
      sizeof(cpfp) * bsize, NULL, NULL);
  this->ocl_relu_vals = clCreateBuffer(this->ocl.oclContext,
      CL_MEM_READ_WRITE, sizeof(char) * outsize_pad, NULL, NULL);
  this->ocl_params = clCreateBuffer(this->ocl.oclContext, CL_MEM_READ_ONLY,
      sizeof(kernel_params), NULL, NULL);

  clEnqueueWriteBuffer(this->ocl.oclCommandQueue, this->ocl_input, CL_TRUE,
      0, sizeof(cpfp) * insize_pad, this->input_pad_cpfp.data(), 0, NULL,
      NULL);
  clEnqueueWriteBuffer(this->ocl.oclCommandQueue, this->ocl_weights, CL_TRUE,
      0, sizeof(cpfp) * wsize_pad, this->weights_pad_cpfp.data(), 0, NULL,
      NULL);
  clEnqueueWriteBuffer(this->ocl.oclCommandQueue, this->ocl_bias, CL_TRUE, 0,
      sizeof(cpfp) * bsize, this->bias_cpfp.data(), 0, NULL, NULL);
  clEnqueueWriteBuffer(this->ocl.oclCommandQueue, this->ocl_output, CL_TRUE,
      0, sizeof(cpfp) * outsize_pad, this->hw_results_cpfp.data(), 0, NULL, 
      NULL);
  clEnqueueWriteBuffer(this->ocl.oclCommandQueue, this->ocl_relu_vals,
      CL_TRUE, 0, sizeof(char) * outsize_pad, this->relu_vals.data(), 0,
      NULL, NULL);
  clEnqueueWriteBuffer(this->ocl.oclCommandQueue, this->ocl_params, CL_TRUE,
      0, sizeof(kernel_params), &params[i], 0, NULL, NULL);

  for (int n = 0; n < batch / params[i].numimages; ++n) {
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
      clSetKernelArg(this->ocl.oclKernel, 6, sizeof(cl_int), &g);
      clSetKernelArg(this->ocl.oclKernel, 7, sizeof(cl_int), &n);
      clEnqueueTask(this->ocl.oclCommandQueue, this->ocl.oclKernel, 0, NULL,
          &(events[n * params[i].numgroups + g]));
    }
  }

  clWaitForEvents(events_size, events.data());

  clEnqueueReadBuffer(this->ocl.oclCommandQueue, this->ocl_output, CL_TRUE,
      0, sizeof(cpfp) * outsize_pad, this->hw_results_cpfp.data(), 0, NULL,
      NULL);
  clEnqueueReadBuffer(this->ocl.oclCommandQueue, this->ocl_relu_vals,
      CL_TRUE, 0, sizeof(char) * outsize_pad, this->relu_vals.data(), 0,
      NULL, NULL);

  for (int j = 0; j < params[i].numimages; ++j)
    for (int k = 0; k < params[i].outchannels; ++k) 
      temp[j * params[i].outchannels + k] =
        this->hw_results_cpfp[k * params[i].numimages + j];
  toFloat(temp, this->hw_results);

  ref_fc_layer(this->input, this->weights, this->bias, this->sw_results,
      params[i]);

  ref_relu_layer(this->sw_results);

  int size = params[i].numimages * params[i].outchannels;
  for (int j = 0; j < size; ++j) {
    EXPECT_TRUE(checkEQ(this->sw_results[j], this->hw_results[j], 1e-1,
          1e-1)); 
  }
  clReleaseMemObject(this->ocl_input);
  clReleaseMemObject(this->ocl_weights);
  clReleaseMemObject(this->ocl_output);
  clReleaseMemObject(this->ocl_bias);
  clReleaseMemObject(this->ocl_relu_vals);
  clReleaseMemObject(this->ocl_params);
}

TYPED_TEST(CRLayerFBCPFPTest, TestFCBias_Update_HALF) {
  typedef typename TypeParam::Dtype Dtype;
  this->ocl.Setup();
  int ksize = 1;
  int ksize_pad = 16;
  std::vector<kernel_params> params = this->params;
  std::vector<cl_event> events;

  int i = 0;

  // Set sizes
  params[i].ksize = ksize;
  params[i].backward = 0;
  params[i].relu = 0;
  params[i].fc = 0;
  params[i].numgroups = 1;
  params[i].inchannels = 48;
  params[i].outchannels = 1;
  params[i].burstchannels = 48;
  params[i].rpo = 1;
  params[i].ydim = 64;
  params[i].xdim = 16;
  params[i].rpofm = 1;
  params[i].burstydim = 64;
  params[i].xtile_pad = 8;
  params[i].numimages = 1;
  int batch = 1;

  int insize = params[i].inchannels * params[i].xdim * params[i].ydim;
  int insize_pad = insize;
  int wsize = params[i].inchannels * ksize_pad;
  int wsize_pad = wsize;
  int outsize = params[i].ydim * params[i].xdim;
  int outsize_pad = outsize;
  int bsize = params[i].outchannels;
  int events_size = batch / params[i].numimages;
  // Clear input vectors
  this->input.clear();
  this->input_pad.clear();
  this->input_pad_cpfp.clear();
  this->weights.clear();
  this->weights_pad.clear();
  this->weights_pad_cpfp.clear();
  this->bias.clear();
  this->bias_cpfp.clear();
  this->hw_results.clear();
  this->hw_results_cpfp.clear();
  this->sw_results.clear();
  this->relu_vals.clear();
  events.clear();
  // Resize vectors
  this->input.resize(insize, 0);
  this->input_pad.resize(insize_pad, 0);
  this->input_pad_cpfp.resize(insize_pad, cpfp(0));
  this->weights.resize(wsize, 1);
  this->weights_pad.resize(wsize_pad, 1);
  this->weights_pad_cpfp.resize(wsize_pad, cpfp(0));
  this->bias.resize(bsize, 0);
  this->bias_cpfp.resize(bsize, cpfp(0));
  this->sw_results.resize(outsize, 0);
  this->hw_results.resize(outsize_pad, 0);
  this->hw_results_cpfp.resize(outsize_pad, cpfp(0));
  this->relu_vals.resize(outsize_pad, 0);
  events.resize(events_size);
  // Populate vectors
  fillVectorCPFP(this->input, 0.0, 1.0);
  copyVector(this->input, this->input_pad, 1, 1);
  copyVector(this->weights, this->weights_pad, 1, 1);

  toCPFP(this->input_pad, this->input_pad_cpfp);
  toCPFP(this->weights_pad, this->weights_pad_cpfp);
  toCPFP(this->bias, this->bias_cpfp);

  // Create buffers
  this->ocl_input = clCreateBuffer(this->ocl.oclContext, CL_MEM_READ_ONLY,
      sizeof(cpfp) * insize_pad, NULL, NULL);
  this->ocl_weights = clCreateBuffer(this->ocl.oclContext, CL_MEM_READ_ONLY,
      sizeof(cpfp) * wsize_pad, NULL, NULL);
  this->ocl_output = clCreateBuffer(this->ocl.oclContext, CL_MEM_READ_WRITE,
      sizeof(cpfp) * outsize_pad, NULL, NULL);
  this->ocl_bias = clCreateBuffer(this->ocl.oclContext, CL_MEM_READ_ONLY,
      sizeof(cpfp) * bsize, NULL, NULL);
  this->ocl_relu_vals = clCreateBuffer(this->ocl.oclContext,
      CL_MEM_READ_WRITE, sizeof(char) * outsize_pad, NULL, NULL);
  this->ocl_params = clCreateBuffer(this->ocl.oclContext, CL_MEM_READ_ONLY,
      sizeof(kernel_params), NULL, NULL);

  clEnqueueWriteBuffer(this->ocl.oclCommandQueue, this->ocl_input, CL_TRUE,
      0, sizeof(cpfp) * insize_pad, this->input_pad_cpfp.data(), 0, NULL,
      NULL);
  clEnqueueWriteBuffer(this->ocl.oclCommandQueue, this->ocl_weights, CL_TRUE,
      0, sizeof(cpfp) * wsize_pad, this->weights_pad_cpfp.data(), 0, NULL,
      NULL);
  clEnqueueWriteBuffer(this->ocl.oclCommandQueue, this->ocl_bias, CL_TRUE, 0,
      sizeof(cpfp) * bsize, this->bias_cpfp.data(), 0, NULL, NULL);
  clEnqueueWriteBuffer(this->ocl.oclCommandQueue, this->ocl_output, CL_TRUE,
      0, sizeof(cpfp) * outsize_pad, this->hw_results_cpfp.data(), 0, NULL, 
      NULL);
  clEnqueueWriteBuffer(this->ocl.oclCommandQueue, this->ocl_relu_vals,
      CL_TRUE, 0, sizeof(char) * outsize_pad, this->relu_vals.data(), 0,
      NULL, NULL);
  clEnqueueWriteBuffer(this->ocl.oclCommandQueue, this->ocl_params, CL_TRUE,
      0, sizeof(kernel_params), &params[i], 0, NULL, NULL);

  for (int n = 0; n < batch / params[i].numimages; ++n) {
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
      clSetKernelArg(this->ocl.oclKernel, 6, sizeof(cl_int), &g);
      clSetKernelArg(this->ocl.oclKernel, 7, sizeof(cl_int), &n);
      clEnqueueTask(this->ocl.oclCommandQueue, this->ocl.oclKernel, 0, NULL,
          &(events[n * params[i].numgroups + g]));
    }
  }

  clWaitForEvents(events_size, events.data());

  clEnqueueReadBuffer(this->ocl.oclCommandQueue, this->ocl_output, CL_TRUE,
      0, sizeof(cpfp) * outsize_pad, this->hw_results_cpfp.data(), 0, NULL,
      NULL);
  clEnqueueReadBuffer(this->ocl.oclCommandQueue, this->ocl_relu_vals,
      CL_TRUE, 0, sizeof(char) * outsize_pad, this->relu_vals.data(), 0,
      NULL, NULL);

  toFloat(this->hw_results_cpfp, this->hw_results);

  for (int j = 0; j < params[i].inchannels; ++j) 
    for (int k = 0; k < params[i].ydim * params[i].xdim; ++k)
      this->sw_results[k] += this->input_pad[j * params[i].ydim *
        params[i].xdim + k];

  int size = params[i].xdim * params[i].ydim;
  for (int j = 0; j < size; ++j) {
    EXPECT_TRUE(checkEQ(this->sw_results[j], this->hw_results[j], 1e-1,
          1e-1)); 
  }
  clReleaseMemObject(this->ocl_input);
  clReleaseMemObject(this->ocl_weights);
  clReleaseMemObject(this->ocl_output);
  clReleaseMemObject(this->ocl_bias);
  clReleaseMemObject(this->ocl_relu_vals);
  clReleaseMemObject(this->ocl_params);
}

TYPED_TEST(CRLayerFBCPFPTest, TestFCB_HALF) {
  typedef typename TypeParam::Dtype Dtype;
  this->ocl.Setup();
  int ksize = 1;
  std::vector<kernel_params> params = this->params;
  std::vector<cl_event> events;

  int i = 0;

  // Set sizes
  params[i].ksize = ksize;
  params[i].backward = 1;
  params[i].relu = 0;
  params[i].fc = 1;
  params[i].numgroups = 1;
  params[i].inchannels = 1;
  params[i].outchannels = 4096;
  params[i].burstchannels = 1;
  params[i].rpo = 1;
  params[i].ydim = 1;
  params[i].xdim = 1024;
  params[i].rpofm = 1;
  params[i].burstydim = 1;
  params[i].xtile_pad = 512;
  params[i].numimages = 64;
  int batch = 64;

  int insize = params[i].numimages * params[i].xtile_pad * 2;
  int insize_pad = insize;
  int wsize = params[i].numimages * params[i].outchannels; 
  int wsize_pad = wsize;
  int outsize = params[i].outchannels * params[i].xtile_pad * 2;
  int outsize_pad = outsize;
  int bsize = params[i].outchannels;
  int events_size = batch / params[i].numimages;
  std::vector<cpfp> temp;
  // Clear input vectors
  this->input.clear();
  this->input_pad.clear();
  this->input_pad_cpfp.clear();
  this->weights.clear();
  this->weights_pad.clear();
  this->weights_pad_cpfp.clear();
  this->bias.clear();
  this->bias_cpfp.clear();
  this->hw_results.clear();
  this->hw_results_cpfp.clear();
  this->sw_results.clear();
  this->relu_vals.clear();
  events.clear();
  // Resize vectors
  this->input.resize(insize, 0);
  this->input_pad.resize(insize_pad, 0);
  this->input_pad_cpfp.resize(insize_pad, cpfp(0));
  this->weights.resize(wsize, 0);
  this->weights_pad.resize(wsize_pad, 0);
  this->weights_pad_cpfp.resize(wsize_pad, cpfp(0));
  this->bias.resize(bsize, 0);
  this->bias_cpfp.resize(bsize, cpfp(0));
  this->sw_results.resize(outsize, 0);
  this->hw_results.resize(outsize_pad, 0);
  this->hw_results_cpfp.resize(outsize_pad, cpfp(0));
  this->relu_vals.resize(outsize_pad, 0);
  events.resize(events_size);
  // Populate vectors
  fillVectorCPFP(this->input, 0.0, 1.0);
  fillVectorCPFP(this->weights, -1.0, 1.0);
  fillVectorCPFP(this->bias, 0.0, 1.0);
  copyVector(this->input, this->input_pad, 1, 1);
  copyVector(this->weights, this->weights_pad, 1, 1);

  toCPFP(this->input_pad, this->input_pad_cpfp);
  toCPFP(this->weights_pad, this->weights_pad_cpfp);

  // Create buffers
  this->ocl_input = clCreateBuffer(this->ocl.oclContext, CL_MEM_READ_ONLY,
      sizeof(cpfp) * insize_pad, NULL, NULL);
  this->ocl_weights = clCreateBuffer(this->ocl.oclContext, CL_MEM_READ_ONLY,
      sizeof(cpfp) * wsize_pad, NULL, NULL);
  this->ocl_output = clCreateBuffer(this->ocl.oclContext, CL_MEM_READ_WRITE,
      sizeof(cpfp) * outsize_pad, NULL, NULL);
  this->ocl_bias = clCreateBuffer(this->ocl.oclContext, CL_MEM_READ_ONLY,
      sizeof(cpfp) * bsize, NULL, NULL);
  this->ocl_relu_vals = clCreateBuffer(this->ocl.oclContext,
      CL_MEM_READ_WRITE, sizeof(char) * outsize_pad, NULL, NULL);
  this->ocl_params = clCreateBuffer(this->ocl.oclContext, CL_MEM_READ_ONLY,
      sizeof(kernel_params), NULL, NULL);

  clEnqueueWriteBuffer(this->ocl.oclCommandQueue, this->ocl_input, CL_TRUE,
      0, sizeof(cpfp) * insize_pad, this->input_pad_cpfp.data(), 0, NULL,
      NULL);
  clEnqueueWriteBuffer(this->ocl.oclCommandQueue, this->ocl_weights, CL_TRUE,
      0, sizeof(cpfp) * wsize_pad, this->weights_pad_cpfp.data(), 0, NULL,
      NULL);
  clEnqueueWriteBuffer(this->ocl.oclCommandQueue, this->ocl_bias, CL_TRUE, 0,
      sizeof(cpfp) * bsize, this->bias_cpfp.data(), 0, NULL, NULL);
  clEnqueueWriteBuffer(this->ocl.oclCommandQueue, this->ocl_output, CL_TRUE,
      0, sizeof(cpfp) * outsize_pad, this->hw_results_cpfp.data(), 0, NULL, 
      NULL);
  clEnqueueWriteBuffer(this->ocl.oclCommandQueue, this->ocl_relu_vals,
      CL_TRUE, 0, sizeof(char) * outsize_pad, this->relu_vals.data(), 0,
      NULL, NULL);
  clEnqueueWriteBuffer(this->ocl.oclCommandQueue, this->ocl_params, CL_TRUE,
      0, sizeof(kernel_params), &params[i], 0, NULL, NULL);

  for (int n = 0; n < batch / params[i].numimages; ++n) {
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
      clSetKernelArg(this->ocl.oclKernel, 6, sizeof(cl_int), &g);
      clSetKernelArg(this->ocl.oclKernel, 7, sizeof(cl_int), &n);
      clEnqueueTask(this->ocl.oclCommandQueue, this->ocl.oclKernel, 0, NULL,
          &(events[n * params[i].numgroups + g]));
    }
  }

  clWaitForEvents(events_size, events.data());

  clEnqueueReadBuffer(this->ocl.oclCommandQueue, this->ocl_output, CL_TRUE,
      0, sizeof(cpfp) * outsize_pad, this->hw_results_cpfp.data(), 0, NULL,
      NULL);
  clEnqueueReadBuffer(this->ocl.oclCommandQueue, this->ocl_relu_vals,
      CL_TRUE, 0, sizeof(char) * outsize_pad, this->relu_vals.data(), 0,
      NULL, NULL);

  toFloat(this->hw_results_cpfp, this->hw_results);

  ref_backward_fc_layer(this->input, this->weights, this->sw_results,
    params[i]);

  int size = params[i].xtile_pad * 2 * params[i].outchannels;
  for (int j = 0; j < size; ++j) {
    EXPECT_TRUE(checkEQ(this->sw_results[j], this->hw_results[j], 1e-1, 1e-1)); 
  }
  clReleaseMemObject(this->ocl_input);
  clReleaseMemObject(this->ocl_weights);
  clReleaseMemObject(this->ocl_output);
  clReleaseMemObject(this->ocl_bias);
  clReleaseMemObject(this->ocl_relu_vals);
  clReleaseMemObject(this->ocl_params);
}*/
