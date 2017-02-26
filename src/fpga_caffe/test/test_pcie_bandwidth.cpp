#include <vector>
#include <string>
#include "gtest/gtest.h"

#include "fpga_caffe/test/test_fpga_caffe_main.hpp"

template <typename TypeParam>
class PCIeBandwidthTest : public OCLDeviceTest<TypeParam> {
  typedef typename TypeParam::Dtype Dtype;

 protected:
  PCIeBandwidthTest()
    : ocl("conv_layer_direct_fc.xclbin", "conv_layer_direct_fb") 
  {}
  virtual void SetUp() {
    params.resize(1);
    params[0].numgroups = 1;
    params[0].inchannels = 256;
    params[0].outchannels = 384;
    params[0].burstchannels = 256;
    params[0].rpo = 1;
    params[0].ydim = 13;
    params[0].xdim = 13;
    params[0].xtile_pad = 8;
    params[0].numimages = 128;
  }

  virtual ~PCIeBandwidthTest() {}
  
  OCLUtil ocl;
  std::vector<Dtype> input;
  std::vector<Dtype> input_pad;
  std::vector<Dtype> hw_results;
  std::vector<Dtype> hw;
  std::vector<kernel_params> params;
  cl_mem ocl_input;
};

TYPED_TEST_CASE(PCIeBandwidthTest, TestOCLDtypesAndDevices);

TYPED_TEST(PCIeBandwidthTest, TestSetup) {
  typedef typename TypeParam::Dtype Dtype;
  this->ocl.Setup_Platform();
}

TYPED_TEST(PCIeBandwidthTest, TestBurst) {
  typedef typename TypeParam::Dtype Dtype;
  this->ocl.Setup_Platform();
  std::vector<kernel_params> params = this->params;
  std::vector<cl_event> events;

  for (int i = 0; i < params.size(); ++i) {
    int insize = params[i].numimages * params[i].inchannels * params[i].ydim *
      params[i].xdim;
    // Clear input vectors
    this->input.clear();
    this->hw_results.clear();
    events.clear();
    // Resize vectors
    this->input.resize(insize, 0);
    this->hw_results.resize(insize, 0);
    // Populate vectors
    fillVector(this->input, 0.0, 1.0);

    // Create buffers
    this->ocl_input = clCreateBuffer(this->ocl.oclContext, CL_MEM_READ_ONLY,
        sizeof(Dtype) * insize, NULL, NULL);

    clEnqueueWriteBuffer(this->ocl.oclCommandQueue, this->ocl_input, CL_TRUE,
        0, sizeof(Dtype) * insize, this->input.data(), 0, NULL, NULL);

    clEnqueueReadBuffer(this->ocl.oclCommandQueue, this->ocl_input, CL_TRUE,
        0, sizeof(Dtype) * insize, this->hw_results.data(), 0, NULL, NULL);

    clReleaseMemObject(this->ocl_input);
  }
}

TYPED_TEST(PCIeBandwidthTest, TestBurst2) {
  typedef typename TypeParam::Dtype Dtype;
  this->ocl.Setup_Platform();
  std::vector<kernel_params> params = this->params;
  std::vector<cl_event> events;

  for (int i = 0; i < params.size(); ++i) {
    int insize = params[i].numimages * params[i].inchannels * params[i].ydim *
      params[i].xdim;
    // Clear input vectors
    this->input.clear();
    this->hw_results.clear();
    events.clear();
    // Resize vectors
    this->input.resize(insize, 0);
    this->hw_results.resize(insize, 0);
    // Populate vectors
    fillVector(this->input, 0.0, 1.0);

    // Create buffers
    this->ocl_input = clCreateBuffer(this->ocl.oclContext, CL_MEM_READ_ONLY,
        sizeof(Dtype) * insize, NULL, NULL);

    clEnqueueWriteBuffer(this->ocl.oclCommandQueue, this->ocl_input, CL_TRUE,
        0, sizeof(Dtype) * insize, this->input.data(), 0, NULL, NULL);

    clEnqueueReadBuffer(this->ocl.oclCommandQueue, this->ocl_input, CL_TRUE,
        0, sizeof(Dtype) * insize, this->hw_results.data(), 0, NULL, NULL);

    clReleaseMemObject(this->ocl_input);
  }
}

TYPED_TEST(PCIeBandwidthTest, TestPadBurst) {
  typedef typename TypeParam::Dtype Dtype;
  this->ocl.Setup_Platform();
  std::vector<kernel_params> params = this->params;
  std::vector<cl_event> events;

  for (int i = 0; i < params.size(); ++i) {
    int insize = params[i].numimages * params[i].inchannels * params[i].ydim *
      params[i].xdim;
    int insize_pad = (insize / params[i].xdim) * params[i].xtile_pad * 2;
    // Clear input vectors
    this->input.clear();
    this->input_pad.clear();
    this->hw_results.clear();
    events.clear();
    // Resize vectors
    this->input.resize(insize, 0);
    this->input_pad.resize(insize_pad, 0);
    this->hw_results.resize(insize_pad, 0);
    this->hw.resize(insize, 0);
    // Populate vectors
    fillVector(this->input, 0.0, 1.0);
 
    // Copy vector
    copyVector(this->input, this->input_pad, params[i].xdim,
        params[i].xtile_pad * 2);

    // Create buffers
    this->ocl_input = clCreateBuffer(this->ocl.oclContext, CL_MEM_READ_ONLY,
        sizeof(Dtype) * insize_pad, NULL, NULL);

    clEnqueueWriteBuffer(this->ocl.oclCommandQueue, this->ocl_input, CL_TRUE,
        0, sizeof(Dtype) * insize_pad, this->input_pad.data(), 0, NULL, NULL);

    clEnqueueReadBuffer(this->ocl.oclCommandQueue, this->ocl_input, CL_TRUE,
        0, sizeof(Dtype) * insize_pad, this->hw_results.data(), 0, NULL,
        NULL);
    for (int c = 0; c < params[i].numimages * params[i].inchannels *
        params[i].ydim; ++c) {
      for (int k = 0; k < params[i].xtile_pad * 2; ++k) {
        int in_mod_idx = c * params[i].xtile_pad * 2 + k;
        int in_idx = c * params[i].xdim + k;
        if (k < params[i].xdim) 
          this->hw[in_idx] = this->hw_results[in_mod_idx];
      }
    }

    clReleaseMemObject(this->ocl_input);
  }
}

TYPED_TEST(PCIeBandwidthTest, TestByChannel) {
  typedef typename TypeParam::Dtype Dtype;
  this->ocl.Setup_Platform();
  std::vector<kernel_params> params = this->params;
  std::vector<cl_event> events;

  std::vector<Dtype> in_mod;

  for (int i = 0; i < params.size(); ++i) {
    int insize = params[i].numimages * params[i].inchannels * params[i].ydim *
      params[i].xdim;
    // Clear input vectors
    this->input.clear();
    events.clear();
    // Resize vectors
    this->input.resize(insize, 0);
    this->hw_results.resize(insize, 0);
    this->hw.resize(insize, 0);
    in_mod.resize(insize, 0);
    // Populate vectors
    fillVector(this->input, 0.0, 1.0);
 
    // Create buffers
    this->ocl_input = clCreateBuffer(this->ocl.oclContext, CL_MEM_READ_ONLY,
        sizeof(Dtype) * insize, NULL, NULL);

    for (int j = 0; j < params[i].numimages; ++j) {
      for (int k = 0; k < params[i].xdim * params[i].ydim; ++k) {
        for (int c = 0; c < params[i].inchannels; ++c) {
          int in_mod_idx = (j * params[i].xdim * params[i].ydim + k) *
            params[i].inchannels + c;
          int in_idx = (j * params[i].inchannels + c) * params[i].xdim *
            params[i].ydim + k;
          in_mod[in_mod_idx] = this->input[in_idx];
        }
      }
    }

    clEnqueueWriteBuffer(this->ocl.oclCommandQueue, this->ocl_input, CL_TRUE,
        0, sizeof(Dtype) * insize, in_mod.data(), 0, NULL, NULL);

    clEnqueueReadBuffer(this->ocl.oclCommandQueue, this->ocl_input, CL_TRUE,
        0, sizeof(Dtype) * insize, this->hw_results.data(), 0, NULL,
        NULL);

    for (int j = 0; j < params[i].numimages; ++j) {
      for (int k = 0; k < params[i].xdim * params[i].ydim; ++k) {
        for (int c = 0; c < params[i].inchannels; ++c) {
          int in_mod_idx = (j * params[i].xdim * params[i].ydim + k) *
            params[i].inchannels + c;
          int in_idx = (j * params[i].inchannels + c) * params[i].xdim *
            params[i].ydim + k;
          this->hw[in_idx] = this->hw_results[in_mod_idx];
        }
      }
    }

    clReleaseMemObject(this->ocl_input);
  }
}
