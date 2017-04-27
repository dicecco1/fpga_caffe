#ifndef FPGA_CAFFE_TEST_TEST_FPGA_CAFFE_MAIN_HPP_
#define FPGA_CAFFE_TEST_TEST_FPGA_CAFFE_MAIN_HPP_

#include <glog/logging.h>
#include <gtest/gtest.h>

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <string>
#include <boost/random/mersenne_twister.hpp>
#include <boost/random/uniform_real_distribution.hpp>
#include <CL/opencl.h>
#include <fstream>
#include <string>
#include "time.h"
#include "fpga_caffe/layer.hpp"
#include "fpga_caffe/half.hpp"

using std::cout;
using std::endl;

template <typename TypeParam>
class OCLDeviceTest : public ::testing::Test {
 public:
  typedef typename TypeParam::Dtype Dtype;
 protected:
  virtual ~OCLDeviceTest() {}
};

template <typename TypeParam>
struct OCLDevice {
  typedef TypeParam Dtype;
};

typedef ::testing::Types<OCLDevice<float> >TestOCLDtypesAndDevices;

class OCLUtil {
 public:
  OCLUtil(std::string xclbin, std::string xclkernel);
  void Setup();
  void Setup_Platform();
  cl_uint oclNumPlatforms;
  std::vector<cl_platform_id> oclPlatform;
  cl_device_id oclDevices;
  cl_context oclContext;
  cl_command_queue oclCommandQueue;
  cl_kernel oclKernel;
  std::string xcl_name;
  std::string kernel;
};

int main(int argc, char** argv);

void ref_relu_layer(std::vector<float>& output);

void ref_conv_layer(std::vector<float> input, std::vector<float> weights,
    std::vector<float> bias, std::vector<float>& output, kernel_params params);

void ref_fc_layer(std::vector<float> input, std::vector<float> weights,
    std::vector<float> bias, std::vector<float>& output, kernel_params params);

void ref_backward_fc_layer(std::vector<float> input,
    std::vector<float> weights, std::vector<float>& output,
    kernel_params params);

void ref_backward_conv_layer(std::vector<float> input, 
    std::vector<float> weights, std::vector<float>& output, 
    kernel_params params);

void fillVector(std::vector<float>& input, float beg, float end);

void fillVectorHalf(std::vector<float>& input, float beg, float end);

void copyVector(std::vector<float> input, std::vector<float>& output, int xsize,
    int xsize_pad);

void copyWeights(std::vector<float> w_input, std::vector<float>& w_output,
    int ksize, int padsize, int size);

void toHalf(std::vector<float> input, std::vector<chalf>& output);

void toFloat(std::vector<chalf> input, std::vector<float>& output);

bool checkEQ(float expected, float result, float epsilon, float absError);

#endif  // FPGA_CAFFE_TEST_TEST_FPGA_CAFFE_MAIN_HPP_
