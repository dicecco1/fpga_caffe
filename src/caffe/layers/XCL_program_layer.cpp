#include <vector>

#include "caffe/common_layers.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void XCLProgramLayer<Dtype>::Forward_ocl(const vector <Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {

  cl_int error;
  
  std::string path(".build_release/opencl/src/caffe/layers/");

  const char *filename = (path + this->layer_param_.xcl_name()).c_str();

  char *sourceStr;
  size_t sourceSize = caffe::convertToString(filename, &sourceStr);

  this->ocl_layer_program = clCreateProgramWithBinary(oclContext, 1, 
      &oclDevices, &sourceSize, (const unsigned char **)&sourceStr, NULL,
      &error);
  clBuildProgram(this->ocl_layer_program, 0, NULL, NULL, NULL, &error);
  delete sourceStr;
  this->ocl_float_kernel = clCreateKernel(this->ocl_layer_program,
      this->layer_param_.kernel_name().c_str(), &error);
  //Call_ocl(bottom, top);
  //clReleaseKernel(this->ocl_float_kernel);
  //clReleaseProgram(this->ocl_layer_program);

}

template <typename Dtype>
void XCLProgramLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
}

#ifdef CPU_ONLY
STUB_GPU(XCLProgramLayer);
#endif

INSTANTIATE_CLASS(XCLProgramLayer);
REGISTER_LAYER_CLASS(XCLProgram);

}  // namespace caffe
