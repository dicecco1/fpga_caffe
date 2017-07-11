#include <string>
#include <vector>

#include "caffe/layers/XCL_program_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

#ifdef USE_OCL

template <typename Dtype>
void XCLProgramLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  program_ = true; 
}

template <typename Dtype>
void XCLProgramLayer<Dtype>::Forward_ocl(const vector <Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  XCLParameter xcl_param = this->layer_param_.xcl_param();
  if (program_) {
    cl_int error;

    string path(".build_release/opencl/src/caffe/layers/");

    const char *filename = (path + xcl_param.xcl_name()).c_str();

    char *sourceStr;
    size_t sourceSize = caffe::convertToString(filename, &sourceStr);

    clReleaseKernel(this->ocl_kernel);
    clReleaseProgram(this->ocl_layer_program);

    this->ocl_layer_program = clCreateProgramWithBinary(oclContext, 1,
        &oclDevices, &sourceSize, (const unsigned char **)&sourceStr, NULL,
        &error);
    clBuildProgram(this->ocl_layer_program, 0, NULL, NULL, NULL, &error);

    delete[] sourceStr;
    this->ocl_kernel = clCreateKernel(this->ocl_layer_program,
        xcl_param.kernel_name().c_str(), &error);
    if (xcl_param.once())
      program_ = false;
  }
}

template <typename Dtype>
void XCLProgramLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
}

INSTANTIATE_CLASS(XCLProgramLayer);
REGISTER_LAYER_CLASS(XCLProgram);
#endif

}  // namespace caffe
