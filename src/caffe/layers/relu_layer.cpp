#include <algorithm>
#include <vector>

#include "caffe/layer.hpp"
#include "caffe/vision_layers.hpp"

namespace caffe {

template <typename Dtype>
void ReLULayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->cpu_data();
  Dtype* top_data = top[0]->mutable_cpu_data();
  const int count = bottom[0]->count();
  Dtype negative_slope = this->layer_param_.relu_param().negative_slope();
  for (int i = 0; i < count; ++i) {
    top_data[i] = std::max(bottom_data[i], Dtype(0))
        + negative_slope * std::min(bottom_data[i], Dtype(0));
  }
}

template <typename Dtype>
void ReLULayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  if (propagate_down[0]) {
    const Dtype* bottom_data = bottom[0]->cpu_data();
    const Dtype* top_diff = top[0]->cpu_diff();
    Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
    const int count = bottom[0]->count();
    Dtype negative_slope = this->layer_param_.relu_param().negative_slope();
    for (int i = 0; i < count; ++i) {
      bottom_diff[i] = top_diff[i] * ((bottom_data[i] > 0)
          + negative_slope * (bottom_data[i] <= 0));
    }
  }
}

#ifdef USE_OCL
template <typename Dtype>
void ReLULayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
        const vector <Blob<Dtype>*>& top) {                                                               
  NeuronLayer<Dtype>::LayerSetUp(bottom, top);   
}  

template <>
void ReLULayer<float>::Call_ocl(const vector<Blob<float>*>& bottom, 
    const vector<Blob<float>*>& top) {
  const float* bottom_data = bottom[0]->ocl_data();
  float* top_data = top[0]->mutable_ocl_data();

  cl_int error; 
  cl_event event;

  error = clSetKernelArg(this->ocl_float_kernel, 0, sizeof(cl_mem),
      (const void *)&bottom_data);
  error = clSetKernelArg(this->ocl_float_kernel, 1, sizeof(cl_mem),
      (const void *)&top_data);
  
  float count = bottom[0]->count();
  int g_size = std::ceil(count / 4096.0);
  size_t global[3] = {g_size, 1, 1};
  size_t local[3] = {1, 1, 1};
  
  clEnqueueNDRangeKernel(oclCommandQueue, this->ocl_float_kernel, 3, NULL, 
     (size_t *)&global, (size_t *)&local, 0, NULL, &event);
  clWaitForEvents(1, &event); 
}

template <>
void ReLULayer<double>::Call_ocl(const vector<Blob<double>*>& bottom,
    const vector<Blob<double>*>& top) {
  Forward_cpu(bottom, top);
}

template <typename Dtype>
void ReLULayer<Dtype>::Forward_ocl(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) { 
  cl_int error;
  std::string path(".build_release/opencl/src/caffe/layers/");
  const char *filename = (path+this->layer_param_.xcl_name()).c_str(); 
  char *sourceStr;
  size_t sourceSize = caffe::convertToString(filename, &sourceStr);
  
  this->ocl_layer_program = clCreateProgramWithBinary(oclContext, 1,
      &oclDevices, &sourceSize, (const unsigned char **)&sourceStr, NULL, 
      &error); 
  clBuildProgram(this->ocl_layer_program, 0, NULL, NULL, NULL, &error);
  delete sourceStr;
  this->ocl_float_kernel = clCreateKernel(this->ocl_layer_program, 
      this->layer_param_.kernel_name().c_str(), &error);
  Call_ocl(bottom, top);
  clReleaseKernel(this->ocl_float_kernel);
  clReleaseProgram(this->ocl_layer_program);
}
#endif


#ifdef CPU_ONLY
STUB_GPU(ReLULayer);
#endif

INSTANTIATE_CLASS(ReLULayer);

}  // namespace caffe
