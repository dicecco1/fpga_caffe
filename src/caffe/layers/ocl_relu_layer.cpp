#include <algorithm>
#include <vector>

#include "caffe/layers/ocl_relu_layer.hpp"

namespace caffe {

#ifdef USE_OCL
template <>
void OCLReLULayer<float>::Call_ocl(const vector<Blob<float>*>& bottom,
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
void OCLReLULayer<double>::Call_ocl(const vector<Blob<double>*>& bottom,
    const vector<Blob<double>*>& top) {
  Forward_cpu(bottom, top);
}

template <typename Dtype>
void OCLReLULayer<Dtype>::Forward_ocl(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  if (this->layer_param_.ocl_enable())
    Call_ocl(bottom, top);
  else
    Forward_cpu(bottom, top);
}

INSTANTIATE_CLASS(OCLReLULayer);
#endif

}  // namespace caffe
