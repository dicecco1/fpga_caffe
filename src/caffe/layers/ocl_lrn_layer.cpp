#include <vector>

#include "caffe/layers/ocl_lrn_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

#ifdef USE_OCL

template <>
void OCLLRNLayer<float>::Call_ocl(const vector<Blob<float>*>& bottom,
    const vector<Blob<float>*>& top) {
  cl_event event;
  cl_int error;
  const float *bottom_data = bottom[0]->ocl_data();
  float *top_data = top[0]->mutable_ocl_data();
  error = clSetKernelArg(this->ocl_float_kernel, 0, sizeof(cl_mem),
      (const void *)&bottom_data);
  error |= clSetKernelArg(this->ocl_float_kernel, 1, sizeof(cl_mem),
      (const void *)&top_data);
  size_t global[3] = {channels_, 1, 1};
  size_t local[3] = {1, 1, 1};
  error = clEnqueueNDRangeKernel(oclCommandQueue, this->ocl_float_kernel, 3,
      NULL, (size_t *)&global, (size_t *)&local, 0, NULL, &event);
  clWaitForEvents(1, &event);
}

template <>
void OCLLRNLayer<double>::Call_ocl(const vector<Blob<double>*>& bottom,
    const vector<Blob<double>*>& top) {
  Forward_cpu(bottom, top);
}

template <typename Dtype>
void OCLLRNLayer<Dtype>::Forward_ocl(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  Dtype* scale_data = this->scale_.mutable_cpu_data();
  // start with the constant value
  for (int i = 0; i < this->scale_.count(); ++i) {
    scale_data[i] = this->k_;
  }
  if (this->layer_param_.ocl_enable())
    Call_ocl(bottom, top);
  else
    Forward_cpu(bottom, top);
}

INSTANTIATE_CLASS(OCLLRNLayer);

#endif
}  // namespace caffe


