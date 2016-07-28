#include <algorithm>
#include <cfloat>
#include <vector>

#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"

namespace caffe {

using std::min;
using std::max;

#ifdef USE_OCL
template <>
void OCLPoolingLayer<float>::Call_ocl(const vector<Blob<float>*>& bottom,
    const vector<Blob<float>*>& top) {
  const float* bottom_data = bottom[0]->ocl_data();  
  float* top_data = top[0]->mutable_ocl_data();
 
  cl_event event;
  cl_int error; 

  size_t global[3] = {bottom[0]->channels() / 8, 1, 1};
  size_t local[3] = {1, 1, 1};

  switch (this->layer_param_.pooling_param().pool()) {
  case PoolingParameter_PoolMethod_MAX:
    clSetKernelArg(this->ocl_float_kernel, 0, sizeof(cl_mem),
      (const void *)&bottom_data);
    clSetKernelArg(this->ocl_float_kernel, 1, sizeof(cl_mem),
      (const void *)&top_data);
    error = clEnqueueNDRangeKernel(oclCommandQueue, this->ocl_float_kernel, 3, 
        NULL, (size_t *)&global, (size_t *)&local, 0, NULL, &event);
    clWaitForEvents(1, &event);
    break;
  case PoolingParameter_PoolMethod_AVE:
    NOT_IMPLEMENTED;
    break;
  case PoolingParameter_PoolMethod_STOCHASTIC:
    NOT_IMPLEMENTED;  
    break;
  default:
    LOG(FATAL) << "Unknown pooling method.";    
  }
}

template <>
void OCLPoolingLayer<double>::Call_ocl(const vector<Blob<double>*>& bottom,
    const vector<Blob<double>*>& top) {
  Forward_cpu(bottom, top);
}

template <typename Dtype>
void OCLPoolingLayer<Dtype>::Forward_ocl(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) { 
  if (this->layer_param_.ocl_enable())
    Call_ocl(bottom, top);
  else
    Forward_cpu(bottom, top);
}

INSTANTIATE_CLASS(OCLPoolingLayer);

#endif
 
} // namespace caffe
