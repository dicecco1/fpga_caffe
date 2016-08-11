#include <vector>

#include "caffe/layers/ocl_inner_product_layer.hpp"
#include "caffe/filler.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

#ifdef USE_OCL
template <>
void OCLInnerProductLayer<float>::Call_ocl(const vector<Blob<float>*>& bottom,
    const vector<Blob<float>*>& top) {
  cl_event event;
  const float* bottom_data = bottom[0]->ocl_data();
  float* top_data = top[0]->mutable_ocl_data();
  const float* weight = this->blobs_[0]->ocl_data();
  clSetKernelArg(this->ocl_float_kernel, 0, sizeof(cl_mem),
      (const void *)&bottom_data);
  clSetKernelArg(this->ocl_float_kernel, 1, sizeof(cl_mem),
      (const void *)&weight);
  clSetKernelArg(this->ocl_float_kernel, 2, sizeof(cl_mem),
      (const void *)&top_data);
  size_t global[3] = {8, 1, 1};
  size_t local[3] = {1, 1, 1};
  if(this->layer_param_.kernel_name() == "fc8_layer")
    global[0] = 5;
  clEnqueueNDRangeKernel(oclCommandQueue, this->ocl_float_kernel, 3, NULL,
      (size_t *)&global, (size_t *)&local, 0, NULL, &event);
  clWaitForEvents(1, &event);
  top_data = top[0]->mutable_cpu_data();
  if (bias_term_) {
    const float *bmult = bias_multiplier_.cpu_data();
    const float *bias_vals = this->blobs_[1]->cpu_data();
    caffe_cpu_gemm(CblasNoTrans, CblasNoTrans, M_, N_, 1, (float)1.0,
        bmult, bias_vals, (float)1.0, top_data);
  }
}

template <>
void OCLInnerProductLayer<double>::Call_ocl(const vector<Blob<double>*>& bottom,
    const vector<Blob<double>*>& top) {
  Forward_cpu(bottom, top);
}

template <typename Dtype>
void OCLInnerProductLayer<Dtype>::Forward_ocl(const vector <Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  if (this->layer_param_.ocl_enable())
    Call_ocl(bottom, top);
  else
    Forward_cpu(bottom, top); 
}
INSTANTIATE_CLASS(OCLInnerProductLayer);

#endif

} // namespace caffe
