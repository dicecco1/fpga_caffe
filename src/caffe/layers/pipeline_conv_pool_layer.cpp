#include <vector>

#include "caffe/vision_layers.hpp"

namespace caffe {

#ifdef USE_OCL
template <typename Dtype>
PipelineConvPoolLayer<Dtype>::~PipelineConvPoolLayer() {
}
#endif

/*
 * Modify compute_output_shape to reflect entire pipeline
 */
template <typename Dtype>
void PipelineConvPoolLayer<Dtype>::compute_output_shape() {
  // Parse pooling parameters
  PoolingParameter pool_param = this->layer_param_.pooling_param();
  const int pool_kernel_shape_data = pool_param.kernel_size();
  const int pool_stride_data = pool_param.stride();
  // Parse convolution parameters
  const int* conv_kernel_shape_data = this->kernel_shape_.cpu_data();
  const int* conv_stride_data = this->stride_.cpu_data();
  const int* conv_pad_data = this->pad_.cpu_data();

  this->output_shape_.clear();
  for (int i = 0; i < this->num_spatial_axes_; ++i) {
    // i + 1 to skip channel axis
    const int input_dim = this->input_shape(i + 1);
    int conv_output_dim = (input_dim + 2 * conv_pad_data[i] - conv_kernel_shape_data[i])
        / conv_stride_data[i] + 1;
	const int output_dim =  (conv_output_dim - pool_kernel_shape_data) / pool_stride_data + 1;
    this->output_shape_.push_back(output_dim + conv_output_dim); // placeholder, fix later
  }

}

template <typename Dtype>
void PipelineConvPoolLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {

}

template <typename Dtype>
void PipelineConvPoolLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {

}

#ifdef USE_OCL
template <>
void PipelineConvPoolLayer<float>::Call_ocl(const vector<Blob<float>*>& bottom,
    const vector<Blob<float>*>& top) { 
  
  // ocl_data returns pointer to cl_mem buffer
  const vector<shared_ptr<Blob<float>>> weight = this->blobs_;
  const float* weight_data = weight[0]->ocl_data(); 			

  // ************ CONV ************
  cl_event event;
  cl_int error;
  int burst_size = 8;

  size_t global[3] = {top[0]->channels() / group_, 1, 1};
  size_t local[3] = {1, 1, 1};

  // bottom.size() is 1 unless using batch processing
  //for (int i = 0; i < bottom.size(); i++) { 
    const float *bottom_data = bottom[0]->ocl_data();
    float *top_data_TEMP = top[0]->mutable_ocl_data();
    clSetKernelArg(this->ocl_float_kernel, 0, sizeof(cl_mem),
        (const void *)&bottom_data);
    clSetKernelArg(this->ocl_float_kernel, 1, sizeof(cl_mem),
        (const void *)&weight_data);
    clSetKernelArg(this->ocl_float_kernel, 2, sizeof(cl_mem), 
        (const void *)&top_data_TEMP);
    clEnqueueNDRangeKernel(oclCommandQueue, this->ocl_float_kernel, 3, NULL,
       (size_t *)&global, (size_t *)&local, 0, NULL, &event);
    //clWaitForEvents(1, &event); 
  //}
  // add biases on CPU side for simplicity
  
  //for (int i = 0; i < bottom.size(); i++) {
  //  float *top_data = top[i]->mutable_cpu_data();
  //  if (this->bias_term_) {
  //    const float* bias_data = weight[1]->cpu_data();
  //    for (int n = 0; n < top[i]->num(); n++) {
  //      for (int o = 0; o < top[i]->channels(); o++) {
  //        for (int y = 0; y < top[i]->height(); y++) {
  //          for (int x = 0; x < top[i]->width(); x++) {
  //            top_data[top[i]->offset(n, o, y, x)] += bias_data[o];
  //          }
  //        }
  //      }
  //    }
  //  }
  //}
  
  // ************ POOL ************

  const float* bottom_data_TEMP = top[0]->ocl_data();  
  float* top_data = top[0]->mutable_ocl_data();
 
  //cl_event event;
  //cl_int error; 

  //size_t global[3] = {bottom[0]->channels() / 8, 1, 1};
  //size_t local[3] = {1, 1, 1};

  clSetKernelArg(this->ocl_float_kernel_2, 0, sizeof(cl_mem),
      (const void *)&bottom_data_TEMP);
  clSetKernelArg(this->ocl_float_kernel_2, 1, sizeof(cl_mem),
      (const void *)&top_data);
  error = clEnqueueNDRangeKernel(oclCommandQueue, this->ocl_float_kernel_2, 3, 
       NULL, (size_t *)&global, (size_t *)&local, 0, NULL, &event);
  //clWaitForEvents(1, &event);
  clFinish(oclCommandQueue);
}

template <>
void PipelineConvPoolLayer<double>::Call_ocl(const vector<Blob<double>*>& bottom,
    const vector<Blob<double>*>& top) { 
  Forward_cpu(bottom, top);
}

template <typename Dtype>
void PipelineConvPoolLayer<Dtype>::Forward_ocl(const vector <Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {

  Call_ocl(bottom, top);
  clReleaseKernel(this->ocl_float_kernel);
  clReleaseKernel(this->ocl_float_kernel_2);
  clReleaseProgram(this->ocl_layer_program);

}
#endif

#ifdef CPU_ONLY
STUB_GPU(PipelineConvPoolLayer);
#endif

INSTANTIATE_CLASS(PipelineConvPoolLayer);
REGISTER_LAYER_CLASS(PipelineConvPool);

}  // namespace caffe

