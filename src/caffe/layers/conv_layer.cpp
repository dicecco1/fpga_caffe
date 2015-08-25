#include <vector>

#include "caffe/filler.hpp"
#include "caffe/layer.hpp"
#include "caffe/util/im2col.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"

namespace caffe {

#ifdef USE_OCL
template <typename Dtype>
ConvolutionLayer<Dtype>::~ConvolutionLayer() {
}
#endif

template <typename Dtype>
void ConvolutionLayer<Dtype>::compute_output_shape() {
  this->height_out_ = (this->height_ + 2 * this->pad_h_ - this->kernel_h_)
      / this->stride_h_ + 1;
  this->width_out_ = (this->width_ + 2 * this->pad_w_ - this->kernel_w_)
      / this->stride_w_ + 1;
}

template <typename Dtype>
void ConvolutionLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  const Dtype* weight = this->blobs_[0]->cpu_data();
  for (int i = 0; i < bottom.size(); ++i) {
    const Dtype* bottom_data = bottom[i]->cpu_data();
    Dtype* top_data = top[i]->mutable_cpu_data();
    for (int n = 0; n < this->num_; ++n) {
      this->forward_cpu_gemm(bottom_data + bottom[i]->offset(n), weight,
          top_data + top[i]->offset(n));
      if (this->bias_term_) {
        const Dtype* bias = this->blobs_[1]->cpu_data();
        this->forward_cpu_bias(top_data + top[i]->offset(n), bias);
      }
    }
  }
}

template <typename Dtype>
void ConvolutionLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  const Dtype* weight = this->blobs_[0]->cpu_data();
  Dtype* weight_diff = this->blobs_[0]->mutable_cpu_diff();
  if (this->param_propagate_down_[0]) {
    caffe_set(this->blobs_[0]->count(), Dtype(0), weight_diff);
  }
  if (this->bias_term_ && this->param_propagate_down_[1]) {
    caffe_set(this->blobs_[1]->count(), Dtype(0),
        this->blobs_[1]->mutable_cpu_diff());
  }
  for (int i = 0; i < top.size(); ++i) {
    const Dtype* top_diff = top[i]->cpu_diff();
    const Dtype* bottom_data = bottom[i]->cpu_data();
    Dtype* bottom_diff = bottom[i]->mutable_cpu_diff();
    // Bias gradient, if necessary.
    if (this->bias_term_ && this->param_propagate_down_[1]) {
      Dtype* bias_diff = this->blobs_[1]->mutable_cpu_diff();
      for (int n = 0; n < this->num_; ++n) {
        this->backward_cpu_bias(bias_diff, top_diff + top[i]->offset(n));
      }
    }
    if (this->param_propagate_down_[0] || propagate_down[i]) {
      for (int n = 0; n < this->num_; ++n) {
        // gradient w.r.t. weight. Note that we will accumulate diffs.
        if (this->param_propagate_down_[0]) {
          this->weight_cpu_gemm(bottom_data + bottom[i]->offset(n),
              top_diff + top[i]->offset(n), weight_diff);
        }
        // gradient w.r.t. bottom data, if necessary.
        if (propagate_down[i]) {
          this->backward_cpu_gemm(top_diff + top[i]->offset(n), weight,
              bottom_diff + bottom[i]->offset(n));
        }
      }
    }
  }
}

#ifdef USE_OCL
template <>
void ConvolutionLayer<float>::Call_ocl(const vector<Blob<float>*>& bottom,
    const vector<Blob<float>*>& top) {  
  const vector<shared_ptr<Blob<float> > > weight = this->blobs_;
  const float* weight_data = weight[0]->ocl_data();
  cl_event event;
  cl_int error;

  size_t global[3] = {1, 1, 1};
  size_t local[3] = {1, 1, 1};

  for (int i = 0; i < bottom.size(); i++) {
    const float *bottom_data = bottom[i]->ocl_data();
    float *top_data = top[i]->mutable_ocl_data();
    clSetKernelArg(this->ocl_float_kernel, 0, sizeof(cl_mem),
        (const void *)&bottom_data);
    clSetKernelArg(this->ocl_float_kernel, 1, sizeof(cl_mem),
        (const void *)&weight_data);
    clSetKernelArg(this->ocl_float_kernel, 2, sizeof(cl_mem), 
        (const void *)&top_data);
    clEnqueueNDRangeKernel(oclCommandQueue, this->ocl_float_kernel, 3, NULL,
       (size_t *)&global, (size_t *)&local, 0, NULL, &event);
    clWaitForEvents(1, &event); 
  }
  for (int i = 0; i < bottom.size(); i++) {
    float *top_data = top[i]->mutable_cpu_data();
    //bias
    if (this->bias_term_) {
      const float* bias_data = weight[1]->cpu_data();
      for (int n = 0; n < top[i]->num(); n++) {
        for (int o = 0; o < top[i]->channels(); o++) {
          for (int y = 0; y < top[i]->height(); y++) {
            for (int x = 0; x < top[i]->width(); x++) {
              top_data[top[i]->offset(n, o, y, x)] += bias_data[o];
            }
          }
        }
      }
    }
  }  
}

template <>
void ConvolutionLayer<double>::Call_ocl(const vector<Blob<double>*>& bottom,
    const vector<Blob<double>*>& top) { 
  Forward_cpu(bottom, top);
}

template <typename Dtype>
void ConvolutionLayer<Dtype>::Forward_ocl(const vector<Blob<Dtype>*>& bottom,
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
  Call_ocl(bottom, top);
  clReleaseKernel(this->ocl_float_kernel);
  clReleaseProgram(this->ocl_layer_program);
}

#endif

#ifdef CPU_ONLY
STUB_GPU(ConvolutionLayer);
#endif

INSTANTIATE_CLASS(ConvolutionLayer);

}  // namespace caffe
