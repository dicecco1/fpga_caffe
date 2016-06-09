#include <vector>

#include "caffe/vision_layers.hpp"

namespace caffe {

template <typename Dtype>
void OCLConvolutionLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top){
  BaseConvolutionLayer<Dtype>::LayerSetUp(bottom, top);
  ConvolutionParameter_SubEngine subengine =
      this->layer_param_.convolution_param().subengine();
  vector<int> shape(4);
  shape[0] = (this->blobs_[0])->shape(0);
  shape[1] = (this->blobs_[0])->shape(1);
  shape[2] = 4;
  shape[3] = 4;
  if (subengine == ConvolutionParameter_SubEngine_WINOGRAD) {
    this->blobs_[0]->Reshape(shape);
    transform_weights();   
  }
  if (subengine == ConvolutionParameter_SubEngine_WINOGRAD) {
    vector<int> outshape(4);
    outshape[0] = shape[0];
    outshape[1] = shape[1] * this->group_;
    outshape[2] = bottom[0]->shape(2);
    if (bottom[0]->shape(3) % 8 != 0) {
      outshape[3] = (bottom[0]->shape(3) / 8 + 1) * 8;
    } else {
      outshape[3] = bottom[0]->shape(3);
    }
    outbuf.Reshape(outshape);
  }
}

template <typename Dtype>
void OCLConvolutionLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  BaseConvolutionLayer<Dtype>::Reshape(bottom, top);
}

template <typename Dtype>
void OCLConvolutionLayer<Dtype>::compute_output_shape() {
  const int* kernel_shape_data = this->kernel_shape_.cpu_data();
  const int* stride_data = this->stride_.cpu_data();
  const int* pad_data = this->pad_.cpu_data();
  this->output_shape_.clear();
  for (int i = 0; i < this->num_spatial_axes_; ++i) {
    // i + 1 to skip channel axis
    const int input_dim = this->input_shape(i + 1);
    const int output_dim = (input_dim + 2 * pad_data[i] - kernel_shape_data[i])
        / stride_data[i] + 1;
    this->output_shape_.push_back(output_dim);
  }
}

template <typename Dtype>
void OCLConvolutionLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  const Dtype* weight = this->blobs_[0]->cpu_data();
  for (int i = 0; i < bottom.size(); ++i) {
    const Dtype* bottom_data = bottom[i]->cpu_data();
    Dtype* top_data = top[i]->mutable_cpu_data();
    for (int n = 0; n < this->num_; ++n) {
      this->forward_cpu_gemm(bottom_data + n * this->bottom_dim_, weight,
          top_data + n * this->top_dim_);
      if (this->bias_term_) {
        const Dtype* bias = this->blobs_[1]->cpu_data();
        this->forward_cpu_bias(top_data + n * this->top_dim_, bias);
      }
    }
  }
}

template <typename Dtype>
void OCLConvolutionLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
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
        this->backward_cpu_bias(bias_diff, top_diff + n * this->top_dim_);
      }
    }
    if (this->param_propagate_down_[0] || propagate_down[i]) {
      for (int n = 0; n < this->num_; ++n) {
        // gradient w.r.t. weight. Note that we will accumulate diffs.
        if (this->param_propagate_down_[0]) {
          this->weight_cpu_gemm(bottom_data + n * this->bottom_dim_,
              top_diff + n * this->top_dim_, weight_diff);
        }
        // gradient w.r.t. bottom data, if necessary.
        if (propagate_down[i]) {
          this->backward_cpu_gemm(top_diff + n * this->top_dim_, weight,
              bottom_diff + n * this->bottom_dim_);
        }
      }
    }
  }
}

#ifdef USE_OCL
template <typename Dtype>
void OCLConvolutionLayer<Dtype>::transform_weights(void) {
  vector<shared_ptr<Blob<Dtype> > > weight = this->blobs_;
  Dtype* weight_data = weight[0]->mutable_cpu_data();
  int woff;
  int wtoff;
  
  Dtype x[16];
  for (int i = weight[0]->shape(0) * weight[0]->shape(1) - 1; i >= 0; --i) {
    woff = i * 9;
    wtoff = i * 16;
    x[0] = weight_data[woff + 1];
    x[1] = weight_data[woff + 0] + weight_data[woff + 2];
    x[2] = weight_data[woff + 6] + weight_data[woff + 8];
    x[3] = weight_data[woff + 3];
    x[4] = weight_data[woff + 5];
    x[5] = weight_data[woff + 1] + weight_data[woff + 7];
    x[6] = weight_data[woff + 0] + weight_data[woff + 6];
    x[7] = weight_data[woff + 3] + weight_data[woff + 5];
    x[8] = weight_data[woff + 7];
    x[9] = weight_data[woff + 4];
    x[10] = x[1] + x[2];
    x[11] = x[10] + x[5];
    x[12] = weight_data[woff + 2] + weight_data[woff + 8];
    x[13] = (x[7] + x[9]);
    x[14] = x[10] - x[5];
    x[15] = (x[7] - x[9]);

    weight_data[wtoff + 0] = weight_data[woff + 0];
    weight_data[wtoff + 3] = weight_data[woff + 2];
    weight_data[wtoff + 12] = weight_data[woff + 6];
    weight_data[wtoff + 15] = weight_data[woff + 8];
    weight_data[wtoff + 1] = 0.5 * (x[1] + x[0]);
    weight_data[wtoff + 2] = 0.5 * (x[1] - x[0]);
    weight_data[wtoff + 4] = 0.5 * (x[6] + x[3]);
    weight_data[wtoff + 5] = 0.25 * (x[11] + x[13]);
    weight_data[wtoff + 6] = 0.25 * (x[14] + x[15]);
    weight_data[wtoff + 7] = 0.5 * (x[12] + x[4]);
    weight_data[wtoff + 8] = 0.5 * (x[6] - x[3]);
    weight_data[wtoff + 9] = 0.25 * (x[11] - x[13]);
    weight_data[wtoff + 10] = 0.25 * (x[14] - x[15]);
    weight_data[wtoff + 11] = 0.5 * (x[12] - x[4]);
    weight_data[wtoff + 13] = 0.5 * (x[2] + x[8]);
    weight_data[wtoff + 14] = 0.5 * (x[2] - x[8]); 
  }
}

template <>
void OCLConvolutionLayer<float>::winograd_conv(
    const vector<Blob<float>*>& bottom, const vector<Blob<float>*>& top) {
  const vector<shared_ptr<Blob<float> > > weight = this->blobs_;
  const float* weight_data = weight[0]->ocl_data();
  const float* bias_data = weight[1]->ocl_data();
  cl_event event;

  int offshape = outbuf.shape(3);
  int ydim = top[0]->shape(2);
  int xdim = top[0]->shape(3);
  int idx_off;
  int idx;

  int inchannels = top[0]->shape(1) / this->group_;
  int outchannels = bottom[0]->shape(1) / this->group_;
  int burstchannels = 128 * 128 / (ydim * offshape);
  int ytile = (ydim + 2 - 1) / 2;
  int xtile = (xdim + 2 - 1) / 2;
  int rpo = inchannels / burstchannels;
  int ytile_pad = offshape / 2;
  int xtile_pad = offshape / 2;
  int rburst = ydim * burstchannels;

  float *tempdata;

  for (int i = 0; i < bottom.size(); i++) {
    const float *bottom_data = bottom[i]->ocl_data();
    float *top_data = top[i]->mutable_cpu_data();
    tempdata = outbuf.mutable_ocl_data();
    for (int g = 0; g < this->group_; ++g) {
      clSetKernelArg(this->ocl_float_kernel, 0, sizeof(cl_mem), &bottom_data);
      clSetKernelArg(this->ocl_float_kernel, 1, sizeof(cl_mem), &weight_data);
      clSetKernelArg(this->ocl_float_kernel, 2, sizeof(cl_mem), &bias_data);
      clSetKernelArg(this->ocl_float_kernel, 3, sizeof(cl_mem), &tempdata);
      clSetKernelArg(this->ocl_float_kernel, 4, sizeof(cl_int), &g);
      clSetKernelArg(this->ocl_float_kernel, 5, sizeof(cl_int), &inchannels);
      clSetKernelArg(this->ocl_float_kernel, 6, sizeof(cl_int), &outchannels);
      clSetKernelArg(this->ocl_float_kernel, 7, sizeof(cl_int), &burstchannels);
      clSetKernelArg(this->ocl_float_kernel, 8, sizeof(cl_int), &rpo);
      clSetKernelArg(this->ocl_float_kernel, 9, sizeof(cl_int), &ydim);
      clSetKernelArg(this->ocl_float_kernel, 10, sizeof(cl_int), &xdim);
      clSetKernelArg(this->ocl_float_kernel, 11, sizeof(cl_int), &ytile);
      clSetKernelArg(this->ocl_float_kernel, 12, sizeof(cl_int), &xtile);
      clSetKernelArg(this->ocl_float_kernel, 13, sizeof(cl_int), &ytile_pad);
      clSetKernelArg(this->ocl_float_kernel, 14, sizeof(cl_int), &xtile_pad);
      clSetKernelArg(this->ocl_float_kernel, 15, sizeof(cl_int), &rburst);
    
      clEnqueueTask(oclCommandQueue, this->ocl_float_kernel, 0,
                            NULL, &event);
      clWaitForEvents(1, &event);
    }

    const float *outdata = outbuf.cpu_data();
    for (int j = 0; j < top[i]->shape(0) * top[i]->shape(1); ++j) {
      for (int y = 0; y < ydim; ++y) {
        for (int x = 0; x < xdim; ++x) {
          idx_off = j * ydim * offshape + y * offshape + x;
          idx = j * ydim * xdim + y * xdim + x;
          top_data[idx] = outdata[idx_off];
        }
      }
    }
  }
}

template <>
void OCLConvolutionLayer<float>::matmul_conv(const vector<Blob<float>*>& bottom,
    const vector<Blob<float>*>& top) {
  const vector<shared_ptr<Blob<float> > > weight = this->blobs_;
  const float* weight_data = weight[0]->ocl_data();
  cl_event event;

  size_t global[3] = {top[0]->channels() / this->group_, 1, 1};
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
void OCLConvolutionLayer<float>::direct_conv(const vector<Blob<float>*>& bottom,
    const vector<Blob<float>*>& top) {
  const vector<shared_ptr<Blob<float> > > weight = this->blobs_;
  const float* weight_data = weight[0]->ocl_data();
  cl_event event;

  size_t global[3] = {top[0]->channels() / this->group_, 1, 1};
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
void OCLConvolutionLayer<double>::winograd_conv(
    const vector<Blob<double>*>& bottom, const vector<Blob<double>*>& top) {
  Forward_cpu(bottom, top);
}

template <>
void OCLConvolutionLayer<double>::matmul_conv(
    const vector<Blob<double>*>& bottom, const vector<Blob<double>*>& top) {
  Forward_cpu(bottom, top);
}

template <>
void OCLConvolutionLayer<double>::direct_conv(
    const vector<Blob<double>*>& bottom, const vector<Blob<double>*>& top) {
  Forward_cpu(bottom, top);  
}


template <>
void OCLConvolutionLayer<float>::Call_ocl(const vector<Blob<float>*>& bottom,
    const vector<Blob<float>*>& top) {  
  const vector<shared_ptr<Blob<float> > > weight = this->blobs_;
  const float* weight_data = weight[0]->ocl_data();
  cl_event event;

  size_t global[3] = {top[0]->channels() / this->group_, 1, 1};
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
void OCLConvolutionLayer<double>::Call_ocl(const vector<Blob<double>*>& bottom,
    const vector<Blob<double>*>& top) { 
  Forward_cpu(bottom, top);
}

template <typename Dtype>
void OCLConvolutionLayer<Dtype>::Forward_ocl(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  /*cl_int error;
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
  */
  ConvolutionParameter_SubEngine subengine = 
      this->layer_param_.convolution_param().subengine();
  if (this->layer_param_.ocl_enable()) {
    if (subengine == ConvolutionParameter_SubEngine_WINOGRAD) 
      winograd_conv(bottom, top);
    else if (subengine == ConvolutionParameter_SubEngine_MATMUL)
      matmul_conv(bottom, top);
    else if (subengine == ConvolutionParameter_SubEngine_DIRECT)
      direct_conv(bottom, top);
    else
      LOG(FATAL) << "Layer " << this->layer_param_.name() << " has unknown subengine.";
  } else
    Forward_cpu(bottom, top);
}
#endif

#ifdef CPU_ONLY
STUB_GPU(OCLConvolutionLayer);
#endif

INSTANTIATE_CLASS(OCLConvolutionLayer);

}  // namespace caffe
