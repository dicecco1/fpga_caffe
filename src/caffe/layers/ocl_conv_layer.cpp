#include <vector>

#include "caffe/layers/ocl_conv_layer.hpp"

namespace caffe {

#ifdef USE_OCL

template <typename Dtype>
void OCLConvolutionLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  BaseConvolutionLayer<Dtype>::LayerSetUp(bottom, top);
  kernel_params *params = &ocl_params_;
  params->ydim = bottom[0]->shape(2);
  params->xdim = bottom[0]->shape(3);

  int num_ = bottom[0]->shape(0);

  if (params->xdim % 16 != 0) {
    params->xtile_pad = (params->xdim / 16 + 1) * 8;
  } else {
    params->xtile_pad = params->xdim / 2;
  }
  params->inchannels = bottom[0]->shape(1) / this->group_;
  params->outchannels = this->num_output_ / this->group_;

  int burstchannels_ = 256 * 256 / (params->ydim * params->xtile_pad * 2);

  if (burstchannels_ > params->inchannels) {
    burstchannels_ = params->inchannels;
  } else {
    int tchannel = burstchannels_;
    while (params->inchannels % tchannel != 0)
      tchannel--;
    burstchannels_ = tchannel;
  }

  int numimages_ = 8 * 256 * 256 / (burstchannels_ * params->ydim *
      params->xtile_pad * 2);

  if (numimages_ > num_) {
    numimages_ = num_;
  } else {
    int tnumimages = numimages_;
    while (num_ % tnumimages != 0)
      tnumimages--;
    numimages_ = tnumimages;
  }

  params->numimages = numimages_;

  int burstydim_ = 512 * 16 / (params->xtile_pad * 2);

  if (burstydim_ > params->ydim) {
    burstydim_ = params->ydim;
  } else {
    int tburstydim = burstydim_;
    while (params->ydim % tburstydim != 0)
      tburstydim--;
    burstydim_ = tburstydim;
  }

  params->burstydim = burstydim_;

  params->ksize = (this->blobs_[0])->shape(3);
  params->rpofm = params->ydim / params->burstydim;
  params->burstchannels = burstchannels_;
  params->rpo = params->inchannels / burstchannels_;
  params->numgroups = this->group_;
  params->fc = 0;
  params->relu = 0;
  batch_ = num_ / params->numimages;
}

template <typename Dtype>
void OCLConvolutionLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  kernel_params *params = &ocl_params_;
  BaseConvolutionLayer<Dtype>::Reshape(bottom, top);
  vector<int> shape(4);
  int ksize = (this->blobs_[0])->shape(3);
  shape[0] = (this->blobs_[0])->shape(0);
  shape[1] = (this->blobs_[0])->shape(1);
  if (ksize == 1) {
    shape[2] = 1;
    shape[3] = 16;
  } else if (ksize == 3) {
    shape[2] = 1;
    shape[3] = 16;
  } else if (ksize == 5) {
    shape[2] = 1;
    shape[3] = 32;
  }
  weights_pad_h.Reshape(shape);
  bias_h.Reshape((this->blobs_[1])->shape());

  shape[0] = this->num_;
  shape[1] = bottom[0]->shape(1);
  shape[2] = bottom[0]->shape(2);
  shape[3] = params->xtile_pad * 2;

  input_pad_h.Reshape(shape);

  shape[1] = top[0]->shape(1);
  shape[2] = top[0]->shape(2);
  shape[3] = params->xtile_pad * 2;

  output_pad_h.Reshape(shape);
  relu_indices.Reshape(shape);
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

template <>
void OCLConvolutionLayer<float>::copyToHalf(const float *input, chalf *output,
    int size, int xdim, int xdim_pad) {
  for (int i = 0; i < size; ++i)
    for (int j = 0; j < xdim_pad; ++j)
      if (j < xdim)
        output[i * xdim_pad + j] = chalf(input[i * xdim + j]);
      else
        output[i * xdim_pad + j] = chalf(0);
}

template <>
void OCLConvolutionLayer<float>::copyToFloat(const chalf *input, float *output,
    int size, int xdim, int xdim_pad) {
  for (int i = 0; i < size; ++i)
    for (int j = 0; j < xdim_pad; ++j)
      if (j < xdim)
        output[i * xdim + j] = float(input[i * xdim_pad + j]);
}

template <>
void OCLConvolutionLayer<float>::copyToHalfWeights(const float *input,
    chalf *output, int size, int ksize, int ksize_pad) {
  for (int i = 0; i < size; ++i) {
    int out_idx = i * ksize_pad;
    int in_idx = i * ksize * ksize;
    if (ksize == 1) {
      output[out_idx] = chalf(input[in_idx]);
    } else if (ksize == 3) {
      for (int j = 0; j < ksize * ksize; ++j) {
        output[out_idx + j] = chalf(input[in_idx + j]);
      }
    } else if (ksize == 5) {
      for (int j = 0; j < 5; ++j) {
        for (int k = 0; k < 3; ++k) {
          output[out_idx + j * 3 + k] = chalf(input[in_idx + j * 5 + k]);
          if (k < 2)
            output[out_idx + 16 + j * 3 + k] =
              chalf(input[in_idx + j * 5 + 3 + k]);
          else
            output[out_idx + 16 + j * 3 + k] = 0;
        }
      }
    }
  }       
}

template <>
void OCLConvolutionLayer<float>::copyToFloatWeights(const chalf *input,
    float *output, int size, int ksize, int ksize_pad) {
  return;
}

template <>
void OCLConvolutionLayer<float>::ocl_conv(
    const vector<Blob<float>*>& bottom, const vector<Blob<float>*>& top) {
  kernel_params *params = &ocl_params_;
  int wsize = params->outchannels * params->numgroups * params->inchannels;
  int insize = params->inchannels * params->numimages * batch_ * params->ydim;
  int outsize = batch_ * params->numimages * params->outchannels *
      params->ydim;
  copyToHalfWeights(this->blobs_[0]->cpu_data(),
      weights_pad_h.mutable_cpu_data(), wsize, params->ksize,
      weights_pad_h.shape(3));
  copyToHalf(this->blobs_[1]->mutable_cpu_data(), bias_h.mutable_cpu_data(),
      params->outchannels, 1, 1);
  const chalf *weight_data = weights_pad_h.ocl_data();
  const chalf *bias_data = bias_h.ocl_data();

  std::vector<cl_event> events(batch_ * this->group_);

  params->backward = 0;
  int numgroups_ = params->numgroups;

  vector<int> shape(4);
  shape[0] = 1;
  shape[1] = 1;
  shape[2] = 1;
  shape[3] = sizeof(kernel_params) / sizeof(int);
  Blob<int> *param_vals = new Blob<int>(shape);

  int *conv_params = param_vals->mutable_cpu_data();

  for (int i = 0; i < shape[3]; ++i) {
    conv_params[i] = ((int *)params)[i];
    std::cout<<conv_params[i]<<std::endl;
  }

  conv_params = param_vals->mutable_ocl_data();

  chalf *bottom_data;
  chalf *top_data;
  char *relu_vals;
  for (int i = 0; i < bottom.size(); i++) {
    copyToHalf(bottom[i]->cpu_data(), input_pad_h.mutable_cpu_data(), insize,
        params->xdim, params->xtile_pad * 2);
    bottom_data = input_pad_h.mutable_ocl_data();
    top_data = output_pad_h.mutable_ocl_data(0);
    relu_vals = relu_indices.mutable_ocl_data(0);
    clSetKernelArg(this->ocl_kernel, 0, sizeof(cl_mem),
      (const void *)&bottom_data);
    clSetKernelArg(this->ocl_kernel, 1, sizeof(cl_mem),
      (const void *)&weight_data);
    clSetKernelArg(this->ocl_kernel, 2, sizeof(cl_mem),
      (const void *)&bias_data);
    clSetKernelArg(this->ocl_kernel, 3, sizeof(cl_mem),
      (const void *)&top_data);
    clSetKernelArg(this->ocl_kernel, 4, sizeof(cl_mem),
      (const void *)&relu_vals);
    clSetKernelArg(this->ocl_kernel, 5, sizeof(cl_mem),
      (const void *)&conv_params);

    for (int n = 0; n < batch_; ++n) {
      for (int g = 0; g < numgroups_; ++g) {
        clSetKernelArg(this->ocl_kernel, 6, sizeof(cl_int),
            (const void *)&g);
        clSetKernelArg(this->ocl_kernel, 7, sizeof(cl_int),
            (const void *)&n);
        clEnqueueTask(oclCommandQueue, this->ocl_kernel, 0,
            NULL, &(events[n * numgroups_ + g]));
      }
    }
    clWaitForEvents(batch_ * numgroups_, events.data());
    top_data = output_pad_h.mutable_cpu_data();
    copyToFloat(top_data, top[i]->mutable_cpu_data(), outsize, params->xdim,
        params->xtile_pad * 2);
  }
}

template <>
void OCLConvolutionLayer<double>::ocl_conv(
    const vector<Blob<double>*>& bottom, const vector<Blob<double>*>& top) {
  Forward_cpu(bottom, top);
}

template <>
void OCLConvolutionLayer<float>::ocl_backward_conv(
    const vector<Blob<float>*>& top, const vector<bool>& propagate_down,
    const vector<Blob<float>*>& bottom) {
/*  transform_weights_rotated();
  const float* weight_data = trans_weights_R.ocl_data();

  float* weight_diff = this->blobs_[0]->mutable_cpu_diff();
  if (this->param_propagate_down_[0]) {
    caffe_set(this->blobs_[0]->count(), static_cast<float>(0), weight_diff);
  }
  if (this->bias_term_ && this->param_propagate_down_[1]) {
    caffe_set(this->blobs_[1]->count(), static_cast<float>(0),
        this->blobs_[1]->mutable_cpu_diff());
  }
  for (int i = 0; i < top.size(); ++i) {
    const float* top_diff = top[i]->cpu_diff();
    // Bias gradient, if necessary.
    if (this->bias_term_ && this->param_propagate_down_[1]) {
      float* bias_diff = this->blobs_[1]->mutable_cpu_diff();
      for (int n = 0; n < this->num_; ++n) {
        this->backward_cpu_bias(bias_diff, top_diff + n * this->top_dim_);
      }
    }
  }
  std::vector<cl_event> events(this->num_ * this->group_);

  int idx_off;
  int idx;
  int ksize = (this->blobs_[0])->shape(3);
  vector<int> outshape(4);

  vector<int> bias_shape(bias_term_, outchannels_ * this->group_);
  Blob<float>* bias = new Blob<float>(bias_shape);
  caffe_set(outchannels_ * this->group_, static_cast<float>(0),
      bias->mutable_cpu_data());
  float *bias_data = bias->mutable_ocl_data();

  float *top_diff;

  for (int i = 0; i < bottom.size(); i++) {
    if (propagate_down[i]) {
      if (bottom[i]->shape(3) % 16 != 0) {
        outshape[0] = bottom[i]->shape(0);
        outshape[1] = bottom[i]->shape(1);
        outshape[2] = bottom[i]->shape(2);
        outshape[3] = offshape_;
        bottom[i]->Reshape(outshape);
      }
      if (top[i]->shape(3) % 16 != 0) {
        outshape[0] = top[i]->shape(0);
        outshape[1] = top[i]->shape(1);
        outshape[2] = top[i]->shape(2);
        outshape[3] = offshape_;
        pad_input.Reshape(outshape);

        const float *input_diff = top[i]->cpu_diff();
        top_diff = pad_input.mutable_cpu_diff();

        for (int n = 0; n < this->num_; ++n) {
          for (int j = 0; j < top[0]->shape(1); ++j) {
            for (int y = 0; y < dim_; ++y) {
              for (int x = 0; x < offshape_; ++x) {
                idx_off = n * outchannels_ * numgroups_ * dim_ * offshape_ +
                  (j * dim_ + y) * offshape_ + x;
                idx = n * outchannels_ * numgroups_ * dim_ * dim_ +
                  (j * dim_ + y) * dim_ + x;
                if (x < dim_) {
                  top_diff[idx_off] = input_diff[idx];
                } else {
                  top_diff[idx_off] = 0;
                }
              }
            }
          }
        }
      } else {
        pad_input.CopyFrom(*top[i], true, true);
      }

      top_diff = pad_input.mutable_ocl_diff();
      float *bottom_diff = bottom[i]->mutable_ocl_diff();

      clSetKernelArg(this->ocl_kernel, 0, sizeof(cl_mem),
        (const void *)&top_diff);
      clSetKernelArg(this->ocl_kernel, 1, sizeof(cl_mem),
        (const void *)&weight_data);
      clSetKernelArg(this->ocl_kernel, 2, sizeof(cl_mem),
        (const void *)&bias_data);
      clSetKernelArg(this->ocl_kernel, 3, sizeof(cl_mem),
        (const void *)&bottom_diff);
      clSetKernelArg(this->ocl_kernel, 5, sizeof(cl_int),
        (const void *)&outchannels_);
      clSetKernelArg(this->ocl_kernel, 6, sizeof(cl_int),
        (const void *)&inchannels_);
      clSetKernelArg(this->ocl_kernel, 7, sizeof(cl_int),
        (const void *)&burstchannels_train_);
      clSetKernelArg(this->ocl_kernel, 8, sizeof(cl_int),
        (const void *)&rpo_train_);
      clSetKernelArg(this->ocl_kernel, 9, sizeof(cl_int),
        (const void *)&dim_);
      clSetKernelArg(this->ocl_kernel, 10, sizeof(cl_int),
        (const void *)&dim_);
      clSetKernelArg(this->ocl_kernel, 11, sizeof(cl_int),
        (const void *)&tile_);
      clSetKernelArg(this->ocl_kernel, 12, sizeof(cl_int),
        (const void *)&tile_pad_);
      clSetKernelArg(this->ocl_kernel, 13, sizeof(cl_int),
        (const void *)&ksize);
      clSetKernelArg(this->ocl_kernel, 15, sizeof(cl_int),
        (const void *)&numgroups_);

      for (int n = 0; n < this->num_; ++n) {
        for (int g = 0; g < numgroups_; ++g) {
          clSetKernelArg(this->ocl_kernel, 4, sizeof(cl_int),
              (const void *)&g);
          clSetKernelArg(this->ocl_kernel, 14, sizeof(cl_int),
              (const void *)&n);

          clEnqueueTask(oclCommandQueue, this->ocl_kernel, 0,
                      NULL, &(events[n * numgroups_ + g]));
        }
      }

      clWaitForEvents(this->num_ * numgroups_, &(events[0]));
      bottom_diff = bottom[i]->mutable_cpu_diff();

      if (bottom[i]->shape(3) != bottom[i]->shape(2)) {
        for (int n = 0; n < this->num_; ++n) {
          for (int j = 0; j < bottom[0]->shape(1); ++j) {
            for (int y = 0; y < dim_; ++y) {
              for (int x = 0; x < dim_; ++x) {
                idx_off = n * inchannels_ * numgroups_ * dim_ * offshape_ +
                  (j * dim_ + y) * offshape_ + x;
                idx = n * inchannels_ * numgroups_ * dim_ * dim_ +
                  (j * dim_ + y) * dim_ + x;
                bottom_diff[idx] = bottom_diff[idx_off];
              }
            }
          }
        }
        outshape[0] = bottom[i]->shape(0);
        outshape[1] = bottom[i]->shape(1);
        outshape[2] = bottom[i]->shape(2);
        outshape[3] = dim_;
        bottom[i]->Reshape(outshape);
      }
    }
  }
  for (int i = 0; i < top.size(); ++i) {
    const float* top_diff = top[i]->cpu_diff();
    const float* bottom_data = bottom[i]->cpu_data();
    if (this->param_propagate_down_[0] || propagate_down[i]) {
      for (int n = 0; n < this->num_; ++n) {
        // gradient w.r.t. weight. Note that we will accumulate diffs.
        if (this->param_propagate_down_[0]) {
          this->weight_cpu_gemm(bottom_data + n * this->bottom_dim_,
              top_diff + n * this->top_dim_, weight_diff);
        }
      }
    }
  }*/
}

template <>
void OCLConvolutionLayer<double>::ocl_backward_conv(
    const vector<Blob<double>*>& top, const vector<bool>& propagate_down,
    const vector<Blob<double>*>& bottom) {
  Backward_cpu(top, propagate_down, bottom);
}

template <typename Dtype>
void OCLConvolutionLayer<Dtype>::Forward_ocl(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  ConvolutionParameter_SubEngine subengine =
    this->layer_param_.convolution_param().subengine();
  if (this->layer_param_.ocl_enable()) {
    if (subengine == ConvolutionParameter_SubEngine_WINOGRAD ||
        subengine == ConvolutionParameter_SubEngine_DIRECT)
      ocl_conv(bottom, top);
    else
      LOG(FATAL) << "Layer " << this->layer_param_.name() <<
        " has unknown subengine.";
  } else {
    Forward_cpu(bottom, top);
  }
}

template <typename Dtype>
void OCLConvolutionLayer<Dtype>::Backward_ocl(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
/*  ConvolutionParameter_SubEngine subengine =
     this->layer_param_.convolution_param().subengine();
  if (this->layer_param_.ocl_enable()) {
    if (subengine == ConvolutionParameter_SubEngine_WINOGRAD ||
        subengine == ConvolutionParameter_SubEngine_DIRECT)
      ocl_backward_conv(top, propagate_down, bottom);
    else
      LOG(FATAL) << "Layer " << this->layer_param_.name() <<
        " has unknown subengine.";
  } else {*/
    Backward_cpu(top, propagate_down, bottom);
  //}
}


INSTANTIATE_CLASS(OCLConvolutionLayer);

#endif
}  // namespace caffe
