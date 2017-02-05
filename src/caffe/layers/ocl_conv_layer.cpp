#include <vector>

#include "caffe/layers/ocl_conv_layer.hpp"

namespace caffe {

#ifdef USE_OCL

template <typename Dtype>
void OCLConvolutionLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  BaseConvolutionLayer<Dtype>::LayerSetUp(bottom, top);
  ConvolutionParameter_SubEngine subengine =
      this->layer_param_.convolution_param().subengine();
  dim_ = bottom[0]->shape(2);
  if (subengine == ConvolutionParameter_SubEngine_WINOGRAD) {
    tile_ = (dim_ + 2 - 1) / 2;
    if (bottom[0]->shape(3) % 16 != 0) {
      offshape_ = (bottom[0]->shape(3) / 16 + 1) * 16;
      if (offshape_ * tile_ / 8 < 12) {
        offshape_ = offshape_ * 2;
      }
    } else {
      offshape_ = bottom[0]->shape(3);
    }

    inchannels_ = bottom[0]->shape(1) / this->group_;
    outchannels_ = this->num_output_ / this->group_;
    burstchannels_ = 128 * 128 / (tile_ * (offshape_ / 2));
    burstchannels_train_ = burstchannels_;
    if (burstchannels_ > inchannels_) {
      burstchannels_ = inchannels_;
    } else {
      int tchannel = burstchannels_;
      while (inchannels_ % tchannel != 0)
        tchannel--;
      burstchannels_ = tchannel;
    }
    if (burstchannels_train_ > outchannels_) {
      burstchannels_train_ = outchannels_;
    } else {
      int tchannel = burstchannels_train_;
      while (outchannels_ % tchannel != 0)
        tchannel--;
      burstchannels_train_ = tchannel;
    }

    rpo_ = inchannels_ / burstchannels_;
    rpo_train_ = outchannels_ / burstchannels_train_;
    tile_pad_ = offshape_ / 2;
    numgroups_ = this->group_;
  } else if (subengine == ConvolutionParameter_SubEngine_DIRECT) {
    dim_ = bottom[0]->shape(2);
    tile_ = dim_;

    if (bottom[0]->shape(3) % 16 != 0) {
      offshape_ = (bottom[0]->shape(3) / 16 + 1) * 16;
      if (offshape_ * tile_ / 8 < 12) {
        offshape_ = offshape_ * 2;
      }
    } else {
      offshape_ = bottom[0]->shape(3);
    }

    inchannels_ = bottom[0]->shape(1) / this->group_;
    outchannels_ = this->num_output_ / this->group_;
    burstchannels_ = 256 * 256 / (tile_ * offshape_);

    if (burstchannels_ > inchannels_) {
      burstchannels_ = inchannels_;
    } else {
      int tchannel = burstchannels_;
      while (inchannels_ % tchannel != 0)
        tchannel--;
      burstchannels_ = tchannel;
    }

    rpo_ = inchannels_ / burstchannels_;
    tile_pad_ = offshape_;

    numgroups_ = this->group_;
  }
}

template <typename Dtype>
void OCLConvolutionLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  BaseConvolutionLayer<Dtype>::Reshape(bottom, top);
  vector<int> shape(4);
  int ksize = (this->blobs_[0])->shape(3);
  shape[0] = (this->blobs_[0])->shape(0);
  shape[1] = (this->blobs_[0])->shape(1);
  if (ksize == 1) {
    shape[2] = 1;
    shape[3] = 16;
  } else if (ksize == 3) {
    shape[2] = 4;
    shape[3] = 4;
  } else if (ksize == 5) {
    shape[2] = 1;
    shape[3] = 32;
  }
  trans_weights.Reshape(shape);
  trans_weights_R.Reshape(shape);
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
void OCLConvolutionLayer<Dtype>::transform_weights(void) {
  vector<shared_ptr<Blob<Dtype> > > weight = this->blobs_;
  const Dtype* weight_data = weight[0]->cpu_data();
  Dtype* trans_data = trans_weights.mutable_cpu_data();
  int woff;
  int wtoff;
  int ksize = (this->blobs_[0])->shape(3);

  for (int i = 0; i < weight[0]->shape(0) * weight[0]->shape(1); ++i) {
    if (ksize == 1) {
      trans_data[i * 16] = weight_data[i];
    } else if (ksize == 3) {
      woff = i * 9;
      wtoff = i * 16;
      for (int j = 0; j < weight[0]->shape(2) * weight[0]->shape(3); ++j) {
        trans_data[wtoff + j] = weight_data[woff + j];
      }
    } else if (ksize == 5) {
      woff = i * ksize * ksize;
      wtoff = i * 32;
      for (int y = 0; y < 5; ++y) {
        for (int x = 0; x < 3; ++x) {
          trans_data[wtoff + y * 3 + x] = weight_data[woff + y * 5 + x];
          if (x < 2) {
            trans_data[wtoff + 16 + y * 3 + x] =
              weight_data[woff + y * 5 + 3 + x];
          } else {
            trans_data[wtoff + 16 + y * 3 + x] = 0;
          }
        }
      }
    }
  }
}

template <typename Dtype>
void OCLConvolutionLayer<Dtype>::transform_weights_rotated(void) {
  vector<shared_ptr<Blob<Dtype> > > weight = this->blobs_;
  const Dtype* weight_data = weight[0]->cpu_data();
  Dtype* trans_data_R = trans_weights_R.mutable_cpu_data();
  int woff;
  int wtoff;

  int ksize = weight[0]->shape(3);

  for (int n = 0; n < weight[0]->shape(0); ++n) {
    for (int m = 0; m < weight[0]->shape(1); ++m) {
      if (ksize == 1) {
        woff = n * weight[0]->shape(1) + m;
        wtoff = (m * weight[0]->shape(0) + n) * 16;
        trans_data_R[wtoff] = weight_data[woff];
      } else if (ksize == 3) {
        woff = (n * weight[0]->shape(1) + m) * 9;
        wtoff = (m * weight[0]->shape(0) + n) * 16;
        for (int j = 0; j < ksize; ++j) {
          for (int k = 0; k < ksize; ++k) {
            trans_data_R[wtoff + j * ksize + k] =
              weight_data[woff + j * ksize + (ksize - 1) - k];
          }
        }
        for (int j = 0; j < ksize / 2; ++j) {
          for (int k = 0; k < ksize; ++k) {
            float temp = trans_data_R[wtoff + j * ksize + k];
            trans_data_R[wtoff + j * ksize + k] =
              trans_data_R[wtoff + (ksize - 1 - j) * ksize + k];
            trans_data_R[wtoff + (ksize - 1 - j) * ksize + k] = temp;
          }
        }
      } else if (ksize == 5) {
        woff = (n * weight[0]->shape(1) + m) * 25;
        wtoff = (m * weight[0]->shape(0) + n) * 32;
        float rotated[32];
        for (int j = 0; j < ksize; ++j) {
          for (int k = 0; k < ksize; ++k) {
            rotated[wtoff + j * ksize + k] =
              weight_data[woff + j * ksize + (ksize - 1) - k];
          }
        }
        for (int j = 0; j < (ksize / 2); ++j) {
          for (int k = 0; k < ksize; ++k) {
            float temp = trans_data_R[wtoff + j * ksize + k];
            rotated[wtoff + j * ksize + k] =
              rotated[wtoff + (ksize - 1 - j) * ksize + k];
            rotated[wtoff + (ksize - 1 - j) * ksize + k] = temp;
          }
        }
        for (int y = 0; y < 5; ++y) {
          for (int x = 0; x < 3; ++x) {
            trans_data_R[wtoff + y * 3 + x] = rotated[y * 5 + x];
            if (x < 2) {
              trans_data_R[wtoff + 16 + y * 3 + x] =
                rotated[y * 5 + 3 + x];
            } else {
              trans_data_R[wtoff + 16 + y * 3 + x] = 0;
            }
          }
        }
      }
    }
  }
}

template <>
void OCLConvolutionLayer<float>::ocl_conv(
    const vector<Blob<float>*>& bottom, const vector<Blob<float>*>& top) {
  transform_weights();
  const float* weight_data = trans_weights.ocl_data();
  const float* bias_data = this->blobs_[1]->ocl_data();

  std::vector<cl_event> events(this->num_ * this->group_);

  int idx_off;
  int idx;
  int ksize = (this->blobs_[0])->shape(3);
  vector<int> outshape(4);
  vector<int> inshape(4);

  float *bottom_data;

  for (int i = 0; i < bottom.size(); i++) {
    if (top[i]->shape(3) % 16 != 0) {
      outshape[0] = top[i]->shape(0);
      outshape[1] = top[i]->shape(1);
      outshape[2] = top[i]->shape(2);
      outshape[3] = offshape_;
      top[i]->Reshape(outshape);
    }
    if (bottom[i]->shape(3) % 16 != 0) {
      inshape[0] = bottom[i]->shape(0);
      inshape[1] = bottom[i]->shape(1);
      inshape[2] = bottom[i]->shape(2);
      inshape[3] = offshape_;
      pad_input.Reshape(inshape);
      const float *input_data = bottom[i]->cpu_data();
      bottom_data = pad_input.mutable_cpu_data();
      for (int n = this->num_ - 1; n >= 0; n--) {
        for (int j = bottom[0]->shape(1) - 1; j >= 0; j--) {
          for (int y = dim_ - 1; y >= 0; y--) {
            for (int x = offshape_ - 1; x >= 0; x--) {
              idx_off = n * inchannels_ * numgroups_ * dim_ * offshape_ +
                (j * dim_ + y) * offshape_ + x;
              idx = n * inchannels_ * numgroups_ * dim_ * dim_ +
                (j * dim_ + y) * dim_ + x;
              if (x < dim_) {
                bottom_data[idx_off] = input_data[idx];
              } else {
                bottom_data[idx_off] = 0;
              }
            }
          }
        }
      }
    } else {
      pad_input.CopyFrom(*bottom[i], false, true);
    }
    bottom_data = pad_input.mutable_ocl_data();
    float *top_data = top[i]->mutable_ocl_data(0);
    clSetKernelArg(this->ocl_float_kernel, 0, sizeof(cl_mem),
      (const void *)&bottom_data);
    clSetKernelArg(this->ocl_float_kernel, 1, sizeof(cl_mem),
      (const void *)&weight_data);
    clSetKernelArg(this->ocl_float_kernel, 2, sizeof(cl_mem),
      (const void *)&bias_data);
    clSetKernelArg(this->ocl_float_kernel, 3, sizeof(cl_mem),
      (const void *)&top_data);
    clSetKernelArg(this->ocl_float_kernel, 5, sizeof(cl_int),
      (const void *)&inchannels_);
    clSetKernelArg(this->ocl_float_kernel, 6, sizeof(cl_int),
      (const void *)&outchannels_);
    clSetKernelArg(this->ocl_float_kernel, 7, sizeof(cl_int),
      (const void *)&burstchannels_);
    clSetKernelArg(this->ocl_float_kernel, 8, sizeof(cl_int),
      (const void *)&rpo_);
    clSetKernelArg(this->ocl_float_kernel, 9, sizeof(cl_int),
      (const void *)&dim_);
    clSetKernelArg(this->ocl_float_kernel, 10, sizeof(cl_int),
      (const void *)&dim_);
    clSetKernelArg(this->ocl_float_kernel, 11, sizeof(cl_int),
      (const void *)&tile_);
    clSetKernelArg(this->ocl_float_kernel, 12, sizeof(cl_int),
      (const void *)&tile_pad_);
    clSetKernelArg(this->ocl_float_kernel, 13, sizeof(cl_int),
      (const void *)&ksize);
    clSetKernelArg(this->ocl_float_kernel, 15, sizeof(cl_int),
      (const void *)&numgroups_);

    for (int n = 0; n < this->num_; ++n) {
      for (int g = 0; g < numgroups_; ++g) {
        clSetKernelArg(this->ocl_float_kernel, 4, sizeof(cl_int),
            (const void *)&g);
        clSetKernelArg(this->ocl_float_kernel, 14, sizeof(cl_int),
            (const void *)&n);
        clEnqueueTask(oclCommandQueue, this->ocl_float_kernel, 0,
            NULL, &(events[n * numgroups_ + g]));
      }
    }
    clWaitForEvents(this->num_ * numgroups_, &(events[0]));
    top_data = top[i]->mutable_cpu_data();
    if (top[i]->shape(3) != top[i]->shape(2)) {
      for (int n = 0; n < this->num_; ++n) {
        for (int j = 0; j < top[0]->shape(1); ++j) {
          for (int y = 0; y < dim_; ++y) {
            for (int x = 0; x < dim_; ++x) {
              idx_off = n * outchannels_ * numgroups_ * dim_ * offshape_ +
                (j * dim_ + y) * offshape_ + x;
              idx = n * outchannels_ * numgroups_ * dim_ * dim_ +
                (j * dim_ + y) * dim_ + x;
              top_data[idx] = top_data[idx_off];
            }
          }
        }
      }
      outshape[0] = top[i]->shape(0);
      outshape[1] = top[i]->shape(1);
      outshape[2] = top[i]->shape(2);
      outshape[3] = dim_;
      top[i]->Reshape(outshape);
    }
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
  transform_weights_rotated();
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

      clSetKernelArg(this->ocl_float_kernel, 0, sizeof(cl_mem),
        (const void *)&top_diff);
      clSetKernelArg(this->ocl_float_kernel, 1, sizeof(cl_mem),
        (const void *)&weight_data);
      clSetKernelArg(this->ocl_float_kernel, 2, sizeof(cl_mem),
        (const void *)&bias_data);
      clSetKernelArg(this->ocl_float_kernel, 3, sizeof(cl_mem),
        (const void *)&bottom_diff);
      clSetKernelArg(this->ocl_float_kernel, 5, sizeof(cl_int),
        (const void *)&outchannels_);
      clSetKernelArg(this->ocl_float_kernel, 6, sizeof(cl_int),
        (const void *)&inchannels_);
      clSetKernelArg(this->ocl_float_kernel, 7, sizeof(cl_int),
        (const void *)&burstchannels_train_);
      clSetKernelArg(this->ocl_float_kernel, 8, sizeof(cl_int),
        (const void *)&rpo_train_);
      clSetKernelArg(this->ocl_float_kernel, 9, sizeof(cl_int),
        (const void *)&dim_);
      clSetKernelArg(this->ocl_float_kernel, 10, sizeof(cl_int),
        (const void *)&dim_);
      clSetKernelArg(this->ocl_float_kernel, 11, sizeof(cl_int),
        (const void *)&tile_);
      clSetKernelArg(this->ocl_float_kernel, 12, sizeof(cl_int),
        (const void *)&tile_pad_);
      clSetKernelArg(this->ocl_float_kernel, 13, sizeof(cl_int),
        (const void *)&ksize);
      clSetKernelArg(this->ocl_float_kernel, 15, sizeof(cl_int),
        (const void *)&numgroups_);

      for (int n = 0; n < this->num_; ++n) {
        for (int g = 0; g < numgroups_; ++g) {
          clSetKernelArg(this->ocl_float_kernel, 4, sizeof(cl_int),
              (const void *)&g);
          clSetKernelArg(this->ocl_float_kernel, 14, sizeof(cl_int),
              (const void *)&n);

          clEnqueueTask(oclCommandQueue, this->ocl_float_kernel, 0,
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
  }
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
  ConvolutionParameter_SubEngine subengine =
     this->layer_param_.convolution_param().subengine();
  if (this->layer_param_.ocl_enable()) {
    if (subengine == ConvolutionParameter_SubEngine_WINOGRAD ||
        subengine == ConvolutionParameter_SubEngine_DIRECT)
      ocl_backward_conv(top, propagate_down, bottom);
    else
      LOG(FATAL) << "Layer " << this->layer_param_.name() <<
        " has unknown subengine.";
  } else {
    Backward_cpu(top, propagate_down, bottom);
  }
}


INSTANTIATE_CLASS(OCLConvolutionLayer);

#endif
}  // namespace caffe
