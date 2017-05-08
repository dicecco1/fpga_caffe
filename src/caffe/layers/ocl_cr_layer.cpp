#include <vector>

#include "caffe/layers/ocl_cr_layer.hpp"

namespace caffe {

#ifdef USE_OCL

template <typename Dtype>
void OCLCRLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  BaseConvolutionLayer<Dtype>::LayerSetUp(bottom, top);
  CRParameter cr_param = this->layer_param_.cr_param(); 
  kernel_params *forward_params = &ocl_params_;
  forward_params->ydim = bottom[0]->shape(2);
  forward_params->xdim = bottom[0]->shape(3);
  forward_params->inchannels = bottom[0]->shape(1) / this->group_;
  forward_params->outchannels = this->num_output_ / this->group_;

  int num_ = bottom[0]->shape(0);

  if (forward_params->xdim % 16 != 0) {
    forward_params->xtile_pad = (forward_params->xdim / 16 + 1) * 8;
  } else {
    forward_params->xtile_pad = forward_params->xdim / 2;
  }

  int burstchannels_ = 256 * 256 / (forward_params->ydim *
      forward_params->xtile_pad * 2);

  if (burstchannels_ > forward_params->inchannels) {
    burstchannels_ = forward_params->inchannels;
  } else {
    int tchannel = burstchannels_;
    while (forward_params->inchannels % tchannel != 0)
      tchannel--;
    burstchannels_ = tchannel;
  }

  int numimages_ = 8 * 256 * 256 / (burstchannels_ * forward_params->ydim *
      forward_params->xtile_pad * 2);

  if (numimages_ > num_) {
    numimages_ = num_;
  } else {
    int tnumimages = numimages_;
    while (num_ % tnumimages != 0)
      tnumimages--;
    numimages_ = tnumimages;
  }
  int burstydim_ = 512 * 16 / (forward_params->xtile_pad * 2);

  if (burstydim_ > forward_params->ydim) {
    burstydim_ = forward_params->ydim;
  } else {
    int tburstydim = burstydim_;
    while (forward_params->ydim % tburstydim != 0)
      tburstydim--;
    burstydim_ = tburstydim;
  }

  forward_params->numimages = numimages_;
  forward_params->burstydim = burstydim_;
  forward_params->ksize = (this->blobs_[0])->shape(3);
  forward_params->rpofm = forward_params->ydim / forward_params->burstydim;
  forward_params->burstchannels = burstchannels_;
  forward_params->rpo = forward_params->inchannels / burstchannels_;
  forward_params->numgroups = this->group_;
  forward_params->fc = 0;
  forward_params->relu = cr_param.relu();
  batch_ = num_ / forward_params->numimages;

  // Backward params
  kernel_params *backward_params = &ocl_params_bi_;
  backward_params->ydim = bottom[0]->shape(2);
  backward_params->xdim = bottom[0]->shape(3);
  backward_params->inchannels = this->num_output_ / this->group_;
  backward_params->outchannels = bottom[0]->shape(1) / this->group_;

  if (backward_params->xdim % 16 != 0) {
    backward_params->xtile_pad = (backward_params->xdim / 16 + 1) * 8;
  } else {
    backward_params->xtile_pad = backward_params->xdim / 2;
  }

  burstchannels_ = 256 * 256 / (backward_params->ydim *
      backward_params->xtile_pad * 2);

  if (burstchannels_ > backward_params->inchannels) {
    burstchannels_ = backward_params->inchannels;
  } else {
    int tchannel = burstchannels_;
    while (backward_params->inchannels % tchannel != 0)
      tchannel--;
    burstchannels_ = tchannel;
  }

  numimages_ = 8 * 256 * 256 / (burstchannels_ * forward_params->ydim *
      forward_params->xtile_pad * 2);

  if (numimages_ > num_) {
    numimages_ = num_;
  } else {
    int tnumimages = numimages_;
    while (num_ % tnumimages != 0)
      tnumimages--;
    numimages_ = tnumimages;
  }
  
  burstydim_ = 512 * 16 / (backward_params->xtile_pad * 2);

  if (burstydim_ > backward_params->ydim) {
    burstydim_ = backward_params->ydim;
  } else {
    int tburstydim = burstydim_;
    while (backward_params->ydim % tburstydim != 0)
      tburstydim--;
    burstydim_ = tburstydim;
  }

  backward_params->numimages = numimages_;
  backward_params->burstydim = burstydim_;
  backward_params->ksize = (this->blobs_[0])->shape(3);
  backward_params->rpofm = backward_params->ydim / backward_params->burstydim;
  backward_params->burstchannels = burstchannels_;
  backward_params->rpo = backward_params->inchannels / burstchannels_;
  backward_params->numgroups = this->group_;
  backward_params->fc = 0;
  backward_params->relu = cr_param.relu();
  batch_bi_ = num_ / backward_params->numimages;

  // Set bias update parameters
  kernel_params *bias_params = &ocl_params_bb_;

  bias_params->ydim = backward_params->ydim;
  bias_params->xdim = backward_params->xdim;
  bias_params->xtile_pad = backward_params->xtile_pad;
  bias_params->inchannels = backward_params->inchannels;
  bias_params->outchannels = 1;
  bias_params->burstchannels = burstchannels_;
  bias_params->numimages = numimages_;
  bias_params->burstydim = burstydim_;
  bias_params->ksize = 1;
  bias_params->rpofm = bias_params->ydim / bias_params->burstydim;
  bias_params->rpo = bias_params->inchannels / burstchannels_;
  bias_params->numgroups = this->group_;
  bias_params->fc = 0;
  bias_params->relu = cr_param.relu();
  batch_bb_ = batch_bi_;
}

template <typename Dtype>
void OCLCRLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
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
    shape[2] = 1;
    shape[3] = 16;
  } else if (ksize == 5) {
    shape[2] = 1;
    shape[3] = 32;
  }
  weights_pad_h.Reshape(shape);
  
  shape[0] = (this->blobs_[0])->shape(1);
  shape[1] = (this->blobs_[0])->shape(0);
  weights_pad_h_r.Reshape(shape);

  bias_h.Reshape((this->blobs_[1])->shape());
  shape[0] = top[0]->shape(0);
  shape[1] = top[0]->shape(1);
  shape[2] = top[0]->shape(2);
  shape[3] = top[0]->shape(3);

  relu_indices.Reshape(shape);
}

template <typename Dtype>
void OCLCRLayer<Dtype>::compute_output_shape() {
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
void OCLCRLayer<Dtype>::copyToHalf(const Dtype *input, chalf *output,
    int size, int xdim, int xdim_pad) {
  for (int i = 0; i < size; ++i)
    for (int j = 0; j < xdim_pad; ++j)
      if (j < xdim)
        output[i * xdim_pad + j] = chalf((float)input[i * xdim + j]);
      else
        output[i * xdim_pad + j] = chalf(0);
}


template <typename Dtype>
void OCLCRLayer<Dtype>::copyToHalfWeights(const Dtype *input,
    chalf *output, int size, int ksize, int ksize_pad) {
  for (int i = 0; i < size; ++i) {
    int out_idx = i * ksize_pad;
    int in_idx = i * ksize * ksize;
    if (ksize == 1) {
      output[out_idx] = chalf((float)input[in_idx]);
    } else if (ksize == 3) {
      for (int j = 0; j < ksize * ksize; ++j) {
        output[out_idx + j] = chalf((float)input[in_idx + j]);
      }
    } else if (ksize == 5) {
      for (int j = 0; j < 5; ++j) {
        for (int k = 0; k < 3; ++k) {
          output[out_idx + j * 3 + k] =
            chalf((float)input[in_idx + j * 5 + k]);
          if (k < 2) {
            output[out_idx + 16 + j * 3 + k] =
              chalf((float)input[in_idx + j * 5 + 3 + k]);
          }
          else
            output[out_idx + 16 + j * 3 + k] = 0;
        }
      }
    }
  }       
}

template <typename Dtype>
void OCLCRLayer<Dtype>::RotateWeightsHalf(const Dtype *input, chalf *output,
    vector<int> shape, int ksize_pad) {
  int ksize = shape[3];
  for (int i = 0; i < shape[0]; ++i) {
    for (int j = 0; j < shape[1]; ++j) {
      int out_idx = (j * shape[0] + i) * ksize_pad;
      int in_idx = (i * shape[1] + j) * ksize * ksize;
      if (ksize == 1) {
        output[out_idx] = chalf((float)input[in_idx]);
      } else if (ksize == 3) {
        for (int k = 0; k < ksize * ksize; ++k) {
          output[out_idx + k] =
            chalf((float)input[in_idx + ksize * ksize - 1 - k]);
        }
      } else if (ksize == 5) {
        for (int k = 0; k < 5; ++k) {
          for (int l = 0; l < 3; ++l) {
            output[out_idx + k * 3 + l] =
              chalf((float)input[in_idx + 24 - (k * 5 + l)]);
            if (l < 2) {
              output[out_idx + 16 + k * 3 + l] =
                chalf((float)input[in_idx + 21 - (k * 5 + l)]);
            } else {
              output[out_idx + 16 + k * 3 + l] = 0;
            }
          }
        }
      }
    }
  }
}

template <typename Dtype>
void OCLCRLayer<Dtype>::copyToFloatWeights(chalf *input,
    Dtype *output, const vector<int> shape, int ksize_pad) {
  int ksize = shape[3];
  for (int i = 0; i < shape[0]; ++i) {
    for (int j = 0; j < shape[1]; ++j) {
      int in_idx = (j * shape[0] + i) * ksize_pad;
      int out_idx = (i * shape[1] + j) * ksize * ksize;
      if (ksize == 1) {
        output[out_idx] = (Dtype)(float(input[in_idx + 1]));
      } else if (ksize == 3) {
        for (int k = 0; k < ksize * ksize; ++k) {
          output[out_idx + k] =
            (Dtype)(float(input[in_idx + 8 - k]));
        }
      } else if (ksize == 5) {
        for (int k = 0; k < 5; ++k) {
          for (int l = 0; l < 3; ++l) {
            output[out_idx + 24 - (k * 5 + l)] =
              (Dtype)(float(input[in_idx + k * 3 + l]));
            if (l < 2) {
              output[out_idx + 21 - (k * 5 + l)] =
                (Dtype)(float(input[in_idx + 16 + k * 3 + l]));
            }
          }
        }
      }
    }
  }
}

template <typename Dtype>
void OCLCRLayer<Dtype>::backward_bias(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  kernel_params *params = &ocl_params_bb_;

  vector<int> shape(4);
  shape[0] = bottom[0]->shape(0);
  shape[1] = 1;
  shape[2] = bottom[0]->shape(2);
  shape[3] = bottom[0]->shape(3);

  Blob<chalf>* weights = new Blob<chalf>(shape);

  for (int i = 0; i < weights->count(); ++i)
    (weights->mutable_cpu_data())[i] = chalf((float)1.0);

  const chalf *weights_data = weights->ocl_data();

  shape[0] = 1;
  shape[1] = 1;
  shape[2] = this->num_output_;
  shape[3] = 16;

  Blob<chalf>* bias = new Blob<chalf>(shape);

  for (int i = 0; i < bias->count(); ++i)
    (bias->mutable_cpu_data())[i] = chalf(0);

  chalf *bias_diff = bias->mutable_ocl_data();

  params->backward = 1;
  int numgroups_ = params->numgroups;

  shape[0] = 1;
  shape[1] = 1;
  shape[2] = 1;
  shape[3] = sizeof(kernel_params) / sizeof(int);
  Blob<int> *param_vals = new Blob<int>(shape);

  int *conv_params = param_vals->mutable_cpu_data();

  for (int i = 0; i < shape[3]; ++i) {
    conv_params[i] = ((int *)params)[i];
  }
  
  const int* cr_params_b = param_vals->ocl_data();

  size_t outsize = sizeof(chalf) * top[0]->count();
  std::vector<cl_event> events;

  int events_size = batch_bi_ * numgroups_;

  const chalf *top_diff;
  const char *relu_vals;
  for (int i = 0; i < bottom.size(); i++) {
    events.resize(events_size, 0);
    top_diff = reinterpret_cast<const chalf *>(top[i]->ocl_diff(outsize));
    relu_vals = relu_indices.ocl_data();
    clSetKernelArg(this->ocl_kernel, 0, sizeof(cl_mem),
      (const void *)&top_diff);
    clSetKernelArg(this->ocl_kernel, 1, sizeof(cl_mem),
      (const void *)&weights_data);
    clSetKernelArg(this->ocl_kernel, 2, sizeof(cl_mem),
      (const void *)&bias_diff);
    clSetKernelArg(this->ocl_kernel, 3, sizeof(cl_mem),
      (const void *)&bias_diff);
    clSetKernelArg(this->ocl_kernel, 4, sizeof(cl_mem),
      (const void *)&relu_vals);
    clSetKernelArg(this->ocl_kernel, 5, sizeof(cl_mem),
      (const void *)&cr_params_b);

    for (int n = 0; n < batch_bi_; ++n) {
      for (int g = 0; g < numgroups_; ++g) {
        clSetKernelArg(this->ocl_kernel, 6, sizeof(cl_int),
            (const void *)&g);
        clSetKernelArg(this->ocl_kernel, 7, sizeof(cl_int),
            (const void *)&n);
        clEnqueueTask(oclCommandQueue, this->ocl_kernel, 0,
            NULL, &(events[n * numgroups_ + g]));
      }
    }
    clWaitForEvents(events.size(), events.data());
  }
  bias_diff = bias->mutable_cpu_data();
  Dtype *bias_diff_out = this->blobs_[1]->mutable_cpu_diff();
  for (int i = 0; i < this->blobs_[1]->count(); ++i) {
    bias_diff_out[i] = (Dtype)float(bias_diff[i * 16]);
  }
}

template <typename Dtype>
void OCLCRLayer<Dtype>::backward_data(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  kernel_params *params = &ocl_params_bi_;
  RotateWeightsHalf(this->blobs_[0]->cpu_data(),
      weights_pad_h_r.mutable_cpu_data(), this->blobs_[0]->shape(),
      weights_pad_h_r.shape(3));

  const chalf *weight_data_r = weights_pad_h_r.ocl_data();
  vector<int> shape(4);
  shape[0] = 1;
  shape[1] = 1;
  shape[2] = 1;
  shape[3] = bottom[0]->shape(1);

  Blob<chalf>* bias = new Blob<chalf>(shape);

  for (int i = 0; i < bias->count(); ++i)
    (bias->mutable_cpu_data())[i] = chalf(0);

  const chalf *bias_data = bias->ocl_data();

  params->backward = 2;
  int numgroups_ = params->numgroups;

  shape[0] = 1;
  shape[1] = 1;
  shape[2] = 1;
  shape[3] = sizeof(kernel_params) / sizeof(int);
  Blob<int> *param_vals = new Blob<int>(shape);

  int *conv_params = param_vals->mutable_cpu_data();

  for (int i = 0; i < shape[3]; ++i) {
    conv_params[i] = ((int *)params)[i];
  }
  
  const int* cr_params_b = param_vals->ocl_data();

  size_t insize = sizeof(chalf) * bottom[0]->count();
  size_t outsize = sizeof(chalf) * top[0]->count();
  std::vector<cl_event> events;

  int events_size = batch_bi_ * numgroups_;

  const chalf *top_diff;
  const char *relu_vals;
  chalf *bottom_diff;
  for (int i = 0; i < bottom.size(); i++) {
    events.resize(events_size, 0);
    bottom_diff =
      reinterpret_cast<chalf *>(bottom[i]->mutable_ocl_diff(0));
    top_diff = reinterpret_cast<const chalf *>(top[i]->ocl_diff(outsize));
    relu_vals = relu_indices.ocl_data();
    clSetKernelArg(this->ocl_kernel, 0, sizeof(cl_mem),
      (const void *)&top_diff);
    clSetKernelArg(this->ocl_kernel, 1, sizeof(cl_mem),
      (const void *)&weight_data_r);
    clSetKernelArg(this->ocl_kernel, 2, sizeof(cl_mem),
      (const void *)&bias_data);
    clSetKernelArg(this->ocl_kernel, 3, sizeof(cl_mem),
      (const void *)&bottom_diff);
    clSetKernelArg(this->ocl_kernel, 4, sizeof(cl_mem),
      (const void *)&relu_vals);
    clSetKernelArg(this->ocl_kernel, 5, sizeof(cl_mem),
      (const void *)&cr_params_b);

    for (int n = 0; n < batch_bi_; ++n) {
      for (int g = 0; g < numgroups_; ++g) {
        clSetKernelArg(this->ocl_kernel, 6, sizeof(cl_int),
            (const void *)&g);
        clSetKernelArg(this->ocl_kernel, 7, sizeof(cl_int),
            (const void *)&n);
        clEnqueueTask(oclCommandQueue, this->ocl_kernel, 0,
            NULL, &(events[n * numgroups_ + g]));
      }
    }
    clWaitForEvents(events.size(), events.data());
  }
}

template <typename Dtype>
void OCLCRLayer<Dtype>::backward_weights(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  kernel_params *params = &ocl_params_bi_;

  Dtype* weight_diff_dtype = this->blobs_[0]->mutable_cpu_diff();
 
  chalf *weight_diff = weights_pad_h.mutable_cpu_diff();

  for (int i = 0; i < weights_pad_h.count(); ++i)
    weight_diff[i] = chalf(0);

  weight_diff = weights_pad_h.mutable_ocl_diff();

  Blob<chalf>* bias = new Blob<chalf>(this->blobs_[1]->shape());

  for (int i = 0; i < bias->count(); ++i)
    (bias->mutable_cpu_data())[i] = chalf(0);

  const chalf *bias_data = bias->ocl_data();

  params->backward = 1;
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
  }
  
  const int* cr_params_b = param_vals->ocl_data();

  size_t insize = sizeof(chalf) * bottom[0]->count();
  size_t outsize = sizeof(chalf) * top[0]->count();
  std::vector<cl_event> events;

  int events_size = batch_bi_ * numgroups_;

  const chalf *top_diff;
  const char *relu_vals;
  const chalf *bottom_data;
  for (int i = 0; i < bottom.size(); i++) {
    events.resize(events_size, 0);
    bottom_data =
      reinterpret_cast<const chalf *>(bottom[i]->ocl_data(insize));
    top_diff = reinterpret_cast<const chalf *>(top[i]->ocl_diff(outsize));
    relu_vals = relu_indices.ocl_data();
    clSetKernelArg(this->ocl_kernel, 0, sizeof(cl_mem),
      (const void *)&top_diff);
    clSetKernelArg(this->ocl_kernel, 1, sizeof(cl_mem),
      (const void *)&bottom_data);
    clSetKernelArg(this->ocl_kernel, 2, sizeof(cl_mem),
      (const void *)&bias_data);
    clSetKernelArg(this->ocl_kernel, 3, sizeof(cl_mem),
      (const void *)&weight_diff);
    clSetKernelArg(this->ocl_kernel, 4, sizeof(cl_mem),
      (const void *)&relu_vals);
    clSetKernelArg(this->ocl_kernel, 5, sizeof(cl_mem),
      (const void *)&cr_params_b);

    for (int n = 0; n < batch_bi_; ++n) {
      for (int g = 0; g < numgroups_; ++g) {
        clSetKernelArg(this->ocl_kernel, 6, sizeof(cl_int),
            (const void *)&g);
        clSetKernelArg(this->ocl_kernel, 7, sizeof(cl_int),
            (const void *)&n);
        clEnqueueTask(oclCommandQueue, this->ocl_kernel, 0,
            NULL, &(events[n * numgroups_ + g]));
      }
    }
    clWaitForEvents(events.size(), events.data());
  }
  weight_diff = weights_pad_h.mutable_cpu_diff();
  copyToFloatWeights(weight_diff, weight_diff_dtype,
      this->blobs_[0]->shape(), weights_pad_h.shape(3));

  delete bias;
  delete param_vals;
}


template <typename Dtype>
void OCLCRLayer<Dtype>::Forward_ocl(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  kernel_params *params = &ocl_params_;
  int wsize = params->outchannels * params->numgroups * params->inchannels;
  copyToHalfWeights(this->blobs_[0]->cpu_data(),
      weights_pad_h.mutable_cpu_data(), wsize, params->ksize,
      weights_pad_h.shape(3));
  copyToHalf(this->blobs_[1]->mutable_cpu_data(), bias_h.mutable_cpu_data(),
      params->outchannels, 1, 1);
  const chalf *weight_data = weights_pad_h.ocl_data();
  const chalf *bias_data = bias_h.ocl_data();

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
  }
  
  const int* cr_params = param_vals->ocl_data();

  size_t insize = sizeof(chalf) * bottom[0]->count();
  size_t outsize = sizeof(chalf) * top[0]->count();
  std::vector<cl_event> events;
  int events_size = batch_ * numgroups_;

  chalf *top_data;
  char *relu_vals;
  for (int i = 0; i < bottom.size(); i++) {
    events.resize(events_size, 0);
    const chalf* bottom_data =
      reinterpret_cast<const chalf *>(bottom[i]->ocl_data(insize));
    top_data = reinterpret_cast<chalf *>(top[i]->mutable_ocl_data(0));
    relu_vals = relu_indices.mutable_ocl_data();
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
      (const void *)&cr_params);

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
    clWaitForEvents(events.size(), events.data());
  }
}

template <typename Dtype>
void OCLCRLayer<Dtype>::Backward_ocl(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  if (this->bias_term_ && this->param_propagate_down_[1])
    backward_bias(top, propagate_down, bottom);
  
  if (this->param_propagate_down_[0])
    backward_weights(top, propagate_down, bottom);

  if (propagate_down[0])
    backward_data(top, propagate_down, bottom);
}


INSTANTIATE_CLASS(OCLCRLayer);
REGISTER_LAYER_CLASS(OCLCR);
#endif
}  // namespace caffe
