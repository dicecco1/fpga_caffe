#include <vector>

#include "caffe/filler.hpp"
#include "caffe/layers/ocl_cr_hwcn_layer.hpp"

namespace caffe {

#ifdef USE_OCL

template <typename Dtype>
void OCLCRHWCNLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  // Configure the kernel size, padding, stride, and inputs.
  ConvolutionParameter conv_param = this->layer_param_.convolution_param();
  const int num_axes = bottom[0]->num_axes();
  this->num_spatial_axes_ = num_axes - 2;
  CHECK_GE(this->num_spatial_axes_, 0);
  vector<int> spatial_dim_blob_shape(1, std::max(this->num_spatial_axes_, 1));
  // Setup filter kernel dimensions (kernel_shape_).
  this->kernel_shape_.Reshape(spatial_dim_blob_shape);
  int* kernel_shape_data = this->kernel_shape_.mutable_cpu_data();
  if (conv_param.has_kernel_h() || conv_param.has_kernel_w()) {
    CHECK_EQ(this->num_spatial_axes_, 2)
        << "kernel_h & kernel_w can only be used for 2D convolution.";
    CHECK_EQ(0, conv_param.kernel_size_size())
        << "Either kernel_size or kernel_h/w should be specified; not both.";
    kernel_shape_data[0] = conv_param.kernel_h();
    kernel_shape_data[1] = conv_param.kernel_w();
  } else {
    const int num_kernel_dims = conv_param.kernel_size_size();
    CHECK(num_kernel_dims == 1 || num_kernel_dims == this->num_spatial_axes_)
        << "kernel_size must be specified once, or once per spatial dimension "
        << "(kernel_size specified " << num_kernel_dims << " times; "
        << this->num_spatial_axes_ << " spatial dims).";
      for (int i = 0; i < this->num_spatial_axes_; ++i) {
        kernel_shape_data[i] =
            conv_param.kernel_size((num_kernel_dims == 1) ? 0 : i);
      }
  }
  for (int i = 0; i < this->num_spatial_axes_; ++i) {
    CHECK_GT(kernel_shape_data[i], 0) << "Filter dimensions must be nonzero.";
  }
  // Setup stride dimensions (stride_).
  this->stride_.Reshape(spatial_dim_blob_shape);
  int* stride_data = this->stride_.mutable_cpu_data();
  if (conv_param.has_stride_h() || conv_param.has_stride_w()) {
    CHECK_EQ(this->num_spatial_axes_, 2)
        << "stride_h & stride_w can only be used for 2D convolution.";
    CHECK_EQ(0, conv_param.stride_size())
        << "Either stride or stride_h/w should be specified; not both.";
    stride_data[0] = conv_param.stride_h();
    stride_data[1] = conv_param.stride_w();
  } else {
    const int num_stride_dims = conv_param.stride_size();
    CHECK(num_stride_dims == 0 || num_stride_dims == 1 ||
          num_stride_dims == this->num_spatial_axes_)
        << "stride must be specified once, or once per spatial dimension "
        << "(stride specified " << num_stride_dims << " times; "
        << this->num_spatial_axes_ << " spatial dims).";
    const int kDefaultStride = 1;
    for (int i = 0; i < this->num_spatial_axes_; ++i) {
      stride_data[i] = (num_stride_dims == 0) ? kDefaultStride :
          conv_param.stride((num_stride_dims == 1) ? 0 : i);
      CHECK_GT(stride_data[i], 0) << "Stride dimensions must be nonzero.";
    }
  }
  // Setup pad dimensions (pad_).
  this->pad_.Reshape(spatial_dim_blob_shape);
  int* pad_data = this->pad_.mutable_cpu_data();
  if (conv_param.has_pad_h() || conv_param.has_pad_w()) {
    CHECK_EQ(this->num_spatial_axes_, 2)
        << "pad_h & pad_w can only be used for 2D convolution.";
    CHECK_EQ(0, conv_param.pad_size())
        << "Either pad or pad_h/w should be specified; not both.";
    pad_data[0] = conv_param.pad_h();
    pad_data[1] = conv_param.pad_w();
  } else {
    const int num_pad_dims = conv_param.pad_size();
    CHECK(num_pad_dims == 0 || num_pad_dims == 1 ||
          num_pad_dims == this->num_spatial_axes_)
        << "pad must be specified once, or once per spatial dimension "
        << "(pad specified " << num_pad_dims << " times; "
        << this->num_spatial_axes_ << " spatial dims).";
    const int kDefaultPad = 0;
    for (int i = 0; i < this->num_spatial_axes_; ++i) {
      pad_data[i] = (num_pad_dims == 0) ? kDefaultPad :
          conv_param.pad((num_pad_dims == 1) ? 0 : i);
    }
  }
  // Configure output channels and groups.
  this->channels_ = bottom[0]->shape(2);
  this->num_output_ = this->layer_param_.convolution_param().num_output();
  CHECK_GT(this->num_output_, 0);
  this->group_ = this->layer_param_.convolution_param().group();
  CHECK_EQ(this->channels_ % this->group_, 0);
  CHECK_EQ(this->num_output_ % this->group_, 0)
      << "Number of output should be multiples of group.";
  if (reverse_dimensions()) {
    conv_out_channels_ = this->channels_;
    conv_in_channels_ = this->num_output_;
  } else {
    conv_out_channels_ = this->num_output_;
    conv_in_channels_ = this->channels_;
  }
  // Handle the parameters: weights and biases.
  // - blobs_[0] holds the filter weights
  // - blobs_[1] holds the biases (optional)
  vector<int> weight_shape(2);
  weight_shape[0] = conv_out_channels_;
  weight_shape[1] = conv_in_channels_ / this->group_;
  for (int i = 0; i < this->num_spatial_axes_; ++i) {
    weight_shape.push_back(kernel_shape_data[i]);
  }
  this->bias_term_ = this->layer_param_.convolution_param().bias_term();
  vector<int> bias_shape(this->bias_term_, this->num_output_);
  if (this->blobs_.size() > 0) {
    CHECK_EQ(1 + this->bias_term_, this->blobs_.size())
        << "Incorrect number of weight blobs.";
    if (weight_shape != this->blobs_[0]->shape()) {
      Blob<Dtype> weight_shaped_blob(weight_shape);
      LOG(FATAL) << "Incorrect weight shape: expected shape "
          << weight_shaped_blob.shape_string() << "; instead, shape was "
          << this->blobs_[0]->shape_string();
    }
    if (this->bias_term_ && bias_shape != this->blobs_[1]->shape()) {
      Blob<Dtype> bias_shaped_blob(bias_shape);
      LOG(FATAL) << "Incorrect bias shape: expected shape "
          << bias_shaped_blob.shape_string() << "; instead, shape was "
          << this->blobs_[1]->shape_string();
    }
    LOG(INFO) << "Skipping parameter initialization";
  } else {
    if (this->bias_term_) {
      this->blobs_.resize(2);
    } else {
      this->blobs_.resize(1);
    }
    // Initialize and fill the weights:
    // output channels x input channels per-group x kernel height x kernel width
    this->blobs_[0].reset(new Blob<Dtype>(weight_shape));
    shared_ptr<Filler<Dtype> > weight_filler(GetFiller<Dtype>(
        this->layer_param_.convolution_param().weight_filler()));
    weight_filler->Fill(this->blobs_[0].get());
    // If necessary, initialize and fill the biases.
    if (this->bias_term_) {
      this->blobs_[1].reset(new Blob<Dtype>(bias_shape));
      shared_ptr<Filler<Dtype> > bias_filler(GetFiller<Dtype>(
          this->layer_param_.convolution_param().bias_filler()));
      bias_filler->Fill(this->blobs_[1].get());
    }
  }
  // Propagate gradients to the parameters (as directed by backward pass).
  this->param_propagate_down_.resize(this->blobs_.size(), true);

  if ((bottom[0]->shape(2) / this->group_) % 16 == 0)
    weight_pad_ = bottom[0]->shape(2) / this->group_;
  else
    weight_pad_ = ((bottom[0]->shape(2) / this->group_) / 16 + 1) * 16;

  CRParameter cr_param = this->layer_param_.cr_param();
  int num_ = bottom[0]->shape(3);
  num_cu_ = cr_param.num_cu();
  num_pe_ = cr_param.num_pe();
  switch(num_pe_) {
    case 4:   burstoc_limit_ = (16 * 256) / num_;
              break;
    case 8:   burstoc_limit_ = (32 * 256) / num_;
              break;
    case 16:  burstoc_limit_ = (64 * 256) / num_;
              break;
  }

  kernel_params *forward_params = &ocl_params_;
  CHECK(num_ % 16 == 0);

  forward_params->ydim = bottom[0]->shape(0);
  forward_params->xdim = bottom[0]->shape(1);
  forward_params->inchannels = bottom[0]->shape(2);
  forward_params->outchannels = this->num_output_ / this->group_;
  forward_params->numimages = num_;
  forward_params->ksize = (this->blobs_[0])->shape(3);  

  int burstchannels_ = 8 * 256 * 256 / (forward_params->ksize *
      forward_params->ksize * forward_params->numimages);

  if (burstchannels_ > forward_params->inchannels) {
    burstchannels_ = forward_params->inchannels;
  } else {
    int tchannel = burstchannels_;
    while (forward_params->inchannels % tchannel != 0)
      tchannel--;
    burstchannels_ = tchannel;
  }
 
  int rpofm = num_cu_;
  int burstoc = 1;
  if (rpofm > forward_params->outchannels) {
    rpofm = forward_params->outchannels;
    burstoc = 1;
  } else {
    while (rpofm * burstoc < forward_params->outchannels) {
      if (burstoc < burstoc_limit_)
        burstoc++;
      else
        rpofm++;
    }
  }
  int ksize = (this->blobs_[0])->shape(3);
  CHECK(burstoc * (num_ / 16) >= 16);
  CHECK(burstoc * burstchannels_ * ksize * ksize >= 16);

  forward_params->rpofm = rpofm;
  forward_params->xtile_pad = 0;
  forward_params->burstydim = burstoc;
  forward_params->stride = stride_data[0];
  forward_params->pad = pad_data[0];
  forward_params->burstchannels = burstchannels_;
  forward_params->rpo = forward_params->inchannels / burstchannels_;
  forward_params->numgroups = this->group_;
  forward_params->fc = 0;
  forward_params->relu = cr_param.relu();
  forward_params->pool = 0;
  forward_params->pksize = 2;

  // Backward params
  this->bottom_shape_ = &bottom[0]->shape();
  compute_output_shape();
  kernel_params *backward_params = &ocl_params_bw_;
  backward_params->ydim = bottom[0]->shape(0);
  backward_params->xdim = bottom[0]->shape(1);
  backward_params->outchannels = this->num_output_ / this->group_;
  backward_params->inchannels = bottom[0]->shape(2);
  backward_params->ksize = (this->blobs_[0])->shape(3);
  backward_params->numimages = num_;
  backward_params->rpofm = rpofm;
  backward_params->xtile_pad = 0;
  backward_params->burstydim = burstoc;
  backward_params->stride = stride_data[0];
  backward_params->pad = pad_data[0];
  backward_params->burstchannels = burstchannels_;
  backward_params->rpo = backward_params->inchannels / burstchannels_;
  backward_params->numgroups = this->group_;
  backward_params->fc = 1;
  backward_params->relu = cr_param.relu();
  backward_params->pool = 0;
  backward_params->pksize = 2;

  // backward wrt data
  kernel_params *backward_params_bi = &ocl_params_bi_;
  backward_params_bi->ydim = this->output_shape_[0];
  backward_params_bi->xdim = this->output_shape_[1];
  backward_params_bi->inchannels = this->num_output_ / this->group_;
  backward_params_bi->outchannels = bottom[0]->shape(2) / this->group_;
  backward_params_bi->ksize = (this->blobs_[0])->shape(3);
  backward_params_bi->numimages = num_;
  backward_params_bi->xtile_pad = 0;
  backward_params_bi->stride = stride_data[0];
  if ((pad_data[0] == 0) && (stride_data[0] == 1)) {
    backward_params_bi->pad = backward_params_bi->ksize - 1;
  } else {
    backward_params_bi->pad = pad_data[0];
  }
  burstchannels_ = 8 * 256 * 256 / (backward_params_bi->ksize *
      backward_params_bi->ksize * backward_params_bi->numimages);

  if (burstchannels_ > backward_params_bi->inchannels) {
    burstchannels_ = backward_params_bi->inchannels;
  } else {
    int tchannel = burstchannels_;
    while (backward_params_bi->inchannels % tchannel != 0)
      tchannel--;
    burstchannels_ = tchannel;
  }
  rpofm = num_cu_;
  burstoc = 1;

  if (backward_params_bi->outchannels < num_cu_) {
    rpofm = 1;
    burstoc = 1; 
    while (rpofm * burstoc < backward_params_bi->outchannels) {
      if (burstoc < burstoc_limit_)
        burstoc++;
      else
        rpofm++;
    }
  } else {
    rpofm = num_cu_;
    burstoc = 1;
    while (rpofm * burstoc < backward_params_bi->outchannels) {
      if (burstoc < burstoc_limit_)
        burstoc++;
      else
        rpofm++;
    }
  }

  CHECK(burstoc * (num_ / 16) >= 16);

  backward_params_bi->burstchannels = burstchannels_;
  backward_params_bi->rpo = backward_params_bi->inchannels / burstchannels_;
  backward_params_bi->numgroups = this->group_;
  backward_params_bi->fc = 0;
  backward_params_bi->relu = cr_param.relu();
  backward_params_bi->pool = 0;
  backward_params_bi->pksize = 2;
  backward_params_bi->rpofm = rpofm;
  backward_params_bi->burstydim = burstoc;

  CHECK(burstchannels_ >= 16);

  // Set bias update parameters
  kernel_params *bias_params = &ocl_params_bb_;
  bias_params->rpofm = 1;
  bias_params->burstydim = 1;
  bias_params->stride = 1;
  bias_params->pad = 0;
  bias_params->ydim = backward_params_bi->ydim;
  bias_params->xdim = backward_params_bi->xdim;
  bias_params->inchannels = backward_params_bi->inchannels;
  bias_params->outchannels = 1;
  bias_params->burstchannels = burstchannels_;
  bias_params->numimages = num_;
  bias_params->ksize = 1;
  bias_params->rpo = bias_params->inchannels / burstchannels_;
  bias_params->numgroups = this->group_;
  bias_params->fc = 0;
  bias_params->relu = cr_param.relu();
  bias_params->pool = 0;
  bias_params->pksize = 2;

  vector<int> shape(4);
  shape[0] = bottom[0]->shape(0);
  shape[1] = bottom[0]->shape(1);
  shape[2] = 1;
  shape[3] = bottom[0]->shape(3);

  weights_placeholder.Reshape(shape);

  for (int i = 0; i < weights_placeholder.count(); ++i)
    (weights_placeholder.mutable_cpu_data())[i] = cpfp((float)1.0);

  shape = (this->blobs_[0])->shape();

  shape[0] = forward_params->rpofm * forward_params->burstydim;
  shape[1] = weight_pad_;

  weights_h.Reshape(shape);

  shape = (this->blobs_[0])->shape();

  if (shape[0] % 16 != 0)
    shape[0] = ((shape[0] / 16) + 1) * 16;
  shape[1] = backward_params_bi->rpofm * backward_params_bi->burstydim;

  weights_h_r.Reshape(shape);

  bias_h.Reshape((this->blobs_[1])->shape());
}

template <typename Dtype>
void OCLCRHWCNLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  // Shape the tops.
  vector<int> top_shape;
  for (int i = 0; i < this->num_spatial_axes_; ++i) {
    top_shape.push_back(this->output_shape_[i]);
  }

  top_shape.push_back(this->num_output_);
  top_shape.push_back(bottom[0]->shape(3));
  for (int top_id = 0; top_id < top.size(); ++top_id) {
    top[top_id]->Reshape(top_shape);
  }

  vector<int> shape(4);

  // Since it's HWCN, N will be shape(3) and should be divisible by 32 

  shape[0] = top[0]->shape(0);
  shape[1] = top[0]->shape(1);
  shape[2] = top[0]->shape(2);
  if (top[0]->shape(3) % 32 != 0)
    shape[3] = top[0]->shape(3) / 32 + 1;
  else
    shape[3] = top[0]->shape(3) / 32;

  relu_indices.Reshape(shape);
  bias_placeholder.Reshape(1, 1, 1, 1);
}

template <typename Dtype>
void OCLCRHWCNLayer<Dtype>::launchKernel(const cpfp *bottom,
    const cpfp *weights, const cpfp *bias, cpfp *top, int *tags,
    const int *params, int numgroups) {
  std::vector<cl_event> events;

  int events_size = numgroups;
  events.resize(events_size, 0);
  
  clSetKernelArg(this->ocl_kernel, 0, sizeof(cl_mem),
    (const void *)&bottom);
  clSetKernelArg(this->ocl_kernel, 1, sizeof(cl_mem),
    (const void *)&weights);
  clSetKernelArg(this->ocl_kernel, 2, sizeof(cl_mem),
    (const void *)&bias);
  clSetKernelArg(this->ocl_kernel, 3, sizeof(cl_mem),
    (const void *)&top);
  clSetKernelArg(this->ocl_kernel, 4, sizeof(cl_mem),
    (const void *)&tags);
  clSetKernelArg(this->ocl_kernel, 5, sizeof(cl_mem),
    (const void *)&params);
  for (int g = 0; g < numgroups; ++g) {
    clSetKernelArg(this->ocl_kernel, 6, sizeof(cl_int),
        (const void *)&g);
    clEnqueueTask(oclCommandQueue, this->ocl_kernel, 0, NULL, &(events[g]));
  }
  clWaitForEvents(events.size(), events.data()); 
}

template <typename Dtype>
void OCLCRHWCNLayer<Dtype>::compute_output_shape() {
  const int* kernel_shape_data = this->kernel_shape_.cpu_data();
  const int* stride_data = this->stride_.cpu_data();
  const int* pad_data = this->pad_.cpu_data();
  this->output_shape_.clear();
  for (int i = 0; i < this->num_spatial_axes_; ++i) {
    // i + 1 to skip channel axis
    const int input_dim = (*this->bottom_shape_)[i];
    const int output_dim = (input_dim + 2 * pad_data[i] - kernel_shape_data[i])
        / stride_data[i] + 1;
    this->output_shape_.push_back(output_dim);
  }
}

template <typename Dtype>
void OCLCRHWCNLayer<Dtype>::copyToHalf(const Dtype *input, cpfp *output,
    int size) {
  for (int i = 0; i < size; ++i)
    output[i] = cpfp((float)input[i]);
}


template <typename Dtype>
void OCLCRHWCNLayer<Dtype>::copyToHalfWeights(const Dtype *input,
    cpfp *output, kernel_params params) {
  int oc = params.outchannels * params.numgroups;
  int ic = params.inchannels;
  int bc = params.burstchannels;
  int ic_new = weight_pad_;
  int bc_new = weight_pad_ / params.rpo;
  int ksize = params.ksize;
  int burstoc = params.burstydim;
  int rpofm = params.rpofm;
  for (int o = 0; o < rpofm; ++o) {
    for (int b = 0; b < burstoc; ++b) {
      for (int n = 0; n < ic_new / bc_new; ++n) {
        for (int k = 0; k < params.ksize * params.ksize; ++k) {
          for (int m = 0; m < bc_new / num_pe_; ++m) {
            for (int j = 0; j < num_pe_; ++j) {
              int burst_idx = k * bc_new + m * num_pe_ + j + b * ksize * ksize
                * bc_new; 
              int in_idx = ((o * burstoc + b) * ic + n * bc + m + j *
                (bc / num_pe_)) * ksize * ksize + k;
              int out_idx = o * burstoc * ksize * ksize * ic_new +
                n * bc_new * ksize * ksize * burstoc + burst_idx;
              if (m < bc / num_pe_ && o * burstoc + b < oc) {
                output[out_idx] = cpfp((float)input[in_idx]);
              } else {
                output[out_idx] = cpfp(0);
              }
            }
          }
        }
      }
    }
  }
}

template <typename Dtype>
void OCLCRHWCNLayer<Dtype>::RotateWeightsHalf(const Dtype *input,
    cpfp *output, kernel_params params) {
  int oc = params.outchannels * params.numgroups;
  int ic = params.inchannels;
  int bc = params.burstchannels;
  int ksize = params.ksize;
  int rpofm = params.rpofm;
  int burstoc = params.burstydim;
  for (int o = 0; o < rpofm; ++o) {
    for (int b = 0; b < burstoc; ++b) {
      for (int n = 0; n < ic / bc; ++n) {
        for (int k = 0; k < ksize * ksize; ++k) {
          for (int m = 0; m < bc / num_pe_; ++m) {
            for (int j = 0; j < num_pe_; ++j) {
              int in_idx = ((n * bc + m + j * bc / num_pe_) * oc + o *
                burstoc + b) * ksize * ksize + k;
              int burst_idx = (ksize * ksize - 1 - k) * bc + m * num_pe_ + j +
                b * ksize * ksize * bc;
              int out_idx = o * burstoc * ksize * ksize * ic +
                n * bc * ksize * ksize * burstoc + burst_idx;
              if (o * burstoc + b < oc) 
                output[out_idx] = cpfp((float)input[in_idx]);
              else
                output[out_idx] = 0;
            }
          }
        }
      }
    }
  }
}

template <typename Dtype>
void OCLCRHWCNLayer<Dtype>::copyToFloatWeights(cpfp *input,
    Dtype *output, const vector<int> shape, kernel_params params) {
  int oc = params.outchannels * params.numgroups;
  int ic = params.inchannels;
  int bc = params.burstchannels;
  int ic_new = weight_pad_;
  int bc_new = weight_pad_ / params.rpo;
  int ksize = params.ksize;
  int rpofm = params.rpofm;
  int burstoc = params.burstydim;

  for (int o = 0; o < rpofm; ++o) {
    for (int b = 0; b < burstoc; ++b) {
      for (int n = 0; n < ic_new / bc_new; ++n) {
        for (int k = 0; k < params.ksize * params.ksize; ++k) {
          for (int m = 0; m < bc / num_pe_; ++m) {
            for (int j = 0; j < num_pe_; ++j) {
              int in_idx = ((o * burstoc + b) * ic + n * bc + m + j *
                (bc / num_pe_)) * ksize * ksize + k;
              int burst_idx = k * bc_new + m * num_pe_ + j + b * ksize * ksize
                * bc_new;
              int out_idx = o * burstoc * ksize * ksize * ic_new + 
                n * ksize * ksize * burstoc * bc_new + burst_idx;
              if (o * burstoc + b < oc)
                output[in_idx] = (Dtype)float(input[out_idx]);
            }
          }
        }
      }
    }
  }
}

template <typename Dtype>
void OCLCRHWCNLayer<Dtype>::backward_bias(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  kernel_params *params = &ocl_params_bb_;

  vector<int> shape(1);
  const cpfp *weights_data = weights_placeholder.ocl_data();

  cpfp *bias_diff = bias_h.mutable_ocl_diff(0);

  params->backward = 1;
  int numgroups = params->numgroups;

  shape[0] = sizeof(kernel_params) / sizeof(int);
  param_vals.Reshape(shape);

  int *conv_params = param_vals.mutable_cpu_data();

  for (int i = 0; i < shape[0]; ++i) {
    conv_params[i] = ((int *)params)[i];
  }
  
  const int* cr_params_b = param_vals.ocl_data();

  size_t outsize = sizeof(cpfp) * top[0]->count();

  const cpfp *top_diff;
  int *relu_vals;
  for (int i = 0; i < bottom.size(); i++) {
    top_diff = reinterpret_cast<const cpfp *>(top[i]->ocl_diff(outsize));
    relu_vals = relu_indices.mutable_ocl_data();
    launchKernel(top_diff, weights_data, (const cpfp *)bias_diff, bias_diff,
        relu_vals, cr_params_b, numgroups);
  } 
  bias_diff = bias_h.mutable_cpu_diff();
  Dtype *bias_diff_out = this->blobs_[1]->mutable_cpu_diff();

  for (int i = 0; i < bias_h.count() / num_pe_; ++i) {
    for (int j = 0; j < num_pe_; ++j) {
      bias_diff_out[i + j * bias_h.count() / num_pe_] =
        (Dtype)float(bias_diff[i * num_pe_ + j]);
    }
  }
}

template <typename Dtype>
void OCLCRHWCNLayer<Dtype>::backward_data(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  kernel_params *params = &ocl_params_bi_;
  RotateWeightsHalf(this->blobs_[0]->cpu_data(),
      weights_h_r.mutable_cpu_data(), ocl_params_bi_);

  const cpfp *weight_data_r = weights_h_r.ocl_data();
  vector<int> shape(1);
  shape[0] = bottom[0]->shape(2);

  const cpfp *bias_data = bias_placeholder.ocl_data();

  params->backward = 2;
  int numgroups = params->numgroups;

  shape[0] = sizeof(kernel_params) / sizeof(int);
  param_vals.Reshape(shape);

  int *conv_params = param_vals.mutable_cpu_data();

  for (int i = 0; i < shape[0]; ++i) {
    conv_params[i] = ((int *)params)[i];
  }
  
  const int* cr_params_b = param_vals.ocl_data();

  size_t insize = sizeof(cpfp) * bottom[0]->count();
  size_t outsize = sizeof(cpfp) * top[0]->count();

  const cpfp *top_diff;
  int *relu_vals;
  cpfp *bottom_diff;
  for (int i = 0; i < bottom.size(); i++) {
    bottom_diff =
      reinterpret_cast<cpfp *>(bottom[i]->mutable_ocl_diff(0, insize));
    top_diff = reinterpret_cast<const cpfp *>(top[i]->ocl_diff(outsize));
    relu_vals = relu_indices.mutable_ocl_data();
    launchKernel(top_diff, weight_data_r, bias_data, bottom_diff, relu_vals,
        cr_params_b, numgroups);
  }
}

template <typename Dtype>
void OCLCRHWCNLayer<Dtype>::backward_weights(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  kernel_params *params = &ocl_params_bw_;

  Dtype* weight_diff_dtype = this->blobs_[0]->mutable_cpu_diff();
 
  cpfp* weight_diff = weights_h.mutable_ocl_diff(0);

  const cpfp *bias_data = bias_placeholder.ocl_data();

  params->backward = 1;
  int numgroups = params->numgroups;

  vector<int> shape(1);
  shape[0] = sizeof(kernel_params) / sizeof(int);
  param_vals.Reshape(shape);

  int *conv_params = param_vals.mutable_cpu_data();

  for (int i = 0; i < shape[0]; ++i) {
    conv_params[i] = ((int *)params)[i];
  }
  
  const int* cr_params_b = param_vals.ocl_data();

  size_t insize = sizeof(cpfp) * bottom[0]->count();
  size_t outsize = sizeof(cpfp) * top[0]->count();

  const cpfp *top_diff;
  int *relu_vals;
  const cpfp *bottom_data;
  for (int i = 0; i < bottom.size(); i++) {
    bottom_data =
      reinterpret_cast<const cpfp *>(bottom[i]->ocl_data(insize));
    top_diff = reinterpret_cast<const cpfp *>(top[i]->ocl_diff(outsize));
    relu_vals = relu_indices.mutable_ocl_data();
    launchKernel(bottom_data, top_diff, bias_data, weight_diff, relu_vals,
        cr_params_b, numgroups);
  }
  weight_diff = weights_h.mutable_cpu_diff();
  copyToFloatWeights(weight_diff, weight_diff_dtype,
      this->blobs_[0]->shape(), ocl_params_bw_);
}


template <typename Dtype>
void OCLCRHWCNLayer<Dtype>::Forward_ocl(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  kernel_params *params = &ocl_params_;
  copyToHalfWeights(this->blobs_[0]->cpu_data(),
      weights_h.mutable_cpu_data(), ocl_params_);
  copyToHalf(this->blobs_[1]->mutable_cpu_data(), bias_h.mutable_cpu_data(),
      params->outchannels * params->numgroups);
  const cpfp *weight_data = weights_h.ocl_data();
  const cpfp *bias_data = bias_h.ocl_data();

  params->backward = 0;
  int numgroups = params->numgroups;

  vector<int> shape(1);
  shape[0] = sizeof(kernel_params) / sizeof(int);
  param_vals.Reshape(shape);

  int *conv_params = param_vals.mutable_cpu_data();

  for (int i = 0; i < shape[0]; ++i) {
    conv_params[i] = ((int *)params)[i];
  }
  const int* cr_params = param_vals.ocl_data();

  size_t insize = sizeof(cpfp) * bottom[0]->count();
  size_t outsize = sizeof(cpfp) * top[0]->count();

  cpfp *top_data;
  int *relu_vals;
  for (int i = 0; i < bottom.size(); i++) {
    const cpfp* bottom_data =
      reinterpret_cast<const cpfp *>(bottom[i]->ocl_data(insize));
    top_data = reinterpret_cast<cpfp *>(top[i]->mutable_ocl_data(0, outsize));
    relu_vals = relu_indices.mutable_ocl_data(0);
    launchKernel(bottom_data, weight_data, bias_data, top_data, relu_vals,
        cr_params, numgroups);
  }
}

template <typename Dtype>
void OCLCRHWCNLayer<Dtype>::Backward_ocl(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  if (this->bias_term_ && this->param_propagate_down_[1])
    backward_bias(top, propagate_down, bottom);
  
  if (this->param_propagate_down_[0])
    backward_weights(top, propagate_down, bottom);

  if (propagate_down[0])
    backward_data(top, propagate_down, bottom);
}


INSTANTIATE_CLASS(OCLCRHWCNLayer);
REGISTER_LAYER_CLASS(OCLCRHWCN);
#endif
}  // namespace caffe
