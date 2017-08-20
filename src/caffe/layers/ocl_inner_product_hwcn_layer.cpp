#include <vector>

#include "caffe/filler.hpp"
#include "caffe/layers/ocl_inner_product_hwcn_layer.hpp"

namespace caffe {

#ifdef USE_OCL

template <typename Dtype>
void OCLHWCNInnerProductLayer<Dtype>::LayerSetUp(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  const int num_output = this->layer_param_.inner_product_param().num_output();
  this->N_ = num_output;
  this->bias_term_ = this->layer_param_.inner_product_param().bias_term();
  this->transpose_ = this->layer_param_.inner_product_param().transpose();
  if (bottom[0]->num_axes() == 2)
    this->M_ = bottom[0]->shape(1);
  else
    this->M_ = bottom[0]->shape(3);

  if (bottom[0]->num_axes() == 2)
    this->K_ = bottom[0]->shape(0);
  else
    this->K_ = bottom[0]->count(0, 3);
  if (this->blobs_.size() > 0) {
    LOG(INFO) << "Skipping parameter initialization";
  } else {
    if (this->bias_term_) {
      this->blobs_.resize(2);
    } else {
      this->blobs_.resize(1);
    }
    // Initialize the weights
    vector<int> weight_shape(2);
    if (this->transpose_) {
      weight_shape[0] = this->K_;
      weight_shape[1] = this->N_;
    } else {
      weight_shape[0] = this->N_;
      weight_shape[1] = this->K_;
    }
    this->blobs_[0].reset(new Blob<Dtype>(weight_shape));
    // fill the weights
    shared_ptr<Filler<Dtype> > weight_filler(GetFiller<Dtype>(
        this->layer_param_.inner_product_param().weight_filler()));
    weight_filler->Fill(this->blobs_[0].get());
    // If necessary, intiialize and fill the bias term
    if (this->bias_term_) {
      vector<int> bias_shape(1, this->N_);
      this->blobs_[1].reset(new Blob<Dtype>(bias_shape));
      shared_ptr<Filler<Dtype> > bias_filler(GetFiller<Dtype>(
          this->layer_param_.inner_product_param().bias_filler()));
      bias_filler->Fill(this->blobs_[1].get());
    }
  }  // parameter initialization
  this->param_propagate_down_.resize(this->blobs_.size(), true);

  CRParameter cr_param = this->layer_param_.cr_param();
  num_cu_ = cr_param.num_cu();
  num_pe_ = cr_param.num_pe();
  switch(num_pe_) {
    case 4:   burstoc_limit_ = 16;
              break;
    case 8:   burstoc_limit_ = 32;
              break;
    case 16:  burstoc_limit_ = 64;
              break;
  }
  kernel_params *params = &ocl_params_;
  params->inchannels = this->K_;
  params->numgroups = 1;
  params->xdim = 1;
  params->ydim = 1;
  params->ksize = 1;
  params->xtile_pad = 0;
  params->fc = 0;
  params->stride = 1;
  params->pad = 0;
  params->relu = cr_param.relu();
  params->outchannels = this->N_;
  params->numimages = this->M_;
  int burstchannels_ = 8 * 256 * 256 / (params->numimages);

  if (burstchannels_ > params->inchannels) {
    burstchannels_ = params->inchannels;
  } else {
    int tchannel = burstchannels_;
    while (params->inchannels % tchannel != 0)
      tchannel--;
    burstchannels_ = tchannel;
  }
  
  int rpofm = num_cu_;
  int burstoc = 1;
  if (rpofm > params->outchannels) {
    rpofm = params->outchannels;
    burstoc = 1;
  } else {
    while (rpofm * burstoc < params->outchannels) {
      if (burstoc < burstoc_limit_)
        burstoc++;
      else
        rpofm++;
    }
  }
  params->rpofm = rpofm;
  params->burstydim = burstoc;
  params->burstchannels = burstchannels_;
  params->rpo = params->inchannels / burstchannels_;
  params->pool = 0;
  params->pksize = 2;

  // Backward params
  kernel_params *backward_params = &ocl_params_bw_;
  backward_params->ydim = 1;
  backward_params->xdim = 1;
  backward_params->outchannels = params->outchannels;
  backward_params->inchannels = params->inchannels;
  backward_params->ksize = 1;
  backward_params->numimages = params->numimages;
  backward_params->xtile_pad = 0;
  backward_params->stride = 1;
  backward_params->pad = 0;
  backward_params->burstchannels = params->burstchannels;
  backward_params->rpo = params->rpo;
  backward_params->numgroups = 1;
  backward_params->fc = 1;
  backward_params->relu = cr_param.relu();
  backward_params->pool = 0;
  backward_params->pksize = 2;
  backward_params->burstydim = params->burstydim;
  backward_params->rpofm = params->rpofm;

  // Backward params inputs
  kernel_params *backward_params_bi = &ocl_params_bi_;
  backward_params_bi->ydim = 1;
  backward_params_bi->xdim = 1;
  backward_params_bi->inchannels = this->N_;
  backward_params_bi->outchannels = this->K_;
  backward_params_bi->ksize = 1;
  backward_params_bi->numimages = this->M_;

  backward_params_bi->xtile_pad = 0;
  backward_params_bi->stride = 1;
  backward_params_bi->pad = 0;
  backward_params_bi->numgroups = 1;
  backward_params_bi->fc = 0;
  backward_params_bi->relu = cr_param.relu();
  backward_params_bi->pool = 0;
  backward_params_bi->pksize = 2;

  rpofm = num_cu_;
  burstoc = 1;
  if (rpofm > backward_params_bi->outchannels) {
    rpofm = backward_params_bi->outchannels;
    burstoc = 1;
  } else {
    while (rpofm * burstoc < backward_params_bi->outchannels) {
      if (burstoc < burstoc_limit_)
        burstoc++;
      else
        rpofm++;
    }
  }
  backward_params_bi->burstydim = burstoc;
  backward_params_bi->rpofm = rpofm;

  burstchannels_ = 8 * 256 * 256 / (backward_params_bi->numimages);

  if (burstchannels_ > backward_params_bi->inchannels) {
    burstchannels_ = backward_params_bi->inchannels;
  } else {
    int tchannel = burstchannels_;
    while (backward_params_bi->inchannels % tchannel != 0)
      tchannel--;
    burstchannels_ = tchannel;
  }

  if (burstoc * burstchannels_ / num_pe_ < 16) {
    backward_params_bi->inchannels = 16;
    burstchannels_ = 16;
    use_aux_ = true;
  }

  if (burstchannels_ % 16 != 0) {
    int rpo = backward_params_bi->inchannels / burstchannels_;
    burstchannels_ = (burstchannels_ / 16 + 1) * 16;
    backward_params_bi->inchannels = burstchannels_ * rpo;
    use_aux_ = true;
  }
  backward_params_bi->burstchannels = burstchannels_;
  backward_params_bi->rpo = backward_params_bi->inchannels / burstchannels_;

  // Set bias update parameters
  kernel_params *bias_params = &ocl_params_bb_;
  bias_params->burstydim = 1;
  bias_params->rpofm = 1;
  bias_params->stride = 1;
  bias_params->pad = 0;
  bias_params->ydim = 1;
  bias_params->xdim = 1;
  bias_params->inchannels = this->N_;
  bias_params->outchannels = 1;
  bias_params->numimages = this->M_;
  burstchannels_ = 8 * 256 * 256 / (bias_params->numimages);

  if (burstchannels_ > bias_params->inchannels) {
    burstchannels_ = bias_params->inchannels;
  } else {
    int tchannel = burstchannels_;
    while (bias_params->inchannels % tchannel != 0)
      tchannel--;
    burstchannels_ = tchannel;
  }

  if (burstchannels_ / num_pe_ < 16) {
    bias_params->inchannels = 16;
    burstchannels_ = 16;
    use_aux_ = true;
  }

  if (burstchannels_ % 16 != 0) {
    int rpo = bias_params->inchannels / burstchannels_;
    burstchannels_ = (burstchannels_ / 16 + 1) * 16;
    bias_params->inchannels = burstchannels_ * rpo;
    use_aux_ = true;
  }

  bias_params->burstchannels = burstchannels_;
  bias_params->ksize = 1;
  bias_params->rpo = bias_params->inchannels / burstchannels_;
  bias_params->numgroups = 1;
  bias_params->fc = 0;
  bias_params->relu = cr_param.relu();
  bias_params->pool = 0;
  bias_params->pksize = 2;
  vector<int> shape(1);
  shape[0] = this->M_;
  weights_placeholder.Reshape(shape);

  for (int i = 0; i < weights_placeholder.count(); ++i)
    (weights_placeholder.mutable_cpu_data())[i] = cpfp((float)1.0);
}

template <typename Dtype>
void OCLHWCNInnerProductLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  kernel_params *bias_params = &ocl_params_bb_;
 
  std::vector<int> top_shape(2);
  top_shape[0] = this->N_;
  top_shape[1] = this->M_;
  top[0]->Reshape(top_shape);

  top_shape[0] = bias_params->inchannels;
  relu_indices.Reshape(top_shape);

  top_shape[0] = 1;
  top_shape[1] = this->N_;

  bias_h.Reshape(top_shape);

  vector<int> weight_shape(2);

  weight_shape[0] = this->N_;
  weight_shape[1] = this->K_;

  if (weight_shape[1] % 16 != 0)
    weight_shape[1] = (weight_shape[1] / 16 + 1) * 16;

  weights_h.Reshape(weight_shape);

  weight_shape[0] = this->K_;
  weight_shape[1] = ocl_params_bi_.inchannels;
  
  if (use_aux_) {
    top_shape[0] = bias_params->inchannels;
    top_shape[1] = this->M_;
    top_aux.Reshape(top_shape);
  }

  weights_h_t.Reshape(weight_shape);
  if (this->bias_term_) {
    vector<int> bias_shape(1, this->M_);
    this->bias_multiplier_.Reshape(bias_shape);
    caffe_set(this->M_, Dtype(1), this->bias_multiplier_.mutable_cpu_data());
  }
  bias_placeholder.Reshape(1, 1, 1, 1);
}

template <typename Dtype>
void OCLHWCNInnerProductLayer<Dtype>::copyToHalf(const Dtype *input,
    cpfp *output, int size, int xdim, int xdim_pad) {
  for (int i = 0; i < size; ++i)
    for (int j = 0; j < xdim_pad; ++j)
      if (j < xdim)
        output[i * xdim_pad + j] = cpfp((float)input[i * xdim + j]);
      else
        output[i * xdim_pad + j] = cpfp(0);
}

template <typename Dtype>
void OCLHWCNInnerProductLayer<Dtype>::launchKernel(const cpfp *bottom,
    const cpfp *weights, const cpfp *bias, cpfp *top, int *tags,
    const int *params) {
  std::vector<cl_event> events;
  int g = 0;
  int events_size = 1;
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
  clSetKernelArg(this->ocl_kernel, 6, sizeof(cl_int),
      (const void *)&g);
  clEnqueueTask(oclCommandQueue, this->ocl_kernel, 0, NULL, &(events[g]));
  clWaitForEvents(events.size(), events.data()); 
}

template <typename Dtype>
void OCLHWCNInnerProductLayer<Dtype>::Forward_ocl(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  kernel_params *params = &ocl_params_;
  const Dtype *weights_dtype = this->blobs_[0]->cpu_data();
  cpfp *weight_data_temp = weights_h.mutable_cpu_data();

  int weight_size = weights_h.count();

  for (int i = 0; i < weight_size; ++i)
    weight_data_temp[i] = cpfp((float)weights_dtype[i]);

  if (this->bias_term_)
    copyToHalf(this->blobs_[1]->mutable_cpu_data(), bias_h.mutable_cpu_data(),
        params->outchannels, 1, 1);
  else
    (bias_h.mutable_cpu_data())[0] = cpfp(0);

  const cpfp *weight_data = weights_h.ocl_data();
  const cpfp *bias_data = bias_h.ocl_data();

  params->backward = 0;

  vector<int> shape(1);
  shape[0] = sizeof(kernel_params) / sizeof(int);
  param_vals.Reshape(shape);

  int *ip_params = param_vals.mutable_cpu_data();

  for (int i = 0; i < shape[0]; ++i) {
    ip_params[i] = ((int *)params)[i];
  }
  
  const int* k_params = param_vals.ocl_data();

  size_t insize = sizeof(cpfp) * bottom[0]->count();
  cpfp *top_data;
  int *relu_vals;
  for (int i = 0; i < bottom.size(); i++) {
    const cpfp *bottom_data =
      reinterpret_cast<const cpfp *>(bottom[i]->ocl_data(insize));
    top_data = reinterpret_cast<cpfp *>(top[i]->mutable_ocl_data(0));
    relu_vals = relu_indices.mutable_ocl_data(0);
    launchKernel(bottom_data, weight_data, bias_data, top_data, relu_vals,
        k_params);
  }
}

template <typename Dtype>
void OCLHWCNInnerProductLayer<Dtype>::backward_weights(
    const vector<Blob<Dtype>*>& top, const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  kernel_params *params = &ocl_params_bw_;
  params->backward = 1;
  Dtype* weight_diff_dtype = this->blobs_[0]->mutable_cpu_diff();
 
  cpfp* weight_diff = weights_h.mutable_ocl_diff(0);
  vector<int> shape(1);
  shape[0] = this->blobs_[0]->shape(1);

  const cpfp *bias_data = bias_placeholder.ocl_data();

  shape[0] = sizeof(kernel_params) / sizeof(int);
  param_vals.Reshape(shape);

  int *ip_params = param_vals.mutable_cpu_data();
  for (int i = 0; i < shape[0]; ++i) {
    ip_params[i] = ((int *)params)[i];
  }
  
  const int* cr_params_b = param_vals.ocl_data();

  size_t insize = sizeof(cpfp) * bottom[0]->count();
  size_t outsize = sizeof(cpfp) * top[0]->count();

  const cpfp *top_diff;
  int *relu_vals;
  const cpfp *bottom_data;

  for (int i = 0; i < bottom.size(); i++) {
    if (use_aux_) {
      outsize = sizeof(cpfp) * top_aux.count();
      top_diff = top_aux.ocl_diff(outsize);
    } else {
      top_diff = reinterpret_cast<const cpfp *>(top[i]->ocl_diff(outsize));
    }
    bottom_data = reinterpret_cast<const cpfp *>(bottom[i]->ocl_data(insize));
    relu_vals = relu_indices.mutable_ocl_data();
    launchKernel(bottom_data, top_diff, bias_data, weight_diff, relu_vals,
        cr_params_b);
  }
  weight_diff = weights_h.mutable_cpu_diff();

  int weight_size = weights_h.count();

  for (int i = 0; i < weight_size; ++i)
    weight_diff_dtype[i] = (Dtype)float(weight_diff[i]);
}

template <typename Dtype>
void OCLHWCNInnerProductLayer<Dtype>::backward_bias(
    const vector<Blob<Dtype>*>& top, const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  kernel_params *params = &ocl_params_bb_;
  params->backward = 1;
  vector<int> shape(1);

  const cpfp *weights_data = weights_placeholder.ocl_data();

  shape[0] = weights_h_t.shape(1);

  cpfp *bias_diff = bias_h.mutable_ocl_diff(0);

  shape[0] = sizeof(kernel_params) / sizeof(int);
  param_vals.Reshape(shape);

  int *ip_params = param_vals.mutable_cpu_data();

  for (int i = 0; i < shape[0]; ++i) {
    ip_params[i] = ((int *)params)[i];
  }
  
  const int* cr_params_b = param_vals.ocl_data();

  size_t outsize = sizeof(cpfp) * top[0]->count();
  const cpfp *top_diff;
  int *relu_vals;
  for (int i = 0; i < bottom.size(); i++) {
    if (use_aux_) {
      outsize = sizeof(cpfp) * top_aux.count();
      top_diff = top_aux.ocl_diff(outsize);
    } else {
      top_diff = reinterpret_cast<const cpfp *>(top[i]->ocl_diff(outsize));
    }
    relu_vals = relu_indices.mutable_ocl_data();
    launchKernel(top_diff, weights_data, (const cpfp *)bias_diff, bias_diff,
        relu_vals, cr_params_b);
  }
  bias_diff = bias_h.mutable_cpu_diff();
  Dtype *bias_diff_out = this->blobs_[1]->mutable_cpu_diff();
  for (int i = 0; i < bias_h.count() / num_pe_; ++i) {
    for (int j = 0; j < num_pe_; ++j)
      if (i + j * bias_h.count() / num_pe_ < this->blobs_[1]->count())
        bias_diff_out[i + j * bias_h.count() / num_pe_] =
          (Dtype)float(bias_diff[i * num_pe_ + j]);
  }
}

template <typename Dtype>
void OCLHWCNInnerProductLayer<Dtype>::backward_data(
    const vector<Blob<Dtype>*>& top, const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  kernel_params *params = &ocl_params_bi_;
  params->backward = 2;
  int outchannels = this->blobs_[0]->shape(0);
  int inchannels = this->blobs_[0]->shape(1);

  const Dtype *weight_data = this->blobs_[0]->cpu_data();
  cpfp *weight_data_h_t = weights_h_t.mutable_cpu_data();
  int outchannels_pad = weights_h_t.shape(1);
  for (int i = 0; i < outchannels_pad / num_pe_; ++i) {
    for (int k = 0; k < num_pe_; ++k) {
      int oc_idx_in = i + k * outchannels_pad / num_pe_;
      int oc_idx_out = i * num_pe_ + k;
      if (i + k * outchannels_pad / num_pe_ < outchannels) {
        for (int j = 0; j < inchannels / num_pe_; ++j) {
          for (int n = 0; n < num_pe_; ++n) {
            int ic_idx_in = j * num_pe_ + n;
            int ic_idx_out = j + n * inchannels / num_pe_;
            weight_data_h_t[ic_idx_out * outchannels_pad + oc_idx_out] =
              cpfp((float)weight_data[oc_idx_in * inchannels + ic_idx_in]);
          }
        }
      } else {
        for (int j = 0; j < inchannels; ++j) {
          weight_data_h_t[j * outchannels_pad + oc_idx_out] = 0;
        }
      }
    }
  }

  const cpfp *weight_data_t = weights_h_t.ocl_data();

  vector<int> shape(1);

  const cpfp *bias_data = bias_placeholder.ocl_data();

  shape[0] = sizeof(kernel_params) / sizeof(int);
  param_vals.Reshape(shape);

  int *ip_params = param_vals.mutable_cpu_data();

  for (int i = 0; i < shape[0]; ++i) {
    ip_params[i] = ((int *)params)[i];
  }
  
  const int* cr_params_b = param_vals.ocl_data();

  size_t insize = sizeof(cpfp) * bottom[0]->count();
  size_t outsize = sizeof(cpfp) * top[0]->count();
  const cpfp *top_diff;
  int *relu_vals;
  cpfp *bottom_diff;

  for (int i = 0; i < bottom.size(); i++) {
    if (use_aux_) {
      outsize = sizeof(cpfp) * top_aux.count();
      top_diff = top_aux.ocl_diff(outsize);
    } else {
      top_diff = reinterpret_cast<const cpfp *>(top[i]->ocl_diff(outsize));
    }
    bottom_diff =
      reinterpret_cast<cpfp *>(bottom[i]->mutable_ocl_diff(0, insize));
    relu_vals = relu_indices.mutable_ocl_data();
    launchKernel(top_diff, weight_data_t, bias_data, bottom_diff, relu_vals,
        cr_params_b);
  }
}


template <typename Dtype>
void OCLHWCNInnerProductLayer<Dtype>::Backward_ocl(
    const vector<Blob<Dtype>*>& top, const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  if (use_aux_) {
    const cpfp *top_diff =
      reinterpret_cast<const cpfp *>(top[0]->cpu_diff());
    cpfp *top_diff_aux = top_aux.mutable_cpu_diff();
    for (int j = 0; j < top_aux.shape(0); ++j)
      for (int k = 0; k < top_aux.shape(1); ++k) 
        if (j < top[0]->shape(0))
          top_diff_aux[j * top_aux.shape(1) + k] =
            top_diff[j * top[0]->shape(1) + k];
        else
          top_diff_aux[j * top_aux.shape(1) + k] = 0;
  }

  if (this->param_propagate_down_[0])
    backward_weights(top, propagate_down, bottom);

  if (propagate_down[0])
    backward_data(top, propagate_down, bottom);

  if (this->bias_term_ && this->param_propagate_down_[1])
    backward_bias(top, propagate_down, bottom);
}


INSTANTIATE_CLASS(OCLHWCNInnerProductLayer);
REGISTER_LAYER_CLASS(OCLHWCNInnerProduct);
#endif
}  // namespace caffe
