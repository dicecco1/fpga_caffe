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
  kernel_params *params = &ocl_params_;
  params->inchannels = this->K_;
  params->numgroups = 1;
  params->xdim = 1;
  params->ydim = 1;
  params->ksize = 1;
  params->rpofm = 0;
  params->burstydim = 0;
  params->xtile_pad = 0;
  params->fc = 1;
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

  params->burstchannels = burstchannels_;
  params->rpo = params->inchannels / burstchannels_;
  params->pool = 0;
  params->pksize = 2;

  // Backward params
  kernel_params *backward_params = &ocl_params_bi_;
  backward_params->ydim = 1;
  backward_params->xdim = 1;
  backward_params->inchannels = this->N_;
  backward_params->outchannels = this->K_;
  backward_params->ksize = 1;
  backward_params->numimages = this->M_;
  burstchannels_ = 8 * 256 * 256 / (backward_params->numimages);

  if (burstchannels_ > backward_params->inchannels) {
    burstchannels_ = backward_params->inchannels;
  } else {
    int tchannel = burstchannels_;
    while (backward_params->inchannels % tchannel != 0)
      tchannel--;
    burstchannels_ = tchannel;
  }

  if (burstchannels_ < 64) {
    backward_params->inchannels = 64;
    burstchannels_ = 64;
    use_aux_ = true;
  }

  if (burstchannels_ % 16 != 0) {
    int rpo = backward_params->inchannels / burstchannels_;
    burstchannels_ = (burstchannels_ / 16 + 1) * 16;
    backward_params->inchannels = burstchannels_ * rpo;
    use_aux_ = true;
  }

  backward_params->rpofm = 0;
  backward_params->xtile_pad = 0;
  backward_params->burstydim = 0;
  backward_params->stride = 1;
  backward_params->pad = 0;
  backward_params->burstchannels = burstchannels_;
  backward_params->rpo = backward_params->inchannels / burstchannels_;
  backward_params->numgroups = 1;
  backward_params->fc = 1;
  backward_params->relu = cr_param.relu();
  backward_params->pool = 0;
  backward_params->pksize = 2;
  // Set bias update parameters
  kernel_params *bias_params = &ocl_params_bb_;

  bias_params->stride = 1;
  bias_params->pad = 0;
  bias_params->ydim = 1;
  bias_params->xdim = 1;
  bias_params->inchannels = backward_params->inchannels;
  bias_params->outchannels = 1;
  bias_params->burstchannels = burstchannels_;
  bias_params->numimages = this->M_;
  bias_params->ksize = 1;
  bias_params->rpo = bias_params->inchannels / burstchannels_;
  bias_params->numgroups = 1;
  bias_params->fc = 1;
  bias_params->relu = cr_param.relu();
  bias_params->pool = 0;
  bias_params->pksize = 2;
  vector<int> shape(1);
  shape[0] = this->M_;
  weights_placeholder.Reshape(shape);

  for (int i = 0; i < weights_placeholder.count(); ++i)
    (weights_placeholder.mutable_cpu_data())[i] = chalf((float)1.0);
}

template <typename Dtype>
void OCLHWCNInnerProductLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  kernel_params *backward_params = &ocl_params_bi_;
 
  std::vector<int> top_shape(2);
  top_shape[0] = this->N_;
  top_shape[1] = this->M_;
  top[0]->Reshape(top_shape);

  top_shape[0] = backward_params->inchannels;
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
  weight_shape[1] = backward_params->inchannels;

  if (use_aux_) {
    top_shape[0] = backward_params->inchannels;
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
    chalf *output, int size, int xdim, int xdim_pad) {
  for (int i = 0; i < size; ++i)
    for (int j = 0; j < xdim_pad; ++j)
      if (j < xdim)
        output[i * xdim_pad + j] = chalf((float)input[i * xdim + j]);
      else
        output[i * xdim_pad + j] = chalf(0);
}

template <typename Dtype>
void OCLHWCNInnerProductLayer<Dtype>::copyToFloat(const chalf *input,
    Dtype *output, int size, int ksize, int ksize_pad) {
  return;
}

template <typename Dtype>
void OCLHWCNInnerProductLayer<Dtype>::Forward_ocl(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  kernel_params *params = &ocl_params_;
  const Dtype *weights_dtype = this->blobs_[0]->cpu_data();
  chalf *weight_data_temp = weights_h.mutable_cpu_data();

  int outchannels = this->blobs_[0]->shape(0);
  int inchannels = this->blobs_[0]->shape(1);
  int inchannels_pad = weights_h.shape(1);

  for (int i = 0; i < outchannels; ++i)
    for (int j = 0; j < inchannels_pad / 4; ++j)
      for (int k = 0; k < 4; ++k)
        if (k * inchannels_pad / 4 + j < inchannels)
          weight_data_temp[i * inchannels_pad + j * 4 + k] =
            chalf((float)weights_dtype[i * inchannels + j +
                k * inchannels_pad / 4]);
        else
          weight_data_temp[i * inchannels_pad + j * 4 + k] = chalf(0);

  if (this->bias_term_)
    copyToHalf(this->blobs_[1]->mutable_cpu_data(), bias_h.mutable_cpu_data(),
        params->outchannels, 1, 1);
  else
    (bias_h.mutable_cpu_data())[0] = chalf(0);

  const chalf *weight_data = weights_h.ocl_data();
  const chalf *bias_data = bias_h.ocl_data();

  params->backward = 0;

  vector<int> shape(1);
  shape[0] = sizeof(kernel_params) / sizeof(int);
  param_vals.Reshape(shape);

  int *ip_params = param_vals.mutable_cpu_data();

  for (int i = 0; i < shape[0]; ++i) {
    ip_params[i] = ((int *)params)[i];
  }
  
  const int* k_params = param_vals.ocl_data();

  size_t insize = sizeof(chalf) * bottom[0]->count();
  std::vector<cl_event> events;

  int events_size = 1;
  int g = 0;
  chalf *top_data;
  int *relu_vals;
  for (int i = 0; i < bottom.size(); i++) {
    events.resize(events_size, 0);
    const chalf *bottom_data =
      reinterpret_cast<const chalf *>(bottom[i]->ocl_data(insize));
    top_data = reinterpret_cast<chalf *>(top[i]->mutable_ocl_data(0));
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
      (const void *)&k_params);
    clSetKernelArg(this->ocl_kernel, 6, sizeof(cl_int),
        (const void *)&g);
    clEnqueueTask(oclCommandQueue, this->ocl_kernel, 0,
        NULL, &(events[g]));
    clWaitForEvents(events.size(), events.data());
  }
}

template <typename Dtype>
void OCLHWCNInnerProductLayer<Dtype>::backward_weights(
    const vector<Blob<Dtype>*>& top, const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  kernel_params *params = &ocl_params_bi_;
  params->backward = 1;
  Dtype* weight_diff_dtype = this->blobs_[0]->mutable_cpu_diff();
 
  chalf *weight_diff = weights_h_t.mutable_cpu_diff();

  for (int i = 0; i < weights_h_t.count(); ++i)
    weight_diff[i] = chalf(0);

  weight_diff = weights_h_t.mutable_ocl_diff();
  vector<int> shape(1);
  shape[0] = this->blobs_[0]->shape(1);

  const chalf *bias_data = bias_placeholder.ocl_data();

  shape[0] = sizeof(kernel_params) / sizeof(int);
  param_vals.Reshape(shape);

  int *ip_params = param_vals.mutable_cpu_data();
  for (int i = 0; i < shape[0]; ++i) {
    ip_params[i] = ((int *)params)[i];
  }
  
  const int* cr_params_b = param_vals.ocl_data();

  size_t insize = sizeof(chalf) * bottom[0]->count();
  size_t outsize = sizeof(chalf) * top[0]->count();
  std::vector<cl_event> events;

  int events_size = 1;
  int g = 0;

  const chalf *top_diff;
  const int *relu_vals;
  const chalf *bottom_data;

  for (int i = 0; i < bottom.size(); i++) {
    if (use_aux_) {
      top_diff = reinterpret_cast<const chalf *>(top[i]->cpu_diff());
      chalf *top_diff_aux = top_aux.mutable_cpu_diff();
      for (int j = 0; j < top_aux.shape(0); ++j)
        for (int k = 0; k < top_aux.shape(1); ++k) 
          if (j < top[i]->shape(0))
            top_diff_aux[j * top_aux.shape(1) + k] =
              top_diff[j * top[i]->shape(1) + k];
          else
            top_diff_aux[j * top_aux.shape(1) + k] = 0;
      outsize = top_aux.count();
      top_diff = top_aux.ocl_diff(outsize);
    } else {
      top_diff = reinterpret_cast<const chalf *>(top[i]->ocl_diff(outsize));
    }
    events.resize(events_size, 0);
    bottom_data = reinterpret_cast<const chalf *>(bottom[i]->ocl_data(insize));
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
    clSetKernelArg(this->ocl_kernel, 6, sizeof(cl_int),
        (const void *)&g);
    clEnqueueTask(oclCommandQueue, this->ocl_kernel, 0,
        NULL, &(events[g]));
    clWaitForEvents(events.size(), events.data());
  }
  weight_diff = weights_h_t.mutable_cpu_diff();
  
  int outchannels = this->blobs_[0]->shape(0);
  int inchannels = this->blobs_[0]->shape(1);

  for (int i = 0; i < weights_h_t.shape(1) / 4; ++i)
    for (int j = 0; j < inchannels; ++j)
      for (int k = 0; k < 4; ++k)
        if (i + k * weights_h_t.shape(1) / 4 < outchannels)
          weight_diff_dtype[(i + k * weights_h_t.shape(1) / 4)
            * inchannels + j] =
            (Dtype)float(weight_diff[j * weights_h_t.shape(1) + i * 4 + k]);

}

template <typename Dtype>
void OCLHWCNInnerProductLayer<Dtype>::backward_bias(
    const vector<Blob<Dtype>*>& top, const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  kernel_params *params = &ocl_params_bb_;
  params->backward = 1;
  vector<int> shape(1);

  const chalf *weights_data = weights_placeholder.ocl_data();

  shape[0] = weights_h_t.shape(1);

  for (int i = 0; i < bias_h.count(); ++i)
    (bias_h.mutable_cpu_diff())[i] = chalf(0);

  chalf *bias_diff = bias_h.mutable_ocl_diff();

  shape[0] = sizeof(kernel_params) / sizeof(int);
  param_vals.Reshape(shape);

  int *ip_params = param_vals.mutable_cpu_data();

  for (int i = 0; i < shape[0]; ++i) {
    ip_params[i] = ((int *)params)[i];
  }
  
  const int* cr_params_b = param_vals.ocl_data();

  size_t outsize = sizeof(chalf) * top[0]->count();
  std::vector<cl_event> events;

  int events_size = 1;
  int g = 0;

  const chalf *top_diff;
  const int *relu_vals;
  for (int i = 0; i < bottom.size(); i++) {
    if (use_aux_) {
      top_diff = reinterpret_cast<const chalf *>(top[i]->cpu_diff());
      chalf *top_diff_aux = top_aux.mutable_cpu_diff();
      for (int j = 0; j < top_aux.shape(0); ++j)
        for (int k = 0; k < top_aux.shape(1); ++k) 
          if (j < top[i]->shape(0))
            top_diff_aux[j * top_aux.shape(1) + k] =
              top_diff[j * top[i]->shape(1) + k];
          else
            top_diff_aux[j * top_aux.shape(1) + k] = 0;
      outsize = top_aux.count();
      top_diff = top_aux.ocl_diff(outsize);
    } else {
      top_diff = reinterpret_cast<const chalf *>(top[i]->ocl_diff(outsize));
    }
    events.resize(events_size, 0);
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
    clSetKernelArg(this->ocl_kernel, 6, sizeof(cl_int),
        (const void *)&g);
    clEnqueueTask(oclCommandQueue, this->ocl_kernel, 0,
        NULL, &(events[g]));
    clWaitForEvents(events.size(), events.data());
  }
  bias_diff = bias_h.mutable_cpu_diff();
  Dtype *bias_diff_out = this->blobs_[1]->mutable_cpu_diff();
  for (int i = 0; i < bias_h.count() / 4; ++i) {
    for (int j = 0; j < 4; ++j)
      if (i + j * bias_h.count() / 4 < this->blobs_[1]->count())
        bias_diff_out[i + j * bias_h.count() / 4] =
          (Dtype)float(bias_diff[i * 4 + j]);
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
  chalf *weight_data_h_t = weights_h_t.mutable_cpu_data();

  for (int i = 0; i < weights_h_t.shape(1) / 4; ++i)
    for (int j = 0; j < inchannels; ++j)
      for (int k = 0; k < 4; ++k)
        if (i + k * weights_h_t.shape(1) / 4 < outchannels)
          weight_data_h_t[j * weights_h_t.shape(1) + i * 4 + k] =
            chalf((float)weight_data[(i + k * weights_h_t.shape(1) / 4) *
                inchannels + j]);
        else
          weight_data_h_t[j * weights_h_t.shape(1) + i * 4 + k] = 0;

  const chalf *weight_data_t = weights_h_t.ocl_data();

  vector<int> shape(1);
  shape[0] = this->K_;

  const chalf *bias_data = bias_placeholder.ocl_data();

  shape[0] = sizeof(kernel_params) / sizeof(int);
  param_vals.Reshape(shape);

  int *ip_params = param_vals.mutable_cpu_data();

  for (int i = 0; i < shape[0]; ++i) {
    ip_params[i] = ((int *)params)[i];
  }
  
  const int* cr_params_b = param_vals.ocl_data();

  size_t insize = sizeof(chalf) * bottom[0]->count();
  size_t outsize = sizeof(chalf) * top[0]->count();
  std::vector<cl_event> events;

  int events_size = 1;
  int g = 0;

  const chalf *top_diff;
  const int *relu_vals;
  chalf *bottom_diff;

  for (int i = 0; i < bottom.size(); i++) {
    if (use_aux_) {
      top_diff = reinterpret_cast<const chalf *>(top[i]->cpu_diff());
      chalf *top_diff_aux = top_aux.mutable_cpu_diff();
      for (int j = 0; j < top_aux.shape(0); ++j)
        for (int k = 0; k < top_aux.shape(1); ++k) 
          if (j < top[i]->shape(0))
            top_diff_aux[j * top_aux.shape(1) + k] =
              top_diff[j * top[i]->shape(1) + k];
          else
            top_diff_aux[j * top_aux.shape(1) + k] = 0;
      outsize = top_aux.count();
      top_diff = top_aux.ocl_diff(outsize);
    } else {
      top_diff = reinterpret_cast<const chalf *>(top[i]->ocl_diff(outsize));
    }
    events.resize(events_size, 0);
    bottom_diff =
      reinterpret_cast<chalf *>(bottom[i]->mutable_ocl_diff(insize));
    relu_vals = relu_indices.ocl_data();
    clSetKernelArg(this->ocl_kernel, 0, sizeof(cl_mem),
      (const void *)&top_diff);
    clSetKernelArg(this->ocl_kernel, 1, sizeof(cl_mem),
      (const void *)&weight_data_t);
    clSetKernelArg(this->ocl_kernel, 2, sizeof(cl_mem),
      (const void *)&bias_data);
    clSetKernelArg(this->ocl_kernel, 3, sizeof(cl_mem),
      (const void *)&bottom_diff);
    clSetKernelArg(this->ocl_kernel, 4, sizeof(cl_mem),
      (const void *)&relu_vals);
    clSetKernelArg(this->ocl_kernel, 5, sizeof(cl_mem),
      (const void *)&cr_params_b);
    clSetKernelArg(this->ocl_kernel, 6, sizeof(cl_int),
        (const void *)&g);
    clEnqueueTask(oclCommandQueue, this->ocl_kernel, 0,
        NULL, &(events[g]));
    clWaitForEvents(events.size(), events.data());
  }
}


template <typename Dtype>
void OCLHWCNInnerProductLayer<Dtype>::Backward_ocl(
    const vector<Blob<Dtype>*>& top, const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  if (this->param_propagate_down_[0])
    backward_weights(top, propagate_down, bottom);

  for (int i = 0; i < bottom[0]->count(); ++i)
    (bottom[0]->mutable_cpu_diff())[i] = 0;
  if (propagate_down[0])
    backward_data(top, propagate_down, bottom);

  if (this->bias_term_ && this->param_propagate_down_[1])
    backward_bias(top, propagate_down, bottom);
}


INSTANTIATE_CLASS(OCLHWCNInnerProductLayer);
REGISTER_LAYER_CLASS(OCLHWCNInnerProduct);
#endif
}  // namespace caffe
