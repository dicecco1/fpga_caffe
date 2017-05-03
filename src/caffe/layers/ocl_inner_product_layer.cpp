#include <vector>

#include "caffe/layers/ocl_inner_product_layer.hpp"

namespace caffe {

#ifdef USE_OCL

template <typename Dtype>
void OCLInnerProductLayer<Dtype>::LayerSetUp(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  InnerProductLayer<Dtype>::LayerSetUp(bottom, top);
  const int num_output = this->layer_param_.inner_product_param().num_output();

  pad_oc_ = (this->blobs_[0])->shape(0);

  if (pad_oc_ % 16 != 0)
    pad_oc_ = (pad_oc_ / 16 + 1) * 16;

  CRParameter cr_param = this->layer_param_.cr_param();
  swap_inputs_ = cr_param.swap_inputs();
  kernel_params *params = &ocl_params_;
  params->inchannels = 1;
  params->xdim = this->K_;
  params->ydim = 1;
  params->ksize = 1;
  params->rpo = 1;
  params->rpofm = 1;
  params->burstydim = 1;
  params->xtile_pad = params->xdim / 2;
  params->burstchannels = 1;
  params->outchannels = num_output;

  int num_ = bottom[0]->shape(0);

  int numimages_ = 8 * 256 * 256 / params->xdim;

  if (numimages_ > num_) {
    numimages_ = num_;
  } else {
    int tnumimages = numimages_;
    while (num_ % tnumimages != 0)
      tnumimages--;
    numimages_ = tnumimages;
  }

  params->numimages = numimages_;
  params->numgroups = 1;
  params->fc = 1;
  params->relu = cr_param.relu();
  batch_ = num_ / params->numimages;

// Set params for backward w.r.t. weights

  params = &ocl_params_bw_;
  params->inchannels = 1;
  params->burstchannels = 1;
  params->ydim = 1;
  params->ksize = 1;
  params->rpo = 1;
  params->rpofm = 1;
  params->burstydim = 1;

  if (cr_param.swap_inputs()) {
    params->xdim = this->K_;
    params->outchannels = num_output;
    params->relu = 0;
  } else {
    params->xdim = num_output;
    params->outchannels = this->K_;
    params->relu = cr_param.relu();
  }
  params->xtile_pad = params->xdim / 2;

  num_ = bottom[0]->shape(0);

  numimages_ = 8 * 256 * 256 / params->xdim;

  if (numimages_ > num_) {
    numimages_ = num_;
  } else {
    int tnumimages = numimages_;
    while (num_ % tnumimages != 0)
      tnumimages--;
    numimages_ = tnumimages;
  }

  params->numimages = numimages_;
  params->numgroups = 1;
  params->fc = 1;
  params->backward = 1;
  batch_bw_ = num_ / params->numimages;

// Set params for backward w.r.t. to bias'

  params = &ocl_params_bb_;
  params->xdim = pad_oc_;
  params->ydim = 1;
  params->ksize = 1;
  params->rpo = 1;
  params->rpofm = 1;
  params->burstydim = 1;
  params->xtile_pad = params->xdim / 2;
  params->outchannels = 1;

  num_ = bottom[0]->shape(0);

  numimages_ = 8 * 256 * 256 / params->xdim;

  if (numimages_ > num_) {
    numimages_ = num_;
  } else {
    int tnumimages = numimages_;
    while (num_ % tnumimages != 0)
      tnumimages--;
    numimages_ = tnumimages;
  }

  params->inchannels = numimages_;
  params->burstchannels = numimages_;
  params->numimages = 1;
  params->numgroups = 1;
  params->fc = 0;
  params->backward = 2;
  params->relu = cr_param.relu();
  batch_bb_ = num_ / numimages_;

// Set params for backward w.r.t. data

  params = &ocl_params_bi_;
  params->inchannels = 1;
  params->xdim = pad_oc_;
  params->ydim = 1;
  params->ksize = 1;
  params->rpo = 1;
  params->rpofm = 1;
  params->burstydim = 1;
  params->xtile_pad = params->xdim / 2;
  params->burstchannels = 1;
  params->outchannels = this->K_;

  num_ = bottom[0]->shape(0);

  numimages_ = 8 * 256 * 256 / params->xdim;

  if (numimages_ > num_) {
    numimages_ = num_;
  } else {
    int tnumimages = numimages_;
    while (num_ % tnumimages != 0)
      tnumimages--;
    numimages_ = tnumimages;
  }

  params->numimages = numimages_;
  params->numgroups = 1;
  params->fc = 1;
  params->backward = 0;
  params->relu = cr_param.relu();
  batch_bi_ = num_ / params->numimages;
}

template <typename Dtype>
void OCLInnerProductLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  InnerProductLayer<Dtype>::Reshape(bottom, top);
  vector<int> shape(2);

  shape[0] = (this->blobs_[0])->shape(0);
  shape[1] = (this->blobs_[0])->shape(1);
  weights_pad_h.Reshape(shape);

  shape[0] = pad_oc_;
  shape[1] = (this->blobs_[0])->shape(1);
  weights_pad_h_t.Reshape(shape);

  shape[0] = 1;
  shape[1] = this->K_;
  if (this->bias_term_)
    bias_h.Reshape((this->blobs_[1])->shape());
  else
    bias_h.Reshape(shape);

  shape[0] = top[0]->shape(0);
  shape[1] = pad_oc_;

  relu_indices.Reshape(shape);
  top_data_h.Reshape(shape);
  bottom_data_h.Reshape(bottom[0]->shape());
}

template <typename Dtype>
void OCLInnerProductLayer<Dtype>::copyPad(const chalf *input, chalf *output,
    int size, int xdim, int xdim_pad) {
  for (int i = 0; i < size; ++i)
    for (int j = 0; j < xdim_pad; ++j)
      if (j < xdim)
        output[i * xdim_pad + j] = input[i * xdim + j];
      else
        output[i * xdim_pad + j] = chalf(0);
}

template <typename Dtype>
void OCLInnerProductLayer<Dtype>::copyToHalf(const Dtype *input, chalf *output,
    int size, int xdim, int xdim_pad) {
  for (int i = 0; i < size; ++i)
    for (int j = 0; j < xdim_pad; ++j)
      if (j < xdim)
        output[i * xdim_pad + j] = chalf((float)input[i * xdim + j]);
      else
        output[i * xdim_pad + j] = chalf(0);
}

template <typename Dtype>
void OCLInnerProductLayer<Dtype>::copyToFloat(const chalf *input,
    Dtype *output, int size, int ksize, int ksize_pad) {
  return;
}

template <typename Dtype>
void OCLInnerProductLayer<Dtype>::Forward_ocl(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  kernel_params *params = &ocl_params_;
  copyToHalf(this->blobs_[0]->mutable_cpu_data(),
      weights_pad_h.mutable_cpu_data(), weights_pad_h.count(), 1, 1);
  if (this->bias_term_)
    copyToHalf(this->blobs_[1]->mutable_cpu_data(), bias_h.mutable_cpu_data(),
        params->outchannels, 1, 1);
  else
    (bias_h.mutable_cpu_data())[0] = chalf(0);

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

  int *ip_params = param_vals->mutable_cpu_data();

  for (int i = 0; i < shape[3]; ++i) {
    ip_params[i] = ((int *)params)[i];
  }
  
  const int* k_params = param_vals->ocl_data();

  size_t insize = sizeof(chalf) * bottom[0]->count();
  std::vector<cl_event> events;

  int events_size = batch_ * numgroups_;

  chalf *top_data;
  chalf *top_data_t;
  char *relu_vals;
  for (int i = 0; i < bottom.size(); i++) {
    events.resize(events_size, 0);
    const chalf *bottom_data =
      reinterpret_cast<const chalf *>(bottom[i]->ocl_data(insize));
    top_data = top_data_h.mutable_ocl_data(0);
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
    top_data = top_data_h.mutable_cpu_data();
    top_data_t = reinterpret_cast<chalf *>(top[i]->mutable_cpu_data());

    for (int j = 0; j < this->M_; ++j)
      for (int k = 0; k < this->N_; ++k)
        top_data_t[j * this->N_ + k] = top_data[k * this->M_ + j];
  }
}

template <typename Dtype>
void OCLInnerProductLayer<Dtype>::backward_weights(
    const vector<Blob<Dtype>*>& top, const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  kernel_params *params = &ocl_params_bw_;

  Dtype* weight_diff_dtype = this->blobs_[0]->mutable_cpu_diff();
 
  chalf *weight_diff = weights_pad_h.mutable_cpu_diff();

  for (int i = 0; i < weights_pad_h.count(); ++i)
    weight_diff[i] = chalf(0);

  weight_diff = weights_pad_h.mutable_ocl_diff();
  vector<int> shape(4);
  shape[0] = 1;
  shape[1] = 1;
  shape[2] = 1;
  shape[3] = this->blobs_[0]->shape(1);
  Blob<chalf>* bias;
  if (swap_inputs_)
    bias = new Blob<chalf>(shape);
  else
    bias = new Blob<chalf>(this->blobs_[1]->shape());

  for (int i = 0; i < bias->count(); ++i)
    (bias->mutable_cpu_data())[i] = chalf(0);

  const chalf *bias_data = bias->ocl_data();

  int numgroups_ = params->numgroups;

  shape[0] = 1;
  shape[1] = 1;
  shape[2] = 1;
  shape[3] = sizeof(kernel_params) / sizeof(int);
  Blob<int> *param_vals = new Blob<int>(shape);

  int *ip_params = param_vals->mutable_cpu_data();

  for (int i = 0; i < shape[3]; ++i) {
    ip_params[i] = ((int *)params)[i];
  }
  
  const int* cr_params_b = param_vals->ocl_data();

  size_t insize = sizeof(chalf) * bottom[0]->count();
  size_t outsize = sizeof(chalf) * top[0]->count();
  std::vector<cl_event> events;

  int events_size = batch_bw_ * numgroups_;

  const chalf *top_diff;
  const char *relu_vals;
  const chalf *bottom_data;
  bottom_data = reinterpret_cast<const chalf *>(bottom[0]->cpu_data());
  top_diff = reinterpret_cast<const chalf *>(top[0]->cpu_diff());

  chalf *bottom_data_t = bottom_data_h.mutable_cpu_data();
  chalf *top_diff_t = top_data_h.mutable_cpu_diff();

  int num_images = bottom[0]->shape(0);
  int outchannels = this->blobs_[0]->shape(0);
  int inchannels = this->blobs_[0]->shape(1);

  for (int i = 0; i < num_images; ++i)
    for (int j = 0; j < inchannels; ++j)
      bottom_data_t[j * num_images + i] = bottom_data[i * inchannels + j];

  for (int i = 0; i < num_images; ++i)
    for (int j = 0; j < outchannels; ++j)
      top_diff_t[j * num_images + i] = top_diff[i * outchannels + j];

  for (int i = 0; i < bottom.size(); i++) {
    events.resize(events_size, 0);
    bottom_data = bottom_data_h.ocl_data();
    top_diff = top_data_h.ocl_diff();
    relu_vals = relu_indices.ocl_data();

    if (swap_inputs_) {
      clSetKernelArg(this->ocl_kernel, 0, sizeof(cl_mem),
        (const void *)&bottom_data);
      clSetKernelArg(this->ocl_kernel, 1, sizeof(cl_mem),
        (const void *)&top_diff);
    } else {
      clSetKernelArg(this->ocl_kernel, 0, sizeof(cl_mem),
        (const void *)&top_diff);
      clSetKernelArg(this->ocl_kernel, 1, sizeof(cl_mem),
        (const void *)&bottom_data);
    }
    clSetKernelArg(this->ocl_kernel, 2, sizeof(cl_mem),
      (const void *)&bias_data);
    clSetKernelArg(this->ocl_kernel, 3, sizeof(cl_mem),
      (const void *)&weight_diff);
    clSetKernelArg(this->ocl_kernel, 4, sizeof(cl_mem),
      (const void *)&relu_vals);
    clSetKernelArg(this->ocl_kernel, 5, sizeof(cl_mem),
      (const void *)&cr_params_b);

    for (int n = 0; n < batch_bw_; ++n) {
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

  for (int i = 0; i < outchannels; ++i)
    for (int j = 0; j < inchannels; ++j)
      if (swap_inputs_) 
        weight_diff_dtype[i * inchannels + j] =
          (Dtype)float(weight_diff[i * inchannels + j]);
      else
        weight_diff_dtype[i * inchannels + j] =
          (Dtype)float(weight_diff[j * outchannels + i]);

  delete bias;
  delete param_vals;
}

template <typename Dtype>
void OCLInnerProductLayer<Dtype>::backward_bias(
    const vector<Blob<Dtype>*>& top, const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  kernel_params *params = &ocl_params_bb_;

  vector<int> shape(4);
  shape[0] = 1;
  shape[1] = 1;
  shape[2] = params->inchannels;
  shape[3] = 16;

  Blob<chalf>* weights = new Blob<chalf>(shape);

  for (int i = 0; i < weights->count(); ++i)
    (weights->mutable_cpu_data())[i] = chalf((float)1.0);

  const chalf *weights_data = weights->ocl_data();

  shape[0] = 1;
  shape[1] = 1;
  shape[2] = 1;
  shape[3] = pad_oc_;

  Blob<chalf>* bias = new Blob<chalf>(shape);

  for (int i = 0; i < bias->count(); ++i)
    (bias->mutable_cpu_data())[i] = chalf(0);

  chalf *bias_diff = bias->mutable_ocl_data();

  int numgroups_ = params->numgroups;

  shape[0] = 1;
  shape[1] = 1;
  shape[2] = 1;
  shape[3] = sizeof(kernel_params) / sizeof(int);
  Blob<int> *param_vals = new Blob<int>(shape);

  int *ip_params = param_vals->mutable_cpu_data();

  for (int i = 0; i < shape[3]; ++i) {
    ip_params[i] = ((int *)params)[i];
  }
  
  const int* cr_params_b = param_vals->ocl_data();

  size_t outsize = sizeof(chalf) * top[0]->count();
  std::vector<cl_event> events;

  int events_size = batch_bb_ * numgroups_;

  const chalf *top_diff;
  const char *relu_vals;
  for (int i = 0; i < bottom.size(); i++) {
    events.resize(events_size, 0);
    top_diff = top_data_h.ocl_diff();
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

    for (int n = 0; n < batch_bb_; ++n) {
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
    bias_diff_out[i] = (Dtype)float(bias_diff[i]);
  }
}

template <typename Dtype>
void OCLInnerProductLayer<Dtype>::backward_data(
    const vector<Blob<Dtype>*>& top, const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  kernel_params *params = &ocl_params_bi_;

  int outchannels = this->blobs_[0]->shape(0);
  int inchannels = this->blobs_[0]->shape(1);

  const Dtype *weight_data = this->blobs_[0]->cpu_data();
  chalf *weight_data_h_t = weights_pad_h_t.mutable_cpu_data();

  for (int i = 0; i < pad_oc_; ++i)
    for (int j = 0; j < inchannels; ++j)
      if (i < outchannels)
        weight_data_h_t[j * pad_oc_ + i] =
          chalf((float)weight_data[i * inchannels + j]);

  const chalf *weight_data_t = weights_pad_h_t.ocl_data();

  vector<int> shape(4);
  shape[0] = 1;
  shape[1] = 1;
  shape[2] = 1;
  shape[3] = bottom[0]->shape(1);

  Blob<chalf>* bias = new Blob<chalf>(shape);

  for (int i = 0; i < bias->count(); ++i)
    (bias->mutable_cpu_data())[i] = chalf(0);

  const chalf *bias_data = bias->ocl_data();

  int numgroups_ = params->numgroups;

  shape[0] = 1;
  shape[1] = 1;
  shape[2] = 1;
  shape[3] = sizeof(kernel_params) / sizeof(int);
  Blob<int> *param_vals = new Blob<int>(shape);

  int *ip_params = param_vals->mutable_cpu_data();

  for (int i = 0; i < shape[3]; ++i) {
    ip_params[i] = ((int *)params)[i];
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
    top_diff = reinterpret_cast<const chalf *>(top[i]->cpu_diff());
    copyPad(top_diff, top_data_h.mutable_cpu_diff(), top_data_h.shape(0),
        top[i]->shape(1), top_data_h.shape(1));

    events.resize(events_size, 0);
    bottom_diff = bottom_data_h.mutable_ocl_diff();
    top_diff = top_data_h.ocl_diff();
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
    bottom_diff = reinterpret_cast<chalf *>(bottom[i]->mutable_cpu_diff());

    chalf *bottom_diff_h = bottom_data_h.mutable_cpu_diff();
    chalf *bottom_diff_t =
      reinterpret_cast<chalf *>(bottom[i]->mutable_cpu_diff());

    for (int j = 0; j < this->M_; ++j)
      for (int k = 0; k < this->K_; ++k)
        bottom_diff_t[j * this->K_ + k] = bottom_diff_h[k * this->M_ + j];
  }
}


template <typename Dtype>
void OCLInnerProductLayer<Dtype>::Backward_ocl(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {  
  if (this->param_propagate_down_[0])
    backward_weights(top, propagate_down, bottom);

  for (int i = 0; i < bottom[0]->count(); ++i)
    (bottom[0]->mutable_cpu_diff())[i] = 0;

  if (propagate_down[0])
    backward_data(top, propagate_down, bottom);

  if (this->bias_term_ && this->param_propagate_down_[1])
    backward_bias(top, propagate_down, bottom);

}


INSTANTIATE_CLASS(OCLInnerProductLayer);
REGISTER_LAYER_CLASS(OCLInnerProduct);
#endif
}  // namespace caffe
