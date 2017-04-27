#include <vector>

#include "caffe/layers/ocl_inner_product_layer.hpp"

namespace caffe {

#ifdef USE_OCL

template <typename Dtype>
void OCLInnerProductLayer<Dtype>::LayerSetUp(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  InnerProductLayer<Dtype>::LayerSetUp(bottom, top);
  const int num_output = this->layer_param_.inner_product_param().num_output();

  CRParameter cr_param = this->layer_param_.cr_param();
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
}

template <typename Dtype>
void OCLInnerProductLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  InnerProductLayer<Dtype>::Reshape(bottom, top);
  vector<int> shape(2);

  shape[0] = (this->blobs_[0])->shape(0);
  shape[1] = (this->blobs_[0])->shape(1);
  weights_pad_h.Reshape(shape);

  shape[0] = (this->blobs_[0])->shape(1);
  shape[1] = (this->blobs_[0])->shape(0);
  weights_pad_h_t.Reshape(shape);

  shape[0] = 1;
  shape[1] = 1;
  if (this->bias_term_)
    bias_h.Reshape((this->blobs_[1])->shape());
  else
    bias_h.Reshape(shape);

  shape[0] = top[0]->shape(0);
  shape[1] = top[0]->shape(1);

  std::cout<<shape[0]<<" "<<shape[1]<<std::endl;

  relu_indices.Reshape(shape);
  top_data_h.Reshape(shape);
  std::cout<<"Done ocl reshape"<<std::endl;
}

template <typename Dtype>
void OCLInnerProductLayer<Dtype>::backward_data(const Dtype* output,
    const Dtype* weights, Dtype* input) {

}

template <typename Dtype>
void OCLInnerProductLayer<Dtype>::backward_weights(const Dtype* input,
    const Dtype* output, Dtype* weights) {

}

template <typename Dtype>
void OCLInnerProductLayer<Dtype>::backward_bias(Dtype* bias,
    const Dtype* input) {

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
    std::cout<<ip_params[i]<<std::endl;
  }
  std::cout<<batch_<<std::endl;
  
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
void OCLInnerProductLayer<Dtype>::Backward_ocl(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  Backward_cpu(top, propagate_down, bottom);
}


INSTANTIATE_CLASS(OCLInnerProductLayer);
REGISTER_LAYER_CLASS(OCLInnerProduct);
#endif
}  // namespace caffe
