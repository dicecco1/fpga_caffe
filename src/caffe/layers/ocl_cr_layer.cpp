#include <vector>

#include "caffe/layers/ocl_cr_layer.hpp"

namespace caffe {

#ifdef USE_OCL

template <typename Dtype>
void OCLCRLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
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
  params->relu = 1;
  batch_ = num_ / params->numimages;
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
          if (k < 2)
            output[out_idx + 16 + j * 3 + k] =
              chalf((float)input[in_idx + j * 5 + 3 + k]);
          else
            output[out_idx + 16 + j * 3 + k] = 0;
        }
      }
    }
  }       
}

template <typename Dtype>
void OCLCRLayer<Dtype>::copyToFloatWeights(const chalf *input,
    Dtype *output, int size, int ksize, int ksize_pad) {
  return;
}

template <typename Dtype>
void OCLCRLayer<Dtype>::Forward_ocl(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  if (this->layer_param_.ocl_enable()) {
    kernel_params *params = &ocl_params_;
    int wsize = params->outchannels * params->numgroups * params->inchannels;
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

    const int* cr_params = param_vals->ocl_data();

    size_t insize = sizeof(chalf) * bottom[0]->count();
    size_t outsize = sizeof(chalf) * top[0]->count();

    chalf *top_data;
    char *relu_vals;
    for (int i = 0; i < bottom.size(); i++) {
      const chalf *bottom_data =
        reinterpret_cast<const chalf *>(bottom[i]->ocl_data(insize));
      top_data = reinterpret_cast<chalf *>(top[i]->mutable_ocl_data(0,
            outsize));
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
      clWaitForEvents(batch_ * numgroups_, events.data());
      top[i]->cpu_data(outsize);
    }
  } else {
    Forward_cpu(bottom, top);
  }
}

template <typename Dtype>
void OCLCRLayer<Dtype>::Backward_ocl(const vector<Blob<Dtype>*>& top,
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


INSTANTIATE_CLASS(OCLCRLayer);
REGISTER_LAYER_CLASS(OCLCR);
#endif
}  // namespace caffe
