#include <algorithm>
#include <vector>

#include "caffe/layers/pad_layer.hpp"

namespace caffe {

template <typename Dtype>
void PadLayer<Dtype>::LayerSetUp(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  PadParameter pad_param =
    this->layer_param_.pad_param();

  pad_val_ = pad_param.pad_val();
  pad_ = pad_param.pad();
  bottom_shape_ = bottom[0]->shape();
  if (bottom_shape_[3] % pad_val_ != 0) {
    xdim_pad_ = (bottom_shape_[3] / pad_val_ + 1) * pad_val_;
  } else {
    xdim_pad_ = bottom_shape_[3];
  }
  if (pad_)
    xdim_ = bottom_shape_[3];
  else
    xdim_ = bottom_shape_[2];
}

template <typename Dtype>
void PadLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
   const vector<Blob<Dtype>*>& top) {

  vector<int> top_shape(bottom_shape_.size());

  top_shape[0] = bottom_shape_[0];
  top_shape[1] = bottom_shape_[1];
  top_shape[2] = bottom_shape_[2];
  if (pad_)
    top_shape[3] = xdim_pad_;
  else
    top_shape[3] = xdim_;

  for (int top_id = 0; top_id < top.size(); ++top_id) {
    top[top_id]->Reshape(top_shape);
  }
} 

template <typename Dtype>
void PadLayer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  int size = bottom_shape_[0] * bottom_shape_[1] * bottom_shape_[2];
  int x_iters;
  int bottom_off, top_off;
  std::cout<<"In forward pad"<<std::endl;
  if (pad_) {
    x_iters = xdim_pad_;
    top_off = xdim_pad_;
    bottom_off = xdim_;
  } else {
    x_iters = xdim_;
    top_off = xdim_;
    bottom_off = xdim_pad_;
  }
  for (int i = 0; i < bottom.size(); ++i) {
    const Dtype *bottom_data = bottom[i]->cpu_data();
    Dtype *top_data = top[i]->mutable_cpu_data();
    for (int j = 0; j < size; ++j) {
      for (int x = 0; x < x_iters; ++x) {
        if (x < xdim_) {
          top_data[j * top_off + x] = bottom_data[j * bottom_off + x];
        } else {
          top_data[j * top_off + x] = 0;
        }
      }
    }
  }
}

template <typename Dtype>
void PadLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  int size = bottom_shape_[0] * bottom_shape_[1] * bottom_shape_[2];
  int x_iters;
  int bottom_off, top_off;
  if (pad_) {
    x_iters = xdim_;
    top_off = xdim_pad_;
    bottom_off = xdim_;
  } else {
    x_iters = xdim_pad_;
    top_off = xdim_;
    bottom_off = xdim_pad_;
  }

  for (int i = 0; i < bottom.size(); ++i) {
    Dtype *bottom_diff = bottom[i]->mutable_cpu_diff();
    const Dtype *top_diff = top[i]->cpu_diff();
    for (int j = 0; j < size; ++j) {
      for (int x = 0; x < x_iters; ++x) {
        if (x < xdim_)
          bottom_diff[j * bottom_off + x] = top_diff[j * top_off + x];
        else
          bottom_diff[j * bottom_off + x] = 0;
      }
    }
  }
}

INSTANTIATE_CLASS(PadLayer);
REGISTER_LAYER_CLASS(Pad);

}  // namespace caffe
