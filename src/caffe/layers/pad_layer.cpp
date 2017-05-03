#include <algorithm>
#include <vector>

#include "caffe/layers/pad_layer.hpp"

namespace caffe {

template <typename Dtype>
void PadLayer<Dtype>::LayerSetUp(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  PadParameter pad_param =
    this->layer_param_.pad_param();

  pad_to_ = pad_param.pad_to();
  pad_val_ = pad_param.pad_val();
  pad_ = pad_param.pad();
  bottom_shape_ = bottom[0]->shape();
  int last_idx = bottom_shape_.size() - 1;

  if (pad_to_ == 0) {
    if (bottom_shape_[last_idx] % pad_val_ != 0) {
      xdim_pad_ = (bottom_shape_[last_idx] / pad_val_ + 1) * pad_val_;
    } else {
      xdim_pad_ = bottom_shape_[last_idx];
    }
  } else {
    if (pad_)
      xdim_pad_ = pad_to_;
    else
      xdim_pad_ = bottom_shape_[last_idx];
  }

  if (pad_) {
    xdim_ = bottom_shape_[last_idx];
  } else {
    if (pad_to_ == 0)
      xdim_ = bottom_shape_[last_idx - 1];
    else
      xdim_ = pad_to_;
  }
}

template <typename Dtype>
void PadLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
   const vector<Blob<Dtype>*>& top) {

  vector<int> top_shape = bottom_shape_;
  int last_idx = bottom_shape_.size() - 1;

  if (pad_)
    top_shape[last_idx] = xdim_pad_;
  else
    top_shape[last_idx] = xdim_;

  for (int top_id = 0; top_id < top.size(); ++top_id) {
    top[top_id]->Reshape(top_shape);
  }
} 

template <typename Dtype>
void PadLayer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  int x_iters;
  int bottom_off, top_off;
  if (pad_) {
    x_iters = xdim_pad_;
    top_off = xdim_pad_;
    bottom_off = xdim_;
  } else {
    x_iters = xdim_;
    top_off = xdim_;
    bottom_off = xdim_pad_;
  }

  int size = top[0]->count() / top_off;

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

  int size = top[0]->count() / top_off;

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
