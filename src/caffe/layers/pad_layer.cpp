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
  axis_ = pad_param.axis();

  if (pad_to_ == 0) {
    if (bottom_shape_[axis_] % pad_val_ != 0)
      dim_pad_ = (bottom_shape_[axis_] / pad_val_ + 1) * pad_val_;
    else
      dim_pad_ = bottom_shape_[axis_];
  } else {
    if (pad_)
      dim_pad_ = pad_to_;
    else
      dim_pad_ = bottom_shape_[axis_];
  }

  if (pad_) {
    dim_ = bottom_shape_[axis_];
  } else {
    if (pad_to_ == 0)
      dim_ = bottom_shape_[axis_ - 1];
    else
      dim_ = pad_to_;
  }
}

template <typename Dtype>
void PadLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
   const vector<Blob<Dtype>*>& top) {

  vector<int> top_shape = bottom_shape_;
  if (pad_)
    top_shape[axis_] = dim_pad_;
  else
    top_shape[axis_] = dim_;

  for (int top_id = 0; top_id < top.size(); ++top_id) {
    top[top_id]->Reshape(top_shape);
  }
} 

template <typename Dtype>
void PadLayer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {

  std::vector<int> shape = bottom[0]->shape();
  std::vector<int> top_shape = top[0]->shape();

  for (int i = 0; i < bottom.size(); ++i) {
    const Dtype *bottom_data = bottom[i]->cpu_data();
    Dtype *top_data = top[i]->mutable_cpu_data();
    for (int n = 0; n < shape[0]; ++n) {
      for (int c = 0; c < shape[1]; ++c) {
        for (int h = 0; h < shape[2]; ++h) {
          for (int w = 0; w < shape[3]; ++w) {
            int bot_idx, top_idx;
            bot_idx = ((n * shape[1] + c) * shape[2] + h) * shape[3] + w;
            top_idx = ((n * top_shape[1] + c) * top_shape[2] + h) *
              top_shape[3] + w;
            if (((axis_ == 0) && n < dim_) || ((axis_ == 1) && c < dim_) ||
                ((axis_ == 2) && h < dim_) || ((axis_ == 3) && w < dim_)) 
              top_data[top_idx] = bottom_data[bot_idx];
            else if (pad_)
              top_data[top_idx] = 0;
          }
        }
      }
    }
  }
}

template <typename Dtype>
void PadLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {

  std::vector<int> shape = bottom[0]->shape();
  std::vector<int> top_shape = top[0]->shape();
  if (propagate_down[0]) {
    for (int i = 0; i < bottom.size(); ++i) {
      Dtype *bottom_diff = bottom[i]->mutable_cpu_diff();
      const Dtype *top_diff = top[i]->cpu_diff();
      for (int n = 0; n < shape[0]; ++n) {
        for (int c = 0; c < shape[1]; ++c) {
          for (int h = 0; h < shape[2]; ++h) {
            for (int w = 0; w < shape[3]; ++w) {
              int bot_idx, top_idx;
              bot_idx = ((n * shape[1] + c) * shape[2] + h) * shape[3] + w;
              top_idx = ((n * top_shape[1] + c) * top_shape[2] + h) *
                top_shape[3] + w;
              if (((axis_ == 0) && n < dim_) || ((axis_ == 1) && c < dim_) ||
                  ((axis_ == 2) && h < dim_) || ((axis_ == 3) && w < dim_))
                bottom_diff[bot_idx] = top_diff[top_idx];
              else if (!pad_)
                bottom_diff[bot_idx] = 0;
            }
          }
        }
      }
    }
  }
}

INSTANTIATE_CLASS(PadLayer);
REGISTER_LAYER_CLASS(Pad);

}  // namespace caffe
