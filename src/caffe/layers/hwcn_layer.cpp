#include <algorithm>
#include <vector>

#include "caffe/layers/hwcn_layer.hpp"

namespace caffe {

template <typename Dtype>
void HWCNLayer<Dtype>::LayerSetUp(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  HWCNParameter hwcn_param =
    this->layer_param_.hwcn_param();

  convert_to_ = hwcn_param.convert_to();
  bottom_shape_ = bottom[0]->shape();
}

template <typename Dtype>
void HWCNLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
   const vector<Blob<Dtype>*>& top) {

  vector<int> top_shape(bottom_shape_.size());

  if (bottom_shape_.size() == 2) {
    for (int i = 0; i < bottom_shape_.size(); ++i)
      top_shape[i] = bottom_shape_[bottom_shape_.size() - 1 - i];
  } else {
    if (convert_to_) {
      top_shape[0] = bottom_shape_[2];
      top_shape[1] = bottom_shape_[3];
      top_shape[2] = bottom_shape_[1];
      top_shape[3] = bottom_shape_[0];
    } else {
      top_shape[0] = bottom_shape_[3];
      top_shape[1] = bottom_shape_[2];
      top_shape[2] = bottom_shape_[0];
      top_shape[3] = bottom_shape_[1];
    }
  }

  for (int top_id = 0; top_id < top.size(); ++top_id) {
    top[top_id]->Reshape(top_shape);
  }
} 

template <typename Dtype>
void HWCNLayer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  for (int i = 0; i < bottom.size(); ++i) {
    const Dtype* bottom_data = bottom[i]->cpu_data();
    Dtype* top_data = top[i]->mutable_cpu_data();
    std::vector<int> shape = bottom_shape_;
    if (convert_to_) {
      if (shape.size() == 2) {
        shape.push_back(1);
        shape.push_back(1);
      }
      for (int n = 0; n < shape[0]; ++n) {
        for (int c = 0; c < shape[1]; ++c) {
          for (int h = 0; h < shape[2]; ++h) {
            for (int w = 0; w < shape[3]; ++w) {
              int bot_idx = ((n * shape[1] + c) * shape[2] + h) * shape[3] + w;
              int top_idx = ((h * shape[3] + w) * shape[1] + c) * shape[0] + n;
              top_data[top_idx] = bottom_data[bot_idx];
            }
          }
        }
      }
    } else {
      if (shape.size() == 2) {
        shape.insert(shape.begin(), 1);
        shape.insert(shape.begin(), 1);
      }
      for (int n = 0; n < shape[3]; ++n) {
        for (int c = 0; c < shape[2]; ++c) {
          for (int h = 0; h < shape[0]; ++h) {
            for (int w = 0; w < shape[1]; ++w) {
              int bot_idx = ((h * shape[1] + w) * shape[2] + c) * shape[3] + n;
              int top_idx = ((n * shape[2] + c) * shape[0] + h) * shape[1] + w;
              top_data[top_idx] = bottom_data[bot_idx];
            }
          }
        }
      }     
    }
  }
}

template <typename Dtype>
void HWCNLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  if (propagate_down[0]) {
    for (int i = 0; i < bottom.size(); ++i) {
      Dtype* bottom_diff = bottom[i]->mutable_cpu_diff();
      const Dtype* top_diff = top[i]->cpu_diff();
      std::vector<int> shape = bottom_shape_;
      if (convert_to_) {
        if (shape.size() == 2) {
          shape.push_back(1);
          shape.push_back(1);
        }
        for (int n = 0; n < shape[0]; ++n) {
          for (int c = 0; c < shape[1]; ++c) {
            for (int h = 0; h < shape[2]; ++h) {
              for (int w = 0; w < shape[3]; ++w) {
                int bot_idx = ((n * shape[1] + c) * shape[2] + h) * shape[3]
                  + w;
                int top_idx = ((h * shape[3] + w) * shape[1] + c) * shape[0]
                  + n;
                bottom_diff[bot_idx] = top_diff[top_idx];
              }
            }
          }
        }
      } else {
        if (shape.size() == 2) {
          shape.insert(shape.begin(), 1);
          shape.insert(shape.begin(), 1);
        }
        for (int n = 0; n < shape[3]; ++n) {
          for (int c = 0; c < shape[2]; ++c) {
            for (int h = 0; h < shape[0]; ++h) {
              for (int w = 0; w < shape[1]; ++w) {
                int bot_idx = ((h * shape[1] + w) * shape[2] + c) * shape[3]
                  + n;
                int top_idx = ((n * shape[2] + c) * shape[0] + h) * shape[1]
                  + w;
                bottom_diff[bot_idx] = top_diff[top_idx];
              }
            }
          }
        }     
      }
    }
  }
}

INSTANTIATE_CLASS(HWCNLayer);
REGISTER_LAYER_CLASS(HWCN);

}  // namespace caffe
