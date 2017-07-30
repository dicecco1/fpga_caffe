#include <algorithm>
#include <vector>

#include "caffe/layers/half_conversion_layer.hpp"

namespace caffe {

template <typename Dtype>
void HalfConversionLayer<Dtype>::LayerSetUp(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  HalfConversionParameter half_param =
    this->layer_param_.half_conversion_param();

  convert_to_ = half_param.convert_to(); 
}

template <typename Dtype>
void HalfConversionLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
   const vector<Blob<Dtype>*>& top) {
  bottom_shape_ = bottom[0]->shape();

  for (int top_id = 0; top_id < top.size(); ++top_id) {
    top[top_id]->Reshape(bottom[0]->shape());
  }
} 

template <typename Dtype>
void HalfConversionLayer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  for (int i = 0; i < bottom.size(); ++i) {
    const int count = bottom[i]->count();
    if (convert_to_) {
      const Dtype *bottom_data = bottom[i]->cpu_data();
      chalf *top_data = reinterpret_cast<chalf *>(top[i]->mutable_cpu_data());
      for (int j = 0; j < count; ++j) {
        top_data[j] = chalf((float)bottom_data[j]);
      }
    } else {
      const chalf *bottom_data =
        reinterpret_cast<const chalf *>(bottom[i]->cpu_data());
      Dtype *top_data = top[i]->mutable_cpu_data();
      for (int j = 0; j < count; ++j) {
        top_data[j] = (Dtype)(float(bottom_data[j]));
      }
    }
  }
}

template <typename Dtype>
void HalfConversionLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  if (propagate_down[0]) {
    for (int i = 0; i < bottom.size(); ++i) {  
      const int count = bottom[i]->count();
      if (convert_to_) {
        Dtype *bottom_diff = bottom[i]->mutable_cpu_diff();
        const chalf *top_diff =
          reinterpret_cast<const chalf *>(top[i]->cpu_diff());
        for (int j = 0; j < count; ++j) {
          bottom_diff[j] = (Dtype)(float(top_diff[j]));
        }
      } else {
        chalf *bottom_diff =
          reinterpret_cast<chalf *>(bottom[i]->mutable_cpu_diff());
        const Dtype *top_diff = top[i]->cpu_diff();
        for (int j = 0; j < count; ++j) {
          bottom_diff[j] = chalf((float)top_diff[j]);
        }
      }
    }
  }
}

INSTANTIATE_CLASS(HalfConversionLayer);
REGISTER_LAYER_CLASS(HalfConversion);

}  // namespace caffe
