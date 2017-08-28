#include <algorithm>
#include <vector>

#include "caffe/layers/cpfp_conversion_layer.hpp"

namespace caffe {

template <typename Dtype>
void CPFPConversionLayer<Dtype>::LayerSetUp(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  CPFPConversionParameter cpfp_param =
    this->layer_param_.cpfp_conversion_param();

  convert_to_ = cpfp_param.convert_to(); 
}

template <typename Dtype>
void CPFPConversionLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
   const vector<Blob<Dtype>*>& top) {
  bottom_shape_ = bottom[0]->shape();

  for (int top_id = 0; top_id < top.size(); ++top_id) {
    top[top_id]->Reshape(bottom[0]->shape());
  }
} 

template <typename Dtype>
void CPFPConversionLayer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  for (int i = 0; i < bottom.size(); ++i) {
    const int count = bottom[i]->count();
    if (convert_to_) {
      int outsize = sizeof(cpfp) * count;
      const Dtype *bottom_data = bottom[i]->cpu_data();
      cpfp *top_data =
        reinterpret_cast<cpfp *>(top[i]->mutable_cpu_data(outsize));
      for (int j = 0; j < count; ++j) {
        top_data[j] = cpfp((float)bottom_data[j]);
      }
    } else {
      int insize = sizeof(cpfp) * count;
      const cpfp *bottom_data =
        reinterpret_cast<const cpfp *>(bottom[i]->cpu_data(insize));
      Dtype *top_data = top[i]->mutable_cpu_data();
      for (int j = 0; j < count; ++j) {
        top_data[j] = (Dtype)(float(bottom_data[j]));
      }
    }
  }
}

template <typename Dtype>
void CPFPConversionLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  if (propagate_down[0]) {
    for (int i = 0; i < bottom.size(); ++i) {  
      const int count = bottom[i]->count();
      if (convert_to_) {
        int outsize = sizeof(cpfp) * count;
        Dtype *bottom_diff = bottom[i]->mutable_cpu_diff();
        const cpfp *top_diff =
          reinterpret_cast<const cpfp *>(top[i]->cpu_diff(outsize));
        for (int j = 0; j < count; ++j) {
          bottom_diff[j] = (Dtype)(float(top_diff[j]));
        }
      } else {
        int insize = sizeof(cpfp) * count;
        cpfp *bottom_diff =
          reinterpret_cast<cpfp *>(bottom[i]->mutable_cpu_diff(insize));
        const Dtype *top_diff = top[i]->cpu_diff();
        for (int j = 0; j < count; ++j) {
          bottom_diff[j] = cpfp((float)top_diff[j]);
        }
      }
    }
  }
}

INSTANTIATE_CLASS(CPFPConversionLayer);
REGISTER_LAYER_CLASS(CPFPConversion);

}  // namespace caffe
