#ifndef CAFFE_OCL_POOLING_HWCN_LAYER_HPP_
#define CAFFE_OCL_POOLING_HWCN_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

#include "caffe/layers/pooling_layer.hpp"

namespace caffe {

#ifdef USE_OCL
template <typename Dtype>
class OCLPoolingHWCNLayer : public PoolingLayer<Dtype> {
 public:
  explicit OCLPoolingHWCNLayer(const LayerParameter& param)
      : PoolingLayer<Dtype>(param) {}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "Pooling"; }

 protected:
  virtual inline bool reverse_dimensions() { return false; }
  virtual void Forward_ocl(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Backward_ocl(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
 private:
  kernel_params ocl_params_;
  kernel_params ocl_params_bi_;
  Blob<int> relu_indices; 
  Blob<chalf> weights_placeholder;
  Blob<chalf> bias_placeholder;
  Blob<int> param_vals;
  int num_;
};
#endif

}  // namespace caffe

#endif  // CAFFE_OCL_POOLING_HWCN_LAYER_HPP_
