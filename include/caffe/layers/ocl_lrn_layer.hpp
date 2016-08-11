#ifndef CAFFE_OCL_LRN_LAYER_HPP_
#define CAFFE_OCL_LRN_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

#include "caffe/layers/lrn_layer.hpp"

namespace caffe {

#ifdef USE_OCL
/**
 * @brief Normalize the input in a local region across or within feature maps.
 *
 * TODO(dox): thorough documentation for Forward, Backward, and proto params.
 */
template <typename Dtype>
class OCLLRNLayer : public LRNLayer<Dtype> {
 public:
  explicit OCLLRNLayer(const LayerParameter& param)
      : LRNLayer<Dtype>(param) {}

  virtual inline const char* type() const { return "LRN"; }
  virtual inline int ExactNumBottomBlobs() const { return 1; }
  virtual inline int ExactNumTopBlobs() const { return 1; }

 protected:
  virtual void Forward_ocl(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Call_ocl(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  
};
#endif

}  // namespace caffe

#endif  // CAFFE_OCL_LRN_LAYER_HPP_
