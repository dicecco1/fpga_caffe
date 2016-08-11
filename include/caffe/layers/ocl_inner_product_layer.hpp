#ifndef CAFFE_OCL_INNER_PRODUCT_LAYER_HPP_
#define CAFFE_OCL_INNER_PRODUCT_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

#include "caffe/layers/inner_product_layer.hpp"

namespace caffe {

#ifdef USE_OCL
/**
 * @brief Also known as a "fully-connected" layer, computes an inner product
 *        with a set of learned weights, and (optionally) adds biases.
 *
 * TODO(dox): thorough documentation for Forward, Backward, and proto params.
 */
template <typename Dtype>
class OCLInnerProductLayer : public InnerProductLayer<Dtype> {
 public:
  explicit OCLInnerProductLayer(const LayerParameter& param)
      : InnerProductLayer<Dtype>(param) {}

  virtual inline const char* type() const { return "InnerProduct"; }
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

#endif  // CAFFE_OCL_INNER_PRODUCT_LAYER_HPP_
