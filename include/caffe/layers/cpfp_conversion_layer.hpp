#ifndef CAFFE_CPFP_CONVERSION_LAYER_HPP_
#define CAFFE_CPFP_CONVERSION_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe {

/**
 * @brief Converts the input blob from Dtype to half precision, but maintains
 * the shape and types.
 * 
 */
template <typename Dtype>
class CPFPConversionLayer : public Layer<Dtype> {
 public:
  /**
   * @param   
   */
  explicit CPFPConversionLayer(const LayerParameter& param)
      : Layer<Dtype>(param) {}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "CPFP Conversion"; }

  virtual inline int MinBottomBlobs() const { return 1; }
  virtual inline int MinTopBlobs() const { return 1; }
  virtual inline bool EqualNumBottomTopBlobs() const { return true; }

 protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
  vector<int> bottom_shape_;
  bool convert_to_;
};

}  // namespace caffe

#endif  // CAFFE_CPFP_CONVERSION_LAYER_HPP_
