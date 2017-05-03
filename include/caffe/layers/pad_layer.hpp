#ifndef CAFFE_PAD_LAYER_HPP_
#define CAFFE_PAD_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe {

/**
 * @brief Pads the input blob on the x-dimension to a multiple of pad_val or
 * unpads the input blob such that the x-dimension is the same as the
 * y-dimension.
 */
template <typename Dtype>
class PadLayer : public Layer<Dtype> {
 public:
  /**
   * @param   
   */
  explicit PadLayer(const LayerParameter& param)
      : Layer<Dtype>(param) {}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "Pad"; }

  virtual inline int MinBottomBlobs() const { return 1; }
  virtual inline int MinTopBlobs() const { return 1; }
  virtual inline bool EqualNumBottomTopBlobs() const { return true; }

 protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
  vector<int> bottom_shape_;
  bool pad_;
  int pad_to_;
  int pad_val_;
  int xdim_;
  int xdim_pad_;
};

}  // namespace caffe

#endif  // CAFFE_PAD_LAYER_HPP_
