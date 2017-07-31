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
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "InnerProduct"; }
  virtual inline int ExactNumBottomBlobs() const { return 1; }
  virtual inline int ExactNumTopBlobs() const { return 1; }

 protected:
  virtual void Forward_ocl(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Backward_ocl(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
  virtual void backward_bias(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
  virtual void backward_data(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
  virtual void backward_weights(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
  void copyToHalf(const Dtype *input, cpfp *output, int size, int xdim,
      int xdim_pad);
  void copyPad(const cpfp *input, cpfp *output, int size, int xdim,
      int xdim_pad);
  void copyToFloat(const cpfp *input, Dtype *output, int size, int xdim,
      int xdim_pad);

 private:
  kernel_params ocl_params_;
  kernel_params ocl_params_bw_;
  kernel_params ocl_params_bb_;
  kernel_params ocl_params_bi_;
  int batch_;
  int batch_bw_;
  int batch_bb_;
  int batch_bi_;
  int pad_oc_;
  bool swap_inputs_;
  Blob<char> relu_indices;
  Blob<cpfp> top_data_h;
  Blob<cpfp> bottom_data_h;
  Blob<cpfp> weights_pad_h;
  Blob<cpfp> weights_pad_h_t;
  Blob<cpfp> bias_h;
};
#endif


}  // namespace caffe

#endif  // CAFFE_OCL_INNER_PRODUCT_LAYER_HPP_
