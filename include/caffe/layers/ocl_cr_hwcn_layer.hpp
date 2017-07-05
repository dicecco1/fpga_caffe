#ifndef CAFFE_OCL_CR_LAYER_HPP_
#define CAFFE_OCL_CR_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

#include "caffe/layers/conv_layer.hpp"

namespace caffe {

#ifdef USE_OCL
/**
 * @brief Convolves the input image with a bank of learned filters,
 *        and (optionally) adds biases.
 *
 *   The input is H x W x C x N convolved with a filter that is K x K x O x C.
 *
 */
template <typename Dtype>
class OCLCRHWCNLayer : public ConvolutionLayer<Dtype> {
 public:
  /**
   * @param param provides ConvolutionParameter convolution_param,
   *    with ConvolutionLayer options:
   *  - num_output. The number of filters.
   *  - kernel_size / kernel_h / kernel_w. The filter dimensions, given by
   *  kernel_h and kernel_w have to be the same (i.e. square kernels only).
   *  - stride / stride_h / stride_w (\b optional, default 1). The filter
   *  stride, in the case of stride_h and stride_w have to be the same.
   *  - pad / pad_h / pad_w (\b optional, default 0). The zero-padding for
   *  convolution, given by pad for equal dimensions, pad_h and pad_w have to 
   *  be the same. 
   *  - group (\b optional, default 1). The number of filter groups. Group
   *  convolution is a method for reducing parameterization by selectively
   *  connecting input and output channels. The input and output channel dimensions must be divisible
   *  by the number of groups. For group @f$ \geq 1 @f$, the
   *  convolutional filters' input and output channels are separated s.t. each
   *  group takes 1 / group of the input channels and makes 1 / group of the
   *  output channels. Concretely 4 input channels, 8 output channels, and
   *  2 groups separate input channels 1-2 and output channels 1-4 into the
   *  first group and input channels 3-4 and output channels 5-8 into the second
   *  group.
   *  - bias_term (\b optional, default true). Whether to have a bias.
   *  - engine: convolution has CAFFE (matrix multiplication) and CUDNN (library
   *    kernels + stream parallelism) engines.
   */
  explicit OCLCRHWCNLayer(const LayerParameter& param)
      : ConvolutionLayer<Dtype>(param) {}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "Convolution"; }

 protected:
  virtual inline bool reverse_dimensions() { return false; }
  virtual void compute_output_shape();
  virtual void Forward_ocl(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Backward_ocl(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
  void backward_bias(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
  virtual void backward_data(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
  void backward_weights(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
  void copyToHalf(const Dtype *input, chalf *output, int size);
  void copyToHalfWeights(const Dtype *input, chalf *output,
      kernel_params params);
  void copyToFloatWeights(chalf *input, Dtype *output, const vector<int>,
      kernel_params params);
  void RotateWeightsHalf(const Dtype *input, chalf *output,
      kernel_params params);
 private:
  kernel_params ocl_params_;
  kernel_params ocl_params_bw_;
  kernel_params ocl_params_bb_;
  kernel_params ocl_params_bi_;
  Blob<int> relu_indices; 
  Blob<chalf> weights_h;
  Blob<chalf> weights_h_r;
  Blob<chalf> bias_h, bias_null;
  int conv_out_channels_;
  int conv_in_channels_;
  int conv_out_spatial_dim_;
};
#endif

}  // namespace caffe

#endif  // CAFFE_OCL_CR_LAYER_HPP_
