#include <vector>

#include "gtest/gtest.h"

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/layers/hwcn_layer.hpp"

#include "caffe/test/test_caffe_main.hpp"

namespace caffe {

template <typename TypeParam>
class HWCNLayerTest : public MultiDeviceTest<TypeParam> {
  typedef typename TypeParam::Dtype Dtype;

 protected:
  HWCNLayerTest()
      : blob_bottom_(new Blob<Dtype>(2, 3, 6, 4)),
        blob_bottom_2_(new Blob<Dtype>(2, 3, 6, 16)),
        blob_top_(new Blob<Dtype>()),
        blob_top_2_(new Blob<Dtype>()) {}
  virtual void SetUp() {
    // fill the values
    FillerParameter filler_param;
    filler_param.set_value(1.);
    GaussianFiller<Dtype> filler(filler_param);
    filler.Fill(this->blob_bottom_);
    filler.Fill(this->blob_bottom_2_);
    blob_bottom_vec_.push_back(blob_bottom_);
    blob_top_vec_.push_back(blob_top_);
  }

  virtual ~HWCNLayerTest() {
    delete blob_bottom_;
    delete blob_bottom_2_;
    delete blob_top_;
    delete blob_top_2_;
  }

  Blob<Dtype>* const blob_bottom_;
  Blob<Dtype>* const blob_bottom_2_;
  Blob<Dtype>* const blob_top_;
  Blob<Dtype>* const blob_top_2_;
  vector<Blob<Dtype>*> blob_bottom_vec_;
  vector<Blob<Dtype>*> blob_top_vec_;
};

TYPED_TEST_CASE(HWCNLayerTest, TestDtypesAndDevices);

TYPED_TEST(HWCNLayerTest, TestForwardHWCNConvertTo) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  HWCNParameter* hwcn_param =
      layer_param.mutable_hwcn_param();
  hwcn_param->set_convert_to(true);
  shared_ptr<Layer<Dtype> > layer(
      new HWCNLayer<Dtype>(layer_param));
  layer->SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  layer->Forward(this->blob_bottom_vec_, this->blob_top_vec_);


  const Dtype *bottom_data = this->blob_bottom_->cpu_data();
  const Dtype *top_data = this->blob_top_->cpu_data();
  
  std::vector<int> shape = this->blob_bottom_->shape();
  for (int n = 0; n < shape[0]; ++n) {
    for (int c = 0; c < shape[1]; ++c) {
      for (int h = 0; h < shape[2]; ++h) {
        for (int w = 0; w < shape[3]; ++w) {
          int bot_idx = ((n * shape[1] + c) * shape[2] + h) * shape[3] + w;
          int top_idx = ((h * shape[3] + w) * shape[1] + c) * shape[0] + n;
          EXPECT_EQ(bottom_data[bot_idx], top_data[top_idx]);
        }
      }
    }
  }
}

TYPED_TEST(HWCNLayerTest, TestForwardHWCNConvertBack) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  HWCNParameter* hwcn_param =
      layer_param.mutable_hwcn_param();
  hwcn_param->set_convert_to(false);
  shared_ptr<Layer<Dtype> > layer(
      new HWCNLayer<Dtype>(layer_param));

  this->blob_bottom_vec_.clear();
  this->blob_top_vec_.clear();

  this->blob_bottom_vec_.push_back(this->blob_bottom_2_);
  this->blob_top_vec_.push_back(this->blob_top_2_);

  layer->SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  layer->Forward(this->blob_bottom_vec_, this->blob_top_vec_);

  const Dtype *bottom_data = this->blob_bottom_2_->cpu_data();
  const Dtype *top_data = this->blob_top_2_->cpu_data();
  
  std::vector<int> shape = this->blob_bottom_2_->shape();
  for (int n = 0; n < shape[3]; ++n) {
    for (int c = 0; c < shape[2]; ++c) {
      for (int h = 0; h < shape[0]; ++h) {
        for (int w = 0; w < shape[1]; ++w) {
          int bot_idx = ((h * shape[1] + w) * shape[2] + c) * shape[3] + n;
          int top_idx = ((n * shape[2] + c) * shape[0] + h) * shape[1] + w;
          EXPECT_EQ(bottom_data[bot_idx], top_data[top_idx]);
        }
      }
    }
  }
}

}  // namespace caffe
