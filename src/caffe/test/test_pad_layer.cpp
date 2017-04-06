#include <vector>

#include "gtest/gtest.h"

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/layers/pad_layer.hpp"

#include "caffe/test/test_caffe_main.hpp"

namespace caffe {

template <typename TypeParam>
class PadLayerTest : public MultiDeviceTest<TypeParam> {
  typedef typename TypeParam::Dtype Dtype;

 protected:
  PadLayerTest()
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

  virtual ~PadLayerTest() {
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

TYPED_TEST_CASE(PadLayerTest, TestDtypesAndDevices);

TYPED_TEST(PadLayerTest, TestForwardPad) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  PadParameter* pad_param =
      layer_param.mutable_pad_param();
  pad_param->set_pad(true);
  shared_ptr<Layer<Dtype> > layer(
      new PadLayer<Dtype>(layer_param));
  layer->SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  layer->Forward(this->blob_bottom_vec_, this->blob_top_vec_);

  int xdim_ = this->blob_bottom_->shape(3);
  int xdim_pad_ = this->blob_top_->shape(3);

  const Dtype *bottom_data = this->blob_bottom_->cpu_data();
  const Dtype *top_data = this->blob_top_->cpu_data();

  for (int i = 0; i < this->blob_top_->count() / xdim_pad_; ++i) {
    for (int x = 0; x < xdim_pad_; ++x) {
      if (x < xdim_) 
        EXPECT_NEAR(bottom_data[i * xdim_ + x], top_data[i * xdim_pad_ + x],
            1e-3);
      else
        EXPECT_NEAR(top_data[i * xdim_pad_ + x], 0, 1e-3);
    }
  }
}

TYPED_TEST(PadLayerTest, TestForwardUnpad) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  PadParameter* pad_param =
      layer_param.mutable_pad_param();
  pad_param->set_pad(false);
  shared_ptr<Layer<Dtype> > layer(
      new PadLayer<Dtype>(layer_param));

  this->blob_bottom_vec_.clear();
  this->blob_top_vec_.clear();

  this->blob_bottom_vec_.push_back(this->blob_bottom_2_);
  this->blob_top_vec_.push_back(this->blob_top_2_);

  layer->SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  layer->Forward(this->blob_bottom_vec_, this->blob_top_vec_);

  int xdim_pad_ = this->blob_bottom_2_->shape(3);
  int xdim_ = this->blob_top_2_->shape(3);

  const Dtype *bottom_data = this->blob_bottom_2_->cpu_data();
  const Dtype *top_data = this->blob_top_2_->cpu_data();

  for (int i = 0; i < this->blob_top_2_->count() / xdim_; ++i) {
    for (int x = 0; x < xdim_pad_; ++x) {
      if (x < xdim_) 
        EXPECT_NEAR(bottom_data[i * xdim_pad_ + x], top_data[i * xdim_ + x],
            1e-3);
    }
  }
}

}  // namespace caffe
