#include <vector>

#include "gtest/gtest.h"

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/layers/half_conversion_layer.hpp"

#include "caffe/test/test_caffe_main.hpp"

namespace caffe {

template <typename TypeParam>
class HalfConversionLayerTest : public MultiDeviceTest<TypeParam> {
  typedef typename TypeParam::Dtype Dtype;

 protected:
  HalfConversionLayerTest()
      : blob_bottom_(new Blob<Dtype>(2, 3, 6, 4)),
        blob_top_(new Blob<Dtype>()) {}
  virtual void SetUp() {
    // fill the values
    FillerParameter filler_param;
    filler_param.set_value(1.);
    GaussianFiller<Dtype> filler(filler_param);
    filler.Fill(this->blob_bottom_);
    blob_bottom_vec_.push_back(blob_bottom_);
    blob_top_vec_.push_back(blob_top_);
  }

  virtual ~HalfConversionLayerTest() {
    delete blob_bottom_;
    delete blob_top_;
  }

  Blob<Dtype>* const blob_bottom_;
  Blob<Dtype>* const blob_top_;
  vector<Blob<Dtype>*> blob_bottom_vec_;
  vector<Blob<Dtype>*> blob_top_vec_;
};

TYPED_TEST_CASE(HalfConversionLayerTest, TestDtypesAndDevices);

TYPED_TEST(HalfConversionLayerTest, TestForward) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  HalfConversionParameter* half_conversion_param =
      layer_param.mutable_half_conversion_param();
  half_conversion_param->set_convert_to(true);
  shared_ptr<Layer<Dtype> > layer(
      new HalfConversionLayer<Dtype>(layer_param));
  layer->SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  layer->Forward(this->blob_bottom_vec_, this->blob_top_vec_);

  const chalf* top_data =
    reinterpret_cast<const chalf *>(this->blob_top_->cpu_data());
  const Dtype* bottom_data = this->blob_bottom_->cpu_data();

  for (int i = 0; i < this->blob_top_->count(); ++i) {
    std::cout<<bottom_data[i]<<" "<<(Dtype)(float(top_data[i]))<<std::endl;
    EXPECT_NEAR(bottom_data[i], (Dtype)(float(top_data[i])), 1e-3);
  }
}

}  // namespace caffe
