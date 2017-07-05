#include <vector>

#include "gtest/gtest.h"

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/layers/half_conversion_layer.hpp"
#include "caffe/layers/hwcn_layer.hpp"
#include "caffe/layers/ocl_inner_product_hwcn_layer.hpp"
#include "caffe/layers/XCL_program_layer.hpp"

#include "caffe/test/test_caffe_main.hpp"
#include "caffe/test/test_gradient_check_util.hpp"

namespace caffe {

#ifdef USE_OCL

template <typename Dtype>
void caffe_inner_product(const Blob<Dtype>* in,
    InnerProductParameter* ip_param,
    const vector<shared_ptr<Blob<Dtype> > >& weights, Blob<Dtype>* out) {

  const int axis = in->CanonicalAxisIndex(ip_param->axis());
  int M_ = in->count(0, axis);
  int K_ = in->count(axis);
  int N_ = ip_param->num_output();

  Dtype *out_data = out->mutable_cpu_data();
  const Dtype *in_data = in->cpu_data();
  const Dtype *weight_data = weights[0]->cpu_data();
  const Dtype *bias_data = weights[1]->cpu_data();
  for (int m = 0; m < M_; ++m)
    for (int n = 0; n < N_; ++n)
      out_data[m * N_ + n] = bias_data[n];

  for (int m = 0; m < M_; ++m) {
    for (int n = 0; n < N_; ++n) {
      for (int k = 0; k < K_; ++k) {
        int out_offset = m * N_ + n;
        int in_offset = m * K_ + k;
        int w_offset = n * K_ + k;
        out_data[out_offset] += in_data[in_offset] * weight_data[w_offset];
      }
    }
  }
}

template void caffe_inner_product(const Blob<float>* in,
    InnerProductParameter* ip_param,
    const vector<shared_ptr<Blob<float> > >& weights,
    Blob<float>* out);
template void caffe_inner_product(const Blob<double>* in,
    InnerProductParameter* ip_param,
    const vector<shared_ptr<Blob<double> > >& weights,
    Blob<double>* out);

template <typename TypeParam>
class OCLHWCNInnerProductLayerTest : public OCLDeviceTest<TypeParam> {
  typedef typename TypeParam::Dtype Dtype;
 protected:
  OCLHWCNInnerProductLayerTest()
      : blob_bottom_(new Blob<Dtype>(256, 128, 2, 1)),
        blob_top_half_out(new Blob<Dtype>()),
        blob_top_hwcn_out(new Blob<Dtype>()),
        blob_top_hwcn_out_2(new Blob<Dtype>()),
        blob_top_ip_out(new Blob<Dtype>()),
        blob_top_half_out_2(new Blob<Dtype>()) {
    // fill the values
    FillerParameter filler_param;
    UniformFiller<Dtype> filler(filler_param);
    filler.Fill(this->blob_bottom_);
    blob_bottom_vec_hwcn.push_back(blob_bottom_);
    blob_top_vec_hwcn.push_back(blob_top_hwcn_out);
    blob_bottom_vec_half.push_back(blob_top_hwcn_out);
    blob_top_vec_half.push_back(blob_top_half_out);
    blob_bottom_vec_ip.push_back(blob_top_half_out);
    blob_top_vec_ip.push_back(blob_top_ip_out);
    blob_bottom_vec_half_2.push_back(blob_top_ip_out);
    blob_top_vec_half_2.push_back(blob_top_half_out_2);
    blob_bottom_vec_hwcn_2.push_back(blob_top_half_out_2);
    blob_top_vec_hwcn_2.push_back(blob_top_hwcn_out_2);
  }

  virtual ~OCLHWCNInnerProductLayerTest() {
    delete blob_bottom_;
    delete blob_top_hwcn_out;
    delete blob_top_half_out;
    delete blob_top_ip_out;
    delete blob_top_half_out_2;
    delete blob_top_hwcn_out_2;
  }

  virtual Blob<Dtype>* MakeReferenceTop(Blob<Dtype>* top) {
    this->ref_blob_top_.reset(new Blob<Dtype>());
    this->ref_blob_top_->ReshapeLike(*top);
    return this->ref_blob_top_.get();
  }

  Blob<Dtype>* const blob_bottom_;
  Blob<Dtype>* const blob_top_half_out;
  Blob<Dtype>* const blob_top_hwcn_out;
  Blob<Dtype>* const blob_top_ip_out;
  Blob<Dtype>* const blob_top_half_out_2;
  Blob<Dtype>* const blob_top_hwcn_out_2;
  shared_ptr<Blob<Dtype> > ref_blob_top_;
  vector<Blob<Dtype>*> blob_bottom_vec_hwcn;
  vector<Blob<Dtype>*> blob_top_vec_hwcn;
  vector<Blob<Dtype>*> blob_bottom_vec_half;
  vector<Blob<Dtype>*> blob_top_vec_half;
  vector<Blob<Dtype>*> blob_bottom_vec_ip;
  vector<Blob<Dtype>*> blob_top_vec_ip;
  vector<Blob<Dtype>*> blob_bottom_vec_half_2;
  vector<Blob<Dtype>*> blob_top_vec_half_2;
  vector<Blob<Dtype>*> blob_bottom_vec_hwcn_2;
  vector<Blob<Dtype>*> blob_top_vec_hwcn_2;
  vector<Blob<Dtype>*> prog_bot_;
  vector<Blob<Dtype>*> prog_top_;
};

TYPED_TEST_CASE(OCLHWCNInnerProductLayerTest, TestDtypesAndDevices);
/*
TYPED_TEST(OCLHWCNInnerProductLayerTest, TestSetUp) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  InnerProductParameter* inner_product_param =
      layer_param.mutable_inner_product_param();
  inner_product_param->set_num_output(10);
  shared_ptr<OCLHWCNInnerProductLayer<Dtype> > layer(
      new OCLHWCNInnerProductLayer<Dtype>(layer_param));
  layer->SetUp(this->blob_bottom_vec_half, this->blob_top_vec_half);
  EXPECT_EQ(this->blob_top_half_out->num(), 16);
  EXPECT_EQ(this->blob_top_half_out->height(), 1);
  EXPECT_EQ(this->blob_top_half_out->width(), 1);
  EXPECT_EQ(this->blob_top_half_out->channels(), 10);
}*/

TYPED_TEST(OCLHWCNInnerProductLayerTest, TestForward) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  InnerProductParameter* inner_product_param =
      layer_param.mutable_inner_product_param();
  inner_product_param->set_num_output(64);
  inner_product_param->mutable_weight_filler()->set_type("gaussian");
  inner_product_param->mutable_weight_filler()->set_std(0.1);
  inner_product_param->mutable_bias_filler()->set_type("constant");
//  inner_product_param->mutable_bias_filler()->set_min(1);
//  inner_product_param->mutable_bias_filler()->set_max(2);
  XCLParameter* xcl_param = layer_param.mutable_xcl_param();

  xcl_param->set_xcl_name("cr_layer_hwcn_half.xclbin");
  xcl_param->set_kernel_name("cr_layer_hwcn_half");

  shared_ptr<Layer<Dtype> > programLayer(
      new XCLProgramLayer<Dtype>(layer_param));
  programLayer->SetUp(this->prog_bot_, this->prog_top_);
  programLayer->Forward(this->prog_bot_, this->prog_top_);

  HWCNParameter* hwcn_param =
      layer_param.mutable_hwcn_param();
  hwcn_param->set_convert_to(true);

  shared_ptr<Layer<Dtype> > hwcn_layer(
      new HWCNLayer<Dtype>(layer_param));
  hwcn_layer->SetUp(this->blob_bottom_vec_hwcn, this->blob_top_vec_hwcn);
  hwcn_layer->Forward(this->blob_bottom_vec_hwcn, this->blob_top_vec_hwcn);

  HalfConversionParameter* half_param =
      layer_param.mutable_half_conversion_param();
  half_param->set_convert_to(true);

  shared_ptr<Layer<Dtype> > half_layer(
      new HalfConversionLayer<Dtype>(layer_param));
  half_layer->SetUp(this->blob_bottom_vec_half, this->blob_top_vec_half);
  half_layer->Forward(this->blob_bottom_vec_half, this->blob_top_vec_half);
  
  shared_ptr<OCLHWCNInnerProductLayer<Dtype> > layer(
      new OCLHWCNInnerProductLayer<Dtype>(layer_param));
  layer->SetUp(this->blob_bottom_vec_ip, this->blob_top_vec_ip);
  layer->Forward(this->blob_bottom_vec_ip, this->blob_top_vec_ip);

  half_param->set_convert_to(false);

  shared_ptr<Layer<Dtype> > half_layer2(
      new HalfConversionLayer<Dtype>(layer_param));
  half_layer2->SetUp(this->blob_bottom_vec_half_2, this->blob_top_vec_half_2);
  half_layer2->Forward(this->blob_bottom_vec_half_2,
      this->blob_top_vec_half_2);

  hwcn_param->set_convert_to(false);

  shared_ptr<Layer<Dtype> > hwcn_layer2(
      new HWCNLayer<Dtype>(layer_param));
  hwcn_layer2->SetUp(this->blob_bottom_vec_hwcn_2, this->blob_top_vec_hwcn_2);
  hwcn_layer2->Forward(this->blob_bottom_vec_hwcn_2, this->blob_top_vec_hwcn_2);

  const Dtype *top_data;
  const Dtype *ref_top_data;

  caffe_inner_product(this->blob_bottom_, inner_product_param, layer->blobs(),
      this->MakeReferenceTop(this->blob_top_hwcn_out_2));

  top_data = this->blob_top_hwcn_out_2->cpu_data();
  ref_top_data = this->ref_blob_top_->cpu_data();
  const int count = this->blob_top_hwcn_out_2->count();
  for (int i = 0; i < count; ++i) {
    std::cout<<top_data[i]<<" "<<ref_top_data[i]<<std::endl;
    EXPECT_NEAR(top_data[i], ref_top_data[i], 1e-1);
  }
}

#endif  // USE_OCL

}  // namespace caffe
