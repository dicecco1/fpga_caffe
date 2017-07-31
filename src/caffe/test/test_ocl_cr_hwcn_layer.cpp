#include <vector>

#include "gtest/gtest.h"

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/layers/hwcn_layer.hpp"
#include "caffe/layers/cpfp_conversion_layer.hpp"
#include "caffe/layers/ocl_cr_hwcn_layer.hpp"
#include "caffe/layers/XCL_program_layer.hpp"
#include "caffe/test/test_caffe_main.hpp"

namespace caffe {

#ifdef USE_OCL
// Reference convolution for checking results:
// accumulate through explicit loops over input, output, and filters.
template <typename Dtype>
void caffe_conv_relu(const Blob<Dtype>* in, ConvolutionParameter* conv_param,
    const vector<shared_ptr<Blob<Dtype> > >& weights,
    Blob<Dtype>* out) {

  const bool has_depth = (out->num_axes() == 5);
  if (!has_depth) { CHECK_EQ(4, out->num_axes()); }
  // Kernel size, stride, and pad
  int kernel_h, kernel_w;
  if (conv_param->has_kernel_h() || conv_param->has_kernel_w()) {
    kernel_h = conv_param->kernel_h();
    kernel_w = conv_param->kernel_w();
  } else {
    kernel_h = kernel_w = conv_param->kernel_size(0);
  }
  int pad_h, pad_w;
  if (conv_param->has_pad_h() || conv_param->has_pad_w()) {
    pad_h = conv_param->pad_h();
    pad_w = conv_param->pad_w();
  } else {
    pad_h = pad_w = conv_param->pad_size() ? conv_param->pad(0) : 0;
  }
  int stride_h, stride_w;
  if (conv_param->has_stride_h() || conv_param->has_stride_w()) {
    stride_h = conv_param->stride_h();
    stride_w = conv_param->stride_w();
  } else {
    stride_h = stride_w = conv_param->stride_size() ? conv_param->stride(0) : 1;
  }
  int dilation_h, dilation_w;
  dilation_h = dilation_w = conv_param->dilation_size() ?
                            conv_param->dilation(0) : 1;
  int kernel_d, pad_d, stride_d, dilation_d;
  if (has_depth) {
    kernel_d = kernel_h;
    stride_d = stride_h;
    pad_d = pad_h;
    dilation_d = dilation_h;
  } else {
    kernel_d = stride_d = dilation_d = 1;
    pad_d = 0;
  }
  // Groups
  int groups = conv_param->group();
  int o_g = out->shape(1) / groups;
  int k_g = in->shape(1) / groups;
  int o_head, k_head;
  // Convolution
  vector<int> weight_offset(4 + has_depth);
  vector<int> in_offset(4 + has_depth);
  vector<int> out_offset(4 + has_depth);
  Dtype* out_data = out->mutable_cpu_data();

  for (int n = 0; n < out->shape(0); n++) {
    for (int g = 0; g < groups; g++) {
      o_head = o_g * g;
      k_head = k_g * g;
      for (int o = 0; o < o_g; o++) {
        for (int k = 0; k < k_g; k++) {
          for (int z = 0; z < (has_depth ? out->shape(2) : 1); z++) {
            for (int y = 0; y < out->shape(2 + has_depth); y++) {
              for (int x = 0; x < out->shape(3 + has_depth); x++) {
                for (int r = 0; r < kernel_d; r++) {
                  for (int p = 0; p < kernel_h; p++) {
                    for (int q = 0; q < kernel_w; q++) {
                      int in_z = z * stride_d - pad_d + r * dilation_d;
                      int in_y = y * stride_h - pad_h + p * dilation_h;
                      int in_x = x * stride_w - pad_w + q * dilation_w;
                      if (in_z >= 0 && in_z < (has_depth ? in->shape(2) : 1)
                          && in_y >= 0 && in_y < in->shape(2 + has_depth)
                          && in_x >= 0 && in_x < in->shape(3 + has_depth)) {
                        weight_offset[0] = o + o_head;
                        weight_offset[1] = k;
                        if (has_depth) { weight_offset[2] = r; }
                        weight_offset[2 + has_depth] = p;
                        weight_offset[3 + has_depth] = q;
                        in_offset[0] = n;
                        in_offset[1] = k + k_head;
                        if (has_depth) { in_offset[2] = in_z; }
                        in_offset[2 + has_depth] = in_y;
                        in_offset[3 + has_depth] = in_x;
                        out_offset[0] = n;
                        out_offset[1] = o + o_head;
                        if (has_depth) { out_offset[2] = z; }
                        out_offset[2 + has_depth] = y;
                        out_offset[3 + has_depth] = x;
                        out_data[out->offset(out_offset)] +=
                            in->data_at(in_offset)
                            * weights[0]->data_at(weight_offset);
                      }
                    }
                  }
                }
              }
            }
          }
        }
      }
    }
  }
  // Bias
  if (conv_param->bias_term()) {
    const Dtype* bias_data = weights[1]->cpu_data();
    for (int n = 0; n < out->shape(0); n++) {
      for (int o = 0; o < out->shape(1); o++) {
        for (int z = 0; z < (has_depth ? out->shape(2) : 1); z++) {
          for (int y = 0; y < out->shape(2 + has_depth); y++) {
            for (int x = 0; x < out->shape(3 + has_depth); x++) {
              out_offset[0] = n;
              out_offset[1] = o;
              if (has_depth) { out_offset[2] = z; }
              out_offset[2 + has_depth] = y;
              out_offset[3 + has_depth] = x;
              out_data[out->offset(out_offset)] += bias_data[o];
//              if (out_data[out->offset(out_offset)] < 0)
//                out_data[out->offset(out_offset)] = 0;
            }
          }
        }
      }
    }
  }
}

template void caffe_conv_relu(const Blob<float>* in,
    ConvolutionParameter* conv_param,
    const vector<shared_ptr<Blob<float> > >& weights,
    Blob<float>* out);
template void caffe_conv_relu(const Blob<double>* in,
    ConvolutionParameter* conv_param,
    const vector<shared_ptr<Blob<double> > >& weights,
    Blob<double>* out);

template <typename TypeParam>
class OCLCRHWCNLayerTest : public OCLDeviceTest<TypeParam> {
  typedef typename TypeParam::Dtype Dtype;

 protected:
  OCLCRHWCNLayerTest()
      : blob_bottom_(new Blob<Dtype>(256, 32, 12, 12)),
        blob_top_hwcn_out(new Blob<Dtype>()),
        blob_top_cpfp_out(new Blob<Dtype>()),
        blob_top_cr_out(new Blob<Dtype>()),
        blob_top_cpfp_out_2(new Blob<Dtype>()),
        blob_top_hwcn_out_2(new Blob<Dtype>()) {}
  virtual void SetUp() {
    // fill the values
    FillerParameter filler_param;
    filler_param.set_value(1.);
    GaussianFiller<Dtype> filler(filler_param);
    filler.Fill(this->blob_bottom_);
    blob_bottom_vec_hwcn.push_back(blob_bottom_);
    blob_top_vec_hwcn.push_back(blob_top_hwcn_out);
    blob_bottom_vec_cpfp.push_back(blob_top_hwcn_out);
    blob_top_vec_cpfp.push_back(blob_top_cpfp_out);
    blob_bottom_vec_cr.push_back(blob_top_cpfp_out);
    blob_top_vec_cr.push_back(blob_top_cr_out);
    blob_bottom_vec_cpfp_2.push_back(blob_top_cr_out);
    blob_top_vec_cpfp_2.push_back(blob_top_cpfp_out_2);
    blob_bottom_vec_hwcn_2.push_back(blob_top_cpfp_out_2);
    blob_top_vec_hwcn_2.push_back(blob_top_hwcn_out_2);
  }

  virtual ~OCLCRHWCNLayerTest() {
    delete blob_bottom_;
    delete blob_top_hwcn_out;
    delete blob_top_cpfp_out;
    delete blob_top_cr_out;
    delete blob_top_cpfp_out_2;
    delete blob_top_hwcn_out_2;
  }

  virtual Blob<Dtype>* MakeReferenceTop(Blob<Dtype>* top) {
    this->ref_blob_top_.reset(new Blob<Dtype>());
    this->ref_blob_top_->ReshapeLike(*top);
    return this->ref_blob_top_.get();
  }

  Blob<Dtype>* const blob_bottom_;
  Blob<Dtype>* const blob_top_hwcn_out;
  Blob<Dtype>* const blob_top_cpfp_out;
  Blob<Dtype>* const blob_top_cr_out;
  Blob<Dtype>* const blob_top_cpfp_out_2;
  Blob<Dtype>* const blob_top_hwcn_out_2;
  shared_ptr<Blob<Dtype> > ref_blob_top_;
  vector<Blob<Dtype>*> blob_bottom_vec_hwcn;
  vector<Blob<Dtype>*> blob_top_vec_hwcn;
  vector<Blob<Dtype>*> blob_bottom_vec_cpfp;
  vector<Blob<Dtype>*> blob_top_vec_cpfp;
  vector<Blob<Dtype>*> blob_bottom_vec_cr;
  vector<Blob<Dtype>*> blob_top_vec_cr;
  vector<Blob<Dtype>*> blob_bottom_vec_cpfp_2;
  vector<Blob<Dtype>*> blob_top_vec_cpfp_2;
  vector<Blob<Dtype>*> blob_bottom_vec_hwcn_2;
  vector<Blob<Dtype>*> blob_top_vec_hwcn_2;
  vector<Blob<Dtype>*> prog_bot_;
  vector<Blob<Dtype>*> prog_top_;
};

TYPED_TEST_CASE(OCLCRHWCNLayerTest, TestOCLDtypesAndDevices);
/*
TYPED_TEST(OCLCRHWCNLayerTest, TestForward1x1) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;

  XCLParameter* xcl_param = layer_param.mutable_xcl_param();

  xcl_param->set_xcl_name("cr_layer_hwcn_cpfp.xclbin");
  xcl_param->set_kernel_name("cr_layer_hwcn_cpfp");
  shared_ptr<Layer<Dtype> > programLayer(
      new XCLProgramLayer<Dtype>(layer_param));
  programLayer->SetUp(this->prog_bot_, this->prog_top_);
  programLayer->Forward(this->prog_bot_, this->prog_top_);
  layer_param.set_ocl_enable(true);
  ConvolutionParameter* convolution_param =
      layer_param.mutable_convolution_param();
  convolution_param->add_kernel_size(1);
  convolution_param->add_stride(1);
  convolution_param->set_num_output(12);
  convolution_param->add_pad(0);
  convolution_param->mutable_weight_filler()->set_type("gaussian");
  convolution_param->mutable_bias_filler()->set_type("gaussian");
  convolution_param->set_engine(ConvolutionParameter_Engine_OCL);
  convolution_param->set_subengine(ConvolutionParameter_SubEngine_WINOGRAD);

  CPFPConversionParameter* cpfp_param =
      layer_param.mutable_cpfp_conversion_param();
  cpfp_param->set_convert_to(true);

  HWCNParameter* hwcn_param =
      layer_param.mutable_hwcn_param();
  hwcn_param->set_convert_to(true);

  shared_ptr<Layer<Dtype> > hwcn_layer(
      new HWCNLayer<Dtype>(layer_param));
  hwcn_layer->SetUp(this->blob_bottom_vec_hwcn, this->blob_top_vec_hwcn);
  hwcn_layer->Forward(this->blob_bottom_vec_hwcn, this->blob_top_vec_hwcn);

  shared_ptr<Layer<Dtype> > cpfp_layer(
      new CPFPConversionLayer<Dtype>(layer_param));
  cpfp_layer->SetUp(this->blob_bottom_vec_cpfp, this->blob_top_vec_cpfp);
  cpfp_layer->Forward(this->blob_bottom_vec_cpfp, this->blob_top_vec_cpfp);

  CRParameter* cr_param = layer_param.mutable_cr_param();
  cr_param->set_relu(0);

  shared_ptr<Layer<Dtype> > cr_layer(
      new OCLCRHWCNLayer<Dtype>(layer_param));
  cr_layer->SetUp(this->blob_bottom_vec_cr, this->blob_top_vec_cr);
  cr_layer->Forward(this->blob_bottom_vec_cr, this->blob_top_vec_cr);

  cpfp_param->set_convert_to(false);

  shared_ptr<Layer<Dtype> > cpfp_layer2(
      new CPFPConversionLayer<Dtype>(layer_param));
  cpfp_layer2->SetUp(this->blob_bottom_vec_cpfp_2, this->blob_top_vec_cpfp_2);
  cpfp_layer2->Forward(this->blob_bottom_vec_cpfp_2,
      this->blob_top_vec_cpfp_2);

  hwcn_param->set_convert_to(false);

  shared_ptr<Layer<Dtype> > hwcn_layer2(
      new HWCNLayer<Dtype>(layer_param));
  hwcn_layer2->SetUp(this->blob_bottom_vec_hwcn_2, this->blob_top_vec_hwcn_2);
  hwcn_layer2->Forward(this->blob_bottom_vec_hwcn_2, this->blob_top_vec_hwcn_2);

  const Dtype* top_data;
  const Dtype* ref_top_data;

  caffe_conv_relu(this->blob_bottom_, convolution_param, cr_layer->blobs(),
      this->MakeReferenceTop(this->blob_top_hwcn_out_2));

  top_data = this->blob_top_hwcn_out_2->cpu_data();
  ref_top_data = this->ref_blob_top_->cpu_data();

  for (int i = 0; i < this->ref_blob_top_->count(); ++i) {
    EXPECT_NEAR(top_data[i], ref_top_data[i], 1e-1);
  }
}

TYPED_TEST(OCLCRHWCNLayerTest, TestForward3x3) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;

  XCLParameter* xcl_param = layer_param.mutable_xcl_param();

  xcl_param->set_xcl_name("cr_layer_hwcn_cpfp.xclbin");
  xcl_param->set_kernel_name("cr_layer_hwcn_cpfp");
  shared_ptr<Layer<Dtype> > programLayer(
      new XCLProgramLayer<Dtype>(layer_param));
  programLayer->SetUp(this->prog_bot_, this->prog_top_);
  programLayer->Forward(this->prog_bot_, this->prog_top_);
  layer_param.set_ocl_enable(true);
  ConvolutionParameter* convolution_param =
      layer_param.mutable_convolution_param();
  convolution_param->add_kernel_size(3);
  convolution_param->add_stride(1);
  convolution_param->set_num_output(4);
  convolution_param->add_pad(1);
  convolution_param->mutable_weight_filler()->set_type("gaussian");
  convolution_param->mutable_bias_filler()->set_type("gaussian");
  convolution_param->set_engine(ConvolutionParameter_Engine_OCL);
  convolution_param->set_subengine(ConvolutionParameter_SubEngine_WINOGRAD);

  CPFPConversionParameter* cpfp_param =
      layer_param.mutable_cpfp_conversion_param();
  cpfp_param->set_convert_to(true);

  HWCNParameter* hwcn_param =
      layer_param.mutable_hwcn_param();
  hwcn_param->set_convert_to(true);

  shared_ptr<Layer<Dtype> > hwcn_layer(
      new HWCNLayer<Dtype>(layer_param));
  hwcn_layer->SetUp(this->blob_bottom_vec_hwcn, this->blob_top_vec_hwcn);
  hwcn_layer->Forward(this->blob_bottom_vec_hwcn, this->blob_top_vec_hwcn);

  shared_ptr<Layer<Dtype> > cpfp_layer(
      new CPFPConversionLayer<Dtype>(layer_param));
  cpfp_layer->SetUp(this->blob_bottom_vec_cpfp, this->blob_top_vec_cpfp);
  cpfp_layer->Forward(this->blob_bottom_vec_cpfp, this->blob_top_vec_cpfp);

  CRParameter* cr_param = layer_param.mutable_cr_param();
  cr_param->set_relu(0);

  shared_ptr<Layer<Dtype> > cr_layer(
      new OCLCRHWCNLayer<Dtype>(layer_param));
  cr_layer->SetUp(this->blob_bottom_vec_cr, this->blob_top_vec_cr);
  cr_layer->Forward(this->blob_bottom_vec_cr, this->blob_top_vec_cr);

  cpfp_param->set_convert_to(false);

  shared_ptr<Layer<Dtype> > cpfp_layer2(
      new CPFPConversionLayer<Dtype>(layer_param));
  cpfp_layer2->SetUp(this->blob_bottom_vec_cpfp_2, this->blob_top_vec_cpfp_2);
  cpfp_layer2->Forward(this->blob_bottom_vec_cpfp_2,
      this->blob_top_vec_cpfp_2);

  hwcn_param->set_convert_to(false);

  shared_ptr<Layer<Dtype> > hwcn_layer2(
      new HWCNLayer<Dtype>(layer_param));
  hwcn_layer2->SetUp(this->blob_bottom_vec_hwcn_2, this->blob_top_vec_hwcn_2);
  hwcn_layer2->Forward(this->blob_bottom_vec_hwcn_2, this->blob_top_vec_hwcn_2);

  const Dtype* top_data;
  const Dtype* ref_top_data;

  caffe_conv_relu(this->blob_bottom_, convolution_param, cr_layer->blobs(),
      this->MakeReferenceTop(this->blob_top_hwcn_out_2));

  top_data = this->blob_top_hwcn_out_2->cpu_data();
  ref_top_data = this->ref_blob_top_->cpu_data();

  for (int i = 0; i < this->ref_blob_top_->count(); ++i) {
    EXPECT_NEAR(top_data[i], ref_top_data[i], 1e-1);
  }
}
*/
TYPED_TEST(OCLCRHWCNLayerTest, TestForward5x5) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;

  XCLParameter* xcl_param = layer_param.mutable_xcl_param();

  xcl_param->set_xcl_name("cr_layer_hwcn_cpfp.xclbin");
  xcl_param->set_kernel_name("cr_layer_hwcn_cpfp");
  shared_ptr<Layer<Dtype> > programLayer(
      new XCLProgramLayer<Dtype>(layer_param));
  programLayer->SetUp(this->prog_bot_, this->prog_top_);
  programLayer->Forward(this->prog_bot_, this->prog_top_);
  layer_param.set_ocl_enable(true);
  ConvolutionParameter* convolution_param =
      layer_param.mutable_convolution_param();
  convolution_param->add_kernel_size(5);
  convolution_param->add_stride(1);
  convolution_param->set_num_output(64);
  convolution_param->add_pad(0);
  convolution_param->mutable_weight_filler()->set_type("gaussian");
  convolution_param->mutable_bias_filler()->set_type("gaussian");
  convolution_param->set_engine(ConvolutionParameter_Engine_OCL);
  convolution_param->set_subengine(ConvolutionParameter_SubEngine_WINOGRAD);

  CPFPConversionParameter* cpfp_param =
      layer_param.mutable_cpfp_conversion_param();
  cpfp_param->set_convert_to(true);

  HWCNParameter* hwcn_param =
      layer_param.mutable_hwcn_param();
  hwcn_param->set_convert_to(true);

  shared_ptr<Layer<Dtype> > hwcn_layer(
      new HWCNLayer<Dtype>(layer_param));
  hwcn_layer->SetUp(this->blob_bottom_vec_hwcn, this->blob_top_vec_hwcn);
  hwcn_layer->Forward(this->blob_bottom_vec_hwcn, this->blob_top_vec_hwcn);

  shared_ptr<Layer<Dtype> > cpfp_layer(
      new CPFPConversionLayer<Dtype>(layer_param));
  cpfp_layer->SetUp(this->blob_bottom_vec_cpfp, this->blob_top_vec_cpfp);
  cpfp_layer->Forward(this->blob_bottom_vec_cpfp, this->blob_top_vec_cpfp);

  CRParameter* cr_param = layer_param.mutable_cr_param();
  cr_param->set_relu(0);

  shared_ptr<Layer<Dtype> > cr_layer(
      new OCLCRHWCNLayer<Dtype>(layer_param));
  cr_layer->SetUp(this->blob_bottom_vec_cr, this->blob_top_vec_cr);
  cr_layer->Forward(this->blob_bottom_vec_cr, this->blob_top_vec_cr);

  cpfp_param->set_convert_to(false);

  shared_ptr<Layer<Dtype> > cpfp_layer2(
      new CPFPConversionLayer<Dtype>(layer_param));
  cpfp_layer2->SetUp(this->blob_bottom_vec_cpfp_2, this->blob_top_vec_cpfp_2);
  cpfp_layer2->Forward(this->blob_bottom_vec_cpfp_2,
      this->blob_top_vec_cpfp_2);

  hwcn_param->set_convert_to(false);

  shared_ptr<Layer<Dtype> > hwcn_layer2(
      new HWCNLayer<Dtype>(layer_param));
  hwcn_layer2->SetUp(this->blob_bottom_vec_hwcn_2, this->blob_top_vec_hwcn_2);
  hwcn_layer2->Forward(this->blob_bottom_vec_hwcn_2, this->blob_top_vec_hwcn_2);

  const Dtype* top_data;
  const Dtype* ref_top_data;

  caffe_conv_relu(this->blob_bottom_, convolution_param, cr_layer->blobs(),
      this->MakeReferenceTop(this->blob_top_hwcn_out_2));

  top_data = this->blob_top_hwcn_out_2->cpu_data();
  ref_top_data = this->ref_blob_top_->cpu_data();

  for (int i = 0; i < this->ref_blob_top_->count(); ++i) {
    EXPECT_NEAR(top_data[i], ref_top_data[i], 1e-1);
  }
}
/*
TYPED_TEST(OCLCRHWCNLayerTest, TestForward3x3_s2) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;

  XCLParameter* xcl_param = layer_param.mutable_xcl_param();

  xcl_param->set_xcl_name("cr_layer_hwcn_cpfp.xclbin");
  xcl_param->set_kernel_name("cr_layer_hwcn_cpfp");
  shared_ptr<Layer<Dtype> > programLayer(
      new XCLProgramLayer<Dtype>(layer_param));
  programLayer->SetUp(this->prog_bot_, this->prog_top_);
  programLayer->Forward(this->prog_bot_, this->prog_top_);
  layer_param.set_ocl_enable(true);
  ConvolutionParameter* convolution_param =
      layer_param.mutable_convolution_param();
  convolution_param->add_kernel_size(3);
  convolution_param->add_stride(2);
  convolution_param->set_num_output(4);
  convolution_param->add_pad(1);
  convolution_param->mutable_weight_filler()->set_type("gaussian");
  convolution_param->mutable_bias_filler()->set_type("gaussian");
  convolution_param->set_engine(ConvolutionParameter_Engine_OCL);
  convolution_param->set_subengine(ConvolutionParameter_SubEngine_WINOGRAD);

  CPFPConversionParameter* cpfp_param =
      layer_param.mutable_cpfp_conversion_param();
  cpfp_param->set_convert_to(true);

  HWCNParameter* hwcn_param =
      layer_param.mutable_hwcn_param();
  hwcn_param->set_convert_to(true);

  shared_ptr<Layer<Dtype> > hwcn_layer(
      new HWCNLayer<Dtype>(layer_param));
  hwcn_layer->SetUp(this->blob_bottom_vec_hwcn, this->blob_top_vec_hwcn);
  hwcn_layer->Forward(this->blob_bottom_vec_hwcn, this->blob_top_vec_hwcn);

  shared_ptr<Layer<Dtype> > cpfp_layer(
      new CPFPConversionLayer<Dtype>(layer_param));
  cpfp_layer->SetUp(this->blob_bottom_vec_cpfp, this->blob_top_vec_cpfp);
  cpfp_layer->Forward(this->blob_bottom_vec_cpfp, this->blob_top_vec_cpfp);

  CRParameter* cr_param = layer_param.mutable_cr_param();
  cr_param->set_relu(0);

  shared_ptr<Layer<Dtype> > cr_layer(
      new OCLCRHWCNLayer<Dtype>(layer_param));
  cr_layer->SetUp(this->blob_bottom_vec_cr, this->blob_top_vec_cr);
  cr_layer->Forward(this->blob_bottom_vec_cr, this->blob_top_vec_cr);

  cpfp_param->set_convert_to(false);

  shared_ptr<Layer<Dtype> > cpfp_layer2(
      new CPFPConversionLayer<Dtype>(layer_param));
  cpfp_layer2->SetUp(this->blob_bottom_vec_cpfp_2, this->blob_top_vec_cpfp_2);
  cpfp_layer2->Forward(this->blob_bottom_vec_cpfp_2,
      this->blob_top_vec_cpfp_2);

  hwcn_param->set_convert_to(false);

  shared_ptr<Layer<Dtype> > hwcn_layer2(
      new HWCNLayer<Dtype>(layer_param));
  hwcn_layer2->SetUp(this->blob_bottom_vec_hwcn_2, this->blob_top_vec_hwcn_2);
  hwcn_layer2->Forward(this->blob_bottom_vec_hwcn_2, this->blob_top_vec_hwcn_2);

  const Dtype* top_data;
  const Dtype* ref_top_data;

  caffe_conv_relu(this->blob_bottom_, convolution_param, cr_layer->blobs(),
      this->MakeReferenceTop(this->blob_top_hwcn_out_2));

  top_data = this->blob_top_hwcn_out_2->cpu_data();
  ref_top_data = this->ref_blob_top_->cpu_data();

  for (int i = 0; i < this->ref_blob_top_->count(); ++i) {
    EXPECT_NEAR(top_data[i], ref_top_data[i], 1e-1);
  }
}*/
#endif  // USE_OCL
}  // namespace caffe
