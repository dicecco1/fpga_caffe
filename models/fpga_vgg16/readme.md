FPGA Caffe Custom-Precision Floating-Point VGG16
=====================================

The figures below shows the accuracy degradation of various custom-precision floating-point settings for inference using a pretrained model on the Imagenet validation set, each of which uses round-to-zero for the multipliers and round-to-nearest for the accumulators. Convolution, max pooling, relu, and fully connected layers are implemented on the FPGA, all other layers are on the CPU.

![vgg_top1](https://github.com/dicecco1/fpga_caffe/blob/master/models/fpga_vgg16/vgg_top1.png)
![vgg_top5](https://github.com/dicecco1/fpga_caffe/blob/master/models/fpga_vgg16/vgg_top5.png)
