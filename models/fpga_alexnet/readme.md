FPGA Caffe Custom-Precision Floating-Point AlexNet
=====================================

The figure below shows the accuracy degradation of various custom-precision floating-point settings for inference using a pretrained model on the Imagenet validationg set, each of which uses round-to-zero for the multipliers and round-to-nearest for the accumulators. Convolution, max pooling, relu, and fully connected layers are implemented on the FPGA, all other layers are on the CPU.

![alexnet_top1](https://github.com/dicecco1/fpga_caffe/blob/master/models/fpga_alexnet/alexnet_top1.png)
![alexnet_top5](https://github.com/dicecco1/fpga_caffe/blob/master/models/fpga_alexnet/alexnet_top5.png)
