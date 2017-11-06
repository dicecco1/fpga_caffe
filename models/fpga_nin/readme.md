FPGA Caffe Custom-Precision Floating-Point Network in Network
=====================================

The figures below shows the accuracy degradation of various custom-precision floating-point settings for inference using a pretrained model on the Imagenet validation set, each of which uses round-to-zero for the multipliers and round-to-nearest for the accumulators. Convolution, max pooling, relu, and fully connected layers are implemented on the FPGA, all other layers are on the CPU.

![nin_top1](https://github.com/dicecco1/fpga_caffe/blob/master/models/fpga_nin/nin_top1.png)
![nin_top5](https://github.com/dicecco1/fpga_caffe/blob/master/models/fpga_nin/nin_top5.png)
