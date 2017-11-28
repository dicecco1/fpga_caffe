# FPGA Caffe

This is a version of Caffe with FPGA kernels for forward and backward: convolution, relu, max pooling, and inner product. These kernels target the Xilinx SDAccel OpenCL environment. The kernels use custom-precision floating-point arithmetic to save area and improve the throughput of the kernels, while also allowing for experimentation with different floating-point precisions and rounding modes for training and inference with CNNs.

Infrastructure has been added to facilitate the use of Xilinx SDAccel kernels within Caffe, while making it essentially seamless to outside users that an FPGA is in use (aside from some additional layers required to program the device). 

The version of SDAccel where most of the custom-precision floating-point results were gathered was 2016.3. Later versions of SDAccel should work too, though low precision multipliers don't seem to map well to DSPs in 2017.1. To overcome this use the 3 input multiplier implementation of the crp layer. 

## License and Citation

The license for this project is the same as that of the original Caffe implementation. Our initial paper related to this work can be found at: http://ieeexplore.ieee.org/document/7929549/

Citation:
@INPROCEEDINGS{7929549, 
author={R. DiCecco and G. Lacey and J. Vasiljevic and P. Chow and G. Taylor and S. Areibi}, 
booktitle={2016 International Conference on Field-Programmable Technology (FPT)}, 
title={Caffeinated FPGAs: FPGA framework For Convolutional Neural Networks}, 
year={2016}, 
pages={265-268}, 
keywords={Computational modeling;Convolution;Field programmable gate arrays;Graphics processing units;Kernel;Parallel processing;Pipelines}, 
doi={10.1109/FPT.2016.7929549}, 
month={Dec},}

The work related to the Custom-Precision Floating-Point Training is set to appear at FPT2017. The repository for the Custom-Precision Floating-Point library is at: https://github.com/dicecco1/fpga_cpfp.

## Build Instructions

In Makefile.config set USE_OCL := 1 and CPU_ONLY := 1 (CPU_ONLY won't be necessary soon) and run make all. 

To build standalone FPGA tests run make testfpga. These tests may not always pass depending on what level of precision is specified because they compare to a single-precision reference. 

To build FPGA layers (or add new layers), in src/fpga_caffe/layers/ run make -f layer.mk KERNEL_NAME=YOUR_KERNEL_NAME, the kernel name currently has to be the same as the .cpp name (e.g. crp_layer_hwcn_cpfp kernel has a .cpp file named crp_layer_hwcn_cpfp.cpp). After the xclbins have been generated, they should be copied to .build_release/opencl/src/caffe/layers/

# Caffe

[![Build Status](https://travis-ci.org/BVLC/caffe.svg?branch=master)](https://travis-ci.org/BVLC/caffe)
[![License](https://img.shields.io/badge/license-BSD-blue.svg)](LICENSE)

Caffe is a deep learning framework made with expression, speed, and modularity in mind.
It is developed by Berkeley AI Research ([BAIR](http://bair.berkeley.edu))/The Berkeley Vision and Learning Center (BVLC) and community contributors.

Check out the [project site](http://caffe.berkeleyvision.org) for all the details like

- [DIY Deep Learning for Vision with Caffe](https://docs.google.com/presentation/d/1UeKXVgRvvxg9OUdh_UiC5G71UMscNPlvArsWER41PsU/edit#slide=id.p)
- [Tutorial Documentation](http://caffe.berkeleyvision.org/tutorial/)
- [BAIR reference models](http://caffe.berkeleyvision.org/model_zoo.html) and the [community model zoo](https://github.com/BVLC/caffe/wiki/Model-Zoo)
- [Installation instructions](http://caffe.berkeleyvision.org/installation.html)

and step-by-step examples.

## Custom distributions

 - [Intel Caffe](https://github.com/BVLC/caffe/tree/intel) (Optimized for CPU and support for multi-node), in particular Xeon processors (HSW, BDW, SKX, Xeon Phi).
- [OpenCL Caffe](https://github.com/BVLC/caffe/tree/opencl) e.g. for AMD or Intel devices.
- [Windows Caffe](https://github.com/BVLC/caffe/tree/windows)

## Community

[![Join the chat at https://gitter.im/BVLC/caffe](https://badges.gitter.im/Join%20Chat.svg)](https://gitter.im/BVLC/caffe?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge&utm_content=badge)

Please join the [caffe-users group](https://groups.google.com/forum/#!forum/caffe-users) or [gitter chat](https://gitter.im/BVLC/caffe) to ask questions and talk about methods and models.
Framework development discussions and thorough bug reports are collected on [Issues](https://github.com/BVLC/caffe/issues).

Happy brewing!

## License and Citation

Caffe is released under the [BSD 2-Clause license](https://github.com/BVLC/caffe/blob/master/LICENSE).
The BAIR/BVLC reference models are released for unrestricted use.

Please cite Caffe in your publications if it helps your research:

    @article{jia2014caffe,
      Author = {Jia, Yangqing and Shelhamer, Evan and Donahue, Jeff and Karayev, Sergey and Long, Jonathan and Girshick, Ross and Guadarrama, Sergio and Darrell, Trevor},
      Journal = {arXiv preprint arXiv:1408.5093},
      Title = {Caffe: Convolutional Architecture for Fast Feature Embedding},
      Year = {2014}
    }
