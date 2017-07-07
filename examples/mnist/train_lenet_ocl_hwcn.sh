#!/usr/bin/env sh
set -e

./build/tools/caffe train --solver=examples/mnist/lenet_solver_ocl_hwcn.prototxt $@ --ocl=1
