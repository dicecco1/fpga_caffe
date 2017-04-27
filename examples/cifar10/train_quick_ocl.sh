#!/usr/bin/env sh
set -e

TOOLS=./build/tools

$TOOLS/caffe train \
  --solver=examples/cifar10/cifar10_quick_solver_ocl.prototxt $@ --ocl=1

# reduce learning rate by factor of 10 after 8 epochs
$TOOLS/caffe train \
  --solver=examples/cifar10/cifar10_quick_solver_lr1_ocl.prototxt \
  --snapshot=examples/cifar10/cifar10_quick_iter_4000.solverstate.h5 $@ --ocl=1
