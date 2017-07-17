#!/usr/bin/env sh
set -e

TOOLS=./build/tools

$TOOLS/caffe train \
    --solver=examples/cifar10/cifar10_full_solver_ocl_hwcn.prototxt $@ --ocl=1

# reduce learning rate by factor of 10
$TOOLS/caffe train \
    --solver=examples/cifar10/cifar10_full_solver_lr1_ocl_hwcn.prototxt \
    --snapshot=examples/cifar10/cifar10_full_iter_8000.solverstate.h5 $@ --ocl=1
