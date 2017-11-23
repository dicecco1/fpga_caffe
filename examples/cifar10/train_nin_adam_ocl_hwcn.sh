#!/usr/bin/env sh
set -e

TOOLS=./build/tools

$TOOLS/caffe train \
    --solver=examples/cifar10/nin_solver_adam_ocl_hwcn.prototxt $@ --ocl=1

$TOOLS/caffe train \
    --solver=examples/cifar10/nin_solver_adam_lr1_ocl_hwcn.prototxt \
    --snapshot=examples/cifar10/cifar10_nin_adam_iter_100000.solverstate.h5 $@ --ocl=1
