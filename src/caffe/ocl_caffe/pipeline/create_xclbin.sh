#!/bin/bash

xocc -g --xdevice xilinx:adm-pcie-7v3:1ddr:1.0 -t sw_emu kernels/UCHPC_Pipeline.cl -o binary/UCHPC_Pipeline_SW.xclbin
