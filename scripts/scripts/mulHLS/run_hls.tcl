############################################################
## This file is generated automatically by Vivado HLS.
## Please DO NOT edit it.
## Copyright (C) 1986-2016 Xilinx, Inc. All Rights Reserved.
############################################################
open_project mul_prj
set_top mul
add_files half.hpp
add_files mul.cpp
add_files mul.hpp
add_files para.hpp
add_files -tb mul_test.c
open_solution "solution1"
set_part {xc7a75tlftg256-2l}
create_clock -period 4 -name default
#source "./adders_prj/solution1/directives.tcl"
# csim_design -compiler gcc
csynth_design
# cosim_design
export_design -flow impl -rtl verilog -format ip_catalog
