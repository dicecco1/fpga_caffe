############################################################
## This file is generated automatically by Vivado HLS.
## Please DO NOT edit it.
## Copyright (C) 1986-2017 Xilinx, Inc. All Rights Reserved.
############################################################
open_project mul_prj
set_top mul
add_files mul.cpp
open_solution "solution1"
set_part {xcku115-flva1517-2-e}
create_clock -period 4 -name default
#source "./adders_prj/solution1/directives.tcl"
#csim_design
csynth_design
#cosim_design
export_design -flow impl -rtl verilog -format ip_catalog