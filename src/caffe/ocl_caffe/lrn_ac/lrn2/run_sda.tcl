
# Define the project for SDAccel
#create_project -name prj_ocl_pooling  -dir . -force
create_solution -name prj_lrn2_ac_layer -dir . -force
#set_property platform vc690-admpcie7v3-1ddr-gen2 [current_project]
add_device -vbnv xilinx:adm-pcie-7v3:1ddr:1.0

# Host Compiler Flags
set_property -name host_cflags -value "-g -Wall -D FPGA_DEVICE" -objects [current_project]

# Host Source Files
add_files "main.c"
add_files "lrn2_ac_layer.h"
set_property file_type "c header files" [get_files "lrn2_ac_layer.h"]

# Kernel Definition
create_kernel lrn2_ac_layer -type clc
add_files -kernel [get_kernels lrn2_ac_layer] "lrn2_ac_layer.cl"

# Define Binary Containers
#set_property max_memory_ports true [get_kernels lrn2_ac_layer]
create_opencl_binary lrn2_ac_layer
set_property region "OCL_REGION_0" [get_opencl_binary lrn2_ac_layer]
create_compute_unit -opencl_binary [get_opencl_binary lrn2_ac_layer] -kernel [get_kernels lrn2_ac_layer] -name ocl_lrn1
create_compute_unit -opencl_binary [get_opencl_binary lrn2_ac_layer] -kernel [get_kernels lrn2_ac_layer] -name ocl_lrn2
create_compute_unit -opencl_binary [get_opencl_binary lrn2_ac_layer] -kernel [get_kernels lrn2_ac_layer] -name ocl_lrn3
create_compute_unit -opencl_binary [get_opencl_binary lrn2_ac_layer] -kernel [get_kernels lrn2_ac_layer] -name ocl_lrn4


#Compile the design for CPU based emulation
compile_emulation -flow cpu -opencl_binary [get_opencl_binary lrn2_ac_layer]

# Run the compiled application in CPU based emulation mode
run_emulation -flow cpu -args "lrn2_ac_layer.xclbin"

report_estimate

# Compile the application to run on the accelerator card
build_system
#
# Package the application binaries
package_system

