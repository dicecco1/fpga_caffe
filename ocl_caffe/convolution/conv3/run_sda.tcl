
# Define the project for SDAccel
#create_project -name prj_ocl_pooling  -dir . -force
create_solution -name prj_ocl_conv3 -dir . -force
#set_property platform vc690-admpcie7v3-1ddr-gen2 [current_project]
add_device -vbnv xilinx:adm-pcie-7v3:1ddr:1.0

# Host Compiler Flags
set_property -name host_cflags -value "-g -Wall -D FPGA_DEVICE" -objects [current_project]

# Host Source Files
add_files "main.c"
add_files "conv3_layer.h"
set_property file_type "c header files" [get_files "conv3_layer.h"]

# Kernel Definition
create_kernel conv3_layer -type clc
add_files -kernel [get_kernels conv3_layer] "conv3_layer.cl"

# Define Binary Containers
#set_property max_memory_ports true [get_kernels conv3_layer]
create_opencl_binary conv3_layer
set_property region "OCL_REGION_0" [get_opencl_binary conv3_layer]
create_compute_unit -opencl_binary [get_opencl_binary conv3_layer] -kernel [get_kernels conv3_layer] -name ocl_conv1
create_compute_unit -opencl_binary [get_opencl_binary conv3_layer] -kernel [get_kernels conv3_layer] -name ocl_conv2
create_compute_unit -opencl_binary [get_opencl_binary conv3_layer] -kernel [get_kernels conv3_layer] -name ocl_conv3
create_compute_unit -opencl_binary [get_opencl_binary conv3_layer] -kernel [get_kernels conv3_layer] -name ocl_conv4
create_compute_unit -opencl_binary [get_opencl_binary conv3_layer] -kernel [get_kernels conv3_layer] -name ocl_conv5
create_compute_unit -opencl_binary [get_opencl_binary conv3_layer] -kernel [get_kernels conv3_layer] -name ocl_conv6
#create_compute_unit -opencl_binary [get_opencl_binary conv3_layer] -kernel [get_kernels conv3_layer] -name ocl_conv7
#create_compute_unit -opencl_binary [get_opencl_binary conv3_layer] -kernel [get_kernels conv3_layer] -name ocl_conv8

#Compile the design for CPU based emulation
compile_emulation -flow cpu -opencl_binary [get_opencl_binary conv3_layer]

# Run the compiled application in CPU based emulation mode
run_emulation -flow cpu -args "conv3_layer.xclbin"

report_estimate

# Compile the application to run on the accelerator card
build_system
#
# Package the application binaries
package_system

