
# Define the project for SDAccel
#create_project -name prj_ocl_pooling  -dir . -force
create_solution -name prj_ocl_fc6 -dir . -force
#set_property platform vc690-admpcie7v3-1ddr-gen2 [current_project]
add_device -vbnv xilinx:adm-pcie-7v3:1ddr:1.0

# Host Compiler Flags
set_property -name host_cflags -value "-g -Wall -D FPGA_DEVICE" -objects [current_project]

# Host Source Files
add_files "main.c"
add_files "fc6_layer.h"
set_property file_type "c header files" [get_files "fc6_layer.h"]

# Kernel Definition
create_kernel fc6_layer -type clc
add_files -kernel [get_kernels fc6_layer] "fc6_layer.cl"

# Define Binary Containers
#set_property max_memory_ports true [get_kernels fc6_layer]
create_opencl_binary fc6_layer
set_property region "OCL_REGION_0" [get_opencl_binary fc6_layer]
create_compute_unit -opencl_binary [get_opencl_binary fc6_layer] -kernel [get_kernels fc6_layer] -name ocl_fc1
create_compute_unit -opencl_binary [get_opencl_binary fc6_layer] -kernel [get_kernels fc6_layer] -name ocl_fc2
#create_compute_unit -opencl_binary [get_opencl_binary fc6_layer] -kernel [get_kernels fc6_layer] -name ocl_pooling3
#create_compute_unit -opencl_binary [get_opencl_binary fc6_layer] -kernel [get_kernels fc6_layer] -name ocl_pooling4
#Compile the design for CPU based emulation
compile_emulation -flow cpu -opencl_binary [get_opencl_binary fc6_layer]

# Run the compiled application in CPU based emulation mode
run_emulation -flow cpu -args "fc6_layer.xclbin"

report_estimate

# Compile the application to run on the accelerator card
build_system
#
# Package the application binaries
package_system

