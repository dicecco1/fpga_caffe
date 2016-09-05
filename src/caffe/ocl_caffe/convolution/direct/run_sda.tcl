# Define the project for SDAccel
create_solution -name prj_direct -dir . -force
#set_property platform vc690-admpcie7v3-1ddr-gen2 [current_project]
add_device -vbnv xilinx:adm-pcie-7v3:1ddr:1.0

# Host Compiler Flags
set_property -name host_cflags -value "-g -Wall -D FPGA_DEVICE -D C_KERNEL" -objects [current_project]

# Host Source Files
add_files "main.c"

# Kernel Definition
create_kernel direct_conv -type c
add_files -kernel [get_kernels direct_conv] "direct_conv.c"

# Define Binary Containers
create_opencl_binary direct_conv
set_property region "OCL_REGION_0" [get_opencl_binary direct_conv]
create_compute_unit -opencl_binary [get_opencl_binary direct_conv] -kernel [get_kernels direct_conv] -name ocl_conv1
#create_compute_unit -opencl_binary [get_opencl_binary direct_conv] -kernel [get_kernels direct_conv] -name ocl_conv2

#Compile the design for CPU based emulation
compile_emulation -flow cpu -opencl_binary [get_opencl_binary direct_conv]

# Run the compiled application in CPU based emulation mode
run_emulation -flow cpu -args "direct_conv.xclbin 1 64 2 1 64 224 224 224 2"
run_emulation -flow cpu -args "direct_conv.xclbin 1 64 2 4 16 112 112 128 2"
run_emulation -flow cpu -args "direct_conv.xclbin 1 512 2 256 2 14 14 16 2"
run_emulation -flow cpu -args "direct_conv.xclbin 1 256 1 256 1 16 16 16 1"

#run_emulation -flow cpu -args "direct_conv.xclbin 1 2 2 2 1 4 4 4 1"

report_estimate

# Compile the application to run on the accelerator card
build_system

# Package the application binaries
package_system
