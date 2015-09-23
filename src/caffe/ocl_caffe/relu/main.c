#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <unistd.h>
#include <assert.h>
#include <stdbool.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <CL/opencl.h>

////////////////////////////////////////////////////////////////////////////////

#define NO_NODES 4096 

#define NUM 1
#define CHANNELS 96 
#define HEIGHT 55
#define WIDTH 55
#define COUNT 290000 

////////////////////////////////////////////////////////////////////////////////

void ref_relu(float *input, float *output) {
  for (int i = 0; i < COUNT; ++i)
    if(input[i] >= 0)
      output[i] = input[i];
    else
      output[i] = 0;
}

int
load_file_to_memory(const char *filename, char **result)
{ 
  int size = 0;
  FILE *f = fopen(filename, "rb");
  if (f == NULL) 
  { 
    *result = NULL;
    return -1; // -1 means file opening fail 
  } 
  fseek(f, 0, SEEK_END);
  size = ftell(f);
  fseek(f, 0, SEEK_SET);
  *result = (char *)malloc(size+1);
  if (size != fread(*result, sizeof(char), size, f)) 
  { 
    free(*result);
    return -2; // -2 means file reading fail 
  } 
  fclose(f);
  (*result)[size] = 0;
  return size;
}

int main(int argc, char** argv)
{
  int err;                            // error code returned from api calls
  cl_platform_id platform_id;         // platform id
  cl_device_id device_id;             // compute device id 
  cl_context context;                 // compute context
  cl_command_queue commands;          // compute command queue
  cl_program program;                 // compute program
  cl_kernel kernel;                   // compute kernel

  size_t global[3];                   // global domain size for our calculation
  size_t local[3];                    // local domain size for our calculation

  char cl_platform_vendor[1001];
  char cl_platform_name[1001];
   

  cl_mem in_array;                     // device memory used for the input array
  //cl_mem synaptic_weights;             // device memory used for the input array
  cl_mem out_array;                    // device memory used for the output array
   
  if (argc != 2){
    printf("%s <inputfile>\n", argv[0]);
    return -1;
  }

	float input[COUNT];
	float output[COUNT];
  float sw_output[COUNT];

  for (int i = 0; i < COUNT; ++i)
    if(i % 2 == 0)
      input[i] = i;
    else
      input[i] = -i;

  //
  // Connect to first platform
  //
  err = clGetPlatformIDs(1,&platform_id,NULL);
  if (err != CL_SUCCESS)
  {
    printf("Error: Failed to find an OpenCL platform!\n");
    printf("Test failed\n");
    return -1;
  }
  err = clGetPlatformInfo(platform_id,CL_PLATFORM_VENDOR,1000,(void *)cl_platform_vendor,NULL);
  if (err != CL_SUCCESS)
  {
    printf("Error: clGetPlatformInfo(CL_PLATFORM_VENDOR) failed!\n");
    printf("Test failed\n");
    return -1;
  }
  printf("CL_PLATFORM_VENDOR %s\n",cl_platform_vendor);
  err = clGetPlatformInfo(platform_id,CL_PLATFORM_NAME,1000,(void *)cl_platform_name,NULL);
  if (err != CL_SUCCESS)
  {
    printf("Error: clGetPlatformInfo(CL_PLATFORM_NAME) failed!\n");
    printf("Test failed\n");
    return -1;
  }
  printf("CL_PLATFORM_NAME %s\n",cl_platform_name);
 
  // Connect to a compute device
  //
  int fpga = 0;
#if defined (FPGA_DEVICE)
  fpga = 1;
#endif
  err = clGetDeviceIDs(platform_id, fpga ? CL_DEVICE_TYPE_ACCELERATOR : CL_DEVICE_TYPE_CPU,
                       1, &device_id, NULL);
  if (err != CL_SUCCESS)
  {
    printf("Error: Failed to create a device group!\n");
    printf("Test failed\n");
    return -1;
  }
  
  //
  // Create a compute context 
  //
  context = clCreateContext(0, 1, &device_id, NULL, NULL, &err);
  if (!context)
  {
    printf("Error: Failed to create a compute context!\n");
    printf("Test failed\n");
    return -1;
  }

  // Create a command commands
  commands = clCreateCommandQueue(context, device_id, 0, &err);
  if (!commands)
  {
    printf("Error: Failed to create a command commands!\n");
    printf("Error: code %i\n",err);
    printf("Test failed\n");
    return -1;
  }

  int status;
  
  // Load binary from disk
  unsigned char *kernelbinary;
  char *xclbin=argv[1];
  printf("loading %s\n", xclbin);
  int n_i = load_file_to_memory(xclbin, (char **) &kernelbinary);
  if (n_i < 0) {
    printf("failed to load kernel from xclbin: %s\n", xclbin);
    printf("Test failed\n");
    return -1;
  }
  size_t n = n_i;
  // Create the compute program from offline
  program = clCreateProgramWithBinary(context, 1, &device_id, &n,
                                      (const unsigned char **) &kernelbinary, &status, &err);
  if ((!program) || (err!=CL_SUCCESS)) {
    printf("Error: Failed to create compute program from binary %d!\n", err);
    printf("Test failed\n");
    printf("err : %d %s\n",err,err);
  }

  // Build the program executable
  //
  err = clBuildProgram(program, 0, NULL, NULL, NULL, NULL);
  if (err != CL_SUCCESS)
  {
    size_t len;
    char buffer[2048];

    printf("Error: Failed to build program executable!\n");
    clGetProgramBuildInfo(program, device_id, CL_PROGRAM_BUILD_LOG, sizeof(buffer), buffer, &len);
    printf("%s\n", buffer);
    printf("Test failed\n");
    return -1;
  }

  // Create the compute kernel in the program we wish to run
  //
  kernel = clCreateKernel(program, "relu_layer", &err);
  if (!kernel || err != CL_SUCCESS)
  {
    printf("Error: Failed to create compute kernel!\n");
    printf("Test failed\n");
    return -1;
  }

  // Create the input and output arrays in device memory for our calculation
  //
  in_array = clCreateBuffer(context,  CL_MEM_READ_ONLY,  sizeof(float) * COUNT, NULL, NULL);
  out_array = clCreateBuffer(context, CL_MEM_WRITE_ONLY, sizeof(float) * COUNT, NULL, NULL);
  if (!in_array || !out_array)
  {
    printf("Error: Failed to allocate device memory!\n");
    printf("Test failed\n");
    return -1;
  }    
      
  //
  // Write our data set into the input array in device memory 
  //
  err = clEnqueueWriteBuffer(commands, in_array, CL_TRUE, 0, sizeof(float) * COUNT, input, 0, NULL, NULL);
  if (err != CL_SUCCESS)
  {
    printf("Error: Failed to write to source array a!\n");
    printf("Test failed\n");
    return -1;
  }
   
  // Set the arguments to our compute kernel
  //
  err = 0;
  err  = clSetKernelArg(kernel, 0, sizeof(cl_mem), &in_array);
  err |= clSetKernelArg(kernel, 1, sizeof(cl_mem), &out_array);
  if (err != CL_SUCCESS)
  {
    printf("Error: Failed to set kernel arguments! %d\n", err);
    printf("Test failed\n");
    return -1;
  }

  // Execute the kernel over the entire range of our 1d input data set
  // using the maximum number of work group items for this device
  //

  global[0] = 71;
  global[1] = 1;
  global[2] = 1;
  local[0] = 1;
  local[1] = 1;
  local[2] = 1;

  err = clEnqueueNDRangeKernel(commands, kernel, 3, NULL, 
                                (size_t*)&global, (size_t*)&local, 0, NULL, NULL);

  if (err)
  {
    printf("Error: Failed to execute kernel! %d\n", err);
    printf("Test failed\n");
    return -1;
  }

  // Read back the results from the device to verify the output
  //
  cl_event readevent;
  err = clEnqueueReadBuffer( commands, out_array, CL_TRUE, 0, sizeof(float) * COUNT, output, 0, NULL, &readevent );  
  if (err != CL_SUCCESS)
  {
    printf("Error: Failed to read output array! %d\n", err);
    printf("Test failed\n");
    return -1;
  }
  int correct = 0;

  ref_relu(input, sw_output);

  clWaitForEvents(1, &readevent);
  for (int i = 0; i < COUNT; ++i)
 	  if (output[i] == sw_output[i]) 
  	  correct++;


  clReleaseMemObject(in_array);
  clReleaseMemObject(out_array);
  clReleaseProgram(program);
  clReleaseKernel(kernel);
  clReleaseCommandQueue(commands);
  clReleaseContext(context);

  if(correct == COUNT){
    printf("Test passed!\n");
    return EXIT_SUCCESS;
  }
  else{
    printf("Test failed\n");
    return -1;
  }
}
