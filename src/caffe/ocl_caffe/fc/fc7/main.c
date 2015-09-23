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
#include "fc7_layer.h"

////////////////////////////////////////////////////////////////////////////////

// Use a static matrix for simplicity
//
#define DATA_SIZE1        M_ * K_
#define FILTER_SIZE1      K_ * N_
#define OUTPUT_SIZE1      M_ * N_

////////////////////////////////////////////////////////////////////////////////

void gemm(float *inputA, float *inputB, float *output) {
  for (int i = 0; i < M_; ++i)
    for (int j = 0; j < N_; ++j)
      for (int k = 0; k < K_; ++k)
        output[i * N_ + j] += inputA[i * K_ + k] * inputB[k * N_ + j];
}

void gemm_t(float *inputA, float *inputB_t, float *output) {
  for (int i = 0; i < M_; ++i)
    for (int j = 0; j < N_; ++j)
      for (int k = 0; k < K_; ++k)
        output[i * N_ + j] += inputA[i * K_ + k] * inputB_t[j * K_ + k];
}

void transpose(float *input, float *output)
{
  for (int i = 0; i < K_; ++i)
    for (int j = 0; j < N_; ++j)
      output[j * K_ + i] = input[i * N_ + j];  
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
     
  float a1[DATA_SIZE1];               // original data set given to device
  //float b1[FILTER_SIZE1];             // original data set given to device
  float *b1 = (float *)malloc(sizeof(float) * FILTER_SIZE1);
  float *b_t = (float *)malloc(sizeof(float) * FILTER_SIZE1);
  float c1[OUTPUT_SIZE1];
  float results1[OUTPUT_SIZE1];       // results returned from device
  float sw_results1[OUTPUT_SIZE1];     // results returned from device
  float sw_results2[OUTPUT_SIZE1];
  unsigned int correct;               // number of correct results returned

  size_t global[3];                   // global domain size for our calculation
  size_t local[3];                    // local domain size for our calculation

  cl_platform_id platform_id;         // platform id
  cl_device_id device_id;             // compute device id 
  cl_context context;                 // compute context
  cl_command_queue commands;          // compute command queue
  cl_program program;                 // compute program
  cl_kernel kernel;                   // compute kernel
   
  char cl_platform_vendor[1001];
  char cl_platform_name[1001];
   
  cl_mem input_a;                     // device memory used for the input array
  cl_mem input_b;                     // device memory used for the input array
  cl_mem output;                      // device memory used for the output array
   
  if (argc != 2){
    printf("%s <inputfile>\n", argv[0]);
    return EXIT_FAILURE;
  }

  // Fill our data sets with pattern
  //
  int i = 0;
  for(i = 0; i < DATA_SIZE1; i++) {
    a1[i] = (float)(rand() % 100 + 1);
  }
  for(i = 0; i < OUTPUT_SIZE1; i++) {
    results1[i] = 0;
    sw_results1[i] = 0;
    sw_results2[i] = 0;
  }
  for(i = 0; i < FILTER_SIZE1; i++) {
    b1[i] = (float)(rand() % 100 + 1);
  }
  for(i = 0; i < OUTPUT_SIZE1; i++) {
    c1[i] = (float)0;
  }
  
  transpose(b1, b_t);
//  gemm(a1, b1, sw_results2);

  // Connect to first platform
  //
  err = clGetPlatformIDs(1,&platform_id,NULL);
  if (err != CL_SUCCESS)
  {
    printf("Error: Failed to find an OpenCL platform!\n");
    printf("Test failed\n");
    return EXIT_FAILURE;
  }
  err = clGetPlatformInfo(platform_id,CL_PLATFORM_VENDOR,1000,(void *)cl_platform_vendor,NULL);
  if (err != CL_SUCCESS)
  {
    printf("Error: clGetPlatformInfo(CL_PLATFORM_VENDOR) failed!\n");
    printf("Test failed\n");
    return EXIT_FAILURE;
  }
  printf("CL_PLATFORM_VENDOR %s\n",cl_platform_vendor);
  err = clGetPlatformInfo(platform_id,CL_PLATFORM_NAME,1000,(void *)cl_platform_name,NULL);
  if (err != CL_SUCCESS)
  {
    printf("Error: clGetPlatformInfo(CL_PLATFORM_NAME) failed!\n");
    printf("Test failed\n");
    return EXIT_FAILURE;
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
    return EXIT_FAILURE;
  }
  
  // Create a compute context 
  //
  context = clCreateContext(0, 1, &device_id, NULL, NULL, &err);
  if (!context)
  {
    printf("Error: Failed to create a compute context!\n");
    printf("Test failed\n");
    return EXIT_FAILURE;
  }

  // Create a command commands
  //
  commands = clCreateCommandQueue(context, device_id, 0, &err);
  if (!commands)
  {
    printf("Error: Failed to create a command commands!\n");
    printf("Error: code %i\n",err);
    printf("Test failed\n");
    return EXIT_FAILURE;
  }

  int status;

  // Create Program Objects
  //
  
  // Load binary from disk
  unsigned char *kernelbinary;
  char *xclbin=argv[1];
  printf("loading %s\n", xclbin);
  int n_i = load_file_to_memory(xclbin, (char **) &kernelbinary);
  if (n_i < 0) {
    printf("failed to load kernel from xclbin: %s\n", xclbin);
    printf("Test failed\n");
    return EXIT_FAILURE;
  }
  size_t n = n_i;
  // Create the compute program from offline
  program = clCreateProgramWithBinary(context, 1, &device_id, &n,
                                      (const unsigned char **) &kernelbinary, &status, &err);
  if ((!program) || (err!=CL_SUCCESS)) {
    printf("Error: Failed to create compute program from binary %d!\n", err);
    printf("Test failed\n");
    return EXIT_FAILURE;
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
    return EXIT_FAILURE;
  }

  // Create the compute kernel in the program we wish to run
  //
  kernel = clCreateKernel(program, "fc7_layer", &err);
  if (!kernel || err != CL_SUCCESS)
  {
    printf("Error: Failed to create compute kernel!\n");
    printf("Test failed\n");
    return EXIT_FAILURE;
  }

  // Create the input and output arrays in device memory for our calculation
  //
  input_b = clCreateBuffer(context,  CL_MEM_READ_ONLY,  sizeof(float) * FILTER_SIZE1, NULL, NULL);
  input_a = clCreateBuffer(context,  CL_MEM_READ_ONLY,  sizeof(float) * DATA_SIZE1, NULL, NULL);
  output = clCreateBuffer(context, CL_MEM_WRITE_ONLY, sizeof(float) * OUTPUT_SIZE1, NULL, NULL);
  if (!input_a || !input_b || !output)
  {
    printf("Error: Failed to allocate device memory!\n");
    printf("Test failed\n");
    return EXIT_FAILURE;
  }    
    
  // Write our data set into the input array in device memory 
  //
  err = clEnqueueWriteBuffer(commands, input_b, CL_TRUE, 0, sizeof(float) * FILTER_SIZE1, b_t, 0, NULL, NULL);
  if (err != CL_SUCCESS)
  {
    printf("Error: Failed to write to source array a!\n");
    printf("Test failed\n");
    return EXIT_FAILURE;
  }

  // Write our data set into the input array in device memory 
  //
  err = clEnqueueWriteBuffer(commands, input_a, CL_TRUE, 0, sizeof(float) * DATA_SIZE1, a1, 0, NULL, NULL);
  if (err != CL_SUCCESS)
  {
    printf("Error: Failed to write to source array b!\n");
    printf("Test failed\n");
    return EXIT_FAILURE;
  }
  err = clEnqueueWriteBuffer(commands, output, CL_TRUE, 0, sizeof(float) * OUTPUT_SIZE1, c1, 0, NULL, NULL);
    
  // Set the arguments to our compute kernel
  //

  err = 0;
  err  = clSetKernelArg(kernel, 0, sizeof(cl_mem), &input_a);
  err |= clSetKernelArg(kernel, 1, sizeof(cl_mem), &input_b);
  err |= clSetKernelArg(kernel, 2, sizeof(cl_mem), &output);
  if (err != CL_SUCCESS)
  {
    printf("Error: Failed to set kernel arguments! %d\n", err);
    printf("Test failed\n");
    return EXIT_FAILURE;
  }

  // Execute the kernel over the entire range of our 1d input data set
  // using the maximum number of work group items for this device
  //

#ifdef C_KERNEL
  err = clEnqueueTask(commands, kernel, 0, NULL, NULL);
#else
  global[0] = 8;
  global[1] = 1;
  global[2] = 1;
  local[0] = 1;
  local[1] = 1;
  local[2] = 1;
  err = clEnqueueNDRangeKernel(commands, kernel, 3, NULL, 
                               (size_t*)&global, (size_t*)&local, 0, NULL, NULL);
#endif
  if (err)
  {
    printf("Error: Failed to execute kernel! %d\n", err);
    printf("Test failed\n");
    return EXIT_FAILURE;
  }

  // Read back the results from the device to verify the output
  //
  cl_event readevent;
  err = clEnqueueReadBuffer( commands, output, CL_TRUE, 0, sizeof(float) * OUTPUT_SIZE1, results1, 0, NULL, &readevent );  
  if (err != CL_SUCCESS)
  {
    printf("Error: Failed to read output array! %d\n", err);
    printf("Test failed\n");
    return EXIT_FAILURE;
  }

  clWaitForEvents(1, &readevent);
  gemm(a1, b1, sw_results1);   
//  gemm(a1, b_t, sw_results2);

  // Validate our results
  //
  correct = 0;
  int correct2 = 0; 
  for (i = 0;i < OUTPUT_SIZE1; i++) 
    if(results1[i] == sw_results1[i])
      correct++;
    else
      printf("%f\t%f\n", results1[i], sw_results1[i]);
 
//  for (i = 0; i < OUTPUT_SIZE1; ++i)
//    if(results1[i] == sw_results2[i])
//      correct2++;
//    else
//      printf("%f\t%f\n", results1[i], sw_results2[i]);


  // Print a brief summary detailing the results
  //
  printf("Computed '%d/%d' correct values!\n", correct, OUTPUT_SIZE1);
//  printf("Computed '%d/%d' correct values! (Transposed)\n", correct2, OUTPUT_SIZE1);
  // Shutdown and cleanup
  //
  clReleaseMemObject(input_a);
  clReleaseMemObject(input_b);
  clReleaseMemObject(output);
  clReleaseProgram(program);
  clReleaseKernel(kernel);
  clReleaseCommandQueue(commands);
  clReleaseContext(context);

  if(correct == OUTPUT_SIZE1){
    printf("Test passed!\n");
    return EXIT_SUCCESS;
  }
  else{
    printf("Test failed\n");
    return EXIT_FAILURE;
  }
}
