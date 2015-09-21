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
#include "lrn2_ac_layer.h"


////////////////////////////////////////////////////////////////////////////////
void ref_lrn_ac(float *input, float *output) {
  for (int n = 0; n < NUM_OF_BOTTOM_BLOBS; ++n) {
    for (int c = 0; c < NUM_CHANNELS; ++c) {
      for (int h = 0; h < IHEIGHT; ++h) {
        for (int w = 0; w < IWIDTH; ++w) {
          int c_start = c - (LOCAL_SIZE - 1) / 2;
          int c_end = fmin(c_start + LOCAL_SIZE, NUM_CHANNELS);
          c_start = fmax(c_start, 0);
          float scale = 1.0;
          for (int i = c_start; i < c_end; ++i) {
            float value = input[((n * NUM_CHANNELS + i) * IHEIGHT + h) * IWIDTH + w];
            scale += value * value * ALPHA / LOCAL_SIZE;
          }
          output[((n * NUM_CHANNELS + c) * OHEIGHT + h) * OWIDTH + w] = 
            input[((n * NUM_CHANNELS + c) * IHEIGHT + h) * IWIDTH + w] / pow(scale, BETA);
        }
      }
    }
  }
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
   
  float input[ISIZE];
  float output[OSIZE];
  float swout[OSIZE]; 

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
  cl_uint frequency;
  cl_uint num_cu;

  cl_mem chin;                     // device memory used for the chin array

  cl_mem chout;                    // device memory used for the chout array
     
  if (argc != 2){
    printf("%s <inputfile>\n", argv[0]);
    return EXIT_FAILURE;
  }

  // Fill input with data pattern
  float tmp;
  for (int x = 0; x < ISIZE; ++x) {
    if(x % 2 == 0)
      input[x] = x;
    else
      input[x] = -x;
  }

  ref_lrn_ac(input, swout);

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
 
  err = clGetDeviceInfo(device_id, CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(cl_uint), &num_cu, NULL);
  err = clGetDeviceInfo(device_id, CL_DEVICE_MAX_CLOCK_FREQUENCY, sizeof(cl_uint), &frequency, NULL);

  printf("Number of Compute Units is %d\n", num_cu);
  printf("Maximum Frequency is %d\n", frequency);

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
  kernel = clCreateKernel(program, "lrn2_ac_layer", &err);
  if (!kernel || err != CL_SUCCESS)
  {
    printf("Error: Failed to create compute kernel!\n");
    printf("Test failed\n");
    return EXIT_FAILURE;
  }

  // Create the input and output arrays in device memory for our calculation
  //Inputs
  chin = clCreateBuffer(context,  CL_MEM_READ_ONLY,  sizeof(float)*(ISIZE + (LOCAL_SIZE + 1) * IHEIGHT * IWIDTH), NULL, NULL);
 
  // Outputs
  chout = clCreateBuffer(context, CL_MEM_WRITE_ONLY, sizeof(float)*(OSIZE + (LOCAL_SIZE + 1) * OHEIGHT * OWIDTH), NULL, NULL);

  if (!chin || !chout)
  {
    printf("Error: Failed to allocate device memory!\n");
    printf("Test failed\n");
    return EXIT_FAILURE;
  }    
    
  // Write our data set into the input array in device memory 
  //
  err = clEnqueueWriteBuffer(commands, chin, CL_TRUE, 0, sizeof(float)*(ISIZE), input, 0, NULL, NULL);
  if (err != CL_SUCCESS)
  {
    printf("Error: Failed to write to source array chin!\n");
    printf("Test failed\n");
    return EXIT_FAILURE;
  }

   
  // Set the arguments to our compute kernel
  //
  err = 0;
  err  = clSetKernelArg(kernel, 0, sizeof(cl_mem), &chin);
  err |= clSetKernelArg(kernel, 1, sizeof(cl_mem), &chout);

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
  global[0] = 256;
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
  err = clEnqueueReadBuffer( commands, chout, CL_TRUE, 0, sizeof(float)*(OSIZE), output, 0, NULL, &readevent );  
  if (err != CL_SUCCESS)
  {
    printf("Error: Failed to read output array! %d\n", err);
    printf("Test failed\n");
    return EXIT_FAILURE;
  }

  clWaitForEvents(1, &readevent);  
  int err_cnt = 0;

  int maxdiff = 0;
   // compare
  
  for (int c = 0; c < NUM_CHANNELS; ++c) {
    for (int y = 0; y < OHEIGHT; ++y) {
      for (int x = 0; x < OWIDTH; ++x) {
        if (abs(swout[(c * OHEIGHT + y) * OWIDTH + x] - output[(c * OHEIGHT + y) * OWIDTH + x]) > 1E-5) {
          printf("SW VALUE %f, HW VALUE %f\n", swout[(c * OHEIGHT + y) * OWIDTH + x], output[(c * OHEIGHT + y) * OWIDTH + x]);
          err_cnt++;
          printf("c %d x %d y %d\n", c, x, y);
        }
        printf("SW VALUE %f, HW VALUE %f\n", swout[(c * OHEIGHT + y) * OWIDTH + x], output[(c * OHEIGHT + y) * OWIDTH + x]);
      }
    }
  }
    
    
  // Shutdown and cleanup
  //
  clReleaseMemObject(chin);
  clReleaseMemObject(chout);
  clReleaseProgram(program);
  clReleaseKernel(kernel);
  clReleaseCommandQueue(commands);
  clReleaseContext(context);

  if(err_cnt == 0){
    printf("Test passed!\n");
    return EXIT_SUCCESS;
  }
  else{
    printf("Test failed\n");
    return EXIT_FAILURE;
  }
}
