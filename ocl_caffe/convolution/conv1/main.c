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
#include "conv1_layer.h"

////////////////////////////////////////////////////////////////////////////////

// Use a static matrix for simplicity
//
#define DATA_SIZE1        TOP_NUM * IN_CHANNEL * (NUM_DATA_ROWS) * (NUM_DATA_COLS)
#define FILTER_SIZE1      K_NUM * K_CHANNEL * NUM_MASK_ROWS * NUM_MASK_COLS
#define OUTPUT_SIZE1      TOP_NUM * OUT_CHANNEL * NUM_OUT_ROWS  * NUM_OUT_COLS

////////////////////////////////////////////////////////////////////////////////

void ref_conv(float *input, float *weights, float *output) {
  int o_head, k_head;
  int out_idx, in_idx, k_idx;
  // Convolution
  for (int i = 0; i < TOP_NUM * OUT_CHANNEL * NUM_OUT_ROWS * NUM_OUT_COLS; ++i)
    output[i] = 0;
  for (int n = 0; n < TOP_NUM; n++) {
    for (int g = 0; g < GROUPS; g++) {
      o_head = O_G * g;
      k_head = K_G * g;
      for (int o = 0; o < O_G; o++) {
        for (int k = 0; k < K_G; k++) {
          for (int y = 0; y < NUM_OUT_ROWS; y++) {
            for (int x = 0; x < NUM_OUT_COLS; x++) {
              for (int p = 0; p < NUM_MASK_ROWS; p++) {
                for (int q = 0; q < NUM_MASK_COLS; q++) {
                  int in_y = y * STRIDE - PAD + p;
                  int in_x = x * STRIDE - PAD + q;
                  if (in_y >= 0 && in_y < NUM_DATA_ROWS
                    && in_x >= 0 && in_x < NUM_DATA_COLS) {
                    out_idx = (((n * OUT_CHANNEL) + o + o_head) * NUM_OUT_ROWS + y) * NUM_OUT_COLS + x;
                    in_idx = (((n * IN_CHANNEL) + k + k_head) * NUM_DATA_ROWS + in_y) * NUM_DATA_COLS + in_x;
                    k_idx = (((o + o_head) * K_CHANNEL + k) * NUM_MASK_ROWS + p) * NUM_MASK_COLS + q;
                    output[out_idx] += input[in_idx] * weights[k_idx];
                  }
                }
              }
            }
          }
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
     
  float a1[DATA_SIZE1];               // original data set given to device
  float b1[FILTER_SIZE1];             // original data set given to device
  float c1[OUTPUT_SIZE1];
  float results1[OUTPUT_SIZE1];       // results returned from device
  float sw_results1[OUTPUT_SIZE1];     // results returned from device

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
//    sw_results1[i] = FILTER_SIZE1;
  }
  for(i = 0; i < FILTER_SIZE1; i++) {
    b1[i] = (float)(rand() % 100 + 1);
  }
  for(i = 0; i < OUTPUT_SIZE1; i++) {
    c1[i] = (float)0;
  }

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
  kernel = clCreateKernel(program, "conv1_layer", &err);
  if (!kernel || err != CL_SUCCESS)
  {
    printf("Error: Failed to create compute kernel!\n");
    printf("Test failed\n");
    return EXIT_FAILURE;
  }

  // Create the input and output arrays in device memory for our calculation
  //
  input_a = clCreateBuffer(context,  CL_MEM_READ_ONLY,  sizeof(float) * DATA_SIZE1, NULL, NULL);
  input_b = clCreateBuffer(context,  CL_MEM_READ_ONLY,  sizeof(float) * FILTER_SIZE1, NULL, NULL);
  output = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(float) * OUTPUT_SIZE1, NULL, NULL);
  if (!input_a || !input_b || !output)
  {
    printf("Error: Failed to allocate device memory!\n");
    printf("Test failed\n");
    return EXIT_FAILURE;
  }    
    
  // Write our data set into the input array in device memory 
  //
  err = clEnqueueWriteBuffer(commands, input_a, CL_TRUE, 0, sizeof(float) * DATA_SIZE1, a1, 0, NULL, NULL);
  if (err != CL_SUCCESS)
  {
    printf("Error: Failed to write to source array a!\n");
    printf("Test failed\n");
    return EXIT_FAILURE;
  }

  // Write our data set into the input array in device memory 
  //
  err = clEnqueueWriteBuffer(commands, input_b, CL_TRUE, 0, sizeof(float) * FILTER_SIZE1, b1, 0, NULL, NULL);
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
  global[0] = 96;
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
  ref_conv(a1, b1, sw_results1);    
  printf("A\n");
/*  for (i=0;i<DATA_SIZE1;i++) {
    printf("%f ",a1[i]);
    if (((i+1) % NUM_DATA_ROWS) == 0)
      printf("\n");
  }*/
  printf("B\n");
/*  for (i=0;i< FILTER_SIZE1;i++) {
    printf("%f ",b1[i]);
    if (((i+1) % NUM_MASK_ROWS) == 0)
      printf("\n");
  }
  */
/*  printf("res\n");
  for (i=0;i< OUTPUT_SIZE1;i++) {
    printf("%f ",results1[i]);
    if (((i+1) % NUM_OUT_ROWS) == 0)
      printf("\n");
  }
*/
    
  // Validate our results
  //
  correct = 0;
  /* for(i = 0; i < OUTPUT_SIZE1; i++)
  {
    int row = i/MATRIX_RANK;
    int col = i%MATRIX_RANK;
    int running = 0;
    int index;
    for (index=0;index<MATRIX_RANK;index++) {
      int aIndex = row*MATRIX_RANK + index;
      int bIndex = col + index*MATRIX_RANK;
      running += a[aIndex] * b[bIndex];
    }
    sw_results[i] = running;
    }*/
    
  for (i = 0;i < OUTPUT_SIZE1; i++) 
    if(results1[i] == sw_results1[i])
      correct++;
  printf("Software\n");
/*  for (i=0;i<OUTPUT_SIZE1;i++) {
    //printf("%0.2f ",sw_results[i]);
    printf("%f ",sw_results1[i]);
    if (((i+1) % NUM_OUT_ROWS) == 0)
      printf("\n");
  }*/
    
    
  // Print a brief summary detailing the results
  //
  printf("Computed '%d/%d' correct values!\n", correct, OUTPUT_SIZE1);
    
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
