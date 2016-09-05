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
#include <limits.h>

////////////////////////////////////////////////////////////////////////////////

#define NUM_MASK_ROWS 3
#define NUM_MASK_COLS 3
#define STRIDE 1
#define PAD 1

/* Reference convolution implementation */

void ref_conv(float *input, float *weights, float *bias, float *ocl_output, 
    int groups, int inchannel, int outchannel, int xsize, int ysize, 
    int numimages) {
  int o_head, k_head;
  int out_idx, in_idx, k_idx;
  // Convolution
  for (int i = 0; i < numimages * outchannel * ysize * xsize; ++i)
    ocl_output[i] = 0;
  for (int n = 0; n < numimages; n++) {
    for (int g = 0; g < groups; g++) {
      o_head = (outchannel / groups) * g;
      k_head = (inchannel / groups) * g;
      int o_g = outchannel / groups;
      int k_g = inchannel / groups;
      for (int o = 0; o < o_g; o++) {
        for (int k = 0; k < k_g; k++) {
          for (int y = 0; y < ysize; y++) {
            for (int x = 0; x < xsize; x++) {
              for (int p = 0; p < NUM_MASK_ROWS; p++) {
                for (int q = 0; q < NUM_MASK_COLS; q++) {
                  int in_y = y * STRIDE - PAD + p;
                  int in_x = x * STRIDE - PAD + q;
                  if (in_y >= 0 && in_y < ysize
                    && in_x >= 0 && in_x < xsize) {
                    out_idx = (((n * outchannel) + o + o_head) * ysize + y) 
                      * xsize + x;
                    in_idx = (((n * inchannel) + k + k_head) * ysize + in_y) 
                      * xsize + in_x;
                    k_idx = (((o + o_head) * (k_g) + k) * NUM_MASK_ROWS + p) 
                      * NUM_MASK_COLS + q;
                    ocl_output[out_idx] += input[in_idx] * weights[k_idx];
                  }
                }
              }
            }
          }
        }
      }
    }
  }
  for (int n = 0; n < numimages; n++) {
    for (int o = 0; o < outchannel; ++o) {
      for (int y = 0; y < ysize; ++y) {
        for (int x = 0; x < xsize; ++x) {
          out_idx = (((n * outchannel) + o) * ysize + y) * xsize + x;
          ocl_output[out_idx] += bias[o];
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
  if (argc != 11){
    printf("%s <inputfile> numgroups inchannels outchannels burstchannels rpo" 
        " dim tilesize padtsize numimages\n", argv[0]);
    return EXIT_FAILURE;
  }

  int err;                            // error code returned from api calls 
  
  // Get Parameters for kernel execution
  int group = 0;
  int numgroups = atoi(argv[2]);
  int inchannels = atoi(argv[3]);
  int outchannels = atoi(argv[4]);
  int burstchannels = atoi(argv[5]);
  int rpo = atoi(argv[6]);
  int ydim = atoi(argv[7]);
  int xdim = atoi(argv[7]);
  int ytile = atoi(argv[8]);
  int xtile = atoi(argv[8]);
  int ytile_pad = atoi(argv[9]);
  int xtile_pad = atoi(argv[9]);
  int rburst = burstchannels * ydim;
  int numimages = atoi(argv[10]);

  int insize = numimages * inchannels * ydim * xdim * numgroups;
  int outsize = numimages * outchannels * ydim * xdim * numgroups;
  int wsize = outchannels * inchannels * numgroups * numgroups * 3 * 3;
  int wsize_pad = outchannels * inchannels * numgroups * numgroups * 4 * 4;
  int outsize_pad = numimages * outchannels * ydim * xtile_pad * numgroups;

  // Data to be sent to the device
  float *input = (float *)malloc(sizeof(float) * insize);
  // Original Weights
  float *weights = (float *)malloc(sizeof(float) * wsize);
  // Padded Weights
  float *weights_pad = (float *)malloc(sizeof(float) * wsize_pad);
  // Bias data
  float *bias = (float *)malloc(sizeof(float) * outchannels * numgroups);
  // Hardware generated results
  float *hw_results = (float *)malloc(sizeof(float) * outsize_pad);
  // Software generated results
  float *sw_results = (float *)malloc(sizeof(float) * outsize);

  unsigned int correct;               // number of correct results returned
  cl_platform_id platform_id;         // platform id
  cl_device_id device_id;             // compute device id 
  cl_context context;                 // compute context
  cl_command_queue commands;          // compute command queue
  cl_program program;                 // compute program
  cl_kernel kernel;                   // compute kernel
   
  char cl_platform_vendor[1001];
  char cl_platform_name[1001];
   
  cl_mem ocl_input;                   // device memory for the input array
  cl_mem ocl_weights;                 // device memory for the weight array
  cl_mem ocl_output;                  // device memory for the output array   
  cl_mem ocl_bias;                    // device memory  for the bias array

  // Fill inputs, weights, and biases
  int i = 0;
  int j = 0;
  int y, x, idx_pad, idx;
  /* Fill the data */
  for(i = 0; i < insize; i++) {
    input[i] = (float)(rand() % 100 + 1) / 100;
  }
 
  /* Fill the weights */
  for(i = 0; i < wsize; i++) {
    weights[i] = (float)(rand() % 100 + 1) / 100;
  }

  for (i = 0; i < wsize_pad / 16; ++i) {
    for (j = 0; j < 9; ++j) {
      weights_pad[i * 16 + j] = weights[i * 9 + j];
    }
  }

  /* Fill the bias */
  for(i = 0; i < outchannels; ++i) {
    bias[i] = (float)(rand() % 100 + 1) / 100;
  }

  // Connect to first platform
  err = clGetPlatformIDs(1, &platform_id, NULL);
  if (err != CL_SUCCESS)
  {
    printf("Error: Failed to find an OpenCL platform!\n");
    printf("Test failed\n");
    return EXIT_FAILURE;
  }
  err = clGetPlatformInfo(platform_id, CL_PLATFORM_VENDOR, 1000, 
      (void *)cl_platform_vendor, NULL);
  if (err != CL_SUCCESS)
  {
    printf("Error: clGetPlatformInfo(CL_PLATFORM_VENDOR) failed!\n");
    printf("Test failed\n");
    return EXIT_FAILURE;
  }
  printf("CL_PLATFORM_VENDOR %s\n",cl_platform_vendor);
  err = clGetPlatformInfo(platform_id, CL_PLATFORM_NAME, 1000, 
      (void *)cl_platform_name, NULL);
  if (err != CL_SUCCESS)
  {
    printf("Error: clGetPlatformInfo(CL_PLATFORM_NAME) failed!\n");
    printf("Test failed\n");
    return EXIT_FAILURE;
  }
  printf("CL_PLATFORM_NAME %s\n",cl_platform_name);
 
  // Connect to a compute device
  int fpga = 0;
#if defined (FPGA_DEVICE)
  fpga = 1;
#endif
  err = clGetDeviceIDs(platform_id, fpga ? CL_DEVICE_TYPE_ACCELERATOR : 
      CL_DEVICE_TYPE_CPU, 1, &device_id, NULL);
  if (err != CL_SUCCESS)
  {
    printf("Error: Failed to create a device group!\n");
    printf("Test failed\n");
    return EXIT_FAILURE;
  }
  
  // Create a compute context 
  context = clCreateContext(0, 1, &device_id, NULL, NULL, &err);
  if (!context)
  {
    printf("Error: Failed to create a compute context!\n");
    printf("Test failed\n");
    return EXIT_FAILURE;
  }

  // Create a command commands
  commands = clCreateCommandQueue(context, device_id, 0, &err);
  if (!commands)
  {
    printf("Error: Failed to create a command commands!\n");
    printf("Error: code %i\n",err);
    printf("Test failed\n");
    return EXIT_FAILURE;
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
  err = clBuildProgram(program, 0, NULL, NULL, NULL, NULL);
  if (err != CL_SUCCESS)
  {
    size_t len;
    char buffer[2048];

    printf("Error: Failed to build program executable!\n");
    clGetProgramBuildInfo(program, device_id, CL_PROGRAM_BUILD_LOG, 
        sizeof(buffer), buffer, &len);
    printf("%s\n", buffer);
    printf("Test failed\n");
    return EXIT_FAILURE;
  }

  // Create the compute kernel in the program we wish to run
  kernel = clCreateKernel(program, "direct_conv", &err);
  if (!kernel || err != CL_SUCCESS)
  {
    printf("Error: Failed to create compute kernel!\n");
    printf("Test failed\n");
    return EXIT_FAILURE;
  }

  // Create the input and ocl_output arrays in device memory for our calculation
  ocl_input = clCreateBuffer(context,  CL_MEM_READ_ONLY,  sizeof(float) 
      * insize, NULL, NULL);
  ocl_weights = clCreateBuffer(context,  CL_MEM_READ_ONLY,  sizeof(float) 
      * wsize_pad, NULL, NULL);
  ocl_output = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(float) 
      * outsize_pad, NULL, NULL);
  ocl_bias = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(float) 
      * outsize_pad, NULL, NULL); 
  
  if (!ocl_input || !ocl_weights || !ocl_output || !ocl_bias)
  {
    printf("Error: Failed to allocate device memory!\n");
    printf("Test failed\n");
    return EXIT_FAILURE;
  }    
    
  // Write our data set into the input array in device memory 
  err = clEnqueueWriteBuffer(commands, ocl_input, CL_TRUE, 0, sizeof(float) 
      * insize, input, 0, NULL, NULL);
  if (err != CL_SUCCESS)
  {
    printf("Error: Failed to write to source array a!\n");
    printf("Test failed\n");
    return EXIT_FAILURE;
  }

  // Write our data set into the input array in device memory 
  err = clEnqueueWriteBuffer(commands, ocl_weights, CL_TRUE, 0, sizeof(float) 
      * wsize_pad, weights_pad, 0, NULL, NULL);
  if (err != CL_SUCCESS)
  {
    printf("Error: Failed to write to source array b!\n");
    printf("Test failed\n");
    return EXIT_FAILURE;
  }

  err = clEnqueueWriteBuffer(commands, ocl_bias, CL_TRUE, 0, sizeof(float) 
      * outchannels, bias, 0, NULL, NULL);
  if (err != CL_SUCCESS)
  {
    printf("Error: Failed to write to source array b!\n");
    printf("Test failed\n");
    return EXIT_FAILURE;
  }

  err = clEnqueueWriteBuffer(commands, ocl_output, CL_TRUE, 0, sizeof(float) 
      * outsize_pad, hw_results, 0, NULL, NULL);
  
  // Set the arguments to our compute kernel
  for (int n = 0; n < numimages; ++n) {
    for (int g = 0; g < numgroups; ++g) {
      err = 0;
      err  = clSetKernelArg(kernel, 0, sizeof(cl_mem), &ocl_input);
      err |= clSetKernelArg(kernel, 1, sizeof(cl_mem), &ocl_weights);
      err |= clSetKernelArg(kernel, 2, sizeof(cl_mem), &ocl_bias);
      err |= clSetKernelArg(kernel, 3, sizeof(cl_mem), &ocl_output);
      err |= clSetKernelArg(kernel, 4, sizeof(cl_int), &g);
      err |= clSetKernelArg(kernel, 5, sizeof(cl_int), &inchannels);
      err |= clSetKernelArg(kernel, 6, sizeof(cl_int), &outchannels);
      err |= clSetKernelArg(kernel, 7, sizeof(cl_int), &burstchannels);
      err |= clSetKernelArg(kernel, 8, sizeof(cl_int), &rpo);
      err |= clSetKernelArg(kernel, 9, sizeof(cl_int), &ydim);
      err |= clSetKernelArg(kernel, 10, sizeof(cl_int), &xdim);
      err |= clSetKernelArg(kernel, 11, sizeof(cl_int), &ytile);
      err |= clSetKernelArg(kernel, 12, sizeof(cl_int), &xtile);
      err |= clSetKernelArg(kernel, 13, sizeof(cl_int), &ytile_pad);
      err |= clSetKernelArg(kernel, 14, sizeof(cl_int), &xtile_pad);
      err |= clSetKernelArg(kernel, 15, sizeof(cl_int), &rburst);
      err |= clSetKernelArg(kernel, 16, sizeof(cl_int), &n);
      err |= clSetKernelArg(kernel, 17, sizeof(cl_int), &numgroups);
      if (err != CL_SUCCESS)
      {
        printf("Error: Failed to set kernel arguments! %d\n", err);
        printf("Test failed\n");
        return EXIT_FAILURE;
      }

      // Execute the kernel over the entire range of our 1d input data set
      // using the maximum number of work group items for this device

      printf("Running kernel\n");
      err = clEnqueueTask(commands, kernel, 0, NULL, NULL);
    }
  }
  if (err)
  {
    printf("Error: Failed to execute kernel! %d\n", err);
    printf("Test failed\n");
    return EXIT_FAILURE;
  }

  // Read back the results from the device to verify the ocl_output
  printf("Waiting for results\n");
  cl_event readevent;
  err = clEnqueueReadBuffer( commands, ocl_output, CL_TRUE, 0, sizeof(float) * 
      outsize_pad, hw_results, 0, NULL, &readevent );  
  if (err != CL_SUCCESS)
  {
    printf("Error: Failed to read ocl_output array! %d\n", err);
    printf("Test failed\n");
    return EXIT_FAILURE;
  }

  clWaitForEvents(1, &readevent);
  ref_conv(input, weights, bias, sw_results, numgroups, inchannels * numgroups, 
      outchannels * numgroups, xdim, ydim, numimages);
   
  // Validate our results
  correct = 0;
  for (n = 0; n < numimages; ++n) { 
    for (i = 0; i < outchannels * numgroups; ++i) {
      for (y = 0; y < ydim; ++y) {
        for (x = 0; x < xtile_pad; ++x) {
          idx_pad = n * outchannels * numgroups * ydim * xtile_pad + 
                    ((i * ydim) + y) * xtile_pad + x;
          idx = n * outchannels * numgroups * ydim * xdim + 
                ((i * ydim) + y) * xdim + x;
          if (x < xdim) {
            if (fabs(hw_results[idx_pad] - sw_results[idx]) / sw_results[idx] 
                < 1e-5)
              correct++;
            else 
              printf("%d %d %d hw_results %f sw_results %f\n", i, y, x, 
                  hw_results[idx_pad], sw_results[idx]);
          }
        }
      }
    }
  }

  if (correct != outsize) {
    for (i = 0; i < outchannels * numgroups; i++) {
      printf("FPGA OUTPUT Channel %d\n", i);
      for (int y = 0; y < ydim; ++y) {
        for (int x = 0; x < xtile_pad * 2; ++x) {
          printf("%f\t", hw_results[i * ydim * xtile_pad + y * xtile_pad + x]);
        }
        printf("\n");
      }
    }

    for (i = 0; i < outchannels; i++) {
      printf("SOFTWARE OUTPUT Channel %d\n", i);
      for (int y = 0; y < ydim; ++y) {
        for (int x = 0; x < xdim; ++x) {
          printf("%f\t", sw_results[i * ydim * xdim + y * xdim + x]);
        }
        printf("\n");
      }
    } 
  }

  // Print a brief summary detailing the results
  printf("Computed '%d/%d' correct values!\n", correct, outsize);
    
  // Shutdown and cleanup
  clReleaseMemObject(ocl_input);
  clReleaseMemObject(ocl_weights);
  clReleaseMemObject(ocl_output);
  clReleaseProgram(program);
  clReleaseKernel(kernel);
  clReleaseCommandQueue(commands);
  clReleaseContext(context);

  if(correct == outsize){
    printf("Test passed!\n");
    return EXIT_SUCCESS;
  }
  else{
    printf("Test failed\n");
    return EXIT_FAILURE;
  }
}
