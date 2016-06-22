#include "timing.h"
#include "cl-helper.h"
#include "string.h"
#include <math.h>

#define FPGA
#define PIPELINE

void ref_conv(float *input, float *weights, float *output) {
  int o_head, k_head;
  int out_idx, in_idx, k_idx;
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

void ref_relu(float *input, float *output) {
  for (int i = 0; i < OUTPUT_SIZE1; ++i)
    if(input[i] >= 0)
      output[i] = input[i];
    else
      output[i] = 0;
}

void ref_pool(float *input, float *output) {
  for (int i = 0; i < POOL_CHANNEL * POOL_OHEIGHT * POOL_OWIDTH; i++)
    output[i] = -1000;

  for (int c = 0; c < POOL_CHANNEL; ++c) {
    for (int ph = 0; ph < POOL_OHEIGHT; ph++) {
      for (int pw = 0; pw < POOL_OWIDTH; pw++) {
        int hstart = ph * POOL_STRIDE;
        int wstart = pw * POOL_STRIDE;
        int hend = fmin(hstart + POOL_NUM_MASK_ROWS, POOL_IHEIGHT);
        int wend = fmin(wstart + POOL_NUM_MASK_COLS, POOL_IWIDTH);
        const int pool_index = ph * POOL_OWIDTH + pw;
        for (int h = hstart; h < hend; h++) {
          for (int w = wstart; w < wend; ++w) {
            const int index = h * POOL_IWIDTH + w;
            if (input[c * POOL_IWIDTH * POOL_IHEIGHT + index] > output[c * POOL_OWIDTH * POOL_OHEIGHT + pool_index])
              output[c * POOL_OWIDTH * POOL_OHEIGHT + pool_index] = input[c * POOL_IWIDTH * POOL_IHEIGHT + index];
          }
        }
      }
    }
  }
}

int main(int argc, char **argv){

	/****************************************************************************
	*                    Declare OpenCL variables                              *                   
	****************************************************************************/

	int PLATFORM_ID = 0;                // platform ID, default value of 0
	int DEVICE_ID = 1;                  // device ID, default value of 1

	cl_uint num_platforms;              // number of available platforms
	cl_uint num_devices;                // number of available platforms
	cl_context ctx;
	cl_command_queue queue;
	cl_platform_id* platforms;          // list of platform ids
	cl_device_id* devices;              // list of compute device ids
	cl_program program;                 // compute program
	int err;                            // error code returned from api calls
	char cl_platform_vendor[1001];      // string to store vendor name
	char cl_platform_name[1001];        // string to store platform name
	char cl_device_name[1001];          // string to store device name
	size_t cl_device_max_workgroup_size;// size_t to store device max workgroup size

	char *xclbin=argv[1];//"binary/test_hw.xclbin";

	/****************************************************************************
	*                    List all platforms and devices                        *                   
	****************************************************************************/

	printf("\n********* LIST ALL DEVICES *********\n\n");
	print_platforms_devices();
	printf("\n");
	printf("Platform ID: ");
	scanf("%d",&PLATFORM_ID);
	printf("Device ID:   ");
	scanf("%d",&DEVICE_ID);


	/****************************************************************************
	*                    Choose a platform and device                          *                   
	****************************************************************************/

	// Find number of available platforms, store in num_platforms
	err = clGetPlatformIDs(0,NULL,&num_platforms);
	if (err != CL_SUCCESS) { printf("Error: Failed to find an OpenCL platform!\n"); return EXIT_FAILURE; }

	// Allocate memory for list of platforms
	platforms = (cl_platform_id *) malloc(num_platforms*sizeof(cl_platform_id));

	// Get info for all platforms
	err = clGetPlatformIDs(num_platforms,platforms,NULL);
	if (err != CL_SUCCESS) { printf("Error: Failed to find an OpenCL platform!\n"); return EXIT_FAILURE; }

	// Get info for selected platform
	err = clGetPlatformInfo(platforms[PLATFORM_ID],CL_PLATFORM_VENDOR,1000,(void *)cl_platform_vendor,NULL);
	if (err != CL_SUCCESS) { printf("Error: clGetPlatformInfo(CL_PLATFORM_VENDOR) failed!\n"); return EXIT_FAILURE; }

	err = clGetPlatformInfo(platforms[PLATFORM_ID],CL_PLATFORM_NAME,1000,(void *)cl_platform_name,NULL);
	if (err != CL_SUCCESS) { printf("Error: clGetPlatformInfo(CL_PLATFORM_NAME) failed!\n"); return EXIT_FAILURE;
	}

	// Find number of available devices for chosen platform, store in num_devices
	err = clGetDeviceIDs(platforms[PLATFORM_ID], CL_DEVICE_TYPE_ALL, 0, NULL, &num_devices);
	if (err != CL_SUCCESS) { printf("Error: Failed to connect to device!\n"); return EXIT_FAILURE; }

	// Allocate memory for list of devices
	devices = (cl_device_id *) malloc(num_devices*sizeof(cl_device_id));

	// Get list of devices in platform
	err = clGetDeviceIDs(platforms[PLATFORM_ID], CL_DEVICE_TYPE_ALL, num_devices, devices, NULL);
	if (err != CL_SUCCESS) { printf("Error: Failed to connect to device!\n"); return EXIT_FAILURE; }

	// Get info for compute device
	err = clGetDeviceInfo(devices[DEVICE_ID], CL_DEVICE_NAME, 1000, cl_device_name, NULL);
	if (err != CL_SUCCESS) { printf("Error: Failed to connect to device!\n"); return EXIT_FAILURE; }
	err = clGetDeviceInfo(devices[DEVICE_ID], CL_DEVICE_MAX_WORK_GROUP_SIZE, 100, &cl_device_max_workgroup_size, NULL);
	if (err != CL_SUCCESS) { printf("Error: Failed to connect to device!\n"); return EXIT_FAILURE; }

	printf("\n********* LIST CHOSEN DEVICE *********\n\n");
	printf("Connected to vendor:   %s\n", cl_platform_vendor);
	printf("Connected to platform: %s\n", cl_platform_name);
	printf("Connected to device:   %s\n", cl_device_name);
	printf("Max workgroup size:    %d\n", cl_device_max_workgroup_size);
  
	/****************************************************************************
	*                    Create a compute context                              *                   
	****************************************************************************/

	ctx = clCreateContext(0, 1, &devices[DEVICE_ID], NULL, NULL, &err);
	if (!ctx) { printf("Error: Failed to create a compute context!\n"); return EXIT_FAILURE;}

	//queue = clCreateCommandQueue(ctx, devices[DEVICE_ID], CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE, &err);
	queue = clCreateCommandQueue(ctx, devices[DEVICE_ID], 0, &err);
	if (!queue) { printf("Error: Failed to create a command commands! Error: code %i\n",err); return EXIT_FAILURE; }
  
	/****************************************************************************
	*                        Load FPGA binary / kernel source                   *                   
	****************************************************************************/

	unsigned char *kernelbinary;             // kernel binary string
	size_t binary_size;                      // kernel binary size
	int binary_size_int;                     // kernel binary size as int
	int status_temp;                         // return status integer

	#ifdef FPGA
	printf("\n********* LOADING FPGA BINARY *********\n\n");  
	// Load binary from disk
	printf("Loading '%s'\n", xclbin);

	// Load binary into string
	binary_size_int = load_file_to_memory(xclbin, (char **) &kernelbinary);
	if (binary_size_int < 0) { printf("failed to load kernel from xclbin: %s\n", xclbin); return EXIT_FAILURE; }
	binary_size = binary_size_int;

	// Create the compute program in offline fashion, since FPGAs do not support clCreateProgramWithSource
	program = clCreateProgramWithBinary(ctx, 1, &devices[DEVICE_ID], &binary_size, (const unsigned char **) &kernelbinary, &status_temp, &err);
	if ((!program) || (err!=CL_SUCCESS)) { printf("Error: Failed to create compute program from binary %d!\n", err); return EXIT_FAILURE; }

	#else
	printf("\n********* LOADING SOURCE *********\n\n");  
	// Load binary from disk
	printf("Loading '%s'\n", xclsrc);

	const char *knl_text = read_file(xclsrc);
	size_t sizes[] = { strlen(knl_text) };
	program = clCreateProgramWithSource(ctx, 1, &knl_text, sizes, &err);
	if ((!program) || (err!=CL_SUCCESS)) { printf("Error: Failed to create compute program from binary %d!\n", err); return EXIT_FAILURE; }

	#endif 

	// Build program 
	// NOTE: THIS IS WHERE FPGA IS PROGRAMMED!
	err = clBuildProgram(program, 0, NULL, NULL, NULL, NULL);
	if (err != CL_SUCCESS) {
		size_t len;
		char buffer[2048];
		printf("Error: Failed to build program executable!\n");
		clGetProgramBuildInfo(program, devices[DEVICE_ID], CL_PROGRAM_BUILD_LOG, sizeof(buffer), buffer, &len);
		printf("%s\n", buffer);
		return EXIT_FAILURE;
	} else {
		#ifdef FPGA
		printf("Successfully loaded '%s'\n", xclbin);
		#else
		printf("Successfully loaded '%s'\n", xclsrc);
		#endif 
	}

	// --------------------------------------------------------------------------
	// create kernel
	// --------------------------------------------------------------------------

	#ifdef PIPELINE
	// Create the compute kernel in the program we wish to run (refer to kernel by function name in .cl!)
	cl_kernel knl = clCreateKernel(program, "conv1_layer", &err);
	if (!knl || err != CL_SUCCESS) { printf("Error: Failed to create compute kernel!\n"); printf("%d\n",err); return EXIT_FAILURE; }
	cl_kernel knl2 = clCreateKernel(program, "pool1_max_layer", &err);
	if (!knl2 || err != CL_SUCCESS) { printf("Error: Failed to create compute kernel!\n"); printf("%d\n",err); return EXIT_FAILURE; }
	#else
	cl_kernel knl = clCreateKernel(program, "conv_pool", &err);
	if (!knl || err != CL_SUCCESS) { printf("Error: Failed to create compute kernel!\n"); printf("%d\n",err); return EXIT_FAILURE; }
	#endif
	/****************************************************************************
	*              Load data into CPU - transfer to DRAM                       *                   
	****************************************************************************/

	// --------------------------------------------------------------------------
	// allocate and initialize CPU memory 
	// --------------------------------------------------------------------------

	float a1[DATA_SIZE1];               // original data set given to device
	float b1[FILTER_SIZE1];             // original data set given to device
	float c1[OUTPUT_SIZE1];
	float results1[OUTPUT_SIZE1];       // results returned from device
	float sw_results1[OUTPUT_SIZE1];    // results returned from software verification
	float sw_results2[OUTPUT_SIZE1];

  	// Fill our data sets with pattern
	int i = 0;
  	for(i = 0; i < DATA_SIZE1; i++) {
		a1[i] = (float)(rand() % 200 - 100 + 1); // generate random numbers from -100 to +100 (need negatives to check ReLU)
  	}
  	for(i = 0; i < OUTPUT_SIZE1; i++) {
    	results1[i] = 0;
  	}
  	for(i = 0; i < FILTER_SIZE1; i++) {
    	b1[i] = (float)(rand() % 100 + 1);
  	}
  	for(i = 0; i < OUTPUT_SIZE1; i++) {
    	c1[i] = (float)0;
  	}

	// POOLING

	float pool_in_host[POOL_CHANNEL * POOL_IWIDTH * POOL_IHEIGHT];
	float pool_out_host[POOL_CHANNEL * POOL_OWIDTH * POOL_OHEIGHT];
	float pool_out_host_sw[POOL_CHANNEL * POOL_OWIDTH * POOL_OHEIGHT];

	for (int i = 0; i < POOL_ISIZE; ++i) {
		pool_in_host[i] = (float)i;
	}

	for (int i = 0; i < POOL_OSIZE; ++i) {
		pool_out_host[i] = 0;
		pool_out_host_sw[i] = 1;
	}

	// --------------------------------------------------------------------------
	// allocate device memory
	// --------------------------------------------------------------------------

	cl_mem input_a;                     // device memory used for the input array
	cl_mem input_b;                     // device memory used for the filter array
	cl_mem output;                      // device memory used for the output array

	input_a = clCreateBuffer(ctx,  CL_MEM_READ_ONLY,  sizeof(float)*DATA_SIZE1, NULL, NULL);
	input_b = clCreateBuffer(ctx,  CL_MEM_READ_ONLY,  sizeof(float)*FILTER_SIZE1, NULL, NULL);
	output = clCreateBuffer(ctx, CL_MEM_READ_WRITE, sizeof(float)*OUTPUT_SIZE1, NULL, NULL);

	cl_mem pool_input_dev;                    
	cl_mem pool_output_dev;                    

	pool_input_dev = clCreateBuffer(ctx,  CL_MEM_READ_ONLY,  sizeof(float)*POOL_ISIZE, NULL, NULL);
	pool_output_dev = clCreateBuffer(ctx, CL_MEM_WRITE_ONLY, sizeof(float)*POOL_OSIZE, NULL, NULL);

	// --------------------------------------------------------------------------
	// transfer to device
	// --------------------------------------------------------------------------

	CALL_CL_GUARDED(clEnqueueWriteBuffer, (queue, input_a, CL_TRUE, 0, sizeof(float)*DATA_SIZE1, a1, 0, NULL, NULL));
	CALL_CL_GUARDED(clEnqueueWriteBuffer, (queue, input_b, CL_TRUE, 0, sizeof(float)*FILTER_SIZE1, b1, 0, NULL, NULL));

	CALL_CL_GUARDED(clEnqueueWriteBuffer, (queue, pool_input_dev, CL_TRUE, 0, sizeof(float)*POOL_ISIZE, pool_in_host, 0, NULL, NULL));

  	// --------------------------------------------------------------------------
  	// run code on device
  	// --------------------------------------------------------------------------

  	CALL_CL_GUARDED(clFinish, (queue));

  	timestamp_type time1, time2;
  	get_timestamp(&time1);
  	cl_event event;

	cl_ulong time_start, time_end;
	double total_time;

	size_t global[] = { 5, 1, 1 };		// global domain size for our calculation
	size_t local[] = { 1, 1, 1 };		// local domain size for our calculation

	#ifdef PIPELINE
  	CALL_CL_GUARDED(clSetKernelArg, (knl, 0, sizeof(cl_mem), &input_a));
  	CALL_CL_GUARDED(clSetKernelArg, (knl, 1, sizeof(cl_mem), &input_b));	
	CALL_CL_GUARDED(clSetKernelArg, (knl, 2, sizeof(cl_mem), &output));   

  	CALL_CL_GUARDED(clSetKernelArg, (knl2, 0, sizeof(cl_mem), &pool_input_dev));
  	CALL_CL_GUARDED(clSetKernelArg, (knl2, 1, sizeof(cl_mem), &pool_output_dev));       

	CALL_CL_GUARDED(clEnqueueNDRangeKernel, (queue, knl, 3, NULL, (size_t*)&global, (size_t*)&local, 0, NULL, &event));
	CALL_CL_GUARDED(clEnqueueNDRangeKernel, (queue, knl2, 3, NULL, (size_t*)&global, (size_t*)&local, 0, NULL, &event));
	#else
  	CALL_CL_GUARDED(clSetKernelArg, (knl, 0, sizeof(cl_mem), &input_a));
  	CALL_CL_GUARDED(clSetKernelArg, (knl, 1, sizeof(cl_mem), &input_b));	
	CALL_CL_GUARDED(clSetKernelArg, (knl, 2, sizeof(cl_mem), &pool_output_dev));  
	CALL_CL_GUARDED(clEnqueueNDRangeKernel, (queue, knl, 3, NULL, (size_t*)&global, (size_t*)&local, 0, NULL, &event));
	#endif

	// Blocks until all previously queued OpenCL commands in a command-queue are issued to the associated device and have completed
	CALL_CL_GUARDED(clFinish, (queue));
	get_timestamp(&time2); 

	double elapsed = timestamp_diff_in_seconds(time1,time2);
	printf("\nTotal kernel run time:   %f s\n", elapsed);
	//printf("Kernel throughput: %f GB/s\n\n",n*n*sizeof(float)/1e9/elapsed);		

	// --------------------------------------------------------------------------
	// transfer back & check
	// --------------------------------------------------------------------------
	
	#ifdef PIPELINE
	CALL_CL_GUARDED(clEnqueueReadBuffer, (queue, output, CL_TRUE, 0, sizeof(float)*OUTPUT_SIZE1, results1, 0, NULL, &event));
	clWaitForEvents(1, &event);
	#endif
	CALL_CL_GUARDED(clEnqueueReadBuffer, (queue, pool_output_dev, CL_TRUE, 0, sizeof(float)*POOL_OSIZE, pool_out_host, 0, NULL, &event));
	clWaitForEvents(1, &event);
	
	ref_conv(a1, b1, sw_results1); 
	ref_relu(sw_results1,sw_results2);
	ref_pool(sw_results2, pool_out_host_sw);
	
 	// Validate our results
  	int convrelu_correct = 0;
  	int pool_correct = 0;
	int same = 0;
	int pipeline_burst = 1;

	#ifdef PIPELINE
  	for (i = 0;i < ((OUTPUT_SIZE1/96)*(global[0])*pipeline_burst); i++) {
    	if(results1[i] == sw_results2[i]) {
      		convrelu_correct++;			
		}		
	}
   
  	// Print a brief summary detailing the results
	if (convrelu_correct==(OUTPUT_SIZE1/96)*(global[0])*pipeline_burst) {
		printf("\nConvolution Test Passed! (%d correct channels)\n",global[0]*pipeline_burst);
		printf("ReLU        Test Passed! (%d correct channels)\n",global[0]*pipeline_burst);
	} else {
		printf("\nConvolution Test Failed! X\n");
  		printf("...Computed '%d/%d' correct values,", convrelu_correct, OUTPUT_SIZE1);
		printf("'%d/%d' correct channels\n", convrelu_correct/(OUTPUT_SIZE1), (global[0]*pipeline_burst));
	}
	#endif	

  	for (i = 0;i < ((POOL_OSIZE/96)*(global[0]*pipeline_burst)); i++) {
    	if(pool_out_host[i] == pool_out_host_sw[i]) {
      		pool_correct++;		
			//printf("CORRECT HOST[%d]: %f, DEV[%d]: %f\n",i,pool_out_host_sw[i],i,pool_out_host[i]);		
		} else {
			printf("WRONG, HOST[%d]: %f, DEV[%d]: %f\n",i,pool_out_host_sw[i],i,pool_out_host[i]);			
		}
	}
   
  	// Print a brief summary detailing the results
	if (pool_correct==((POOL_OSIZE/96)*(global[0]*pipeline_burst))) {
		printf("Pooling     Test Passed! (%d correct channels)\n",global[0]*pipeline_burst);
	} else {
		printf("Pooling     Test Failed! X\n");
  		printf("...Computed '%d/%d' correct values,", pool_correct, POOL_OSIZE);
		printf("'%d/%d' correct channels\n", pool_correct/(POOL_OSIZE), global[0]*pipeline_burst);
	}

	// --------------------------------------------------------------------------
	// clean up
	// --------------------------------------------------------------------------
  	CALL_CL_GUARDED(clReleaseMemObject, (input_a));
  	CALL_CL_GUARDED(clReleaseMemObject, (input_b));
  	CALL_CL_GUARDED(clReleaseMemObject, (output));
  	CALL_CL_GUARDED(clReleaseMemObject, (pool_input_dev));
  	CALL_CL_GUARDED(clReleaseMemObject, (pool_output_dev));
	CALL_CL_GUARDED(clReleaseKernel, (knl));
	CALL_CL_GUARDED(clReleaseCommandQueue, (queue));
	CALL_CL_GUARDED(clReleaseContext, (ctx));

  	return 0;
}
