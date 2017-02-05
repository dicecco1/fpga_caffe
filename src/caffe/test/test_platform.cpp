#include <cstdio>
#include <cstdlib>
#include <vector>

#include "glog/logging.h"
#include "gtest/gtest.h"

#include "caffe/test/test_caffe_main.hpp"

namespace caffe {

class PlatformTest : public ::testing::Test {};

TEST_F(PlatformTest, TestInitialization) {
#ifndef CPU_ONLY
  extern cudaDeviceProp CAFFE_TEST_CUDA_PROP;
  printf("Major revision number:         %d\n",  CAFFE_TEST_CUDA_PROP.major);
  printf("Minor revision number:         %d\n",  CAFFE_TEST_CUDA_PROP.minor);
  printf("Name:                          %s\n",  CAFFE_TEST_CUDA_PROP.name);
  printf("Total global memory:           %lu\n",
         CAFFE_TEST_CUDA_PROP.totalGlobalMem);
  printf("Total shared memory per block: %lu\n",
         CAFFE_TEST_CUDA_PROP.sharedMemPerBlock);
  printf("Total registers per block:     %d\n",
         CAFFE_TEST_CUDA_PROP.regsPerBlock);
  printf("Warp size:                     %d\n",
         CAFFE_TEST_CUDA_PROP.warpSize);
  printf("Maximum memory pitch:          %lu\n",
         CAFFE_TEST_CUDA_PROP.memPitch);
  printf("Maximum threads per block:     %d\n",
         CAFFE_TEST_CUDA_PROP.maxThreadsPerBlock);
  for (int i = 0; i < 3; ++i)
    printf("Maximum dimension %d of block:  %d\n", i,
           CAFFE_TEST_CUDA_PROP.maxThreadsDim[i]);
  for (int i = 0; i < 3; ++i)
    printf("Maximum dimension %d of grid:   %d\n", i,
           CAFFE_TEST_CUDA_PROP.maxGridSize[i]);
  printf("Clock rate:                    %d\n", CAFFE_TEST_CUDA_PROP.clockRate);
  printf("Total constant memory:         %lu\n",
         CAFFE_TEST_CUDA_PROP.totalConstMem);
  printf("Texture alignment:             %lu\n",
         CAFFE_TEST_CUDA_PROP.textureAlignment);
  printf("Concurrent copy and execution: %s\n",
         (CAFFE_TEST_CUDA_PROP.deviceOverlap ? "Yes" : "No"));
  printf("Number of multiprocessors:     %d\n",
         CAFFE_TEST_CUDA_PROP.multiProcessorCount);
  printf("Kernel execution timeout:      %s\n",
         (CAFFE_TEST_CUDA_PROP.kernelExecTimeoutEnabled ? "Yes" : "No"));
  printf("Unified virtual addressing:    %s\n",
         (CAFFE_TEST_CUDA_PROP.unifiedAddressing ? "Yes" : "No"));
  EXPECT_TRUE(true);
#endif  // CPU_ONLY

#ifdef USE_OCL
  cl_uint numPlatforms;
  std::vector<cl_platform_id> all_platforms;
  clGetPlatformIDs(0, NULL, &numPlatforms);
  all_platforms.resize(1);
  clGetPlatformIDs(1, &(all_platforms[0]), NULL);
  std::cout << "Number of platforms is: " << numPlatforms << std::endl;
  EXPECT_NE(numPlatforms, 0);

  cl_device_id all_devices;
  cl_uint numDevices;
  clGetDeviceIDs(all_platforms[0], CL_DEVICE_TYPE_ALL, 0, NULL, &numDevices);
  clGetDeviceIDs(all_platforms[0], CL_DEVICE_TYPE_ALL, 1, &all_devices, NULL);
  std::cout << "Number of devices is: " << numDevices << std::endl;
  EXPECT_NE(numDevices, 0);

#endif  // USE_OCL
}

}  // namespace caffe
