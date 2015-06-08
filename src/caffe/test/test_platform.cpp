#include <cstdio>
#include <cstdlib>

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
#endif // CPU_ONLY

#ifdef USE_OCL
  std::vector<cl::Platform> all_platforms;
  cl::Platform::get(&all_platforms);
  std::cout<<"Number of platforms is: "<<all_platforms.size()<<std::endl;
  EXPECT_TRUE(all_platforms.size() != 0);
  
  cl::Platform default_platform = all_platforms[0];
  std::cout<<"Using platform: "
    <<default_platform.getInfo<CL_PLATFORM_NAME>()<<std::endl;
  
  std::vector<cl::Device> all_devices;
  default_platform.getDevices(CL_DEVICE_TYPE_ALL, &all_devices);
  std::cout<<"Number of devices is: "<<all_devices.size()<<std::endl;
  EXPECT_TRUE(all_devices.size() != 0);
  
  cl::Device default_device = all_devices[0];
  std::cout<<"Using device: "
    <<default_device.getInfo<CL_DEVICE_NAME>()<<std::endl;
#endif // USE_OCL

}

} // namespace caffe
