// The main caffe test code. Your test cpp code should include this hpp
// to allow a main function to be compiled into the binary.

#include "caffe/caffe.hpp"
#include "caffe/test/test_caffe_main.hpp"



namespace caffe {
#ifndef CPU_ONLY
  cudaDeviceProp CAFFE_TEST_CUDA_PROP;
#endif
}

#ifndef CPU_ONLY
using caffe::CAFFE_TEST_CUDA_PROP;
#endif

#ifdef USE_OCL
using caffe::oclNumPlatforms;
using caffe::oclPlatform;
using caffe::oclDevices;
using caffe::oclContext;
using caffe::oclCommandQueue;
using caffe::oclProgram;
using caffe::oclKernel;
#endif

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  caffe::GlobalInit(&argc, &argv);
#ifndef CPU_ONLY
  // Before starting testing, let's first print out a few cuda defice info.
  int device;
  cudaGetDeviceCount(&device);
  cout << "Cuda number of devices: " << device << endl;
  if (argc > 1) {
    // Use the given device
    device = atoi(argv[1]);
    cudaSetDevice(device);
    cout << "Setting to use device " << device << endl;
  } else if (CUDA_TEST_DEVICE >= 0) {
    // Use the device assigned in build configuration; but with a lower priority
    device = CUDA_TEST_DEVICE;
  }
  cudaGetDevice(&device);
  cout << "Current device id: " << device << endl;
  cudaGetDeviceProperties(&CAFFE_TEST_CUDA_PROP, device);
#endif

#ifdef USE_OCL
  caffe::Caffe::SetOCLDevice();
  /*oclPlatform.resize(1);
  clGetPlatformIDs(0, NULL, &oclNumPlatforms);
  clGetPlatformIDs(1, &(oclPlatform[0]), NULL);
  clGetDeviceIDs(oclPlatform[0], CL_DEVICE_TYPE_CPU, 1, &oclDevices, NULL);
  oclContext = clCreateContext(NULL, 1, &oclDevices, NULL, NULL, NULL);
  oclCommandQueue = clCreateCommandQueue(oclContext, oclDevices, 0, NULL);
  const char *filename = "src/caffe/layers/conv_layer.cl";
  std::string sourceStr;
  caffe::convertToString(filename, sourceStr);
	const char *source = sourceStr.c_str();
	size_t sourceSize[] = {strlen(source)};

  oclProgram.push_back(clCreateProgramWithSource(oclContext, 1, &source, 
        sourceSize, NULL));
  clBuildProgram(oclProgram[0], 1, &oclDevices, NULL, NULL, NULL);
  
  oclKernel.push_back(clCreateKernel(oclProgram[0], "conv_forward_float", NULL));
  oclKernel.push_back(clCreateKernel(oclProgram[0], "conv_forward_double", NULL));*/
#endif // USE_OCL

  // invoke the test.
  return RUN_ALL_TESTS();
}
