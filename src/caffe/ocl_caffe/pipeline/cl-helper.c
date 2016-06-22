#include "cl-helper.h"
#include <string.h>
#include <stdbool.h>
#include <math.h>

#define MAX_NAME_LEN 1000

const char *cl_error_to_str(cl_int e)
{
  switch (e)
  {
    case CL_SUCCESS: return "success";
    case CL_DEVICE_NOT_FOUND: return "device not found";
    case CL_DEVICE_NOT_AVAILABLE: return "device not available";
#if !(defined(CL_PLATFORM_NVIDIA) && CL_PLATFORM_NVIDIA == 0x3001)
    case CL_COMPILER_NOT_AVAILABLE: return "device compiler not available";
#endif
    case CL_MEM_OBJECT_ALLOCATION_FAILURE: return "mem object allocation failure";
    case CL_OUT_OF_RESOURCES: return "out of resources";
    case CL_OUT_OF_HOST_MEMORY: return "out of host memory";
    case CL_PROFILING_INFO_NOT_AVAILABLE: return "profiling info not available";
    case CL_MEM_COPY_OVERLAP: return "mem copy overlap";
    case CL_IMAGE_FORMAT_MISMATCH: return "image format mismatch";
    case CL_IMAGE_FORMAT_NOT_SUPPORTED: return "image format not supported";
    case CL_BUILD_PROGRAM_FAILURE: return "build program failure";
    case CL_MAP_FAILURE: return "map failure";

    case CL_INVALID_VALUE: return "invalid value";
    case CL_INVALID_DEVICE_TYPE: return "invalid device type";
    case CL_INVALID_PLATFORM: return "invalid platform";
    case CL_INVALID_DEVICE: return "invalid device";
    case CL_INVALID_CONTEXT: return "invalid context";
    case CL_INVALID_QUEUE_PROPERTIES: return "invalid queue properties";
    case CL_INVALID_COMMAND_QUEUE: return "invalid command queue";
    case CL_INVALID_HOST_PTR: return "invalid host ptr";
    case CL_INVALID_MEM_OBJECT: return "invalid mem object";
    case CL_INVALID_IMAGE_FORMAT_DESCRIPTOR: return "invalid image format descriptor";
    case CL_INVALID_IMAGE_SIZE: return "invalid image size";
    case CL_INVALID_SAMPLER: return "invalid sampler";
    case CL_INVALID_BINARY: return "invalid binary";
    case CL_INVALID_BUILD_OPTIONS: return "invalid build options";
    case CL_INVALID_PROGRAM: return "invalid program";
    case CL_INVALID_PROGRAM_EXECUTABLE: return "invalid program executable";
    case CL_INVALID_KERNEL_NAME: return "invalid kernel name";
    case CL_INVALID_KERNEL_DEFINITION: return "invalid kernel definition";
    case CL_INVALID_KERNEL: return "invalid kernel";
    case CL_INVALID_ARG_INDEX: return "invalid arg index";
    case CL_INVALID_ARG_VALUE: return "invalid arg value";
    case CL_INVALID_ARG_SIZE: return "invalid arg size";
    case CL_INVALID_KERNEL_ARGS: return "invalid kernel args";
    case CL_INVALID_WORK_DIMENSION: return "invalid work dimension";
    case CL_INVALID_WORK_GROUP_SIZE: return "invalid work group size";
    case CL_INVALID_WORK_ITEM_SIZE: return "invalid work item size";
    case CL_INVALID_GLOBAL_OFFSET: return "invalid global offset";
    case CL_INVALID_EVENT_WAIT_LIST: return "invalid event wait list";
    case CL_INVALID_EVENT: return "invalid event";
    case CL_INVALID_OPERATION: return "invalid operation";
    case CL_INVALID_GL_OBJECT: return "invalid gl object";
    case CL_INVALID_BUFFER_SIZE: return "invalid buffer size";
    case CL_INVALID_MIP_LEVEL: return "invalid mip level";

#if defined(cl_khr_gl_sharing) && (cl_khr_gl_sharing >= 1)
    case CL_INVALID_GL_SHAREGROUP_REFERENCE_KHR: return "invalid gl sharegroup reference number";
#endif

#ifdef CL_VERSION_1_1
    case CL_MISALIGNED_SUB_BUFFER_OFFSET: return "misaligned sub-buffer offset";
    case CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST: return "exec status error for events in wait list";
    case CL_INVALID_GLOBAL_WORK_SIZE: return "invalid global work size";
#endif

    default: return "invalid/unknown error code";
  }
}




void print_platforms_devices() {
	// get number of platforms
	cl_uint plat_count;
	
	//CALL_CL_GUARDED(clGetPlatformIDs, (0, NULL, &plat_count));
	clGetPlatformIDs(0, NULL, &plat_count);

	// allocate memory, get list of platforms
	cl_platform_id *platforms =
	(cl_platform_id *) malloc(plat_count*sizeof(cl_platform_id));
	CHECK_SYS_ERROR(!platforms, "allocating platform array");

	CALL_CL_GUARDED(clGetPlatformIDs, (plat_count, platforms, NULL));

	

  //printf("Platform count: %d\n",(int)plat_count);
  // iterate over platforms
  for (cl_uint i = 0; i < plat_count; ++i)
  {
    // get platform vendor name
    char buf[MAX_NAME_LEN];
    CALL_CL_GUARDED(clGetPlatformInfo, (platforms[i], CL_PLATFORM_VENDOR,
          sizeof(buf), buf, NULL));
    printf("platform %d: vendor '%s'\n", i, buf);

    // get number of devices in platform
    cl_uint dev_count;
    clGetDeviceIDs(platforms[i], CL_DEVICE_TYPE_ALL, 0, NULL, &dev_count);

    if (dev_count == 0) {
        printf("No devices found!\n");
    } else {
        cl_device_id *devices =
          (cl_device_id *) malloc(dev_count*sizeof(cl_device_id));
        CHECK_SYS_ERROR(!devices, "allocating device array");

        // get list of devices in platform
        CALL_CL_GUARDED(clGetDeviceIDs, (platforms[i], CL_DEVICE_TYPE_ALL,
              dev_count, devices, NULL));

        // iterate over devices
        for (cl_uint j = 0; j < dev_count; ++j)
        {
          char buf[MAX_NAME_LEN];
          CALL_CL_GUARDED(clGetDeviceInfo, (devices[j], CL_DEVICE_NAME,
                sizeof(buf), buf, NULL));
          printf("  device %d: '%s'\n", j, buf);
        }
        free(devices);
    }
  }
  free(platforms);
}

//Xilinx function
int load_file_to_memory(const char *filename, char **result) { 
  size_t size = 0;
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

// Reading Source File
char *read_file(const char *filename) {
  FILE *f = fopen(filename, "r");
  CHECK_SYS_ERROR(!f, "read_file: opening file");

  // figure out file size
  CHECK_SYS_ERROR(fseek(f, 0, SEEK_END) < 0, "read_file: seeking to end");
  size_t size = ftell(f);

  CHECK_SYS_ERROR(fseek(f, 0, SEEK_SET) != 0,
      "read_file: seeking to start");

  // allocate memory, slurp in entire file
  char *result = (char *) malloc(size+1);
  CHECK_SYS_ERROR(!result, "read_file: allocating file contents");
  CHECK_SYS_ERROR(fread(result, 1, size, f) < size,
      "read_file: reading file contents");

  // close, return
  CHECK_SYS_ERROR(fclose(f), "read_file: closing file");
  result[size] = '\0';

  return result;
}

