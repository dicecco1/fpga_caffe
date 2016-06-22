#ifndef NYUHPC_CL_HELPER
#define NYUHPC_CL_HELPER

#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>

#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/cl.h>
#endif

// Convolution
#define NUM_DATA_ROWS (227)
#define NUM_DATA_COLS (227)
#define PAD           (0)
#define NUM_MASK_ROWS (11)
#define NUM_MASK_COLS (11)
#define STRIDE (4)
#define IN_CHANNEL 3 
#define OUT_CHANNEL 96 
#define K_CHANNEL 3
#define GROUPS 1
#define TOP_NUM 1 
#define K_NUM 96
#define O_G 96
#define K_G 3
#define NUM_OUT_COLS (((NUM_DATA_COLS - NUM_MASK_COLS + (2*PAD) )/ (STRIDE)) + 1)
#define NUM_OUT_ROWS (((NUM_DATA_ROWS - NUM_MASK_ROWS + (2*PAD) )/ (STRIDE)) + 1)

#define DATA_SIZE1        TOP_NUM * IN_CHANNEL * (NUM_DATA_ROWS) * (NUM_DATA_COLS)
#define FILTER_SIZE1      K_NUM * K_CHANNEL * NUM_MASK_ROWS * NUM_MASK_COLS
#define OUTPUT_SIZE1      TOP_NUM * OUT_CHANNEL * NUM_OUT_ROWS  * NUM_OUT_COLS

// ReLU
#define COUNT 55*55 

// Pool
#define POOL_CHANNEL 96
#define POOL_NUM_MASK_ROWS 3
#define POOL_NUM_MASK_COLS 3
#define POOL_STRIDE 2
#define POOL_IDX2C(i,j,ld) (((j)*(ld))+(i))
#define POOL_IWIDTH 55
#define POOL_IHEIGHT 55
#define POOL_OWIDTH 27
#define POOL_OHEIGHT 27
#define POOL_BURST 96
#define POOL_WORKITEMS 69984
#define POOL_ISIZE POOL_CHANNEL*POOL_IWIDTH*POOL_IHEIGHT
#define POOL_OSIZE POOL_CHANNEL*POOL_OWIDTH*POOL_OHEIGHT

// Functions
const char *cl_error_to_str(cl_int e);
void print_platforms_devices();
int load_file_to_memory(const char *filename, char **result);
char *read_file(const char *filename);

/* An error check macro for OpenCL.
 *
 * Usage:
 * CHECK_CL_ERROR(status_code_from_a_cl_operation, "function_name")
 *
 * It will abort with a message if an error occurred.
 */

#define CHECK_CL_ERROR(STATUS_CODE, WHAT) \
  if ((STATUS_CODE) != CL_SUCCESS) \
  { \
    fprintf(stderr, \
        "*** '%s' in '%s' on line %d failed with error '%s'.\n", \
        WHAT, __FILE__, __LINE__, \
        cl_error_to_str(STATUS_CODE)); \
    abort(); \
  }

/* A more automated error check macro for OpenCL, for use with clXxxx
 * functions that return status codes. (Not all of them do, notably 
 * clCreateXxx do not.)
 *
 * Usage:
 * CALL_CL_GUARDED(clFunction, (arg1, arg2));
 *
 * Note the slightly strange comma between the function name and the
 * argument list.
 */

#define CALL_CL_GUARDED(NAME, ARGLIST) \
  { \
    cl_int status_code; \
      status_code = NAME ARGLIST; \
    CHECK_CL_ERROR(status_code, #NAME); \
  }

/* An error check macro for Unix system functions. If "COND" is true, then the
 * last system error ("errno") is printed along with MSG, which is supposed to
 * be a string describing what you were doing.
 *
 * Example:
 * CHECK_SYS_ERROR(dave != 0, "opening hatch");
 */
#define CHECK_SYS_ERROR(COND, MSG) \
  if (COND) \
  { \
    perror(MSG); \
    abort(); \
  }

#endif



