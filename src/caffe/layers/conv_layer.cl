__kernel void conv_forward_float(__global float *bottom, __global float *top,
                               __global float *weight_data)
{
  printf("Hello World float from OpenCL %f %f %f\n", bottom[0], top[0],
         weight_data[0]);
}

__kernel void conv_forward_double(__global double *bottom, __global double *top,
                                __global double *weight_data)
{
  printf("Hello World double from OpenCL %f %f %f\n", bottom[0], top[0],
         weight_data[0]);
}
//test
