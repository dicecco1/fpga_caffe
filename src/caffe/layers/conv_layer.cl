#pragma OPENCL EXTENSION cl_khr_fp64 : enable

int offset(int channels, int height, int width, int n, int c, int h, int w)
{ 
  return ((n * channels + c) * height + h)* width +w;
}

__kernel void conv_forward_float(__global float *bottom, __global float *top,
                               __global float *weight_data, int groups, 
                               int kernel_h, int kernel_w, int pad_h, 
                               int pad_w, int stride_h, int stride_w,
                               int top_h, int top_w, int bottom_h,
                               int bottom_w, int top_num, int bias_term,
                               int top_channel, int bottom_channel,
                               __global float *bias_data, int weight_c)
{
  int o_head, k_head, o_g, k_g;

  o_g = top_channel/groups;
  k_g = bottom_channel/groups;
 
  for(int i = 0; i < top_num*top_channel*top_h*top_w; i++)
    top[i] = 0;
  for (int n = 0; n < top_num; n++) {
    for (int g = 0; g < groups; g++) {
      o_head = o_g * g; 
      k_head = k_g * g;
      for (int o = 0; o < o_g; o++) {
        for (int k = 0; k < k_g; k++) {
          for (int y = 0; y < top_h; y++) {
            for (int x = 0; x < top_w; x++) {
              for (int p = 0; p < kernel_h; p++) {
                for (int q = 0; q < kernel_w; q++) {
                  int in_y = y * stride_h - pad_h + p;
                  int in_x = x * stride_w - pad_w + q;
                  if (in_y >= 0 && in_y < bottom_h
                    && in_x >= 0 && in_x < bottom_w) {
                        top[offset(top_channel, top_h, top_w, n, o + o_head, y, x)] +=
                          bottom[offset(bottom_channel, bottom_h, bottom_w, n, k + k_head, in_y, in_x)]
                          * weight_data[offset(weight_c, kernel_h, kernel_w, o + o_head, k, p, q)];
                  }
                }
              }
            }
          }
        }
      }
    }
  }
  if (bias_term) {
    for (int n = 0; n < top_num; n++) {
      for (int o = 0; o < top_channel; o++) {
        for (int y = 0; y < top_h; y++) {
          for (int x = 0; x < top_w; x++) {
            top[offset(top_channel, top_h, top_w, n, o, y, x)] += bias_data[o];
          }
        }
      }
    }
  }
}

__kernel void conv_forward_double(__global double *bottom, __global double *top,
                                __global double *weight_data, int groups, 
                               int kernel_h, int kernel_w, int pad_h, 
                               int pad_w, int stride_h, int stride_w,
                               int top_h, int top_w, int bottom_h,
                               int bottom_w, int top_num, int bias_term,
                               int top_channel, int bottom_channel,
                               __global double *bias_data, int weight_c)
{
  int o_head, k_head, o_g, k_g;

  o_g = top_channel/groups;
  k_g = bottom_channel/groups;
 
  for(int i = 0; i < top_num*top_channel*top_h*top_w; i++)
    top[i] = 0;
  for (int n = 0; n < top_num; n++) {
    for (int g = 0; g < groups; g++) {
      o_head = o_g * g; 
      k_head = k_g * g;
      for (int o = 0; o < o_g; o++) {
        for (int k = 0; k < k_g; k++) {
          for (int y = 0; y < top_h; y++) {
            for (int x = 0; x < top_w; x++) {
              for (int p = 0; p < kernel_h; p++) {
                for (int q = 0; q < kernel_w; q++) {
                  int in_y = y * stride_h - pad_h + p;
                  int in_x = x * stride_w - pad_w + q;
                  if (in_y >= 0 && in_y < bottom_h
                    && in_x >= 0 && in_x < bottom_w) {
                        top[offset(top_channel, top_h, top_w, n, o + o_head, y, x)] +=
                          bottom[offset(bottom_channel, bottom_h, bottom_w, n, k + k_head, in_y, in_x)]
                          * weight_data[offset(weight_c, kernel_h, kernel_w, o + o_head, k, p, q)];
                  }
                }
              }
            }
          }
        }
      }
    }
  }
  if (bias_term) {
    for (int n = 0; n < top_num; n++) {
      for (int o = 0; o < top_channel; o++) {
        for (int y = 0; y < top_h; y++) {
          for (int x = 0; x < top_w; x++) {
            top[offset(top_channel, top_h, top_w, n, o, y, x)] += bias_data[o];
          }
        }
      }
    }
  }
}
//test123
