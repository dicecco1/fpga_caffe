#define NUM 1
#define CHANNELS 96
#define HEIGHT 55
#define WIDTH 55
#define SIZE 4096 

__kernel __attribute__((reqd_work_group_size(1, 1, 1)))
void relu_layer(__global float16 *input, __global float16 *output)
{
  __local float16 inbuf[SIZE / 16];
  __local float16 outbuf[SIZE / 16];
  
  int offset = get_global_id(0);
  
  async_work_group_copy(inbuf, input + offset * SIZE / 16, SIZE / 16, 0);

  __attribute__((xcl_pipeline_loop))
  for (int i = 0; i < SIZE / 16; ++i) {
    outbuf[i].s0 = (inbuf[i].s0 < 0) ? 0 : inbuf[i].s0;
    outbuf[i].s1 = (inbuf[i].s1 < 0) ? 0 : inbuf[i].s1;
    outbuf[i].s2 = (inbuf[i].s2 < 0) ? 0 : inbuf[i].s2;
    outbuf[i].s3 = (inbuf[i].s3 < 0) ? 0 : inbuf[i].s3;
    outbuf[i].s4 = (inbuf[i].s4 < 0) ? 0 : inbuf[i].s4;
    outbuf[i].s5 = (inbuf[i].s5 < 0) ? 0 : inbuf[i].s5;
    outbuf[i].s6 = (inbuf[i].s6 < 0) ? 0 : inbuf[i].s6;
    outbuf[i].s7 = (inbuf[i].s7 < 0) ? 0 : inbuf[i].s7;
    outbuf[i].s8 = (inbuf[i].s8 < 0) ? 0 : inbuf[i].s8;
    outbuf[i].s9 = (inbuf[i].s9 < 0) ? 0 : inbuf[i].s9;
    outbuf[i].sa = (inbuf[i].sa < 0) ? 0 : inbuf[i].sa;
    outbuf[i].sb = (inbuf[i].sb < 0) ? 0 : inbuf[i].sb;
    outbuf[i].sc = (inbuf[i].sc < 0) ? 0 : inbuf[i].sc;
    outbuf[i].sd = (inbuf[i].sd < 0) ? 0 : inbuf[i].sd;
    outbuf[i].se = (inbuf[i].se < 0) ? 0 : inbuf[i].se;
    outbuf[i].sf = (inbuf[i].sf < 0) ? 0 : inbuf[i].sf;
  }

  async_work_group_copy(output + offset * SIZE / 16, outbuf, SIZE / 16, 0);
} 

