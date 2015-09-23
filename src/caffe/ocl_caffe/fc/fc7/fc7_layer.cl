#define M_ 1
#define N_ 4096 
#define K_ 4096
#define BURST 512

__kernel __attribute__ ((reqd_work_group_size(1, 1, 1)))
void fc7_layer(__global float8 *a, __global float8 *b, __global float8 *output)
{
  __local float8 inputA[K_ / 8];
  __local float8 inputB[K_ / 8];
  float sum[BURST];
  float8 inter[K_ / 8];
  __local float8 outbuf[BURST / 8];
  float psum[K_ / 8];
  float psum2[K_ / 64];
  int j = get_global_id(0);
  int i = get_global_id(1);
  float temp;
  async_work_group_copy(inputA, a + i * K_ / 8, K_ / 8, 0);

  for (int off = 0; off < BURST; ++off) {
    temp = 0;
    async_work_group_copy(inputB, b + ((j * BURST) + off) * K_ / 8, K_ / 8, 0);
    __attribute__((xcl_pipeline_loop))
    for (int k = 0; k < K_ / 8; ++k) {
        inter[k] = inputA[k] * inputB[k];
   }
    __attribute__((xcl_pipeline_loop))
    for (int k = 0; k < K_ / 8; ++k) {
      psum[k] = inter[k].s0 + inter[k].s1 + inter[k].s2 + inter[k].s3 + inter[k].s4 
                 + inter[k].s5 + inter[k].s6 + inter[k].s7;
    }
    __attribute__((xcl_pipeline_loop))
    for (int k = 0; k < K_ / 64; ++k) {
      psum2[k] = 0;
      for (int n = 0; n < 8; ++n)
        psum2[k] += psum[k * 8 + n];
    }
    __attribute__((xcl_pipeline_loop))
    for (int k = 0; k < K_ / 64; ++k)
      temp += psum2[k];
    sum[off] = temp;
  }
  __attribute__((xcl_pipeline_loop))
  for (int x = 0; x < BURST / 8; ++x) {
    outbuf[x].s0 = sum[x * 8 + 0];
    outbuf[x].s1 = sum[x * 8 + 1];
    outbuf[x].s2 = sum[x * 8 + 2];
    outbuf[x].s3 = sum[x * 8 + 3];
    outbuf[x].s4 = sum[x * 8 + 4];
    outbuf[x].s5 = sum[x * 8 + 5];
    outbuf[x].s6 = sum[x * 8 + 6];
    outbuf[x].s7 = sum[x * 8 + 7];
  }
  async_work_group_copy(output + (j * BURST) / 8, outbuf, BURST / 8, 0);

  return;
}
