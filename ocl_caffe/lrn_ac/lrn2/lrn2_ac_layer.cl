#define IWIDTH 27
#define IHEIGHT 27
#define OWIDTH 27
#define OHEIGHT 27

#define NUM_OF_BOTTOM_BLOBS 1
#define NUM_CHANNELS 256 
#define ALPHA 0.0001
#define BETA 0.75
#define LOCAL_SIZE 5
#define INV_SIZE 0.2

#define ISIZE NUM_CHANNELS * IWIDTH * IHEIGHT
#define OSIZE NUM_CHANNELS * OWIDTH * OHEIGHT

__kernel __attribute__((reqd_work_group_size(1, 1, 1)))
void lrn2_ac_layer(__global float *input, __global float *output) {
  int off;
  float base, arg, scale, value;

  __local float inbuf[LOCAL_SIZE * IHEIGHT * IWIDTH];// __attribute__((xcl_array_partition(complete, 1)));
  __local float outbuf[OHEIGHT * OWIDTH];// __attribute__((xcl_array_partition(complete, 1)));
  
  int c = get_global_id(0);
  int c_start = c - ((LOCAL_SIZE - 1) / 2);
  int c_end = (c_start + LOCAL_SIZE) < NUM_CHANNELS ? c_start + LOCAL_SIZE : NUM_CHANNELS;
  c_start = c_start > 0 ? c_start : 0;
  int c_idx = c - c_start;
  off = c_end - c_start;
  
  async_work_group_copy(inbuf, input + (c_start * IHEIGHT * IWIDTH), LOCAL_SIZE * IHEIGHT * IWIDTH, 0);

  for (int h = 0; h < IHEIGHT; ++h) {
    __attribute__((xcl_pipeline_loop))
    for (int w = 0; w < IWIDTH; ++w) {
      scale = 1.0;
      for (int i = 0; i < LOCAL_SIZE; ++i) {
        if(i < off) {
          value = inbuf[(i * IHEIGHT + h) * IWIDTH + w]; 
          scale += value * value;
        }
      }
      scale = scale * ALPHA * INV_SIZE;
      base = native_log(scale);
      arg = -1 * BETA * base;
      outbuf[h * OWIDTH + w] = inbuf[(c_idx * IHEIGHT + h) * IWIDTH + w] * native_exp(arg); 
    }
  }

  async_work_group_copy(output + c * OHEIGHT * OWIDTH, outbuf, IHEIGHT * IWIDTH, 0);
}
