#define IWIDTH 55 
#define IHEIGHT 55
#define OWIDTH 55
#define OHEIGHT 55

#define NUM_OF_BOTTOM_BLOBS 1
#define NUM_CHANNELS 96 
#define ALPHA 0.0001
#define BETA 0.75
#define LOCAL_SIZE 5

#define ISIZE NUM_CHANNELS * IWIDTH * IHEIGHT
#define OSIZE NUM_CHANNELS * OWIDTH * OHEIGHT

__kernel __attribute__((reqd_work_group_size(24, 1, 1)))
void lrn1_ac_layer(__global float *input, __global float *output) {
  int off;
  float alpha = ALPHA;
  float beta = BETA;
  int size = LOCAL_SIZE;

  float inbuf[LOCAL_SIZE][IWIDTH];// __attribute__((xcl_array_partition(complete, 1)));
  float outbuf[LOCAL_SIZE][OWIDTH];// __attribute__((xcl_array_partition(complete, 1)));
  
  int c = get_global_id(0);
  int c_start = c - ((size - 1) / 2);
  int c_end = (c_start + size) < NUM_CHANNELS ? c_start + size : NUM_CHANNELS;
  c_start = c_start > 0 ? c_start : 0;
  int c_idx = 0;


  for (int h = 0; h < IHEIGHT + 1; ++h) {
    int c_idx = 0;
    __attribute__((xcl_pipeline_loop))
    for (int w = 0; w < IWIDTH; ++w) {
      for (int i = 0; i < size; ++i) {
        if(c_start + i < c_end)
          inbuf[i][w] = input[((c_start + i) * IHEIGHT + h) * IWIDTH + w];
        if(c_start + i == c)
          c_idx = i;
      }
      float scale = 1.0;
      for (int i = 0; i < size; ++i) {
        off = c_start + i;
        if(off < c_end) {
          float value = inbuf[i][w]; 
          scale += value * value * alpha / size;
        }
      }
      float base = native_log(scale);
      float arg = beta * base;
      //outbuf[c][w] = inbuf[c_idx][w] * native_exp(-1 * arg);
      output[(c * OHEIGHT + h) * OWIDTH + w] = inbuf[c_idx][w] * native_exp(-1 * arg); //outbuf[c][w];
    }
  }
}
