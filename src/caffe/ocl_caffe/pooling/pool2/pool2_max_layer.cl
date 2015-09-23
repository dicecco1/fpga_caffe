#define TOP_NUM 1 
#define OUT_CHANNEL 256
#define NUM_MASK_ROWS 3
#define NUM_MASK_COLS 3
#define STRIDE 2
#define IWIDTH 27 
#define IHEIGHT 27
#define OWIDTH 13
#define OHEIGHT 13
#define BURST 8 

__kernel __attribute__((reqd_work_group_size(1, 1, 1)))
void pool2_max_layer(__global float *in, __global float *out) {
  __local float inbuf[BURST * IHEIGHT * IWIDTH];
  float interbuf[BURST * IHEIGHT * OWIDTH];
  __local float outbuf[BURST * OHEIGHT * OWIDTH];
  float m;
  int in_x, in_y;
  float val;
  int k = get_global_id(0);

  async_work_group_copy(inbuf, in + (k << 3) * IHEIGHT * IWIDTH, BURST * IHEIGHT * IWIDTH, 0);

  for (int blk = 0; blk < BURST; ++blk) {
//    __attribute__((xcl_pipeline_loop))
    for (int row = 0; row < IHEIGHT; ++row) {
      __attribute__((xcl_pipeline_loop))
      for (int col = 0; col < OWIDTH; ++col) {
        in_x = col * STRIDE;
        m = inbuf[blk * IHEIGHT * IWIDTH + row * IWIDTH + in_x];
        for (int pcol = 1; (pcol < NUM_MASK_COLS && in_x + pcol < IWIDTH); ++pcol) {
          val = inbuf[blk * IHEIGHT * IWIDTH + row * IWIDTH + in_x + pcol];
          if (val > m)
            m = val;
        }
        interbuf[blk * IHEIGHT * OWIDTH + row * OWIDTH + col] = m;
      }
    }
  
//    __attribute__((xcl_pipeline_loop))
    __attribute__((xcl_pipeline_loop))
    for (int row = 0; row < OHEIGHT; ++row) {
      for (int col = 0; col < OWIDTH; ++col) {
        in_y = row * STRIDE;
        m = interbuf[blk * IHEIGHT * OWIDTH + in_y * OWIDTH + col];
        for (int prow = 1; (prow < NUM_MASK_ROWS && in_y + prow < IHEIGHT); ++prow) {
          val = interbuf[blk * IHEIGHT * OWIDTH + (in_y + prow) * OWIDTH + col];
          if (val > m)
            m = val;
        }
        outbuf[blk * OHEIGHT * OWIDTH + row * OWIDTH + col] = m;
      }
    }
  }
  async_work_group_copy(out + (k << 3) * OHEIGHT * OWIDTH, outbuf, BURST * OHEIGHT * OWIDTH, 0); 
}



