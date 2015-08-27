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

#define DATA_SIZE_KERN      TOP_NUM * IN_CHANNEL * (NUM_DATA_ROWS) * (NUM_DATA_COLS)
#define FILTER_SIZE_KERN    k_NUM * K_CHANNEL * NUM_MASK_ROWS * NUM_MASK_COLS
#define OUTPUT_SIZE_KERN    TOP_NUM * OUT_CHANNEL * NUM_OUT_ROWS * NUM_OUT_COLS

__kernel __attribute__ ((reqd_work_group_size(1, 1, 1)))
void conv1_layer(__global float *a, __global float *b, __global float *output)
{
  __local float window[K_CHANNEL * NUM_MASK_ROWS * NUM_MASK_COLS];// __attribute__((xcl_array_partition(complete, 3)));
  __local float input[NUM_DATA_ROWS * NUM_DATA_COLS];// __attribute__((xcl_array_partition(complete, 3)));
  __local float bufout[NUM_OUT_ROWS][NUM_OUT_COLS];// __attribute__((xcl_array_partition(complete, 2)));  
  __local float inter[NUM_OUT_ROWS][NUM_OUT_COLS][NUM_MASK_ROWS][NUM_MASK_COLS];// __attribute__((xcl_array_partition(complete, 4)));  

  int o_head = 0;
  int k_head = 0;
  int in_y = 0;
  int in_x = 0;
  int out_idx_t = 0;
  int data_idx_t = 0;
  int filter_idx_t = 0;
  for (int g = 0; g < GROUPS; ++g) {
    o_head = O_G * g;
    k_head = K_G * g;
    for (int o = 0; o < O_G; ++o) {
      o_head = O_G * g;
      k_head = K_G * g;
      out_idx_t = (o + o_head) * NUM_OUT_ROWS;
      filter_idx_t = (o + o_head) * K_CHANNEL;
      __attribute__((xcl_pipeline_loop))
      for (int i = 0; i < NUM_OUT_ROWS; ++i) 
        for (int j = 0; j < NUM_OUT_COLS; ++j)
          bufout[i][j] = 0;      

//      __attribute__((xcl_pipeline_loop))
//      for (int k = 0; k < K_CHANNEL; ++k)
//        for (int i = 0; i < NUM_MASK_ROWS; ++i) 
//          for (int j = 0; j < NUM_MASK_COLS; ++j)
//            window[k][i][j] = b[((filter_idx_t + k) * NUM_MASK_ROWS + i) * NUM_MASK_COLS + j];

//      __attribute__((xcl_pipeline_loop))
//      for (int k = 0; k < IN_CHANNEL; ++k)
//        for (int i = 0; i < NUM_DATA_ROWS; ++i)
//          for (int j = 0; j < NUM_DATA_COLS; ++j)
//            input[k][i][j] = a[(k * NUM_DATA_ROWS + i) * NUM_DATA_COLS + j];
      async_work_group_copy(window, b + filter_idx_t * NUM_MASK_ROWS * NUM_MASK_COLS, K_CHANNEL * NUM_MASK_ROWS * NUM_MASK_COLS, 0);
      for (int k = 0; k < K_G; ++k) {
        async_work_group_copy(input, a + ((k + k_head) * NUM_DATA_ROWS * NUM_DATA_COLS), NUM_DATA_ROWS * NUM_DATA_COLS, 0); 
//        __attribute__((xcl_pipeline_loop))
        for (int y = 0; y < NUM_OUT_ROWS; ++y) {
          for (int x = 0; x < NUM_OUT_COLS; ++x) { 
            for (int p = 0; p < NUM_MASK_ROWS; ++p) {
              for (int q = 0; q < NUM_MASK_COLS; ++q) {   
                in_y = y * STRIDE - PAD + p;
                in_x = x * STRIDE - PAD + q;
                if(in_y >= 0 && in_y < NUM_DATA_ROWS && in_x >= 0 && in_x < NUM_DATA_COLS) {
                  inter[y][x][p][q] = input[in_y * NUM_DATA_COLS + in_x] * window[(k * NUM_MASK_ROWS + p) * NUM_MASK_COLS + q];
                }
                else
                  inter[y][x][p][q] = 0;              
              } 
            }
          }
        }
//        __attribute__((xcl_pipeline_loop))
        for (int y = 0; y < NUM_OUT_ROWS; ++y) {
          for (int x = 0; x < NUM_OUT_COLS; ++x) {
            for (int p = 0; p < NUM_MASK_ROWS; ++p) {
              for (int q = 1; q < NUM_MASK_COLS; ++q)
                inter[y][x][p][0] += inter[y][x][p][q];
              bufout[y][x] += inter[y][x][p][0];
            }
          }
        }
      }
//      __attribute__((xcl_pipeline_loop))
      for(int i = 0; i < NUM_OUT_ROWS; ++i)
        for(int j = 0; j < NUM_OUT_COLS; ++j)
          output[(out_idx_t + i) * NUM_OUT_COLS + j] = bufout[i][j];
    }
  }
  return;
}
