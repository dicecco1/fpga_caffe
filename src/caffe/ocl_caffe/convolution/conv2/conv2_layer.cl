#define NUM_DATA_ROWS (27)
#define NUM_DATA_COLS (27)
#define PAD           (2)
#define NUM_MASK_ROWS (5)
#define NUM_MASK_COLS (5)
#define STRIDE (1)
#define IN_CHANNEL 96 
#define OUT_CHANNEL 256
#define K_CHANNEL 48
#define GROUPS 2
#define TOP_NUM 1 
#define K_NUM 256
#define O_G 128
#define K_G 48 
#define BURST 48 

#define NUM_OUT_COLS (((NUM_DATA_COLS - NUM_MASK_COLS + (2*PAD) )/ (STRIDE)) + 1)
#define NUM_OUT_ROWS (((NUM_DATA_ROWS - NUM_MASK_ROWS + (2*PAD) )/ (STRIDE)) + 1)

#define DATA_SIZE_KERN      TOP_NUM * IN_CHANNEL * (NUM_DATA_ROWS) * (NUM_DATA_COLS)
#define FILTER_SIZE_KERN    K_NUM * K_CHANNEL * NUM_MASK_ROWS * NUM_MASK_COLS
#define OUTPUT_SIZE_KERN    TOP_NUM * OUT_CHANNEL * NUM_OUT_ROWS * NUM_OUT_COLS

__kernel __attribute__ ((reqd_work_group_size(1, 1, 1)))
void conv2_layer(__global float *a, __global float *b, __global float *output)
{
  __local float window[BURST * NUM_MASK_ROWS * NUM_MASK_COLS];
  __local float input[NUM_DATA_ROWS * NUM_DATA_COLS];
  __local float bufout[NUM_OUT_ROWS * NUM_OUT_COLS]; 

  int o_head = 0;
  int k_head = 0;
  int in_y = 0;
  int in_x = 0;
  int out_idx_t = 0;
  int data_idx_t = 0;
  int filter_idx_t = 0;
  int out_idx = 0;
  int data_idx = 0;
  int filter_idx = 0;

  int o = get_global_id(0);

  float temp;
  
  for (int g = 0; g < GROUPS; ++g) {
    o_head = (g == 0) ? 0 : O_G;
    k_head = (g == 0) ? 0 : K_G;
    filter_idx_t = (o + o_head) * K_CHANNEL * NUM_MASK_ROWS * NUM_MASK_COLS;
    
    for (int i = 0; i < NUM_OUT_ROWS * NUM_OUT_COLS; ++i)
      bufout[i] = 0;

    async_work_group_copy(window, b + filter_idx_t,  BURST * NUM_MASK_ROWS * NUM_MASK_COLS, 0);    
    
    for (int i = 0; i < BURST; ++i) {
      async_work_group_copy(input, a + ((k_head) + i) * NUM_DATA_ROWS * NUM_DATA_COLS, NUM_DATA_ROWS * NUM_DATA_COLS, 0);
      filter_idx_t = i * NUM_MASK_ROWS * NUM_MASK_COLS;
      for (int y = 0; y < NUM_OUT_ROWS; ++y) {
        for (int p = 0; p < NUM_MASK_ROWS; ++p) {
          in_y = y * STRIDE - PAD + p;
          //if (in_y >= 0 && in_y < NUM_DATA_ROWS) {
            __attribute__((xcl_pipeline_loop))
            for (int x = 0; x < NUM_OUT_COLS; ++x) {
              //if(in_y >= 0 && in_y < NUM_DATA_ROWS) {
              out_idx = y * NUM_OUT_COLS + x;
              temp = 0;
              for (int q = 0; q < NUM_MASK_COLS; ++q) {
                in_x = x * STRIDE - PAD + q;
                data_idx = data_idx_t + in_y * NUM_DATA_COLS + in_x;
                filter_idx = filter_idx_t + p * NUM_MASK_COLS + q;
                if(in_y >= 0 && in_y < NUM_DATA_ROWS && in_x >= 0 && in_x < NUM_DATA_COLS) {
                  temp += input[data_idx] * window[filter_idx];
                }
              }
              bufout[out_idx] += temp;
            }
          //}
        }
      }
    }
    async_work_group_copy(output + (o + o_head) * NUM_OUT_ROWS * NUM_OUT_COLS, bufout, NUM_OUT_ROWS * NUM_OUT_COLS, 0);
  }
  return;
}
