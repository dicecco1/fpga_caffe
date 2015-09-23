#ifndef __POOL2_MAX_LAYER_H__
#define __POOL2_MAX_LAYER_H__

#include <math.h>
#include <limits.h>

#define CHANNEL 256
#define NUM_MASK_ROWS 3
#define NUM_MASK_COLS 3
#define STRIDE 2
#define IDX2C(i,j,ld) (((j)*(ld))+(i))
#define IWIDTH 27
#define IHEIGHT 27
#define OWIDTH 13
#define OHEIGHT 13

// Prototype of top level function for C-synthesis
void pool2_max_layer(float *in,float *out);

#endif // __POOL2_MAX_LAYER_H__ not defined
