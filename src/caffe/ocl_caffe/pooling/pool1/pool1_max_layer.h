#ifndef __POOL1_MAX_LAYER_H__
#define __POOL1_MAX_LAYER_H__

#include <math.h>
#include <limits.h>

#define CHANNEL 96
#define NUM_MASK_ROWS 3
#define NUM_MASK_COLS 3
#define STRIDE 2
#define IDX2C(i,j,ld) (((j)*(ld))+(i))
#define IWIDTH 55
#define IHEIGHT 55
#define OWIDTH 27
#define OHEIGHT 27

// Prototype of top level function for C-synthesis
void pool1_max_layer(float *in,float *out);

#endif // __POOL1_MAX_LAYER_H__ not defined
