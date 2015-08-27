#ifndef __POOL3_MAX_FLOAT_H__
#define __POOL3_MAX_FLOAT_H__

#include <math.h>
#include <limits.h>

#define CHANNEL 256 
#define NUM_MASK_ROWS 3
#define NUM_MASK_COLS 3
#define STRIDE 2
#define IDX2C(i,j,ld) (((j)*(ld))+(i))
#define IWIDTH 13 
#define IHEIGHT 13
#define OWIDTH 6
#define OHEIGHT 6

// Prototype of top level function for C-synthesis
void pool3_max_layer(float *in,float *out);

#endif // __POOL3_MAX_FLOAT_H__ not defined
