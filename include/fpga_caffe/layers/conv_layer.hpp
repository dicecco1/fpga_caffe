#ifndef CONV_LAYER_HPP_
#define CONV_LAYER_HPP_

typedef struct {
  int inchannels;
  int outchannels;
  int burstchannels;
  int rpo;
  int ydim;
  int xdim; 
  int xtile_pad;
  int ksize;
  int numgroups;
  int numimages;
  int backward;
} kernel_params;

#endif
