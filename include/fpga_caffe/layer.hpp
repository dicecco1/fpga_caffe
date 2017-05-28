#ifndef LAYER_HPP_
#define LAYER_HPP_

#define FADD_LATENCY 13
#define HADD_LATENCY 12

typedef struct {
  int inchannels;
  int outchannels;
  int burstchannels;
  int rpo;
  int rpofm;
  int burstydim;
  int ydim;
  int xdim; 
  int xtile_pad;
  int ksize;
  int numgroups;
  int numimages;
  int fc;
  int relu;
  int backward;
  int stride;
  int pad;
} kernel_params;

#endif  // LAYER_HPP_
