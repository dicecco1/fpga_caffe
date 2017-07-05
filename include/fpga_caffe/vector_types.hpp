#ifndef VECTOR_TYPES_HPP_
#define VECTOR_TYPES_HPP_

#include "half.hpp"

struct char16{
  char s0;
  char s1;
  char s2;
  char s3;
  char s4;
  char s5;
  char s6;
  char s7;
  char s8;
  char s9;
  char sa;
  char sb;
  char sc;
  char sd;
  char se;
  char sf;
};

struct short16 {
  short s0;
  short s1;
  short s2;
  short s3;
  short s4;
  short s5;
  short s6;
  short s7;
  short s8;
  short s9;
  short sa;
  short sb;
  short sc;
  short sd;
  short se;
  short sf;
  
  short16& operator=(const short& rhs) {
    s0 = rhs;
    s1 = rhs;
    s2 = rhs;
    s3 = rhs;
    s4 = rhs;
    s5 = rhs;
    s6 = rhs;
    s7 = rhs;
    s8 = rhs;
    s9 = rhs;
    sa = rhs;
    sb = rhs;
    sc = rhs;
    sd = rhs;
    se = rhs;
    sf = rhs;
    return *this;
  }
};

struct chalf16 {
  chalf s0;
  chalf s1;
  chalf s2;
  chalf s3;
  chalf s4;
  chalf s5;
  chalf s6;
  chalf s7;
  chalf s8;
  chalf s9;
  chalf sa;
  chalf sb;
  chalf sc;
  chalf sd;
  chalf se;
  chalf sf;
  
  chalf16& operator=(const chalf& rhs) {
    s0 = rhs;
    s1 = rhs;
    s2 = rhs;
    s3 = rhs;
    s4 = rhs;
    s5 = rhs;
    s6 = rhs;
    s7 = rhs;
    s8 = rhs;
    s9 = rhs;
    sa = rhs;
    sb = rhs;
    sc = rhs;
    sd = rhs;
    se = rhs;
    sf = rhs;
    return *this;
  }

  chalf16& operator+=(const chalf rhs[16]) { 
#pragma HLS INLINE
    s0 += rhs[0];
    s1 += rhs[1];
    s2 += rhs[2];
    s3 += rhs[3];
    s4 += rhs[4];
    s5 += rhs[5];
    s6 += rhs[6];
    s7 += rhs[7];
    s8 += rhs[8];
    s9 += rhs[9];
    sa += rhs[10];
    sb += rhs[11];
    sc += rhs[12];
    sd += rhs[13];
    se += rhs[14];
    sf += rhs[15];
    return *this;
  }
};

chalf16 max(const chalf16 rhs) {
#pragma HLS INLINE
  chalf16 val;
  val.s0 = max(rhs.s0);
  val.s1 = max(rhs.s1);
  val.s2 = max(rhs.s2);
  val.s3 = max(rhs.s3);
  val.s4 = max(rhs.s4);
  val.s5 = max(rhs.s5);
  val.s6 = max(rhs.s6);
  val.s7 = max(rhs.s7);
  val.s8 = max(rhs.s8);
  val.s9 = max(rhs.s9);
  val.sa = max(rhs.sa);
  val.sb = max(rhs.sb);
  val.sc = max(rhs.sc);
  val.sd = max(rhs.sd);
  val.se = max(rhs.se);
  val.sf = max(rhs.sf);
  return val;
}

chalf16 max(const chalf16 T, const chalf16 U) {
#pragma HLS INLINE
  chalf16 val;
  val.s0 = max(T.s0, U.s0);
  val.s1 = max(T.s1, U.s1);
  val.s2 = max(T.s2, U.s2);
  val.s3 = max(T.s3, U.s3);
  val.s4 = max(T.s4, U.s4);
  val.s5 = max(T.s5, U.s5);
  val.s6 = max(T.s6, U.s6);
  val.s7 = max(T.s7, U.s7);
  val.s8 = max(T.s8, U.s8);
  val.s9 = max(T.s9, U.s9);
  val.sa = max(T.sa, U.sa);
  val.sb = max(T.sb, U.sb);
  val.sc = max(T.sc, U.sc);
  val.sd = max(T.sd, U.sd);
  val.se = max(T.se, U.se);
  val.sf = max(T.sf, U.sf);
  return val;
}

chalf16 max(const chalf16 T, const chalf16 U, const short16 Tmask,
    const short16 Umask, short16 *out_mask) {
#pragma HLS INLINE
  chalf16 val;
  short16 res_mask;
  val.s0 = max(T.s0, U.s0, Tmask.s0, Umask.s0, &(res_mask.s0));
  val.s1 = max(T.s1, U.s1, Tmask.s1, Umask.s1, &(res_mask.s1));
  val.s2 = max(T.s2, U.s2, Tmask.s2, Umask.s2, &(res_mask.s2));
  val.s3 = max(T.s3, U.s3, Tmask.s3, Umask.s3, &(res_mask.s3));
  val.s4 = max(T.s4, U.s4, Tmask.s4, Umask.s4, &(res_mask.s4));
  val.s5 = max(T.s5, U.s5, Tmask.s5, Umask.s5, &(res_mask.s5));
  val.s6 = max(T.s6, U.s6, Tmask.s6, Umask.s6, &(res_mask.s6));
  val.s7 = max(T.s7, U.s7, Tmask.s7, Umask.s7, &(res_mask.s7));
  val.s8 = max(T.s8, U.s8, Tmask.s8, Umask.s8, &(res_mask.s8));
  val.s9 = max(T.s9, U.s9, Tmask.s9, Umask.s9, &(res_mask.s9));
  val.sa = max(T.sa, U.sa, Tmask.sa, Umask.sa, &(res_mask.sa));
  val.sb = max(T.sb, U.sb, Tmask.sb, Umask.sb, &(res_mask.sb));
  val.sc = max(T.sc, U.sc, Tmask.sc, Umask.sc, &(res_mask.sc));
  val.sd = max(T.sd, U.sd, Tmask.sd, Umask.sd, &(res_mask.sd));
  val.se = max(T.se, U.se, Tmask.se, Umask.se, &(res_mask.se));
  val.sf = max(T.sf, U.sf, Tmask.sf, Umask.sf, &(res_mask.sf));
  *out_mask = res_mask;
  return val;
}


struct chalf32 {
  chalf16 l, u;
};

#endif // HVECTOR_TYPES_HPP
