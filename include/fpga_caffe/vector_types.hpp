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

struct chalf32 {
  chalf16 l, u;
};

#endif // HVECTOR_TYPES_HPP
