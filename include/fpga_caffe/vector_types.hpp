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

struct chalf16{
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
};
#endif // HVECTOR_TYPES_HPP
