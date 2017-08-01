#ifndef CPFP_HPP_
#define CPFP_HPP_

#include <algorithm>
#include <iostream>
#include <limits>
#include <climits>
#include <cmath>
#include <cstring>
#include <stdint.h>
#include <stdbool.h>

#ifdef SYNTHESIS
#include "ap_int.h"
#endif

#define EXP_SIZE 6
#define MANT_SIZE 5 
#define EXP_OFFSET ((1 << (EXP_SIZE - 1)) - 1)
#define MAX_EXP ((1 << EXP_SIZE) - 1)
#define MAX_MANT ((1 << MANT_SIZE) - 1)
#define MANT_MASK MAX_MANT
#define MANT_NORM (1 << MANT_SIZE)
#define EXP_SHIFT MANT_SIZE
#define EXP_MASK (MAX_EXP << MANT_SIZE)
#define PRODUCT_SIZE ((MANT_SIZE + 1) * 2)
#define SIGN_SHIFT (EXP_SIZE + MANT_SIZE)
#define SIGN_MASK (1 << SIGN_SHIFT)
#define FP_WIDTH (EXP_SIZE + MANT_SIZE + 1)
#define ROUND_NEAREST_MULT 0
#define ROUND_NEAREST_ADD 1 
#define CPFP_MIN_VAL (1 << SIGN_SHIFT) | (MAX_EXP << EXP_SHIFT) | MAX_MANT
#define CPFP_MAX_VAL (MAX_EXP << EXP_SHIFT) | MAX_MANT

#if (EXP_SIZE + MANT_SIZE + 1) > 16
  #define DIFF_SIZE (32 - EXP_SIZE - MANT_SIZE - 1)
#else
  #define DIFF_SIZE (16 - EXP_SIZE - MANT_SIZE - 1)
#endif

typedef uint16_t uint16;
typedef uint32_t uint32;
typedef int16_t int16;
typedef int32_t int32;

class cpfp;

cpfp operator*(cpfp T, float U);
cpfp operator*(cpfp T, int U);
cpfp operator*(cpfp T, cpfp U);
cpfp operator+(cpfp T, cpfp U);
cpfp operator/(cpfp T, cpfp U);
cpfp operator/(cpfp T, int U);
bool operator<(cpfp T, cpfp U);
bool operator<=(cpfp T, cpfp U);
bool operator>(cpfp T, cpfp U);
bool operator>=(cpfp T, cpfp U);
bool operator==(cpfp T, cpfp U);
bool operator!=(cpfp T, cpfp U);
cpfp max(cpfp T, cpfp U, short Tmask, short Umask, short *out_mask);
cpfp max(cpfp T, cpfp U);
cpfp max(cpfp T);

/// Convert IEEE single-precision to cpfp-precision.
inline uint32 float2cpfp_impl(float value)
{
#if CPFP_ENABLE_CPP11_STATIC_ASSERT
  static_assert(std::numeric_limits<float>::is_iec559, "float to cpfp conversion needs IEEE 754 conformant 'float' type");
  static_assert(sizeof(uint32)==sizeof(float), "float to cpfp conversion needs unsigned integer type of exactly the size of a 'float'");
#endif
  uint32 bits;		//violating strict aliasing!
  float *temp = &value;
  bits = *((uint32 *)temp);
  int32 exp = ((bits >> 23) & 0xFF) - 127;
  uint32 sign = (bits >> (31 - SIGN_SHIFT)) & SIGN_MASK;
  uint32 mant = (bits & 0x7FFFFF);
  uint32 guard = (mant >> (22 - MANT_SIZE)) & 0x1;
  uint32 round = (mant >> (21 - MANT_SIZE)) & 0x1;
  uint32 mant_noround = mant >> (23 - MANT_SIZE);
  uint32 last = mant_noround & 0x1;
  uint32 sticky = (mant & (~(MAX_MANT << (23 - MANT_SIZE))) & (~(MAX_MANT << (21 - MANT_SIZE)))) > 0;
  uint32 rnd_val = guard & (round | sticky | last);
  uint32 mant_round = (mant_noround != MAX_MANT) ? mant_noround + rnd_val : ((exp < EXP_OFFSET) && rnd_val) ? 0 : mant_noround;
  uint32 exp_add = (mant_noround != MAX_MANT) ? 0 : ((exp < EXP_OFFSET) && rnd_val) ? 1 : 0;
  uint32 eresf = (exp < (-1 * EXP_OFFSET + 1)) ? 0 : (exp <= EXP_OFFSET) ? ((exp + EXP_OFFSET) + exp_add) << MANT_SIZE : (MAX_EXP - 1) << MANT_SIZE;
  uint32 mantf = (exp < (-1 * EXP_OFFSET + 1)) ? 0 : (exp <= EXP_OFFSET) ? mant_round : MAX_MANT;
  uint32 hbits = (sign | eresf | mantf);
  return hbits;
}

/// Convert single-precision to cpfp-precision.
inline uint32 float2cpfp(float value)
{
  return float2cpfp_impl(value);
}

/// Convert cpfp-precision to IEEE single-precision.
inline float cpfp2float_impl(uint32 value)
{
#if CPFP_ENABLE_CPP11_STATIC_ASSERT
  static_assert(std::numeric_limits<float>::is_iec559, "cpfp to float conversion needs IEEE 754 conformant 'float' type");
  static_assert(sizeof(uint32)==sizeof(float), "cpfp to float conversion needs unsigned integer type of exactly the size of a 'float'");
#endif
  float out;
  uint32 sign = (value & SIGN_MASK) << (31 - SIGN_SHIFT);
  uint32 mant = value & MANT_MASK;
  uint32 exp = (value >> MANT_SIZE) & MAX_EXP;
  uint32 eresf = (exp != 0) ? (exp + 127 - EXP_OFFSET) << 23 : 0;
  uint32 mantf = (exp != 0) ? (mant) << (23 - MANT_SIZE) : 0;
  uint32 bits = sign | eresf | mantf;
  uint32 *temp = &bits;

  out = *((float *)temp);
  return out;
}

/// Convert cpfp-precision to single-precision.
inline float cpfp2float(uint32 value)
{
  return cpfp2float_impl(value);
}

class cpfp {
  friend cpfp operator*(cpfp T, float U);
  friend cpfp operator*(cpfp T, int U);
  friend cpfp operator*(cpfp T, cpfp U);
  friend cpfp operator+(cpfp T, cpfp U);
  friend cpfp max(cpfp T, cpfp U, short Tmask, short Umask,
      short *out_mask);
  friend cpfp max(cpfp T, cpfp U);
  friend cpfp max(cpfp T);
  friend cpfp operator/(cpfp T, cpfp U);
  friend cpfp operator/(cpfp T, int U);
  friend bool operator<(cpfp T, cpfp U);
  friend bool operator<=(cpfp T, cpfp U);
  friend bool operator>(cpfp T, cpfp U);
  friend bool operator>=(cpfp T, cpfp U);
  friend bool operator==(cpfp T, cpfp U);
  friend bool operator!=(cpfp T, cpfp U);
  public:
    cpfp() : data_() {}

    cpfp(float rhs) : data_(float2cpfp(rhs)) {}
    
    cpfp(uint16 rhs) : data_(rhs) {}

    cpfp(int rhs) : data_(rhs) {}

    cpfp(uint32 rhs) : data_(rhs) {}

#ifdef SYNTHESIS
    cpfp(ap_uint<FP_WIDTH> rhs) : data_(rhs) {}
#endif

#ifdef SYNTHESIS
    ap_uint<FP_WIDTH> getdata() const {
      return data_;
    }
#endif

    operator float() const {
      return cpfp2float(data_);
    }

    operator uint32() const {
      return data_;
    }

    operator uint16() const {
      return data_;
    }

    cpfp& operator=(const int& rhs) {
      this->data_ = rhs;
      return *this;
    }

    cpfp& operator+=(const cpfp& rhs) {
      *this = *this + rhs;
      return *this;
    }

    cpfp& operator/=(const cpfp& rhs) {
      *this = *this / rhs;
      return *this;
    }

    cpfp& operator/=(const int& rhs) {
      *this = *this / rhs;
      return *this;
    }
 
  private:
#if FP_WIDTH <= 16
    uint16 data_;
#else
    uint32 data_;
#endif
};

#ifndef SYNTHESIS
inline
#endif
cpfp operator*(cpfp T, cpfp U) {
#ifdef SYNTHESIS
#pragma HLS INLINE off
#pragma HLS pipeline
  ap_uint<FP_WIDTH> Tdata_ = T.data_;
  ap_uint<FP_WIDTH> Udata_ = U.data_;
  ap_uint<EXP_SIZE> e1 = (Tdata_) >> EXP_SHIFT;
  ap_uint<EXP_SIZE> e2 = (Udata_) >> EXP_SHIFT;
  ap_uint<MANT_SIZE + 1> mant1 = Tdata_ | MANT_NORM;// 11 bits
  ap_uint<MANT_SIZE + 1> mant2 = Udata_ | MANT_NORM;// 11 bits
  ap_uint<1> sign1 = (Tdata_) >> SIGN_SHIFT;
  ap_uint<1> sign2 = (Udata_) >> SIGN_SHIFT;
  ap_uint<1> sign_res = sign1 ^ sign2;

  ap_uint<FP_WIDTH> sign = sign_res;
  ap_uint<MANT_SIZE + 2> mantres;
  ap_uint<MANT_SIZE> mantresf;
  ap_uint<FP_WIDTH> eresf;
  ap_uint<PRODUCT_SIZE> product = mant1 * mant2; // 22 bits
  mantres = product >> MANT_SIZE; // 11 bits
  ap_int<EXP_SIZE + 2> eres = e1 + e2 - EXP_OFFSET;
  ap_uint<1> last = (product >> MANT_SIZE) & 0x1;
  ap_uint<1> guard = (product >> (MANT_SIZE - 1)) & 0x1;
  ap_uint<1> sticky = ((product & (MAX_MANT >> 1)) > 0);


  // normalize
  if ((mantres >> (MANT_SIZE + 1)) & 0x1) {
    last = (product >> (MANT_SIZE + 1)) & 0x1;
    sticky |= guard;
    guard = (product >> MANT_SIZE) & 0x1;
    mantres = (product >> (MANT_SIZE + 1));
    eres++;
  }

#if ROUND_NEAREST_MULT == 1
  if (guard & (sticky | last)) {
    if (mantres == (MAX_MANT | MANT_NORM))
      eres++;
    mantres++;
  }
#endif

  ap_uint<EXP_SIZE> eres_t;

  eres_t = eres;
  mantresf = mantres;
  if (eres >= MAX_EXP) {
    // saturate results
    eres_t = MAX_EXP - 1;
    mantresf = MAX_MANT;
  } else if ((e1 == 0) || (e2 == 0) || (eres <= 0)) {
    // 0 * val, underflow
    eres_t = 0;
    mantresf = 0;
  }

  eresf = eres_t;

  ap_uint<FP_WIDTH> res;

  res = ((sign << SIGN_SHIFT) & SIGN_MASK) |
    ((eresf << EXP_SHIFT) & EXP_MASK) | mantresf;

  return cpfp(res);
#else
  return cpfp(float(T) * float(U));
#endif
}

#ifdef SYNTHESIS
#if MANT_SIZE == 1
ap_uint<3> LOD(ap_uint<MANT_SIZE + 2> sum_cpath, ap_uint<1> *zero_flag){
  ap_uint<1> a[MANT_SIZE + 2];
  a[2] = (sum_cpath >> 2) & 0x1;
  a[1] = (sum_cpath >> 1) & 0x1;
  a[0] = (sum_cpath >> 0) & 0x1;

  *zero_flag = ~a[2] & ~a[1] & ~a[0];
  ap_uint<3> result;
  ap_uint<3> b[3];
  b[2] = 0;
  b[1] = ~a[2] & ~a[1] & (a[0]);
  b[0] = ~a[2] & a[1];
  result = ((b[2] & 0x1) << 2) | ((b[1] & 0x1) << 1) | (b[0] & 0x1);
  return result;
}
#endif

#if MANT_SIZE == 2
ap_uint<3> LOD(ap_uint<MANT_SIZE + 2> sum_cpath, ap_uint<1> *zero_flag){
  ap_uint<1> a[MANT_SIZE + 2];
  a[3] = (sum_cpath >> 3) & 0x1;
  a[2] = (sum_cpath >> 2) & 0x1;
  a[1] = (sum_cpath >> 1) & 0x1;
  a[0] = (sum_cpath >> 0) & 0x1;

  *zero_flag = ~a[3] & ~a[2] & ~a[1] & ~a[0];
  ap_uint<3> result;
  ap_uint<3> b[3];
  b[2] = 0;
  b[1] = ~a[3] & ~a[2] & (a[1] | a[0]);
  b[0] = ~a[3] & (a[2] | ( ~a[1] & (a[0])));
  result = ((b[2] & 0x1) << 2) | ((b[1] & 0x1) << 1) | (b[0] & 0x1);
  return result;
}
#endif

#if MANT_SIZE == 3 
ap_uint<3> LOD(ap_uint<MANT_SIZE + 2> sum_cpath, ap_uint<1> *zero_flag){
  ap_uint<1> a[MANT_SIZE + 2];
  a[4] = (sum_cpath >> 4) & 0x1;
  a[3] = (sum_cpath >> 3) & 0x1;
  a[2] = (sum_cpath >> 2) & 0x1;
  a[1] = (sum_cpath >> 1) & 0x1;
  a[0] = (sum_cpath >> 0) & 0x1;

  *zero_flag = ~a[4] & ~a[3] & ~a[2] & ~a[1] & ~a[0];
  ap_uint<3> result;
  ap_uint<3> b[3];
  b[2] = ~a[4] & ~a[3] & ~a[2] & ~a[1] & (a[0]);
  b[1] = ~a[4] & ~a[3] & (a[2] | a[1]);
  b[0] = ~a[4] & (a[3] | ( ~a[2] & (a[1])));
  result = ((b[2] & 0x1) << 2) | ((b[1] & 0x1) << 1) | (b[0] & 0x1);
  return result;
}
#endif

#if MANT_SIZE == 4
ap_uint<3> LOD(ap_uint<MANT_SIZE + 2> sum_cpath, ap_uint<1> *zero_flag){
  ap_uint<1> a[MANT_SIZE + 2];
  a[5] = (sum_cpath >> 5) & 0x1;
  a[4] = (sum_cpath >> 4) & 0x1;
  a[3] = (sum_cpath >> 3) & 0x1;
  a[2] = (sum_cpath >> 2) & 0x1;
  a[1] = (sum_cpath >> 1) & 0x1;
  a[0] = (sum_cpath >> 0) & 0x1;

  *zero_flag = ~a[5] & ~a[4] & ~a[3] & ~a[2] & ~a[1] & ~a[0];
  ap_uint<3> result;
  ap_uint<3> b[3];
  b[2] = ~a[5] & ~a[4] & ~a[3] & ~a[2] & (a[1] | a[0]);
  b[1] = ~a[5] & ~a[4] & (a[3] | a[2]);
  b[0] = ~a[5] & (a[4] | ( ~a[3] & (a[2] | (~a[1] & (a[0])))));
  result = ((b[2] & 0x1) << 2) | ((b[1] & 0x1) << 1) | (b[0] & 0x1);
  return result;
}
#endif

#if MANT_SIZE == 5
ap_uint<3> LOD (ap_uint<MANT_SIZE + 2> sum_cpath, ap_uint<1> *zero_flag){
#pragma HLS INLINE
  ap_uint<1> a[MANT_SIZE + 2];
  a[6] = (sum_cpath >> 6) & 0x1;
  a[5] = (sum_cpath >> 5) & 0x1;
  a[4] = (sum_cpath >> 4) & 0x1;
  a[3] = (sum_cpath >> 3) & 0x1;
  a[2] = (sum_cpath >> 2) & 0x1;
  a[1] = (sum_cpath >> 1) & 0x1;
  a[0] = (sum_cpath >> 0) & 0x1;

  *zero_flag = ~a[6] & ~a[5] & ~a[4] & ~a[3] & ~a[2] & ~a[1] & ~a[0];
  ap_uint<3> result;
  ap_uint<3> b[3];
  b[2] = ~a[6] & ~a[5] & ~a[4] & ~a[3] & (a[2] | a[1] | a[0]);
  b[1] = ~a[6] & ~a[5] & ((a[4] | a[3]) | (~a[4] & ~a[3] & ~a[2] & ~a[1] &
         (a[0])));
  b[0] = ~a[6] & (a[5] | ( ~a[4] & (a[3] | (~a[2] & (a[1])))));
  result = ((b[2] & 0x1) << 2) | ((b[1] & 0x1) << 1) | (b[0] & 0x1);
  return result;
}
#endif

#if MANT_SIZE == 6
ap_uint<3> LOD (ap_uint<MANT_SIZE + 2> sum_cpath, ap_uint<1> * zero_flag){
#pragma HLS INLINE
  ap_uint<1> a[MANT_SIZE + 2];
  a[7] = (sum_cpath >> 7) & 0x1;
  a[6] = (sum_cpath >> 6) & 0x1;
  a[5] = (sum_cpath >> 5) & 0x1;
  a[4] = (sum_cpath >> 4) & 0x1;
  a[3] = (sum_cpath >> 3) & 0x1;
  a[2] = (sum_cpath >> 2) & 0x1;
  a[1] = (sum_cpath >> 1) & 0x1;
  a[0] = (sum_cpath >> 0) & 0x1;

  *zero_flag = ~a[7] & ~a[6] & ~a[5] & ~a[4] & ~a[3] & ~a[2] & ~a[1] & ~a[0];
  ap_uint<3> result;
  ap_uint<3> b[3];
  b[2] = ~a[7] & ~a[6] & ~a[5] & ~a[4] & (a[3] | a[2] | a[1] | a[0]);
  b[1] = ~a[7] & ~a[6] & ((a[5] | a[4]) | (~a[5] & ~a[4] & ~a[3] & ~a[2] &
         (a[1] | a[0])));
  b[0] = ~a[7] & (a[6] | (~a[5] & (a[4] | (~a[3] & (a[2] |
         (~a[1] | a[0]))))));
  result = ((b[2] & 0x1) << 2) | ((b[1] & 0x1) << 1) | (b[0] & 0x1);
  return result;
}
#endif

#if MANT_SIZE == 7
ap_uint<4> LOD(ap_uint<MANT_SIZE + 2> sum_cpath, ap_uint<1> *zero_flag) {
#pragma HLS INLINE
  ap_uint<1> a[MANT_SIZE + 2];
  ap_uint<4> b[4];
  a[8] = (sum_cpath >> 8) & 0x1;
  a[7] = (sum_cpath >> 7) & 0x1;
  a[6] = (sum_cpath >> 6) & 0x1;
  a[5] = (sum_cpath >> 5) & 0x1;
  a[4] = (sum_cpath >> 4) & 0x1;
  a[3] = (sum_cpath >> 3) & 0x1;
  a[2] = (sum_cpath >> 2) & 0x1;
  a[1] = (sum_cpath >> 1) & 0x1;
  a[0] = (sum_cpath >> 0) & 0x1;

  *zero_flag = ~(a[8] | a[7] | a[6] | a[5] | a[4] | a[3] | a[2] | a[1] | a[0]);
  b[3] = ~a[8] & ~a[7] & ~a[6] & ~a[5] & ~a[4] & ~a[3] & ~a[2] & ~a[1] & a[0];
  b[2] = ~a[8] & ~a[7] & ~a[6] & ~a[5] & (a[4] | a[3] | a[2] | a[1]);
  b[1] = ~a[8] & ~a[7] & ((a[6] | a[5]) | (~a[6] & ~a[5] & ~a[4] & ~a[3] &
        (a[2] | a[1])));
  b[0] = ~a[8] & (a[7] | (~a[6] & (a[5] | (~a[4] & (a[3] | (~a[2] & a[1]))))));

  ap_uint<4> result =  ((b[3] & 0x1) << 3) | ((b[2] & 0x1) << 2) |
    ((b[1] & 0x1) << 1) | (b[0] & 0x1);
  return result;
}
#endif

#if MANT_SIZE == 8
ap_uint<4> LOD(ap_uint<MANT_SIZE + 2> sum_cpath, ap_uint<1> *zero_flag) {
#pragma HLS INLINE
  ap_uint<1> a[MANT_SIZE + 2];
  ap_uint<4> b[4];
  a[9] = (sum_cpath >> 9) & 0x1;
  a[8] = (sum_cpath >> 8) & 0x1;
  a[7] = (sum_cpath >> 7) & 0x1;
  a[6] = (sum_cpath >> 6) & 0x1;
  a[5] = (sum_cpath >> 5) & 0x1;
  a[4] = (sum_cpath >> 4) & 0x1;
  a[3] = (sum_cpath >> 3) & 0x1;
  a[2] = (sum_cpath >> 2) & 0x1;
  a[1] = (sum_cpath >> 1) & 0x1;
  a[0] = (sum_cpath >> 0) & 0x1;

  *zero_flag = ~(a[9] | a[8] | a[7] | a[6] | a[5] | a[4] | a[3] | a[2] | a[1]
      | a[0]);
  b[3] = ~a[9] & ~a[8] & ~a[7] & ~a[6] & ~a[5] & ~a[4] & ~a[3] & ~a[2] &
    (a[1] | a[0]);
  b[2] = ~a[9] & ~a[8] & ~a[7] & ~a[6] & (a[5] | a[4] | a[3] | a[2]);
  b[1] = ~a[9] & ~a[8] & ((a[7] | a[6]) | (~a[7] & ~a[6] & ~a[5] & ~a[4] &
        (a[3] | a[2])));
  b[0] = ~a[9] & (a[8] | (~a[7] & (a[6] | (~a[5] & (a[4] | (~a[3] &
        (a[2] | (~a[1] & a[0]))))))));

  ap_uint<4> result =  ((b[3] & 0x1) << 3) | ((b[2] & 0x1) << 2) |
    ((b[1] & 0x1) << 1) | (b[0] & 0x1);
  return result;
}
#endif

#if MANT_SIZE == 9 
ap_uint<4> LOD(ap_uint<MANT_SIZE + 2> sum_cpath, ap_uint<1> *zero_flag) {
#pragma HLS INLINE
  ap_uint<1> a[MANT_SIZE + 2];
  ap_uint<4> b[4];
  a[10] = (sum_cpath >> 10) & 0x1;
  a[9] = (sum_cpath >> 9) & 0x1;
  a[8] = (sum_cpath >> 8) & 0x1;
  a[7] = (sum_cpath >> 7) & 0x1;
  a[6] = (sum_cpath >> 6) & 0x1;
  a[5] = (sum_cpath >> 5) & 0x1;
  a[4] = (sum_cpath >> 4) & 0x1;
  a[3] = (sum_cpath >> 3) & 0x1;
  a[2] = (sum_cpath >> 2) & 0x1;
  a[1] = (sum_cpath >> 1) & 0x1;
  a[0] = (sum_cpath >> 0) & 0x1;

  *zero_flag = ~(a[10] | a[9] | a[8] | a[7] | a[6] | a[5] | a[4] | a[3] | a[2]
      | a[1] | a[0]);
  b[3] = ~a[10] & ~a[9] & ~a[8] & ~a[7] & ~a[6] & ~a[5] & ~a[4] & ~a[3] &
    (a[2] | a[1] | a[0]);
  b[2] = ~a[10] & ~a[9] & ~a[8] & ~a[7] & (a[6] | a[5] | a[4] | a[3]);
  b[1] = ~a[10] & ~a[9] & ((a[8] | a[7]) | (~a[8] & ~a[7] & ~a[6] & ~a[5] &
        ((a[4] | a[3]) | (~a[4] & ~a[3] & ~a[2] & ~a[1] & a[0]))));
  b[0] = ~a[10] & (a[9] | (~a[8] & (a[7] | (~a[6] & (a[5] | (~a[4] &
        (a[3] | (~a[2] & a[1]))))))));

  ap_uint<4> result =  ((b[3] & 0x1) << 3) | ((b[2] & 0x1) << 2) |
    ((b[1] & 0x1) << 1) | (b[0] & 0x1);
  return result;
}
#endif

#if MANT_SIZE == 10
ap_uint<4> LOD(ap_uint<MANT_SIZE + 2> sum_cpath, ap_uint<1> *zero_flag) {
#pragma HLS INLINE
  ap_uint<1> a[MANT_SIZE + 2];
  ap_uint<4> b[4];
  a[11] = (sum_cpath >> 11) & 0x1;
  a[10] = (sum_cpath >> 10) & 0x1;
  a[9] = (sum_cpath >> 9) & 0x1;
  a[8] = (sum_cpath >> 8) & 0x1;
  a[7] = (sum_cpath >> 7) & 0x1;
  a[6] = (sum_cpath >> 6) & 0x1;
  a[5] = (sum_cpath >> 5) & 0x1;
  a[4] = (sum_cpath >> 4) & 0x1;
  a[3] = (sum_cpath >> 3) & 0x1;
  a[2] = (sum_cpath >> 2) & 0x1;
  a[1] = (sum_cpath >> 1) & 0x1;
  a[0] = (sum_cpath >> 0) & 0x1;

  *zero_flag = ~(a[10] | a[9] | a[8] | a[7] | a[6] | a[5] | a[4] | a[3] | a[2]
      | a[1] | a[0] | a[11]);
  b[3] = ~a[11] & ~a[10] & ~a[9] & ~a[8] & ~a[7] & ~a[6] & ~a[5] & ~a[4] &
    (a[3] | a[2] | a[1] | a[0]);
  b[2] = ~a[11] & ~a[10] & ~a[9] & ~a[8] & (a[7] | a[6] | a[5] | a[4]);
  b[1] = ~a[11] & ~a[10] & ((a[9] | a[8]) | (~a[9] & ~a[8] & ~a[7] & ~a[6] &
        ((a[5] | a[4]) | (~a[5] & ~a[4] & ~a[3] & ~a[2] & (a[1] | a[0])))));
  b[0] = ~a[11] & (a[10] | (~a[9] & (a[8] | (~a[7] & (a[6] | (~a[5] &
        (a[4] | (~a[3] & (a[2] | (~a[1] & a[0]))))))))));

  ap_uint<4> result =  ((b[3] & 0x1) << 3) | ((b[2] & 0x1) << 2) |
    ((b[1] & 0x1) << 1) | (b[0] & 0x1);
  return result;
}
#endif

#if MANT_SIZE == 11
ap_uint<4> LOD(ap_uint<MANT_SIZE + 2> sum_cpath, ap_uint<1> *zero_flag) {
#pragma HLS INLINE
  ap_uint<1> a[MANT_SIZE + 2];
  ap_uint<4> b[4];
  a[12] = (sum_cpath >> 12) & 0x1;
  a[11] = (sum_cpath >> 11) & 0x1;
  a[10] = (sum_cpath >> 10) & 0x1;
  a[9] = (sum_cpath >> 9) & 0x1;
  a[8] = (sum_cpath >> 8) & 0x1;
  a[7] = (sum_cpath >> 7) & 0x1;
  a[6] = (sum_cpath >> 6) & 0x1;
  a[5] = (sum_cpath >> 5) & 0x1;
  a[4] = (sum_cpath >> 4) & 0x1;
  a[3] = (sum_cpath >> 3) & 0x1;
  a[2] = (sum_cpath >> 2) & 0x1;
  a[1] = (sum_cpath >> 1) & 0x1;
  a[0] = (sum_cpath >> 0) & 0x1;

  *zero_flag = ~(a[10] | a[9] | a[8] | a[7] | a[6] | a[5] | a[4] | a[3] | a[2]
      | a[1] | a[0] | a[11] | a[12]);
  b[3] = ~a[12] & ~a[11] & ~a[10] & ~a[9] & ~a[8] & ~a[7] & ~a[6] & ~a[5] &
    (a[4] | a[3] | a[2] | a[1] | a[0]);
  b[2] = ~a[12] & ~a[11] & ~a[10] & ~a[9] & ((a[8] | a[7] | a[6] | a[5]) |
      (~a[8] & ~a[7] & ~a[6] & ~a[5] & ~a[4] & ~a[3] & ~a[2] & ~a[1] & a[0]));
  b[1] = ~a[12] & ~a[11] & ((a[10] | a[9]) | (~a[10] & ~a[9] & ~a[8] & ~a[7] &
        ((a[6] | a[5]) | (~a[6] & ~a[5] & ~a[4] & ~a[3] & (a[2] | a[1])))));
  b[0] = ~a[12] & (a[11] | (~a[10] & (a[9] | (~a[8] & (a[7] | (~a[6] &
        (a[5] | (~a[4] & (a[3] | (~a[2] & a[1]))))))))));

  ap_uint<4> result =  ((b[3] & 0x1) << 3) | ((b[2] & 0x1) << 2) |
    ((b[1] & 0x1) << 1) | (b[0] & 0x1);
  return result;
}
#endif

#if MANT_SIZE == 12
ap_uint<4> LOD(ap_uint<MANT_SIZE + 2> sum_cpath, ap_uint<1> *zero_flag) {
#pragma HLS INLINE
  ap_uint<1> a[MANT_SIZE + 2];
  ap_uint<4> b[4];
  a[13] = (sum_cpath >> 13) & 0x1;
  a[12] = (sum_cpath >> 12) & 0x1;
  a[11] = (sum_cpath >> 11) & 0x1;
  a[10] = (sum_cpath >> 10) & 0x1;
  a[9] = (sum_cpath >> 9) & 0x1;
  a[8] = (sum_cpath >> 8) & 0x1;
  a[7] = (sum_cpath >> 7) & 0x1;
  a[6] = (sum_cpath >> 6) & 0x1;
  a[5] = (sum_cpath >> 5) & 0x1;
  a[4] = (sum_cpath >> 4) & 0x1;
  a[3] = (sum_cpath >> 3) & 0x1;
  a[2] = (sum_cpath >> 2) & 0x1;
  a[1] = (sum_cpath >> 1) & 0x1;
  a[0] = (sum_cpath >> 0) & 0x1;

  *zero_flag = ~(a[10] | a[9] | a[8] | a[7] | a[6] | a[5] | a[4] | a[3] | a[2]
      | a[1] | a[0] | a[11] | a[12] | a[13]);
  b[3] = ~a[13] & ~a[12] & ~a[11] & ~a[10] & ~a[9] & ~a[8] & ~a[7] & ~a[6] &
    (a[5] | a[4] | a[3] | a[2] | a[1] | a[0]);
  b[2] = ~a[13] & ~a[12] & ~a[11] & ~a[10] & ((a[9] | a[8] | a[7] | a[6]) |
      (~a[9] & ~a[8] & ~a[7] & ~a[6] & ~a[5] & ~a[4] & ~a[3] & ~a[2] &
      (a[1] | a[0])));
  b[1] = ~a[13] & ~a[12] & ((a[11] | a[10]) | (~a[11] & ~a[10] & ~a[9] & ~a[8] &
        ((a[7] | a[6]) | (~a[7] & ~a[6] & ~a[5] & ~a[4] & (a[3] | a[2])))));
  b[0] = ~a[13] & (a[12] | (~a[11] & (a[10] | (~a[9] & (a[8] | (~a[7] &
        (a[6] | (~a[5] & (a[4] | (~a[3] & (a[2] | (~a[1] & a[0]))))))))))));

  ap_uint<4> result =  ((b[3] & 0x1) << 3) | ((b[2] & 0x1) << 2) |
    ((b[1] & 0x1) << 1) | (b[0] & 0x1);
  return result;
}
#endif

#if MANT_SIZE == 13
ap_uint<4> LOD(ap_uint<MANT_SIZE + 2> sum_cpath, ap_uint<1> *zero_flag) {
#pragma HLS INLINE
  ap_uint<1> a[MANT_SIZE + 2];
  ap_uint<4> b[4];
  a[14] = (sum_cpath >> 14) & 0x1;
  a[13] = (sum_cpath >> 13) & 0x1;
  a[12] = (sum_cpath >> 12) & 0x1;
  a[11] = (sum_cpath >> 11) & 0x1;
  a[10] = (sum_cpath >> 10) & 0x1;
  a[9] = (sum_cpath >> 9) & 0x1;
  a[8] = (sum_cpath >> 8) & 0x1;
  a[7] = (sum_cpath >> 7) & 0x1;
  a[6] = (sum_cpath >> 6) & 0x1;
  a[5] = (sum_cpath >> 5) & 0x1;
  a[4] = (sum_cpath >> 4) & 0x1;
  a[3] = (sum_cpath >> 3) & 0x1;
  a[2] = (sum_cpath >> 2) & 0x1;
  a[1] = (sum_cpath >> 1) & 0x1;
  a[0] = (sum_cpath >> 0) & 0x1;

  *zero_flag = ~(a[10] | a[9] | a[8] | a[7] | a[6] | a[5] | a[4] | a[3] | a[2]
      | a[1] | a[0] | a[11] | a[12] | a[13] | a[14]);
  b[3] = ~a[14] & ~a[13] & ~a[12] & ~a[11] & ~a[10] & ~a[9] & ~a[8] & ~a[7] &
    (a[6] | a[5] | a[4] | a[3] | a[2] | a[1] | a[0]);
  b[2] = ~a[14] & ~a[13] & ~a[12] & ~a[11] & ((a[10] | a[9] | a[8] | a[7]) |
      (~a[10] & ~a[9] & ~a[8] & ~a[7] & ~a[6] & ~a[5] & ~a[4] & ~a[3] &
      (a[2] | a[1] | a[0])));
  b[1] = ~a[14] & ~a[13] & ((a[12] | a[11]) | (~a[12] & ~a[11] & ~a[10] &
        ~a[9] & ((a[8] | a[7]) | (~a[8] & ~a[7] & ~a[6] & ~a[5] &
        ((a[4] | a[3]) | (~a[4] & ~a[3] & ~a[2] & ~a[1] & a[0]))))));
  b[0] = ~a[14] & (a[13] | (~a[12] & (a[11] | (~a[10] & (a[9] | (~a[8] &
        (a[7] | (~a[6] & (a[5] | (~a[4] & (a[3] | (~a[2] & a[1]))))))))))));

  ap_uint<4> result =  ((b[3] & 0x1) << 3) | ((b[2] & 0x1) << 2) |
    ((b[1] & 0x1) << 1) | (b[0] & 0x1);
  return result;
}
#endif

#if MANT_SIZE == 14
ap_uint<4> LOD(ap_uint<MANT_SIZE + 2> sum_cpath, ap_uint<1> *zero_flag) {
#pragma HLS INLINE
  ap_uint<1> a[MANT_SIZE + 2];
  ap_uint<4> b[4];
  a[15] = (sum_cpath >> 15) & 0x1;
  a[14] = (sum_cpath >> 14) & 0x1;
  a[13] = (sum_cpath >> 13) & 0x1;
  a[12] = (sum_cpath >> 12) & 0x1;
  a[11] = (sum_cpath >> 11) & 0x1;
  a[10] = (sum_cpath >> 10) & 0x1;
  a[9] = (sum_cpath >> 9) & 0x1;
  a[8] = (sum_cpath >> 8) & 0x1;
  a[7] = (sum_cpath >> 7) & 0x1;
  a[6] = (sum_cpath >> 6) & 0x1;
  a[5] = (sum_cpath >> 5) & 0x1;
  a[4] = (sum_cpath >> 4) & 0x1;
  a[3] = (sum_cpath >> 3) & 0x1;
  a[2] = (sum_cpath >> 2) & 0x1;
  a[1] = (sum_cpath >> 1) & 0x1;
  a[0] = (sum_cpath >> 0) & 0x1;

  *zero_flag = ~(a[10] | a[9] | a[8] | a[7] | a[6] | a[5] | a[4] | a[3] | a[2]
      | a[1] | a[0] | a[11] | a[12] | a[13] | a[14] | a[15]);
  b[3] = ~a[15] & ~a[14] & ~a[13] & ~a[12] & ~a[11] & ~a[10] & ~a[9] & ~a[8] &
    (a[7] | a[6] | a[5] | a[4] | a[3] | a[2] | a[1] | a[0]);
  b[2] = ~a[15] & ~a[14] & ~a[13] & ~a[12] & ((a[11] | a[10] | a[9] | a[8]) |
      (~a[11] & ~a[10] & ~a[9] & ~a[8] & ~a[7] & ~a[6] & ~a[5] & ~a[4] &
      (a[3] | a[2] | a[1] | a[0])));
  b[1] = ~a[15] & ~a[14] & ((a[13] | a[12]) | (~a[13] & ~a[12] & ~a[11] &
        ~a[10] & ((a[9] | a[8]) | (~a[9] & ~a[8] & ~a[7] & ~a[6] &
        ((a[5] | a[4]) | (~a[5] & ~a[4] & ~a[3] & ~a[2] & (a[1] | a[0])))))));
  b[0] = ~a[15] & (a[14] | (~a[13] & (a[12] | (~a[11] & (a[10] | (~a[9] &
        (a[8] | (~a[7] & (a[6] | (~a[5] & (a[4] | (~a[3] & (a[2] |
        (~a[1] & a[0]))))))))))))));

  ap_uint<4> result =  ((b[3] & 0x1) << 3) | ((b[2] & 0x1) << 2) |
    ((b[1] & 0x1) << 1) | (b[0] & 0x1);
  return result;
}
#endif

#endif

#ifndef SYNTHESIS
inline
#endif
cpfp operator+(cpfp T, cpfp U) {
#ifdef SYNTHESIS
#pragma HLS INLINE off
#pragma HLS pipeline
  ap_uint<FP_WIDTH> Tdata_ = T.data_;
  ap_uint<FP_WIDTH> Udata_ = U.data_;
  ap_uint<EXP_SIZE> e1 = Tdata_ >> EXP_SHIFT;
  ap_uint<EXP_SIZE> e2 = Udata_ >> EXP_SHIFT;
  ap_uint<MANT_SIZE> mant1 = Tdata_;
  ap_uint<MANT_SIZE> mant2 = Udata_;
  ap_uint<1> sign1 = Tdata_ >> SIGN_SHIFT;
  ap_uint<1> sign2 = Udata_ >> SIGN_SHIFT;

  // EOP = 1 -> add, EOP = 0 -> sub
  ap_uint<1> EOP = sign1 == sign2; 

  // 1 if e1 is bigger, 0 if e2 is bigger
  ap_uint<1> exp_cmp = (e1 >= e2) ? 1 : 0;
  ap_uint<1> guard, round;
  ap_uint<MANT_SIZE> mantresf;
  ap_uint<MANT_SIZE + EXP_SIZE> eresf;

  ap_uint<MANT_SIZE + 5> sum_fpath;
  ap_uint<1> sum_fpath_sign = 0;

  ap_int<2> Rshifter = 0;
  ap_uint<5> Lshifter = 0;
 
  ap_uint<MANT_SIZE> mant1_s, mant2_s;
  ap_uint<1> sign1_s, sign2_s;
  ap_uint<EXP_SIZE> e1_s, e2_s;

  mant1_s = (exp_cmp) ? mant1 : mant2;
  mant2_s = (exp_cmp) ? mant2 : mant1;
  e1_s = (exp_cmp) ? e1 : e2;
  e2_s = (exp_cmp) ? e2 : e1;
  sign1_s = (exp_cmp) ? sign1 : sign2;
  sign2_s = (exp_cmp) ? sign2 : sign1;

  ap_uint<EXP_SIZE> eres = e1_s;
  ap_uint<EXP_SIZE> diff = e1_s - e2_s; 

  ap_uint<1> fpath_flag = (diff > 1) || EOP;

  ap_uint<MANT_SIZE + 4> mant1_large = (e1_s != 0) ?
    (ap_uint<MANT_SIZE + 4>)(mant1_s | MANT_NORM) : (ap_uint<MANT_SIZE + 4>)0;
  ap_uint<PRODUCT_SIZE + 6> mant2_large = (e2_s != 0) ?
    (ap_uint<PRODUCT_SIZE>)(mant2_s | MANT_NORM) : (ap_uint<PRODUCT_SIZE>)0;

  // Close path, sub and (diff = 0 or diff = 1)
  ap_uint<MANT_SIZE + 2> mant1_cpath;
  ap_uint<MANT_SIZE + 2> mant2_cpath;

  mant1_cpath = mant1_large << 1;

  if (diff == 1)
    mant2_cpath = mant2_large;
  else
    mant2_cpath = mant2_large << 1;

  ap_int<MANT_SIZE + 4> sum_cpath_t;
 
  sum_cpath_t = mant1_cpath - mant2_cpath;

  ap_uint<1> sum_cpath_sign;

  ap_uint<MANT_SIZE + 2> sum_cpath;

  if (sum_cpath_t < 0) {
    sum_cpath = -1 * sum_cpath_t;
    sum_cpath_sign = sign2_s; 
  } else {
    sum_cpath = sum_cpath_t;
    sum_cpath_sign = sign1_s;
  }

  ap_uint<1> zero_flag = 0;
  Lshifter = LOD(sum_cpath, &zero_flag);

  ap_uint<MANT_SIZE> sum_cpath_f = ((sum_cpath) << Lshifter) >> 1;

  // Far path

  // saturate difference at 11 bits

  ap_uint<EXP_SIZE> diff_sat = (diff > (MANT_SIZE + 4)) ?
    (ap_uint<EXP_SIZE>)(MANT_SIZE + 4) : diff;

  ap_uint<PRODUCT_SIZE + 6> mant2_a =
    (mant2_large) << ((MANT_SIZE + 4) - diff_sat);

  ap_uint<1> sticky;
 
#if ROUND_NEAREST_ADD == 1
  sticky = (mant2_a & ((1 << (MANT_SIZE + 1)) - 1)) > 0;
#else
  sticky = 0;
#endif

  ap_uint<MANT_SIZE + 4> mant1_fpath = (mant1_large) << 3;
  ap_uint<MANT_SIZE + 4> mant2_fpath = (mant2_a >> (MANT_SIZE + 1)) | sticky;

  if (EOP) 
    sum_fpath = mant1_fpath + mant2_fpath;
  else
    sum_fpath = mant1_fpath - mant2_fpath; 

  ap_uint<MANT_SIZE + 2> sum_t = (sum_fpath >> 3);
  guard = (sum_fpath >> 2) & 0x1;
  round = (sum_fpath >> 1) & 0x1;
  sticky = sum_fpath & 0x1;

  if ((sum_t >> (MANT_SIZE + 1)) & 0x1) {
    Rshifter = 1;
    sticky |= round;
    round = guard;
    guard = sum_t & 0x1;
    sum_t = (sum_fpath >> 4);
  } else if (((sum_t >> (MANT_SIZE)) & 0x1) == 0) {
    Rshifter = -1;
    guard = round;
    round = 0;
    sum_t = sum_fpath >> 2;
  }

  ap_uint<1> last = sum_t & 0x1;
  ap_uint<1> rnd_ovfl = 0;

#if ROUND_NEAREST_ADD == 1
  if (guard & (last | round | sticky)) {
    if (sum_t == (MAX_MANT | MANT_NORM))
      rnd_ovfl = 1;
    sum_t++;
  }
#endif

  ap_uint<MANT_SIZE> sum_fpath_f = sum_t;

  ap_uint<FP_WIDTH> sign = (fpath_flag) ? sign1_s :
    sum_cpath_sign;

  ap_uint<EXP_SIZE> eres_t;

  ap_uint<EXP_SIZE> eres_fpath_f = eres + Rshifter + rnd_ovfl;
  ap_uint<EXP_SIZE> eres_cpath_f = eres - Lshifter;
  if (fpath_flag) {
    eres_t = eres_fpath_f;
    mantresf = sum_fpath_f;
    if (eres + Rshifter + rnd_ovfl >= MAX_EXP) {
      eres_t = MAX_EXP - 1;
      mantresf = MAX_MANT;
    } else if (eres + Rshifter + rnd_ovfl <= 0) {
      eres_t = 0;
      mantresf = 0;
    } else {
      eres_t = eres_fpath_f;
      mantresf = sum_fpath_f;
    }
  } else {
    if ((eres - Lshifter < 1) || (zero_flag == 1)) {
      eres_t = 0;
      mantresf = 0;
    } else {
      eres_t = eres_cpath_f;
      mantresf = sum_cpath_f;
    }
  }

  eresf = eres_t;

  ap_uint<FP_WIDTH> res;
  res = ((sign << SIGN_SHIFT) & SIGN_MASK) |
    ((eresf << EXP_SHIFT) & EXP_MASK) | mantresf;

  return cpfp(res);
#else
  return cpfp(float(T) + float(U));
#endif
}

#ifndef SYNTHESIS
inline
#endif
bool operator==(cpfp T, cpfp U) {
#ifdef SYNTHESIS
  ap_uint<FP_WIDTH> Tdata_ = T.data_;
  ap_uint<FP_WIDTH> Udata_ = U.data_;
#else
  int32 Tdata_ = T.data_;
  int32 Udata_ = U.data_;
#endif
  return (Tdata_ == Udata_);
}

#ifndef SYNTHESIS
inline
#endif
bool operator!=(cpfp T, cpfp U) {
#ifdef SYNTHESIS
  ap_uint<FP_WIDTH> Tdata_ = T.data_;
  ap_uint<FP_WIDTH> Udata_ = U.data_;
#else
  uint32 Tdata_ = T.data_;
  uint32 Udata_ = U.data_;
#endif
  return (Tdata_ != Udata_);
}

#ifndef SYNTHESIS
inline
#endif
cpfp max(cpfp T, cpfp U) {
#ifdef SYNTHESIS
#pragma HLS INLINE off
#pragma HLS pipeline
  cpfp res;

  if (T < U)
    res = U;
  else
    res = T;  
#else
  cpfp res;
  if (T < U)
    res = U;
  else
    res = T;
#endif
  return res;
}

#ifndef SYNTHESIS
inline
#endif
cpfp max(cpfp T, cpfp U, short Tmask, short Umask, short *out_mask) {
#ifdef SYNTHESIS
#pragma HLS INLINE off
#pragma HLS pipeline
  cpfp res;
  short res_mask;
  if (T < U) {
    res = U;
    res_mask = Umask;
  } else {
    res = T;
    res_mask = Tmask;
  }  
#else
  cpfp res;
  short res_mask;
  if (T < U) {
    res = U;
    res_mask = Umask;
  } else {
    res = T;
    res_mask = Tmask;
  }
#endif
  *out_mask = res_mask;
  return res;
}

// Compares T with 0
#ifndef SYNTHESIS
inline
#endif
cpfp max(cpfp T) {
#ifdef SYNTHESIS
#pragma HLS INLINE off
#pragma HLS pipeline
  ap_uint<FP_WIDTH> Tdata_ = T.data_;
  ap_uint<1> sign1 = Tdata_ >> SIGN_SHIFT;

  cpfp res;
  if (sign1)
    res = cpfp(0);
  else
    res = T;
#else
  cpfp res;
  uint32 sign = T.data_ >> SIGN_SHIFT;
  if (sign == 1)
    res = cpfp(0);
  else
    res = T;
#endif
  return res;
}

inline cpfp operator*(cpfp T, float U) {
  return cpfp(float(T) * U);
}

inline cpfp operator*(cpfp T, int U) {
  return cpfp(float(T) * U);
}

inline cpfp operator/(cpfp T, cpfp U) {
  return cpfp(float(T) / float(U));
}

inline cpfp operator/(cpfp T, int U) {
  return cpfp(float(T) / U);
}

inline bool operator>(cpfp T, cpfp U) {
#if (EXP_SIZE + MANT_SIZE + 1) > 16
  uint32 Tdata, Udata, Tsign, Usign;
#else
  uint16 Tdata, Udata, Tsign, Usign;
#endif
  Tsign = T.data_ >> SIGN_SHIFT;
  Usign = U.data_ >> SIGN_SHIFT;
  Tdata = T.data_ ^ SIGN_MASK;
  Udata = U.data_ ^ SIGN_MASK;
  return ((Tdata > Udata) && (Tsign == 0) && (Usign == 0)) ||
    ((Tdata < Udata) && (Tsign == 1) && (Usign == 1)) ||
    ((Tsign == 0) && (Usign == 1));
}

inline bool operator>=(cpfp T, cpfp U) {
  return (T > U) || (T == U);
}

inline bool operator<(cpfp T, cpfp U) {
#if (EXP_SIZE + MANT_SIZE + 1) > 16
  uint32 Tdata, Udata, Tsign, Usign;
#else
  uint16 Tdata, Udata, Tsign, Usign;
#endif
  Tsign = T.data_ >> SIGN_SHIFT;
  Usign = U.data_ >> SIGN_SHIFT;
  Tdata = T.data_ ^ SIGN_MASK;
  Udata = U.data_ ^ SIGN_MASK;
  return ((Tdata < Udata) && (Tsign == 0) && (Usign == 0)) ||
    ((Tdata > Udata) && (Tsign == 1) && (Usign == 1)) ||
    ((Tsign == 1) && (Usign == 0));
}

inline bool operator<=(cpfp T, cpfp U) {
  return (T < U) || (T == U);
}

#endif  // CPFP_HPP_
