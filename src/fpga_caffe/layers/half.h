#include <algorithm>
#include <iostream>
#include <limits>
#include <climits>
#include <cmath>
#include <cstring>
#include <stdint.h>
#include <stdbool.h>
#include "ap_int.h"

#define SIGN_MASK_HP 0x8000
#define EXP_SIZE 6 
#define MANT_SIZE 9
#define EXP_OFFSET ((1 << (EXP_SIZE - 1)) - 1)
#define MAX_EXP ((1 << EXP_SIZE) - 1)
#define MAX_MANT ((1 << MANT_SIZE) - 1)
#define MANT_MASK_HP MAX_MANT
#define MANT_NORM_HP (1 << MANT_SIZE)
#define EXP_SHIFT_HP MANT_SIZE
#define EXP_MASK_HP (MAX_EXP << MANT_SIZE)
#define PRODUCT_SIZE ((MANT_SIZE + 1) * 2)

typedef uint16_t uint16;
typedef uint32_t uint32;


#ifndef HALF_ROUND_STYLE
	#define HALF_ROUND_STYLE	1			// = std::round_indeterminate
#endif

class chalf;

chalf operator*(chalf T, chalf U);
chalf operator+(chalf T, chalf U);
bool operator==(chalf T, chalf U);
bool operator!=(chalf T, chalf U);
chalf max(chalf T, chalf U);
chalf max(chalf T);
/// Convert IEEE single-precision to chalf-precision.
/// Credit for this goes to [Jeroen van der Zijp](ftp://ftp.fox-toolkit.org/pub/fastchalffloatconversion.pdf).
/// \tparam R rounding mode to use, `std::round_indeterminate` for fastest rounding
/// \param value single-precision value
/// \return binary representation of chalf-precision value
uint16 float2chalf_impl(float value)
{
#if HALF_ENABLE_CPP11_STATIC_ASSERT
  static_assert(std::numeric_limits<float>::is_iec559, "float to chalf conversion needs IEEE 754 conformant 'float' type");
  static_assert(sizeof(uint32)==sizeof(float), "float to chalf conversion needs unsigned integer type of exactly the size of a 'float'");
#endif
  uint32 bits;		//violating strict aliasing!
  float *temp = &value;
  bits = *((uint32 *)temp);
  short exp = ((bits >> 23) & 0xFF) - 127;
  uint16 sign = (bits >> 16) & 0x8000;
  uint32 mant = (bits & 0x7FFFFF);
  uint16 expf = (exp < (-1 * EXP_OFFSET + 1)) ? 0 : (exp <= EXP_OFFSET) ? (exp + EXP_OFFSET) << MANT_SIZE : (MAX_EXP - 1) << MANT_SIZE;
  uint16 mantf = (exp < (-1 * EXP_OFFSET + 1)) ? 0 : (exp <= EXP_OFFSET) ? (mant >> (23 - MANT_SIZE)) : MAX_MANT;
  uint16 hbits = sign | expf | mantf;
  return hbits;
}

/// Convert single-precision to chalf-precision.
/// \param value single-precision value
/// \return binary representation of chalf-precision value
uint16 float2chalf(float value)
{
  return float2chalf_impl(value);
}

/// Convert chalf-precision to IEEE single-precision.
/// Credit for this goes to [Jeroen van der Zijp](ftp://ftp.fox-toolkit.org/pub/fastchalffloatconversion.pdf).
/// \param value binary representation of chalf-precision value
/// \return single-precision value
inline float chalf2float_impl(uint16 value)
{
#if HALF_ENABLE_CPP11_STATIC_ASSERT
  static_assert(std::numeric_limits<float>::is_iec559, "chalf to float conversion needs IEEE 754 conformant 'float' type");
  static_assert(sizeof(uint32)==sizeof(float), "chalf to float conversion needs unsigned integer type of exactly the size of a 'float'");
#endif
  float out;
  uint32 sign = (value & 0x8000) << 16;
  uint16 mant = value & MANT_MASK_HP;
  uint16 exp = (value >> MANT_SIZE) & MAX_EXP;
  uint32 expf = (exp != 0) ? (exp + 127 - EXP_OFFSET) << 23 : 0;
  uint32 mantf = (exp != 0) ? (mant) << (23 - MANT_SIZE) : 0;
  uint32 bits = sign | expf | mantf;
  uint32 *temp = &bits;

  out = *((float *)temp);
  return out;
}

/// Convert chalf-precision to single-precision.
/// \param value binary representation of chalf-precision value
/// \return single-precision value
inline float chalf2float(uint16 value)
{
  return chalf2float_impl(value);
}

class chalf {
  friend chalf operator*(chalf T, chalf U);
  friend chalf operator+(chalf T, chalf U);
  friend chalf max(chalf T, chalf U);
  friend chalf max(chalf T);
  friend bool operator==(chalf T, chalf U);
  friend bool operator!=(chalf T, chalf U);
  public:
    chalf() : data_() {}

    chalf(float rhs) : data_(float2chalf(rhs)) {}
    
    chalf(uint16 rhs) : data_(rhs) {}

    chalf(int rhs) : data_(rhs) {}

    operator float() const {
      return chalf2float(data_);
    }

    operator uint16() const {
      return data_;
    }

    chalf& operator=(const int& rhs) {
      this->data_ = rhs;
      return *this;
    }

    chalf& operator+=(const chalf& rhs) {
      *this = *this + rhs;
      return *this;
    }
 
  private:
    uint16 data_;
};

chalf operator*(chalf T, chalf U) {
#pragma HLS INLINE off
#pragma HLS pipeline
  uint16 Tdata_ = T.data_;
  uint16 Udata_ = U.data_;
  ap_uint<EXP_SIZE> e1 = (Tdata_) >> EXP_SHIFT_HP;
  ap_uint<EXP_SIZE> e2 = (Udata_) >> EXP_SHIFT_HP;
  ap_uint<MANT_SIZE + 1> mant1 = Tdata_ | MANT_NORM_HP;// 11 bits
  ap_uint<MANT_SIZE + 1> mant2 = Udata_ | MANT_NORM_HP;// 11 bits
  ap_uint<1> sign1 = (Tdata_) >> 15;
  ap_uint<1> sign2 = (Udata_) >> 15;
  ap_uint<1> sign_res = sign1 ^ sign2;

  ap_uint<16> sign = sign_res;
  ap_uint<MANT_SIZE + 2> mantres;
  ap_uint<MANT_SIZE> mantresf;
  uint16 eresf;
  ap_uint<PRODUCT_SIZE> product = mant1 * mant2; // 22 bits
  mantres = product >> MANT_SIZE; // 11 bits
  ap_int<EXP_SIZE + 2> eres = e1 + e2 - EXP_OFFSET;

  // normalize
  if ((mantres >> (MANT_SIZE + 1)) & 0x1) {
    mantres = (product >> (MANT_SIZE + 1));
    eres++;
  }

  ap_uint<EXP_SIZE> eres_t;

  eres_t = eres;
  mantresf = mantres;
  if (eres >= MAX_EXP) {
    // saturate results
    eres_t = MAX_EXP - 1;
    mantresf = MAX_MANT;
  } else if ((e1 == 0) || (e2 == 0) || (eres < 0)) {
    // 0 * val, underflow
    eres_t = 0;
    mantresf = 0;
  }

  eresf = eres_t;

  uint16 res;

  res = ((sign << 15) & SIGN_MASK_HP) |
    ((eresf << EXP_SHIFT_HP) & EXP_MASK_HP) | mantresf;

  return chalf(res);
}

ap_uint<4> LZD(ap_uint<12> sum_cpath) {
#pragma HLS INLINE 
  ap_uint<4> Lshifter;
  ap_uint<4> b_1_o, b_0_o;
  ap_uint<1> a[12];
  ap_uint<4> b_1[4];
  ap_uint<4> b_0[4];
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

  ap_uint<1> b_1_sel = a[11] | a[10] | a[9] | a[8] | a[7] | a[6];
  ap_uint<1> b_0_sel = a[5] | a[4] | a[3] | a[2] | a[1] | a[0];

  b_1[3] = 0;
  b_1[2] = ~a[11] & ~a[10] & ~a[9] & ~a[8] & (a[7] | a[6]);
  b_1[1] = ~a[11] & ~a[10] & (a[9] | a[8]);
  b_1[0] = ~a[11] & (~a[9] & (~a[7] | a[8]) | a[10]);

  b_1_o = ((b_1[3] & 0x1) << 3) | ((b_1[2] & 0x1) << 2) |
    ((b_1[1] & 0x1) << 1) | (b_1[0] & 0x1);

  b_0[3] = ~a[5] & ~a[4];
  b_0[2] = a[5] | a[4];
  b_0[1] = ~(~a[5] & ~a[4] & (a[3] | a[2]));
  b_0[0] = ~a[5] & a[4] | (~a[5] & ~a[4] & ~a[3] & a[2]) |
    (~a[5] & ~a[4] & ~a[3] & ~a[2] & ~a[1] & a[0]);

  b_0_o = ((b_0[3] & 0x1) << 3) | ((b_0[2] & 0x1) << 2) |
    ((b_0[1] & 0x1) << 1) | (b_0[0] & 0x1);

  if (b_1_sel)
    Lshifter = b_1_o;
  else if (b_0_sel) 
    Lshifter = b_0_o;
  else 
    Lshifter = 12;

  return Lshifter;
}

chalf operator+(chalf T, chalf U) {
#pragma HLS INLINE off
#pragma HLS pipeline
  uint16 Tdata_ = T.data_;
  uint16 Udata_ = U.data_;
  ap_uint<EXP_SIZE> e1 = Tdata_ >> EXP_SHIFT_HP;
  ap_uint<EXP_SIZE> e2 = Udata_ >> EXP_SHIFT_HP;
  ap_uint<MANT_SIZE> mant1 = Tdata_;
  ap_uint<MANT_SIZE> mant2 = Udata_;
  ap_uint<1> sign1 = Tdata_ >> 15;
  ap_uint<1> sign2 = Udata_ >> 15;

  // EOP = 1 -> add, EOP = 0 -> sub
  ap_uint<1> EOP = sign1 == sign2; 

  // 1 if e1 is bigger, 0 if e2 is bigger
  ap_uint<1> exp_cmp = (e1 >= e2) ? 1 : 0;
  ap_uint<1> guard, round;
  ap_uint<MANT_SIZE> mantresf;
  ap_uint<15> eresf;

  ap_uint<MANT_SIZE + 4> sum_fpath;
  ap_uint<1> sum_fpath_sign = 0;

  ap_int<2> Rshifter = 0;
  ap_uint<4> Lshifter = 0;
 
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

  ap_uint<13> mant1_large = mant1_s | MANT_NORM_HP;
  ap_uint<PRODUCT_SIZE> mant2_large = mant2_s | MANT_NORM_HP;

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

  ap_uint<13> sum_cpath;

  if (sum_cpath_t < 0) {
    sum_cpath = -1 * sum_cpath_t;
    sum_cpath_sign = sign2_s; 
  } else {
    sum_cpath = sum_cpath_t;
    sum_cpath_sign = sign1_s;
  }

  ap_uint<4> lshift_t;
  lshift_t = LZD(sum_cpath);

  Lshifter = lshift_t - (10 - MANT_SIZE);

  ap_uint<MANT_SIZE> sum_cpath_f = ((sum_cpath) << Lshifter) >> 1;

  // Far path

  // saturate difference at 11 bits

  ap_uint<EXP_SIZE> diff_sat = (diff > (MANT_SIZE + 1)) ?
    (ap_uint<EXP_SIZE>)(MANT_SIZE + 1) : diff;

  ap_uint<PRODUCT_SIZE> mant2_a =
    (mant2_large) << ((MANT_SIZE + 1) - diff_sat);

  ap_uint<MANT_SIZE + 3> mant1_fpath = (mant1_large) << 2;
  ap_uint<MANT_SIZE + 3> mant2_fpath = (mant2_a >> (MANT_SIZE - 1));
 
  guard = (mant2_fpath >> 1) & 0x1;
  round = (mant2_fpath) & 0x1;
  ap_uint<1> last = (mant2_fpath >> 2) & 0x1;

  ap_uint<1> rnd_flag = guard & (round | last);

  ap_uint<3> off = (rnd_flag) ? 4 : 0;

  if (EOP) 
    sum_fpath = mant1_fpath + mant2_fpath + off;
  else
    sum_fpath = mant1_fpath - mant2_fpath; 

  ap_uint<MANT_SIZE + 2> sum_t = (sum_fpath >> 2);
  guard = (sum_fpath >> 2) & 0x1;
  round = (sum_fpath >> 1) & 0x1;
  rnd_flag = (guard & round);

  if ((sum_t >> (MANT_SIZE + 1)) & 0x1) {
    Rshifter = 1;
    sum_t = (sum_fpath >> 3) | (rnd_flag);
  } else if (((sum_t >> (MANT_SIZE)) & 0x1) == 0) {
    Rshifter = -1;
    sum_t = sum_fpath >> 1;
  }

  ap_uint<MANT_SIZE> sum_fpath_f = sum_t;

  ap_uint<16> sign = (fpath_flag) ? sign1_s : sum_cpath_sign;

  ap_uint<EXP_SIZE> eres_t;

  ap_uint<EXP_SIZE> eres_fpath_f = eres + Rshifter;
  ap_uint<EXP_SIZE> eres_cpath_f = eres - Lshifter;

  if (fpath_flag) {
    eres_t = eres_fpath_f;
    mantresf = sum_fpath_f;
    if (eres + Rshifter >= MAX_EXP) {
      eres_t = MAX_EXP - 1;
      mantresf = MAX_MANT;
    } else {
      eres_t = eres_fpath_f;
      mantresf = sum_fpath_f;
    }
  } else {
    if ((eres - Lshifter < 1) || (Lshifter == (MANT_SIZE + 2))) {
      eres_t = 0;
      mantresf = 0;
    } else {
      eres_t = eres_cpath_f;
      mantresf = sum_cpath_f;
    }
  }

  eresf = eres_t;

  uint16 res;
  res = ((sign << 15) & SIGN_MASK_HP) |
    ((eresf << EXP_SHIFT_HP) & EXP_MASK_HP) | mantresf;

  return chalf(res);
}

bool operator==(chalf T, chalf U) {
  uint16 Tdata_ = T.data_;
  uint16 Udata_ = U.data_;

  return (Tdata_ == Udata_);
}

bool operator!=(chalf T, chalf U) {
  uint16 Tdata_ = T.data_;
  uint16 Udata_ = U.data_;

  return (Tdata_ != Udata_);
}

chalf max(chalf T, chalf U) {
#pragma HLS INLINE off
#pragma HLS pipeline
  short Tdata_ = T.data_;
  short Udata_ = U.data_;
  chalf res;

  if (Tdata_ < Udata_)
    res = U;
  else
    res = T;  
  return res;
}

// Compares T with 0
chalf max(chalf T) {
#pragma HLS INLINE off
#pragma HLS pipeline
  uint16 Tdata_ = T.data_;
  ap_uint<1> sign1 = Tdata_ >> 15;

  chalf res;
  if (sign1)
    res = chalf(0);
  else
    res = T;
  return res;
}
