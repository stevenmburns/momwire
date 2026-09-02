// Cancellation-free spellings, C++ side. The twin of `_stable.py`, which
// carries the sibling table and the measurements (momwire#799).
//
// Every function here is written operation for operation against its numpy
// counterpart, in the same multi-step order, because the cross-lane agreement
// tests bound the difference between the two and a kernel that shares a
// spelling with its twin has nothing to bound but the reduction order.
//
// Its own header rather than a block in `_accel_common.h`: that file is
// included by every translation unit, and the #687 split exists so that
// editing one kernel does not ccache-miss all of them.
#pragma once

#include <cmath>

// cos(y) - 1 = -2 sin^2(y/2). The literal subtraction returns this to an
// absolute epsilon: at y = 1e-3 the answer is -5e-7 out of terms of size 1,
// so its relative error is 2e-10 and grows as 1/y^2.
//
// Costs nothing where it replaces a `cos`: the callers here want sin(y) as
// well, so `cos(y), sin(y)` becomes `sin(y/2), sin(y)` -- two transcendental
// calls either way.
static inline double stable_cos_minus_one(double y) {
    const double half = std::sin(0.5 * y);
    return -2.0 * half * half;
}

// e^{-jkR} - 1 with k = k_re + j*k_im, Im k <= 0. `_stable.expm1_neg_jkR`'s
// bracket: with a = k_im*R and y = k_re*R,
//
//   exp(-jkR) - 1 = [ expm1(a) cos(y) - 2 sin^2(y/2) ] - j [ exp(a) sin(y) ]
//
// `exp` and `expm1` are called separately rather than reconstructing one from
// the other, so the imaginary part is bitwise what the numpy lane's
// `np.exp(a) * np.sin(y)` computes.
static inline void stable_expm1_neg_jkR(double k_re, double k_im, double R,
                                        double *re, double *im) {
    const double a = k_im * R;
    const double y = k_re * R;
    double r = std::expm1(a) * std::cos(y);
    r = r + stable_cos_minus_one(y);
    *re = r;
    *im = -(std::exp(a) * std::sin(y));
}

// sqrt(u1^2 + rho^2) - sqrt(u0^2 + rho^2), rationalised to
// (u1^2 - u0^2)/(r1 + r0). An exact identity, not a series.
static inline double stable_sqrt_diff(double u0, double u1, double r0,
                                      double r1) {
    const double du = u1 - u0;
    const double su = u1 + u0;
    return du * su / (r1 + r0);
}

// asinh(u1/rho) - asinh(u0/rho) through log1p of the ratio, with
// p_i = u_i + r_i rationalised as rho^2/(r_i - u_i) where u_i < 0. The full
// derivation is in `_stable.asinh_diff`.
static inline double stable_asinh_diff(double u0, double u1, double rho2,
                                       double r0, double r1) {
    const double p0 = (u0 >= 0.0) ? (u0 + r0) : (rho2 / (r0 - u0));
    const double p1 = (u1 >= 0.0) ? (u1 + r1) : (rho2 / (r1 - u1));
    const double du = u1 - u0;
    const double num = du * (p1 + p0);
    return std::log1p(num / ((r1 + r0) * p0));
}
