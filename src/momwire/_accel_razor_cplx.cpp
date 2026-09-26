// razor's complex-k bracket, vectorised (Design D4). See the header.
//
// WHY THIS IS FAST
// ----------------
// The complex-k moment loop cost 5x the real-k loop per entry, and all of
// it was the transcendentals: the real loop vectorises onto libmvec (two
// `_ZGVdN4v_sin` per four nodes) while the complex one made four SCALAR libm
// calls per node -- expm1, exp, sincos and sin(y/2). Splitting the loop
// without changing the calls bought nothing (DESIGN-D §2, variant D-split),
// so the only lever is the calls themselves.
//
// This spells the bracket with three vector calls to two functions glibc
// 2.28's libmvec has -- exp and sin -- and nothing else:
//
//     h = sin(y/2)
//     cos(y) - 1 = -2 h^2           (the cancellation-free form, momwire#799)
//     sin(y)     =  sin(y)
//     exp(a)     =  exp(a)          (one vector call, reused below)
//     expm1(a)   =  Taylor(a)       a >= -0.35
//               =  exp(a) - 1      a <  -0.35
//
// `expm1` itself is NOT on the list, and that is the whole reason for the
// series: libmvec's `_ZGVdN4v_expm1` is GLIBC_2.35, and the Linux wheels
// build on manylinux_2_28 (glibc 2.28), whose libmvec has exp/log/pow/sin/
// cos/sincos only. A kernel that called it would link against a new glibc
// and fail to IMPORT from the wheel, never locally. tests/test_glibc_floor.py
// holds that line.
//
// ACCURACY
// --------
// expm1 for a in [-0.35, 0]: the degree-13 Taylor polynomial by Horner, in
// fused steps. Its truncation error is |a|^14/14! < 5.6e-18 absolute at
// a = -0.35, i.e. < 2e-17 relative to |expm1(a)| >= 0.295, and at the other
// end the relative error of `p * a` is a few ulp as a -> 0, which is the
// point of expm1. Below -0.35, exp(a) - 1 subtracts two numbers of which the
// larger is 1 and the result is at least 0.295 in magnitude, so the literal
// subtraction costs about two bits (|exp(a)|/|exp(a)-1| <= 2.4).
//
// The half-angle form is the one `_stable.expm1_neg_jkR` uses for the `- 1`
// already, and sin(y) is called directly, as there. All of it is a few ulp
// per node. It is NOT bit-identical to the scalar
// spelling it replaces (libmvec's exp/sin differ from scalar libm in the
// last bits anyway), which is why this change is gated on Z rather than
// proven by array_equal.
//
// THE SELECT
// ----------
// The switch at -0.35 is a bitwise blend on an all-ones/all-zeros mask, not
// `a >= -0.35 ? p : e`. GCC lowers the conditional to a branch, sinks each
// arm's arithmetic into it, and then will not if-convert it back under
// -ftrapping-math (the default), so the loop stays scalar ("control flow in
// loop"). The two ways around that the prototype considered were
// `__attribute__((optimize("no-trapping-math")))`, which the GCC manual
// calls unsuitable for production code, and -fno-trapping-math on this TU,
// which setuptools has no per-source hook for. The blend needs neither: both
// arms are computed unconditionally anyway, it changes no value (it selects
// one of two finished doubles by their bits), and it compiles to
// vandpd/vandnpd/vorpd. Measured on GCC 13: the conditional form vectorises
// only with -fno-trapping-math; this form vectorises at the build's flags.
//
// NO cos(y/2)
// -----------
// The prototype took sin(y) as 2 sin(y/2) cos(y/2). That vectorises on GCC 13
// and NOT on GCC 11 (Ubuntu 22.04): there the sincos pass merges sin and cos
// of one argument into a scalar `sincos` before the vectoriser runs, which
// has no simd declaration, and the whole loop stays scalar ("no vectype").
// sin(y/2) and sin(y) have different arguments, so nothing merges -- the
// real-k loop's pattern -- and the loop vectorises on both.
//
// PORTABILITY
// -----------
// The `declare simd` block is gated exactly as `_accel_common.h`'s: glibc
// libmvec only. Under MSVC, on macOS and in the x86-64 baseline (`_sse2`)
// build the loop is scalar -- or the platform's own vectoriser's -- and the
// same arithmetic runs through the platform libm. Those builds therefore
// differ from the AVX2 build in the last bits at complex k, as they already
// do at real k.
#include "_accel_razor_cplx.h"

#include <cmath>
#include <cstdint>
#include <cstring>

#include "_fma_inline.h"

#if defined(__GNUC__) && !defined(_MSC_VER) && !defined(__APPLE__)
#pragma omp declare simd notinbranch simdlen(4)
extern "C" double exp(double);

#pragma omp declare simd notinbranch simdlen(4)
extern "C" double sin(double);
#endif

// MSVC's /openmp:llvm rejects `omp simd` (see `_accel_common.h`).
#if defined(_MSC_VER)
#define MW_RC_SIMD
#else
#define MW_RC_SIMD _Pragma("omp simd aligned(R, re, im : 32)")
#endif

// `take_x ? x : y`, selected by bits (see THE SELECT above).
static inline double razor_cplx_blend(bool take_x, double x, double y) {
    uint64_t bx, by;
    std::memcpy(&bx, &x, sizeof bx);
    std::memcpy(&by, &y, sizeof by);
    const uint64_t m = static_cast<uint64_t>(0) - static_cast<uint64_t>(take_x);
    const uint64_t r = (bx & m) | (by & ~m);
    double out;
    std::memcpy(&out, &r, sizeof out);
    return out;
}

// The series' switch point. Above it (toward 0) the Taylor polynomial;
// below it exp(a) - 1.
static constexpr double RAZOR_CPLX_EXPM1_SWITCH = -0.35;

void razor_cplx_brackets(const double *R, size_t n, double k_re, double k_im,
                         double *re, double *im) {
    MW_RC_SIMD
    for (size_t q = 0; q < n; q++) {
        const double a = k_im * R[q];
        const double y = k_re * R[q];
        const double ea = std::exp(a);
        // expm1(a) = a * (1 + a/2! + a^2/3! + ... + a^12/13!), Horner.
        double p = 1.0 / 6227020800.0;                // 1/13!
        p = mw_fma::fma(p, a, 1.0 / 479001600.0);     // 1/12!
        p = mw_fma::fma(p, a, 1.0 / 39916800.0);      // 1/11!
        p = mw_fma::fma(p, a, 1.0 / 3628800.0);       // 1/10!
        p = mw_fma::fma(p, a, 1.0 / 362880.0);        // 1/9!
        p = mw_fma::fma(p, a, 1.0 / 40320.0);         // 1/8!
        p = mw_fma::fma(p, a, 1.0 / 5040.0);          // 1/7!
        p = mw_fma::fma(p, a, 1.0 / 720.0);           // 1/6!
        p = mw_fma::fma(p, a, 1.0 / 120.0);           // 1/5!
        p = mw_fma::fma(p, a, 1.0 / 24.0);            // 1/4!
        p = mw_fma::fma(p, a, 1.0 / 6.0);             // 1/3!
        p = mw_fma::fma(p, a, 0.5);                   // 1/2!
        p = mw_fma::fma(p, a, 1.0);                   // 1/1!
        const double em = razor_cplx_blend(a >= RAZOR_CPLX_EXPM1_SWITCH, p * a,
                                           ea - 1.0);
        const double h = std::sin(0.5 * y);
        const double sy = std::sin(y);
        const double cm1 = -2.0 * h * h;  // cos(y) - 1
        const double cy = 1.0 + cm1;      // cos(y)
        re[q] = em * cy + cm1;
        im[q] = -(ea * sy);
    }
}
