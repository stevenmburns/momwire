// The complex-k Galerkin far fill's transcendental sweep, vectorised
// (momwire#1224). See the header for the contract.
//
// The shape is `_accel_razor_cplx.cpp`'s (Design D4), and its notes apply
// unchanged; in brief:
//
//   * ONLY exp and sin. glibc 2.28's libmvec (the manylinux_2_28 wheels'
//     floor, tests/test_glibc_floor.py) has exp/log/pow/sin/cos/sincos and
//     nothing else, so expm1 is a degree-13 Taylor series near 0 and
//     exp(a) - 1 beyond it.
//   * NO cos. GCC 11 merges sin and cos of one argument into a scalar
//     `sincos` before the vectoriser runs and the loop then stays scalar;
//     sin(y/2) and sin(y) have different arguments, so nothing merges, and
//     cos y is rebuilt by the caller as 1 - 2 sin^2(y/2).
//   * The series switch is a bitwise blend, not `?:`, which GCC lowers to a
//     branch it will not if-convert under -ftrapping-math.
//
// The switch here is |a| <= 0.35 rather than razor's a >= -0.35, because this
// table carries SIGNED distances (the quadrature nodes' offsets from their
// endpoint, and the E_rho reference angle's), so a = Im(k)*d is positive as
// often as it is negative. On the far side of the switch exp(a) - 1 costs at
// most ~2 bits below (|e^a|/|e^a - 1| <= 2.4 at a = -0.35) and none above.
// The series' truncation error is |a|^14/14! < 5.6e-18 at |a| = 0.35.
//
// PORTABILITY is `_accel_common.h`'s: the `declare simd` block is glibc
// libmvec only. Under MSVC, on macOS and in the x86-64 baseline (`_sse2`)
// build the loop is scalar (or the platform's own vectoriser's) and runs
// through the platform libm, so those builds differ from the AVX2 build in
// the last bits at complex k, exactly as they already do at real k.
#include "_accel_sinusoidal_cplx.h"

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
#define MW_SGC_SIMD
#else
#define MW_SGC_SIMD _Pragma("omp simd aligned(d, ea, em, s, h : 32)")
#endif

// `take_x ? x : y`, selected by bits (razor_cplx_blend's spelling).
static inline double sg_cplx_blend(bool take_x, double x, double y) {
    uint64_t bx, by;
    std::memcpy(&bx, &x, sizeof bx);
    std::memcpy(&by, &y, sizeof by);
    const uint64_t m = static_cast<uint64_t>(0) - static_cast<uint64_t>(take_x);
    const uint64_t r = (bx & m) | (by & ~m);
    double out;
    std::memcpy(&out, &r, sizeof out);
    return out;
}

static constexpr double SG_CPLX_EXPM1_SWITCH = 0.35;

void sg_cplx_phase_sweep(const double *d, size_t n, double k_re, double k_im,
                         double *ea, double *em, double *s, double *h) {
    MW_SGC_SIMD
    for (size_t i = 0; i < n; i++) {
        const double a = k_im * d[i];
        const double y = k_re * d[i];
        const double e = std::exp(a);
        // expm1(a) = a * (1 + a/2! + ... + a^12/13!), Horner, fused steps.
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
        ea[i] = e;
        em[i] = sg_cplx_blend(std::fabs(a) <= SG_CPLX_EXPM1_SWITCH, p * a,
                              e - 1.0);
        s[i] = std::sin(y);
        h[i] = std::sin(0.5 * y);
    }
}
