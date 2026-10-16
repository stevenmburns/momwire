// The Sommerfeld branch cut, in one place (momwire#714).
//
// `_sommerfeld._gamma`: (lam^2 - k^2)^{1/2} with vertical cuts running DOWN
// from +k and UP from -k. Transcribed as the two-sqrt product, NOT collapsed
// to sqrt(lam^2 - k^2): the collapsed form has its cut on the SEGMENT between
// the branch points, and the legitimacy of both the head's first-quadrant
// detour and the rotated rays is that they cross neither of the vertical ones.
// The spelling IS the branch choice.
//
// That last sentence is why this file exists. Until #714 the spelling stood as
// three formula-equivalent internal-linkage copies -- `somm::gam`,
// `mw568_below::gamma_cut`, `mw680::gamma_cut` -- across two extensions. Two
// copies of a branch choice can drift; one cannot. A drifted copy would not
// fail loudly: it would move the cut in ONE translation unit and return
// wrong-sheet integrals that only that section's gates could catch.
//
// Deliberately leaner than `_accel_somm_proj_inline.h` (which #714 first
// proposed as the home): one of the three call sites is in the SEPARATE
// `_near_interface_accel` extension, whose include set is kept minimal, so
// this header takes `<cmath>`, `<complex>` and `_fma_inline.h` (itself
// `<cmath>` and `<complex>` only) and nothing else -- no pybind11, no OpenMP
// declarations, no cancellation machinery. Same discipline as
// `_contour_engine_inline.h`: it must not depend on the includer having
// defined `_USE_MATH_DEFINES` first (MSVC).
//
// `static inline` keeps the monolith's semantics exactly: each including TU
// gets its own internal-linkage copy, so section-local inlining is preserved
// and there is no ODR surface across the two extensions.
#pragma once

#include <cmath>
#include <complex>

#include "_fma_inline.h"

namespace mw_branch {

// The imaginary unit. Named as in the call sites this replaces (`CI` in the
// somm family, `MW_BJ` in the mw568/mw680 twins -- both were (0.0, 1.0)).
static const std::complex<double> MW_BRANCH_J(0.0, 1.0);

// The principal square root the cut is built from, without the library call
// (momwire#1441 step 2). glibc's csqrt was 15 % of the below/below table
// fill's self time, its hypot another 8 %, for four square roots per contour
// node, all of them on operands many decades inside the double range.
//
// This is glibc's own algorithm (s_csqrt_template.c), branch for branch, with
// two things taken out: hypot becomes sqrt(x*x + y*y), and the operand
// scaling and inf/nan/zero classification go to a guard. Inside the guard
// (max(|x|, |y|) in [2^-500, 2^500]) nothing it squares can over- or
// underflow, so the only difference from glibc is hypot's rounding against
// the plain sum's: at most an ulp in |z|, which reaches the root as a few
// ulps at most. Outside the guard (zero, non-finite, extreme) std::sqrt
// itself runs.
//
// The cut is std::sqrt's, which is what gamma_cut's two-sqrt spelling relies
// on: Re >= 0, and Im takes the sign of the operand's Im, signed zero
// included, so a point on the negative real axis lands on +j or -j by the
// sign of its zero. A purely imaginary operand (x == 0) gets glibc's
// dedicated branch, (t, +-t) from one sqrt: the general formula would form
// the second part as y / (2t) and can differ from t in the last bit, which
// is the numpy-vs-cmath trap for purely imaginary roots.
// `tests/test_lean_sqrt_1441.py` pins all of this against std::sqrt.
static inline std::complex<double> lean_sqrt(const std::complex<double> &z) {
    const double x = z.real(), y = z.imag();
    const double ax = std::fabs(x), ay = std::fabs(y);
    const double m = ax > ay ? ax : ay;
    if (!(m >= 3.054936363499605e-151 && m <= 3.273390607896142e+150))
        return std::sqrt(z);
    if (x == 0.0) {
        const double t = std::sqrt(0.5 * ay);
        return std::complex<double>(t, std::copysign(t, y));
    }
    const double d = std::sqrt(x * x + y * y);
    if (x > 0.0) {
        const double r = std::sqrt(0.5 * (d + x));
        const double s = 0.5 * (y / r);
        return std::complex<double>(r, std::copysign(s, y));
    }
    const double s = std::sqrt(0.5 * (d - x));
    const double r = std::fabs(0.5 * (y / s));
    return std::complex<double>(r, std::copysign(s, y));
}

static inline std::complex<double> gamma_cut(const std::complex<double> &lam,
                                             const std::complex<double> &k) {
    // The three products are the fused mw_fma::mul (momwire#1194): this runs
    // twice per node of every Sommerfeld, below and near-interface integrand,
    // and was the largest unfused remainder in the grid fill once the build
    // stopped contracting. Same factors, same order, same branch. The roots
    // are lean_sqrt (momwire#1441): std::sqrt's cut, without the call.
    return mw_fma::mul(lean_sqrt(mw_fma::mul(-MW_BRANCH_J, lam - k)),
                       lean_sqrt(mw_fma::mul(MW_BRANCH_J, lam + k)));
}

}  // namespace mw_branch
