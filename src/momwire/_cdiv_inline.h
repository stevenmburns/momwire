// Complex division without the libgcc call (momwire#1441 step 1).
//
// With GCC, `std::complex<double>` division is a call to libgcc's
// `__divdc3` at every site: GCC only inlines it under -fcx-limited-range or
// -fcx-fortran-rules, and both of those are whole-TU switches. The
// Sommerfeld integrands divide four times per contour node (the D1/D2 pair),
// and the call alone was 11.5 % of the below/below table fill's self time.
//
// `mw_cdiv::div` is that division spelled out, BIT FOR BIT what `__divdc3`
// returns wherever it takes the inline path. libgcc (GCC 12 and later) is
// Smith's algorithm on the larger of |Re y|, |Im y|:
//
//     |c| < |d|:  ratio = c / d;  denom = c * ratio + d
//                 re = (a * ratio + b) / denom;  im = (b * ratio - a) / denom
//     otherwise:  ratio = d / c;  denom = d * ratio + c
//                 re = (b * ratio + a) / denom;  im = (b - a * ratio) / denom
//
// for x = a + jb, y = c + jd, plus branches that scale all four operands by a
// power of two (when |y| is tiny or huge, or a numerator part is below
// DBL_MIN, which includes an exact zero) and an alternative order when
// `ratio` is subnormal. A power-of-two scaling is exact and cancels in the
// quotient as long as nothing over- or underflows, so inside the guarded
// range below every one of those branches rounds exactly like the plain
// formula. A real or imaginary y (one part exactly zero) takes libgcc's
// subnormal-ratio order, spelled the same way. Outside that (y zero or
// non-finite, a nonzero part of x or y outside [2^-500, 2^500], or
// 0 < |ratio| < 2^-500) the operator itself runs.
// No build flag is involved, so no other division in any TU changes.
//
// The integrands' operands are wavenumbers, gammas and Bessel values in SI
// units, many decades inside that range, so the fallback is for robustness,
// not a path they take. `tests/test_cdiv_1441.py` pins the identity against
// the operator over random and edge operands.
//
// Clang (macOS) calls compiler-rt's `__divdc3`, a different algorithm
// (logb/scalbn scaling, C99 Annex G), and MSVC divides inline in its own
// header; neither would round like the formula above. Those compilers keep
// the operator, so their bits do not move.
#pragma once

#include <cmath>
#include <complex>

#include "_fma_inline.h"

namespace mw_cdiv {

#if defined(__GNUC__) && !defined(__clang__)
// 2^-500 and 2^500: every product and quotient the formula forms from
// operands in this range is a normal double.
static const double MW_CDIV_LO = 3.054936363499605e-151;
static const double MW_CDIV_HI = 3.273390607896142e+150;

MW_FMA_INLINE bool in_range(double v) {
    const double a = std::fabs(v);
    return a >= MW_CDIV_LO && a <= MW_CDIV_HI;
}

MW_FMA_INLINE bool zero_or_in_range(double v) {
    return v == 0.0 || in_range(v);
}

MW_FMA_INLINE std::complex<double> div(const std::complex<double> &x,
                                       const std::complex<double> &y) {
    const double a = x.real(), b = x.imag();
    const double c = y.real(), d = y.imag();
    if (zero_or_in_range(a) && zero_or_in_range(b)) {
        if (std::fabs(c) < std::fabs(d)) {
            if (in_range(d) && zero_or_in_range(c)) {
                const double ratio = c / d;
                const double denom = (c * ratio) + d;
                if (std::fabs(ratio) >= MW_CDIV_LO)
                    return std::complex<double>(((a * ratio) + b) / denom,
                                                ((b * ratio) - a) / denom);
                if (ratio == 0.0)  // c == 0: libgcc's subnormal-ratio order
                    return std::complex<double>(((c * (a / d)) + b) / denom,
                                                ((c * (b / d)) - a) / denom);
            }
        } else if (in_range(c) && zero_or_in_range(d)) {
            const double ratio = d / c;
            const double denom = (d * ratio) + c;
            if (std::fabs(ratio) >= MW_CDIV_LO)
                return std::complex<double>(((b * ratio) + a) / denom,
                                            (b - (a * ratio)) / denom);
            if (ratio == 0.0)  // d == 0: libgcc's subnormal-ratio order
                return std::complex<double>((a + (d * (b / c))) / denom,
                                            (b - (d * (a / c))) / denom);
        }
    }
    // Outside the guarded range: the operator, i.e. libgcc itself.
    return x / y;
}
#else
MW_FMA_INLINE std::complex<double> div(const std::complex<double> &x,
                                       const std::complex<double> &y) {
    return x / y;
}
#endif

// The real-scalar instantiation of the templated callers (the contour
// engine's Bessel kernels run at T = double on the tail): a plain divide.
MW_FMA_INLINE double div(double x, double y) { return x / y; }

// s / y for a real s: libstdc++ forms it as complex(s, 0) / y, the same
// `__divdc3` call with b = 0.
MW_FMA_INLINE std::complex<double> div(double s, const std::complex<double> &y) {
    return div(std::complex<double>(s, 0.0), y);
}

}  // namespace mw_cdiv
