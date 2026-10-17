#include "_accel_common.h"
#include "_fma_inline.h"

#include <stdexcept>
#include <string>
#if defined(_MSC_VER) && defined(__AVX2__)
#include <immintrin.h>
#endif

// The fused elementwise complex multiply (`momwire._cmul.cmul`).
//
// numpy's elementwise complex `*` rounds differently by CPU (momwire#1497).
// Its x86-64-v3, AVX-512 and ARM ASIMD loops are fused:
//
//     re = fma(a.re, b.re, -(a.im * b.im))
//     im = fma(a.re, b.im,   a.im * b.re)
//
// while its baseline x86-64 loop rounds the product and the sum separately.
// This is that fused formula, spelled out, so a run's last bits no longer
// depend on which loop numpy dispatched. On the CPUs where numpy fuses, the
// result is numpy's to the bit.
//
// Every build fuses, which is the difference from `_fma_inline.h`, whose
// helpers fall back to the unfused expression where there is no hardware FMA:
//
// - GCC/clang, AVX2 variant (`-mfma`): `std::fma` is one vfmadd instruction.
// - GCC/clang, x86-64 baseline `_sse2` variant (no `-mfma`): `mw_fma::exact`,
//   the inline emulated fma of `_fma_inline.h` (libm's `fma` is software
//   there, and slow); correctly rounded, so the same bits.
// - arm64 (macOS, Linux): `std::fma` is one fmadd instruction.
// - MSVC with /arch:AVX2 (the Windows `_avx2` variant): `_mm_fmadd_sd`, so
//   the fused instruction is written into the source rather than left to how
//   /fp:fast treats a call to `fma`.
// - MSVC without it (the Windows `_sse2` variant): `mw_fma::exact`'s
//   emulation again, in place of the UCRT's software `fma`.
//
// The plain products `a.im * b.im` and `a.im * b.re` must round on their own,
// as numpy's do. The GCC/clang builds pass -ffp-contract=off, so nothing is
// fused that the source does not spell; MSVC's /fp:fast is switched off for
// this file by the pragmas below, as in `_accel_row_scatter.cpp`.
#if defined(_MSC_VER)
#pragma float_control(precise, on)
#pragma fp_contract(off)
#endif

namespace cmul {

typedef std::complex<double> cd;

static inline double fused(double a, double b, double c) {
#if defined(_MSC_VER) && defined(__AVX2__)
    return _mm_cvtsd_f64(_mm_fmadd_sd(_mm_set_sd(a), _mm_set_sd(b), _mm_set_sd(c)));
#else
    return mw_fma::exact(a, b, c);
#endif
}

// a * b elementwise over two 1-D complex128 arrays of one length. The Python
// side (`momwire._cmul`) broadcasts and flattens first.
static py::array_t<cd> fused_cmul(py::array_t<cd, py::array::c_style | py::array::forcecast> a,
                                  py::array_t<cd, py::array::c_style | py::array::forcecast> b) {
    if (a.ndim() != 1 || b.ndim() != 1 || a.size() != b.size())
        throw std::runtime_error("fused_cmul: a and b must be 1-D and of one length");
    const py::ssize_t n = a.size();
    py::array_t<cd> out(n);
    const double *x = reinterpret_cast<const double *>(a.data());
    const double *y = reinterpret_cast<const double *>(b.data());
    double *z = reinterpret_cast<double *>(out.mutable_data());
    for (py::ssize_t i = 0; i < n; ++i) {
        const double ar = x[2 * i], ai = x[2 * i + 1];
        const double br = y[2 * i], bi = y[2 * i + 1];
        const double ii = ai * bi;
        const double ir = ai * br;
        z[2 * i] = fused(ar, br, -ii);
        z[2 * i + 1] = fused(ar, bi, ir);
    }
    return out;
}

// The test hooks of `mw_fma::exact` (tests/test_exact_fma.py): fma over
// three 1-D float64 arrays of one length, by `route` -- "exact" (what the
// call sites get on this build), "soft" (the emulation itself, on any build
// that compiles it) or "libm" (`std::fma`, the reference).
static py::array_t<double> fma_route(const std::string &route,
                                     py::array_t<double, py::array::c_style | py::array::forcecast> a,
                                     py::array_t<double, py::array::c_style | py::array::forcecast> b,
                                     py::array_t<double, py::array::c_style | py::array::forcecast> c) {
    if (a.ndim() != 1 || b.ndim() != 1 || c.ndim() != 1 || a.size() != b.size() ||
        a.size() != c.size())
        throw std::runtime_error("fma_route: a, b and c must be 1-D and of one length");
    const py::ssize_t n = a.size();
    py::array_t<double> out(n);
    const double *x = a.data(), *y = b.data(), *z = c.data();
    double *o = out.mutable_data();
    if (route == "exact") {
        for (py::ssize_t i = 0; i < n; ++i) o[i] = mw_fma::exact(x[i], y[i], z[i]);
    } else if (route == "soft") {
#if MW_FMA_HAVE_SOFT_EXACT
        for (py::ssize_t i = 0; i < n; ++i) o[i] = mw_fma::soft_exact(x[i], y[i], z[i]);
#else
        throw std::runtime_error("fma_route: this build has no soft_exact");
#endif
    } else if (route == "libm") {
        for (py::ssize_t i = 0; i < n; ++i) o[i] = std::fma(x[i], y[i], z[i]);
    } else {
        throw std::runtime_error("fma_route: route is exact, soft or libm");
    }
    return out;
}

}  // namespace cmul

void register_cmul(py::module_ &m) {
    m.def("fused_cmul", &cmul::fused_cmul, py::arg("a"), py::arg("b"),
          "a * b over two 1-D complex128 arrays of one length, each part one fused "
          "multiply-add: re = fma(a.re, b.re, -(a.im*b.im)), im = fma(a.re, b.im, "
          "a.im*b.re), numpy's own formula on its FMA loops (momwire#1497).");
    m.def("fma_route", &cmul::fma_route, py::arg("route"), py::arg("a"), py::arg("b"),
          py::arg("c"),
          "fma(a, b, c) elementwise by route: 'exact' (mw_fma::exact as this build "
          "compiles it), 'soft' (its emulation) or 'libm' (std::fma). Test hook.");
    m.attr("fma_exact_soft") = bool(MW_FMA_EXACT_SOFT);
    m.attr("fma_have_soft_exact") = bool(MW_FMA_HAVE_SOFT_EXACT);
}
