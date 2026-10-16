// `_wire_loading.series_impedance_per_wire` in C++ -- see the header.
#include "_razor_wire_loading_core.h"

// MSVC builds every TU /fp:fast (setup.py); this one replays numpy's
// arithmetic one IEEE operation at a time. GCC / clang: -ffp-contract=off.
#if defined(_MSC_VER)
#pragma float_control(precise, on)
#pragma fp_contract(off)
#endif

#include "xsf/bessel.h"

#include <cfloat>
#include <cmath>
#include <stdexcept>

namespace mw {
namespace razor {

namespace {

constexpr double PI = 3.141592653589793;
constexpr double MU0 = 1.25663706127e-6;  // `_wire_loading.MU0`

// numpy's array-loop complex multiply (fused, its x86-64-v3 loop: the C++
// route runs only where numpy takes it).
cd fused_mul(cd a, cd b) {
    return cd(std::fma(a.real(), b.real(), -(a.imag() * b.imag())),
              std::fma(a.real(), b.imag(), a.imag() * b.real()));
}

// numpy's scalar math (`cdouble_ctype_multiply`): two roundings.
cd scalar_mul(cd a, cd b) {
    return cd(a.real() * b.real() - a.imag() * b.imag(), a.real() * b.imag() + a.imag() * b.real());
}

// numpy's complex divide (`CDOUBLE_divide`; the scalar divide is the same).
cd np_div(cd a, cd b) {
    const double in1r = a.real(), in1i = a.imag(), in2r = b.real(), in2i = b.imag();
    const double in2r_abs = std::fabs(in2r), in2i_abs = std::fabs(in2i);
    if (in2r_abs >= in2i_abs) {
        if (in2r_abs == 0 && in2i_abs == 0) return cd(in1r / in2r_abs, in1i / in2i_abs);
        const double rat = in2i / in2r;
        const double scl = 1.0 / (in2r + in2i * rat);
        return cd((in1r + in1i * rat) * scl, (in1i - in1r * rat) * scl);
    }
    const double rat = in2r / in2i;
    const double scl = 1.0 / (in2i + in2r * rat);
    return cd((in1r * rat + in1i) * scl, (in1i * rat - in1r) * scl);
}

// numpy's complex sqrt: libm's `csqrt` (measured equal on 29,000 values,
// the purely imaginary arguments a conductor's (j omega mu sigma) is among
// them -- where CPython's `cmath.sqrt` gives other bits on half of them).
cd np_sqrt(cd z) {
#if defined(__GNUC__)
    __complex__ double c;
    __real__ c = z.real();
    __imag__ c = z.imag();
    c = __builtin_csqrt(c);
    return cd(__real__ c, __imag__ c);
#else
    return std::sqrt(z);
#endif
}

inline cd add(cd a, cd b) { return cd(a.real() + b.real(), a.imag() + b.imag()); }
inline bool truthy(double v) { return v != 0.0 || std::isnan(v); }

// `wire_internal_impedance(omega, radius, sigma)`: `omega` a 0-d array, so
// `1j * omega` is numpy's array loop and every step after it scalar math.
cd wire_internal_impedance(double omega, double radius, double sigma) {
    if (!(radius > 0.0)) throw std::invalid_argument("wire loading: a wire radius <= 0");
    if (!(sigma > 0.0)) throw std::invalid_argument("wire loading: a conductivity <= 0");
    cd t = fused_mul(cd(0.0, 1.0), cd(omega, 0.0));
    t = scalar_mul(t, cd(MU0, 0.0));
    t = scalar_mul(t, cd(sigma, 0.0));
    const cd kc = np_sqrt(t);
    const cd z = scalar_mul(kc, cd(radius, 0.0));
    const cd ratio = np_div(xsf::cyl_bessel_ie(0.0, z), xsf::cyl_bessel_ie(1.0, z));
    const double d = ((2.0 * PI) * radius) * sigma;
    return scalar_mul(np_div(kc, cd(d, 0.0)), ratio);
}

// `rlc_impedance(kind, r, l, c, omega)` with `omega` a 0-d array (the
// per-metre reading, `DistributedRLC.impedance_per_metre`).
cd rlc_per_metre(const WireRLC &e, double omega) {
    const auto j_omega = [&](double v) {
        return scalar_mul(fused_mul(cd(0.0, 1.0), cd(omega, 0.0)), cd(v, 0.0));
    };
    if (e.kind == WireRLC::series) {
        const cd p = fused_mul(cd(0.0, 0.0), cd(omega, 0.0));  // `0j * omega`
        cd z(e.r + p.real(), 0.0 + p.imag());
        if (truthy(e.l)) z = add(z, j_omega(e.l));
        if (truthy(e.c)) z = add(z, np_div(cd(1.0, 0.0), j_omega(e.c)));
        return z;
    }
    cd y(0.0, 0.0);
    if (truthy(e.r)) y = cd(y.real() + 1.0 / e.r, y.imag() + 0.0);
    if (truthy(e.l)) y = add(y, np_div(cd(1.0, 0.0), j_omega(e.l)));
    if (truthy(e.c)) y = add(y, j_omega(e.c));
    return np_div(cd(1.0, 0.0), y);
}

}  // namespace

std::vector<cd> series_impedance_per_wire(double omega, const std::vector<double> &radius,
                                          const WireLoading &loading) {
    const std::size_t n_w = radius.size();
    std::vector<cd> out(n_w, cd(0.0, 0.0));
    for (std::size_t w = 0; w < n_w; w++) {
        if (loading.has_conductivity && std::isfinite(loading.conductivity[w]))
            out[w] = add(out[w], wire_internal_impedance(omega, radius[w], loading.conductivity[w]));
        if (loading.has_rlc && loading.rlc[w].kind != WireRLC::none)
            out[w] = add(out[w], rlc_per_metre(loading.rlc[w], omega));
    }
    return out;
}

}  // namespace razor
}  // namespace mw
