// The sinusoidal basis' point-matched kernels, Python-free (momwire#1607 B1):
// every field tensor `_accel_sinusoidal.cpp` binds, by its bound name and
// with its bound arguments, over `mw::nd` arrays (`_mw_nd.h`) instead of
// numpy's. The arithmetic is `_sinusoidal_core.cpp`'s, which is the text the
// extension TU compiled before the split; the extension now calls these.
//
// Arrays: an argument is a `mw::nd::array_t` over the caller's memory
// (`array_t::borrow`), C-contiguous; a result is allocated by the current
// host (`mw::nd::current_host()`: the default host allocates with `new`; the
// extension installs numpy's). Each returns the three tensors of the source
// shapes (const, sin, cos).
//
// Errors are C++ exceptions with the extension's wording: std::runtime_error
// for a bad call, std::domain_error / std::out_of_range for pybind11's own
// array checks, and the cancellation machinery's `mw::AbortedError`.
#pragma once

#include "_mw_nd.h"

#include <complex>
#include <cstddef>
#include <cstdint>
#include <tuple>

namespace mw {
namespace sinusoidal {

std::tuple<nd::array_t<std::complex<double>>, nd::array_t<std::complex<double>>, nd::array_t<std::complex<double>>>
sinusoidal_field_tensor(
    nd::array_t<double, nd::array::c_style | nd::array::forcecast> obs_centers,
    nd::array_t<double, nd::array::c_style | nd::array::forcecast> obs_tangents,
    nd::array_t<double, nd::array::c_style | nd::array::forcecast> src_centers,
    nd::array_t<double, nd::array::c_style | nd::array::forcecast> src_tangents,
    nd::array_t<double, nd::array::c_style | nd::array::forcecast> seg_h,
    double a,
    double k,
    double eta,
    nd::array_t<double, nd::array::c_style | nd::array::forcecast> gl_t,
    nd::array_t<double, nd::array::c_style | nd::array::forcecast> gl_w,
    uintptr_t cancel_flag);

std::tuple<nd::array_t<std::complex<double>>, nd::array_t<std::complex<double>>, nd::array_t<std::complex<double>>>
sinusoidal_field_tensor_refl(
    nd::array_t<double, nd::array::c_style | nd::array::forcecast> obs_centers,
    nd::array_t<double, nd::array::c_style | nd::array::forcecast> obs_tangents,
    nd::array_t<double, nd::array::c_style | nd::array::forcecast> src_centers,
    nd::array_t<double, nd::array::c_style | nd::array::forcecast> src_tangents,
    nd::array_t<double, nd::array::c_style | nd::array::forcecast> seg_h,
    double a,
    double k,
    double eta,
    nd::array_t<double, nd::array::c_style | nd::array::forcecast> gl_t,
    nd::array_t<double, nd::array::c_style | nd::array::forcecast> gl_w,
    nd::array_t<double, nd::array::c_style | nd::array::forcecast> cos_th,
    nd::array_t<double, nd::array::c_style | nd::array::forcecast> px,
    nd::array_t<double, nd::array::c_style | nd::array::forcecast> py_,
    nd::array_t<double, nd::array::c_style | nd::array::forcecast> tm_p,
    nd::array_t<double, nd::array::c_style | nd::array::forcecast> tn_p,
    std::complex<double> eps_t,
    uintptr_t cancel_flag);

std::tuple<nd::array_t<std::complex<double>>, nd::array_t<std::complex<double>>, nd::array_t<std::complex<double>>>
sinusoidal_field_tensor_ek(
    nd::array_t<double, nd::array::c_style | nd::array::forcecast> obs_centers,
    nd::array_t<double, nd::array::c_style | nd::array::forcecast> obs_tangents,
    nd::array_t<double, nd::array::c_style | nd::array::forcecast> src_centers,
    nd::array_t<double, nd::array::c_style | nd::array::forcecast> src_tangents,
    nd::array_t<double, nd::array::c_style | nd::array::forcecast> seg_h,
    double a,
    double k,
    double eta,
    nd::array_t<double, nd::array::c_style | nd::array::forcecast> gl_t,
    nd::array_t<double, nd::array::c_style | nd::array::forcecast> gl_w,
    nd::array_t<double, nd::array::c_style | nd::array::forcecast> src_a,
    nd::array_t<int8_t, nd::array::c_style | nd::array::forcecast> ind1,
    nd::array_t<int8_t, nd::array::c_style | nd::array::forcecast> ind2,
    uintptr_t cancel_flag);

std::tuple<nd::array_t<std::complex<double>>, nd::array_t<std::complex<double>>, nd::array_t<std::complex<double>>>
sinusoidal_field_tensor_ek_refl(
    nd::array_t<double, nd::array::c_style | nd::array::forcecast> obs_centers,
    nd::array_t<double, nd::array::c_style | nd::array::forcecast> obs_tangents,
    nd::array_t<double, nd::array::c_style | nd::array::forcecast> src_centers,
    nd::array_t<double, nd::array::c_style | nd::array::forcecast> src_tangents,
    nd::array_t<double, nd::array::c_style | nd::array::forcecast> seg_h,
    double a,
    double k,
    double eta,
    nd::array_t<double, nd::array::c_style | nd::array::forcecast> gl_t,
    nd::array_t<double, nd::array::c_style | nd::array::forcecast> gl_w,
    nd::array_t<double, nd::array::c_style | nd::array::forcecast> src_a,
    nd::array_t<int8_t, nd::array::c_style | nd::array::forcecast> ind1,
    nd::array_t<int8_t, nd::array::c_style | nd::array::forcecast> ind2,
    nd::array_t<double, nd::array::c_style | nd::array::forcecast> cos_th,
    nd::array_t<double, nd::array::c_style | nd::array::forcecast> px,
    nd::array_t<double, nd::array::c_style | nd::array::forcecast> py_,
    nd::array_t<double, nd::array::c_style | nd::array::forcecast> tm_p,
    nd::array_t<double, nd::array::c_style | nd::array::forcecast> tn_p,
    std::complex<double> eps_t,
    uintptr_t cancel_flag);

std::tuple<nd::array_t<std::complex<double>>, nd::array_t<std::complex<double>>, nd::array_t<std::complex<double>>>
sinusoidal_field_tensor_cplx(
    nd::array_t<double, nd::array::c_style | nd::array::forcecast> obs_centers,
    nd::array_t<double, nd::array::c_style | nd::array::forcecast> obs_tangents,
    nd::array_t<double, nd::array::c_style | nd::array::forcecast> obs_radius,
    nd::array_t<double, nd::array::c_style | nd::array::forcecast> src_centers,
    nd::array_t<double, nd::array::c_style | nd::array::forcecast> src_tangents,
    nd::array_t<double, nd::array::c_style | nd::array::forcecast> seg_h,
    std::complex<double> k,
    std::complex<double> eta,
    nd::array_t<double, nd::array::c_style | nd::array::forcecast> gl_t,
    nd::array_t<double, nd::array::c_style | nd::array::forcecast> gl_w,
    bool folded,
    uintptr_t cancel_flag);

// One CSC matrix of the Φ → Z contraction (`reduce_phi_band`'s operand):
// `indptr` (cols + 1), `indices`, and the complex data interleaved (re, im).
struct FillRowsCsc {
    const int64_t *ptr;
    const int64_t *ind;
    const double *val;
};

// `sinusoidal_fill_rows`' operands, all borrowed, C-contiguous where they are
// arrays. The kernel is the reduced one, or the extended one when `ek` (then
// `src_a`, `ind1`, `ind2` are the (N,) tables `sinusoidal_field_tensor_ek`
// takes). `img_c` / `img_t` are the PEC image's mirrored sources, or null
// for free space.
struct FillRowsArgs {
    size_t M = 0, N = 0, n_qp = 0, cols = 0;
    const double *obs_c = nullptr, *obs_t = nullptr;  // (M, 3)
    const double *a_rows = nullptr;                   // (M,) observer radius
    const double *src_c = nullptr, *src_t = nullptr;  // (N, 3)
    const double *img_c = nullptr, *img_t = nullptr;  // (N, 3) or null
    const double *seg_h = nullptr;                    // (N,)
    double k = 0, eta = 0;
    const double *gl_t = nullptr, *gl_w = nullptr;    // (n_qp,)
    bool ek = false;
    const double *src_a = nullptr;
    const int8_t *ind1 = nullptr, *ind2 = nullptr;
    FillRowsCsc csc[3] = {};
    std::complex<double> *out = nullptr;  // out[m * out_rs + j * out_cs]
    std::ptrdiff_t out_rs = 0, out_cs = 0;
    int n_threads = 0;  // 0: OpenMP's default team
    uintptr_t cancel_flag = 0;
};

// momwire#2069: Z's rows for M observers, `Φ₀@M₀ + Φ₁@M₁ + Φ₂@M₂` with Φ the
// real-k literal-cos field tensor (less the PEC image's when `img_c` is set),
// computed a row at a time and contracted in `reduce_phi_band`'s order, so
// the two-pass fill's bits with no Φ band. The caller holds no GIL here.
void sinusoidal_fill_rows(const FillRowsArgs &A);

}  // namespace sinusoidal
}  // namespace mw
