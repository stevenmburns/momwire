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

}  // namespace sinusoidal
}  // namespace mw
