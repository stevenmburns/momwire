// The NEC-4 dialect's radius-step junction for the point-matched sinusoidal
// basis (momwire#1644), Python-free: the NEC-4 thin-wire kernel, the end-cap
// charge's field, and the junction-charge solve. Public source: the NEC-4
// theory manual, UCRL-MA-109338 Pt. II, Sections 3.2, 3.4, 4.1, 4.4 and
// Appendices A and E; `_sinusoidal_nec4.py` and `_nec4_junction.py` carry the
// derivation and the readings licensed NEC-4.2 decided.
//
// - `sinusoidal_field_tensor_nec4` and `sinusoidal_cap_field` are KERNELS:
//   `_accel_sinusoidal_nec4.cpp` binds them, the Python solver calls them
//   whenever the extension is loaded, and the C++ fill
//   (`_sinusoidal_fill_core.cpp`) calls the same code, so the two agree by
//   construction. Their numpy twins in `_sinusoidal_nec4.py` are the
//   reference and the fallback.
// - `junction_charge_factors` is the FRONT half's: a scalar copy of
//   `_nec4_junction.junction_charge_factors`, operation for operation, so
//   the C++ front (`_sinusoidal_front_core.cpp`) gives Python's bits.
//
// Its own TU so that no existing kernel's codegen can move with it.
#pragma once

#include "_mw_nd.h"

#include <array>
#include <complex>
#include <cstdint>
#include <tuple>
#include <vector>

namespace mw {
namespace sinusoidal {

// (Phi_const, Phi_sin, Phi_cos), each (M, N): the tangential fields of the
// three source shapes under the NEC-4 kernel (source radius `src_a` per
// source, no segment-end charges, bare 1/rho radial forms).
std::tuple<nd::array_t<std::complex<double>>, nd::array_t<std::complex<double>>, nd::array_t<std::complex<double>>>
sinusoidal_field_tensor_nec4(
    nd::array_t<double, nd::array::c_style | nd::array::forcecast> obs_centers,
    nd::array_t<double, nd::array::c_style | nd::array::forcecast> obs_tangents,
    nd::array_t<double, nd::array::c_style | nd::array::forcecast> src_centers,
    nd::array_t<double, nd::array::c_style | nd::array::forcecast> src_tangents,
    nd::array_t<double, nd::array::c_style | nd::array::forcecast> seg_h,
    nd::array_t<double, nd::array::c_style | nd::array::forcecast> src_a,
    double k,
    double eta,
    nd::array_t<double, nd::array::c_style | nd::array::forcecast> gl_t,
    nd::array_t<double, nd::array::c_style | nd::array::forcecast> gl_w,
    uintptr_t cancel_flag);

// (M,): the tangential field at the observers of one charged flat end cap at
// `pos` (3,), outward normal `normal` (3,), radius `a`, with C0 = q/(4 pi eps)
// per unit eta (`_sinusoidal_nec4.end_caps`).
nd::array_t<std::complex<double>> sinusoidal_cap_field(
    nd::array_t<double, nd::array::c_style | nd::array::forcecast> obs_centers,
    nd::array_t<double, nd::array::c_style | nd::array::forcecast> obs_tangents,
    nd::array_t<double, nd::array::c_style | nd::array::forcecast> pos,
    nd::array_t<double, nd::array::c_style | nd::array::forcecast> normal,
    double a,
    double k,
    double eta,
    std::complex<double> c0);

// `_nec4_junction.junction_charge_factors(k, dirs, radii, lengths,
// first_factor)`, to the bit. Throws std::domain_error with Python's
// ValueError sentence when a member is too far off another's axis, and
// std::runtime_error for a singular system (Python's ZeroDivisionError).
std::vector<double> junction_charge_factors(double k, const std::vector<std::array<double, 3>> &dirs,
                                            const std::vector<double> &radii, const std::vector<double> &lengths,
                                            double first_factor);

}  // namespace sinusoidal
}  // namespace mw
