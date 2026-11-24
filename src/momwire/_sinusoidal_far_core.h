// NEC-2's interaction approximation on the point-matched sinusoidal fill
// (the opt-in `nec2_far_approx`): the field tensors of
// `_sinusoidal_core.h` with every (observer, source) pair whose segment
// centres are more than `far_r` metres apart taken from the source segment's
// lumped current, an infinitesimal dipole at its centre. See
// `_sinusoidal_far_core.cpp` for the form and `momwire._nec2_far_approx` for
// the evidence.
//
// Its own TU, as the NEC-4 kernel's is, so that no shipped kernel's codegen
// can move with it.
#pragma once

#include "_mw_nd.h"
#include "_sinusoidal_core.h"

#include <complex>
#include <cstdint>
#include <tuple>

namespace mw {
namespace sinusoidal {

// The reduced kernel (`sinusoidal_field_tensor`) with the far pairs lumped.
std::tuple<nd::array_t<std::complex<double>>, nd::array_t<std::complex<double>>, nd::array_t<std::complex<double>>>
sinusoidal_field_tensor_far(
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
    double far_r,
    uintptr_t cancel_flag);

// The reflection-coefficient image (`sinusoidal_field_tensor_refl`) with the
// far pairs lumped: the lumped field takes the same Fresnel dyad.
std::tuple<nd::array_t<std::complex<double>>, nd::array_t<std::complex<double>>, nd::array_t<std::complex<double>>>
sinusoidal_field_tensor_refl_far(
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
    double far_r,
    uintptr_t cancel_flag);

// The far pairs' lumped tensors written over `base` (three (M, N) tensors a
// shipped kernel filled for the same pairs, the extended kernel's), the near
// pairs copied from it. The kernels with no lumped twin of their own (EK,
// EK + Fresnel) take the approximation's answer this way, at their own cost.
// `refl` selects the Fresnel tail, whose tables are then required; pass
// empty (0, 0) tables otherwise. `cos_minus_one`: the base's third shape is
// the folded cos(k xi) - 1 (`cos_shape="cos-1"`), and so is the lumped one.
std::tuple<nd::array_t<std::complex<double>>, nd::array_t<std::complex<double>>, nd::array_t<std::complex<double>>>
sinusoidal_far_overlay(
    nd::array_t<std::complex<double>, nd::array::c_style | nd::array::forcecast> base_const,
    nd::array_t<std::complex<double>, nd::array::c_style | nd::array::forcecast> base_sin,
    nd::array_t<std::complex<double>, nd::array::c_style | nd::array::forcecast> base_cos,
    nd::array_t<double, nd::array::c_style | nd::array::forcecast> obs_centers,
    nd::array_t<double, nd::array::c_style | nd::array::forcecast> obs_tangents,
    nd::array_t<double, nd::array::c_style | nd::array::forcecast> src_centers,
    nd::array_t<double, nd::array::c_style | nd::array::forcecast> src_tangents,
    nd::array_t<double, nd::array::c_style | nd::array::forcecast> seg_h,
    double a,
    double k,
    double eta,
    bool refl,
    bool cos_minus_one,
    nd::array_t<double, nd::array::c_style | nd::array::forcecast> cos_th,
    nd::array_t<double, nd::array::c_style | nd::array::forcecast> px,
    nd::array_t<double, nd::array::c_style | nd::array::forcecast> py_,
    nd::array_t<double, nd::array::c_style | nd::array::forcecast> tm_p,
    nd::array_t<double, nd::array::c_style | nd::array::forcecast> tn_p,
    std::complex<double> eps_t,
    double far_r,
    uintptr_t cancel_flag);

// `sinusoidal_fill_rows` (`_sinusoidal_core.h`, momwire#2069) with the far
// pairs lumped at `far_r` metres (momwire#2082 step 1): Z's rows with the
// contraction fused in, the bits of `sinusoidal_field_tensor_far` (direct,
// less the PEC image's when `img_c` is set) reduced by `reduce_phi_band`. The
// reduced kernel only (`A.ek` refuses). The caller holds no GIL here.
void sinusoidal_fill_rows_far(const FillRowsArgs &A, double far_r);

// How many (observer, source) pairs of the given arrays have a centre
// distance within `rel` of `far_r` (|d - far_r| <= rel * far_r): the pairs
// whose side of the range depends on rounding (`_nec2_far_approx.ties`).
int64_t far_range_ties(
    nd::array_t<double, nd::array::c_style | nd::array::forcecast> obs_centers,
    nd::array_t<double, nd::array::c_style | nd::array::forcecast> src_centers,
    double far_r,
    double rel);

}  // namespace sinusoidal
}  // namespace mw
