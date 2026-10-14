// The razor kernels' Python-free interface: `_razor_core.cpp` and
// `_razor_t2_core.cpp` define it, `_accel_razor.cpp` and `_accel_razor_t2.cpp`
// bind it.
//
// Every array is a non-owning `mw::view` over memory the CALLER owns. Inputs
// are read through their strides, except the 1-D index, flag and weight
// arrays the kernels read as runs, which must be C-contiguous (checked).
// Outputs are written into caller-allocated arrays, C-contiguous where the
// kernel says so. Nothing here allocates a result, takes the GIL, or knows
// about numpy. An empty view (size 0) is how an optional array says
// "absent", as the bindings' empty arrays always did.
//
// Each function validates its arguments before it reads them, in the
// pre-split order and wording.
//
// Errors are C++ exceptions: std::runtime_error / std::invalid_argument for a
// bad call, AbortedError when `cancel_flag` (the raw address of an int32,
// 0 = none) trips mid-fill. The extension's binding translates them; a C++
// caller catches them.
#pragma once

#include "_mw_core.h"

namespace mw {
namespace razor {

// M0[p, s] = int_0^h g dtau, M1[p, s] = int_0^h tau g dtau (real k).
// m0_out, m1_out: C-contiguous (n_obs, n_seg); m1_out may be (0, 0) when
// !need_m1.
void seg_moments(
    view<const double, 2> obs, view<const double, 2> seg_p0,
    view<const double, 2> seg_t, view<const double, 1> seg_h,
    view<const double, 1> a, view<const double, 1> xg, view<const double, 1> wg,
    double k, bool need_m1, view<const int64_t, 1> group_i,
    view<const int64_t, 1> group_j, view<const double, 1> a_ek,
    uintptr_t cancel_flag, view<const double, 1> b_obs,
    view<cd, 2> m0_out, view<cd, 2> m1_out);

// The in-medium twin (momwire#796): Im k <= 0, else std::invalid_argument
// from require_decaying_k, checked before anything else.
void seg_moments_cplx(
    view<const double, 2> obs, view<const double, 2> seg_p0,
    view<const double, 2> seg_t, view<const double, 1> seg_h,
    view<const double, 1> a, view<const double, 1> xg, view<const double, 1> wg,
    cd k, bool need_m1, view<const int64_t, 1> group_i,
    view<const int64_t, 1> group_j, view<const double, 1> a_ek,
    uintptr_t cancel_flag, view<const double, 1> b_obs,
    view<cd, 2> m0_out, view<cd, 2> m1_out);

void require_decaying_k(cd k);

// e^{-jkR} - 1 at n radii (Design D4), into out[n].
void cplx_brackets(const double *R, size_t n, cd k, cd *out);

// T1 of one row window: out is (n_obs / n_path, n_basis), any strides.
void assemble_t1(
    view<const cd, 2> M0, view<const cd, 2> M1,
    view<const int64_t, 1> s_a, view<const int64_t, 1> s_b,
    view<const double, 1> h_a, view<const double, 1> h_b,
    view<const bool, 1> fall_a, view<const bool, 1> fall_b,
    view<const double, 2> t_out, view<const double, 2> td_a,
    view<const double, 2> td_b, view<const double, 1> wts, size_t n_path,
    view<const int64_t, 1> obs_row, view<cd, 2> out);

void assemble_t1_weighted(
    view<const cd, 2> M0, view<const cd, 2> M1,
    view<const int64_t, 1> s_a, view<const int64_t, 1> s_b,
    view<const double, 1> h_a, view<const double, 1> h_b,
    view<const bool, 1> fall_a, view<const bool, 1> fall_b,
    view<const double, 1> sig_a, view<const double, 1> sig_b,
    view<const double, 2> obs_c, view<const double, 2> obs_t,
    view<const double, 2> src_c, view<const double, 2> src_t,
    view<const double, 1> wts, size_t n_path, int window_kind,
    cd eps_t, double ground_z, cd coefficient,
    view<const int64_t, 1> obs_row, view<cd, 2> out);

// T2 of one row window and (t1 given) the block's final combination,
// written into `out` through its strides (a window of a column-major Z).
// `t1` may alias `out`. n_tab = M0w rows + M0k rows.
void t2_rows(view<cd, 2> out, const view<const cd, 2> *t1,
             view<const cd, 2> M0w, view<const cd, 2> M0k,
             view<const int64_t, 1> plus, view<const int64_t, 1> minus,
             view<const int64_t, 1> s_a, view<const int64_t, 1> s_b,
             view<const double, 1> q_a, view<const double, 1> q_b,
             double c_a_imag, double c_phi_imag);

// The composing ground's remainder rows Q: f is C-contiguous (n_u, n_seg, 2);
// out is C-contiguous (n_obs / n_path, n_basis).
void q_rows(view<const cd, 3> f, view<const int64_t, 1> obs_row,
            view<const int64_t, 1> s_a, view<const int64_t, 1> s_b,
            view<const double, 1> h_a, view<const double, 1> h_b,
            view<const bool, 1> fall_a, view<const bool, 1> fall_b,
            view<const double, 1> sig_a, view<const double, 1> sig_b,
            view<const double, 1> wts, ptrdiff_t n_path, view<cd, 2> out);

}  // namespace razor
}  // namespace mw
