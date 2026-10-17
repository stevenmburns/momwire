// The B-spline basis' accelerator kernels, Python-free (momwire#1517 step 0):
// every function `_accel_bspline.cpp` binds, by its bound name and with its
// bound arguments, over `mw::nd` arrays (`_mw_nd.h`) instead of numpy's. The
// arithmetic is `_bspline_core.cpp`'s, which is the text the extension TU
// compiled before the split; the extension now calls these.
//
// Arrays: an argument is a `mw::nd::array_t` over the caller's memory
// (`array_t::borrow`), any strides where the bound function took any (a
// write-through Z), C-contiguous where it took `c_style`; a result is
// allocated by the current host (`mw::nd::current_host()`: the default host
// allocates with `new`; the extension installs numpy's). A `mw::nd::object`
// argument is None (`mw::nd::none()`) or an array the host casts.
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
namespace bspline {

// `BSPLINE_FAR_MAX_P` and `BSPLINE_MOMENT_MAX_D`, which the extension
// exports (momwire#999).
extern const int FAR_MAX_P;
extern const int MOMENT_MAX_D;
// Whether this build's windowed assemblers and off-edge moments carry the
// AVX2 lane kernels (momwire#1290), which the extension reports.
extern const bool WINDOWED_LANES_1290;
extern const bool OFFEDGE_LANES_1290;


double bspline_j_static_moment_py(
    int p,
    int q,
    double alpha,
    double beta,
    double A,
    double B,
    double a);

double bspline_d_ek_moment_py(
    int p,
    int q,
    double alpha,
    double beta,
    double A,
    double B,
    double a);

double bspline_j_static_far_py(
    int p,
    int q,
    double alpha,
    double beta,
    double A,
    double B,
    double a);

nd::array_t<std::complex<double>> seg_seg_reg_moments_bspline_swept(
    nd::array_t<double, nd::array::c_style | nd::array::forcecast> R,
    nd::array_t<double, nd::array::c_style | nd::array::forcecast> wu_pow,
    nd::array_t<double, nd::array::c_style | nd::array::forcecast> k_array);

nd::array_t<std::complex<double>> seg_seg_reg_moments_bspline_swept_ek(
    nd::array_t<double, nd::array::c_style | nd::array::forcecast> R,
    nd::array_t<double, nd::array::c_style | nd::array::forcecast> wu_pow,
    nd::array_t<double, nd::array::c_style | nd::array::forcecast> k_array,
    double a_ek);

nd::array_t<std::complex<double>> seg_seg_reg_moments_bspline_swept_window(
    nd::array_t<double, nd::array::c_style | nd::array::forcecast> R,
    nd::array_t<double, nd::array::c_style | nd::array::forcecast> wu_row,
    nd::array_t<double, nd::array::c_style | nd::array::forcecast> wu_col,
    nd::array_t<double, nd::array::c_style | nd::array::forcecast> k_array);

nd::array_t<std::complex<double>> seg_seg_reg_moments_bspline_swept_ek_window(
    nd::array_t<double, nd::array::c_style | nd::array::forcecast> R,
    nd::array_t<double, nd::array::c_style | nd::array::forcecast> wu_row,
    nd::array_t<double, nd::array::c_style | nd::array::forcecast> wu_col,
    nd::array_t<double, nd::array::c_style | nd::array::forcecast> k_array,
    double a_ek);

nd::array_t<std::complex<double>> seg_seg_full_moments_bspline(
    nd::array_t<double, nd::array::c_style | nd::array::forcecast> seg_l_i,
    nd::array_t<double, nd::array::c_style | nd::array::forcecast> seg_r_i,
    nd::array_t<double, nd::array::c_style | nd::array::forcecast> seg_l_j,
    nd::array_t<double, nd::array::c_style | nd::array::forcecast> seg_r_j,
    double a_squared,
    double k,
    int max_d,
    nd::array_t<double, nd::array::c_style | nd::array::forcecast> gl_t,
    nd::array_t<double, nd::array::c_style | nd::array::forcecast> gl_w,
    bool reference);

nd::array_t<std::complex<double>> seg_seg_full_moments_bspline_cplx(
    nd::array_t<double, nd::array::c_style | nd::array::forcecast> seg_l_i,
    nd::array_t<double, nd::array::c_style | nd::array::forcecast> seg_r_i,
    nd::array_t<double, nd::array::c_style | nd::array::forcecast> seg_l_j,
    nd::array_t<double, nd::array::c_style | nd::array::forcecast> seg_r_j,
    double a_squared,
    std::complex<double> k,
    int max_d,
    nd::array_t<double, nd::array::c_style | nd::array::forcecast> gl_t,
    nd::array_t<double, nd::array::c_style | nd::array::forcecast> gl_w,
    bool reference);

nd::array_t<std::complex<double>> seg_seg_full_moments_bspline_tiered(
    nd::array_t<double, nd::array::c_style | nd::array::forcecast> seg_l_i,
    nd::array_t<double, nd::array::c_style | nd::array::forcecast> seg_r_i,
    nd::array_t<double, nd::array::c_style | nd::array::forcecast> seg_l_j,
    nd::array_t<double, nd::array::c_style | nd::array::forcecast> seg_r_j,
    double a_squared,
    double k,
    int max_d,
    nd::array_t<double, nd::array::c_style | nd::array::forcecast> tier_t,
    nd::array_t<double, nd::array::c_style | nd::array::forcecast> tier_w,
    nd::array_t<int64_t, nd::array::c_style | nd::array::forcecast> tier_n_qp,
    nd::array_t<double, nd::array::c_style | nd::array::forcecast> tier_ratio,
    bool reference);

nd::array_t<std::complex<double>> seg_seg_full_moments_bspline_cplx_tiered(
    nd::array_t<double, nd::array::c_style | nd::array::forcecast> seg_l_i,
    nd::array_t<double, nd::array::c_style | nd::array::forcecast> seg_r_i,
    nd::array_t<double, nd::array::c_style | nd::array::forcecast> seg_l_j,
    nd::array_t<double, nd::array::c_style | nd::array::forcecast> seg_r_j,
    double a_squared,
    std::complex<double> k,
    int max_d,
    nd::array_t<double, nd::array::c_style | nd::array::forcecast> tier_t,
    nd::array_t<double, nd::array::c_style | nd::array::forcecast> tier_w,
    nd::array_t<int64_t, nd::array::c_style | nd::array::forcecast> tier_n_qp,
    nd::array_t<double, nd::array::c_style | nd::array::forcecast> tier_ratio,
    bool reference);

nd::array_t<std::complex<double>> seg_seg_full_moments_sinusoidal_tiered(
    nd::array_t<double, nd::array::c_style | nd::array::forcecast> seg_l_i,
    nd::array_t<double, nd::array::c_style | nd::array::forcecast> seg_r_i,
    nd::array_t<double, nd::array::c_style | nd::array::forcecast> seg_l_j,
    nd::array_t<double, nd::array::c_style | nd::array::forcecast> seg_r_j,
    double a_squared,
    std::complex<double> k,
    nd::array_t<double, nd::array::c_style | nd::array::forcecast> tier_t,
    nd::array_t<double, nd::array::c_style | nd::array::forcecast> tier_w,
    nd::array_t<int64_t, nd::array::c_style | nd::array::forcecast> tier_n_qp,
    nd::array_t<double, nd::array::c_style | nd::array::forcecast> tier_ratio,
    bool reference);

nd::array_t<std::complex<double>> seg_seg_full_moments_sinusoidal_tiered_ek(
    nd::array_t<double, nd::array::c_style | nd::array::forcecast> seg_l_i,
    nd::array_t<double, nd::array::c_style | nd::array::forcecast> seg_r_i,
    nd::array_t<double, nd::array::c_style | nd::array::forcecast> seg_l_j,
    nd::array_t<double, nd::array::c_style | nd::array::forcecast> seg_r_j,
    double a_squared,
    double k,
    nd::array_t<double, nd::array::c_style | nd::array::forcecast> tier_t,
    nd::array_t<double, nd::array::c_style | nd::array::forcecast> tier_w,
    nd::array_t<int64_t, nd::array::c_style | nd::array::forcecast> tier_n_qp,
    nd::array_t<double, nd::array::c_style | nd::array::forcecast> tier_ratio,
    nd::array_t<int64_t, nd::array::c_style | nd::array::forcecast> group_i,
    nd::array_t<int64_t, nd::array::c_style | nd::array::forcecast> group_j,
    double a_ek,
    nd::object b_j_obj);

void assemble_Z_sinusoidal_windowed(
    nd::array_t<std::complex<double>, nd::array::c_style | nd::array::forcecast> J,
    nd::array_t<int64_t, nd::array::c_style | nd::array::forcecast> rows_seg,
    nd::array_t<int64_t, nd::array::c_style | nd::array::forcecast> cols_seg,
    nd::array_t<int64_t, nd::array::c_style | nd::array::forcecast> starts,
    nd::array_t<int64_t, nd::array::c_style | nd::array::forcecast> jbasis,
    nd::array_t<std::complex<double>, nd::array::c_style | nd::array::forcecast> coef,
    nd::array_t<std::complex<double>, nd::array::c_style | nd::array::forcecast> dcoef,
    nd::array_t<double, nd::array::c_style | nd::array::forcecast> tangents_rows,
    nd::array_t<double, nd::array::c_style | nd::array::forcecast> tangents_cols,
    std::complex<double> c_a,
    std::complex<double> c_phi,
    nd::object w_a_obj,
    nd::object w_phi_obj,
    nd::array_t<std::complex<double>> Z,
    uintptr_t cancel_flag);

nd::array_t<std::complex<double>> field_pair_moments_sinusoidal(
    nd::array_t<std::complex<double>, nd::array::c_style | nd::array::forcecast> F,
    nd::array_t<std::complex<double>, nd::array::c_style | nd::array::forcecast> W_obs,
    nd::array_t<std::complex<double>, nd::array::c_style | nd::array::forcecast> W_src);

nd::array_t<std::complex<double>> parallel_pair_moments_sinusoidal(
    nd::array_t<double, nd::array::c_style | nd::array::forcecast> c_i,
    nd::array_t<double, nd::array::c_style | nd::array::forcecast> t_i,
    nd::array_t<double, nd::array::c_style | nd::array::forcecast> h_i,
    nd::array_t<double, nd::array::c_style | nd::array::forcecast> c_j,
    nd::array_t<double, nd::array::c_style | nd::array::forcecast> t_j,
    nd::array_t<double, nd::array::c_style | nd::array::forcecast> h_j,
    nd::array_t<double, nd::array::c_style | nd::array::forcecast> a2,
    std::complex<double> k,
    nd::array_t<double, nd::array::c_style | nd::array::forcecast> gt,
    nd::array_t<double, nd::array::c_style | nd::array::forcecast> gw,
    nd::array_t<double, nd::array::c_style | nd::array::forcecast> gx,
    nd::array_t<double, nd::array::c_style | nd::array::forcecast> gwx,
    nd::object a_ek_obj,
    nd::object b_ek_obj);

nd::array_t<std::complex<double>> seg_seg_full_moments_bspline_swept(
    nd::array_t<double, nd::array::c_style | nd::array::forcecast> seg_l_i,
    nd::array_t<double, nd::array::c_style | nd::array::forcecast> seg_r_i,
    nd::array_t<double, nd::array::c_style | nd::array::forcecast> seg_l_j,
    nd::array_t<double, nd::array::c_style | nd::array::forcecast> seg_r_j,
    double a_squared,
    nd::array_t<double, nd::array::c_style | nd::array::forcecast> k_array,
    int max_d,
    nd::array_t<double, nd::array::c_style | nd::array::forcecast> gl_t,
    nd::array_t<double, nd::array::c_style | nd::array::forcecast> gl_w);

nd::array_t<std::complex<double>> seg_seg_full_moments_bspline_ek(
    nd::array_t<double, nd::array::c_style | nd::array::forcecast> seg_l_i,
    nd::array_t<double, nd::array::c_style | nd::array::forcecast> seg_r_i,
    nd::array_t<double, nd::array::c_style | nd::array::forcecast> seg_l_j,
    nd::array_t<double, nd::array::c_style | nd::array::forcecast> seg_r_j,
    double a_squared,
    double k,
    int max_d,
    nd::array_t<double, nd::array::c_style | nd::array::forcecast> gl_t,
    nd::array_t<double, nd::array::c_style | nd::array::forcecast> gl_w,
    nd::array_t<int64_t, nd::array::c_style | nd::array::forcecast> group_i,
    nd::array_t<int64_t, nd::array::c_style | nd::array::forcecast> group_j,
    double a_ek,
    bool reference,
    double a_ek_src);

nd::array_t<std::complex<double>> seg_seg_full_moments_bspline_ek_tiered(
    nd::array_t<double, nd::array::c_style | nd::array::forcecast> seg_l_i,
    nd::array_t<double, nd::array::c_style | nd::array::forcecast> seg_r_i,
    nd::array_t<double, nd::array::c_style | nd::array::forcecast> seg_l_j,
    nd::array_t<double, nd::array::c_style | nd::array::forcecast> seg_r_j,
    double a_squared,
    double k,
    int max_d,
    nd::array_t<double, nd::array::c_style | nd::array::forcecast> tier_t,
    nd::array_t<double, nd::array::c_style | nd::array::forcecast> tier_w,
    nd::array_t<int64_t, nd::array::c_style | nd::array::forcecast> tier_n_qp,
    nd::array_t<double, nd::array::c_style | nd::array::forcecast> tier_ratio,
    nd::array_t<int64_t, nd::array::c_style | nd::array::forcecast> group_i,
    nd::array_t<int64_t, nd::array::c_style | nd::array::forcecast> group_j,
    double a_ek,
    bool reference,
    double a_ek_src);

nd::array_t<std::complex<double>> seg_seg_full_moments_bspline_swept_ek(
    nd::array_t<double, nd::array::c_style | nd::array::forcecast> seg_l_i,
    nd::array_t<double, nd::array::c_style | nd::array::forcecast> seg_r_i,
    nd::array_t<double, nd::array::c_style | nd::array::forcecast> seg_l_j,
    nd::array_t<double, nd::array::c_style | nd::array::forcecast> seg_r_j,
    double a_squared,
    nd::array_t<double, nd::array::c_style | nd::array::forcecast> k_array,
    int max_d,
    nd::array_t<double, nd::array::c_style | nd::array::forcecast> gl_t,
    nd::array_t<double, nd::array::c_style | nd::array::forcecast> gl_w,
    nd::array_t<int64_t, nd::array::c_style | nd::array::forcecast> group_i,
    nd::array_t<int64_t, nd::array::c_style | nd::array::forcecast> group_j,
    double a_ek,
    bool reference,
    double a_ek_src);

nd::array_t<std::complex<double>> seg_seg_full_moments_bspline_swept_ek_tiered(
    nd::array_t<double, nd::array::c_style | nd::array::forcecast> seg_l_i,
    nd::array_t<double, nd::array::c_style | nd::array::forcecast> seg_r_i,
    nd::array_t<double, nd::array::c_style | nd::array::forcecast> seg_l_j,
    nd::array_t<double, nd::array::c_style | nd::array::forcecast> seg_r_j,
    double a_squared,
    nd::array_t<double, nd::array::c_style | nd::array::forcecast> k_array,
    int max_d,
    nd::array_t<double, nd::array::c_style | nd::array::forcecast> tier_t,
    nd::array_t<double, nd::array::c_style | nd::array::forcecast> tier_w,
    nd::array_t<int64_t, nd::array::c_style | nd::array::forcecast> tier_n_qp,
    nd::array_t<double, nd::array::c_style | nd::array::forcecast> tier_ratio,
    nd::array_t<int64_t, nd::array::c_style | nd::array::forcecast> group_i,
    nd::array_t<int64_t, nd::array::c_style | nd::array::forcecast> group_j,
    double a_ek,
    bool reference,
    double a_ek_src);

nd::array_t<double> seg_seg_static_moments_bspline_uniform(
    double h,
    double a,
    size_t N,
    int max_d,
    uintptr_t cancel_flag,
    uintptr_t poll_probe);

nd::array_t<double> seg_seg_static_moments_bspline_uniform_ek(
    double h,
    double a,
    size_t N,
    int max_d,
    double a_ek,
    uintptr_t cancel_flag);

nd::array_t<double> toeplitz_gather_rows(
    nd::array_t<double, nd::array::c_style | nd::array::forcecast> table,
    size_t N,
    size_t r0,
    size_t r1);

nd::array_t<double> reg_distance_table(
    nd::array_t<double, nd::array::c_style | nd::array::forcecast> s_row,
    nd::array_t<double, nd::array::c_style | nd::array::forcecast> s_col,
    double a2);

nd::array_t<std::complex<double>> same_edge_corr_combine(
    nd::array_t<double, nd::array::c_style | nd::array::forcecast> A_st,
    nd::array_t<std::complex<double>, nd::array::c_style | nd::array::forcecast> A_reg,
    nd::array_t<std::complex<double>, nd::array::c_style | nd::array::forcecast> J);

nd::array_t<double> seg_seg_static_moments_bspline_uniform_table(
    double h,
    double a,
    size_t N,
    int max_d);

nd::array_t<double> seg_seg_static_moments_bspline_uniform_ek_table(
    double h,
    double a,
    size_t N,
    int max_d,
    double a_ek);

void assemble_Z_bspline_weighted_windowed(
    nd::array_t<std::complex<double>, nd::array::c_style | nd::array::forcecast> J_chunk,
    nd::array_t<int64_t, nd::array::c_style | nd::array::forcecast> support_seg,
    nd::array_t<double, nd::array::c_style | nd::array::forcecast> polys,
    nd::array_t<std::complex<double>, nd::array::c_style | nd::array::forcecast> wA_win,
    nd::array_t<std::complex<double>, nd::array::c_style | nd::array::forcecast> wPhi_win,
    nd::array_t<int64_t, nd::array::c_style | nd::array::forcecast> m_idx,
    nd::array_t<int64_t, nd::array::c_style | nd::array::forcecast> n_idx,
    int64_t i0,
    int64_t i1,
    int64_t j0,
    int64_t j1,
    double omega,
    double eps_,
    double mu_,
    std::complex<double> scale,
    nd::array_t<std::complex<double>> Z,
    uintptr_t cancel_flag,
    nd::object row_of,
    bool reference);

void assemble_Z_bspline_windowed_cplx_eps(
    nd::array_t<std::complex<double>, nd::array::c_style | nd::array::forcecast> J_chunk,
    nd::array_t<int64_t, nd::array::c_style | nd::array::forcecast> support_seg,
    nd::array_t<double, nd::array::c_style | nd::array::forcecast> polys,
    nd::array_t<double, nd::array::c_style | nd::array::forcecast> tangents,
    nd::array_t<int64_t, nd::array::c_style | nd::array::forcecast> m_idx,
    nd::array_t<int64_t, nd::array::c_style | nd::array::forcecast> n_idx,
    int64_t i0,
    int64_t i1,
    int64_t j0,
    int64_t j1,
    double omega,
    std::complex<double> eps_,
    double mu_,
    nd::array_t<std::complex<double>> Z,
    uintptr_t cancel_flag,
    nd::object row_of,
    bool reference);

void assemble_Z_bspline_weighted_windowed_cplx_eps(
    nd::array_t<std::complex<double>, nd::array::c_style | nd::array::forcecast> J_chunk,
    nd::array_t<int64_t, nd::array::c_style | nd::array::forcecast> support_seg,
    nd::array_t<double, nd::array::c_style | nd::array::forcecast> polys,
    nd::array_t<std::complex<double>, nd::array::c_style | nd::array::forcecast> wA_win,
    nd::array_t<std::complex<double>, nd::array::c_style | nd::array::forcecast> wPhi_win,
    nd::array_t<int64_t, nd::array::c_style | nd::array::forcecast> m_idx,
    nd::array_t<int64_t, nd::array::c_style | nd::array::forcecast> n_idx,
    int64_t i0,
    int64_t i1,
    int64_t j0,
    int64_t j1,
    double omega,
    std::complex<double> eps_,
    double mu_,
    std::complex<double> scale,
    nd::array_t<std::complex<double>> Z,
    uintptr_t cancel_flag,
    nd::object row_of,
    bool reference);

void assemble_Z_bspline_windowed(
    nd::array_t<std::complex<double>, nd::array::c_style | nd::array::forcecast> J_chunk,
    nd::array_t<int64_t, nd::array::c_style | nd::array::forcecast> support_seg,
    nd::array_t<double, nd::array::c_style | nd::array::forcecast> polys,
    nd::array_t<double, nd::array::c_style | nd::array::forcecast> tangents,
    nd::array_t<int64_t, nd::array::c_style | nd::array::forcecast> m_idx,
    nd::array_t<int64_t, nd::array::c_style | nd::array::forcecast> n_idx,
    int64_t i0,
    int64_t i1,
    int64_t j0,
    int64_t j1,
    double omega,
    double eps_,
    double mu_,
    nd::array_t<std::complex<double>> Z,
    uintptr_t cancel_flag,
    nd::object row_of,
    bool reference);

nd::array_t<std::complex<double>> assemble_Z_bspline(
    nd::array_t<std::complex<double>, nd::array::c_style | nd::array::forcecast> J,
    nd::array_t<int64_t, nd::array::c_style | nd::array::forcecast> support_seg,
    nd::array_t<double, nd::array::c_style | nd::array::forcecast> polys,
    nd::array_t<double, nd::array::c_style | nd::array::forcecast> td_all,
    double omega,
    double eps_,
    double mu_,
    int max_d,
    uintptr_t cancel_flag);

nd::array_t<std::complex<double>> assemble_Z_bspline_weighted(
    nd::array_t<std::complex<double>, nd::array::c_style | nd::array::forcecast> J,
    nd::array_t<int64_t, nd::array::c_style | nd::array::forcecast> support_seg,
    nd::array_t<double, nd::array::c_style | nd::array::forcecast> polys,
    nd::array_t<std::complex<double>, nd::array::c_style | nd::array::forcecast> wA_all,
    nd::array_t<std::complex<double>, nd::array::c_style | nd::array::forcecast> wPhi_all,
    double omega,
    double eps_,
    double mu_,
    int max_d,
    uintptr_t cancel_flag);

nd::array_t<std::complex<double>> assemble_Z_bspline_cplx_eps(
    nd::array_t<std::complex<double>, nd::array::c_style | nd::array::forcecast> J,
    nd::array_t<int64_t, nd::array::c_style | nd::array::forcecast> support_seg,
    nd::array_t<double, nd::array::c_style | nd::array::forcecast> polys,
    nd::array_t<double, nd::array::c_style | nd::array::forcecast> td_all,
    double omega,
    std::complex<double> eps_,
    double mu_,
    int max_d,
    uintptr_t cancel_flag);

nd::array_t<std::complex<double>> assemble_Z_bspline_weighted_cplx_eps(
    nd::array_t<std::complex<double>, nd::array::c_style | nd::array::forcecast> J,
    nd::array_t<int64_t, nd::array::c_style | nd::array::forcecast> support_seg,
    nd::array_t<double, nd::array::c_style | nd::array::forcecast> polys,
    nd::array_t<std::complex<double>, nd::array::c_style | nd::array::forcecast> wA_all,
    nd::array_t<std::complex<double>, nd::array::c_style | nd::array::forcecast> wPhi_all,
    double omega,
    std::complex<double> eps_,
    double mu_,
    int max_d,
    uintptr_t cancel_flag);

nd::array_t<std::complex<double>> assemble_Z_bspline_swept(
    nd::array_t<std::complex<double>, nd::array::c_style | nd::array::forcecast> J,
    nd::array_t<int64_t, nd::array::c_style | nd::array::forcecast> support_seg,
    nd::array_t<double, nd::array::c_style | nd::array::forcecast> polys,
    nd::array_t<double, nd::array::c_style | nd::array::forcecast> tangents_row,
    nd::array_t<double, nd::array::c_style | nd::array::forcecast> tangents_col,
    nd::array_t<double, nd::array::c_style | nd::array::forcecast> omega_array,
    double eps_,
    double mu_,
    int max_d);

nd::array_t<std::complex<double>> bspline_assemble_offedge_block(
    nd::array_t<int64_t, nd::array::c_style | nd::array::forcecast> supp_I,
    nd::array_t<double, nd::array::c_style | nd::array::forcecast> polys_I,
    nd::array_t<double, nd::array::c_style | nd::array::forcecast> segl_I,
    nd::array_t<double, nd::array::c_style | nd::array::forcecast> segr_I,
    nd::array_t<double, nd::array::c_style | nd::array::forcecast> tan_I,
    nd::array_t<int64_t, nd::array::c_style | nd::array::forcecast> supp_J,
    nd::array_t<double, nd::array::c_style | nd::array::forcecast> polys_J,
    nd::array_t<double, nd::array::c_style | nd::array::forcecast> segl_J,
    nd::array_t<double, nd::array::c_style | nd::array::forcecast> segr_J,
    nd::array_t<double, nd::array::c_style | nd::array::forcecast> tan_J,
    double a_squared,
    double k,
    double omega,
    double eps_,
    double mu_,
    int max_d,
    nd::array_t<double, nd::array::c_style | nd::array::forcecast> gl_t,
    nd::array_t<double, nd::array::c_style | nd::array::forcecast> gl_w,
    uintptr_t cancel_flag);

nd::array_t<std::complex<double>> bspline_assemble_offedge_block_refl(
    nd::array_t<int64_t, nd::array::c_style | nd::array::forcecast> supp_I,
    nd::array_t<double, nd::array::c_style | nd::array::forcecast> polys_I,
    nd::array_t<double, nd::array::c_style | nd::array::forcecast> segl_I,
    nd::array_t<double, nd::array::c_style | nd::array::forcecast> segr_I,
    nd::array_t<double, nd::array::c_style | nd::array::forcecast> tan_I,
    nd::array_t<int64_t, nd::array::c_style | nd::array::forcecast> supp_J,
    nd::array_t<double, nd::array::c_style | nd::array::forcecast> polys_J,
    nd::array_t<double, nd::array::c_style | nd::array::forcecast> segl_J,
    nd::array_t<double, nd::array::c_style | nd::array::forcecast> segr_J,
    nd::array_t<double, nd::array::c_style | nd::array::forcecast> tan_J,
    double a_squared,
    double k,
    double omega,
    double eps_,
    double mu_,
    int max_d,
    nd::array_t<double, nd::array::c_style | nd::array::forcecast> gl_t,
    nd::array_t<double, nd::array::c_style | nd::array::forcecast> gl_w,
    std::complex<double> eps_t,
    std::complex<double> phi_c0,
    std::complex<double> phi_c1,
    uintptr_t cancel_flag);

nd::array_t<std::complex<double>> bspline_assemble_offedge_block_ek(
    nd::array_t<int64_t, nd::array::c_style | nd::array::forcecast> supp_I,
    nd::array_t<double, nd::array::c_style | nd::array::forcecast> polys_I,
    nd::array_t<double, nd::array::c_style | nd::array::forcecast> segl_I,
    nd::array_t<double, nd::array::c_style | nd::array::forcecast> segr_I,
    nd::array_t<double, nd::array::c_style | nd::array::forcecast> tan_I,
    nd::array_t<int64_t, nd::array::c_style | nd::array::forcecast> supp_J,
    nd::array_t<double, nd::array::c_style | nd::array::forcecast> polys_J,
    nd::array_t<double, nd::array::c_style | nd::array::forcecast> segl_J,
    nd::array_t<double, nd::array::c_style | nd::array::forcecast> segr_J,
    nd::array_t<double, nd::array::c_style | nd::array::forcecast> tan_J,
    double a_squared,
    double k,
    double omega,
    double eps_,
    double mu_,
    int max_d,
    nd::array_t<double, nd::array::c_style | nd::array::forcecast> gl_t,
    nd::array_t<double, nd::array::c_style | nd::array::forcecast> gl_w,
    nd::array_t<int64_t, nd::array::c_style | nd::array::forcecast> group_I,
    nd::array_t<int64_t, nd::array::c_style | nd::array::forcecast> group_J,
    double a_ek,
    uintptr_t cancel_flag,
    bool reference);

nd::array_t<std::complex<double>> bspline_assemble_offedge_block_ek_tiered(
    nd::array_t<int64_t, nd::array::c_style | nd::array::forcecast> supp_I,
    nd::array_t<double, nd::array::c_style | nd::array::forcecast> polys_I,
    nd::array_t<double, nd::array::c_style | nd::array::forcecast> segl_I,
    nd::array_t<double, nd::array::c_style | nd::array::forcecast> segr_I,
    nd::array_t<double, nd::array::c_style | nd::array::forcecast> tan_I,
    nd::array_t<int64_t, nd::array::c_style | nd::array::forcecast> supp_J,
    nd::array_t<double, nd::array::c_style | nd::array::forcecast> polys_J,
    nd::array_t<double, nd::array::c_style | nd::array::forcecast> segl_J,
    nd::array_t<double, nd::array::c_style | nd::array::forcecast> segr_J,
    nd::array_t<double, nd::array::c_style | nd::array::forcecast> tan_J,
    double a_squared,
    double k,
    double omega,
    double eps_,
    double mu_,
    int max_d,
    nd::array_t<double, nd::array::c_style | nd::array::forcecast> tier_t,
    nd::array_t<double, nd::array::c_style | nd::array::forcecast> tier_w,
    nd::array_t<int64_t, nd::array::c_style | nd::array::forcecast> tier_n_qp,
    nd::array_t<double, nd::array::c_style | nd::array::forcecast> tier_ratio,
    nd::array_t<int64_t, nd::array::c_style | nd::array::forcecast> group_I,
    nd::array_t<int64_t, nd::array::c_style | nd::array::forcecast> group_J,
    double a_ek,
    nd::array_t<uint8_t, nd::array::c_style | nd::array::forcecast> phase_ok_I,
    nd::array_t<uint8_t, nd::array::c_style | nd::array::forcecast> phase_ok_J,
    int64_t limited_below,
    uintptr_t cancel_flag,
    bool reference);

nd::array_t<std::complex<double>> bspline_assemble_offedge_block_refl_ek(
    nd::array_t<int64_t, nd::array::c_style | nd::array::forcecast> supp_I,
    nd::array_t<double, nd::array::c_style | nd::array::forcecast> polys_I,
    nd::array_t<double, nd::array::c_style | nd::array::forcecast> segl_I,
    nd::array_t<double, nd::array::c_style | nd::array::forcecast> segr_I,
    nd::array_t<double, nd::array::c_style | nd::array::forcecast> tan_I,
    nd::array_t<int64_t, nd::array::c_style | nd::array::forcecast> supp_J,
    nd::array_t<double, nd::array::c_style | nd::array::forcecast> polys_J,
    nd::array_t<double, nd::array::c_style | nd::array::forcecast> segl_J,
    nd::array_t<double, nd::array::c_style | nd::array::forcecast> segr_J,
    nd::array_t<double, nd::array::c_style | nd::array::forcecast> tan_J,
    double a_squared,
    double k,
    double omega,
    double eps_,
    double mu_,
    int max_d,
    nd::array_t<double, nd::array::c_style | nd::array::forcecast> gl_t,
    nd::array_t<double, nd::array::c_style | nd::array::forcecast> gl_w,
    nd::array_t<int64_t, nd::array::c_style | nd::array::forcecast> group_I,
    nd::array_t<int64_t, nd::array::c_style | nd::array::forcecast> group_J,
    double a_ek,
    std::complex<double> eps_t,
    std::complex<double> phi_c0,
    std::complex<double> phi_c1,
    uintptr_t cancel_flag,
    bool reference);

nd::array_t<std::complex<double>> bspline_assemble_offedge_block_refl_ek_tiered(
    nd::array_t<int64_t, nd::array::c_style | nd::array::forcecast> supp_I,
    nd::array_t<double, nd::array::c_style | nd::array::forcecast> polys_I,
    nd::array_t<double, nd::array::c_style | nd::array::forcecast> segl_I,
    nd::array_t<double, nd::array::c_style | nd::array::forcecast> segr_I,
    nd::array_t<double, nd::array::c_style | nd::array::forcecast> tan_I,
    nd::array_t<int64_t, nd::array::c_style | nd::array::forcecast> supp_J,
    nd::array_t<double, nd::array::c_style | nd::array::forcecast> polys_J,
    nd::array_t<double, nd::array::c_style | nd::array::forcecast> segl_J,
    nd::array_t<double, nd::array::c_style | nd::array::forcecast> segr_J,
    nd::array_t<double, nd::array::c_style | nd::array::forcecast> tan_J,
    double a_squared,
    double k,
    double omega,
    double eps_,
    double mu_,
    int max_d,
    nd::array_t<double, nd::array::c_style | nd::array::forcecast> tier_t,
    nd::array_t<double, nd::array::c_style | nd::array::forcecast> tier_w,
    nd::array_t<int64_t, nd::array::c_style | nd::array::forcecast> tier_n_qp,
    nd::array_t<double, nd::array::c_style | nd::array::forcecast> tier_ratio,
    nd::array_t<int64_t, nd::array::c_style | nd::array::forcecast> group_I,
    nd::array_t<int64_t, nd::array::c_style | nd::array::forcecast> group_J,
    double a_ek,
    std::complex<double> eps_t,
    std::complex<double> phi_c0,
    std::complex<double> phi_c1,
    nd::array_t<uint8_t, nd::array::c_style | nd::array::forcecast> phase_ok_I,
    nd::array_t<uint8_t, nd::array::c_style | nd::array::forcecast> phase_ok_J,
    int64_t limited_below,
    uintptr_t cancel_flag,
    bool reference);

std::tuple<nd::array_t<std::complex<double>>, nd::array_t<std::complex<double>>, nd::array_t<std::complex<double>>> assemble_Z_enrich(
    nd::array_t<int64_t, nd::array::c_style | nd::array::forcecast> spec_seg,
    nd::array_t<int64_t, nd::array::c_style | nd::array::forcecast> spec_origin,
    nd::array_t<double, nd::array::c_style | nd::array::forcecast> seg_l,
    nd::array_t<double, nd::array::c_style | nd::array::forcecast> seg_r,
    nd::array_t<double, nd::array::c_style | nd::array::forcecast> h_per_seg,
    nd::array_t<double, nd::array::c_style | nd::array::forcecast> tangents,
    nd::array_t<int64_t, nd::array::c_style | nd::array::forcecast> supp_seg_poly,
    nd::array_t<double, nd::array::c_style | nd::array::forcecast> polys_poly,
    double a_squared,
    double k,
    double omega,
    double eps_,
    double mu_,
    nd::array_t<double, nd::array::c_style | nd::array::forcecast> gl_t01,
    nd::array_t<double, nd::array::c_style | nd::array::forcecast> gl_w01,
    nd::array_t<double, nd::array::c_style | nd::array::forcecast> proj_coeffs);

}  // namespace bspline
}  // namespace mw
