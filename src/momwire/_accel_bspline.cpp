// bspline section of the accelerator: the BINDING half (momwire#1517 step 0).
// The kernels, their documentation and their arithmetic live in
// `_bspline_core.cpp`, which includes no Python header. What stays here is
// only what is Python's: the names, keyword arguments, defaults and
// docstrings Python calls; numpy arrays in, with the forcecast / c_style
// conversion the signatures always had; and the host (`_mw_nd_binding.h`)
// through which the core allocates its outputs with numpy (so tracemalloc
// still sees them), releases the GIL where it always did, and casts an
// optional array argument at the point it always was.
#include "_accel_common.h"
#include "_mw_nd_binding.h"
#include "_bspline_core.h"

#include <complex>
#include <tuple>


static double
bspline_j_static_moment_py(
    int p,
    int q,
    double alpha,
    double beta,
    double A,
    double B,
    double a) {
    ndb::NumpyHost host;
    mw::nd::HostScope scope(host);
    return mw::bspline::bspline_j_static_moment_py(p, q, alpha, beta, A, B, a);
}

static double
bspline_d_ek_moment_py(
    int p,
    int q,
    double alpha,
    double beta,
    double A,
    double B,
    double a) {
    ndb::NumpyHost host;
    mw::nd::HostScope scope(host);
    return mw::bspline::bspline_d_ek_moment_py(p, q, alpha, beta, A, B, a);
}

static double
bspline_j_static_far_py(
    int p,
    int q,
    double alpha,
    double beta,
    double A,
    double B,
    double a) {
    ndb::NumpyHost host;
    mw::nd::HostScope scope(host);
    return mw::bspline::bspline_j_static_far_py(p, q, alpha, beta, A, B, a);
}

static py::array_t<std::complex<double>>
seg_seg_reg_moments_bspline_swept(
    py::array_t<double, py::array::c_style | py::array::forcecast> R,
    py::array_t<double, py::array::c_style | py::array::forcecast> wu_pow,
    py::array_t<double, py::array::c_style | py::array::forcecast> k_array) {
    ndb::NumpyHost host;
    mw::nd::HostScope scope(host);
    return ndb::out(
        mw::bspline::seg_seg_reg_moments_bspline_swept(ndb::in(R), ndb::in(wu_pow),
            ndb::in(k_array)));
}

static py::array_t<std::complex<double>>
seg_seg_reg_moments_bspline_swept_ek(
    py::array_t<double, py::array::c_style | py::array::forcecast> R,
    py::array_t<double, py::array::c_style | py::array::forcecast> wu_pow,
    py::array_t<double, py::array::c_style | py::array::forcecast> k_array,
    double a_ek) {
    ndb::NumpyHost host;
    mw::nd::HostScope scope(host);
    return ndb::out(
        mw::bspline::seg_seg_reg_moments_bspline_swept_ek(ndb::in(R), ndb::in(wu_pow),
            ndb::in(k_array), a_ek));
}

static py::array_t<std::complex<double>>
seg_seg_reg_moments_bspline_swept_window(
    py::array_t<double, py::array::c_style | py::array::forcecast> R,
    py::array_t<double, py::array::c_style | py::array::forcecast> wu_row,
    py::array_t<double, py::array::c_style | py::array::forcecast> wu_col,
    py::array_t<double, py::array::c_style | py::array::forcecast> k_array) {
    ndb::NumpyHost host;
    mw::nd::HostScope scope(host);
    return ndb::out(
        mw::bspline::seg_seg_reg_moments_bspline_swept_window(ndb::in(R), ndb::in(wu_row),
            ndb::in(wu_col), ndb::in(k_array)));
}

static py::array_t<std::complex<double>>
seg_seg_reg_moments_bspline_swept_ek_window(
    py::array_t<double, py::array::c_style | py::array::forcecast> R,
    py::array_t<double, py::array::c_style | py::array::forcecast> wu_row,
    py::array_t<double, py::array::c_style | py::array::forcecast> wu_col,
    py::array_t<double, py::array::c_style | py::array::forcecast> k_array,
    double a_ek) {
    ndb::NumpyHost host;
    mw::nd::HostScope scope(host);
    return ndb::out(
        mw::bspline::seg_seg_reg_moments_bspline_swept_ek_window(ndb::in(R), ndb::in(wu_row),
            ndb::in(wu_col), ndb::in(k_array), a_ek));
}

static py::array_t<std::complex<double>>
seg_seg_full_moments_bspline(
    py::array_t<double, py::array::c_style | py::array::forcecast> seg_l_i,
    py::array_t<double, py::array::c_style | py::array::forcecast> seg_r_i,
    py::array_t<double, py::array::c_style | py::array::forcecast> seg_l_j,
    py::array_t<double, py::array::c_style | py::array::forcecast> seg_r_j,
    double a_squared,
    double k,
    int max_d,
    py::array_t<double, py::array::c_style | py::array::forcecast> gl_t,
    py::array_t<double, py::array::c_style | py::array::forcecast> gl_w,
    bool reference) {
    ndb::NumpyHost host;
    mw::nd::HostScope scope(host);
    return ndb::out(
        mw::bspline::seg_seg_full_moments_bspline(ndb::in(seg_l_i), ndb::in(seg_r_i),
            ndb::in(seg_l_j), ndb::in(seg_r_j), a_squared, k, max_d, ndb::in(gl_t), ndb::in(gl_w),
            reference));
}

static py::array_t<std::complex<double>>
seg_seg_full_moments_bspline_cplx(
    py::array_t<double, py::array::c_style | py::array::forcecast> seg_l_i,
    py::array_t<double, py::array::c_style | py::array::forcecast> seg_r_i,
    py::array_t<double, py::array::c_style | py::array::forcecast> seg_l_j,
    py::array_t<double, py::array::c_style | py::array::forcecast> seg_r_j,
    double a_squared,
    std::complex<double> k,
    int max_d,
    py::array_t<double, py::array::c_style | py::array::forcecast> gl_t,
    py::array_t<double, py::array::c_style | py::array::forcecast> gl_w,
    bool reference) {
    ndb::NumpyHost host;
    mw::nd::HostScope scope(host);
    return ndb::out(
        mw::bspline::seg_seg_full_moments_bspline_cplx(ndb::in(seg_l_i), ndb::in(seg_r_i),
            ndb::in(seg_l_j), ndb::in(seg_r_j), a_squared, k, max_d, ndb::in(gl_t), ndb::in(gl_w),
            reference));
}

static py::array_t<std::complex<double>>
seg_seg_full_moments_bspline_tiered(
    py::array_t<double, py::array::c_style | py::array::forcecast> seg_l_i,
    py::array_t<double, py::array::c_style | py::array::forcecast> seg_r_i,
    py::array_t<double, py::array::c_style | py::array::forcecast> seg_l_j,
    py::array_t<double, py::array::c_style | py::array::forcecast> seg_r_j,
    double a_squared,
    double k,
    int max_d,
    py::array_t<double, py::array::c_style | py::array::forcecast> tier_t,
    py::array_t<double, py::array::c_style | py::array::forcecast> tier_w,
    py::array_t<int64_t, py::array::c_style | py::array::forcecast> tier_n_qp,
    py::array_t<double, py::array::c_style | py::array::forcecast> tier_ratio,
    bool reference) {
    ndb::NumpyHost host;
    mw::nd::HostScope scope(host);
    return ndb::out(
        mw::bspline::seg_seg_full_moments_bspline_tiered(ndb::in(seg_l_i), ndb::in(seg_r_i),
            ndb::in(seg_l_j), ndb::in(seg_r_j), a_squared, k, max_d, ndb::in(tier_t),
            ndb::in(tier_w), ndb::in(tier_n_qp), ndb::in(tier_ratio), reference));
}

static py::array_t<std::complex<double>>
seg_seg_full_moments_bspline_cplx_tiered(
    py::array_t<double, py::array::c_style | py::array::forcecast> seg_l_i,
    py::array_t<double, py::array::c_style | py::array::forcecast> seg_r_i,
    py::array_t<double, py::array::c_style | py::array::forcecast> seg_l_j,
    py::array_t<double, py::array::c_style | py::array::forcecast> seg_r_j,
    double a_squared,
    std::complex<double> k,
    int max_d,
    py::array_t<double, py::array::c_style | py::array::forcecast> tier_t,
    py::array_t<double, py::array::c_style | py::array::forcecast> tier_w,
    py::array_t<int64_t, py::array::c_style | py::array::forcecast> tier_n_qp,
    py::array_t<double, py::array::c_style | py::array::forcecast> tier_ratio,
    bool reference) {
    ndb::NumpyHost host;
    mw::nd::HostScope scope(host);
    return ndb::out(
        mw::bspline::seg_seg_full_moments_bspline_cplx_tiered(ndb::in(seg_l_i), ndb::in(seg_r_i),
            ndb::in(seg_l_j), ndb::in(seg_r_j), a_squared, k, max_d, ndb::in(tier_t),
            ndb::in(tier_w), ndb::in(tier_n_qp), ndb::in(tier_ratio), reference));
}

static py::array_t<int64_t> bspline_pair_tiers(
    py::array_t<double, py::array::c_style | py::array::forcecast> seg_l_i,
    py::array_t<double, py::array::c_style | py::array::forcecast> seg_r_i,
    py::array_t<double, py::array::c_style | py::array::forcecast> seg_l_j,
    py::array_t<double, py::array::c_style | py::array::forcecast> seg_r_j,
    py::array_t<double, py::array::c_style | py::array::forcecast> tier_t,
    py::array_t<double, py::array::c_style | py::array::forcecast> tier_w,
    py::array_t<int64_t, py::array::c_style | py::array::forcecast> tier_n_qp,
    py::array_t<double, py::array::c_style | py::array::forcecast> tier_ratio) {
    ndb::NumpyHost host;
    mw::nd::HostScope scope(host);
    return ndb::out(mw::bspline::bspline_pair_tiers(ndb::in(seg_l_i), ndb::in(seg_r_i), ndb::in(seg_l_j),
                                                    ndb::in(seg_r_j), ndb::in(tier_t), ndb::in(tier_w),
                                                    ndb::in(tier_n_qp), ndb::in(tier_ratio)));
}

static py::array_t<std::complex<double>>
seg_seg_full_moments_sinusoidal_tiered(
    py::array_t<double, py::array::c_style | py::array::forcecast> seg_l_i,
    py::array_t<double, py::array::c_style | py::array::forcecast> seg_r_i,
    py::array_t<double, py::array::c_style | py::array::forcecast> seg_l_j,
    py::array_t<double, py::array::c_style | py::array::forcecast> seg_r_j,
    double a_squared,
    std::complex<double> k,
    py::array_t<double, py::array::c_style | py::array::forcecast> tier_t,
    py::array_t<double, py::array::c_style | py::array::forcecast> tier_w,
    py::array_t<int64_t, py::array::c_style | py::array::forcecast> tier_n_qp,
    py::array_t<double, py::array::c_style | py::array::forcecast> tier_ratio,
    bool reference) {
    ndb::NumpyHost host;
    mw::nd::HostScope scope(host);
    return ndb::out(
        mw::bspline::seg_seg_full_moments_sinusoidal_tiered(ndb::in(seg_l_i), ndb::in(seg_r_i),
            ndb::in(seg_l_j), ndb::in(seg_r_j), a_squared, k, ndb::in(tier_t), ndb::in(tier_w),
            ndb::in(tier_n_qp), ndb::in(tier_ratio), reference));
}

static py::array_t<std::complex<double>>
seg_seg_full_moments_sinusoidal_tiered_ek(
    py::array_t<double, py::array::c_style | py::array::forcecast> seg_l_i,
    py::array_t<double, py::array::c_style | py::array::forcecast> seg_r_i,
    py::array_t<double, py::array::c_style | py::array::forcecast> seg_l_j,
    py::array_t<double, py::array::c_style | py::array::forcecast> seg_r_j,
    double a_squared,
    double k,
    py::array_t<double, py::array::c_style | py::array::forcecast> tier_t,
    py::array_t<double, py::array::c_style | py::array::forcecast> tier_w,
    py::array_t<int64_t, py::array::c_style | py::array::forcecast> tier_n_qp,
    py::array_t<double, py::array::c_style | py::array::forcecast> tier_ratio,
    py::array_t<int64_t, py::array::c_style | py::array::forcecast> group_i,
    py::array_t<int64_t, py::array::c_style | py::array::forcecast> group_j,
    double a_ek,
    py::object b_j_obj) {
    ndb::NumpyHost host;
    mw::nd::HostScope scope(host);
    return ndb::out(
        mw::bspline::seg_seg_full_moments_sinusoidal_tiered_ek(ndb::in(seg_l_i), ndb::in(seg_r_i),
            ndb::in(seg_l_j), ndb::in(seg_r_j), a_squared, k, ndb::in(tier_t), ndb::in(tier_w),
            ndb::in(tier_n_qp), ndb::in(tier_ratio), ndb::in(group_i), ndb::in(group_j), a_ek,
            ndb::object(b_j_obj)));
}

static void
assemble_Z_sinusoidal_windowed(
    py::array_t<std::complex<double>, py::array::c_style | py::array::forcecast> J,
    py::array_t<int64_t, py::array::c_style | py::array::forcecast> rows_seg,
    py::array_t<int64_t, py::array::c_style | py::array::forcecast> cols_seg,
    py::array_t<int64_t, py::array::c_style | py::array::forcecast> starts,
    py::array_t<int64_t, py::array::c_style | py::array::forcecast> jbasis,
    py::array_t<std::complex<double>, py::array::c_style | py::array::forcecast> coef,
    py::array_t<std::complex<double>, py::array::c_style | py::array::forcecast> dcoef,
    py::array_t<double, py::array::c_style | py::array::forcecast> tangents_rows,
    py::array_t<double, py::array::c_style | py::array::forcecast> tangents_cols,
    std::complex<double> c_a,
    std::complex<double> c_phi,
    py::object w_a_obj,
    py::object w_phi_obj,
    py::array_t<std::complex<double>> Z,
    uintptr_t cancel_flag) {
    ndb::NumpyHost host;
    mw::nd::HostScope scope(host);
    mw::bspline::assemble_Z_sinusoidal_windowed(ndb::in(J), ndb::in(rows_seg), ndb::in(cols_seg),
        ndb::in(starts), ndb::in(jbasis), ndb::in(coef), ndb::in(dcoef), ndb::in(tangents_rows),
        ndb::in(tangents_cols), c_a, c_phi, ndb::object(w_a_obj), ndb::object(w_phi_obj),
        ndb::in(Z), cancel_flag);
}

static py::array_t<std::complex<double>>
field_pair_moments_sinusoidal(
    py::array_t<std::complex<double>, py::array::c_style | py::array::forcecast> F,
    py::array_t<std::complex<double>, py::array::c_style | py::array::forcecast> W_obs,
    py::array_t<std::complex<double>, py::array::c_style | py::array::forcecast> W_src) {
    ndb::NumpyHost host;
    mw::nd::HostScope scope(host);
    return ndb::out(
        mw::bspline::field_pair_moments_sinusoidal(ndb::in(F), ndb::in(W_obs), ndb::in(W_src)));
}

static py::array_t<std::complex<double>>
parallel_pair_moments_sinusoidal(
    py::array_t<double, py::array::c_style | py::array::forcecast> c_i,
    py::array_t<double, py::array::c_style | py::array::forcecast> t_i,
    py::array_t<double, py::array::c_style | py::array::forcecast> h_i,
    py::array_t<double, py::array::c_style | py::array::forcecast> c_j,
    py::array_t<double, py::array::c_style | py::array::forcecast> t_j,
    py::array_t<double, py::array::c_style | py::array::forcecast> h_j,
    py::array_t<double, py::array::c_style | py::array::forcecast> a2,
    std::complex<double> k,
    py::array_t<double, py::array::c_style | py::array::forcecast> gt,
    py::array_t<double, py::array::c_style | py::array::forcecast> gw,
    py::array_t<double, py::array::c_style | py::array::forcecast> gx,
    py::array_t<double, py::array::c_style | py::array::forcecast> gwx,
    py::object a_ek_obj,
    py::object b_ek_obj) {
    ndb::NumpyHost host;
    mw::nd::HostScope scope(host);
    return ndb::out(
        mw::bspline::parallel_pair_moments_sinusoidal(ndb::in(c_i), ndb::in(t_i), ndb::in(h_i),
            ndb::in(c_j), ndb::in(t_j), ndb::in(h_j), ndb::in(a2), k, ndb::in(gt), ndb::in(gw),
            ndb::in(gx), ndb::in(gwx), ndb::object(a_ek_obj), ndb::object(b_ek_obj)));
}

static py::array_t<std::complex<double>>
seg_seg_full_moments_bspline_swept(
    py::array_t<double, py::array::c_style | py::array::forcecast> seg_l_i,
    py::array_t<double, py::array::c_style | py::array::forcecast> seg_r_i,
    py::array_t<double, py::array::c_style | py::array::forcecast> seg_l_j,
    py::array_t<double, py::array::c_style | py::array::forcecast> seg_r_j,
    double a_squared,
    py::array_t<double, py::array::c_style | py::array::forcecast> k_array,
    int max_d,
    py::array_t<double, py::array::c_style | py::array::forcecast> gl_t,
    py::array_t<double, py::array::c_style | py::array::forcecast> gl_w) {
    ndb::NumpyHost host;
    mw::nd::HostScope scope(host);
    return ndb::out(
        mw::bspline::seg_seg_full_moments_bspline_swept(ndb::in(seg_l_i), ndb::in(seg_r_i),
            ndb::in(seg_l_j), ndb::in(seg_r_j), a_squared, ndb::in(k_array), max_d, ndb::in(gl_t),
            ndb::in(gl_w)));
}

static py::array_t<std::complex<double>>
seg_seg_full_moments_bspline_ek(
    py::array_t<double, py::array::c_style | py::array::forcecast> seg_l_i,
    py::array_t<double, py::array::c_style | py::array::forcecast> seg_r_i,
    py::array_t<double, py::array::c_style | py::array::forcecast> seg_l_j,
    py::array_t<double, py::array::c_style | py::array::forcecast> seg_r_j,
    double a_squared,
    double k,
    int max_d,
    py::array_t<double, py::array::c_style | py::array::forcecast> gl_t,
    py::array_t<double, py::array::c_style | py::array::forcecast> gl_w,
    py::array_t<int64_t, py::array::c_style | py::array::forcecast> group_i,
    py::array_t<int64_t, py::array::c_style | py::array::forcecast> group_j,
    double a_ek,
    bool reference,
    double a_ek_src) {
    ndb::NumpyHost host;
    mw::nd::HostScope scope(host);
    return ndb::out(
        mw::bspline::seg_seg_full_moments_bspline_ek(ndb::in(seg_l_i), ndb::in(seg_r_i),
            ndb::in(seg_l_j), ndb::in(seg_r_j), a_squared, k, max_d, ndb::in(gl_t), ndb::in(gl_w),
            ndb::in(group_i), ndb::in(group_j), a_ek, reference, a_ek_src));
}

static py::array_t<std::complex<double>>
seg_seg_full_moments_bspline_ek_tiered(
    py::array_t<double, py::array::c_style | py::array::forcecast> seg_l_i,
    py::array_t<double, py::array::c_style | py::array::forcecast> seg_r_i,
    py::array_t<double, py::array::c_style | py::array::forcecast> seg_l_j,
    py::array_t<double, py::array::c_style | py::array::forcecast> seg_r_j,
    double a_squared,
    double k,
    int max_d,
    py::array_t<double, py::array::c_style | py::array::forcecast> tier_t,
    py::array_t<double, py::array::c_style | py::array::forcecast> tier_w,
    py::array_t<int64_t, py::array::c_style | py::array::forcecast> tier_n_qp,
    py::array_t<double, py::array::c_style | py::array::forcecast> tier_ratio,
    py::array_t<int64_t, py::array::c_style | py::array::forcecast> group_i,
    py::array_t<int64_t, py::array::c_style | py::array::forcecast> group_j,
    double a_ek,
    bool reference,
    double a_ek_src) {
    ndb::NumpyHost host;
    mw::nd::HostScope scope(host);
    return ndb::out(
        mw::bspline::seg_seg_full_moments_bspline_ek_tiered(ndb::in(seg_l_i), ndb::in(seg_r_i),
            ndb::in(seg_l_j), ndb::in(seg_r_j), a_squared, k, max_d, ndb::in(tier_t),
            ndb::in(tier_w), ndb::in(tier_n_qp), ndb::in(tier_ratio), ndb::in(group_i),
            ndb::in(group_j), a_ek, reference, a_ek_src));
}

static py::array_t<std::complex<double>>
seg_seg_full_moments_bspline_swept_ek(
    py::array_t<double, py::array::c_style | py::array::forcecast> seg_l_i,
    py::array_t<double, py::array::c_style | py::array::forcecast> seg_r_i,
    py::array_t<double, py::array::c_style | py::array::forcecast> seg_l_j,
    py::array_t<double, py::array::c_style | py::array::forcecast> seg_r_j,
    double a_squared,
    py::array_t<double, py::array::c_style | py::array::forcecast> k_array,
    int max_d,
    py::array_t<double, py::array::c_style | py::array::forcecast> gl_t,
    py::array_t<double, py::array::c_style | py::array::forcecast> gl_w,
    py::array_t<int64_t, py::array::c_style | py::array::forcecast> group_i,
    py::array_t<int64_t, py::array::c_style | py::array::forcecast> group_j,
    double a_ek,
    bool reference,
    double a_ek_src) {
    ndb::NumpyHost host;
    mw::nd::HostScope scope(host);
    return ndb::out(
        mw::bspline::seg_seg_full_moments_bspline_swept_ek(ndb::in(seg_l_i), ndb::in(seg_r_i),
            ndb::in(seg_l_j), ndb::in(seg_r_j), a_squared, ndb::in(k_array), max_d, ndb::in(gl_t),
            ndb::in(gl_w), ndb::in(group_i), ndb::in(group_j), a_ek, reference, a_ek_src));
}

static py::array_t<std::complex<double>>
seg_seg_full_moments_bspline_swept_ek_tiered(
    py::array_t<double, py::array::c_style | py::array::forcecast> seg_l_i,
    py::array_t<double, py::array::c_style | py::array::forcecast> seg_r_i,
    py::array_t<double, py::array::c_style | py::array::forcecast> seg_l_j,
    py::array_t<double, py::array::c_style | py::array::forcecast> seg_r_j,
    double a_squared,
    py::array_t<double, py::array::c_style | py::array::forcecast> k_array,
    int max_d,
    py::array_t<double, py::array::c_style | py::array::forcecast> tier_t,
    py::array_t<double, py::array::c_style | py::array::forcecast> tier_w,
    py::array_t<int64_t, py::array::c_style | py::array::forcecast> tier_n_qp,
    py::array_t<double, py::array::c_style | py::array::forcecast> tier_ratio,
    py::array_t<int64_t, py::array::c_style | py::array::forcecast> group_i,
    py::array_t<int64_t, py::array::c_style | py::array::forcecast> group_j,
    double a_ek,
    bool reference,
    double a_ek_src) {
    ndb::NumpyHost host;
    mw::nd::HostScope scope(host);
    return ndb::out(
        mw::bspline::seg_seg_full_moments_bspline_swept_ek_tiered(ndb::in(seg_l_i),
            ndb::in(seg_r_i), ndb::in(seg_l_j), ndb::in(seg_r_j), a_squared, ndb::in(k_array),
            max_d, ndb::in(tier_t), ndb::in(tier_w), ndb::in(tier_n_qp), ndb::in(tier_ratio),
            ndb::in(group_i), ndb::in(group_j), a_ek, reference, a_ek_src));
}

static py::array_t<double>
seg_seg_static_moments_bspline_uniform(
    double h,
    double a,
    size_t N,
    int max_d,
    uintptr_t cancel_flag,
    uintptr_t poll_probe) {
    ndb::NumpyHost host;
    mw::nd::HostScope scope(host);
    return ndb::out(
        mw::bspline::seg_seg_static_moments_bspline_uniform(h, a, N, max_d, cancel_flag,
            poll_probe));
}

static py::array_t<double>
seg_seg_static_moments_bspline_uniform_ek(
    double h,
    double a,
    size_t N,
    int max_d,
    double a_ek,
    uintptr_t cancel_flag) {
    ndb::NumpyHost host;
    mw::nd::HostScope scope(host);
    return ndb::out(
        mw::bspline::seg_seg_static_moments_bspline_uniform_ek(h, a, N, max_d, a_ek, cancel_flag));
}

static py::array_t<double>
toeplitz_gather_rows(
    py::array_t<double, py::array::c_style | py::array::forcecast> table,
    size_t N,
    size_t r0,
    size_t r1) {
    ndb::NumpyHost host;
    mw::nd::HostScope scope(host);
    return ndb::out(
        mw::bspline::toeplitz_gather_rows(ndb::in(table), N, r0, r1));
}

static py::array_t<double>
reg_distance_table(
    py::array_t<double, py::array::c_style | py::array::forcecast> s_row,
    py::array_t<double, py::array::c_style | py::array::forcecast> s_col,
    double a2) {
    ndb::NumpyHost host;
    mw::nd::HostScope scope(host);
    return ndb::out(
        mw::bspline::reg_distance_table(ndb::in(s_row), ndb::in(s_col), a2));
}

static py::array_t<std::complex<double>>
seg_seg_reg_moments_bspline_cplx(
    py::array_t<double, py::array::c_style | py::array::forcecast> R,
    py::array_t<double, py::array::c_style | py::array::forcecast> wu_pow,
    std::complex<double> k) {
    ndb::NumpyHost host;
    mw::nd::HostScope scope(host);
    return ndb::out(mw::bspline::seg_seg_reg_moments_bspline_cplx(ndb::in(R), ndb::in(wu_pow), k));
}

static py::array_t<std::complex<double>>
same_edge_corr_combine(
    py::array_t<double, py::array::c_style | py::array::forcecast> A_st,
    py::array_t<std::complex<double>, py::array::c_style | py::array::forcecast> A_reg,
    py::array_t<std::complex<double>, py::array::c_style | py::array::forcecast> J) {
    ndb::NumpyHost host;
    mw::nd::HostScope scope(host);
    return ndb::out(
        mw::bspline::same_edge_corr_combine(ndb::in(A_st), ndb::in(A_reg), ndb::in(J)));
}

static py::array_t<double>
seg_seg_static_moments_bspline_uniform_table(
    double h,
    double a,
    size_t N,
    int max_d) {
    ndb::NumpyHost host;
    mw::nd::HostScope scope(host);
    return ndb::out(
        mw::bspline::seg_seg_static_moments_bspline_uniform_table(h, a, N, max_d));
}

static py::array_t<double>
seg_seg_static_moments_bspline_uniform_ek_table(
    double h,
    double a,
    size_t N,
    int max_d,
    double a_ek) {
    ndb::NumpyHost host;
    mw::nd::HostScope scope(host);
    return ndb::out(
        mw::bspline::seg_seg_static_moments_bspline_uniform_ek_table(h, a, N, max_d, a_ek));
}

static void
assemble_Z_bspline_weighted_windowed(
    py::array_t<std::complex<double>, py::array::c_style | py::array::forcecast> J_chunk,
    py::array_t<int64_t, py::array::c_style | py::array::forcecast> support_seg,
    py::array_t<double, py::array::c_style | py::array::forcecast> polys,
    py::array_t<std::complex<double>, py::array::c_style | py::array::forcecast> wA_win,
    py::array_t<std::complex<double>, py::array::c_style | py::array::forcecast> wPhi_win,
    py::array_t<int64_t, py::array::c_style | py::array::forcecast> m_idx,
    py::array_t<int64_t, py::array::c_style | py::array::forcecast> n_idx,
    int64_t i0,
    int64_t i1,
    int64_t j0,
    int64_t j1,
    double omega,
    double eps_,
    double mu_,
    std::complex<double> scale,
    py::array_t<std::complex<double>> Z,
    uintptr_t cancel_flag,
    py::object row_of,
    bool reference) {
    ndb::NumpyHost host;
    mw::nd::HostScope scope(host);
    mw::bspline::assemble_Z_bspline_weighted_windowed(ndb::in(J_chunk), ndb::in(support_seg),
        ndb::in(polys), ndb::in(wA_win), ndb::in(wPhi_win), ndb::in(m_idx), ndb::in(n_idx), i0, i1,
        j0, j1, omega, eps_, mu_, scale, ndb::in(Z), cancel_flag, ndb::object(row_of), reference);
}

static void
assemble_Z_bspline_windowed_cplx_eps(
    py::array_t<std::complex<double>, py::array::c_style | py::array::forcecast> J_chunk,
    py::array_t<int64_t, py::array::c_style | py::array::forcecast> support_seg,
    py::array_t<double, py::array::c_style | py::array::forcecast> polys,
    py::array_t<double, py::array::c_style | py::array::forcecast> tangents,
    py::array_t<int64_t, py::array::c_style | py::array::forcecast> m_idx,
    py::array_t<int64_t, py::array::c_style | py::array::forcecast> n_idx,
    int64_t i0,
    int64_t i1,
    int64_t j0,
    int64_t j1,
    double omega,
    std::complex<double> eps_,
    double mu_,
    py::array_t<std::complex<double>> Z,
    uintptr_t cancel_flag,
    py::object row_of,
    bool reference) {
    ndb::NumpyHost host;
    mw::nd::HostScope scope(host);
    mw::bspline::assemble_Z_bspline_windowed_cplx_eps(ndb::in(J_chunk), ndb::in(support_seg),
        ndb::in(polys), ndb::in(tangents), ndb::in(m_idx), ndb::in(n_idx), i0, i1, j0, j1, omega,
        eps_, mu_, ndb::in(Z), cancel_flag, ndb::object(row_of), reference);
}

static void
assemble_Z_bspline_weighted_windowed_cplx_eps(
    py::array_t<std::complex<double>, py::array::c_style | py::array::forcecast> J_chunk,
    py::array_t<int64_t, py::array::c_style | py::array::forcecast> support_seg,
    py::array_t<double, py::array::c_style | py::array::forcecast> polys,
    py::array_t<std::complex<double>, py::array::c_style | py::array::forcecast> wA_win,
    py::array_t<std::complex<double>, py::array::c_style | py::array::forcecast> wPhi_win,
    py::array_t<int64_t, py::array::c_style | py::array::forcecast> m_idx,
    py::array_t<int64_t, py::array::c_style | py::array::forcecast> n_idx,
    int64_t i0,
    int64_t i1,
    int64_t j0,
    int64_t j1,
    double omega,
    std::complex<double> eps_,
    double mu_,
    std::complex<double> scale,
    py::array_t<std::complex<double>> Z,
    uintptr_t cancel_flag,
    py::object row_of,
    bool reference) {
    ndb::NumpyHost host;
    mw::nd::HostScope scope(host);
    mw::bspline::assemble_Z_bspline_weighted_windowed_cplx_eps(ndb::in(J_chunk),
        ndb::in(support_seg), ndb::in(polys), ndb::in(wA_win), ndb::in(wPhi_win), ndb::in(m_idx),
        ndb::in(n_idx), i0, i1, j0, j1, omega, eps_, mu_, scale, ndb::in(Z), cancel_flag,
        ndb::object(row_of), reference);
}

static void
assemble_Z_bspline_windowed(
    py::array_t<std::complex<double>, py::array::c_style | py::array::forcecast> J_chunk,
    py::array_t<int64_t, py::array::c_style | py::array::forcecast> support_seg,
    py::array_t<double, py::array::c_style | py::array::forcecast> polys,
    py::array_t<double, py::array::c_style | py::array::forcecast> tangents,
    py::array_t<int64_t, py::array::c_style | py::array::forcecast> m_idx,
    py::array_t<int64_t, py::array::c_style | py::array::forcecast> n_idx,
    int64_t i0,
    int64_t i1,
    int64_t j0,
    int64_t j1,
    double omega,
    double eps_,
    double mu_,
    py::array_t<std::complex<double>> Z,
    uintptr_t cancel_flag,
    py::object row_of,
    bool reference) {
    ndb::NumpyHost host;
    mw::nd::HostScope scope(host);
    mw::bspline::assemble_Z_bspline_windowed(ndb::in(J_chunk), ndb::in(support_seg),
        ndb::in(polys), ndb::in(tangents), ndb::in(m_idx), ndb::in(n_idx), i0, i1, j0, j1, omega,
        eps_, mu_, ndb::in(Z), cancel_flag, ndb::object(row_of), reference);
}

static py::array_t<std::complex<double>>
assemble_Z_bspline(
    py::array_t<std::complex<double>, py::array::c_style | py::array::forcecast> J,
    py::array_t<int64_t, py::array::c_style | py::array::forcecast> support_seg,
    py::array_t<double, py::array::c_style | py::array::forcecast> polys,
    py::array_t<double, py::array::c_style | py::array::forcecast> td_all,
    double omega,
    double eps_,
    double mu_,
    int max_d,
    uintptr_t cancel_flag) {
    ndb::NumpyHost host;
    mw::nd::HostScope scope(host);
    return ndb::out(
        mw::bspline::assemble_Z_bspline(ndb::in(J), ndb::in(support_seg), ndb::in(polys),
            ndb::in(td_all), omega, eps_, mu_, max_d, cancel_flag));
}

static py::array_t<std::complex<double>>
assemble_Z_bspline_weighted(
    py::array_t<std::complex<double>, py::array::c_style | py::array::forcecast> J,
    py::array_t<int64_t, py::array::c_style | py::array::forcecast> support_seg,
    py::array_t<double, py::array::c_style | py::array::forcecast> polys,
    py::array_t<std::complex<double>, py::array::c_style | py::array::forcecast> wA_all,
    py::array_t<std::complex<double>, py::array::c_style | py::array::forcecast> wPhi_all,
    double omega,
    double eps_,
    double mu_,
    int max_d,
    uintptr_t cancel_flag) {
    ndb::NumpyHost host;
    mw::nd::HostScope scope(host);
    return ndb::out(
        mw::bspline::assemble_Z_bspline_weighted(ndb::in(J), ndb::in(support_seg), ndb::in(polys),
            ndb::in(wA_all), ndb::in(wPhi_all), omega, eps_, mu_, max_d, cancel_flag));
}

static py::array_t<std::complex<double>>
assemble_Z_bspline_cplx_eps(
    py::array_t<std::complex<double>, py::array::c_style | py::array::forcecast> J,
    py::array_t<int64_t, py::array::c_style | py::array::forcecast> support_seg,
    py::array_t<double, py::array::c_style | py::array::forcecast> polys,
    py::array_t<double, py::array::c_style | py::array::forcecast> td_all,
    double omega,
    std::complex<double> eps_,
    double mu_,
    int max_d,
    uintptr_t cancel_flag) {
    ndb::NumpyHost host;
    mw::nd::HostScope scope(host);
    return ndb::out(
        mw::bspline::assemble_Z_bspline_cplx_eps(ndb::in(J), ndb::in(support_seg), ndb::in(polys),
            ndb::in(td_all), omega, eps_, mu_, max_d, cancel_flag));
}

static py::array_t<std::complex<double>>
assemble_Z_bspline_weighted_cplx_eps(
    py::array_t<std::complex<double>, py::array::c_style | py::array::forcecast> J,
    py::array_t<int64_t, py::array::c_style | py::array::forcecast> support_seg,
    py::array_t<double, py::array::c_style | py::array::forcecast> polys,
    py::array_t<std::complex<double>, py::array::c_style | py::array::forcecast> wA_all,
    py::array_t<std::complex<double>, py::array::c_style | py::array::forcecast> wPhi_all,
    double omega,
    std::complex<double> eps_,
    double mu_,
    int max_d,
    uintptr_t cancel_flag) {
    ndb::NumpyHost host;
    mw::nd::HostScope scope(host);
    return ndb::out(
        mw::bspline::assemble_Z_bspline_weighted_cplx_eps(ndb::in(J), ndb::in(support_seg),
            ndb::in(polys), ndb::in(wA_all), ndb::in(wPhi_all), omega, eps_, mu_, max_d,
            cancel_flag));
}

static py::array_t<std::complex<double>>
assemble_Z_bspline_swept(
    py::array_t<std::complex<double>, py::array::c_style | py::array::forcecast> J,
    py::array_t<int64_t, py::array::c_style | py::array::forcecast> support_seg,
    py::array_t<double, py::array::c_style | py::array::forcecast> polys,
    py::array_t<double, py::array::c_style | py::array::forcecast> tangents_row,
    py::array_t<double, py::array::c_style | py::array::forcecast> tangents_col,
    py::array_t<double, py::array::c_style | py::array::forcecast> omega_array,
    double eps_,
    double mu_,
    int max_d) {
    ndb::NumpyHost host;
    mw::nd::HostScope scope(host);
    return ndb::out(
        mw::bspline::assemble_Z_bspline_swept(ndb::in(J), ndb::in(support_seg), ndb::in(polys),
            ndb::in(tangents_row), ndb::in(tangents_col), ndb::in(omega_array), eps_, mu_, max_d));
}

static py::array_t<std::complex<double>>
bspline_assemble_offedge_block(
    py::array_t<int64_t, py::array::c_style | py::array::forcecast> supp_I,
    py::array_t<double, py::array::c_style | py::array::forcecast> polys_I,
    py::array_t<double, py::array::c_style | py::array::forcecast> segl_I,
    py::array_t<double, py::array::c_style | py::array::forcecast> segr_I,
    py::array_t<double, py::array::c_style | py::array::forcecast> tan_I,
    py::array_t<int64_t, py::array::c_style | py::array::forcecast> supp_J,
    py::array_t<double, py::array::c_style | py::array::forcecast> polys_J,
    py::array_t<double, py::array::c_style | py::array::forcecast> segl_J,
    py::array_t<double, py::array::c_style | py::array::forcecast> segr_J,
    py::array_t<double, py::array::c_style | py::array::forcecast> tan_J,
    double a_squared,
    double k,
    double omega,
    double eps_,
    double mu_,
    int max_d,
    py::array_t<double, py::array::c_style | py::array::forcecast> gl_t,
    py::array_t<double, py::array::c_style | py::array::forcecast> gl_w,
    uintptr_t cancel_flag) {
    ndb::NumpyHost host;
    mw::nd::HostScope scope(host);
    return ndb::out(
        mw::bspline::bspline_assemble_offedge_block(ndb::in(supp_I), ndb::in(polys_I),
            ndb::in(segl_I), ndb::in(segr_I), ndb::in(tan_I), ndb::in(supp_J), ndb::in(polys_J),
            ndb::in(segl_J), ndb::in(segr_J), ndb::in(tan_J), a_squared, k, omega, eps_, mu_,
            max_d, ndb::in(gl_t), ndb::in(gl_w), cancel_flag));
}

static py::array_t<std::complex<double>>
bspline_assemble_offedge_block_refl(
    py::array_t<int64_t, py::array::c_style | py::array::forcecast> supp_I,
    py::array_t<double, py::array::c_style | py::array::forcecast> polys_I,
    py::array_t<double, py::array::c_style | py::array::forcecast> segl_I,
    py::array_t<double, py::array::c_style | py::array::forcecast> segr_I,
    py::array_t<double, py::array::c_style | py::array::forcecast> tan_I,
    py::array_t<int64_t, py::array::c_style | py::array::forcecast> supp_J,
    py::array_t<double, py::array::c_style | py::array::forcecast> polys_J,
    py::array_t<double, py::array::c_style | py::array::forcecast> segl_J,
    py::array_t<double, py::array::c_style | py::array::forcecast> segr_J,
    py::array_t<double, py::array::c_style | py::array::forcecast> tan_J,
    double a_squared,
    double k,
    double omega,
    double eps_,
    double mu_,
    int max_d,
    py::array_t<double, py::array::c_style | py::array::forcecast> gl_t,
    py::array_t<double, py::array::c_style | py::array::forcecast> gl_w,
    std::complex<double> eps_t,
    std::complex<double> phi_c0,
    std::complex<double> phi_c1,
    uintptr_t cancel_flag) {
    ndb::NumpyHost host;
    mw::nd::HostScope scope(host);
    return ndb::out(
        mw::bspline::bspline_assemble_offedge_block_refl(ndb::in(supp_I), ndb::in(polys_I),
            ndb::in(segl_I), ndb::in(segr_I), ndb::in(tan_I), ndb::in(supp_J), ndb::in(polys_J),
            ndb::in(segl_J), ndb::in(segr_J), ndb::in(tan_J), a_squared, k, omega, eps_, mu_,
            max_d, ndb::in(gl_t), ndb::in(gl_w), eps_t, phi_c0, phi_c1, cancel_flag));
}

static py::array_t<std::complex<double>>
bspline_assemble_offedge_block_ek(
    py::array_t<int64_t, py::array::c_style | py::array::forcecast> supp_I,
    py::array_t<double, py::array::c_style | py::array::forcecast> polys_I,
    py::array_t<double, py::array::c_style | py::array::forcecast> segl_I,
    py::array_t<double, py::array::c_style | py::array::forcecast> segr_I,
    py::array_t<double, py::array::c_style | py::array::forcecast> tan_I,
    py::array_t<int64_t, py::array::c_style | py::array::forcecast> supp_J,
    py::array_t<double, py::array::c_style | py::array::forcecast> polys_J,
    py::array_t<double, py::array::c_style | py::array::forcecast> segl_J,
    py::array_t<double, py::array::c_style | py::array::forcecast> segr_J,
    py::array_t<double, py::array::c_style | py::array::forcecast> tan_J,
    double a_squared,
    double k,
    double omega,
    double eps_,
    double mu_,
    int max_d,
    py::array_t<double, py::array::c_style | py::array::forcecast> gl_t,
    py::array_t<double, py::array::c_style | py::array::forcecast> gl_w,
    py::array_t<int64_t, py::array::c_style | py::array::forcecast> group_I,
    py::array_t<int64_t, py::array::c_style | py::array::forcecast> group_J,
    double a_ek,
    uintptr_t cancel_flag,
    bool reference) {
    ndb::NumpyHost host;
    mw::nd::HostScope scope(host);
    return ndb::out(
        mw::bspline::bspline_assemble_offedge_block_ek(ndb::in(supp_I), ndb::in(polys_I),
            ndb::in(segl_I), ndb::in(segr_I), ndb::in(tan_I), ndb::in(supp_J), ndb::in(polys_J),
            ndb::in(segl_J), ndb::in(segr_J), ndb::in(tan_J), a_squared, k, omega, eps_, mu_,
            max_d, ndb::in(gl_t), ndb::in(gl_w), ndb::in(group_I), ndb::in(group_J), a_ek,
            cancel_flag, reference));
}

static py::array_t<std::complex<double>>
bspline_assemble_offedge_block_ek_tiered(
    py::array_t<int64_t, py::array::c_style | py::array::forcecast> supp_I,
    py::array_t<double, py::array::c_style | py::array::forcecast> polys_I,
    py::array_t<double, py::array::c_style | py::array::forcecast> segl_I,
    py::array_t<double, py::array::c_style | py::array::forcecast> segr_I,
    py::array_t<double, py::array::c_style | py::array::forcecast> tan_I,
    py::array_t<int64_t, py::array::c_style | py::array::forcecast> supp_J,
    py::array_t<double, py::array::c_style | py::array::forcecast> polys_J,
    py::array_t<double, py::array::c_style | py::array::forcecast> segl_J,
    py::array_t<double, py::array::c_style | py::array::forcecast> segr_J,
    py::array_t<double, py::array::c_style | py::array::forcecast> tan_J,
    double a_squared,
    double k,
    double omega,
    double eps_,
    double mu_,
    int max_d,
    py::array_t<double, py::array::c_style | py::array::forcecast> tier_t,
    py::array_t<double, py::array::c_style | py::array::forcecast> tier_w,
    py::array_t<int64_t, py::array::c_style | py::array::forcecast> tier_n_qp,
    py::array_t<double, py::array::c_style | py::array::forcecast> tier_ratio,
    py::array_t<int64_t, py::array::c_style | py::array::forcecast> group_I,
    py::array_t<int64_t, py::array::c_style | py::array::forcecast> group_J,
    double a_ek,
    py::array_t<uint8_t, py::array::c_style | py::array::forcecast> phase_ok_I,
    py::array_t<uint8_t, py::array::c_style | py::array::forcecast> phase_ok_J,
    int64_t limited_below,
    uintptr_t cancel_flag,
    bool reference) {
    ndb::NumpyHost host;
    mw::nd::HostScope scope(host);
    return ndb::out(
        mw::bspline::bspline_assemble_offedge_block_ek_tiered(ndb::in(supp_I), ndb::in(polys_I),
            ndb::in(segl_I), ndb::in(segr_I), ndb::in(tan_I), ndb::in(supp_J), ndb::in(polys_J),
            ndb::in(segl_J), ndb::in(segr_J), ndb::in(tan_J), a_squared, k, omega, eps_, mu_,
            max_d, ndb::in(tier_t), ndb::in(tier_w), ndb::in(tier_n_qp), ndb::in(tier_ratio),
            ndb::in(group_I), ndb::in(group_J), a_ek, ndb::in(phase_ok_I), ndb::in(phase_ok_J),
            limited_below, cancel_flag, reference));
}

static py::array_t<std::complex<double>>
bspline_assemble_offedge_block_refl_ek(
    py::array_t<int64_t, py::array::c_style | py::array::forcecast> supp_I,
    py::array_t<double, py::array::c_style | py::array::forcecast> polys_I,
    py::array_t<double, py::array::c_style | py::array::forcecast> segl_I,
    py::array_t<double, py::array::c_style | py::array::forcecast> segr_I,
    py::array_t<double, py::array::c_style | py::array::forcecast> tan_I,
    py::array_t<int64_t, py::array::c_style | py::array::forcecast> supp_J,
    py::array_t<double, py::array::c_style | py::array::forcecast> polys_J,
    py::array_t<double, py::array::c_style | py::array::forcecast> segl_J,
    py::array_t<double, py::array::c_style | py::array::forcecast> segr_J,
    py::array_t<double, py::array::c_style | py::array::forcecast> tan_J,
    double a_squared,
    double k,
    double omega,
    double eps_,
    double mu_,
    int max_d,
    py::array_t<double, py::array::c_style | py::array::forcecast> gl_t,
    py::array_t<double, py::array::c_style | py::array::forcecast> gl_w,
    py::array_t<int64_t, py::array::c_style | py::array::forcecast> group_I,
    py::array_t<int64_t, py::array::c_style | py::array::forcecast> group_J,
    double a_ek,
    std::complex<double> eps_t,
    std::complex<double> phi_c0,
    std::complex<double> phi_c1,
    uintptr_t cancel_flag,
    bool reference) {
    ndb::NumpyHost host;
    mw::nd::HostScope scope(host);
    return ndb::out(
        mw::bspline::bspline_assemble_offedge_block_refl_ek(ndb::in(supp_I), ndb::in(polys_I),
            ndb::in(segl_I), ndb::in(segr_I), ndb::in(tan_I), ndb::in(supp_J), ndb::in(polys_J),
            ndb::in(segl_J), ndb::in(segr_J), ndb::in(tan_J), a_squared, k, omega, eps_, mu_,
            max_d, ndb::in(gl_t), ndb::in(gl_w), ndb::in(group_I), ndb::in(group_J), a_ek, eps_t,
            phi_c0, phi_c1, cancel_flag, reference));
}

static py::array_t<std::complex<double>>
bspline_assemble_offedge_block_refl_ek_tiered(
    py::array_t<int64_t, py::array::c_style | py::array::forcecast> supp_I,
    py::array_t<double, py::array::c_style | py::array::forcecast> polys_I,
    py::array_t<double, py::array::c_style | py::array::forcecast> segl_I,
    py::array_t<double, py::array::c_style | py::array::forcecast> segr_I,
    py::array_t<double, py::array::c_style | py::array::forcecast> tan_I,
    py::array_t<int64_t, py::array::c_style | py::array::forcecast> supp_J,
    py::array_t<double, py::array::c_style | py::array::forcecast> polys_J,
    py::array_t<double, py::array::c_style | py::array::forcecast> segl_J,
    py::array_t<double, py::array::c_style | py::array::forcecast> segr_J,
    py::array_t<double, py::array::c_style | py::array::forcecast> tan_J,
    double a_squared,
    double k,
    double omega,
    double eps_,
    double mu_,
    int max_d,
    py::array_t<double, py::array::c_style | py::array::forcecast> tier_t,
    py::array_t<double, py::array::c_style | py::array::forcecast> tier_w,
    py::array_t<int64_t, py::array::c_style | py::array::forcecast> tier_n_qp,
    py::array_t<double, py::array::c_style | py::array::forcecast> tier_ratio,
    py::array_t<int64_t, py::array::c_style | py::array::forcecast> group_I,
    py::array_t<int64_t, py::array::c_style | py::array::forcecast> group_J,
    double a_ek,
    std::complex<double> eps_t,
    std::complex<double> phi_c0,
    std::complex<double> phi_c1,
    py::array_t<uint8_t, py::array::c_style | py::array::forcecast> phase_ok_I,
    py::array_t<uint8_t, py::array::c_style | py::array::forcecast> phase_ok_J,
    int64_t limited_below,
    uintptr_t cancel_flag,
    bool reference) {
    ndb::NumpyHost host;
    mw::nd::HostScope scope(host);
    return ndb::out(
        mw::bspline::bspline_assemble_offedge_block_refl_ek_tiered(ndb::in(supp_I),
            ndb::in(polys_I), ndb::in(segl_I), ndb::in(segr_I), ndb::in(tan_I), ndb::in(supp_J),
            ndb::in(polys_J), ndb::in(segl_J), ndb::in(segr_J), ndb::in(tan_J), a_squared, k,
            omega, eps_, mu_, max_d, ndb::in(tier_t), ndb::in(tier_w), ndb::in(tier_n_qp),
            ndb::in(tier_ratio), ndb::in(group_I), ndb::in(group_J), a_ek, eps_t, phi_c0, phi_c1,
            ndb::in(phase_ok_I), ndb::in(phase_ok_J), limited_below, cancel_flag, reference));
}

static std::tuple<py::array_t<std::complex<double>>, py::array_t<std::complex<double>>, py::array_t<std::complex<double>>>
assemble_Z_enrich(
    py::array_t<int64_t, py::array::c_style | py::array::forcecast> spec_seg,
    py::array_t<int64_t, py::array::c_style | py::array::forcecast> spec_origin,
    py::array_t<double, py::array::c_style | py::array::forcecast> seg_l,
    py::array_t<double, py::array::c_style | py::array::forcecast> seg_r,
    py::array_t<double, py::array::c_style | py::array::forcecast> h_per_seg,
    py::array_t<double, py::array::c_style | py::array::forcecast> tangents,
    py::array_t<int64_t, py::array::c_style | py::array::forcecast> supp_seg_poly,
    py::array_t<double, py::array::c_style | py::array::forcecast> polys_poly,
    double a_squared,
    double k,
    double omega,
    double eps_,
    double mu_,
    py::array_t<double, py::array::c_style | py::array::forcecast> gl_t01,
    py::array_t<double, py::array::c_style | py::array::forcecast> gl_w01,
    py::array_t<double, py::array::c_style | py::array::forcecast> proj_coeffs) {
    ndb::NumpyHost host;
    mw::nd::HostScope scope(host);
    return ndb::out_tuple(
        mw::bspline::assemble_Z_enrich(ndb::in(spec_seg), ndb::in(spec_origin), ndb::in(seg_l),
            ndb::in(seg_r), ndb::in(h_per_seg), ndb::in(tangents), ndb::in(supp_seg_poly),
            ndb::in(polys_poly), a_squared, k, omega, eps_, mu_, ndb::in(gl_t01), ndb::in(gl_w01),
            ndb::in(proj_coeffs)));
}

void register_bspline(py::module_ &m) {

    // Read by momwire._accel so the Python routing guard cannot drift from the
    // kernels' real ceiling (momwire#769).
    m.attr("BSPLINE_MAX_N_QP") = py::int_(BSPLINE_MAX_N_QP);
    m.attr("BSPLINE_SAME_EDGE_MAX_N_QP") = py::int_(BSPLINE_SAME_EDGE_MAX_N_QP);

    // Same contract as the two above, for the far series' shared domain:
    // exported so a test can pin it to `_bspline_static_moments.MAX_D` rather
    // than a comment claiming they match (momwire#999).
    m.attr("BSPLINE_FAR_MAX_P") = py::int_(mw::bspline::FAR_MAX_P);
    m.attr("BSPLINE_MOMENT_MAX_D") = py::int_(mw::bspline::MOMENT_MAX_D);

    m.def("bspline_j_static_moment", &bspline_j_static_moment_py,
          "One same-edge static moment, dispatching near/far exactly as the "
          "Toeplitz builders do -- the C++ twin of "
          "_bspline_static_far.J_static_stable. Exposed for the cross-lane "
          "gate, which otherwise cannot reach p or q = 3.",
          py::arg("p"), py::arg("q"), py::arg("alpha"), py::arg("beta"),
          py::arg("A"), py::arg("B"), py::arg("a"));

    m.def("bspline_d_ek_moment", &bspline_d_ek_moment_py,
          "One same-edge EXTENDED-kernel static correction -- the C++ twin of "
          "_bspline_ek_moments.D_ek_moment. No far branch: D_ek has only the "
          "closed form, unlike J. Same reason for existing as the twin above.",
          py::arg("p"), py::arg("q"), py::arg("alpha"), py::arg("beta"),
          py::arg("A"), py::arg("B"), py::arg("a"));

    m.def("bspline_j_static_far", &bspline_j_static_far_py,
          "One same-edge static moment by the centred multipole series -- the "
          "C++ twin of _bspline_static_far.J_static_far, same truncation "
          "(BSPLINE_FAR_TERMS) and same domain guard. Correct only where "
          "bspline_far_ratio <= BSPLINE_FAR_RATIO; the caller checks, exactly "
          "as the numpy twin's docstring says. Exposed for the cross-lane "
          "gate, which otherwise has no path to p = 3.",
          py::arg("p"), py::arg("q"), py::arg("alpha"), py::arg("beta"),
          py::arg("A"), py::arg("B"), py::arg("a"));

    m.def("seg_seg_reg_moments_bspline_swept",
          &seg_seg_reg_moments_bspline_swept,
          "Streaming swept reg-moment kernel for the B-spline Galerkin MoM. "
          "From the precomputed pair-distance table R (N*n_qp, N*n_qp) and "
          "weight-folded local-coordinate powers wu_pow (n_d, N, n_qp), "
          "compute J[k,p,P,i,j] = sum_{q,r} wu_pow[p,i,q] (exp(-jkR)-1)/(4 pi "
          "R) wu_pow[P,j,r] for every k. Returns (n_k, n_d, n_d, N, N) "
          "complex — the streaming C++ replacement for the numpy einsum in "
          "_seg_seg_reg_moments_from_geometry_swept.",
          py::arg("R"), py::arg("wu_pow"), py::arg("k_array"));
    m.def("seg_seg_reg_moments_bspline_swept_ek",
          &seg_seg_reg_moments_bspline_swept_ek,
          "Extended-thin-wire-kernel twin of seg_seg_reg_moments_bspline_swept "
          "(momwire#270 unit 1). Same (R, wu_pow, k_array) contract and the "
          "same (n_k, n_d, n_d, N, N) output, but the smooth remainder is "
          "[(exp(-jkR) - 1)*fac + extra] / (4 pi R) with NEC Eq 89's coaxial "
          "equal-radius factor fac = 1 + T1*C2 - T2*C1 about the EK radius "
          "`a_ek` and extra = fac - fac_static — a literal transcription of "
          "_bspline_kernels._ek_reg_kernel, whose static half rides in "
          "seg_seg_static_moments_bspline_uniform_ek. Same-edge blocks are "
          "eligible in their entirety, so there are no per-pair group labels.",
          py::arg("R"), py::arg("wu_pow"), py::arg("k_array"), py::arg("a_ek"));
    m.def("seg_seg_reg_moments_bspline_swept_window",
          &seg_seg_reg_moments_bspline_swept_window,
          "Rectangular twin of seg_seg_reg_moments_bspline_swept "
          "(momwire#968): ONE OBSERVER WINDOW of a same-edge block. R is "
          "(n_row*n_qp, n_col*n_qp), wu_row (n_d, n_row, n_qp) carries the "
          "window's segments and wu_col (n_d, n_col, n_qp) the whole edge's; "
          "the result is (n_k, n_d, n_d, n_row, n_col). Same kernel body as "
          "the square call, which delegates to it with one array on both "
          "axes — the observer and source axes were always independent.",
          py::arg("R"), py::arg("wu_row"), py::arg("wu_col"),
          py::arg("k_array"));
    m.def("seg_seg_reg_moments_bspline_swept_ek_window",
          &seg_seg_reg_moments_bspline_swept_ek_window,
          "Rectangular twin of seg_seg_reg_moments_bspline_swept_ek "
          "(momwire#968). Same windowing contract as "
          "seg_seg_reg_moments_bspline_swept_window, same EK remainder as the "
          "square call.",
          py::arg("R"), py::arg("wu_row"), py::arg("wu_col"),
          py::arg("k_array"), py::arg("a_ek"));
    m.def("seg_seg_full_moments_bspline", &seg_seg_full_moments_bspline,
          "Single-k full-kernel polynomial moment integrals for the B-spline "
          "Galerkin MoM. Returns J of shape (max_d+1, max_d+1, N_i, N_j) "
          "complex. Templated on max_d at compile time; currently "
          "instantiated for max_d in {1, 2}.",
          py::arg("seg_l_i"), py::arg("seg_r_i"),
          py::arg("seg_l_j"), py::arg("seg_r_j"),
          py::arg("a_squared"), py::arg("k"),
          py::arg("max_d"),
          py::arg("gl_t"), py::arg("gl_w"),
          py::arg("reference") = false);
    m.def("seg_seg_full_moments_bspline_cplx",
          &seg_seg_full_moments_bspline_cplx,
          "In-medium (complex-k) twin of seg_seg_full_moments_bspline "
          "(momwire#778). Same contract and same (max_d+1, max_d+1, N_i, N_j) "
          "output, with k a complex wavenumber k_re + j*k_im, Im k <= 0. The "
          "continuation is exp(-jkR) = exp(k_im*R) * exp(-j*k_re*R): one extra "
          "real exp() per quadrature point, all loops still real-lane SIMD.",
          py::arg("seg_l_i"), py::arg("seg_r_i"),
          py::arg("seg_l_j"), py::arg("seg_r_j"),
          py::arg("a_squared"), py::arg("k"),
          py::arg("max_d"),
          py::arg("gl_t"), py::arg("gl_w"),
          py::arg("reference") = false);
    m.def("seg_seg_full_moments_bspline_tiered",
          &seg_seg_full_moments_bspline_tiered,
          "Distance-adaptive twin of seg_seg_full_moments_bspline "
          "(momwire#906). Same contract and output, but the Gauss-Legendre "
          "rule is a LADDER: tier 0 (the first tier_n_qp[0] entries of "
          "tier_t / tier_w) is the base order, and tier t >= 1 serves every "
          "pair whose centre distance over the longer segment is >= "
          "tier_ratio[t] (thresholds strictly ascending). One tier reproduces "
          "the plain entry bit for bit.",
          py::arg("seg_l_i"), py::arg("seg_r_i"),
          py::arg("seg_l_j"), py::arg("seg_r_j"),
          py::arg("a_squared"), py::arg("k"),
          py::arg("max_d"),
          py::arg("tier_t"), py::arg("tier_w"),
          py::arg("tier_n_qp"), py::arg("tier_ratio"),
          py::arg("reference") = false);
    m.def("bspline_pair_tiers", &bspline_pair_tiers,
          "The pair-order ladder's tier of every pair (i, j) of a block, "
          "(N_i, N_j) int64 (momwire#1772): the table the tiered off-edge "
          "kernels route by, for the tests.",
          py::arg("seg_l_i"), py::arg("seg_r_i"), py::arg("seg_l_j"), py::arg("seg_r_j"),
          py::arg("tier_t"), py::arg("tier_w"), py::arg("tier_n_qp"), py::arg("tier_ratio"));
    m.def("seg_seg_full_moments_bspline_cplx_tiered",
          &seg_seg_full_moments_bspline_cplx_tiered,
          "In-medium (complex-k) twin of seg_seg_full_moments_bspline_tiered "
          "(momwire#906): the ladder contract above with the #778 complex-k "
          "continuation, Im k <= 0.",
          py::arg("seg_l_i"), py::arg("seg_r_i"),
          py::arg("seg_l_j"), py::arg("seg_r_j"),
          py::arg("a_squared"), py::arg("k"),
          py::arg("max_d"),
          py::arg("tier_t"), py::arg("tier_w"),
          py::arg("tier_n_qp"), py::arg("tier_ratio"),
          py::arg("reference") = false);
    m.def("seg_seg_full_moments_sinusoidal_tiered",
          &seg_seg_full_moments_sinusoidal_tiered,
          "The tiered pair moments on the sinusoidal Galerkin basis's folded "
          "shape set {1, sin k xi, cos k xi - 1} (momwire#1354): the B-spline "
          "ladder contract with the shapes evaluated at the nodes in place of "
          "the monomials, (3, 3, N_i, N_j). k complex, Im k <= 0; a real k "
          "takes the real-shape instantiation.",
          py::arg("seg_l_i"), py::arg("seg_r_i"),
          py::arg("seg_l_j"), py::arg("seg_r_j"),
          py::arg("a_squared"), py::arg("k"),
          py::arg("tier_t"), py::arg("tier_w"),
          py::arg("tier_n_qp"), py::arg("tier_ratio"),
          py::arg("reference") = false);
    m.def("seg_seg_full_moments_sinusoidal_tiered_ek",
          &seg_seg_full_moments_sinusoidal_tiered_ek,
          "The extended-kernel twin of seg_seg_full_moments_sinusoidal_tiered "
          "(momwire#1362): NEC Eq 89's coaxial factor on G for the pairs whose "
          "group labels match. Real k only.",
          py::arg("seg_l_i"), py::arg("seg_r_i"),
          py::arg("seg_l_j"), py::arg("seg_r_j"),
          py::arg("a_squared"), py::arg("k"),
          py::arg("tier_t"), py::arg("tier_w"),
          py::arg("tier_n_qp"), py::arg("tier_ratio"),
          py::arg("group_i"), py::arg("group_j"), py::arg("a_ek"),
          py::arg("b_j") = py::none());
    m.def("assemble_Z_sinusoidal_windowed", &assemble_Z_sinusoidal_windowed,
          "Accumulate one window of sinusoidal pair moments into Z "
          "(momwire#1354): the windowed B-spline assembler with the basis as "
          "CSR-by-segment coefficient tables (coef, dcoef) on the folded "
          "shape set, per-pair complex weights optional. Threads own basis "
          "rows.",
          py::arg("J"), py::arg("rows_seg"), py::arg("cols_seg"),
          py::arg("starts"), py::arg("jbasis"), py::arg("coef"), py::arg("dcoef"),
          py::arg("tangents_rows"), py::arg("tangents_cols"),
          py::arg("c_a"), py::arg("c_phi"),
          py::arg("w_a") = py::none(), py::arg("w_phi") = py::none(),
          py::arg("Z"), py::arg("cancel_flag") = 0);
    m.def("field_pair_moments_sinusoidal", &field_pair_moments_sinusoidal,
          "Field-form pair moments sum_{q,r} W_obs[p,i,q] F[iq,jr] W_src[P,j,r] "
          "with complex shape weights, (3, 3, n_obs, n_src) (momwire#1354).",
          py::arg("F"), py::arg("W_obs"), py::arg("W_src"));
    m.def("parallel_pair_moments_sinusoidal", &parallel_pair_moments_sinusoidal,
          "The sinusoidal fill's parallel-pair reduction (momwire#1354): "
          "(3, 3, n) moments of n parallel segment pairs by the sinh-substituted "
          "separation integral, a pair per thread.",
          py::arg("c_i"), py::arg("t_i"), py::arg("h_i"), py::arg("c_j"), py::arg("t_j"),
          py::arg("h_j"), py::arg("a2"), py::arg("k"), py::arg("gt"), py::arg("gw"),
          py::arg("gx"), py::arg("gwx"), py::arg("a_ek") = py::none(),
          py::arg("b_ek") = py::none());
    m.def("seg_seg_full_moments_bspline_swept",
          &seg_seg_full_moments_bspline_swept,
          "Batched (swept-k) off-edge full-kernel polynomial moments for the "
          "B-spline Galerkin MoM. The per-(i,j) geometry (R table, moment "
          "weights) is built once and reused across k_array; only exp(-jkR) "
          "varies per frequency. Returns (n_k, max_d+1, max_d+1, N_i, N_j) "
          "complex; per-pair R tables are reused across the k axis.",
          py::arg("seg_l_i"), py::arg("seg_r_i"),
          py::arg("seg_l_j"), py::arg("seg_r_j"),
          py::arg("a_squared"), py::arg("k_array"),
          py::arg("max_d"),
          py::arg("gl_t"), py::arg("gl_w"));
    m.def("seg_seg_full_moments_bspline_ek", &seg_seg_full_moments_bspline_ek,
          "Extended-thin-wire-kernel twin of seg_seg_full_moments_bspline "
          "(momwire#270 unit 2). Same (seg_l_i, seg_r_i, seg_l_j, seg_r_j, "
          "a_squared, k, max_d, gl_t, gl_w) contract, plus per-segment "
          "coaxial-and-equal-radius group labels group_i (N_i,) / group_j "
          "(N_j,) int64 and the plain (unsquared) EK radius a_ek. A pair "
          "(i, j) is eligible iff group_i[i] == group_j[j] >= 0, evaluated "
          "once per segment pair and applied to every quadrature sub-pair "
          "inside it; G = exp(-jkR)/(4 pi R) is multiplied by NEC Eq 89's "
          "coaxial factor fac = 1 + T1*C2 - T2*C1 on eligible pairs and left "
          "alone otherwise — a literal transcription of "
          "_bspline_kernels._seg_seg_full_moments_offedge's `ek is not "
          "None` branch, evaluated eagerly instead of via np.where. "
          "reference=True walks every pair alone where the AVX2 lane path "
          "is compiled (momwire#1362), the gate that path is held to.",
          py::arg("seg_l_i"), py::arg("seg_r_i"),
          py::arg("seg_l_j"), py::arg("seg_r_j"),
          py::arg("a_squared"), py::arg("k"),
          py::arg("max_d"),
          py::arg("gl_t"), py::arg("gl_w"),
          py::arg("group_i"), py::arg("group_j"), py::arg("a_ek"),
          py::arg("reference") = false, py::arg("a_ek_src") = -1.0);
    m.def("seg_seg_full_moments_bspline_ek_tiered",
          &seg_seg_full_moments_bspline_ek_tiered,
          "Distance-adaptive twin of seg_seg_full_moments_bspline_ek "
          "(momwire#1362): the pair-order ladder contract of "
          "seg_seg_full_moments_bspline_tiered (tier 0 the base order, tier "
          "t >= 1 for pairs at centre distance over the longer segment >= "
          "tier_ratio[t]) with the EK group labels and radius. The coaxial "
          "factor is applied at whatever order a pair runs; one tier "
          "reproduces seg_seg_full_moments_bspline_ek bit for bit.",
          py::arg("seg_l_i"), py::arg("seg_r_i"),
          py::arg("seg_l_j"), py::arg("seg_r_j"),
          py::arg("a_squared"), py::arg("k"),
          py::arg("max_d"),
          py::arg("tier_t"), py::arg("tier_w"),
          py::arg("tier_n_qp"), py::arg("tier_ratio"),
          py::arg("group_i"), py::arg("group_j"), py::arg("a_ek"),
          py::arg("reference") = false, py::arg("a_ek_src") = -1.0);
    m.def("seg_seg_full_moments_bspline_swept_ek",
          &seg_seg_full_moments_bspline_swept_ek,
          "Batched (swept-k) twin of seg_seg_full_moments_bspline_ek "
          "(momwire#270 unit 2). Same (max_d+1, max_d+1, N_i, N_j) "
          "per-(i,j) geometry hoist as seg_seg_full_moments_bspline_swept, "
          "plus the EK eligibility mask and the k-independent T1/T2 halves "
          "of the coaxial factor hoisted out of the k loop the same way the "
          "same-edge swept twin hoists them. Returns (n_k, max_d+1, max_d+1, "
          "N_i, N_j) complex. reference=True walks every pair alone where the "
          "AVX2 lane path is compiled (momwire#1362), the gate that path is "
          "held to.",
          py::arg("seg_l_i"), py::arg("seg_r_i"),
          py::arg("seg_l_j"), py::arg("seg_r_j"),
          py::arg("a_squared"), py::arg("k_array"),
          py::arg("max_d"),
          py::arg("gl_t"), py::arg("gl_w"),
          py::arg("group_i"), py::arg("group_j"), py::arg("a_ek"),
          py::arg("reference") = false, py::arg("a_ek_src") = -1.0);
    m.def("seg_seg_full_moments_bspline_swept_ek_tiered",
          &seg_seg_full_moments_bspline_swept_ek_tiered,
          "Distance-adaptive twin of seg_seg_full_moments_bspline_swept_ek "
          "(momwire#1362): the pair-order ladder contract of "
          "seg_seg_full_moments_bspline_ek_tiered over a k_array. One tier "
          "reproduces seg_seg_full_moments_bspline_swept_ek bit for bit; "
          "reference=True walks every pair alone where the AVX2 lane path is "
          "compiled.",
          py::arg("seg_l_i"), py::arg("seg_r_i"),
          py::arg("seg_l_j"), py::arg("seg_r_j"),
          py::arg("a_squared"), py::arg("k_array"),
          py::arg("max_d"),
          py::arg("tier_t"), py::arg("tier_w"),
          py::arg("tier_n_qp"), py::arg("tier_ratio"),
          py::arg("group_i"), py::arg("group_j"), py::arg("a_ek"),
          py::arg("reference") = false, py::arg("a_ek_src") = -1.0);
    m.def("seg_seg_static_moments_bspline_uniform",
          &seg_seg_static_moments_bspline_uniform,
          "Closed-form same-edge static-kernel polynomial moments J_pq for a "
          "uniform-h edge with N segments. Uses Toeplitz structure (2N-1 "
          "unique values per moment) and inlined sympy-derived closed forms. "
          "Returns J_static of shape (max_d+1, max_d+1, N, N), with the "
          "1/(4π) prefactor folded in.",
          py::arg("h"), py::arg("a"), py::arg("N"), py::arg("max_d"),
          py::arg("cancel_flag") = 0,
          // Test seam (momwire#1158): see the probe comment in the impl.
          py::arg("poll_probe") = 0);
    m.def("seg_seg_static_moments_bspline_uniform_ek",
          &seg_seg_static_moments_bspline_uniform_ek,
          "Extended-thin-wire-kernel twin of "
          "seg_seg_static_moments_bspline_uniform (momwire#270 unit 1). Adds "
          "the generated closed-form correction D_pq^EK — the moments of "
          "-a_ek^2/(2R^3) + 3 a_ek^4/(4R^5), the k -> 0 limit of Eq 89's "
          "coaxial factor minus 1 — to each J_pq before the 1/(4 pi) "
          "prefactor. D is translation-invariant along the edge exactly as J "
          "is, so it rides the same 2N-1 Toeplitz table. `a_ek` is separate "
          "from the regularization radius `a` because _EK.a may override it; "
          "on every eligible pair they are equal.",
          py::arg("h"), py::arg("a"), py::arg("N"), py::arg("max_d"),
          py::arg("a_ek"), py::arg("cancel_flag") = 0);
    m.def("toeplitz_gather_rows", &toeplitz_gather_rows,
          py::arg("table"), py::arg("N"), py::arg("r0"), py::arg("r1"),
          "Rows [r0, r1) of the (nd, nd, N, N) Toeplitz gather of an (nd, nd, "
          "2N-1) table, in parallel; a copy, bit-identical to numpy's "
          "(momwire#1441).");
    m.def("reg_distance_table", &reg_distance_table,
          py::arg("s_row"), py::arg("s_col"), py::arg("a2"),
          "sqrt((s_row[:, None] - s_col[None, :])**2 + a2), numpy's in-place "
          "order, in parallel; bit-identical (momwire#1441).");
    m.def("seg_seg_reg_moments_bspline_cplx", &seg_seg_reg_moments_bspline_cplx,
          py::arg("R"), py::arg("wu_pow"), py::arg("k"),
          "_seg_seg_reg_moments_from_geometry at a complex k on the square "
          "geometry: numpy's expm1_neg_jkR / (4 pi R) and its einsum, bit for "
          "bit, in parallel (momwire#1752).");
    m.def("same_edge_corr_combine", &same_edge_corr_combine,
          py::arg("A_st"), py::arg("A_reg"), py::arg("J"),
          "(A_st + A_reg) - J with numpy's float-to-complex promotion, in "
          "parallel; bit-identical (momwire#1441).");
    m.def("seg_seg_static_moments_bspline_uniform_table",
          &seg_seg_static_moments_bspline_uniform_table,
          "The 2N-1 Toeplitz TABLE behind "
          "seg_seg_static_moments_bspline_uniform (momwire#968), shape "
          "(max_d+1, max_d+1, 2N-1) with the 1/(4 pi) prefactor folded in. The "
          "square call gathers it into (max_d+1, max_d+1, N, N); a caller "
          "streaming one edge in observer windows gathers whatever rows it "
          "wants instead, and gets the square answer's sub-block bit for bit. "
          "Exposed rather than a row-window gather because the TABLE is the "
          "work: rebuilding it per window measured 3.7x slower across an edge.",
          py::arg("h"), py::arg("a"), py::arg("N"), py::arg("max_d"));
    m.def("seg_seg_static_moments_bspline_uniform_ek_table",
          &seg_seg_static_moments_bspline_uniform_ek_table,
          "EK twin of seg_seg_static_moments_bspline_uniform_table "
          "(momwire#968): each entry carries the D_pq^EK correction the square "
          "EK call adds.",
          py::arg("h"), py::arg("a"), py::arg("N"), py::arg("max_d"),
          py::arg("a_ek"));
    // momwire#1132: the four windowed assemblers take `row_of`, a ROW-COMPACT
    // Z target. Its own flag: a .so built before this answers `row_of=` with a
    // TypeError, so the sector route's compact fill gates on THIS.
    m.attr("windowed_row_of_1132") = true;
    // momwire#1290: whether the windowed assemblers carry the AVX2 lane
    // kernel; either way they take `reference`, which runs the per-entry
    // loop that kernel is gated against.
    m.attr("windowed_lanes_1290") = mw::bspline::WINDOWED_LANES_1290;
    // momwire#1290: whether the off-edge moment kernel (plain, cplx and both
    // tiered entries) carries the AVX2 lane path; either way they take
    // `reference`, which walks every pair alone.
    m.attr("offedge_lanes_1290") = mw::bspline::OFFEDGE_LANES_1290;
    m.def("assemble_Z_bspline_weighted_windowed", &assemble_Z_bspline_weighted_windowed,
          "Weighted + scaled windowed accumulator: like "
          "assemble_Z_bspline_windowed but with complex per-pair weights "
          "wA_win / wPhi_win on the A and charge terms and a complex scale "
          "on the window's contribution before the +=. Serves the chunked "
          "ground-image builds (PEC mirror-dot, Fresnel refl-coef, "
          "Sommerfeld constant-C2) with scale = -1 for the Z -= image "
          "convention. The weights are WINDOWS of shape (i1-i0, j1-j0) "
          "aligned with J_chunk's trailing axes, not global (N, N) tables: "
          "lookups are window-relative, so the caller never keeps two "
          "global complex (N, N) tables alive across the fill (issue #323). "
          "max_d inferred from support_seg. On an AVX2+FMA x86 build a lane "
          "kernel fills eight entries a step; reference=True runs the "
          "per-entry loop instead, the same bits.",
          py::arg("J_chunk"), py::arg("support_seg"),
          py::arg("polys"), py::arg("wA_win"), py::arg("wPhi_win"),
          py::arg("m_idx"), py::arg("n_idx"),
          py::arg("i0"), py::arg("i1"), py::arg("j0"), py::arg("j1"),
          py::arg("omega"), py::arg("eps_"), py::arg("mu_"),
          py::arg("scale"), py::arg("Z"), py::arg("cancel_flag") = 0,
          py::arg("row_of") = py::none(),
          py::arg("reference") = false);
    m.def("assemble_Z_bspline_windowed_cplx_eps", &assemble_Z_bspline_windowed_cplx_eps,
          "In-medium twin of assemble_Z_bspline_windowed (momwire#915): the "
          "same window contract with eps a COMPLEX permittivity, and the same "
          "`reference` switch (momwire#1290).",
          py::arg("J_chunk"), py::arg("support_seg"), py::arg("polys"),
          py::arg("tangents"), py::arg("m_idx"), py::arg("n_idx"),
          py::arg("i0"), py::arg("i1"), py::arg("j0"), py::arg("j1"),
          py::arg("omega"), py::arg("eps"), py::arg("mu"),
          py::arg("Z"), py::arg("cancel_flag") = 0,
          py::arg("row_of") = py::none(),
          py::arg("reference") = false);
    m.def("assemble_Z_bspline_weighted_windowed_cplx_eps",
          &assemble_Z_bspline_weighted_windowed_cplx_eps,
          "In-medium twin of assemble_Z_bspline_weighted_windowed "
          "(momwire#915): the same window contract with eps a COMPLEX "
          "permittivity.",
          py::arg("J_chunk"), py::arg("support_seg"), py::arg("polys"),
          py::arg("wA_win"), py::arg("wPhi_win"), py::arg("m_idx"), py::arg("n_idx"),
          py::arg("i0"), py::arg("i1"), py::arg("j0"), py::arg("j1"),
          py::arg("omega"), py::arg("eps"), py::arg("mu"), py::arg("scale"),
          py::arg("Z"), py::arg("cancel_flag") = 0,
          py::arg("row_of") = py::none(),
          py::arg("reference") = false);
    m.def("assemble_Z_bspline_windowed", &assemble_Z_bspline_windowed,
          "Accumulate one rectangular segment window's contribution into a "
          "caller-provided Z from a chunked moment tensor J_chunk of shape "
          "(D+1, D+1, i1-i0, j1-j0). m_idx / n_idx select the basis rows / "
          "cols with support in the window; the (zA, zPhi) -> Z mixing is "
          "linear so per-window accumulation equals the all-at-once "
          "assembly. Lets the dense build skip the full (D+1, D+1, N, N) "
          "tensor (issue #136). `tangents` is the per-segment unit tangent "
          "table, (n_segs, 3): the pair tangent dot is formed here from two "
          "rows rather than read out of an (N, N) table the caller would "
          "have to keep alive across the whole fill (issue #318). max_d is "
          "inferred from support_seg. On the AVX2 build the fill runs eight "
          "entries at a time over cache tiles (momwire#1290); "
          "reference=True runs the per-entry loop instead, the same bits.",
          py::arg("J_chunk"), py::arg("support_seg"),
          py::arg("polys"), py::arg("tangents"),
          py::arg("m_idx"), py::arg("n_idx"),
          py::arg("i0"), py::arg("i1"), py::arg("j0"), py::arg("j1"),
          py::arg("omega"), py::arg("eps_"), py::arg("mu_"),
          py::arg("Z"), py::arg("cancel_flag") = 0,
          py::arg("row_of") = py::none(),
          py::arg("reference") = false);
    m.def("assemble_Z_bspline", &assemble_Z_bspline,
          "Assemble the (n_basis, n_basis) Z matrix from the polynomial-"
          "moment tensor J, per-basis polynomial coefficients, support-segment "
          "map, and tangent-dot table. Templated on max_d at compile time; "
          "currently instantiated for max_d in {1, 2}. Single-k.",
          py::arg("J"), py::arg("support_seg"),
          py::arg("polys"), py::arg("td_all"),
          py::arg("omega"), py::arg("eps"), py::arg("mu"),
          py::arg("max_d"),
          py::arg("cancel_flag") = 0);
    m.def("assemble_Z_bspline_weighted", &assemble_Z_bspline_weighted,
          "Weighted assemble_Z_bspline for the reflection-coefficient "
          "finite ground: complex per-segment-pair weight tables on both "
          "terms — wA_all (Fresnel dyad tangent table) on the A term, "
          "wPhi_all (image-charge weight) on the Φ term. Templated on "
          "max_d in {1, 2}; single-k.",
          py::arg("J"), py::arg("support_seg"),
          py::arg("polys"), py::arg("wA_all"), py::arg("wPhi_all"),
          py::arg("omega"), py::arg("eps"), py::arg("mu"),
          py::arg("max_d"),
          py::arg("cancel_flag") = 0);
    m.def("assemble_Z_bspline_cplx_eps", &assemble_Z_bspline_cplx_eps,
          "In-medium twin of assemble_Z_bspline (momwire#910): the same "
          "contract with eps a COMPLEX permittivity, for the buried fill's "
          "charge term divided by eps0 * eps~.",
          py::arg("J"), py::arg("support_seg"),
          py::arg("polys"), py::arg("td_all"),
          py::arg("omega"), py::arg("eps"), py::arg("mu"),
          py::arg("max_d"),
          py::arg("cancel_flag") = 0);
    m.def("assemble_Z_bspline_weighted_cplx_eps",
          &assemble_Z_bspline_weighted_cplx_eps,
          "In-medium twin of assemble_Z_bspline_weighted (momwire#910): the "
          "same contract with eps a COMPLEX permittivity, for the buried "
          "image block.",
          py::arg("J"), py::arg("support_seg"),
          py::arg("polys"), py::arg("wA_all"), py::arg("wPhi_all"),
          py::arg("omega"), py::arg("eps"), py::arg("mu"),
          py::arg("max_d"),
          py::arg("cancel_flag") = 0);
    m.def("assemble_Z_bspline_swept", &assemble_Z_bspline_swept,
          "Batched (swept-k) assemble: J is (n_k, max_d+1, max_d+1, N, N) and "
          "omega is an array; the basis tables are k-independent. "
          "tangents_row / tangents_col are (n_segs, 3) per-segment tangent "
          "tables — the kernel forms each pair's dot in-kernel rather than "
          "reading an (N, N) table (issue #333); pass (tangents, tangents) "
          "for free space and (tangents, mirrored_tangents) for the PEC "
          "image term. Returns (n_k, n_basis, n_basis) — the bspline analog "
          "of triangular's batched assemble_Z.",
          py::arg("J"), py::arg("support_seg"),
          py::arg("polys"), py::arg("tangents_row"), py::arg("tangents_col"),
          py::arg("omega_array"), py::arg("eps"), py::arg("mu"),
          py::arg("max_d"));
    m.def("bspline_assemble_offedge_block", &bspline_assemble_offedge_block,
          "Fused off-edge Z[I, J] block assembly for the H-matrix / ACA "
          "solver: quadratures the a²-regularised full-kernel moments and "
          "performs the EFIE Galerkin combine in one pass, with no "
          "intermediate moment tensor. Segments are the per-block union "
          "referenced by the I/J bases; support_*_local index into them. "
          "Templated on max_d in {1, 2}; single-k.",
          py::arg("supp_I"), py::arg("polys_I"), py::arg("segl_I"),
          py::arg("segr_I"), py::arg("tan_I"),
          py::arg("supp_J"), py::arg("polys_J"), py::arg("segl_J"),
          py::arg("segr_J"), py::arg("tan_J"),
          py::arg("a_squared"), py::arg("k"), py::arg("omega"),
          py::arg("eps"), py::arg("mu"), py::arg("max_d"),
          py::arg("gl_t"), py::arg("gl_w"),
          py::arg("cancel_flag") = 0);
    m.def("bspline_assemble_offedge_block_refl", &bspline_assemble_offedge_block_refl,
          "Reflection-coefficient finite-ground image variant of "
          "bspline_assemble_offedge_block: J side passed pre-mirrored, the "
          "Fresnel dyad (from eps_t) weights the A term per segment pair and "
          "w_Phi = phi_c0 + phi_c1*rho_v weights the charge term. Templated "
          "on max_d in {1, 2}; single-k.",
          py::arg("supp_I"), py::arg("polys_I"), py::arg("segl_I"),
          py::arg("segr_I"), py::arg("tan_I"),
          py::arg("supp_J"), py::arg("polys_J"), py::arg("segl_J"),
          py::arg("segr_J"), py::arg("tan_J"),
          py::arg("a_squared"), py::arg("k"), py::arg("omega"),
          py::arg("eps"), py::arg("mu"), py::arg("max_d"),
          py::arg("gl_t"), py::arg("gl_w"),
          py::arg("eps_t"), py::arg("phi_c0"), py::arg("phi_c1"),
          py::arg("cancel_flag") = 0);
    m.def("bspline_assemble_offedge_block_ek", &bspline_assemble_offedge_block_ek,
          "Extended-thin-wire-kernel twin of bspline_assemble_offedge_block "
          "(momwire#270 unit 3). Same fused off-edge Z[I, J] block assembly "
          "contract, plus per-segment coaxial-and-equal-radius group labels "
          "group_I (nSegI,) / group_J (nSegJ,) int64 over the SAME segment "
          "unions segl_I/segl_J index, and the plain (unsquared) EK radius "
          "a_ek. A segment pair (smi, snj) is eligible iff "
          "group_I[smi] == group_J[snj] >= 0, evaluated once per basis-pair "
          "wing and applied to every quadrature sub-pair inside it before "
          "the Galerkin combine — the fused-assembler analog of "
          "seg_seg_full_moments_bspline_ek. Serves free space and the "
          "mirror_J PEC-ground image (J-side positions/tangents pre-mirrored "
          "by the caller); the reflection-coefficient image is the separate "
          "bspline_assemble_offedge_block_refl_ek. Templated on "
          "max_d in {1, 2}; single-k.",
          py::arg("supp_I"), py::arg("polys_I"), py::arg("segl_I"),
          py::arg("segr_I"), py::arg("tan_I"),
          py::arg("supp_J"), py::arg("polys_J"), py::arg("segl_J"),
          py::arg("segr_J"), py::arg("tan_J"),
          py::arg("a_squared"), py::arg("k"), py::arg("omega"),
          py::arg("eps"), py::arg("mu"), py::arg("max_d"),
          py::arg("gl_t"), py::arg("gl_w"),
          py::arg("group_I"), py::arg("group_J"), py::arg("a_ek"),
          py::arg("cancel_flag") = 0, py::arg("reference") = false);
    m.def("bspline_assemble_offedge_block_ek_tiered",
          &bspline_assemble_offedge_block_ek_tiered,
          "Distance-adaptive twin of bspline_assemble_offedge_block_ek "
          "(momwire#1362): the pair-order ladder contract of "
          "seg_seg_full_moments_bspline_tiered, with the per-pair phase guard "
          "answered in the kernel -- a tier of fewer than limited_below points "
          "serves a segment pair only when phase_ok_I[si] and phase_ok_J[sj] "
          "are both set (uint8 per segment of each union; empty = all pass). "
          "One tier reproduces bspline_assemble_offedge_block_ek bit for bit.",
          py::arg("supp_I"), py::arg("polys_I"), py::arg("segl_I"),
          py::arg("segr_I"), py::arg("tan_I"),
          py::arg("supp_J"), py::arg("polys_J"), py::arg("segl_J"),
          py::arg("segr_J"), py::arg("tan_J"),
          py::arg("a_squared"), py::arg("k"), py::arg("omega"),
          py::arg("eps"), py::arg("mu"), py::arg("max_d"),
          py::arg("tier_t"), py::arg("tier_w"),
          py::arg("tier_n_qp"), py::arg("tier_ratio"),
          py::arg("group_I"), py::arg("group_J"), py::arg("a_ek"),
          py::arg("phase_ok_I"), py::arg("phase_ok_J"), py::arg("limited_below"),
          py::arg("cancel_flag") = 0, py::arg("reference") = false);
    m.def("bspline_assemble_offedge_block_refl_ek",
          &bspline_assemble_offedge_block_refl_ek,
          "bspline_assemble_offedge_block_ek and "
          "bspline_assemble_offedge_block_refl composed (momwire#269): the "
          "reflection-coefficient finite-ground image block under the "
          "extended thin-wire kernel. The coaxial factor multiplies G on "
          "eligible segment pairs before the Galerkin contraction; the "
          "Fresnel dyad (from eps_t) and w_Phi = phi_c0 + phi_c1*rho_v "
          "weight the contracted A / charge terms after it. J side passed "
          "pre-mirrored. Templated on max_d in {1, 2}; single-k.",
          py::arg("supp_I"), py::arg("polys_I"), py::arg("segl_I"),
          py::arg("segr_I"), py::arg("tan_I"),
          py::arg("supp_J"), py::arg("polys_J"), py::arg("segl_J"),
          py::arg("segr_J"), py::arg("tan_J"),
          py::arg("a_squared"), py::arg("k"), py::arg("omega"),
          py::arg("eps"), py::arg("mu"), py::arg("max_d"),
          py::arg("gl_t"), py::arg("gl_w"),
          py::arg("group_I"), py::arg("group_J"), py::arg("a_ek"),
          py::arg("eps_t"), py::arg("phi_c0"), py::arg("phi_c1"),
          py::arg("cancel_flag") = 0, py::arg("reference") = false);
    m.def("bspline_assemble_offedge_block_refl_ek_tiered",
          &bspline_assemble_offedge_block_refl_ek_tiered,
          "bspline_assemble_offedge_block_ek_tiered's ladder and phase-guard "
          "contract on the reflection-coefficient image block of "
          "bspline_assemble_offedge_block_refl_ek (momwire#1362). One tier "
          "reproduces that entry bit for bit.",
          py::arg("supp_I"), py::arg("polys_I"), py::arg("segl_I"),
          py::arg("segr_I"), py::arg("tan_I"),
          py::arg("supp_J"), py::arg("polys_J"), py::arg("segl_J"),
          py::arg("segr_J"), py::arg("tan_J"),
          py::arg("a_squared"), py::arg("k"), py::arg("omega"),
          py::arg("eps"), py::arg("mu"), py::arg("max_d"),
          py::arg("tier_t"), py::arg("tier_w"),
          py::arg("tier_n_qp"), py::arg("tier_ratio"),
          py::arg("group_I"), py::arg("group_J"), py::arg("a_ek"),
          py::arg("eps_t"), py::arg("phi_c0"), py::arg("phi_c1"),
          py::arg("phase_ok_I"), py::arg("phase_ok_J"), py::arg("limited_below"),
          py::arg("cancel_flag") = 0, py::arg("reference") = false);
    m.def("assemble_Z_enrich", &assemble_Z_enrich,
          "Assemble (Z_pe, Z_ep, Z_ee) for the stable XFEM singular basis "
          "enrichment at K≥3 junctions. Each enrichment basis is "
          "Φ_sing_stable(t) = t·log(t) − Σ_p proj_coeffs[p]·t^p with "
          "t = u_norm = u/h (origin=0) or 1 − u/h (origin=1), so the "
          "enrichment is L²-orthogonal to the local polynomial space on "
          "each segment. proj_coeffs must have length degree+1 and match "
          "the polys_poly third dim. Z_ep is computed independently from "
          "Z_pe (no .T shortcut). tangents is the (n_segs, 3) per-segment "
          "unit tangent table (#334) — the tangent dot is formed in-kernel "
          "per (m, e)/(e, f) pair rather than reading an (N, N) td_all "
          "table. Single-k.",
          py::arg("spec_seg"), py::arg("spec_origin"),
          py::arg("seg_l"), py::arg("seg_r"),
          py::arg("h_per_seg"), py::arg("tangents"),
          py::arg("supp_seg_poly"), py::arg("polys_poly"),
          py::arg("a_squared"), py::arg("k"),
          py::arg("omega"), py::arg("eps"), py::arg("mu"),
          py::arg("gl_t01"), py::arg("gl_w01"),
          py::arg("proj_coeffs"));
}

