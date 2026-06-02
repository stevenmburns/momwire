#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <complex.h>
#include "math.h"

#include <iostream>
#include <tuple>
#include <vector>

#include "_bspline_static_moments_inline.h"

namespace py = pybind11;

// Ubuntu/glibc <cmath> headers don't carry `omp declare simd` markers for the
// libmvec routines, so GCC's auto-vectorizer can't substitute the vectorized
// `_ZGVdN4v_sin` / `_ZGVdN4v_cos` (AVX2, 4 doubles) inside an `omp simd` loop
// without these explicit declarations. The std::cos / std::sin overloads in
// <cmath> still resolve to these underlying extern-C symbols, so the rest of
// the file's calls pick up the simd-vectorized form for free once the linker
// has libmvec available (-lmvec in setup.py).
#pragma omp declare simd notinbranch simdlen(4)
extern "C" double cos(double);

#pragma omp declare simd notinbranch simdlen(4)
extern "C" double sin(double);

// Batched cross-segment Gauss-Legendre quadrature in 3D.
//
// For each k in k_array, and each (i, j) segment pair, compute:
//   J_pq[k, i, j] = sum_{q, r} w_i_q * u_i_q^p * G(k, R_qr) * w_j_r * u_j_r^q
// where
//   R_qr = sqrt(|pos_i(t_q) - pos_j(t_r)|^2 + a_squared)
//   G(k, R) = exp(-j*k*R) / (4*pi*R)
//   pos_i(t) = (1-t) * seg_l_i + t * seg_r_i  (similarly for j)
//   w_i_q = len_i * gl_w_q;   u_i_q = len_i * gl_t_q
//
// a_squared = a^2 for V's off-edge case (corner regularization), 0 for the
// Yagi cross-wire case (no shared point so the kernel is already non-singular).
//
// Parallelism: each (i, j) pair is independent — OpenMP over the (i, j) grid.
// For each pair we precompute R[q, r] once, then loop k over all wavenumbers,
// reusing R from the per-thread cache.
static std::tuple<py::array_t<std::complex<double>>,
                  py::array_t<std::complex<double>>,
                  py::array_t<std::complex<double>>,
                  py::array_t<std::complex<double>>>
seg_seg_quad_batch_3d(
    py::array_t<double, py::array::c_style | py::array::forcecast> seg_l_i,
    py::array_t<double, py::array::c_style | py::array::forcecast> seg_r_i,
    py::array_t<double, py::array::c_style | py::array::forcecast> seg_l_j,
    py::array_t<double, py::array::c_style | py::array::forcecast> seg_r_j,
    double a_squared,
    py::array_t<double, py::array::c_style | py::array::forcecast> k_array,
    py::array_t<double, py::array::c_style | py::array::forcecast> gl_t,
    py::array_t<double, py::array::c_style | py::array::forcecast> gl_w
) {
    auto sli = seg_l_i.unchecked<2>();
    auto sri = seg_r_i.unchecked<2>();
    auto slj = seg_l_j.unchecked<2>();
    auto srj = seg_r_j.unchecked<2>();
    auto ka  = k_array.unchecked<1>();
    auto glt = gl_t.unchecked<1>();
    auto glw = gl_w.unchecked<1>();

    if (sli.shape(1) != 3 || sri.shape(1) != 3 ||
        slj.shape(1) != 3 || srj.shape(1) != 3) {
        throw std::runtime_error("segment endpoint arrays must have shape (N, 3)");
    }
    if (sli.shape(0) != sri.shape(0) || slj.shape(0) != srj.shape(0)) {
        throw std::runtime_error("seg_l and seg_r must have matching N");
    }
    if (glt.shape(0) != glw.shape(0)) {
        throw std::runtime_error("gl_t and gl_w must have matching length");
    }

    size_t N_i = sli.shape(0);
    size_t N_j = slj.shape(0);
    size_t n_k = ka.shape(0);
    size_t n_qp = glt.shape(0);

    py::array_t<std::complex<double>> J00({n_k, N_i, N_j});
    py::array_t<std::complex<double>> J10({n_k, N_i, N_j});
    py::array_t<std::complex<double>> J01({n_k, N_i, N_j});
    py::array_t<std::complex<double>> J11({n_k, N_i, N_j});
    auto j00 = J00.mutable_unchecked<3>();
    auto j10 = J10.mutable_unchecked<3>();
    auto j01 = J01.mutable_unchecked<3>();
    auto j11 = J11.mutable_unchecked<3>();

    const double inv_4pi = 1.0 / (4.0 * M_PI);

    // Per-segment quadrature-point positions and lengths -- k-independent so
    // compute once outside the parallel region.
    std::vector<double> pos_i(N_i * n_qp * 3);
    std::vector<double> pos_j(N_j * n_qp * 3);
    std::vector<double> len_i(N_i);
    std::vector<double> len_j(N_j);
    for (size_t i = 0; i < N_i; i++) {
        double dx = sri(i,0) - sli(i,0);
        double dy = sri(i,1) - sli(i,1);
        double dz = sri(i,2) - sli(i,2);
        len_i[i] = std::sqrt(dx*dx + dy*dy + dz*dz);
        for (size_t q = 0; q < n_qp; q++) {
            double t = glt(q);
            pos_i[(i*n_qp + q)*3 + 0] = (1.0 - t) * sli(i,0) + t * sri(i,0);
            pos_i[(i*n_qp + q)*3 + 1] = (1.0 - t) * sli(i,1) + t * sri(i,1);
            pos_i[(i*n_qp + q)*3 + 2] = (1.0 - t) * sli(i,2) + t * sri(i,2);
        }
    }
    for (size_t j = 0; j < N_j; j++) {
        double dx = srj(j,0) - slj(j,0);
        double dy = srj(j,1) - slj(j,1);
        double dz = srj(j,2) - slj(j,2);
        len_j[j] = std::sqrt(dx*dx + dy*dy + dz*dz);
        for (size_t r = 0; r < n_qp; r++) {
            double t = glt(r);
            pos_j[(j*n_qp + r)*3 + 0] = (1.0 - t) * slj(j,0) + t * srj(j,0);
            pos_j[(j*n_qp + r)*3 + 1] = (1.0 - t) * slj(j,1) + t * srj(j,1);
            pos_j[(j*n_qp + r)*3 + 2] = (1.0 - t) * slj(j,2) + t * srj(j,2);
        }
    }

    #pragma omp parallel for collapse(2) schedule(static)
    for (size_t i = 0; i < N_i; i++) {
        for (size_t j = 0; j < N_j; j++) {
            // n_qp <= 8 in practice, so n_qp^2 <= 64. Stack-allocated, aligned
            // for 32-byte AVX2 loads. The hot loop is the sincos at line ~Y
            // below; the explicit per-(qr) array layout lets `#pragma omp simd`
            // batch the sincos calls into the libmvec vectorized form
            // (_ZGVdN4v_sin / _ZGVdN4v_cos) instead of N_pairs individual calls.
            alignas(32) double R[64];
            alignas(32) double inv_R_4pi[64];
            alignas(32) double phases[64];
            alignas(32) double cos_phases[64];
            alignas(32) double sin_phases[64];
            alignas(32) double wi_arr[64], wj_arr[64], ui_arr[64], uj_arr[64];

            const double *pi = &pos_i[i * n_qp * 3];
            const double *pj = &pos_j[j * n_qp * 3];
            for (size_t q = 0; q < n_qp; q++) {
                double pix = pi[q*3 + 0];
                double piy = pi[q*3 + 1];
                double piz = pi[q*3 + 2];
                for (size_t r = 0; r < n_qp; r++) {
                    double dx = pix - pj[r*3 + 0];
                    double dy = piy - pj[r*3 + 1];
                    double dz = piz - pj[r*3 + 2];
                    R[q*n_qp + r] = std::sqrt(dx*dx + dy*dy + dz*dz + a_squared);
                }
            }

            double Li = len_i[i];
            double Lj = len_j[j];
            size_t n_pairs = n_qp * n_qp;

            // k-independent precompute -- weights, u-coords, and 1/(4πR).
            for (size_t q = 0; q < n_qp; q++) {
                double wi = glw(q) * Li;
                double ui = glt(q) * Li;
                for (size_t r = 0; r < n_qp; r++) {
                    size_t qr = q*n_qp + r;
                    wi_arr[qr] = wi;
                    wj_arr[qr] = glw(r) * Lj;
                    ui_arr[qr] = ui;
                    uj_arr[qr] = glt(r) * Lj;
                }
            }
            #pragma omp simd
            for (size_t qr = 0; qr < n_pairs; qr++) {
                inv_R_4pi[qr] = inv_4pi / R[qr];
            }

            for (size_t kk = 0; kk < n_k; kk++) {
                double k = ka(kk);

                // Stage 1: phase = -k * R, fully vectorizable.
                #pragma omp simd
                for (size_t qr = 0; qr < n_pairs; qr++) {
                    phases[qr] = -k * R[qr];
                }

                // Stage 2: vectorized cos and sin via libmvec. Split into two
                // loops on purpose — if cos/sin appear in the same loop body
                // on the same input, GCC fuses them into a single scalar
                // `sincos` call (smart for serial code), but libmvec has no
                // vector `sincos`, only `_ZGVdN4v_cos` and `_ZGVdN4v_sin`.
                // Splitting keeps them as independent vectorizable calls.
                #pragma omp simd
                for (size_t qr = 0; qr < n_pairs; qr++) {
                    cos_phases[qr] = std::cos(phases[qr]);
                }
                #pragma omp simd
                for (size_t qr = 0; qr < n_pairs; qr++) {
                    sin_phases[qr] = std::sin(phases[qr]);
                }

                // Stage 3: scalar reductions of the 4 J integrals. The compiler
                // vectorizes the per-component accumulation across qr.
                double s00_re = 0.0, s00_im = 0.0;
                double s10_re = 0.0, s10_im = 0.0;
                double s01_re = 0.0, s01_im = 0.0;
                double s11_re = 0.0, s11_im = 0.0;
                #pragma omp simd reduction(+:s00_re,s00_im,s10_re,s10_im,s01_re,s01_im,s11_re,s11_im)
                for (size_t qr = 0; qr < n_pairs; qr++) {
                    double iR = inv_R_4pi[qr];
                    double G_re = cos_phases[qr] * iR;
                    double G_im = sin_phases[qr] * iR;
                    double wij  = wi_arr[qr] * wj_arr[qr];
                    double wG_re = wij * G_re;
                    double wG_im = wij * G_im;
                    s00_re += wG_re;
                    s00_im += wG_im;
                    s10_re += ui_arr[qr] * wG_re;
                    s10_im += ui_arr[qr] * wG_im;
                    s01_re += uj_arr[qr] * wG_re;
                    s01_im += uj_arr[qr] * wG_im;
                    double uij = ui_arr[qr] * uj_arr[qr];
                    s11_re += uij * wG_re;
                    s11_im += uij * wG_im;
                }
                j00(kk, i, j) = std::complex<double>(s00_re, s00_im);
                j10(kk, i, j) = std::complex<double>(s10_re, s10_im);
                j01(kk, i, j) = std::complex<double>(s01_re, s01_im);
                j11(kk, i, j) = std::complex<double>(s11_re, s11_im);
            }
        }
    }

    return std::make_tuple(J00, J10, J01, J11);
}


// Batched same-wire Gauss-Legendre quadrature on the *regularized* kernel
//   G_reg(R, k) = (exp(-j k R) - 1) / (4 pi R),    R = sqrt(diff^2 + a^2)
// for all segment pairs (i, j) on a shared 1D arc-length line.
//
// Inputs:
//   seg_endpoints : (N+1,) arc-length / position of segment boundaries
//   a             : wire radius (corner regularization)
//   k_array       : (n_k,) wavenumbers
//   gl_t, gl_w    : Gauss-Legendre nodes/weights mapped to [0, 1]
//                   (i.e. t = (xi + 1)/2 and w = w_legendre / 2)
//
// For each pair (i, j) we precompute the per-pair R table once and reuse it
// across the k axis. OpenMP parallelizes over (i, j).
//
// Output:
//   (J00, J10, J01, J11), each (n_k, N, N) complex.
static std::tuple<py::array_t<std::complex<double>>,
                  py::array_t<std::complex<double>>,
                  py::array_t<std::complex<double>>,
                  py::array_t<std::complex<double>>>
seg_seg_reg_quad_batch_1d(
    py::array_t<double, py::array::c_style | py::array::forcecast> seg_endpoints,
    double a,
    py::array_t<double, py::array::c_style | py::array::forcecast> k_array,
    py::array_t<double, py::array::c_style | py::array::forcecast> gl_t,
    py::array_t<double, py::array::c_style | py::array::forcecast> gl_w
) {
    auto se  = seg_endpoints.unchecked<1>();
    auto ka  = k_array.unchecked<1>();
    auto glt = gl_t.unchecked<1>();
    auto glw = gl_w.unchecked<1>();

    if (glt.shape(0) != glw.shape(0)) {
        throw std::runtime_error("gl_t and gl_w must have matching length");
    }
    if (se.shape(0) < 2) {
        throw std::runtime_error("seg_endpoints must have at least 2 entries");
    }

    size_t N    = se.shape(0) - 1;
    size_t n_k  = ka.shape(0);
    size_t n_qp = glt.shape(0);
    if (n_qp * n_qp > 64) {
        throw std::runtime_error("n_qp too large (n_qp^2 must be <= 64)");
    }

    const double a_sq    = a * a;
    const double inv_4pi = 1.0 / (4.0 * M_PI);

    py::array_t<std::complex<double>> J00({n_k, N, N});
    py::array_t<std::complex<double>> J10({n_k, N, N});
    py::array_t<std::complex<double>> J01({n_k, N, N});
    py::array_t<std::complex<double>> J11({n_k, N, N});
    auto j00 = J00.mutable_unchecked<3>();
    auto j10 = J10.mutable_unchecked<3>();
    auto j01 = J01.mutable_unchecked<3>();
    auto j11 = J11.mutable_unchecked<3>();

    // Per-segment quadrature positions and lengths.
    std::vector<double> pos(N * n_qp);
    std::vector<double> Lvec(N);
    for (size_t i = 0; i < N; i++) {
        double sl = se(i);
        double sr = se(i + 1);
        double Li = sr - sl;
        Lvec[i] = Li;
        for (size_t q = 0; q < n_qp; q++) {
            pos[i * n_qp + q] = sl + Li * glt(q);
        }
    }

    #pragma omp parallel for collapse(2) schedule(static)
    for (size_t i = 0; i < N; i++) {
        for (size_t j = 0; j < N; j++) {
            // Same split-loop layout as seg_seg_quad_batch_3d: the per-(qr)
            // arrays let `#pragma omp simd` substitute libmvec's
            // _ZGVdN4v_cos / _ZGVdN4v_sin for the inner sincos calls.
            alignas(32) double R[64];
            alignas(32) double inv_R_4pi[64];
            alignas(32) double phases[64];
            alignas(32) double cos_phases[64];
            alignas(32) double sin_phases[64];
            alignas(32) double wi_arr[64], wj_arr[64], ui_arr[64], uj_arr[64];

            double Li = Lvec[i];
            double Lj = Lvec[j];
            const double *pi_q = &pos[i * n_qp];
            const double *pj_r = &pos[j * n_qp];
            for (size_t q = 0; q < n_qp; q++) {
                double piq = pi_q[q];
                for (size_t r = 0; r < n_qp; r++) {
                    double d = piq - pj_r[r];
                    R[q * n_qp + r] = std::sqrt(d * d + a_sq);
                }
            }

            size_t n_pairs = n_qp * n_qp;
            for (size_t q = 0; q < n_qp; q++) {
                double wi = glw(q) * Li;
                double ui = glt(q) * Li;
                for (size_t r = 0; r < n_qp; r++) {
                    size_t qr = q * n_qp + r;
                    wi_arr[qr] = wi;
                    wj_arr[qr] = glw(r) * Lj;
                    ui_arr[qr] = ui;
                    uj_arr[qr] = glt(r) * Lj;
                }
            }
            #pragma omp simd
            for (size_t qr = 0; qr < n_pairs; qr++) {
                inv_R_4pi[qr] = inv_4pi / R[qr];
            }

            for (size_t kk = 0; kk < n_k; kk++) {
                double k = ka(kk);

                #pragma omp simd
                for (size_t qr = 0; qr < n_pairs; qr++) {
                    phases[qr] = -k * R[qr];
                }
                #pragma omp simd
                for (size_t qr = 0; qr < n_pairs; qr++) {
                    cos_phases[qr] = std::cos(phases[qr]);
                }
                #pragma omp simd
                for (size_t qr = 0; qr < n_pairs; qr++) {
                    sin_phases[qr] = std::sin(phases[qr]);
                }

                // exp(-j k R) - 1 = (cos(-kR) - 1) + j sin(-kR)
                double s00_re = 0.0, s00_im = 0.0;
                double s10_re = 0.0, s10_im = 0.0;
                double s01_re = 0.0, s01_im = 0.0;
                double s11_re = 0.0, s11_im = 0.0;
                #pragma omp simd reduction(+:s00_re,s00_im,s10_re,s10_im,s01_re,s01_im,s11_re,s11_im)
                for (size_t qr = 0; qr < n_pairs; qr++) {
                    double iR = inv_R_4pi[qr];
                    double Greg_re = (cos_phases[qr] - 1.0) * iR;
                    double Greg_im = sin_phases[qr] * iR;
                    double wij = wi_arr[qr] * wj_arr[qr];
                    double wG_re = wij * Greg_re;
                    double wG_im = wij * Greg_im;
                    s00_re += wG_re;
                    s00_im += wG_im;
                    s10_re += ui_arr[qr] * wG_re;
                    s10_im += ui_arr[qr] * wG_im;
                    s01_re += uj_arr[qr] * wG_re;
                    s01_im += uj_arr[qr] * wG_im;
                    double uij = ui_arr[qr] * uj_arr[qr];
                    s11_re += uij * wG_re;
                    s11_im += uij * wG_im;
                }
                j00(kk, i, j) = std::complex<double>(s00_re, s00_im);
                j10(kk, i, j) = std::complex<double>(s10_re, s10_im);
                j01(kk, i, j) = std::complex<double>(s01_re, s01_im);
                j11(kk, i, j) = std::complex<double>(s11_re, s11_im);
            }
        }
    }

    return std::make_tuple(J00, J10, J01, J11);
}


// Assemble the (n_k, n_basis, n_basis) Z matrix from the four J tensors,
// per-segment h, the segment tangent-dot-product table, the left/right
// segment index of each basis, and omega(k).
//
// For each basis pair (m, n) and each wavenumber k:
//   Z[k, m, n] = (j w mu) * I_A + S / (j w eps)
// with
//   m_l = left_seg[m],  m_r = right_seg[m]
//   n_l = left_seg[n],  n_r = right_seg[n]
//   h*_m = h[m_*],      h*_n = h[n_*]
//   td_xy = td_all[m_x, n_y]
//   S   =  J00_ll/(hl_m hl_n) - J00_lr/(hl_m hr_n)
//        - J00_rl/(hr_m hl_n) + J00_rr/(hr_m hr_n)
//   I_A =  td_ll *  J11_ll/(hl_m hl_n)
//        + td_lr * (J10_lr/hl_m   - J11_lr/(hl_m hr_n))
//        + td_rl * (J01_rl/hl_n   - J11_rl/(hr_m hl_n))
//        + td_rr * (J00_rr - J10_rr/hr_m - J01_rr/hr_n + J11_rr/(hr_m hr_n))
//
// Loop order: collapse(2) parallel for (k, m); the inner n loop reads
// contiguous rows of the J tensors (when left_seg/right_seg are consecutive,
// which is the common case for both V and Yagi within a wire).
static py::array_t<std::complex<double>>
assemble_Z(
    py::array_t<std::complex<double>, py::array::c_style | py::array::forcecast> J00,
    py::array_t<std::complex<double>, py::array::c_style | py::array::forcecast> J10,
    py::array_t<std::complex<double>, py::array::c_style | py::array::forcecast> J01,
    py::array_t<std::complex<double>, py::array::c_style | py::array::forcecast> J11,
    py::array_t<double, py::array::c_style | py::array::forcecast> h_per_seg,
    py::array_t<double, py::array::c_style | py::array::forcecast> td_all,
    py::array_t<int64_t, py::array::c_style | py::array::forcecast> left_seg,
    py::array_t<int64_t, py::array::c_style | py::array::forcecast> right_seg,
    py::array_t<double, py::array::c_style | py::array::forcecast> omega_array,
    double eps,
    double mu
) {
    auto j00_info = J00.request();
    auto j10_info = J10.request();
    auto j01_info = J01.request();
    auto j11_info = J11.request();

    if (j00_info.ndim != 3) throw std::runtime_error("J tensors must be 3D");
    size_t n_k = (size_t)j00_info.shape[0];
    size_t N   = (size_t)j00_info.shape[1];
    if ((size_t)j00_info.shape[2] != N) {
        throw std::runtime_error("J tensors must be square (n_k, N, N)");
    }

    auto h   = h_per_seg.unchecked<1>();
    auto td  = td_all.unchecked<2>();
    auto ls  = left_seg.unchecked<1>();
    auto rs  = right_seg.unchecked<1>();
    auto om  = omega_array.unchecked<1>();

    if ((size_t)h.shape(0) != N) {
        throw std::runtime_error("h_per_seg length must match J tensor N");
    }
    if ((size_t)td.shape(0) != N || (size_t)td.shape(1) != N) {
        throw std::runtime_error("td_all must be (N, N)");
    }
    if (ls.shape(0) != rs.shape(0)) {
        throw std::runtime_error("left_seg / right_seg must have matching length");
    }
    if ((size_t)om.shape(0) != n_k) {
        throw std::runtime_error("omega_array length must match n_k");
    }
    size_t n_basis = (size_t)ls.shape(0);

    py::array_t<std::complex<double>> Z({n_k, n_basis, n_basis});
    auto z_info = Z.request();

    const std::complex<double>* j00_ptr = static_cast<const std::complex<double>*>(j00_info.ptr);
    const std::complex<double>* j10_ptr = static_cast<const std::complex<double>*>(j10_info.ptr);
    const std::complex<double>* j01_ptr = static_cast<const std::complex<double>*>(j01_info.ptr);
    const std::complex<double>* j11_ptr = static_cast<const std::complex<double>*>(j11_info.ptr);
    std::complex<double>* z_ptr = static_cast<std::complex<double>*>(z_info.ptr);

    const std::complex<double> j_unit(0.0, 1.0);

    #pragma omp parallel for collapse(2) schedule(static)
    for (size_t kk = 0; kk < n_k; kk++) {
        for (size_t m = 0; m < n_basis; m++) {
            int64_t m_l = ls(m);
            int64_t m_r = rs(m);
            double hl_m = h(m_l);
            double hr_m = h(m_r);
            double inv_hl_m = 1.0 / hl_m;
            double inv_hr_m = 1.0 / hr_m;

            double omega_k = om(kk);
            std::complex<double> jw_mu    = j_unit * (omega_k * mu);
            std::complex<double> inv_jw_eps = 1.0 / (j_unit * (omega_k * eps));

            size_t base_kk = kk * N * N;
            const std::complex<double> *j00_ml = j00_ptr + base_kk + (size_t)m_l * N;
            const std::complex<double> *j00_mr = j00_ptr + base_kk + (size_t)m_r * N;
            const std::complex<double> *j10_ml = j10_ptr + base_kk + (size_t)m_l * N;
            const std::complex<double> *j10_mr = j10_ptr + base_kk + (size_t)m_r * N;
            const std::complex<double> *j01_mr = j01_ptr + base_kk + (size_t)m_r * N;
            const std::complex<double> *j11_ml = j11_ptr + base_kk + (size_t)m_l * N;
            const std::complex<double> *j11_mr = j11_ptr + base_kk + (size_t)m_r * N;

            const double *td_ml = &td(m_l, 0);
            const double *td_mr = &td(m_r, 0);

            std::complex<double> *z_row = z_ptr + kk * n_basis * n_basis + m * n_basis;

            for (size_t n = 0; n < n_basis; n++) {
                int64_t n_l = ls(n);
                int64_t n_r = rs(n);
                double inv_hl_n = 1.0 / h(n_l);
                double inv_hr_n = 1.0 / h(n_r);

                double td_ll = td_ml[n_l];
                double td_lr = td_ml[n_r];
                double td_rl = td_mr[n_l];
                double td_rr = td_mr[n_r];

                std::complex<double> J00_ll = j00_ml[n_l];
                std::complex<double> J00_lr = j00_ml[n_r];
                std::complex<double> J00_rl = j00_mr[n_l];
                std::complex<double> J00_rr = j00_mr[n_r];

                std::complex<double> J10_lr = j10_ml[n_r];
                std::complex<double> J10_rr = j10_mr[n_r];

                std::complex<double> J01_rl = j01_mr[n_l];
                std::complex<double> J01_rr = j01_mr[n_r];

                std::complex<double> J11_ll = j11_ml[n_l];
                std::complex<double> J11_lr = j11_ml[n_r];
                std::complex<double> J11_rl = j11_mr[n_l];
                std::complex<double> J11_rr = j11_mr[n_r];

                double inv_hlm_hln = inv_hl_m * inv_hl_n;
                double inv_hlm_hrn = inv_hl_m * inv_hr_n;
                double inv_hrm_hln = inv_hr_m * inv_hl_n;
                double inv_hrm_hrn = inv_hr_m * inv_hr_n;

                std::complex<double> S =
                      J00_ll * inv_hlm_hln
                    - J00_lr * inv_hlm_hrn
                    - J00_rl * inv_hrm_hln
                    + J00_rr * inv_hrm_hrn;

                std::complex<double> I_A =
                      td_ll * (J11_ll * inv_hlm_hln)
                    + td_lr * (J10_lr * inv_hl_m - J11_lr * inv_hlm_hrn)
                    + td_rl * (J01_rl * inv_hl_n - J11_rl * inv_hrm_hln)
                    + td_rr * (J00_rr - J10_rr * inv_hr_m - J01_rr * inv_hr_n + J11_rr * inv_hrm_hrn);

                z_row[n] = jw_mu * I_A + S * inv_jw_eps;
            }
        }
    }

    return Z;
}


// Assemble Z from per-basis (segment, L, R) support arrays (n_basis, 2). Each
// basis has up to 2 wings, with arbitrary level and slope per wing. Interior
// tent bases encode (left, right) as ((sm_l, 0, 1), (sm_r, 1, 0)); junction
// directional bases use wing 0 only and zero out wing 1 (L=R=0 → slope=0 → no
// contribution). The general 2×2 (a, b) sum per (m, n):
//
//   slope[m, a] = (R[m, a] - L[m, a]) / h[support_seg[m, a]]
//   S[k, m, n]   = Σ_a Σ_b slope[m,a]*slope[n,b] * J00[k, sm_a, sn_b]
//   I_A[k, m, n] = Σ_a Σ_b td_all[sm_a, sn_b] * (
//                     L[m,a]*L[n,b]*J00 + L[m,a]*slope[n,b]*J01
//                   + slope[m,a]*L[n,b]*J10 + slope[m,a]*slope[n,b]*J11 )
//   Z[k, m, n]   = jωμ I_A + S / (jωε)
//
// Parallel collapse(2) over (k, m); inner n loop reads contiguous J rows.
static py::array_t<std::complex<double>>
assemble_Z_general(
    py::array_t<std::complex<double>, py::array::c_style | py::array::forcecast> J00,
    py::array_t<std::complex<double>, py::array::c_style | py::array::forcecast> J10,
    py::array_t<std::complex<double>, py::array::c_style | py::array::forcecast> J01,
    py::array_t<std::complex<double>, py::array::c_style | py::array::forcecast> J11,
    py::array_t<double, py::array::c_style | py::array::forcecast> h_per_seg,
    py::array_t<double, py::array::c_style | py::array::forcecast> td_all,
    py::array_t<int64_t, py::array::c_style | py::array::forcecast> support_seg,
    py::array_t<double, py::array::c_style | py::array::forcecast> support_L,
    py::array_t<double, py::array::c_style | py::array::forcecast> support_R,
    py::array_t<double, py::array::c_style | py::array::forcecast> omega_array,
    double eps,
    double mu
) {
    auto j00_info = J00.request();
    auto j10_info = J10.request();
    auto j01_info = J01.request();
    auto j11_info = J11.request();

    if (j00_info.ndim != 3) throw std::runtime_error("J tensors must be 3D");
    size_t n_k = (size_t)j00_info.shape[0];
    size_t N   = (size_t)j00_info.shape[1];
    if ((size_t)j00_info.shape[2] != N) {
        throw std::runtime_error("J tensors must be square (n_k, N, N)");
    }

    auto h   = h_per_seg.unchecked<1>();
    auto td  = td_all.unchecked<2>();
    auto ssg = support_seg.unchecked<2>();
    auto sL  = support_L.unchecked<2>();
    auto sR  = support_R.unchecked<2>();
    auto om  = omega_array.unchecked<1>();

    if ((size_t)h.shape(0) != N) {
        throw std::runtime_error("h_per_seg length must match J tensor N");
    }
    if ((size_t)td.shape(0) != N || (size_t)td.shape(1) != N) {
        throw std::runtime_error("td_all must be (N, N)");
    }
    if (ssg.shape(1) != 2 || sL.shape(1) != 2 || sR.shape(1) != 2) {
        throw std::runtime_error("support_seg/L/R must have shape (n_basis, 2)");
    }
    if (ssg.shape(0) != sL.shape(0) || ssg.shape(0) != sR.shape(0)) {
        throw std::runtime_error("support_seg/L/R must have matching n_basis");
    }
    if ((size_t)om.shape(0) != n_k) {
        throw std::runtime_error("omega_array length must match n_k");
    }
    size_t n_basis = (size_t)ssg.shape(0);

    std::vector<int64_t> ssg_flat(n_basis * 2);
    std::vector<double>  L_flat(n_basis * 2);
    std::vector<double>  slope_flat(n_basis * 2);
    for (size_t m = 0; m < n_basis; m++) {
        for (size_t a = 0; a < 2; a++) {
            int64_t s = ssg(m, a);
            double Lv = sL(m, a);
            double Rv = sR(m, a);
            ssg_flat[m * 2 + a] = s;
            L_flat[m * 2 + a]   = Lv;
            slope_flat[m * 2 + a] = (Rv - Lv) / h(s);
        }
    }

    py::array_t<std::complex<double>> Z({n_k, n_basis, n_basis});
    auto z_info = Z.request();

    const std::complex<double>* j00_ptr = static_cast<const std::complex<double>*>(j00_info.ptr);
    const std::complex<double>* j10_ptr = static_cast<const std::complex<double>*>(j10_info.ptr);
    const std::complex<double>* j01_ptr = static_cast<const std::complex<double>*>(j01_info.ptr);
    const std::complex<double>* j11_ptr = static_cast<const std::complex<double>*>(j11_info.ptr);
    std::complex<double>* z_ptr = static_cast<std::complex<double>*>(z_info.ptr);

    const std::complex<double> j_unit(0.0, 1.0);

    #pragma omp parallel for collapse(2) schedule(static)
    for (size_t kk = 0; kk < n_k; kk++) {
        for (size_t m = 0; m < n_basis; m++) {
            double omega_k = om(kk);
            std::complex<double> jw_mu      = j_unit * (omega_k * mu);
            std::complex<double> inv_jw_eps = 1.0 / (j_unit * (omega_k * eps));

            int64_t sm0 = ssg_flat[m * 2 + 0];
            int64_t sm1 = ssg_flat[m * 2 + 1];
            double Lm0 = L_flat[m * 2 + 0];
            double Lm1 = L_flat[m * 2 + 1];
            double Sm0 = slope_flat[m * 2 + 0];
            double Sm1 = slope_flat[m * 2 + 1];

            size_t base_kk = kk * N * N;
            const std::complex<double> *j00_m0 = j00_ptr + base_kk + (size_t)sm0 * N;
            const std::complex<double> *j00_m1 = j00_ptr + base_kk + (size_t)sm1 * N;
            const std::complex<double> *j10_m0 = j10_ptr + base_kk + (size_t)sm0 * N;
            const std::complex<double> *j10_m1 = j10_ptr + base_kk + (size_t)sm1 * N;
            const std::complex<double> *j01_m0 = j01_ptr + base_kk + (size_t)sm0 * N;
            const std::complex<double> *j01_m1 = j01_ptr + base_kk + (size_t)sm1 * N;
            const std::complex<double> *j11_m0 = j11_ptr + base_kk + (size_t)sm0 * N;
            const std::complex<double> *j11_m1 = j11_ptr + base_kk + (size_t)sm1 * N;

            const double *td_m0 = &td(sm0, 0);
            const double *td_m1 = &td(sm1, 0);

            std::complex<double> *z_row = z_ptr + kk * n_basis * n_basis + m * n_basis;

            for (size_t n = 0; n < n_basis; n++) {
                int64_t sn0 = ssg_flat[n * 2 + 0];
                int64_t sn1 = ssg_flat[n * 2 + 1];
                double Ln0 = L_flat[n * 2 + 0];
                double Ln1 = L_flat[n * 2 + 1];
                double Sn0 = slope_flat[n * 2 + 0];
                double Sn1 = slope_flat[n * 2 + 1];

                std::complex<double> J00_00 = j00_m0[sn0];
                std::complex<double> J01_00 = j01_m0[sn0];
                std::complex<double> J10_00 = j10_m0[sn0];
                std::complex<double> J11_00 = j11_m0[sn0];
                double td00 = td_m0[sn0];

                std::complex<double> J00_01 = j00_m0[sn1];
                std::complex<double> J01_01 = j01_m0[sn1];
                std::complex<double> J10_01 = j10_m0[sn1];
                std::complex<double> J11_01 = j11_m0[sn1];
                double td01 = td_m0[sn1];

                std::complex<double> J00_10 = j00_m1[sn0];
                std::complex<double> J01_10 = j01_m1[sn0];
                std::complex<double> J10_10 = j10_m1[sn0];
                std::complex<double> J11_10 = j11_m1[sn0];
                double td10 = td_m1[sn0];

                std::complex<double> J00_11 = j00_m1[sn1];
                std::complex<double> J01_11 = j01_m1[sn1];
                std::complex<double> J10_11 = j10_m1[sn1];
                std::complex<double> J11_11 = j11_m1[sn1];
                double td11 = td_m1[sn1];

                std::complex<double> S =
                      (Sm0 * Sn0) * J00_00
                    + (Sm0 * Sn1) * J00_01
                    + (Sm1 * Sn0) * J00_10
                    + (Sm1 * Sn1) * J00_11;

                std::complex<double> I_A =
                      td00 * (Lm0*Ln0*J00_00 + Lm0*Sn0*J01_00 + Sm0*Ln0*J10_00 + Sm0*Sn0*J11_00)
                    + td01 * (Lm0*Ln1*J00_01 + Lm0*Sn1*J01_01 + Sm0*Ln1*J10_01 + Sm0*Sn1*J11_01)
                    + td10 * (Lm1*Ln0*J00_10 + Lm1*Sn0*J01_10 + Sm1*Ln0*J10_10 + Sm1*Sn0*J11_10)
                    + td11 * (Lm1*Ln1*J00_11 + Lm1*Sn1*J01_11 + Sm1*Ln1*J10_11 + Sm1*Sn1*J11_11);

                z_row[n] = jw_mu * I_A + S * inv_jw_eps;
            }
        }
    }

    return Z;
}


py::array_t<double> dist_outer_product(py::array_t<double> input0,
				       py::array_t<double> input1) {
    auto buf0 = input0.request();
    auto buf1 = input1.request();

    if (buf0.ndim != 2)
      throw std::runtime_error("Number of dimensions must be two");

    if (buf1.ndim != 2)
      throw std::runtime_error("Number of dimensions must be two");

    if (buf0.shape[1] != buf1.shape[1])
      throw std::runtime_error("Inputs must have same sized second dimension");
    
    size_t rows = buf0.shape[0];
    size_t cols = buf1.shape[0];
    size_t vsize = buf0.shape[1];

    auto result = py::array_t<double>({rows, cols});
    auto result_buf = result.request();

    double *ptr0 = static_cast<double *>(buf0.ptr);
    double *ptr1 = static_cast<double *>(buf1.ptr);
    double *result_ptr = static_cast<double *>(result_buf.ptr);

    #pragma omp parallel for
    for (size_t i = 0; i < rows; i++) {
      for (size_t j = 0; j < cols; j++) {
	auto sumsq = 0.0;
	for (size_t k = 0; k < vsize; k++) {
	  auto diff = ptr0[i*vsize+k] - ptr1[j*vsize+k];
	  sumsq += diff*diff;
	}
        result_ptr[i*cols+j] = sqrt(sumsq);
      }
    }

    return result;
}

// Templated B-spline moment-integral kernel.
//
// For each (i, j) segment pair, compute the (D+1)^2 polynomial moments
//   J[p, P, i, j] = sum_{q, r} wi[q] * ui[q]^p * wj[r] * uj[r]^P * G(R_qr)
// where
//   R_qr = sqrt(|pos_i(t_q) - pos_j(t_r)|^2 + a_squared)
//   G(R) = exp(-j*k*R) / (4*pi*R)
//   ui[q] = t_q * len_i,  uj[r] = t_r * len_j  (local arc lengths)
//
// Used by BSplinePySim._build_J_blocks for the all-pairs off-edge piece
// (same a^2 wire-radius regularization handles touching segments at kinks
// and at junctions). Single-k for now (BSplinePySim hasn't grown a swept
// path yet); add a batched k_array variant later if/when needed.
//
// Template parameter D = B-spline degree (1 or 2 currently — explicit
// instantiations below). Hardcoding D as a compile-time constant lets the
// compiler fully unroll the (D+1)^2 polynomial-moment inner loop, getting
// the same scalar-unrolled tight assembly the d=1 seg_seg_quad_batch_3d
// achieves with hand-rolled s00 / s10 / s01 / s11 accumulators.
//
// n_qp <= 8 assumed (n_qp^2 <= 64 scratch buffer size).
template<int D>
static py::array_t<std::complex<double>>
seg_seg_full_moments_bspline_kernel(
    py::array_t<double, py::array::c_style | py::array::forcecast> seg_l_i,
    py::array_t<double, py::array::c_style | py::array::forcecast> seg_r_i,
    py::array_t<double, py::array::c_style | py::array::forcecast> seg_l_j,
    py::array_t<double, py::array::c_style | py::array::forcecast> seg_r_j,
    double a_squared,
    double k,
    py::array_t<double, py::array::c_style | py::array::forcecast> gl_t,
    py::array_t<double, py::array::c_style | py::array::forcecast> gl_w
) {
    static constexpr int NM = D + 1;          // moments per axis
    static constexpr int NMM = NM * NM;       // total moments

    auto sli = seg_l_i.unchecked<2>();
    auto sri = seg_r_i.unchecked<2>();
    auto slj = seg_l_j.unchecked<2>();
    auto srj = seg_r_j.unchecked<2>();
    auto glt = gl_t.unchecked<1>();
    auto glw = gl_w.unchecked<1>();

    if (sli.shape(1) != 3 || sri.shape(1) != 3 ||
        slj.shape(1) != 3 || srj.shape(1) != 3) {
        throw std::runtime_error("segment endpoint arrays must have shape (N, 3)");
    }
    if (sli.shape(0) != sri.shape(0) || slj.shape(0) != srj.shape(0)) {
        throw std::runtime_error("seg_l and seg_r must have matching N");
    }
    if (glt.shape(0) != glw.shape(0)) {
        throw std::runtime_error("gl_t and gl_w must have matching length");
    }
    size_t n_qp_in = glt.shape(0);
    if (n_qp_in > 8) {
        throw std::runtime_error("n_qp > 8 not supported (scratch buffer size)");
    }

    size_t N_i = sli.shape(0);
    size_t N_j = slj.shape(0);
    size_t n_qp = n_qp_in;

    py::array_t<std::complex<double>> J({(size_t)NM, (size_t)NM, N_i, N_j});
    auto j_view = J.mutable_unchecked<4>();

    const double inv_4pi = 1.0 / (4.0 * M_PI);

    // Per-segment quadrature-point positions and lengths -- k-independent,
    // computed once outside the parallel region.
    std::vector<double> pos_i(N_i * n_qp * 3);
    std::vector<double> pos_j(N_j * n_qp * 3);
    std::vector<double> len_i(N_i);
    std::vector<double> len_j(N_j);
    for (size_t i = 0; i < N_i; i++) {
        double dx = sri(i,0) - sli(i,0);
        double dy = sri(i,1) - sli(i,1);
        double dz = sri(i,2) - sli(i,2);
        len_i[i] = std::sqrt(dx*dx + dy*dy + dz*dz);
        for (size_t q = 0; q < n_qp; q++) {
            double t = glt(q);
            pos_i[(i*n_qp + q)*3 + 0] = (1.0 - t) * sli(i,0) + t * sri(i,0);
            pos_i[(i*n_qp + q)*3 + 1] = (1.0 - t) * sli(i,1) + t * sri(i,1);
            pos_i[(i*n_qp + q)*3 + 2] = (1.0 - t) * sli(i,2) + t * sri(i,2);
        }
    }
    for (size_t j = 0; j < N_j; j++) {
        double dx = srj(j,0) - slj(j,0);
        double dy = srj(j,1) - slj(j,1);
        double dz = srj(j,2) - slj(j,2);
        len_j[j] = std::sqrt(dx*dx + dy*dy + dz*dz);
        for (size_t r = 0; r < n_qp; r++) {
            double t = glt(r);
            pos_j[(j*n_qp + r)*3 + 0] = (1.0 - t) * slj(j,0) + t * srj(j,0);
            pos_j[(j*n_qp + r)*3 + 1] = (1.0 - t) * slj(j,1) + t * srj(j,1);
            pos_j[(j*n_qp + r)*3 + 2] = (1.0 - t) * slj(j,2) + t * srj(j,2);
        }
    }

    #pragma omp parallel for collapse(2) schedule(static)
    for (size_t i = 0; i < N_i; i++) {
        for (size_t j = 0; j < N_j; j++) {
            alignas(32) double R[64];
            alignas(32) double inv_R_4pi[64];
            alignas(32) double phases[64];
            alignas(32) double cos_phases[64];
            alignas(32) double sin_phases[64];
            alignas(32) double G_re[64], G_im[64];
            // wuwu[pP, qr]: precomputed wi[q]*ui[q]^p * wj[r]*uj[r]^P,
            // flattened with pP = p*NM + P. For D=2: NMM*64 = 576 doubles = 4.5KB,
            // fits comfortably in L1.
            alignas(32) double wuwu[NMM * 64];

            const double *pi = &pos_i[i * n_qp * 3];
            const double *pj = &pos_j[j * n_qp * 3];
            for (size_t q = 0; q < n_qp; q++) {
                double pix = pi[q*3 + 0];
                double piy = pi[q*3 + 1];
                double piz = pi[q*3 + 2];
                for (size_t r = 0; r < n_qp; r++) {
                    double dx = pix - pj[r*3 + 0];
                    double dy = piy - pj[r*3 + 1];
                    double dz = piz - pj[r*3 + 2];
                    R[q*n_qp + r] = std::sqrt(dx*dx + dy*dy + dz*dz + a_squared);
                }
            }

            double Li = len_i[i];
            double Lj = len_j[j];
            size_t n_pairs = n_qp * n_qp;

            // Build wuwu[pP, qr]: pP indexes the moment (p*NM + P), qr the
            // quadrature pair (q*n_qp + r). The NM is a template constant so
            // ui_pow[p] / uj_pow[P] arrays unroll.
            for (size_t q = 0; q < n_qp; q++) {
                double wi = glw(q) * Li;
                double ui = glt(q) * Li;
                double ui_pow[NM];
                ui_pow[0] = 1.0;
                for (int p = 1; p < NM; p++) ui_pow[p] = ui_pow[p-1] * ui;
                for (size_t r = 0; r < n_qp; r++) {
                    double wj = glw(r) * Lj;
                    double uj = glt(r) * Lj;
                    double uj_pow[NM];
                    uj_pow[0] = 1.0;
                    for (int P = 1; P < NM; P++) uj_pow[P] = uj_pow[P-1] * uj;
                    double wij = wi * wj;
                    size_t qr = q * n_qp + r;
                    for (int p = 0; p < NM; p++) {
                        for (int P = 0; P < NM; P++) {
                            wuwu[(p * NM + P) * n_pairs + qr] = wij * ui_pow[p] * uj_pow[P];
                        }
                    }
                }
            }

            // Stage 1: phases = -k * R, then sincos via libmvec.
            #pragma omp simd
            for (size_t qr = 0; qr < n_pairs; qr++) {
                phases[qr] = -k * R[qr];
            }
            #pragma omp simd
            for (size_t qr = 0; qr < n_pairs; qr++) {
                cos_phases[qr] = std::cos(phases[qr]);
            }
            #pragma omp simd
            for (size_t qr = 0; qr < n_pairs; qr++) {
                sin_phases[qr] = std::sin(phases[qr]);
            }
            #pragma omp simd
            for (size_t qr = 0; qr < n_pairs; qr++) {
                inv_R_4pi[qr] = inv_4pi / R[qr];
                G_re[qr] = cos_phases[qr] * inv_R_4pi[qr];
                G_im[qr] = sin_phases[qr] * inv_R_4pi[qr];
            }

            // Stage 2: NMM moment reductions, each a vectorizable sum over qr.
            for (int pP = 0; pP < NMM; pP++) {
                double sr = 0.0, si = 0.0;
                const double *w_row = &wuwu[pP * n_pairs];
                #pragma omp simd reduction(+:sr,si)
                for (size_t qr = 0; qr < n_pairs; qr++) {
                    sr += w_row[qr] * G_re[qr];
                    si += w_row[qr] * G_im[qr];
                }
                j_view(pP / NM, pP % NM, i, j) = std::complex<double>(sr, si);
            }
        }
    }

    return J;
}

// Templated B-spline Z assembly kernel.
//
// For each (m, n) basis pair, assembles the EFIE Galerkin entry from the
// polynomial-moment tensor J and the per-(basis, wing, poly-degree)
// coefficient table:
//   Z[m,n] = j*omega*mu * sum_{a,b} (t·t)[sm, sn]
//            * sum_{p, q} polys[m, a, p] * polys[n, b, q] * J[p, q, sm, sn]
//          + (1/jωε)    * sum_{a,b}
//            * sum_{p≥1, q≥1} p*q * polys[m, a, p] * polys[n, b, q]
//                           * J[p-1, q-1, sm, sn]
// where sm = support_seg[m, a], sn = support_seg[n, b].
//
// Inactive wings of boundary / junction-directional bases have polys = 0
// at every p, so they contribute nothing — no special handling needed.
//
// Template parameter D = B-spline degree (1 or 2). NM = D+1 wings per basis
// and D+1 polynomial moments per wing. Hardcoding NM as a compile-time
// constant unrolls the (D+1)^4 inner muladd loop.
//
// Single-k for now (BSplinePySim doesn't have a swept path yet); the inputs
// are scalar omega instead of an omega_array.
template<int D>
static py::array_t<std::complex<double>>
assemble_Z_bspline_kernel(
    py::array_t<std::complex<double>, py::array::c_style | py::array::forcecast> J,
    py::array_t<int64_t, py::array::c_style | py::array::forcecast> support_seg,
    py::array_t<double, py::array::c_style | py::array::forcecast> polys,
    py::array_t<double, py::array::c_style | py::array::forcecast> td_all,
    double omega,
    double eps_,
    double mu_
) {
    static constexpr int NM = D + 1;

    auto j_view = J.unchecked<4>();
    auto ss_view = support_seg.unchecked<2>();
    auto p_view = polys.unchecked<3>();
    auto td_view = td_all.unchecked<2>();

    size_t n_basis = (size_t)support_seg.shape(0);
    if (support_seg.shape(1) != NM) {
        throw std::runtime_error("support_seg.shape(1) must equal D+1");
    }
    if (polys.shape(0) != (long)n_basis || polys.shape(1) != NM ||
        polys.shape(2) != NM) {
        throw std::runtime_error("polys.shape must be (n_basis, D+1, D+1)");
    }
    if (J.shape(0) != NM || J.shape(1) != NM) {
        throw std::runtime_error("J.shape(0:2) must be (D+1, D+1)");
    }

    py::array_t<std::complex<double>> Z({n_basis, n_basis});
    auto z_view = Z.mutable_unchecked<2>();

    // Z = j*omega*mu * Z_A_accum + (1/(j*omega*eps)) * Z_Phi_accum
    // For Z_A_accum = re + j*im:    j*omega*mu * (re + j*im) = -omega*mu*im + j*omega*mu*re
    // For Z_Phi_accum = re + j*im:  (re + j*im)/(j*omega*eps) = im/(omega*eps) - j*re/(omega*eps)
    const double omega_mu = omega * mu_;
    const double inv_omega_eps = 1.0 / (omega * eps_);

    #pragma omp parallel for collapse(2) schedule(static)
    for (size_t m = 0; m < n_basis; m++) {
        for (size_t n = 0; n < n_basis; n++) {
            double zA_re = 0.0, zA_im = 0.0;
            double zPhi_re = 0.0, zPhi_im = 0.0;

            for (int a = 0; a < NM; a++) {
                int64_t sm = ss_view(m, a);
                for (int b = 0; b < NM; b++) {
                    int64_t sn = ss_view(n, b);
                    double td = td_view(sm, sn);

                    double wA_re = 0.0, wA_im = 0.0;
                    double wPhi_re = 0.0, wPhi_im = 0.0;

                    for (int p = 0; p < NM; p++) {
                        double mp_ap = p_view(m, a, p);
                        for (int q = 0; q < NM; q++) {
                            double nq_bq = p_view(n, b, q);
                            std::complex<double> Jpq = j_view(p, q, sm, sn);
                            double prod = mp_ap * nq_bq;
                            wA_re += prod * Jpq.real();
                            wA_im += prod * Jpq.imag();
                            // p, q in {1..D}: Z_Phi contribution
                            if (p >= 1 && q >= 1) {
                                std::complex<double> Jpm1qm1 = j_view(p - 1, q - 1, sm, sn);
                                double pq = (double)(p * q) * prod;
                                wPhi_re += pq * Jpm1qm1.real();
                                wPhi_im += pq * Jpm1qm1.imag();
                            }
                        }
                    }

                    zA_re += td * wA_re;
                    zA_im += td * wA_im;
                    zPhi_re += wPhi_re;
                    zPhi_im += wPhi_im;
                }
            }

            double Zre = -omega_mu * zA_im + zPhi_im * inv_omega_eps;
            double Zim = omega_mu * zA_re - zPhi_re * inv_omega_eps;
            z_view(m, n) = std::complex<double>(Zre, Zim);
        }
    }

    return Z;
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
    int max_d
) {
    switch (max_d) {
        case 1:
            return assemble_Z_bspline_kernel<1>(J, support_seg, polys, td_all, omega, eps_, mu_);
        case 2:
            return assemble_Z_bspline_kernel<2>(J, support_seg, polys, td_all, omega, eps_, mu_);
        default:
            throw std::runtime_error(
                "assemble_Z_bspline: max_d must be 1 or 2");
    }
}


// Runtime dispatch wrapper. Picks the right template instantiation based on
// max_d (the maximum polynomial moment degree, == B-spline degree D).
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
    py::array_t<double, py::array::c_style | py::array::forcecast> gl_w
) {
    switch (max_d) {
        case 1:
            return seg_seg_full_moments_bspline_kernel<1>(
                seg_l_i, seg_r_i, seg_l_j, seg_r_j, a_squared, k, gl_t, gl_w);
        case 2:
            return seg_seg_full_moments_bspline_kernel<2>(
                seg_l_i, seg_r_i, seg_l_j, seg_r_j, a_squared, k, gl_t, gl_w);
        default:
            throw std::runtime_error(
                "seg_seg_full_moments_bspline: max_d must be 1 or 2 "
                "(add an explicit template instantiation in _accelerators.cpp)");
    }
}


// Toeplitz fast-path B-spline static-moment evaluation.
//
// For a single straight edge with uniform-h segments, the J_pq^static[i, j]
// integrals are translation-invariant in the arc direction — the matrix is
// Toeplitz with 2N-1 unique values per (p, q) moment. This function computes
// those 2N-1 values via the sympy-derived closed forms (inlined from
// _bspline_static_moments_inline.h) and gathers them to the (max_d+1,
// max_d+1, N, N) output.
//
// Replaces the per-edge numpy loop in `_seg_seg_static_moments` — that path
// took ~5 ms / call mainly from numpy dispatch overhead; the C++ inlined
// closed forms run in ~0.1 ms / call. Big win on multi-edge polylines like
// the hentenna where the static moments dominate after the all-pairs J kernel.
//
// max_d ∈ {0, 1, 2} currently — extends automatically when the header file
// is regenerated for larger MAX_D in scripts/derive_bspline_static_moments.py
// (and the case-list in J_static_dispatch below is extended).
static double J_static_dispatch(int p, int q,
                                double alpha, double beta,
                                double A, double B, double a) {
    int pq = p * 3 + q;
    switch (pq) {
        case 0: return J_static_pq_0_0(alpha, beta, A, B, a);
        case 1: return J_static_pq_0_1(alpha, beta, A, B, a);
        case 2: return J_static_pq_0_2(alpha, beta, A, B, a);
        case 3: return J_static_pq_1_0(alpha, beta, A, B, a);
        case 4: return J_static_pq_1_1(alpha, beta, A, B, a);
        case 5: return J_static_pq_1_2(alpha, beta, A, B, a);
        case 6: return J_static_pq_2_0(alpha, beta, A, B, a);
        case 7: return J_static_pq_2_1(alpha, beta, A, B, a);
        case 8: return J_static_pq_2_2(alpha, beta, A, B, a);
        default:
            throw std::runtime_error("J_static: (p, q) out of inline range");
    }
}

static py::array_t<double>
seg_seg_static_moments_bspline_uniform(double h, double a, size_t N, int max_d) {
    if (max_d < 0 || max_d > 2) {
        throw std::runtime_error("max_d out of range [0, 2]");
    }
    size_t NM = (size_t)(max_d + 1);
    py::array_t<double> out({NM, NM, N, N});
    auto v = out.mutable_unchecked<4>();

    // 2N-1 unique Toeplitz values per moment, indexed by Δ = j - i ∈ [-(N-1), N-1].
    // delta_idx = Δ + (N - 1) ∈ [0, 2N-2].
    size_t n_delta = 2 * N - 1;
    const double inv_4pi = 1.0 / (4.0 * M_PI);

    // Build (NM, NM, n_delta) Toeplitz table
    std::vector<double> table(NM * NM * n_delta);
    for (size_t p = 0; p < NM; p++) {
        for (size_t q = 0; q < NM; q++) {
            for (size_t di = 0; di < n_delta; di++) {
                long long delta = (long long)di - (long long)(N - 1);
                double alpha = 0.0;
                double beta = h;
                double A_ = (double)delta * h;
                double B_ = ((double)delta + 1.0) * h;
                double val = J_static_dispatch((int)p, (int)q, alpha, beta, A_, B_, a);
                table[(p * NM + q) * n_delta + di] = val * inv_4pi;
            }
        }
    }

    // Gather: v(p, q, i, j) = table[p, q, j - i + (N - 1)]
    for (size_t p = 0; p < NM; p++) {
        for (size_t q = 0; q < NM; q++) {
            const double *row = &table[(p * NM + q) * n_delta];
            for (size_t i = 0; i < N; i++) {
                for (size_t j = 0; j < N; j++) {
                    size_t di = (size_t)((long long)j - (long long)i + (long long)(N - 1));
                    v(p, q, i, j) = row[di];
                }
            }
        }
    }
    return out;
}


PYBIND11_MODULE(_accelerators, m) {
    m.def("dist_outer_product", &dist_outer_product, "Compute point to point euclidean distance");
    m.def("seg_seg_quad_batch_3d", &seg_seg_quad_batch_3d,
          "Batched 3D cross-segment Gauss-Legendre quadrature over a k vector. "
          "Returns (J00, J10, J01, J11) each (n_k, N_i, N_j) complex.",
          py::arg("seg_l_i"), py::arg("seg_r_i"),
          py::arg("seg_l_j"), py::arg("seg_r_j"),
          py::arg("a_squared"), py::arg("k_array"),
          py::arg("gl_t"), py::arg("gl_w"));
    m.def("seg_seg_reg_quad_batch_1d", &seg_seg_reg_quad_batch_1d,
          "Batched same-wire Gauss-Legendre quadrature on the regularized "
          "kernel (exp(-jkR)-1)/(4 pi R) over a k vector. "
          "Returns (J00, J10, J01, J11) each (n_k, N, N) complex.",
          py::arg("seg_endpoints"), py::arg("a"),
          py::arg("k_array"),
          py::arg("gl_t"), py::arg("gl_w"));
    m.def("assemble_Z", &assemble_Z,
          "Assemble the (n_k, n_basis, n_basis) Z matrix from the four J "
          "tensors, per-segment h, tangent-dot table, left/right basis-to-"
          "segment mappings, and omega(k). Returns Z complex.",
          py::arg("J00"), py::arg("J10"),
          py::arg("J01"), py::arg("J11"),
          py::arg("h_per_seg"), py::arg("td_all"),
          py::arg("left_seg"), py::arg("right_seg"),
          py::arg("omega_array"),
          py::arg("eps"), py::arg("mu"));
    m.def("assemble_Z_general", &assemble_Z_general,
          "Assemble Z from per-basis (support_seg, support_L, support_R) "
          "(n_basis, 2) arrays. Handles arbitrary 2-wing basis layouts "
          "including junction directional bases (one wing inactive with "
          "L=R=0). Returns Z complex (n_k, n_basis, n_basis).",
          py::arg("J00"), py::arg("J10"),
          py::arg("J01"), py::arg("J11"),
          py::arg("h_per_seg"), py::arg("td_all"),
          py::arg("support_seg"),
          py::arg("support_L"), py::arg("support_R"),
          py::arg("omega_array"),
          py::arg("eps"), py::arg("mu"));
    m.def("seg_seg_full_moments_bspline", &seg_seg_full_moments_bspline,
          "Single-k full-kernel polynomial moment integrals for the B-spline "
          "Galerkin MoM. Returns J of shape (max_d+1, max_d+1, N_i, N_j) "
          "complex. Templated on max_d at compile time; currently "
          "instantiated for max_d in {1, 2}.",
          py::arg("seg_l_i"), py::arg("seg_r_i"),
          py::arg("seg_l_j"), py::arg("seg_r_j"),
          py::arg("a_squared"), py::arg("k"),
          py::arg("max_d"),
          py::arg("gl_t"), py::arg("gl_w"));
    m.def("seg_seg_static_moments_bspline_uniform",
          &seg_seg_static_moments_bspline_uniform,
          "Closed-form same-edge static-kernel polynomial moments J_pq for a "
          "uniform-h edge with N segments. Uses Toeplitz structure (2N-1 "
          "unique values per moment) and inlined sympy-derived closed forms. "
          "Returns J_static of shape (max_d+1, max_d+1, N, N), with the "
          "1/(4π) prefactor folded in.",
          py::arg("h"), py::arg("a"), py::arg("N"), py::arg("max_d"));
    m.def("assemble_Z_bspline", &assemble_Z_bspline,
          "Assemble the (n_basis, n_basis) Z matrix from the polynomial-"
          "moment tensor J, per-basis polynomial coefficients, support-segment "
          "map, and tangent-dot table. Templated on max_d at compile time; "
          "currently instantiated for max_d in {1, 2}. Single-k.",
          py::arg("J"), py::arg("support_seg"),
          py::arg("polys"), py::arg("td_all"),
          py::arg("omega"), py::arg("eps"), py::arg("mu"),
          py::arg("max_d"));
}
