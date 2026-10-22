// The sinusoidal basis' POINT-MATCHED accelerator kernels with no Python in
// them (momwire#1607 B1): the field tensors `_accel_sinusoidal.cpp` binds --
// `sinusoidal_field_tensor` and its reflection-coefficient, extended-kernel
// and complex-wavenumber twins -- see `_sinusoidal_core.h`.
//
// The kernel text below is the extension TU's, moved verbatim, on the
// B-spline core's pattern (`_bspline_core.cpp`, momwire#1517 step 0): it
// spells pybind11's array vocabulary (`py::array_t`, `unchecked<N>()`, an
// output array built from its shape, `py::gil_scoped_release`), and here `py`
// names `mw::nd` (`_mw_nd.h`), which is that vocabulary over memory a host
// provides. Through the extension the host is numpy's and the GIL's
// (`_mw_nd_binding.h`); standalone, the host allocates with `new`. The entry
// points at the end are the bound functions, by name.
//
// The Galerkin far fills stay in the extension TU until the SG phase; the two
// series helpers both use are `_sinusoidal_series_inline.h`.
//
// MSVC builds every TU /fp:fast (setup.py), as it built this text in the
// extension TU; nothing here changes that.

// M_PI: see `_accel_common.h`; must stay above the first math header.
#define _USE_MATH_DEFINES

#include <complex>

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <tuple>
#include <vector>
#ifdef _OPENMP
#include <omp.h>
#endif

// Explicit fused multiply-adds, since the build no longer contracts on its
// own (momwire#1194).
#include "_fma_inline.h"

#include "_mw_core.h"
#include "_mw_nd.h"
#include "_sinusoidal_core.h"

namespace py = mw::nd;

#include "_stable_inline.h"
#include "_sinusoidal_series_inline.h"


// ---------------------------------------------------------------------------
// Fresnel field-dyad projection tail — the reflection-coefficient finite
// ground (NEC IPERF=0). Called by the EXTENDED-kernel refl entry point
// (`sinusoidal_field_tensor_ek_refl`, momwire#259); the reduced one
// (`sinusoidal_field_tensor_refl`) carries a byte-frozen open-coded copy of
// exactly this algebra — see the note at its REFL branch for why it is not
// routed through here.
//
// At each (observer m, IMAGE source n) pair the specular ray's incidence
// cosine cos θ fixes the Fresnel coefficients (principal-branch sqrt, matching
// `_ground_refl.fresnel_rho`)
//   ρ_v = (ε̃ cosθ − √(ε̃ − sin²θ)) / (ε̃ cosθ + √(ε̃ − sin²θ))
//   ρ_h = (   cosθ − √(ε̃ − sin²θ)) / (   cosθ + √(ε̃ − sin²θ))
// and the dyad D = ρ_v(I − p̂p̂) − ρ_h p̂p̂ collapses under the observer
// projection to
//   t_m · D · E = ρ_v·(t_m·E) − (ρ_v + ρ_h)·(t_m·p̂)·(E·p̂)
// with both scalars read off the UNPROJECTED (E_z, E_ρ) tables:
//   t_m·E = td·E_z + rho_proj·E_ρ      (the plain projection)
//   E·p̂  = (t_n·p̂)·E_z + (ρ̂·p̂)·E_ρ    (t_img·p̂ = t_n·p̂; ρ̂ = rho_vec/rho_eval)
//
// That is exactly why one tail serves both kernels: the extended kernel
// changes WHAT E_z/E_ρ are, not how they are weighted, and the four geometric
// factors fed in (td, rho_proj, t_n·p̂, ρ̂·p̂) are EFLD's, computed before NEC
// picks between EKSC and EKSCX and therefore identical in both — in
// particular ρ̂ and rho_proj use the UNSWAPPED radial distance RHX, never
// EKSCX's ordered RH. The numpy side says the same thing by construction:
// `_field_tensor_image_refl`'s `_project_weighted` is ONE expression, applied
// to whichever tables `_field_components_bcast` returned, and it is the oracle
// for the term placement below.
struct ReflPair {
    std::complex<double> rho_v;
    std::complex<double> rvh;  // ρ_v + ρ_h — → 0 in the PEC limit
    double tm_p;               // t_m · p̂
    double tn_p;               // t_n · p̂
    double rho_p;              // ρ̂ · p̂
    double td;                 // t_m · t_n
    double rho_proj;           // (rho_vec · t_m) / rho_eval
};

static inline ReflPair refl_pair(std::complex<double> eps_t, double ct,
                                 double tm_p, double tn_p, double rho_p,
                                 double td, double rho_proj) {
    std::complex<double> root = std::sqrt(eps_t - (1.0 - ct * ct));
    std::complex<double> rho_v = (eps_t * ct - root) / (eps_t * ct + root);
    std::complex<double> rho_h = (ct - root) / (ct + root);
    ReflPair p;
    p.rho_v = rho_v;
    p.rvh = rho_v + rho_h;
    p.tm_p = tm_p;
    p.tn_p = tn_p;
    p.rho_p = rho_p;
    p.td = td;
    p.rho_proj = rho_proj;
    return p;
}

static inline std::complex<double> refl_project(const ReflPair &p,
                                                std::complex<double> Ez,
                                                std::complex<double> Erho) {
    std::complex<double> tm_E = p.td * Ez + p.rho_proj * Erho;
    std::complex<double> E_p = p.tn_p * Ez + p.rho_p * Erho;
    return p.rho_v * tm_E - p.rvh * (p.tm_p * E_p);
}


// Sinusoidal-basis (NEC2 three-term) tangential-field tensor.
//
// For each (m=obs, n=src) pair of segments, compute the three scalar tensors
//   Phi_const[m, n] = ŝ_m · E^const_n(c_m)
//   Phi_sin  [m, n] = ŝ_m · E^sin_n  (c_m)
//   Phi_cos  [m, n] = ŝ_m · E^cos_n  (c_m)
// where the source's local frame is centered on segment n with z-axis along
// src_tangents[n]; the const/sin/cos sources are I(z')=1 / sin(k z') /
// cos(k z') over z' ∈ [-H_n, +H_n], H_n = h_n/2. Result is in NEC's natural-
// arc convention (σ accounting is the caller's job).
//
// Closed forms for the const-source `int G_0 dz'` are 1/r_0 singularity
// extraction: ∫ 1/r_0 dz' = arcsinh((H-z)/ρ) - arcsinh((-H-z)/ρ); regular
// remainder via Gauss-Legendre on the (G_0 - 1/r_0) integrand. Sin/cos
// sources are fully closed-form per Eqs 76-79 of the LLNL theory manual
// (mirrored by the numpy reference in src/momwire/sinusoidal.py _field_tensor).
//
// Parallelism: each (m, n) pair is independent. OpenMP collapse(2) over the
// (m, n) grid; per-n constants (H_n, sin(kH_n), cos(kH_n)) are precomputed
// outside the parallel region.
//
// REFL=true is the reflection-coefficient finite-ground variant
// (`sinusoidal_field_tensor_refl`): the caller passes the k-independent
// per-pair specular tables (cos_th, px, py, tm_p, tn_p — see
// _ground_refl.specular_ray_tables and SinusoidalSolver._image_refl_band,
// built fresh per observer band since momwire#332 unit B)
// plus the complex ε̃, and the projection tail applies NEC's field dyad
//     t_m · D · E = ρ_v·(t_m·E) − (ρ_v + ρ_h)·(t_m·p̂)·(E·p̂)
// with the Fresnel ρ_v/ρ_h computed in-kernel at each pair's specular
// angle (principal-branch sqrt, matching _ground_refl.fresnel_rho). The
// unprojected E_z/E_ρ physics (Eqs 76-79) is shared with REFL=false —
// this template exists so the reflection path stops paying the pure-numpy
// `_field_components` fill (~35× the kernel cost, super-quadratic once
// its (M,N,n_qp) temporaries fall out of cache).
template<bool REFL>
static std::tuple<py::array_t<std::complex<double>>,
                  py::array_t<std::complex<double>>,
                  py::array_t<std::complex<double>>>
sinusoidal_field_tensor_impl(
    py::array_t<double, py::array::c_style | py::array::forcecast> obs_centers,
    py::array_t<double, py::array::c_style | py::array::forcecast> obs_tangents,
    py::array_t<double, py::array::c_style | py::array::forcecast> src_centers,
    py::array_t<double, py::array::c_style | py::array::forcecast> src_tangents,
    py::array_t<double, py::array::c_style | py::array::forcecast> seg_h,
    double a, double k, double eta,
    py::array_t<double, py::array::c_style | py::array::forcecast> gl_t,
    py::array_t<double, py::array::c_style | py::array::forcecast> gl_w,
    py::array_t<double, py::array::c_style | py::array::forcecast> cos_th,
    py::array_t<double, py::array::c_style | py::array::forcecast> px_t,
    py::array_t<double, py::array::c_style | py::array::forcecast> py_t,
    py::array_t<double, py::array::c_style | py::array::forcecast> tm_p,
    py::array_t<double, py::array::c_style | py::array::forcecast> tn_p,
    std::complex<double> eps_t,
    uintptr_t cancel_flag = 0
) {
    auto oc = obs_centers.unchecked<2>();
    auto ot = obs_tangents.unchecked<2>();
    auto sc = src_centers.unchecked<2>();
    auto st = src_tangents.unchecked<2>();
    auto sh = seg_h.unchecked<1>();
    auto glt = gl_t.unchecked<1>();
    auto glw = gl_w.unchecked<1>();

    if (oc.shape(1) != 3 || ot.shape(1) != 3 ||
        sc.shape(1) != 3 || st.shape(1) != 3) {
        throw std::runtime_error("center/tangent arrays must have shape (N, 3)");
    }
    if (oc.shape(0) != ot.shape(0)) {
        throw std::runtime_error("obs_centers and obs_tangents must have matching N");
    }
    if (sc.shape(0) != st.shape(0) || sc.shape(0) != sh.shape(0)) {
        throw std::runtime_error("src arrays must all have matching N");
    }
    if (glt.shape(0) != glw.shape(0)) {
        throw std::runtime_error("gl_t and gl_w must have matching length");
    }

    size_t M = oc.shape(0);
    size_t N = sc.shape(0);
    size_t n_qp = glt.shape(0);

    // REFL-only per-pair specular tables, flat (M, N) row-major.
    const double *cth_p = nullptr, *px_p = nullptr, *py_p = nullptr;
    const double *tmp_p = nullptr, *tnp_p = nullptr;
    if (REFL) {
        for (const auto *t : {&cos_th, &px_t, &py_t, &tm_p, &tn_p}) {
            if ((*t).ndim() != 2 || (size_t)(*t).shape(0) != M ||
                (size_t)(*t).shape(1) != N) {
                throw std::runtime_error(
                    "refl tables must all have shape (M_obs, N_src)");
            }
        }
        cth_p = cos_th.data();
        px_p = px_t.data();
        py_p = py_t.data();
        tmp_p = tm_p.data();
        tnp_p = tn_p.data();
    }

    py::array_t<std::complex<double>> Phi_const({M, N});
    py::array_t<std::complex<double>> Phi_sin({M, N});
    py::array_t<std::complex<double>> Phi_cos({M, N});
    auto pc = Phi_const.mutable_unchecked<2>();
    auto ps = Phi_sin.mutable_unchecked<2>();
    auto pco = Phi_cos.mutable_unchecked<2>();

    // Phase 0: release the GIL for the heavy compute region below.
    py::gil_scoped_release release;

    // Per-source-segment precompute: H_n = h_n/2, sin(kH_n), cos(kH_n).
    std::vector<double> H_n(N), sin_kH(N), cos_kH(N);
    for (size_t n = 0; n < N; n++) {
        H_n[n] = 0.5 * sh(n);
        sin_kH[n] = std::sin(k * H_n[n]);
        cos_kH[n] = std::cos(k * H_n[n]);
    }

    // Cache GL nodes/weights in std::vector so the per-pair loops avoid
    // bouncing through the pybind11 unchecked accessor in the inner loop.
    std::vector<double> glt_v(n_qp), glw_v(n_qp);
    for (size_t q = 0; q < n_qp; q++) {
        glt_v[q] = glt(q);
        glw_v[q] = glw(q);
    }

    // Scalar prefactors. pref_z = +j eta / (4 pi k); pref_rho_const has the
    // same form (the per-pair 1/rho_eval scaling lives inside pref_rho).
    const double four_pi_k = 4.0 * M_PI * k;
    const double pref_z_im = eta / four_pi_k;
    const double pref_rho_const_im = -eta / four_pi_k;
    const double a_sq = a * a;

    // The oscillatory sincos is batched across all N source segments (per
    // observer m) into one flat buffer so it vectorizes — the per-pair scalar
    // calls were unvectorizable. Three stages per m: (A) geometry + phases,
    // (B) one omp-simd sincos sweep over the buffer, (C) assembly.
    //
    // Phases per source segment: 2 boundary (r0_2, r0_1) + n_qp quadrature nodes.
    const size_t S = n_qp + 2;
    const size_t P = N * S;

    MW_CANCEL_SETUP(cancel_flag);
    #pragma omp parallel for schedule(static)
    for (size_t m = 0; m < M; m++) {
        MW_CANCEL_POLL();
        // Per-iteration scratch (per-thread under the parallel-for). Sizes are
        // tiny (P ~ N*(n_qp+2)); allocation cost is negligible next to the
        // sincos + assembly work it feeds.
        std::vector<double> ph(P), cphb(P), sphb(P);
        std::vector<double> rho_eval_a(N), dz1_a(N), dz2_a(N),
                            r0_1_a(N), r0_2_a(N), td_a(N), rpf_a(N);
        std::vector<double> r0q_inv_a(N * n_qp);
        // REFL: ρ̂·p̂ per source segment for this observer row (ρ̂ =
        // rho_vec/rho_eval; p̂ is horizontal so only x/y contribute).
        std::vector<double> rho_p_a(REFL ? N : 0);

        double cmx = oc(m, 0), cmy = oc(m, 1), cmz = oc(m, 2);
        double tmx = ot(m, 0), tmy = ot(m, 1), tmz = ot(m, 2);

        // ---- Stage A: geometry + phase generation -------------------------
        for (size_t n = 0; n < N; n++) {
            double cnx = sc(n, 0), cny = sc(n, 1), cnz = sc(n, 2);
            double tnx = st(n, 0), tny = st(n, 1), tnz = st(n, 2);
            double rvx = cmx - cnx, rvy = cmy - cny, rvz = cmz - cnz;
            double z_eval = rvx * tnx + rvy * tny + rvz * tnz;
            double rho_vx = rvx - z_eval * tnx;
            double rho_vy = rvy - z_eval * tny;
            double rho_vz = rvz - z_eval * tnz;
            double rho_axis = std::sqrt(rho_vx*rho_vx + rho_vy*rho_vy + rho_vz*rho_vz);
            double rho_eval = std::sqrt(rho_axis*rho_axis + a_sq);
            double td = tmx*tnx + tmy*tny + tmz*tnz;
            double rho_dot_tobs = rho_vx*tmx + rho_vy*tmy + rho_vz*tmz;

            double H = H_n[n];
            double dz2 = z_eval - H;
            double dz1 = z_eval + H;
            double r0_2 = std::sqrt(rho_eval*rho_eval + dz2*dz2);
            double r0_1 = std::sqrt(rho_eval*rho_eval + dz1*dz1);

            rho_eval_a[n] = rho_eval;
            dz1_a[n] = dz1; dz2_a[n] = dz2;
            r0_1_a[n] = r0_1; r0_2_a[n] = r0_2;
            td_a[n] = td; rpf_a[n] = rho_dot_tobs / rho_eval;
            if (REFL) {
                size_t mn = m * N + n;
                rho_p_a[n] = (rho_vx * px_p[mn] + rho_vy * py_p[mn]) / rho_eval;
            }

            size_t base = n * S;
            ph[base + 0] = -k * r0_2;
            ph[base + 1] = -k * r0_1;
            for (size_t q = 0; q < n_qp; q++) {
                double z_q = H * glt_v[q];
                double dz_q = z_eval - z_q;
                double r0_q = std::sqrt(rho_eval*rho_eval + dz_q*dz_q);
                // HALF phase (momwire#799). This slice of the table serves
                // only the smooth remainder, whose real part is `cos − 1` =
                // −2 sin²(y/2); the full-angle sine comes back from the same
                // two values as 2 sin(y/2) cos(y/2), so nothing extra is
                // evaluated. Same reconstruction idiom as `IX_H2`/`IX_H1` in
                // the swept kernel below.
                ph[base + 2 + q] = -0.5 * k * r0_q;
                r0q_inv_a[n * n_qp + q] = 1.0 / r0_q;
            }
        }

        // ---- Stage B: vectorized sincos over every phase ------------------
        // Split cos and sin into separate omp-simd loops (libmvec has no vector
        // sincos) so each body stays vectorizable to _ZGVdN4v_{cos,sin}
        // (AVX2, 4 doubles per call).
        MW_OMP_SIMD()
        for (size_t i = 0; i < P; i++) cphb[i] = std::cos(ph[i]);
        MW_OMP_SIMD()
        for (size_t i = 0; i < P; i++) sphb[i] = std::sin(ph[i]);

        // ---- Stage C: assembly --------------------------------------------
        for (size_t n = 0; n < N; n++) {
            size_t base = n * S;
            double cph_2 = cphb[base + 0], sph_2 = sphb[base + 0];
            double cph_1 = cphb[base + 1], sph_1 = sphb[base + 1];
            double rho_eval = rho_eval_a[n];
            double dz1 = dz1_a[n], dz2 = dz2_a[n];
            double r0_1 = r0_1_a[n], r0_2 = r0_2_a[n];
            double td = td_a[n], rho_proj_factor = rpf_a[n];
            double H = H_n[n];
            double inv_r0_2 = 1.0 / r0_2;
            double inv_r0_1 = 1.0 / r0_1;
            double G0_2_re = cph_2 * inv_r0_2, G0_2_im = sph_2 * inv_r0_2;
            double G0_1_re = cph_1 * inv_r0_1, G0_1_im = sph_1 * inv_r0_1;

            // (1 + j k r0) / r0² split into re/im
            double inv_r0_2_sq = inv_r0_2 * inv_r0_2;
            double inv_r0_1_sq = inv_r0_1 * inv_r0_1;
            double one_jkr_2_re = inv_r0_2_sq;
            double one_jkr_2_im = k * r0_2 * inv_r0_2_sq;
            double one_jkr_1_re = inv_r0_1_sq;
            double one_jkr_1_im = k * r0_1 * inv_r0_1_sq;

            // ---- Const source -------------------------------------------------
            // Erho_const = pref_rho_const * (
            //     (1 + j k r0_2) * rho_eval * G0_2 / r0_2²
            //   - (1 + j k r0_1) * rho_eval * G0_1 / r0_1²
            // )
            // Let A_2 = (1 + jk r0_2) / r0_2² = one_jkr_2 (complex).
            // term_2 = A_2 * G0_2 — complex product.
            auto cmul = [](double ar, double ai, double br, double bi,
                           double &cr, double &ci) {
                cr = ar*br - ai*bi;
                ci = ar*bi + ai*br;
            };

            double term_const2_re, term_const2_im;
            cmul(one_jkr_2_re, one_jkr_2_im, G0_2_re, G0_2_im,
                 term_const2_re, term_const2_im);
            double term_const1_re, term_const1_im;
            cmul(one_jkr_1_re, one_jkr_1_im, G0_1_re, G0_1_im,
                 term_const1_re, term_const1_im);
            double rho_diff_re = rho_eval * (term_const2_re - term_const1_re);
            double rho_diff_im = rho_eval * (term_const2_im - term_const1_im);
            // pref_rho_const = j * pref_rho_const_im  (pure imaginary scalar)
            double Erho_const_re = -pref_rho_const_im * rho_diff_im;
            double Erho_const_im =  pref_rho_const_im * rho_diff_re;

            double inv_rho_eval = 1.0 / rho_eval;  // used by pref_rho below
            // int_inv_r0.  H - z_eval = -dz2;  -H - z_eval = -dz1. Through
            // `stable_asinh_diff` (momwire#799) — the literal difference of
            // two O(log) asinh values loses ε·|asinh|·z/(2H) on a collinear
            // deck's far pairs, 8.4e-13 at 801 segments.
            double int_inv_r0 = stable_asinh_diff(-dz1, -dz2,
                                                  rho_eval * rho_eval,
                                                  r0_1, r0_2);

            // Quadrature for the smooth remainder of int_G0:
            //   reg(q) = (exp(-jk r0_q) - 1) / r0_q,   r0_q = sqrt(ρ² + (z - H gx[q])²)
            //   int_reg = H * Σ_q reg(q) * gw[q]
            double int_reg_re = 0.0, int_reg_im = 0.0;
            for (size_t q = 0; q < n_qp; q++) {
                double hc_q = cphb[base + 2 + q];
                double hs_q = sphb[base + 2 + q];
                double inv_r0_q = r0q_inv_a[n * n_qp + q];
                // (exp(jphase) - 1) / r0, cancellation-free (momwire#799):
                // cos y − 1 = −2 sin²(y/2), sin y = 2 sin(y/2) cos(y/2).
                double reg_re = (-2.0 * hs_q * hs_q) * inv_r0_q;
                double reg_im = (2.0 * hs_q * hc_q) * inv_r0_q;
                int_reg_re += reg_re * glw_v[q];
                int_reg_im += reg_im * glw_v[q];
            }
            int_reg_re *= H;
            int_reg_im *= H;
            double int_G0_re = int_inv_r0 + int_reg_re;
            double int_G0_im = int_reg_im;

            // Ez_const_boundary = (1+jk r0_2) dz2 G0_2 / r0_2² - (1+jk r0_1) dz1 G0_1 / r0_1²
            double Ez_boundary_re = dz2 * term_const2_re - dz1 * term_const1_re;
            double Ez_boundary_im = dz2 * term_const2_im - dz1 * term_const1_im;
            double k_sq = k * k;
            // Ez_const = -pref_z * (Ez_boundary + k² int_G0). pref_z = j * pref_z_im.
            double inside_re = Ez_boundary_re + k_sq * int_G0_re;
            double inside_im = Ez_boundary_im + k_sq * int_G0_im;
            // Multiply by -j * pref_z_im
            double Ez_const_re =  pref_z_im * inside_im;
            double Ez_const_im = -pref_z_im * inside_re;

            // ---- Sine source (Eq 76, 77) --------------------------------------
            double sin2 = sin_kH[n];
            double cos2 = cos_kH[n];
            double sin1 = -sin2;
            double cos1 =  cos2;

            // bracket_sin_2 = G0_2 * (k dz2 cos2 + (1 - dz2² (1+jk r0_2)/r0_2²) sin2)
            double inner_2_re = 1.0 - dz2*dz2 * one_jkr_2_re;
            double inner_2_im =     - dz2*dz2 * one_jkr_2_im;
            double bracket_sin_2_re = k*dz2*cos2 + inner_2_re*sin2;
            double bracket_sin_2_im =              inner_2_im*sin2;
            double bsin2_re, bsin2_im;
            cmul(G0_2_re, G0_2_im, bracket_sin_2_re, bracket_sin_2_im,
                 bsin2_re, bsin2_im);

            double inner_1_re = 1.0 - dz1*dz1 * one_jkr_1_re;
            double inner_1_im =     - dz1*dz1 * one_jkr_1_im;
            double bracket_sin_1_re = k*dz1*cos1 + inner_1_re*sin1;
            double bracket_sin_1_im =              inner_1_im*sin1;
            double bsin1_re, bsin1_im;
            cmul(G0_1_re, G0_1_im, bracket_sin_1_re, bracket_sin_1_im,
                 bsin1_re, bsin1_im);
            double Erho_sin_inner_re = bsin2_re - bsin1_re;
            double Erho_sin_inner_im = bsin2_im - bsin1_im;
            // pref_rho = -j eta / (4 pi k rho_eval)
            double pref_rho_im = pref_rho_const_im * inv_rho_eval;
            double Erho_sin_re = -pref_rho_im * Erho_sin_inner_im;
            double Erho_sin_im =  pref_rho_im * Erho_sin_inner_re;

            // bracket_sin_z = G0 * (k cos - (1+jk r0) dz / r0² sin)
            double bracket_sin_z_2_re = k*cos2 - dz2*one_jkr_2_re*sin2;
            double bracket_sin_z_2_im =        - dz2*one_jkr_2_im*sin2;
            double bszin2_re, bszin2_im;
            cmul(G0_2_re, G0_2_im, bracket_sin_z_2_re, bracket_sin_z_2_im,
                 bszin2_re, bszin2_im);
            double bracket_sin_z_1_re = k*cos1 - dz1*one_jkr_1_re*sin1;
            double bracket_sin_z_1_im =        - dz1*one_jkr_1_im*sin1;
            double bszin1_re, bszin1_im;
            cmul(G0_1_re, G0_1_im, bracket_sin_z_1_re, bracket_sin_z_1_im,
                 bszin1_re, bszin1_im);
            double Ez_sin_inner_re = bszin2_re - bszin1_re;
            double Ez_sin_inner_im = bszin2_im - bszin1_im;
            // pref_z = +j pref_z_im
            double Ez_sin_re = -pref_z_im * Ez_sin_inner_im;
            double Ez_sin_im =  pref_z_im * Ez_sin_inner_re;

            // ---- Cosine source ------------------------------------------------
            // bracket_cos_2 = G0_2 * (-k dz2 sin2 + (1 - dz2² (1+jk r0_2)/r0_2²) cos2)
            double bracket_cos_2_re = -k*dz2*sin2 + inner_2_re*cos2;
            double bracket_cos_2_im =              inner_2_im*cos2;
            double bcos2_re, bcos2_im;
            cmul(G0_2_re, G0_2_im, bracket_cos_2_re, bracket_cos_2_im,
                 bcos2_re, bcos2_im);
            double bracket_cos_1_re = -k*dz1*sin1 + inner_1_re*cos1;
            double bracket_cos_1_im =              inner_1_im*cos1;
            double bcos1_re, bcos1_im;
            cmul(G0_1_re, G0_1_im, bracket_cos_1_re, bracket_cos_1_im,
                 bcos1_re, bcos1_im);
            double Erho_cos_inner_re = bcos2_re - bcos1_re;
            double Erho_cos_inner_im = bcos2_im - bcos1_im;
            double Erho_cos_re = -pref_rho_im * Erho_cos_inner_im;
            double Erho_cos_im =  pref_rho_im * Erho_cos_inner_re;

            double bracket_cos_z_2_re = -k*sin2 - dz2*one_jkr_2_re*cos2;
            double bracket_cos_z_2_im =         - dz2*one_jkr_2_im*cos2;
            double bczin2_re, bczin2_im;
            cmul(G0_2_re, G0_2_im, bracket_cos_z_2_re, bracket_cos_z_2_im,
                 bczin2_re, bczin2_im);
            double bracket_cos_z_1_re = -k*sin1 - dz1*one_jkr_1_re*cos1;
            double bracket_cos_z_1_im =         - dz1*one_jkr_1_im*cos1;
            double bczin1_re, bczin1_im;
            cmul(G0_1_re, G0_1_im, bracket_cos_z_1_re, bracket_cos_z_1_im,
                 bczin1_re, bczin1_im);
            double Ez_cos_inner_re = bczin2_re - bczin1_re;
            double Ez_cos_inner_im = bczin2_im - bczin1_im;
            double Ez_cos_re = -pref_z_im * Ez_cos_inner_im;
            double Ez_cos_im =  pref_z_im * Ez_cos_inner_re;

            if (REFL) {
                // Fresnel ρ_v / ρ_h at this pair's specular angle
                // (principal-branch sqrt — matches _ground_refl.fresnel_rho),
                // then NEC's field dyad:
                //   t_m · D · E = ρ_v·(t_m·E) − (ρ_v + ρ_h)·(t_m·p̂)·(E·p̂)
                //   t_m·E = td·E_z + rho_proj·E_ρ
                //   E·p̂  = (t_n·p̂)·E_z + (ρ̂·p̂)·E_ρ
                //
                // `refl_pair` / `refl_project` above are this same algebra,
                // and the extended kernel (momwire#259) calls them — but THIS
                // copy stays open-coded on purpose. Routing the reduced path
                // through the helpers is a byte-level change: the factors then
                // reach the multiply-adds through a struct and GCC contracts
                // them differently, which measured out at ≤3.6e-15 on the
                // tensor and 7.1e-14 on a grounded Z. The reduced refl path is
                // frozen armor (#233 gate 4 / #259 gate 3), so the duplication
                // is the cheaper price. Both copies are independently anchored
                // to the same numpy oracle, `_field_tensor_image_refl`'s
                // `_project_weighted` — that, not code sharing, is what keeps
                // them from drifting.
                size_t mn = m * N + n;
                double ct = cth_p[mn];
                std::complex<double> root =
                    std::sqrt(eps_t - (1.0 - ct * ct));
                std::complex<double> rho_v =
                    (eps_t * ct - root) / (eps_t * ct + root);
                std::complex<double> rho_h = (ct - root) / (ct + root);
                std::complex<double> rvh = rho_v + rho_h;
                double tmp = tmp_p[mn], tnp = tnp_p[mn], rp = rho_p_a[n];

                auto weighted = [&](double Ez_re, double Ez_im,
                                    double Erho_re, double Erho_im) {
                    std::complex<double> Ez(Ez_re, Ez_im);
                    std::complex<double> Erho(Erho_re, Erho_im);
                    std::complex<double> tm_E =
                        td * Ez + rho_proj_factor * Erho;
                    std::complex<double> E_p = tnp * Ez + rp * Erho;
                    return rho_v * tm_E - rvh * (tmp * E_p);
                };
                pc(m, n) = weighted(Ez_const_re, Ez_const_im,
                                    Erho_const_re, Erho_const_im);
                ps(m, n) = weighted(Ez_sin_re, Ez_sin_im,
                                    Erho_sin_re, Erho_sin_im);
                pco(m, n) = weighted(Ez_cos_re, Ez_cos_im,
                                     Erho_cos_re, Erho_cos_im);
            } else {
                // Project to obs tangent: Phi = td * Ez + rho_proj * Erho.
                pc(m, n) = std::complex<double>(
                    td * Ez_const_re + rho_proj_factor * Erho_const_re,
                    td * Ez_const_im + rho_proj_factor * Erho_const_im);
                ps(m, n) = std::complex<double>(
                    td * Ez_sin_re + rho_proj_factor * Erho_sin_re,
                    td * Ez_sin_im + rho_proj_factor * Erho_sin_im);
                pco(m, n) = std::complex<double>(
                    td * Ez_cos_re + rho_proj_factor * Erho_cos_re,
                    td * Ez_cos_im + rho_proj_factor * Erho_cos_im);
            }
        }
    }

    MW_THROW_IF_ABORTED();
    return std::make_tuple(Phi_const, Phi_sin, Phi_cos);
}

// Thin non-template wrappers for pybind registration. The plain tensor
// passes empty refl tables; the refl variant forwards them.
static std::tuple<py::array_t<std::complex<double>>,
                  py::array_t<std::complex<double>>,
                  py::array_t<std::complex<double>>>
sinusoidal_field_tensor(
    py::array_t<double, py::array::c_style | py::array::forcecast> obs_centers,
    py::array_t<double, py::array::c_style | py::array::forcecast> obs_tangents,
    py::array_t<double, py::array::c_style | py::array::forcecast> src_centers,
    py::array_t<double, py::array::c_style | py::array::forcecast> src_tangents,
    py::array_t<double, py::array::c_style | py::array::forcecast> seg_h,
    double a, double k, double eta,
    py::array_t<double, py::array::c_style | py::array::forcecast> gl_t,
    py::array_t<double, py::array::c_style | py::array::forcecast> gl_w,
    uintptr_t cancel_flag = 0
) {
    // Declared with the parameters' own flags (c_style | forcecast): pybind11
    // converted the extension's plain `array_t<double>` at the call, mw::nd
    // does not convert between flag sets (momwire#1607 B1). The array is
    // the same empty (0, 0) placeholder.
    py::array_t<double, py::array::c_style | py::array::forcecast> empty(std::vector<py::ssize_t>{0, 0});
    return sinusoidal_field_tensor_impl<false>(
        obs_centers, obs_tangents, src_centers, src_tangents, seg_h,
        a, k, eta, gl_t, gl_w,
        empty, empty, empty, empty, empty,
        std::complex<double>(0.0, 0.0), cancel_flag);
}

static std::tuple<py::array_t<std::complex<double>>,
                  py::array_t<std::complex<double>>,
                  py::array_t<std::complex<double>>>
sinusoidal_field_tensor_refl(
    py::array_t<double, py::array::c_style | py::array::forcecast> obs_centers,
    py::array_t<double, py::array::c_style | py::array::forcecast> obs_tangents,
    py::array_t<double, py::array::c_style | py::array::forcecast> src_centers,
    py::array_t<double, py::array::c_style | py::array::forcecast> src_tangents,
    py::array_t<double, py::array::c_style | py::array::forcecast> seg_h,
    double a, double k, double eta,
    py::array_t<double, py::array::c_style | py::array::forcecast> gl_t,
    py::array_t<double, py::array::c_style | py::array::forcecast> gl_w,
    py::array_t<double, py::array::c_style | py::array::forcecast> cos_th,
    py::array_t<double, py::array::c_style | py::array::forcecast> px,
    py::array_t<double, py::array::c_style | py::array::forcecast> py_,
    py::array_t<double, py::array::c_style | py::array::forcecast> tm_p,
    py::array_t<double, py::array::c_style | py::array::forcecast> tn_p,
    std::complex<double> eps_t,
    uintptr_t cancel_flag = 0
) {
    return sinusoidal_field_tensor_impl<true>(
        obs_centers, obs_tangents, src_centers, src_tangents, seg_h,
        a, k, eta, gl_t, gl_w,
        cos_th, px, py_, tm_p, tn_p, eps_t, cancel_flag);
}


// NEC's EXTENDED thin-wire kernel for the sinusoidal field tensor (#245).
//
// `sinusoidal_field_tensor` above is EKSC (nec2-1.2.1.2.f:3124-3166) — the
// REDUCED kernel, rearranged into per-endpoint brackets. This entry point is
// EKSCX (f.3170-3234): the same brackets, with the per-end reduced routine GX
// (f.4842-4852) swapped for the second-order tube-average GXX (f.4857-4897)
// wherever the fill's per-end gating says the source current really continues
// straight through that end, plus EKSCX's ungated (kB/2)² correction on the
// constant-current term. Theory manual Eqs 84-98 (Burke & Poggio Part I,
// pp. 30-34).
//
// This is a TRANSCRIPTION of `SinusoidalSolver._ek_end_gxx` / `_ek_end_gx` /
// `_extended_kernel_fields`, which stay the reference implementation and the
// oracle: tests/test_extended_kernel.py holds the two paths to 1e-13 on every
// gating branch. Nothing here may be "improved" relative to the numpy spelling
// without moving that oracle first — see the IRA note below.
//
// Two tables the reduced entry point does not take:
//   * `src_a` (N,) — the SOURCE segment's radius, NEC's BX = BI(J). The
//     reduced kernel only ever needs the OBSERVER's (necpp EFLD's `ai`, the
//     scalar `a` argument), which is why THAT entry point can serve mixed
//     per-wire radii one constant-radius observer-row run at a time. The O(a²)
//     expansion is about the SOURCE conductor's girth, so it needs the other
//     radius, per source, inside every run.
//   * `ind1` / `ind2` (N,) int8 — the per-source-end gating codes of
//     `_ek_gating` (f.2019-2053). Codes 0 and 1 take GXX, code 2 takes GX,
//     exactly as EKSCX routes at f.3193 (`IF( INX1.EQ.2) GOTO 3`) and f.3199.
//     They are computed in Python because they are a connectivity walk, not
//     arithmetic.
//
// EKSCX's IRA (f.3186-3192) is decided PER PAIR, in stage A, from this pair's
// own `rhx < src_a` — the same comparison that already orders (rh, b) right
// beside it. #245 could not do that: `_extended_kernel_fields` reduced the
// test to one `np.any` before calling `_ek_end_gxx`, so every pair took the
// same arm, and this kernel took a build-wide `want_swapped` scalar to
// reproduce that faithfully rather than silently repair it. momwire#258
// repaired it on both sides, so the scalar is GONE from this signature — the
// arm now rides the same local `sw` the ordering does, which is the only
// spelling in which the two cannot disagree at a knife-edge pair. Python
// detects the new arity through the `ek_ira_per_pair` module attribute (set
// in `register_sinusoidal`, at the end of this file); a stale extension
// exports the symbols with the old arity and is refused there rather than
// called wrongly.
//
// Parallelism / staging follow `sinusoidal_field_tensor_impl` exactly: OpenMP
// over observer rows, per-row scratch, and the same three stages (geometry +
// phase generation, one omp-simd sincos sweep over the row's phase buffer,
// assembly). Nothing (M, N, n_qp)-sized is ever materialized, which is the
// other half of #245 — the numpy path's source-quadrature temporaries are what
// make large EK meshes memory-bound.
//
// REFL=true (momwire#259) is the same marriage the reduced kernel's REFL arm
// makes: EKSCX's E_z/E_ρ tables with the Fresnel field dyad applied instead of
// the plain tangential projection, from the same k-independent specular tables
// (cos_th, px, py, tm_p, tn_p) and the same complex ε̃. It closes the last
// numpy-speed extended-kernel path on the sinusoidal family — before it,
// `extended_kernel=True` with `ground_eps` (and no `ground_model=
// "sommerfeld"`) took the pure-numpy image block. PEC and Sommerfeld grounds
// need nothing here: the PEC image is `sinusoidal_field_tensor_ek` with
// mirrored sources, and the Sommerfeld image block is that same PEC tensor
// scaled by the scalar C₂, so both already ride REFL=false.

// Minimal POD complex for the extended-kernel inner loop. libstdc++'s
// std::complex<double> multiplication lowers to a libgcc `__muldc3` CALL (the
// Inf/NaN fixup path) at our flag set, and each pair here does ~60 complex
// products; the naive four-multiply form below is what the reduced kernel's
// `cmul` lambda already open-codes inline.
struct EkC {
    double re, im;
};
static inline EkC operator+(EkC a, EkC b) { return EkC{a.re + b.re, a.im + b.im}; }
static inline EkC operator-(EkC a, EkC b) { return EkC{a.re - b.re, a.im - b.im}; }
static inline EkC operator-(EkC a) { return EkC{-a.re, -a.im}; }
static inline EkC operator+(double s, EkC a) { return EkC{s + a.re, a.im}; }
static inline EkC operator-(EkC a, double s) { return EkC{a.re - s, a.im}; }
static inline EkC operator*(EkC a, EkC b) {
    return EkC{a.re * b.re - a.im * b.im, a.re * b.im + a.im * b.re};
}
static inline EkC operator*(EkC a, double s) { return EkC{a.re * s, a.im * s}; }
static inline EkC operator*(double s, EkC a) { return EkC{s * a.re, s * a.im}; }
static inline EkC operator/(EkC a, double s) { return EkC{a.re / s, a.im / s}; }
// (j·s)·x. CON (f.3181) is pure imaginary, and numpy's complex product with a
// zero real part is bit-identical to this: 0·x is an exact zero and ±0 + y == y.
static inline EkC ek_mul_j(double s, EkC x) { return EkC{-s * x.im, s * x.re}; }

// The six per-end quantities EKSCX consumes, in NEC's argument order:
// (G1, G1P, G2, G2P, G3, GZP) — the extended scalar kernel (Eq 89), its
// z-derivative, the ρ-kernel and its ρ- and z-derivatives (Eqs 90-96), and the
// reduced-kernel z-derivative that only the constant-current correction uses.
struct EkEnd {
    EkC g1, g1p, g2, g2p, g3, gzp;
};

// GXX (f.4857-4897) — `SinusoidalSolver._ek_end_gxx`. `r2`/`r` are the already
// formed R² = zz² + rh² and R; `cph`/`sph` are cos(−kR)/sin(−kR) out of the
// batched sincos sweep. `rh >= b` is the caller's ordering (f.3186-3192), and
// `want_swapped` is THIS pair's IRA out of that same ordering (momwire#258).
static inline EkEnd ek_end_gxx(double k, double zz, double rh, double b,
                               double r2, double r, double cph, double sph,
                               bool want_swapped) {
    double kr = k * r;
    double kr2 = kr * kr;
    // C1, C2, C3 of f.4869-4871 — the polynomial coefficients of the
    // second-order Taylor terms of Eqs 86-88.
    EkC c1 = EkC{1.0, kr};
    EkC c2 = 3.0 * c1 - kr2;
    EkC c3 = EkC{6.0, kr} * kr2 - 15.0 * c1;
    double a2 = b * b;
    // T1, T2 of f.4867-4868: the two O(a²) shape factors of Eq 89.
    double rh2 = rh * rh;
    double r4 = r2 * r2;
    double t1 = 0.25 * a2 * rh2 / r4;
    double t2 = 0.5 * a2 / r2;
    EkC gz = EkC{cph / r, sph / r};  // reduced kernel e^{-jkR}/R
    // Eq 89: the circumferentially averaged kernel, ρ-flavoured (G2) and
    // z-flavoured (G1, which carries the extra -T2·C1 term).
    EkC g2 = gz * (1.0 + t1 * c2);
    EkC g1 = g2 - t2 * c1 * gz;
    EkC gzr = gz / r2;
    EkC g2p = gzr * (t1 * c3 - c1);
    EkC gzp_t = t2 * c2 * gzr;
    EkC g3 = g2p + gzp_t;
    EkEnd out;
    out.g1 = g1;
    out.g1p = g3 * zz;
    // GZP: the plain reduced-kernel z-derivative. It is a-independent and only
    // enters EKSCX's constant-current term, scaled by (ka/2)².
    out.gzp = (-zz) * c1 * gzr;
    if (want_swapped) {
        // IRA == 1 (f.4886-4896): observation point inside the conductor.
        double t2b = 0.5 * b;
        out.g2 = (-t2b) * c1 * gzr;
        EkC g2p_s = (t2b * gzr) * c2 / r2;
        out.g3 = rh2 * g2p_s - (b * gzr) * c1;
        out.g2p = g2p_s * zz;
    } else {
        // IRA == 0 (f.4879-4885), the ordinary case. `rh` is never zero in
        // momwire — it is at least the observer radius — so NEC's RH < 1e-10
        // guard at f.4881 is unreachable here.
        out.g3 = (g3 + gzp_t) * rh;
        out.g2 = g2 / rh;
        out.g2p = g2p * zz / rh;
    }
    return out;
}

// GX (f.4842-4852) repackaged into GXX's six-value contract —
// `SinusoidalSolver._ek_end_gx`. EKSCX's `INX == 2` arm (f.3195-3207) passes
// RHX, the ORIGINAL radial distance, not the possibly-swapped RH, and zeroes
// GZP so the end contributes nothing to the constant-current correction.
static inline EkEnd ek_end_gx(double zz, double rhx, double r2, double r,
                              double k, double cph, double sph) {
    double kr = k * r;
    EkC c1 = EkC{1.0, kr};
    EkC gz = EkC{cph / r, sph / r};
    EkC gp = (-c1) * gz / r2;
    EkC g1p = gp * zz;
    EkEnd out;
    out.g1 = gz;
    out.g1p = g1p;
    out.g2 = gz / rhx;
    out.g2p = g1p / rhx;
    out.g3 = gp * rhx;
    out.gzp = EkC{0.0, 0.0};
    return out;
}

// Per-(observer row, source) geometry carried from stage A to stage C.
struct EkGeomRow {
    double z_eval, rhx, rh, b, z1, z2;
    double r2_1, r_1, r2_2, r_2;
    double td, rpf;
    bool ext1, ext2;
    // EKSCX's IRA for THIS pair (momwire#258): the `rhx < src_a` that ordered
    // (rh, b) just above, carried through to the arm it selects.
    bool swapped;
};

template<bool REFL>
static std::tuple<py::array_t<std::complex<double>>,
                  py::array_t<std::complex<double>>,
                  py::array_t<std::complex<double>>>
sinusoidal_field_tensor_ek_impl(
    py::array_t<double, py::array::c_style | py::array::forcecast> obs_centers,
    py::array_t<double, py::array::c_style | py::array::forcecast> obs_tangents,
    py::array_t<double, py::array::c_style | py::array::forcecast> src_centers,
    py::array_t<double, py::array::c_style | py::array::forcecast> src_tangents,
    py::array_t<double, py::array::c_style | py::array::forcecast> seg_h,
    double a, double k, double eta,
    py::array_t<double, py::array::c_style | py::array::forcecast> gl_t,
    py::array_t<double, py::array::c_style | py::array::forcecast> gl_w,
    py::array_t<double, py::array::c_style | py::array::forcecast> src_a,
    py::array_t<int8_t, py::array::c_style | py::array::forcecast> ind1,
    py::array_t<int8_t, py::array::c_style | py::array::forcecast> ind2,
    py::array_t<double, py::array::c_style | py::array::forcecast> cos_th,
    py::array_t<double, py::array::c_style | py::array::forcecast> px_t,
    py::array_t<double, py::array::c_style | py::array::forcecast> py_t,
    py::array_t<double, py::array::c_style | py::array::forcecast> tm_p,
    py::array_t<double, py::array::c_style | py::array::forcecast> tn_p,
    std::complex<double> eps_t,
    uintptr_t cancel_flag = 0
) {
    auto oc = obs_centers.unchecked<2>();
    auto ot = obs_tangents.unchecked<2>();
    auto sc = src_centers.unchecked<2>();
    auto st = src_tangents.unchecked<2>();
    auto sh = seg_h.unchecked<1>();
    auto glt = gl_t.unchecked<1>();
    auto glw = gl_w.unchecked<1>();
    auto sa = src_a.unchecked<1>();
    auto i1 = ind1.unchecked<1>();
    auto i2 = ind2.unchecked<1>();

    if (oc.shape(1) != 3 || ot.shape(1) != 3 ||
        sc.shape(1) != 3 || st.shape(1) != 3) {
        throw std::runtime_error("center/tangent arrays must have shape (N, 3)");
    }
    if (oc.shape(0) != ot.shape(0)) {
        throw std::runtime_error("obs_centers and obs_tangents must have matching N");
    }
    if (sc.shape(0) != st.shape(0) || sc.shape(0) != sh.shape(0) ||
        sc.shape(0) != sa.shape(0) || sc.shape(0) != i1.shape(0) ||
        sc.shape(0) != i2.shape(0)) {
        throw std::runtime_error(
            "src arrays (centers, tangents, seg_h, src_a, ind1, ind2) must all "
            "have matching N");
    }
    if (glt.shape(0) != glw.shape(0)) {
        throw std::runtime_error("gl_t and gl_w must have matching length");
    }

    size_t M = oc.shape(0);
    size_t N = sc.shape(0);
    size_t n_qp = glt.shape(0);

    // REFL-only per-pair specular tables, flat (M, N) row-major — the same
    // contract `sinusoidal_field_tensor_impl<true>` takes.
    const double *cth_p = nullptr, *px_p = nullptr, *py_p = nullptr;
    const double *tmp_p = nullptr, *tnp_p = nullptr;
    if (REFL) {
        for (const auto *t : {&cos_th, &px_t, &py_t, &tm_p, &tn_p}) {
            if ((*t).ndim() != 2 || (size_t)(*t).shape(0) != M ||
                (size_t)(*t).shape(1) != N) {
                throw std::runtime_error(
                    "refl tables must all have shape (M_obs, N_src)");
            }
        }
        cth_p = cos_th.data();
        px_p = px_t.data();
        py_p = py_t.data();
        tmp_p = tm_p.data();
        tnp_p = tn_p.data();
    }

    py::array_t<std::complex<double>> Phi_const({M, N});
    py::array_t<std::complex<double>> Phi_sin({M, N});
    py::array_t<std::complex<double>> Phi_cos({M, N});
    auto pc = Phi_const.mutable_unchecked<2>();
    auto ps = Phi_sin.mutable_unchecked<2>();
    auto pco = Phi_cos.mutable_unchecked<2>();

    py::gil_scoped_release release;

    // Per-source-segment precompute: H_n = h_n/2, sin(kH_n), cos(kH_n) — NEC's
    // SS/CS at f.3183-3184.
    std::vector<double> H_n(N), sin_kH(N), cos_kH(N), sa_v(N);
    std::vector<unsigned char> ext1_v(N), ext2_v(N);
    for (size_t n = 0; n < N; n++) {
        H_n[n] = 0.5 * sh(n);
        sin_kH[n] = std::sin(k * H_n[n]);
        cos_kH[n] = std::cos(k * H_n[n]);
        sa_v[n] = sa(n);
        ext1_v[n] = (i1(n) != 2) ? 1 : 0;
        ext2_v[n] = (i2(n) != 2) ? 1 : 0;
    }

    std::vector<double> glt_v(n_qp), glw_v(n_qp);
    for (size_t q = 0; q < n_qp; q++) {
        glt_v[q] = glt(q);
        glw_v[q] = glw(q);
    }

    // CON of f.3181 — NEC's DATA CONX/0., 4.771341189/ is jη/(4πk) with NEC's
    // k = 2π. Pure imaginary, so only the magnitude is carried (see ek_mul_j).
    const double con_im = eta / (4.0 * M_PI * k);
    const double a_sq = a * a;
    const double k_sq = k * k;

    // Phases per source segment: 2 segment ends + n_qp const-term quadrature
    // nodes. Same layout as `sinusoidal_field_tensor_impl`, with slot 0 = end 1
    // (NEC's Z1 side) and slot 1 = end 2.
    const size_t S = n_qp + 2;
    const size_t P = N * S;

    MW_CANCEL_SETUP(cancel_flag);
    #pragma omp parallel for schedule(static)
    for (size_t m = 0; m < M; m++) {
        MW_CANCEL_POLL();
        std::vector<double> ph(P), cphb(P), sphb(P);
        std::vector<EkGeomRow> gmr(N);
        std::vector<double> r0q_inv_a(N * n_qp);
        // REFL: ρ̂·p̂ per source segment for this observer row, off the
        // UNSWAPPED rho_eval (= RHX) — p̂ is horizontal so only x/y contribute.
        std::vector<double> rho_p_a(REFL ? N : 0);

        double cmx = oc(m, 0), cmy = oc(m, 1), cmz = oc(m, 2);
        double tmx = ot(m, 0), tmy = ot(m, 1), tmz = ot(m, 2);

        // ---- Stage A: geometry + phase generation -------------------------
        for (size_t n = 0; n < N; n++) {
            double cnx = sc(n, 0), cny = sc(n, 1), cnz = sc(n, 2);
            double tnx = st(n, 0), tny = st(n, 1), tnz = st(n, 2);
            double rvx = cmx - cnx, rvy = cmy - cny, rvz = cmz - cnz;
            double z_eval = rvx * tnx + rvy * tny + rvz * tnz;
            double rho_vx = rvx - z_eval * tnx;
            double rho_vy = rvy - z_eval * tny;
            double rho_vz = rvz - z_eval * tnz;
            double rho_axis = std::sqrt(rho_vx*rho_vx + rho_vy*rho_vy + rho_vz*rho_vz);
            // EFLD's RHX: the a-regularized radial distance, on the OBSERVER's
            // radius. Shared with the reduced kernel — NEC computes it before
            // it chooses between EKSC and EKSCX (f.2919-2962), and the ρ̂
            // projection uses this UNSWAPPED distance either way.
            double rho_eval = std::sqrt(rho_axis*rho_axis + a_sq);
            double td = tmx*tnx + tmy*tny + tmz*tnz;
            double rho_dot_tobs = rho_vx*tmx + rho_vy*tmy + rho_vz*tmz;
            if (REFL) {
                size_t mn = m * N + n;
                rho_p_a[n] = (rho_vx * px_p[mn] + rho_vy * py_p[mn]) / rho_eval;
            }

            double H = H_n[n];
            double rhx = rho_eval;
            double src_an = sa_v[n];
            // f.3186-3192. One comparison decides both the ordering and the
            // IRA arm, per pair — see the IRA note at the top of this block.
            bool sw = rhx < src_an;
            double rh = sw ? src_an : rhx;
            double b = sw ? rhx : src_an;
            // NEC's Z2 = SH - Z and Z1 = -(SH + Z); momwire's dz_i is -Z_i.
            double z2 = H - z_eval;
            double z1 = -(H + z_eval);
            bool ext1 = ext1_v[n] != 0;
            bool ext2 = ext2_v[n] != 0;
            // GXX integrates at the ORDERED radius, GX at the original one.
            double rad1 = ext1 ? rh : rhx;
            double rad2 = ext2 ? rh : rhx;
            double r2_1 = z1*z1 + rad1*rad1;
            double r2_2 = z2*z2 + rad2*rad2;
            double r_1 = std::sqrt(r2_1);
            double r_2 = std::sqrt(r2_2);

            EkGeomRow &g = gmr[n];
            g.z_eval = z_eval; g.rhx = rhx; g.rh = rh; g.b = b;
            g.z1 = z1; g.z2 = z2;
            g.r2_1 = r2_1; g.r_1 = r_1; g.r2_2 = r2_2; g.r_2 = r_2;
            g.td = td; g.rpf = rho_dot_tobs / rho_eval;
            g.ext1 = ext1; g.ext2 = ext2;
            g.swapped = sw;

            size_t base = n * S;
            ph[base + 0] = -k * r_1;
            ph[base + 1] = -k * r_2;
            // INTX (f.3223-3227) integrates e^{-jkR}/R along the segment at the
            // ORDERED radius RH, not at RHX.
            for (size_t q = 0; q < n_qp; q++) {
                double z_q = H * glt_v[q];
                double dz_q = z_eval - z_q;
                double r0_q = std::sqrt(rh*rh + dz_q*dz_q);
                // HALF phase, for the remainder's stable spelling below
                // (momwire#799) — the reduced kernel's treatment verbatim.
                ph[base + 2 + q] = -0.5 * k * r0_q;
                r0q_inv_a[n * n_qp + q] = 1.0 / r0_q;
            }
        }

        // ---- Stage B: vectorized sincos over every phase ------------------
        MW_OMP_SIMD()
        for (size_t i = 0; i < P; i++) cphb[i] = std::cos(ph[i]);
        MW_OMP_SIMD()
        for (size_t i = 0; i < P; i++) sphb[i] = std::sin(ph[i]);

        // ---- Stage C: assembly --------------------------------------------
        for (size_t n = 0; n < N; n++) {
            const EkGeomRow &g = gmr[n];
            size_t base = n * S;
            double H = H_n[n];
            double ss = sin_kH[n];
            double cs = cos_kH[n];

            EkEnd e1 = g.ext1
                ? ek_end_gxx(k, g.z1, g.rh, g.b, g.r2_1, g.r_1,
                             cphb[base + 0], sphb[base + 0], g.swapped)
                : ek_end_gx(g.z1, g.rhx, g.r2_1, g.r_1, k,
                            cphb[base + 0], sphb[base + 0]);
            EkEnd e2 = g.ext2
                ? ek_end_gxx(k, g.z2, g.rh, g.b, g.r2_2, g.r_2,
                             cphb[base + 1], sphb[base + 1], g.swapped)
                : ek_end_gx(g.z2, g.rhx, g.r2_2, g.r_2, k,
                            cphb[base + 1], sphb[base + 1]);

            // f.3213-3222, verbatim — algebraically the same brackets the
            // reduced path spells per endpoint; only the G-quantities changed.
            EkC ez_sin = ek_mul_j(con_im, (e2.g1 - e1.g1) * cs * k
                                          - (e2.g1p + e1.g1p) * ss);
            EkC ez_cos = ek_mul_j(-con_im, (e2.g1 + e1.g1) * ss * k
                                           + (e2.g1p - e1.g1p) * cs);
            EkC erho_sin = ek_mul_j(
                -con_im,
                (g.z2 * e2.g2p + g.z1 * e1.g2p + e2.g2 + e1.g2) * ss
                    - (g.z2 * e2.g2 - g.z1 * e1.g2) * cs * k);
            EkC erho_cos = ek_mul_j(
                -con_im,
                (g.z2 * e2.g2p - g.z1 * e1.g2p + e2.g2 - e1.g2) * cs
                    + (g.z2 * e2.g2 + g.z1 * e1.g2) * ss * k);
            EkC erho_const = ek_mul_j(con_im, e2.g3 - e1.g3);

            // f.3223-3227: the constant-current term. INTX integrates
            // e^{-jkR}/R along the segment at RH, and the extended kernel
            // scales that integral by (1 - (kB/2)²) while adding back a
            // (kB/2)²-weighted reduced-kernel end difference — the O(a²) piece
            // of Eq 98 that the ρ-expansion of the axial term leaves behind.
            // Unlike the per-end substitutions this factor is NOT gated: it
            // applies whenever EK is on, even when both ends fell back to GX.
            // Spelled as two divisions, matching the numpy `(H - z)/rh` and
            // `(-H - z)/rh`, rather than one reciprocal and two products.
            const double rh2 = g.rh * g.rh;
            const double rr2 = std::sqrt(rh2 + g.z2 * g.z2);
            const double rr1 = std::sqrt(rh2 + g.z1 * g.z1);
            double int_inv_r0 = stable_asinh_diff(g.z1, g.z2, rh2, rr1, rr2);
            double int_reg_re = 0.0, int_reg_im = 0.0;
            for (size_t q = 0; q < n_qp; q++) {
                double inv_r0_q = r0q_inv_a[n * n_qp + q];
                double hc_q = cphb[base + 2 + q], hs_q = sphb[base + 2 + q];
                double reg_re = (-2.0 * hs_q * hs_q) * inv_r0_q;
                double reg_im = (2.0 * hs_q * hc_q) * inv_r0_q;
                int_reg_re += reg_re * glw_v[q];
                int_reg_im += reg_im * glw_v[q];
            }
            EkC int_G0 = EkC{int_inv_r0 + int_reg_re * H, int_reg_im * H};
            double bk = k * g.b;
            double bk2 = 0.25 * bk * bk;
            EkC ez_const = ek_mul_j(
                -con_im,
                e2.g1p - e1.g1p + (k_sq * (1.0 - bk2)) * int_G0
                    - bk2 * (e2.gzp - e1.gzp));

            double td = g.td, rpf = g.rpf;
            if (REFL) {
                // The Fresnel tail, byte-for-byte the reduced kernel's (see
                // `refl_pair` / `refl_project`). The EK tables replace E_z/E_ρ
                // and nothing else: `td`, `rpf` and `rho_p_a` are EFLD's
                // pre-kernel geometry, so the dyad sees the same four factors
                // it sees with EK off — which is exactly how the numpy path
                // spells it, `_field_tensor_image_refl`'s `_project_weighted`
                // consuming whichever tables `_field_components_bcast`
                // returned.
                size_t mn = m * N + n;
                ReflPair rp = refl_pair(eps_t, cth_p[mn], tmp_p[mn], tnp_p[mn],
                                        rho_p_a[n], td, rpf);
                auto weighted = [&](EkC ez, EkC erho) {
                    return refl_project(rp,
                                        std::complex<double>(ez.re, ez.im),
                                        std::complex<double>(erho.re, erho.im));
                };
                pc(m, n) = weighted(ez_const, erho_const);
                ps(m, n) = weighted(ez_sin, erho_sin);
                pco(m, n) = weighted(ez_cos, erho_cos);
            } else {
                // The consuming projection is the reduced kernel's:
                //   Phi = td·E_z + rho_proj·E_ρ   (`_field_tensor`'s tail).
                pc(m, n) = std::complex<double>(
                    td * ez_const.re + rpf * erho_const.re,
                    td * ez_const.im + rpf * erho_const.im);
                ps(m, n) = std::complex<double>(
                    td * ez_sin.re + rpf * erho_sin.re,
                    td * ez_sin.im + rpf * erho_sin.im);
                pco(m, n) = std::complex<double>(
                    td * ez_cos.re + rpf * erho_cos.re,
                    td * ez_cos.im + rpf * erho_cos.im);
            }
        }
    }

    MW_THROW_IF_ABORTED();
    return std::make_tuple(Phi_const, Phi_sin, Phi_cos);
}


// Thin non-template wrappers for pybind registration, mirroring the reduced
// kernel's pair: the plain EK entry point passes empty refl tables, the refl
// one forwards them.
static std::tuple<py::array_t<std::complex<double>>,
                  py::array_t<std::complex<double>>,
                  py::array_t<std::complex<double>>>
sinusoidal_field_tensor_ek(
    py::array_t<double, py::array::c_style | py::array::forcecast> obs_centers,
    py::array_t<double, py::array::c_style | py::array::forcecast> obs_tangents,
    py::array_t<double, py::array::c_style | py::array::forcecast> src_centers,
    py::array_t<double, py::array::c_style | py::array::forcecast> src_tangents,
    py::array_t<double, py::array::c_style | py::array::forcecast> seg_h,
    double a, double k, double eta,
    py::array_t<double, py::array::c_style | py::array::forcecast> gl_t,
    py::array_t<double, py::array::c_style | py::array::forcecast> gl_w,
    py::array_t<double, py::array::c_style | py::array::forcecast> src_a,
    py::array_t<int8_t, py::array::c_style | py::array::forcecast> ind1,
    py::array_t<int8_t, py::array::c_style | py::array::forcecast> ind2,
    uintptr_t cancel_flag = 0
) {
    // Declared with the parameters' own flags (c_style | forcecast): pybind11
    // converted the extension's plain `array_t<double>` at the call, mw::nd
    // does not convert between flag sets (momwire#1607 B1). The array is
    // the same empty (0, 0) placeholder.
    py::array_t<double, py::array::c_style | py::array::forcecast> empty(std::vector<py::ssize_t>{0, 0});
    return sinusoidal_field_tensor_ek_impl<false>(
        obs_centers, obs_tangents, src_centers, src_tangents, seg_h,
        a, k, eta, gl_t, gl_w, src_a, ind1, ind2,
        empty, empty, empty, empty, empty,
        std::complex<double>(0.0, 0.0), cancel_flag);
}

static std::tuple<py::array_t<std::complex<double>>,
                  py::array_t<std::complex<double>>,
                  py::array_t<std::complex<double>>>
sinusoidal_field_tensor_ek_refl(
    py::array_t<double, py::array::c_style | py::array::forcecast> obs_centers,
    py::array_t<double, py::array::c_style | py::array::forcecast> obs_tangents,
    py::array_t<double, py::array::c_style | py::array::forcecast> src_centers,
    py::array_t<double, py::array::c_style | py::array::forcecast> src_tangents,
    py::array_t<double, py::array::c_style | py::array::forcecast> seg_h,
    double a, double k, double eta,
    py::array_t<double, py::array::c_style | py::array::forcecast> gl_t,
    py::array_t<double, py::array::c_style | py::array::forcecast> gl_w,
    py::array_t<double, py::array::c_style | py::array::forcecast> src_a,
    py::array_t<int8_t, py::array::c_style | py::array::forcecast> ind1,
    py::array_t<int8_t, py::array::c_style | py::array::forcecast> ind2,
    py::array_t<double, py::array::c_style | py::array::forcecast> cos_th,
    py::array_t<double, py::array::c_style | py::array::forcecast> px,
    py::array_t<double, py::array::c_style | py::array::forcecast> py_,
    py::array_t<double, py::array::c_style | py::array::forcecast> tm_p,
    py::array_t<double, py::array::c_style | py::array::forcecast> tn_p,
    std::complex<double> eps_t,
    uintptr_t cancel_flag = 0
) {
    return sinusoidal_field_tensor_ek_impl<true>(
        obs_centers, obs_tangents, src_centers, src_tangents, seg_h,
        a, k, eta, gl_t, gl_w, src_a, ind1, ind2,
        cos_th, px, py_, tm_p, tn_p, eps_t, cancel_flag);
}

// `cexp_i`: the point-matched complex twin's exp(j z), in its scalar
// spelling (momwire#1222); moved here from beside the Galerkin twin's
// body, which never calls it.
static inline std::complex<double> cexp_i(std::complex<double> z) {
    return std::exp(std::complex<double>(0.0, 1.0) * z);
}

// ---------------------------------------------------------------------------
// momwire#1222: the complex-k twin of the point-matched `sinusoidal_field_tensor`.
//
// The buried point-matched lane (#1220 stage 1) fills at k_m = k0*sqrt(eps~),
// Im k_m <= 0, with eta_m. The real kernel takes `double k` and so every
// complex fill fell to the numpy reference. This is the same tensor — Phi[3,
// M, N], the tangential field of each source segment's three shapes at each
// observer — with complex k and eta.
//
// Its per-pair arithmetic is `galerkin_far_fill_cplx_impl`'s (step E, gated
// against numpy at 1e-12), with that body's test reduction removed: one
// observer is one row. Kept a SEPARATE body, not a shared inline: the real
// kernels' bytes are frozen armor and a shared helper moves them through
// inlining (momwire#1193). `folded` selects the third shape exactly as the
// numpy `cos_shape` does: `false` is the literal cos k(xi) (the real
// kernel's, spelled complex), `true` the #205 folded cos k(xi) - 1.
//
// `obs_radius` is per OBSERVER (the necpp EFLD convention the real kernel
// serves one constant-radius run at a time), so mixed radii need no runs here.
// `seg_h` is the source FULL length, as in `sinusoidal_field_tensor`.
static std::tuple<py::array_t<std::complex<double>>,
                  py::array_t<std::complex<double>>,
                  py::array_t<std::complex<double>>>
sinusoidal_field_tensor_cplx(
    py::array_t<double, py::array::c_style | py::array::forcecast> obs_centers,
    py::array_t<double, py::array::c_style | py::array::forcecast> obs_tangents,
    py::array_t<double, py::array::c_style | py::array::forcecast> obs_radius,
    py::array_t<double, py::array::c_style | py::array::forcecast> src_centers,
    py::array_t<double, py::array::c_style | py::array::forcecast> src_tangents,
    py::array_t<double, py::array::c_style | py::array::forcecast> seg_h,
    std::complex<double> k, std::complex<double> eta,
    py::array_t<double, py::array::c_style | py::array::forcecast> gl_t,
    py::array_t<double, py::array::c_style | py::array::forcecast> gl_w,
    bool folded,
    uintptr_t cancel_flag = 0
) {
    typedef std::complex<double> C;
    const C J(0.0, 1.0);

    auto oc = obs_centers.unchecked<2>();
    auto ot = obs_tangents.unchecked<2>();
    auto ar = obs_radius.unchecked<1>();
    auto sc = src_centers.unchecked<2>();
    auto st = src_tangents.unchecked<2>();
    auto sh = seg_h.unchecked<1>();
    auto glt = gl_t.unchecked<1>();
    auto glw = gl_w.unchecked<1>();

    if (oc.shape(1) != 3 || ot.shape(1) != 3 ||
        sc.shape(1) != 3 || st.shape(1) != 3) {
        throw std::runtime_error("center/tangent arrays must have shape (N, 3)");
    }
    if (oc.shape(0) != ot.shape(0) || oc.shape(0) != ar.shape(0)) {
        throw std::runtime_error(
            "obs_centers, obs_tangents and obs_radius must have matching M");
    }
    if (sc.shape(0) != st.shape(0) || sc.shape(0) != sh.shape(0)) {
        throw std::runtime_error("src arrays must all have matching N");
    }
    if (glt.shape(0) != glw.shape(0)) {
        throw std::runtime_error("gl_t and gl_w must have matching length");
    }
    if (k.imag() > 0.0) {
        throw std::runtime_error(
            "complex k with Im k > 0: e^{+jwt} requires Im k <= 0 so that "
            "e^{-jkR} decays");
    }

    size_t M = oc.shape(0);
    size_t N = sc.shape(0);
    size_t n_qp = glt.shape(0);

    py::array_t<std::complex<double>> Phi_const({M, N});
    py::array_t<std::complex<double>> Phi_sin({M, N});
    py::array_t<std::complex<double>> Phi_cos({M, N});
    C *pc = Phi_const.mutable_data();
    C *ps = Phi_sin.mutable_data();
    C *pco = Phi_cos.mutable_data();

    py::gil_scoped_release release;

    std::vector<double> H_n(N);
    std::vector<C> sin_kH(N), cos_kH(N), cos_kH_m1(N), smarg_kH(N);
    for (size_t n = 0; n < N; n++) {
        H_n[n] = 0.5 * sh(n);
        C kH = k * H_n[n];
        sin_kH[n] = std::sin(kH);
        cos_kH[n] = std::cos(kH);
        C hs = std::sin(0.5 * kH);
        cos_kH_m1[n] = -2.0 * hs * hs;
        smarg_kH[n] = sin_minus_arg(kH);
    }
    std::vector<double> glt_v(n_qp), glw_v(n_qp), gl_step(n_qp);
    std::vector<char> gl_near2(n_qp);
    double w_hi = 0.0, w_lo = 0.0;
    for (size_t q = 0; q < n_qp; q++) {
        glt_v[q] = glt(q);
        glw_v[q] = glw(q);
        gl_near2[q] = glt_v[q] >= 0.0;
        gl_step[q] = gl_near2[q] ? (1.0 - glt_v[q]) : -(1.0 + glt_v[q]);
        (gl_near2[q] ? w_hi : w_lo) += glw_v[q];
    }
    w_hi -= 1.0;
    w_lo -= 1.0;

    const C four_pi_k = 4.0 * M_PI * k;
    const C pref_z = eta / four_pi_k;
    const C pref_rho_const = -eta / four_pi_k;
    const C k_sq = k * k;

    MW_CANCEL_SETUP(cancel_flag);
    #pragma omp parallel for schedule(static)
    for (size_t m = 0; m < M; m++) {
        MW_CANCEL_POLL();
        double cmx = oc(m, 0), cmy = oc(m, 1), cmz = oc(m, 2);
        double tmx = ot(m, 0), tmy = ot(m, 1), tmz = ot(m, 2);
        double a_sq = ar(m) * ar(m);
        for (size_t n = 0; n < N; n++) {
            double cnx = sc(n, 0), cny = sc(n, 1), cnz = sc(n, 2);
            double tnx = st(n, 0), tny = st(n, 1), tnz = st(n, 2);
            double rvx = cmx - cnx, rvy = cmy - cny, rvz = cmz - cnz;
            double z_eval = rvx * tnx + rvy * tny + rvz * tnz;
            double rho_vx = rvx - z_eval * tnx;
            double rho_vy = rvy - z_eval * tny;
            double rho_vz = rvz - z_eval * tnz;
            double rho_axis =
                std::sqrt(rho_vx*rho_vx + rho_vy*rho_vy + rho_vz*rho_vz);
            double rho_eval = std::sqrt(rho_axis*rho_axis + a_sq);
            double td = tmx*tnx + tmy*tny + tmz*tnz;
            double rho_proj_factor = (rho_vx*tmx + rho_vy*tmy + rho_vz*tmz)
                                     / rho_eval;
            double H = H_n[n];
            double dz2 = z_eval - H;
            double dz1 = z_eval + H;
            double rho2 = rho_eval * rho_eval;
            double r0_2 = std::sqrt(rho2 + dz2*dz2);
            double r0_1 = std::sqrt(rho2 + dz1*dz1);
            double inv_r0_2 = 1.0 / r0_2, inv_r0_1 = 1.0 / r0_1;

            C ph2 = -0.5 * k * r0_2;
            C ph1 = -0.5 * k * r0_1;
            C eh2 = cexp_i(ph2), eh1 = cexp_i(ph1);
            C ef2 = eh2 * eh2, ef1 = eh1 * eh1;   // e^{-jkr}

            C G0_2 = ef2 * inv_r0_2;
            C G0_1 = ef1 * inv_r0_1;
            C one_jkr_2 = (1.0 + J * k * r0_2) * (inv_r0_2 * inv_r0_2);
            C one_jkr_1 = (1.0 + J * k * r0_1) * (inv_r0_1 * inv_r0_1);

            // ---- Const source (Eqs 78, 79) -------------------------------
            C term_const2 = one_jkr_2 * G0_2;
            C term_const1 = one_jkr_1 * G0_1;
            C rho_diff = rho_eval * (term_const2 - term_const1);
            C Erho_const = J * pref_rho_const * rho_diff;

            double int_inv_r0 = stable_asinh_diff(-dz1, -dz2, rho2, r0_1, r0_2);
            C int_reg(0.0, 0.0);
            for (size_t q = 0; q < n_qp; q++) {
                double z_q = H * glt_v[q];
                double dz_q = z_eval - z_q;
                double r0_q = std::sqrt(rho2 + dz_q*dz_q);
                double inv_r0_q = 1.0 / r0_q;
                double dz_ref = gl_near2[q] ? dz2 : dz1;
                double r_ref  = gl_near2[q] ? r0_2 : r0_1;
                double delta = H * gl_step[q] * (dz_q + dz_ref) / (r0_q + r_ref);
                C ph_d = -0.5 * k * delta;
                C ph_q = (gl_near2[q] ? ph2 : ph1) + ph_d;
                C em1_q = 2.0 * J * std::sin(ph_q) * cexp_i(ph_q);
                int_reg += em1_q * inv_r0_q * glw_v[q];
            }
            int_reg *= H;
            C int_G0 = int_inv_r0 + int_reg;

            C Ez_boundary = dz2 * term_const2 - dz1 * term_const1;
            C inside = Ez_boundary + k_sq * int_G0;
            C Ez_const = -J * pref_z * inside;

            // ---- Sine source (Eqs 76, 77) --------------------------------
            C sin2 = sin_kH[n], cos2 = cos_kH[n];
            C sin1 = -sin2, cos1 = cos2;
            C inner_2 = 1.0 - dz2*dz2 * one_jkr_2;
            C inner_1 = 1.0 - dz1*dz1 * one_jkr_1;
            C bsin2 = G0_2 * (k*dz2*cos2 + inner_2*sin2);
            C bsin1 = G0_1 * (k*dz1*cos1 + inner_1*sin1);
            C pref_rho = pref_rho_const / rho_eval;
            C Erho_sin = J * pref_rho * (bsin2 - bsin1);
            C bszin2 = G0_2 * (k*cos2 - dz2*one_jkr_2*sin2);
            C bszin1 = G0_1 * (k*cos1 - dz1*one_jkr_1*sin1);
            C Ez_sin = J * pref_z * (bszin2 - bszin1);

            C Erho_cos, Ez_cos;
            if (!folded) {
                // ---- Literal cosine source (the real kernel's, complex) --
                C bcos2 = G0_2 * (-k*dz2*sin2 + inner_2*cos2);
                C bcos1 = G0_1 * (-k*dz1*sin1 + inner_1*cos1);
                Erho_cos = J * pref_rho * (bcos2 - bcos1);
                C bczin2 = G0_2 * (-k*sin2 - dz2*one_jkr_2*cos2);
                C bczin1 = G0_1 * (-k*sin1 - dz1*one_jkr_1*cos1);
                Ez_cos = J * pref_z * (bczin2 - bczin1);
            } else {
                // ---- Folded source (I = cos k(xi) - 1), #205 --------------
                C s_h2 = std::sin(ph2), s_h1 = std::sin(ph1);
                C phi_ang = 2.0 * k * H * z_eval / (r0_1 + r0_2);
                C cm1 = cos_kH_m1[n];
                double X = (dz1 * dz2 >= 0.0)
                    ? 2.0 * H * (dz1 + dz2) / (dz1 * r0_2 + dz2 * r0_1)
                    : (dz1 * r0_2 - dz2 * r0_1) / rho2;
                double t_asx = std::asinh(X);
                double t_sing =
                    (std::fabs(X) < 1.0)
                        ? asinh_minus_arg_from_t(t_asx)
                              + H * rho2 * X * X / ((r0_1 + r0_2) * r0_1 * r0_2)
                        : t_asx - H * (inv_r0_1 + inv_r0_2);
                C g2 = 2.0 * J * s_h2 * eh2 * inv_r0_2;
                C g1 = 2.0 * J * s_h1 * eh1 * inv_r0_1;
                C m_reg = w_hi * g2 + w_lo * g1;
                for (size_t q = 0; q < n_qp; q++) {
                    double z_q = H * glt_v[q];
                    double dz_q = z_eval - z_q;
                    double r0_q = std::sqrt(rho2 + dz_q*dz_q);
                    double inv_r0_q = 1.0 / r0_q;
                    bool hi = gl_near2[q];
                    double dz_ref = hi ? dz2 : dz1;
                    double r_ref  = hi ? r0_2 : r0_1;
                    double delta = H * gl_step[q] * (dz_q + dz_ref) / (r0_q + r_ref);
                    C ph_d = -0.5 * k * delta;
                    C em1 = 2.0 * J * std::sin(ph_d) * cexp_i(ph_d);
                    C e_ref = hi ? ef2 : ef1;
                    C gr = hi ? g2 : g1;
                    double w = glw_v[q] * inv_r0_q;
                    m_reg += w * (e_ref * em1 - gr * delta);
                }
                C smarg = smarg_kH[n];
                C d_int = t_sing + H * m_reg - (smarg / k) * (G0_1 + G0_2);
                C inner_cos = k_sq * d_int - cm1 * Ez_boundary;
                Ez_cos = J * pref_z * inner_cos;
                C kH = k * H;
                C A_ang = kH + phi_ang, B_ang = kH - phi_ang;
                double d_lin = -8.0 * H * H * H * z_eval * rho2
                               / ((rho2 + dz1 * dz2 + r0_1 * r0_2)
                                  * (r0_1 + r0_2) * r0_1 * r0_2);
                C cph_p = std::cos(phi_ang), sph_p = std::sin(phi_ang);
                C w_even =
                    (A_ang * sin_minus_arg(B_ang) - B_ang * sin_minus_arg(A_ang))
                        / kH
                    + (d_lin / H) * sin2 * cph_p;
                C w_odd = sin2 * (-(rho2 * X) / (r0_1 * r0_2)) * sph_p;
                C W = (ef2 * cexp_i(-phi_ang)) * (w_even + J * w_odd);
                C b_rho = -k * W + rho2 * cm1 * (term_const2 - term_const1);
                Erho_cos = J * pref_rho * b_rho;
            }

            size_t mn = m * N + n;
            pc[mn]  = td * Ez_const + rho_proj_factor * Erho_const;
            ps[mn]  = td * Ez_sin   + rho_proj_factor * Erho_sin;
            pco[mn] = td * Ez_cos   + rho_proj_factor * Erho_cos;
        }
    }
    MW_THROW_IF_ABORTED();
    return std::make_tuple(Phi_const, Phi_sin, Phi_cos);
}

// ===========================================================================
// The entry points: the bound functions, by name (`_sinusoidal_core.h`).

namespace mw {
namespace sinusoidal {

std::tuple<py::array_t<std::complex<double>>, py::array_t<std::complex<double>>, py::array_t<std::complex<double>>>
sinusoidal_field_tensor(
    py::array_t<double, py::array::c_style | py::array::forcecast> obs_centers,
    py::array_t<double, py::array::c_style | py::array::forcecast> obs_tangents,
    py::array_t<double, py::array::c_style | py::array::forcecast> src_centers,
    py::array_t<double, py::array::c_style | py::array::forcecast> src_tangents,
    py::array_t<double, py::array::c_style | py::array::forcecast> seg_h,
    double a,
    double k,
    double eta,
    py::array_t<double, py::array::c_style | py::array::forcecast> gl_t,
    py::array_t<double, py::array::c_style | py::array::forcecast> gl_w,
    uintptr_t cancel_flag) {
    return ::sinusoidal_field_tensor(obs_centers, obs_tangents, src_centers, src_tangents, seg_h, a, k, eta, gl_t, gl_w, cancel_flag);
}

std::tuple<py::array_t<std::complex<double>>, py::array_t<std::complex<double>>, py::array_t<std::complex<double>>>
sinusoidal_field_tensor_refl(
    py::array_t<double, py::array::c_style | py::array::forcecast> obs_centers,
    py::array_t<double, py::array::c_style | py::array::forcecast> obs_tangents,
    py::array_t<double, py::array::c_style | py::array::forcecast> src_centers,
    py::array_t<double, py::array::c_style | py::array::forcecast> src_tangents,
    py::array_t<double, py::array::c_style | py::array::forcecast> seg_h,
    double a,
    double k,
    double eta,
    py::array_t<double, py::array::c_style | py::array::forcecast> gl_t,
    py::array_t<double, py::array::c_style | py::array::forcecast> gl_w,
    py::array_t<double, py::array::c_style | py::array::forcecast> cos_th,
    py::array_t<double, py::array::c_style | py::array::forcecast> px,
    py::array_t<double, py::array::c_style | py::array::forcecast> py_,
    py::array_t<double, py::array::c_style | py::array::forcecast> tm_p,
    py::array_t<double, py::array::c_style | py::array::forcecast> tn_p,
    std::complex<double> eps_t,
    uintptr_t cancel_flag) {
    return ::sinusoidal_field_tensor_refl(obs_centers, obs_tangents, src_centers, src_tangents, seg_h, a, k, eta, gl_t, gl_w, cos_th, px, py_, tm_p, tn_p, eps_t, cancel_flag);
}

std::tuple<py::array_t<std::complex<double>>, py::array_t<std::complex<double>>, py::array_t<std::complex<double>>>
sinusoidal_field_tensor_ek(
    py::array_t<double, py::array::c_style | py::array::forcecast> obs_centers,
    py::array_t<double, py::array::c_style | py::array::forcecast> obs_tangents,
    py::array_t<double, py::array::c_style | py::array::forcecast> src_centers,
    py::array_t<double, py::array::c_style | py::array::forcecast> src_tangents,
    py::array_t<double, py::array::c_style | py::array::forcecast> seg_h,
    double a,
    double k,
    double eta,
    py::array_t<double, py::array::c_style | py::array::forcecast> gl_t,
    py::array_t<double, py::array::c_style | py::array::forcecast> gl_w,
    py::array_t<double, py::array::c_style | py::array::forcecast> src_a,
    py::array_t<int8_t, py::array::c_style | py::array::forcecast> ind1,
    py::array_t<int8_t, py::array::c_style | py::array::forcecast> ind2,
    uintptr_t cancel_flag) {
    return ::sinusoidal_field_tensor_ek(obs_centers, obs_tangents, src_centers, src_tangents, seg_h, a, k, eta, gl_t, gl_w, src_a, ind1, ind2, cancel_flag);
}

std::tuple<py::array_t<std::complex<double>>, py::array_t<std::complex<double>>, py::array_t<std::complex<double>>>
sinusoidal_field_tensor_ek_refl(
    py::array_t<double, py::array::c_style | py::array::forcecast> obs_centers,
    py::array_t<double, py::array::c_style | py::array::forcecast> obs_tangents,
    py::array_t<double, py::array::c_style | py::array::forcecast> src_centers,
    py::array_t<double, py::array::c_style | py::array::forcecast> src_tangents,
    py::array_t<double, py::array::c_style | py::array::forcecast> seg_h,
    double a,
    double k,
    double eta,
    py::array_t<double, py::array::c_style | py::array::forcecast> gl_t,
    py::array_t<double, py::array::c_style | py::array::forcecast> gl_w,
    py::array_t<double, py::array::c_style | py::array::forcecast> src_a,
    py::array_t<int8_t, py::array::c_style | py::array::forcecast> ind1,
    py::array_t<int8_t, py::array::c_style | py::array::forcecast> ind2,
    py::array_t<double, py::array::c_style | py::array::forcecast> cos_th,
    py::array_t<double, py::array::c_style | py::array::forcecast> px,
    py::array_t<double, py::array::c_style | py::array::forcecast> py_,
    py::array_t<double, py::array::c_style | py::array::forcecast> tm_p,
    py::array_t<double, py::array::c_style | py::array::forcecast> tn_p,
    std::complex<double> eps_t,
    uintptr_t cancel_flag) {
    return ::sinusoidal_field_tensor_ek_refl(obs_centers, obs_tangents, src_centers, src_tangents, seg_h, a, k, eta, gl_t, gl_w, src_a, ind1, ind2, cos_th, px, py_, tm_p, tn_p, eps_t, cancel_flag);
}

std::tuple<py::array_t<std::complex<double>>, py::array_t<std::complex<double>>, py::array_t<std::complex<double>>>
sinusoidal_field_tensor_cplx(
    py::array_t<double, py::array::c_style | py::array::forcecast> obs_centers,
    py::array_t<double, py::array::c_style | py::array::forcecast> obs_tangents,
    py::array_t<double, py::array::c_style | py::array::forcecast> obs_radius,
    py::array_t<double, py::array::c_style | py::array::forcecast> src_centers,
    py::array_t<double, py::array::c_style | py::array::forcecast> src_tangents,
    py::array_t<double, py::array::c_style | py::array::forcecast> seg_h,
    std::complex<double> k,
    std::complex<double> eta,
    py::array_t<double, py::array::c_style | py::array::forcecast> gl_t,
    py::array_t<double, py::array::c_style | py::array::forcecast> gl_w,
    bool folded,
    uintptr_t cancel_flag) {
    return ::sinusoidal_field_tensor_cplx(obs_centers, obs_tangents, obs_radius, src_centers, src_tangents, seg_h, k, eta, gl_t, gl_w, folded, cancel_flag);
}

}  // namespace sinusoidal
}  // namespace mw
