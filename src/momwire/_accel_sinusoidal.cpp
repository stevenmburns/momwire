#include "_accel_common.h"

// sinusoidal section of the former _accelerators.cpp monolith (momwire#687).
// Code below is byte-identical to the monolith's lines 3349-5401.

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

    PYSIM_CANCEL_SETUP(cancel_flag);
    #pragma omp parallel for schedule(static)
    for (size_t m = 0; m < M; m++) {
        PYSIM_CANCEL_POLL();
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
                ph[base + 2 + q] = -k * r0_q;
                r0q_inv_a[n * n_qp + q] = 1.0 / r0_q;
            }
        }

        // ---- Stage B: vectorized sincos over every phase ------------------
        // Split cos and sin into separate omp-simd loops (libmvec has no vector
        // sincos) so each body stays vectorizable to _ZGVdN4v_{cos,sin}
        // (AVX2, 4 doubles per call).
        PYSIM_OMP_SIMD()
        for (size_t i = 0; i < P; i++) cphb[i] = std::cos(ph[i]);
        PYSIM_OMP_SIMD()
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

            // u2, u1, int_inv_r0.  H - z_eval = -dz2;  -H - z_eval = -dz1.
            double inv_rho_eval = 1.0 / rho_eval;
            double u2 = -dz2 * inv_rho_eval;
            double u1 = -dz1 * inv_rho_eval;
            double int_inv_r0 = std::asinh(u2) - std::asinh(u1);

            // Quadrature for the smooth remainder of int_G0:
            //   reg(q) = (exp(-jk r0_q) - 1) / r0_q,   r0_q = sqrt(ρ² + (z - H gx[q])²)
            //   int_reg = H * Σ_q reg(q) * gw[q]
            double int_reg_re = 0.0, int_reg_im = 0.0;
            for (size_t q = 0; q < n_qp; q++) {
                double cph_q = cphb[base + 2 + q];
                double sph_q = sphb[base + 2 + q];
                double inv_r0_q = r0q_inv_a[n * n_qp + q];
                // (exp(jphase) - 1) / r0
                double reg_re = (cph_q - 1.0) * inv_r0_q;
                double reg_im = sph_q * inv_r0_q;
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

    PYSIM_THROW_IF_ABORTED();
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
    py::array_t<double> empty(std::vector<py::ssize_t>{0, 0});
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
// detects the new arity through the `ek_ira_per_pair` module attribute below;
// a stale extension exports the symbols with the old arity and is refused
// there rather than called wrongly.
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

    PYSIM_CANCEL_SETUP(cancel_flag);
    #pragma omp parallel for schedule(static)
    for (size_t m = 0; m < M; m++) {
        PYSIM_CANCEL_POLL();
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
                ph[base + 2 + q] = -k * r0_q;
                r0q_inv_a[n * n_qp + q] = 1.0 / r0_q;
            }
        }

        // ---- Stage B: vectorized sincos over every phase ------------------
        PYSIM_OMP_SIMD()
        for (size_t i = 0; i < P; i++) cphb[i] = std::cos(ph[i]);
        PYSIM_OMP_SIMD()
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
            double int_inv_r0 = std::asinh(g.z2 / g.rh) - std::asinh(g.z1 / g.rh);
            double int_reg_re = 0.0, int_reg_im = 0.0;
            for (size_t q = 0; q < n_qp; q++) {
                double inv_r0_q = r0q_inv_a[n * n_qp + q];
                double reg_re = (cphb[base + 2 + q] - 1.0) * inv_r0_q;
                double reg_im = sphb[base + 2 + q] * inv_r0_q;
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

    PYSIM_THROW_IF_ABORTED();
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
    py::array_t<double> empty(std::vector<py::ssize_t>{0, 0});
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


// Fused far fill for the sinusoidal GALERKIN solver (momwire#194).
//
// The Galerkin row for test segment m is a quadrature over m, so its field
// evaluation is the tensor above with the observers at m's nq Gauss points
// instead of its centre — and then reduced against the test weights:
//
//   contrib[e, n] = Σ_q w_entry[e, q] · Phi[m(e), q, n]
//
// where `e` runs over the support entries (segment, basis) of the CSR basis
// view and Phi is the plainly-projected tensor of the three source shapes.
// Evaluating Phi and reducing it in ONE kernel is the whole point: the numpy
// path has to materialize the (rows·nq, N) tables (and the (rows·nq, N,
// n_qp_const) source-quadrature scratch under them) before it can contract
// the q axis away, which is what forces the Python fill to block over test
// segments at all. Here nothing but the (nnz, N) result is ever stored.
//
// Serves the PLAIN projection only — the free-space block and the PEC-image
// block (`src_*` mirrored), which is where the fill time is. The
// reflection-coefficient projector's per-pair Fresnel tables and the
// Sommerfeld remainder stay on the numpy path.
//
// Parallelism: over test segments m. Each m owns rows starts[m]:starts[m+1]
// of the output exclusively, so the accumulation needs no reduction and no
// atomics; the rows are contiguous, so they stay cache-resident across the
// nq observers that write them. Per observer the kernel repeats
// `sinusoidal_field_tensor_impl`'s three stages (geometry+phases, one
// omp-simd sincos sweep, assembly).
//
// THIRD SHAPE (momwire#205): the const and sin blocks are the point-matched
// kernel's arithmetic verbatim; the third is the (cos kξ − 1) source shape,
// not cos kξ — the folded contract the Galerkin product pairs with σC. It is
// a transcription of `SinusoidalSolver._folded_cos_fields`, whose docstring
// carries the derivation and the reason the literal difference cannot be
// used. The two helpers below are that method's `_sin_minus_arg` /
// `_asinh_minus_arg`, and the phase table grows to carry the half-angles the
// spelling needs (see `S` at the top of the fill).

// sin(u) − u, without the u²/6 cancellation of the literal difference.
static inline double sin_minus_arg(double u) {
    double u2 = u * u;
    if (std::fabs(u) < 0.1) {
        return -(u * u2) / 6.0 *
               (1.0 - u2 / 20.0 *
                          (1.0 - u2 / 42.0 *
                                     (1.0 - u2 / 72.0 * (1.0 - u2 / 110.0))));
    }
    return std::sin(u) - u;
}

// asinh(x) − x, taking the caller's t = asinh(x) so the fill pays for one
// asinh either way. The series is in t because sinh t − t converges
// factorially where the asinh series does not.
static inline double asinh_minus_arg_from_t(double t) {
    double t2 = t * t;
    if (std::fabs(t) < 1.0) {
        return -(t * t2) / 6.0 *
               (1.0 + t2 / 20.0 *
                          (1.0 + t2 / 42.0 *
                                     (1.0 + t2 / 72.0 *
                                                (1.0 + t2 / 110.0 *
                                                           (1.0 + t2 / 156.0)))));
    }
    return -(std::sinh(t) - t);
}

// The extended kernel's payload for the fused far fill (momwire#246 unit C).
// `src_a` is one radius per SOURCE segment (N) and `gx`/`gw` are the composite
// sinh-mapped rule `SinusoidalSolver._ek_delta_rule` built, passed in rather
// than rebuilt here so the two backends integrate against the same nodes by
// construction.
//
// Eligibility arrives as the pair rule's GROUP LABELS rather than as its mask
// (momwire#358): `g_obs` per TEST segment (M) and `g_src` per source segment
// (N), with the pair rule evaluated in the sweep as
//
//     eligible(observer row of test segment m, source n)
//         = g_obs[m] >= 0 && g_obs[m] == g_src[n]
//
// which is `SinusoidalGalerkinSolver._ek_pairs`' `(gi == gj) & (gi >= 0)`
// verbatim. The mask this replaces was an (M*nq, N) bool — 11.5 MB of
// resident input at N = 1200 — whose rows were constant over one test
// segment's nq observers by construction, because the caller built it by
// indexing the observer axis through `m_of_obs = repeat(arange(M), nq)` and
// the kernel already requires `obs_centers` to have exactly M*nq rows in that
// order. So the labels carry the same information at 8·(M + N) bytes, the
// blocked numpy fill's own spelling (its per-block mask has always been a
// slice of the same predicate), and the test segment's label is hoisted out
// of the source loop instead of being re-read from memory per pair.
struct GalerkinEkBlock {
    const double *src_a;
    const int64_t *g_obs;
    const int64_t *g_src;
    const double *gx;
    const double *gw;
    size_t n_gl;
};

// The FOLD destination for the fused far fill (momwire#356). Without one the
// fill allocates its own three (nnz, N) arrays and the caller differences them
// off its own triple afterwards, so any accelerated grounded block floors at
// two triples live. With one, the caller's arrays ARE the output buffers and
// each test segment's rows are folded on as
//
//     dst[e, n] += scale * value[e, n]
//
// with `scale` on the LEFT of the complex product — `sinusoidal.py`'s
// documented C2 convention, because complex multiply evaluates the imaginary
// part as x.re*y.im + x.im*y.re and swapping the operands reorders that sum
// and moves the last bit.
//
// The value folded on is the FULLY reduced entry, not a running partial: the
// test-quadrature accumulation runs into a per-test-segment scratch band and
// the fold happens once, after it. Accumulating `dst -= t_qt` node by node
// instead would be `((dst − t1) − t2) − …` where the caller's spelling is
// `dst − (t1 + t2 + …)`, which is a reassociation and not a fold.
//
// `scale` with a zero imaginary part takes a real path — `dst.re += s·v.re`,
// `dst.im += s·v.im`. At the fold's own scale, −1, that is bit-for-bit
// `np.subtract(dst, value, out=dst)`: IEEE addition of an exactly-negated
// operand is the subtraction, signed zeros included. The general complex path
// spells numpy's `np.multiply(scale, value)` term for term.
//
// A cancelled fill leaves the destination partly folded. That is the same
// contract the allocating path has — the caller drops what the kernel was
// building when `SolveAborted` comes out of it — one array further along.
struct GalerkinFoldBlock {
    py::array_t<std::complex<double>> dst_const, dst_sin, dst_cos;
    double s_re, s_im;
};

// One implementation, two instantiations. `WITH_EK` is a compile-time
// constant, so the reduced entry point below compiles with every line of the
// delta gone — which is what keeps `sinusoidal_galerkin_far_fill` byte-frozen
// against its pre-#246 build (gate G-C2) while the two paths stay one piece of
// code. The alternative — a copied 450-line twin — freezes the reduced fill
// just as well and rots twice as fast.
template <bool WITH_EK>
static std::tuple<py::array_t<std::complex<double>>,
                  py::array_t<std::complex<double>>,
                  py::array_t<std::complex<double>>>
galerkin_far_fill_impl(
    py::array_t<double, py::array::c_style | py::array::forcecast> obs_centers,
    py::array_t<double, py::array::c_style | py::array::forcecast> obs_tangents,
    py::array_t<double, py::array::c_style | py::array::forcecast> obs_radius,
    py::array_t<double, py::array::c_style | py::array::forcecast> src_centers,
    py::array_t<double, py::array::c_style | py::array::forcecast> src_tangents,
    py::array_t<double, py::array::c_style | py::array::forcecast> src_hh,
    double k, double eta,
    py::array_t<double, py::array::c_style | py::array::forcecast> gl_t,
    py::array_t<double, py::array::c_style | py::array::forcecast> gl_w,
    py::array_t<std::complex<double>,
                py::array::c_style | py::array::forcecast> w_entry,
    py::array_t<int64_t, py::array::c_style | py::array::forcecast> starts,
    uintptr_t cancel_flag,
    const GalerkinEkBlock *ekb,
    const GalerkinFoldBlock *fold
) {
    auto oc = obs_centers.unchecked<2>();
    auto ot = obs_tangents.unchecked<2>();
    auto ar = obs_radius.unchecked<1>();
    auto sc = src_centers.unchecked<2>();
    auto st = src_tangents.unchecked<2>();
    auto sh = src_hh.unchecked<1>();
    auto glt = gl_t.unchecked<1>();
    auto glw = gl_w.unchecked<1>();
    auto we = w_entry.unchecked<2>();
    auto st_ = starts.unchecked<1>();

    if (oc.shape(1) != 3 || ot.shape(1) != 3 ||
        sc.shape(1) != 3 || st.shape(1) != 3) {
        throw std::runtime_error("center/tangent arrays must have shape (N, 3)");
    }
    if (sc.shape(0) != st.shape(0) || sc.shape(0) != sh.shape(0)) {
        throw std::runtime_error("src arrays must all have matching N");
    }
    if (glt.shape(0) != glw.shape(0)) {
        throw std::runtime_error("gl_t and gl_w must have matching length");
    }
    if (st_.shape(0) < 1) {
        throw std::runtime_error("starts must have length n_test_segments + 1");
    }

    size_t M = (size_t)st_.shape(0) - 1;          // test segments
    size_t nq = (size_t)we.shape(1);              // test quadrature nodes
    size_t nnz = (size_t)we.shape(0);             // CSR support entries
    size_t N = (size_t)sc.shape(0);               // source segments
    size_t n_qp = (size_t)glt.shape(0);           // source quadrature nodes

    if ((size_t)oc.shape(0) != M * nq || (size_t)ot.shape(0) != M * nq ||
        (size_t)ar.shape(0) != M * nq) {
        throw std::runtime_error(
            "obs_centers/obs_tangents/obs_radius must have M*nq rows");
    }
    if ((size_t)st_(st_.shape(0) - 1) != nnz) {
        throw std::runtime_error("starts[-1] must equal w_entry's row count");
    }

    // Folding, the caller's arrays ARE the output buffers — nothing is
    // allocated here and nothing is returned that the caller did not already
    // hold. Not folding, the three arrays below are this fill's own, exactly
    // as they were before momwire#356.
    const bool folding = (fold != nullptr);
    py::array_t<std::complex<double>> out_const =
        folding ? fold->dst_const : py::array_t<std::complex<double>>({nnz, N});
    py::array_t<std::complex<double>> out_sin =
        folding ? fold->dst_sin : py::array_t<std::complex<double>>({nnz, N});
    py::array_t<std::complex<double>> out_cos =
        folding ? fold->dst_cos : py::array_t<std::complex<double>>({nnz, N});
    const double fold_re = folding ? fold->s_re : 1.0;
    const double fold_im = folding ? fold->s_im : 0.0;
    const bool fold_real = (fold_im == 0.0);
    std::complex<double> *oc_p = out_const.mutable_data();
    std::complex<double> *os_p = out_sin.mutable_data();
    std::complex<double> *oco_p = out_cos.mutable_data();
    const std::complex<double> *w_p = w_entry.data();

    py::gil_scoped_release release;

    std::vector<double> H_n(N), sin_kH(N), cos_kH(N), cos_kH_m1(N), smarg_kH(N);
    for (size_t n = 0; n < N; n++) {
        H_n[n] = sh(n);
        sin_kH[n] = std::sin(k * H_n[n]);
        cos_kH[n] = std::cos(k * H_n[n]);
        // cos kH − 1 and sin kH − kH, both at their own size (#205).
        double hs = std::sin(0.5 * k * H_n[n]);
        cos_kH_m1[n] = -2.0 * hs * hs;
        smarg_kH[n] = sin_minus_arg(k * H_n[n]);
    }
    std::vector<double> glt_v(n_qp), glw_v(n_qp);
    // Which endpoint each source-quadrature node is referred to (#205), and
    // the two halves' weight sums — the rule's own departure from 1 each,
    // which is part of the value the const shape already carries.
    std::vector<double> gl_step(n_qp);
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

    const double four_pi_k = 4.0 * M_PI * k;
    const double pref_z_im = eta / four_pi_k;
    const double pref_rho_const_im = -eta / four_pi_k;

    // Phase table per source segment. The folded third shape (#205) needs
    // HALF angles the const and sin shapes do not: cos kr − 1 has to be
    // −2sin²(kr/2) rather than a subtraction, and each node's e^{−jkδ_q} − 1
    // is taken at that node's offset δ_q from its own endpoint. So the table
    // carries the halves and derives the full angles from them by the
    // double-angle identities — e^{−jkr} = (e^{−jkr/2})², two multiplies —
    // rather than evaluating both. That keeps the sweep at n_qp + 3 phases
    // against the pre-#205 n_qp + 2: the whole cost of the folded shape is
    // one extra sincos per (observer, source) pair, for the E_ρ reference
    // angle φ = k(r₁−r₂)/2. Evaluating the full angles as well measured 2×
    // slower on the N=2401 dipole, and the sweep is most of the kernel.
    //
    // The reconstruction costs the const and sin shapes ~1 ulp on their
    // Green's values against the numpy path's own exp() — the two paths have
    // agreed at reassociation level since #194 and still do.
    const size_t IX_H2 = 0;         // half-phase of r₀_2
    const size_t IX_H1 = 1;         // half-phase of r₀_1
    const size_t IX_PHI = 2;        // φ
    const size_t IX_DQ = 3;         // half-phases of the node offsets
    const size_t S = n_qp + 3;
    const size_t P = N * S;

    PYSIM_CANCEL_SETUP(cancel_flag);
    #pragma omp parallel for schedule(static)
    for (size_t m = 0; m < M; m++) {
        PYSIM_CANCEL_POLL();
        size_t e0 = (size_t)st_(m), e1 = (size_t)st_(m + 1);
        if (e1 == e0) continue;  // segment carries no basis support

        // This test segment's extended-kernel group label, read once per
        // segment rather than per pair (momwire#358). Every observer row this
        // iteration fills belongs to segment `m`, so the eligibility the mask
        // used to carry is `g_obs_m` against each source's own label — see
        // `GalerkinEkBlock`. `-1` is the never-extend value of the label
        // convention, so the reduced instantiation (null `ekb`) reaches the
        // delta's guard with a label that can never match.
        const int64_t g_obs_m = WITH_EK ? ekb->g_obs[m] : (int64_t)-1;

        // Rows e0:e1 belong to this test segment alone, so the accumulation
        // below owns them outright and needs them zeroed first.
        //
        // Not folding, they are zeroed IN the output — which is why the fill
        // allocates an uninitialized array and not a zeroed one. Folding, the
        // output already holds the caller's own values and the accumulation
        // runs into a scratch band of this segment's rows instead, so that
        // the fold below sees each entry's finished sum exactly once
        // (momwire#356). The band is one test segment wide — a few support
        // entries by N — so it is a working set and not a triple: at N = 400
        // with three entries a segment it is 58 kB a thread against the
        // 23 MB triple it replaces.
        size_t nrows = e1 - e0;
        std::vector<std::complex<double>> band;
        std::complex<double> *bc, *bs, *bco;
        if (folding) {
            band.assign(3 * nrows * N, std::complex<double>(0.0, 0.0));
            bc = band.data();
            bs = bc + nrows * N;
            bco = bs + nrows * N;
        } else {
            bc = oc_p + e0 * N;
            bs = os_p + e0 * N;
            bco = oco_p + e0 * N;
            std::fill(bc, bc + nrows * N, std::complex<double>(0.0, 0.0));
            std::fill(bs, bs + nrows * N, std::complex<double>(0.0, 0.0));
            std::fill(bco, bco + nrows * N, std::complex<double>(0.0, 0.0));
        }

        std::vector<double> ph(P), cphb(P), sphb(P);
        std::vector<double> rho_eval_a(N), dz1_a(N), dz2_a(N),
                            r0_1_a(N), r0_2_a(N), td_a(N), rpf_a(N), z_a(N);
        std::vector<double> r0q_inv_a(N * n_qp), delta_a(N * n_qp);
        // Projected field of the three source shapes at ONE observer, split
        // re/im so the reduction below stays vectorizable.
        std::vector<double> phi_c_re(N), phi_c_im(N), phi_s_re(N), phi_s_im(N),
                            phi_co_re(N), phi_co_im(N);

        for (size_t qt = 0; qt < nq; qt++) {
            size_t o = m * nq + qt;
            double cmx = oc(o, 0), cmy = oc(o, 1), cmz = oc(o, 2);
            double tmx = ot(o, 0), tmy = ot(o, 1), tmz = ot(o, 2);
            double a_sq = ar(o) * ar(o);

            // ---- Stage A: geometry + phase generation ---------------------
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
                double rho_dot_tobs = rho_vx*tmx + rho_vy*tmy + rho_vz*tmz;

                double H = H_n[n];
                double dz2 = z_eval - H;
                double dz1 = z_eval + H;
                double r0_2 = std::sqrt(rho_eval*rho_eval + dz2*dz2);
                double r0_1 = std::sqrt(rho_eval*rho_eval + dz1*dz1);

                rho_eval_a[n] = rho_eval;
                z_a[n] = z_eval;
                dz1_a[n] = dz1; dz2_a[n] = dz2;
                r0_1_a[n] = r0_1; r0_2_a[n] = r0_2;
                td_a[n] = td; rpf_a[n] = rho_dot_tobs / rho_eval;

                size_t base = n * S;
                ph[base + IX_H2] = -0.5 * k * r0_2;
                ph[base + IX_H1] = -0.5 * k * r0_1;
                // φ from r₁ − r₂ = 4Hz/(r₁+r₂), which is exact where the
                // subtraction would not be (#205).
                ph[base + IX_PHI] = 2.0 * k * H * z_eval / (r0_1 + r0_2);
                for (size_t q = 0; q < n_qp; q++) {
                    double z_q = H * glt_v[q];
                    double dz_q = z_eval - z_q;
                    double r0_q = std::sqrt(rho_eval*rho_eval + dz_q*dz_q);
                    r0q_inv_a[n * n_qp + q] = 1.0 / r0_q;
                    // δ_q = r_q − r_ref for the endpoint on the node's own
                    // side. Δz_q − Δz_ref is ±H(1 ∓ t_q) exactly — the
                    // observer cancels out of it — so δ_q is exact however
                    // far away the observer sits.
                    double dz_ref = gl_near2[q] ? dz2 : dz1;
                    double r_ref = gl_near2[q] ? r0_2 : r0_1;
                    double delta = H * gl_step[q] * (dz_q + dz_ref)
                                   / (r0_q + r_ref);
                    delta_a[n * n_qp + q] = delta;
                    ph[base + IX_DQ + q] = -0.5 * k * delta;
                }
            }

            // ---- Stage B: vectorized sincos over every phase --------------
            PYSIM_OMP_SIMD()
            for (size_t i = 0; i < P; i++) cphb[i] = std::cos(ph[i]);
            PYSIM_OMP_SIMD()
            for (size_t i = 0; i < P; i++) sphb[i] = std::sin(ph[i]);

            // ---- Stage C: assembly + plain tangential projection ----------
            for (size_t n = 0; n < N; n++) {
                size_t base = n * S;
                // Full boundary phases from their halves: e^{−jkr} =
                // (e^{−jkr/2})². The halves are what the folded shape needs
                // (below), and squaring is cheaper than a second sincos.
                double ch2 = cphb[base + IX_H2], sh2 = sphb[base + IX_H2];
                double ch1 = cphb[base + IX_H1], sh1 = sphb[base + IX_H1];
                double cph_2 = ch2 * ch2 - sh2 * sh2, sph_2 = 2.0 * ch2 * sh2;
                double cph_1 = ch1 * ch1 - sh1 * sh1, sph_1 = 2.0 * ch1 * sh1;
                double rho_eval = rho_eval_a[n];
                double dz1 = dz1_a[n], dz2 = dz2_a[n];
                double r0_1 = r0_1_a[n], r0_2 = r0_2_a[n];
                double td = td_a[n], rho_proj_factor = rpf_a[n];
                double H = H_n[n];
                double inv_r0_2 = 1.0 / r0_2;
                double inv_r0_1 = 1.0 / r0_1;
                double G0_2_re = cph_2 * inv_r0_2, G0_2_im = sph_2 * inv_r0_2;
                double G0_1_re = cph_1 * inv_r0_1, G0_1_im = sph_1 * inv_r0_1;

                double inv_r0_2_sq = inv_r0_2 * inv_r0_2;
                double inv_r0_1_sq = inv_r0_1 * inv_r0_1;
                double one_jkr_2_re = inv_r0_2_sq;
                double one_jkr_2_im = k * r0_2 * inv_r0_2_sq;
                double one_jkr_1_re = inv_r0_1_sq;
                double one_jkr_1_im = k * r0_1 * inv_r0_1_sq;

                auto cmul = [](double ar_, double ai, double br, double bi,
                               double &cr, double &ci) {
                    cr = ar_*br - ai*bi;
                    ci = ar_*bi + ai*br;
                };

                // ---- Const source (Eqs 78, 79) ---------------------------
                double term_const2_re, term_const2_im;
                cmul(one_jkr_2_re, one_jkr_2_im, G0_2_re, G0_2_im,
                     term_const2_re, term_const2_im);
                double term_const1_re, term_const1_im;
                cmul(one_jkr_1_re, one_jkr_1_im, G0_1_re, G0_1_im,
                     term_const1_re, term_const1_im);
                double rho_diff_re = rho_eval * (term_const2_re - term_const1_re);
                double rho_diff_im = rho_eval * (term_const2_im - term_const1_im);
                double Erho_const_re = -pref_rho_const_im * rho_diff_im;
                double Erho_const_im =  pref_rho_const_im * rho_diff_re;

                double inv_rho_eval = 1.0 / rho_eval;
                double u2 = -dz2 * inv_rho_eval;
                double u1 = -dz1 * inv_rho_eval;
                double int_inv_r0 = std::asinh(u2) - std::asinh(u1);

                double int_reg_re = 0.0, int_reg_im = 0.0;
                for (size_t q = 0; q < n_qp; q++) {
                    // e^{−jkr_q} = e^{−jkr_ref}·(e^{−jkδ_q/2})², the node's
                    // own endpoint carrying the reference phase (#205). The
                    // half-angle is the one the folded shape needs below.
                    double cd = cphb[base + IX_DQ + q];
                    double sd = sphb[base + IX_DQ + q];
                    double ed_re = cd * cd - sd * sd;
                    double ed_im = 2.0 * cd * sd;
                    bool hi = gl_near2[q];
                    double er = hi ? cph_2 : cph_1;
                    double ei = hi ? sph_2 : sph_1;
                    double cph_q = er * ed_re - ei * ed_im;
                    double sph_q = er * ed_im + ei * ed_re;
                    double inv_r0_q = r0q_inv_a[n * n_qp + q];
                    int_reg_re += (cph_q - 1.0) * inv_r0_q * glw_v[q];
                    int_reg_im += sph_q * inv_r0_q * glw_v[q];
                }
                int_reg_re *= H;
                int_reg_im *= H;
                double int_G0_re = int_inv_r0 + int_reg_re;
                double int_G0_im = int_reg_im;

                double Ez_boundary_re = dz2 * term_const2_re - dz1 * term_const1_re;
                double Ez_boundary_im = dz2 * term_const2_im - dz1 * term_const1_im;
                double k_sq = k * k;
                double inside_re = Ez_boundary_re + k_sq * int_G0_re;
                double inside_im = Ez_boundary_im + k_sq * int_G0_im;
                double Ez_const_re =  pref_z_im * inside_im;
                double Ez_const_im = -pref_z_im * inside_re;

                // ---- Sine source (Eqs 76, 77) ----------------------------
                double sin2 = sin_kH[n];
                double cos2 = cos_kH[n];
                double sin1 = -sin2;
                double cos1 =  cos2;

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
                double pref_rho_im = pref_rho_const_im * inv_rho_eval;
                double Erho_sin_re = -pref_rho_im * (bsin2_im - bsin1_im);
                double Erho_sin_im =  pref_rho_im * (bsin2_re - bsin1_re);

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
                double Ez_sin_re = -pref_z_im * (bszin2_im - bszin1_im);
                double Ez_sin_im =  pref_z_im * (bszin2_re - bszin1_re);

                // ---- Folded source (I = cos kξ − 1), #205 ----------------
                // Transcription of `SinusoidalSolver._folded_cos_fields`;
                // the derivation, and why the two closed forms may not
                // simply be subtracted, are in that method's docstring.
                double rho2 = rho_eval * rho_eval;
                double cm1 = cos_kH_m1[n];

                // X: sinh of the arcsinh difference ∫(1/r)dξ, rationalized
                // on whichever endpoint pair does not cancel.
                double X = (dz1 * dz2 >= 0.0)
                    ? 2.0 * H * (dz1 + dz2) / (dz1 * r0_2 + dz2 * r0_1)
                    : (dz1 * r0_2 - dz2 * r0_1) / rho2;
                double t_asx = std::asinh(X);
                double t_sing =
                    (std::fabs(X) < 1.0)
                        ? asinh_minus_arg_from_t(t_asx)
                              + H * rho2 * X * X / ((r0_1 + r0_2) * r0_1 * r0_2)
                        : t_asx - H * (inv_r0_1 + inv_r0_2);

                // g_e = (e^{−jkr_e} − 1)/r_e at the two endpoints, real part
                // from the half-angle (cos kr − 1 = −2sin²(kr/2)) so it is
                // relative rather than absolute.
                double g2_re = -2.0 * sh2 * sh2 * inv_r0_2;
                double g2_im = sph_2 * inv_r0_2;
                double g1_re = -2.0 * sh1 * sh1 * inv_r0_1;
                double g1_im = sph_1 * inv_r0_1;

                // Σ_q gw_q·g(ξ_q) − (g₁+g₂), each node against the endpoint
                // on its own side: g(r_ref+δ) − g(r_ref), both terms O(kδ).
                double m_reg_re = w_hi * g2_re + w_lo * g1_re;
                double m_reg_im = w_hi * g2_im + w_lo * g1_im;
                for (size_t q = 0; q < n_qp; q++) {
                    bool hi = gl_near2[q];
                    double e_ref_re = hi ? cph_2 : cph_1;
                    double e_ref_im = hi ? sph_2 : sph_1;
                    double gr_re = hi ? g2_re : g1_re;
                    double gr_im = hi ? g2_im : g1_im;
                    double sd = sphb[base + IX_DQ + q];
                    double cd = cphb[base + IX_DQ + q];
                    // e^{−jkδ} − 1 from the half-angle: −2s² + 2jsc.
                    double em1_re = -2.0 * sd * sd;
                    double em1_im = 2.0 * sd * cd;
                    double t_re, t_im;
                    cmul(e_ref_re, e_ref_im, em1_re, em1_im, t_re, t_im);
                    double dl = delta_a[n * n_qp + q];
                    double inv_r0_q = r0q_inv_a[n * n_qp + q];
                    double w = glw_v[q] * inv_r0_q;
                    m_reg_re += w * (t_re - gr_re * dl);
                    m_reg_im += w * (t_im - gr_im * dl);
                }

                double smarg = smarg_kH[n];
                double d_int_re = t_sing + H * m_reg_re
                                  - smarg / k * (G0_1_re + G0_2_re);
                double d_int_im = H * m_reg_im - smarg / k * (G0_1_im + G0_2_im);
                double inner_cos_re =
                    k_sq * d_int_re - cm1 * Ez_boundary_re;
                double inner_cos_im =
                    k_sq * d_int_im - cm1 * Ez_boundary_im;
                double Ez_cos_re = -pref_z_im * inner_cos_im;
                double Ez_cos_im =  pref_z_im * inner_cos_re;

                // E_ρ: both endpoint terms referred to the pair's mean
                // radius, which splits them into an even part that
                // telescopes onto sin_minus_arg and an odd part in c₂ − c₁.
                double cph_p = cphb[base + IX_PHI], sph_p = sphb[base + IX_PHI];
                double phi_ang = ph[base + IX_PHI];
                double kH = k * H;
                double A_ang = kH + phi_ang, B_ang = kH - phi_ang;
                // H(c₁+c₂) − (r₁−r₂), cancellation-free.
                double d_lin = -8.0 * H * H * H * z_a[n] * rho2
                               / ((rho2 + dz1 * dz2 + r0_1 * r0_2)
                                  * (r0_1 + r0_2) * r0_1 * r0_2);
                double w_even =
                    (A_ang * sin_minus_arg(B_ang) - B_ang * sin_minus_arg(A_ang))
                        / kH
                    + (d_lin / H) * sin2 * cph_p;
                // c₂ − c₁ = −ρ²X/(r₁r₂), from the same numerator as X.
                double w_odd = sin2 * (-(rho2 * X) / (r0_1 * r0_2)) * sph_p;
                // W = e^{−jk(r₁+r₂)/2}·(w_even + j·w_odd), the reference
                // phase built from e^{−jkr₂} and φ rather than a new exp.
                double ref_re = cph_2 * cph_p + sph_2 * sph_p;
                double ref_im = sph_2 * cph_p - cph_2 * sph_p;
                double W_re, W_im;
                cmul(ref_re, ref_im, w_even, w_odd, W_re, W_im);
                double b_rho_re = -k * W_re + rho2 * cm1
                                  * (term_const2_re - term_const1_re);
                double b_rho_im = -k * W_im + rho2 * cm1
                                  * (term_const2_im - term_const1_im);
                double Erho_cos_re = -pref_rho_im * b_rho_im;
                double Erho_cos_im =  pref_rho_im * b_rho_re;

                // ---- Extended kernel: the folded delta (momwire#246) ------
                // Transcription of `SinusoidalSolver._folded_ek_delta_fields`;
                // the derivation of the delta kernel W = a²g′ + a²ρ²g″, the
                // two field operators, and the floor underneath the whole
                // decomposition are in that method's docstring. Nothing above
                // this point moves: the reduced tables are #205's own, and the
                // delta is a separate sum at its own size added to them before
                // the projection — exactly where the numpy seam adds it.
                if (WITH_EK && g_obs_m >= 0 && ekb->g_src[n] == g_obs_m) {
                    // ζ = ρ·sinh t, R = ρ·cosh t. The variable is NOT ξ: in ξ
                    // the delta is a spike of width ρ inside a segment of half
                    // length H, so a fixed rule's accuracy is governed by ρ/H
                    // and a 16-node rule at the self pair is wrong by 1e6 at
                    // Δ/a = 122. In t the spike is O(1) wide however thin the
                    // wire, and the kernel's poles sit at t = ±jπ/2 whatever ρ
                    // is. `gx`/`gw` are the caller's composite rule over that
                    // mapped interval (`_ek_delta_rule`).
                    double t_lo = std::asinh((z_a[n] - H) / rho_eval);
                    double t_hi = std::asinh((z_a[n] + H) / rho_eval);
                    double t_mid = 0.5 * (t_hi + t_lo);
                    double t_half = 0.5 * (t_hi - t_lo);
                    double src_a = ekb->src_a[n];
                    double dz_c_re = 0.0, dz_c_im = 0.0;  // E_z,   const shape
                    double dr_c_re = 0.0, dr_c_im = 0.0;  // E_ρ,   const shape
                    double dz_s_re = 0.0, dz_s_im = 0.0;  // sin shape
                    double dr_s_re = 0.0, dr_s_im = 0.0;
                    double dz_o_re = 0.0, dz_o_im = 0.0;  // folded cos shape
                    double dr_o_re = 0.0, dr_o_im = 0.0;
                    for (size_t qd = 0; qd < ekb->n_gl; qd++) {
                        double t = t_mid + t_half * ekb->gx[qd];
                        double cosh_t = std::cosh(t);
                        // R = ρ·cosh t exactly, so the near-singular
                        // denominator never goes through a difference of
                        // squares; ξ = z − ζ recovers the source arc.
                        double R = rho_eval * cosh_t;
                        double zeta = rho_eval * std::sinh(t);
                        double xi = z_a[n] - zeta;
                        double wq = (t_half * ekb->gw[qd]) * (rho_eval * cosh_t);
                        double r2 = R * R;

                        // g′…g⁗ of e^{−jkR}/R in u = R², through the reverse
                        // Bessel polynomials Aₙ(x) at x = jkR. x is purely
                        // imaginary, so x², x³, x⁴ are exactly real or exactly
                        // imaginary and the four polynomials split by hand
                        // into the same values numpy's complex products give.
                        double kR = k * R;
                        double kr2 = kR * kR;
                        double kr3 = kr2 * kR;
                        double kr4 = kr2 * kr2;
                        double a1_re = 1.0,               a1_im = kR;
                        double a2_re = 3.0 - kr2,         a2_im = 3.0 * kR;
                        double a3_re = 15.0 - 6.0 * kr2,  a3_im = 15.0 * kR - kr3;
                        double a4_re = 105.0 - 45.0 * kr2 + kr4;
                        double a4_im = 105.0 * kR - 10.0 * kr3;

                        double inv2 = 1.0 / r2;
                        double inv4 = inv2 * inv2;
                        double inv6 = inv4 * inv2;
                        double inv8 = inv6 * inv2;
                        double base_re =  std::cos(kR) / R;
                        double base_im = -std::sin(kR) / R;
                        double gd1_re, gd1_im, gd2_re, gd2_im;
                        double gd3_re, gd3_im, gd4_re, gd4_im;
                        // Named steps in the numpy path's own order — the
                        // scale factor last — so the two backends' roundings
                        // stay one reassociation apart and no further.
                        cmul(base_re, base_im, a1_re, a1_im, gd1_re, gd1_im);
                        gd1_re *= inv2;   gd1_im *= inv2;
                        gd1_re *= -0.5;   gd1_im *= -0.5;
                        cmul(base_re, base_im, a2_re, a2_im, gd2_re, gd2_im);
                        gd2_re *= inv4;   gd2_im *= inv4;
                        gd2_re *= 0.25;   gd2_im *= 0.25;
                        cmul(base_re, base_im, a3_re, a3_im, gd3_re, gd3_im);
                        gd3_re *= inv6;   gd3_im *= inv6;
                        gd3_re *= -0.125; gd3_im *= -0.125;
                        cmul(base_re, base_im, a4_re, a4_im, gd4_re, gd4_im);
                        gd4_re *= inv8;   gd4_im *= inv8;
                        gd4_re *= 0.0625; gd4_im *= 0.0625;

                        // The two operators, with a² held back to the end:
                        //   (k² + ∂²_z)W = a²[k²(g′+ρ²g″) + 2(g″+ρ²g‴)
                        //                     + 4ζ²(g‴+ρ²g⁗)]
                        //   ∂²_{ρz} W    = a²·ρζ·[8g‴ + 4ρ²g⁗]
                        double z4 = 4.0 * zeta * zeta;
                        double lz_re = k_sq * (gd1_re + rho2 * gd2_re)
                                       + 2.0 * (gd2_re + rho2 * gd3_re)
                                       + z4 * (gd3_re + rho2 * gd4_re);
                        double lz_im = k_sq * (gd1_im + rho2 * gd2_im)
                                       + 2.0 * (gd2_im + rho2 * gd3_im)
                                       + z4 * (gd3_im + rho2 * gd4_im);
                        double lr_re = ((8.0 * gd3_re + 4.0 * rho2 * gd4_re)
                                        * rho_eval) * zeta;
                        double lr_im = ((8.0 * gd3_im + 4.0 * rho2 * gd4_im)
                                        * rho_eval) * zeta;

                        // The folded shape POINTWISE as −2sin²(kξ/2) — never
                        // cos − 1 by subtraction. There is no cancellation
                        // discipline to keep here because there is no
                        // difference of closed forms anywhere in the delta.
                        // The sin shape is evaluated at the full angle rather
                        // than reconstructed from the half one: measured, the
                        // reconstruction saves nothing here (the loop is
                        // bound by cosh/sinh, not by sincos) and costs a ulp.
                        double kxi = k * xi;
                        double hsx = std::sin(0.5 * kxi);
                        double sw_s = std::sin(kxi) * wq;
                        double sw_o = (-2.0 * hsx * hsx) * wq;
                        dz_c_re += wq * lz_re;    dz_c_im += wq * lz_im;
                        dr_c_re += wq * lr_re;    dr_c_im += wq * lr_im;
                        dz_s_re += sw_s * lz_re;  dz_s_im += sw_s * lz_im;
                        dr_s_re += sw_s * lr_re;  dr_s_im += sw_s * lr_im;
                        dz_o_re += sw_o * lz_re;  dz_o_im += sw_o * lz_im;
                        dr_o_re += sw_o * lr_re;  dr_o_im += sw_o * lr_im;
                    }
                    // −pref_z·a²: one multiplication at the end, which is what
                    // makes the whole correction exactly 0.0 at a = 0 in IEEE
                    // and not merely in the limit. `src_a` is the SOURCE
                    // segment's radius, the a of NEC Eq 89.
                    double gain = pref_z_im * (src_a * src_a);
                    Ez_const_re  += gain * dz_c_im;
                    Ez_const_im  -= gain * dz_c_re;
                    Erho_const_re += gain * dr_c_im;
                    Erho_const_im -= gain * dr_c_re;
                    Ez_sin_re  += gain * dz_s_im;
                    Ez_sin_im  -= gain * dz_s_re;
                    Erho_sin_re += gain * dr_s_im;
                    Erho_sin_im -= gain * dr_s_re;
                    Ez_cos_re  += gain * dz_o_im;
                    Ez_cos_im  -= gain * dz_o_re;
                    Erho_cos_re += gain * dr_o_im;
                    Erho_cos_im -= gain * dr_o_re;
                }

                phi_c_re[n]  = td * Ez_const_re + rho_proj_factor * Erho_const_re;
                phi_c_im[n]  = td * Ez_const_im + rho_proj_factor * Erho_const_im;
                phi_s_re[n]  = td * Ez_sin_re   + rho_proj_factor * Erho_sin_re;
                phi_s_im[n]  = td * Ez_sin_im   + rho_proj_factor * Erho_sin_im;
                phi_co_re[n] = td * Ez_cos_re   + rho_proj_factor * Erho_cos_re;
                phi_co_im[n] = td * Ez_cos_im   + rho_proj_factor * Erho_cos_im;
            }

            // ---- Test reduction: contrib[e, :] += w_entry[e, qt] * Phi ----
            // One axpy per support entry of this test segment, over the whole
            // source row. Accumulating quadrature node by quadrature node
            // matches `_tested_contrib_rows`' summation order.
            for (size_t e = e0; e < e1; e++) {
                double wr = w_p[e * nq + qt].real();
                double wi = w_p[e * nq + qt].imag();
                double *rc  = reinterpret_cast<double *>(bc + (e - e0) * N);
                double *rs  = reinterpret_cast<double *>(bs + (e - e0) * N);
                double *rco = reinterpret_cast<double *>(bco + (e - e0) * N);
                PYSIM_OMP_SIMD()
                for (size_t n = 0; n < N; n++) {
                    rc[2*n]     += wr * phi_c_re[n]  - wi * phi_c_im[n];
                    rc[2*n + 1] += wr * phi_c_im[n]  + wi * phi_c_re[n];
                    rs[2*n]     += wr * phi_s_re[n]  - wi * phi_s_im[n];
                    rs[2*n + 1] += wr * phi_s_im[n]  + wi * phi_s_re[n];
                    rco[2*n]    += wr * phi_co_re[n] - wi * phi_co_im[n];
                    rco[2*n + 1]+= wr * phi_co_im[n] + wi * phi_co_re[n];
                }
            }
        }

        // ---- The fold: dst[e0:e1] += scale * band, scale on the LEFT -----
        // Once per test segment, off the finished sums (momwire#356). Rows
        // e0:e1 belong to this m alone, so this writes where no other thread
        // reads and needs no more synchronisation than the accumulation did.
        if (folding) {
            const std::complex<double> *src[3] = {bc, bs, bco};
            std::complex<double> *dst[3] = {oc_p + e0 * N, os_p + e0 * N,
                                            oco_p + e0 * N};
            for (int t = 0; t < 3; t++) {
                const double *b = reinterpret_cast<const double *>(src[t]);
                double *d = reinterpret_cast<double *>(dst[t]);
                if (fold_real) {
                    // Re and im are the same scaling, so the interleaved
                    // buffer is one flat axpy. At scale −1 this is exactly
                    // `np.subtract(dst, value, out=dst)`.
                    PYSIM_OMP_SIMD()
                    for (size_t i = 0; i < 2 * nrows * N; i++) {
                        d[i] += fold_re * b[i];
                    }
                } else {
                    PYSIM_OMP_SIMD()
                    for (size_t i = 0; i < nrows * N; i++) {
                        double vr = b[2*i], vi = b[2*i + 1];
                        d[2*i]     += fold_re * vr - fold_im * vi;
                        d[2*i + 1] += fold_re * vi + fold_im * vr;
                    }
                }
            }
        }
    }

    PYSIM_THROW_IF_ABORTED();
    return std::make_tuple(out_const, out_sin, out_cos);
}

// One of the three fold destinations, checked (momwire#356). It has to be
// exactly the array the allocating path would have returned — (nnz, N)
// complex128, C-contiguous and writeable — because the kernel folds into its
// buffer in place. A forcecast `array_t` would silently accept a copy here
// and the fold would land in the copy, so the cast below is the strict one
// and the shape/layout checks are explicit.
static py::array_t<std::complex<double>> galerkin_fold_dest(
    py::handle h, py::ssize_t nnz, py::ssize_t N, const char *which
) {
    if (!py::isinstance<py::array_t<std::complex<double>>>(h)) {
        throw std::runtime_error(
            std::string("out[") + which + "] must be a complex128 array");
    }
    auto a = py::reinterpret_borrow<py::array_t<std::complex<double>>>(h);
    if (a.ndim() != 2 || a.shape(0) != nnz || a.shape(1) != N) {
        throw std::runtime_error(
            std::string("out[") + which + "] must have shape (nnz, N)");
    }
    if ((a.flags() & py::array::c_style) == 0 || !a.writeable()) {
        throw std::runtime_error(
            std::string("out[") + which +
            " must be C-contiguous and writeable");
    }
    return a;
}

// `out` (a 3-sequence of destinations, or None) and `scale` into a fold
// block. Returns false for `out=None`, which is the pre-#356 behaviour: the
// fill allocates and returns its own triple and never touches a caller array.
static bool galerkin_fold_block(
    const py::object &out, std::complex<double> scale,
    py::ssize_t nnz, py::ssize_t N, GalerkinFoldBlock &blk
) {
    if (out.is_none()) return false;
    auto seq = py::cast<py::sequence>(out);
    if (py::len(seq) != 3) {
        throw std::runtime_error(
            "out must be (const, sin, cos) — three (nnz, N) complex arrays");
    }
    blk.dst_const = galerkin_fold_dest(seq[0], nnz, N, "const");
    blk.dst_sin = galerkin_fold_dest(seq[1], nnz, N, "sin");
    blk.dst_cos = galerkin_fold_dest(seq[2], nnz, N, "cos");
    blk.s_re = scale.real();
    blk.s_im = scale.imag();
    return true;
}

// The reduced entry point. Byte-frozen against its pre-#246 build: the
// instantiation below has WITH_EK false, so not one line of the delta is
// compiled into it (gate G-C2). momwire#356's `out`/`scale` are additive and
// default to `None`/1, which reaches `galerkin_far_fill_impl` with a null
// fold block — the allocating path, unchanged.
static std::tuple<py::array_t<std::complex<double>>,
                  py::array_t<std::complex<double>>,
                  py::array_t<std::complex<double>>>
sinusoidal_galerkin_far_fill(
    py::array_t<double, py::array::c_style | py::array::forcecast> obs_centers,
    py::array_t<double, py::array::c_style | py::array::forcecast> obs_tangents,
    py::array_t<double, py::array::c_style | py::array::forcecast> obs_radius,
    py::array_t<double, py::array::c_style | py::array::forcecast> src_centers,
    py::array_t<double, py::array::c_style | py::array::forcecast> src_tangents,
    py::array_t<double, py::array::c_style | py::array::forcecast> src_hh,
    double k, double eta,
    py::array_t<double, py::array::c_style | py::array::forcecast> gl_t,
    py::array_t<double, py::array::c_style | py::array::forcecast> gl_w,
    py::array_t<std::complex<double>,
                py::array::c_style | py::array::forcecast> w_entry,
    py::array_t<int64_t, py::array::c_style | py::array::forcecast> starts,
    uintptr_t cancel_flag = 0,
    py::object out = py::none(),
    std::complex<double> scale = std::complex<double>(1.0, 0.0)
) {
    GalerkinFoldBlock fold;
    bool folding = galerkin_fold_block(
        out, scale, w_entry.shape(0), src_centers.shape(0), fold);
    return galerkin_far_fill_impl<false>(
        obs_centers, obs_tangents, obs_radius, src_centers, src_tangents,
        src_hh, k, eta, gl_t, gl_w, w_entry, starts, cancel_flag, nullptr,
        folding ? &fold : nullptr);
}

// The extended-kernel twin (momwire#246 unit C). Same arguments plus the EK
// payload — one source radius per segment, the pair rule's GROUP LABELS
// (momwire#358: per test segment and per source segment, the mask derived in
// the sweep), and the delta quadrature's composite rule — and the same three
// arrays out.
static std::tuple<py::array_t<std::complex<double>>,
                  py::array_t<std::complex<double>>,
                  py::array_t<std::complex<double>>>
sinusoidal_galerkin_far_fill_ek(
    py::array_t<double, py::array::c_style | py::array::forcecast> obs_centers,
    py::array_t<double, py::array::c_style | py::array::forcecast> obs_tangents,
    py::array_t<double, py::array::c_style | py::array::forcecast> obs_radius,
    py::array_t<double, py::array::c_style | py::array::forcecast> src_centers,
    py::array_t<double, py::array::c_style | py::array::forcecast> src_tangents,
    py::array_t<double, py::array::c_style | py::array::forcecast> src_hh,
    double k, double eta,
    py::array_t<double, py::array::c_style | py::array::forcecast> gl_t,
    py::array_t<double, py::array::c_style | py::array::forcecast> gl_w,
    py::array_t<std::complex<double>,
                py::array::c_style | py::array::forcecast> w_entry,
    py::array_t<int64_t, py::array::c_style | py::array::forcecast> starts,
    py::array_t<double, py::array::c_style | py::array::forcecast> src_a,
    py::array_t<int64_t, py::array::c_style | py::array::forcecast> group_obs,
    py::array_t<int64_t, py::array::c_style | py::array::forcecast> group_src,
    py::array_t<double, py::array::c_style | py::array::forcecast> ek_gx,
    py::array_t<double, py::array::c_style | py::array::forcecast> ek_gw,
    uintptr_t cancel_flag = 0,
    py::object out = py::none(),
    std::complex<double> scale = std::complex<double>(1.0, 0.0)
) {
    auto sa = src_a.unchecked<1>();
    auto go = group_obs.unchecked<1>();
    auto gs = group_src.unchecked<1>();
    auto ex = ek_gx.unchecked<1>();
    auto ew = ek_gw.unchecked<1>();
    if (sa.shape(0) != src_centers.shape(0)) {
        throw std::runtime_error("src_a must have one radius per source segment");
    }
    // One observer label per TEST segment, not per observer row: the sweep
    // reads it once per segment and the rows of one segment shared a mask row
    // anyway (momwire#358). `starts` is the CSR's per-test-segment row index,
    // so it has M + 1 entries.
    if (go.shape(0) != starts.shape(0) - 1) {
        throw std::runtime_error(
            "group_obs must have one label per test segment (len(starts) - 1)");
    }
    if (gs.shape(0) != src_centers.shape(0)) {
        throw std::runtime_error(
            "group_src must have one label per source segment");
    }
    if (ex.shape(0) != ew.shape(0) || ex.shape(0) < 1) {
        throw std::runtime_error("ek_gx and ek_gw must have matching length");
    }
    GalerkinEkBlock ekb;
    ekb.src_a = src_a.data();
    ekb.g_obs = group_obs.data();
    ekb.g_src = group_src.data();
    ekb.gx = ek_gx.data();
    ekb.gw = ek_gw.data();
    ekb.n_gl = (size_t)ex.shape(0);
    GalerkinFoldBlock fold;
    bool folding = galerkin_fold_block(
        out, scale, w_entry.shape(0), src_centers.shape(0), fold);
    return galerkin_far_fill_impl<true>(
        obs_centers, obs_tangents, obs_radius, src_centers, src_tangents,
        src_hh, k, eta, gl_t, gl_w, w_entry, starts, cancel_flag, &ekb,
        folding ? &fold : nullptr);
}


// ==========================================================================
// Sommerfeld ground: batched six-integral evaluation (sommerfeld-perf-plan
// Phase 3). A faithful port of _sommerfeld.py's `_six_integrals` machinery
// (same 24-point Gauss rule, same adaptive bisection, same fig 13/14
// contours and tail handling), evaluated per (rho, h) node with OpenMP
// across nodes — the nodes are independent, and the pure-Python fill spends
// ~75% of its time on interpreter/ufunc dispatch this port removes.
//
// Clean-room note: ported from momwire's own _sommerfeld.py (itself
// implemented from the public-domain NEC-2 theory manual equations); the
// complex-argument Bessel/Hankel functions below are implemented from the
// Abramowitz & Stegun 9.1/9.2 ascending series and asymptotic expansions.
// No GPL Sommerfeld source (nec2c, nec2++/PyNEC, somnec) was consulted.
// ==========================================================================


void register_sinusoidal(py::module_ &m) {

    m.def("sinusoidal_field_tensor", &sinusoidal_field_tensor,
          "Tangential field tensor for the NEC2 three-term basis. Returns "
          "(Phi_const, Phi_sin, Phi_cos), each (M, N) complex. obs_*/src_* "
          "can be the same arrays (free-space build) or src_* mirrored "
          "(PEC image build).",
          py::arg("obs_centers"), py::arg("obs_tangents"),
          py::arg("src_centers"), py::arg("src_tangents"),
          py::arg("seg_h"),
          py::arg("a"), py::arg("k"), py::arg("eta"),
          py::arg("gl_t"), py::arg("gl_w"),
          py::arg("cancel_flag") = 0);
    m.def("sinusoidal_field_tensor_refl", &sinusoidal_field_tensor_refl,
          "Reflection-coefficient finite-ground variant of "
          "sinusoidal_field_tensor: src_* are the MIRRORED image sources, "
          "and the tangential projection applies NEC's Fresnel field dyad "
          "t_m·D·E = rho_v·(t_m·E) − (rho_v+rho_h)·(t_m·p̂)(E·p̂) with "
          "rho_v/rho_h computed in-kernel at each pair's specular angle "
          "from eps_t and the k-independent tables (cos_th, px, py, tm_p, "
          "tn_p — see SinusoidalSolver._image_refl_band). Returns "
          "(Phi_const, Phi_sin, Phi_cos), each (M, N) complex.",
          py::arg("obs_centers"), py::arg("obs_tangents"),
          py::arg("src_centers"), py::arg("src_tangents"),
          py::arg("seg_h"),
          py::arg("a"), py::arg("k"), py::arg("eta"),
          py::arg("gl_t"), py::arg("gl_w"),
          py::arg("cos_th"), py::arg("px"), py::arg("py"),
          py::arg("tm_p"), py::arg("tn_p"),
          py::arg("eps_t"),
          py::arg("cancel_flag") = 0);
    m.def("sinusoidal_field_tensor_ek", &sinusoidal_field_tensor_ek,
          "NEC extended-thin-wire-kernel variant of sinusoidal_field_tensor "
          "(EKSCX, nec2-1.2.1.2.f:3170-3234). Same (Phi_const, Phi_sin, "
          "Phi_cos) contract and the same observer-side scalar radius `a`, "
          "plus three per-SOURCE tables: `src_a` (NEC's BX, the source "
          "conductor radius the O(a²) expansion is about) and the int8 "
          "per-end gating codes `ind1`/`ind2` from "
          "SinusoidalSolver._ek_gating (0/1 -> GXX, 2 -> GX). "
          "EKSCX's IRA is NOT an argument: it is resolved per pair inside "
          "the kernel from that pair's own `rhx < src_a`, the same "
          "comparison that orders (rh, b) — see momwire#258, which removed "
          "the build-wide `want_swapped` scalar #245 took. "
          "src_* may be the MIRRORED image sources (the gating tables are "
          "unchanged there, as in NEC's KSYMP loop).",
          py::arg("obs_centers"), py::arg("obs_tangents"),
          py::arg("src_centers"), py::arg("src_tangents"),
          py::arg("seg_h"),
          py::arg("a"), py::arg("k"), py::arg("eta"),
          py::arg("gl_t"), py::arg("gl_w"),
          py::arg("src_a"), py::arg("ind1"), py::arg("ind2"),
          py::arg("cancel_flag") = 0);
    m.def("sinusoidal_field_tensor_ek_refl", &sinusoidal_field_tensor_ek_refl,
          "Extended-thin-wire-kernel (EKSCX) variant of "
          "sinusoidal_field_tensor_refl: the EK tables of "
          "sinusoidal_field_tensor_ek (src_a / ind1 / ind2, with EKSCX's IRA "
          "resolved per pair in-kernel) "
          "with the Fresnel field dyad projection tail of "
          "sinusoidal_field_tensor_refl (cos_th, px, py, tm_p, tn_p, eps_t) "
          "in place of the plain tangential one. src_* are the MIRRORED "
          "image sources. Returns (Phi_const, Phi_sin, Phi_cos), each "
          "(M, N) complex.",
          py::arg("obs_centers"), py::arg("obs_tangents"),
          py::arg("src_centers"), py::arg("src_tangents"),
          py::arg("seg_h"),
          py::arg("a"), py::arg("k"), py::arg("eta"),
          py::arg("gl_t"), py::arg("gl_w"),
          py::arg("src_a"), py::arg("ind1"), py::arg("ind2"),
          py::arg("cos_th"), py::arg("px"), py::arg("py"),
          py::arg("tm_p"), py::arg("tn_p"),
          py::arg("eps_t"),
          py::arg("cancel_flag") = 0);
    m.def("sinusoidal_galerkin_far_fill", &sinusoidal_galerkin_far_fill,
          "Fused far fill for SinusoidalGalerkinSolver: the plainly-projected "
          "Eqs 76-79 tensor at each test segment's nq Gauss points, reduced "
          "on the fly against the test weights. Returns (contrib_const, "
          "contrib_sin, contrib_cos), each (nnz, N) complex, where nnz is the "
          "CSR support-entry count (`starts` is that CSR's per-test-segment "
          "row index). src_* are the geometry's segments (free-space block) "
          "or the MIRRORED ones (PEC image block); the reflection-coefficient "
          "and Sommerfeld blocks stay on the numpy path. With `out` — three "
          "(nnz, N) complex128 C-contiguous arrays — nothing is allocated "
          "and each entry is folded on as `out += scale * value`, scale on "
          "the LEFT of the complex product; the same three arrays come back. "
          "`out=None` with `scale=1` is the pre-momwire#356 behaviour to the "
          "bit.",
          py::arg("obs_centers"), py::arg("obs_tangents"),
          py::arg("obs_radius"),
          py::arg("src_centers"), py::arg("src_tangents"), py::arg("src_hh"),
          py::arg("k"), py::arg("eta"),
          py::arg("gl_t"), py::arg("gl_w"),
          py::arg("w_entry"), py::arg("starts"),
          py::arg("cancel_flag") = 0,
          py::arg("out") = py::none(),
          py::arg("scale") = std::complex<double>(1.0, 0.0));
    m.def("sinusoidal_galerkin_far_fill_ek", &sinusoidal_galerkin_far_fill_ek,
          "Extended-kernel twin of sinusoidal_galerkin_far_fill "
          "(momwire#246): the same fused far fill, plus the folded EK delta "
          "on the pairs the group labels select. `src_a` is one radius per "
          "SOURCE segment; `group_obs` (one label per TEST segment, i.e. "
          "len(starts) - 1) and `group_src` (one per source segment) are the "
          "pair rule's coaxial-and-equal-radius labels, from which the sweep "
          "derives eligibility as `g_obs[m] >= 0 and g_obs[m] == g_src[n]` "
          "(momwire#358 — the (obs rows, N) mask this replaced was constant "
          "over each test segment's observer rows). `ek_gx`/`ek_gw` are the "
          "composite sinh-mapped rule "
          "`SinusoidalSolver._ek_delta_rule` built — passed in, not rebuilt "
          "here, so both backends integrate the delta on the same nodes. "
          "Returns the same three (nnz, N) arrays, and takes the same "
          "momwire#356 `out`/`scale` fold destination.",
          py::arg("obs_centers"), py::arg("obs_tangents"),
          py::arg("obs_radius"),
          py::arg("src_centers"), py::arg("src_tangents"), py::arg("src_hh"),
          py::arg("k"), py::arg("eta"),
          py::arg("gl_t"), py::arg("gl_w"),
          py::arg("w_entry"), py::arg("starts"),
          py::arg("src_a"), py::arg("group_obs"), py::arg("group_src"),
          py::arg("ek_gx"), py::arg("ek_gw"),
          py::arg("cancel_flag") = 0,
          py::arg("out") = py::none(),
          py::arg("scale") = std::complex<double>(1.0, 0.0));
}

