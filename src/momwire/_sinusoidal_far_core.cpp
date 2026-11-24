// NEC-2's interaction approximation on the point-matched sinusoidal fill
// (the opt-in `nec2_far_approx`; `_sinusoidal_far_core.h`).
//
// THE FORM. NEC-2 fills a pair whose segment centres are more than RKH
// wavelengths apart (the KH card; 1 wavelength when a deck sends none, measured
// on nec2c 1.3.1, `momwire._nec2_far_approx`) from an approximate integration
// instead of the full one. Here that is the source segment's current lumped
// into an infinitesimal dipole at its centre, along its tangent, with the
// moment of each basis shape over [-H, H]:
//
//     const:  2H            sin:  0            cos:  2 sin(kH) / k
//
// and the dipole's whole field (all three of its 1/R, 1/R^2, 1/R^3 terms),
//
//     E = -j k eta G [A t + B (t . r) r],   G = exp(-jkR) / (4 pi R),
//     A = 1 - j/(kR) - 1/(kR)^2,   B = -(1 - 3j/(kR) - 3/(kR)^2),
//
// at the observer point the exact kernel uses (the segment-centre offset by
// the observer radius, rho_eval = sqrt(rho^2 + a^2)), projected onto the
// observer tangent exactly as the exact kernel projects its E_z / E_rho, or
// through the same Fresnel dyad on the reflection-coefficient image. It is
// the exact kernel's large-R limit: the two differ by O((kH)^2) and
// O((H/R)^2) relative. The black-box agreement with nec2c that fixed this
// choice (and ruled out a lumped constant term alone) is in
// `momwire._nec2_far_approx`.
//
// The decision is the CENTRE distance, |c_m - c_n| > far_r, without the
// observer radius: NEC-2's printed rule ("segments that are more than RKH
// wavelengths apart").
//
// THE NEAR PAIRS are `_sinusoidal_core.cpp`'s reduced kernel text (stage A, B,
// C), compacted per observer row onto the near sources so the far ones cost
// one sincos and a few multiplies instead of n_qp + 2 sincos and the closed
// forms. Their arithmetic is the shipped kernel's, term for term; the only
// difference is which lane of the vector sincos a phase lands in, which can
// move a value by an ulp.

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

#include "_fma_inline.h"

#include "_mw_core.h"
#include "_mw_nd.h"
#include "_sinusoidal_core.h"
#include "_sinusoidal_far_core.h"

namespace py = mw::nd;

#include "_stable_inline.h"

namespace {

typedef py::array_t<double, py::array::c_style | py::array::forcecast> F64;
typedef py::array_t<std::complex<double>, py::array::c_style | py::array::forcecast> C128;

// A far pair's geometry, in the source's frame: `z` along the source tangent,
// R the distance from its centre to the observer point (the observer centre
// offset by the observer radius, as the exact kernel evaluates), and the
// projections the tail needs. The radial distance itself never appears: the
// E_rho terms reach the result only through (rho_vec . t_m) / rho_eval times
// rho_eval / R, and (rho_vec . p) / rho_eval times the same, so the tail
// carries rdt = rho_vec . t_m and rvp = rho_vec . p over R instead (one sqrt
// and one division per pair).
struct FarPair {
    double z, R, td, rdt;
    double rvp;  // REFL only
    size_t n;
};

// The far pair's three tensors: the lumped dipole's field (unit moment)
//     E = -j k eta G [A t + B (t . r) r],   G = exp(-jkR) / (4 pi R),
//     A = 1 - j/(kR) - 1/(kR)^2,   B = -(1 - 3j/(kR) - 3/(kR)^2),
// as E_z = c (A + B u^2) and E_rho = c B u v (u = z/R, v = rho/R, c = -j k
// eta G), projected as the exact kernel projects (plain, or the Fresnel
// dyad's), times each shape's moment.
template <bool REFL>
static inline void far_tail(const FarPair &f, double cph, double sph, double k,
                            double eta, double H, double sin_kH,
                            std::complex<double> eps_t, double ct, double tmp,
                            double tnp, std::complex<double> &out_c,
                            std::complex<double> &out_s,
                            std::complex<double> &out_co,
                            bool cos_minus_one = false) {
    const double inv_R = 1.0 / f.R;
    const double g = inv_R * (1.0 / (4.0 * M_PI));
    const double g_re = cph * g, g_im = sph * g;
    const double c_re = k * eta * g_im, c_im = -k * eta * g_re;
    const double inv_kr = inv_R / k;
    const double inv_kr2 = inv_kr * inv_kr;
    const double A_re = 1.0 - inv_kr2, A_im = -inv_kr;
    const double B_re = -(1.0 - 3.0 * inv_kr2), B_im = 3.0 * inv_kr;
    const double u = f.z * inv_R;
    const double uu = u * u;
    // E_z = c (A + B u^2); W = c B u / R, so that rpf * E_rho = rdt * W.
    const double pz_re = A_re + B_re * uu, pz_im = A_im + B_im * uu;
    const double ez_re = c_re * pz_re - c_im * pz_im;
    const double ez_im = c_re * pz_im + c_im * pz_re;
    const double ui = u * inv_R;
    const double cb_re = c_re * B_re - c_im * B_im;
    const double cb_im = c_re * B_im + c_im * B_re;
    const double w_re = cb_re * ui, w_im = cb_im * ui;
    std::complex<double> phi;
    if (REFL) {
        std::complex<double> root = std::sqrt(eps_t - (1.0 - ct * ct));
        std::complex<double> rho_v = (eps_t * ct - root) / (eps_t * ct + root);
        std::complex<double> rho_h = (ct - root) / (ct + root);
        std::complex<double> rvh = rho_v + rho_h;
        std::complex<double> Ez(ez_re, ez_im), W(w_re, w_im);
        std::complex<double> tm_E = f.td * Ez + f.rdt * W;
        std::complex<double> E_p = tnp * Ez + f.rvp * W;
        phi = rho_v * tm_E - rvh * (tmp * E_p);
    } else {
        phi = std::complex<double>(f.td * ez_re + f.rdt * w_re,
                                   f.td * ez_im + f.rdt * w_im);
    }
    const double m_const = 2.0 * H;
    // The folded third shape, cos(k xi) - 1 (`cos_shape="cos-1"`), lumps to
    // the difference of the two moments.
    const double m_cos = cos_minus_one ? 2.0 * sin_kH / k - m_const
                                       : 2.0 * sin_kH / k;
    out_c = phi * m_const;
    out_s = std::complex<double>(0.0, 0.0);
    out_co = phi * m_cos;
}

// A far pair from its centre-to-centre vector and the two tangents.
template <bool REFL>
static inline FarPair far_pair(double rvx, double rvy, double rvz, double d2,
                               double tnx, double tny, double tnz, double tmx,
                               double tmy, double tmz, double a_sq, double px,
                               double py, size_t n) {
    FarPair f;
    f.z = rvx * tnx + rvy * tny + rvz * tnz;
    f.td = tmx * tnx + tmy * tny + tmz * tnz;
    f.R = std::sqrt(d2 + a_sq);
    // rho_vec = rv - z t_n.
    f.rdt = (rvx * tmx + rvy * tmy + rvz * tmz) - f.z * f.td;
    f.rvp = REFL ? (rvx * px + rvy * py) - f.z * (tnx * px + tny * py) : 0.0;
    f.n = n;
    return f;
}

static void check_pair_arrays(const F64 &obs_centers, const F64 &obs_tangents,
                              const F64 &src_centers, const F64 &src_tangents,
                              const F64 &seg_h) {
    if (obs_centers.ndim() != 2 || obs_tangents.ndim() != 2 ||
        src_centers.ndim() != 2 || src_tangents.ndim() != 2 ||
        obs_centers.shape(1) != 3 || obs_tangents.shape(1) != 3 ||
        src_centers.shape(1) != 3 || src_tangents.shape(1) != 3) {
        throw std::runtime_error("center/tangent arrays must have shape (N, 3)");
    }
    if (obs_centers.shape(0) != obs_tangents.shape(0)) {
        throw std::runtime_error("obs_centers and obs_tangents must have matching N");
    }
    if (src_centers.shape(0) != src_tangents.shape(0) ||
        src_centers.shape(0) != seg_h.shape(0)) {
        throw std::runtime_error("src arrays must all have matching N");
    }
}

static void check_far_r(double far_r) {
    if (!(far_r >= 0.0)) {
        throw std::runtime_error("far_r must be a non-negative distance");
    }
}

// The sources' per-segment precompute, H = seg_h / 2 and sin(kH), cos(kH):
// ONE out-of-line function for both routes. Its loop is a candidate for the
// compiler's vectorizer, whose libmvec sin/cos can differ from libm's in the
// last bit; inlined into two callers it was compiled two ways (measured: the
// fused route's cos(kH) moved one ulp on a stepped-length deck, its Phi_cos by
// 2e-16), so the far route and the fused route take it from here.
__attribute__((noinline)) static void far_source_tables(size_t N, const double *seg_h,
                                                        double k, double *H_n,
                                                        double *sin_kH, double *cos_kH) {
    for (size_t n = 0; n < N; n++) {
        H_n[n] = 0.5 * seg_h[n];
        sin_kH[n] = std::sin(k * H_n[n]);
        cos_kH[n] = std::cos(k * H_n[n]);
    }
}

// A far-aware row's per-call constants, read-only and shared by the threads:
// the sources' per-segment precompute, the rule and the prefactors, each as
// `far_impl` forms it.
struct FarRowConst {
    size_t n_qp, S;
    const double *H_n, *sin_kH, *cos_kH;
    const double *glt, *glw;
    double k, eta, pref_z_im, pref_rho_const_im;
};

// A far-aware row's per-thread scratch: `near_cap` near pairs' stage-A
// slots and `far_cap` far pairs'. `far_impl` sizes it for a whole row (N and
// N); the fused fill (`sinusoidal_fill_rows_far`) for one flush of each.
struct FarRowScratch {
    std::vector<double> ph, cphb, sphb;
    std::vector<double> rho_eval_a, dz1_a, dz2_a, r0_1_a, r0_2_a, td_a, rpf_a;
    std::vector<double> r0q_inv_a;
    std::vector<double> rho_p_a;
    std::vector<size_t> near_n;
    std::vector<FarPair> far_a;
    std::vector<double> fph, fcph, fsph;
    FarRowScratch(size_t near_cap, size_t far_cap, size_t n_qp, bool refl)
        : ph(near_cap * (n_qp + 2)), cphb(near_cap * (n_qp + 2)),
          sphb(near_cap * (n_qp + 2)), rho_eval_a(near_cap), dz1_a(near_cap),
          dz2_a(near_cap), r0_1_a(near_cap), r0_2_a(near_cap), td_a(near_cap),
          rpf_a(near_cap), r0q_inv_a(near_cap * n_qp), rho_p_a(refl ? near_cap : 0),
          near_n(near_cap), far_a(far_cap), fph(far_cap), fcph(far_cap), fsph(far_cap) {}
};

// Stage A of one NEAR pair into slot `j`: its geometry and its n_qp + 2
// phases. `rv` is the observer centre less the source centre, `tn` / `tm` the
// source and observer tangents, `a_sq` the observer radius squared, and
// `px` / `py` the pair's reflection-table entries (REFL only).
template <bool REFL>
static inline void near_stage_a(const FarRowConst &K, FarRowScratch &s, size_t j,
                                size_t n, double rvx, double rvy, double rvz,
                                double tnx, double tny, double tnz, double tmx,
                                double tmy, double tmz, double a_sq, double px,
                                double py) {
    const double k = K.k;
    const size_t n_qp = K.n_qp;
    const size_t S = K.S;
    const double *H_n = K.H_n, *glt_v = K.glt;
    double *ph = s.ph.data();
    double *rho_eval_a = s.rho_eval_a.data(), *dz1_a = s.dz1_a.data(),
           *dz2_a = s.dz2_a.data(), *r0_1_a = s.r0_1_a.data(),
           *r0_2_a = s.r0_2_a.data(), *td_a = s.td_a.data(), *rpf_a = s.rpf_a.data();
    double *r0q_inv_a = s.r0q_inv_a.data();
    double *rho_p_a = s.rho_p_a.data();
    size_t *near_n = s.near_n.data();
    double z_eval = rvx * tnx + rvy * tny + rvz * tnz;
    double rho_vx = rvx - z_eval * tnx;
    double rho_vy = rvy - z_eval * tny;
    double rho_vz = rvz - z_eval * tnz;
    double rho_axis = std::sqrt(rho_vx*rho_vx + rho_vy*rho_vy + rho_vz*rho_vz);
    double rho_eval = std::sqrt(rho_axis*rho_axis + a_sq);
    double td = tmx*tnx + tmy*tny + tmz*tnz;
    double rho_dot_tobs = rho_vx*tmx + rho_vy*tmy + rho_vz*tmz;
    double rho_p = 0.0;
    if (REFL) {
        rho_p = (rho_vx * px + rho_vy * py) / rho_eval;
    }

    near_n[j] = n;
    double H = H_n[n];
    double dz2 = z_eval - H;
    double dz1 = z_eval + H;
    double r0_2 = std::sqrt(rho_eval*rho_eval + dz2*dz2);
    double r0_1 = std::sqrt(rho_eval*rho_eval + dz1*dz1);

    rho_eval_a[j] = rho_eval;
    dz1_a[j] = dz1; dz2_a[j] = dz2;
    r0_1_a[j] = r0_1; r0_2_a[j] = r0_2;
    td_a[j] = td; rpf_a[j] = rho_dot_tobs / rho_eval;
    if (REFL) rho_p_a[j] = rho_p;

    size_t base = j * S;
    ph[base + 0] = -k * r0_2;
    ph[base + 1] = -k * r0_1;
    for (size_t q = 0; q < n_qp; q++) {
        double z_q = H * glt_v[q];
        double dz_q = z_eval - z_q;
        double r0_q = std::sqrt(rho_eval*rho_eval + dz_q*dz_q);
        ph[base + 2 + q] = -0.5 * k * r0_q;
        r0q_inv_a[j * n_qp + q] = 1.0 / r0_q;
    }
}

// Stage B: the vectorized sincos over the first `P` near phases and the
// first `n_far` far ones. Each sweep starts at its buffer's start, as the
// whole-row sweep did, so a phase lands in the vector or the scalar part of
// the libmvec split by its position alone (see `kNearFlush`).
static inline void far_row_sincos(FarRowScratch &s, size_t P, size_t n_far) {
    const double *ph = s.ph.data(), *fph = s.fph.data();
    double *cphb = s.cphb.data(), *sphb = s.sphb.data();
    double *fcph = s.fcph.data(), *fsph = s.fsph.data();
    MW_OMP_SIMD()
    for (size_t i = 0; i < P; i++) cphb[i] = std::cos(ph[i]);
    MW_OMP_SIMD()
    for (size_t i = 0; i < P; i++) sphb[i] = std::sin(ph[i]);
    MW_OMP_SIMD()
    for (size_t i = 0; i < n_far; i++) fcph[i] = std::cos(fph[i]);
    MW_OMP_SIMD()
    for (size_t i = 0; i < n_far; i++) fsph[i] = std::sin(fph[i]);
}

// Stage C of the NEAR pair in slot `j` (source `n`): the shipped reduced
// kernel's assembly, projected plainly or (REFL) through the Fresnel dyad at
// the pair's `ct`, `tmp`, `tnp`.
template <bool REFL>
static inline void near_stage_c(const FarRowConst &K, const FarRowScratch &s,
                                size_t j, size_t n, double ct, double tmp,
                                double tnp, std::complex<double> eps_t,
                                std::complex<double> &out_c,
                                std::complex<double> &out_s,
                                std::complex<double> &out_co) {
    const double k = K.k;
    const size_t n_qp = K.n_qp;
    const size_t S = K.S;
    const double pref_z_im = K.pref_z_im;
    const double pref_rho_const_im = K.pref_rho_const_im;
    const double *H_n = K.H_n, *sin_kH = K.sin_kH, *cos_kH = K.cos_kH;
    const double *glw_v = K.glw;
    const double *cphb = s.cphb.data(), *sphb = s.sphb.data();
    const double *rho_eval_a = s.rho_eval_a.data(), *dz1_a = s.dz1_a.data(),
                 *dz2_a = s.dz2_a.data(), *r0_1_a = s.r0_1_a.data(),
                 *r0_2_a = s.r0_2_a.data(), *td_a = s.td_a.data(),
                 *rpf_a = s.rpf_a.data();
    const double *r0q_inv_a = s.r0q_inv_a.data();
    const double *rho_p_a = s.rho_p_a.data();
    size_t base = j * S;
    double cph_2 = cphb[base + 0], sph_2 = sphb[base + 0];
    double cph_1 = cphb[base + 1], sph_1 = sphb[base + 1];
    double rho_eval = rho_eval_a[j];
    double dz1 = dz1_a[j], dz2 = dz2_a[j];
    double r0_1 = r0_1_a[j], r0_2 = r0_2_a[j];
    double td = td_a[j], rho_proj_factor = rpf_a[j];
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
    double Erho_const_re = -pref_rho_const_im * rho_diff_im;
    double Erho_const_im =  pref_rho_const_im * rho_diff_re;

    double inv_rho_eval = 1.0 / rho_eval;
    double int_inv_r0 = stable_asinh_diff(-dz1, -dz2,
                                          rho_eval * rho_eval,
                                          r0_1, r0_2);

    double int_reg_re = 0.0, int_reg_im = 0.0;
    for (size_t q = 0; q < n_qp; q++) {
        double hc_q = cphb[base + 2 + q];
        double hs_q = sphb[base + 2 + q];
        double inv_r0_q = r0q_inv_a[j * n_qp + q];
        double reg_re = (-2.0 * hs_q * hs_q) * inv_r0_q;
        double reg_im = (2.0 * hs_q * hc_q) * inv_r0_q;
        int_reg_re += reg_re * glw_v[q];
        int_reg_im += reg_im * glw_v[q];
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
    double Erho_sin_inner_re = bsin2_re - bsin1_re;
    double Erho_sin_inner_im = bsin2_im - bsin1_im;
    double pref_rho_im = pref_rho_const_im * inv_rho_eval;
    double Erho_sin_re = -pref_rho_im * Erho_sin_inner_im;
    double Erho_sin_im =  pref_rho_im * Erho_sin_inner_re;

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
    double Ez_sin_re = -pref_z_im * Ez_sin_inner_im;
    double Ez_sin_im =  pref_z_im * Ez_sin_inner_re;

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
        std::complex<double> root =
            std::sqrt(eps_t - (1.0 - ct * ct));
        std::complex<double> rho_v =
            (eps_t * ct - root) / (eps_t * ct + root);
        std::complex<double> rho_h = (ct - root) / (ct + root);
        std::complex<double> rvh = rho_v + rho_h;
        double rp = rho_p_a[j];

        auto weighted = [&](double Ez_re, double Ez_im,
                            double Erho_re, double Erho_im) {
            std::complex<double> Ez(Ez_re, Ez_im);
            std::complex<double> Erho(Erho_re, Erho_im);
            std::complex<double> tm_E =
                td * Ez + rho_proj_factor * Erho;
            std::complex<double> E_p = tnp * Ez + rp * Erho;
            return rho_v * tm_E - rvh * (tmp * E_p);
        };
        out_c = weighted(Ez_const_re, Ez_const_im,
                            Erho_const_re, Erho_const_im);
        out_s = weighted(Ez_sin_re, Ez_sin_im,
                            Erho_sin_re, Erho_sin_im);
        out_co = weighted(Ez_cos_re, Ez_cos_im,
                             Erho_cos_re, Erho_cos_im);
    } else {
        out_c = std::complex<double>(
            td * Ez_const_re + rho_proj_factor * Erho_const_re,
            td * Ez_const_im + rho_proj_factor * Erho_const_im);
        out_s = std::complex<double>(
            td * Ez_sin_re + rho_proj_factor * Erho_sin_re,
            td * Ez_sin_im + rho_proj_factor * Erho_sin_im);
        out_co = std::complex<double>(
            td * Ez_cos_re + rho_proj_factor * Erho_cos_re,
            td * Ez_cos_im + rho_proj_factor * Erho_cos_im);
    }
}

template <bool REFL>
static std::tuple<py::array_t<std::complex<double>>,
                  py::array_t<std::complex<double>>,
                  py::array_t<std::complex<double>>>
far_impl(F64 obs_centers, F64 obs_tangents, F64 src_centers, F64 src_tangents,
         F64 seg_h, double a, double k, double eta, F64 gl_t, F64 gl_w,
         F64 cos_th, F64 px_t, F64 py_t, F64 tm_p, F64 tn_p,
         std::complex<double> eps_t, double far_r, uintptr_t cancel_flag) {
    check_pair_arrays(obs_centers, obs_tangents, src_centers, src_tangents, seg_h);
    check_far_r(far_r);
    auto oc = obs_centers.unchecked<2>();
    auto ot = obs_tangents.unchecked<2>();
    auto sc = src_centers.unchecked<2>();
    auto st = src_tangents.unchecked<2>();
    auto glt = gl_t.unchecked<1>();
    auto glw = gl_w.unchecked<1>();
    if (glt.shape(0) != glw.shape(0)) {
        throw std::runtime_error("gl_t and gl_w must have matching length");
    }

    size_t M = oc.shape(0);
    size_t N = sc.shape(0);
    size_t n_qp = glt.shape(0);

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

    std::vector<double> H_n(N), sin_kH(N), cos_kH(N);
    far_source_tables(N, seg_h.data(), k, H_n.data(), sin_kH.data(), cos_kH.data());
    std::vector<double> glt_v(n_qp), glw_v(n_qp);
    for (size_t q = 0; q < n_qp; q++) {
        glt_v[q] = glt(q);
        glw_v[q] = glw(q);
    }

    const double four_pi_k = 4.0 * M_PI * k;
    const double pref_z_im = eta / four_pi_k;
    const double pref_rho_const_im = -eta / four_pi_k;
    const double a_sq = a * a;
    const double far_r2 = far_r * far_r;

    const size_t S = n_qp + 2;
    const FarRowConst K{n_qp, S, H_n.data(), sin_kH.data(), cos_kH.data(),
                        glt_v.data(), glw_v.data(), k, eta, pref_z_im,
                        pref_rho_const_im};

    MW_CANCEL_SETUP(cancel_flag);
    #pragma omp parallel
    {
    // Per-thread scratch, sized for every source being near (or far), and
    // allocated once per thread rather than per row: a row writes the first
    // n_near (n_far) entries before it reads them. Per row, the zero-filling
    // allocation was most of a far pair's cost.
    FarRowScratch s(N, N, n_qp, REFL);
    #pragma omp for schedule(static)
    for (size_t m = 0; m < M; m++) {
        MW_CANCEL_POLL();
        size_t n_near = 0, n_far = 0;

        double cmx = oc(m, 0), cmy = oc(m, 1), cmz = oc(m, 2);
        double tmx = ot(m, 0), tmy = ot(m, 1), tmz = ot(m, 2);

        // ---- Stage A: geometry, the near/far split, phases ----------------
        for (size_t n = 0; n < N; n++) {
            double cnx = sc(n, 0), cny = sc(n, 1), cnz = sc(n, 2);
            double tnx = st(n, 0), tny = st(n, 1), tnz = st(n, 2);
            double rvx = cmx - cnx, rvy = cmy - cny, rvz = cmz - cnz;
            double d2 = rvx*rvx + rvy*rvy + rvz*rvz;
            size_t mn = m * N + n;
            if (d2 > far_r2) {
                s.far_a[n_far] = far_pair<REFL>(
                    rvx, rvy, rvz, d2, tnx, tny, tnz, tmx, tmy, tmz, a_sq,
                    REFL ? px_p[mn] : 0.0, REFL ? py_p[mn] : 0.0, n);
                s.fph[n_far] = -k * s.far_a[n_far].R;
                n_far++;
                continue;
            }
            near_stage_a<REFL>(K, s, n_near++, n, rvx, rvy, rvz, tnx, tny, tnz,
                               tmx, tmy, tmz, a_sq, REFL ? px_p[mn] : 0.0,
                               REFL ? py_p[mn] : 0.0);
        }

        // ---- Stage B: vectorized sincos over the near and far phases ------
        far_row_sincos(s, n_near * S, n_far);

        // ---- Stage C (far): the lumped dipole ------------------------------
        for (size_t i = 0; i < n_far; i++) {
            const FarPair &f = s.far_a[i];
            size_t n = f.n;
            double ct = 0.0, tmp = 0.0, tnp = 0.0;
            if (REFL) {
                size_t mn = m * N + n;
                ct = cth_p[mn];
                tmp = tmp_p[mn];
                tnp = tnp_p[mn];
            }
            far_tail<REFL>(f, s.fcph[i], s.fsph[i], k, eta, H_n[n], sin_kH[n],
                           eps_t, ct, tmp, tnp, pc(m, n), ps(m, n), pco(m, n));
        }

        // ---- Stage C (near): the shipped kernel's assembly -----------------
        for (size_t j = 0; j < n_near; j++) {
            size_t n = s.near_n[j];
            double ct = 0.0, tmp = 0.0, tnp = 0.0;
            if (REFL) {
                size_t mn = m * N + n;
                ct = cth_p[mn];
                tmp = tmp_p[mn];
                tnp = tnp_p[mn];
            }
            near_stage_c<REFL>(K, s, j, n, ct, tmp, tnp, eps_t, pc(m, n),
                               ps(m, n), pco(m, n));
        }
    }
    }  // omp parallel

    MW_THROW_IF_ABORTED();
    return std::make_tuple(Phi_const, Phi_sin, Phi_cos);
}

template <bool REFL>
static std::tuple<py::array_t<std::complex<double>>,
                  py::array_t<std::complex<double>>,
                  py::array_t<std::complex<double>>>
overlay_impl(C128 base_const, C128 base_sin, C128 base_cos, F64 obs_centers,
             F64 obs_tangents, F64 src_centers, F64 src_tangents, F64 seg_h,
             double a, double k, double eta, bool cos_minus_one, F64 cos_th,
             F64 px_t, F64 py_t, F64 tm_p, F64 tn_p, std::complex<double> eps_t,
             double far_r, uintptr_t cancel_flag) {
    check_pair_arrays(obs_centers, obs_tangents, src_centers, src_tangents, seg_h);
    check_far_r(far_r);
    auto oc = obs_centers.unchecked<2>();
    auto ot = obs_tangents.unchecked<2>();
    auto sc = src_centers.unchecked<2>();
    auto st = src_tangents.unchecked<2>();
    auto sh = seg_h.unchecked<1>();
    size_t M = oc.shape(0);
    size_t N = sc.shape(0);
    for (const auto *b : {&base_const, &base_sin, &base_cos}) {
        if ((*b).ndim() != 2 || (size_t)(*b).shape(0) != M ||
            (size_t)(*b).shape(1) != N) {
            throw std::runtime_error("base tensors must all have shape (M_obs, N_src)");
        }
    }
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
    const std::complex<double> *bc = base_const.data();
    const std::complex<double> *bs = base_sin.data();
    const std::complex<double> *bco = base_cos.data();

    py::array_t<std::complex<double>> Phi_const({M, N});
    py::array_t<std::complex<double>> Phi_sin({M, N});
    py::array_t<std::complex<double>> Phi_cos({M, N});
    auto pc = Phi_const.mutable_unchecked<2>();
    auto ps = Phi_sin.mutable_unchecked<2>();
    auto pco = Phi_cos.mutable_unchecked<2>();

    py::gil_scoped_release release;

    std::vector<double> H_n(N), sin_kH(N);
    for (size_t n = 0; n < N; n++) {
        H_n[n] = 0.5 * sh(n);
        sin_kH[n] = std::sin(k * H_n[n]);
    }
    const double a_sq = a * a;
    const double far_r2 = far_r * far_r;

    MW_CANCEL_SETUP(cancel_flag);
    #pragma omp parallel for schedule(static)
    for (size_t m = 0; m < M; m++) {
        MW_CANCEL_POLL();
        double cmx = oc(m, 0), cmy = oc(m, 1), cmz = oc(m, 2);
        double tmx = ot(m, 0), tmy = ot(m, 1), tmz = ot(m, 2);
        for (size_t n = 0; n < N; n++) {
            size_t mn = m * N + n;
            double cnx = sc(n, 0), cny = sc(n, 1), cnz = sc(n, 2);
            double rvx = cmx - cnx, rvy = cmy - cny, rvz = cmz - cnz;
            double d2 = rvx*rvx + rvy*rvy + rvz*rvz;
            if (!(d2 > far_r2)) {
                pc(m, n) = bc[mn];
                ps(m, n) = bs[mn];
                pco(m, n) = bco[mn];
                continue;
            }
            double tnx = st(n, 0), tny = st(n, 1), tnz = st(n, 2);
            FarPair f = far_pair<REFL>(rvx, rvy, rvz, d2, tnx, tny, tnz, tmx,
                                       tmy, tmz, a_sq, REFL ? px_p[mn] : 0.0,
                                       REFL ? py_p[mn] : 0.0, n);
            double phase = -k * f.R;
            far_tail<REFL>(f, std::cos(phase), std::sin(phase), k, eta, H_n[n],
                           sin_kH[n], eps_t, REFL ? cth_p[mn] : 0.0,
                           REFL ? tmp_p[mn] : 0.0, REFL ? tnp_p[mn] : 0.0,
                           pc(m, n), ps(m, n), pco(m, n), cos_minus_one);
        }
    }
    MW_THROW_IF_ABORTED();
    return std::make_tuple(Phi_const, Phi_sin, Phi_cos);
}

// ---------------------------------------------------------------------------
// The fused fill's far-aware rows (momwire#2082, step 1): `sinusoidal_fill_rows`
// (`_sinusoidal_core.cpp`, momwire#2069) with NEC-2's interaction
// approximation, so that the approximation keeps the fill fused.
//
// THE BITS are the two-pass far route's (`sinusoidal_field_tensor_far`, then
// `reduce_phi_band`), by construction:
//  - a pair's Φ entries come from the same three stages `far_impl` runs
//    (`near_stage_a`, `far_row_sincos`, `near_stage_c`, and `far_pair` /
//    `far_tail` for a far pair), one function for both routes;
//  - the stage-B sweeps keep each phase's place in the libmvec split. The
//    whole-row sweep `far_impl` runs takes whole vectors from the start of its
//    compacted near (far) phases and libm's scalar functions on the last
//    (count mod VF). Here the near pairs are compacted as they are found and
//    swept each time `kNearFlush` of them are waiting, the far pairs every
//    `kFarFlush`. Both are multiples of 16, so every flush but a row's last
//    starts and ends on a whole vector of every width (AVX-512's 8 with GCC's
//    2x unroll included), and the last flush's scalar tail is exactly the
//    whole-row sweep's: the near phases are the same `kNearFlush`·S-aligned
//    slices of the same compacted sequence. This is #2070's tiling argument
//    (`kSrcTile`) applied to the compacted sequence instead of the source axis;
//  - the PEC image (img_c set) is the same route over the mirrored sources,
//    subtracted element by element as `Phi_c -= Phi_i[0]` subtracts it, after
//    the whole direct row is in place;
//  - the contraction is `sinusoidal_fill_rows`' copy of `reduce_phi_band`'s
//    loop, copied again below operation for operation.
//
// Why compact rather than tile the source axis as #2070 does: about 70 % of
// a large deck's pairs are far, and a far pair costs one sincos against a
// near pair's n_qp + 2. A tile of 256 sources would sweep its near phases in
// short, ragged runs whose tails could not line up with the whole row's; the
// compacted runs are full and line up by construction. The scratch is a
// flush's worth, not a row's: about 92 KB of near slots and 48 KB of far ones
// per thread, beside the row's three Φ (48·N bytes).
//
// The reduced kernel only: the extended kernel's far pairs are overlaid on
// its shipped tensors (`sinusoidal_far_overlay`) and keep the two-pass band
// until it has a lumped twin (#2082, follow-up 3).
constexpr size_t kNearFlush = 256;
constexpr size_t kFarFlush = 1024;
static_assert(kNearFlush % 16 == 0 && kFarFlush % 16 == 0,
              "a flush must end on a whole vector of every width");

// One observer row against every source, far-aware, into x0/x1/x2: written
// (SUB false, the direct block) or subtracted from (SUB true, the image).
template <bool SUB>
static inline void far_row_fused(const FarRowConst &K, FarRowScratch &s,
                                 const double *cm, const double *tm, double a_sq,
                                 const double *sc, const double *st, size_t N,
                                 double far_r2, std::complex<double> *x0,
                                 std::complex<double> *x1,
                                 std::complex<double> *x2) {
    using cd = std::complex<double>;
    const double k = K.k, eta = K.eta;
    const size_t S = K.S;
    const double tmx = tm[0], tmy = tm[1], tmz = tm[2];
    auto put = [](cd &dst, const cd &v) {
        if (SUB)
            dst -= v;
        else
            dst = v;
    };
    auto flush = [&](size_t n_near, size_t n_far) {
        far_row_sincos(s, n_near * S, n_far);
        for (size_t i = 0; i < n_far; i++) {
            const FarPair &f = s.far_a[i];
            const size_t n = f.n;
            cd c, sn, co;
            far_tail<false>(f, s.fcph[i], s.fsph[i], k, eta, K.H_n[n],
                            K.sin_kH[n], cd(0.0, 0.0), 0.0, 0.0, 0.0, c, sn, co);
            put(x0[n], c);
            put(x1[n], sn);
            put(x2[n], co);
        }
        for (size_t j = 0; j < n_near; j++) {
            const size_t n = s.near_n[j];
            cd c, sn, co;
            near_stage_c<false>(K, s, j, n, 0.0, 0.0, 0.0, cd(0.0, 0.0), c, sn, co);
            put(x0[n], c);
            put(x1[n], sn);
            put(x2[n], co);
        }
    };
    size_t n_near = 0, n_far = 0;
    for (size_t n = 0; n < N; n++) {
        const double tnx = st[3 * n + 0], tny = st[3 * n + 1], tnz = st[3 * n + 2];
        const double rvx = cm[0] - sc[3 * n + 0], rvy = cm[1] - sc[3 * n + 1],
                     rvz = cm[2] - sc[3 * n + 2];
        const double d2 = rvx*rvx + rvy*rvy + rvz*rvz;
        if (d2 > far_r2) {
            s.far_a[n_far] = far_pair<false>(rvx, rvy, rvz, d2, tnx, tny, tnz,
                                             tmx, tmy, tmz, a_sq, 0.0, 0.0, n);
            s.fph[n_far] = -k * s.far_a[n_far].R;
            if (++n_far == kFarFlush) {
                flush(0, n_far);
                n_far = 0;
            }
            continue;
        }
        near_stage_a<false>(K, s, n_near, n, rvx, rvy, rvz, tnx, tny, tnz, tmx,
                            tmy, tmz, a_sq, 0.0, 0.0);
        if (++n_near == kNearFlush) {
            flush(n_near, 0);
            n_near = 0;
        }
    }
    flush(n_near, n_far);
}

}  // namespace

namespace mw {
namespace sinusoidal {

std::tuple<nd::array_t<std::complex<double>>, nd::array_t<std::complex<double>>, nd::array_t<std::complex<double>>>
sinusoidal_field_tensor_far(F64 obs_centers, F64 obs_tangents, F64 src_centers,
                            F64 src_tangents, F64 seg_h, double a, double k,
                            double eta, F64 gl_t, F64 gl_w, double far_r,
                            uintptr_t cancel_flag) {
    F64 empty(std::vector<py::ssize_t>{0, 0});
    return far_impl<false>(obs_centers, obs_tangents, src_centers, src_tangents,
                           seg_h, a, k, eta, gl_t, gl_w, empty, empty, empty,
                           empty, empty, std::complex<double>(0.0, 0.0), far_r,
                           cancel_flag);
}

std::tuple<nd::array_t<std::complex<double>>, nd::array_t<std::complex<double>>, nd::array_t<std::complex<double>>>
sinusoidal_field_tensor_refl_far(F64 obs_centers, F64 obs_tangents,
                                 F64 src_centers, F64 src_tangents, F64 seg_h,
                                 double a, double k, double eta, F64 gl_t,
                                 F64 gl_w, F64 cos_th, F64 px, F64 py_, F64 tm_p,
                                 F64 tn_p, std::complex<double> eps_t,
                                 double far_r, uintptr_t cancel_flag) {
    return far_impl<true>(obs_centers, obs_tangents, src_centers, src_tangents,
                          seg_h, a, k, eta, gl_t, gl_w, cos_th, px, py_, tm_p,
                          tn_p, eps_t, far_r, cancel_flag);
}

std::tuple<nd::array_t<std::complex<double>>, nd::array_t<std::complex<double>>, nd::array_t<std::complex<double>>>
sinusoidal_far_overlay(C128 base_const, C128 base_sin, C128 base_cos,
                       F64 obs_centers, F64 obs_tangents, F64 src_centers,
                       F64 src_tangents, F64 seg_h, double a, double k,
                       double eta, bool refl, bool cos_minus_one, F64 cos_th,
                       F64 px, F64 py_,
                       F64 tm_p, F64 tn_p, std::complex<double> eps_t,
                       double far_r, uintptr_t cancel_flag) {
    if (refl) {
        return overlay_impl<true>(base_const, base_sin, base_cos, obs_centers,
                                  obs_tangents, src_centers, src_tangents, seg_h,
                                  a, k, eta, cos_minus_one, cos_th, px, py_, tm_p,
                                  tn_p, eps_t, far_r, cancel_flag);
    }
    return overlay_impl<false>(base_const, base_sin, base_cos, obs_centers,
                               obs_tangents, src_centers, src_tangents, seg_h, a,
                               k, eta, cos_minus_one, cos_th, px, py_, tm_p, tn_p,
                               eps_t, far_r, cancel_flag);
}

// momwire#2082 step 1: `sinusoidal_fill_rows` with NEC-2's interaction
// approximation at `far_r` metres (`kNearFlush` and `far_row_fused` above
// say why the bits are the two-pass far route's).
void sinusoidal_fill_rows_far(const FillRowsArgs &A, double far_r) {
    if (A.ek) {
        throw std::runtime_error(
            "sinusoidal_fill_rows: far_r serves the reduced kernel only; the "
            "extended kernel keeps the two-pass band");
    }
    check_far_r(far_r);
    const size_t M = A.M, N = A.N, n_qp = A.n_qp, cols = A.cols;
    const double k = A.k, eta = A.eta;
    const bool image = A.img_c != nullptr;

    // Per-source-segment precompute, `far_impl`'s own function.
    std::vector<double> H_n(N), sin_kH(N), cos_kH(N);
    far_source_tables(N, A.seg_h, k, H_n.data(), sin_kH.data(), cos_kH.data());
    std::vector<double> glt_v(n_qp), glw_v(n_qp);
    for (size_t q = 0; q < n_qp; q++) {
        glt_v[q] = A.gl_t[q];
        glw_v[q] = A.gl_w[q];
    }
    const double four_pi_k = 4.0 * M_PI * k;
    const double pref_z_im = eta / four_pi_k;
    const double pref_rho_const_im = -eta / four_pi_k;
    const double far_r2 = far_r * far_r;
    const size_t S = n_qp + 2;
    const FarRowConst K{n_qp, S, H_n.data(), sin_kH.data(), cos_kH.data(),
                        glt_v.data(), glw_v.data(), k, eta, pref_z_im,
                        pref_rho_const_im};

    int nt = 1;
#ifdef _OPENMP
    nt = A.n_threads > 0 ? A.n_threads : omp_get_max_threads();
#endif
    if (nt < 1) nt = 1;

    using cd = std::complex<double>;
    MW_CANCEL_SETUP(A.cancel_flag);
    #pragma omp parallel num_threads(nt)
    {
        // One thread's working set, allocated once: a flush of near and of
        // far slots, and the row's three Φ (48·N bytes).
        FarRowScratch s(kNearFlush, kFarFlush, n_qp, false);
        std::vector<cd> row(3 * N);
        #pragma omp for schedule(static)
        for (size_t m = 0; m < M; m++) {
            MW_CANCEL_POLL();
            const double a = A.a_rows[m];
            const double a_sq = a * a;
            const double *cm = A.obs_c + 3 * m, *tm = A.obs_t + 3 * m;
            cd *x0 = row.data(), *x1 = x0 + N, *x2 = x1 + N;
            far_row_fused<false>(K, s, cm, tm, a_sq, A.src_c, A.src_t, N, far_r2,
                                 x0, x1, x2);
            // `Phi_c -= Phi_i[0]` (and _s, _co), element by element.
            if (image)
                far_row_fused<true>(K, s, cm, tm, a_sq, A.img_c, A.img_t, N,
                                    far_r2, x0, x1, x2);
            // `reduce_phi_band`'s loop, element for element: the copy in
            // `sinusoidal_fill_rows` (`_sinusoidal_core.cpp`), verbatim.
            const double *x[3] = {reinterpret_cast<const double *>(x0),
                                  reinterpret_cast<const double *>(x1),
                                  reinterpret_cast<const double *>(x2)};
            cd *o = A.out + static_cast<std::ptrdiff_t>(m) * A.out_rs;
            for (size_t j = 0; j < cols; ++j) {
                double sr[3], si[3];
                for (int c = 0; c < 3; c++) {
                    // csr_matvecs: a zeroed y, then `y += a * x` per stored entry
                    double yr = 0.0, yi = 0.0;
                    const FillRowsCsc &mc = A.csc[c];
                    for (int64_t e = mc.ptr[j]; e < mc.ptr[j + 1]; ++e) {
                        const double ar = mc.val[2 * e], ai = mc.val[2 * e + 1];
                        const double *xe = x[c] + 2 * mc.ind[e];
                        const double xr = xe[0], xi = xe[1];
                        const double pr = ar * xr - ai * xi;
                        const double pi = ar * xi + ai * xr;
                        yr = yr + pr;
                        yi = yi + pi;
                    }
                    sr[c] = yr;
                    si[c] = yi;
                }
                o[static_cast<std::ptrdiff_t>(j) * A.out_cs] =
                    cd((sr[0] + sr[1]) + sr[2], (si[0] + si[1]) + si[2]);
            }
        }
    }
    MW_THROW_IF_ABORTED();
}

int64_t far_range_ties(F64 obs_centers, F64 src_centers, double far_r, double rel) {
    if (obs_centers.ndim() != 2 || src_centers.ndim() != 2 ||
        obs_centers.shape(1) != 3 || src_centers.shape(1) != 3) {
        throw std::runtime_error("center arrays must have shape (N, 3)");
    }
    check_far_r(far_r);
    auto oc = obs_centers.unchecked<2>();
    auto sc = src_centers.unchecked<2>();
    const int64_t M = static_cast<int64_t>(oc.shape(0));
    const int64_t N = static_cast<int64_t>(sc.shape(0));
    // On squared distances: |d - R| <= rel R  <=>  d^2 in [(1-rel)^2, (1+rel)^2] R^2.
    const double lo = (1.0 - rel) * (1.0 - rel) * far_r * far_r;
    const double hi = (1.0 + rel) * (1.0 + rel) * far_r * far_r;
    int64_t count = 0;
    py::gil_scoped_release release;
    #pragma omp parallel for schedule(static) reduction(+:count)
    for (int64_t m = 0; m < M; m++) {
        double cmx = oc(m, 0), cmy = oc(m, 1), cmz = oc(m, 2);
        for (int64_t n = 0; n < N; n++) {
            double rvx = cmx - sc(n, 0), rvy = cmy - sc(n, 1), rvz = cmz - sc(n, 2);
            double d2 = rvx*rvx + rvy*rvy + rvz*rvz;
            if (d2 >= lo && d2 <= hi) count++;
        }
    }
    return count;
}

}  // namespace sinusoidal
}  // namespace mw
