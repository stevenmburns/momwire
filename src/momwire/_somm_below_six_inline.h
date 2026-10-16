// The below/below Sommerfeld integrand and one node of its contour
// (`_sommerfeld_below._integrand_six_below` / `_six_integrals_below`), with no
// Python: shared by the extension's `below_six_integrals_batch` and the
// executable's own below/below table (momwire#1512 B5), so both run the same
// code and the same bits. Moved verbatim from `_accel_mw568.cpp`.
#pragma once

#include <algorithm>
#include <cmath>
#include <complex>

#include "_branch_cut_inline.h"
#include "_cdiv_inline.h"
#include "_contour_engine_inline.h"
#include "_contour_lanes_inline.h"
#include "_fma_inline.h"

namespace mw568_below {
using mw_contour::cd;

// The branch cut: one definition for all three call sites, across both
// extensions -- the full rationale (the two-sqrt spelling IS the branch
// choice) lives with the definition in _branch_cut_inline.h (#714).
using mw_branch::gamma_cut;


// `_sommerfeld._d12(lam, k1, k2)` -- NEC eqs 154-155 (the 2s of eqs 141-142
// are inside). The below family calls it SWAPPED, (k1, k2) = (k_p, k_m), so
// the GROUND sits in the decay slot: `_d12` builds its kernels around
// gamma_2 = gamma(k2) as the decay gamma and k1 as the other medium, and that
// swapped reading IS the below/below D-pair (measured at 0.0 relative
// difference against the #524 phase-0 prototype's independent generalized
// form). `g2` is returned because the integrand needs gamma_m itself.
static inline void d12(const cd &lam, const cd &k1, const cd &k2, cd &d1,
                       cd &d2, cd &g2) {
    const cd g1 = gamma_cut(lam, k1);
    g2 = gamma_cut(lam, k2);
    // Fused complex products (momwire#1194, _fma_inline.h), as in the
    // above-ground twin `somm::integrand_six`.
    const cd k1s = mw_fma::mul(k1, k1);
    const cd k2s = mw_fma::mul(k2, k2);
    const cd g2ks = mw_fma::mul(g2, k1s + k2s);
    // The four divides are mw_cdiv::div, libgcc's own rounding without the
    // call (momwire#1441).
    d1 = mw_cdiv::div(2.0, g1 + g2) - mw_cdiv::div(2.0 * k2s, g2ks);
    d2 = mw_cdiv::div(2.0, mw_fma::mul_add(k1s, g2, mw_fma::mul(k2s, g1))) -
         mw_cdiv::div(2.0, g2ks);
}

// `_integrand_six_below`, term for term and in its order:
//
//   0: d2 V/drho^2      [D2 e^{-gamma_m h} lam^3 (J1/x - J0)]
//   1: d2 V/dz^2        [D2 gamma_m^2 e^{-gamma_m h} J0 lam]
//   2: d2 V/drho dz     [-D2 gamma_m e^{-gamma_m h} J1 lam^2]   <- the +/-=- sign
//   3: (1/rho) dV/drho  [-D2 e^{-gamma_m h} (J1/x) lam^3]
//   4: V                [D2 e^{-gamma_m h} J0 lam]
//   5: U                [D1 e^{-gamma_m h} J0 lam]
//
// INDEX 2 IS THE WHOLE OF THE SIGN STORY and the one line of this file that a
// mutation test exists for. h = |z + z'| with z + z' < 0 below the interface,
// so d/dz e^{-gamma_m|z + z'|} = +gamma_m e where the +/-=+ case gets
// -gamma_2 e; with dJ0/drho = -lam J1 the product lands at -D2 gamma_m J1 lam^2,
// the NEGATIVE of `_sommerfeld._integrand_six`'s index 2. Every other
// component is identical between the families once the kernel pair is the
// swapped one -- which is what `test_gu2_1_the_integrand_z_derivative_flips_
// and_nothing_else_does` asserts on the numpy side and what
// `test_g5686_the_index_2_sign_is_load_bearing` asserts on this one.
//
// Bessel form only: there is no Hankel twin here, because there is no fig-14
// contour here.
struct SixBelow {
    double rho;
    double h;
    cd k_p;
    cd k_m;
    // Forced inline (momwire#1194): with its products written as mw_fma::mul
    // GCC 11 priced this body over its inline limit and called it out of line
    // from the contour engine, once per quadrature node, which cost the below
    // grid fill ~5 %; inlined, it is within 1.5 % of the contracted build.
    MW_FMA_ALWAYS_INLINE void operator()(const cd &lam, cd *out) const {
        cd d1, d2, g_m;
        d12(lam, k_p, k_m, d1, d2, g_m);
        const cd e = std::exp(-g_m * h);
        const cd x = lam * rho;
        cd b0, b1x;
        mw_contour::bessel_j0_j1x(x, b0, b1x);
        using mw_fma::mul;  // fused products (momwire#1194)
        const cd l2 = mul(lam, lam);
        const cd l3 = mul(l2, lam);
        const cd common = mul(d2, e);
        const cd cg = mul(common, g_m);
        out[0] = mul(mul(common, b1x - b0), l3);
        out[1] = mul(mul(mul(cg, g_m), b0), lam);
        out[2] = -mul(mul(cg, mul(b1x, x)), l2);
        out[3] = -mul(mul(common, b1x), l3);
        out[4] = mul(mul(common, b0), lam);
        out[5] = mul(mul(mul(d1, e), b0), lam);
    }
    // Lanes (momwire#1441 step 3): four contour nodes at a time, each lane
    // this operator() bit for bit (_contour_lanes_inline.h). The contour
    // engine's TailWrap hands over real abscissae t (lam = (t, +0)), its
    // HeadWrap the detour's lam; a lane the lanes do not serve (the Bessel
    // pair's |x| < 1e-6 branch at rho = 0) is recomputed here by the scalar
    // operator().
    static const bool kLanes = true;
    bool use_lanes = true;
    bool lanes_on() const { return use_lanes; }
    void lanes_at(const mw_sl::C &lam, bool real_axis, cd *fv) const {
        mw_sl::C o[6];
        const int left = mw_contour_lanes::six_below(rho, h, k_p, k_m, lam, real_axis, o);
        for (int c = 0; c < 6; ++c) {
            const mw_sl::Buf re = mw_sl::spill(o[c].re), im = mw_sl::spill(o[c].im);
            for (int l = 0; l < mw_sl::W; ++l) fv[l * 6 + c] = cd(re.v[l], im.v[l]);
        }
        if (left) {
            for (int l = 0; l < mw_sl::W; ++l)
                if (left & (1 << l)) (*this)(mw_sl::lane(lam, l), fv + l * 6);
        }
    }
    void tail_lanes(const double *t, cd *fv) const {
        lanes_at(mw_sl::C{mw_sl::loadu(t), mw_sl::set1(0.0)}, true, fv);
    }
    void head_lanes(const double *lr, const double *li, cd *fv) const {
        lanes_at(mw_sl::C{mw_sl::loadu(lr), mw_sl::loadu(li)}, false, fv);
    }
};
// What one node of `_six_integrals_below` produces: the six values plus
// everything `Health.note` / `Health.note_selfconv` are handed on the numpy
// side. `selfconv` is -1.0 when the coarse machine was not asked for, so the
// Python layer can tell "not measured" from "measured zero".
struct SixResult {
    cd val[6];
    int head_panels;
    int tail_panels;
    bool converged;
    bool accel;
    double selfconv;
};

// One node: the fine contour, then optionally the coarse self-convergence
// twin (Gauss-16, rtol x100, a shallower detour -- `_six_integrals_below`'s
// `selfconv=True` branch). The componentwise relative spread is the numpy
// spelling exactly: max over components of |fine - coarse| / max(|fine|, 1e-300).
static void six_below_one(double rho, double h, double k_p, const cd &k_m,
                          double rtol_fine, int depth, double detour,
                          const double *gx, const double *gw, int ng,
                          bool selfconv, double rtol_coarse, int depth_coarse,
                          double detour_coarse, const double *gxc,
                          const double *gwc, int ngc, int max_panels,
                          const mw_contour::WaRule *wa, SixResult &r,
                          bool lanes = mw_sl::kActive) {
    SixBelow f;
    f.use_lanes = lanes;
    f.rho = rho;
    f.h = h;
    f.k_p = cd(k_p, 0.0);
    f.k_m = k_m;
    const mw_contour::ContourHealth hh = mw_contour::run_contour<6>(
        f, k_p, k_m, rho, h, rtol_fine, depth, detour, gx, gw, ng, max_panels,
        r.val, wa);
    r.head_panels = hh.head_panels;
    r.tail_panels = hh.tail_panels;
    r.converged = hh.converged;
    r.accel = hh.accel;
    r.selfconv = -1.0;
    if (selfconv) {
        cd coarse[6];
        mw_contour::run_contour<6>(f, k_p, k_m, rho, h, rtol_coarse,
                                   depth_coarse, detour_coarse, gxc, gwc, ngc,
                                   max_panels, coarse, wa);
        double worst = 0.0;
        for (int c = 0; c < 6; ++c) {
            const double scale = std::max(std::abs(r.val[c]), 1e-300);
            const double rel = std::abs(r.val[c] - coarse[c]) / scale;
            if (rel > worst) worst = rel;
        }
        r.selfconv = worst;
    }
}

}  // namespace mw568_below
