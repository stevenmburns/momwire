// The transmitted Sommerfeld family's integrand, one node of its contour and
// the projected pair kernel (`_sommerfeld_transmitted._integrand_six_
// transmitted` / `_six_integrals_transmitted` / `transmitted_field_proj_*`),
// with no Python: shared by the extension's `transmitted_six_integrals_batch`
// and `transmitted_field_proj_batch` and by the executables' transmitted
// table (`_somm_transmitted_core.cpp`, momwire#1718 S12), so both run the same
// code and the same bits. Moved verbatim from `_accel_mw568.cpp` (the pybind
// index type spelled `std::ptrdiff_t`, which is `py::ssize_t`; the branch cut
// named by its definition, `mw_branch::gamma_cut`, which `mw568_below`'s
// using-declaration names).
#pragma once

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstddef>

#include "_accel_somm_proj_inline.h"
#include "_branch_cut_inline.h"
#include "_cdiv_inline.h"
#include "_contour_engine_inline.h"

namespace mw568_trans {
using mw_contour::cd;

// `1j` (the `mw568_below::MW_BJ` of `_accel_mw568.cpp`, the same value).
static const cd MW_BJ_T(0.0, 1.0);

// `_sommerfeld._gamma` is SHARED with U2 rather than re-transcribed
// (`mw_branch::gamma_cut` — since #714 the definition itself lives in
// _branch_cut_inline.h, one copy for both extensions): it is the two-sqrt
// product
// sqrt(-j(lam - k)) sqrt(j(lam + k)), and the spelling IS the branch choice --
// vertical cuts running DOWN from +k and UP from -k, neither of which the
// head's first-quadrant detour crosses. Two copies of that could drift; one
// cannot. This family calls it TWICE per evaluation, at k_p and at k_m,
// because its two exponential legs live in different media.

// `_integrand_six_transmitted`, term for term and in its order:
//
//   0: d2 V_T/drho dz     [a (-lam J1) d_z]
//   1: (d2/dz2 + k_p^2)V_T [a lam^2 J0]
//   2: d2 V_T/drho^2      [a lam^2 (J1/x - J0)]
//   3: (1/rho) dV_T/drho  [-a lam^2 (J1/x)]
//   4: d2 V_T/drho dz'    [a (-lam J1) d_zp]
//   5: U_T                [u J0]
//
// with a = 2 lam e / (k_m^2 gamma_p + k_p^2 gamma_m) the (7f) V_T kernel and
// u = 2 lam e / (gamma_m + gamma_p) the (7g) U_T one, e the two-leg
// exponential. All five derivatives are ANALYTIC under the integral sign
// (d/drho J0 = -lam J1, d2/drho2 via the Bessel ODE identity, d/dz -> x(-g_p),
// d/dz' -> x(+g_m)) -- never a differenced interpolant, which is the
// LLNL-TR-490316 rule the phase-0 comment records.
//
// INDEX 1 IS SPELLED lam^2 AND THAT IS LOAD-BEARING. (d2/dz2 + k^2) is
// gamma^2 + k^2 in either medium, and gamma^2 + k^2 is a difference of two
// O(k^2) numbers whose result is O(lam^2): at the bottom of the contour
// (lam ~ 1e-3, |k_m| ~ 0.6) it cancels away ELEVEN digits. The swapped and
// unswapped spellings of the same quantity were measured 3.5e-11 apart before
// the numpy line said lam^2. Writing `g_p*g_p + k_p*k_p` here would not fail
// loudly; it would quietly cost a grazing gate. `test_g56812_index_1_is_
// lambda_squared_not_gamma_squared_plus_k_squared` is the mutation gate.
//
// INDEX 4 IS THE FIFTH SURFACE. d/dz' multiplies by +gamma_m where d/dz
// multiplies by -gamma_p, and the two are different functions of lam, so
// index 4 is NOT a multiple of index 0 and T_z^H is not -cos(phi) T_rho^V.
// That collapse is a +-=+ identity that does not survive here; the #553 arc
// gated it hardest because the licensed engine departs from the phase-0
// prototype on exactly this component by O(1), with empymod siding with the
// prototype.
struct SixTransmitted {
    double rho;
    double z;
    double zp;
    double k_p;
    cd k_m;
    bool swap;
    void operator()(const cd &lam, cd *out) const {
        const cd kp(k_p, 0.0);
        const cd g_p = mw_branch::gamma_cut(lam, kp);
        const cd g_m = mw_branch::gamma_cut(lam, k_m);
        const double azp = std::fabs(zp);
        cd e, d_z, d_zp;
        if (swap) {
            // Source ABOVE at z' > 0 carrying gamma_p, observer BELOW at
            // z < 0 carrying gamma_m: the same integral with the legs
            // exchanged, which is the reciprocity identity phase 0 measured
            // at 0.0 relative.
            e = 2.0 * std::exp(-g_p * azp - g_m * std::fabs(z)) * lam;
            d_z = g_m;
            d_zp = -g_p;
        } else {
            e = 2.0 * std::exp(-g_m * azp - g_p * z) * lam;
            d_z = -g_p;
            d_zp = g_m;
        }
        const cd zzk = lam * lam;  // NOT g^2 + k^2 -- see the note above
        const double kp2 = k_p * k_p;
        const cd a = mw_cdiv::div(e, k_m * k_m * g_p + kp2 * g_m);  // (7f)
        const cd u = mw_cdiv::div(e, g_m + g_p);                    // (7g)
        const cd x = lam * rho;
        cd b0, b1x;
        mw_contour::bessel_j0_j1x(x, b0, b1x);
        const cd j1 = b1x * x;
        const cd dr = -(lam * j1);  // d/drho J0 = -lam J1
        out[0] = a * dr * d_z;
        out[1] = a * zzk * b0;
        out[2] = a * lam * lam * (b1x - b0);
        out[3] = a * (-(lam * lam * b1x));
        out[4] = a * dr * d_zp;
        out[5] = u * b0;
    }
};

// What one node of `_six_integrals_transmitted` produces: the six values plus
// everything `Health.note` / `Health.note_selfconv` are handed on the numpy
// side. `selfconv` is -1.0 when the coarse machine was not asked for, so the
// Python layer can tell "not measured" from "measured zero".
struct SixResultT {
    cd val[6];
    int head_panels;
    int tail_panels;
    bool converged;
    bool accel;
    double selfconv;
};

// One node: the fine contour, then optionally the coarse self-convergence
// twin (Gauss-16, rtol x100, a shallower detour).
//
// `h_decay = |z'| + |z|` is the TAIL'S DECAY LENGTH, not a coordinate: both
// gammas go to lam at large lam, so e^{-g_m|z'| - g_p z} -> e^{-lam(|z'|+z)}
// exactly as the below family's e^{-g_m h} -> e^{-lam h}. That one argument is
// the whole of the sharing with U2's contour, and `max_panels` is the other
// handle -- this family's grazing rows need 6000 where any below/below
// geometry is done in a few hundred.
static void six_transmitted_one(double rho, double z, double zp, double k_p,
                                const cd &k_m, bool swap, double rtol_fine,
                                int depth, double detour, const double *gx,
                                const double *gw, int ng, bool selfconv,
                                double rtol_coarse, int depth_coarse,
                                double detour_coarse, const double *gxc,
                                const double *gwc, int ngc, int max_panels,
                                SixResultT &r) {
    SixTransmitted f;
    f.rho = rho;
    f.z = z;
    f.zp = zp;
    f.k_p = k_p;
    f.k_m = k_m;
    f.swap = swap;
    const double h_decay = std::fabs(zp) + std::fabs(z);
    const mw_contour::ContourHealth hh = mw_contour::run_contour<6>(
        f, k_p, k_m, rho, h_decay, rtol_fine, depth, detour, gx, gw, ng,
        max_panels, r.val);
    r.head_panels = hh.head_panels;
    r.tail_panels = hh.tail_panels;
    r.converged = hh.converged;
    r.accel = hh.accel;
    r.selfconv = -1.0;
    if (selfconv) {
        cd coarse[6];
        mw_contour::run_contour<6>(f, k_p, k_m, rho, h_decay, rtol_coarse,
                                   depth_coarse, detour_coarse, gxc, gwc, ngc,
                                   max_panels, coarse);
        double worst = 0.0;
        for (int c = 0; c < 6; ++c) {
            const double scale = std::max(std::abs(r.val[c]), 1e-300);
            const double rel = std::abs(r.val[c] - coarse[c]) / scale;
            if (rel > worst) worst = rel;
        }
        r.selfconv = worst;
    }
}

// `TransmittedGrid`'s tabulation, flattened for the inner loop.
//
// NOT `somm_proj::GridView`, and not a widening of it. That view is built for
// the +-=+ / +-=- REMAINDER grids: four surfaces, four-or-six (R1, theta)
// REGIONS selected by (r_break, th_split, r_near), and a LINEAR radial axis
// per region. The transmitted tabulation is one uniform lattice in
// (ln R, theta) -- logarithmic, because the transmitted surface is the whole
// field and carries the source's own 1/R^2 near zone, so there is no finite
// R1 -> 0 limit node the way the remainder families have -- with FIVE surfaces
// and a third axis, the log|z'| ladder. Bending `build_grid_view` around a
// different region model, a different axis law, a different surface count and
// an extra dimension would have made one struct serve two unrelated layouts;
// this is the small honest one.
struct TGridView {
    const cd *vals;  // (n_zp, 5, n_r, n_th), C-contiguous
    std::ptrdiff_t n_zp, n_r, n_th;
    double lnr0, dlnr, th0, dth, lnz0, dlnz;
    double tiny, half_pi;
};

// `TransmittedGrid.eval`'s interpolation, WITHOUT its four refusals: cubic
// Lagrange in ln R and theta, and -- when the ladder has more than one rung --
// cubic in ln|z'| on the divided-out surfaces, which is phase 0's measured
// scheme. The refusals stay in Python; see the comment on
// `transmitted_field_proj_batch`.
static inline void tgrid_eval(const TGridView &G, double R, double theta,
                              double azp, cd *surf) {
    const double fr = (std::log(std::max(R, 1e-300)) - G.lnr0) / G.dlnr;
    int i0 = (int)std::floor(fr) - 1;
    if (i0 < 0) i0 = 0;
    else if (i0 > (int)G.n_r - 4) i0 = (int)G.n_r - 4;
    double thc = theta;
    if (thc < 0.0) thc = 0.0;
    else if (thc > G.half_pi) thc = G.half_pi;
    const double ft = (thc - G.th0) / G.dth;
    int j0 = (int)std::floor(ft) - 1;
    if (j0 < 0) j0 = 0;
    else if (j0 > (int)G.n_th - 4) j0 = (int)G.n_th - 4;
    double wr[4], wt[4];
    somm_proj::lagrange4(fr - i0, wr);
    somm_proj::lagrange4(ft - j0, wt);

    const std::ptrdiff_t nth = G.n_th, nr = G.n_r;
    const std::ptrdiff_t plane = nr * nth;       // one surface at one rung
    const std::ptrdiff_t rung = 5 * plane;       // all five at one rung

    if (G.n_zp == 1) {
        for (int s = 0; s < 5; ++s) {
            const cd *V = G.vals + (std::ptrdiff_t)s * plane;
            cd acc(0.0, 0.0);
            for (int i = 0; i < 4; ++i) {
                const cd *row = V + (std::ptrdiff_t)(i0 + i) * nth + j0;
                const cd rs = row[0] * wt[0] + row[1] * wt[1] + row[2] * wt[2] +
                              row[3] * wt[3];
                acc += rs * wr[i];
            }
            surf[s] = acc;
        }
        return;
    }

    const double fz = (std::log(std::max(azp, 1e-300)) - G.lnz0) / G.dlnz;
    int k0 = (int)std::floor(fz) - 1;
    if (k0 < 0) k0 = 0;
    else if (k0 > (int)G.n_zp - 4) k0 = (int)G.n_zp - 4;
    double wz[4];
    somm_proj::lagrange4(fz - k0, wz);
    for (int s = 0; s < 5; ++s) {
        cd acc(0.0, 0.0);
        for (int k = 0; k < 4; ++k) {
            const cd *V = G.vals + (std::ptrdiff_t)(k0 + k) * rung +
                          (std::ptrdiff_t)s * plane;
            cd sub(0.0, 0.0);
            for (int i = 0; i < 4; ++i) {
                const cd *row = V + (std::ptrdiff_t)(i0 + i) * nth + j0;
                const cd rs = row[0] * wt[0] + row[1] * wt[1] + row[2] * wt[2] +
                              row[3] * wt[3];
                sub += rs * wr[i];
            }
            acc += sub * wz[k];
        }
        surf[s] = acc;
    }
}

// One (above, below) pair: `_crossing_geometry` + `TransmittedGrid.eval` +
// `divide_out_transmitted` + `_combine_transmitted_proj`, with no
// intermediates.
//
// The geometry is ALWAYS built from the above point and the below point, in
// that role order, whichever way the field is travelling: R and theta are the
// ABOVE point's polar coordinates about the below point's ground projection,
// |z'| is the below point's depth, and (dhx, dhy) is the horizontal direction
// from the BELOW point to the ABOVE one in BOTH directions of travel. That
// last one is the single silent sign error available in this family -- the
// surfaces are tabulated with the below point as the source and sin(phi) is
// ODD, so reading d-hat from source-minus-observer would flip T_phi^H on one
// of the two directions only.
//
// `transposed` swaps T_rho^V with T_z^H and NOTHING else. That is the entire
// content of "above->below is the reciprocity transpose": the dyad's 2x2
// horizontal block is already symmetric, so transposing exchanges exactly the
// vertical source's radial row with the horizontal source's vertical row --
// which is the FIFTH surface, and why a four-surface family could not have
// served both directions. E_z's horizontal row therefore reads T_z^H (or
// T_rho^V when transposed) and never -cos(phi) T_rho^V; that collapse is a
// +-=+ identity and it does not hold here.
//
// `to`/`ts` are the OBSERVER's and SOURCE's tangents, so the caller supplies
// them in travel order while the geometry stays in role order. The query's
// (R, theta, |z'|) go back to the caller so `TransmittedGrid.eval` can raise
// the four refusals in its own words; nothing here transcribes that prose.
static inline cd proj_one_transmitted(const TGridView &G, double ground_z,
                                      double k_p, const cd &k_m,
                                      double ax, double ay, double az,
                                      double bx, double by, double bz,
                                      const double *to, const double *ts,
                                      bool transposed, double &r_out,
                                      double &th_out, double &zp_out) {
    const double z = az - ground_z;         // the above point's height, >= 0
    const double depth = ground_z - bz;     // the below point's depth, > 0
    const double dx = ax - bx;
    const double dy = ay - by;
    const double rho = std::hypot(dx, dy);
    const double r_obs = std::hypot(rho, z);
    const double theta = std::atan2(z, rho);
    r_out = r_obs;
    th_out = theta;
    zp_out = depth;

    cd surf[5];
    tgrid_eval(G, r_obs, theta, depth, surf);
    const cd TrhoV = surf[0], TzV = surf[1], TrhoH = surf[2], TphiH = surf[3],
             TzH = surf[4];

    // `divide_out_transmitted`: two legs, because the transmitted ray has two
    // -- |z'| straight up through the ground at k_m, then the rest of the way
    // through the AIR at the real k_p over the true separation R, with the 1/R
    // spherical spreading every point-source field carries.
    const double dz = z + depth;
    const double rr = std::sqrt(rho * rho + dz * dz);
    const cd g = std::exp(-MW_BJ_T * (k_m * depth + cd(k_p * rr, 0.0))) / rr;

    const bool safe_r = rho > G.tiny;
    const double inv_rho = safe_r ? 1.0 / rho : 0.0;
    const double dhx = safe_r ? dx * inv_rho : 1.0;
    const double dhy = safe_r ? dy * inv_rho : 0.0;
    const double cph = ts[0] * dhx + ts[1] * dhy;
    const double sph = ts[0] * dhy - ts[1] * dhx;
    const double pz = ts[2];

    cd e_rho, e_z;
    if (transposed) {
        e_rho = g * (pz * TzH + cph * TrhoH);
        e_z = g * (pz * TzV + cph * TrhoV);
    } else {
        e_rho = g * (pz * TrhoV + cph * TrhoH);
        e_z = g * (pz * TzV + cph * TzH);
    }
    const cd e_phi = g * (sph * TphiH);
    return to[0] * (dhx * e_rho - dhy * e_phi) +
           to[1] * (dhy * e_rho + dhx * e_phi) + to[2] * e_z;
}
}  // namespace mw568_trans
