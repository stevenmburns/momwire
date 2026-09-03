// Shared projection helpers for the split accelerator TUs (momwire#687).
//
// This namespace was inside the _accelerators.cpp monolith, where the
// transmitted/below fills in `mw568*` could reach it for free. Those now
// live in their own translation unit, so the helpers hoist here -- the
// ONE real cross-section dependency the split turned up (the issue
// predicted the sections were independent; the compiler found otherwise).
//
// Everything stays `static`/`static inline`, exactly as in the monolith:
// each including TU gets its own internal-linkage copy, so section-local
// inlining is preserved and there is no ODR surface.
#pragma once

#include "_accel_common.h"

namespace somm_proj {
using cd = std::complex<double>;

// Cubic Lagrange weights for nodes at 0,1,2,3 evaluated at u (== _lagrange4).
static inline void lagrange4(double u, double *w) {
    const double u0 = u, u1 = u - 1.0, u2 = u - 2.0, u3 = u - 3.0;
    w[0] = -u1 * u2 * u3 / 6.0;
    w[1] = u0 * u2 * u3 / 2.0;
    w[2] = -u0 * u1 * u3 / 2.0;
    w[3] = u0 * u1 * u2 / 6.0;
}

// The tabulated SommerfeldGrid, flattened for the inner loop: raw pointers to
// the three near (plus optionally two far, issue #159) regions'
// (4, n_r, n_th) C-contiguous value tables plus their axis origins/spacings,
// and the region-select breakpoints. Populated from the pybind arrays by both
// callers (build_grid_view).
// Sized 9 for the below/below family's momwire#838 layout (3 R1 zones x 3
// theta bands). The +-=+ family still uses 4 or 6 of these; the extra slots
// cost a few hundred bytes on a struct built once per batch.
struct GridView {
    const cd *vptr[9];
    py::ssize_t nR[9], nTh[9];
    double rr0[9], rdr[9], rth0[9], rdth[9];
    double r1_max, r_break, th_split, r_near, tiny, half_pi;
};

static GridView build_grid_view(
    double r1_max, double r_break, double th_split, double r_near,
    const py::detail::unchecked_reference<double, 1> &r0b,
    const py::detail::unchecked_reference<double, 1> &drb,
    const py::detail::unchecked_reference<double, 1> &th0b,
    const py::detail::unchecked_reference<double, 1> &dthb,
    const std::vector<py::array_t<cd, py::array::c_style | py::array::forcecast>>
        &reg_vals) {
    // 4/6 since the momwire#443 inner-zone theta split, and 9 since
    // momwire#838 gave the below/below family three theta bands across three
    // R1 zones; any other count means a stale momwire/_sommerfeld*.py — fail
    // loudly either way.
    const size_t n_reg = reg_vals.size();
    if (n_reg != 4 && n_reg != 6 && n_reg != 9)
        throw std::runtime_error("expected 4, 6 or 9 region value tables");
    GridView G;
    for (size_t g = 0; g < n_reg; ++g) {
        auto v = reg_vals[g].template unchecked<3>();
        if (v.shape(0) != 4)
            throw std::runtime_error("region values must have shape (4, n_r, n_th)");
        G.vptr[g] = reg_vals[g].data();
        G.nR[g] = v.shape(1);
        G.nTh[g] = v.shape(2);
        G.rr0[g] = r0b(g);
        G.rdr[g] = drb(g);
        G.rth0[g] = th0b(g);
        G.rdth[g] = dthb(g);
    }
    G.r1_max = r1_max;
    G.r_break = r_break;
    G.th_split = th_split;
    // 4-region grids have r_near == r1_max, so clamped queries never route
    // far; guard anyway so a stale r_near can't index missing tables. The
    // 9-region below layout carries a real far zone and needs the true value.
    G.r_near = (n_reg == 6 || n_reg == 9) ? r_near : r1_max;
    G.tiny = 1e-12 * r1_max;
    G.half_pi = 0.5 * M_PI;
    return G;
}

// Interpolated + projected smooth-remainder field for ONE (observer, source)
// pair: t_obs . F(r_obs, r_src) . t_src. `sux/suy/sthsrc/stzsrc` are the
// source tangent's horizontal-unit / horizontal-magnitude / vertical parts
// (precomputed once per source point by the caller). Inlines
// SommerfeldGrid.eval + remainder_field_proj's eqs 143-147 with no
// intermediates. Bit-for-bit the numpy body.
static inline cd proj_one(
    const GridView &G, double ground_z, double k,
    double ox, double oy, double oz, double tox, double toy, double toz,
    double sx, double sy, double sz, double sux, double suy, double sthsrc,
    double stzsrc) {
    const double dx = ox - sx;
    const double dy = oy - sy;
    const double rho = std::hypot(dx, dy);
    const double hh = (oz - ground_z) + (sz - ground_z);
    const double r1 = std::sqrt(rho * rho + hh * hh);

    // --- inline SommerfeldGrid.eval(r1, theta) ---
    double theta = std::atan2(hh, rho);
    if (theta < 0.0) theta = 0.0; else if (theta > G.half_pi) theta = G.half_pi;
    const double r1c = r1 > G.r1_max ? G.r1_max : r1;  // interp clamps; g uses r1
    const int reg = (r1c <= G.r_break)
                        ? (theta <= G.th_split ? 0 : 1)
                        : (r1c <= G.r_near ? (theta <= G.th_split ? 2 : 3)
                                           : (theta <= G.th_split ? 4 : 5));
    const double fr = (r1c - G.rr0[reg]) / G.rdr[reg];
    const double ft = (theta - G.rth0[reg]) / G.rdth[reg];
    int i0 = (int)std::floor(fr) - 1;
    int j0 = (int)std::floor(ft) - 1;
    if (i0 < 0) i0 = 0; else if (i0 > G.nR[reg] - 4) i0 = (int)G.nR[reg] - 4;
    if (j0 < 0) j0 = 0; else if (j0 > G.nTh[reg] - 4) j0 = (int)G.nTh[reg] - 4;
    double wr[4], wt[4];
    lagrange4(fr - i0, wr);
    lagrange4(ft - j0, wt);
    const cd *V = G.vptr[reg];
    const py::ssize_t nth = G.nTh[reg], nr = G.nR[reg];
    cd surf[4];
    for (int s = 0; s < 4; ++s) {
        const cd *plane = V + (py::ssize_t)s * nr * nth;
        cd acc(0.0, 0.0);
        for (int i = 0; i < 4; ++i) {
            const cd *row = plane + (py::ssize_t)(i0 + i) * nth + j0;
            cd rs = row[0] * wt[0] + row[1] * wt[1] + row[2] * wt[2] +
                    row[3] * wt[3];
            acc += rs * wr[i];
        }
        surf[s] = acc;
    }
    const cd IrhoV = surf[0], IzV = surf[1], IrhoH = surf[2], IphiH = surf[3];

    // --- projection (eqs 143-147) ---
    const cd g = std::polar(1.0 / r1, -k * r1);
    const bool safe_r = rho > G.tiny;
    const double inv_rho = safe_r ? 1.0 / rho : 0.0;
    const double dhx = safe_r ? dx * inv_rho : sux;
    const double dhy = safe_r ? dy * inv_rho : suy;
    const double cphi = sux * dhx + suy * dhy;
    const double sphi = sux * dhy - suy * dhx;
    const cd e_rho = g * (stzsrc * IrhoV + sthsrc * cphi * IrhoH);
    const cd e_phi = g * (sthsrc * sphi * IphiH);
    const cd e_z = g * (stzsrc * IzV - sthsrc * cphi * IrhoV);
    return tox * (dhx * e_rho - dhy * e_phi) +
           toy * (dhy * e_rho + dhx * e_phi) + toz * e_z;
}

// Decompose a tangent (tx,ty,tz) into (horizontal unit x,y; horizontal
// magnitude; vertical), matching remainder_field_proj's th_src/ux/uy/tz_src.
static inline void tangent_decomp(double tx, double ty, double tz, double &ux,
                                  double &uy, double &th, double &tzc) {
    th = std::hypot(tx, ty);
    const bool safe = th > 1e-12;
    ux = safe ? tx / th : 1.0;
    uy = safe ? ty / th : 0.0;
    tzc = tz;
}
}  // namespace somm_proj
