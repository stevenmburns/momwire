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
// Sized `MAX_REGIONS` for the below/below family's momwire#1064 layout (3 R1
// zones x 5 theta bands). The +-=+ family still uses 4 or 6 of these; the
// extra slots cost a few hundred bytes on a struct built once per batch.
//
// The capacity is a NAMED constant and `build_grid_view` bounds-checks
// against it, because the failure mode when it is too small is silent: the
// fill loop runs to `reg_vals.size()` and simply writes past every array.
// #935 took this from 9 to 12 and #1064 from 12 to 15; the count check below
// is what makes a future band addition say so instead of corrupting the heap.
static constexpr size_t MAX_REGIONS = 15;

// The large-R1 continuation past the table's edge (momwire#1258): the
// constants `_sommerfeld.far_cpp_pack` computes once per grid, in its order.
// `on` false (an empty pack: the below family, or free space) serves the edge
// value frozen, which is what every grid did before #1258.
struct FarPack {
    bool on = false;
    cd eps, k1, c2, c1k, ap, res_fv, g2k1;
    double k2 = 0.0;
    // Derived once in set_far: sin/cos(ap), sqrt(cos(ap)), sin/cos(ap/2).
    cd sap, cap, sqcap, s_hap, c_hap;
};

struct GridView {
    const cd *vptr[MAX_REGIONS];
    py::ssize_t nR[MAX_REGIONS], nTh[MAX_REGIONS];
    double rr0[MAX_REGIONS], rdr[MAX_REGIONS], rth0[MAX_REGIONS],
        rdth[MAX_REGIONS];
    double r1_max, r_break, th_split, r_near, tiny, half_pi;
    FarPack far;
};

static GridView build_grid_view(
    double r1_max, double r_break, double th_split, double r_near,
    const py::detail::unchecked_reference<double, 1> &r0b,
    const py::detail::unchecked_reference<double, 1> &drb,
    const py::detail::unchecked_reference<double, 1> &th0b,
    const py::detail::unchecked_reference<double, 1> &dthb,
    const std::vector<py::array_t<cd, py::array::c_style | py::array::forcecast>>
        &reg_vals) {
    // 4/6 since the momwire#443 inner-zone theta split; 9 since momwire#838
    // gave the below/below family three theta bands across three R1 zones;
    // 12 since momwire#935 made that four bands, and 15 since momwire#1064
    // made it five. Any other count means a stale momwire/_sommerfeld*.py —
    // fail loudly either way.
    //
    // This is the SHARED builder, so it accepts every family's count and the
    // per-family kernel is what pins one. `proj_one_below` checks 15 exactly;
    // #935 found this second gate only by tripping it, which is the check
    // working — a below grid that grew a band reached here first.
    const size_t n_reg = reg_vals.size();
    if (n_reg != 4 && n_reg != 6 && n_reg != 9 && n_reg != 12 && n_reg != 15)
        throw std::runtime_error("expected 4, 6, 9, 12 or 15 region value tables");
    static_assert(MAX_REGIONS >= 15, "GridView too small for the #1064 layout");
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
    // 9-, 12- and 15-region below layouts carry a real far zone and need the
    // true value. Omitting 15 here does NOT fail loudly — it collapses the far
    // zone into the near one and every far query silently reads the wrong
    // table, so this list must grow with every layout added above.
    G.r_near = (n_reg == 6 || n_reg == 9 || n_reg == 12 || n_reg == 15)
                   ? r_near
                   : r1_max;
    G.tiny = 1e-12 * r1_max;
    G.half_pi = 0.5 * M_PI;
    return G;
}

// Load `_sommerfeld.far_cpp_pack(grid)` into G. Eight complex entries in the
// order below, or none (the continuation off). Any other length is a stale
// `_sommerfeld.py` and fails loudly.
static inline void set_far(
    GridView &G,
    const py::array_t<cd, py::array::c_style | py::array::forcecast> &far) {
    const py::ssize_t n = far.size();
    if (n == 0) {
        G.far.on = false;
        return;
    }
    if (n != 8)
        throw std::runtime_error(
            "far pack must hold 8 complex constants (momwire#1258) or none");
    const cd *f = far.data();
    G.far.on = true;
    G.far.eps = f[0];
    G.far.k1 = f[1];
    G.far.c2 = f[2];
    G.far.c1k = f[3];
    G.far.ap = f[4];
    G.far.res_fv = f[5];
    G.far.g2k1 = f[6];
    G.far.k2 = f[7].real();
    G.far.sap = std::sin(G.far.ap);
    G.far.cap = std::cos(G.far.ap);
    G.far.sqcap = std::sqrt(G.far.cap);
    G.far.s_hap = std::sin(0.5 * G.far.ap);
    G.far.c_hap = std::cos(0.5 * G.far.ap);
}

// Faddeeva w(z) = e^{-z^2} erfc(-jz). Weideman's rational approximation
// (SIAM J. Numer. Anal. 31, 1497, 1994) with N = 40 terms on the closed upper
// half-plane -- measured against scipy.special.wofz at 1.7e-15 absolute over
// |Re z|, Im z <= 12 -- and w(z) = 2e^{-z^2} - w(-z) below it. The coefficients
// are the paper's FFT, done here as a direct DFT once.
struct WeidemanCoef {
    static constexpr int N = 40;
    double a[N];
    double L;
    WeidemanCoef() {
        const int M = 2 * N, M2 = 2 * M;
        L = std::sqrt(N / std::sqrt(2.0));
        // f over k = -M+1 .. M-1, with a leading 0: length M2; fftshift
        // rotates it by M.
        std::vector<double> f(M2, 0.0), sh(M2);
        for (int k = -M + 1; k <= M - 1; ++k) {
            const double t = L * std::tan(0.5 * k * M_PI / M);
            f[k + M] = std::exp(-t * t) * (L * L + t * t);
        }
        for (int j = 0; j < M2; ++j) sh[j] = f[(j + M) % M2];
        for (int n = 1; n <= N; ++n) {
            double re = 0.0;
            for (int j = 0; j < M2; ++j)
                re += sh[j] * std::cos(2.0 * M_PI * (double)j * n / M2);
            a[n - 1] = re / M2;  // p(Z) = sum_n a[n-1] Z^(n-1)
        }
    }
};

static inline cd faddeeva_upper(cd z) {
    static const WeidemanCoef C;
    const cd iz(-z.imag(), z.real());
    const cd lm = C.L - iz;
    const cd Z = (C.L + iz) / lm;
    cd p(0.0, 0.0);
    for (int n = WeidemanCoef::N - 1; n >= 0; --n) p = p * Z + C.a[n];
    return 2.0 * p / (lm * lm) + (1.0 / std::sqrt(M_PI)) / lm;
}

static inline cd faddeeva(cd z) {
    if (z.imag() >= 0.0) return faddeeva_upper(z);
    return 2.0 * std::exp(-z * z) - faddeeva_upper(-z);
}

// 1/s + j sqrt(pi) w(s): `_sommerfeld._pole_bracket`, including its switch to
// the asymptotic series past |s| = 10 (`_FAR_BRACKET_SERIES`).
static inline cd pole_bracket(cd s) {
    const cd j(0.0, 1.0);
    if (std::abs(s) <= 10.0) return 1.0 / s + j * std::sqrt(M_PI) * faddeeva(s);
    const cd inv2 = 1.0 / (s * s);
    cd term = 0.5 * inv2 / s;
    cd acc = term;
    for (int n = 2; n < 10; ++n) {
        term = term * (double)(2 * n - 1) * 0.5 * inv2;
        acc = acc + term;
    }
    cd corr(0.0, 0.0);
    if (s.imag() < 0.0) corr = 2.0 * j * std::sqrt(M_PI) * std::exp(-s * s);
    return -acc + corr;
}

// `_sommerfeld.far_surfaces` in two halves, so the continuation evaluates the
// theta-only half once per pair and the range-dependent half at R1 and at
// the edge. Mirrors the numpy body term for term; the block comment above
// `far_surfaces` is the derivation.
struct FarTheta {
    double c, s, wt;
    cd fv, n_rz, n_zzv, n_rr, n_u;  // the saddle, bar its 1/R1 term
    cd pole;  // wt * res_fv * sqrt(cos(ap)/c)
    cd shalf;  // sin((ap - theta)/2)
};

static inline FarTheta far_theta(const FarPack &F, double th) {
    const cd j(0.0, 1.0);
    const double k = F.k2, ks = k * k;
    const cd k1s = F.k1 * F.k1;
    FarTheta T;
    T.c = std::cos(th);
    T.s = std::sin(th);
    const double c = T.c, s = T.s;
    const cd g2 = j * k * s;
    const cd g1 = std::sqrt(cd(ks * c * c, 0.0) - k1s);
    const cd rtm = (k1s * g2 - ks * g1) / (k1s * g2 + ks * g1);
    const cd rte = (g2 - g1) / (g2 + g1);
    T.fv = (rtm - F.c2) / k1s;
    T.n_rz = -ks * c * s * T.fv;
    T.n_zzv = ks * c * c * T.fv;
    T.n_rr = -ks * c * c * T.fv;
    T.n_u = rte + F.c2;
    // `_far_taper`: 1 below 50 deg, a raised cosine to 0 at 70.
    const double lo = 50.0 * M_PI / 180.0, hi = 70.0 * M_PI / 180.0;
    double x = (th - lo) / (hi - lo);
    x = x < 0.0 ? 0.0 : (x > 1.0 ? 1.0 : x);
    T.wt = 0.5 * (1.0 + std::cos(M_PI * x));
    if (T.wt > 0.0) {
        // sqrt(cos(ap)/c) and sin((ap - th)/2) from per-grid constants and
        // real functions of th.
        T.pole = T.wt * F.res_fv * F.sqcap / std::sqrt(c);
        const double ch = std::cos(0.5 * th), sh = std::sin(0.5 * th);
        T.shalf = F.s_hap * ch - F.c_hap * sh;
    }
    return T;
}

static inline void far_at(const FarPack &F, const FarTheta &T, double r1,
                          cd out[4]) {
    const cd j(0.0, 1.0);
    const double k = F.k2, ks = k * k;
    const cd k1 = F.k1, k1s = k1 * k1;
    const double c = T.c, s = T.s;
    cd n_rz = T.n_rz, n_zzv = T.n_zzv, n_rr = T.n_rr, n_u = T.n_u;
    cd n_r1 = -j * k / r1 * T.fv;
    if (T.wt > 0.0) {
        const double om = k * r1;
        const cd sap = F.sap, cap = F.cap;
        const cd em(std::sqrt(0.5), -std::sqrt(0.5));  // e^{-j pi/4}
        const double r2om = std::sqrt(2.0 * om);
        const cd sp = r2om * em * T.shalf;
        const cd inv_a0 = (0.5 * r2om) * em;  // sqrt(om/2) e^{-j pi/4}
        const cd amp = T.pole * inv_a0 * pole_bracket(sp);
        n_rz += -ks * cap * sap * amp;
        n_zzv += ks * cap * cap * amp;
        n_rr += -ks * cap * cap * amp;
        n_r1 += -j * k * cap / (r1 * c) * amp;

        const cd g2k1 = F.g2k1;
        const double rho = r1 * c, h = r1 * s;
        if (g2k1.real() * h - k1.imag() * rho < 50.0 && rho > 0.0) {
            const cd rp = rho - j * h * k1 / g2k1;
            const cd base = -j * T.wt *
                            std::exp(-j * k1 * rho - g2k1 * h + j * k * r1) *
                            r1 / (std::sqrt(rho) * (rp * std::sqrt(rp)));
            const cd dd2 = -2.0 * ks / ((k1s * g2k1) * (k1s * g2k1));
            const cd dd1 = -2.0 / (g2k1 * g2k1);
            n_rr += dd2 * (-k1 * k1s) * base;
            n_rz += dd2 * (j * g2k1 * k1s) * base;
            n_zzv += dd2 * (g2k1 * g2k1 * k1 + ks * k1) * base;
            n_r1 += dd2 * (-j * k1s / rho) * base;
            n_u += dd1 * k1 * base;
        }
    }
    const cd c1 = F.c1k / ks;
    out[0] = c1 * k1s * n_rz;
    out[1] = c1 * k1s * n_zzv;
    out[2] = c1 * ks * (n_rr + n_u);
    out[3] = -c1 * ks * (n_r1 + n_u);
}

// `SommerfeldGrid._continue_past_edge` for one query: `surf` holds the
// table's value at the edge on entry and the continued value on return.
static void far_continue(const GridView &G, double r1, double theta, cd surf[4]) {
    const FarTheta T = far_theta(G.far, theta);
    cd a[4], e[4];
    far_at(G.far, T, r1, a);
    far_at(G.far, T, G.r1_max, e);
    const double w = G.r1_max / r1;
    for (int q = 0; q < 4; ++q) surf[q] = a[q] + (surf[q] - e[q]) * w;
}

// The pair-symmetric half of `proj_one`: everything that depends on the pair
// only through rho = |horizontal separation| and hh = (z_obs - ground) +
// (z_src - ground) -- the grid interpolation of the four surfaces, the
// momwire#1258 continuation and the image-distance phase factor g. Both
// arguments are bit-identical under swapping observer and source (hypot reads
// |dx|, |dy|, and dx, dy merely change sign; hh is one commutative add), so
// one call serves the (m, n) AND (n, m) projections of a node pair
// (`sommerfeld_remainder_bspline_Q`'s symmetric route).
static inline void proj_core(const GridView &G, double k, double rho, double hh,
                             cd surf[4], cd &g) {
    const double r1 = std::sqrt(rho * rho + hh * hh);

    // --- inline SommerfeldGrid.eval(r1, theta) ---
    double theta = std::atan2(hh, rho);
    if (theta < 0.0) theta = 0.0; else if (theta > G.half_pi) theta = G.half_pi;
    const double r1c = r1 > G.r1_max ? G.r1_max : r1;  // the edge; see below
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
    // Past the edge (momwire#1258): continue from the edge value rather than
    // serving it frozen. `g` below keeps the true distance either way.
    if (G.far.on && r1 > G.r1_max) far_continue(G, r1, theta, surf);
    g = std::polar(1.0 / r1, -k * r1);
}

// The orientation-dependent half of `proj_one`: the eqs 143-147 azimuth
// factors and the projection on the observer tangent, from `proj_core`'s
// surfaces and g. (dx, dy) = observer - source, horizontally; `rho` is
// hypot(dx, dy) as `proj_core` received it.
static inline cd proj_project(const GridView &G, double dx, double dy,
                              double rho, const cd surf[4], cd g, double tox,
                              double toy, double toz, double sux, double suy,
                              double sthsrc, double stzsrc) {
    const cd IrhoV = surf[0], IzV = surf[1], IrhoH = surf[2], IphiH = surf[3];

    // --- projection (eqs 143-147) ---
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

// Interpolated + projected smooth-remainder field for ONE (observer, source)
// pair: t_obs . F(r_obs, r_src) . t_src. `sux/suy/sthsrc/stzsrc` are the
// source tangent's horizontal-unit / horizontal-magnitude / vertical parts
// (precomputed once per source point by the caller). Inlines
// SommerfeldGrid.eval + remainder_field_proj's eqs 143-147 with no
// intermediates. Bit-for-bit the numpy body.
//
// Split into `proj_core` + `proj_project` with every expression kept as
// written; the builds pass -ffp-contract=off (momwire#1194), so the split
// moves no bits.
static inline cd proj_one(
    const GridView &G, double ground_z, double k,
    double ox, double oy, double oz, double tox, double toy, double toz,
    double sx, double sy, double sz, double sux, double suy, double sthsrc,
    double stzsrc) {
    const double dx = ox - sx;
    const double dy = oy - sy;
    const double rho = std::hypot(dx, dy);
    const double hh = (oz - ground_z) + (sz - ground_z);
    cd surf[4], g;
    proj_core(G, k, rho, hh, surf, g);
    return proj_project(G, dx, dy, rho, surf, g, tox, toy, toz, sux, suy,
                        sthsrc, stzsrc);
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
