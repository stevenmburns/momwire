#include "_accel_common.h"
#include "_branch_cut_inline.h"
#include "_cdiv_inline.h"
#include "_fma_inline.h"
#include "_somm_six_lanes_inline.h"

#include <cstring>
#ifdef _OPENMP
#include <omp.h>
#endif

// somm section of the former _accelerators.cpp monolith (momwire#687).
// Code below is byte-identical to the monolith's lines 5402-6272, with TWO
// exceptions: somm_proj (monolith 5846-5982) was hoisted verbatim into
// _accel_somm_proj_inline.h — shared with the mw568 TU — and is pulled back
// in by the mid-file #include below; and the local `gam` branch-cut helper
// became the shared `mw_branch::gamma_cut` (#714), renaming its two call
// sites in integrand_six.

#include "_somm_kernel_core.h"

// Batched entry point: the six NEC lambda-integrals at each (rho[i], h[i]),
// OpenMP across nodes (each node's adaptive quadrature is independent).
// Returns (n, 6) complex in _six_integrals order (Vrr, Vzz, Vrz, Vr1, V, U).
static py::array_t<std::complex<double>> somm_six_integrals_batch(
    std::complex<double> eps_t, double k2,
    py::array_t<double, py::array::c_style | py::array::forcecast> rho,
    py::array_t<double, py::array::c_style | py::array::forcecast> h,
    double rtol, int form, uintptr_t cancel_flag = 0, bool lanes = true) {
    auto rb = rho.unchecked<1>();
    auto hb = h.unchecked<1>();
    const py::ssize_t n = rb.shape(0);
    if (hb.shape(0) != n)
        throw std::invalid_argument("rho and h must have the same length");
    if (form < 0 || form > 2)
        throw std::invalid_argument("form must be 0 (auto), 1 (J) or 2 (H)");
    for (py::ssize_t i = 0; i < n; ++i) {
        if (rb(i) < 0.0 || hb(i) < 0.0 || (rb(i) == 0.0 && hb(i) == 0.0))
            throw std::invalid_argument(
                "need rho, h >= 0 and R1 > 0 at every node");
    }
    py::array_t<std::complex<double>> out({n, py::ssize_t(6)});
    auto ob = out.mutable_unchecked<2>();
    const std::complex<double> et(eps_t);
    MW_CANCEL_SETUP(cancel_flag);

    #pragma omp parallel for schedule(dynamic)
    for (py::ssize_t i = 0; i < n; ++i) {
        MW_CANCEL_POLL();
        std::complex<double> res[6];
        somm_core::six_integrals(et, k2, rb(i), hb(i), rtol, form, res, lanes);
        for (int j = 0; j < 6; ++j) ob(i, j) = res[j];
    }
    MW_THROW_IF_ABORTED();
    return out;
}

// ---------------------------------------------------------------------------
// Sommerfeld remainder assembly (sommerfeld-perf-plan Phase 4b).
//
// Fused C++ port of _sommerfeld.remainder_field_proj (which internally calls
// SommerfeldGrid.eval). Per (observer m, source n) pair: interpolate the four
// smooth-remainder surfaces from the tabulated grid with the same 4x4 Lagrange
// (bivariate cubic) stencil, combine per the theory-manual eqs 143-147 azimuth
// factors, and project onto the observer tangent -- with NO materialized
// (4,n,4,4) intermediate (the numpy bottleneck). One OpenMP loop over observer
// rows. Bit-for-bit the same arithmetic as the Python fallback; cross-checked
// in tests/test_sommerfeld_accel.py.
//
// Clean-room: ported from momwire's own _sommerfeld.py. No GPL Sommerfeld
// source (nec2c, nec2++/PyNEC, somnec) was consulted.
#include "_accel_somm_proj_inline.h"
// The near-field remainder's kernel (`reflected_field_batch` below binds it).
#include "_somm_field_core.h"

// ---------------------------------------------------------------------------
// The remainder projection's blocked loop and its lanes (perf item 9; on the
// lane layer, `_lanes.h`, since momwire#1372).
//
// `proj_one` per pair is ~10 divisions (the stencil's two and lagrange4's
// eight), sixty-four complex-by-real products in the surface read, the
// projection's complex products, and three libm calls (hypot, atan2, the
// polar's sincos). SG above x16 spent 7.2 s of 21 s here on Haswell. The
// blocked loop takes a row's sources in blocks of `ABOVE_BLOCK` and runs
// `proj_one`'s stages over a block: the geometry and the libm calls per pair,
// scalar; the stencil W pairs to a vector; the surface read W/2 surfaces to
// a vector (re, im of each in a lane pair); the projection W pairs to a
// vector; and the momwire#1258 continuation, where a pair needs it, per pair
// on the lanes' surfaces. Every lane does the scalar stage's operations in its
// order — the same selects, correctly rounded divisions, and (the build being
// -ffp-contract=off, and these helpers using std::complex, not the `mw_fma`
// ones) separate multiplies and adds, never a fused one — and no reduction
// crosses lanes, so a lane's value is `proj_one`'s. A block's last nb mod W
// pairs take `proj_one` itself. `lanes=false` is the per-pair loop, the
// reference the lanes are gated against (tests/test_somm_proj_lanes_1290.py);
// the baseline and arm64 builds have only it. On MSVC (/fp:fast) the gate is
// the derived win32 tolerance, not bit equality (momwire#1371).
#if MW_LANES_SIMD && MW_LANES_PACKED
#define MW_SOMM_LANES 1
#else
#define MW_SOMM_LANES 0
#endif

namespace somm_lanes {
using somm_proj::cd;
constexpr int ABOVE_BLOCK = 64;

#if MW_SOMM_LANES
using namespace mw_lanes;

// `proj_core`'s stencil for pairs b .. b+W-1: theta and r1 clamped, the
// region, then the shared tail (the first stencil node clamped, lagrange4 at
// both offsets).
static inline void stencil_lanes(const somm_proj::GridView &G, const int *nR32,
                                 const int *nTh32, const double *r1,
                                 const double *th, int *reg_o, int *i0_o,
                                 int *j0_o, double (*wr)[ABOVE_BLOCK],
                                 double (*wt)[ABOVE_BLOCK], int b) {
    const vd t = loadu(th + b);
    vd rc = loadu(r1 + b);
    const vd c_r1max = set1(G.r1_max);
    // r1c = r1 > r1_max ? r1_max : r1
    rc = blend(rc, c_r1max, cmp_gt(rc, c_r1max));
    // reg = r1c <= r_break ? (t <= split ? 0 : 1)
    //     : (r1c <= r_near ? (t <= split ? 2 : 3) : (t <= split ? 4 : 5))
    const vm low = cmp_le(t, set1(G.th_split));
    const vd side = blend(set1(1.0), zero(), low);
    vd zone = blend(set1(4.0), set1(2.0), cmp_le(rc, set1(G.r_near)));
    zone = blend(zone, zero(), cmp_le(rc, set1(G.r_break)));
    somm_proj_lanes::stencil_tail<ABOVE_BLOCK>(G, nR32, nTh32, rc, t,
                                               cvtt_i32(add(zone, side)), reg_o,
                                               i0_o, j0_o, wr, wt, b);
}

// `proj_project` for pairs b .. b+W-1, std::complex's operators spelled per
// part (complex x complex as (ac - bd, ad + bc), complex x real
// componentwise); writes the W entries.
static inline void project_lanes(const somm_proj::GridView &G,
                                 const double (*sre)[ABOVE_BLOCK],
                                 const double (*sim)[ABOVE_BLOCK],
                                 const double *gre, const double *gim,
                                 const double *rho_, const double *dx_,
                                 const double *dy_, double tox, double toy,
                                 double toz, const double *ux, const double *uy,
                                 const double *thsrc, const double *tzsrc, int b,
                                 cd *out) {
    auto L = [&](const double *p) { return loadu(p + b); };
    const vd rho = L(rho_), dx = L(dx_), dy = L(dy_);
    const vd sux = loadu(ux), suy = loadu(uy);
    const vd sth = loadu(thsrc), stz = loadu(tzsrc);
    const vd gr = L(gre), gi = L(gim);
    const vm safe = cmp_gt(rho, set1(G.tiny));
    const vd inv = blend(zero(), div(set1(1.0), rho), safe);
    const vd dhx = blend(sux, mul(dx, inv), safe);
    const vd dhy = blend(suy, mul(dy, inv), safe);
    const vd cphi = add(mul(sux, dhx), mul(suy, dhy));
    const vd sphi = sub(mul(sux, dhy), mul(suy, dhx));
    const vd Vr = L(sre[0]), Vi = L(sim[0]), Zr = L(sre[1]), Zi = L(sim[1]);
    const vd Hr = L(sre[2]), Hi = L(sim[2]), Pr = L(sre[3]), Pi = L(sim[3]);
    auto cmr = [](vd ar, vd ai, vd br, vd bi) { return sub(mul(ar, br), mul(ai, bi)); };
    auto cmi = [](vd ar, vd ai, vd br, vd bi) { return add(mul(ar, bi), mul(ai, br)); };
    const vd sc = mul(sth, cphi);
    // e_rho = g * (stzsrc * IrhoV + (sthsrc * cphi) * IrhoH)
    const vd tr = add(mul(stz, Vr), mul(sc, Hr));
    const vd ti = add(mul(stz, Vi), mul(sc, Hi));
    const vd er = cmr(gr, gi, tr, ti), ei = cmi(gr, gi, tr, ti);
    // e_phi = g * ((sthsrc * sphi) * IphiH)
    const vd f = mul(sth, sphi);
    const vd ur = mul(f, Pr), ui = mul(f, Pi);
    const vd pr = cmr(gr, gi, ur, ui), pi = cmi(gr, gi, ur, ui);
    // e_z = g * (stzsrc * IzV - (sthsrc * cphi) * IrhoV)
    const vd cr = sub(mul(stz, Zr), mul(sc, Vr));
    const vd ci = sub(mul(stz, Zi), mul(sc, Vi));
    const vd zr = cmr(gr, gi, cr, ci), zi = cmi(gr, gi, cr, ci);
    // tox * (dhx * e_rho - dhy * e_phi) + toy * (dhy * e_rho + dhx * e_phi)
    //   + toz * e_z
    const vd vtox = set1(tox), vtoy = set1(toy), vtoz = set1(toz);
    const vd d1r = sub(mul(dhx, er), mul(dhy, pr));
    const vd d1i = sub(mul(dhx, ei), mul(dhy, pi));
    const vd d2r = add(mul(dhy, er), mul(dhx, pr));
    const vd d2i = add(mul(dhy, ei), mul(dhx, pi));
    const vd o_r = add(add(mul(vtox, d1r), mul(vtoy, d2r)), mul(vtoz, zr));
    const vd o_i = add(add(mul(vtox, d1i), mul(vtoy, d2i)), mul(vtoz, zi));
    store_interleaved(reinterpret_cast<double *>(out), o_r, o_i);
}
#endif  // MW_SOMM_LANES
}  // namespace somm_lanes

static std::atomic<unsigned long long> g_somm_proj_lane_pairs{0};

// obs/t_obs (M,3), src/t_src (S,3); returns (M,S) complex. The grid is passed
// flattened: the regions' (r0, dr, th0, dth) as per-region arrays, plus a
// list of four-or-six (4, n_r, n_th) complex value tables. r_break / th_split select
// the region exactly as SommerfeldGrid.eval.
static py::array_t<std::complex<double>> remainder_field_proj_batch(
    py::array_t<double, py::array::c_style | py::array::forcecast> obs,
    py::array_t<double, py::array::c_style | py::array::forcecast> t_obs,
    py::array_t<double, py::array::c_style | py::array::forcecast> src,
    py::array_t<double, py::array::c_style | py::array::forcecast> t_src,
    double ground_z, double k, double r1_max, double r_break, double th_split,
    double r_near,
    py::array_t<double, py::array::c_style | py::array::forcecast> reg_r0,
    py::array_t<double, py::array::c_style | py::array::forcecast> reg_dr,
    py::array_t<double, py::array::c_style | py::array::forcecast> reg_th0,
    py::array_t<double, py::array::c_style | py::array::forcecast> reg_dth,
    std::vector<py::array_t<std::complex<double>,
                            py::array::c_style | py::array::forcecast>> reg_vals,
    py::array_t<std::complex<double>, py::array::c_style | py::array::forcecast>
        far,
    uintptr_t cancel_flag = 0, bool lanes = true) {
    using somm_proj::cd;
    auto ob = obs.unchecked<2>();
    auto tob = t_obs.unchecked<2>();
    auto sb = src.unchecked<2>();
    auto tsb = t_src.unchecked<2>();
    if (ob.shape(1) != 3 || tob.shape(1) != 3 || sb.shape(1) != 3 ||
        tsb.shape(1) != 3)
        throw std::runtime_error("obs/src/tangent arrays must have shape (*, 3)");
    if (ob.shape(0) != tob.shape(0) || sb.shape(0) != tsb.shape(0))
        throw std::runtime_error("points and tangents must have matching length");

    const py::ssize_t M = ob.shape(0);
    const py::ssize_t S = sb.shape(0);
    somm_proj::GridView G = somm_proj::build_grid_view(
        r1_max, r_break, th_split, r_near, reg_r0.unchecked<1>(),
        reg_dr.unchecked<1>(), reg_th0.unchecked<1>(), reg_dth.unchecked<1>(),
        reg_vals);
    somm_proj::set_far(G, far);

    py::array_t<std::complex<double>> out({M, S});
    auto out_m = out.mutable_unchecked<2>();

    py::gil_scoped_release release;

    // Per-source constants (n-only): position + tangent decomposition.
    std::vector<double> sx(S), sy(S), sz(S), ux(S), uy(S), thsrc(S), tzsrc(S);
    for (py::ssize_t n = 0; n < S; ++n) {
        sx[n] = sb(n, 0);
        sy[n] = sb(n, 1);
        sz[n] = sb(n, 2);
        somm_proj::tangent_decomp(tsb(n, 0), tsb(n, 1), tsb(n, 2), ux[n], uy[n],
                                  thsrc[n], tzsrc[n]);
    }

#if MW_SOMM_LANES
    int nR32[somm_proj::MAX_REGIONS], nTh32[somm_proj::MAX_REGIONS];
    for (size_t g = 0; g < reg_vals.size(); ++g) {
        nR32[g] = static_cast<int>(G.nR[g]);
        nTh32[g] = static_cast<int>(G.nTh[g]);
    }
    const bool use_lanes = lanes;
    if (use_lanes) g_somm_proj_lane_pairs += (unsigned long long)(M * S);
#else
    (void)lanes;
#endif

    MW_CANCEL_SETUP(cancel_flag);
    // Rows dynamically: an entry is its own pair's arithmetic, so where a row
    // runs moves no bit, and a call's few dozen rows split unevenly over a
    // static schedule.
    #pragma omp parallel for schedule(dynamic)
    for (py::ssize_t m = 0; m < M; ++m) {
        MW_CANCEL_POLL();
        const double ox = ob(m, 0), oy = ob(m, 1), oz = ob(m, 2);
        const double tox = tob(m, 0), toy = tob(m, 1), toz = tob(m, 2);
#if MW_SOMM_LANES
        if (use_lanes) {
            using namespace somm_lanes;
            constexpr int B = ABOVE_BLOCK;
            double dx[B], dy[B], rho[B], r1[B], th[B], gre[B], gim[B];
            int l_reg[B], l_i0[B], l_j0[B];
            double l_wr[4][B], l_wt[4][B], l_sre[4][B], l_sim[4][B];
            cd *orow = &out_m(m, 0);
            for (py::ssize_t n0 = 0; n0 < S; n0 += B) {
                const int nb = (int)std::min<py::ssize_t>(B, S - n0);
                const int nq4 = nb - nb % W;
                // Geometry and the libm calls, per pair: `proj_one`'s
                // dx/dy/rho/hh and `proj_core`'s r1, theta (clamped) and g.
                for (int b = 0; b < nq4; ++b) {
                    const py::ssize_t n = n0 + b;
                    double hh;
                    somm_proj::pair_geometry(ground_z, ox, oy, oz, sx[n], sy[n], sz[n],
                                             dx[b], dy[b], rho[b], hh);
                    somm_proj::proj_r1_theta(G, rho[b], hh, r1[b], th[b]);
                    const cd g = std::polar(1.0 / r1[b], -k * r1[b]);
                    gre[b] = g.real();
                    gim[b] = g.imag();
                }
                for (int b = 0; b < nq4; b += W)
                    stencil_lanes(G, nR32, nTh32, r1, th, l_reg, l_i0, l_j0,
                                  l_wr, l_wt, b);
                for (int b = 0; b < nq4; ++b)
                    somm_proj_lanes::surfaces_lanes<B, false>(
                        G, l_reg[b], l_i0[b], l_j0[b], l_wr, l_wt, l_sre, l_sim, b);
                // Past the table's edge (momwire#1258): the continuation, per
                // pair, on the read surfaces, exactly as `proj_core` applies it.
                if (G.far.on) {
                    for (int b = 0; b < nq4; ++b) {
                        if (!(r1[b] > G.r1_max)) continue;
                        cd surf[4];
                        for (int q = 0; q < 4; ++q) surf[q] = cd(l_sre[q][b], l_sim[q][b]);
                        somm_proj::far_continue(G, r1[b], th[b], surf);
                        for (int q = 0; q < 4; ++q) {
                            l_sre[q][b] = surf[q].real();
                            l_sim[q][b] = surf[q].imag();
                        }
                    }
                }
                for (int b = 0; b < nq4; b += W)
                    project_lanes(G, l_sre, l_sim, gre, gim, rho, dx, dy, tox,
                                  toy, toz, &ux[n0 + b], &uy[n0 + b],
                                  &thsrc[n0 + b], &tzsrc[n0 + b], b,
                                  orow + n0 + b);
                for (int b = nq4; b < nb; ++b) {
                    const py::ssize_t n = n0 + b;
                    orow[n] = somm_proj::proj_one(
                        G, ground_z, k, ox, oy, oz, tox, toy, toz, sx[n], sy[n],
                        sz[n], ux[n], uy[n], thsrc[n], tzsrc[n]);
                }
            }
            continue;
        }
#endif
        for (py::ssize_t n = 0; n < S; ++n) {
            out_m(m, n) = somm_proj::proj_one(
                G, ground_z, k, ox, oy, oz, tox, toy, toz, sx[n], sy[n], sz[n],
                ux[n], uy[n], thsrc[n], tzsrc[n]);
        }
    }
    MW_THROW_IF_ABORTED();
    return out;
}

// The near-field remainder at observation points (`_somm_field_core.cpp`):
// E, or H by the curl when `step` > 0. points (N, 3), src_mid (S, 3),
// src_moment (S, 3) complex; returns (N, 3) complex. The grid is passed as
// `remainder_field_proj_batch` takes it.
static py::array_t<std::complex<double>> reflected_field_batch_impl(
    py::array_t<double, py::array::c_style | py::array::forcecast> points,
    py::array_t<double, py::array::c_style | py::array::forcecast> src_mid,
    py::array_t<std::complex<double>, py::array::c_style | py::array::forcecast> src_moment,
    double ground_z, double k, bool magnetic, double omega, double mu, double step,
    double r1_max, double r_break, double th_split, double r_near,
    py::array_t<double, py::array::c_style | py::array::forcecast> reg_r0,
    py::array_t<double, py::array::c_style | py::array::forcecast> reg_dr,
    py::array_t<double, py::array::c_style | py::array::forcecast> reg_th0,
    py::array_t<double, py::array::c_style | py::array::forcecast> reg_dth,
    std::vector<py::array_t<std::complex<double>,
                            py::array::c_style | py::array::forcecast>> reg_vals,
    py::array_t<std::complex<double>, py::array::c_style | py::array::forcecast> far) {
    if (points.ndim() != 2 || points.shape(1) != 3 || src_mid.ndim() != 2 ||
        src_mid.shape(1) != 3 || src_moment.ndim() != 2 || src_moment.shape(1) != 3 ||
        src_moment.shape(0) != src_mid.shape(0))
        throw std::runtime_error("points, src_mid and src_moment must be (*, 3), the sources alike");
    somm_proj::GridView G = somm_proj::build_grid_view(
        r1_max, r_break, th_split, r_near, reg_r0.unchecked<1>(), reg_dr.unchecked<1>(),
        reg_th0.unchecked<1>(), reg_dth.unchecked<1>(), reg_vals);
    somm_proj::set_far(G, far);
    const py::ssize_t n = points.shape(0), s = src_mid.shape(0);
    py::array_t<std::complex<double>> out({n, py::ssize_t(3)});
    {
        py::gil_scoped_release release;
        if (magnetic)
            somm_core::reflected_h_field(G, n, points.data(), s, src_mid.data(), src_moment.data(),
                                         ground_z, k, omega, mu, step, out.mutable_data());
        else
            somm_core::reflected_field(G, n, points.data(), s, src_mid.data(), src_moment.data(),
                                       ground_z, k, out.mutable_data());
    }
    return out;
}

// The lazy grids' census (momwire#1512): a mask per region (n_r, n_th) of
// the nodes the reads of `remainder_field_proj_batch` over obs x src take --
// both of its loops go through `pair_geometry`, `proj_r1_theta` and the
// address `proj_address` gives (the lanes' `stencil_lanes` is that address
// lane by lane) -- or, `reflected`, those of `reflected_field_batch` at
// `points` (E; with magnetic, H's six offset points), likewise.
static std::vector<py::array_t<uint8_t>> census_masks(
    const somm_proj::GridView &G, size_t n_reg, somm_proj::NodeMarks &marks) {
    std::vector<py::array_t<uint8_t>> masks;
    marks.n_reg = n_reg;
    for (size_t r = 0; r < n_reg; r++) {
        py::array_t<uint8_t> a({G.nR[r], G.nTh[r]});
        std::fill(a.mutable_data(), a.mutable_data() + G.nR[r] * G.nTh[r], uint8_t(0));
        marks.m[r] = a.mutable_data();
        marks.nR[r] = static_cast<std::ptrdiff_t>(G.nR[r]);
        marks.nTh[r] = static_cast<std::ptrdiff_t>(G.nTh[r]);
        masks.push_back(a);
    }
    return masks;
}

static std::vector<py::array_t<uint8_t>> remainder_field_proj_nodes(
    py::array_t<double, py::array::c_style | py::array::forcecast> obs,
    py::array_t<double, py::array::c_style | py::array::forcecast> src, double ground_z,
    double r1_max, double r_break, double th_split, double r_near,
    py::array_t<double, py::array::c_style | py::array::forcecast> reg_r0,
    py::array_t<double, py::array::c_style | py::array::forcecast> reg_dr,
    py::array_t<double, py::array::c_style | py::array::forcecast> reg_th0,
    py::array_t<double, py::array::c_style | py::array::forcecast> reg_dth,
    std::vector<py::array_t<std::complex<double>,
                            py::array::c_style | py::array::forcecast>> reg_vals) {
    if (obs.ndim() != 2 || obs.shape(1) != 3 || src.ndim() != 2 || src.shape(1) != 3)
        throw std::runtime_error("obs/src arrays must have shape (*, 3)");
    const somm_proj::GridView G = somm_proj::build_grid_view(
        r1_max, r_break, th_split, r_near, reg_r0.unchecked<1>(), reg_dr.unchecked<1>(),
        reg_th0.unchecked<1>(), reg_dth.unchecked<1>(), reg_vals);
    somm_proj::NodeMarks marks;
    std::vector<py::array_t<uint8_t>> masks = census_masks(G, reg_vals.size(), marks);
    const py::ssize_t M = obs.shape(0), S = src.shape(0);
    const double *o = obs.data(), *sp = src.data();
    py::gil_scoped_release release;
    for (py::ssize_t m = 0; m < M; ++m)
        for (py::ssize_t n = 0; n < S; ++n) {
            double dx, dy, rho, hh, r1, theta, fr, ft;
            int i0, j0;
            somm_proj::pair_geometry(ground_z, o[3 * m], o[3 * m + 1], o[3 * m + 2], sp[3 * n],
                                     sp[3 * n + 1], sp[3 * n + 2], dx, dy, rho, hh);
            somm_proj::proj_r1_theta(G, rho, hh, r1, theta);
            const int reg = somm_proj::proj_address(G, r1, theta, fr, ft, i0, j0);
            marks.mark(reg, i0, j0);
        }
    return masks;
}

static std::vector<py::array_t<uint8_t>> reflected_field_nodes_impl(
    py::array_t<double, py::array::c_style | py::array::forcecast> points,
    py::array_t<double, py::array::c_style | py::array::forcecast> src_mid, double ground_z,
    bool magnetic, double step, double r1_max, double r_break, double th_split, double r_near,
    py::array_t<double, py::array::c_style | py::array::forcecast> reg_r0,
    py::array_t<double, py::array::c_style | py::array::forcecast> reg_dr,
    py::array_t<double, py::array::c_style | py::array::forcecast> reg_th0,
    py::array_t<double, py::array::c_style | py::array::forcecast> reg_dth,
    std::vector<py::array_t<std::complex<double>,
                            py::array::c_style | py::array::forcecast>> reg_vals) {
    if (points.ndim() != 2 || points.shape(1) != 3 || src_mid.ndim() != 2 ||
        src_mid.shape(1) != 3)
        throw std::runtime_error("points and src_mid must be (*, 3)");
    const somm_proj::GridView G = somm_proj::build_grid_view(
        r1_max, r_break, th_split, r_near, reg_r0.unchecked<1>(), reg_dr.unchecked<1>(),
        reg_th0.unchecked<1>(), reg_dth.unchecked<1>(), reg_vals);
    somm_proj::NodeMarks marks;
    std::vector<py::array_t<uint8_t>> masks = census_masks(G, reg_vals.size(), marks);
    const py::ssize_t n = points.shape(0), s = src_mid.shape(0);
    py::gil_scoped_release release;
    if (magnetic)
        somm_core::reflected_h_field_nodes(G, n, points.data(), s, src_mid.data(), ground_z, step,
                                           marks);
    else
        somm_core::reflected_field_nodes(G, n, points.data(), s, src_mid.data(), ground_z, marks);
    return masks;
}

// Fully-fused b-spline Galerkin Sommerfeld remainder (Phase 4b, stage 2):
// returns the (n_basis, n_basis) block Q directly, absorbing the moment
// quadrature and the basis assembly that the Python code did with two einsums
// and a large fancy-index gather. Segments are the shared obs=src set.
//
//   Jf[p,P,i,j] = sum_{qi,rj} W[p,i,qi] * proj(node[i,qi], node[j,rj]) * W[P,j,rj]
//   Q[m,n]      = sum_{a,b,p,P} polys[m,a,p] * Jf[p,P, supp[m,a], supp[n,b]]
//                                            * polys[n,b,P]
//
// Rectangular obs/src form: the dense b-spline block is the symmetric case
// (obs == src, loc == supp_seg), the ACA sampler passes thin obs/src segment
// subsets with local support maps. obs_nodes (nsI, q, 3), obs_tang (nsI, 3),
// W_obs (d+1, nsI, q); likewise src_*; loc_I (nI, d+1) int64 indexes obs
// segments, pI (nI, d+1, d+1); loc_J/pJ index src. Grid passed as in
// remainder_field_proj_batch. Returns Q (nI, nJ).
//
// Observer banding (momwire#343). Jf over the FULL (nsI, nsJ) rectangle is
// 16*(d+1)^2*nsI*nsJ bytes -- 144 N^2 at d=2, ~9x the dense Z it contributes
// to, and ~10 GB at N=8320. The two stages are instead run band by band over
// the observer segments: stage 1 fills Jf for the band [i0, i1) only, stage 2
// immediately accumulates that band's wing contributions into Q, and the slab
// is reused for the next band. Peak Jf residency is therefore
//   16 * (d+1)^2 * band * nsJ  <=  MAX_JF_SLAB_BYTES
// with `band` derived from that budget (>= 1, capped at nsI). One band means
// the old single-shot behavior, so thin-obs callers (the ACA sampler) are
// unaffected.
//
// Exactness of the split: Q[m,n] is a plain sum over the (a, b) wing pairs of
// basis m and basis n, and banding partitions ONLY the `a` axis (each wing a
// of m lands in exactly one band, the one holding segment loc_I[m,a]). Every
// pair is therefore visited exactly once. The accumulation is also kept in
// the original order: a band seeds its local accumulator from the current
// Q[m,n] and adds its in-band (a, b) terms in increasing (a, b), so as long
// as loc_I[m,:] is non-decreasing (true for b-spline supports and for the
// searchsorted local maps the ACA sampler builds) the summation order across
// bands is identical to the unbanded loop, and Q is bit-identical.
static constexpr size_t MAX_JF_SLAB_BYTES = 64u << 20;  // 64 MiB

// ---- The symmetric route (the dense obs == src block) ----------------------
//
// Reciprocity makes the remainder dyad symmetric, but Q is NOT symmetric bit
// for bit -- the projection of (m, n) and of (n, m) round differently -- so
// this route does not mirror Q. What it shares is narrower and exact: for a
// node pair the costly half of `proj_one` (grid interpolation, continuation,
// g: `somm_proj::proj_core`) reads only rho and hh, and both are bit-identical
// with observer and source swapped. Each unordered node pair therefore pays
// for one core and two `proj_project`s instead of two whole `proj_one`s, and
// every Jf entry and every Q entry is then formed by the rectangular kernel's
// own expressions in its own order. Q is bit-identical to the banded kernel's
// (`tests/test_remainder_symmetric_q.py` holds the two routes equal).
//
// Sharing across (i, j) and (j, i) needs both moment blocks live at once,
// which the observer bands cannot give (a band's mirror lives in a later
// band), so this route tiles the BASIS axis instead: a tile pair (A, B),
// A <= B, fills Jf over (segments under A) x (segments under B) and its mirror
// together, then finishes Q's (A, B) and (B, A) blocks outright. A Q entry is
// complete inside one tile pair, so its wing sum can be replayed in the banded
// kernel's exact order: wings a by (band of loc[m, a], a), each against b in
// increasing order -- the same order the bands produce whatever the support
// maps look like. A segment under two tiles (a support's halo) is filled by
// both: a tile's segment set is ~TILE + d, so ~2 d / TILE of the work is
// repeated (3 % at d = 2 and the default tile).
//
// Scratch is per thread, two (d+1)^2 x |S_A| x |S_B| blocks laid out with
// the (p, P) moments innermost, so stage 2 reads each wing pair's moments
// contiguously; the tile is sized so all threads' scratch fits the same
// `max_jf_bytes` budget the bands honour.
static constexpr py::ssize_t SYM_TILE_MAX = 128;

namespace somm_sym {
using somm_proj::cd;

struct Tile {
    py::ssize_t m0 = 0, m1 = 0;     // basis rows [m0, m1)
    std::vector<py::ssize_t> segs;  // the rows' support segments, sorted unique
    std::vector<int32_t> loc;       // (m1 - m0, d1): index of loc[m, a] in segs
};

// Per-segment data, read by raw pointer (every input is c_style + forcecast).
struct Side {
    const double *nodes;  // (ns, q, 3)
    const double *tang;   // (ns, 3)
    const double *W;      // (d1, ns, q)
    py::ssize_t ns, q;
    std::vector<double> ux, uy, th, tz;  // tangent_decomp per segment
    double node(py::ssize_t i, py::ssize_t qi, int c) const {
        return nodes[(i * q + qi) * 3 + c];
    }
    double w(int p, py::ssize_t i, py::ssize_t qi) const {
        return W[(p * ns + i) * q + qi];
    }
};

// Jf[p, P] of observer segment i against source segment j from the node-pair
// block f[qi * q + rj] (observer node qi, source node rj): the rectangular
// kernel's stage-1 contraction, expression for expression, except that its
// inner sum over rj -- which does not read p -- is formed once per (P, qi)
// into `rows` (d1 * q scratch) instead of once per (p, P, qi). The same sums
// in the same order, so the same bits.
static inline void contract(const Side &S, int d1, py::ssize_t i, py::ssize_t j,
                            const cd *f, cd *rows, cd *out) {
    const py::ssize_t q = S.q;
    for (int P = 0; P < d1; ++P) {
        for (py::ssize_t qi = 0; qi < q; ++qi) {
            cd row(0.0, 0.0);
            for (py::ssize_t rj = 0; rj < q; ++rj)
                row += f[qi * q + rj] * S.w(P, j, rj);
            rows[P * q + qi] = row;
        }
    }
    for (int p = 0; p < d1; ++p) {
        for (int P = 0; P < d1; ++P) {
            cd acc(0.0, 0.0);
            for (py::ssize_t qi = 0; qi < q; ++qi) {
                const double wp = S.w(p, i, qi);
                acc += wp * rows[P * q + qi];
            }
            out[p * d1 + P] = acc;
        }
    }
}

// Both orientations of the segment pair (i, j): fw[qi * q + rj] is observer
// node (i, qi) against source node (j, rj), bw[rj * q + qi] the reverse. One
// core per node pair; for i == j the pair (qi, rj) and (rj, qi) is one node
// pair too, and fw alone is filled (it is its own mirror).
static inline void node_block(const somm_proj::GridView &G, double ground_z,
                              double k, const Side &S, py::ssize_t i,
                              py::ssize_t j, cd *fw, cd *bw) {
    const py::ssize_t q = S.q;
    const double *ti = S.tang + i * 3, *tj = S.tang + j * 3;
    for (py::ssize_t qi = 0; qi < q; ++qi) {
        const double ox = S.node(i, qi, 0), oy = S.node(i, qi, 1),
                     oz = S.node(i, qi, 2);
        const py::ssize_t r0 = (i == j) ? qi : 0;
        for (py::ssize_t rj = r0; rj < q; ++rj) {
            const double sx = S.node(j, rj, 0), sy = S.node(j, rj, 1),
                         sz = S.node(j, rj, 2);
            const double dx = ox - sx;
            const double dy = oy - sy;
            const double rho = std::hypot(dx, dy);
            const double hh = (oz - ground_z) + (sz - ground_z);
            cd surf[4], g;
            somm_proj::proj_core(G, k, rho, hh, surf, g);
            fw[qi * q + rj] = somm_proj::proj_project(
                G, dx, dy, rho, surf, g, ti[0], ti[1], ti[2], S.ux[j], S.uy[j],
                S.th[j], S.tz[j]);
            if (i == j && rj == qi) continue;
            // The reverse pair's own differences, as `proj_one` would form
            // them (exactly -dx, -dy).
            const double rdx = sx - ox;
            const double rdy = sy - oy;
            const cd v = somm_proj::proj_project(
                G, rdx, rdy, rho, surf, g, tj[0], tj[1], tj[2], S.ux[i],
                S.uy[i], S.th[i], S.tz[i]);
            if (i == j)
                fw[rj * q + qi] = v;
            else
                bw[rj * q + qi] = v;
        }
    }
}

// Q[m, n] for m under tile T (row role, its local wing indices xl) and n
// under tile U, from J laid out [(x * ny + y) * dd + p * d1 + P]. Wings of m
// in `ord` order, each against b ascending; inner sums as the banded kernel.
static inline void finish_block(const Tile &T, const Tile &U, py::ssize_t ny,
                                int d1, const int32_t *ord, const double *pl,
                                const cd *J, cd *Q, py::ssize_t nb) {
    const int dd = d1 * d1;
    for (py::ssize_t m = T.m0; m < T.m1; ++m) {
        const int32_t *xm = &T.loc[(size_t)(m - T.m0) * d1];
        const int32_t *om = &ord[(size_t)m * d1];
        for (py::ssize_t n = U.m0; n < U.m1; ++n) {
            const int32_t *yn = &U.loc[(size_t)(n - U.m0) * d1];
            cd qmn(0.0, 0.0);
            for (int r = 0; r < d1; ++r) {
                const int a = om[r];
                const double *pma_row = &pl[((size_t)m * d1 + a) * d1];
                const cd *jx = &J[(size_t)xm[a] * ny * dd];
                for (int b = 0; b < d1; ++b) {
                    const double *pnb = &pl[((size_t)n * d1 + b) * d1];
                    const cd *jf = jx + (size_t)yn[b] * dd;
                    cd inner(0.0, 0.0);
                    for (int p = 0; p < d1; ++p) {
                        const double pma = pma_row[p];
                        cd s(0.0, 0.0);
                        for (int P = 0; P < d1; ++P)
                            s += jf[p * d1 + P] * pnb[P];
                        inner += pma * s;
                    }
                    qmn += inner;
                }
            }
            Q[(size_t)m * nb + n] = qmn;
        }
    }
}
}  // namespace somm_sym

// The symmetric route's driver; see the block comment above. `band` is the
// banded kernel's observer band height, read only for the wing order.
static void remainder_Q_symmetric(
    const somm_proj::GridView &G, double ground_z, double k,
    somm_sym::Side &S, int d1, const int64_t *loc, const double *pl,
    py::ssize_t nb, py::ssize_t band, size_t budget, std::complex<double> *Q,
    const volatile int32_t *pysim_cancel, std::atomic<bool> &pysim_aborted) {
    using somm_proj::cd;
    using somm_sym::Tile;
    const py::ssize_t q = S.q;
    const int dd = d1 * d1;

    // Wing order per basis: (band of loc[m, a], a), stable.
    std::vector<int32_t> ord((size_t)nb * d1);
    for (py::ssize_t m = 0; m < nb; ++m) {
        int32_t *om = &ord[(size_t)m * d1];
        for (int a = 0; a < d1; ++a) om[a] = a;
        std::stable_sort(om, om + d1, [&](int32_t x, int32_t y) {
            return loc[m * d1 + x] / band < loc[m * d1 + y] / band;
        });
    }

    int n_threads = 1;
#ifdef _OPENMP
    n_threads = omp_get_max_threads();
#endif
    // Tile height: all threads' two scratch blocks within the budget, with a
    // support halo of d1 - 1 segments on either side of the tile's own.
    py::ssize_t tile = SYM_TILE_MAX;
    while (tile > 1) {
        const double side = (double)(tile + 2 * (d1 - 1));
        const double bytes =
            (double)n_threads * 2.0 * dd * side * side * sizeof(cd);
        if (bytes <= (double)budget) break;
        tile = tile / 2;
    }

    std::vector<Tile> tiles;
    py::ssize_t smax = 0;
    for (py::ssize_t m0 = 0; m0 < nb; m0 += tile) {
        Tile T;
        T.m0 = m0;
        T.m1 = std::min<py::ssize_t>(m0 + tile, nb);
        for (py::ssize_t m = T.m0; m < T.m1; ++m)
            for (int a = 0; a < d1; ++a) T.segs.push_back(loc[m * d1 + a]);
        std::sort(T.segs.begin(), T.segs.end());
        T.segs.erase(std::unique(T.segs.begin(), T.segs.end()), T.segs.end());
        T.loc.resize((size_t)(T.m1 - T.m0) * d1);
        for (py::ssize_t m = T.m0; m < T.m1; ++m)
            for (int a = 0; a < d1; ++a)
                T.loc[(size_t)(m - T.m0) * d1 + a] = (int32_t)(
                    std::lower_bound(T.segs.begin(), T.segs.end(),
                                     loc[m * d1 + a]) -
                    T.segs.begin());
        smax = std::max<py::ssize_t>(smax, (py::ssize_t)T.segs.size());
        tiles.push_back(std::move(T));
    }
    const py::ssize_t n_tiles = (py::ssize_t)tiles.size();
    std::vector<std::pair<py::ssize_t, py::ssize_t>> pairs;
    pairs.reserve((size_t)(n_tiles * (n_tiles + 1) / 2));
    for (py::ssize_t A = 0; A < n_tiles; ++A)
        for (py::ssize_t B = A; B < n_tiles; ++B) pairs.emplace_back(A, B);
    const py::ssize_t n_pairs = (py::ssize_t)pairs.size();

    #pragma omp parallel
    {
        std::vector<cd> JAB((size_t)dd * smax * smax), JBA(JAB.size());
        std::vector<cd> fw((size_t)q * q), bw((size_t)q * q), rows((size_t)d1 * q);
        #pragma omp for schedule(dynamic)
        for (py::ssize_t t = 0; t < n_pairs; ++t) {
            MW_CANCEL_POLL();
            const py::ssize_t A = pairs[(size_t)t].first;
            const py::ssize_t B = pairs[(size_t)t].second;
            const Tile &TA = tiles[(size_t)A], &TB = tiles[(size_t)B];
            const py::ssize_t nx = (py::ssize_t)TA.segs.size();
            const py::ssize_t ny = (py::ssize_t)TB.segs.size();
            const bool diag = A == B;
            for (py::ssize_t x = 0; x < nx; ++x) {
                const py::ssize_t si = TA.segs[(size_t)x];
                // On the diagonal tile pair the (y, x) entry is the mirror of
                // (x, y), so only y >= x is walked.
                for (py::ssize_t y = diag ? x : 0; y < ny; ++y) {
                    const py::ssize_t sj = TB.segs[(size_t)y];
                    cd *jxy = &JAB[((size_t)x * ny + y) * dd];
                    somm_sym::node_block(G, ground_z, k, S, si, sj, fw.data(),
                                         bw.data());
                    somm_sym::contract(S, d1, si, sj, fw.data(), rows.data(), jxy);
                    if (si == sj) {
                        if (!diag)
                            std::copy(jxy, jxy + dd,
                                      &JBA[((size_t)y * nx + x) * dd]);
                        continue;
                    }
                    cd *jyx = diag ? &JAB[((size_t)y * ny + x) * dd]
                                   : &JBA[((size_t)y * nx + x) * dd];
                    somm_sym::contract(S, d1, sj, si, bw.data(), rows.data(), jyx);
                }
            }
            somm_sym::finish_block(TA, TB, ny, d1, ord.data(), pl, JAB.data(),
                                   Q, nb);
            if (!diag)
                somm_sym::finish_block(TB, TA, nx, d1, ord.data(), pl,
                                       JBA.data(), Q, nb);
        }
    }
}

// Calls served by the symmetric route, so a gate can show the route ran
// rather than infer it from agreement (an untaken route agrees trivially).
static std::atomic<long long> g_remainder_symmetric_calls{0};

// True when the obs and src halves of the call are the same data -- the
// dense block's call shape -- so the symmetric route serves it.
template <typename T>
static bool same_array(const py::array_t<T, py::array::c_style | py::array::forcecast> &a,
                       const py::array_t<T, py::array::c_style | py::array::forcecast> &b) {
    if (a.ndim() != b.ndim()) return false;
    for (py::ssize_t d = 0; d < a.ndim(); ++d)
        if (a.shape(d) != b.shape(d)) return false;
    if (a.data() == b.data()) return true;
    return std::memcmp(a.data(), b.data(), sizeof(T) * (size_t)a.size()) == 0;
}

static py::array_t<std::complex<double>> sommerfeld_remainder_bspline_Q(
    py::array_t<double, py::array::c_style | py::array::forcecast> obs_nodes,
    py::array_t<double, py::array::c_style | py::array::forcecast> obs_tang,
    py::array_t<double, py::array::c_style | py::array::forcecast> W_obs,
    py::array_t<double, py::array::c_style | py::array::forcecast> src_nodes,
    py::array_t<double, py::array::c_style | py::array::forcecast> src_tang,
    py::array_t<double, py::array::c_style | py::array::forcecast> W_src,
    py::array_t<int64_t, py::array::c_style | py::array::forcecast> loc_I,
    py::array_t<double, py::array::c_style | py::array::forcecast> pI,
    py::array_t<int64_t, py::array::c_style | py::array::forcecast> loc_J,
    py::array_t<double, py::array::c_style | py::array::forcecast> pJ,
    double ground_z, double k, double r1_max, double r_break, double th_split,
    double r_near,
    py::array_t<double, py::array::c_style | py::array::forcecast> reg_r0,
    py::array_t<double, py::array::c_style | py::array::forcecast> reg_dr,
    py::array_t<double, py::array::c_style | py::array::forcecast> reg_th0,
    py::array_t<double, py::array::c_style | py::array::forcecast> reg_dth,
    std::vector<py::array_t<std::complex<double>,
                            py::array::c_style | py::array::forcecast>> reg_vals,
    py::array_t<std::complex<double>, py::array::c_style | py::array::forcecast>
        far,
    uintptr_t cancel_flag = 0, size_t max_jf_bytes = 0, bool reference = false) {
    using somm_proj::cd;
    auto ndI = obs_nodes.unchecked<3>();
    auto tgI = obs_tang.unchecked<2>();
    auto WvI = W_obs.unchecked<3>();
    auto ndJ = src_nodes.unchecked<3>();
    auto tgJ = src_tang.unchecked<2>();
    auto WvJ = W_src.unchecked<3>();
    auto lI = loc_I.unchecked<2>();
    auto lJ = loc_J.unchecked<2>();
    auto plI = pI.unchecked<3>();
    auto plJ = pJ.unchecked<3>();

    const py::ssize_t nsI = ndI.shape(0);
    const py::ssize_t nsJ = ndJ.shape(0);
    const py::ssize_t q = ndI.shape(1);
    const int d1 = (int)WvI.shape(0);  // degree + 1
    const py::ssize_t nI = lI.shape(0);
    const py::ssize_t nJ = lJ.shape(0);
    if (ndI.shape(2) != 3 || ndJ.shape(2) != 3 || ndJ.shape(1) != q)
        throw std::runtime_error("obs/src nodes must be (n_seg, q, 3), same q");
    if (WvI.shape(1) != nsI || WvI.shape(2) != q || WvJ.shape(1) != nsJ ||
        WvJ.shape(2) != q || (int)WvJ.shape(0) != d1)
        throw std::runtime_error("W_obs/W_src inconsistent with nodes/degree");
    if (lI.shape(1) != d1 || lJ.shape(1) != d1 || plI.shape(1) != d1 ||
        plI.shape(2) != d1 || plJ.shape(1) != d1 || plJ.shape(2) != d1)
        throw std::runtime_error("loc/polys inconsistent with degree");

    somm_proj::GridView G = somm_proj::build_grid_view(
        r1_max, r_break, th_split, r_near, reg_r0.unchecked<1>(),
        reg_dr.unchecked<1>(), reg_th0.unchecked<1>(), reg_dth.unchecked<1>(),
        reg_vals);
    somm_proj::set_far(G, far);

    // The dense block's call shape (obs == src throughout) takes the
    // symmetric route unless `reference` asks for this banded kernel, which
    // stays the route of every rectangular call and the oracle the symmetric
    // one is held to.
    const bool symmetric =
        !reference && same_array(obs_nodes, src_nodes) &&
        same_array(obs_tang, src_tang) && same_array(W_obs, W_src) &&
        same_array(loc_I, loc_J) && same_array(pI, pJ);

    py::array_t<std::complex<double>> Q({nI, nJ});
    auto Qm = Q.mutable_unchecked<2>();

    // Observer band size: the largest number of obs segments whose Jf slab
    // fits the budget (#343). `band == nsI` reproduces the unbanded kernel.
    const size_t budget = max_jf_bytes ? max_jf_bytes : MAX_JF_SLAB_BYTES;
    const size_t row_bytes = sizeof(cd) * (size_t)d1 * d1 * (size_t)nsJ;
    py::ssize_t band = nsI;
    if (row_bytes > 0) {
        size_t fit = budget / row_bytes;
        if (fit < 1) fit = 1;
        if ((py::ssize_t)fit < band) band = (py::ssize_t)fit;
    }
    if (band < 1) band = 1;

    if (symmetric) {
        g_remainder_symmetric_calls.fetch_add(1, std::memory_order_relaxed);
        somm_sym::Side S;
        S.nodes = obs_nodes.data();
        S.tang = obs_tang.data();
        S.W = W_obs.data();
        S.ns = nsI;
        S.q = q;
        S.ux.resize(nsI);
        S.uy.resize(nsI);
        S.th.resize(nsI);
        S.tz.resize(nsI);
        for (py::ssize_t j = 0; j < nsI; ++j)
            somm_proj::tangent_decomp(tgI(j, 0), tgI(j, 1), tgI(j, 2), S.ux[j],
                                      S.uy[j], S.th[j], S.tz[j]);
        cd *Qp = Q.mutable_data();
        const int64_t *locp = loc_I.data();
        const double *plp = pI.data();
        py::gil_scoped_release release;
        MW_CANCEL_SETUP(cancel_flag);
        remainder_Q_symmetric(G, ground_z, k, S, d1, locp, plp, nI, band,
                              budget, Qp, pysim_cancel, pysim_aborted);
        MW_THROW_IF_ABORTED();
        return Q;
    }

    std::fill(Q.mutable_data(), Q.mutable_data() + (size_t)nI * nJ, cd(0.0, 0.0));

    py::gil_scoped_release release;

    // Per-src-segment tangent decomposition.
    std::vector<double> sux(nsJ), suy(nsJ), sth(nsJ), stz(nsJ);
    for (py::ssize_t j = 0; j < nsJ; ++j)
        somm_proj::tangent_decomp(tgJ(j, 0), tgJ(j, 1), tgJ(j, 2), sux[j],
                                  suy[j], sth[j], stz[j]);

    // Stage 1 slab: Jf[p,P,i-i0,j] over the (band, nsJ) segment rectangle.
    // Flat index (((p*d1+P)*ib)+(i-i0))*nsJ+j, ib = this band's height.
    std::vector<cd> Jf((size_t)d1 * d1 * (size_t)band * nsJ);
    std::vector<py::ssize_t> rows;
    rows.reserve((size_t)std::min<py::ssize_t>(nI, band * d1 + d1));

    MW_CANCEL_SETUP(cancel_flag);
    for (py::ssize_t i0 = 0; i0 < nsI; i0 += band) {
        const py::ssize_t i1 = std::min<py::ssize_t>(i0 + band, nsI);
        const py::ssize_t ib = i1 - i0;
        const size_t seg2 = (size_t)ib * nsJ;

        // Stage 1: fill the band's moment slab.
        #pragma omp parallel for schedule(dynamic)
        for (py::ssize_t i = i0; i < i1; ++i) {
            MW_CANCEL_POLL();
            const double tox = tgI(i, 0), toy = tgI(i, 1), toz = tgI(i, 2);
            std::vector<cd> fblk((size_t)q * q);
            for (py::ssize_t j = 0; j < nsJ; ++j) {
                for (py::ssize_t qi = 0; qi < q; ++qi) {
                    const double ox = ndI(i, qi, 0), oy = ndI(i, qi, 1),
                                 oz = ndI(i, qi, 2);
                    for (py::ssize_t rj = 0; rj < q; ++rj) {
                        fblk[qi * q + rj] = somm_proj::proj_one(
                            G, ground_z, k, ox, oy, oz, tox, toy, toz,
                            ndJ(j, rj, 0), ndJ(j, rj, 1), ndJ(j, rj, 2),
                            sux[j], suy[j], sth[j], stz[j]);
                    }
                }
                for (int p = 0; p < d1; ++p) {
                    for (int P = 0; P < d1; ++P) {
                        cd acc(0.0, 0.0);
                        for (py::ssize_t qi = 0; qi < q; ++qi) {
                            const double wp = WvI(p, i, qi);
                            cd row(0.0, 0.0);
                            for (py::ssize_t rj = 0; rj < q; ++rj)
                                row += fblk[qi * q + rj] * WvJ(P, j, rj);
                            acc += wp * row;
                        }
                        Jf[((size_t)(p * d1 + P) * ib + (i - i0)) * nsJ + j] =
                            acc;
                    }
                }
            }
        }
        MW_THROW_IF_ABORTED();

        // Which basis rows have at least one wing landing in this band? A
        // row is listed once however many of its wings are in-band, so the
        // parallel stage-2 loop below owns each Q row exclusively.
        rows.clear();
        for (py::ssize_t m = 0; m < nI; ++m) {
            for (int a = 0; a < d1; ++a) {
                const py::ssize_t si = lI(m, a);
                if (si >= i0 && si < i1) {
                    rows.push_back(m);
                    break;
                }
            }
        }

        // Stage 2: accumulate this band's wing contributions into Q. The
        // out-of-band wings are skipped here and picked up by the band that
        // owns them, so every (a, b) pair is summed exactly once.
        const py::ssize_t n_rows = (py::ssize_t)rows.size();
        #pragma omp parallel for schedule(static)
        for (py::ssize_t r = 0; r < n_rows; ++r) {
            MW_CANCEL_POLL();
            const py::ssize_t m = rows[(size_t)r];
            for (py::ssize_t n = 0; n < nJ; ++n) {
                cd qmn = Qm(m, n);  // seeded, so the a-order is preserved
                for (int a = 0; a < d1; ++a) {
                    const py::ssize_t si = lI(m, a);
                    if (si < i0 || si >= i1) continue;
                    const py::ssize_t sl = si - i0;
                    for (int b = 0; b < d1; ++b) {
                        const py::ssize_t sj = lJ(n, b);
                        cd inner(0.0, 0.0);
                        for (int p = 0; p < d1; ++p) {
                            const double pma = plI(m, a, p);
                            const cd *jfp =
                                &Jf[((size_t)(p * d1) * ib + sl) * nsJ + sj];
                            cd s(0.0, 0.0);
                            for (int P = 0; P < d1; ++P)
                                s += jfp[(size_t)P * seg2] * plJ(n, b, P);
                            inner += pma * s;
                        }
                        qmn += inner;
                    }
                }
                Qm(m, n) = qmn;
            }
        }
        MW_THROW_IF_ABORTED();
    }
    return Q;
}


void register_somm(py::module_ &m) {

    m.def("somm_six_integrals_batch", &somm_six_integrals_batch,
          "Batched Sommerfeld six-integral evaluation at (rho[i], h[i]) "
          "nodes; OpenMP across nodes. form: 0 auto, 1 Bessel, 2 Hankel. "
          "Returns (n, 6) complex in _six_integrals order.",
          py::arg("eps_t"), py::arg("k2"), py::arg("rho"), py::arg("h"),
          py::arg("rtol") = 1e-9, py::arg("form") = 0,
          py::arg("cancel_flag") = 0, py::arg("lanes") = true);
    // momwire#1441 step 3: the integrand runs four contour nodes at a time;
    // `lanes=False` is the scalar walk it is gated against, bit for bit.
    m.attr("somm_six_lanes_1441") = true;
    m.def("remainder_field_proj_batch", &remainder_field_proj_batch,
          "Fused Sommerfeld smooth-remainder assembly: interpolate the four "
          "grid surfaces (4x4 Lagrange) and project t_m.F(r_m,r_n).t_n per "
          "(observer, source) pair; OpenMP over observer rows. Returns (M, S) "
          "complex. The grid is passed flattened (per-region r0/dr/th0/dth "
          "arrays + a list of 3 near — or 5 with the #159 far zone — "
          "(4,n_r,n_th) value tables), then `far`: the momwire#1258 "
          "continuation past r1_max (`_sommerfeld.far_cpp_pack`; empty "
          "serves the edge value frozen).",
          py::arg("obs"), py::arg("t_obs"), py::arg("src"), py::arg("t_src"),
          py::arg("ground_z"), py::arg("k"), py::arg("r1_max"),
          py::arg("r_break"), py::arg("th_split"), py::arg("r_near"),
          py::arg("reg_r0"), py::arg("reg_dr"), py::arg("reg_th0"),
          py::arg("reg_dth"), py::arg("reg_vals"), py::arg("far"),
          py::arg("cancel_flag") = 0, py::arg("lanes") = true);
    m.def("reflected_field_batch", &reflected_field_batch_impl,
          "The Sommerfeld ground's near-field REMAINDER at observation points "
          "(`_somm_field_core.cpp`): E, or with magnetic=True H = curl(E) / "
          "(-j omega mu) by central differences at `step`. points (N, 3), "
          "src_mid (S, 3), src_moment (S, 3) complex; returns (N, 3) complex. "
          "The grid flattened as `remainder_field_proj_batch` takes it, with "
          "its far pack. GIL released, OpenMP over points.",
          py::arg("points"), py::arg("src_mid"), py::arg("src_moment"),
          py::arg("ground_z"), py::arg("k"), py::arg("magnetic"), py::arg("omega"),
          py::arg("mu"), py::arg("step"), py::arg("r1_max"), py::arg("r_break"),
          py::arg("th_split"), py::arg("r_near"), py::arg("reg_r0"), py::arg("reg_dr"),
          py::arg("reg_th0"), py::arg("reg_dth"), py::arg("reg_vals"), py::arg("far"));
    m.def("remainder_field_proj_nodes", &remainder_field_proj_nodes,
          "The lazy grids' census (momwire#1512): per region a (n_r, n_th) "
          "uint8 mask of the nodes remainder_field_proj_batch's reads over "
          "obs x src take (its own geometry and stencil address). The grid "
          "as remainder_field_proj_batch takes it, without `far`.",
          py::arg("obs"), py::arg("src"), py::arg("ground_z"), py::arg("r1_max"),
          py::arg("r_break"), py::arg("th_split"), py::arg("r_near"), py::arg("reg_r0"),
          py::arg("reg_dr"), py::arg("reg_th0"), py::arg("reg_dth"), py::arg("reg_vals"));
    m.def("reflected_field_nodes", &reflected_field_nodes_impl,
          "The lazy grids' census of reflected_field_batch (momwire#1512): the "
          "nodes its reads at `points` take (with magnetic, H's six offset "
          "points), a (n_r, n_th) uint8 mask per region.",
          py::arg("points"), py::arg("src_mid"), py::arg("ground_z"), py::arg("magnetic"),
          py::arg("step"), py::arg("r1_max"), py::arg("r_break"), py::arg("th_split"),
          py::arg("r_near"), py::arg("reg_r0"), py::arg("reg_dr"), py::arg("reg_th0"),
          py::arg("reg_dth"), py::arg("reg_vals"));
    m.def("somm_proj_lanes_built", []() { return (bool)MW_SOMM_LANES; },
          "Whether this build has remainder_field_proj_batch's AVX2 lanes "
          "(perf item 9); without them `lanes=True` runs the per-pair loop.");
    m.def("somm_proj_lane_pairs",
          []() { return g_somm_proj_lane_pairs.load(); },
          "How many pairs remainder_field_proj_batch has sent through its "
          "AVX2 lanes in this process (perf item 9): the lanes' output equals "
          "the per-pair loop's, so only this says which ran.");
    m.def("sommerfeld_remainder_bspline_Q", &sommerfeld_remainder_bspline_Q,
          "Fully-fused b-spline Galerkin Sommerfeld remainder over an obs/src "
          "rectangle: interpolate + project + moment-quadrature + basis-assemble "
          "into the (nI, nJ) Q block directly (no Jf / einsum intermediates). "
          "Dense block = symmetric case (obs==src, loc==supp_seg); the ACA "
          "sampler passes thin segment subsets with local support maps. Grid as "
          "in remainder_field_proj_batch. The internal moment slab is banded "
          "over observer segments so its residency is bounded by "
          "`max_jf_bytes` (0 = the 64 MiB default), never the full "
          "(d+1)^2 * nsI * nsJ tensor (momwire#343); the banding is exact and "
          "order-preserving, not an approximation. A call whose obs and src "
          "halves are the same data (the dense block) takes the symmetric "
          "route, one grid interpolation per unordered node pair, bit-identical "
          "to the banded kernel; `reference=True` forces the banded kernel.",
          py::arg("obs_nodes"), py::arg("obs_tang"), py::arg("W_obs"),
          py::arg("src_nodes"), py::arg("src_tang"), py::arg("W_src"),
          py::arg("loc_I"), py::arg("pI"), py::arg("loc_J"), py::arg("pJ"),
          py::arg("ground_z"), py::arg("k"),
          py::arg("r1_max"), py::arg("r_break"), py::arg("th_split"),
          py::arg("r_near"), py::arg("reg_r0"), py::arg("reg_dr"), py::arg("reg_th0"),
          py::arg("reg_dth"), py::arg("reg_vals"), py::arg("far"),
          py::arg("cancel_flag") = 0, py::arg("max_jf_bytes") = 0,
          py::arg("reference") = false);
    m.def("remainder_q_symmetric_calls",
          []() { return g_remainder_symmetric_calls.load(); },
          "How many sommerfeld_remainder_bspline_Q calls the symmetric route "
          "has served in this process (a gate's evidence that it ran).");
}

