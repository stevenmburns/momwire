// The Sommerfeld remainder's projection and moment reduction, Python-free
// (momwire#1479 S3), so the standalone executable runs the same kernels the
// extension does. Moved verbatim from `_accel_somm.cpp` (the projection's
// loop and lanes) and `_accel_mw568.cpp` (the moment reduction); those TUs
// keep only the bindings. `_mw_core.h` comes first for the TU-wide
// `omp declare simd` declarations (momwire#1194), as in every kernel TU.
#include "_study_timers.h"
#include "_mw_core.h"
#include "_somm_proj_core.h"

#ifdef _OPENMP
#include <omp.h>
#endif

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
#if MW_LANES_SIMD && MW_LANES_PACKED && !defined(MW_STUDY_SSE2_LANES_NI_ONLY)
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

namespace somm_core {

bool proj_lanes_built() { return (bool)MW_SOMM_LANES; }
unsigned long long proj_lane_pairs() { return g_somm_proj_lane_pairs.load(); }

void field_proj_batch(const somm_proj::GridView &G, size_t n_reg,
                      std::ptrdiff_t M, std::ptrdiff_t S, const double *obs,
                      const double *t_obs, const double *src,
                      const double *t_src, double ground_z, double k,
                      std::complex<double> *out, uintptr_t cancel_flag,
                      bool lanes) {
    STUDY_T("proj.above_batch");
    using somm_proj::cd;
    // Per-source constants (n-only): position + tangent decomposition.
    std::vector<double> sx(S), sy(S), sz(S), ux(S), uy(S), thsrc(S), tzsrc(S);
    for (std::ptrdiff_t n = 0; n < S; ++n) {
        sx[n] = src[3 * n];
        sy[n] = src[3 * n + 1];
        sz[n] = src[3 * n + 2];
        somm_proj::tangent_decomp(t_src[3 * n], t_src[3 * n + 1],
                                  t_src[3 * n + 2], ux[n], uy[n],
                                  thsrc[n], tzsrc[n]);
    }

#if MW_SOMM_LANES
    int nR32[somm_proj::MAX_REGIONS], nTh32[somm_proj::MAX_REGIONS];
    for (size_t g = 0; g < n_reg; ++g) {
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
    for (std::ptrdiff_t m = 0; m < M; ++m) {
        MW_CANCEL_POLL();
        const double ox = obs[3 * m], oy = obs[3 * m + 1],
                     oz = obs[3 * m + 2];
        const double tox = t_obs[3 * m], toy = t_obs[3 * m + 1],
                     toz = t_obs[3 * m + 2];
#if MW_SOMM_LANES
        if (use_lanes) {
            using namespace somm_lanes;
            constexpr int B = ABOVE_BLOCK;
            double dx[B], dy[B], rho[B], r1[B], th[B], gre[B], gim[B];
            int l_reg[B], l_i0[B], l_j0[B];
            double l_wr[4][B], l_wt[4][B], l_sre[4][B], l_sim[4][B];
            cd *orow = out + m * S;
            for (std::ptrdiff_t n0 = 0; n0 < S; n0 += B) {
                const int nb = (int)std::min<std::ptrdiff_t>(B, S - n0);
                const int nq4 = nb - nb % W;
                // Geometry and the libm calls, per pair: `proj_one`'s
                // dx/dy/rho/hh and `proj_core`'s r1, theta (clamped) and g.
                for (int b = 0; b < nq4; ++b) {
                    const std::ptrdiff_t n = n0 + b;
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
                    const std::ptrdiff_t n = n0 + b;
                    orow[n] = somm_proj::proj_one(
                        G, ground_z, k, ox, oy, oz, tox, toy, toz, sx[n], sy[n],
                        sz[n], ux[n], uy[n], thsrc[n], tzsrc[n]);
                }
            }
            continue;
        }
#endif
        for (std::ptrdiff_t n = 0; n < S; ++n) {
            out[m * S + n] = somm_proj::proj_one(
                G, ground_z, k, ox, oy, oz, tox, toy, toz, sx[n], sy[n], sz[n],
                ux[n], uy[n], thsrc[n], tzsrc[n]);
        }
    }
    MW_THROW_IF_ABORTED();
}

void field_proj_nodes(const somm_proj::GridView &G, std::ptrdiff_t M, std::ptrdiff_t S,
                      const double *obs, const double *src, double ground_z,
                      somm_proj::NodeMarks &marks) {
    // Every pair `field_proj_batch` reads (both of its loops take
    // `pair_geometry`, `proj_r1_theta` and the address `proj_address` gives,
    // the lanes' through `stencil_lanes`, which is the same address lane by
    // lane), marked at its stencil.
    for (std::ptrdiff_t m = 0; m < M; ++m) {
        const double ox = obs[3 * m], oy = obs[3 * m + 1], oz = obs[3 * m + 2];
        for (std::ptrdiff_t n = 0; n < S; ++n) {
            double dx, dy, rho, hh, r1, theta, fr, ft;
            int i0, j0;
            somm_proj::pair_geometry(ground_z, ox, oy, oz, src[3 * n], src[3 * n + 1],
                                     src[3 * n + 2], dx, dy, rho, hh);
            somm_proj::proj_r1_theta(G, rho, hh, r1, theta);
            const int reg = somm_proj::proj_address(G, r1, theta, fr, ft, i0, j0);
            marks.mark(reg, i0, j0);
        }
    }
}

void moment_reduce(const std::complex<double> *proj, const double *W,
                   std::ptrdiff_t P, std::ptrdiff_t N, std::ptrdiff_t q,
                   std::ptrdiff_t M, std::complex<double> *out) {
    STUDY_T("proj.moment_reduce");
    typedef std::complex<double> cd;
    #pragma omp parallel for schedule(static)
    for (std::ptrdiff_t o = 0; o < M; ++o) {
        const cd *row = proj + o * N * q;
        cd *orow = out + o * N * P;
        for (std::ptrdiff_t j = 0; j < N; ++j) {
            const cd *pn = row + j * q;
            for (std::ptrdiff_t m = 0; m < P; ++m) {
                const double *wn = W + (m * N + j) * q;
                double are = 0.0, aim = 0.0;
                for (std::ptrdiff_t k = 0; k < q; ++k) {
                    const double pr = pn[k].real(), pi = pn[k].imag();
                    const double w = wn[k], z = 0.0;
                    const double re = pr * w - pi * z;
                    const double im = pr * z + pi * w;
                    are = re + are;
                    aim = im + aim;
                }
                orow[j * P + m] = cd(are, aim);
            }
        }
    }
}

}  // namespace somm_core
