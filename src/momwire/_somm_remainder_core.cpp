// See `_somm_remainder_core.h`. Moved verbatim from `_accel_somm.cpp`
// (momwire#1517 step 3); `_mw_core.h` first, as in every kernel TU, for the
// TU-wide `omp declare simd` declarations (momwire#1194) and the cancel
// macros.
#include "_mw_core.h"
#include "_somm_remainder_core.h"

#include <algorithm>
#include <atomic>
#include <cstring>
#include <utility>
#include <vector>
#ifdef _OPENMP
#include <omp.h>
#endif

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
static constexpr std::ptrdiff_t SYM_TILE_MAX = 128;

namespace somm_sym {
using somm_proj::cd;

struct Tile {
    std::ptrdiff_t m0 = 0, m1 = 0;     // basis rows [m0, m1)
    std::vector<std::ptrdiff_t> segs;  // the rows' support segments, sorted unique
    std::vector<int32_t> loc;       // (m1 - m0, d1): index of loc[m, a] in segs
};

// Per-segment data, read by raw pointer (every input is c_style + forcecast).
struct Side {
    const double *nodes;  // (ns, q, 3)
    const double *tang;   // (ns, 3)
    const double *W;      // (d1, ns, q)
    std::ptrdiff_t ns, q;
    std::vector<double> ux, uy, th, tz;  // tangent_decomp per segment
    double node(std::ptrdiff_t i, std::ptrdiff_t qi, int c) const {
        return nodes[(i * q + qi) * 3 + c];
    }
    double w(int p, std::ptrdiff_t i, std::ptrdiff_t qi) const {
        return W[(p * ns + i) * q + qi];
    }
};

// Jf[p, P] of observer segment i against source segment j from the node-pair
// block f[qi * q + rj] (observer node qi, source node rj): the rectangular
// kernel's stage-1 contraction, expression for expression, except that its
// inner sum over rj -- which does not read p -- is formed once per (P, qi)
// into `rows` (d1 * q scratch) instead of once per (p, P, qi). The same sums
// in the same order, so the same bits.
static inline void contract(const Side &S, int d1, std::ptrdiff_t i, std::ptrdiff_t j,
                            const cd *f, cd *rows, cd *out) {
    const std::ptrdiff_t q = S.q;
    for (int P = 0; P < d1; ++P) {
        for (std::ptrdiff_t qi = 0; qi < q; ++qi) {
            cd row(0.0, 0.0);
            for (std::ptrdiff_t rj = 0; rj < q; ++rj)
                row += f[qi * q + rj] * S.w(P, j, rj);
            rows[P * q + qi] = row;
        }
    }
    for (int p = 0; p < d1; ++p) {
        for (int P = 0; P < d1; ++P) {
            cd acc(0.0, 0.0);
            for (std::ptrdiff_t qi = 0; qi < q; ++qi) {
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
                              double k, const Side &S, std::ptrdiff_t i,
                              std::ptrdiff_t j, cd *fw, cd *bw) {
    const std::ptrdiff_t q = S.q;
    const double *ti = S.tang + i * 3, *tj = S.tang + j * 3;
    for (std::ptrdiff_t qi = 0; qi < q; ++qi) {
        const double ox = S.node(i, qi, 0), oy = S.node(i, qi, 1),
                     oz = S.node(i, qi, 2);
        const std::ptrdiff_t r0 = (i == j) ? qi : 0;
        for (std::ptrdiff_t rj = r0; rj < q; ++rj) {
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
static inline void finish_block(const Tile &T, const Tile &U, std::ptrdiff_t ny,
                                int d1, const int32_t *ord, const double *pl,
                                const cd *J, cd *Q, std::ptrdiff_t nb) {
    const int dd = d1 * d1;
    for (std::ptrdiff_t m = T.m0; m < T.m1; ++m) {
        const int32_t *xm = &T.loc[(size_t)(m - T.m0) * d1];
        const int32_t *om = &ord[(size_t)m * d1];
        for (std::ptrdiff_t n = U.m0; n < U.m1; ++n) {
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
    std::ptrdiff_t nb, std::ptrdiff_t band, size_t budget, std::complex<double> *Q,
    const volatile int32_t *pysim_cancel, std::atomic<bool> &pysim_aborted) {
    using somm_proj::cd;
    using somm_sym::Tile;
    const std::ptrdiff_t q = S.q;
    const int dd = d1 * d1;

    // Wing order per basis: (band of loc[m, a], a), stable.
    std::vector<int32_t> ord((size_t)nb * d1);
    for (std::ptrdiff_t m = 0; m < nb; ++m) {
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
    std::ptrdiff_t tile = SYM_TILE_MAX;
    while (tile > 1) {
        const double side = (double)(tile + 2 * (d1 - 1));
        const double bytes =
            (double)n_threads * 2.0 * dd * side * side * sizeof(cd);
        if (bytes <= (double)budget) break;
        tile = tile / 2;
    }

    std::vector<Tile> tiles;
    std::ptrdiff_t smax = 0;
    for (std::ptrdiff_t m0 = 0; m0 < nb; m0 += tile) {
        Tile T;
        T.m0 = m0;
        T.m1 = std::min<std::ptrdiff_t>(m0 + tile, nb);
        for (std::ptrdiff_t m = T.m0; m < T.m1; ++m)
            for (int a = 0; a < d1; ++a) T.segs.push_back(loc[m * d1 + a]);
        std::sort(T.segs.begin(), T.segs.end());
        T.segs.erase(std::unique(T.segs.begin(), T.segs.end()), T.segs.end());
        T.loc.resize((size_t)(T.m1 - T.m0) * d1);
        for (std::ptrdiff_t m = T.m0; m < T.m1; ++m)
            for (int a = 0; a < d1; ++a)
                T.loc[(size_t)(m - T.m0) * d1 + a] = (int32_t)(
                    std::lower_bound(T.segs.begin(), T.segs.end(),
                                     loc[m * d1 + a]) -
                    T.segs.begin());
        smax = std::max<std::ptrdiff_t>(smax, (std::ptrdiff_t)T.segs.size());
        tiles.push_back(std::move(T));
    }
    const std::ptrdiff_t n_tiles = (std::ptrdiff_t)tiles.size();
    std::vector<std::pair<std::ptrdiff_t, std::ptrdiff_t>> pairs;
    pairs.reserve((size_t)(n_tiles * (n_tiles + 1) / 2));
    for (std::ptrdiff_t A = 0; A < n_tiles; ++A)
        for (std::ptrdiff_t B = A; B < n_tiles; ++B) pairs.emplace_back(A, B);
    const std::ptrdiff_t n_pairs = (std::ptrdiff_t)pairs.size();

    #pragma omp parallel
    {
        std::vector<cd> JAB((size_t)dd * smax * smax), JBA(JAB.size());
        std::vector<cd> fw((size_t)q * q), bw((size_t)q * q), rows((size_t)d1 * q);
        #pragma omp for schedule(dynamic)
        for (std::ptrdiff_t t = 0; t < n_pairs; ++t) {
            MW_CANCEL_POLL();
            const std::ptrdiff_t A = pairs[(size_t)t].first;
            const std::ptrdiff_t B = pairs[(size_t)t].second;
            const Tile &TA = tiles[(size_t)A], &TB = tiles[(size_t)B];
            const std::ptrdiff_t nx = (std::ptrdiff_t)TA.segs.size();
            const std::ptrdiff_t ny = (std::ptrdiff_t)TB.segs.size();
            const bool diag = A == B;
            for (std::ptrdiff_t x = 0; x < nx; ++x) {
                const std::ptrdiff_t si = TA.segs[(size_t)x];
                // On the diagonal tile pair the (y, x) entry is the mirror of
                // (x, y), so only y >= x is walked.
                for (std::ptrdiff_t y = diag ? x : 0; y < ny; ++y) {
                    const std::ptrdiff_t sj = TB.segs[(size_t)y];
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


namespace somm_core {

void remainder_bspline_Q(const somm_proj::GridView &G, double ground_z, double k,
                         const double *obs_nodes, const double *obs_tang, const double *W_obs,
                         std::ptrdiff_t nsI, const double *src_nodes, const double *src_tang,
                         const double *W_src, std::ptrdiff_t nsJ, std::ptrdiff_t q, int d1,
                         const int64_t *loc_I, const double *pI, std::ptrdiff_t nI,
                         const int64_t *loc_J, const double *pJ, std::ptrdiff_t nJ, bool symmetric,
                         size_t max_jf_bytes, std::complex<double> *Q, uintptr_t cancel_flag) {
    using somm_proj::cd;
    // the binding's accessors, over the raw C-ordered arrays
    const auto ndI = [&](std::ptrdiff_t i, std::ptrdiff_t qi, int c) { return obs_nodes[(i * q + qi) * 3 + c]; };
    const auto tgI = [&](std::ptrdiff_t i, int c) { return obs_tang[i * 3 + c]; };
    const auto WvI = [&](int p, std::ptrdiff_t i, std::ptrdiff_t qi) { return W_obs[(p * nsI + i) * q + qi]; };
    const auto ndJ = [&](std::ptrdiff_t j, std::ptrdiff_t rj, int c) { return src_nodes[(j * q + rj) * 3 + c]; };
    const auto tgJ = [&](std::ptrdiff_t j, int c) { return src_tang[j * 3 + c]; };
    const auto WvJ = [&](int p, std::ptrdiff_t j, std::ptrdiff_t rj) { return W_src[(p * nsJ + j) * q + rj]; };
    const auto lI = [&](std::ptrdiff_t m, int a) { return loc_I[m * d1 + a]; };
    const auto lJ = [&](std::ptrdiff_t n, int b) { return loc_J[n * d1 + b]; };
    const auto plI = [&](std::ptrdiff_t m, int a, int p) { return pI[(m * d1 + a) * d1 + p]; };
    const auto plJ = [&](std::ptrdiff_t n, int b, int P) { return pJ[(n * d1 + b) * d1 + P]; };
    const auto Qm = [&](std::ptrdiff_t m, std::ptrdiff_t n) -> cd & { return Q[m * nJ + n]; };

    // Observer band size: the largest number of obs segments whose Jf slab
    // fits the budget (#343). `band == nsI` reproduces the unbanded kernel.
    const size_t budget = max_jf_bytes ? max_jf_bytes : MAX_JF_SLAB_BYTES;
    const size_t row_bytes = sizeof(cd) * (size_t)d1 * d1 * (size_t)nsJ;
    std::ptrdiff_t band = nsI;
    if (row_bytes > 0) {
        size_t fit = budget / row_bytes;
        if (fit < 1) fit = 1;
        if ((std::ptrdiff_t)fit < band) band = (std::ptrdiff_t)fit;
    }
    if (band < 1) band = 1;

    if (symmetric) {
        somm_sym::Side S;
        S.nodes = obs_nodes;
        S.tang = obs_tang;
        S.W = W_obs;
        S.ns = nsI;
        S.q = q;
        S.ux.resize(nsI);
        S.uy.resize(nsI);
        S.th.resize(nsI);
        S.tz.resize(nsI);
        for (std::ptrdiff_t j = 0; j < nsI; ++j)
            somm_proj::tangent_decomp(tgI(j, 0), tgI(j, 1), tgI(j, 2), S.ux[j],
                                      S.uy[j], S.th[j], S.tz[j]);
        MW_CANCEL_SETUP(cancel_flag);
        remainder_Q_symmetric(G, ground_z, k, S, d1, loc_I, pI, nI, band,
                              budget, Q, pysim_cancel, pysim_aborted);
        MW_THROW_IF_ABORTED();
        return;
    }

    std::fill(Q, Q + (size_t)nI * nJ, cd(0.0, 0.0));

    // Per-src-segment tangent decomposition.
    std::vector<double> sux(nsJ), suy(nsJ), sth(nsJ), stz(nsJ);
    for (std::ptrdiff_t j = 0; j < nsJ; ++j)
        somm_proj::tangent_decomp(tgJ(j, 0), tgJ(j, 1), tgJ(j, 2), sux[j],
                                  suy[j], sth[j], stz[j]);

    // Stage 1 slab: Jf[p,P,i-i0,j] over the (band, nsJ) segment rectangle.
    // Flat index (((p*d1+P)*ib)+(i-i0))*nsJ+j, ib = this band's height.
    std::vector<cd> Jf((size_t)d1 * d1 * (size_t)band * nsJ);
    std::vector<std::ptrdiff_t> rows;
    rows.reserve((size_t)std::min<std::ptrdiff_t>(nI, band * d1 + d1));

    MW_CANCEL_SETUP(cancel_flag);
    for (std::ptrdiff_t i0 = 0; i0 < nsI; i0 += band) {
        const std::ptrdiff_t i1 = std::min<std::ptrdiff_t>(i0 + band, nsI);
        const std::ptrdiff_t ib = i1 - i0;
        const size_t seg2 = (size_t)ib * nsJ;

        // Stage 1: fill the band's moment slab.
        #pragma omp parallel for schedule(dynamic)
        for (std::ptrdiff_t i = i0; i < i1; ++i) {
            MW_CANCEL_POLL();
            const double tox = tgI(i, 0), toy = tgI(i, 1), toz = tgI(i, 2);
            std::vector<cd> fblk((size_t)q * q);
            for (std::ptrdiff_t j = 0; j < nsJ; ++j) {
                for (std::ptrdiff_t qi = 0; qi < q; ++qi) {
                    const double ox = ndI(i, qi, 0), oy = ndI(i, qi, 1),
                                 oz = ndI(i, qi, 2);
                    for (std::ptrdiff_t rj = 0; rj < q; ++rj) {
                        fblk[qi * q + rj] = somm_proj::proj_one(
                            G, ground_z, k, ox, oy, oz, tox, toy, toz,
                            ndJ(j, rj, 0), ndJ(j, rj, 1), ndJ(j, rj, 2),
                            sux[j], suy[j], sth[j], stz[j]);
                    }
                }
                for (int p = 0; p < d1; ++p) {
                    for (int P = 0; P < d1; ++P) {
                        cd acc(0.0, 0.0);
                        for (std::ptrdiff_t qi = 0; qi < q; ++qi) {
                            const double wp = WvI(p, i, qi);
                            cd row(0.0, 0.0);
                            for (std::ptrdiff_t rj = 0; rj < q; ++rj)
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
        for (std::ptrdiff_t m = 0; m < nI; ++m) {
            for (int a = 0; a < d1; ++a) {
                const std::ptrdiff_t si = lI(m, a);
                if (si >= i0 && si < i1) {
                    rows.push_back(m);
                    break;
                }
            }
        }

        // Stage 2: accumulate this band's wing contributions into Q. The
        // out-of-band wings are skipped here and picked up by the band that
        // owns them, so every (a, b) pair is summed exactly once.
        const std::ptrdiff_t n_rows = (std::ptrdiff_t)rows.size();
        #pragma omp parallel for schedule(static)
        for (std::ptrdiff_t r = 0; r < n_rows; ++r) {
            MW_CANCEL_POLL();
            const std::ptrdiff_t m = rows[(size_t)r];
            for (std::ptrdiff_t n = 0; n < nJ; ++n) {
                cd qmn = Qm(m, n);  // seeded, so the a-order is preserved
                for (int a = 0; a < d1; ++a) {
                    const std::ptrdiff_t si = lI(m, a);
                    if (si < i0 || si >= i1) continue;
                    const std::ptrdiff_t sl = si - i0;
                    for (int b = 0; b < d1; ++b) {
                        const std::ptrdiff_t sj = lJ(n, b);
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
}

}  // namespace somm_core
