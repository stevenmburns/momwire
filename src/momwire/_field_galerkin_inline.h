// The field-form Galerkin assembly's compute half (`bspline.
// _field_galerkin_block`'s inner two contractions, momwire#914 unit 2, the
// strided target #1115, the row-compact one #1132 and the mirror #1290):
// moved verbatim from `_accel_mw568.cpp`'s `assemble_field_galerkin`, whose
// binding keeps the argument checks and calls `mw_fg::assemble`, so bs2's
// C++ buried fill (momwire#1517 step 4) assembles through the same code.
// The derivation of the fusion and the threading rule are at the binding.
#pragma once

#include "_lanes.h"

#include <cmath>
#include <complex>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <utility>
#include <vector>

#if MW_LANES_SIMD && MW_LANES_FUSED && MW_LANES_PACKED
#define MW_FG_LANES 1
#else
#define MW_FG_LANES 0
#endif

namespace mw_fg {

typedef std::complex<double> cd_fg;
constexpr std::ptrdiff_t FG_BATCH = 4;
// h is built on the stack per column; q is a quadrature order (6 here), not
// a mesh dimension, so a fixed bound is honest rather than a limit anyone
// meets. Checked below.
constexpr std::ptrdiff_t FG_MAX_Q = 64;

// One accumulation target of the assembly: Q itself, or (momwire#1290) the
// symmetric route's mirror, Q.T with its own source positions. `cols` are the
// basis columns stage 2 visits.
struct FgTarget {
    cd_fg *Qp;
    std::ptrdiff_t sr, sc;
    const std::int64_t *ps;
    std::vector<std::int64_t> js_of;
    std::vector<double> hcol;
    std::vector<std::ptrdiff_t> cols;
};

// One batch of row-wings sharing an observer index: their basis rows, the Q
// rows they land in, and (in `fg_g`) their g q-vectors.
struct FgBatch {
    std::ptrdiff_t ic, nk;
    std::ptrdiff_t mb[FG_BATCH], mr[FG_BATCH];
};

// Stage 2 for one batch into one target: one basis column per thread, so
// every Q entry a thread writes is its own.
static void fg_stage2(FgTarget &T, const FgBatch &bt, const cd_fg *F,
                      std::ptrdiff_t nsq, std::ptrdiff_t q, std::ptrdiff_t A,
                      double scale) {
    const std::ptrdiff_t ncol = static_cast<std::ptrdiff_t>(T.cols.size());
    const std::ptrdiff_t nk = bt.nk;
    #pragma omp parallel for schedule(static)
    for (std::ptrdiff_t c = 0; c < ncol; ++c) {
        const std::ptrdiff_t n = T.cols[c];
        cd_fg s[FG_BATCH];
        for (std::ptrdiff_t k = 0; k < nk; ++k) s[k] = cd_fg(0.0, 0.0);
        for (std::ptrdiff_t b = 0; b < A; ++b) {
            const std::int64_t js = T.js_of[n * A + b];
            if (js < 0) continue;
            const double *h = &T.hcol[(n * A + b) * q];
            for (std::ptrdiff_t k = 0; k < nk; ++k) {
                const cd_fg *fp = &F[k * nsq + js * q];
                cd_fg t(0.0, 0.0);
                for (std::ptrdiff_t r = 0; r < q; ++r) t += fp[r] * h[r];
                s[k] += t;
            }
        }
        // `scale` multiplies each CONTRIBUTION as it lands, never Q and
        // never a factor. Never Q: this is one chunk of an observer loop,
        // and scaling the target would re-scale every chunk already in it.
        // Never a factor (g, h or the projected table): that would move the
        // rounding, and scale = 1.0 has to reproduce the unscaled numbers bit
        // for bit.
        for (std::ptrdiff_t k = 0; k < nk; ++k)
            T.Qp[bt.mr[k] * T.sr + n * T.sc] += scale * s[k];
    }
}

#if MW_FG_LANES
// Stage 1 for W/2 adjacent (j, r) entries, NK row-wings, (re, im) per lane
// pair: each lane is the scalar step `acc += g * v` -- the real product of
// each part, then the add, q in order from +0.0 -- on its own part.
template <int NK>
static inline void fg_stage1_lanes(const cd_fg *col, std::ptrdiff_t nsq,
                                   std::ptrdiff_t q, const double *g, cd_fg *F,
                                   std::ptrdiff_t jr) {
    using namespace mw_lanes;
    vd acc[NK];
    for (int k = 0; k < NK; ++k) acc[k] = zero();
    for (std::ptrdiff_t iq = 0; iq < q; ++iq) {
        const vd v = loadu(reinterpret_cast<const double *>(col + iq * nsq + jr));
        for (int k = 0; k < NK; ++k)
            acc[k] = add(acc[k], mul(set1(g[k * q + iq]), v));
    }
    for (int k = 0; k < NK; ++k)
        storeu(reinterpret_cast<double *>(F + k * nsq + jr), acc[k]);
}
#endif

// The validated arguments of one observer chunk: `proj` (nc q, ns q), the
// moment weights (P, nc, q) / (P, ns, q), the basis (nb, A) wings and their
// (nb, A, P) polynomials, the axis positions of every segment (`pos_o`,
// `pos_s`, n_seg), the target Q (element strides; `q_rows` rows, and
// `rof` the row-compact map or empty) and the mirror (Q2 / ps2, or null).
struct Args {
    const std::complex<double> *pj = nullptr;
    const double *wo = nullptr, *ws = nullptr, *pl = nullptr;
    const std::int64_t *ss = nullptr, *po = nullptr, *ps = nullptr;
    std::ptrdiff_t P = 0, nc = 0, q = 0, ns = 0, nb = 0, A = 0, i0 = 0, q_rows = 0;
    std::complex<double> *Qp = nullptr;
    std::ptrdiff_t sr = 0, sc = 0;
    std::vector<std::int64_t> rof;
    bool fused = true;
    double scale = 1.0;
    std::complex<double> *Q2 = nullptr;  // the mirror target, or null
    std::ptrdiff_t sr2 = 0, sc2 = 0;
    const std::int64_t *ps2 = nullptr;
    bool lanes = true;
};

// Throws std::invalid_argument where a row this chunk writes has no compact
// row (checked before any write).
inline void assemble(const Args &args) {
    typedef std::complex<double> cd;
    const std::complex<double> *pj = args.pj;
    const double *wo = args.wo, *ws = args.ws, *pl = args.pl;
    const std::int64_t *ss = args.ss, *po = args.po, *ps = args.ps;
    const std::ptrdiff_t P = args.P, nc = args.nc, q = args.q, ns = args.ns, nb = args.nb, A = args.A,
                         i0 = args.i0, q_rows = args.q_rows;
    const std::ptrdiff_t nsq = ns * q;
    cd *Qp = args.Qp;
    const std::ptrdiff_t sr = args.sr, sc = args.sc;
    const std::vector<std::int64_t> &rof = args.rof;
    const bool fused = args.fused, lanes = args.lanes;
    const double scale = args.scale;
    const bool has_mirror = args.Q2 != nullptr;
    (void)lanes;

    std::vector<FgTarget> tg(has_mirror ? 2 : 1);
    tg[0].Qp = Qp;
    tg[0].sr = sr;
    tg[0].sc = sc;
    tg[0].ps = ps;
    if (has_mirror) {
        tg[1].Qp = args.Q2;
        tg[1].sr = args.sr2;
        tg[1].sc = args.sc2;
        tg[1].ps = args.ps2;
    }
    // The column wings, once per target: their source index and their
    // h-vector. js < 0 is a wing belonging to the other medium, which drops
    // out of the block rather than being clamped into it.
    for (auto &t : tg) {
        t.js_of.assign(static_cast<size_t>(nb) * A, 0);
        t.hcol.assign(static_cast<size_t>(nb) * A * q, 0.0);
    }
    // Row-wings whose observer segment is in this chunk, bucketed by local
    // observer index so one pass over `proj` serves every one of them.
    std::vector<std::vector<std::pair<std::ptrdiff_t, std::ptrdiff_t>>> at(nc);
    for (std::ptrdiff_t a = 0; a < A; ++a)
        for (std::ptrdiff_t m = 0; m < nb; ++m) {
            const std::int64_t io = po[ss[m * A + a]];
            if (io >= i0 && io < i0 + nc) {
                // A row this chunk writes must have a compact row to land in:
                // checked before any write, not trusted. The
                // one exception is a PADDED wing -- supp_seg is zero-padded,
                // so an unlive slot names segment 0 and lands in whichever
                // chunk holds it, contributing an exact zero. On the square
                // target that zero is added to a row nobody reads; a compact
                // target has no such row, so the wing is dropped. Dropping it
                // cannot move a held row: each slot's sum is its own, and a
                // held row's adds keep their order.
                if (!rof.empty() && (rof[m] < 0 || rof[m] >= q_rows)) {
                    bool live = false;
                    for (std::ptrdiff_t p = 0; p < P; ++p)
                        if (pl[(m * A + a) * P + p] != 0.0) live = true;
                    if (!live && rof[m] < 0) continue;
                    throw std::invalid_argument(
                        "row_of maps a basis this chunk writes outside the "
                        "compact Q");
                }
                at[io - i0].emplace_back(m, a);
            }
        }

    std::vector<cd> F, T, Jc;
    if (!fused) {
        T.assign(static_cast<size_t>(P) * nc * nsq, cd(0.0, 0.0));
        Jc.assign(static_cast<size_t>(P) * P * nc * ns, cd(0.0, 0.0));
    }
    // The fused route's batches, formed up front (they read only the
    // inputs): row-wings of one observer index, at most FG_BATCH of them and
    // never one basis row twice, each with its g q-vectors in `gb`.
    std::vector<FgBatch> batches;
    std::vector<double> gb;
    if (fused) {
        for (std::ptrdiff_t ic = 0; ic < nc; ++ic) {
            const auto &bucket = at[ic];
            std::ptrdiff_t taken = 0;
            while (taken < static_cast<std::ptrdiff_t>(bucket.size())) {
                FgBatch bt;
                bt.ic = ic;
                std::ptrdiff_t nk = 0;
                const size_t g0 = gb.size();
                gb.resize(g0 + static_cast<size_t>(FG_BATCH) * q, 0.0);
                while (taken < static_cast<std::ptrdiff_t>(bucket.size()) &&
                       nk < FG_BATCH) {
                    const std::ptrdiff_t m = bucket[taken].first;
                    const std::ptrdiff_t a = bucket[taken].second;
                    bool dup = false;
                    for (std::ptrdiff_t u = 0; u < nk; ++u)
                        if (bt.mb[u] == m) dup = true;
                    if (dup) break;
                    bt.mb[nk] = m;
                    bt.mr[nk] = rof.empty() ? m : static_cast<std::ptrdiff_t>(rof[m]);
                    for (std::ptrdiff_t iq = 0; iq < q; ++iq) {
                        double acc = 0.0;
                        for (std::ptrdiff_t p = 0; p < P; ++p)
                            acc += pl[(m * A + a) * P + p] * wo[(p * nc + ic) * q + iq];
                        gb[g0 + nk * q + iq] = acc;
                    }
                    ++nk;
                    ++taken;
                }
                bt.nk = nk;
                batches.push_back(bt);
            }
        }
        // One F per batch when a mirror will read it again, else one reused.
        const size_t n_f = has_mirror ? batches.size() : 1;
        F.assign(n_f * static_cast<size_t>(FG_BATCH) * nsq, cd(0.0, 0.0));
    }
    // momwire#1290: stage 2 visits only the columns with a live wing, when
    // that is exact. A column whose every wing drops out contributes
    // `scale * (+0, +0)`, and for a scale whose sign bit is set that is
    // (-0, -0), which adds to any double as the identity -- so skipping the
    // add moves no bit, and saves a strided write per dead column (half the
    // columns, on average, of the symmetric route's chunks). For a positive
    // scale the add of +0 would turn a -0 entry into +0, so every column is
    // still visited.
    const bool skip_dead = lanes && std::signbit(scale);
#if MW_FG_LANES
    const bool use_lanes = lanes;
#endif

    {
        for (auto &t : tg) {
            std::int64_t *js_of = t.js_of.data();
            double *hcol = t.hcol.data();
            const std::int64_t *pss = t.ps;
            #pragma omp parallel for schedule(static)
            for (std::ptrdiff_t e = 0; e < nb * A; ++e) {
                const std::int64_t js = pss[ss[e]];
                js_of[e] = js;
                if (js < 0) continue;
                for (std::ptrdiff_t r = 0; r < q; ++r) {
                    double acc = 0.0;
                    for (std::ptrdiff_t p = 0; p < P; ++p)
                        acc += pl[e * P + p] * ws[(p * ns + js) * q + r];
                    hcol[e * q + r] = acc;
                }
            }
            t.cols.reserve(nb);
            for (std::ptrdiff_t n = 0; n < nb; ++n) {
                bool live = !skip_dead;
                for (std::ptrdiff_t b = 0; b < A && !live; ++b)
                    live = js_of[n * A + b] >= 0;
                if (live) t.cols.push_back(n);
            }
        }

        if (fused) {
            const std::ptrdiff_t nbt = static_cast<std::ptrdiff_t>(batches.size());
            for (std::ptrdiff_t bi = 0; bi < nbt; ++bi) {
                const FgBatch &bt = batches[bi];
                const std::ptrdiff_t ic = bt.ic, nk = bt.nk;
                const double *g = &gb[static_cast<size_t>(bi) * FG_BATCH * q];
                cd *Fb = &F[(has_mirror ? static_cast<size_t>(bi) : 0) *
                            FG_BATCH * nsq];
                const cd *col = pj + ic * q * nsq;
                // Stage 1: F[k, j, r] = sum_q g[k,q] * proj[ic*q+q, j*q+r]
#if MW_FG_LANES
                if (use_lanes) {
                    constexpr std::ptrdiff_t CPV = mw_lanes::W / 2;
                    const std::ptrdiff_t ngrp = nsq / CPV;
                    #pragma omp parallel for schedule(static)
                    for (std::ptrdiff_t jp = 0; jp < ngrp; ++jp) {
                        const std::ptrdiff_t jr = CPV * jp;
                        switch (nk) {
                            case 1: fg_stage1_lanes<1>(col, nsq, q, g, Fb, jr); break;
                            case 2: fg_stage1_lanes<2>(col, nsq, q, g, Fb, jr); break;
                            case 3: fg_stage1_lanes<3>(col, nsq, q, g, Fb, jr); break;
                            default: fg_stage1_lanes<4>(col, nsq, q, g, Fb, jr); break;
                        }
                    }
                    for (std::ptrdiff_t jr = ngrp * CPV; jr < nsq; ++jr) {
                        cd acc[FG_BATCH];
                        for (std::ptrdiff_t k = 0; k < nk; ++k) acc[k] = cd(0.0, 0.0);
                        for (std::ptrdiff_t iq = 0; iq < q; ++iq) {
                            const cd v = col[iq * nsq + jr];
                            for (std::ptrdiff_t k = 0; k < nk; ++k)
                                acc[k] += g[k * q + iq] * v;
                        }
                        for (std::ptrdiff_t k = 0; k < nk; ++k) Fb[k * nsq + jr] = acc[k];
                    }
                } else
#endif
                {
                    #pragma omp parallel for schedule(static)
                    for (std::ptrdiff_t jr = 0; jr < nsq; ++jr) {
                        cd acc[FG_BATCH];
                        for (std::ptrdiff_t k = 0; k < nk; ++k) acc[k] = cd(0.0, 0.0);
                        for (std::ptrdiff_t iq = 0; iq < q; ++iq) {
                            const cd v = col[iq * nsq + jr];
                            for (std::ptrdiff_t k = 0; k < nk; ++k)
                                acc[k] += g[k * q + iq] * v;
                        }
                        for (std::ptrdiff_t k = 0; k < nk; ++k) Fb[k * nsq + jr] = acc[k];
                    }
                }
                fg_stage2(tg[0], bt, Fb, nsq, q, A, scale);
            }
            if (has_mirror)
                for (std::ptrdiff_t bi = 0; bi < nbt; ++bi)
                    fg_stage2(tg[1], batches[bi],
                              &F[static_cast<size_t>(bi) * FG_BATCH * nsq], nsq,
                              q, A, scale);
        } else {
            // The literal transcription, Jc materialised: the alternative the
            // fused route above was measured against.
            const std::int64_t *js_of = tg[0].js_of.data();
            const std::int64_t *rp = rof.empty() ? nullptr : rof.data();
            #pragma omp parallel for collapse(2) schedule(static)
            for (std::ptrdiff_t p = 0; p < P; ++p)
                for (std::ptrdiff_t ic = 0; ic < nc; ++ic) {
                    cd *out = &T[(p * nc + ic) * nsq];
                    for (std::ptrdiff_t iq = 0; iq < q; ++iq) {
                        const double w = wo[(p * nc + ic) * q + iq];
                        const cd *row = &pj[(ic * q + iq) * nsq];
                        for (std::ptrdiff_t jr = 0; jr < nsq; ++jr) out[jr] += w * row[jr];
                    }
                }

            #pragma omp parallel for collapse(2) schedule(static)
            for (std::ptrdiff_t p = 0; p < P; ++p)
                for (std::ptrdiff_t Pp = 0; Pp < P; ++Pp)
                    for (std::ptrdiff_t ic = 0; ic < nc; ++ic)
                        for (std::ptrdiff_t j = 0; j < ns; ++j) {
                            const cd *tp = &T[(p * nc + ic) * nsq + j * q];
                            cd acc(0.0, 0.0);
                            for (std::ptrdiff_t r = 0; r < q; ++r)
                                acc += tp[r] * ws[(Pp * ns + j) * q + r];
                            Jc[((p * P + Pp) * nc + ic) * ns + j] = acc;
                        }

            for (std::ptrdiff_t ic = 0; ic < nc; ++ic)
                for (const auto &ma : at[ic]) {
                    const std::ptrdiff_t m = ma.first, a = ma.second;
                    const std::ptrdiff_t mrow = rp ? static_cast<std::ptrdiff_t>(rp[m]) : m;
                    #pragma omp parallel for schedule(static)
                    for (std::ptrdiff_t n = 0; n < nb; ++n) {
                        cd s(0.0, 0.0);
                        for (std::ptrdiff_t b = 0; b < A; ++b) {
                            const std::int64_t js = js_of[n * A + b];
                            if (js < 0) continue;
                            for (std::ptrdiff_t p = 0; p < P; ++p) {
                                const double cpa = pl[(m * A + a) * P + p];
                                for (std::ptrdiff_t Pp = 0; Pp < P; ++Pp)
                                    s += cpa * pl[(n * A + b) * P + Pp] *
                                         Jc[((p * P + Pp) * nc + ic) * ns + js];
                            }
                        }
                        // The same per-contribution scaling as the fused
                        // route's stage 2. The two routes are gated against
                        // each other, so they cannot part on where it lands.
                        Qp[mrow * sr + n * sc] += scale * s;
                    }
                }
        }
    }
}

}  // namespace mw_fg
