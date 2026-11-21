// razor2p's crossing blocks in C++: see `_razor_crossing_core.h`. Every
// step names the `_crossing_fill.py` / razor.py function it replays.
#include "_razor_crossing_core.h"
#include "_cpu_count.h"
#include "_crossing_shared_inline.h"
#include "_fma_inline.h"

#include "_near_interface_eval.h"
#include "_somm_below_grid_core.h"
#include "_somm_grid_core.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <map>
#include <utility>
#include <set>
#include <stdexcept>
#include <mutex>
#include <thread>

#if defined(_OPENMP)
#include <omp.h>
#endif

namespace mw {
namespace razor {

namespace {

using ni_eval::Six;
using namespace mw::crossing;
using mw::crossing::End;  // the axis end record, not the front's wire end

[[noreturn]] void raise_value(const std::string &m) {
    throw FrontError(FrontErrorKind::value_error, m, true);
}

// The shared pieces' ValueError: razor's FrontError.
struct RazorPolicy {
    [[noreturn]] static void raise(const std::string &m) { raise_value(m); }
};
double on_plane_side(double z, bool above, const char *what) { return crossing::on_plane_side<RazorPolicy>(z, above, what); }

// The crossing context's medium (`buried_medium`) and constants.
struct Ctx {
    const Front *f;
    double gz, omega, mu, eps;
    cd eps_t, eps_m, k_m, c2, a_m;
    double k_p;
    cd c1;
    // `AxisGeometry`: seg_l = seg_p0, seg_r = seg_p0 + h t (full geometry).
    std::vector<double> seg_r;
    // `BasisPolynomials`: per basis and wing, (c0, c1).
    std::vector<double> polys;  // (n_basis, 2, 2)
};

Ctx make_ctx(const Front &f) {
    Ctx c;
    c.f = &f;
    c.gz = f.ground_z;
    c.omega = f.omega;
    c.mu = MU;
    c.eps = EPS;
    c.k_p = f.k;
    // `_ground_refl.eps_tilde`, `buried_medium`: CPython complex arithmetic.
    c.eps_t = cd(f.ground_eps_r_, (-f.ground_sigma_) / (f.omega * EPS));
    c.eps_m = py_prod(cd(EPS, 0.0), c.eps_t);
    c.k_m = somm_below::k_medium(c.eps_t, f.k);
    c.c2 = somm_core::py_quot(cd(c.eps_t.real() - 1.0, c.eps_t.imag() - 0.0),
                              cd(c.eps_t.real() + 1.0, c.eps_t.imag() + 0.0));
    c.a_m = somm_core::py_quot(cd(1.0 - c.eps_t.real(), 0.0 - c.eps_t.imag()),
                               cd(1.0 + c.eps_t.real(), 0.0 + c.eps_t.imag()));
    // `_c1_moment`: -1j * omega * mu / (4.0 * pi)
    const cd m1j(-0.0, -1.0);
    const cd a = py_prod(py_prod(m1j, cd(f.omega, 0.0)), cd(MU, 0.0));
    c.c1 = somm_core::py_quot(a, cd(4.0 * M_PI, 0.0));
    const size_t ns = static_cast<size_t>(f.n_seg);
    c.seg_r.resize(3 * ns);
    for (size_t g = 0; g < ns; g++)
        for (int k = 0; k < 3; k++) c.seg_r[3 * g + k] = f.seg_p0[3 * g + k] + f.seg_h[g] * f.seg_t[3 * g + k];
    const size_t n = static_cast<size_t>(f.n_basis);
    c.polys.assign(4 * n, 0.0);
    for (size_t m = 0; m < n; m++)
        for (int j = 0; j < 2; j++) {
            const int64_t seg = f.wing_seg[2 * m + j];
            const double h = f.seg_h[static_cast<size_t>(seg)];
            const double sig = f.wing_sigma[2 * m + j];
            if (f.wing_rise.p[2 * m + j]) {
                c.polys[4 * m + 2 * j] = 0.0;
                c.polys[4 * m + 2 * j + 1] = sig / h;
            } else {
                c.polys[4 * m + 2 * j] = sig;
                c.polys[4 * m + 2 * j + 1] = (-sig) / h;
            }
        }
    return c;
}

// ---------------------------------------------------------------------------
// Axes

// An end's nonzero basis values (what `End::fv` holds densely), ascending
// row order: razor's axes keep them here and leave `fv` empty (momwire#1541:
// a path-tested axis carries two ends per basis, and two dense n_basis rows
// per basis were 16 n_basis^2 bytes, a whole Z, on the below axis).
struct EndNz {
    std::vector<int64_t> idx;
    std::vector<double> val;
};
// An axis with its ends' values sparse, `nz[e]` for `ends[e]`.
struct RAxis : Axis {
    std::vector<EndNz> nz;
};

// `_path_test_rows` + `path_test_axis`: one record per row -- its path's
// points (the two wing centroids, or one half with the knot), tangents,
// weights, and the T2 endpoints.
struct Rec {
    int64_t m;
    std::vector<double> pts, tans, wts;
    std::vector<int64_t> seg;
    double before[3], after[3];
};

RAxis path_test_axis(int64_t n_basis, const std::vector<Rec> &recs) {
    RAxis ax;
    ax.n_basis = n_basis;
    ax.path_tested = true;
    std::vector<std::tuple<int64_t, int64_t, double>> trip;
    int64_t off = 0;
    for (const Rec &r : recs) {
        const size_t q = r.wts.size();
        ax.nodes.insert(ax.nodes.end(), r.pts.begin(), r.pts.end());
        ax.t.insert(ax.t.end(), r.tans.begin(), r.tans.end());
        ax.w.insert(ax.w.end(), r.wts.begin(), r.wts.end());
        for (size_t k = 0; k < q; k++) trip.emplace_back(r.m, off + static_cast<int64_t>(k), 1.0);
        off += static_cast<int64_t>(q);
        ax.segof.insert(ax.segof.end(), r.seg.begin(), r.seg.end());
        // Each end's fv is one 1.0 at the record's own row: kept sparse
        // (`EndNz`), not as two dense n_basis rows per record (momwire#1541).
        End b, a;
        std::copy(r.before, r.before + 3, b.pt);
        b.sign = -1.0;
        std::copy(r.after, r.after + 3, a.pt);
        a.sign = +1.0;
        ax.ends.push_back(b);
        ax.ends.push_back(a);
        EndNz nz;
        nz.idx.push_back(r.m);
        nz.val.push_back(1.0);
        ax.nz.push_back(nz);
        ax.nz.push_back(nz);
    }
    ax.F = to_csr(n_basis, off, trip);
    ax.Fd = to_csr(n_basis, off, {});
    return ax;
}

// `_crossing_path_axis(geom, tents, side)`: the medium's own tents whole,
// each crossing tent's live half (the knot its shared endpoint).
RAxis crossing_path_axis(const Front &f, const std::vector<std::pair<int64_t, int>> &tents, bool below) {
    std::vector<char> mine(static_cast<size_t>(f.n_seg), 0);
    for (int64_t w = 0; w < f.n_wires; w++)
        if ((f.media_below[static_cast<size_t>(w)] != 0) == below)
            for (int64_t g = f.seg_offsets[w]; g < f.seg_offsets[w + 1]; g++) mine[static_cast<size_t>(g)] = 1;
    std::set<int64_t> crossing;
    for (const auto &t : tents) crossing.insert(t.first);
    auto cent = [&](int64_t s, double *o) {
        const double half = 0.5 * f.seg_h[static_cast<size_t>(s)];
        for (int k = 0; k < 3; k++) o[k] = f.seg_p0[3 * s + k] + half * f.seg_t[3 * s + k];
    };
    std::vector<Rec> recs;
    for (int64_t m = 0; m < f.n_basis; m++) {
        const int64_t sa = f.wing_seg[2 * m], sb = f.wing_seg[2 * m + 1];
        int halves = -1;  // -1 none, 0 both, 1 A, 2 B
        if (crossing.count(m)) {
            const int alive = mine[static_cast<size_t>(sa)] ? 0 : 1;
            halves = alive == 0 ? 1 : 2;
        } else if (mine[static_cast<size_t>(sa)]) {
            halves = 0;
        }
        if (halves < 0) continue;
        Rec r;
        r.m = m;
        double ca[3], cb[3], knot[3];
        cent(sa, ca);
        cent(sb, cb);
        {
            const double along = f.wing_rise.p[2 * m] ? f.seg_h[static_cast<size_t>(sa)] : 0.0;
            for (int k = 0; k < 3; k++) knot[k] = f.seg_p0[3 * sa + k] + along * f.seg_t[3 * sa + k];
        }
        const double ga = f.wing_sigma[2 * m], gb = f.wing_sigma[2 * m + 1];
        auto push = [&](const double *p, double g, int64_t s) {
            r.pts.insert(r.pts.end(), p, p + 3);
            for (int k = 0; k < 3; k++) r.tans.push_back(g * f.seg_t[3 * s + k]);
            r.wts.push_back(0.5 * f.seg_h[static_cast<size_t>(s)]);
            r.seg.push_back(s);
        };
        if (halves == 0) {
            push(ca, ga, sa);
            push(cb, gb, sb);
            std::copy(ca, ca + 3, r.before);
            std::copy(cb, cb + 3, r.after);
        } else if (halves == 1) {
            push(ca, ga, sa);
            std::copy(ca, ca + 3, r.before);
            std::copy(knot, knot + 3, r.after);
        } else {
            push(cb, gb, sb);
            std::copy(knot, knot + 3, r.before);
            std::copy(cb, cb + 3, r.after);
        }
        recs.push_back(std::move(r));
    }
    return path_test_axis(f.n_basis, recs);
}

// `axis_data(ctx, seg_idx, grade_near_plane=True, growth=2.0, panel_order=8,
// q=12)` with razor's tent polynomials.
RAxis axis_data(const Ctx &c, const std::vector<int64_t> &seg_idx, double a_wire) {
    const Front &f = *c.f;
    RAxis ax;
    ax.n_basis = f.n_basis;
    std::vector<double> xg, wg, gx, gw;
    leggauss(12, xg, wg);
    leggauss(8, gx, gw);
    const double growth = 2.0;
    std::vector<double> u_phys;
    std::map<int64_t, std::pair<int64_t, int64_t>> runs;  // seg -> (start, count)
    for (int64_t g : seg_idx) {
        const double *sl = &f.seg_p0[3 * g];
        const double *sr = &c.seg_r[3 * g];
        const double h = f.seg_h[static_cast<size_t>(g)];
        const double *tang = &f.seg_t[3 * g];
        const bool touch_lo = std::fabs(sl[2] - c.gz) < PLANE_TOL;
        const bool touch_hi = std::fabs(sr[2] - c.gz) < PLANE_TOL;
        const double d_lo = std::fabs(sl[2] - c.gz), d_hi = std::fabs(sr[2] - c.gz);
        std::vector<double> u, w;
        if (touch_lo || touch_hi) {
            graded_u(h, !touch_lo, a_wire, growth, gx, gw, u, w);
        } else if (std::max(d_lo, d_hi) > 2.0 * std::min(d_lo, d_hi)) {
            const double d = std::min(d_lo, d_hi);
            graded_u(h, !(d_lo <= d_hi), std::max(d, a_wire), growth, gx, gw, u, w);
        } else {
            for (size_t k = 0; k < xg.size(); k++) {
                const double tq = 0.5 * (xg[k] + 1.0);
                u.push_back(h * tq);
                w.push_back(0.5 * h * wg[k]);
            }
        }
        runs[g] = {static_cast<int64_t>(ax.w.size()), static_cast<int64_t>(u.size())};
        for (size_t k = 0; k < u.size(); k++) {
            const double s = u[k] / h;
            for (int q = 0; q < 3; q++) ax.nodes.push_back(sl[q] + s * (sr[q] - sl[q]));
            for (int q = 0; q < 3; q++) ax.t.push_back(tang[q]);
            ax.w.push_back(w[k]);
            ax.segof.push_back(g);
            u_phys.push_back(u[k]);
        }
    }
    // `_basis_samples`: every live wing's polynomial and derivative at its
    // segment's nodes; `seg_rows`.
    std::vector<std::tuple<int64_t, int64_t, double>> tf, tfd;
    for (int64_t m = 0; m < f.n_basis; m++)
        for (int j = 0; j < 2; j++) {
            const double c0 = c.polys[static_cast<size_t>(4 * m + 2 * j)];
            const double c1 = c.polys[static_cast<size_t>(4 * m + 2 * j + 1)];
            if (!(c0 != 0.0 || c1 != 0.0)) continue;
            const int64_t seg = f.wing_seg[2 * m + j];
            auto it = runs.find(seg);
            if (it == runs.end()) continue;
            for (int64_t k = 0; k < it->second.second; k++) {
                const int64_t node = it->second.first + k;
                const double u = u_phys[static_cast<size_t>(node)];
                double fv = 0.0;
                fv += c0 * 1.0;
                fv += c1 * u;
                double fdv = 0.0;
                fdv += (1.0 * c1) * 1.0;
                tf.emplace_back(m, node, fv);
                tfd.emplace_back(m, node, fdv);
            }
            ax.seg_rows[seg].push_back(m);
        }
    for (auto &kv : ax.seg_rows) {
        std::sort(kv.second.begin(), kv.second.end());
        kv.second.erase(std::unique(kv.second.begin(), kv.second.end()), kv.second.end());
    }
    const int64_t n_nodes = static_cast<int64_t>(ax.w.size());
    ax.F = to_csr(f.n_basis, n_nodes, tf);
    ax.Fd = to_csr(f.n_basis, n_nodes, tfd);
    // The ends table (`_end_values`): every wire whose first segment is on
    // the axis, its two ends where some basis is nonzero.
    std::set<int64_t> on_axis(seg_idx.begin(), seg_idx.end());
    for (int64_t w = 0; w < f.n_wires; w++) {
        const int64_t first = f.seg_offsets[w], last = f.seg_offsets[w + 1] - 1;
        if (!on_axis.count(first)) continue;
        for (int pass = 0; pass < 2; pass++) {
            const int64_t gseg = pass == 0 ? first : last;
            const double hh = f.seg_h[static_cast<size_t>(gseg)];
            const double u = pass == 0 ? 0.0 : hh;
            End e;
            e.sign = pass == 0 ? -1.0 : 1.0;
            for (int q = 0; q < 3; q++)
                e.pt[q] = f.seg_p0[3 * gseg + q] + (u / hh) * (c.seg_r[3 * gseg + q] - f.seg_p0[3 * gseg + q]);
            e.fv.assign(static_cast<size_t>(f.n_basis), 0.0);
            bool any = false;
            for (int j = 0; j < 2; j++)
                for (int64_t m = 0; m < f.n_basis; m++) {
                    if (f.wing_seg[2 * m + j] != gseg) continue;
                    const double c0 = c.polys[static_cast<size_t>(4 * m + 2 * j)];
                    const double c1 = c.polys[static_cast<size_t>(4 * m + 2 * j + 1)];
                    if (!(c0 != 0.0 || c1 != 0.0)) continue;
                    double acc = 0.0;
                    acc = acc + c0 * 1.0;
                    acc = acc + c1 * u;
                    e.fv[static_cast<size_t>(m)] += acc;
                }
            for (double v : e.fv) any |= v != 0.0;
            if (any) {
                // Kept sparse (`EndNz`): the entries the folds read, in
                // ascending row order, each `fv != 0.0`.
                EndNz nz;
                for (size_t m = 0; m < e.fv.size(); m++)
                    if (e.fv[m] != 0.0) {
                        nz.idx.push_back(static_cast<int64_t>(m));
                        nz.val.push_back(e.fv[m]);
                    }
                std::vector<double>().swap(e.fv);
                ax.ends.push_back(std::move(e));
                ax.nz.push_back(std::move(nz));
            }
        }
    }
    return ax;
}

// One cross block, compact (momwire#1541): only the entries the fold
// `Z[rows, cols] -= t` reads, row-major (|rows| x |cols|). `rpos` / `cpos`
// map a basis to its local row / column (-1: not folded). Every kept entry
// sees the operations the whole (n x n) block gave it, in the same order;
// the rest were computed and dropped, so leaving them out moves no bit.
struct Block {
    std::vector<int64_t> rpos, cpos;
    std::vector<int64_t> rows, cols;  // local index -> basis
    std::vector<cd> v;
    void shape(int64_t n_basis, const std::vector<int64_t> &r, const std::vector<int64_t> &cc) {
        rows = r;
        cols = cc;
        rpos.assign(static_cast<size_t>(n_basis), -1);
        cpos.assign(static_cast<size_t>(n_basis), -1);
        for (size_t i = 0; i < rows.size(); i++) rpos[static_cast<size_t>(rows[i])] = static_cast<int64_t>(i);
        for (size_t j = 0; j < cols.size(); j++) cpos[static_cast<size_t>(cols[j])] = static_cast<int64_t>(j);
        v.assign(rows.size() * cols.size(), cd(0.0, 0.0));
    }
    cd *at(int64_t r_basis, int64_t c_basis) {
        const int64_t r = rpos[static_cast<size_t>(r_basis)], q = cpos[static_cast<size_t>(c_basis)];
        if (r < 0 || q < 0) return nullptr;
        return &v[static_cast<size_t>(r) * cols.size() + static_cast<size_t>(q)];
    }
};

// The main sandwich's node pairs per window (momwire#1541): the grid tables
// are formed a window of A's nodes at a time, never over the whole grid.
constexpr size_t SANDWICH_WINDOW_PAIRS = size_t(1) << 21;
// Up to this many node pairs, pass 1 keeps every pair's memo index (4 bytes
// a pair, 16 MB at the cap) and pass 2 reads them instead of forming each
// pair's triple again and finding it (momwire#1956: the second `pair_row`
// and lookup were about half the block's bookkeeping). Over it, pass 2
// re-finds them a window at a time, as before.
constexpr size_t SANDWICH_KEEP_PAIRS = size_t(1) << 22;

// One (A node, B node) pair's table row, as the whole-grid fill formed it.
inline void pair_row(const Ctx &c, double a_wire, const Axis &A, const Axis &B, size_t a, size_t b,
                     double *row) {
    const double rho = std::hypot(A.nodes[3 * a] - B.nodes[3 * b], A.nodes[3 * a + 1] - B.nodes[3 * b + 1]);
    row[0] = std::hypot(rho, a_wire);
    row[1] = A.nodes[3 * a + 2] - c.gz;
    row[2] = B.nodes[3 * b + 2] - c.gz;
}

// `designed_index`'s claim over one window of the grid's rows (labels off),
// continued across windows: `rows` gathers the fresh distinct triples in
// first-appearance order over every window so far (the order one call over
// the whole grid claims them), each at index `base` + its place there, and
// `nan_rows` is the call's NaN dedup. Nothing is evaluated here:
// `claim_finish` evaluates the whole list as the one call would, because a
// triple's value can depend on which others share its exact-rho column
// (`_chunked_tables`' argument), so the evaluation must see the same set.
//
// `idx` (when given, n entries): each point's index, the one the claim
// stored or found for its triple, which is what `Memo::find` (then the NaN
// dedup's `find`) answers for it afterwards: the evaluation only fills
// values, so no key moves (momwire#1956).
void claim_window(BlockEval &be, uint32_t base, const double *re, const double *z, const double *zp,
                  size_t n, ni_eval::TripleIndex &nan_rows, std::vector<double> &rows,
                  uint32_t *idx = nullptr) {
    using ni_eval::TripleIndex;
    TripleIndex &keys = be.memo.keys();
    auto claim = [&](size_t i) {
        const double row[3] = {re[i], z[i], zp[i]};
        const uint32_t next = base + static_cast<uint32_t>(rows.size() / 3);
        bool ins = false;
        const bool nan = std::isnan(row[0]) || std::isnan(row[1]) || std::isnan(row[2]);
        const uint32_t v = nan ? nan_rows.find_or_insert(row, next, ins) : keys.find_or_insert(row, next, ins);
        if (ins) rows.insert(rows.end(), row, row + 3);
        return v;
    };
    int nt = 1;
#if defined(_OPENMP)
    nt = team(be, n);
    nt = std::min(nt, omp_get_max_threads());
    if (n < 16384) nt = 1;
    if (nt > 1) {
        // As `designed_index`: each chunk's points deduplicated on their own,
        // then claimed in chunk order at their first appearance, which is the
        // serial walk's order. No memo lookup first (momwire#1541): a triple
        // an earlier window claimed is found by the claim itself, once per
        // chunk-distinct triple rather than once per point.
        // With `idx`, each point first holds its chunk-local index, then
        // the index its chunk's first appearance claimed.
        std::vector<std::vector<size_t>> first(static_cast<size_t>(nt));
        #pragma omp parallel for schedule(static, 1) num_threads(nt)
        for (int ch = 0; ch < nt; ch++) {
            const size_t i0 = n * static_cast<size_t>(ch) / static_cast<size_t>(nt);
            const size_t i1 = n * static_cast<size_t>(ch + 1) / static_cast<size_t>(nt);
            TripleIndex li;
            li.reserve((i1 - i0) / 8);
            std::vector<size_t> &fc = first[static_cast<size_t>(ch)];
            for (size_t i = i0; i < i1; i++) {
                const double row[3] = {re[i], z[i], zp[i]};
                bool ins = false;
                const uint32_t local = li.find_or_insert(row, static_cast<uint32_t>(fc.size()), ins);
                if (ins) fc.push_back(i);
                if (idx) idx[i] = local;
            }
        }
        std::vector<std::vector<uint32_t>> claimed(static_cast<size_t>(nt));
        for (int ch = 0; ch < nt; ch++) {
            std::vector<uint32_t> &cl = claimed[static_cast<size_t>(ch)];
            cl.reserve(first[static_cast<size_t>(ch)].size());
            for (size_t i : first[static_cast<size_t>(ch)]) cl.push_back(claim(i));
        }
        if (idx) {
            #pragma omp parallel for schedule(static, 1) num_threads(nt)
            for (int ch = 0; ch < nt; ch++) {
                const size_t i0 = n * static_cast<size_t>(ch) / static_cast<size_t>(nt);
                const size_t i1 = n * static_cast<size_t>(ch + 1) / static_cast<size_t>(nt);
                const std::vector<uint32_t> &cl = claimed[static_cast<size_t>(ch)];
                for (size_t i = i0; i < i1; i++) idx[i] = cl[idx[i]];
            }
        }
        return;
    }
#else
    (void)nt;
#endif
    for (size_t i = 0; i < n; i++) {
        const uint32_t v = claim(i);
        if (idx) idx[i] = v;
    }
}

// Pass 1's claim over the WHOLE grid by its structure (momwire#1956), when
// that is cheap: `claim_window` over every window, the same calls in the same
// order, without forming or hashing every pair's triple.
//
// A pair's triple is (hypot(rho, a), z, z'), rho from the two nodes' (x, y):
// so A nodes sharing (x, y) (a vertical wire's) share every pair's rho, and
// the triple is a function of (A's z class, the key (re, z') of A's (x, y)
// group against B's node). The keys are found once per (group, B node), the
// z classes once per A node, and the walk over the grid in row-major order
// reads a dense table of (z class, key) instead of a hash. A table entry is
// claimed at its first appearance, from that pair's own row, so the memo
// sees `claim_window`'s find_or_insert of each distinct triple's first pair,
// in first-appearance order, and nothing else: the same keys, indices,
// fresh rows and NaN dedup, bit for bit. The z classes and the keys compare
// as the memo does (each component's `x + 0.0` bits), and a group's members
// differ at most in the sign of a zero coordinate, which `hypot` does not
// see, so a (z class, key) entry is exactly one of the memo's triples.
//
// False (nothing claimed) when the structure does not pay: fewer than two A
// nodes a group, or a table over SANDWICH_KEEP_PAIRS entries; the caller then
// claims window by window. `idx` as `claim_window`'s, over the whole grid.
bool claim_grouped(const Ctx &c, BlockEval &be, double a_wire, const Axis &A, const Axis &B, uint32_t base,
                   ni_eval::TripleIndex &nan_rows, std::vector<double> &rows, uint32_t *idx) {
    using ni_eval::TripleIndex;
    const size_t nA = A.n_nodes(), nB = B.n_nodes();
    if (nA == 0 || nB == 0) return false;
    // A's (x, y) groups (first node of each) and z classes, keyed as the
    // memo keys its components.
    std::vector<uint32_t> ga(nA), za(nA);
    std::vector<size_t> grep;
    size_t n_z = 0;
    {
        TripleIndex gi, zi;
        for (size_t a = 0; a < nA; a++) {
            bool ins = false;
            const double kxy[3] = {A.nodes[3 * a], A.nodes[3 * a + 1], 0.0};
            ga[a] = gi.find_or_insert(kxy, static_cast<uint32_t>(grep.size()), ins);
            if (ins) grep.push_back(a);
            const double kz[3] = {A.nodes[3 * a + 2] - c.gz, 0.0, 0.0};
            za[a] = zi.find_or_insert(kz, static_cast<uint32_t>(n_z), ins);
            if (ins) n_z++;
        }
    }
    const size_t G = grep.size();
    if (2 * G > nA || G * nB > SANDWICH_KEEP_PAIRS) return false;
    // Each (group, B node)'s key (re, z'), as `pair_row` forms them.
    std::vector<uint32_t> kid(G * nB);
    size_t n_k = 0;
    {
        std::vector<double> re(G * nB);
        const std::ptrdiff_t ng = static_cast<std::ptrdiff_t>(G);
        const int nt = team(be, G * nB);
#if defined(_OPENMP)
        #pragma omp parallel for schedule(static) num_threads(nt) if (nt > 1)
#else
        (void)nt;
#endif
        for (std::ptrdiff_t g = 0; g < ng; g++) {
            const size_t a = grep[static_cast<size_t>(g)];
            for (size_t b = 0; b < nB; b++) {
                double row[3];
                pair_row(c, a_wire, A, B, a, b, row);
                re[static_cast<size_t>(g) * nB + b] = row[0];
            }
        }
        TripleIndex ki;
        ki.reserve(G * nB / 4);
        for (size_t g = 0; g < G; g++)
            for (size_t b = 0; b < nB; b++) {
                bool ins = false;
                const double k3[3] = {re[g * nB + b], B.nodes[3 * b + 2] - c.gz, 0.0};
                kid[g * nB + b] = ki.find_or_insert(k3, static_cast<uint32_t>(n_k), ins);
                if (ins) n_k++;
            }
        if (n_z * n_k > SANDWICH_KEEP_PAIRS) return false;
    }
    // The walk, in `claim_window`'s order (a * nB + b).
    TripleIndex &keys = be.memo.keys();
    std::vector<uint32_t> tab(n_z * n_k, TripleIndex::NONE);
    for (size_t a = 0; a < nA; a++) {
        const uint32_t *kr = &kid[static_cast<size_t>(ga[a]) * nB];
        uint32_t *tr = &tab[static_cast<size_t>(za[a]) * n_k];
        uint32_t *ir = idx ? idx + a * nB : nullptr;
        for (size_t b = 0; b < nB; b++) {
            uint32_t &t = tr[kr[b]];
            if (t == TripleIndex::NONE) {
                double row[3];
                pair_row(c, a_wire, A, B, a, b, row);
                const uint32_t next = base + static_cast<uint32_t>(rows.size() / 3);
                bool ins = false;
                const bool nan = std::isnan(row[0]) || std::isnan(row[1]) || std::isnan(row[2]);
                t = nan ? nan_rows.find_or_insert(row, next, ins) : keys.find_or_insert(row, next, ins);
                if (ins) rows.insert(rows.end(), row, row + 3);
            }
            if (ir) ir[b] = t;
        }
    }
    return true;
}

// The rows one evaluation batch takes (whole exact-rho columns, so a batch
// can run over by its last column).
constexpr size_t EVAL_BATCH_ROWS = size_t(1) << 18;

// The claimed rows evaluated and appended to the memo, as `designed_index`
// ends: their values written straight into the memo's own storage, and
// evaluated a batch of whole exact-rho COLUMNS at a time (momwire#1541), not
// in one call over every row, whose internals held several copies of the
// values beside the memo's.
//
// Why a batch of whole columns is the one call's bits: every row is either
// served from a sheet or by the column twin. A sheet row is interpolated at
// its own point from a sheet grown by whole strips, each new cell evaluated
// as its own columns, so how far a call grows it moves no cell. A twin row
// is its exact-rho column's: the column's rule (its cache key, from the
// column's own members' smallest z - z') and its members' own sums, so the
// column moves no bit wherever its members sit, as long as they arrive
// together and in their order. Batches take the columns in the one call's
// first-seen order, each column's members in claim order. With
// momwire#1860 L5 (`crossing_table`) on, a call decides from its own row
// count whether it takes the table, so the rows stay ONE call.
void claim_finish(BlockEval &be, std::vector<double> &rows, size_t room) {
    const size_t m = rows.size() / 3;
    if (m == 0) return;
    Six *vals = be.memo.grow_values(m, room);
    const ni_eval::SheetPlan *plan = be.memo.have_plan ? &be.memo.plan : nullptr;
    if (m <= EVAL_BATCH_ROWS || ni_eval::active_features().crossing_table) {
        ni_eval::evaluate_fresh(be.ev, rows.data(), m, nullptr, plan, be.memo.cache.get(), vals);
        std::vector<double>().swap(rows);
        return;
    }
    // The columns (exact rho, -0.0 == 0.0) in first-seen order, and each
    // row's: members chained in claim order.
    std::vector<uint32_t> head, tail, next(m, ni_eval::TripleIndex::NONE);
    {
        ni_eval::TripleIndex col_of;  // keyed (rho, 0, 0): `x + 0.0` bits
        for (size_t i = 0; i < m; i++) {
            const double key[3] = {rows[3 * i], 0.0, 0.0};
            bool ins = false;
            const uint32_t c = col_of.find_or_insert(key, static_cast<uint32_t>(head.size()), ins);
            if (ins) {
                head.push_back(static_cast<uint32_t>(i));
                tail.push_back(static_cast<uint32_t>(i));
            } else {
                next[tail[c]] = static_cast<uint32_t>(i);
                tail[c] = static_cast<uint32_t>(i);
            }
        }
    }
    std::vector<uint32_t>().swap(tail);
    std::vector<double> sub;
    std::vector<uint32_t> at;
    std::vector<Six> got;
    size_t c = 0;
    while (c < head.size()) {
        sub.clear();
        at.clear();
        for (; c < head.size() && at.size() < EVAL_BATCH_ROWS; c++)
            for (uint32_t i = head[c]; i != ni_eval::TripleIndex::NONE; i = next[i]) {
                sub.insert(sub.end(), &rows[3 * static_cast<size_t>(i)], &rows[3 * static_cast<size_t>(i)] + 3);
                at.push_back(i);
            }
        got.resize(at.size());
        ni_eval::evaluate_fresh(be.ev, sub.data(), at.size(), nullptr, plan, be.memo.cache.get(), got.data());
        for (size_t k = 0; k < at.size(); k++) vals[at[k]] = got[k];
    }
    std::vector<double>().swap(rows);
}

// y[c] += a * key(tab[ix[c]]): `axpy_gather` over one key of the memo's
// values read in place (momwire#1541: no plane copies of them), the same
// product and sum per part.
void axpy_gather_key(cd *__restrict y, const Six *__restrict tab, int key, const uint32_t *__restrict ix,
                     double a, size_t n) {
    double *__restrict yd = reinterpret_cast<double *>(y);
    const double *__restrict td = reinterpret_cast<const double *>(tab) + 2 * key;
    for (size_t cc = 0; cc < n; cc++) {
        const size_t k = 2 * static_cast<size_t>(ni_eval::N_KEYS) * static_cast<size_t>(ix[cc]);
        yd[2 * cc] = yd[2 * cc] + a * td[k];
        yd[2 * cc + 1] = yd[2 * cc + 1] + a * td[k + 1];
    }
}

// The same over `k2 V + dzpW`, each part formed as the plane held it.
void axpy_gather_x(cd *__restrict y, const Six *__restrict tab, double k2sq, const uint32_t *__restrict ix,
                   double a, size_t n) {
    double *__restrict yd = reinterpret_cast<double *>(y);
    const double *__restrict tv = reinterpret_cast<const double *>(tab) + 2 * K_V;
    const double *__restrict tx = reinterpret_cast<const double *>(tab) + 2 * K_DZPW;
    for (size_t cc = 0; cc < n; cc++) {
        const size_t k = 2 * static_cast<size_t>(ni_eval::N_KEYS) * static_cast<size_t>(ix[cc]);
        const double xr = k2sq * tv[k] + tx[k], xi = k2sq * tv[k + 1] + tx[k + 1];
        yd[2 * cc] = yd[2 * cc] + a * xr;
        yd[2 * cc + 1] = yd[2 * cc + 1] + a * xi;
    }
}

// `_main_sandwich(ctx, A, B, ...)` with the plan: the (nA basis x nB basis)
// block of the five-term contraction, times c1 (zero outside rA x rB, then
// times c1 as numpy multiplies the whole block), written into `T`: entry
// (A row, B row) at T(A row, B row), or at T(B row, A row) when `transposed`
// (the reversed block, `_main_sandwich(ctx, Q, P).T`).
//
// momwire#1541: the grid tables are never held whole. Pass 1 forms them a
// window of A's nodes at a time and only CLAIMS each window's fresh triples
// (`claim_window`), so the memo ends holding the triples, in the order, the
// whole-grid `designed_index` gave it, and their values from the same one
// evaluation. Pass 2 forms, a window of rA's rows at a time, only the table
// rows (A nodes) those rows read, each pair's index found in the memo, and
// contracts each row as before. The memo, its key planes and every
// contracted entry are therefore the whole-grid route's bits.
void main_sandwich(const Ctx &c, BlockEval &be, double a_wire, const Axis &A, const Axis &B, Block &T,
                   bool transposed, size_t room) {
    const size_t nA = A.n_nodes(), nB = B.n_nodes();
    // The sheet plan, once per fill, from the node pairs.
    {
        std::vector<double> pa(A.nodes), pb(B.nodes);
        for (size_t i = 0; i < nA; i++) pa[3 * i + 2] -= c.gz;
        for (size_t i = 0; i < nB; i++) pb[3 * i + 2] -= c.gz;
        be.memo.plan = ni_eval::sheet_plan(c.k_p, c.k_m, {ni_eval::NodePair{pa.data(), nA, pb.data(), nB}}, a_wire);
        be.memo.have_plan = true;
    }
    const size_t per = std::max<size_t>(1, SANDWICH_WINDOW_PAIRS / std::max<size_t>(1, nB));
    ni_eval::TripleIndex nan_rows;
    // Every pair's memo index from pass 1 (row-major, a * nB + b), or empty.
    const bool keep = nA * nB <= SANDWICH_KEEP_PAIRS;
    std::vector<uint32_t> kept(keep ? nA * nB : 0);
    // Pass 1: every pair's triple claimed, window by window, in row-major
    // grid order (a * nB + b), then the one evaluation.
    {
        const uint32_t base = static_cast<uint32_t>(be.memo.size());
        std::vector<double> fresh, re, z, zp;
        const bool grouped = claim_grouped(c, be, a_wire, A, B, base, nan_rows, fresh, keep ? kept.data() : nullptr);
        for (size_t a0 = 0; a0 < nA && !grouped; a0 += per) {
            const size_t a1 = std::min(nA, a0 + per), m = (a1 - a0) * nB;
            re.resize(m);
            z.resize(m);
            zp.resize(m);
            const std::ptrdiff_t na = static_cast<std::ptrdiff_t>(a1 - a0);
            const int nt = team(be, m);
#if defined(_OPENMP)
            #pragma omp parallel for schedule(static) num_threads(nt) if (nt > 1)
#else
            (void)nt;
#endif
            for (std::ptrdiff_t ap = 0; ap < na; ap++) {
                const size_t a = a0 + static_cast<size_t>(ap);
                for (size_t b = 0; b < nB; b++) {
                    const size_t q = static_cast<size_t>(ap) * nB + b;
                    double row[3];
                    pair_row(c, a_wire, A, B, a, b, row);
                    re[q] = row[0];
                    z[q] = row[1];
                    zp[q] = row[2];
                }
            }
            claim_window(be, base, re.data(), z.data(), zp.data(), m, nan_rows, fresh,
                         keep ? kept.data() + a0 * nB : nullptr);
        }
        std::vector<double>().swap(re);
        std::vector<double>().swap(z);
        std::vector<double>().swap(zp);
        // The end terms append their own fresh values after these: room for
        // as many as they have pairs, at most this evaluation's count again
        // (the capacity a doubling would have taken, without its copy).
        claim_finish(be, fresh, std::min(room, fresh.size() / 3));
    }
    const double k2sq = c.k_p * c.k_p;
    const std::vector<int64_t> rA = support_rows(A), rB = support_rows(B);
    const std::array<CSR, 4> P = row_weights(A, rA), Q = row_weights(B, rB);
    // The four keys the contraction reads, gathered from the memo's values
    // in place (they were copied into one plane each); `k2 V + dzpW` is a
    // function of the value alone, formed as its plane was.
    const Six *MV = be.memo.size() ? &be.memo.at(0) : nullptr;
    // Pass 2, a window of rA's rows at a time: the A nodes those rows read
    // (their four weight rows' columns), each one's table row of memo
    // indices, then each row's left products (`left_products_gathered`'s
    // sums, six (nB) rows) into the thread's own buffer and `combine_rows`
    // against Q's rows: ((((T0 + T1) + T2) + T3) + T4) - T5.
    static const int qsel[6] = {0, 1, 2, 3, 2, 3};
    const size_t nr = rA.size();
    std::vector<int64_t> slot(nA, -1);
    std::vector<size_t> need;
    std::vector<uint32_t> K;
    size_t i0 = 0;
    while (i0 < nr) {
        // The window: rows while their nodes fit (at least one row); with
        // `kept`, every row at once, reading A node a's indices at a * nB.
        need.clear();
        size_t i1 = i0;
        if (keep) i1 = nr;
        while (i1 < nr) {
            size_t added = 0;
            for (int pm = 0; pm < 4; pm++) {
                const CSR &M = P[static_cast<size_t>(pm)];
                for (int64_t jj = M.indptr[i1]; jj < M.indptr[i1 + 1]; jj++) {
                    const size_t node = static_cast<size_t>(M.indices[static_cast<size_t>(jj)]);
                    if (slot[node] < 0) {
                        slot[node] = static_cast<int64_t>(need.size());
                        need.push_back(node);
                        added++;
                    }
                }
            }
            i1++;
            if (i1 > i0 + 1 && need.size() > per) {
                // This row overflowed the window: hand it to the next one.
                for (size_t k = need.size() - added; k < need.size(); k++) slot[need[k]] = -1;
                need.resize(need.size() - added);
                i1--;
                break;
            }
            if (need.size() >= per) break;
        }
        // The window's table rows: each pair's memo index (a NaN row's from
        // the claim's own NaN dedup, as `designed_index` answered it).
        if (!keep) K.resize(need.size() * nB);
        if (!keep) {
            const std::ptrdiff_t nnp = static_cast<std::ptrdiff_t>(need.size());
            const int nt = team(be, need.size() * nB);
#if defined(_OPENMP)
            #pragma omp parallel for schedule(static) num_threads(nt) if (nt > 1)
#else
            (void)nt;
#endif
            for (std::ptrdiff_t u = 0; u < nnp; u++) {
                const size_t a = need[static_cast<size_t>(u)];
                uint32_t *kr = &K[static_cast<size_t>(u) * nB];
                for (size_t b = 0; b < nB; b++) {
                    double row[3];
                    pair_row(c, a_wire, A, B, a, b, row);
                    uint32_t ix = be.memo.find(row);
                    if (ix == ni_eval::TripleIndex::NONE) ix = nan_rows.find(row);
                    kr[b] = ix;
                }
            }
        }
        if (std::find(K.begin(), K.end(), ni_eval::TripleIndex::NONE) != K.end())
            throw std::runtime_error("razor crossing: a table pair missing from the memo (momwire#1541)");
        // A's node `node`'s table row: its kept indices, or the window's.
        const uint32_t *Kb = keep ? kept.data() : K.data();
        const auto row_of = [&](int64_t node) -> const uint32_t * {
            const size_t at = keep ? static_cast<size_t>(node) : static_cast<size_t>(slot[static_cast<size_t>(node)]);
            return &Kb[at * nB];
        };
        const std::ptrdiff_t w0 = static_cast<std::ptrdiff_t>(i0), w1 = static_cast<std::ptrdiff_t>(i1);
        const int nt = team(be, (i1 - i0) * nB);
#if defined(_OPENMP)
        #pragma omp parallel num_threads(nt) if (nt > 1)
#else
        (void)nt;
#endif
        {
        std::vector<cd> Lrow(6 * nB);
#if defined(_OPENMP)
        #pragma omp for schedule(dynamic)
#endif
        for (std::ptrdiff_t ip = w0; ip < w1; ip++) {
            const size_t i = static_cast<size_t>(ip);
            std::fill(Lrow.begin(), Lrow.end(), cd(0.0, 0.0));
            cd *L[6];
            for (int m = 0; m < 6; m++) L[m] = &Lrow[static_cast<size_t>(m) * nB];
            for (int pm = 0; pm < 3; pm++) {
                const CSR &M = P[static_cast<size_t>(pm)];
                for (int64_t jj = M.indptr[i]; jj < M.indptr[i + 1]; jj++) {
                    const double a = M.data[static_cast<size_t>(jj)];
                    const uint32_t *kx = row_of(M.indices[static_cast<size_t>(jj)]);
                    if (pm < 2) {
                        axpy_gather_key(L[pm], MV, K_U, kx, a, nB);
                    } else {
                        axpy_gather_x(L[2], MV, k2sq, kx, a, nB);
                        axpy_gather_key(L[3], MV, K_W, kx, a, nB);
                    }
                }
            }
            const CSR &M = P[3];
            for (int64_t jj = M.indptr[i]; jj < M.indptr[i + 1]; jj++) {
                const double a = M.data[static_cast<size_t>(jj)];
                const uint32_t *kx = row_of(M.indices[static_cast<size_t>(jj)]);
                axpy_gather_key(L[4], MV, K_W, kx, a, nB);
                axpy_gather_key(L[5], MV, K_V, kx, a, nB);
            }
            for (size_t j = 0; j < rB.size(); j++) {
                cd *dst = transposed ? T.at(rB[j], rA[i]) : T.at(rA[i], rB[j]);
                if (dst == nullptr) continue;
                double tr[6], ti[6];
                for (int m = 0; m < 6; m++) {
                    const CSR &Mq = Q[static_cast<size_t>(qsel[m])];
                    csr_row_dot(Mq.indices.data(), Mq.data.data(), Mq.indptr[j], Mq.indptr[j + 1],
                                reinterpret_cast<const double *>(L[m]), tr[m], ti[m]);
                }
                *dst = cd(((((tr[0] + tr[1]) + tr[2]) + tr[3]) + tr[4]) - tr[5],
                          ((((ti[0] + ti[1]) + ti[2]) + ti[3]) + ti[4]) - ti[5]);
            }
        }
        }
        for (size_t node : need) slot[node] = -1;
        i0 = i1;
    }
    {
        const std::ptrdiff_t nt2 = static_cast<std::ptrdiff_t>(T.v.size());
        const int ntc = team(be, T.v.size());
#if defined(_OPENMP)
        #pragma omp parallel for schedule(static) num_threads(ntc) if (ntc > 1)
#else
        (void)ntc;
#endif
        for (std::ptrdiff_t q = 0; q < nt2; q++) T.v[static_cast<size_t>(q)] = np_cmul(T.v[static_cast<size_t>(q)], c.c1);
    }
}

// The team cap: MOMWIRE_THREADS clamped to the CPUs the affinity mask
// allows, else one per physical core in the mask (`_cpu_count.h`). The block's
// values do not depend on it (`team`: disjoint outputs, serial order each).
int n_physical_threads() { return mw::cpu::pool_threads(); }

// The (end, line node) pairs `add_end_terms_kept(R, C)` asks tables of: a
// bound on the fresh values it adds to the block's memo.
size_t end_pairs(const RAxis &R, const RAxis &C) {
    return R.ends.size() * C.n_nodes() + C.ends.size() * R.n_nodes();
}

// `add_end_terms<RazorPolicy>(c, be, a_wire, R, C, r_above, t)` (razor's: no
// corner, both loops) into the compact block (momwire#1541): E_r and E_c are
// formed only over the rows and columns `T` keeps, and the ends' fv read
// sparse (`RAxis::nz`). Every kept entry of E_r, E_c and T takes the terms
// the shared spelling gives it, in the same order (ends in end order, each
// loop's two passes, then `answer`'s two adds), so the fold reads the same
// bits; the dropped entries only fed entries the fold never reads.
template <class P>
void add_end_terms_kept(const Ctx &c, BlockEval &be, double a_wire, const RAxis &R, const RAxis &C,
                        bool r_above, Block &T) {
    const int64_t nR = R.n_basis, nC = C.n_basis;
    auto live = [](const RAxis &ax) {
        std::set<int64_t> s;
        for (const EndNz &e : ax.nz) s.insert(e.idx.begin(), e.idx.end());
        return std::vector<int64_t>(s.begin(), s.end());
    };
    const std::vector<int64_t> LA = live(R), LB = live(C);
    // The kept live rows (LA in T's rows) and live columns (LB in T's cols).
    std::vector<int64_t> la, lb;  // basis
    for (int64_t a : LA)
        if (T.rpos[static_cast<size_t>(a)] >= 0) la.push_back(a);
    for (int64_t b : LB)
        if (T.cpos[static_cast<size_t>(b)] >= 0) lb.push_back(b);
    std::vector<int64_t> posLA(static_cast<size_t>(nR), -1), posLB(static_cast<size_t>(nC), -1);
    for (size_t i = 0; i < la.size(); i++) posLA[static_cast<size_t>(la[i])] = static_cast<int64_t>(i);
    for (size_t i = 0; i < lb.size(); i++) posLB[static_cast<size_t>(lb[i])] = static_cast<int64_t>(i);
    const size_t nq = T.cols.size(), nr = T.rows.size();
    std::vector<cd> E_r(la.size() * nq, cd(0.0, 0.0));  // (la, T.cols)
    std::vector<cd> E_c(nr * lb.size(), cd(0.0, 0.0));  // (T.rows, lb)
    const cd neg_c1(-c.c1.real(), -c.c1.imag());
    // Row loop: R's ends against C's line, folded a batch of ends at a time.
    size_t e_row = 0;
    auto row_fold = [&](const std::vector<EndVec> &evs) {
        const size_t g0 = e_row;
        e_row += evs.size();
        const int nt = team(be, evs.size() * static_cast<size_t>(nC));
#if defined(_OPENMP)
        #pragma omp parallel for schedule(static, 1) num_threads(nt) if (nt > 1)
#endif
        for (int ch = 0; ch < nt; ch++) {
            const size_t q0 = nq * static_cast<size_t>(ch) / static_cast<size_t>(nt);
            const size_t q1 = nq * static_cast<size_t>(ch + 1) / static_cast<size_t>(nt);
            for (size_t e = 0; e < evs.size(); e++) {
                const EndVec &ev = evs[e];
                const EndNz &nz = R.nz[g0 + e];
                const cd s_plus = py_prod(c.c1, cd(ev.sign, 0.0));
                const cd s_minus = py_prod(neg_c1, cd(ev.sign, 0.0));
                for (int pass = 0; pass < 2; pass++) {
                    const std::vector<cd> &vec = pass == 0 ? ev.vV : ev.vW;
                    const cd scale = pass == 0 ? s_plus : s_minus;
                    for (size_t k = 0; k < nz.idx.size(); k++) {
                        const int64_t a = posLA[static_cast<size_t>(nz.idx[k])];
                        if (a < 0) continue;
                        const double fvi = nz.val[k];
                        cd *row = &E_r[static_cast<size_t>(a) * nq];
                        for (size_t q = q0; q < q1; q++) {
                            const int64_t j = T.cols[q];
                            const cd o = np_cmul(np_cmul(cd(fvi, 0.0), vec[static_cast<size_t>(j)]), scale);
                            row[q] = cd(row[q].real() + o.real(), row[q].imag() + o.imag());
                        }
                    }
                }
            }
        }
    };
    end_vectors_each<P>(c, be, a_wire, R.ends, C, r_above, row_fold);
    // Column loop: C's ends against R's line (W, then V), batched the same way.
    size_t e_col = 0;
    auto col_fold = [&](const std::vector<EndVec> &evs) {
        const size_t g0 = e_col;
        e_col += evs.size();
        const size_t nla = la.size();
        const int nt = team(be, evs.size() * static_cast<size_t>(nR));
#if defined(_OPENMP)
        #pragma omp parallel for schedule(static, 1) num_threads(nt) if (nt > 1)
#endif
        for (int ch = 0; ch < nt; ch++) {
            const size_t a0 = nla * static_cast<size_t>(ch) / static_cast<size_t>(nt);
            const size_t a1 = nla * static_cast<size_t>(ch + 1) / static_cast<size_t>(nt);
            const size_t r0 = nr * static_cast<size_t>(ch) / static_cast<size_t>(nt);
            const size_t r1 = nr * static_cast<size_t>(ch + 1) / static_cast<size_t>(nt);
            for (size_t e = 0; e < evs.size(); e++) {
                const EndVec &ev = evs[e];
                const EndNz &nz = C.nz[g0 + e];
                const cd s_plus = py_prod(c.c1, cd(ev.sign, 0.0));
                const cd s_minus = py_prod(neg_c1, cd(ev.sign, 0.0));
                for (int pass = 0; pass < 2; pass++) {
                    const std::vector<cd> &vec = pass == 0 ? ev.vW : ev.vV;
                    const cd scale = pass == 0 ? s_minus : s_plus;
                    for (size_t k = 0; k < nz.idx.size(); k++) {
                        const int64_t jn = nz.idx[k];
                        const int64_t q = T.cpos[static_cast<size_t>(jn)];
                        if (q < 0) continue;  // a column the fold never reads
                        const double fvj = nz.val[k];
                        // E_r's columns (rows la), then E_c's (every kept row).
                        for (size_t i = a0; i < a1; i++) {
                            const cd o = np_cmul(np_cmul(vec[static_cast<size_t>(la[i])], cd(fvj, 0.0)), scale);
                            cd &x = E_r[i * nq + static_cast<size_t>(q)];
                            x = cd(x.real() + o.real(), x.imag() + o.imag());
                        }
                        const size_t lbj = static_cast<size_t>(posLB[static_cast<size_t>(jn)]);
                        for (size_t r = r0; r < r1; r++) {
                            const cd o = np_cmul(np_cmul(vec[static_cast<size_t>(T.rows[r])], cd(fvj, 0.0)), scale);
                            cd &x = E_c[r * lb.size() + lbj];
                            x = cd(x.real() + o.real(), x.imag() + o.imag());
                        }
                    }
                }
            }
        }
    };
    end_vectors_each<P>(c, be, a_wire, C.ends, R, !r_above, col_fold);
    // `answer`: E_c[LA, :] = 0; dest[LA, :] += E_r; dest[:, LB] += E_c.
    for (int64_t a : la) {
        const size_t r = static_cast<size_t>(T.rpos[static_cast<size_t>(a)]);
        for (size_t j = 0; j < lb.size(); j++) E_c[r * lb.size() + j] = cd(0.0, 0.0);
    }
    for (size_t i = 0; i < la.size(); i++) {
        cd *dst = &T.v[static_cast<size_t>(T.rpos[static_cast<size_t>(la[i])]) * nq];
        for (size_t q = 0; q < nq; q++) {
            const cd v = E_r[i * nq + q];
            dst[q] = cd(dst[q].real() + v.real(), dst[q].imag() + v.imag());
        }
    }
    for (size_t r = 0; r < nr; r++)
        for (size_t j = 0; j < lb.size(); j++) {
            cd &d = T.v[r * nq + static_cast<size_t>(T.cpos[static_cast<size_t>(lb[j])])];
            const cd v = E_c[r * lb.size() + j];
            d = cd(d.real() + v.real(), d.imag() + v.imag());
        }
}

// `_crossing_node_charges(geom, tents, ctx, A, P, omega)`: the (n_basis,
// n_tents) addition to the crossing tents' columns, `-sgn q (fam - c1 V)` at
// each T2 endpoint of A (above rows) and P (below rows), np.add.at'ed.
std::vector<cd> node_charges(const Ctx &c, const RAxis &A, const RAxis &P,
                             const std::vector<std::pair<int64_t, int>> &tents, uintptr_t cancel_flag) {
    const Front &f = *c.f;
    const size_t T = tents.size();
    const int64_t n = f.n_basis;
    std::vector<int64_t> cols(T);
    std::vector<double> q_a(T), q_b(T), a_above(T), a_below(T), nodes(3 * T);
    for (size_t t = 0; t < T; t++) {
        const int64_t m = tents[t].first;
        const int jb = tents[t].second, ja = 1 - jb;
        cols[t] = m;
        auto seg_r = [&](int64_t seg) {
            int64_t w = 0;
            while (f.seg_offsets[static_cast<size_t>(w) + 1] <= seg) w++;
            return f.radius_per_wire[static_cast<size_t>(w)];
        };
        a_above[t] = seg_r(f.wing_seg[2 * m + ja]);
        a_below[t] = seg_r(f.wing_seg[2 * m + jb]);
        // -wg * where(rise, 1, -1)
        q_a[t] = -f.wing_sigma[2 * m + ja] * (f.wing_rise.p[2 * m + ja] ? 1.0 : -1.0);
        q_b[t] = -f.wing_sigma[2 * m + jb] * (f.wing_rise.p[2 * m + jb] ? 1.0 : -1.0);
        const int64_t sa = f.wing_seg[2 * m];
        const double along = f.wing_rise.p[2 * m] ? f.seg_h[static_cast<size_t>(sa)] : 0.0;
        for (int k = 0; k < 3; k++) nodes[3 * t + k] = f.seg_p0[3 * sa + k] + along * f.seg_t[3 * sa + k];
    }
    for (size_t t = 0; t < T; t++) {
        // `np.allclose(q_a, -q_b)`
        const double a = q_a[t], b = -q_b[t];
        if (!(std::fabs(a - b) <= 1e-8 + 1e-5 * std::fabs(b)))
            throw std::runtime_error("crossing tent whose half tents' node charges do not cancel (momwire#1149 U2)");
    }
    std::vector<cd> out(static_cast<size_t>(n) * T, cd(0.0, 0.0));  // row-major (n, T)
    BlockEval be;  // `_near_interface.TripleMemo()`: no sheet plan
    be.ev.k_p = c.k_p;
    be.ev.k_m = c.k_m;
    be.ev.n_threads = n_physical_threads();
    be.ev.cancel_flag = cancel_flag;
    for (int pass = 0; pass < 2; pass++) {
        const bool above = pass == 0;
        const RAxis &ax = above ? A : P;
        const size_t E = ax.ends.size();
        if (E == 0) continue;
        const std::vector<double> &a_half = above ? a_above : a_below;
        std::vector<double> rho(E * T), R(E * T), zt(E);
        std::vector<int64_t> row(E);
        for (size_t e = 0; e < E; e++) {
            const End &en = ax.ends[e];
            // The first row whose fv is nonzero (`EndNz` holds them in order).
            if (!ax.nz[e].idx.empty()) row[e] = ax.nz[e].idx.front();
            for (size_t t = 0; t < T; t++) {
                const double d0 = en.pt[0] - nodes[3 * t], d1 = en.pt[1] - nodes[3 * t + 1],
                             d2 = en.pt[2] - nodes[3 * t + 2];
                rho[e * T + t] = std::hypot(d0, d1);
                // `np.einsum("etk,etk->et", d, d)`: numpy pairs the first and
                // last terms, then adds the middle one (measured, every shape
                // and both CPUs).
                const double dd = (d0 * d0 + d2 * d2) + d1 * d1;
                R[e * T + t] = std::sqrt(dd + a_half[t] * a_half[t]);
            }
            zt[e] = on_plane_side(en.pt[2] - c.gz, above, "T2 endpoint");
        }
        // fam, per (e, t)
        std::vector<cd> fam(E * T);
        const cd one_minus = above ? cd(1.0 - c.c2.real(), 0.0 - c.c2.imag())
                                   : cd(1.0 - c.a_m.real(), 0.0 - c.a_m.imag());
        const cd m1j(-0.0, -1.0);
        const cd jk = above ? py_prod(m1j, cd(c.k_p, 0.0)) : py_prod(m1j, c.k_m);
        const cd j_omega = py_prod(cd(0.0, 1.0), cd(c.omega, 0.0));
        const cd denom = above ? py_prod(j_omega, cd(c.eps, 0.0)) : py_prod(j_omega, c.eps_m);
        for (size_t q = 0; q < E * T; q++) {
            const cd arg = np_cmul(jk, cd(R[q], 0.0));
            const cd ex = std::exp(arg);
            cd v = np_cmul(one_minus, ex);
            v = np_cdiv(v, cd(4.0 * M_PI * R[q], 0.0));
            fam[q] = np_cdiv(v, denom);
        }
        // V per radius group (ascending), `_tables(ctx_r, ...)` over (E, T_r).
        std::set<double> radii(a_half.begin(), a_half.end());
        std::vector<cd> V(E * T);
        for (double r : radii) {
            std::vector<size_t> ts;
            for (size_t t = 0; t < T; t++)
                if (a_half[t] == r) ts.push_back(t);
            std::vector<double> rr, zz, zzp;
            for (size_t e = 0; e < E; e++)
                for (size_t t : ts) {
                    rr.push_back(rho[e * T + t]);
                    if (above) {
                        zz.push_back(zt[e]);
                        zzp.push_back(0.0);
                    } else {
                        zz.push_back(0.0);
                        zzp.push_back(zt[e]);
                    }
                }
            std::vector<Six> te;
            tables(be, r, rr, zz, zzp, nullptr, te);
            size_t q = 0;
            for (size_t e = 0; e < E; e++)
                for (size_t t : ts) V[e * T + t] = te[q++][K_V];
        }
        const std::vector<double> &qq = above ? q_a : q_b;
        for (size_t e = 0; e < E; e++) {
            const double sg = ax.ends[e].sign;
            for (size_t t = 0; t < T; t++) {
                const cd cv = np_cmul(c.c1, V[e * T + t]);
                const cd diff(fam[e * T + t].real() - cv.real(), fam[e * T + t].imag() - cv.imag());
                const double coef = (-sg) * qq[t];
                const cd val = np_cmul(cd(coef, 0.0), diff);
                cd &o = out[static_cast<size_t>(row[e]) * T + t];
                o = cd(o.real() + val.real(), o.imag() + val.imag());
            }
        }
    }
    return out;
}

// The cross blocks and node charges at the front's own wavenumber: the
// shipped route, and what `crossing_blocks` runs whenever the anchors do not
// serve the fill.
void crossing_blocks_exact(const Front &f, std::vector<cd> &z,
                           const std::vector<std::pair<int64_t, int>> &tents, uintptr_t cancel_flag) {
    const Ctx c = make_ctx(f);
    const int64_t n = f.n_basis;
    const RAxis A = crossing_path_axis(f, tents, false);
    const RAxis P = crossing_path_axis(f, tents, true);
    // Each medium's rows (`_medium_geometry`'s `rows`): its wires' interior
    // bases, then the junction bases with a wing on its segments.
    auto rows_of = [&](bool below) {
        std::vector<int64_t> rows;
        std::vector<char> mine(static_cast<size_t>(f.n_seg), 0);
        for (int64_t w = 0; w < f.n_wires; w++)
            if ((f.media_below[static_cast<size_t>(w)] != 0) == below) {
                for (int64_t g = f.seg_offsets[w]; g < f.seg_offsets[w + 1]; g++) mine[static_cast<size_t>(g)] = 1;
                for (int64_t m = f.basis_offsets[w]; m < f.basis_offsets[w + 1]; m++) rows.push_back(m);
            }
        for (int64_t m = f.basis_offsets.back(); m < n; m++)
            if (mine[static_cast<size_t>(f.wing_seg[2 * m])] || mine[static_cast<size_t>(f.wing_seg[2 * m + 1])])
                rows.push_back(m);
        return rows;
    };
    const std::vector<int64_t> rows_a = rows_of(false), rows_b = rows_of(true);
    // The source partitions by radius (`_seg_radius`), ascending.
    auto parts_of = [&](bool below) {
        std::map<double, std::vector<int64_t>> parts;
        for (int64_t w = 0; w < f.n_wires; w++) {
            if ((f.media_below[static_cast<size_t>(w)] != 0) != below) continue;
            const double r = f.radius_per_wire[static_cast<size_t>(w)];
            for (int64_t g = f.seg_offsets[w]; g < f.seg_offsets[w + 1]; g++) parts[r].push_back(g);
        }
        return parts;
    };
    for (int dir = 0; dir < 2; dir++) {
        const bool src_below = dir == 0;
        const std::vector<int64_t> &rows = src_below ? rows_a : rows_b;
        const std::vector<int64_t> &cols = src_below ? rows_b : rows_a;
        if (rows.empty()) continue;
        for (const auto &part : parts_of(src_below)) {
            const double r = part.first;
            const RAxis ax = axis_data(c, part.second, r);
            BlockEval be;
            be.ev.k_p = c.k_p;
            be.ev.k_m = c.k_m;
            be.ev.n_threads = n_physical_threads();
            be.ev.cancel_flag = cancel_flag;
            // The block (rows = test axis), held only over the entries the
            // fold reads (momwire#1541; it was a whole n x n block).
            Block t;
            t.shape(n, rows, cols);
            if (src_below) {
                // `cross_complete_block(ctx_r, A, ax)`: above test rows x below
                // source columns.
                //
                // Ordering (momwire#1860 L5, `crossing_table`): Python's fused
                // streamed route asks for this block's slow end terms in a
                // prepass BEFORE its main tiles (`_FusedEnds._slow_prepass`),
                // while this route forms them after `main_sandwich`. Whether a
                // twin call reads the crossing table depends on whether an
                // earlier call in the fill already took it, so the end terms
                // must decide on the state the main tiles started from, not
                // the one they left: `CrossingEndsFirst` restores it around
                // them. In the fill's first such block that state is "not
                // taken", so the end terms weigh their own rows, as Python's
                // prepass does. With the feature off nothing reads it.
                const bool taken_before = ni_eval::crossing_table_taken();
                main_sandwich(c, be, r, A, ax, t, false, end_pairs(A, ax));
                {
                    const ni_eval::CrossingEndsFirst ends_first(taken_before);
                    add_end_terms_kept<RazorPolicy>(c, be, r, A, ax, true, t);
                }
            } else {
                // `cross_complete_block_reversed(ctx_r, P, ax)`: the main
                // sandwich over (above source Q = ax, below test P),
                // transposed as it is written; the ends with the rows below.
                main_sandwich(c, be, r, ax, P, t, true, end_pairs(P, ax));
                add_end_terms_kept<RazorPolicy>(c, be, r, P, ax, false, t);
            }
            // The fold: `Z[rows, cols] -= t`, columns apart.
            const std::ptrdiff_t ncols = static_cast<std::ptrdiff_t>(cols.size());
            const size_t nrows = rows.size();
            const int ntf = team(be, cols.size() * rows.size());
#if defined(_OPENMP)
            #pragma omp parallel for schedule(static) num_threads(ntf) if (ntf > 1)
#else
            (void)ntf;
#endif
            for (std::ptrdiff_t cq = 0; cq < ncols; cq++) {
                const int64_t ci = cols[static_cast<size_t>(cq)];
                for (size_t rq = 0; rq < nrows; rq++) {
                    const int64_t ri = rows[rq];
                    cd &e = z[static_cast<size_t>(ci * n + ri)];
                    const cd v = t.v[rq * static_cast<size_t>(ncols) + static_cast<size_t>(cq)];
                    e = cd(e.real() - v.real(), e.imag() - v.imag());
                }
            }
        }
    }
    // 5. The node charges into the crossing tents' columns: Z[:, cols] += charges.
    if (!tents.empty()) {
        const std::vector<cd> ch = node_charges(c, A, P, tents, cancel_flag);
        const size_t T = tents.size();
        for (size_t t = 0; t < T; t++) {
            const int64_t col = tents[t].first;
            for (int64_t i = 0; i < n; i++) {
                cd &e = z[static_cast<size_t>(col * n + i)];
                const cd v = ch[static_cast<size_t>(i) * T + t];
                e = cd(e.real() + v.real(), e.imag() + v.imag());
            }
        }
    }
}

// ---------------------------------------------------------------------------
// The opt-in feature `crossing_anchors` (see the header).
//
// Over a sweep the cross blocks' pairs are fixed and only the kernel moves:
// k_p, and k_m through k and eps~(omega). Over a narrow band each entry is a
// smooth function of k, so the entries are computed exactly at n_a
// Chebyshev-Lobatto wavenumbers spanning the plan (its two ends among them)
// and every other sweep point's are the barycentric interpolant of those. A
// fill whose k IS an anchor's (the plan's two ends) adds that anchor's
// entries as computed. (Taking each entry's propagation phase exp(-j kappa R)
// out first, kappa = k_m or k, R the bases' centre distance, was measured and
// made the interpolant WORSE on the target deck -- 3.7e-11 / 2.1e-11 against
// 9.1e-12 at n_a = 5 over 3 % -- so the values are interpolated as they are.)

// The fields `crossing_blocks_exact` reads, at wavenumber `k` (c, k, omega as
// `Front::build` spells them, so an anchor at a plan wavenumber is that
// frequency's front bit for bit). Front itself is move-only.
void crossing_front_at(const Front &f, double k, Front &o) {
    o.n_basis = f.n_basis;
    o.n_seg = f.n_seg;
    o.n_wires = f.n_wires;
    o.basis_offsets = f.basis_offsets;
    o.seg_offsets = f.seg_offsets;
    o.media_below = f.media_below;
    o.radius_per_wire = f.radius_per_wire;
    o.seg_p0 = f.seg_p0;
    o.seg_t = f.seg_t;
    o.seg_h = f.seg_h;
    o.wing_seg = f.wing_seg;
    o.wing_sigma = f.wing_sigma;
    o.wing_rise.assign(f.wing_rise.n, false);
    for (size_t i = 0; i < f.wing_rise.n; i++) o.wing_rise.p[i] = f.wing_rise.p[i];
    o.ground_z = f.ground_z;
    o.ground_eps_r_ = f.ground_eps_r_;
    o.ground_sigma_ = f.ground_sigma_;
    o.buried = f.buried;
    o.c = f.c;
    o.k = k;
    o.omega = o.c * k;
}

// The error rule's calibration (`anchors::bound`). Fitted on 43 synthetic
// crossing decks (a vertical over 2-8 buried radials; 1.8-28 MHz; 1-9.5 %
// bands; radials 0.1-1 wavelength, to 40 m; depth 0.02-1.5 m; sigma 1e-3 to
// 3e-2, eps_r 5-30; some sloped), each at a forced n_a = 5, 7, 9 against
// off, the nec5-razor2p executable's full-precision Zin: every point above
// the 3e-11 rounding floor satisfies |dZ|/|Z| <= 1.39e-2 X + 1.25 P (X, P
// the two terms in `bound`). C_phase is twice that; C_origin four times,
// because the origin term rests on two small-deck points (a held-out set of
// 36 more put one at 0.54 of twice the fit). With these, every point of the
// 79 decks sits at most 0.46 of the bound. The target sits a factor of three
// under the registry's declared drift.
constexpr double ANCHOR_BOUND_C_PHASE = 3e-2;
constexpr double ANCHOR_BOUND_C_ORIGIN = 5.0;
constexpr double ANCHOR_BOUND_TARGET = 5e-8 / 3.0;
// The anchors' working memory: their values (n_a x the cross entries) and
// the one n x n scratch an exact fill writes into, at 16 bytes an entry. A
// plan whose anchors would pass this declines ("memory"), so the feature
// never adds more than this to a fill's peak (the buried memory bound,
// 512 MB, is the reason it is not larger).
constexpr double ANCHOR_MEMORY_BYTES = 256.0 * 1024 * 1024;

// The internal knobs, read when a plan is made: MOMWIRE_CROSSING_ANCHOR_NODES
// (a study knob: n_a forced, the error rule below bypassed) and
// MOMWIRE_CROSSING_ANCHOR_BAND ((k_max - k_min) / k_mid above which a plan
// declines, default 0.10).
//
// The anchor count is otherwise CHOSEN, from an a-priori bound on the
// interpolation error (`anchors::bound`): the smallest n_a in
// [ANCHOR_NODES_MIN, ANCHOR_NODES_MAX] whose bound is at most
// ANCHOR_BOUND_TARGET, or a decline ("electrically-large") when none is. The
// band alone does not bound the error: the cross entries' phase runs over
// the structure's span, so the error grows as (dk_m * D)^n_a with D the
// largest above-to-below distance, and a long, low-loss deck at a band the
// cap allows moved Zin 1.2e-6 at a fixed n_a = 5 (the 11-17 review).
constexpr int ANCHOR_NODES_MIN = 5;
constexpr int ANCHOR_NODES_MAX = 12;
constexpr double ANCHOR_MAX_BAND_DEFAULT = 0.10;

int anchor_nodes_knob() {
    const char *v = std::getenv("MOMWIRE_CROSSING_ANCHOR_NODES");
    if (v == nullptr || *v == '\0') return 0;
    const long n = std::strtol(v, nullptr, 10);
    return n >= 2 && n <= 32 ? static_cast<int>(n) : 0;
}

double anchor_band_knob() {
    const char *v = std::getenv("MOMWIRE_CROSSING_ANCHOR_BAND");
    if (v == nullptr || *v == '\0') return ANCHOR_MAX_BAND_DEFAULT;
    const double b = std::strtod(v, nullptr);
    return b > 0.0 && b < 2.0 ? b : ANCHOR_MAX_BAND_DEFAULT;
}

// |d k_m / d k| at k: k_m = k sqrt(eps~), eps~ = eps_r - j sigma Z0 / k (Z0 =
// mu0 c), so d k_m / d k = (2 eps_r - j sigma Z0 / k) / (2 sqrt(eps~)). How
// much faster than k_p the below medium's phase turns with frequency. 1 with
// no lower medium given.
double km_slope(double k, const anchors::Size &s) {
    if (!(s.eps_r > 0.0) || !(s.c > 0.0)) return 1.0;
    const double z0 = 4e-7 * M_PI * s.c;
    const std::complex<double> eps_t(s.eps_r, -s.sigma * z0 / k);
    const std::complex<double> num(2.0 * s.eps_r, -s.sigma * z0 / k);
    return std::abs(num / (2.0 * std::sqrt(eps_t)));
}

}  // namespace

namespace anchors {

const double BOUND_TARGET = ANCHOR_BOUND_TARGET;
const int NODES_MAX = ANCHOR_NODES_MAX;

double span_of(int64_t n_seg, const double *seg_p0, const double *seg_t, const double *seg_h, double ground_z) {
    // Each segment's two ends, sorted by side on its midpoint (a crossing
    // deck's segments each lie wholly in one medium).
    std::vector<double> up, down;
    for (int64_t g = 0; g < n_seg; g++) {
        const double *p = seg_p0 + 3 * g, *t = seg_t + 3 * g;
        const double h = seg_h[g];
        const double mid = p[2] + 0.5 * h * t[2];
        std::vector<double> &side = mid < ground_z ? down : up;
        for (int e = 0; e < 2; e++)
            for (int i = 0; i < 3; i++) side.push_back(p[i] + (e ? h * t[i] : 0.0));
    }
    double d2 = 0.0;
    for (size_t a = 0; a < up.size(); a += 3)
        for (size_t b = 0; b < down.size(); b += 3) {
            const double dx = up[a] - down[b], dy = up[a + 1] - down[b + 1], dz = up[a + 2] - down[b + 2];
            d2 = std::max(d2, dx * dx + dy * dy + dz * dz);
        }
    return std::sqrt(d2);
}

bool fits(int n_a, int64_t entries, int64_t n) {
    const double bytes = 16.0 * (static_cast<double>(n_a) * static_cast<double>(entries) +
                                 static_cast<double>(n) * static_cast<double>(n));
    return bytes <= ANCHOR_MEMORY_BYTES;
}

double bound(int n, double x, double rho) {
    // C_phase X + C_origin P, the two error terms of Chebyshev interpolation
    // of degree n - 1 on the band: X = 2 (x/2)^n / n! for an entry that
    // turns its phase by x over the half band (e^{j x t}, entire), and
    // P = 2 rho^-n for the nearest singularity of the kernel's k-dependence,
    // the origin (the 1/(j omega eps) scale, the branch point of k_m), at
    // Bernstein radius rho. The constants are the calibration above.
    double phase = 2.0;
    for (int i = 1; i <= n; i++) phase *= 0.5 * x / i;
    return ANCHOR_BOUND_C_PHASE * phase + ANCHOR_BOUND_C_ORIGIN * 2.0 * std::pow(rho, -n);
}

Plan plan_of(const std::vector<double> &wavelengths, const Size &size) {
    Plan p;
    for (double wl : wavelengths) p.k.push_back(2.0 * M_PI / wl);  // `Front::build`'s k
    std::sort(p.k.begin(), p.k.end());
    p.k.erase(std::unique(p.k.begin(), p.k.end()), p.k.end());
    const int forced = anchor_nodes_knob();
    p.n_a = forced > 0 ? forced : ANCHOR_NODES_MIN;
    bool finite = std::isfinite(size.span) && size.span >= 0.0;
    for (double k : p.k) finite = finite && std::isfinite(k) && k > 0.0;
    if (p.k.empty()) {
        p.why = "";
        return p;
    }
    if (!finite) {
        p.why = "non-finite";
        return p;
    }
    if (static_cast<long long>(p.k.size()) <= p.n_a + 1) {
        p.why = "few-frequencies";
        return p;
    }
    const double k_lo = p.k.front(), k_hi = p.k.back();
    const double mid = 0.5 * (k_lo + k_hi), half = 0.5 * (k_hi - k_lo);
    if ((k_hi - k_lo) / mid > anchor_band_knob()) {
        p.why = "wide-band";
        return p;
    }
    p.x = half * size.span * std::max({1.0, km_slope(k_lo, size), km_slope(k_hi, size)});
    p.rho = (mid + std::sqrt(mid * mid - half * half)) / half;
    if (forced == 0) {
        int n = ANCHOR_NODES_MIN;
        while (n <= ANCHOR_NODES_MAX && bound(n, p.x, p.rho) > ANCHOR_BOUND_TARGET) n++;
        if (n > ANCHOR_NODES_MAX) {
            p.why = "electrically-large";
            return p;
        }
        p.n_a = n;
        if (static_cast<long long>(p.k.size()) <= p.n_a + 1) {
            p.why = "few-frequencies";
            return p;
        }
    }
    p.est = bound(p.n_a, p.x, p.rho);
    p.why = "ok";
    p.ok = true;
    const int na = p.n_a;
    for (int j = 0; j < na; j++) {
        // Lobatto nodes, descending: j = 0 is k_hi, j = na - 1 k_lo, both
        // the plan's own bits.
        p.anchor_k.push_back(j == 0 ? k_hi : j == na - 1 ? k_lo : mid + half * std::cos(M_PI * j / (na - 1)));
        p.bary_w.push_back(((j % 2) ? -1.0 : 1.0) * ((j == 0 || j == na - 1) ? 0.5 : 1.0));
    }
    return p;
}

bool in_plan(const Plan &p, double k) { return p.ok && std::binary_search(p.k.begin(), p.k.end(), k); }

std::vector<int64_t> entries(int64_t n, const std::vector<int64_t> &rows_a, const std::vector<int64_t> &rows_b,
                             const std::vector<int64_t> &tent_cols) {
    std::vector<int64_t> idx;
    idx.reserve(2 * rows_a.size() * rows_b.size() + tent_cols.size() * static_cast<size_t>(n));
    for (int64_t r : rows_a)
        for (int64_t c : rows_b) {
            idx.push_back(c * n + r);
            idx.push_back(r * n + c);
        }
    for (int64_t t : tent_cols)
        for (int64_t i = 0; i < n; i++) idx.push_back(t * n + i);
    std::sort(idx.begin(), idx.end());
    idx.erase(std::unique(idx.begin(), idx.end()), idx.end());
    return idx;
}

int apply(cd *z, const int64_t *idx, size_t m, const cd *raw, const Plan &p, double k) {
    const size_t na = p.anchor_k.size();
    for (size_t j = 0; j < na; j++)
        if (k == p.anchor_k[j]) {
            // On an anchor: its own entries, as computed.
            const cd *r = raw + j * m;
            for (size_t e = 0; e < m; e++) z[idx[e]] += r[e];
            return 0;
        }
    std::vector<double> lam(na);
    double sum = 0.0;
    for (size_t j = 0; j < na; j++) {
        lam[j] = p.bary_w[j] / (k - p.anchor_k[j]);
        sum += lam[j];
    }
    for (size_t j = 0; j < na; j++) lam[j] /= sum;
    for (size_t e = 0; e < m; e++) {
        cd v(0.0, 0.0);
        for (size_t j = 0; j < na; j++) v += lam[j] * raw[j * m + e];
        z[idx[e]] += v;
    }
    return 1;
}

}  // namespace anchors

namespace {

struct AnchorState {
    std::mutex mu;
    anchors::Plan plan;
    // The anchors, once built: the geometry they are of, the entries they
    // cover (column-major indices into Z, ascending) and each anchor's
    // values there (n_a x entries).
    bool built = false;
    Front key;
    std::vector<int64_t> idx;
    std::vector<cd> raw;
    AnchorCounters counters;
};

AnchorState &anchor_state() {
    static AnchorState s;
    return s;
}

// Each medium's rows, as `crossing_blocks_exact` forms them.
std::vector<int64_t> medium_rows(const Front &f, bool below) {
    const int64_t n = f.n_basis;
    std::vector<int64_t> rows;
    std::vector<char> mine(static_cast<size_t>(f.n_seg), 0);
    for (int64_t w = 0; w < f.n_wires; w++)
        if ((f.media_below[static_cast<size_t>(w)] != 0) == below) {
            for (int64_t g = f.seg_offsets[w]; g < f.seg_offsets[w + 1]; g++) mine[static_cast<size_t>(g)] = 1;
            for (int64_t m = f.basis_offsets[w]; m < f.basis_offsets[w + 1]; m++) rows.push_back(m);
        }
    for (int64_t m = f.basis_offsets.back(); m < n; m++)
        if (mine[static_cast<size_t>(f.wing_seg[2 * m])] || mine[static_cast<size_t>(f.wing_seg[2 * m + 1])])
            rows.push_back(m);
    return rows;
}

// Builds the anchors on `f`'s geometry (the caller holds the lock), in the
// plan's anchor order.
bool build_anchors(AnchorState &s, const Front &f, const std::vector<std::pair<int64_t, int>> &tents,
                   uintptr_t cancel_flag) {
    const int64_t n = f.n_basis;
    std::vector<int64_t> tent_cols;
    for (const auto &t : tents) tent_cols.push_back(t.first);
    std::vector<int64_t> idx = anchors::entries(n, medium_rows(f, false), medium_rows(f, true), tent_cols);
    const size_t na = s.plan.anchor_k.size();
    if (!anchors::fits(static_cast<int>(na), static_cast<int64_t>(idx.size()), n)) {
        // Too large to hold: the plan declines for the rest of the sweep.
        s.plan.ok = false;
        s.plan.why = "memory";
        return false;
    }
    std::vector<cd> raw(na * idx.size());
    // Each anchor decides on the crossing table's state this fill reached
    // the cross blocks with, as the exact route at that k would.
    const bool taken_entry = ni_eval::crossing_table_taken();
    std::vector<cd> scratch;
    for (size_t j = 0; j < na; j++) {
        Front fa;
        crossing_front_at(f, s.plan.anchor_k[j], fa);
        scratch.assign(static_cast<size_t>(n * n), cd(0.0, 0.0));
        {
            const ni_eval::CrossingEndsFirst state(taken_entry);
            crossing_blocks_exact(fa, scratch, tents, cancel_flag);
        }
        cd *out = raw.data() + j * idx.size();
        for (size_t e = 0; e < idx.size(); e++) out[e] = scratch[static_cast<size_t>(idx[e])];
        s.counters.anchors_built++;
    }
    s.key = Front();
    crossing_front_at(f, f.k, s.key);
    s.idx = std::move(idx);
    s.raw = std::move(raw);
    s.built = true;
    return true;
}

// MOMWIRE_CROSSING_ANCHOR_TRACE=1: one stderr line per fill the feature is
// on for, naming what it did (a gate's proof that the branch ran, on any
// route, the executables' included).
bool anchor_trace_on() {
    static const bool on = [] {
        const char *v = std::getenv("MOMWIRE_CROSSING_ANCHOR_TRACE");
        return v != nullptr && std::string(v) == "1";
    }();
    return on;
}

void anchor_trace(const char *what, const AnchorState &s) {
    if (!anchor_trace_on()) return;
    std::fprintf(stderr,
                 "crossing_anchors: %s plan=%s n_a=%d built=%lld interpolated=%lld copied=%lld declined=%lld "
                 "x=%.6g rho=%.6g est=%.3g\n",
                 what, s.plan.why.empty() ? "none" : s.plan.why.c_str(), s.plan.n_a, s.counters.anchors_built,
                 s.counters.interpolated, s.counters.copied, s.counters.declined, s.plan.x, s.plan.rho, s.plan.est);
}

bool same_vectors(const Front &a, const Front &b) {
    if (a.n_basis != b.n_basis || a.n_seg != b.n_seg || a.n_wires != b.n_wires) return false;
    if (a.basis_offsets != b.basis_offsets || a.seg_offsets != b.seg_offsets || a.media_below != b.media_below ||
        a.radius_per_wire != b.radius_per_wire || a.seg_p0 != b.seg_p0 || a.seg_t != b.seg_t ||
        a.seg_h != b.seg_h || a.wing_seg != b.wing_seg || a.wing_sigma != b.wing_sigma)
        return false;
    if (a.wing_rise.n != b.wing_rise.n) return false;
    for (size_t i = 0; i < a.wing_rise.n; i++)
        if (a.wing_rise.p[i] != b.wing_rise.p[i]) return false;
    // Bitwise on the scalars (NaN never reaches here: the front refuses it).
    return a.ground_z == b.ground_z && a.ground_eps_r_ == b.ground_eps_r_ && a.ground_sigma_ == b.ground_sigma_ &&
           a.c == b.c && a.buried == b.buried;
}

}  // namespace

bool same_crossing_geometry(const Front &a, const Front &b) { return same_vectors(a, b); }

void set_band_plan(const std::vector<double> &wavelengths, const anchors::Size &size) {
    AnchorState &s = anchor_state();
    std::lock_guard<std::mutex> lock(s.mu);
    anchors::Plan p = anchors::plan_of(wavelengths, size);
    if (p.k == s.plan.k && p.n_a == s.plan.n_a && p.ok == s.plan.ok) return;  // the same plan
    s.plan = std::move(p);
    s.built = false;
    s.key = Front();
    s.idx.clear();
    s.raw.clear();
    s.counters = AnchorCounters();
    s.counters.plan = s.plan.why;
}

AnchorCounters anchor_counters() {
    AnchorState &s = anchor_state();
    std::lock_guard<std::mutex> lock(s.mu);
    return s.counters;
}

void crossing_blocks(const Front &f, std::vector<cd> &z,
                     const std::vector<std::pair<int64_t, int>> &tents, uintptr_t cancel_flag) {
    if (!ni_eval::active_features().crossing_anchors) {
        crossing_blocks_exact(f, z, tents, cancel_flag);
        return;
    }
    AnchorState &s = anchor_state();
    std::unique_lock<std::mutex> lock(s.mu);
    const bool in = anchors::in_plan(s.plan, f.k);
    if (!in || (s.built && !same_vectors(s.key, f))) {
        s.counters.declined++;
        anchor_trace(!s.plan.ok ? "declined" : !in ? "outside-plan" : "other-geometry", s);
        lock.unlock();
        crossing_blocks_exact(f, z, tents, cancel_flag);
        return;
    }
    if (!s.built && !build_anchors(s, f, tents, cancel_flag)) {
        s.counters.declined++;
        anchor_trace("declined", s);
        lock.unlock();
        crossing_blocks_exact(f, z, tents, cancel_flag);
        return;
    }
    const int how = anchors::apply(z.data(), s.idx.data(), s.idx.size(), s.raw.data(), s.plan, f.k);
    (how == 0 ? s.counters.copied : s.counters.interpolated)++;
    anchor_trace(how == 0 ? "copied" : "interpolated", s);
}

}  // namespace razor
}  // namespace mw
