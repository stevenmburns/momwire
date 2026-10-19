// momwire#1517 step 4b-2: bs2's crossing serve in C++ -- see
// `_bspline_crossing_core.h`. Every step names the `_crossing_fill.py`
// function it replays.
#include "_bspline_crossing_core.h"

#include "_crossing_shared_inline.h"
#include "_fma_inline.h"
#include "_near_interface_core.h"
#include "_near_interface_eval.h"
#include "_razor_front_core.h"  // `leggauss`: `_quadrature.leggauss`'s frozen table
#include "_somm_grid_core.h"

#include <algorithm>
#include <array>
#include <limits>
#include <cmath>
#include <map>
#include <numeric>
#include <set>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace mw {
namespace bspline {

namespace {

namespace cx = mw::crossing;
using cd = std::complex<double>;
using nec5::replay::np_cmul;
using nec5::replay::np_cdiv;

constexpr double PLANE_TOL = 1e-12;        // `_PLANE_TOL`
constexpr double CORNER_RTOL = 1e-10;      // `_CORNER_RTOL`
constexpr double SAME_NODE_RHO = 1e-9;     // `_SAME_NODE_RHO`
constexpr int NEAR_Q = 4;                  // `_NEAR_Q`
constexpr double NEAR_GROWTH = 4.0;        // `_NEAR_GROWTH`
constexpr int NEAR_PANEL = 4;              // `_NEAR_GX, _NEAR_GW = leggauss(4)`
constexpr size_t CLUSTER_LEAF = 3;         // `_CLUSTER_LEAF_SEGS`
constexpr double ADM_ETA = 1.0;            // `_ADM_ETA`
constexpr double ACA_COST_GUARD = 48.0;    // `_ACA_COST_GUARD`
constexpr int64_t PAIR_GROUP_BUDGET = 600000;  // `_pair_groups(budget_pairs=)`
constexpr double ENDS_TO_NODES_MB = 48.0;  // `_ends_to_nodes(budget_mb=)`
constexpr int64_t ENDS_TO_NODES_BYTES_PER_PAIR = 112;
// `_near_interface`'s walk constants (`_sommerfeld_below`'s): the corner's
constexpr double LAM_MULT = 8.0;
constexpr int ADAPT_DEPTH = 16;
constexpr double DETOUR = 2.0;
constexpr int GAUSS_N = 24;

struct Raise {
    [[noreturn]] static void raise(const std::string &m) { throw FrontNotPorted("crossing-refusal: " + m); }
};

// CPython's complex product and quotient
inline cd py_prod(cd a, cd b) {
    return cd(a.real() * b.real() - a.imag() * b.imag(), a.real() * b.imag() + a.imag() * b.real());
}
inline cd py_quot(cd a, cd b) { return somm_core::py_quot(a, b); }
inline cd c_add(cd a, cd b) { return cd(a.real() + b.real(), a.imag() + b.imag()); }
inline cd c_sub(cd a, cd b) { return cd(a.real() - b.real(), a.imag() - b.imag()); }

// What the shared end terms read off the context.
struct XCtx {
    double gz = 0;
    cd c1;
};

int physical_threads() {
    const unsigned n = std::thread::hardware_concurrency();
    return n > 1 ? static_cast<int>(n / 2) : 1;
}

// ---------------------------------------------------------------------------
// The axes (`axis_data` with the B-spline basis's samples)

// `axis_data(ctx, seg_idx)` at the near knobs (q 4, panels G4, growth 4),
// `_basis_samples` / `_end_values` over the polynomials.
cx::Axis axis_of(const Geometry &g, const Basis &b, const std::vector<int64_t> &seg_idx, double gz, double a_wire) {
    const int nd = b.degree + 1;
    cx::Axis ax;
    ax.n_basis = b.n_basis;
    std::vector<double> xg, wg, gx, gw;
    razor::leggauss(NEAR_Q, xg, wg);
    razor::leggauss(NEAR_PANEL, gx, gw);
    std::vector<double> u_phys;
    std::map<int64_t, std::pair<int64_t, int64_t>> runs;
    for (int64_t s : seg_idx) {
        const double *sl = &g.seg_l[3 * static_cast<size_t>(s)];
        const double *sr = &g.seg_r[3 * static_cast<size_t>(s)];
        const double h = g.h_per_seg[static_cast<size_t>(s)];
        const double *tang = &g.tangents[3 * static_cast<size_t>(s)];
        const bool touch_lo = std::fabs(sl[2] - gz) < PLANE_TOL;
        const bool touch_hi = std::fabs(sr[2] - gz) < PLANE_TOL;
        std::vector<double> u, w;
        if (touch_lo || touch_hi) {
            cx::graded_u(h, !touch_lo, a_wire, NEAR_GROWTH, gx, gw, u, w);
        } else {
            for (size_t k = 0; k < xg.size(); k++) {
                const double tq = 0.5 * (xg[k] + 1.0);
                u.push_back(h * tq);
                w.push_back(0.5 * h * wg[k]);
            }
        }
        runs[s] = {static_cast<int64_t>(ax.w.size()), static_cast<int64_t>(u.size())};
        for (size_t k = 0; k < u.size(); k++) {
            const double t = u[k] / h;
            for (int q = 0; q < 3; q++) ax.nodes.push_back(sl[q] + t * (sr[q] - sl[q]));
            for (int q = 0; q < 3; q++) ax.t.push_back(tang[q]);
            ax.w.push_back(w[k]);
            ax.segof.push_back(s);
            u_phys.push_back(u[k]);
        }
    }
    // `_basis_samples`: every live wing at its segment's nodes; `seg_rows`
    std::vector<std::tuple<int64_t, int64_t, double>> tf, tfd;
    for (int64_t m = 0; m < b.n_basis; m++)
        for (int a = 0; a < nd; a++) {
            const double *c = &b.polys[static_cast<size_t>((m * nd + a) * nd)];
            bool live = false;
            for (int p = 0; p < nd; p++) live = live || c[p] != 0.0;
            if (!live) continue;
            const int64_t seg = b.supp_seg[static_cast<size_t>(m * nd + a)];
            auto it = runs.find(seg);
            if (it == runs.end()) continue;
            for (int64_t k = 0; k < it->second.second; k++) {
                const int64_t node = it->second.first + k;
                const double u = u_phys[static_cast<size_t>(node)];
                double fv = 0.0, fdv = 0.0;
                for (int p = 0; p < nd; p++) {
                    // `_upow` on an array: 1, u, u * u, then libm's pow
                    const double up = p == 0 ? 1.0 : p == 1 ? u : p == 2 ? u * u : std::pow(u, double(p));
                    fv = fv + c[p] * up;
                    if (p >= 1) {
                        const int q = p - 1;
                        const double uq = q == 0 ? 1.0 : q == 1 ? u : q == 2 ? u * u : std::pow(u, double(q));
                        fdv = fdv + (static_cast<double>(p) * c[p]) * uq;
                    }
                }
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
    ax.F = cx::to_csr(b.n_basis, n_nodes, tf);
    ax.Fd = cx::to_csr(b.n_basis, n_nodes, tfd);
    // the signed wire-end table (`_end_values`: Python float powers)
    std::set<int64_t> on_axis(seg_idx.begin(), seg_idx.end());
    for (size_t w = 0; w < g.per_wire.size(); w++) {
        const int64_t first = g.seg_offsets[w], last = g.seg_offsets[w + 1] - 1;
        if (!on_axis.count(first)) continue;
        for (int pass = 0; pass < 2; pass++) {
            const int64_t gs = pass == 0 ? first : last;
            const double hh = g.h_per_seg[static_cast<size_t>(gs)];
            const double u = pass == 0 ? 0.0 : hh;
            cx::End e;
            e.sign = pass == 0 ? -1.0 : 1.0;
            for (int q = 0; q < 3; q++) {
                const double l = g.seg_l[3 * static_cast<size_t>(gs) + q], r = g.seg_r[3 * static_cast<size_t>(gs) + q];
                e.pt[q] = l + (u / hh) * (r - l);
            }
            e.fv.assign(static_cast<size_t>(b.n_basis), 0.0);
            for (int a = 0; a < nd; a++)
                for (int64_t m = 0; m < b.n_basis; m++) {
                    if (b.supp_seg[static_cast<size_t>(m * nd + a)] != gs) continue;
                    const double *c = &b.polys[static_cast<size_t>((m * nd + a) * nd)];
                    bool live = false;
                    for (int p = 0; p < nd; p++) live = live || c[p] != 0.0;
                    if (!live) continue;
                    double acc = 0.0;
                    for (int p = 0; p < nd; p++) {
                        // a Python float's `u ** p`: 1.0, u, then CPython's pow
                        const double up = p == 0 ? 1.0 : p == 1 ? u : std::pow(u, double(p));
                        acc = acc + c[p] * up;
                    }
                    e.fv[static_cast<size_t>(m)] += acc;
                }
            bool any = false;
            for (double v : e.fv) any = any || v != 0.0;
            if (any) ax.ends.push_back(std::move(e));
        }
    }
    return ax;
}

// ---------------------------------------------------------------------------
// The admissibility partition (`_aca.build_cluster_tree`, `build_block_tree`)

struct Cluster {
    std::vector<int64_t> idx;  // positions into the axis's segment list
    double lo[3], hi[3];
    int left = -1, right = -1;
    double diam() const {
        const double x = hi[0] - lo[0], y = hi[1] - lo[1], z = hi[2] - lo[2];
        return std::sqrt((x * x + y * y) + z * z);
    }
};

// `np.median` of a copy: the middle value, or the mean of the middle two
double np_median(std::vector<double> v) {
    const size_t n = v.size();
    std::sort(v.begin(), v.end());
    if (n % 2 == 1) return v[n / 2] / 1.0;
    return (v[n / 2 - 1] + v[n / 2]) / 2.0;
}

int build_tree(std::vector<Cluster> &out, std::vector<int64_t> idx, const std::vector<double> &blo,
               const std::vector<double> &bhi, size_t leaf) {
    Cluster c;
    c.idx = idx;
    for (int q = 0; q < 3; q++) {
        c.lo[q] = blo[3 * static_cast<size_t>(idx[0]) + q];
        c.hi[q] = bhi[3 * static_cast<size_t>(idx[0]) + q];
        for (int64_t i : idx) {
            c.lo[q] = std::min(c.lo[q], blo[3 * static_cast<size_t>(i) + q]);
            c.hi[q] = std::max(c.hi[q], bhi[3 * static_cast<size_t>(i) + q]);
        }
    }
    const int me = static_cast<int>(out.size());
    out.push_back(c);
    if (idx.size() <= leaf) return me;
    int axis = 0;
    for (int q = 1; q < 3; q++)
        if (c.hi[q] - c.lo[q] > c.hi[axis] - c.lo[axis]) axis = q;
    std::vector<double> centers(idx.size());
    for (size_t k = 0; k < idx.size(); k++) {
        const size_t i = static_cast<size_t>(idx[k]);
        centers[k] = 0.5 * (blo[3 * i + axis] + bhi[3 * i + axis]);
    }
    const double split = np_median(centers);
    std::vector<char> left(idx.size());
    size_t n_left = 0;
    for (size_t k = 0; k < idx.size(); k++) n_left += (left[k] = centers[k] <= split);
    if (n_left == idx.size() || n_left == 0) {
        std::vector<size_t> order(idx.size());
        std::iota(order.begin(), order.end(), 0);
        std::stable_sort(order.begin(), order.end(), [&](size_t a, size_t b2) { return centers[a] < centers[b2]; });
        std::fill(left.begin(), left.end(), 0);
        for (size_t k = 0; k < idx.size() / 2; k++) left[order[k]] = 1;
    }
    std::vector<int64_t> li, ri;
    for (size_t k = 0; k < idx.size(); k++) (left[k] ? li : ri).push_back(idx[k]);
    const int l = build_tree(out, li, blo, bhi, leaf);
    const int r = build_tree(out, ri, blo, bhi, leaf);
    out[static_cast<size_t>(me)].left = l;
    out[static_cast<size_t>(me)].right = r;
    return me;
}

// `_axis_segment_tree`: boxes are the segments' endpoint extents
std::vector<Cluster> segment_tree(const Geometry &g, const std::vector<int64_t> &seg_idx) {
    std::vector<double> lo(3 * seg_idx.size()), hi(3 * seg_idx.size());
    for (size_t k = 0; k < seg_idx.size(); k++)
        for (int q = 0; q < 3; q++) {
            const double l = g.seg_l[3 * static_cast<size_t>(seg_idx[k]) + q],
                         r = g.seg_r[3 * static_cast<size_t>(seg_idx[k]) + q];
            lo[3 * k + q] = std::min(l, r);
            hi[3 * k + q] = std::max(l, r);
        }
    std::vector<int64_t> all(seg_idx.size());
    std::iota(all.begin(), all.end(), 0);
    std::vector<Cluster> out;
    build_tree(out, all, lo, hi, CLUSTER_LEAF);
    return out;
}

bool admissible(const Cluster &a, const Cluster &b) {
    double dd[3];
    for (int q = 0; q < 3; q++) dd[q] = std::max(std::max(a.lo[q] - b.hi[q], b.lo[q] - a.hi[q]), 0.0);
    const double d = std::sqrt((dd[0] * dd[0] + dd[1] * dd[1]) + dd[2] * dd[2]);
    if (d == 0.0) return false;
    return std::min(a.diam(), b.diam()) <= ADM_ETA * d;
}

void block_tree(const std::vector<Cluster> &S, const std::vector<Cluster> &T,
                std::vector<std::pair<int, int>> &far, std::vector<std::pair<int, int>> &near) {
    std::vector<std::pair<int, int>> stack{{0, 0}};
    while (!stack.empty()) {
        const auto pr = stack.back();
        stack.pop_back();
        const Cluster &cs = S[static_cast<size_t>(pr.first)], &ct = T[static_cast<size_t>(pr.second)];
        const bool s_leaf = cs.left < 0, t_leaf = ct.left < 0;
        if (admissible(cs, ct)) {
            far.push_back(pr);
        } else if (s_leaf && t_leaf) {
            near.push_back(pr);
        } else if (s_leaf) {
            stack.push_back({pr.first, ct.left});
            stack.push_back({pr.first, ct.right});
        } else if (t_leaf) {
            stack.push_back({cs.left, pr.second});
            stack.push_back({cs.right, pr.second});
        } else {
            stack.push_back({cs.left, ct.left});
            stack.push_back({cs.left, ct.right});
            stack.push_back({cs.right, ct.left});
            stack.push_back({cs.right, ct.right});
        }
    }
}

// `_nodes_of(ax, segs)`: the nodes of `segs`, ascending
std::vector<int64_t> nodes_of(const cx::Axis &ax, const std::map<int64_t, std::pair<int64_t, int64_t>> &runs,
                              const std::vector<int64_t> &segs) {
    std::vector<int64_t> out;
    for (int64_t s : segs) {
        auto it = runs.find(s);
        if (it == runs.end()) continue;
        for (int64_t k = 0; k < it->second.second; k++) out.push_back(it->second.first + k);
    }
    std::sort(out.begin(), out.end());
    (void)ax;
    return out;
}

std::map<int64_t, std::pair<int64_t, int64_t>> runs_of(const cx::Axis &ax) {
    std::map<int64_t, std::pair<int64_t, int64_t>> runs;
    for (size_t k = 0; k < ax.segof.size(); k++) {
        auto it = runs.find(ax.segof[k]);
        if (it == runs.end())
            runs[ax.segof[k]] = {static_cast<int64_t>(k), 1};
        else
            it->second.second++;
    }
    return runs;
}

// `_support_rows(ax, ii)`: the rows with a live wing on ii's segments
std::vector<int64_t> support_rows_of(const cx::Axis &ax, const std::vector<int64_t> &ii) {
    std::set<int64_t> segs, rows;
    for (int64_t i : ii) segs.insert(ax.segof[static_cast<size_t>(i)]);
    for (int64_t s : segs) {
        auto it = ax.seg_rows.find(s);
        if (it != ax.seg_rows.end()) rows.insert(it->second.begin(), it->second.end());
    }
    return std::vector<int64_t>(rows.begin(), rows.end());
}

// `_row_weights(ax, ii, rows)`: (F w tx, F w ty, F w tz, Fd w) on rows x ii,
// the columns renumbered to positions in ii
std::array<cx::CSR, 4> row_weights_of(const cx::Axis &ax, const std::vector<int64_t> &ii,
                                       const std::vector<int64_t> &rows) {
    std::vector<int64_t> pos(ax.w.size(), -1);
    for (size_t k = 0; k < ii.size(); k++) pos[static_cast<size_t>(ii[k])] = static_cast<int64_t>(k);
    std::array<cx::CSR, 4> out;
    for (int m = 0; m < 4; m++) {
        cx::CSR &o = out[static_cast<size_t>(m)];
        o.n_rows = static_cast<int64_t>(rows.size());
        o.n_cols = static_cast<int64_t>(ii.size());
        o.indptr.assign(rows.size() + 1, 0);
        const cx::CSR &src = m < 3 ? ax.F : ax.Fd;
        for (size_t i = 0; i < rows.size(); i++) {
            const int64_t r = rows[i];
            for (int64_t jj = src.indptr[static_cast<size_t>(r)]; jj < src.indptr[static_cast<size_t>(r) + 1]; jj++) {
                const int64_t col = src.indices[static_cast<size_t>(jj)];
                const int64_t p = pos[static_cast<size_t>(col)];
                if (p < 0) continue;
                double v = src.data[static_cast<size_t>(jj)] * ax.w[static_cast<size_t>(col)];
                if (m < 3) v = v * ax.t[3 * static_cast<size_t>(col) + m];
                o.indices.push_back(p);
                o.data.push_back(v);
            }
            o.indptr[i + 1] = static_cast<int64_t>(o.indices.size());
        }
    }
    return out;
}

// One direct spec: (iA, iB) node subsets of the two axes.
struct Spec {
    std::vector<int64_t> iA, iB;
    bool coarse = false;  // a small far block, on the coarse axes
};

// `_sandwich_dense(A, B, iA, iB, K, k2sq, out=t)`: the left products and the
// five-term combine over the spec's tables (memo indices `K`, row a * |iB| +
// b), added into t at (rA, rB).
void sandwich(const cx::Axis &A, const cx::Axis &B, const Spec &sp, const uint32_t *K,
              const std::vector<cd> &KU, const std::vector<cd> &KX, const std::vector<cd> &KW,
              const std::vector<cd> &KV, std::vector<cd> &t) {
    const size_t nB = sp.iB.size();
    const std::vector<int64_t> rA = support_rows_of(A, sp.iA), rB = support_rows_of(B, sp.iB);
    const std::array<cx::CSR, 4> P = row_weights_of(A, sp.iA, rA), Q = row_weights_of(B, sp.iB, rB);
    static const int qsel[6] = {0, 1, 2, 3, 2, 3};
    std::vector<cd> Lrow(6 * nB);
    for (size_t i = 0; i < rA.size(); i++) {
        std::fill(Lrow.begin(), Lrow.end(), cd(0.0, 0.0));
        cd *L[6];
        for (int m = 0; m < 6; m++) L[m] = &Lrow[static_cast<size_t>(m) * nB];
        for (int pm = 0; pm < 3; pm++) {
            const cx::CSR &M = P[static_cast<size_t>(pm)];
            for (int64_t jj = M.indptr[i]; jj < M.indptr[i + 1]; jj++) {
                const double a = M.data[static_cast<size_t>(jj)];
                const uint32_t *kx = &K[static_cast<size_t>(M.indices[static_cast<size_t>(jj)]) * nB];
                if (pm < 2) {
                    cx::axpy_gather(L[pm], KU.data(), kx, a, nB);
                } else {
                    cx::axpy_gather(L[2], KX.data(), kx, a, nB);
                    cx::axpy_gather(L[3], KW.data(), kx, a, nB);
                }
            }
        }
        const cx::CSR &M = P[3];
        for (int64_t jj = M.indptr[i]; jj < M.indptr[i + 1]; jj++) {
            const double a = M.data[static_cast<size_t>(jj)];
            const uint32_t *kx = &K[static_cast<size_t>(M.indices[static_cast<size_t>(jj)]) * nB];
            cx::axpy_gather(L[4], KW.data(), kx, a, nB);
            cx::axpy_gather(L[5], KV.data(), kx, a, nB);
        }
        for (size_t j = 0; j < rB.size(); j++) {
            double tr[6], ti[6];
            for (int m = 0; m < 6; m++) {
                const cx::CSR &Mq = Q[static_cast<size_t>(qsel[m])];
                cx::csr_row_dot(Mq.indices.data(), Mq.data.data(), Mq.indptr[j], Mq.indptr[j + 1],
                                reinterpret_cast<const double *>(L[m]), tr[m], ti[m]);
            }
            const cd v(((((tr[0] + tr[1]) + tr[2]) + tr[3]) + tr[4]) - tr[5],
                       ((((ti[0] + ti[1]) + ti[2]) + ti[3]) + ti[4]) - ti[5]);
            cd &e = t[static_cast<size_t>(rA[i] * B.n_basis + rB[j])];
            e = c_add(e, v);
        }
    }
}

// numpy's complex absolute on an array (AVX2/AVX512F `simd_cabsolute`):
// larger * sqrt(fma(r, r, 1)), r = smaller / larger
inline double np_cabs(cd z) {
    const double re = std::fabs(z.real()), im = std::fabs(z.imag());
    if (std::isnan(re) || std::isnan(im)) return std::numeric_limits<double>::quiet_NaN();
    if (std::isinf(re) || std::isinf(im)) return std::numeric_limits<double>::infinity();
    const double larger = std::max(re, im), smaller = std::min(im, re);
    if (larger == 0.0) return larger;
    const double r = smaller / larger;
    return std::sqrt(mw_fma::exact(r, r, 1.0)) * larger;
}

// `float(np.linalg.norm(u))` for a complex vector: the real and imaginary
// parts' ddot at stride 2, added, rooted
double np_cnorm(const nec5::NumpyLinalg &np, const std::vector<cd> &u) {
    const double *p = reinterpret_cast<const double *>(u.data());
    const int64_t n = static_cast<int64_t>(u.size());
    const double sq = np.ddot(n, p, 2, p, 2) + np.ddot(n, p + 1, 2, p + 1, 2);
    return std::sqrt(sq);
}

// `np.vdot(a, b)`
cd np_vdot(const nec5::NumpyLinalg &np, const std::vector<cd> &a, const std::vector<cd> &b) {
    cd out(0.0, 0.0);
    np.zdotc_sub(static_cast<int64_t>(a.size()), a.data(), 1, b.data(), 1, &out);
    return out;
}

// `_aca.aca_partial(get_row, get_col, m, n, tol)`: U (m each), V (n each)
template <class RowF, class ColF>
void aca_partial(const nec5::NumpyLinalg &np, RowF get_row, ColF get_col, int64_t m, int64_t n, double tol,
                 std::vector<std::vector<cd>> &U, std::vector<std::vector<cd>> &V) {
    const int64_t max_rank = std::min(m, n);
    std::vector<char> used_rows(static_cast<size_t>(m), 0), used_cols(static_cast<size_t>(n), 0);
    double approx = 0.0;
    int64_t i_star = 0;
    for (int64_t it = 0; it < max_rank; it++) {
        std::vector<cd> row = get_row(i_star);
        const size_t rank = U.size();
        for (size_t k = 0; k < rank; k++) {
            const cd a = U[k][static_cast<size_t>(i_star)];
            for (int64_t j = 0; j < n; j++) row[static_cast<size_t>(j)] = c_sub(row[static_cast<size_t>(j)], np_cmul(a, V[k][static_cast<size_t>(j)]));
        }
        used_rows[static_cast<size_t>(i_star)] = 1;
        int64_t j_star = 0;
        double best = -std::numeric_limits<double>::infinity();
        bool have = false;
        for (int64_t j = 0; j < n; j++) {
            const double v = used_cols[static_cast<size_t>(j)] ? -1.0 : np_cabs(row[static_cast<size_t>(j)]);
            if (!have || v > best || (std::isnan(v) && !std::isnan(best))) {
                if (have && std::isnan(best)) continue;
                best = v;
                j_star = j;
                have = true;
            }
        }
        const cd delta = row[static_cast<size_t>(j_star)];
        if (std::hypot(delta.real(), delta.imag()) < 1e-300 || best <= 0.0) {
            int64_t rem = -1;
            for (int64_t i = 0; i < m; i++)
                if (!used_rows[static_cast<size_t>(i)]) {
                    rem = i;
                    break;
                }
            if (rem < 0) break;
            i_star = rem;
            continue;
        }
        std::vector<cd> v(static_cast<size_t>(n));
        for (int64_t j = 0; j < n; j++) v[static_cast<size_t>(j)] = np_cdiv(row[static_cast<size_t>(j)], delta);
        std::vector<cd> col = get_col(j_star);
        for (size_t k = 0; k < rank; k++) {
            const cd a = V[k][static_cast<size_t>(j_star)];
            for (int64_t i = 0; i < m; i++) col[static_cast<size_t>(i)] = c_sub(col[static_cast<size_t>(i)], np_cmul(a, U[k][static_cast<size_t>(i)]));
        }
        used_cols[static_cast<size_t>(j_star)] = 1;
        const double un = np_cnorm(np, col), vn = np_cnorm(np, v);
        double cross = 0.0;
        for (size_t k = 0; k < rank; k++) cross += py_prod(np_vdot(np, U[k], col), np_vdot(np, V[k], v)).real();
        approx += 2.0 * cross + std::pow(un * vn, 2.0);
        U.push_back(col);
        V.push_back(std::move(v));
        if (approx <= 0.0 || un * vn <= tol * std::sqrt(approx)) break;
        bool any_unused = false;
        for (int64_t i = 0; i < m; i++) any_unused = any_unused || !used_rows[static_cast<size_t>(i)];
        if (!any_unused) break;
        int64_t i_next = 0;
        best = -std::numeric_limits<double>::infinity();
        have = false;
        for (int64_t i = 0; i < m; i++) {
            const double a = used_rows[static_cast<size_t>(i)] ? -1.0 : np_cabs(U.back()[static_cast<size_t>(i)]);
            if (!have || a > best || (std::isnan(a) && !std::isnan(best))) {
                if (have && std::isnan(best)) continue;
                best = a;
                i_next = i;
                have = true;
            }
        }
        i_star = i_next;
    }
}

// One far block (`_main_split`'s ACA branch): per kernel a low-rank
// factorisation over the block's designed rows / columns (sampled once,
// through the fill's memo, cached across the four kernels), then the
// seven terms `P_i @ U_f @ V_f @ Q_j.T` added into t at (rA, rB).
void far_block(const nec5::NumpyLinalg &np, const cx::Axis &A, const cx::Axis &B, const Spec &sp, cx::BlockEval &be,
               double gz, double a_wire, double k2sq, std::vector<cd> &t) {
    const int64_t m = static_cast<int64_t>(sp.iA.size()), n = static_cast<int64_t>(sp.iB.size());
    std::vector<double> zA(static_cast<size_t>(m)), zB(static_cast<size_t>(n)), rho(static_cast<size_t>(m * n));
    for (int64_t i = 0; i < m; i++) zA[static_cast<size_t>(i)] = A.nodes[3 * static_cast<size_t>(sp.iA[static_cast<size_t>(i)]) + 2] - gz;
    for (int64_t j = 0; j < n; j++) zB[static_cast<size_t>(j)] = B.nodes[3 * static_cast<size_t>(sp.iB[static_cast<size_t>(j)]) + 2] - gz;
    for (int64_t i = 0; i < m; i++)
        for (int64_t j = 0; j < n; j++) {
            const double *pa = &A.nodes[3 * static_cast<size_t>(sp.iA[static_cast<size_t>(i)])];
            const double *pb = &B.nodes[3 * static_cast<size_t>(sp.iB[static_cast<size_t>(j)])];
            rho[static_cast<size_t>(i * n + j)] = std::hypot(pa[0] - pb[0], pa[1] - pb[1]);
        }
    static const int KEYS[4] = {cx::K_U, cx::K_V, cx::K_W, cx::K_DZPW};
    std::map<int64_t, std::array<std::vector<cd>, 4>> row_s, col_s;
    auto sample = [&](const std::vector<double> &r, const std::vector<double> &z, const std::vector<double> &zp) {
        std::vector<cx::Six> te;
        cx::tables(be, a_wire, r, z, zp, nullptr, te);
        std::array<std::vector<cd>, 4> out;
        for (int q = 0; q < 4; q++) {
            out[static_cast<size_t>(q)].resize(te.size());
            for (size_t k = 0; k < te.size(); k++) out[static_cast<size_t>(q)][k] = te[k][static_cast<size_t>(KEYS[q])];
        }
        return out;
    };
    auto row6 = [&](int64_t i) -> const std::array<std::vector<cd>, 4> & {
        auto it = row_s.find(i);
        if (it != row_s.end()) return it->second;
        std::vector<double> r(rho.begin() + i * n, rho.begin() + (i + 1) * n), z(static_cast<size_t>(n), zA[static_cast<size_t>(i)]);
        return row_s[i] = sample(r, z, zB);
    };
    auto col6 = [&](int64_t j) -> const std::array<std::vector<cd>, 4> & {
        auto it = col_s.find(j);
        if (it != col_s.end()) return it->second;
        std::vector<double> r(static_cast<size_t>(m)), zp(static_cast<size_t>(m), zB[static_cast<size_t>(j)]);
        for (int64_t i = 0; i < m; i++) r[static_cast<size_t>(i)] = rho[static_cast<size_t>(i * n + j)];
        return col_s[j] = sample(r, zA, zp);
    };
    // Uf (m, r) and Vf (r, n), C order, per kernel
    std::array<std::vector<cd>, 4> Uf, Vf;
    std::array<int64_t, 4> rk{};
    for (int q = 0; q < 4; q++) {
        std::vector<std::vector<cd>> U, V;
        aca_partial(np, [&](int64_t i) { return row6(i)[static_cast<size_t>(q)]; },
                    [&](int64_t j) { return col6(j)[static_cast<size_t>(q)]; }, m, n, 1e-7, U, V);
        const int64_t r = static_cast<int64_t>(U.size());
        rk[static_cast<size_t>(q)] = r;
        Uf[static_cast<size_t>(q)].assign(static_cast<size_t>(m * r), cd(0.0, 0.0));
        Vf[static_cast<size_t>(q)].assign(static_cast<size_t>(r * n), cd(0.0, 0.0));
        for (int64_t k = 0; k < r; k++) {
            for (int64_t i = 0; i < m; i++) Uf[static_cast<size_t>(q)][static_cast<size_t>(i * r + k)] = U[static_cast<size_t>(k)][static_cast<size_t>(i)];
            for (int64_t j = 0; j < n; j++) Vf[static_cast<size_t>(q)][static_cast<size_t>(k * n + j)] = V[static_cast<size_t>(k)][static_cast<size_t>(j)];
        }
    }
    const std::vector<int64_t> rA = support_rows_of(A, sp.iA), rB = support_rows_of(B, sp.iB);
    const std::array<cx::CSR, 4> Pw = row_weights_of(A, sp.iA, rA), Qw = row_weights_of(B, sp.iB, rB);
    const int64_t nrA = static_cast<int64_t>(rA.size()), nrB = static_cast<int64_t>(rB.size());
    std::map<std::pair<int, int>, std::vector<cd>> PU, VQ;  // VQ held as (|rB|, r): its transpose
    auto lr = [&](int pi, int q, int qi) {
        const int64_t r = rk[static_cast<size_t>(q)];
        auto pk = std::make_pair(pi, q);
        if (!PU.count(pk)) {
            // `Pw[pi] @ Uf`: scipy's csr_matvecs, the data promoted to complex
            std::vector<cd> y(static_cast<size_t>(nrA * r), cd(0.0, 0.0));
            const cx::CSR &M = Pw[static_cast<size_t>(pi)];
            for (int64_t i = 0; i < nrA; i++)
                for (int64_t jj = M.indptr[static_cast<size_t>(i)]; jj < M.indptr[static_cast<size_t>(i) + 1]; jj++) {
                    const double d = M.data[static_cast<size_t>(jj)];
                    const cd *x = &Uf[static_cast<size_t>(q)][static_cast<size_t>(M.indices[static_cast<size_t>(jj)] * r)];
                    for (int64_t k = 0; k < r; k++) {
                        cd &e = y[static_cast<size_t>(i * r + k)];
                        e = cd(e.real() + d * x[k].real(), e.imag() + d * x[k].imag());
                    }
                }
            PU[pk] = std::move(y);
        }
        auto vk = std::make_pair(q, qi);
        if (!VQ.count(vk)) {
            // `Vf @ Qw[qi].T` = (Qw[qi] @ Vf.T).T
            std::vector<cd> y(static_cast<size_t>(nrB * r), cd(0.0, 0.0));
            const cx::CSR &M = Qw[static_cast<size_t>(qi)];
            for (int64_t j = 0; j < nrB; j++)
                for (int64_t jj = M.indptr[static_cast<size_t>(j)]; jj < M.indptr[static_cast<size_t>(j) + 1]; jj++) {
                    const double d = M.data[static_cast<size_t>(jj)];
                    const int64_t c = M.indices[static_cast<size_t>(jj)];
                    for (int64_t k = 0; k < r; k++) {
                        const cd x = Vf[static_cast<size_t>(q)][static_cast<size_t>(k * n + c)];
                        cd &e = y[static_cast<size_t>(j * r + k)];
                        e = cd(e.real() + d * x.real(), e.imag() + d * x.imag());
                    }
                }
            VQ[vk] = std::move(y);
        }
        std::vector<cd> out(static_cast<size_t>(nrA * nrB));
        const std::vector<cd> &L = PU[pk], &R = VQ[vk];
        nec5::replay::np_matmul(np, nec5::Op{L.data(), nrA, r, r, 1}, nec5::Op{R.data(), r, nrB, 1, r}, out.data(), nrB, 1);
        return out;
    };
    // `_terms()`: U, U, k2 V, dzpW, W, W, -V
    const std::vector<cd> t0 = lr(0, 0, 0), t1 = lr(1, 0, 1), t2 = lr(2, 1, 2), t3 = lr(2, 3, 2), t4 = lr(2, 2, 3),
                          t5 = lr(3, 2, 2), t6 = lr(3, 1, 3);
    for (int64_t i = 0; i < nrA; i++)
        for (int64_t j = 0; j < nrB; j++) {
            const size_t q = static_cast<size_t>(i * nrB + j);
            cd s = c_add(t0[q], t1[q]);
            s = c_add(s, cd(k2sq * t2[q].real(), k2sq * t2[q].imag()));
            s = c_add(s, t3[q]);
            s = c_add(s, t4[q]);
            s = c_add(s, t5[q]);
            s = c_sub(s, t6[q]);
            cd &e = t[static_cast<size_t>(rA[static_cast<size_t>(i)] * B.n_basis + rB[static_cast<size_t>(j)])];
            e = c_add(e, s);
        }
}

// `_main_split(ctx, a_idx, b_idx, A, B, ...)` without the restriction: the
// partition, the direct groups through one memo, c1 * t_main.
// `Ac` / `Bc` are the coarse axes (`axis_data(coarse=True)`): the dense ones
// at one radius, rebuilt at the block's radius at a two-radius node.
std::vector<cd> main_split(const Geometry &g, const cx::Axis &A, const cx::Axis &B, const cx::Axis &Ac,
                           const cx::Axis &Bc, const std::vector<int64_t> &a_idx, const std::vector<int64_t> &b_idx,
                           cx::BlockEval &be, double gz, double k_p, cd c1, double a_wire,
                           const nec5::NumpyLinalg &np) {
    const std::vector<Cluster> TA = segment_tree(g, a_idx), TB = segment_tree(g, b_idx);
    std::vector<std::pair<int, int>> far, near;
    block_tree(TA, TB, far, near);
    const auto runA = runs_of(A), runB = runs_of(B);
    const auto runAc = runs_of(Ac), runBc = runs_of(Bc);
    auto segs_of = [](const Cluster &c, const std::vector<int64_t> &idx) {
        std::vector<int64_t> s;
        for (int64_t i : c.idx) s.push_back(idx[static_cast<size_t>(i)]);
        return s;
    };
    std::vector<Spec> direct, far_aca;
    for (const auto &pr : near) {
        Spec sp{nodes_of(A, runA, segs_of(TA[static_cast<size_t>(pr.first)], a_idx)),
                nodes_of(B, runB, segs_of(TB[static_cast<size_t>(pr.second)], b_idx))};
        if (!sp.iA.empty() && !sp.iB.empty()) direct.push_back(std::move(sp));
    }
    // the far blocks ride the coarse axes, the same sampling as the dense
    // ones at these knobs (`axis_data(coarse=True, share_from=)`)
    for (const auto &pr : far) {
        Spec sp{nodes_of(Ac, runAc, segs_of(TA[static_cast<size_t>(pr.first)], a_idx)),
                nodes_of(Bc, runBc, segs_of(TB[static_cast<size_t>(pr.second)], b_idx))};
        sp.coarse = true;
        if (sp.iA.empty() || sp.iB.empty()) continue;
        const double na = static_cast<double>(sp.iA.size()), nb = static_cast<double>(sp.iB.size());
        if (na * nb > ACA_COST_GUARD * (na + nb))
            far_aca.push_back(std::move(sp));
        else
            direct.push_back(std::move(sp));
    }
    // `_plan_sheets`: the fill's sheets from the direct specs' node pairs
    {
        std::vector<std::vector<double>> keep;
        std::vector<ni_eval::NodePair> pairs;
        for (const Spec &sp : direct) {
            const cx::Axis &SA = sp.coarse ? Ac : A, &SB = sp.coarse ? Bc : B;
            std::vector<double> pa, pb;
            for (int64_t i : sp.iA)
                for (int q = 0; q < 3; q++) pa.push_back(SA.nodes[3 * static_cast<size_t>(i) + q] - (q == 2 ? gz : 0.0));
            for (int64_t i : sp.iB)
                for (int q = 0; q < 3; q++) pb.push_back(SB.nodes[3 * static_cast<size_t>(i) + q] - (q == 2 ? gz : 0.0));
            keep.push_back(std::move(pa));
            keep.push_back(std::move(pb));
        }
        for (size_t k = 0; k < direct.size(); k++)
            pairs.push_back({keep[2 * k].data(), direct[k].iA.size(), keep[2 * k + 1].data(), direct[k].iB.size()});
        be.memo.plan = ni_eval::sheet_plan(k_p, be.ev.k_m, pairs, a_wire);
        be.memo.have_plan = true;
    }
    std::vector<cd> t(static_cast<size_t>(A.n_basis * B.n_basis), cd(0.0, 0.0));
    const double k2sq = k_p * k_p;
    // `_pair_groups`: consecutive specs under the pair budget
    size_t start = 0;
    while (start < direct.size()) {
        size_t stop = start;
        int64_t run = 0;
        while (stop < direct.size()) {
            const int64_t n = static_cast<int64_t>(direct[stop].iA.size() * direct[stop].iB.size());
            if (run && run + n > PAIR_GROUP_BUDGET) break;
            run += n;
            stop++;
        }
        // `_direct_group`: one table call over the group's pairs
        std::vector<double> re, z, zp;
        for (size_t k = start; k < stop; k++) {
            const Spec &sp = direct[k];
            const cx::Axis &SA = sp.coarse ? Ac : A, &SB = sp.coarse ? Bc : B;
            for (int64_t ia : sp.iA)
                for (int64_t ib : sp.iB) {
                    const double *pa = &SA.nodes[3 * static_cast<size_t>(ia)];
                    const double *pb = &SB.nodes[3 * static_cast<size_t>(ib)];
                    const double rho = std::hypot(pa[0] - pb[0], pa[1] - pb[1]);
                    re.push_back(std::hypot(rho, a_wire));
                    z.push_back(pa[2] - gz);
                    zp.push_back(pb[2] - gz);
                }
        }
        std::vector<uint32_t> K;
        cx::table_index(be, re, z, zp, nullptr, K);
        const std::vector<cd> KU = cx::key_plane(be.memo, cx::K_U), KV = cx::key_plane(be.memo, cx::K_V),
                              KW = cx::key_plane(be.memo, cx::K_W), KD = cx::key_plane(be.memo, cx::K_DZPW);
        std::vector<cd> KX(KV.size());
        for (size_t j = 0; j < KX.size(); j++)
            KX[j] = cd(k2sq * KV[j].real() + KD[j].real(), k2sq * KV[j].imag() + KD[j].imag());
        size_t off = 0;
        for (size_t k = start; k < stop; k++) {
            sandwich(direct[k].coarse ? Ac : A, direct[k].coarse ? Bc : B, direct[k], &K[off], KU, KX, KW, KV, t);
            off += direct[k].iA.size() * direct[k].iB.size();
        }
        start = stop;
    }
    // the large far blocks: coarse axes, low-rank ACA per kernel
    if (!far_aca.empty() && (np.ddot == nullptr || np.zdotc_sub == nullptr))
        throw FrontNotPorted("no BLAS for the crossing ACA");
    for (const Spec &sp : far_aca) far_block(np, Ac, Bc, sp, be, gz, a_wire, k2sq, t);
    for (cd &v : t) v = np_cmul(c1, v);
    return t;
}

// `_corner_terms`: every in-plane end pair, -sig_a sig_b c1 V(rho_eff) times
// the outer product of the two ends' values, into E_r.
struct Corner {
    const cx::Axis *R, *C;
    double gz, a_wire, k_p;
    cd k_m, c1;
    void operator()(std::vector<cd> &E_r, const std::vector<int64_t> &posLA, int64_t nC) const {
        std::map<double, cd> v_at;
        std::vector<double> gx, gw;
        razor::leggauss(GAUSS_N, gx, gw);
        auto corner_v = [&](double rho) {
            const double key = rho < SAME_NODE_RHO ? 0.0 : rho;
            auto it = v_at.find(key);
            if (it != v_at.end()) return it->second;
            const double r_eff = key == 0.0 ? a_wire : std::hypot(rho, a_wire);
            const double z0 = 0.0, zp0 = 0.0;
            std::complex<double> six[6];
            ni_core::six_batch(k_p, k_m, &r_eff, &z0, &zp0, 1, CORNER_RTOL, LAM_MULT, ADAPT_DEPTH, DETOUR, gx.data(),
                               gw.data(), GAUSS_N, six);
            v_at[key] = six[1];
            return six[1];
        };
        for (const cx::End &ea : R->ends) {
            if (std::fabs(ea.pt[2] - gz) > 1e-12) continue;
            for (const cx::End &eb : C->ends) {
                if (std::fabs(eb.pt[2] - gz) > 1e-12) continue;
                const double rho = std::hypot(ea.pt[0] - eb.pt[0], ea.pt[1] - eb.pt[1]);
                const cd v = corner_v(rho);
                const cd scale = py_prod(py_prod(cd((-ea.sign) * eb.sign, 0.0), c1), v);
                for (size_t i = 0; i < ea.fv.size(); i++) {
                    if (ea.fv[i] == 0.0) continue;
                    cd *row = &E_r[static_cast<size_t>(posLA[i]) * static_cast<size_t>(nC)];
                    for (size_t j = 0; j < eb.fv.size(); j++) {
                        if (eb.fv[j] == 0.0) continue;
                        row[j] = c_add(row[j], np_cmul(scale, cd(ea.fv[i] * eb.fv[j], 0.0)));
                    }
                }
            }
        }
    }
};

// `_g_of_r(k, R)` for one R: exp(-1j k R) / R
inline cd g_of_r(cd mjk, double R) { return np_cdiv(std::exp(np_cmul(mjk, cd(R, 0.0))), cd(R, 0.0)); }

// `_ends_to_nodes(k, a2, obs, src, Fsp)`: (E, n) row-major
std::vector<cd> ends_to_nodes(cd mjk, double a2, const std::vector<double> &obs, const std::vector<double> &src,
                              const cx::CSR &Fsp) {
    const int64_t E = static_cast<int64_t>(obs.size() / 3), P = static_cast<int64_t>(src.size() / 3), n = Fsp.n_rows;
    const int64_t per_node = std::max<int64_t>(E * ENDS_TO_NODES_BYTES_PER_PAIR, 1);
    const int64_t block = std::max<int64_t>(
        1, std::min<int64_t>(P, static_cast<int64_t>(std::floor(ENDS_TO_NODES_MB * (1 << 20) / static_cast<double>(per_node)))));
    auto Ge = [&](int64_t e, int64_t p) {
        const double d0 = src[3 * p] - obs[3 * e], d1 = src[3 * p + 1] - obs[3 * e + 1], d2 = src[3 * p + 2] - obs[3 * e + 2];
        return g_of_r(mjk, std::sqrt(a2 + ((d0 * d0 + d2 * d2) + d1 * d1)));
    };
    std::vector<cd> Gt(static_cast<size_t>(E * n), cd(0.0, 0.0));
    if (block >= P) {
        std::vector<cd> G(static_cast<size_t>(E * P));
        for (int64_t e = 0; e < E; e++)
            for (int64_t p = 0; p < P; p++) G[static_cast<size_t>(e * P + p)] = Ge(e, p);
        // scipy's csr_matvecs, y fresh: per row, its stored entries in order
        for (int64_t r = 0; r < n; r++)
            for (int64_t jj = Fsp.indptr[static_cast<size_t>(r)]; jj < Fsp.indptr[static_cast<size_t>(r) + 1]; jj++) {
                const double a = Fsp.data[static_cast<size_t>(jj)];
                const int64_t p = Fsp.indices[static_cast<size_t>(jj)];
                for (int64_t e = 0; e < E; e++) {
                    const cd x = G[static_cast<size_t>(e * P + p)];
                    cd &y = Gt[static_cast<size_t>(e * n + r)];
                    y = cd(y.real() + a * x.real(), y.imag() + a * x.imag());
                }
            }
        return Gt;
    }
    // node blocks through the CSC (csc_matvecs: columns in order, each
    // column's rows in order), each block's product added into Gt
    std::vector<std::vector<std::pair<int64_t, double>>> cols(static_cast<size_t>(P));
    for (int64_t r = 0; r < n; r++)
        for (int64_t jj = Fsp.indptr[static_cast<size_t>(r)]; jj < Fsp.indptr[static_cast<size_t>(r) + 1]; jj++)
            cols[static_cast<size_t>(Fsp.indices[static_cast<size_t>(jj)])].push_back({r, Fsp.data[static_cast<size_t>(jj)]});
    std::vector<cd> res(static_cast<size_t>(n * E));
    for (int64_t p0 = 0; p0 < P; p0 += block) {
        const int64_t p1 = std::min(p0 + block, P);
        std::fill(res.begin(), res.end(), cd(0.0, 0.0));
        for (int64_t p = p0; p < p1; p++) {
            std::vector<cd> gcol(static_cast<size_t>(E));
            for (int64_t e = 0; e < E; e++) gcol[static_cast<size_t>(e)] = Ge(e, p);
            for (const auto &rc : cols[static_cast<size_t>(p)])
                for (int64_t e = 0; e < E; e++) {
                    const cd x = gcol[static_cast<size_t>(e)];
                    cd &y = res[static_cast<size_t>(rc.first * E + e)];
                    y = cd(y.real() + rc.second * x.real(), y.imag() + rc.second * x.imag());
                }
        }
        for (int64_t r = 0; r < n; r++)
            for (int64_t e = 0; e < E; e++) {
                cd &y = Gt[static_cast<size_t>(e * n + r)];
                y = c_add(y, res[static_cast<size_t>(r * E + e)]);
            }
    }
    return Gt;
}

// `_bnd_and_corner(ax, k, a_wire, gz, mirror)` and `_CompletionScatter.add`
// for one family pass, into E_r / E_c.
// `a_col`: the radius of the column term (the family's line observers), `a_wire`
// the row term's and the corner's (point tests at the node): one radius but at
// a two-radius node (`self_completions_two_radius`, whose second
// `_bnd_and_corner` call is the column term's alone).
void bnd_and_corner(const cx::Axis &ax, cd k, double a_wire, double gz, bool mirror, cd beta,
                    const nec5::NumpyLinalg &np, const std::vector<int64_t> &posL, const std::vector<int64_t> &Lu,
                    std::vector<cd> &E_r, std::vector<cd> &E_c, double a_col = -1.0) {
    const int64_t n = ax.n_basis;
    const int64_t E = static_cast<int64_t>(ax.ends.size());
    if (E == 0) return;
    std::vector<int64_t> live;
    for (int64_t m = 0; m < n; m++) {
        bool any = false;
        for (const cx::End &e : ax.ends) any = any || e.fv[static_cast<size_t>(m)] != 0.0;
        if (any) live.push_back(m);
    }
    if (live.empty()) return;
    const int64_t L = static_cast<int64_t>(live.size());
    // sf = sig[:, None] * fvT[:, live]: numpy's product of (sig, 0) and (fv, 0)
    std::vector<cd> sf(static_cast<size_t>(E * L));
    for (int64_t e = 0; e < E; e++)
        for (int64_t l = 0; l < L; l++)
            sf[static_cast<size_t>(e * L + l)] =
                np_cmul(cd(ax.ends[static_cast<size_t>(e)].sign, 0.0),
                        cd(ax.ends[static_cast<size_t>(e)].fv[static_cast<size_t>(live[static_cast<size_t>(l)])], 0.0));
    const size_t P = ax.w.size();
    std::vector<double> pts(ax.nodes), src(ax.nodes), ptE(3 * static_cast<size_t>(E)), pe(3 * static_cast<size_t>(E));
    for (int64_t e = 0; e < E; e++)
        for (int q = 0; q < 3; q++) ptE[3 * static_cast<size_t>(e) + q] = pe[3 * static_cast<size_t>(e) + q] = ax.ends[static_cast<size_t>(e)].pt[q];
    if (mirror) {
        for (size_t p = 0; p < P; p++) src[3 * p + 2] = 2.0 * gz - src[3 * p + 2];
        for (int64_t e = 0; e < E; e++) pe[3 * static_cast<size_t>(e) + 2] = 2.0 * gz - pe[3 * static_cast<size_t>(e) + 2];
    }
    const double a2 = a_wire * a_wire;
    // `_fdw_sparse`: Fd with each column scaled by w
    cx::CSR Fsp = ax.Fd;
    for (size_t jj = 0; jj < Fsp.data.size(); jj++) Fsp.data[jj] = Fsp.data[jj] * ax.w[static_cast<size_t>(Fsp.indices[jj])];
    const cd mjk = py_prod(cd(-0.0, -1.0), k);
    std::vector<cd> Gee(static_cast<size_t>(E * E));
    for (int64_t i = 0; i < E; i++)
        for (int64_t j = 0; j < E; j++) {
            const double d0 = ptE[3 * i] - pe[3 * j], d1 = ptE[3 * i + 1] - pe[3 * j + 1], d2 = ptE[3 * i + 2] - pe[3 * j + 2];
            Gee[static_cast<size_t>(i * E + j)] = g_of_r(mjk, std::sqrt(a2 + ((d0 * d0 + d2 * d2) + d1 * d1)));
        }
    const std::vector<cd> Gt = ends_to_nodes(mjk, a2, ptE, src, Fsp);  // (E, n)
    const std::vector<cd> Gs =
        ends_to_nodes(mjk, a_col < 0.0 ? a2 : a_col * a_col, pe, pts, Fsp);  // (E, n)
    using nec5::Op;
    // row_term = -(sf.T @ Gt) (L, n); col_term = -(Gs.T @ sf) (n, L);
    // corner = (sf.T @ Gee) @ sf (L, L)
    std::vector<cd> row(static_cast<size_t>(L * n)), col(static_cast<size_t>(n * L)), tmp(static_cast<size_t>(L * E)),
        corner(static_cast<size_t>(L * L));
    const Op sfT{sf.data(), L, E, 1, L}, sfO{sf.data(), E, L, L, 1};
    nec5::replay::np_matmul(np, sfT, Op{Gt.data(), E, n, n, 1}, row.data(), n, 1);
    nec5::replay::np_matmul(np, Op{Gs.data(), n, E, 1, n}, sfO, col.data(), L, 1);
    nec5::replay::np_matmul(np, sfT, Op{Gee.data(), E, E, E, 1}, tmp.data(), E, 1);
    nec5::replay::np_matmul(np, Op{tmp.data(), L, E, E, 1}, sfO, corner.data(), L, 1);
    for (cd &v : row) v = cd(-v.real(), -v.imag());
    for (cd &v : col) v = cd(-v.real(), -v.imag());
    // `_CompletionScatter.add`
    const int64_t nL = static_cast<int64_t>(Lu.size());
    for (int64_t l = 0; l < L; l++) {
        const int64_t pl = posL[static_cast<size_t>(live[static_cast<size_t>(l)])];
        for (int64_t j = 0; j < n; j++) {
            cd &e = E_r[static_cast<size_t>(pl * n + j)];
            e = c_add(e, np_cmul(beta, row[static_cast<size_t>(l * n + j)]));
        }
    }
    for (int64_t i = 0; i < nL; i++)
        for (int64_t l = 0; l < L; l++) {
            cd &e = E_r[static_cast<size_t>(i * n + live[static_cast<size_t>(l)])];
            e = c_add(e, np_cmul(beta, col[static_cast<size_t>(Lu[static_cast<size_t>(i)] * L + l)]));
        }
    for (int64_t r = 0; r < n; r++)
        for (int64_t l = 0; l < L; l++) {
            const int64_t pl = posL[static_cast<size_t>(live[static_cast<size_t>(l)])];
            cd &e = E_c[static_cast<size_t>(r * nL + pl)];
            e = c_add(e, np_cmul(beta, col[static_cast<size_t>(r * L + l)]));
        }
    for (int64_t a = 0; a < L; a++) {
        const int64_t pl = posL[static_cast<size_t>(live[static_cast<size_t>(a)])];
        for (int64_t l = 0; l < L; l++) {
            cd &e = E_r[static_cast<size_t>(pl * n + live[static_cast<size_t>(l)])];
            e = c_add(e, np_cmul(beta, corner[static_cast<size_t>(a * L + l)]));
        }
    }
}

// `self_completions(ctx, ax_b, ax_a, out=Z)`
// At a two-radius node (`a_below` > 0): `self_completions_two_radius`, every
// row term and corner at `a_below`, each family's column term at its own
// radius (`a_above` for the above family).
void self_completions(std::vector<cd> &dest, const cx::Axis &B, const cx::Axis &A, const BuriedMedium &bm,
                      double a_wire, double gz, double omega, double eps0, const nec5::NumpyLinalg &np,
                      double a_above = 0.0, double a_below = 0.0) {
    const int64_t n = B.n_basis;
    std::set<int64_t> lu;
    for (const cx::Axis *ax : {&B, &A})
        for (const cx::End &e : ax->ends)
            for (size_t m = 0; m < e.fv.size(); m++)
                if (e.fv[m] != 0.0) lu.insert(static_cast<int64_t>(m));
    const std::vector<int64_t> Lu(lu.begin(), lu.end());
    std::vector<int64_t> posL(static_cast<size_t>(n), -1);
    for (size_t i = 0; i < Lu.size(); i++) posL[static_cast<size_t>(Lu[i])] = static_cast<int64_t>(i);
    const int64_t nL = static_cast<int64_t>(Lu.size());
    std::vector<cd> E_r(static_cast<size_t>(nL * n), cd(0.0, 0.0)), E_c(static_cast<size_t>(n * nL), cd(0.0, 0.0));
    for (int fam = 0; fam < 2; fam++) {
        const cx::Axis &ax = fam == 0 ? B : A;
        const cd k = fam == 0 ? bm.k_m : cd(bm.k_p, 0.0);
        const cd wgt = fam == 0 ? bm.a_m : bm.c2;
        const cd ep = fam == 0 ? bm.eps_m : cd(eps0, 0.0);
        // `1j * omega * eps * 4 * np.pi`, CPython's products
        cd den = py_prod(cd(0.0, 1.0), cd(omega, 0.0));
        den = py_prod(den, ep);
        den = py_prod(den, cd(4.0, 0.0));
        den = py_prod(den, cd(3.141592653589793, 0.0));
        const cd beta_dir = py_quot(cd(1.0, 0.0), den);
        const cd beta_img = py_quot(wgt, den);
        // the medium's own k (k_p a Python float: its product with -1j)
        const cd kk = fam == 0 ? k : cd(bm.k_p, 0.0);
        if (a_below > 0.0) {
            const double a_line = fam == 0 ? a_below : a_above;
            const double a_col = a_line != a_below ? a_line : -1.0;
            bnd_and_corner(ax, kk, a_below, gz, false, beta_dir, np, posL, Lu, E_r, E_c, a_col);
            bnd_and_corner(ax, kk, a_below, gz, true, cd(-beta_img.real(), -beta_img.imag()), np, posL, Lu, E_r,
                           E_c, a_col);
        } else {
            bnd_and_corner(ax, kk, a_wire, gz, false, beta_dir, np, posL, Lu, E_r, E_c);
            bnd_and_corner(ax, kk, a_wire, gz, true, cd(-beta_img.real(), -beta_img.imag()), np, posL, Lu, E_r, E_c);
        }
    }
    // `flush`: E_c[L, :] = 0; dest[L, :] += E_r; dest[:, L] += E_c
    for (int64_t i = 0; i < nL; i++)
        for (int64_t l = 0; l < nL; l++) E_c[static_cast<size_t>(Lu[static_cast<size_t>(i)] * nL + l)] = cd(0.0, 0.0);
    for (int64_t i = 0; i < nL; i++)
        for (int64_t j = 0; j < n; j++) {
            cd &d = dest[static_cast<size_t>(Lu[static_cast<size_t>(i)] * n + j)];
            d = c_add(d, E_r[static_cast<size_t>(i * n + j)]);
        }
    for (int64_t r = 0; r < n; r++)
        for (int64_t l = 0; l < nL; l++) {
            cd &d = dest[static_cast<size_t>(r * n + Lu[static_cast<size_t>(l)])];
            d = c_add(d, E_c[static_cast<size_t>(r * nL + l)]);
        }
}

}  // namespace

void crossing_serve(std::vector<cdouble> &Z, const Geometry &g, const Basis &b, double a_wire, const BuriedMedium &bm,
                    double ground_z, double omega, double mu, double eps, const std::vector<int64_t> &a_idx,
                    const std::vector<int64_t> &b_idx, const nec5::NumpyLinalg &np, CrossingStages *stages,
                    double a_above, double a_below) {
    const int64_t n = b.n_basis;
    const double gz = ground_z;
    const cx::Axis A = axis_of(g, b, a_idx, gz, a_wire), B = axis_of(g, b, b_idx, gz, a_wire);
    // `_c1_moment`: -1j * omega * mu / (4.0 * pi), CPython's arithmetic
    const cd c1 = py_quot(py_prod(py_prod(cd(-0.0, -1.0), cd(omega, 0.0)), cd(mu, 0.0)), cd(4.0 * M_PI, 0.0));
    XCtx xc{gz, c1};
    if (a_below > 0.0) {
        // A TWO-RADIUS node (`cross_complete_blocks_two_radius`, antennaknobs
        // plan U5): the below rows' block at `a_below` (one-radius, read by
        // column), the above rows' main sandwich and source ends at
        // `a_above`, their point tests at the node and the corner at
        // `a_below`; `Z -= t_above; Z -= t_below.T`.
        std::vector<cd> t_below;
        {
            // `cross_complete_block_split(ctx_below)`: its own memo, its
            // coarse axes at `a_below`
            cx::BlockEval be;
            be.ev.k_p = bm.k_p;
            be.ev.k_m = bm.k_m;
            be.ev.n_threads = physical_threads();
            const cx::Axis Ac = axis_of(g, b, a_idx, gz, a_below), Bc = axis_of(g, b, b_idx, gz, a_below);
            t_below = main_split(g, A, B, Ac, Bc, a_idx, b_idx, be, gz, bm.k_p, c1, a_below, np);
            const Corner corner{&A, &B, gz, a_below, bm.k_p, bm.k_m, c1};
            cx::add_end_terms<Raise>(xc, be, a_below, A, B, true, t_below, corner);
        }
        std::vector<cd> t_above;
        {
            cx::BlockEval be;
            be.ev.k_p = bm.k_p;
            be.ev.k_m = bm.k_m;
            be.ev.n_threads = physical_threads();
            const cx::Axis Ac = axis_of(g, b, a_idx, gz, a_above), Bc = axis_of(g, b, b_idx, gz, a_above);
            t_above = main_split(g, A, B, Ac, Bc, a_idx, b_idx, be, gz, bm.k_p, c1, a_above, np);
            // the source ends at `a_above` (no corner, no test ends) ...
            cx::add_end_terms<Raise>(xc, be, a_above, A, B, true, t_above, cx::NoCorner(), false, true);
            // ... the test ends and the corner at `a_below`, a fresh memo
            cx::BlockEval be2;
            be2.ev.k_p = bm.k_p;
            be2.ev.k_m = bm.k_m;
            be2.ev.n_threads = physical_threads();
            const Corner corner{&A, &B, gz, a_below, bm.k_p, bm.k_m, c1};
            cx::add_end_terms<Raise>(xc, be2, a_below, A, B, true, t_above, corner, true, false);
        }
        if (stages) stages->t_ab = t_above;
        for (size_t q = 0; q < Z.size(); q++) Z[q] = c_sub(Z[q], t_above[q]);
        for (int64_t i = 0; i < n; i++)
            for (int64_t j = 0; j < n; j++) {
                cd &z = Z[static_cast<size_t>(i * n + j)];
                z = c_sub(z, t_below[static_cast<size_t>(j * n + i)]);
            }
        if (stages) {
            std::vector<cd> only(static_cast<size_t>(n * n), cd(0.0, 0.0));
            self_completions(only, B, A, bm, a_wire, gz, omega, eps, np, a_above, a_below);
            stages->completions = only;
        }
        self_completions(Z, B, A, bm, a_wire, gz, omega, eps, np, a_above, a_below);
        return;
    }
    cx::BlockEval be;
    be.ev.k_p = bm.k_p;
    be.ev.k_m = bm.k_m;
    be.ev.n_threads = physical_threads();
    std::vector<cd> t = main_split(g, A, B, A, B, a_idx, b_idx, be, gz, bm.k_p, c1, a_wire, np);
    if (stages) stages->t_main = t;
    const Corner corner{&A, &B, gz, a_wire, bm.k_p, bm.k_m, c1};
    cx::add_end_terms<Raise>(xc, be, a_wire, A, B, true, t, corner);
    if (stages) stages->t_ab = t;
    // `Z -= t_ab; Z -= t_ab.T`
    for (size_t q = 0; q < Z.size(); q++) Z[q] = c_sub(Z[q], t[q]);
    for (int64_t i = 0; i < n; i++)
        for (int64_t j = 0; j < n; j++) {
            cd &z = Z[static_cast<size_t>(i * n + j)];
            z = c_sub(z, t[static_cast<size_t>(j * n + i)]);
        }
    if (stages) {
        std::vector<cd> only(static_cast<size_t>(n * n), cd(0.0, 0.0));
        self_completions(only, B, A, bm, a_wire, gz, omega, eps, np);
        stages->completions = only;
    }
    self_completions(Z, B, A, bm, a_wire, gz, omega, eps, np);
}

}  // namespace bspline
}  // namespace mw
