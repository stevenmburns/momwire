// The crossing fill's split and self-completion pieces (`_crossing_fill.py`),
// shared by bs2's crossing serve (`_bspline_crossing_core.cpp`, momwire#1517
// step 4b-2, whence they moved) and the Sinusoidal-Galerkin one
// (`_sinusoidal_galerkin_crossing_core.cpp`, momwire#1718 S11b): the
// admissibility partition, the ACA, the row weights, and the self
// completions. `W` is the sampled basis' value type: a B-spline's real
// samples (`double`), or the sinusoidal sampler's complex ones, whose
// products are numpy's and sparsetools' complex ones.
#pragma once

#include "_crossing_shared_inline.h"
#include "_nec5_printout_core.h"  // `nec5::NumpyLinalg`, `nec5::replay`
#include "_somm_grid_core.h"      // `somm_core::py_quot`

#include <algorithm>
#include <array>
#include <cmath>
#include <complex>
#include <cstdint>
#include <limits>
#include <map>
#include <numeric>
#include <set>
#include <utility>
#include <vector>

namespace mw {
namespace crossing {
namespace {

constexpr size_t CLUSTER_LEAF = 3;         // `_CLUSTER_LEAF_SEGS`
constexpr double ADM_ETA = 1.0;            // `_ADM_ETA`
constexpr double ACA_COST_GUARD = 48.0;    // `_ACA_COST_GUARD`
constexpr double ACA_RULE_COST = 16.0;     // `_ACA_RULE_COST` (momwire#1595)
constexpr int64_t DENSE_FAR_MAX_PAIRS = 600000;  // `_DENSE_FAR_MAX_PAIRS`
constexpr double ENDS_TO_NODES_MB = 48.0;  // `_ends_to_nodes(budget_mb=)`
constexpr int64_t ENDS_TO_NODES_BYTES_PER_PAIR = 112;

inline cd py_quot(cd a, cd b) { return somm_core::py_quot(a, b); }
inline cd c_add(cd a, cd b) { return cd(a.real() + b.real(), a.imag() + b.imag()); }
inline cd c_sub(cd a, cd b) { return cd(a.real() - b.real(), a.imag() - b.imag()); }

// A sample as numpy holds it beside complex values: a real one promoted.
inline cd as_c(double x) { return cd(x, 0.0); }
inline cd as_c(cd x) { return x; }
// `_scale_cols(M, v)`'s `data * v[indices]`: a complex sample's product
// with (v, 0) is numpy's.
inline double scale_w(double d, double v) { return d * v; }
inline cd scale_w(cd d, double v) { return nec5::replay::np_cmul(d, cd(v, 0.0)); }
// scipy's `y += a * x` (sparsetools' `csr_matvecs` / `csc_matvecs`): a real
// sample's product part by part; a complex one sparsetools' complex product
// (plain on x86-64, contracted on arm64), then the sum.
inline cd axpy_w(cd y, double a, cd x) { return cd(y.real() + a * x.real(), y.imag() + a * x.imag()); }
inline cd axpy_w(cd y, cd a, cd x) {
#if MW_NUMPY_CONTRACTS
    const double pr = mw_fma::exact(a.real(), x.real(), -(a.imag() * x.imag()));
    const double pi = mw_fma::exact(a.real(), x.imag(), a.imag() * x.real());
#else
    const double pr = a.real() * x.real() - a.imag() * x.imag();
    const double pi = a.real() * x.imag() + a.imag() * x.real();
#endif
    return cd(y.real() + pr, y.imag() + pi);
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
template <class AX>
std::vector<int64_t> nodes_of(const AX &ax, const std::map<int64_t, std::pair<int64_t, int64_t>> &runs,
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

template <class AX>
std::map<int64_t, std::pair<int64_t, int64_t>> runs_of(const AX &ax) {
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
template <class AX>
std::vector<int64_t> support_rows_of(const AX &ax, const std::vector<int64_t> &ii) {
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
template <class W>
std::array<CSRT<W>, 4> row_weights_of(const AxisT<W> &ax, const std::vector<int64_t> &ii,
                                       const std::vector<int64_t> &rows) {
    std::vector<int64_t> pos(ax.w.size(), -1);
    for (size_t k = 0; k < ii.size(); k++) pos[static_cast<size_t>(ii[k])] = static_cast<int64_t>(k);
    std::array<CSRT<W>, 4> out;
    for (int m = 0; m < 4; m++) {
        CSRT<W> &o = out[static_cast<size_t>(m)];
        o.n_rows = static_cast<int64_t>(rows.size());
        o.n_cols = static_cast<int64_t>(ii.size());
        o.indptr.assign(rows.size() + 1, 0);
        const CSRT<W> &src = m < 3 ? ax.F : ax.Fd;
        for (size_t i = 0; i < rows.size(); i++) {
            const int64_t r = rows[i];
            for (int64_t jj = src.indptr[static_cast<size_t>(r)]; jj < src.indptr[static_cast<size_t>(r) + 1]; jj++) {
                const int64_t col = src.indices[static_cast<size_t>(jj)];
                const int64_t p = pos[static_cast<size_t>(col)];
                if (p < 0) continue;
                W v = scale_w(src.data[static_cast<size_t>(jj)], ax.w[static_cast<size_t>(col)]);
                if (m < 3) v = scale_w(v, ax.t[3 * static_cast<size_t>(col) + m]);
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
            for (int64_t j = 0; j < n; j++) row[static_cast<size_t>(j)] = c_sub(row[static_cast<size_t>(j)], nec5::replay::np_cmul(a, V[k][static_cast<size_t>(j)]));
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
        for (int64_t j = 0; j < n; j++) v[static_cast<size_t>(j)] = nec5::replay::np_cdiv(row[static_cast<size_t>(j)], delta);
        std::vector<cd> col = get_col(j_star);
        for (size_t k = 0; k < rank; k++) {
            const cd a = V[k][static_cast<size_t>(j_star)];
            for (int64_t i = 0; i < m; i++) col[static_cast<size_t>(i)] = c_sub(col[static_cast<size_t>(i)], nec5::replay::np_cmul(a, U[k][static_cast<size_t>(i)]));
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

// `_main_split`'s partition: the near cluster pairs on the dense axes, the
// far ones on the coarse axes (`axis_data(coarse=True)`), each far block
// ACA past the guard unless `_dense_beats_aca`, else direct.
template <class AX>
void split_specs(const std::vector<Cluster> &TA, const std::vector<Cluster> &TB, const AX &A, const AX &B,
                 const AX &Ac, const AX &Bc, const std::vector<int64_t> &a_idx, const std::vector<int64_t> &b_idx,
                 double a_wire, std::vector<Spec> &direct, std::vector<Spec> &far_aca) {
    std::vector<std::pair<int, int>> far, near;
    block_tree(TA, TB, far, near);
    const auto runA = runs_of(A), runB = runs_of(B);
    const auto runAc = runs_of(Ac), runBc = runs_of(Bc);
    auto segs_of = [](const Cluster &c, const std::vector<int64_t> &idx) {
        std::vector<int64_t> s;
        for (int64_t i : c.idx) s.push_back(idx[static_cast<size_t>(i)]);
        return s;
    };
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
        if (na * nb > ACA_COST_GUARD * (na + nb) && !dense_beats_aca(Ac, Bc, sp, a_wire))
            far_aca.push_back(std::move(sp));
        else
            direct.push_back(std::move(sp));
    }
}

// `_plan_sheets`: the fill's sheets from the direct specs' node pairs
template <class AX>
void plan_sheets(BlockEval &be, const std::vector<Spec> &direct, const AX &A, const AX &B, const AX &Ac,
                 const AX &Bc, double gz, double k_p, double a_wire) {
    std::vector<std::vector<double>> keep;
    std::vector<ni_eval::NodePair> pairs;
    for (const Spec &sp : direct) {
        const AX &SA = sp.coarse ? Ac : A, &SB = sp.coarse ? Bc : B;
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

// `_pair_groups` + `_direct_group`: consecutive specs under the pair budget,
// one table call over each group's pairs; `group(start, stop, K)` takes the
// group's memo indices, spec after spec, row a * |iB| + b.
template <class AX, class F>
void direct_groups(BlockEval &be, const std::vector<Spec> &direct, const AX &A, const AX &B, const AX &Ac,
                   const AX &Bc, double gz, double a_wire, int64_t budget, F &&group) {
    size_t start = 0;
    while (start < direct.size()) {
        size_t stop = start;
        int64_t run = 0;
        while (stop < direct.size()) {
            const int64_t n = static_cast<int64_t>(direct[stop].iA.size() * direct[stop].iB.size());
            if (run && run + n > budget) break;
            run += n;
            stop++;
        }
        std::vector<double> re, z, zp;
        for (size_t k = start; k < stop; k++) {
            const Spec &sp = direct[k];
            const AX &SA = sp.coarse ? Ac : A, &SB = sp.coarse ? Bc : B;
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
        table_index(be, re, z, zp, nullptr, K);
        group(start, stop, K);
        start = stop;
    }
}

// `_main_split`'s far block factorisation: per kernel {U, V, W, dzpW} an
// `aca_partial` over the block's designed rows / columns (sampled once,
// through the fill's memo, cached across the four kernels): `Uf` (m, r) and
// `Vf` (r, n), C order, `rk` the ranks.
template <class AX>
void aca_factors(const nec5::NumpyLinalg &np, const AX &A, const AX &B, const Spec &sp, BlockEval &be, double gz,
                 double a_wire, std::array<std::vector<cd>, 4> &Uf, std::array<std::vector<cd>, 4> &Vf,
                 std::array<int64_t, 4> &rk) {
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
    static const int KEYS[4] = {K_U, K_V, K_W, K_DZPW};
    std::map<int64_t, std::array<std::vector<cd>, 4>> row_s, col_s;
    auto sample = [&](const std::vector<double> &r, const std::vector<double> &z, const std::vector<double> &zp) {
        std::vector<Six> te;
        tables(be, a_wire, r, z, zp, nullptr, te);
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
}

// `_dense_beats_aca(Ac, Bc, iA, iB, a_wire)` (momwire#1595): whether a far
// block past the guard is still cheaper dense once the column twin's rule
// sharing is counted. The Python docstring carries the cost model; with every
// rho distinct it is the guard itself, so only blocks whose rho repeats move.
// rho by the same two hypots as the samples, n_rho by value as np.unique.
template <class AX>
bool dense_beats_aca(const AX &Ac, const AX &Bc, const Spec &sp, double a_wire) {
    const size_t m = sp.iA.size(), n = sp.iB.size();
    if (static_cast<int64_t>(m * n) > DENSE_FAR_MAX_PAIRS) return false;
    std::vector<double> re;
    re.reserve(m * n);
    for (size_t i = 0; i < m; i++) {
        const double *pa = &Ac.nodes[3 * static_cast<size_t>(sp.iA[i])];
        for (size_t j = 0; j < n; j++) {
            const double *pb = &Bc.nodes[3 * static_cast<size_t>(sp.iB[j])];
            re.push_back(std::hypot(std::hypot(pa[0] - pb[0], pa[1] - pb[1]), a_wire));
        }
    }
    std::sort(re.begin(), re.end());
    const double n_rho = static_cast<double>(std::unique(re.begin(), re.end()) - re.begin());
    const double c = ACA_RULE_COST, g = ACA_COST_GUARD;
    const double dm = static_cast<double>(m), dn = static_cast<double>(n);
    const double dense = dm * dn + c * n_rho;
    const double aca = g * ((dn + c * std::min(dn, n_rho)) + (dm + c * std::min(dm, n_rho)));
    return dense <= aca;
}

// `_g_of_r(k, R)` for one R: exp(-1j k R) / R
inline cd g_of_r(cd mjk, double R) { return nec5::replay::np_cdiv(std::exp(nec5::replay::np_cmul(mjk, cd(R, 0.0))), cd(R, 0.0)); }

// `_ends_to_nodes(k, a2, obs, src, Fsp)`: (E, n) row-major
template <class W>
std::vector<cd> ends_to_nodes(cd mjk, double a2, const std::vector<double> &obs, const std::vector<double> &src,
                              const CSRT<W> &Fsp) {
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
                const W a = Fsp.data[static_cast<size_t>(jj)];
                const int64_t p = Fsp.indices[static_cast<size_t>(jj)];
                for (int64_t e = 0; e < E; e++) {
                    const cd x = G[static_cast<size_t>(e * P + p)];
                    cd &y = Gt[static_cast<size_t>(e * n + r)];
                    y = axpy_w(y, a, x);
                }
            }
        return Gt;
    }
    // node blocks through the CSC (csc_matvecs: columns in order, each
    // column's rows in order), each block's product added into Gt
    std::vector<std::vector<std::pair<int64_t, W>>> cols(static_cast<size_t>(P));
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
                    y = axpy_w(y, rc.second, x);
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
template <class W>
void bnd_and_corner(const AxisT<W> &ax, cd k, double a_wire, double gz, bool mirror, cd beta,
                    const nec5::NumpyLinalg &np, const std::vector<int64_t> &posL, const std::vector<int64_t> &Lu,
                    std::vector<cd> &E_r, std::vector<cd> &E_c, double a_col = -1.0) {
    const int64_t n = ax.n_basis;
    const int64_t E = static_cast<int64_t>(ax.ends.size());
    if (E == 0) return;
    std::vector<int64_t> live;
    for (int64_t m = 0; m < n; m++) {
        bool any = false;
        for (const EndT<W> &e : ax.ends) any = any || e.fv[static_cast<size_t>(m)] != 0.0;
        if (any) live.push_back(m);
    }
    if (live.empty()) return;
    const int64_t L = static_cast<int64_t>(live.size());
    // sf = sig[:, None] * fvT[:, live]: numpy's product of (sig, 0) and (fv, 0)
    std::vector<cd> sf(static_cast<size_t>(E * L));
    for (int64_t e = 0; e < E; e++)
        for (int64_t l = 0; l < L; l++)
            sf[static_cast<size_t>(e * L + l)] =
                nec5::replay::np_cmul(cd(ax.ends[static_cast<size_t>(e)].sign, 0.0),
                        as_c(ax.ends[static_cast<size_t>(e)].fv[static_cast<size_t>(live[static_cast<size_t>(l)])]));
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
    CSRT<W> Fsp = ax.Fd;
    for (size_t jj = 0; jj < Fsp.data.size(); jj++) Fsp.data[jj] = scale_w(Fsp.data[jj], ax.w[static_cast<size_t>(Fsp.indices[jj])]);
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
            e = c_add(e, nec5::replay::np_cmul(beta, row[static_cast<size_t>(l * n + j)]));
        }
    }
    for (int64_t i = 0; i < nL; i++)
        for (int64_t l = 0; l < L; l++) {
            cd &e = E_r[static_cast<size_t>(i * n + live[static_cast<size_t>(l)])];
            e = c_add(e, nec5::replay::np_cmul(beta, col[static_cast<size_t>(Lu[static_cast<size_t>(i)] * L + l)]));
        }
    for (int64_t r = 0; r < n; r++)
        for (int64_t l = 0; l < L; l++) {
            const int64_t pl = posL[static_cast<size_t>(live[static_cast<size_t>(l)])];
            cd &e = E_c[static_cast<size_t>(r * nL + pl)];
            e = c_add(e, nec5::replay::np_cmul(beta, col[static_cast<size_t>(r * L + l)]));
        }
    for (int64_t a = 0; a < L; a++) {
        const int64_t pl = posL[static_cast<size_t>(live[static_cast<size_t>(a)])];
        for (int64_t l = 0; l < L; l++) {
            cd &e = E_r[static_cast<size_t>(pl * n + live[static_cast<size_t>(l)])];
            e = c_add(e, nec5::replay::np_cmul(beta, corner[static_cast<size_t>(a * L + l)]));
        }
    }
}

// `self_completions(ctx, ax_b, ax_a, out=Z)`
// At a two-radius node (`a_below` > 0): `self_completions_two_radius`, every
// row term and corner at `a_below`, each family's column term at its own
// radius (`a_above` for the above family).
// `dest` element (r, c) at r * ds0 + c * ds1 (C order: (n, 1)).
template <class W, class Med>
void self_completions(std::vector<cd> &dest, int64_t ds0, int64_t ds1, const AxisT<W> &B, const AxisT<W> &A,
                      const Med &bm, double a_wire, double gz, double omega, double eps0,
                      const nec5::NumpyLinalg &np, double a_above = 0.0, double a_below = 0.0) {
    const int64_t n = B.n_basis;
    std::set<int64_t> lu;
    for (const AxisT<W> *ax : {&B, &A})
        for (const EndT<W> &e : ax->ends)
            for (size_t m = 0; m < e.fv.size(); m++)
                if (e.fv[m] != 0.0) lu.insert(static_cast<int64_t>(m));
    const std::vector<int64_t> Lu(lu.begin(), lu.end());
    std::vector<int64_t> posL(static_cast<size_t>(n), -1);
    for (size_t i = 0; i < Lu.size(); i++) posL[static_cast<size_t>(Lu[i])] = static_cast<int64_t>(i);
    const int64_t nL = static_cast<int64_t>(Lu.size());
    std::vector<cd> E_r(static_cast<size_t>(nL * n), cd(0.0, 0.0)), E_c(static_cast<size_t>(n * nL), cd(0.0, 0.0));
    for (int fam = 0; fam < 2; fam++) {
        const AxisT<W> &ax = fam == 0 ? B : A;
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
            cd &d = dest[static_cast<size_t>(Lu[static_cast<size_t>(i)] * ds0 + j * ds1)];
            d = c_add(d, E_r[static_cast<size_t>(i * n + j)]);
        }
    for (int64_t r = 0; r < n; r++)
        for (int64_t l = 0; l < nL; l++) {
            cd &d = dest[static_cast<size_t>(r * ds0 + Lu[static_cast<size_t>(l)] * ds1)];
            d = c_add(d, E_c[static_cast<size_t>(r * nL + l)]);
        }
}

}  // namespace
}  // namespace crossing
}  // namespace mw
