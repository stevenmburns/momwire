// momwire#1517 step 1: see `_bspline_fill_core.h`. Each step is
// `_bspline_kernels.py` / `bspline.py` as written; the kernels are the
// extension's own (`mw::bspline::*`, the default host).
#include "_bspline_fill_core.h"
#include "_bspline_crossing_core.h"

#include "_bspline_core.h"
#include "_exact_kernel_coaxial_core.h"
#include "_razor_exact_core.h"
#include "_somm_grid_core.h"
#include "_somm_proj_core.h"
#include "_somm_below_remainder_inline.h"
#include "_somm_below_fill_core.h"
#include "_field_galerkin_inline.h"
#include "_somm_pairs_core.h"
#include "_somm_remainder_core.h"
#include "_mw_times.h"
#include "_razor_front_core.h"  // `leggauss`: `_quadrature.leggauss`'s frozen table

#if defined(_MSC_VER)
#pragma float_control(precise, on)
#pragma fp_contract(off)
#endif

#include <algorithm>
#include <array>
#include <climits>
#include <cmath>
#include <cstring>
#include <limits>
#include <map>
#include <set>
#include <stdexcept>
#include <type_traits>

namespace mw {
namespace bspline {

namespace {

using CF = nd::array_t<double, nd::array::c_style | nd::array::forcecast>;
using CI = nd::array_t<int64_t, nd::array::c_style | nd::array::forcecast>;

template <class C, class T = std::remove_const_t<std::remove_pointer_t<decltype(std::declval<const C &>().data())>>>
nd::array_t<T, nd::array::c_style | nd::array::forcecast> borrow(const C &v, std::vector<nd::ssize_t> shape) {
    nd::Buffer b;
    b.ptr = const_cast<T *>(v.data());
    b.ndim = static_cast<int>(shape.size());
    nd::ssize_t acc = sizeof(T);
    for (int q = b.ndim - 1; q >= 0; q--) {
        b.shape[q] = shape[static_cast<size_t>(q)];
        b.strides[q] = acc;
        acc *= shape[static_cast<size_t>(q)];
    }
    b.writeable = false;
    return nd::array_t<T, nd::array::c_style | nd::array::forcecast>::borrow(b);
}

// A writeable view (the assembler's output Z).
template <class T>
nd::array_t<T> borrow_mut(std::vector<T> &v, std::vector<nd::ssize_t> shape) {
    nd::Buffer b;
    b.ptr = v.data();
    b.ndim = static_cast<int>(shape.size());
    nd::ssize_t acc = sizeof(T);
    for (int q = b.ndim - 1; q >= 0; q--) {
        b.shape[q] = shape[static_cast<size_t>(q)];
        b.strides[q] = acc;
        acc *= shape[static_cast<size_t>(q)];
    }
    b.writeable = true;
    return nd::array_t<T>::borrow(b);
}

constexpr size_t OFFEDGE_SPLIT_MAX_EDGES = 32;
constexpr double PHASE_KL_CEILING = 0.5;     // `_LADDER_PHASE_KL_CEILING`
constexpr int64_t PHASE_LIMITED_BELOW = 8;   // `_LADDER_PHASE_LIMITED_BELOW`
constexpr int BSPLINE_ACCEL_MAX_D = 3;

using Ladder = std::vector<std::pair<double, int64_t>>;

// `_gl01(n)`: Gauss-Legendre on [0, 1].
void gl01(int64_t n, std::vector<double> &t, std::vector<double> &w) {
    std::vector<double> x, ww;
    if (!razor::leggauss(n, x, ww)) throw FrontNotPorted("an untabled Gauss order");
    t.resize(x.size());
    w.resize(x.size());
    for (size_t i = 0; i < x.size(); i++) {
        t[i] = 0.5 * (x[i] + 1.0);
        w[i] = 0.5 * ww[i];
    }
}

// Segments as (n, 3) row-major, with each row's length `norm3(r - l)`.
struct Segs {
    std::vector<double> l, r;
    size_t n() const { return l.size() / 3; }
    double len(size_t i) const {
        const double x = r[3 * i] - l[3 * i], y = r[3 * i + 1] - l[3 * i + 1],
                     z = r[3 * i + 2] - l[3 * i + 2];
        return std::sqrt((x * x + y * y) + z * z);
    }
    Segs take(const std::vector<size_t> &idx) const {
        Segs s;
        for (size_t i : idx)
            for (int c = 0; c < 3; c++) {
                s.l.push_back(l[3 * i + c]);
                s.r.push_back(r[3 * i + c]);
            }
        return s;
    }
};

// The extended-kernel spec (`_EK`) of one block: labels and the per-source
// tube radius (`b_j`: absent, one value, or one per column).
struct EK {
    bool on = false;
    std::vector<int64_t> group_i, group_j;
    enum BKind { none, scalar, per_col } b_kind = none;
    double b = 0;
    std::vector<double> b_j;
};

EK ek_cut(const EK &ek, const std::vector<size_t> *rows, const std::vector<size_t> *cols) {
    EK o = ek;
    if (!ek.on) return o;
    if (rows) {
        o.group_i.clear();
        for (size_t i : *rows) o.group_i.push_back(ek.group_i[i]);
    }
    if (cols) {
        o.group_j.clear();
        for (size_t j : *cols) o.group_j.push_back(ek.group_j[j]);
        if (ek.b_kind == EK::per_col) {
            o.b_j.clear();
            for (size_t j : *cols) o.b_j.push_back(ek.b_j[j]);
        }
    }
    return o;
}

using Block = CArray;  // (nd, nd, Ni, Nj) C order

template <class K>
Block offedge(const Segs &si, const Segs &sj, const std::vector<double> &a_rows, K k, int d, int64_t n_qp,
              const EK &ek, const Ladder &ladder_in);

// `abs(k)`: a real k's magnitude, or CPython's complex abs (hypot) of the
// lower medium's
inline double kmag(double k) { return std::fabs(k); }
inline double kmag(cdouble k) { return std::hypot(k.real(), k.imag()); }

// Scatter `sub` (nd, nd, |ri|, |cj|) into `out` (nd, nd, Ni, Nj).
void place(Block &out, size_t Ni, size_t Nj, const Block &sub, const std::vector<size_t> &ri,
           const std::vector<size_t> &cj, int nd) {
    // the column list as runs of consecutive indices, each copied whole
    std::vector<std::pair<size_t, size_t>> runs;  // (start in cj, length)
    for (size_t b = 0; b < cj.size();) {
        size_t e = b + 1;
        while (e < cj.size() && cj[e] == cj[e - 1] + 1) e++;
        runs.push_back({b, e - b});
        b = e;
    }
    for (int p = 0; p < nd; p++)
        for (int q = 0; q < nd; q++)
            for (size_t a = 0; a < ri.size(); a++) {
                cdouble *dst = &out[((static_cast<size_t>(p) * nd + q) * Ni + ri[a]) * Nj];
                const cdouble *src = &sub[((static_cast<size_t>(p) * nd + q) * ri.size() + a) * cj.size()];
                for (const auto &rn : runs) std::copy(src + rn.first, src + rn.first + rn.second, dst + cj[rn.first]);
            }
}

// the kernel's output as it is (no copy)
Block from_nd(const nd::array_t<std::complex<double>> &a) {
    return Block::adopt(a);
}

// `_seg_seg_full_moments_offedge`: a real k, or a complex one (the lower
// medium's, momwire#778: the plain kernel's COMPLEX_K instantiation, no
// extended kernel).
template <class K>
Block offedge(const Segs &si, const Segs &sj, const std::vector<double> &a_rows, K k, int d, int64_t n_qp,
              const EK &ek, const Ladder &ladder) {
    constexpr bool CPLX = std::is_same<K, cdouble>::value;
    if (CPLX && ek.on) throw FrontNotPorted("the extended kernel at a complex k");
    const int nd = d + 1;
    const size_t Ni = si.n(), Nj = sj.n();
    // `_normalize_row_radius`
    bool uniform = true;
    for (double v : a_rows)
        if (!(v == a_rows[0])) uniform = false;
    // `_phase_split_needed`
    bool all_high = true;
    for (const auto &t : ladder) all_high = all_high && t.second >= PHASE_LIMITED_BELOW;
    if (!ladder.empty() && !all_high) {
        std::vector<size_t> ok_i, no_i, ok_j, no_j;
        for (size_t i = 0; i < Ni; i++) (si.len(i) * kmag(k) <= PHASE_KL_CEILING ? ok_i : no_i).push_back(i);
        for (size_t j = 0; j < Nj; j++) (sj.len(j) * kmag(k) <= PHASE_KL_CEILING ? ok_j : no_j).push_back(j);
        const bool all_ok = no_i.empty() && no_j.empty();
        const bool any_ok = !ok_i.empty() && !ok_j.empty();
        if (!(all_ok || !any_ok)) {
            Block out(static_cast<size_t>(nd * nd) * Ni * Nj);
            for (const auto *ri : {&ok_i, &no_i}) {
                if (ri->empty()) continue;
                for (const auto *cj : {&ok_j, &no_j}) {
                    if (cj->empty()) continue;
                    std::vector<double> a_sub;
                    for (size_t i : *ri) a_sub.push_back(a_rows[i]);
                    place(out, Ni, Nj,
                          offedge(si.take(*ri), sj.take(*cj), a_sub, k, d, n_qp, ek_cut(ek, ri, cj), ladder),
                          *ri, *cj, nd);
                }
            }
            return out;
        }
    }
    // `_ladder_for_block`
    Ladder lad = ladder;
    if (!lad.empty()) {
        double len_max = 0.0;
        for (size_t i = 0; i < Ni; i++) len_max = std::max(len_max, si.len(i));
        double lj = 0.0;
        for (size_t j = 0; j < Nj; j++) lj = std::max(lj, sj.len(j));
        // `max(float(norm3(..).max()), float(norm3(..).max()))`
        len_max = std::max(len_max, lj);
        if (!(kmag(k) * len_max <= PHASE_KL_CEILING)) {
            Ladder kept;
            for (const auto &t : lad)
                if (t.second >= PHASE_LIMITED_BELOW) kept.push_back(t);
            lad = kept;
        }
    }
    if (d > BSPLINE_ACCEL_MAX_D) throw FrontNotPorted("degree beyond the C++ kernels");
    // radius runs
    if (!uniform) {
        Block out(static_cast<size_t>(nd * nd) * Ni * Nj);
        size_t s = 0;
        while (s < Ni) {
            size_t e = s + 1;
            while (e < Ni && a_rows[e] == a_rows[s]) e++;
            std::vector<size_t> rows;
            for (size_t i = s; i < e; i++) rows.push_back(i);
            std::vector<size_t> cols;
            for (size_t j = 0; j < Nj; j++) cols.push_back(j);
            std::vector<double> a_sub(e - s, a_rows[s]);
            place(out, Ni, Nj, offedge(si.take(rows), sj, a_sub, k, d, n_qp, ek_cut(ek, &rows, nullptr), lad),
                  rows, cols, nd);
            s = e;
        }
        return out;
    }
    const double a = a_rows.empty() ? 0.0 : a_rows[0];
    // `_ek_src_runs`: one call per distinct source-tube radius
    if (ek.on && ek.b_kind == EK::per_col) {
        std::map<double, std::vector<size_t>> runs;  // np.unique: ascending
        for (size_t j = 0; j < Nj; j++) runs[ek.b_j[j]].push_back(j);
        Block out(static_cast<size_t>(nd * nd) * Ni * Nj);
        std::vector<size_t> rows;
        for (size_t i = 0; i < Ni; i++) rows.push_back(i);
        for (const auto &kv : runs) {
            EK sub = ek_cut(ek, nullptr, &kv.second);
            sub.b_kind = EK::scalar;
            sub.b = kv.first;
            sub.b_j.clear();
            place(out, Ni, Nj, offedge(si, sj.take(kv.second), a_rows, k, d, n_qp, sub, lad), rows,
                  kv.second, nd);
        }
        return out;
    }
    const CF li = borrow(si.l, {static_cast<nd::ssize_t>(Ni), 3}),
             ri = borrow(si.r, {static_cast<nd::ssize_t>(Ni), 3}),
             lj = borrow(sj.l, {static_cast<nd::ssize_t>(Nj), 3}),
             rj = borrow(sj.r, {static_cast<nd::ssize_t>(Nj), 3});
    const double a2 = a * a;
    const double a_src = !ek.on || ek.b_kind == EK::none ? -1.0 : ek.b;
    std::vector<double> t01, w01;
    gl01(n_qp, t01, w01);
    if (!lad.empty()) {
        // `_ladder_arrays`
        std::vector<double> tt, ww, ratios{0.0};
        std::vector<int64_t> orders{n_qp};
        for (const auto &t : lad) {
            orders.push_back(t.second);
            ratios.push_back(t.first);
        }
        for (int64_t o : orders) {
            std::vector<double> t, w;
            gl01(o, t, w);
            tt.insert(tt.end(), t.begin(), t.end());
            ww.insert(ww.end(), w.begin(), w.end());
        }
        const CF ta = borrow(tt, {static_cast<nd::ssize_t>(tt.size())}),
                 wa = borrow(ww, {static_cast<nd::ssize_t>(ww.size())}),
                 ra = borrow(ratios, {static_cast<nd::ssize_t>(ratios.size())});
        const CI oa = borrow(orders, {static_cast<nd::ssize_t>(orders.size())});
        if constexpr (CPLX) {
            return from_nd(seg_seg_full_moments_bspline_cplx_tiered(li, ri, lj, rj, a2, k, d, ta, wa, oa, ra, false));
        } else {
        if (!ek.on)
            return from_nd(seg_seg_full_moments_bspline_tiered(li, ri, lj, rj, a2, k, d, ta, wa, oa, ra, false));
        const CI gi = borrow(ek.group_i, {static_cast<nd::ssize_t>(Ni)}),
                 gj = borrow(ek.group_j, {static_cast<nd::ssize_t>(Nj)});
        return from_nd(seg_seg_full_moments_bspline_ek_tiered(li, ri, lj, rj, a2, k, d, ta, wa, oa, ra, gi,
                                                               gj, a, false, a_src));
        }
    }
    const CF ta = borrow(t01, {static_cast<nd::ssize_t>(t01.size())}),
             wa = borrow(w01, {static_cast<nd::ssize_t>(w01.size())});
    if constexpr (CPLX) {
        return from_nd(seg_seg_full_moments_bspline_cplx(li, ri, lj, rj, a2, k, d, ta, wa, false));
    } else {
        if (!ek.on) return from_nd(seg_seg_full_moments_bspline(li, ri, lj, rj, a2, k, d, ta, wa, false));
        const CI gi = borrow(ek.group_i, {static_cast<nd::ssize_t>(Ni)}),
                 gj = borrow(ek.group_j, {static_cast<nd::ssize_t>(Nj)});
        return from_nd(seg_seg_full_moments_bspline_ek(li, ri, lj, rj, a2, k, d, ta, wa, gi, gj, a, false, a_src));
    }
}

// `_static_is_toeplitz`: np.allclose(h, h[0], rtol=1e-12, atol=1e-15).
bool is_toeplitz(const std::vector<double> &arc) {
    if (arc.size() < 2) return false;
    const double h0 = arc[1] - arc[0];
    for (size_t i = 1; i + 1 < arc.size(); i++) {
        const double h = arc[i + 1] - arc[i];
        if (!(std::fabs(h - h0) <= 1e-15 + 1e-12 * std::fabs(h0))) return false;
    }
    return true;
}

}  // namespace

std::vector<std::pair<double, int64_t>> default_ladder(int64_t n_qp_pair) {
    // `DEFAULT_PAIR_ORDER_LADDER = ((16.0, 4),)`, tiers below the base
    std::vector<std::pair<double, int64_t>> out;
    if (4 < n_qp_pair) out.push_back({16.0, 4});
    return out;
}

CArray build_J_blocks(const Geometry &g, const std::vector<double> &radius_per_wire,
                                    const FillParams &fp) {
    const int d = fp.degree, nd = d + 1;
    const size_t N = static_cast<size_t>(g.n_segs_total);
    std::vector<double> seg_a;  // `_seg_radius`
    for (size_t w = 0; w < g.per_wire.size(); w++)
        for (int64_t s = 0; s < g.per_wire[w].n_total; s++) seg_a.push_back(radius_per_wire[w]);
    // the same-edge slices
    std::vector<std::pair<size_t, size_t>> slices;
    for (size_t w = 0; w < g.per_wire.size(); w++) {
        const auto &eo = g.per_wire[w].edge_offsets;
        for (size_t e = 0; e + 1 < eo.size(); e++)
            slices.push_back({static_cast<size_t>(g.seg_offsets[w] + eo[e]),
                              static_cast<size_t>(g.seg_offsets[w] + eo[e + 1])});
    }
    Block J;
    if (slices.size() == 1 && slices[0].first == 0 && slices[0].second == N) {
        J = Block(static_cast<size_t>(nd * nd) * N * N);
    } else {
        Segs s;
        s.l = g.seg_l;
        s.r = g.seg_r;
        EK ek;
        if (fp.extended_kernel) {
            ek.on = true;
            ek.group_i.assign(N, 0);  // `_ek_axis_groups`: one label (momwire#1368)
            ek.group_j.assign(N, 0);
            bool all_eq = true;
            for (double v : seg_a) all_eq = all_eq && v == seg_a[0];
            if (!all_eq) {
                ek.b_kind = EK::per_col;
                ek.b_j = seg_a;
            }
        }
        // The off-edge pairs only (momwire#1517 speed): each edge's rows
        // against every column off it. `_build_J_blocks` fills every pair and
        // then overwrites the same-edge blocks below, and a pair's moments do
        // not depend on which pairs share its call (the per-pair phase guard
        // and tier, momwire#920), so this is the same J with the overwritten
        // half never computed. Few edges only: a deck of many short edges
        // saves little and would pay a call per edge.
        if (slices.size() <= OFFEDGE_SPLIT_MAX_EDGES) {
            // every entry written below: off-edge blocks here, the same-edge
            // ones by the overwrite that follows
            J = Block(static_cast<size_t>(nd * nd) * N * N, false);
            for (const auto &sl : slices) {
                std::vector<size_t> rows, cols;
                for (size_t q = sl.first; q < sl.second; q++) rows.push_back(q);
                for (size_t q = 0; q < N; q++)
                    if (q < sl.first || q >= sl.second) cols.push_back(q);
                if (cols.empty()) continue;
                std::vector<double> a_rows;
                for (size_t q : rows) a_rows.push_back(seg_a[q]);
                const Block sub = offedge(s.take(rows), s.take(cols), a_rows, fp.k, d, fp.n_qp_pair,
                                          ek_cut(ek, &rows, &cols), fp.ladder);
                place(J, N, N, sub, rows, cols, nd);
            }
        } else {
            J = offedge(s, s, seg_a, fp.k, d, fp.n_qp_pair, ek, fp.ladder);
        }
    }
    // the same-edge blocks: `A_st + A_reg`
    for (size_t w = 0; w < g.per_wire.size(); w++) {
        const WireGeometry &pw = g.per_wire[w];
        const double a_w = radius_per_wire[w];
        for (size_t e = 0; e + 1 < pw.edge_offsets.size(); e++) {
            const std::vector<double> &arc = pw.edge_arc_edges[e];
            const size_t base = static_cast<size_t>(g.seg_offsets[w] + pw.edge_offsets[e]);
            const size_t n = arc.size() - 1;
            if (!is_toeplitz(arc)) throw FrontNotPorted("a non-uniform edge");
            const double h = arc[1] - arc[0];
            const nd::array_t<double> A_st =
                fp.extended_kernel
                    ? seg_seg_static_moments_bspline_uniform_ek(h, a_w, n, d, a_w, 0)
                    : seg_seg_static_moments_bspline_uniform(h, a_w, n, d, 0, 0);
            // `_seg_seg_reg_geometry`
            std::vector<double> t01, w01;
            gl01(fp.n_qp_same_edge, t01, w01);
            const size_t q = t01.size();
            std::vector<double> s_q(n * q), u_q(n * q), w_q(n * q);
            for (size_t i = 0; i < n; i++) {
                const double hs = arc[i + 1] - arc[i];
                for (size_t j = 0; j < q; j++) {
                    s_q[i * q + j] = arc[i] + t01[j] * hs;
                    u_q[i * q + j] = (t01[j] * hs) * 1.0;
                    w_q[i * q + j] = (w01[j] * hs) * 1.0;
                }
            }
            const CF sq = borrow(s_q, {static_cast<nd::ssize_t>(n * q)});
            const nd::array_t<double> R = reg_distance_table(sq, sq, a_w * a_w);
            std::vector<double> wu(static_cast<size_t>(nd) * n * q);
            for (int p = 0; p < nd; p++)
                for (size_t i = 0; i < n * q; i++) {
                    // `_upow(u_q, p)`: p <= 2 numpy's fast paths (1, u, u*u)
                    const double u = u_q[i];
                    const double up = p == 0 ? 1.0 : p == 1 ? u : p == 2 ? u * u : std::pow(u, p);
                    wu[static_cast<size_t>(p) * n * q + i] = w_q[i] * up;
                }
            const CF wua = borrow(wu, {nd, static_cast<nd::ssize_t>(n), static_cast<nd::ssize_t>(q)});
            const std::vector<double> kv{fp.k};
            const CF ka = borrow(kv, {1});
            const CF Rv = CF::borrow(R.buffer());
            const nd::array_t<std::complex<double>> A_reg =
                fp.extended_kernel ? seg_seg_reg_moments_bspline_swept_ek(Rv, wua, ka, a_w)
                                   : seg_seg_reg_moments_bspline_swept(Rv, wua, ka);
            // J[:, :, sl, sl] = A_st + A_reg[0]
            const double *st = A_st.data();
            const std::complex<double> *rg = A_reg.data();
            for (int p = 0; p < nd; p++)
                for (int pp = 0; pp < nd; pp++)
                    for (size_t i = 0; i < n; i++)
                        for (size_t j = 0; j < n; j++) {
                            const size_t loc = ((static_cast<size_t>(p) * nd + pp) * n + i) * n + j;
                            const cdouble r = rg[loc];
                            J[((static_cast<size_t>(p) * nd + pp) * N + base + i) * N + base + j] =
                                cdouble(st[loc] + r.real(), 0.0 + r.imag());
                        }
        }
    }
    return J;
}

namespace {

// numpy's real `@` for one (m, n) @ (n, p) product of C-ordered or
// transposed operands (strides in elements), as `DOUBLE_matmul` dispatches
// it: ddot for a scalar result, dgemv for a vector-matrix / matrix-vector
// product, dgemm otherwise. Only the shapes the loading Gram forms (n = 3,
// every operand blasable).
constexpr int CB_ROW = 101, CB_COL = 102, CB_NO = 111, CB_T = 112;
bool blasable(int64_t s1, int64_t s2, int64_t d2) { return s2 == 1 && s1 >= d2; }

void dmatmul(const nec5::NumpyLinalg &np, const double *a, int64_t m, int64_t n, int64_t a0, int64_t a1,
             const double *b, int64_t p, int64_t b0, int64_t b1, double *out) {
    const int64_t os_m = p, os_p = 1;  // a fresh C-ordered result
    const bool i1b = blasable(a0, a1, n) || blasable(a1, a0, m);
    const bool i2b = blasable(b0, b1, p) || blasable(b1, b0, n);
    if (n == 1) throw FrontNotPorted("a real product numpy does not hand BLAS");
    if (m == 1 && p == 1) {
        out[0] = 0.0 + np.ddot(n, a, a1, b, b0);  // `DOUBLE_dot`: sum += ddot
        return;
    }
    if (m == 1 && i2b && blasable(a1, 1, 1)) {  // vector @ matrix: gemv on b's transpose
        int order;
        int64_t lda;
        if (blasable(b1, b0, n)) {
            order = CB_COL;
            lda = b1;
        } else {
            order = CB_ROW;
            lda = b0;
        }
        np.dgemv(order, CB_T, n, p, 1.0, b, lda, a, a1, 0.0, out, os_p);
        return;
    }
    if (p == 1 && i1b && blasable(b0, 1, 1)) {  // matrix @ vector
        int order;
        int64_t lda;
        if (blasable(a0, a1, n)) {
            order = CB_COL;
            lda = a0;
        } else {
            order = CB_ROW;
            lda = a1;
        }
        np.dgemv(order, CB_T, n, m, 1.0, a, lda, b, b0, 0.0, out, os_m);
        return;
    }
    if (m == 1 || p == 1 || !i1b || !i2b) throw FrontNotPorted("a real product numpy does not hand BLAS");
    int t1, t2;
    int64_t lda, ldb;
    if (blasable(a0, a1, n)) {
        t1 = CB_NO;
        lda = a0;
    } else {
        t1 = CB_T;
        lda = a1;
    }
    if (blasable(b0, b1, p)) {
        t2 = CB_NO;
        ldb = b0;
    } else {
        t2 = CB_T;
        ldb = b1;
    }
    if (a == b && m == p && a0 == b1 && a1 == b0 && t1 != t2) throw FrontNotPorted("the real syrk case");
    np.dgemm(CB_ROW, t1, t2, m, p, n, 1.0, a, lda, b, ldb, 0.0, out, os_m);
}

}  // namespace

Triplets loading_gram(const Geometry &g, const Basis &b, const nec5::NumpyLinalg &np) {
    if (np.dgemm == nullptr || np.dgemv == nullptr || np.ddot == nullptr)
        throw FrontNotPorted("no real BLAS for the loading Gram");
    const int nd = b.degree + 1;
    const int64_t n_poly = nd;
    // segment -> [(basis, wing)], in the order the dict first meets each
    std::vector<int64_t> order;
    std::map<int64_t, std::vector<std::pair<int64_t, int>>> seg_map;
    for (int64_t m = 0; m < b.n_basis; m++)
        for (int a = 0; a < nd; a++) {
            bool nz = false;
            for (int q = 0; q < nd; q++)
                nz = nz || b.polys[static_cast<size_t>((m * nd + a) * nd + q)] != 0.0;
            if (!nz) continue;
            const int64_t sg = b.supp_seg[static_cast<size_t>(m * nd + a)];
            auto it = seg_map.find(sg);
            if (it == seg_map.end()) {
                order.push_back(sg);
                seg_map[sg].push_back({m, a});
            } else {
                it->second.push_back({m, a});
            }
        }
    Triplets t;
    for (int64_t sg : order) {
        const auto &entries = seg_map[sg];
        const double hs = g.h_per_seg[static_cast<size_t>(sg)];
        // H[p, q] = np.float_power(hs, p + q + 1) / (p + q + 1)
        std::vector<double> H(static_cast<size_t>(n_poly * n_poly));
        for (int64_t p = 0; p < n_poly; p++)
            for (int64_t q = 0; q < n_poly; q++) {
                const double e = static_cast<double>(p + q + 1);
                H[static_cast<size_t>(p * n_poly + q)] = std::pow(hs, e) / e;
            }
        const int64_t ne = static_cast<int64_t>(entries.size());
        std::vector<double> C(static_cast<size_t>(ne * n_poly));
        for (int64_t i = 0; i < ne; i++)
            for (int64_t q = 0; q < n_poly; q++)
                C[static_cast<size_t>(i * n_poly + q)] =
                    b.polys[static_cast<size_t>((entries[i].first * nd + entries[i].second) * nd + q)];
        // `C @ H @ C.T`
        std::vector<double> CH(static_cast<size_t>(ne * n_poly)), M(static_cast<size_t>(ne * ne));
        dmatmul(np, C.data(), ne, n_poly, n_poly, 1, H.data(), n_poly, n_poly, 1, CH.data());
        dmatmul(np, CH.data(), ne, n_poly, n_poly, 1, C.data(), ne, 1, n_poly, M.data());
        // `np.searchsorted(seg_offsets, s, side="right") - 1`
        const int64_t w =
            static_cast<int64_t>(std::upper_bound(g.seg_offsets.begin(), g.seg_offsets.end(), sg) -
                                 g.seg_offsets.begin()) - 1;
        for (int64_t i = 0; i < ne; i++)
            for (int64_t j = 0; j < ne; j++) {
                t.rows.push_back(entries[i].first);
                t.cols.push_back(entries[j].first);
                t.vals.push_back(M[static_cast<size_t>(i * ne + j)]);
                t.wire.push_back(w);
            }
    }
    return t;
}

void add_wire_scaled(std::vector<cdouble> &Z, int64_t n_basis, const Triplets &t,
                     const std::vector<cdouble> &z_wire) {
    for (size_t q = 0; q < t.vals.size(); q++) {
        const cdouble v = nec5::replay::np_cmul(z_wire[static_cast<size_t>(t.wire[q])], cdouble(t.vals[q], 0.0));
        cdouble &z = Z[static_cast<size_t>(t.rows[q] * n_basis + t.cols[q])];
        z = nec5::replay::c_add(z, v);
    }
}

double wire_loss_power(const Triplets &t, const std::vector<cdouble> &coeffs,
                       const std::vector<cdouble> &z_wire, size_t n_wires) {
    std::vector<double> per_wire(n_wires, 0.0);
    for (size_t q = 0; q < t.vals.size(); q++) {
        const size_t w = static_cast<size_t>(t.wire[q]);
        const cdouble a = coeffs[static_cast<size_t>(t.rows[q])], bb = coeffs[static_cast<size_t>(t.cols[q])];
        const double re = nec5::replay::np_cmul(std::conj(a), bb).real();
        const double contrib = ((0.5 * z_wire[w].real()) * re) * t.vals[q];
        per_wire[w] += contrib;
    }
    return nec5::replay::np_pairwise(per_wire.data(), static_cast<int64_t>(n_wires));
}

bool dense_tensor_fits(int degree, int64_t n_segs, int64_t swept_mem_mb) {
    const int64_t nd = degree + 1;
    return nd * nd * n_segs * n_segs * 16 <= (swept_mem_mb << 20);
}

std::vector<cdouble> compute_Z_chunked(const Geometry &g, const Basis &b,
                                       const std::vector<double> &radius_per_wire, const FillParams &fp,
                                       double omega, double eps, double mu, double swept_mem_mb) {
    const int d = fp.degree, nd = d + 1;
    const int64_t N = g.n_segs_total, n_basis = b.n_basis;
    std::vector<double> a_row;  // `_seg_radius`
    for (size_t w = 0; w < g.per_wire.size(); w++)
        for (int64_t q = 0; q < g.per_wire[w].n_total; q++) a_row.push_back(radius_per_wire[w]);
    Segs all;
    all.l = g.seg_l;
    all.r = g.seg_r;
    EK ek;  // `_ek_spec(geom)`
    if (fp.extended_kernel) {
        ek.on = true;
        ek.group_i.assign(static_cast<size_t>(N), 0);
        ek.group_j.assign(static_cast<size_t>(N), 0);
        bool all_eq = true;
        for (double v : a_row) all_eq = all_eq && v == a_row[0];
        if (!all_eq) {
            ek.b_kind = EK::per_col;
            ek.b_j = a_row;
        }
    }
    std::vector<cdouble> Z(static_cast<size_t>(n_basis * n_basis), cdouble(0.0, 0.0));
    nd::array_t<std::complex<double>> Za = borrow_mut(Z, {static_cast<nd::ssize_t>(n_basis),
                                                          static_cast<nd::ssize_t>(n_basis)});
    const CI sa = borrow(b.supp_seg, {static_cast<nd::ssize_t>(n_basis), nd});
    const CF pa = borrow(b.polys, {static_cast<nd::ssize_t>(n_basis), nd, nd});
    const CF ta = borrow(g.tangents, {static_cast<nd::ssize_t>(N), 3});
    std::vector<int64_t> all_n(static_cast<size_t>(n_basis));
    for (int64_t m = 0; m < n_basis; m++) all_n[static_cast<size_t>(m)] = m;
    const auto touching = [&](int64_t lo, int64_t hi) {  // `_bases_touching`
        std::vector<int64_t> out;
        for (int64_t m = 0; m < n_basis; m++)
            for (int q = 0; q < nd; q++) {
                const int64_t sg = b.supp_seg[static_cast<size_t>(m * nd + q)];
                if (sg >= lo && sg < hi) {
                    out.push_back(m);
                    break;
                }
            }
        return out;
    };
    const auto accumulate = [&](const Block &J_win, int64_t i0, int64_t i1, int64_t j0, int64_t j1,
                                const std::vector<int64_t> &m_idx, const std::vector<int64_t> &n_idx) {
        const nd::array_t<std::complex<double>, nd::array::c_style | nd::array::forcecast> Jw =
            borrow(J_win, {nd, nd, static_cast<nd::ssize_t>(i1 - i0), static_cast<nd::ssize_t>(j1 - j0)});
        const CI mi = borrow(m_idx, {static_cast<nd::ssize_t>(m_idx.size())});
        const CI ni = borrow(n_idx, {static_cast<nd::ssize_t>(n_idx.size())});
        assemble_Z_bspline_windowed(Jw, sa, pa, ta, mi, ni, i0, i1, j0, j1, omega, eps, mu, Za, 0, nd::none(),
                                    false);
    };
    const auto range = [](int64_t lo, int64_t hi) {
        std::vector<size_t> v;
        for (int64_t q = lo; q < hi; q++) v.push_back(static_cast<size_t>(q));
        return v;
    };
    // `_schedule.mb_rows(swept_mem_mb, row_bytes)`: max(1, int(mb * 1024 * 1024 // row_bytes))
    // (an int budget, as the solver holds it; a float one, which only a gate
    // hands `_compute_Z_dense_chunked`, floors the true quotient)
    const auto mb_rows = [&](int64_t row_bytes) {
        if (swept_mem_mb == std::floor(swept_mem_mb))
            return std::max<int64_t>(1, (static_cast<int64_t>(swept_mem_mb) * 1024 * 1024) / row_bytes);
        return std::max<int64_t>(
            1, static_cast<int64_t>(std::floor((swept_mem_mb * 1024 * 1024) / static_cast<double>(row_bytes))));
    };
    // the sweep: every observer window against every source
    const int64_t chunk = mb_rows(static_cast<int64_t>(nd * nd) * N * 16);
    for (int64_t w0 = 0; w0 < N; w0 += chunk) {
        const int64_t w1 = std::min(w0 + chunk, N);
        const std::vector<size_t> rows = range(w0, w1);
        std::vector<double> a_sub(a_row.begin() + w0, a_row.begin() + w1);
        const Block J_chunk = offedge(all.take(rows), all, a_sub, fp.k, d, fp.n_qp_pair,
                                      ek_cut(ek, &rows, nullptr), fp.ladder);
        accumulate(J_chunk, w0, w1, 0, N, touching(w0, w1), all_n);
    }
    // the same-edge fixup, per edge in wire order, in row windows
    std::vector<double> t01, w01;
    gl01(fp.n_qp_same_edge, t01, w01);
    const size_t q = t01.size();
    for (size_t w = 0; w < g.per_wire.size(); w++) {
        const WireGeometry &pw = g.per_wire[w];
        const double a_w = radius_per_wire[w];
        for (size_t e = 0; e + 1 < pw.edge_offsets.size(); e++) {
            const std::vector<double> &arc = pw.edge_arc_edges[e];
            const int64_t s0 = g.seg_offsets[w] + pw.edge_offsets[e], s1 = g.seg_offsets[w] + pw.edge_offsets[e + 1];
            const size_t n = arc.size() - 1;
            if (!is_toeplitz(arc)) throw FrontNotPorted("a non-uniform edge");
            const double h = arc[1] - arc[0];
            const nd::array_t<double> table =
                fp.extended_kernel ? seg_seg_static_moments_bspline_uniform_ek_table(h, a_w, n, d, a_w)
                                   : seg_seg_static_moments_bspline_uniform_table(h, a_w, n, d);
            const CF tab = CF::borrow(table.buffer());
            // `_seg_seg_reg_geometry`'s O(N) half
            std::vector<double> s_q(n * q), u_q(n * q), w_q(n * q);
            for (size_t i = 0; i < n; i++) {
                const double hs = arc[i + 1] - arc[i];
                for (size_t j = 0; j < q; j++) {
                    s_q[i * q + j] = arc[i] + t01[j] * hs;
                    u_q[i * q + j] = (t01[j] * hs) * 1.0;
                    w_q[i * q + j] = (w01[j] * hs) * 1.0;
                }
            }
            std::vector<double> wu(static_cast<size_t>(nd) * n * q);
            for (int p = 0; p < nd; p++)
                for (size_t i = 0; i < n * q; i++) {
                    const double u = u_q[i];
                    const double up = p == 0 ? 1.0 : p == 1 ? u : p == 2 ? u * u : std::pow(u, p);
                    wu[static_cast<size_t>(p) * n * q + i] = w_q[i] * up;
                }
            const CF sq_all = borrow(s_q, {static_cast<nd::ssize_t>(n * q)});
            const CF wu_col = borrow(wu, {nd, static_cast<nd::ssize_t>(n), static_cast<nd::ssize_t>(q)});
            const std::vector<int64_t> e_idx = touching(s0, s1);
            const std::vector<size_t> cols = range(s0, s1);
            const int64_t corr_chunk = mb_rows(static_cast<int64_t>(nd * nd) * static_cast<int64_t>(n) * 16);
            for (int64_t r0 = s0; r0 < s1; r0 += corr_chunk) {
                const int64_t r1 = std::min(r0 + corr_chunk, s1);
                const std::vector<size_t> rows = range(r0, r1);
                std::vector<double> a_sub(a_row.begin() + r0, a_row.begin() + r1);
                const Block J_win = offedge(all.take(rows), all.take(cols), a_sub, fp.k, d, fp.n_qp_pair,
                                            ek_cut(ek, &rows, &cols), fp.ladder);
                const size_t lo = static_cast<size_t>(r0 - s0), hi = static_cast<size_t>(r1 - s0);
                const nd::array_t<double> A_st_w = toeplitz_gather_rows(tab, n, lo, hi);
                // the row window's reg geometry: R (n_row q, n q), wu_row
                std::vector<double> s_row(s_q.begin() + static_cast<std::ptrdiff_t>(lo * q),
                                          s_q.begin() + static_cast<std::ptrdiff_t>(hi * q));
                const CF sr = borrow(s_row, {static_cast<nd::ssize_t>(s_row.size())});
                const nd::array_t<double> R = reg_distance_table(sr, sq_all, a_w * a_w);
                std::vector<double> wu_r(static_cast<size_t>(nd) * (hi - lo) * q);
                for (int p = 0; p < nd; p++)
                    for (size_t i = 0; i < (hi - lo) * q; i++)
                        wu_r[(static_cast<size_t>(p) * (hi - lo)) * q + i] =
                            wu[static_cast<size_t>(p) * n * q + lo * q + i];
                const CF wu_row = borrow(wu_r, {nd, static_cast<nd::ssize_t>(hi - lo), static_cast<nd::ssize_t>(q)});
                const std::vector<double> kv{fp.k};
                const CF ka = borrow(kv, {1});
                const CF Rv = CF::borrow(R.buffer());
                const nd::array_t<std::complex<double>> A_reg =
                    fp.extended_kernel ? seg_seg_reg_moments_bspline_swept_ek_window(Rv, wu_row, wu_col, ka, a_w)
                                       : seg_seg_reg_moments_bspline_swept_window(Rv, wu_row, wu_col, ka);
                // `[0]`: the k axis off (a contiguous leading slice)
                std::vector<cdouble> reg0(A_reg.data(), A_reg.data() + A_reg.size());
                const nd::array_t<std::complex<double>, nd::array::c_style | nd::array::forcecast> regw =
                    borrow(reg0, {nd, nd, static_cast<nd::ssize_t>(hi - lo), static_cast<nd::ssize_t>(n)});
                const nd::array_t<std::complex<double>, nd::array::c_style | nd::array::forcecast> Jw =
                    borrow(J_win, {nd, nd, static_cast<nd::ssize_t>(hi - lo), static_cast<nd::ssize_t>(n)});
                const CF stw = CF::borrow(A_st_w.buffer());
                const nd::array_t<std::complex<double>> corr = same_edge_corr_combine(stw, regw, Jw);
                accumulate(Block::adopt(corr), r0, r1, s0, s1, touching(r0, r1), e_idx);
            }
        }
    }
    return Z;
}

std::vector<cdouble> assemble_Z(const CArray &J, const Basis &b, const Geometry &g,
                                double omega, double eps, double mu, dsyrk_fn syrk) {
    const int d = b.degree, nd = d + 1;
    const int64_t N = g.n_segs_total;
    std::vector<double> td(static_cast<size_t>(N * N), 0.0);
    // numpy's DOUBLE_matmul syrk case: CblasRowMajor, CblasUpper, NoTrans
    syrk(101, 121, 111, N, 3, 1.0, g.tangents.data(), 3, 0.0, td.data(), N);
    for (int64_t i = 0; i < N; i++)
        for (int64_t j = i + 1; j < N; j++) td[static_cast<size_t>(j * N + i)] = td[static_cast<size_t>(i * N + j)];
    const nd::array_t<std::complex<double>, nd::array::c_style | nd::array::forcecast> Ja =
        borrow(J, {nd, nd, static_cast<nd::ssize_t>(N), static_cast<nd::ssize_t>(N)});
    const CI sa = borrow(b.supp_seg, {static_cast<nd::ssize_t>(b.n_basis), nd});
    const CF pa = borrow(b.polys, {static_cast<nd::ssize_t>(b.n_basis), nd, nd});
    const CF ta = borrow(td, {static_cast<nd::ssize_t>(N), static_cast<nd::ssize_t>(N)});
    const nd::array_t<std::complex<double>> Z = assemble_Z_bspline(Ja, sa, pa, ta, omega, eps, mu, d, 0);
    return std::vector<cdouble>(Z.data(), Z.data() + Z.size());
}

PortSolve solve_ports(const std::vector<cdouble> &Z, int64_t n_b, const std::vector<double> &B,
                      int64_t n_p, const std::vector<double> &kcl, int64_t n_c,
                      const razor::Lapack &lp, const nec5::NumpyLinalg &np, const SymLapack *sym) {
    const mw_times::Scope timed(mw_times::factor);
    using nec5::Op;
    PortSolve out;
    const size_t nb = static_cast<size_t>(n_b), np_ = static_cast<size_t>(n_p), nc = static_cast<size_t>(n_c);
    // `lu_factor`: Z's Fortran copy
    std::vector<cdouble> zf(nb * nb);
    for (size_t i = 0; i < nb; i++)
        for (size_t j = 0; j < nb; j++) zf[j * nb + i] = Z[i * nb + j];
    razor::SolveContext ctx;
    ctx.set_z(zf.data(), static_cast<ptrdiff_t>(n_b));
    // B as numpy holds it: complex128, C order (`np.hstack`)
    std::vector<cdouble> Bc(nb * np_);
    for (size_t q = 0; q < Bc.size(); q++) Bc[q] = cdouble(B[q], 0.0);
    std::vector<cdouble> Xc(nb * np_);  // the result, C order
    if (n_c == 0) {
        // `_lu_solve(Z, V)`: V's Fortran copy, solved in place
        std::vector<cdouble> x(nb * np_);
        for (size_t i = 0; i < nb; i++)
            for (size_t j = 0; j < np_; j++) x[j * nb + i] = Bc[i * np_ + j];
        out.info = ctx.lu_solve(lp, x.data(), static_cast<int>(n_p));
        if (out.info != 0) return out;
        out.Y.assign(np_ * np_, cdouble(0.0, 0.0));
        nec5::replay::np_matmul(np, Op{Bc.data(), n_p, n_b, 1, n_p}, Op{x.data(), n_b, n_p, 1, n_b},
                                out.Y.data(), n_p, 1);
        out.x_c_order = false;
        out.X = std::move(x);
        return out;
    }
    // rhs = [V | A^T], F order
    const size_t nr = np_ + nc;
    std::vector<cdouble> sol(nb * nr);
    for (size_t i = 0; i < nb; i++) {
        for (size_t j = 0; j < np_; j++) sol[j * nb + i] = Bc[i * np_ + j];
        for (size_t c = 0; c < nc; c++) sol[(np_ + c) * nb + i] = cdouble(kcl[c * nb + i], 0.0);
    }
    out.info = ctx.lu_solve(lp, sol.data(), static_cast<int>(nr));
    if (out.info != 0) return out;
    const cdouble *W = sol.data(), *X = sol.data() + np_ * nb;  // F views
    // `kcl_A @ X`, `kcl_A @ W`: the float operand cast to a complex C copy
    std::vector<cdouble> Ac(nc * nb);
    for (size_t q = 0; q < Ac.size(); q++) Ac[q] = cdouble(kcl[q], 0.0);
    std::vector<cdouble> AX(nc * nc), AW(nc * np_);
    nec5::replay::np_matmul(np, Op{Ac.data(), n_c, n_b, n_b, 1}, Op{X, n_b, n_c, 1, n_b}, AX.data(), n_c, 1);
    nec5::replay::np_matmul(np, Op{Ac.data(), n_c, n_b, n_b, 1}, Op{W, n_b, n_p, 1, n_b}, AW.data(), n_p, 1);
    // `scipy.linalg.solve(AX, AW)`
    std::vector<cdouble> Lam(nc * np_);
    if (n_c == 1) {
        if (AX[0] == cdouble(0.0, 0.0)) {
            out.info = 1;
            return out;
        }
        for (size_t j = 0; j < np_; j++) Lam[j] = nec5::replay::np_cdiv(AW[j], AX[0]);
    } else {
        // general: no exact zero triangle, no exact symmetry (else scipy
        // detects a structure and takes another factorization)
        bool lower0 = true, upper0 = true, is_sym = true, herm = true;
        for (size_t i = 0; i < nc; i++)
            for (size_t j = 0; j < nc; j++) {
                const cdouble a = AX[i * nc + j], at = AX[j * nc + i];
                if (i > j && a != cdouble(0.0, 0.0)) lower0 = false;
                if (i < j && a != cdouble(0.0, 0.0)) upper0 = false;
                if (a != at) is_sym = false;
                if (a != std::conj(at)) herm = false;
            }
        // a triangular / diagonal / Hermitian system takes another of
        // scipy's solvers: not ported
        if (lower0 || upper0 || herm || (is_sym && (sym == nullptr || sym->sytrf == nullptr)))
            throw FrontNotPorted("a structured KCL system");
        std::vector<cdouble> af(nc * nc), bf(nc * np_);
        for (size_t i = 0; i < nc; i++) {
            for (size_t j = 0; j < nc; j++) af[j * nc + i] = AX[i * nc + j];
            for (size_t j = 0; j < np_; j++) bf[j * nc + i] = AW[i * np_ + j];
        }
        if (is_sym) {
            // `assume_a="sym"`: zsytrf (upper, the optimal workspace) + zsytrs
            char uplo = 'U';
            int n = static_cast<int>(n_c), lda = n, info = 0, lwork = -1, nrhs = static_cast<int>(n_p);
            std::vector<int> ipiv(nc);
            cdouble wq(0.0, 0.0);
            sym->sytrf(&uplo, &n, af.data(), &lda, ipiv.data(), &wq, &lwork, &info);
            lwork = std::max(1, static_cast<int>(wq.real()));
            std::vector<cdouble> work(static_cast<size_t>(lwork));
            sym->sytrf(&uplo, &n, af.data(), &lda, ipiv.data(), work.data(), &lwork, &info);
            if (info != 0) {
                out.info = 1;
                return out;
            }
            sym->sytrs(&uplo, &n, &nrhs, af.data(), &lda, ipiv.data(), bf.data(), &n, &info);
            if (info != 0) {
                out.info = 1;
                return out;
            }
        } else {
            razor::SolveContext small;
            small.set_z(af.data(), static_cast<ptrdiff_t>(n_c));
            if (small.lu_solve(lp, bf.data(), static_cast<int>(n_p)) != 0) {
                out.info = 1;
                return out;
            }
        }
        for (size_t i = 0; i < nc; i++)
            for (size_t j = 0; j < np_; j++) Lam[i * np_ + j] = bf[j * nc + i];
    }
    // `W - X @ Lam`
    std::vector<cdouble> XL(nb * np_);
    nec5::replay::np_matmul(np, Op{X, n_b, n_c, 1, n_b}, Op{Lam.data(), n_c, n_p, n_p, 1}, XL.data(), n_p, 1);
    for (size_t i = 0; i < nb; i++)
        for (size_t j = 0; j < np_; j++) {
            const cdouble w = W[j * nb + i], xl = XL[i * np_ + j];
            Xc[i * np_ + j] = cdouble(w.real() - xl.real(), w.imag() - xl.imag());
        }
    // `Y = B.T @ X`
    out.Y.assign(np_ * np_, cdouble(0.0, 0.0));
    nec5::replay::np_matmul(np, Op{Bc.data(), n_p, n_b, 1, n_p}, Op{Xc.data(), n_b, n_p, n_p, 1},
                            out.Y.data(), n_p, 1);
    out.X = Xc;
    return out;
}

std::vector<cdouble> spline_eval_wire(const Basis &b, const Geometry &g, const std::vector<cdouble> &coeffs,
                                      size_t w, std::vector<double> s, bool derivative) {
    using nec5::replay::np_cmul;
    const int d = b.degree, nd = d + 1;

    const std::vector<double> &arc = g.per_wire[w].arc_at_knot;
    const std::vector<double> &kv = b.wire_knots[w];
    const double lo = kv.front(), hi = kv.back();
    std::vector<double> s_eval = std::move(s);
    for (double &v : s_eval) v = std::min(std::max(v, lo), hi);  // np.clip (NaN stays NaN)
    const size_t n = s_eval.size();
    std::vector<cdouble> out(n, cdouble(0.0, 0.0));
    if (n == 0) return out;
    if (!derivative)
        for (double v : s_eval)
            if (std::isnan(v)) throw FrontNotPorted("a NaN current sample position");
    // `_wire_span_polynomials`: band[s, b] the polynomial of basis s + b
    const size_t n_seg = arc.size() - 1;
    std::vector<double> band(n_seg * nd * nd, 0.0);
    std::vector<cdouble> band_c(n_seg * nd, cdouble(0.0, 0.0));
    for (const auto &jm : b.wire_globals[w]) {
        const int64_t j = jm.first, m = jm.second;
        const int64_t seg_lo = std::max<int64_t>(0, j - d), seg_hi = std::min<int64_t>(static_cast<int64_t>(n_seg), j + 1);
        for (int wi = 0; wi < nd; wi++) {
            const int64_t span = seg_lo + wi;
            if (!(span < seg_hi)) continue;
            const int64_t bb = j - span;
            for (int p = 0; p < nd; p++)
                band[(static_cast<size_t>(span) * nd + bb) * nd + p] =
                    b.polys[(static_cast<size_t>(m) * nd + wi) * nd + p];
            band_c[static_cast<size_t>(span) * nd + bb] = coeffs[static_cast<size_t>(m)];
        }
    }
    std::vector<cdouble> sp(n_seg * nd, cdouble(0.0, 0.0));
    for (int bb = 0; bb < nd; bb++)
        for (size_t s2 = 0; s2 < n_seg; s2++)
            for (int p = 0; p < nd; p++) {
                const cdouble t = np_cmul(band_c[s2 * nd + bb], cdouble(band[(s2 * nd + bb) * nd + p], 0.0));
                cdouble &o = sp[s2 * nd + p];
                o = cdouble(o.real() + t.real(), o.imag() + t.imag());
            }
    // `_eval_span_polynomials`
    for (size_t q = 0; q < n; q++) {
        const double sv = s_eval[q];
        int64_t span = static_cast<int64_t>(std::upper_bound(arc.begin(), arc.end(), sv) - arc.begin()) - 1;
        span = std::min<int64_t>(std::max<int64_t>(span, 0), static_cast<int64_t>(n_seg) - 1);
        const double u = sv - arc[static_cast<size_t>(span)];
        const cdouble *coef = &sp[static_cast<size_t>(span) * nd];
        cdouble val;
        if (!derivative) {
            val = coef[d];
            for (int p = d - 1; p >= 0; p--) {
                const cdouble t = np_cmul(val, cdouble(u, 0.0));
                val = cdouble(t.real() + coef[p].real(), t.imag() + coef[p].imag());
            }
            out[q] = cdouble(0.0 + val.real(), 0.0 + val.imag());  // `I_out += ...`
        } else {
            val = np_cmul(cdouble(static_cast<double>(d), 0.0), coef[d]);
            for (int p = d - 1; p >= 1; p--) {
                const cdouble t = np_cmul(val, cdouble(u, 0.0));
                const cdouble pc = np_cmul(cdouble(static_cast<double>(p), 0.0), coef[p]);
                val = cdouble(t.real() + pc.real(), t.imag() + pc.imag());
            }
            if (std::isnan(sv)) val = cdouble(std::nan(""), std::nan(""));
            out[q] = val;
        }
    }
    return out;
}

std::vector<std::vector<cdouble>> spline_readout(const Basis &b, const Geometry &g,
                                                 const WireDedup &dd,
                                                 const std::vector<cdouble> &coeffs,
                                                 const std::vector<std::vector<double>> *s_array,
                                                 bool derivative) {
    const size_t n_solved = g.per_wire.size();
    // the solved wires' sample positions (`dedup.to_solved`)
    std::vector<std::vector<double>> s_solved;
    if (s_array != nullptr) {
        if (dd.active) {
            for (int64_t w : dd.kept) s_solved.push_back((*s_array)[static_cast<size_t>(w)]);
        } else {
            s_solved = *s_array;
        }
    }
    std::vector<std::vector<cdouble>> solved(n_solved);
    for (size_t w = 0; w < n_solved; w++)
        solved[w] = spline_eval_wire(b, g, coeffs, w, s_array == nullptr ? g.per_wire[w].arc_at_knot : s_solved[w],
                                     derivative);
    if (!dd.active) return solved;
    // `dedup.to_input`: a dropped copy reads zeros
    std::vector<std::vector<cdouble>> input(static_cast<size_t>(dd.n_input));
    for (size_t i = 0; i < dd.kept.size(); i++) input[static_cast<size_t>(dd.kept[i])] = solved[i];
    for (const auto &r : dd.removed) {
        const size_t n = s_array != nullptr ? (*s_array)[static_cast<size_t>(r.copy)].size()
                                            : input[static_cast<size_t>(r.twin)].size();
        input[static_cast<size_t>(r.copy)].assign(n, cdouble(0.0, 0.0));
    }
    return input;
}

// ---------------------------------------------------------------------------
// The exact ring-kernel correction (momwire#1517 step 2)
// ---------------------------------------------------------------------------
namespace {

inline double ndot3(const double *a, const double *b) { return (a[0] * b[0] + a[1] * b[1]) + a[2] * b[2]; }
inline double nnorm3(const double *v) { return std::sqrt((v[0] * v[0] + v[1] * v[1]) + v[2] * v[2]); }

// CPython's float `//` (`float_floor_div`).
double py_floordiv(double vx, double wx) {
    double mod = std::fmod(vx, wx);
    double div = (vx - mod) / wx;
    if (mod != 0.0) {
        if ((wx < 0) != (mod < 0)) div -= 1.0;
    }
    double floordiv;
    if (div != 0.0) {
        floordiv = std::floor(div);
        if (div - floordiv > 0.5) floordiv += 1.0;
    } else {
        floordiv = std::copysign(0.0, vx / wx);
    }
    return floordiv;
}

using Ranges = std::vector<std::pair<int64_t, int64_t>>;

Ranges subtract_ranges(const Ranges &ranges, Ranges holes) {
    Ranges out;
    std::sort(holes.begin(), holes.end());
    for (const auto &ab : ranges) {
        int64_t cur = ab.first;
        for (const auto &cd_ : holes) {
            if (cd_.second <= cur || cd_.first >= ab.second) continue;
            if (cd_.first > cur) out.push_back({cur, cd_.first});
            cur = std::max(cur, cd_.second);
        }
        if (cur < ab.second) out.push_back({cur, ab.second});
    }
    return out;
}

Ranges merge_ranges(Ranges r) {
    std::sort(r.begin(), r.end());
    Ranges out;
    for (const auto &ab : r) {
        if (!out.empty() && out.back().second == ab.first)
            out.back().second = ab.second;
        else
            out.push_back(ab);
    }
    return out;
}

constexpr int64_t TOEPLITZ_MIN_PAIRS = 4096;
constexpr double RUN_TOL_EPS = 16.0;
constexpr double TIE_REL = 1e-9;

// One boundary pair per diagonal (`_boundary_pairs`): for each m, the first
// of row 0, row nA - 1, column 0, column nB - 1 that holds it.
struct Boundary {
    bool row;
    int64_t at;
    std::vector<size_t> idx;      // positions into m
    std::vector<int64_t> other;   // the other index
};
std::vector<Boundary> boundary_pairs(const std::vector<int64_t> &m, int64_t nA, int64_t nB, int64_t sA,
                                     int64_t sB) {
    std::vector<Boundary> out;
    std::vector<bool> left(m.size(), true);
    const std::pair<bool, int64_t> lines[4] = {{true, 0}, {true, nA - 1}, {false, 0}, {false, nB - 1}};
    for (const auto &ln : lines) {
        bool any_left = false;
        for (bool v : left) any_left = any_left || v;
        if (!any_left) break;
        Boundary bd{ln.first, ln.second, {}, {}};
        for (size_t q = 0; q < m.size(); q++) {
            if (!left[q]) continue;
            int64_t other;
            bool ok;
            if (ln.first) {
                other = sB * (m[q] + sA * ln.second);
                ok = other >= 0 && other < nB;
            } else {
                other = sA * (sB * ln.second - m[q]);
                ok = other >= 0 && other < nA;
            }
            if (ok) {
                bd.idx.push_back(q);
                bd.other.push_back(other);
                left[q] = false;
            }
        }
        if (!bd.idx.empty()) out.push_back(std::move(bd));
    }
    for (bool v : left)
        if (v) throw std::logic_error("a diagonal missed every boundary line");
    return out;
}

// `_exact_kernel_base.BaseRows`, without the cross mode and without its
// cross-window cache (a cached diagonal is the same representative's value,
// so the cache moves no bit).
class BaseRows {
  public:
    struct Edge {
        int64_t l0, l1;
        const std::vector<double> *arc;
        double a_w;
    };
    BaseRows(const Geometry &g, const std::vector<double> &seg_a, const std::vector<int64_t> &S,
             std::vector<Edge> edges, const FillParams &fp, const EK &ek)
        : g_(g), seg_a_(seg_a), S_(S), n_(static_cast<int64_t>(S.size())), fp_(fp), ek_(ek),
          nd_(fp.degree + 1), edges_(std::move(edges)) {
        const int64_t n = n_;
        std::vector<double> edge_of(static_cast<size_t>(n), -1.0);
        std::vector<int64_t> edge_of_i(static_cast<size_t>(n), -1);
        for (size_t e = 0; e < edges_.size(); e++)
            for (int64_t q = edges_[e].l0; q < edges_[e].l1; q++) {
                edge_of[static_cast<size_t>(q)] = static_cast<double>(e);
                edge_of_i[static_cast<size_t>(q)] = static_cast<int64_t>(e);
            }
        for (const auto &e : edges_) edge_bounds_.push_back(e.l0);
        std::sort(edge_bounds_.begin(), edge_bounds_.end());
        const double *axis = &g.tangents[3 * static_cast<size_t>(S[0])];
        std::vector<double> Ls(3 * static_cast<size_t>(n)), Rs(3 * static_cast<size_t>(n)), u(3 * static_cast<size_t>(n));
        std::vector<double> h(static_cast<size_t>(n));
        double X = 0.0;
        for (int64_t i = 0; i < n; i++)
            for (int c = 0; c < 3; c++) {
                Ls[3 * i + c] = g.seg_l[3 * static_cast<size_t>(S[i]) + c];
                Rs[3 * i + c] = g.seg_r[3 * static_cast<size_t>(S[i]) + c];
            }
        double xl = 0.0, xr = 0.0;
        for (size_t q = 0; q < Ls.size(); q++) {
            xl = std::max(xl, std::fabs(Ls[q]));
            xr = std::max(xr, std::fabs(Rs[q]));
            u[q] = Rs[q] - Ls[q];
        }
        X = n ? std::max(xl, xr) : 0.0;
        for (int64_t i = 0; i < n; i++) h[i] = nnorm3(&u[3 * i]);
        rho_ = (RUN_TOL_EPS * 2.220446049250313e-16) * X;
        const double nan = std::nan("");
        std::vector<double> sgn(static_cast<size_t>(n)), a(static_cast<size_t>(n)), gi(static_cast<size_t>(n)),
            gj(static_cast<size_t>(n)), bi(static_cast<size_t>(n), nan), bj(static_cast<size_t>(n));
        for (int64_t i = 0; i < n; i++) {
            const size_t s = static_cast<size_t>(S[i]);
            sgn[i] = ndot3(&g.tangents[3 * s], axis) >= 0.0 ? 1.0 : -1.0;
            a[i] = seg_a[s];
            gi[i] = ek.on ? static_cast<double>(ek.group_i[s]) : nan;
            gj[i] = ek.on ? static_cast<double>(ek.group_j[s]) : nan;
            bj[i] = (ek.on && ek.b_kind == EK::per_col) ? ek.b_j[s] : (ek.on && ek.b_kind == EK::scalar ? ek.b : nan);
        }
        // runs: the exact fields (RUN_FIELDS less "h") ...
        std::vector<bool> change(static_cast<size_t>(n), false);
        for (const std::vector<double> *f : {&edge_of, &sgn, &a, &gi, &gj, &bi, &bj})
            for (int64_t i = 1; i < n; i++) {
                const double x = (*f)[i], y = (*f)[i - 1];
                const bool same = x == y || (std::isnan(x) && std::isnan(y));
                if (!same) change[i] = true;
            }
        std::vector<int64_t> starts{0};
        for (int64_t i = 1; i < n; i++)
            if (change[i]) starts.push_back(i);
        Ranges pieces;
        if (n)
            for (size_t q = 0; q < starts.size(); q++)
                pieces.push_back({starts[q], q + 1 < starts.size() ? starts[q + 1] : n});
        // ... then the geometric rule: `_same_vector`, `_progressions`
        Ranges cut;
        for (const auto &p : pieces) {
            int64_t aa = p.first;
            const int64_t s1 = p.second;
            while (aa < s1) {
                int64_t b = aa + 1, block = 64;
                while (b < s1) {
                    const int64_t e = std::min(s1, b + block);
                    int64_t off = -1;
                    for (int64_t q = b; q < e; q++) {
                        double d[3];
                        for (int c = 0; c < 3; c++) d[c] = u[3 * q + c] - u[3 * aa + c];
                        if ((d[0] * d[0] + d[1] * d[1]) + d[2] * d[2] > rho_ * rho_) {
                            off = q;
                            break;
                        }
                    }
                    if (off >= 0) {
                        b = off;
                        break;
                    }
                    b = e;
                    block *= 2;
                }
                cut.push_back({aa, b});
                aa = b;
            }
        }
        Ranges prog;
        for (const auto &c0 : cut) {
            Ranges out, todo{c0};
            while (!todo.empty()) {
                const auto ab = todo.back();
                todo.pop_back();
                const int64_t aa = ab.first, b = ab.second;
                if (b - aa > 1) {
                    double step[3];
                    for (int c = 0; c < 3; c++)
                        step[c] = (Ls[3 * (b - 1) + c] - Ls[3 * aa + c]) / static_cast<double>(b - 1 - aa);
                    double dev = -1.0;
                    for (const std::vector<double> *E : {&Ls, &Rs}) {
                        double mx = -1.0;
                        for (int64_t i = aa; i < b; i++) {
                            double ssum = 0.0;
                            for (int c = 0; c < 3; c++) {
                                const double line = static_cast<double>(i - aa) * step[c];
                                const double d = ((*E)[3 * i + c] - (*E)[3 * aa + c]) - line;
                                ssum += d * d;
                            }
                            mx = std::max(mx, ssum);
                        }
                        dev = std::max(dev, mx);
                    }
                    if (dev > rho_ * rho_) {
                        const int64_t mid = (aa + b) / 2;
                        todo.push_back({aa, mid});
                        todo.push_back({mid, b});
                        continue;
                    }
                }
                out.push_back(ab);
            }
            std::sort(out.begin(), out.end());
            prog.insert(prog.end(), out.begin(), out.end());
        }
        for (const auto &p : prog) {
            Run r{p.first, p.second, h[p.first], sgn[p.first], edge_of_i[p.first]};
            runs_.push_back(r);
            std::array<double, 3> st;
            for (int c = 0; c < 3; c++)
                st[c] = p.second - p.first > 1
                            ? (Ls[3 * (p.second - 1) + c] - Ls[3 * p.first + c]) / static_cast<double>(p.second - 1 - p.first)
                            : u[3 * p.first + c];
            step_.push_back(st);
        }
        run_of_.assign(static_cast<size_t>(n), 0);
        for (size_t r = 0; r < runs_.size(); r++)
            for (int64_t q = runs_[r].s0; q < runs_[r].s1; q++) run_of_[q] = static_cast<int64_t>(r);
        bool phase_limited = false;
        for (const auto &t : fp.ladder) phase_limited = phase_limited || t.second < PHASE_LIMITED_BELOW;
        for (const auto &r : runs_)
            phase_tie_.push_back(phase_limited &&
                                 std::fabs(std::fabs(fp.k) * r.h - PHASE_KL_CEILING) <= TIE_REL * PHASE_KL_CEILING);
        for (const auto &t : fp.ladder) thresholds_.push_back(t.first);
        std::vector<int64_t> order(runs_.size());
        for (size_t r = 0; r < order.size(); r++) order[r] = static_cast<int64_t>(r);
        std::stable_sort(order.begin(), order.end(), [&](int64_t x, int64_t y) { return runs_[x].h < runs_[y].h; });
        class_of_.assign(runs_.size(), 0);
        for (int64_t r : order) {
            if (!classes_.empty() && runs_[r].h - runs_[classes_.back()[0]].h <= rho_)
                classes_.back().push_back(r);
            else
                classes_.push_back({r});
            class_of_[r] = static_cast<int64_t>(classes_.size()) - 1;
        }
        for (auto &c : classes_)
            std::stable_sort(c.begin(), c.end(), [&](int64_t x, int64_t y) {
                return -(runs_[x].s1 - runs_[x].s0) < -(runs_[y].s1 - runs_[y].s0);
            });
        edge_runs_.assign(edges_.size(), {});
        for (size_t r = 0; r < runs_.size(); r++) edge_runs_[runs_[r].edge].push_back(static_cast<int64_t>(r));
    }

    int64_t cache_entries_needed() const {
        int64_t need = 0;
        for (const auto &c : classes_)
            for (int64_t A : c) {
                const int64_t la = runs_[A].s1 - runs_[A].s0;
                for (int64_t B : c) {
                    const int64_t lb = runs_[B].s1 - runs_[B].s0;
                    if (!eligible(la, lb)) break;
                    if (same_step(A, B)) need += la + lb - 1;
                }
            }
        return need;
    }

    // out (nd, nd, r1 - r0, n) -= the base rows [r0, r1)
    void subtract_from(std::vector<cdouble> &out, int64_t r0, int64_t r1) {
        out_ = &out;
        r0_ = r0;
        nr_ = r1 - r0;
        if (edges_.size() > 1) fill(r0, r1, false, -1);
        for (size_t e = 0; e < edges_.size(); e++) {
            const int64_t l0 = edges_[e].l0, l1 = edges_[e].l1;
            const int64_t lo = std::max(l0, r0), hi = std::min(l1, r1);
            if (lo >= hi) continue;
            const int64_t A = run_of_[l0];
            if (eligible(hi - lo, l1 - l0) && foldable(static_cast<int64_t>(e), A)) {
                toeplitz(A, lo, hi, A, true, static_cast<int64_t>(e), true);
                continue;
            }
            const Blk st = static_rows(static_cast<int64_t>(e), lo - l0, hi - l0);
            if (edge_runs_[e].size() == 1 && !eligible(hi - lo, l1 - l0)) {
                Blk rg = reg(static_cast<int64_t>(e), iota(lo, hi), iota(l0, l1));
                // `static + reg`: the real block promoted, then added
                for (size_t q = 0; q < rg.v.size(); q++)
                    rg.v[q] = cdouble(st.v[q].real() + rg.v[q].real(), 0.0 + rg.v[q].imag());
                put(lo, hi, l0, l1, rg, 0, 0);
                continue;
            }
            put_real(lo, hi, l0, l1, st);
            fill(lo, hi, true, static_cast<int64_t>(e));
        }
        out_ = nullptr;
    }

  private:
    struct Run {
        int64_t s0, s1;
        double h, sgn;
        int64_t edge;
    };
    // a (nd, nd, nr, nc) block, C order
    struct Blk {
        int64_t nr = 0, nc = 0;
        std::vector<cdouble> v;
        cdouble &at(int nd, int p, int q, int64_t i, int64_t j) {
            return v[static_cast<size_t>(((static_cast<int64_t>(p) * nd + q) * nr + i) * nc + j)];
        }
    };

    static std::vector<int64_t> iota(int64_t a, int64_t b) {
        std::vector<int64_t> v;
        for (int64_t q = a; q < b; q++) v.push_back(q);
        return v;
    }
    static bool eligible(int64_t nr, int64_t nc) { return nr * nc >= std::max<int64_t>(TOEPLITZ_MIN_PAIRS, 2); }

    bool same_step(int64_t A, int64_t B) const {
        if (A == B) return true;
        const Run &ra = runs_[A], &rb = runs_[B];
        const int64_t steps = std::min(ra.s1 - ra.s0, rb.s1 - rb.s0) - 1;
        if (steps <= 0) return true;
        const double f = ra.sgn * rb.sgn;
        double d[3];
        for (int c = 0; c < 3; c++) d[c] = f * step_[B][c] - step_[A][c];
        return static_cast<double>(steps) * nnorm3(d) <= rho_;
    }

    bool foldable(int64_t e, int64_t A) {
        const Run &r = runs_[A];
        if (r.s0 != edges_[e].l0 || r.s1 != edges_[e].l1) return false;
        return is_toeplitz(*edges_[e].arc);
    }

    // `_seg_seg_static_moments(arc, a_w, d, ek_se, rows=[i0, i1))`: the
    // Toeplitz table's gather (a non-uniform edge is numpy's dense branch)
    const nd::array_t<double> &table(int64_t e) {
        auto it = tables_.find(e);
        if (it != tables_.end()) return it->second;
        const auto &arc = *edges_[e].arc;
        if (!is_toeplitz(arc)) throw FrontNotPorted("a non-uniform edge");
        const double h = arc[1] - arc[0], a_w = edges_[e].a_w;
        const size_t N = arc.size() - 1;
        nd::array_t<double> t = fp_.extended_kernel
                                    ? seg_seg_static_moments_bspline_uniform_ek_table(h, a_w, N, fp_.degree, a_w)
                                    : seg_seg_static_moments_bspline_uniform_table(h, a_w, N, fp_.degree);
        return tables_.emplace(e, std::move(t)).first->second;
    }
    // real (nd, nd, i1 - i0, N), held in the real parts
    Blk static_rows(int64_t e, int64_t i0, int64_t i1) {
        const nd::array_t<double> &t = table(e);
        const int64_t N = static_cast<int64_t>(edges_[e].arc->size()) - 1;
        Blk b;
        b.nr = i1 - i0;
        b.nc = N;
        b.v.resize(static_cast<size_t>(nd_ * nd_ * b.nr * b.nc));
        const double *tp = t.data();
        const int64_t L = 2 * N - 1;
        for (int p = 0; p < nd_; p++)
            for (int q = 0; q < nd_; q++)
                for (int64_t i = i0; i < i1; i++)
                    for (int64_t j = 0; j < N; j++)
                        b.at(nd_, p, q, i - i0, j) = cdouble(tp[(static_cast<int64_t>(p) * nd_ + q) * L + (j - i + N - 1)], 0.0);
        return b;
    }

    Blk offedge_rows(const std::vector<int64_t> &rows, const std::vector<int64_t> &cols) {
        Segs si, sj;
        std::vector<size_t> gr, gc;
        for (int64_t r : rows) gr.push_back(static_cast<size_t>(S_[r]));
        for (int64_t c : cols) gc.push_back(static_cast<size_t>(S_[c]));
        Segs all;
        all.l = g_.seg_l;
        all.r = g_.seg_r;
        si = all.take(gr);
        sj = all.take(gc);
        std::vector<double> a_rows;
        for (size_t s : gr) a_rows.push_back(seg_a_[s]);
        const EK ekc = ek_cut(ek_, &gr, &gc);
        Blk b;
        b.nr = static_cast<int64_t>(rows.size());
        b.nc = static_cast<int64_t>(cols.size());
        const Block o = offedge(si, sj, a_rows, fp_.k, fp_.degree, fp_.n_qp_pair, ekc, fp_.ladder);
        b.v.assign(o.begin(), o.end());
        return b;
    }

    // same-edge smooth-kernel moments of edge e (rows / cols group-local)
    Blk reg(int64_t e, const std::vector<int64_t> &rows, const std::vector<int64_t> &cols) {
        const int64_t l0 = edges_[e].l0;
        const auto &arc = *edges_[e].arc;
        const double a_w = edges_[e].a_w;
        if (nodes_e_ != e) {
            // `_seg_seg_reg_nodes`
            std::vector<double> t01, w01;
            gl01(fp_.n_qp_same_edge, t01, w01);
            q_ = t01.size();
            const size_t N = arc.size() - 1;
            s_q_.assign(N * q_, 0.0);
            wu_.assign(static_cast<size_t>(nd_) * N * q_, 0.0);
            std::vector<double> u_q(N * q_), w_q(N * q_);
            for (size_t i = 0; i < N; i++) {
                const double hs = arc[i + 1] - arc[i];
                for (size_t j = 0; j < q_; j++) {
                    s_q_[i * q_ + j] = arc[i] + t01[j] * hs;
                    u_q[i * q_ + j] = (t01[j] * hs) * 1.0;
                    w_q[i * q_ + j] = (w01[j] * hs) * 1.0;
                }
            }
            for (int p = 0; p < nd_; p++)
                for (size_t i = 0; i < N * q_; i++) {
                    const double uu = u_q[i];
                    const double up = p == 0 ? 1.0 : p == 1 ? uu : p == 2 ? uu * uu : std::pow(uu, p);
                    wu_[static_cast<size_t>(p) * N * q_ + i] = w_q[i] * up;
                }
            n_edge_ = N;
            nodes_e_ = e;
        }
        const size_t q = q_, N = n_edge_;
        const size_t nr = rows.size(), nc = cols.size();
        std::vector<double> R(nr * q * nc * q);
        const double a2 = a_w * a_w;
        for (size_t i = 0; i < nr; i++)
            for (size_t qi = 0; qi < q; qi++) {
                const double sr = s_q_[static_cast<size_t>(rows[i] - l0) * q + qi];
                for (size_t j = 0; j < nc; j++)
                    for (size_t qj = 0; qj < q; qj++) {
                        double d = sr - s_q_[static_cast<size_t>(cols[j] - l0) * q + qj];
                        d *= d;
                        d += a2;
                        R[(i * q + qi) * nc * q + j * q + qj] = std::sqrt(d);
                    }
            }
        std::vector<double> wr(static_cast<size_t>(nd_) * nr * q), wc(static_cast<size_t>(nd_) * nc * q);
        for (int p = 0; p < nd_; p++) {
            for (size_t i = 0; i < nr; i++)
                for (size_t qi = 0; qi < q; qi++)
                    wr[(static_cast<size_t>(p) * nr + i) * q + qi] =
                        wu_[static_cast<size_t>(p) * N * q + static_cast<size_t>(rows[i] - l0) * q + qi];
            for (size_t j = 0; j < nc; j++)
                for (size_t qj = 0; qj < q; qj++)
                    wc[(static_cast<size_t>(p) * nc + j) * q + qj] =
                        wu_[static_cast<size_t>(p) * N * q + static_cast<size_t>(cols[j] - l0) * q + qj];
        }
        const CF Ra = borrow(R, {static_cast<nd::ssize_t>(nr * q), static_cast<nd::ssize_t>(nc * q)});
        const CF wra = borrow(wr, {nd_, static_cast<nd::ssize_t>(nr), static_cast<nd::ssize_t>(q)});
        const CF wca = borrow(wc, {nd_, static_cast<nd::ssize_t>(nc), static_cast<nd::ssize_t>(q)});
        const std::vector<double> kv{fp_.k};
        const CF ka = borrow(kv, {1});
        const nd::array_t<std::complex<double>> A =
            fp_.extended_kernel ? seg_seg_reg_moments_bspline_swept_ek_window(Ra, wra, wca, ka, a_w)
                                : seg_seg_reg_moments_bspline_swept_window(Ra, wra, wca, ka);
        Blk b;
        b.nr = static_cast<int64_t>(nr);
        b.nc = static_cast<int64_t>(nc);
        b.v.assign(A.data(), A.data() + static_cast<size_t>(nd_ * nd_) * nr * nc);
        return b;
    }

    Blk eval(bool is_reg, int64_t e, const std::vector<int64_t> &rows, const std::vector<int64_t> &cols) {
        return is_reg ? reg(e, rows, cols) : offedge_rows(rows, cols);
    }

    // out[rows a:b, cols c:f] -= blk[rows bi.., cols bj..] (a sub-block of blk)
    void put(int64_t a, int64_t b, int64_t c, int64_t f, Blk &blk, int64_t bi, int64_t bj) {
        std::vector<cdouble> &o = *out_;
        for (int p = 0; p < nd_; p++)
            for (int q = 0; q < nd_; q++)
                for (int64_t i = a; i < b; i++)
                    for (int64_t j = c; j < f; j++) {
                        cdouble &d = o[static_cast<size_t>(((static_cast<int64_t>(p) * nd_ + q) * nr_ + (i - r0_)) * n_ + j)];
                        const cdouble s = blk.at(nd_, p, q, bi + (i - a), bj + (j - c));
                        d = cdouble(d.real() - s.real(), d.imag() - s.imag());
                    }
    }
    // out -= a real block (complex minus float: the imaginary part minus 0)
    void put_real(int64_t a, int64_t b, int64_t c, int64_t f, const Blk &blk) {
        std::vector<cdouble> &o = *out_;
        for (int p = 0; p < nd_; p++)
            for (int q = 0; q < nd_; q++)
                for (int64_t i = a; i < b; i++)
                    for (int64_t j = c; j < f; j++) {
                        cdouble &d = o[static_cast<size_t>(((static_cast<int64_t>(p) * nd_ + q) * nr_ + (i - r0_)) * n_ + j)];
                        const double s = blk.v[static_cast<size_t>(((static_cast<int64_t>(p) * nd_ + q) * blk.nr + (i - a)) * blk.nc + (j - c))].real();
                        d = cdouble(d.real() - s, d.imag() - 0.0);
                    }
    }

    Ranges span(bool is_reg, int64_t e) const {
        const int64_t l0 = edges_[e].l0, l1 = edges_[e].l1;
        if (is_reg) return {{l0, l1}};
        Ranges r;
        if (0 < l0) r.push_back({0, l0});
        if (l1 < n_) r.push_back({l1, n_});
        return r;
    }

    void direct(const Ranges &row_ranges, const Ranges &col_ranges, bool is_reg, int64_t e) {
        std::vector<int64_t> rows, cols;
        for (const auto &r : row_ranges)
            for (int64_t q = r.first; q < r.second; q++) rows.push_back(q);
        for (const auto &c : col_ranges)
            for (int64_t q = c.first; q < c.second; q++) cols.push_back(q);
        Blk vals = eval(is_reg, e, rows, cols);
        int64_t i = 0;
        for (const auto &r : row_ranges) {
            int64_t j = 0;
            for (const auto &c : col_ranges) {
                put(r.first, r.second, c.first, c.second, vals, i, j);
                j += c.second - c.first;
            }
            i += r.second - r.first;
        }
    }

    void fill(int64_t lo, int64_t hi, bool is_reg, int64_t e) {
        Ranges shorts;
        for (int64_t A = run_of_[lo]; A <= run_of_[hi - 1]; A++) {
            const Run &ra = runs_[A];
            const int64_t ra0 = std::max(ra.s0, lo), ra1 = std::min(ra.s1, hi);
            std::vector<int64_t> toe;
            if (is_reg) {
                for (int64_t B : edge_runs_[ra.edge]) {
                    const Run &rb = runs_[B];
                    if (class_of_[B] == class_of_[A] && eligible(ra1 - ra0, rb.s1 - rb.s0) && same_step(A, B))
                        toe.push_back(B);
                }
            } else if (!phase_tie_[A]) {
                for (int64_t B : classes_[class_of_[A]]) {
                    const Run &rb = runs_[B];
                    if (!eligible(ra1 - ra0, rb.s1 - rb.s0)) break;
                    if (rb.edge != ra.edge && !phase_tie_[B] && same_step(A, B)) toe.push_back(B);
                }
            }
            if (toe.empty()) {
                shorts.push_back({ra0, ra1});
                continue;
            }
            for (int64_t B : toe) toeplitz(A, ra0, ra1, B, is_reg, e, false);
            Ranges holes;
            for (int64_t B : toe) holes.push_back({runs_[B].s0, runs_[B].s1});
            const Ranges rest = subtract_ranges(span(is_reg, ra.edge), holes);
            if (!rest.empty()) direct({{ra0, ra1}}, rest, is_reg, e);
        }
        if (shorts.empty()) return;
        shorts = merge_ranges(shorts);
        if (is_reg) {
            direct(shorts, {{edges_[e].l0, edges_[e].l1}}, is_reg, e);
            return;
        }
        std::vector<int64_t> rows;
        for (const auto &r : shorts)
            for (int64_t q = r.first; q < r.second; q++) rows.push_back(q);
        Blk vals = eval(false, e, rows, iota(0, n_));
        int64_t i = 0;
        for (const auto &r : shorts) {
            const int64_t a = r.first, b = r.second;
            // `_split_by_edge`: [a, b) cut at the edge starts
            std::vector<int64_t> pts{a};
            for (int64_t x : edge_bounds_)
                if (a < x && x < b) pts.push_back(x);
            pts.push_back(b);
            for (size_t k2 = 0; k2 + 1 < pts.size(); k2++) {
                const int64_t x = pts[k2], y = pts[k2 + 1];
                const Edge &ed = edges_[runs_[run_of_[x]].edge];
                for (int p = 0; p < nd_; p++)
                    for (int q = 0; q < nd_; q++)
                        for (int64_t ii = i + x - a; ii < i + y - a; ii++)
                            for (int64_t j = ed.l0; j < ed.l1; j++) vals.at(nd_, p, q, ii, j) = cdouble(0.0, 0.0);
            }
            put(a, b, 0, n_, vals, i, 0);
            i += b - a;
        }
    }

    std::vector<int64_t> tie_diagonals(int64_t a0, int64_t b0, int64_t nA, int64_t nB, int64_t sA, int64_t sB,
                                       int64_t m_min, int64_t size) const {
        std::vector<int64_t> m(static_cast<size_t>(size));
        for (int64_t q = 0; q < size; q++) m[q] = q + m_min;
        std::vector<int64_t> ii(static_cast<size_t>(size)), jj(static_cast<size_t>(size));
        for (const Boundary &bd : boundary_pairs(m, nA, nB, sA, sB))
            for (size_t t = 0; t < bd.idx.size(); t++) {
                ii[bd.idx[t]] = bd.row ? bd.at : bd.other[t];
                jj[bd.idx[t]] = bd.row ? bd.other[t] : bd.at;
            }
        std::vector<int64_t> out;
        for (int64_t q = 0; q < size; q++) {
            const size_t I = static_cast<size_t>(S_[a0 + ii[q]]), J = static_cast<size_t>(S_[b0 + jj[q]]);
            double dv[3], hi_[3], hj_[3];
            for (int c = 0; c < 3; c++) {
                dv[c] = 0.5 * (g_.seg_l[3 * I + c] + g_.seg_r[3 * I + c]) - 0.5 * (g_.seg_l[3 * J + c] + g_.seg_r[3 * J + c]);
                hi_[c] = g_.seg_r[3 * I + c] - g_.seg_l[3 * I + c];
                hj_[c] = g_.seg_r[3 * J + c] - g_.seg_l[3 * J + c];
            }
            const double ratio = nnorm3(dv) / std::max(nnorm3(hi_), nnorm3(hj_));
            bool near = false;
            for (double thr : thresholds_) near = near || std::fabs(ratio - thr) <= TIE_REL * thr;
            if (near) out.push_back(q);
        }
        return out;
    }

    void toeplitz(int64_t A, int64_t ra0, int64_t ra1, int64_t B, bool is_reg, int64_t e, bool with_static) {
        const Run &rA = runs_[A], &rB = runs_[B];
        const int64_t a0 = rA.s0, a1 = rA.s1, b0 = rB.s0, b1 = rB.s1;
        int64_t sA = static_cast<int64_t>(rA.sgn), sB = static_cast<int64_t>(rB.sgn);
        if (is_reg) sA = sB = 1;
        const int64_t nA = a1 - a0, nB = b1 - b0;
        const int64_t m_min = std::min<int64_t>(0, sB * (nB - 1)) - std::max<int64_t>(0, sA * (nA - 1));
        const int64_t size = nA + nB - 1;
        const int nd2 = nd_ * nd_;
        std::vector<cdouble> F(static_cast<size_t>(nd2 * size), cdouble(0.0, 0.0));  // (nd, nd, size)
        std::vector<bool> have(static_cast<size_t>(size), false);
        std::vector<int64_t> ties;
        if (!is_reg && !thresholds_.empty()) {
            ties = tie_diagonals(a0, b0, nA, nB, sA, sB, m_min, size);
            for (int64_t t : ties) have[t] = true;
        }
        int64_t tlo = INT64_MAX, thi = INT64_MIN;
        for (int64_t ip = ra0 - a0; ip < ra1 - a0; ip++)
            for (int64_t t : {-sA * ip - m_min, sB * (nB - 1) - sA * ip - m_min}) {
                tlo = std::min(tlo, t);
                thi = std::max(thi, t);
            }
        std::vector<int64_t> need;
        for (int64_t t = tlo; t <= thi; t++)
            if (!have[t]) need.push_back(t);
        if (!need.empty()) {
            std::vector<int64_t> mm;
            for (int64_t t : need) mm.push_back(t + m_min);
            for (const Boundary &bd : boundary_pairs(mm, nA, nB, sA, sB)) {
                if (bd.row) {
                    std::vector<int64_t> cols;
                    for (int64_t o : bd.other) cols.push_back(b0 + o);
                    Blk vals = eval(is_reg, e, {a0 + bd.at}, cols);
                    for (size_t t = 0; t < bd.idx.size(); t++)
                        for (int pq = 0; pq < nd2; pq++)
                            F[static_cast<size_t>(pq * size + need[bd.idx[t]])] = vals.v[static_cast<size_t>(pq) * cols.size() + t];
                } else {
                    std::vector<int64_t> rows;
                    for (int64_t o : bd.other) rows.push_back(a0 + o);
                    Blk vals = eval(is_reg, e, rows, {b0 + bd.at});
                    for (size_t t = 0; t < bd.idx.size(); t++)
                        for (int pq = 0; pq < nd2; pq++)
                            F[static_cast<size_t>(pq * size + need[bd.idx[t]])] = vals.v[static_cast<size_t>(pq) * rows.size() + t];
                }
            }
            if (with_static) {
                // the static table IS the 2N - 1 diagonals (rows N-1 and 0 of its gather)
                const nd::array_t<double> &tb = table(e);
                const double *tp = tb.data();
                for (int pq = 0; pq < nd2; pq++)
                    for (int64_t t : need) {
                        cdouble &f = F[static_cast<size_t>(pq * size + t)];
                        f = cdouble(f.real() + tp[pq * size + t], f.imag() + 0.0);
                    }
            }
        }
        for (int64_t i = ra0; i < ra1; i++) {
            const int64_t t0 = -sA * (i - a0) - m_min;
            std::vector<cdouble> &o = *out_;
            for (int pq = 0; pq < nd2; pq++)
                for (int64_t j = 0; j < nB; j++) {
                    const int64_t t = sB > 0 ? t0 + j : t0 - j;
                    const cdouble s = F[static_cast<size_t>(pq * size + t)];
                    cdouble &d = o[static_cast<size_t>((pq * nr_ + (i - r0_)) * n_ + b0 + j)];
                    d = cdouble(d.real() - s.real(), d.imag() - s.imag());
                }
        }
        if (!ties.empty()) {
            for (int64_t i = ra0; i < ra1; i++) {
                std::vector<int64_t> jp;
                for (int64_t t : ties) {
                    const int64_t j = sB * ((t + m_min) + sA * (i - a0));
                    if (j >= 0 && j < nB) jp.push_back(j);
                }
                if (jp.empty()) continue;
                std::vector<int64_t> cols;
                for (int64_t j : jp) cols.push_back(b0 + j);
                Blk vals = eval(is_reg, e, {i}, cols);
                for (size_t c = 0; c < jp.size(); c++) put(i, i + 1, b0 + jp[c], b0 + jp[c] + 1, vals, 0, static_cast<int64_t>(c));
            }
        }
    }

    const Geometry &g_;
    const std::vector<double> &seg_a_;
    std::vector<int64_t> S_;
    int64_t n_;
    FillParams fp_;
    EK ek_;
    int nd_;
    std::vector<Edge> edges_;
    std::vector<int64_t> edge_bounds_;
    double rho_ = 0;
    std::vector<Run> runs_;
    std::vector<std::array<double, 3>> step_;
    std::vector<int64_t> run_of_;
    std::vector<bool> phase_tie_;
    std::vector<double> thresholds_;
    std::vector<std::vector<int64_t>> classes_;
    std::vector<int64_t> class_of_;
    std::vector<std::vector<int64_t>> edge_runs_;
    std::map<int64_t, nd::array_t<double>> tables_;
    int64_t nodes_e_ = -1;
    size_t q_ = 0, n_edge_ = 0;
    std::vector<double> s_q_, wu_;
    std::vector<cdouble> *out_ = nullptr;
    int64_t r0_ = 0, nr_ = 0;
};

}  // namespace

bool exact_kernel_auto(const Geometry &g, const std::vector<double> &radius_per_wire) {
    for (size_t w = 0; w < g.per_wire.size(); w++)
        for (int64_t q = 0; q < g.per_wire[w].n_total; q++) {
            const size_t s = static_cast<size_t>(g.seg_offsets[w] + q);
            double d[3];
            for (int c = 0; c < 3; c++) d[c] = g.seg_r[3 * s + c] - g.seg_l[3 * s + c];
            if (nnorm3(d) < 3.0 * radius_per_wire[w]) return true;
        }
    return false;
}

bool exact_kernel_has_cross_job(const Geometry &g, const std::vector<double> &radius_per_wire) {
    std::vector<double> seg_a;
    for (size_t w = 0; w < g.per_wire.size(); w++)
        for (int64_t q = 0; q < g.per_wire[w].n_total; q++) seg_a.push_back(radius_per_wire[w]);
    const std::vector<int64_t> labels = razor::coaxial_labels(g.seg_l, g.seg_r, g.tangents, seg_a);
    double amax = seg_a.empty() ? 0.0 : seg_a[0];
    for (double v : seg_a) amax = std::max(amax, v);
    const std::vector<int64_t> line =
        razor::coaxial_labels(g.seg_l, g.seg_r, g.tangents, std::vector<double>(seg_a.size(), amax));
    std::map<int64_t, int64_t> first;
    for (size_t s = 0; s < line.size(); s++) {
        auto it = first.find(line[s]);
        if (it == first.end())
            first[line[s]] = labels[s];
        else if (it->second != labels[s])
            return true;
    }
    return false;
}

void add_exact_correction(std::vector<cdouble> &Z, const Geometry &g, const Basis &b,
                          const std::vector<double> &radius_per_wire, const FillParams &fp, double omega,
                          double eps, double mu, double swept_mem_mb) {
    const int d = fp.degree, nd = d + 1;
    const int64_t n_total = g.n_segs_total, n_basis = b.n_basis;
    std::vector<double> seg_a;
    for (size_t w = 0; w < g.per_wire.size(); w++)
        for (int64_t q = 0; q < g.per_wire[w].n_total; q++) seg_a.push_back(radius_per_wire[w]);
    const std::vector<int64_t> labels = razor::coaxial_labels(g.seg_l, g.seg_r, g.tangents, seg_a);
    if (exact_kernel_has_cross_job(g, radius_per_wire))
        throw FrontNotPorted("a coaxial line with a radius step (the cross job)");
    // the fill's EK spec
    EK ek;
    if (fp.extended_kernel) {
        ek.on = true;
        ek.group_i.assign(static_cast<size_t>(n_total), 0);
        ek.group_j.assign(static_cast<size_t>(n_total), 0);
        bool all_eq = true;
        for (double v : seg_a) all_eq = all_eq && v == seg_a[0];
        if (!all_eq) {
            ek.b_kind = EK::per_col;
            ek.b_j = seg_a;
        }
    }
    // the groups, in label order, each with its edges (`_edge_ingredients`)
    std::map<int64_t, std::vector<int64_t>> groups;
    for (int64_t s = 0; s < n_total; s++) groups[labels[s]].push_back(s);
    struct GEdge {
        int64_t start, stop;
        const std::vector<double> *arc;
        double a_w;
    };
    std::map<int64_t, std::vector<GEdge>> edges;
    for (size_t w = 0; w < g.per_wire.size(); w++) {
        const auto &pw = g.per_wire[w];
        for (size_t e = 0; e + 1 < pw.edge_offsets.size(); e++) {
            const int64_t s0 = g.seg_offsets[w] + pw.edge_offsets[e], s1 = g.seg_offsets[w] + pw.edge_offsets[e + 1];
            edges[labels[s0]].push_back({s0, s1, &pw.edge_arc_edges[e], radius_per_wire[w]});
        }
    }
    const mw::ek::PointRules rules = razor::point_rules();
    const int64_t budget = static_cast<int64_t>(swept_mem_mb) * 1024 * 1024;
    nd::array_t<std::complex<double>> Za =
        borrow_mut(Z, {static_cast<nd::ssize_t>(n_basis), static_cast<nd::ssize_t>(n_basis)});
    const CF pa = borrow(b.polys, {static_cast<nd::ssize_t>(n_basis), nd, nd});
    for (const auto &gS : groups) {
        const std::vector<int64_t> &S = gS.second;
        const int64_t n = static_cast<int64_t>(S.size());
        const size_t s0 = static_cast<size_t>(S[0]);
        const double *axis = &g.tangents[3 * s0];
        // `group_frame`
        std::vector<double> x0(static_cast<size_t>(n)), sgn(static_cast<size_t>(n)), h(static_cast<size_t>(n));
        double hmax = 0.0;
        for (int64_t i = 0; i < n; i++) {
            const size_t s = static_cast<size_t>(S[i]);
            double dv[3], ev[3];
            for (int c = 0; c < 3; c++) {
                dv[c] = g.seg_l[3 * s + c] - g.seg_l[3 * s0 + c];
                ev[c] = g.seg_r[3 * s + c] - g.seg_l[3 * s + c];
            }
            x0[i] = ndot3(dv, axis);
            sgn[i] = ndot3(&g.tangents[3 * s], axis) >= 0.0 ? 1.0 : -1.0;
            h[i] = nnorm3(ev);
            hmax = std::max(hmax, h[i]);
        }
        const int64_t max_cached = std::max<int64_t>(1, budget / (nd * nd * 16 + 256));
        mw::ek::CoaxialRows exact(x0.data(), sgn.data(), h.data(), n, seg_a[s0], fp.k, nd, max_cached,
                                  n ? 1e-11 * hmax : 1.0, true, rules, 0);
        // the bases relabelled onto the group's local order (sentinel n)
        std::vector<int64_t> loc(static_cast<size_t>(n_total), n);
        for (int64_t i = 0; i < n; i++) loc[static_cast<size_t>(S[i])] = i;
        std::vector<int64_t> supp_loc(static_cast<size_t>(n_basis * nd));
        std::vector<int64_t> Bs;
        for (int64_t m = 0; m < n_basis; m++) {
            bool any = false;
            for (int a = 0; a < nd; a++) {
                const int64_t sg = b.supp_seg[static_cast<size_t>(m * nd + a)];
                const int64_t v = sg >= 0 ? loc[static_cast<size_t>(sg)] : n;
                supp_loc[static_cast<size_t>(m * nd + a)] = v;
                any = any || v < n;
            }
            if (any) Bs.push_back(m);
        }
        std::vector<double> tan_loc(static_cast<size_t>(3 * (n + 1)), 0.0);
        for (int64_t i = 0; i < n; i++)
            for (int c = 0; c < 3; c++) tan_loc[3 * i + c] = g.tangents[3 * static_cast<size_t>(S[i]) + c];
        std::vector<BaseRows::Edge> ge;
        for (const GEdge &e : edges[gS.first]) {
            const int64_t l0 = loc[static_cast<size_t>(e.start)];
            ge.push_back({l0, l0 + (e.stop - e.start), e.arc, e.a_w});
        }
        BaseRows base(g, seg_a, S, ge, fp, ek);
        const int64_t entry = nd * nd * 16 + 1;
        const int64_t reserve = std::min<int64_t>(budget / 16, base.cache_entries_needed() * entry);
        const double window_mb = static_cast<double>(budget - reserve) / (1024.0 * 1024.0);
        const int64_t nd2 = nd * nd;
        const int64_t per_pair = 3 * nd2 * 16 + nd2 * 8 + 2 * 8 * fp.n_qp_same_edge * fp.n_qp_same_edge + 200;
        const int64_t row_bytes = n * per_pair;
        // `_schedule.mb_rows(window_mb, row_bytes)`: max(1, int(mb * 1024 * 1024 // row_bytes))
        const int64_t chunk = std::max<int64_t>(
            1, static_cast<int64_t>(py_floordiv(window_mb * 1024.0 * 1024.0, static_cast<double>(row_bytes))));
        const CI sa = borrow(supp_loc, {static_cast<nd::ssize_t>(n_basis), nd});
        const CF ta = borrow(tan_loc, {static_cast<nd::ssize_t>(n + 1), 3});
        const CI Ba = borrow(Bs, {static_cast<nd::ssize_t>(Bs.size())});
        for (int64_t r0 = 0; r0 < n; r0 += chunk) {
            const int64_t r1 = std::min(n, r0 + chunk);
            std::vector<cdouble> dJ(static_cast<size_t>(nd2 * (r1 - r0) * n));
            exact.rows(r0, r1, dJ.data());
            base.subtract_from(dJ, r0, r1);
            std::vector<int64_t> M;
            for (int64_t m : Bs) {
                bool hit = false;
                for (int a = 0; a < nd; a++) {
                    const int64_t v = supp_loc[static_cast<size_t>(m * nd + a)];
                    hit = hit || (v >= r0 && v < r1);
                }
                if (hit) M.push_back(m);
            }
            const nd::array_t<std::complex<double>, nd::array::c_style | nd::array::forcecast> Jw =
                borrow(dJ, {nd, nd, static_cast<nd::ssize_t>(r1 - r0), static_cast<nd::ssize_t>(n)});
            const CI Ma = borrow(M, {static_cast<nd::ssize_t>(M.size())});
            assemble_Z_bspline_windowed(Jw, sa, pa, ta, Ma, Ba, r0, r1, 0, n, omega, eps, mu, Za, 0, nd::none(),
                                        false);
        }
    }
}


// ---------------------------------------------------------------------------
// The image term above a ground (momwire#1517 step 3)
// ---------------------------------------------------------------------------
void accumulate_image(std::vector<cdouble> &Z, const Geometry &g, const Basis &b,
                      const std::vector<double> &radius_per_wire, const FillParams &fp, double omega,
                      double eps, double mu, const ImageGround &gr, double swept_mem_mb,
                      const nec5::NumpyLinalg &np) {
    if (np.dgemm == nullptr || np.dgemv == nullptr || np.ddot == nullptr)
        throw FrontNotPorted("no real BLAS for the image weights");
    const int d = fp.degree, nd = d + 1;
    const int64_t N = g.n_segs_total, n_basis = b.n_basis;
    std::vector<double> a_row;
    for (size_t w = 0; w < g.per_wire.size(); w++)
        for (int64_t q = 0; q < g.per_wire[w].n_total; q++) a_row.push_back(radius_per_wire[w]);
    Segs real, img;
    real.l = g.seg_l;
    real.r = g.seg_r;
    img.l = g.seg_l;
    img.r = g.seg_r;
    for (int64_t s = 0; s < N; s++) {  // `mirror_positions`: 2 * ground_z - z
        img.l[3 * s + 2] = 2.0 * gr.ground_z - g.seg_l[3 * s + 2];
        img.r[3 * s + 2] = 2.0 * gr.ground_z - g.seg_r[3 * s + 2];
    }
    EK ek;  // `_ek_spec(geom, mirror=True)`: one joint label (momwire#1368)
    if (fp.extended_kernel) {
        ek.on = true;
        ek.group_i.assign(static_cast<size_t>(N), 0);
        ek.group_j.assign(static_cast<size_t>(N), 0);
        bool all_eq = true;
        for (double v : a_row) all_eq = all_eq && v == a_row[0];
        if (!all_eq) {
            ek.b_kind = EK::per_col;
            ek.b_j = a_row;
        }
    }
    // `mirror_tangents(src_t)`: (tx, ty, -tz), and its transpose's strides
    std::vector<double> mt(g.tangents);
    for (int64_t s = 0; s < N; s++)
        for (int c = 0; c < 3; c++) mt[3 * s + c] = g.tangents[3 * s + c] * (c == 2 ? -1.0 : 1.0);
    // `weights_fn(i0, i1)`: (w_A, w_Phi), (i1 - i0, N) C order
    const auto weights = [&](int64_t i0, int64_t i1, std::vector<cdouble> &wA, std::vector<cdouble> &wP) {
        const int64_t m = i1 - i0;
        std::vector<double> td(static_cast<size_t>(m * N));
        dmatmul(np, &g.tangents[3 * static_cast<size_t>(i0)], m, 3, 3, 1, mt.data(), N, 1, 3, td.data());
        wA.resize(td.size());
        wP.resize(td.size());
        for (size_t q = 0; q < td.size(); q++) {
            if (gr.compose) {
                wA[q] = nec5::replay::np_cmul(gr.c2, cdouble(td[q], 0.0));
                wP[q] = gr.c2;
            } else {
                wA[q] = cdouble(td[q], 0.0);
                wP[q] = cdouble(1.0, 0.0);
            }
        }
    };
    nd::array_t<std::complex<double>> Za =
        borrow_mut(Z, {static_cast<nd::ssize_t>(n_basis), static_cast<nd::ssize_t>(n_basis)});
    const CI sa = borrow(b.supp_seg, {static_cast<nd::ssize_t>(n_basis), nd});
    const CF pa = borrow(b.polys, {static_cast<nd::ssize_t>(n_basis), nd, nd});
    std::vector<int64_t> all_n(static_cast<size_t>(n_basis));
    for (int64_t m = 0; m < n_basis; m++) all_n[static_cast<size_t>(m)] = m;
    const auto touching = [&](int64_t lo, int64_t hi) {
        std::vector<int64_t> out;
        for (int64_t m = 0; m < n_basis; m++)
            for (int q = 0; q < nd; q++) {
                const int64_t sg = b.supp_seg[static_cast<size_t>(m * nd + q)];
                if (sg >= lo && sg < hi) {
                    out.push_back(m);
                    break;
                }
            }
        return out;
    };
    const auto assemble = [&](const Block &J, const std::vector<cdouble> &wA, const std::vector<cdouble> &wP,
                              const std::vector<int64_t> &mi, const std::vector<int64_t> &ni, int64_t i0, int64_t i1,
                              int64_t j0, int64_t j1) {
        using CC = nd::array_t<std::complex<double>, nd::array::c_style | nd::array::forcecast>;
        const CC Jw = borrow(J, {nd, nd, static_cast<nd::ssize_t>(i1 - i0), static_cast<nd::ssize_t>(j1 - j0)});
        const CC wa = borrow(wA, {static_cast<nd::ssize_t>(i1 - i0), static_cast<nd::ssize_t>(j1 - j0)});
        const CC wp = borrow(wP, {static_cast<nd::ssize_t>(i1 - i0), static_cast<nd::ssize_t>(j1 - j0)});
        const CI ma = borrow(mi, {static_cast<nd::ssize_t>(mi.size())});
        const CI na = borrow(ni, {static_cast<nd::ssize_t>(ni.size())});
        assemble_Z_bspline_weighted_windowed(Jw, sa, pa, wa, wp, ma, na, i0, i1, j0, j1, omega, eps, mu,
                                             std::complex<double>(-1.0, 0.0), Za, 0, nd::none(), false);
    };
    const auto range = [](int64_t lo, int64_t hi) {
        std::vector<size_t> v;
        for (int64_t q = lo; q < hi; q++) v.push_back(static_cast<size_t>(q));
        return v;
    };
    // the sweep: `mb_rows(swept_mem_mb, (d+1)^2 N 16 + 2 N 16)`
    const int64_t row_bytes = static_cast<int64_t>(nd * nd) * N * 16 + 2 * N * 16;
    int64_t chunk;
    if (swept_mem_mb == std::floor(swept_mem_mb))
        chunk = std::max<int64_t>(1, (static_cast<int64_t>(swept_mem_mb) * 1024 * 1024) / row_bytes);
    else
        chunk = std::max<int64_t>(
            1, static_cast<int64_t>(std::floor((swept_mem_mb * 1024 * 1024) / static_cast<double>(row_bytes))));
    for (int64_t i0 = 0; i0 < N; i0 += chunk) {
        const int64_t i1 = std::min(i0 + chunk, N);
        const std::vector<size_t> rows = range(i0, i1);
        std::vector<double> a_sub(a_row.begin() + i0, a_row.begin() + i1);
        const Block J = offedge(real.take(rows), img, a_sub, fp.k, d, fp.n_qp_pair, ek_cut(ek, &rows, nullptr),
                                fp.ladder);
        std::vector<cdouble> wA, wP;
        weights(i0, i1, wA, wP);
        assemble(J, wA, wP, touching(i0, i1), all_n, i0, i1, 0, N);
    }
    // the near-image fixup (`_near_image_edge_blocks`), wire order
    std::vector<double> t01, w01;
    gl01(fp.n_qp_same_edge, t01, w01);
    const size_t q = t01.size();
    for (size_t w = 0; w < g.per_wire.size(); w++) {
        const WireGeometry &pw = g.per_wire[w];
        const double a_w = radius_per_wire[w];
        for (size_t e = 0; e + 1 < pw.edge_offsets.size(); e++) {
            const int64_t s0 = g.seg_offsets[w] + pw.edge_offsets[e], s1 = g.seg_offsets[w] + pw.edge_offsets[e + 1];
            const std::vector<double> &arc = pw.edge_arc_edges[e];
            const double h = g.seg_l[3 * static_cast<size_t>(s0) + 2] - gr.ground_z;
            const double atol = std::max(a_w, 1e-12);
            bool flat = true;
            for (int64_t s = s0; s < s1; s++)
                for (double z : {g.seg_l[3 * static_cast<size_t>(s) + 2], g.seg_r[3 * static_cast<size_t>(s) + 2]}) {
                    const double v = z - gr.ground_z;
                    if (!(std::fabs(v - h) <= atol + 0.0 * std::fabs(h))) flat = false;
                }
            if (!flat || h <= 0.0) continue;
            const double two_h = 2.0 * h;
            double delta = arc[1] - arc[0];
            for (size_t i = 1; i + 1 < arc.size(); i++) delta = std::min(delta, arc[i + 1] - arc[i]);
            if (delta <= 0.5 * two_h) continue;
            const double a_eff = std::sqrt(a_w * a_w + two_h * two_h);
            const size_t n = arc.size() - 1;
            const std::vector<size_t> rows = range(s0, s1);
            std::vector<double> a_sub(a_row.begin() + s0, a_row.begin() + s1);
            const Block J_edge = offedge(real.take(rows), img.take(rows), a_sub, fp.k, d, fp.n_qp_pair,
                                         ek_cut(ek, &rows, &rows), fp.ladder);
            // `_near_image_analytic_block`: static + reg at a_eff, no EK
            if (!is_toeplitz(arc)) throw FrontNotPorted("a non-uniform grazing edge");
            const double hh = arc[1] - arc[0];
            const nd::array_t<double> A_st = seg_seg_static_moments_bspline_uniform(hh, a_eff, n, d, 0, 0);
            std::vector<double> s_q(n * q), u_q(n * q), w_q(n * q);
            for (size_t i = 0; i < n; i++) {
                const double hs = arc[i + 1] - arc[i];
                for (size_t j = 0; j < q; j++) {
                    s_q[i * q + j] = arc[i] + t01[j] * hs;
                    u_q[i * q + j] = (t01[j] * hs) * 1.0;
                    w_q[i * q + j] = (w01[j] * hs) * 1.0;
                }
            }
            const CF sq = borrow(s_q, {static_cast<nd::ssize_t>(n * q)});
            const nd::array_t<double> R = reg_distance_table(sq, sq, a_eff * a_eff);
            std::vector<double> wu(static_cast<size_t>(nd) * n * q);
            for (int p = 0; p < nd; p++)
                for (size_t i = 0; i < n * q; i++) {
                    const double u = u_q[i];
                    const double up = p == 0 ? 1.0 : p == 1 ? u : p == 2 ? u * u : std::pow(u, p);
                    wu[static_cast<size_t>(p) * n * q + i] = w_q[i] * up;
                }
            const CF wua = borrow(wu, {nd, static_cast<nd::ssize_t>(n), static_cast<nd::ssize_t>(q)});
            const std::vector<double> kv{fp.k};
            const CF ka = borrow(kv, {1});
            const CF Rv = CF::borrow(R.buffer());
            const nd::array_t<std::complex<double>> A_reg = seg_seg_reg_moments_bspline_swept(Rv, wua, ka);
            Block corr(static_cast<size_t>(nd * nd) * n * n);
            const double *st = A_st.data();
            const std::complex<double> *rg = A_reg.data();
            for (size_t t = 0; t < corr.size(); t++) {
                const cdouble an(st[t] + rg[t].real(), 0.0 + rg[t].imag());
                corr[t] = cdouble(an.real() - J_edge[t].real(), an.imag() - J_edge[t].imag());
            }
            std::vector<cdouble> wA, wP;
            weights(s0, s1, wA, wP);
            std::vector<cdouble> wAs(n * n), wPs(n * n);
            for (size_t i = 0; i < n; i++)
                for (size_t j = 0; j < n; j++) {
                    wAs[i * n + j] = wA[i * static_cast<size_t>(N) + static_cast<size_t>(s0) + j];
                    wPs[i * n + j] = wP[i * static_cast<size_t>(N) + static_cast<size_t>(s0) + j];
                }
            const std::vector<int64_t> e_idx = touching(s0, s1);
            assemble(corr, wAs, wPs, e_idx, e_idx, s0, s1, s0, s1);
        }
    }
}


// ---------------------------------------------------------------------------
// The Sommerfeld remainder (momwire#1517 step 3b)
// ---------------------------------------------------------------------------
namespace {

// `_quadrature.leggauss(n)` mapped to [0, 1]: (tq, wg) with tq = 0.5 (x + 1)
void remainder_rule(int64_t q, std::vector<double> &tq, std::vector<double> &wg) {
    std::vector<double> xg;
    if (!razor::leggauss(q, xg, wg)) throw FrontNotPorted("a remainder order past the frozen rules");
    tq.resize(xg.size());
    for (size_t j = 0; j < xg.size(); j++) tq[j] = 0.5 * (xg[j] + 1.0);
}

// `seg_l[:, None, :] + tq[None, :, None] * (seg_r - seg_l)[:, None, :]`
std::vector<double> remainder_nodes(const Geometry &g, const std::vector<double> &tq) {
    const size_t N = static_cast<size_t>(g.n_segs_total), q = tq.size();
    std::vector<double> nodes(N * q * 3);
    for (size_t s = 0; s < N; s++)
        for (size_t j = 0; j < q; j++)
            for (int c = 0; c < 3; c++)
                nodes[(s * q + j) * 3 + c] = g.seg_l[3 * s + c] + tq[j] * (g.seg_r[3 * s + c] - g.seg_l[3 * s + c]);
    return nodes;
}

constexpr double REMAINDER_GRADED_C = 4.0;  // `_REMAINDER_GRADED_C`

}  // namespace

bool remainder_pairs_cleared(const Geometry &g, double ground_z, int64_t n_qp_sommerfeld) {
    const size_t N = static_cast<size_t>(g.n_segs_total);
    if (N == 0) return true;
    std::vector<double> tq, wg;
    remainder_rule(n_qp_sommerfeld, tq, wg);
    const std::vector<double> nodes = remainder_nodes(g, tq);
    double lmax = -1.0, h_l = INFINITY, h_r = INFINITY, h_o = INFINITY;
    for (size_t s = 0; s < N; s++) {
        double d[3];
        for (int c = 0; c < 3; c++) d[c] = g.seg_r[3 * s + c] - g.seg_l[3 * s + c];
        lmax = std::max(lmax, nnorm3(d));
        h_l = std::min(h_l, g.seg_l[3 * s + 2] - ground_z);
        h_r = std::min(h_r, g.seg_r[3 * s + 2] - ground_z);
    }
    for (size_t t = 0; t < nodes.size() / 3; t++) h_o = std::min(h_o, nodes[3 * t + 2] - ground_z);
    const double h_min = std::min(std::min(h_l, h_r), h_o);
    if (!(h_min > 0.0)) return false;
    const double bound = lmax / (2.0 * h_min);
    return static_cast<int64_t>(std::ceil(REMAINDER_GRADED_C * bound)) <= n_qp_sommerfeld;
}

// ---------------------------------------------------------------------------
// The Sommerfeld remainder's raised pairs (momwire#1517 step 3c):
// `_remainder_qp_pairs`, `_remainder_pair_correction` on the graded route
// (`_remainder_graded.pair_moments`), numpy's arithmetic replayed
// ---------------------------------------------------------------------------
namespace {

constexpr int64_t REMAINDER_QP_CAP = 192;          // `_REMAINDER_QP_CAP`
constexpr double REMAINDER_GRADED_EDGE_TOL = 1e-9;  // `_REMAINDER_GRADED_EDGE_TOL`
constexpr double GRADED_SIGMA = 0.25;               // `_remainder_graded.SIGMA`
constexpr int64_t GRADED_N_PANEL = 8;               // `N_PANEL`
constexpr double GRADED_SCALE_FLOOR = 1e-6;         // `SCALE_FLOOR`
constexpr int64_t PAIR_SCATTER_CHUNK = int64_t(1) << 15;

// numpy's NaN-propagating maximum / minimum, and `np.clip`.
inline double npmax(double a, double b) { return (a != a) ? a : (b != b) ? b : (a >= b ? a : b); }
inline double npmin(double a, double b) { return (a != a) ? a : (b != b) ? b : (a <= b ? a : b); }
inline double npclip(double x, double lo, double hi) { return npmin(npmax(x, lo), hi); }
// `einsum("ij,ij->i")` over three contiguous components: numpy's two-lane
// loop, the third term into the first lane (measured: (a0 b0 + a2 b2) + a1 b1),
// and over two strided ones in order
inline double edot3(const double *a, const double *b) { return (a[0] * b[0] + a[2] * b[2]) + a[1] * b[1]; }
inline double edot2(double ax, double ay, double bx, double by) { return (0.0 + ax * bx) + ay * by; }
inline double vnorm3(double x, double y, double z) { return std::sqrt((x * x + y * y) + z * z); }

struct V3 {
    double x = 0, y = 0, z = 0;
};

// `_roots01(a, b, c)`: (task, u) of the real roots in (0, 1)
void roots01(const std::vector<double> &a, const std::vector<double> &b, const std::vector<double> &c,
             std::vector<int64_t> &ko, std::vector<double> &uo) {
    const size_t n = a.size();
    std::vector<int64_t> kl, kq;
    std::vector<double> ul, r1, r2;
    for (size_t i = 0; i < n; i++) {
        const double scale = (std::fabs(a[i]) + std::fabs(b[i])) + std::fabs(c[i]);
        const bool lin = std::fabs(a[i]) <= 1e-14 * scale;
        if (lin) {
            if (std::fabs(b[i]) > 0.0) {
                kl.push_back(static_cast<int64_t>(i));
                ul.push_back(-c[i] / b[i]);
            }
            continue;
        }
        const double disc = b[i] * b[i] - (4.0 * a[i]) * c[i];
        if (!(disc >= 0.0)) continue;
        const double sq = std::sqrt(disc);
        const double bb = b[i];
        const double q = -0.5 * (bb + (bb >= 0.0 ? sq : -sq));
        kq.push_back(static_cast<int64_t>(i));
        r1.push_back(q / a[i]);
        r2.push_back(q != 0.0 ? c[i] / q : std::nan(""));
    }
    const auto keep = [&](int64_t k, double u) {
        if (std::isfinite(u) && u > 0.0 && u < 1.0) {
            ko.push_back(k);
            uo.push_back(u);
        }
    };
    for (size_t t = 0; t < kl.size(); t++) keep(kl[t], ul[t]);
    for (size_t t = 0; t < kq.size(); t++) keep(kq[t], r1[t]);
    for (size_t t = 0; t < kq.size(); t++) keep(kq[t], r2[t]);
}

struct Lattice {  // `_lattice(grid)`
    double th_split = 0;
    std::vector<double> breaks;
    struct Row {
        double r0, dr;
        int64_t n_r;
        double lo, hi;
        bool upper;
    };
    std::vector<Row> rows;
};

Lattice lattice_of(const somm_core::Grid &g) {
    Lattice L;
    L.th_split = 30.0 * (M_PI / 180.0);  // math.radians(_SOMM_TH_SPLIT_DEG)
    const double rb = g.r_break, rn = g.r_near, r1 = g.r1_max;
    const double bounds[3][2] = {{0.0, rb}, {rb, rn}, {rn, r1}};
    for (size_t idx = 0; idx < g.regions.size(); idx++) {
        const auto &reg = g.regions[idx];
        L.rows.push_back({reg.r0, reg.dr, reg.n_r, bounds[idx / 2][0], bounds[idx / 2][1], idx % 2 == 1});
    }
    for (double r : {rb, rn})
        if (0.0 < r && r < r1) L.breaks.push_back(r);
    return L;
}

// `_crossings(P, C, D, gz, lattice)`: (task, u) where the pair (P, C + u D)
// crosses an interpolant seam
void crossings(const std::vector<V3> &P, const std::vector<V3> &C, const std::vector<V3> &D, double gz,
               const Lattice &lat, std::vector<int64_t> &task, std::vector<double> &root) {
    const size_t T = P.size();
    std::vector<double> h0x(T), h0y(T), hdx(T), hdy(T), c0(T), c1(T), hh0(T), hdd(T), h0d(T);
    for (size_t i = 0; i < T; i++) {
        h0x[i] = P[i].x - C[i].x;
        h0y[i] = P[i].y - C[i].y;
        hdx[i] = -D[i].x;
        hdy[i] = -D[i].y;
        c0[i] = (P[i].z - gz) + (C[i].z - gz);
        c1[i] = D[i].z;
        hh0[i] = edot2(h0x[i], h0y[i], h0x[i], h0y[i]);
        hdd[i] = edot2(hdx[i], hdy[i], hdx[i], hdy[i]);
        h0d[i] = edot2(h0x[i], h0y[i], hdx[i], hdy[i]);
    }
    const double t2 = std::pow(std::tan(lat.th_split), 2.0);
    {
        std::vector<double> a(T), b(T), c(T);
        for (size_t i = 0; i < T; i++) {
            a[i] = c1[i] * c1[i] - t2 * hdd[i];
            b[i] = (2.0 * c0[i]) * c1[i] - (2.0 * t2) * h0d[i];
            c[i] = c0[i] * c0[i] - t2 * hh0[i];
        }
        roots01(a, b, c, task, root);
    }
    std::vector<double> A(T), Bq(T), C0(T), r_lo(T), r_hi(T);
    for (size_t i = 0; i < T; i++) {
        A[i] = hdd[i] + c1[i] * c1[i];
        Bq[i] = 2.0 * h0d[i] + (2.0 * c0[i]) * c1[i];
        C0[i] = hh0[i] + c0[i] * c0[i];
        const double us = npclip(A[i] > 0.0 ? -Bq[i] / (2.0 * A[i]) : 0.0, 0.0, 1.0);
        const auto r2 = [&](double uu) { return ((A[i] * uu) * uu + Bq[i] * uu) + C0[i]; };
        r_lo[i] = std::sqrt(npmax(r2(us), 0.0));
        r_hi[i] = std::sqrt(npmax(npmax(r2(0.0), r2(1.0)), 0.0));
    }
    for (double r : lat.breaks) {
        std::vector<double> c(T);
        for (size_t i = 0; i < T; i++) c[i] = C0[i] - r * r;
        roots01(A, Bq, c, task, root);
    }
    for (const auto &row : lat.rows) {
        std::vector<int64_t> own;
        std::vector<double> kk;
        for (size_t i = 0; i < T; i++) {
            const double a_lo = npmax(r_lo[i], row.lo), a_hi = npmin(r_hi[i], row.hi);
            const double k0 = npmax(std::ceil((a_lo - row.r0) / row.dr), 1.0);
            const double k1 = npmin(std::floor((a_hi - row.r0) / row.dr), static_cast<double>(row.n_r - 1));
            const double nn = npmax(k1 - k0 + 1.0, 0.0);
            const int64_t n = static_cast<int64_t>(nn);
            for (int64_t t = 0; t < n; t++) {
                own.push_back(static_cast<int64_t>(i));
                kk.push_back(k0 + static_cast<double>(t));
            }
        }
        if (own.empty()) continue;
        std::vector<double> a(own.size()), b(own.size()), c(own.size());
        for (size_t t = 0; t < own.size(); t++) {
            const size_t i = static_cast<size_t>(own[t]);
            const double rr = row.r0 + kk[t] * row.dr;
            a[t] = A[i];
            b[t] = Bq[i];
            c[t] = C0[i] - rr * rr;
        }
        std::vector<int64_t> k;
        std::vector<double> u;
        roots01(a, b, c, k, u);
        for (size_t t = 0; t < k.size(); t++) {
            const size_t i = static_cast<size_t>(own[static_cast<size_t>(k[t])]);
            const double rho = std::hypot(h0x[i] + u[t] * hdx[i], h0y[i] + u[t] * hdy[i]);
            const double th = std::atan2(c0[i] + u[t] * c1[i], rho);
            const bool in_band = row.upper ? (th > lat.th_split) : (th <= lat.th_split);
            if (in_band) {
                task.push_back(static_cast<int64_t>(i));
                root.push_back(u[t]);
            }
        }
    }
}

// `_panels(task, bp, scale, sigma)`: (owner, x0, x1), stable by owner
void panels(std::vector<int64_t> task, std::vector<double> bp, std::vector<double> scale,
            std::vector<int64_t> &own_o, std::vector<double> &x0_o, std::vector<double> &x1_o) {
    const size_t n = task.size();
    std::vector<size_t> order(n);
    for (size_t i = 0; i < n; i++) order[i] = i;
    std::stable_sort(order.begin(), order.end(), [&](size_t x, size_t y) {
        if (task[x] != task[y]) return task[x] < task[y];
        return bp[x] < bp[y];
    });
    std::vector<int64_t> t2;
    std::vector<double> b2, s2;
    for (size_t r = 0; r < n; r++) {
        const size_t i = order[r];
        const bool first = r == 0 || task[i] != task[order[r - 1]] || bp[i] != bp[order[r - 1]];
        if (first) {
            t2.push_back(task[i]);
            b2.push_back(bp[i]);
            s2.push_back(scale[i]);
        } else {
            s2.back() = std::min(s2.back(), scale[i]);  // `np.minimum.reduceat`
        }
    }
    std::vector<double> a, b, da, db, L;
    std::vector<int64_t> t;
    for (size_t r = 0; r + 1 < t2.size(); r++) {
        if (t2[r + 1] != t2[r]) continue;
        const double len = b2[r + 1] - b2[r];
        if (!(len > 0.0)) continue;
        a.push_back(b2[r]);
        b.push_back(b2[r + 1]);
        da.push_back(s2[r]);
        db.push_back(s2[r + 1]);
        t.push_back(t2[r]);
        L.push_back(len);
    }
    std::vector<double> anchor, direc, length, delta;
    std::vector<int64_t> owner;
    const size_t m = L.size();
    for (size_t i = 0; i < m; i++)
        if (da[i] >= L[i] && db[i] >= L[i]) {
            anchor.push_back(a[i]);
            direc.push_back(1.0);
            length.push_back(L[i]);
            delta.push_back(INFINITY);
            owner.push_back(t[i]);
        }
    for (size_t i = 0; i < m; i++)
        if (!(da[i] >= L[i] && db[i] >= L[i])) {
            anchor.push_back(a[i]);
            direc.push_back(1.0);
            length.push_back(0.5 * L[i]);
            delta.push_back(da[i]);
            owner.push_back(t[i]);
        }
    for (size_t i = 0; i < m; i++)
        if (!(da[i] >= L[i] && db[i] >= L[i])) {
            anchor.push_back(b[i]);
            direc.push_back(-1.0);
            length.push_back(0.5 * L[i]);
            delta.push_back(db[i]);
            owner.push_back(t[i]);
        }
    const double inv_log = std::log(1.0 / GRADED_SIGMA);  // math.log(1.0 / sigma)
    std::vector<int64_t> own;
    std::vector<double> x0, x1;
    for (size_t i = 0; i < anchor.size(); i++) {
        const int64_t K =
            delta[i] >= length[i] ? 0 : static_cast<int64_t>(std::ceil(std::log(length[i] / delta[i]) / inv_log));
        for (int64_t k = 0; k <= K; k++) {
            const double far = length[i] * std::pow(GRADED_SIGMA, static_cast<double>(k));
            const double near = k == K ? 0.0 : far * GRADED_SIGMA;
            const double lo = direc[i] > 0.0 ? near : -far;
            const double hi = direc[i] > 0.0 ? far : -near;
            own.push_back(owner[i]);
            x0.push_back(anchor[i] + lo);
            x1.push_back(anchor[i] + hi);
        }
    }
    std::vector<size_t> srt(own.size());
    for (size_t i = 0; i < srt.size(); i++) srt[i] = i;
    std::stable_sort(srt.begin(), srt.end(), [&](size_t x, size_t y) { return own[x] < own[y]; });
    own_o.clear();
    x0_o.clear();
    x1_o.clear();
    for (size_t i : srt) {
        own_o.push_back(own[i]);
        x0_o.push_back(x0[i]);
        x1_o.push_back(x1[i]);
    }
}

// `_seg_dist(P, A, D)`: (distance, t raw)
inline double seg_dist(const V3 &P, const V3 &A, const V3 &D, double &traw) {
    const double d[3] = {D.x, D.y, D.z};
    const double pa[3] = {P.x - A.x, P.y - A.y, P.z - A.z};
    const double dd = edot3(d, d);
    traw = dd > 0.0 ? edot3(pa, d) / dd : 0.0;
    const double t = npclip(traw, 0.0, 1.0);
    return vnorm3(P.x - (A.x + t * D.x), P.y - (A.y + t * D.y), P.z - (A.z + t * D.z));
}

inline V3 mirror(const V3 &p, double gz) { return {p.x, p.y, 2.0 * gz - p.z}; }

struct GradedCtx {
    const Geometry &g;
    double gz, k;
    const somm_proj::GridView &view;
    Lattice lat;
    std::vector<double> xg, wg;  // `leggauss(N_PANEL)`
    int d1;
    V3 at(const std::vector<double> &a, int64_t s) const {
        return {a[3 * static_cast<size_t>(s)], a[3 * static_cast<size_t>(s) + 1], a[3 * static_cast<size_t>(s) + 2]};
    }
};

// `_inner_moments`: v (n_o, d1) for one block of outer nodes
void inner_moments(const GradedCtx &c, const std::vector<V3> &obs, const std::vector<V3> &t_obs,
                   const std::vector<V3> &A_o, const std::vector<V3> &D_o, const std::vector<V3> &t_src,
                   const std::vector<double> &Lj_o, std::vector<cdouble> &v) {
    const size_t n_o = obs.size();
    std::vector<V3> mobs(n_o);
    std::vector<double> traw(n_o);
    for (size_t i = 0; i < n_o; i++) {
        mobs[i] = mirror(obs[i], c.gz);
        (void)seg_dist(mobs[i], A_o[i], D_o[i], traw[i]);
    }
    std::vector<int64_t> kx;
    std::vector<double> ux;
    crossings(obs, A_o, D_o, c.gz, c.lat, kx, ux);
    std::vector<int64_t> tid;
    std::vector<double> tv;
    for (size_t i = 0; i < n_o; i++) {
        tid.push_back(static_cast<int64_t>(i));
        tv.push_back(0.0);
    }
    for (size_t i = 0; i < n_o; i++) {
        tid.push_back(static_cast<int64_t>(i));
        tv.push_back(1.0);
    }
    for (size_t i = 0; i < n_o; i++)
        if (traw[i] > 0.0 && traw[i] < 1.0) {
            tid.push_back(static_cast<int64_t>(i));
            tv.push_back(traw[i]);
        }
    for (size_t t = 0; t < kx.size(); t++) {
        tid.push_back(kx[t]);
        tv.push_back(ux[t]);
    }
    std::vector<double> scale(tid.size());
    for (size_t t = 0; t < tid.size(); t++) {
        const size_t i = static_cast<size_t>(tid[t]);
        const V3 spt{A_o[i].x + tv[t] * D_o[i].x, A_o[i].y + tv[t] * D_o[i].y, A_o[i].z + tv[t] * D_o[i].z};
        const double r1 = vnorm3(mobs[i].x - spt.x, mobs[i].y - spt.y, mobs[i].z - spt.z);
        scale[t] = npmax(r1 / Lj_o[i], GRADED_SCALE_FLOOR);
    }
    std::vector<int64_t> p_own;
    std::vector<double> p0, p1;
    panels(tid, tv, scale, p_own, p0, p1);
    // `ptr = np.searchsorted(p_own, np.arange(n_o + 1))`
    std::vector<int64_t> ptr(n_o + 1);
    for (size_t m = 0; m <= n_o; m++)
        ptr[m] = static_cast<int64_t>(std::lower_bound(p_own.begin(), p_own.end(), static_cast<int64_t>(m)) -
                                      p_own.begin());
    std::vector<double> ob(3 * n_o), tob(3 * n_o), ao(3 * n_o), dvo(3 * n_o), tso(3 * n_o);
    for (size_t i = 0; i < n_o; i++) {
        const V3 *src[5] = {&obs[i], &t_obs[i], &A_o[i], &D_o[i], &t_src[i]};
        std::vector<double> *dst[5] = {&ob, &tob, &ao, &dvo, &tso};
        for (int q = 0; q < 5; q++) {
            (*dst[q])[3 * i] = src[q]->x;
            (*dst[q])[3 * i + 1] = src[q]->y;
            (*dst[q])[3 * i + 2] = src[q]->z;
        }
    }
    v.assign(n_o * static_cast<size_t>(c.d1), cdouble(0.0, 0.0));
    somm_core::remainder_graded_inner(c.view, c.gz, c.k, ob.data(), tob.data(), ao.data(), dvo.data(), tso.data(),
                                      Lj_o.data(), ptr.data(), static_cast<std::ptrdiff_t>(n_o), p0.data(), p1.data(),
                                      c.xg.data(), c.wg.data(), static_cast<std::ptrdiff_t>(c.xg.size()), c.d1,
                                      v.data(), 0);
}

// `_chunk_moments`: Jf (npair, d1, d1) for pairs (I[n] <= J[n])
struct ChunkTrace {
    std::vector<int64_t> o_pair;
    std::vector<double> s_o, w_o;
    std::vector<cdouble> v;
};
std::vector<cdouble> chunk_moments(const GradedCtx &c, const std::vector<int64_t> &I, const std::vector<int64_t> &J,
                                   ChunkTrace *trace = nullptr) {
    const Geometry &g = c.g;
    const size_t npair = I.size();
    const int d1 = c.d1;
    std::vector<V3> oA(npair), oD(npair), sA(npair), sD(npair), mA(npair), mD(npair);
    std::vector<double> Li(npair), Lj(npair);
    for (size_t n = 0; n < npair; n++) {
        const V3 l = c.at(g.seg_l, I[n]), r = c.at(g.seg_r, I[n]);
        oA[n] = l;
        oD[n] = {r.x - l.x, r.y - l.y, r.z - l.z};
        const V3 sl = c.at(g.seg_l, J[n]), sr = c.at(g.seg_r, J[n]);
        sA[n] = sl;
        sD[n] = {sr.x - sl.x, sr.y - sl.y, sr.z - sl.z};
        Li[n] = g.h_per_seg[static_cast<size_t>(I[n])];
        Lj[n] = g.h_per_seg[static_cast<size_t>(J[n])];
        mA[n] = mirror(oA[n], c.gz);
        mD[n] = {oD[n].x, oD[n].y, -oD[n].z};
    }
    std::vector<int64_t> pid;
    std::vector<double> sv;
    std::vector<uint8_t> kind_seg;
    std::vector<V3> anchor;
    const auto add = [&](int64_t p, double s, bool kind, V3 an) {
        pid.push_back(p);
        sv.push_back(s);
        kind_seg.push_back(kind ? 1 : 0);
        anchor.push_back(an);
    };
    for (size_t n = 0; n < npair; n++) add(static_cast<int64_t>(n), 0.0, true, V3{});
    for (size_t n = 0; n < npair; n++) add(static_cast<int64_t>(n), 1.0, true, V3{});
    std::vector<double> sdd(npair), t0(npair), alpha(npair);
    for (size_t n = 0; n < npair; n++) {
        const double s[3] = {sD[n].x, sD[n].y, sD[n].z};
        const double w[3] = {mA[n].x - sA[n].x, mA[n].y - sA[n].y, mA[n].z - sA[n].z};
        const double md[3] = {mD[n].x, mD[n].y, mD[n].z};
        sdd[n] = edot3(s, s);
        t0[n] = edot3(w, s) / sdd[n];
        alpha[n] = edot3(md, s) / sdd[n];
    }
    for (double edge : {0.0, 1.0})
        for (size_t n = 0; n < npair; n++) {
            const bool moving = std::fabs(alpha[n]) > 1e-12;
            const double s_e = moving ? (edge - t0[n]) / alpha[n] : -1.0;
            if (s_e > 0.0 && s_e < 1.0) add(static_cast<int64_t>(n), s_e, true, V3{});
        }
    for (size_t n = 0; n < npair; n++) {
        const double md[3] = {mD[n].x, mD[n].y, mD[n].z};
        const double s[3] = {sD[n].x, sD[n].y, sD[n].z};
        const double w0[3] = {mA[n].x - sA[n].x, mA[n].y - sA[n].y, mA[n].z - sA[n].z};
        const double aa = edot3(md, md), bb = edot3(md, s);
        const double den = aa * sdd[n] - bb * bb;
        const bool skew = den > (1e-12 * aa) * sdd[n];
        const double s_c = skew ? (bb * edot3(s, w0) - sdd[n] * edot3(md, w0)) / den : -1.0;
        if (s_c > 0.0 && s_c < 1.0) add(static_cast<int64_t>(n), s_c, true, V3{});
    }
    for (int e = 0; e < 2; e++) {
        std::vector<V3> E(npair);
        for (size_t n = 0; n < npair; n++)
            E[n] = e == 0 ? sA[n] : V3{sA[n].x + sD[n].x, sA[n].y + sD[n].y, sA[n].z + sD[n].z};
        std::vector<int64_t> kk;
        std::vector<double> uu;
        crossings(E, oA, oD, c.gz, c.lat, kk, uu);
        for (size_t t = 0; t < kk.size(); t++) add(kk[t], uu[t], false, E[static_cast<size_t>(kk[t])]);
    }
    std::vector<double> scale(pid.size());
    for (size_t t = 0; t < pid.size(); t++) {
        const size_t n = static_cast<size_t>(pid[t]);
        const V3 mpt{mA[n].x + sv[t] * mD[n].x, mA[n].y + sv[t] * mD[n].y, mA[n].z + sv[t] * mD[n].z};
        double tr;
        const double d_seg = seg_dist(mpt, sA[n], sD[n], tr);
        const double d_pt = vnorm3(mpt.x - anchor[t].x, mpt.y - anchor[t].y, mpt.z - anchor[t].z);
        scale[t] = npmax((kind_seg[t] ? d_seg : d_pt) / Li[n], GRADED_SCALE_FLOOR);
    }
    std::vector<int64_t> o_pan;
    std::vector<double> x0, x1;
    panels(pid, sv, scale, o_pan, x0, x1);
    // `_nodes`
    const size_t ng = c.xg.size();
    std::vector<int64_t> o_pair;
    std::vector<double> s_o, w_o;
    for (size_t p = 0; p < o_pan.size(); p++) {
        const double half = 0.5 * (x1[p] - x0[p]);
        for (size_t q = 0; q < ng; q++) {
            o_pair.push_back(o_pan[p]);
            s_o.push_back(x0[p] + half * (c.xg[q] + 1.0));
            w_o.push_back(half * c.wg[q]);
        }
    }
    const size_t n_o = o_pair.size();
    std::vector<V3> obs(n_o), t_obs(n_o), A_o(n_o), D_o(n_o), t_src(n_o);
    std::vector<double> Lj_o(n_o);
    for (size_t i = 0; i < n_o; i++) {
        const size_t n = static_cast<size_t>(o_pair[i]);
        obs[i] = {oA[n].x + s_o[i] * oD[n].x, oA[n].y + s_o[i] * oD[n].y, oA[n].z + s_o[i] * oD[n].z};
        t_obs[i] = c.at(g.tangents, I[n]);
        t_src[i] = c.at(g.tangents, J[n]);
        A_o[i] = sA[n];
        D_o[i] = sD[n];
        Lj_o[i] = Lj[n];
    }
    std::vector<cdouble> v(n_o * static_cast<size_t>(d1));
    for (size_t b0 = 0; b0 < n_o; b0 += 2048) {  // `OUTER_BLOCK`
        const size_t b1 = std::min(n_o, b0 + 2048);
        const auto sl = [&](const auto &x) { return std::vector<typename std::decay_t<decltype(x)>::value_type>(x.begin() + b0, x.begin() + b1); };
        std::vector<cdouble> vb;
        inner_moments(c, sl(obs), sl(t_obs), sl(A_o), sl(D_o), sl(t_src), sl(Lj_o), vb);
        std::copy(vb.begin(), vb.end(), v.begin() + static_cast<std::ptrdiff_t>(b0 * static_cast<size_t>(d1)));
    }
    if (trace != nullptr) *trace = ChunkTrace{o_pair, s_o, w_o, v};
    // wo = (li_n * w_o)[:, None] * u_o[:, None] ** powers, then
    // `np.add.reduceat(wo[:, :, None] * v[:, None, :], starts, axis=0)`: each
    // entry its pair's first node plus numpy's pairwise sum of the rest
    std::vector<cdouble> out(npair * static_cast<size_t>(d1 * d1), cdouble(0.0, 0.0));
    size_t i = 0;
    std::vector<cdouble> col;
    while (i < n_o) {
        size_t e = i + 1;
        while (e < n_o && o_pair[e] == o_pair[i]) e++;
        const size_t n = static_cast<size_t>(o_pair[i]);
        const double li = Li[n];
        cdouble *dst = &out[n * static_cast<size_t>(d1 * d1)];
        for (int p = 0; p < d1; p++)
            for (int P = 0; P < d1; P++) {
                col.clear();
                for (size_t r = i; r < e; r++) {
                    const double u = li * s_o[r];
                    const double wo = (li * w_o[r]) * std::pow(u, static_cast<double>(p));
                    col.push_back(nec5::replay::np_cmul(cdouble(wo, 0.0), v[r * static_cast<size_t>(d1) + P]));
                }
                cdouble o = col[0];
                if (col.size() > 1) {
                    double rr, ri;
                    nec5::replay::np_cpairwise(col.data() + 1, static_cast<int64_t>(col.size() - 1), rr, ri);
                    o = cdouble(o.real() + rr, o.imag() + ri);
                }
                dst[p * d1 + P] = o;
            }
        i = e;
    }
    return out;
}

// `_remainder_graded.pair_moments`: Jf (n, d1, d1) for pairs (I[n], J[n])
std::vector<cdouble> graded_pair_moments(const GradedCtx &c, const std::vector<int64_t> &I, const std::vector<int64_t> &J) {
    const size_t n = I.size();
    const int d1 = c.d1;
    std::vector<cdouble> out(n * static_cast<size_t>(d1 * d1), cdouble(0.0, 0.0));
    if (n == 0) return out;
    int64_t mx = 0;
    for (size_t t = 0; t < n; t++) mx = std::max(mx, std::max(I[t], J[t]));
    std::vector<int64_t> key(n);
    for (size_t t = 0; t < n; t++) key[t] = std::min(I[t], J[t]) * (mx + 1) + std::max(I[t], J[t]);
    std::vector<int64_t> ukey(key);
    std::sort(ukey.begin(), ukey.end());
    ukey.erase(std::unique(ukey.begin(), ukey.end()), ukey.end());
    std::vector<int64_t> Iu(ukey.size()), Ju(ukey.size());
    for (size_t t = 0; t < ukey.size(); t++) {
        Iu[t] = ukey[t] / (mx + 1);
        Ju[t] = ukey[t] % (mx + 1);
    }
    std::vector<cdouble> Jf(ukey.size() * static_cast<size_t>(d1 * d1));
    for (size_t c0 = 0; c0 < Iu.size(); c0 += 256) {  // `PAIR_CHUNK`
        const size_t c1 = std::min(Iu.size(), c0 + 256);
        const std::vector<int64_t> ii(Iu.begin() + c0, Iu.begin() + c1), jj(Ju.begin() + c0, Ju.begin() + c1);
        const std::vector<cdouble> part = chunk_moments(c, ii, jj);
        std::copy(part.begin(), part.end(), Jf.begin() + static_cast<std::ptrdiff_t>(c0 * static_cast<size_t>(d1 * d1)));
    }
    for (size_t t = 0; t < n; t++) {
        const size_t u = static_cast<size_t>(std::lower_bound(ukey.begin(), ukey.end(), key[t]) - ukey.begin());
        const cdouble *src = &Jf[u * static_cast<size_t>(d1 * d1)];
        cdouble *dst = &out[t * static_cast<size_t>(d1 * d1)];
        const bool fwd = I[t] <= J[t];
        for (int p = 0; p < d1; p++)
            for (int P = 0; P < d1; P++) dst[p * d1 + P] = fwd ? src[p * d1 + P] : src[P * d1 + p];
    }
    return out;
}

// `_remainder_pair_moments(geom, obs_segs, [src], q, grid)[:, 0]`: (no, d1, d1)
std::vector<cdouble> pair_moments_fixed(const Geometry &g, const somm_proj::GridView &view, double gz, double k,
                                        int d1, const std::vector<int64_t> &obs_segs, int64_t src, int64_t q) {
    std::vector<double> xg, wg, tq;
    if (!razor::leggauss(q, xg, wg)) throw FrontNotPorted("a remainder order past the frozen rules");
    for (double x : xg) tq.push_back(0.5 * (x + 1.0));
    const auto side = [&](const std::vector<int64_t> &segs, std::vector<double> &nodes, std::vector<double> &tang,
                          std::vector<double> &W) {
        const size_t ns = segs.size();
        nodes.assign(ns * static_cast<size_t>(q) * 3, 0.0);
        tang.assign(ns * 3, 0.0);
        W.assign(static_cast<size_t>(d1) * ns * static_cast<size_t>(q), 0.0);
        for (size_t i = 0; i < ns; i++) {
            const size_t s = static_cast<size_t>(segs[i]);
            const double h = g.h_per_seg[s];
            for (int cc = 0; cc < 3; cc++) tang[3 * i + cc] = g.tangents[3 * s + cc];
            for (int64_t j = 0; j < q; j++) {
                for (int cc = 0; cc < 3; cc++)
                    nodes[(i * static_cast<size_t>(q) + static_cast<size_t>(j)) * 3 + cc] =
                        g.seg_l[3 * s + cc] + tq[static_cast<size_t>(j)] * (g.seg_r[3 * s + cc] - g.seg_l[3 * s + cc]);
                const double u = h * tq[static_cast<size_t>(j)];
                const double w = (0.5 * h) * wg[static_cast<size_t>(j)];
                for (int p = 0; p < d1; p++)
                    W[(static_cast<size_t>(p) * ns + i) * static_cast<size_t>(q) + static_cast<size_t>(j)] =
                        w * std::pow(u, static_cast<double>(p));
            }
        }
    };
    std::vector<double> no_, to_, wo_, ns_, ts_, ws_;
    side(obs_segs, no_, to_, wo_);
    side({src}, ns_, ts_, ws_);
    const auto pseudo = [&](size_t n, std::vector<int64_t> &loc, std::vector<double> &poly) {
        loc.assign(n * static_cast<size_t>(d1) * static_cast<size_t>(d1), 0);
        poly.assign(n * static_cast<size_t>(d1 * d1 * d1), 0.0);
        for (size_t s = 0; s < n; s++)
            for (int p = 0; p < d1; p++) {
                const size_t m = s * static_cast<size_t>(d1) + static_cast<size_t>(p);
                for (int a = 0; a < d1; a++) loc[m * static_cast<size_t>(d1) + static_cast<size_t>(a)] = static_cast<int64_t>(s);
                poly[(m * static_cast<size_t>(d1) + 0) * static_cast<size_t>(d1) + static_cast<size_t>(p)] = 1.0;
            }
    };
    const size_t no = obs_segs.size();
    std::vector<int64_t> lo_, ls_;
    std::vector<double> po_, ps_;
    pseudo(no, lo_, po_);
    pseudo(1, ls_, ps_);
    // the binding's `same_array` test: the symmetric route when every pair of
    // halves is the same bytes
    const auto same = [](const auto &x, const auto &y) {
        return x.size() == y.size() && std::memcmp(x.data(), y.data(), x.size() * sizeof(x[0])) == 0;
    };
    const bool symmetric = same(no_, ns_) && same(to_, ts_) && same(wo_, ws_) && same(lo_, ls_) && same(po_, ps_);
    const std::ptrdiff_t nI = static_cast<std::ptrdiff_t>(no) * d1, nJ = d1;
    std::vector<cdouble> Qb(static_cast<size_t>(nI * nJ));
    somm_core::remainder_bspline_Q(view, gz, k, no_.data(), to_.data(), wo_.data(), static_cast<std::ptrdiff_t>(no),
                                   ns_.data(), ts_.data(), ws_.data(), 1, q, d1, lo_.data(), po_.data(), nI, ls_.data(),
                                   ps_.data(), nJ, symmetric, 0, Qb.data(), 0);
    // out.reshape(no, d1, 1, d1).transpose(0, 2, 1, 3)[:, 0]
    std::vector<cdouble> out(no * static_cast<size_t>(d1 * d1));
    for (size_t o = 0; o < no; o++)
        for (int p = 0; p < d1; p++)
            for (int P = 0; P < d1; P++)
                out[(o * static_cast<size_t>(d1) + static_cast<size_t>(p)) * static_cast<size_t>(d1) + static_cast<size_t>(P)] =
                    Qb[(o * static_cast<size_t>(d1) + static_cast<size_t>(p)) * static_cast<size_t>(nJ) + static_cast<size_t>(P)];
    return out;
}

}  // namespace

// `_quadrature.remainder_qp_pairs` (+ `_remainder_qp_pairs`' symmetrise,
// dilate and per-pair max): the raised pairs (I, J, Q), sorted by I then J.
void remainder_raised_pairs(const Geometry &g, double gz, int64_t base, const nec5::NumpyLinalg &np,
                            std::vector<int64_t> &I, std::vector<int64_t> &J, std::vector<int64_t> &Q) {
    I.clear();
    J.clear();
    Q.clear();
    const int64_t n = g.n_segs_total;
    if (n == 0) return;
    if (remainder_pairs_cleared(g, gz, base)) return;
    std::vector<double> tq, wg;
    remainder_rule(base, tq, wg);
    const std::vector<double> obs = remainder_nodes(g, tq);
    const int64_t nn = static_cast<int64_t>(tq.size());
    const std::ptrdiff_t M = static_cast<std::ptrdiff_t>(n * nn);
    std::vector<double> mir_l(3 * static_cast<size_t>(n)), d(3 * static_cast<size_t>(n)), dd(static_cast<size_t>(n)),
        lengths(static_cast<size_t>(n));
    for (int64_t s = 0; s < n; s++) {
        double mr[3];
        for (int c = 0; c < 3; c++) {
            mir_l[3 * s + c] = g.seg_l[3 * s + c];
            mr[c] = g.seg_r[3 * s + c];
        }
        mir_l[3 * s + 2] = 2.0 * gz - g.seg_l[3 * s + 2];
        mr[2] = 2.0 * gz - g.seg_r[3 * s + 2];
        for (int c = 0; c < 3; c++) d[3 * s + c] = mr[c] - mir_l[3 * s + c];
        dd[s] = edot3(&d[3 * s], &d[3 * s]);
        lengths[s] = vnorm3(g.seg_r[3 * s] - g.seg_l[3 * s], g.seg_r[3 * s + 1] - g.seg_l[3 * s + 1],
                            g.seg_r[3 * s + 2] - g.seg_l[3 * s + 2]);
    }
    const double c = REMAINDER_GRADED_C, edge_tol = REMAINDER_GRADED_EDGE_TOL;
    std::vector<std::vector<somm_core::QpHit>> hits;
    std::vector<unsigned char> unsure;
    somm_core::remainder_qp_pairs_scan(obs.data(), M, nn, mir_l.data(), d.data(), dd.data(), lengths.data(), n, base,
                                       REMAINDER_QP_CAP, c, edge_tol, hits, unsure);
    std::vector<int64_t> I0, J0, Q0;
    for (int64_t j = 0; j < n; j++)
        for (const auto &h : hits[static_cast<size_t>(j)]) {
            I0.push_back(h.row);
            J0.push_back(j);
            Q0.push_back(h.q);
        }
    // the numpy body for every source the scan could not decide
    for (int64_t j = 0; j < n; j++) {
        if (!unsure[static_cast<size_t>(j)]) continue;
        if (lengths[j] <= 0.0 || !(dd[j] > 0.0)) continue;
        std::vector<double> ap(3 * static_cast<size_t>(M)), tr(static_cast<size_t>(M));
        for (std::ptrdiff_t o = 0; o < M; o++)
            for (int cc = 0; cc < 3; cc++) ap[3 * o + cc] = obs[3 * o + cc] - mir_l[3 * j + cc];
        // `ap @ d[j]`: numpy's matrix-vector product
        dmatmul(np, ap.data(), M, 3, 3, 1, &d[3 * j], 1, 1, 0, tr.data());
        std::vector<double> rmin(static_cast<size_t>(n), INFINITY);
        bool any = false;
        for (std::ptrdiff_t o = 0; o < M; o++) {
            const double t_raw = tr[o] / dd[j];
            const bool interior = t_raw > -edge_tol && t_raw < 1.0 + edge_tol;
            if (!interior) continue;
            any = true;
            const double t = npclip(t_raw, 0.0, 1.0);
            const double dist = vnorm3(obs[3 * o] - (mir_l[3 * j] + t * d[3 * j]),
                                       obs[3 * o + 1] - (mir_l[3 * j + 1] + t * d[3 * j + 1]),
                                       obs[3 * o + 2] - (mir_l[3 * j + 2] + t * d[3 * j + 2]));
            const size_t sg = static_cast<size_t>(o / nn);
            rmin[sg] = npmin(rmin[sg], dist);
        }
        if (!any) continue;
        for (int64_t s = 0; s < n; s++) {
            const double r = rmin[static_cast<size_t>(s)];
            const bool hit = std::isfinite(r);
            const double ratio = (hit && r > 0.0) ? lengths[j] / r : 0.0;
            const double need = ratio > 0.0 ? std::ceil(c * ratio) : 0.0;
            int64_t qv = static_cast<int64_t>(npmin(npmax(need, static_cast<double>(base)), static_cast<double>(REMAINDER_QP_CAP)));
            if (hit && r <= 0.0) qv = REMAINDER_QP_CAP;
            if (qv > base) {
                I0.push_back(s);
                J0.push_back(j);
                Q0.push_back(qv);
            }
        }
    }
    if (I0.empty()) return;
    // `np.lexsort((J, I))`, then symmetrised
    std::vector<int64_t> Is(I0), Js(J0), Qs(Q0);
    Is.insert(Is.end(), J0.begin(), J0.end());
    Js.insert(Js.end(), I0.begin(), I0.end());
    Qs.insert(Qs.end(), Q0.begin(), Q0.end());
    // `_pair_max`: the distinct pairs, sorted by I then J, each its largest Q
    const auto pair_max = [n](const std::vector<int64_t> &a, const std::vector<int64_t> &b, const std::vector<int64_t> &qv,
                              std::vector<int64_t> &oa, std::vector<int64_t> &ob, std::vector<int64_t> &oq) {
        std::map<int64_t, int64_t> m;
        for (size_t t = 0; t < a.size(); t++) {
            const int64_t kkey = a[t] * n + b[t];
            auto it = m.find(kkey);
            if (it == m.end())
                m[kkey] = std::max<int64_t>(0, qv[t]);
            else
                it->second = std::max(it->second, qv[t]);
        }
        oa.clear();
        ob.clear();
        oq.clear();
        for (const auto &kv : m) {
            oa.push_back(kv.first / n);
            ob.push_back(kv.first % n);
            oq.push_back(kv.second);
        }
    };
    // `_segment_touch_lists`: segments sharing an endpoint (within 1e-9 of the longest), each with itself
    double lmax = 0.0;
    for (int64_t s = 0; s < n; s++) lmax = std::max(lmax, lengths[s]);
    const double tol = 1e-9 * std::max(lmax, 1e-300);
    std::vector<std::set<int64_t>> touch(static_cast<size_t>(n));
    for (int64_t s = 0; s < n; s++) touch[s].insert(s);
    {
        std::vector<std::pair<double, int64_t>> ends;  // by x, sweep
        for (int64_t s = 0; s < n; s++) {
            ends.push_back({g.seg_l[3 * s], 2 * s});
            ends.push_back({g.seg_r[3 * s], 2 * s + 1});
        }
        std::sort(ends.begin(), ends.end());
        const auto pt = [&](int64_t e) {
            const std::vector<double> &src = (e % 2 == 0) ? g.seg_l : g.seg_r;
            const int64_t s = e / 2;
            return V3{src[3 * s], src[3 * s + 1], src[3 * s + 2]};
        };
        for (size_t a = 0; a < ends.size(); a++)
            for (size_t b = a + 1; b < ends.size() && ends[b].first - ends[a].first <= tol; b++) {
                const V3 p = pt(ends[a].second), q2 = pt(ends[b].second);
                const double dx = p.x - q2.x, dy = p.y - q2.y, dz = p.z - q2.z;
                if (std::sqrt((dx * dx + dy * dy) + dz * dz) <= tol) {
                    const int64_t sa = ends[a].second / 2, sb = ends[b].second / 2;
                    if (sa != sb) {
                        touch[sa].insert(sb);
                        touch[sb].insert(sa);
                    }
                }
            }
    }
    // dilate: rows, then columns (`_csr_expand` + `_pair_max` each)
    std::vector<int64_t> a1, b1, q1;
    {
        std::vector<int64_t> ea, eb, eq;
        for (size_t t = 0; t < Is.size(); t++)
            for (int64_t m : touch[Is[t]]) {
                ea.push_back(m);
                eb.push_back(Js[t]);
                eq.push_back(Qs[t]);
            }
        pair_max(ea, eb, eq, a1, b1, q1);
    }
    {
        std::vector<int64_t> ea, eb, eq;
        for (size_t t = 0; t < a1.size(); t++)
            for (int64_t m : touch[b1[t]]) {
                ea.push_back(a1[t]);
                eb.push_back(m);
                eq.push_back(q1[t]);
            }
        pair_max(ea, eb, eq, I, J, Q);
    }
}

// `_remainder_pair_correction(Q, geom, supp_seg, polys, grid, pairs)` on the
// graded route: Q += P (Jf_graded - Jf_base) P over each pair's wings
void remainder_pair_correction(std::vector<cdouble> &Qm, const Geometry &g, const Basis &b, double gz, double k,
                               const somm_proj::GridView &view, const somm_core::Grid &grid,
                               const std::vector<int64_t> &I, const std::vector<int64_t> &J, int64_t base,
                               const nec5::NumpyLinalg &np) {
    const int d1 = b.degree + 1;
    const int64_t n_seg = g.n_segs_total, n_basis = b.n_basis;
    GradedCtx ctx{g, gz, k, view, lattice_of(grid), {}, {}, d1};
    if (!razor::leggauss(GRADED_N_PANEL, ctx.xg, ctx.wg)) throw FrontNotPorted("no graded rule");
    std::vector<cdouble> dJ = graded_pair_moments(ctx, I, J);
    // runs by J (stable), each against the base order
    std::vector<size_t> grp(I.size());
    for (size_t t = 0; t < grp.size(); t++) grp[t] = t;
    std::stable_sort(grp.begin(), grp.end(), [&](size_t x, size_t y) { return J[x] < J[y]; });
    const size_t dd2 = static_cast<size_t>(d1 * d1);
    for (size_t r0 = 0; r0 < grp.size();) {
        size_t r1 = r0 + 1;
        while (r1 < grp.size() && J[grp[r1]] == J[grp[r0]]) r1++;
        std::vector<int64_t> obs;
        for (size_t t = r0; t < r1; t++) obs.push_back(I[grp[t]]);
        const std::vector<cdouble> lo = pair_moments_fixed(g, view, gz, k, d1, obs, J[grp[r0]], base);
        for (size_t t = r0; t < r1; t++)
            for (size_t e = 0; e < dd2; e++) {
                cdouble &x = dJ[grp[t] * dd2 + e];
                x = nec5::replay::c_sub(x, lo[(t - r0) * dd2 + e]);
            }
        r0 = r1;
    }
    // `_segment_wing_table`: per segment its real wings (flat basis*d1+wing)
    std::vector<std::vector<int64_t>> wings(static_cast<size_t>(n_seg));
    for (int64_t m = 0; m < n_basis; m++)
        for (int a = 0; a < d1; a++) {
            bool real = false;
            for (int p = 0; p < d1; p++) real = real || b.polys[static_cast<size_t>((m * d1 + a) * d1 + p)] != 0.0;
            if (real) wings[static_cast<size_t>(b.supp_seg[static_cast<size_t>(m * d1 + a)])].push_back(m * d1 + a);
        }
    size_t width = 0;
    for (const auto &w : wings) width = std::max(width, w.size());
    // pw[s, x, p]: polys[m_of, a_of, p] * valid (a padded slot: polys[0, 0, p] * 0)
    const auto pw = [&](int64_t s, size_t x, int p) {
        const auto &w = wings[static_cast<size_t>(s)];
        if (x < w.size()) return b.polys[static_cast<size_t>(w[x] * d1 + p)] * 1.0;
        return b.polys[static_cast<size_t>(p)] * 0.0;
    };
    const auto m_of = [&](int64_t s, size_t x) -> int64_t {
        const auto &w = wings[static_cast<size_t>(s)];
        return x < w.size() ? w[x] / d1 : 0;
    };
    const size_t W = width;
    for (size_t c0 = 0; c0 < I.size(); c0 += static_cast<size_t>(PAIR_SCATTER_CHUNK)) {
        const size_t c1 = std::min(I.size(), c0 + static_cast<size_t>(PAIR_SCATTER_CHUNK));
        // contrib = einsum("kup,kpP,kvP->kuv", pw[I], dJ, pw[J]), as numpy's
        // bmm path runs it: T = dJ^T-view @ pwI^T (numpy's matmul per batch),
        // then T-view @ pwJ^T
        for (size_t kk = c0; kk < c1; kk++) {
            const int64_t si = I[kk], sj = J[kk];
            std::vector<cdouble> bI(static_cast<size_t>(d1) * W), bJ(static_cast<size_t>(d1) * W);
            for (int p = 0; p < d1; p++)
                for (size_t x = 0; x < W; x++) {
                    bI[static_cast<size_t>(p) * W + x] = cdouble(pw(si, x, p), 0.0);
                    bJ[static_cast<size_t>(p) * W + x] = cdouble(pw(sj, x, p), 0.0);
                }
            const cdouble *dk = &dJ[kk * dd2];
            std::vector<cdouble> T(static_cast<size_t>(d1) * W);  // (P, u) C order
            nec5::replay::np_matmul(np, nec5::Op{dk, d1, d1, 1, d1}, nec5::Op{bI.data(), d1, static_cast<int64_t>(W), static_cast<int64_t>(W), 1},
                                    T.data(), static_cast<int64_t>(W), 1);
            std::vector<cdouble> out(W * W);
            nec5::replay::np_matmul(np, nec5::Op{T.data(), static_cast<int64_t>(W), d1, 1, static_cast<int64_t>(W)},
                                    nec5::Op{bJ.data(), d1, static_cast<int64_t>(W), static_cast<int64_t>(W), 1}, out.data(),
                                    static_cast<int64_t>(W), 1);
            const auto &wi = wings[static_cast<size_t>(si)];
            const auto &wj = wings[static_cast<size_t>(sj)];
            for (size_t u = 0; u < W; u++)
                for (size_t v = 0; v < W; v++) {
                    if (u >= wi.size() || v >= wj.size()) continue;  // keep: both wings real
                    cdouble &qq = Qm[static_cast<size_t>(m_of(si, u) * n_basis + m_of(sj, v))];
                    qq = nec5::replay::c_add(qq, out[u * W + v]);
                }
        }
    }
}

void subtract_sommerfeld_remainder(std::vector<cdouble> &Z, const Geometry &g, const Basis &b,
                                   double ground_z, cdouble eps_t, double k, double omega, double mu,
                                   int64_t n_qp_sommerfeld, const nec5::NumpyLinalg &np) {
    const int nd = b.degree + 1;
    const int64_t N = g.n_segs_total, n_basis = b.n_basis;
    std::vector<double> tq, wg;
    remainder_rule(n_qp_sommerfeld, tq, wg);
    const int64_t q = static_cast<int64_t>(tq.size());
    const std::vector<double> nodes = remainder_nodes(g, tq);
    // W[p, s, j] = (0.5 * h[s] * wg[j]) * np.float_power(h[s] * tq[j], p)
    std::vector<double> W(static_cast<size_t>(nd * N * q));
    for (int p = 0; p < nd; p++)
        for (int64_t s = 0; s < N; s++)
            for (int64_t j = 0; j < q; j++) {
                const double h = g.h_per_seg[static_cast<size_t>(s)];
                const double u = h * tq[static_cast<size_t>(j)];
                const double w = (0.5 * h) * wg[static_cast<size_t>(j)];
                W[static_cast<size_t>((p * N + s) * q + j)] = w * std::pow(u, static_cast<double>(p));
            }
    const double r1_max = razor::max_image_distance(g.seg_l.data(), g.seg_r.data(), static_cast<size_t>(N), ground_z);
    const somm_core::Grid &G0 = somm_core::get_grid(eps_t, k, r1_max, omega, mu);
    const size_t n_reg = G0.regions.size();
    if (n_reg > somm_proj::MAX_REGIONS) throw FrontNotPorted("a grid layout past the view's regions");
    double r0[somm_proj::MAX_REGIONS], dr[somm_proj::MAX_REGIONS], th0[somm_proj::MAX_REGIONS],
        dth[somm_proj::MAX_REGIONS];
    const cdouble *vals[somm_proj::MAX_REGIONS];
    std::ptrdiff_t shape0[somm_proj::MAX_REGIONS], nr[somm_proj::MAX_REGIONS], nth[somm_proj::MAX_REGIONS];
    for (size_t r = 0; r < n_reg; r++) {
        const somm_core::GridRegion &reg = G0.regions[r];
        r0[r] = reg.r0;
        dr[r] = reg.dr;
        th0[r] = reg.th0;
        dth[r] = reg.dth;
        vals[r] = reg.vals.data();
        shape0[r] = 4;
        nr[r] = static_cast<std::ptrdiff_t>(reg.n_r);
        nth[r] = static_cast<std::ptrdiff_t>(reg.n_th);
    }
    // `grid_cpp_args(grid)`: theta split `math.radians(30.0)`, and the far pack
    somm_proj::GridView view = somm_proj::build_grid_view(G0.r1_max, G0.r_break, 30.0 * (M_PI / 180.0), G0.r_near,
                                                          r0, dr, th0, dth, n_reg, vals, shape0, nr, nth);
    const std::vector<cdouble> far = somm_core::far_pack(G0);
    somm_proj::set_far(view, far.data(), static_cast<std::ptrdiff_t>(far.size()));
    // `_remainder_qp_pairs`: the pairs raised above the base order
    std::vector<int64_t> pI, pJ, pQ;
    remainder_raised_pairs(g, ground_z, n_qp_sommerfeld, np, pI, pJ, pQ);
    std::vector<cdouble> Q(static_cast<size_t>(n_basis * n_basis));
    somm_core::remainder_bspline_Q(view, ground_z, k, nodes.data(), g.tangents.data(), W.data(), N, nodes.data(),
                                   g.tangents.data(), W.data(), N, q, nd, b.supp_seg.data(), b.polys.data(),
                                   n_basis, b.supp_seg.data(), b.polys.data(), n_basis, true, 0, Q.data(), 0);
    if (!pI.empty()) remainder_pair_correction(Q, g, b, ground_z, k, view, G0, pI, pJ, n_qp_sommerfeld, np);
    for (size_t t = 0; t < Z.size(); t++) Z[t] = cdouble(Z[t].real() - Q[t].real(), Z[t].imag() - Q[t].imag());
}

void debug_chunk_pair(const Geometry &g, double gz, double k, const somm_core::Grid &grid, int d1, int64_t i,
                      int64_t j, std::vector<int64_t> &o_pair, std::vector<double> &s_o, std::vector<double> &w_o,
                      std::vector<cdouble> &v) {
    const size_t n_reg = grid.regions.size();
    double r0[somm_proj::MAX_REGIONS], dr[somm_proj::MAX_REGIONS], th0[somm_proj::MAX_REGIONS],
        dth[somm_proj::MAX_REGIONS];
    const cdouble *vals[somm_proj::MAX_REGIONS];
    std::ptrdiff_t shape0[somm_proj::MAX_REGIONS], nr[somm_proj::MAX_REGIONS], nth[somm_proj::MAX_REGIONS];
    for (size_t r = 0; r < n_reg; r++) {
        const somm_core::GridRegion &reg = grid.regions[r];
        r0[r] = reg.r0;
        dr[r] = reg.dr;
        th0[r] = reg.th0;
        dth[r] = reg.dth;
        vals[r] = reg.vals.data();
        shape0[r] = 4;
        nr[r] = static_cast<std::ptrdiff_t>(reg.n_r);
        nth[r] = static_cast<std::ptrdiff_t>(reg.n_th);
    }
    somm_proj::GridView view = somm_proj::build_grid_view(grid.r1_max, grid.r_break, 30.0 * (M_PI / 180.0),
                                                          grid.r_near, r0, dr, th0, dth, n_reg, vals, shape0, nr, nth);
    const std::vector<cdouble> far = somm_core::far_pack(grid);
    somm_proj::set_far(view, far.data(), static_cast<std::ptrdiff_t>(far.size()));
    GradedCtx ctx{g, gz, k, view, lattice_of(grid), {}, {}, d1};
    razor::leggauss(GRADED_N_PANEL, ctx.xg, ctx.wg);
    ChunkTrace tr;
    (void)chunk_moments(ctx, {std::min(i, j)}, {std::max(i, j)}, &tr);
    o_pair = tr.o_pair;
    s_o = tr.s_o;
    w_o = tr.w_o;
    v = tr.v;
}

void debug_pair_moments(const Geometry &g, double gz, double k, const somm_core::Grid &grid, int d1,
                        const std::vector<int64_t> &I, const std::vector<int64_t> &J, int64_t base,
                        std::vector<cdouble> &graded, std::vector<cdouble> &fixed) {
    const size_t n_reg = grid.regions.size();
    double r0[somm_proj::MAX_REGIONS], dr[somm_proj::MAX_REGIONS], th0[somm_proj::MAX_REGIONS],
        dth[somm_proj::MAX_REGIONS];
    const cdouble *vals[somm_proj::MAX_REGIONS];
    std::ptrdiff_t shape0[somm_proj::MAX_REGIONS], nr[somm_proj::MAX_REGIONS], nth[somm_proj::MAX_REGIONS];
    for (size_t r = 0; r < n_reg; r++) {
        const somm_core::GridRegion &reg = grid.regions[r];
        r0[r] = reg.r0;
        dr[r] = reg.dr;
        th0[r] = reg.th0;
        dth[r] = reg.dth;
        vals[r] = reg.vals.data();
        shape0[r] = 4;
        nr[r] = static_cast<std::ptrdiff_t>(reg.n_r);
        nth[r] = static_cast<std::ptrdiff_t>(reg.n_th);
    }
    somm_proj::GridView view = somm_proj::build_grid_view(grid.r1_max, grid.r_break, 30.0 * (M_PI / 180.0),
                                                          grid.r_near, r0, dr, th0, dth, n_reg, vals, shape0, nr, nth);
    const std::vector<cdouble> far = somm_core::far_pack(grid);
    somm_proj::set_far(view, far.data(), static_cast<std::ptrdiff_t>(far.size()));
    GradedCtx ctx{g, gz, k, view, lattice_of(grid), {}, {}, d1};
    razor::leggauss(GRADED_N_PANEL, ctx.xg, ctx.wg);
    graded = graded_pair_moments(ctx, I, J);
    fixed.clear();
    for (size_t t = 0; t < I.size(); t++) {
        const std::vector<cdouble> lo = pair_moments_fixed(g, view, gz, k, d1, {I[t]}, J[t], base);
        fixed.insert(fixed.end(), lo.begin(), lo.end());
    }
}

// ---------------------------------------------------------------------------
// The buried fill (momwire#1517 step 4): `_below_interface.
// compute_Z_operator_buried` on a deck whose every wire is below the plane
// ---------------------------------------------------------------------------
namespace {

constexpr double PI_ = 3.141592653589793;

// numpy's complex division (`CDOUBLE_divide`, Smith's)
inline cdouble np_cdiv(cdouble a, cdouble b) {
    const double in1r = a.real(), in1i = a.imag(), in2r = b.real(), in2i = b.imag();
    const double in2r_abs = std::fabs(in2r), in2i_abs = std::fabs(in2i);
    if (in2r_abs >= in2i_abs) {
        if (in2r_abs == 0 && in2i_abs == 0) return cdouble(in1r / in2r_abs, in1i / in2i_abs);
        const double rat = in2i / in2r;
        const double scl = 1.0 / (in2r + in2i * rat);
        return cdouble((in1r + in1i * rat) * scl, (in1i - in1r * rat) * scl);
    }
    const double rat = in2r / in2i;
    const double scl = 1.0 / (in2i + in2r * rat);
    return cdouble((in1r * rat + in1i) * scl, (in1i * rat - in1r) * scl);
}

// `_seg_seg_reg_moments(arc, a, k, max_d, n_qp)` at a complex k (the lower
// medium's): numpy's branch. R from `reg_distance_table`, the kernel
// `expm1_neg_jkR(k, R) / (4 pi R)` element by element (libm's expm1, cos,
// sin, exp; `re - 1j * im`; numpy's complex division by the real
// denominator), then `einsum("piq,iqjr,Pjr->pPij", wu, G, wu)` as numpy's
// iterator sums it: each output's (q, r) terms ((w G.re) v, (w G.im) v), r
// innermost from zero and each q's partial added to the output in turn --
// one run over (q, r) when the source axis has extent 1, which the iterator
// drops. (nd, nd, n, n) C order.
Block reg_moments_cplx(const std::vector<double> &arc, double a_w, cdouble k, int d, int64_t n_qp) {
    const int nd = d + 1;
    std::vector<double> t01, w01;
    gl01(n_qp, t01, w01);
    const size_t q = t01.size(), n = arc.size() - 1;
    std::vector<double> s_q(n * q), u_q(n * q), w_q(n * q);
    for (size_t i = 0; i < n; i++) {
        const double hs = arc[i + 1] - arc[i];
        for (size_t j = 0; j < q; j++) {
            s_q[i * q + j] = arc[i] + t01[j] * hs;
            u_q[i * q + j] = (t01[j] * hs) * 1.0;
            w_q[i * q + j] = (w01[j] * hs) * 1.0;
        }
    }
    const CF sq = borrow(s_q, {static_cast<nd::ssize_t>(n * q)});
    const nd::array_t<double> R = reg_distance_table(sq, sq, a_w * a_w);
    const double *Rp = R.data();
    std::vector<double> wu(static_cast<size_t>(nd) * n * q);
    for (int p = 0; p < nd; p++)
        for (size_t i = 0; i < n * q; i++) {
            const double u = u_q[i];
            const double up = p == 0 ? 1.0 : p == 1 ? u : p == 2 ? u * u : std::pow(u, static_cast<double>(p));
            wu[static_cast<size_t>(p) * n * q + i] = w_q[i] * up;
        }
    const size_t nq = n * q;
    std::vector<double> Gr(nq * nq), Gi(nq * nq);
    const double four_pi = 4.0 * PI_;
    for (size_t t = 0; t < nq * nq; t++) {
        const double r = Rp[t];
        const double a = k.imag() * r, y = k.real() * r;
        const double half = std::sin(0.5 * y);
        double re = std::expm1(a) * std::cos(y);
        re = re - 2.0 * half * half;
        const double im = std::exp(a) * std::sin(y);
        const cdouble g = np_cdiv(cdouble(re, 0.0 - im), cdouble(four_pi * r, 0.0));
        Gr[t] = g.real();
        Gi[t] = g.imag();
    }
    Block out(static_cast<size_t>(nd * nd) * n * n, false);
    for (int p = 0; p < nd; p++)
        for (int P = 0; P < nd; P++)
            for (size_t i = 0; i < n; i++)
                for (size_t j = 0; j < n; j++) {
                    double ar = 0.0, ai = 0.0;
                    if (n == 1) {
                        for (size_t a = 0; a < q; a++)
                            for (size_t b2 = 0; b2 < q; b2++) {
                                const double w = wu[(static_cast<size_t>(p) * n + i) * q + a];
                                const double v = wu[(static_cast<size_t>(P) * n + j) * q + b2];
                                const size_t t = (i * q + a) * nq + j * q + b2;
                                ar += (w * Gr[t]) * v;
                                ai += (w * Gi[t]) * v;
                            }
                    } else {
                        for (size_t a = 0; a < q; a++) {
                            const double w = wu[(static_cast<size_t>(p) * n + i) * q + a];
                            double lr = 0.0, li = 0.0;
                            for (size_t b2 = 0; b2 < q; b2++) {
                                const double v = wu[(static_cast<size_t>(P) * n + j) * q + b2];
                                const size_t t = (i * q + a) * nq + j * q + b2;
                                lr += (w * Gr[t]) * v;
                                li += (w * Gi[t]) * v;
                            }
                            ar += lr;
                            ai += li;
                        }
                    }
                    out[((static_cast<size_t>(p) * nd + P) * n + i) * n + j] = cdouble(ar, ai);
                }
    return out;
}

// `_seg_seg_reg_moments(arc, a, k, max_d, n_qp, ek=...)` at a real k: the
// whole edge's R table and the swept kernel (its EK twin under the extended
// kernel), the length-1 k axis off. (nd, nd, n, n) C order.
Block reg_moments_real(const std::vector<double> &arc, double a_w, double k, int d, int64_t n_qp, bool ek) {
    const int nd = d + 1;
    std::vector<double> t01, w01;
    gl01(n_qp, t01, w01);
    const size_t q = t01.size(), n = arc.size() - 1;
    std::vector<double> s_q(n * q), u_q(n * q), w_q(n * q);
    for (size_t i = 0; i < n; i++) {
        const double hs = arc[i + 1] - arc[i];
        for (size_t j = 0; j < q; j++) {
            s_q[i * q + j] = arc[i] + t01[j] * hs;
            u_q[i * q + j] = (t01[j] * hs) * 1.0;
            w_q[i * q + j] = (w01[j] * hs) * 1.0;
        }
    }
    const CF sq = borrow(s_q, {static_cast<nd::ssize_t>(n * q)});
    const nd::array_t<double> R = reg_distance_table(sq, sq, a_w * a_w);
    std::vector<double> wu(static_cast<size_t>(nd) * n * q);
    for (int p = 0; p < nd; p++)
        for (size_t i = 0; i < n * q; i++) {
            const double u = u_q[i];
            const double up = p == 0 ? 1.0 : p == 1 ? u : p == 2 ? u * u : std::pow(u, static_cast<double>(p));
            wu[static_cast<size_t>(p) * n * q + i] = w_q[i] * up;
        }
    const CF wua = borrow(wu, {nd, static_cast<nd::ssize_t>(n), static_cast<nd::ssize_t>(q)});
    const std::vector<double> kv{k};
    const CF ka = borrow(kv, {1});
    const CF Rv = CF::borrow(R.buffer());
    return Block::adopt(ek ? seg_seg_reg_moments_bspline_swept_ek(Rv, wua, ka, a_w)
                           : seg_seg_reg_moments_bspline_swept(Rv, wua, ka));
}

// `numpy.searchsorted`-free `_schedule.runs`: a sorted index set's runs
std::vector<std::pair<int64_t, int64_t>> index_runs(const std::vector<int64_t> &idx) {
    std::vector<std::pair<int64_t, int64_t>> out;
    for (size_t s0 = 0; s0 < idx.size();) {
        size_t s1 = s0 + 1;
        while (s1 < idx.size() && idx[s1] == idx[s1 - 1] + 1) s1++;
        out.push_back({idx[s0], idx[s1 - 1] + 1});
        s0 = s1;
    }
    return out;
}

}  // namespace

namespace {

// `_accumulate_Z_subset_chunked` at a real k with a real eps (the above
// medium, `ek` the above-ground spec cut to the subset or off) or a complex
// k with a complex eps~ (the lower medium, no extended kernel).
template <class K, class E>
void accumulate_subset(std::vector<cdouble> &Z, const Geometry &g, const Basis &b,
                       const std::vector<double> &radius_per_wire, const FillParams &fp, K k,
                       const std::vector<int64_t> &seg_idx, bool mirror, E eps, double scale, bool weighted,
                       cdouble weight, double omega, double mu, double ground_z, int64_t swept_mem_mb,
                       const nec5::NumpyLinalg &np, const EK &ek) {
    constexpr bool CPLX = std::is_same<K, cdouble>::value;
    const int d = fp.degree, nd = d + 1;
    const int64_t N = g.n_segs_total, n_basis = b.n_basis;
    std::vector<double> a_row;  // `_seg_radius`
    for (size_t w = 0; w < g.per_wire.size(); w++)
        for (int64_t q = 0; q < g.per_wire[w].n_total; q++) a_row.push_back(radius_per_wire[w]);
    Segs real, src;
    real.l = g.seg_l;
    real.r = g.seg_r;
    src = real;
    std::vector<double> src_t(g.tangents);
    if (mirror) {  // `_image_positions`, `mirror_tangents`
        for (int64_t s2 = 0; s2 < N; s2++) {
            src.l[3 * s2 + 2] = 2.0 * ground_z - g.seg_l[3 * s2 + 2];
            src.r[3 * s2 + 2] = 2.0 * ground_z - g.seg_r[3 * s2 + 2];
            src_t[3 * s2 + 2] = g.tangents[3 * s2 + 2] * -1.0;
        }
    }
    if (weighted && (np.dgemm == nullptr || np.dgemv == nullptr || np.ddot == nullptr))
        throw FrontNotPorted("no real BLAS for the image weights");
    nd::array_t<std::complex<double>> Za =
        borrow_mut(Z, {static_cast<nd::ssize_t>(n_basis), static_cast<nd::ssize_t>(n_basis)});
    const CI sa = borrow(b.supp_seg, {static_cast<nd::ssize_t>(n_basis), nd});
    const CF pa = borrow(b.polys, {static_cast<nd::ssize_t>(n_basis), nd, nd});
    const CF ta = borrow(g.tangents, {static_cast<nd::ssize_t>(N), 3});
    const auto touching = [&](int64_t lo, int64_t hi) {  // `_bases_touching`
        std::vector<int64_t> out;
        for (int64_t m = 0; m < n_basis; m++)
            for (int q = 0; q < nd; q++) {
                const int64_t sg = b.supp_seg[static_cast<size_t>(m * nd + q)];
                if (sg >= lo && sg < hi) {
                    out.push_back(m);
                    break;
                }
            }
        return out;
    };
    using CC = nd::array_t<std::complex<double>, nd::array::c_style | nd::array::forcecast>;
    const auto accumulate = [&](Block J_win, int64_t i0, int64_t i1, int64_t j0, int64_t j1) {
        const std::vector<int64_t> m_idx = touching(i0, i1), n_idx = touching(j0, j1);
        if (m_idx.empty() || n_idx.empty()) return;
        const CI mi = borrow(m_idx, {static_cast<nd::ssize_t>(m_idx.size())});
        const CI ni = borrow(n_idx, {static_cast<nd::ssize_t>(n_idx.size())});
        if (!weighted) {
            if (scale != 1.0)
                for (cdouble &v : J_win) v = cdouble(v.real() * scale, v.imag() * scale);
            const CC Jw = borrow(J_win, {nd, nd, static_cast<nd::ssize_t>(i1 - i0), static_cast<nd::ssize_t>(j1 - j0)});
            if constexpr (std::is_same<E, cdouble>::value)
                assemble_Z_bspline_windowed_cplx_eps(Jw, sa, pa, ta, mi, ni, i0, i1, j0, j1, omega, eps, mu, Za, 0,
                                                     nd::none(), false);
            else
                assemble_Z_bspline_windowed(Jw, sa, pa, ta, mi, ni, i0, i1, j0, j1, omega, eps, mu, Za, 0, nd::none(),
                                            false);
            return;
        }
        // w_A = weight * (tangents[i0:i1] @ src_t[j0:j1].T), w_Phi = weight
        const int64_t m = i1 - i0, nn = j1 - j0;
        std::vector<double> td(static_cast<size_t>(m * nn));
        dmatmul(np, &g.tangents[3 * static_cast<size_t>(i0)], m, 3, 3, 1, &src_t[3 * static_cast<size_t>(j0)], nn, 1,
                3, td.data());
        std::vector<cdouble> wA(td.size()), wP(td.size(), weight);
        for (size_t t = 0; t < td.size(); t++) wA[t] = nec5::replay::np_cmul(weight, cdouble(td[t], 0.0));
        const CC Jw = borrow(J_win, {nd, nd, static_cast<nd::ssize_t>(m), static_cast<nd::ssize_t>(nn)});
        const CC wa = borrow(wA, {static_cast<nd::ssize_t>(m), static_cast<nd::ssize_t>(nn)});
        const CC wp = borrow(wP, {static_cast<nd::ssize_t>(m), static_cast<nd::ssize_t>(nn)});
        if constexpr (std::is_same<E, cdouble>::value)
            assemble_Z_bspline_weighted_windowed_cplx_eps(Jw, sa, pa, wa, wp, mi, ni, i0, i1, j0, j1, omega, eps, mu,
                                                          cdouble(scale, 0.0), Za, 0, nd::none(), false);
        else
            assemble_Z_bspline_weighted_windowed(Jw, sa, pa, wa, wp, mi, ni, i0, i1, j0, j1, omega, eps, mu,
                                                 cdouble(scale, 0.0), Za, 0, nd::none(), false);
    };
    const auto range = [](int64_t lo, int64_t hi) {
        std::vector<size_t> v;
        for (int64_t q = lo; q < hi; q++) v.push_back(static_cast<size_t>(q));
        return v;
    };
    // the sweep: every (observer chunk x source run) window
    const std::vector<std::pair<int64_t, int64_t>> runs = index_runs(seg_idx);
    const int64_t n_sub = static_cast<int64_t>(seg_idx.size());
    const int64_t row_bytes = static_cast<int64_t>(nd * nd) * n_sub * 16;
    const int64_t chunk = std::max<int64_t>(1, (swept_mem_mb * 1024 * 1024) / row_bytes);
    for (const auto &run : runs)
        for (int64_t c0 = run.first; c0 < run.second; c0 += chunk) {
            const int64_t c1 = std::min(c0 + chunk, run.second);
            const std::vector<size_t> rows = range(c0, c1);
            const std::vector<double> a_sub(a_row.begin() + c0, a_row.begin() + c1);
            for (const auto &srun : runs) {
                const std::vector<size_t> cols = range(srun.first, srun.second);
                accumulate(offedge(real.take(rows), src.take(cols), a_sub, k, d, fp.n_qp_pair, ek_cut(ek, &rows, &cols),
                                   fp.ladder),
                           c0, c1, srun.first, srun.second);
            }
        }
    if (mirror) return;
    // the same-edge fixup, per edge of each subset wire (wire order)
    std::vector<bool> on_subset(static_cast<size_t>(N), false);
    for (int64_t s2 : seg_idx) on_subset[static_cast<size_t>(s2)] = true;
    for (size_t w = 0; w < g.per_wire.size(); w++) {
        const WireGeometry &pw = g.per_wire[w];
        if (!on_subset[static_cast<size_t>(g.seg_offsets[w])]) continue;
        const double a_w = radius_per_wire[w];
        for (size_t e = 0; e + 1 < pw.edge_offsets.size(); e++) {
            const std::vector<double> &arc = pw.edge_arc_edges[e];
            const int64_t s0 = g.seg_offsets[w] + pw.edge_offsets[e], s1 = g.seg_offsets[w] + pw.edge_offsets[e + 1];
            const size_t n = arc.size() - 1;
            if (!is_toeplitz(arc)) throw FrontNotPorted("a non-uniform edge");
            const double h = arc[1] - arc[0];
            // `_EK_SAME_EDGE` with the extended kernel on this block
            const nd::array_t<double> A_st = ek.on ? seg_seg_static_moments_bspline_uniform_ek(h, a_w, n, d, a_w, 0)
                                                   : seg_seg_static_moments_bspline_uniform(h, a_w, n, d, 0, 0);
            Block A_reg;
            if constexpr (CPLX) {
                A_reg = reg_moments_cplx(arc, a_w, k, d, fp.n_qp_same_edge);
            } else {
                A_reg = reg_moments_real(arc, a_w, k, d, fp.n_qp_same_edge, ek.on);
            }
            const std::vector<size_t> rows = range(s0, s1);
            const std::vector<double> a_sub(a_row.begin() + s0, a_row.begin() + s1);
            const Block J_edge =
                offedge(real.take(rows), real.take(rows), a_sub, k, d, fp.n_qp_pair, ek_cut(ek, &rows, &rows), fp.ladder);
            // corr = (A_st + A_reg) - J_edge
            Block corr(A_reg.size(), false);
            const double *st = A_st.data();
            for (size_t t = 0; t < corr.size(); t++) {
                const cdouble an(st[t] + A_reg[t].real(), 0.0 + A_reg[t].imag());
                corr[t] = cdouble(an.real() - J_edge[t].real(), an.imag() - J_edge[t].imag());
            }
            accumulate(corr, s0, s1, s0, s1);
        }
    }
}

}  // namespace

void accumulate_subset_chunked(std::vector<cdouble> &Z, const Geometry &g, const Basis &b,
                               const std::vector<double> &radius_per_wire, const FillParams &fp, cdouble k,
                               const std::vector<int64_t> &seg_idx, bool mirror, cdouble eps, double scale,
                               bool weighted, cdouble weight, double omega, double mu, double ground_z,
                               int64_t swept_mem_mb, const nec5::NumpyLinalg &np) {
    accumulate_subset(Z, g, b, radius_per_wire, fp, k, seg_idx, mirror, eps, scale, weighted, weight, omega, mu,
                      ground_z, swept_mem_mb, np, EK());
}


namespace {

// a refusal on the buried route: Python's ValueError, which the caller
// declines rather than words (the plan refuses every geometry first)
struct BuriedRaise {
    [[noreturn]] static void value(const std::string &m) { throw FrontNotPorted("buried-refusal: " + m); }
};

// `_below_interface.field_nodes(seg_l, seg_r, tangents, h, q)` over the
// segments `idx` with the polynomial moment weights `W[p, i, j] = w_node *
// float_power(u, p)` (`_buried_nodes`): nodes and tangents (n q, 3), W
// (nd, n, q).
void buried_nodes(const Geometry &g, const std::vector<int64_t> &idx, int64_t q, int nd, std::vector<double> &nodes,
                  std::vector<double> &tans, std::vector<double> &W) {
    std::vector<double> xg, wg;
    if (!razor::leggauss(q, xg, wg)) throw FrontNotPorted("an untabled field order");
    const size_t n = idx.size(), Q = static_cast<size_t>(q);
    nodes.assign(3 * n * Q, 0.0);
    tans.assign(3 * n * Q, 0.0);
    W.assign(static_cast<size_t>(nd) * n * Q, 0.0);
    for (size_t i = 0; i < n; i++) {
        const size_t s2 = static_cast<size_t>(idx[i]);
        const double h = g.h_per_seg[s2];
        for (size_t j = 0; j < Q; j++) {
            const double tq = 0.5 * (xg[j] + 1.0);
            for (int c = 0; c < 3; c++) {
                const double l = g.seg_l[3 * s2 + c], r = g.seg_r[3 * s2 + c];
                nodes[3 * (i * Q + j) + c] = l + tq * (r - l);
                tans[3 * (i * Q + j) + c] = g.tangents[3 * s2 + c];
            }
            const double u = h * tq, w_node = (0.5 * h) * wg[j];
            for (int p = 0; p < nd; p++)
                W[(static_cast<size_t>(p) * n + i) * Q + j] = w_node * std::pow(u, static_cast<double>(p));
        }
    }
}

}  // namespace

BuriedMedium buried_medium(double eps_r, double sigma, double omega, double eps, double k) {
    BuriedMedium m;
    // `_ground_refl.eps_tilde`: complex(eps_r, -sigma / (omega * eps0))
    m.eps_t = cdouble(eps_r, (-sigma) / (omega * eps));
    m.k_p = k;
    m.k_m = somm_below::k_medium(m.eps_t, k);
    // `eps * eps_t`, `(eps_t - 1.0) / (eps_t + 1.0)`, `(1.0 - eps_t) /
    // (1.0 + eps_t)`: CPython's complex arithmetic, a float promoted
    m.eps_m = cdouble(eps * m.eps_t.real() - 0.0 * m.eps_t.imag(), eps * m.eps_t.imag() + 0.0 * m.eps_t.real());
    m.c2 = somm_core::py_quot(cdouble(m.eps_t.real() - 1.0, m.eps_t.imag() - 0.0),
                              cdouble(m.eps_t.real() + 1.0, m.eps_t.imag() + 0.0));
    m.a_m = somm_core::py_quot(cdouble(1.0 - m.eps_t.real(), 0.0 - m.eps_t.imag()),
                               cdouble(1.0 + m.eps_t.real(), 0.0 + m.eps_t.imag()));
    return m;
}

std::vector<cdouble> fill_buried(const Geometry &g, const Basis &b, const std::vector<double> &radius_per_wire,
                                 const FillParams &fp, double omega, double eps, double mu, double ground_z,
                                 double eps_r, double sigma, int64_t n_qp_sommerfeld, int64_t swept_mem_mb,
                                 const nec5::NumpyLinalg &np, bool stop_before_cross, CrossingStages *stages) {
    const int d = fp.degree, nd = d + 1;
    const int64_t N = g.n_segs_total, n_basis = b.n_basis;
    const BuriedMedium bm = buried_medium(eps_r, sigma, omega, eps, fp.k);
    const double gz = ground_z;
    // `plan_buried`: the media per segment, the pair classes' index sets
    std::vector<uint8_t> below(static_cast<size_t>(N), 0);
    for (size_t w = 0; w < g.per_wire.size(); w++)
        for (int64_t s2 = g.seg_offsets[w]; s2 < g.seg_offsets[w + 1]; s2++)
            below[static_cast<size_t>(s2)] = b.below_wire[w];
    std::vector<int64_t> a_idx, b_idx;
    for (int64_t s2 = 0; s2 < N; s2++) (below[static_cast<size_t>(s2)] ? b_idx : a_idx).push_back(s2);
    const bool crossing = !a_idx.empty() && !b.crossing.empty();
    if (!a_idx.empty() && !crossing) throw FrontNotPorted("buried-transmitted");
    const int64_t q = std::max<int64_t>(n_qp_sommerfeld, 6);  // `n_qp_buried_field`
    std::vector<double> obs_a, t_a, W_a, obs_b, t_b, W_b;
    buried_nodes(g, a_idx, q, nd, obs_a, t_a, W_a);
    buried_nodes(g, b_idx, q, nd, obs_b, t_b, W_b);
    const size_t n_obs = obs_b.size() / 3;
    // `serve_plan`: the below/below extents over the nodes (`_pair_extents_below`)
    std::vector<double> d_b(n_obs);
    for (size_t i = 0; i < n_obs; i++) d_b[i] = gz - obs_b[3 * i + 2];
    double r1sq_max = 0.0, q_min = std::numeric_limits<double>::infinity();
    for (size_t i = 0; i < n_obs; i++)
        for (size_t j = i; j < n_obs; j++) {
            const double dx = obs_b[3 * i] - obs_b[3 * j], dy = obs_b[3 * i + 1] - obs_b[3 * j + 1];
            const double rho2 = dx * dx + dy * dy;
            const double hh = d_b[i] + d_b[j];
            const double hh2 = hh * hh;
            const double r1sq = rho2 + hh2;
            if (r1sq > r1sq_max) r1sq_max = r1sq;
            const double qq = hh2 / rho2;
            if (qq < q_min) q_min = qq;
        }
    const double r1_max = std::sqrt(r1sq_max), th_min = std::atan(std::sqrt(q_min));
    const double lam_m = (2.0 * PI_) / std::hypot(bm.k_m.real(), bm.k_m.imag());
    const double r1_cap = (somm_below::R1_CAP_LAMBDA_M * lam_m) * (1.0 + 1e-9);
    const double floor = (0.05 - 2.0 * (0.05 / 3.0)) * (PI_ / 180.0);
    if (r1_max > r1_cap && th_min < floor) throw FrontNotPorted("buried-capped-plan");
    if (th_min < floor) throw FrontNotPorted("buried-grazing-refusal");
    // the above/above sizing (`max_image_distance` over the above segments)
    double r1_above = 0.0;
    if (!a_idx.empty()) {
        std::vector<double> la, ra;
        for (int64_t s2 : a_idx)
            for (int c = 0; c < 3; c++) {
                la.push_back(g.seg_l[3 * static_cast<size_t>(s2) + c]);
                ra.push_back(g.seg_r[3 * static_cast<size_t>(s2) + c]);
            }
        r1_above = razor::max_image_distance(la.data(), ra.data(), a_idx.size(), gz);
    }

    // the direct and image blocks, each in its own medium, chunked
    // (`accumulate_Z_subset_chunked`): the above ones at k_p with the
    // above-ground extended kernel (`_ek_spec_above`: below segments -1)
    std::vector<cdouble> Z(static_cast<size_t>(n_basis * n_basis), cdouble(0.0, 0.0));
    accumulate_subset(Z, g, b, radius_per_wire, fp, bm.k_m, b_idx, false, bm.eps_m, 1.0, false, cdouble(0.0, 0.0),
                      omega, mu, gz, swept_mem_mb, np, EK());
    if (!a_idx.empty()) {
        EK ek;
        if (fp.extended_kernel) {
            ek.on = true;
            ek.group_i.assign(static_cast<size_t>(N), 0);
            for (int64_t s2 = 0; s2 < N; s2++)
                if (below[static_cast<size_t>(s2)]) ek.group_i[static_cast<size_t>(s2)] = -1;
            ek.group_j = ek.group_i;
            std::vector<double> a_row;
            for (size_t w = 0; w < g.per_wire.size(); w++)
                for (int64_t s2 = 0; s2 < g.per_wire[w].n_total; s2++) a_row.push_back(radius_per_wire[w]);
            bool all_eq = true;
            for (double v : a_row) all_eq = all_eq && v == a_row[0];
            if (!all_eq) {
                ek.b_kind = EK::per_col;
                ek.b_j = a_row;
            }
        }
        accumulate_subset(Z, g, b, radius_per_wire, fp, bm.k_p, a_idx, false, eps, 1.0, false, cdouble(0.0, 0.0),
                          omega, mu, gz, swept_mem_mb, np, ek);
        accumulate_subset(Z, g, b, radius_per_wire, fp, bm.k_p, a_idx, true, eps, -1.0, true, bm.c2, omega, mu, gz,
                          swept_mem_mb, np, ek);
    }
    accumulate_subset(Z, g, b, radius_per_wire, fp, bm.k_m, b_idx, true, bm.eps_m, -1.0, true, bm.a_m, omega, mu, gz,
                      swept_mem_mb, np, EK());

    std::vector<int64_t> pos_a(static_cast<size_t>(N), -1), pos(static_cast<size_t>(N), -1);
    for (size_t i = 0; i < a_idx.size(); i++) pos_a[static_cast<size_t>(a_idx[i])] = static_cast<int64_t>(i);
    for (size_t i = 0; i < b_idx.size(); i++) pos[static_cast<size_t>(b_idx[i])] = static_cast<int64_t>(i);
    const size_t Q = static_cast<size_t>(q);
    // the above/above remainder, field form (`_field_galerkin_block` over
    // `remainder_field_proj` on the above grid)
    if (!a_idx.empty()) {
        // the lazy view (momwire#1512): only the nodes the remainder reads
        // over every above node pair, as `lazy_above_grid` fills them
        const somm_core::Grid &G0 = somm_core::get_grid_lazy(bm.eps_t, bm.k_p, r1_above, omega, mu);
        const size_t n_reg = G0.regions.size();
        if (n_reg > somm_proj::MAX_REGIONS) throw FrontNotPorted("a grid layout past the view's regions");
        {
            std::vector<std::vector<uint8_t>> need(n_reg);
            double r0[somm_proj::MAX_REGIONS], dr[somm_proj::MAX_REGIONS], th0[somm_proj::MAX_REGIONS],
                dth[somm_proj::MAX_REGIONS];
            const cdouble *vals[somm_proj::MAX_REGIONS];
            std::ptrdiff_t shape0[somm_proj::MAX_REGIONS], nr[somm_proj::MAX_REGIONS], nth[somm_proj::MAX_REGIONS];
            somm_proj::NodeMarks marks;
            marks.n_reg = n_reg;
            for (size_t r = 0; r < n_reg; r++) {
                const somm_core::GridRegion &reg = G0.regions[r];
                r0[r] = reg.r0;
                dr[r] = reg.dr;
                th0[r] = reg.th0;
                dth[r] = reg.dth;
                vals[r] = reg.vals.data();
                shape0[r] = 4;
                nr[r] = static_cast<std::ptrdiff_t>(reg.n_r);
                nth[r] = static_cast<std::ptrdiff_t>(reg.n_th);
                need[r].assign(static_cast<size_t>(reg.n_r * reg.n_th), 0);
                marks.m[r] = need[r].data();
                marks.nR[r] = nr[r];
                marks.nTh[r] = nth[r];
            }
            const somm_proj::GridView cv = somm_proj::build_grid_view(
                G0.r1_max, G0.r_break, 30.0 * (M_PI / 180.0), G0.r_near, r0, dr, th0, dth, n_reg, vals, shape0, nr, nth);
            const std::ptrdiff_t na = static_cast<std::ptrdiff_t>(obs_a.size() / 3);
            if (somm_core::census_pays(static_cast<double>(na) * static_cast<double>(na),
                                       somm_core::unfilled_nodes(G0)))
                somm_core::field_proj_nodes(cv, na, na, obs_a.data(), obs_a.data(), gz, marks);
            else
                need = somm_core::all_nodes_of(G0);  // the census would cost more than it saves
            somm_core::ensure_nodes(G0, need);
        }
        double r0[somm_proj::MAX_REGIONS], dr[somm_proj::MAX_REGIONS], th0[somm_proj::MAX_REGIONS],
            dth[somm_proj::MAX_REGIONS];
        const cdouble *vals[somm_proj::MAX_REGIONS];
        std::ptrdiff_t shape0[somm_proj::MAX_REGIONS], nr[somm_proj::MAX_REGIONS], nth[somm_proj::MAX_REGIONS];
        for (size_t r = 0; r < n_reg; r++) {
            const somm_core::GridRegion &reg = G0.regions[r];
            r0[r] = reg.r0;
            dr[r] = reg.dr;
            th0[r] = reg.th0;
            dth[r] = reg.dth;
            vals[r] = reg.vals.data();
            shape0[r] = 4;
            nr[r] = static_cast<std::ptrdiff_t>(reg.n_r);
            nth[r] = static_cast<std::ptrdiff_t>(reg.n_th);
        }
        somm_proj::GridView view = somm_proj::build_grid_view(G0.r1_max, G0.r_break, 30.0 * (M_PI / 180.0), G0.r_near,
                                                              r0, dr, th0, dth, n_reg, vals, shape0, nr, nth);
        const std::vector<cdouble> far = somm_core::far_pack(G0);
        somm_proj::set_far(view, far.data(), static_cast<std::ptrdiff_t>(far.size()));
        const int64_t n_a = static_cast<int64_t>(a_idx.size());
        const int64_t chunk = std::max<int64_t>(1, (int64_t(1) << 19) / std::max<int64_t>(n_a * q * q, 1));
        const size_t S = static_cast<size_t>(n_a) * Q;
        for (int64_t i0 = 0; i0 < n_a; i0 += chunk) {
            const int64_t i1 = std::min(i0 + chunk, n_a);
            const size_t m = static_cast<size_t>(i1 - i0) * Q;
            std::vector<cdouble> proj(m * S);
            somm_core::field_proj_batch(view, n_reg, static_cast<std::ptrdiff_t>(m), static_cast<std::ptrdiff_t>(S),
                                        &obs_a[3 * static_cast<size_t>(i0) * Q], &t_a[3 * static_cast<size_t>(i0) * Q],
                                        obs_a.data(), t_a.data(), gz, bm.k_p, proj.data(), 0, true);
            std::vector<double> W_rows(static_cast<size_t>(nd) * m);
            for (int p = 0; p < nd; p++)
                std::copy(W_a.begin() + (static_cast<std::ptrdiff_t>(p) * n_a + i0) * q,
                          W_a.begin() + (static_cast<std::ptrdiff_t>(p) * n_a + i1) * q,
                          W_rows.begin() + static_cast<std::ptrdiff_t>(p) * (i1 - i0) * q);
            mw_fg::Args args;
            args.pj = proj.data();
            args.wo = W_rows.data();
            args.ws = W_a.data();
            args.pl = b.polys.data();
            args.ss = b.supp_seg.data();
            args.po = pos_a.data();
            args.ps = pos_a.data();
            args.P = nd;
            args.nc = i1 - i0;
            args.q = q;
            args.ns = n_a;
            args.nb = n_basis;
            args.A = nd;
            args.i0 = i0;
            args.q_rows = n_basis;
            args.Qp = Z.data();
            args.sr = n_basis;
            args.sc = 1;
            args.fused = true;
            args.scale = -1.0;
            args.lanes = true;
            mw_fg::assemble(args);
        }
    }

    // the below/below remainder, field form, each unordered pair once
    // (`_field_galerkin_block_symmetric`)
    somm_below::Source *src = somm_below::source();
    if (src == nullptr) throw FrontNotPorted("below-table");
    somm_below::Remainder<BuriedRaise> rem;
    try {
        rem.grid = &src->get(bm.eps_t, bm.k_p, r1_max, omega, mu, somm_below::SPACING_FILL);
    } catch (const somm_below::TailCapped &e) {
        throw FrontNotPorted(std::string("buried-tail-capped: ") + e.what());
    }
    rem.build_view();
    rem.gz = gz;
    rem.k_p = bm.k_p;
    rem.k_m = bm.k_m;
    const int64_t n_b = static_cast<int64_t>(b_idx.size());
    const int64_t chunk = std::max<int64_t>(1, (int64_t(1) << 19) / std::max<int64_t>(n_b * q * q, 1));
    for (int64_t i0 = 0; i0 < n_b; i0 += chunk) {
        const int64_t i1 = std::min(i0 + chunk, n_b);
        const size_t m = static_cast<size_t>(i1 - i0) * Q, S = static_cast<size_t>(n_b - i0) * Q;
        rem.src.assign(obs_b.begin() + 3 * static_cast<std::ptrdiff_t>(i0) * q, obs_b.end());
        rem.t_src.assign(t_b.begin() + 3 * static_cast<std::ptrdiff_t>(i0) * q, t_b.end());
        rem.d_src.resize(S);
        for (size_t j = 0; j < S; j++) rem.d_src[j] = gz - rem.src[3 * j + 2];
        std::vector<cdouble> proj(m * S);
        rem.project(&obs_b[3 * static_cast<size_t>(i0) * Q], &t_b[3 * static_cast<size_t>(i0) * Q],
                    static_cast<std::ptrdiff_t>(m), proj.data());
        std::vector<double> W_rows(static_cast<size_t>(nd) * (i1 - i0) * Q), W_cols(static_cast<size_t>(nd) * S);
        for (int p = 0; p < nd; p++) {
            std::copy(W_b.begin() + (static_cast<std::ptrdiff_t>(p) * n_b + i0) * q,
                      W_b.begin() + (static_cast<std::ptrdiff_t>(p) * n_b + i1) * q,
                      W_rows.begin() + static_cast<std::ptrdiff_t>(p) * (i1 - i0) * q);
            std::copy(W_b.begin() + (static_cast<std::ptrdiff_t>(p) * n_b + i0) * q,
                      W_b.begin() + (static_cast<std::ptrdiff_t>(p) + 1) * n_b * q,
                      W_cols.begin() + static_cast<std::ptrdiff_t>(p) * (n_b - i0) * q);
        }
        std::vector<int64_t> pos_s(pos.size()), pos_m(pos.size());
        for (size_t t = 0; t < pos.size(); t++) {
            pos_s[t] = pos[t] >= i0 ? pos[t] - i0 : -1;
            pos_m[t] = pos[t] >= i1 ? pos[t] - i0 : -1;
        }
        mw_fg::Args args;
        args.pj = proj.data();
        args.wo = W_rows.data();
        args.ws = W_cols.data();
        args.pl = b.polys.data();
        args.ss = b.supp_seg.data();
        args.po = pos.data();
        args.ps = pos_s.data();
        args.P = nd;
        args.nc = i1 - i0;
        args.q = q;
        args.ns = n_b - i0;
        args.nb = n_basis;
        args.A = nd;
        args.i0 = i0;
        args.q_rows = n_basis;
        args.Qp = Z.data();
        args.sr = n_basis;
        args.sc = 1;
        args.fused = true;
        args.scale = -1.0;
        if (i1 < n_b) {  // the mirror through Z.T, in the same call (momwire#1290)
            args.Q2 = Z.data();
            args.sr2 = 1;
            args.sc2 = n_basis;
            args.ps2 = pos_m.data();
        }
        args.lanes = true;
        mw_fg::assemble(args);
    }
    // the crossing serve (`_crossing_fill`, momwire#524 phase 2): the cross
    // block's split fill and the self completions, on a one-radius deck
    if (crossing && !stop_before_cross) {
        // `_two_radius_crossing` (antennaknobs plan U5): with radii that
        // differ, `crossing_side_radii(media, radii, crossing_above_member(
        // ...))` -- the above member at the node's radius and the buried
        // side's one radius; the axes take the smaller (`ctx.a_wire`)
        double rmin = radius_per_wire[0], rmax = radius_per_wire[0];
        for (double r : radius_per_wire) {
            rmin = std::min(rmin, r);
            rmax = std::max(rmax, r);
        }
        if (rmax - rmin > 0.0) {
            int64_t above_member = -1;
            if (b.crossing.size() == 1)
                for (size_t w = 0; w < b.below_wire.size() && above_member < 0; w++)
                    if (!b.below_wire[w] &&
                        (b.start_status[w] == b.crossing[0] || b.end_status[w] == b.crossing[0]))
                        above_member = static_cast<int64_t>(w);
            double side_r[2] = {0.0, 0.0};
            for (int side = 0; side < 2; side++) {
                if (side == 0 && above_member >= 0) {
                    side_r[0] = radius_per_wire[static_cast<size_t>(above_member)];
                    continue;
                }
                bool have = false;
                double lo = 0.0, hi = 0.0;
                for (size_t w = 0; w < b.below_wire.size(); w++) {
                    if ((b.below_wire[w] != 0) != (side == 1)) continue;
                    const double r = radius_per_wire[w];
                    if (!have) {
                        lo = hi = r;
                        side_r[side] = r;
                        have = true;
                    } else {
                        lo = std::min(lo, r);
                        hi = std::max(hi, r);
                    }
                }
                // a spread within one side: Python's NotImplementedError
                if (hi - lo > 0.0) throw FrontNotPorted("buried-two-radius-spread");
            }
            // several crossing nodes at two radii: refused (plan U9)
            if (b.crossing.size() > 1) throw FrontNotPorted("buried-two-radius-nodes");
            crossing_serve(Z, g, b, std::min(side_r[0], side_r[1]), bm, gz, omega, mu, eps, a_idx, b_idx, np,
                           stages, side_r[0], side_r[1]);
        } else {
            crossing_serve(Z, g, b, radius_per_wire[0], bm, gz, omega, mu, eps, a_idx, b_idx, np, stages);
        }
    }
    return Z;
}
}  // namespace bspline
}  // namespace mw
