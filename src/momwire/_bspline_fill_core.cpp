// momwire#1517 step 1: see `_bspline_fill_core.h`. Each step is
// `_bspline_kernels.py` / `bspline.py` as written; the kernels are the
// extension's own (`mw::bspline::*`, the default host).
#include "_bspline_fill_core.h"

#include "_bspline_core.h"
#include "_razor_front_core.h"  // `leggauss`: `_quadrature.leggauss`'s frozen table

#if defined(_MSC_VER)
#pragma float_control(precise, on)
#pragma fp_contract(off)
#endif

#include <algorithm>
#include <cmath>
#include <map>
#include <stdexcept>

namespace mw {
namespace bspline {

namespace {

using CF = nd::array_t<double, nd::array::c_style | nd::array::forcecast>;
using CI = nd::array_t<int64_t, nd::array::c_style | nd::array::forcecast>;

template <class T>
nd::array_t<T, nd::array::c_style | nd::array::forcecast> borrow(const std::vector<T> &v,
                                                                 std::vector<nd::ssize_t> shape) {
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

using Block = std::vector<cdouble>;  // (nd, nd, Ni, Nj) C order

Block offedge(const Segs &si, const Segs &sj, const std::vector<double> &a_rows, double k, int d,
              int64_t n_qp, const EK &ek, const Ladder &ladder_in);

// Scatter `sub` (nd, nd, |ri|, |cj|) into `out` (nd, nd, Ni, Nj).
void place(Block &out, size_t Ni, size_t Nj, const Block &sub, const std::vector<size_t> &ri,
           const std::vector<size_t> &cj, int nd) {
    for (int p = 0; p < nd; p++)
        for (int q = 0; q < nd; q++)
            for (size_t a = 0; a < ri.size(); a++)
                for (size_t b = 0; b < cj.size(); b++)
                    out[((static_cast<size_t>(p) * nd + q) * Ni + ri[a]) * Nj + cj[b]] =
                        sub[((static_cast<size_t>(p) * nd + q) * ri.size() + a) * cj.size() + b];
}

Block from_nd(const nd::array_t<std::complex<double>> &a) {
    return Block(a.data(), a.data() + a.size());
}

// `_seg_seg_full_moments_offedge`, real k.
Block offedge(const Segs &si, const Segs &sj, const std::vector<double> &a_rows, double k, int d,
              int64_t n_qp, const EK &ek, const Ladder &ladder) {
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
        for (size_t i = 0; i < Ni; i++) (si.len(i) * std::fabs(k) <= PHASE_KL_CEILING ? ok_i : no_i).push_back(i);
        for (size_t j = 0; j < Nj; j++) (sj.len(j) * std::fabs(k) <= PHASE_KL_CEILING ? ok_j : no_j).push_back(j);
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
        if (!(std::fabs(k) * len_max <= PHASE_KL_CEILING)) {
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
        if (!ek.on)
            return from_nd(seg_seg_full_moments_bspline_tiered(li, ri, lj, rj, a2, k, d, ta, wa, oa, ra, false));
        const CI gi = borrow(ek.group_i, {static_cast<nd::ssize_t>(Ni)}),
                 gj = borrow(ek.group_j, {static_cast<nd::ssize_t>(Nj)});
        return from_nd(seg_seg_full_moments_bspline_ek_tiered(li, ri, lj, rj, a2, k, d, ta, wa, oa, ra, gi,
                                                               gj, a, false, a_src));
    }
    const CF ta = borrow(t01, {static_cast<nd::ssize_t>(t01.size())}),
             wa = borrow(w01, {static_cast<nd::ssize_t>(w01.size())});
    if (!ek.on) return from_nd(seg_seg_full_moments_bspline(li, ri, lj, rj, a2, k, d, ta, wa, false));
    const CI gi = borrow(ek.group_i, {static_cast<nd::ssize_t>(Ni)}),
             gj = borrow(ek.group_j, {static_cast<nd::ssize_t>(Nj)});
    return from_nd(seg_seg_full_moments_bspline_ek(li, ri, lj, rj, a2, k, d, ta, wa, gi, gj, a, false, a_src));
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

std::vector<cdouble> build_J_blocks(const Geometry &g, const std::vector<double> &radius_per_wire,
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
    std::vector<cdouble> J;
    if (slices.size() == 1 && slices[0].first == 0 && slices[0].second == N) {
        J.assign(static_cast<size_t>(nd * nd) * N * N, cdouble(0.0, 0.0));
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
        J = offedge(s, s, seg_a, fp.k, d, fp.n_qp_pair, ek, fp.ladder);
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
    if (!i1b || !i2b || n == 1) throw FrontNotPorted("a real product numpy does not hand BLAS");
    if (m == 1 && p == 1) {
        out[0] = 0.0 + np.ddot(n, a, a1, b, b0);  // `DOUBLE_dot`: sum += ddot
        return;
    }
    if (m == 1 && blasable(a1, 1, 1)) {  // vector @ matrix: gemv on b's transpose
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
    if (p == 1 && blasable(b0, 1, 1)) {  // matrix @ vector
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
    if (m == 1 || p == 1) throw FrontNotPorted("a real product numpy does not hand BLAS");
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
                accumulate(Block(corr.data(), corr.data() + corr.size()), r0, r1, s0, s1, touching(r0, r1), e_idx);
            }
        }
    }
    return Z;
}

std::vector<cdouble> assemble_Z(const std::vector<cdouble> &J, const Basis &b, const Geometry &g,
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

}  // namespace bspline
}  // namespace mw
