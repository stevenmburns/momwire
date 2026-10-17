// momwire#1517 step 1: see `_bspline_fill_core.h`. Each step is
// `_bspline_kernels.py` / `bspline.py` as written; the kernels are the
// extension's own (`mw::bspline::*`, the default host).
#include "_bspline_fill_core.h"

#include "_bspline_core.h"
#include "_exact_kernel_coaxial_core.h"
#include "_razor_exact_core.h"
#include "_razor_front_core.h"  // `leggauss`: `_quadrature.leggauss`'s frozen table

#if defined(_MSC_VER)
#pragma float_control(precise, on)
#pragma fp_contract(off)
#endif

#include <algorithm>
#include <array>
#include <climits>
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
        b.v = offedge(si, sj, a_rows, fp_.k, fp_.degree, fp_.n_qp_pair, ekc, fp_.ladder);
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

}  // namespace bspline
}  // namespace mw
