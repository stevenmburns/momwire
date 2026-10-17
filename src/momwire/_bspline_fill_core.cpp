// PROTOTYPE (momwire#1517 step 1): see `_bspline_fill_core.h`. Each step is
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

}  // namespace bspline
}  // namespace mw
