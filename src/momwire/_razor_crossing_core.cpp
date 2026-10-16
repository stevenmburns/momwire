// razor-2p's crossing blocks in C++: see `_razor_crossing_core.h`. Every
// step names the `_crossing_fill.py` / razor.py function it replays.
#include "_razor_crossing_core.h"

#include "_near_interface_eval.h"
#include "_somm_below_grid_core.h"
#include "_somm_grid_core.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <map>
#include <set>
#include <stdexcept>
#include <thread>

#if defined(_OPENMP)
#include <omp.h>
#endif

namespace mw {
namespace razor {

namespace {

using ni_eval::Six;

constexpr double PLANE_TOL = 1e-12;     // `_crossing_fill._PLANE_TOL`
constexpr int64_t END_BATCH_PAIRS = 1 << 16;
constexpr int K_U = 0, K_V = 1, K_W = 2, K_DZPW = 5;  // KEYS order

// numpy's fused complex product (`_cmul.cmul`'s formula, operand order kept).
inline cd np_cmul(cd a, cd b) {
    return cd(std::fma(a.real(), b.real(), -(a.imag() * b.imag())),
              std::fma(a.real(), b.imag(), a.imag() * b.real()));
}
// numpy's complex division (`CDOUBLE_divide`, Smith's).
inline cd np_cdiv(cd a, cd b) {
    const double in1r = a.real(), in1i = a.imag(), in2r = b.real(), in2i = b.imag();
    const double in2r_abs = std::fabs(in2r), in2i_abs = std::fabs(in2i);
    if (in2r_abs >= in2i_abs) {
        if (in2r_abs == 0 && in2i_abs == 0) return cd(in1r / in2r_abs, in1i / in2i_abs);
        const double rat = in2i / in2r;
        const double scl = 1.0 / (in2r + in2i * rat);
        return cd((in1r + in1i * rat) * scl, (in1i - in1r * rat) * scl);
    }
    const double rat = in2r / in2i;
    const double scl = 1.0 / (in2i + in2r * rat);
    return cd((in1r * rat + in1i) * scl, (in1i * rat - in1r) * scl);
}
// CPython's complex product.
inline cd py_prod(cd a, cd b) {
    return cd(a.real() * b.real() - a.imag() * b.imag(), a.real() * b.imag() + a.imag() * b.real());
}

[[noreturn]] void raise_value(const std::string &m) {
    throw FrontError(FrontErrorKind::value_error, m, true);
}

std::string py_g6(double x) {
    char buf[64];
    std::snprintf(buf, sizeof buf, "%.6g", x);
    return buf;
}

// `_on_plane_side(zrel, side, what)` on one value.
double on_plane_side(double z, bool above, const char *what) {
    const bool wrong = above ? z < 0.0 : z > 0.0;
    if (!wrong) return z;
    if (std::fabs(z) >= PLANE_TOL) {
        const char *side = above ? "above" : "below";
        raise_value(std::string("crossing fill: the ") + side + " axis's " + what + " sits " +
                    py_g6(std::fabs(z)) + " on the wrong side of ground_z, past the 1e-12 "
                    "tolerance that says a point IS the interface. The designed tables are "
                    "derived for z >= 0 >= z', so there is no honest side to put this on: an " +
                    side + " member must not cross the plane. Check the deck's crossing "
                    "junction -- exactly one above member and the rest below (momwire#852)");
    }
    return 0.0;
}

struct CSR {
    int64_t n_rows = 0, n_cols = 0;
    std::vector<int64_t> indptr, indices;
    std::vector<double> data;
};

// Rows (r, c, v) to a canonical CSR (indices sorted within each row; no
// duplicates in our inputs).
CSR to_csr(int64_t n_rows, int64_t n_cols, std::vector<std::tuple<int64_t, int64_t, double>> t) {
    std::stable_sort(t.begin(), t.end(), [](const std::tuple<int64_t, int64_t, double> &a,
                                            const std::tuple<int64_t, int64_t, double> &b) {
        return std::get<0>(a) != std::get<0>(b) ? std::get<0>(a) < std::get<0>(b)
                                                : std::get<1>(a) < std::get<1>(b);
    });
    CSR m;
    m.n_rows = n_rows;
    m.n_cols = n_cols;
    m.indptr.assign(static_cast<size_t>(n_rows + 1), 0);
    for (const auto &e : t) m.indptr[static_cast<size_t>(std::get<0>(e)) + 1]++;
    for (int64_t r = 0; r < n_rows; r++) m.indptr[r + 1] += m.indptr[r];
    for (const auto &e : t) {
        m.indices.push_back(std::get<1>(e));
        m.data.push_back(std::get<2>(e));
    }
    return m;
}

struct End {
    double pt[3];
    double sign;
    std::vector<double> fv;  // (n_basis)
};

struct Axis {
    int64_t n_basis = 0;
    std::vector<double> nodes, t, w;  // (n, 3), (n, 3), (n)
    CSR F, Fd;
    std::vector<End> ends;
    bool path_tested = false;
    std::vector<int64_t> segof;
    std::map<int64_t, std::vector<int64_t>> seg_rows;
    size_t n_nodes() const { return w.size(); }
};

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

// `_path_test_rows` + `path_test_axis`: one record per row -- its path's
// points (the two wing centroids, or one half with the knot), tangents,
// weights, and the T2 endpoints.
struct Rec {
    int64_t m;
    std::vector<double> pts, tans, wts;
    std::vector<int64_t> seg;
    double before[3], after[3];
};

Axis path_test_axis(int64_t n_basis, const std::vector<Rec> &recs) {
    Axis ax;
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
        End b, a;
        std::copy(r.before, r.before + 3, b.pt);
        b.sign = -1.0;
        b.fv.assign(static_cast<size_t>(n_basis), 0.0);
        b.fv[static_cast<size_t>(r.m)] = 1.0;
        std::copy(r.after, r.after + 3, a.pt);
        a.sign = +1.0;
        a.fv = b.fv;
        ax.ends.push_back(b);
        ax.ends.push_back(a);
    }
    ax.F = to_csr(n_basis, off, trip);
    ax.Fd = to_csr(n_basis, off, {});
    return ax;
}

// `_crossing_path_axis(geom, tents, side)`: the medium's own tents whole,
// each crossing tent's live half (the knot its shared endpoint).
Axis crossing_path_axis(const Front &f, const std::vector<std::pair<int64_t, int>> &tents, bool below) {
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

// `_graded_u(h, toward_end, a, growth, gx, gw)`.
void graded_u(double h, bool toward_hi, double a, double growth, const std::vector<double> &gx,
              const std::vector<double> &gw, std::vector<double> &u, std::vector<double> &w) {
    std::vector<double> edges{0.0};
    double step = a;
    while (edges.back() + step < h) {
        edges.push_back(edges.back() + step);
        step *= growth;
    }
    edges.push_back(h);
    if (toward_hi) {
        std::vector<double> e(edges.size());
        for (size_t i = 0; i < edges.size(); i++) e[i] = h - edges[edges.size() - 1 - i];
        edges.swap(e);
    }
    for (size_t p = 0; p + 1 < edges.size(); p++) {
        const double mid = 0.5 * (edges[p] + edges[p + 1]);
        const double half = 0.5 * (edges[p + 1] - edges[p]);
        for (size_t k = 0; k < gx.size(); k++) {
            u.push_back(mid + half * gx[k]);
            w.push_back(half * gw[k]);
        }
    }
}

// `axis_data(ctx, seg_idx, grade_near_plane=True, growth=2.0, panel_order=8,
// q=12)` with razor's tent polynomials.
Axis axis_data(const Ctx &c, const std::vector<int64_t> &seg_idx, double a_wire) {
    const Front &f = *c.f;
    Axis ax;
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
            if (any) ax.ends.push_back(std::move(e));
        }
    }
    return ax;
}

// `_support_rows(ax, all nodes)`.
std::vector<int64_t> support_rows(const Axis &ax) {
    std::vector<int64_t> out;
    if (ax.path_tested) {
        for (int64_t r = 0; r < ax.F.n_rows; r++)
            if (ax.F.indptr[r + 1] > ax.F.indptr[r] || ax.Fd.indptr[r + 1] > ax.Fd.indptr[r]) out.push_back(r);
        return out;
    }
    std::set<int64_t> rows;
    std::set<int64_t> segs(ax.segof.begin(), ax.segof.end());
    for (int64_t g : segs) {
        auto it = ax.seg_rows.find(g);
        if (it != ax.seg_rows.end()) rows.insert(it->second.begin(), it->second.end());
    }
    return std::vector<int64_t>(rows.begin(), rows.end());
}

// `_row_weights(ax, all, rows)`: (F w tx, F w ty, F w tz, Fd w) restricted
// to `rows`.
std::array<CSR, 4> row_weights(const Axis &ax, const std::vector<int64_t> &rows) {
    std::array<CSR, 4> out;
    for (int m = 0; m < 4; m++) {
        CSR &o = out[static_cast<size_t>(m)];
        o.n_rows = static_cast<int64_t>(rows.size());
        o.n_cols = ax.F.n_cols;
        o.indptr.assign(rows.size() + 1, 0);
        const CSR &src = m < 3 ? ax.F : ax.Fd;
        for (size_t i = 0; i < rows.size(); i++) {
            const int64_t r = rows[i];
            for (int64_t jj = src.indptr[r]; jj < src.indptr[r + 1]; jj++) {
                const int64_t col = src.indices[jj];
                double v = src.data[jj] * ax.w[static_cast<size_t>(col)];
                if (m < 3) v = v * ax.t[3 * static_cast<size_t>(col) + m];
                o.indices.push_back(col);
                o.data.push_back(v);
            }
            o.indptr[i + 1] = static_cast<int64_t>(o.indices.size());
        }
    }
    return out;
}

// One block's memo and evaluator.
struct BlockEval {
    ni_eval::Evaluator ev;
    ni_eval::Memo memo;
};

// The OpenMP team for the block's own loops: the evaluator's cap, held to
// what OpenMP would give (1 in the serial copy). Every loop it splits writes
// disjoint outputs, each one's arithmetic its serial order's.
int team(const BlockEval &be, size_t work) {
#if defined(_OPENMP)
    int nt = omp_get_max_threads();
    if (be.ev.n_threads > 0) nt = std::min(nt, be.ev.n_threads);
    return work < 4096 ? 1 : std::max(1, nt);
#else
    (void)be;
    (void)work;
    return 1;
#endif
}

// `_tables(ctx_r, eps_t, k_p, rho, z, zp, rtol, memo, group_labels)` over n
// asked points: rho folded with the radius.
void tables(BlockEval &be, double a_wire, const std::vector<double> &rho, const std::vector<double> &z,
            const std::vector<double> &zp, const std::vector<int64_t> *labels, std::vector<Six> &out) {
    std::vector<double> re(rho.size());
    for (size_t i = 0; i < rho.size(); i++) re[i] = std::hypot(rho[i], a_wire);
    out.resize(rho.size());
    ni_eval::designed_tables(be.ev, &be.memo, re.data(), z.data(), zp.data(), rho.size(),
                             labels ? labels->data() : nullptr, out.data());
}

// The same, each point answered as its value's index in the block's memo
// (`rho` already folded with the radius).
void table_index(BlockEval &be, const std::vector<double> &re, const std::vector<double> &z,
                 const std::vector<double> &zp, const std::vector<int64_t> *labels, std::vector<uint32_t> &idx) {
    idx.resize(re.size());
    ni_eval::designed_index(be.ev, be.memo, re.data(), z.data(), zp.data(), re.size(),
                            labels ? labels->data() : nullptr, idx.data(), team(be, re.size()));
}

// One key of every memo value, contiguous (the contractions' gathers).
std::vector<cd> key_plane(const ni_eval::Memo &memo, int key) {
    std::vector<cd> out(memo.size());
    for (size_t j = 0; j < out.size(); j++) out[j] = memo.at(static_cast<uint32_t>(j))[static_cast<size_t>(key)];
    return out;
}

// One real CSR row against complex v: each part's products summed in the
// row's order (`(M @ v.real) + 1j (M @ v.imag)`, scipy's row loop). Plain
// restrict-qualified pointers: through the vectors' accessors the loop ran
// ten times slower, with the same arithmetic.
inline void csr_row_dot(const int64_t *__restrict ix, const double *__restrict d, int64_t j0, int64_t j1,
                        const double *__restrict v, double &re_out, double &im_out) {
    double re = 0.0, im = 0.0;
    for (int64_t jj = j0; jj < j1; jj++) {
        const double *x = v + 2 * ix[jj];
        const double a = d[jj];
        re += a * x[0];
        im += a * x[1];
    }
    re_out = re;
    im_out = im;
}

// y[c] += a * tab[ix[c]], each part one product and one sum (no FMA): the
// left products' inner loop, with nothing aliased so it vectorizes alike in
// the serial and the threaded copy (elementwise, so its bits never depend on
// how it is vectorized).
void axpy_gather(cd *__restrict y, const cd *__restrict tab, const uint32_t *__restrict ix, double a,
                 size_t n) {
    double *__restrict yd = reinterpret_cast<double *>(y);
    const double *__restrict td = reinterpret_cast<const double *>(tab);
    for (size_t cc = 0; cc < n; cc++) {
        const size_t k = 2 * static_cast<size_t>(ix[cc]);
        yd[2 * cc] = yd[2 * cc] + a * td[k];
        yd[2 * cc + 1] = yd[2 * cc + 1] + a * td[k + 1];
    }
}

// `_main_sandwich(ctx, A, B, ...)` with the plan: the (nA basis x nB basis)
// block of the five-term contraction, times c1 (zero outside rA x rB, then
// times c1 as numpy multiplies the whole block).
std::vector<cd> main_sandwich(const Ctx &c, BlockEval &be, double a_wire, const Axis &A, const Axis &B) {
    const size_t nA = A.n_nodes(), nB = B.n_nodes();
    // The sheet plan, once per fill, from the node pairs.
    {
        std::vector<double> pa(A.nodes), pb(B.nodes);
        for (size_t i = 0; i < nA; i++) pa[3 * i + 2] -= c.gz;
        for (size_t i = 0; i < nB; i++) pb[3 * i + 2] -= c.gz;
        be.memo.plan = ni_eval::sheet_plan(c.k_p, c.k_m, {ni_eval::NodePair{pa.data(), nA, pb.data(), nB}}, a_wire);
        be.memo.have_plan = true;
    }
    // The tables over the whole grid, row a * nB + b.
    std::vector<uint32_t> K;
    {
        std::vector<double> re(nA * nB), z(nA * nB), zp(nA * nB);
        const std::ptrdiff_t nAp = static_cast<std::ptrdiff_t>(nA);
        const int nt = team(be, nA * nB);
#if defined(_OPENMP)
        #pragma omp parallel for schedule(static) num_threads(nt) if (nt > 1)
#else
        (void)nt;
#endif
        for (std::ptrdiff_t ap = 0; ap < nAp; ap++) {
            const size_t a = static_cast<size_t>(ap);
            for (size_t b = 0; b < nB; b++) {
                const size_t q = a * nB + b;
                const double rho = std::hypot(A.nodes[3 * a] - B.nodes[3 * b], A.nodes[3 * a + 1] - B.nodes[3 * b + 1]);
                re[q] = std::hypot(rho, a_wire);
                z[q] = A.nodes[3 * a + 2] - c.gz;
                zp[q] = B.nodes[3 * b + 2] - c.gz;
            }
        }
        table_index(be, re, z, zp, nullptr, K);
    }
    const double k2sq = c.k_p * c.k_p;
    const std::vector<int64_t> rA = support_rows(A), rB = support_rows(B);
    const std::array<CSR, 4> P = row_weights(A, rA), Q = row_weights(B, rB);
    // The four keys the contraction reads, one plane each over the memo's
    // values; `k2 V + dzpW` is a function of the value alone.
    const std::vector<cd> KU = key_plane(be.memo, K_U), KV = key_plane(be.memo, K_V),
                          KW = key_plane(be.memo, K_W);
    std::vector<cd> KX(KV.size());
    {
        const std::vector<cd> KD = key_plane(be.memo, K_DZPW);
        for (size_t j = 0; j < KX.size(); j++)
            KX[j] = cd(k2sq * KV[j].real() + KD[j].real(), k2sq * KV[j].imag() + KD[j].imag());
    }
    // Per row of rA (rows apart, any thread): its left products
    // (`left_products_gathered`'s sums, six (nB) rows) into the thread's own
    // buffer, then `combine_rows` against Q's rows: ((((T0 + T1) + T2) + T3)
    // + T4) - T5. Only the row's own products feed its combine, so nothing
    // (nr x nB) is ever held.
    static const int qsel[6] = {0, 1, 2, 3, 2, 3};
    std::vector<cd> t(static_cast<size_t>(A.n_basis * B.n_basis), cd(0.0, 0.0));
    const size_t nr = rA.size();
    const std::ptrdiff_t nrp = static_cast<std::ptrdiff_t>(nr);
    const int nt = team(be, nr * nB);
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
    for (std::ptrdiff_t ip = 0; ip < nrp; ip++) {
        const size_t i = static_cast<size_t>(ip);
        std::fill(Lrow.begin(), Lrow.end(), cd(0.0, 0.0));
        cd *L[6];
        for (int m = 0; m < 6; m++) L[m] = &Lrow[static_cast<size_t>(m) * nB];
        for (int pm = 0; pm < 3; pm++) {
            const CSR &M = P[static_cast<size_t>(pm)];
            for (int64_t jj = M.indptr[i]; jj < M.indptr[i + 1]; jj++) {
                const double a = M.data[static_cast<size_t>(jj)];
                const uint32_t *kx = &K[static_cast<size_t>(M.indices[static_cast<size_t>(jj)]) * nB];
                if (pm < 2) {
                    axpy_gather(L[pm], KU.data(), kx, a, nB);
                } else {
                    axpy_gather(L[2], KX.data(), kx, a, nB);
                    axpy_gather(L[3], KW.data(), kx, a, nB);
                }
            }
        }
        const CSR &M = P[3];
        for (int64_t jj = M.indptr[i]; jj < M.indptr[i + 1]; jj++) {
            const double a = M.data[static_cast<size_t>(jj)];
            const uint32_t *kx = &K[static_cast<size_t>(M.indices[static_cast<size_t>(jj)]) * nB];
            axpy_gather(L[4], KW.data(), kx, a, nB);
            axpy_gather(L[5], KV.data(), kx, a, nB);
        }
        for (size_t j = 0; j < rB.size(); j++) {
            double tr[6], ti[6];
            for (int m = 0; m < 6; m++) {
                const CSR &Mq = Q[static_cast<size_t>(qsel[m])];
                csr_row_dot(Mq.indices.data(), Mq.data.data(), Mq.indptr[j], Mq.indptr[j + 1],
                            reinterpret_cast<const double *>(L[m]), tr[m], ti[m]);
            }
            t[static_cast<size_t>(rA[i] * B.n_basis + rB[j])] =
                cd(((((tr[0] + tr[1]) + tr[2]) + tr[3]) + tr[4]) - tr[5],
                   ((((ti[0] + ti[1]) + ti[2]) + ti[3]) + ti[4]) - ti[5]);
        }
    }
    }
    {
        const std::ptrdiff_t nt2 = static_cast<std::ptrdiff_t>(t.size());
        const int ntc = team(be, t.size());
#if defined(_OPENMP)
        #pragma omp parallel for schedule(static) num_threads(ntc) if (ntc > 1)
#else
        (void)ntc;
#endif
        for (std::ptrdiff_t q = 0; q < nt2; q++) t[static_cast<size_t>(q)] = np_cmul(t[static_cast<size_t>(q)], c.c1);
    }
    return t;
}

// `_end_vectors` over one loop's ends against `line`: per end, (sign, fv,
// vV, vW) with vV = Fd_line (w V), vW = F_line (w tz W); the tables from
// `_end_tables`' labelled spans through the block's memo. `end_above`: the
// end in the z slot (`_above_end_args`), else in z' (`_below_end_args`).
struct EndVec {
    double sign;
    const std::vector<double> *fv;
    std::vector<cd> vV, vW;
};

std::vector<cd> real_matvec(const CSR &M, const std::vector<cd> &v) {
    std::vector<cd> out(static_cast<size_t>(M.n_rows));
    const int64_t *ip = M.indptr.data();
    const int64_t *ix = M.indices.data();
    const double *d = M.data.data();
    const double *vd = reinterpret_cast<const double *>(v.data());
    double *od = reinterpret_cast<double *>(out.data());
    for (int64_t i = 0; i < M.n_rows; i++) csr_row_dot(ix, d, ip[i], ip[i + 1], vd, od[2 * i], od[2 * i + 1]);
    return out;
}

std::vector<EndVec> end_vectors(const Ctx &c, BlockEval &be, double a_wire, const std::vector<End> &ends,
                                const Axis &line, bool end_above) {
    std::vector<EndVec> out;
    const size_t n = line.n_nodes();
    if (ends.empty()) return out;
    // `_LineZ`: the line's plane-side z, formed once (refusing at the first end).
    std::vector<double> line_z(n);
    for (size_t k = 0; k < n; k++)
        line_z[k] = on_plane_side(line.nodes[3 * k + 2] - c.gz, !end_above, "quadrature node");
    std::vector<double> w(n), w_tz(n);
    for (size_t k = 0; k < n; k++) {
        w[k] = line.w[k];
        w_tz[k] = line.w[k] * line.t[3 * k + 2];
    }
    const size_t per = std::max<size_t>(1, static_cast<size_t>(END_BATCH_PAIRS) / std::max<size_t>(1, n));
    out.resize(ends.size());
    std::vector<double> re, z, zp;
    std::vector<int64_t> lab;
    std::vector<uint32_t> te;
    for (size_t g0 = 0; g0 < ends.size(); g0 += per) {
        const size_t g1 = std::min(ends.size(), g0 + per);
        const size_t ne = g1 - g0;
        // The ends' own z, refused in order before any point is formed.
        std::vector<double> endz(ne);
        for (size_t e = g0; e < g1; e++) endz[e - g0] = on_plane_side(ends[e].pt[2] - c.gz, end_above, "end point");
        re.resize(ne * n);
        z.resize(ne * n);
        zp.resize(ne * n);
        lab.resize(ne * n);
        const std::ptrdiff_t nep = static_cast<std::ptrdiff_t>(ne);
        const int nt = team(be, ne * n);
#if defined(_OPENMP)
        #pragma omp parallel for schedule(static) num_threads(nt) if (nt > 1)
#else
        (void)nt;
#endif
        for (std::ptrdiff_t ep = 0; ep < nep; ep++) {
            const size_t e = static_cast<size_t>(ep);
            const End &E = ends[g0 + e];
            for (size_t k = 0; k < n; k++) {
                const double *q = &line.nodes[3 * k];
                const size_t i = e * n + k;
                const double rho = end_above ? std::hypot(E.pt[0] - q[0], E.pt[1] - q[1])
                                             : std::hypot(q[0] - E.pt[0], q[1] - E.pt[1]);
                re[i] = std::hypot(rho, a_wire);
                z[i] = end_above ? endz[e] : line_z[k];
                zp[i] = end_above ? line_z[k] : endz[e];
                lab[i] = static_cast<int64_t>(e);
            }
        }
        table_index(be, re, z, zp, ne > 1 ? &lab : nullptr, te);
#if defined(_OPENMP)
        #pragma omp parallel num_threads(nt) if (nt > 1)
#endif
        {
        std::vector<cd> V(n), W(n);  // each thread's, reused end to end
#if defined(_OPENMP)
        #pragma omp for schedule(dynamic)
#endif
        for (std::ptrdiff_t ep = 0; ep < nep; ep++) {
            const size_t e = static_cast<size_t>(ep);
            for (size_t k = 0; k < n; k++) {
                const Six &s = be.memo.at(te[e * n + k]);
                // `w * V`: numpy's (w + 0j)(V): each part one rounding.
                V[k] = np_cmul(cd(w[k], 0.0), s[K_V]);
                W[k] = np_cmul(cd(w_tz[k], 0.0), s[K_W]);
            }
            EndVec &ev = out[g0 + e];
            ev.sign = ends[g0 + e].sign;
            ev.fv = &ends[g0 + e].fv;
            ev.vV = real_matvec(line.Fd, V);
            ev.vW = real_matvec(line.F, W);
        }
        }
    }
    return out;
}

// `_ends_and_corner_rc(..., corner=False)` + `_EndTerms.answer(out=t)`: the
// row loop (R's ends against C's line; V then W) and the column loop (C's
// ends against R's line; W then V), each a rank-1 update into E_r / E_c, then
// scattered into t (nR x nC, row-major).
void add_end_terms(const Ctx &c, BlockEval &be, double a_wire, const Axis &R, const Axis &C,
                   bool r_above, std::vector<cd> &t) {
    const int64_t nR = R.n_basis, nC = C.n_basis;
    auto live = [](const Axis &ax) {
        std::set<int64_t> s;
        for (const End &e : ax.ends)
            for (size_t i = 0; i < e.fv.size(); i++)
                if (e.fv[i] != 0.0) s.insert(static_cast<int64_t>(i));
        return std::vector<int64_t>(s.begin(), s.end());
    };
    const std::vector<int64_t> LA = live(R), LB = live(C);
    std::vector<int64_t> posLA(static_cast<size_t>(nR), -1), posLB(static_cast<size_t>(nC), -1);
    for (size_t i = 0; i < LA.size(); i++) posLA[static_cast<size_t>(LA[i])] = static_cast<int64_t>(i);
    for (size_t i = 0; i < LB.size(); i++) posLB[static_cast<size_t>(LB[i])] = static_cast<int64_t>(i);
    std::vector<cd> E_r(LA.size() * static_cast<size_t>(nC), cd(0.0, 0.0));
    std::vector<cd> E_c(static_cast<size_t>(nR) * LB.size(), cd(0.0, 0.0));
    const cd neg_c1(-c.c1.real(), -c.c1.imag());
    // Row loop: R's ends against C's line.
    {
        const std::vector<EndVec> evs = end_vectors(c, be, a_wire, R.ends, C, r_above);
        // Columns split over the team; each entry takes the ends' rank-1
        // terms in end order, as the serial loop adds them.
        const int nt = team(be, evs.size() * static_cast<size_t>(nC));
#if defined(_OPENMP)
        #pragma omp parallel for schedule(static, 1) num_threads(nt) if (nt > 1)
#endif
        for (int ch = 0; ch < nt; ch++) {
            const int64_t j0 = nC * ch / nt, j1 = nC * (ch + 1) / nt;
            for (const EndVec &ev : evs) {
                const cd s_plus = py_prod(c.c1, cd(ev.sign, 0.0));
                const cd s_minus = py_prod(neg_c1, cd(ev.sign, 0.0));
                for (int pass = 0; pass < 2; pass++) {
                    const std::vector<cd> &vec = pass == 0 ? ev.vV : ev.vW;
                    const cd scale = pass == 0 ? s_plus : s_minus;
                    for (size_t i = 0; i < ev.fv->size(); i++) {
                        const double fvi = (*ev.fv)[i];
                        if (fvi == 0.0) continue;
                        cd *row = &E_r[static_cast<size_t>(posLA[i]) * static_cast<size_t>(nC)];
                        for (int64_t j = j0; j < j1; j++) {
                            const cd o = np_cmul(np_cmul(cd(fvi, 0.0), vec[static_cast<size_t>(j)]), scale);
                            row[j] = cd(row[j].real() + o.real(), row[j].imag() + o.imag());
                        }
                    }
                }
            }
        }
    }
    // Column loop: C's ends against R's line (W, then V).
    {
        const std::vector<EndVec> evs = end_vectors(c, be, a_wire, C.ends, R, !r_above);
        // Rows split over the team (E_r's rows LA and E_c's rows apart);
        // each entry takes the ends' terms in end order, as serially.
        const int64_t nLA = static_cast<int64_t>(LA.size());
        const int nt = team(be, evs.size() * static_cast<size_t>(nR));
#if defined(_OPENMP)
        #pragma omp parallel for schedule(static, 1) num_threads(nt) if (nt > 1)
#endif
        for (int ch = 0; ch < nt; ch++) {
            const int64_t a0 = nLA * ch / nt, a1 = nLA * (ch + 1) / nt;
            const int64_t r0 = nR * ch / nt, r1 = nR * (ch + 1) / nt;
            for (const EndVec &ev : evs) {
                const cd s_plus = py_prod(c.c1, cd(ev.sign, 0.0));
                const cd s_minus = py_prod(neg_c1, cd(ev.sign, 0.0));
                for (int pass = 0; pass < 2; pass++) {
                    const std::vector<cd> &vec = pass == 0 ? ev.vW : ev.vV;
                    const cd scale = pass == 0 ? s_minus : s_plus;
                    for (size_t jn = 0; jn < ev.fv->size(); jn++) {
                        const double fvj = (*ev.fv)[jn];
                        if (fvj == 0.0) continue;
                        // E_r's columns (rows LA), then E_c's (every row).
                        for (int64_t i = a0; i < a1; i++) {
                            const cd o = np_cmul(np_cmul(vec[static_cast<size_t>(LA[static_cast<size_t>(i)])], cd(fvj, 0.0)), scale);
                            cd &e = E_r[static_cast<size_t>(i) * static_cast<size_t>(nC) + jn];
                            e = cd(e.real() + o.real(), e.imag() + o.imag());
                        }
                        for (int64_t i = r0; i < r1; i++) {
                            const cd o = np_cmul(np_cmul(vec[static_cast<size_t>(i)], cd(fvj, 0.0)), scale);
                            cd &e = E_c[static_cast<size_t>(i) * LB.size() + static_cast<size_t>(posLB[jn])];
                            e = cd(e.real() + o.real(), e.imag() + o.imag());
                        }
                    }
                }
            }
        }
    }
    // `answer`: E_c[LA, :] = 0; dest[LA, :] += E_r; dest[:, LB] += E_c.
    for (int64_t a : LA)
        for (size_t j = 0; j < LB.size(); j++) E_c[static_cast<size_t>(a) * LB.size() + j] = cd(0.0, 0.0);
    for (size_t i = 0; i < LA.size(); i++)
        for (int64_t j = 0; j < nC; j++) {
            cd &d = t[static_cast<size_t>(LA[i] * nC + j)];
            const cd v = E_r[i * static_cast<size_t>(nC) + static_cast<size_t>(j)];
            d = cd(d.real() + v.real(), d.imag() + v.imag());
        }
    for (int64_t i = 0; i < nR; i++)
        for (size_t j = 0; j < LB.size(); j++) {
            cd &d = t[static_cast<size_t>(i * nC + LB[j])];
            const cd v = E_c[static_cast<size_t>(i) * LB.size() + j];
            d = cd(d.real() + v.real(), d.imag() + v.imag());
        }
}

int n_physical_threads() {
    const unsigned n = std::thread::hardware_concurrency();
    return n > 1 ? static_cast<int>(n / 2) : 1;
}

// `_crossing_node_charges(geom, tents, ctx, A, P, omega)`: the (n_basis,
// n_tents) addition to the crossing tents' columns, `-sgn q (fam - c1 V)` at
// each T2 endpoint of A (above rows) and P (below rows), np.add.at'ed.
std::vector<cd> node_charges(const Ctx &c, const Axis &A, const Axis &P,
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
        const Axis &ax = above ? A : P;
        const size_t E = ax.ends.size();
        if (E == 0) continue;
        const std::vector<double> &a_half = above ? a_above : a_below;
        std::vector<double> rho(E * T), R(E * T), zt(E);
        std::vector<int64_t> row(E);
        for (size_t e = 0; e < E; e++) {
            const End &en = ax.ends[e];
            for (size_t i = 0; i < en.fv.size(); i++)
                if (en.fv[i] != 0.0) {
                    row[e] = static_cast<int64_t>(i);
                    break;
                }
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

}  // namespace

void crossing_blocks(const Front &f, std::vector<cd> &z,
                     const std::vector<std::pair<int64_t, int>> &tents, uintptr_t cancel_flag) {
    const Ctx c = make_ctx(f);
    const int64_t n = f.n_basis;
    const Axis A = crossing_path_axis(f, tents, false);
    const Axis P = crossing_path_axis(f, tents, true);
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
            const Axis ax = axis_data(c, part.second, r);
            BlockEval be;
            be.ev.k_p = c.k_p;
            be.ev.k_m = c.k_m;
            be.ev.n_threads = n_physical_threads();
            be.ev.cancel_flag = cancel_flag;
            std::vector<cd> t;  // (n x n) row-major, rows = test axis
            if (src_below) {
                // `cross_complete_block(ctx_r, A, ax)`: above test rows x below
                // source columns.
                t = main_sandwich(c, be, r, A, ax);
                add_end_terms(c, be, r, A, ax, true, t);
            } else {
                // `cross_complete_block_reversed(ctx_r, P, ax)`: the main
                // sandwich over (above source Q = ax, below test P),
                // transposed; the ends with the rows below.
                const std::vector<cd> tba = main_sandwich(c, be, r, ax, P);
                t.resize(static_cast<size_t>(n * n));
                const int ntt = team(be, t.size());
#if defined(_OPENMP)
                #pragma omp parallel for schedule(static) num_threads(ntt) if (ntt > 1)
#else
                (void)ntt;
#endif
                for (int64_t i = 0; i < n; i++)
                    for (int64_t j = 0; j < n; j++) t[static_cast<size_t>(i * n + j)] = tba[static_cast<size_t>(j * n + i)];
                add_end_terms(c, be, r, P, ax, false, t);
            }
            // The fold: `Z[rows, cols] -= t`, columns apart.
            const std::ptrdiff_t ncols = static_cast<std::ptrdiff_t>(cols.size());
            const int ntf = team(be, cols.size() * rows.size());
#if defined(_OPENMP)
            #pragma omp parallel for schedule(static) num_threads(ntf) if (ntf > 1)
#else
            (void)ntf;
#endif
            for (std::ptrdiff_t cq = 0; cq < ncols; cq++) {
                const int64_t ci = cols[static_cast<size_t>(cq)];
                for (int64_t ri : rows) {
                    cd &e = z[static_cast<size_t>(ci * n + ri)];
                    const cd v = t[static_cast<size_t>(ri * n + ci)];
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

}  // namespace razor
}  // namespace mw
