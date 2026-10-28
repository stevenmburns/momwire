// The Sinusoidal-Galerkin crossing pair in C++ (momwire#1718 S11b); see the
// header. Every step names the `_crossing_fill.py` function it replays.

#if defined(_MSC_VER) && !defined(__clang__)
#pragma float_control(precise, on)
#pragma fp_contract(off)
#endif

#include "_sinusoidal_galerkin_crossing_core.h"

#include "_crossing_shared_inline.h"
#include "_crossing_split_inline.h"
#include "_fma_inline.h"
#include "_libm_complex_inline.h"
#include "_near_interface_core.h"
#include "_near_interface_eval.h"
#include "_razor_front_core.h"  // `leggauss`: `_quadrature.leggauss`'s frozen table

#include <algorithm>
#include <array>
#include <cmath>
#include <map>
#include <numeric>
#include <set>
#include <string>
#include <thread>
#include <utility>
#include <vector>

namespace mw {
namespace sinusoidal_galerkin {

namespace {

typedef long long ll;
typedef std::complex<double> cd;
namespace cx = mw::crossing;
using nec4::Point3;
using sinusoidal::Front;
using sinusoidal::Geometry;
using sinusoidal::SegView;
using Ax = cx::AxisT<cd>;
using Csr = cx::CSRT<cd>;

constexpr double PLANE_TOL = 1e-12;        // `_PLANE_TOL`
constexpr double CORNER_RTOL = 1e-10;      // `_CORNER_RTOL`
constexpr double SAME_NODE_RHO = 1e-9;     // `_SAME_NODE_RHO`
constexpr int NEAR_Q = 4;                  // `_NEAR_Q` (`_FAR_Q` the same)
constexpr double NEAR_GROWTH = 4.0;        // `_NEAR_GROWTH` (`_FAR_GROWTH` the same)
constexpr int NEAR_PANEL = 4;              // `_NEAR_GX, _NEAR_GW = leggauss(4)` (`_GX4`)
constexpr int64_t PAIR_GROUP_BUDGET = 600000;  // `_pair_groups(budget_pairs=)`
constexpr double SOLVER_MU = 1.25663706127e-6;  // SinusoidalSolver.mu
constexpr double SOLVER_EPS = 8.8541878188e-12;  // SinusoidalSolver.eps
// the corner's walk constants (`_near_interface`'s, bs2's `Corner`)
constexpr double LAM_MULT = 8.0;
constexpr int ADAPT_DEPTH = 16;
constexpr double DETOUR = 2.0;
constexpr int GAUSS_N = 24;

[[noreturn]] void decline(const std::string &m) { throw nec4::Declined(m); }

// `_on_plane_side`'s ValueError: declined by name (its sentence is not copied)
struct Raise {
    [[noreturn]] static void raise(const std::string &) {
        decline("a crossing axis on the wrong side of the plane (the reference raises)");
    }
};

inline cd xmul(cd a, cd b) { return cx::np_cmul(a, b); }  // numpy's array product
inline cd re(double x) { return cd(x, 0.0); }
inline cd xadd(cd a, cd b) { return cd(a.real() + b.real(), a.imag() + b.imag()); }
inline cd xsub(cd a, cd b) { return cd(a.real() - b.real(), a.imag() - b.imag()); }
inline bool nonzero(cd z) { return z.real() != 0.0 || z.imag() != 0.0; }

int physical_threads() {
    const unsigned n = std::thread::hardware_concurrency();
    return n > 1 ? static_cast<int>(n / 2) : 1;
}

// What the shared end terms read off the context.
struct XCtx {
    double gz = 0;
    cd c1;
};

// `_c1_moment(omega, mu)`: -1j * omega * mu / (4.0 * pi), CPython's
cd c1_moment(double omega) {
    return cx::py_quot(cx::py_prod(cx::py_prod(cd(-0.0, -1.0), cd(omega, 0.0)), cd(SOLVER_MU, 0.0)),
                       cd(4.0 * M_PI, 0.0));
}

// `SinusoidalBasisSampler._value_and_slope` of one entry at one xi:
// `_basis_value(sig AC, B, sig C, k, xi)` and `k (B cos k xi - sig C sin k xi)`
void value_and_slope(cd sigAC, cd B, cd sigC, cd k, double xi, cd &f, cd &fd) {
    const cd half = mw_libm_c::csin(xmul(xmul(re(0.5), k), re(xi)));
    const cd kx = xmul(k, re(xi));
    const cd s = mw_libm_c::csin(kx), c = mw_libm_c::ccos(kx);
    f = xsub(xadd(sigAC, xmul(B, s)), xmul(xmul(re(2.0), sigC), xmul(half, half)));
    fd = xmul(k, xsub(xmul(B, c), xmul(sigC, s)));
}

// `axis_data(ctx, seg_idx)` (or `coarse=True`, the same sampling at these
// knobs) with `SinusoidalBasisSampler`'s samples: the nodes (G4 panels
// graded toward the plane on a segment touching it, q 4 elsewhere), F / Fd
// (`samples`: COO -> CSR, zeros eliminated), `seg_rows`, the signed end table
Ax sg_axis(const Front &fr, const std::vector<ll> &seg_idx, double a_wire) {
    const Geometry &g = fr.geometry;
    const SegView &v = fr.basis;
    const double gz = 0.0;
    Ax ax;
    ax.n_basis = g.n_segs + fr.n_wings;
    std::vector<double> xg, wg, gx, gw;
    razor::leggauss(NEAR_Q, xg, wg);
    razor::leggauss(NEAR_PANEL, gx, gw);
    std::vector<double> u_phys;
    std::map<ll, std::pair<ll, ll>> runs;  // `_segment_runs`: ascending segments
    for (ll s : seg_idx) {
        const std::size_t z = static_cast<std::size_t>(s);
        const Point3 &sl = g.seg_l[z], &sr = g.seg_r[z];
        const double h = g.seg_h[z];
        const Point3 &tang = g.seg_tangents[z];
        const bool touch_lo = std::fabs(sl[2] - gz) < PLANE_TOL;
        const bool touch_hi = std::fabs(sr[2] - gz) < PLANE_TOL;
        std::vector<double> u, w;
        if (touch_lo || touch_hi) {
            cx::graded_u(h, !touch_lo, a_wire, NEAR_GROWTH, gx, gw, u, w);
        } else {
            for (std::size_t k = 0; k < xg.size(); k++) {
                const double tq = 0.5 * (xg[k] + 1.0);
                u.push_back(h * tq);
                w.push_back(0.5 * h * wg[k]);
            }
        }
        runs[s] = {static_cast<ll>(ax.w.size()), static_cast<ll>(u.size())};
        for (std::size_t k = 0; k < u.size(); k++) {
            const double t = u[k] / h;
            for (std::size_t q = 0; q < 3; q++) ax.nodes.push_back(sl[q] + t * (sr[q] - sl[q]));
            for (std::size_t q = 0; q < 3; q++) ax.t.push_back(tang[q]);
            ax.w.push_back(w[k]);
            ax.segof.push_back(s);
            u_phys.push_back(u[k]);
        }
    }
    // the sampler's per-entry arrays (`sig * AC`, `sig * C`: array products)
    const std::size_t ne = v.jbasis.size();
    if (v.k_entry.size() != ne) decline("a crossing view without its entries' k");
    std::vector<cd> sigAC(ne), sigC(ne);
    for (std::size_t e = 0; e < ne; e++) {
        const cd sg(static_cast<double>(v.sigma[e]), 0.0);
        sigAC[e] = xmul(sg, v.AC[e]);
        sigC[e] = xmul(sg, v.C[e]);
    }
    const ll n_nodes = static_cast<ll>(ax.w.size());
    const std::size_t nb = static_cast<std::size_t>(ax.n_basis);
    std::vector<std::map<ll, cd>> f_rows(nb), fd_rows(nb);
    for (const auto &run : runs) {
        const std::size_t s = static_cast<std::size_t>(run.first);
        const ll s0 = run.second.first, cnt = run.second.second;
        std::vector<ll> rows;
        for (ll e = v.starts[s]; e < v.starts[s + 1]; e++) rows.push_back(v.jbasis[static_cast<std::size_t>(e)]);
        if (rows.empty()) continue;
        std::vector<ll> uq(rows);
        std::sort(uq.begin(), uq.end());
        if (std::unique(uq.begin(), uq.end()) != uq.end()) decline("a segment carrying a basis in two entries");
        for (ll e = v.starts[s]; e < v.starts[s + 1]; e++) {
            const std::size_t ez = static_cast<std::size_t>(e);
            const std::size_t row = static_cast<std::size_t>(v.jbasis[ez]);
            for (ll k = 0; k < cnt; k++) {
                const double xi = u_phys[static_cast<std::size_t>(s0 + k)] - 0.5 * g.seg_h[s];
                cd f, fd;
                value_and_slope(sigAC[ez], v.B[ez], sigC[ez], v.k_entry[ez], xi, f, fd);
                f_rows[row][s0 + k] = f;
                fd_rows[row][s0 + k] = fd;
            }
        }
        ax.seg_rows[run.first] = std::vector<int64_t>(uq.begin(), uq.end());
    }
    for (int which = 0; which < 2; which++) {
        Csr &M = which == 0 ? ax.F : ax.Fd;
        const std::vector<std::map<ll, cd>> &rows = which == 0 ? f_rows : fd_rows;
        M.n_rows = ax.n_basis;
        M.n_cols = n_nodes;
        M.indptr.assign(nb + 1, 0);
        for (std::size_t r = 0; r < nb; r++) {
            for (const auto &kv : rows[r])
                if (nonzero(kv.second)) {  // `eliminate_zeros`
                    M.indices.push_back(kv.first);
                    M.data.push_back(kv.second);
                }
            M.indptr[r + 1] = static_cast<int64_t>(M.indices.size());
        }
    }
    // the signed wire-end table: sigma = -1 at a wire's first segment's u = 0,
    // +1 at its last segment's u = h, kept where some basis is nonzero
    std::set<ll> on_axis(seg_idx.begin(), seg_idx.end());
    for (std::size_t w = 0; w < g.wire_first.size(); w++) {
        const ll first = g.wire_first[w], last = g.wire_last[w];
        if (!on_axis.count(first)) continue;
        for (int pass = 0; pass < 2; pass++) {
            const ll gs = pass == 0 ? first : last;
            const std::size_t z = static_cast<std::size_t>(gs);
            const double hh = g.seg_h[z];
            const double u = pass == 0 ? 0.0 : hh;
            cx::EndT<cd> e;
            e.sign = pass == 0 ? -1.0 : 1.0;
            for (std::size_t q = 0; q < 3; q++) e.pt[q] = g.seg_l[z][q] + (u / hh) * (g.seg_r[z][q] - g.seg_l[z][q]);
            e.fv.assign(nb, cd(0.0, 0.0));
            const double xi = u - 0.5 * hh;  // `float(u) - 0.5 * h`
            bool any = false;
            for (ll ee = v.starts[z]; ee < v.starts[z + 1]; ee++) {
                const std::size_t ez = static_cast<std::size_t>(ee);
                cd f, fd;
                value_and_slope(sigAC[ez], v.B[ez], sigC[ez], v.k_entry[ez], xi, f, fd);
                e.fv[static_cast<std::size_t>(v.jbasis[ez])] = f;
            }
            for (const cd &x : e.fv) any = any || x != 0.0;  // `np.any(fv != 0.0)`
            if (any) ax.ends.push_back(std::move(e));
        }
    }
    return ax;
}

// `_axis_segment_tree`: boxes are the segments' endpoint extents
std::vector<cx::Cluster> segment_tree(const Geometry &g, const std::vector<ll> &seg_idx) {
    std::vector<double> lo(3 * seg_idx.size()), hi(3 * seg_idx.size());
    for (std::size_t k = 0; k < seg_idx.size(); k++)
        for (std::size_t q = 0; q < 3; q++) {
            const double l = g.seg_l[static_cast<std::size_t>(seg_idx[k])][q],
                         r = g.seg_r[static_cast<std::size_t>(seg_idx[k])][q];
            lo[3 * k + q] = std::min(l, r);
            hi[3 * k + q] = std::max(l, r);
        }
    std::vector<int64_t> all(seg_idx.size());
    std::iota(all.begin(), all.end(), 0);
    std::vector<cx::Cluster> out;
    cx::build_tree(out, all, lo, hi, cx::CLUSTER_LEAF);
    return out;
}

// `_in_rows(rows, idx)`: per entry of `idx`, its position in the sorted
// `rows`, or -1
std::vector<ll> in_rows(const std::vector<ll> &rows, const std::vector<int64_t> &idx) {
    std::vector<ll> pos(idx.size(), -1);
    for (std::size_t i = 0; i < idx.size(); i++) {
        auto it = std::lower_bound(rows.begin(), rows.end(), static_cast<ll>(idx[i]));
        if (it != rows.end() && *it == idx[i]) pos[i] = static_cast<ll>(it - rows.begin());
    }
    return pos;
}

// One `(rows, n_cols)` / `(n_rows, cols)` pair, C order: `_main_split(rows=)`'s
// and `_EndTerms(rows=)`'s answer
struct RowsPair {
    ll nR = 0, nC = 0, nr = 0;  // block (nR x nC), |rows|
    std::vector<cd> t_r, t_c;   // (nr, nC), (nR, nr)
    void init(ll R, ll C, ll r) {
        nR = R;
        nC = C;
        nr = r;
        t_r.assign(static_cast<std::size_t>(r * C), cd(0.0, 0.0));
        t_c.assign(static_cast<std::size_t>(R * r), cd(0.0, 0.0));
    }
    cd &r_at(ll i, ll j) { return t_r[static_cast<std::size_t>(i * nC + j)]; }
    cd &c_at(ll i, ll j) { return t_c[static_cast<std::size_t>(i * nr + j)]; }
};

// `_sandwich_dense(A, B, iA, iB, K, k2sq, out=, rows=, out_cols=)` on the
// complex weights: the six left products `P_i @ K_x` (scipy's `csr_matvecs`,
// from zero, entry by entry), the five-term `_combine` (each `L @ Q.T` as
// scipy's `(Q @ L.T).T`: per Q row its stored entries in order, from zero),
// summed in the reference order, added into the row and column halves.
void sandwich_rows(const Ax &A, const Ax &B, const cx::Spec &sp, const uint32_t *K, const std::vector<cd> &KU,
                   const std::vector<cd> &KX, const std::vector<cd> &KW, const std::vector<cd> &KV,
                   const std::vector<ll> &rows, RowsPair &out) {
    const std::size_t nB = sp.iB.size();
    const std::vector<int64_t> rA = cx::support_rows_of(A, sp.iA), rB = cx::support_rows_of(B, sp.iB);
    if (rA.empty() || rB.empty()) return;
    const std::array<Csr, 4> P = cx::row_weights_of(A, sp.iA, rA), Q = cx::row_weights_of(B, sp.iB, rB);
    // L[m] (|rA|, nB) C order
    std::array<std::vector<cd>, 6> L;
    for (auto &x : L) x.assign(rA.size() * nB, cd(0.0, 0.0));
    struct Prod {
        int pm;
        const std::vector<cd> *tab;
    };
    const Prod prods[6] = {{0, &KU}, {1, &KU}, {2, &KX}, {2, &KW}, {3, &KW}, {3, &KV}};
    for (int m = 0; m < 6; m++) {
        const Csr &M = P[static_cast<std::size_t>(prods[m].pm)];
        const cd *tab = prods[m].tab->data();
        for (std::size_t i = 0; i < rA.size(); i++) {
            cd *y = &L[static_cast<std::size_t>(m)][i * nB];
            for (int64_t jj = M.indptr[i]; jj < M.indptr[i + 1]; jj++) {
                const cd a = M.data[static_cast<std::size_t>(jj)];
                const uint32_t *kx = &K[static_cast<std::size_t>(M.indices[static_cast<std::size_t>(jj)]) * nB];
                for (std::size_t b = 0; b < nB; b++) y[b] = cx::axpy_w(y[b], a, tab[kx[b]]);
            }
        }
    }
    static const int qsel[6] = {0, 1, 2, 3, 2, 3};
    auto combine = [&](std::size_t i, std::size_t j) {
        cd t[6];
        for (int m = 0; m < 6; m++) {
            const Csr &Mq = Q[static_cast<std::size_t>(qsel[m])];
            const cd *x = &L[static_cast<std::size_t>(m)][i * nB];
            cd y(0.0, 0.0);
            for (int64_t jj = Mq.indptr[j]; jj < Mq.indptr[j + 1]; jj++)
                y = cx::axpy_w(y, Mq.data[static_cast<std::size_t>(jj)], x[Mq.indices[static_cast<std::size_t>(jj)]]);
            t[m] = y;
        }
        return xsub(xadd(xadd(xadd(xadd(t[0], t[1]), t[2]), t[3]), t[4]), t[5]);
    };
    const std::vector<ll> pos = in_rows(rows, rA), posc = in_rows(rows, rB);
    for (std::size_t i = 0; i < rA.size(); i++) {
        if (pos[i] < 0) continue;
        for (std::size_t j = 0; j < rB.size(); j++) {
            cd &e = out.r_at(pos[i], rB[j]);
            e = xadd(e, combine(i, j));
        }
    }
    for (std::size_t i = 0; i < rA.size(); i++)
        for (std::size_t j = 0; j < rB.size(); j++) {
            if (posc[j] < 0) continue;
            cd &e = out.c_at(rA[i], posc[j]);
            e = xadd(e, combine(i, j));
        }
}

// One far block in the restricted shape (`_main_split`'s ACA branch, `rows`):
// the shared factorisation, then `P @ Uf` and `Vf @ Q.T` (scipy's complex
// products), the row half `left[sel] @ right` and the column half `left @
// right[:, selc]` (numpy's matmul of exactly those shapes), seven terms each.
void far_block_rows(const nec5::NumpyLinalg &np, const Ax &A, const Ax &B, const cx::Spec &sp, cx::BlockEval &be,
                    double gz, double a_wire, double k2sq, const std::vector<ll> &rows, RowsPair &out) {
    const int64_t n = static_cast<int64_t>(sp.iB.size());
    std::array<std::vector<cd>, 4> Uf, Vf;
    std::array<int64_t, 4> rk{};
    cx::aca_factors(np, A, B, sp, be, gz, a_wire, Uf, Vf, rk);
    const std::vector<int64_t> rA = cx::support_rows_of(A, sp.iA), rB = cx::support_rows_of(B, sp.iB);
    const std::array<Csr, 4> Pw = cx::row_weights_of(A, sp.iA, rA), Qw = cx::row_weights_of(B, sp.iB, rB);
    const int64_t nrA = static_cast<int64_t>(rA.size()), nrB = static_cast<int64_t>(rB.size());
    std::map<std::pair<int, int>, std::vector<cd>> PU, VQ;  // PU (|rA|, r); VQ (|rB|, r): right's transpose
    auto factors = [&](int pi, int q, int qi) {
        const int64_t r = rk[static_cast<std::size_t>(q)];
        const auto pk = std::make_pair(pi, q);
        if (!PU.count(pk)) {
            std::vector<cd> y(static_cast<std::size_t>(nrA * r), cd(0.0, 0.0));
            const Csr &M = Pw[static_cast<std::size_t>(pi)];
            for (int64_t i = 0; i < nrA; i++)
                for (int64_t jj = M.indptr[static_cast<std::size_t>(i)]; jj < M.indptr[static_cast<std::size_t>(i) + 1]; jj++) {
                    const cd d = M.data[static_cast<std::size_t>(jj)];
                    const cd *x = &Uf[static_cast<std::size_t>(q)][static_cast<std::size_t>(M.indices[static_cast<std::size_t>(jj)] * r)];
                    for (int64_t k = 0; k < r; k++) {
                        cd &e = y[static_cast<std::size_t>(i * r + k)];
                        e = cx::axpy_w(e, d, x[k]);
                    }
                }
            PU[pk] = std::move(y);
        }
        const auto vk = std::make_pair(q, qi);
        if (!VQ.count(vk)) {
            std::vector<cd> y(static_cast<std::size_t>(nrB * r), cd(0.0, 0.0));
            const Csr &M = Qw[static_cast<std::size_t>(qi)];
            for (int64_t j = 0; j < nrB; j++)
                for (int64_t jj = M.indptr[static_cast<std::size_t>(j)]; jj < M.indptr[static_cast<std::size_t>(j) + 1]; jj++) {
                    const cd d = M.data[static_cast<std::size_t>(jj)];
                    const int64_t c = M.indices[static_cast<std::size_t>(jj)];
                    for (int64_t k = 0; k < r; k++) {
                        cd &e = y[static_cast<std::size_t>(j * r + k)];
                        e = cx::axpy_w(e, d, Vf[static_cast<std::size_t>(q)][static_cast<std::size_t>(k * n + c)]);
                    }
                }
            VQ[vk] = std::move(y);
        }
    };
    const std::vector<ll> pos = in_rows(rows, rA), posc = in_rows(rows, rB);
    std::vector<int64_t> sel, selc;
    for (int64_t i = 0; i < nrA; i++)
        if (pos[static_cast<std::size_t>(i)] >= 0) sel.push_back(i);
    for (int64_t j = 0; j < nrB; j++)
        if (posc[static_cast<std::size_t>(j)] >= 0) selc.push_back(j);
    // `_terms(rsel=)` / `_terms(csel=)`: U, U, k2 V, dzpW, W, W, -V
    static const int TERMS[7][3] = {{0, 0, 0}, {1, 0, 1}, {2, 1, 2}, {2, 3, 2}, {2, 2, 3}, {3, 2, 2}, {3, 1, 3}};
    auto terms = [&](bool row_half) {
        const int64_t m_out = row_half ? static_cast<int64_t>(sel.size()) : nrA;
        const int64_t p_out = row_half ? nrB : static_cast<int64_t>(selc.size());
        std::vector<cd> s(static_cast<std::size_t>(m_out * p_out));
        for (int t = 0; t < 7; t++) {
            const int pi = TERMS[t][0], q = TERMS[t][1], qi = TERMS[t][2];
            factors(pi, q, qi);
            const int64_t r = rk[static_cast<std::size_t>(q)];
            const std::vector<cd> &PUv = PU[std::make_pair(pi, q)], &VQv = VQ[std::make_pair(q, qi)];
            std::vector<cd> prod(static_cast<std::size_t>(m_out * p_out));
            if (row_half) {
                std::vector<cd> left(static_cast<std::size_t>(m_out * r));  // `left[rsel]`, C order
                for (int64_t a = 0; a < m_out; a++)
                    std::copy(PUv.begin() + sel[static_cast<std::size_t>(a)] * r,
                              PUv.begin() + (sel[static_cast<std::size_t>(a)] + 1) * r,
                              left.begin() + a * r);
                // right = VQ.T, F order (r, |rB|)
                nec5::replay::np_matmul(np, nec5::Op{left.data(), m_out, r, r, 1}, nec5::Op{VQv.data(), r, nrB, 1, r},
                                        prod.data(), p_out, 1);
            } else {
                // `right[:, csel]`: the selected columns of the F-ordered view,
                // F order (r, |selc|)
                std::vector<cd> right(static_cast<std::size_t>(r * p_out));
                for (int64_t c = 0; c < p_out; c++)
                    std::copy(VQv.begin() + selc[static_cast<std::size_t>(c)] * r,
                              VQv.begin() + (selc[static_cast<std::size_t>(c)] + 1) * r, right.begin() + c * r);
                nec5::replay::np_matmul(np, nec5::Op{PUv.data(), m_out, r, r, 1}, nec5::Op{right.data(), r, p_out, 1, r},
                                        prod.data(), p_out, 1);
            }
            for (std::size_t e = 0; e < s.size(); e++) {
                if (t == 0) {
                    s[e] = prod[e];
                } else if (t == 2) {
                    s[e] = xadd(s[e], xmul(re(k2sq), prod[e]));
                } else if (t == 6) {
                    s[e] = xsub(s[e], prod[e]);
                } else {
                    s[e] = xadd(s[e], prod[e]);
                }
            }
        }
        return s;
    };
    if (!sel.empty()) {
        const std::vector<cd> s = terms(true);
        for (std::size_t a = 0; a < sel.size(); a++)
            for (int64_t j = 0; j < nrB; j++) {
                cd &e = out.r_at(pos[static_cast<std::size_t>(sel[a])], rB[static_cast<std::size_t>(j)]);
                e = xadd(e, s[a * static_cast<std::size_t>(nrB) + static_cast<std::size_t>(j)]);
            }
    }
    if (!selc.empty()) {
        const std::vector<cd> s = terms(false);
        for (int64_t i = 0; i < nrA; i++)
            for (std::size_t c = 0; c < selc.size(); c++) {
                cd &e = out.c_at(rA[static_cast<std::size_t>(i)], posc[static_cast<std::size_t>(selc[c])]);
                e = xadd(e, s[static_cast<std::size_t>(i) * selc.size() + c]);
            }
    }
}

// `_main_split(ctx, a_idx, b_idx, A, B, ..., rows=)`: `(c1 t[rows, :], c1 t[:,
// rows])`. `Ac` / `Bc` are the coarse axes (the dense ones at the block's
// radius, rebuilt at another).
RowsPair main_split_rows(const Geometry &g, const Ax &A, const Ax &B, const Ax &Ac, const Ax &Bc,
                         const std::vector<ll> &a_idx, const std::vector<ll> &b_idx, cx::BlockEval &be, double gz,
                         double k_p, cd c1, double a_wire, const nec5::NumpyLinalg &np, const std::vector<ll> &rows) {
    const std::vector<cx::Cluster> TA = segment_tree(g, a_idx), TB = segment_tree(g, b_idx);
    const std::vector<int64_t> ai(a_idx.begin(), a_idx.end()), bi(b_idx.begin(), b_idx.end());
    std::vector<cx::Spec> direct, far_aca;
    cx::split_specs(TA, TB, A, B, Ac, Bc, ai, bi, a_wire, direct, far_aca);
    cx::plan_sheets(be, direct, A, B, Ac, Bc, gz, k_p, a_wire);
    RowsPair out;
    out.init(A.n_basis, B.n_basis, static_cast<ll>(rows.size()));
    const double k2sq = k_p * k_p;
    cx::direct_groups(be, direct, A, B, Ac, Bc, gz, a_wire, PAIR_GROUP_BUDGET,
                      [&](std::size_t start, std::size_t stop, const std::vector<uint32_t> &K) {
                          const std::vector<cd> KU = cx::key_plane(be.memo, cx::K_U),
                                                KV = cx::key_plane(be.memo, cx::K_V),
                                                KW = cx::key_plane(be.memo, cx::K_W),
                                                KD = cx::key_plane(be.memo, cx::K_DZPW);
                          // `k2sq * K["V"] + K["dzpW"]`: numpy's product, then the sum
                          std::vector<cd> KX(KV.size());
                          for (std::size_t j = 0; j < KX.size(); j++) KX[j] = xadd(xmul(re(k2sq), KV[j]), KD[j]);
                          std::size_t off = 0;
                          for (std::size_t k = start; k < stop; k++) {
                              sandwich_rows(direct[k].coarse ? Ac : A, direct[k].coarse ? Bc : B, direct[k], &K[off],
                                            KU, KX, KW, KV, rows, out);
                              off += direct[k].iA.size() * direct[k].iB.size();
                          }
                      });
    if (!far_aca.empty() && (np.ddot == nullptr || np.zdotc_sub == nullptr))
        decline("no BLAS for the crossing ACA");
    for (const cx::Spec &sp : far_aca) far_block_rows(np, Ac, Bc, sp, be, gz, a_wire, k2sq, rows, out);
    for (cd &x : out.t_r) x = xmul(c1, x);
    for (cd &x : out.t_c) x = xmul(c1, x);
    return out;
}

// `_corner_v(cache, eps_t, k_p, a_wire, rho)`: V at the regularized end
// separation, once per distinct rho (`six_point_served` at `_CORNER_RTOL`)
struct CornerV {
    double k_p, a_wire;
    cd k_m;
    std::map<double, cd> v_at;
    cd operator()(double rho) {
        const double key = rho < SAME_NODE_RHO ? 0.0 : rho;
        auto it = v_at.find(key);
        if (it != v_at.end()) return it->second;
        std::vector<double> gx, gw;
        razor::leggauss(GAUSS_N, gx, gw);
        const double r_eff = key == 0.0 ? a_wire : std::hypot(rho, a_wire);
        const double z0 = 0.0, zp0 = 0.0;
        std::complex<double> six[6];
        ni_core::six_batch(k_p, k_m, &r_eff, &z0, &zp0, 1, CORNER_RTOL, LAM_MULT, ADAPT_DEPTH, DETOUR, gx.data(),
                           gw.data(), GAUSS_N, six);
        v_at[key] = six[1];
        return six[1];
    }
};

// `_rank1_add` into a half: `t[pos] += (multiply.outer(a, b)) * scale`
// (numpy's products, the outer's operands in its order)
std::vector<ll> nonzeros(const std::vector<cd> &fv) {
    std::vector<ll> nz;
    for (std::size_t i = 0; i < fv.size(); i++)
        if (fv[i] != 0.0) nz.push_back(static_cast<ll>(i));
    return nz;
}

// `_ends_and_corner(ctx, A, B, ..., rows=)`: the end terms and the corner of
// the forward block (rows above, columns below) in the restricted shape,
// `(t[rows, :], t[:, rows])` from zero: the row loop (A's ends against B's
// line, V then W), the column loop (B's ends against A's line, W then V),
// then the corner, each `_EndTerms` write in its order.
RowsPair ends_rows(const XCtx &xc, cx::BlockEval &be, double a_wire, const Ax &R, const Ax &C, bool corner,
                   bool row_ends, bool col_ends, const std::vector<ll> &rows, double k_p, cd k_m) {
    RowsPair T;
    T.init(R.n_basis, C.n_basis, static_cast<ll>(rows.size()));
    const ll nr = static_cast<ll>(rows.size());
    const cd neg_c1(-xc.c1.real(), -xc.c1.imag());
    auto pos_of = [&](ll i) {
        auto it = std::lower_bound(rows.begin(), rows.end(), i);
        return it != rows.end() && *it == i ? static_cast<ll>(it - rows.begin()) : -1;
    };
    // `add_rows(nz, fv[nz], vec, scale)`
    auto add_rows = [&](const std::vector<ll> &nz, const std::vector<cd> &fv, const std::vector<cd> &vec, cd scale) {
        for (ll i : nz) {
            const ll p = pos_of(i);
            if (p < 0) continue;
            const cd a = fv[static_cast<std::size_t>(i)];
            for (ll j = 0; j < T.nC; j++) {
                cd &e = T.r_at(p, j);
                e = xadd(e, xmul(xmul(a, vec[static_cast<std::size_t>(j)]), scale));
            }
        }
        for (ll i : nz) {
            const cd a = fv[static_cast<std::size_t>(i)];
            for (ll k = 0; k < nr; k++) {
                cd &e = T.c_at(i, k);
                e = xadd(e, xmul(xmul(a, vec[static_cast<std::size_t>(rows[static_cast<std::size_t>(k)])]), scale));
            }
        }
    };
    // `add_cols(nz, vec, fv[nz], scale)`
    auto add_cols = [&](const std::vector<ll> &nz, const std::vector<cd> &vec, const std::vector<cd> &fv, cd scale) {
        for (ll k = 0; k < nr; k++) {
            const cd a = vec[static_cast<std::size_t>(rows[static_cast<std::size_t>(k)])];
            for (ll j : nz) {
                cd &e = T.r_at(k, j);
                e = xadd(e, xmul(xmul(a, fv[static_cast<std::size_t>(j)]), scale));
            }
        }
        std::vector<std::pair<ll, ll>> sel;  // (nz entry, its position in rows)
        for (ll j : nz) {
            const ll p = pos_of(j);
            if (p >= 0) sel.push_back({j, p});
        }
        if (sel.empty()) return;
        for (ll i = 0; i < T.nR; i++) {
            const cd a = vec[static_cast<std::size_t>(i)];
            for (const auto &jp : sel) {
                cd &e = T.c_at(i, jp.second);
                e = xadd(e, xmul(xmul(a, fv[static_cast<std::size_t>(jp.first)]), scale));
            }
        }
    };
    {
        const auto evs = row_ends ? cx::end_vectors<Raise>(xc, be, a_wire, R.ends, C, true)
                                  : std::vector<cx::EndVecT<cd>>();
        for (const auto &ev : evs) {
            const std::vector<ll> nz = nonzeros(*ev.fv);
            add_rows(nz, *ev.fv, ev.vV, cx::py_prod(xc.c1, cd(ev.sign, 0.0)));
            add_rows(nz, *ev.fv, ev.vW, cx::py_prod(neg_c1, cd(ev.sign, 0.0)));
        }
    }
    {
        const auto evs = col_ends ? cx::end_vectors<Raise>(xc, be, a_wire, C.ends, R, false)
                                  : std::vector<cx::EndVecT<cd>>();
        for (const auto &ev : evs) {
            const std::vector<ll> nz = nonzeros(*ev.fv);
            add_cols(nz, ev.vW, *ev.fv, cx::py_prod(neg_c1, cd(ev.sign, 0.0)));
            add_cols(nz, ev.vV, *ev.fv, cx::py_prod(xc.c1, cd(ev.sign, 0.0)));
        }
    }
    if (corner) {
        // `_corner_terms`: every in-plane end pair, `scale * np.outer(fva, fvb)`
        CornerV cv{k_p, a_wire, k_m, {}};
        for (const auto &ea : R.ends) {
            if (std::fabs(ea.pt[2] - xc.gz) > 1e-12) continue;
            for (const auto &eb : C.ends) {
                if (std::fabs(eb.pt[2] - xc.gz) > 1e-12) continue;
                const double rho = std::hypot(ea.pt[0] - eb.pt[0], ea.pt[1] - eb.pt[1]);
                const cd v = cv(rho);
                const cd scale = cx::py_prod(cx::py_prod(cd((-ea.sign) * eb.sign, 0.0), xc.c1), v);
                const std::vector<ll> nza = nonzeros(ea.fv), nzb = nonzeros(eb.fv);
                for (ll i : nza) {
                    const ll p = pos_of(i);
                    if (p < 0) continue;
                    for (ll j : nzb) {
                        cd &e = T.r_at(p, j);
                        e = xadd(e, xmul(scale, xmul(ea.fv[static_cast<std::size_t>(i)], eb.fv[static_cast<std::size_t>(j)])));
                    }
                }
                std::vector<std::pair<ll, ll>> selb;
                for (ll j : nzb) {
                    const ll p = pos_of(j);
                    if (p >= 0) selb.push_back({j, p});
                }
                for (ll i : nza)
                    for (const auto &jp : selb) {
                        cd &e = T.c_at(i, jp.second);
                        e = xadd(e, xmul(scale, xmul(ea.fv[static_cast<std::size_t>(i)], eb.fv[static_cast<std::size_t>(jp.first)])));
                    }
            }
        }
    }
    return T;
}

void add_into(std::vector<cd> &dst, const std::vector<cd> &src) {
    for (std::size_t q = 0; q < dst.size(); q++) dst[q] = xadd(dst[q], src[q]);
}

cx::BlockEval block_eval(const sinusoidal::Medium &md) {
    cx::BlockEval be;
    be.ev.k_p = md.k_p;
    be.ev.k_m = md.k_m;
    be.ev.n_threads = physical_threads();
    return be;
}

// The deck's crossing context: the class segments, the axes at `a_wire`
// (the smaller radius at a two-radius node), the support rows
struct Ctx {
    std::vector<ll> a_idx, b_idx, rows;
    double a_wire = 0;
    Ax A, B;
};

Ctx context(const Front &fr) {
    const Geometry &g = fr.geometry;
    Ctx c;
    for (ll s = 0; s < g.n_segs; s++) (fr.below_seg[static_cast<std::size_t>(s)] ? c.b_idx : c.a_idx).push_back(s);
    if (c.a_idx.empty() || c.b_idx.empty()) decline("a crossing deck with one side empty");
    c.a_wire = fr.two_radius ? std::min(fr.a_above, fr.a_below) : fr.radius_per_wire[0];
    c.A = sg_axis(fr, c.a_idx, c.a_wire);
    c.B = sg_axis(fr, c.b_idx, c.a_wire);
    // `rows = np.unique(jbasis[~below[seg_of_entry]])`
    std::set<ll> rs;
    const SegView &v = fr.basis;
    for (ll s : c.a_idx)
        for (ll e = v.starts[static_cast<std::size_t>(s)]; e < v.starts[static_cast<std::size_t>(s) + 1]; e++)
            rs.insert(v.jbasis[static_cast<std::size_t>(e)]);
    c.rows.assign(rs.begin(), rs.end());
    return c;
}

}  // namespace

CrossingRows crossing_rows(const Front &fr, const nec5::NumpyLinalg &np) {
    if (!fr.crossing) throw std::logic_error("crossing_rows on a deck with no crossing junction");
    const Geometry &g = fr.geometry;
    const sinusoidal::Medium &md = fr.medium;
    const double gz = 0.0;
    const Ctx c = context(fr);
    const cd c1 = c1_moment(fr.omega);
    const XCtx xc{gz, c1};
    CrossingRows out;
    out.rows = c.rows;
    if (!fr.two_radius) {
        // `cross_complete_block_split(ctx_x, a_idx, b_idx, ax_a, ax_b, rows=)`:
        // one memo; the coarse axes are the dense ones; main, then ends
        cx::BlockEval be = block_eval(md);
        RowsPair main = main_split_rows(g, c.A, c.B, c.A, c.B, c.a_idx, c.b_idx, be, gz, md.k_p, c1, c.a_wire, np,
                                        c.rows);
        const RowsPair ends = ends_rows(xc, be, c.a_wire, c.A, c.B, true, true, true, c.rows, md.k_p, md.k_m);
        add_into(main.t_r, ends.t_r);
        out.t_rows = main.t_r;
        out.t_cols = main.t_r;
        return out;
    }
    // `cross_complete_blocks_two_radius(..., below_half="rows")`
    {
        // `t_below`: `cross_complete_block_split(ctx_below, ..., rows=)[0]`,
        // its own memo, its coarse axes at `a_below`
        cx::BlockEval be = block_eval(md);
        const double a = fr.a_below;
        const Ax Ac = sg_axis(fr, c.a_idx, a), Bc = sg_axis(fr, c.b_idx, a);
        RowsPair main = main_split_rows(g, c.A, c.B, Ac, Bc, c.a_idx, c.b_idx, be, gz, md.k_p, c1, a, np, c.rows);
        const RowsPair ends = ends_rows(xc, be, a, c.A, c.B, true, true, true, c.rows, md.k_p, md.k_m);
        add_into(main.t_r, ends.t_r);
        out.t_cols = main.t_r;
    }
    {
        // `t_above`: the main split at `a_above` through the block's memo,
        // the source ends at `a_above` (no corner), then the test ends and
        // the corner at `a_below` through a fresh memo
        cx::BlockEval be = block_eval(md);
        const double a = fr.a_above;
        const Ax Ac = sg_axis(fr, c.a_idx, a), Bc = sg_axis(fr, c.b_idx, a);
        RowsPair main = main_split_rows(g, c.A, c.B, Ac, Bc, c.a_idx, c.b_idx, be, gz, md.k_p, c1, a, np, c.rows);
        const RowsPair e1 = ends_rows(xc, be, a, c.A, c.B, false, false, true, c.rows, md.k_p, md.k_m);
        add_into(main.t_r, e1.t_r);
        cx::BlockEval be2 = block_eval(md);
        const RowsPair e2 = ends_rows(xc, be2, fr.a_below, c.A, c.B, true, true, false, c.rows, md.k_p, md.k_m);
        add_into(main.t_r, e2.t_r);
        out.t_rows = main.t_r;
    }
    return out;
}

void crossing_completions(const Front &fr, const nec5::NumpyLinalg &np, std::vector<cd> &G) {
    const Ctx c = context(fr);
    const ll n = c.A.n_basis;
    if (G.size() != static_cast<std::size_t>(n * n)) throw std::logic_error("crossing_completions: G's shape");
    for (cd &x : G) x = cd(-x.real(), -x.imag());  // `np.negative(G, out=G)`
    if (fr.two_radius)
        cx::self_completions(G, 1, n, c.B, c.A, fr.medium, c.a_wire, 0.0, fr.omega, SOLVER_EPS, np, fr.a_above,
                             fr.a_below);
    else
        cx::self_completions(G, 1, n, c.B, c.A, fr.medium, c.a_wire, 0.0, fr.omega, SOLVER_EPS, np);
    for (cd &x : G) x = cd(-x.real(), -x.imag());
}

}  // namespace sinusoidal_galerkin
}  // namespace mw
