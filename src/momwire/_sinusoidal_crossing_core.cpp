// The point-matched sinusoidal lane's crossing rows in C++ (momwire#1607
// B6b); see the header.

#include "_sinusoidal_crossing_core.h"

#include "_libm_complex_inline.h"

#include "_crossing_shared_inline.h"
#include "_razor_front_core.h"
#include "_somm_below_grid_core.h"

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstdint>
#include <map>
#include <thread>
#include <vector>

namespace mw {
namespace sinusoidal {

namespace {

typedef long long ll;
typedef std::complex<double> cd;
using nec4::Point3;
namespace cx = mw::crossing;

const ll kNearQ = 4;            // `_crossing_fill._NEAR_Q`
const ll kNearPanel = 4;        // `_NEAR_GX, _NEAR_GW = leggauss(4)`
const double kNearGrowth = 4.0;  // `_NEAR_GROWTH`
const double kPlaneTol = 1e-12;  // `_PLANE_TOL`
const double kMu = 1.25663706127e-6;  // SinusoidalSolver.mu

[[noreturn]] void decline(const std::string &m) { throw nec4::Declined(m); }

// numpy's fused array product, CPython's own complex arithmetic
cd xmul(cd a, cd b) { return cx::np_cmul(a, b); }
cd re(double x) { return cd(x, 0.0); }
cd xadd(cd a, cd b) { return cd(a.real() + b.real(), a.imag() + b.imag()); }
cd xsub(cd a, cd b) { return cd(a.real() - b.real(), a.imag() - b.imag()); }
cd py_prod(cd a, cd b) { return cx::py_prod(a, b); }
cd py_quot(cd a, cd b) {
    const double abs_breal = b.real() < 0 ? -b.real() : b.real();
    const double abs_bimag = b.imag() < 0 ? -b.imag() : b.imag();
    if (abs_breal >= abs_bimag) {
        if (abs_breal == 0.0) decline("a complex division by zero (the reference raises)");
        const double ratio = b.imag() / b.real();
        const double denom = b.real() + b.imag() * ratio;
        return cd((a.real() + a.imag() * ratio) / denom, (a.imag() - a.real() * ratio) / denom);
    }
    const double ratio = b.real() / b.imag();
    const double denom = b.real() * ratio + b.imag();
    return cd((a.real() * ratio + a.imag()) / denom, (a.imag() * ratio - a.real()) / denom);
}

int physical_threads() {
    const unsigned n = std::thread::hardware_concurrency();
    return n > 1 ? static_cast<int>(n / 2) : 1;
}

// A complex CSR (`csr_array`, canonical: columns ascending in a row, the
// exact zeros eliminated).
struct CCSR {
    ll n_rows = 0, n_cols = 0;
    std::vector<ll> indptr, indices;
    std::vector<cd> data;
};

// `axis_data(ctx, seg_idx)` with `SinusoidalBasisSampler`'s samples.
struct SinAxis {
    std::vector<double> nodes, t, w;  // (n, 3), (n, 3), (n)
    CCSR F, Fd;
    struct End {
        double pt[3];
        double sign;
        std::vector<cd> fv;  // (n_basis)
    };
    std::vector<End> ends;
    ll n_basis = 0;
    std::size_t n_nodes() const { return w.size(); }
};

// `SinusoidalBasisSampler._value_and_slope` of one entry at one xi:
// `_basis_value(sig AC, B, sig C, k, xi)` and `k (B cos k xi - sig C sin k xi)`
void value_and_slope(cd sigAC, cd B, cd sigC, cd k, double xi, cd &f, cd &fd) {
    const cd half = mw_libm_c::csin(xmul(xmul(re(0.5), k), re(xi)));
    const cd kx = xmul(k, re(xi));
    const cd s = mw_libm_c::csin(kx), c = mw_libm_c::ccos(kx);
    f = xsub(xadd(sigAC, xmul(B, s)), xmul(xmul(re(2.0), sigC), xmul(half, half)));
    fd = xmul(k, xsub(xmul(B, c), xmul(sigC, s)));
}

SinAxis axis_data(const Front &fr, const std::vector<ll> &seg_idx) {
    const Geometry &g = fr.geometry;
    const SegView &v = fr.basis;
    const double gz = 0.0;
    const double a_wire = fr.radius_per_wire[0];
    SinAxis ax;
    ax.n_basis = g.n_segs;
    std::vector<double> xg, wg, gx, gw;
    razor::leggauss(kNearQ, xg, wg);
    razor::leggauss(kNearPanel, gx, gw);
    std::vector<double> u_phys;
    std::vector<std::pair<ll, std::pair<ll, ll>>> runs;  // (segment, (first node, count)), axis order
    for (ll s : seg_idx) {
        const std::size_t z = static_cast<std::size_t>(s);
        const Point3 &sl = g.seg_l[z], &sr = g.seg_r[z];
        const double h = g.seg_h[z];
        const Point3 &tang = g.seg_tangents[z];
        const bool touch_lo = std::fabs(sl[2] - gz) < kPlaneTol;
        const bool touch_hi = std::fabs(sr[2] - gz) < kPlaneTol;
        std::vector<double> u, w;
        if (touch_lo || touch_hi) {
            cx::graded_u(h, !touch_lo, a_wire, kNearGrowth, gx, gw, u, w);
        } else {
            for (std::size_t k = 0; k < xg.size(); k++) {
                const double tq = 0.5 * (xg[k] + 1.0);
                u.push_back(h * tq);
                w.push_back(0.5 * h * wg[k]);
            }
        }
        runs.push_back({s, {static_cast<ll>(ax.w.size()), static_cast<ll>(u.size())}});
        for (std::size_t k = 0; k < u.size(); k++) {
            const double t = u[k] / h;
            for (int q = 0; q < 3; q++) ax.nodes.push_back(sl[static_cast<std::size_t>(q)] + t * (sr[static_cast<std::size_t>(q)] - sl[static_cast<std::size_t>(q)]));
            for (int q = 0; q < 3; q++) ax.t.push_back(tang[static_cast<std::size_t>(q)]);
            ax.w.push_back(w[k]);
            u_phys.push_back(u[k]);
        }
    }
    // the sampler's per-entry arrays
    const std::size_t ne = v.jbasis.size();
    std::vector<cd> sigAC(ne), sigC(ne);
    for (std::size_t e = 0; e < ne; e++) {
        const cd sg(static_cast<double>(v.sigma[e]), 0.0);
        sigAC[e] = xmul(sg, v.AC[e]);
        sigC[e] = xmul(sg, v.C[e]);
    }
    if (v.k_entry.size() != ne) decline("a crossing view without its entries' k");
    // `samples(seg_runs, u_phys)`: COO -> CSR (no duplicates: a segment's
    // rows are unique and segments own disjoint node runs), zeros eliminated
    const ll n_nodes = static_cast<ll>(ax.w.size());
    std::vector<std::map<ll, cd>> fr_rows(static_cast<std::size_t>(g.n_segs)), fd_rows(static_cast<std::size_t>(g.n_segs));
    for (const auto &run : runs) {
        const std::size_t s = static_cast<std::size_t>(run.first);
        const ll s0 = run.second.first, cnt = run.second.second;
        std::vector<ll> seen;
        for (ll e = v.starts[s]; e < v.starts[s + 1]; e++) seen.push_back(v.jbasis[static_cast<std::size_t>(e)]);
        std::vector<ll> uq(seen);
        std::sort(uq.begin(), uq.end());
        if (std::unique(uq.begin(), uq.end()) != uq.end()) decline("a segment carrying a basis in two entries");
        for (ll e = v.starts[s]; e < v.starts[s + 1]; e++) {
            const std::size_t ez = static_cast<std::size_t>(e);
            const ll row = v.jbasis[ez];
            for (ll k = 0; k < cnt; k++) {
                const double xi = u_phys[static_cast<std::size_t>(s0 + k)] - 0.5 * g.seg_h[s];
                cd f, fd;
                value_and_slope(sigAC[ez], v.B[ez], sigC[ez], v.k_entry[ez], xi, f, fd);
                fr_rows[static_cast<std::size_t>(row)][s0 + k] = f;
                fd_rows[static_cast<std::size_t>(row)][s0 + k] = fd;
            }
        }
    }
    for (int which = 0; which < 2; which++) {
        CCSR &M = which == 0 ? ax.F : ax.Fd;
        const std::vector<std::map<ll, cd>> &rows = which == 0 ? fr_rows : fd_rows;
        M.n_rows = g.n_segs;
        M.n_cols = n_nodes;
        M.indptr.assign(static_cast<std::size_t>(g.n_segs + 1), 0);
        for (ll r = 0; r < g.n_segs; r++) {
            for (const auto &kv : rows[static_cast<std::size_t>(r)])
                if (kv.second.real() != 0.0 || kv.second.imag() != 0.0) {  // `eliminate_zeros`
                    M.indices.push_back(kv.first);
                    M.data.push_back(kv.second);
                }
            M.indptr[static_cast<std::size_t>(r) + 1] = static_cast<ll>(M.indices.size());
        }
    }
    // the signed wire-end table: sigma = -1 at a wire's first segment's u = 0,
    // +1 at its last segment's u = h, kept where some basis is nonzero
    std::vector<bool> on_axis(static_cast<std::size_t>(g.n_segs), false);
    for (ll s : seg_idx) on_axis[static_cast<std::size_t>(s)] = true;
    for (std::size_t w = 0; w < g.wire_first.size(); w++) {
        const ll first = g.wire_first[w], last = g.wire_last[w];
        if (!on_axis[static_cast<std::size_t>(first)]) continue;
        for (int pass = 0; pass < 2; pass++) {
            const ll gs = pass == 0 ? first : last;
            const std::size_t z = static_cast<std::size_t>(gs);
            const double hh = g.seg_h[z];
            const double u = pass == 0 ? 0.0 : hh;
            SinAxis::End e;
            e.sign = pass == 0 ? -1.0 : 1.0;
            for (int q = 0; q < 3; q++)
                e.pt[q] = g.seg_l[z][static_cast<std::size_t>(q)] + (u / hh) * (g.seg_r[z][static_cast<std::size_t>(q)] - g.seg_l[z][static_cast<std::size_t>(q)]);
            e.fv.assign(static_cast<std::size_t>(g.n_segs), cd(0.0, 0.0));
            const double xi = u - 0.5 * hh;  // `float(u) - 0.5 * h`
            bool any = false;
            for (ll ee = v.starts[z]; ee < v.starts[z + 1]; ee++) {
                const std::size_t ez = static_cast<std::size_t>(ee);
                cd f, fd;
                value_and_slope(sigAC[ez], v.B[ez], sigC[ez], v.k_entry[ez], xi, f, fd);
                e.fv[static_cast<std::size_t>(v.jbasis[ez])] = f;
                any = any || f.real() != 0.0 || f.imag() != 0.0;
            }
            if (any) ax.ends.push_back(e);
        }
    }
    return ax;
}

// The (observers x points) grid's folded triples, row-major: (rho_eff, z, z')
// with z the above side's height and z' the below side's
struct Grid {
    std::vector<double> dx, dy, rho, re, z, zp;
};
Grid pair_grid(const double *P, std::size_t nA, const double *pts, std::size_t nB, bool above, double a_wire) {
    const double gz = 0.0;
    Grid gr;
    const std::size_t n = nA * nB;
    gr.dx.resize(n);
    gr.dy.resize(n);
    gr.rho.resize(n);
    gr.re.resize(n);
    gr.z.resize(n);
    gr.zp.resize(n);
    for (std::size_t i = 0; i < nA; i++)
        for (std::size_t j = 0; j < nB; j++) {
            const std::size_t q = i * nB + j;
            gr.dx[q] = P[3 * i] - pts[3 * j];
            gr.dy[q] = P[3 * i + 1] - pts[3 * j + 1];
            gr.rho[q] = std::hypot(gr.dx[q], gr.dy[q]);
            gr.re[q] = std::hypot(gr.rho[q], a_wire);  // `radius_fold`
            const double zo = P[3 * i + 2] - gz, zs = pts[3 * j + 2] - gz;
            gr.z[q] = above ? zo : zs;
            gr.zp[q] = above ? zs : zo;
        }
    return gr;
}

// The distinct folded triples of `gr` in first-appearance order (-0.0 and
// 0.0 one key, the first appearance's floats kept), and each pair's index
void unique_rows(const Grid &gr, std::vector<double> &rows, std::vector<uint32_t> &inv) {
    ni_eval::TripleIndex idx;
    const std::size_t n = gr.re.size();
    inv.resize(n);
    rows.clear();
    for (std::size_t q = 0; q < n; q++) {
        const double key[3] = {gr.re[q] + 0.0, gr.z[q] + 0.0, gr.zp[q] + 0.0};
        bool inserted = false;
        const uint32_t next = static_cast<uint32_t>(rows.size() / 3);
        const uint32_t at = std::isnan(key[0]) || std::isnan(key[1]) || std::isnan(key[2])
                                ? ni_eval::TripleIndex::NONE
                                : idx.find_or_insert(key, next, inserted);
        if (at == ni_eval::TripleIndex::NONE) decline("a NaN crossing triple");
        if (inserted) {
            rows.push_back(gr.re[q]);
            rows.push_back(gr.z[q]);
            rows.push_back(gr.zp[q]);
        }
        inv[q] = at;
    }
}

// The six tables (through the block's memo) and the point family
// (`point_tables`, memo-free) at every pair of `gr`
void kernels(cx::BlockEval &be, const Grid &gr, std::vector<ni_eval::Six> &six, std::vector<std::array<cd, 4>> &pk) {
    std::vector<double> rows;
    std::vector<uint32_t> inv;
    unique_rows(gr, rows, inv);
    const std::size_t m = rows.size() / 3;
    std::vector<double> ure(m), uz(m), uzp(m);
    for (std::size_t i = 0; i < m; i++) {
        ure[i] = rows[3 * i];
        uz[i] = rows[3 * i + 1];
        uzp[i] = rows[3 * i + 2];
    }
    std::vector<uint32_t> six_idx;
    cx::table_index(be, ure, uz, uzp, nullptr, six_idx);
    std::vector<cd> pv(m * 4);
    ni_eval::point_tables(be.ev, be.memo.have_plan ? &be.memo.plan : nullptr, rows.data(), m, pv.data());
    const std::size_t n = inv.size();
    six.resize(n);
    pk.resize(n);
    for (std::size_t q = 0; q < n; q++) {
        six[q] = be.memo.at(six_idx[inv[q]]);
        for (int k = 0; k < 4; k++) pk[q][static_cast<std::size_t>(k)] = pv[4 * inv[q] + static_cast<std::size_t>(k)];
    }
}

constexpr int kU = 0, kV = 1, kW = 2, kDzW = 3, kDzpW = 5;           // KEYS
constexpr int pRhoV = 0, pZV = 1, pRhoW = 2, pZpV = 3;               // POINT_KEYS

// `point_observer_block(ctx, obs_pts, obs_t, src, observers_above, into=G,
// into_rows=rows)` on the reference spelling (the whole grid's evaluation,
// then the ends', then the contraction row by row)
void point_observer_block(const Front &fr, const std::vector<ll> &obs_rows, const SinAxis &src, bool above,
                          std::vector<cd> &G) {
    const Geometry &g = fr.geometry;
    const Medium &md = fr.medium;
    const ll N = g.n_segs;
    const double a_wire = fr.radius_per_wire[0];
    const double k_p = md.k_p;
    const double k2sq = k_p * k_p;
    // `_c1_moment(omega, mu)`: -1j * omega * mu / (4.0 * np.pi), CPython's
    const cd c1 = py_quot(py_prod(py_prod(cd(-0.0, -1.0), cd(fr.omega, 0.0)), cd(kMu, 0.0)), cd(4.0 * M_PI, 0.0));
    cx::BlockEval be;
    be.ev.k_p = k_p;
    be.ev.k_m = md.k_m;
    be.ev.n_threads = physical_threads();
    const std::size_t nA = obs_rows.size(), nB = src.n_nodes();
    std::vector<double> P(3 * nA), A(3 * nA);
    for (std::size_t i = 0; i < nA; i++)
        for (int c = 0; c < 3; c++) {
            P[3 * i + static_cast<std::size_t>(c)] = g.seg_centers[static_cast<std::size_t>(obs_rows[i])][static_cast<std::size_t>(c)];
            A[3 * i + static_cast<std::size_t>(c)] = g.seg_tangents[static_cast<std::size_t>(obs_rows[i])][static_cast<std::size_t>(c)];
        }
    // `_plan_point_sheets`: (above nodes, below nodes), z relative to ground
    {
        std::vector<double> obs_rel(P), nodes_rel(src.nodes);
        for (std::size_t i = 0; i < nA; i++) obs_rel[3 * i + 2] -= 0.0;
        for (std::size_t j = 0; j < nB; j++) nodes_rel[3 * j + 2] -= 0.0;
        std::vector<ni_eval::NodePair> pairs;
        if (above)
            pairs.push_back({obs_rel.data(), nA, nodes_rel.data(), nB});
        else
            pairs.push_back({nodes_rel.data(), nB, obs_rel.data(), nA});
        be.memo.plan = ni_eval::sheet_plan(k_p, md.k_m, pairs, a_wire);
        be.memo.have_plan = true;
    }
    // the main block's kernels over the whole grid
    const Grid gm = pair_grid(P.data(), nA, src.nodes.data(), nB, above, a_wire);
    std::vector<ni_eval::Six> six;
    std::vector<std::array<cd, 4>> pk;
    kernels(be, gm, six, pk);
    // the ends' (`_point_kernels_dense` over the end points), after the main
    // block's evaluation: e_term = -az W - (az dz V + g dRho V)
    const std::size_t nE = src.ends.size();
    std::vector<cd> e_term(nA * nE);
    if (nE) {
        std::vector<double> e_pts;
        for (const auto &e : src.ends) e_pts.insert(e_pts.end(), e.pt, e.pt + 3);
        const Grid ge = pair_grid(P.data(), nA, e_pts.data(), nE, above, a_wire);
        std::vector<ni_eval::Six> six_e;
        std::vector<std::array<cd, 4>> pk_e;
        kernels(be, ge, six_e, pk_e);
        for (std::size_t i = 0; i < nA; i++) {
            const double ax = A[3 * i], ay = A[3 * i + 1], az = A[3 * i + 2];
            for (std::size_t j = 0; j < nE; j++) {
                const std::size_t q = i * nE + j;
                const double gch = (ax * ge.dx[q] + ay * ge.dy[q]) / ge.re[q];
                const cd dz = above ? pk_e[q][pZV] : pk_e[q][pZpV];
                const cd a_grad_v = xadd(xmul(re(az), dz), xmul(re(gch), pk_e[q][pRhoV]));
                e_term[q] = xsub(xmul(re(-az), six_e[q][kW]), a_grad_v);
            }
        }
    }
    // the contraction: tc = (F (ax U w tx)^T) + (F (ay U w ty)^T); tc = tc +
    // (F (k3 w tz)^T) + (Fd (k4 w)^T), each output scipy's `csr_matvecs`
    // row sum (per stored entry the product, then the add, from zero; the
    // product fused where SciPy's build contracts it, `MW_NUMPY_CONTRACTS`)
    std::vector<cd> X[4];
    for (int t = 0; t < 4; t++) X[t].resize(nA * nB);
    for (std::size_t i = 0; i < nA; i++) {
        const double ax = A[3 * i], ay = A[3 * i + 1], az = A[3 * i + 2];
        for (std::size_t j = 0; j < nB; j++) {
            const std::size_t q = i * nB + j;
            const double gch = (ax * gm.dx[q] + ay * gm.dy[q]) / gm.re[q];
            const ni_eval::Six &s6 = six[q];
            const std::array<cd, 4> &p4 = pk[q];
            const cd dzv = above ? p4[pZV] : p4[pZpV];
            const cd a_grad_v = xadd(xmul(re(az), dzv), xmul(re(gch), p4[pRhoV]));
            cd k3;
            if (above) {
                const cd a_grad_w = xadd(xmul(re(az), s6[kDzW]), xmul(re(gch), p4[pRhoW]));
                k3 = xsub(xmul(re(az), xadd(xmul(re(k2sq), s6[kV]), s6[kDzpW])), a_grad_w);
            } else {
                k3 = xsub(xmul(re(az * k2sq), s6[kV]), xmul(re(gch), p4[pRhoW]));
            }
            const cd k4 = xadd(xmul(re(az), s6[kW]), a_grad_v);
            const double w = src.w[j];
            const double tx = src.t[3 * j], ty = src.t[3 * j + 1], tz = src.t[3 * j + 2];
            X[0][q] = xmul(xmul(re(ax), s6[kU]), re(w * tx));
            X[1][q] = xmul(xmul(re(ay), s6[kU]), re(w * ty));
            X[2][q] = xmul(k3, re(w * tz));
            X[3][q] = xmul(k4, re(w));
        }
    }
    auto matvec = [&](const CCSR &M, const std::vector<cd> &x, ll row, std::size_t i) {
        double yr = 0.0, yi = 0.0;
        for (ll e = M.indptr[static_cast<std::size_t>(row)]; e < M.indptr[static_cast<std::size_t>(row) + 1]; e++) {
            const cd a = M.data[static_cast<std::size_t>(e)];
            const cd xv = x[i * nB + static_cast<std::size_t>(M.indices[static_cast<std::size_t>(e)])];
#if MW_NUMPY_CONTRACTS  // sparsetools' complex product, contracted (arm64)
            const double pr = mw_fma::exact(a.real(), xv.real(), -(a.imag() * xv.imag()));
            const double pi = mw_fma::exact(a.real(), xv.imag(), a.imag() * xv.real());
#else
            const double pr = a.real() * xv.real() - a.imag() * xv.imag();
            const double pi = a.real() * xv.imag() + a.imag() * xv.real();
#endif
            yr = yr + pr;
            yi = yi + pi;
        }
        return cd(yr, yi);
    };
    std::vector<cd> rc(static_cast<std::size_t>(N));
    for (std::size_t i = 0; i < nA; i++) {
        for (ll b = 0; b < N; b++) {
            const cd t0 = matvec(src.F, X[0], b, i), t1 = matvec(src.F, X[1], b, i);
            const cd t2 = matvec(src.F, X[2], b, i), t3 = matvec(src.Fd, X[3], b, i);
            rc[static_cast<std::size_t>(b)] = xadd(xadd(xadd(t0, t1), t2), t3);
        }
        // the ends' terms in end order: rc[:, nz] += (sign * e_term) * fv[nz]
        for (std::size_t j = 0; j < nE; j++) {
            const SinAxis::End &e = src.ends[j];
            const cd se = xmul(re(e.sign), e_term[i * nE + j]);
            for (std::size_t b = 0; b < e.fv.size(); b++) {
                const cd fv = e.fv[b];
                if (fv.real() == 0.0 && fv.imag() == 0.0) continue;  // `np.flatnonzero(fv)`
                rc[b] = xadd(rc[b], xmul(se, fv));
            }
        }
        cd *grow = G.data() + obs_rows[i] * N;
        for (ll b = 0; b < N; b++) {
            const cd v = xmul(rc[static_cast<std::size_t>(b)], c1);  // `rc *= c1`
            grow[b] = xadd(grow[b], v);                               // `into[rows] += rc`
        }
    }
}

}  // namespace

void crossing_point_rows(const Front &fr, std::vector<cd> &G) {
    const Geometry &g = fr.geometry;
    std::vector<ll> a_idx, b_idx;
    for (ll s = 0; s < g.n_segs; s++) (fr.below_seg[static_cast<std::size_t>(s)] ? b_idx : a_idx).push_back(s);
    const SinAxis ax_a = axis_data(fr, a_idx), ax_b = axis_data(fr, b_idx);
    point_observer_block(fr, a_idx, ax_b, true, G);
    point_observer_block(fr, b_idx, ax_a, false, G);
}

}  // namespace sinusoidal
}  // namespace mw
