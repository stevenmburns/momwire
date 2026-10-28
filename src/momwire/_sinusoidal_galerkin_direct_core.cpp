// The Sinusoidal-Galerkin direct-field fill (see
// `_sinusoidal_galerkin_direct_core.h`). Each function names the Python it
// copies (`sinusoidal_galerkin.py`, and `sinusoidal.py` for the field
// kernels it inherits); the far fill is the S1 core's.
//
// numpy's arithmetic is written in numpy's order (measured on this numpy,
// as `_sinusoidal_galerkin_fill_core.cpp` records): a complex product is
// `np_cmul` in the written operand order with a float operand promoted to
// (x, 0); a complex quotient `np_cdiv`; a float plus a complex (x + re,
// 0.0 + im); `np.exp` of a complex libm's cexp, and `np.sin` / `cos` /
// `sqrt` / `arcsinh` / `sinh` / `cosh` / `log1p` libm's under the AVX-512
// pins; `einsum("...d,...d->...")` over three (x0 y0 + x2 y2) + x1 y1 and
// `.sum(-1)` over three (p0 + p1) + p2; a two-operand complex (or real by
// complex) einsum reduction a sequential sum from zero of unfused complex
// products. A Python-scalar expression (`1j * k`, `-1j * eta / (4 pi k)`)
// is CPython's `_Py_c_prod` / `_Py_c_quot`. scipy's sparse products are
// `csr_matvecs`: each output from zero plus its stored entries in order,
// one unfused complex multiply-add each. No product here depends on numpy's
// temporary elision (the contact kernel's that did are spelled out of it in
// Python, momwire#1718 S7).

#if defined(_MSC_VER) && !defined(__clang__)
#pragma float_control(precise, on)
#pragma fp_contract(off)
#endif

#include "_sinusoidal_galerkin_direct_core.h"

#include "_libm_complex_inline.h"  // `rsin` / `rcos`: sin and cos as two calls (momwire#1733)
#include "_razor_front_core.h"  // `leggauss`, `max_image_distance`
#include "_razor_exact_core.h"  // `coaxial_labels`
#include "_sinusoidal_galerkin_core.h"
#include "_sinusoidal_series_inline.h"
#include "_somm_grid_core.h"
#include "_somm_proj_core.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <memory>
#include <numeric>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

namespace mw {
namespace sinusoidal_galerkin {

namespace {

typedef long long ll;
typedef std::complex<double> cd;
namespace rp = nec5::replay;
using nec4::Declined;
using sinusoidal::Front;
using sinusoidal::Geometry;
using sinusoidal::SegView;
using CF = nd::array_t<double, nd::array::c_style | nd::array::forcecast>;
using CI = nd::array_t<int64_t, nd::array::c_style | nd::array::forcecast>;
using CC = nd::array_t<cd, nd::array::c_style | nd::array::forcecast>;
using V = std::vector<double>;
using VC = std::vector<cd>;
using VI = std::vector<ll>;

[[noreturn]] void decline(const std::string &m) { throw Declined(m); }

// --- numpy's and CPython's scalar arithmetic ---------------------------------

inline cd cm(cd a, cd b) { return rp::np_cmul(a, b); }
inline cd rc(double x, cd b) { return cm(cd(x, 0.0), b); }  // float * complex
inline cd cr(cd a, double x) { return cm(a, cd(x, 0.0)); }  // complex * float
inline cd cdr(cd a, double x) { return rp::np_cdiv(a, cd(x, 0.0)); }
inline cd cadd(cd a, cd b) { return cd(a.real() + b.real(), a.imag() + b.imag()); }
inline cd csub(cd a, cd b) { return cd(a.real() - b.real(), a.imag() - b.imag()); }
inline cd radd(double x, cd b) { return cd(x + b.real(), 0.0 + b.imag()); }   // float + complex
inline cd rsub(double x, cd b) { return cd(x - b.real(), 0.0 - b.imag()); }   // float - complex
inline cd csubr(cd a, double x) { return cd(a.real() - x, a.imag() - 0.0); }  // complex - float
inline cd cneg(cd a) { return cd(-a.real(), -a.imag()); }
// numpy's einsum / scipy's complex_wrapper product: never fused
inline cd e_mul(cd a, cd b) {
    return cd(a.real() * b.real() - a.imag() * b.imag(), a.real() * b.imag() + a.imag() * b.real());
}
inline void acc_add(cd p, double &ar, double &ai) {
    ar = p.real() + ar;
    ai = p.imag() + ai;
}

// CPython 3.12's `_Py_c_prod` / `_Py_c_quot`
cd py_mul(cd a, cd b) {
    return cd(a.real() * b.real() - a.imag() * b.imag(), a.real() * b.imag() + a.imag() * b.real());
}
cd py_div(cd a, cd b) {
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

const cd kJ(0.0, 1.0);         // 1j
const cd kMinusJ(-0.0, -1.0);  // -1j: unary minus of 1j
const double kPi = 3.141592653589793;

// numpy's NaN-propagating maximum / minimum
inline double npmax(double a, double b) { return (a != a) ? a : (b != b) ? b : (a >= b ? a : b); }
inline double npmin(double a, double b) { return (a != a) ? a : (b != b) ? b : (a <= b ? a : b); }

inline double norm3(const double *v) { return std::sqrt((v[0] * v[0] + v[1] * v[1]) + v[2] * v[2]); }
// einsum("...d,...d->...") over three
inline double edot3(const double *a, const double *b) { return (a[0] * b[0] + a[2] * b[2]) + a[1] * b[1]; }
// (a * b).sum(axis=-1) over three
inline double sdot3(const double *a, const double *b) { return (a[0] * b[0] + a[1] * b[1]) + a[2] * b[2]; }

// CPython's float floor division `a // b` (`float_floor_div`), b > 0
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

void leggauss(ll n, V &x, V &w) {
    if (!razor::leggauss(n, x, w)) decline("an untabled Gauss order");
}

// --- constants -------------------------------------------------------------

constexpr ll kNQpTest = 8;            // SinusoidalGalerkinSolver(n_qp_test=8)
constexpr ll kNQpNear = 8;            // n_qp_near=8
constexpr double kNearFactor = 0.5;   // near_factor=0.5
constexpr ll kNQpConst = 8;           // SinusoidalSolver(n_qp_const=8)
constexpr ll kNQpSommerfeld = 3;      // n_qp_sommerfeld=3
constexpr ll kSweptMemMb = 256;       // swept_mem_mb=256
constexpr ll kDenseThreshold = 60;    // _DENSE_ASSEMBLY_THRESHOLD
constexpr ll kBandMinSegments = 64;   // _BAND_MIN_SEGMENTS
constexpr ll kNQpEkDelta = 16;        // _N_QP_EK_DELTA
constexpr ll kNPanelEkNear = 8;       // _N_PANEL_EK_DELTA_NEAR
constexpr ll kPairBlock = 512;        // _PAIR_BLOCK
constexpr ll kEkBracketBandBytes = ll(1) << 25;  // _EK_BRACKET_BAND_BYTES
constexpr ll kRemainderChunkElems = ll(1) << 19;  // _REMAINDER_CHUNK_ELEMS
constexpr double kMu = 1.25663706127e-6;          // SinusoidalSolver.mu
constexpr double kEps0 = 8.8541878188e-12;        // SinusoidalSolver.eps

// --- arrays over std containers ----------------------------------------------

nd::Buffer buffer(const void *p, std::size_t itemsize, std::initializer_list<ll> shape, bool writeable = false) {
    nd::Buffer b;
    b.ptr = const_cast<void *>(p);
    b.ndim = static_cast<int>(shape.size());
    int k = 0;
    for (ll s : shape) b.shape[k++] = static_cast<nd::ssize_t>(s);
    nd::ssize_t acc = static_cast<nd::ssize_t>(itemsize);
    for (int d = b.ndim - 1; d >= 0; d--) {
        b.strides[d] = acc;
        acc *= b.shape[d];
    }
    b.writeable = writeable;
    return b;
}
CF f64(const double *p, std::initializer_list<ll> shape) { return CF::borrow(buffer(p, sizeof(double), shape)); }
CI i64(const int64_t *p, std::initializer_list<ll> shape) { return CI::borrow(buffer(p, sizeof(int64_t), shape)); }
CC c128(const cd *p, std::initializer_list<ll> shape) { return CC::borrow(buffer(p, sizeof(cd), shape)); }

// `_basis_value(sigAC, B, sigC, k, xi)` at one entry
inline cd basis_value(cd sigAC, cd B, cd sigC, double k, double xi) {
    const double half = std::sin(0.5 * k * xi);
    const double s = std::sin(k * xi);
    return csub(cadd(sigAC, cr(B, s)), cr(rc(2.0, sigC), half * half));
}

// --- the deck and the test context --------------------------------------------

struct Deck {
    ll N = 0, n_basis = 0, nnz = 0;
    double k = 0, eta = 0;
    V c, t, h, hh, a;  // (N, 3), (N, 3), (N), (N), per-segment radius
    bool uniform = true;
    double a_uniform = 0;
    // the CSR view
    VI starts, m_of_entry, i_of_entry;
    VC B, sigC, sigAC;
};

// `_test_context(geom, seg_view, k)`
struct Ctx {
    ll nq = 0;
    V gx, gw;
    V obs_c, obs_t, a_obs;  // (N nq, 3), (N nq, 3), (N nq)
    VC w_entry;             // (nnz, nq)
};

Deck deck_of(const Front &fr, const SegView &v, ll n_basis) {
    const Geometry &g = fr.geometry;
    Deck d;
    d.N = g.n_segs;
    d.n_basis = n_basis;
    d.k = fr.k;
    d.eta = fr.eta;
    const std::size_t N = static_cast<std::size_t>(d.N);
    d.c.resize(3 * N);
    d.t.resize(3 * N);
    d.h = g.seg_h;
    d.hh.resize(N);
    for (std::size_t i = 0; i < N; i++) {
        for (int q = 0; q < 3; q++) {
            d.c[3 * i + q] = g.seg_centers[i][static_cast<std::size_t>(q)];
            d.t[3 * i + q] = g.seg_tangents[i][static_cast<std::size_t>(q)];
        }
        d.hh[i] = 0.5 * g.seg_h[i];
    }
    d.uniform = fr.uniform_radius;
    d.a_uniform = fr.radius_per_wire.empty() ? 0.0 : fr.radius_per_wire[0];
    d.a.resize(N);
    for (std::size_t w = 0; w < g.wire_first.size(); w++)
        for (ll i = g.wire_first[w]; i <= g.wire_last[w]; i++) d.a[static_cast<std::size_t>(i)] = fr.radius_per_wire[w];
    d.starts.assign(v.starts.begin(), v.starts.end());
    if (static_cast<ll>(d.starts.size()) != d.N + 1) decline("a basis view whose CSR is not over the segments");
    d.nnz = d.starts.back();
    d.i_of_entry.assign(v.jbasis.begin(), v.jbasis.end());
    d.m_of_entry.resize(static_cast<std::size_t>(d.nnz));
    for (ll m = 0; m < d.N; m++)
        for (ll e = d.starts[static_cast<std::size_t>(m)]; e < d.starts[static_cast<std::size_t>(m) + 1]; e++)
            d.m_of_entry[static_cast<std::size_t>(e)] = m;
    d.B = v.B;
    d.sigC.resize(static_cast<std::size_t>(d.nnz));
    d.sigAC.resize(static_cast<std::size_t>(d.nnz));
    for (ll e = 0; e < d.nnz; e++) {
        const std::size_t ez = static_cast<std::size_t>(e);
        const cd sig(static_cast<double>(v.sigma[ez]), 0.0);
        d.sigC[ez] = cm(sig, v.C[ez]);
        d.sigAC[ez] = cm(sig, v.AC[ez]);
    }
    return d;
}

Ctx test_context(const Deck &d) {
    Ctx x;
    leggauss(kNQpTest, x.gx, x.gw);
    x.nq = static_cast<ll>(x.gx.size());
    const std::size_t N = static_cast<std::size_t>(d.N), nq = static_cast<std::size_t>(x.nq);
    x.obs_c.resize(N * nq * 3);
    x.obs_t.resize(N * nq * 3);
    x.a_obs.resize(N * nq);
    for (std::size_t m = 0; m < N; m++)
        for (std::size_t q = 0; q < nq; q++) {
            const double xi = d.hh[m] * x.gx[q];
            for (int c = 0; c < 3; c++) {
                x.obs_c[(m * nq + q) * 3 + c] = d.c[3 * m + c] + xi * d.t[3 * m + c];
                x.obs_t[(m * nq + q) * 3 + c] = d.t[3 * m + c];
            }
            x.a_obs[m * nq + q] = d.uniform ? d.a_uniform : d.a[m];
        }
    x.w_entry.resize(static_cast<std::size_t>(d.nnz) * nq);
    for (ll e = 0; e < d.nnz; e++) {
        const std::size_t ez = static_cast<std::size_t>(e);
        const std::size_t m = static_cast<std::size_t>(d.m_of_entry[ez]);
        for (std::size_t q = 0; q < nq; q++) {
            const double xi = d.hh[m] * x.gx[q];
            const cd fval = basis_value(d.sigAC[ez], d.B[ez], d.sigC[ez], d.k, xi);
            x.w_entry[ez * nq + q] = rc(x.gw[q] * d.hh[m], fval);
        }
    }
    return x;
}

// `_coincident_segments(seg_l, seg_r)`: the first two segments sharing both
// endpoints (nanometre keys, then 1e-12 of the coordinate scale), or none.
bool coincident_segments(const Front &fr, ll &m_out, ll &n_out) {
    const Geometry &g = fr.geometry;
    const std::size_t N = static_cast<std::size_t>(g.n_segs);
    if (N < 2) return false;
    std::vector<std::array<int64_t, 6>> rows(N);
    double amax = 0.0, bmax = 0.0;
    bool first = true;
    for (std::size_t i = 0; i < N; i++) {
        int64_t ka[3], kb[3];
        for (int c = 0; c < 3; c++) {
            const double av = g.seg_l[i][static_cast<std::size_t>(c)], bv = g.seg_r[i][static_cast<std::size_t>(c)];
            ka[c] = static_cast<int64_t>(std::nearbyint(av * 1e9));
            kb[c] = static_cast<int64_t>(std::nearbyint(bv * 1e9));
            const double fa = std::fabs(av), fb = std::fabs(bv);
            if (first) {
                amax = fa;
                bmax = fb;
                first = false;
            } else {
                amax = npmax(amax, fa);
                bmax = npmax(bmax, fb);
            }
        }
        const bool eq0 = ka[0] == kb[0], eq1 = ka[1] == kb[1];
        const bool swap = (ka[0] > kb[0]) || (eq0 && ka[1] > kb[1]) || (eq0 && eq1 && ka[2] > kb[2]);
        for (int c = 0; c < 3; c++) {
            rows[i][static_cast<std::size_t>(c)] = swap ? kb[c] : ka[c];
            rows[i][static_cast<std::size_t>(c) + 3] = swap ? ka[c] : kb[c];
        }
    }
    std::vector<std::size_t> order(N);
    std::iota(order.begin(), order.end(), std::size_t(0));
    std::stable_sort(order.begin(), order.end(), [&](std::size_t x, std::size_t y) { return rows[x] < rows[y]; });
    const double tol = 1e-12 * std::max(1.0, std::max(amax, bmax));
    auto dmax = [&](const nec4::Point3 &p, const nec4::Point3 &q) {
        double r = std::fabs(p[0] - q[0]);
        r = npmax(r, std::fabs(p[1] - q[1]));
        return npmax(r, std::fabs(p[2] - q[2]));
    };
    for (std::size_t s = 0; s < N;) {
        std::size_t e = s + 1;
        while (e < N && rows[order[e]] == rows[order[s]]) e++;
        if (e - s > 1) {
            std::vector<std::size_t> members(order.begin() + static_cast<std::ptrdiff_t>(s),
                                             order.begin() + static_cast<std::ptrdiff_t>(e));
            std::sort(members.begin(), members.end());
            const std::size_t m = members[0];
            for (std::size_t q = 1; q < members.size(); q++) {
                const std::size_t n = members[q];
                const double same = std::max(dmax(g.seg_l[m], g.seg_l[n]), dmax(g.seg_r[m], g.seg_r[n]));
                const double flip = std::max(dmax(g.seg_l[m], g.seg_r[n]), dmax(g.seg_r[m], g.seg_l[n]));
                if (std::min(same, flip) <= tol) {
                    m_out = static_cast<ll>(m);
                    n_out = static_cast<ll>(n);
                    return true;
                }
            }
        }
        s = e;
    }
    return false;
}

// --- the extended kernel's labels (`_ek_axis_labels`, `_ek_reduced_ends`) -----

struct Labels {
    VI obs, src;          // (N): the test half and the source half
    std::vector<bool> bad_lo, bad_hi;
};

Labels ek_labels(const Front &fr, const Deck &d, bool mirror) {
    const std::size_t N = static_cast<std::size_t>(d.N);
    // seg_l / seg_r rebuilt from the centres: c -/+ hh t
    V l(3 * N), r(3 * N);
    for (std::size_t i = 0; i < N; i++)
        for (int c = 0; c < 3; c++) {
            const double ht = d.hh[i] * d.t[3 * i + c];
            l[3 * i + c] = d.c[3 * i + c] - ht;
            r[3 * i + c] = d.c[3 * i + c] + ht;
        }
    Labels L;
    if (!mirror) {
        const std::vector<int64_t> lab = razor::coaxial_labels(l, r, d.t, d.a);
        L.obs.assign(lab.begin(), lab.end());
        L.src = L.obs;
    } else {
        V jl = l, jr = r, jt = d.t, ja = d.a;
        for (std::size_t i = 0; i < N; i++) {
            jl.insert(jl.end(), {l[3 * i], l[3 * i + 1], 2.0 * 0.0 - l[3 * i + 2]});
            jr.insert(jr.end(), {r[3 * i], r[3 * i + 1], 2.0 * 0.0 - r[3 * i + 2]});
            jt.insert(jt.end(), {d.t[3 * i] * 1.0, d.t[3 * i + 1] * 1.0, d.t[3 * i + 2] * -1.0});
        }
        ja.insert(ja.end(), d.a.begin(), d.a.end());
        const std::vector<int64_t> lab = razor::coaxial_labels(jl, jr, jt, ja);
        L.obs.assign(lab.begin(), lab.begin() + static_cast<std::ptrdiff_t>(N));
        L.src.assign(lab.begin() + static_cast<std::ptrdiff_t>(N), lab.end());
    }
    // `_ek_reduced_ends`: K >= 3 at the node, or a member of another label
    const Geometry &g = fr.geometry;
    for (int side = 0; side < 2; side++) {
        const std::vector<ll> &count = side == 0 ? g.nm_count : g.np_count;
        const std::vector<ll> &basis = side == 0 ? g.nm_basis : g.np_basis;
        const std::vector<ll> &seg = side == 0 ? g.nm_seg : g.np_seg;
        std::vector<bool> bad(N);
        for (std::size_t i = 0; i < N; i++) bad[i] = count[i] >= 2 || L.src[i] < 0;
        for (std::size_t e = 0; e < basis.size(); e++)
            if (L.src[static_cast<std::size_t>(basis[e])] != L.src[static_cast<std::size_t>(seg[e])])
                bad[static_cast<std::size_t>(basis[e])] = true;
        (side == 0 ? L.bad_lo : L.bad_hi) = bad;
    }
    return L;
}

// --- `_near_pairs(geom, src_c, src_t)` ---------------------------------------

void near_pairs(const Deck &d, const V &sc, const V &st, VI &mm, VI &nn) {
    const std::size_t N = static_cast<std::size_t>(d.N);
    const double thresh = 1.0 + kNearFactor;
    VI cm_, cn_;
    for (std::size_t i = 0; i < N; i++)
        for (std::size_t j = 0; j < N; j++) {
            const double dv[3] = {d.c[3 * i] - sc[3 * j], d.c[3 * i + 1] - sc[3 * j + 1], d.c[3 * i + 2] - sc[3 * j + 2]};
            const double dc = norm3(dv);  // scipy's cdist: ((d0^2 + d1^2) + d2^2)
            if (dc <= thresh * (d.hh[i] + d.hh[j])) {
                cm_.push_back(static_cast<ll>(i));
                cn_.push_back(static_cast<ll>(j));
            }
        }
    static const double s[5] = {-1.0, -0.5, 0.0, 0.5, 1.0};  // np.linspace(-1, 1, 5)
    // `_gap(ai, bi, ca, ta, cb, tb)`
    auto gap = [&](std::size_t ai, std::size_t bi, const double *ca, const double *ta, const double *cb,
                   const double *tb) {
        double best = 0.0;
        for (int q = 0; q < 5; q++) {
            const double sh = d.hh[ai] * s[q];
            double dd[3];
            for (int c = 0; c < 3; c++) dd[c] = (ca[c] + sh * ta[c]) - cb[c];
            double u = edot3(dd, tb);
            u = npmin(npmax(u, -d.hh[bi]), d.hh[bi]);
            double perp[3];
            for (int c = 0; c < 3; c++) perp[c] = dd[c] - u * tb[c];
            const double nv = norm3(perp);
            best = q == 0 ? nv : npmin(best, nv);
        }
        return best;
    };
    mm.clear();
    nn.clear();
    for (std::size_t p = 0; p < cm_.size(); p++) {
        const std::size_t m = static_cast<std::size_t>(cm_[p]), n = static_cast<std::size_t>(cn_[p]);
        const double g1 = gap(m, n, &d.c[3 * m], &d.t[3 * m], &sc[3 * n], &st[3 * n]);
        const double g2 = gap(n, m, &sc[3 * n], &st[3 * n], &d.c[3 * m], &d.t[3 * m]);
        const double gp = npmin(g1, g2);
        if (gp <= kNearFactor * (d.hh[m] + d.hh[n])) {
            mm.push_back(cm_[p]);
            nn.push_back(cn_[p]);
        }
    }
}

// `_graded_endpoint_rule(eps, n_per_panel, leggauss)` on [-1, 1]
void graded_endpoint_rule(double eps, ll n_per_panel, V &x, V &w) {
    eps = npmin(npmax(eps, 1e-9), 0.5);
    V left{-1.0};
    for (int q = 0; q < 64; q++) {
        const double dv = eps * std::pow(2.0, static_cast<double>(q));
        if (dv <= 0.5) left.push_back(-1.0 + dv);
    }
    left.push_back(0.0);
    V all = left;
    for (auto it = left.rbegin(); it != left.rend(); ++it) all.push_back(-*it);
    std::sort(all.begin(), all.end());
    V edges;
    for (double v : all)
        if (edges.empty() || !(v == edges.back())) edges.push_back(v);
    V gx, gw;
    leggauss(n_per_panel, gx, gw);
    x.clear();
    w.clear();
    for (std::size_t p = 0; p + 1 < edges.size(); p++) {
        const double lo = edges[p], hi = edges[p + 1];
        const double mid = 0.5 * (lo + hi), half = 0.5 * (hi - lo);
        for (std::size_t q = 0; q < gx.size(); q++) {
            x.push_back(mid + half * gx[q]);
            w.push_back(half * gw[q]);
        }
    }
}

double min_a_over_hh(const Deck &d) {
    double m = 0.0;
    for (std::size_t i = 0; i < d.a.size(); i++) m = i == 0 ? d.a[i] / d.hh[i] : npmin(m, d.a[i] / d.hh[i]);
    return m;
}

// --- `_field_components_bcast(cos_shape="cos-1")`, one (observer, source) ----

struct KConst {
    double k = 0, eta = 0;
    cd jk, mjk, c_e, pz, prc;  // 1j k, -1j k, -1j eta, pref_z, pref_rho_const
    double four_pi_k = 0;
    V gx, gw;            // n_qp_const
    double sw_hi = 0, sw_lo = 0;  // gw[gx >= 0].sum(), gw[gx < 0].sum()
    V ek_gx, ek_gw;      // `_ek_delta_rule(16, 8)`
};

KConst kconst(double k, double eta) {
    KConst K;
    K.k = k;
    K.eta = eta;
    K.jk = py_mul(kJ, cd(k, 0.0));
    K.mjk = py_mul(kMinusJ, cd(k, 0.0));
    K.c_e = py_mul(kMinusJ, cd(eta, 0.0));
    K.four_pi_k = 4.0 * kPi * k;
    K.pz = py_div(py_mul(kJ, cd(eta, 0.0)), cd(K.four_pi_k, 0.0));
    K.prc = py_div(K.c_e, cd(K.four_pi_k, 0.0));
    leggauss(kNQpConst, K.gx, K.gw);
    V hi, lo;
    for (std::size_t q = 0; q < K.gx.size(); q++) (K.gx[q] >= 0.0 ? hi : lo).push_back(K.gw[q]);
    K.sw_hi = rp::np_pairwise(hi.data(), static_cast<int64_t>(hi.size()));
    K.sw_lo = rp::np_pairwise(lo.data(), static_cast<int64_t>(lo.size()));
    V gx, gw;
    leggauss(kNQpEkDelta, gx, gw);
    const ll np_ = kNPanelEkNear;
    for (ll p = 0; p < np_; p++) {
        // np.linspace(-1, 1, n_panels + 1): step * i + start, the last exact
        const double step = 2.0 / static_cast<double>(np_);
        const double e0 = static_cast<double>(p) * step + -1.0;
        const double e1 = p + 1 == np_ ? 1.0 : static_cast<double>(p + 1) * step + -1.0;
        const double mid = 0.5 * (e0 + e1), half = 0.5 * (e1 - e0);
        for (std::size_t q = 0; q < gx.size(); q++) {
            K.ek_gx.push_back(mid + half * gx[q]);
            K.ek_gw.push_back(half * gw[q]);
        }
    }
    return K;
}

// `_stable.expm1_neg_j(w)`: re - 1j * im
inline cd expm1_neg_j(double w) {
    const double half = std::sin(0.5 * w);
    const double re = -2.0 * half * half;
    const double im = std::sin(w);
    return rsub(re, cm(kJ, cd(im, 0.0)));
}
// `_stable.expm1_neg_j_from_half(half)`: re + 1j * im
inline cd expm1_neg_j_from_half(double half) {
    const double h = mw_libm_c::rsin(half), c = mw_libm_c::rcos(half);
    const double re = -2.0 * h * h;
    const double im = 2.0 * h * c;
    return radd(re, cm(kJ, cd(im, 0.0)));
}
// `_stable.asinh_diff(u0, u1, rho2, r0, r1)`
inline double asinh_diff(double u0, double u1, double rho2, double r0, double r1) {
    const double p0 = u0 >= 0.0 ? u0 + r0 : rho2 / (r0 + std::fabs(u0));
    const double p1 = u1 >= 0.0 ? u1 + r1 : rho2 / (r1 + std::fabs(u1));
    const double du = u1 - u0;
    const double num = du * (p1 + p0);
    return std::log1p(num / ((r1 + r0) * p0));
}
// `_sin_minus_arg(u)`, `_asinh_minus_arg(x)`
inline double sin_minus(double u) { return sin_minus_arg(u); }
inline double asinh_minus(double x) { return asinh_minus_arg_from_t(std::asinh(x)); }

struct PairGeom {
    double z, rho, rho_vec[3], td, rpf;
};

// `_pair_geometry(obs_c, obs_t, a, src_c, src_t)`
inline PairGeom pair_geometry(const double *o, const double *to, double a, const double *sc, const double *st) {
    PairGeom p;
    const double rv[3] = {o[0] - sc[0], o[1] - sc[1], o[2] - sc[2]};
    p.z = edot3(rv, st);
    for (int c = 0; c < 3; c++) p.rho_vec[c] = rv[c] - p.z * st[c];
    const double rho_axis = norm3(p.rho_vec);
    p.rho = std::sqrt(rho_axis * rho_axis + a * a);
    p.td = sdot3(to, st);
    const double rdt = sdot3(p.rho_vec, to);
    p.rpf = rdt / p.rho;
    return p;
}

struct F6 {
    cd erc, ezc, ers, ezs, ero, ezo;  // Erho / Ez of the const, sin and cos-1 shapes
};

// The reduced fields on the folded shape set: `_field_components_bcast`
// below its EK branch, with `_folded_cos_fields` for the third shape.
F6 reduced_cos1(const KConst &K, double z, double rho, double H) {
    const double k = K.k;
    const double dz2 = z - H, dz1 = z + H;
    const double r2 = std::sqrt(rho * rho + dz2 * dz2);
    const double r1 = std::sqrt(rho * rho + dz1 * dz1);
    const cd G2 = cdr(std::exp(cm(K.mjk, cd(r2, 0.0))), r2);
    const cd G1 = cdr(std::exp(cm(K.mjk, cd(r1, 0.0))), r1);
    const cd pref_rho = rp::np_cdiv(K.c_e, cd(K.four_pi_k * rho, 0.0));
    const cd c12 = radd(1.0, cm(K.jk, cd(r2, 0.0)));  // 1.0 + 1j k r0_2
    const cd c11 = radd(1.0, cm(K.jk, cd(r1, 0.0)));
    F6 f;
    // Erho_const
    f.erc = cm(K.prc, csub(cdr(cm(cr(c12, rho), G2), r2 * r2), cdr(cm(cr(c11, rho), G1), r1 * r1)));
    // int_G0
    const double int_inv_r0 = asinh_diff(-dz1, -dz2, rho * rho, r1, r2);
    const std::size_t nq = K.gx.size();
    double ar = 0.0, ai = 0.0;
    double dz_qp[16], r_qp[16];
    for (std::size_t q = 0; q < nq; q++) {
        const double z_qp = H * K.gx[q];
        dz_qp[q] = z - z_qp;
        r_qp[q] = std::sqrt(rho * rho + dz_qp[q] * dz_qp[q]);
        const cd reg = cdr(expm1_neg_j_from_half((-0.5 * k) * r_qp[q]), r_qp[q]);
        acc_add(e_mul(reg, cd(K.gw[q], 0.0)), ar, ai);
    }
    const cd int_G0 = radd(int_inv_r0, cr(cd(ar, ai), H));
    const cd ezb = csub(cdr(cm(cr(c12, dz2), G2), r2 * r2), cdr(cm(cr(c11, dz1), G1), r1 * r1));
    f.ezc = cm(cneg(K.pz), cadd(ezb, rc(k * k, int_G0)));
    // sine
    const double sin2 = mw_libm_c::rsin(k * H), cos2 = mw_libm_c::rcos(k * H);
    const double sin1 = mw_libm_c::rsin(-k * H), cos1 = mw_libm_c::rcos(-k * H);
    const cd b_sin_2 = radd((k * dz2) * cos2, cr(rsub(1.0, cdr(rc(dz2 * dz2, c12), r2 * r2)), sin2));
    const cd b_sin_1 = radd((k * dz1) * cos1, cr(rsub(1.0, cdr(rc(dz1 * dz1, c11), r1 * r1)), sin1));
    const cd d_sin = csub(cm(G2, b_sin_2), cm(G1, b_sin_1));
    f.ers = cm(pref_rho, d_sin);
    const cd b_sin_z_2 = rsub(k * cos2, cr(cdr(cr(c12, dz2), r2 * r2), sin2));
    const cd b_sin_z_1 = rsub(k * cos1, cr(cdr(cr(c11, dz1), r1 * r1), sin1));
    f.ezs = cm(K.pz, csub(cm(G2, b_sin_z_2), cm(G1, b_sin_z_1)));
    // `_folded_cos_fields`
    const double kH = k * H;
    const double sh = std::sin(0.5 * kH);
    const double cm1 = -2.0 * (sh * sh);
    const cd T2 = cdr(cm(c12, G2), r2 * r2);
    const cd T1 = cdr(cm(c11, G1), r1 * r1);
    const bool outside = dz1 * dz2 >= 0.0;
    const double s_den = outside ? dz1 * r2 + dz2 * r1 : 1.0;
    const double X = outside ? ((2.0 * H) * (dz1 + dz2)) / s_den : (dz1 * r2 - dz2 * r1) / (rho * rho);
    const double t_sing = std::fabs(X) < 1.0
                              ? asinh_minus(X) + ((((H * rho) * rho) * X) * X) / (((r1 + r2) * r1) * r2)
                              : std::asinh(X) - H * (1.0 / r1 + 1.0 / r2);
    const cd g1 = cdr(expm1_neg_j(k * r1), r1);
    const cd g2 = cdr(expm1_neg_j(k * r2), r2);
    const cd e2 = cr(G2, r2), e1 = cr(G1, r1);
    double mr = 0.0, mi = 0.0;
    for (std::size_t q = 0; q < nq; q++) {
        const bool near2 = K.gx[q] >= 0.0;
        const double r_ref = near2 ? r2 : r1;
        const double dz_ref = near2 ? dz2 : dz1;
        const cd g_ref = near2 ? g2 : g1;
        const cd e_ref = near2 ? e2 : e1;
        const double d_step = near2 ? H * (1.0 - K.gx[q]) : (-H) * (1.0 + K.gx[q]);
        const double delta = (d_step * (dz_qp[q] + dz_ref)) / (r_qp[q] + r_ref);
        const cd dg_phase = expm1_neg_j(k * delta);
        const cd dg_num = csub(cm(e_ref, dg_phase), cr(g_ref, delta));
        const cd dg = cdr(dg_num, r_qp[q]);
        acc_add(e_mul(dg, cd(K.gw[q], 0.0)), mr, mi);
    }
    const cd m_reg = cadd(cadd(cd(mr, mi), rc(K.sw_hi - 1.0, g2)), rc(K.sw_lo - 1.0, g1));
    const cd d_int = csub(radd(t_sing, rc(H, m_reg)), rc(sin_minus(kH) / k, cadd(G1, G2)));
    f.ezo = cm(K.pz, csub(rc(k * k, d_int), rc(cm1, csub(rc(dz2, T2), rc(dz1, T1)))));
    const double dr = ((4.0 * H) * z) / (r1 + r2);
    const double phi = (0.5 * k) * dr;
    const double A = kH + phi, Bv = kH - phi;
    const double d_lin = ((((-8.0 * std::pow(H, 3.0)) * z) * rho) * rho) /
                         ((((rho * rho + dz1 * dz2) + r1 * r2) * (r1 + r2)) * r1 * r2);
    const double cph = mw_libm_c::rcos(phi), sph = mw_libm_c::rsin(phi);
    const double w_even = (A * sin_minus(Bv) - Bv * sin_minus(A)) / kH + ((d_lin / H) * sin2) * cph;
    const cd w_odd = cr(cr(cm(kJ, cd(sin2, 0.0)), -((rho * rho) * X) / (r1 * r2)), sph);
    const cd W = cm(cm(cr(G2, r2), rsub(cph, cm(kJ, cd(sph, 0.0)))), radd(w_even, w_odd));
    const cd e_rho_brk = cadd(rc(-k, W), rc((rho * rho) * cm1, csub(T2, T1)));
    f.ero = cm(pref_rho, e_rho_brk);
    return f;
}

// `_folded_ek_delta_fields(k, H, z, rho, src_a, "cos-1", n_panels=8)`
F6 ek_delta_cos1(const KConst &K, double z, double rho, double H, double src_a) {
    const double k = K.k;
    const double t_lo = std::asinh((z - H) / rho), t_hi = std::asinh((z + H) / rho);
    const double t_mid = 0.5 * (t_hi + t_lo), t_half = 0.5 * (t_hi - t_lo);
    const cd gain = cm(K.prc, cd(src_a * src_a, 0.0));
    const double rho2 = rho * rho;
    double s[6][2] = {{0, 0}, {0, 0}, {0, 0}, {0, 0}, {0, 0}, {0, 0}};  // Ez/Erho x const/sin/cos
    for (std::size_t q = 0; q < K.ek_gx.size(); q++) {
        const double t = t_mid + t_half * K.ek_gx[q];
        const double cosh_t = std::cosh(t);
        const double R = rho * cosh_t;
        const double zeta = rho * std::sinh(t);
        const double xi = z - zeta;
        const double w = (t_half * K.ek_gw[q]) * (rho * cosh_t);
        const double r2 = R * R;
        const cd x = cm(kJ, cd(k * R, 0.0));
        const cd x2 = cm(x, x), x3 = cm(x2, x), x4 = cm(x2, x2);
        const cd a1 = radd(1.0, x);
        const cd a2 = cadd(radd(3.0, rc(3.0, x)), x2);
        const cd a3 = cadd(cadd(radd(15.0, rc(15.0, x)), rc(6.0, x2)), x3);
        const cd a4 = cadd(cadd(cadd(radd(105.0, rc(105.0, x)), rc(45.0, x2)), rc(10.0, x3)), x4);
        const double inv2 = 1.0 / r2;
        const cd phase = std::exp(cm(kMinusJ, cd(k * R, 0.0)));
        const cd base = cdr(phase, R);
        const cd g1 = rc(-0.5, cr(cm(base, a1), inv2));
        const cd g2 = rc(0.25, cr(cm(base, a2), inv2 * inv2));
        const cd g3 = rc(-0.125, cr(cm(base, a3), (inv2 * inv2) * inv2));
        const cd g4 = rc(0.0625, cr(cm(base, a4), ((inv2 * inv2) * inv2) * inv2));
        const cd t_k = rc(k * k, cadd(g1, rc(rho2, g2)));
        const cd t_c = rc(2.0, cadd(g2, rc(rho2, g3)));
        const cd t_z = rc((4.0 * zeta) * zeta, cadd(g3, rc(rho2, g4)));
        const cd l_z = cadd(cadd(t_k, t_c), t_z);
        const cd l_r = cr(cr(cadd(rc(8.0, g3), rc(4.0 * rho2, g4)), rho), zeta);
        const double kxi = k * xi;
        const double sh = std::sin(0.5 * kxi);
        const double sv[3] = {1.0, std::sin(kxi), -2.0 * (sh * sh)};
        for (int sp = 0; sp < 3; sp++) {
            const double sw = sp == 0 ? w : sv[sp] * w;
            const cd pz = e_mul(cd(sw, 0.0), l_z), pr = e_mul(cd(sw, 0.0), l_r);
            s[2 * sp][0] = pz.real() + s[2 * sp][0];
            s[2 * sp][1] = pz.imag() + s[2 * sp][1];
            s[2 * sp + 1][0] = pr.real() + s[2 * sp + 1][0];
            s[2 * sp + 1][1] = pr.imag() + s[2 * sp + 1][1];
        }
    }
    F6 f;
    f.ezc = cm(gain, cd(s[0][0], s[0][1]));
    f.erc = cm(gain, cd(s[1][0], s[1][1]));
    f.ezs = cm(gain, cd(s[2][0], s[2][1]));
    f.ers = cm(gain, cd(s[3][0], s[3][1]));
    f.ezo = cm(gain, cd(s[4][0], s[4][1]));
    f.ero = cm(gain, cd(s[5][0], s[5][1]));
    return f;
}

// the plain projection: td * Ez + rho_proj_factor * Erho, per shape
inline void project(const F6 &f, double td, double rpf, cd out[3]) {
    out[0] = cadd(rc(td, f.ezc), rc(rpf, f.erc));
    out[1] = cadd(rc(td, f.ezs), rc(rpf, f.ers));
    out[2] = cadd(rc(td, f.ezo), rc(rpf, f.ero));
}

// --- the near cells (`_apply_near_correction`, `_near_cells`) ---------------

struct NearCells {
    VI entry, seg;
    VC vals[3];
};

// One source block's graded near values over the whole deck, in Python's
// (pair, then support entry) order. `labels` non-null: the extended kernel's
// pair rule on that block.
NearCells near_cells(const Deck &d, const KConst &K, const V &src_c, const V &src_t, const Labels *labels) {
    NearCells out;
    VI mm, nn;
    near_pairs(d, src_c, src_t, mm, nn);
    if (mm.empty()) return out;
    V xg, wg;
    graded_endpoint_rule(min_a_over_hh(d), kNQpNear, xg, wg);
    const std::size_t G = xg.size();
    // the cells' layout: pair p's entries are its test segment's
    VI pair_start(mm.size() + 1, 0);
    for (std::size_t p = 0; p < mm.size(); p++) {
        const std::size_t m = static_cast<std::size_t>(mm[p]);
        pair_start[p + 1] = pair_start[p] + (d.starts[m + 1] - d.starts[m]);
    }
    const std::size_t n_cells = static_cast<std::size_t>(pair_start.back());
    out.entry.resize(n_cells);
    out.seg.resize(n_cells);
    for (int s = 0; s < 3; s++) out.vals[s].resize(n_cells);
    const ll n_pairs = static_cast<ll>(mm.size());
#pragma omp parallel for schedule(dynamic, 4)
    for (ll pl = 0; pl < n_pairs; pl++) {
        const std::size_t p = static_cast<std::size_t>(pl);
        const std::size_t m = static_cast<std::size_t>(mm[p]), n = static_cast<std::size_t>(nn[p]);
        const double a_obs = d.uniform ? d.a_uniform : d.a[m];
        const bool elig = labels != nullptr && labels->obs[m] == labels->src[n] && labels->obs[m] >= 0;
        std::vector<cd> phi(3 * G);
        for (std::size_t g = 0; g < G; g++) {
            const double xi = d.hh[m] * xg[g];
            double o[3];
            for (int c = 0; c < 3; c++) o[c] = d.c[3 * m + c] + xi * d.t[3 * m + c];
            const PairGeom pg = pair_geometry(o, &d.t[3 * m], a_obs, &src_c[3 * n], &src_t[3 * n]);
            F6 f = reduced_cos1(K, pg.z, pg.rho, d.hh[n]);
            if (elig) {
                const F6 dl = ek_delta_cos1(K, pg.z, pg.rho, d.hh[n], d.a[n]);
                f.erc = cadd(f.erc, dl.erc);
                f.ezc = cadd(f.ezc, dl.ezc);
                f.ers = cadd(f.ers, dl.ers);
                f.ezs = cadd(f.ezs, dl.ezs);
                f.ero = cadd(f.ero, dl.ero);
                f.ezo = cadd(f.ezo, dl.ezo);
            }
            cd v[3];
            project(f, pg.td, pg.rpf, v);
            for (int s = 0; s < 3; s++) phi[static_cast<std::size_t>(s) * G + g] = v[s];
        }
        std::size_t cell = static_cast<std::size_t>(pair_start[p]);
        for (ll e = d.starts[m]; e < d.starts[m + 1]; e++, cell++) {
            const std::size_t ez = static_cast<std::size_t>(e);
            double acc[3][2] = {{0, 0}, {0, 0}, {0, 0}};
            for (std::size_t g = 0; g < G; g++) {
                const double xi = d.hh[m] * xg[g];
                const cd fval = basis_value(d.sigAC[ez], d.B[ez], d.sigC[ez], d.k, xi);
                const cd w = rc(wg[g] * d.hh[m], fval);
                for (int s = 0; s < 3; s++) {
                    const cd pr = e_mul(w, phi[static_cast<std::size_t>(s) * G + g]);
                    acc[s][0] = pr.real() + acc[s][0];
                    acc[s][1] = pr.imag() + acc[s][1];
                }
            }
            out.entry[cell] = e;
            out.seg[cell] = static_cast<ll>(n);
            for (int s = 0; s < 3; s++) out.vals[s][cell] = cd(acc[s][0], acc[s][1]);
        }
    }
    return out;
}

// --- the far fill (`_far_fill_accel`) over one observer band -----------------

// A band of test segments [m0, m1) and their entries [e0, e1): the
// sub-context `_sub_view` builds, against every source of the deck.
struct Band {
    ll m0 = 0, m1 = 0, e0 = 0, e1 = 0;
};

// The band's far fill into `out` (three (e1-e0, N) C-order arrays): written
// (`fold` false) or folded on as `out += -1 * value` (the PEC image).
void far_fill(const Deck &d, const Ctx &x, const Band &b, const V &src_c, const V &src_t, const Labels *labels,
              VC out[3], bool fold) {
    const ll nq = x.nq, M = b.m1 - b.m0, ne = b.e1 - b.e0, N = d.N;
    const std::size_t o0 = static_cast<std::size_t>(b.m0 * nq) * 3;
    CF oc = f64(x.obs_c.data() + o0, {M * nq, 3}), ot = f64(x.obs_t.data() + o0, {M * nq, 3});
    CF oa = f64(x.a_obs.data() + b.m0 * nq, {M * nq});
    CF sc = f64(src_c.data(), {N, 3}), st = f64(src_t.data(), {N, 3}), sh = f64(d.hh.data(), {N});
    V cgx, cgw;  // `_leggauss_cached(self.n_qp_const)`
    leggauss(kNQpConst, cgx, cgw);
    CF gt = f64(cgx.data(), {static_cast<ll>(cgx.size())}), gw = f64(cgw.data(), {static_cast<ll>(cgw.size())});
    CC we = c128(x.w_entry.data() + b.e0 * nq, {ne, nq});
    std::vector<int64_t> st64(static_cast<std::size_t>(M + 1));  // the band's own CSR starts
    for (ll m = 0; m <= M; m++) st64[static_cast<std::size_t>(m)] = d.starts[static_cast<std::size_t>(b.m0 + m)] - b.e0;
    CI sts = i64(st64.data(), {M + 1});
    std::unique_ptr<GalerkinFoldBlock> fb;
    if (fold) {
        fb.reset(new GalerkinFoldBlock{
            nd::array_t<cd>::borrow(buffer(out[0].data(), sizeof(cd), {ne, N}, true)),
            nd::array_t<cd>::borrow(buffer(out[1].data(), sizeof(cd), {ne, N}, true)),
            nd::array_t<cd>::borrow(buffer(out[2].data(), sizeof(cd), {ne, N}, true)), -1.0, 0.0});
    }
    std::tuple<nd::array_t<cd>, nd::array_t<cd>, nd::array_t<cd>> r;
    if (labels == nullptr) {
        r = sinusoidal_galerkin_far_fill(oc, ot, oa, sc, st, sh, d.k, d.eta, gt, gw, we, sts, 0, fb.get(), false);
    } else {
        std::vector<int64_t> g_obs(labels->obs.begin() + b.m0, labels->obs.begin() + b.m1);
        std::vector<int64_t> g_src(labels->src.begin(), labels->src.end());
        V egx, egw;
        leggauss(kNQpEkDelta, egx, egw);  // `_ek_delta_rule(16, 1)`
        r = sinusoidal_galerkin_far_fill_ek(oc, ot, oa, sc, st, sh, d.k, d.eta, gt, gw, we, sts, f64(d.a.data(), {N}),
                                            i64(g_obs.data(), {M}), i64(g_src.data(), {N}),
                                            f64(egx.data(), {static_cast<ll>(egx.size())}),
                                            f64(egw.data(), {static_cast<ll>(egw.size())}), 0, fb.get());
    }
    if (!fold) {
        const nd::array_t<cd> *parts[3] = {&std::get<0>(r), &std::get<1>(r), &std::get<2>(r)};
        for (int s = 0; s < 3; s++) {
            if (parts[s]->size() != ne * N) throw std::logic_error("far fill: unexpected shape");
            out[s].assign(parts[s]->data(), parts[s]->data() + ne * N);
        }
    }
}

// --- the Sommerfeld remainder (`_tested_sommerfeld_remainder`) ---------------

struct Somm {
    cd eps_t, c2;
    ll q = 0, n_src = 0;
    V srcf, t_src, shp_w;  // (n_src, 3), (n_src, 3), (3, N, q)
    std::shared_ptr<somm_proj::GridView> view;
    std::vector<cd> far;
    std::size_t n_reg = 0;
};

// `_sommerfeld_remainder_prepare(geom, k, eps_t, cos_shape="cos-1")`
void somm_prepare(const Front &fr, const Deck &d, Somm &sm) {
    const Geometry &g = fr.geometry;
    const ll N = d.N;
    for (ll i = 0; i < N; i++) {
        const std::size_t z = static_cast<std::size_t>(i);
        if (g.seg_l[z][2] - 0.0 < -1e-12 || g.seg_r[z][2] - 0.0 < -1e-12)
            decline("a wire below the Sommerfeld ground (the reference raises)");
    }
    V gx, gw;
    leggauss(kNQpSommerfeld, gx, gw);
    const ll q = kNQpSommerfeld;
    sm.q = q;
    sm.n_src = N * q;
    const double k = d.k;
    sm.srcf.resize(static_cast<std::size_t>(3 * N * q));
    sm.t_src.resize(static_cast<std::size_t>(3 * N * q));
    sm.shp_w.resize(static_cast<std::size_t>(3 * N * q));
    for (ll n = 0; n < N; n++) {
        const std::size_t nz = static_cast<std::size_t>(n);
        const double h_half = 0.5 * g.seg_h[nz];
        for (ll j = 0; j < q; j++) {
            const std::size_t jz = static_cast<std::size_t>(j);
            const double zloc = h_half * gx[jz];
            const std::size_t at = static_cast<std::size_t>(n * q + j);
            for (int c = 0; c < 3; c++) {
                sm.srcf[3 * at + static_cast<std::size_t>(c)] = d.c[3 * nz + c] + zloc * d.t[3 * nz + c];
                sm.t_src[3 * at + static_cast<std::size_t>(c)] = d.t[3 * nz + c];
            }
            const double w_node = h_half * gw[jz];
            const double s = std::sin(0.5 * k * zloc);
            const double shp[3] = {1.0, std::sin(k * zloc), -2.0 * (s * s)};
            for (int c = 0; c < 3; c++) sm.shp_w[static_cast<std::size_t>((c * N + n) * q + j)] = shp[c] * w_node;
        }
    }
    V sl(static_cast<std::size_t>(3 * N)), sr(static_cast<std::size_t>(3 * N));
    for (ll n = 0; n < N; n++)
        for (int c = 0; c < 3; c++) {
            sl[static_cast<std::size_t>(3 * n + c)] = g.seg_l[static_cast<std::size_t>(n)][static_cast<std::size_t>(c)];
            sr[static_cast<std::size_t>(3 * n + c)] = g.seg_r[static_cast<std::size_t>(n)][static_cast<std::size_t>(c)];
        }
    const double r1_max = razor::max_image_distance(sl.data(), sr.data(), static_cast<std::size_t>(N), 0.0);
    const somm_core::Grid &grid = somm_core::get_grid(sm.eps_t, k, r1_max, fr.omega, kMu);
    sm.n_reg = grid.regions.size();
    if (sm.n_reg > somm_proj::MAX_REGIONS) decline("a Sommerfeld grid layout this copy does not view");
    double r0[somm_proj::MAX_REGIONS], dr[somm_proj::MAX_REGIONS], th0[somm_proj::MAX_REGIONS],
        dth[somm_proj::MAX_REGIONS];
    const cd *vals[somm_proj::MAX_REGIONS];
    std::ptrdiff_t shape0[somm_proj::MAX_REGIONS], nr[somm_proj::MAX_REGIONS], nth[somm_proj::MAX_REGIONS];
    for (std::size_t r = 0; r < sm.n_reg; r++) {
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
    sm.view = std::make_shared<somm_proj::GridView>(somm_proj::build_grid_view(
        grid.r1_max, grid.r_break, 30.0 * (kPi / 180.0), grid.r_near, r0, dr, th0, dth, sm.n_reg, vals, shape0, nr,
        nth));
    sm.far = somm_core::far_pack(grid);
    somm_proj::set_far(*sm.view, sm.far.data(), static_cast<std::ptrdiff_t>(sm.far.size()));
}

// The remainder replayed at the band's test observers, each chunk reduced
// against the test functions (`_tested_contrib_rows`) and subtracted from
// `img` (the band's c2-scaled image triple).
void somm_subtract(const Deck &d, const Ctx &x, const Band &b, const Somm &sm, VC img[3]) {
    const ll nq = x.nq, N = d.N, q = sm.q;
    const ll M = (b.m1 - b.m0) * nq;
    ll chunk = std::max<ll>(1, kRemainderChunkElems / std::max<ll>(1, sm.n_src));  // `elem_rows`
    if (nq > 1) chunk = std::max(nq, (chunk / nq) * nq);
    std::vector<cd> table, blk;
    for (ll a0 = 0; a0 < M; a0 += chunk) {
        const ll a1 = std::min(a0 + chunk, M), m = a1 - a0;
        const double *obs = x.obs_c.data() + static_cast<std::size_t>((b.m0 * nq + a0) * 3);
        const double *tob = x.obs_t.data() + static_cast<std::size_t>((b.m0 * nq + a0) * 3);
        table.assign(static_cast<std::size_t>(m * sm.n_src), cd(0.0, 0.0));
        somm_core::field_proj_batch(*sm.view, sm.n_reg, m, sm.n_src, obs, tob, sm.srcf.data(), sm.t_src.data(), 0.0,
                                    d.k, table.data(), 0, true);
        // `remainder_shape_reduce(shp_w, table)`: block (3, m, N)
        blk.assign(static_cast<std::size_t>(3 * m * N), cd(0.0, 0.0));
        for (ll i = 0; i < m; i++) {
            const cd *row = table.data() + i * N * q;
            for (int s = 0; s < 3; s++) {
                const double *w = sm.shp_w.data() + s * N * q;
                for (ll n = 0; n < N; n++) {
                    double are = 0.0, aim = 0.0;
                    for (ll kq = 0; kq < q; kq++) acc_add(e_mul(cd(w[n * q + kq], 0.0), row[n * q + kq]), are, aim);
                    blk[static_cast<std::size_t>((s * m + i) * N + n)] = cd(are, aim);
                }
            }
        }
        // the chunk's whole test segments, and their entries
        const ll ms0 = b.m0 + a0 / nq, ms1 = b.m0 + a1 / nq;
        for (ll mseg = ms0; mseg < ms1; mseg++)
            for (ll e = d.starts[static_cast<std::size_t>(mseg)]; e < d.starts[static_cast<std::size_t>(mseg) + 1]; e++) {
                const std::size_t ez = static_cast<std::size_t>(e);
                const ll ml = mseg - ms0;
                for (int s = 0; s < 3; s++) {
                    cd *dst = img[s].data() + static_cast<std::size_t>((e - b.e0) * N);
                    for (ll n = 0; n < N; n++) {
                        cd acc(0.0, 0.0);
                        for (ll qq = 0; qq < nq; qq++)
                            acc = cadd(acc, cm(x.w_entry[ez * static_cast<std::size_t>(nq) + static_cast<std::size_t>(qq)],
                                               blk[static_cast<std::size_t>((s * m + ml * nq + qq) * N + n)]));
                        dst[n] = csub(dst[n], acc);
                    }
                }
            }
    }
}

// --- one band's triple: the free block, the image, the fold -------------------

struct Ground {
    bool pec = false, compose = false;
    cd c2;
};

struct Blocks {
    V img_c, img_t;
    Labels lab_free, lab_img;
    bool ek = false;
    NearCells near_free, near_img;
    std::unique_ptr<Somm> somm;
};

// Places the cells the band holds: assigned, or subtracted (`sub`)
void place_near(const NearCells &nc, const Band &b, ll N, VC t[3], bool sub, std::vector<std::size_t> *placed) {
    for (std::size_t c = 0; c < nc.entry.size(); c++) {
        const ll e = nc.entry[c];
        if (e < b.e0 || e >= b.e1) continue;
        const std::size_t at = static_cast<std::size_t>((e - b.e0) * N + nc.seg[c]);
        for (int s = 0; s < 3; s++) t[s][at] = sub ? csub(t[s][at], nc.vals[s][c]) : nc.vals[s][c];
        if (placed != nullptr) placed->push_back(at);
    }
}

void band_triple(const Deck &d, const Ctx &x, const Band &b, const Ground &gr, const Blocks &bl, VC t[3]) {
    const ll N = d.N;
    far_fill(d, x, b, d.c, d.t, bl.ek ? &bl.lab_free : nullptr, t, false);
    place_near(bl.near_free, b, N, t, false, nullptr);
    if (gr.pec) {
        std::vector<std::size_t> cells;
        place_near(bl.near_img, b, N, t, true, &cells);
        std::vector<cd> held[3];
        for (int s = 0; s < 3; s++)
            for (std::size_t at : cells) held[s].push_back(t[s][at]);
        far_fill(d, x, b, bl.img_c, bl.img_t, bl.ek ? &bl.lab_img : nullptr, t, true);
        for (int s = 0; s < 3; s++)
            for (std::size_t c = 0; c < cells.size(); c++) t[s][cells[c]] = held[s][c];
    } else if (gr.compose) {
        VC img[3];
        far_fill(d, x, b, bl.img_c, bl.img_t, bl.ek ? &bl.lab_img : nullptr, img, false);
        place_near(bl.near_img, b, N, img, false, nullptr);
        for (int s = 0; s < 3; s++)
            for (cd &v : img[s]) v = cm(gr.c2, v);  // np.multiply(coef, a, out=a)
        somm_subtract(d, x, b, *bl.somm, img);
        for (int s = 0; s < 3; s++)
            for (std::size_t i = 0; i < img[s].size(); i++) t[s][i] = csub(t[s][i], img[s][i]);
    }
}

// --- the scatter ---------------------------------------------------------------

// `_scatter_coef_product` below the dense threshold: `T = zeros; np.add.at(T,
// i_of_entry, contrib)`, `M[m, i] = coef`, `G = sum(T @ M)` (numpy's zgemm),
// into F order.
std::vector<cd> scatter_dense(const Deck &d, const VC t[3], const nec5::NumpyLinalg &np) {
    const ll N = d.N, nb = d.n_basis;
    const std::size_t Nz = static_cast<std::size_t>(N), nbz = static_cast<std::size_t>(nb);
    std::vector<cd> G(nbz * nbz, cd(0.0, 0.0)), acc(nbz * nbz), P(nbz * nbz);
    const VC *coefs[3] = {&d.sigAC, &d.B, &d.sigC};
    for (int s = 0; s < 3; s++) {
        std::vector<cd> T(nbz * Nz, cd(0.0, 0.0)), M(Nz * nbz, cd(0.0, 0.0));
        for (ll e = 0; e < d.nnz; e++) {
            const std::size_t ez = static_cast<std::size_t>(e), i = static_cast<std::size_t>(d.i_of_entry[ez]);
            for (std::size_t n = 0; n < Nz; n++) T[i * Nz + n] = cadd(T[i * Nz + n], t[s][ez * Nz + n]);
            M[static_cast<std::size_t>(d.m_of_entry[ez]) * nbz + i] = (*coefs[s])[ez];
        }
        const nec5::Op a{T.data(), nb, N, N, 1};
        const nec5::Op bo{M.data(), N, nb, nb, 1};
        rp::np_matmul(np, a, bo, P.data(), nb, 1);
        for (std::size_t q = 0; q < nbz * nbz; q++) acc[q] = s == 0 ? cadd(cd(0.0, 0.0), P[q]) : cadd(acc[q], P[q]);
    }
    for (std::size_t i = 0; i < nbz; i++)
        for (std::size_t j = 0; j < nbz; j++) G[i + j * nbz] = acc[i * nbz + j];
    return G;
}

// A csc factor's columns: for basis j its stored (row, coef) in row order.
struct CscCols {
    std::vector<std::vector<std::pair<ll, cd>>> col;
};

// `scipy.sparse.csc_matrix((coef[sel], (row[sel], i_of_entry[sel])))`
CscCols csc_of(const Deck &d, const VC &coef, const VI &row_of_seg) {
    CscCols M;
    M.col.resize(static_cast<std::size_t>(d.n_basis));
    for (ll e = 0; e < d.nnz; e++) {
        const std::size_t ez = static_cast<std::size_t>(e);
        const ll row = row_of_seg[static_cast<std::size_t>(d.m_of_entry[ez])];
        if (row < 0) continue;
        M.col[static_cast<std::size_t>(d.i_of_entry[ez])].emplace_back(row, coef[ez]);
    }
    for (auto &c : M.col)
        std::stable_sort(c.begin(), c.end(), [](const std::pair<ll, cd> &a, const std::pair<ll, cd> &b) {
            return a.first < b.first;
        });
    return M;
}

// one cell of `dense_row @ csc`: zero plus the column's entries, unfused
inline cd row_csc(const cd *row, const std::vector<std::pair<ll, cd>> &col) {
    double ar = 0.0, ai = 0.0;
    for (const auto &e : col) {
        const cd p = e_mul(e.second, row[e.first]);
        ar = ar + p.real();
        ai = ai + p.imag();
    }
    return cd(ar, ai);
}

// `_assemble_G_banded`'s scatter state and `_scatter_band`
struct BandRows {
    CscCols Ms[3];
    VI last;
    std::vector<std::vector<cd>> carry[3];  // per basis, or empty
    std::vector<bool> open;
};

void scatter_band(const Deck &d, BandRows &R, const Band &b, const VC t[3], std::vector<cd> &G) {
    const ll N = d.N, nb = d.n_basis;
    const std::size_t Nz = static_cast<std::size_t>(N);
    std::vector<ll> touched;
    for (ll e = b.e0; e < b.e1; e++) touched.push_back(d.i_of_entry[static_cast<std::size_t>(e)]);
    std::sort(touched.begin(), touched.end());
    touched.erase(std::unique(touched.begin(), touched.end()), touched.end());
    for (ll bi : touched) {
        const std::size_t bz = static_cast<std::size_t>(bi);
        std::vector<cd> T[3];
        for (int s = 0; s < 3; s++) {
            if (R.open[bz]) {
                T[s] = std::move(R.carry[s][bz]);
            } else {
                T[s].assign(Nz, cd(0.0, 0.0));  // a fresh row starts from +0
            }
        }
        R.open[bz] = false;
        for (ll e = b.e0; e < b.e1; e++) {
            if (d.i_of_entry[static_cast<std::size_t>(e)] != bi) continue;
            for (int s = 0; s < 3; s++) {
                const cd *src = t[s].data() + static_cast<std::size_t>(e - b.e0) * Nz;
                for (std::size_t n = 0; n < Nz; n++) T[s][n] = cadd(T[s][n], src[n]);
            }
        }
        if (R.last[bz] < b.e1) {
            for (ll j = 0; j < nb; j++) {
                const std::size_t jz = static_cast<std::size_t>(j);
                cd v = cadd(cd(0.0, 0.0), row_csc(T[0].data(), R.Ms[0].col[jz]));
                v = cadd(v, row_csc(T[1].data(), R.Ms[1].col[jz]));
                v = cadd(v, row_csc(T[2].data(), R.Ms[2].col[jz]));
                G[bz + jz * static_cast<std::size_t>(nb)] = v;
            }
        } else {
            for (int s = 0; s < 3; s++) R.carry[s][bz] = std::move(T[s]);
            R.open[bz] = true;
        }
    }
}

// `_segment_bands(ctx, n_basis, N, n_triples)`
std::vector<Band> segment_bands(const Deck &d, int n_triples) {
    const ll N = d.N, nb = d.n_basis;
    const double entries_per_seg = static_cast<double>(d.nnz) / static_cast<double>(std::max<ll>(1, N));
    const double per_seg =
        16.0 * ((((static_cast<double>(3 * n_triples) * entries_per_seg) * static_cast<double>(N)) +
                 static_cast<double>(3 * N)) +
                static_cast<double>(4 * nb));
    const ll z_bytes = 16 * nb * nb;
    const ll budget = std::min<ll>(kSweptMemMb * 1024 * 1024, z_bytes / 4);
    const ll step = std::max<ll>(std::min(kBandMinSegments, N), static_cast<ll>(py_floordiv(static_cast<double>(budget), per_seg)));
    std::vector<Band> out;
    for (ll m0 = 0; m0 < N; m0 += step) {
        Band b;
        b.m0 = m0;
        b.m1 = std::min(m0 + step, N);
        b.e0 = d.starts[static_cast<std::size_t>(b.m0)];
        b.e1 = b.m1 == N ? d.nnz : d.starts[static_cast<std::size_t>(b.m1)];
        out.push_back(b);
    }
    return out;
}

ll band_budget_bytes(ll n_basis) { return std::min<ll>(kSweptMemMb * 1024 * 1024, 16 * n_basis * n_basis / 4); }

// --- `_ek_bracket_correction_tested` -------------------------------------------

struct BracketPlan {
    const V *src_c, *src_t;
    cd scale;
    const Labels *lab;
    std::vector<ll> near_key;  // sorted
    VI cols;
};

// `_ek_end_bracket_fields(k, H, z, rho, src_a, sign, "cos-1")` at one element
F6 end_bracket(const KConst &K, double H, double z, double rho, double src_a, double sign) {
    const double k = K.k;
    const double xi_end = sign * H;
    const double zeta = z - xi_end;
    const double R = std::sqrt(rho * rho + zeta * zeta);
    const cd x = cm(kJ, cd(k * R, 0.0));
    const cd x2 = cm(x, x), x3 = cm(x2, x);
    const cd a1 = radd(1.0, x);
    const cd a2 = cadd(radd(3.0, rc(3.0, x)), x2);
    const cd a3 = cadd(cadd(radd(15.0, rc(15.0, x)), rc(6.0, x2)), x3);
    const double inv2 = 1.0 / (R * R);
    const cd base = cdr(std::exp(cm(kMinusJ, cd(k * R, 0.0))), R);
    const cd g1 = cr(rc(-0.5, cm(base, a1)), inv2);
    const cd g2 = cr(rc(0.25, cm(base, a2)), inv2 * inv2);
    const cd g3 = cr(rc(-0.125, cm(base, a3)), (inv2 * inv2) * inv2);
    const double rho2 = rho * rho;
    const cd w_val = cadd(g1, rc(rho2, g2));
    const cd dw_dxi = rc(-2.0 * zeta, cadd(g2, rc(rho2, g3)));
    const cd dw_drho = rc(2.0 * rho, cadd(rc(2.0, g2), rc(rho2, g3)));
    const cd gain = cr(cm(K.prc, cd(src_a * src_a, 0.0)), sign);
    const double kxi = k * xi_end;
    const double s_sin = mw_libm_c::rsin(kxi), c_cos = mw_libm_c::rcos(kxi);
    const double hs = std::sin(0.5 * kxi);
    const double s_cos = -2.0 * (hs * hs), sp_cos = -k * s_sin;
    F6 f;
    f.ezc = cm(gain, rc(1.0, dw_dxi));
    f.erc = cm(cneg(gain), rc(1.0, dw_drho));
    f.ezs = cm(gain, csub(rc(s_sin, dw_dxi), rc(k * c_cos, w_val)));
    f.ers = cm(cneg(gain), rc(s_sin, dw_drho));
    f.ezo = cm(gain, csub(rc(s_cos, dw_dxi), rc(sp_cos, w_val)));
    f.ero = cm(cneg(gain), rc(s_cos, dw_drho));
    return f;
}

void ek_bracket_block(const Deck &d, const Ctx &x, const KConst &K, const BracketPlan &plan, const VI &col_of,
                      const V &xg, const V &wg, ll m0, ll m1, VC corr[3], ll ncols) {
    const ll N = d.N, nq = x.nq;
    const ll e_base = d.starts[static_cast<std::size_t>(m0)];
    const Labels &L = *plan.lab;
    for (int side = 0; side < 2; side++) {
        const double sign = side == 0 ? -1.0 : 1.0;
        const std::vector<bool> &bad = side == 0 ? L.bad_lo : L.bad_hi;
        VI nb;
        for (ll n = 0; n < N; n++)
            if (bad[static_cast<std::size_t>(n)]) nb.push_back(n);
        if (nb.empty()) continue;
        VI m_sel, n_sel;
        for (ll m = m0; m < m1; m++) {
            const ll go = L.obs[static_cast<std::size_t>(m)];
            for (ll n : nb)
                if (go == L.src[static_cast<std::size_t>(n)] && go >= 0) {
                    m_sel.push_back(m);
                    n_sel.push_back(n);
                }
        }
        if (m_sel.empty()) continue;
        std::vector<bool> is_near(m_sel.size());
        for (std::size_t p = 0; p < m_sel.size(); p++)
            is_near[p] = std::binary_search(plan.near_key.begin(), plan.near_key.end(), m_sel[p] * N + n_sel[p]);
        for (int graded = 0; graded < 2; graded++) {
            for (std::size_t p = 0; p < m_sel.size(); p++) {
                if (is_near[p] != (graded == 1)) continue;
                const std::size_t m = static_cast<std::size_t>(m_sel[p]), n = static_cast<std::size_t>(n_sel[p]);
                const double a_obs = d.uniform ? d.a_uniform : d.a[m];
                const std::size_t G = graded ? xg.size() : static_cast<std::size_t>(nq);
                std::vector<cd> phi(3 * G);
                for (std::size_t g = 0; g < G; g++) {
                    double o[3];
                    if (graded) {
                        const double xi = d.hh[m] * xg[g];
                        for (int c = 0; c < 3; c++) o[c] = d.c[3 * m + c] + xi * d.t[3 * m + c];
                    } else {
                        for (int c = 0; c < 3; c++) o[c] = x.obs_c[(m * static_cast<std::size_t>(nq) + g) * 3 + c];
                    }
                    const PairGeom pg = pair_geometry(o, &d.t[3 * m], a_obs, &(*plan.src_c)[3 * n], &(*plan.src_t)[3 * n]);
                    const F6 f = end_bracket(K, d.hh[n], pg.z, pg.rho, d.a[n], sign);
                    cd v[3];
                    project(f, pg.td, pg.rpf, v);
                    for (int s = 0; s < 3; s++) phi[static_cast<std::size_t>(s) * G + g] = v[s];
                }
                const ll col = col_of[n];
                for (ll e = d.starts[m]; e < d.starts[m + 1]; e++) {
                    const std::size_t ez = static_cast<std::size_t>(e);
                    double acc[3][2] = {{0, 0}, {0, 0}, {0, 0}};
                    for (std::size_t g = 0; g < G; g++) {
                        cd w;
                        if (graded) {
                            const double xi = d.hh[m] * xg[g];
                            w = rc(wg[g] * d.hh[m], basis_value(d.sigAC[ez], d.B[ez], d.sigC[ez], d.k, xi));
                        } else {
                            w = x.w_entry[ez * static_cast<std::size_t>(nq) + g];
                        }
                        for (int s = 0; s < 3; s++) {
                            const cd pr = e_mul(w, phi[static_cast<std::size_t>(s) * G + g]);
                            acc[s][0] = pr.real() + acc[s][0];
                            acc[s][1] = pr.imag() + acc[s][1];
                        }
                    }
                    const std::size_t at = static_cast<std::size_t>((e - e_base) * ncols + col);
                    for (int s = 0; s < 3; s++) corr[s][at] = cadd(corr[s][at], cm(plan.scale, cd(acc[s][0], acc[s][1])));
                }
            }
        }
    }
}

void ek_bracket_correction(const Deck &d, const Ctx &x, const KConst &K, const Ground &gr, const Blocks &bl,
                           std::vector<cd> &G, const nec5::NumpyLinalg &np) {
    const ll N = d.N, nb = d.n_basis;
    V xg, wg;
    graded_endpoint_rule(min_a_over_hh(d), kNQpNear, xg, wg);
    std::vector<BracketPlan> plans;
    auto add_plan = [&](const V &sc, const V &st, cd scale, const Labels &lab) {
        bool any = false;
        for (ll n = 0; n < N; n++) any = any || lab.bad_lo[static_cast<std::size_t>(n)] || lab.bad_hi[static_cast<std::size_t>(n)];
        if (!any) return;
        BracketPlan p;
        p.src_c = &sc;
        p.src_t = &st;
        p.scale = scale;
        p.lab = &lab;
        VI mm, nn;
        near_pairs(d, sc, st, mm, nn);
        for (std::size_t q = 0; q < mm.size(); q++) p.near_key.push_back(mm[q] * N + nn[q]);
        std::sort(p.near_key.begin(), p.near_key.end());
        for (ll n = 0; n < N; n++)
            if (lab.bad_lo[static_cast<std::size_t>(n)] || lab.bad_hi[static_cast<std::size_t>(n)]) p.cols.push_back(n);
        plans.push_back(std::move(p));
    };
    add_plan(d.c, d.t, cd(1.0, 0.0), bl.lab_free);
    if (gr.pec) add_plan(bl.img_c, bl.img_t, cd(-1.0, 0.0), bl.lab_img);  // -fg.image_coefficient (1.0)
    if (gr.compose) add_plan(bl.img_c, bl.img_t, cneg(gr.c2), bl.lab_img);
    if (plans.empty()) return;
    VI cols;
    for (const auto &p : plans) cols.insert(cols.end(), p.cols.begin(), p.cols.end());
    std::sort(cols.begin(), cols.end());
    cols.erase(std::unique(cols.begin(), cols.end()), cols.end());
    const ll nc = static_cast<ll>(cols.size());
    VI col_of(static_cast<std::size_t>(N), -1);
    for (ll c = 0; c < nc; c++) col_of[static_cast<std::size_t>(cols[static_cast<std::size_t>(c)])] = c;
    std::vector<cd> T[3];
    for (int s = 0; s < 3; s++) T[s].assign(static_cast<std::size_t>(nb * nc), cd(0.0, 0.0));
    const ll nnz = d.nnz;
    const ll per_seg = 48 * nc * std::max<ll>(1, nnz / N);
    const ll blk = std::min(std::max<ll>(1, kEkBracketBandBytes / per_seg), std::max<ll>(1, (nb * N) / nnz));
    for (ll m0 = 0; m0 < N; m0 += blk) {
        const ll m1 = std::min(m0 + blk, N);
        const ll e0 = d.starts[static_cast<std::size_t>(m0)], e1 = m1 == N ? nnz : d.starts[static_cast<std::size_t>(m1)];
        VC corr[3];
        for (int s = 0; s < 3; s++) corr[s].assign(static_cast<std::size_t>((e1 - e0) * nc), cd(0.0, 0.0));
        for (const auto &p : plans) ek_bracket_block(d, x, K, p, col_of, xg, wg, m0, m1, corr, nc);
        bool any = false;
        for (int s = 0; s < 3 && !any; s++)
            for (const cd &v : corr[s])
                if (v.real() != 0.0 || v.imag() != 0.0) {
                    any = true;
                    break;
                }
        if (!any) continue;
        for (ll e = e0; e < e1; e++) {
            const std::size_t i = static_cast<std::size_t>(d.i_of_entry[static_cast<std::size_t>(e)]);
            for (int s = 0; s < 3; s++)
                for (ll c = 0; c < nc; c++) {
                    cd &dst = T[s][i * static_cast<std::size_t>(nc) + static_cast<std::size_t>(c)];
                    dst = cadd(dst, corr[s][static_cast<std::size_t>((e - e0) * nc + c)]);
                }
        }
    }
    bool any = false;
    for (int s = 0; s < 3 && !any; s++)
        for (const cd &v : T[s])
            if (v.real() != 0.0 || v.imag() != 0.0) {
                any = true;
                break;
            }
    if (!any) return;
    const VC *coefs[3] = {&d.sigAC, &d.B, &d.sigC};
    const std::size_t nbz = static_cast<std::size_t>(nb);
    if (N < kDenseThreshold) {
        // `_bracket_coef_product`: dense, `sum(t @ M)`; `G -= 0.5 * (C + C.T)`
        std::vector<cd> C(nbz * nbz), P(nbz * nbz);
        for (int s = 0; s < 3; s++) {
            std::vector<cd> M(static_cast<std::size_t>(nc) * nbz, cd(0.0, 0.0));
            for (ll e = 0; e < nnz; e++) {
                const std::size_t ez = static_cast<std::size_t>(e);
                const ll row = col_of[static_cast<std::size_t>(d.m_of_entry[ez])];
                if (row < 0) continue;
                M[static_cast<std::size_t>(row) * nbz + static_cast<std::size_t>(d.i_of_entry[ez])] = (*coefs[s])[ez];
            }
            const nec5::Op a{T[s].data(), nb, nc, nc, 1};
            const nec5::Op bo{M.data(), nc, nb, nb, 1};
            rp::np_matmul(np, a, bo, P.data(), nb, 1);
            for (std::size_t q = 0; q < nbz * nbz; q++) C[q] = s == 0 ? cadd(cd(0.0, 0.0), P[q]) : cadd(C[q], P[q]);
        }
        for (std::size_t i = 0; i < nbz; i++)
            for (std::size_t j = 0; j < nbz; j++) {
                cd &g = G[i + j * nbz];
                g = csub(g, rc(0.5, cadd(C[i * nbz + j], C[j * nbz + i])));
            }
        return;
    }
    // banded: rows i0:i1 of C, and the same columns of C, by the csc factors
    CscCols Ms[3];
    for (int s = 0; s < 3; s++) Ms[s] = csc_of(d, *coefs[s], col_of);
    auto cell = [&](std::size_t r, std::size_t j) {
        cd v = cadd(cd(0.0, 0.0), row_csc(T[0].data() + r * static_cast<std::size_t>(nc), Ms[0].col[j]));
        v = cadd(v, row_csc(T[1].data() + r * static_cast<std::size_t>(nc), Ms[1].col[j]));
        return cadd(v, row_csc(T[2].data() + r * static_cast<std::size_t>(nc), Ms[2].col[j]));
    };
    std::vector<cd> crow(nbz);
    for (std::size_t i = 0; i < nbz; i++) {
        for (std::size_t j = 0; j < nbz; j++) crow[j] = cell(i, j);
        for (std::size_t j = 0; j < nbz; j++) {
            cd &g = G[i + j * nbz];
            g = csub(g, rc(0.5, cadd(crow[j], cell(j, i))));
        }
    }
}

// --- `_contact_charge_correction_tested` (a finite ground) --------------------

struct End {
    cd g1, g1p, g2, g2p, g3, gzz;
};

// `_ek_end_gxx(k, zz, rh, b, want_swapped)` at one element (the sinusoidal
// fill core's replay of the same function)
End ek_end_gxx(double k, double zz, double rh, double b, bool swapped) {
    const double r2 = zz * zz + rh * rh;
    const double r = std::sqrt(r2);
    const double kr = k * r;
    const double kr2 = kr * kr;
    const cd jkr = cm(kJ, cd(kr, 0.0));
    const cd c1 = radd(1.0, jkr);
    const cd c2 = csubr(rc(3.0, c1), kr2);
    const cd c3 = csub(cr(radd(6.0, jkr), kr2), rc(15.0, c1));
    const double a2 = b * b;
    const double rh2 = rh * rh;
    const double r4 = r2 * r2;
    const double t1 = 0.25 * a2 * rh2 / r4;
    const double t2 = 0.5 * a2 / r2;
    const cd gz = cdr(std::exp(cm(kMinusJ, cd(kr, 0.0))), r);
    const cd f2 = radd(1.0, rc(t1, c2));
    const cd g2 = cm(gz, f2);
    const cd g1 = csub(g2, cm(rc(t2, c1), gz));
    const cd gzr = cdr(gz, r2);
    const cd f2p = csub(rc(t1, c3), c1);
    const cd g2p = cm(gzr, f2p);
    const cd gzp_t = cm(rc(t2, c2), gzr);
    const cd g3 = cadd(g2p, gzp_t);
    const cd g1p = cr(g3, zz);
    End e;
    e.g1 = g1;
    e.g1p = g1p;
    e.gzz = cm(rc(-zz, c1), gzr);
    if (swapped) {
        const double t2b = 0.5 * b;
        e.g2 = cm(rc(-t2b, c1), gzr);
        const cd g2p_s = cdr(cm(rc(t2b, gzr), c2), r2);
        e.g2p = cr(g2p_s, zz);
        e.g3 = csub(rc(rh2, g2p_s), cm(rc(b, gzr), c1));
    } else {
        e.g2 = cdr(g2, rh);
        e.g2p = cdr(cr(g2p, zz), rh);
        e.g3 = cr(cadd(g3, gzp_t), rh);
    }
    return e;
}

// numpy's real `(M, 3) @ (3,)`: `DOUBLE_matmul`'s gemv
V mat3_vec(const nec5::NumpyLinalg &np, const V &a, const double *x) {
    const ll n = static_cast<ll>(a.size() / 3);
    V y(static_cast<std::size_t>(n));
    if (n == 1) {
        if (np.ddot == nullptr) decline("no real BLAS for the contact correction");
        y[0] = 0.0 + np.ddot(3, a.data(), 1, x, 1);
        return y;
    }
    if (np.dgemv == nullptr) decline("no real BLAS for the contact correction");
    np.dgemv(102, 112, 3, n, 1.0, a.data(), 3, x, 1, 0.0, y.data(), 1);
    return y;
}

void contact_correction(const Front &fr, const Deck &d, const KConst &K, const Ground &gr, const Blocks &bl,
                        const SegView &view, std::vector<cd> &G, const nec5::NumpyLinalg &np) {
    const Geometry &g = fr.geometry;
    const ll N = d.N;
    const std::size_t Nz = static_cast<std::size_t>(N), nbz = static_cast<std::size_t>(d.n_basis);
    struct Node {
        ll i;
        double sgn, p[3];
    };
    std::vector<Node> nodes;
    for (int side = 0; side < 2; side++) {
        const std::vector<bool> &mask = side == 0 ? g.ground_minus : g.ground_plus;
        const double sgn = side == 0 ? -1.0 : 1.0;
        for (std::size_t i = 0; i < Nz; i++)
            if (mask[i]) {
                Node nd;
                nd.i = static_cast<ll>(i);
                nd.sgn = sgn;
                const double sc = sgn * 0.5 * g.seg_h[i];
                for (int c = 0; c < 3; c++) nd.p[c] = d.c[3 * i + c] + sc * d.t[3 * i + c];
                nodes.push_back(nd);
            }
    }
    if (nodes.empty()) return;
    V xg, wg;
    graded_endpoint_rule(min_a_over_hh(d), kNQpNear, xg, wg);
    const std::size_t nq = xg.size(), M = Nz * nq;
    V obs_c(3 * M), obs_t(3 * M), a_obs(M);
    std::vector<ll> obs_seg(M);
    for (std::size_t m = 0; m < Nz; m++)
        for (std::size_t q = 0; q < nq; q++) {
            const double xi = d.hh[m] * xg[q];
            for (int c = 0; c < 3; c++) {
                obs_c[(m * nq + q) * 3 + c] = d.c[3 * m + c] + xi * d.t[3 * m + c];
                obs_t[(m * nq + q) * 3 + c] = d.t[3 * m + c];
            }
            a_obs[m * nq + q] = d.a[m];
            obs_seg[m * nq + q] = static_cast<ll>(m);
        }
    std::vector<cd> w_entry(static_cast<std::size_t>(d.nnz) * nq);
    for (ll e = 0; e < d.nnz; e++) {
        const std::size_t ez = static_cast<std::size_t>(e), m = static_cast<std::size_t>(d.m_of_entry[ez]);
        for (std::size_t q = 0; q < nq; q++)
            w_entry[ez * nq + q] = rc(wg[q] * d.hh[m], basis_value(d.sigAC[ez], d.B[ez], d.sigC[ez], d.k, d.hh[m] * xg[q]));
    }
    const double k = d.k;
    const cd con = py_div(py_mul(kJ, cd(d.eta, 0.0)), cd(K.four_pi_k, 0.0));
    const cd mcon = cneg(con);
    const cd one_minus_c2(1.0 - gr.c2.real(), 0.0 - gr.c2.imag());
    for (const Node &nd : nodes) {
        const std::size_t iz = static_cast<std::size_t>(nd.i);
        std::vector<cd> pref(M), R(M);
        V t_d(M);
        for (std::size_t m = 0; m < M; m++) {
            double dv[3];
            for (int c = 0; c < 3; c++) dv[c] = obs_c[3 * m + c] - nd.p[c];
            const double r2 = edot3(dv, dv);
            const double r0 = std::sqrt(r2 + a_obs[m] * a_obs[m]);
            const cd c1 = radd(1.0, cm(K.jk, cd(r0, 0.0)));
            const cd e = std::exp(cm(K.mjk, cd(r0, 0.0)));
            pref[m] = cdr(cm(cm(K.prc, c1), e), (r0 * r0) * r0);
            t_d[m] = edot3(&obs_t[3 * m], dv);
            R[m] = cr(cm(pref[m], one_minus_c2), t_d[m]);
        }
        // `_contact_ek_masks` (#246's pair rule, per observer)
        bool any_real = false, any_img = false;
        std::vector<bool> use_real(M), use_img(M);
        if (bl.ek) {
            for (std::size_t m = 0; m < M; m++) {
                const std::size_t s = static_cast<std::size_t>(obs_seg[m]);
                use_real[m] = bl.lab_free.obs[s] == bl.lab_free.src[iz] && bl.lab_free.obs[s] >= 0;
                use_img[m] = bl.lab_img.obs[s] == bl.lab_img.src[iz] && bl.lab_img.obs[s] >= 0;
                any_real = any_real || use_real[m];
                any_img = any_img || use_img[m];
            }
        }
        if (any_real || any_img) {
            // `_contact_charge_end_gradient` + `_contact_charge_ek_delta`
            const double *src_c = &d.c[3 * iz], *src_t = &d.t[3 * iz];
            const double src_hh = 0.5 * g.seg_h[iz];
            const double src_a = d.uniform ? d.a_uniform : d.a[iz];
            V rvec(3 * M);
            for (std::size_t m = 0; m < M; m++)
                for (int c = 0; c < 3; c++) rvec[3 * m + c] = obs_c[3 * m + c] - src_c[c];
            const V z_eval = mat3_vec(np, rvec, src_t);
            const V ts = mat3_vec(np, obs_t, src_t);
            std::vector<double> rho_eval(M);
            std::vector<double> rho_vec(3 * M);
            for (std::size_t m = 0; m < M; m++) {
                for (int c = 0; c < 3; c++) rho_vec[3 * m + c] = rvec[3 * m + c] - z_eval[m] * src_t[c];
                const double rho_axis = norm3(&rho_vec[3 * m]);
                rho_eval[m] = std::sqrt(rho_axis * rho_axis + a_obs[m] * a_obs[m]);
            }
            for (std::size_t m = 0; m < M; m++) {
                const double zz = nd.sgn > 0 ? (src_hh - z_eval[m]) : -(src_hh + z_eval[m]);
                const bool swap = rho_eval[m] < src_a;
                const double rh = swap ? src_a : rho_eval[m];
                const double b = swap ? rho_eval[m] : src_a;
                const End e = ek_end_gxx(k, zz, rh, b, swap);
                const cd ez = cm(mcon, e.g1p);
                const cd erho_hat = cdr(cm(con, e.g3), rho_eval[m]);
                const double rho_t = edot3(&rho_vec[3 * m], &obs_t[3 * m]);
                const cd dt = csub(cadd(cr(ez, ts[m]), cr(erho_hat, rho_t)), cr(pref[m], t_d[m]));
                const cd zero(0.0, 0.0);
                const cd a = use_real[m] ? dt : zero;
                const cd bimg = use_img[m] ? dt : zero;
                R[m] = cadd(R[m], csub(a, cm(gr.c2, bimg)));
            }
        }
        // `_tested_contrib_rows(w_entry, m_of_entry, nq, R)` then
        // `np.add.at(T, i_of_entry, R_t)`
        std::vector<cd> T(nbz, cd(0.0, 0.0));
        for (ll e = 0; e < d.nnz; e++) {
            const std::size_t ez = static_cast<std::size_t>(e), m = static_cast<std::size_t>(d.m_of_entry[ez]);
            cd acc(0.0, 0.0);
            for (std::size_t q = 0; q < nq; q++) acc = cadd(acc, cm(w_entry[ez * nq + q], R[m * nq + q]));
            cd &tv = T[static_cast<std::size_t>(d.i_of_entry[ez])];
            tv = cadd(tv, acc);
        }
        // `_contact_node_values`
        const double sseg = nd.sgn * 0.5 * g.seg_h[iz];
        const double half = std::sin(0.5 * k * sseg);
        const double sks = std::sin(k * sseg);
        for (ll q = view.starts[iz]; q < view.starts[iz + 1]; q++) {
            const std::size_t qz = static_cast<std::size_t>(q);
            const cd sig(static_cast<double>(view.sigma[qz]), 0.0);
            const cd val = csub(cadd(cm(sig, view.AC[qz]), cr(view.B[qz], sks)),
                                cr(cr(cr(cm(sig, view.C[qz]), 2.0), half), half));
            const std::size_t col = static_cast<std::size_t>(view.jbasis[qz]);
            for (std::size_t r = 0; r < nbz; r++) {
                const cd t = cm(cm(cd(nd.sgn, 0.0), T[r]), val);  // sgn * T[:, None] * val[None, :]
                cd &x = G[r + col * nbz];
                x = csub(x, t);
            }
        }
    }
}

}  // namespace

bool direct_route(const Front &fr, const nec4::SolverArgs &args) {
    (void)args;
    const Geometry &g = fr.geometry;
    for (std::size_t i = 0; i < g.ground_minus.size(); i++)
        if (g.ground_minus[i] || g.ground_plus[i]) return true;
    return false;
}

std::vector<cd> assemble_Z_direct(const Front &fr, const nec4::SolverArgs &args, const nec5::NumpyLinalg *np) {
    return assemble_Z_direct(fr, args, fr.basis, fr.geometry.n_segs, np);
}

std::vector<cd> assemble_Z_direct(const Front &fr, const nec4::SolverArgs &args, const SegView &view, ll n_basis,
                                  const nec5::NumpyLinalg *np) {
    const bool pec = args.ground.kind == nec4::Ground::pec;
    const bool finite = args.ground.kind == nec4::Ground::finite;
    if (args.ground.kind != nec4::Ground::none && !pec && !finite)
        decline("a reflection-coefficient ground (not the NEC-4.2 slot's)");
    if (np == nullptr) decline("no numpy products for the direct fill");
    // `_refuse_coincident_segments`
    ll cm_ = 0, cn_ = 0;
    if (coincident_segments(fr, cm_, cn_))
        throw SolverRefusal(
            "ValueError",
            "SinusoidalGalerkinSolver: segments " + std::to_string(cm_) + " and " + std::to_string(cn_) +
                " coincide, end for end -- two conductors written on one path -- and the direct-field fill cannot "
                "solve coincident conductors: for two segments on one path its answer depends on rounding "
                "(momwire#1344: the rise-spelled fan read -5.6 - 3326j ohm as written, 3977 - 26802j ohm moved "
                "0.3 m, and 32.8 - 341.1j ohm, the right answer, moved 1 m). The mixed-potential fill solves "
                "them: it is this family's default and serves free space, a PEC ground and a Sommerfeld ground, "
                "but not fill=\"direct\", junction_ports, a wire end in the ground plane or a "
                "reflection-coefficient ground. Solve the deck with BSplineSolver, or spell it so that the "
                "mixed-potential fill serves it - momwire#1344");
    const Deck d = deck_of(fr, view, n_basis);
    const Ctx x = test_context(d);
    const KConst K = kconst(d.k, d.eta);
    Ground gr;
    gr.pec = pec;
    gr.compose = finite;
    Blocks bl;
    bl.ek = args.extended_kernel;
    if (pec || finite) {
        const std::size_t N = static_cast<std::size_t>(d.N);
        bl.img_c.resize(3 * N);
        bl.img_t.resize(3 * N);
        for (std::size_t i = 0; i < N; i++) {
            bl.img_c[3 * i] = d.c[3 * i];
            bl.img_c[3 * i + 1] = d.c[3 * i + 1];
            bl.img_c[3 * i + 2] = 2.0 * 0.0 - d.c[3 * i + 2];
            bl.img_t[3 * i] = d.t[3 * i] * 1.0;
            bl.img_t[3 * i + 1] = d.t[3 * i + 1] * 1.0;
            bl.img_t[3 * i + 2] = d.t[3 * i + 2] * -1.0;
        }
    }
    if (finite) {
        bl.somm.reset(new Somm);
        bl.somm->eps_t = cd(args.ground.eps_r, (-args.ground.sigma) / (fr.omega * kEps0));
        const cd e = bl.somm->eps_t;
        gr.c2 = py_div(cd(e.real() - 1.0, e.imag() - 0.0), cd(e.real() + 1.0, e.imag() + 0.0));
        bl.somm->c2 = gr.c2;
    }
    if (bl.ek) {
        bl.lab_free = ek_labels(fr, d, false);
        if (pec || finite) bl.lab_img = ek_labels(fr, d, true);
    }
    bl.near_free = near_cells(d, K, d.c, d.t, bl.ek ? &bl.lab_free : nullptr);
    if (pec || finite) bl.near_img = near_cells(d, K, bl.img_c, bl.img_t, bl.ek ? &bl.lab_img : nullptr);
    if (finite) somm_prepare(fr, d, *bl.somm);

    const ll N = d.N, nb = d.n_basis;
    const std::size_t nbz = static_cast<std::size_t>(nb);
    std::vector<cd> G;
    if (N < kDenseThreshold) {
        Band b{0, N, 0, d.nnz};
        VC t[3];
        band_triple(d, x, b, gr, bl, t);
        G = scatter_dense(d, t, *np);
    } else {
        G.assign(nbz * nbz, cd(0.0, 0.0));
        BandRows R;
        VI all(static_cast<std::size_t>(N));
        std::iota(all.begin(), all.end(), ll(0));
        const VC *coefs[3] = {&d.sigAC, &d.B, &d.sigC};
        for (int s = 0; s < 3; s++) {
            R.Ms[s] = csc_of(d, *coefs[s], all);
            R.carry[s].resize(nbz);
        }
        R.open.assign(nbz, false);
        R.last.assign(nbz, -1);
        for (ll e = 0; e < d.nnz; e++) {
            ll &l = R.last[static_cast<std::size_t>(d.i_of_entry[static_cast<std::size_t>(e)])];
            l = std::max(l, e);
        }
        for (const Band &b : segment_bands(d, finite ? 2 : 1)) {
            if (b.e1 == b.e0) continue;
            VC t[3];
            band_triple(d, x, b, gr, bl, t);
            scatter_band(d, R, b, t, G);
        }
        for (bool o : R.open)
            if (o) decline("a basis row left unfinished by the band scatter");
    }
    if (bl.ek) ek_bracket_correction(d, x, K, gr, bl, G, *np);
    if (finite) contact_correction(fr, d, K, gr, bl, view, G, *np);
    return G;
}

}  // namespace sinusoidal_galerkin
}  // namespace mw
