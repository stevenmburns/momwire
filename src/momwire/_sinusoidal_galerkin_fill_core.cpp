// The Sinusoidal-Galerkin mixed-potential fill and solve (see
// `_sinusoidal_galerkin_fill_core.h`). Each step is `_sinusoidal_mp.py` /
// `sinusoidal_galerkin.py` as written; the kernels are the B-spline core's
// sinusoidal ones (`mw::bspline::*`), which the bs2 executable already links.
//
// The numpy arithmetic is written in numpy's own order (measured on this
// numpy, as the sinusoidal fill core records): a complex product is
// `np_cmul` in the written operand order with a float promoted to (x, 0);
// a complex quotient `np_cdiv`; `np.exp` of a complex is libm's cexp;
// `einsum("ij,ij->i")` over three is (x0 y0 + x2 y2) + x1 y1 (contracted
// on arm64, `MW_NUMPY_CONTRACTS`); `.sum(-1)`
// over three (p0 + p1) + p2; the three-operand
// complex einsum of `_pair_moments_two_rules_batch` accumulates its (q, r)
// terms from zero, q outer and r inner, each term `(a b) c` with the
// unfused complex product on x86-64 and the contracted one on arm64
// (measured against numpy 2.5's einsum). No product
// here depends on numpy's temporary elision, which not every interpreter
// performs: the one that did is spelled out of it in Python. An
// expression of Python scalars alone (`1j * k * eta`) is CPython's
// `_Py_c_prod` / `_Py_c_quot`.

#if defined(_MSC_VER) && !defined(__clang__)
#pragma float_control(precise, on)
#pragma fp_contract(off)
#endif

#include "_sinusoidal_galerkin_fill_core.h"

#include "_sinusoidal_galerkin_crossing_core.h"
#include "_sinusoidal_galerkin_direct_core.h"

#include "_bspline_core.h"
#include "_bspline_fill_core.h"  // the graded remainder pairs (bs2's)
#include "_somm_grid_core.h"
#include "_somm_pairs_owned_core.h"
#include "_somm_remainder_core.h"
#include "_sinusoidal_series_inline.h"  // `sin_minus_arg`
#include "_razor_front_core.h"  // `leggauss`: `_quadrature.leggauss`'s frozen table
#include "_fma_inline.h"
#include "_libm_complex_inline.h"  // libm's cexp, `MW_NUMPY_CONTRACTS`
#include "_somm_below_remainder_inline.h"  // the below remainder's projection (momwire#1718 S11a)

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <map>
#include <memory>
#include <set>
#include <string>
#include <tuple>
#include <type_traits>
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

[[noreturn]] void decline(const std::string &m) { throw Declined(m); }

inline cd cm(cd a, cd b) { return rp::np_cmul(a, b); }
inline cd rc(double x, cd b) { return cm(cd(x, 0.0), b); }  // float * complex
inline cd cr(cd a, double x) { return cm(a, cd(x, 0.0)); }  // complex * float
inline cd cdr(cd a, double x) { return rp::np_cdiv(a, cd(x, 0.0)); }
inline cd cadd(cd a, cd b) { return cd(a.real() + b.real(), a.imag() + b.imag()); }
inline cd csub(cd a, cd b) { return cd(a.real() - b.real(), a.imag() - b.imag()); }
inline cd radd(double x, cd b) { return cd(x + b.real(), 0.0 + b.imag()); }   // float + complex
inline cd csubr(cd a, double x) { return cd(a.real() - x, a.imag() - 0.0); }  // complex - float
inline cd caddr(cd a, double x) { return cd(a.real() + x, a.imag() + 0.0); }  // complex + float
inline cd cneg(cd a) { return cd(-a.real(), -a.imag()); }

// CPython 3.12's `_Py_c_prod` / `_Py_c_quot` (Smith's method)
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
// numpy's einsum inner product of complex operands (`sum_of_products`),
// plain C, so its bits are numpy's build's (`MW_NUMPY_CONTRACTS`): unfused
// on x86-64; on arm64 each part contracted, `a.r * b.r - a.i * b.i` into
// fma(a.r, b.r, -(a.i b.i)) and `a.r * b.i + a.i * b.r` into
// fma(a.r, b.i, a.i b.r) (measured on macOS 15 arm64, numpy 2.5.4: 12 of
// 12 "pnq,nqr,Pnr->pPn" and 7 of 7 "ep,epP,eP->e" outputs, momwire#1718)
inline cd e_mul(cd a, cd b) {
#if MW_NUMPY_CONTRACTS
    return cd(mw_fma::exact(a.real(), b.real(), -(a.imag() * b.imag())),
              mw_fma::exact(a.real(), b.imag(), a.imag() * b.real()));
#else
    return cd(a.real() * b.real() - a.imag() * b.imag(), a.real() * b.imag() + a.imag() * b.real());
#endif
}

// numpy's NaN-propagating maximum, and `np.clip`
inline double npmax(double a, double b) { return (a != a) ? a : (b != b) ? b : (a >= b ? a : b); }
inline double npmin(double a, double b) { return (a != a) ? a : (b != b) ? b : (a <= b ? a : b); }

inline double norm3(const double *v) { return std::sqrt((v[0] * v[0] + v[1] * v[1]) + v[2] * v[2]); }
// `einsum("ij,ij->i")` over three: (x0 y0 + x2 y2) + x1 y1 on x86-64; on
// arm64 fma(x2, y2, x0 y0) + x1 y1 (4000 of 4000 measured, momwire#1718)
inline double edot3(const double *a, const double *b) {
#if MW_NUMPY_CONTRACTS
    return mw_fma::exact(a[2], b[2], a[0] * b[0]) + a[1] * b[1];
#else
    return (a[0] * b[0] + a[2] * b[2]) + a[1] * b[1];
#endif
}
// `einsum("m,mjn->jn")`'s step: acc + a b, fused on arm64 (12 of 12)
inline double e_acc(double acc, double a, double b) {
#if MW_NUMPY_CONTRACTS
    return mw_fma::exact(a, b, acc);
#else
    return acc + a * b;
#endif
}

// --- constants (`_sinusoidal_mp`, `bspline`) ---------------------------------

constexpr double NEAR_RATIO = 2.0;
constexpr int64_t PARALLEL_N_T = 24;
constexpr int64_t PARALLEL_N_XI = 8;
constexpr int64_t GRADED_N_PER_PANEL = 8;
constexpr int64_t NEAR_N_QP = 32;
constexpr double PAIR_TOL = 1e-9;  // `classify_pairs`' tol
constexpr int64_t WINDOW_BYTES = int64_t(128) << 20;
constexpr int64_t N_SHAPES = 3;
constexpr double DIRECT_SIGN = -1.0;
constexpr int64_t DEFAULT_N_QP_PAIR = 8;
const std::vector<std::pair<double, int64_t>> &default_ladder() {  // `DEFAULT_PAIR_ORDER_LADDER`
    static const std::vector<std::pair<double, int64_t>> l{{16.0, 4}};
    return l;
}
// numpy's iterator buffer, in elements (NPY_BUFSIZE)
constexpr std::size_t EINSUM_BUFSIZE = 8192;

// --- arrays over std containers ------------------------------------------------

template <class C, class T = std::remove_const_t<std::remove_pointer_t<decltype(std::declval<const C &>().data())>>>
nd::array_t<T, nd::array::c_style | nd::array::forcecast> borrow(const C &v, std::vector<nd::ssize_t> shape) {
    nd::Buffer b;
    b.ptr = const_cast<T *>(v.data());
    b.ndim = static_cast<int>(shape.size());
    nd::ssize_t acc = sizeof(T);
    for (int q = b.ndim - 1; q >= 0; q--) {
        b.shape[q] = shape[static_cast<std::size_t>(q)];
        b.strides[q] = acc;
        acc *= shape[static_cast<std::size_t>(q)];
    }
    b.writeable = false;
    return nd::array_t<T, nd::array::c_style | nd::array::forcecast>::borrow(b);
}

// The kernels' `object` arguments (an optional array): a host whose handles
// are `nd::Buffer`s, installed for the duration of a kernel call.
class BufferHost : public nd::Host {
  public:
    nd::Buffer cast_array(const void *handle, nd::dtype type, int flags) override {
        (void)type;
        (void)flags;
        return *static_cast<const nd::Buffer *>(handle);
    }
};

nd::Buffer buffer_of(const std::vector<double> &v) {
    nd::Buffer b;
    b.ptr = const_cast<double *>(v.data());
    b.ndim = 1;
    b.shape[0] = static_cast<nd::ssize_t>(v.size());
    b.strides[0] = sizeof(double);
    b.writeable = false;
    return b;
}

// `_gl01(n)`: Gauss-Legendre on [0, 1]
void gl01(int64_t n, std::vector<double> &t, std::vector<double> &w) {
    std::vector<double> x, ww;
    if (!razor::leggauss(n, x, ww)) decline("an untabled Gauss order");
    t.resize(x.size());
    w.resize(x.size());
    for (std::size_t i = 0; i < x.size(); i++) {
        t[i] = 0.5 * (x[i] + 1.0);
        w[i] = 0.5 * ww[i];
    }
}

void leggauss(int64_t n, std::vector<double> &x, std::vector<double> &w) {
    if (!razor::leggauss(n, x, w)) decline("an untabled Gauss order");
}

// `_ladder_arrays(n_qp, ladder)`
struct LadderArrays {
    std::vector<double> t, w, ratio;
    std::vector<int64_t> n;
};
LadderArrays ladder_arrays(int64_t n_qp, const std::vector<std::pair<double, int64_t>> &ladder) {
    LadderArrays L;
    L.n.push_back(n_qp);
    L.ratio.push_back(0.0);
    for (const auto &r : ladder) {
        L.n.push_back(r.second);
        L.ratio.push_back(r.first);
    }
    for (int64_t o : L.n) {
        std::vector<double> t, w;
        gl01(o, t, w);
        L.t.insert(L.t.end(), t.begin(), t.end());
        L.w.insert(L.w.end(), w.begin(), w.end());
    }
    return L;
}

// --- `basis_csr(seg_view, k)` -------------------------------------------------

struct Csr {
    std::vector<int64_t> starts, jbasis;
    std::vector<cd> coef, dcoef;  // (nnz, 3) C order
};

Csr basis_csr(const SegView &v, double k) {
    Csr c;
    c.starts.assign(v.starts.begin(), v.starts.end());
    c.jbasis.assign(v.jbasis.begin(), v.jbasis.end());
    const std::size_t nnz = v.jbasis.size();
    c.coef.resize(3 * nnz);
    c.dcoef.resize(3 * nnz);
    const cd kk(k, 0.0), mk(-k, 0.0);
    for (std::size_t e = 0; e < nnz; e++) {
        const cd sig(static_cast<double>(v.sigma[e]), 0.0);
        const cd c0 = cm(sig, v.AC[e]);
        const cd c1 = v.B[e];
        const cd c2 = cm(sig, v.C[e]);
        c.coef[3 * e] = c0;
        c.coef[3 * e + 1] = c1;
        c.coef[3 * e + 2] = c2;
        // `dshape_coefs`: k c1, -k c2, k c1
        c.dcoef[3 * e] = cm(kk, c1);
        c.dcoef[3 * e + 1] = cm(mk, c2);
        c.dcoef[3 * e + 2] = cm(kk, c1);
    }
    return c;
}

// --- the segment tables --------------------------------------------------------

struct Segs {
    std::size_t n = 0;
    std::vector<double> c, t, h, a;  // (n, 3), (n, 3), (n), (n)
    std::vector<double> l, r;        // `segment_ends`
};

void segment_ends(Segs &s) {
    s.l.resize(3 * s.n);
    s.r.resize(3 * s.n);
    for (std::size_t i = 0; i < s.n; i++) {
        const double half = 0.5 * s.h[i];
        for (int d = 0; d < 3; d++) {
            const double ht = half * s.t[3 * i + d];
            s.l[3 * i + d] = s.c[3 * i + d] - ht;
            s.r[3 * i + d] = s.c[3 * i + d] + ht;
        }
    }
}

// --- the near pairs ------------------------------------------------------------

struct Pairs {
    std::vector<int64_t> ii, jj;
    std::vector<bool> parallel, touching;
    std::vector<int> end_i, end_j;
};

// `classify_pairs(c_i, t_i, h_i, c_j, t_j, h_j)` over observer rows
// [i0, i1) of `o` against every row of `s`. The candidates are the strict
// test's own set (the cKDTree query at `reach` offers a superset of it; the
// route census checked the corpora for a pair it misses).
Pairs classify_pairs(const Segs &o, std::size_t i0, std::size_t i1, const Segs &s) {
    Pairs p;
    double hmax_i = -HUGE_VAL, hmax_j = -HUGE_VAL;
    for (std::size_t i = i0; i < i1; i++) hmax_i = npmax(hmax_i, o.h[i]);
    for (std::size_t j = 0; j < s.n; j++) hmax_j = npmax(hmax_j, s.h[j]);
    const double reach = NEAR_RATIO * std::max(hmax_i, hmax_j);
    // a conservative cull: a pair whose centres are farther apart than the
    // reach on any axis is past the strict test too
    const double cull = reach * (1.0 + 1e-6);
    for (std::size_t i = i0; i < i1; i++) {
        const double *ci = &o.c[3 * i];
        for (std::size_t j = 0; j < s.n; j++) {
            const double *cj = &s.c[3 * j];
            const double d3[3] = {ci[0] - cj[0], ci[1] - cj[1], ci[2] - cj[2]};
            if (std::fabs(d3[0]) > cull || std::fabs(d3[1]) > cull || std::fabs(d3[2]) > cull) continue;
            const double dist = norm3(d3);
            if (dist < NEAR_RATIO * npmax(o.h[i], s.h[j])) {
                p.ii.push_back(static_cast<int64_t>(i - i0));
                p.jj.push_back(static_cast<int64_t>(j));
            }
        }
    }
    const std::size_t n = p.ii.size();
    p.parallel.resize(n);
    p.touching.resize(n);
    p.end_i.assign(n, -1);
    p.end_j.assign(n, -1);
    for (std::size_t q = 0; q < n; q++) {
        const std::size_t i = i0 + static_cast<std::size_t>(p.ii[q]), j = static_cast<std::size_t>(p.jj[q]);
        const double dot = edot3(&o.t[3 * i], &s.t[3 * j]);
        p.parallel[q] = std::fabs(std::fabs(dot) - 1.0) <= PAIR_TOL;
        const double scale = npmax(o.h[i], s.h[j]);
        bool close[2][2];
        bool any = false;
        for (int a = 0; a < 2; a++)
            for (int b = 0; b < 2; b++) {
                const double *ea = a == 0 ? &o.l[3 * i] : &o.r[3 * i];
                const double *eb = b == 0 ? &s.l[3 * j] : &s.r[3 * j];
                const double d3[3] = {ea[0] - eb[0], ea[1] - eb[1], ea[2] - eb[2]};
                close[a][b] = norm3(d3) <= PAIR_TOL * scale;
                any = any || close[a][b];
            }
        p.touching[q] = any && !p.parallel[q];
        if (p.touching[q]) {
            p.end_i[q] = (close[0][0] || close[0][1]) ? 0 : 1;
            p.end_j[q] = (close[0][0] || close[1][0]) ? 0 : 1;
        }
    }
    return p;
}

// `graded_endpoint_rule01(eps, n_per_panel, end)`
void graded_endpoint_rule01(double eps, int64_t n_per_panel, int end, std::vector<double> &x,
                            std::vector<double> &w) {
    eps = npmin(npmax(eps, 1e-9), 0.5);  // float(np.clip(eps, 1e-9, 0.5))
    std::vector<double> edges{0.0};
    for (int q = 0; q < 64; q++) {
        const double d = eps * std::pow(2.0, static_cast<double>(q));
        if (d < 1.0) edges.push_back(d);
    }
    edges.push_back(1.0);
    std::vector<double> gx, gw;
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
    if (end == 1)
        for (double &v : x) v = 1.0 - v;
}

// CPython's `round(x, 12)`: the correctly rounded 12-decimal string, read back
double py_round12(double x) {
    if (!std::isfinite(x)) return x;
    char buf[512];
    std::snprintf(buf, sizeof buf, "%.12f", x);
    return std::strtod(buf, nullptr);
}

// The extended kernel's per-pair spec of a product-rule batch: every pair
// eligible (momwire#1368), and the source tube per pair or none.
struct EkBatch {
    bool on = false;
    bool has_b = false;
    std::vector<double> b;  // per pair
};

// `_ek_factor_floored(R, rho, k, b)` at one element (numpy's elementwise
// order, `_bspline_kernels._ek_factor`)
cd ek_factor(double R, double a, double k, bool has_b, double b) {
    if (has_b) R = npmax(R, b);
    const double r2 = R * R;
    const double r4 = r2 * r2;
    const double kr = k * R;
    const double kr2 = kr * kr;
    const double a2 = a * a;
    const double b2 = has_b ? b * b : a2;
    const double a4 = b2 * a2;
    const cd c1 = radd(1.0, cm(cd(0.0, 1.0), cd(kr, 0.0)));
    const cd c2 = csubr(rc(3.0, c1), kr2);
    const double t1 = 0.25 * a4 / r4;
    const double t2 = 0.5 * b2 / r2;
    cd fac = rc(t1, c2);
    fac = csub(fac, rc(t2, c1));
    fac = caddr(fac, 1.0);
    return fac;
}

// `_pair_moments_two_rules_batch(l_i, r_i, l_j, r_j, a2, k, ti, wi, tj, wj, ek)`
// over the batch's pairs `(gi, gj)` (rows of the window's / source tables):
// (3, 3, n) into `out` at the batch's own pair columns `cols`.
void two_rules_batch(const Segs &o, std::size_t i0, const Segs &s, const std::vector<int64_t> &gi,
                     const std::vector<int64_t> &gj, const std::vector<double> &a2, double k,
                     const std::vector<double> &ti, const std::vector<double> &wi, const std::vector<double> &tj,
                     const std::vector<double> &wj, const EkBatch &ek, std::vector<cd> &J,
                     std::size_t n_pairs_total, const std::vector<std::size_t> &cols) {
    const std::size_t n = gi.size(), qi = ti.size(), qj = tj.size();
    // measured against numpy's einsum for a row of at most 8192 / 32 nodes;
    // the rules here have at most 31 panels of 8
    if (qj > 256 || qi > 256) decline("a near-pair rule wider than the einsum replay is measured for");
    const std::size_t chunk = qi * qj <= EINSUM_BUFSIZE ? qi : std::max<std::size_t>(1, EINSUM_BUFSIZE / qj);
    const double four_pi = 4.0 * M_PI;
    const cd mjk = py_mul(cd(-0.0, -1.0), cd(k, 0.0));  // `-1j * k`
    std::vector<cd> ws_i(3 * qi), ws_j(3 * qj), G(qi * qj);
    std::vector<double> pos_i(3 * qi), pos_j(3 * qj);
    for (std::size_t m = 0; m < n; m++) {
        const std::size_t i = i0 + static_cast<std::size_t>(gi[m]), j = static_cast<std::size_t>(gj[m]);
        const double *li = &o.l[3 * i], *ri = &o.r[3 * i], *lj = &s.l[3 * j], *rj = &s.r[3 * j];
        const double di[3] = {ri[0] - li[0], ri[1] - li[1], ri[2] - li[2]};
        const double dj[3] = {rj[0] - lj[0], rj[1] - lj[1], rj[2] - lj[2]};
        const double len_i = norm3(di), len_j = norm3(dj);
        auto side = [&](const double *l, const double *r, double len, const std::vector<double> &t,
                        const std::vector<double> &w, std::vector<double> &pos, std::vector<cd> &ws) {
            const std::size_t q = t.size();
            for (std::size_t a = 0; a < q; a++) {
                const double om = 1.0 - t[a];
                for (int d = 0; d < 3; d++) pos[3 * a + d] = om * l[d] + t[a] * r[d];
                const double wl = w[a] * len;
                const double xi = (t[a] - 0.5) * len;
                const double arg = k * xi;
                const double half = std::sin(0.5 * arg);
                const double S[3] = {1.0, std::sin(arg), -2.0 * half * half};
                for (int p = 0; p < 3; p++) ws[static_cast<std::size_t>(p) * q + a] = cm(cd(wl, 0.0), cd(S[p], 0.0));
            }
        };
        side(li, ri, len_i, ti, wi, pos_i, ws_i);
        side(lj, rj, len_j, tj, wj, pos_j, ws_j);
        const double rho = std::sqrt(a2[m]);
        for (std::size_t a = 0; a < qi; a++)
            for (std::size_t b = 0; b < qj; b++) {
                const double d0 = pos_i[3 * a] - pos_j[3 * b], d1 = pos_i[3 * a + 1] - pos_j[3 * b + 1],
                             d2 = pos_i[3 * a + 2] - pos_j[3 * b + 2];
                const double R = std::sqrt(((d0 * d0 + d1 * d1) + d2 * d2) + a2[m]);
                const cd e = mw_libm_c::cexp(cm(mjk, cd(R, 0.0)));
                cd g = cdr(e, four_pi * R);
                if (ek.on) g = cm(g, ek_factor(R, rho, k, ek.has_b, ek.has_b ? ek.b[m] : 0.0));
                G[a * qj + b] = g;
            }
        // einsum("pnq,nqr,Pnr->pPn", ws_i, G, ws_j): each output's (q, r)
        // terms in order, q outer, from zero -- in runs of `chunk` q rows
        // when the reduction outgrows the iterator's buffer (NPY_BUFSIZE =
        // 8192 elements), each run's sum from zero, then added to the output
        for (int p = 0; p < 3; p++)
            for (int P = 0; P < 3; P++) {
                double ar = 0.0, ai = 0.0;
                for (std::size_t a0 = 0; a0 < qi; a0 += chunk) {
                    double pr = 0.0, pi = 0.0;
                    for (std::size_t a = a0; a < std::min(a0 + chunk, qi); a++) {
                        const cd wa = ws_i[static_cast<std::size_t>(p) * qi + a];
                        for (std::size_t b = 0; b < qj; b++) {
                            const cd t =
                                e_mul(e_mul(wa, G[a * qj + b]), ws_j[static_cast<std::size_t>(P) * qj + b]);
                            pr = t.real() + pr;
                            pi = t.imag() + pi;
                        }
                    }
                    ar = pr + ar;
                    ai = pi + ai;
                }
                J[(static_cast<std::size_t>(p) * 3 + static_cast<std::size_t>(P)) * n_pairs_total + cols[m]] = cd(ar, ai);
            }
    }
}

// --- numpy's complex arithmetic at a complex k (momwire#1718 S11a) -----------
//
// A wholly-buried deck's fill runs at k_m, a CPython complex, with eta_m a
// numpy complex128 scalar: an array product is numpy's fused one (`cm`), a
// product of two scalars numpy's scalar math (`smul`), an array quotient
// `CDOUBLE_divide` (`xdiv`); the scalar math and the quotient are numpy's
// plain C, contracted on arm64 (`MW_NUMPY_CONTRACTS`, as the front's copies,
// momwire#1607 B6a); CPython's own scalars take `py_mul` / `py_div`.

cd smul(cd a, cd b) {  // numpy 0-d scalars (`cdouble_ctype_multiply`)
#if MW_NUMPY_CONTRACTS
    return cm(a, b);
#else
    return py_mul(a, b);
#endif
}

cd xdiv(cd a, cd b) {  // `CDOUBLE_divide`
    const double in1r = a.real(), in1i = a.imag(), in2r = b.real(), in2i = b.imag();
    const double in2r_abs = std::fabs(in2r), in2i_abs = std::fabs(in2i);
    if (in2r_abs >= in2i_abs) {
        if (in2r_abs == 0 && in2i_abs == 0) return cd(in1r / in2r_abs, in1i / in2i_abs);
        const double rat = in2i / in2r;
#if MW_NUMPY_CONTRACTS
        const double scl = 1.0 / mw_fma::exact(in2i, rat, in2r);
        return cd(mw_fma::exact(in1i, rat, in1r) * scl, mw_fma::exact(-in1r, rat, in1i) * scl);
#else
        const double scl = 1.0 / (in2r + in2i * rat);
        return cd((in1r + in1i * rat) * scl, (in1i - in1r * rat) * scl);
#endif
    }
    const double rat = in2r / in2i;
#if MW_NUMPY_CONTRACTS
    const double scl = 1.0 / mw_fma::exact(in2r, rat, in2i);
    return cd(mw_fma::exact(in1r, rat, in1i) * scl, mw_fma::exact(in1i, rat, -in1r) * scl);
#else
    const double scl = 1.0 / (in2i + in2r * rat);
    return cd((in1r * rat + in1i) * scl, (in1i * rat - in1r) * scl);
#endif
}

// `shape_values(k, xi)` at a complex k, one arc: `arg = k * xi`, `half =
// np.sin(0.5 * arg)`, (1, sin arg, -2.0 * half * half), each an array op
void shapes_c(cd k, double xi, cd S[3]) {
    const cd arg = cm(k, cd(xi, 0.0));
    const cd half = mw_libm_c::csin(cm(cd(0.5, 0.0), arg));
    S[0] = cd(1.0, 0.0);
    S[1] = mw_libm_c::csin(arg);
    S[2] = cm(cm(cd(-2.0, 0.0), half), half);
}

// `_pair_moments_two_rules_batch` at a complex k (no extended kernel: a
// wholly-buried deck fills reduced): as `two_rules_batch`, with the shapes
// complex (`shapes_c`, scaled `(w len) * S`), `-1j * k` CPython's product,
// `np.exp` libm's cexp and the quotient by `4 pi R` numpy's.
void two_rules_batch_c(const Segs &o, std::size_t i0, const Segs &s, const std::vector<int64_t> &gi,
                       const std::vector<int64_t> &gj, const std::vector<double> &a2, cd k,
                       const std::vector<double> &ti, const std::vector<double> &wi, const std::vector<double> &tj,
                       const std::vector<double> &wj, std::vector<cd> &J, std::size_t n_pairs_total,
                       const std::vector<std::size_t> &cols) {
    const std::size_t n = gi.size(), qi = ti.size(), qj = tj.size();
    if (qj > 256 || qi > 256) decline("a near-pair rule wider than the einsum replay is measured for");
    const std::size_t chunk = qi * qj <= EINSUM_BUFSIZE ? qi : std::max<std::size_t>(1, EINSUM_BUFSIZE / qj);
    const double four_pi = 4.0 * M_PI;
    const cd mjk = py_mul(cd(-0.0, -1.0), k);  // `-1j * k`
    std::vector<cd> ws_i(3 * qi), ws_j(3 * qj), G(qi * qj);
    std::vector<double> pos_i(3 * qi), pos_j(3 * qj);
    for (std::size_t m = 0; m < n; m++) {
        const std::size_t i = i0 + static_cast<std::size_t>(gi[m]), j = static_cast<std::size_t>(gj[m]);
        const double *li = &o.l[3 * i], *ri = &o.r[3 * i], *lj = &s.l[3 * j], *rj = &s.r[3 * j];
        const double di[3] = {ri[0] - li[0], ri[1] - li[1], ri[2] - li[2]};
        const double dj[3] = {rj[0] - lj[0], rj[1] - lj[1], rj[2] - lj[2]};
        const double len_i = norm3(di), len_j = norm3(dj);
        auto side = [&](const double *l, const double *r, double len, const std::vector<double> &t,
                        const std::vector<double> &w, std::vector<double> &pos, std::vector<cd> &ws) {
            const std::size_t q = t.size();
            for (std::size_t a = 0; a < q; a++) {
                const double om = 1.0 - t[a];
                for (int d = 0; d < 3; d++) pos[3 * a + d] = om * l[d] + t[a] * r[d];
                const double wl = w[a] * len;
                const double xi = (t[a] - 0.5) * len;
                cd S[3];
                shapes_c(k, xi, S);
                for (int p = 0; p < 3; p++) ws[static_cast<std::size_t>(p) * q + a] = cm(cd(wl, 0.0), S[p]);
            }
        };
        side(li, ri, len_i, ti, wi, pos_i, ws_i);
        side(lj, rj, len_j, tj, wj, pos_j, ws_j);
        for (std::size_t a = 0; a < qi; a++)
            for (std::size_t b = 0; b < qj; b++) {
                const double d0 = pos_i[3 * a] - pos_j[3 * b], d1 = pos_i[3 * a + 1] - pos_j[3 * b + 1],
                             d2 = pos_i[3 * a + 2] - pos_j[3 * b + 2];
                const double R = std::sqrt(((d0 * d0 + d1 * d1) + d2 * d2) + a2[m]);
                const cd e = mw_libm_c::cexp(cm(mjk, cd(R, 0.0)));
                G[a * qj + b] = xdiv(e, cd(four_pi * R, 0.0));
            }
        for (int p = 0; p < 3; p++)
            for (int P = 0; P < 3; P++) {
                double ar = 0.0, ai = 0.0;
                for (std::size_t a0 = 0; a0 < qi; a0 += chunk) {
                    double pr = 0.0, pi = 0.0;
                    for (std::size_t a = a0; a < std::min(a0 + chunk, qi); a++) {
                        const cd wa = ws_i[static_cast<std::size_t>(p) * qi + a];
                        for (std::size_t b = 0; b < qj; b++) {
                            const cd t =
                                e_mul(e_mul(wa, G[a * qj + b]), ws_j[static_cast<std::size_t>(P) * qj + b]);
                            pr = t.real() + pr;
                            pi = t.imag() + pi;
                        }
                    }
                    ar = pr + ar;
                    ai = pi + ai;
                }
                J[(static_cast<std::size_t>(p) * 3 + static_cast<std::size_t>(P)) * n_pairs_total + cols[m]] = cd(ar, ai);
            }
    }
}

// `near_pair_moments(c_i, t_i, h_i, c_j, t_j, h_j, a2, k, ek=...)` for the
// window [i0, i1): the pairs and J (3, 3, n_near), C order.
struct NearBlock {
    Pairs pairs;
    std::vector<cd> J;
};

NearBlock near_pair_moments(const Segs &o, std::size_t i0, std::size_t i1, const Segs &s, cd k, bool cplx, bool ek,
                            bool has_b_src, const std::vector<double> &b_src) {
    NearBlock nb;
    nb.pairs = classify_pairs(o, i0, i1, s);
    const Pairs &P = nb.pairs;
    const std::size_t n = P.ii.size();
    nb.J.assign(9 * n, cd(0.0, 0.0));
    if (n == 0) return nb;
    // a2_row = a_row[i0:i1] ** 2
    std::vector<double> a2_row(i1 - i0);
    for (std::size_t i = i0; i < i1; i++) a2_row[i - i0] = o.a[i] * o.a[i];
    // the parallel pairs, through the kernel
    std::vector<std::size_t> par;
    for (std::size_t q = 0; q < n; q++)
        if (P.parallel[q]) par.push_back(q);
    if (!par.empty()) {
        const std::size_t m = par.size();
        std::vector<double> ci(3 * m), ti(3 * m), hi(m), cj(3 * m), tj(3 * m), hj(m), a2(m), a_ek, b_ek;
        for (std::size_t r = 0; r < m; r++) {
            const std::size_t q = par[r];
            const std::size_t i = i0 + static_cast<std::size_t>(P.ii[q]), j = static_cast<std::size_t>(P.jj[q]);
            for (int d = 0; d < 3; d++) {
                ci[3 * r + d] = o.c[3 * i + d];
                ti[3 * r + d] = o.t[3 * i + d];
                cj[3 * r + d] = s.c[3 * j + d];
                tj[3 * r + d] = s.t[3 * j + d];
            }
            hi[r] = o.h[i];
            hj[r] = s.h[j];
            a2[r] = a2_row[static_cast<std::size_t>(P.ii[q])];
            if (ek) {
                a_ek.push_back(std::sqrt(a2[r]));
                if (has_b_src) b_ek.push_back(b_src[j]);
            }
        }
        std::vector<double> gt, gw, gx, gwx;
        leggauss(PARALLEL_N_T, gt, gw);
        leggauss(PARALLEL_N_XI, gx, gwx);
        const nd::ssize_t M = static_cast<nd::ssize_t>(m);
        BufferHost host;
        nd::HostScope scope(host);
        const nd::Buffer ba = buffer_of(a_ek), bb = buffer_of(b_ek);
        const nd::object oa = ek ? nd::object(&ba) : nd::none();
        const nd::object ob = ek && has_b_src ? nd::object(&bb) : nd::none();
        const nd::array_t<cd> out = bspline::parallel_pair_moments_sinusoidal(
            borrow(ci, {M, 3}), borrow(ti, {M, 3}), borrow(hi, {M}), borrow(cj, {M, 3}), borrow(tj, {M, 3}),
            borrow(hj, {M}), borrow(a2, {M}), k, borrow(gt, {static_cast<nd::ssize_t>(gt.size())}),
            borrow(gw, {static_cast<nd::ssize_t>(gw.size())}), borrow(gx, {static_cast<nd::ssize_t>(gx.size())}),
            borrow(gwx, {static_cast<nd::ssize_t>(gwx.size())}), oa, ob);
        const cd *op = out.data();
        for (std::size_t pq = 0; pq < 9; pq++)
            for (std::size_t r = 0; r < m; r++) nb.J[pq * n + par[r]] = op[pq * m + r];
    }
    // the rest, grouped by rule in first-appearance order
    std::vector<double> t01, w01;
    gl01(NEAR_N_QP, t01, w01);
    struct Key {
        bool touching;
        int ei, ej;
        double eps_i, eps_j;
    };
    std::vector<Key> keys;
    std::vector<std::vector<std::size_t>> groups;
    for (std::size_t q = 0; q < n; q++) {
        if (P.parallel[q]) continue;
        Key key{false, 0, 0, 0.0, 0.0};
        if (P.touching[q]) {
            const std::size_t i = i0 + static_cast<std::size_t>(P.ii[q]), j = static_cast<std::size_t>(P.jj[q]);
            const double sa = std::sqrt(a2_row[static_cast<std::size_t>(P.ii[q])]);
            key = Key{true, P.end_i[q], P.end_j[q], py_round12(sa / o.h[i]), py_round12(sa / s.h[j])};
        }
        std::size_t g = 0;
        for (; g < keys.size(); g++) {
            const Key &k2 = keys[g];
            if (k2.touching != key.touching) continue;
            if (!key.touching) break;
            if (k2.ei == key.ei && k2.ej == key.ej && k2.eps_i == key.eps_i && k2.eps_j == key.eps_j) break;
        }
        if (g == keys.size()) {
            keys.push_back(key);
            groups.emplace_back();
        }
        groups[g].push_back(q);
    }
    for (std::size_t g = 0; g < keys.size(); g++) {
        std::vector<double> ti, wi, tj, wj;
        if (!keys[g].touching) {
            ti = tj = t01;
            wi = wj = w01;
        } else {
            graded_endpoint_rule01(keys[g].eps_i, GRADED_N_PER_PANEL, keys[g].ei, ti, wi);
            graded_endpoint_rule01(keys[g].eps_j, GRADED_N_PER_PANEL, keys[g].ej, tj, wj);
        }
        std::vector<int64_t> gi, gj;
        std::vector<double> a2;
        EkBatch eb;
        eb.on = ek;
        eb.has_b = ek && has_b_src;
        for (std::size_t q : groups[g]) {
            gi.push_back(P.ii[q]);
            gj.push_back(P.jj[q]);
            a2.push_back(a2_row[static_cast<std::size_t>(P.ii[q])]);
            if (eb.has_b) eb.b.push_back(b_src[static_cast<std::size_t>(P.jj[q])]);
        }
        if (cplx)
            two_rules_batch_c(o, i0, s, gi, gj, a2, k, ti, wi, tj, wj, nb.J, n, groups[g]);
        else
            two_rules_batch(o, i0, s, gi, gj, a2, k.real(), ti, wi, tj, wj, eb, nb.J, n, groups[g]);
    }
    return nb;
}

// The constructor's arguments as the solver reads them, per segment.
struct Deck {
    Segs segs;
    bool uniform = true;
    double radius = 0;
};

Deck deck_of(const Front &fr) {
    const Geometry &g = fr.geometry;
    Deck d;
    Segs &s = d.segs;
    s.n = static_cast<std::size_t>(g.n_segs);
    s.c.resize(3 * s.n);
    s.t.resize(3 * s.n);
    s.h = g.seg_h;
    s.a.resize(s.n);
    for (std::size_t i = 0; i < s.n; i++)
        for (int q = 0; q < 3; q++) {
            s.c[3 * i + q] = g.seg_centers[i][static_cast<std::size_t>(q)];
            s.t[3 * i + q] = g.seg_tangents[i][static_cast<std::size_t>(q)];
        }
    d.uniform = fr.uniform_radius;
    if (fr.uniform_radius) {
        d.radius = fr.radius_per_wire.empty() ? 0.0 : fr.radius_per_wire[0];
        std::fill(s.a.begin(), s.a.end(), d.radius);
    } else {
        // `_seg_radius(geom)`: each wire's radius over its segments
        for (std::size_t w = 0; w < g.wire_first.size(); w++)
            for (ll i = g.wire_first[w]; i <= g.wire_last[w]; i++) s.a[static_cast<std::size_t>(i)] = fr.radius_per_wire[w];
    }
    segment_ends(s);
    return d;
}

// `WindowFill(...).accumulate(c, t, h, a_row, src_c, src_t, h, scale, ek)` into
// G (F order): the deck's segments `s` as the observers, `src` the sources
// (the same segments, or their images). `scale` is the block's Python scalar
// (a float as (x, 0)).
// The block's walk at the window's own (k, c_a * scale, c_phi * scale), the
// products as the caller's operand types make them: `cplx` is a complex k (a
// wholly-buried deck's k_m, momwire#1718 S11a), whose near pairs take the
// complex product rule.
// A class block of a crossing deck (momwire#1718 S11b): the pair order
// and ladder (bspline's `BURIED_*` in the medium), and the GLOBAL segment ids
// of the observer and source lists (`obs_idx` / `src_idx`, identity when null)
struct WinOpts {
    int64_t n_qp = DEFAULT_N_QP_PAIR;
    const std::vector<std::pair<double, int64_t>> *ladder = nullptr;  // null: `DEFAULT_PAIR_ORDER_LADDER`
    const std::vector<int64_t> *obs_idx = nullptr, *src_idx = nullptr;
};

void window_fill_at(std::vector<cd> &G, std::size_t n_basis, const Csr &csr, const Segs &s, const Segs &src, cd k,
                    bool cplx, cd c_a_s, cd c_phi_s, bool ek, bool has_b_src, const std::vector<double> &b_src,
                    const WinOpts &wo = WinOpts());

void window_fill(std::vector<cd> &G, std::size_t n_basis, const Csr &csr, const Segs &s, const Segs &src,
                 double k, double eta, cd scale, bool ek, bool has_b_src, const std::vector<double> &b_src,
                 const WinOpts &wo = WinOpts()) {
    const cd c_a = py_mul(py_mul(cd(0.0, 1.0), cd(k, 0.0)), cd(eta, 0.0));  // 1j * k * eta
    const cd c_phi = py_div(cd(eta, 0.0), py_mul(cd(0.0, 1.0), cd(k, 0.0)));  // eta / (1j * k)
    const cd c_a_s = py_mul(c_a, scale);
    const cd c_phi_s = py_mul(c_phi, scale);
    window_fill_at(G, n_basis, csr, s, src, cd(k, 0.0), false, c_a_s, c_phi_s, ek, has_b_src, b_src, wo);
}

void window_fill_at(std::vector<cd> &G, std::size_t n_basis, const Csr &csr, const Segs &s, const Segs &src, cd k,
                    bool cplx, cd c_a_s, cd c_phi_s, bool ek, bool has_b_src, const std::vector<double> &b_src,
                    const WinOpts &wo) {
    const std::size_t n_obs = s.n, n_src = src.n;
    const int64_t rows = std::max<int64_t>(
        1, WINDOW_BYTES / (N_SHAPES * N_SHAPES * 16 * std::max<int64_t>(static_cast<int64_t>(n_src), 1)));
    const LadderArrays L = ladder_arrays(wo.n_qp, wo.ladder ? *wo.ladder : default_ladder());
    const nd::ssize_t NS = static_cast<nd::ssize_t>(n_src);
    const CF tt = borrow(L.t, {static_cast<nd::ssize_t>(L.t.size())}),
             tw = borrow(L.w, {static_cast<nd::ssize_t>(L.w.size())}),
             tr = borrow(L.ratio, {static_cast<nd::ssize_t>(L.ratio.size())});
    const CI tn = borrow(L.n, {static_cast<nd::ssize_t>(L.n.size())});
    const CF sl_s = borrow(src.l, {NS, 3}), sr_s = borrow(src.r, {NS, 3});
    std::vector<int64_t> every(n_src, 0);
    const CI gj = borrow(every, {NS});
    std::vector<int64_t> all_cols(n_src);
    for (std::size_t j = 0; j < n_src; j++) all_cols[j] = wo.src_idx ? (*wo.src_idx)[j] : static_cast<int64_t>(j);
    const CI cols = borrow(all_cols, {NS});
    const CI st = borrow(csr.starts, {static_cast<nd::ssize_t>(csr.starts.size())}),
             jb = borrow(csr.jbasis, {static_cast<nd::ssize_t>(csr.jbasis.size())});
    const nd::ssize_t nnz = static_cast<nd::ssize_t>(csr.jbasis.size());
    const CC cf = borrow(csr.coef, {nnz, 3}), dcf = borrow(csr.dcoef, {nnz, 3});
    const CF tc = borrow(src.t, {NS, 3});
    // G as the assembler's writeable Z, F order
    nd::Buffer zb;
    zb.ptr = G.data();
    zb.ndim = 2;
    zb.shape[0] = zb.shape[1] = static_cast<nd::ssize_t>(n_basis);
    zb.strides[0] = sizeof(cd);
    zb.strides[1] = static_cast<nd::ssize_t>(sizeof(cd) * n_basis);
    zb.writeable = true;
    const nd::array_t<cd> Z = nd::array_t<cd>::borrow(zb);
    const nd::Buffer b_buf = buffer_of(b_src);
    for (std::size_t i0 = 0; i0 < n_obs; i0 += static_cast<std::size_t>(rows)) {
        const std::size_t i1 = std::min(i0 + static_cast<std::size_t>(rows), n_obs);
        const std::size_t nw = i1 - i0;
        // `_window_moments`: one call per run of equal observer radius
        std::vector<cd> J(9 * nw * n_src);
        std::size_t r0 = i0;
        while (r0 < i1) {
            std::size_t r1 = r0 + 1;
            while (r1 < i1 && s.a[r1] == s.a[r0]) r1++;
            const nd::ssize_t NR = static_cast<nd::ssize_t>(r1 - r0);
            std::vector<double> l(s.l.begin() + static_cast<std::ptrdiff_t>(3 * r0),
                                  s.l.begin() + static_cast<std::ptrdiff_t>(3 * r1)),
                r(s.r.begin() + static_cast<std::ptrdiff_t>(3 * r0), s.r.begin() + static_cast<std::ptrdiff_t>(3 * r1));
            const double a = s.a[r0];
            const double a2 = a * a;  // float(a) ** 2
            nd::array_t<cd> out;
            if (ek) {
                std::vector<int64_t> gi(r1 - r0, 0);
                BufferHost host;
                nd::HostScope scope(host);
                out = bspline::seg_seg_full_moments_sinusoidal_tiered_ek(
                    borrow(l, {NR, 3}), borrow(r, {NR, 3}), sl_s, sr_s, a2, k.real(), tt, tw, tn, tr, borrow(gi, {NR}), gj,
                    a, has_b_src ? nd::object(&b_buf) : nd::none());
            } else {
                out = bspline::seg_seg_full_moments_sinusoidal_tiered(borrow(l, {NR, 3}), borrow(r, {NR, 3}), sl_s,
                                                                      sr_s, a2, k, tt, tw, tn, tr, false);
            }
            // concatenate along axis 2
            const cd *op = out.data();
            const std::size_t nr = r1 - r0;
            for (std::size_t pq = 0; pq < 9; pq++)
                for (std::size_t m = 0; m < nr; m++)
                    std::copy(op + (pq * nr + m) * n_src, op + (pq * nr + m + 1) * n_src,
                              J.begin() + static_cast<std::ptrdiff_t>((pq * nw + (r0 - i0) + m) * n_src));
            r0 = r1;
        }
        // the near pairs overwrite their moments
        const NearBlock nb = near_pair_moments(s, i0, i1, src, k, cplx, ek, has_b_src, b_src);
        const std::size_t nn = nb.pairs.ii.size();
        for (std::size_t q = 0; q < nn; q++) {
            const std::size_t m = static_cast<std::size_t>(nb.pairs.ii[q]), j = static_cast<std::size_t>(nb.pairs.jj[q]);
            for (std::size_t pq = 0; pq < 9; pq++) J[(pq * nw + m) * n_src + j] = nb.J[pq * nn + q];
        }
        std::vector<int64_t> rows_seg(nw);
        for (std::size_t m = 0; m < nw; m++) rows_seg[m] = wo.obs_idx ? (*wo.obs_idx)[i0 + m] : static_cast<int64_t>(i0 + m);
        std::vector<double> t_rows(s.t.begin() + static_cast<std::ptrdiff_t>(3 * i0),
                                   s.t.begin() + static_cast<std::ptrdiff_t>(3 * i1));
        const nd::ssize_t NW = static_cast<nd::ssize_t>(nw);
        bspline::assemble_Z_sinusoidal_windowed(borrow(J, {3, 3, NW, NS}), borrow(rows_seg, {NW}), cols, st, jb, cf,
                                                dcf, borrow(t_rows, {NW, 3}), tc, c_a_s, c_phi_s, nd::none(),
                                                nd::none(), Z, 0);
    }
}

// `_loading_integrals(k, h)` at one segment, real k: (∫(cos kξ − 1),
// ∫sin² kξ, ∫(cos kξ − 1)²), numpy's elementwise order
void loading_integrals(double k, double h, double &i_g, double &i_ss, double &i_gg) {
    const double x = (0.5 * k) * h;
    const double s1 = sin_minus_arg(x);
    const double s2 = sin_minus_arg(2.0 * x);
    i_g = (2.0 / k) * s1;
    i_ss = -s2 / (2.0 * k);
    const double x2 = x * x;
    // Horner over x², highest order first; each coefficient as CPython
    // computes `(-1) ** m * (4.0 ** m - 4.0) / math.factorial(2 * m + 1)`
    static const double fact[16] = {1.0, 1.0, 2.0, 6.0, 24.0, 120.0, 720.0, 5040.0, 40320.0, 362880.0,
                                    3628800.0, 39916800.0, 479001600.0, 6227020800.0, 87178291200.0,
                                    1307674368000.0};
    double series = 0.0;
    for (int m = 7; m > 1; m--) {
        const double sign = m % 2 == 0 ? 1.0 : -1.0;
        const double coef = sign * (std::pow(4.0, static_cast<double>(m)) - 4.0) / fact[2 * m + 1];
        series = series * x2 + coef;
    }
    series = series * x2 * x2 * x;
    i_gg = (std::fabs(x) < 0.25 ? series : 0.5 * s2 - 4.0 * s1) / k;
}

// `_apply_loading(G, geom, seg_view, k)`: the Galerkin overlap of the series
// wire impedance, `np.subtract.at` over every ordered pair of support entries
// sharing a segment (`_shared_segment_pairs`' order), G F order.
void apply_loading(std::vector<cd> &G, std::size_t n_basis, const Front &fr, const std::vector<cd> &z_wire) {
    const Geometry &g = fr.geometry;
    const SegView &v = fr.basis;
    const double k = fr.k;
    std::vector<cd> z_seg(static_cast<std::size_t>(g.n_segs));
    for (std::size_t w = 0; w < g.wire_first.size(); w++)
        for (ll i = g.wire_first[w]; i <= g.wire_last[w]; i++) z_seg[static_cast<std::size_t>(i)] = z_wire.at(w);
    const std::size_t nnz = v.jbasis.size();
    std::vector<cd> U(nnz), R(nnz);
    for (std::size_t e = 0; e < nnz; e++) {
        const cd sg(static_cast<double>(v.sigma[e]), 0.0);
        U[e] = cm(sg, v.AC[e]);
        R[e] = cm(sg, v.C[e]);
    }
    for (std::size_t m = 0; m < static_cast<std::size_t>(g.n_segs); m++) {
        const double h = g.seg_h[m];
        double i_g, i_ss, i_gg;
        loading_integrals(k, h, i_g, i_ss, i_gg);
        for (ll e = v.starts[m]; e < v.starts[m + 1]; e++)
            for (ll f = v.starts[m]; f < v.starts[m + 1]; f++) {
                const std::size_t a = static_cast<std::size_t>(e), b = static_cast<std::size_t>(f);
                const cd t1 = cr(cm(U[a], U[b]), h);
                const cd t2 = cr(cadd(cm(U[a], R[b]), cm(R[a], U[b])), i_g);
                const cd t3 = cr(cm(v.B[a], v.B[b]), i_ss);
                const cd t4 = cr(cm(R[a], R[b]), i_gg);
                const cd val = cm(cadd(cadd(cadd(t1, t2), t3), t4), z_seg[m]);
                cd &slot = G[static_cast<std::size_t>(v.jbasis[a]) +
                             static_cast<std::size_t>(v.jbasis[b]) * n_basis];
                slot = csub(slot, val);
            }
    }
}

// --- the Sommerfeld remainder above the plane (`_mp_remainder`, S5b) ---------

// `_sinusoidal_mp.REMAINDER_Q_BYTES`, `REMAINDER_TAYLOR_DEGREE + 1`, bspline's
// `_REMAINDER_GRADED_C` / `_REMAINDER_GRADED_EDGE_TOL`
constexpr std::size_t REMAINDER_Q_BYTES = std::size_t(256) << 20;
constexpr int REMAINDER_D1 = 13;
constexpr double REMAINDER_GRADED_C = 4.0;
constexpr double REMAINDER_GRADED_EDGE_TOL = 1e-9;
constexpr double SOLVER_MU = 1.25663706127e-6;  // SinusoidalSolver.mu

struct GridHold {  // `grid_cpp_args(grid)`'s view over a grid
    somm_proj::GridView view;
    std::vector<cd> far;
};

void grid_view(const somm_core::Grid &G0, GridHold &h) {
    const std::size_t n_reg = G0.regions.size();
    if (n_reg > somm_proj::MAX_REGIONS) decline("a Sommerfeld grid layout past the view's regions");
    double r0[somm_proj::MAX_REGIONS], dr[somm_proj::MAX_REGIONS], th0[somm_proj::MAX_REGIONS],
        dth[somm_proj::MAX_REGIONS];
    const cd *vals[somm_proj::MAX_REGIONS];
    std::ptrdiff_t shape0[somm_proj::MAX_REGIONS], nr[somm_proj::MAX_REGIONS], nth[somm_proj::MAX_REGIONS];
    for (std::size_t r = 0; r < n_reg; r++) {
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
    // theta split `math.radians(30.0)`, and the far pack
    h.view = somm_proj::build_grid_view(G0.r1_max, G0.r_break, 30.0 * (M_PI / 180.0), G0.r_near, r0, dr, th0, dth,
                                        n_reg, vals, shape0, nr, nth);
    h.far = somm_core::far_pack(G0);
    somm_proj::set_far(h.view, h.far.data(), static_cast<std::ptrdiff_t>(h.far.size()));
}

// `taylor_shape_coefs(k, h)`: tau (3, D1, n) C order
std::vector<double> taylor_shape_coefs(double k, const std::vector<double> &h) {
    const int D1 = REMAINDER_D1;
    const std::size_t n = h.size();
    double fact[REMAINDER_D1], kp[REMAINDER_D1], s_xi[REMAINDER_D1], c_xi[REMAINDER_D1], one[REMAINDER_D1];
    double f = 1.0;
    for (int m = 0; m < D1; m++) {
        if (m > 0) f = f * static_cast<double>(m);  // math.factorial, exact here
        fact[m] = f;
    }
    for (int m = 0; m < D1; m++) {
        kp[m] = std::pow(k, static_cast<double>(m)) / fact[m];
        const double sgn = (m / 2) % 2 == 0 ? 1.0 : -1.0;
        s_xi[m] = (m % 2 == 1 ? sgn : 0.0) * kp[m];
        c_xi[m] = (m % 2 == 0 && m > 0 ? sgn : 0.0) * kp[m];
        one[m] = m == 0 ? 1.0 : 0.0;
    }
    double binom[REMAINDER_D1][REMAINDER_D1];
    for (int mm = 0; mm < D1; mm++) {
        double c = 1.0;  // math.comb(mm, j), exact here
        for (int j = 0; j < D1; j++) {
            binom[mm][j] = j <= mm ? c : 0.0;
            if (j < mm) c = c * static_cast<double>(mm - j) / static_cast<double>(j + 1);
        }
    }
    const double *series[3] = {one, s_xi, c_xi};
    std::vector<double> out(3 * static_cast<std::size_t>(D1) * n);
    for (std::size_t s = 0; s < n; s++) {
        const double base = -0.5 * h[s];
        double T[REMAINDER_D1][REMAINDER_D1];
        for (int m = 0; m < D1; m++)
            for (int j = 0; j < D1; j++) T[m][j] = binom[m][j] * std::pow(base, static_cast<double>(std::max(m - j, 0)));
        for (int p = 0; p < 3; p++)
            for (int j = 0; j < D1; j++) {
                double acc = 0.0;  // einsum("m,mjn->jn"): from zero, m in order
                for (int m = 0; m < D1; m++) acc = e_acc(acc, series[p][m], T[m][j]);
                out[(static_cast<std::size_t>(p) * D1 + static_cast<std::size_t>(j)) * n + s] = acc;
            }
    }
    return out;
}

// `_mp_remainder(G, geom, seg_view, k, fg)` at a real k: `remainder_Q_above`
// with `scale = REMAINDER_SIGN` (1.0), G F order.
void mp_remainder(std::vector<cd> &G, std::size_t n_basis, const Csr &csr, const Segs &s, double k, cd eps_t,
                  double omega, int64_t q, const nec5::NumpyLinalg &np) {
    const std::size_t N = s.n;
    const double gz = 0.0;  // the NEC-4.2 dialect's interface
    const double r1_max = razor::max_image_distance(s.l.data(), s.r.data(), N, gz);
    const somm_core::Grid &G0 = somm_core::get_grid(eps_t, k, r1_max, omega, SOLVER_MU);
    GridHold gh;
    grid_view(G0, gh);
    std::vector<double> t01, w01;
    gl01(q, t01, w01);
    const std::size_t qn = t01.size();
    // nodes = seg_l + t01 (seg_r - seg_l); W = (h w01) S_p(xi), real
    std::vector<double> nodes(N * qn * 3), W(3 * N * qn);
    for (std::size_t i = 0; i < N; i++)
        for (std::size_t a = 0; a < qn; a++) {
            for (int d = 0; d < 3; d++)
                nodes[(i * qn + a) * 3 + d] = s.l[3 * i + d] + t01[a] * (s.r[3 * i + d] - s.l[3 * i + d]);
            const double xi = (t01[a] - 0.5) * s.h[i];
            const double arg = k * xi;
            const double half = std::sin(0.5 * arg);
            const double S[3] = {1.0, std::sin(arg), -2.0 * half * half};
            const double hw = s.h[i] * w01[a];
            for (int p = 0; p < 3; p++) W[(static_cast<std::size_t>(p) * N + i) * qn + a] = hw * S[p];
        }
    for (const cd &c : csr.coef)
        if (c.imag() != 0.0) decline("complex basis coefficients on the remainder (a complex k)");
    // `basis_wings(starts, jbasis, coef, n_basis)`
    const std::size_t nnz = csr.jbasis.size();
    std::vector<int64_t> seg_of(nnz);
    for (std::size_t i = 0; i < N; i++)
        for (int64_t e = csr.starts[i]; e < csr.starts[i + 1]; e++) seg_of[static_cast<std::size_t>(e)] = static_cast<int64_t>(i);
    std::vector<std::size_t> order(nnz);
    for (std::size_t e = 0; e < nnz; e++) order[e] = e;
    std::stable_sort(order.begin(), order.end(), [&](std::size_t x, std::size_t y) { return csr.jbasis[x] < csr.jbasis[y]; });
    std::vector<int64_t> counts(n_basis, 0);
    for (std::size_t e = 0; e < nnz; e++) counts[static_cast<std::size_t>(csr.jbasis[e])]++;
    std::vector<int64_t> rows_basis, row_start(n_basis + 1, 0);
    for (std::size_t b = 0; b < n_basis; b++) {
        const int64_t nr = (counts[b] + 2) / 3;
        row_start[b + 1] = row_start[b] + nr;
        for (int64_t r = 0; r < nr; r++) rows_basis.push_back(static_cast<int64_t>(b));
    }
    const std::size_t R = rows_basis.size();
    if (16 * R * R > REMAINDER_Q_BYTES) decline("a remainder block past REMAINDER_Q_BYTES (its banded fold)");
    std::vector<int64_t> loc(R * 3, 0);
    std::vector<double> pl(R * 9, 0.0);
    {
        std::vector<int64_t> pos_in(n_basis, 0);
        for (std::size_t o = 0; o < nnz; o++) {
            const std::size_t e = order[o];
            const std::size_t b = static_cast<std::size_t>(csr.jbasis[e]);
            const int64_t pos = pos_in[b]++;
            const std::size_t row = static_cast<std::size_t>(row_start[b] + pos / 3), slot = static_cast<std::size_t>(pos % 3);
            loc[row * 3 + slot] = seg_of[e];
            for (int p = 0; p < 3; p++) pl[(row * 3 + slot) * 3 + static_cast<std::size_t>(p)] = csr.coef[3 * e + static_cast<std::size_t>(p)].real();
        }
    }
    std::vector<cd> Qr(R * R);
    somm_core::remainder_bspline_Q(gh.view, gz, k, nodes.data(), s.t.data(), W.data(), static_cast<std::ptrdiff_t>(N),
                                   nodes.data(), s.t.data(), W.data(), static_cast<std::ptrdiff_t>(N),
                                   static_cast<std::ptrdiff_t>(qn), 3, loc.data(), pl.data(),
                                   static_cast<std::ptrdiff_t>(R), loc.data(), pl.data(),
                                   static_cast<std::ptrdiff_t>(R), true, 0, Qr.data(), 0);
    // `fold_rows_into(G, Qr, rows_basis, scale)`: onto the bases with a row
    // (every basis on a whole deck; a class block's own, momwire#1718 S11b)
    {
        std::vector<std::size_t> first(n_basis);
        for (std::size_t b = 0; b < n_basis; b++) first[b] = static_cast<std::size_t>(row_start[b]);
        std::vector<std::size_t> bases;
        for (std::size_t b = 0; b < n_basis; b++)
            if (counts[b] > 0) bases.push_back(b);
        std::vector<std::size_t> extra;
        for (std::size_t r = 0; r < R; r++)
            if (static_cast<int64_t>(r) != row_start[static_cast<std::size_t>(rows_basis[r])]) extra.push_back(r);
        if (!extra.empty()) {
            std::vector<cd> rows(extra.size() * R);  // `Qr[extra]`, copied first
            for (std::size_t i = 0; i < extra.size(); i++)
                std::copy(Qr.begin() + static_cast<std::ptrdiff_t>(extra[i] * R),
                          Qr.begin() + static_cast<std::ptrdiff_t>((extra[i] + 1) * R),
                          rows.begin() + static_cast<std::ptrdiff_t>(i * R));
            for (std::size_t i = 0; i < extra.size(); i++) {
                const std::size_t t = first[static_cast<std::size_t>(rows_basis[extra[i]])];
                for (std::size_t j = 0; j < R; j++) Qr[t * R + j] = cadd(Qr[t * R + j], rows[i * R + j]);
            }
            std::vector<cd> cols(extra.size() * R);  // `Qr.T[extra]`
            for (std::size_t i = 0; i < extra.size(); i++)
                for (std::size_t r = 0; r < R; r++) cols[i * R + r] = Qr[r * R + extra[i]];
            for (std::size_t i = 0; i < extra.size(); i++) {
                const std::size_t t = first[static_cast<std::size_t>(rows_basis[extra[i]])];
                for (std::size_t r = 0; r < R; r++) Qr[r * R + t] = cadd(Qr[r * R + t], cols[i * R + r]);
            }
        }
        for (std::size_t i : bases)
            for (std::size_t j : bases) {
                const cd blk = cm(Qr[first[i] * R + first[j]], cd(1.0, 0.0));  // `blk *= scale`
                cd &gij = G[i + j * n_basis];
                gij = cadd(gij, blk);
            }
    }
    // the grazing pairs, on graded panels
    std::vector<int64_t> I, J, Qp;
    bspline::raised_pairs_on(s.l, s.r, static_cast<int64_t>(N), gz, q, REMAINDER_GRADED_C, REMAINDER_GRADED_EDGE_TOL,
                             np, I, J, Qp);
    if (I.empty()) return;
    const std::size_t n = I.size();
    const int D1 = REMAINDER_D1;
    bspline::Geometry bg;
    bg.n_segs_total = static_cast<int64_t>(N);
    bg.seg_l = s.l;
    bg.seg_r = s.r;
    bg.tangents = s.t;
    bg.h_per_seg = s.h;
    // `_remainder_graded.pair_moments(..., D1)` through the bs2 core's exported
    // hook, without the base-order moments it can also return (this route
    // takes them from `remainder_field_proj_owned` below, as Python does):
    // computing them pair by pair was ~90 % of an above-ground screen's fill
    std::vector<cd> dJm, unread;
    bspline::debug_pair_moments(bg, gz, k, G0, D1, I, J, q, dJm, unread, false);
    const std::vector<double> tau = taylor_shape_coefs(k, s.h);
    // dJ = einsum("pmn,nmM,PMn->npP", tau[:, :, I], dJ_mono, tau[:, :, J])
    std::vector<cd> dJ(n * 9);
    for (std::size_t t = 0; t < n; t++) {
        const std::size_t si = static_cast<std::size_t>(I[t]), sj = static_cast<std::size_t>(J[t]);
        for (int p = 0; p < 3; p++)
            for (int P = 0; P < 3; P++) {
                double ar = 0.0, ai = 0.0;
                for (int m = 0; m < D1; m++) {
                    const cd a(tau[(static_cast<std::size_t>(p) * D1 + static_cast<std::size_t>(m)) * N + si], 0.0);
                    for (int M = 0; M < D1; M++) {
                        const cd c(tau[(static_cast<std::size_t>(P) * D1 + static_cast<std::size_t>(M)) * N + sj], 0.0);
                        const cd v = e_mul(e_mul(a, dJm[(t * D1 + static_cast<std::size_t>(m)) * D1 + static_cast<std::size_t>(M)]), c);
                        ar = v.real() + ar;
                        ai = v.imag() + ai;
                    }
                }
                dJ[t * 9 + static_cast<std::size_t>(p) * 3 + static_cast<std::size_t>(P)] = cd(ar, ai);
            }
    }
    // the base-order moments these pairs were filled with: each observer node
    // of I[n] against the q source nodes of J[n] (`remainder_field_proj_owned`)
    const std::size_t S = n * qn * qn;
    std::vector<double> obs(n * qn * 3), t_obs(n * qn * 3), src(S * 3), t_src(S * 3);
    std::vector<int64_t> owner(S);
    for (std::size_t t = 0; t < n; t++) {
        const std::size_t si = static_cast<std::size_t>(I[t]), sj = static_cast<std::size_t>(J[t]);
        for (std::size_t a = 0; a < qn; a++) {
            const std::size_t o = t * qn + a;
            for (int d = 0; d < 3; d++) {
                obs[o * 3 + d] = nodes[(si * qn + a) * 3 + d];
                t_obs[o * 3 + d] = s.t[3 * si + d];
            }
            for (std::size_t b = 0; b < qn; b++) {
                const std::size_t r = o * qn + b;
                owner[r] = static_cast<int64_t>(o);
                for (int d = 0; d < 3; d++) {
                    src[r * 3 + d] = nodes[(sj * qn + b) * 3 + d];
                    t_src[r * 3 + d] = s.t[3 * sj + d];
                }
            }
        }
    }
    std::vector<cd> F(S);
    somm_core::remainder_field_proj_owned(gh.view, gz, k, obs.data(), t_obs.data(), src.data(), t_src.data(),
                                          owner.data(), static_cast<std::ptrdiff_t>(S), F.data(), 0);
    // lo = einsum("pnq,nqr,Pnr->npP", W[:, I, :], F, W[:, J, :]); dJ -= lo
    for (std::size_t t = 0; t < n; t++) {
        const std::size_t si = static_cast<std::size_t>(I[t]), sj = static_cast<std::size_t>(J[t]);
        for (int p = 0; p < 3; p++)
            for (int P = 0; P < 3; P++) {
                double ar = 0.0, ai = 0.0;
                for (std::size_t a = 0; a < qn; a++) {
                    const cd wa(W[(static_cast<std::size_t>(p) * N + si) * qn + a], 0.0);
                    for (std::size_t b = 0; b < qn; b++) {
                        const cd wb(W[(static_cast<std::size_t>(P) * N + sj) * qn + b], 0.0);
                        const cd v = e_mul(e_mul(wa, F[(t * qn + a) * qn + b]), wb);
                        ar = v.real() + ar;
                        ai = v.imag() + ai;
                    }
                }
                cd &x = dJ[t * 9 + static_cast<std::size_t>(p) * 3 + static_cast<std::size_t>(P)];
                x = csub(x, cd(ar, ai));
            }
    }
    // `scatter_pair_moments(G, starts, jbasis, coef, I, J, dJ, scale)`
    for (std::size_t t = 0; t < n; t++) {
        const int64_t a0 = csr.starts[static_cast<std::size_t>(I[t])], a1 = csr.starts[static_cast<std::size_t>(I[t]) + 1];
        const int64_t b0 = csr.starts[static_cast<std::size_t>(J[t])], b1 = csr.starts[static_cast<std::size_t>(J[t]) + 1];
        for (int64_t ei = a0; ei < a1; ei++)
            for (int64_t ej = b0; ej < b1; ej++) {
                double ar = 0.0, ai = 0.0;  // einsum("ep,epP,eP->e")
                for (int p = 0; p < 3; p++)
                    for (int P = 0; P < 3; P++) {
                        const cd v = e_mul(e_mul(csr.coef[3 * static_cast<std::size_t>(ei) + static_cast<std::size_t>(p)],
                                                 dJ[t * 9 + static_cast<std::size_t>(p) * 3 + static_cast<std::size_t>(P)]),
                                           csr.coef[3 * static_cast<std::size_t>(ej) + static_cast<std::size_t>(P)]);
                        ar = v.real() + ar;
                        ai = v.imag() + ai;
                    }
                const cd val = cm(cd(1.0, 0.0), cd(ar, ai));  // `scale * val`
                cd &gij = G[static_cast<std::size_t>(csr.jbasis[static_cast<std::size_t>(ei)]) +
                            static_cast<std::size_t>(csr.jbasis[static_cast<std::size_t>(ej)]) * n_basis];
                gij = cadd(gij, val);
            }
    }
}

// --- a wholly-buried deck (momwire#1718 S11a, `_operating_medium`) ----------
//
// The whole solve at the lower medium's k_m (a CPython complex) and eta_m =
// np.sqrt(mu / eps_m) (CPython's quotient, numpy's csqrt), sized by the serve
// plan's below/below extents (`r1_below`), then `_assemble_Z_mp` at k_m: the
// free block and the image weighted A_m (`WindowFill` at a complex k), and
// the below family's remainder (`_mp_remainder_below`) through bspline's
// field-form route (`_sinusoidal_mp.remainder_Q_field`): the shared below
// projection (`somm_below::Remainder`, the grid from `somm_below::source()`),
// `field_pair_moments_sinusoidal`, the windowed assembler, and the pairs the
// geometry raises above the base order (`remainder_raised_pairs`)
// re-integrated at their own order minus what the base fill gave them.

// `basis_csr(seg_view, k_m)`: `dshape_coefs` with k a 0-d complex array (an
// array product, and numpy's negative)
Csr basis_csr_c(const SegView &v, cd k) {
    Csr c;
    c.starts.assign(v.starts.begin(), v.starts.end());
    c.jbasis.assign(v.jbasis.begin(), v.jbasis.end());
    const std::size_t nnz = v.jbasis.size();
    c.coef.resize(3 * nnz);
    c.dcoef.resize(3 * nnz);
    const cd mk(-k.real(), -k.imag());
    for (std::size_t e = 0; e < nnz; e++) {
        const cd sig(static_cast<double>(v.sigma[e]), 0.0);
        const cd c0 = cm(sig, v.AC[e]);
        const cd c1 = v.B[e];
        const cd c2 = cm(sig, v.C[e]);
        c.coef[3 * e] = c0;
        c.coef[3 * e + 1] = c1;
        c.coef[3 * e + 2] = c2;
        c.dcoef[3 * e] = cm(k, c1);
        c.dcoef[3 * e + 1] = cm(mk, c2);
        c.dcoef[3 * e + 2] = cm(k, c1);
    }
    return c;
}

// `_operating_medium`'s `serve_plan` over the field rule's nodes
// (`field_nodes`, q = `n_qp_buried_field(3)` = 6) with bspline's
// `_pair_extents_below`: `r1_below`, or the grazing refusal, which is not
// worded here (it declines). The inherited plan, the point-matched lane's
// `serve_plan_below` (`_sinusoidal_fill_core.cpp`) line for line, kept out
// of that shared TU.
double serve_plan_below(const Front &fr) {
    const Geometry &g = fr.geometry;
    const double gz = 0.0;
    const ll q = 6;  // max(n_qp_sommerfeld, N_QP_BURIED_FIELD) * near_q_factor(None, None)
    std::vector<double> xg, wg;
    leggauss(q, xg, wg);
    std::vector<double> x, y, d;
    for (ll i = 0; i < g.n_segs; i++) {
        const auto &l = g.seg_l[static_cast<std::size_t>(i)], &r = g.seg_r[static_cast<std::size_t>(i)];
        for (ll j = 0; j < q; j++) {
            const double tq = 0.5 * (xg[static_cast<std::size_t>(j)] + 1.0);
            double p[3];
            for (int c = 0; c < 3; c++)
                p[c] = l[static_cast<std::size_t>(c)] + tq * (r[static_cast<std::size_t>(c)] - l[static_cast<std::size_t>(c)]);
            x.push_back(p[0]);
            y.push_back(p[1]);
            d.push_back(gz - p[2]);
        }
    }
    // `_pair_extents_below_all`: the upper triangle, diagonal included
    double r1sq_max = 0.0, q_min = std::numeric_limits<double>::infinity();
    const std::size_t n = x.size();
    for (std::size_t i = 0; i < n; i++)
        for (std::size_t j = i; j < n; j++) {
            const double dx = x[i] - x[j], dy = y[i] - y[j];
            const double rho2 = dx * dx + dy * dy;
            const double hh = d[i] + d[j];
            const double hh2 = hh * hh;
            const double r1sq = rho2 + hh2;
            if (r1sq > r1sq_max) r1sq_max = r1sq;
            const double qq = hh2 / rho2;
            if (qq < q_min) q_min = qq;
        }
    const double r1_max = std::sqrt(r1sq_max), th_min = std::atan(std::sqrt(q_min));
    const double lam_m = (2.0 * M_PI) / std::hypot(fr.medium.k_m.real(), fr.medium.k_m.imag());
    const double r1_cap = (somm_below::R1_CAP_LAMBDA_M * lam_m) * (1.0 + 1e-9);  // `plan_r1_cap(below_r1_cap(k_m))`
    const double floor_deg = 0.05 - 2.0 * (0.05 / 3.0);  // _SOMM_BELOW_TH_MIN_DEG
    const double floor = floor_deg * (M_PI / 180.0);     // math.radians
    if (th_min < floor) {
        // past the cap Python asks the floor of the pairs inside it only
        if (r1_max > r1_cap) decline("a grazing below/below pair past the cap (the capped walk is not copied)");
        decline("a below/below pair under the grazing floor (the refusal's wording is not copied)");
    }
    return r1_max;
}

// A ValueError the below remainder raises (the solve's), in its words.
struct SgBelowRefusal {
    [[noreturn]] static void value(const std::string &m) { throw SolverRefusal("ValueError", m); }
};
using BelowRemainder = somm_below::Remainder<SgBelowRefusal>;

// `_below_interface.field_nodes(seg_l, seg_r, tangents, h, q)` over the
// segments `idx` of `s` (its `segment_ends`): the nodes (n q, 3), their
// tangents, and `u_phys` / `w_node` (n, q)
struct FieldNodes {
    std::size_t n = 0, q = 0;
    std::vector<double> nodes, t, u, w;
};
FieldNodes field_nodes(const Segs &s, const std::vector<int64_t> &idx, int64_t q) {
    std::vector<double> xg, wg;
    leggauss(q, xg, wg);
    FieldNodes f;
    f.n = idx.size();
    f.q = static_cast<std::size_t>(q);
    f.nodes.resize(3 * f.n * f.q);
    f.t.resize(3 * f.n * f.q);
    f.u.resize(f.n * f.q);
    f.w.resize(f.n * f.q);
    for (std::size_t i = 0; i < f.n; i++) {
        const std::size_t si = static_cast<std::size_t>(idx[i]);
        const double h = s.h[si];
        for (std::size_t a = 0; a < f.q; a++) {
            const double tq = 0.5 * (xg[a] + 1.0);
            const std::size_t o = i * f.q + a;
            for (int c = 0; c < 3; c++) {
                f.nodes[3 * o + c] = s.l[3 * si + c] + tq * (s.r[3 * si + c] - s.l[3 * si + c]);
                f.t[3 * o + c] = s.t[3 * si + c];
            }
            f.u[o] = h * tq;
            f.w[o] = (0.5 * h) * wg[a];
        }
    }
    return f;
}

// `w_node[None] * shape_values(k, u_phys - 0.5 * h[:, None])`: (3, n, q)
std::vector<cd> shape_weights_c(cd k, const Segs &s, const std::vector<int64_t> &idx, const FieldNodes &f) {
    std::vector<cd> W(3 * f.n * f.q);
    for (std::size_t i = 0; i < f.n; i++) {
        const double hh = 0.5 * s.h[static_cast<std::size_t>(idx[i])];
        for (std::size_t a = 0; a < f.q; a++) {
            const std::size_t o = i * f.q + a;
            cd S[3];
            shapes_c(k, f.u[o] - hh, S);
            for (std::size_t p = 0; p < 3; p++) W[(p * f.n + i) * f.q + a] = cm(cd(f.w[o], 0.0), S[p]);
        }
    }
    return W;
}

// The below family's projection with the nodes `f` as its sources:
// `remainder_field_proj_below(obs, t_obs, src, t_src, gz, k_p, k_m, grid)`
void below_remainder_for(BelowRemainder &rem, const sinusoidal::Medium &md, somm_below::Grid *grid,
                         const FieldNodes &f) {
    rem.gz = 0.0;
    rem.k_p = md.k_p;
    rem.k_m = md.k_m;
    rem.grid = grid;
    rem.src = f.nodes;
    rem.t_src = f.t;
    rem.d_src.resize(f.n * f.q);
    for (std::size_t j = 0; j < rem.d_src.size(); j++) rem.d_src[j] = rem.gz - rem.src[3 * j + 2];
    rem.build_view();
}

// `field_pair_moments(F, W_obs, W_src)`: (3, 3, n_obs, n_src) C order
std::vector<cd> field_moments(const std::vector<cd> &F, const std::vector<cd> &Wo, std::size_t n_obs,
                              const std::vector<cd> &Ws, std::size_t n_src, std::size_t q) {
    const nd::ssize_t M = static_cast<nd::ssize_t>(n_obs * q), S = static_cast<nd::ssize_t>(n_src * q);
    const nd::array_t<cd> J = bspline::field_pair_moments_sinusoidal(
        borrow(F, {M, S}), borrow(Wo, {3, static_cast<nd::ssize_t>(n_obs), static_cast<nd::ssize_t>(q)}),
        borrow(Ws, {3, static_cast<nd::ssize_t>(n_src), static_cast<nd::ssize_t>(q)}));
    if (static_cast<std::size_t>(J.size()) != 9 * n_obs * n_src) throw std::logic_error("field moments: shape");
    return std::vector<cd>(J.data(), J.data() + J.size());
}

// `scatter_pair_moments(G, starts, jbasis, coef, I, J, dJ, scale=1.0)`: each
// listed pair's `einsum("ep,epP,eP->e")` (p outer, P inner, from zero),
// `scale * val`, `np.add.at` in list order; G F order
void scatter_pair_moments(std::vector<cd> &G, std::size_t n_basis, const Csr &csr, const std::vector<int64_t> &I,
                          const std::vector<int64_t> &J, const std::vector<cd> &dJ) {
    for (std::size_t t = 0; t < I.size(); t++) {
        const int64_t a0 = csr.starts[static_cast<std::size_t>(I[t])], a1 = csr.starts[static_cast<std::size_t>(I[t]) + 1];
        const int64_t b0 = csr.starts[static_cast<std::size_t>(J[t])], b1 = csr.starts[static_cast<std::size_t>(J[t]) + 1];
        for (int64_t ei = a0; ei < a1; ei++)
            for (int64_t ej = b0; ej < b1; ej++) {
                double ar = 0.0, ai = 0.0;
                for (int p = 0; p < 3; p++)
                    for (int P = 0; P < 3; P++) {
                        const cd v = e_mul(e_mul(csr.coef[3 * static_cast<std::size_t>(ei) + static_cast<std::size_t>(p)],
                                                 dJ[t * 9 + static_cast<std::size_t>(p) * 3 + static_cast<std::size_t>(P)]),
                                           csr.coef[3 * static_cast<std::size_t>(ej) + static_cast<std::size_t>(P)]);
                        ar = v.real() + ar;
                        ai = v.imag() + ai;
                    }
                const cd val = cm(cd(1.0, 0.0), cd(ar, ai));
                cd &gij = G[static_cast<std::size_t>(csr.jbasis[static_cast<std::size_t>(ei)]) +
                            static_cast<std::size_t>(csr.jbasis[static_cast<std::size_t>(ej)]) * n_basis];
                gij = cadd(gij, val);
            }
    }
}

// `_mp_remainder_below(G, geom, seg_view, ctx, fg, keep=all, medium, r1_below)`:
// `remainder_Q_field(G, ..., segs=all, k_m, n_qp_sommerfeld=3, proj, gz=0,
// scale=REMAINDER_SIGN)`, G F order
// `segs`: the class's GLOBAL segment ids, `s` the class's own arrays in that
// order, `csr` the whole view's tables (a crossing deck's below class,
// momwire#1718 S11b; every segment on a wholly-buried deck)
void mp_remainder_below(std::vector<cd> &G, std::size_t n_basis, const Csr &csr, const Segs &s,
                        const std::vector<int64_t> &segs, const sinusoidal::Medium &md, double r1_below, double omega,
                        const nec5::NumpyLinalg &np) {
    const std::size_t N = s.n;
    const int64_t q = 3;  // n_qp_sommerfeld
    const std::size_t Q = static_cast<std::size_t>(q);
    const cd k = md.k_m;
    somm_below::Source *source = somm_below::source();
    if (source == nullptr) decline("no below/below table source (the buried fill's grid)");
    somm_below::Grid *grid = &source->get(md.eps_t, md.k_p, r1_below, omega, SOLVER_MU, somm_below::SPACING_FILL);
    std::vector<int64_t> local(N);
    for (std::size_t i = 0; i < N; i++) local[i] = static_cast<int64_t>(i);
    const FieldNodes f = field_nodes(s, local, q);
    const std::vector<cd> W = shape_weights_c(k, s, local, f);
    BelowRemainder rem;
    below_remainder_for(rem, md, grid, f);
    // the assembler's tables: unit tangents (F carries the projection), no
    // derivative coefficients
    std::vector<double> unit(3 * N, 0.0);
    for (std::size_t i = 0; i < N; i++) unit[3 * i] = 1.0;
    const std::vector<cd> zero_d(csr.dcoef.size(), cd(0.0, 0.0));
    const nd::ssize_t NS = static_cast<nd::ssize_t>(N), nnz = static_cast<nd::ssize_t>(csr.jbasis.size());
    const CI st = borrow(csr.starts, {static_cast<nd::ssize_t>(csr.starts.size())}), jb = borrow(csr.jbasis, {nnz});
    const CC cf = borrow(csr.coef, {nnz, 3}), dz = borrow(zero_d, {nnz, 3});
    const CI cols = borrow(segs, {NS});
    const CF tc = borrow(unit, {NS, 3});
    nd::Buffer zb;
    zb.ptr = G.data();
    zb.ndim = 2;
    zb.shape[0] = zb.shape[1] = static_cast<nd::ssize_t>(n_basis);
    zb.strides[0] = sizeof(cd);
    zb.strides[1] = static_cast<nd::ssize_t>(sizeof(cd) * n_basis);
    zb.writeable = true;
    const nd::array_t<cd> Z = nd::array_t<cd>::borrow(zb);
    const std::size_t chunk = std::max<std::size_t>(1, (std::size_t(1) << 19) / std::max<std::size_t>(N * Q * Q, 1));
    for (std::size_t i0 = 0; i0 < N; i0 += chunk) {
        const std::size_t i1 = std::min(i0 + chunk, N), m = (i1 - i0) * Q;
        std::vector<cd> proj(m * N * Q);
        rem.project(&f.nodes[3 * i0 * Q], &f.t[3 * i0 * Q], static_cast<std::ptrdiff_t>(m), proj.data());
        std::vector<cd> Wo(3 * (i1 - i0) * Q);
        for (std::size_t p = 0; p < 3; p++)
            std::copy(W.begin() + static_cast<std::ptrdiff_t>((p * N + i0) * Q),
                      W.begin() + static_cast<std::ptrdiff_t>((p * N + i1) * Q),
                      Wo.begin() + static_cast<std::ptrdiff_t>(p * (i1 - i0) * Q));
        const std::vector<cd> J = field_moments(proj, Wo, i1 - i0, W, N, Q);
        const nd::ssize_t NW = static_cast<nd::ssize_t>(i1 - i0);
        std::vector<int64_t> rows(segs.begin() + static_cast<std::ptrdiff_t>(i0), segs.begin() + static_cast<std::ptrdiff_t>(i1));
        std::vector<double> t_rows(unit.begin() + static_cast<std::ptrdiff_t>(3 * i0),
                                   unit.begin() + static_cast<std::ptrdiff_t>(3 * i1));
        bspline::assemble_Z_sinusoidal_windowed(borrow(J, {3, 3, NW, NS}), borrow(rows, {NW}), cols, st, jb, cf, dz,
                                                borrow(t_rows, {NW, 3}), tc, cd(1.0, 0.0), cd(0.0, 0.0), nd::none(),
                                                nd::none(), Z, 0);
    }
    // the pairs the geometry raises (`remainder_raised_pairs`), each at its
    // own order minus the base fill's. Listed over the class's own arrays,
    // then read through `loc` (global id -> class position) as the
    // reference reads them, and kept where both ends land in the class
    std::vector<int64_t> I0, J0, Q0;
    bspline::raised_pairs_on(s.l, s.r, static_cast<int64_t>(N), 0.0, q, 1.0, 0.0, np, I0, J0, Q0);
    if (I0.empty()) return;
    const std::size_t n_total = csr.starts.size() - 1;
    std::vector<int64_t> loc(n_total, -1);
    for (std::size_t i = 0; i < N; i++) loc[static_cast<std::size_t>(segs[i])] = static_cast<int64_t>(i);
    std::vector<int64_t> I, Jl, Qp;
    for (std::size_t t = 0; t < I0.size(); t++) {
        const int64_t a = loc.at(static_cast<std::size_t>(I0[t])), b = loc.at(static_cast<std::size_t>(J0[t]));
        if (a < 0 || b < 0) continue;
        I.push_back(a);
        Jl.push_back(b);
        Qp.push_back(Q0[t]);
    }
    const std::size_t n = I.size();
    std::vector<cd> dJ(9 * n, cd(0.0, 0.0));
    auto moments = [&](const std::vector<int64_t> &obs, const std::vector<int64_t> &src, int64_t qq) {
        const FieldNodes fo = field_nodes(s, obs, qq), fs = field_nodes(s, src, qq);
        const std::vector<cd> Wo = shape_weights_c(k, s, obs, fo), Ws = shape_weights_c(k, s, src, fs);
        BelowRemainder r2;
        below_remainder_for(r2, md, grid, fs);
        std::vector<cd> F(fo.n * fo.q * fs.n * fs.q);
        r2.project(fo.nodes.data(), fo.t.data(), static_cast<std::ptrdiff_t>(fo.n * fo.q), F.data());
        return field_moments(F, Wo, fo.n, Ws, fs.n, static_cast<std::size_t>(qq));
    };
    std::vector<int64_t> orders(Qp);
    std::sort(orders.begin(), orders.end());
    orders.erase(std::unique(orders.begin(), orders.end()), orders.end());
    for (int64_t qq : orders) {
        std::vector<std::size_t> sel;
        for (std::size_t t = 0; t < n; t++)
            if (Qp[t] == qq) sel.push_back(t);
        std::vector<int64_t> uI, uJ;
        for (std::size_t t : sel) {
            uI.push_back(I[t]);
            uJ.push_back(Jl[t]);
        }
        std::sort(uI.begin(), uI.end());
        uI.erase(std::unique(uI.begin(), uI.end()), uI.end());
        std::sort(uJ.begin(), uJ.end());
        uJ.erase(std::unique(uJ.begin(), uJ.end()), uJ.end());
        const std::vector<cd> hi = moments(uI, uJ, qq), lo = moments(uI, uJ, q);
        const std::size_t nI = uI.size(), nJ = uJ.size();
        for (std::size_t t : sel) {
            const std::size_t a = static_cast<std::size_t>(std::lower_bound(uI.begin(), uI.end(), I[t]) - uI.begin());
            const std::size_t b = static_cast<std::size_t>(std::lower_bound(uJ.begin(), uJ.end(), Jl[t]) - uJ.begin());
            for (std::size_t pq = 0; pq < 9; pq++) {
                const std::size_t at = (pq * nI + a) * nJ + b;
                dJ[t * 9 + pq] = csub(hi[at], lo[at]);
            }
        }
    }
    std::vector<int64_t> Ig(n), Jg(n);
    for (std::size_t t = 0; t < n; t++) {
        Ig[t] = segs[static_cast<std::size_t>(I[t])];
        Jg[t] = segs[static_cast<std::size_t>(Jl[t])];
    }
    scatter_pair_moments(G, n_basis, csr, Ig, Jg, dJ);
}

// numpy's complex absolute (`simd_cabsolute`): larger * sqrt(fma(r, r, 1)),
// r = smaller / larger (the point-matched lane's `np_cabs`)
double np_cabs(cd z) {
    const double re = std::fabs(z.real()), im = std::fabs(z.imag());
    if (std::isnan(re) || std::isnan(im)) return std::numeric_limits<double>::quiet_NaN();
    if (std::isinf(re) || std::isinf(im)) return std::numeric_limits<double>::infinity();
    const double larger = std::max(re, im), smaller = std::min(im, re);
    if (larger == 0.0) return larger;
    const double r = smaller / larger;
    return std::sqrt(mw_fma::exact(r, r, 1.0)) * larger;
}

inline cd rsub(double x, cd b) { return cd(x - b.real(), 0.0 - b.imag()); }  // float - complex array
inline cd xdivr(cd a, double x) { return xdiv(a, cd(x, 0.0)); }               // complex array / float

// `_sin_minus_arg(u)` at one complex element: the series and the literal
// difference both evaluated, `np.where(np.abs(u) < 0.1, ...)`
cd sin_minus_arg_c(cd u) {
    const cd u2 = cm(u, u);
    const cd inner = rsub(1.0, cm(xdivr(u2, 72.0), rsub(1.0, xdivr(u2, 110.0))));
    const cd mid = rsub(1.0, cm(xdivr(u2, 42.0), inner));
    const cd bracket = rsub(1.0, cm(xdivr(u2, 20.0), mid));
    const cd series = cm(xdivr(cneg(cm(u, u2)), 6.0), bracket);
    return np_cabs(u) < 0.1 ? series : csub(mw_libm_c::csin(u), u);
}

// `_loading_integrals(k, h)` at a CPython complex k, one segment: (i_g,
// i_ss, i_gg), numpy's elementwise order
void loading_integrals_c(cd k, double h, cd &i_g, cd &i_ss, cd &i_gg) {
    const cd x = cm(cm(cd(0.5, 0.0), k), cd(h, 0.0));  // 0.5 * np.asarray(k) * np.asarray(h)
    const cd s1 = sin_minus_arg_c(x);
    const cd s2 = sin_minus_arg_c(cm(cd(2.0, 0.0), x));
    i_g = cm(py_div(cd(2.0, 0.0), k), s1);           // (2.0 / k) * s1
    i_ss = xdiv(cneg(s2), py_mul(cd(2.0, 0.0), k));  // -s2 / (2.0 * k)
    const cd x2 = cm(x, x);
    static const double fact[16] = {1.0, 1.0, 2.0, 6.0, 24.0, 120.0, 720.0, 5040.0, 40320.0, 362880.0,
                                    3628800.0, 39916800.0, 479001600.0, 6227020800.0, 87178291200.0,
                                    1307674368000.0};
    cd series(0.0, 0.0);
    bool first = true;
    for (int m = 7; m > 1; m--) {
        const double sign = m % 2 == 0 ? 1.0 : -1.0;
        const double coef = sign * (std::pow(4.0, static_cast<double>(m)) - 4.0) / fact[2 * m + 1];
        // `series * x2 + coef`: the float 0.0 first, an array after
        const cd prod = first ? cm(cd(0.0, 0.0), x2) : cm(series, x2);
        series = cd(prod.real() + coef, prod.imag() + 0.0);
        first = false;
    }
    series = cm(cm(cm(series, x2), x2), x);
    const cd lit = csub(cm(cd(0.5, 0.0), s2), cm(cd(4.0, 0.0), s1));
    i_gg = xdiv(np_cabs(x) < 0.25 ? series : lit, k);
}

// `_apply_loading(G, geom, seg_view, k_m)` inside `_operating_medium` (no
// `k_entry` on the fill's view): the overlap's shapes at k_m, the series
// impedance read at the solve's real omega (`medium.k_p * c`)
void apply_loading_c(std::vector<cd> &G, std::size_t n_basis, const Front &fr, cd k, const std::vector<cd> &z_wire) {
    const Geometry &g = fr.geometry;
    const SegView &v = fr.basis;
    std::vector<cd> z_seg(static_cast<std::size_t>(g.n_segs));
    for (std::size_t w = 0; w < g.wire_first.size(); w++)
        for (ll i = g.wire_first[w]; i <= g.wire_last[w]; i++) z_seg[static_cast<std::size_t>(i)] = z_wire.at(w);
    const std::size_t nnz = v.jbasis.size();
    std::vector<cd> U(nnz), R(nnz);
    for (std::size_t e = 0; e < nnz; e++) {
        const cd sg(static_cast<double>(v.sigma[e]), 0.0);
        U[e] = cm(sg, v.AC[e]);
        R[e] = cm(sg, v.C[e]);
    }
    for (std::size_t m = 0; m < static_cast<std::size_t>(g.n_segs); m++) {
        const double h = g.seg_h[m];
        cd i_g, i_ss, i_gg;
        loading_integrals_c(k, h, i_g, i_ss, i_gg);
        for (ll e = v.starts[m]; e < v.starts[m + 1]; e++)
            for (ll f = v.starts[m]; f < v.starts[m + 1]; f++) {
                const std::size_t a = static_cast<std::size_t>(e), b = static_cast<std::size_t>(f);
                const cd t1 = cr(cm(U[a], U[b]), h);
                const cd t2 = cm(cadd(cm(U[a], R[b]), cm(R[a], U[b])), i_g);
                const cd t3 = cm(cm(v.B[a], v.B[b]), i_ss);
                const cd t4 = cm(cm(R[a], R[b]), i_gg);
                const cd val = cm(cadd(cadd(cadd(t1, t2), t3), t4), z_seg[m]);
                cd &slot = G[static_cast<std::size_t>(v.jbasis[a]) +
                             static_cast<std::size_t>(v.jbasis[b]) * n_basis];
                slot = csub(slot, val);
            }
    }
}

// `_assemble_Z_mp(geom, k_m, eta_m)` inside `_operating_medium`: G F order
std::vector<cd> assemble_Z_buried(const Front &fr, const nec4::SolverArgs &args, const nec5::NumpyLinalg *np) {
    // the dialects build the reduced kernel when every wire is below (the
    // buried partial's full fallback); a solver that still asks for the
    // extended kernel takes `_advise_ek_above`'s advisory, not copied here
    if (args.extended_kernel) decline("the extended kernel on a wholly-buried deck (its advisory is not copied)");
    if (np == nullptr) decline("no numpy products for the below remainder's pair scan");
    const sinusoidal::Medium &md = fr.medium;
    const double r1_below = serve_plan_below(fr);
    const cd k = md.k_m;
    const cd eta = mw_libm_c::csqrt(py_div(cd(SOLVER_MU, 0.0), md.eps_m));  // np.sqrt(self.mu / medium.eps_m)
    const Csr csr = basis_csr_c(fr.basis, k);
    const Deck d = deck_of(fr);
    const std::size_t N = d.segs.n, n_basis = N;
    std::vector<cd> G(n_basis * n_basis, cd(0.0, 0.0));
    // `WindowFill`'s c_a = 1j * k * eta (CPython's 1j * k, then numpy's
    // scalar product) and c_phi = eta / (1j * k) (numpy's scalar quotient);
    // each block's `complex(c * scale)` numpy's scalar product
    const cd jk = py_mul(cd(0.0, 1.0), k);
    const cd c_a = smul(jk, eta), c_phi = xdiv(eta, jk);
    const std::vector<double> none_b;
    window_fill_at(G, n_basis, csr, d.segs, d.segs, k, true, smul(c_a, cd(DIRECT_SIGN, 0.0)),
                   smul(c_phi, cd(DIRECT_SIGN, 0.0)), false, false, none_b);
    // the image, `fg.image_sources()` (mirrored in z = 0), scale
    // `-DIRECT_SIGN * fg.image_coefficient` = 1.0 * A_m (CPython's product)
    Segs img;
    img.n = N;
    img.c = d.segs.c;
    img.t = d.segs.t;
    img.h = d.segs.h;
    img.a = d.segs.a;
    for (std::size_t i = 0; i < N; i++) {
        img.c[3 * i + 2] = 2.0 * 0.0 - d.segs.c[3 * i + 2];
        img.t[3 * i] = d.segs.t[3 * i] * 1.0;
        img.t[3 * i + 1] = d.segs.t[3 * i + 1] * 1.0;
        img.t[3 * i + 2] = d.segs.t[3 * i + 2] * -1.0;
    }
    segment_ends(img);
    const cd scale = py_mul(cd(-DIRECT_SIGN, 0.0), md.a_m);
    window_fill_at(G, n_basis, csr, d.segs, img, k, true, smul(c_a, scale), smul(c_phi, scale), false, false, none_b);
    std::vector<int64_t> every(N);
    for (std::size_t i = 0; i < N; i++) every[i] = static_cast<int64_t>(i);
    mp_remainder_below(G, n_basis, csr, d.segs, every, md, r1_below, fr.omega, *np);
    // `_apply_loading(G, geom, seg_view, k_m)`: the overlap's integrals at the
    // complex k are not copied yet
    std::vector<cd> z_wire;
    if (sinusoidal::wire_impedance_per_wire(fr, args, md.k_p * fr.c, z_wire)) apply_loading_c(G, n_basis, fr, k, z_wire);
    return G;
}

// `_basis_value(sigAC, B, sigC, k, xi)` at one entry
cd basis_value(cd sigAC, cd B, cd sigC, double k, double xi) {
    const double half = std::sin(0.5 * k * xi);
    const double s = std::sin(k * xi);
    const cd t1 = cr(B, s);
    const cd t2 = cr(rc(2.0, sigC), half * half);
    return csub(cadd(sigAC, t1), t2);
}

// `_basis_value(sigAC, B, sigC, k, xi)` at a CPython complex k (a wholly-
// buried deck's k_m) and a float xi: `0.5 * k * xi` and `k * xi` CPython's
// products, `np.sin` libm's csin into numpy scalars, `half * half` numpy's
// scalar product, the rest array products
cd basis_value_c(cd sigAC, cd B, cd sigC, cd k, double xi) {
    const cd half = mw_libm_c::csin(py_mul(py_mul(cd(0.5, 0.0), k), cd(xi, 0.0)));
    const cd s = mw_libm_c::csin(py_mul(k, cd(xi, 0.0)));
    const cd t1 = cm(B, s);
    const cd t2 = cm(cm(cd(2.0, 0.0), sigC), smul(half, half));
    return csub(cadd(sigAC, t1), t2);
}

// `_basis_value(sigAC, B, sigC, k, xi)` with `k` an array (`_entry_k` on a
// view carrying `k_entry`, a crossing deck's, momwire#1718 S11b): numpy's
// array products throughout
cd basis_value_e(cd sigAC, cd B, cd sigC, cd k, double xi) {
    const cd half = mw_libm_c::csin(cm(cm(cd(0.5, 0.0), k), cd(xi, 0.0)));
    const cd s = mw_libm_c::csin(cm(k, cd(xi, 0.0)));
    return csub(cadd(sigAC, cm(B, s)), cm(cm(cd(2.0, 0.0), sigC), cm(half, half)));
}

// CPython 3.12's `cmath.sin(z)` at a finite z of moderate size:
// `-1j * sinh(1j * z)`, its sinh `(cos y sinh x, sin y cosh x)` at x + jy =
// -Im z + j Re z (`cmath_sinh_impl` past its special values)
cd cmath_sin(cd z) {
    const double x = -z.imag(), y = z.real();
    if (!std::isfinite(x) || !std::isfinite(y) || std::fabs(x) > 708.0)
        decline("cmath.sin past its ordinary branch");
    const double r_re = mw_libm_c::rcos(y) * std::sinh(x), r_im = mw_libm_c::rsin(y) * std::cosh(x);
    return cd(r_im, -r_re);
}

// `_feed_segment_current(alpha, seg_view, feed_seg, xi)` at a CPython complex
// k off the segment centre (a wholly-buried deck's k_m): `cmath.sin` of
// CPython's products, then numpy's array arithmetic and sum
cd feed_segment_current_c(const SegView &v, const std::vector<cd> &alpha, std::size_t off, ll fi, cd k, double xi) {
    const std::size_t f = static_cast<std::size_t>(fi);
    const cd half = cmath_sin(py_mul(py_mul(cd(0.5, 0.0), k), cd(xi, 0.0)));
    const cd s = cmath_sin(py_mul(k, cd(xi, 0.0)));
    std::vector<cd> terms;
    for (ll e = v.starts[f]; e < v.starts[f + 1]; e++) {
        const std::size_t ez = static_cast<std::size_t>(e);
        const cd amp = alpha[off + static_cast<std::size_t>(v.jbasis[ez])];
        const double sg = static_cast<double>(v.sigma[ez]);
        // sig*AC + B*sin(k xi) - 2.0*sig*C*half*half
        const cd a1 = cadd(cm(cd(sg, 0.0), v.AC[ez]), cm(v.B[ez], s));
        const cd a2 = cm(cm(rc(2.0 * sg, v.C[ez]), half), half);
        terms.push_back(cm(amp, csub(a1, a2)));
    }
    return rp::np_csum(terms);
}

// `_feed_segment_current(alpha, seg_view, feed_seg, xi)` off the centre on a
// view carrying `k_entry`: `_basis_value` at each entry's own k, then
// `(amp * f).sum()`
cd feed_segment_current_e(const SegView &v, const std::vector<cd> &alpha, std::size_t off, ll fi, double xi) {
    const std::size_t f = static_cast<std::size_t>(fi);
    std::vector<cd> terms;
    for (ll e = v.starts[f]; e < v.starts[f + 1]; e++) {
        const std::size_t ez = static_cast<std::size_t>(e);
        const cd sg(static_cast<double>(v.sigma[ez]), 0.0);
        const cd amp = alpha[off + static_cast<std::size_t>(v.jbasis[ez])];
        terms.push_back(cm(amp, basis_value_e(cm(sg, v.AC[ez]), v.B[ez], cm(sg, v.C[ez]), v.k_entry[ez], xi)));
    }
    return rp::np_csum(terms);
}

// `_feed_segment_current(alpha, seg_view, feed_seg, xi)` for a real k
cd feed_segment_current(const SegView &v, const std::vector<cd> &alpha, std::size_t off, ll fi, double k, double xi) {
    const std::size_t f = static_cast<std::size_t>(fi);
    std::vector<cd> terms;
    for (ll e = v.starts[f]; e < v.starts[f + 1]; e++) {
        const std::size_t ez = static_cast<std::size_t>(e);
        const cd amp = alpha[off + static_cast<std::size_t>(v.jbasis[ez])];
        const double sg = static_cast<double>(v.sigma[ez]);
        if (xi == 0.0) {
            terms.push_back(cm(cm(amp, cd(sg, 0.0)), v.AC[ez]));
        } else {
            const double half = std::sin(0.5 * k * xi);
            const double s = std::sin(k * xi);
            // sig*AC + B*sin(k xi) - 2.0*sig*C*half*half
            const cd a1 = cadd(cm(cd(sg, 0.0), v.AC[ez]), cr(v.B[ez], s));
            const cd a2 = cr(cr(rc(2.0 * sg, v.C[ez]), half), half);
            terms.push_back(cm(amp, csub(a1, a2)));
        }
    }
    return rp::np_csum(terms);
}


// --- a crossing deck (momwire#1718 S11b, `_assemble_Z_mp_mixed`) -----------

const std::vector<std::pair<double, int64_t>> &buried_ladder() {  // `BURIED_PAIR_ORDER_LADDER`
    static const std::vector<std::pair<double, int64_t>> l{{2.0, 8}, {16.0, 4}};
    return l;
}
constexpr int64_t BURIED_N_QP_PAIR = 32;

// `basis_csr(seg_view, k_p)` on a view carrying `k_entry`: `dshape_coefs`
// at each entry's own k (`-k` the array's negation)
Csr basis_csr_entry(const SegView &v) {
    Csr c;
    c.starts.assign(v.starts.begin(), v.starts.end());
    c.jbasis.assign(v.jbasis.begin(), v.jbasis.end());
    const std::size_t nnz = v.jbasis.size();
    c.coef.resize(3 * nnz);
    c.dcoef.resize(3 * nnz);
    for (std::size_t e = 0; e < nnz; e++) {
        const cd k = v.k_entry[e], mk(-k.real(), -k.imag());
        const cd sig(static_cast<double>(v.sigma[e]), 0.0);
        const cd c1 = v.B[e];
        const cd c2 = cm(sig, v.C[e]);
        c.coef[3 * e] = cm(sig, v.AC[e]);
        c.coef[3 * e + 1] = c1;
        c.coef[3 * e + 2] = c2;
        c.dcoef[3 * e] = cm(k, c1);
        c.dcoef[3 * e + 1] = cm(mk, c2);
        c.dcoef[3 * e + 2] = cm(k, c1);
    }
    return c;
}

// `csr_subset(starts, jbasis, coef, dcoef, idx)`: the segments `idx`' entries,
// bases unrenumbered
Csr csr_subset(const Csr &c, const std::vector<int64_t> &idx) {
    Csr o;
    o.starts.push_back(0);
    for (int64_t s : idx) {
        for (int64_t e = c.starts[static_cast<std::size_t>(s)]; e < c.starts[static_cast<std::size_t>(s) + 1]; e++) {
            const std::size_t ez = static_cast<std::size_t>(e);
            o.jbasis.push_back(c.jbasis[ez]);
            for (int p = 0; p < 3; p++) {
                o.coef.push_back(c.coef[3 * ez + static_cast<std::size_t>(p)]);
                o.dcoef.push_back(c.dcoef[3 * ez + static_cast<std::size_t>(p)]);
            }
        }
        o.starts.push_back(static_cast<int64_t>(o.jbasis.size()));
    }
    return o;
}

// One class's segment tables (`c[idx]`, `t[idx]`, `h[idx]`, `a_row[idx]`),
// and the class's image (`fg.image_sources()[idx]`)
Segs class_segs(const Segs &all, const std::vector<int64_t> &idx) {
    Segs s;
    s.n = idx.size();
    for (int64_t i : idx) {
        const std::size_t z = static_cast<std::size_t>(i);
        for (int q = 0; q < 3; q++) {
            s.c.push_back(all.c[3 * z + static_cast<std::size_t>(q)]);
            s.t.push_back(all.t[3 * z + static_cast<std::size_t>(q)]);
        }
        s.h.push_back(all.h[z]);
        s.a.push_back(all.a[z]);
    }
    segment_ends(s);
    return s;
}
Segs image_of(const Segs &s) {
    Segs img = s;
    for (std::size_t i = 0; i < s.n; i++) {
        img.c[3 * i + 2] = 2.0 * 0.0 - s.c[3 * i + 2];
        img.t[3 * i] = s.t[3 * i] * 1.0;
        img.t[3 * i + 1] = s.t[3 * i + 1] * 1.0;
        img.t[3 * i + 2] = s.t[3 * i + 2] * -1.0;
    }
    segment_ends(img);
    return img;
}

// `_mixed_serve_plan(..., crossing=True)`'s below/below section: bspline's
// `_pair_extents_below` over the below segments' field nodes (q 6) and their
// Galerkin test nodes (`n_qp_test` 8, the `ctx["obs_c"]` the plan joins)
double serve_plan_crossing(const Front &fr, const std::vector<int64_t> &b_idx) {
    const Geometry &g = fr.geometry;
    const double gz = 0.0;
    std::vector<double> x, y, d;
    {
        std::vector<double> xg, wg;
        leggauss(6, xg, wg);
        for (int64_t i : b_idx) {
            const auto &l = g.seg_l[static_cast<std::size_t>(i)], &r = g.seg_r[static_cast<std::size_t>(i)];
            for (std::size_t j = 0; j < xg.size(); j++) {
                const double tq = 0.5 * (xg[j] + 1.0);
                double p[3];
                for (int c = 0; c < 3; c++)
                    p[c] = l[static_cast<std::size_t>(c)] + tq * (r[static_cast<std::size_t>(c)] - l[static_cast<std::size_t>(c)]);
                x.push_back(p[0]);
                y.push_back(p[1]);
                d.push_back(gz - p[2]);
            }
        }
    }
    {
        std::vector<double> gx, gw;
        leggauss(8, gx, gw);
        for (int64_t i : b_idx) {
            const std::size_t z = static_cast<std::size_t>(i);
            const double hh = 0.5 * g.seg_h[z];
            for (std::size_t q = 0; q < gx.size(); q++) {
                const double xi = hh * gx[q];
                double p[3];
                for (int c = 0; c < 3; c++)
                    p[c] = g.seg_centers[z][static_cast<std::size_t>(c)] + xi * g.seg_tangents[z][static_cast<std::size_t>(c)];
                x.push_back(p[0]);
                y.push_back(p[1]);
                d.push_back(gz - p[2]);
            }
        }
    }
    double r1sq_max = 0.0, q_min = std::numeric_limits<double>::infinity();
    const std::size_t n = x.size();
    for (std::size_t i = 0; i < n; i++)
        for (std::size_t j = i; j < n; j++) {
            const double dx = x[i] - x[j], dy = y[i] - y[j];
            const double rho2 = dx * dx + dy * dy;
            const double hh = d[i] + d[j];
            const double hh2 = hh * hh;
            const double r1sq = rho2 + hh2;
            if (r1sq > r1sq_max) r1sq_max = r1sq;
            const double qq = hh2 / rho2;
            if (qq < q_min) q_min = qq;
        }
    const double r1_max = std::sqrt(r1sq_max), th_min = std::atan(std::sqrt(q_min));
    const double lam_m = (2.0 * M_PI) / std::hypot(fr.medium.k_m.real(), fr.medium.k_m.imag());
    const double r1_cap = (somm_below::R1_CAP_LAMBDA_M * lam_m) * (1.0 + 1e-9);
    const double floor_deg = 0.05 - 2.0 * (0.05 / 3.0);  // _SOMM_BELOW_TH_MIN_DEG
    const double floor = floor_deg * (M_PI / 180.0);
    if (th_min < floor && r1_max > r1_cap) {
        // `_pair_extents_below_numpy(..., r1_cap)`: the floor over the pairs
        // inside the cap only (momwire#1187), every ordered pair, hh / rho
        const double cap2 = r1_cap * r1_cap;
        double ratio_min = std::numeric_limits<double>::infinity();
        for (std::size_t i = 0; i < n; i++)
            for (std::size_t j = 0; j < n; j++) {
                const double dx = x[i] - x[j], dy = y[i] - y[j];
                const double rho2 = dx * dx + dy * dy;
                const double hh = d[i] + d[j];
                const double r1sq = rho2 + hh * hh;
                const double rho = std::sqrt(rho2);
                if (!(rho > 0.0) || !(r1sq <= cap2)) continue;
                const double ratio = hh / rho;
                if (ratio < ratio_min) ratio_min = ratio;
            }
        if (std::atan(ratio_min) >= floor) return r1_max;
    }
    if (th_min < floor) decline("a below/below pair under the grazing floor (the refusal's wording is not copied)");
    return r1_max;
}

// `_loading_integrals(k, h)` with `k` an array (a mixed view's `k_entry` per
// pair), one element: the array's quotients and products
void loading_integrals_e(cd k, double h, cd &i_g, cd &i_ss, cd &i_gg) {
    const cd x = cm(cm(cd(0.5, 0.0), k), cd(h, 0.0));
    const cd s1 = sin_minus_arg_c(x);
    const cd s2 = sin_minus_arg_c(cm(cd(2.0, 0.0), x));
    i_g = cm(xdiv(cd(2.0, 0.0), k), s1);           // (2.0 / k) * s1
    i_ss = xdiv(cneg(s2), cm(cd(2.0, 0.0), k));    // -s2 / (2.0 * k)
    const cd x2 = cm(x, x);
    static const double fact[16] = {1.0, 1.0, 2.0, 6.0, 24.0, 120.0, 720.0, 5040.0, 40320.0, 362880.0,
                                    3628800.0, 39916800.0, 479001600.0, 6227020800.0, 87178291200.0,
                                    1307674368000.0};
    cd series(0.0, 0.0);
    bool first = true;
    for (int m = 7; m > 1; m--) {
        const double sign = m % 2 == 0 ? 1.0 : -1.0;
        const double coef = sign * (std::pow(4.0, static_cast<double>(m)) - 4.0) / fact[2 * m + 1];
        const cd prod = first ? cm(cd(0.0, 0.0), x2) : cm(series, x2);
        series = cd(prod.real() + coef, prod.imag() + 0.0);
        first = false;
    }
    series = cm(cm(cm(series, x2), x2), x);
    const cd lit = csub(cm(cd(0.5, 0.0), s2), cm(cd(4.0, 0.0), s1));
    i_gg = xdiv(np_cabs(x) < 0.25 ? series : lit, k);
}

// `_apply_loading(G, geom, seg_view, None, medium=medium)` on a view with
// `k_entry` (momwire#1156): each pair's shapes at its left entry's k, the
// series impedance at the real omega `medium.k_p * c`, `np.subtract.at` in
// `_shared_segment_pairs`' order
void apply_loading_e(std::vector<cd> &G, std::size_t n_basis, const Front &fr, const std::vector<cd> &z_wire) {
    const Geometry &g = fr.geometry;
    const SegView &v = fr.basis;
    std::vector<cd> z_seg(static_cast<std::size_t>(g.n_segs));
    for (std::size_t w = 0; w < g.wire_first.size(); w++)
        for (ll i = g.wire_first[w]; i <= g.wire_last[w]; i++) z_seg[static_cast<std::size_t>(i)] = z_wire.at(w);
    const std::size_t nnz = v.jbasis.size();
    std::vector<cd> U(nnz), R(nnz);
    for (std::size_t e = 0; e < nnz; e++) {
        const cd sg(static_cast<double>(v.sigma[e]), 0.0);
        U[e] = cm(sg, v.AC[e]);
        R[e] = cm(sg, v.C[e]);
    }
    for (std::size_t m = 0; m < static_cast<std::size_t>(g.n_segs); m++) {
        const double h = g.seg_h[m];
        for (ll e = v.starts[m]; e < v.starts[m + 1]; e++) {
            cd i_g, i_ss, i_gg;
            loading_integrals_e(v.k_entry[static_cast<std::size_t>(e)], h, i_g, i_ss, i_gg);
            for (ll f = v.starts[m]; f < v.starts[m + 1]; f++) {
                const std::size_t a = static_cast<std::size_t>(e), b = static_cast<std::size_t>(f);
                const cd t1 = cr(cm(U[a], U[b]), h);
                const cd t2 = cm(cadd(cm(U[a], R[b]), cm(R[a], U[b])), i_g);
                const cd t3 = cm(cm(v.B[a], v.B[b]), i_ss);
                const cd t4 = cm(cm(R[a], R[b]), i_gg);
                const cd val = cm(cadd(cadd(cadd(t1, t2), t3), t4), z_seg[m]);
                cd &slot = G[static_cast<std::size_t>(v.jbasis[a]) + static_cast<std::size_t>(v.jbasis[b]) * n_basis];
                slot = csub(slot, val);
            }
        }
    }
}

// `_assemble_Z_mp_mixed(geom, eta, below, medium, seg_view, ctx, plan)`: the
// two within-medium classes in mixed-potential form (above at k_p with the
// extended kernel's labels when asked, its image and the Sommerfeld
// remainder; below at k_m with bspline's buried pair ladder, the image
// weighted A_m and the below remainder), then the crossing pair
// (`_crossing_rows`, `G[rows] += t_rows; G[:, rows] += t_cols.T`), the self
// completions with G's sign, and the loading. G (n_basis, n_basis) F order.
std::vector<cd> assemble_Z_crossing(const Front &fr, const nec4::SolverArgs &args, const nec5::NumpyLinalg *np) {
    if (np == nullptr) decline("no numpy products for the crossing fill");
    if (args.ground.kind != nec4::Ground::finite) decline("a crossing deck over a ground with no lower medium");
    const sinusoidal::Medium &md = fr.medium;
    const Deck d = deck_of(fr);
    const std::size_t N = d.segs.n, n_basis = N + static_cast<std::size_t>(fr.n_wings);
    std::vector<int64_t> a_idx, b_idx;
    for (std::size_t i = 0; i < N; i++) (fr.below_seg[i] ? b_idx : a_idx).push_back(static_cast<int64_t>(i));
    const double r1_below = serve_plan_crossing(fr, b_idx);
    // the crossing pair first (its transients never sit beside G)
    const CrossingRows cr = crossing_rows(fr, *np);
    const Csr csr = basis_csr_entry(fr.basis);
    std::vector<cd> G(n_basis * n_basis, cd(0.0, 0.0));
    // the above class: k_p, the free route with the Sommerfeld composition
    {
        const Segs s = class_segs(d.segs, a_idx);
        const double k = md.k_p;
        bool has_b = false;
        if (args.extended_kernel)
            for (std::size_t i = 0; i < s.n; i++)
                if (s.a[i] != s.a[0]) has_b = true;
        WinOpts wo;
        wo.obs_idx = wo.src_idx = &a_idx;
        window_fill(G, n_basis, csr, s, s, k, fr.eta, cd(DIRECT_SIGN, 0.0), args.extended_kernel, has_b, s.a, wo);
        const cd eps_t(args.ground.eps_r, (-args.ground.sigma) / (fr.omega * 8.8541878188e-12));
        const cd c2 = py_div(cd(eps_t.real() - 1.0, eps_t.imag() - 0.0), cd(eps_t.real() + 1.0, eps_t.imag() + 0.0));
        const Segs img = image_of(s);
        window_fill(G, n_basis, csr, s, img, k, fr.eta, py_mul(cd(-DIRECT_SIGN, 0.0), c2), args.extended_kernel,
                    has_b, s.a, wo);
        const Csr sub = csr_subset(csr, a_idx);
        mp_remainder(G, n_basis, sub, s, k, eps_t, fr.omega, 3, *np);  // n_qp_sommerfeld = 3
    }
    // the below class: k_m / eta_m, bspline's buried pair order and ladder
    {
        const Segs s = class_segs(d.segs, b_idx);
        const cd k = md.k_m;
        const cd eta = mw_libm_c::csqrt(py_div(cd(SOLVER_MU, 0.0), md.eps_m));  // np.sqrt(self.mu / medium.eps_m)
        const cd jk = py_mul(cd(0.0, 1.0), k);
        const cd c_a = smul(jk, eta), c_phi = xdiv(eta, jk);
        WinOpts wo;
        wo.n_qp = BURIED_N_QP_PAIR;
        wo.ladder = &buried_ladder();
        wo.obs_idx = wo.src_idx = &b_idx;
        const std::vector<double> none_b;
        window_fill_at(G, n_basis, csr, s, s, k, true, smul(c_a, cd(DIRECT_SIGN, 0.0)),
                       smul(c_phi, cd(DIRECT_SIGN, 0.0)), false, false, none_b, wo);
        const Segs img = image_of(s);
        const cd scale = py_mul(cd(-DIRECT_SIGN, 0.0), md.a_m);
        window_fill_at(G, n_basis, csr, s, img, k, true, smul(c_a, scale), smul(c_phi, scale), false, false, none_b,
                       wo);
        mp_remainder_below(G, n_basis, csr, s, b_idx, md, r1_below, fr.omega, *np);
    }
    // the crossing pair, as `_add_crossing_blocks` adds it
    const std::size_t nr = cr.rows.size();
    for (std::size_t i = 0; i < nr; i++) {
        const std::size_t r = static_cast<std::size_t>(cr.rows[i]);
        for (std::size_t j = 0; j < n_basis; j++) {
            cd &x = G[r + j * n_basis];
            x = cadd(x, cr.t_rows[i * n_basis + j]);
        }
    }
    for (std::size_t i = 0; i < nr; i++) {
        const std::size_t c = static_cast<std::size_t>(cr.rows[i]);
        for (std::size_t j = 0; j < n_basis; j++) {
            cd &x = G[j + c * n_basis];
            x = cadd(x, cr.t_cols[i * n_basis + j]);
        }
    }
    crossing_completions(fr, *np, G);
    std::vector<cd> z_wire;
    if (sinusoidal::wire_impedance_per_wire(fr, args, md.k_p * fr.c, z_wire)) apply_loading_e(G, n_basis, fr, z_wire);
    return G;
}

}  // namespace

void check_constructor(const Front &fr, const nec4::SolverArgs &args) {
    // momwire#398 D2: the extended kernel at a junction whose members do not
    // share one radius
    if (args.extended_kernel && !fr.junctions.empty()) {
        for (const auto &jw : fr.junctions) {
            std::vector<double> radii;
            for (const auto &m : jw) radii.push_back(fr.radius_per_wire.at(static_cast<std::size_t>(m.first)));
            if (std::any_of(radii.begin(), radii.end(), [&](double r) { return r != radii[0]; }))
                throw SolverRefusal("NotImplementedError", nec4::kSgEkSteppedJunction);
        }
    }
}

std::vector<cd> assemble_Z(const Front &fr, const nec4::SolverArgs &args, const nec5::NumpyLinalg *np) {
    // GN 1 (and GD's perfect plane, whose cliff is the readout's): the image
    // block; a finite ground composes the Sommerfeld remainder (S5b)
    const bool pec = args.ground.kind == nec4::Ground::pec;
    const bool finite = args.ground.kind == nec4::Ground::finite;  // GN 2 / GN 3: Sommerfeld
    if (args.ground.kind != nec4::Ground::none && !pec && !finite)
        decline("a reflection-coefficient ground (not the NEC-4.2 slot's)");
    // a wire end on the ground plane: the direct-field fill (S7), then the
    // same `_apply_loading`
    // a wholly-buried deck (momwire#1718 S11a): the mixed-potential fill in
    // the lower medium; a mixed or crossing deck declines in the front
    if (fr.crossing) return assemble_Z_crossing(fr, args, np);  // momwire#1718 S11b
    if (fr.buried) {
        if (direct_route(fr, args)) decline("a wire end on the plane beside a buried wire");
        return assemble_Z_buried(fr, args, np);
    }
    if (direct_route(fr, args)) {
        std::vector<cd> G = assemble_Z_direct(fr, args, np);
        std::vector<cd> z_wire;
        if (sinusoidal::wire_impedance_per_wire(fr, args, fr.k * fr.c, z_wire))
            apply_loading(G, static_cast<std::size_t>(fr.geometry.n_segs), fr, z_wire);
        return G;
    }
    const double k = fr.k;
    const Csr csr = basis_csr(fr.basis, k);
    const Deck d = deck_of(fr);
    const std::size_t N = d.segs.n;
    const std::size_t n_basis = N;
    std::vector<cd> G(n_basis * n_basis, cd(0.0, 0.0));
    // the extended kernel's every-pair labels, and the source tube on a
    // stepped radius (None on a uniform one: b = a on every pair)
    bool has_b = false;
    if (args.extended_kernel)
        for (std::size_t i = 0; i < N; i++)
            if (d.segs.a[i] != d.segs.a[0]) has_b = true;
    window_fill(G, n_basis, csr, d.segs, d.segs, k, fr.eta, cd(DIRECT_SIGN, 0.0), args.extended_kernel, has_b,
                d.segs.a);
    // the Sommerfeld composition's eps~ and image coefficient C2 = (eps~ - 1) /
    // (eps~ + 1), CPython complex arithmetic (`_ground_spec.ground_config`)
    const cd eps_t(args.ground.eps_r, (-args.ground.sigma) / (fr.omega * 8.8541878188e-12));
    const cd c2 = py_div(cd(eps_t.real() - 1.0, eps_t.imag() - 0.0), cd(eps_t.real() + 1.0, eps_t.imag() + 0.0));
    if (pec || finite) {
        // `fg.image_sources()`: centres mirrored in z = 0 (`2 * ground_z - z`),
        // tangents z-flipped (`t * MIRROR`); the image keeps each segment's
        // length and radius. Scale `-DIRECT_SIGN * fg.image_coefficient`.
        Segs img;
        img.n = N;
        img.c = d.segs.c;
        img.t = d.segs.t;
        img.h = d.segs.h;
        img.a = d.segs.a;
        for (std::size_t i = 0; i < N; i++) {
            img.c[3 * i + 2] = 2.0 * 0.0 - d.segs.c[3 * i + 2];
            img.t[3 * i] = d.segs.t[3 * i] * 1.0;
            img.t[3 * i + 1] = d.segs.t[3 * i + 1] * 1.0;
            img.t[3 * i + 2] = d.segs.t[3 * i + 2] * -1.0;
        }
        segment_ends(img);
        const cd scale = pec ? cd(-DIRECT_SIGN * 1.0, 0.0) : py_mul(cd(-DIRECT_SIGN, 0.0), c2);
        window_fill(G, n_basis, csr, d.segs, img, k, fr.eta, scale, args.extended_kernel, has_b, d.segs.a);
        if (finite) {
            if (np == nullptr) decline("no numpy products for the remainder's pair scan");
            mp_remainder(G, n_basis, csr, d.segs, k, eps_t, fr.omega, 3, *np);  // n_qp_sommerfeld = 3
        }
    }
    // `_apply_loading`: read at `k * self.c`, the fill's own k (an insulated
    // wire declines)
    std::vector<cd> z_wire;
    if (sinusoidal::wire_impedance_per_wire(fr, args, k * fr.c, z_wire)) apply_loading(G, n_basis, fr, z_wire);
    return G;
}

namespace {

// SinusoidalGalerkinSolver's constructor over `node_gaps=`
// (`_wire_spec.normalize_node_gaps`, then the entries as node ports and
// `_validate_node_ports` / `_check_shared_junction`): per gap, (junction,
// side_a). ValueError in the constructor's words.
struct NodePort {
    std::size_t junction = 0;
    std::vector<std::size_t> side_a;
};

std::vector<NodePort> node_ports_of(const Front &fr, const std::vector<NodeGap> &gaps) {
    std::vector<NodePort> out;
    const auto &J = fr.junctions;
    const auto end_name = [](bool start) { return std::string(start ? "start" : "end"); };
    std::vector<std::pair<long long, bool>> seen_members;
    std::vector<std::size_t> seen_junctions;
    for (std::size_t i = 0; i < gaps.size(); i++) {
        const NodeGap &gp = gaps[i];
        const std::string at = "node_gaps[" + std::to_string(i) + "]: ";
        if (gp.wire < 0 || gp.wire >= static_cast<long long>(fr.radius_per_wire.size()))
            throw SolverRefusal("ValueError", at + "wire_index " + std::to_string(gp.wire) + " out of range [0, " +
                                                  std::to_string(fr.radius_per_wire.size()) + ")");
        long long jfound = -1;
        std::size_t mfound = 0;
        for (std::size_t j = 0; j < J.size() && jfound < 0; j++)
            for (std::size_t m = 0; m < J[j].size(); m++)
                if (J[j][m].first == gp.wire && J[j][m].second == gp.start) {
                    jfound = static_cast<long long>(j);
                    mfound = m;
                    break;
                }
        if (jfound < 0)
            throw SolverRefusal("ValueError", at + "the " + end_name(gp.start) + " of wire " +
                                                  std::to_string(gp.wire) +
                                                  " is not a member of any junction group \xe2\x80\x94 a series "
                                                  "node gap lives at a junction; for a feed inside a wire use feeds=");
        const std::size_t j = static_cast<std::size_t>(jfound);
        if (J[j].size() < 2) decline("a node gap at a one-member junction");
        for (const auto &sm : seen_members)
            if (sm.first == gp.wire && sm.second == gp.start)
                throw SolverRefusal("ValueError", at + "the " + end_name(gp.start) + " of wire " +
                                                      std::to_string(gp.wire) + " is listed twice");
        if (J[j].size() == 2 && std::find(seen_junctions.begin(), seen_junctions.end(), j) != seen_junctions.end())
            decline("a second node gap at a two-wire junction");
        seen_members.push_back({gp.wire, gp.start});
        seen_junctions.push_back(j);
        NodePort np;
        np.junction = j;
        for (std::size_t m = 0; m < J[j].size(); m++)
            if (m != mfound) np.side_a.push_back(m);
        out.push_back(np);
    }
    // `_check_shared_junction`: a second port at one junction only as another
    // branch cut (the seam's single-member spelling always cuts one branch)
    std::vector<std::vector<std::size_t>> branches(J.size());
    for (const NodePort &np : out) {
        const std::size_t K = J[np.junction].size();
        std::size_t branch = 0;  // side_a is every member but one: that one
        for (std::size_t m = 0; m < K; m++)
            if (std::find(np.side_a.begin(), np.side_a.end(), m) == np.side_a.end()) branch = m;
        auto &b = branches[np.junction];
        if (!b.empty() && (K == 2 || std::find(b.begin(), b.end(), branch) != b.end()))
            decline("a node port listed twice at one junction");
        b.push_back(branch);
    }
    return out;
}

}  // namespace

void check_node_gaps(const Front &fr, const std::vector<NodeGap> &node_gaps) { (void)node_ports_of(fr, node_gaps); }

namespace {

// `_solve_in_place(A, B)`: `lu_factor(A, overwrite_a=True)` (the finite
// check, getrf, the diagonal check) and `lu_solve` (B's finite check, getrs),
// A (n, n) and B (n, nrhs) F order; B overwritten with the solution
void solve_in_place(const sinusoidal::FillLinalg &la, std::vector<cd> &A, std::size_t N, std::vector<cd> &B,
                    std::size_t P) {
    for (const cd &z : A)
        if (!std::isfinite(z.real()) || !std::isfinite(z.imag()))
            throw SolverRefusal("ValueError", "array must not contain infs or NaNs");
    std::vector<int> ipiv(std::max<std::size_t>(N, 1));
    int n = static_cast<int>(N), info = 0;
    la.lu.getrf(&n, &n, A.data(), &n, ipiv.data(), &info);
    if (info < 0) decline("getrf: an illegal argument");
    for (std::size_t i = 0; i < N; i++)
        if (A[i * N + i] == cd(0.0, 0.0)) throw SolverRefusal("LinAlgError", "singular matrix");
    if (P == 0) return;
    for (const cd &z : B)
        if (!std::isfinite(z.real()) || !std::isfinite(z.imag()))
            throw SolverRefusal("ValueError", "array must not contain infs or NaNs");
    int nrhs = static_cast<int>(P);
    char trans = 'N';
    la.lu.getrs(&trans, &n, &nrhs, A.data(), &n, ipiv.data(), B.data(), &n, &info);
    if (info != 0) decline("getrs failed");
}

// `_node_inflow(geom, seg_view, k, j, every member)`: each basis's current
// into the node along every member, at each entry's own k (a crossing view)
std::vector<cd> node_inflow_all(const Front &fr, std::size_t j, std::size_t n_basis) {
    const Geometry &g = fr.geometry;
    const SegView &v = fr.basis;
    std::vector<cd> out(n_basis, cd(0.0, 0.0));
    for (const auto &mem : fr.junctions[j]) {
        const std::size_t w = static_cast<std::size_t>(mem.first);
        const long long m = mem.second ? g.wire_first[w] : g.wire_last[w];
        const double sgn = mem.second ? -1.0 : 1.0;
        const double half = 0.5 * g.seg_h[static_cast<std::size_t>(m)];
        for (ll e = v.starts[static_cast<std::size_t>(m)]; e < v.starts[static_cast<std::size_t>(m) + 1]; e++) {
            const std::size_t ez = static_cast<std::size_t>(e);
            const cd sg(static_cast<double>(v.sigma[ez]), 0.0);
            const cd val = basis_value_e(cm(sg, v.AC[ez]), v.B[ez], cm(sg, v.C[ez]), v.k_entry[ez], half * sgn);
            cd &slot = out[static_cast<std::size_t>(v.jbasis[ez])];
            slot = cadd(slot, cm(cd(sgn, 0.0), val));
        }
    }
    return out;
}

// `_solve_constrained(G, U, C)` (momwire#1282 / #1630): G x = U on C x = 0,
// C the continuity rows of `cont`. The pivots from `scipy.linalg.qr(C,
// pivoting=True)` (zgeqp3 after its workspace query), T = -solve(C_p, C_r)
// (numpy's zgesv), the reduced system `G_rr + T^T G_pr + G_rp T + (T^T G_pp)
// T` and right-hand side through numpy's matmul, `_solve_in_place`, then
// x_p = T x_r. G F order (N, N), U C order (N, P); the answer F order.
std::vector<cd> solve_constrained(const Front &fr, const sinusoidal::FillLinalg &la, const std::vector<cd> &G,
                                  std::size_t N, const std::vector<cd> &U, std::size_t P,
                                  const std::vector<long long> &cont) {
    if (la.lu.geqp3 == nullptr) decline("no zgeqp3 for the crossing node's constrained solve");
    if (la.np.zgesv == nullptr) decline("no zgesv for the crossing node's constrained solve");
    const std::size_t m = cont.size(), n = N;
    if (m >= n) decline("a constrained solve with no free unknowns");
    std::vector<cd> C(m * n);  // C order
    for (std::size_t i = 0; i < m; i++) {
        const std::vector<cd> row = node_inflow_all(fr, static_cast<std::size_t>(cont[i]), n);
        std::copy(row.begin(), row.end(), C.begin() + static_cast<std::ptrdiff_t>(i * n));
    }
    for (const cd &z : C)
        if (!std::isfinite(z.real()) || !std::isfinite(z.imag()))
            throw SolverRefusal("ValueError", "array must not contain infs or NaNs");
    // the pivots
    std::vector<cd> a(m * n);
    for (std::size_t i = 0; i < m; i++)
        for (std::size_t j = 0; j < n; j++) a[i + j * m] = C[i * n + j];
    int mi = static_cast<int>(m), ni = static_cast<int>(n), lda = mi, info = 0, lwork = -1;
    std::vector<int> jpvt(n, 0);
    std::vector<cd> tau(std::max<std::size_t>(std::min(m, n), 1)), wq(1);
    std::vector<double> rwork(2 * n);
    la.lu.geqp3(&mi, &ni, a.data(), &lda, jpvt.data(), tau.data(), wq.data(), &lwork, rwork.data(), &info);
    if (info != 0) decline("zgeqp3's workspace query failed");
    lwork = static_cast<int>(static_cast<long long>(wq[0].real()));  // `.real.astype(numpy.int_)`
    std::vector<cd> work(static_cast<std::size_t>(std::max(lwork, 1)));
    la.lu.geqp3(&mi, &ni, a.data(), &lda, jpvt.data(), tau.data(), work.data(), &lwork, rwork.data(), &info);
    if (info != 0) decline("zgeqp3 failed");
    std::vector<int64_t> p_idx, r_idx;
    std::vector<bool> is_p(n, false);
    for (std::size_t i = 0; i < m; i++) is_p[static_cast<std::size_t>(jpvt[i] - 1)] = true;
    for (std::size_t j = 0; j < n; j++) (is_p[j] ? p_idx : r_idx).push_back(static_cast<int64_t>(j));
    const std::size_t nr = r_idx.size();
    // T = -np.linalg.solve(C[:, p], C[:, r]): (m, nr) C order
    std::vector<cd> T(m * nr);
    {
        std::vector<cd> af(m * m), bf(m * nr);
        for (std::size_t j = 0; j < m; j++)
            for (std::size_t i = 0; i < m; i++) af[i + j * m] = C[i * n + static_cast<std::size_t>(p_idx[j])];
        for (std::size_t j = 0; j < nr; j++)
            for (std::size_t i = 0; i < m; i++) bf[i + j * m] = C[i * n + static_cast<std::size_t>(r_idx[j])];
        const nec5::np_int nn = static_cast<nec5::np_int>(m), nrhs = static_cast<nec5::np_int>(nr);
        std::vector<nec5::np_int> ipiv(std::max<std::size_t>(m, 1));
        nec5::np_int ld = nn, inf = 0;
        la.np.zgesv(&nn, &nrhs, af.data(), &ld, ipiv.data(), bf.data(), &ld, &inf);
        if (inf > 0) decline("a singular continuity block (the reference raises)");
        for (std::size_t i = 0; i < m; i++)
            for (std::size_t j = 0; j < nr; j++) T[i * nr + j] = cneg(bf[i + j * m]);
    }
    const auto Gat = [&](std::size_t i, std::size_t j) { return G[i + j * N]; };
    std::vector<cd> Grr(nr * nr), Gpr(m * nr), Grp(nr * m), Gpp(m * m);  // `G[np.ix_(...)]`, C order
    for (std::size_t i = 0; i < nr; i++)
        for (std::size_t j = 0; j < nr; j++) Grr[i * nr + j] = Gat(static_cast<std::size_t>(r_idx[i]), static_cast<std::size_t>(r_idx[j]));
    for (std::size_t i = 0; i < m; i++)
        for (std::size_t j = 0; j < nr; j++) {
            Gpr[i * nr + j] = Gat(static_cast<std::size_t>(p_idx[i]), static_cast<std::size_t>(r_idx[j]));
            Grp[j * m + i] = Gat(static_cast<std::size_t>(r_idx[j]), static_cast<std::size_t>(p_idx[i]));
        }
    for (std::size_t i = 0; i < m; i++)
        for (std::size_t j = 0; j < m; j++) Gpp[i * m + j] = Gat(static_cast<std::size_t>(p_idx[i]), static_cast<std::size_t>(p_idx[j]));
    const int64_t M = static_cast<int64_t>(m), NR = static_cast<int64_t>(nr), PP = static_cast<int64_t>(P);
    const nec5::Op TT{T.data(), NR, M, 1, NR};  // `T.T`, F order
    const nec5::Op Tc{T.data(), M, NR, NR, 1};
    std::vector<cd> A1(nr * nr), A2(nr * nr), A3a(nr * m), A3(nr * nr);
    nec5::replay::np_matmul(la.np, TT, nec5::Op{Gpr.data(), M, NR, NR, 1}, A1.data(), NR, 1);
    nec5::replay::np_matmul(la.np, nec5::Op{Grp.data(), NR, M, M, 1}, Tc, A2.data(), NR, 1);
    nec5::replay::np_matmul(la.np, TT, nec5::Op{Gpp.data(), M, M, M, 1}, A3a.data(), M, 1);
    nec5::replay::np_matmul(la.np, nec5::Op{A3a.data(), NR, M, M, 1}, Tc, A3.data(), NR, 1);
    // `np.asfortranarray(Gc)`
    std::vector<cd> Gc(nr * nr);
    for (std::size_t i = 0; i < nr; i++)
        for (std::size_t j = 0; j < nr; j++) {
            const std::size_t q = i * nr + j;
            Gc[i + j * nr] = cadd(cadd(cadd(Grr[q], A1[q]), A2[q]), A3[q]);
        }
    // rhs_c = rhs[r] + mm(T.T, rhs[p])
    std::vector<cd> Up(m * P), B1(nr * P);
    for (std::size_t i = 0; i < m; i++)
        for (std::size_t j = 0; j < P; j++) Up[i * P + j] = U[static_cast<std::size_t>(p_idx[i]) * P + j];
    if (P > 0) nec5::replay::np_matmul(la.np, TT, nec5::Op{Up.data(), M, PP, PP, 1}, B1.data(), PP, 1);
    std::vector<cd> xr(nr * P);  // F order: `lu_solve`'s Fortran copy of rhs_c
    for (std::size_t i = 0; i < nr; i++)
        for (std::size_t j = 0; j < P; j++)
            xr[i + j * nr] = cadd(U[static_cast<std::size_t>(r_idx[i]) * P + j], B1[i * P + j]);
    solve_in_place(la, Gc, nr, xr, P);
    // x[p] = mm(T, x_r)
    std::vector<cd> xp(m * P);
    if (P > 0) nec5::replay::np_matmul(la.np, Tc, nec5::Op{xr.data(), NR, PP, 1, NR}, xp.data(), PP, 1);
    std::vector<cd> x(N * P);
    for (std::size_t j = 0; j < P; j++) {
        for (std::size_t i = 0; i < nr; i++) x[static_cast<std::size_t>(r_idx[i]) + j * N] = xr[i + j * nr];
        for (std::size_t i = 0; i < m; i++) x[static_cast<std::size_t>(p_idx[i]) + j * N] = xp[i * P + j];
    }
    return x;
}

}  // namespace

PortSolution port_solution(const Front &fr, const nec4::SolverArgs &args, const sinusoidal::FillLinalg &la,
                           const std::vector<NodeGap> &node_gaps) {
    if (la.lu.getrf == nullptr || la.lu.getrs == nullptr) decline("no LU routines");
    check_constructor(fr, args);
    const std::vector<NodePort> nports = node_ports_of(fr, node_gaps);
    if (!nports.empty() && fr.dedup.active) decline("a node gap on a deck with a wire listed twice");
    if (!nports.empty() && args.ground.kind != nec4::Ground::none)
        for (const NodePort &np : nports)
            // a crossing junction is grounded by geometry but not shorted to
            // the plane: its node port takes the continuity row (momwire#1282)
            if (std::binary_search(fr.geometry.grounded_junctions.begin(), fr.geometry.grounded_junctions.end(),
                                   static_cast<long long>(np.junction)) &&
                std::find(fr.crossing_junctions.begin(), fr.crossing_junctions.end(),
                          static_cast<long long>(np.junction)) == fr.crossing_junctions.end())
                throw SolverRefusal("ValueError",
                                    "junction " + std::to_string(np.junction) +
                                        " is both grounded and a node port \xe2\x80\x94 a node in the ground "
                                        "plane carries current into its own image (#151), so the members' "
                                        "currents do not close on each other and there is no through-current "
                                        "to drive");
    PortSolution ps;
    ps.G = sinusoidal_galerkin::assemble_Z(fr, args, &la.np);
    const Geometry &g = fr.geometry;
    const SegView &v = fr.basis;
    // a crossing deck's basis carries the node wings after the N segment
    // bases (`_n_extra_cols`, momwire#1718 S11b)
    const std::size_t N = static_cast<std::size_t>(g.n_segs + fr.n_wings), NF = g.feed_segs.size(),
                      P = NF + nports.size();
    // `_crossing_continuity_rows`: the crossing junctions a node port names,
    // and every crossing junction of a two-radius deck, ascending
    std::vector<long long> cont;
    if (fr.crossing) {
        std::set<long long> cj;
        for (const NodePort &np : nports)
            if (std::find(fr.crossing_junctions.begin(), fr.crossing_junctions.end(),
                          static_cast<long long>(np.junction)) != fr.crossing_junctions.end())
                cj.insert(static_cast<long long>(np.junction));
        if (fr.two_radius) cj.insert(fr.crossing_junctions.begin(), fr.crossing_junctions.end());
        cont.assign(cj.begin(), cj.end());
    }
    ps.n = static_cast<long long>(N);
    ps.n_ports = static_cast<long long>(P);
    const double k = fr.k;
    // `_drive_columns`: U (N, P), C order as numpy allocates it
    std::vector<cd> U(N * P, cd(0.0, 0.0));
    // node port p: U[:, NF + p] = -(`_node_cut_vectors`), each member's
    // current into the node at xi = sigma_m h_m / 2, `np.add.at` order
    for (std::size_t p = 0; p < nports.size(); p++) {
        std::vector<cd> cut(N, cd(0.0, 0.0));
        for (std::size_t mi : nports[p].side_a) {
            const auto &mem = fr.junctions[nports[p].junction][mi];
            const std::size_t w = static_cast<std::size_t>(mem.first);
            const long long m = mem.second ? g.wire_first[w] : g.wire_last[w];
            const double sgn = mem.second ? -1.0 : 1.0;
            const double half = 0.5 * g.seg_h[static_cast<std::size_t>(m)];
            for (ll e = v.starts[static_cast<std::size_t>(m)]; e < v.starts[static_cast<std::size_t>(m) + 1]; e++) {
                const std::size_t ez = static_cast<std::size_t>(e);
                const cd sg(static_cast<double>(v.sigma[ez]), 0.0);
                // at k_m on a wholly-buried deck, each entry's own k on a
                // crossing one (`_entry_k` of the fill's view)
                const cd val =
                    fr.crossing ? basis_value_e(cm(sg, v.AC[ez]), v.B[ez], cm(sg, v.C[ez]), v.k_entry[ez], half * sgn)
                    : fr.buried ? basis_value_c(cm(sg, v.AC[ez]), v.B[ez], cm(sg, v.C[ez]), fr.medium.k_m, half * sgn)
                                : basis_value(cm(sg, v.AC[ez]), v.B[ez], cm(sg, v.C[ez]), k, half * sgn);
                cd &slot = cut[static_cast<std::size_t>(v.jbasis[ez])];
                slot = cadd(slot, cm(cd(sgn, 0.0), val));
            }
        }
        for (std::size_t i = 0; i < N; i++) U[i * P + NF + p] = cneg(cut[i]);
    }
    for (std::size_t j = 0; j < NF; j++) {
        const std::size_t f = static_cast<std::size_t>(g.feed_segs[j]);
        const double xi = g.feed_xi[j];
        for (ll e = v.starts[f]; e < v.starts[f + 1]; e++) {
            const std::size_t ez = static_cast<std::size_t>(e);
            const cd sg(static_cast<double>(v.sigma[ez]), 0.0);
            // at k_m on a wholly-buried deck (`_operating_medium`'s self.k);
            // at each entry's own k on a crossing deck (`_entry_k`)
            const cd col = cneg(fr.crossing ? basis_value_e(cm(sg, v.AC[ez]), v.B[ez], cm(sg, v.C[ez]), v.k_entry[ez], xi)
                                : fr.buried ? basis_value_c(cm(sg, v.AC[ez]), v.B[ez], cm(sg, v.C[ez]), fr.medium.k_m, xi)
                                            : basis_value(cm(sg, v.AC[ez]), v.B[ez], cm(sg, v.C[ez]), k, xi));
            cd &slot = U[static_cast<std::size_t>(v.jbasis[ez]) * P + j];
            slot = cadd(slot, col);
        }
    }
    if (!cont.empty()) {
        ps.coeffs = solve_constrained(fr, la, ps.G, N, U, P, cont);
        ps.c_order = true;  // `np.empty` then the two slices
    } else {
    // `_solve_in_place(G, U)`: lu_factor(G, overwrite_a=True), the diagonal
    // check, lu_solve
    // scipy's `asarray_chkfinite` and `_solve_in_place`'s singular check
    // raise, and the slot refuses in their words
    for (const cd &z : ps.G)
        if (!std::isfinite(z.real()) || !std::isfinite(z.imag()))
            throw SolverRefusal("ValueError", "array must not contain infs or NaNs");
    std::vector<int> ipiv(std::max<std::size_t>(N, 1));
    int n = static_cast<int>(N), info = 0;
    std::vector<cd> lu = ps.G;  // the solve overwrites G in place; keep G for the dump
    la.lu.getrf(&n, &n, lu.data(), &n, ipiv.data(), &info);
    if (info < 0) decline("getrf: an illegal argument");
    for (std::size_t i = 0; i < N; i++)
        if (lu[i * N + i] == cd(0.0, 0.0)) throw SolverRefusal("LinAlgError", "singular matrix");
    ps.coeffs.assign(N * P, cd(0.0, 0.0));
    for (std::size_t i = 0; i < N; i++)
        for (std::size_t j = 0; j < P; j++) ps.coeffs[i + j * N] = U[i * P + j];
    if (P > 0) {
        for (const cd &z : U)
            if (!std::isfinite(z.real()) || !std::isfinite(z.imag()))
                throw SolverRefusal("ValueError", "array must not contain infs or NaNs");
        int nrhs = static_cast<int>(P);
        char trans = 'N';
        la.lu.getrs(&trans, &n, &nrhs, lu.data(), &n, ipiv.data(), ps.coeffs.data(), &n, &info);
        if (info != 0) decline("getrs failed");
    }
    }
    // Y[:, j] = _port_currents(alphas[:, j], ...): a gap feed's segment
    // current at `feed_xi`; a node port's `-(U[:, port] @ alpha)`, numpy's
    // vector-vector matmul (`cblas_zdotu_sub` over U's strided column)
    ps.y.assign(P * P, cd(0.0, 0.0));
    if (!nports.empty() && la.np.zdotu_sub == nullptr) decline("no zdotu for the node ports' readout");
    for (std::size_t j = 0; j < P; j++)
        for (std::size_t i = 0; i < P; i++) {
            if (i < NF) {
                ps.y[i * P + j] =
                    fr.crossing && g.feed_xi[i] != 0.0
                        // off the centre on a crossing deck: each entry at its own k
                        ? feed_segment_current_e(v, ps.coeffs, j * N, g.feed_segs[i], g.feed_xi[i])
                    : fr.buried && g.feed_xi[i] != 0.0
                        // off the centre at k_m: `cmath.sin` (the readout's #980 branch)
                        ? feed_segment_current_c(v, ps.coeffs, j * N, g.feed_segs[i], fr.medium.k_m, g.feed_xi[i])
                        : feed_segment_current(v, ps.coeffs, j * N, g.feed_segs[i], k, g.feed_xi[i]);
            } else {
                cd dot(0.0, 0.0);
                la.np.zdotu_sub(static_cast<nec5::np_int>(N), &U[i], static_cast<nec5::np_int>(P),
                                &ps.coeffs[j * N], 1, &dot);
                ps.y[i * P + j] = cneg(dot);
            }
        }
    return ps;
}

// --- the comparison format ---------------------------------------------------

namespace {

struct Out {
    std::string s;
    void line(const std::string &l) { s += l + "\n"; }
    void c(const std::string &p, cd v) {
        line(p + " = complex " + nec5::dump_bits(v.real()) + " " + nec5::dump_bits(v.imag()));
    }
};

std::string at(const std::string &p, std::size_t k) { return p + "[" + std::to_string(k) + "]"; }

std::string digest(const cd *row, std::size_t n) {
    const std::uint64_t P = 0x100000001b3ULL;
    std::uint64_t h = 0, pw = 1;
    for (std::size_t k = 0; k < 2 * n; k++) {
        double d = k % 2 == 0 ? row[k / 2].real() : row[k / 2].imag();
        std::uint64_t w;
        std::memcpy(&w, &d, sizeof w);
        std::uint64_t x = w ^ (w >> 29);
        x *= 0xbf58476d1ce4e5b9ULL;
        x ^= x >> 32;
        pw *= P;
        h += x * pw;
    }
    char buf[19];
    std::snprintf(buf, sizeof buf, "%016llx", static_cast<unsigned long long>(h));
    return buf;
}

// `m` (rows, cols) in F order
void matrix(Out &o, const std::string &p, const std::vector<cd> &m, std::size_t rows, std::size_t cols,
            std::size_t whole) {
    o.line(p + " : shape " + std::to_string(rows) + " " + std::to_string(cols));
    std::vector<cd> row(cols);
    for (std::size_t i = 0; i < rows; i++) {
        for (std::size_t j = 0; j < cols; j++) row[j] = m[i + j * rows];
        if (cols <= whole) {
            for (std::size_t j = 0; j < cols; j++) o.c(at(at(p, i), j), row[j]);
        } else {
            o.line(at(p, i) + " = digest " + digest(row.data(), cols));
        }
    }
}

}  // namespace

std::string dump_fill(const std::string &text, const sinusoidal::FillLinalg &la, std::size_t whole, bool g_only) {
    Out o;
    nec4::Build b;
    try {
        b = nec4::build(text, "sinusoidal");
    } catch (const Declined &e) {
        return "fill ? declined " + nec5::dump_hex(e.what()) + "\n";
    }
    o.line("fill.structures : list " + std::to_string(b.structures.size()));
    for (std::size_t s = 0; s < b.structures.size(); s++) {
        const nec4::StructureBuild &sb = b.structures[s];
        std::string p = "fill.structures[" + std::to_string(s) + "]";
        if (sb.kind == nec4::StructureBuild::declined) {
            o.line(p + " ? declined " + nec5::dump_hex(sb.message));
            continue;
        }
        o.line(p + ".points : list " + std::to_string(sb.points.size()));
        for (std::size_t j = 0; j < sb.points.size(); j++) {
            std::string q = at(p + ".points", j);
            try {
                // SinusoidalGalerkinSolver takes no dialect junction rule
                nec4::SolverArgs args = sb.points[j].args;
                args.junction_rule_nec42 = false;
                const Front fr = sinusoidal::front(args, true);
                if (g_only) {
                    check_constructor(fr, fr.args);
                    const std::vector<cd> G = sinusoidal_galerkin::assemble_Z(fr, fr.args, nullptr);
                    const std::size_t n = static_cast<std::size_t>(std::llround(std::sqrt(double(G.size()))));
                    matrix(o, q + ".G", G, n, n, whole);
                    continue;
                }
                const PortSolution ps = sinusoidal_galerkin::port_solution(fr, fr.args, la);  // the solved wires
                const std::size_t n = static_cast<std::size_t>(ps.n), np_ = static_cast<std::size_t>(ps.n_ports);
                matrix(o, q + ".G", ps.G, n, n, whole);
                matrix(o, q + ".coeffs", ps.coeffs, n, np_, whole);
                std::vector<cd> yf(np_ * np_);
                for (std::size_t r = 0; r < np_; r++)
                    for (std::size_t c = 0; c < np_; c++) yf[r + c * np_] = ps.y[r * np_ + c];
                matrix(o, q + ".y", yf, np_, np_, whole);
            } catch (const sinusoidal::FrontError &e) {
                o.line(q + " ! " + e.kind + " " + nec5::dump_hex(e.message));
            } catch (const SolverRefusal &e) {
                o.line(q + " ! " + e.kind + " " + nec5::dump_hex(e.message));
            } catch (const Declined &e) {
                o.line(q + " ? declined " + nec5::dump_hex(e.what()));
            }
        }
    }
    return o.s;
}

}  // namespace sinusoidal_galerkin
}  // namespace mw
