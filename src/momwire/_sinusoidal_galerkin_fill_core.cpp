// The Sinusoidal-Galerkin mixed-potential fill and solve (see
// `_sinusoidal_galerkin_fill_core.h`). Each step is `_sinusoidal_mp.py` /
// `sinusoidal_galerkin.py` as written; the kernels are the B-spline core's
// sinusoidal ones (`mw::bspline::*`), which the bs2 executable already links.
//
// The numpy arithmetic is written in numpy's own order (measured on this
// numpy, as the sinusoidal fill core records): a complex product is
// `np_cmul` in the written operand order with a float promoted to (x, 0);
// a complex quotient `np_cdiv`; `np.exp` of a complex is libm's cexp;
// `einsum("ij,ij->i")` over three is (x0 y0 + x2 y2) + x1 y1; `.sum(-1)`
// over three (p0 + p1) + p2; the three-operand
// complex einsum of `_pair_moments_two_rules_batch` accumulates its (q, r)
// terms from zero, q outer and r inner, each term `(a b) c` with CPython's
// unfused complex product (measured against numpy 2.5's einsum). No product
// here depends on numpy's temporary elision, which not every interpreter
// performs: the one that did is spelled out of it in Python. An
// expression of Python scalars alone (`1j * k * eta`) is CPython's
// `_Py_c_prod` / `_Py_c_quot`.

#if defined(_MSC_VER) && !defined(__clang__)
#pragma float_control(precise, on)
#pragma fp_contract(off)
#endif

#include "_sinusoidal_galerkin_fill_core.h"

#include "_sinusoidal_galerkin_direct_core.h"

#include "_bspline_core.h"
#include "_bspline_fill_core.h"  // the graded remainder pairs (bs2's)
#include "_somm_grid_core.h"
#include "_somm_pairs_owned_core.h"
#include "_somm_remainder_core.h"
#include "_sinusoidal_series_inline.h"  // `sin_minus_arg`
#include "_razor_front_core.h"  // `leggauss`: `_quadrature.leggauss`'s frozen table

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <memory>
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
// numpy's einsum inner product of complex operands (`sum_of_products`):
// never fused
inline cd e_mul(cd a, cd b) {
    return cd(a.real() * b.real() - a.imag() * b.imag(), a.real() * b.imag() + a.imag() * b.real());
}

// numpy's NaN-propagating maximum, and `np.clip`
inline double npmax(double a, double b) { return (a != a) ? a : (b != b) ? b : (a >= b ? a : b); }
inline double npmin(double a, double b) { return (a != a) ? a : (b != b) ? b : (a <= b ? a : b); }

inline double norm3(const double *v) { return std::sqrt((v[0] * v[0] + v[1] * v[1]) + v[2] * v[2]); }
inline double edot3(const double *a, const double *b) { return (a[0] * b[0] + a[2] * b[2]) + a[1] * b[1]; }

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
                const cd e = std::exp(cm(mjk, cd(R, 0.0)));
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

// `near_pair_moments(c_i, t_i, h_i, c_j, t_j, h_j, a2, k, ek=...)` for the
// window [i0, i1): the pairs and J (3, 3, n_near), C order.
struct NearBlock {
    Pairs pairs;
    std::vector<cd> J;
};

NearBlock near_pair_moments(const Segs &o, std::size_t i0, std::size_t i1, const Segs &s, double k, bool ek,
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
            borrow(hj, {M}), borrow(a2, {M}), cd(k, 0.0), borrow(gt, {static_cast<nd::ssize_t>(gt.size())}),
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
        two_rules_batch(o, i0, s, gi, gj, a2, k, ti, wi, tj, wj, eb, nb.J, n, groups[g]);
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
void window_fill(std::vector<cd> &G, std::size_t n_basis, const Csr &csr, const Segs &s, const Segs &src,
                 double k, double eta, cd scale, bool ek, bool has_b_src, const std::vector<double> &b_src) {
    const cd c_a = py_mul(py_mul(cd(0.0, 1.0), cd(k, 0.0)), cd(eta, 0.0));  // 1j * k * eta
    const cd c_phi = py_div(cd(eta, 0.0), py_mul(cd(0.0, 1.0), cd(k, 0.0)));  // eta / (1j * k)
    const cd c_a_s = py_mul(c_a, scale);
    const cd c_phi_s = py_mul(c_phi, scale);
    const std::size_t n_obs = s.n, n_src = src.n;
    const int64_t rows = std::max<int64_t>(
        1, WINDOW_BYTES / (N_SHAPES * N_SHAPES * 16 * std::max<int64_t>(static_cast<int64_t>(n_src), 1)));
    const LadderArrays L = ladder_arrays(DEFAULT_N_QP_PAIR, default_ladder());
    const nd::ssize_t NS = static_cast<nd::ssize_t>(n_src);
    const CF tt = borrow(L.t, {static_cast<nd::ssize_t>(L.t.size())}),
             tw = borrow(L.w, {static_cast<nd::ssize_t>(L.w.size())}),
             tr = borrow(L.ratio, {static_cast<nd::ssize_t>(L.ratio.size())});
    const CI tn = borrow(L.n, {static_cast<nd::ssize_t>(L.n.size())});
    const CF sl_s = borrow(src.l, {NS, 3}), sr_s = borrow(src.r, {NS, 3});
    std::vector<int64_t> every(n_src, 0);
    const CI gj = borrow(every, {NS});
    std::vector<int64_t> all_cols(n_src);
    for (std::size_t j = 0; j < n_src; j++) all_cols[j] = static_cast<int64_t>(j);
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
                    borrow(l, {NR, 3}), borrow(r, {NR, 3}), sl_s, sr_s, a2, k, tt, tw, tn, tr, borrow(gi, {NR}), gj,
                    a, has_b_src ? nd::object(&b_buf) : nd::none());
            } else {
                out = bspline::seg_seg_full_moments_sinusoidal_tiered(borrow(l, {NR, 3}), borrow(r, {NR, 3}), sl_s,
                                                                      sr_s, a2, cd(k, 0.0), tt, tw, tn, tr, false);
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
        const NearBlock nb = near_pair_moments(s, i0, i1, src, k, ek, has_b_src, b_src);
        const std::size_t nn = nb.pairs.ii.size();
        for (std::size_t q = 0; q < nn; q++) {
            const std::size_t m = static_cast<std::size_t>(nb.pairs.ii[q]), j = static_cast<std::size_t>(nb.pairs.jj[q]);
            for (std::size_t pq = 0; pq < 9; pq++) J[(pq * nw + m) * n_src + j] = nb.J[pq * nn + q];
        }
        std::vector<int64_t> rows_seg(nw);
        for (std::size_t m = 0; m < nw; m++) rows_seg[m] = static_cast<int64_t>(i0 + m);
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
                for (int m = 0; m < D1; m++) acc = acc + series[p][m] * T[m][j];
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
    // `fold_rows_into(G, Qr, rows_basis, scale)`: every basis has a row here
    {
        std::vector<std::size_t> first(n_basis);
        for (std::size_t b = 0; b < n_basis; b++) first[b] = static_cast<std::size_t>(row_start[b]);
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
        for (std::size_t i = 0; i < n_basis; i++)
            for (std::size_t j = 0; j < n_basis; j++) {
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
    // hook, which also returns the base-order moments this route does not
    // read: a new export there would move the codegen of that TU's kernels
    std::vector<cd> dJm, unread;
    bspline::debug_pair_moments(bg, gz, k, G0, D1, I, J, q, dJm, unread);
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

// `_basis_value(sigAC, B, sigC, k, xi)` at one entry
cd basis_value(cd sigAC, cd B, cd sigC, double k, double xi) {
    const double half = std::sin(0.5 * k * xi);
    const double s = std::sin(k * xi);
    const cd t1 = cr(B, s);
    const cd t2 = cr(rc(2.0, sigC), half * half);
    return csub(cadd(sigAC, t1), t2);
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

PortSolution port_solution(const Front &fr, const nec4::SolverArgs &args, const sinusoidal::FillLinalg &la) {
    if (la.lu.getrf == nullptr || la.lu.getrs == nullptr) decline("no LU routines");
    check_constructor(fr, args);
    PortSolution ps;
    ps.G = sinusoidal_galerkin::assemble_Z(fr, args, &la.np);
    const Geometry &g = fr.geometry;
    const SegView &v = fr.basis;
    const std::size_t N = static_cast<std::size_t>(g.n_segs), P = g.feed_segs.size();
    ps.n = static_cast<long long>(N);
    ps.n_ports = static_cast<long long>(P);
    const double k = fr.k;
    // `_drive_columns`: U (N, P), C order as numpy allocates it
    std::vector<cd> U(N * P, cd(0.0, 0.0));
    for (std::size_t j = 0; j < P; j++) {
        const std::size_t f = static_cast<std::size_t>(g.feed_segs[j]);
        const double xi = g.feed_xi[j];
        for (ll e = v.starts[f]; e < v.starts[f + 1]; e++) {
            const std::size_t ez = static_cast<std::size_t>(e);
            const cd sg(static_cast<double>(v.sigma[ez]), 0.0);
            const cd col = cneg(basis_value(cm(sg, v.AC[ez]), v.B[ez], cm(sg, v.C[ez]), k, xi));
            cd &slot = U[static_cast<std::size_t>(v.jbasis[ez]) * P + j];
            slot = cadd(slot, col);
        }
    }
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
    // Y[:, j] = _port_currents(alphas[:, j], ...)
    ps.y.assign(P * P, cd(0.0, 0.0));
    for (std::size_t j = 0; j < P; j++)
        for (std::size_t i = 0; i < P; i++)
            ps.y[i * P + j] = feed_segment_current(v, ps.coeffs, j * N, g.feed_segs[i], k, g.feed_xi[i]);
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

std::string dump_fill(const std::string &text, const sinusoidal::FillLinalg &la, std::size_t whole) {
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
                const PortSolution ps = sinusoidal_galerkin::port_solution(fr, fr.args, la);  // the solved wires
                const std::size_t n = static_cast<std::size_t>(ps.n), np_ = static_cast<std::size_t>(ps.n_ports);
                matrix(o, q + ".G", ps.G, n, n, whole);
                matrix(o, q + ".coeffs", ps.coeffs, n, np_, whole);
                std::vector<cd> yf(np_ * np_);
                for (std::size_t r = 0; r < np_; r++)
                    for (std::size_t c = 0; c < np_; c++) yf[r + c * np_] = ps.y[r * np_ + c];
                matrix(o, q + ".y", yf, np_, np_, whole);
            } catch (const sinusoidal::FrontError &e) {
                o.line(q + " ! ValueError " + nec5::dump_hex(e.message));
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
