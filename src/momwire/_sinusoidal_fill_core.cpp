// The sinusoidal free-space fill and solve in C++ (see
// `_sinusoidal_fill_core.h`). Each function names the Python it copies;
// numpy's and scipy's arithmetic is written out per element in their order,
// with no contraction.

#if defined(_MSC_VER) && !defined(__clang__)
#pragma float_control(precise, on)
#pragma fp_contract(off)
#endif

#include "_libm_complex_inline.h"
#include "_sinusoidal_fill_core.h"
#include "_sinusoidal_crossing_core.h"
#include "_sinusoidal_transmitted_core.h"
#include "_somm_transmitted_core.h"

#include "_razor_front_core.h"
#include "_razor_wire_loading_core.h"
#include "_sinusoidal_core.h"
#include "_sinusoidal_nec4_core.h"
#include "_somm_grid_core.h"
#include "_somm_proj_core.h"
#include "_somm_below_fill_core.h"
#include "_somm_below_remainder_inline.h"
#include "_sinusoidal_series_inline.h"  // `sin_minus_arg`, `asinh_minus_arg_from_t`
#include "_fma_inline.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <limits>
#include <map>
#include <memory>
#include <set>
#include <string>
#include <utility>
#include <vector>

namespace mw {
namespace sinusoidal {

namespace {

typedef long long ll;
typedef std::complex<double> cd;
using nec4::Declined;
using nec4::Point3;
using CF = nd::array_t<double, nd::array::c_style | nd::array::forcecast>;
using CI8 = nd::array_t<int8_t, nd::array::c_style | nd::array::forcecast>;
using Tensors = std::tuple<nd::array_t<cd>, nd::array_t<cd>, nd::array_t<cd>>;

static_assert(sizeof(Point3) == 3 * sizeof(double), "Point3 is three packed doubles");

const ll kDenseAssemblyThreshold = 60;  // _DENSE_ASSEMBLY_THRESHOLD
const ll kReduceTile = 32;              // _REDUCE_TILE
const double kWellScaledKd = 1.5e-3;    // _WELL_SCALED_KD
const double kPi = 3.141592653589793;     // math.pi
const ll kSweptMemMb = 256;             // SinusoidalSolver(swept_mem_mb=256)
const ll kNQpConst = 8;                 // SinusoidalSolver(n_qp_const=8)

[[noreturn]] void decline(const std::string &m) { throw Declined(m); }

// A borrowed C-contiguous float64 / int8 array over the caller's memory.
nd::Buffer buffer(const void *p, std::size_t itemsize, std::initializer_list<ll> shape) {
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
    b.writeable = false;
    return b;
}
CF f64(const double *p, std::initializer_list<ll> shape) { return CF::borrow(buffer(p, sizeof(double), shape)); }
CF pts(const std::vector<Point3> &v, ll i0, ll i1) {
    return f64(v.empty() ? nullptr : v[static_cast<std::size_t>(i0)].data(), {i1 - i0, 3});
}

// numpy's elementwise complex multiply (fused, `nec5::replay::np_cmul`) of
// an int8 operand promoted to complex by a complex one: `sigma * A`.
cd sig_mul(int8_t s, cd a) { return nec5::replay::np_cmul(cd(static_cast<double>(s), 0.0), a); }

// scipy's `complex_wrapper` multiply-add (`axpy`'s `y += a * x`), unfused.
inline void axpy1(cd a, cd x, double &yr, double &yi) {
    const double pr = a.real() * x.real() - a.imag() * x.imag();
    const double pi = a.real() * x.imag() + a.imag() * x.real();
    yr = yr + pr;
    yi = yi + pi;
}

// `_seg_radius(geom)`: each segment its wire's radius.
std::vector<double> seg_radius(const Front &fr) {
    const Geometry &g = fr.geometry;
    std::vector<double> a;
    a.reserve(static_cast<std::size_t>(g.n_segs));
    for (std::size_t w = 0; w < g.wire_first.size(); w++)
        for (ll i = g.wire_first[w]; i <= g.wire_last[w]; i++) a.push_back(fr.radius_per_wire[w]);
    return a;
}

// `_ek_gating(geom)`: the EK end labels (ind1 from the N- side, ind2 from N+).
void ek_gating(const Front &fr, const std::vector<double> &rad, std::vector<int8_t> &ind1,
               std::vector<int8_t> &ind2) {
    const Geometry &g = fr.geometry;
    const std::size_t n = static_cast<std::size_t>(g.n_segs);
    const std::vector<Point3> &t = g.seg_tangents;
    std::vector<bool> perp(n);
    for (std::size_t i = 0; i < n; i++) perp[i] = (t[i][0] * t[i][0] + t[i][1] * t[i][1]) <= 1e-8;
    for (int side = 0; side < 2; side++) {
        const std::vector<ll> &count = side == 0 ? g.nm_count : g.np_count;
        const std::vector<ll> &basis = side == 0 ? g.nm_basis : g.np_basis;
        const std::vector<ll> &seg = side == 0 ? g.nm_seg : g.np_seg;
        const std::vector<bool> &ground = side == 0 ? g.ground_minus : g.ground_plus;
        std::vector<int8_t> &ind = side == 0 ? ind1 : ind2;
        ind.assign(n, 2);
        for (std::size_t i = 0; i < n; i++)
            if (count[i] == 0 && !ground[i]) ind[i] = 1;
        for (std::size_t i = 0; i < n; i++)
            if (ground[i] && perp[i]) ind[i] = 0;
        std::vector<ll> nbr(n, -1);  // `nbr[basis] = seg`: the last write wins
        for (std::size_t e = 0; e < basis.size(); e++) nbr[static_cast<std::size_t>(basis[e])] = seg[e];
        for (std::size_t i = 0; i < n; i++) {
            if (count[i] != 1) continue;
            const ll jj = nbr[i];
            if (jj < 0) decline("an EK neighbour table numpy would read uninitialised");
            const std::size_t j = static_cast<std::size_t>(jj);
            // np.einsum("id,id->i") over three: (x0 y0 + x2 y2) + x1 y1
            const double dot = std::fabs((t[i][0] * t[j][0] + t[i][2] * t[j][2]) + t[i][1] * t[j][1]);
            const bool ok = dot >= 0.999999 && std::fabs(rad[j] / rad[i] - 1.0) <= 1e-6;
            if (ok) ind[i] = 0;
        }
    }
}

struct Kernel {
    const Front *fr = nullptr;
    bool ek = false;
    bool nec4 = false;  // momwire#1644: the NEC-4 kernel, `src_a` per source
    std::vector<double> gx, gw, src_a;
    std::vector<int8_t> ind1, ind2;
    // radius runs: (start, stop, a)
    std::vector<std::pair<std::pair<ll, ll>, double>> runs;
};

Tensors call(const Kernel &K, ll r0, ll r1, double a, const std::vector<Point3> &src_c,
             const std::vector<Point3> &src_t) {
    const Geometry &g = K.fr->geometry;
    const ll n = g.n_segs;
    const ll q = static_cast<ll>(K.gx.size());
    CF oc = pts(g.seg_centers, r0, r1), ot = pts(g.seg_tangents, r0, r1);
    CF sc = pts(src_c, 0, n), st = pts(src_t, 0, n);
    CF h = f64(g.seg_h.data(), {n});
    CF gx = f64(K.gx.data(), {q}), gw = f64(K.gw.data(), {q});
    if (K.nec4) {
        (void)a;
        CF sa = f64(K.src_a.data(), {n});
        return sinusoidal_field_tensor_nec4(oc, ot, sc, st, h, sa, K.fr->k, K.fr->eta, gx, gw, 0);
    }
    if (K.ek) {
        CF sa = f64(K.src_a.data(), {n});
        CI8 i1 = CI8::borrow(buffer(K.ind1.data(), 1, {n})), i2 = CI8::borrow(buffer(K.ind2.data(), 1, {n}));
        return sinusoidal_field_tensor_ek(oc, ot, sc, st, h, a, K.fr->k, K.fr->eta, gx, gw, sa, i1, i2, 0);
    }
    return sinusoidal_field_tensor(oc, ot, sc, st, h, a, K.fr->k, K.fr->eta, gx, gw, 0);
}

// `_field_tensor(geom, k, obs_rows=(i0, i1))` on the accelerated literal-cos
// route: (Φ_const, Φ_sin, Φ_cos), each (i1 - i0, N) in C order.
std::vector<std::vector<cd>> field_tensor(const Kernel &K, ll i0, ll i1, const std::vector<Point3> &src_c,
                                          const std::vector<Point3> &src_t) {
    const ll n = K.fr->geometry.n_segs;
    std::vector<std::vector<cd>> out(3, std::vector<cd>(static_cast<std::size_t>((i1 - i0) * n)));
    auto put = [&](const Tensors &t, ll r0, ll r1) {
        const nd::array_t<cd> *parts[3] = {&std::get<0>(t), &std::get<1>(t), &std::get<2>(t)};
        for (int c = 0; c < 3; c++) {
            if (parts[c]->size() != (r1 - r0) * n) throw std::logic_error("field tensor: unexpected shape");
            std::copy(parts[c]->data(), parts[c]->data() + (r1 - r0) * n,
                      out[static_cast<std::size_t>(c)].begin() + (r0 - i0) * n);
        }
    };
    if (K.fr->uniform_radius || K.nec4) {
        // `_nec4_field_tensor`: one call over the band, the radius per source
        put(call(K, i0, i1, K.fr->radius_per_wire[0], src_c, src_t), i0, i1);
        return out;
    }
    // `_radius_runs(geom, obs_rows)`, concatenated on the observer axis
    for (const auto &run : K.runs) {
        const ll s = run.first.first, e = run.first.second;
        if (!(s < i1 && e > i0)) continue;
        const ll r0 = std::max(s, i0), r1 = std::min(e, i1);
        put(call(K, r0, r1, run.second, src_c, src_t), r0, r1);
    }
    return out;
}

// --- the folded cos-1 fill: numpy's reference route, element by element ----
//
// `_field_tensor(cos_shape="cos-1")` takes `_field_components` for every
// pair (the accelerated kernels serve the literal cos only): `_pair_geometry`,
// then `_extended_kernel_fields` with its cos-1 branch (`_ek_end_gxx`,
// `_ek_end_gx`, `_ek_rho_cos1_gap`, `asinh_diff`,
// `expm1_neg_j_from_half`), then `td * Ez + rho_proj_factor * Erho`. Every
// numpy operation is elementwise, so each pair is computed alone, each
// operation as numpy performs it (measured on this numpy):
// - a complex product is `np_cmul(a, b)` in the written operand order, a
//   float operand promoted to (x, 0) first; a complex quotient is
//   `np_cdiv` (numpy multiplies by the divisor's reciprocal);
// - a float plus a complex is (x + re, 0.0 + im), a complex minus a float
//   (re - x, im - 0.0);
// - `np.exp` of a complex is libm's cexp; `np.sin`, `np.cos`, `np.sqrt`,
//   `np.log1p` are libm's under the AVX-512 pins;
// - `einsum("...d,...d->...")` over three is (x0 y0 + x2 y2) + x1 y1, and
//   `.sum(axis=-1)` over three (p0 + p1) + p2; `einsum("...q,q->...")` of a
//   complex row by the float weights is a sequential complex multiply-add
//   from zero, unfused;
// - an expression of Python scalars alone (`1j * k`, `con`) is CPython's
//   `_Py_c_prod` / `_Py_c_quot`, before numpy sees it;
// - numpy's TEMPORARY ELISION: `x * <temporary>` with the temporary a fresh
//   complex array of at least 256 KiB (`NPY_MIN_ELIDE_BYTES`, so M·N ≥ 16384
//   pairs) is computed in place in the temporary, as `<temporary> * x`. The
//   fused complex product is not symmetric in its operands, so each product
//   whose right operand is a temporary takes `cmt`, which swaps them at that
//   size (measured: 16383 elements as written, 16384 swapped). A float
//   operand makes the order immaterial, and so does an add.

inline cd cm(cd a, cd b) { return nec5::replay::np_cmul(a, b); }
// `a * <temporary b>`: elided into `b * a` when `elide`
inline cd cmt(bool elide, cd a, cd b) { return elide ? cm(b, a) : cm(a, b); }
inline cd rc(double x, cd b) { return cm(cd(x, 0.0), b); }  // float * complex
inline cd cr(cd a, double x) { return cm(a, cd(x, 0.0)); }  // complex * float
inline cd cdr(cd a, double x) { return nec5::replay::np_cdiv(a, cd(x, 0.0)); }
inline cd cadd(cd a, cd b) { return cd(a.real() + b.real(), a.imag() + b.imag()); }
inline cd csub(cd a, cd b) { return cd(a.real() - b.real(), a.imag() - b.imag()); }
inline cd radd(double x, cd b) { return cd(x + b.real(), 0.0 + b.imag()); }  // float + complex
inline cd rsub(double x, cd b) { return cd(x - b.real(), 0.0 - b.imag()); }  // float - complex
inline cd csubr(cd a, double x) { return cd(a.real() - x, a.imag() - 0.0); }  // complex - float
inline cd cneg(cd a) { return cd(-a.real(), -a.imag()); }

// CPython 3.12's `_Py_c_prod` / `_Py_c_quot` (a zero divisor never reaches
// here: every divisor is 4 pi k with k > 0).
cd py_mul(cd a, cd b) {
    return cd(a.real() * b.real() - a.imag() * b.imag(), a.real() * b.imag() + a.imag() * b.real());
}
cd py_div(cd a, cd b) {
    const double abs_breal = b.real() < 0 ? -b.real() : b.real();
    const double abs_bimag = b.imag() < 0 ? -b.imag() : b.imag();
    if (abs_breal >= abs_bimag) {
        const double ratio = b.imag() / b.real();
        const double denom = b.real() + b.imag() * ratio;
        return cd((a.real() + a.imag() * ratio) / denom, (a.imag() - a.real() * ratio) / denom);
    }
    const double ratio = b.real() / b.imag();
    const double denom = b.real() * ratio + b.imag();
    return cd((a.real() * ratio + a.imag()) / denom, (a.imag() * ratio - a.real()) / denom);
}

const cd kJ(0.0, 1.0);        // 1j
const cd kMinusJ(-0.0, -1.0);  // -1j, as the compiler folds it

// `_stable.expm1_neg_j(w)`: re - 1j * im
cd expm1_neg_j(double w) {
    const double half = std::sin(0.5 * w);
    const double re = -2.0 * half * half;
    const double im = std::sin(w);
    return rsub(re, cm(kJ, cd(im, 0.0)));
}

// `_stable.expm1_neg_j_from_half(half)`: re + 1j * im
cd expm1_neg_j_from_half(double half) {
    const double h = mw_libm_c::rsin(half), c = mw_libm_c::rcos(half);
    const double re = -2.0 * h * h;
    const double im = 2.0 * h * c;
    return radd(re, cm(kJ, cd(im, 0.0)));
}

// `_stable.asinh_diff(u0, u1, rho2, r0, r1)`
double asinh_diff(double u0, double u1, double rho2, double r0, double r1) {
    const double p0 = u0 >= 0.0 ? u0 + r0 : rho2 / (r0 + std::fabs(u0));
    const double p1 = u1 >= 0.0 ? u1 + r1 : rho2 / (r1 + std::fabs(u1));
    const double du = u1 - u0;
    const double num = du * (p1 + p0);
    return std::log1p(num / ((r1 + r0) * p0));
}

struct End {  // (g1, g1p, g2, g2p, g3, gzz)
    cd g1, g1p, g2, g2p, g3, gzz;
};

// `_ek_end_gxx(k, zz, rh, b, want_swapped)` at one pair
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
    const cd gzp_out = cm(rc(-zz, c1), gzr);
    End e;
    e.g1 = g1;
    e.g1p = g1p;
    e.gzz = gzp_out;
    if (swapped) {  // _ira1
        const double t2b = 0.5 * b;
        e.g2 = cm(rc(-t2b, c1), gzr);
        const cd g2p_s = cdr(cm(rc(t2b, gzr), c2), r2);
        e.g2p = cr(g2p_s, zz);
        e.g3 = csub(rc(rh2, g2p_s), cm(rc(b, gzr), c1));
    } else {  // _ira0
        e.g2 = cdr(g2, rh);
        e.g2p = cdr(cr(g2p, zz), rh);
        e.g3 = cr(cadd(g3, gzp_t), rh);
    }
    return e;
}

// `_ek_end_gx(k, zz, rhx)` at one pair
End ek_end_gx(double k, double zz, double rhx) {
    const double r2 = zz * zz + rhx * rhx;
    const double r = std::sqrt(r2);
    const double kr = k * r;
    const cd c1 = radd(1.0, cm(kJ, cd(kr, 0.0)));
    const cd gz = cdr(std::exp(cm(kMinusJ, cd(kr, 0.0))), r);
    const cd gp = cdr(cm(cneg(c1), gz), r2);
    const cd g1p = cr(gp, zz);
    End e;
    e.g1 = gz;
    e.g1p = g1p;
    e.g2 = cdr(gz, rhx);
    e.g2p = cdr(g1p, rhx);
    e.g3 = cr(gp, rhx);
    e.gzz = cd(0.0, 0.0);
    return e;
}

// `_ek_rho_cos1_gap(k, z1, z2, rhx, rh, b, ind1, ind2, swap)` at one pair;
// `mjk` is `-1j * k` and `jk` `1j * k` (Python scalars).
cd ek_rho_cos1_gap(bool elide, double k, cd jk, cd mjk, double z1, double z2, double rhx, double rh,
                   double b, bool ext1, bool ext2, bool swap) {
    const double r1 = std::sqrt(z1 * z1 + rhx * rhx);
    const double r2 = std::sqrt(z2 * z2 + rhx * rhx);
    const cd e1 = std::exp(cm(mjk, cd(r1, 0.0)));
    const cd e2 = std::exp(cm(mjk, cd(r2, 0.0)));
    const bool lead1 = !(ext1 && swap);
    const bool lead2 = !(ext2 && swap);
    const double dr = (z2 * z2 - z1 * z1) / (r1 + r2);
    cd d_lead;
    if (lead1 && lead2) {
        d_lead = cmt(elide, e1, expm1_neg_j(k * dr));
    } else {
        d_lead = csub(lead2 ? e2 : cd(0.0, 0.0), lead1 ? e1 : cd(0.0, 0.0));
    }
    cd out = cdr(cm(mjk, d_lead), rhx);
    const double kk = k * k;
    auto q = [&](double rr, cd ee, bool ext) -> cd {
        if (!ext) return cd(0.0, 0.0);
        if (swap) return cdr(rc(-0.5 * b * kk, ee), rr);  // q_ira1
        const double r2_ = rr * rr;
        const cd c1 = radd(1.0, cm(jk, cd(rr, 0.0)));
        const double t1 = 0.25 * (b * b) * (rh * rh) / (r2_ * r2_);
        return cdr(cm(rc(kk * rr * t1, c1), ee), rh);  // q_ira0
    };
    const double r1h = swap ? std::sqrt(z1 * z1 + rh * rh) : r1;
    const double r2h = swap ? std::sqrt(z2 * z2 + rh * rh) : r2;
    const cd e1h = swap ? std::exp(cm(mjk, cd(r1h, 0.0))) : e1;
    const cd e2h = swap ? std::exp(cm(mjk, cd(r2h, 0.0))) : e2;
    out = csub(cadd(out, q(r2h, e2h, ext2)), q(r1h, e1h, ext1));
    return out;
}

// `_field_tensor(geom, k, obs_rows=(i0, i1), cos_shape="cos-1")` with the
// extended kernel: numpy's route, (Φ_const, Φ_sin, Φ_cos) in C order.
std::vector<std::vector<cd>> field_tensor_cos1(const Kernel &K, ll i0, ll i1, const std::vector<Point3> &src_c,
                                               const std::vector<Point3> &src_t) {
    const Front &fr = *K.fr;
    const Geometry &g = fr.geometry;
    const ll n = g.n_segs;
    const double k = fr.k;
    const cd jk = py_mul(kJ, cd(k, 0.0));        // 1j * k
    const cd mjk = py_mul(kMinusJ, cd(k, 0.0));  // -1j * k
    // con = 1j * eta / (4.0 * np.pi * k)
    const cd con = py_div(py_mul(kJ, cd(fr.eta, 0.0)), cd(4.0 * 3.141592653589793 * k, 0.0));
    const cd mcon = cneg(con);
    const std::size_t q = K.gx.size();
    const bool elide = (i1 - i0) * n >= 16384;  // a complex (M, N) temporary of 256 KiB
    std::vector<std::vector<cd>> out(3, std::vector<cd>(static_cast<std::size_t>((i1 - i0) * n)));
    for (ll i = i0; i < i1; i++) {
        const Point3 &oc = g.seg_centers[static_cast<std::size_t>(i)];
        const Point3 &ot = g.seg_tangents[static_cast<std::size_t>(i)];
        const double a = fr.uniform_radius ? fr.radius_per_wire[0] : K.src_a[static_cast<std::size_t>(i)];
        const double aa = a * a;
        for (ll j = 0; j < n; j++) {
            const std::size_t jz = static_cast<std::size_t>(j);
            const Point3 &sc = src_c[jz];
            const Point3 &st = src_t[jz];
            const double H = 0.5 * g.seg_h[jz];
            const double src_a = fr.uniform_radius ? fr.radius_per_wire[0] : K.src_a[jz];
            // _pair_geometry
            const double rv[3] = {oc[0] - sc[0], oc[1] - sc[1], oc[2] - sc[2]};
            const double z_eval = (rv[0] * st[0] + rv[2] * st[2]) + rv[1] * st[1];
            double rho_vec[3];
            for (int c = 0; c < 3; c++) rho_vec[c] = rv[c] - z_eval * st[c];
            const double rho_axis =
                std::sqrt((rho_vec[0] * rho_vec[0] + rho_vec[1] * rho_vec[1]) + rho_vec[2] * rho_vec[2]);
            const double rho_eval = std::sqrt(rho_axis * rho_axis + aa);
            const double td = (ot[0] * st[0] + ot[1] * st[1]) + ot[2] * st[2];
            const double rho_dot_tobs = (rho_vec[0] * ot[0] + rho_vec[1] * ot[1]) + rho_vec[2] * ot[2];
            const double rpf = rho_dot_tobs / rho_eval;
            // _extended_kernel_fields(cos_shape="cos-1")
            const double rhx = rho_eval;
            const bool swap = rhx < src_a;
            const double rh = swap ? src_a : rhx;
            const double b = swap ? rhx : src_a;
            const double z2 = H - z_eval;
            const double z1 = -(H + z_eval);
            const double ss = std::sin(k * H);
            const double cs = std::cos(k * H);
            const bool ext1 = K.ind1[jz] != 2, ext2 = K.ind2[jz] != 2;
            const End e1 = ext1 ? ek_end_gxx(k, z1, rh, b, swap) : ek_end_gx(k, z1, rhx);
            const End e2 = ext2 ? ek_end_gxx(k, z2, rh, b, swap) : ek_end_gx(k, z2, rhx);
            const cd ez_sin = cmt(elide, con, csub(cr(cr(csub(e2.g1, e1.g1), cs), k), cr(cadd(e2.g1p, e1.g1p), ss)));
            const cd erho_sin = cmt(elide, mcon, csub(cr(cadd(cadd(cadd(rc(z2, e2.g2p), rc(z1, e1.g2p)), e2.g2), e1.g2), ss),
                                              cr(cr(csub(rc(z2, e2.g2), rc(z1, e1.g2)), cs), k)));
            const cd erho_const = cmt(elide, con, csub(e2.g3, e1.g3));
            const double u2 = H - z_eval;
            const double u1 = -H - z_eval;
            const double rh2 = rh * rh;
            const double rr2 = std::sqrt(rh2 + u2 * u2);
            const double rr1 = std::sqrt(rh2 + u1 * u1);
            const double int_inv_r0 = asinh_diff(u1, u2, rh2, rr1, rr2);
            double er = 0.0, ei = 0.0;  // einsum("...q,q->...", reg_qp, gw)
            for (std::size_t p = 0; p < q; p++) {
                const double z_qp = H * K.gx[p];
                const double dz_qp = z_eval - z_qp;
                const double r0_qp = std::sqrt(rh * rh + dz_qp * dz_qp);
                const cd reg = cdr(expm1_neg_j_from_half(-0.5 * k * r0_qp), r0_qp);
                const double w = K.gw[p];
                er = er + (reg.real() * w - reg.imag() * 0.0);
                ei = ei + (reg.real() * 0.0 + reg.imag() * w);
            }
            const cd int_G0 = radd(int_inv_r0, cr(cd(er, ei), H));
            const double bk = k * b;
            const double bk2 = 0.25 * bk * bk;
            const cd d_gzz = csub(e2.gzz, e1.gzz);
            const double kk1 = k * k * (1.0 - bk2);
            const cd ez_const =
                cmt(elide, mcon, csub(cadd(csub(e2.g1p, e1.g1p), rc(kk1, int_G0)), rc(bk2, d_gzz)));
            const double half_h = std::sin(0.5 * k * H);
            const double cs_m1 = -2.0 * half_h * half_h;
            const cd ez_cos = cmt(elide, mcon, cadd(csub(cadd(cr(cr(cadd(e2.g1, e1.g1), ss), k), cr(csub(e2.g1p, e1.g1p), cs_m1)),
                                                 rc(kk1, int_G0)),
                                            rc(bk2, d_gzz)));
            const cd d_g2p = csub(cadd(csub(rc(z2, e2.g2p), rc(z1, e1.g2p)), e2.g2), e1.g2);
            const cd x_gap = ek_rho_cos1_gap(elide, k, jk, mjk, z1, z2, rhx, rh, b, ext1, ext2, swap);
            const cd erho_cos =
                cmt(elide, mcon, cadd(cadd(x_gap, cr(d_g2p, cs_m1)), cr(cr(cadd(rc(z2, e2.g2), rc(z1, e1.g2)), ss), k)));
            // Phi = td * Ez + rho_proj_factor * Erho
            const std::size_t at = static_cast<std::size_t>((i - i0) * n + j);
            out[0][at] = cadd(rc(td, ez_const), rc(rpf, erho_const));
            out[1][at] = cadd(rc(td, ez_sin), rc(rpf, erho_sin));
            out[2][at] = cadd(rc(td, ez_cos), rc(rpf, erho_cos));
        }
    }
    return out;
}

// --- the reduced kernel on the folded shape, numpy's route -------------------
//
// `_field_tensor(geom_src, k, src_centers, src_tangents, obs_rows=(i0, i1),
// cos_shape="cos-1")` with the reduced kernel at a real k: the accelerated
// kernels serve the literal cos only, so Python takes `_field_components`:
// `_pair_geometry`, `_field_components_bcast` below its EK branch with
// `_folded_cos_fields` for the third shape, then `td * Ez + rpf * Erho`.
// Every numpy operation is elementwise, so each pair is computed alone, as
// `field_tensor_cos1` above does for the extended kernel. Python names every
// complex product's right-hand operand on this route (momwire#392), so
// numpy's temporary elision moves nothing at any band size. On arm64 the
// `...d,...d` einsum is contracted (`MW_NUMPY_CONTRACTS`); its complex
// einsum against the real weights rounds the same either way.

// `einsum("...d,...d->...")` over three (momwire#1718's measurement)
inline double edot3(const double *a, const double *b) {
#if MW_NUMPY_CONTRACTS
    return mw_fma::exact(a[2], b[2], a[0] * b[0]) + a[1] * b[1];
#else
    return (a[0] * b[0] + a[2] * b[2]) + a[1] * b[1];
#endif
}
// `(u * v).sum(axis=-1)` over three
inline double sdot3(const double *a, const double *b) { return (a[0] * b[0] + a[1] * b[1]) + a[2] * b[2]; }

struct Reduced {  // `_field_components_bcast`'s six tables at one pair
    cd erc, ezc, ers, ezs, ero, ezo;
};

struct RedConst {  // the Python scalars and the rule the route reads
    double k = 0;
    cd jk, mjk, pz, prc, c_e;
    double four_pi_k = 0, sw_hi = 0, sw_lo = 0;
    std::vector<double> gx, gw;
};

RedConst red_const(double k, double eta, const std::vector<double> &gx, const std::vector<double> &gw) {
    RedConst K;
    K.k = k;
    K.jk = py_mul(kJ, cd(k, 0.0));        // 1j * k
    K.mjk = py_mul(kMinusJ, cd(k, 0.0));  // -1j * k
    K.c_e = py_mul(kMinusJ, cd(eta, 0.0));  // -1j * eta
    K.four_pi_k = 4.0 * kPi * k;
    K.pz = py_div(py_mul(kJ, cd(eta, 0.0)), cd(K.four_pi_k, 0.0));  // 1j * eta / (4 pi k)
    K.prc = py_div(K.c_e, cd(K.four_pi_k, 0.0));                     // -1j * eta / (4 pi k)
    K.gx = gx;
    K.gw = gw;
    std::vector<double> hi, lo;  // gw[gx >= 0].sum(), gw[gx < 0].sum()
    for (std::size_t q = 0; q < gx.size(); q++) (gx[q] >= 0.0 ? hi : lo).push_back(gw[q]);
    K.sw_hi = nec5::replay::np_pairwise(hi.data(), static_cast<int64_t>(hi.size()));
    K.sw_lo = nec5::replay::np_pairwise(lo.data(), static_cast<int64_t>(lo.size()));
    return K;
}

// `_field_components_bcast(..., cos_shape="cos-1", ek=None)` at one pair
Reduced reduced_cos1(const RedConst &K, double z, double rho, double H) {
    const double k = K.k;
    const double dz2 = z - H, dz1 = z + H;
    const double r2 = std::sqrt(rho * rho + dz2 * dz2);
    const double r1 = std::sqrt(rho * rho + dz1 * dz1);
    const cd G2 = cdr(mw_libm_c::cexp(cm(K.mjk, cd(r2, 0.0))), r2);
    const cd G1 = cdr(mw_libm_c::cexp(cm(K.mjk, cd(r1, 0.0))), r1);
    // pref_rho = -1j * eta / (4.0 * np.pi * k * rho_eval)
    const cd pref_rho = nec5::replay::np_cdiv(K.c_e, cd(K.four_pi_k * rho, 0.0));
    const cd c12 = radd(1.0, cm(K.jk, cd(r2, 0.0)));  // 1.0 + 1j k r0_2
    const cd c11 = radd(1.0, cm(K.jk, cd(r1, 0.0)));
    Reduced f;
    f.erc = cm(K.prc, csub(cdr(cm(cr(c12, rho), G2), r2 * r2), cdr(cm(cr(c11, rho), G1), r1 * r1)));
    const double int_inv_r0 = asinh_diff(-dz1, -dz2, rho * rho, r1, r2);
    const std::size_t nq = K.gx.size();
    std::vector<double> dz_qp(nq), r_qp(nq);
    double ar = 0.0, ai = 0.0;  // einsum("...q,q->...", reg_qp, gw), from zero
    for (std::size_t q = 0; q < nq; q++) {
        const double z_qp = H * K.gx[q];
        dz_qp[q] = z - z_qp;
        r_qp[q] = std::sqrt(rho * rho + dz_qp[q] * dz_qp[q]);
        const cd reg = cdr(expm1_neg_j_from_half((-0.5 * k) * r_qp[q]), r_qp[q]);
        const double w = K.gw[q];
        ar = ar + (reg.real() * w - reg.imag() * 0.0);
        ai = ai + (reg.real() * 0.0 + reg.imag() * w);
    }
    const cd int_G0 = radd(int_inv_r0, cr(cd(ar, ai), H));
    const cd ezb = csub(cdr(cm(cr(c12, dz2), G2), r2 * r2), cdr(cm(cr(c11, dz1), G1), r1 * r1));
    f.ezc = cm(cneg(K.pz), cadd(ezb, rc(k * k, int_G0)));
    // the sine source
    const double sin2 = mw_libm_c::rsin(k * H), cos2 = mw_libm_c::rcos(k * H);
    const double sin1 = mw_libm_c::rsin(-k * H), cos1 = mw_libm_c::rcos(-k * H);
    const cd b_sin_2 = radd((k * dz2) * cos2, cr(rsub(1.0, cdr(rc(dz2 * dz2, c12), r2 * r2)), sin2));
    const cd b_sin_1 = radd((k * dz1) * cos1, cr(rsub(1.0, cdr(rc(dz1 * dz1, c11), r1 * r1)), sin1));
    f.ers = cm(pref_rho, csub(cm(G2, b_sin_2), cm(G1, b_sin_1)));
    const cd b_sin_z_2 = rsub(k * cos2, cr(cdr(cr(c12, dz2), r2 * r2), sin2));
    const cd b_sin_z_1 = rsub(k * cos1, cr(cdr(cr(c11, dz1), r1 * r1), sin1));
    f.ezs = cm(K.pz, csub(cm(G2, b_sin_z_2), cm(G1, b_sin_z_1)));
    // `_folded_cos_fields`
    const double kH = k * H;
    const double sh = mw_libm_c::rsin(0.5 * kH);
    const double cm1 = -2.0 * (sh * sh);
    const cd T2 = cdr(cm(c12, G2), r2 * r2);
    const cd T1 = cdr(cm(c11, G1), r1 * r1);
    const bool outside = dz1 * dz2 >= 0.0;
    const double s_den = outside ? dz1 * r2 + dz2 * r1 : 1.0;
    const double X = outside ? ((2.0 * H) * (dz1 + dz2)) / s_den : (dz1 * r2 - dz2 * r1) / (rho * rho);
    const double t_sing = std::fabs(X) < 1.0
                              ? asinh_minus_arg_from_t(std::asinh(X)) +
                                    ((((H * rho) * rho) * X) * X) / (((r1 + r2) * r1) * r2)
                              : std::asinh(X) - H * (1.0 / r1 + 1.0 / r2);
    const cd g1 = cdr(expm1_neg_j(k * r1), r1);
    const cd g2 = cdr(expm1_neg_j(k * r2), r2);
    const cd e2 = cr(G2, r2), e1 = cr(G1, r1);
    double mr = 0.0, mi = 0.0;  // einsum("...q,q->...", dg, gw), from zero
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
        const double w = K.gw[q];
        mr = mr + (dg.real() * w - dg.imag() * 0.0);
        mi = mi + (dg.real() * 0.0 + dg.imag() * w);
    }
    const cd m_reg = cadd(cadd(cd(mr, mi), rc(K.sw_hi - 1.0, g2)), rc(K.sw_lo - 1.0, g1));
    const cd d_int = csub(radd(t_sing, rc(H, m_reg)), rc(sin_minus_arg(kH) / k, cadd(G1, G2)));
    f.ezo = cm(K.pz, csub(rc(k * k, d_int), rc(cm1, csub(rc(dz2, T2), rc(dz1, T1)))));
    const double dr = ((4.0 * H) * z) / (r1 + r2);
    const double phi = (0.5 * k) * dr;
    const double A = kH + phi, Bv = kH - phi;
    const double d_lin = ((((-8.0 * std::pow(H, 3.0)) * z) * rho) * rho) /
                         ((((rho * rho + dz1 * dz2) + r1 * r2) * (r1 + r2)) * r1 * r2);
    const double cph = mw_libm_c::rcos(phi), sph = mw_libm_c::rsin(phi);
    const double w_even = (A * sin_minus_arg(Bv) - Bv * sin_minus_arg(A)) / kH + ((d_lin / H) * sin2) * cph;
    const cd w_odd = cr(cr(cm(kJ, cd(sin2, 0.0)), -((rho * rho) * X) / (r1 * r2)), sph);
    const cd W = cm(cm(cr(G2, r2), rsub(cph, cm(kJ, cd(sph, 0.0)))), radd(w_even, w_odd));
    const cd e_rho_brk = cadd(rc(-k, W), rc((rho * rho) * cm1, csub(T2, T1)));
    f.ero = cm(pref_rho, e_rho_brk);
    return f;
}

// `_field_tensor(geom_src, k, src_centers=sc, src_tangents=st, obs_rows=(i0,
// i1), cos_shape="cos-1")` at a real k with the reduced kernel and one
// radius: (Phi_const, Phi_sin, Phi_cos), (i1 - i0, n) in C order, `sh` the
// sources' full lengths
std::vector<std::vector<cd>> field_tensor_reduced_cos1(const Front &fr, const Kernel &K, ll i0, ll i1,
                                                       const std::vector<Point3> &sc_, const std::vector<Point3> &st_,
                                                       const std::vector<double> &sh, double k, double eta) {
    const Geometry &g = fr.geometry;
    const ll n = static_cast<ll>(sc_.size());
    const RedConst C = red_const(k, eta, K.gx, K.gw);
    const double a = fr.radius_per_wire[0];
    std::vector<std::vector<cd>> out(3, std::vector<cd>(static_cast<std::size_t>((i1 - i0) * n)));
    for (ll i = i0; i < i1; i++) {
        const Point3 &oc = g.seg_centers[static_cast<std::size_t>(i)];
        const Point3 &ot = g.seg_tangents[static_cast<std::size_t>(i)];
        for (ll j = 0; j < n; j++) {
            const std::size_t jz = static_cast<std::size_t>(j);
            const Point3 &sc = sc_[jz];
            const Point3 &st = st_[jz];
            // `_pair_geometry`
            const double rv[3] = {oc[0] - sc[0], oc[1] - sc[1], oc[2] - sc[2]};
            const double z = edot3(rv, st.data());
            double rho_vec[3];
            for (int c = 0; c < 3; c++) rho_vec[c] = rv[c] - z * st[static_cast<std::size_t>(c)];
            const double rho_axis = std::sqrt((rho_vec[0] * rho_vec[0] + rho_vec[1] * rho_vec[1]) + rho_vec[2] * rho_vec[2]);
            const double rho = std::sqrt(rho_axis * rho_axis + a * a);
            const double td = sdot3(ot.data(), st.data());
            const double rpf = sdot3(rho_vec, ot.data()) / rho;
            const Reduced f = reduced_cos1(C, z, rho, 0.5 * sh[jz]);
            const std::size_t at = static_cast<std::size_t>((i - i0) * n + j);
            out[0][at] = cadd(rc(td, f.ezc), rc(rpf, f.erc));
            out[1][at] = cadd(rc(td, f.ezs), rc(rpf, f.ers));
            out[2][at] = cadd(rc(td, f.ezo), rc(rpf, f.ero));
        }
    }
    return out;
}

// --- the composing Sommerfeld ground (`fg.mode == "compose"`) --------------
//
// `_sommerfeld_remainder_prepare` and `_replay_sommerfeld_remainder`: each
// source segment's `n_qp_sommerfeld` Gauss nodes, the three source shapes
// weighted (`shp_w`), the grid (`_sommerfeld.get_grid`, the shared cores'
// view with every node filled: the eager grid's values), and per observer
// chunk the projected remainder table (`remainder_field_proj_batch`,
// `somm_core::field_proj_batch`) reduced against the shapes as
// `remainder_shape_reduce` does it.
const ll kNQpSommerfeld = 3;              // SinusoidalSolver(n_qp_sommerfeld=3)
const ll kRemainderChunkElems = 1 << 19;  // _REMAINDER_CHUNK_ELEMS
const double kMu = 1.25663706127e-6;      // SinusoidalSolver.mu

struct Somm {
    cd eps_t, c2;  // the medium and the image coefficient (eps_t - 1) / (eps_t + 1)
    ll q = 0, n_src = 0;
    std::vector<double> srcf, t_src;  // (n_src, 3)
    std::vector<double> shp_w;        // (3, N, q)
    const somm_core::Grid *grid = nullptr;
    std::shared_ptr<somm_proj::GridView> view;
    std::size_t n_reg = 0;
};

// `_ground_refl.eps_tilde((eps_r, sigma), omega, eps0)` and CPython's
// `(eps_t - 1.0) / (eps_t + 1.0)`
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

void somm_prepare(const Front &fr, const nec4::SolverArgs &args, bool cos1, Somm &sm) {
    const Geometry &g = fr.geometry;
    const ll N = g.n_segs;
    const double gz = 0.0;  // the NEC-4.2 dialect's interface
    sm.eps_t = cd(args.ground.eps_r, (-args.ground.sigma) / (fr.omega * 8.8541878188e-12));
    sm.c2 = py_quot(cd(sm.eps_t.real() - 1.0, sm.eps_t.imag() - 0.0),
                    cd(sm.eps_t.real() + 1.0, sm.eps_t.imag() + 0.0));
    for (ll i = 0; i < N; i++) {
        const std::size_t z = static_cast<std::size_t>(i);
        if (g.seg_l[z][2] - gz < -1e-12 || g.seg_r[z][2] - gz < -1e-12)
            decline("a wire below the Sommerfeld ground (the reference raises)");
    }
    std::vector<double> gx, gw;
    if (!razor::leggauss(kNQpSommerfeld, gx, gw)) throw std::logic_error("leggauss(3) is tabled");
    const ll q = kNQpSommerfeld;
    sm.q = q;
    sm.n_src = N * q;
    const double k = fr.k;
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
                sm.srcf[3 * at + static_cast<std::size_t>(c)] =
                    g.seg_centers[nz][static_cast<std::size_t>(c)] + zloc * g.seg_tangents[nz][static_cast<std::size_t>(c)];
                sm.t_src[3 * at + static_cast<std::size_t>(c)] = g.seg_tangents[nz][static_cast<std::size_t>(c)];
            }
            const double w_node = h_half * gw[jz];
            double shp_cos;
            if (cos1) {
                const double s = std::sin(0.5 * k * zloc);
                shp_cos = -2.0 * (s * s);
            } else {
                shp_cos = std::cos(k * zloc);
            }
            const double shp[3] = {1.0, std::sin(k * zloc), shp_cos};
            for (int c = 0; c < 3; c++)
                sm.shp_w[static_cast<std::size_t>((c * N + n) * q + j)] = shp[c] * w_node;
        }
    }
    // `_sommerfeld.max_image_distance(seg_l, seg_r, gz)`
    std::vector<double> sl(static_cast<std::size_t>(3 * N)), sr(static_cast<std::size_t>(3 * N));
    for (ll n = 0; n < N; n++)
        for (int c = 0; c < 3; c++) {
            sl[static_cast<std::size_t>(3 * n + c)] = g.seg_l[static_cast<std::size_t>(n)][static_cast<std::size_t>(c)];
            sr[static_cast<std::size_t>(3 * n + c)] = g.seg_r[static_cast<std::size_t>(n)][static_cast<std::size_t>(c)];
        }
    const double r1_max = razor::max_image_distance(sl.data(), sr.data(), static_cast<std::size_t>(N), gz);
    const somm_core::Grid &grid = somm_core::get_grid_lazy(sm.eps_t, k, r1_max, fr.omega, kMu);
    // the eager grid: every node (`get_grid(..., lazy=False)`), the same values
    somm_core::ensure_nodes(grid, somm_core::all_nodes_of(grid));
    sm.grid = &grid;
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
    // `grid_cpp_args(grid)`: theta split at `math.radians(30.0)`, the far pack
    sm.view = std::make_shared<somm_proj::GridView>(somm_proj::build_grid_view(
        grid.r1_max, grid.r_break, 30.0 * (M_PI / 180.0), grid.r_near, r0, dr, th0, dth, sm.n_reg, vals, shape0,
        nr, nth));
    const std::vector<cd> far = somm_core::far_pack(grid);
    somm_proj::set_far(*sm.view, far.data(), static_cast<std::ptrdiff_t>(far.size()));
}

// `_replay_sommerfeld_remainder` for observer rows [i0, i1): S (3, M, N)
std::vector<std::vector<cd>> somm_replay(const Front &fr, const Somm &sm, ll i0, ll i1) {
    const Geometry &g = fr.geometry;
    const ll N = g.n_segs, M = i1 - i0, q = sm.q;
    std::vector<std::vector<cd>> S(3, std::vector<cd>(static_cast<std::size_t>(M * N)));
    const ll chunk = std::max<ll>(1, kRemainderChunkElems / std::max<ll>(1, sm.n_src));  // `elem_rows`
    std::vector<double> obs, tob;
    for (ll a = 0; a < M; a += chunk) {
        const ll b = std::min(a + chunk, M), m = b - a;
        obs.assign(static_cast<std::size_t>(3 * m), 0.0);
        tob.assign(static_cast<std::size_t>(3 * m), 0.0);
        for (ll r = 0; r < m; r++)
            for (int c = 0; c < 3; c++) {
                obs[static_cast<std::size_t>(3 * r + c)] = g.seg_centers[static_cast<std::size_t>(i0 + a + r)][static_cast<std::size_t>(c)];
                tob[static_cast<std::size_t>(3 * r + c)] = g.seg_tangents[static_cast<std::size_t>(i0 + a + r)][static_cast<std::size_t>(c)];
            }
        std::vector<cd> table(static_cast<std::size_t>(m * sm.n_src));
        somm_core::field_proj_batch(*sm.view, sm.n_reg, m, sm.n_src, obs.data(), tob.data(), sm.srcf.data(),
                                    sm.t_src.data(), 0.0, fr.k, table.data(), 0, true);
        // `remainder_shape_reduce(shp_w, table)`: per k numpy's unfused
        // complex product against (w + 0j), added to a +0.0 accumulator
        for (ll i = 0; i < m; i++) {
            const cd *row = table.data() + i * N * q;
            for (int s = 0; s < 3; s++) {
                const double *w = sm.shp_w.data() + s * N * q;
                for (ll n = 0; n < N; n++) {
                    double are = 0.0, aim = 0.0;
                    for (ll k = 0; k < q; k++) {
                        const double r0 = w[n * q + k], i0w = 0.0;
                        const double r1 = row[n * q + k].real(), i1 = row[n * q + k].imag();
                        const double re = r0 * r1 - i0w * i1;
                        const double im = r0 * i1 + i0w * r1;
                        are = re + are;
                        aim = im + aim;
                    }
                    S[static_cast<std::size_t>(s)][static_cast<std::size_t>((a + i) * N + n)] = cd(are, aim);
                }
            }
        }
    }
    return S;
}

// One band's Φ triple (`_field_tensor` on the real sources), less the
// image's over a perfect ground (`Phi_c -= Phi_i[0]`, ...), or over the
// composing Sommerfeld ground less `c2 * Phi_img - S`.
std::vector<std::vector<cd>> band(const Kernel &K, ll i0, ll i1, bool cos1, bool pec,
                                  const std::vector<Point3> &img_c, const std::vector<Point3> &img_t,
                                  const Somm *somm = nullptr) {
    const Geometry &g = K.fr->geometry;
    std::vector<std::vector<cd>> phi = cos1 ? field_tensor_cos1(K, i0, i1, g.seg_centers, g.seg_tangents)
                                            : field_tensor(K, i0, i1, g.seg_centers, g.seg_tangents);
    if (!pec && somm == nullptr) return phi;
    std::vector<std::vector<cd>> img =
        cos1 ? field_tensor_cos1(K, i0, i1, img_c, img_t) : field_tensor(K, i0, i1, img_c, img_t);
    if (somm != nullptr) {
        const std::vector<std::vector<cd>> S = somm_replay(*K.fr, *somm, i0, i1);
        for (int c = 0; c < 3; c++)
            for (std::size_t q = 0; q < img[static_cast<std::size_t>(c)].size(); q++) {
                cd &y = img[static_cast<std::size_t>(c)][q];
                y = nec5::replay::np_cmul(somm->c2, y);  // np.multiply(c2, Pi, out=Pi)
                const cd s = S[static_cast<std::size_t>(c)][q];
                y = cd(y.real() - s.real(), y.imag() - s.imag());  // Pi -= Si
            }
    }
    for (int c = 0; c < 3; c++)
        for (std::size_t q = 0; q < phi[static_cast<std::size_t>(c)].size(); q++) {
            cd &x = phi[static_cast<std::size_t>(c)][q];
            const cd y = img[static_cast<std::size_t>(c)][q];
            x = cd(x.real() - y.real(), x.imag() - y.imag());
        }
    return phi;
}


// --- a wholly-buried deck (momwire#1607 B6a, `_operating_medium`) -----------
//
// The whole solve at the lower medium's k_m / eta_m: `_field_tensor` on the
// complex twin (`sinusoidal_field_tensor_cplx`, both shape sets, a radius per
// observer), the image at k_m weighted A_m, and the below-family remainder
// (`_somm_remainder_below_prepare` / `_replay_somm_remainder_below`: the
// below grid, `remainder_field_proj_below`, the source shapes at k_m),
// composed `Phi - (A_m * Phi_img - S)` as the Sommerfeld band composes it.

// numpy's complex absolute (`simd_cabsolute`, AVX2 and AVX512F alike):
// larger * sqrt(fma(r, r, 1)), r = smaller / larger
double np_cabs(cd z) {
    const double re = std::fabs(z.real()), im = std::fabs(z.imag());
    if (std::isnan(re) || std::isnan(im)) return std::numeric_limits<double>::quiet_NaN();
    if (std::isinf(re) || std::isinf(im)) return std::numeric_limits<double>::infinity();
    const double larger = std::max(re, im), smaller = std::min(im, re);
    if (larger == 0.0) return larger;
    const double r = smaller / larger;
    return std::sqrt(mw_fma::exact(r, r, 1.0)) * larger;
}

// CPython 3.12's complex product (`_Py_c_prod`, a float promoted to (x, 0))
cd py_prod(cd a, cd b) {
    return cd(a.real() * b.real() - a.imag() * b.imag(), a.real() * b.imag() + a.imag() * b.real());
}

// numpy's complex sqrt (`npy_csqrt`: libm's csqrt)
cd np_csqrt(cd z) {
#if defined(__GNUC__)
    __complex__ double c;
    __real__ c = z.real();
    __imag__ c = z.imag();
    c = __builtin_csqrt(c);
    return cd(__real__ c, __imag__ c);
#else
    return std::sqrt(z);
#endif
}

// A ValueError the below remainder raises, as the refusal printout's words.
struct BelowRefusal {
    [[noreturn]] static void value(const std::string &m) { throw FrontError(m); }
};

// `_field_tensor(geom, k, src_c, src_t, obs_rows=(i0, i1), cos_shape)` at a
// complex k: the complex twin over the band
std::vector<std::vector<cd>> field_tensor_cplx(const Front &fr, const Kernel &K, ll i0, ll i1,
                                               const std::vector<Point3> &src_c, const std::vector<Point3> &src_t,
                                               cd k, cd eta, bool folded, const std::vector<double> &rad) {
    const Geometry &g = fr.geometry;
    const ll n = g.n_segs, m = i1 - i0;
    std::vector<double> a_obs(static_cast<std::size_t>(m));
    for (ll i = 0; i < m; i++)
        a_obs[static_cast<std::size_t>(i)] = fr.uniform_radius ? fr.radius_per_wire[0] : rad[static_cast<std::size_t>(i0 + i)];
    CF oc = pts(g.seg_centers, i0, i1), ot = pts(g.seg_tangents, i0, i1);
    CF ao = f64(a_obs.data(), {m});
    CF sc = pts(src_c, 0, n), st = pts(src_t, 0, n);
    CF h = f64(g.seg_h.data(), {n});
    const ll q = static_cast<ll>(K.gx.size());
    CF gx = f64(K.gx.data(), {q}), gw = f64(K.gw.data(), {q});
    const Tensors t = sinusoidal_field_tensor_cplx(oc, ot, ao, sc, st, h, k, eta, gx, gw, folded, 0);
    std::vector<std::vector<cd>> out(3);
    const nd::array_t<cd> *parts[3] = {&std::get<0>(t), &std::get<1>(t), &std::get<2>(t)};
    for (int c = 0; c < 3; c++) {
        if (parts[c]->size() != m * n) throw std::logic_error("field tensor: unexpected shape");
        out[static_cast<std::size_t>(c)].assign(parts[c]->data(), parts[c]->data() + m * n);
    }
    return out;
}

struct SommBelow {
    cd k_m, a_m;
    double k_p = 0;
    ll q = 0, n_src = 0;
    std::vector<cd> shp_w;  // (3, N, q)
    somm_below::Remainder<BelowRefusal> rem;
};

// `_below_interface.serve_plan(...)`'s below/below section over the field
// rule's nodes (`field_nodes`, q = `n_qp_buried_field(3)` = 6) with
// bspline's `_pair_extents_below`: `r1_below`, or the grazing refusal
// over the below segments `segs` (every segment when null); `centres` adds
// the below segments' midpoints, the collocation points a mixed deck's plan
// joins (`_mixed_serve_plan`'s `test_obs`)
double serve_plan_below(const Front &fr, const std::vector<ll> *segs = nullptr, bool centres = false, ll q = 6) {
    const Geometry &g = fr.geometry;
    std::vector<ll> all;
    if (segs == nullptr) {
        for (ll i = 0; i < g.n_segs; i++) all.push_back(i);
        segs = &all;
    }
    const double gz = 0.0;
    // `q`: max(n_qp_sommerfeld, N_QP_BURIED_FIELD) * near_q_factor(...), 6
    // unless a mixed deck's cross pairs raise it (momwire#1718 S12)
    std::vector<double> xg, wg;
    if (!razor::leggauss(q, xg, wg)) decline("an untabled buried field order");
    std::vector<double> x, y, d;
    for (ll i : *segs) {
        const Point3 &l = g.seg_l[static_cast<std::size_t>(i)], &r = g.seg_r[static_cast<std::size_t>(i)];
        for (ll j = 0; j < q; j++) {
            const double tq = 0.5 * (xg[static_cast<std::size_t>(j)] + 1.0);
            double p[3];
            for (int c = 0; c < 3; c++) p[c] = l[static_cast<std::size_t>(c)] + tq * (r[static_cast<std::size_t>(c)] - l[static_cast<std::size_t>(c)]);
            x.push_back(p[0]);
            y.push_back(p[1]);
            d.push_back(gz - p[2]);
        }
    }
    if (centres)
        for (ll i : *segs) {
            const Point3 &c = g.seg_centers[static_cast<std::size_t>(i)];
            x.push_back(c[0]);
            y.push_back(c[1]);
            d.push_back(gz - c[2]);
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
    const double lam_m = (2.0 * kPi) / std::hypot(fr.medium.k_m.real(), fr.medium.k_m.imag());
    const double r1_cap = (somm_below::R1_CAP_LAMBDA_M * lam_m) * (1.0 + 1e-9);  // `plan_r1_cap(below_r1_cap(k_m))`
    const double floor_deg = 0.05 - 2.0 * (0.05 / 3.0);  // _SOMM_BELOW_TH_MIN_DEG
    const double floor = floor_deg * (kPi / 180.0);     // math.radians
    if (th_min < floor) {
        // past the cap Python asks the floor of the pairs inside it only
        if (r1_max > r1_cap) decline("a grazing below/below pair past the cap (the capped walk is not copied)");
        decline("a below/below pair under the grazing floor (the refusal's wording is not copied)");
    }
    return r1_max;
}

void somm_below_prepare(const Front &fr, bool cos1, double r1_below, SommBelow &sb) {
    const Geometry &g = fr.geometry;
    const ll N = g.n_segs;
    const double gz = 0.0;
    const Medium &md = fr.medium;
    sb.k_m = md.k_m;
    sb.k_p = md.k_p;
    sb.a_m = md.a_m;
    std::vector<double> gx, gw;
    if (!razor::leggauss(kNQpSommerfeld, gx, gw)) throw std::logic_error("leggauss(3) is tabled");
    const ll q = kNQpSommerfeld;
    sb.q = q;
    sb.n_src = N * q;
    std::vector<double> &src = sb.rem.src, &t_src = sb.rem.t_src;
    src.resize(static_cast<std::size_t>(3 * N * q));
    t_src.resize(static_cast<std::size_t>(3 * N * q));
    sb.shp_w.resize(static_cast<std::size_t>(3 * N * q));
    const cd half_k = py_prod(cd(0.5, 0.0), md.k_m);  // `0.5 * k_m`, CPython's
    for (ll n = 0; n < N; n++) {
        const std::size_t nz = static_cast<std::size_t>(n);
        const double h_half = 0.5 * g.seg_h[nz];
        for (ll j = 0; j < q; j++) {
            const std::size_t jz = static_cast<std::size_t>(j);
            const double zloc = h_half * gx[jz];
            const std::size_t at = static_cast<std::size_t>(n * q + j);
            for (int c = 0; c < 3; c++) {
                src[3 * at + static_cast<std::size_t>(c)] =
                    g.seg_centers[nz][static_cast<std::size_t>(c)] + zloc * g.seg_tangents[nz][static_cast<std::size_t>(c)];
                t_src[3 * at + static_cast<std::size_t>(c)] = g.seg_tangents[nz][static_cast<std::size_t>(c)];
            }
            const double w_node = h_half * gw[jz];
            const cd kz = nec5::replay::np_cmul(md.k_m, cd(zloc, 0.0));  // `k_m * zloc`
            cd shp_cos;
            if (cos1) {
                const cd s = mw_libm_c::csin(nec5::replay::np_cmul(half_k, cd(zloc, 0.0)));
                shp_cos = nec5::replay::np_cmul(cd(-2.0, 0.0), nec5::replay::np_cmul(s, s));  // -2 * sin(.)**2
            } else {
                shp_cos = mw_libm_c::ccos(kz);
            }
            const cd shp[3] = {cd(1.0, 0.0), mw_libm_c::csin(kz), shp_cos};
            for (int c = 0; c < 3; c++)
                sb.shp_w[static_cast<std::size_t>((c * N + n) * q + j)] = nec5::replay::np_cmul(shp[c], cd(w_node, 0.0));
        }
    }
    sb.rem.gz = gz;
    sb.rem.k_p = md.k_p;
    sb.rem.k_m = md.k_m;
    sb.rem.d_src.resize(static_cast<std::size_t>(N * q));
    for (std::size_t j = 0; j < sb.rem.d_src.size(); j++) sb.rem.d_src[j] = gz - src[3 * j + 2];
    somm_below::Source *source = somm_below::source();
    if (source == nullptr) decline("no below/below table source (the buried fill's grid)");
    sb.rem.grid = &source->get(md.eps_t, md.k_p, r1_below, fr.omega, kMu, somm_below::SPACING_FILL);
    sb.rem.build_view();
}

// `_replay_somm_remainder_below` for observer rows [i0, i1): S (3, M, N)
std::vector<std::vector<cd>> somm_below_replay(const Front &fr, SommBelow &sb, ll i0, ll i1) {
    const Geometry &g = fr.geometry;
    const ll q = sb.q, N = sb.n_src / std::max<ll>(1, q), M = i1 - i0;  // the sources' count
    std::vector<std::vector<cd>> S(3, std::vector<cd>(static_cast<std::size_t>(M * N)));
    const ll chunk = std::max<ll>(1, kRemainderChunkElems / std::max<ll>(1, sb.n_src));  // `elem_rows`
    std::vector<double> obs, tob;
    for (ll a = 0; a < M; a += chunk) {
        const ll b = std::min(a + chunk, M), m = b - a;
        obs.assign(static_cast<std::size_t>(3 * m), 0.0);
        tob.assign(static_cast<std::size_t>(3 * m), 0.0);
        for (ll r = 0; r < m; r++)
            for (int c = 0; c < 3; c++) {
                obs[static_cast<std::size_t>(3 * r + c)] = g.seg_centers[static_cast<std::size_t>(i0 + a + r)][static_cast<std::size_t>(c)];
                tob[static_cast<std::size_t>(3 * r + c)] = g.seg_tangents[static_cast<std::size_t>(i0 + a + r)][static_cast<std::size_t>(c)];
            }
        std::vector<cd> table(static_cast<std::size_t>(m * sb.n_src));
        sb.rem.project(obs.data(), tob.data(), m, table.data());
        // `remainder_shape_reduce(shp_w, table)`: per k numpy's unfused
        // complex product, added to a +0.0 accumulator
        for (ll i = 0; i < m; i++) {
            const cd *row = table.data() + i * N * q;
            for (int s = 0; s < 3; s++) {
                const cd *w = sb.shp_w.data() + s * N * q;
                for (ll n = 0; n < N; n++) {
                    double are = 0.0, aim = 0.0;
                    for (ll k = 0; k < q; k++) {
                        const double r0 = w[n * q + k].real(), i0w = w[n * q + k].imag();
                        const double r1 = row[n * q + k].real(), i1 = row[n * q + k].imag();
                        const double re = r0 * r1 - i0w * i1;
                        const double im = r0 * i1 + i0w * r1;
                        are = re + are;
                        aim = im + aim;
                    }
                    S[static_cast<std::size_t>(s)][static_cast<std::size_t>((a + i) * N + n)] = cd(are, aim);
                }
            }
        }
    }
    return S;
}

// One band of a wholly-buried fill: `Phi - (A_m * Phi_img - S)`
std::vector<std::vector<cd>> band_buried(const Front &fr, const Kernel &K, ll i0, ll i1, bool cos1, cd eta,
                                         const std::vector<Point3> &img_c, const std::vector<Point3> &img_t,
                                         SommBelow &sb, const std::vector<double> &rad) {
    const Geometry &g = fr.geometry;
    std::vector<std::vector<cd>> phi =
        field_tensor_cplx(fr, K, i0, i1, g.seg_centers, g.seg_tangents, sb.k_m, eta, cos1, rad);
    std::vector<std::vector<cd>> img = field_tensor_cplx(fr, K, i0, i1, img_c, img_t, sb.k_m, eta, cos1, rad);
    const std::vector<std::vector<cd>> S = somm_below_replay(fr, sb, i0, i1);
    for (int c = 0; c < 3; c++)
        for (std::size_t qq = 0; qq < img[static_cast<std::size_t>(c)].size(); qq++) {
            cd &y = img[static_cast<std::size_t>(c)][qq];
            y = nec5::replay::np_cmul(sb.a_m, y);  // np.multiply(a_m, Pi, out=Pi)
            const cd s = S[static_cast<std::size_t>(c)][qq];
            y = cd(y.real() - s.real(), y.imag() - s.imag());  // Pi -= Si
        }
    for (int c = 0; c < 3; c++)
        for (std::size_t qq = 0; qq < phi[static_cast<std::size_t>(c)].size(); qq++) {
            cd &x = phi[static_cast<std::size_t>(c)][qq];
            const cd y = img[static_cast<std::size_t>(c)][qq];
            x = cd(x.real() - y.real(), x.imag() - y.imag());
        }
    return phi;
}


// --- a crossing deck (momwire#1607 B6b, `_assemble_Z_mixed`) ---------------
//
// Each medium's own block, `_class_bands`: its sources (the class's
// segments) seen at its own midpoints, at its own (k, eta), with its own
// ground, `Phi - (coef * Phi_img - S)`, the remainder prepared over the
// class's geometry: above at k_p (the reduced kernel, the Sommerfeld
// remainder, C2), below at k_m (the complex twin, the below remainder,
// A_m). The cross quadrants are the point rows (`_crossing_point_rows`,
// `_sinusoidal_crossing_core.cpp`), added to Z after the reduction.

// `_sommerfeld_remainder_prepare(cls_geom, k_p, eps_t, cos_shape, r1_max)`
// over the class's segments `idx`
void somm_prepare_cls(const Front &fr, const nec4::SolverArgs &args, bool cos1, const std::vector<ll> &idx,
                      Somm &sm) {
    const Geometry &g = fr.geometry;
    const ll N = static_cast<ll>(idx.size());
    const double gz = 0.0;
    sm.eps_t = cd(args.ground.eps_r, (-args.ground.sigma) / (fr.omega * 8.8541878188e-12));
    sm.c2 = py_quot(cd(sm.eps_t.real() - 1.0, sm.eps_t.imag() - 0.0),
                    cd(sm.eps_t.real() + 1.0, sm.eps_t.imag() + 0.0));
    for (ll s : idx) {
        const std::size_t z = static_cast<std::size_t>(s);
        if (g.seg_l[z][2] - gz < -1e-12 || g.seg_r[z][2] - gz < -1e-12)
            decline("an above class below the Sommerfeld ground (the reference raises)");
    }
    std::vector<double> gx, gw;
    if (!razor::leggauss(kNQpSommerfeld, gx, gw)) throw std::logic_error("leggauss(3) is tabled");
    const ll q = kNQpSommerfeld;
    sm.q = q;
    sm.n_src = N * q;
    const double k = fr.k;
    sm.srcf.resize(static_cast<std::size_t>(3 * N * q));
    sm.t_src.resize(static_cast<std::size_t>(3 * N * q));
    sm.shp_w.resize(static_cast<std::size_t>(3 * N * q));
    for (ll n = 0; n < N; n++) {
        const std::size_t nz = static_cast<std::size_t>(idx[static_cast<std::size_t>(n)]);
        const double h_half = 0.5 * g.seg_h[nz];
        for (ll j = 0; j < q; j++) {
            const std::size_t jz = static_cast<std::size_t>(j);
            const double zloc = h_half * gx[jz];
            const std::size_t at = static_cast<std::size_t>(n * q + j);
            for (int c = 0; c < 3; c++) {
                sm.srcf[3 * at + static_cast<std::size_t>(c)] =
                    g.seg_centers[nz][static_cast<std::size_t>(c)] + zloc * g.seg_tangents[nz][static_cast<std::size_t>(c)];
                sm.t_src[3 * at + static_cast<std::size_t>(c)] = g.seg_tangents[nz][static_cast<std::size_t>(c)];
            }
            const double w_node = h_half * gw[jz];
            double shp_cos;
            if (cos1) {
                const double s = std::sin(0.5 * k * zloc);
                shp_cos = -2.0 * (s * s);
            } else {
                shp_cos = std::cos(k * zloc);
            }
            const double shp[3] = {1.0, std::sin(k * zloc), shp_cos};
            for (int c = 0; c < 3; c++)
                sm.shp_w[static_cast<std::size_t>((c * N + n) * q + j)] = shp[c] * w_node;
        }
    }
    std::vector<double> sl(static_cast<std::size_t>(3 * N)), sr(static_cast<std::size_t>(3 * N));
    for (ll n = 0; n < N; n++)
        for (int c = 0; c < 3; c++) {
            const std::size_t z = static_cast<std::size_t>(idx[static_cast<std::size_t>(n)]);
            sl[static_cast<std::size_t>(3 * n + c)] = g.seg_l[z][static_cast<std::size_t>(c)];
            sr[static_cast<std::size_t>(3 * n + c)] = g.seg_r[z][static_cast<std::size_t>(c)];
        }
    const double r1_max = razor::max_image_distance(sl.data(), sr.data(), static_cast<std::size_t>(N), gz);
    const somm_core::Grid &grid = somm_core::get_grid_lazy(sm.eps_t, k, r1_max, fr.omega, kMu);
    somm_core::ensure_nodes(grid, somm_core::all_nodes_of(grid));
    sm.grid = &grid;
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
        grid.r1_max, grid.r_break, 30.0 * (M_PI / 180.0), grid.r_near, r0, dr, th0, dth, sm.n_reg, vals, shape0,
        nr, nth));
    const std::vector<cd> far = somm_core::far_pack(grid);
    somm_proj::set_far(*sm.view, far.data(), static_cast<std::ptrdiff_t>(far.size()));
}

// `_replay_sommerfeld_remainder` over the class's sources, for global
// observer rows [i0, i1): S (3, M, N_class)
std::vector<std::vector<cd>> somm_replay_cls(const Front &fr, const Somm &sm, ll N, ll i0, ll i1) {
    const Geometry &g = fr.geometry;
    const ll M = i1 - i0, q = sm.q;
    std::vector<std::vector<cd>> S(3, std::vector<cd>(static_cast<std::size_t>(M * N)));
    const ll chunk = std::max<ll>(1, kRemainderChunkElems / std::max<ll>(1, sm.n_src));
    std::vector<double> obs, tob;
    for (ll a = 0; a < M; a += chunk) {
        const ll b = std::min(a + chunk, M), m = b - a;
        obs.assign(static_cast<std::size_t>(3 * m), 0.0);
        tob.assign(static_cast<std::size_t>(3 * m), 0.0);
        for (ll r = 0; r < m; r++)
            for (int c = 0; c < 3; c++) {
                obs[static_cast<std::size_t>(3 * r + c)] = g.seg_centers[static_cast<std::size_t>(i0 + a + r)][static_cast<std::size_t>(c)];
                tob[static_cast<std::size_t>(3 * r + c)] = g.seg_tangents[static_cast<std::size_t>(i0 + a + r)][static_cast<std::size_t>(c)];
            }
        std::vector<cd> table(static_cast<std::size_t>(m * sm.n_src));
        somm_core::field_proj_batch(*sm.view, sm.n_reg, m, sm.n_src, obs.data(), tob.data(), sm.srcf.data(),
                                    sm.t_src.data(), 0.0, fr.k, table.data(), 0, true);
        for (ll i = 0; i < m; i++) {
            const cd *row = table.data() + i * N * q;
            for (int s = 0; s < 3; s++) {
                const double *w = sm.shp_w.data() + s * N * q;
                for (ll n = 0; n < N; n++) {
                    double are = 0.0, aim = 0.0;
                    for (ll k = 0; k < q; k++) {
                        const double r0 = w[n * q + k], i0w = 0.0;
                        const double r1 = row[n * q + k].real(), i1 = row[n * q + k].imag();
                        const double re = r0 * r1 - i0w * i1;
                        const double im = r0 * i1 + i0w * r1;
                        are = re + are;
                        aim = im + aim;
                    }
                    S[static_cast<std::size_t>(s)][static_cast<std::size_t>((a + i) * N + n)] = cd(are, aim);
                }
            }
        }
    }
    return S;
}

// The class's sources (centres, tangents, lengths) and their mirror images.
struct ClassSources {
    std::vector<ll> idx;
    std::vector<Point3> c, t, img_c, img_t;
    std::vector<double> h;
};
ClassSources class_sources(const Front &fr, bool below_side) {
    const Geometry &g = fr.geometry;
    ClassSources cs;
    for (ll s = 0; s < g.n_segs; s++) {
        const std::size_t z = static_cast<std::size_t>(s);
        if (fr.below_seg[z] != below_side) continue;
        cs.idx.push_back(s);
        cs.c.push_back(g.seg_centers[z]);
        cs.t.push_back(g.seg_tangents[z]);
        cs.h.push_back(g.seg_h[z]);
        // `_ground_mirror`: `2 * ground_z - z`, `tangents * [1, 1, -1]`
        cs.img_c.push_back({g.seg_centers[z][0], g.seg_centers[z][1], 2.0 * 0.0 - g.seg_centers[z][2]});
        cs.img_t.push_back({g.seg_tangents[z][0] * 1.0, g.seg_tangents[z][1] * 1.0, g.seg_tangents[z][2] * -1.0});
    }
    return cs;
}

// `_field_tensor(geom_src, k, src_centers, src_tangents, obs_rows=(i0, i1))`
// over a class's sources: the reduced kernel at a real k (one radius), or
// the complex twin at k_m
std::vector<std::vector<cd>> class_tensor(const Front &fr, const Kernel &K, ll i0, ll i1, const std::vector<Point3> &sc_,
                                          const std::vector<Point3> &st_, const std::vector<double> &sh, bool complex_k,
                                          cd k, cd eta, bool folded) {
    const Geometry &g = fr.geometry;
    const ll n = static_cast<ll>(sc_.size()), m = i1 - i0;
    CF oc = pts(g.seg_centers, i0, i1), ot = pts(g.seg_tangents, i0, i1);
    CF sc = pts(sc_, 0, n), st = pts(st_, 0, n);
    CF h = f64(sh.data(), {n});
    const ll q = static_cast<ll>(K.gx.size());
    CF gx = f64(K.gx.data(), {q}), gw = f64(K.gw.data(), {q});
    Tensors t;
    if (complex_k) {
        std::vector<double> a_obs(static_cast<std::size_t>(m), fr.radius_per_wire[0]);
        CF ao = f64(a_obs.data(), {m});
        t = sinusoidal_field_tensor_cplx(oc, ot, ao, sc, st, h, k, eta, gx, gw, folded, 0);
    } else if (folded) {
        // the accelerated kernel serves the literal cos only: numpy's route
        return field_tensor_reduced_cos1(fr, K, i0, i1, sc_, st_, sh, k.real(), eta.real());
    } else {
        t = sinusoidal_field_tensor(oc, ot, sc, st, h, fr.radius_per_wire[0], k.real(), eta.real(), gx, gw, 0);
    }
    std::vector<std::vector<cd>> out(3);
    const nd::array_t<cd> *parts[3] = {&std::get<0>(t), &std::get<1>(t), &std::get<2>(t)};
    for (int c = 0; c < 3; c++) {
        if (parts[c]->size() != m * n) throw std::logic_error("field tensor: unexpected shape");
        out[static_cast<std::size_t>(c)].assign(parts[c]->data(), parts[c]->data() + m * n);
    }
    return out;
}

// --- `_contact_charge_correction` over a Sommerfeld ground ------------------
//
// A wire end on the finite ground (`_contact_nodes`: every `ground_minus`
// segment's start, then every `ground_plus` segment's end) leaves an end
// charge whose field the image does not cancel there: its kernel
// (`_contact_charge_kernel`, the Sommerfeld branch), plus the EK end's delta
// where the end is extended (`_contact_charge_ek_delta`), times the end's
// basis values (`_contact_node_values`), subtracted from the columns of its
// basis: `G[:, jb] -= sgn * R[:, None] * val[None, :]`. Every operation is
// numpy's over the N observers. No product here depends on numpy's temporary
// elision: the two whose right operand was a fresh temporary are spelled out
// of it in Python (momwire#1718 S7), so they are computed as written.

// numpy's real matmul `(N, 3) @ (3,)`: `DOUBLE_matmul`'s matrix-vector gemv,
// `cblas_dgemv(ColMajor, Trans, 3, N, 1, a, 3, x, 1, 0, y, 1)`
std::vector<double> mat3_vec(const nec5::NumpyLinalg &np, const std::vector<double> &a, const double *x) {
    const ll n = static_cast<ll>(a.size() / 3);
    std::vector<double> y(static_cast<std::size_t>(n));
    if (n == 1) {
        // numpy's `DOUBLE_dot` for a (1, 3) @ (3,): `sum += ddot`
        if (np.ddot == nullptr) decline("no real BLAS for the contact correction");
        y[0] = 0.0 + np.ddot(3, a.data(), 1, x, 1);
        return y;
    }
    if (np.dgemv == nullptr) decline("no real BLAS for the contact correction");
    np.dgemv(102, 112, 3, n, 1.0, a.data(), 3, x, 1, 0.0, y.data(), 1);
    return y;
}

void contact_correction(const Front &fr, const nec4::SolverArgs &args, const Kernel &K, const nec5::NumpyLinalg &np,
                        std::vector<cd> &G) {
    const Geometry &g = fr.geometry;
    const SegView &sv = fr.basis;
    const ll N = g.n_segs;
    const std::size_t Nz = static_cast<std::size_t>(N);
    const double k = fr.k;
    const double eta = fr.eta;
    // CPython scalars: -1j * eta / (4 pi k), 1j * k, -1j * k, con = 1j * eta / (4 pi k)
    const double four_pi_k = 4.0 * 3.141592653589793 * k;
    const cd pre0 = py_quot(cd(-0.0 * eta - (-1.0) * 0.0, -0.0 * 0.0 + (-1.0) * eta), cd(four_pi_k, 0.0));
    const cd jk(0.0 * k - 1.0 * 0.0, 0.0 * 0.0 + 1.0 * k), mjk(-0.0 * k - (-1.0) * 0.0, -0.0 * 0.0 + (-1.0) * k);
    const cd con = py_quot(cd(0.0 * eta - 1.0 * 0.0, 0.0 * 0.0 + 1.0 * eta), cd(four_pi_k, 0.0));
    const cd mcon(-con.real(), -con.imag());
    const cd eps_t(args.ground.eps_r, (-args.ground.sigma) / (fr.omega * 8.8541878188e-12));
    const cd c2 = py_quot(cd(eps_t.real() - 1.0, eps_t.imag() - 0.0), cd(eps_t.real() + 1.0, eps_t.imag() + 0.0));
    const cd one_minus_c2(1.0 - c2.real(), 0.0 - c2.imag());
    const std::vector<double> &a_obs = K.src_a.empty() ? std::vector<double>() : K.src_a;
    std::vector<double> rad = a_obs;
    if (rad.empty()) decline("a contact correction without the segment radii");
    std::vector<double> obs_t_flat(3 * Nz);
    for (std::size_t m = 0; m < Nz; m++)
        for (int c = 0; c < 3; c++) obs_t_flat[3 * m + static_cast<std::size_t>(c)] = g.seg_tangents[m][static_cast<std::size_t>(c)];
    // `_contact_nodes`
    struct Node {
        ll i;
        double sgn;
        double p[3];
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
                for (int c = 0; c < 3; c++)
                    nd.p[c] = g.seg_centers[i][static_cast<std::size_t>(c)] + sc * g.seg_tangents[i][static_cast<std::size_t>(c)];
                nodes.push_back(nd);
            }
    }
    for (const Node &nd : nodes) {
        const std::size_t iz = static_cast<std::size_t>(nd.i);
        // `_contact_charge_kernel` (Sommerfeld): pref * (1 - c2) * t_d
        std::vector<cd> pref(Nz), R(Nz);
        std::vector<double> t_d(Nz);
        for (std::size_t m = 0; m < Nz; m++) {
            double d[3];
            for (int c = 0; c < 3; c++) d[c] = g.seg_centers[m][static_cast<std::size_t>(c)] - nd.p[c];
            const double r2 = (d[0] * d[0] + d[2] * d[2]) + d[1] * d[1];
            const double r0 = std::sqrt(r2 + rad[m] * rad[m]);
            const cd t1 = radd(1.0, cm(jk, cd(r0, 0.0)));
            const cd e = std::exp(cm(mjk, cd(r0, 0.0)));
            pref[m] = cdr(cm(cm(pre0, t1), e), (r0 * r0) * r0);
            const Point3 &ot = g.seg_tangents[m];
            t_d[m] = (ot[0] * d[0] + ot[2] * d[2]) + ot[1] * d[1];
            R[m] = cr(cm(pref[m], one_minus_c2), t_d[m]);
        }
        // `_contact_ek_masks`: the extended end's delta
        const bool extended = args.extended_kernel && (nd.sgn > 0 ? K.ind2 : K.ind1)[iz] != 2;
        if (extended) {
            // `_contact_charge_end_gradient`
            const Point3 &src_c = g.seg_centers[iz], &src_t = g.seg_tangents[iz];
            const double src_hh = 0.5 * g.seg_h[iz];
            const double src_a = fr.uniform_radius ? fr.radius_per_wire[0] : rad[iz];
            std::vector<double> rvec(3 * Nz);
            for (std::size_t m = 0; m < Nz; m++)
                for (int c = 0; c < 3; c++)
                    rvec[3 * m + static_cast<std::size_t>(c)] =
                        g.seg_centers[m][static_cast<std::size_t>(c)] - src_c[static_cast<std::size_t>(c)];
            const std::vector<double> z_eval = mat3_vec(np, rvec, src_t.data());   // rvec @ src_t
            const std::vector<double> ts = mat3_vec(np, obs_t_flat, src_t.data());  // obs_t @ src_t
            for (std::size_t m = 0; m < Nz; m++) {
                double rho_vec[3];
                for (int c = 0; c < 3; c++)
                    rho_vec[c] = rvec[3 * m + static_cast<std::size_t>(c)] - z_eval[m] * src_t[static_cast<std::size_t>(c)];
                const double rho_axis =
                    std::sqrt((rho_vec[0] * rho_vec[0] + rho_vec[1] * rho_vec[1]) + rho_vec[2] * rho_vec[2]);
                const double rho_eval = std::sqrt(rho_axis * rho_axis + rad[m] * rad[m]);
                const double zz = nd.sgn > 0 ? (src_hh - z_eval[m]) : -(src_hh + z_eval[m]);
                const bool swap = rho_eval < src_a;
                const double rh = swap ? src_a : rho_eval;
                const double b = swap ? rho_eval : src_a;
                const End e = ek_end_gxx(k, zz, rh, b, swap);
                const cd ez = cm(mcon, e.g1p);
                const cd erho_hat = cdr(cm(con, e.g3), rho_eval);
                // `_contact_charge_ek_delta`
                const Point3 &ot = g.seg_tangents[m];
                const double rho_t = (rho_vec[0] * ot[0] + rho_vec[2] * ot[2]) + rho_vec[1] * ot[1];
                const cd dt = csub(cadd(cr(ez, ts[m]), cr(erho_hat, rho_t)), cr(pref[m], t_d[m]));
                const cd delta = csub(dt, cm(c2, dt));
                R[m] = cadd(R[m], delta);
            }
        }
        // `_contact_node_values`
        const double sseg = nd.sgn * 0.5 * g.seg_h[iz];
        const double half = std::sin(0.5 * k * sseg);
        const double sks = std::sin(k * sseg);
        for (ll q = sv.starts[iz]; q < sv.starts[iz + 1]; q++) {
            const std::size_t qz = static_cast<std::size_t>(q);
            const cd sig(static_cast<double>(sv.sigma[qz]), 0.0);
            const cd val = csub(cadd(cm(sig, sv.AC[qz]), cr(sv.B[qz], sks)),
                                cr(cr(cr(cm(sig, sv.C[qz]), 2.0), half), half));
            const std::size_t col = static_cast<std::size_t>(sv.jbasis[qz]);
            for (std::size_t m = 0; m < Nz; m++) {
                const cd t = cm(cm(cd(nd.sgn, 0.0), R[m]), val);  // sgn * R[:, None] * val[None, :]
                cd &x = G[m * Nz + col];
                x = cd(x.real() - t.real(), x.imag() - t.imag());
            }
        }
    }
}

// The (segment, basis) entries of `_assemble_Z`'s M_A / M_B / M_C.
struct Entries {
    std::vector<ll> n, j;
    std::vector<cd> v[3];
};

// `_reduce_tiles(n)`
std::vector<std::pair<ll, ll>> reduce_tiles(ll n) {
    std::vector<ll> starts;
    for (ll s = 0; s < n; s += std::max<ll>(2, kReduceTile)) starts.push_back(s);
    if (n > 1 && n - starts.back() == 1) starts.pop_back();
    std::vector<std::pair<ll, ll>> t;
    for (std::size_t k = 0; k < starts.size(); k++)
        t.push_back({starts[k], k + 1 < starts.size() ? starts[k + 1] : n});
    return t;
}

// `configure_loading`'s wire loading from the solver arguments (LD 5's
// conductivity, LD 2 / LD 3's distributed RLC); a conductivity <= 0 is the
// constructor's ValueError, which this copy declines.
razor::WireLoading wire_loading(const nec4::SolverArgs &args) {
    razor::WireLoading w;
    if (args.has_conductivity) {
        w.has_conductivity = true;
        w.conductivity = args.conductivity;
        for (double c : w.conductivity)
            if (std::isfinite(c) && c <= 0.0) decline("a wire conductivity <= 0 (the reference raises)");
    }
    if (args.has_distributed) {
        w.has_rlc = true;
        for (std::size_t i = 0; i < args.distributed.size(); i++) {
            razor::WireRLC r;
            if (i < args.distributed_present.size() && args.distributed_present[i]) {
                const nec4::DistributedRLC &d = args.distributed[i];
                if (d.kind == "series")
                    r.kind = razor::WireRLC::series;
                else if (d.kind == "parallel")
                    r.kind = razor::WireRLC::parallel;
                else
                    decline("a distributed load kind this copy does not read");
                r.r = d.r;
                r.l = d.l;
                r.c = d.c;
            }
            w.rlc.push_back(r);
        }
    }
    return w;
}

// The per-metre series impedance each wire carries
// (`series_impedance_per_wire` at `omega = k * c`, the conductor radius).
std::vector<cd> wire_impedance(const Front &fr, const razor::WireLoading &loading, double omega) {
    return razor::series_impedance_per_wire(omega, fr.radius_per_wire, loading);
}

// `_apply_loading(G, geom, seg_view, k)`: `G[rows, cols] -= z_seg[rows] *
// (sigma * AC)` (not A + C, #606), the entries in their CSR order.
void apply_loading(const Front &fr, const razor::WireLoading &loading, const Entries &E, std::vector<cd> &G) {
    if (!loading.active()) return;
    const std::vector<cd> z_wire = wire_impedance(fr, loading, fr.k * fr.c);
    const Geometry &g = fr.geometry;
    const SegView &sv = fr.basis;
    const ll N = g.n_segs;
    std::vector<cd> z_seg(static_cast<std::size_t>(N));
    for (std::size_t w = 0; w < g.wire_first.size(); w++)
        for (ll i = g.wire_first[w]; i <= g.wire_last[w]; i++) z_seg[static_cast<std::size_t>(i)] = z_wire.at(w);
    // numpy's `G[rows, cols] -= v`: every entry read before any is written,
    // so a repeated (row, col) keeps the LAST entry's difference
    std::vector<std::pair<std::size_t, cd>> writes;
    for (ll n = 0; n < N; n++)
        for (ll q = sv.starts[static_cast<std::size_t>(n)]; q < sv.starts[static_cast<std::size_t>(n) + 1]; q++) {
            const std::size_t qz = static_cast<std::size_t>(q);
            const cd ic = sig_mul(sv.sigma[qz], sv.AC[qz]);
            const cd t = nec5::replay::np_cmul(z_seg[static_cast<std::size_t>(n)], ic);
            const std::size_t at = static_cast<std::size_t>(n * N + sv.jbasis[qz]);
            const cd x = G[at];
            writes.push_back({at, cd(x.real() - t.real(), x.imag() - t.imag())});
        }
    for (const auto &w : writes) G[w.first] = w.second;
    (void)E;
}

// `SinusoidalSolver._nec4_cap_columns` (momwire#1644): every free end's cap
// (`_sinusoidal_nec4.end_caps`, scalar float arithmetic), its field from the
// shared kernel, less its image under a ground (`c2 * img` under Sommerfeld,
// numpy's fused product), added into its basis column.
void nec4_cap_columns(const Front &fr, bool pec, const Somm *somm, std::vector<cd> &G) {
    const Geometry &g = fr.geometry;
    const SegView &sv = fr.basis;
    const ll N = g.n_segs;
    const std::size_t Nz = static_cast<std::size_t>(N);
    const double k = fr.k;
    const std::vector<double> rad = seg_radius(fr);
    CF oc = pts(g.seg_centers, 0, N), ot = pts(g.seg_tangents, 0, N);
    for (std::size_t i = 0; i < Nz; i++) {
        const bool free_m = g.nm_count[i] == 0 && !g.ground_minus[i] && !g.crossing_minus[i];
        const bool free_p = g.np_count[i] == 0 && !g.ground_plus[i] && !g.crossing_plus[i];
        if (!free_m && !free_p) continue;
        ll e = -1;
        for (ll q = sv.starts[i]; q < sv.starts[i + 1]; q++)
            if (sv.jbasis[static_cast<std::size_t>(q)] == static_cast<ll>(i)) {
                e = q;
                break;
            }
        if (e < 0) throw std::logic_error("a segment without its own basis entry");
        const std::size_t ez = static_cast<std::size_t>(e);
        const double A = sv.A[ez].real(), B = sv.B[ez].real(), C = sv.C[ez].real();
        const double h = g.seg_h[i];
        const double u = 0.5 * k * h;
        const Point3 &c = g.seg_centers[i], &t = g.seg_tangents[i];
        const bool frees[2] = {free_m, free_p};
        const double sgns[2] = {-1.0, 1.0};
        for (int side = 0; side < 2; side++) {
            if (!frees[side]) continue;
            const double sgn = sgns[side];
            const double i_end = A + B * std::sin(sgn * u) + C * std::cos(u);
            const double S = -sgn;
            const double off = sgn * 0.5 * h;
            const double pos[3] = {c[0] + off * t[0], c[1] + off * t[1], c[2] + off * t[2]};
            const double nrm[3] = {sgn * t[0], sgn * t[1], sgn * t[2]};
            const cd c0(0.0, S * i_end / (4.0 * kPi * k));
            const nd::array_t<cd> col =
                sinusoidal_cap_field(oc, ot, f64(pos, {3}), f64(nrm, {3}), rad[i], k, fr.eta, c0);
            std::vector<cd> v(col.data(), col.data() + N);
            if (pec || somm != nullptr) {
                const double ipos[3] = {pos[0], pos[1], 2.0 * 0.0 - pos[2]};
                const double inrm[3] = {nrm[0] * 1.0, nrm[1] * 1.0, nrm[2] * -1.0};
                const nd::array_t<cd> img =
                    sinusoidal_cap_field(oc, ot, f64(ipos, {3}), f64(inrm, {3}), rad[i], k, fr.eta, c0);
                for (std::size_t m = 0; m < Nz; m++) {
                    cd y = img.data()[m];
                    if (somm != nullptr) y = nec5::replay::np_cmul(somm->c2, y);
                    v[m] = cd(v[m].real() - y.real(), v[m].imag() - y.imag());
                }
            }
            for (std::size_t m = 0; m < Nz; m++) {
                cd &x = G[m * Nz + i];
                x = cd(x.real() + v[m].real(), x.imag() + v[m].imag());
            }
        }
    }
}


// `_mixed_serve_plan(geom, below, medium, crossing=False)`'s cross section
// (momwire#1718 S12): the field nodes at order `q` and the midpoints of each
// class (`test_obs=None`, one per segment), through `serve_plan`'s
// cross-medium extents
somm_trans::CrossPlan mixed_cross_plan(const Front &fr, const std::vector<ll> &a_idx, const std::vector<ll> &b_idx,
                                       ll q) {
    const Geometry &g = fr.geometry;
    std::vector<double> xg, wg;
    if (!razor::leggauss(q, xg, wg)) decline("an untabled buried field order");
    const auto points = [&](const std::vector<ll> &idx) {
        std::vector<double> p;
        for (ll i : idx) {
            const Point3 &l = g.seg_l[static_cast<std::size_t>(i)], &r = g.seg_r[static_cast<std::size_t>(i)];
            for (ll j = 0; j < q; j++) {
                const double tq = 0.5 * (xg[static_cast<std::size_t>(j)] + 1.0);
                for (int c = 0; c < 3; c++)
                    p.push_back(l[static_cast<std::size_t>(c)] +
                                tq * (r[static_cast<std::size_t>(c)] - l[static_cast<std::size_t>(c)]));
            }
        }
        for (ll i : idx)
            for (int c = 0; c < 3; c++) p.push_back(g.seg_centers[static_cast<std::size_t>(i)][static_cast<std::size_t>(c)]);
        return p;
    };
    const std::vector<double> pa = points(a_idx), pb = points(b_idx);
    try {
        return somm_trans::cross_extents(pa.data(), pa.size() / 3, pb.data(), pb.size() / 3, 0.0, fr.medium.k_p,
                                         fr.medium.k_m);
    } catch (const somm_trans::Refused &e) {
        decline(std::string("a mixed deck's cross pairs Python refuses (") + e.what() + ")");
    }
}

// `_assemble_Z_mixed(geom, eta)` on a crossing deck (momwire#1223), and on a
// mixed one -- wires on both sides with no crossing junction (momwire#1718
// S12): the two transmitted pair classes over the below->above table in
// place of the crossing's point rows
std::vector<cd> assemble_Z_crossing(const Front &fr, const nec4::SolverArgs &args, const nec5::NumpyLinalg &np) {
    if (args.has_insulation) decline("an insulated wire (momwire#1607 B5b)");
    if (args.extended_kernel) decline("the extended kernel below the interface (the reference refuses)");
    const razor::WireLoading loading = wire_loading(args);
    const Geometry &g = fr.geometry;
    const SegView &sv = fr.basis;
    const ll N = g.n_segs;
    const std::size_t Nz = static_cast<std::size_t>(N);
    const Medium &md = fr.medium;
    const ClassSources above = class_sources(fr, false), below = class_sources(fr, true);
    const bool mixed = !fr.crossing;
    // `_mixed_serve_plan`: on a mixed deck the cross block's order first
    // (`n_qp_buried_field` over `cross_pair_separation`, momwire#1004)
    ll q_x = 6;
    if (mixed) {
        std::vector<double> sl, sr;
        for (ll i = 0; i < N; i++)
            for (int c = 0; c < 3; c++) {
                sl.push_back(g.seg_l[static_cast<std::size_t>(i)][static_cast<std::size_t>(c)]);
                sr.push_back(g.seg_r[static_cast<std::size_t>(i)][static_cast<std::size_t>(c)]);
            }
        const std::vector<int64_t> ai(above.idx.begin(), above.idx.end()), bi(below.idx.begin(), below.idx.end());
        try {
            q_x = 6 * somm_trans::near_q_factor(sl.data(), sr.data(), ai, bi);
        } catch (const somm_trans::Refused &e) {
            decline(std::string("a mixed deck's cross order (") + e.what() + ")");
        }
    }
    // the below/below extents over the field nodes and the below midpoints
    const double r1_below = serve_plan_below(fr, &below.idx, true, q_x);
    somm_trans::CrossPlan cross_plan;
    if (mixed) cross_plan = mixed_cross_plan(fr, above.idx, below.idx, q_x);
    // the shape set over every segment at its own |k| (`abs(k_m)`: CPython's)
    const double k_m_abs = std::hypot(md.k_m.real(), md.k_m.imag());
    double kd_min = std::numeric_limits<double>::infinity();
    bool first = true;
    for (std::size_t i = 0; i < Nz; i++) {
        const double v = (fr.below_seg[i] ? k_m_abs : md.k_p) * g.seg_h[i];
        if (first || v < kd_min || std::isnan(v)) kd_min = v;
        first = false;
    }
    if (std::isnan(kd_min)) decline("a non-finite k*h");
    const bool cos1 = kd_min < kWellScaledKd;
    // a mixed deck's transmitted blocks on the folded shape set
    // (`-2 sin^2(k z / 2)` at the source medium's k) are not copied
    if (cos1 && mixed)
        decline("the folded cos-1 shape on a mixed deck (the transmitted blocks' folded shapes are not copied)");
    Entries E;
    std::map<std::pair<ll, ll>, int> seen;
    for (ll i = 0; i < N; i++)
        for (ll e = sv.starts[static_cast<std::size_t>(i)]; e < sv.starts[static_cast<std::size_t>(i) + 1]; e++) {
            const std::size_t ez = static_cast<std::size_t>(e);
            if (++seen[{i, sv.jbasis[ez]}] > 1) decline("a repeated (segment, basis) entry on a crossing deck");
            E.n.push_back(i);
            E.j.push_back(sv.jbasis[ez]);
            E.v[0].push_back(sig_mul(sv.sigma[ez], cos1 ? sv.AC[ez] : sv.A[ez]));
            E.v[1].push_back(sv.B[ez]);
            E.v[2].push_back(sig_mul(sv.sigma[ez], sv.C[ez]));
        }
    Kernel K;
    K.fr = &fr;
    if (!razor::leggauss(kNQpConst, K.gx, K.gw)) throw std::logic_error("leggauss(8) is tabled");
    if (!fr.uniform_radius)
        decline(mixed ? "a mixed deck with several radii (the class kernels read one)"
                      : "a crossing deck with several radii (the reference refuses)");
    const cd eta_p(fr.eta, 0.0);
    const cd eta_m = np_csqrt(py_quot(cd(kMu, 0.0), md.eps_m));
    // the two classes' remainders, over each class's own geometry
    Somm s_above;
    somm_prepare_cls(fr, args, cos1, above.idx, s_above);
    SommBelow s_below;
    {
        // `_somm_remainder_below_prepare(cls_geom, medium, r1_below)`
        Front cls;  // the class geometry: only its segment arrays are read
        cls.geometry.n_segs = static_cast<ll>(below.idx.size());
        for (ll s : below.idx) {
            const std::size_t z = static_cast<std::size_t>(s);
            cls.geometry.seg_centers.push_back(g.seg_centers[z]);
            cls.geometry.seg_tangents.push_back(g.seg_tangents[z]);
            cls.geometry.seg_h.push_back(g.seg_h[z]);
        }
        cls.medium = md;
        cls.omega = fr.omega;
        somm_below_prepare(cls, cos1, r1_below, s_below);
    }
    // a mixed deck's two transmitted pair classes, compact
    // (`_transmitted_block`), before the class bands; `cross[side]` is the
    // block whose OBSERVERS are that side's class, (3, n_obs, n_src)
    std::vector<std::vector<cd>> cross[2];
    if (mixed) {
        cross[0] = transmitted_block(fr, cross_plan, q_x, kMu, false, above.idx, below.idx);
        cross[1] = transmitted_block(fr, cross_plan, q_x, kMu, true, below.idx, above.idx);
    }
    // `P[:, cols] += t` onto a row band's zeros: rows `r0..r1` (global) of
    // `side`'s class, its class row `r - first` taken from the class order
    std::vector<ll> class_row(Nz, -1);
    for (std::size_t i = 0; i < above.idx.size(); i++) class_row[static_cast<std::size_t>(above.idx[i])] = static_cast<ll>(i);
    for (std::size_t i = 0; i < below.idx.size(); i++) class_row[static_cast<std::size_t>(below.idx[i])] = static_cast<ll>(i);
    auto add_cross = [&](int side, ll r, cd *const phi_row[3]) {
        if (!mixed) return;
        const ClassSources &other = side ? above : below;
        const std::size_t nc = other.idx.size(), a = static_cast<std::size_t>(class_row[static_cast<std::size_t>(r)]);
        for (int c = 0; c < 3; c++)
            for (std::size_t j = 0; j < nc; j++) {
                cd &x = phi_row[c][static_cast<std::size_t>(other.idx[j])];
                const cd t = cross[side][static_cast<std::size_t>(c)][a * nc + j];
                x = cd(x.real() + t.real(), x.imag() + t.imag());
            }
    };
    // One class band's D: `P - (coef * Pi - Si)` over the class's sources,
    // for global observer rows [s, e)
    auto class_band = [&](bool is_below, ll s, ll e) {
        const ClassSources &cs = is_below ? below : above;
        const cd k = is_below ? md.k_m : cd(md.k_p, 0.0);
        const cd eta = is_below ? eta_m : eta_p;
        std::vector<std::vector<cd>> P = class_tensor(fr, K, s, e, cs.c, cs.t, cs.h, is_below, k, eta, cos1);
        std::vector<std::vector<cd>> Pi = class_tensor(fr, K, s, e, cs.img_c, cs.img_t, cs.h, is_below, k, eta, cos1);
        const ll nc = static_cast<ll>(cs.idx.size());
        const std::vector<std::vector<cd>> S =
            is_below ? somm_below_replay(fr, s_below, s, e) : somm_replay_cls(fr, s_above, nc, s, e);
        const cd coef = is_below ? md.a_m : s_above.c2;
        for (int c = 0; c < 3; c++)
            for (std::size_t q = 0; q < Pi[static_cast<std::size_t>(c)].size(); q++) {
                cd &y = Pi[static_cast<std::size_t>(c)][q];
                y = nec5::replay::np_cmul(coef, y);
                const cd si = S[static_cast<std::size_t>(c)][q];
                y = cd(y.real() - si.real(), y.imag() - si.imag());
                const cd x = P[static_cast<std::size_t>(c)][q];
                y = cd(x.real() - y.real(), x.imag() - y.imag());  // np.subtract(P, Pi, out=Pi)
            }
        return Pi;
    };
    // The class's rows run by run (`_schedule.runs(idx)`): each row its own
    // reduction, so any band height gives the same floats
    auto runs_of = [](const std::vector<ll> &idx) {
        std::vector<std::pair<ll, ll>> r;
        for (std::size_t i = 0; i < idx.size();) {
            std::size_t j = i + 1;
            while (j < idx.size() && idx[j] == idx[j - 1] + 1) j++;
            r.push_back({idx[i], idx[j - 1] + 1});
            i = j;
        }
        return r;
    };
    std::vector<cd> G(Nz * Nz, cd(0.0, 0.0));
    const bool dense = N < kDenseAssemblyThreshold;
    if (dense) {
        std::vector<std::vector<cd>> Phi(3, std::vector<cd>(Nz * Nz, cd(0.0, 0.0)));
        for (int side = 0; side < 2; side++) {
            const ClassSources &cs = side ? below : above;
            const std::size_t nc = cs.idx.size();
            for (const auto &run : runs_of(cs.idx)) {
                const std::vector<std::vector<cd>> D = class_band(side == 1, run.first, run.second);
                for (int c = 0; c < 3; c++)
                    for (ll r = run.first; r < run.second; r++)
                        for (std::size_t j = 0; j < nc; j++)
                            Phi[static_cast<std::size_t>(c)][static_cast<std::size_t>(r) * Nz + static_cast<std::size_t>(cs.idx[j])] =
                                D[static_cast<std::size_t>(c)][static_cast<std::size_t>(r - run.first) * nc + j];
                for (ll r = run.first; r < run.second; r++) {
                    cd *const rows[3] = {Phi[0].data() + r * N, Phi[1].data() + r * N, Phi[2].data() + r * N};
                    add_cross(side, r, rows);
                }
            }
        }
        std::vector<std::vector<cd>> M(3, std::vector<cd>(Nz * Nz, cd(0.0, 0.0)));
        for (int c = 0; c < 3; c++)
            for (std::size_t e = 0; e < E.n.size(); e++)
                M[static_cast<std::size_t>(c)][static_cast<std::size_t>(E.n[e] * N + E.j[e])] = E.v[c][e];
        std::vector<cd> tmp(Nz * Nz);
        for (int c = 0; c < 3; c++) {
            const nec5::Op a{Phi[static_cast<std::size_t>(c)].data(), N, N, N, 1};
            const nec5::Op b{M[static_cast<std::size_t>(c)].data(), N, N, N, 1};
            nec5::replay::np_matmul(np, a, b, c == 0 ? G.data() : tmp.data(), N, 1);
            if (c > 0)
                for (std::size_t i = 0; i < Nz * Nz; i++) G[i] = nec5::replay::c_add(G[i], tmp[i]);
        }
    } else {
        // CSC by column, rows ascending
        std::vector<ll> cptr(Nz + 1, 0);
        for (ll j : E.j) cptr[static_cast<std::size_t>(j) + 1]++;
        for (std::size_t j = 0; j < Nz; j++) cptr[j + 1] += cptr[j];
        std::vector<ll> crow(E.n.size()), fillp(cptr.begin(), cptr.end() - 1);
        std::vector<cd> cval[3];
        for (int c = 0; c < 3; c++) cval[c].resize(E.n.size());
        for (std::size_t e = 0; e < E.n.size(); e++) {
            const std::size_t at = static_cast<std::size_t>(fillp[static_cast<std::size_t>(E.j[e])]++);
            crow[at] = E.n[e];
            for (int c = 0; c < 3; c++) cval[c][at] = E.v[c][e];
        }
        // Z bands of each class's runs (`_mixed_band_rows`): the band's Φ
        // rows, its class's columns written, then the tiled reduction
        const ll zband = std::max<ll>(1, (kSweptMemMb * 1024 * 1024) / ((3 + 2) * 16 * N));
        for (int side = 0; side < 2; side++) {
            const ClassSources &cs = side ? below : above;
            const std::size_t nc = cs.idx.size();
            for (const auto &run : runs_of(cs.idx))
                for (ll z0 = run.first; z0 < run.second; z0 += zband) {
                    const ll z1 = std::min(z0 + zband, run.second), m = z1 - z0;
                    const std::vector<std::vector<cd>> D = class_band(side == 1, z0, z1);
                    std::vector<std::vector<cd>> Phi(3, std::vector<cd>(static_cast<std::size_t>(m) * Nz, cd(0.0, 0.0)));
                    for (int c = 0; c < 3; c++)
                        for (ll r = 0; r < m; r++)
                            for (std::size_t j = 0; j < nc; j++)
                                Phi[static_cast<std::size_t>(c)][static_cast<std::size_t>(r) * Nz + static_cast<std::size_t>(cs.idx[j])] =
                                    D[static_cast<std::size_t>(c)][static_cast<std::size_t>(r) * nc + j];
                    for (ll r = 0; r < m; r++) {
                        cd *const rows[3] = {Phi[0].data() + r * N, Phi[1].data() + r * N, Phi[2].data() + r * N};
                        add_cross(side, z0 + r, rows);
                    }
                    for (ll r = 0; r < m; r++) {
                        cd *grow = G.data() + (z0 + r) * N;
                        const cd *phi[3] = {Phi[0].data() + r * N, Phi[1].data() + r * N, Phi[2].data() + r * N};
                        for (ll j = 0; j < N; j++) {
                            double sr[3], si[3];
                            for (int c = 0; c < 3; c++) {
                                double yr = 0.0, yi = 0.0;
                                for (ll e = cptr[static_cast<std::size_t>(j)]; e < cptr[static_cast<std::size_t>(j) + 1]; e++)
                                    axpy1(cval[c][static_cast<std::size_t>(e)], phi[c][crow[static_cast<std::size_t>(e)]], yr, yi);
                                sr[c] = yr;
                                si[c] = yi;
                            }
                            grow[j] = cd((sr[0] + sr[1]) + sr[2], (si[0] + si[1]) + si[2]);
                        }
                    }
                }
        }
    }
    // the cross quadrants of a crossing deck: the point rows, added to Z
    if (!mixed) crossing_point_rows(fr, G);
    apply_loading(fr, loading, E, G);
    return G;
}

}  // namespace

bool wire_impedance_per_wire(const Front &fr, const nec4::SolverArgs &args, double omega, std::vector<cd> &z) {
    if (args.has_insulation) decline("an insulated wire (momwire#1607 B5b)");
    const razor::WireLoading loading = wire_loading(args);
    if (!loading.active()) return false;
    z = wire_impedance(fr, loading, omega);
    return true;
}

std::vector<cd> assemble_Z(const Front &fr, const nec4::SolverArgs &args, const nec5::NumpyLinalg &np) {
    if (fr.crossing || fr.mixed) return assemble_Z_crossing(fr, args, np);
    // GN 1: the perfect ground's image fold (`fg.mode == "fold"`, unweighted);
    // GN 2 / GN 3: the Sommerfeld composition (B5c)
    const bool pec = args.ground.kind == nec4::Ground::pec;
    const bool finite = args.ground.kind == nec4::Ground::finite;  // GN 2 / GN 3: Sommerfeld
    if (args.ground.kind != nec4::Ground::none && !pec && !finite)
        decline("a reflection-coefficient ground (not the NEC-4.2 slot's)");

    if (args.has_insulation) decline("an insulated wire (momwire#1607 B5b)");
    const razor::WireLoading loading = wire_loading(args);
    // The NEC-4.2 slot builds every solver with the extended kernel; the
    // reduced-kernel route (`sinusoidal_field_tensor`) is wired below but no
    // deck reaches it, so it is declined rather than served ungated.
    if (!args.extended_kernel && !fr.nec4 && !fr.buried)
        decline("the reduced-kernel fill (the NEC-4.2 slot never builds one)");
    // `_operating_medium`'s `refuse_out_of_scope`: the extended kernel has no
    // medium below the interface (the reference refuses; not worded here)
    if (fr.buried && args.extended_kernel) decline("the extended kernel below the interface (the reference refuses)");
    const Geometry &g = fr.geometry;
    const SegView &sv = fr.basis;
    const ll N = g.n_segs;
    if (N < 1) decline("a structure with no segments");
    const std::size_t Nz = static_cast<std::size_t>(N);

    // kd_min = min(|k| h), and the shape it picks (a wholly-buried deck's k
    // is k_m, its `np.abs` numpy's complex absolute)
    const double abs_k = fr.buried ? np_cabs(fr.medium.k_m) : std::fabs(fr.k);
    double kd_min = abs_k * g.seg_h[0];
    for (std::size_t i = 1; i < Nz; i++) {
        const double v = abs_k * g.seg_h[i];
        if (v < kd_min || std::isnan(v)) kd_min = v;
    }
    if (std::isnan(kd_min)) decline("a non-finite k*h");
    // momwire#1644: the NEC-4 kernel serves the literal shapes only
    const bool cos1 = !fr.nec4 && kd_min < kWellScaledKd;  // cos_shape == "cos-1"

    // M_A, M_B, M_C's entries: n_idx = repeat(arange(N), diff(starts)). A
    // (segment, basis) pair can repeat (a wire's two ends in one junction,
    // as a dropped duplicate leaves it, momwire#1042): the dense M keeps the
    // LAST value numpy's fancy assignment writes, scipy's CSC sums them.
    Entries E;
    std::map<std::pair<ll, ll>, int> seen;
    for (ll i = 0; i < N; i++)
        for (ll e = sv.starts[static_cast<std::size_t>(i)]; e < sv.starts[static_cast<std::size_t>(i) + 1]; e++) {
            const std::size_t ez = static_cast<std::size_t>(e);
            if (++seen[{i, sv.jbasis[ez]}] > 2) decline("a (segment, basis) entry repeated three times");
            E.n.push_back(i);
            E.j.push_back(sv.jbasis[ez]);
            E.v[0].push_back(sig_mul(sv.sigma[ez], cos1 ? sv.AC[ez] : sv.A[ez]));
            E.v[1].push_back(sv.B[ez]);
            E.v[2].push_back(sig_mul(sv.sigma[ez], sv.C[ez]));
        }

    Kernel K;
    K.fr = &fr;
    K.nec4 = fr.nec4;
    K.ek = args.extended_kernel && !K.nec4;
    if (!razor::leggauss(kNQpConst, K.gx, K.gw)) throw std::logic_error("leggauss(8) is tabled");
    const std::vector<double> rad = seg_radius(fr);
    if (K.nec4) K.src_a = rad;
    if (K.ek) {
        K.src_a = rad;
        ek_gating(fr, rad, K.ind1, K.ind2);
    }
    if (!fr.uniform_radius) {
        ll s = 0;
        for (ll i = 1; i <= N; i++)
            if (i == N || rad[static_cast<std::size_t>(i)] - rad[static_cast<std::size_t>(i) - 1] != 0.0) {
                K.runs.push_back({{s, i}, rad[static_cast<std::size_t>(s)]});
                s = i;
            }
    }

    // `_image_source_centers_tangents`: the sources mirrored in z = 0
    // (`2 * ground_z - z`, `tangents * [1, 1, -1]`)
    std::vector<Point3> img_c, img_t;
    Somm somm;
    if (finite && !fr.buried) somm_prepare(fr, args, cos1, somm);
    const Somm *sp = finite && !fr.buried ? &somm : nullptr;
    // a wholly-buried deck: eta_m = np.sqrt(mu / eps_m) (CPython's quotient,
    // numpy's csqrt), the serve plan's r1_below, the below remainder
    cd eta_b(0.0, 0.0);
    SommBelow below;
    if (fr.buried) {
        eta_b = np_csqrt(py_quot(cd(kMu, 0.0), fr.medium.eps_m));
        const double r1_below = serve_plan_below(fr);
        somm_below_prepare(fr, cos1, r1_below, below);
    }
    auto bandf = [&](ll i0, ll i1) {
        return fr.buried ? band_buried(fr, K, i0, i1, cos1, eta_b, img_c, img_t, below, rad)
                         : band(K, i0, i1, cos1, pec, img_c, img_t, sp);
    };
    if (pec || finite)
        for (std::size_t i = 0; i < Nz; i++) {
            const Point3 &c = g.seg_centers[i], &t = g.seg_tangents[i];
            img_c.push_back({c[0], c[1], 2.0 * 0.0 - c[2]});
            img_t.push_back({t[0] * 1.0, t[1] * 1.0, t[2] * -1.0});
        }
    std::vector<cd> G(Nz * Nz, cd(0.0, 0.0));
    const bool dense = N < kDenseAssemblyThreshold;
    if (dense) {
        // M[n_idx, j_idx] = val over zeros, then `out = Φ0@M0; out += Φ1@M1; out += Φ2@M2`
        std::vector<std::vector<cd>> M(3, std::vector<cd>(Nz * Nz, cd(0.0, 0.0)));
        for (int c = 0; c < 3; c++)
            for (std::size_t e = 0; e < E.n.size(); e++)
                M[static_cast<std::size_t>(c)][static_cast<std::size_t>(E.n[e] * N + E.j[e])] = E.v[c][e];
        const std::vector<std::vector<cd>> Phi = bandf(0, N);
        std::vector<cd> tmp(Nz * Nz);
        for (int c = 0; c < 3; c++) {
            const nec5::Op a{Phi[static_cast<std::size_t>(c)].data(), N, N, N, 1};
            const nec5::Op b{M[static_cast<std::size_t>(c)].data(), N, N, N, 1};
            nec5::replay::np_matmul(np, a, b, c == 0 ? G.data() : tmp.data(), N, 1);
            if (c > 0)
                for (std::size_t i = 0; i < Nz * Nz; i++) G[i] = nec5::replay::c_add(G[i], tmp[i]);
        }
        if (fr.nec4) nec4_cap_columns(fr, pec, sp, G);
        if (finite && !fr.nec4 && !fr.buried) contact_correction(fr, args, K, np, G);
        apply_loading(fr, loading, E, G);
        return G;
    }

    // CSC by column, rows ascending (scipy's canonical order; the COO entries
    // arrive n-major, so a stable bucket by column is already sorted)
    std::vector<ll> cptr(Nz + 1, 0);
    for (ll j : E.j) cptr[static_cast<std::size_t>(j) + 1]++;
    for (std::size_t j = 0; j < Nz; j++) cptr[j + 1] += cptr[j];
    std::vector<ll> crow(E.n.size()), fillp(cptr.begin(), cptr.end() - 1);
    std::vector<cd> cval[3];
    for (int c = 0; c < 3; c++) cval[c].resize(E.n.size());
    for (std::size_t e = 0; e < E.n.size(); e++) {
        const std::size_t at = static_cast<std::size_t>(fillp[static_cast<std::size_t>(E.j[e])]++);
        crow[at] = E.n[e];
        for (int c = 0; c < 3; c++) cval[c][at] = E.v[c][e];
    }
    if (E.n.size() != seen.size()) {
        // `sum_duplicates`: a repeated row in a column is one entry, the sum
        // of the two (in either order: one addition)
        std::vector<ll> ptr2(Nz + 1, 0), row2;
        std::vector<cd> val2[3];
        for (std::size_t j = 0; j < Nz; j++) {
            for (ll e = cptr[j]; e < cptr[j + 1]; e++) {
                const std::size_t ez = static_cast<std::size_t>(e);
                if (e > cptr[j] && crow[ez] == row2.back()) {
                    for (int c = 0; c < 3; c++) val2[c].back() = nec5::replay::c_add(val2[c].back(), cval[c][ez]);
                    continue;
                }
                row2.push_back(crow[ez]);
                for (int c = 0; c < 3; c++) val2[c].push_back(cval[c][ez]);
            }
            ptr2[j + 1] = static_cast<ll>(row2.size());
        }
        cptr.swap(ptr2);
        crow.swap(row2);
        for (int c = 0; c < 3; c++) cval[c].swap(val2[c]);
    }

    // `_schedule.mb_rows(swept_mem_mb, _fill_row_bytes(N))` in free space
    const ll blocks = 1 + ((pec || finite) ? 1 : 0) + (finite ? 1 : 0);  // 1 + (ground_z is not None) + sommerfeld
    const ll row_bytes = std::max<ll>(3 * blocks, 3 + 2) * N * 16;
    const ll chunk = std::max<ll>(1, (kSweptMemMb * 1024 * 1024) / row_bytes);
    for (ll i0 = 0; i0 < N; i0 += chunk) {
        const ll i1 = std::min(i0 + chunk, N);
        const std::vector<std::vector<cd>> Phi = bandf(i0, i1);
        // `_reduce_phi_band` over this band: per tile, P_c = Mcᵀ @ Φc[a:b]ᵀ
        // (csr_matvecs into zeros), then (P0 + P1) + P2, written back transposed
        for (const auto &tile : reduce_tiles(i1 - i0)) {
            for (ll r = tile.first; r < tile.second; r++) {
                cd *grow = G.data() + (i0 + r) * N;
                const cd *phi[3] = {Phi[0].data() + r * N, Phi[1].data() + r * N, Phi[2].data() + r * N};
                for (ll j = 0; j < N; j++) {
                    double sr[3], si[3];
                    for (int c = 0; c < 3; c++) {
                        double yr = 0.0, yi = 0.0;
                        for (ll e = cptr[static_cast<std::size_t>(j)]; e < cptr[static_cast<std::size_t>(j) + 1]; e++)
                            axpy1(cval[c][static_cast<std::size_t>(e)], phi[c][crow[static_cast<std::size_t>(e)]], yr, yi);
                        sr[c] = yr;
                        si[c] = yi;
                    }
                    grow[j] = cd((sr[0] + sr[1]) + sr[2], (si[0] + si[1]) + si[2]);
                }
            }
        }
    }
    if (fr.nec4) nec4_cap_columns(fr, pec, sp, G);
    if (finite && !fr.nec4 && !fr.buried) contact_correction(fr, args, K, np, G);
    apply_loading(fr, loading, E, G);
    return G;
}

PortSolution port_solution(const Front &fr, const nec4::SolverArgs &args, const FillLinalg &la) {
    if (la.lu.getrf == nullptr || la.lu.getrs == nullptr) decline("no LU routines");
    PortSolution ps;
    ps.G = assemble_Z(fr, args, la.np);
    const Geometry &g = fr.geometry;
    const SegView &sv = fr.basis;
    const ll N = g.n_segs, P = static_cast<ll>(g.feed_segs.size());
    const std::size_t Nz = static_cast<std::size_t>(N), Pz = static_cast<std::size_t>(P);
    ps.n = N;
    ps.n_ports = P;
    // B[fi, j] = -1 / seg_h[fi]
    std::vector<cd> x(Nz * Pz, cd(0.0, 0.0));  // Fortran order (N, P)
    for (std::size_t j = 0; j < Pz; j++) {
        const std::size_t fi = static_cast<std::size_t>(g.feed_segs[j]);
        x[fi + j * Nz] = cd(-1.0 / g.seg_h[fi], 0.0);
    }
    if (P > 0) {
        // `scipy.linalg.solve(G, B)`: finite inputs, then the structure it detects
        for (const cd &v : ps.G)
            if (!std::isfinite(v.real()) || !std::isfinite(v.imag())) decline("a non-finite Z");
        if (N == 1) {
            if (ps.G[0] == cd(0.0, 0.0)) decline("a singular 1x1 Z");
            for (std::size_t j = 0; j < Pz; j++) x[j] = nec5::replay::np_cdiv(x[j], ps.G[0]);
        } else {
            // scipy takes a diagonal, triangular or tridiagonal matrix (lower
            // or upper bandwidth at most 1), a symmetric and a Hermitian one
            // to other drivers (measured: a wider band, a block-diagonal one
            // and scattered zeros all take getrf / getrs)
            bool sym = true, her = true;
            std::size_t lower = 0, upper = 0;
            for (std::size_t i = 0; i < Nz; i++)
                for (std::size_t j = 0; j < Nz; j++) {
                    const cd a = ps.G[i * Nz + j], b = ps.G[j * Nz + i];
                    if (a != cd(0.0, 0.0)) {
                        if (i > j) lower = std::max(lower, i - j);
                        if (j > i) upper = std::max(upper, j - i);
                    }
                    if (j > i) {
                        if (a != b) sym = false;
                        if (a != std::conj(b)) her = false;
                    } else if (j == i && a.imag() != 0.0) {
                        her = false;
                    }
                }
            if (lower <= 1 || upper <= 1) decline("a banded or triangular Z (scipy's structure detection)");
            if (sym || her) decline("a symmetric or Hermitian Z (scipy's structure detection)");
            std::vector<cd> lu(Nz * Nz);
            for (std::size_t i = 0; i < Nz; i++)
                for (std::size_t j = 0; j < Nz; j++) lu[i + j * Nz] = ps.G[i * Nz + j];
            std::vector<int> ipiv(Nz);
            int n = static_cast<int>(N), nrhs = static_cast<int>(P), info = 0;
            la.lu.getrf(&n, &n, lu.data(), &n, ipiv.data(), &info);
            if (info != 0) decline("a singular Z (scipy's LinAlgError)");
            char trans = 'N';
            la.lu.getrs(&trans, &n, &nrhs, lu.data(), &n, ipiv.data(), x.data(), &n, &info);
            if (info != 0) decline("getrs failed");
        }
    }
    ps.coeffs.assign(Nz * Pz, cd(0.0, 0.0));
    for (std::size_t i = 0; i < Nz; i++)
        for (std::size_t j = 0; j < Pz; j++) ps.coeffs[i * Pz + j] = x[i + j * Nz];
    // Y[i, j] = _feed_segment_current(alphas[:, j], seg_view, fi) at xi = 0:
    // (amp * sig * AC).sum()
    ps.y.assign(Pz * Pz, cd(0.0, 0.0));
    for (std::size_t j = 0; j < Pz; j++)
        for (std::size_t i = 0; i < Pz; i++) {
            const std::size_t fi = static_cast<std::size_t>(g.feed_segs[i]);
            std::vector<cd> terms;
            for (ll e = sv.starts[fi]; e < sv.starts[fi + 1]; e++) {
                const std::size_t ez = static_cast<std::size_t>(e);
                const cd amp = x[static_cast<std::size_t>(sv.jbasis[ez]) + j * Nz];
                const cd as = nec5::replay::np_cmul(amp, cd(static_cast<double>(sv.sigma[ez]), 0.0));
                terms.push_back(nec5::replay::np_cmul(as, sv.AC[ez]));
            }
            ps.y[i * Pz + j] = nec5::replay::np_csum(terms);
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

// A row's digest (`sinusoidal_fill_compare._digest`): its float64 words w,
// x = splitmix-style mix of w, h = sum of x_i P^(i+1) mod 2^64.
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

void matrix(Out &o, const std::string &p, const std::vector<cd> &m, std::size_t rows, std::size_t cols,
            std::size_t whole) {
    o.line(p + " : shape " + std::to_string(rows) + " " + std::to_string(cols));
    for (std::size_t i = 0; i < rows; i++) {
        if (cols <= whole) {
            for (std::size_t j = 0; j < cols; j++) o.c(at(at(p, i), j), m[i * cols + j]);
        } else {
            o.line(at(p, i) + " = digest " + digest(m.data() + i * cols, cols));
        }
    }
}

}  // namespace

std::string dump_fill(const std::string &text, const FillLinalg &la, std::size_t whole) {
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
                const Front fr = front(sb.points[j].args);
                const PortSolution ps = port_solution(fr, fr.args, la);
                const std::size_t n = static_cast<std::size_t>(ps.n), np_ = static_cast<std::size_t>(ps.n_ports);
                matrix(o, q + ".G", ps.G, n, n, whole);
                matrix(o, q + ".coeffs", ps.coeffs, n, np_, whole);
                matrix(o, q + ".y", ps.y, np_, np_, whole);
            } catch (const FrontError &e) {
                o.line(q + " ! " + e.kind + " " + nec5::dump_hex(e.message));
            } catch (const Declined &e) {
                o.line(q + " ? declined " + nec5::dump_hex(e.what()));
            }
        }
    }
    return o.s;
}

}  // namespace sinusoidal
}  // namespace mw
