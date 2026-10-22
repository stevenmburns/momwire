// The sinusoidal free-space fill and solve in C++ (see
// `_sinusoidal_fill_core.h`). Each function names the Python it copies;
// numpy's and scipy's arithmetic is written out per element in their order,
// with no contraction.

#if defined(_MSC_VER) && !defined(__clang__)
#pragma float_control(precise, on)
#pragma fp_contract(off)
#endif

#include "_sinusoidal_fill_core.h"

#include "_razor_front_core.h"
#include "_sinusoidal_core.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
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
    std::vector<double> gx, gw, src_a;
    std::vector<int8_t> ind1, ind2;
    // radius runs: (start, stop, a)
    std::vector<std::pair<std::pair<ll, ll>, double>> runs;
};

Tensors call(const Kernel &K, ll r0, ll r1, double a) {
    const Geometry &g = K.fr->geometry;
    const ll n = g.n_segs;
    const ll q = static_cast<ll>(K.gx.size());
    CF oc = pts(g.seg_centers, r0, r1), ot = pts(g.seg_tangents, r0, r1);
    CF sc = pts(g.seg_centers, 0, n), st = pts(g.seg_tangents, 0, n);
    CF h = f64(g.seg_h.data(), {n});
    CF gx = f64(K.gx.data(), {q}), gw = f64(K.gw.data(), {q});
    if (K.ek) {
        CF sa = f64(K.src_a.data(), {n});
        CI8 i1 = CI8::borrow(buffer(K.ind1.data(), 1, {n})), i2 = CI8::borrow(buffer(K.ind2.data(), 1, {n}));
        return sinusoidal_field_tensor_ek(oc, ot, sc, st, h, a, K.fr->k, K.fr->eta, gx, gw, sa, i1, i2, 0);
    }
    return sinusoidal_field_tensor(oc, ot, sc, st, h, a, K.fr->k, K.fr->eta, gx, gw, 0);
}

// `_field_tensor(geom, k, obs_rows=(i0, i1))` on the accelerated literal-cos
// route: (Φ_const, Φ_sin, Φ_cos), each (i1 - i0, N) in C order.
std::vector<std::vector<cd>> field_tensor(const Kernel &K, ll i0, ll i1) {
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
    if (K.fr->uniform_radius) {
        put(call(K, i0, i1, K.fr->radius_per_wire[0]), i0, i1);
        return out;
    }
    // `_radius_runs(geom, obs_rows)`, concatenated on the observer axis
    for (const auto &run : K.runs) {
        const ll s = run.first.first, e = run.first.second;
        if (!(s < i1 && e > i0)) continue;
        const ll r0 = std::max(s, i0), r1 = std::min(e, i1);
        put(call(K, r0, r1, run.second), r0, r1);
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
    const double h = std::sin(half), c = std::cos(half);
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
std::vector<std::vector<cd>> field_tensor_cos1(const Kernel &K, ll i0, ll i1) {
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
            const Point3 &sc = g.seg_centers[jz];
            const Point3 &st = g.seg_tangents[jz];
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

}  // namespace

std::vector<cd> assemble_Z(const Front &fr, const nec4::SolverArgs &args, const nec5::NumpyLinalg &np) {
    if (args.ground.kind != nec4::Ground::none) decline("a ground (momwire#1607 B5)");
    if (args.has_conductivity || args.has_insulation || args.has_distributed)
        decline("wire loading (momwire#1607 B5)");
    // The NEC-4.2 slot builds every solver with the extended kernel; the
    // reduced-kernel route (`sinusoidal_field_tensor`) is wired below but no
    // deck reaches it, so it is declined rather than served ungated.
    if (!args.extended_kernel) decline("the reduced-kernel fill (the NEC-4.2 slot never builds one)");
    const Geometry &g = fr.geometry;
    const SegView &sv = fr.basis;
    const ll N = g.n_segs;
    if (N < 1) decline("a structure with no segments");
    const std::size_t Nz = static_cast<std::size_t>(N);

    // kd_min = min(|k| h), and the shape it picks
    double kd_min = std::fabs(fr.k) * g.seg_h[0];
    for (std::size_t i = 1; i < Nz; i++) {
        const double v = std::fabs(fr.k) * g.seg_h[i];
        if (v < kd_min || std::isnan(v)) kd_min = v;
    }
    if (std::isnan(kd_min)) decline("a non-finite k*h");
    const bool cos1 = kd_min < kWellScaledKd;  // cos_shape == "cos-1"

    // M_A, M_B, M_C's entries: n_idx = repeat(arange(N), diff(starts))
    Entries E;
    std::set<std::pair<ll, ll>> seen;
    for (ll i = 0; i < N; i++)
        for (ll e = sv.starts[static_cast<std::size_t>(i)]; e < sv.starts[static_cast<std::size_t>(i) + 1]; e++) {
            const std::size_t ez = static_cast<std::size_t>(e);
            if (!seen.insert({i, sv.jbasis[ez]}).second) decline("a repeated (segment, basis) entry");
            E.n.push_back(i);
            E.j.push_back(sv.jbasis[ez]);
            E.v[0].push_back(sig_mul(sv.sigma[ez], cos1 ? sv.AC[ez] : sv.A[ez]));
            E.v[1].push_back(sv.B[ez]);
            E.v[2].push_back(sig_mul(sv.sigma[ez], sv.C[ez]));
        }

    Kernel K;
    K.fr = &fr;
    K.ek = args.extended_kernel;
    if (!razor::leggauss(kNQpConst, K.gx, K.gw)) throw std::logic_error("leggauss(8) is tabled");
    const std::vector<double> rad = seg_radius(fr);
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

    std::vector<cd> G(Nz * Nz, cd(0.0, 0.0));
    const bool dense = N < kDenseAssemblyThreshold;
    if (dense) {
        // M[n_idx, j_idx] = val over zeros, then `out = Φ0@M0; out += Φ1@M1; out += Φ2@M2`
        std::vector<std::vector<cd>> M(3, std::vector<cd>(Nz * Nz, cd(0.0, 0.0)));
        for (int c = 0; c < 3; c++)
            for (std::size_t e = 0; e < E.n.size(); e++)
                M[static_cast<std::size_t>(c)][static_cast<std::size_t>(E.n[e] * N + E.j[e])] = E.v[c][e];
        const std::vector<std::vector<cd>> Phi = cos1 ? field_tensor_cos1(K, 0, N) : field_tensor(K, 0, N);
        std::vector<cd> tmp(Nz * Nz);
        for (int c = 0; c < 3; c++) {
            const nec5::Op a{Phi[static_cast<std::size_t>(c)].data(), N, N, N, 1};
            const nec5::Op b{M[static_cast<std::size_t>(c)].data(), N, N, N, 1};
            nec5::replay::np_matmul(np, a, b, c == 0 ? G.data() : tmp.data(), N, 1);
            if (c > 0)
                for (std::size_t i = 0; i < Nz * Nz; i++) G[i] = nec5::replay::c_add(G[i], tmp[i]);
        }
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

    // `_schedule.mb_rows(swept_mem_mb, _fill_row_bytes(N))` in free space
    const ll row_bytes = std::max<ll>(3 * 1, 3 + 2) * N * 16;
    const ll chunk = std::max<ll>(1, (kSweptMemMb * 1024 * 1024) / row_bytes);
    for (ll i0 = 0; i0 < N; i0 += chunk) {
        const ll i1 = std::min(i0 + chunk, N);
        const std::vector<std::vector<cd>> Phi = cos1 ? field_tensor_cos1(K, i0, i1) : field_tensor(K, i0, i1);
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
                const PortSolution ps = port_solution(fr, sb.points[j].args, la);
                const std::size_t n = static_cast<std::size_t>(ps.n), np_ = static_cast<std::size_t>(ps.n_ports);
                matrix(o, q + ".G", ps.G, n, n, whole);
                matrix(o, q + ".coeffs", ps.coeffs, n, np_, whole);
                matrix(o, q + ".y", ps.y, np_, np_, whole);
            } catch (const FrontError &e) {
                o.line(q + " ! ValueError " + nec5::dump_hex(e.message));
            } catch (const Declined &e) {
                o.line(q + " ? declined " + nec5::dump_hex(e.what()));
            }
        }
    }
    return o.s;
}

}  // namespace sinusoidal
}  // namespace mw
