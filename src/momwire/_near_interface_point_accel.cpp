// momwire#1224 (sinusoidal buried, stage 3), unit 1: the C++ twin of
// `_near_interface.point_keys_columns` -- the four POINT-observer integrals
// (POINT_KEYS: dRhoV, dzV, dRhoW, dz'V) over concatenated rho columns, for the
// point-matched lane's crossing node (#1223 U3a/U4).
//
// Stage 2 left this family in numpy by decision (dense, correctness first),
// and on the buried radial vertical it is 88 % of the solve (#1224 profile):
// per column a J1 weight rebuild plus an (nz x K) complex exponential. This is
// `near_interface_six_columns`' shape exactly -- the SAME column rule (nodes
// and J0 weights, `mw899::column_rule`), one more weight vector (J1 on those
// nodes), and four factor rows instead of six against the same exponential
// e^{gamma_m z' - gamma_p z}.
//
// Its OWN extension on purpose (the #1193/#1194/#1209 lessons): the shared
// machinery lives in `_near_interface_columns_inline.h`, moved there
// verbatim, so adding this kernel cannot move a single bit of the six-kernel
// twin's code. The numpy route stays the reference; the gate against it is
// RELATIVE (1e-13), never bit (libmvec, Amos and the ordered lane sum differ
// in the last bits -- the house rule).

#define _USE_MATH_DEFINES

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstdint>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

#include "_contour_engine_inline.h"
#include "_branch_cut_inline.h"
#include "xsf/bessel.h"

namespace py = pybind11;

// The libmvec declarations and the MSVC `omp simd` neutralization, as in
// `_near_interface_accel.cpp` (see there for why each exists).
#if defined(__GNUC__) && !defined(_MSC_VER) && !defined(__APPLE__)
#pragma omp declare simd notinbranch simdlen(4)
extern "C" double exp(double);

#pragma omp declare simd notinbranch simdlen(4)
extern "C" double cos(double);

#pragma omp declare simd notinbranch simdlen(4)
extern "C" double sin(double);
#endif

#if defined(_MSC_VER)
#define MW_NI_SIMD(clauses)
#else
#define MW_NI_PRAGMA_(x) _Pragma(#x)
#define MW_NI_SIMD(clauses) MW_NI_PRAGMA_(omp simd clauses)
#endif

#if defined(__GNUC__)
// The shared header carries the six-kernel twin's single-point walk too,
// which this TU never calls.
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-function"
#endif
#include "_near_interface_columns_inline.h"
#if defined(__GNUC__)
#pragma GCC diagnostic pop
#endif

namespace mw1224 {
using mw_branch::gamma_cut;
using mw_contour::cd;
using mw_contour::MW_PI;

static const int NP = 4;  // POINT_KEYS

// `_column_rule_j1`'s second weight vector, on `mw899::column_rule`'s own
// nodes: the J0/H0 factor of each weight replaced by the J1 one --
// lam rho (J1/lam rho) in the head and mid, H1/2 and H2/2 on the rays. Like
// the numpy route it REBUILDS the path pieces rather than dividing the J0
// factor out (which would fail at a J0 zero), and it checks the rebuilt nodes
// against the rule's own: a J1 weight on a different node than its J0 twin
// would be a silent error. Returns false on a mismatch; the caller raises.
static bool column_rule_j1(double rho, double k_p, const cd &k_m, double s_min,
                           double lam_mult, int p, double detour,
                           const double *gx, const double *gw, int ng,
                           mw899::Scratch &s, std::vector<cd> &w1) {
    mw899::column_rule(rho, k_p, k_m, s_min, lam_mult, p, detour, gx, gw, ng,
                       s);
    const double kk = std::max(k_p, std::abs(k_m));
    double a_head = 1.1 * kk;
    double lam_top = lam_mult * kk;
    if (s_min > 0.0 && mw680::MW_FAR_PAIR_KILL / s_min < lam_top) {
        const double lam_kill = mw680::MW_FAR_PAIR_KILL / s_min;
        a_head = std::max(2.2 * k_p, std::min(a_head, lam_kill));
        lam_top = std::max(1.5 * a_head, lam_kill);
    }
    w1.clear();
    w1.reserve(s.lam.size());
    std::vector<double> edges, seeded, t, wt;
    size_t idx = 0;
    auto same = [&](const cd &l) { return idx < s.lam.size() && s.lam[idx++] == l; };

    double H = std::min(0.35 * a_head, detour / std::max(rho, 1e-12));
    H = std::max(H, 1e-6 * a_head);
    edges.push_back(0.0);
    edges.push_back(a_head);
    for (int i = 1; i < 7; ++i) edges.push_back(a_head * i / 7.0);
    const double marks[2] = {k_p, std::fabs(k_m.real())};
    const double ws[5] = {0.0, -0.15, 0.15, -0.4, 0.4};
    for (int m = 0; m < 2; ++m)
        for (int j = 0; j < 5; ++j) {
            const double v = marks[m] * (1.0 + ws[j]);
            if (v > 0.0 && v < a_head) edges.push_back(v);
        }
    std::sort(edges.begin(), edges.end());
    edges.erase(std::unique(edges.begin(), edges.end()), edges.end());
    mw899::sub_seed(edges, rho, seeded);
    mw899::fixed_gauss(seeded, p, gx, gw, ng, t, wt);
    for (size_t i = 0; i < t.size(); ++i) {
        const cd l(t[i], H * std::sin(MW_PI * t[i] / a_head));
        const cd dl(1.0, H * (MW_PI / a_head) * std::cos(MW_PI * t[i] / a_head));
        if (!same(l)) return false;
        cd b0, b1x;
        mw_contour::bessel_j0_j1x(l * rho, b0, b1x);
        w1.push_back(wt[i] * dl * (l * rho) * b1x);
    }

    edges.clear();
    edges.push_back(a_head);
    edges.push_back(lam_top);
    mw899::sub_seed(edges, rho, seeded);
    mw899::fixed_gauss(seeded, p, gx, gw, ng, t, wt);
    for (size_t i = 0; i < t.size(); ++i) {
        const cd l(t[i], 0.0);
        if (!same(l)) return false;
        cd b0, b1x;
        mw_contour::bessel_j0_j1x(l * rho, b0, b1x);
        w1.push_back(wt[i] * (l * rho) * b1x);
    }

    const double scale = std::sqrt(2.0) / (s_min + rho);
    double step = std::min(0.25 * scale, lam_top);
    edges.clear();
    edges.push_back(0.0);
    while (edges.back() < mw680::MW_FAR_PAIR_KILL * scale) {
        edges.push_back(edges.back() + step);
        step *= 2.0;
    }
    mw899::fixed_gauss(edges, p, gx, gw, ng, t, wt);
    const cd ray = std::exp(cd(0.0, 0.25 * MW_PI));
    if (rho == 0.0) {
        // One up-ray, J1(0) = 0: the rho keys are an exact zero on the axis.
        for (size_t i = 0; i < t.size(); ++i) {
            if (!same(lam_top + t[i] * ray)) return false;
            w1.push_back(cd(0.0, 0.0));
        }
    } else {
        for (size_t i = 0; i < t.size(); ++i) {
            const cd up = lam_top + t[i] * ray;
            if (!same(up)) return false;
            w1.push_back(wt[i] * ray * 0.5 * xsf::cyl_hankel_1(1.0, up * rho));
        }
        for (size_t i = 0; i < t.size(); ++i) {
            const cd dn = lam_top + t[i] * std::conj(ray);
            if (!same(dn)) return false;
            w1.push_back(wt[i] * std::conj(ray) * 0.5 *
                         xsf::cyl_hankel_2(1.0, dn * rho));
        }
    }
    return idx == s.lam.size();
}

// `point_keys_columns`' factor rows with the weights folded in, such that
// point(z, z') = F @ exp(gamma_m z' - gamma_p z). POINT_KEYS order:
//   0 dRhoV = -lam v w1,  1 dzV = -gamma_p v w0,
//   2 dRhoW = -lam wv w1, 3 dz'V = +gamma_m v w0,
// v = 2 lam / (k_m^2 gamma_p + k_p^2 gamma_m), wv = (gamma_p - gamma_m) v.
static void point_factors(const cd &k_p, const cd &k_m,
                          const std::vector<cd> &w1, mw899::Scratch &s) {
    const size_t K = s.lam.size();
    const cd kp2 = k_p * k_p, km2 = k_m * k_m;
    s.gpr.resize(K);
    s.gpi.resize(K);
    s.gmr.resize(K);
    s.gmi.resize(K);
    s.fr.resize(NP * K);
    s.fi.resize(NP * K);
    for (size_t k = 0; k < K; ++k) {
        const cd l = s.lam[k], w0 = s.w[k], w1k = w1[k];
        const cd g_p = gamma_cut(l, k_p);
        const cd g_m = gamma_cut(l, k_m);
        using mw_fma::mul;
        const cd v = 2.0 * l / (mul(km2, g_p) + mul(kp2, g_m));
        const cd wv = mul(g_p - g_m, v);
        const cd f[NP] = {-mul(mul(l, v), w1k), -mul(mul(g_p, v), w0),
                          -mul(mul(l, wv), w1k), mul(mul(g_m, v), w0)};
        for (int c = 0; c < NP; ++c) {
            s.fr[c * K + k] = f[c].real();
            s.fi[c * K + k] = f[c].imag();
        }
        s.gpr[k] = g_p.real();
        s.gpi[k] = g_p.imag();
        s.gmr[k] = g_m.real();
        s.gmi[k] = g_m.imag();
    }
}

// `mw899::column_member` for NP rows: the same underflow skip and the same
// source-fixed lane order (momwire#1193), four sums instead of six.
static void point_member(const mw899::Scratch &s, size_t K, double z,
                         double zp, cd *out) {
    using mw899::MW_BLOCK;
    using mw899::MW_EXP_ZERO;
    using mw899::MW_LANES;
    using mw899::mw_lanes;
    double accr[NP] = {0, 0, 0, 0}, acci[NP] = {0, 0, 0, 0};
    double ar[MW_BLOCK], ai[MW_BLOCK], er[MW_BLOCK], ei[MW_BLOCK];
    const double *gpr = s.gpr.data(), *gpi = s.gpi.data();
    const double *gmr = s.gmr.data(), *gmi = s.gmi.data();
    for (size_t k0 = 0; k0 < K; k0 += MW_BLOCK) {
        const int nb = static_cast<int>(std::min<size_t>(MW_BLOCK, K - k0));
        double amax = -std::numeric_limits<double>::infinity();
        for (int j = 0; j < nb; ++j) {
            ar[j] = mw_fma::fma(gmr[k0 + j], zp, -(gpr[k0 + j] * z));
            amax = std::max(amax, ar[j]);
        }
        if (amax < MW_EXP_ZERO) continue;
        MW_NI_SIMD()
        for (int j = 0; j < nb; ++j) {
            ai[j] = mw_fma::fma(gmi[k0 + j], zp, -(gpi[k0 + j] * z));
            const double ex = exp(ar[j]);
            er[j] = ex * cos(ai[j]);
            ei[j] = ex * sin(ai[j]);
        }
        static_assert(MW_LANES == 4, "the lane combine is written for four");
        for (int c = 0; c < NP; ++c) {
            const double *fr = s.fr.data() + c * K + k0;
            const double *fi = s.fi.data() + c * K + k0;
            mw_lanes sr = {0.0, 0.0, 0.0, 0.0}, si = {0.0, 0.0, 0.0, 0.0};
            int j = 0;
            for (; j + MW_LANES <= nb; j += MW_LANES) {
#if defined(__GNUC__)
                mw_lanes a, b, x, y;
                std::memcpy(&a, er + j, sizeof a);
                std::memcpy(&b, ei + j, sizeof b);
                std::memcpy(&x, fr + j, sizeof x);
                std::memcpy(&y, fi + j, sizeof y);
                const mw_lanes by = b * y, bx = b * x;
                mw_lanes pp, q;
                for (int l = 0; l < MW_LANES; ++l) {
                    pp[l] = mw_fma::fma(a[l], x[l], -by[l]);
                    q[l] = mw_fma::fma(a[l], y[l], bx[l]);
                }
                sr += pp;
                si += q;
#else
                for (int l = 0; l < MW_LANES; ++l) {
                    sr[l] += mw_fma::fma(er[j + l], fr[j + l],
                                         -(ei[j + l] * fi[j + l]));
                    si[l] += mw_fma::fma(er[j + l], fi[j + l],
                                         ei[j + l] * fr[j + l]);
                }
#endif
            }
            for (int l = 0; j < nb; ++j, ++l) {
                sr[l] += mw_fma::fma(er[j], fr[j], -(ei[j] * fi[j]));
                si[l] += mw_fma::fma(er[j], fi[j], ei[j] * fr[j]);
            }
            accr[c] += (sr[0] + sr[1]) + (sr[2] + sr[3]);
            acci[c] += (si[0] + si[1]) + (si[2] + si[3]);
        }
    }
    for (int c = 0; c < NP; ++c) out[c] = cd(accr[c], acci[c]);
}
}  // namespace mw1224

// `point_keys_columns` over CONCATENATED columns, `near_interface_six_columns`'
// contract with four columns out instead of six: column c owns members
// offsets[c] .. offsets[c+1] of (z, zp), all at rho[c]; the answer keeps the
// members' order. Parallel over columns and, for a column bigger than one
// thread's share, over its members -- the six twin's split, for its reason.
static py::array_t<std::complex<double>> near_interface_point_columns(
    double k_p, std::complex<double> k_m,
    py::array_t<double, py::array::c_style | py::array::forcecast> rho,
    py::array_t<py::ssize_t, py::array::c_style | py::array::forcecast> offsets,
    py::array_t<double, py::array::c_style | py::array::forcecast> z,
    py::array_t<double, py::array::c_style | py::array::forcecast> zp,
    double lam_mult, int p, double detour, int n_threads,
    py::array_t<double, py::array::c_style | py::array::forcecast> gx,
    py::array_t<double, py::array::c_style | py::array::forcecast> gw) {
    if (rho.ndim() != 1 || offsets.ndim() != 1 || z.ndim() != 1 ||
        zp.ndim() != 1)
        throw std::invalid_argument("rho, offsets, z and zp must be 1-D");
    auto rb = rho.unchecked<1>();
    auto ob = offsets.unchecked<1>();
    auto zb = z.unchecked<1>();
    auto pb = zp.unchecked<1>();
    const py::ssize_t nc = rb.shape(0);
    const py::ssize_t n = zb.shape(0);
    if (ob.shape(0) != nc + 1)
        throw std::invalid_argument("offsets must have len(rho) + 1 entries");
    if (pb.shape(0) != n)
        throw std::invalid_argument("z and zp must have the same length");
    if (ob(0) != 0 || ob(nc) != n)
        throw std::invalid_argument("offsets must span 0 .. len(z)");
    for (py::ssize_t c = 0; c < nc; ++c)
        if (ob(c + 1) < ob(c))
            throw std::invalid_argument("offsets must be non-decreasing");
    if (gx.ndim() != 1 || gw.ndim() != 1 || gx.shape(0) != gw.shape(0) ||
        gx.shape(0) < 1)
        throw std::invalid_argument("bad Gauss rule");
    if (p < 0 || p > 20) throw std::invalid_argument("column p out of range");
    const double *gxp = gx.data();
    const double *gwp = gw.data();
    const int ng = static_cast<int>(gx.shape(0));
    for (py::ssize_t c = 0; c < nc; ++c) {
        for (py::ssize_t i = ob(c); i < ob(c + 1); ++i) {
            if (!(zb(i) >= 0.0 && pb(i) <= 0.0))
                throw std::invalid_argument("need z >= 0 >= zp");
            if (rb(c) < 0.0 || (zb(i) - pb(i)) + rb(c) <= 0.0)
                throw std::invalid_argument("need R > 0");
        }
    }

    py::array_t<std::complex<double>> vals({n, py::ssize_t(mw1224::NP)});
    auto vb = vals.mutable_unchecked<2>();
    const mw_contour::cd kpc(k_p, 0.0);
    const mw_contour::cd km(k_m);
    int nt = 1;
#ifdef _OPENMP
    nt = omp_get_max_threads();
    if (n_threads > 0) nt = std::min(nt, n_threads);
    nt = std::max(nt, 1);
#endif
    const py::ssize_t share = (n + nt - 1) / nt;
    std::vector<py::ssize_t> by_column, by_member;
    for (py::ssize_t c = 0; c < nc; ++c) {
        if (ob(c + 1) == ob(c)) continue;
        (ob(c + 1) - ob(c) > share ? by_member : by_column).push_back(c);
    }
    const py::ssize_t n_col = static_cast<py::ssize_t>(by_column.size());
    const py::ssize_t n_mem = static_cast<py::ssize_t>(by_member.size());
    bool nodes_moved = false;
    {
        py::gil_scoped_release release;
        #pragma omp parallel for schedule(dynamic) num_threads(nt)
        for (py::ssize_t j = 0; j < n_col; ++j) {
            const py::ssize_t c = by_column[j];
            const py::ssize_t lo = ob(c), hi = ob(c + 1);
            mw899::Scratch s;
            std::vector<mw_contour::cd> w1;
            if (!mw1224::column_rule_j1(rb(c), k_p, km,
                                        mw899::s_min_of(zb, pb, lo, hi),
                                        lam_mult, p, detour, gxp, gwp, ng, s,
                                        w1)) {
                #pragma omp atomic write
                nodes_moved = true;
                continue;
            }
            mw1224::point_factors(kpc, km, w1, s);
            const size_t K = s.lam.size();
            for (py::ssize_t i = lo; i < hi; ++i) {
                mw_contour::cd out[mw1224::NP];
                mw1224::point_member(s, K, zb(i), pb(i), out);
                for (int q = 0; q < mw1224::NP; ++q) vb(i, q) = out[q];
            }
        }
        for (py::ssize_t j = 0; j < n_mem && !nodes_moved; ++j) {
            const py::ssize_t c = by_member[j];
            const py::ssize_t lo = ob(c), hi = ob(c + 1);
            mw899::Scratch s;
            std::vector<mw_contour::cd> w1;
            if (!mw1224::column_rule_j1(rb(c), k_p, km,
                                        mw899::s_min_of(zb, pb, lo, hi),
                                        lam_mult, p, detour, gxp, gwp, ng, s,
                                        w1)) {
                nodes_moved = true;
                break;
            }
            mw1224::point_factors(kpc, km, w1, s);
            const size_t K = s.lam.size();
            #pragma omp parallel for schedule(dynamic, 32) num_threads(nt)
            for (py::ssize_t i = lo; i < hi; ++i) {
                mw_contour::cd out[mw1224::NP];
                mw1224::point_member(s, K, zb(i), pb(i), out);
                for (int q = 0; q < mw1224::NP; ++q) vb(i, q) = out[q];
            }
        }
    }
    if (nodes_moved)
        throw std::runtime_error(
            "the J1 weights no longer follow _column_rule's nodes");
    return vals;
}

// Its OWN extension, not a second TU of `_near_interface_accel`: the vendored
// xsf headers define some functions non-inline, so two TUs that include them
// cannot link into one module, and keeping the six twin's module untouched is
// the point anyway. The name is a build parameter, as the sibling's is
// (momwire#1032's AVX2 / baseline pair).
#ifndef MOMWIRE_MODULE_NAME
#define MOMWIRE_MODULE_NAME _near_interface_point_accel
#endif

PYBIND11_MODULE(MOMWIRE_MODULE_NAME, m) {
    m.doc() =
        "momwire#1224: the C++ column twin of "
        "_near_interface.point_keys_columns. Optional; _near_interface falls "
        "back to the numpy route without it.";
    // Its own flag: the capability `_near_interface` keys the route on.
    m.attr("point_columns_1224") = true;
    m.def("near_interface_point_columns", &near_interface_point_columns,
          py::arg("k_p"), py::arg("k_m"), py::arg("rho"), py::arg("offsets"),
          py::arg("z"), py::arg("zp"), py::arg("lam_mult"), py::arg("p"),
          py::arg("detour"), py::arg("n_threads"), py::arg("gx"),
          py::arg("gw"),
          "point_keys_columns over CONCATENATED columns -> (n, 4) complex in "
          "POINT_KEYS order. near_interface_six_columns' contract otherwise.");
}
