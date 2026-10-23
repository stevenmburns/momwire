// The point-observer column twin, Python-free (momwire#1653): the kernel of
// `near_interface_point_columns` (momwire#1224) MOVED VERBATIM from
// `_near_interface_point_accel.cpp`, which now binds it, so the buried
// near-field readout (`_near_buried_core.cpp`) and the standalone executable
// call the very code the extension does. The TU keeps that file's preamble
// (the libmvec `omp declare simd` block and `MW_NI_SIMD` before
// `_near_interface_columns_inline.h`), so the columns vectorize as they did.
#define _USE_MATH_DEFINES

#include "_near_interface_point_core.h"

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
#include "_contour_lanes_inline.h"
#include "_branch_cut_inline.h"
// The vendored xsf defines non-inline functions, so one link unit may hold
// one copy of each: this TU's are renamed, as `_near_interface_core.cpp`'s
// are (the extension, `_accelerators` and the executable each link both).
#undef xsf
#define MW_NP_CAT2(a, b) a##b
#define MW_NP_CAT(a, b) MW_NP_CAT2(a, b)
#define xsf MW_NP_CAT(mw, _np_xsf)
#include "xsf/bessel.h"
// faddeeva.h keeps its own namespace, outside xsf: renamed too.
#undef Faddeeva
#define Faddeeva MW_NP_CAT(mw, _np_Faddeeva)
#include "xsf/faddeeva.h"

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
#include "_vmath_inline.h"
#include "_near_interface_columns_inline.h"
#include "_near_interface_sdp_inline.h"
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
    // The extents `column_rule` itself used, D1 included (momwire#1595).
    double a_head, lam_top;
    mw899::column_extents(rho, k_p, k_m, s_min, lam_mult, a_head, lam_top);
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

    if (lam_top > a_head) {  // no mid where the rays start at a_head
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
    }

    const double scale = std::sqrt(2.0) / (s_min + rho);
    double step = std::min(0.25 * scale, lam_top);
    edges.clear();
    edges.push_back(0.0);
    while (edges.back() < mw680::MW_FAR_PAIR_KILL * scale) {
        edges.push_back(edges.back() + step);
        step *= 2.0;
    }
    // The tail's own 16-point rule, as `column_tail` builds it (momwire#1595 D2).
    mw899::fixed_gauss(edges, p, mw899::MW_TAIL_GX, mw899::MW_TAIL_GW, mw899::MW_TAIL_NG, t, wt);
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
#if MW_NI_VMATH
        for (int j = 0; j < nb; ++j)
            ai[j] = mw_fma::fma(gmi[k0 + j], zp, -(gpi[k0 + j] * z));
        mw_vmath::exp_cis_lanes<mw899::vmath_backend>(ar, ai, er, ei, nb);
#else
        MW_NI_SIMD()
        for (int j = 0; j < nb; ++j) {
            ai[j] = mw_fma::fma(gmi[k0 + j], zp, -(gpi[k0 + j] * z));
            const double ex = exp(ar[j]);
            er[j] = ex * cos(ai[j]);
            ei[j] = ex * sin(ai[j]);
        }
#endif
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

// The point family's far columns on the steepest-descent path (momwire#1653):
// `_near_interface._point_columns_sdp`'s twin, on the six twin's path, rules
// and crossover (`_near_interface_sdp_inline.h`, momwire#1595 D1b). The J0
// rows take 1/2 H0^(2) on the path and the J1 rows 1/2 H1^(2) (the parity
// note at `_sdp_point_factors`), on the rule built into the Scratch planes
// `point_member` sums.
namespace mw1653 {
using mw_contour::cd;
using mw899::Scratch;

static inline cd hankel2_1(const cd &x) { return xsf::cyl_hankel_2(1.0, x); }

// `_point_column_factors` at one node: POINT_KEYS order, w0 on the z rows
// and w1 on the rho rows.
static inline void factors(const cd &lam, const cd &w0, const cd &w1, const cd &g_p, const cd &g_m,
                           const cd &k_p, const cd &k_m, cd *f) {
    const cd v = 2.0 * lam / (k_m * k_m * g_p + k_p * k_p * g_m);
    const cd wv = (g_p - g_m) * v;
    f[0] = -lam * v * w1;
    f[1] = -g_p * v * w0;
    f[2] = -lam * wv * w1;
    f[3] = g_m * v * w0;
}

// `_sdp_point_factors` at one path point s, times the quadrature weight `ws`.
static inline void path_node(const cd &s, double theta, double rho, double k_p, const cd &k_m,
                             const cd &ws, cd *f, cd &g_p, cd &g_m) {
    cd w, dpsi;
    mw1595::path(theta, s, w, dpsi);
    const cd lam = k_p * std::sin(w);
    const cd d = k_p * std::cos(w) * dpsi;
    const cd w0 = 0.5 * mw1595::hankel2_0(lam * rho) * d;
    const cd w1 = 0.5 * hankel2_1(lam * rho) * d;
    g_p = cd(0.0, 1.0) * k_p * std::cos(w);
    g_m = mw_branch::gamma_cut(lam, k_m);
    factors(lam, w0 * ws, w1 * ws, g_p, g_m, cd(k_p, 0.0), k_m, f);
}

// `_sdp_pole(..., factors=_sdp_point_factors)`: s_p and the residue (4,).
static bool pole(double k_p, const cd &k_m, double theta, double rho, cd &sp, cd *res) {
    const cd j(0.0, 1.0);
    const cd lam = k_p * k_m / std::sqrt(k_p * k_p + k_m * k_m);
    const cd cw = j * k_p * mw_branch::gamma_cut(lam, k_m) / (k_m * k_m);
    const cd wp = -j * std::log(cw + j * lam / k_p);
    sp = std::sqrt(-j * (1.0 - std::cos(wp - theta)));
    const cd guess = (wp - theta) / cd(1.0, 1.0);
    if (std::abs(-sp - guess) < std::abs(sp - guess)) sp = -sp;
    {
        cd w, dpsi;
        mw1595::path(theta, sp, w, dpsi);
        const cd lam_p = k_p * std::sin(w);
        const cd d = k_m * k_m * (j * k_p * std::cos(w)) + k_p * k_p * mw_branch::gamma_cut(lam_p, k_m);
        if (std::abs(d) > 1e-9 * std::norm(k_m) * k_p) return false;
    }
    const int NP = mw1224::NP;
    auto contour = [&](double radius, cd *out) {
        const int m = 64;
        for (int c = 0; c < NP; ++c) out[c] = 0.0;
        for (int i = 0; i < m; ++i) {
            const double ang = 2.0 * M_PI * (i + 0.5) / m;
            const cd off = radius * cd(std::cos(ang), std::sin(ang));
            cd f[mw1224::NP], gp, gm;
            path_node(sp + off, theta, rho, k_p, k_m, cd(1.0, 0.0), f, gp, gm);
            for (int c = 0; c < NP; ++c) out[c] += f[c] * off;
        }
        bool finite = true;
        for (int c = 0; c < NP; ++c) {
            out[c] /= double(m);
            finite = finite && std::isfinite(out[c].real()) && std::isfinite(out[c].imag());
        }
        return finite;
    };
    double radius = (sp != cd(0.0, 0.0)) ? std::min(1e-2, 0.5 * std::abs(sp)) : 1e-2;
    if (!contour(radius, res)) return false;
    for (int it = 0; it < 6; ++it) {
        cd half[mw1224::NP];
        if (!contour(0.5 * radius, half)) return false;
        double dmax = 0.0, rmax = 0.0;
        for (int c = 0; c < NP; ++c) {
            dmax = std::max(dmax, std::abs(half[c] - res[c]));
            rmax = std::max(rmax, std::abs(res[c]));
        }
        if (dmax <= 1e-12 * std::max(rmax, 1e-300)) return true;
        radius *= 0.5;
        for (int c = 0; c < NP; ++c) res[c] = half[c];
    }
    return false;
}

// `_sdp_point_nodes` into the Scratch planes (`point_factors`' layout).
static size_t build(double rho, double k_p, const cd &k_m, double theta, double kr, Scratch &s) {
    const int NP = mw1224::NP;
    const cd j(0.0, 1.0);
    const mw1595::Rule &g = mw1595::rule_n();
    const double S = std::sqrt(mw1595::SDP_TAIL / kr);
    std::vector<cd> F, GP, GM;
    auto push = [&](const cd *f, const cd &gp, const cd &gm) {
        for (int c = 0; c < NP; ++c) F.push_back(f[c]);
        GP.push_back(gp);
        GM.push_back(gm);
    };
    for (int i = 0; i < mw1595::SDP_N; ++i) {
        cd f[mw1224::NP], gp, gm;
        path_node(cd(S * g.x[i], 0.0), theta, rho, k_p, k_m, cd(S * g.w[i], 0.0), f, gp, gm);
        push(f, gp, gm);
    }
    cd sp, res[mw1224::NP];
    if (pole(k_p, k_m, theta, rho, sp, res) && std::abs(std::sqrt(kr) * sp) <= mw1595::SDP_A_MAX) {
        const cd a = std::sqrt(kr) * sp;
        const cd whole = a.imag() > 0.0 ? j * M_PI * Faddeeva::w(a, 0.0) : -j * M_PI * Faddeeva::w(-a, 0.0);
        cd gsum = 0.0;
        for (int i = 0; i < mw1595::SDP_N; ++i) {
            const double si = S * g.x[i];
            gsum += std::exp(-kr * (si * si - sp * sp)) / (si - sp) * (S * g.w[i]);
        }
        const cd corr = std::exp(kr * sp * sp) * whole - gsum;
        cd f1[mw1224::NP], gp1, gm1;
        path_node(sp, theta, rho, k_p, k_m, cd(1.0, 0.0), f1, gp1, gm1);
        cd f[mw1224::NP];
        for (int c = 0; c < NP; ++c) f[c] = res[c] * corr;
        push(f, gp1, gm1);
    }
    if (k_m.imag() < 0.0 && mw1595::encloses(k_p, k_m, theta, S)) {
        const mw1595::Rule &c = mw1595::rule_cut();
        const double t_max = mw1595::SDP_CUT_T / rho;
        const double rt = std::sqrt(t_max);
        std::vector<cd> fb, gpb, gmb;
        for (int i = 0; i < mw1595::SDP_NCUT; ++i) {
            const double u = 0.5 * (c.x[i] + 1.0) * rt;
            const double dt = 2.0 * u * 0.5 * rt * c.w[i];
            const cd lam(k_m.real(), k_m.imag() - u * u);
            const cd gp = mw_branch::gamma_cut(lam, cd(k_p, 0.0));
            const cd gm = mw_branch::gamma_cut(lam, k_m);
            const cd h0 = 0.5 * mw1595::hankel2_0(lam * rho) * j * dt;
            const cd h1 = 0.5 * hankel2_1(lam * rho) * j * dt;
            cd f[mw1224::NP];
            factors(lam, h0, h1, gp, gm, cd(k_p, 0.0), k_m, f);
            push(f, gp, gm);
            factors(lam, -h0, -h1, gp, -gm, cd(k_p, 0.0), k_m, f);
            for (int q = 0; q < NP; ++q) fb.push_back(f[q]);
            gpb.push_back(gp);
            gmb.push_back(-gm);
        }
        // the walk's order: one cut side's 32 nodes, then the other's
        const size_t n0 = GP.size() - mw1595::SDP_NCUT;
        std::vector<cd> Fa(F.begin() + NP * n0, F.end()), gpa(GP.begin() + n0, GP.end()),
            gma(GM.begin() + n0, GM.end());
        F.resize(NP * n0);
        GP.resize(n0);
        GM.resize(n0);
        F.insert(F.end(), Fa.begin(), Fa.end());
        F.insert(F.end(), fb.begin(), fb.end());
        GP.insert(GP.end(), gpa.begin(), gpa.end());
        GP.insert(GP.end(), gpb.begin(), gpb.end());
        GM.insert(GM.end(), gma.begin(), gma.end());
        GM.insert(GM.end(), gmb.begin(), gmb.end());
    }
    const size_t K = GP.size();
    s.lam.assign(K, cd(0.0, 0.0));  // sized for `point_member`'s K only
    s.gpr.resize(K);
    s.gpi.resize(K);
    s.gmr.resize(K);
    s.gmi.resize(K);
    s.fr.resize(NP * K);
    s.fi.resize(NP * K);
    for (size_t k = 0; k < K; ++k) {
        s.gpr[k] = GP[k].real();
        s.gpi[k] = GP[k].imag();
        s.gmr[k] = GM[k].real();
        s.gmi[k] = GM[k].imag();
        for (int c = 0; c < NP; ++c) {
            s.fr[c * K + k] = F[NP * k + c].real();
            s.fi[c * K + k] = F[NP * k + c].imag();
        }
    }
    return K;
}

// One column's rule into `s`: the steepest-descent rule for a far column,
// else `_column_rule_j1` and the point factors. Returns false when the J1
// weights left the rule's nodes; K is the rule's node count.
template <class ZB, class PB>
static bool column(double rho, double k_p, const cd &kpc, const cd &km, const ZB &zb, const PB &pb,
                   const double *z, std::ptrdiff_t lo, std::ptrdiff_t hi, double lam_mult, int p, double detour,
                   const double *gx, const double *gw, int ng, bool sdp, Scratch &s, size_t &K) {
    double theta, kr;
    if (sdp && mw1595::far_column(rho, z, lo, hi, k_p, theta, kr)) {
        K = build(rho, k_p, km, theta, kr, s);
        return true;
    }
    std::vector<cd> w1;
    if (!mw1224::column_rule_j1(rho, k_p, km, mw899::s_min_of(zb, pb, lo, hi), lam_mult, p, detour, gx, gw, ng,
                                s, w1))
        return false;
    mw1224::point_factors(kpc, km, w1, s);
    K = s.lam.size();
    return true;
}
}  // namespace mw1653

namespace ni_point {

void point_columns(double k_p, std::complex<double> k_m, const double *rho,
                   const std::ptrdiff_t *offsets, std::ptrdiff_t nc, const double *z,
                   const double *zp, std::ptrdiff_t n, double lam_mult, int p, double detour,
                   int n_threads, const double *gx, const double *gw, int ng,
                   std::complex<double> *out, bool sdp) {
    // The binding's checks, in its words.
    if (offsets[0] != 0 || offsets[nc] != n)
        throw std::invalid_argument("offsets must span 0 .. len(z)");
    for (std::ptrdiff_t c = 0; c < nc; ++c)
        if (offsets[c + 1] < offsets[c])
            throw std::invalid_argument("offsets must be non-decreasing");
    if (ng < 1) throw std::invalid_argument("bad Gauss rule");
    if (p < 0 || p > 20) throw std::invalid_argument("column p out of range");
    for (std::ptrdiff_t c = 0; c < nc; ++c) {
        for (std::ptrdiff_t i = offsets[c]; i < offsets[c + 1]; ++i) {
            if (!(z[i] >= 0.0 && zp[i] <= 0.0))
                throw std::invalid_argument("need z >= 0 >= zp");
            if (rho[c] < 0.0 || (z[i] - zp[i]) + rho[c] <= 0.0)
                throw std::invalid_argument("need R > 0");
        }
    }
    const auto ob = [offsets](std::ptrdiff_t c) { return offsets[c]; };
    const auto rb = [rho](std::ptrdiff_t c) { return rho[c]; };
    const auto zb = [z](std::ptrdiff_t i) { return z[i]; };
    const auto pb = [zp](std::ptrdiff_t i) { return zp[i]; };
    const auto vb_set = [out](std::ptrdiff_t i, int q, const mw_contour::cd &v) {
        out[i * mw1224::NP + q] = v;
    };
    const double *gxp = gx;
    const double *gwp = gw;
    const mw_contour::cd kpc(k_p, 0.0);
    const mw_contour::cd km(k_m);
    int nt = 1;
#ifdef _OPENMP
    nt = omp_get_max_threads();
    if (n_threads > 0) nt = std::min(nt, n_threads);
    nt = std::max(nt, 1);
#endif
    const std::ptrdiff_t share = (n + nt - 1) / nt;
    std::vector<std::ptrdiff_t> by_column, by_member;
    for (std::ptrdiff_t c = 0; c < nc; ++c) {
        if (ob(c + 1) == ob(c)) continue;
        (ob(c + 1) - ob(c) > share ? by_member : by_column).push_back(c);
    }
    const std::ptrdiff_t n_col = static_cast<std::ptrdiff_t>(by_column.size());
    const std::ptrdiff_t n_mem = static_cast<std::ptrdiff_t>(by_member.size());
    bool nodes_moved = false;
    #pragma omp parallel for schedule(dynamic) num_threads(nt)
    for (std::ptrdiff_t j = 0; j < n_col; ++j) {
        const std::ptrdiff_t c = by_column[j];
        const std::ptrdiff_t lo = ob(c), hi = ob(c + 1);
        mw899::Scratch s;
        size_t K = 0;
        if (!mw1653::column(rb(c), k_p, kpc, km, zb, pb, z, lo, hi, lam_mult, p, detour, gxp, gwp, ng, sdp,
                            s, K)) {
            #pragma omp atomic write
            nodes_moved = true;
            continue;
        }
        for (std::ptrdiff_t i = lo; i < hi; ++i) {
            mw_contour::cd o[mw1224::NP];
            mw1224::point_member(s, K, zb(i), pb(i), o);
            for (int q = 0; q < mw1224::NP; ++q) vb_set(i, q, o[q]);
        }
    }
    for (std::ptrdiff_t j = 0; j < n_mem && !nodes_moved; ++j) {
        const std::ptrdiff_t c = by_member[j];
        const std::ptrdiff_t lo = ob(c), hi = ob(c + 1);
        mw899::Scratch s;
        size_t K = 0;
        if (!mw1653::column(rb(c), k_p, kpc, km, zb, pb, z, lo, hi, lam_mult, p, detour, gxp, gwp, ng, sdp,
                            s, K)) {
            nodes_moved = true;
            break;
        }
        #pragma omp parallel for schedule(dynamic, 32) num_threads(nt)
        for (std::ptrdiff_t i = lo; i < hi; ++i) {
            mw_contour::cd o[mw1224::NP];
            mw1224::point_member(s, K, zb(i), pb(i), o);
            for (int q = 0; q < mw1224::NP; ++q) vb_set(i, q, o[q]);
        }
    }
    if (nodes_moved) throw std::runtime_error("the J1 weights no longer follow _column_rule's nodes");
}

}  // namespace ni_point
