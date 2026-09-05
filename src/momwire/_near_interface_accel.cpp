// The C++ twins of the designed near-interface walk: `six_point` per triple
// (momwire#680 U2) and `six_columns` per rho column (momwire#899 item 1).
//
// TWO ENTRIES, ONE INTEGRAND. `near_interface_six_batch` walks each triple
// adaptively; `near_interface_six_columns` builds the FIXED per-column rule
// once and spends one exponential per node per member. They are the two
// routes `_near_interface.designed_tables` dispatches between, and each is
// the twin of the numpy walk of the same name — never of the other.
//
// Both are WALK ports, not integrand ports (the house twin rule: shared
// walk + limit, never pointwise): the head/mid machinery is the very same
// templated engine `_accelerators.cpp` rides (`_contour_engine_inline.h`,
// momwire#568 U1 — `adaptive_segment`, `head_contour`, the complex J0/J1
// kernels), instantiated here with the crossing serve's six-component
// spectral core, plus a transcription of the ONE structure the engine does
// not carry: the rotated-ray tail lam = lam_top + t e^{+-j pi/4}. Every
// structural rule of the Python walk survives verbatim:
//
//   * head [0, 1.1K] via the first-quadrant half-sine detour (branch points
//     + transmitted pole inside), mid [1.1K, 8K] real-axis adaptive Gauss,
//     tail = rotated rays;
//   * ray panels START at the lam0 scale and double toward the decay scale
//     sqrt(2)/(s+rho) — starting at the decay scale is a measured 44 %
//     silent W-log error at s = 1e-5;
//   * exact-underflow ray panels count as QUIET (high-sigma far pairs
//     underflow the whole tail to exact 0.0 — the tail IS zero, not a
//     stall); two consecutive quiet panels end the ray;
//   * the far-pair kill cap lam <= 60/s, with the head keeping the k_p
//     branch point + transmitted pole inside;
//   * rho = 0 takes ONE up-ray with J0 = 1; rho > 0 splits J0 into Hankel
//     halves, H1 up-ray / H2 down-ray (conjugate ray direction);
//   * derivative bookkeeping: dz <-> -gamma_p, dz' <-> +gamma_m; z' <= 0 so
//     e^{-gamma_m|z'|} = e^{+gamma_m z'}.
//
// The complex-argument Hankel pair comes from the VENDORED scipy/xsf
// (extern/xsf, header-only C++17, scipy.special's own Amos translation —
// see extern/xsf/VENDOR-PIN.md). Its underflow contract is load-bearing:
// Amos underflow returns the exact 0.0 rather than NaN, which is precisely
// the quiet-panel rule above. This is also why the module is compiled at
// C++17 in its OWN extension: `_accelerators` stays at gnu++11, untouched.
//
// The numpy walk (`_near_interface.six_point`) stays production's reference;
// the gates against it are RELATIVE (1e-12), never bit — the transcendental
// libraries and the Gauss dot products differ in the last bits (house rule:
// no cross-build bit equality, momwire#249/#270).
//
// REENTRANCY: the engine header is allocation-free with no mutable static
// (its #568 U1 contract), the spectral core below holds only per-point
// scalars, and xsf's Amos kernels are pure functions — so the batch loop
// runs OpenMP `dynamic` over points with the GIL released, exactly the
// `below_six_integrals_batch` pattern. Domain validation happens BEFORE the
// parallel region (a throw cannot cross an omp boundary); the one in-loop
// failure mode (a ray that never goes quiet) sets a flag that is raised
// afterwards in the Python walk's own words.
//
// The COLUMN entry (#899 item 1) carries the rule as well as the sum, and
// that is the whole point of it: on the measured BLE column a third of the
// numpy route's wall is `_column_rule` + `_column_factors` (complex-argument
// Bessel/Hankel through scipy, the `_sub_seed` loop in Python, the Gauss
// panels) and two thirds the (nz x K) exponential, so a twin that only did
// exp + dot would leave the third behind. Its parallel unit is the COLUMN,
// not the point: a column's rule is built by the thread that owns it, every
// scratch buffer is thread-local, and there is no shared mutable state — so
// it also supplies the thread scaling the numpy route has none of (#898).

#define _USE_MATH_DEFINES

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

#include <algorithm>
#include <cmath>
#include <complex>
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

// The column entry's inner loop is exp + sincos per node per member, which
// is exactly what glibc's libmvec vectorizes (`-lmvec` is already on this
// extension's link line). Ubuntu's <cmath> carries no `omp declare simd`
// markers for them, so GCC cannot substitute `_ZGVdN4v_exp` / `_ZGVdN4v_cos`
// / `_ZGVdN4v_sin` inside an `omp simd` loop without these declarations; the
// std:: overloads still resolve to the same extern-C symbols, so the calls
// below pick the vectorized form up for free. Same block, same gating and
// same reason as `_accel_common.h`'s (the sibling extension's) — MSVC has no
// libmvec and would choke on redeclaring the CRT's exp/sin/cos, and macOS
// has neither libmvec nor the AVX2 simdlen(4) form.
#if defined(__GNUC__) && !defined(_MSC_VER) && !defined(__APPLE__)
#pragma omp declare simd notinbranch simdlen(4)
extern "C" double exp(double);

#pragma omp declare simd notinbranch simdlen(4)
extern "C" double cos(double);

#pragma omp declare simd notinbranch simdlen(4)
extern "C" double sin(double);
#endif

// `omp simd` neutralization for MSVC, whose /openmp:llvm rejects the
// directive outright (the `_accel_common.h` note has the long form). The
// pragmas below are a vectorization hint and nothing else: the arithmetic is
// a plain reduction either way, so dropping them costs speed, never bits.
#if defined(_MSC_VER)
#define MW_NI_SIMD(clauses)
#else
#define MW_NI_PRAGMA_(x) _Pragma(#x)
#define MW_NI_SIMD(clauses) MW_NI_PRAGMA_(omp simd clauses)
#endif

namespace mw680 {
using mw_contour::cd;

// The Python walk's fixed shape constants (`_near_interface`): the ray
// direction e^{+j pi/4}, the panel budget, and the far-pair kill cap.
// Transcribed as constants rather than parameters because they are part of
// the walk's SHAPE — a twin that let them drift would not be a twin.
static const int MW_MAX_RAY_PANELS = 90;
static const double MW_FAR_PAIR_KILL = 60.0;

// The branch cut: one definition for all three call sites, across both
// extensions — the full rationale (the two-sqrt spelling IS the branch
// choice; head detour and rotated rays cross neither vertical cut) lives
// with the definition in _branch_cut_inline.h (#714). Calls in this
// namespace are unqualified; the using-declaration binds them.
using mw_branch::gamma_cut;

// `_near_interface._core`: the six spectral factors x (2 E~ lam), WITHOUT
// the Bessel factor. z' <= 0, so e^{-gamma_m|z'|} = e^{+gamma_m z'}.
// Stacked: 0 U, 1 V, 2 W, 3 dzW (= -gamma_p W~ under the integral),
// 4 dz dz' V (= -gamma_p gamma_m V~), 5 dz'W (= +gamma_m W~).
struct CoreSix {
    double z;
    double zp;
    cd k_p;
    cd k_m;
    cd kp2;  // k_p^2, hoisted
    cd km2;  // k_m^2, hoisted
    void operator()(const cd &lam, cd *out) const {
        const cd g_p = gamma_cut(lam, k_p);
        const cd g_m = gamma_cut(lam, k_m);
        const cd e = 2.0 * std::exp(g_m * zp - g_p * z) * lam;
        const cd u = e / (g_p + g_m);
        const cd v = e / (km2 * g_p + kp2 * g_m);
        const cd w = (g_p - g_m) * v;
        out[0] = u;
        out[1] = v;
        out[2] = w;
        out[3] = -g_p * w;
        out[4] = -(g_p * g_m) * v;
        out[5] = g_m * w;
    }
};

// The head/mid integrand: core(lam) * J0(lam rho), on the engine's own
// complex J0 kernels (gated against scipy.special.jv at #568 U1).
struct J0Core {
    const CoreSix &core;
    double rho;
    J0Core(const CoreSix &c, double r) : core(c), rho(r) {}
    void operator()(const cd &lam, cd *out) const {
        core(lam, out);
        cd b0, b1x;
        mw_contour::bessel_j0_j1x(lam * rho, b0, b1x);
        for (int c = 0; c < 6; ++c) out[c] *= b0;
    }
};

// One rotated ray's integrand over the REAL panel parameter t:
// core(lam0 + t dir) * factor * dir, factor in {1, H1_0(lam rho)/2,
// H2_0(lam rho)/2}. The Hankel halves are xsf's Amos pair at complex
// argument; on the up-ray lam rho gains positive imaginary part and H1
// decays, on the (conjugate-direction) down-ray H2 does — underflow to
// exact 0.0 included, which the quiet rule wants.
enum RayKind { RAY_ONE = 0, RAY_H1 = 1, RAY_H2 = 2 };

struct RayIntegrand {
    const CoreSix &core;
    cd lam0;
    cd dir;
    double rho;
    int kind;
    void operator()(double t, cd *out) const {
        const cd lam = lam0 + t * dir;
        core(lam, out);
        cd fac = dir;
        if (kind == RAY_H1) fac *= 0.5 * xsf::cyl_hankel_1(0.0, lam * rho);
        else if (kind == RAY_H2) fac *= 0.5 * xsf::cyl_hankel_2(0.0, lam * rho);
        for (int c = 0; c < 6; ++c) out[c] *= fac;
    }
};

// `_near_interface._ray_integral`: geometric panels, each adaptive Gauss;
// stops when two consecutive panels contribute < rtol of the running total.
// Panels START at min(0.25 * scale, lam0) and double (the 44 % trap above).
// Returns false when the panel budget is exhausted without going quiet —
// the caller raises in the Python walk's words, outside the omp region.
static bool ray_integral(const RayIntegrand &g, double lam0, double scale,
                         double rtol, int depth, const double *gx,
                         const double *gw, int ng, cd *acc) {
    for (int c = 0; c < 6; ++c) acc[c] = cd(0.0, 0.0);
    double t_lo = 0.0;
    double step = std::min(0.25 * scale, lam0);
    int quiet = 0;
    cd part[6];
    for (int p = 0; p < MW_MAX_RAY_PANELS; ++p) {
        const double t_hi = t_lo + step;
        mw_contour::adaptive_segment<6>(g, t_lo, t_hi, rtol, depth, gx, gw, ng,
                                        0, 0.0, part);
        double ref = 0.0, pmax = 0.0;
        for (int c = 0; c < 6; ++c) {
            acc[c] += part[c];
            ref = std::max(ref, std::abs(acc[c]));
            pmax = std::max(pmax, std::abs(part[c]));
        }
        // ref == 0.0: the whole ray underflows to exact 0 (high-sigma far
        // pairs); consecutive all-zero panels are quiet, the tail IS zero.
        if (ref == 0.0 || pmax < rtol * ref) {
            if (++quiet >= 2) return true;
        } else {
            quiet = 0;
        }
        t_lo = t_hi;
        step *= 2.0;
    }
    return false;
}

// `six_point` for ONE (rho, z, zp): head + mid + tail. Returns false on a
// non-quiet ray (see above); every other pathway matches the Python walk's
// arithmetic term for term.
static bool six_point_one(double k_p, const cd &k_m, double rho, double z,
                          double zp, double rtol, double lam_mult, int depth,
                          double detour, const double *gx, const double *gw,
                          int ng, cd *out) {
    const double s = z - zp;
    const double kk = std::max(k_p, std::abs(k_m));
    double a_head = 1.1 * kk;
    double lam_top = lam_mult * kk;
    // Far-pair kill cap (sigma = 5 class, |k_m| >> k_p): beyond
    // lam ~ 60/s the integrand is e^{-60} of the total — dead range. Cap
    // the extents there, keeping the k_p branch point + transmitted pole
    // (|lam_p| ~ k_p) inside the head. Inactive for every near-interface
    // pair, so nothing the corner probes pinned changes.
    if (s > 0.0 && MW_FAR_PAIR_KILL / s < lam_top) {
        const double lam_kill = MW_FAR_PAIR_KILL / s;
        a_head = std::max(2.2 * k_p, std::min(a_head, lam_kill));
        lam_top = std::max(1.5 * a_head, lam_kill);
    }

    CoreSix core;
    core.z = z;
    core.zp = zp;
    core.k_p = cd(k_p, 0.0);
    core.k_m = k_m;
    core.kp2 = core.k_p * core.k_p;
    core.km2 = k_m * k_m;
    const J0Core f_j0(core, rho);

    cd head[6], mid[6];
    const double marks[2] = {k_p, std::fabs(k_m.real())};
    mw_contour::head_contour<6>(f_j0, a_head, rho, marks, 2, rtol, depth,
                                detour, gx, gw, ng, head);
    const mw_contour::TailWrap<6, J0Core> f_mid(f_j0);
    mw_contour::adaptive_segment<6>(f_mid, a_head, lam_top, rtol, depth, gx,
                                    gw, ng, 0, 0.0, mid);

    const double scale = std::sqrt(2.0) / (s + rho);
    const cd ray_dir = std::exp(cd(0.0, 0.25 * mw_contour::MW_PI));
    cd tail[6];
    if (rho == 0.0) {
        RayIntegrand g{core, cd(lam_top, 0.0), ray_dir, rho, RAY_ONE};
        if (!ray_integral(g, lam_top, scale, rtol, depth, gx, gw, ng, tail))
            return false;
    } else {
        RayIntegrand up{core, cd(lam_top, 0.0), ray_dir, rho, RAY_H1};
        RayIntegrand dn{core, cd(lam_top, 0.0), std::conj(ray_dir), rho,
                        RAY_H2};
        cd dn_acc[6];
        if (!ray_integral(up, lam_top, scale, rtol, depth, gx, gw, ng, tail))
            return false;
        if (!ray_integral(dn, lam_top, scale, rtol, depth, gx, gw, ng, dn_acc))
            return false;
        for (int c = 0; c < 6; ++c) tail[c] += dn_acc[c];
    }
    for (int c = 0; c < 6; ++c) out[c] = head[c] + mid[c] + tail[c];
    return true;
}
}  // namespace mw680

// ---------------------------------------------------------------------------
// momwire#899 item 1: the twin of `_near_interface.six_columns` — the FIXED
// per-rho column rule, built once per column and shared by every member.
// ---------------------------------------------------------------------------
namespace mw899 {
using mw_branch::gamma_cut;
using mw_contour::cd;
using mw_contour::MW_PI;

// The exponent's real part below which e^{Re} is EXACTLY 0.0 in IEEE double:
// the smallest subnormal is e^{-744.44}, so anything under -746 rounds to
// zero and the whole term e^{Re}(cos, sin) is (+-0, +-0). Skipping such a
// node is therefore bit-identical to evaluating it, not a tolerance — it is
// the walk's own quiet rule read from the other end. The column's extents
// are converged for its SMALLEST s (`_column_rule`), so every member with a
// larger s carries tail nodes past its own decay, and on a high-sigma far
// pair that is the whole tail (`six_columns`' underflow comment).
static const double MW_EXP_ZERO = -746.0;

// Nodes are processed in blocks of this many so the underflow skip can be
// taken a BLOCK at a time: the exponent grows along the contour, so the dead
// nodes are a contiguous run, and a per-node branch would sit inside the
// vectorized loop instead of in front of it.
static const int MW_BLOCK = 64;

// `_fixed_gauss`: the shipped Gauss rule on each [e_i, e_{i+1}], each split
// into 2^p equal panels. numpy's linspace is start + i*(stop - start)/n with
// the last node set to stop exactly; spelled that way here so the panels are
// the same floats and not merely the same partition.
static void fixed_gauss(const std::vector<double> &edges, int p,
                        const double *gx, const double *gw, int ng,
                        std::vector<double> &t, std::vector<double> &wt) {
    t.clear();
    wt.clear();
    const int npan = 1 << p;
    for (size_t i = 0; i + 1 < edges.size(); ++i) {
        const double e0 = edges[i], e1 = edges[i + 1];
        const double d = (e1 - e0) / static_cast<double>(npan);
        for (int j = 0; j < npan; ++j) {
            const double a = e0 + static_cast<double>(j) * d;
            const double b =
                (j + 1 == npan) ? e1 : e0 + static_cast<double>(j + 1) * d;
            const double mid = 0.5 * (a + b), half = 0.5 * (b - a);
            for (int q = 0; q < ng; ++q) {
                t.push_back(mid + half * gx[q]);
                wt.push_back(gw[q] * half);
            }
        }
    }
}

// `_sub_seed`: no interval spans more than a doubling in lam or one
// oscillation of J0(lam rho). BOTH halves are load-bearing and the doubling
// one is easy to get subtly wrong — the step is min(e - lo, lat, lo), with
// the `lo` term dropped only at lo = 0, and the inner loop terminates on
// lo + step >= e rather than on a count. Neuter it and the 41 m radial reads
// 110 % wrong (`test_g895_7` is the red/green proof).
static void sub_seed(const std::vector<double> &edges, double rho,
                     std::vector<double> &out) {
    const double lat = (rho > 0.0) ? (2.0 * MW_PI / rho)
                                   : std::numeric_limits<double>::infinity();
    out.clear();
    out.push_back(edges[0]);
    for (size_t i = 1; i < edges.size(); ++i) {
        const double e = edges[i];
        for (;;) {
            const double lo = out.back();
            double step = std::min(e - lo, lat);
            if (lo > 0.0) step = std::min(step, lo);
            if (lo + step >= e) break;
            out.push_back(lo + step);
        }
        out.push_back(e);
    }
}

// Per-column scratch. Held by the thread that owns the column, so nothing
// here is shared and nothing needs a lock; the vectors are members only so a
// column's allocations are one block rather than a dozen.
struct Scratch {
    std::vector<double> edges, seeded, t, wt;
    std::vector<cd> lam, w;
    // The factors, SPLIT into real and imaginary parts: the member loop is a
    // real-arithmetic reduction, and interleaved std::complex<double> would
    // hand the vectorizer a stride-2 gather on every operand.
    std::vector<double> gpr, gpi, gmr, gmi, fr, fi;
};

// The column's smallest s = z - z', which is the one thing the rule needs
// from its members: the slowest-decaying member, and so the one the extents
// must be converged for.
template <class ZB, class PB>
static double s_min_of(const ZB &zb, const PB &pb, py::ssize_t lo,
                       py::ssize_t hi) {
    double s_min = zb(lo) - pb(lo);
    for (py::ssize_t i = lo + 1; i < hi; ++i)
        s_min = std::min(s_min, zb(i) - pb(i));
    return s_min;
}

// `_column_rule`: nodes and weights for one rho column, path derivative and
// Bessel/Hankel factor folded in. Every path decision is `six_point`'s,
// evaluated once for the column; the only thing the column chooses for
// itself is `s_min`, the smallest s = z - z' in it, which is the
// slowest-decaying member and so the one the extents must be converged for.
static void column_rule(double rho, double k_p, const cd &k_m, double s_min,
                        double lam_mult, int p, double detour, const double *gx,
                        const double *gw, int ng, Scratch &s) {
    const double kk = std::max(k_p, std::abs(k_m));
    double a_head = 1.1 * kk;
    double lam_top = lam_mult * kk;
    // `six_point`'s far-pair kill cap, on the column's SMALLEST s: that is
    // the largest 60/s, so the extents are the least capped any member would
    // ask for and no member loses range it needed.
    if (s_min > 0.0 && mw680::MW_FAR_PAIR_KILL / s_min < lam_top) {
        const double lam_kill = mw680::MW_FAR_PAIR_KILL / s_min;
        a_head = std::max(2.2 * k_p, std::min(a_head, lam_kill));
        lam_top = std::max(1.5 * a_head, lam_kill);
    }
    s.lam.clear();
    s.w.clear();

    // --- head: `_head`'s detour, H rule and seeded edges (2 endpoints, 6
    // sevenths, 5 seeds per mark), then sub-seeded. H depends on rho alone,
    // so it is column-shared exactly. The Python side builds a SET and sorts
    // it; sort + exact-equality unique is the same thing, and it matters —
    // a duplicate edge would be a zero-width panel.
    double H = std::min(0.35 * a_head, detour / std::max(rho, 1e-12));
    H = std::max(H, 1e-6 * a_head);
    s.edges.clear();
    s.edges.push_back(0.0);
    s.edges.push_back(a_head);
    for (int i = 1; i < 7; ++i) s.edges.push_back(a_head * i / 7.0);
    const double marks[2] = {k_p, std::fabs(k_m.real())};
    const double ws[5] = {0.0, -0.15, 0.15, -0.4, 0.4};
    for (int m = 0; m < 2; ++m) {
        for (int j = 0; j < 5; ++j) {
            const double v = marks[m] * (1.0 + ws[j]);
            if (v > 0.0 && v < a_head) s.edges.push_back(v);
        }
    }
    std::sort(s.edges.begin(), s.edges.end());
    s.edges.erase(std::unique(s.edges.begin(), s.edges.end()), s.edges.end());
    sub_seed(s.edges, rho, s.seeded);
    fixed_gauss(s.seeded, p, gx, gw, ng, s.t, s.wt);
    for (size_t i = 0; i < s.t.size(); ++i) {
        const double t = s.t[i];
        const cd l(t, H * std::sin(MW_PI * t / a_head));
        const cd dl(1.0, H * (MW_PI / a_head) * std::cos(MW_PI * t / a_head));
        cd b0, b1x;
        mw_contour::bessel_j0_j1x(l * rho, b0, b1x);
        s.lam.push_back(l);
        s.w.push_back(s.wt[i] * dl * b0);
    }

    // --- mid: the real axis [a_head, lam_top], same J0 factor. `six_point`
    // hands the WHOLE range to one adaptive segment, so unlike the head it
    // carries no seeding of its own and `sub_seed` supplies all of it.
    s.edges.clear();
    s.edges.push_back(a_head);
    s.edges.push_back(lam_top);
    sub_seed(s.edges, rho, s.seeded);
    fixed_gauss(s.seeded, p, gx, gw, ng, s.t, s.wt);
    for (size_t i = 0; i < s.t.size(); ++i) {
        const cd l(s.t[i], 0.0);
        cd b0, b1x;
        mw_contour::bessel_j0_j1x(l * rho, b0, b1x);
        s.lam.push_back(l);
        s.w.push_back(s.wt[i] * b0);
    }

    // --- tail: `_ray_integral`'s geometric panels — starting at the lam0
    // scale and doubling toward the decay scale, which is what resolves the
    // 1/lam log content when s + rho is tiny — run out to 60 decay lengths
    // instead of to the adaptive quiet test. e^{-60} = 9e-27 of the total,
    // 16 decades inside any rtol a caller asks for, which is why this rule
    // takes no rtol at all.
    const double scale = std::sqrt(2.0) / (s_min + rho);
    double step = std::min(0.25 * scale, lam_top);
    s.edges.clear();
    s.edges.push_back(0.0);
    while (s.edges.back() < mw680::MW_FAR_PAIR_KILL * scale) {
        s.edges.push_back(s.edges.back() + step);
        step *= 2.0;
    }
    fixed_gauss(s.edges, p, gx, gw, ng, s.t, s.wt);
    const cd ray = std::exp(cd(0.0, 0.25 * MW_PI));
    if (rho == 0.0) {
        // `six_point`: the single up-ray, J0(0) = 1, no Hankel split.
        for (size_t i = 0; i < s.t.size(); ++i) {
            s.lam.push_back(lam_top + s.t[i] * ray);
            s.w.push_back(s.wt[i] * ray);
        }
    } else {
        for (size_t i = 0; i < s.t.size(); ++i) {
            const cd up = lam_top + s.t[i] * ray;
            s.lam.push_back(up);
            s.w.push_back(s.wt[i] * ray * 0.5 *
                          xsf::cyl_hankel_1(0.0, up * rho));
        }
        for (size_t i = 0; i < s.t.size(); ++i) {
            const cd dn = lam_top + s.t[i] * std::conj(ray);
            s.lam.push_back(dn);
            s.w.push_back(s.wt[i] * std::conj(ray) * 0.5 *
                          xsf::cyl_hankel_2(0.0, dn * rho));
        }
    }
}

// `_column_factors`: `_core` at every node with its z-dependent exponential
// factored out and the weights folded in, such that
// six(z, z') = F @ exp(gamma_m z' - gamma_p z). Index order is `_core`'s:
// 0 U, 1 V, 2 W, 3 dzW, 4 dz dz' V, 5 dz'W.
static void column_factors(const cd &k_p, const cd &k_m, Scratch &s) {
    const size_t K = s.lam.size();
    const cd kp2 = k_p * k_p, km2 = k_m * k_m;
    s.gpr.resize(K);
    s.gpi.resize(K);
    s.gmr.resize(K);
    s.gmi.resize(K);
    s.fr.resize(6 * K);
    s.fi.resize(6 * K);
    for (size_t k = 0; k < K; ++k) {
        const cd l = s.lam[k], wk = s.w[k];
        const cd g_p = gamma_cut(l, k_p);
        const cd g_m = gamma_cut(l, k_m);
        const cd u = 2.0 * l / (g_p + g_m);
        const cd v = 2.0 * l / (km2 * g_p + kp2 * g_m);
        const cd wv = (g_p - g_m) * v;
        const cd f[6] = {u * wk,
                         v * wk,
                         wv * wk,
                         (-g_p * wv) * wk,
                         (-(g_p * g_m) * v) * wk,
                         (g_m * wv) * wk};
        for (int c = 0; c < 6; ++c) {
            s.fr[c * K + k] = f[c].real();
            s.fi[c * K + k] = f[c].imag();
        }
        s.gpr[k] = g_p.real();
        s.gpi[k] = g_p.imag();
        s.gmr[k] = g_m.real();
        s.gmi[k] = g_m.imag();
    }
}

// One member of a built column: the fused exponential and six dot products,
// out[c] = sum_k F[c][k] e^{gamma_m z' - gamma_p z}.
//
// Fused deliberately. The numpy route materialises the (nz x K) exponential
// and hands it to a gemm, which on the measured column is 1.7 MB written and
// read back for 0.06 ms of arithmetic; here a member's exponentials live in
// L1 for the length of its own reduction and nothing but the answer is
// stored. The z' factor is NOT hoisted per distinct z' as the numpy route
// hoists it: there it saves a whole (nz x K) complex product, here it would
// save one multiply against a transcendental pair.
static void column_member(const Scratch &s, size_t K, double z, double zp,
                          cd *out) {
    double accr[6] = {0, 0, 0, 0, 0, 0}, acci[6] = {0, 0, 0, 0, 0, 0};
    double ar[MW_BLOCK], ai[MW_BLOCK], er[MW_BLOCK], ei[MW_BLOCK];
    const double *gpr = s.gpr.data(), *gpi = s.gpi.data();
    const double *gmr = s.gmr.data(), *gmi = s.gmi.data();
    for (size_t k0 = 0; k0 < K; k0 += MW_BLOCK) {
        const int nb =
            static_cast<int>(std::min<size_t>(MW_BLOCK, K - k0));
        double amax = -std::numeric_limits<double>::infinity();
        for (int j = 0; j < nb; ++j) {
            ar[j] = gmr[k0 + j] * zp - gpr[k0 + j] * z;
            amax = std::max(amax, ar[j]);
        }
        if (amax < MW_EXP_ZERO) continue;  // exactly zero, see MW_EXP_ZERO
        MW_NI_SIMD()
        for (int j = 0; j < nb; ++j) {
            ai[j] = gmi[k0 + j] * zp - gpi[k0 + j] * z;
            const double ex = exp(ar[j]);
            er[j] = ex * cos(ai[j]);
            ei[j] = ex * sin(ai[j]);
        }
        for (int c = 0; c < 6; ++c) {
            const double *fr = s.fr.data() + c * K + k0;
            const double *fi = s.fi.data() + c * K + k0;
            double sr = 0.0, si = 0.0;
            MW_NI_SIMD(reduction(+ : sr, si))
            for (int j = 0; j < nb; ++j) {
                sr += er[j] * fr[j] - ei[j] * fi[j];
                si += er[j] * fi[j] + ei[j] * fr[j];
            }
            accr[c] += sr;
            acci[c] += si;
        }
    }
    for (int c = 0; c < 6; ++c) out[c] = cd(accr[c], acci[c]);
}
}  // namespace mw899

// `six_point` over parallel (rho, z, zp) arrays: the (n, 6) table, OpenMP
// across points with the GIL released. The wavenumbers arrive DERIVED
// (k_p real, k_m complex on the Im <= 0 branch) rather than as eps~, so the
// branch choice stays in `k_medium` where it is written down once; the
// domain raises keep the Python walk's words. The U1 exact-triple memo
// stays in Python — this entry expects the caller to hand it UNIQUE
// triples (that list is also U3's parallel unit).
static py::array_t<std::complex<double>> near_interface_six_batch(
    double k_p, std::complex<double> k_m,
    py::array_t<double, py::array::c_style | py::array::forcecast> rho,
    py::array_t<double, py::array::c_style | py::array::forcecast> z,
    py::array_t<double, py::array::c_style | py::array::forcecast> zp,
    double rtol, double lam_mult, int depth, double detour,
    py::array_t<double, py::array::c_style | py::array::forcecast> gx,
    py::array_t<double, py::array::c_style | py::array::forcecast> gw) {
    auto rb = rho.unchecked<1>();
    auto zb = z.unchecked<1>();
    auto pb = zp.unchecked<1>();
    const py::ssize_t n = rb.shape(0);
    if (zb.shape(0) != n || pb.shape(0) != n)
        throw std::invalid_argument("rho, z and zp must have the same length");
    if (gx.ndim() != 1 || gw.ndim() != 1 || gx.shape(0) != gw.shape(0) ||
        gx.shape(0) < 1)
        throw std::invalid_argument("bad Gauss rule");
    const double *gxp = gx.data();
    const double *gwp = gw.data();
    const int ng = static_cast<int>(gx.shape(0));

    // The Python walk's domain raises, verbatim and BEFORE the parallel
    // region (a throw cannot cross an omp boundary).
    for (py::ssize_t i = 0; i < n; ++i) {
        if (!(zb(i) >= 0.0 && pb(i) <= 0.0))
            throw std::invalid_argument("need z >= 0 >= zp");
        if (rb(i) < 0.0 || (zb(i) - pb(i)) + rb(i) <= 0.0)
            throw std::invalid_argument("need R > 0");
    }

    py::array_t<std::complex<double>> vals({n, py::ssize_t(6)});
    auto vb = vals.mutable_unchecked<2>();
    const mw_contour::cd km(k_m);
    bool ray_failed = false;

    {
        py::gil_scoped_release release;
        // `dynamic`: a corner point costs many more adaptive levels than a
        // mid-range pair, and a fill's unique triples arrive sorted by
        // geometry — static scheduling would hand one thread the hard band.
        #pragma omp parallel for schedule(dynamic)
        for (py::ssize_t i = 0; i < n; ++i) {
            mw_contour::cd out[6];
            if (!mw680::six_point_one(k_p, km, rb(i), zb(i), pb(i), rtol,
                                      lam_mult, depth, detour, gxp, gwp, ng,
                                      out)) {
                #pragma omp atomic write
                ray_failed = true;
                continue;
            }
            for (int c = 0; c < 6; ++c) vb(i, c) = out[c];
        }
    }
    if (ray_failed)
        throw std::runtime_error(
            "rotated tail did not go quiet inside the panel budget");
    return vals;
}

// `six_columns` over CONCATENATED columns: column c owns the members
// offsets[c] .. offsets[c+1] of (z, zp), all at rho[c]. One call for a whole
// fill's grouping, because the parallel unit is the column and a per-column
// call would hand OpenMP one column at a time — which is precisely the
// scaling the numpy route does not have (#898). Returns the (n, 6) table in
// the members' own order, so the caller scatters with the same offsets.
static py::array_t<std::complex<double>> near_interface_six_columns(
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
    // `p` is the resolution ladder, not a size knob: 2^p panels per seeded
    // interval. Bounded so a typo asks for a refusal rather than a terabyte.
    if (p < 0 || p > 20) throw std::invalid_argument("column p out of range");
    const double *gxp = gx.data();
    const double *gwp = gw.data();
    const int ng = static_cast<int>(gx.shape(0));

    // The Python walk's domain raises, BEFORE the parallel region (a throw
    // cannot cross an omp boundary) and before any column is built — a bad
    // member anywhere refuses the whole call, as `six_columns` does.
    for (py::ssize_t c = 0; c < nc; ++c) {
        for (py::ssize_t i = ob(c); i < ob(c + 1); ++i) {
            if (!(zb(i) >= 0.0 && pb(i) <= 0.0))
                throw std::invalid_argument("need z >= 0 >= zp");
            if (rb(c) < 0.0 || (zb(i) - pb(i)) + rb(c) <= 0.0)
                throw std::invalid_argument("need R > 0");
        }
    }

    py::array_t<std::complex<double>> vals({n, py::ssize_t(6)});
    auto vb = vals.mutable_unchecked<2>();
    const mw_contour::cd kpc(k_p, 0.0);
    const mw_contour::cd km(k_m);

    // The thread count is the CALLER's policy, held to what OpenMP would
    // have used anyway so a pin below it survives: `_near_interface` passes
    // the PHYSICAL core count, for the reason #898 measured on the numpy
    // route's gemm and this kernel repeats — libmvec exp/sincos saturates a
    // core's FPU, so the hyperthread siblings contend rather than add. On
    // this 4c/8t box the BLE N = 4 kernel reads 95 ms at 1 thread, 40 ms at
    // 4 and 46 ms at 8.
    int nt = 1;
#ifdef _OPENMP
    nt = omp_get_max_threads();
    if (n_threads > 0) nt = std::min(nt, n_threads);
    nt = std::max(nt, 1);
#endif

    // TWO parallel units, because a fill's columns differ by three orders in
    // member count and one unit cannot serve both. Measured on BLE 45 ft
    // N = 4 (the #899 census): 84 columns over four `designed_tables` calls,
    // and the largest call is one column of 3,024 members beside forty of
    // 108. Parallelising over columns alone left that one column — 40 % of
    // the call's members — on a single thread, and the kernel measured 1.0x
    // from one thread to four.
    //
    // The split is a FAIR SHARE, not a fixed size: a column carrying more
    // members than one thread's share of the whole call cannot be a unit,
    // because no schedule can divide it. Everything else stays a column,
    // which matters in the other direction — those forty 108-member columns
    // are better parallelised as columns than as forty serial rule builds.
    // At one thread the share is the whole call and nothing splits, which is
    // the right answer there too.
    //
    //   * by_column: `dynamic`, since K grows with rho (one panel per J0
    //     oscillation) and a fill's columns arrive in geometry order;
    //   * by_member: the rule built once in front of the loop, serial, then
    //     every member of it on all threads.
    const py::ssize_t share = (n + nt - 1) / nt;
    std::vector<py::ssize_t> by_column, by_member;
    for (py::ssize_t c = 0; c < nc; ++c) {
        if (ob(c + 1) == ob(c)) continue;
        (ob(c + 1) - ob(c) > share ? by_member : by_column).push_back(c);
    }
    const py::ssize_t n_col = static_cast<py::ssize_t>(by_column.size());
    const py::ssize_t n_mem = static_cast<py::ssize_t>(by_member.size());

    {
        py::gil_scoped_release release;
        #pragma omp parallel for schedule(dynamic) num_threads(nt)
        for (py::ssize_t j = 0; j < n_col; ++j) {
            const py::ssize_t c = by_column[j];
            const py::ssize_t lo = ob(c), hi = ob(c + 1);
            mw899::Scratch s;
            mw899::column_rule(rb(c), k_p, km, mw899::s_min_of(zb, pb, lo, hi),
                               lam_mult, p, detour, gxp, gwp, ng, s);
            mw899::column_factors(kpc, km, s);
            const size_t K = s.lam.size();
            for (py::ssize_t i = lo; i < hi; ++i) {
                mw_contour::cd out[6];
                mw899::column_member(s, K, zb(i), pb(i), out);
                for (int q = 0; q < 6; ++q) vb(i, q) = out[q];
            }
        }
        for (py::ssize_t j = 0; j < n_mem; ++j) {
            const py::ssize_t c = by_member[j];
            const py::ssize_t lo = ob(c), hi = ob(c + 1);
            mw899::Scratch s;
            mw899::column_rule(rb(c), k_p, km, mw899::s_min_of(zb, pb, lo, hi),
                               lam_mult, p, detour, gxp, gwp, ng, s);
            mw899::column_factors(kpc, km, s);
            const size_t K = s.lam.size();
            // `dynamic`: members of one column do NOT cost the same. The
            // rule is converged for the column's smallest s, so a member
            // with a larger s underflows part of its own tail and skips it
            // (see MW_EXP_ZERO) — the exact saving the numpy route cannot
            // take, and it makes the cheap members the deep ones.
            #pragma omp parallel for schedule(dynamic, 32) num_threads(nt)
            for (py::ssize_t i = lo; i < hi; ++i) {
                mw_contour::cd out[6];
                mw899::column_member(s, K, zb(i), pb(i), out);
                for (int q = 0; q < 6; ++q) vb(i, q) = out[q];
            }
        }
    }
    return vals;
}

PYBIND11_MODULE(_near_interface_accel, m) {
    m.doc() =
        "momwire#680 U2 and #899 item 1: the C++ twins of "
        "_near_interface.six_point and _near_interface.six_columns. "
        "Optional; _near_interface falls back to the numpy walks without it.";
    // The capability flag `_near_interface._HAVE_NEAR_INTERFACE_ACCEL`
    // keys on. Its OWN name (not contour_engine_568 / below_fills_568): a
    // .so built at an earlier arc exports those but not this entry, and a
    // shared flag would claim a contract it cannot serve.
    m.attr("near_interface_680") = true;
    // The column twin's flag, and for the same reason its OWN name: a .so
    // built between #680 and #899 exports `near_interface_680` and not this
    // one, so the two entries have to be asked about separately.
    m.attr("near_interface_columns_899") = true;
    m.def("near_interface_six_batch", &near_interface_six_batch,
          py::arg("k_p"), py::arg("k_m"), py::arg("rho"), py::arg("z"),
          py::arg("zp"), py::arg("rtol"), py::arg("lam_mult"),
          py::arg("depth"), py::arg("detour"), py::arg("gx"), py::arg("gw"),
          "six_point over parallel (rho, z, zp) arrays -> (n, 6) complex. "
          "k_p/k_m arrive derived (k_medium keeps the branch choice); the "
          "U1 memo layer in Python hands this UNIQUE triples only.");
    m.def("near_interface_six_columns", &near_interface_six_columns,
          py::arg("k_p"), py::arg("k_m"), py::arg("rho"), py::arg("offsets"),
          py::arg("z"), py::arg("zp"), py::arg("lam_mult"), py::arg("p"),
          py::arg("detour"), py::arg("n_threads"), py::arg("gx"),
          py::arg("gw"),
          "six_columns over CONCATENATED columns -> (n, 6) complex. Column c "
          "owns members offsets[c]..offsets[c+1], all at rho[c]; the answer "
          "keeps the members' order. Parallel over columns, and over the "
          "MEMBERS of any column too big to be a unit, so a whole fill's "
          "grouping belongs in ONE call. `n_threads` <= 0 takes OpenMP's own "
          "count and is never raised above it. No rtol: the fixed rule's "
          "resolution is `p` and its extents the e^{-60} dead range.");
}
