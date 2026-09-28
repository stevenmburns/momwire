// The near-interface column machinery, shared by the six-kernel twin
// (`_near_interface_accel.cpp`) and the point-observer twin
// (`_near_interface_point_accel.cpp`, momwire#1224).
//
// MOVED VERBATIM from `_near_interface_accel.cpp` (namespaces mw680 and
// mw899), and included there at the very spot it was cut from, so that
// translation unit's preprocessed text -- and so its code -- is what it was
// (the #1193/#1194 codegen-drift rule; the gate is the six kernels
// bit-identical before and after). Edit it as the two TUs' shared source:
// every includer rebuilds, and both are listed in setup.py's _NEAR_HEADERS.
//
// Not self-contained: the includer supplies the pybind11 / <complex> /
// contour-engine / branch-cut / xsf includes and the `py` alias first.
#pragma once

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

// Accumulator lanes of `column_member`'s ordered sum (momwire#1193). The
// lane combine there is written out for exactly four, so changing this is a
// deliberate one-time move of every near-interface table, not a tuning knob.
//
// Under GCC/clang the lanes are a vector TYPE, not an array the vectorizer
// may or may not map to a register: `a + b` on it is four independent IEEE
// operations, so the order is fixed by the type and the speed does not
// depend on the vectorizer's reading of the loop. Measured on Haswell
// (GCC 11), a `double[4]` with the lane loop left to the vectorizer ran
// this kernel 11-18 % slower than the `reduction(+)` it replaces; the
// vector type ran it at parity (-1 % at four threads, +2 % on one). MSVC
// has no vector extension and gets the array, with the identical per-lane
// expression.
static const int MW_LANES = 4;
#if defined(__GNUC__)
typedef double mw_lanes __attribute__((vector_size(MW_LANES * sizeof(double))));
#else
typedef double mw_lanes[MW_LANES];
#endif

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
        // Fused complex products (momwire#1214, _fma_inline.h): the rule
        // build is most of this kernel's time (a column's members share it),
        // and GCC contracted every one of these products before #1194. Each
        // `mul` keeps the operands in the order written; the divisions stay
        // library calls, as they always were.
        using mw_fma::mul;
        const cd u = 2.0 * l / (g_p + g_m);
        const cd v = 2.0 * l / (mul(km2, g_p) + mul(kp2, g_m));
        const cd wv = mul(g_p - g_m, v);
        const cd f[6] = {mul(u, wk),
                         mul(v, wk),
                         mul(wv, wk),
                         mul(mul(-g_p, wv), wk),
                         mul(mul(-mul(g_p, g_m), v), wk),
                         mul(mul(g_m, wv), wk)};
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
        // Explicit fused multiply-adds (momwire#1214, _fma_inline.h), here and
        // in the six sums below: the sites GCC contracted before
        // -ffp-contract=off (#1194), in the split it chose (the first product
        // fused, the second rounded). The lane sums themselves stay adds, in
        // the order the lanes fix.
        for (int j = 0; j < nb; ++j) {
            ar[j] = mw_fma::fma(gmr[k0 + j], zp, -(gpr[k0 + j] * z));
            amax = std::max(amax, ar[j]);
        }
        if (amax < MW_EXP_ZERO) continue;  // exactly zero, see MW_EXP_ZERO
        MW_NI_SIMD()
        for (int j = 0; j < nb; ++j) {
            ai[j] = mw_fma::fma(gmi[k0 + j], zp, -(gpi[k0 + j] * z));
            const double ex = exp(ar[j]);
            er[j] = ex * cos(ai[j]);
            ei[j] = ex * sin(ai[j]);
        }
        // The block's six sums, in an order the SOURCE fixes (momwire#1193,
        // the #781 rule): node j adds into lane j % MW_LANES of its column,
        // sequentially within the lane, and the lanes combine pairwise as
        // written after the loop. No `omp simd reduction(+)`: that clause
        // licenses reassociation, so the order would be the compiler's
        // vectorization choice, and an unrelated edit to this TU or a
        // compiler upgrade could move every table at the ulp level. (MSVC
        // builds with /fp:fast, which licenses reassociation TU-wide; the
        // order here is fixed wherever the compiler keeps IEEE semantics.)
        //
        // Lanes rather than one serial accumulator per column, for speed:
        // the serial spelling, even with the six columns interleaved for
        // independent chains, cost +6 % on this kernel and +3 % on razor's
        // hub16 x16 fill (Haswell); the lanes cost nothing measurable
        // (see mw_lanes for why they are a vector type).
        static_assert(MW_LANES == 4, "the lane combine is written for four");
        for (int c = 0; c < 6; ++c) {
            const double *fr = s.fr.data() + c * K + k0;
            const double *fi = s.fi.data() + c * K + k0;
            mw_lanes sr = {0.0, 0.0, 0.0, 0.0}, si = {0.0, 0.0, 0.0, 0.0};
            int j = 0;
            for (; j + MW_LANES <= nb; j += MW_LANES) {
#if defined(__GNUC__)
                mw_lanes a, b, x, y;  // memcpy: unaligned loads, no aliasing
                std::memcpy(&a, er + j, sizeof a);
                std::memcpy(&b, ei + j, sizeof b);
                std::memcpy(&x, fr + j, sizeof x);
                std::memcpy(&y, fi + j, sizeof y);
                // Lane by lane through the scalar mw_fma::fma, which GCC
                // packs back into one vfmadd per sum under -mfma.
                const mw_lanes by = b * y, bx = b * x;
                mw_lanes p, q;
                for (int l = 0; l < MW_LANES; ++l) {
                    p[l] = mw_fma::fma(a[l], x[l], -by[l]);
                    q[l] = mw_fma::fma(a[l], y[l], bx[l]);
                }
                sr += p;
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
    for (int c = 0; c < 6; ++c) out[c] = cd(accr[c], acci[c]);
}
}  // namespace mw899
