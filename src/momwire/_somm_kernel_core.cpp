// The Sommerfeld six-integral kernel, Python-free (momwire#1479's Sommerfeld
// stage S2): the contours, the 24-point Gauss rule, the adaptive panels and
// the Bessel / Hankel evaluations `_sommerfeld._six_integrals` describes,
// moved verbatim out of `_accel_somm.cpp` (its binding) so the standalone
// executable links it with no Python. It includes `_mw_core.h` for the same
// TU-wide `omp declare simd` declarations the binding TU sees, so the move
// cannot change which `exp` / `sin` / `cos` a loop calls (momwire#1194).
#include "_somm_kernel_core.h"

#include "_mw_core.h"
#include "_branch_cut_inline.h"
#include "_cdiv_inline.h"
#include "_fma_inline.h"
#include "_somm_six_lanes_inline.h"

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstring>

namespace somm {

using cd = std::complex<double>;
static const cd CI(0.0, 1.0);
static const double SPI = 3.14159265358979323846;
static const double EULER_GAMMA = 0.57721566490153286061;
static const int ADAPT_DEPTH = 14;  // = _sommerfeld._ADAPT_DEPTH

// 24-point Gauss-Legendre rule — identical to _sommerfeld._GX/_GW.
static const double GX[24] = {
    -9.95187219997021311e-01, -9.74728555971309474e-01, -9.38274552002732798e-01,
    -8.86415527004401071e-01, -8.20001985973902947e-01, -7.40124191578554358e-01,
    -6.48093651936975546e-01, -5.45421471388839563e-01, -4.33793507626045127e-01,
    -3.15042679696163397e-01, -1.91118867473616311e-01, -6.40568928626056300e-02,
    6.40568928626056300e-02, 1.91118867473616311e-01, 3.15042679696163397e-01,
    4.33793507626045127e-01, 5.45421471388839563e-01, 6.48093651936975546e-01,
    7.40124191578554358e-01, 8.20001985973902947e-01, 8.86415527004401071e-01,
    9.38274552002732798e-01, 9.74728555971309474e-01, 9.95187219997021311e-01,
};
static const double GW[24] = {
    1.23412297999886903e-02, 2.85313886289335593e-02, 4.42774388174194122e-02,
    5.92985849154363601e-02, 7.33464814110801611e-02, 8.61901615319532050e-02,
    9.76186521041139260e-02, 1.07444270115965565e-01, 1.15505668053725516e-01,
    1.21670472927803294e-01, 1.25837456346828247e-01, 1.27938195346752021e-01,
    1.27938195346752021e-01, 1.25837456346828247e-01, 1.21670472927803294e-01,
    1.15505668053725516e-01, 1.07444270115965565e-01, 9.76186521041139260e-02,
    8.61901615319532050e-02, 7.33464814110801611e-02, 5.92985849154363601e-02,
    4.42774388174194122e-02, 2.85313886289335593e-02, 1.23412297999886903e-02,
};

// ---- complex-argument Bessel/Hankel, orders 0 and 1 ----------------------
//
// Domain (measured from real contour fills): |x| up to ~110, arg(x) in
// [-100 deg, +45 deg] — never near the negative-real-axis branch cut, so
// principal-branch log/sqrt are safe throughout. Ascending series for
// |x| <= 12 (~3 digits of cancellation, ~1e-12 abs), A&S 9.2 asymptotic
// expansions with optimal truncation beyond (~1e-11 at the switch, far
// better further out). Both are validated pointwise against scipy over the
// sampled domain in tests/test_sommerfeld_accel.py.

// These routines are the measured hot spot of the whole grid fill
// (sommerfeld-perf-plan Phase 8): ~85% of the per-contour-point cost, and
// the fill is ~95% of a cold Sommerfeld solve. Three shares are exploited
// below, none of which changes the mathematics:
//
//   * every truncation test is "is this term small next to the running
//     sum", a predicate identical under squaring — so |.| (a libm hypot,
//     ~75 of them per small-|x| Hankel call) becomes three flops;
//   * the Y series ride the same two ladders as the J series, so one pass
//     yields all four sums instead of two passes rebuilding the ladders;
//   * in the asymptotic regime the two orders share the (i/z) ladder and
//     the sqrt/exp prefactor, and the kind-1 terms are the kind-2 terms
//     with alternating signs.

// |z|^2 — the convergence-test currency (see above).
static inline double cnorm(cd z) {
    return mw_fma::norm2(z);  // fused (momwire#1194)
}
static const double SER_EPS2 = 1e-34;  // (1e-17)^2

// J0 and J1/z ascending series (A&S 9.1.10/9.1.12): safe as z -> 0.
static void j01_series(cd z, cd &j0, cd &j1x) {
    const cd q = mw_fma::mul(-0.25 * z, z);  // fused (momwire#1194)
    cd t0(1.0, 0.0), t1(0.5, 0.0);
    j0 = t0;
    j1x = t1;
    for (int k = 1; k <= 60; ++k) {
        // Fused ladder products (momwire#1194, _fma_inline.h).
        t0 = mw_fma::mul(t0, q / double(k * k));
        t1 = mw_fma::mul(t1, q / double(k * (k + 1)));
        j0 += t0;
        j1x += t1;
        if (cnorm(t0) <= SER_EPS2 * cnorm(j0) &&
            cnorm(t1) <= SER_EPS2 * cnorm(j1x))
            break;
    }
}

// J0, J1/z, Y0 and Y1 from ONE ascending-series pass (A&S 9.1.10-9.1.13,
// principal log).
//
// With q = -(z/2)^2 the J ladders are t0_k = q^k/(k!)^2 and t1_k =
// (1/2) q^k/(k!(k+1)!), i.e. t0_k = (-1)^k (z/2)^{2k}/(k!)^2 and 2 t1_k =
// (-1)^k (z/2)^{2k}/(k!(k+1)!). The manual's alternating Y sums are then
// exactly those ladders reweighted by harmonic numbers:
//   sum_{k>=1} (-1)^{k+1} H_k (z/2)^{2k}/(k!)^2        = -sum_k H_k t0_k
//   sum_{k>=0} (-1)^k (H_k+H_{k+1}) (z/2)^{2k}/(k!(k+1)!)
//                                            = 2 sum_k (H_k+H_{k+1}) t1_k
// so the four sums cost two complex multiplies per term, not four.
static void jy01_series(cd z, cd &j0, cd &j1x, cd &y0, cd &y1) {
    const cd q = mw_fma::mul(-0.25 * z, z);  // fused (momwire#1194)
    cd t0(1.0, 0.0), t1(0.5, 0.0);
    j0 = t0;
    j1x = t1;
    cd s0(0.0, 0.0), s1 = 2.0 * t1;  // k = 0: H_0 = 0, H_0 + H_1 = 1
    double hk = 0.0, hk1 = 1.0;
    for (int k = 1; k <= 60; ++k) {
        // Fused ladder products (momwire#1194, _fma_inline.h).
        t0 = mw_fma::mul(t0, q / double(k * k));
        t1 = mw_fma::mul(t1, q / double(k * (k + 1)));
        hk += 1.0 / double(k);
        hk1 += 1.0 / double(k + 1);
        j0 += t0;
        j1x += t1;
        const cd term0 = hk * t0;
        const cd term1 = (2.0 * (hk + hk1)) * t1;
        s0 -= term0;
        s1 += term1;
        if (cnorm(t0) <= SER_EPS2 * cnorm(j0) &&
            cnorm(t1) <= SER_EPS2 * cnorm(j1x) &&
            cnorm(term0) <= SER_EPS2 * (cnorm(s0) + 1.0) &&
            cnorm(term1) <= SER_EPS2 * (cnorm(s1) + 1.0))
            break;
    }
    const cd lg = std::log(0.5 * z) + EULER_GAMMA;
    y0 = (2.0 / SPI) * mw_fma::mul_add(lg, j0, s0);
    y1 = (2.0 / SPI) * (mw_fma::mul(lg, mw_fma::mul(j1x, z)) - mw_cdiv::div(1.0, z)) -
         mw_fma::mul(z / (2.0 * SPI), s1);
}

// The A&S 9.2 asymptotic sums for orders 0 and 1 at once, each with its own
// optimal truncation:
//   H^(kind)_nu(z) ~ sqrt(2/(pi z)) e^{s i (z - nu pi/2 - pi/4)} sum_k t^nu_k
//   t^nu_0 = 1,  t^nu_{k+1} = t^nu_k (4 nu^2 - (2k+1)^2)/(8(k+1)) (s i/z)
// with s = +1 for kind 1, -1 for kind 2. Valid away from the negative real
// axis, which the contours never approach.
//
// The orders differ only in the numerator 4 nu^2 - (2k+1)^2, so one loop
// over the shared (i/z) ladder yields both; and t^nu_k carries (s i/z)^k,
// so the kind-1 terms are the kind-2 terms with alternating signs — no
// second pass. `want_plus` skips the kind-1 accumulation for the Hankel
// form, which never needs it. Each order keeps its own divergence-onset
// break, so both truncate exactly where a per-order loop would.
static void hankel_asym_sums(cd z, bool want_plus, cd &s0m, cd &s1m, cd &s0p,
                             cd &s1p) {
    const cd iz = mw_cdiv::div(-CI, z);  // s = -1
    cd t0(1.0, 0.0), t1(1.0, 0.0);
    s0m = s1m = s0p = s1p = cd(1.0, 0.0);
    double prev0 = 1e300, prev1 = 1e300;
    bool live0 = true, live1 = true;
    for (int k = 0; k < 40 && (live0 || live1); ++k) {
        const double odd = double(2 * k + 1);
        const double odd2 = odd * odd;
        const double den = 8.0 * double(k + 1);
        const double sgn = (k & 1) ? 1.0 : -1.0;  // (-1)^{k+1}: term k+1
        if (live0) {
            t0 = mw_fma::mul(t0, (-odd2 / den) * iz);  // fused (#1194)
            const double a = cnorm(t0);
            if (a >= prev0) {
                live0 = false;  // divergence onset: stop at the optimum
            } else {
                s0m += t0;
                if (want_plus) s0p = mw_fma::mul_add(t0, sgn, s0p);
                prev0 = a;
                if (a <= SER_EPS2 * cnorm(s0m)) live0 = false;
            }
        }
        if (live1) {
            t1 = mw_fma::mul(t1, ((4.0 - odd2) / den) * iz);
            const double a = cnorm(t1);
            if (a >= prev1) {
                live1 = false;
            } else {
                s1m += t1;
                if (want_plus) s1p = mw_fma::mul_add(t1, sgn, s1p);
                prev1 = a;
                if (a <= SER_EPS2 * cnorm(s1m)) live1 = false;
            }
        }
    }
}

static const double BESSEL_SWITCH2 = 144.0;  // (12.0)^2

// (J0(x), J1(x)/x) — the Bessel-form pair of _bessel_j0_j1x.
static void bessel_j0_j1x(cd x, cd &b0, cd &b1x) {
    if (cnorm(x) <= BESSEL_SWITCH2) {
        j01_series(x, b0, b1x);
        return;
    }
    // J_nu = (H1_nu + H2_nu)/2. With W = sqrt(2/(pi x)) and E = e^{-i(x -
    // pi/4)}: H2_0 = W E S0m, H2_1 = i W E S1m, H1_0 = (W/E) S0p and
    // H1_1 = -i (W/E) S1p — one sqrt and one exp for all four, where the
    // per-order calls did four of each.
    cd s0m, s1m, s0p, s1p;
    hankel_asym_sums(x, true, s0m, s1m, s0p, s1p);
    const cd w = std::sqrt(mw_cdiv::div(2.0, SPI * x));
    const cd e = std::exp(mw_fma::mul(-CI, x - 0.25 * SPI));
    const cd p = mw_fma::mul(w, e), pinv = mw_cdiv::div(w, e);
    b0 = 0.5 * mw_fma::mul_add(pinv, s0p, mw_fma::mul(p, s0m));
    b1x = mw_cdiv::div(
        mw_fma::mul(0.5 * CI, mw_fma::mul(p, s1m) - mw_fma::mul(pinv, s1p)), x);
}

// (H2_0(x)/2, H2_1(x)/(2x)) — the Hankel-form pair of _integrand_six.
static void hankel2_half(cd x, cd &b0, cd &b1x) {
    if (cnorm(x) <= BESSEL_SWITCH2) {
        cd j0, j1x, y0, y1;
        jy01_series(x, j0, j1x, y0, y1);
        b0 = 0.5 * (j0 - mw_fma::mul(CI, y0));
        b1x = mw_cdiv::div(0.5 * (mw_fma::mul(j1x, x) - mw_fma::mul(CI, y1)), x);
        return;
    }
    cd s0m, s1m, s0p, s1p;
    hankel_asym_sums(x, false, s0m, s1m, s0p, s1p);
    const cd p = mw_fma::mul(std::sqrt(mw_cdiv::div(2.0, SPI * x)),
                             std::exp(mw_fma::mul(-CI, x - 0.25 * SPI)));
    b0 = mw_fma::mul(0.5 * p, s0m);
    b1x = mw_cdiv::div(mw_fma::mul(mw_fma::mul(0.5 * CI, p), s1m), x);
}

// ---- the six integrands and quadrature (ports of the Python names) -------

// gamma(lam, k) = sqrt(-j(lam-k)) sqrt(j(lam+k)): NEC's vertical cuts.
// One definition for all three call sites, across both extensions (#714).
using mw_branch::gamma_cut;

struct Six {
    cd v[6];
    Six() : v{} {}
    Six &operator+=(const Six &o) {
        for (int i = 0; i < 6; ++i) v[i] += o.v[i];
        return *this;
    }
    Six &operator-=(const Six &o) {
        for (int i = 0; i < 6; ++i) v[i] -= o.v[i];
        return *this;
    }
};
static inline Six operator+(Six a, const Six &b) { return a += b; }

static inline double absmax(const Six &s) {
    double m = 0.0;
    for (int i = 0; i < 6; ++i) m = std::max(m, std::abs(s.v[i]));
    return m;
}
static inline double absmax_diff(const Six &a, const Six &b) {
    double m = 0.0;
    for (int i = 0; i < 6; ++i) m = std::max(m, std::abs(a.v[i] - b.v[i]));
    return m;
}

struct SommCtx {
    double rho, h;
    cd k1, k2;
    bool bessel;
    bool lanes;  // the rule's nodes four at a time (momwire#1441 step 3)
};

// The six lambda-integrands of NEC eqs 148-153 (= _integrand_six).
static inline void integrand_six(const SommCtx &c, cd lam, cd out[6]) {
    const cd g1 = gamma_cut(lam, c.k1);
    const cd g2 = gamma_cut(lam, c.k2);
    // Every complex product here is the fused mw_fma::mul (momwire#1194):
    // this integrand is most of the Sommerfeld grid fill, which lost 10-12 %
    // on Haswell when the build stopped contracting. Same factors, same
    // left-to-right order as the operator* spelling it replaces. It takes
    // ALL of them -- these, the branch cut's and the Bessel ladders' -- to
    // get back to parity: fusing only the ladders and the accumulation left
    // the fill 8-12 % behind.
    using mw_fma::mul;
    const cd k1s = mul(c.k1, c.k1);
    const cd k2s = mul(c.k2, c.k2);
    const cd g2ks = mul(g2, k1s + k2s);
    // The four divides are mw_cdiv::div, libgcc's own rounding without the
    // call (momwire#1441).
    const cd d1 = mw_cdiv::div(2.0, g1 + g2) - mw_cdiv::div(2.0 * k2s, g2ks);
    const cd d2 = mw_cdiv::div(2.0, mw_fma::mul_add(k1s, g2, mul(k2s, g1))) -
                  mw_cdiv::div(2.0, g2ks);
    const cd e = std::exp(-g2 * c.h);
    const cd x = lam * c.rho;
    cd b0, b1x;
    if (c.bessel)
        bessel_j0_j1x(x, b0, b1x);
    else
        hankel2_half(x, b0, b1x);
    const cd l2 = mul(lam, lam);
    const cd l3 = mul(l2, lam);
    const cd common = mul(d2, e);
    const cd cg2 = mul(common, g2);
    out[0] = mul(mul(common, b1x - b0), l3);
    out[1] = mul(mul(mul(cg2, g2), b0), lam);
    out[2] = mul(mul(cg2, mul(b1x, x)), l2);
    out[3] = mul(mul(-common, b1x), l3);
    out[4] = mul(mul(common, b0), lam);
    out[5] = mul(mul(mul(d1, e), b0), lam);
}

static Six gauss_segment(const SommCtx &c, cd z0, cd z1) {
    const cd mid = 0.5 * (z0 + z1);
    const cd half = 0.5 * (z1 - z0);
    Six acc;
    if (c.lanes) {
        // The 24 integrand values four nodes at a time, each lane the walk
        // below bit for bit (_somm_six_lanes_inline.h), then the same
        // weighted sum in the same node order.
        cd fv[24 * 6];
        somm_six_lanes::gauss_values<24>(c.rho, c.h, c.k1, c.k2, c.bessel, mid,
                                         half, GX, fv);
        for (int q = 0; q < 24; ++q) {
            const cd w = GW[q] * half;
            for (int i = 0; i < 6; ++i)
                acc.v[i] = mw_fma::mul_add(fv[q * 6 + i], w, acc.v[i]);
        }
        return acc;
    }
    cd f[6];
    for (int q = 0; q < 24; ++q) {
        integrand_six(c, mw_fma::mul_add(half, GX[q], mid), f);
        const cd w = GW[q] * half;
        // Fused accumulation (momwire#1194, _fma_inline.h).
        for (int i = 0; i < 6; ++i) acc.v[i] = mw_fma::mul_add(f[i], w, acc.v[i]);
    }
    return acc;
}

// Recursive bisection Gauss quadrature, relative tolerance
// (= _adaptive_segment; see its docstring for why relative).
static Six adaptive_segment(const SommCtx &c, cd z0, cd z1, double rtol,
                            int depth, const Six *whole_in) {
    Six whole = whole_in ? *whole_in : gauss_segment(c, z0, z1);
    const cd mid = 0.5 * (z0 + z1);
    Six left = gauss_segment(c, z0, mid);
    Six right = gauss_segment(c, mid, z1);
    Six better = left + right;
    const double err = absmax_diff(better, whole);
    const double scale = absmax(better);
    if (depth <= 0 || err <= rtol * std::max(scale, 1e-300)) return better;
    return adaptive_segment(c, z0, mid, rtol, depth - 1, &left) +
           adaptive_segment(c, mid, z1, rtol, depth - 1, &right);
}

// Panel tail with geometric ramp from panel0 (<= 0 means "none") — see
// _tail's docstring for the ramp rationale and the `== 0.0` quiet trigger.
static Six tail(const SommCtx &c, cd z0, cd direction, double panel,
                double rtol, double ref_scale, double panel0) {
    const int max_panels = 800;
    Six total;
    int quiet = 0;
    cd z = z0;
    double step = (panel0 > 0.0) ? std::min(panel0, panel) : panel;
    for (int i = 0; i < max_panels; ++i) {
        const cd z_next = z + step * direction;
        Six contrib = gauss_segment(c, z, z_next);
        total += contrib;
        z = z_next;
        step = std::min(2.0 * step, panel);
        const double scale = std::max(absmax(total), ref_scale);
        const double cmax = absmax(contrib);
        if (cmax == 0.0 || cmax < rtol * scale) {
            if (++quiet >= 2) break;
        } else {
            quiet = 0;
        }
    }
    return total;
}

// = _six_integrals for one (rho, h); form: 0 auto (rho < 2h -> Bessel),
// 1 force Bessel, 2 force Hankel. eps_t == 1 is short-circuited by the
// caller (and again here for safety).
static void six_integrals(cd eps_t, double k2d, double rho, double h,
                          double rtol, int form, cd out[6], bool lanes = true) {
    for (int i = 0; i < 6; ++i) out[i] = cd(0.0, 0.0);
    if (eps_t == cd(1.0, 0.0)) return;
    const cd k2(k2d, 0.0);
    cd k1 = k2d * std::sqrt(eps_t);
    if (k1.imag() > 0) k1 = std::conj(k1);
    const double scale = std::max(rho, h);
    const double panel = 0.2 * SPI / scale;
    const double kmax = std::max(std::abs(k1), k2d);
    const double kcap = 1.2 * k2d + 50.0 / scale;
    const double kmax_eff = std::min(kmax, kcap);
    const double qtol = std::min(rtol, 1e-11);
    const bool use_bessel = (form == 0) ? (rho < 2.0 * h) : (form == 1);
    SommCtx c{rho, h, k1, k2, use_bessel, lanes};
    Six total;
    if (use_bessel) {
        const double p = std::min(rho > 0.0 ? 1.0 / rho : 1e300, 1.0 / h);
        const cd brk(p, p);
        const cd end_adapt(1.3 * kmax_eff + 3.0 * p, p);
        total = adaptive_segment(c, cd(0.0, 0.0), brk, qtol, ADAPT_DEPTH, nullptr);
        cd tail_start = brk;
        if (end_adapt.real() > brk.real()) {
            total += adaptive_segment(c, brk, end_adapt, qtol, ADAPT_DEPTH, nullptr);
            tail_start = end_adapt;
        }
        total += tail(c, tail_start, cd(1.0, 0.0), panel, rtol, absmax(total), -1.0);
    } else {
        const double r1 = std::hypot(rho, h);
        const cd dir_right = cd(c.h, -c.rho) / r1;
        const cd dir_left = cd(-c.h, -c.rho) / r1;
        const cd a(0.0, -0.4 * k2d);
        const cd b = cd(0.6, 0.2) * k2d;
        const cd cc = cd(1.02, 0.2) * k2d;
        // Waypoint d must clear the k1 branch point: gamma_1's cut runs
        // straight DOWN from +k1, so a d left of k1.real starts the
        // descending tail on the far side of that cut and flips gamma_1's
        // sign over the live part of the contour (issue #161). `kcap` is
        // keyed to max(rho, h) and at grazing falls below k1, so cap only
        // once the branch point is numerically dead -- the a->d run
        // carries e^{-gamma_2 h} * H0(2)(lam*rho) ~ e^{-(k1r*h - k1i*rho)}
        // there. See _sommerfeld._six_integrals for the full rationale and
        // the |k1| > 200 k2 (PEC-limit) escape.
        const bool k1_dead = k1.real() * h - k1.imag() * rho >= 50.0;
        const double cap_d = (k1_dead || std::abs(k1) > 200.0 * k2d)
                                 ? kcap
                                 : std::max(kcap, 1.01 * k1.real());
        cd d;
        if (1.01 * k1.real() <= cap_d)
            d = cd(1.01 * k1.real(), 0.99 * std::max(k1.imag(), -cap_d));
        else
            d = cd(cap_d, 0.0);
        if (d.real() < 1.1 * k2d) d = cd(1.1 * k2d, d.imag());
        total = adaptive_segment(c, a, b, qtol, ADAPT_DEPTH, nullptr);
        total += adaptive_segment(c, b, cc, qtol, ADAPT_DEPTH, nullptr);
        total += adaptive_segment(c, cc, d, qtol, ADAPT_DEPTH, nullptr);
        const double ref = absmax(total);
        const double p0 = 0.5 * kmax;
        total += tail(c, d, dir_right, panel, rtol, ref, p0);
        total -= tail(c, a, dir_left, panel, rtol, ref, p0);
    }
    for (int i = 0; i < 6; ++i) out[i] = total.v[i];
}

}  // namespace somm

namespace somm_core {

void six_integrals(std::complex<double> eps_t, double k2, double rho, double h, double rtol,
                   int form, std::complex<double> out[6], bool lanes) {
    // The lanes run where this build has them (mw_sl::kActive: off under
    // MSVC and without a vector backend); `lanes=false` is the scalar walk.
    somm::six_integrals(eps_t, k2, rho, h, rtol, form, out, lanes && mw_sl::kActive);
}

}  // namespace somm_core
