#include "_accel_common.h"
#include "_branch_cut_inline.h"
#include "_fma_inline.h"

#include <cstring>
#ifdef _OPENMP
#include <omp.h>
#endif

// somm section of the former _accelerators.cpp monolith (momwire#687).
// Code below is byte-identical to the monolith's lines 5402-6272, with TWO
// exceptions: somm_proj (monolith 5846-5982) was hoisted verbatim into
// _accel_somm_proj_inline.h — shared with the mw568 TU — and is pulled back
// in by the mid-file #include below; and the local `gam` branch-cut helper
// became the shared `mw_branch::gamma_cut` (#714), renaming its two call
// sites in integrand_six.

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
    y1 = (2.0 / SPI) * (mw_fma::mul(lg, mw_fma::mul(j1x, z)) - 1.0 / z) -
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
    const cd iz = -CI / z;  // s = -1
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
    const cd w = std::sqrt(2.0 / (SPI * x));
    const cd e = std::exp(mw_fma::mul(-CI, x - 0.25 * SPI));
    const cd p = mw_fma::mul(w, e), pinv = w / e;
    b0 = 0.5 * mw_fma::mul_add(pinv, s0p, mw_fma::mul(p, s0m));
    b1x = mw_fma::mul(0.5 * CI, mw_fma::mul(p, s1m) - mw_fma::mul(pinv, s1p)) / x;
}

// (H2_0(x)/2, H2_1(x)/(2x)) — the Hankel-form pair of _integrand_six.
static void hankel2_half(cd x, cd &b0, cd &b1x) {
    if (cnorm(x) <= BESSEL_SWITCH2) {
        cd j0, j1x, y0, y1;
        jy01_series(x, j0, j1x, y0, y1);
        b0 = 0.5 * (j0 - mw_fma::mul(CI, y0));
        b1x = 0.5 * (mw_fma::mul(j1x, x) - mw_fma::mul(CI, y1)) / x;
        return;
    }
    cd s0m, s1m, s0p, s1p;
    hankel_asym_sums(x, false, s0m, s1m, s0p, s1p);
    const cd p = mw_fma::mul(std::sqrt(2.0 / (SPI * x)),
                             std::exp(mw_fma::mul(-CI, x - 0.25 * SPI)));
    b0 = mw_fma::mul(0.5 * p, s0m);
    b1x = mw_fma::mul(mw_fma::mul(0.5 * CI, p), s1m) / x;
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
    const cd d1 = 2.0 / (g1 + g2) - 2.0 * k2s / g2ks;
    const cd d2 = 2.0 / mw_fma::mul_add(k1s, g2, mul(k2s, g1)) - 2.0 / g2ks;
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
                          double rtol, int form, cd out[6]) {
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
    SommCtx c{rho, h, k1, k2, use_bessel};
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

// Batched entry point: the six NEC lambda-integrals at each (rho[i], h[i]),
// OpenMP across nodes (each node's adaptive quadrature is independent).
// Returns (n, 6) complex in _six_integrals order (Vrr, Vzz, Vrz, Vr1, V, U).
static py::array_t<std::complex<double>> somm_six_integrals_batch(
    std::complex<double> eps_t, double k2,
    py::array_t<double, py::array::c_style | py::array::forcecast> rho,
    py::array_t<double, py::array::c_style | py::array::forcecast> h,
    double rtol, int form, uintptr_t cancel_flag = 0) {
    auto rb = rho.unchecked<1>();
    auto hb = h.unchecked<1>();
    const py::ssize_t n = rb.shape(0);
    if (hb.shape(0) != n)
        throw std::invalid_argument("rho and h must have the same length");
    if (form < 0 || form > 2)
        throw std::invalid_argument("form must be 0 (auto), 1 (J) or 2 (H)");
    for (py::ssize_t i = 0; i < n; ++i) {
        if (rb(i) < 0.0 || hb(i) < 0.0 || (rb(i) == 0.0 && hb(i) == 0.0))
            throw std::invalid_argument(
                "need rho, h >= 0 and R1 > 0 at every node");
    }
    py::array_t<std::complex<double>> out({n, py::ssize_t(6)});
    auto ob = out.mutable_unchecked<2>();
    const somm::cd et(eps_t);
    MW_CANCEL_SETUP(cancel_flag);

    #pragma omp parallel for schedule(dynamic)
    for (py::ssize_t i = 0; i < n; ++i) {
        MW_CANCEL_POLL();
        somm::cd res[6];
        somm::six_integrals(et, k2, rb(i), hb(i), rtol, form, res);
        for (int j = 0; j < 6; ++j) ob(i, j) = res[j];
    }
    MW_THROW_IF_ABORTED();
    return out;
}

// ---------------------------------------------------------------------------
// Sommerfeld remainder assembly (sommerfeld-perf-plan Phase 4b).
//
// Fused C++ port of _sommerfeld.remainder_field_proj (which internally calls
// SommerfeldGrid.eval). Per (observer m, source n) pair: interpolate the four
// smooth-remainder surfaces from the tabulated grid with the same 4x4 Lagrange
// (bivariate cubic) stencil, combine per the theory-manual eqs 143-147 azimuth
// factors, and project onto the observer tangent -- with NO materialized
// (4,n,4,4) intermediate (the numpy bottleneck). One OpenMP loop over observer
// rows. Bit-for-bit the same arithmetic as the Python fallback; cross-checked
// in tests/test_sommerfeld_accel.py.
//
// Clean-room: ported from momwire's own _sommerfeld.py. No GPL Sommerfeld
// source (nec2c, nec2++/PyNEC, somnec) was consulted.
#include "_accel_somm_proj_inline.h"

// ---------------------------------------------------------------------------
// The remainder projection's blocked loop and its lanes (perf item 9; on the
// lane layer, `_lanes.h`, since momwire#1372).
//
// `proj_one` per pair is ~10 divisions (the stencil's two and lagrange4's
// eight), sixty-four complex-by-real products in the surface read, the
// projection's complex products, and three libm calls (hypot, atan2, the
// polar's sincos). SG above x16 spent 7.2 s of 21 s here on Haswell. The
// blocked loop takes a row's sources in blocks of `ABOVE_BLOCK` and runs
// `proj_one`'s stages over a block: the geometry and the libm calls per pair,
// scalar; the stencil W pairs to a vector; the surface read W/2 surfaces to
// a vector (re, im of each in a lane pair); the projection W pairs to a
// vector; and the momwire#1258 continuation, where a pair needs it, per pair
// on the lanes' surfaces. Every lane does the scalar stage's operations in its
// order — the same selects, correctly rounded divisions, and (the build being
// -ffp-contract=off, and these helpers using std::complex, not the `mw_fma`
// ones) separate multiplies and adds, never a fused one — and no reduction
// crosses lanes, so a lane's value is `proj_one`'s. A block's last nb mod W
// pairs take `proj_one` itself. `lanes=false` is the per-pair loop, the
// reference the lanes are gated against (tests/test_somm_proj_lanes_1290.py);
// the baseline, arm64 and MSVC builds have only it.
#if MW_LANES_SIMD && MW_LANES_PACKED
#define MW_SOMM_LANES 1
#else
#define MW_SOMM_LANES 0
#endif

namespace somm_lanes {
using somm_proj::cd;
constexpr int ABOVE_BLOCK = 64;

#if MW_SOMM_LANES
using namespace mw_lanes;

// `proj_core`'s stencil for pairs b .. b+W-1: theta and r1 clamped, the
// region, then the shared tail (the first stencil node clamped, lagrange4 at
// both offsets).
static inline void stencil_lanes(const somm_proj::GridView &G, const int *nR32,
                                 const int *nTh32, const double *r1,
                                 const double *th, int *reg_o, int *i0_o,
                                 int *j0_o, double (*wr)[ABOVE_BLOCK],
                                 double (*wt)[ABOVE_BLOCK], int b) {
    const vd t = loadu(th + b);
    vd rc = loadu(r1 + b);
    const vd c_r1max = set1(G.r1_max);
    // r1c = r1 > r1_max ? r1_max : r1
    rc = blend(rc, c_r1max, cmp_gt(rc, c_r1max));
    // reg = r1c <= r_break ? (t <= split ? 0 : 1)
    //     : (r1c <= r_near ? (t <= split ? 2 : 3) : (t <= split ? 4 : 5))
    const vm low = cmp_le(t, set1(G.th_split));
    const vd side = blend(set1(1.0), zero(), low);
    vd zone = blend(set1(4.0), set1(2.0), cmp_le(rc, set1(G.r_near)));
    zone = blend(zone, zero(), cmp_le(rc, set1(G.r_break)));
    somm_proj_lanes::stencil_tail<ABOVE_BLOCK>(G, nR32, nTh32, rc, t,
                                               cvtt_i32(add(zone, side)), reg_o,
                                               i0_o, j0_o, wr, wt, b);
}

// `proj_project` for pairs b .. b+W-1, std::complex's operators spelled per
// part (complex x complex as (ac - bd, ad + bc), complex x real
// componentwise); writes the W entries.
static inline void project_lanes(const somm_proj::GridView &G,
                                 const double (*sre)[ABOVE_BLOCK],
                                 const double (*sim)[ABOVE_BLOCK],
                                 const double *gre, const double *gim,
                                 const double *rho_, const double *dx_,
                                 const double *dy_, double tox, double toy,
                                 double toz, const double *ux, const double *uy,
                                 const double *thsrc, const double *tzsrc, int b,
                                 cd *out) {
    auto L = [&](const double *p) { return loadu(p + b); };
    const vd rho = L(rho_), dx = L(dx_), dy = L(dy_);
    const vd sux = loadu(ux), suy = loadu(uy);
    const vd sth = loadu(thsrc), stz = loadu(tzsrc);
    const vd gr = L(gre), gi = L(gim);
    const vm safe = cmp_gt(rho, set1(G.tiny));
    const vd inv = blend(zero(), div(set1(1.0), rho), safe);
    const vd dhx = blend(sux, mul(dx, inv), safe);
    const vd dhy = blend(suy, mul(dy, inv), safe);
    const vd cphi = add(mul(sux, dhx), mul(suy, dhy));
    const vd sphi = sub(mul(sux, dhy), mul(suy, dhx));
    const vd Vr = L(sre[0]), Vi = L(sim[0]), Zr = L(sre[1]), Zi = L(sim[1]);
    const vd Hr = L(sre[2]), Hi = L(sim[2]), Pr = L(sre[3]), Pi = L(sim[3]);
    auto cmr = [](vd ar, vd ai, vd br, vd bi) { return sub(mul(ar, br), mul(ai, bi)); };
    auto cmi = [](vd ar, vd ai, vd br, vd bi) { return add(mul(ar, bi), mul(ai, br)); };
    const vd sc = mul(sth, cphi);
    // e_rho = g * (stzsrc * IrhoV + (sthsrc * cphi) * IrhoH)
    const vd tr = add(mul(stz, Vr), mul(sc, Hr));
    const vd ti = add(mul(stz, Vi), mul(sc, Hi));
    const vd er = cmr(gr, gi, tr, ti), ei = cmi(gr, gi, tr, ti);
    // e_phi = g * ((sthsrc * sphi) * IphiH)
    const vd f = mul(sth, sphi);
    const vd ur = mul(f, Pr), ui = mul(f, Pi);
    const vd pr = cmr(gr, gi, ur, ui), pi = cmi(gr, gi, ur, ui);
    // e_z = g * (stzsrc * IzV - (sthsrc * cphi) * IrhoV)
    const vd cr = sub(mul(stz, Zr), mul(sc, Vr));
    const vd ci = sub(mul(stz, Zi), mul(sc, Vi));
    const vd zr = cmr(gr, gi, cr, ci), zi = cmi(gr, gi, cr, ci);
    // tox * (dhx * e_rho - dhy * e_phi) + toy * (dhy * e_rho + dhx * e_phi)
    //   + toz * e_z
    const vd vtox = set1(tox), vtoy = set1(toy), vtoz = set1(toz);
    const vd d1r = sub(mul(dhx, er), mul(dhy, pr));
    const vd d1i = sub(mul(dhx, ei), mul(dhy, pi));
    const vd d2r = add(mul(dhy, er), mul(dhx, pr));
    const vd d2i = add(mul(dhy, ei), mul(dhx, pi));
    const vd o_r = add(add(mul(vtox, d1r), mul(vtoy, d2r)), mul(vtoz, zr));
    const vd o_i = add(add(mul(vtox, d1i), mul(vtoy, d2i)), mul(vtoz, zi));
    store_interleaved(reinterpret_cast<double *>(out), o_r, o_i);
}
#endif  // MW_SOMM_LANES
}  // namespace somm_lanes

static std::atomic<unsigned long long> g_somm_proj_lane_pairs{0};

// obs/t_obs (M,3), src/t_src (S,3); returns (M,S) complex. The grid is passed
// flattened: the regions' (r0, dr, th0, dth) as per-region arrays, plus a
// list of four-or-six (4, n_r, n_th) complex value tables. r_break / th_split select
// the region exactly as SommerfeldGrid.eval.
static py::array_t<std::complex<double>> remainder_field_proj_batch(
    py::array_t<double, py::array::c_style | py::array::forcecast> obs,
    py::array_t<double, py::array::c_style | py::array::forcecast> t_obs,
    py::array_t<double, py::array::c_style | py::array::forcecast> src,
    py::array_t<double, py::array::c_style | py::array::forcecast> t_src,
    double ground_z, double k, double r1_max, double r_break, double th_split,
    double r_near,
    py::array_t<double, py::array::c_style | py::array::forcecast> reg_r0,
    py::array_t<double, py::array::c_style | py::array::forcecast> reg_dr,
    py::array_t<double, py::array::c_style | py::array::forcecast> reg_th0,
    py::array_t<double, py::array::c_style | py::array::forcecast> reg_dth,
    std::vector<py::array_t<std::complex<double>,
                            py::array::c_style | py::array::forcecast>> reg_vals,
    py::array_t<std::complex<double>, py::array::c_style | py::array::forcecast>
        far,
    uintptr_t cancel_flag = 0, bool lanes = true) {
    using somm_proj::cd;
    auto ob = obs.unchecked<2>();
    auto tob = t_obs.unchecked<2>();
    auto sb = src.unchecked<2>();
    auto tsb = t_src.unchecked<2>();
    if (ob.shape(1) != 3 || tob.shape(1) != 3 || sb.shape(1) != 3 ||
        tsb.shape(1) != 3)
        throw std::runtime_error("obs/src/tangent arrays must have shape (*, 3)");
    if (ob.shape(0) != tob.shape(0) || sb.shape(0) != tsb.shape(0))
        throw std::runtime_error("points and tangents must have matching length");

    const py::ssize_t M = ob.shape(0);
    const py::ssize_t S = sb.shape(0);
    somm_proj::GridView G = somm_proj::build_grid_view(
        r1_max, r_break, th_split, r_near, reg_r0.unchecked<1>(),
        reg_dr.unchecked<1>(), reg_th0.unchecked<1>(), reg_dth.unchecked<1>(),
        reg_vals);
    somm_proj::set_far(G, far);

    py::array_t<std::complex<double>> out({M, S});
    auto out_m = out.mutable_unchecked<2>();

    py::gil_scoped_release release;

    // Per-source constants (n-only): position + tangent decomposition.
    std::vector<double> sx(S), sy(S), sz(S), ux(S), uy(S), thsrc(S), tzsrc(S);
    for (py::ssize_t n = 0; n < S; ++n) {
        sx[n] = sb(n, 0);
        sy[n] = sb(n, 1);
        sz[n] = sb(n, 2);
        somm_proj::tangent_decomp(tsb(n, 0), tsb(n, 1), tsb(n, 2), ux[n], uy[n],
                                  thsrc[n], tzsrc[n]);
    }

#if MW_SOMM_LANES
    int nR32[somm_proj::MAX_REGIONS], nTh32[somm_proj::MAX_REGIONS];
    for (size_t g = 0; g < reg_vals.size(); ++g) {
        nR32[g] = static_cast<int>(G.nR[g]);
        nTh32[g] = static_cast<int>(G.nTh[g]);
    }
    const bool use_lanes = lanes;
    if (use_lanes) g_somm_proj_lane_pairs += (unsigned long long)(M * S);
#else
    (void)lanes;
#endif

    MW_CANCEL_SETUP(cancel_flag);
    // Rows dynamically: an entry is its own pair's arithmetic, so where a row
    // runs moves no bit, and a call's few dozen rows split unevenly over a
    // static schedule.
    #pragma omp parallel for schedule(dynamic)
    for (py::ssize_t m = 0; m < M; ++m) {
        MW_CANCEL_POLL();
        const double ox = ob(m, 0), oy = ob(m, 1), oz = ob(m, 2);
        const double tox = tob(m, 0), toy = tob(m, 1), toz = tob(m, 2);
#if MW_SOMM_LANES
        if (use_lanes) {
            using namespace somm_lanes;
            constexpr int B = ABOVE_BLOCK;
            double dx[B], dy[B], rho[B], r1[B], th[B], gre[B], gim[B];
            int l_reg[B], l_i0[B], l_j0[B];
            double l_wr[4][B], l_wt[4][B], l_sre[4][B], l_sim[4][B];
            cd *orow = &out_m(m, 0);
            for (py::ssize_t n0 = 0; n0 < S; n0 += B) {
                const int nb = (int)std::min<py::ssize_t>(B, S - n0);
                const int nq4 = nb - nb % W;
                // Geometry and the libm calls, per pair: `proj_one`'s
                // dx/dy/rho/hh and `proj_core`'s r1, theta (clamped) and g.
                for (int b = 0; b < nq4; ++b) {
                    const py::ssize_t n = n0 + b;
                    dx[b] = ox - sx[n];
                    dy[b] = oy - sy[n];
                    rho[b] = std::hypot(dx[b], dy[b]);
                    const double hh = (oz - ground_z) + (sz[n] - ground_z);
                    r1[b] = std::sqrt(rho[b] * rho[b] + hh * hh);
                    double theta = std::atan2(hh, rho[b]);
                    if (theta < 0.0) theta = 0.0;
                    else if (theta > G.half_pi) theta = G.half_pi;
                    th[b] = theta;
                    const cd g = std::polar(1.0 / r1[b], -k * r1[b]);
                    gre[b] = g.real();
                    gim[b] = g.imag();
                }
                for (int b = 0; b < nq4; b += W)
                    stencil_lanes(G, nR32, nTh32, r1, th, l_reg, l_i0, l_j0,
                                  l_wr, l_wt, b);
                for (int b = 0; b < nq4; ++b)
                    somm_proj_lanes::surfaces_lanes<B, false>(
                        G, l_reg[b], l_i0[b], l_j0[b], l_wr, l_wt, l_sre, l_sim, b);
                // Past the table's edge (momwire#1258): the continuation, per
                // pair, on the read surfaces, exactly as `proj_core` applies it.
                if (G.far.on) {
                    for (int b = 0; b < nq4; ++b) {
                        if (!(r1[b] > G.r1_max)) continue;
                        cd surf[4];
                        for (int q = 0; q < 4; ++q) surf[q] = cd(l_sre[q][b], l_sim[q][b]);
                        somm_proj::far_continue(G, r1[b], th[b], surf);
                        for (int q = 0; q < 4; ++q) {
                            l_sre[q][b] = surf[q].real();
                            l_sim[q][b] = surf[q].imag();
                        }
                    }
                }
                for (int b = 0; b < nq4; b += W)
                    project_lanes(G, l_sre, l_sim, gre, gim, rho, dx, dy, tox,
                                  toy, toz, &ux[n0 + b], &uy[n0 + b],
                                  &thsrc[n0 + b], &tzsrc[n0 + b], b,
                                  orow + n0 + b);
                for (int b = nq4; b < nb; ++b) {
                    const py::ssize_t n = n0 + b;
                    orow[n] = somm_proj::proj_one(
                        G, ground_z, k, ox, oy, oz, tox, toy, toz, sx[n], sy[n],
                        sz[n], ux[n], uy[n], thsrc[n], tzsrc[n]);
                }
            }
            continue;
        }
#endif
        for (py::ssize_t n = 0; n < S; ++n) {
            out_m(m, n) = somm_proj::proj_one(
                G, ground_z, k, ox, oy, oz, tox, toy, toz, sx[n], sy[n], sz[n],
                ux[n], uy[n], thsrc[n], tzsrc[n]);
        }
    }
    MW_THROW_IF_ABORTED();
    return out;
}

// Fully-fused b-spline Galerkin Sommerfeld remainder (Phase 4b, stage 2):
// returns the (n_basis, n_basis) block Q directly, absorbing the moment
// quadrature and the basis assembly that the Python code did with two einsums
// and a large fancy-index gather. Segments are the shared obs=src set.
//
//   Jf[p,P,i,j] = sum_{qi,rj} W[p,i,qi] * proj(node[i,qi], node[j,rj]) * W[P,j,rj]
//   Q[m,n]      = sum_{a,b,p,P} polys[m,a,p] * Jf[p,P, supp[m,a], supp[n,b]]
//                                            * polys[n,b,P]
//
// Rectangular obs/src form: the dense b-spline block is the symmetric case
// (obs == src, loc == supp_seg), the ACA sampler passes thin obs/src segment
// subsets with local support maps. obs_nodes (nsI, q, 3), obs_tang (nsI, 3),
// W_obs (d+1, nsI, q); likewise src_*; loc_I (nI, d+1) int64 indexes obs
// segments, pI (nI, d+1, d+1); loc_J/pJ index src. Grid passed as in
// remainder_field_proj_batch. Returns Q (nI, nJ).
//
// Observer banding (momwire#343). Jf over the FULL (nsI, nsJ) rectangle is
// 16*(d+1)^2*nsI*nsJ bytes -- 144 N^2 at d=2, ~9x the dense Z it contributes
// to, and ~10 GB at N=8320. The two stages are instead run band by band over
// the observer segments: stage 1 fills Jf for the band [i0, i1) only, stage 2
// immediately accumulates that band's wing contributions into Q, and the slab
// is reused for the next band. Peak Jf residency is therefore
//   16 * (d+1)^2 * band * nsJ  <=  MAX_JF_SLAB_BYTES
// with `band` derived from that budget (>= 1, capped at nsI). One band means
// the old single-shot behavior, so thin-obs callers (the ACA sampler) are
// unaffected.
//
// Exactness of the split: Q[m,n] is a plain sum over the (a, b) wing pairs of
// basis m and basis n, and banding partitions ONLY the `a` axis (each wing a
// of m lands in exactly one band, the one holding segment loc_I[m,a]). Every
// pair is therefore visited exactly once. The accumulation is also kept in
// the original order: a band seeds its local accumulator from the current
// Q[m,n] and adds its in-band (a, b) terms in increasing (a, b), so as long
// as loc_I[m,:] is non-decreasing (true for b-spline supports and for the
// searchsorted local maps the ACA sampler builds) the summation order across
// bands is identical to the unbanded loop, and Q is bit-identical.
static constexpr size_t MAX_JF_SLAB_BYTES = 64u << 20;  // 64 MiB

// ---- The symmetric route (the dense obs == src block) ----------------------
//
// Reciprocity makes the remainder dyad symmetric, but Q is NOT symmetric bit
// for bit -- the projection of (m, n) and of (n, m) round differently -- so
// this route does not mirror Q. What it shares is narrower and exact: for a
// node pair the costly half of `proj_one` (grid interpolation, continuation,
// g: `somm_proj::proj_core`) reads only rho and hh, and both are bit-identical
// with observer and source swapped. Each unordered node pair therefore pays
// for one core and two `proj_project`s instead of two whole `proj_one`s, and
// every Jf entry and every Q entry is then formed by the rectangular kernel's
// own expressions in its own order. Q is bit-identical to the banded kernel's
// (`tests/test_remainder_symmetric_q.py` holds the two routes equal).
//
// Sharing across (i, j) and (j, i) needs both moment blocks live at once,
// which the observer bands cannot give (a band's mirror lives in a later
// band), so this route tiles the BASIS axis instead: a tile pair (A, B),
// A <= B, fills Jf over (segments under A) x (segments under B) and its mirror
// together, then finishes Q's (A, B) and (B, A) blocks outright. A Q entry is
// complete inside one tile pair, so its wing sum can be replayed in the banded
// kernel's exact order: wings a by (band of loc[m, a], a), each against b in
// increasing order -- the same order the bands produce whatever the support
// maps look like. A segment under two tiles (a support's halo) is filled by
// both: a tile's segment set is ~TILE + d, so ~2 d / TILE of the work is
// repeated (3 % at d = 2 and the default tile).
//
// Scratch is per thread, two (d+1)^2 x |S_A| x |S_B| blocks laid out with
// the (p, P) moments innermost, so stage 2 reads each wing pair's moments
// contiguously; the tile is sized so all threads' scratch fits the same
// `max_jf_bytes` budget the bands honour.
static constexpr py::ssize_t SYM_TILE_MAX = 128;

namespace somm_sym {
using somm_proj::cd;

struct Tile {
    py::ssize_t m0 = 0, m1 = 0;     // basis rows [m0, m1)
    std::vector<py::ssize_t> segs;  // the rows' support segments, sorted unique
    std::vector<int32_t> loc;       // (m1 - m0, d1): index of loc[m, a] in segs
};

// Per-segment data, read by raw pointer (every input is c_style + forcecast).
struct Side {
    const double *nodes;  // (ns, q, 3)
    const double *tang;   // (ns, 3)
    const double *W;      // (d1, ns, q)
    py::ssize_t ns, q;
    std::vector<double> ux, uy, th, tz;  // tangent_decomp per segment
    double node(py::ssize_t i, py::ssize_t qi, int c) const {
        return nodes[(i * q + qi) * 3 + c];
    }
    double w(int p, py::ssize_t i, py::ssize_t qi) const {
        return W[(p * ns + i) * q + qi];
    }
};

// Jf[p, P] of observer segment i against source segment j from the node-pair
// block f[qi * q + rj] (observer node qi, source node rj): the rectangular
// kernel's stage-1 contraction, expression for expression, except that its
// inner sum over rj -- which does not read p -- is formed once per (P, qi)
// into `rows` (d1 * q scratch) instead of once per (p, P, qi). The same sums
// in the same order, so the same bits.
static inline void contract(const Side &S, int d1, py::ssize_t i, py::ssize_t j,
                            const cd *f, cd *rows, cd *out) {
    const py::ssize_t q = S.q;
    for (int P = 0; P < d1; ++P) {
        for (py::ssize_t qi = 0; qi < q; ++qi) {
            cd row(0.0, 0.0);
            for (py::ssize_t rj = 0; rj < q; ++rj)
                row += f[qi * q + rj] * S.w(P, j, rj);
            rows[P * q + qi] = row;
        }
    }
    for (int p = 0; p < d1; ++p) {
        for (int P = 0; P < d1; ++P) {
            cd acc(0.0, 0.0);
            for (py::ssize_t qi = 0; qi < q; ++qi) {
                const double wp = S.w(p, i, qi);
                acc += wp * rows[P * q + qi];
            }
            out[p * d1 + P] = acc;
        }
    }
}

// Both orientations of the segment pair (i, j): fw[qi * q + rj] is observer
// node (i, qi) against source node (j, rj), bw[rj * q + qi] the reverse. One
// core per node pair; for i == j the pair (qi, rj) and (rj, qi) is one node
// pair too, and fw alone is filled (it is its own mirror).
static inline void node_block(const somm_proj::GridView &G, double ground_z,
                              double k, const Side &S, py::ssize_t i,
                              py::ssize_t j, cd *fw, cd *bw) {
    const py::ssize_t q = S.q;
    const double *ti = S.tang + i * 3, *tj = S.tang + j * 3;
    for (py::ssize_t qi = 0; qi < q; ++qi) {
        const double ox = S.node(i, qi, 0), oy = S.node(i, qi, 1),
                     oz = S.node(i, qi, 2);
        const py::ssize_t r0 = (i == j) ? qi : 0;
        for (py::ssize_t rj = r0; rj < q; ++rj) {
            const double sx = S.node(j, rj, 0), sy = S.node(j, rj, 1),
                         sz = S.node(j, rj, 2);
            const double dx = ox - sx;
            const double dy = oy - sy;
            const double rho = std::hypot(dx, dy);
            const double hh = (oz - ground_z) + (sz - ground_z);
            cd surf[4], g;
            somm_proj::proj_core(G, k, rho, hh, surf, g);
            fw[qi * q + rj] = somm_proj::proj_project(
                G, dx, dy, rho, surf, g, ti[0], ti[1], ti[2], S.ux[j], S.uy[j],
                S.th[j], S.tz[j]);
            if (i == j && rj == qi) continue;
            // The reverse pair's own differences, as `proj_one` would form
            // them (exactly -dx, -dy).
            const double rdx = sx - ox;
            const double rdy = sy - oy;
            const cd v = somm_proj::proj_project(
                G, rdx, rdy, rho, surf, g, tj[0], tj[1], tj[2], S.ux[i],
                S.uy[i], S.th[i], S.tz[i]);
            if (i == j)
                fw[rj * q + qi] = v;
            else
                bw[rj * q + qi] = v;
        }
    }
}

// Q[m, n] for m under tile T (row role, its local wing indices xl) and n
// under tile U, from J laid out [(x * ny + y) * dd + p * d1 + P]. Wings of m
// in `ord` order, each against b ascending; inner sums as the banded kernel.
static inline void finish_block(const Tile &T, const Tile &U, py::ssize_t ny,
                                int d1, const int32_t *ord, const double *pl,
                                const cd *J, cd *Q, py::ssize_t nb) {
    const int dd = d1 * d1;
    for (py::ssize_t m = T.m0; m < T.m1; ++m) {
        const int32_t *xm = &T.loc[(size_t)(m - T.m0) * d1];
        const int32_t *om = &ord[(size_t)m * d1];
        for (py::ssize_t n = U.m0; n < U.m1; ++n) {
            const int32_t *yn = &U.loc[(size_t)(n - U.m0) * d1];
            cd qmn(0.0, 0.0);
            for (int r = 0; r < d1; ++r) {
                const int a = om[r];
                const double *pma_row = &pl[((size_t)m * d1 + a) * d1];
                const cd *jx = &J[(size_t)xm[a] * ny * dd];
                for (int b = 0; b < d1; ++b) {
                    const double *pnb = &pl[((size_t)n * d1 + b) * d1];
                    const cd *jf = jx + (size_t)yn[b] * dd;
                    cd inner(0.0, 0.0);
                    for (int p = 0; p < d1; ++p) {
                        const double pma = pma_row[p];
                        cd s(0.0, 0.0);
                        for (int P = 0; P < d1; ++P)
                            s += jf[p * d1 + P] * pnb[P];
                        inner += pma * s;
                    }
                    qmn += inner;
                }
            }
            Q[(size_t)m * nb + n] = qmn;
        }
    }
}
}  // namespace somm_sym

// The symmetric route's driver; see the block comment above. `band` is the
// banded kernel's observer band height, read only for the wing order.
static void remainder_Q_symmetric(
    const somm_proj::GridView &G, double ground_z, double k,
    somm_sym::Side &S, int d1, const int64_t *loc, const double *pl,
    py::ssize_t nb, py::ssize_t band, size_t budget, std::complex<double> *Q,
    const volatile int32_t *pysim_cancel, std::atomic<bool> &pysim_aborted) {
    using somm_proj::cd;
    using somm_sym::Tile;
    const py::ssize_t q = S.q;
    const int dd = d1 * d1;

    // Wing order per basis: (band of loc[m, a], a), stable.
    std::vector<int32_t> ord((size_t)nb * d1);
    for (py::ssize_t m = 0; m < nb; ++m) {
        int32_t *om = &ord[(size_t)m * d1];
        for (int a = 0; a < d1; ++a) om[a] = a;
        std::stable_sort(om, om + d1, [&](int32_t x, int32_t y) {
            return loc[m * d1 + x] / band < loc[m * d1 + y] / band;
        });
    }

    int n_threads = 1;
#ifdef _OPENMP
    n_threads = omp_get_max_threads();
#endif
    // Tile height: all threads' two scratch blocks within the budget, with a
    // support halo of d1 - 1 segments on either side of the tile's own.
    py::ssize_t tile = SYM_TILE_MAX;
    while (tile > 1) {
        const double side = (double)(tile + 2 * (d1 - 1));
        const double bytes =
            (double)n_threads * 2.0 * dd * side * side * sizeof(cd);
        if (bytes <= (double)budget) break;
        tile = tile / 2;
    }

    std::vector<Tile> tiles;
    py::ssize_t smax = 0;
    for (py::ssize_t m0 = 0; m0 < nb; m0 += tile) {
        Tile T;
        T.m0 = m0;
        T.m1 = std::min<py::ssize_t>(m0 + tile, nb);
        for (py::ssize_t m = T.m0; m < T.m1; ++m)
            for (int a = 0; a < d1; ++a) T.segs.push_back(loc[m * d1 + a]);
        std::sort(T.segs.begin(), T.segs.end());
        T.segs.erase(std::unique(T.segs.begin(), T.segs.end()), T.segs.end());
        T.loc.resize((size_t)(T.m1 - T.m0) * d1);
        for (py::ssize_t m = T.m0; m < T.m1; ++m)
            for (int a = 0; a < d1; ++a)
                T.loc[(size_t)(m - T.m0) * d1 + a] = (int32_t)(
                    std::lower_bound(T.segs.begin(), T.segs.end(),
                                     loc[m * d1 + a]) -
                    T.segs.begin());
        smax = std::max<py::ssize_t>(smax, (py::ssize_t)T.segs.size());
        tiles.push_back(std::move(T));
    }
    const py::ssize_t n_tiles = (py::ssize_t)tiles.size();
    std::vector<std::pair<py::ssize_t, py::ssize_t>> pairs;
    pairs.reserve((size_t)(n_tiles * (n_tiles + 1) / 2));
    for (py::ssize_t A = 0; A < n_tiles; ++A)
        for (py::ssize_t B = A; B < n_tiles; ++B) pairs.emplace_back(A, B);
    const py::ssize_t n_pairs = (py::ssize_t)pairs.size();

    #pragma omp parallel
    {
        std::vector<cd> JAB((size_t)dd * smax * smax), JBA(JAB.size());
        std::vector<cd> fw((size_t)q * q), bw((size_t)q * q), rows((size_t)d1 * q);
        #pragma omp for schedule(dynamic)
        for (py::ssize_t t = 0; t < n_pairs; ++t) {
            MW_CANCEL_POLL();
            const py::ssize_t A = pairs[(size_t)t].first;
            const py::ssize_t B = pairs[(size_t)t].second;
            const Tile &TA = tiles[(size_t)A], &TB = tiles[(size_t)B];
            const py::ssize_t nx = (py::ssize_t)TA.segs.size();
            const py::ssize_t ny = (py::ssize_t)TB.segs.size();
            const bool diag = A == B;
            for (py::ssize_t x = 0; x < nx; ++x) {
                const py::ssize_t si = TA.segs[(size_t)x];
                // On the diagonal tile pair the (y, x) entry is the mirror of
                // (x, y), so only y >= x is walked.
                for (py::ssize_t y = diag ? x : 0; y < ny; ++y) {
                    const py::ssize_t sj = TB.segs[(size_t)y];
                    cd *jxy = &JAB[((size_t)x * ny + y) * dd];
                    somm_sym::node_block(G, ground_z, k, S, si, sj, fw.data(),
                                         bw.data());
                    somm_sym::contract(S, d1, si, sj, fw.data(), rows.data(), jxy);
                    if (si == sj) {
                        if (!diag)
                            std::copy(jxy, jxy + dd,
                                      &JBA[((size_t)y * nx + x) * dd]);
                        continue;
                    }
                    cd *jyx = diag ? &JAB[((size_t)y * ny + x) * dd]
                                   : &JBA[((size_t)y * nx + x) * dd];
                    somm_sym::contract(S, d1, sj, si, bw.data(), rows.data(), jyx);
                }
            }
            somm_sym::finish_block(TA, TB, ny, d1, ord.data(), pl, JAB.data(),
                                   Q, nb);
            if (!diag)
                somm_sym::finish_block(TB, TA, nx, d1, ord.data(), pl,
                                       JBA.data(), Q, nb);
        }
    }
}

// Calls served by the symmetric route, so a gate can show the route ran
// rather than infer it from agreement (an untaken route agrees trivially).
static std::atomic<long long> g_remainder_symmetric_calls{0};

// True when the obs and src halves of the call are the same data -- the
// dense block's call shape -- so the symmetric route serves it.
template <typename T>
static bool same_array(const py::array_t<T, py::array::c_style | py::array::forcecast> &a,
                       const py::array_t<T, py::array::c_style | py::array::forcecast> &b) {
    if (a.ndim() != b.ndim()) return false;
    for (py::ssize_t d = 0; d < a.ndim(); ++d)
        if (a.shape(d) != b.shape(d)) return false;
    if (a.data() == b.data()) return true;
    return std::memcmp(a.data(), b.data(), sizeof(T) * (size_t)a.size()) == 0;
}

static py::array_t<std::complex<double>> sommerfeld_remainder_bspline_Q(
    py::array_t<double, py::array::c_style | py::array::forcecast> obs_nodes,
    py::array_t<double, py::array::c_style | py::array::forcecast> obs_tang,
    py::array_t<double, py::array::c_style | py::array::forcecast> W_obs,
    py::array_t<double, py::array::c_style | py::array::forcecast> src_nodes,
    py::array_t<double, py::array::c_style | py::array::forcecast> src_tang,
    py::array_t<double, py::array::c_style | py::array::forcecast> W_src,
    py::array_t<int64_t, py::array::c_style | py::array::forcecast> loc_I,
    py::array_t<double, py::array::c_style | py::array::forcecast> pI,
    py::array_t<int64_t, py::array::c_style | py::array::forcecast> loc_J,
    py::array_t<double, py::array::c_style | py::array::forcecast> pJ,
    double ground_z, double k, double r1_max, double r_break, double th_split,
    double r_near,
    py::array_t<double, py::array::c_style | py::array::forcecast> reg_r0,
    py::array_t<double, py::array::c_style | py::array::forcecast> reg_dr,
    py::array_t<double, py::array::c_style | py::array::forcecast> reg_th0,
    py::array_t<double, py::array::c_style | py::array::forcecast> reg_dth,
    std::vector<py::array_t<std::complex<double>,
                            py::array::c_style | py::array::forcecast>> reg_vals,
    py::array_t<std::complex<double>, py::array::c_style | py::array::forcecast>
        far,
    uintptr_t cancel_flag = 0, size_t max_jf_bytes = 0, bool reference = false) {
    using somm_proj::cd;
    auto ndI = obs_nodes.unchecked<3>();
    auto tgI = obs_tang.unchecked<2>();
    auto WvI = W_obs.unchecked<3>();
    auto ndJ = src_nodes.unchecked<3>();
    auto tgJ = src_tang.unchecked<2>();
    auto WvJ = W_src.unchecked<3>();
    auto lI = loc_I.unchecked<2>();
    auto lJ = loc_J.unchecked<2>();
    auto plI = pI.unchecked<3>();
    auto plJ = pJ.unchecked<3>();

    const py::ssize_t nsI = ndI.shape(0);
    const py::ssize_t nsJ = ndJ.shape(0);
    const py::ssize_t q = ndI.shape(1);
    const int d1 = (int)WvI.shape(0);  // degree + 1
    const py::ssize_t nI = lI.shape(0);
    const py::ssize_t nJ = lJ.shape(0);
    if (ndI.shape(2) != 3 || ndJ.shape(2) != 3 || ndJ.shape(1) != q)
        throw std::runtime_error("obs/src nodes must be (n_seg, q, 3), same q");
    if (WvI.shape(1) != nsI || WvI.shape(2) != q || WvJ.shape(1) != nsJ ||
        WvJ.shape(2) != q || (int)WvJ.shape(0) != d1)
        throw std::runtime_error("W_obs/W_src inconsistent with nodes/degree");
    if (lI.shape(1) != d1 || lJ.shape(1) != d1 || plI.shape(1) != d1 ||
        plI.shape(2) != d1 || plJ.shape(1) != d1 || plJ.shape(2) != d1)
        throw std::runtime_error("loc/polys inconsistent with degree");

    somm_proj::GridView G = somm_proj::build_grid_view(
        r1_max, r_break, th_split, r_near, reg_r0.unchecked<1>(),
        reg_dr.unchecked<1>(), reg_th0.unchecked<1>(), reg_dth.unchecked<1>(),
        reg_vals);
    somm_proj::set_far(G, far);

    // The dense block's call shape (obs == src throughout) takes the
    // symmetric route unless `reference` asks for this banded kernel, which
    // stays the route of every rectangular call and the oracle the symmetric
    // one is held to.
    const bool symmetric =
        !reference && same_array(obs_nodes, src_nodes) &&
        same_array(obs_tang, src_tang) && same_array(W_obs, W_src) &&
        same_array(loc_I, loc_J) && same_array(pI, pJ);

    py::array_t<std::complex<double>> Q({nI, nJ});
    auto Qm = Q.mutable_unchecked<2>();

    // Observer band size: the largest number of obs segments whose Jf slab
    // fits the budget (#343). `band == nsI` reproduces the unbanded kernel.
    const size_t budget = max_jf_bytes ? max_jf_bytes : MAX_JF_SLAB_BYTES;
    const size_t row_bytes = sizeof(cd) * (size_t)d1 * d1 * (size_t)nsJ;
    py::ssize_t band = nsI;
    if (row_bytes > 0) {
        size_t fit = budget / row_bytes;
        if (fit < 1) fit = 1;
        if ((py::ssize_t)fit < band) band = (py::ssize_t)fit;
    }
    if (band < 1) band = 1;

    if (symmetric) {
        g_remainder_symmetric_calls.fetch_add(1, std::memory_order_relaxed);
        somm_sym::Side S;
        S.nodes = obs_nodes.data();
        S.tang = obs_tang.data();
        S.W = W_obs.data();
        S.ns = nsI;
        S.q = q;
        S.ux.resize(nsI);
        S.uy.resize(nsI);
        S.th.resize(nsI);
        S.tz.resize(nsI);
        for (py::ssize_t j = 0; j < nsI; ++j)
            somm_proj::tangent_decomp(tgI(j, 0), tgI(j, 1), tgI(j, 2), S.ux[j],
                                      S.uy[j], S.th[j], S.tz[j]);
        cd *Qp = Q.mutable_data();
        const int64_t *locp = loc_I.data();
        const double *plp = pI.data();
        py::gil_scoped_release release;
        MW_CANCEL_SETUP(cancel_flag);
        remainder_Q_symmetric(G, ground_z, k, S, d1, locp, plp, nI, band,
                              budget, Qp, pysim_cancel, pysim_aborted);
        MW_THROW_IF_ABORTED();
        return Q;
    }

    std::fill(Q.mutable_data(), Q.mutable_data() + (size_t)nI * nJ, cd(0.0, 0.0));

    py::gil_scoped_release release;

    // Per-src-segment tangent decomposition.
    std::vector<double> sux(nsJ), suy(nsJ), sth(nsJ), stz(nsJ);
    for (py::ssize_t j = 0; j < nsJ; ++j)
        somm_proj::tangent_decomp(tgJ(j, 0), tgJ(j, 1), tgJ(j, 2), sux[j],
                                  suy[j], sth[j], stz[j]);

    // Stage 1 slab: Jf[p,P,i-i0,j] over the (band, nsJ) segment rectangle.
    // Flat index (((p*d1+P)*ib)+(i-i0))*nsJ+j, ib = this band's height.
    std::vector<cd> Jf((size_t)d1 * d1 * (size_t)band * nsJ);
    std::vector<py::ssize_t> rows;
    rows.reserve((size_t)std::min<py::ssize_t>(nI, band * d1 + d1));

    MW_CANCEL_SETUP(cancel_flag);
    for (py::ssize_t i0 = 0; i0 < nsI; i0 += band) {
        const py::ssize_t i1 = std::min<py::ssize_t>(i0 + band, nsI);
        const py::ssize_t ib = i1 - i0;
        const size_t seg2 = (size_t)ib * nsJ;

        // Stage 1: fill the band's moment slab.
        #pragma omp parallel for schedule(dynamic)
        for (py::ssize_t i = i0; i < i1; ++i) {
            MW_CANCEL_POLL();
            const double tox = tgI(i, 0), toy = tgI(i, 1), toz = tgI(i, 2);
            std::vector<cd> fblk((size_t)q * q);
            for (py::ssize_t j = 0; j < nsJ; ++j) {
                for (py::ssize_t qi = 0; qi < q; ++qi) {
                    const double ox = ndI(i, qi, 0), oy = ndI(i, qi, 1),
                                 oz = ndI(i, qi, 2);
                    for (py::ssize_t rj = 0; rj < q; ++rj) {
                        fblk[qi * q + rj] = somm_proj::proj_one(
                            G, ground_z, k, ox, oy, oz, tox, toy, toz,
                            ndJ(j, rj, 0), ndJ(j, rj, 1), ndJ(j, rj, 2),
                            sux[j], suy[j], sth[j], stz[j]);
                    }
                }
                for (int p = 0; p < d1; ++p) {
                    for (int P = 0; P < d1; ++P) {
                        cd acc(0.0, 0.0);
                        for (py::ssize_t qi = 0; qi < q; ++qi) {
                            const double wp = WvI(p, i, qi);
                            cd row(0.0, 0.0);
                            for (py::ssize_t rj = 0; rj < q; ++rj)
                                row += fblk[qi * q + rj] * WvJ(P, j, rj);
                            acc += wp * row;
                        }
                        Jf[((size_t)(p * d1 + P) * ib + (i - i0)) * nsJ + j] =
                            acc;
                    }
                }
            }
        }
        MW_THROW_IF_ABORTED();

        // Which basis rows have at least one wing landing in this band? A
        // row is listed once however many of its wings are in-band, so the
        // parallel stage-2 loop below owns each Q row exclusively.
        rows.clear();
        for (py::ssize_t m = 0; m < nI; ++m) {
            for (int a = 0; a < d1; ++a) {
                const py::ssize_t si = lI(m, a);
                if (si >= i0 && si < i1) {
                    rows.push_back(m);
                    break;
                }
            }
        }

        // Stage 2: accumulate this band's wing contributions into Q. The
        // out-of-band wings are skipped here and picked up by the band that
        // owns them, so every (a, b) pair is summed exactly once.
        const py::ssize_t n_rows = (py::ssize_t)rows.size();
        #pragma omp parallel for schedule(static)
        for (py::ssize_t r = 0; r < n_rows; ++r) {
            MW_CANCEL_POLL();
            const py::ssize_t m = rows[(size_t)r];
            for (py::ssize_t n = 0; n < nJ; ++n) {
                cd qmn = Qm(m, n);  // seeded, so the a-order is preserved
                for (int a = 0; a < d1; ++a) {
                    const py::ssize_t si = lI(m, a);
                    if (si < i0 || si >= i1) continue;
                    const py::ssize_t sl = si - i0;
                    for (int b = 0; b < d1; ++b) {
                        const py::ssize_t sj = lJ(n, b);
                        cd inner(0.0, 0.0);
                        for (int p = 0; p < d1; ++p) {
                            const double pma = plI(m, a, p);
                            const cd *jfp =
                                &Jf[((size_t)(p * d1) * ib + sl) * nsJ + sj];
                            cd s(0.0, 0.0);
                            for (int P = 0; P < d1; ++P)
                                s += jfp[(size_t)P * seg2] * plJ(n, b, P);
                            inner += pma * s;
                        }
                        qmn += inner;
                    }
                }
                Qm(m, n) = qmn;
            }
        }
        MW_THROW_IF_ABORTED();
    }
    return Q;
}


void register_somm(py::module_ &m) {

    m.def("somm_six_integrals_batch", &somm_six_integrals_batch,
          "Batched Sommerfeld six-integral evaluation at (rho[i], h[i]) "
          "nodes; OpenMP across nodes. form: 0 auto, 1 Bessel, 2 Hankel. "
          "Returns (n, 6) complex in _six_integrals order.",
          py::arg("eps_t"), py::arg("k2"), py::arg("rho"), py::arg("h"),
          py::arg("rtol") = 1e-9, py::arg("form") = 0,
          py::arg("cancel_flag") = 0);
    m.def("remainder_field_proj_batch", &remainder_field_proj_batch,
          "Fused Sommerfeld smooth-remainder assembly: interpolate the four "
          "grid surfaces (4x4 Lagrange) and project t_m.F(r_m,r_n).t_n per "
          "(observer, source) pair; OpenMP over observer rows. Returns (M, S) "
          "complex. The grid is passed flattened (per-region r0/dr/th0/dth "
          "arrays + a list of 3 near — or 5 with the #159 far zone — "
          "(4,n_r,n_th) value tables), then `far`: the momwire#1258 "
          "continuation past r1_max (`_sommerfeld.far_cpp_pack`; empty "
          "serves the edge value frozen).",
          py::arg("obs"), py::arg("t_obs"), py::arg("src"), py::arg("t_src"),
          py::arg("ground_z"), py::arg("k"), py::arg("r1_max"),
          py::arg("r_break"), py::arg("th_split"), py::arg("r_near"),
          py::arg("reg_r0"), py::arg("reg_dr"), py::arg("reg_th0"),
          py::arg("reg_dth"), py::arg("reg_vals"), py::arg("far"),
          py::arg("cancel_flag") = 0, py::arg("lanes") = true);
    m.def("somm_proj_lanes_built", []() { return (bool)MW_SOMM_LANES; },
          "Whether this build has remainder_field_proj_batch's AVX2 lanes "
          "(perf item 9); without them `lanes=True` runs the per-pair loop.");
    m.def("somm_proj_lane_pairs",
          []() { return g_somm_proj_lane_pairs.load(); },
          "How many pairs remainder_field_proj_batch has sent through its "
          "AVX2 lanes in this process (perf item 9): the lanes' output equals "
          "the per-pair loop's, so only this says which ran.");
    m.def("sommerfeld_remainder_bspline_Q", &sommerfeld_remainder_bspline_Q,
          "Fully-fused b-spline Galerkin Sommerfeld remainder over an obs/src "
          "rectangle: interpolate + project + moment-quadrature + basis-assemble "
          "into the (nI, nJ) Q block directly (no Jf / einsum intermediates). "
          "Dense block = symmetric case (obs==src, loc==supp_seg); the ACA "
          "sampler passes thin segment subsets with local support maps. Grid as "
          "in remainder_field_proj_batch. The internal moment slab is banded "
          "over observer segments so its residency is bounded by "
          "`max_jf_bytes` (0 = the 64 MiB default), never the full "
          "(d+1)^2 * nsI * nsJ tensor (momwire#343); the banding is exact and "
          "order-preserving, not an approximation. A call whose obs and src "
          "halves are the same data (the dense block) takes the symmetric "
          "route, one grid interpolation per unordered node pair, bit-identical "
          "to the banded kernel; `reference=True` forces the banded kernel.",
          py::arg("obs_nodes"), py::arg("obs_tang"), py::arg("W_obs"),
          py::arg("src_nodes"), py::arg("src_tang"), py::arg("W_src"),
          py::arg("loc_I"), py::arg("pI"), py::arg("loc_J"), py::arg("pJ"),
          py::arg("ground_z"), py::arg("k"),
          py::arg("r1_max"), py::arg("r_break"), py::arg("th_split"),
          py::arg("r_near"), py::arg("reg_r0"), py::arg("reg_dr"), py::arg("reg_th0"),
          py::arg("reg_dth"), py::arg("reg_vals"), py::arg("far"),
          py::arg("cancel_flag") = 0, py::arg("max_jf_bytes") = 0,
          py::arg("reference") = false);
    m.def("remainder_q_symmetric_calls",
          []() { return g_remainder_symmetric_calls.load(); },
          "How many sommerfeld_remainder_bspline_Q calls the symmetric route "
          "has served in this process (a gate's evidence that it ran).");
}

