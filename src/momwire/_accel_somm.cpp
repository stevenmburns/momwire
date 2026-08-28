#include "_accel_common.h"

// somm section of the former _accelerators.cpp monolith (momwire#687).
// Code below is byte-identical to the monolith's lines 5402-6272.

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
    return z.real() * z.real() + z.imag() * z.imag();
}
static const double SER_EPS2 = 1e-34;  // (1e-17)^2

// J0 and J1/z ascending series (A&S 9.1.10/9.1.12): safe as z -> 0.
static void j01_series(cd z, cd &j0, cd &j1x) {
    const cd q = -0.25 * z * z;
    cd t0(1.0, 0.0), t1(0.5, 0.0);
    j0 = t0;
    j1x = t1;
    for (int k = 1; k <= 60; ++k) {
        t0 *= q / double(k * k);
        t1 *= q / double(k * (k + 1));
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
    const cd q = -0.25 * z * z;
    cd t0(1.0, 0.0), t1(0.5, 0.0);
    j0 = t0;
    j1x = t1;
    cd s0(0.0, 0.0), s1 = 2.0 * t1;  // k = 0: H_0 = 0, H_0 + H_1 = 1
    double hk = 0.0, hk1 = 1.0;
    for (int k = 1; k <= 60; ++k) {
        t0 *= q / double(k * k);
        t1 *= q / double(k * (k + 1));
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
    y0 = (2.0 / SPI) * (lg * j0 + s0);
    y1 = (2.0 / SPI) * (lg * (j1x * z) - 1.0 / z) - (z / (2.0 * SPI)) * s1;
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
            t0 *= (-odd2 / den) * iz;
            const double a = cnorm(t0);
            if (a >= prev0) {
                live0 = false;  // divergence onset: stop at the optimum
            } else {
                s0m += t0;
                if (want_plus) s0p += sgn * t0;
                prev0 = a;
                if (a <= SER_EPS2 * cnorm(s0m)) live0 = false;
            }
        }
        if (live1) {
            t1 *= ((4.0 - odd2) / den) * iz;
            const double a = cnorm(t1);
            if (a >= prev1) {
                live1 = false;
            } else {
                s1m += t1;
                if (want_plus) s1p += sgn * t1;
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
    const cd e = std::exp(-CI * (x - 0.25 * SPI));
    const cd p = w * e, pinv = w / e;
    b0 = 0.5 * (pinv * s0p + p * s0m);
    b1x = (0.5 * CI) * (p * s1m - pinv * s1p) / x;
}

// (H2_0(x)/2, H2_1(x)/(2x)) — the Hankel-form pair of _integrand_six.
static void hankel2_half(cd x, cd &b0, cd &b1x) {
    if (cnorm(x) <= BESSEL_SWITCH2) {
        cd j0, j1x, y0, y1;
        jy01_series(x, j0, j1x, y0, y1);
        b0 = 0.5 * (j0 - CI * y0);
        b1x = 0.5 * (j1x * x - CI * y1) / x;
        return;
    }
    cd s0m, s1m, s0p, s1p;
    hankel_asym_sums(x, false, s0m, s1m, s0p, s1p);
    const cd p = std::sqrt(2.0 / (SPI * x)) * std::exp(-CI * (x - 0.25 * SPI));
    b0 = 0.5 * p * s0m;
    b1x = (0.5 * CI) * p * s1m / x;
}

// ---- the six integrands and quadrature (ports of the Python names) -------

// gamma(lam, k) = sqrt(-j(lam-k)) sqrt(j(lam+k)): NEC's vertical cuts.
static inline cd gam(cd lam, cd k) {
    return std::sqrt(-CI * (lam - k)) * std::sqrt(CI * (lam + k));
}

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
    const cd g1 = gam(lam, c.k1);
    const cd g2 = gam(lam, c.k2);
    const cd k1s = c.k1 * c.k1;
    const cd k2s = c.k2 * c.k2;
    const cd d1 = 2.0 / (g1 + g2) - 2.0 * k2s / (g2 * (k1s + k2s));
    const cd d2 = 2.0 / (k1s * g2 + k2s * g1) - 2.0 / (g2 * (k1s + k2s));
    const cd e = std::exp(-g2 * c.h);
    const cd x = lam * c.rho;
    cd b0, b1x;
    if (c.bessel)
        bessel_j0_j1x(x, b0, b1x);
    else
        hankel2_half(x, b0, b1x);
    const cd l2 = lam * lam;
    const cd l3 = l2 * lam;
    const cd common = d2 * e;
    out[0] = common * (b1x - b0) * l3;
    out[1] = common * g2 * g2 * b0 * lam;
    out[2] = common * g2 * (b1x * x) * l2;
    out[3] = -common * b1x * l3;
    out[4] = common * b0 * lam;
    out[5] = d1 * e * b0 * lam;
}

static Six gauss_segment(const SommCtx &c, cd z0, cd z1) {
    const cd mid = 0.5 * (z0 + z1);
    const cd half = 0.5 * (z1 - z0);
    Six acc;
    cd f[6];
    for (int q = 0; q < 24; ++q) {
        integrand_six(c, mid + half * GX[q], f);
        const cd w = GW[q] * half;
        for (int i = 0; i < 6; ++i) acc.v[i] += f[i] * w;
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
    PYSIM_CANCEL_SETUP(cancel_flag);

    #pragma omp parallel for schedule(dynamic)
    for (py::ssize_t i = 0; i < n; ++i) {
        PYSIM_CANCEL_POLL();
        somm::cd res[6];
        somm::six_integrals(et, k2, rb(i), hb(i), rtol, form, res);
        for (int j = 0; j < 6; ++j) ob(i, j) = res[j];
    }
    PYSIM_THROW_IF_ABORTED();
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
    uintptr_t cancel_flag = 0) {
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

    PYSIM_CANCEL_SETUP(cancel_flag);
    #pragma omp parallel for schedule(static)
    for (py::ssize_t m = 0; m < M; ++m) {
        PYSIM_CANCEL_POLL();
        const double ox = ob(m, 0), oy = ob(m, 1), oz = ob(m, 2);
        const double tox = tob(m, 0), toy = tob(m, 1), toz = tob(m, 2);
        for (py::ssize_t n = 0; n < S; ++n) {
            out_m(m, n) = somm_proj::proj_one(
                G, ground_z, k, ox, oy, oz, tox, toy, toz, sx[n], sy[n], sz[n],
                ux[n], uy[n], thsrc[n], tzsrc[n]);
        }
    }
    PYSIM_THROW_IF_ABORTED();
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
    uintptr_t cancel_flag = 0, size_t max_jf_bytes = 0) {
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

    py::array_t<std::complex<double>> Q({nI, nJ});
    auto Qm = Q.mutable_unchecked<2>();
    std::fill(Q.mutable_data(), Q.mutable_data() + (size_t)nI * nJ, cd(0.0, 0.0));

    py::gil_scoped_release release;

    // Per-src-segment tangent decomposition.
    std::vector<double> sux(nsJ), suy(nsJ), sth(nsJ), stz(nsJ);
    for (py::ssize_t j = 0; j < nsJ; ++j)
        somm_proj::tangent_decomp(tgJ(j, 0), tgJ(j, 1), tgJ(j, 2), sux[j],
                                  suy[j], sth[j], stz[j]);

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

    // Stage 1 slab: Jf[p,P,i-i0,j] over the (band, nsJ) segment rectangle.
    // Flat index (((p*d1+P)*ib)+(i-i0))*nsJ+j, ib = this band's height.
    std::vector<cd> Jf((size_t)d1 * d1 * (size_t)band * nsJ);
    std::vector<py::ssize_t> rows;
    rows.reserve((size_t)std::min<py::ssize_t>(nI, band * d1 + d1));

    PYSIM_CANCEL_SETUP(cancel_flag);
    for (py::ssize_t i0 = 0; i0 < nsI; i0 += band) {
        const py::ssize_t i1 = std::min<py::ssize_t>(i0 + band, nsI);
        const py::ssize_t ib = i1 - i0;
        const size_t seg2 = (size_t)ib * nsJ;

        // Stage 1: fill the band's moment slab.
        #pragma omp parallel for schedule(dynamic)
        for (py::ssize_t i = i0; i < i1; ++i) {
            PYSIM_CANCEL_POLL();
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
        PYSIM_THROW_IF_ABORTED();

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
            PYSIM_CANCEL_POLL();
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
        PYSIM_THROW_IF_ABORTED();
    }
    return Q;
}


// --------------------------------------------------------------------------
// momwire#568 unit 1 -- test instantiations of the shared contour engine.
//
// These exist so `tests/test_contour_engine_568.py` can gate the engine and
// its complex Bessel pair from Python BEFORE any production fill rides on
// them. Nothing in momwire's dispatch calls into this block; U2 (below fills)
// and U3 (transmitted fills) will instantiate `mw_contour::run_contour` with
// their own integrands and never go through these entry points.
// --------------------------------------------------------------------------


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
          "(4,n_r,n_th) value tables).",
          py::arg("obs"), py::arg("t_obs"), py::arg("src"), py::arg("t_src"),
          py::arg("ground_z"), py::arg("k"), py::arg("r1_max"),
          py::arg("r_break"), py::arg("th_split"), py::arg("r_near"),
          py::arg("reg_r0"), py::arg("reg_dr"), py::arg("reg_th0"),
          py::arg("reg_dth"), py::arg("reg_vals"),
          py::arg("cancel_flag") = 0);
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
          "order-preserving, not an approximation.",
          py::arg("obs_nodes"), py::arg("obs_tang"), py::arg("W_obs"),
          py::arg("src_nodes"), py::arg("src_tang"), py::arg("W_src"),
          py::arg("loc_I"), py::arg("pI"), py::arg("loc_J"), py::arg("pJ"),
          py::arg("ground_z"), py::arg("k"),
          py::arg("r1_max"), py::arg("r_break"), py::arg("th_split"),
          py::arg("r_near"), py::arg("reg_r0"), py::arg("reg_dr"), py::arg("reg_th0"),
          py::arg("reg_dth"), py::arg("reg_vals"),
          py::arg("cancel_flag") = 0, py::arg("max_jf_bytes") = 0);
}

