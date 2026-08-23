// The shared adaptive-contour engine, in C++ (momwire#568 unit 1).
//
// This is a faithful port of the five pure-numpy functions that carry every
// minute of the buried/transmitted Sommerfeld fills --
// `_sommerfeld_below._gauss_segment`, `_adaptive_segment`, `_wynn_epsilon`,
// `_head`, `_tail_below` and their `_run_contour` driver -- templated on the
// integrand so the below/below and transmitted families can each instantiate
// it with their own C++ integrand. The numpy engine stays the production path
// and the reference; the gates against it are TOLERANCES, never bit equality
// (the Gauss dot product, the complex divides and the transcendental library
// all differ in the last bits from numpy's).
//
// Structure, in the order the numpy module writes it:
//
//   gauss_segment   fixed Gauss rule on [z0, z1], vector-valued (NC components)
//   adaptive_segment recursive bisection with a ref-floored relative test
//   wynn_epsilon     best even column, elementwise, finite-guarded
//   head_contour     first-quadrant half-sine detour lam(t) = t + jH sin(pi t/a)
//   tail_below       real-axis tail panelled at the J0(lam rho) zeros
//   run_contour      a = 1.1 max(|k_p|, |k_m|), marks = (k_p, |Re k_m|)
//
// THE THREE MEASURED GUARDS in `tail_below` are transcribed verbatim, because
// every one of them was a real defect in the #524 phase-0 prototype:
//
//   * `+ 1e-9` inside the panel-index floor. Without it a z that lands exactly
//     on the Bessel-zero lattice floors back onto itself, two consecutive
//     zero-length panels contribute exactly 0.0, and the quiet counter reads
//     that as convergence -- the measured symptom was a tail truncating at
//     lam ~ 1.4 when the answer needed lam ~ 40, reported CONVERGED.
//   * The `1.5 / h` e-folding cap. One Gauss rule spanning 14 e-foldings of
//     e^{-lam h} was wrong by 8e-4.
//   * `cmax == 0.0` tested explicitly, and the relative quiet test floored by
//     `ref` (the head's scale), so a remainder decades below the divided-out
//     factor terminates on an absolute bar instead of chasing its own ulps.
//
// REENTRANCY (hard requirement -- U2/U3 call this from OpenMP threads with the
// GIL released): there is no mutable static, no global and no heap allocation
// anywhere below. Every buffer is a fixed-size automatic array whose extent is
// a compile-time constant (NC, the Gauss rule's node count is iterated rather
// than buffered, and the tail's partial-sum history is a 41-deep ring because
// that is all `_wynn_epsilon` is ever handed). Peak stack is ~10 kB per
// `wynn_epsilon` call at NC = 6 plus ~0.6 kB per `adaptive_segment` frame.
//
// Portability: C++11 (the build is -std=gnu++11), no GCC builtins, no VLAs,
// nothing that needs _USE_MATH_DEFINES from the includer.

#pragma once

#include <algorithm>
#include <cmath>
#include <complex>
#include <limits>

namespace mw_contour {

typedef std::complex<double> cd;

// Not M_PI: this header must not depend on the includer having defined
// _USE_MATH_DEFINES first (MSVC).
const double MW_PI = 3.14159265358979323846;

// The tail keeps at most this many partial sums, because `_wynn_epsilon` is
// only ever handed `sums[-min(len(sums), 41):]`. The numpy code accumulates
// the whole list; keeping a ring of the last 41 is behaviourally identical
// (the only other use of the list is its LENGTH, which is `parts`) and bounds
// the engine's footprint independently of `max_panels`.
const int MW_TAIL_KEEP = 41;

// ---------------------------------------------------------------------------
// Complex J0(x) and J1(x)/x
// ---------------------------------------------------------------------------
//
// The numpy side is `_sommerfeld._bessel_j0_j1x`: scipy.special.jv(0|1, x) at
// COMPLEX x, with a series switch at |x| < 1e-6 so rho -> 0 is safe. Three
// regimes, chosen to keep the worst cancellation bounded:
//
//   |x| < 1e-6   the two-term series the numpy code uses, transcribed
//   |x| < 8      ascending power series (worst cancellation ~3 digits)
//   |x| < 25     Miller's downward recurrence, normalized by
//                J0 + 2(J2 + J4 + ...) = 1 -- no cancellation at all, which
//                is exactly what the 8..25 window needs (the power series
//                loses ~10 digits at |x| = 25, the asymptotic series has not
//                yet reached its 1e-16 floor there)
//   otherwise    the Hankel P/Q asymptotic expansion
//
// The asymptotic branch deliberately does NOT form cos(z - pi/4) by
// subtracting pi/4 from z: at |z| ~ 500 that subtraction rounds at 3e-14 and
// poisons the answer near the zeros of J0. Using the angle-sum identities
// instead leaves only std::cos(z) / std::sin(z), whose own argument reduction
// is sub-ulp.
//
// J0 is even and J1(z)/z is even, so arguments with Re x < 0 are reflected.
//
// The three kernels are TEMPLATED ON THE SCALAR (`double` or `cd`) and the
// dispatcher takes the real path whenever Im x == 0. That is not a micro-
// optimization: the tail runs along the real axis by construction and carries
// the overwhelming majority of the panels, and the arithmetic in the inner
// recurrences is a serial latency chain, not a throughput problem. Measured
// on an i7-8550U, ns per evaluation (real path / complex path):
//
//     |x| = 1     24 / 58        (series)
//     |x| = 7     48 / 103       (series)
//     |x| = 10    80 / 222       (Miller)
//     |x| = 24   122 / 311       (Miller)
//     |x| = 30   132 / 356       (asymptotic, worst case: most P/Q terms)
//     |x| = 300   62 / 227       (asymptotic)
//
// The Bessel pair is ~55-60% of the whole engine's cost on a production-like
// contour, so these are the numbers that set the C++/numpy ratio.

static inline bool mw_finite(const cd &z) {
    return std::isfinite(z.real()) && std::isfinite(z.imag());
}

// |v|^2 without hypot, and a cheap "how big is this" for the rescale test.
static inline double mw_norm2(double v) { return v * v; }
static inline double mw_norm2(const cd &v) {
    return v.real() * v.real() + v.imag() * v.imag();
}
static inline double mw_mag(double v) { return std::fabs(v); }
static inline double mw_mag(const cd &v) {
    return std::max(std::fabs(v.real()), std::fabs(v.imag()));
}

// 1/m for the two term recursions below. Both loops otherwise carry a
// floating-point DIVIDE per iteration, and the divider is neither pipelined
// nor vectorized -- on this box those divides were the single largest line
// item in the whole engine. Every entry is a constant expression, so the
// compiler folds it to the correctly-rounded double at compile time; the
// table is const, hence shareable across OpenMP threads without a guard.
static const double MW_INV[82] = {
    0.0,       1.0 / 1.0,  1.0 / 2.0,  1.0 / 3.0,  1.0 / 4.0,  1.0 / 5.0,
    1.0 / 6.0,  1.0 / 7.0,  1.0 / 8.0,  1.0 / 9.0,  1.0 / 10.0, 1.0 / 11.0,
    1.0 / 12.0, 1.0 / 13.0, 1.0 / 14.0, 1.0 / 15.0, 1.0 / 16.0, 1.0 / 17.0,
    1.0 / 18.0, 1.0 / 19.0, 1.0 / 20.0, 1.0 / 21.0, 1.0 / 22.0, 1.0 / 23.0,
    1.0 / 24.0, 1.0 / 25.0, 1.0 / 26.0, 1.0 / 27.0, 1.0 / 28.0, 1.0 / 29.0,
    1.0 / 30.0, 1.0 / 31.0, 1.0 / 32.0, 1.0 / 33.0, 1.0 / 34.0, 1.0 / 35.0,
    1.0 / 36.0, 1.0 / 37.0, 1.0 / 38.0, 1.0 / 39.0, 1.0 / 40.0, 1.0 / 41.0,
    1.0 / 42.0, 1.0 / 43.0, 1.0 / 44.0, 1.0 / 45.0, 1.0 / 46.0, 1.0 / 47.0,
    1.0 / 48.0, 1.0 / 49.0, 1.0 / 50.0, 1.0 / 51.0, 1.0 / 52.0, 1.0 / 53.0,
    1.0 / 54.0, 1.0 / 55.0, 1.0 / 56.0, 1.0 / 57.0, 1.0 / 58.0, 1.0 / 59.0,
    1.0 / 60.0, 1.0 / 61.0, 1.0 / 62.0, 1.0 / 63.0, 1.0 / 64.0, 1.0 / 65.0,
    1.0 / 66.0, 1.0 / 67.0, 1.0 / 68.0, 1.0 / 69.0, 1.0 / 70.0, 1.0 / 71.0,
    1.0 / 72.0, 1.0 / 73.0, 1.0 / 74.0, 1.0 / 75.0, 1.0 / 76.0, 1.0 / 77.0,
    1.0 / 78.0, 1.0 / 79.0, 1.0 / 80.0, 1.0 / 81.0};

// Ascending series. Terms are tracked against their own running peak (not
// against the partial sum) so the stopping test stays meaningful when the sum
// itself is near a zero of J0. Magnitude tests are on the SQUARED modulus.
template <class T>
static inline void mw_bessel_series(const T &z, T &j0, T &j1) {
    const T w = -0.25 * (z * z);
    T t0(1.0), s0(1.0);
    T t1(0.5), s1(0.5);
    double p0 = 1.0, p1 = 0.25;  // squared peaks
    for (int m = 1; m < 80; ++m) {
        const double rm = MW_INV[m];
        t0 *= w * (rm * rm);
        t1 *= w * (rm * MW_INV[m + 1]);
        s0 += t0;
        s1 += t1;
        const double a0 = mw_norm2(t0), a1 = mw_norm2(t1);
        if (a0 > p0) p0 = a0;
        if (a1 > p1) p1 = a1;
        if (a0 <= 1e-38 * p0 && a1 <= 1e-38 * p1) break;
    }
    j0 = s0;
    j1 = s1 * z;  // s1 is J1(z)/z
}

// Miller's algorithm: downward recurrence from a starting order well past the
// turning point, normalized by the generating-function identity
// J0(z) + 2 sum_{k>=1} J_{2k}(z) = 1, which holds for all complex z.
//
// M = 2*floor(0.75|z| + 16) is comfortably past the turning point over this
// branch's whole window (|z| < 25): J_M(z) ~ (e|z| / 2M)^M / sqrt(2 pi M)
// there, which is 1e-27 relative at |z| = 8 (M = 44) and 1e-20 at |z| = 25
// (M = 68) -- both far below the double's reach. 2/z is hoisted; a divide
// inside the recurrence cost more than the rest of the loop put together.
template <class T>
static inline void mw_bessel_miller(const T &z, double az, T &j0, T &j1) {
    const int M = 2 * static_cast<int>(0.75 * az + 16.0);
    const T inv2z = T(2.0) / z;
    T fp1(0.0), fn(1e-280);
    T s(0.0), j0v(0.0), j1v(0.0);
    for (int n = M; n >= 1; --n) {
        if ((n & 1) == 0) s += 2.0 * fn;  // fn carries index n
        const T fm1 = (static_cast<double>(n) * inv2z) * fn - fp1;
        fp1 = fn;
        fn = fm1;  // fn now carries index n - 1
        if (n == 2) j1v = fn;
        if (n == 1) j0v = fn;
        if (mw_mag(fn) > 1e250) {
            const double sc = 1e-250;
            fn *= sc;
            fp1 *= sc;
            s *= sc;
            j1v *= sc;
            j0v *= sc;
        }
    }
    s += j0v;
    const T inv_s = T(1.0) / s;
    j0 = j0v * inv_s;
    j1 = j1v * inv_s;
}

// Hankel's P/Q asymptotic coefficients for order nu, mu = 4 nu^2
// (A&S 9.2.9-9.2.10), summed by the term recursion
// c_k = c_{k-1} (mu - (2k-1)^2) / (8 k z), P = c0 - c2 + c4 - ...,
// Q = c1 - c3 + c5 - ...; truncated at the smallest term.
template <class T>
static inline void mw_bessel_pq(const T &z, double mu, T &P, T &Q) {
    P = T(1.0);
    Q = T(0.0);
    T c(1.0);
    const T iz = T(1.0) / z;
    double prev = std::numeric_limits<double>::infinity();
    for (int k = 1; k <= 40; ++k) {
        const double kk = 2.0 * k - 1.0;
        c *= ((mu - kk * kk) * 0.125 * MW_INV[k]) * iz;
        const double m = mw_norm2(c);  // squared: no hypot in the inner loop
        if (!(m < prev)) break;        // the series has started to diverge
        prev = m;
        if (k & 1) {
            if ((((k - 1) / 2) & 1) == 0) Q += c;
            else Q -= c;
        } else {
            if (((k / 2) & 1) == 0) P += c;
            else P -= c;
        }
        if (m < 1e-40) break;
    }
}

template <class T>
static inline void mw_bessel_asymptotic(const T &z, T &j0, T &j1) {
    T P0, Q0, P1, Q1;
    mw_bessel_pq(z, 0.0, P0, Q0);
    mw_bessel_pq(z, 4.0, P1, Q1);
    const T cz = std::cos(z);
    const T sz = std::sin(z);
    const T pref = std::sqrt(T(1.0) / (MW_PI * z));
    // J0 = sqrt(2/(pi z)) [P0 cos(z - pi/4) - Q0 sin(z - pi/4)] with
    // cos(z - pi/4) = (cos z + sin z)/sqrt2, sin(z - pi/4) = (sin z - cos z)/sqrt2.
    j0 = pref * ((P0 + Q0) * cz + (P0 - Q0) * sz);
    // J1 = sqrt(2/(pi z)) [P1 cos(z - 3pi/4) - Q1 sin(z - 3pi/4)] with
    // cos(z - 3pi/4) = (sin z - cos z)/sqrt2, sin(z - 3pi/4) = -(sin z + cos z)/sqrt2.
    j1 = pref * ((P1 + Q1) * sz + (Q1 - P1) * cz);
}

// The regime switch, shared by both scalars. `nz` is |z|^2.
template <class T>
static inline void mw_bessel_j0_j1(const T &z, double nz, T &b0, T &b1) {
    if (nz < 64.0) mw_bessel_series(z, b0, b1);
    else if (nz < 625.0) mw_bessel_miller(z, std::sqrt(nz), b0, b1);
    else mw_bessel_asymptotic(z, b0, b1);
}

// (J0(x), J1(x)/x). The small-|x| branch is `_bessel_j0_j1x`'s, spelled the
// same way and switching at the same 1e-6.
static inline void bessel_j0_j1x(const cd &x, cd &j0, cd &j1x) {
    const double nx = mw_norm2(x);  // |x|^2 -- the branch thresholds squared
    if (nx < 1e-12) {
        const cd xx = x * x;
        j0 = 1.0 - 0.25 * xx;
        j1x = 0.5 - xx / 16.0;
        return;
    }
    if (x.imag() == 0.0) {  // the tail's abscissa, by construction
        const double ax = std::fabs(x.real());  // both results are even in x
        double b0, b1;
        mw_bessel_j0_j1(ax, nx, b0, b1);
        j0 = cd(b0, 0.0);
        j1x = cd(b1 / ax, 0.0);
        return;
    }
    const cd z = (x.real() < 0.0) ? -x : x;
    cd b0, b1;
    mw_bessel_j0_j1(z, nx, b0, b1);
    j0 = b0;
    j1x = b1 / z;
}

// ---------------------------------------------------------------------------
// The engine
// ---------------------------------------------------------------------------

// What `_run_contour` reports alongside the value: enough for a caller to tell
// a converged answer from a lucky one.
struct ContourHealth {
    int head_panels;
    int tail_panels;
    bool converged;
    bool accel;
};

// `_gauss_segment`: f(mid + half*gx) @ (gw * half), vector-valued.
// `G` is a real-abscissa integrand: void operator()(double t, cd *out) const.
template <int NC, class G>
static inline void gauss_segment(const G &g, double z0, double z1,
                                 const double *gx, const double *gw, int ng,
                                 cd *out) {
    const double mid = 0.5 * (z0 + z1);
    const double half = 0.5 * (z1 - z0);
    for (int c = 0; c < NC; ++c) out[c] = cd(0.0, 0.0);
    cd fv[NC];
    for (int i = 0; i < ng; ++i) {
        g(mid + half * gx[i], fv);
        const double w = gw[i] * half;
        for (int c = 0; c < NC; ++c) out[c] += fv[c] * w;
    }
}

// `_adaptive_segment`: recursive bisection Gauss on z0 -> z1.
//
// `whole` may be null (the numpy `whole=None` default), in which case the
// undivided rule is evaluated here. `ref` floors the relative test: on a
// high-loss ground the remainder can sit decades below a neighbouring
// section's magnitude and a purely local relative target then chases noise.
template <int NC, class G>
static void adaptive_segment(const G &g, double z0, double z1, double rtol,
                             int depth, const double *gx, const double *gw,
                             int ng, const cd *whole, double ref, cd *out) {
    cd wh[NC];
    if (whole == 0) {
        gauss_segment<NC>(g, z0, z1, gx, gw, ng, wh);
        whole = wh;
    }
    const double mid = 0.5 * (z0 + z1);
    cd left[NC], right[NC], better[NC];
    gauss_segment<NC>(g, z0, mid, gx, gw, ng, left);
    gauss_segment<NC>(g, mid, z1, gx, gw, ng, right);
    double err = 0.0, bmax = 0.0;
    for (int c = 0; c < NC; ++c) {
        better[c] = left[c] + right[c];
        const double e = std::abs(better[c] - whole[c]);
        if (e > err) err = e;
        const double b = std::abs(better[c]);
        if (b > bmax) bmax = b;
    }
    const double scale = std::max(bmax, ref);
    if (depth <= 0 || err <= rtol * std::max(scale, 1e-300)) {
        for (int c = 0; c < NC; ++c) out[c] = better[c];
        return;
    }
    cd la[NC], rb[NC];
    adaptive_segment<NC>(g, z0, mid, rtol, depth - 1, gx, gw, ng, left, ref, la);
    adaptive_segment<NC>(g, mid, z1, rtol, depth - 1, gx, gw, ng, right, ref, rb);
    for (int c = 0; c < NC; ++c) out[c] = la[c] + rb[c];
}

// `_wynn_epsilon`: Wynn's epsilon over a sequence of complex vectors; the best
// even column, elementwise. `s` is (n, NC) row-major, n <= MW_TAIL_KEEP.
template <int NC>
static inline void wynn_epsilon(const cd *s, int n, cd *best) {
    for (int c = 0; c < NC; ++c) best[c] = s[(n - 1) * NC + c];
    if (n < 5) return;
    cd prev[MW_TAIL_KEEP * NC];
    cd cur[MW_TAIL_KEEP * NC];
    cd nxt[MW_TAIL_KEEP * NC];
    for (int i = 0; i < n * NC; ++i) {
        prev[i] = cd(0.0, 0.0);
        cur[i] = s[i];
    }
    int m = n;
    for (int k = 1; k < n; ++k) {
        for (int i = 0; i + 1 < m; ++i) {
            for (int c = 0; c < NC; ++c) {
                cd d = cur[(i + 1) * NC + c] - cur[i * NC + c];
                if (std::abs(d) < 1e-300) d = cd(1e-300, 0.0);
                nxt[i * NC + c] = prev[i * NC + c] + cd(1.0, 0.0) / d;
            }
        }
        const int mm = m - 1;
        for (int i = 0; i < mm * NC; ++i) prev[i] = cur[i];
        for (int i = 0; i < mm * NC; ++i) cur[i] = nxt[i];
        m = mm;
        if ((k % 2) == 0 && m >= 1) {
            bool ok = true;
            for (int c = 0; c < NC; ++c)
                if (!mw_finite(cur[(m - 1) * NC + c])) ok = false;
            if (ok)
                for (int c = 0; c < NC; ++c) best[c] = cur[(m - 1) * NC + c];
        }
        if (m <= 1) break;
    }
}

// The head's parametrized integrand: lam(t) = t + jH sin(pi t / a), carrying
// its own dlam/dt. `F` is the caller's integrand,
// void operator()(cd lam, cd *out) const.
template <int NC, class F>
struct HeadWrap {
    const F &f;
    double a;
    double H;
    HeadWrap(const F &f_, double a_, double H_) : f(f_), a(a_), H(H_) {}
    void operator()(double t, cd *out) const {
        const double s = std::sin(MW_PI * t / a);
        const double c = std::cos(MW_PI * t / a);
        const cd lam(t, H * s);
        const cd dl(1.0, H * (MW_PI / a) * c);
        f(lam, out);
        for (int i = 0; i < NC; ++i) out[i] *= dl;
    }
};

// The tail runs along the real axis; the caller's integrand still takes a
// complex lam (numpy's `np.asarray(lam, dtype=complex128)` gives +0.0 imag).
template <int NC, class F>
struct TailWrap {
    const F &f;
    explicit TailWrap(const F &f_) : f(f_) {}
    void operator()(double z, cd *out) const { f(cd(z, 0.0), out); }
};

// `_head`: the integral over [0, a], deformed into the FIRST quadrant.
// Returns the panel count; `marks` are the branch points' real parts.
template <int NC, class F>
static inline int head_contour(const F &f, double a, double rho,
                               const double *marks, int nmarks, double rtol,
                               int depth, double detour, const double *gx,
                               const double *gw, int ng, cd *out) {
    double H = std::min(0.35 * a, detour / std::max(rho, 1e-12));
    H = std::max(H, 1e-6 * a);
    const HeadWrap<NC, F> g(f, a, H);

    // 2 endpoints + 6 sevenths + 5 seeds per mark. `_head` builds a Python set
    // and sorts it; sort + exact-equality unique is the same thing.
    const int MAXE = 8 + 5 * 4;
    double e[8 + 5 * 4];
    int ne = 0;
    e[ne++] = 0.0;
    e[ne++] = a;
    for (int i = 1; i < 7; ++i) e[ne++] = a * i / 7.0;
    const double ws[5] = {0.0, -0.15, 0.15, -0.4, 0.4};
    for (int m = 0; m < nmarks && ne + 5 <= MAXE; ++m) {
        for (int w = 0; w < 5; ++w) {
            const double v = marks[m] * (1.0 + ws[w]);
            if (v > 0.0 && v < a) e[ne++] = v;
        }
    }
    std::sort(e, e + ne);
    ne = static_cast<int>(std::unique(e, e + ne) - e);

    cd part[NC];
    for (int c = 0; c < NC; ++c) out[c] = cd(0.0, 0.0);
    for (int i = 0; i + 1 < ne; ++i) {
        adaptive_segment<NC>(g, e[i], e[i + 1], rtol, depth, gx, gw, ng, 0, 0.0,
                             part);
        if (i == 0)
            for (int c = 0; c < NC; ++c) out[c] = part[c];
        else
            for (int c = 0; c < NC; ++c) out[c] += part[c];
    }
    return ne - 1;
}

// `_tail_below`: the integral from a to infinity along the real axis,
// panelled at the J0(lam rho) zeros. See the header comment for the three
// guards; each one is load-bearing.
template <int NC, class F>
static inline void tail_below(const F &f, double a, double rho, double h,
                              const cd *ref, double rtol, const double *gx,
                              const double *gw, int ng, int max_panels,
                              cd *out, int *parts_out, bool *converged_out,
                              bool *accel_out) {
    const double INF = std::numeric_limits<double>::infinity();
    double dl_decay = (h > 0.0) ? (1.5 / h) : INF;
    if (!std::isfinite(dl_decay)) dl_decay = std::max(1.0, 10.0 * a);
    const double lat = (rho > 0.0) ? (MW_PI / rho) : INF;
    const double off = (rho > 0.0) ? (-0.25 * MW_PI / rho) : 0.0;
    const TailWrap<NC, F> g(f);

    double ref_scale = 0.0;
    if (ref != 0)
        for (int c = 0; c < NC; ++c)
            ref_scale = std::max(ref_scale, std::abs(ref[c]));

    cd acc[NC];
    for (int c = 0; c < NC; ++c) acc[c] = cd(0.0, 0.0);
    cd contrib[NC];
    cd ring[MW_TAIL_KEEP * NC];

    double z = a;
    int parts = 0, quiet = 0;
    bool converged = false;
    while (parts < max_panels) {
        double nz = INF;
        if (std::isfinite(lat)) {
            // The `+ 1e-9` is the zero-length-panel guard: without it a z
            // sitting exactly on the lattice floors back onto itself.
            const double n = std::floor((z - off) / lat + 1e-9) + 1.0;
            nz = off + n * lat;
        }
        const double znext = std::min(nz, z + dl_decay);
        adaptive_segment<NC>(g, z, znext, rtol, 6, gx, gw, ng, 0, ref_scale,
                             contrib);
        for (int c = 0; c < NC; ++c) acc[c] += contrib[c];
        {
            cd *slot = ring + (parts % MW_TAIL_KEEP) * NC;
            for (int c = 0; c < NC; ++c) slot[c] = acc[c];
        }
        z = znext;
        ++parts;
        double cmax = 0.0, smax = 0.0;
        for (int c = 0; c < NC; ++c) {
            cmax = std::max(cmax, std::abs(contrib[c]));
            smax = std::max(smax, std::abs(acc[c]));
        }
        smax = std::max(std::max(smax, ref_scale), 1e-300);
        if (cmax == 0.0 || cmax < rtol * smax) {
            ++quiet;
            if (quiet >= 2) {
                converged = true;
                break;
            }
        } else {
            quiet = 0;
        }
    }

    bool accel = false;
    if (!converged && parts >= 8) {
        const int keep = std::min(parts, MW_TAIL_KEEP);
        cd hist[MW_TAIL_KEEP * NC];
        for (int i = 0; i < keep; ++i) {
            const cd *slot = ring + ((parts - keep + i) % MW_TAIL_KEEP) * NC;
            for (int c = 0; c < NC; ++c) hist[i * NC + c] = slot[c];
        }
        wynn_epsilon<NC>(hist, keep, acc);
        accel = true;
    }

    for (int c = 0; c < NC; ++c) out[c] = acc[c];
    *parts_out = parts;
    *converged_out = converged;
    *accel_out = accel;
}

// `_run_contour`: head + tail for ONE contour.
//
// `h` is the TAIL'S DECAY LENGTH, not a coordinate -- the below/below family
// passes |z| + |z'|, the transmitted family passes |z'| + z. `max_panels` is
// the transmitted family's handle; its grazing rows need a bigger budget than
// any below/below geometry does.
template <int NC, class F>
static inline ContourHealth run_contour(const F &f, double k_p, const cd &k_m,
                                        double rho, double h, double rtol,
                                        int depth, double detour,
                                        const double *gx, const double *gw,
                                        int ng, int max_panels, cd *out) {
    const double a = 1.1 * std::max(std::abs(k_p), std::abs(k_m));
    const double marks[2] = {k_p, std::abs(k_m.real())};
    cd head[NC], tail[NC];
    ContourHealth hh;
    hh.head_panels = head_contour<NC>(f, a, rho, marks, 2, rtol, depth, detour,
                                      gx, gw, ng, head);
    tail_below<NC>(f, a, rho, h, head, rtol, gx, gw, ng, max_panels, tail,
                   &hh.tail_panels, &hh.converged, &hh.accel);
    for (int c = 0; c < NC; ++c) out[c] = head[c] + tail[c];
    return hh;
}

}  // namespace mw_contour
