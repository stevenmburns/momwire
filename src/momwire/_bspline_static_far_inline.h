// Far-field spelling of the same-edge static moments — momwire#808.
// The C++ twin of `_bspline_static_far.py`, written operation for operation
// against it. That module's docstring carries the derivation, the measured
// accuracy either side of the switch, and why the switch ratio is read off
// the kernel's poles rather than fitted.
//
// Hand-written, NOT generated: `_bspline_static_moments_inline.h` is sympy's
// closed form and stays exactly as it is. This is the other regime, and the
// two are dispatched between by `far_ratio` below.
#pragma once

#include <cmath>

// Series terms, and the switch. Both must match `_bspline_static_far.py`'s
// N_TERMS and FAR_RATIO — the two lanes agree only if they truncate in the
// same place and change regime on the same pairs.
static constexpr int BSPLINE_FAR_TERMS = 64;
static constexpr double BSPLINE_FAR_RATIO = 0.5;

// The domain the twin lanes agree to share. `_bspline_static_far.py` reads it
// from the generated `_bspline_static_moments.MAX_D` and raises outside
// [0, MAX_P]^2; this is the C++ copy of that number.
//
// The series is GENERIC in p -- the centred moments below expand (u + h/2)^p
// for any p -- so this bounds the two spellings' domains equally rather than
// bounding the derivation. That is exactly why it is dangerous to mirror by
// hand: momwire#883 raised the python side to 3 and this file kept a 3x3
// table, and nothing said so. `_accelerators.BSPLINE_FAR_MAX_P` exports it and
// a test pins the two equal, so the next bump fails a gate instead of aiming a
// read past an array.
static constexpr int BSPLINE_FAR_MAX_P = 3;

// r = (h1+h2)/2 / sqrt(xi0^2 + a^2), the ratio the radius of convergence
// sets. Below 1 the expansion converges; the dispatch takes it at half that.
static inline double bspline_far_ratio(double alpha, double beta, double A,
                                       double B, double a) {
    const double h1 = beta - alpha;
    const double h2 = B - A;
    const double xi0 = 0.5 * (alpha + beta) - 0.5 * (A + B);
    return 0.5 * (h1 + h2) / std::sqrt(xi0 * xi0 + a * a);
}

// M[m] = int_0^h u^p (u - h/2)^m du, every surviving term positive and the
// odd ones identically zero. `_centred_moments` in the numpy twin.
//
// C(p, j) is STEPPED in j, not tabulated: C(p, 0) = 1 and
// C(p, j) = C(p, j-1) * (p - j + 1) / j. Every partial product is itself a
// binomial coefficient, so it is an exact integer well under 2^53 for any p
// this family reaches, and the doubles agree with the numpy twin's exact
// `math.comb` bit for bit. The outer coefficient loop in `bspline_J_static_far`
// already steps C(n, m) the same way, for the same reason.
//
// This was a `static const double binom[3][3]` until momwire#999, carrying the
// comment "C(p, j) for p <= 2, which is what the generated family covers" --
// true when written, false from momwire#883 onward, which raised the generated
// family to 3. `binom[p][j]` at p = 3 read a row past the array: undefined
// behaviour, no throw, and no bound check anywhere on the path, because
// `J_static_dispatch` reaches this through its far-ratio early return BEFORE
// its own switch. It was unreachable only because `_BSPLINE_ACCEL_MAX_D = 2`,
// three modules away, happened to stop degree 3 first -- i.e. the gate that
// made it safe is the gate that raising the degree axis removes. A spelling
// with no array in it cannot be indexed out of.
static inline void bspline_centred_moments(int p, double h, int nmax,
                                           double *out) {
    for (int m = 0; m <= nmax; m++) {
        double acc = 0.0;
        double c_pj = 1.0;  // C(p, j), stepped with j below
        for (int j = 0; j <= p; j++) {
            // Step BEFORE the parity skip: `continue` must not skip the
            // recurrence, or every j after the first odd k is wrong.
            if (j > 0) c_pj = c_pj * (double)(p - j + 1) / (double)j;
            const int k = m + j;
            if (k % 2) continue;
            acc += c_pj * std::pow(h / 2.0, p - j) *
                   std::pow(h, k + 1) / (std::pow(2.0, k) * (k + 1));
        }
        out[m] = acc;
    }
}

// J_pq by the centred multipole series. Correct only where `bspline_far_ratio`
// is at or under BSPLINE_FAR_RATIO; the caller checks.
static inline double bspline_J_static_far(int p, int q, double alpha,
                                          double beta, double A, double B,
                                          double a) {
    const double h1 = beta - alpha;
    const double h2 = B - A;
    const double xi0 = 0.5 * (alpha + beta) - 0.5 * (A + B);
    const double den = xi0 * xi0 + a * a;

    // g[n] = f^(n)(xi0)/n!, the factorial carried INSIDE the recurrence:
    //   g[n+1] = -[ (2n+1) xi0 g[n] + n g[n-1] ] / ((n+1)(xi0^2+a^2))
    // f^(n) alone reaches 1e174 at 64 terms on a 401-segment edge, so forming
    // it and dividing by 64! afterwards would put the arithmetic in the one
    // place it can overflow. See the numpy twin's docstring.
    double g[BSPLINE_FAR_TERMS + 2];
    g[0] = 1.0 / std::sqrt(den);
    g[1] = -xi0 / (den * std::sqrt(den));
    for (int n = 1; n <= BSPLINE_FAR_TERMS; n++) {
        g[n + 1] = -((2 * n + 1) * xi0 * g[n] + (double)n * g[n - 1]) /
                   (den * (double)(n + 1));
    }

    double Mp[BSPLINE_FAR_TERMS + 1], Mq[BSPLINE_FAR_TERMS + 1];
    bspline_centred_moments(p, h1, BSPLINE_FAR_TERMS, Mp);
    bspline_centred_moments(q, h2, BSPLINE_FAR_TERMS, Mq);

    double total = 0.0;
    for (int n = 0; n <= BSPLINE_FAR_TERMS; n++) {
        double coef = 0.0;
        double c_nm = 1.0;  // C(n, m), stepped in m as the numpy twin steps it
        for (int m = 0; m <= n; m++) {
            if (m > 0) c_nm = c_nm * (double)(n - m + 1) / (double)m;
            const double term = c_nm * Mp[m] * Mq[n - m];
            coef += ((n - m) % 2) ? -term : term;
        }
        total += g[n] * coef;
    }
    return total;
}
