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
static inline void bspline_centred_moments(int p, double h, int nmax,
                                           double *out) {
    // C(p, j) for p <= 2, which is what the generated family covers.
    static const double binom[3][3] = {{1, 0, 0}, {1, 1, 0}, {1, 2, 1}};
    for (int m = 0; m <= nmax; m++) {
        double acc = 0.0;
        for (int j = 0; j <= p; j++) {
            const int k = m + j;
            if (k % 2) continue;
            acc += binom[p][j] * std::pow(h / 2.0, p - j) *
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
