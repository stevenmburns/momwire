// The two-radius exact ring kernel K2(z; a1, a2) (momwire#1438): the ring
// kernel between coaxial rings of two radii, near and far, in C++. Shared by
// the point moments across a radius step (`_exact_kernel_cross_inline.h`,
// razor-2p's) and the Galerkin pair moments across one
// (`_exact_kernel_coaxial_cross_inline.h`, bs2's, momwire#1615), so the two
// formulations evaluate K2 with one text.
//
// Included INSIDE each TU's namespace, after `_exact_kernel_ring_inline.h`
// (agm_KE, near_terms, far_polys, the constants); it includes nothing itself.

// K2 near the source ring, |z| < T_SWITCH sqrt(a1 a2): the one-radius near
// series at rho^2 = z^2 + (a1 + a2)^2, m = 4 a1 a2 / rho^2,
// m1 = (z^2 + (a1 - a2)^2) / rho^2 (`_exact_kernel._near2`).
static cd near_eval2(double z, double a1, double a2, double k) {
    const double s = a1 + a2, dl = a1 - a2;
    const double rho2 = z * z + s * s;
    const double rho = std::sqrt(rho2);
    const double m = 4.0 * a1 * a2 / rho2;
    const double m1 = (z * z + dl * dl) / rho2;
    double Km, Em;
    agm_KE(m1, Km, Em);
    const int N = near_terms(k * rho);
    const int nmax = (N - 1) / 2 + 2;
    double Rh[48], Ji[48];
    Rh[0] = Km;
    Rh[1] = Em;
    Ji[0] = PI_ / 2.0;
    Ji[1] = (2.0 - m) * PI_ / 4.0;
    for (int i = 1; i < nmax; ++i) {
        double nu = i - 0.5;
        const double c0 = 2.0 * nu * (m - 1.0), c1 = (2.0 * nu + 1.0) * (2.0 - m),
                     c2 = 2.0 * nu + 2.0;
        Rh[i + 1] = (c0 * Rh[i - 1] + c1 * Rh[i]) / c2;
        nu = static_cast<double>(i);
        Ji[i + 1] = (2.0 * nu * (m - 1.0) * Ji[i - 1] + (2.0 * nu + 1.0) * (2.0 - m) * Ji[i]) /
                    (2.0 * nu + 2.0);
    }
    cd reg(0.0, 0.0);
    cd coef(1.0, 0.0);
    const cd mjk(0.0, -k);
    double rp = 1.0 / rho;  // rho^{n-1}
    for (int n = 0; n <= N; ++n) {
        if ((n & 1) == 0)
            reg += coef * (rp * Rh[n / 2]);
        else
            reg += coef * (rp * Ji[(n - 1) / 2]);
        coef = coef * mjk / static_cast<double>(n + 1);
        rp = rp * rho;
    }
    return reg * (2.0 / (4.0 * PI_ * PI_));
}

// K2 far from it: the series about s0 = z^2 + a1^2 + a2^2 in powers of
// a1 a2 (`_exact_kernel._far2`).
static cd far_eval2(double z, double a1, double a2, double k) {
    const FarPolys &P = far_polys();
    const double s0 = z * z + a1 * a1 + a2 * a2;
    const double r = std::sqrt(s0);
    const double p = a1 * a2;
    const cd x(0.0, k * r);
    cd acc(0.0, 0.0);
    for (int n = 0; n <= 8; n += 2) {
        cd y(0.0, 0.0);
        for (int i = n; i >= 0; --i) y = y * x + P.c[n][i];
        acc += (P.C[n] * std::pow(p, n)) * y / std::pow(r, 2 * n + 1);
    }
    return std::exp(-x) * acc / (4.0 * PI_);
}

// `_exact_kernel.ring_kernel2`: the observer's ring of radius a1, the
// source's of radius a2, at axial separation z.
static inline cd ring_kernel2(double z, double a1, double a2, double k) {
    z = std::fabs(z);
    if (z < T_SWITCH * std::sqrt(a1 * a2)) return near_eval2(z, a1, a2, k);
    return far_eval2(z, a1, a2, k);
}
