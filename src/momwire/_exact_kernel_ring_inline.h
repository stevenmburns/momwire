// The exact ring kernel K(z), its exact log split, and the quadrature rules
// handed in from Python: shared by the coaxial pair-moment TU
// (`_accel_exact_kernel.cpp`, momwire#1410) and the point-moment TU
// (`_accel_exact_kernel_point.cpp`, momwire#1433). Textually what the 1410 TU
// held at this point, so its preprocessed token stream (and codegen) is
// unchanged. Included INSIDE each TU's namespace; it includes nothing itself
// (the TU has <complex>, <cmath>, <cstring>, <vector> via _accel_common.h).
typedef std::complex<double> cd;

static const int MAXND = 6;           // degree <= 5
static const int NM = MAXND * MAXND;  // a moment matrix, stride MAXND
static const double T_SWITCH = 6.0;   // near form below |z|/a, far series at and above
static const double GRADE = 3.0;      // geometric chunk ratio
static const double PI_ = 3.14159265358979323846;

// ---------------------------------------------------------------------------
// quadrature rules, handed in from Python (`_gl01` / `_gl01_log`) so both
// routes integrate on the same nodes
// ---------------------------------------------------------------------------
struct Rules {
    std::vector<double> xg, wg;       // Gauss-Legendre on [0, 1], smooth pieces
    std::vector<double> xs, ws, wls;  // the split rule: GL nodes, GL and log weights
};

// ---------------------------------------------------------------------------
// elliptic integrals by the AGM: K(m), E(m) at m = 1 - m1, m1 passed exactly
// ---------------------------------------------------------------------------
static inline void agm_KE(double m1, double &K, double &E) {
    double a = 1.0, b = std::sqrt(m1), s = 0.5 * (1.0 - m1), p = 0.5;
    for (int it = 0; it < 40; ++it) {
        const double c = 0.5 * (a - b);
        const double an = 0.5 * (a + b);
        b = std::sqrt(a * b);
        a = an;
        p *= 2.0;
        s = s + p * c * c;
        if (std::fabs(c) <= 1e-17 * a) break;
    }
    K = PI_ / (2.0 * a);
    E = K * (1.0 - s);
}

static inline int near_terms(double kr) {
    int n = 0;
    double term = 1.0;
    while (term > 1e-17 && n < 80) {
        ++n;
        term *= kr / n;
    }
    return n < 4 ? 4 : n;
}

// The near closed form at z >= 0. split=false: A <- K. split=true: K = A ln z + B.
static void near_eval(double z, double a, double k, bool split, cd &A, cd &B) {
    const double rho2 = z * z + 4.0 * a * a;
    const double rho = std::sqrt(rho2);
    const double m = 4.0 * a * a / rho2;
    const double m1 = z * z / rho2;
    double Km, Em;
    agm_KE(m1, Km, Em);
    const int N = near_terms(k * rho);
    const int nmax = (N - 1) / 2 + 2;
    double Rh[48], Lh[48], Ji[48];
    if (split) {
        double Kc, Ec;
        agm_KE(m, Kc, Ec);
        Lh[0] = -Kc / PI_;
        Lh[1] = -(Kc - Ec) / PI_;
        if (m1 == 0.0) {
            Rh[0] = std::log(4.0);
            Rh[1] = 1.0;
        } else {
            const double lnm1 = std::log(m1);
            Rh[0] = Km - Lh[0] * lnm1;
            Rh[1] = Em - Lh[1] * lnm1;
        }
    } else {
        Rh[0] = Km;
        Rh[1] = Em;
    }
    Ji[0] = PI_ / 2.0;
    Ji[1] = (2.0 - m) * PI_ / 4.0;
    for (int i = 1; i < nmax; ++i) {
        double nu = i - 0.5;
        const double c0 = 2.0 * nu * (m - 1.0), c1 = (2.0 * nu + 1.0) * (2.0 - m),
                     c2 = 2.0 * nu + 2.0;
        Rh[i + 1] = (c0 * Rh[i - 1] + c1 * Rh[i]) / c2;
        if (split) Lh[i + 1] = (c0 * Lh[i - 1] + c1 * Lh[i]) / c2;
        nu = static_cast<double>(i);
        Ji[i + 1] = (2.0 * nu * (m - 1.0) * Ji[i - 1] + (2.0 * nu + 1.0) * (2.0 - m) * Ji[i]) /
                    (2.0 * nu + 2.0);
    }
    cd reg(0.0, 0.0), lg(0.0, 0.0);
    cd coef(1.0, 0.0);
    const cd mjk(0.0, -k);
    double rp = 1.0 / rho;  // rho^{n-1}
    for (int n = 0; n <= N; ++n) {
        if ((n & 1) == 0) {
            reg += coef * (rp * Rh[n / 2]);
            if (split) lg += coef * (rp * Lh[n / 2]);
        } else {
            reg += coef * (rp * Ji[(n - 1) / 2]);
        }
        coef = coef * mjk / static_cast<double>(n + 1);
        rp = rp * rho;
    }
    const double scale = 2.0 / (4.0 * PI_ * PI_);
    if (!split) {
        A = reg * scale;
        B = cd(0.0, 0.0);
        return;
    }
    A = 2.0 * lg * scale;
    B = (reg - lg * std::log(rho2)) * scale;
}

// P_n with d^n/ds^n [e^{-jk sqrt s}/sqrt s] = e^{-x} P_n(x) / r^{2n+1}, x = jkr;
// ascending coefficients. Dyadic rationals, exact in double.
struct FarPolys {
    double c[9][10];
    double C[9];  // <eps^n>/n! in units of a^{2n}
    FarPolys() {
        std::memset(c, 0, sizeof(c));
        c[0][0] = 1.0;
        for (int n = 0; n < 8; ++n) {
            for (int i = 0; i <= n + 1; ++i) {
                const double xd = (i <= n) ? i * c[n][i] : 0.0;           // x P'
                const double xp = (i >= 1) ? c[n][i - 1] : 0.0;           // x P
                const double cp = (i <= n) ? (2 * n + 1) * c[n][i] : 0.0;  // (2n+1) P
                c[n + 1][i] = (xd - (xp + cp)) / 2.0;
            }
        }
        for (int n = 0; n <= 8; ++n) {
            double comb = 1.0, fact = 1.0;
            for (int i = 1; i <= n; ++i) fact *= i;
            for (int i = 0; i < n / 2; ++i) comb = comb * (n - i) / (i + 1);
            C[n] = std::pow(2.0, n) * comb / std::pow(2.0, n) / fact;
        }
    }
};

static const FarPolys &far_polys() {
    static const FarPolys P;  // thread-safe initialisation (C++11)
    return P;
}

static inline cd far_eval(double z, double a, double k) {
    const FarPolys &P = far_polys();
    const double s0 = z * z + 2.0 * a * a;
    const double r = std::sqrt(s0);
    const cd x(0.0, k * r);
    cd acc(0.0, 0.0);
    for (int n = 0; n <= 8; n += 2) {
        cd y(0.0, 0.0);
        for (int i = n; i >= 0; --i) y = y * x + P.c[n][i];
        acc += (P.C[n] * std::pow(a, 2 * n)) * y / std::pow(r, 2 * n + 1);
    }
    return std::exp(-x) * acc / (4.0 * PI_);
}

static inline cd ring_kernel(double z, double a, double k) {
    z = std::fabs(z);
    if (z < T_SWITCH * a) {
        cd K, unused;
        near_eval(z, a, k, false, K, unused);
        return K;
    }
    return far_eval(z, a, k);
}
