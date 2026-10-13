#include "_accel_common.h"

#include <cstring>
#include <stdexcept>
#ifdef _OPENMP
#include <omp.h>
#endif

// momwire#1454 (step 3): the exact ring kernel's point moments as a power
// series in k with geometry-only coefficients.
//
// A port of `_exact_kernel_series.py`, which is the reference and carries the
// mathematics (derivation: docs/design/exact-kernel-static-dynamic-split.md):
//
//     N_q(c, h; k) = e^{-jk Rc} sum_n (-jk)^n / n! * mu_{n,q}(c, h),
//     mu_{n,q}(c, h) = (1/4pi) integral_0^h v^q <(R - Rc)^n / R>(c + v) dv,
//
// Rc = 0 (n = 0 the static kernel, n >= 1 the dynamic remainder) for a pair
// near the observer or inside 6a, the middle of the pair's range of R for one
// that is neither, so its series runs in k (R - Rc). Every mu is real and
// a function of the pair's geometry alone, so `PointSeriesAccel` holds them
// for its lifetime and a wavenumber costs one Horner evaluation per unique
// pair. The pair key, its mirror and the first-met cache are the 1433 point
// TU's (`_accel_exact_kernel_point.cpp`), with k moved from the constructor
// to the call.
//
// NOT BIT-EXACT WITH THE NUMPY ROUTE, by the 1410 decision: the two are held
// at a derived tolerance (tests/test_exact_kernel_series_1454.py). Every
// value is a function of its own pair alone, so a run is bit-stable whatever
// the thread count.
//
// Its own TU so that no existing kernel's codegen can move with it
// (momwire#1187); the shared headers are included verbatim.

namespace exact_kernel_series_1454 {

#include "_exact_kernel_ring_inline.h"

#include "_exact_kernel_table_inline.h"

static const int NCAP = 40;  // `_exact_kernel_series.N_CAP`: terms n = 0..NCAP-1

// N with x^N/N! <= tol (at least 4): the series runs n = 0..N.
static inline int n_terms(double x, double tol) {
    int n = 0;
    double term = 1.0;
    while (term > tol && n < NCAP - 1) {
        ++n;
        term *= x / n;
    }
    return n < 4 ? 4 : n;
}

// binom((n-1)/2, i) * C(i, i/2), i = 0, 2, .., 8.
struct FarBinom {
    double b[NCAP][5];
    FarBinom() {
        for (int n = 0; n < NCAP; ++n) {
            const double nu = 0.5 * (n - 1);
            double bin = 1.0, cc = 1.0;  // binom(nu, i), C(i, i/2)
            for (int i = 0; i < 9; ++i) {
                if ((i & 1) == 0) {
                    if (i) cc = cc * (i - 1) * 4.0 / i;  // C(i, i/2) from C(i-2, i/2-1)
                    b[n][i / 2] = bin * cc;
                }
                bin = bin * (nu - i) / (i + 1);
            }
        }
    }
};

static const FarBinom &far_binom() {
    static const FarBinom B;  // thread-safe initialisation (C++11)
    return B;
}

// P[n] = <R^{n-1}> / (4 pi) at |z|, n = 0..N.
static void ring_powers(double z, double a, int N, double *P) {
    z = std::fabs(z);
    if (z < T_SWITCH * a) {
        const double rho2 = z * z + 4.0 * a * a;
        const double rho = std::sqrt(rho2);
        const double m = 4.0 * a * a / rho2;
        double Rh[NCAP / 2 + 3], Ji[NCAP / 2 + 3];
        agm_KE(z * z / rho2, Rh[0], Rh[1]);
        Ji[0] = PI_ / 2.0;
        Ji[1] = (2.0 - m) * PI_ / 4.0;
        const int nmax = (N - 1) / 2 + 2;
        for (int i = 1; i < nmax; ++i) {
            double nu = i - 0.5;
            const double c0 = 2.0 * nu * (m - 1.0), c1 = (2.0 * nu + 1.0) * (2.0 - m),
                         c2 = 2.0 * nu + 2.0;
            Rh[i + 1] = (c0 * Rh[i - 1] + c1 * Rh[i]) / c2;
            nu = static_cast<double>(i);
            Ji[i + 1] = (2.0 * nu * (m - 1.0) * Ji[i - 1] + (2.0 * nu + 1.0) * (2.0 - m) * Ji[i]) /
                        (2.0 * nu + 2.0);
        }
        const double scale = 2.0 / (4.0 * (PI_ * PI_));
        double rp = 1.0 / rho;
        for (int n = 0; n <= N; ++n) {
            const double J = (n & 1) ? Ji[(n - 1) / 2] : Rh[n / 2];
            P[n] = scale * (rp * J);
            rp = rp * rho;
        }
        return;
    }
    const FarBinom &B = far_binom();
    const double s0 = z * z + 2.0 * a * a;
    const double r = std::sqrt(s0);
    const double e1 = a * a / s0;
    const double e2 = e1 * e1;
    double rp = 1.0 / r;
    for (int n = 0; n <= N; ++n) {
        double acc = B.b[n][4];
        for (int i = 3; i >= 0; --i) acc = acc * e2 + B.b[n][i];
        P[n] = rp * acc / (4.0 * PI_);
        rp = rp * r;
    }
}

// The switch between the two odd-power rules: the pair lies closer to the
// observer than its own length (the observer inside it included).
static inline bool pair_is_near(double c, double h) {
    if (c <= 0.0 && 0.0 <= c + h) return true;
    return std::min(std::fabs(c), std::fabs(c + h)) < h;
}

// mu_{n,q} for even n <= N and q = 0, 1 of a NEAR pair: the closed form in z
// under the ring average, the graded rule in theta. Writes mu[2*n + q].
static void near_odd(double c, double h, double a, int N, const Rules &R, double *mu) {
    const double z0 = c, z1 = c + h;
    const int J = N / 2;
    const int s0 = (z0 > 0.0) - (z0 < 0.0), s1 = (z1 > 0.0) - (z1 < 0.0);
    const int nlog = s1 - s0;
    double t;
    if (z0 == 0.0)
        t = std::fabs(z1);
    else if (z1 == 0.0)
        t = std::fabs(z0);
    else
        t = std::min(std::fabs(z0), std::fabs(z1));
    t /= 2.0 * a;
    const double d2 = h * (z1 + z0);  // R1^2 - R0^2
    double acc0[NCAP / 2 + 1], acc1[NCAP / 2 + 1];
    for (int j = 0; j <= J; ++j) acc0[j] = acc1[j] = 0.0;
    const double top = 0.5 * PI_;
    double lo = 0.0, hi = std::min(t, top);
    while (true) {
        const bool first = nlog && lo == 0.0;
        const std::vector<double> &X = first ? R.xs : R.xg;
        const std::vector<double> &W = first ? R.ws : R.wg;
        const double lhi = first ? std::log(hi) : 0.0;
        for (size_t i = 0; i < X.size(); ++i) {
            const double th = lo + (hi - lo) * X[i];
            const double w = (hi - lo) * W[i];
            const double b = 2.0 * a * std::sin(th);
            const double b2 = b * b;
            const double R0 = std::sqrt(z0 * z0 + b2), R1 = std::sqrt(z1 * z1 + b2);
            double I0, B0 = 0.0, wl = 0.0;
            if (nlog == 0) {
                if (z0 > 0.0)
                    I0 = std::log1p(h * (1.0 + (z1 + z0) / (R1 + R0)) / (z0 + R0));
                else
                    I0 = std::log1p(h * (1.0 - (z1 + z0) / (R1 + R0)) / (R1 - z1));
            } else if (first) {
                // I = A + B ln(theta), both smooth: the split rule's weights
                I0 = -nlog * std::log(b / th);
                if (s1) I0 = I0 + std::log(z1 + R1);
                if (s0) I0 = I0 + std::log(R0 - z0);
                B0 = -static_cast<double>(nlog);
                wl = hi * (R.ws[i] * lhi + R.wls[i]);
            } else {
                I0 = std::asinh(z1 / b) - std::asinh(z0 / b);
            }
            // E_p = R1^p - R0^p, D_p = z1 R1^p - z0 R0^p, p odd, as sums of
            // same-signed terms
            double E = d2 / (R1 + R0);
            double R0p = R0, R1p = R1;
            for (int j = 0; j <= J; ++j) {
                const int p = 2 * j - 1;
                if (j) {
                    const double D = h * R1p + z0 * E;
                    I0 = (D + p * b2 * I0) / (p + 1);
                    if (first) B0 = p * b2 * B0 / (p + 1);
                    E = R1 * R1 * E + R0p * d2;
                    R0p = R0p * R0 * R0;
                    R1p = R1p * R1 * R1;
                }
                const double I1 = E / (p + 2);
                acc0[j] += w * I0;
                acc1[j] += w * (I1 - z0 * I0);
                if (first) {
                    const double lg = wl * B0;
                    acc0[j] += lg;
                    acc1[j] -= z0 * lg;
                }
            }
        }
        if (hi >= top) break;
        lo = hi;
        hi = std::min(hi * GRADE, top);
    }
    const double scale = (2.0 / PI_) / (4.0 * PI_);
    for (int j = 0; j <= J; ++j) {
        mu[2 * (2 * j)] = acc0[j] * scale;
        mu[2 * (2 * j) + 1] = acc1[j] * scale;
    }
}

enum { NEAR = 0, CENTRED = 1, PLAIN_FAR = 2 };

static inline int pair_kind(double c, double h, double a) {
    if (pair_is_near(c, h)) return NEAR;
    if (std::min(std::fabs(c), std::fabs(c + h)) >= T_SWITCH * a) return CENTRED;
    return PLAIN_FAR;
}

static const int NRING = 12;  // `_exact_kernel_series.N_RING`

// cos(phi) at the rule's six distinct points, phi = 15, 45, .. 165 degrees:
// (sqrt6 + sqrt2)/4, 1/sqrt2, (sqrt6 - sqrt2)/4 and their negatives. Literals,
// the reference's own: a loop of std::cos here is vectorised into libmvec,
// which the wheel's glibc floor does not have (tests/test_glibc_floor.py).
struct RingCos {
    double c[NRING / 2];
};

static const RingCos &ring_cos() {
    static const RingCos C = {{0.9659258262890683, 0.7071067811865476, 0.25881904510252074,
                               -0.25881904510252074, -0.7071067811865476,
                               -0.9659258262890683}};
    return C;
}

// The largest |R - Rc| over the pair: what k multiplies in its series' bound.
static inline double pair_reach(double c, double h, double a, int kind) {
    const double zmin = std::min(std::fabs(c), std::fabs(c + h));
    const double zmax = std::max(std::fabs(c), std::fabs(c + h));
    const double r = std::hypot(zmax, 2.0 * a);
    return kind == CENTRED ? 0.5 * (r - zmin) : r;
}

// mu[2*n + q], n = 0..N, q = 0, 1, of one pair, and its Rc; N is the kind's.
static void point_series(double c, double h, double a, int kind, int N, const Rules &R,
                         double *mu, double &Rc) {
    for (int i = 0; i < 2 * (N + 1); ++i) mu[i] = 0.0;
    if (kind == CENTRED) {
        const RingCos &C = ring_cos();
        const double zmin = std::min(std::fabs(c), std::fabs(c + h));
        const double zmax = std::max(std::fabs(c), std::fabs(c + h));
        Rc = 0.5 * (zmin + std::hypot(zmax, 2.0 * a));
        for (size_t i = 0; i < R.xg.size(); ++i) {
            const double v = h * R.xg[i];
            const double w0 = h * R.wg[i], w1 = w0 * v;
            const double z = c + v;
            const double s0 = z * z + 2.0 * a * a;
            double term[NRING / 2], d[NRING / 2];
            for (int r = 0; r < NRING / 2; ++r) {
                const double Rr = std::sqrt(s0 - 2.0 * a * a * C.c[r]);
                d[r] = Rr - Rc;
                term[r] = 1.0 / Rr;
            }
            for (int n = 0; n <= N; ++n) {
                double T = 0.0;
                for (int r = 0; r < NRING / 2; ++r) {
                    T += term[r];
                    term[r] = term[r] * d[r];
                }
                T = T / (NRING / 2) / (4.0 * PI_);
                mu[2 * n] += T * w0;
                mu[2 * n + 1] += T * w1;
            }
        }
        return;
    }
    Rc = 0.0;
    double P[NCAP];
    for (size_t i = 0; i < R.xg.size(); ++i) {
        const double v = h * R.xg[i];
        const double w0 = h * R.wg[i], w1 = w0 * v;
        ring_powers(c + v, a, N, P);
        for (int n = 0; n <= N; ++n) {
            mu[2 * n] += P[n] * w0;
            mu[2 * n + 1] += P[n] * w1;
        }
    }
    if (kind == NEAR) near_odd(c, h, a, N, R, mu);
}

// N_q(k) = e^{-jk Rc} sum_n (-jk)^n/n! mu_{n,q}, Horner from the highest term.
static inline void evaluate(const double *mu, int N, int nq, const cd &x, double Rc, cd *out) {
    for (int q = 0; q < nq; ++q) {
        cd acc(0.0, 0.0);
        for (int n = N; n >= 1; --n) acc = (acc + mu[2 * n + q]) * (x / static_cast<double>(n));
        out[q] = acc + mu[q];
    }
    if (Rc != 0.0) {
        const cd ph = std::exp(x * Rc);
        for (int q = 0; q < nq; ++q) out[q] = out[q] * ph;
    }
}

static inline int64_t qround(double x) { return static_cast<int64_t>(std::nearbyint(x)); }

// The canonical key of (c, h) under the mirror c -> -c - h (the 1433 key).
static inline void canon(double c, double h, double q, Key &key, bool &M, double &cc) {
    const int64_t C = qround(c / q), H = qround(h / q);
    const int64_t Cm = -C - H;
    key.hi = H;
    key.hj = 0;
    key.s = 0;
    if (Cm < C) {
        key.c = Cm;
        M = true;
        cc = -c - h;
    } else {
        key.c = C;
        M = false;
        cc = c;
    }
}

// N(c) from the canonical value V = N(-c - h): v = h - u.
static inline void mirror(const cd *V, int nq, double h, cd *X) {
    X[0] = V[0];
    if (nq > 1) X[1] = h * V[0] - V[1];
}

static int resolve_threads(int n_threads) {
    int nt = 1;
#ifdef _OPENMP
    nt = omp_get_max_threads();
    if (n_threads > 0) nt = std::min(nt, n_threads);
#else
    (void)n_threads;
#endif
    return nt;
}

typedef py::array_t<double, py::array::c_style | py::array::forcecast> darr;

static Rules make_rules(darr xg, darr wg, darr xs, darr ws, darr wls) {
    if (xg.size() != wg.size() || xs.size() != ws.size() || xs.size() != wls.size() ||
        xg.size() == 0 || xs.size() == 0)
        throw std::runtime_error("exact kernel: inconsistent quadrature rules");
    Rules R;
    R.xg.assign(xg.data(), xg.data() + xg.size());
    R.wg.assign(wg.data(), wg.data() + wg.size());
    R.xs.assign(xs.data(), xs.data() + xs.size());
    R.ws.assign(ws.data(), ws.data() + ws.size());
    R.wls.assign(wls.data(), wls.data() + wls.size());
    return R;
}

class PointSeriesAccel {
  public:
    PointSeriesAccel(darr x0, darr sgn, darr h, double a, int nq, int64_t max_cached, double q,
                     Rules rules, double x_max, double x_centred, double tol, int n_threads)
        : a_(a), nq_(nq), max_cached_(max_cached), q_(q), R_(rules), x_max_(x_max),
          x_centred_(x_centred), n_threads_(n_threads), built_(0), reused_(0), evaluated_(0),
          have_k_(false) {
        if (nq < 1 || nq > 2)
            throw std::runtime_error("ExactKernelPointSeries: nq must be 1 or 2");
        const py::ssize_t n = x0.size();
        if (sgn.size() != n || h.size() != n)
            throw std::runtime_error("ExactKernelPointSeries: x0, sgn, h must share a length");
        if (!(q > 0.0)) throw std::runtime_error("ExactKernelPointSeries: q must be positive");
        if (!(a > 0.0)) throw std::runtime_error("ExactKernelPointSeries: a must be positive");
        n_plain_ = n_terms(x_max, tol);
        n_centred_ = n_terms(x_centred, tol);
        if (n_plain_ >= NCAP - 1 || n_centred_ >= NCAP - 1)
            throw std::runtime_error("ExactKernelPointSeries: the bounds need more terms than N_CAP");
        kinds_[0] = kinds_[1] = kinds_[2] = 0;
        x0_.assign(x0.data(), x0.data() + n);
        sgn_.assign(sgn.data(), sgn.data() + n);
        h_.assign(h.data(), h.data() + n);
        off_.push_back(0);
    }

    int64_t n() const { return static_cast<int64_t>(x0_.size()); }
    int64_t built() const { return built_; }
    int64_t reused() const { return reused_; }
    int64_t evaluated() const { return evaluated_; }
    int64_t cached() const { return static_cast<int64_t>(cache_.size()); }
    int64_t nbytes() const {
        return static_cast<int64_t>(coef_.size() * sizeof(double) +
                                    cache_.size() * (2 * sizeof(Key) + 4 * sizeof(int64_t) +
                                                     2 * sizeof(cd)));
    }
    py::tuple kinds() const { return py::make_tuple(kinds_[0], kinds_[1], kinds_[2]); }

    void rows(darr x_obs, std::complex<double> k, py::array out) {
        const int64_t n = this->n();
        const int nq = nq_;
        const int64_t m = static_cast<int64_t>(x_obs.size());
        if (out.ndim() != 3 || out.shape(0) != nq || out.shape(1) != m || out.shape(2) != n)
            throw std::runtime_error("ExactKernelPointSeries.rows: out must be (nq, m, n)");
        if (!out.dtype().is(py::dtype::of<std::complex<double>>()) ||
            !(out.flags() & py::array::c_style) || !out.writeable())
            throw std::runtime_error(
                "ExactKernelPointSeries.rows: out must be complex128, C-contiguous, writeable");
        cd *O = static_cast<cd *>(out.mutable_data());
        const double *X = x_obs.data();
        if (m == 0 || n == 0) return;
        const int nt = resolve_threads(n_threads_);
        const int64_t npairs = m * n;
        const cd x = cd(0.0, -1.0) * k;  // -jk

        // The values at this k of the pairs already built: kept while k is
        // the same, so a repeat call evaluates nothing.
        if (!have_k_ || k != k_) {
            for (size_t s = 0; s < kok_.size(); ++s) kok_[s] = 0;
            k_ = k;
            have_k_ = true;
        }

        // 1. keys, looked up in the persistent cache (read only)
        NpVec<int64_t> slot;
        slot.resize(static_cast<size_t>(npairs));
        {
            py::gil_scoped_release release;
#pragma omp parallel for schedule(static) num_threads(nt)
            for (int64_t o = 0; o < m; ++o) {
                Key key;
                bool M;
                double cc;
                for (int64_t j = 0; j < n; ++j) {
                    canon(sgn_[j] * (x0_[j] - X[o]), h_[j], q_, key, M, cc);
                    slot[o * n + j] = cache_.find(key);
                }
            }
        }
        // 2. the call's new keys, first met first; and which built pairs
        //    still need their value at this k
        KeyTable wmap;
        NpVec<Key> wkeys;
        NpVec<double> wgeo;   // canonical c, h per new key
        NpVec<int64_t> need;  // built slots to evaluate
        {
            Key key;
            bool M;
            double cc;
            for (int64_t o = 0; o < m; ++o)
                for (int64_t j = 0; j < n; ++j) {
                    int64_t &s = slot[o * n + j];
                    if (s >= 0) {
                        if (kok_[static_cast<size_t>(s)] == 0) {
                            kok_[static_cast<size_t>(s)] = 1;
                            need.push_back(s);
                        }
                        if (kok_[static_cast<size_t>(s)] != 3) {
                            kok_[static_cast<size_t>(s)] |= 2;  // counted as reused this call
                            ++reused_;
                            touched_.push_back(s);
                        }
                        continue;
                    }
                    canon(sgn_[j] * (x0_[j] - X[o]), h_[j], q_, key, M, cc);
                    int64_t u = wmap.find(key);
                    if (u < 0) {
                        u = static_cast<int64_t>(wkeys.size());
                        wmap.insert(key, u);
                        wkeys.push_back(key);
                        wgeo.push_back(cc);
                        wgeo.push_back(h_[j]);
                    }
                    s = -(u + 1);
                }
        }
        for (size_t i = 0; i < touched_.size(); ++i) kok_[static_cast<size_t>(touched_[i])] &= 1;
        touched_.resize(0);
        // 3. the new keys: how many terms each, and room for the ones kept
        const int64_t U = static_cast<int64_t>(wkeys.size());
        NpVec<int> wkind;
        wkind.resize(static_cast<size_t>(U));
        const double ak = std::abs(k);
        for (int64_t u = 0; u < U; ++u) {
            const int kind = pair_kind(wgeo[2 * u], wgeo[2 * u + 1], a_);
            wkind[u] = kind;
            // the bound the term count was sized for must hold at this k
            if (!(ak * pair_reach(wgeo[2 * u], wgeo[2 * u + 1], a_, kind) <=
                  (kind == CENTRED ? x_centred_ : x_max_))) {
                have_k_ = false;  // step 2 marked values this call will now never compute
                throw std::runtime_error(
                    "ExactKernelPointSeries.rows: k is beyond a pair's series bound");
            }
        }
        const int64_t base = static_cast<int64_t>(cache_.size());
        int64_t keep = U;
        if (max_cached_ >= 0) keep = std::max<int64_t>(0, std::min(U, max_cached_ - base));
        for (int64_t u = 0; u < keep; ++u) {
            cache_.insert(wkeys[u], base + u);
            off_.push_back(off_[off_.size() - 1] + 2 * (terms(wkind[u]) + 1));
            rc_.push_back(0.0);
            kok_.push_back(1);
        }
        coef_.resize(static_cast<size_t>(off_[off_.size() - 1]));
        kval_.resize(static_cast<size_t>(cache_.size() * 2));
        NpVec<cd> wvals;  // the values at k of the new keys NOT kept
        wvals.resize(static_cast<size_t>((U - keep) * 2));
        {
            py::gil_scoped_release release;
#pragma omp parallel for schedule(dynamic, 16) num_threads(nt)
            for (int64_t u = 0; u < U; ++u) {
                double tmp[2 * NCAP];
                const bool kept = u < keep;
                const int N = terms(wkind[u]);
                double *mu = kept ? &coef_[static_cast<size_t>(off_[base + u])] : tmp;
                double Rc;
                point_series(wgeo[2 * u], wgeo[2 * u + 1], a_, wkind[u], N, R_, mu, Rc);
                if (kept) rc_[static_cast<size_t>(base + u)] = Rc;
                cd *v = kept ? &kval_[static_cast<size_t>(2 * (base + u))]
                             : &wvals[static_cast<size_t>(2 * (u - keep))];
                v[1] = cd(0.0, 0.0);
                evaluate(mu, N, nq, x, Rc, v);
            }
            // 4. the built pairs first asked for at this k
            const int64_t NE = static_cast<int64_t>(need.size());
#pragma omp parallel for schedule(static) num_threads(nt)
            for (int64_t i = 0; i < NE; ++i) {
                const int64_t s = need[i];
                const int N = static_cast<int>((off_[s + 1] - off_[s]) / 2) - 1;
                kval_[static_cast<size_t>(2 * s + 1)] = cd(0.0, 0.0);
                evaluate(&coef_[static_cast<size_t>(off_[s])], N, nq, x,
                         rc_[static_cast<size_t>(s)], &kval_[static_cast<size_t>(2 * s)]);
            }
        }
        built_ += U;
        for (int64_t u = 0; u < U; ++u) ++kinds_[wkind[u]];
        evaluated_ += U + static_cast<int64_t>(need.size());
        py::gil_scoped_release release;
        // 5. every pair's moments from its key's value, mirrored back if need be
#pragma omp parallel for schedule(static) num_threads(nt)
        for (int64_t o = 0; o < m; ++o) {
            Key key;
            bool M;
            double cc;
            cd Xv[2];
            for (int64_t j = 0; j < n; ++j) {
                int64_t s = slot[o * n + j];
                const cd *V;
                if (s >= 0) {
                    V = &kval_[static_cast<size_t>(2 * s)];
                } else {
                    s = -s - 1;
                    V = s < keep ? &kval_[static_cast<size_t>(2 * (base + s))]
                                 : &wvals[static_cast<size_t>(2 * (s - keep))];
                }
                canon(sgn_[j] * (x0_[j] - X[o]), h_[j], q_, key, M, cc);
                const int64_t at = o * n + j;
                if (M) {
                    mirror(V, nq, h_[j], Xv);
                    for (int qq = 0; qq < nq; ++qq) O[qq * npairs + at] = Xv[qq];
                } else {
                    for (int qq = 0; qq < nq; ++qq) O[qq * npairs + at] = V[qq];
                }
            }
        }
    }

  private:
    double a_;
    int nq_;
    int64_t max_cached_;
    double q_;
    Rules R_;
    double x_max_, x_centred_;
    int n_plain_, n_centred_;
    int n_threads_;
    int64_t built_, reused_, evaluated_;
    int64_t kinds_[3];
    int terms(int kind) const { return kind == CENTRED ? n_centred_ : n_plain_; }
    std::vector<double> x0_, sgn_, h_;
    KeyTable cache_;
    NpVec<int64_t> off_;    // coefficient offsets, one past the last at the end
    NpVec<double> coef_;    // mu[2*n + q] per kept pair
    NpVec<double> rc_;      // its reference distance
    NpVec<cd> kval_;        // the kept pairs' values at k_
    NpVec<uint8_t> kok_;    // bit 0: kval_ holds k_; bit 1: scratch
    NpVec<int64_t> touched_;
    std::complex<double> k_;
    bool have_k_;
};

// (Rc, (N + 1, 2) coefficients) of one pair: the reference's `point_series`,
// for the tests that hold the two routes together.
static py::tuple point_series_py(double c, double h, double a, double x_max, double x_centred,
                                 double tol, Rules rules) {
    const int kind = pair_kind(c, h, a);
    const int N = n_terms(kind == CENTRED ? x_centred : x_max, tol);
    if (N >= NCAP - 1) throw std::runtime_error("point series: the bounds need more terms than N_CAP");
    py::array_t<double> out({static_cast<py::ssize_t>(N + 1), static_cast<py::ssize_t>(2)});
    double Rc;
    point_series(c, h, a, kind, N, rules, out.mutable_data(), Rc);
    return py::make_tuple(Rc, out);
}

}  // namespace exact_kernel_series_1454

void register_exact_kernel_series(py::module_ &m) {
    using namespace exact_kernel_series_1454;
    py::class_<Rules>(m, "ExactKernelSeriesRules")
        .def(py::init(&make_rules), py::arg("x_gl"), py::arg("w_gl"), py::arg("x_split"),
             py::arg("w_split"), py::arg("wlog_split"));
    m.def("exact_kernel_point_series", &point_series_py,
          "(Rc, (N + 1, 2) geometry-only series coefficients) of one pair. momwire#1454.",
          py::arg("c"), py::arg("h"), py::arg("a"), py::arg("x_max"), py::arg("x_centred"),
          py::arg("tol"), py::arg("rules"));
    py::class_<PointSeriesAccel>(m, "ExactKernelPointSeries")
        .def(py::init<darr, darr, darr, double, int, int64_t, double, Rules, double, double,
                      double, int>(),
             py::arg("x0"), py::arg("sgn"), py::arg("h"), py::arg("a"), py::arg("nq"),
             py::arg("max_cached"), py::arg("q"), py::arg("rules"), py::arg("x_max"),
             py::arg("x_centred"), py::arg("tol"), py::arg("n_threads") = 0)
        .def("rows", &PointSeriesAccel::rows,
             "Write the moments at observers x_obs (m,) and wavenumber k into out "
             "(nq, m, n), complex128, C-contiguous.",
             py::arg("x_obs"), py::arg("k"), py::arg("out"))
        .def_property_readonly("built", &PointSeriesAccel::built)
        .def_property_readonly("reused", &PointSeriesAccel::reused)
        .def_property_readonly("evaluated", &PointSeriesAccel::evaluated)
        .def_property_readonly("cached", &PointSeriesAccel::cached)
        .def_property_readonly("nbytes", &PointSeriesAccel::nbytes)
        .def_property_readonly("kinds", &PointSeriesAccel::kinds)
        .def_property_readonly("n", &PointSeriesAccel::n);
    m.attr("exact_kernel_series_1454") = true;
}
