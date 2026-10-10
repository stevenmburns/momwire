#include "_accel_common.h"

#include <cstring>
#include <stdexcept>
#ifdef _OPENMP
#include <omp.h>
#endif

// momwire#1410: the exact ring kernel's coaxial pair moments in C++.
//
// A port of `_exact_kernel.py` (momwire#1408): the AGM near form with its
// exact log split, the five-term far series, the closed-form piecewise
// polynomial weight W_pq(y), product integration on the piece touching the
// singular point, geometric grading beyond it, and one Gauss-Legendre rule per
// piece on far pairs. Plus `CoaxialRows`' observer-row windows (momwire#1411)
// with the pair dedup done by hashing instead of `np.unique`'s sort, which was
// most of the numpy route's cost on a uniform run (6.9 s of 13.5 s at 3,000
// segments).
//
// NOT BIT-EXACT WITH THE NUMPY ROUTE, by decision (Steve, 10-10): numpy's
// vectorised exp/log/sqrt, its einsum orders and its batch-wide stopping rules
// (the AGM runs until EVERY element converged, the near series' length is set
// by the batch's largest rho) are not reproduced. The agreement is gated at a
// derived tolerance (tests/test_exact_kernel_accel_1410.py). Within this
// kernel every value is a function of its own pair alone -- no reduction
// spans pairs or threads -- so a run is bit-stable whatever the thread count.
//
// Its own TU so that no existing kernel's codegen can move with it
// (the momwire#1187 lesson).
//
// THE DEDUP, extended (momwire#1410). numpy keys a pair on its quantised
// (h_i, h_j, c, sigma_i, sigma_j). Here a pair is first written with both
// segments FORWARD (a reversed segment's moments are its forward moments
// under u -> h - u, a binomial map on the polynomial index), and then keyed
// on the least of its four images under the two exact symmetries of a
// coaxial pair: the transpose (swap the roles of i and j: J -> J^T,
// c -> -c) and the mirror of the axis (both local coordinates reverse,
// c -> -c + h_i - h_j). On a uniform run that halves the 2n - 1 unique pairs
// to n; on a graded mesh, whose pairs are otherwise all distinct, the
// transpose alone halves the kernel work and a mesh symmetric about its
// centre halves it again. `symmetric=false` keeps numpy's own key, so the
// tests can hold the two apart.

namespace exact_kernel_1410 {

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

// ---------------------------------------------------------------------------
// the 1-D weight W_pq(y)
// ---------------------------------------------------------------------------
static const double BINOM[MAXND + 1][MAXND + 1] = {
    {1, 0, 0, 0, 0, 0, 0},  {1, 1, 0, 0, 0, 0, 0},   {1, 2, 1, 0, 0, 0, 0},
    {1, 3, 3, 1, 0, 0, 0},  {1, 4, 6, 4, 1, 0, 0},   {1, 5, 10, 10, 5, 1, 0},
    {1, 6, 15, 20, 15, 6, 1}};

// branch: -1 = natural (max/min), else bit0 = lower end on the v-bound,
// bit1 = upper end on the v-bound (`_weight`'s y_branch, held).
static inline void weight(double y, double hi, double hj, double si, double sj, int nd,
                          int branch, double *W) {
    const double s = si * sj;
    const double v0 = sj * y;
    const double lo_v = s > 0 ? -v0 : v0 - hj;
    const double hi_v = s > 0 ? hj - v0 : v0;
    double u0, u1;
    if (branch < 0) {
        u0 = std::max(0.0, lo_v);
        u1 = std::max(std::min(hi, hi_v), u0);
    } else {
        u0 = (branch & 1) ? lo_v : 0.0;
        u1 = (branch & 2) ? hi_v : hi;
    }
    double U0[2 * MAXND + 1], U1[2 * MAXND + 1], V0[MAXND + 1];
    U0[0] = U1[0] = V0[0] = 1.0;
    for (int e = 1; e <= 2 * nd; ++e) {
        U0[e] = U0[e - 1] * u0;
        U1[e] = U1[e - 1] * u1;
    }
    for (int e = 1; e <= nd; ++e) V0[e] = V0[e - 1] * v0;
    for (int p = 0; p < nd; ++p)
        for (int q = 0; q < nd; ++q) {
            double acc = 0.0;
            double sr = 1.0;
            for (int r = 0; r <= q; ++r) {
                const int e = p + r + 1;
                acc = acc + BINOM[q][r] * V0[q - r] * sr * (U1[e] - U0[e]) / e;
                sr *= s;
            }
            W[p * MAXND + q] = acc;
        }
}

static inline int branch_at(double y_branch, double hi, double hj, double si, double sj) {
    const double s = si * sj;
    const double v0 = sj * y_branch;
    const double bl = s > 0 ? -v0 : v0 - hj;
    const double bh = s > 0 ? hj - v0 : v0;
    return (bl > 0.0 ? 1 : 0) | (bh < hi ? 2 : 0);
}

// ---------------------------------------------------------------------------
// pair moments
// ---------------------------------------------------------------------------
struct Geo {
    double hi, hj, si, sj;
    int nd;
    double a, k;
    const Rules *R;
};

static inline void zero(cd *M) {
    for (int t = 0; t < NM; ++t) M[t] = cd(0.0, 0.0);
}

// The integral over zeta of W(y_star + sgn*zeta) K(zeta) on one piece.
struct Piece {
    const Geo *g;
    double y_star, sgn;
    int branch;

    void smooth(double lo, double hi_, cd *out) const {
        zero(out);
        const Rules &R = *g->R;
        const int nd = g->nd;
        double W[NM];
        for (size_t i = 0; i < R.xg.size(); ++i) {
            const double z = lo + (hi_ - lo) * R.xg[i];
            const double w = (hi_ - lo) * R.wg[i];
            const cd f = w * ring_kernel(z, g->a, g->k);
            weight(y_star + sgn * z, g->hi, g->hj, g->si, g->sj, nd, branch, W);
            for (int p = 0; p < nd; ++p)
                for (int q = 0; q < nd; ++q) out[p * MAXND + q] += f * W[p * MAXND + q];
        }
    }

    void graded(double lo, double e, cd *out) const {
        zero(out);
        cd part[NM];
        double c0 = lo;
        while (c0 < e) {
            double c1 = std::min(c0 * GRADE, e);
            if (e - c1 < 1e-3 * (c1 - c0)) c1 = e;
            smooth(c0, c1, part);
            for (int t = 0; t < NM; ++t) out[t] += part[t];
            c0 = c1;
        }
    }

    // F(b) = integral over [0, b]: the log split on [0, min(b, 2a)], graded beyond.
    void from_zero(double b, cd *out) const {
        zero(out);
        const Rules &R = *g->R;
        const int nd = g->nd;
        const double b0 = std::min(b, 2.0 * g->a);
        const double lb0 = std::log(b0);
        double W[NM];
        for (size_t i = 0; i < R.xs.size(); ++i) {
            const double z = b0 * R.xs[i];
            cd A, B;
            near_eval(z, g->a, g->k, true, A, B);
            const cd f = b0 * (R.ws[i] * (B + A * lb0) + R.wls[i] * A);
            weight(y_star + sgn * z, g->hi, g->hj, g->si, g->sj, nd, branch, W);
            for (int p = 0; p < nd; ++p)
                for (int q = 0; q < nd; ++q) out[p * MAXND + q] += f * W[p * MAXND + q];
        }
        if (b > b0) {
            cd part[NM];
            graded(b0, b, part);
            for (int t = 0; t < NM; ++t) out[t] += part[t];
        }
    }

    void integrate(double d, double ell, cd *out) const {
        zero(out);
        if (ell <= 0.0) return;
        if (d >= ell) return smooth(d, d + ell, out);
        if (d >= 2.0 * g->a) return graded(d, d + ell, out);
        from_zero(d + ell, out);
        if (d > 0.0) {
            cd part[NM];
            from_zero(d, part);
            for (int t = 0; t < NM; ++t) out[t] -= part[t];
        }
    }
};

static inline void sort4(double *v) {
    for (int i = 1; i < 4; ++i)
        for (int j = i; j > 0 && v[j] < v[j - 1]; --j) std::swap(v[j], v[j - 1]);
}

// `pair_moments`: any separation, the singular point handled exactly.
static void pair_moments_near(const Geo &g, double c, cd *out) {
    zero(out);
    const double y_star = -c;
    double cor[4] = {0.0, g.sj * g.hj, -g.si * g.hi, g.sj * g.hj - g.si * g.hi};
    sort4(cor);
    cd part[NM];
    for (int s = 0; s < 3; ++s) {
        const double y0 = cor[s], y1 = cor[s + 1];
        if (y1 - y0 <= 0.0) continue;
        double subs[2][2] = {{y0, y1}, {0.0, 0.0}};
        int ns = 1;
        if (y0 < y_star && y_star < y1) {
            subs[0][1] = y_star;
            subs[1][0] = y_star;
            subs[1][1] = y1;
            ns = 2;
        }
        for (int t = 0; t < ns; ++t) {
            const double ya = subs[t][0], yb = subs[t][1];
            Piece P;
            P.g = &g;
            P.y_star = y_star;
            P.branch = branch_at(0.5 * (ya + yb), g.hi, g.hj, g.si, g.sj);
            double d;
            if (y_star <= ya) {
                P.sgn = 1.0;
                d = ya - y_star;
            } else {
                P.sgn = -1.0;
                d = y_star - yb;
            }
            P.integrate(std::max(d, 0.0), yb - ya, part);
            for (int u = 0; u < NM; ++u) out[u] += part[u];
        }
    }
}

// `pair_moments_far`: one GL rule per piece, no grading.
static void pair_moments_far(const Geo &g, double c, cd *out) {
    zero(out);
    const Rules &R = *g.R;
    const int nd = g.nd;
    double cor[4] = {0.0, g.sj * g.hj, -g.si * g.hi, g.sj * g.hj - g.si * g.hi};
    sort4(cor);
    double W[NM];
    for (int s = 0; s < 3; ++s) {
        const double lo = cor[s], ln = cor[s + 1] - cor[s];
        if (ln <= 0.0) continue;
        for (size_t i = 0; i < R.xg.size(); ++i) {
            const double y = lo + ln * R.xg[i];
            const cd f = (ln * R.wg[i]) * ring_kernel(c + y, g.a, g.k);
            weight(y, g.hi, g.hj, g.si, g.sj, nd, -1, W);
            for (int p = 0; p < nd; ++p)
                for (int q = 0; q < nd; ++q) out[p * MAXND + q] += f * W[p * MAXND + q];
        }
    }
}

static inline bool is_far(const Geo &g, double c) {
    const double y[4] = {0.0, g.sj * g.hj, -g.si * g.hi, g.sj * g.hj - g.si * g.hi};
    double ylo = y[0], yhi = y[0];
    for (int i = 1; i < 4; ++i) {
        ylo = std::min(ylo, y[i]);
        yhi = std::max(yhi, y[i]);
    }
    const double zlo = c + ylo, zhi = c + yhi;
    const double dmin =
        (zlo <= 0.0 && zhi >= 0.0) ? 0.0 : std::min(std::fabs(zlo), std::fabs(zhi));
    return dmin >= std::max(g.hi, g.hj);
}

static void pair_moments(const Geo &g, double c, cd *out) {
    if (is_far(g, c))
        pair_moments_far(g, c, out);
    else
        pair_moments_near(g, c, out);
}

// tracemalloc sees numpy's buffers but not a std::vector's, and the memgate
// lane measures the correction with tracemalloc (tests/
// test_exact_kernel_rows_1411.py). So every buffer here that scales with the
// window or the cache is a numpy array (numpy reports its allocations to
// tracemalloc): without that the gate would pass on the C++ route by not
// seeing its memory at all. (Reporting a std::vector through
// PyTraceMalloc_Track is not portable: CPython 3.12.3's header declares it
// without C linkage, so a C++ caller links against a mangled name.)
//
// A growable array of a trivially copyable T. Every operation that
// allocates, and the destructor, need the GIL; element access does not.
template <class T>
class NpVec {
  public:
    NpVec() : n_(0), cap_(0), p_(0) {}
    void reserve(size_t c) {
        if (c <= cap_) return;
        py::array_t<uint8_t> a(static_cast<py::ssize_t>(c * sizeof(T)));
        T *q = reinterpret_cast<T *>(a.mutable_data());
        if (n_) std::memcpy(q, p_, n_ * sizeof(T));
        buf_ = a;
        p_ = q;
        cap_ = c;
    }
    void resize(size_t n) {  // new elements uninitialised
        reserve(n);
        n_ = n;
    }
    void assign(size_t n, const T &v) {
        resize(n);
        for (size_t i = 0; i < n; ++i) p_[i] = v;
    }
    void push_back(const T &v) {
        if (n_ == cap_) reserve(cap_ ? 2 * cap_ : 16);
        p_[n_++] = v;
    }
    T &operator[](size_t i) { return p_[i]; }
    const T &operator[](size_t i) const { return p_[i]; }
    size_t size() const { return n_; }
    bool empty() const { return n_ == 0; }
    T *data() { return p_; }
    const T *data() const { return p_; }

  private:
    size_t n_, cap_;
    T *p_;
    py::object buf_;
};

// ---------------------------------------------------------------------------
// the dedup key and its table
// ---------------------------------------------------------------------------
struct Key {
    int64_t hi, hj, c;
    int64_t s;
};

static inline bool key_eq(const Key &x, const Key &y) {
    return x.hi == y.hi && x.hj == y.hj && x.c == y.c && x.s == y.s;
}

static inline bool key_lt(const Key &x, const Key &y) {
    if (x.hi != y.hi) return x.hi < y.hi;
    if (x.hj != y.hj) return x.hj < y.hj;
    if (x.c != y.c) return x.c < y.c;
    return x.s < y.s;
}

static inline uint64_t mix64(uint64_t z) {
    z += 0x9e3779b97f4a7c15ULL;
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
    z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
    return z ^ (z >> 31);
}

static inline uint64_t key_hash(const Key &k) {
    uint64_t h = mix64(static_cast<uint64_t>(k.hi));
    h = mix64(h ^ static_cast<uint64_t>(k.hj));
    h = mix64(h ^ static_cast<uint64_t>(k.c));
    return mix64(h ^ static_cast<uint64_t>(k.s));
}

// Open addressing, linear probing, power-of-two capacity, load <= 1/2.
class KeyTable {
  public:
    KeyTable() : mask_(0), count_(0) {}
    void reserve(size_t n) {
        size_t cap = 16;
        while (cap < 2 * n) cap <<= 1;
        if (cap > keys_.size()) rehash(cap);
    }
    int64_t find(const Key &k) const {
        if (keys_.empty()) return -1;
        size_t i = key_hash(k) & mask_;
        while (true) {
            const int64_t v = vals_[i];
            if (v < 0) return -1;
            if (key_eq(keys_[i], k)) return v;
            i = (i + 1) & mask_;
        }
    }
    void insert(const Key &k, int64_t v) {  // k must be absent
        if (2 * (count_ + 1) > keys_.size()) rehash(keys_.empty() ? 16 : 2 * keys_.size());
        size_t i = key_hash(k) & mask_;
        while (vals_[i] >= 0) i = (i + 1) & mask_;
        keys_[i] = k;
        vals_[i] = v;
        ++count_;
    }
    size_t size() const { return count_; }
  private:
    void rehash(size_t cap) {
        NpVec<Key> ok = keys_;
        NpVec<int64_t> ov = vals_;
        keys_ = NpVec<Key>();
        vals_ = NpVec<int64_t>();
        keys_.assign(cap, Key());
        vals_.assign(cap, -1);
        mask_ = cap - 1;
        count_ = 0;
        for (size_t i = 0; i < ov.size(); ++i)
            if (ov[i] >= 0) insert(ok[i], ov[i]);
    }
    NpVec<Key> keys_;
    NpVec<int64_t> vals_;
    size_t mask_, count_;
};

// A pair's canonical key, the transform from the canonical value to its own,
// and the canonical geometry (evaluated for the representative only).
struct Canon {
    Key key;
    bool T, Li, Lj;  // transpose; reverse the left / right polynomial index
    double hi, hj, c, si, sj;  // the geometry the canonical value is evaluated at
};

static inline int64_t qround(double x) { return static_cast<int64_t>(std::nearbyint(x)); }

static inline void canon(double hi, double hj, double xi, double xj, double si, double sj,
                         double q, bool sym, Canon &o) {
    if (!sym) {
        const double c = xj - xi;
        o.key.hi = qround(hi / q);
        o.key.hj = qround(hj / q);
        o.key.c = qround(c / q);
        o.key.s = (si > 0 ? 1 : 0) | (sj > 0 ? 2 : 0);
        o.T = o.Li = o.Lj = false;
        o.hi = hi;
        o.hj = hj;
        o.c = c;
        o.si = si;
        o.sj = sj;
        return;
    }
    const double fi = si > 0 ? xi : xi - hi;
    const double fj = sj > 0 ? xj : xj - hj;
    const double c = fj - fi;
    const int64_t Hi = qround(hi / q), Hj = qround(hj / q), C = qround(c / q);
    Key cand[4];
    cand[0].hi = Hi, cand[0].hj = Hj, cand[0].c = C;                 // identity
    cand[1].hi = Hj, cand[1].hj = Hi, cand[1].c = -C;                // transpose
    cand[2].hi = Hi, cand[2].hj = Hj, cand[2].c = -C + Hi - Hj;      // mirror
    cand[3].hi = Hj, cand[3].hj = Hi, cand[3].c = C - Hi + Hj;       // both
    int g = 0;
    for (int t = 0; t < 4; ++t) cand[t].s = 3;
    for (int t = 1; t < 4; ++t)
        if (key_lt(cand[t], cand[g])) g = t;
    o.key = cand[g];
    o.T = (g & 1) != 0;
    const bool M = g >= 2;
    o.Li = (si < 0) != M;
    o.Lj = (sj < 0) != M;
    o.si = o.sj = 1.0;
    switch (g) {
        case 0: o.hi = hi, o.hj = hj, o.c = c; break;
        case 1: o.hi = hj, o.hj = hi, o.c = -c; break;
        case 2: o.hi = hi, o.hj = hj, o.c = -c + hi - hj; break;
        default: o.hi = hj, o.hj = hi, o.c = c - hi + hj; break;
    }
}

// R(h)[p][r] = C(p, r) h^{p-r} (-1)^r: u^p in the reversed coordinate.
static inline void reversal(double h, int nd, double *R) {
    double hp[MAXND];
    hp[0] = 1.0;
    for (int i = 1; i < nd; ++i) hp[i] = hp[i - 1] * h;
    for (int p = 0; p < nd; ++p)
        for (int r = 0; r < nd; ++r)
            R[p * MAXND + r] = r <= p ? BINOM[p][r] * hp[p - r] * ((r & 1) ? -1.0 : 1.0) : 0.0;
}

// The pair's own moments from the canonical value V (compact, stride nd).
static inline void apply(const cd *V, int nd, const Canon &o, double hi, double hj, cd *X) {
    for (int p = 0; p < nd; ++p)
        for (int q = 0; q < nd; ++q) X[p * MAXND + q] = o.T ? V[q * nd + p] : V[p * nd + q];
    if (o.Li) {
        double R[NM];
        reversal(hi, nd, R);
        cd Y[NM];
        for (int p = 0; p < nd; ++p)
            for (int q = 0; q < nd; ++q) {
                cd acc(0.0, 0.0);
                for (int r = 0; r <= p; ++r) acc += R[p * MAXND + r] * X[r * MAXND + q];
                Y[p * MAXND + q] = acc;
            }
        for (int t = 0; t < NM; ++t) X[t] = Y[t];
    }
    if (o.Lj) {
        double R[NM];
        reversal(hj, nd, R);
        cd Y[NM];
        for (int p = 0; p < nd; ++p)
            for (int q = 0; q < nd; ++q) {
                cd acc(0.0, 0.0);
                for (int r = 0; r <= q; ++r) acc += X[p * MAXND + r] * R[q * MAXND + r];
                Y[p * MAXND + q] = acc;
            }
        for (int t = 0; t < NM; ++t) X[t] = Y[t];
    }
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

static Rules make_rules(py::array_t<double, py::array::c_style | py::array::forcecast> xg,
                        py::array_t<double, py::array::c_style | py::array::forcecast> wg,
                        py::array_t<double, py::array::c_style | py::array::forcecast> xs,
                        py::array_t<double, py::array::c_style | py::array::forcecast> ws,
                        py::array_t<double, py::array::c_style | py::array::forcecast> wls) {
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

// ---------------------------------------------------------------------------
// CoaxialRows: observer-row windows with a cross-window cache
// ---------------------------------------------------------------------------
class CoaxialRowsAccel {
  public:
    CoaxialRowsAccel(py::array_t<double, py::array::c_style | py::array::forcecast> x0,
                     py::array_t<double, py::array::c_style | py::array::forcecast> sgn,
                     py::array_t<double, py::array::c_style | py::array::forcecast> h, double a,
                     double k, int nd, int64_t max_cached, double q, bool symmetric,
                     Rules rules, int n_threads)
        : a_(a), k_(k), nd_(nd), max_cached_(max_cached), q_(q), sym_(symmetric),
          R_(rules), n_threads_(n_threads), evaluated_(0) {
        if (nd < 1 || nd > MAXND)
            throw std::runtime_error("CoaxialRowsAccel: nd must be in [1, 6]");
        const py::ssize_t n = x0.size();
        if (sgn.size() != n || h.size() != n)
            throw std::runtime_error("CoaxialRowsAccel: x0, sgn, h must share a length");
        if (!(q > 0.0)) throw std::runtime_error("CoaxialRowsAccel: q must be positive");
        x0_.assign(x0.data(), x0.data() + n);
        sgn_.assign(sgn.data(), sgn.data() + n);
        h_.assign(h.data(), h.data() + n);
    }

    int64_t n() const { return static_cast<int64_t>(x0_.size()); }
    int64_t evaluated() const { return evaluated_; }
    int64_t cached() const { return static_cast<int64_t>(cache_.size()); }

    void rows(int64_t r0, int64_t r1, py::array out) {
        const int64_t n = this->n();
        const int nd = nd_;
        if (r0 < 0 || r1 > n || r1 < r0)
            throw std::runtime_error("CoaxialRowsAccel.rows: bad row window");
        const int64_t m = r1 - r0;
        if (out.ndim() != 4 || out.shape(0) != nd || out.shape(1) != nd || out.shape(2) != m ||
            out.shape(3) != n)
            throw std::runtime_error("CoaxialRowsAccel.rows: out must be (nd, nd, r1-r0, n)");
        // A py::array (not array_t) so a wrong dtype or layout is refused
        // rather than converted into a temporary the writes would vanish into.
        if (!out.dtype().is(py::dtype::of<std::complex<double>>()) ||
            !(out.flags() & py::array::c_style) || !out.writeable())
            throw std::runtime_error(
                "CoaxialRowsAccel.rows: out must be complex128, C-contiguous, writeable");
        cd *O = static_cast<cd *>(out.mutable_data());
        if (m == 0 || n == 0) return;
        const int nt = resolve_threads(n_threads_);
        const int64_t npairs = m * n;
        const int64_t plane = npairs;  // out[p, q] is one (m, n) plane

        // The parallel phases run without the GIL; the serial ones hold it,
        // since they allocate (numpy-backed buffers, see NpVec).
        // 1. canonical keys, looked up in the persistent cache (read only)
        NpVec<int64_t> slot;
        slot.resize(static_cast<size_t>(npairs));
        {
            py::gil_scoped_release release;
#pragma omp parallel for schedule(static) num_threads(nt)
            for (int64_t ii = 0; ii < m; ++ii) {
                const int64_t i = r0 + ii;
                Canon o;
                for (int64_t j = 0; j < n; ++j) {
                    canon(h_[i], h_[j], x0_[i], x0_[j], sgn_[i], sgn_[j], q_, sym_, o);
                    slot[ii * n + j] = cache_.find(o.key);
                }
            }
        }
        // 2. the window's new keys, in row-major order: each is evaluated at
        // its first pair, as the numpy route does
        KeyTable wmap;
        NpVec<Key> wkeys;
        NpVec<double> wgeo;  // hi, hj, c, si, sj per new key
        {
            Canon o;
            for (int64_t ii = 0; ii < m; ++ii) {
                const int64_t i = r0 + ii;
                for (int64_t j = 0; j < n; ++j) {
                    int64_t &s = slot[ii * n + j];
                    if (s >= 0) continue;
                    canon(h_[i], h_[j], x0_[i], x0_[j], sgn_[i], sgn_[j], q_, sym_, o);
                    int64_t u = wmap.find(o.key);
                    if (u < 0) {
                        u = static_cast<int64_t>(wkeys.size());
                        wmap.insert(o.key, u);
                        wkeys.push_back(o.key);
                        wgeo.push_back(o.hi);
                        wgeo.push_back(o.hj);
                        wgeo.push_back(o.c);
                        wgeo.push_back(o.si);
                        wgeo.push_back(o.sj);
                    }
                    s = -(u + 1);
                }
            }
        }
        // 3. evaluate them, each from its own geometry alone
        const int64_t U = static_cast<int64_t>(wkeys.size());
        const int nd2 = nd * nd;
        NpVec<cd> wvals;
        wvals.resize(static_cast<size_t>(U * nd2));
        {
            py::gil_scoped_release release;
#pragma omp parallel for schedule(dynamic, 16) num_threads(nt)
            for (int64_t u = 0; u < U; ++u) {
                const double *G = &wgeo[static_cast<size_t>(5 * u)];
                Geo g;
                g.hi = G[0];
                g.hj = G[1];
                g.si = G[3];
                g.sj = G[4];
                g.nd = nd;
                g.a = a_;
                g.k = k_;
                g.R = &R_;
                cd M[NM];
                pair_moments(g, G[2], M);
                for (int p = 0; p < nd; ++p)
                    for (int qq = 0; qq < nd; ++qq)
                        wvals[u * nd2 + p * nd + qq] = M[p * MAXND + qq];
            }
        }
        evaluated_ += U;
        // 4. remember them, first met first, up to the cap
        for (int64_t u = 0; u < U; ++u) {
            if (max_cached_ >= 0 && static_cast<int64_t>(cache_.size()) >= max_cached_) break;
            cache_.insert(wkeys[u], static_cast<int64_t>(cache_.size()));
            for (int t = 0; t < nd2; ++t) vals_.push_back(wvals[u * nd2 + t]);
        }
        py::gil_scoped_release release;
        // 5. every pair's moments, from its key's value and its own transform
#pragma omp parallel for schedule(static) num_threads(nt)
        for (int64_t ii = 0; ii < m; ++ii) {
            const int64_t i = r0 + ii;
            Canon oc;
            cd X[NM];
            for (int64_t j = 0; j < n; ++j) {
                const int64_t s = slot[ii * n + j];
                const cd *V = s >= 0 ? &vals_[static_cast<size_t>(s * nd2)]
                                     : &wvals[static_cast<size_t>((-s - 1) * nd2)];
                const int64_t at = ii * n + j;
                if (!sym_) {
                    for (int p = 0; p < nd; ++p)
                        for (int qq = 0; qq < nd; ++qq) O[(p * nd + qq) * plane + at] = V[p * nd + qq];
                    continue;
                }
                canon(h_[i], h_[j], x0_[i], x0_[j], sgn_[i], sgn_[j], q_, sym_, oc);
                apply(V, nd, oc, h_[i], h_[j], X);
                for (int p = 0; p < nd; ++p)
                    for (int qq = 0; qq < nd; ++qq) O[(p * nd + qq) * plane + at] = X[p * MAXND + qq];
            }
        }
    }

  private:
    double a_, k_;
    int nd_;
    int64_t max_cached_;
    double q_;
    bool sym_;
    Rules R_;
    int n_threads_;
    int64_t evaluated_;
    std::vector<double> x0_, sgn_, h_;
    KeyTable cache_;
    NpVec<cd> vals_;
};

// ---------------------------------------------------------------------------
// bindings
// ---------------------------------------------------------------------------
static py::array_t<std::complex<double>> ring_kernel_py(
    py::array_t<double, py::array::c_style | py::array::forcecast> z, double a, double k) {
    py::array_t<std::complex<double>> out(z.request().shape);
    const double *Z = z.data();
    cd *O = out.mutable_data();
    for (py::ssize_t i = 0; i < z.size(); ++i) O[i] = ring_kernel(Z[i], a, k);
    return out;
}

static py::tuple ring_kernel_split_py(
    py::array_t<double, py::array::c_style | py::array::forcecast> z, double a, double k) {
    py::array_t<std::complex<double>> A(z.request().shape), B(z.request().shape);
    const double *Z = z.data();
    cd *PA = A.mutable_data();
    cd *PB = B.mutable_data();
    for (py::ssize_t i = 0; i < z.size(); ++i) {
        const double zi = std::fabs(Z[i]);
        if (zi >= T_SWITCH * a)
            throw std::runtime_error("exact_kernel_split serves |z| < 6a only");
        near_eval(zi, a, k, true, PA[i], PB[i]);
    }
    return py::make_tuple(A, B);
}

// One pair, any separation; far=-1 classifies as the rows do, 0 forces the
// near route, 1 the far rule.
static py::array_t<std::complex<double>> pair_moments_py(double hi, double hj, double c,
                                                         double si, double sj, double a,
                                                         double k, int nd, Rules rules,
                                                         int far) {
    if (nd < 1 || nd > MAXND) throw std::runtime_error("pair moments: nd must be in [1, 6]");
    Geo g;
    g.hi = hi;
    g.hj = hj;
    g.si = si;
    g.sj = sj;
    g.nd = nd;
    g.a = a;
    g.k = k;
    g.R = &rules;
    cd M[NM];
    if (far < 0)
        pair_moments(g, c, M);
    else if (far == 0)
        pair_moments_near(g, c, M);
    else
        pair_moments_far(g, c, M);
    py::array_t<std::complex<double>> out({nd, nd});
    cd *O = out.mutable_data();
    for (int p = 0; p < nd; ++p)
        for (int q = 0; q < nd; ++q) O[p * nd + q] = M[p * MAXND + q];
    return out;
}

// The dedup key's transform for one pair, for the tests that hold every
// branch of `canon` / `apply` to having run: (T, Li, Lj, canonical key).
static py::tuple canon_py(double hi, double hj, double xi, double xj, double si, double sj,
                          double q, bool symmetric) {
    Canon o;
    canon(hi, hj, xi, xj, si, sj, q, symmetric, o);
    return py::make_tuple(o.T, o.Li, o.Lj,
                          py::make_tuple(o.key.hi, o.key.hj, o.key.c, o.key.s));
}

}  // namespace exact_kernel_1410

void register_exact_kernel(py::module_ &m) {
    using namespace exact_kernel_1410;
    py::class_<Rules>(m, "ExactKernelRules")
        .def(py::init(&make_rules), py::arg("x_gl"), py::arg("w_gl"), py::arg("x_split"),
             py::arg("w_split"), py::arg("wlog_split"));
    m.def("exact_kernel_ring", &ring_kernel_py,
          "The exact ring kernel K(z) (momwire#1408), elementwise. momwire#1410.",
          py::arg("z"), py::arg("a"), py::arg("k"));
    m.def("exact_kernel_ring_split", &ring_kernel_split_py,
          "(A, B) with K = A ln|z| + B, |z| < 6a. momwire#1410.", py::arg("z"), py::arg("a"),
          py::arg("k"));
    m.def("exact_kernel_pair_moments", &pair_moments_py,
          "(nd, nd) exact-kernel moments of one coaxial pair. momwire#1410.", py::arg("hi"),
          py::arg("hj"), py::arg("c"), py::arg("si"), py::arg("sj"), py::arg("a"), py::arg("k"),
          py::arg("nd"), py::arg("rules"), py::arg("far") = -1);
    m.def("exact_kernel_canon", &canon_py,
          "(transpose, reverse_i, reverse_j, key) of one pair's dedup key. momwire#1410.",
          py::arg("hi"), py::arg("hj"), py::arg("xi"), py::arg("xj"), py::arg("si"),
          py::arg("sj"), py::arg("q"), py::arg("symmetric"));
    py::class_<CoaxialRowsAccel>(m, "ExactKernelCoaxialRows")
        .def(py::init<py::array_t<double, py::array::c_style | py::array::forcecast>,
                      py::array_t<double, py::array::c_style | py::array::forcecast>,
                      py::array_t<double, py::array::c_style | py::array::forcecast>, double,
                      double, int, int64_t, double, bool, Rules, int>(),
             py::arg("x0"), py::arg("sgn"), py::arg("h"), py::arg("a"), py::arg("k"),
             py::arg("nd"), py::arg("max_cached"), py::arg("q"), py::arg("symmetric"),
             py::arg("rules"), py::arg("n_threads") = 0)
        .def("rows", &CoaxialRowsAccel::rows,
             "Write observer rows [r0, r1) of the group's exact moment block into out "
             "(nd, nd, r1 - r0, n), complex128, C-contiguous.",
             py::arg("r0"), py::arg("r1"), py::arg("out"))
        .def_property_readonly("evaluated", &CoaxialRowsAccel::evaluated)
        .def_property_readonly("cached", &CoaxialRowsAccel::cached)
        .def_property_readonly("n", &CoaxialRowsAccel::n);
    m.attr("exact_kernel_1410") = true;
}
