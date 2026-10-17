// The exact ring kernel's COAXIAL pair moments (momwire#1410) and
// `CoaxialRows`' observer-row windows (momwire#1411), shared by the
// extension's binding TU (`_accel_exact_kernel.cpp`) and the Python-free core
// TU (`_exact_kernel_coaxial_core.cpp`, momwire#1517 step 0): one text, so the
// standalone route and the extension compute the same bits. Included INSIDE
// each TU's namespace, after `_exact_kernel_ring_inline.h`; it includes
// `_exact_kernel_table_inline.h` at the point the 1410 TU always did.
//
// MW_EK_RELEASE_GIL is the binding's `py::gil_scoped_release` (declared in
// the scope the GIL is to be released for); the core TU defines it empty.
// Every buffer that scales with the window or the cache is an `NpVec`
// (numpy-backed in the binding so tracemalloc sees it; a plain vector in the
// core TU, `_exact_kernel_table_inline.h`).

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

#include "_exact_kernel_table_inline.h"

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

// ---------------------------------------------------------------------------
// CoaxialRows: observer-row windows with a cross-window cache
// ---------------------------------------------------------------------------
class CoaxialRowsCore {
  public:
    // x0, sgn, h: n_x0, n_sgn, n_h values (they must agree).
    CoaxialRowsCore(const double *x0, int64_t n_x0, const double *sgn, int64_t n_sgn,
                    const double *h, int64_t n_h, double a, double k, int nd,
                    int64_t max_cached, double q, bool symmetric, Rules rules, int n_threads)
        : a_(a), k_(k), nd_(nd), max_cached_(max_cached), q_(q), sym_(symmetric),
          R_(rules), n_threads_(n_threads), evaluated_(0) {
        if (nd < 1 || nd > MAXND)
            throw std::runtime_error("CoaxialRowsAccel: nd must be in [1, 6]");
        const int64_t n = n_x0;
        if (n_sgn != n || n_h != n)
            throw std::runtime_error("CoaxialRowsAccel: x0, sgn, h must share a length");
        if (!(q > 0.0)) throw std::runtime_error("CoaxialRowsAccel: q must be positive");
        x0_.assign(x0, x0 + n);
        sgn_.assign(sgn, sgn + n);
        h_.assign(h, h + n);
    }

    int64_t n() const { return static_cast<int64_t>(x0_.size()); }
    int64_t evaluated() const { return evaluated_; }
    int64_t cached() const { return static_cast<int64_t>(cache_.size()); }

    // Observer rows [r0, r1) of the group's exact moment block into O,
    // (nd, nd, r1 - r0, n) C order; the caller has checked O's shape.
    void rows(int64_t r0, int64_t r1, cd *O) {
        const int64_t n = this->n();
        const int nd = nd_;
        if (r0 < 0 || r1 > n || r1 < r0)
            throw std::runtime_error("CoaxialRowsAccel.rows: bad row window");
        const int64_t m = r1 - r0;
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
            MW_EK_RELEASE_GIL
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
            MW_EK_RELEASE_GIL
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
        MW_EK_RELEASE_GIL
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
