// The exact ring kernel's POINT moments and the point-tested assembly
// (momwire#1433), shared by the extension's binding TU
// (`_accel_exact_kernel_point.cpp`) and the Python-free core TU
// (`_exact_kernel_point_core.cpp`, momwire#1479's standalone path): one
// text, so the C++ route and the extension compute the same bits. Included
// INSIDE each TU's namespace, after `_exact_kernel_ring_inline.h` and
// `_exact_kernel_table_inline.h`.
//
// MW_EK_RELEASE_GIL is the binding's `py::gil_scoped_release` (declared in
// the scope the GIL is to be released for); the core TU defines it empty.
// Every buffer that scales with the window or the cache is an `NpVec`
// (numpy-backed in the binding so tracemalloc sees it; a plain vector in the
// core TU, `_exact_kernel_table_inline.h`).

// The integral over zeta of f(v0 + sgn*zeta) K(zeta), f = v^q, q < nq.
struct PPiece {
    double v0, sgn, a, k;
    int nq;
    const Rules *R;

    inline void add(double z, const cd &f, cd *out) const {
        const double v = v0 + sgn * z;
        double p = 1.0;
        for (int q = 0; q < nq; ++q) {
            out[q] += f * p;
            p *= v;
        }
    }

    void smooth(double lo, double hi, cd *out) const {
        for (size_t i = 0; i < R->xg.size(); ++i) {
            const double z = lo + (hi - lo) * R->xg[i];
            add(z, ((hi - lo) * R->wg[i]) * ring_kernel(z, a, k), out);
        }
    }

    void graded(double lo, double e, cd *out) const {
        double c0 = lo;
        while (c0 < e) {
            double c1 = std::min(c0 * GRADE, e);
            if (e - c1 < 1e-3 * (c1 - c0)) c1 = e;
            smooth(c0, c1, out);
            c0 = c1;
        }
    }

    // F(b) = integral over [0, b], added into out with sign s.
    void from_zero(double b, double s, cd *out) const {
        const double b0 = std::min(b, 2.0 * a);
        const double lb0 = std::log(b0);
        cd tmp[MAXND];
        for (int q = 0; q < nq; ++q) tmp[q] = cd(0.0, 0.0);
        for (size_t i = 0; i < R->xs.size(); ++i) {
            const double z = b0 * R->xs[i];
            cd A, B;
            near_eval(z, a, k, true, A, B);
            add(z, b0 * (R->ws[i] * (B + A * lb0) + R->wls[i] * A), tmp);
        }
        if (b > b0) graded(b0, b, tmp);
        for (int q = 0; q < nq; ++q) out[q] += s * tmp[q];
    }

    void integrate(double d, double ell, cd *out) const {
        if (ell <= 0.0) return;
        if (d >= ell) return smooth(d, d + ell, out);
        if (d >= 2.0 * a) return graded(d, d + ell, out);
        from_zero(d + ell, 1.0, out);
        if (d > 0.0) from_zero(d, -1.0, out);
    }
};

// N_q(c, h), q < nq, into out[0..nq).
static void point_moments(double c, double h, int nq, double a, double k, const Rules &R,
                          cd *out) {
    for (int q = 0; q < nq; ++q) out[q] = cd(0.0, 0.0);
    const double vs = -c;
    PPiece P;
    P.v0 = vs;
    P.a = a;
    P.k = k;
    P.nq = nq;
    P.R = &R;
    if (0.0 < vs && vs < h) {
        P.sgn = -1.0;
        P.integrate(0.0, vs, out);
        P.sgn = 1.0;
        P.integrate(0.0, h - vs, out);
    } else if (vs <= 0.0) {
        P.sgn = 1.0;
        P.integrate(-vs, h, out);
    } else {
        P.sgn = -1.0;
        P.integrate(vs - h, h, out);
    }
}

static inline int64_t qround(double x) { return static_cast<int64_t>(std::nearbyint(x)); }

// The canonical key of (c, h) under the mirror c -> -c - h, whether the
// canonical value must be mirrored back, and the canonical c.
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

// N(c) from the canonical value V = N(-c - h): v^q = (h - u)^q.
static inline void mirror(const cd *V, int nq, double h, cd *X) {
    static const double BIN[MAXND][MAXND] = {{1, 0, 0, 0, 0, 0},  {1, 1, 0, 0, 0, 0},
                                             {1, 2, 1, 0, 0, 0},  {1, 3, 3, 1, 0, 0},
                                             {1, 4, 6, 4, 1, 0},  {1, 5, 10, 10, 5, 1}};
    double hp[MAXND];
    hp[0] = 1.0;
    for (int i = 1; i < nq; ++i) hp[i] = hp[i - 1] * h;
    for (int q = 0; q < nq; ++q) {
        cd acc(0.0, 0.0);
        for (int r = 0; r <= q; ++r)
            acc += (BIN[q][r] * hp[q - r] * ((r & 1) ? -1.0 : 1.0)) * V[r];
        X[q] = acc;
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
// PointRows: the moments of one coaxial group's segments at observer points,
// with a cross-call cache (checks are the caller's)
// ---------------------------------------------------------------------------
class PointRowsCore {
  public:
    PointRowsCore(const double *x0, const double *sgn, const double *h, int64_t n, double a,
                  double k, int nq, int64_t max_cached, double q, Rules rules, int n_threads)
        : a_(a), k_(k), nq_(nq), max_cached_(max_cached), q_(q), R_(rules),
          n_threads_(n_threads), evaluated_(0) {
        x0_.assign(x0, x0 + n);
        sgn_.assign(sgn, sgn + n);
        h_.assign(h, h + n);
    }

    int64_t n() const { return static_cast<int64_t>(x0_.size()); }
    int nq() const { return nq_; }
    int64_t evaluated() const { return evaluated_; }
    int64_t cached() const { return static_cast<int64_t>(cache_.size()); }

    // The moments at observers X[0..m) into O, (nq, m, n) C order.
    void rows(const double *X, int64_t m, cd *O) {
        const int64_t n = this->n();
        const int nq = nq_;
        if (m == 0 || n == 0) return;
        const int nt = resolve_threads(n_threads_);
        const int64_t npairs = m * n;

        // 1. keys, looked up in the persistent cache (read only)
        NpVec<int64_t> slot;
        slot.resize(static_cast<size_t>(npairs));
        {
            MW_EK_RELEASE_GIL
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
        // 2. the call's new keys, first met first
        KeyTable wmap;
        NpVec<Key> wkeys;
        NpVec<double> wgeo;  // canonical c, h per new key
        {
            Key key;
            bool M;
            double cc;
            for (int64_t o = 0; o < m; ++o)
                for (int64_t j = 0; j < n; ++j) {
                    int64_t &s = slot[o * n + j];
                    if (s >= 0) continue;
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
        // 3. evaluate them, each from its own geometry alone
        const int64_t U = static_cast<int64_t>(wkeys.size());
        NpVec<cd> wvals;
        wvals.resize(static_cast<size_t>(U * nq));
        {
            MW_EK_RELEASE_GIL
#pragma omp parallel for schedule(dynamic, 16) num_threads(nt)
            for (int64_t u = 0; u < U; ++u)
                point_moments(wgeo[2 * u], wgeo[2 * u + 1], nq, a_, k_, R_,
                              &wvals[static_cast<size_t>(u * nq)]);
        }
        evaluated_ += U;
        // 4. remember them, up to the cap
        for (int64_t u = 0; u < U; ++u) {
            if (max_cached_ >= 0 && static_cast<int64_t>(cache_.size()) >= max_cached_) break;
            cache_.insert(wkeys[u], static_cast<int64_t>(cache_.size()));
            for (int t = 0; t < nq; ++t) vals_.push_back(wvals[u * nq + t]);
        }
        MW_EK_RELEASE_GIL
        // 5. every pair's moments from its key's value, mirrored back if need be
#pragma omp parallel for schedule(static) num_threads(nt)
        for (int64_t o = 0; o < m; ++o) {
            Key key;
            bool M;
            double cc;
            cd Xv[MAXND];
            for (int64_t j = 0; j < n; ++j) {
                const int64_t s = slot[o * n + j];
                const cd *V = s >= 0 ? &vals_[static_cast<size_t>(s * nq)]
                                     : &wvals[static_cast<size_t>((-s - 1) * nq)];
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
    double a_, k_;
    int nq_;
    int64_t max_cached_;
    double q_;
    Rules R_;
    int n_threads_;
    int64_t evaluated_;
    std::vector<double> x0_, sgn_, h_;
    KeyTable cache_;
    NpVec<cd> vals_;
};

// ---------------------------------------------------------------------------
// The point-tested correction's assembly (razor-2p's wing algebra, #1433):
//
//   Z[m, n] += cA_r * sum_{e in n} t_e dmom_e(o_r)  -  cP_r * sum_{e in n} q_e dM0(o_r, j_e)
//
// for every row entry r = (m, o, cA, cP) and every column n with its wing
// entries e = (j, t, rising, q): dmom = dM1/h on a rising wing, dM0 - dM1/h on
// a falling one. Parallel over COLUMNS, so no two threads write one entry:
// Z is column-major (leading dimension ld) and a column's writes are one
// thread's. Checks are the caller's.
// ---------------------------------------------------------------------------
static void point_assemble(const cd *M0, const cd *M1, int64_t n, const int64_t *rm,
                           const int64_t *ro, const cd *cA, const cd *cP, int64_t R,
                           const int64_t *cp, const int64_t *cn, int64_t NB, const int64_t *ej,
                           const double *et, const uint8_t *er, const double *eq,
                           const double *H, cd *Zp, int64_t ld, int nt) {
#pragma omp parallel for schedule(static) num_threads(nt)
    for (int64_t b = 0; b < NB; ++b) {
        cd *col = Zp + cn[b] * ld;
        for (int64_t i = 0; i < R; ++i) {
            const cd *m0 = M0 + ro[i] * n, *m1 = M1 + ro[i] * n;
            cd a(0.0, 0.0), p(0.0, 0.0);
            for (int64_t e = cp[b]; e < cp[b + 1]; ++e) {
                const int64_t j = ej[e];
                const cd d1 = m1[j] / H[j];
                a += et[e] * (er[e] ? d1 : m0[j] - d1);
                p += eq[e] * m0[j];
            }
            col[rm[i]] += cA[i] * a - cP[i] * p;
        }
    }
}
