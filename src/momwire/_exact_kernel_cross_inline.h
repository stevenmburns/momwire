// The exact ring kernel ACROSS a radius step (momwire#1438): the two-radius
// kernel K2(z; a1, a2), its point moments and their rows over one coaxial
// line (`_exact_kernel.PointCrossRows`), in C++ so that the solvers and the
// standalone razor-2p path (momwire#1479) evaluate them with one text.
//
// The numpy route stays the reference the tests hold this to; this is a
// second implementation of the same formulas, not a bit-for-bit replay of
// numpy's array arithmetic (that route chose its series length per CALL,
// from the largest rho in the batch, so its bits depended on the batch).
// Here every value depends on its own pair alone.
//
// Included INSIDE the point TU's namespace, after `_exact_kernel_ring_inline.h`
// (agm_KE, near_terms, far_polys, Rules, the constants),
// `_exact_kernel_table_inline.h` (Key, KeyTable, NpVec) and the point
// moments' qround / resolve_threads; MW_EK_RELEASE_GIL is the scope's GIL
// release (empty where there is no GIL).

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

// `_point_piece2`: the integral over zeta of (v0 + sgn zeta)^q K2(zeta).
// K2 is analytic on the real axis with branch points at +-i|a1 - a2|, so a
// piece takes one rule on [d, |delta|] and geometric chunks beyond
// (`_integrate_two_radius`); no log split.
struct CPiece {
    double v0, sgn, a1, a2, k, dl;
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
            add(z, ((hi - lo) * R->wg[i]) * ring_kernel2(z, a1, a2, k), out);
        }
    }

    void graded(double lo, double e, cd *out) const {
        if (!(lo > 0.0)) throw std::logic_error("exact kernel: graded chunks from zero");
        double c0 = lo;
        while (c0 < e) {
            double c1 = std::min(c0 * GRADE, e);
            if (e - c1 < 1e-3 * (c1 - c0)) c1 = e;
            smooth(c0, c1, out);
            c0 = c1;
        }
    }

    void integrate(double d, double ell, cd *out) const {
        if (ell <= 0.0) return;
        if (d >= ell) return smooth(d, d + ell, out);
        if (d >= dl) return graded(d, d + ell, out);
        const double e = std::min(dl, d + ell);
        smooth(d, e, out);
        if (d + ell > e) graded(e, d + ell, out);
    }
};

// `point_moments2` (near) and `point_moments2_far` (the source at least its
// own length from the observer: one Gauss-Legendre rule over it), N_q,
// q < nq, into out[0..nq).
static void point_moments2(double c, double h, double a1, double a2, double k, int nq,
                           const Rules &R, cd *out) {
    for (int q = 0; q < nq; ++q) out[q] = cd(0.0, 0.0);
    const double zlo = c, zhi = c + h;
    const double dmin =
        (zlo <= 0.0 && zhi >= 0.0) ? 0.0 : std::min(std::fabs(zlo), std::fabs(zhi));
    if (dmin >= h) {
        for (size_t i = 0; i < R.xg.size(); ++i) {
            const double v = h * R.xg[i];
            const cd wv = (h * R.wg[i]) * ring_kernel2(c + v, a1, a2, k);
            double p = 1.0;
            for (int q = 0; q < nq; ++q) {
                out[q] += wv * p;
                p *= v;
            }
        }
        return;
    }
    const double vs = -c;
    CPiece P;
    P.v0 = vs;
    P.a1 = a1;
    P.a2 = a2;
    P.k = k;
    P.dl = std::fabs(a1 - a2);
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

// ---------------------------------------------------------------------------
// PointCrossRows: the two-radius point moments of one coaxial LINE at
// observer points, zero on every pair whose observer and source carry the
// same radius label (the one-radius groups own those). Keyed on the
// quantised (c, h, a_obs, a_src), new keys first met first, each evaluated
// from its first pair's geometry, remembered across calls up to the cap.
// Checks are the caller's.
// ---------------------------------------------------------------------------
class PointCrossRowsCore {
  public:
    PointCrossRowsCore(const double *x0, const double *sgn, const double *h, const double *a,
                       const int64_t *lab, int64_t n, double k, int nq, int64_t max_cached,
                       double q, double qa, Rules rules, int n_threads)
        : k_(k), nq_(nq), max_cached_(max_cached), q_(q), qa_(qa), R_(rules),
          n_threads_(n_threads), evaluated_(0) {
        x0_.assign(x0, x0 + n);
        sgn_.assign(sgn, sgn + n);
        h_.assign(h, h + n);
        a_.assign(a, a + n);
        lab_.assign(lab, lab + n);
    }

    int64_t n() const { return static_cast<int64_t>(x0_.size()); }
    int nq() const { return nq_; }
    int64_t evaluated() const { return evaluated_; }
    int64_t cached() const { return static_cast<int64_t>(cache_.size()); }

    // The moments at observers X[0..m) of radii A[0..m) and labels L[0..m)
    // into O, (nq, m, n) C order, every entry written.
    void rows(const double *X, const double *A, const int64_t *L, int64_t m, cd *O) {
        const int64_t n = this->n();
        const int nq = nq_;
        const int64_t npairs = m * n;
        for (int64_t i = 0; i < nq * npairs; ++i) O[i] = cd(0.0, 0.0);
        if (m == 0 || n == 0) return;
        const int nt = resolve_threads(n_threads_);
        const int64_t SAME = std::numeric_limits<int64_t>::min();

        // 1. keys, looked up in the persistent cache (read only)
        NpVec<int64_t> slot;
        slot.resize(static_cast<size_t>(npairs));
        {
            MW_EK_RELEASE_GIL
#pragma omp parallel for schedule(static) num_threads(nt)
            for (int64_t o = 0; o < m; ++o)
                for (int64_t j = 0; j < n; ++j)
                    slot[o * n + j] = L[o] == lab_[j] ? SAME : cache_.find(key(X[o], A[o], j));
        }
        // 2. the call's new keys, first met first
        KeyTable wmap;
        NpVec<Key> wkeys;
        NpVec<double> wgeo;  // c, h, a_obs, a_src per new key
        for (int64_t o = 0; o < m; ++o)
            for (int64_t j = 0; j < n; ++j) {
                int64_t &s = slot[o * n + j];
                if (s >= 0 || s == SAME) continue;
                const Key kk = key(X[o], A[o], j);
                int64_t u = wmap.find(kk);
                if (u < 0) {
                    u = static_cast<int64_t>(wkeys.size());
                    wmap.insert(kk, u);
                    wkeys.push_back(kk);
                    wgeo.push_back(sgn_[j] * (x0_[j] - X[o]));
                    wgeo.push_back(h_[j]);
                    wgeo.push_back(A[o]);
                    wgeo.push_back(a_[j]);
                }
                s = -(u + 1);
            }
        // 3. evaluate them, each from its own geometry alone
        const int64_t U = static_cast<int64_t>(wkeys.size());
        NpVec<cd> wvals;
        wvals.resize(static_cast<size_t>(U * nq));
        {
            MW_EK_RELEASE_GIL
#pragma omp parallel for schedule(dynamic, 16) num_threads(nt)
            for (int64_t u = 0; u < U; ++u)
                point_moments2(wgeo[4 * u], wgeo[4 * u + 1], wgeo[4 * u + 2], wgeo[4 * u + 3], k_,
                               nq, R_, &wvals[static_cast<size_t>(u * nq)]);
        }
        evaluated_ += U;
        // 4. remember them, up to the cap
        for (int64_t u = 0; u < U; ++u) {
            if (max_cached_ >= 0 && static_cast<int64_t>(cache_.size()) >= max_cached_) break;
            cache_.insert(wkeys[u], static_cast<int64_t>(cache_.size()));
            for (int t = 0; t < nq; ++t) vals_.push_back(wvals[u * nq + t]);
        }
        MW_EK_RELEASE_GIL
        // 5. every cross pair's moments from its key's value
#pragma omp parallel for schedule(static) num_threads(nt)
        for (int64_t o = 0; o < m; ++o)
            for (int64_t j = 0; j < n; ++j) {
                const int64_t s = slot[o * n + j];
                if (s == SAME) continue;
                const cd *V = s >= 0 ? &vals_[static_cast<size_t>(s * nq)]
                                     : &wvals[static_cast<size_t>((-s - 1) * nq)];
                for (int qq = 0; qq < nq; ++qq) O[qq * npairs + o * n + j] = V[qq];
            }
    }

  private:
    double k_;
    int nq_;
    int64_t max_cached_;
    double q_, qa_;
    Rules R_;
    int n_threads_;
    int64_t evaluated_;
    std::vector<double> x0_, sgn_, h_, a_;
    std::vector<int64_t> lab_;
    KeyTable cache_;
    NpVec<cd> vals_;

    Key key(double x, double a_obs, int64_t j) const {
        Key kk;
        kk.c = qround(sgn_[j] * (x0_[j] - x) / q_);
        kk.hi = qround(h_[j] / q_);
        kk.hj = qround(a_obs / qa_);
        kk.s = qround(a_[j] / qa_);
        return kk;
    }
};
