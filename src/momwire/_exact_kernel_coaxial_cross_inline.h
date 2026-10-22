// The exact ring kernel's Galerkin pair moments ACROSS a radius step
// (momwire#1438 for the numpy route, momwire#1615 here) and the observer-row
// windows over one coaxial line (`_exact_kernel.CoaxialCrossRows`): bs2's
// cross job, shared by the extension's binding TU (`_accel_exact_kernel.cpp`)
// and the Python-free core TU (`_exact_kernel_coaxial_core.cpp`, the
// standalone bs2 route), so the two compute the same bits.
//
// The numpy `CoaxialCrossRows` stays the reference the tests hold this to;
// this is a second implementation of the same formulas (`pair_moments2`,
// `pair_moments2_far`, the two-radius piece rule `_integrate_two_radius`),
// not a bit-for-bit replay of numpy's array arithmetic, the way
// `PointCrossRowsCore` is razor-2p's. Every value depends on its own pair
// alone, never on the window, the batch or the thread count.
//
// Included INSIDE each TU's namespace, after `_exact_kernel_ring_inline.h`,
// `_exact_kernel_ring2_inline.h` (ring_kernel2) and
// `_exact_kernel_coaxial_inline.h` (weight, branch_at, sort4, Rules, NpVec,
// qround, resolve_threads); MW_EK_RELEASE_GIL is the scope's GIL release.
// It includes nothing itself: the TU has <map>, <limits> and <stdexcept>.

// One pair's geometry: the observer segment i of radius a1, the source j of
// radius a2 (a1 != a2).
struct Geo2 {
    double hi, hj, si, sj;
    int nd;
    double a1, a2, k;
    const Rules *R;
};

// `_Piece2`: the integral over zeta of W(y_star + sgn*zeta) K2(zeta) on one
// piece. K2 is analytic on the real axis (branch points at +-i|a1 - a2|), so
// a piece takes one rule on [d, |delta|] and geometric chunks beyond, no log
// split (`_integrate_two_radius`).
struct Piece2 {
    const Geo2 *g;
    double y_star, sgn, dl;
    int branch;

    void smooth(double lo, double hi_, cd *out) const {
        zero(out);
        const Rules &R = *g->R;
        const int nd = g->nd;
        double W[NM];
        for (size_t i = 0; i < R.xg.size(); ++i) {
            const double z = lo + (hi_ - lo) * R.xg[i];
            const double w = (hi_ - lo) * R.wg[i];
            const cd f = w * ring_kernel2(z, g->a1, g->a2, g->k);
            weight(y_star + sgn * z, g->hi, g->hj, g->si, g->sj, nd, branch, W);
            for (int p = 0; p < nd; ++p)
                for (int q = 0; q < nd; ++q) out[p * MAXND + q] += f * W[p * MAXND + q];
        }
    }

    void graded(double lo, double e, cd *out) const {
        zero(out);
        if (!(lo > 0.0)) throw std::logic_error("exact kernel: graded chunks from zero");
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

    void integrate(double d, double ell, cd *out) const {
        zero(out);
        if (ell <= 0.0) return;
        if (d >= ell) return smooth(d, d + ell, out);
        if (d >= dl) return graded(d, d + ell, out);
        const double e = std::min(dl, d + ell);
        smooth(d, e, out);
        if (d + ell > e) {
            cd part[NM];
            graded(e, d + ell, part);
            for (int t = 0; t < NM; ++t) out[t] += part[t];
        }
    }
};

// `pair_moments2`: any separation.
static void pair_moments2_near(const Geo2 &g, double c, cd *out) {
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
            Piece2 P;
            P.g = &g;
            P.y_star = y_star;
            P.dl = std::fabs(g.a1 - g.a2);
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

// `pair_moments2_far`: one GL rule per piece, no grading.
static void pair_moments2_far(const Geo2 &g, double c, cd *out) {
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
            const cd f = (ln * R.wg[i]) * ring_kernel2(c + y, g.a1, g.a2, g.k);
            weight(y, g.hi, g.hj, g.si, g.sj, nd, -1, W);
            for (int p = 0; p < nd; ++p)
                for (int q = 0; q < nd; ++q) out[p * MAXND + q] += f * W[p * MAXND + q];
        }
    }
}

// `CoaxialCrossRows.rows`' split: far when the pair is at least its own
// length from z = 0 (`_min_abs_z(...) >= max(hi, hj)`).
static inline bool is_far2(const Geo2 &g, double c) {
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

static void pair_moments2(const Geo2 &g, double c, cd *out) {
    if (is_far2(g, c))
        pair_moments2_far(g, c, out);
    else
        pair_moments2_near(g, c, out);
}

// ---------------------------------------------------------------------------
// CoaxialCrossRows: the two-radius moments of one coaxial LINE, by observer
// row windows, zero on every pair whose two segments carry the same radius
// label (the one-radius groups own those). Keyed on the quantised
// (h_i, h_j, c, sigma_i, sigma_j, a_i, a_j) as the numpy class keys them; new
// keys first met first, each evaluated from its first pair's geometry,
// remembered across windows up to the cap. Checks are the caller's.
// ---------------------------------------------------------------------------
struct CrossKey {
    int64_t v[6];  // h_i, h_j, c, (sigma_i > 0) | (sigma_j > 0) << 1, a_i, a_j
    bool operator<(const CrossKey &o) const {
        for (int t = 0; t < 6; ++t)
            if (v[t] != o.v[t]) return v[t] < o.v[t];
        return false;
    }
};

class CoaxialCrossRowsCore {
  public:
    CoaxialCrossRowsCore(const double *x0, const double *sgn, const double *h, const double *a,
                         const int64_t *lab, int64_t n, double k, int nd, int64_t max_cached,
                         double q, double qa, Rules rules, int n_threads)
        : k_(k), nd_(nd), max_cached_(max_cached), q_(q), qa_(qa), R_(rules),
          n_threads_(n_threads), evaluated_(0) {
        if (nd < 1 || nd > MAXND)
            throw std::runtime_error("CoaxialCrossRowsAccel: nd must be in [1, 6]");
        if (!(q > 0.0) || !(qa > 0.0))
            throw std::runtime_error("CoaxialCrossRowsAccel: q and qa must be positive");
        x0_.assign(x0, x0 + n);
        sgn_.assign(sgn, sgn + n);
        h_.assign(h, h + n);
        a_.assign(a, a + n);
        lab_.assign(lab, lab + n);
    }

    int64_t n() const { return static_cast<int64_t>(x0_.size()); }
    int64_t evaluated() const { return evaluated_; }
    int64_t cached() const { return static_cast<int64_t>(cache_.size()); }

    // Observer rows [r0, r1) into O, (nd, nd, r1 - r0, n) C order, every
    // entry written (zero on the same-radius pairs).
    void rows(int64_t r0, int64_t r1, cd *O) {
        const int64_t n = this->n();
        const int nd = nd_;
        if (r0 < 0 || r1 > n || r1 < r0)
            throw std::runtime_error("CoaxialCrossRowsAccel.rows: bad row window");
        const int64_t m = r1 - r0;
        const int64_t plane = m * n;
        for (int64_t t = 0; t < nd * nd * plane; ++t) O[t] = cd(0.0, 0.0);
        if (m == 0 || n == 0) return;
        const int nt = resolve_threads(n_threads_);
        const int64_t SAME = std::numeric_limits<int64_t>::min();
        const int nd2 = nd * nd;
        // 1. keys, looked up in the persistent cache; 2. the window's new
        // keys, row-major, first met first (serial: the map is not shared)
        NpVec<int64_t> slot;
        slot.resize(static_cast<size_t>(plane));
        std::map<CrossKey, int64_t> wmap;
        std::vector<double> wgeo;  // hi, hj, c, si, sj, ai, aj per new key
        std::vector<CrossKey> wkeys;
        for (int64_t ii = 0; ii < m; ++ii) {
            const int64_t i = r0 + ii;
            for (int64_t j = 0; j < n; ++j) {
                int64_t &s = slot[ii * n + j];
                if (lab_[i] == lab_[j]) {
                    s = SAME;
                    continue;
                }
                const double c = x0_[j] - x0_[i];
                const CrossKey kk = key(i, j, c);
                auto hit = cache_.find(kk);
                if (hit != cache_.end()) {
                    s = hit->second;
                    continue;
                }
                auto w = wmap.find(kk);
                int64_t u;
                if (w == wmap.end()) {
                    u = static_cast<int64_t>(wkeys.size());
                    wmap.emplace(kk, u);
                    wkeys.push_back(kk);
                    for (double v : {h_[i], h_[j], c, sgn_[i], sgn_[j], a_[i], a_[j]}) wgeo.push_back(v);
                } else {
                    u = w->second;
                }
                s = -(u + 1);
            }
        }
        // 3. evaluate them, each from its own geometry alone
        const int64_t U = static_cast<int64_t>(wkeys.size());
        NpVec<cd> wvals;
        wvals.resize(static_cast<size_t>(U * nd2));
        {
            MW_EK_RELEASE_GIL
#pragma omp parallel for schedule(dynamic, 16) num_threads(nt)
            for (int64_t u = 0; u < U; ++u) {
                const double *G = &wgeo[static_cast<size_t>(7 * u)];
                Geo2 g;
                g.hi = G[0];
                g.hj = G[1];
                g.si = G[3];
                g.sj = G[4];
                g.a1 = G[5];
                g.a2 = G[6];
                g.nd = nd;
                g.k = k_;
                g.R = &R_;
                cd M[NM];
                pair_moments2(g, G[2], M);
                for (int p = 0; p < nd; ++p)
                    for (int qq = 0; qq < nd; ++qq) wvals[u * nd2 + p * nd + qq] = M[p * MAXND + qq];
            }
        }
        evaluated_ += U;
        // 4. remember them, first met first, up to the cap
        for (int64_t u = 0; u < U; ++u) {
            if (max_cached_ >= 0 && static_cast<int64_t>(cache_.size()) >= max_cached_) break;
            cache_.emplace(wkeys[u], static_cast<int64_t>(cache_.size()));
            for (int t = 0; t < nd2; ++t) vals_.push_back(wvals[u * nd2 + t]);
        }
        // 5. every cross pair's moments from its key's value
        for (int64_t ii = 0; ii < m; ++ii)
            for (int64_t j = 0; j < n; ++j) {
                const int64_t s = slot[ii * n + j];
                if (s == SAME) continue;
                const cd *V = s >= 0 ? &vals_[static_cast<size_t>(s * nd2)]
                                     : &wvals[static_cast<size_t>((-s - 1) * nd2)];
                const int64_t at = ii * n + j;
                for (int pq = 0; pq < nd2; ++pq) O[pq * plane + at] = V[pq];
            }
    }

  private:
    double k_;
    int nd_;
    int64_t max_cached_;
    double q_, qa_;
    Rules R_;
    int n_threads_;
    int64_t evaluated_;
    std::vector<double> x0_, sgn_, h_, a_;
    std::vector<int64_t> lab_;
    std::map<CrossKey, int64_t> cache_;
    NpVec<cd> vals_;

    CrossKey key(int64_t i, int64_t j, double c) const {
        CrossKey kk;
        kk.v[0] = qround(h_[i] / q_);
        kk.v[1] = qround(h_[j] / q_);
        kk.v[2] = qround(c / q_);
        kk.v[3] = (sgn_[i] > 0 ? 1 : 0) | (sgn_[j] > 0 ? 2 : 0);
        kk.v[4] = qround(a_[i] / qa_);
        kk.v[5] = qround(a_[j] / qa_);
        return kk;
    }
};
