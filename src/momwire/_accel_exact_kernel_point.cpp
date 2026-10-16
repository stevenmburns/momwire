#include "_accel_common.h"

#include <cstring>
#include <limits>
#include <stdexcept>
#ifdef _OPENMP
#include <omp.h>
#endif

// momwire#1433: the exact ring kernel's POINT moments in C++ -- the 1-D
// primitive a point- or path-tested solver needs (razor-2p's centroids
// first; sin's match points next, momwire#1434).
//
// A port of `_exact_kernel.point_moments` / `PointRows`: for an observer at
// the origin of a coaxial source segment's line and the segment running
// forward from c over length h,
//
//     N_q(c, h) = integral_0^h v^q K(c + v) dv,   q < nq,
//
// the singular point v* = -c split out, the piece touching it integrated by
// the exact log split within 2a and geometric grading beyond, a piece at
// least its own length away by one Gauss-Legendre rule. The kernel, its split
// and the rules are `_exact_kernel_ring_inline.h`, the same text the 1410
// pair-moment TU compiles.
//
// THE DEDUP. A pair is keyed on its quantised (c, h). The mirror of the axis
// (v -> h - v, K even) maps N(c, h) to N(-c - h, h) by a binomial transform
// of the polynomial index, so a pair is keyed on the lesser of c and -c - h:
// a uniform run of n segments seen from its n centroids has n unique keys.
//
// NOT BIT-EXACT WITH THE NUMPY ROUTE, by the 1410 decision: the agreement is
// gated at a derived tolerance (tests/test_exact_kernel_razor_1433.py). Every
// value is a function of its own pair alone, so a run is bit-stable whatever
// the thread count.
//
// Its own TU so that no existing kernel's codegen can move with it
// (momwire#1187); the shared headers are included verbatim into both TUs.

namespace exact_kernel_point_1433 {

#include "_exact_kernel_ring_inline.h"

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

#include "_exact_kernel_table_inline.h"

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

// momwire#1438's two-radius point moments (`PointCrossRows`), in the GIL
// release scope's spelling this TU uses.
#define MW_EK_RELEASE_GIL py::gil_scoped_release release;
#include "_exact_kernel_cross_inline.h"

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
// PointRows: the moments of one coaxial group's segments at observer points,
// with a cross-call cache
// ---------------------------------------------------------------------------
class PointRowsAccel {
  public:
    PointRowsAccel(py::array_t<double, py::array::c_style | py::array::forcecast> x0,
                   py::array_t<double, py::array::c_style | py::array::forcecast> sgn,
                   py::array_t<double, py::array::c_style | py::array::forcecast> h, double a,
                   double k, int nq, int64_t max_cached, double q, Rules rules, int n_threads)
        : a_(a), k_(k), nq_(nq), max_cached_(max_cached), q_(q), R_(rules),
          n_threads_(n_threads), evaluated_(0) {
        if (nq < 1 || nq > MAXND)
            throw std::runtime_error("ExactKernelPointRows: nq must be in [1, 6]");
        const py::ssize_t n = x0.size();
        if (sgn.size() != n || h.size() != n)
            throw std::runtime_error("ExactKernelPointRows: x0, sgn, h must share a length");
        if (!(q > 0.0)) throw std::runtime_error("ExactKernelPointRows: q must be positive");
        x0_.assign(x0.data(), x0.data() + n);
        sgn_.assign(sgn.data(), sgn.data() + n);
        h_.assign(h.data(), h.data() + n);
    }

    int64_t n() const { return static_cast<int64_t>(x0_.size()); }
    int64_t evaluated() const { return evaluated_; }
    int64_t cached() const { return static_cast<int64_t>(cache_.size()); }

    void rows(py::array_t<double, py::array::c_style | py::array::forcecast> x_obs,
              py::array out) {
        const int64_t n = this->n();
        const int nq = nq_;
        const int64_t m = static_cast<int64_t>(x_obs.size());
        if (out.ndim() != 3 || out.shape(0) != nq || out.shape(1) != m || out.shape(2) != n)
            throw std::runtime_error("ExactKernelPointRows.rows: out must be (nq, m, n)");
        if (!out.dtype().is(py::dtype::of<std::complex<double>>()) ||
            !(out.flags() & py::array::c_style) || !out.writeable())
            throw std::runtime_error(
                "ExactKernelPointRows.rows: out must be complex128, C-contiguous, writeable");
        cd *O = static_cast<cd *>(out.mutable_data());
        const double *X = x_obs.data();
        if (m == 0 || n == 0) return;
        const int nt = resolve_threads(n_threads_);
        const int64_t npairs = m * n;

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
            py::gil_scoped_release release;
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
        py::gil_scoped_release release;
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

static py::array_t<std::complex<double>> point_moments_py(double c, double h, double a,
                                                          double k, int nq, Rules rules) {
    if (nq < 1 || nq > MAXND) throw std::runtime_error("point moments: nq must be in [1, 6]");
    cd M[MAXND];
    point_moments(c, h, nq, a, k, rules, M);
    py::array_t<std::complex<double>> out(nq);
    cd *O = out.mutable_data();
    for (int q = 0; q < nq; ++q) O[q] = M[q];
    return out;
}

static py::array_t<std::complex<double>> point_moments2_py(double c, double h, double a1,
                                                           double a2, double k, int nq,
                                                           Rules rules) {
    if (nq < 1 || nq > MAXND) throw std::runtime_error("point moments: nq must be in [1, 6]");
    cd M[MAXND];
    point_moments2(c, h, a1, a2, k, nq, rules, M);
    py::array_t<std::complex<double>> out(nq);
    cd *O = out.mutable_data();
    for (int q = 0; q < nq; ++q) O[q] = M[q];
    return out;
}

static py::array_t<std::complex<double>> ring_kernel2_py(
    py::array_t<double, py::array::c_style | py::array::forcecast> z, double a1, double a2,
    double k) {
    py::array_t<std::complex<double>> out(z.size());
    const double *Z = z.data();
    cd *O = out.mutable_data();
    for (py::ssize_t i = 0; i < z.size(); ++i) O[i] = ring_kernel2(Z[i], a1, a2, k);
    return out;
}

// PointCrossRows through the core class, its arguments checked here.
class PointCrossRowsAccel {
  public:
    PointCrossRowsAccel(py::array_t<double, py::array::c_style | py::array::forcecast> x0,
                        py::array_t<double, py::array::c_style | py::array::forcecast> sgn,
                        py::array_t<double, py::array::c_style | py::array::forcecast> h,
                        py::array_t<double, py::array::c_style | py::array::forcecast> a,
                        py::array_t<int64_t, py::array::c_style | py::array::forcecast> lab,
                        double k, int nq, int64_t max_cached, double q, double qa, Rules rules,
                        int n_threads)
        : impl_(make(x0, sgn, h, a, lab, k, nq, max_cached, q, qa, rules, n_threads)) {}

    int64_t n() const { return impl_.n(); }
    int64_t evaluated() const { return impl_.evaluated(); }
    int64_t cached() const { return impl_.cached(); }

    void rows(py::array_t<double, py::array::c_style | py::array::forcecast> x_obs,
              py::array_t<double, py::array::c_style | py::array::forcecast> a_obs,
              py::array_t<int64_t, py::array::c_style | py::array::forcecast> lab_obs,
              py::array out) {
        const int64_t m = static_cast<int64_t>(x_obs.size());
        if (a_obs.size() != m || lab_obs.size() != m)
            throw std::runtime_error(
                "ExactKernelPointCrossRows.rows: x_obs, a_obs, lab_obs must share a length");
        if (out.ndim() != 3 || out.shape(0) != impl_.nq() || out.shape(1) != m ||
            out.shape(2) != impl_.n())
            throw std::runtime_error("ExactKernelPointCrossRows.rows: out must be (nq, m, n)");
        if (!out.dtype().is(py::dtype::of<std::complex<double>>()) ||
            !(out.flags() & py::array::c_style) || !out.writeable())
            throw std::runtime_error(
                "ExactKernelPointCrossRows.rows: out must be complex128, C-contiguous, "
                "writeable");
        impl_.rows(x_obs.data(), a_obs.data(), lab_obs.data(), m,
                   static_cast<cd *>(out.mutable_data()));
    }

  private:
    PointCrossRowsCore impl_;

    static PointCrossRowsCore make(
        py::array_t<double, py::array::c_style | py::array::forcecast> &x0,
        py::array_t<double, py::array::c_style | py::array::forcecast> &sgn,
        py::array_t<double, py::array::c_style | py::array::forcecast> &h,
        py::array_t<double, py::array::c_style | py::array::forcecast> &a,
        py::array_t<int64_t, py::array::c_style | py::array::forcecast> &lab, double k, int nq,
        int64_t max_cached, double q, double qa, Rules &rules, int n_threads) {
        if (nq < 1 || nq > MAXND)
            throw std::runtime_error("ExactKernelPointCrossRows: nq must be in [1, 6]");
        const py::ssize_t n = x0.size();
        if (sgn.size() != n || h.size() != n || a.size() != n || lab.size() != n)
            throw std::runtime_error(
                "ExactKernelPointCrossRows: x0, sgn, h, a, lab must share a length");
        if (!(q > 0.0) || !(qa > 0.0))
            throw std::runtime_error("ExactKernelPointCrossRows: q and qa must be positive");
        return PointCrossRowsCore(x0.data(), sgn.data(), h.data(), a.data(), lab.data(),
                                  static_cast<int64_t>(n), k, nq, max_cached, q, qa, rules,
                                  n_threads);
    }
};

// ---------------------------------------------------------------------------
// The point-tested correction's assembly (razor-2p's wing algebra, #1433):
//
//   Z[m, n] += cA_r * sum_{e in n} t_e dmom_e(o_r)  -  cP_r * sum_{e in n} q_e dM0(o_r, j_e)
//
// for every row entry r = (m, o, cA, cP) and every column n with its wing
// entries e = (j, t, rising, q): dmom = dM1/h on a rising wing, dM0 - dM1/h on
// a falling one. Parallel over COLUMNS, so no two threads write one entry:
// Z is column-major and a column's writes are one thread's.
// ---------------------------------------------------------------------------
static void assemble_py(py::array_t<std::complex<double>, py::array::c_style> dM0,
                        py::array_t<std::complex<double>, py::array::c_style> dM1,
                        py::array_t<int64_t, py::array::c_style> row_m,
                        py::array_t<int64_t, py::array::c_style> row_o,
                        py::array_t<std::complex<double>, py::array::c_style> row_cA,
                        py::array_t<std::complex<double>, py::array::c_style> row_cP,
                        py::array_t<int64_t, py::array::c_style> col_ptr,
                        py::array_t<int64_t, py::array::c_style> col_n,
                        py::array_t<int64_t, py::array::c_style> ent_j,
                        py::array_t<double, py::array::c_style> ent_t,
                        py::array_t<uint8_t, py::array::c_style> ent_rise,
                        py::array_t<double, py::array::c_style> ent_q,
                        py::array_t<double, py::array::c_style> h, py::array Z,
                        int n_threads) {
    if (dM0.ndim() != 2 || dM1.ndim() != 2 || dM0.shape(0) != dM1.shape(0) ||
        dM0.shape(1) != dM1.shape(1))
        throw std::runtime_error("point assemble: dM0, dM1 must share a (r, n) shape");
    const int64_t r = dM0.shape(0), n = dM0.shape(1);
    if (Z.ndim() != 2 || !Z.dtype().is(py::dtype::of<std::complex<double>>()) ||
        !(Z.flags() & py::array::f_style) || !Z.writeable())
        throw std::runtime_error("point assemble: Z must be complex128, Fortran order, writeable");
    const int64_t ld = Z.shape(0), ncol = Z.shape(1);
    const int64_t R = row_m.size(), NB = col_n.size();
    if (row_o.size() != R || row_cA.size() != R || row_cP.size() != R ||
        col_ptr.size() != NB + 1 || ent_t.size() != ent_j.size() ||
        ent_rise.size() != ent_j.size() || ent_q.size() != ent_j.size() || h.size() != n)
        throw std::runtime_error("point assemble: inconsistent table lengths");
    const cd *M0 = dM0.data(), *M1 = dM1.data();
    const int64_t *rm = row_m.data(), *ro = row_o.data(), *cp = col_ptr.data(),
                  *cn = col_n.data(), *ej = ent_j.data();
    const cd *cA = row_cA.data(), *cP = row_cP.data();
    const double *et = ent_t.data(), *eq = ent_q.data(), *H = h.data();
    const uint8_t *er = ent_rise.data();
    for (int64_t i = 0; i < R; ++i)
        if (rm[i] < 0 || rm[i] >= ld || ro[i] < 0 || ro[i] >= r)
            throw std::runtime_error("point assemble: row entry out of range");
    for (int64_t b = 0; b < NB; ++b) {
        if (cn[b] < 0 || cn[b] >= ncol || cp[b] > cp[b + 1])
            throw std::runtime_error("point assemble: column entry out of range");
        for (int64_t e = cp[b]; e < cp[b + 1]; ++e)
            if (ej[e] < 0 || ej[e] >= n) throw std::runtime_error("point assemble: bad segment");
    }
    cd *Zp = static_cast<cd *>(Z.mutable_data());
    const int nt = resolve_threads(n_threads);
    py::gil_scoped_release release;
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

}  // namespace exact_kernel_point_1433

void register_exact_kernel_point(py::module_ &m) {
    using namespace exact_kernel_point_1433;
    py::class_<Rules>(m, "ExactKernelPointRules")
        .def(py::init(&make_rules), py::arg("x_gl"), py::arg("w_gl"), py::arg("x_split"),
             py::arg("w_split"), py::arg("wlog_split"));
    m.def("exact_kernel_point_moments", &point_moments_py,
          "(nq,) exact-kernel point moments N_q(c, h) of one segment. momwire#1433.",
          py::arg("c"), py::arg("h"), py::arg("a"), py::arg("k"), py::arg("nq"),
          py::arg("rules"));
    py::class_<PointRowsAccel>(m, "ExactKernelPointRows")
        .def(py::init<py::array_t<double, py::array::c_style | py::array::forcecast>,
                      py::array_t<double, py::array::c_style | py::array::forcecast>,
                      py::array_t<double, py::array::c_style | py::array::forcecast>, double,
                      double, int, int64_t, double, Rules, int>(),
             py::arg("x0"), py::arg("sgn"), py::arg("h"), py::arg("a"), py::arg("k"),
             py::arg("nq"), py::arg("max_cached"), py::arg("q"), py::arg("rules"),
             py::arg("n_threads") = 0)
        .def("rows", &PointRowsAccel::rows,
             "Write the moments at observers x_obs (m,) into out (nq, m, n), complex128, "
             "C-contiguous.",
             py::arg("x_obs"), py::arg("out"))
        .def_property_readonly("evaluated", &PointRowsAccel::evaluated)
        .def_property_readonly("cached", &PointRowsAccel::cached)
        .def_property_readonly("n", &PointRowsAccel::n);
    m.def("exact_kernel_point_assemble", &assemble_py,
          "Z += the point-tested correction of one observer window. momwire#1433.",
          py::arg("dM0"), py::arg("dM1"), py::arg("row_m"), py::arg("row_o"),
          py::arg("row_cA"), py::arg("row_cP"), py::arg("col_ptr"), py::arg("col_n"),
          py::arg("ent_j"), py::arg("ent_t"), py::arg("ent_rise"), py::arg("ent_q"),
          py::arg("h"), py::arg("Z"), py::arg("n_threads") = 0);
    m.def("exact_kernel_point_moments2", &point_moments2_py,
          "(nq,) two-radius point moments of one segment (observer ring a1, source a2). "
          "momwire#1438.",
          py::arg("c"), py::arg("h"), py::arg("a1"), py::arg("a2"), py::arg("k"), py::arg("nq"),
          py::arg("rules"));
    m.def("exact_kernel_ring_kernel2", &ring_kernel2_py,
          "The two-radius ring kernel K2(z; a1, a2) at each z. momwire#1438.", py::arg("z"),
          py::arg("a1"), py::arg("a2"), py::arg("k"));
    py::class_<PointCrossRowsAccel>(m, "ExactKernelPointCrossRows")
        .def(py::init<py::array_t<double, py::array::c_style | py::array::forcecast>,
                      py::array_t<double, py::array::c_style | py::array::forcecast>,
                      py::array_t<double, py::array::c_style | py::array::forcecast>,
                      py::array_t<double, py::array::c_style | py::array::forcecast>,
                      py::array_t<int64_t, py::array::c_style | py::array::forcecast>, double,
                      int, int64_t, double, double, Rules, int>(),
             py::arg("x0"), py::arg("sgn"), py::arg("h"), py::arg("a"), py::arg("lab"),
             py::arg("k"), py::arg("nq"), py::arg("max_cached"), py::arg("q"), py::arg("qa"),
             py::arg("rules"), py::arg("n_threads") = 0)
        .def("rows", &PointCrossRowsAccel::rows,
             "Write the two-radius moments at observers x_obs (radii a_obs, labels lab_obs) "
             "into out (nq, m, n), complex128, C-contiguous; zero on same-label pairs.",
             py::arg("x_obs"), py::arg("a_obs"), py::arg("lab_obs"), py::arg("out"))
        .def_property_readonly("evaluated", &PointCrossRowsAccel::evaluated)
        .def_property_readonly("cached", &PointCrossRowsAccel::cached)
        .def_property_readonly("n", &PointCrossRowsAccel::n);
    m.attr("exact_kernel_point_1433") = true;
    m.attr("exact_kernel_point_cross_1438") = true;
}
