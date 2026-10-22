#include "_accel_common.h"

#include <cstring>
#include <limits>
#include <map>
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

#include "_exact_kernel_ring_inline.h"

#define MW_EK_RELEASE_GIL py::gil_scoped_release release;
#include "_exact_kernel_coaxial_inline.h"
// the cross job across a radius step (momwire#1615), on the same rules
#include "_exact_kernel_ring2_inline.h"
#include "_exact_kernel_coaxial_cross_inline.h"

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
// CoaxialRows: the binding over `CoaxialRowsCore`
// ---------------------------------------------------------------------------
class CoaxialRowsAccel {
  public:
    CoaxialRowsAccel(py::array_t<double, py::array::c_style | py::array::forcecast> x0,
                     py::array_t<double, py::array::c_style | py::array::forcecast> sgn,
                     py::array_t<double, py::array::c_style | py::array::forcecast> h, double a,
                     double k, int nd, int64_t max_cached, double q, bool symmetric,
                     Rules rules, int n_threads)
        : core_(x0.data(), static_cast<int64_t>(x0.size()), sgn.data(),
                static_cast<int64_t>(sgn.size()), h.data(), static_cast<int64_t>(h.size()), a, k,
                nd, max_cached, q, symmetric, rules, n_threads),
          nd_(nd) {}

    int64_t n() const { return core_.n(); }
    int64_t evaluated() const { return core_.evaluated(); }
    int64_t cached() const { return core_.cached(); }

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
        core_.rows(r0, r1, static_cast<cd *>(out.mutable_data()));
    }

  private:
    CoaxialRowsCore core_;
    int nd_;
};

// ---------------------------------------------------------------------------
// CoaxialCrossRows: the binding over `CoaxialCrossRowsCore` (momwire#1615)
// ---------------------------------------------------------------------------
class CoaxialCrossRowsAccel {
  public:
    CoaxialCrossRowsAccel(py::array_t<double, py::array::c_style | py::array::forcecast> x0,
                          py::array_t<double, py::array::c_style | py::array::forcecast> sgn,
                          py::array_t<double, py::array::c_style | py::array::forcecast> h,
                          py::array_t<double, py::array::c_style | py::array::forcecast> a,
                          py::array_t<int64_t, py::array::c_style | py::array::forcecast> lab,
                          double k, int nd, int64_t max_cached, double q, double qa, Rules rules,
                          int n_threads)
        : core_(checked(x0, sgn, h, a, lab), sgn.data(), h.data(), a.data(), lab.data(),
                static_cast<int64_t>(x0.size()), k, nd, max_cached, q, qa, rules, n_threads),
          nd_(nd) {}

    int64_t n() const { return core_.n(); }
    int64_t evaluated() const { return core_.evaluated(); }
    int64_t cached() const { return core_.cached(); }

    void rows(int64_t r0, int64_t r1, py::array out) {
        const int64_t n = this->n();
        const int nd = nd_;
        if (r0 < 0 || r1 > n || r1 < r0)
            throw std::runtime_error("CoaxialCrossRowsAccel.rows: bad row window");
        const int64_t m = r1 - r0;
        if (out.ndim() != 4 || out.shape(0) != nd || out.shape(1) != nd || out.shape(2) != m ||
            out.shape(3) != n)
            throw std::runtime_error("CoaxialCrossRowsAccel.rows: out must be (nd, nd, r1-r0, n)");
        if (!out.dtype().is(py::dtype::of<std::complex<double>>()) ||
            !(out.flags() & py::array::c_style) || !out.writeable())
            throw std::runtime_error(
                "CoaxialCrossRowsAccel.rows: out must be complex128, C-contiguous, writeable");
        core_.rows(r0, r1, static_cast<cd *>(out.mutable_data()));
    }

  private:
    static const double *checked(
        const py::array_t<double, py::array::c_style | py::array::forcecast> &x0,
        const py::array_t<double, py::array::c_style | py::array::forcecast> &sgn,
        const py::array_t<double, py::array::c_style | py::array::forcecast> &h,
        const py::array_t<double, py::array::c_style | py::array::forcecast> &a,
        const py::array_t<int64_t, py::array::c_style | py::array::forcecast> &lab) {
        const py::ssize_t n = x0.size();
        if (sgn.size() != n || h.size() != n || a.size() != n || lab.size() != n)
            throw std::runtime_error("CoaxialCrossRowsAccel: x0, sgn, h, a, lab must share a length");
        return x0.data();
    }
    CoaxialCrossRowsCore core_;
    int nd_;
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
    py::class_<CoaxialCrossRowsAccel>(m, "ExactKernelCoaxialCrossRows")
        .def(py::init<py::array_t<double, py::array::c_style | py::array::forcecast>,
                      py::array_t<double, py::array::c_style | py::array::forcecast>,
                      py::array_t<double, py::array::c_style | py::array::forcecast>,
                      py::array_t<double, py::array::c_style | py::array::forcecast>,
                      py::array_t<int64_t, py::array::c_style | py::array::forcecast>, double, int,
                      int64_t, double, double, Rules, int>(),
             py::arg("x0"), py::arg("sgn"), py::arg("h"), py::arg("a"), py::arg("lab"),
             py::arg("k"), py::arg("nd"), py::arg("max_cached"), py::arg("q"), py::arg("qa"),
             py::arg("rules"), py::arg("n_threads") = 0)
        .def("rows", &CoaxialCrossRowsAccel::rows,
             "Write observer rows [r0, r1) of the line's two-radius moment block into out "
             "(nd, nd, r1 - r0, n), complex128, C-contiguous; zero on same-label pairs. "
             "momwire#1615.",
             py::arg("r0"), py::arg("r1"), py::arg("out"))
        .def_property_readonly("evaluated", &CoaxialCrossRowsAccel::evaluated)
        .def_property_readonly("cached", &CoaxialCrossRowsAccel::cached)
        .def_property_readonly("n", &CoaxialCrossRowsAccel::n);
    m.attr("exact_kernel_1410") = true;
}
