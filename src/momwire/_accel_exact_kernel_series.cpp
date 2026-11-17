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

#define MW_EK_RELEASE_GIL py::gil_scoped_release release;

// The series and its row source: one text with the Python-free core TU.
#include "_exact_kernel_series_inline.h"

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
        : impl_(make(x0, sgn, h, a, nq, max_cached, q, rules, x_max, x_centred, tol,
                     n_threads)) {}

    int64_t n() const { return impl_.n(); }
    int64_t built() const { return impl_.built(); }
    int64_t reused() const { return impl_.reused(); }
    int64_t evaluated() const { return impl_.evaluated(); }
    int64_t cached() const { return impl_.cached(); }
    int64_t nbytes() const { return impl_.nbytes(); }
    py::tuple kinds() const {
        return py::make_tuple(impl_.kind_count(0), impl_.kind_count(1), impl_.kind_count(2));
    }

    void rows(darr x_obs, std::complex<double> k, py::array out) {
        const int64_t n = this->n();
        const int nq = impl_.nq();
        const int64_t m = static_cast<int64_t>(x_obs.size());
        if (out.ndim() != 3 || out.shape(0) != nq || out.shape(1) != m || out.shape(2) != n)
            throw std::runtime_error("ExactKernelPointSeries.rows: out must be (nq, m, n)");
        if (!out.dtype().is(py::dtype::of<std::complex<double>>()) ||
            !(out.flags() & py::array::c_style) || !out.writeable())
            throw std::runtime_error(
                "ExactKernelPointSeries.rows: out must be complex128, C-contiguous, writeable");
        impl_.rows(x_obs.data(), m, k, static_cast<cd *>(out.mutable_data()));
    }

  private:
    static PointSeriesCore make(darr x0, darr sgn, darr h, double a, int nq, int64_t max_cached,
                                double q, const Rules &rules, double x_max, double x_centred,
                                double tol, int n_threads) {
        const py::ssize_t n = x0.size();
        if (sgn.size() != n || h.size() != n)
            throw std::runtime_error("ExactKernelPointSeries: x0, sgn, h must share a length");
        return PointSeriesCore(x0.data(), sgn.data(), h.data(), static_cast<int64_t>(n), a, nq,
                               max_cached, q, rules, x_max, x_centred, tol, n_threads);
    }
    PointSeriesCore impl_;
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
