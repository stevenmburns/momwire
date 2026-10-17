#include "_accel_common.h"
#include "_somm_pairs_core.h"

#include <limits>
#include <stdexcept>
#ifdef _OPENMP
#include <omp.h>
#endif

// `_quadrature.remainder_qp_pairs`' per-source scan, off the Python loop
// (momwire queue item 4).
//
// The numpy body is one Python iteration per SOURCE segment, each a dozen
// array passes over every observer node of the deck: on SG's inverted-L x16
// (2,930 segments) that was 1.7 s of a 16 s solve, most of it the passes'
// memory traffic and 35k ufunc reductions. Here one source's scan is one pass
// over the observer nodes, OpenMP across sources.
//
// THE ANSWER IS AN INTEGER ORDER, AND IT IS NUMPY'S EXACTLY. Not by
// transcribing numpy's floats -- the projection `ap @ d[j]` is a BLAS gemv,
// whose summation order and fusion are the BLAS build's, not ours -- but by
// knowing where ours could disagree. Every float here is within a few
// rounding errors of numpy's, and the order is a STEP function of them: it
// moves only where a foot crosses a segment end (t_raw against 0 or 1, or
// the edge tolerance), or where c * len / R_min crosses an integer, or where
// R_min reaches zero. So the scan computes, beside each decision, a bound on
// how far numpy's float could sit from ours, and a source with ANY decision
// inside its bound is flagged UNCERTAIN instead of answered. The caller runs
// the numpy body for exactly those sources. Every source this returns is
// therefore numpy's answer, and every other source IS the numpy body -- the
// list is the numpy route's list on every build, whatever its BLAS does.
//
// The bounds are deliberately loose (1e-10 relative, against an arithmetic
// whose disagreement is a few times 1e-16): an uncertain source costs one
// numpy iteration, a wrong certain one costs the identity.
//
// On a hub, a radial's end lands on another radial's line exactly, so
// those few sources are the uncertain ones (the #1201 comment's last bit of
// a cos(pi/2)); everything else is answered here.

namespace qp_pairs {

// The scan and its relative slack are `_somm_pairs_core.cpp`'s
// (momwire#1517 step 3).
using Hit = somm_core::QpHit;

}  // namespace qp_pairs

// obs: (n_obs_seg * n_node, 3) observer nodes, grouped by segment.
// mir_l, d: (S, 3) mirrored source start and mirrored direction; dd, lengths:
// (S,). Returns (I, J, Q, uncertain): the listed pairs of every CERTAIN
// source, J ascending and I ascending within a source (the numpy loop's
// order), and the sources the caller must run through the numpy body.
static py::tuple remainder_qp_pairs_scan(
    py::array_t<double, py::array::c_style | py::array::forcecast> obs,
    int64_t n_node,
    py::array_t<double, py::array::c_style | py::array::forcecast> mir_l,
    py::array_t<double, py::array::c_style | py::array::forcecast> d,
    py::array_t<double, py::array::c_style | py::array::forcecast> dd,
    py::array_t<double, py::array::c_style | py::array::forcecast> lengths,
    int64_t base, int64_t cap, double c, double edge_tol) {
    using qp_pairs::Hit;
    if (obs.ndim() != 2 || obs.shape(1) != 3 || mir_l.ndim() != 2 ||
        mir_l.shape(1) != 3 || d.ndim() != 2 || d.shape(1) != 3)
        throw std::invalid_argument("obs, mir_l and d must be (*, 3)");
    const py::ssize_t S = mir_l.shape(0);
    if (d.shape(0) != S || dd.ndim() != 1 || dd.shape(0) != S ||
        lengths.ndim() != 1 || lengths.shape(0) != S)
        throw std::invalid_argument("mir_l, d, dd and lengths must agree in length");
    const py::ssize_t M = obs.shape(0);
    if (n_node < 1 || M % n_node != 0)
        throw std::invalid_argument("obs rows must be a whole number of segments");
    const double *O = obs.data();
    const double *ML = mir_l.data();
    const double *D = d.data();
    const double *DD = dd.data();
    const double *L = lengths.data();

    std::vector<std::vector<Hit>> hits;
    std::vector<unsigned char> unsure;
    {
        py::gil_scoped_release release;
        somm_core::remainder_qp_pairs_scan(O, M, n_node, ML, D, DD, L, S, base, cap, c, edge_tol, hits, unsure);
    }

    size_t total = 0;
    int64_t n_unsure = 0;
    for (py::ssize_t j = 0; j < S; ++j) {
        total += hits[j].size();
        n_unsure += unsure[j];
    }
    py::array_t<int64_t> I(static_cast<py::ssize_t>(total)),
        J(static_cast<py::ssize_t>(total)), Q(static_cast<py::ssize_t>(total)),
        U(static_cast<py::ssize_t>(n_unsure));
    int64_t *ip = I.mutable_data(), *jp = J.mutable_data(), *qp = Q.mutable_data(),
            *up = U.mutable_data();
    size_t w = 0;
    int64_t u = 0;
    for (py::ssize_t j = 0; j < S; ++j) {
        if (unsure[j]) up[u++] = j;
        for (const Hit &h : hits[j]) {
            ip[w] = h.row;
            jp[w] = j;
            qp[w] = h.q;
            ++w;
        }
    }
    return py::make_tuple(I, J, Q, U);
}

void register_qp_pairs(py::module_ &m) {
    m.def("remainder_qp_pairs_scan", &remainder_qp_pairs_scan,
          "`_quadrature.remainder_qp_pairs`' per-source scan: (I, J, Q) of the "
          "certain sources and the sources the numpy body must decide "
          "(see _accel_qp_pairs.cpp).",
          py::arg("obs"), py::arg("n_node"), py::arg("mir_l"), py::arg("d"),
          py::arg("dd"), py::arg("lengths"), py::arg("base"), py::arg("cap"),
          py::arg("c"), py::arg("edge_tol"));
}
