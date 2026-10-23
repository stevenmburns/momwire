// The buried near-field readout's binding (momwire#1653): `near_buried_parts`
// is `_near_buried_core.cpp`'s `parts`, the regimes `_near_buried.py`
// composes across and below a finite ground. The Python route calls it when
// it is built (`_near_buried._CXX`), so the launchers and the standalone
// executables answer from the same code; `_near_buried.py`'s numpy spelling
// is the reference the gates hold it to.
#include "_accel_common.h"
#include "_near_buried_core.h"
#include "_somm_below_fill_core.h"
#include "_somm_below_grid_core.h"

#include <complex>
#include <memory>
#include <stdexcept>
#include <vector>

namespace {

using cd = std::complex<double>;

// The process's below/below grid source for this readout (the in-process
// memo of exact grids `fill_source` keeps), installed for each call when no
// other caller has one installed, and put back after.
std::shared_ptr<somm_below::Source> &readout_source() {
    static std::shared_ptr<somm_below::Source> s = somm_below::fill_source();
    return s;
}

py::array_t<cd> near_buried_parts(py::array_t<double, py::array::c_style | py::array::forcecast> points,
                                  py::array_t<double, py::array::c_style | py::array::forcecast> mid,
                                  py::array_t<cd, py::array::c_style | py::array::forcecast> moment,
                                  py::array_t<double, py::array::c_style | py::array::forcecast> nodes,
                                  py::array_t<cd, py::array::c_style | py::array::forcecast> delta,
                                  py::array_t<int64_t, py::array::c_style | py::array::forcecast> counts,
                                  int subdiv, cd eps_t, double ground_z, double k, double omega, double mu,
                                  double radius, bool magnetic, int n_threads) {
    if (points.ndim() != 2 || points.shape(1) != 3) throw std::invalid_argument("points must be (n, 3)");
    near_buried::Elements el;
    el.mid = mid.data();
    el.moment = moment.data();
    el.n_el = static_cast<size_t>(mid.shape(0));
    el.nodes = nodes.data();
    el.delta = delta.data();
    el.n_nodes = static_cast<size_t>(nodes.shape(0));
    el.counts = counts.data();
    el.n_wires = static_cast<size_t>(counts.shape(0));
    el.subdiv = subdiv;
    size_t total = 0;
    for (size_t w = 0; w < el.n_wires; w++) total += static_cast<size_t>(el.counts[w]);
    if (total != el.n_el || total + el.n_wires != el.n_nodes)
        throw std::invalid_argument("element counts do not match the element walk");
    const size_t n = static_cast<size_t>(points.shape(0));
    py::array_t<cd> out({static_cast<py::ssize_t>(n), py::ssize_t(3)});
    cd *o = out.mutable_data();
    const double *p = points.data();
    {
        py::gil_scoped_release release;
        struct Restore {
            bool installed = false;
            ~Restore() {
                if (installed) somm_below::set_source(nullptr);
            }
        } restore;
        if (somm_below::source() == nullptr) {
            somm_below::set_source(readout_source());
            restore.installed = true;
        }
        near_buried::parts(p, n, el, eps_t, ground_z, k, omega, mu, radius, magnetic, n_threads, o);
    }
    return out;
}

}  // namespace

void register_near_buried(py::module_ &m) {
    m.attr("near_buried_1653") = true;
    m.def("near_buried_parts", &near_buried_parts, py::arg("points"), py::arg("mid"), py::arg("moment"),
          py::arg("nodes"), py::arg("delta"), py::arg("counts"), py::arg("subdiv"), py::arg("eps_t"),
          py::arg("ground_z"), py::arg("k"), py::arg("omega"), py::arg("mu"), py::arg("radius"),
          py::arg("magnetic"), py::arg("n_threads"),
          "momwire#1653: the near field's two new regimes at every point -- the "
          "transmitted field of the buried sources at a point at or above the "
          "plane (the caller adds the above/above composition first), below(...) "
          "+ the transmitted field of the above sources at a point below it. "
          "(n, 3) complex; counts is each wire's element count at `subdiv`.");
}
