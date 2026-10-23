// momwire#1224 (sinusoidal buried, stage 3), unit 1: the binding of the
// C++ twin of `_near_interface.point_keys_columns` -- the four POINT-observer
// integrals (POINT_KEYS: dRhoV, dzV, dRhoW, dz'V) over concatenated rho
// columns. The kernel itself is `_near_interface_point_core.cpp`
// (momwire#1653: moved there verbatim, Python-free, so the buried near-field
// readout and the standalone executable call the same code). The numpy route
// stays the reference; the gate against it is RELATIVE (1e-13), never bit.

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

#include <complex>
#include <cstdint>
#include <stdexcept>

#include "_near_interface_point_core.h"

namespace py = pybind11;

// `point_keys_columns` over CONCATENATED columns, `near_interface_six_columns`'
// contract with four columns out instead of six: column c owns members
// offsets[c] .. offsets[c+1] of (z, zp), all at rho[c]; the answer keeps the
// members' order.
static py::array_t<std::complex<double>> near_interface_point_columns(
    double k_p, std::complex<double> k_m,
    py::array_t<double, py::array::c_style | py::array::forcecast> rho,
    py::array_t<py::ssize_t, py::array::c_style | py::array::forcecast> offsets,
    py::array_t<double, py::array::c_style | py::array::forcecast> z,
    py::array_t<double, py::array::c_style | py::array::forcecast> zp,
    double lam_mult, int p, double detour, int n_threads,
    py::array_t<double, py::array::c_style | py::array::forcecast> gx,
    py::array_t<double, py::array::c_style | py::array::forcecast> gw,
    bool sdp) {
    if (rho.ndim() != 1 || offsets.ndim() != 1 || z.ndim() != 1 ||
        zp.ndim() != 1)
        throw std::invalid_argument("rho, offsets, z and zp must be 1-D");
    const py::ssize_t nc = rho.shape(0);
    const py::ssize_t n = z.shape(0);
    if (offsets.shape(0) != nc + 1)
        throw std::invalid_argument("offsets must have len(rho) + 1 entries");
    if (zp.shape(0) != n)
        throw std::invalid_argument("z and zp must have the same length");
    if (gx.ndim() != 1 || gw.ndim() != 1 || gx.shape(0) != gw.shape(0) ||
        gx.shape(0) < 1)
        throw std::invalid_argument("bad Gauss rule");
    py::array_t<std::complex<double>> vals({n, py::ssize_t(ni_point::N_POINT_KEYS)});
    std::complex<double> *out = vals.mutable_data();
    const double *r = rho.data();
    const py::ssize_t *o = offsets.data();
    const double *zz = z.data();
    const double *pp = zp.data();
    const double *gxp = gx.data();
    const double *gwp = gw.data();
    const int ng = static_cast<int>(gx.shape(0));
    {
        py::gil_scoped_release release;
        ni_point::point_columns(k_p, k_m, r, reinterpret_cast<const std::ptrdiff_t *>(o), nc, zz, pp,
                                n, lam_mult, p, detour, n_threads, gxp, gwp, ng, out, sdp);
    }
    return vals;
}

// Its OWN extension, not a second TU of `_near_interface_accel`: keeping the
// six twin's module untouched is the point. The name is a build parameter,
// as the sibling's is (momwire#1032's AVX2 / baseline pair).
#ifndef MOMWIRE_MODULE_NAME
#define MOMWIRE_MODULE_NAME _near_interface_point_accel
#endif

PYBIND11_MODULE(MOMWIRE_MODULE_NAME, m) {
    m.doc() =
        "momwire#1224: the C++ column twin of "
        "_near_interface.point_keys_columns. Optional; _near_interface falls "
        "back to the numpy route without it.";
    // Its own flag: the capability `_near_interface` keys the route on.
    m.attr("point_columns_1224") = true;
    // momwire#1653: far columns on the steepest-descent rule (D1b's route);
    // `sdp=False` is the head / mid / tail rule everywhere.
    m.attr("point_columns_sdp_1653") = true;
    m.def("near_interface_point_columns", &near_interface_point_columns,
          py::arg("k_p"), py::arg("k_m"), py::arg("rho"), py::arg("offsets"),
          py::arg("z"), py::arg("zp"), py::arg("lam_mult"), py::arg("p"),
          py::arg("detour"), py::arg("n_threads"), py::arg("gx"),
          py::arg("gw"), py::arg("sdp") = true,
          "point_keys_columns over CONCATENATED columns -> (n, 4) complex in "
          "POINT_KEYS order. near_interface_six_columns' contract otherwise.");
}
