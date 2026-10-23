#include "_accel_common.h"
#include "_mw_nd_binding.h"
#include "_sinusoidal_nec4_core.h"

#include <array>
#include <stdexcept>
#include <vector>

// The NEC-4 dialect's radius-step junction (momwire#1644): the bindings of
// `_sinusoidal_nec4_core.cpp`. Its own TU, as the core is, so that no
// existing kernel's codegen can move with it.

typedef py::array_t<double, py::array::c_style | py::array::forcecast> F64;
typedef std::complex<double> cd;

static std::tuple<py::array_t<cd>, py::array_t<cd>, py::array_t<cd>>
sinusoidal_field_tensor_nec4(F64 obs_centers, F64 obs_tangents, F64 src_centers, F64 src_tangents, F64 seg_h,
                             F64 src_a, double k, double eta, F64 gl_t, F64 gl_w, uintptr_t cancel_flag = 0) {
    ndb::NumpyHost host;
    mw::nd::HostScope scope(host);
    return ndb::out_tuple(mw::sinusoidal::sinusoidal_field_tensor_nec4(
        ndb::in(obs_centers), ndb::in(obs_tangents), ndb::in(src_centers), ndb::in(src_tangents), ndb::in(seg_h),
        ndb::in(src_a), k, eta, ndb::in(gl_t), ndb::in(gl_w), cancel_flag));
}

static py::array_t<cd> sinusoidal_cap_field(F64 obs_centers, F64 obs_tangents, F64 pos, F64 normal, double a,
                                            double k, double eta, cd c0) {
    ndb::NumpyHost host;
    mw::nd::HostScope scope(host);
    return ndb::out(mw::sinusoidal::sinusoidal_cap_field(ndb::in(obs_centers), ndb::in(obs_tangents), ndb::in(pos),
                                                         ndb::in(normal), a, k, eta, c0));
}

static std::vector<double> nec4_junction_factors(double k, const std::vector<std::array<double, 3>> &dirs,
                                                 const std::vector<double> &radii,
                                                 const std::vector<double> &lengths, double first_factor) {
    try {
        return mw::sinusoidal::junction_charge_factors(k, dirs, radii, lengths, first_factor);
    } catch (const std::domain_error &e) {
        throw py::value_error(e.what());
    }
}

void register_sinusoidal_nec4(py::module_ &m) {
    m.def("sinusoidal_field_tensor_nec4", &sinusoidal_field_tensor_nec4,
          "The NEC-4 thin-wire kernel for the point-matched sinusoidal basis "
          "(momwire#1644): (Phi_const, Phi_sin, Phi_cos), each (M, N), with the "
          "SOURCE radius per segment (`src_a`), no segment-end charges and bare "
          "1/rho radial forms. `_sinusoidal_nec4.nec4_field_components` is the "
          "numpy reference.",
          py::arg("obs_centers"), py::arg("obs_tangents"), py::arg("src_centers"), py::arg("src_tangents"),
          py::arg("seg_h"), py::arg("src_a"), py::arg("k"), py::arg("eta"), py::arg("gl_t"), py::arg("gl_w"),
          py::arg("cancel_flag") = 0);
    m.def("sinusoidal_cap_field", &sinusoidal_cap_field,
          "The tangential field (M,) of one charged flat end cap (momwire#1644, "
          "`_sinusoidal_nec4.cap_field` is the numpy reference).",
          py::arg("obs_centers"), py::arg("obs_tangents"), py::arg("pos"), py::arg("normal"), py::arg("a"),
          py::arg("k"), py::arg("eta"), py::arg("c0"));
    m.def("nec4_junction_factors", &nec4_junction_factors,
          "The C++ front half's copy of `_nec4_junction.junction_charge_factors` "
          "(momwire#1644), for the test that holds the two to the bit.",
          py::arg("k"), py::arg("dirs"), py::arg("radii"), py::arg("lengths"), py::arg("first_factor"));
}
