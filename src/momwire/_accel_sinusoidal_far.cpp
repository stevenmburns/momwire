#include "_accel_common.h"
#include "_mw_nd_binding.h"
#include "_sinusoidal_far_core.h"

// NEC-2's interaction approximation (`nec2_far_approx`): the
// bindings of `_sinusoidal_far_core.cpp`. Its own TU, as the core is, so that
// no shipped kernel's codegen can move with it.

typedef py::array_t<double, py::array::c_style | py::array::forcecast> F64;
typedef py::array_t<std::complex<double>, py::array::c_style | py::array::forcecast> C128;
typedef std::complex<double> cd;
typedef std::tuple<py::array_t<cd>, py::array_t<cd>, py::array_t<cd>> Tensors;

static Tensors sinusoidal_field_tensor_far(F64 obs_centers, F64 obs_tangents, F64 src_centers,
                                           F64 src_tangents, F64 seg_h, double a, double k, double eta,
                                           F64 gl_t, F64 gl_w, double far_r, uintptr_t cancel_flag = 0) {
    ndb::NumpyHost host;
    mw::nd::HostScope scope(host);
    return ndb::out_tuple(mw::sinusoidal::sinusoidal_field_tensor_far(
        ndb::in(obs_centers), ndb::in(obs_tangents), ndb::in(src_centers), ndb::in(src_tangents),
        ndb::in(seg_h), a, k, eta, ndb::in(gl_t), ndb::in(gl_w), far_r, cancel_flag));
}

static Tensors sinusoidal_field_tensor_refl_far(F64 obs_centers, F64 obs_tangents, F64 src_centers,
                                                F64 src_tangents, F64 seg_h, double a, double k,
                                                double eta, F64 gl_t, F64 gl_w, F64 cos_th, F64 px,
                                                F64 py_, F64 tm_p, F64 tn_p, cd eps_t, double far_r,
                                                uintptr_t cancel_flag = 0) {
    ndb::NumpyHost host;
    mw::nd::HostScope scope(host);
    return ndb::out_tuple(mw::sinusoidal::sinusoidal_field_tensor_refl_far(
        ndb::in(obs_centers), ndb::in(obs_tangents), ndb::in(src_centers), ndb::in(src_tangents),
        ndb::in(seg_h), a, k, eta, ndb::in(gl_t), ndb::in(gl_w), ndb::in(cos_th), ndb::in(px),
        ndb::in(py_), ndb::in(tm_p), ndb::in(tn_p), eps_t, far_r, cancel_flag));
}

static Tensors sinusoidal_far_overlay(C128 base_const, C128 base_sin, C128 base_cos, F64 obs_centers,
                                      F64 obs_tangents, F64 src_centers, F64 src_tangents, F64 seg_h,
                                      double a, double k, double eta, bool refl, bool cos_minus_one,
                                      F64 cos_th, F64 px,
                                      F64 py_, F64 tm_p, F64 tn_p, cd eps_t, double far_r,
                                      uintptr_t cancel_flag = 0) {
    ndb::NumpyHost host;
    mw::nd::HostScope scope(host);
    return ndb::out_tuple(mw::sinusoidal::sinusoidal_far_overlay(
        ndb::in(base_const), ndb::in(base_sin), ndb::in(base_cos), ndb::in(obs_centers),
        ndb::in(obs_tangents), ndb::in(src_centers), ndb::in(src_tangents), ndb::in(seg_h), a, k, eta,
        refl, cos_minus_one, ndb::in(cos_th), ndb::in(px), ndb::in(py_), ndb::in(tm_p), ndb::in(tn_p), eps_t, far_r,
        cancel_flag));
}

static int64_t far_range_ties(F64 obs_centers, F64 src_centers, double far_r, double rel) {
    ndb::NumpyHost host;
    mw::nd::HostScope scope(host);
    return mw::sinusoidal::far_range_ties(ndb::in(obs_centers), ndb::in(src_centers), far_r, rel);
}

void register_sinusoidal_far(py::module_ &m) {
    m.def("sinusoidal_field_tensor_far", &sinusoidal_field_tensor_far,
          "sinusoidal_field_tensor with NEC-2's interaction approximation "
          "(`nec2_far_approx`): a pair whose segment centres are more than far_r "
          "apart takes the source's lumped (infinitesimal-dipole) field.",
          py::arg("obs_centers"), py::arg("obs_tangents"), py::arg("src_centers"),
          py::arg("src_tangents"), py::arg("seg_h"), py::arg("a"), py::arg("k"), py::arg("eta"),
          py::arg("gl_t"), py::arg("gl_w"), py::arg("far_r"), py::arg("cancel_flag") = 0);
    m.def("sinusoidal_field_tensor_refl_far", &sinusoidal_field_tensor_refl_far,
          "sinusoidal_field_tensor_refl with NEC-2's interaction approximation "
          "(`nec2_far_approx`).",
          py::arg("obs_centers"), py::arg("obs_tangents"), py::arg("src_centers"),
          py::arg("src_tangents"), py::arg("seg_h"), py::arg("a"), py::arg("k"), py::arg("eta"),
          py::arg("gl_t"), py::arg("gl_w"), py::arg("cos_th"), py::arg("px"), py::arg("py"),
          py::arg("tm_p"), py::arg("tn_p"), py::arg("eps_t"), py::arg("far_r"),
          py::arg("cancel_flag") = 0);
    m.def("sinusoidal_far_overlay", &sinusoidal_far_overlay,
          "The far pairs' lumped tensors over a shipped kernel's base tensors "
          "(`nec2_far_approx`), the near pairs copied.",
          py::arg("base_const"), py::arg("base_sin"), py::arg("base_cos"), py::arg("obs_centers"),
          py::arg("obs_tangents"), py::arg("src_centers"), py::arg("src_tangents"), py::arg("seg_h"),
          py::arg("a"), py::arg("k"), py::arg("eta"), py::arg("refl"), py::arg("cos_minus_one"),
          py::arg("cos_th"),
          py::arg("px"), py::arg("py"), py::arg("tm_p"), py::arg("tn_p"), py::arg("eps_t"),
          py::arg("far_r"), py::arg("cancel_flag") = 0);
    m.def("far_range_ties", &far_range_ties,
          "How many pairs have a centre distance within rel of far_r.",
          py::arg("obs_centers"), py::arg("src_centers"), py::arg("far_r"), py::arg("rel"));
}
