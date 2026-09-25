#include "_accel_common.h"
#include "_accel_somm_proj_inline.h"

// momwire#1201: the Sommerfeld remainder at LISTED (observer, source) pairs.
//
// `remainder_field_proj_batch` fills a full (M, S) table because the fixed-order
// rules share one source node set across every observer. The graded rule of
// `_remainder_graded` does not: each observer node carries its own source
// nodes, split and graded toward that node's image. This is the same
// `somm_proj::proj_one` evaluated at (obs[owner[n]], src[n]) for each n.
//
// Its own translation unit on purpose. `proj_one` is `static inline` and the
// two existing callers live in _accel_somm.cpp; a third caller there could move
// the compiler's unit-wide inlining budget and with it the codegen of kernels
// every Sommerfeld solve runs (the momwire#1194 hazard). Here it cannot.
namespace somm_pairs {

static py::array_t<std::complex<double>> remainder_field_proj_owned(
    py::array_t<double, py::array::c_style | py::array::forcecast> obs,
    py::array_t<double, py::array::c_style | py::array::forcecast> t_obs,
    py::array_t<double, py::array::c_style | py::array::forcecast> src,
    py::array_t<double, py::array::c_style | py::array::forcecast> t_src,
    py::array_t<int64_t, py::array::c_style | py::array::forcecast> owner,
    double ground_z, double k, double r1_max, double r_break, double th_split,
    double r_near,
    py::array_t<double, py::array::c_style | py::array::forcecast> reg_r0,
    py::array_t<double, py::array::c_style | py::array::forcecast> reg_dr,
    py::array_t<double, py::array::c_style | py::array::forcecast> reg_th0,
    py::array_t<double, py::array::c_style | py::array::forcecast> reg_dth,
    std::vector<py::array_t<std::complex<double>,
                            py::array::c_style | py::array::forcecast>> reg_vals,
    uintptr_t cancel_flag = 0) {
    using somm_proj::cd;
    auto ob = obs.unchecked<2>();
    auto tob = t_obs.unchecked<2>();
    auto sb = src.unchecked<2>();
    auto tsb = t_src.unchecked<2>();
    auto own = owner.unchecked<1>();
    if (ob.shape(1) != 3 || tob.shape(1) != 3 || sb.shape(1) != 3 ||
        tsb.shape(1) != 3)
        throw std::runtime_error("obs/src/tangent arrays must have shape (*, 3)");
    if (ob.shape(0) != tob.shape(0) || sb.shape(0) != tsb.shape(0) ||
        own.shape(0) != sb.shape(0))
        throw std::runtime_error(
            "points, tangents and owner must have matching lengths");
    const py::ssize_t M = ob.shape(0);
    const py::ssize_t S = sb.shape(0);
    for (py::ssize_t n = 0; n < S; ++n)
        if (own(n) < 0 || own(n) >= M)
            throw std::runtime_error("owner index out of range");

    somm_proj::GridView G = somm_proj::build_grid_view(
        r1_max, r_break, th_split, r_near, reg_r0.unchecked<1>(),
        reg_dr.unchecked<1>(), reg_th0.unchecked<1>(), reg_dth.unchecked<1>(),
        reg_vals);

    py::array_t<std::complex<double>> out(S);
    auto out_m = out.mutable_unchecked<1>();

    py::gil_scoped_release release;
    MW_CANCEL_SETUP(cancel_flag);
    #pragma omp parallel for schedule(static)
    for (py::ssize_t n = 0; n < S; ++n) {
        MW_CANCEL_POLL();
        const py::ssize_t m = own(n);
        double ux, uy, th, tz;
        somm_proj::tangent_decomp(tsb(n, 0), tsb(n, 1), tsb(n, 2), ux, uy, th,
                                  tz);
        out_m(n) = somm_proj::proj_one(G, ground_z, k, ob(m, 0), ob(m, 1),
                                       ob(m, 2), tob(m, 0), tob(m, 1), tob(m, 2),
                                       sb(n, 0), sb(n, 1), sb(n, 2), ux, uy, th,
                                       tz);
    }
    MW_THROW_IF_ABORTED();
    return out;
}

}  // namespace somm_pairs

void register_somm_pairs(py::module_ &m) {
    m.def("remainder_field_proj_owned", &somm_pairs::remainder_field_proj_owned,
          "Sommerfeld smooth-remainder t_m.F(r_m, r_n).t_n at listed pairs: "
          "entry n pairs src[n] with obs[owner[n]] (momwire#1201's graded "
          "rule). Same per-pair arithmetic as remainder_field_proj_batch; "
          "OpenMP over sources. Returns (S,) complex.",
          py::arg("obs"), py::arg("t_obs"), py::arg("src"), py::arg("t_src"),
          py::arg("owner"), py::arg("ground_z"), py::arg("k"),
          py::arg("r1_max"), py::arg("r_break"), py::arg("th_split"),
          py::arg("r_near"), py::arg("reg_r0"), py::arg("reg_dr"),
          py::arg("reg_th0"), py::arg("reg_dth"), py::arg("reg_vals"),
          py::arg("cancel_flag") = 0);
}
