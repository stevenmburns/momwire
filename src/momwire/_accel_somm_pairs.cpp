#include "_accel_common.h"
#include "_accel_somm_proj_py.h"
#include "_somm_pairs_core.h"
#include "_somm_pairs_owned_core.h"

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
    py::array_t<std::complex<double>, py::array::c_style | py::array::forcecast>
        far,
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
    somm_proj::set_far(G, far);

    py::array_t<std::complex<double>> out(S);
    std::complex<double> *outp = out.mutable_data();
    const double *obp = obs.data(), *tobp = t_obs.data(), *sbp = src.data(), *tsbp = t_src.data();
    const int64_t *ownp = owner.data();
    py::gil_scoped_release release;
    // the kernel is the core's (momwire#1718 S5b), so the C++ route's
    // graded correction calls the same text
    somm_core::remainder_field_proj_owned(G, ground_z, k, obp, tobp, sbp, tsbp, ownp, S, outp, cancel_flag);
    return out;
}

// The graded rule's inner integral, fused (momwire#1201). Observer node m
// owns the source panels [ptr[m], ptr[m+1]) of the segment a0[m] + t*dvec[m],
// t in [0, 1], of physical length len[m]; each panel [p0, p1] gets the
// Gauss rule (xg, wg) on [-1, 1]. Returns
//   v[m, P] = sum over m's nodes of  w * len * F(obs[m], src(t)) * (len*t)^P
// for P < d1 — the source-side moments `_remainder_graded._inner_moments`
// takes in numpy, without materialising a single node array. One OpenMP
// iteration per observer node, its panels summed in order, so the result
// does not depend on the thread count.
static py::array_t<std::complex<double>> remainder_graded_inner(
    py::array_t<double, py::array::c_style | py::array::forcecast> obs,
    py::array_t<double, py::array::c_style | py::array::forcecast> t_obs,
    py::array_t<double, py::array::c_style | py::array::forcecast> a0,
    py::array_t<double, py::array::c_style | py::array::forcecast> dvec,
    py::array_t<double, py::array::c_style | py::array::forcecast> t_src,
    py::array_t<double, py::array::c_style | py::array::forcecast> len,
    py::array_t<int64_t, py::array::c_style | py::array::forcecast> ptr,
    py::array_t<double, py::array::c_style | py::array::forcecast> p0,
    py::array_t<double, py::array::c_style | py::array::forcecast> p1,
    py::array_t<double, py::array::c_style | py::array::forcecast> xg,
    py::array_t<double, py::array::c_style | py::array::forcecast> wg,
    int d1, double ground_z, double k, double r1_max, double r_break,
    double th_split, double r_near,
    py::array_t<double, py::array::c_style | py::array::forcecast> reg_r0,
    py::array_t<double, py::array::c_style | py::array::forcecast> reg_dr,
    py::array_t<double, py::array::c_style | py::array::forcecast> reg_th0,
    py::array_t<double, py::array::c_style | py::array::forcecast> reg_dth,
    std::vector<py::array_t<std::complex<double>,
                            py::array::c_style | py::array::forcecast>> reg_vals,
    py::array_t<std::complex<double>, py::array::c_style | py::array::forcecast>
        far,
    uintptr_t cancel_flag = 0) {
    using somm_proj::cd;
    auto ob = obs.unchecked<2>();
    auto tob = t_obs.unchecked<2>();
    auto A = a0.unchecked<2>();
    auto D = dvec.unchecked<2>();
    auto ts = t_src.unchecked<2>();
    auto Ln = len.unchecked<1>();
    auto pt = ptr.unchecked<1>();
    auto P0 = p0.unchecked<1>();
    auto P1 = p1.unchecked<1>();
    auto X = xg.unchecked<1>();
    auto Wg = wg.unchecked<1>();
    const py::ssize_t M = ob.shape(0);
    const py::ssize_t NP = P0.shape(0);
    const py::ssize_t NG = X.shape(0);
    if (ob.shape(1) != 3 || tob.shape(1) != 3 || A.shape(1) != 3 ||
        D.shape(1) != 3 || ts.shape(1) != 3)
        throw std::runtime_error("point/tangent arrays must have shape (M, 3)");
    if (tob.shape(0) != M || A.shape(0) != M || D.shape(0) != M ||
        ts.shape(0) != M || Ln.shape(0) != M || pt.shape(0) != M + 1)
        throw std::runtime_error("per-observer arrays must have length M (ptr M+1)");
    if (P1.shape(0) != NP || Wg.shape(0) != NG || d1 < 1)
        throw std::runtime_error("panel / rule arrays inconsistent");
    if (pt(0) != 0 || pt(M) != NP)
        throw std::runtime_error("ptr must run from 0 to the panel count");
    for (py::ssize_t m = 0; m < M; ++m)
        if (pt(m + 1) < pt(m))
            throw std::runtime_error("ptr must be non-decreasing");

    somm_proj::GridView G = somm_proj::build_grid_view(
        r1_max, r_break, th_split, r_near, reg_r0.unchecked<1>(),
        reg_dr.unchecked<1>(), reg_th0.unchecked<1>(), reg_dth.unchecked<1>(),
        reg_vals);
    somm_proj::set_far(G, far);

    py::array_t<std::complex<double>> out({M, (py::ssize_t)d1});
    cd *outp = out.mutable_data();
    const double *obp = obs.data(), *tobp = t_obs.data(), *ap = a0.data(), *dp = dvec.data(),
                 *tsp = t_src.data(), *lp = len.data(), *p0p = p0.data(), *p1p = p1.data(),
                 *xp = xg.data(), *wp = wg.data();
    const int64_t *ptp = ptr.data();
    py::gil_scoped_release release;
    somm_core::remainder_graded_inner(G, ground_z, k, obp, tobp, ap, dp, tsp, lp, ptp, M, p0p, p1p, xp, wp, NG,
                                      d1, outp, cancel_flag);
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
          py::arg("far"), py::arg("cancel_flag") = 0);
    m.def("remainder_graded_inner", &somm_pairs::remainder_graded_inner,
          "momwire#1201's graded source-side moments: per observer node m, "
          "sum over its source panels [ptr[m], ptr[m+1]) of the Gauss rule "
          "(xg, wg) of w*len*F(obs[m], a0[m]+t*dvec[m])*(len*t)^P, P < d1. "
          "OpenMP over observer nodes; returns (M, d1) complex. Grid as in "
          "remainder_field_proj_batch.",
          py::arg("obs"), py::arg("t_obs"), py::arg("a0"), py::arg("dvec"),
          py::arg("t_src"), py::arg("len"), py::arg("ptr"), py::arg("p0"),
          py::arg("p1"), py::arg("xg"), py::arg("wg"), py::arg("d1"),
          py::arg("ground_z"), py::arg("k"), py::arg("r1_max"),
          py::arg("r_break"), py::arg("th_split"), py::arg("r_near"),
          py::arg("reg_r0"), py::arg("reg_dr"), py::arg("reg_th0"),
          py::arg("reg_dth"), py::arg("reg_vals"), py::arg("far"),
          py::arg("cancel_flag") = 0);
}
