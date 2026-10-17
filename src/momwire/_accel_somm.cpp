#include "_accel_common.h"
#include "_branch_cut_inline.h"
#include "_cdiv_inline.h"
#include "_fma_inline.h"
#include "_somm_six_lanes_inline.h"

#include <cstring>
#ifdef _OPENMP
#include <omp.h>
#endif

// somm section of the former _accelerators.cpp monolith (momwire#687).
// Code below is byte-identical to the monolith's lines 5402-6272, with TWO
// exceptions: somm_proj (monolith 5846-5982) was hoisted verbatim into
// _accel_somm_proj_inline.h — shared with the mw568 TU — and is pulled back
// in by the mid-file #include below; and the local `gam` branch-cut helper
// became the shared `mw_branch::gamma_cut` (#714), renaming its two call
// sites in integrand_six.

#include "_somm_kernel_core.h"

// Batched entry point: the six NEC lambda-integrals at each (rho[i], h[i]),
// OpenMP across nodes (each node's adaptive quadrature is independent).
// Returns (n, 6) complex in _six_integrals order (Vrr, Vzz, Vrz, Vr1, V, U).
static py::array_t<std::complex<double>> somm_six_integrals_batch(
    std::complex<double> eps_t, double k2,
    py::array_t<double, py::array::c_style | py::array::forcecast> rho,
    py::array_t<double, py::array::c_style | py::array::forcecast> h,
    double rtol, int form, uintptr_t cancel_flag = 0, bool lanes = true) {
    auto rb = rho.unchecked<1>();
    auto hb = h.unchecked<1>();
    const py::ssize_t n = rb.shape(0);
    if (hb.shape(0) != n)
        throw std::invalid_argument("rho and h must have the same length");
    if (form < 0 || form > 2)
        throw std::invalid_argument("form must be 0 (auto), 1 (J) or 2 (H)");
    for (py::ssize_t i = 0; i < n; ++i) {
        if (rb(i) < 0.0 || hb(i) < 0.0 || (rb(i) == 0.0 && hb(i) == 0.0))
            throw std::invalid_argument(
                "need rho, h >= 0 and R1 > 0 at every node");
    }
    py::array_t<std::complex<double>> out({n, py::ssize_t(6)});
    auto ob = out.mutable_unchecked<2>();
    const std::complex<double> et(eps_t);
    MW_CANCEL_SETUP(cancel_flag);

    #pragma omp parallel for schedule(dynamic)
    for (py::ssize_t i = 0; i < n; ++i) {
        MW_CANCEL_POLL();
        std::complex<double> res[6];
        somm_core::six_integrals(et, k2, rb(i), hb(i), rtol, form, res, lanes);
        for (int j = 0; j < 6; ++j) ob(i, j) = res[j];
    }
    MW_THROW_IF_ABORTED();
    return out;
}

// ---------------------------------------------------------------------------
// Sommerfeld remainder assembly (sommerfeld-perf-plan Phase 4b).
//
// Fused C++ port of _sommerfeld.remainder_field_proj (which internally calls
// SommerfeldGrid.eval). Per (observer m, source n) pair: interpolate the four
// smooth-remainder surfaces from the tabulated grid with the same 4x4 Lagrange
// (bivariate cubic) stencil, combine per the theory-manual eqs 143-147 azimuth
// factors, and project onto the observer tangent -- with NO materialized
// (4,n,4,4) intermediate (the numpy bottleneck). One OpenMP loop over observer
// rows. Bit-for-bit the same arithmetic as the Python fallback; cross-checked
// in tests/test_sommerfeld_accel.py.
//
// Clean-room: ported from momwire's own _sommerfeld.py. No GPL Sommerfeld
// source (nec2c, nec2++/PyNEC, somnec) was consulted.
#include "_accel_somm_proj_py.h"
// The projection's loop and lanes are in `_somm_proj_core.cpp` (momwire#1479
// S3); `remainder_field_proj_batch` below binds them.
#include "_somm_proj_core.h"
// The near-field remainder's kernel (`reflected_field_batch` below binds it).
#include "_somm_field_core.h"


// obs/t_obs (M,3), src/t_src (S,3); returns (M,S) complex. The grid is passed
// flattened: the regions' (r0, dr, th0, dth) as per-region arrays, plus a
// list of four-or-six (4, n_r, n_th) complex value tables. r_break / th_split select
// the region exactly as SommerfeldGrid.eval.
static py::array_t<std::complex<double>> remainder_field_proj_batch(
    py::array_t<double, py::array::c_style | py::array::forcecast> obs,
    py::array_t<double, py::array::c_style | py::array::forcecast> t_obs,
    py::array_t<double, py::array::c_style | py::array::forcecast> src,
    py::array_t<double, py::array::c_style | py::array::forcecast> t_src,
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
    uintptr_t cancel_flag = 0, bool lanes = true) {
    using somm_proj::cd;
    auto ob = obs.unchecked<2>();
    auto tob = t_obs.unchecked<2>();
    auto sb = src.unchecked<2>();
    auto tsb = t_src.unchecked<2>();
    if (ob.shape(1) != 3 || tob.shape(1) != 3 || sb.shape(1) != 3 ||
        tsb.shape(1) != 3)
        throw std::runtime_error("obs/src/tangent arrays must have shape (*, 3)");
    if (ob.shape(0) != tob.shape(0) || sb.shape(0) != tsb.shape(0))
        throw std::runtime_error("points and tangents must have matching length");

    const py::ssize_t M = ob.shape(0);
    const py::ssize_t S = sb.shape(0);
    somm_proj::GridView G = somm_proj::build_grid_view(
        r1_max, r_break, th_split, r_near, reg_r0.unchecked<1>(),
        reg_dr.unchecked<1>(), reg_th0.unchecked<1>(), reg_dth.unchecked<1>(),
        reg_vals);
    somm_proj::set_far(G, far);

    py::array_t<std::complex<double>> out({M, S});

    py::gil_scoped_release release;

    somm_core::field_proj_batch(G, reg_vals.size(), M, S, obs.data(),
                                t_obs.data(), src.data(), t_src.data(),
                                ground_z, k, out.mutable_data(), cancel_flag,
                                lanes);
    return out;
}

// The near-field remainder at observation points (`_somm_field_core.cpp`):
// E, or H by the curl when `step` > 0. points (N, 3), src_mid (S, 3),
// src_moment (S, 3) complex; returns (N, 3) complex. The grid is passed as
// `remainder_field_proj_batch` takes it.
static py::array_t<std::complex<double>> reflected_field_batch_impl(
    py::array_t<double, py::array::c_style | py::array::forcecast> points,
    py::array_t<double, py::array::c_style | py::array::forcecast> src_mid,
    py::array_t<std::complex<double>, py::array::c_style | py::array::forcecast> src_moment,
    double ground_z, double k, bool magnetic, double omega, double mu, double step,
    double r1_max, double r_break, double th_split, double r_near,
    py::array_t<double, py::array::c_style | py::array::forcecast> reg_r0,
    py::array_t<double, py::array::c_style | py::array::forcecast> reg_dr,
    py::array_t<double, py::array::c_style | py::array::forcecast> reg_th0,
    py::array_t<double, py::array::c_style | py::array::forcecast> reg_dth,
    std::vector<py::array_t<std::complex<double>,
                            py::array::c_style | py::array::forcecast>> reg_vals,
    py::array_t<std::complex<double>, py::array::c_style | py::array::forcecast> far) {
    if (points.ndim() != 2 || points.shape(1) != 3 || src_mid.ndim() != 2 ||
        src_mid.shape(1) != 3 || src_moment.ndim() != 2 || src_moment.shape(1) != 3 ||
        src_moment.shape(0) != src_mid.shape(0))
        throw std::runtime_error("points, src_mid and src_moment must be (*, 3), the sources alike");
    somm_proj::GridView G = somm_proj::build_grid_view(
        r1_max, r_break, th_split, r_near, reg_r0.unchecked<1>(), reg_dr.unchecked<1>(),
        reg_th0.unchecked<1>(), reg_dth.unchecked<1>(), reg_vals);
    somm_proj::set_far(G, far);
    const py::ssize_t n = points.shape(0), s = src_mid.shape(0);
    py::array_t<std::complex<double>> out({n, py::ssize_t(3)});
    {
        py::gil_scoped_release release;
        if (magnetic)
            somm_core::reflected_h_field(G, n, points.data(), s, src_mid.data(), src_moment.data(),
                                         ground_z, k, omega, mu, step, out.mutable_data());
        else
            somm_core::reflected_field(G, n, points.data(), s, src_mid.data(), src_moment.data(),
                                       ground_z, k, out.mutable_data());
    }
    return out;
}

// The lazy grids' census (momwire#1512): a mask per region (n_r, n_th) of
// the nodes the reads of `remainder_field_proj_batch` over obs x src take,
// or those of `reflected_field_batch` at `points` (E; with magnetic, H's six
// offset points). Both are the cores' own (`somm_core::field_proj_nodes`,
// `somm_core::reflected_field_nodes`), the one copy the executable runs.
static std::vector<py::array_t<uint8_t>> census_masks(
    const somm_proj::GridView &G, size_t n_reg, somm_proj::NodeMarks &marks) {
    std::vector<py::array_t<uint8_t>> masks;
    marks.n_reg = n_reg;
    for (size_t r = 0; r < n_reg; r++) {
        py::array_t<uint8_t> a({G.nR[r], G.nTh[r]});
        std::fill(a.mutable_data(), a.mutable_data() + G.nR[r] * G.nTh[r], uint8_t(0));
        marks.m[r] = a.mutable_data();
        marks.nR[r] = static_cast<std::ptrdiff_t>(G.nR[r]);
        marks.nTh[r] = static_cast<std::ptrdiff_t>(G.nTh[r]);
        masks.push_back(a);
    }
    return masks;
}

static std::vector<py::array_t<uint8_t>> remainder_field_proj_nodes(
    py::array_t<double, py::array::c_style | py::array::forcecast> obs,
    py::array_t<double, py::array::c_style | py::array::forcecast> src, double ground_z,
    double r1_max, double r_break, double th_split, double r_near,
    py::array_t<double, py::array::c_style | py::array::forcecast> reg_r0,
    py::array_t<double, py::array::c_style | py::array::forcecast> reg_dr,
    py::array_t<double, py::array::c_style | py::array::forcecast> reg_th0,
    py::array_t<double, py::array::c_style | py::array::forcecast> reg_dth,
    std::vector<py::array_t<std::complex<double>,
                            py::array::c_style | py::array::forcecast>> reg_vals) {
    if (obs.ndim() != 2 || obs.shape(1) != 3 || src.ndim() != 2 || src.shape(1) != 3)
        throw std::runtime_error("obs/src arrays must have shape (*, 3)");
    const somm_proj::GridView G = somm_proj::build_grid_view(
        r1_max, r_break, th_split, r_near, reg_r0.unchecked<1>(), reg_dr.unchecked<1>(),
        reg_th0.unchecked<1>(), reg_dth.unchecked<1>(), reg_vals);
    somm_proj::NodeMarks marks;
    std::vector<py::array_t<uint8_t>> masks = census_masks(G, reg_vals.size(), marks);
    py::gil_scoped_release release;
    somm_core::field_proj_nodes(G, obs.shape(0), src.shape(0), obs.data(), src.data(), ground_z,
                                marks);
    return masks;
}

static std::vector<py::array_t<uint8_t>> reflected_field_nodes_impl(
    py::array_t<double, py::array::c_style | py::array::forcecast> points,
    py::array_t<double, py::array::c_style | py::array::forcecast> src_mid, double ground_z,
    bool magnetic, double step, double r1_max, double r_break, double th_split, double r_near,
    py::array_t<double, py::array::c_style | py::array::forcecast> reg_r0,
    py::array_t<double, py::array::c_style | py::array::forcecast> reg_dr,
    py::array_t<double, py::array::c_style | py::array::forcecast> reg_th0,
    py::array_t<double, py::array::c_style | py::array::forcecast> reg_dth,
    std::vector<py::array_t<std::complex<double>,
                            py::array::c_style | py::array::forcecast>> reg_vals) {
    if (points.ndim() != 2 || points.shape(1) != 3 || src_mid.ndim() != 2 ||
        src_mid.shape(1) != 3)
        throw std::runtime_error("points and src_mid must be (*, 3)");
    const somm_proj::GridView G = somm_proj::build_grid_view(
        r1_max, r_break, th_split, r_near, reg_r0.unchecked<1>(), reg_dr.unchecked<1>(),
        reg_th0.unchecked<1>(), reg_dth.unchecked<1>(), reg_vals);
    somm_proj::NodeMarks marks;
    std::vector<py::array_t<uint8_t>> masks = census_masks(G, reg_vals.size(), marks);
    const py::ssize_t n = points.shape(0), s = src_mid.shape(0);
    py::gil_scoped_release release;
    if (magnetic)
        somm_core::reflected_h_field_nodes(G, n, points.data(), s, src_mid.data(), ground_z, step,
                                           marks);
    else
        somm_core::reflected_field_nodes(G, n, points.data(), s, src_mid.data(), ground_z, marks);
    return masks;
}

// The fully-fused B-spline Galerkin Sommerfeld remainder's kernel is in
// `_somm_remainder_core.cpp` (momwire#1517 step 3), with its block comments;
// `sommerfeld_remainder_bspline_Q` below binds it.
#include "_somm_remainder_core.h"

// Calls served by the symmetric route, so a gate can show the route ran
// rather than infer it from agreement (an untaken route agrees trivially).
static std::atomic<long long> g_remainder_symmetric_calls{0};

// True when the obs and src halves of the call are the same data -- the
// dense block's call shape -- so the symmetric route serves it.
template <typename T>
static bool same_array(const py::array_t<T, py::array::c_style | py::array::forcecast> &a,
                       const py::array_t<T, py::array::c_style | py::array::forcecast> &b) {
    if (a.ndim() != b.ndim()) return false;
    for (py::ssize_t d = 0; d < a.ndim(); ++d)
        if (a.shape(d) != b.shape(d)) return false;
    if (a.data() == b.data()) return true;
    return std::memcmp(a.data(), b.data(), sizeof(T) * (size_t)a.size()) == 0;
}

static py::array_t<std::complex<double>> sommerfeld_remainder_bspline_Q(
    py::array_t<double, py::array::c_style | py::array::forcecast> obs_nodes,
    py::array_t<double, py::array::c_style | py::array::forcecast> obs_tang,
    py::array_t<double, py::array::c_style | py::array::forcecast> W_obs,
    py::array_t<double, py::array::c_style | py::array::forcecast> src_nodes,
    py::array_t<double, py::array::c_style | py::array::forcecast> src_tang,
    py::array_t<double, py::array::c_style | py::array::forcecast> W_src,
    py::array_t<int64_t, py::array::c_style | py::array::forcecast> loc_I,
    py::array_t<double, py::array::c_style | py::array::forcecast> pI,
    py::array_t<int64_t, py::array::c_style | py::array::forcecast> loc_J,
    py::array_t<double, py::array::c_style | py::array::forcecast> pJ,
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
    uintptr_t cancel_flag = 0, size_t max_jf_bytes = 0, bool reference = false) {
    using somm_proj::cd;
    auto ndI = obs_nodes.unchecked<3>();
    auto tgI = obs_tang.unchecked<2>();
    auto WvI = W_obs.unchecked<3>();
    auto ndJ = src_nodes.unchecked<3>();
    auto tgJ = src_tang.unchecked<2>();
    auto WvJ = W_src.unchecked<3>();
    auto lI = loc_I.unchecked<2>();
    auto lJ = loc_J.unchecked<2>();
    auto plI = pI.unchecked<3>();
    auto plJ = pJ.unchecked<3>();

    const py::ssize_t nsI = ndI.shape(0);
    const py::ssize_t nsJ = ndJ.shape(0);
    const py::ssize_t q = ndI.shape(1);
    const int d1 = (int)WvI.shape(0);  // degree + 1
    const py::ssize_t nI = lI.shape(0);
    const py::ssize_t nJ = lJ.shape(0);
    if (ndI.shape(2) != 3 || ndJ.shape(2) != 3 || ndJ.shape(1) != q)
        throw std::runtime_error("obs/src nodes must be (n_seg, q, 3), same q");
    if (WvI.shape(1) != nsI || WvI.shape(2) != q || WvJ.shape(1) != nsJ ||
        WvJ.shape(2) != q || (int)WvJ.shape(0) != d1)
        throw std::runtime_error("W_obs/W_src inconsistent with nodes/degree");
    if (lI.shape(1) != d1 || lJ.shape(1) != d1 || plI.shape(1) != d1 ||
        plI.shape(2) != d1 || plJ.shape(1) != d1 || plJ.shape(2) != d1)
        throw std::runtime_error("loc/polys inconsistent with degree");

    somm_proj::GridView G = somm_proj::build_grid_view(
        r1_max, r_break, th_split, r_near, reg_r0.unchecked<1>(),
        reg_dr.unchecked<1>(), reg_th0.unchecked<1>(), reg_dth.unchecked<1>(),
        reg_vals);
    somm_proj::set_far(G, far);

    // The dense block's call shape (obs == src throughout) takes the
    // symmetric route unless `reference` asks for this banded kernel, which
    // stays the route of every rectangular call and the oracle the symmetric
    // one is held to.
    const bool symmetric =
        !reference && same_array(obs_nodes, src_nodes) &&
        same_array(obs_tang, src_tang) && same_array(W_obs, W_src) &&
        same_array(loc_I, loc_J) && same_array(pI, pJ);

    py::array_t<std::complex<double>> Q({nI, nJ});
    if (symmetric) g_remainder_symmetric_calls.fetch_add(1, std::memory_order_relaxed);
    cd *Qp = Q.mutable_data();
    const double *on = obs_nodes.data(), *ot = obs_tang.data(), *ow = W_obs.data();
    const double *sn = src_nodes.data(), *st = src_tang.data(), *sw = W_src.data();
    const int64_t *li = loc_I.data(), *lj = loc_J.data();
    const double *pi = pI.data(), *pj = pJ.data();
    py::gil_scoped_release release;
    somm_core::remainder_bspline_Q(G, ground_z, k, on, ot, ow, nsI, sn, st, sw, nsJ, q, d1, li, pi,
                                   nI, lj, pj, nJ, symmetric, max_jf_bytes, Qp, cancel_flag);
    return Q;
}


void register_somm(py::module_ &m) {

    m.def("somm_six_integrals_batch", &somm_six_integrals_batch,
          "Batched Sommerfeld six-integral evaluation at (rho[i], h[i]) "
          "nodes; OpenMP across nodes. form: 0 auto, 1 Bessel, 2 Hankel. "
          "Returns (n, 6) complex in _six_integrals order.",
          py::arg("eps_t"), py::arg("k2"), py::arg("rho"), py::arg("h"),
          py::arg("rtol") = 1e-9, py::arg("form") = 0,
          py::arg("cancel_flag") = 0, py::arg("lanes") = true);
    // momwire#1441 step 3: the integrand runs four contour nodes at a time;
    // `lanes=False` is the scalar walk it is gated against, bit for bit.
    m.attr("somm_six_lanes_1441") = true;
    m.def("remainder_field_proj_batch", &remainder_field_proj_batch,
          "Fused Sommerfeld smooth-remainder assembly: interpolate the four "
          "grid surfaces (4x4 Lagrange) and project t_m.F(r_m,r_n).t_n per "
          "(observer, source) pair; OpenMP over observer rows. Returns (M, S) "
          "complex. The grid is passed flattened (per-region r0/dr/th0/dth "
          "arrays + a list of 3 near — or 5 with the #159 far zone — "
          "(4,n_r,n_th) value tables), then `far`: the momwire#1258 "
          "continuation past r1_max (`_sommerfeld.far_cpp_pack`; empty "
          "serves the edge value frozen).",
          py::arg("obs"), py::arg("t_obs"), py::arg("src"), py::arg("t_src"),
          py::arg("ground_z"), py::arg("k"), py::arg("r1_max"),
          py::arg("r_break"), py::arg("th_split"), py::arg("r_near"),
          py::arg("reg_r0"), py::arg("reg_dr"), py::arg("reg_th0"),
          py::arg("reg_dth"), py::arg("reg_vals"), py::arg("far"),
          py::arg("cancel_flag") = 0, py::arg("lanes") = true);
    m.def("reflected_field_batch", &reflected_field_batch_impl,
          "The Sommerfeld ground's near-field REMAINDER at observation points "
          "(`_somm_field_core.cpp`): E, or with magnetic=True H = curl(E) / "
          "(-j omega mu) by central differences at `step`. points (N, 3), "
          "src_mid (S, 3), src_moment (S, 3) complex; returns (N, 3) complex. "
          "The grid flattened as `remainder_field_proj_batch` takes it, with "
          "its far pack. GIL released, OpenMP over points.",
          py::arg("points"), py::arg("src_mid"), py::arg("src_moment"),
          py::arg("ground_z"), py::arg("k"), py::arg("magnetic"), py::arg("omega"),
          py::arg("mu"), py::arg("step"), py::arg("r1_max"), py::arg("r_break"),
          py::arg("th_split"), py::arg("r_near"), py::arg("reg_r0"), py::arg("reg_dr"),
          py::arg("reg_th0"), py::arg("reg_dth"), py::arg("reg_vals"), py::arg("far"));
    m.def("remainder_field_proj_nodes", &remainder_field_proj_nodes,
          "The lazy grids' census (momwire#1512): per region a (n_r, n_th) "
          "uint8 mask of the nodes remainder_field_proj_batch's reads over "
          "obs x src take (its own geometry and stencil address). The grid "
          "as remainder_field_proj_batch takes it, without `far`.",
          py::arg("obs"), py::arg("src"), py::arg("ground_z"), py::arg("r1_max"),
          py::arg("r_break"), py::arg("th_split"), py::arg("r_near"), py::arg("reg_r0"),
          py::arg("reg_dr"), py::arg("reg_th0"), py::arg("reg_dth"), py::arg("reg_vals"));
    m.def("reflected_field_nodes", &reflected_field_nodes_impl,
          "The lazy grids' census of reflected_field_batch (momwire#1512): the "
          "nodes its reads at `points` take (with magnetic, H's six offset "
          "points), a (n_r, n_th) uint8 mask per region.",
          py::arg("points"), py::arg("src_mid"), py::arg("ground_z"), py::arg("magnetic"),
          py::arg("step"), py::arg("r1_max"), py::arg("r_break"), py::arg("th_split"),
          py::arg("r_near"), py::arg("reg_r0"), py::arg("reg_dr"), py::arg("reg_th0"),
          py::arg("reg_dth"), py::arg("reg_vals"));
    m.def("somm_proj_lanes_built", []() { return somm_core::proj_lanes_built(); },
          "Whether this build has remainder_field_proj_batch's AVX2 lanes "
          "(perf item 9); without them `lanes=True` runs the per-pair loop.");
    m.def("somm_proj_lane_pairs",
          []() { return somm_core::proj_lane_pairs(); },
          "How many pairs remainder_field_proj_batch has sent through its "
          "AVX2 lanes in this process (perf item 9): the lanes' output equals "
          "the per-pair loop's, so only this says which ran.");
    m.def("sommerfeld_remainder_bspline_Q", &sommerfeld_remainder_bspline_Q,
          "Fully-fused b-spline Galerkin Sommerfeld remainder over an obs/src "
          "rectangle: interpolate + project + moment-quadrature + basis-assemble "
          "into the (nI, nJ) Q block directly (no Jf / einsum intermediates). "
          "Dense block = symmetric case (obs==src, loc==supp_seg); the ACA "
          "sampler passes thin segment subsets with local support maps. Grid as "
          "in remainder_field_proj_batch. The internal moment slab is banded "
          "over observer segments so its residency is bounded by "
          "`max_jf_bytes` (0 = the 64 MiB default), never the full "
          "(d+1)^2 * nsI * nsJ tensor (momwire#343); the banding is exact and "
          "order-preserving, not an approximation. A call whose obs and src "
          "halves are the same data (the dense block) takes the symmetric "
          "route, one grid interpolation per unordered node pair, bit-identical "
          "to the banded kernel; `reference=True` forces the banded kernel.",
          py::arg("obs_nodes"), py::arg("obs_tang"), py::arg("W_obs"),
          py::arg("src_nodes"), py::arg("src_tang"), py::arg("W_src"),
          py::arg("loc_I"), py::arg("pI"), py::arg("loc_J"), py::arg("pJ"),
          py::arg("ground_z"), py::arg("k"),
          py::arg("r1_max"), py::arg("r_break"), py::arg("th_split"),
          py::arg("r_near"), py::arg("reg_r0"), py::arg("reg_dr"), py::arg("reg_th0"),
          py::arg("reg_dth"), py::arg("reg_vals"), py::arg("far"),
          py::arg("cancel_flag") = 0, py::arg("max_jf_bytes") = 0,
          py::arg("reference") = false);
    m.def("remainder_q_symmetric_calls",
          []() { return g_remainder_symmetric_calls.load(); },
          "How many sommerfeld_remainder_bspline_Q calls the symmetric route "
          "has served in this process (a gate's evidence that it ran).");
}

