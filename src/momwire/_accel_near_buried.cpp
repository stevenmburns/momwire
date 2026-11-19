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
#include "_somm_transmitted_core.h"

#include <complex>
#include <memory>
#include <stdexcept>
#include <string>
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

// momwire#1718 S12: the executables' transmitted table and projection
// (`_somm_transmitted_core.cpp`), bound for the unit gates against
// `_sommerfeld_transmitted`. Python's own route does not call these.
py::dict transmitted_grid_core(cd eps_t, double k2, double r_max, double zp_min, double zp_max, double omega,
                               double mu, py::object r_min) {
    const bool has = !r_min.is_none();
    const double rm = has ? r_min.cast<double>() : 0.0;
    somm_trans::Grid g;
    {
        py::gil_scoped_release release;
        g = somm_trans::make_grid(eps_t, k2, r_max, zp_min, zp_max, omega, mu, has, rm, 0);
    }
    py::array_t<cd> vals({static_cast<py::ssize_t>(g.n_zp), py::ssize_t(5), static_cast<py::ssize_t>(g.n_r),
                          static_cast<py::ssize_t>(g.n_th)});
    std::copy(g.vals.begin(), g.vals.end(), vals.mutable_data());
    py::array_t<double> zp_nodes(static_cast<py::ssize_t>(g.zp_nodes.size()));
    std::copy(g.zp_nodes.begin(), g.zp_nodes.end(), zp_nodes.mutable_data());
    py::dict d;
    d["vals"] = vals;
    d["zp_nodes"] = zp_nodes;
    d["k_m"] = g.k_m;
    d["lam_p"] = g.lam_p;
    d["lam_m"] = g.lam_m;
    d["r_min"] = g.r_min;
    d["r_cap"] = g.r_cap;
    d["r_max"] = g.r_max;
    d["zp_min"] = g.zp_min;
    d["zp_max"] = g.zp_max;
    d["lnr0"] = g.lnr0;
    d["dlnr"] = g.dlnr;
    d["n_r"] = g.n_r;
    d["th_min"] = g.th_min;
    d["th0"] = g.th0;
    d["dth"] = g.dth;
    d["n_th"] = g.n_th;
    d["lnz0"] = g.lnz0;
    d["dlnz"] = g.dlnz;
    return d;
}

// `serve_plan`'s cross section and the cross-block order, as the
// executables compute them: (r_cross_max, r_cross_min, zp_min, zp_max), or
// the refusal's name (a str).
py::object transmitted_cross_plan_core(py::array_t<double, py::array::c_style | py::array::forcecast> obs_a,
                                       py::array_t<double, py::array::c_style | py::array::forcecast> obs_b,
                                       double ground_z, double k_p, cd k_m) {
    try {
        const somm_trans::CrossPlan p =
            somm_trans::cross_extents(obs_a.data(), static_cast<size_t>(obs_a.shape(0)), obs_b.data(),
                                      static_cast<size_t>(obs_b.shape(0)), ground_z, k_p, k_m);
        return py::make_tuple(p.r_cross_max, p.r_cross_min, p.zp_min, p.zp_max);
    } catch (const somm_trans::Refused &e) {
        return py::str(e.what());
    }
}

int64_t transmitted_near_q_factor_core(py::array_t<double, py::array::c_style | py::array::forcecast> seg_l,
                                       py::array_t<double, py::array::c_style | py::array::forcecast> seg_r,
                                       std::vector<int64_t> a_idx, std::vector<int64_t> b_idx) {
    return somm_trans::near_q_factor(seg_l.data(), seg_r.data(), a_idx, b_idx);
}

// The executables' projection over the grid `transmitted_grid_core` fills
// for the same arguments (filled again here): the (A, B) role-order table
// and the eval refusal's name ('' when served).
py::tuple transmitted_proj_core(cd eps_t, double k2, double r_max, double zp_min, double zp_max, double omega,
                                double mu, double r_min,
                                py::array_t<double, py::array::c_style | py::array::forcecast> above,
                                py::array_t<double, py::array::c_style | py::array::forcecast> t_above,
                                py::array_t<double, py::array::c_style | py::array::forcecast> below,
                                py::array_t<double, py::array::c_style | py::array::forcecast> t_below,
                                double ground_z, double k_p, cd k_m, bool transposed) {
    const size_t A = static_cast<size_t>(above.shape(0)), B = static_cast<size_t>(below.shape(0));
    py::array_t<cd> out({static_cast<py::ssize_t>(A), static_cast<py::ssize_t>(B)});
    std::string why;
    {
        py::gil_scoped_release release;
        const somm_trans::Grid &g = somm_trans::get_grid(eps_t, k2, r_max, zp_min, zp_max, omega, mu, r_min, 0);
        somm_trans::Extremes ext;
        somm_trans::project(g, above.data(), t_above.data(), A, below.data(), t_below.data(), B, ground_z, k_p,
                            k_m, transposed, out.mutable_data(), ext);
        if (A && B) why = somm_trans::eval_refusal(g, ext);
    }
    return py::make_tuple(out, why);
}

// momwire#1847: the C++ route's `_transmitted_anchors.get_grid`, every node
// filled, for the unit gates against Python's. None where Python declines
// (with `why`, its name, under the key "declined"); with `parts`, also the
// eight anchors' tables it summed, in stencil order, and the blend on the
// family's lattice before it is resampled onto the exact grid's nodes.
py::dict grid_state(const somm_trans::Grid &x) {
    py::dict d;
    py::array_t<cd> v({static_cast<py::ssize_t>(x.n_zp), py::ssize_t(5), static_cast<py::ssize_t>(x.n_r),
                       static_cast<py::ssize_t>(x.n_th)});
    std::copy(x.vals.begin(), x.vals.end(), v.mutable_data());
    d["vals"] = v;
    py::array_t<double> zp_nodes(static_cast<py::ssize_t>(x.zp_nodes.size()));
    std::copy(x.zp_nodes.begin(), x.zp_nodes.end(), zp_nodes.mutable_data());
    d["zp_nodes"] = zp_nodes;
    d["eps_t"] = x.eps_t;
    d["k2"] = x.k2;
    d["omega"] = x.omega;
    d["k_m"] = x.k_m;
    d["lam_p"] = x.lam_p;
    d["lam_m"] = x.lam_m;
    d["r_min"] = x.r_min;
    d["r_cap"] = x.r_cap;
    d["r_max"] = x.r_max;
    d["zp_min"] = x.zp_min;
    d["zp_max"] = x.zp_max;
    d["lnr0"] = x.lnr0;
    d["dlnr"] = x.dlnr;
    d["n_r"] = x.n_r;
    d["th_min"] = x.th_min;
    d["th0"] = x.th0;
    d["dth"] = x.dth;
    d["n_th"] = x.n_th;
    d["lnz0"] = x.lnz0;
    d["dlnz"] = x.dlnz;
    return d;
}

py::object transmitted_anchors_grid_core(cd eps_t, double k2, double r_max, double zp_min, double zp_max,
                                         double omega, double mu, py::object r_min, bool parts) {
    const bool has = !r_min.is_none();
    const double rm = has ? r_min.cast<double>() : 0.0;
    somm_trans::AnchorParts held;
    std::string why;
    std::shared_ptr<const somm_trans::Grid> g;
    {
        py::gil_scoped_release release;
        g = somm_trans::anchored_grid(eps_t, k2, r_max, zp_min, zp_max, omega, mu, has, rm, 0,
                                      parts ? &held : nullptr, &why);
    }
    if (g == nullptr) {
        py::dict d;
        d["declined"] = why;
        return d;
    }
    py::dict d = grid_state(*g);
    if (parts) {
        py::list a;
        for (const auto &x : held.anchors) {
            py::dict e;
            e["vals"] = grid_state(*x)["vals"];
            e["eps_t"] = x->eps_t;
            e["k2"] = x->k2;
            e["omega"] = x->omega;
            a.append(e);
        }
        d["anchors"] = a;
        d["blend"] = grid_state(*held.blend);
        d["r_first"] = held.r_first;
    }
    return d;
}

// momwire#1847: `_on_exact_nodes`' resampling alone, from a lattice the test
// spells (`vals` (n_zp, 5, n_r, n_th) on lnr0 + i dlnr, th0 + j dth and the
// ladder `zp_nodes`) onto the served axes (`to_*`), for the gate against
// `_resample_matrix` and its einsum on the same values. Returns the served
// values and which in-plane axis the contraction took first.
py::tuple transmitted_anchors_resample(py::array_t<cd, py::array::c_style | py::array::forcecast> vals,
                                       double lnr0, double th0, double dth, std::vector<double> zp_nodes,
                                       double lnz0, double dlnz, double to_lnr0, int64_t to_n_r, double to_th0,
                                       double to_dth, int64_t to_n_th, std::vector<double> to_zp_nodes) {
    if (vals.ndim() != 4 || vals.shape(1) != 5 || vals.shape(0) != static_cast<py::ssize_t>(zp_nodes.size()))
        throw std::invalid_argument("vals must be (len(zp_nodes), 5, n_r, n_th)");
    somm_trans::Grid lat;
    lat.n_zp = vals.shape(0);
    lat.n_r = vals.shape(2);
    lat.n_th = vals.shape(3);
    lat.lnr0 = lnr0;
    lat.dlnr = 0.10;  // `_R_STEP_LOG`, which the resampling's R targets step by too
    lat.th0 = lat.th_min = th0;
    lat.dth = dth;
    lat.zp_nodes = std::move(zp_nodes);
    lat.lnz0 = lnz0;
    lat.dlnz = dlnz;
    lat.vals.assign(vals.data(), vals.data() + vals.size());
    lat.have.assign(static_cast<size_t>(lat.n_zp * lat.n_r * lat.n_th), 1);
    bool r_first = true;
    somm_trans::Grid s;
    {
        py::gil_scoped_release release;
        s = somm_trans::resample_onto(lat, to_lnr0, to_n_r, to_th0, to_dth, to_n_th, to_zp_nodes, &r_first);
    }
    return py::make_tuple(grid_state(s)["vals"], r_first);
}

py::dict transmitted_anchors_stats(bool reset, int64_t anchors_max) {
    if (reset) somm_trans::anchor_reset();
    if (anchors_max > 0) somm_trans::anchor_capacity(static_cast<size_t>(anchors_max));
    const somm_trans::AnchorStats s = somm_trans::anchor_stats();
    py::dict d;
    d["anchor_fills"] = s.anchor_fills;
    d["blends"] = s.blends;
    d["declined"] = s.declined;
    d["nodes_filled"] = s.nodes_filled;
    return d;
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
    m.def("transmitted_grid_core", &transmitted_grid_core, py::arg("eps_t"), py::arg("k2"), py::arg("r_max"),
          py::arg("zp_min"), py::arg("zp_max"), py::arg("omega"), py::arg("mu"), py::arg("r_min") = py::none(),
          "momwire#1718 S12: `TransmittedGrid(...)` as the executables fill it (a dict of its state).");
    m.def("transmitted_cross_plan_core", &transmitted_cross_plan_core, py::arg("obs_a"), py::arg("obs_b"),
          py::arg("ground_z"), py::arg("k_p"), py::arg("k_m"),
          "momwire#1718 S12: `serve_plan`'s cross-medium extents as the executables compute them.");
    m.def("transmitted_near_q_factor_core", &transmitted_near_q_factor_core, py::arg("seg_l"), py::arg("seg_r"),
          py::arg("a_idx"), py::arg("b_idx"),
          "momwire#1718 S12: `near_q_factor(*cross_pair_separation(...))` as the executables compute it.");
    m.def("transmitted_proj_core", &transmitted_proj_core, py::arg("eps_t"), py::arg("k2"), py::arg("r_max"),
          py::arg("zp_min"), py::arg("zp_max"), py::arg("omega"), py::arg("mu"), py::arg("r_min"),
          py::arg("above"), py::arg("t_above"), py::arg("below"), py::arg("t_below"), py::arg("ground_z"),
          py::arg("k_p"), py::arg("k_m"), py::arg("transposed"),
          "momwire#1718 S12: the executables' get_grid + projection (role-order table, eval refusal name).");
    m.def("transmitted_anchors_grid_core", &transmitted_anchors_grid_core, py::arg("eps_t"), py::arg("k2"),
          py::arg("r_max"), py::arg("zp_min"), py::arg("zp_max"), py::arg("omega"), py::arg("mu"),
          py::arg("r_min") = py::none(), py::arg("parts") = false,
          "momwire#1847: `_transmitted_anchors.get_grid` as the C++ route blends it (a dict of its "
          "state; {'declined': name} where Python returns None; with parts, the anchors summed and the "
          "blend before its resampling).");
    m.def("transmitted_anchors_resample", &transmitted_anchors_resample, py::arg("vals"), py::arg("lnr0"),
          py::arg("th0"), py::arg("dth"), py::arg("zp_nodes"), py::arg("lnz0"), py::arg("dlnz"),
          py::arg("to_lnr0"), py::arg("to_n_r"), py::arg("to_th0"), py::arg("to_dth"), py::arg("to_n_th"),
          py::arg("to_zp_nodes"),
          "momwire#1847: `_on_exact_nodes`' resampling alone, from the given lattice table onto the given "
          "served axes; (served vals, whether the R axis was contracted first).");
    m.def("transmitted_anchors_stats", &transmitted_anchors_stats, py::arg("reset") = false,
          py::arg("anchors_max") = 0,
          "momwire#1847: the C++ anchored route's counts (`_transmitted_anchors.STATS`); reset forgets "
          "every anchor and blend first (and restores the anchor store's capacity, 96); anchors_max > 0 "
          "then sets that capacity (a test's, to saturate the FIFO).");
}
