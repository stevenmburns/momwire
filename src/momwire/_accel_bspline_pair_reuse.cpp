// bs2's fill-pair reuse for the Python route (momwire#2063 stage 3): the
// binding of `_bspline_pair_reuse_core.h`. Its own TU: copies and integers
// only, no kernel.
#include "_accel_common.h"

#include "_bspline_pair_reuse_core.h"

#include <memory>
#include <stdexcept>

namespace {

namespace br = mw::bspline::reuse;
using cd = std::complex<double>;
using F64 = py::array_t<double, py::array::c_style | py::array::forcecast>;
using I64 = py::array_t<int64_t, py::array::c_style | py::array::forcecast>;
using I32 = py::array_t<int32_t, py::array::c_style | py::array::forcecast>;
using C128 = py::array_t<cd, py::array::c_style>;

struct PlanRef {
    std::weak_ptr<const br::Plan> plan;
};

struct PlanHandle {
    std::shared_ptr<const br::Plan> plan;
    bool built = false;
    int64_t n() const { return plan->fam.at(0).spec.rows; }
};

PlanHandle reuse_plan(F64 seg_l, F64 seg_r, F64 a, I32 edge, py::object group_i, py::object group_j,
                      py::object b_src, std::vector<double> thresholds, int degree) {
    const int64_t n = static_cast<int64_t>(a.size());
    if (seg_l.ndim() != 2 || seg_l.shape(0) != n || seg_l.shape(1) != 3 || seg_r.ndim() != 2 ||
        seg_r.shape(0) != n || seg_r.shape(1) != 3 || edge.size() != n)
        throw std::invalid_argument("bspline pair_reuse: (n, 3) end points, n radii and n edges");
    I64 gi, gj;
    F64 bs;
    br::Geometry g;
    g.n = n;
    g.seg_l = seg_l.data();
    g.seg_r = seg_r.data();
    g.a = a.data();
    g.edge = edge.data();
    if (!group_i.is_none()) {
        gi = group_i.cast<I64>();
        gj = group_j.cast<I64>();
        if (gi.size() != n || gj.size() != n) throw std::invalid_argument("bspline pair_reuse: n EK labels");
        g.group_i = gi.data();
        g.group_j = gj.data();
    }
    if (!b_src.is_none()) {
        bs = b_src.cast<F64>();
        if (bs.size() != n) throw std::invalid_argument("bspline pair_reuse: n source tube radii");
        g.b_src = bs.data();
    }
    g.thresholds = std::move(thresholds);
    g.degree = degree;
    PlanHandle h;
    {
        py::gil_scoped_release release;
        h.plan = br::plan_for(g, &h.built);
    }
    return h;
}

PlanHandle plan_of_classes(I32 cls) {
    if (cls.ndim() != 2 || cls.shape(0) != cls.shape(1))
        throw std::invalid_argument("bspline pair_reuse: an (n, n) class map");
    PlanHandle h;
    h.plan = br::plan_of_classes(cls.data(), static_cast<int64_t>(cls.shape(0)));
    h.built = true;
    return h;
}

py::list groups(const PlanHandle &h) {
    py::list out;
    for (const br::Group &G : h.plan->fam.at(0).groups) {
        auto arr = [](const std::vector<int32_t> &v) {
            py::array_t<int64_t> a(static_cast<py::ssize_t>(v.size()));
            int64_t *p = a.mutable_data();
            for (size_t i = 0; i < v.size(); i++) p[i] = v[i];
            return a;
        };
        out.append(py::make_tuple(arr(G.obs), arr(G.src), arr(G.cls), arr(G.at)));
    }
    return out;
}

void gather(const PlanHandle &h, C128 values, int64_t r0, int64_t r1, C128 out) {
    const int64_t n = h.n(), classes = h.plan->fam.at(0).classes;
    if (values.ndim() != 3 || values.shape(0) != classes || values.shape(1) != values.shape(2))
        throw std::invalid_argument("bspline pair_reuse: values are (classes, d + 1, d + 1)");
    const int d1 = static_cast<int>(values.shape(1));
    if (out.ndim() != 4 || out.shape(0) != d1 || out.shape(1) != d1 || out.shape(2) != r1 - r0 ||
        out.shape(3) != n)
        throw std::invalid_argument("bspline pair_reuse: out is (d + 1, d + 1, r1 - r0, n)");
    const cd *v = values.data();
    cd *o = out.mutable_data();
    py::gil_scoped_release release;
    br::gather(*h.plan, r0, r1, v, d1, o);
}

py::tuple remainder_blocks(F64 seg_l, F64 seg_r, F64 a, I64 edge_seg0, double ground_z) {
    const int64_t n = static_cast<int64_t>(a.size()), E = static_cast<int64_t>(edge_seg0.size()) - 1;
    if (seg_l.ndim() != 2 || seg_l.shape(0) != n || seg_l.shape(1) != 3 || seg_r.ndim() != 2 ||
        seg_r.shape(0) != n || seg_r.shape(1) != 3 || E < 1 || edge_seg0.data()[E] != n)
        throw std::invalid_argument("bspline pair_reuse: (n, 3) end points, n radii, E + 1 edge starts ending at n");
    br::RemainderGeometry g;
    g.seg_l = seg_l.data();
    g.seg_r = seg_r.data();
    g.a = a.data();
    g.edge_seg0 = edge_seg0.data();
    g.n_edges = E;
    g.ground_z = ground_z;
    std::vector<int64_t> ta, tb, sa, sb;
    {
        py::gil_scoped_release release;
        br::remainder_blocks(g, ta, tb, sa, sb);
    }
    auto arr = [](const std::vector<int64_t> &v) {
        py::array_t<int64_t> out(static_cast<py::ssize_t>(v.size()));
        std::copy(v.begin(), v.end(), out.mutable_data());
        return out;
    };
    return py::make_tuple(arr(ta), arr(tb), arr(sa), arr(sb));
}

py::object wire_map(const PlanRef &r, I32 wire_of_seg, int64_t n_wires) {
    const std::shared_ptr<const br::Plan> p = r.plan.lock();
    if (!p) return py::none();
    const int64_t n = p->fam.at(0).spec.rows;
    if (wire_of_seg.size() != n) throw std::invalid_argument("bspline pair_reuse: one wire per segment");
    if (n_wires < 1) throw std::invalid_argument("bspline pair_reuse: n_wires must be positive");
    const int32_t *w = wire_of_seg.data();
    for (int64_t s = 0; s < n; s++)
        if (w[s] < 0 || w[s] >= n_wires) throw std::invalid_argument("bspline pair_reuse: wire out of range");
    py::array_t<int64_t> out({static_cast<py::ssize_t>(n_wires), static_cast<py::ssize_t>(n_wires),
                              static_cast<py::ssize_t>(3)});
    int64_t *o = out.mutable_data();
    {
        py::gil_scoped_release release;
        br::wire_map(*p, w, n_wires, o);
    }
    return out;
}

py::dict stats(const PlanHandle &h) {
    const br::Plan &p = *h.plan;
    const mw::pair_reuse::FamilyPlan &f = p.fam.at(0);
    py::dict out;
    out["n_seg"] = f.spec.rows;
    out["pairs"] = f.spec.rows * f.spec.cols;
    out["classes"] = f.classes;
    out["near_pairs"] = br::near_pairs(p);
    out["same_edge_pairs"] = br::same_edge_pairs(p);
    out["rep_evals"] = f.rep_evals;
    out["groups"] = static_cast<int64_t>(f.groups.size());
    out["census_seconds"] = p.census_seconds;
    out["nbytes"] = static_cast<int64_t>(p.nbytes());
    out["budget_bytes"] = p.budget_bytes;
    out["census_peak_bytes"] = p.census_peak_bytes;
    out["declined"] = p.declined.empty() ? py::object(py::none()) : py::object(py::str(p.declined));
    return out;
}

}  // namespace

void register_bspline_pair_reuse(py::module_ &m) {
    py::class_<PlanRef>(m, "BSplineReusePlanRef", "A bs2 reuse plan held weakly (the fill report's).")
        .def_property_readonly("alive", [](const PlanRef &r) { return !r.plan.expired(); },
                               "Whether the process's cache (or a fill) still holds the plan.")
        .def("wire_map", &wire_map, py::arg("wire_of_seg"), py::arg("n_wires"),
             "The (W, W, 3) int64 map of observer wire x source wire x (computed, reused, near), "
             "or None when the plan is gone.");
    py::class_<PlanHandle>(m, "BSplineReusePlan",
                           "bs2's fill-pair reuse plan (momwire#2063): the classes of the (observer "
                           "segment, source segment) pairs of the moment fill, from the process's cache.")
        .def_readonly("built", &PlanHandle::built, "Whether this call's census built the plan.")
        .def_property_readonly("declined", [](const PlanHandle &h) -> py::object {
            if (h.plan->declined.empty()) return py::none();
            return py::str(h.plan->declined);
        }, "Why this plan does not serve the deck (it then has no classes), or None.")
        .def_property_readonly("classes", [](const PlanHandle &h) { return h.plan->fam.at(0).classes; })
        .def_property_readonly("plan_bytes", [](const PlanHandle &h) { return h.plan->budget_bytes; },
                               "The kept plan's bytes: its maps and groups, and one wavenumber's values.")
        .def_property_readonly("census_peak_bytes", [](const PlanHandle &h) { return h.plan->census_peak_bytes; })
        .def("stats", &stats)
        .def("groups", &groups,
             "The representatives' evaluation groups: (observer rows, sources, classes, at), "
             "class cls[q]'s value at entry at[q] = row * |sources| + column of the group's call.")
        .def("gather", &gather, py::arg("values"), py::arg("r0"), py::arg("r1"), py::arg("out"),
             "Observer rows r0..r1 of the moment tensor, (d + 1, d + 1, r1 - r0, n), from the "
             "class values (classes, d + 1, d + 1).")
        .def("ref", [](const PlanHandle &h) { return PlanRef{h.plan}; }, "The plan, held weakly.");
    m.def("bspline_reuse_plan", &reuse_plan, py::arg("seg_l"), py::arg("seg_r"), py::arg("a"),
          py::arg("edge"), py::arg("group_i"), py::arg("group_j"), py::arg("b_src"), py::arg("thresholds"),
          py::arg("degree"), "The reuse plan for this geometry, from the process's cache or built.");
    m.def("bspline_reuse_remainder_blocks", &remainder_blocks, py::arg("seg_l"), py::arg("seg_r"), py::arg("a"),
          py::arg("edge_seg0"), py::arg("ground_z"),
          "The Sommerfeld remainder's block copies, from its census of edge pairs: (ta, tb, sa, sb).");
    m.def("bspline_reuse_plan_of_classes", &plan_of_classes, py::arg("cls"),
          "TEST HOOK (the ExternalClasses sub-gate): a plan of the given (n, n) classes.");
    m.def("bspline_reuse_cache", []() {
        const auto c = br::cache_counters();
        py::dict d;
        d["built"] = c.built;
        d["found"] = c.found;
        d["declined"] = c.declined;
        return d;
    }, "The process's bs2 reuse plans built and found, and how many of those built declined.");
    m.def("bspline_reuse_cache_clear", &br::clear_cache, "Drop every cached bs2 reuse plan (TEST HOOK).");
    m.def("bspline_reuse_class_limit", &br::set_class_limit, py::arg("classes"),
          "TEST HOOK: later plans decline past this many classes (0: the coding's own limit).");
    m.def("bspline_reuse_plan_allowance", &br::set_plan_allowance, py::arg("bytes"),
          "TEST HOOK (momwire#2123): later plans decline past `bytes` of kept plan (<= 0: the derived "
          "allowance, 4 B per pair plus `bspline_reuse_kept_bytes_per_class` per class the 4/3x cap allows).");
    m.def("bspline_reuse_small_deck_segs", &br::small_deck_segs,
          "Segments below which bs2's pair reuse steps aside.");
    m.def("bspline_reuse_set_small_deck_segs", &br::set_small_deck_segs, py::arg("segs"),
          "TEST HOOK: move the small-deck threshold (negative restores the constant).");
    m.def("bspline_reuse_length_bits", [](int b) {
        if (b != 0) br::set_length_bits(b);
        return br::length_bits();
    }, py::arg("bits") = 0, "TEST HOOK: the key's length resolution for later plans (< 0 restores; 0 reads).");
    m.def("bspline_reuse_kept_bytes_per_class", &br::kept_bytes_per_class, py::arg("degree"),
          "What a class costs the kept plan at this degree: its (d + 1)^2 complex values, its group "
          "entry, code and list entries.");
    m.attr("bspline_reuse_key_bits") = br::KEY_BITS;
    m.attr("bspline_reuse_near_lengths") = br::NEAR_LENGTHS;
    m.attr("bspline_pair_reuse_2063") = true;
}
