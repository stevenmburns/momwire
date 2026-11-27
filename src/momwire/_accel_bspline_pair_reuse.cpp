// bs2's fill-pair reuse for the Python route (momwire#2063 stage 3): the
// binding of `_bspline_pair_reuse_core.h`. Its own TU: copies and integers
// only, no kernel; `assemble` calls the windowed assembly across the TU
// boundary (`_bspline_core.h`).
#include "_accel_common.h"
#include "_mw_nd_binding.h"

#include "_bspline_core.h"
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

using I8 = py::array_t<int8_t, py::array::c_style | py::array::forcecast>;

PlanHandle reuse_plan(F64 seg_l, F64 seg_r, F64 a, I32 edge, py::object group_i, py::object group_j,
                      py::object b_src, std::vector<double> thresholds, int degree, py::object img_l,
                      py::object img_r, py::object img_mode) {
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
    F64 il, ir;
    I8 im;
    if (!img_l.is_none()) {
        il = img_l.cast<F64>();
        ir = img_r.cast<F64>();
        im = img_mode.cast<I8>();
        if (il.ndim() != 2 || il.shape(0) != n || il.shape(1) != 3 || ir.ndim() != 2 || ir.shape(0) != n ||
            ir.shape(1) != 3 || im.size() != n)
            throw std::invalid_argument("bspline pair_reuse: (n, 3) mirrored end points and n image modes");
        g.img_l = il.data();
        g.img_r = ir.data();
        g.img_mode = im.data();
    }
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

// TEST HOOK: the classes of pairs (rows[i], cols[j]) of one block.
py::array_t<int32_t> classes_of(const PlanHandle &h, int block, I64 rows, I64 cols) {
    const mw::pair_reuse::FamilyPlan &f = h.plan->fam.at(0);
    if (block < 0 || block >= f.spec.n_blocks || f.map[block].empty())
        throw std::invalid_argument("bspline pair_reuse: no such block");
    const int64_t n = f.spec.cols;
    py::array_t<int32_t> out({rows.size(), cols.size()});
    int32_t *o = out.mutable_data();
    for (py::ssize_t i = 0; i < rows.size(); i++)
        for (py::ssize_t j = 0; j < cols.size(); j++) {
            const int64_t r = rows.data()[i], c = cols.data()[j];
            if (r < 0 || r >= n || c < 0 || c >= n) throw std::invalid_argument("bspline pair_reuse: pair out of range");
            o[i * cols.size() + j] = f.map[block][static_cast<size_t>(r * n + c)] >> f.spec.code_bits;
        }
    return out;
}

py::array_t<int64_t> group_blocks(const PlanHandle &h) {
    const std::vector<br::Group> &gs = h.plan->fam.at(0).groups;
    py::array_t<int64_t> out(static_cast<py::ssize_t>(gs.size()));
    int64_t *p = out.mutable_data();
    for (size_t i = 0; i < gs.size(); i++) p[i] = gs[i].block;
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

// momwire#2166: observer rows r0..r1 of one edge (own columns o0..o1)
// assembled into Z from the class values, the window never written whole:
// `assemble_Z_bspline_windowed(J_win, ...)` on the window `gather` writes
// with its own columns replaced by `own` (or zero when None), bit for bit.
void assemble(const PlanHandle &h, C128 values, int64_t r0, int64_t r1, int64_t o0, int64_t o1, py::object own,
              I64 support_seg, F64 polys, F64 tangents, I64 m_idx, I64 n_idx, double omega, double eps, double mu,
              py::array_t<cd> Z, uintptr_t cancel_flag) {
    const int64_t n = h.n(), classes = h.plan->fam.at(0).classes;
    if (values.ndim() != 3 || values.shape(0) != classes || values.shape(1) != values.shape(2))
        throw std::invalid_argument("bspline pair_reuse: values are (classes, d + 1, d + 1)");
    br::Window w;
    w.plan = h.plan.get();
    w.values = values.data();
    w.d1 = static_cast<int>(values.shape(1));
    w.r0 = r0;
    w.r1 = r1;
    w.o0 = o0;
    w.o1 = o1;
    br::check_window(w);
    if (support_seg.ndim() != 2 || support_seg.shape(1) != w.d1)
        throw std::invalid_argument("bspline pair_reuse: support_seg is (n_basis, d + 1)");
    if (tangents.ndim() != 2 || tangents.shape(0) != n)
        throw std::invalid_argument("bspline pair_reuse: one tangent per segment of the plan");
    C128 own_arr;
    if (!own.is_none()) {
        own_arr = own.cast<C128>();
        if (own_arr.ndim() != 4 || own_arr.shape(0) != w.d1 || own_arr.shape(1) != w.d1 ||
            own_arr.shape(2) != r1 - r0 || own_arr.shape(3) != o1 - o0)
            throw std::invalid_argument("bspline pair_reuse: own is (d + 1, d + 1, r1 - r0, o1 - o0)");
        w.own = own_arr.data();
    }
    ndb::NumpyHost host;
    mw::nd::HostScope scope(host);
    mw::bspline::assemble_Z_bspline_windowed_tiles(
        [&w](const int64_t *rows, size_t nr, const int64_t *cols, size_t nc, cd *out) {
            br::window_tile(w, rows, nr, cols, nc, out);
        },
        ndb::in(support_seg), ndb::in(polys), ndb::in(tangents), ndb::in(m_idx), ndb::in(n_idx), r0, r1, 0, n,
        omega, eps, mu, ndb::in(Z), cancel_flag);
}

// momwire#2063 step 2a: observer rows r0..r1 against every source, the
// image block's weighted assembly into Z from the class values
// (`assemble_Z_bspline_weighted_windowed` on the window the image tile
// writes), the weights the rows' (r1 - r0, n) windows.
void assemble_image(const PlanHandle &h, C128 values, int64_t r0, int64_t r1, I64 support_seg, F64 polys,
                    py::array_t<cd, py::array::c_style | py::array::forcecast> w_A,
                    py::array_t<cd, py::array::c_style | py::array::forcecast> w_Phi, I64 m_idx, I64 n_idx, double omega, double eps, double mu, cd scale,
                    py::array_t<cd> Z, uintptr_t cancel_flag) {
    const int64_t n = h.n(), classes = h.plan->fam.at(0).classes;
    if (values.ndim() != 3 || values.shape(0) != classes || values.shape(1) != values.shape(2))
        throw std::invalid_argument("bspline pair_reuse: values are (classes, d + 1, d + 1)");
    br::ImageWindow w;
    w.plan = h.plan.get();
    w.values = values.data();
    w.d1 = static_cast<int>(values.shape(1));
    w.zero_class = br::sentinel_class(*h.plan);
    br::check_image_window(w);
    if (!(0 <= r0 && r0 < r1 && r1 <= n)) throw std::invalid_argument("bspline pair_reuse: rows out of range");
    if (support_seg.ndim() != 2 || support_seg.shape(1) != w.d1)
        throw std::invalid_argument("bspline pair_reuse: support_seg is (n_basis, d + 1)");
    ndb::NumpyHost host;
    mw::nd::HostScope scope(host);
    mw::bspline::assemble_Z_bspline_weighted_windowed_tiles(
        [&w](const int64_t *rows, size_t nr, const int64_t *cols, size_t nc, cd *out) {
            br::image_tile(w, rows, nr, cols, nc, out);
        },
        ndb::in(support_seg), ndb::in(polys), ndb::in(w_A), ndb::in(w_Phi), ndb::in(m_idx), ndb::in(n_idx), n, r0,
        r1, 0, n, omega, eps, mu, scale, ndb::in(Z), cancel_flag);
}

// momwire#2063 step 2b: the Sommerfeld remainder's plan of segment pairs.
// `raised` the ordered pairs whose remainder order is raised, flat o * n + s.
PlanHandle remainder_plan(F64 seg_l, F64 seg_r, F64 h, double ground_z, I64 raised, int degree) {
    const int64_t n = static_cast<int64_t>(h.size());
    if (seg_l.ndim() != 2 || seg_l.shape(0) != n || seg_l.shape(1) != 3 || seg_r.ndim() != 2 ||
        seg_r.shape(0) != n || seg_r.shape(1) != 3)
        throw std::invalid_argument("bspline pair_reuse: (n, 3) end points and n lengths");
    br::RemainderPlanGeometry g;
    g.n = n;
    g.seg_l = seg_l.data();
    g.seg_r = seg_r.data();
    g.h = h.data();
    g.ground_z = ground_z;
    g.raised = raised.data();
    g.n_raised = static_cast<int64_t>(raised.size());
    g.degree = degree;
    PlanHandle out;
    {
        py::gil_scoped_release release;
        out.plan = br::remainder_plan_for(g, &out.built);
    }
    return out;
}

// The remainder plan's class map, representatives, their codes and the
// singles' bits as the remainder kernel's class route reads them: views of
// the plan's own arrays (no copy), each holding the plan alive.
py::tuple remainder_arrays(const PlanHandle &h) {
    const br::RemainderClasses c = br::remainder_classes(*h.plan);
    if (!c.map) throw std::invalid_argument("bspline pair_reuse: not a served remainder plan");
    py::object base = py::cast(h);
    const py::ssize_t n = static_cast<py::ssize_t>(c.n), C = static_cast<py::ssize_t>(c.classes);
    py::array_t<int32_t> map({n, n}, {static_cast<py::ssize_t>(n * sizeof(int32_t)),
                                      static_cast<py::ssize_t>(sizeof(int32_t))},
                             c.map, base);
    py::array_t<int64_t> rep(C, c.rep_flat, base);
    py::array_t<uint8_t> code(C, c.rep_code, base);
    py::object single = py::none();
    if (c.single) single = py::array_t<uint64_t>((C + 63) / 64, c.single, base);
    for (py::array a : {py::array(map), py::array(rep), py::array(code)})
        py::detail::array_proxy(a.ptr())->flags &= ~py::detail::npy_api::NPY_ARRAY_WRITEABLE_;
    return py::make_tuple(map, rep, code, single);
}

py::object remainder_wire_map(const PlanRef &r, py::object tied, I32 wire_of_seg, int64_t n_wires) {
    const std::shared_ptr<const br::Plan> p = r.plan.lock();
    if (!p) return py::none();
    const br::RemainderClasses c = br::remainder_classes(*p);
    if (!c.map) throw std::invalid_argument("bspline pair_reuse: not a served remainder plan");
    if (wire_of_seg.size() != c.n) throw std::invalid_argument("bspline pair_reuse: one wire per segment");
    if (n_wires < 1) throw std::invalid_argument("bspline pair_reuse: n_wires must be positive");
    const int32_t *w = wire_of_seg.data();
    for (int64_t s = 0; s < c.n; s++)
        if (w[s] < 0 || w[s] >= n_wires) throw std::invalid_argument("bspline pair_reuse: wire out of range");
    py::array_t<uint8_t, py::array::c_style | py::array::forcecast> t;
    if (!tied.is_none()) {
        t = tied.cast<py::array_t<uint8_t, py::array::c_style | py::array::forcecast>>();
        if (t.size() != c.classes) throw std::invalid_argument("bspline pair_reuse: one tied flag per class");
    }
    py::array_t<int64_t> out({static_cast<py::ssize_t>(n_wires), static_cast<py::ssize_t>(n_wires),
                              static_cast<py::ssize_t>(3)});
    int64_t *o = out.mutable_data();
    {
        py::gil_scoped_release release;
        br::remainder_wire_map(*p, tied.is_none() ? nullptr : t.data(), w, n_wires, o);
    }
    return out;
}

py::object wire_map(const PlanRef &r, I32 wire_of_seg, int64_t n_wires, int block) {
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
        br::wire_map(*p, w, n_wires, o, block);
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
    // Per block, by the block of each class's representative (momwire#2063
    // step 2a): block 0 the direct pairs, block 1 the image's.
    out["blocks"] = f.spec.n_blocks;
    out["classes_block"] = py::make_tuple(f.cls_block[0], f.cls_block[1]);
    out["near_block"] = py::make_tuple(f.near_block[0], f.near_block[1]);
    out["evals_block"] = py::make_tuple(f.evals_block[0], f.evals_block[1]);
    out["image_sentinel_pairs"] = br::image_sentinel_pairs(p);
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
        .def("wire_map", &wire_map, py::arg("wire_of_seg"), py::arg("n_wires"), py::arg("block") = 0,
             "The (W, W, 3) int64 map of observer wire x source wire x (computed, reused, near) of one "
             "block (0 direct, 1 image), or None when the plan is gone.")
        .def("remainder_wire_map", &remainder_wire_map, py::arg("tied"), py::arg("wire_of_seg"),
             py::arg("n_wires"),
             "A remainder plan's (W, W, 3) map of computed, reused and 0 (momwire#2063 step 2b), the "
             "classes `tied` names ((classes,) uint8, or None) counted computed; None when the plan is gone.");
    py::class_<PlanHandle>(m, "BSplineReusePlan",
                           "bs2's fill-pair reuse plan (momwire#2063): the classes of the (observer "
                           "segment, source segment) pairs of the moment fill, from the process's cache.")
        .def_readonly("built", &PlanHandle::built, "Whether this call's census built the plan.")
        .def_property_readonly("declined", [](const PlanHandle &h) -> py::object {
            if (h.plan->declined.empty()) return py::none();
            return py::str(h.plan->declined);
        }, "Why this plan does not serve the deck (it then has no classes), or None.")
        .def_property_readonly("classes", [](const PlanHandle &h) { return h.plan->fam.at(0).classes; })
        .def_property_readonly("image", [](const PlanHandle &h) { return br::has_image(*h.plan); },
                               "Whether the plan has the image block (momwire#2063 step 2a).")
        .def_property_readonly("plan_bytes", [](const PlanHandle &h) { return h.plan->budget_bytes; },
                               "The kept plan's bytes: its maps and groups, and one wavenumber's values.")
        .def_property_readonly("census_peak_bytes", [](const PlanHandle &h) { return h.plan->census_peak_bytes; })
        .def("stats", &stats)
        .def("groups", &groups,
             "The representatives' evaluation groups: (observer rows, sources, classes, at), "
             "class cls[q]'s value at entry at[q] = row * |sources| + column of the group's call.")
        .def("group_blocks", &group_blocks, "Each group's block, in groups() order (1: the image's).")
        .def("classes_of", &classes_of, py::arg("block"), py::arg("rows"), py::arg("cols"),
             "TEST HOOK: the (len(rows), len(cols)) classes of one block's pairs.")
        .def("assemble_image", &assemble_image, py::arg("values"), py::arg("r0"), py::arg("r1"),
             py::arg("support_seg"), py::arg("polys"), py::arg("w_A"), py::arg("w_Phi"), py::arg("m_idx"),
             py::arg("n_idx"), py::arg("omega"), py::arg("eps"), py::arg("mu"), py::arg("scale"), py::arg("Z"),
             py::arg("cancel_flag") = 0,
             "momwire#2063 step 2a: observer rows r0..r1 of the image block added into Z from the class "
             "values with the rows' (r1 - r0, n) weights, as assemble_Z_bspline_weighted_windowed adds "
             "the window, without writing the window.")
        .def("gather", &gather, py::arg("values"), py::arg("r0"), py::arg("r1"), py::arg("out"),
             "Observer rows r0..r1 of the moment tensor, (d + 1, d + 1, r1 - r0, n), from the "
             "class values (classes, d + 1, d + 1).")
        .def("assemble", &assemble, py::arg("values"), py::arg("r0"), py::arg("r1"), py::arg("o0"),
             py::arg("o1"), py::arg("own"), py::arg("support_seg"), py::arg("polys"), py::arg("tangents"),
             py::arg("m_idx"), py::arg("n_idx"), py::arg("omega"), py::arg("eps"), py::arg("mu"), py::arg("Z"),
             py::arg("cancel_flag") = 0,
             "momwire#2166: observer rows r0..r1 of one edge (its own columns o0..o1: `own`, "
             "(d + 1, d + 1, r1 - r0, o1 - o0), or zero for None) added into Z from the class values, "
             "as assemble_Z_bspline_windowed adds the gathered window, without writing the window.")
        .def("remainder_arrays", &remainder_arrays,
             "A remainder plan's (class map (n, n) int32, representatives (C,) int64, their codes (C,) "
             "uint8, singles' bits (C + 63) // 64 uint64 or None), read-only views the remainder "
             "kernel's class route takes (momwire#2063 step 2b).")
        .def("ref", [](const PlanHandle &h) { return PlanRef{h.plan}; }, "The plan, held weakly.");
    m.def("bspline_reuse_plan", &reuse_plan, py::arg("seg_l"), py::arg("seg_r"), py::arg("a"),
          py::arg("edge"), py::arg("group_i"), py::arg("group_j"), py::arg("b_src"), py::arg("thresholds"),
          py::arg("degree"), py::arg("img_l") = py::none(), py::arg("img_r") = py::none(),
          py::arg("img_mode") = py::none(),
          "The reuse plan for this geometry, from the process's cache or built; with the mirrored end "
          "points and each segment's own-image mode, its image block too (momwire#2063 step 2a).");
    m.def("bspline_reuse_remainder_plan", &remainder_plan, py::arg("seg_l"), py::arg("seg_r"), py::arg("h"),
          py::arg("ground_z"), py::arg("raised"), py::arg("degree"),
          "The Sommerfeld remainder's plan of segment pairs by plane invariants (momwire#2063 step 2b), "
          "from the process's cache or built.");
    m.def("bspline_reuse_set_remainder_key_probe", &br::set_remainder_key_probe, py::arg("probe"),
          "TEST HOOK (momwire#2063 step 2b): 1 keys the remainder without the height sum, 2 on the "
          "tangents' fixed-frame components, 3 on the height difference; 0 restores.");
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
    m.def("bspline_reuse_window_counts", []() {
        const mw::bspline::WindowTileCounts c = mw::bspline::window_tile_counts();
        py::dict d;
        d["tiled"] = c.tiled;
        d["whole"] = c.whole;
        d["tiles"] = c.tiles;
        d["weighted_tiled"] = c.wtiled;
        d["weighted_whole"] = c.wwhole;
        return d;
    }, "Process totals of the fused windowed assembly (TEST HOOK, momwire#2166): calls run tile by "
       "tile, calls whose window was materialised (no lane kernel), and tiles.");
    m.def("bspline_reuse_set_window_probe", &br::set_window_probe, py::arg("probe"),
          "TEST HOOK (momwire#2166): 1 copies a reciprocal member untransposed in the fused "
          "window; 0 restores.");
    m.def("bspline_reuse_cache_clear", &br::clear_cache, "Drop every cached bs2 reuse plan (TEST HOOK).");
    m.def("bspline_reuse_set_image_key_probe", &br::set_image_key_probe, py::arg("probe"),
          "TEST HOOK (momwire#2063 step 2a): 1 keys the image block on the source's unmirrored "
          "direction; 0 restores.");
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
