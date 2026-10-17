#include "_accel_common.h"
#include "_mw_binding.h"
#include "_nec5_razor_seam_core.h"
#include "_nec5_bspline_seam_core.h"
#include "_bspline_front_core.h"
#include "_bspline_fill_core.h"
#include "_razor_buried_core.h"
#include "_somm_below_grid_core.h"
#include "_somm_below_fill_core.h"
#include "_razor_front_core.h"
#include "_razor_solve_core.h"

// The binding half of `_razor_solve_core.cpp`: `RazorSolveContext`, a Python
// handle on one `mw::razor::SolveContext` (its buffer pool and counters).
//
// LAPACK comes from SciPy itself. `scipy.linalg.cython_lapack` exports every
// routine as a PyCapsule holding the C function pointer of the Fortran
// symbol in SciPy's bundled OpenBLAS -- the library `scipy.linalg.lu_factor`
// and `lu_solve` reach through their f2py wrappers. The pointers are read
// here, once, and handed to the core as plain function pointers: the core
// stays Python-free and the factorisation is the same routine on the same
// library as the Python route's.

using mw::binding::flat;
using mw::binding::in;
using cd = std::complex<double>;
using cdarr = py::array_t<cd, py::array::c_style | py::array::forcecast>;
using darr = py::array_t<double, py::array::c_style | py::array::forcecast>;
using iarr = py::array_t<int64_t, py::array::c_style | py::array::forcecast>;
using barr = py::array_t<bool, py::array::c_style | py::array::forcecast>;

namespace {

void *capsule_pointer(const py::dict &capi, const char *name) {
    py::object cap = capi[name];
    PyObject *p = cap.ptr();
    if (!PyCapsule_CheckExact(p))
        throw std::runtime_error(std::string("cython_lapack.") + name + " is not a capsule");
    void *fn = PyCapsule_GetPointer(p, PyCapsule_GetName(p));
    if (fn == nullptr) throw py::error_already_set();
    return fn;
}

const mw::razor::Lapack &scipy_lapack() {
    static mw::razor::Lapack lp;
    if (lp.getrf == nullptr) {
        py::dict capi = py::module_::import("scipy.linalg.cython_lapack").attr("__pyx_capi__");
        lp.getrf = reinterpret_cast<mw::razor::zgetrf_fn>(capsule_pointer(capi, "zgetrf"));
        lp.getrs = reinterpret_cast<mw::razor::zgetrs_fn>(capsule_pointer(capi, "zgetrs"));
    }
    return lp;
}

struct Held {
    // Keeps every converted array alive for the call's duration.
    std::vector<py::object> keep;
    template <class A>
    A hold(A a) {
        keep.push_back(a);
        return a;
    }
};

mw::razor::SourceSet source_set(Held &h, const py::tuple &t) {
    // (seg_p0, seg_t, group_j, a_ek, td_a, td_b)
    if (t.size() != 6) throw std::invalid_argument("razor_solve: a source set has 6 arrays");
    mw::razor::SourceSet s;
    s.seg_p0 = in<double, 2>(h.hold(t[0].cast<darr>()));
    s.seg_t = in<double, 2>(h.hold(t[1].cast<darr>()));
    s.group_j = flat(h.hold(t[2].cast<iarr>()));
    s.a_ek = flat(h.hold(t[3].cast<darr>()));
    s.td_a = in<double, 2>(h.hold(t[4].cast<darr>()));
    s.td_b = in<double, 2>(h.hold(t[5].cast<darr>()));
    return s;
}

py::tuple port_solution(mw::razor::SolveContext &ctx, darr cent, iarr group_i, darr b_obs,
                        darr seg_h, darr a, darr xg, darr wg, iarr s_a, iarr s_b,
                        darr h_a, darr h_b, darr q_a, darr q_b, barr fall_a,
                        barr fall_b, darr tans, darr wts, iarr grounded, double k,
                        double c_a_imag, double c_phi_imag, py::tuple real,
                        py::object image, ptrdiff_t window_rows, cdarr rhs,
                        uintptr_t cancel_flag, bool want_z) {
    Held h;
    mw::razor::SolveInputs inp;
    inp.cent = in<double, 2>(cent);
    inp.group_i = flat(group_i);
    inp.b_obs = flat(b_obs);
    inp.seg_h = in<double, 1>(seg_h);
    inp.a = in<double, 1>(a);
    inp.xg = in<double, 1>(xg);
    inp.wg = in<double, 1>(wg);
    inp.s_a = in<int64_t, 1>(s_a);
    inp.s_b = in<int64_t, 1>(s_b);
    inp.h_a = in<double, 1>(h_a);
    inp.h_b = in<double, 1>(h_b);
    inp.q_a = in<double, 1>(q_a);
    inp.q_b = in<double, 1>(q_b);
    inp.fall_a = in<bool, 1>(fall_a);
    inp.fall_b = in<bool, 1>(fall_b);
    inp.tans = in<double, 2>(tans);
    inp.wts = in<double, 1>(wts);
    inp.grounded = flat(grounded);
    inp.n_path = 2;
    inp.k = k;
    inp.c_a_imag = c_a_imag;
    inp.c_phi_imag = c_phi_imag;
    inp.real = source_set(h, real);
    inp.has_image = !image.is_none();
    if (inp.has_image) inp.image = source_set(h, image.cast<py::tuple>());
    inp.window_rows = window_rows;
    inp.cancel_flag = cancel_flag;

    const ptrdiff_t n = s_a.shape(0);
    if (rhs.ndim() != 2 || rhs.shape(0) != n)
        throw std::invalid_argument("razor_solve: rhs must be (n_basis, n_rhs)");
    const ptrdiff_t m = rhs.shape(1);
    const mw::razor::Lapack &lp = scipy_lapack();
    // X starts as the right-hand side, column-major, and getrs overwrites it
    // (`lu_solve`'s F-ordered copy of b).
    py::array_t<cd, py::array::f_style> X({n, m});
    {
        auto src = rhs.unchecked<2>();
        cd *x = X.mutable_data();
        for (ptrdiff_t j = 0; j < m; j++)
            for (ptrdiff_t i = 0; i < n; i++) x[j * n + i] = src(i, j);
    }
    py::object Z = py::none();
    int info = 0;
    {
        py::gil_scoped_release release;
        ctx.fill(inp);
    }
    if (want_z) {
        py::array_t<cd, py::array::f_style> Zc({n, n});
        std::copy(ctx.z(), ctx.z() + n * n, Zc.mutable_data());
        Z = Zc;
    }
    {
        cd *x = X.mutable_data();
        py::gil_scoped_release release;
        info = ctx.lu_solve(lp, x, static_cast<int>(m));
    }
    return py::make_tuple(X, info, Z);
}

py::dict counters(const mw::razor::SolveContext &ctx) {
    const auto &c = ctx.counters;
    py::dict d;
    d["solves"] = c.solves;
    d["windows"] = c.windows;
    d["image_windows"] = c.image_windows;
    d["compose_windows"] = c.compose_windows;
    d["seg_moments_calls"] = c.seg_moments_calls;
    d["assemble_t1_calls"] = c.assemble_t1_calls;
    d["t2_rows_calls"] = c.t2_rows_calls;
    d["lu_calls"] = c.lu_calls;
    d["pool_grows"] = c.pool_grows;
    d["fill_seconds"] = c.fill_seconds;
    d["lu_seconds"] = c.lu_seconds;
    return d;
}

// ---------------------------------------------------------------------------
// The front half (`_razor_front_core.h`): RazorSolver's own constructor
// arguments in, geometry -> prepared tables -> port columns -> fill -> LU in
// C++, the solved coefficients out.

// The arguments did not convert the way the constructor reads them: the
// Python front half answers (and refuses in its own words).
struct Malformed {};

bool is_sequence(const py::handle &h) {
    return py::isinstance<py::list>(h) || py::isinstance<py::tuple>(h) ||
           (py::isinstance<py::array>(h) && py::reinterpret_borrow<py::array>(h).ndim() >= 1);
}

// An integer as the constructor's `int(v)` / range checks read it: Python
// and numpy integers only (a float index is the Python front half's).
int64_t as_index(const py::handle &h) {
    if (py::isinstance<py::bool_>(h)) throw Malformed{};
    py::object idx = py::reinterpret_borrow<py::object>(h);
    if (PyIndex_Check(idx.ptr()) == 0) throw Malformed{};
    return py::int_(idx).cast<int64_t>();
}

// `int(v)` of a segment count, which truncates a float like the
// constructor's `[int(v) for v in npe]`.
int64_t as_count(const py::handle &h) {
    PyObject *o = PyNumber_Long(h.ptr());
    if (o == nullptr) {
        PyErr_Clear();
        throw Malformed{};
    }
    py::int_ v = py::reinterpret_steal<py::int_>(o);
    return v.cast<int64_t>();
}

double as_float(const py::handle &h) {
    PyObject *o = PyNumber_Float(h.ptr());
    if (o == nullptr) {
        PyErr_Clear();
        throw Malformed{};
    }
    return py::reinterpret_steal<py::float_>(o).cast<double>();
}

std::complex<double> as_complex(const py::handle &h) {
    py::object c = py::module_::import("builtins").attr("complex");
    try {
        return c(h).cast<std::complex<double>>();
    } catch (py::error_already_set &) {
        throw Malformed{};
    }
}

mw::razor::End as_end(const py::handle &h) {
    if (!py::isinstance<py::str>(h)) throw Malformed{};
    const std::string e = h.cast<std::string>();
    if (e == "start") return mw::razor::End::start;
    if (e == "end") return mw::razor::End::end;
    throw Malformed{};
}

// RazorSolver's keyword arguments (the subset the front half reads) ->
// FrontInputs, as `__init__` reads each one.
mw::razor::FrontInputs front_inputs(const py::dict &kw) {
    using namespace mw::razor;
    FrontInputs in;
    auto get = [&](const char *k) -> py::object {
        return kw.contains(k) ? py::reinterpret_borrow<py::object>(kw[k]) : py::none();
    };
    py::object wires = get("wires");
    if (!wires.is_none()) {
        for (py::handle w : wires) {
            py::array_t<double, py::array::c_style | py::array::forcecast> a;
            try {
                a = py::array_t<double, py::array::c_style | py::array::forcecast>::ensure(w);
            } catch (py::error_already_set &) {
                throw Malformed{};
            }
            if (!a || a.ndim() != 2 || a.shape(1) != 3) throw Malformed{};
            const double *d = a.data();
            in.points.insert(in.points.end(), d, d + a.size());
            in.wire_points.push_back(a.shape(0));
        }
    }
    py::object npe = get("n_per_edge_per_wire");
    if (!npe.is_none()) {
        in.counts_given = true;
        for (py::handle e : npe) {
            EdgeCounts ec;
            if (e.is_none()) {
                ec.kind = EdgeCounts::none;
            } else if (is_sequence(e)) {
                ec.kind = EdgeCounts::list;
                for (py::handle v : e) ec.v.push_back(as_count(v));
            } else {
                ec.kind = EdgeCounts::scalar;
                ec.v.push_back(as_count(e));
            }
            in.counts.push_back(ec);
        }
    }
    if (kw.contains("nsegs")) in.nsegs = as_count(kw["nsegs"]);
    {
        py::object r = kw.contains("wire_radius") ? get("wire_radius") : py::float_(0.0005);
        py::array_t<double, py::array::c_style | py::array::forcecast> a;
        try {
            a = py::array_t<double, py::array::c_style | py::array::forcecast>::ensure(r);
        } catch (py::error_already_set &) {
            throw Malformed{};
        }
        if (!a) throw Malformed{};
        for (ptrdiff_t k = 0; k < a.ndim(); k++) in.radius_shape.push_back(a.shape(k));
        in.radius.assign(a.data(), a.data() + a.size());
    }
    py::object feeds = get("feeds");
    if (feeds.is_none()) {
        in.feeds_given = false;
        in.feed_wire_index = kw.contains("feed_wire_index") ? as_index(kw["feed_wire_index"]) : 0;
        py::object arc = get("feed_arclength");
        in.feed_midpoint = arc.is_none();
        if (!arc.is_none()) in.feed_arclength = as_float(arc);
    } else {
        for (py::handle f : feeds) {
            if (!is_sequence(f) || py::len(f) != 3) throw Malformed{};
            py::sequence t = py::reinterpret_borrow<py::sequence>(f);
            FeedSpec fs;
            fs.wire = as_index(t[0]);
            py::object arc = t[1];
            fs.midpoint = arc.is_none();
            if (!arc.is_none()) fs.arc = as_float(arc);
            fs.volts = as_complex(t[2]);
            in.feeds.push_back(fs);
        }
    }
    py::object gaps = get("node_gaps");
    if (!gaps.is_none()) {
        for (py::handle g : gaps) {
            if (!is_sequence(g) || py::len(g) != 3) throw Malformed{};
            py::sequence t = py::reinterpret_borrow<py::sequence>(g);
            NodeGapSpec ns;
            ns.wire = as_index(t[0]);
            ns.end = as_end(t[1]);
            ns.volts = as_complex(t[2]);
            in.node_gaps.push_back(ns);
        }
    }
    py::object gz = get("ground_z");
    in.has_ground = !gz.is_none();
    if (in.has_ground) in.ground_z = as_float(gz);
    py::object geps = get("ground_eps");
    if (!geps.is_none()) {
        // `(eps_r, sigma)` only: a complex eps~ is not the seam's spelling.
        if (!is_sequence(geps) || py::len(geps) != 2) throw Malformed{};
        py::sequence t = py::reinterpret_borrow<py::sequence>(geps);
        in.ground_eps_given = true;
        in.ground_eps_r = as_float(t[0]);
        in.ground_sigma = as_float(t[1]);
    }
    py::object gmodel = get("ground_model");
    if (!gmodel.is_none()) {
        if (!py::isinstance<py::str>(gmodel)) throw Malformed{};
        const std::string m = gmodel.cast<std::string>();
        if (m != "refl-coef" && m != "sommerfeld") throw Malformed{};
        in.ground_sommerfeld = m == "sommerfeld";
    }
    in.wavelength = kw.contains("wavelength") ? as_float(kw["wavelength"]) : 22.0;
    in.extended_kernel = kw.contains("extended_kernel") && py::bool_(get("extended_kernel"));
    // The wire loading (`_solver_for`'s): an LD 5 conductivity per piece (NaN
    // where none) and LD 2 / 3 `DistributedRLC`s (or None).
    py::object cond = get("wire_conductivity");
    if (!cond.is_none()) {
        in.loading.has_conductivity = true;
        for (py::handle v : cond) in.loading.conductivity.push_back(v.cast<double>());
    }
    py::object rlc = get("distributed_rlc");
    if (!rlc.is_none()) {
        in.loading.has_rlc = true;
        for (py::handle v : rlc) {
            mw::razor::WireRLC e;
            if (!v.is_none()) {
                const std::string kind = v.attr("kind").cast<std::string>();
                e.kind = kind == "series" ? mw::razor::WireRLC::series : mw::razor::WireRLC::parallel;
                e.r = v.attr("r").cast<double>();
                e.l = v.attr("l").cast<double>();
                e.c = v.attr("c").cast<double>();
            }
            in.loading.rlc.push_back(e);
        }
    }
    py::object ek = get("exact_kernel");
    if (py::isinstance<py::str>(ek)) {
        if (ek.cast<std::string>() != "auto") throw Malformed{};
        in.exact_kernel = ExactKernel::automatic;
    } else {
        in.exact_kernel = (!ek.is_none() && py::bool_(ek)) ? ExactKernel::on : ExactKernel::off;
    }
    py::object nqp = get("n_qp_path");
    in.n_qp_path_given = !nqp.is_none();
    if (in.n_qp_path_given) in.n_qp_path = as_count(nqp);
    if (kw.contains("n_qp_source")) in.n_qp_source = as_count(kw["n_qp_source"]);
    if (kw.contains("n_qp_sommerfeld")) in.n_qp_sommerfeld = as_count(kw["n_qp_sommerfeld"]);
    return in;
}

template <class T>
py::array_t<T> np1(const std::vector<T> &v) {
    py::array_t<T> a(static_cast<ptrdiff_t>(v.size()));
    std::copy(v.begin(), v.end(), a.mutable_data());
    return a;
}

template <class T>
py::array_t<T> np2(const std::vector<T> &v, ptrdiff_t rows, ptrdiff_t cols) {
    py::array_t<T> a({rows, cols});
    std::copy(v.begin(), v.end(), a.mutable_data());
    return a;
}

py::array_t<bool> npb(const mw::razor::BoolArray &b, ptrdiff_t rows, ptrdiff_t cols) {
    py::array_t<bool> a = cols > 0 ? py::array_t<bool>({rows, cols}) : py::array_t<bool>(rows);
    std::copy(b.data(), b.data() + b.n, a.mutable_data());
    return a;
}

py::tuple end_label(const std::pair<int64_t, mw::razor::End> &e) {
    return py::make_tuple(e.first, e.second == mw::razor::End::start ? "start" : "end");
}

// `_build_geometry`'s dict, key for key and type for type.
py::dict geometry_dict(const mw::razor::Front &f) {
    py::dict g;
    py::list per_wire, seg_off, basis_off;
    for (int64_t w = 0; w < f.n_wires; w++) {
        const int64_t a = f.seg_offsets[w] + w, b = f.seg_offsets[w + 1] + w + 1;
        py::array_t<double> arc(b - a);
        std::copy(f.arc_at_knot.begin() + a, f.arc_at_knot.begin() + b, arc.mutable_data());
        py::dict pw;
        pw["arc_at_knot"] = arc;
        pw["n_total"] = f.seg_offsets[w + 1] - f.seg_offsets[w];
        per_wire.append(pw);
    }
    for (int64_t v : f.seg_offsets) seg_off.append(v);
    for (int64_t v : f.basis_offsets) basis_off.append(v);
    py::list junctions;
    for (const auto &jn : f.junctions) {
        py::list ends, bases;
        for (const auto &e : jn.ends) ends.append(end_label(e));
        for (int64_t b : jn.bases) bases.append(b);
        py::dict d;
        d["ends"] = ends;
        d["bases"] = bases;
        d["grounded"] = jn.grounded;
        junctions.append(d);
    }
    g["per_wire"] = per_wire;
    g["seg_offsets"] = seg_off;
    g["basis_offsets"] = basis_off;
    g["n_segs_total"] = f.n_seg;
    g["n_basis_interior"] = f.n_interior;
    g["n_basis_total"] = f.n_basis;
    g["seg_p0"] = np2(f.seg_p0, f.n_seg, 3);
    g["seg_t"] = np2(f.seg_t, f.n_seg, 3);
    g["seg_h"] = np1(f.seg_h);
    g["wing_seg"] = np2(f.wing_seg, f.n_basis, 2);
    g["wing_rise"] = npb(f.wing_rise, f.n_basis, 2);
    g["wing_sigma"] = np2(f.wing_sigma, f.n_basis, 2);
    g["junctions"] = junctions;
    g["grounded_bases"] = np1(f.grounded_bases);
    return g;
}

// Every table the fill reads, and the constructor's normalised state, for
// the byte-for-byte gates against the Python front half.
py::dict front_dict(const mw::razor::Front &f) {
    const ptrdiff_t n = f.n_basis, ns = f.n_seg;
    py::dict d;
    d["s_a"] = np1(f.s_a);
    d["s_b"] = np1(f.s_b);
    d["h_a"] = np1(f.h_a);
    d["h_b"] = np1(f.h_b);
    d["q_a"] = np1(f.q_a);
    d["q_b"] = np1(f.q_b);
    d["cent"] = np2(f.cent, ns, 3);
    d["tans"] = np2(f.tans, 2 * n, 3);
    d["wts"] = np1(f.wts);
    d["td_a"] = np2(f.td_a, 3, n);
    d["td_b"] = np2(f.td_b, 3, n);
    d["fall_a"] = npb(f.fall_a, n, 0);
    d["fall_b"] = npb(f.fall_b, n, 0);
    d["a"] = np1(f.a_col);
    d["group"] = np1(f.group);
    d["a_ek"] = np1(f.a_ek);
    d["b_obs"] = np1(f.b_obs);
    d["xg"] = np1(f.xg);
    d["wg"] = np1(f.wg);
    if (f.has_image) {
        d["img_seg_p0"] = np2(f.img_seg_p0, ns, 3);
        d["img_seg_t"] = np2(f.img_seg_t, ns, 3);
        d["img_td_a"] = np2(f.img_td_a, 3, n);
        d["img_td_b"] = np2(f.img_td_b, 3, n);
    }
    if (f.compose) {
        // The composing ground's per-solve half (momwire#1479 S3).
        d["eps_t"] = py::cast(f.eps_t);
        d["c2"] = py::cast(f.c2);
        d["obs_pts"] = np2(f.obs_pts, 2 * n, 3);
        d["sig_a"] = np1(f.sig_a);
        d["sig_b"] = np1(f.sig_b);
        d["r1_max"] = f.r1_max;
        d["n_qp_rem"] = f.n_qp_rem;
    }
    d["window_rows"] = f.window_rows;
    d["k"] = f.k;
    d["c"] = f.c;
    d["c_a_imag"] = f.c_a_imag;
    d["c_phi_imag"] = f.c_phi_imag;
    d["radius_per_wire"] = np1(f.radius_per_wire);
    d["uniform_radius"] = f.uniform_radius ? py::object(py::float_(f.radius_value)) : py::none();
    py::list npe;
    for (const auto &w : f.npe) {
        py::list e;
        for (int64_t c : w) e.append(c);
        npe.append(e);
    }
    d["n_per_edge_per_wire"] = npe;
    py::list feeds, gaps;
    for (const auto &fs : f.feeds)
        feeds.append(py::make_tuple(fs.wire, fs.midpoint ? py::object(py::none())
                                                         : py::object(py::float_(fs.arc)),
                                    fs.volts));
    for (const auto &g : f.node_gaps)
        gaps.append(py::make_tuple(g.wire, g.end == mw::razor::End::start ? "start" : "end",
                                   g.volts));
    d["feeds"] = feeds;
    d["node_gaps"] = gaps;
    d["feed_basis"] = np1(f.feed_basis);
    return d;
}

py::list ties_list(const mw::razor::Front &f) {
    py::list out;
    for (const auto &t : f.ties)
        out.append(py::make_tuple(t.feed, t.wire, t.target, t.lo, t.hi, t.picked));
    return out;
}

// Build the front half from RazorSolver's keyword arguments. Returns
// (decline, front): decline a str naming what is not covered (front None),
// or None with the Front built. FrontError raises as the Python side does.
std::pair<std::string, std::unique_ptr<mw::razor::Front>> build_front(const py::dict &kw,
                                                                     int64_t chunk_elems) {
    mw::razor::FrontInputs in;
    try {
        in = front_inputs(kw);
    } catch (Malformed &) {
        return {"malformed-args", nullptr};
    }
    in.chunk_elems = chunk_elems;
    std::unique_ptr<mw::razor::Front> f(new mw::razor::Front());
    std::string why;
    {
        py::gil_scoped_release release;
        why = f->build(in);
    }
    if (!why.empty()) return {why, nullptr};
    return {"", std::move(f)};
}

// momwire#1512 step 3's gate (`scripts/core_split/buried_z_compare.py`): a
// buried solve's matrix at each stage the C++ fill reaches, from the
// arguments the seam handed RazorSolver.
py::dict razor_buried_stages(py::dict kw, int64_t chunk_elems, std::string below_tables) {
    py::dict out;
    struct Restore {
        ~Restore() { somm_below::set_source(nullptr); }
    } restore;
    // "fill": the table filled in C++ (B5); else a bridge directory.
    if (below_tables == "fill")
        somm_below::set_source(somm_below::fill_source());
    else if (!below_tables.empty())
        somm_below::set_source(somm_below::bridge_source(below_tables));
    mw::razor::FrontInputs in;
    try {
        in = front_inputs(kw);
    } catch (Malformed &) {
        out["route"] = "malformed-args";
        return out;
    }
    in.chunk_elems = chunk_elems;
    mw::razor::Front f;
    std::string why;
    mw::razor::BuriedStages st;
    std::vector<cd> z;
    {
        py::gil_scoped_release release;
        why = f.build(in);
        if ((why.empty() || why == mw::razor::BURIED_FILL_PENDING) &&
            f.buried != mw::razor::Buried::none)
            why = mw::razor::fill_buried(f, z, &st, 0);
    }
    out["route"] = why.empty() ? std::string("filled") : "stopped:" + why;
    const ptrdiff_t n = f.n_basis;
    for (size_t i = 0; i < st.reached.size(); i++) {
        py::array_t<cd, py::array::f_style> a({n, n});
        std::copy(st.z[i].begin(), st.z[i].end(), a.mutable_data());
        out[py::str(st.reached[i])] = a;
    }
    return out;
}

py::dict front_port_solution(mw::razor::SolveContext &ctx, py::dict kw, int64_t chunk_elems,
                             uintptr_t cancel_flag, bool want_z, bool want_front) {
    py::dict out;
    auto built = build_front(kw, chunk_elems);
    if (!built.first.empty()) {
        out["decline"] = built.first;
        return out;
    }
    const mw::razor::Front &f = *built.second;
    const ptrdiff_t n = f.n_basis, m = f.n_ports;
    const mw::razor::Lapack &lp = scipy_lapack();  // with the GIL: it may import
    py::array_t<cd, py::array::f_style> X({n, m});
    int info = 0;
    {
        cd *x = X.mutable_data();
        py::gil_scoped_release release;
        for (ptrdiff_t j = 0; j < m; j++)
            for (ptrdiff_t i = 0; i < n; i++) x[j * n + i] = cd(f.cols[i * m + j], 0.0);
        mw::razor::fill_front(f, ctx, cancel_flag);
    }
    if (want_z) {
        py::array_t<cd, py::array::f_style> Zc({n, n});
        std::copy(ctx.z(), ctx.z() + n * n, Zc.mutable_data());
        out["Z"] = Zc;
    }
    {
        cd *x = X.mutable_data();
        py::gil_scoped_release release;
        info = ctx.lu_solve(lp, x, static_cast<int>(m));
    }
    out["decline"] = py::none();
    out["X"] = X;
    out["info"] = info;
    out["cols"] = np2(f.cols, n, m);
    out["geom"] = geometry_dict(f);
    out["ties"] = ties_list(f);
    if (want_front) out["front"] = front_dict(f);
    return out;
}

py::dict razor_front_build(py::dict kw, int64_t chunk_elems) {
    py::dict out;
    auto built = build_front(kw, chunk_elems);
    if (!built.first.empty()) {
        out["decline"] = built.first;
        return out;
    }
    const mw::razor::Front &f = *built.second;
    out["decline"] = py::none();
    out["cols"] = np2(f.cols, f.n_basis, f.n_ports);
    out["geom"] = geometry_dict(f);
    out["ties"] = ties_list(f);
    out["front"] = front_dict(f);
    return out;
}

// ---------------------------------------------------------------------------
// The NEC-5 seam's structure half (`_nec5_razor_seam_core.h`), test-facing:
// deck text in; per structure and run, what the C++ seam did; per
// preparation, the keyword arguments it hands RazorSolver (spelled exactly as
// `deck._solver.construct_solver` spells them, Python types included), the
// mesh and the structure counts, and on request the solved coefficients.

py::dict seam_kwargs(const mw::nec5::Prepared &p) {
    const mw::razor::FrontInputs &in = mw::nec5::razor_part(p).inputs;
    py::dict kw;
    py::list wires, npe;
    for (size_t w = 0; w < in.wire_points.size(); w++) {
        py::array_t<double> a({static_cast<ptrdiff_t>(2), static_cast<ptrdiff_t>(3)});
        std::copy(in.points.begin() + 6 * w, in.points.begin() + 6 * w + 6, a.mutable_data());
        wires.append(a);
        py::list c;
        for (int64_t v : in.counts[w].v) c.append(v);
        npe.append(c);
    }
    kw["wires"] = wires;
    kw["n_per_edge_per_wire"] = npe;
    py::list feeds;
    for (const auto &f : in.feeds) feeds.append(py::make_tuple(f.wire, f.arc, f.volts));
    kw["feeds"] = feeds;
    kw["wavelength"] = in.wavelength;
    if (in.radius_shape.empty()) {
        kw["wire_radius"] = in.radius[0];
    } else {
        py::list r;
        for (double v : in.radius) r.append(v);
        kw["wire_radius"] = r;
    }
    kw["cancel"] = py::none();
    if (!in.node_gaps.empty()) {
        py::list gaps;
        for (const auto &g : in.node_gaps)
            gaps.append(py::make_tuple(g.wire, g.end == mw::razor::End::start ? "start" : "end",
                                       g.volts));
        kw["node_gaps"] = gaps;
    }
    if (in.extended_kernel) kw["extended_kernel"] = true;
    if (in.has_ground) kw["ground_z"] = in.ground_z;
    if (in.ground_eps_given) {
        kw["ground_eps"] = py::make_tuple(in.ground_eps_r, in.ground_sigma);
        kw["ground_model"] = in.ground_sommerfeld ? "sommerfeld" : "refl-coef";
    }
    // `_solver_for`'s loading: an LD 5 conductivity array and an LD 2 / 3
    // list of `DistributedRLC` (or None), one entry per piece.
    if (in.loading.has_conductivity) {
        py::array_t<double> a(static_cast<ptrdiff_t>(in.loading.conductivity.size()));
        std::copy(in.loading.conductivity.begin(), in.loading.conductivity.end(), a.mutable_data());
        kw["wire_conductivity"] = a;
    }
    if (in.loading.has_rlc) {
        py::object rlc_type = py::module_::import("momwire._wire_loading").attr("DistributedRLC");
        py::list rlc;
        for (const auto &e : in.loading.rlc) {
            if (e.kind == mw::razor::WireRLC::none)
                rlc.append(py::none());
            else
                rlc.append(rlc_type(e.kind == mw::razor::WireRLC::series ? "series" : "parallel",
                                    e.r, e.l, e.c));
        }
        kw["distributed_rlc"] = rlc;
    }
    kw["nec5_quadrature"] = true;
    kw["exact_kernel"] = "auto";
    return kw;
}

py::object node_tuple(const mw::nec5::Node &n) {
    return py::make_tuple(py::int_(py::str(n.tag.str())), py::int_(py::str(n.node.str())));
}

py::dict seam_mesh(const mw::nec5::Prepared &p) {
    const mw::nec5::SeamMesh &m = p.mesh;
    py::dict d;
    py::list pieces, junctions, sites, feeds, gaps;
    for (const auto &pc : m.pieces)
        pieces.append(py::make_tuple(py::int_(py::str(pc.tag.str())), pc.first_node,
                                     pc.last_node, pc.radius, pc.cut_start, pc.cut_end,
                                     py::make_tuple(pc.points[0], pc.points[1], pc.points[2]),
                                     py::make_tuple(pc.points[3], pc.points[4], pc.points[5])));
    for (const auto &g : m.junctions) {
        py::list members;
        for (const auto &e : g)
            members.append(py::make_tuple(e.first, e.second == mw::nec5::End::start ? "start" : "end"));
        junctions.append(members);
    }
    static const char *spell[] = {"gap", "node", "virtual"};
    static const char *ends[] = {"start", "end", "interior", "none"};
    for (const auto &s : m.sites) {
        py::dict x;
        x["at"] = node_tuple(s.at);
        x["spelling"] = spell[s.spelling];
        x["piece"] = s.piece;
        x["end"] = ends[s.end];
        x["sign"] = s.sign;
        x["arclength"] = s.arclength;
        x["contact"] = s.contact;
        x["crossing"] = s.crossing;
        x["index"] = s.index;
        x["column"] = s.column;
        x["weight"] = s.weight;
        sites.append(x);
    }
    for (long long i : m.feeds) feeds.append(i);
    for (long long i : m.gaps) gaps.append(i);
    d["pieces"] = pieces;
    d["junctions"] = junctions;
    d["sites"] = sites;
    d["feeds"] = feeds;
    d["gaps"] = gaps;
    py::list element_of;
    for (const auto &e : m.element_of) element_of.append(py::make_tuple(e.first, e.second));
    d["element_of"] = element_of;
    py::dict split;
    for (const auto &kv : m.split_elements)
        split[py::int_(kv.first)] = py::make_tuple(kv.second.lower, kv.second.upper,
                                                    kv.second.centre_piece, kv.second.centre_s);
    d["split_elements"] = split;
    d["node_count"] = p.structure.node_count;
    d["wire_element_count"] = p.structure.wire_element_count;
    d["unknown_count"] = p.structure.unknown_count;
    d["ground_plane"] = p.structure.ground_plane;
    d["frequency"] = p.frequency_mhz;
    d["wavelength"] = p.wavelength;
    d["omega"] = p.omega;
    if (p.has_medium)
        d["medium"] = py::make_tuple(p.medium_eps_r, p.medium_sigma,
                                     std::complex<double>(p.medium_eps_c[0], p.medium_eps_c[1]));
    else
        d["medium"] = py::none();
    py::list src;
    for (const auto &n : p.source_sites) src.append(node_tuple(n));
    d["source_sites"] = src;
    py::list cards;
    for (const auto &c : p.cards) {
        py::list payload;
        for (double v : c.payload) payload.append(v);
        cards.append(py::make_tuple(c.kind == mw::nec5::SeamCard::tl ? "TL" : "NT", py::tuple(payload),
                                    c.site_a, c.site_b, py::int_(py::str(c.tag_from.str())),
                                    c.segment_from, py::int_(py::str(c.tag_to.str())), c.segment_to));
    }
    d["cards"] = cards;
    py::list phantom;
    for (const auto &t : p.phantom_tags) phantom.append(py::int_(py::str(t.str())));
    d["phantom_tags"] = phantom;
    return d;
}

py::dict nec5_razor_seam(py::bytes latin1, bool solve) {
    const std::string text = latin1;
    mw::nec5::SeamResult r;
    {
        py::gil_scoped_release release;
        r = mw::nec5::razor_seam(text);
    }
    static const char *route[] = {"served", "refused", "raised", "declined"};
    static const char *read[] = {"refused", "raised", "single", "multi"};
    static const char *run[] = {"prepared", "shared", "refused", "raised", "declined"};
    py::dict out;
    out["route"] = route[r.route];
    out["message"] = r.message;
    out["error_kind"] = r.error_kind;
    out["chained"] = r.chained;
    py::list structures;
    for (const auto &so : r.structures) {
        py::list runs;
        for (const auto &o : so.runs) {
            py::dict x;
            x["kind"] = run[o.kind];
            x["message"] = o.message;
            x["error_kind"] = o.error_kind;
            x["prepared"] = o.prepared_index;
            runs.append(x);
        }
        py::dict sd;
        sd["read"] = read[so.read];
        sd["runs"] = runs;
        structures.append(sd);
    }
    out["structures"] = structures;
    py::list prepared;
    for (const auto &p : r.prepared) {
        py::dict pd;
        pd["kwargs"] = seam_kwargs(p);
        pd["mesh"] = seam_mesh(p);
        const mw::razor::Front &f = *mw::nec5::razor_part(p).front;
        pd["cols"] = np2(f.cols, f.n_basis, f.n_ports);
        pd["ties"] = ties_list(f);
        if (solve && !p.unsolved) {
            const ptrdiff_t n = f.n_basis, m = f.n_ports;
            const mw::razor::Lapack &lp = scipy_lapack();
            py::array_t<cd, py::array::f_style> X({n, m});
            int info = 0;
            {
                cd *x = X.mutable_data();
                py::gil_scoped_release release;
                mw::razor::SolveContext ctx;
                info = mw::razor::solve_ports(f, ctx, lp, x, 0);
            }
            pd["X"] = X;
            pd["info"] = info;
        }
        prepared.append(pd);
    }
    out["prepared"] = prepared;
    return out;
}

// PROTOTYPE (momwire#1517 step 1): the bspline seam's structure half,
// test-facing, as `nec5_razor_seam` is razor's: the route, each run's
// outcome, and per preparation BSplineSolver's keyword arguments (spelled as
// `construct_solver` spells them) and the mesh.
py::dict bspline_kwargs(const mw::nec5::Prepared &p) {
    const mw::nec5::BSplineFrontInputs &in = mw::nec5::bspline_part(p).inputs;
    py::dict kw;
    py::list wires, npe, feeds;
    for (size_t w = 0; w < in.n_per_edge.size(); w++) {
        py::array_t<double> a({static_cast<ptrdiff_t>(2), static_cast<ptrdiff_t>(3)});
        std::copy(in.points.begin() + 6 * w, in.points.begin() + 6 * w + 6, a.mutable_data());
        wires.append(a);
        py::list c;
        c.append(in.n_per_edge[w]);
        npe.append(c);
    }
    kw["wires"] = wires;
    kw["n_per_edge_per_wire"] = npe;
    for (const auto &f : in.feeds) feeds.append(py::make_tuple(f.wire, f.arc, cd(0.0, 0.0)));
    kw["feeds"] = feeds;
    kw["wavelength"] = in.wavelength;
    if (!in.radius_per_wire) {
        kw["wire_radius"] = in.radius[0];
    } else {
        py::list r;
        for (double v : in.radius) r.append(v);
        kw["wire_radius"] = r;
    }
    kw["cancel"] = py::none();
    if (!in.node_gaps.empty()) {
        py::list gaps;
        for (const auto &g : in.node_gaps)
            gaps.append(py::make_tuple(g.wire, g.end == mw::nec5::End::start ? "start" : "end",
                                       cd(0.0, 0.0)));
        kw["node_gaps"] = gaps;
    }
    if (in.junctions.empty()) {
        kw["junctions"] = py::none();
    } else {
        py::list js;
        for (const auto &g : in.junctions) {
            py::list members;
            for (const auto &e : g)
                members.append(py::make_tuple(e.first, e.second == mw::nec5::End::start ? "start" : "end"));
            js.append(members);
        }
        kw["junctions"] = js;
    }
    if (in.extended_kernel) kw["extended_kernel"] = true;
    if (in.has_conductivity) {
        py::array_t<double> a(static_cast<ptrdiff_t>(in.conductivity.size()));
        std::copy(in.conductivity.begin(), in.conductivity.end(), a.mutable_data());
        kw["wire_conductivity"] = a;
    }
    if (in.has_rlc) {
        py::object rlc_type = py::module_::import("momwire._wire_loading").attr("DistributedRLC");
        py::list rlc;
        for (const auto &e : in.rlc) {
            if (e.kind == 0)
                rlc.append(py::none());
            else
                rlc.append(rlc_type(e.kind == 1 ? "series" : "parallel", e.r, e.l, e.c));
        }
        kw["distributed_rlc"] = rlc;
    }
    if (in.has_ground) kw["ground_z"] = in.ground_z;
    if (in.ground_eps_given) {
        kw["ground_eps"] = py::make_tuple(in.ground_eps_r, in.ground_sigma);
        kw["ground_model"] = "sommerfeld";
    }
    kw["exact_kernel"] = "auto";
    return kw;
}

py::dict nec5_bspline_seam(py::bytes latin1) {
    const std::string text = latin1;
    mw::nec5::SeamResult r;
    {
        py::gil_scoped_release release;
        r = mw::nec5::bspline_seam(text);
    }
    static const char *route[] = {"served", "refused", "raised", "declined"};
    static const char *run[] = {"prepared", "shared", "refused", "raised", "declined"};
    py::dict out;
    out["route"] = route[r.route];
    out["message"] = r.message;
    out["error_kind"] = r.error_kind;
    py::list structures;
    for (const auto &so : r.structures) {
        py::list runs;
        for (const auto &o : so.runs) {
            py::dict x;
            x["kind"] = run[o.kind];
            x["message"] = o.message;
            x["error_kind"] = o.error_kind;
            x["prepared"] = o.prepared_index;
            runs.append(x);
        }
        structures.append(runs);
    }
    out["structures"] = structures;
    py::list prepared;
    for (const auto &p : r.prepared) {
        py::dict pd;
        pd["kwargs"] = bspline_kwargs(p);
        pd["mesh"] = seam_mesh(p);
        prepared.append(pd);
    }
    out["prepared"] = prepared;
    return out;
}

// PROTOTYPE (momwire#1517 step 1): the bspline front half's geometry for a
// deck's preparations (`_bspline_front_core.h`), test-facing.
py::list bspline_front_geometry(py::bytes latin1, uintptr_t syrk_addr,
                                std::vector<uintptr_t> blas,
                                std::vector<std::vector<double>> samples) {
    const std::string text = latin1;
    const mw::nec5::SeamResult r = mw::nec5::bspline_seam(text);
    py::list out;
    for (const auto &p : r.prepared) {
        const mw::nec5::BSplineFrontInputs &in = mw::nec5::bspline_part(p).inputs;
        mw::bspline::WireArgs wa;
        for (size_t w = 0; w < in.n_per_edge.size(); w++) {
            wa.polylines.emplace_back(in.points.begin() + 6 * w, in.points.begin() + 6 * w + 6);
            wa.n_per_edge.push_back({in.n_per_edge[w]});
        }
        wa.radius = in.radius;
        wa.radius_per_wire = in.radius_per_wire;
        wa.has_conductivity = in.has_conductivity;
        wa.conductivity = in.conductivity;
        wa.has_rlc = in.has_rlc;
        for (const auto &e : in.rlc) wa.rlc.push_back({e.kind, e.r, e.l, e.c});
        for (const auto &f : in.feeds) wa.feeds.push_back({f.wire, f.arc});
        for (const auto &g : in.node_gaps) wa.node_gaps.push_back({g.wire, g.end == mw::nec5::End::start});
        wa.junctions_given = !in.junctions.empty();
        for (const auto &g : in.junctions) {
            std::vector<std::pair<int64_t, bool>> mm;
            for (const auto &e : g) mm.push_back({e.first, e.second == mw::nec5::End::start});
            wa.junctions.push_back(mm);
        }
        py::dict entry;
        entry["kwargs"] = bspline_kwargs(p);
        mw::bspline::WireDedup dd;
        try {
            dd = mw::bspline::drop_duplicated_wires(wa);
            mw::bspline::check_ports(wa);
        } catch (const mw::bspline::FrontValueError &e) {
            entry["raised"] = py::make_tuple("ValueError", e.what());
            out.append(entry);
            continue;
        } catch (const mw::bspline::FrontNotPorted &e) {
            entry["declined"] = e.what();
            out.append(entry);
            continue;
        }
        if (dd.active) {
            py::list kept, removed;
            for (int64_t w : dd.kept) kept.append(w);
            for (const auto &r : dd.removed) removed.append(py::make_tuple(r.copy, r.twin, r.reversed));
            entry["dedup"] = py::make_tuple(dd.n_input, py::tuple(kept), py::tuple(removed));
        } else {
            entry["dedup"] = py::none();
        }
        const mw::bspline::Geometry g = mw::bspline::build_geometry(wa.polylines, wa.n_per_edge);
        auto arr = [](const std::vector<double> &v, bool rows3) {
            const ptrdiff_t n = static_cast<ptrdiff_t>(v.size());
            py::array_t<double> a = rows3 ? py::array_t<double>({n / 3, static_cast<ptrdiff_t>(3)})
                                          : py::array_t<double>(n);
            std::copy(v.begin(), v.end(), a.mutable_data());
            return a;
        };
        py::dict d;
        py::list per_wire;
        for (const auto &wg : g.per_wire) {
            py::dict x;
            x["seg_l"] = arr(wg.seg_l, true);
            x["seg_r"] = arr(wg.seg_r, true);
            x["tangents"] = arr(wg.tangents, true);
            x["h_per_seg"] = arr(wg.h_per_seg, false);
            py::list eo, ea;
            for (int64_t v : wg.edge_offsets) eo.append(v);
            for (const auto &e : wg.edge_arc_edges) ea.append(arr(e, false));
            x["edge_offsets"] = eo;
            x["edge_arc_edges"] = ea;
            x["arc_at_knot"] = arr(wg.arc_at_knot, false);
            x["n_total"] = wg.n_total;
            per_wire.append(x);
        }
        py::list so;
        for (int64_t v : g.seg_offsets) so.append(v);
        d["per_wire"] = per_wire;
        d["seg_offsets"] = so;
        d["n_segs_total"] = g.n_segs_total;
        d["h_per_seg"] = arr(g.h_per_seg, false);
        d["tangents"] = arr(g.tangents, true);
        d["seg_l"] = arr(g.seg_l, true);
        d["seg_r"] = arr(g.seg_r, true);
        entry["geom"] = d;
        try {
            std::vector<double> radii = wa.radius;
            if (!wa.radius_per_wire) radii.assign(wa.polylines.size(), wa.radius[0]);
            const mw::bspline::Basis bs = mw::bspline::build_basis(
                g, wa.polylines, wa.junctions, radii, in.has_ground, in.ground_z, 2);
            const ptrdiff_t nb = static_cast<ptrdiff_t>(bs.n_basis), W = 3;
            py::array_t<int64_t> ss({nb, W});
            std::copy(bs.supp_seg.begin(), bs.supp_seg.end(), ss.mutable_data());
            py::array_t<double> pp({nb, W, W});
            std::copy(bs.polys.begin(), bs.polys.end(), pp.mutable_data());
            py::array_t<double> ka({static_cast<ptrdiff_t>(bs.n_kcl), nb});
            std::copy(bs.kcl_A.begin(), bs.kcl_A.end(), ka.mutable_data());
            py::list knots;
            for (const auto &k : bs.wire_knots) knots.append(arr(k, false));
            entry["basis"] = py::make_tuple(ss, pp, ka, knots);
            const std::vector<double> gc = mw::bspline::node_gap_columns(wa, bs);
            py::array_t<double> gcol({nb, static_cast<ptrdiff_t>(wa.node_gaps.size())});
            std::copy(gc.begin(), gc.end(), gcol.mutable_data());
            entry["node_gap_cols"] = gcol;
            py::list js, ngs, fs;
            for (const auto &g2 : wa.junctions) {
                py::list mm;
                for (const auto &m : g2) mm.append(py::make_tuple(m.first, m.second ? "start" : "end"));
                js.append(mm);
            }
            for (const auto &g2 : wa.node_gaps)
                ngs.append(py::make_tuple(g2.wire, g2.at_start ? "start" : "end", cd(0.0, 0.0)));
            for (const auto &f : wa.feeds) fs.append(py::make_tuple(f.wire, f.arc, cd(0.0, 0.0)));
            entry["spec"] = py::make_tuple(js, ngs, fs);
            const std::vector<double> B = mw::bspline::port_columns(wa, g, bs);
            const ptrdiff_t n_p = static_cast<ptrdiff_t>(wa.feeds.size() + wa.node_gaps.size());
            py::array_t<cd> Bc({nb, n_p});
            for (size_t q = 0; q < B.size(); q++) Bc.mutable_data()[q] = cd(B[q], 0.0);
            entry["port_cols"] = Bc;
            // the free-space fill (proto): `_build_J_blocks(geom, k)`
            if (!in.has_ground) {
                mw::bspline::FillParams fp;
                const double c = 1.0 / std::sqrt(8.8541878188e-12 * 1.25663706127e-6);
                const double freq = c / in.wavelength;
                const double omega = (2.0 * 3.141592653589793) * freq;
                fp.k = omega / c;
                fp.degree = 2;
                fp.n_qp_pair = 8;
                fp.ladder = mw::bspline::default_ladder(8);
                fp.n_qp_same_edge = 4;
                fp.extended_kernel = in.extended_kernel;
                const std::vector<std::complex<double>> J = mw::bspline::build_J_blocks(g, radii, fp);
                py::array_t<cd> Ja({static_cast<ptrdiff_t>(3), static_cast<ptrdiff_t>(3),
                                    static_cast<ptrdiff_t>(g.n_segs_total),
                                    static_cast<ptrdiff_t>(g.n_segs_total)});
                std::copy(J.begin(), J.end(), Ja.mutable_data());
                entry["J"] = Ja;
                if (syrk_addr != 0) {
                    const std::vector<std::complex<double>> Z = mw::bspline::assemble_Z(
                        J, bs, g, omega, 8.8541878188e-12, 1.25663706127e-6,
                        reinterpret_cast<mw::bspline::dsyrk_fn>(syrk_addr));
                    py::array_t<cd> Za({nb, nb});
                    std::copy(Z.begin(), Z.end(), Za.mutable_data());
                    entry["Z"] = Za;
                    if (blas.size() == 4) {
                        mw::nec5::NumpyLinalg npl;
                        npl.zgemm = reinterpret_cast<mw::nec5::cblas_zgemm_fn>(blas[0]);
                        npl.zgemv = reinterpret_cast<mw::nec5::cblas_zgemv_fn>(blas[1]);
                        npl.zdotu_sub = reinterpret_cast<mw::nec5::cblas_zdotu_sub_fn>(blas[2]);
                        npl.zgesv = reinterpret_cast<mw::nec5::zgesv_fn>(blas[3]);
                        const int64_t n_p = static_cast<int64_t>(wa.feeds.size() + wa.node_gaps.size());
                        const mw::bspline::PortSolve ps = mw::bspline::solve_ports(
                            Z, bs.n_basis, B, n_p, bs.kcl_A, bs.n_kcl, scipy_lapack(), npl);
                        entry["solve_info"] = ps.info;
                        if (ps.info == 0) {
                            py::array_t<cd> Xa({nb, static_cast<ptrdiff_t>(n_p)});
                            for (int64_t i = 0; i < bs.n_basis; i++)
                                for (int64_t j = 0; j < n_p; j++)
                                    Xa.mutable_data()[i * n_p + j] =
                                        ps.X[static_cast<size_t>(ps.x_c_order ? i * n_p + j : j * bs.n_basis + i)];
                            py::array_t<cd> Ya({static_cast<ptrdiff_t>(n_p), static_cast<ptrdiff_t>(n_p)});
                            std::copy(ps.Y.begin(), ps.Y.end(), Ya.mutable_data());
                            entry["X"] = Xa;
                            entry["Y"] = Ya;
                            // the readouts on coeffs = X[:, 0]
                            std::vector<std::complex<double>> c0(static_cast<size_t>(bs.n_basis));
                            for (int64_t m = 0; m < bs.n_basis; m++) c0[static_cast<size_t>(m)] = ps.X[static_cast<size_t>(ps.x_c_order ? m * n_p : m)];
                            auto lists = [](const std::vector<std::vector<std::complex<double>>> &v) {
                                py::list l;
                                for (const auto &a : v) {
                                    py::array_t<cd> x(static_cast<ptrdiff_t>(a.size()));
                                    std::copy(a.begin(), a.end(), x.mutable_data());
                                    l.append(x);
                                }
                                return l;
                            };
                            entry["knot_currents"] = lists(mw::bspline::spline_readout(bs, g, dd, c0, nullptr, false));
                            if (!samples.empty()) {
                                entry["sample_currents"] = lists(mw::bspline::spline_readout(bs, g, dd, c0, &samples, false));
                                entry["sample_slopes"] = lists(mw::bspline::spline_readout(bs, g, dd, c0, &samples, true));
                            }
                        }
                    }
                }
            }
        } catch (const mw::bspline::FrontValueError &e) {
            entry["basis_raised"] = py::make_tuple("ValueError", e.what());
        } catch (const mw::bspline::FrontNotPorted &e) {
            entry["basis_declined"] = e.what();
        }
        out.append(entry);
    }
    return out;
}

}  // namespace

void register_razor_solve(py::module_ &m) {
    m.attr("razor_solve_proto") = true;
    py::class_<mw::razor::SolveContext>(m, "RazorSolveContext",
        "razor-2p's k-dependent solve in one C++ call (prototype): every row "
        "window of the fill (free space or a perfect ground's image fold) "
        "written into Z, then SciPy's own zgetrf/zgetrs. Holds the buffer "
        "pool reused across windows and solves, and counters.")
        .def(py::init<>())
        .def("port_solution", &port_solution, py::arg("cent"), py::arg("group_i"),
             py::arg("b_obs"), py::arg("seg_h"), py::arg("a"), py::arg("xg"),
             py::arg("wg"), py::arg("s_a"), py::arg("s_b"), py::arg("h_a"),
             py::arg("h_b"), py::arg("q_a"), py::arg("q_b"), py::arg("fall_a"),
             py::arg("fall_b"), py::arg("tans"), py::arg("wts"), py::arg("grounded"),
             py::arg("k"), py::arg("c_a_imag"), py::arg("c_phi_imag"), py::arg("real"),
             py::arg("image"), py::arg("window_rows"), py::arg("rhs"),
             py::arg("cancel_flag") = 0, py::arg("want_z") = false,
             "Fill Z and solve Z X = rhs. Returns (X, info, Z): X (n, n_rhs) "
             "column-major; info > 0 when U(info, info) is exactly zero (X "
             "is then the right-hand side, unsolved); Z the filled matrix "
             "before factorisation when want_z, else None.")
        .def("release", &mw::razor::SolveContext::release,
             "Free Z and the window pool; the counters are kept.")
        .def_property_readonly("counters", &counters)
        .def_property_readonly("n_basis", &mw::razor::SolveContext::n_basis)
        .def("front_port_solution", &front_port_solution, py::arg("kwargs"),
             py::arg("chunk_elems") = 2000000, py::arg("cancel_flag") = 0,
             py::arg("want_z") = false, py::arg("want_front") = false,
             "razor-2p's whole solve from RazorSolver's own keyword arguments: "
             "construction, geometry, the prepared tables, the port columns, "
             "the fill and the LU, all in C++ (`_razor_front_core.h`). Returns "
             "a dict: 'decline' names what the front half does not cover (and "
             "nothing else is set), else X (n_basis, n_ports) column-major, "
             "info (> 0: U(info, info) exactly zero), cols, geom "
             "(`_build_geometry`'s dict), ties, and Z / front on request.");

    m.attr("razor_front_proto") = true;
    m.def("razor_buried_stages", &razor_buried_stages, py::arg("kwargs"),
          py::arg("chunk_elems") = 2000000, py::arg("below_tables") = std::string(),
          "momwire#1512 step 3's gate: a buried solve's matrix at each stage the "
          "C++ fill reaches ('above', 'below', 'cross', 'final'), column-major.");
    m.def("razor_front_build", &razor_front_build, py::arg("kwargs"),
          py::arg("chunk_elems") = 2000000,
          "The front half alone (no fill, no LU): the same dict as "
          "RazorSolveContext.front_port_solution's, without X, with 'front'.");
    m.def("razor_front_format_numbers",
          [](std::vector<double> v) { return mw::razor::format_numbers(v); },
          "TEST HOOK: the core's `_wire_spec.format_numbers`.");
    m.def("razor_front_leggauss",
          [](int64_t n) -> py::object {
              std::vector<double> x, w;
              if (!mw::razor::leggauss(n, x, w)) return py::none();
              return py::make_tuple(np1(x), np1(w));
          },
          "TEST HOOK: the core's frozen Gauss-Legendre rule of order n (None "
          "beyond the table).");
    m.def("bspline_front_geometry", &bspline_front_geometry, py::arg("latin1"),
          py::arg("syrk_addr") = 0, py::arg("blas") = std::vector<uintptr_t>(),
          py::arg("samples") = std::vector<std::vector<double>>(),
          "PROTOTYPE HOOK (momwire#1517 step 1): per bspline preparation, its "
          "BSplineSolver keyword arguments and the C++ front half's geometry.");
    m.def("nec5_bspline_seam", &nec5_bspline_seam, py::arg("latin1"),
          "PROTOTYPE HOOK (momwire#1517 step 1): the NEC-5 seam's structure half "
          "for bspline in C++: the route, each run's outcome, and per "
          "preparation BSplineSolver's keyword arguments and the mesh.");
    m.attr("nec5_razor_seam_proto") = true;
    m.def("nec5_razor_seam", &nec5_razor_seam, py::arg("latin1"), py::arg("solve") = false,
          "TEST HOOK: the NEC-5 seam's structure half for razor-2p in C++ "
          "(`_nec5_razor_seam_core.h`): a deck's latin-1 bytes in; the route, "
          "each structure's and run's outcome, and per preparation the "
          "RazorSolver keyword arguments, the mesh, the port columns and, "
          "with solve, the solved coefficients X (n_basis, n_ports).");
    m.def("nec5_seam_constants", []() {
        py::dict d;
        d["speed_of_light_mhz_m"] = mw::nec5::SPEED_OF_LIGHT_MHZ_M;
        d["node_eps"] = mw::nec5::NODE_EPS;
        d["probe_volts"] = mw::nec5::PROBE_VOLTS;
        d["epsc_conductivity_factor"] = mw::nec5::EPSC_CONDUCTIVITY_FACTOR;
        return d;
    }, "TEST HOOK: the constants the C++ seam carries in place of Python's.");
    m.def("razor_front_constants", []() {
        py::dict d;
        d["eps"] = mw::razor::EPS;
        d["mu"] = mw::razor::MU;
        d["junction_tol"] = mw::razor::JUNCTION_TOL;
        d["ambiguity_tol_frac"] = mw::razor::AMBIGUITY_TOL_FRAC;
        d["exact_kernel_auto_h_over_a"] = mw::razor::EXACT_KERNEL_AUTO_H_OVER_A;
        d["chunk_elems"] = mw::razor::FrontInputs().chunk_elems;
        d["n_qp_source"] = mw::razor::FrontInputs().n_qp_source;
        return d;
    }, "TEST HOOK: the constants the front half carries in place of Python's.");
    // The front half's refusals, as the Python front half raises them.
    py::register_exception_translator([](std::exception_ptr p) {
        try {
            if (p) std::rethrow_exception(p);
        } catch (const mw::razor::FrontError &e) {
            PyErr_SetString(e.kind == mw::razor::FrontErrorKind::not_implemented
                                ? PyExc_NotImplementedError
                                : PyExc_ValueError,
                            e.what());
        }
    });
}
