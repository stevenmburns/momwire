#include "_accel_common.h"
#include "_mw_binding.h"
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

const mw::razor::Blas &scipy_blas() {
    // SciPy's own ddot, the norm `np.linalg.norm` reaches (measured equal to
    // numpy's bundled OpenBLAS on Haswell, Skylake and an AVX-512
    // Rocketlake, where the tail of the kernel fuses into FMAs).
    static mw::razor::Blas b;
    if (b.ddot == nullptr) {
        py::dict capi = py::module_::import("scipy.linalg.cython_blas").attr("__pyx_capi__");
        b.ddot = reinterpret_cast<mw::razor::ddot_fn>(capsule_pointer(capi, "ddot"));
    }
    return b;
}

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
    in.wavelength = kw.contains("wavelength") ? as_float(kw["wavelength"]) : 22.0;
    in.extended_kernel = kw.contains("extended_kernel") && py::bool_(get("extended_kernel"));
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
std::pair<std::string, std::unique_ptr<mw::razor::Front>> build_front(
    const py::dict &kw, py::object xg, py::object wg, int64_t chunk_elems) {
    mw::razor::FrontInputs in;
    try {
        in = front_inputs(kw);
    } catch (Malformed &) {
        return {"malformed-args", nullptr};
    }
    if (!xg.is_none()) {
        auto x = py::array_t<double, py::array::c_style | py::array::forcecast>::ensure(xg);
        auto w = py::array_t<double, py::array::c_style | py::array::forcecast>::ensure(wg);
        in.xg.assign(x.data(), x.data() + x.size());
        in.wg.assign(w.data(), w.data() + w.size());
    }
    in.chunk_elems = chunk_elems;
    std::unique_ptr<mw::razor::Front> f(new mw::razor::Front());
    const mw::razor::Blas &blas = scipy_blas();  // with the GIL: it may import
    std::string why;
    {
        py::gil_scoped_release release;
        why = f->build(in, blas);
    }
    if (!why.empty()) return {why, nullptr};
    return {"", std::move(f)};
}

py::dict front_port_solution(mw::razor::SolveContext &ctx, py::dict kw, py::object xg,
                             py::object wg, int64_t chunk_elems, uintptr_t cancel_flag,
                             bool want_z, bool want_front) {
    py::dict out;
    auto built = build_front(kw, xg, wg, chunk_elems);
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
        ctx.fill(f.solve_inputs(cancel_flag));
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

py::dict razor_front_build(py::dict kw, py::object xg, py::object wg, int64_t chunk_elems) {
    py::dict out;
    auto built = build_front(kw, xg, wg, chunk_elems);
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
             py::arg("xg") = py::none(), py::arg("wg") = py::none(),
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
    m.def("razor_front_build", &razor_front_build, py::arg("kwargs"),
          py::arg("xg") = py::none(), py::arg("wg") = py::none(),
          py::arg("chunk_elems") = 2000000,
          "The front half alone (no fill, no LU): the same dict as "
          "RazorSolveContext.front_port_solution's, without X, with 'front'.");
    m.def("razor_front_numpy_str",
          [](std::vector<double> v) { return mw::razor::numpy_array_str(v); },
          "TEST HOOK: the core's spelling of numpy's str() of a 1-D float array.");
    m.def("razor_front_constants", []() {
        py::dict d;
        d["eps"] = mw::razor::EPS;
        d["mu"] = mw::razor::MU;
        d["junction_tol"] = mw::razor::JUNCTION_TOL;
        d["ambiguity_tol_frac"] = mw::razor::AMBIGUITY_TOL_FRAC;
        d["exact_kernel_auto_h_over_a"] = mw::razor::EXACT_KERNEL_AUTO_H_OVER_A;
        d["chunk_elems"] = mw::razor::FrontInputs().chunk_elems;
        d["n_qp_source"] = mw::razor::FrontInputs().n_qp_source;
        d["leggauss12_x"] = std::vector<double>(mw::razor::LEGGAUSS12_X, mw::razor::LEGGAUSS12_X + 12);
        d["leggauss12_w"] = std::vector<double>(mw::razor::LEGGAUSS12_W, mw::razor::LEGGAUSS12_W + 12);
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
