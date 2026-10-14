#include "_accel_common.h"
#include "_mw_binding.h"
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
        .def_property_readonly("n_basis", &mw::razor::SolveContext::n_basis);
}
