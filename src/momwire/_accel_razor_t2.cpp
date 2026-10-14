#include "_accel_common.h"
#include "_mw_binding.h"
#include "_razor_core.h"

#include <stdexcept>
#include <string>

// momwire#1290's razor T2 rows and Q rows: the BINDING half
// (explore/core-binding-split). The kernels and the derivation of their bits
// live in `_razor_t2_core.cpp`, which includes no Python header.

using cd = std::complex<double>;
using mw::binding::in;
using mw::binding::opt;

// A complex128 2-D array taken AS IS, any strides: `out` is written in place,
// and pybind11 would answer a column-major window with a C-contiguous copy
// (momwire#1115's trap) if it were taken as an array_t. Refuses anything it
// cannot address rather than copying it.
static mw::view<cd, 2> view_of(const py::array &a, const char *what, bool write) {
    if (!py::isinstance<py::array_t<cd>>(a) || a.ndim() != 2)
        throw std::invalid_argument(std::string("razor_t2_rows: ") + what +
                                    " must be a 2-D complex128 array");
    if (write && !a.writeable())
        throw std::invalid_argument(std::string("razor_t2_rows: ") + what +
                                    " must be writeable: it is written in place");
    const ptrdiff_t shp[2] = {a.shape(0), a.shape(1)};
    const ptrdiff_t st[2] = {a.strides(0), a.strides(1)};
    return mw::view<cd, 2>(static_cast<cd *>(const_cast<void *>(a.data())), shp, st);
}

static void razor_t2_rows(
    py::array out, py::object t1_obj, py::array m0w, py::array m0k,
    py::array_t<int64_t, py::array::c_style | py::array::forcecast> plus,
    py::array_t<int64_t, py::array::c_style | py::array::forcecast> minus,
    py::array_t<int64_t, py::array::c_style | py::array::forcecast> s_a,
    py::array_t<int64_t, py::array::c_style | py::array::forcecast> s_b,
    py::array_t<double, py::array::c_style | py::array::forcecast> q_a,
    py::array_t<double, py::array::c_style | py::array::forcecast> q_b,
    double c_a_imag, double c_phi_imag) {
    const mw::view<cd, 2> o = view_of(out, "out", true);
    mw::view<const cd, 2> t;
    const bool fuse = !t1_obj.is_none();
    if (fuse) {
        mw::view<cd, 2> tt = view_of(t1_obj.cast<py::array>(), "t1", false);
        t = mw::view<const cd, 2>(tt.ptr, tt.n, tt.st);
    }
    mw::view<cd, 2> w = view_of(m0w, "M0w", false), k = view_of(m0k, "M0k", false);
    if (plus.ndim() != 1 || minus.ndim() != 1 || plus.shape(0) != o.shape(0) ||
        minus.shape(0) != o.shape(0))
        throw std::invalid_argument("razor_t2_rows: plus/minus need one entry per row");
    if (s_a.ndim() != 1 || s_b.ndim() != 1 || q_a.ndim() != 1 || q_b.ndim() != 1)
        throw std::invalid_argument("razor_t2_rows: per-basis arrays disagree on n_basis");
    auto vp = in<int64_t, 1>(plus), vm = in<int64_t, 1>(minus);
    auto vsa = in<int64_t, 1>(s_a), vsb = in<int64_t, 1>(s_b);
    auto vqa = in<double, 1>(q_a), vqb = in<double, 1>(q_b);
    py::gil_scoped_release nogil;
    mw::razor::t2_rows(o, fuse ? &t : nullptr,
                       mw::view<const cd, 2>(w.ptr, w.n, w.st),
                       mw::view<const cd, 2>(k.ptr, k.n, k.st), vp, vm, vsa, vsb,
                       vqa, vqb, c_a_imag, c_phi_imag);
}

static py::array_t<cd> razor_q_rows(
    py::array_t<cd, py::array::c_style | py::array::forcecast> f,
    py::array_t<int64_t, py::array::c_style | py::array::forcecast> obs_row,
    py::array_t<int64_t, py::array::c_style | py::array::forcecast> s_a,
    py::array_t<int64_t, py::array::c_style | py::array::forcecast> s_b,
    py::array_t<double, py::array::c_style | py::array::forcecast> h_a,
    py::array_t<double, py::array::c_style | py::array::forcecast> h_b,
    py::array_t<bool, py::array::c_style | py::array::forcecast> fall_a,
    py::array_t<bool, py::array::c_style | py::array::forcecast> fall_b,
    py::array_t<double, py::array::c_style | py::array::forcecast> sig_a,
    py::array_t<double, py::array::c_style | py::array::forcecast> sig_b,
    py::array_t<double, py::array::c_style | py::array::forcecast> wts,
    py::ssize_t n_path) {
    if (f.ndim() != 3 || f.shape(2) != 2)
        throw std::invalid_argument("razor_q_rows: f must be (n_u, n_seg, 2)");
    const py::ssize_t n_obs = obs_row.size();
    const py::ssize_t n_rows = (n_path >= 1 && n_obs % n_path == 0) ? n_obs / n_path : 0;
    py::array_t<cd> res(std::vector<py::ssize_t>{n_rows, s_a.size()});
    auto vf = in<cd, 3>(f);
    auto vo = mw::binding::out<cd, 2>(res);
    auto vrow = opt<int64_t>(obs_row), vsa = opt<int64_t>(s_a), vsb = opt<int64_t>(s_b);
    auto vha = opt<double>(h_a), vhb = opt<double>(h_b);
    auto vfa = opt<bool>(fall_a), vfb = opt<bool>(fall_b);
    auto vga = opt<double>(sig_a), vgb = opt<double>(sig_b), vw = opt<double>(wts);
    {
        py::gil_scoped_release nogil;
        mw::razor::q_rows(vf, vrow, vsa, vsb, vha, vhb, vfa, vfb, vga, vgb, vw, n_path, vo);
    }
    return res;
}

void register_razor_t2(py::module_ &m) {
    m.def("razor_q_rows", &razor_q_rows,
          "The composing ground's remainder rows Q of one razor row window: "
          "rem = f[obs_row[o], s, 1] / h (f[., s, 0] - that on a falling "
          "wing), Q[r, j] = sum over the row's n_path observers of "
          "(rem_a*sig_a + rem_b*sig_b)*w[o], in numpy's operation order.",
          py::arg("f"), py::arg("obs_row"), py::arg("s_a"), py::arg("s_b"),
          py::arg("h_a"), py::arg("h_b"), py::arg("fall_a"), py::arg("fall_b"),
          py::arg("sig_a"), py::arg("sig_b"), py::arg("wts"), py::arg("n_path"));
    m.attr("razor_q_rows_1290") = true;
    m.def("razor_t2_rows", &razor_t2_rows,
          "One razor row window's T2 and its final combination "
          "(momwire#1290). dM0[r] = tab[plus[r]] - tab[minus[r]] over the "
          "stacked table [M0w; M0k] (minus < 0: no subtraction), T2[r, j] = "
          "dM0[r, s_a[j]]*q_a[j] + dM0[r, s_b[j]]*q_b[j]. With t1 given, "
          "writes out = c_A*t1 - T2/c_Phi for c_A = i*c_a_imag and c_Phi = "
          "i*c_phi_imag (both purely imaginary, the case whose numpy "
          "arithmetic has an exact closed form); with t1 None, writes T2. "
          "`out` is written through its own strides.",
          py::arg("out"), py::arg("t1"), py::arg("M0w"), py::arg("M0k"),
          py::arg("plus"), py::arg("minus"), py::arg("s_a"), py::arg("s_b"),
          py::arg("q_a"), py::arg("q_b"), py::arg("c_a_imag"),
          py::arg("c_phi_imag"));
    m.attr("razor_t2_1290") = true;
}

