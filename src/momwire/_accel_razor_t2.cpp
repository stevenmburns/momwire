#include "_accel_common.h"

#include <stdexcept>
#include <string>

// momwire#1290: razor's T2 rows and the block's final combination, fused.
//
// `RazorSolver._source_block_rows` finishes each row window as
//
//     rows = c_A * T1 - T2 / c_Phi,
//     T2   = dM0[:, s_a] * q_a + dM0[:, s_b] * q_b,
//     dM0  = M0w[plus] - M0w[minus]          (row-wise; no minus = no subtraction)
//
// numpy spells that with two (rows, n_seg) gathers and their difference, two
// (rows, n_basis) column gathers, and five more (rows, n_basis) temporaries,
// then copies the result into the window of a column-major Z. At the free
// 8-dipole array, N = 2816, that glue was ~0.28 s of a 1.27 s solve. Here
// each entry is read where it lies and written once.
//
// THE BITS ARE THE NUMPY ROUTE'S, BY DERIVATION, on any build that does not
// contract (every non-MSVC build of this TU passes -ffp-contract=off; MSVC
// gets the pragmas below):
//
//   * dM0 is one complex subtraction, componentwise, as numpy's.
//   * `dM0 * q` with q real is numpy's complex multiply by q + 0j:
//     (dr*q - di*0, dr*0 + di*q). The 0-products are exact zeros, so with or
//     without a fused multiply-add (numpy's AVX2 loop uses fmaddsub; its
//     baseline loop does not) each part is ONE rounding of dr*q, resp. di*q,
//     up to the sign of an exact zero. The sum of the two wings is a
//     componentwise add. So T2 is fixed by the operands on every platform.
//   * `fuse` (the free-space / above-ground case): c_A = i*a and c_Phi = i*b,
//     both with an EXACTLY zero real part. Then c_A * x is
//     (0*xr - a*xi, 0*xi + a*xr) = (-(a*xi), a*xr), one rounding each, with
//     or without fma, for the same reason. numpy's complex division is Smith's:
//     for |br| < |bi| it takes rat = br/bi = 0 and scl = 1/(bi + br*rat) =
//     1/bi, and returns ((xr*rat + xi)*scl, (xi*rat - xr)*scl) =
//     (xi*scl, (-xr)*scl): the products with rat are exact zeros, so a
//     contracting build (arm64 clang) gets the same numbers too. The
//     combination is therefore
//
//         re = -(a*t1i) - t2i*scl,   im = a*t1r + t2r*scl,
//
//     each operation rounded once, which is what is written below.
//   * Without `fuse` (a lossy medium's complex eps, so c_Phi has a real
//     part), Smith's division has no exact shortcut and a contracting numpy
//     build rounds it differently from a non-contracting one. The kernel then
//     writes T2 alone and the caller divides and combines in numpy, so that
//     half stays numpy's own arithmetic by construction.
//
// Signed zeros aside, the fused route is the numpy route bit for bit; the
// tests hold it to a bound derived from the operands rather than to `==`,
// because the bits of the numpy side belong to numpy's build, not to this one
// (tests/test_razor_t2_accel_1290.py).
#if defined(_MSC_VER)
#pragma float_control(precise, on)
#pragma fp_contract(off)
#endif

namespace razor_t2 {

using cd = std::complex<double>;

// A complex128 2-D array read or written through its own strides, in
// elements. Refuses anything it cannot address rather than copying it.
struct View {
    char *base = nullptr;
    py::ssize_t s0 = 0, s1 = 0, rows = 0, cols = 0;
    inline double *at(py::ssize_t r, py::ssize_t c) const {
        return reinterpret_cast<double *>(base + r * s0 + c * s1);
    }
};

static View view_of(const py::array &a, const char *what, bool write) {
    if (!py::isinstance<py::array_t<cd>>(a) || a.ndim() != 2)
        throw std::invalid_argument(std::string("razor_t2_rows: ") + what +
                                    " must be a 2-D complex128 array");
    if (write && !a.writeable())
        throw std::invalid_argument(std::string("razor_t2_rows: ") + what +
                                    " must be writeable: it is written in place");
    View v;
    v.base = static_cast<char *>(const_cast<void *>(a.data()));
    v.s0 = a.strides(0);
    v.s1 = a.strides(1);
    v.rows = a.shape(0);
    v.cols = a.shape(1);
    return v;
}

// `out` is written in place, through its strides: the caller hands the row
// window of a column-major Z, and pybind11 would answer that with a
// C-contiguous copy (momwire#1115's trap) if it were taken as an array_t.
// `t1` may be `out` itself (the numpy T1 branch builds it there); every entry
// is read before it is written, by the same iteration.
static void razor_t2_rows(
    py::array out,                                                          // (n_rows, n_basis)
    py::object t1_obj,                                                      // (n_rows, n_basis) or None
    py::array m0w,                                                          // (n_w, n_seg)
    py::array m0k,                                                          // (n_k, n_seg), n_k may be 0
    py::array_t<int64_t, py::array::c_style | py::array::forcecast> plus,   // (n_rows,)
    py::array_t<int64_t, py::array::c_style | py::array::forcecast> minus,  // (n_rows,), < 0: none
    py::array_t<int64_t, py::array::c_style | py::array::forcecast> s_a,    // (n_basis,)
    py::array_t<int64_t, py::array::c_style | py::array::forcecast> s_b,
    py::array_t<double, py::array::c_style | py::array::forcecast> q_a,
    py::array_t<double, py::array::c_style | py::array::forcecast> q_b,
    double c_a_imag,
    double c_phi_imag
) {
    const View o = view_of(out, "out", true);
    const bool fuse = !t1_obj.is_none();
    View t;
    if (fuse) {
        t = view_of(t1_obj.cast<py::array>(), "t1", false);
        if (t.rows != o.rows || t.cols != o.cols)
            throw std::invalid_argument("razor_t2_rows: t1 must have out's shape");
        if (!(c_phi_imag != 0.0))
            throw std::invalid_argument("razor_t2_rows: c_Phi must be nonzero");
    }
    const View w = view_of(m0w, "M0w", false);
    const View k = view_of(m0k, "M0k", false);
    const py::ssize_t n_rows = o.rows, n_basis = o.cols, n_seg = w.cols;
    const py::ssize_t n_w = w.rows, n_tab = w.rows + k.rows;
    if (k.rows > 0 && k.cols != n_seg)
        throw std::invalid_argument("razor_t2_rows: M0k must have M0w's columns");
    if (plus.ndim() != 1 || minus.ndim() != 1 || plus.shape(0) != n_rows ||
        minus.shape(0) != n_rows)
        throw std::invalid_argument("razor_t2_rows: plus/minus need one entry per row");
    if (s_a.ndim() != 1 || s_b.ndim() != 1 || q_a.ndim() != 1 || q_b.ndim() != 1 ||
        s_a.shape(0) != n_basis || s_b.shape(0) != n_basis ||
        q_a.shape(0) != n_basis || q_b.shape(0) != n_basis)
        throw std::invalid_argument("razor_t2_rows: per-basis arrays disagree on n_basis");
    const int64_t *pp = plus.data(), *pm = minus.data();
    const int64_t *sa = s_a.data(), *sb = s_b.data();
    const double *qa = q_a.data(), *qb = q_b.data();
    // Every index is checked HERE, before the GIL is released, so a caller
    // bug is an exception and not a read off the end of a table.
    for (py::ssize_t r = 0; r < n_rows; r++)
        if (pp[r] < 0 || pp[r] >= n_tab || pm[r] >= n_tab)
            throw std::invalid_argument("razor_t2_rows: row table index out of range");
    for (py::ssize_t j = 0; j < n_basis; j++)
        if (sa[j] < 0 || sa[j] >= n_seg || sb[j] < 0 || sb[j] >= n_seg)
            throw std::invalid_argument("razor_t2_rows: wing segment index out of range");

    // Row r of the stacked table [M0w; M0k], resolved once per row.
    std::vector<const char *> row_p(n_rows), row_m(n_rows);
    for (py::ssize_t r = 0; r < n_rows; r++) {
        row_p[r] = pp[r] < n_w ? w.base + pp[r] * w.s0 : k.base + (pp[r] - n_w) * k.s0;
        row_m[r] = pm[r] < 0 ? nullptr
                   : pm[r] < n_w ? w.base + pm[r] * w.s0
                                 : k.base + (pm[r] - n_w) * k.s0;
    }
    // The column stride of each row pointer's own table: the two tables need
    // not share a layout.
    std::vector<py::ssize_t> cs_p(n_rows), cs_m(n_rows);
    for (py::ssize_t r = 0; r < n_rows; r++) {
        cs_p[r] = pp[r] < n_w ? w.s1 : k.s1;
        cs_m[r] = (pm[r] >= 0 && pm[r] >= n_w) ? k.s1 : w.s1;
    }

    const double a = c_a_imag;
    const double scl = fuse ? 1.0 / c_phi_imag : 0.0;
    // Walk the window in `out`'s memory order: rows innermost when its rows
    // are the short stride (a window of the column-major Z), columns
    // innermost otherwise. Every entry is independent, so the order and the
    // thread count move no value.
    const bool rows_inner = std::llabs(static_cast<long long>(o.s0)) <
                            std::llabs(static_cast<long long>(o.s1));

    auto one = [&](py::ssize_t r, py::ssize_t j) {
        const char *P = row_p[r];
        const char *M = row_m[r];
        const double *pa = reinterpret_cast<const double *>(P + sa[j] * cs_p[r]);
        const double *pb = reinterpret_cast<const double *>(P + sb[j] * cs_p[r]);
        double dar = pa[0], dai = pa[1], dbr = pb[0], dbi = pb[1];
        if (M != nullptr) {
            const double *ma = reinterpret_cast<const double *>(M + sa[j] * cs_m[r]);
            const double *mb = reinterpret_cast<const double *>(M + sb[j] * cs_m[r]);
            dar = dar - ma[0];
            dai = dai - ma[1];
            dbr = dbr - mb[0];
            dbi = dbi - mb[1];
        }
        const double t2r = dar * qa[j] + dbr * qb[j];
        const double t2i = dai * qa[j] + dbi * qb[j];
        double *y = o.at(r, j);
        if (fuse) {
            const double *x = t.at(r, j);
            const double xr = x[0], xi = x[1];
            y[0] = -(a * xi) - t2i * scl;
            y[1] = a * xr + t2r * scl;
        } else {
            y[0] = t2r;
            y[1] = t2i;
        }
    };

    py::gil_scoped_release nogil;
    if (rows_inner) {
#pragma omp parallel for schedule(static)
        for (py::ssize_t j = 0; j < n_basis; j++)
            for (py::ssize_t r = 0; r < n_rows; r++) one(r, j);
    } else {
#pragma omp parallel for schedule(static)
        for (py::ssize_t r = 0; r < n_rows; r++)
            for (py::ssize_t j = 0; j < n_basis; j++) one(r, j);
    }
}

// The composing ground's remainder rows (`_source_block_rows`' Q), from the
// projected field moments `f` of the window's observers:
//
//     rem_a = f[o, s_a, 1] / h_a,   rem_a = f[o, s_a, 0] - rem_a  (falling wing)
//     rem_b = the same on s_b,
//     Q[r]  = sum over the r-th n_path observers o of (rem_a*sig_a + rem_b*sig_b)*w[o]
//
// with `f` row `obs_row[o]` for observer o (the window's distinct observers,
// `_potential_ground._distinct_observers`), so the gathered (n_obs, n_seg, 2)
// copy is never formed.
//
// THE BITS ARE THE NUMPY ROUTE'S: each operation is numpy's on the same
// operands, spelled as numpy's complex loops spell it with a real operand
// promoted to x + 0j. The division is Smith's with a zero imaginary part
// (rat = 0/h = +0, scl = 1/(h + 0*rat)), so each part is (x + y*rat)*scl;
// the multiplies by sig and by w are (ar*s - ai*0, ar*0 + ai*s). Every
// product with the zero is an exact zero, so a build that fuses them into an
// fma rounds the same numbers, and the terms are written in numpy's order,
// so the signs of zeros are its too. The path sum is numpy's reduction over
// a strided axis, sequential from its first element.
static py::array_t<cd> razor_q_rows(
    py::array_t<cd, py::array::c_style | py::array::forcecast> f,             // (n_u, n_seg, 2)
    py::array_t<int64_t, py::array::c_style | py::array::forcecast> obs_row,  // (n_obs,)
    py::array_t<int64_t, py::array::c_style | py::array::forcecast> s_a,      // (n_basis,)
    py::array_t<int64_t, py::array::c_style | py::array::forcecast> s_b,
    py::array_t<double, py::array::c_style | py::array::forcecast> h_a,
    py::array_t<double, py::array::c_style | py::array::forcecast> h_b,
    py::array_t<bool, py::array::c_style | py::array::forcecast> fall_a,
    py::array_t<bool, py::array::c_style | py::array::forcecast> fall_b,
    py::array_t<double, py::array::c_style | py::array::forcecast> sig_a,
    py::array_t<double, py::array::c_style | py::array::forcecast> sig_b,
    py::array_t<double, py::array::c_style | py::array::forcecast> wts,       // (n_obs,)
    py::ssize_t n_path
) {
    if (f.ndim() != 3 || f.shape(2) != 2)
        throw std::invalid_argument("razor_q_rows: f must be (n_u, n_seg, 2)");
    const py::ssize_t n_u = f.shape(0), n_seg = f.shape(1);
    const py::ssize_t n_obs = obs_row.size();
    const py::ssize_t n_basis = s_a.size();
    if (n_path < 1 || n_obs % n_path != 0)
        throw std::invalid_argument("razor_q_rows: n_obs must be whole paths");
    if (wts.size() != n_obs)
        throw std::invalid_argument("razor_q_rows: one weight per observer");
    if (s_b.size() != n_basis || h_a.size() != n_basis || h_b.size() != n_basis ||
        fall_a.size() != n_basis || fall_b.size() != n_basis ||
        sig_a.size() != n_basis || sig_b.size() != n_basis)
        throw std::invalid_argument("razor_q_rows: per-basis arrays disagree on n_basis");
    const int64_t *orow = obs_row.data();
    const int64_t *sa = s_a.data(), *sb = s_b.data();
    for (py::ssize_t o = 0; o < n_obs; o++)
        if (orow[o] < 0 || orow[o] >= n_u)
            throw std::invalid_argument("razor_q_rows: observer row out of range");
    for (py::ssize_t j = 0; j < n_basis; j++)
        if (sa[j] < 0 || sa[j] >= n_seg || sb[j] < 0 || sb[j] >= n_seg)
            throw std::invalid_argument("razor_q_rows: wing segment index out of range");
    const double *F = reinterpret_cast<const double *>(f.data());
    const double *ha = h_a.data(), *hb = h_b.data();
    const bool *fa = fall_a.data(), *fb = fall_b.data();
    const double *ga = sig_a.data(), *gb = sig_b.data(), *w = wts.data();
    const py::ssize_t n_rows = n_obs / n_path;
    py::array_t<cd> out(std::vector<py::ssize_t>{n_rows, n_basis});
    double *Y = reinterpret_cast<double *>(out.mutable_data());

    // One wing's rem: f[., s, 1] / h by Smith's with a zero imaginary part,
    // and on a falling wing f[., s, 0] minus that.
    auto wing = [&](const double *frow, int64_t s, double h, bool fall, double *r) {
        const double *m = frow + 4 * s;  // (n_seg, 2) complex: m[0..1] moment 0, m[2..3] moment 1
        const double rat = 0.0 / h;
        const double scl = 1.0 / (h + 0.0 * rat);
        double xr = (m[2] + m[3] * rat) * scl;
        double xi = (m[3] - m[2] * rat) * scl;
        if (fall) {
            xr = m[0] - xr;
            xi = m[1] - xi;
        }
        r[0] = xr;
        r[1] = xi;
    };

    py::gil_scoped_release nogil;
#pragma omp parallel for schedule(static)
    for (py::ssize_t r = 0; r < n_rows; r++) {
        for (py::ssize_t j = 0; j < n_basis; j++) {
            double accr = 0.0, acci = 0.0;
            for (py::ssize_t p = 0; p < n_path; p++) {
                const py::ssize_t o = r * n_path + p;
                const double *frow = F + 4 * n_seg * orow[o];
                double ra[2], rb[2];
                wing(frow, sa[j], ha[j], fa[j], ra);
                wing(frow, sb[j], hb[j], fb[j], rb);
                const double s1 = ga[j], s2 = gb[j];
                // rem_a*sig_a + rem_b*sig_b, each a complex multiply by s + 0j.
                const double ir = (ra[0] * s1 - ra[1] * 0.0) + (rb[0] * s2 - rb[1] * 0.0);
                const double ii = (ra[0] * 0.0 + ra[1] * s1) + (rb[0] * 0.0 + rb[1] * s2);
                // *= w[o], a complex multiply by w + 0j.
                const double vr = ir * w[o] - ii * 0.0;
                const double vi = ir * 0.0 + ii * w[o];
                if (p == 0) {
                    accr = vr;
                    acci = vi;
                } else {
                    accr = accr + vr;
                    acci = acci + vi;
                }
            }
            Y[2 * (r * n_basis + j)] = accr;
            Y[2 * (r * n_basis + j) + 1] = acci;
        }
    }
    return out;
}

}  // namespace razor_t2

void register_razor_t2(py::module_ &m) {
    m.def("razor_q_rows", &razor_t2::razor_q_rows,
          "The composing ground's remainder rows Q of one razor row window: "
          "rem = f[obs_row[o], s, 1] / h (f[., s, 0] - that on a falling "
          "wing), Q[r, j] = sum over the row's n_path observers of "
          "(rem_a*sig_a + rem_b*sig_b)*w[o], in numpy's operation order.",
          py::arg("f"), py::arg("obs_row"), py::arg("s_a"), py::arg("s_b"),
          py::arg("h_a"), py::arg("h_b"), py::arg("fall_a"), py::arg("fall_b"),
          py::arg("sig_a"), py::arg("sig_b"), py::arg("wts"), py::arg("n_path"));
    m.attr("razor_q_rows_1290") = true;
    m.def("razor_t2_rows", &razor_t2::razor_t2_rows,
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
