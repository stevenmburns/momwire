#include "_accel_common.h"

#include <stdexcept>
#ifdef _OPENMP
#include <omp.h>
#endif

// momwire#1905 step 1: the point-matched sinusoidal fill's Φ → Z reduction,
// `out = Φ₀@M₀; out += Φ₁@M₁; out += Φ₂@M₂` over a band of observer rows
// (`sinusoidal._reduce_phi_band`), threaded over those rows.
//
// The scipy route it replaces ran on one thread: per tile of observer rows,
// `Mᵀ @ Φᵀ` is scipy's `csr_matvecs`, which for each output column j starts
// from a zeroed result and walks column j's stored entries of M in CSR order
// doing `y += a · x` (scipy's `complex_wrapper` multiply, then the two real
// adds) — 0.44 s of a 0.79 s fill on the 4,392-segment ELT whip at any thread
// count.
//
// THE BITS ARE THAT ROUTE'S, by construction. Each output element here is its
// own `0.0 + p₀ + p₁ + ...` over its column's stored entries in stored order,
// each `p = a · x` the unfused
//     (a.re·x.re − a.im·x.im,  a.re·x.im + a.im·x.re)
// that `complex_wrapper::operator*` spells (a = M's entry, x = Φ's), then the
// three products summed `(A + B) + C` elementwise, as `R = A; R += B; R += C`
// did. Nothing reassociates: there is no reduction ACROSS elements, and an
// element's chain is never split. `-ffp-contract=off` (setup.py) keeps every
// a·b + c two roundings, as scipy's x86 build has them; MSVC's /fp:fast is
// overridden below as in `_accel_row_scatter.cpp`. The exe's copy of this sum
// is `axpy1` in `_sinusoidal_fill_core.cpp`, the same order.
//
// Threading: one output ROW per iteration, and a row reads only its own Φ
// rows, so rows are independent and the thread count moves nothing. A row's
// sums go to a per-thread buffer first and are copied into `out` after, so
// `out` may be `Φ₀` itself (`_reduce_phi_rows_in_place` writes Z into Φ₀'s
// storage): the row's Φ₀ entries are all read before any is overwritten.
// Gated bit for bit against the scipy route (tests/test_sin_phi_reduce_1905.py).
#if defined(_MSC_VER)
#pragma float_control(precise, on)
#pragma fp_contract(off)
#endif

namespace phi_reduce {

using cd = std::complex<double>;
using I64 = py::array_t<int64_t, py::array::c_style | py::array::forcecast>;

struct Csc {
    const int64_t *ptr;
    const int64_t *ind;
    const double *val;  // interleaved (re, im)
};

static void reduce_phi_band(py::array out, py::array phi0, py::array phi1, py::array phi2,
                            I64 ptr0, I64 ind0, py::array val0, I64 ptr1, I64 ind1,
                            py::array val1, I64 ptr2, I64 ind2, py::array val2,
                            int n_threads) {
    const auto cplx = py::dtype::of<cd>();
    py::array phi[3] = {phi0, phi1, phi2};
    I64 ptr[3] = {ptr0, ptr1, ptr2}, ind[3] = {ind0, ind1, ind2};
    py::array val[3] = {val0, val1, val2};
    if (out.ndim() != 2 || !out.dtype().is(cplx))
        throw std::runtime_error("reduce_phi_band: out must be a 2-D complex128 array");
    if (!out.writeable()) throw std::runtime_error("reduce_phi_band: out is read-only");
    const py::ssize_t rows = out.shape(0), cols = out.shape(1);
    const py::ssize_t cs = static_cast<py::ssize_t>(sizeof(cd));
    if (cols > 1 && out.strides(1) != cs)
        throw std::runtime_error("reduce_phi_band: out rows must be contiguous");
    py::ssize_t n_src = -1;
    Csc M[3];
    const char *P[3];
    py::ssize_t ps[3];
    for (int c = 0; c < 3; c++) {
        const py::array &f = phi[c];
        if (f.ndim() != 2 || !f.dtype().is(cplx) || f.shape(0) != rows)
            throw std::runtime_error(
                "reduce_phi_band: each phi must be complex128 (rows of out, n_src)");
        if (n_src < 0) n_src = f.shape(1);
        if (f.shape(1) != n_src)
            throw std::runtime_error("reduce_phi_band: the three phi differ in width");
        if (n_src > 1 && f.strides(1) != cs)
            throw std::runtime_error("reduce_phi_band: phi rows must be contiguous");
        P[c] = static_cast<const char *>(f.data());
        ps[c] = f.strides(0);
        const py::array &v = val[c];
        if (v.ndim() != 1 || !v.dtype().is(cplx) || (v.size() > 1 && v.strides(0) != cs))
            throw std::runtime_error("reduce_phi_band: M data must be contiguous complex128");
        if (ptr[c].ndim() != 1 || ptr[c].size() != cols + 1 || ind[c].ndim() != 1 ||
            ind[c].size() != v.size())
            throw std::runtime_error(
                "reduce_phi_band: M must be CSC with out's columns (indptr cols + 1, "
                "indices as long as data)");
        const int64_t *p = ptr[c].data(), *ix = ind[c].data();
        if (p[0] < 0 || p[cols] > static_cast<int64_t>(v.size()))
            throw std::runtime_error("reduce_phi_band: indptr out of range");
        for (py::ssize_t j = 0; j < cols; j++)
            if (p[j + 1] < p[j]) throw std::runtime_error("reduce_phi_band: indptr not monotone");
        for (py::ssize_t e = 0; e < ind[c].size(); e++)
            if (ix[e] < 0 || ix[e] >= n_src)
                throw std::runtime_error("reduce_phi_band: index out of range");
        M[c] = Csc{p, ix, static_cast<const double *>(v.data())};
    }
    char *O = static_cast<char *>(out.mutable_data());
    const py::ssize_t os = out.strides(0);
    {
        py::gil_scoped_release nogil;
        // n_threads > 0 is taken as asked, past OMP_NUM_THREADS, so the
        // thread-count gate means something on an xdist worker (pinned to 1)
        int nt = 1;
#ifdef _OPENMP
        nt = n_threads > 0 ? n_threads : omp_get_max_threads();
#else
        (void)n_threads;
#endif
        if (nt < 1) nt = 1;
        std::vector<double> bufs(static_cast<size_t>(nt) * 2 * static_cast<size_t>(cols));
#pragma omp parallel for schedule(static) num_threads(nt)
        for (py::ssize_t r = 0; r < rows; ++r) {
            int t = 0;
#ifdef _OPENMP
            t = omp_get_thread_num();
#endif
            double *buf = bufs.data() + static_cast<size_t>(t) * 2 * static_cast<size_t>(cols);
            const double *x[3];
            for (int c = 0; c < 3; c++) x[c] = reinterpret_cast<const double *>(P[c] + r * ps[c]);
            for (py::ssize_t j = 0; j < cols; ++j) {
                double sr[3], si[3];
                for (int c = 0; c < 3; c++) {
                    // csr_matvecs: a zeroed y, then `y += a * x` per stored entry
                    double yr = 0.0, yi = 0.0;
                    const Csc &m = M[c];
                    for (int64_t e = m.ptr[j]; e < m.ptr[j + 1]; ++e) {
                        const double ar = m.val[2 * e], ai = m.val[2 * e + 1];
                        const double *xe = x[c] + 2 * m.ind[e];
                        const double xr = xe[0], xi = xe[1];
                        const double pr = ar * xr - ai * xi;
                        const double pi = ar * xi + ai * xr;
                        yr = yr + pr;
                        yi = yi + pi;
                    }
                    sr[c] = yr;
                    si[c] = yi;
                }
                buf[2 * j] = (sr[0] + sr[1]) + sr[2];
                buf[2 * j + 1] = (si[0] + si[1]) + si[2];
            }
            double *o = reinterpret_cast<double *>(O + r * os);
            for (py::ssize_t q = 0; q < 2 * cols; ++q) o[q] = buf[q];
        }
    }
}

}  // namespace phi_reduce

void register_phi_reduce(py::module_ &m) {
    m.def("reduce_phi_band", &phi_reduce::reduce_phi_band,
          "`out = Φ₀@M₀; out += Φ₁@M₁; out += Φ₂@M₂` for CSC Ms, to scipy's bit: "
          "each element its own zero plus its column's stored entries in order, "
          "one unfused complex multiply-add each, then (A + B) + C. out and the "
          "three phi are complex128 (rows, ·) with contiguous rows; out may be "
          "phi0 itself. OpenMP over out's rows; n_threads > 0 sets the team "
          "size, 0 takes OpenMP's default. momwire#1905.",
          py::arg("out"), py::arg("phi0"), py::arg("phi1"), py::arg("phi2"),
          py::arg("indptr0"), py::arg("indices0"), py::arg("data0"),
          py::arg("indptr1"), py::arg("indices1"), py::arg("data1"),
          py::arg("indptr2"), py::arg("indices2"), py::arg("data2"),
          py::arg("n_threads") = 0);
    m.attr("phi_reduce_1905") = true;
}
