#include "_accel_common.h"

#include <stdexcept>
#ifdef _OPENMP
#include <omp.h>
#endif

// momwire#1916 part A: `_lapack.solve`'s preparation of A, in one threaded
// pass instead of three single-threaded ones.
//
// What the solve did before getrf, on the ELT whip's 4392 x 4392 complex Z
// (Rocketlake, 8 threads, per frequency): `np.isfinite(A).all()` 0.016 s, the
// C-to-Fortran copy 0.061 s, scipy's `lange("1", A)` 0.120 s — 0.20 s beside
// a 0.455 s getrf. All three read the same 300 MB; here they are one read.
//
// NOTHING HERE DECIDES A VALUE. The copy is a copy (bytes in, bytes out), so
// getrf and getrs see exactly the matrix they saw before. The finiteness flag
// raises the same ValueError. The 1-norm only feeds gecon, whose estimate
// decides the ill-conditioning warning; it is written as LAPACK's zlange /
// dlange spells it — per column j, `sum = 0; sum = sum + |a(i, j)|` for i
// ascending, then the maximum over columns — with |.| the C library's
// cabs / fabs that a gfortran-built LAPACK's ABS lowers to. Each column's
// chain is one thread's, in that order, and a maximum is exact, so the
// thread count moves nothing. It is gated equal, bit for bit, to scipy's
// lange (tests/test_lapack_solve_prep_1916.py).
//
// LAPACK's NaN rule (a NaN column sum wins the maximum) is never reached: a
// non-finite A raises before the norm is read.
#if defined(_MSC_VER)
#pragma float_control(precise, on)
#pragma fp_contract(off)
#endif

namespace solve_prep {

// Columns per thread task and rows per tile: a 64 x 64 complex tile is 64 KB
// read and 64 KB written, so the transpose stays in L2.
static constexpr py::ssize_t TILE = 64;

template <typename T>
static double mag(const T &v);

template <>
double mag<double>(const double &v) {
    return std::fabs(v);
}

template <>
double mag<std::complex<double>>(const std::complex<double> &v) {
    return std::abs(v);  // libstdc++ / MSVC: cabs (hypot) on the two parts
}

template <typename T>
static bool finite(const T &v);

template <>
bool finite<double>(const double &v) {
    return std::isfinite(v);
}

template <>
bool finite<std::complex<double>>(const std::complex<double> &v) {
    return std::isfinite(v.real()) && std::isfinite(v.imag());
}

template <typename T>
static py::tuple fortran_copy_1norm(py::array a, int n_threads) {
    if (a.ndim() != 2) throw std::runtime_error("fortran_copy_1norm: a must be 2-D");
    if (!(a.flags() & py::array::c_style))
        throw std::runtime_error("fortran_copy_1norm: a must be C-contiguous");
    const py::ssize_t m = a.shape(0), n = a.shape(1);
    py::array_t<T, py::array::f_style> out({m, n});
    const T *src = static_cast<const T *>(a.data());
    T *dst = out.mutable_data();
    std::vector<double> colsum(static_cast<size_t>(n), 0.0);
    int bad = 0;
    {
        py::gil_scoped_release nogil;
        int nt = 1;
#ifdef _OPENMP
        nt = n_threads > 0 ? n_threads : omp_get_max_threads();
#else
        (void)n_threads;
#endif
        if (nt < 1) nt = 1;
        const py::ssize_t n_strips = (n + TILE - 1) / TILE;
#pragma omp parallel for schedule(dynamic, 1) num_threads(nt) reduction(| : bad)
        for (py::ssize_t s = 0; s < n_strips; ++s) {
            const py::ssize_t j0 = s * TILE, j1 = std::min(n, j0 + TILE);
            double sums[TILE];
            for (py::ssize_t j = j0; j < j1; ++j) sums[j - j0] = 0.0;
            for (py::ssize_t i0 = 0; i0 < m; i0 += TILE) {
                const py::ssize_t i1 = std::min(m, i0 + TILE);
                for (py::ssize_t i = i0; i < i1; ++i) {
                    const T *row = src + i * n;
                    for (py::ssize_t j = j0; j < j1; ++j) {
                        const T v = row[j];
                        dst[j * m + i] = v;
                        if (!finite(v)) bad |= 1;
                        sums[j - j0] = sums[j - j0] + mag(v);
                    }
                }
            }
            for (py::ssize_t j = j0; j < j1; ++j) colsum[j] = sums[j - j0];
        }
    }
    double anorm = 0.0;
    for (py::ssize_t j = 0; j < n; ++j)
        if (anorm < colsum[j]) anorm = colsum[j];
    return py::make_tuple(out, bad == 0, anorm);
}

static py::tuple dispatch(py::array a, int n_threads) {
    if (a.dtype().is(py::dtype::of<std::complex<double>>()))
        return fortran_copy_1norm<std::complex<double>>(a, n_threads);
    if (a.dtype().is(py::dtype::of<double>())) return fortran_copy_1norm<double>(a, n_threads);
    throw std::runtime_error("fortran_copy_1norm: a must be complex128 or float64");
}

}  // namespace solve_prep

void register_solve_prep(py::module_ &m) {
    m.def("fortran_copy_1norm", &solve_prep::dispatch,
          "(F, finite, anorm) for a C-contiguous 2-D complex128 or float64 `a`: "
          "F its Fortran-ordered copy, finite whether every entry is finite, "
          "anorm its 1-norm as LAPACK's lange('1') sums it (each column in row "
          "order, then the maximum). One threaded pass; OpenMP over column "
          "strips, n_threads > 0 sets the team. momwire#1916.",
          py::arg("a"), py::arg("n_threads") = 0);
    m.attr("solve_prep_1916") = true;
}
