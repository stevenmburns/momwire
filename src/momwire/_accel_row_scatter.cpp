#include "_accel_common.h"

#include <stdexcept>
#ifdef _OPENMP
#include <omp.h>
#endif

// momwire#1290: SG's band scatter, `np.add.at(dest, idx, src)` along dest's
// first axis, read and written in place.
//
// `_ordered_row_scatter` reaches `np.add.at`'s order rank by rank with a
// buffered fancy-indexed `+=`, and what it still costs is memory traffic:
// each rank gathers its destination rows and its source rows into
// temporaries, adds, and scatters the result back. At N = 2816 on the
// free-space array that was 0.65 s of a 6.2 s solve for ~1.1 GB of source
// rows that need reading once. Here each destination row is visited once and
// each source row is read where it lives.
//
// THE BITS ARE `np.add.at`'s ON EVERY PLATFORM, by construction rather than
// by matching a library: `np.add.at` performs `dest[idx[e]] += src[e]` for e
// ascending, so a destination cell receives its entries' values one IEEE add
// at a time in ascending entry order, starting from what dest held. The loop
// below groups the entries by destination (a counting sort, stable, so each
// group is ascending) and, per row, performs exactly those adds in exactly
// that order. A complex add is the two real adds of its parts, so there is
// nothing to contract or reassociate: no multiply, and no reduction (the
// inner loop runs ACROSS a row's cells, each its own chain). Rows are
// independent, so threading over them moves nothing.
//
// `fresh` marks rows whose prior value is +0 without dest being read: the
// first add writes `0.0 + s`, which is what `np.zeros` then `+=` produced,
// signed zeros included (+0 + -0 is +0). That lets the caller hand over
// uninitialised rows (`np.empty`) instead of zeroing memory it is about to
// overwrite. MSVC builds with /fp:fast, under which `0.0 + s` may fold to
// `s`; the pragmas below restore the value-safe semantics, as in
// `_accel_left_gather.cpp`. Gated bit-for-bit against `np.add.at` and
// `_ordered_row_scatter` (tests/test_sg_row_scatter_accel_1290.py).
#if defined(_MSC_VER)
#pragma float_control(precise, on)
#pragma fp_contract(off)
#endif

namespace row_scatter {

static void ordered_row_scatter(
    py::array dest, py::array_t<int64_t, py::array::c_style | py::array::forcecast> idx,
    py::array src, py::array_t<bool, py::array::c_style | py::array::forcecast> fresh,
    int n_threads) {
    const auto cplx = py::dtype::of<std::complex<double>>();
    if (dest.ndim() != 2 || !dest.dtype().is(cplx) || src.ndim() != 2 ||
        !src.dtype().is(cplx))
        throw std::runtime_error(
            "ordered_row_scatter: dest and src must be 2-D complex128 arrays");
    if (!dest.writeable())
        throw std::runtime_error("ordered_row_scatter: dest is read-only");
    const py::ssize_t rows = dest.shape(0), w = dest.shape(1);
    const py::ssize_t ne = idx.size();
    if (idx.ndim() != 1 || src.shape(0) != ne || src.shape(1) != w)
        throw std::runtime_error(
            "ordered_row_scatter: idx must be 1-D and src (len(idx), dest width)");
    const py::ssize_t cs = static_cast<py::ssize_t>(sizeof(std::complex<double>));
    if (w > 1 && (dest.strides(1) != cs || src.strides(1) != cs))
        throw std::runtime_error(
            "ordered_row_scatter: rows must be contiguous (unit column stride)");
    const bool have_fresh = fresh.size() > 0;
    if (have_fresh && (fresh.ndim() != 1 || fresh.size() != rows))
        throw std::runtime_error("ordered_row_scatter: fresh must be (dest rows,)");

    const int64_t *ix = idx.data();
    for (py::ssize_t e = 0; e < ne; ++e)
        if (ix[e] < 0 || ix[e] >= rows)
            throw std::runtime_error("ordered_row_scatter: index out of range");

    // Entries grouped by destination row, ascending within each group.
    std::vector<int64_t> ptr(static_cast<size_t>(rows) + 1, 0);
    for (py::ssize_t e = 0; e < ne; ++e) ++ptr[static_cast<size_t>(ix[e]) + 1];
    for (py::ssize_t r = 0; r < rows; ++r) ptr[r + 1] += ptr[r];
    std::vector<int64_t> order(static_cast<size_t>(ne));
    {
        std::vector<int64_t> fill(ptr.begin(), ptr.end() - 1);
        for (py::ssize_t e = 0; e < ne; ++e) order[fill[ix[e]]++] = e;
    }

    const bool *fr = have_fresh ? fresh.data() : nullptr;
    char *D = static_cast<char *>(dest.mutable_data());
    const char *S = static_cast<const char *>(src.data());
    const py::ssize_t ds = dest.strides(0), ss = src.strides(0);
    const py::ssize_t n = 2 * w;  // doubles per row
    {
        py::gil_scoped_release nogil;
        int nt = 1;
#ifdef _OPENMP
        nt = omp_get_max_threads();
        if (n_threads > 0) nt = std::min(nt, n_threads);
#endif
#pragma omp parallel for schedule(dynamic, 1) num_threads(nt)
        for (py::ssize_t r = 0; r < rows; ++r) {
            double *d = reinterpret_cast<double *>(D + r * ds);
            int64_t k = ptr[r];
            const int64_t k1 = ptr[r + 1];
            if (fr != nullptr && fr[r]) {
                if (k == k1) {
                    for (py::ssize_t c = 0; c < n; ++c) d[c] = 0.0;
                    continue;
                }
                const double *s = reinterpret_cast<const double *>(S + order[k] * ss);
                for (py::ssize_t c = 0; c < n; ++c) d[c] = 0.0 + s[c];
                ++k;
            }
            for (; k < k1; ++k) {
                const double *s = reinterpret_cast<const double *>(S + order[k] * ss);
                for (py::ssize_t c = 0; c < n; ++c) d[c] += s[c];
            }
        }
    }
}

}  // namespace row_scatter

void register_row_scatter(py::module_ &m) {
    m.def("ordered_row_scatter", &row_scatter::ordered_row_scatter,
          "`np.add.at(dest, idx, src)` along dest's first axis, to the bit: "
          "each dest row receives its entries' rows one add at a time in "
          "ascending entry order. dest (rows, w) and src (len(idx), w) are "
          "complex128 with contiguous rows; rows where `fresh` is true are "
          "taken as +0 without being read. OpenMP over dest rows. momwire#1290.",
          py::arg("dest"), py::arg("idx"), py::arg("src"), py::arg("fresh"),
          py::arg("n_threads") = 0);
    m.attr("row_scatter_1290") = true;
}
