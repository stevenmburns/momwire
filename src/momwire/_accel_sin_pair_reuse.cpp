#include "_accel_common.h"

#include <stdexcept>
#ifdef _OPENMP
#include <omp.h>
#endif

// momwire#1916 part B: `pair_reuse`'s gather fused into the Φ → Z reduction
// (`_sin_pair_reuse.reduce_band`), so a band's per-pair Φ is never
// materialised: each observer row's three Φ rows are gathered by class index
// into a per-thread buffer, the near pairs written over them, and the row
// reduced from there — necpp's shape for its fill-pair reuse.
//
// What it replaces: `_sin_pair_reuse.scatter` (three `np.take`s into three
// band-sized Φ, then the near pairs) followed by `_acc.reduce_phi_band`; on
// the ELT whip at 8 threads (Rocketlake) 0.119 s + 0.078 s per frequency,
// the gather being memory traffic through three Φ the size of the band.
//
// THE BITS ARE THAT ROUTE'S, by construction. The gather is a copy:
// `np.take(rep[s], cls, mode="clip")` is the clamped class's value, and the
// near pairs then overwrite theirs, as `scatter` writes them after the take.
// Under the PEC image the image block is gathered the same way and
// subtracted, `Φ -= Φ_img`, one IEEE subtraction per part as numpy's complex
// `-=`. The reduction is `reduce_phi_band`'s (`_accel_phi_reduce.cpp`)
// written out again: per output element a zero plus its column's stored
// entries of M in stored order, one unfused multiply-add each, then
// `(A + B) + C`. Rows are independent, so the thread count moves nothing;
// LANES rows share each step of the walk over M, each its own chain.
// `-ffp-contract=off` (setup.py) and the MSVC pragmas below keep every
// product and sum its own rounding. Gated bit for bit against scatter +
// reduce_phi_band (tests/test_sin_pair_reuse_1916.py).
#if defined(_MSC_VER)
#pragma float_control(precise, on)
#pragma fp_contract(off)
#endif

namespace sin_pair_reuse {

using cd = std::complex<double>;
using I64 = py::array_t<int64_t, py::array::c_style | py::array::forcecast>;
using I32 = py::array_t<int32_t, py::array::c_style>;
using C128 = py::array_t<cd, py::array::c_style>;

struct Csc {
    const int64_t *ptr;
    const int64_t *ind;
    const double *val;  // interleaved (re, im)
};

// One block's class map over the band and where its values come from.
struct Block {
    const int32_t *cls = nullptr;  // (rows, n_src), row-major
    const double *rep = nullptr;   // (n_cls, 3) complex, interleaved
    int64_t n_cls = 0;
    const int64_t *near_idx = nullptr;  // (k,) sorted flat r * n_src + n, band-local
    const double *near_val = nullptr;   // (k, 3) complex
    int64_t n_near = 0;
};

static Block block_of(const I32 &cls, const C128 &rep, const I64 &near_idx,
                      const C128 &near_val, py::ssize_t rows, py::ssize_t n_src,
                      const char *what) {
    if (cls.ndim() != 2 || cls.shape(0) != rows || cls.shape(1) != n_src)
        throw std::runtime_error(std::string("reduce_band: ") + what +
                                 " classes must be int32 (rows of out, n_src)");
    if (rep.ndim() != 2 || rep.shape(1) != 3 || rep.shape(0) < 1)
        throw std::runtime_error(std::string("reduce_band: ") + what +
                                 " class values must be (n_cls >= 1, 3) complex128");
    if (near_idx.ndim() != 1 || near_val.ndim() != 2 || near_val.shape(1) != 3 ||
        near_val.shape(0) != near_idx.shape(0))
        throw std::runtime_error(std::string("reduce_band: ") + what +
                                 " near pairs must be (k,) int64 and (k, 3) complex128");
    Block b;
    b.cls = cls.data();
    b.rep = reinterpret_cast<const double *>(rep.data());
    b.n_cls = rep.shape(0);
    b.near_idx = near_idx.data();
    b.near_val = reinterpret_cast<const double *>(near_val.data());
    b.n_near = near_idx.shape(0);
    const int64_t total = static_cast<int64_t>(rows) * static_cast<int64_t>(n_src);
    for (int64_t e = 0; e < b.n_near; ++e) {
        if (b.near_idx[e] < 0 || b.near_idx[e] >= total)
            throw std::runtime_error("reduce_band: a near pair lies outside the band");
        if (e && b.near_idx[e] <= b.near_idx[e - 1])
            throw std::runtime_error("reduce_band: near pairs must be sorted and distinct");
    }
    return b;
}

// Observer rows reduced together: the reduction's every step is done for
// LANES rows at once on the same M entry, each row's chain exactly as
// `reduce_phi_band` runs it, so a lane is that kernel's arithmetic and the
// compiler may put the lanes in one vector register. That is the whole
// reason for the layout below; it moves no operation within a chain.
static constexpr int LANES = 4;

// Row r's three Φ rows into lane `l` of x[0..2], each laid out
// [source n][re, im][lane]: the take, then the near pairs over it.
static void gather_row(const Block &b, py::ssize_t r, py::ssize_t n_src, int l,
                       double *x[3]) {
    const int32_t *c = b.cls + r * n_src;
    const int64_t hi = b.n_cls - 1;
    for (py::ssize_t n = 0; n < n_src; ++n) {
        int64_t k = c[n];
        k = k < 0 ? 0 : (k > hi ? hi : k);  // np.take(mode="clip")
        const double *v = b.rep + 6 * k;
        const py::ssize_t at = 2 * LANES * n + l;
        x[0][at] = v[0];
        x[0][at + LANES] = v[1];
        x[1][at] = v[2];
        x[1][at + LANES] = v[3];
        x[2][at] = v[4];
        x[2][at + LANES] = v[5];
    }
    if (b.n_near == 0) return;
    const int64_t lo = static_cast<int64_t>(r) * n_src, up = lo + n_src;
    const int64_t *e0 = std::lower_bound(b.near_idx, b.near_idx + b.n_near, lo);
    const int64_t *e1 = std::lower_bound(e0, b.near_idx + b.n_near, up);
    for (const int64_t *e = e0; e < e1; ++e) {
        const py::ssize_t n = static_cast<py::ssize_t>(*e - lo);
        const double *v = b.near_val + 6 * (e - b.near_idx);
        const py::ssize_t at = 2 * LANES * n + l;
        x[0][at] = v[0];
        x[0][at + LANES] = v[1];
        x[1][at] = v[2];
        x[1][at + LANES] = v[3];
        x[2][at] = v[4];
        x[2][at + LANES] = v[5];
    }
}

static void reduce_band(py::array out, I32 cls, C128 rep, I64 near_idx, C128 near_val,
                        py::object image, I64 ptr0, I64 ind0, py::array val0, I64 ptr1,
                        I64 ind1, py::array val1, I64 ptr2, I64 ind2, py::array val2,
                        int n_threads) {
    const auto cplx = py::dtype::of<cd>();
    if (out.ndim() != 2 || !out.dtype().is(cplx))
        throw std::runtime_error("reduce_band: out must be a 2-D complex128 array");
    if (!out.writeable()) throw std::runtime_error("reduce_band: out is read-only");
    const py::ssize_t rows = out.shape(0), cols = out.shape(1);
    const py::ssize_t cs = static_cast<py::ssize_t>(sizeof(cd));
    if (cols > 1 && out.strides(1) != cs)
        throw std::runtime_error("reduce_band: out rows must be contiguous");
    const py::ssize_t n_src = cls.ndim() == 2 ? cls.shape(1) : -1;
    Block d = block_of(cls, rep, near_idx, near_val, rows, n_src, "the direct");
    Block im;
    bool has_image = !image.is_none();
    // Keep the image arrays alive (and their conversions owned) to the end.
    I32 icls;
    C128 irep, inval;
    I64 inidx;
    if (has_image) {
        auto t = image.cast<py::tuple>();
        if (t.size() != 4)
            throw std::runtime_error(
                "reduce_band: image must be (classes, values, near index, near values)");
        icls = t[0].cast<I32>();
        irep = t[1].cast<C128>();
        inidx = t[2].cast<I64>();
        inval = t[3].cast<C128>();
        im = block_of(icls, irep, inidx, inval, rows, n_src, "the image");
    }
    I64 ptr[3] = {ptr0, ptr1, ptr2}, ind[3] = {ind0, ind1, ind2};
    py::array val[3] = {val0, val1, val2};
    Csc M[3];
    for (int c = 0; c < 3; c++) {
        const py::array &v = val[c];
        if (v.ndim() != 1 || !v.dtype().is(cplx) || (v.size() > 1 && v.strides(0) != cs))
            throw std::runtime_error("reduce_band: M data must be contiguous complex128");
        if (ptr[c].ndim() != 1 || ptr[c].size() != cols + 1 || ind[c].ndim() != 1 ||
            ind[c].size() != v.size())
            throw std::runtime_error(
                "reduce_band: M must be CSC with out's columns (indptr cols + 1, "
                "indices as long as data)");
        const int64_t *p = ptr[c].data(), *ix = ind[c].data();
        if (p[0] < 0 || p[cols] > static_cast<int64_t>(v.size()))
            throw std::runtime_error("reduce_band: indptr out of range");
        for (py::ssize_t j = 0; j < cols; j++)
            if (p[j + 1] < p[j]) throw std::runtime_error("reduce_band: indptr not monotone");
        for (py::ssize_t e = 0; e < ind[c].size(); e++)
            if (ix[e] < 0 || ix[e] >= n_src)
                throw std::runtime_error("reduce_band: index out of range");
        M[c] = Csc{p, ix, static_cast<const double *>(v.data())};
    }
    char *O = static_cast<char *>(out.mutable_data());
    const py::ssize_t os = out.strides(0);
    {
        py::gil_scoped_release nogil;
        int nt = 1;
#ifdef _OPENMP
        nt = n_threads > 0 ? n_threads : omp_get_max_threads();
#else
        (void)n_threads;
#endif
        if (nt < 1) nt = 1;
        const size_t per = static_cast<size_t>(2 * LANES * n_src);
        const size_t blocks = has_image ? 6 : 3;
        std::vector<double> bufs(static_cast<size_t>(nt) * blocks * per);
        const py::ssize_t tiles = (rows + LANES - 1) / LANES;
#pragma omp parallel for schedule(static) num_threads(nt)
        for (py::ssize_t tile = 0; tile < tiles; ++tile) {
            int t = 0;
#ifdef _OPENMP
            t = omp_get_thread_num();
#endif
            const py::ssize_t r0 = tile * LANES;
            const int nl = static_cast<int>(std::min<py::ssize_t>(LANES, rows - r0));
            double *base = bufs.data() + static_cast<size_t>(t) * blocks * per;
            double *x[3] = {base, base + per, base + 2 * per};
            // A short last tile's idle lanes compute on zeros, never stored.
            if (nl < LANES) std::fill(base, base + blocks * per, 0.0);
            for (int l = 0; l < nl; ++l) gather_row(d, r0 + l, n_src, l, x);
            if (has_image) {
                double *y[3] = {base + 3 * per, base + 4 * per, base + 5 * per};
                for (int l = 0; l < nl; ++l) gather_row(im, r0 + l, n_src, l, y);
                // `Phi_c -= Phi_i[0]` (and the other two): per part.
                for (int c = 0; c < 3; c++)
                    for (size_t q = 0; q < per; ++q) x[c][q] = x[c][q] - y[c][q];
            }
            double *o[LANES];
            for (int l = 0; l < LANES; ++l)
                o[l] = reinterpret_cast<double *>(O + (r0 + std::min(l, nl - 1)) * os);
            for (py::ssize_t j = 0; j < cols; ++j) {
                double sr[3][LANES], si[3][LANES];
                for (int c = 0; c < 3; c++) {
                    // reduce_phi_band's chain per lane: a zeroed y, then
                    // `y += a * x` per stored entry, unfused.
                    double yr[LANES], yi[LANES];
                    for (int l = 0; l < LANES; ++l) yr[l] = 0.0, yi[l] = 0.0;
                    const Csc &m = M[c];
                    for (int64_t e = m.ptr[j]; e < m.ptr[j + 1]; ++e) {
                        const double ar = m.val[2 * e], ai = m.val[2 * e + 1];
                        const double *xr = x[c] + 2 * LANES * m.ind[e];
                        const double *xi = xr + LANES;
                        for (int l = 0; l < LANES; ++l) {
                            const double pr = ar * xr[l] - ai * xi[l];
                            const double pi = ar * xi[l] + ai * xr[l];
                            yr[l] = yr[l] + pr;
                            yi[l] = yi[l] + pi;
                        }
                    }
                    for (int l = 0; l < LANES; ++l) sr[c][l] = yr[l], si[c][l] = yi[l];
                }
                for (int l = 0; l < nl; ++l) {
                    o[l][2 * j] = (sr[0][l] + sr[1][l]) + sr[2][l];
                    o[l][2 * j + 1] = (si[0][l] + si[1][l]) + si[2][l];
                }
            }
        }
    }
}

}  // namespace sin_pair_reuse

void register_sin_pair_reuse(py::module_ &m) {
    m.def("pair_reuse_reduce_band", &sin_pair_reuse::reduce_band,
          "pair_reuse's gather fused into the Φ → Z reduction, to the bit of "
          "`_sin_pair_reuse.scatter` then `reduce_phi_band`: per observer row, "
          "Φ gathered from the class values (n_cls, 3) by the int32 class map "
          "(clamped, as np.take mode='clip'), the band's near pairs (sorted "
          "band-local flat index, (k, 3) values) written over it, the image "
          "block's likewise subtracted when `image` is (classes, values, near "
          "index, near values), then reduced against the three CSC Ms. OpenMP "
          "over out's rows; n_threads > 0 sets the team. momwire#1916.",
          py::arg("out"), py::arg("cls"), py::arg("rep"), py::arg("near_idx"),
          py::arg("near_val"), py::arg("image"), py::arg("indptr0"), py::arg("indices0"),
          py::arg("data0"), py::arg("indptr1"), py::arg("indices1"), py::arg("data1"),
          py::arg("indptr2"), py::arg("indices2"), py::arg("data2"),
          py::arg("n_threads") = 0);
    m.attr("pair_reuse_reduce_1916") = true;
}
