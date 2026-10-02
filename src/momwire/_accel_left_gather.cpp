#include "_accel_common.h"

#include <cstring>
#include <stdexcept>
#ifdef _OPENMP
#include <omp.h>
#endif

// momwire#1224: the crossing main sandwich's six left products, formed
// straight from the product tiles' evaluated values.
//
// `_crossing_fill._left_products` is six sparse @ dense products,
// (P1 U, P2 U, P3 (k2 V + dz'W), P3 W, P4 W, P4 V), over table columns that
// `_ProductTiles._gather` first copies out of the tile block, the held store
// and the V/W store, one (nA, |cols|) complex array per kernel. At razor's
// inverted-L x8 the gathers and the products were ~0.5 s of a 3 s crossing
// fill, most of it moving the gathered tables through memory. Here each
// table entry is read where it lives, when a product needs it.
//
// The matrices must be REAL float64: they are taken without `forcecast`,
// so a complex basis's samples are refused with a TypeError rather than
// silently cast (the callers route those to numpy).
//
// THE BITS ARE THE NUMPY ROUTE'S, on builds that do not contract (every
// non-MSVC build of this TU passes -ffp-contract=off; MSVC gets the pragmas
// below). scipy's CSR @ dense product forms output (i, c) from zero as
// `y += a_jj * x[j, c]` over row i's stored entries in stored order; with a
// real matrix and a complex table that is (a*xr - 0*xi, a*xi + 0*xr), which
// for finite x is (a*xr, a*xi) up to the sign of an exact zero -- and a
// signed zero cannot survive a sum that starts from +0 unless every term is
// zero. So the loop below, the same entries in the same order from +0, one
// rounding per multiply and per add, is that product. The combined kernel
// `k2 V + dz'W` is numpy's real-scalar complex multiply-add, which is
// (k2*Vr + dr, k2*Vi + di) on the same terms (k2 + 0j is the scalar). Gated
// bit-for-bit against the numpy route (tests/test_left_gather_1224.py);
// never pinned across builds (scipy's own build decides whether ITS loop
// contracts).
#if defined(_MSC_VER)
#pragma float_control(precise, on)
#pragma fp_contract(off)
#endif

namespace left_gather {

struct Csr {
    const int64_t *indptr;
    const int64_t *indices;
    const double *data;
};

// A complex (rows, kernels) table read through its strides.
struct Table {
    const char *base = nullptr;
    py::ssize_t s0 = 0, s1 = 0;
    py::ssize_t rows = 0, cols = 0;
    inline const double *at(int64_t r, int k) const {
        return reinterpret_cast<const double *>(base + r * s0 + k * s1);
    }
};

static Table table_of(const py::array &a, const char *what) {
    Table t;
    if (a.size() == 0) return t;
    if (a.ndim() != 2 || !a.dtype().is(py::dtype::of<std::complex<double>>()))
        throw std::runtime_error(std::string("left_products_gathered: ") + what +
                                 " must be a 2-D complex128 array");
    t.base = static_cast<const char *>(a.data());
    t.s0 = a.strides(0);
    t.s1 = a.strides(1);
    t.rows = a.shape(0);
    t.cols = a.shape(1);
    return t;
}

static py::tuple left_products_gathered(
    std::vector<py::array_t<int64_t, py::array::c_style | py::array::forcecast>>
        indptrs,
    std::vector<py::array_t<int64_t, py::array::c_style | py::array::forcecast>>
        indicess,
    std::vector<py::array_t<double, py::array::c_style>>
        datas,
    py::ssize_t n_out, double k2sq,
    py::array_t<int32_t, py::array::c_style | py::array::forcecast> li,
    py::array_t<int32_t, py::array::c_style | py::array::forcecast> hp,
    py::array tb, py::array held, int kU, int kV, int kW, int kdz,
    py::array_t<int64_t, py::array::c_style | py::array::forcecast> sidx,
    py::array store, int n_threads,
    std::vector<py::array_t<std::complex<double>,
                            py::array::c_style | py::array::forcecast>>
        dense) {
    if (indptrs.size() != 4 || indicess.size() != 4 || datas.size() != 4)
        throw std::runtime_error("left_products_gathered: four matrices");
    // DENSE mode: the four kernels' (nA, nc) tables themselves, (U, V, W,
    // dz'W) -- the whole-table and grid routes' dict -- read by (a, c), so
    // every route's left products are this loop's (momwire#1224).
    const bool by_dense = !dense.empty();
    if (by_dense) {
        if (dense.size() != 4 || dense[0].ndim() != 2)
            throw std::runtime_error("left_products_gathered: four dense tables");
        for (int m = 1; m < 4; ++m)
            if (dense[m].ndim() != 2 || dense[m].shape(0) != dense[0].shape(0) ||
                dense[m].shape(1) != dense[0].shape(1))
                throw std::runtime_error("left_products_gathered: dense shapes");
    } else if (li.ndim() != 2) {
        throw std::runtime_error("left_products_gathered: li must be (nA, nc)");
    }
    const py::ssize_t nA = by_dense ? dense[0].shape(0) : li.shape(0);
    const py::ssize_t nc = by_dense ? dense[0].shape(1) : li.shape(1);
    const bool have_hp = hp.size() != 0;
    if (have_hp && (hp.ndim() != 2 || hp.shape(0) != nA || hp.shape(1) != nc))
        throw std::runtime_error("left_products_gathered: hp must match li");
    const bool vw_store = !by_dense && kV < 0;
    if (vw_store && (sidx.ndim() != 2 || sidx.shape(0) != nA || sidx.shape(1) != nc))
        throw std::runtime_error("left_products_gathered: sidx must match li");
    Csr P[4];
    for (int m = 0; m < 4; ++m) {
        if (indptrs[m].size() != n_out + 1)
            throw std::runtime_error("left_products_gathered: indptr length");
        P[m] = Csr{indptrs[m].data(), indicess[m].data(), datas[m].data()};
        const int64_t nnz = P[m].indptr[n_out];
        if (indicess[m].size() < nnz || datas[m].size() < nnz)
            throw std::runtime_error("left_products_gathered: short CSR arrays");
        for (int64_t jj = 0; jj < nnz; ++jj)
            if (P[m].indices[jj] < 0 || P[m].indices[jj] >= nA)
                throw std::runtime_error("left_products_gathered: column out of range");
    }
    const Table T = table_of(tb, "tb"), H = table_of(held, "held"),
                S = table_of(store, "store");
    const int32_t *L = by_dense ? nullptr : li.data();
    const int32_t *Hp = have_hp ? hp.data() : nullptr;
    const int64_t *Si = vw_store ? sidx.data() : nullptr;
    const double *D[4] = {nullptr, nullptr, nullptr, nullptr};
    if (by_dense)
        for (int m = 0; m < 4; ++m)
            D[m] = reinterpret_cast<const double *>(dense[m].data());
    // Every table entry the products can read must be in hand: a row in
    // the tile, or a held slot (the numpy route's AssertionError).
    for (py::ssize_t q = 0; !by_dense && q < nA * nc; ++q) {
        const int32_t l = L[q];
        if (l >= 0) {
            if (l >= T.rows) throw std::runtime_error("left_products_gathered: tile row");
        } else if (!have_hp || Hp[q] < 0 || Hp[q] >= H.rows) {
            throw std::runtime_error(
                "left_products_gathered: a ready column reads a row not in hand");
        }
        if (vw_store && (Si[q] < 0 || Si[q] >= S.rows))
            throw std::runtime_error("left_products_gathered: store row");
    }

    std::vector<py::array_t<std::complex<double>>> outs;
    for (int m = 0; m < 6; ++m) {
        outs.emplace_back(std::vector<py::ssize_t>{n_out, nc});
    }
    double *Y[6];
    for (int m = 0; m < 6; ++m) Y[m] = reinterpret_cast<double *>(outs[m].mutable_data());

    {
        py::gil_scoped_release nogil;
        // The table entry (a, c) of kernel k: the tile's row, or the held
        // store's; V and W from the row-ordered store when the tiles keep one.
        auto tile_val = [&](py::ssize_t q, int k) -> const double * {
            if (by_dense) return D[k] + 2 * q;
            const int32_t l = L[q];
            return l >= 0 ? T.at(l, k) : H.at(Hp[q], k);
        };
        auto vw_val = [&](py::ssize_t q, int k, int store_col) -> const double * {
            return vw_store ? S.at(Si[q], store_col) : tile_val(q, k);
        };
        int nt = 1;
#ifdef _OPENMP
        nt = omp_get_max_threads();
        if (n_threads > 0) nt = std::min(nt, n_threads);
#endif
        // Rows are independent: each output row is written by one thread,
        // its terms in the CSR's stored order, so the thread count cannot
        // move a bit.
#pragma omp parallel for schedule(dynamic, 8) num_threads(nt)
        for (py::ssize_t i = 0; i < n_out; ++i) {
            for (int m = 0; m < 6; ++m)
                std::memset(Y[m] + 2 * i * nc, 0, sizeof(double) * 2 * nc);
            // (P1, U) -> 0, (P2, U) -> 1, (P3, k2 V + dz'W) -> 2, (P3, W) -> 3
            for (int pm = 0; pm < 3; ++pm) {
                const Csr &M = P[pm];
                for (int64_t jj = M.indptr[i]; jj < M.indptr[i + 1]; ++jj) {
                    const double a = M.data[jj];
                    const py::ssize_t row = M.indices[jj] * nc;
                    if (pm < 2) {
                        double *y = Y[pm] + 2 * i * nc;
                        for (py::ssize_t c = 0; c < nc; ++c) {
                            const double *u = tile_val(row + c, kU);
                            y[2 * c] += a * u[0];
                            y[2 * c + 1] += a * u[1];
                        }
                    } else {
                        double *y2 = Y[2] + 2 * i * nc;
                        double *y3 = Y[3] + 2 * i * nc;
                        for (py::ssize_t c = 0; c < nc; ++c) {
                            const double *v = vw_val(row + c, kV, 0);
                            const double *d = tile_val(row + c, kdz);
                            const double xr = k2sq * v[0] + d[0];
                            const double xi = k2sq * v[1] + d[1];
                            y2[2 * c] += a * xr;
                            y2[2 * c + 1] += a * xi;
                            const double *w = vw_val(row + c, kW, 1);
                            y3[2 * c] += a * w[0];
                            y3[2 * c + 1] += a * w[1];
                        }
                    }
                }
            }
            // (P4, W) -> 4, (P4, V) -> 5
            const Csr &M = P[3];
            for (int64_t jj = M.indptr[i]; jj < M.indptr[i + 1]; ++jj) {
                const double a = M.data[jj];
                const py::ssize_t row = M.indices[jj] * nc;
                double *y4 = Y[4] + 2 * i * nc;
                double *y5 = Y[5] + 2 * i * nc;
                for (py::ssize_t c = 0; c < nc; ++c) {
                    const double *w = vw_val(row + c, kW, 1);
                    y4[2 * c] += a * w[0];
                    y4[2 * c + 1] += a * w[1];
                    const double *v = vw_val(row + c, kV, 0);
                    y5[2 * c] += a * v[0];
                    y5[2 * c + 1] += a * v[1];
                }
            }
        }
    }
    py::tuple res(6);
    for (int m = 0; m < 6; ++m) res[m] = outs[m];
    return res;
}

}  // namespace left_gather

void register_left_gather(py::module_ &m) {
    m.def("left_products_gathered", &left_gather::left_products_gathered,
          "The crossing main sandwich's six left products (P1 U, P2 U, "
          "P3 (k2 V + dz'W), P3 W, P4 W, P4 V) of four (n_out, nA) CSR "
          "matrices against table columns read in place: entry (a, c) of "
          "kernel k is tb[li[a, c], k], or held[hp[a, c], k] where li < 0; V "
          "and W come from store[sidx[a, c], 0/1] when kV < 0. scipy's "
          "CSR product order, from zero; OpenMP over output rows. momwire#1224.",
          py::arg("indptrs"), py::arg("indices"), py::arg("data"),
          py::arg("n_out"), py::arg("k2sq"), py::arg("li"), py::arg("hp"),
          py::arg("tb"), py::arg("held"), py::arg("kU"), py::arg("kV"),
          py::arg("kW"), py::arg("kdz"), py::arg("sidx"), py::arg("store"),
          py::arg("n_threads"),
          py::arg("dense") = std::vector<py::array_t<std::complex<double>,
                                                       py::array::c_style |
                                                           py::array::forcecast>>());
    m.attr("left_gather_1224") = true;
}
