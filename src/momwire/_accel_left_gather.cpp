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
// The matrices (and the end loops' weights) must be REAL float64: they are
// taken without `forcecast`, so a complex basis's samples are refused with a
// TypeError rather than silently cast (the callers route those to numpy).
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

// The crossing end loops' matvecs, `_real_matvec_c(M, w * X[e])` for every
// end e at once, with X[e, j] read where it is stored: the tile block (by
// li, the held store at li < 0 by hp) or the row-ordered V/W store (by
// sidx). Per term numpy's `w * X` is (w + 0j)(Xr + iXi) = (w*Xr, w*Xi) up to
// the sign of an exact zero, and `_real_matvec_c` sums a * (w*X) over the
// row's stored entries from +0 in stored order, real and imaginary parts
// apart, then `re + 1j*im` -- which, the sums being finite and never -0,
// is (re, im). This is that loop, contraction off, as above.
static py::array_t<std::complex<double>> end_matvecs(
    py::array_t<int64_t, py::array::c_style | py::array::forcecast> indptr,
    py::array_t<int64_t, py::array::c_style | py::array::forcecast> indices,
    py::array_t<double, py::array::c_style> data,
    py::ssize_t n_out,
    py::array_t<double, py::array::c_style> w,
    py::array_t<int32_t, py::array::c_style | py::array::forcecast> li,
    py::array_t<int32_t, py::array::c_style | py::array::forcecast> hp,
    py::array tb, py::array held, int k,
    py::array_t<int64_t, py::array::c_style | py::array::forcecast> sidx,
    py::array store, int store_col, int n_threads) {
    const py::ssize_t n = w.size();
    const bool by_store = store_col >= 0;
    py::ssize_t E = 0;
    if (by_store) {
        if (sidx.ndim() != 2 || sidx.shape(1) != n)
            throw std::runtime_error("end_matvecs: sidx must be (E, n)");
        E = sidx.shape(0);
    } else {
        if (li.ndim() != 2 || li.shape(1) != n)
            throw std::runtime_error("end_matvecs: li must be (E, n)");
        E = li.shape(0);
    }
    const bool have_hp = hp.size() != 0;
    if (have_hp && (hp.ndim() != 2 || hp.shape(0) != E || hp.shape(1) != n))
        throw std::runtime_error("end_matvecs: hp must match li");
    if (indptr.size() != n_out + 1)
        throw std::runtime_error("end_matvecs: indptr length");
    const int64_t *ip = indptr.data();
    const int64_t *ix = indices.data();
    const double *dv = data.data();
    const int64_t nnz = ip[n_out];
    if (indices.size() < nnz || data.size() < nnz)
        throw std::runtime_error("end_matvecs: short CSR arrays");
    for (int64_t jj = 0; jj < nnz; ++jj)
        if (ix[jj] < 0 || ix[jj] >= n)
            throw std::runtime_error("end_matvecs: column out of range");
    const Table T = table_of(tb, "tb"), H = table_of(held, "held"),
                S = table_of(store, "store");
    const int32_t *L = by_store ? nullptr : li.data();
    const int32_t *Hp = have_hp ? hp.data() : nullptr;
    const int64_t *Si = by_store ? sidx.data() : nullptr;
    for (py::ssize_t q = 0; q < E * n; ++q) {
        if (by_store) {
            if (Si[q] < 0 || Si[q] >= S.rows)
                throw std::runtime_error("end_matvecs: store row");
        } else if (L[q] >= 0) {
            if (L[q] >= T.rows) throw std::runtime_error("end_matvecs: tile row");
        } else if (!have_hp || Hp[q] < 0 || Hp[q] >= H.rows) {
            throw std::runtime_error("end_matvecs: an end reads a row not in hand");
        }
    }
    const double *wp = w.data();
    py::array_t<std::complex<double>> out(std::vector<py::ssize_t>{E, n_out});
    double *Y = reinterpret_cast<double *>(out.mutable_data());
    {
        py::gil_scoped_release nogil;
        int nt = 1;
#ifdef _OPENMP
        nt = omp_get_max_threads();
        if (n_threads > 0) nt = std::min(nt, n_threads);
#endif
#pragma omp parallel for schedule(dynamic, 4) num_threads(nt)
        for (py::ssize_t e = 0; e < E; ++e) {
            const py::ssize_t base = e * n;
            double *y = Y + 2 * e * n_out;
            for (py::ssize_t i = 0; i < n_out; ++i) {
                double re = 0.0, im = 0.0;
                for (int64_t jj = ip[i]; jj < ip[i + 1]; ++jj) {
                    const int64_t j = ix[jj];
                    const py::ssize_t q = base + j;
                    const double *x;
                    if (by_store)
                        x = S.at(Si[q], store_col);
                    else
                        x = L[q] >= 0 ? T.at(L[q], k) : H.at(Hp[q], k);
                    const double xr = wp[j] * x[0];
                    const double xi = wp[j] * x[1];
                    re += dv[jj] * xr;
                    im += dv[jj] * xi;
                }
                y[2 * i] = re;
                y[2 * i + 1] = im;
            }
        }
    }
    return out;
}

// The main sandwich's five-term combine (`_crossing_fill._combine`) of the
// six left products L_i against the right weights, for the basis rows `J`
// of four CSR matrices Q (U.x, U.y, V/W, W/V: the six terms read Q1, Q2,
// Q3, Q4, Q3, Q4):
//
//   out[r, j] = L0 Q1[J_j]^T + L1 Q2^T + L2 Q3^T + L3 Q4^T + L4 Q3^T
//               - L5 Q4^T                                   (row r of L)
//
// with column c of the products read through `colmap`: L_i[r, colmap[c]]
// of `lnew` when colmap[c] >= 0, else of `lheld` at -1 - colmap[c] (the
// streamed sandwich's chunk and its held columns, momwire#1168), so neither
// the gathered (|rA|, |need|) products nor the sliced Q[J][:, need] are
// formed.
//
// THE BITS ARE THE NUMPY ROUTE'S on builds that do not contract. `L @ S.T`
// with a real CSR S is scipy's `csr_matvecs` over S's rows: each output
// (j, r) a running sum from +0 of (a + 0j) * L[r, col] over row j's stored
// entries in stored order, and column-slicing S keeps each row's entries and
// their order, so reading the full row through `colmap` reads the same terms
// in the same order. A term's parts are a*xr - 0*xi and a*xi + 0*xr there
// and a*xr, a*xi here: equal unless the product is a zero, where only a
// zero's sign can differ, and a running sum from +0 cannot hold -0, so no
// partial sum differs. The combine is numpy's elementwise left-to-right
// ((((T0 + T1) + T2) + T3) + T4) - T5, real and imaginary parts apart.
static py::array_t<std::complex<double>> combine_rows(
    std::vector<py::array_t<std::complex<double>, py::array::c_style>> lnew,
    std::vector<py::array_t<std::complex<double>, py::array::c_style>> lheld,
    py::array_t<int64_t, py::array::c_style | py::array::forcecast> colmap,
    py::array_t<int64_t, py::array::c_style | py::array::forcecast> J,
    std::vector<py::array_t<int64_t, py::array::c_style | py::array::forcecast>>
        indptrs,
    std::vector<py::array_t<int64_t, py::array::c_style | py::array::forcecast>>
        indicess,
    std::vector<py::array_t<double, py::array::c_style>> datas, int n_threads) {
    if (lnew.size() != 6 || (!lheld.empty() && lheld.size() != 6))
        throw std::runtime_error("combine_rows: six left products (and six held)");
    if (indptrs.size() != 4 || indicess.size() != 4 || datas.size() != 4)
        throw std::runtime_error("combine_rows: four matrices");
    if (lnew[0].ndim() != 2) throw std::runtime_error("combine_rows: L must be 2-D");
    const py::ssize_t rA = lnew[0].shape(0);
    const py::ssize_t n_new = lnew[0].shape(1);
    const py::ssize_t n_held = lheld.empty() ? 0 : lheld[0].shape(1);
    for (int i = 0; i < 6; ++i) {
        if (lnew[i].ndim() != 2 || lnew[i].shape(0) != rA || lnew[i].shape(1) != n_new)
            throw std::runtime_error("combine_rows: lnew shapes");
        if (!lheld.empty() &&
            (lheld[i].ndim() != 2 || lheld[i].shape(0) != rA ||
             lheld[i].shape(1) != n_held))
            throw std::runtime_error("combine_rows: lheld shapes");
    }
    const py::ssize_t n_cols = colmap.size();
    const int64_t *cm = colmap.data();
    Csr Q[4];
    py::ssize_t n_rows_q = -1;
    for (int m = 0; m < 4; ++m) {
        const py::ssize_t nr = indptrs[m].size() - 1;
        if (nr < 0 || (n_rows_q >= 0 && nr != n_rows_q))
            throw std::runtime_error("combine_rows: matrices of unequal height");
        n_rows_q = nr;
        Q[m] = Csr{indptrs[m].data(), indicess[m].data(), datas[m].data()};
    }
    const py::ssize_t nJ = J.size();
    const int64_t *Jp = J.data();
    // Every column a row in J stores must be in hand, in `lnew` or `lheld`
    // (the streamed route's AssertionError when it is not).
    for (py::ssize_t j = 0; j < nJ; ++j) {
        if (Jp[j] < 0 || Jp[j] >= n_rows_q)
            throw std::runtime_error("combine_rows: row out of range");
        for (int m = 0; m < 4; ++m)
            for (int64_t jj = Q[m].indptr[Jp[j]]; jj < Q[m].indptr[Jp[j] + 1]; ++jj) {
                if (jj >= indicess[m].size() || jj >= datas[m].size())
                    throw std::runtime_error("combine_rows: short CSR arrays");
                const int64_t c = Q[m].indices[jj];
                if (c < 0 || c >= n_cols)
                    throw std::runtime_error("combine_rows: column out of range");
                const int64_t at = cm[c];
                if (at >= n_new || (at < 0 && -1 - at >= n_held))
                    throw std::runtime_error(
                        "combine_rows: a row reads a column not in hand");
            }
    }
    const double *Ln[6], *Lh[6];
    for (int i = 0; i < 6; ++i) {
        Ln[i] = reinterpret_cast<const double *>(lnew[i].data());
        Lh[i] = lheld.empty() ? nullptr
                              : reinterpret_cast<const double *>(lheld[i].data());
    }
    py::array_t<std::complex<double>> out(std::vector<py::ssize_t>{rA, nJ});
    double *Y = reinterpret_cast<double *>(out.mutable_data());
    {
        py::gil_scoped_release nogil;
        int nt = 1;
#ifdef _OPENMP
        nt = omp_get_max_threads();
        if (n_threads > 0) nt = std::min(nt, n_threads);
#endif
        static const int qsel[6] = {0, 1, 2, 3, 2, 3};
        // Rows of L are independent: each output row is written by one
        // thread, its terms in stored order, so the thread count cannot move
        // a bit.
#pragma omp parallel for schedule(dynamic, 8) num_threads(nt)
        for (py::ssize_t r = 0; r < rA; ++r) {
            double *y = Y + 2 * r * nJ;
            for (py::ssize_t j = 0; j < nJ; ++j) {
                double tr[6], ti[6];
                for (int i = 0; i < 6; ++i) {
                    const Csr &M = Q[qsel[i]];
                    double sr = 0.0, si = 0.0;
                    for (int64_t jj = M.indptr[Jp[j]]; jj < M.indptr[Jp[j] + 1]; ++jj) {
                        const double a = M.data[jj];
                        const int64_t c = cm[M.indices[jj]];
                        const double *x = c >= 0 ? Ln[i] + 2 * (r * n_new + c)
                                                 : Lh[i] + 2 * (r * n_held + (-1 - c));
                        sr += a * x[0];
                        si += a * x[1];
                    }
                    tr[i] = sr;
                    ti[i] = si;
                }
                y[2 * j] = ((((tr[0] + tr[1]) + tr[2]) + tr[3]) + tr[4]) - tr[5];
                y[2 * j + 1] = ((((ti[0] + ti[1]) + ti[2]) + ti[3]) + ti[4]) - ti[5];
            }
        }
    }
    return out;
}

}  // namespace left_gather

void register_left_gather(py::module_ &m) {
    m.def("combine_rows", &left_gather::combine_rows,
          "The crossing main sandwich's five-term combine of six left "
          "products against rows J of four CSR matrices (Q1, Q2, Q3, Q4, Q3, "
          "Q4), product column c read at colmap[c] of lnew (>= 0) or at "
          "-1 - colmap[c] of lheld. scipy's CSR product order from zero, "
          "numpy's combine order; OpenMP over L's rows. Returns (rA, |J|) "
          "complex. momwire#1290.",
          py::arg("lnew"), py::arg("lheld"), py::arg("colmap"), py::arg("J"),
          py::arg("indptrs"), py::arg("indices"), py::arg("data"),
          py::arg("n_threads"));
    m.attr("combine_rows_1290") = true;
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
    m.def("end_matvecs", &left_gather::end_matvecs,
          "`_real_matvec_c(M, w * X[e])` for every end e: M an (n_out, n) "
          "CSR, X[e, j] = tb[li[e, j], k] (held[hp[e, j], k] where li < 0), "
          "or store[sidx[e, j], store_col] when store_col >= 0. Returns "
          "(E, n_out) complex; OpenMP over ends. momwire#1224.",
          py::arg("indptr"), py::arg("indices"), py::arg("data"),
          py::arg("n_out"), py::arg("w"), py::arg("li"), py::arg("hp"),
          py::arg("tb"), py::arg("held"), py::arg("k"), py::arg("sidx"),
          py::arg("store"), py::arg("store_col"), py::arg("n_threads"));
    m.attr("left_gather_1224") = true;
    m.attr("end_matvecs_1224") = true;
}
