#include "_accel_common.h"

#include <complex>
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

// `end_matvecs` over a tile's product ROWS (momwire#1335): X[e, j] is the
// tile block's kernel k at loc[R[e, j]], or the held store's at
// hpos[R[e, j]] where loc < 0 -- the gather `_crossing_fill._tile_matvecs`
// made in numpy before calling `end_matvecs`, done here per element. The
// sums are `end_matvecs`' loop, operand for operand: each output entry from
// 0.0, the row's stored entries in order, (w * x) then data * that, added.
template <class RI>
static py::array_t<std::complex<double>> end_matvecs_rows_t(
    const int64_t *ip, const int64_t *ix, const double *dv, py::ssize_t n_out,
    const double *wp, py::ssize_t n, const RI *Rp, py::ssize_t E,
    const int32_t *lp, py::ssize_t n_rows, const int32_t *hq, const Table &T,
    const Table &H, int k, int n_threads) {
    for (py::ssize_t q = 0; q < E * n; ++q) {
        const int64_t r = static_cast<int64_t>(Rp[q]);
        if (r < 0 || r >= n_rows)
            throw std::runtime_error("end_matvecs_rows: product row out of range");
        const int32_t l = lp[r];
        if (l >= 0) {
            if (l >= T.rows) throw std::runtime_error("end_matvecs_rows: tile row");
        } else if (hq == nullptr || hq[r] < 0 || hq[r] >= H.rows) {
            throw std::runtime_error("end_matvecs_rows: an end reads a row not in hand");
        }
    }
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
                    const int64_t r = static_cast<int64_t>(Rp[base + j]);
                    const int32_t l = lp[r];
                    const double *x = l >= 0 ? T.at(l, k) : H.at(hq[r], k);
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

static py::array_t<std::complex<double>> end_matvecs_rows(
    py::array_t<int64_t, py::array::c_style | py::array::forcecast> indptr,
    py::array_t<int64_t, py::array::c_style | py::array::forcecast> indices,
    py::array_t<double, py::array::c_style> data, py::ssize_t n_out,
    py::array_t<double, py::array::c_style> w, py::array R,
    py::array_t<int32_t, py::array::c_style> loc,
    py::array_t<int32_t, py::array::c_style> hpos, py::array tb, py::array held,
    int k, int n_threads) {
    const py::ssize_t n = w.size();
    if (R.ndim() != 2 || R.shape(1) != n)
        throw std::runtime_error("end_matvecs_rows: R must be (E, n)");
    if (!(R.flags() & py::array::c_style))
        throw std::runtime_error("end_matvecs_rows: R must be C-contiguous");
    const py::ssize_t E = R.shape(0);
    if (indptr.size() != n_out + 1)
        throw std::runtime_error("end_matvecs_rows: indptr length");
    const int64_t *ip = indptr.data();
    const int64_t *ix = indices.data();
    const double *dv = data.data();
    const int64_t nnz = ip[n_out];
    if (indices.size() < nnz || data.size() < nnz)
        throw std::runtime_error("end_matvecs_rows: short CSR arrays");
    for (int64_t jj = 0; jj < nnz; ++jj)
        if (ix[jj] < 0 || ix[jj] >= n)
            throw std::runtime_error("end_matvecs_rows: column out of range");
    const py::ssize_t n_rows = loc.size();
    const bool have_h = hpos.size() != 0;
    if (have_h && hpos.size() != n_rows)
        throw std::runtime_error("end_matvecs_rows: hpos must match loc");
    const Table T = table_of(tb, "tb"), H = table_of(held, "held");
    const int32_t *lp = loc.data();
    const int32_t *hq = have_h ? hpos.data() : nullptr;
    const double *wp = w.data();
    if (R.dtype().is(py::dtype::of<int32_t>()))
        return end_matvecs_rows_t<int32_t>(ip, ix, dv, n_out, wp, n,
                                           static_cast<const int32_t *>(R.data()), E, lp,
                                           n_rows, hq, T, H, k, n_threads);
    if (R.dtype().is(py::dtype::of<int64_t>()))
        return end_matvecs_rows_t<int64_t>(ip, ix, dv, n_out, wp, n,
                                           static_cast<const int64_t *>(R.data()), E, lp,
                                           n_rows, hq, T, H, k, n_threads);
    throw std::runtime_error("end_matvecs_rows: R must be int32 or int64");
}

// `end_matvecs_rows` with the rows never formed (momwire#1335): row
// R[e, j] = rowflat[base_e[e] + base_j[j] + T[rsel[e], csel[j]]] -- the
// fused grouped ends' rows (group g's z row against the needed line nodes'
// local keys: base_e = off[g] + zl * nk[g], T = kl_rank, rsel = g, csel =
// the nodes) and the line ends' (each grouped node's z row at the end's key
// in its group: base_j = off[grank] + zl_rank * nk[grank], T = the ends'
// key table, rsel = the end, csel = grank). Each x is then read as
// `end_matvecs_rows` reads it, and summed by the same loop. Every index is
// checked inside the loop and a bad one refused after it.
template <class TI>
static py::array_t<std::complex<double>> end_matvecs_table_t(
    const int64_t *ip, const int64_t *ix, const double *dv, py::ssize_t n_out,
    const double *wp, py::ssize_t n, py::ssize_t E, const int64_t *be,
    const int64_t *bj, const TI *Tp, py::ssize_t t_rows, py::ssize_t t_cols,
    const int64_t *rs, const int64_t *cs, const int32_t *rf, int64_t n_flat,
    const int32_t *lp, py::ssize_t n_rows, const int32_t *hq, const Table &T,
    const Table &H, int k, int n_threads) {
    for (py::ssize_t e = 0; e < E; ++e)
        if (rs[e] < 0 || rs[e] >= t_rows)
            throw std::runtime_error("end_matvecs_table: row selector out of range");
    for (py::ssize_t j = 0; j < n; ++j)
        if (cs[j] < 0 || cs[j] >= t_cols)
            throw std::runtime_error("end_matvecs_table: column selector out of range");
    py::array_t<std::complex<double>> out(std::vector<py::ssize_t>{E, n_out});
    double *Y = reinterpret_cast<double *>(out.mutable_data());
    int bad = 0;
    {
        py::gil_scoped_release nogil;
        int nt = 1;
#ifdef _OPENMP
        nt = omp_get_max_threads();
        if (n_threads > 0) nt = std::min(nt, n_threads);
#endif
#pragma omp parallel for schedule(dynamic, 4) num_threads(nt) reduction(|:bad)
        for (py::ssize_t e = 0; e < E; ++e) {
            double *y = Y + 2 * e * n_out;
            const TI *trow = Tp + rs[e] * t_cols;
            const int64_t b0 = be[e];
            for (py::ssize_t i = 0; i < n_out; ++i) {
                double re = 0.0, im = 0.0;
                for (int64_t jj = ip[i]; jj < ip[i + 1]; ++jj) {
                    const int64_t j = ix[jj];
                    const int64_t f = b0 + bj[j] + static_cast<int64_t>(trow[cs[j]]);
                    if (f < 0 || f >= n_flat) {
                        bad |= 1;
                        continue;
                    }
                    const int64_t r = rf[f];
                    if (r < 0 || r >= n_rows) {
                        bad |= 1;
                        continue;
                    }
                    const int32_t l = lp[r];
                    const double *x;
                    if (l >= 0 && l < T.rows) {
                        x = T.at(l, k);
                    } else if (l < 0 && hq != nullptr && hq[r] >= 0 && hq[r] < H.rows) {
                        x = H.at(hq[r], k);
                    } else {
                        bad |= 2;
                        continue;
                    }
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
    if (bad & 1) throw std::runtime_error("end_matvecs_table: index out of range");
    if (bad & 2)
        throw std::runtime_error("end_matvecs_table: an end reads a row not in hand");
    return out;
}

static py::array_t<std::complex<double>> end_matvecs_table(
    py::array_t<int64_t, py::array::c_style | py::array::forcecast> indptr,
    py::array_t<int64_t, py::array::c_style | py::array::forcecast> indices,
    py::array_t<double, py::array::c_style> data, py::ssize_t n_out,
    py::array_t<double, py::array::c_style> w,
    py::array_t<int64_t, py::array::c_style | py::array::forcecast> base_e,
    py::array_t<int64_t, py::array::c_style | py::array::forcecast> base_j,
    py::array table,
    py::array_t<int64_t, py::array::c_style | py::array::forcecast> rsel,
    py::array_t<int64_t, py::array::c_style | py::array::forcecast> csel,
    py::array_t<int32_t, py::array::c_style> rowflat,
    py::array_t<int32_t, py::array::c_style> loc,
    py::array_t<int32_t, py::array::c_style> hpos, py::array tb, py::array held,
    int k, int n_threads) {
    const py::ssize_t n = w.size();
    const py::ssize_t E = base_e.size();
    if (rsel.size() != E) throw std::runtime_error("end_matvecs_table: one rsel per end");
    if (base_j.size() != n || csel.size() != n)
        throw std::runtime_error("end_matvecs_table: one base_j and csel per node");
    if (table.ndim() != 2 || !(table.flags() & py::array::c_style))
        throw std::runtime_error("end_matvecs_table: table must be 2-D C-contiguous");
    if (indptr.size() != n_out + 1)
        throw std::runtime_error("end_matvecs_table: indptr length");
    const int64_t *ip = indptr.data();
    const int64_t *ix = indices.data();
    const double *dv = data.data();
    const int64_t nnz = ip[n_out];
    if (indices.size() < nnz || data.size() < nnz)
        throw std::runtime_error("end_matvecs_table: short CSR arrays");
    for (int64_t jj = 0; jj < nnz; ++jj)
        if (ix[jj] < 0 || ix[jj] >= n)
            throw std::runtime_error("end_matvecs_table: column out of range");
    const py::ssize_t n_rows = loc.size();
    const bool have_h = hpos.size() != 0;
    if (have_h && hpos.size() != n_rows)
        throw std::runtime_error("end_matvecs_table: hpos must match loc");
    const Table T = table_of(tb, "tb"), H = table_of(held, "held");
    const int32_t *hq = have_h ? hpos.data() : nullptr;
    if (table.dtype().is(py::dtype::of<int32_t>()))
        return end_matvecs_table_t<int32_t>(
            ip, ix, dv, n_out, w.data(), n, E, base_e.data(), base_j.data(),
            static_cast<const int32_t *>(table.data()), table.shape(0), table.shape(1),
            rsel.data(), csel.data(), rowflat.data(), rowflat.size(), loc.data(), n_rows,
            hq, T, H, k, n_threads);
    if (table.dtype().is(py::dtype::of<int64_t>()))
        return end_matvecs_table_t<int64_t>(
            ip, ix, dv, n_out, w.data(), n, E, base_e.data(), base_j.data(),
            static_cast<const int64_t *>(table.data()), table.shape(0), table.shape(1),
            rsel.data(), csel.data(), rowflat.data(), rowflat.size(), loc.data(), n_rows,
            hq, T, H, k, n_threads);
    throw std::runtime_error("end_matvecs_table: table must be int32 or int64");
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
    std::vector<py::array> lheld,
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
    // The held set is read through its strides (`table_of`): the streamed
    // route keeps it column-major, a column per held slot.
    Table H[6];
    for (int i = 0; i < 6; ++i) {
        if (lnew[i].ndim() != 2 || lnew[i].shape(0) != rA || lnew[i].shape(1) != n_new)
            throw std::runtime_error("combine_rows: lnew shapes");
        if (!lheld.empty()) {
            H[i] = table_of(lheld[i], "lheld");
            if (H[i].rows != rA || H[i].cols != n_held)
                throw std::runtime_error("combine_rows: lheld shapes");
        }
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
    const double *Ln[6];
    for (int i = 0; i < 6; ++i) Ln[i] = reinterpret_cast<const double *>(lnew[i].data());
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
                                                 : H[i].at(r, static_cast<int>(-1 - c));
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

// A product chunk's table indices (`_ProductTiles._gather`), in one pass:
// for grouped node a and line column c = cols[j] of a slot-"z" product,
//
//     row      = rowflat[base[a] + kl_rank[grank[a], c]]   (`chunk_idx`)
//     li[a, j] = loc[row]                                  (its place in the tile)
//     hp[a, j] = hpos[row] where li < 0, else -1           (its held slot)
//     sidx     = row, when the V/W store is read           (`want_sidx`)
//
// Integers only: what it replaces is numpy's gathers of the same arrays,
// and nothing is computed from a float. A row read before its tile with no
// held slot is refused, as `_gather` refuses it.
static py::tuple product_chunk_index(
    py::array_t<int32_t, py::array::c_style> rowflat,
    py::array_t<int32_t, py::array::c_style> kl_rank,  // (groups, line)
    py::array_t<int64_t, py::array::c_style | py::array::forcecast> grank,
    py::array_t<int64_t, py::array::c_style | py::array::forcecast> base,
    py::array_t<int64_t, py::array::c_style | py::array::forcecast> cols,
    py::array_t<int32_t, py::array::c_style> loc,
    py::array_t<int32_t, py::array::c_style> hpos,  // empty: nothing held
    bool want_sidx, int n_threads) {
    if (kl_rank.ndim() != 2)
        throw std::runtime_error("product_chunk_index: kl_rank must be 2-D");
    const py::ssize_t nG = kl_rank.shape(0), nL = kl_rank.shape(1);
    const py::ssize_t nA = grank.size(), nc = cols.size();
    if (base.size() != nA)
        throw std::runtime_error("product_chunk_index: one base per grouped node");
    const py::ssize_t n_flat = rowflat.size(), n_rows = loc.size();
    const bool have_h = hpos.size() != 0;
    if (have_h && hpos.size() != n_rows)
        throw std::runtime_error("product_chunk_index: hpos must match loc");
    const int64_t *g = grank.data(), *b = base.data(), *cc = cols.data();
    for (py::ssize_t a = 0; a < nA; ++a)
        if (g[a] < 0 || g[a] >= nG)
            throw std::runtime_error("product_chunk_index: group out of range");
    for (py::ssize_t j = 0; j < nc; ++j)
        if (cc[j] < 0 || cc[j] >= nL)
            throw std::runtime_error("product_chunk_index: column out of range");
    const int32_t *rf = rowflat.data(), *kl = kl_rank.data();
    const int32_t *lp = loc.data(), *hq = have_h ? hpos.data() : nullptr;
    py::array_t<int32_t> li(std::vector<py::ssize_t>{nA, nc});
    py::array_t<int32_t> hp(std::vector<py::ssize_t>{nA, nc});
    py::array_t<int64_t> sidx(want_sidx ? std::vector<py::ssize_t>{nA, nc}
                                        : std::vector<py::ssize_t>{0, 0});
    int32_t *L = li.mutable_data(), *H = hp.mutable_data();
    int64_t *S = want_sidx ? sidx.mutable_data() : nullptr;
    int bad = 0, any_miss = 0;
    {
        py::gil_scoped_release nogil;
        int nt = 1;
#ifdef _OPENMP
        nt = omp_get_max_threads();
        if (n_threads > 0) nt = std::min(nt, n_threads);
#endif
#pragma omp parallel for schedule(static) num_threads(nt) reduction(|:bad, any_miss)
        for (py::ssize_t a = 0; a < nA; ++a) {
            const int32_t *klr = kl + g[a] * nL;
            for (py::ssize_t j = 0; j < nc; ++j) {
                const int64_t f = b[a] + klr[cc[j]];
                if (f < 0 || f >= n_flat) {
                    bad |= 1;
                    continue;
                }
                const int32_t row = rf[f];
                if (row < 0 || row >= n_rows) {
                    bad |= 1;
                    continue;
                }
                const py::ssize_t q = a * nc + j;
                const int32_t l = lp[row];
                L[q] = l;
                if (l >= 0) {
                    H[q] = -1;
                } else {
                    any_miss |= 1;
                    const int32_t h = have_h ? hq[row] : -1;
                    if (h < 0) bad |= 2;
                    H[q] = h;
                }
                if (S) S[q] = row;
            }
        }
    }
    if (bad & 1) throw std::runtime_error("product_chunk_index: index out of range");
    if (bad & 2)
        throw std::runtime_error(
            "product_chunk_index: a ready column reads a row not in hand");
    return py::make_tuple(li, any_miss ? py::object(hp) : py::object(py::none()),
                          sidx);
}


// The product tiles' per-tile bookkeeping (momwire#1377), copies only.
//
// `product_rows`: `_ProductPlan.rows(ids)` of a plan that kept each row's
// flat grid position -- (a, b) = divmod(kept_pos[id], nB), the grouped node
// and line node (g, l) = (grank[a], b) on slot "z", (grank[b], a) on "zp",
// and the row (key_r[kid[g, l]], zA[a], zB[b]). Every float is an element
// copied from its array, so the rows are numpy's to the bit.
template <class KP>
static py::array_t<double> product_rows_impl(
    py::array_t<int64_t, py::array::c_style | py::array::forcecast> ids,
    py::array_t<KP, py::array::c_style> kept_pos, int64_t nB, bool slot_z,
    py::array_t<int64_t, py::array::c_style | py::array::forcecast> grank,
    py::array_t<int32_t, py::array::c_style> kid,
    py::array_t<double, py::array::c_style> key_r,
    py::array_t<double, py::array::c_style> zA,
    py::array_t<double, py::array::c_style> zB) {
    if (kid.ndim() != 2) throw std::runtime_error("product_rows: kid must be 2-D");
    if (nB <= 0) throw std::runtime_error("product_rows: nB must be positive");
    const py::ssize_t m = ids.size(), nG = kid.shape(0), nL = kid.shape(1);
    const py::ssize_t n_rows = kept_pos.size(), n_key = key_r.size();
    const py::ssize_t nA = zA.size(), nBs = zB.size(), n_gr = grank.size();
    const int64_t *I = ids.data(), *GR = grank.data();
    const KP *KPp = kept_pos.data();
    const int32_t *K = kid.data();
    const double *R = key_r.data(), *ZA = zA.data(), *ZB = zB.data();
    py::array_t<double> out(std::vector<py::ssize_t>{m, 3});
    double *O = out.mutable_data();
    int bad = 0;
    {
        py::gil_scoped_release nogil;
#pragma omp parallel for schedule(static) reduction(| : bad)
        for (py::ssize_t i = 0; i < m; ++i) {
            const int64_t id = I[i];
            if (id < 0 || id >= n_rows) {
                bad |= 1;
                continue;
            }
            const int64_t f = static_cast<int64_t>(KPp[id]);
            const int64_t a = f / nB, b = f % nB;
            if (f < 0 || a >= nA || b >= nBs) {
                bad |= 1;
                continue;
            }
            const int64_t gnode = slot_z ? a : b, l = slot_z ? b : a;
            if (gnode >= n_gr || l >= nL) {
                bad |= 1;
                continue;
            }
            const int64_t g = GR[gnode];
            if (g < 0 || g >= nG) {
                bad |= 1;
                continue;
            }
            const int32_t k = K[g * nL + l];
            if (k < 0 || k >= n_key) {
                bad |= 1;
                continue;
            }
            O[3 * i] = R[k];
            O[3 * i + 1] = ZA[a];
            O[3 * i + 2] = ZB[b];
        }
    }
    if (bad) throw std::runtime_error("product_rows: index out of range");
    return out;
}

// `tile_block`: one tile's block and stores, what `_ProductTiles.chunks`
// did with numpy on the fused route --
//     refuse a row already done, mark the tile's rows done;
//     tb[:, j] = vals[pos][:, kcols[j]]       (pos empty: vals in row order)
//     held[hpos[ids[i]], :] = tb[i, :]        where that slot is >= 0
//     loc[ids[i]] = i
// -- each value an element copied, so the stores are the same bits. Returns
// (tb in column-major order, whether a row was asked twice); nothing is
// written when one was.
template <class P>
static py::tuple tile_block_impl(
    py::array_t<std::complex<double>> vals, py::array_t<P, py::array::c_style> pos,
    std::vector<int> kcols, py::array_t<int32_t, py::array::c_style> ids,
    py::array_t<bool, py::array::c_style> done, py::array_t<int32_t, py::array::c_style> loc,
    py::array_t<int32_t, py::array::c_style> hpos, py::array_t<std::complex<double>> held) {
    if (vals.ndim() != 2 || held.ndim() != 2)
        throw std::runtime_error("tile_block: vals and held must be 2-D");
    const py::ssize_t m = ids.size(), nv = vals.shape(0), nk = static_cast<py::ssize_t>(kcols.size());
    const py::ssize_t n_rows = done.size();
    const bool have_pos = pos.size() != 0, have_h = hpos.size() != 0;
    if (nv != m || (have_pos && pos.size() != m))
        throw std::runtime_error("tile_block: one value row per id");
    if (loc.size() != n_rows || (have_h && hpos.size() != n_rows))
        throw std::runtime_error("tile_block: loc and hpos must be row-length");
    if (held.shape(1) != nk) throw std::runtime_error("tile_block: held width");
    for (int k : kcols)
        if (k < 0 || k >= vals.shape(1)) throw std::runtime_error("tile_block: kernel column");
    const int32_t *I = ids.data();
    const P *Pp = have_pos ? pos.data() : nullptr;
    for (py::ssize_t i = 0; i < m; ++i) {
        if (I[i] < 0 || I[i] >= n_rows) throw std::runtime_error("tile_block: id out of range");
        if (Pp && (Pp[i] < 0 || Pp[i] >= m)) throw std::runtime_error("tile_block: pos out of range");
    }
    bool *D = done.mutable_data();
    for (py::ssize_t i = 0; i < m; ++i)
        if (D[I[i]]) return py::make_tuple(py::none(), true);
    const int32_t *H = have_h ? hpos.data() : nullptr;
    const py::ssize_t n_held = held.shape(0);
    if (H)
        for (py::ssize_t i = 0; i < m; ++i)
            if (H[I[i]] >= n_held) throw std::runtime_error("tile_block: held slot out of range");
    auto V = vals.unchecked<2>();
    auto HD = held.mutable_unchecked<2>();
    py::array_t<std::complex<double>, py::array::f_style> tb(std::vector<py::ssize_t>{m, nk});
    std::complex<double> *T = tb.mutable_data();
    int32_t *L = loc.mutable_data();
    {
        py::gil_scoped_release nogil;
#pragma omp parallel for schedule(static)
        for (py::ssize_t i = 0; i < m; ++i) {
            const py::ssize_t src = Pp ? static_cast<py::ssize_t>(Pp[i]) : i;
            const int32_t r = I[i];
            const int32_t h = H ? H[r] : -1;
            for (py::ssize_t j = 0; j < nk; ++j) {
                const std::complex<double> v = V(src, kcols[j]);
                T[j * m + i] = v;
                if (h >= 0) HD(h, j) = v;
            }
            D[r] = true;
            L[r] = static_cast<int32_t>(i);
        }
    }
    return py::make_tuple(tb, false);
}

// `loc[ids] = value`, the tile's places cleared after its columns are served.
static void fill_rows_i32(py::array_t<int32_t, py::array::c_style> ids,
                          py::array_t<int32_t, py::array::c_style> loc, int32_t value) {
    const py::ssize_t m = ids.size(), n = loc.size();
    const int32_t *I = ids.data();
    for (py::ssize_t i = 0; i < m; ++i)
        if (I[i] < 0 || I[i] >= n) throw std::runtime_error("fill_rows_i32: id out of range");
    int32_t *L = loc.mutable_data();
    py::gil_scoped_release nogil;
#pragma omp parallel for schedule(static)
    for (py::ssize_t i = 0; i < m; ++i) L[I[i]] = value;
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
    m.def("product_chunk_index", &left_gather::product_chunk_index,
          "A slot-z product chunk's table indices in one pass: li = "
          "loc[row], hp = hpos[row] where li < 0 (None when no row misses), "
          "and sidx = row when asked, for row = rowflat[base[a] + "
          "kl_rank[grank[a], cols[j]]]. momwire#1290.",
          py::arg("rowflat"), py::arg("kl_rank"), py::arg("grank"),
          py::arg("base"), py::arg("cols"), py::arg("loc"), py::arg("hpos"),
          py::arg("want_sidx"), py::arg("n_threads"));
    m.attr("product_chunk_index_1290") = true;
    m.def("product_rows", &left_gather::product_rows_impl<int32_t>,
          "`_ProductPlan.rows(ids)` of a plan with kept grid positions: rows "
          "(key_r[kid[g, l]], zA[a], zB[b]) for (a, b) = divmod(kept_pos[id], "
          "nB), copies only. momwire#1377.",
          py::arg("ids"), py::arg("kept_pos"), py::arg("nB"), py::arg("slot_z"),
          py::arg("grank"), py::arg("kid"), py::arg("key_r"), py::arg("zA"),
          py::arg("zB"));
    m.def("product_rows", &left_gather::product_rows_impl<int64_t>,
          py::arg("ids"), py::arg("kept_pos"), py::arg("nB"), py::arg("slot_z"),
          py::arg("grank"), py::arg("kid"), py::arg("key_r"), py::arg("zA"),
          py::arg("zB"));
    m.def("tile_block", &left_gather::tile_block_impl<int64_t>,
          "One product tile's block, held store, places and done marks, as "
          "`_ProductTiles.chunks` formed them, copies only: (tb, asked_twice). "
          "momwire#1377.",
          py::arg("vals"), py::arg("pos"), py::arg("kcols"), py::arg("ids"),
          py::arg("done"), py::arg("loc"), py::arg("hpos"), py::arg("held"));
    m.def("tile_block", &left_gather::tile_block_impl<int32_t>,
          py::arg("vals"), py::arg("pos"), py::arg("kcols"), py::arg("ids"),
          py::arg("done"), py::arg("loc"), py::arg("hpos"), py::arg("held"));
    m.def("fill_rows_i32", &left_gather::fill_rows_i32,
          "loc[ids] = value for int32 arrays. momwire#1377.",
          py::arg("ids"), py::arg("loc"), py::arg("value"));
    m.attr("product_tiles_1377") = true;
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
    m.def("end_matvecs_rows", &left_gather::end_matvecs_rows,
          "`end_matvecs` reading X[e, j] at product row R[e, j]: the tile "
          "block's kernel k at loc[R] (int32), else the held store's at "
          "hpos[R]. The same sums, the gather done per element. Returns "
          "(E, n_out) complex; OpenMP over ends. momwire#1335.",
          py::arg("indptr"), py::arg("indices"), py::arg("data"),
          py::arg("n_out"), py::arg("w"), py::arg("R"), py::arg("loc"),
          py::arg("hpos"), py::arg("tb"), py::arg("held"), py::arg("k"),
          py::arg("n_threads"));
    m.attr("end_matvecs_rows_1335") = true;
    m.def("end_matvecs_table", &left_gather::end_matvecs_table,
          "`end_matvecs_rows` with R[e, j] = rowflat[base_e[e] + base_j[j] + "
          "table[rsel[e], csel[j]]] formed in the loop, never stored. "
          "momwire#1335.",
          py::arg("indptr"), py::arg("indices"), py::arg("data"),
          py::arg("n_out"), py::arg("w"), py::arg("base_e"), py::arg("base_j"),
          py::arg("table"), py::arg("rsel"), py::arg("csel"), py::arg("rowflat"),
          py::arg("loc"), py::arg("hpos"), py::arg("tb"), py::arg("held"),
          py::arg("k"), py::arg("n_threads"));
    m.attr("end_matvecs_table_1335") = true;
}
