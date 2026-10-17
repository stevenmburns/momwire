// The crossing fill's shared C++ pieces (`_crossing_fill.py`): the axis
// records and their CSR samples, the graded panels, the designed tables
// through a block's memo, the main sandwich's gathers, and the by-parts end
// terms (`_end_vectors`, `_EndTerms`). Moved from `_razor_crossing_core.cpp`
// (momwire#1517 step 4) so bs2's crossing fill runs the same code; `P::raise`
// raises Python's ValueError in the caller's terms, and `C` is the caller's
// context (`gz`, `c1`).
#pragma once

#include "_fma_inline.h"
#include "_near_interface_eval.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <complex>
#include <cstdint>
#include <cstdio>
#include <map>
#include <set>
#include <string>
#include <tuple>
#include <vector>

#if defined(_OPENMP)
#include <omp.h>
#endif

namespace mw {
namespace crossing {
namespace {

using cd = std::complex<double>;
using ni_eval::Six;

constexpr double PLANE_TOL = 1e-12;     // `_crossing_fill._PLANE_TOL`
constexpr int64_t END_BATCH_PAIRS = 1 << 16;
constexpr int K_U = 0, K_V = 1, K_W = 2, K_DZPW = 5;  // KEYS order

// numpy's fused complex product (`_cmul.cmul`'s formula, operand order kept).
inline cd np_cmul(cd a, cd b) {
    return cd(mw_fma::exact(a.real(), b.real(), -(a.imag() * b.imag())),
              mw_fma::exact(a.real(), b.imag(), a.imag() * b.real()));
}
// numpy's complex division (`CDOUBLE_divide`, Smith's).
inline cd np_cdiv(cd a, cd b) {
    const double in1r = a.real(), in1i = a.imag(), in2r = b.real(), in2i = b.imag();
    const double in2r_abs = std::fabs(in2r), in2i_abs = std::fabs(in2i);
    if (in2r_abs >= in2i_abs) {
        if (in2r_abs == 0 && in2i_abs == 0) return cd(in1r / in2r_abs, in1i / in2i_abs);
        const double rat = in2i / in2r;
        const double scl = 1.0 / (in2r + in2i * rat);
        return cd((in1r + in1i * rat) * scl, (in1i - in1r * rat) * scl);
    }
    const double rat = in2r / in2i;
    const double scl = 1.0 / (in2i + in2r * rat);
    return cd((in1r * rat + in1i) * scl, (in1i * rat - in1r) * scl);
}
// CPython's complex product.
inline cd py_prod(cd a, cd b) {
    return cd(a.real() * b.real() - a.imag() * b.imag(), a.real() * b.imag() + a.imag() * b.real());
}

// `%.6g`, Python's `:.6g`
std::string crossing_g6(double x) {
    char buf[64];
    std::snprintf(buf, sizeof buf, "%.6g", x);
    return buf;
}

// `_on_plane_side(zrel, side, what)` on one value.
template <class P>
double on_plane_side(double z, bool above, const char *what) {
    const bool wrong = above ? z < 0.0 : z > 0.0;
    if (!wrong) return z;
    if (std::fabs(z) >= PLANE_TOL) {
        const char *side = above ? "above" : "below";
        P::raise(std::string("crossing fill: the ") + side + " axis's " + what + " sits " +
                    crossing_g6(std::fabs(z)) + " on the wrong side of ground_z, past the 1e-12 "
                    "tolerance that says a point IS the interface. The designed tables are "
                    "derived for z >= 0 >= z', so there is no honest side to put this on: an " +
                    side + " member must not cross the plane. Check the deck's crossing "
                    "junction -- exactly one above member and the rest below (momwire#852)");
    }
    return 0.0;
}

struct CSR {
    int64_t n_rows = 0, n_cols = 0;
    std::vector<int64_t> indptr, indices;
    std::vector<double> data;
};

// Rows (r, c, v) to a canonical CSR (indices sorted within each row; no
// duplicates in our inputs).
CSR to_csr(int64_t n_rows, int64_t n_cols, std::vector<std::tuple<int64_t, int64_t, double>> t) {
    std::stable_sort(t.begin(), t.end(), [](const std::tuple<int64_t, int64_t, double> &a,
                                            const std::tuple<int64_t, int64_t, double> &b) {
        return std::get<0>(a) != std::get<0>(b) ? std::get<0>(a) < std::get<0>(b)
                                                : std::get<1>(a) < std::get<1>(b);
    });
    CSR m;
    m.n_rows = n_rows;
    m.n_cols = n_cols;
    m.indptr.assign(static_cast<size_t>(n_rows + 1), 0);
    for (const auto &e : t) m.indptr[static_cast<size_t>(std::get<0>(e)) + 1]++;
    for (int64_t r = 0; r < n_rows; r++) m.indptr[r + 1] += m.indptr[r];
    for (const auto &e : t) {
        m.indices.push_back(std::get<1>(e));
        m.data.push_back(std::get<2>(e));
    }
    return m;
}

struct End {
    double pt[3];
    double sign;
    std::vector<double> fv;  // (n_basis)
};

struct Axis {
    int64_t n_basis = 0;
    std::vector<double> nodes, t, w;  // (n, 3), (n, 3), (n)
    CSR F, Fd;
    std::vector<End> ends;
    bool path_tested = false;
    std::vector<int64_t> segof;
    std::map<int64_t, std::vector<int64_t>> seg_rows;
    size_t n_nodes() const { return w.size(); }
};

// `_graded_u(h, toward_end, a, growth, gx, gw)`.
void graded_u(double h, bool toward_hi, double a, double growth, const std::vector<double> &gx,
              const std::vector<double> &gw, std::vector<double> &u, std::vector<double> &w) {
    std::vector<double> edges{0.0};
    double step = a;
    while (edges.back() + step < h) {
        edges.push_back(edges.back() + step);
        step *= growth;
    }
    edges.push_back(h);
    if (toward_hi) {
        std::vector<double> e(edges.size());
        for (size_t i = 0; i < edges.size(); i++) e[i] = h - edges[edges.size() - 1 - i];
        edges.swap(e);
    }
    for (size_t p = 0; p + 1 < edges.size(); p++) {
        const double mid = 0.5 * (edges[p] + edges[p + 1]);
        const double half = 0.5 * (edges[p + 1] - edges[p]);
        for (size_t k = 0; k < gx.size(); k++) {
            u.push_back(mid + half * gx[k]);
            w.push_back(half * gw[k]);
        }
    }
}

// `_support_rows(ax, all nodes)`.
std::vector<int64_t> support_rows(const Axis &ax) {
    std::vector<int64_t> out;
    if (ax.path_tested) {
        for (int64_t r = 0; r < ax.F.n_rows; r++)
            if (ax.F.indptr[r + 1] > ax.F.indptr[r] || ax.Fd.indptr[r + 1] > ax.Fd.indptr[r]) out.push_back(r);
        return out;
    }
    std::set<int64_t> rows;
    std::set<int64_t> segs(ax.segof.begin(), ax.segof.end());
    for (int64_t g : segs) {
        auto it = ax.seg_rows.find(g);
        if (it != ax.seg_rows.end()) rows.insert(it->second.begin(), it->second.end());
    }
    return std::vector<int64_t>(rows.begin(), rows.end());
}

// `_row_weights(ax, all, rows)`: (F w tx, F w ty, F w tz, Fd w) restricted
// to `rows`.
std::array<CSR, 4> row_weights(const Axis &ax, const std::vector<int64_t> &rows) {
    std::array<CSR, 4> out;
    for (int m = 0; m < 4; m++) {
        CSR &o = out[static_cast<size_t>(m)];
        o.n_rows = static_cast<int64_t>(rows.size());
        o.n_cols = ax.F.n_cols;
        o.indptr.assign(rows.size() + 1, 0);
        const CSR &src = m < 3 ? ax.F : ax.Fd;
        for (size_t i = 0; i < rows.size(); i++) {
            const int64_t r = rows[i];
            for (int64_t jj = src.indptr[r]; jj < src.indptr[r + 1]; jj++) {
                const int64_t col = src.indices[jj];
                double v = src.data[jj] * ax.w[static_cast<size_t>(col)];
                if (m < 3) v = v * ax.t[3 * static_cast<size_t>(col) + m];
                o.indices.push_back(col);
                o.data.push_back(v);
            }
            o.indptr[i + 1] = static_cast<int64_t>(o.indices.size());
        }
    }
    return out;
}

// One block's memo and evaluator.
struct BlockEval {
    ni_eval::Evaluator ev;
    ni_eval::Memo memo;
};

// The OpenMP team for the block's own loops: the evaluator's cap, held to
// what OpenMP would give (1 in the serial copy). Every loop it splits writes
// disjoint outputs, each one's arithmetic its serial order's.
int team(const BlockEval &be, size_t work) {
#if defined(_OPENMP)
    int nt = omp_get_max_threads();
    if (be.ev.n_threads > 0) nt = std::min(nt, be.ev.n_threads);
    return work < 4096 ? 1 : std::max(1, nt);
#else
    (void)be;
    (void)work;
    return 1;
#endif
}

// `_tables(ctx_r, eps_t, k_p, rho, z, zp, rtol, memo, group_labels)` over n
// asked points: rho folded with the radius.
void tables(BlockEval &be, double a_wire, const std::vector<double> &rho, const std::vector<double> &z,
            const std::vector<double> &zp, const std::vector<int64_t> *labels, std::vector<Six> &out) {
    std::vector<double> re(rho.size());
    for (size_t i = 0; i < rho.size(); i++) re[i] = std::hypot(rho[i], a_wire);
    out.resize(rho.size());
    ni_eval::designed_tables(be.ev, &be.memo, re.data(), z.data(), zp.data(), rho.size(),
                             labels ? labels->data() : nullptr, out.data());
}

// The same, each point answered as its value's index in the block's memo
// (`rho` already folded with the radius).
void table_index(BlockEval &be, const std::vector<double> &re, const std::vector<double> &z,
                 const std::vector<double> &zp, const std::vector<int64_t> *labels, std::vector<uint32_t> &idx) {
    idx.resize(re.size());
    ni_eval::designed_index(be.ev, be.memo, re.data(), z.data(), zp.data(), re.size(),
                            labels ? labels->data() : nullptr, idx.data(), team(be, re.size()));
}

// One key of every memo value, contiguous (the contractions' gathers).
std::vector<cd> key_plane(const ni_eval::Memo &memo, int key) {
    std::vector<cd> out(memo.size());
    for (size_t j = 0; j < out.size(); j++) out[j] = memo.at(static_cast<uint32_t>(j))[static_cast<size_t>(key)];
    return out;
}

// One real CSR row against complex v: each part's products summed in the
// row's order (`(M @ v.real) + 1j (M @ v.imag)`, scipy's row loop). Plain
// restrict-qualified pointers: through the vectors' accessors the loop ran
// ten times slower, with the same arithmetic.
inline void csr_row_dot(const int64_t *__restrict ix, const double *__restrict d, int64_t j0, int64_t j1,
                        const double *__restrict v, double &re_out, double &im_out) {
    double re = 0.0, im = 0.0;
    for (int64_t jj = j0; jj < j1; jj++) {
        const double *x = v + 2 * ix[jj];
        const double a = d[jj];
        re += a * x[0];
        im += a * x[1];
    }
    re_out = re;
    im_out = im;
}

// y[c] += a * tab[ix[c]], each part one product and one sum (no FMA): the
// left products' inner loop, with nothing aliased so it vectorizes alike in
// the serial and the threaded copy (elementwise, so its bits never depend on
// how it is vectorized).
void axpy_gather(cd *__restrict y, const cd *__restrict tab, const uint32_t *__restrict ix, double a,
                 size_t n) {
    double *__restrict yd = reinterpret_cast<double *>(y);
    const double *__restrict td = reinterpret_cast<const double *>(tab);
    for (size_t cc = 0; cc < n; cc++) {
        const size_t k = 2 * static_cast<size_t>(ix[cc]);
        yd[2 * cc] = yd[2 * cc] + a * td[k];
        yd[2 * cc + 1] = yd[2 * cc + 1] + a * td[k + 1];
    }
}

// `_end_vectors` over one loop's ends against `line`: per end, (sign, fv,
// vV, vW) with vV = Fd_line (w V), vW = F_line (w tz W); the tables from
// `_end_tables`' labelled spans through the block's memo. `end_above`: the
// end in the z slot (`_above_end_args`), else in z' (`_below_end_args`).
struct EndVec {
    double sign;
    const std::vector<double> *fv;
    std::vector<cd> vV, vW;
};

std::vector<cd> real_matvec(const CSR &M, const std::vector<cd> &v) {
    std::vector<cd> out(static_cast<size_t>(M.n_rows));
    const int64_t *ip = M.indptr.data();
    const int64_t *ix = M.indices.data();
    const double *d = M.data.data();
    const double *vd = reinterpret_cast<const double *>(v.data());
    double *od = reinterpret_cast<double *>(out.data());
    for (int64_t i = 0; i < M.n_rows; i++) csr_row_dot(ix, d, ip[i], ip[i + 1], vd, od[2 * i], od[2 * i + 1]);
    return out;
}

template <class P, class CX>
std::vector<EndVec> end_vectors(const CX &c, BlockEval &be, double a_wire, const std::vector<End> &ends,
                                const Axis &line, bool end_above) {
    std::vector<EndVec> out;
    const size_t n = line.n_nodes();
    if (ends.empty()) return out;
    // `_LineZ`: the line's plane-side z, formed once (refusing at the first end).
    std::vector<double> line_z(n);
    for (size_t k = 0; k < n; k++)
        line_z[k] = on_plane_side<P>(line.nodes[3 * k + 2] - c.gz, !end_above, "quadrature node");
    std::vector<double> w(n), w_tz(n);
    for (size_t k = 0; k < n; k++) {
        w[k] = line.w[k];
        w_tz[k] = line.w[k] * line.t[3 * k + 2];
    }
    const size_t per = std::max<size_t>(1, static_cast<size_t>(END_BATCH_PAIRS) / std::max<size_t>(1, n));
    out.resize(ends.size());
    std::vector<double> re, z, zp;
    std::vector<int64_t> lab;
    std::vector<uint32_t> te;
    for (size_t g0 = 0; g0 < ends.size(); g0 += per) {
        const size_t g1 = std::min(ends.size(), g0 + per);
        const size_t ne = g1 - g0;
        // The ends' own z, refused in order before any point is formed.
        std::vector<double> endz(ne);
        for (size_t e = g0; e < g1; e++) endz[e - g0] = on_plane_side<P>(ends[e].pt[2] - c.gz, end_above, "end point");
        re.resize(ne * n);
        z.resize(ne * n);
        zp.resize(ne * n);
        lab.resize(ne * n);
        const std::ptrdiff_t nep = static_cast<std::ptrdiff_t>(ne);
        const int nt = team(be, ne * n);
#if defined(_OPENMP)
        #pragma omp parallel for schedule(static) num_threads(nt) if (nt > 1)
#else
        (void)nt;
#endif
        for (std::ptrdiff_t ep = 0; ep < nep; ep++) {
            const size_t e = static_cast<size_t>(ep);
            const End &E = ends[g0 + e];
            for (size_t k = 0; k < n; k++) {
                const double *q = &line.nodes[3 * k];
                const size_t i = e * n + k;
                const double rho = end_above ? std::hypot(E.pt[0] - q[0], E.pt[1] - q[1])
                                             : std::hypot(q[0] - E.pt[0], q[1] - E.pt[1]);
                re[i] = std::hypot(rho, a_wire);
                z[i] = end_above ? endz[e] : line_z[k];
                zp[i] = end_above ? line_z[k] : endz[e];
                lab[i] = static_cast<int64_t>(e);
            }
        }
        table_index(be, re, z, zp, ne > 1 ? &lab : nullptr, te);
#if defined(_OPENMP)
        #pragma omp parallel num_threads(nt) if (nt > 1)
#endif
        {
        std::vector<cd> V(n), W(n);  // each thread's, reused end to end
#if defined(_OPENMP)
        #pragma omp for schedule(dynamic)
#endif
        for (std::ptrdiff_t ep = 0; ep < nep; ep++) {
            const size_t e = static_cast<size_t>(ep);
            for (size_t k = 0; k < n; k++) {
                const Six &s = be.memo.at(te[e * n + k]);
                // `w * V`: numpy's (w + 0j)(V): each part one rounding.
                V[k] = np_cmul(cd(w[k], 0.0), s[K_V]);
                W[k] = np_cmul(cd(w_tz[k], 0.0), s[K_W]);
            }
            EndVec &ev = out[g0 + e];
            ev.sign = ends[g0 + e].sign;
            ev.fv = &ends[g0 + e].fv;
            ev.vV = real_matvec(line.Fd, V);
            ev.vW = real_matvec(line.F, W);
        }
        }
    }
    return out;
}

// No corner (razor's path-tested rows).
struct NoCorner {
    void operator()(std::vector<cd> &, const std::vector<int64_t> &, int64_t) const {}
};

// `_ends_and_corner_rc(..., corner=False)` + `_EndTerms.answer(out=t)`: the
// row loop (R's ends against C's line; V then W) and the column loop (C's
// ends against R's line; W then V), each a rank-1 update into E_r / E_c, then
// scattered into t (nR x nC, row-major).
// `corner(E_r, posLA, nC)` adds the designed corner into E_r ((|LA|, nC)
// row-major, R's live rows) after both loops, before the scatter.
template <class P, class CX, class CornerFn = NoCorner>
void add_end_terms(const CX &c, BlockEval &be, double a_wire, const Axis &R, const Axis &C,
                   bool r_above, std::vector<cd> &t, const CornerFn &corner = CornerFn()) {
    const int64_t nR = R.n_basis, nC = C.n_basis;
    auto live = [](const Axis &ax) {
        std::set<int64_t> s;
        for (const End &e : ax.ends)
            for (size_t i = 0; i < e.fv.size(); i++)
                if (e.fv[i] != 0.0) s.insert(static_cast<int64_t>(i));
        return std::vector<int64_t>(s.begin(), s.end());
    };
    const std::vector<int64_t> LA = live(R), LB = live(C);
    std::vector<int64_t> posLA(static_cast<size_t>(nR), -1), posLB(static_cast<size_t>(nC), -1);
    for (size_t i = 0; i < LA.size(); i++) posLA[static_cast<size_t>(LA[i])] = static_cast<int64_t>(i);
    for (size_t i = 0; i < LB.size(); i++) posLB[static_cast<size_t>(LB[i])] = static_cast<int64_t>(i);
    std::vector<cd> E_r(LA.size() * static_cast<size_t>(nC), cd(0.0, 0.0));
    std::vector<cd> E_c(static_cast<size_t>(nR) * LB.size(), cd(0.0, 0.0));
    const cd neg_c1(-c.c1.real(), -c.c1.imag());
    // Row loop: R's ends against C's line.
    {
        const std::vector<EndVec> evs = end_vectors<P>(c, be, a_wire, R.ends, C, r_above);
        // Columns split over the team; each entry takes the ends' rank-1
        // terms in end order, as the serial loop adds them.
        const int nt = team(be, evs.size() * static_cast<size_t>(nC));
#if defined(_OPENMP)
        #pragma omp parallel for schedule(static, 1) num_threads(nt) if (nt > 1)
#endif
        for (int ch = 0; ch < nt; ch++) {
            const int64_t j0 = nC * ch / nt, j1 = nC * (ch + 1) / nt;
            for (const EndVec &ev : evs) {
                const cd s_plus = py_prod(c.c1, cd(ev.sign, 0.0));
                const cd s_minus = py_prod(neg_c1, cd(ev.sign, 0.0));
                for (int pass = 0; pass < 2; pass++) {
                    const std::vector<cd> &vec = pass == 0 ? ev.vV : ev.vW;
                    const cd scale = pass == 0 ? s_plus : s_minus;
                    for (size_t i = 0; i < ev.fv->size(); i++) {
                        const double fvi = (*ev.fv)[i];
                        if (fvi == 0.0) continue;
                        cd *row = &E_r[static_cast<size_t>(posLA[i]) * static_cast<size_t>(nC)];
                        for (int64_t j = j0; j < j1; j++) {
                            const cd o = np_cmul(np_cmul(cd(fvi, 0.0), vec[static_cast<size_t>(j)]), scale);
                            row[j] = cd(row[j].real() + o.real(), row[j].imag() + o.imag());
                        }
                    }
                }
            }
        }
    }
    // Column loop: C's ends against R's line (W, then V).
    {
        const std::vector<EndVec> evs = end_vectors<P>(c, be, a_wire, C.ends, R, !r_above);
        // Rows split over the team (E_r's rows LA and E_c's rows apart);
        // each entry takes the ends' terms in end order, as serially.
        const int64_t nLA = static_cast<int64_t>(LA.size());
        const int nt = team(be, evs.size() * static_cast<size_t>(nR));
#if defined(_OPENMP)
        #pragma omp parallel for schedule(static, 1) num_threads(nt) if (nt > 1)
#endif
        for (int ch = 0; ch < nt; ch++) {
            const int64_t a0 = nLA * ch / nt, a1 = nLA * (ch + 1) / nt;
            const int64_t r0 = nR * ch / nt, r1 = nR * (ch + 1) / nt;
            for (const EndVec &ev : evs) {
                const cd s_plus = py_prod(c.c1, cd(ev.sign, 0.0));
                const cd s_minus = py_prod(neg_c1, cd(ev.sign, 0.0));
                for (int pass = 0; pass < 2; pass++) {
                    const std::vector<cd> &vec = pass == 0 ? ev.vW : ev.vV;
                    const cd scale = pass == 0 ? s_minus : s_plus;
                    for (size_t jn = 0; jn < ev.fv->size(); jn++) {
                        const double fvj = (*ev.fv)[jn];
                        if (fvj == 0.0) continue;
                        // E_r's columns (rows LA), then E_c's (every row).
                        for (int64_t i = a0; i < a1; i++) {
                            const cd o = np_cmul(np_cmul(vec[static_cast<size_t>(LA[static_cast<size_t>(i)])], cd(fvj, 0.0)), scale);
                            cd &e = E_r[static_cast<size_t>(i) * static_cast<size_t>(nC) + jn];
                            e = cd(e.real() + o.real(), e.imag() + o.imag());
                        }
                        for (int64_t i = r0; i < r1; i++) {
                            const cd o = np_cmul(np_cmul(vec[static_cast<size_t>(i)], cd(fvj, 0.0)), scale);
                            cd &e = E_c[static_cast<size_t>(i) * LB.size() + static_cast<size_t>(posLB[jn])];
                            e = cd(e.real() + o.real(), e.imag() + o.imag());
                        }
                    }
                }
            }
        }
    }
    corner(E_r, posLA, nC);
    // `answer`: E_c[LA, :] = 0; dest[LA, :] += E_r; dest[:, LB] += E_c.
    for (int64_t a : LA)
        for (size_t j = 0; j < LB.size(); j++) E_c[static_cast<size_t>(a) * LB.size() + j] = cd(0.0, 0.0);
    for (size_t i = 0; i < LA.size(); i++)
        for (int64_t j = 0; j < nC; j++) {
            cd &d = t[static_cast<size_t>(LA[i] * nC + j)];
            const cd v = E_r[i * static_cast<size_t>(nC) + static_cast<size_t>(j)];
            d = cd(d.real() + v.real(), d.imag() + v.imag());
        }
    for (int64_t i = 0; i < nR; i++)
        for (size_t j = 0; j < LB.size(); j++) {
            cd &d = t[static_cast<size_t>(i * nC + LB[j])];
            const cd v = E_c[static_cast<size_t>(i) * LB.size() + j];
            d = cd(d.real() + v.real(), d.imag() + v.imag());
        }
}

}  // namespace
}  // namespace crossing
}  // namespace mw
