// The C++ twins of the designed near-interface walk: `six_point` per triple
// (momwire#680 U2) and `six_columns` per rho column (momwire#899 item 1).
//
// TWO ENTRIES, ONE INTEGRAND. `near_interface_six_batch` walks each triple
// adaptively; `near_interface_six_columns` builds the FIXED per-column rule
// once and spends one exponential per node per member. They are the two
// routes `_near_interface.designed_tables` dispatches between, and each is
// the twin of the numpy walk of the same name — never of the other.
//
// Both are WALK ports, not integrand ports (the house twin rule: shared
// walk + limit, never pointwise): the head/mid machinery is the very same
// templated engine `_accelerators.cpp` rides (`_contour_engine_inline.h`,
// momwire#568 U1 — `adaptive_segment`, `head_contour`, the complex J0/J1
// kernels), instantiated here with the crossing serve's six-component
// spectral core, plus a transcription of the ONE structure the engine does
// not carry: the rotated-ray tail lam = lam_top + t e^{+-j pi/4}. Every
// structural rule of the Python walk survives verbatim:
//
//   * head [0, 1.1K] via the first-quadrant half-sine detour (branch points
//     + transmitted pole inside), mid [1.1K, 8K] real-axis adaptive Gauss,
//     tail = rotated rays;
//   * ray panels START at the lam0 scale and double toward the decay scale
//     sqrt(2)/(s+rho) — starting at the decay scale is a measured 44 %
//     silent W-log error at s = 1e-5;
//   * exact-underflow ray panels count as QUIET (high-sigma far pairs
//     underflow the whole tail to exact 0.0 — the tail IS zero, not a
//     stall); two consecutive quiet panels end the ray;
//   * the far-pair kill cap lam <= 60/s, with the head keeping the k_p
//     branch point + transmitted pole inside;
//   * rho = 0 takes ONE up-ray with J0 = 1; rho > 0 splits J0 into Hankel
//     halves, H1 up-ray / H2 down-ray (conjugate ray direction);
//   * derivative bookkeeping: dz <-> -gamma_p, dz' <-> +gamma_m; z' <= 0 so
//     e^{-gamma_m|z'|} = e^{+gamma_m z'}.
//
// The complex-argument Hankel pair comes from the VENDORED scipy/xsf
// (extern/xsf, header-only C++17, scipy.special's own Amos translation —
// see extern/xsf/VENDOR-PIN.md). Its underflow contract is load-bearing:
// Amos underflow returns the exact 0.0 rather than NaN, which is precisely
// the quiet-panel rule above. This is also why the module is compiled at
// C++17 in its OWN extension: `_accelerators` stays at gnu++11, untouched.
//
// The numpy walk (`_near_interface.six_point`) stays production's reference;
// the gates against it are RELATIVE (1e-12), never bit — the transcendental
// libraries and the Gauss dot products differ in the last bits (house rule:
// no cross-build bit equality, momwire#249/#270).
//
// REENTRANCY: the engine header is allocation-free with no mutable static
// (its #568 U1 contract), the spectral core below holds only per-point
// scalars, and xsf's Amos kernels are pure functions — so the batch loop
// runs OpenMP `dynamic` over points with the GIL released, exactly the
// `below_six_integrals_batch` pattern. Domain validation happens BEFORE the
// parallel region (a throw cannot cross an omp boundary); the one in-loop
// failure mode (a ray that never goes quiet) sets a flag that is raised
// afterwards in the Python walk's own words.
//
// The COLUMN entry (#899 item 1) carries the rule as well as the sum, and
// that is the whole point of it: on the measured BLE column a third of the
// numpy route's wall is `_column_rule` + `_column_factors` (complex-argument
// Bessel/Hankel through scipy, the `_sub_seed` loop in Python, the Gauss
// panels) and two thirds the (nz x K) exponential, so a twin that only did
// exp + dot would leave the third behind. Its parallel unit is the COLUMN,
// not the point: a column's rule is built by the thread that owns it, every
// scratch buffer is thread-local, and there is no shared mutable state — so
// it also supplies the thread scaling the numpy route has none of (#898).

#define _USE_MATH_DEFINES

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstdint>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

#include "_contour_engine_inline.h"
#include "_branch_cut_inline.h"
#include "xsf/bessel.h"

namespace py = pybind11;

// The column entry's inner loop is exp + sincos per node per member, which
// is exactly what glibc's libmvec vectorizes (`-lmvec` is already on this
// extension's link line). Ubuntu's <cmath> carries no `omp declare simd`
// markers for them, so GCC cannot substitute `_ZGVdN4v_exp` / `_ZGVdN4v_cos`
// / `_ZGVdN4v_sin` inside an `omp simd` loop without these declarations; the
// std:: overloads still resolve to the same extern-C symbols, so the calls
// below pick the vectorized form up for free. Same block, same gating and
// same reason as `_accel_common.h`'s (the sibling extension's) — MSVC has no
// libmvec and would choke on redeclaring the CRT's exp/sin/cos, and macOS
// has neither libmvec nor the AVX2 simdlen(4) form.
#if defined(__GNUC__) && !defined(_MSC_VER) && !defined(__APPLE__)
#pragma omp declare simd notinbranch simdlen(4)
extern "C" double exp(double);

#pragma omp declare simd notinbranch simdlen(4)
extern "C" double cos(double);

#pragma omp declare simd notinbranch simdlen(4)
extern "C" double sin(double);
#endif

// `omp simd` neutralization for MSVC, whose /openmp:llvm rejects the
// directive outright (the `_accel_common.h` note has the long form). The
// pragmas below are a vectorization hint and nothing else: the arithmetic is
// a plain reduction either way, so dropping them costs speed, never bits.
#if defined(_MSC_VER)
#define MW_NI_SIMD(clauses)
#else
#define MW_NI_PRAGMA_(x) _Pragma(#x)
#define MW_NI_SIMD(clauses) MW_NI_PRAGMA_(omp simd clauses)
#endif

#include "_near_interface_columns_inline.h"

// `six_point` over parallel (rho, z, zp) arrays: the (n, 6) table, OpenMP
// across points with the GIL released. The wavenumbers arrive DERIVED
// (k_p real, k_m complex on the Im <= 0 branch) rather than as eps~, so the
// branch choice stays in `k_medium` where it is written down once; the
// domain raises keep the Python walk's words. The U1 exact-triple memo
// stays in Python — this entry expects the caller to hand it UNIQUE
// triples (that list is also U3's parallel unit).
static py::array_t<std::complex<double>> near_interface_six_batch(
    double k_p, std::complex<double> k_m,
    py::array_t<double, py::array::c_style | py::array::forcecast> rho,
    py::array_t<double, py::array::c_style | py::array::forcecast> z,
    py::array_t<double, py::array::c_style | py::array::forcecast> zp,
    double rtol, double lam_mult, int depth, double detour,
    py::array_t<double, py::array::c_style | py::array::forcecast> gx,
    py::array_t<double, py::array::c_style | py::array::forcecast> gw) {
    auto rb = rho.unchecked<1>();
    auto zb = z.unchecked<1>();
    auto pb = zp.unchecked<1>();
    const py::ssize_t n = rb.shape(0);
    if (zb.shape(0) != n || pb.shape(0) != n)
        throw std::invalid_argument("rho, z and zp must have the same length");
    if (gx.ndim() != 1 || gw.ndim() != 1 || gx.shape(0) != gw.shape(0) ||
        gx.shape(0) < 1)
        throw std::invalid_argument("bad Gauss rule");
    const double *gxp = gx.data();
    const double *gwp = gw.data();
    const int ng = static_cast<int>(gx.shape(0));

    // The Python walk's domain raises, verbatim and BEFORE the parallel
    // region (a throw cannot cross an omp boundary).
    for (py::ssize_t i = 0; i < n; ++i) {
        if (!(zb(i) >= 0.0 && pb(i) <= 0.0))
            throw std::invalid_argument("need z >= 0 >= zp");
        if (rb(i) < 0.0 || (zb(i) - pb(i)) + rb(i) <= 0.0)
            throw std::invalid_argument("need R > 0");
    }

    py::array_t<std::complex<double>> vals({n, py::ssize_t(6)});
    auto vb = vals.mutable_unchecked<2>();
    const mw_contour::cd km(k_m);
    bool ray_failed = false;

    {
        py::gil_scoped_release release;
        // `dynamic`: a corner point costs many more adaptive levels than a
        // mid-range pair, and a fill's unique triples arrive sorted by
        // geometry — static scheduling would hand one thread the hard band.
        #pragma omp parallel for schedule(dynamic)
        for (py::ssize_t i = 0; i < n; ++i) {
            mw_contour::cd out[6];
            if (!mw680::six_point_one(k_p, km, rb(i), zb(i), pb(i), rtol,
                                      lam_mult, depth, detour, gxp, gwp, ng,
                                      out)) {
                #pragma omp atomic write
                ray_failed = true;
                continue;
            }
            for (int c = 0; c < 6; ++c) vb(i, c) = out[c];
        }
    }
    if (ray_failed)
        throw std::runtime_error(
            "rotated tail did not go quiet inside the panel budget");
    return vals;
}

// `six_columns` over CONCATENATED columns: column c owns the members
// offsets[c] .. offsets[c+1] of (z, zp), all at rho[c]. One call for a whole
// fill's grouping, because the parallel units live INSIDE it (see the split
// below) and a per-column call would hand OpenMP one column at a time —
// which is precisely the scaling the numpy route does not have (#898).
// Returns the (n, 6) table in the members' own order, so the caller
// scatters with the same offsets.
static py::array_t<std::complex<double>> near_interface_six_columns(
    double k_p, std::complex<double> k_m,
    py::array_t<double, py::array::c_style | py::array::forcecast> rho,
    py::array_t<py::ssize_t, py::array::c_style | py::array::forcecast> offsets,
    py::array_t<double, py::array::c_style | py::array::forcecast> z,
    py::array_t<double, py::array::c_style | py::array::forcecast> zp,
    double lam_mult, int p, double detour, int n_threads,
    py::array_t<double, py::array::c_style | py::array::forcecast> gx,
    py::array_t<double, py::array::c_style | py::array::forcecast> gw) {
    if (rho.ndim() != 1 || offsets.ndim() != 1 || z.ndim() != 1 ||
        zp.ndim() != 1)
        throw std::invalid_argument("rho, offsets, z and zp must be 1-D");
    auto rb = rho.unchecked<1>();
    auto ob = offsets.unchecked<1>();
    auto zb = z.unchecked<1>();
    auto pb = zp.unchecked<1>();
    const py::ssize_t nc = rb.shape(0);
    const py::ssize_t n = zb.shape(0);
    if (ob.shape(0) != nc + 1)
        throw std::invalid_argument("offsets must have len(rho) + 1 entries");
    if (pb.shape(0) != n)
        throw std::invalid_argument("z and zp must have the same length");
    if (ob(0) != 0 || ob(nc) != n)
        throw std::invalid_argument("offsets must span 0 .. len(z)");
    for (py::ssize_t c = 0; c < nc; ++c)
        if (ob(c + 1) < ob(c))
            throw std::invalid_argument("offsets must be non-decreasing");
    if (gx.ndim() != 1 || gw.ndim() != 1 || gx.shape(0) != gw.shape(0) ||
        gx.shape(0) < 1)
        throw std::invalid_argument("bad Gauss rule");
    // `p` is the resolution ladder, not a size knob: 2^p panels per seeded
    // interval. Bounded so a typo asks for a refusal rather than a terabyte.
    if (p < 0 || p > 20) throw std::invalid_argument("column p out of range");
    const double *gxp = gx.data();
    const double *gwp = gw.data();
    const int ng = static_cast<int>(gx.shape(0));

    // The Python walk's domain raises, BEFORE the parallel region (a throw
    // cannot cross an omp boundary) and before any column is built — a bad
    // member anywhere refuses the whole call, as `six_columns` does.
    for (py::ssize_t c = 0; c < nc; ++c) {
        for (py::ssize_t i = ob(c); i < ob(c + 1); ++i) {
            if (!(zb(i) >= 0.0 && pb(i) <= 0.0))
                throw std::invalid_argument("need z >= 0 >= zp");
            if (rb(c) < 0.0 || (zb(i) - pb(i)) + rb(c) <= 0.0)
                throw std::invalid_argument("need R > 0");
        }
    }

    py::array_t<std::complex<double>> vals({n, py::ssize_t(6)});
    auto vb = vals.mutable_unchecked<2>();
    const mw_contour::cd kpc(k_p, 0.0);
    const mw_contour::cd km(k_m);

    // The thread count is the CALLER's policy, held to what OpenMP would
    // have used anyway so a pin below it survives: `_near_interface` passes
    // the PHYSICAL core count, for the reason #898 measured on the numpy
    // route's gemm and this kernel repeats — libmvec exp/sincos saturates a
    // core's FPU, so the hyperthread siblings contend rather than add. On
    // this 4c/8t box the BLE N = 4 kernel reads 95 ms at 1 thread, 40 ms at
    // 4 and 46 ms at 8.
    int nt = 1;
#ifdef _OPENMP
    nt = omp_get_max_threads();
    if (n_threads > 0) nt = std::min(nt, n_threads);
    nt = std::max(nt, 1);
#endif

    // TWO parallel units, because a fill's columns differ by three orders in
    // member count and one unit cannot serve both. Measured on BLE 45 ft
    // N = 4 (the #899 census): 84 columns over four `designed_tables` calls,
    // and the largest call is one column of 3,024 members beside forty of
    // 108. Parallelising over columns alone left that one column — 40 % of
    // the call's members — on a single thread, and the kernel measured 1.0x
    // from one thread to four.
    //
    // The split is a FAIR SHARE, not a fixed size: a column carrying more
    // members than one thread's share of the whole call cannot be a unit,
    // because no schedule can divide it. Everything else stays a column,
    // which matters in the other direction — those forty 108-member columns
    // are better parallelised as columns than as forty serial rule builds.
    // At one thread the share is the whole call and nothing splits, which is
    // the right answer there too.
    //
    //   * by_column: `dynamic`, since K grows with rho (one panel per J0
    //     oscillation) and a fill's columns arrive in geometry order;
    //   * by_member: the rule built once in front of the loop, serial, then
    //     every member of it on all threads.
    const py::ssize_t share = (n + nt - 1) / nt;
    std::vector<py::ssize_t> by_column, by_member;
    for (py::ssize_t c = 0; c < nc; ++c) {
        if (ob(c + 1) == ob(c)) continue;
        (ob(c + 1) - ob(c) > share ? by_member : by_column).push_back(c);
    }
    const py::ssize_t n_col = static_cast<py::ssize_t>(by_column.size());
    const py::ssize_t n_mem = static_cast<py::ssize_t>(by_member.size());

    {
        py::gil_scoped_release release;
        #pragma omp parallel for schedule(dynamic) num_threads(nt)
        for (py::ssize_t j = 0; j < n_col; ++j) {
            const py::ssize_t c = by_column[j];
            const py::ssize_t lo = ob(c), hi = ob(c + 1);
            mw899::Scratch s;
            mw899::column_rule(rb(c), k_p, km, mw899::s_min_of(zb, pb, lo, hi),
                               lam_mult, p, detour, gxp, gwp, ng, s);
            mw899::column_factors(kpc, km, s);
            const size_t K = s.lam.size();
            for (py::ssize_t i = lo; i < hi; ++i) {
                mw_contour::cd out[6];
                mw899::column_member(s, K, zb(i), pb(i), out);
                for (int q = 0; q < 6; ++q) vb(i, q) = out[q];
            }
        }
        for (py::ssize_t j = 0; j < n_mem; ++j) {
            const py::ssize_t c = by_member[j];
            const py::ssize_t lo = ob(c), hi = ob(c + 1);
            mw899::Scratch s;
            mw899::column_rule(rb(c), k_p, km, mw899::s_min_of(zb, pb, lo, hi),
                               lam_mult, p, detour, gxp, gwp, ng, s);
            mw899::column_factors(kpc, km, s);
            const size_t K = s.lam.size();
            // `dynamic`: members of one column do NOT cost the same. The
            // rule is converged for the column's smallest s, so a member
            // with a larger s underflows part of its own tail and skips it
            // (see MW_EXP_ZERO) — the exact saving the numpy route cannot
            // take, and it makes the cheap members the deep ones.
            #pragma omp parallel for schedule(dynamic, 32) num_threads(nt)
            for (py::ssize_t i = lo; i < hi; ++i) {
                mw_contour::cd out[6];
                mw899::column_member(s, K, zb(i), pb(i), out);
                for (int q = 0; q < 6; ++q) vb(i, q) = out[q];
            }
        }
    }
    return vals;
}

// ---------------------------------------------------------------------------
// Sheets (momwire#1173 Design E).
//
// A PLANE sheet is one source depth z' = -d: the six kernels are analytic in
// (rho, z) over the whole above half-plane z >= 0, their only nearby singular
// point (rho, z) = (0, z'), at least d away. A HEIGHT sheet (`height`) is its
// mirror, one observer height z = h over (rho, z' <= 0), at least h from its
// singular point (0, h). Either way the table is in (rho, s), s = z - z' the
// vertical separation (s >= d), and the singular point is (0, 0).
//
// `_near_interface.PlaneSheet` lays the table on a tensor grid of cells: rho
// edges and s edges from dyadic strips (each strip at least as far from the
// singular point as it is wide, split to a fraction of the wavelength alive
// there), p x p Chebyshev (first-kind) nodes per cell, each node evaluated by
// the column twin above -- a rho node shares its column setup with every s
// node of its dyadic s strip. Cell (a, b)'s block starts at node cell_off[a,
// b] and is laid out [rho node][s node][6 kernels].
//
// The stored values are f * R^pw with R = hypot(rho, s), pw = 1 for U, V, W
// and 2 for the three derivatives, so the table carries the smooth part and
// the 1/R^pw is put back here, exactly. Interpolation is barycentric (second
// form) in each direction: no Vandermonde, no conditioning question.
//
// Each row is independent and reduces over nothing shared, so the answer does
// not depend on the thread count. The multiply-adds are spelled through
// `mw_fma` (the build passes -ffp-contract=off), so they round as written.
// Only scalar hypot is called: no vector libm, so no symbol newer than the
// Linux wheels' glibc floor (test_glibc_floor.py).
// The widest table `near_interface_grid_sheet` takes (momwire#1221). Six
// kernels today; the headroom is for the point-observer keys #1220 stage 2
// adds.
#define MW_SHEET_MAX_KEYS 16

namespace mw_sheet {

static inline void bary(double t, double lo, double hi, const double *x,
                        const double *bw, int p, double *w) {
    const double xx = (2.0 * t - (lo + hi)) / (hi - lo);
    double sum = 0.0;
    for (int a = 0; a < p; ++a) {
        const double df = xx - x[a];
        if (std::fabs(df) < 1e-15) {  // on a node: its value, exactly
            for (int b = 0; b < p; ++b) w[b] = (a == b) ? 1.0 : 0.0;
            return;
        }
        w[a] = bw[a] / df;
        sum += w[a];
    }
    const double inv = 1.0 / sum;
    for (int a = 0; a < p; ++a) w[a] *= inv;
}

// The cell of t among edges e[0] < ... < e[n], or -1 outside [e[0], e[n]]
// by more than a relative 1e-12 (the ends clamp into the table).
static inline py::ssize_t cell_of(double t, const double *e, py::ssize_t n) {
    const double tol = 1e-12 * std::max(1.0, std::fabs(e[n]));
    if (!(t >= e[0] - tol && t <= e[n] + tol)) return -1;
    py::ssize_t j =
        static_cast<py::ssize_t>(std::upper_bound(e, e + n + 1, t) - e) - 1;
    if (j < 0) j = 0;
    if (j > n - 1) j = n - 1;
    return j;
}

}  // namespace mw_sheet

static void near_interface_grid_sheet(
    py::array_t<double, py::array::c_style | py::array::forcecast> sub,
    py::array_t<py::ssize_t, py::array::c_style | py::array::forcecast> idx,
    double fixed, bool height,
    py::array_t<double, py::array::c_style | py::array::forcecast> rho_edges,
    py::array_t<double, py::array::c_style | py::array::forcecast> s_edges,
    py::array_t<std::int64_t, py::array::c_style | py::array::forcecast>
        cell_off,
    py::array_t<double, py::array::c_style | py::array::forcecast> x,
    py::array_t<double, py::array::c_style | py::array::forcecast> bw,
    py::array_t<std::complex<double>, py::array::c_style | py::array::forcecast>
        vals,
    py::array_t<std::complex<double>, py::array::c_style> out, int n_threads,
    py::array_t<std::int64_t, py::array::c_style | py::array::forcecast> rpow) {
    if (sub.ndim() != 2 || sub.shape(1) != 3)
        throw std::invalid_argument("sub must be (m, 3)");
    const py::ssize_t m = sub.shape(0);
    // The table's WIDTH is the caller's (momwire#1221): one column per
    // kernel, `rpow[k]` the power of R that column was tabulated times. Six
    // today (`_near_interface.KEYS`); a later stage appends keys without
    // touching this loop. Each column is interpolated on its own, so the
    // width moves no bit of any column.
    if (rpow.ndim() != 1 || rpow.shape(0) < 1 || rpow.shape(0) > MW_SHEET_MAX_KEYS)
        throw std::invalid_argument("rpow must be 1-D with 1..16 entries");
    const int nk = static_cast<int>(rpow.shape(0));
    const int nd = 2 * nk;  // interleaved doubles per node
    const std::int64_t *rp = rpow.data();
    for (int k = 0; k < nk; ++k)
        if (rp[k] < 0 || rp[k] > 4)
            throw std::invalid_argument("rpow entries must be in 0..4");
    if (out.ndim() != 2 || out.shape(0) != m || out.shape(1) != nk ||
        !out.writeable())
        throw std::invalid_argument(
            "out must be a writeable (m, len(rpow)) complex array");
    if (idx.ndim() != 1) throw std::invalid_argument("idx must be 1-D");
    if (height ? !(fixed > 0.0) : !(fixed < 0.0))
        throw std::invalid_argument(height ? "a height sheet needs h > 0"
                                           : "a plane sheet needs zp < 0");
    if (rho_edges.ndim() != 1 || s_edges.ndim() != 1 || rho_edges.shape(0) < 2 ||
        s_edges.shape(0) < 2)
        throw std::invalid_argument("edges must be 1-D with at least 2 entries");
    const py::ssize_t nr = rho_edges.shape(0) - 1, ns = s_edges.shape(0) - 1;
    if (cell_off.ndim() != 2 || cell_off.shape(0) != nr || cell_off.shape(1) != ns)
        throw std::invalid_argument("cell_off must be (len(rho_edges) - 1, "
                                    "len(s_edges) - 1)");
    const int p = static_cast<int>(x.shape(0));
    if (p < 1 || p > 32 || bw.shape(0) != p)
        throw std::invalid_argument("bad Chebyshev nodes (1..32)");
    if (vals.ndim() != 2 || vals.shape(1) != nk)
        throw std::invalid_argument("vals must be (nodes, len(rpow))");
    const double *re = rho_edges.data(), *se = s_edges.data();
    for (py::ssize_t j = 0; j < nr; ++j)
        if (!(re[j + 1] > re[j])) throw std::invalid_argument("rho edges must rise");
    for (py::ssize_t j = 0; j < ns; ++j)
        if (!(se[j + 1] > se[j])) throw std::invalid_argument("s edges must rise");
    const std::int64_t *co = cell_off.data();
    const py::ssize_t n_nodes = vals.shape(0);
    const std::int64_t per = static_cast<std::int64_t>(p) * p;
    for (py::ssize_t c = 0; c < nr * ns; ++c)
        if (co[c] < 0 || co[c] + per > n_nodes)
            throw std::invalid_argument("inconsistent sheet cells");
    const py::ssize_t n = idx.shape(0);
    const py::ssize_t *ix = idx.data();
    for (py::ssize_t r = 0; r < n; ++r)
        if (ix[r] < 0 || ix[r] >= m)
            throw std::invalid_argument("idx out of range");
    const double *xp = x.data(), *bwp = bw.data();
    // (nodes, nk) complex as interleaved doubles: node q's kernel k at
    // nd q + 2 k (re) and + 1 (im).
    const double *V = reinterpret_cast<const double *>(vals.data());
    const double *sb = sub.data();
    double *ob = reinterpret_cast<double *>(out.mutable_data());

    int nt = 1;
#ifdef _OPENMP
    nt = omp_get_max_threads();
    if (n_threads > 0) nt = std::min(nt, n_threads);
    nt = std::max(nt, 1);
#endif
    // A row the table cannot answer (off the sheet, on the wrong side, beyond
    // its cells) is flagged in the loop -- a throw cannot cross the omp
    // boundary -- and refused after.
    int bad = 0;
    {
        py::gil_scoped_release release;
        #pragma omp parallel for schedule(static) num_threads(nt) reduction(|:bad)
        for (py::ssize_t r = 0; r < n; ++r) {
            double wr[32], ws[32];
            const py::ssize_t row = ix[r];
            const double rho = sb[row * 3], z = sb[row * 3 + 1];
            const double zq = sb[row * 3 + 2];
            double *o = ob + row * nd;
            const bool on =
                height ? (z == fixed && zq <= 0.0) : (zq == fixed && z >= 0.0);
            const double s = z - zq;
            const py::ssize_t a = on && rho >= 0.0 ? mw_sheet::cell_of(rho, re, nr) : -1;
            const py::ssize_t b = a >= 0 ? mw_sheet::cell_of(s, se, ns) : -1;
            if (b < 0) {
                bad = 1;
                for (int k = 0; k < nd; ++k) o[k] = 0.0;
                continue;
            }
            mw_sheet::bary(rho, re[a], re[a + 1], xp, bwp, p, wr);
            mw_sheet::bary(s, se[b], se[b + 1], xp, bwp, p, ws);
            const double *blk = V + co[a * ns + b] * nd;
            double acc[2 * MW_SHEET_MAX_KEYS] = {0.0};
            for (int i = 0; i < p; ++i) {
                double inner[2 * MW_SHEET_MAX_KEYS] = {0.0};
                const double *Vi = blk + static_cast<std::int64_t>(i) * p * nd;
                for (int j = 0; j < p; ++j) {
                    const double w = ws[j];
                    const double *Vj = Vi + j * nd;
                    for (int k = 0; k < nd; ++k)
                        inner[k] = mw_fma::fma(w, Vj[k], inner[k]);
                }
                const double wi = wr[i];
                for (int k = 0; k < nd; ++k)
                    acc[k] = mw_fma::fma(wi, inner[k], acc[k]);
            }
            // 1/R^pw per column, as a product of 1/R's: pw = 2 is i1 * i1,
            // the spelling the fixed six-column loop used.
            const double i1 = 1.0 / std::hypot(rho, s);
            for (int k = 0; k < nk; ++k) {
                double f = 1.0;
                if (rp[k] > 0) {
                    f = i1;
                    for (std::int64_t e = 1; e < rp[k]; ++e) f *= i1;
                }
                o[2 * k] = acc[2 * k] * f;
                o[2 * k + 1] = acc[2 * k + 1] * f;
            }
        }
    }
    if (bad)
        throw std::invalid_argument(
            "a row lies off the sheet (need z' == the plane's and z >= 0, or on "
            "a height sheet z == its h and z' <= 0; rho >= 0 and (rho, z - z') "
            "inside the table)");
}

// momwire#1032: the module NAME is a build parameter, so the same sources can
// be compiled twice — once with AVX2/FMA and once at the x86-64 baseline — and
// loaded by name at import time on a CPU that can run one but not the other.
// `PYBIND11_MODULE` pastes this token into the init symbol, so a `-D` here is
// what makes `PyInit__accelerators_sse2` exist. The default keeps an
// out-of-tree or single-variant build (macOS, arm64) building under the name
// it has always had.
#ifndef MOMWIRE_MODULE_NAME
#define MOMWIRE_MODULE_NAME _near_interface_accel
#endif

PYBIND11_MODULE(MOMWIRE_MODULE_NAME, m) {
    m.doc() =
        "momwire#680 U2 and #899 item 1: the C++ twins of "
        "_near_interface.six_point and _near_interface.six_columns. "
        "Optional; _near_interface falls back to the numpy walks without it.";
    // The capability flag `_near_interface._HAVE_NEAR_INTERFACE_ACCEL`
    // keys on. Its OWN name (not contour_engine_568 / below_fills_568): a
    // .so built at an earlier arc exports those but not this entry, and a
    // shared flag would claim a contract it cannot serve.
    m.attr("near_interface_680") = true;
    // The column twin's flag, and for the same reason its OWN name: a .so
    // built between #680 and #899 exports `near_interface_680` and not this
    // one, so the two entries have to be asked about separately.
    m.attr("near_interface_columns_899") = true;
    m.def("near_interface_six_batch", &near_interface_six_batch,
          py::arg("k_p"), py::arg("k_m"), py::arg("rho"), py::arg("z"),
          py::arg("zp"), py::arg("rtol"), py::arg("lam_mult"),
          py::arg("depth"), py::arg("detour"), py::arg("gx"), py::arg("gw"),
          "six_point over parallel (rho, z, zp) arrays -> (n, 6) complex. "
          "k_p/k_m arrive derived (k_medium keeps the branch choice); the "
          "U1 memo layer in Python hands this UNIQUE triples only.");
    m.def("near_interface_six_columns", &near_interface_six_columns,
          py::arg("k_p"), py::arg("k_m"), py::arg("rho"), py::arg("offsets"),
          py::arg("z"), py::arg("zp"), py::arg("lam_mult"), py::arg("p"),
          py::arg("detour"), py::arg("n_threads"), py::arg("gx"),
          py::arg("gw"),
          "six_columns over CONCATENATED columns -> (n, 6) complex. Column c "
          "owns members offsets[c]..offsets[c+1], all at rho[c]; the answer "
          "keeps the members' order. Parallel over columns, and over the "
          "MEMBERS of any column too big to be a unit, so a whole fill's "
          "grouping belongs in ONE call. `n_threads` <= 0 takes OpenMP's own "
          "count and is never raised above it. No rtol: the fixed rule's "
          "resolution is `p` and its extents the e^{-60} dead range.");
    // The plane sheet's flag (momwire#1173 Design E), its own name for the
    // same reason as the two above.
    // Design E's sheets, on the strip grid (phase 2): their own flag, for the
    // same reason as the two above.
    m.attr("grid_sheet_1173") = true;
    // momwire#1221: the sheet takes its width (and each column's R power)
    // from the caller. A distinct flag, so a stale build is refused by name
    // rather than handed an argument it does not know.
    m.attr("grid_sheet_width_1221") = true;
    m.def("near_interface_grid_sheet", &near_interface_grid_sheet,
          py::arg("sub"), py::arg("idx"), py::arg("fixed"), py::arg("height"),
          py::arg("rho_edges"), py::arg("s_edges"), py::arg("cell_off"),
          py::arg("x"), py::arg("bw"), py::arg("vals"), py::arg("out"),
          py::arg("n_threads"), py::arg("rpow"),
          "Interpolate the rows sub[idx] (rho, z, zp) of one sheet -- the plane "
          "z' = fixed < 0, or with height=True the height z = fixed > 0 -- "
          "from a _near_interface.PlaneSheet strip grid into out[idx] ((m, "
          "len(rpow)) complex, KEYS order; column k was tabulated times "
          "R^rpow[k]). Row-parallel; the answer does not depend on the "
          "thread count.");
}
