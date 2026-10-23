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
// the quiet-panel rule above. This is why the module was the first one
// compiled at C++17; `_accelerators` has since followed.
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
#include <deque>
#include <limits>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <unordered_map>
#include <vector>

#include "_lanes.h"

#ifdef _OPENMP
#include <omp.h>
#endif


namespace py = pybind11;

#include "_near_interface_core.h"

// Cooperative cancellation (momwire#1348 scope), the `_accel_common.h`
// convention spelled locally: this companion does not include that header,
// and its exception is its OWN type, registered on this module, so no
// C++ type has to be matched across two shared objects. `_accel` remaps
// this module's `AcceleratorAborted` to `momwire.SolveAborted` exactly as it
// does `_accelerators`'. `cancel_flag` is a CancelToken's raw int32 address
// (0 = none), read with a volatile load per work item; inside a parallel
// region the drain pattern (no exception may cross an omp boundary).
#include <atomic>
namespace {
using NiAborted = ni_core::Aborted;
}  // namespace
#define NI_CANCEL_SETUP(flag_addr)                                          \
    const volatile int32_t *ni_cancel =                                        \
        reinterpret_cast<const volatile int32_t *>(flag_addr);                 \
    std::atomic<bool> ni_aborted { false }
#define NI_CANCEL_POLL()                                                    \
    if (ni_aborted.load(std::memory_order_relaxed)) continue;                  \
    if (ni_cancel && *ni_cancel) {                                             \
        ni_aborted.store(true, std::memory_order_relaxed);                     \
        continue;                                                              \
    }
#define NI_THROW_IF_ABORTED()                                               \
    if (ni_aborted.load(std::memory_order_relaxed) || (ni_cancel && *ni_cancel)) \
        throw NiAborted {}

// The column rule's head and mid, shared between columns of one rho.
//
// A column's rule is head + mid + tail (`column_rule`). The head and mid
// nodes, and so their factors, are a function of rho and the extents
// (a_head, lam_top) alone; the column's smallest s reaches them only through
// the extents, and on every column the far-pair kill cap leaves alone the
// extents are the same for every s. A crossing fill asks for the same rho in
// call after call, each time for a different few members (its memo hands the
// twin only fresh triples): on the bs2 inverted-L x16 fill, 20,772 columns
// over 328 calls carried 6,520 distinct rho, 2.3 members a column, and the
// head and mid (J0 at 465 nodes, and their factors) were 28 % of the
// kernel's time on one thread.
//
// So the fill holds one of these (on its SheetPlan) and the twin keeps the
// head + mid FACTORS here, keyed by the exact bits of (rho, a_head, lam_top),
// for the context they were built in (k_p, k_m, p, detour and the Gauss
// rule; a different context empties it). A hit copies the arrays
// `column_factors` would have written for those nodes; the tail is built as
// before; `column_factors_joined` lays the two out exactly as
// `column_factors` lays out the whole rule. Every float a member reads is
// therefore the float it read before -- the cache moves when a value is
// computed, never what it is -- and the gate is uint64 equality against the
// uncached call (tests/test_near_interface_column_cache.py).
//
// Bounded: entries are dropped oldest first past `max_bytes`. A miss costs
// what the twin cost before, so the bound trades memory for hits, never
// bits. Thread-safe: lookups and inserts take the mutex, and the twin does
// both outside its parallel regions.

static py::ssize_t near_interface_hankel_conj_mismatches(
    py::array_t<std::complex<double>, py::array::c_style | py::array::forcecast> w) {
    return ni_core::hankel_conj_mismatches(w.data(), w.size());
}

static py::array_t<std::complex<double>> near_interface_six_batch(
    double k_p, std::complex<double> k_m,
    py::array_t<double, py::array::c_style | py::array::forcecast> rho,
    py::array_t<double, py::array::c_style | py::array::forcecast> z,
    py::array_t<double, py::array::c_style | py::array::forcecast> zp,
    double rtol, double lam_mult, int depth, double detour,
    py::array_t<double, py::array::c_style | py::array::forcecast> gx,
    py::array_t<double, py::array::c_style | py::array::forcecast> gw) {
    const py::ssize_t n = rho.shape(0);
    if (z.shape(0) != n || zp.shape(0) != n)
        throw std::invalid_argument("rho, z and zp must have the same length");
    if (gx.ndim() != 1 || gw.ndim() != 1 || gx.shape(0) != gw.shape(0) ||
        gx.shape(0) < 1)
        throw std::invalid_argument("bad Gauss rule");
    py::array_t<std::complex<double>> vals({n, py::ssize_t(6)});
    {
        py::gil_scoped_release release;
        ni_core::six_batch(k_p, k_m, rho.data(), z.data(), zp.data(), n, rtol, lam_mult, depth,
                           detour, gx.data(), gw.data(), static_cast<int>(gx.shape(0)),
                           vals.mutable_data());
    }
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
    py::array_t<double, py::array::c_style | py::array::forcecast> gw,
    uintptr_t cancel_flag = 0, py::object cache = py::none(), bool lanes = true,
    bool sdp = true) {
    if (rho.ndim() != 1 || offsets.ndim() != 1 || z.ndim() != 1 ||
        zp.ndim() != 1)
        throw std::invalid_argument("rho, offsets, z and zp must be 1-D");
    const py::ssize_t nc = rho.shape(0);
    const py::ssize_t n = z.shape(0);
    if (offsets.shape(0) != nc + 1)
        throw std::invalid_argument("offsets must have len(rho) + 1 entries");
    if (zp.shape(0) != n)
        throw std::invalid_argument("z and zp must have the same length");
    if (gx.ndim() != 1 || gw.ndim() != 1 || gx.shape(0) != gw.shape(0) ||
        gx.shape(0) < 1)
        throw std::invalid_argument("bad Gauss rule");
    if (p < 0 || p > 20) throw std::invalid_argument("column p out of range");
    ni_core::ColumnCache *cc =
        cache.is_none() ? nullptr : cache.cast<ni_core::ColumnCache *>();
    py::array_t<std::complex<double>> vals({n, py::ssize_t(6)});
    {
        // The kernel's own checks (offsets, members) run before any column;
        // the GIL is released for the columns.
        py::gil_scoped_release release;
        ni_core::six_columns(k_p, k_m, rho.data(), offsets.data(), nc, z.data(), zp.data(), n,
                             lam_mult, p, detour, n_threads, gx.data(), gw.data(),
                             static_cast<int>(gx.shape(0)), cancel_flag, cc,
                             vals.mutable_data(), lanes, sdp);
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
    py::array_t<std::int64_t, py::array::c_style | py::array::forcecast> rpow,
    bool generic) {
    const bool fixed_ok = !generic;
    (void)fixed_ok;  // read only where the fixed route is built
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
    {
        py::gil_scoped_release release;
        ni_core::grid_sheet(sub.data(), m, ix, n, fixed, height, re, nr, se, ns, co, x.data(),
                            bw.data(), p, vals.data(), out.mutable_data(), n_threads, rp, nk,
                            generic);
    }
}

// `ConeSheet.interpolate` (momwire#1595 option A): out[r] = one cone
// surface's kernels at u[r] and height zs[zi[r]] (`ni_core::cone_line`).
static void near_interface_cone_line(
    py::array_t<double, py::array::c_style | py::array::forcecast> u,
    py::array_t<std::int64_t, py::array::c_style | py::array::forcecast> zi,
    py::array_t<double, py::array::c_style | py::array::forcecast> zs, double d,
    py::array_t<double, py::array::c_style | py::array::forcecast> u_edges,
    py::array_t<std::int64_t, py::array::c_style | py::array::forcecast> cell_off,
    py::array_t<double, py::array::c_style | py::array::forcecast> x,
    py::array_t<double, py::array::c_style | py::array::forcecast> bw,
    py::array_t<std::complex<double>, py::array::c_style | py::array::forcecast> vals,
    py::array_t<std::complex<double>, py::array::c_style> out, int n_threads,
    py::array_t<std::int64_t, py::array::c_style | py::array::forcecast> rpow) {
    if (u.ndim() != 1 || zi.ndim() != 1 || zi.shape(0) != u.shape(0))
        throw std::invalid_argument("u and zi must be 1-D of one length");
    const py::ssize_t n = u.shape(0);
    if (rpow.ndim() != 1 || rpow.shape(0) < 1 || rpow.shape(0) > MW_SHEET_MAX_KEYS)
        throw std::invalid_argument("rpow must be 1-D with 1..16 entries");
    const int nk = static_cast<int>(rpow.shape(0));
    for (int k = 0; k < nk; ++k)
        if (rpow.data()[k] < 0 || rpow.data()[k] > 4)
            throw std::invalid_argument("rpow entries must be in 0..4");
    if (out.ndim() != 2 || out.shape(0) != n || out.shape(1) != nk || !out.writeable())
        throw std::invalid_argument("out must be a writeable (n, len(rpow)) complex array");
    if (zs.ndim() != 1 || zs.shape(0) < 1) throw std::invalid_argument("zs must be 1-D");
    const py::ssize_t n_z = zs.shape(0);
    if (u_edges.ndim() != 1 || u_edges.shape(0) < 2)
        throw std::invalid_argument("edges must be 1-D with at least 2 entries");
    const py::ssize_t nu = u_edges.shape(0) - 1;
    if (cell_off.ndim() != 1 || cell_off.shape(0) != nu)
        throw std::invalid_argument("cell_off must be (len(u_edges) - 1,)");
    const int p = static_cast<int>(x.shape(0));
    if (p < 1 || p > 32 || bw.shape(0) != p)
        throw std::invalid_argument("bad Chebyshev nodes (1..32)");
    if (vals.ndim() != 2 || vals.shape(1) != nk || vals.shape(0) % n_z != 0)
        throw std::invalid_argument("vals must be (nodes * len(zs), len(rpow))");
    const py::ssize_t n_nodes = vals.shape(0) / n_z;
    for (py::ssize_t c = 0; c < nu; ++c)
        if (cell_off.data()[c] < 0 || cell_off.data()[c] + p > n_nodes)
            throw std::invalid_argument("inconsistent cone sheet cells");
    py::gil_scoped_release release;
    ni_core::cone_line(u.data(), zi.data(), n, zs.data(), n_z, d, u_edges.data(), nu,
                       cell_off.data(), x.data(), bw.data(), p, vals.data(), out.mutable_data(),
                       n_threads, rpow.data(), nk);
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
    py::register_exception<ni_core::Aborted>(m, "AcceleratorAborted");
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
    m.def("near_interface_hankel_conj_mismatches",
          &near_interface_hankel_conj_mismatches, py::arg("w"),
          "How many w break H2_0(conj w) == conj(H1_0(w)) as uint64 -- the "
          "identity the column tail's down-ray rests on (TEST-ONLY).");
    // The fill-held head + mid cache of the column twin (`mw899::ColumnCache`).
    py::class_<ni_core::ColumnCache>(m, "ColumnCache")
        .def(py::init<size_t>(), py::arg("max_bytes"))
        .def_property_readonly("hits", &ni_core::ColumnCache::hits)
        .def_property_readonly("misses", &ni_core::ColumnCache::misses)
        .def_property_readonly("evictions", &ni_core::ColumnCache::evictions)
        .def_property_readonly("entries", &ni_core::ColumnCache::entries)
        .def_property_readonly("bytes", &ni_core::ColumnCache::bytes)
        .def_property_readonly("max_bytes", &ni_core::ColumnCache::max_bytes);
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
          py::arg("gw"), py::arg("cancel_flag") = 0,
          py::arg("cache") = py::none(), py::arg("lanes") = true,
          py::arg("sdp") = true,
          "six_columns over CONCATENATED columns -> (n, 6) complex. Column c "
          "owns members offsets[c]..offsets[c+1], all at rho[c]; the answer "
          "keeps the members' order. Parallel over columns, and over the "
          "MEMBERS of any column too big to be a unit, so a whole fill's "
          "grouping belongs in ONE call. `n_threads` <= 0 takes OpenMP's own "
          "count and is never raised above it. No rtol: the fixed rule's "
          "resolution is `p` and its extents the e^{-60} dead range.");
    // momwire#1441 step 3: the rule's Bessel pairs and factors run four nodes
    // at a time; `lanes=False` is the scalar walk, gated bit for bit.
    m.attr("six_columns_lanes_1441") = true;
    // momwire#1595 D1b: far columns on the steepest-descent rule; `sdp=False`
    // is the head / mid / tail rule everywhere.
    m.attr("six_columns_sdp_1595") = true;
    // The plane sheet's flag (momwire#1173 Design E), its own name for the
    // same reason as the two above.
    // Design E's sheets, on the strip grid (phase 2): their own flag, for the
    // same reason as the two above.
    m.attr("grid_sheet_1173") = true;
    // momwire#1221: the sheet takes its width (and each column's R power)
    // from the caller. A distinct flag, so a stale build is refused by name
    // rather than handed an argument it does not know.
    m.attr("grid_sheet_width_1221") = true;
    // momwire#1595 option A: the cone line sheets' interpolator.
    m.attr("cone_line_1595") = true;
    m.def("near_interface_cone_line", &near_interface_cone_line, py::arg("u"), py::arg("zi"),
          py::arg("zs"), py::arg("d"), py::arg("u_edges"), py::arg("cell_off"), py::arg("x"),
          py::arg("bw"), py::arg("vals"), py::arg("out"), py::arg("n_threads"), py::arg("rpow"),
          "Interpolate one cone sheet surface (momwire#1595) at rows (u, height index).");
    // momwire#1290: whether this build has the fixed-width cell sums (the
    // AVX2 variant does); either way `generic=True` reaches the width-generic
    // loop they are gated against.
    m.attr("grid_sheet_fixed_1290") = ni_core::grid_sheet_fixed();
    m.def("near_interface_grid_sheet", &near_interface_grid_sheet,
          py::arg("sub"), py::arg("idx"), py::arg("fixed"), py::arg("height"),
          py::arg("rho_edges"), py::arg("s_edges"), py::arg("cell_off"),
          py::arg("x"), py::arg("bw"), py::arg("vals"), py::arg("out"),
          py::arg("n_threads"), py::arg("rpow"), py::arg("generic") = false,
          "Interpolate the rows sub[idx] (rho, z, zp) of one sheet -- the plane "
          "z' = fixed < 0, or with height=True the height z = fixed > 0 -- "
          "from a _near_interface.PlaneSheet strip grid into out[idx] ((m, "
          "len(rpow)) complex, KEYS order; column k was tabulated times "
          "R^rpow[k]). Row-parallel; the answer does not depend on the "
          "thread count. `generic` (TEST-ONLY) takes the width-generic cell "
          "sum, the reference the fixed-width ones are gated against.");
}
