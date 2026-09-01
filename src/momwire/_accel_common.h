// Shared preamble for the split _accelerators translation units (momwire#687).
// Generated from the monolith's lines 1-120, then slimmed by the #710 review:
// the kernel headers (`_bspline_*_moments_inline.h`, `_contour_engine_inline.h`)
// moved into the TUs that use them, so editing one kernel no longer
// ccache-misses every TU — per-TU staleness is the split's whole point.
#pragma once

// M_PI is not in the C++ standard. GCC/glibc define it unconditionally, but
// MSVC only exposes it when _USE_MATH_DEFINES is set *before* the first math
// header is pulled in (directly or transitively via pybind11). Must stay at
// the very top of the file.
#define _USE_MATH_DEFINES

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <complex>

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <tuple>
#include <vector>

namespace py = pybind11;

// The per-section registration seam. Declared here so every section TU
// compiles its definition against the same prototype the module TU calls —
// a drift would otherwise surface only as an undefined-symbol link error
// (or, for a forgotten call, not at all).
void register_bspline(py::module_ &m);
void register_sinusoidal(py::module_ &m);
void register_somm(py::module_ &m);
void register_mw568(py::module_ &m);
void register_razor(py::module_ &m);

// Ubuntu/glibc <cmath> headers don't carry `omp declare simd` markers for the
// libmvec routines, so GCC's auto-vectorizer can't substitute the vectorized
// `_ZGVdN4v_sin` / `_ZGVdN4v_cos` (AVX2, 4 doubles) inside an `omp simd` loop
// without these explicit declarations. The std::cos / std::sin overloads in
// <cmath> still resolve to these underlying extern-C symbols, so the rest of
// the file's calls pick up the simd-vectorized form for free once the linker
// has libmvec available (-lmvec in setup.py).
//
// Gated to GNU-compatible, non-MSVC compilers: this trick targets glibc's
// libmvec specifically. MSVC has no libmvec and would choke on redeclaring the
// CRT's cos/sin; there the sincos calls stay scalar/autovectorized. macOS is
// also excluded: Apple clang defines __GNUC__ but there is no libmvec on
// macOS (and arm64 has no AVX2 simdlen(4) form either), so the sincos calls
// stay scalar/NEON-autovectorized there too.
#if defined(__GNUC__) && !defined(_MSC_VER) && !defined(__APPLE__)
#pragma omp declare simd notinbranch simdlen(4)
extern "C" double cos(double);

#pragma omp declare simd notinbranch simdlen(4)
extern "C" double sin(double);
#endif

// The B-spline pair kernels' qr blocking width, in ONE place.
//
// The scratch arrays are `[64]` (and `wuwu` is `[NMM * 64]`) because 64 doubles
// keeps a pair block in L1 — a cache-blocking constant, not an arbitrary limit.
// It used to double as a CEILING on n_qp, since n_qp*n_qp had to fit; six
// kernels threw above it and the crossing/lossy-soil class that needs high
// order could not be computed on the fast path at all (momwire#760).
//
// momwire#762 made it a tile width instead. Growing the buffers was the wrong
// fix and is measured as such in that issue: at n_qp^2 = 4096 `wuwu` alone is
// 295 KB, an order of magnitude past L1, and the fixed sizes are what the full
// unroll and PYSIM_OMP_SIMD vectorization depend on. Nothing carries state
// across quadrature pairs, so walking qr in chunks of this width is exact —
// and at n_qp <= 8 it is ONE chunk, so the arithmetic and its order are
// unchanged and the output is bit-identical to the untiled kernel.
constexpr size_t BSPLINE_QR_TILE = 64;

// The ceiling the Python routing guard reads (momwire#769). Since momwire#762
// the pair kernels TILE the qr range, so there is no ceiling any more and this
// says so: the guard's `n_qp <= MAX_N_QP` is then always true and no fill is
// diverted. The constant and the export stay because the guard, its warning and
// its drift test are the safety net for the NEXT kernel limit, and because
// removing them would leave the routing decision spelled in two places again.
// The OFF-EDGE pair kernels' ceiling, read by the Python routing guard
// (momwire#769). All six are tiled as of momwire#762, so there is no ceiling
// left and this says so: the guard's `n_qp <= MAX_N_QP` is then always true
// and no off-edge fill is diverted. The constant and its export stay because
// the guard, its warning and its drift test are the safety net for the next
// kernel limit — and because the SAME-EDGE reg-moment kernel below still has
// one.
constexpr size_t BSPLINE_MAX_N_QP = static_cast<size_t>(-1);

// The SAME-EDGE reg-moment kernel's ceiling, which momwire#762 did NOT lift.
// That kernel takes R from a precomputed (N*n_qp, N*n_qp) array and reduces
// through a nested wu(p,i,q)*wu(P,j,r) product rather than a flat wuwu row, so
// it needs its own transformation rather than the one the six share. Until
// then `n_qp_pair_same_edge` above this routes to numpy instead of raising —
// it used to raise, with wording ("n_qp^2 must be <= 64") different enough
// from the off-edge kernels' that momwire#769 missed the site entirely.
constexpr size_t BSPLINE_SAME_EDGE_MAX_N_QP = 8;

// Stringizing helpers so a token sequence can be turned into a _Pragma.
#define PYSIM_STR_(x) #x
#define PYSIM_PRAGMA_(x) _Pragma(PYSIM_STR_(x))

// The (i, j)-grid parallel loops below use `collapse(2)` for finer load
// balancing. We keep the Windows OpenMP usage conservative and drop it to plain
// outer-loop parallelism under MSVC — same results, only coarser scheduling
// across the grid. GCC (and MSVC's /openmp:llvm) keeps collapse(2).
#if defined(_MSC_VER)
#  define PYSIM_OMP_PARALLEL_FOR_COLLAPSE2 _Pragma("omp parallel for schedule(static)")
#else
#  define PYSIM_OMP_PARALLEL_FOR_COLLAPSE2 \
       _Pragma("omp parallel for collapse(2) schedule(static)")
#endif

// `omp simd` neutralization for MSVC. MSVC's /openmp:llvm (which we build with,
// for its collapse + unsigned-index support) rejects the `omp simd` directive
// outright, and /openmp:experimental silently drops simd `reduction` clauses —
// a correctness hazard. Turn the simd directives into no-ops under MSVC and let
// /arch:AVX2 autovectorize the inner loops as correct scalar-reduction code.
// GCC keeps real `omp simd` (bound to libmvec via the declare-simd block above).
// Reduction-clause commas are protected by the inner parens, so they pass as a
// single macro argument.
#if defined(_MSC_VER)
#  define PYSIM_OMP_SIMD(clauses)
#else
#  define PYSIM_OMP_SIMD(clauses) PYSIM_PRAGMA_(omp simd clauses)
#endif

// --------------------------------------------------------------------------
// Cooperative cancellation (Phase 2).
//
// Long kernels take a trailing `uintptr_t cancel_flag` (0 = no cancellation).
// It is the raw address of a CancelToken's int32 flag on the Python side. A
// monotonic 0 -> 1 write from any thread means "abort"; we read it with a
// `volatile` load per outer loop iteration (naturally-aligned int32, single
// writer -> safe in practice, and avoids the C++20 std::atomic_ref / Apple
// Clang libc++ support gap that would jeopardise the macOS wheel).
//
// An exception must never escape an OpenMP parallel region (UB), so we use the
// standard drain pattern: on cancel, set a shared atomic<bool> and `continue`
// so every remaining iteration becomes a no-op; the loop finishes normally and
// we throw AFTER it. AbortedError is registered as the Python exception
// `AcceleratorAborted`, which the `_accel.py` wrappers remap to SolveAborted.
struct AbortedError : std::exception {
    const char *what() const noexcept override { return "accelerator solve aborted"; }
};

#define PYSIM_CANCEL_SETUP(flag_addr)                                          \
    const volatile int32_t *pysim_cancel =                                     \
        reinterpret_cast<const volatile int32_t *>(flag_addr);                 \
    std::atomic<bool> pysim_aborted { false }

// Drain-poll. Place at the very top of a parallel loop body; `continue` targets
// the enclosing for-loop by design, so use only inside a braced loop body.
#define PYSIM_CANCEL_POLL()                                                    \
    if (pysim_aborted.load(std::memory_order_relaxed)) continue;              \
    if (pysim_cancel && *pysim_cancel) {                                       \
        pysim_aborted.store(true, std::memory_order_relaxed);                  \
        continue;                                                              \
    }

#define PYSIM_THROW_IF_ABORTED()                                               \
    if (pysim_aborted.load(std::memory_order_relaxed)) throw AbortedError {}
