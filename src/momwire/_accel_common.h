// Shared preamble for the split _accelerators translation units (momwire#687).
// Generated from the monolith's lines 1-120; contents byte-identical.
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
#include <iostream>
#include <tuple>
#include <vector>

#include "_bspline_static_moments_inline.h"
// The extended-thin-wire static correction D_pq^EK (momwire#249's codegen,
// wired in by #270 unit 1). Pulls in the J header itself; the duplicate
// include above is harmless (`#pragma once`) and kept for legibility.
#include "_bspline_ek_moments_inline.h"
// The shared adaptive-contour engine (momwire#568 unit 1): the C++ twin of
// `_sommerfeld_below`'s head + tail machinery, templated on the integrand.
// Header-only, reentrant, allocation-free -- U2/U3 instantiate it with their
// own integrands under OpenMP with the GIL released. The bindings at the end
// of this file are TEST instantiations only; nothing in the production Python
// dispatch calls them.
#include "_contour_engine_inline.h"

namespace py = pybind11;

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
