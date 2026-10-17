// The Python-free half of the accelerator preamble. Everything here compiles
// WITHOUT pybind11 or Python.h, so a kernel TU that includes only this header
// (a "core" TU, e.g. `_razor_core.cpp`) can be linked into a standalone C++
// program as well as into the extension. tests/test_razor_core_standalone.py
// holds that to be true.
//
// `_accel_common.h` includes this after pybind11, at the position its content
// used to occupy, so every TU that is not split yet sees the same
// declarations in the same order.
#pragma once

// See `_accel_common.h`: MSVC exposes M_PI only with this set before the
// first math header.
#ifndef _USE_MATH_DEFINES
#define _USE_MATH_DEFINES
#endif

#include <complex>

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <initializer_list>
#include <stdexcept>
#include <string>
#include <tuple>
#include <type_traits>
#include <vector>

// Explicit fused multiply-adds, since the build no longer contracts on its
// own (momwire#1194).
#include "_fma_inline.h"

namespace mw {

// A non-owning N-d view over memory someone else owns, addressed through BYTE
// strides exactly as pybind11's `unchecked<N>()` proxy is, so a kernel body
// written against `x(i, j)` and `x.shape(k)` is textually unchanged when its
// argument type changes from the proxy to this. It never allocates, never
// copies and never checks bounds; the caller (a binding, or a C++ caller that
// owns the arrays) states shape and strides once. It keeps nothing alive: a
// core function must not hold a view past its return.
//
// A view promises only alignof(T). Aligned SIMD loads and stores stay on
// kernel-owned `alignas(32)` scratch; caller memory goes through `loadu`.
template <class T, int N>
struct view {
    T *ptr = nullptr;
    ptrdiff_t n[N > 0 ? N : 1] = {};
    ptrdiff_t st[N > 0 ? N : 1] = {};  // bytes

    view() = default;
    view(T *p, const ptrdiff_t *shape, const ptrdiff_t *strides_bytes) : ptr(p) {
        for (int k = 0; k < N; k++) {
            n[k] = shape[k];
            st[k] = strides_bytes[k];
        }
    }
    // A writable view reads as a read-only one.
    template <class U, class = typename std::enable_if<
                           std::is_same<const U, T>::value &&
                           !std::is_same<U, T>::value>::type>
    view(const view<U, N> &w) : view(w.ptr, w.n, w.st) {}
    // C-contiguous over `shape`.
    static view contiguous(T *p, std::initializer_list<ptrdiff_t> shape) {
        view v;
        v.ptr = p;
        int k = 0;
        for (ptrdiff_t s : shape) v.n[k++] = s;
        ptrdiff_t acc = static_cast<ptrdiff_t>(sizeof(T));
        for (int j = N - 1; j >= 0; j--) {
            v.st[j] = acc;
            acc *= v.n[j];
        }
        return v;
    }

    ptrdiff_t shape(int k) const { return n[k]; }
    ptrdiff_t ndim() const { return N; }
    ptrdiff_t size() const {
        ptrdiff_t s = 1;
        for (int k = 0; k < N; k++) s *= n[k];
        return s;
    }
    T *data() const { return ptr; }
    bool c_contiguous() const {
        if (size() == 0) return true;
        ptrdiff_t acc = static_cast<ptrdiff_t>(sizeof(T));
        for (int j = N - 1; j >= 0; j--) {
            if (n[j] != 1 && st[j] != acc) return false;
            acc *= n[j];
        }
        return true;
    }

    // The offset is spelled per rank, the way pybind11's proxy spells it
    // (index times byte stride, summed, then added to the base), and not as a
    // loop over the rank. The spelling is part of the codegen contract: with
    // a loop, GCC vectorised razor's T1 assembler differently, 162 FP ops
    // where this spelling gives the proxy's 43, and the call ran 4.5 %
    // slower (same bits). A change here is gated like a kernel change:
    // scripts/core_split/objdiff.py and the paired benchmark.
    T &operator()(ptrdiff_t i) const { return at(i * st[0]); }
    T &operator()(ptrdiff_t i, ptrdiff_t j) const { return at(i * st[0] + j * st[1]); }
    T &operator()(ptrdiff_t i, ptrdiff_t j, ptrdiff_t k) const {
        return at(i * st[0] + (j * st[1] + k * st[2]));
    }

  private:
    T &at(ptrdiff_t byte_offset) const {
        typedef typename std::remove_const<T>::type U;
        return *reinterpret_cast<T *>(
            reinterpret_cast<char *>(const_cast<U *>(ptr)) + byte_offset);
    }
};

// A core kernel that reads a view's elements as one run, through data(),
// says so with this; a binding's c_style arrays always pass.
template <class T, int N>
inline void require_contiguous(const view<T, N> &v, const char *who, const char *what) {
    if (!v.c_contiguous())
        throw std::invalid_argument(std::string(who) + ": " + what + " must be C-contiguous");
}

using cd = std::complex<double>;

}  // namespace mw

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
#if defined(MW_STUDY_SSE2_MVEC) && !defined(__AVX2__)
#pragma omp declare simd notinbranch simdlen(2)
#else
#pragma omp declare simd notinbranch simdlen(4)
#endif
extern "C" double cos(double);

#if defined(MW_STUDY_SSE2_MVEC) && !defined(__AVX2__)
#pragma omp declare simd notinbranch simdlen(2)
#else
#pragma omp declare simd notinbranch simdlen(4)
#endif
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
// unroll and MW_OMP_SIMD vectorization depend on. Nothing carries state
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
#define MW_STR_(x) #x
#define MW_PRAGMA_(x) _Pragma(MW_STR_(x))

// The (i, j)-grid parallel loops below use `collapse(2)` for finer load
// balancing. We keep the Windows OpenMP usage conservative and drop it to plain
// outer-loop parallelism under MSVC — same results, only coarser scheduling
// across the grid. GCC (and MSVC's /openmp:llvm) keeps collapse(2).
#if defined(_MSC_VER)
#  define MW_OMP_PARALLEL_FOR_COLLAPSE2 _Pragma("omp parallel for schedule(static)")
// The work-sharing half alone. The off-edge kernel's per-pair walk keeps
// main's structure on MSVC and never uses it; the off-edge LANE kernels do,
// since momwire#1371 switched the lane layer on there. They keep collapse(2):
// a lane block's rows can be fewer than the threads, and /openmp:llvm takes
// the clause (setup.py's flag note).
#  define MW_OMP_FOR_COLLAPSE2 _Pragma("omp for collapse(2) schedule(static)")
#else
#  define MW_OMP_PARALLEL_FOR_COLLAPSE2 \
       _Pragma("omp parallel for collapse(2) schedule(static)")
// The work-sharing half alone, for a loop inside an `omp parallel` region
// that keeps per-thread state across its iterations.
#  define MW_OMP_FOR_COLLAPSE2 _Pragma("omp for collapse(2) schedule(static)")
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
#  define MW_OMP_SIMD(clauses)
#else
#  define MW_OMP_SIMD(clauses) MW_PRAGMA_(omp simd clauses)
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

#define MW_CANCEL_SETUP(flag_addr)                                          \
    const volatile int32_t *pysim_cancel =                                     \
        reinterpret_cast<const volatile int32_t *>(flag_addr);                 \
    std::atomic<bool> pysim_aborted { false }

// Drain-poll. Place at the very top of a parallel loop body; `continue` targets
// the enclosing for-loop by design, so use only inside a braced loop body.
#define MW_CANCEL_POLL()                                                    \
    if (pysim_aborted.load(std::memory_order_relaxed)) continue;              \
    if (pysim_cancel && *pysim_cancel) {                                       \
        pysim_aborted.store(true, std::memory_order_relaxed);                  \
        continue;                                                              \
    }

#define MW_THROW_IF_ABORTED()                                               \
    if (pysim_aborted.load(std::memory_order_relaxed)) throw AbortedError {}

// The serial twin: a single-threaded loop has no parallel region to drain,
// so it throws at the poll itself. One volatile load per poll; place it at a
// unit of work (a row, a block), not in an innermost loop.
#define MW_CANCEL_SERIAL_SETUP(flag_addr)                                   \
    const volatile int32_t *pysim_cancel =                                     \
        reinterpret_cast<const volatile int32_t *>(flag_addr)

#define MW_CANCEL_SERIAL_POLL()                                             \
    if (pysim_cancel && *pysim_cancel) throw AbortedError {}
