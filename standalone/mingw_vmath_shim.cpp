// libmvec's three AVX2 names for the MinGW-w64 executables (optimisation 4 of
// WINDOWS-SINGLE-FILE.md): Windows has no libmvec, so with the `omp declare
// simd` blocks switched on (MW_MINGW_VMATH_SHIM, the avx2 cores only) gcc's
// vectorised sin / cos / exp loops call the definitions here, which run
// mw_vmath's AVX2 lanes (`_vmath_inline.h`, momwire#1441).
//
// THE CALL. gcc emits a simd clone call as an ordinary call of a function of
// type `__m256d (__m256d)`, so under the Microsoft x64 ABI a 32-byte vector
// goes through memory: RCX points at the result, RDX at the argument, RAX
// returns RCX, and neither pointer is 32-byte aligned (the callers' frames
// are aligned to 16). XMM6-XMM15 are callee-saved. Compiling these definitions
// with the same gcc gives all of that by construction;
// scripts/windows_static/vmath_shim_check.cpp holds it to it at run time.
//
// THE RANGE. mw_vmath covers |x| <= MW_VMATH_TRIG_MAX for sin / cos and
// x <= 709 for exp. A vector with any lane outside, or with a NaN (or, for
// sin / cos, an infinity), goes whole to the C runtime's scalar functions
// (ucrtbase's under UCRT64), so such a lane's answer is what it was without
// this file. exp's lanes clamp below at -1000, where the answer is +0 anyway.
//
// NOTHING ELSE MAY BE DEFINED HERE. In the fat executable this object is
// compiled -mavx2 and linked beside the sse2 code: a dynamic initializer would
// run AVX2 code before main() on a CPU without it, and an out-of-line inline
// function could be picked for the sse2 copies. windows-static.yml checks the
// object's defined symbols and sections.
#if !defined(__MINGW32__) || !defined(__AVX2__)
#error "mingw_vmath_shim.cpp is the MinGW-w64 avx2 build's only"
#endif

#include <immintrin.h>

#include <cmath>

#include "_vmath_inline.h"

#if defined(MW_VMATH_SHIM_COUNT)
// The proof that the shim ran (a timing-free build, windows-static.yml): calls
// per function, fast path and whole-vector fallback, reported on stderr at
// exit. Registered at the first call, so inside main: the atexit table is
// last-in first-out, so the report runs before the executable's exit handler
// ends the process.
#include <atomic>
#include <cstdint>
#include <cstdio>
#include <cstdlib>

namespace {
std::atomic<uint64_t> g_counts[6];
std::atomic<int> g_registered{0};
void report_counts() {
    std::fprintf(stderr,
                 "mw_vmath_shim: sin fast=%llu fallback=%llu cos fast=%llu fallback=%llu "
                 "exp fast=%llu fallback=%llu\n",
                 (unsigned long long)g_counts[0].load(), (unsigned long long)g_counts[1].load(),
                 (unsigned long long)g_counts[2].load(), (unsigned long long)g_counts[3].load(),
                 (unsigned long long)g_counts[4].load(), (unsigned long long)g_counts[5].load());
    std::fflush(stderr);
}
inline void count(int slot) {
    g_counts[slot].fetch_add(1, std::memory_order_relaxed);
    if (g_registered.load(std::memory_order_relaxed) == 0 && g_registered.exchange(1) == 0) {
        std::atexit(report_counts);
    }
}
}  // namespace
extern "C" void mw_vmath_shim_counts(uint64_t out[6]) {
    for (int i = 0; i < 6; ++i) out[i] = g_counts[i].load();
}
#define MW_SHIM_COUNT(slot) count(slot)
#else
#define MW_SHIM_COUNT(slot) ((void)0)
#endif

namespace {

typedef mw_vmath::avx2 B;

// exp's overflow bound: mw_vmath has no overflow branch (its header's contract)
constexpr double EXP_FAST_MAX = 709.0;

// Whole vector through the C runtime. Out of line: the rare path stays out of
// the hot function's frame.
__attribute__((noinline)) __m256d scalar_lanes(double (*f)(double), __m256d x) {
    alignas(32) double v[4];
    _mm256_store_pd(v, x);
    for (int l = 0; l < 4; ++l) v[l] = f(v[l]);
    return _mm256_load_pd(v);
}

inline bool trig_in_range(__m256d x) {
    return B::all_le_abs(x, B::set1(mw_vmath::MW_VMATH_TRIG_MAX));
}

}  // namespace

extern "C" __m256d _ZGVdN4v_sin(__m256d x) {
    if (__builtin_expect(!trig_in_range(x), 0)) {
        MW_SHIM_COUNT(1);
        return scalar_lanes(::sin, x);
    }
    MW_SHIM_COUNT(0);
    __m256d s, c;
    mw_vmath::sincos_lanes<B>(x, s, c);
    // sin(-0.0) is -0.0: the reduction's sum would give +0.0
    return _mm256_blendv_pd(s, x, _mm256_cmp_pd(x, _mm256_setzero_pd(), _CMP_EQ_OQ));
}

extern "C" __m256d _ZGVdN4v_cos(__m256d x) {
    if (__builtin_expect(!trig_in_range(x), 0)) {
        MW_SHIM_COUNT(3);
        return scalar_lanes(::cos, x);
    }
    MW_SHIM_COUNT(2);
    __m256d s, c;
    mw_vmath::sincos_lanes<B>(x, s, c);
    return c;
}

extern "C" __m256d _ZGVdN4v_exp(__m256d x) {
    // false on NaN, so a NaN lane falls back too
    const __m256d in_range = _mm256_cmp_pd(x, _mm256_set1_pd(EXP_FAST_MAX), _CMP_LE_OQ);
    if (__builtin_expect(_mm256_movemask_pd(in_range) != 15, 0)) {
        MW_SHIM_COUNT(5);
        return scalar_lanes(::exp, x);
    }
    MW_SHIM_COUNT(4);
    return mw_vmath::exp_lanes<B>(x);
}
