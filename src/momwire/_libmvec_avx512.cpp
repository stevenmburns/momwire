// libmvec's 4-lane entry points under the AVX-512 build (momwire#1370).
//
// The `omp declare simd notinbranch simdlen(4)` blocks that bind cos/sin/exp
// to glibc's libmvec (`_accel_common.h`, `_accel_razor_cplx.cpp`,
// `_accel_sinusoidal_cplx.cpp`, both near-interface TUs) name the clone by the
// ISA the TU is compiled for. Under -mavx2 that is `_ZGVdN4v_*`, which glibc
// exports. Under -mavx512f GCC names `_ZGVeN4v_*` instead -- and glibc exports
// no 4-lane AVX-512 entry, only `_ZGVeN8v_*`. A shared object links with the
// reference undefined and then fails to import on exactly the CPUs the build
// is for.
//
// The `_avx512` variant keeps the 4-lane declaration on purpose, and this TU
// supplies the missing names as tail calls into the AVX2 entries. Why not
// declare simdlen(8) and bind `_ZGVeN8v_*`, which is what libmvec offers?
// Because libmvec's routines are not where the bits move. Measured on the
// Rocket Lake box (glibc 2.39, 16.7M arguments each across the fills' phase
// range): `_ZGVeN8v_{cos,sin,exp}` and `_ZGVdN4v_*` agree to the bit on every
// argument, and both differ from scalar libm on 36-65 % of them. What moves is
// WHICH points of a loop take the vector entry and which fall to the scalar
// tail, and that is the loop length modulo the vectorisation factor. At 8 the
// walk's and the lanes' tails stop coinciding (an even order n with n = 2 mod
// 4 gives m = n^2 = 4 mod 8), which is the 32 bit-gate failures step 1 saw.
// So this build vectorises its autovectorised loops at the AVX2 factor
// (setup.py passes -mprefer-vector-width=256) and calls the same libmvec
// entries the AVX2 build does: every loop has the AVX2 build's tails and every
// point the AVX2 build's routine, and only the lane layer (`_lanes.h`, under
// MW_LANES_ENABLE_AVX512) runs 8 wide.
//
// Calling convention: a 4-lane double clone takes and returns __m256d in ymm0
// under either ISA letter, so each thunk is one `jmp`. Hidden, so the name
// never leaves this extension and never meets another library's.
//
// Empty everywhere else: the AVX2 and baseline builds, MSVC (no libmvec) and
// macOS (no libmvec; no declare simd blocks either).
#if defined(__AVX512F__) && defined(__GNUC__) && !defined(_MSC_VER) && \
    !defined(__APPLE__)
#include <immintrin.h>

extern "C" {
__m256d _ZGVdN4v_cos(__m256d);
__m256d _ZGVdN4v_sin(__m256d);
__m256d _ZGVdN4v_exp(__m256d);

__attribute__((visibility("hidden"))) __m256d _ZGVeN4v_cos(__m256d x) {
    return _ZGVdN4v_cos(x);
}
__attribute__((visibility("hidden"))) __m256d _ZGVeN4v_sin(__m256d x) {
    return _ZGVdN4v_sin(x);
}
__attribute__((visibility("hidden"))) __m256d _ZGVeN4v_exp(__m256d x) {
    return _ZGVdN4v_exp(x);
}
}
#endif
