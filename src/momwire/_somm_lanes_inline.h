// Lanes for the Sommerfeld integrands (momwire#1441 step 3).
//
// The integrands of the Sommerfeld contour engines are latency-bound one node
// at a time: a node is a ~900-cycle chain of square roots, divides, Bessel
// ladders and exponentials, far longer than the reorder window, so the core
// never overlaps two nodes and its second FMA pipe sits idle (BURIED-STUDY,
// antennaknobs scratch, 2026-10-15). What helps is evaluating several nodes
// of one Gauss rule at once, one node per SIMD lane.
//
// THE CONTRACT is `_lanes.h`'s: every operation is one IEEE operation per
// lane, rounded once, exactly as the scalar expression it stands for, and
// nothing reduces across lanes. `fma` here is `mw_fma::fma` per lane: one
// fused op where that is one (GCC/clang with -mfma, arm64), the unfused
// a*b + c where it is not (the x86-64 baseline `_sse2` build, MSVC). So a
// lane computes the bits its node's scalar walk computes, on every build, and
// the lane kernels below are gated against the scalar ones as uint64.
//
// The transcendental functions are the one thing a lane cannot do itself
// without changing bits: libm's exp/sin/cos/log and glibc's csqrt are called
// per lane, on the lane's own operand, through `opaque` so that no vectorizer
// can swap in a libmvec clone (`_mw_core.h` declares sin/cos `omp declare
// simd` TU-wide, and a vectorized clone rounds differently). Square roots and
// divides are IEEE operations and run in the vector unit.
//
// Why not `_lanes.h`. Its baseline build has no backend at all (MW_LANES_SIMD
// is 0 under `_sse2`), and giving it one would switch on every kernel written
// against it in that build. These kernels want the baseline too, since the
// baseline is the build a CPU without AVX2 runs. So this header is
// self-contained and its backends are chosen by what the TU was compiled for:
//
//   avx2      one __m256d per W = 4 lanes (GCC/clang -mavx2 -mfma, MSVC
//             /arch:AVX2).
//   sse2      two __m128d per W = 4 lanes: the x86-64 baseline. Four lanes
//             rather than two because these chains are latency-bound: the
//             second register pair is independent work the core overlaps.
//   portable  double[4], each lane the scalar expression (arm64, anything
//             else; MW_SL_FORCE_PORTABLE selects it anywhere, for a check).
//
// Python-free (<cmath>, <complex>, the two inline headers): the standalone
// executable compiles the same kernels.
#pragma once

#include <cmath>
#include <complex>
#include <cstdint>

#include "_branch_cut_inline.h"
#include "_cdiv_inline.h"
#include "_fma_inline.h"

#if defined(MW_SL_FORCE_PORTABLE)
#define MW_SL_PORTABLE 1
#elif (defined(__AVX2__) && (defined(__FMA__) || defined(_MSC_VER)))
#define MW_SL_AVX2 1
#include <immintrin.h>
#elif defined(__SSE2__) || defined(_M_X64) || \
    (defined(_M_IX86_FP) && _M_IX86_FP >= 2)
#define MW_SL_SSE2 1
#include <emmintrin.h>
#else
#define MW_SL_PORTABLE 1
#endif

// Whether mw_fma::fma is one fused operation on this build (_fma_inline.h's
// own test): the lanes must fuse exactly where it does.
#if (defined(__FMA__) || defined(__ARM_FEATURE_FMA)) && !defined(_MSC_VER)
#define MW_SL_FUSED 1
#else
#define MW_SL_FUSED 0
#endif

#if defined(__GNUC__)
#define MW_SL_INLINE static inline __attribute__((always_inline))
#elif defined(_MSC_VER)
#define MW_SL_INLINE static __forceinline
#else
#define MW_SL_INLINE static inline
#endif

// momwire#1441 step 4: the opt-in vector-math route (`vecmath` on the
// bindings, `_LANES_VECMATH` in Python). Off -- the default -- every lane calls
// libm exactly as above and the lanes are the scalar walk's bits. On, the
// lanes' transcendental functions run on glibc's libmvec, one vector call per
// four lanes instead of four libm calls: _ZGVdN4v_* on the avx2 build, two
// _ZGVbN2v_* on the sse2 one. libmvec is not libm (its functions are within a
// few ulps, not correctly rounded alike), so ON is a different table, gated by
// measurement (the census on momwire#1441), and the Python side puts the mode
// in every key of a table it fills. libmvec is glibc-only: MSVC, macOS, musl
// and the portable backend compile the route away, `kVecmathBuilt` is 0, and a
// request for it is the libm route (Python reads the built mask and keys on
// what actually ran).
//
// The mode is a bit mask, so the census can price each family on its own:
#define MW_SL_VM_CEXP 1   // complex exp: exp(Re) * (cos, sin)(Im)
#define MW_SL_VM_TRIG 2   // the Bessel asymptotics' sin / cos, real and complex
#define MW_SL_VM_CLOG 4   // complex log: (log hypot, atan2)
#define MW_SL_VM_CSQRT 8  // csqrt: the lanes' own principal root (lean_sqrt)
#define MW_SL_VM_ALL 15
#if !defined(MW_SL_NO_VECMATH) && defined(__GLIBC__) && !defined(_MSC_VER) && \
    (defined(MW_SL_AVX2) || defined(MW_SL_SSE2)) && defined(__x86_64__)
#define MW_SL_VECMATH 1
#include <atomic>
// Only functions libmvec has had since glibc 2.22: the Linux wheels build on
// manylinux_2_28, and a 2.35 function (sinh, cosh, hypot, atan2, log1p, ...)
// links locally and then fails to IMPORT from the wheel
// (tests/test_glibc_floor.py). hypot, log1p, sinh and cosh are composed
// below from these and IEEE operations; atan2 stays libm per lane.
extern "C" {
#if defined(MW_SL_AVX2)
__m256d _ZGVdN4v_exp(__m256d);
__m256d _ZGVdN4v_sin(__m256d);
__m256d _ZGVdN4v_cos(__m256d);
__m256d _ZGVdN4v_log(__m256d);
#else
__m128d _ZGVbN2v_exp(__m128d);
__m128d _ZGVbN2v_sin(__m128d);
__m128d _ZGVbN2v_cos(__m128d);
__m128d _ZGVbN2v_log(__m128d);
#endif
}
#else
#define MW_SL_VECMATH 0
#endif

namespace mw_sl {

constexpr int W = 4;
// Whether the lanes run here. The portable backend is exact but no faster
// than the walk (arrays of four, left to the autovectorizer), so a build
// without a vector backend keeps the walk unless the portable lanes were
// asked for by name. MSVC keeps the walk too: its /arch:AVX2 /fp:fast build
// of these lanes compiled but missed the walk by 4e3 relative on the
// above-ground gate (wheels.yml, 2026-10-16), and a GCC build with MSVC's
// unfused multiply-add and operator divide matches it to 1e-11, so the cause
// is MSVC-specific and still open. Windows needs no bit-compatibility, but it
// does need the right answer.
#if (defined(MW_SL_AVX2) || defined(MW_SL_SSE2) || defined(MW_SL_FORCE_PORTABLE)) && \
    !defined(_MSC_VER)
constexpr bool kActive = true;
#else
constexpr bool kActive = false;
#endif
typedef std::complex<double> cd;

// The vector-math mode of the call in flight (MW_SL_VM_* bits). One per
// module: a binding that takes `vecmath` sets it for its own duration
// (VecmathScope) and every other binding finds it 0. A process-wide value
// rather than an argument threaded through every lane function, because it
// is an A/B switch for the census and the tests -- two concurrent calls with
// DIFFERENT modes in one process are not supported (one mode per process is).
#if MW_SL_VECMATH
constexpr int kVecmathBuilt = kActive ? MW_SL_VM_ALL : 0;
inline std::atomic<int> g_vecmath{0};
MW_SL_INLINE bool vecmath_on(int bit) {
    return (g_vecmath.load(std::memory_order_relaxed) & bit) != 0;
}
#else
constexpr int kVecmathBuilt = 0;
MW_SL_INLINE bool vecmath_on(int) { return false; }
#endif
struct VecmathScope {
    explicit VecmathScope(int mode) {
#if MW_SL_VECMATH
        g_vecmath.store(mode & kVecmathBuilt, std::memory_order_relaxed);
#else
        (void)mode;
#endif
    }
    ~VecmathScope() {
#if MW_SL_VECMATH
        g_vecmath.store(0, std::memory_order_relaxed);
#endif
    }
    VecmathScope(const VecmathScope &) = delete;
    VecmathScope &operator=(const VecmathScope &) = delete;
};

#if defined(MW_SL_AVX2)
// ---------------------------------------------------------------- avx2 ----
struct V {
    __m256d v;
};
typedef V M;  // all-ones / all-zeros per lane

MW_SL_INLINE V set1(double x) { return V{_mm256_set1_pd(x)}; }
MW_SL_INLINE V loadu(const double *p) { return V{_mm256_loadu_pd(p)}; }
MW_SL_INLINE void storeu(double *p, V a) { _mm256_storeu_pd(p, a.v); }
MW_SL_INLINE V add(V a, V b) { return V{_mm256_add_pd(a.v, b.v)}; }
MW_SL_INLINE V sub(V a, V b) { return V{_mm256_sub_pd(a.v, b.v)}; }
MW_SL_INLINE V mul(V a, V b) { return V{_mm256_mul_pd(a.v, b.v)}; }
MW_SL_INLINE V div(V a, V b) { return V{_mm256_div_pd(a.v, b.v)}; }
MW_SL_INLINE V sqrt(V a) { return V{_mm256_sqrt_pd(a.v)}; }
MW_SL_INLINE V fma(V a, V b, V c) {
#if MW_SL_FUSED
    return V{_mm256_fmadd_pd(a.v, b.v, c.v)};
#else
    return V{_mm256_add_pd(_mm256_mul_pd(a.v, b.v), c.v)};
#endif
}
// The sign bit as an integer pattern, not the literal -0.0: MSVC's /fp:fast
// does not keep the sign of a zero constant, and a +0.0 here would turn neg
// and fabs into the identity.
MW_SL_INLINE __m256d sign_bit() {
    return _mm256_castsi256_pd(_mm256_set1_epi64x(static_cast<long long>(0x8000000000000000ULL)));
}
MW_SL_INLINE V neg(V a) { return V{_mm256_xor_pd(a.v, sign_bit())}; }
MW_SL_INLINE V fabs(V a) { return V{_mm256_andnot_pd(sign_bit(), a.v)}; }
// |mag| with the sign bit of sgn: std::copysign, bit for bit.
MW_SL_INLINE V copysign(V mag, V sgn) {
    const __m256d s = sign_bit();
    return V{_mm256_or_pd(_mm256_andnot_pd(s, mag.v), _mm256_and_pd(s, sgn.v))};
}
// Ordered, quiet: false on NaN, as the scalar `<`, `<=`, ... are.
MW_SL_INLINE M lt(V a, V b) { return V{_mm256_cmp_pd(a.v, b.v, _CMP_LT_OQ)}; }
MW_SL_INLINE M le(V a, V b) { return V{_mm256_cmp_pd(a.v, b.v, _CMP_LE_OQ)}; }
MW_SL_INLINE M gt(V a, V b) { return V{_mm256_cmp_pd(a.v, b.v, _CMP_GT_OQ)}; }
MW_SL_INLINE M ge(V a, V b) { return V{_mm256_cmp_pd(a.v, b.v, _CMP_GE_OQ)}; }
MW_SL_INLINE M eq(V a, V b) { return V{_mm256_cmp_pd(a.v, b.v, _CMP_EQ_OQ)}; }
MW_SL_INLINE M m_and(M a, M b) { return V{_mm256_and_pd(a.v, b.v)}; }
MW_SL_INLINE M m_or(M a, M b) { return V{_mm256_or_pd(a.v, b.v)}; }
// a and not b
MW_SL_INLINE M m_andnot(M a, M b) { return V{_mm256_andnot_pd(b.v, a.v)}; }
MW_SL_INLINE M m_true() {
    return V{_mm256_castsi256_pd(_mm256_set1_epi64x(-1))};
}
MW_SL_INLINE M m_false() { return V{_mm256_setzero_pd()}; }
MW_SL_INLINE int bits(M m) { return _mm256_movemask_pd(m.v); }
// m ? b : a per lane.
MW_SL_INLINE V blend(V a, V b, M m) { return V{_mm256_blendv_pd(a.v, b.v, m.v)}; }
// (a > b) ? a : b per lane, the scalar `if (a > b) b = a` (NaN keeps b).
MW_SL_INLINE V vmax(V a, V b) { return V{_mm256_max_pd(a.v, b.v)}; }

#elif defined(MW_SL_SSE2)
// ---------------------------------------------------------------- sse2 ----
struct V {
    __m128d lo, hi;
};
typedef V M;

MW_SL_INLINE V set1(double x) {
    const __m128d s = _mm_set1_pd(x);
    return V{s, s};
}
MW_SL_INLINE V loadu(const double *p) { return V{_mm_loadu_pd(p), _mm_loadu_pd(p + 2)}; }
MW_SL_INLINE void storeu(double *p, V a) {
    _mm_storeu_pd(p, a.lo);
    _mm_storeu_pd(p + 2, a.hi);
}
#define MW_SL_BIN_(name, op) \
    MW_SL_INLINE V name(V a, V b) { return V{op(a.lo, b.lo), op(a.hi, b.hi)}; }
MW_SL_BIN_(add, _mm_add_pd)
MW_SL_BIN_(sub, _mm_sub_pd)
MW_SL_BIN_(mul, _mm_mul_pd)
MW_SL_BIN_(div, _mm_div_pd)
MW_SL_BIN_(lt, _mm_cmplt_pd)
MW_SL_BIN_(le, _mm_cmple_pd)
MW_SL_BIN_(gt, _mm_cmpgt_pd)
MW_SL_BIN_(ge, _mm_cmpge_pd)
MW_SL_BIN_(eq, _mm_cmpeq_pd)
MW_SL_BIN_(m_and, _mm_and_pd)
MW_SL_BIN_(m_or, _mm_or_pd)
#undef MW_SL_BIN_
MW_SL_INLINE V sqrt(V a) { return V{_mm_sqrt_pd(a.lo), _mm_sqrt_pd(a.hi)}; }
// The sign bit as an integer pattern (see the avx2 backend's note).
MW_SL_INLINE __m128d sign_bit() {
    return _mm_castsi128_pd(_mm_set1_epi64x(static_cast<long long>(0x8000000000000000ULL)));
}
// No FMA hardware at the baseline: mw_fma::fma is a*b + c there.
MW_SL_INLINE V fma(V a, V b, V c) { return add(mul(a, b), c); }
MW_SL_INLINE V neg(V a) {
    const __m128d s = sign_bit();
    return V{_mm_xor_pd(a.lo, s), _mm_xor_pd(a.hi, s)};
}
MW_SL_INLINE V fabs(V a) {
    const __m128d s = sign_bit();
    return V{_mm_andnot_pd(s, a.lo), _mm_andnot_pd(s, a.hi)};
}
MW_SL_INLINE V copysign(V mag, V sgn) {
    const __m128d s = sign_bit();
    return V{_mm_or_pd(_mm_andnot_pd(s, mag.lo), _mm_and_pd(s, sgn.lo)),
             _mm_or_pd(_mm_andnot_pd(s, mag.hi), _mm_and_pd(s, sgn.hi))};
}
MW_SL_INLINE M m_andnot(M a, M b) {
    return V{_mm_andnot_pd(b.lo, a.lo), _mm_andnot_pd(b.hi, a.hi)};
}
MW_SL_INLINE M m_true() {
    const __m128d t = _mm_castsi128_pd(_mm_set1_epi32(-1));
    return V{t, t};
}
MW_SL_INLINE M m_false() { return V{_mm_setzero_pd(), _mm_setzero_pd()}; }
MW_SL_INLINE int bits(M m) {
    return _mm_movemask_pd(m.lo) | (_mm_movemask_pd(m.hi) << 2);
}
MW_SL_INLINE V blend(V a, V b, M m) {
    return V{_mm_or_pd(_mm_and_pd(m.lo, b.lo), _mm_andnot_pd(m.lo, a.lo)),
             _mm_or_pd(_mm_and_pd(m.hi, b.hi), _mm_andnot_pd(m.hi, a.hi))};
}
MW_SL_INLINE V vmax(V a, V b) { return V{_mm_max_pd(a.lo, b.lo), _mm_max_pd(a.hi, b.hi)}; }

#else
// ------------------------------------------------------------ portable ----
struct V {
    double v[W];
};
struct M {
    bool v[W];
};
#define MW_SL_MAP_(T, expr)                       \
    T r;                                          \
    for (int l = 0; l < W; ++l) r.v[l] = (expr); \
    return r
MW_SL_INLINE V set1(double x) { MW_SL_MAP_(V, x); }
MW_SL_INLINE V loadu(const double *p) { MW_SL_MAP_(V, p[l]); }
MW_SL_INLINE void storeu(double *p, V a) {
    for (int l = 0; l < W; ++l) p[l] = a.v[l];
}
MW_SL_INLINE V add(V a, V b) { MW_SL_MAP_(V, a.v[l] + b.v[l]); }
MW_SL_INLINE V sub(V a, V b) { MW_SL_MAP_(V, a.v[l] - b.v[l]); }
MW_SL_INLINE V mul(V a, V b) { MW_SL_MAP_(V, a.v[l] * b.v[l]); }
MW_SL_INLINE V div(V a, V b) { MW_SL_MAP_(V, a.v[l] / b.v[l]); }
MW_SL_INLINE V sqrt(V a) { MW_SL_MAP_(V, std::sqrt(a.v[l])); }
MW_SL_INLINE V fma(V a, V b, V c) { MW_SL_MAP_(V, mw_fma::fma(a.v[l], b.v[l], c.v[l])); }
MW_SL_INLINE V neg(V a) { MW_SL_MAP_(V, -a.v[l]); }
MW_SL_INLINE V fabs(V a) { MW_SL_MAP_(V, std::fabs(a.v[l])); }
MW_SL_INLINE V copysign(V m, V s) { MW_SL_MAP_(V, std::copysign(m.v[l], s.v[l])); }
MW_SL_INLINE M lt(V a, V b) { MW_SL_MAP_(M, a.v[l] < b.v[l]); }
MW_SL_INLINE M le(V a, V b) { MW_SL_MAP_(M, a.v[l] <= b.v[l]); }
MW_SL_INLINE M gt(V a, V b) { MW_SL_MAP_(M, a.v[l] > b.v[l]); }
MW_SL_INLINE M ge(V a, V b) { MW_SL_MAP_(M, a.v[l] >= b.v[l]); }
MW_SL_INLINE M eq(V a, V b) { MW_SL_MAP_(M, a.v[l] == b.v[l]); }
MW_SL_INLINE M m_and(M a, M b) { MW_SL_MAP_(M, a.v[l] && b.v[l]); }
MW_SL_INLINE M m_or(M a, M b) { MW_SL_MAP_(M, a.v[l] || b.v[l]); }
MW_SL_INLINE M m_andnot(M a, M b) { MW_SL_MAP_(M, a.v[l] && !b.v[l]); }
MW_SL_INLINE M m_true() { MW_SL_MAP_(M, true); }
MW_SL_INLINE M m_false() { MW_SL_MAP_(M, false); }
MW_SL_INLINE int bits(M m) {
    int b = 0;
    for (int l = 0; l < W; ++l) b |= (m.v[l] ? 1 : 0) << l;
    return b;
}
MW_SL_INLINE V blend(V a, V b, M m) { MW_SL_MAP_(V, m.v[l] ? b.v[l] : a.v[l]); }
MW_SL_INLINE V vmax(V a, V b) { MW_SL_MAP_(V, a.v[l] > b.v[l] ? a.v[l] : b.v[l]); }
#undef MW_SL_MAP_
#endif

static const int ALL = (1 << W) - 1;
MW_SL_INLINE bool any(M m) { return bits(m) != 0; }
MW_SL_INLINE bool all(M m) { return bits(m) == ALL; }
MW_SL_INLINE M m_not(M m) { return m_andnot(m_true(), m); }

// ---- per-lane access (the scalar fallbacks and libm calls) ---------------
struct Buf {
    double v[W];
};
MW_SL_INLINE Buf spill(V a) {
    Buf b;
    storeu(b.v, a);
    return b;
}

// ---- complex lanes --------------------------------------------------------
struct C {
    V re, im;
};

MW_SL_INLINE C cset1(const cd &z) { return C{set1(z.real()), set1(z.imag())}; }
MW_SL_INLINE C cadd(C a, C b) { return C{add(a.re, b.re), add(a.im, b.im)}; }
MW_SL_INLINE C csub(C a, C b) { return C{sub(a.re, b.re), sub(a.im, b.im)}; }
MW_SL_INLINE C cneg(C a) { return C{neg(a.re), neg(a.im)}; }
// s * z and z * s for a real s: std::complex's componentwise product.
MW_SL_INLINE C cscale(V s, C z) { return C{mul(s, z.re), mul(s, z.im)}; }
MW_SL_INLINE C cscale(double s, C z) { return cscale(set1(s), z); }
// z / s for a real s: componentwise.
MW_SL_INLINE C cdivr(C z, V s) { return C{div(z.re, s), div(z.im, s)}; }
// z + s, z - s for a real s: the real part only.
MW_SL_INLINE C caddr(C z, double s) { return C{add(z.re, set1(s)), z.im}; }
MW_SL_INLINE C csubr(C z, double s) { return C{sub(z.re, set1(s)), z.im}; }
MW_SL_INLINE C cblend(C a, C b, M m) { return C{blend(a.re, b.re, m), blend(a.im, b.im, m)}; }

// mw_fma::mul: re = fma(xr, yr, -(xi yi)), im = fma(xr, yi, xi yr).
MW_SL_INLINE C cmul(C x, C y) {
    return C{fma(x.re, y.re, neg(mul(x.im, y.im))), fma(x.re, y.im, mul(x.im, y.re))};
}
// mw_fma::mul_add(x, y, acc) for complex y: the fused product, then one add
// per part.
MW_SL_INLINE C cmul_add(C x, C y, C acc) {
    const C p = cmul(x, y);
    return C{add(acc.re, p.re), add(acc.im, p.im)};
}
// mw_fma::mul_add(x, s, acc) for a real s: one fused op per part.
MW_SL_INLINE C cmul_add(C x, V s, C acc) {
    return C{fma(x.re, s, acc.re), fma(x.im, s, acc.im)};
}
// mw_fma::mul_sub(x, y, z): x * y - z with the scalar's split.
MW_SL_INLINE C cmul_sub(C x, C y, C z) {
    return C{fma(x.re, y.re, fma(neg(x.im), y.im, neg(z.re))),
             fma(x.re, y.im, fma(x.im, y.re, neg(z.im)))};
}
// mw_fma::norm2.
MW_SL_INLINE V norm2(C z) { return fma(z.re, z.re, mul(z.im, z.im)); }

MW_SL_INLINE cd lane(C z, int l) {
    const Buf r = spill(z.re), i = spill(z.im);
    return cd(r.v[l], i.v[l]);
}

// Apply a scalar complex function to each lane. `opaque` on every operand and
// result part keeps the call scalar: no loop here may be vectorized onto a
// libmvec clone.
template <class F>
MW_SL_INLINE C per_lane(C z, F f) {
    Buf r = spill(z.re), i = spill(z.im);
    for (int l = 0; l < W; ++l) {
        const cd v = f(cd(mw_fma::opaque(r.v[l]), mw_fma::opaque(i.v[l])));
        r.v[l] = mw_fma::opaque(v.real());
        i.v[l] = mw_fma::opaque(v.imag());
    }
    return C{loadu(r.v), loadu(i.v)};
}
template <class F>
MW_SL_INLINE V per_lane_r(V x, F f) {
    Buf r = spill(x);
    for (int l = 0; l < W; ++l) r.v[l] = mw_fma::opaque(f(mw_fma::opaque(r.v[l])));
    return loadu(r.v);
}

// The same, on the lanes of `act` only (bits(M)); the others keep z, a value
// nobody reads. A lane outside a regime costs no libm call there.
template <class F>
MW_SL_INLINE C per_lane(C z, F f, int act) {
    if (act == ALL) return per_lane(z, f);
    Buf r = spill(z.re), i = spill(z.im);
    for (int l = 0; l < W; ++l) {
        if (!(act & (1 << l))) continue;
        const cd v = f(cd(mw_fma::opaque(r.v[l]), mw_fma::opaque(i.v[l])));
        r.v[l] = mw_fma::opaque(v.real());
        i.v[l] = mw_fma::opaque(v.imag());
    }
    return C{loadu(r.v), loadu(i.v)};
}

// std::exp / std::sqrt / std::log of each lane: glibc's cexp, csqrt, clog.
struct CexpF {
    cd operator()(const cd &v) const { return std::exp(v); }
};
struct CsqrtF {
    cd operator()(const cd &v) const { return std::sqrt(v); }
};
struct ClogF {
    cd operator()(const cd &v) const { return std::log(v); }
};
MW_SL_INLINE C cexp_libm(C z, int act = ALL) { return per_lane(z, CexpF(), act); }
MW_SL_INLINE C csqrt_per_lane(C z, int act = ALL) { return per_lane(z, CsqrtF(), act); }
MW_SL_INLINE C clog_libm(C z, int act = ALL) { return per_lane(z, ClogF(), act); }

// mw_cdiv::div, per lane bit for bit. libgcc's Smith step is written once,
// with the operands swapped per lane, so a lane runs the three divides its
// own branch runs. A lane outside the guarded range, or one that needs the
// subnormal-ratio order (a real or imaginary divisor, an exact zero ratio),
// is recomputed by the scalar mw_cdiv::div itself.
MW_SL_INLINE C cdiv(C x, C y) {
#if defined(__GNUC__) && !defined(__clang__)
    const V lo = set1(mw_cdiv::MW_CDIV_LO), hi = set1(mw_cdiv::MW_CDIV_HI);
    const V zero = set1(0.0);
    const V aa = fabs(x.re), ab = fabs(x.im), ac = fabs(y.re), ad = fabs(y.im);
    // in_range(v): lo <= |v| <= hi; zero_or_in_range: v == 0 or in range.
    const M ra = m_or(eq(x.re, zero), m_and(ge(aa, lo), le(aa, hi)));
    const M rb = m_or(eq(x.im, zero), m_and(ge(ab, lo), le(ab, hi)));
    const M a_branch = lt(ac, ad);  // |c| < |d|
    // A: ratio = c/d, needs in_range(d) and zero_or_in_range(c);
    // B: ratio = d/c, needs in_range(c) and zero_or_in_range(d).
    const V p = blend(y.im, y.re, a_branch);  // A: c ; B: d
    const V q = blend(y.re, y.im, a_branch);  // A: d ; B: c
    const V ap = fabs(p), aq = fabs(q);
    const M rq = m_and(ge(aq, lo), le(aq, hi));
    const M rp = m_or(eq(p, zero), m_and(ge(ap, lo), le(ap, hi)));
    const V ratio = div(p, q);
    const V denom = add(mul(p, ratio), q);
    // re: A (a ratio + b), B (b ratio + a)
    const V u = blend(x.im, x.re, a_branch);  // A: a ; B: b
    const V w = blend(x.re, x.im, a_branch);  // A: b ; B: a
    const V re = div(add(mul(u, ratio), w), denom);
    // im: A (b ratio - a), B (b - a ratio)
    const V m1 = mul(w, ratio);  // A: b ratio ; B: a ratio
    const V s1 = blend(x.im, m1, a_branch);  // A: b ratio ; B: b
    const V s2 = blend(m1, x.re, a_branch);  // A: a ; B: a ratio
    const V im = div(sub(s1, s2), denom);
    const M ok = m_and(m_and(m_and(ra, rb), m_and(rq, rp)), ge(fabs(ratio), lo));
    C r{re, im};
    const int b = bits(ok);
    if (b != ALL) {
        Buf rr = spill(r.re), ri = spill(r.im);
        const Buf xr = spill(x.re), xi = spill(x.im), yr = spill(y.re), yi = spill(y.im);
        for (int l = 0; l < W; ++l) {
            if (b & (1 << l)) continue;
            const cd v = mw_cdiv::div(cd(xr.v[l], xi.v[l]), cd(yr.v[l], yi.v[l]));
            rr.v[l] = v.real();
            ri.v[l] = v.imag();
        }
        r = C{loadu(rr.v), loadu(ri.v)};
    }
    return r;
#else
    // Clang and MSVC keep the operator (see _cdiv_inline.h): per lane.
    const Buf xr = spill(x.re), xi = spill(x.im), yr = spill(y.re), yi = spill(y.im);
    Buf rr, ri;
    for (int l = 0; l < W; ++l) {
        const cd v = mw_cdiv::div(cd(xr.v[l], xi.v[l]), cd(yr.v[l], yi.v[l]));
        rr.v[l] = v.real();
        ri.v[l] = v.imag();
    }
    return C{loadu(rr.v), loadu(ri.v)};
#endif
}
MW_SL_INLINE C cdiv(double s, C y) { return cdiv(C{set1(s), set1(0.0)}, y); }

// mw_branch::lean_sqrt, per lane bit for bit. Both of its general branches
// are one formula once |x| is written for the x < 0 case's d - x (the same
// rounding: x - y and x + (-y) are one IEEE operation). A lane with x == 0
// or outside the guard is recomputed by the scalar lean_sqrt.
MW_SL_INLINE C lean_sqrt(C z) {
    const V x = z.re, y = z.im;
    const V ax = fabs(x), ay = fabs(y);
    const M xgt = gt(ax, ay);
    const V m = blend(ay, ax, xgt);
    const M ok = m_and(m_and(ge(m, set1(3.054936363499605e-151)),
                             le(m, set1(3.273390607896142e+150))),
                       m_not(eq(x, set1(0.0))));
    const V d = sqrt(add(mul(x, x), mul(y, y)));
    const V t = sqrt(mul(set1(0.5), add(d, ax)));
    const V u = mul(set1(0.5), div(y, t));
    const M pos = gt(x, set1(0.0));
    // x > 0: (t, copysign(u, y));  x < 0: (|u|, copysign(t, y)).
    C r{blend(fabs(u), t, pos), copysign(blend(t, u, pos), y)};
    const int b = bits(ok);
    if (b != ALL) {
        Buf rr = spill(r.re), ri = spill(r.im);
        const Buf xr = spill(x), yi = spill(y);
        for (int l = 0; l < W; ++l) {
            if (b & (1 << l)) continue;
            const cd v = mw_branch::lean_sqrt(cd(xr.v[l], yi.v[l]));
            rr.v[l] = v.real();
            ri.v[l] = v.imag();
        }
        r = C{loadu(rr.v), loadu(ri.v)};
    }
    return r;
}

// mw_branch::gamma_cut: the same three fused products, the same two roots.
MW_SL_INLINE C gamma_cut(C lam, const cd &k) {
    const C kk = cset1(k);
    const C mj = cset1(-mw_branch::MW_BRANCH_J);
    const C j = cset1(mw_branch::MW_BRANCH_J);
    return cmul(lean_sqrt(cmul(mj, csub(lam, kk))), lean_sqrt(cmul(j, cadd(lam, kk))));
}


// ---- the vector-math route (momwire#1441 step 4) --------------------------
#if MW_SL_VECMATH
namespace vm {
#if defined(MW_SL_AVX2)
#define MW_SL_VM1_(name, sym) \
    MW_SL_INLINE V name(V a) { return V{sym(a.v)}; }
MW_SL_VM1_(exp, _ZGVdN4v_exp)
MW_SL_VM1_(sin, _ZGVdN4v_sin)
MW_SL_VM1_(cos, _ZGVdN4v_cos)
MW_SL_VM1_(log, _ZGVdN4v_log)
#else
#define MW_SL_VM1_(name, sym) \
    MW_SL_INLINE V name(V a) { return V{sym(a.lo), sym(a.hi)}; }
MW_SL_VM1_(exp, _ZGVbN2v_exp)
MW_SL_VM1_(sin, _ZGVbN2v_sin)
MW_SL_VM1_(cos, _ZGVbN2v_cos)
MW_SL_VM1_(log, _ZGVbN2v_log)
#endif
#undef MW_SL_VM1_

// a*a = p + e exactly: the fused residual where fma is one op, Veltkamp's
// split where it is not (the sse2 baseline).
MW_SL_INLINE void two_square(V a, V &p, V &e) {
    p = mul(a, a);
#if MW_SL_FUSED
    e = fma(a, a, neg(p));
#else
    const V c = mul(set1(134217729.0), a);
    const V hi = sub(c, sub(c, a));
    const V lo = sub(a, hi);
    e = add(add(sub(mul(hi, hi), p), mul(set1(2.0), mul(hi, lo))), mul(lo, lo));
#endif
}

// |z| without overflow or underflow of the squares: m sqrt(1 + (n/m)^2).
MW_SL_INLINE V hypot(V x, V y) {
    const V ax = fabs(x), ay = fabs(y);
    const V m = vmax(ax, ay);
    const V n = blend(ax, ay, gt(ax, ay));  // the smaller
    const V r = div(n, m);
    return mul(m, sqrt(fma(r, r, set1(1.0))));
}

// The lanes of `act` that `ok` refuses, recomputed by the scalar f: the
// guards keep the vector formulas to the range where they are the function
// (no overflow of an intermediate, finite operands); outside it, libm.
template <class F>
MW_SL_INLINE C patch(C r, C z, M ok, int act, F f) {
    const int bad = act & ~bits(ok);
    if (!bad) return r;
    Buf rr = spill(r.re), ri = spill(r.im);
    const Buf zr = spill(z.re), zi = spill(z.im);
    for (int l = 0; l < W; ++l) {
        if (!(bad & (1 << l))) continue;
        const cd v = f(cd(zr.v[l], zi.v[l]));
        rr.v[l] = v.real();
        ri.v[l] = v.imag();
    }
    return C{loadu(rr.v), loadu(ri.v)};
}

// exp(x + iy) = e^x (cos y, sin y): glibc's cexp in this range, with
// libmvec's exp, cos and sin.
MW_SL_INLINE C cexp(C z, int act) {
    const V e = exp(z.re);
    const C r{mul(e, cos(z.im)), mul(e, sin(z.im))};
    const M ok = m_and(le(z.re, set1(700.0)), le(fabs(z.im), set1(1e9)));
    return patch(r, z, ok, act, CexpF());
}

// log(z) = (log |z|, atan2(y, x)), atan2 by libm per lane (libmvec's is
// GLIBC_2.35). Near |z| = 1, log |z| is a small number that log(hypot) gets
// only to an absolute ulp of 1; there it is log1p(x^2 - 1 + y^2) / 2, with
// both squares' rounding errors kept (Dekker) so the cancellation leaves the
// right digits, and log1p(u) = log(w) u / (w - 1), w = 1 + u (Goldberg).
MW_SL_INLINE C clog(C z, int act) {
    const V h = hypot(z.re, z.im);
    V lr = log(h);
    const M near1 = m_and(gt(h, set1(0.5)), lt(h, set1(2.0)));
    if (any(near1)) {
        V px, ex, py, ey;
        two_square(z.re, px, ex);
        two_square(z.im, py, ey);
        const V u = add(add(sub(px, set1(1.0)), py), add(ex, ey));
        const V w = add(set1(1.0), u);
        const V wm1 = sub(w, set1(1.0));
        const M exact1 = eq(wm1, set1(0.0));
        const V lp = blend(div(mul(log(w), u), blend(wm1, set1(1.0), exact1)), u, exact1);
        lr = blend(lr, mul(set1(0.5), lp), near1);
    }
    Buf ai = spill(z.im);
    const Buf zr = spill(z.re);
    for (int l = 0; l < W; ++l)
        if (act & (1 << l)) ai.v[l] = mw_fma::opaque(std::atan2(ai.v[l], zr.v[l]));
    const C r{lr, loadu(ai.v)};
    const M ok = m_and(ge(h, set1(2.2250738585072014e-308)), le(h, set1(1e300)));
    return patch(r, z, ok, act, ClogF());
}

// cosh y and sinh y from one exp: (e + 1/e)/2 and (e - 1/e)/2 with e =
// exp|y|; for |y| < 1/2, where that difference cancels, sinh by its odd
// series (the y^19 term is below 1e-22 relative there).
MW_SL_INLINE void cosh_sinh(V y, V &ch, V &sh) {
    const V ay = fabs(y);
    const V e = exp(ay), ei = div(set1(1.0), e);
    ch = mul(set1(0.5), add(e, ei));
    V big = mul(set1(0.5), sub(e, ei));
    const V y2 = mul(ay, ay);
    // Horner in y^2: 1 + y2/3! + y2^2/5! + ... + y2^8/17!
    V p = set1(1.0 / 355687428096000.0);
    p = fma(p, y2, set1(1.0 / 1307674368000.0));
    p = fma(p, y2, set1(1.0 / 6227020800.0));
    p = fma(p, y2, set1(1.0 / 39916800.0));
    p = fma(p, y2, set1(1.0 / 362880.0));
    p = fma(p, y2, set1(1.0 / 5040.0));
    p = fma(p, y2, set1(1.0 / 120.0));
    p = fma(p, y2, set1(1.0 / 6.0));
    const V small = add(ay, mul(mul(ay, y2), p));
    sh = copysign(blend(big, small, lt(ay, set1(0.5))), y);
}

// cos z = (cos x cosh y, -sin x sinh y), sin z = (sin x cosh y, cos x sinh y).
struct CcosF_ {
    cd operator()(const cd &v) const { return std::cos(v); }
};
struct CsinF_ {
    cd operator()(const cd &v) const { return std::sin(v); }
};
MW_SL_INLINE void ccos_csin(C z, int act, C &cz, C &sz) {
    const V cx = cos(z.re), sx = sin(z.re);
    V ch, sh;
    cosh_sinh(z.im, ch, sh);
    cz = C{mul(cx, ch), neg(mul(sx, sh))};
    sz = C{mul(sx, ch), mul(cx, sh)};
    const M ok = m_and(le(fabs(z.im), set1(700.0)), le(fabs(z.re), set1(1e9)));
    cz = patch(cz, z, ok, act, CcosF_());
    sz = patch(sz, z, ok, act, CsinF_());
}
}  // namespace vm
#endif

// The lanes' transcendental functions: libm per lane (the default, the scalar
// walk's bits), or the vector-math route when the call in flight asked for it.
MW_SL_INLINE C cexp(C z, int act = ALL) {
#if MW_SL_VECMATH
    if (vecmath_on(MW_SL_VM_CEXP)) return vm::cexp(z, act);
#endif
    return cexp_libm(z, act);
}
MW_SL_INLINE C clog(C z, int act = ALL) {
#if MW_SL_VECMATH
    if (vecmath_on(MW_SL_VM_CLOG)) return vm::clog(z, act);
#endif
    return clog_libm(z, act);
}
// glibc's csqrt per lane; under the vector route, the lanes' own principal
// root (lean_sqrt, above: IEEE square roots and divides, a different rounding
// from glibc's csqrt, the same branch).
MW_SL_INLINE C csqrt_libm(C z, int act = ALL) {
#if MW_SL_VECMATH
    if (vecmath_on(MW_SL_VM_CSQRT)) return lean_sqrt(z);
#endif
    return csqrt_per_lane(z, act);
}
// std::cos / std::sin of each lane of `act` (complex).
struct CcosF {
    cd operator()(const cd &v) const { return std::cos(v); }
};
struct CsinF {
    cd operator()(const cd &v) const { return std::sin(v); }
};
MW_SL_INLINE void ccos_csin(C z, int act, C &cz, C &sz) {
#if MW_SL_VECMATH
    if (vecmath_on(MW_SL_VM_TRIG)) {
        vm::ccos_csin(z, act, cz, sz);
        return;
    }
#endif
    cz = per_lane(z, CcosF(), act);
    sz = per_lane(z, CsinF(), act);
}
// std::cos / std::sin of each lane of `act` (real); the others keep x.
MW_SL_INLINE V cos_lanes(V x, int act) {
#if MW_SL_VECMATH
    if (vecmath_on(MW_SL_VM_TRIG)) return vm::cos(x);
#endif
    Buf r = spill(x);
    for (int l = 0; l < W; ++l)
        if (act & (1 << l)) r.v[l] = mw_fma::opaque(std::cos(mw_fma::opaque(r.v[l])));
    return loadu(r.v);
}
MW_SL_INLINE V sin_lanes(V x, int act) {
#if MW_SL_VECMATH
    if (vecmath_on(MW_SL_VM_TRIG)) return vm::sin(x);
#endif
    Buf r = spill(x);
    for (int l = 0; l < W; ++l)
        if (act & (1 << l)) r.v[l] = mw_fma::opaque(std::sin(mw_fma::opaque(r.v[l])));
    return loadu(r.v);
}

}  // namespace mw_sl
