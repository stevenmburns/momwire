// The SIMD lane layer (momwire#1372): one double vector type and the dozen
// operations the hand-written lane kernels use, so each kernel is written
// once against a width W and a backend, not once per instruction set.
//
// THE CONTRACT. Every operation is one IEEE operation per lane, rounded once,
// exactly as the scalar expression it stands for: add/sub/mul/div/sqrt are
// correctly rounded; `fmadd`/`fnmadd` are ONE fused op (std::fma per lane) and
// exist only where a kernel spells the fusion on purpose; nothing here
// contracts, reassociates or reduces across lanes. So a lane's floats are a
// function of that lane's operands and the order the kernel writes, never of
// W or of the backend, and the kernels' walk-vs-lanes gates stay exact. The
// one place lanes are mixed -- the (re, im)-packed complex helpers at the end
// -- moves values between lanes, never combines two lanes' values in one
// rounding other than the product that the scalar complex multiply also
// rounds.
//
// BACKENDS, chosen once here:
//   avx2      W = 4, __m256d. GCC/clang with -mavx2 -mfma (the `_avx2`
//             variant). The reference: the kernels were written against it.
//   avx512    W = 8, __m512d with __mmask8 masks. NOT switched on: it is
//             selected only under MW_LANES_ENABLE_AVX512, which no build sets
//             (momwire#1370 adds the variant and the dispatch).
//   portable  W = MW_LANES_PORTABLE_WIDTH (default 1, i.e. scalar), plain
//             arrays and std:: math. Selected by MW_LANES_FORCE_PORTABLE; the
//             walk-shaped reference a kernel can be instantiated at, and how
//             the layer is checked on a machine without the instruction set.
// With none of them selected MW_LANES_SIMD is 0 and every kernel keeps its
// scalar path alone: the `_sse2` baseline and macOS arm64 builds.
//
// MSVC (momwire#1371): the `_avx2` variant (/arch:AVX2) runs the avx2 backend
// too. It uses only intrinsics MSVC has, and the compiler differences (the
// force-inline spelling, the FMA macro MSVC does not define under /arch:AVX2)
// are resolved below. Exactness does NOT carry over: the Windows build is
// /fp:fast, which contracts and reassociates the scalar reference loops on its
// own terms, and its `mw_fma::fma` is the unfused a*b + c. So on Windows a
// kernel's lanes are gated against its walk by a DERIVED tolerance
// (tests/_lane_gate.py), not bit equality: Windows needs no bit-compatibility
// (2026-10-03), and the fused kernels run there too (2026-10-07; see
// MW_LANES_FUSED below). Nothing here changes what GCC or clang compile, so
// Linux and macOS keep their exact gates.
//
// The (re, im)-packed helpers (`cmul_packed`, `load_cpairs`) need an even W:
// they hold W/2 complex values per vector. MW_LANES_PACKED says whether the
// backend has them; a kernel built on them is not instantiable at W = 1.
#ifndef MOMWIRE_LANES_H
#define MOMWIRE_LANES_H

#include <cmath>
#include <cstddef>
#include <cstdint>

#if defined(__GNUC__)
#define MW_LANES_INLINE static inline __attribute__((always_inline))
#elif defined(_MSC_VER)
#define MW_LANES_INLINE static __forceinline
#else
#define MW_LANES_INLINE static inline
#endif

// MSVC has no __FMA__; its /arch:AVX2 implies FMA3.
#if defined(__AVX2__) && (defined(__FMA__) || defined(_MSC_VER))
#define MW_LANES_HAVE_AVX2 1
#else
#define MW_LANES_HAVE_AVX2 0
#endif

#if defined(MW_LANES_FORCE_PORTABLE)
#define MW_LANES_PORTABLE 1
#elif defined(MW_LANES_ENABLE_AVX512) && defined(__AVX512F__) && \
    defined(__AVX512DQ__) && defined(__AVX512VL__)
#define MW_LANES_AVX512 1
#elif MW_LANES_HAVE_AVX2
#define MW_LANES_AVX2 1
#endif

#if defined(MW_LANES_AVX2) || defined(MW_LANES_AVX512) || \
    defined(MW_LANES_PORTABLE)
#define MW_LANES_SIMD 1
#else
#define MW_LANES_SIMD 0
#endif

// Whether `fmadd` rounds as the scalar `mw_fma::fma` (_fma_inline.h) does on
// this build: a kernel mirroring an mw_fma chain is exact only then. True for
// the vector backends under GCC/clang (their builds have FMA hardware, and
// mw_fma::fma is std::fma); for the portable one it follows _fma_inline.h's
// own test. False under MSVC, whose mw_fma::fma is the unfused a*b + c.
#if ((defined(MW_LANES_AVX2) || defined(MW_LANES_AVX512)) &&            \
     !defined(_MSC_VER)) ||                                              \
    (defined(MW_LANES_PORTABLE) &&                                       \
     (defined(__FMA__) || defined(__ARM_FEATURE_FMA)) && !defined(_MSC_VER))
#define MW_LANES_FMA_EXACT 1
#else
#define MW_LANES_FMA_EXACT 0
#endif

// Whether the FUSED kernels run their lanes: those whose lanes spell `fmadd`
// to mirror an mw_fma chain (the windowed assembly, the below-interface
// stages, the near-interface sheets). They run where the fusion is exact
// (MW_LANES_FMA_EXACT), and on MSVC's vector backend, where it is not and the
// lanes are gated against the walk by the derived win32 tolerance
// (momwire#1371). A portable build without exact fusion keeps them off, so
// the layer's portable check stays a bit gate. Under GCC/clang this IS
// MW_LANES_FMA_EXACT.
#if MW_LANES_FMA_EXACT ||                                                \
    ((defined(MW_LANES_AVX2) || defined(MW_LANES_AVX512)) && defined(_MSC_VER))
#define MW_LANES_FUSED 1
#else
#define MW_LANES_FUSED 0
#endif

#if defined(MW_LANES_AVX2) || defined(MW_LANES_AVX512)
#include <immintrin.h>
#endif

namespace mw_lanes {

#if defined(MW_LANES_AVX2)
// ---------------------------------------------------------------- avx2 ----
constexpr int W = 4;
typedef __m256d vd;
typedef __m256d vm;   // all-ones / all-zeros per lane; blendv reads the sign
typedef __m128i vi;   // W int32
typedef __m128i vmi;  // int32 compare result
#define MW_LANES_PACKED 1

MW_LANES_INLINE vd zero() { return _mm256_setzero_pd(); }
MW_LANES_INLINE vd set1(double x) { return _mm256_set1_pd(x); }
MW_LANES_INLINE vd load(const double *p) { return _mm256_load_pd(p); }
MW_LANES_INLINE vd loadu(const double *p) { return _mm256_loadu_pd(p); }
MW_LANES_INLINE void store(double *p, vd a) { _mm256_store_pd(p, a); }
MW_LANES_INLINE void storeu(double *p, vd a) { _mm256_storeu_pd(p, a); }

MW_LANES_INLINE vd add(vd a, vd b) { return _mm256_add_pd(a, b); }
MW_LANES_INLINE vd sub(vd a, vd b) { return _mm256_sub_pd(a, b); }
MW_LANES_INLINE vd mul(vd a, vd b) { return _mm256_mul_pd(a, b); }
MW_LANES_INLINE vd div(vd a, vd b) { return _mm256_div_pd(a, b); }
MW_LANES_INLINE vd sqrt(vd a) { return _mm256_sqrt_pd(a); }
MW_LANES_INLINE vd floor(vd a) { return _mm256_floor_pd(a); }
// a*b + c and c - a*b, one rounding each.
MW_LANES_INLINE vd fmadd(vd a, vd b, vd c) { return _mm256_fmadd_pd(a, b, c); }
MW_LANES_INLINE vd fnmadd(vd a, vd b, vd c) { return _mm256_fnmadd_pd(a, b, c); }
// Sign flip, exact (an xor of the sign bit, so -0.0 and NaN payloads too).
MW_LANES_INLINE vd neg(vd a) { return _mm256_xor_pd(a, _mm256_set1_pd(-0.0)); }
// x86 semantics, per lane: min(a, b) = a < b ? a : b, max(a, b) = a > b ? a : b
// -- so b when either is NaN, which the kernels rely on.
MW_LANES_INLINE vd min(vd a, vd b) { return _mm256_min_pd(a, b); }
MW_LANES_INLINE vd max(vd a, vd b) { return _mm256_max_pd(a, b); }

// Ordered, quiet compares: false on NaN.
MW_LANES_INLINE vm cmp_lt(vd a, vd b) { return _mm256_cmp_pd(a, b, _CMP_LT_OQ); }
MW_LANES_INLINE vm cmp_le(vd a, vd b) { return _mm256_cmp_pd(a, b, _CMP_LE_OQ); }
MW_LANES_INLINE vm cmp_gt(vd a, vd b) { return _mm256_cmp_pd(a, b, _CMP_GT_OQ); }
// m ? b : a per lane.
MW_LANES_INLINE vd blend(vd a, vd b, vm m) { return _mm256_blendv_pd(a, b, m); }
MW_LANES_INLINE vm mask_from(const bool *e) {
    return _mm256_castsi256_pd(_mm256_set_epi64x(e[3] ? -1 : 0, e[2] ? -1 : 0,
                                                 e[1] ? -1 : 0, e[0] ? -1 : 0));
}
// W int64 flags, -1 on and 0 off.
MW_LANES_INLINE vm load_mask(const int64_t *p) {
    return _mm256_loadu_pd(reinterpret_cast<const double *>(p));
}

// int32 lanes: truncating convert, exact convert back, gathers by index.
MW_LANES_INLINE vi cvtt_i32(vd a) { return _mm256_cvttpd_epi32(a); }
MW_LANES_INLINE vd cvt_f64(vi a) { return _mm256_cvtepi32_pd(a); }
MW_LANES_INLINE vd gather(const double *base, vi idx) {
    return _mm256_i32gather_pd(base, idx, 8);
}
MW_LANES_INLINE vi gather_i32(const int *base, vi idx) {
    return _mm_i32gather_epi32(base, idx, 4);
}
MW_LANES_INLINE vi set1_i32(int x) { return _mm_set1_epi32(x); }
MW_LANES_INLINE vi zero_i32() { return _mm_setzero_si128(); }
MW_LANES_INLINE vi sub_i32(vi a, vi b) { return _mm_sub_epi32(a, b); }
MW_LANES_INLINE vmi cmpgt_i32(vi a, vi b) { return _mm_cmpgt_epi32(a, b); }
MW_LANES_INLINE vi blend_i32(vi a, vi b, vmi m) { return _mm_blendv_epi8(a, b, m); }
MW_LANES_INLINE void storeu_i32(int *p, vi a) {
    _mm_storeu_si128(reinterpret_cast<__m128i *>(p), a);
}

// (re, im)-packed: lanes 2c, 2c + 1 hold complex value c.
// w x (componentwise): [wr xr - wi xi, wr xi + wi xr], the four products
// rounded, then one subtract / one add -- std::complex's finite expansion.
MW_LANES_INLINE vd cmul_packed(vd w, vd x) {
    return _mm256_addsub_pd(
        _mm256_mul_pd(_mm256_movedup_pd(w), x),
        _mm256_mul_pd(_mm256_permute_pd(w, 0xF), _mm256_permute_pd(x, 0x5)));
}
// W/2 complex values, value c from the (re, im) pair at p[c].
MW_LANES_INLINE vd load_cpairs(const double *const *p) {
    return _mm256_insertf128_pd(_mm256_castpd128_pd256(_mm_loadu_pd(p[0])),
                                _mm_loadu_pd(p[1]), 1);
}
// re, im as W complex values, (re0 im0 re1 im1 ...), 2W doubles at o.
MW_LANES_INLINE void store_interleaved(double *o, vd re, vd im) {
    const __m256d lo = _mm256_unpacklo_pd(re, im);  // re0 im0 re2 im2
    const __m256d hi = _mm256_unpackhi_pd(re, im);  // re1 im1 re3 im3
    _mm256_storeu_pd(o, _mm256_permute2f128_pd(lo, hi, 0x20));
    _mm256_storeu_pd(o + 4, _mm256_permute2f128_pd(lo, hi, 0x31));
}

#elif defined(MW_LANES_AVX512)
// -------------------------------------------------------------- avx512 ----
// For momwire#1370. Compile-checked, never run: no build defines
// MW_LANES_ENABLE_AVX512. Masks are k-registers, so `blend` is a masked move,
// not blendv; there is no addsub, so `cmul_packed` is a masked add over a
// subtract -- the same two roundings per lane.
constexpr int W = 8;
typedef __m512d vd;
typedef __mmask8 vm;
typedef __m256i vi;
typedef __m256i vmi;
#define MW_LANES_PACKED 1

MW_LANES_INLINE vd zero() { return _mm512_setzero_pd(); }
MW_LANES_INLINE vd set1(double x) { return _mm512_set1_pd(x); }
MW_LANES_INLINE vd load(const double *p) { return _mm512_load_pd(p); }
MW_LANES_INLINE vd loadu(const double *p) { return _mm512_loadu_pd(p); }
MW_LANES_INLINE void store(double *p, vd a) { _mm512_store_pd(p, a); }
MW_LANES_INLINE void storeu(double *p, vd a) { _mm512_storeu_pd(p, a); }

MW_LANES_INLINE vd add(vd a, vd b) { return _mm512_add_pd(a, b); }
MW_LANES_INLINE vd sub(vd a, vd b) { return _mm512_sub_pd(a, b); }
MW_LANES_INLINE vd mul(vd a, vd b) { return _mm512_mul_pd(a, b); }
MW_LANES_INLINE vd div(vd a, vd b) { return _mm512_div_pd(a, b); }
MW_LANES_INLINE vd sqrt(vd a) { return _mm512_sqrt_pd(a); }
MW_LANES_INLINE vd floor(vd a) {
    return _mm512_roundscale_pd(a, _MM_FROUND_TO_NEG_INF | _MM_FROUND_NO_EXC);
}
MW_LANES_INLINE vd fmadd(vd a, vd b, vd c) { return _mm512_fmadd_pd(a, b, c); }
MW_LANES_INLINE vd fnmadd(vd a, vd b, vd c) { return _mm512_fnmadd_pd(a, b, c); }
MW_LANES_INLINE vd neg(vd a) { return _mm512_xor_pd(a, _mm512_set1_pd(-0.0)); }
MW_LANES_INLINE vd min(vd a, vd b) { return _mm512_min_pd(a, b); }
MW_LANES_INLINE vd max(vd a, vd b) { return _mm512_max_pd(a, b); }

MW_LANES_INLINE vm cmp_lt(vd a, vd b) { return _mm512_cmp_pd_mask(a, b, _CMP_LT_OQ); }
MW_LANES_INLINE vm cmp_le(vd a, vd b) { return _mm512_cmp_pd_mask(a, b, _CMP_LE_OQ); }
MW_LANES_INLINE vm cmp_gt(vd a, vd b) { return _mm512_cmp_pd_mask(a, b, _CMP_GT_OQ); }
MW_LANES_INLINE vd blend(vd a, vd b, vm m) { return _mm512_mask_blend_pd(m, a, b); }
MW_LANES_INLINE vm mask_from(const bool *e) {
    unsigned m = 0;
    for (int l = 0; l < W; ++l) m |= (e[l] ? 1u : 0u) << l;
    return (vm)m;
}
MW_LANES_INLINE vm load_mask(const int64_t *p) {
    return _mm512_cmplt_epi64_mask(_mm512_loadu_si512(p), _mm512_setzero_si512());
}

MW_LANES_INLINE vi cvtt_i32(vd a) { return _mm512_cvttpd_epi32(a); }
MW_LANES_INLINE vd cvt_f64(vi a) { return _mm512_cvtepi32_pd(a); }
MW_LANES_INLINE vd gather(const double *base, vi idx) {
    return _mm512_i32gather_pd(idx, base, 8);
}
MW_LANES_INLINE vi gather_i32(const int *base, vi idx) {
    return _mm256_i32gather_epi32(base, idx, 4);
}
MW_LANES_INLINE vi set1_i32(int x) { return _mm256_set1_epi32(x); }
MW_LANES_INLINE vi zero_i32() { return _mm256_setzero_si256(); }
MW_LANES_INLINE vi sub_i32(vi a, vi b) { return _mm256_sub_epi32(a, b); }
MW_LANES_INLINE vmi cmpgt_i32(vi a, vi b) { return _mm256_cmpgt_epi32(a, b); }
MW_LANES_INLINE vi blend_i32(vi a, vi b, vmi m) { return _mm256_blendv_epi8(a, b, m); }
MW_LANES_INLINE void storeu_i32(int *p, vi a) {
    _mm256_storeu_si256(reinterpret_cast<__m256i *>(p), a);
}

MW_LANES_INLINE vd cmul_packed(vd w, vd x) {
    const vd t = _mm512_mul_pd(_mm512_movedup_pd(w), x);
    const vd u = _mm512_mul_pd(_mm512_permute_pd(w, 0xFF), _mm512_permute_pd(x, 0x55));
    // even lanes (re) t - u, odd lanes (im) t + u
    return _mm512_mask_add_pd(_mm512_sub_pd(t, u), (__mmask8)0xAA, t, u);
}
MW_LANES_INLINE vd load_cpairs(const double *const *p) {
    vd v = _mm512_castpd128_pd512(_mm_loadu_pd(p[0]));
    v = _mm512_insertf64x2(v, _mm_loadu_pd(p[1]), 1);
    v = _mm512_insertf64x2(v, _mm_loadu_pd(p[2]), 2);
    return _mm512_insertf64x2(v, _mm_loadu_pd(p[3]), 3);
}
MW_LANES_INLINE void store_interleaved(double *o, vd re, vd im) {
    const vd lo = _mm512_unpacklo_pd(re, im);  // re0 im0 re2 im2 re4 im4 re6 im6
    const vd hi = _mm512_unpackhi_pd(re, im);  // re1 im1 re3 im3 re5 im5 re7 im7
    const __m512i i0 = _mm512_set_epi64(11, 10, 3, 2, 9, 8, 1, 0);
    const __m512i i1 = _mm512_set_epi64(15, 14, 7, 6, 13, 12, 5, 4);
    _mm512_storeu_pd(o, _mm512_permutex2var_pd(lo, i0, hi));
    _mm512_storeu_pd(o + 8, _mm512_permutex2var_pd(lo, i1, hi));
}

#elif defined(MW_LANES_PORTABLE)
// ------------------------------------------------------------ portable ----
// Plain arrays and std:: math, each lane the scalar expression. With the
// build's -ffp-contract=off nothing here fuses; `fmadd` is std::fma.
#ifndef MW_LANES_PORTABLE_WIDTH
#define MW_LANES_PORTABLE_WIDTH 1
#endif
constexpr int W = MW_LANES_PORTABLE_WIDTH;
struct vd { double v[W]; };
struct vm { bool v[W]; };
struct vi { int32_t v[W]; };
typedef vm vmi;
#if MW_LANES_PORTABLE_WIDTH % 2 == 0
#define MW_LANES_PACKED 1
#else
#define MW_LANES_PACKED 0
#endif

#define MW_LANES_MAP_(T, expr)            \
    T r;                                  \
    for (int l = 0; l < W; ++l) r.v[l] = (expr); \
    return r

MW_LANES_INLINE vd zero() { MW_LANES_MAP_(vd, 0.0); }
MW_LANES_INLINE vd set1(double x) { MW_LANES_MAP_(vd, x); }
MW_LANES_INLINE vd load(const double *p) { MW_LANES_MAP_(vd, p[l]); }
MW_LANES_INLINE vd loadu(const double *p) { MW_LANES_MAP_(vd, p[l]); }
MW_LANES_INLINE void store(double *p, vd a) { for (int l = 0; l < W; ++l) p[l] = a.v[l]; }
MW_LANES_INLINE void storeu(double *p, vd a) { for (int l = 0; l < W; ++l) p[l] = a.v[l]; }

MW_LANES_INLINE vd add(vd a, vd b) { MW_LANES_MAP_(vd, a.v[l] + b.v[l]); }
MW_LANES_INLINE vd sub(vd a, vd b) { MW_LANES_MAP_(vd, a.v[l] - b.v[l]); }
MW_LANES_INLINE vd mul(vd a, vd b) { MW_LANES_MAP_(vd, a.v[l] * b.v[l]); }
MW_LANES_INLINE vd div(vd a, vd b) { MW_LANES_MAP_(vd, a.v[l] / b.v[l]); }
MW_LANES_INLINE vd sqrt(vd a) { MW_LANES_MAP_(vd, std::sqrt(a.v[l])); }
MW_LANES_INLINE vd floor(vd a) { MW_LANES_MAP_(vd, std::floor(a.v[l])); }
MW_LANES_INLINE vd fmadd(vd a, vd b, vd c) { MW_LANES_MAP_(vd, std::fma(a.v[l], b.v[l], c.v[l])); }
MW_LANES_INLINE vd fnmadd(vd a, vd b, vd c) { MW_LANES_MAP_(vd, std::fma(-a.v[l], b.v[l], c.v[l])); }
MW_LANES_INLINE vd neg(vd a) { MW_LANES_MAP_(vd, -a.v[l]); }
MW_LANES_INLINE vd min(vd a, vd b) { MW_LANES_MAP_(vd, a.v[l] < b.v[l] ? a.v[l] : b.v[l]); }
MW_LANES_INLINE vd max(vd a, vd b) { MW_LANES_MAP_(vd, a.v[l] > b.v[l] ? a.v[l] : b.v[l]); }

MW_LANES_INLINE vm cmp_lt(vd a, vd b) { MW_LANES_MAP_(vm, a.v[l] < b.v[l]); }
MW_LANES_INLINE vm cmp_le(vd a, vd b) { MW_LANES_MAP_(vm, a.v[l] <= b.v[l]); }
MW_LANES_INLINE vm cmp_gt(vd a, vd b) { MW_LANES_MAP_(vm, a.v[l] > b.v[l]); }
MW_LANES_INLINE vd blend(vd a, vd b, vm m) { MW_LANES_MAP_(vd, m.v[l] ? b.v[l] : a.v[l]); }
MW_LANES_INLINE vm mask_from(const bool *e) { MW_LANES_MAP_(vm, e[l]); }
MW_LANES_INLINE vm load_mask(const int64_t *p) { MW_LANES_MAP_(vm, p[l] < 0); }

MW_LANES_INLINE vi cvtt_i32(vd a) { MW_LANES_MAP_(vi, (int32_t)a.v[l]); }
MW_LANES_INLINE vd cvt_f64(vi a) { MW_LANES_MAP_(vd, (double)a.v[l]); }
MW_LANES_INLINE vd gather(const double *base, vi idx) { MW_LANES_MAP_(vd, base[idx.v[l]]); }
MW_LANES_INLINE vi gather_i32(const int *base, vi idx) { MW_LANES_MAP_(vi, base[idx.v[l]]); }
MW_LANES_INLINE vi set1_i32(int x) { MW_LANES_MAP_(vi, x); }
MW_LANES_INLINE vi zero_i32() { MW_LANES_MAP_(vi, 0); }
MW_LANES_INLINE vi sub_i32(vi a, vi b) { MW_LANES_MAP_(vi, a.v[l] - b.v[l]); }
MW_LANES_INLINE vmi cmpgt_i32(vi a, vi b) { MW_LANES_MAP_(vmi, a.v[l] > b.v[l]); }
MW_LANES_INLINE vi blend_i32(vi a, vi b, vmi m) { MW_LANES_MAP_(vi, m.v[l] ? b.v[l] : a.v[l]); }
MW_LANES_INLINE void storeu_i32(int *p, vi a) { for (int l = 0; l < W; ++l) p[l] = a.v[l]; }

#if MW_LANES_PACKED
MW_LANES_INLINE vd cmul_packed(vd w, vd x) {
    vd r;
    for (int l = 0; l < W; l += 2) {
        const double t0 = w.v[l] * x.v[l], u0 = w.v[l + 1] * x.v[l + 1];
        const double t1 = w.v[l] * x.v[l + 1], u1 = w.v[l + 1] * x.v[l];
        r.v[l] = t0 - u0;
        r.v[l + 1] = t1 + u1;
    }
    return r;
}
MW_LANES_INLINE vd load_cpairs(const double *const *p) {
    vd r;
    for (int c = 0; c < W / 2; ++c) {
        r.v[2 * c] = p[c][0];
        r.v[2 * c + 1] = p[c][1];
    }
    return r;
}
#endif
MW_LANES_INLINE void store_interleaved(double *o, vd re, vd im) {
    for (int l = 0; l < W; ++l) {
        o[2 * l] = re.v[l];
        o[2 * l + 1] = im.v[l];
    }
}
#undef MW_LANES_MAP_
#endif

#if MW_LANES_SIMD
// Byte alignment of a [..][W] lane buffer read with `load`.
constexpr size_t ALIGN = W * sizeof(double) < 16 ? 16 : W * sizeof(double);
#endif

}  // namespace mw_lanes

#endif  // MOMWIRE_LANES_H
