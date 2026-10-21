// Vector exp and sincos for the lane-shaped kernels that libmvec cannot reach
// (momwire#1441): the x86-64 baseline (`_sse2`) build, where GCC's
// `omp declare simd simdlen(4)` names a 4-wide SSE2 variant libmvec does not
// have, and every MSVC build, which has no libmvec at all and compiles the
// `omp simd` hints out. On those builds the near-interface column kernel
// called libm once per node, and that was ~44 % of an sse2 buried fill.
//
// `exp_cis(ar, ai, er, ei, n)`: er + i ei = e^{ar} (cos ai + i sin ai), for
// n doubles, the column kernel's one transcendental step.
//
// THE ALGORITHMS, lane by lane:
//   exp   table-driven, as glibc's: n = round(x 64 / ln 2), r = x - n ln2/64
//         (a 32-bit HI, so n HI is exact), e^x = 2^(n/64) (1 + q(r)) with a
//         64-entry 2^(j/64) table and a degree-5 q, then 2^(n div 64) applied
//         in two halves so results down in the subnormals round once. x is
//         clamped below at -1000 (-> +0), NaN passing through; there is no
//         overflow branch (x <= 709 is the caller's contract: the column
//         exponents are <= 0).
//   sin / cos   n = round(x 2/pi), r = ((x - n p1) - n p2) - n p2t (fdlibm's
//         33-bit splits, so n p1 and n p2 are exact for |n| < 2^20), then
//         __kernel_sin / __kernel_cos on [-pi/4, pi/4] and the quadrant.
// Accuracy: exp within 1 ulp and sin / cos within 2 ulp of glibc's libm
// (tests/test_vmath_1441.py); NOT bit-identical to it, which is why the GCC
// avx2 build keeps libmvec and only the builds that were scalar take this.
// The SSE2 and AVX2 lanes are bit-identical to each other under GCC.
//
// RANGE: |ai| <= MW_VMATH_TRIG_MAX for the reduction above. A block with a
// lane beyond it, or a non-finite angle, falls back to libm whole, so the
// answer there is what it was. The near-interface columns reach 1.6e4.
//
// Every operation is IEEE with no contraction (the TU flags carry
// -ffp-contract=off; MSVC's /fp:fast may contract, which Windows tolerates by
// policy), so a lane's result depends on its own inputs only, never on the
// width or on its neighbours.
#ifndef MOMWIRE_VMATH_INLINE_H
#define MOMWIRE_VMATH_INLINE_H

#include <cmath>
#include <cstdint>
#include <cstring>

#if defined(__SSE2__) || defined(_M_X64) || (defined(_M_IX86_FP) && _M_IX86_FP >= 2)
#define MW_VMATH_X86 1
#include <emmintrin.h>
#if defined(__AVX2__)
#include <immintrin.h>
#endif
#else
#define MW_VMATH_X86 0
#endif

#if defined(__GNUC__)
#define MW_VMATH_INLINE static inline __attribute__((always_inline))
#elif defined(_MSC_VER)
#define MW_VMATH_INLINE static __forceinline
#else
#define MW_VMATH_INLINE static inline
#endif

namespace mw_vmath {

constexpr double MW_VMATH_TRIG_MAX = 1.0e5;

namespace k {
// round-to-nearest by the 1.5 * 2^52 shifter: (x + S) - S, exact for |x| < 2^51
constexpr double SHIFT = 6755399441055744.0;
constexpr int64_t SHIFT_BITS = 0x4338000000000000LL;
// exp: n = round(x 64 / ln 2), x = n ln2/64 + r with |r| <= ln2/128; the
// ln2/64 split keeps 32 bits in HI, so n HI is exact for |n| < 2^21.
constexpr double INV64LN2 = 92.33248261689366;
constexpr double LN2_64HI = 0.010830424695086549;
constexpr double LN2_64LO = 1.162596423439437e-12;
// e^r - 1 - r = r^2 (1/2 + r/6 + r^2/24 + r^3/120); the r^6/720 term left out
// is < 0.2 ulp on |r| <= ln2/128.
constexpr double E2 = 0.5;
constexpr double E3 = 1.0 / 6.0;
constexpr double E4 = 1.0 / 24.0;
constexpr double E5 = 1.0 / 120.0;
// 2^(j/64), j = 0..63, each correctly rounded (generated at 60 digits).
alignas(64) static const double EXP_TAB[64] = {
    1.0, 1.0108892860517005, 1.0218971486541166, 1.0330248790212284, 1.0442737824274138, 1.0556451783605572, 1.0671404006768237, 1.0787607977571199,
    1.0905077326652577, 1.102382583307841, 1.1143867425958924, 1.1265216186082418, 1.1387886347566916, 1.1511892299529827, 1.1637248587775775, 1.1763969916502812,
    1.189207115002721, 1.202156731452703, 1.215247359980469, 1.22848053610687, 1.241857812073484, 1.255380757024691, 1.2690509571917332, 1.2828700160787783,
    1.2968395546510096, 1.3109612115247644, 1.3252366431597413, 1.339667524053303, 1.3542555469368927, 1.3690024229745905, 1.383909881963832, 1.3989796725383112,
    1.4142135623730951, 1.42961333839197, 1.4451808069770467, 1.460917794180647, 1.4768261459394993, 1.4929077282912648, 1.5091644275934228, 1.5255981507445384,
    1.5422108254079407, 1.559004400237837, 1.5759808451078865, 1.593142151342267, 1.6104903319492543, 1.6280274218573478, 1.645755478153965, 1.6636765803267364,
    1.681792830507429, 1.7001063537185235, 1.718619298122478, 1.7373338352737062, 1.7562521603732995, 1.7753764925265212, 1.7947090750031072, 1.8142521755003989,
    1.8340080864093424, 1.8539791250833855, 1.8741676341103, 1.8945759815869656, 1.9152065613971474, 1.9360617934922943, 1.9571441241754002, 1.978456026387951};
constexpr double EXP_FLOOR = -1000.0;
// sin / cos
constexpr double INVPIO2 = 6.36619772367581382433e-01;
constexpr double PIO2_1 = 1.57079632673412561417e+00;
constexpr double PIO2_2 = 6.07710050630396597660e-11;
constexpr double PIO2_2T = 2.02226624879595063154e-21;
constexpr double S1 = -1.66666666666666324348e-01;
constexpr double S2 = 8.33333333332248946124e-03;
constexpr double S3 = -1.98412698298579493134e-04;
constexpr double S4 = 2.75573137070700676789e-06;
constexpr double S5 = -2.50507602534068634195e-08;
constexpr double S6 = 1.58969099521155010221e-10;
constexpr double C1 = 4.16666666666666019037e-02;
constexpr double C2 = -1.38888888888741095749e-03;
constexpr double C3 = 2.48015872894767294178e-05;
constexpr double C4 = -2.75573143513906633035e-07;
constexpr double C5 = 2.08757232129817482790e-09;
constexpr double C6 = -1.13596475577881948265e-11;
}  // namespace k

#if MW_VMATH_X86
// ------------------------------------------------------------ SSE2, W = 2 --
struct sse2 {
    static constexpr int W = 2;
    typedef __m128d vd;
    typedef __m128i vi;
    static vd set1(double x) { return _mm_set1_pd(x); }
    static vd loadu(const double *p) { return _mm_loadu_pd(p); }
    static void storeu(double *p, vd a) { _mm_storeu_pd(p, a); }
    static vd add(vd a, vd b) { return _mm_add_pd(a, b); }
    static vd sub(vd a, vd b) { return _mm_sub_pd(a, b); }
    static vd mul(vd a, vd b) { return _mm_mul_pd(a, b); }
    static vd div(vd a, vd b) { return _mm_div_pd(a, b); }
    // x86 max: b when either is NaN
    static vd max(vd a, vd b) { return _mm_max_pd(a, b); }
    static vi bits(vd a) { return _mm_castpd_si128(a); }
    static vd from_bits(vi a) { return _mm_castsi128_pd(a); }
    static vi add64(vi a, vi b) { return _mm_add_epi64(a, b); }
    static vi sub64(vi a, vi b) { return _mm_sub_epi64(a, b); }
    static vi and_(vi a, vi b) { return _mm_and_si128(a, b); }
    static vi xor_(vi a, vi b) { return _mm_xor_si128(a, b); }
    static vi or_(vi a, vi b) { return _mm_or_si128(a, b); }
    static vi andnot(vi m, vi a) { return _mm_andnot_si128(m, a); }  // ~m & a
    static vi set1_64(int64_t x) { return _mm_set1_epi64x(x); }
    static vi zero_i() { return _mm_setzero_si128(); }
    template <int N> static vi slli64(vi a) { return _mm_slli_epi64(a, N); }
    // tab[idx] per lane, idx in [0, 63] as int64 lanes; jd = idx as double
    static vd lookup_f64(const double *tab, vi idx, vd *jd) {
        const int i0 = _mm_cvtsi128_si32(idx);
        const int i1 = _mm_cvtsi128_si32(_mm_unpackhi_epi64(idx, idx));
        *jd = _mm_set_pd(static_cast<double>(i1), static_cast<double>(i0));
        return _mm_set_pd(tab[i1], tab[i0]);
    }
    // lanes whose |a| <= m (false on NaN)
    static int all_le_abs(vd a, vd m) {
        const vd ab = _mm_andnot_pd(_mm_set1_pd(-0.0), a);
        return _mm_movemask_pd(_mm_cmple_pd(ab, m)) == 3;
    }
};
#endif

#if MW_VMATH_X86 && defined(__AVX2__)
// ------------------------------------------------------------ AVX2, W = 4 --
struct avx2 {
    static constexpr int W = 4;
    typedef __m256d vd;
    typedef __m256i vi;
    static vd set1(double x) { return _mm256_set1_pd(x); }
    static vd loadu(const double *p) { return _mm256_loadu_pd(p); }
    static void storeu(double *p, vd a) { _mm256_storeu_pd(p, a); }
    static vd add(vd a, vd b) { return _mm256_add_pd(a, b); }
    static vd sub(vd a, vd b) { return _mm256_sub_pd(a, b); }
    static vd mul(vd a, vd b) { return _mm256_mul_pd(a, b); }
    static vd div(vd a, vd b) { return _mm256_div_pd(a, b); }
    static vd max(vd a, vd b) { return _mm256_max_pd(a, b); }
    static vi bits(vd a) { return _mm256_castpd_si256(a); }
    static vd from_bits(vi a) { return _mm256_castsi256_pd(a); }
    static vi add64(vi a, vi b) { return _mm256_add_epi64(a, b); }
    static vi sub64(vi a, vi b) { return _mm256_sub_epi64(a, b); }
    static vi and_(vi a, vi b) { return _mm256_and_si256(a, b); }
    static vi xor_(vi a, vi b) { return _mm256_xor_si256(a, b); }
    static vi or_(vi a, vi b) { return _mm256_or_si256(a, b); }
    static vi andnot(vi m, vi a) { return _mm256_andnot_si256(m, a); }
    static vi set1_64(int64_t x) { return _mm256_set1_epi64x(x); }
    static vi zero_i() { return _mm256_setzero_si256(); }
    template <int N> static vi slli64(vi a) { return _mm256_slli_epi64(a, N); }
    static vd lookup_f64(const double *tab, vi idx, vd *jd) {
        // idx < 64: the low dword of each lane is the index
        const __m128i i32 = _mm256_castsi256_si128(
            _mm256_permutevar8x32_epi32(idx, _mm256_setr_epi32(0, 2, 4, 6, 0, 2, 4, 6)));
        *jd = _mm256_cvtepi32_pd(i32);
        return _mm256_i64gather_pd(tab, idx, 8);
    }
    static int all_le_abs(vd a, vd m) {
        const vd ab = _mm256_andnot_pd(_mm256_set1_pd(-0.0), a);
        return _mm256_movemask_pd(_mm256_cmp_pd(ab, m, _CMP_LE_OQ)) == 15;
    }
};
#endif

#if MW_VMATH_X86
// 2^k per lane from k held as a double (an integer in [-1022, 1023]).
template <class B>
MW_VMATH_INLINE typename B::vd pow2_lanes(typename B::vd kd) {
    const typename B::vi kb =
        B::sub64(B::bits(B::add(kd, B::set1(k::SHIFT))), B::set1_64(k::SHIFT_BITS));
    return B::from_bits(B::template slli64<52>(B::add64(kb, B::set1_64(1023))));
}

template <class B>
MW_VMATH_INLINE typename B::vd exp_lanes(typename B::vd x) {
    using namespace k;
    typedef typename B::vd vd;
    x = B::max(B::set1(EXP_FLOOR), x);  // NaN stays x (max returns b)
    const vd S = B::set1(SHIFT);
    const vd t = B::add(B::mul(x, B::set1(INV64LN2)), S);
    const vd nd = B::sub(t, S);
    const typename B::vi jv =
        B::and_(B::sub64(B::bits(t), B::set1_64(SHIFT_BITS)), B::set1_64(63));
    const vd r = B::sub(B::sub(x, B::mul(nd, B::set1(LN2_64HI))), B::mul(nd, B::set1(LN2_64LO)));
    vd p = B::add(B::set1(E4), B::mul(r, B::set1(E5)));
    p = B::add(B::set1(E3), B::mul(r, p));
    p = B::add(B::set1(E2), B::mul(r, p));
    const vd q = B::add(r, B::mul(B::mul(r, r), p));
    vd jdd;
    const vd T = B::lookup_f64(EXP_TAB, jv, &jdd);
    const vd y = B::add(T, B::mul(T, q));
    const vd kd = B::mul(B::sub(nd, jdd), B::set1(1.0 / 64.0));
    const vd h = B::sub(B::add(B::mul(kd, B::set1(0.5)), S), S);
    const vd h2 = B::sub(kd, h);
    return B::mul(B::mul(y, pow2_lanes<B>(h)), pow2_lanes<B>(h2));
}

template <class B>
MW_VMATH_INLINE void sincos_lanes(typename B::vd x, typename B::vd &s, typename B::vd &c) {
    using namespace k;
    typedef typename B::vd vd;
    typedef typename B::vi vi;
    const vd S = B::set1(SHIFT);
    const vd t = B::add(B::mul(x, B::set1(INVPIO2)), S);
    const vd nd = B::sub(t, S);
    const vi q = B::sub64(B::bits(t), B::set1_64(SHIFT_BITS));  // n, two's complement
    const vd r = B::sub(B::sub(B::sub(x, B::mul(nd, B::set1(PIO2_1))), B::mul(nd, B::set1(PIO2_2))),
                        B::mul(nd, B::set1(PIO2_2T)));
    const vd z = B::mul(r, r);
    vd ps = B::add(B::set1(S5), B::mul(z, B::set1(S6)));
    ps = B::add(B::set1(S4), B::mul(z, ps));
    ps = B::add(B::set1(S3), B::mul(z, ps));
    ps = B::add(B::set1(S2), B::mul(z, ps));
    ps = B::add(B::set1(S1), B::mul(z, ps));
    const vd sp = B::add(r, B::mul(B::mul(r, z), ps));
    vd pc = B::add(B::set1(C5), B::mul(z, B::set1(C6)));
    pc = B::add(B::set1(C4), B::mul(z, pc));
    pc = B::add(B::set1(C3), B::mul(z, pc));
    pc = B::add(B::set1(C2), B::mul(z, pc));
    pc = B::add(B::set1(C1), B::mul(z, pc));
    const vd hz = B::mul(B::set1(0.5), z), w = B::sub(B::set1(1.0), hz);
    const vd cp = B::add(w, B::add(B::sub(B::sub(B::set1(1.0), w), hz), B::mul(B::mul(z, z), pc)));
    // quadrant: swap where n is odd; sin negated where n & 2, cos where (n+1) & 2
    const vi one = B::set1_64(1), two = B::set1_64(2);
    const vi swap = B::sub64(B::zero_i(), B::and_(q, one));  // all ones where odd
    const vi sp_b = B::bits(sp), cp_b = B::bits(cp);
    vi sv = B::or_(B::andnot(swap, sp_b), B::and_(swap, cp_b));
    vi cv = B::or_(B::andnot(swap, cp_b), B::and_(swap, sp_b));
    sv = B::xor_(sv, B::template slli64<62>(B::and_(q, two)));
    cv = B::xor_(cv, B::template slli64<62>(B::and_(B::add64(q, one), two)));
    s = B::from_bits(sv);
    c = B::from_bits(cv);
}

// One vector of the step: libm whole when any lane's angle is beyond
// MW_VMATH_TRIG_MAX or not finite, so such a lane's answer is what it was.
template <class B>
MW_VMATH_INLINE void exp_cis_vec(const double *ar, const double *ai, double *er, double *ei) {
    typedef typename B::vd vd;
    const vd a = B::loadu(ar), b = B::loadu(ai);
    if (!B::all_le_abs(b, B::set1(MW_VMATH_TRIG_MAX))) {
        for (int l = 0; l < B::W; ++l) {
            const double ex = std::exp(ar[l]);
            er[l] = ex * std::cos(ai[l]);
            ei[l] = ex * std::sin(ai[l]);
        }
        return;
    }
    const vd ex = exp_lanes<B>(a);
    vd s, c;
    sincos_lanes<B>(b, s, c);
    B::storeu(er, B::mul(ex, c));
    B::storeu(ei, B::mul(ex, s));
}

// er + i ei = e^{ar} (cos ai + i sin ai) for n doubles, B::W at a time. The
// tail runs the same vector code on a zero-padded copy, so every element's
// bits are a function of its own inputs alone -- never of n, of its position
// or of the width -- and there is no scalar transcription of the arithmetic
// for MSVC's /fp:fast to re-associate (the round-to-nearest shifter
// (x c + S) - S is exactly what such a rewrite would fold away; the
// intrinsics are opaque to it).
template <class B>
MW_VMATH_INLINE void exp_cis_lanes(const double *ar, const double *ai, double *er,
                                   double *ei, int n) {
    int j = 0;
    for (; j + B::W <= n; j += B::W) exp_cis_vec<B>(ar + j, ai + j, er + j, ei + j);
    if (j < n) {
        double ta[B::W] = {}, tb[B::W] = {}, te[B::W], tf[B::W];
        for (int l = 0; l < n - j; ++l) {
            ta[l] = ar[j + l];
            tb[l] = ai[j + l];
        }
        exp_cis_vec<B>(ta, tb, te, tf);
        for (int l = 0; l < n - j; ++l) {
            er[j + l] = te[l];
            ei[j + l] = tf[l];
        }
    }
}
#endif

}  // namespace mw_vmath

#endif  // MOMWIRE_VMATH_INLINE_H
