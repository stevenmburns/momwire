// Explicit fused multiply-adds (momwire#1194).
//
// The GCC/clang builds pass -ffp-contract=off, so the compiler never fuses
// a*b+c on its own: an expression rounds exactly as it is written, whatever
// the inliner decides. Before that, GCC's default -ffp-contract=fast fused
// wherever it saw the shape after inlining, and inlining follows
// translation-unit-wide budgets, so an edit to one function changed which
// multiply-adds fused in others (#1187 moved hub_deck(16)'s Z through grid-fill
// integrands it never touched).
//
// Where a hot loop wants the fused op for speed, it asks for it here, in the
// source. The fusion decision then belongs to the line that spells it and
// cannot move when something else in the file changes.
//
// On a build without hardware FMA (the x86-64 baseline `_sse2` variant, and
// MSVC, whose /fp:fast contracts on its own terms) these fall back to the
// unfused expression: `std::fma` there is a libm call, correctly rounded in
// software, which would cost far more than the fusion saves. So the AVX2 and
// baseline builds give different last bits wherever these are used, as they
// did before (the baseline never had an FMA to contract into).
//
// A site that must fuse on EVERY build -- the numpy-shaped complex product --
// spells `mw_fma::exact` instead, never `std::fma`: correctly rounded
// everywhere, and on a build without hardware FMA an inline emulation that
// costs a fraction of libm's software fma (`exact`, below).
#ifndef MOMWIRE_FMA_INLINE_H
#define MOMWIRE_FMA_INLINE_H

#include <cfloat>
#include <cmath>
#include <complex>
#include <cstdint>
#include <cstring>

// Always inlined. These are a handful of instructions each, but GCC's
// inliner prices the asm barrier below as expensive and, left to itself,
// called `mul` out of line from the Sommerfeld integrand: a call per complex
// product, which cost more than the fusion saved. MW_FMA_ALWAYS_INLINE is the
// same attribute for a caller's own integrand (a member function, so no
// `static`), where the helpers' expansion pushed it over the inline limit.
#if defined(__GNUC__)
#define MW_FMA_ALWAYS_INLINE inline __attribute__((always_inline))
#define MW_FMA_INLINE static inline __attribute__((always_inline))
#elif defined(_MSC_VER)
#define MW_FMA_ALWAYS_INLINE __forceinline
#define MW_FMA_INLINE static __forceinline
#else
#define MW_FMA_ALWAYS_INLINE inline
#define MW_FMA_INLINE static inline
#endif

namespace mw_fma {

// __FMA__: x86 built with -mfma (the AVX2 variant). __ARM_FEATURE_FMA: every
// arm64 target, where std::fma is one fmadd instruction.
#if (defined(__FMA__) || defined(__ARM_FEATURE_FMA)) && !defined(_MSC_VER)
MW_FMA_INLINE double fma(double a, double b, double c) { return std::fma(a, b, c); }
#else
MW_FMA_INLINE double fma(double a, double b, double c) { return a * b + c; }
#endif

// ---------------------------------------------------------------------------
// `exact`: fma(a, b, c) correctly rounded (a*b + c, ONE rounding) on EVERY
// build. It is what an explicit `std::fma` site means -- the sites that must
// fuse whatever the hardware, such as the numpy-shaped complex product
// (`np_cmul`, momwire#1497) -- and those sites spell it this way instead.
//
// Where the build has hardware FMA it IS `std::fma`: one instruction under
// GCC/clang with -mfma or on arm64; under MSVC /arch:AVX2 the UCRT's `fma`,
// as before. Without it -- the x86-64 baseline `_sse2` variant under GCC,
// clang or MSVC -- `std::fma` is a call into libm's software fma, and glibc's
// costs ~57 ns a call on a CPU without the instruction: 1.66 s of a 3.87 s
// buried cold solve on a Dan-class machine (proto/razor-sse2-study). There it
// is `soft_exact` below, which returns the same double: it is not an
// approximation, and the baseline build's bits do not move.
//
// soft_exact is the emulated FMA of Boldo & Melquiond ("Emulation of FMA and
// correctly rounded sums: proved algorithms using rounding to odd", IEEE TC
// 57(4), 2008): Dekker's exact product a*b = p + e (Veltkamp split), Knuth's
// TwoSum c + p = th + tl, the low part v = e + tl rounded TO ODD (round to
// nearest, then one ulp towards the exact value when that was inexact and
// landed on an even significand), and one round-to-nearest th + v. Rounding
// the low part to odd is what makes the final rounding a single one.
//
// Its fast path holds the operands where every step is exact. Everything
// else goes to `std::fma`, so special values and extreme ranges get libm's
// IEEE answer by construction, not by a re-derivation here:
//   * a or b zero: a*b is an exact (signed) zero and a*b + c is fma's answer,
//     signs of zero included, for any c (inf and NaN too). Handled inline,
//     because the numpy-shaped products multiply by a literal 0.0.
//   * a or b subnormal, inf or NaN; c inf or NaN: std::fma.
//   * |a| or |b| >= 2^996 (the split C*x would overflow), the product's
//     exponents e_a + e_b outside [-960, 1000] (Dekker's error term must not
//     underflow; a*b and c + a*b must not overflow), or |c| >= 2^1001:
//     std::fma.
// Inside it the result is correctly rounded, a subnormal one included: with
// e_a + e_b >= -960 every term is a multiple of 2^-1074, and a sum that
// lands below 2^-1022 has c ~ -a*b, where TwoSum's sum is exact (Sterbenz).
// It assumes round-to-nearest and binary64 evaluation (FLT_EVAL_METHOD 0, so
// no x87); a build where the latter does not hold keeps std::fma. Every
// operation must round exactly as written, which the GCC/clang builds'
// -ffp-contract=off guarantees (and a baseline x86 build has no FMA to
// contract into); under MSVC (/fp:fast) the function is compiled precise and
// kept out of line, so /fp:fast cannot reassociate the error-free transforms.
//
// Gate: tests/test_exact_fma.py (random triples, cancellation, every edge
// class above) against libm's fma and exact rational arithmetic, on both x86
// variants.
//
// soft_exact itself is compiled wherever it can be exact (not MSVC /arch:AVX2,
// whose /fp:fast build has no use for it, and not an x87 build), so the gate
// also runs it on the FMA builds; `exact` takes it only where there is no
// hardware FMA (MW_FMA_EXACT_SOFT).
// (MSVC reports FLT_EVAL_METHOD -1 under /fp:fast, but its x64 target
// evaluates every double in SSE2 registers, and the function is compiled
// precise: _M_X64 is its test.)
#if (defined(_MSC_VER) && !defined(__AVX2__) && defined(_M_X64)) || \
    (!defined(_MSC_VER) && defined(FLT_EVAL_METHOD) && FLT_EVAL_METHOD == 0)
#define MW_FMA_HAVE_SOFT_EXACT 1

MW_FMA_INLINE std::uint64_t bits_of(double x) {
    std::uint64_t u;
    std::memcpy(&u, &x, sizeof u);
    return u;
}

#if defined(_MSC_VER)
#pragma float_control(push)
#pragma float_control(precise, on)
#define MW_FMA_SOFT_FN static __declspec(noinline)
#else
#define MW_FMA_SOFT_FN MW_FMA_INLINE
#endif

MW_FMA_SOFT_FN double soft_exact(double a, double b, double c) {
    // Biased exponent fields: 0 = zero/subnormal, 2047 = inf/NaN, 1023 = 2^0.
    const int ea = static_cast<int>((bits_of(a) >> 52) & 0x7ff);
    const int eb = static_cast<int>((bits_of(b) >> 52) & 0x7ff);
    const int ec = static_cast<int>((bits_of(c) >> 52) & 0x7ff);
    const int eab = ea + eb;  // e_a + e_b + 2046
    if (ea >= 1 && eb >= 1 && ea <= 2018 && eb <= 2018 && eab >= 1086 && eab <= 3046 &&
        ec <= 2023) {
        // Dekker / Veltkamp: a*b = p + e exactly.
        const double split = 134217729.0;  // 2^27 + 1
        const double p = a * b;
        const double ca = split * a, ah = ca - (ca - a), al = a - ah;
        const double cb = split * b, bh = cb - (cb - b), bl = b - bh;
        const double e = ((ah * bh - p) + ah * bl + al * bh) + al * bl;
        // TwoSum: c + p = th + tl exactly.
        const double th = c + p;
        const double tz = th - c;
        const double tl = (c - (th - tz)) + (p - tz);
        // v = RO(e + tl): round to nearest, with the TwoSum residual saying
        // whether (and which way) that was inexact.
        const double v = e + tl;
        const double vz = v - e;
        const double vr = (e - (v - vz)) + (tl - vz);
        double vo = v;
        if (vr != 0.0) {
            std::uint64_t u = bits_of(v);
            if ((u & 1u) == 0) {
                // Away from zero when the residual has v's sign, else towards.
                if ((v > 0.0) == (vr > 0.0))
                    u += 1;
                else
                    u -= 1;
                std::memcpy(&vo, &u, sizeof vo);
            }
        }
        return th + vo;
    }
    if (a == 0.0 || b == 0.0) return a * b + c;
    return std::fma(a, b, c);
}

#if defined(_MSC_VER)
#pragma float_control(pop)
#endif
#undef MW_FMA_SOFT_FN
#else
#define MW_FMA_HAVE_SOFT_EXACT 0
#endif

#if ((defined(__FMA__) || defined(__ARM_FEATURE_FMA)) && !defined(_MSC_VER)) || \
    (defined(_MSC_VER) && defined(__AVX2__)) || !MW_FMA_HAVE_SOFT_EXACT
#define MW_FMA_EXACT_SOFT 0
MW_FMA_INLINE double exact(double a, double b, double c) { return std::fma(a, b, c); }
#else
#define MW_FMA_EXACT_SOFT 1
MW_FMA_INLINE double exact(double a, double b, double c) { return soft_exact(a, b, c); }
#endif

typedef std::complex<double> cdd;

// An optimisation barrier on one double: the value is unchanged, but the
// compiler can no longer see where it came from. It exists for the complex
// helpers below, and it is measured, not decorative. Written with std::fma,
// the real and imaginary halves of a complex product are two isomorphic
// fused ops, and GCC's SLP vectorizer packs them into one 128-bit register
// with shuffles (vunpck/vmovddup/vxorpd) around them. In a ladder like
// t = t * q, where each step waits for the last, those shuffles sit on the
// dependency chain. Measured on the J0/J1 series ladder alone (GCC 13, one
// thread): contracted operator* 0.0459 s, unfused 0.0528 s, fused-and-packed
// 0.0484 s, fused with this barrier 0.0458 s. Hiding one half from the
// vectorizer keeps both halves scalar, which is what GCC's own contraction
// used to emit (that pass runs after SLP has already declined). No
// instruction is generated for it.
MW_FMA_INLINE double opaque(double v) {
#if defined(__GNUC__) && (defined(__x86_64__) || defined(__i386__))
    __asm__("" : "+x"(v));
#elif defined(__GNUC__) && defined(__aarch64__)
    __asm__("" : "+w"(v));
#endif
    return v;
}

// Real x * y: nothing to fuse, here so the templated ladders read the same.
MW_FMA_INLINE double mul(double x, double y) { return x * y; }

// x * y with each part one fused op:
//     re = fma(xr, yr, -(xi*yi)),  im = fma(xr, yi, xi*yr).
// Unlike operator*, it has no Annex G recovery for an (inf, nan) product (the
// __muldc3 call GCC emits behind a NaN test). Apart from the rounding, which
// is the point, the two differ only when an operand is already infinite,
// which the sites that use this never hold on a finite contour. Skipping
// that test is part of why it is cheaper than operator*.
MW_FMA_INLINE cdd mul(const cdd &x, const cdd &y) {
    return cdd(opaque(fma(x.real(), y.real(), -(x.imag() * y.imag()))),
               fma(x.real(), y.imag(), x.imag() * y.real()));
}

// |z|^2 = re*re + im*im with the second product fused into the add: the
// convergence tests' currency (`cnorm`, `mw_norm2`).
MW_FMA_INLINE double norm2(const cdd &z) {
    return fma(z.real(), z.real(), z.imag() * z.imag());
}

// acc + x * y: the fused product `mul`, then one add per part. Not the cross
// terms fused into the add (fma(xr, yr, fma(-xi, yi, accr))): that would put
// two dependent fused ops on the accumulator's chain where this has one add.
MW_FMA_INLINE cdd mul_add(const cdd &x, const cdd &y, const cdd &acc) {
    const cdd p = mul(x, y);
    return cdd(acc.real() + p.real(), acc.imag() + p.imag());
}

// acc + x * s for a real s: one fused op per part.
MW_FMA_INLINE cdd mul_add(const cdd &x, double s, const cdd &acc) {
    return cdd(fma(x.real(), s, acc.real()), fma(x.imag(), s, acc.imag()));
}
MW_FMA_INLINE double mul_add(double x, double s, double acc) { return fma(x, s, acc); }

// x * y - z, in both scalar types (the Miller recurrence's step).
MW_FMA_INLINE double mul_sub(double x, double y, double z) { return fma(x, y, -z); }
MW_FMA_INLINE cdd mul_sub(const cdd &x, const cdd &y, const cdd &z) {
    return cdd(opaque(fma(x.real(), y.real(), fma(-x.imag(), y.imag(), -z.real()))),
               fma(x.real(), y.imag(), fma(x.imag(), y.real(), -z.imag())));
}

}  // namespace mw_fma

#endif  // MOMWIRE_FMA_INLINE_H
