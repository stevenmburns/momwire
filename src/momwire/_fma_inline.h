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
#ifndef MOMWIRE_FMA_INLINE_H
#define MOMWIRE_FMA_INLINE_H

#include <cmath>
#include <complex>

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
