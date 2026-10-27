// numpy's complex sin / cos / log / exp / sqrt are the C library's csin /
// ccos / clog / cexp / csqrt (npymath wraps them wherever the platform has
// them, macOS included). `std::sin(std::complex<double>)` is not always that
// call: libstdc++ forwards to the C99 builtins, but libc++ (Apple clang)
// computes its own formulas, which differ from Apple's libm by bits on
// complex arguments (momwire#1607 B6a, the macOS arm64 lane). These call the
// C library on every GNU-compatible compiler, gcc and clang alike, and fall
// back to std:: elsewhere (MSVC, where no bit identity is promised).
#pragma once

#include <cmath>
#include <complex>

// Whether the platform's numpy / SciPy build contracts its plain-C complex
// arithmetic: numpy's `CDOUBLE_divide` and scalar product
// (`cdouble_ctype_multiply`), SciPy's sparsetools complex product. An x86-64
// wheel's baseline has no FMA to contract into (unfused); an arm64 wheel's
// compiler contracts each `a * b + c` into one fused multiply-add. Measured on
// macOS 15 arm64 (numpy 2.5.4, SciPy 1.18.1): 4000 of 4000 random quotients
// and scalar products, and 480 of 480 `csr_matvec(s)` outputs, in exactly the
// fused forms the copies spell (momwire#1607 B6a / B6b). Linux aarch64's gcc
// contracts by default too; that platform is unmeasured.
#if defined(__aarch64__)
#define MW_NUMPY_CONTRACTS 1
#else
#define MW_NUMPY_CONTRACTS 0
#endif

namespace mw_libm_c {

using cd = std::complex<double>;

// A real sine and cosine, each its own C library call. Apple clang merges a
// `sin(x)` and a `cos(x)` of one argument into one `__sincos_stret` call,
// whose sine is not `sin`'s (68 of 40000 arguments in [0, 4) on macOS 15
// arm64); numpy and CPython call `sin` and `cos` apart, so a copy that holds
// their bits keeps the two calls in separate functions there (momwire#1733).
// Elsewhere these are `std::sin` / `std::cos` exactly.
#if defined(__APPLE__) && defined(__GNUC__)
__attribute__((noinline)) inline double rsin(double x) { return std::sin(x); }
__attribute__((noinline)) inline double rcos(double x) { return std::cos(x); }
#else
inline double rsin(double x) { return std::sin(x); }
inline double rcos(double x) { return std::cos(x); }
#endif

#if defined(__GNUC__)
#define MW_LIBM_C_CALL(builtin)                       \
    __complex__ double c;                             \
    __real__ c = z.real();                            \
    __imag__ c = z.imag();                            \
    c = builtin(c);                                   \
    return cd(__real__ c, __imag__ c)

inline cd csin(cd z) { MW_LIBM_C_CALL(__builtin_csin); }
inline cd ccos(cd z) { MW_LIBM_C_CALL(__builtin_ccos); }
inline cd clog(cd z) { MW_LIBM_C_CALL(__builtin_clog); }
inline cd cexp(cd z) { MW_LIBM_C_CALL(__builtin_cexp); }
inline cd csqrt(cd z) { MW_LIBM_C_CALL(__builtin_csqrt); }

#undef MW_LIBM_C_CALL
#else
inline cd csin(cd z) { return std::sin(z); }
inline cd ccos(cd z) { return std::cos(z); }
inline cd clog(cd z) { return std::log(z); }
inline cd cexp(cd z) { return std::exp(z); }
inline cd csqrt(cd z) { return std::sqrt(z); }
#endif

}  // namespace mw_libm_c
