// numpy's complex sin / cos / log / exp / sqrt are the C library's csin /
// ccos / clog / cexp / csqrt (npymath wraps them wherever the platform has
// them, macOS included). `std::sin(std::complex<double>)` is not always that
// call: libstdc++ forwards to the C99 builtins, but libc++ (Apple clang)
// computes its own formulas, which differ from Apple's libm by bits on
// complex arguments (momwire#1607 B6a, the macOS arm64 lane). These call the
// C library on every GNU-compatible compiler, gcc and clang alike, and fall
// back to std:: elsewhere (MSVC, where no bit identity is promised).
#pragma once

#include <complex>

namespace mw_libm_c {

using cd = std::complex<double>;

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
