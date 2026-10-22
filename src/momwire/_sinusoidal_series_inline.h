// The two series helpers the sinusoidal kernels share (momwire#1607 B1),
// moved verbatim from `_accel_sinusoidal.cpp`, where the Galerkin far
// fills use them, so the point-matched core (`_sinusoidal_core.cpp`) can
// use them too: each TU instantiates its own copy, as before.
#pragma once

#include <cmath>
#include <complex>

// sin(u) − u, without the u²/6 cancellation of the literal difference.
//
// Templated on the scalar type for momwire#980 step E: the complex-k twin
// needs the same series at std::complex<double>, and these two helpers are
// the ONLY code the real fill and the twin share (the fills themselves are
// separate bodies — see `galerkin_far_fill_cplx_impl` and the issue it
// cites). The branch compares std::abs, which is |u| for a complex argument
// and exactly `fabs` for a real one, matching what `_sin_minus_arg` does on
// the Python side. The real path's bytes are unchanged by the templating and
// that is checked by rebuilding and hashing the fill's output, not by
// reading this diff (the momwire#762 protocol).
template <typename T>
static inline T sin_minus_arg(T u) {
    T u2 = u * u;
    if (std::abs(u) < 0.1) {
        return -(u * u2) / 6.0 *
               (1.0 - u2 / 20.0 *
                          (1.0 - u2 / 42.0 *
                                     (1.0 - u2 / 72.0 * (1.0 - u2 / 110.0))));
    }
    return std::sin(u) - u;
}

// asinh(x) − x, taking the caller's t = asinh(x) so the fill pays for one
// asinh either way. The series is in t because sinh t − t converges
// factorially where the asinh series does not.
template <typename T>
static inline T asinh_minus_arg_from_t(T t) {
    T t2 = t * t;
    if (std::abs(t) < 1.0) {
        return -(t * t2) / 6.0 *
               (1.0 + t2 / 20.0 *
                          (1.0 + t2 / 42.0 *
                                     (1.0 + t2 / 72.0 *
                                                (1.0 + t2 / 110.0 *
                                                           (1.0 + t2 / 156.0)))));
    }
    return -(std::sinh(t) - t);
}
