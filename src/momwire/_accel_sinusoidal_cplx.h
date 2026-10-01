// The complex-k Galerkin far fill's transcendental sweep (momwire#1224).
//
// `galerkin_far_fill_cplx_impl` (`_accel_sinusoidal.cpp`) used to evaluate
// complex `std::exp` / `std::sin` per (observer, source) pair and per source
// quadrature node, all scalar. This is its Stage B: every transcendental the
// pair needs, from ONE flat table of real distances, in one vector loop onto
// glibc's libmvec (exp and sin, both in glibc 2.28's libmvec).
//
// Defined in its OWN translation unit, `_accel_sinusoidal_cplx.cpp`, for the
// reason `_accel_razor_cplx.h` gives: the loop needs `exp` declared
// `omp declare simd`, that declaration is TU-wide, and in the sinusoidal TU it
// could reach the REAL fill's codegen, whose bytes are a gate (#762). Here it
// reaches nothing else.
#pragma once

#include <cstddef>

// The multiple the caller pads every call to. With the table padded to a
// multiple of this, every entry takes the vector body and none the scalar
// remainder, whose libm differs from libmvec in the last bits — so an entry's
// value depends on its own distance alone, never on where it sits in the
// table. GCC runs the loop 4 lanes wide at AVX2 (libmvec's `_ZGVdN4v_*`); 8 is
// a multiple of that and of every other width a compiler might pick for it
// (2 for SSE2 or NEON, 8 for an unrolled or AVX-512 loop), at a cost of at
// most 7 dummy entries per observer. That is what keeps the complex fill
// COLUMN-INDEPENDENT (see `_SIMD_TAIL_PERIOD` in sinusoidal_galerkin.py): the
// real fill's sweep is not padded (its bytes are frozen) and needs the
// caller's source padding instead; this one needs none.
constexpr size_t SG_CPLX_LANES = 8;

// For i in [0, n), with a = k_im*d[i] and y = k_re*d[i] (so k*d = y + j*a):
//
//     ea[i] = exp(a)
//     em[i] = expm1(a)     degree-13 Taylor for |a| <= 0.35, exp(a) - 1 beyond
//     s[i]  = sin(y)
//     h[i]  = sin(y / 2)
//
// from which the caller spells, cancellation-free,
//
//     e^{-jkd}     = ea*(c - j*s),                 c = cos y = 1 - 2h^2
//     e^{-jkd} - 1 = (em*c - 2h^2) - j*ea*s
//     sin(kd)      = s*cosh(a) + j*c*sinh(a),     sinh a = em*(ea+1)/(2ea)
//
// All five arrays 32-byte aligned. Any `n` computes these values, but only a
// multiple of SG_CPLX_LANES keeps every entry in the vector body, which is
// the position-independence above; the far fill always passes one.
void sg_cplx_phase_sweep(const double *d, size_t n, double k_re, double k_im,
                         double *ea, double *em, double *s, double *h);
