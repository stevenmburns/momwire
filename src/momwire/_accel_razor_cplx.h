// The complex-k bracket of razor's segment moments, vectorised (Design D4).
//
// e^{-jkR} - 1 at every node of one entry, k = k_re + j*k_im with Im k <= 0.
// Defined in its OWN translation unit, `_accel_razor_cplx.cpp`, because it
// needs `exp` declared `omp declare simd` and that declaration is TU-wide:
// placed in `_accel_razor.cpp` it changed the codegen of that TU's other
// kernels (DESIGN-D §4). Here it can reach nothing else.
#pragma once

#include <cstddef>

// Nodes per vector call. The caller pads every call to a multiple of this,
// so every node of every entry takes the vector body and none takes the
// scalar remainder, whose libm differs from libmvec in the last bits: a
// node's value then does not depend on n_qp mod 4.
constexpr size_t RAZOR_CPLX_LANES = 4;

// Nodes per call, and the size of the caller's per-thread buffers. A
// multiple of RAZOR_CPLX_LANES. The nodes are independent, so an order above
// this is walked in chunks and the answer does not depend on the width.
constexpr size_t RAZOR_CPLX_CHUNK = 64;

// For q in [0, n): with a = k_im*R[q], y = k_re*R[q],
//
//     re[q] + j*im[q] = e^{-jkR[q]} - 1
//                     = [ expm1(a) cos(y) - 2 sin^2(y/2) ] - j [ exp(a) sin(y) ]
//
// `n` must be a multiple of RAZOR_CPLX_LANES and at most RAZOR_CPLX_CHUNK,
// and the three arrays 32-byte aligned. R[q] > 0.
void razor_cplx_brackets(const double *R, size_t n, double k_re, double k_im,
                         double *re, double *im);
