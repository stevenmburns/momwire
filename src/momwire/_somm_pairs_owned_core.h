// The Sommerfeld remainder's projection at LISTED (observer, source) pairs
// (`_sommerfeld.remainder_field_proj_owned`, momwire#1201), with no Python
// (momwire#1718 S5b): the kernel `_accel_somm_pairs.cpp` bound, moved here so
// the Sinusoidal-Galerkin route's graded correction runs the same text. Its
// own translation unit, as the binding's was, for `proj_one`'s inlining
// (momwire#1194): a second caller beside `remainder_graded_inner` would move
// that kernel's codegen.
#pragma once

#include "_accel_somm_proj_inline.h"

#include <complex>
#include <cstddef>
#include <cstdint>

namespace somm_core {

// out[n] = t_obs[owner[n]] . F(obs[owner[n]], src[n]) . t_src[n] for the S
// listed sources (obs / t_obs (M, 3), src / t_src (S, 3)). OpenMP over
// sources; throws AbortedError when `cancel_flag` is raised.
void remainder_field_proj_owned(const somm_proj::GridView &G, double ground_z, double k, const double *obs,
                                const double *t_obs, const double *src, const double *t_src, const int64_t *owner,
                                std::ptrdiff_t S, std::complex<double> *out, uintptr_t cancel_flag);

}  // namespace somm_core
