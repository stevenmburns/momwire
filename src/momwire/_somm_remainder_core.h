// The B-spline Sommerfeld remainder block with no Python (momwire#1517 step
// 3): `sommerfeld_remainder_bspline_Q`'s kernel -- the banded rectangular
// route and the symmetric route of the dense obs == src block -- moved
// verbatim from `_accel_somm.cpp`, which keeps only the binding (argument
// checks, the grid view, the route choice). See the block comments in
// `_somm_remainder_core.cpp`.
#pragma once

#include "_accel_somm_proj_inline.h"

#include <complex>
#include <cstddef>
#include <cstdint>

namespace somm_core {

// Q (nI, nJ) C order: obs_nodes (nsI, q, 3), obs_tang (nsI, 3), W_obs
// (d1, nsI, q); src likewise; loc_I (nI, d1) into the obs segments with
// pI (nI, d1, d1); loc_J / pJ into the src segments. `symmetric`: the dense
// block's call (obs == src throughout), which takes the symmetric route.
// `max_jf_bytes` 0: the 64 MiB default slab. Cancellation polls
// `cancel_flag` (0 = none) and throws AbortedError after the loops drain.
void remainder_bspline_Q(const somm_proj::GridView &G, double ground_z, double k,
                         const double *obs_nodes, const double *obs_tang, const double *W_obs,
                         std::ptrdiff_t nsI, const double *src_nodes, const double *src_tang,
                         const double *W_src, std::ptrdiff_t nsJ, std::ptrdiff_t q, int d1,
                         const int64_t *loc_I, const double *pI, std::ptrdiff_t nI,
                         const int64_t *loc_J, const double *pJ, std::ptrdiff_t nJ, bool symmetric,
                         size_t max_jf_bytes, std::complex<double> *Q, uintptr_t cancel_flag);

}  // namespace somm_core
