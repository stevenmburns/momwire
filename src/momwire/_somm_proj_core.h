// The Sommerfeld remainder's projection and moment reduction with no Python
// (`_somm_proj_core.cpp`, momwire#1479 S3): the kernels behind
// `remainder_field_proj_batch` and `remainder_moment_reduce`, which bind them.
#pragma once

#include "_accel_somm_proj_inline.h"

namespace somm_core {

// out[m * S + n] = t_obs[m] . F(obs[m], src[n]) . t_src[n] over the grid view;
// points and tangents (M, 3) / (S, 3) row-major. n_reg is the view's region
// count. lanes as the binding's (the per-pair loop when false or not built).
// Cancellation polls `cancel_flag` (0 = none) and throws AbortedError after
// the loop drains.
void field_proj_batch(const somm_proj::GridView &G, size_t n_reg,
                      std::ptrdiff_t M, std::ptrdiff_t S, const double *obs,
                      const double *t_obs, const double *src,
                      const double *t_src, double ground_z, double k,
                      std::complex<double> *out, uintptr_t cancel_flag,
                      bool lanes);

// The lazy grids' census of `field_proj_batch` (momwire#1512): every node
// its reads take over the M x S pairs, marked (tangents do not move a read).
void field_proj_nodes(const somm_proj::GridView &G, std::ptrdiff_t M, std::ptrdiff_t S,
                      const double *obs, const double *src, double ground_z,
                      somm_proj::NodeMarks &marks);

// Whether this build has the projection's lanes, and how many pairs went
// through them in this process.
bool proj_lanes_built();
unsigned long long proj_lane_pairs();

// mom[o, j, p] = sum_k proj[o, j*q + k] * W[p, j, k]: proj (M, N*q), W
// (P, N, q) real, out (M, N, P); numpy's unfused, k-ordered arithmetic.
void moment_reduce(const std::complex<double> *proj, const double *W,
                   std::ptrdiff_t P, std::ptrdiff_t N, std::ptrdiff_t q,
                   std::ptrdiff_t M, std::complex<double> *out);

}  // namespace somm_core
