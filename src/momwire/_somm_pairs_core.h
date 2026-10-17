// The Sommerfeld remainder at LISTED (observer, source) pairs, with no
// Python (momwire#1517 step 3): the graded rule's fused inner integral
// (`remainder_graded_inner`, momwire#1201), moved verbatim from
// `_accel_somm_pairs.cpp`, which keeps the binding. Its own translation unit
// for the reason that file gives (`proj_one`'s inlining, momwire#1194).
#pragma once

#include "_accel_somm_proj_inline.h"

#include <complex>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace somm_core {

// v[m, P] (M, d1) C order: observer node m (obs, t_obs (M, 3)) owns the
// source panels [ptr[m], ptr[m+1]) of the segment a0[m] + t dvec[m] of
// physical length len[m]; each panel [p0, p1] takes the rule (xg, wg) on
// [-1, 1] (ng nodes). Throws AbortedError when `cancel_flag` is raised.
void remainder_graded_inner(const somm_proj::GridView &G, double ground_z, double k, const double *obs,
                            const double *t_obs, const double *a0, const double *dvec, const double *t_src,
                            const double *len, const int64_t *ptr, std::ptrdiff_t M, const double *p0,
                            const double *p1, const double *xg, const double *wg, std::ptrdiff_t ng, int d1,
                            std::complex<double> *out, uintptr_t cancel_flag);

// `_quadrature.remainder_qp_pairs`' per-source scan (moved verbatim from
// `_accel_qp_pairs.cpp`): for each source j, the observer segments (n_node
// nodes each, obs (n_seg n_node, 3)) whose order is certainly above `base`,
// as (row, q), or `unsure[j]` set when the numpy body must decide.
struct QpHit {
    int64_t row;
    int64_t q;
};
void remainder_qp_pairs_scan(const double *obs, std::ptrdiff_t M, int64_t n_node, const double *mir_l,
                             const double *d, const double *dd, const double *lengths, std::ptrdiff_t S,
                             int64_t base, int64_t cap, double c, double edge_tol,
                             std::vector<std::vector<QpHit>> &hits, std::vector<unsigned char> &unsure);

}  // namespace somm_core
