// The crossing table (momwire#1860 L5, the opt-in feature `crossing_table`),
// Python-free: `_crossing_table_core.cpp`; `_near_interface_accel.cpp` binds
// it. Its nodes come from the column twin (`ni_core::six_columns`).
#pragma once

#include <complex>
#include <cstddef>
#include <cstdint>

namespace ni_core {

// The table on the uniform grid rho_i = i h[0], z_j = j h[1], z'_k = -k h[2]
// (i < n[0], j < n[1], k < n[2]): vals (n0, n1, n2, 6) row-major holds
// R (f - S) per kernel, f from the column twin (one column per rho node, its
// members every (z, z') node, on the factored route) and S the closed form
// of `_crossing_table_inline.h`. The node at R = 0 holds 0.
void crossing_table_build(double k_p, std::complex<double> k_m, const double *h,
                          const std::ptrdiff_t *n, double lam_mult, int p, double detour,
                          int n_threads, const double *gx, const double *gw, int ng,
                          uintptr_t cancel_flag, std::complex<double> *vals);

// The table's kernels at m rows (rho, z, z'), each inside the grid: the
// tricubic read of R (f - S), divided by R, plus S. out is (m, 6) row-major.
void crossing_table_lookup(double k_p, std::complex<double> k_m, const double *h,
                           const std::ptrdiff_t *n, const std::complex<double> *vals,
                           const double *rho, const double *z, const double *zp,
                           std::ptrdiff_t m, int n_threads, std::complex<double> *out);

}  // namespace ni_core
