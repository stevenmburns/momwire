// The point-observer column twin, Python-free (momwire#1653):
// `_near_interface_point_core.cpp`; `_near_interface_point_accel.cpp` binds
// it (`near_interface_point_columns`), and the buried near-field readout
// (`_near_buried_core.cpp`) calls it directly.
#pragma once

#include <complex>
#include <cstddef>

namespace ni_point {

constexpr int N_POINT_KEYS = 4;  // POINT_KEYS: dRhoV, dzV, dRhoW, dz'V

// `point_keys_columns` over `nc` concatenated exact-rho columns: column c's
// members are [offsets[c], offsets[c+1]) of (z, zp); out is (n, 4) row-major
// in POINT_KEYS order. `sdp`: a far column takes the steepest-descent rule
// (momwire#1653, D1b's route and crossover), as `point_keys_columns` does
// under `_SDP`; false is the head / mid / tail rule everywhere. Throws std::invalid_argument on a bad member and
// std::runtime_error if the J1 weights leave the rule's nodes.
void point_columns(double k_p, std::complex<double> k_m, const double *rho,
                   const std::ptrdiff_t *offsets, std::ptrdiff_t nc, const double *z,
                   const double *zp, std::ptrdiff_t n, double lam_mult, int p, double detour,
                   int n_threads, const double *gx, const double *gw, int ng,
                   std::complex<double> *out, bool sdp = true);

}  // namespace ni_point
