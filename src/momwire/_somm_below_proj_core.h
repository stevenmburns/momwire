// The below/below remainder's projection, Python-free (momwire#1512 B2):
// `_somm_below_proj_core.cpp`; `remainder_field_proj_batch_below` binds it.
#pragma once

#include "_accel_somm_proj_inline.h"

#include <complex>
#include <cstddef>

namespace somm_core {

// out[m * S + n] = t_obs[m] . F_below(obs[m], src[n]) . t_src[n] over the
// below grid's 15-region view (points and tangents (M, 3) / (S, 3) row
// major), and the query's extremes (max R1, min and max theta) over every
// pair, reduced in item order. `blocked` / `lanes` as the binding's.
void below_field_proj_batch(const somm_proj::GridView &G, size_t n_reg, double th_min,
                            double th_band_floor_hi, double th_band_lo_hi, double th_band_hi,
                            std::ptrdiff_t M, std::ptrdiff_t S, const double *obs,
                            const double *t_obs, const double *src, const double *t_src,
                            double ground_z, double k_p, std::complex<double> k_m,
                            std::complex<double> *out, double &mx_r1, double &mn_th,
                            double &mx_th, bool blocked, bool lanes);

// Whether the held-modulus divide-out (glibc's cexp split) is this libm's.
bool below_cexp_split_ok();

// `_zero_past_cap`'s test for one pair of `below_pair_geometry`'s (rho, hh):
// numpy's hypot then the root, past the cap. One copy, for the zeroing and
// the lazy grid's census.
inline bool below_past_cap(double rho, double hh, double r1_cap) {
    return std::sqrt(rho * rho + hh * hh) > r1_cap;
}

// The lazy below grid's census of `below_field_proj_batch` (momwire#1512):
// every node its reads take, marked in `past` for a pair `_zero_past_cap`
// would zero at `r1_cap` and in `kept` otherwise, and the batch's extremes
// (max R1, min and max theta), the same floats it reduces.
void below_field_proj_nodes(const somm_proj::GridView &G, double th_min, double th_band_floor_hi,
                            double th_band_lo_hi, double th_band_hi, std::ptrdiff_t M,
                            std::ptrdiff_t S, const double *obs, const double *src,
                            double ground_z, double r1_cap, somm_proj::NodeMarks &kept,
                            somm_proj::NodeMarks &past, double &mx_r1, double &mn_th,
                            double &mx_th);

}  // namespace somm_core
