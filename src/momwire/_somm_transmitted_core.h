// The transmitted Sommerfeld family (`_sommerfeld_transmitted`) with no
// Python (momwire#1718 S12): the below->above grid a mixed deck's cross
// blocks read -- wires on BOTH sides of the interface with no crossing
// junction -- and its projected pair table, shared by every executable whose
// basis fills those blocks (bs2's field-form Galerkin, the point-matched
// sinusoidal lane, the Sinusoidal-Galerkin fill).
//
//   cross_extents   `serve_plan`'s cross-medium section: the observer radius
//                   and source-depth extents over the plan's nodes and its
//                   three refusals (range, depth, grazing), reported by name
//                   for the caller to decline;
//   get_grid        `get_grid_below_above`: the exact in-process key, the
//                   radius bucket, and `TransmittedGrid(...)` -- the ladder,
//                   the (ln R, theta) lattice, `t_surfaces_direct` per rung
//                   over the six integrals (`_somm_transmitted_six_inline.h`,
//                   the code the extension's `transmitted_six_integrals_batch`
//                   runs) and the divide-out, op for op as numpy computes it;
//   project         `transmitted_field_proj_below_to_above` /
//                   `_above_to_below` on the accelerated route: the pair
//                   kernel the extension's `transmitted_field_proj_batch`
//                   runs, and the query's extremes, which `eval_refusal`
//                   reads back as `TransmittedGrid.eval`'s four refusals.
//
// Every refusal here is a NAME for the caller's decline: the executables
// serve only what Python serves, and a geometry Python refuses inside the
// fill is declined rather than worded.
#pragma once

#include <complex>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

namespace somm_trans {

using cd = std::complex<double>;

// A geometry `_sommerfeld_transmitted` (or `serve_plan`'s cross section)
// raises a ValueError on: the caller declines it by name.
struct Refused : std::runtime_error {
    using std::runtime_error::runtime_error;
};

// `TransmittedGrid`'s state as `eval` and the projection read it.
struct Grid {
    cd eps_t, k_m;
    double k2 = 0, omega = 0, mu = 0, lam_p = 0, lam_m = 0;
    double r_min = 0, r_cap = 0, r_max = 0;
    std::vector<double> zp_nodes;
    int64_t n_zp = 0;
    double zp_min = 0, zp_max = 0;
    double lnr0 = 0, dlnr = 0;
    int64_t n_r = 0;
    double th_min = 0, th0 = 0, dth = 0;
    int64_t n_th = 0;
    double lnz0 = 0, dlnz = 0;
    std::vector<cd> vals;  // (n_zp, 5, n_r, n_th), C order: TrhoV TzV TrhoH TphiH TzH
};

// `r_min_bucket(r_min, lam_p)`; `has_r_min` false is `r_min=None`.
double r_min_bucket(bool has_r_min, double r_min, double lam_p);
// `grazing_floor(r_max, zp_min)`, radians.
double grazing_floor(double r_max, double zp_min);
// `grid_extent(k2, r_max, zp_min, r_min=r_min)`.
void grid_extent(double k2, double r_max, double zp_min, bool has_r_min, double r_min, double &rmin,
                 double &rmax, double &th_floor);
// `ladder_nodes(zp_min, zp_max, lam_m)`: the |z'| rungs, ascending.
std::vector<double> ladder_nodes(double zp_min, double zp_max, double lam_m);

// `t_surfaces_direct(eps_t, k2, rho, z, zp, rtol=1e-9, omega=, mu=)` on the
// accelerated route over n points sharing one source depth `zp` (< 0): `out`
// is (5, n), the surfaces in `_SURF_KEYS_T` order.
void t_surfaces(cd eps_t, double k2, double omega, double mu, const double *rho, const double *z, double zp,
                size_t n, cd *out, uintptr_t cancel_flag);

// `TransmittedGrid(eps_t, k2, r_max, zp_min, zp_max, rtol=1e-9, omega=,
// mu=, r_min=)`.
Grid make_grid(cd eps_t, double k2, double r_max, double zp_min, double zp_max, double omega, double mu,
               bool has_r_min, double r_min, uintptr_t cancel_flag);

// `get_grid_below_above(eps_t, k2, r_max, zp_min, zp_max, omega, mu=mu,
// r_min=r_min)`: the process's memo under the Python key (eps~, k2, the
// radius bucket, the bucketed r_min, |zp_min|, |zp_max|, omega, mu), so a
// repeat frequency reads the grid it filled.
const Grid &get_grid(cd eps_t, double k2, double r_max, double zp_min, double zp_max, double omega, double mu,
                     double r_min, uintptr_t cancel_flag = 0);

// The query's extremes over one projection call (`transmitted_field_proj_
// batch`'s six returns).
struct Extremes {
    double r_lo = 0, r_hi = 0, th_lo = 0, th_hi = 0, zp_lo = 0, zp_hi = 0;
};

// The (A, B) projected table in ROLE order -- `above` (A, 3) with its
// tangents, `below` (B, 3) with its -- as `transmitted_field_proj_batch`
// computes it: `transposed` false is the upward table (observer above),
// true the downward one's role-order table (its caller transposes it into
// (n_obs, n_src)). `out` is A * B, row major.
void project(const Grid &g, const double *above, const double *t_above, size_t A, const double *below,
             const double *t_below, size_t B, double ground_z, double k_p, cd k_m, bool transposed, cd *out,
             Extremes &ext);

// `TransmittedGrid.eval` of the extremes (`_proj_accel`'s check): the name of
// the refusal it raises, or empty when the query is served.
std::string eval_refusal(const Grid &g, const Extremes &ext);

// `serve_plan`'s cross-medium section over the plan's nodes: `obs_a` (n_a, 3)
// above, `obs_b` (n_b, 3) below. Throws `Refused` with the refusal's name
// (range, depth, grazing) where Python raises.
struct CrossPlan {
    double r_cross_max = 0, r_cross_min = 0, zp_min = 0, zp_max = 0;
};
CrossPlan cross_extents(const double *obs_a, size_t n_a, const double *obs_b, size_t n_b, double ground_z,
                        double k_p, cd k_m);

// `near_q_factor(*cross_pair_separation(seg_l, seg_r, a_idx, b_idx))`:
// `seg_l` / `seg_r` (n, 3) over every segment.
int64_t near_q_factor(const double *seg_l, const double *seg_r, const std::vector<int64_t> &a_idx,
                      const std::vector<int64_t> &b_idx);

}  // namespace somm_trans
