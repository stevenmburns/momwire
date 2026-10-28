// The point-matched sinusoidal lane's two transmitted pair classes
// (momwire#1718 S12): `SinusoidalSolver._transmitted_block` on a mixed deck
// -- wires on both sides of the interface and no crossing junction -- read at
// the segment midpoints (`row_group = 1`) over the below->above table
// `_somm_transmitted_core` fills. Its own TU so the fill core's other
// kernels keep their codegen.
#pragma once

#include "_sinusoidal_front_core.h"
#include "_somm_transmitted_core.h"

#include <complex>
#include <vector>

namespace mw {
namespace sinusoidal {

// `_transmitted_block(geom, medium, plan, src_keep, obs_keep, obs_below,
// seg_c, seg_t, row_group=1, cos_shape="cos")`: the (3, n_obs, n_src)
// block (each part row major), observers the midpoints of `obs_idx` and
// sources the segments `src_idx`, at the plan's field order `q`. Declines
// (`nec4::Declined`) where the grid's eval refuses.
std::vector<std::vector<std::complex<double>>> transmitted_block(const Front &fr, const somm_trans::CrossPlan &plan,
                                                                 long long q, double mu, bool obs_below,
                                                                 const std::vector<long long> &obs_idx,
                                                                 const std::vector<long long> &src_idx);

}  // namespace sinusoidal
}  // namespace mw
