// PROTOTYPE (momwire#1517 step 1, not merged): bs2's free-space fill in C++,
// the orchestration over the Python-free kernels (`_bspline_core.h`):
// `BSplineSolver._build_J_blocks` (off-edge moments with the pair-order
// ladder, the phase-guard / radius-run / source-tube splits and the extended
// kernel; same-edge static + regularised blocks) and `_assemble_Z`.
#pragma once

#include "_bspline_front_core.h"

#include <complex>
#include <cstdint>
#include <utility>
#include <vector>

namespace mw {
namespace bspline {

using cdouble = std::complex<double>;

struct FillParams {
    double k = 0;             // the solver's real wavenumber
    int degree = 2;
    int64_t n_qp_pair = 8;    // `DEFAULT_N_QP_PAIR` (free space)
    std::vector<std::pair<double, int64_t>> ladder;  // `pair_order_ladder`
    int64_t n_qp_same_edge = 4;
    bool extended_kernel = true;
};

// `_build_J_blocks(geom, k)`: (d+1, d+1, N, N) C order. `radius_per_wire`:
// the conductor radius of each wire.
std::vector<cdouble> build_J_blocks(const Geometry &g, const std::vector<double> &radius_per_wire,
                                    const FillParams &fp);

// `BSplineSolver.pair_order_ladder` for a free-space deck: the default
// ladder's tiers below `n_qp_pair`, normalised.
std::vector<std::pair<double, int64_t>> default_ladder(int64_t n_qp_pair);

}  // namespace bspline
}  // namespace mw
