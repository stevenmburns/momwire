// The Sinusoidal-Galerkin basis' crossing pair in C++ (momwire#1718 S11b),
// with no Python in it: what `SinusoidalGalerkinSolver._assemble_Z_mp_mixed`
// takes from `_crossing_fill` on a crossing deck (a wire through the ground
// plane at a crossing junction), over the front's stitched view with its node
// wings (`_stitch_basis_coefs`, `_crossing_wing_view`):
// - `_crossing_rows`: the cross block's rows with support above, through
//   `cross_complete_block_split(ctx, a_idx, b_idx, A, B, rows=)` (one radius)
//   or `cross_complete_blocks_two_radius(..., below_half="rows")` (a
//   two-radius node, antennaknobs plan U5);
// - `self_completions` / `self_completions_two_radius` onto G, negated around
//   the call as the fill does.
//
// The axes are `axis_data` with `SinusoidalBasisSampler`'s COMPLEX samples, so
// every product the split takes is the reference's numpy / scipy route for a
// complex operand (`_left_products`' and `_combine`'s sparse products, the
// numpy form of `_real_matvec_c`), not the real accelerators bs2's crossing
// serve replays. The partition, the ACA, the designed tables through each
// block's memo, the end tables and the completions are the shared pieces
// (`_crossing_shared_inline.h`, `_crossing_split_inline.h`).
//
// Python stays the reference: gated through the fill compare
// (`scripts/core_split/sinusoidal_galerkin_fill_compare.py`) and the
// printout audits.
#pragma once

#include "_nec5_printout_core.h"
#include "_sinusoidal_front_core.h"

#include <complex>
#include <vector>

namespace mw {
namespace sinusoidal_galerkin {

// `_crossing_rows(geom, seg_view, medium, below)`: the basis rows with
// support above (`rows`, ascending), and the cross block's rows `t_rows`
// ((|rows|, n_basis), C order) and the block G takes the transpose of into
// those columns (`t_cols`, the same at one radius). nec4::Declined where this
// copy declines.
struct CrossingRows {
    std::vector<long long> rows;
    std::vector<std::complex<double>> t_rows, t_cols;
};
CrossingRows crossing_rows(const sinusoidal::Front &fr, const nec5::NumpyLinalg &np);

// `self_completions[_two_radius](ctx_x, ax_b, ax_a, out=-G)`, G (n, n) F
// order, negated around the call.
void crossing_completions(const sinusoidal::Front &fr, const nec5::NumpyLinalg &np,
                          std::vector<std::complex<double>> &G);

}  // namespace sinusoidal_galerkin
}  // namespace mw
