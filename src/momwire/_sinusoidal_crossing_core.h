// The point-matched sinusoidal lane's crossing rows in C++ (momwire#1607
// B6b), with no Python in it: `SinusoidalSolver._crossing_point_rows`, each
// medium's midpoints observing the other medium's part of every basis,
// through `_crossing_fill.point_observer_block` over the two axes
// `axis_data` builds with `SinusoidalBasisSampler`'s samples.
//
// The designed tables and the point family are the shared cores'
// (`ni_eval::designed_index` through one block's memo, `ni_eval::point_tables`,
// the block's `sheet_plan`); this file is the glue, in Python's order:
//   * the axes: per segment Gauss nodes (the near knobs: q 4, growth 4,
//     G4 panels graded toward the plane on a segment touching it), the CSR
//     samples F / Fd (`samples`, zeros eliminated) and the signed end table;
//   * per direction a fresh memo, its sheet plan over the (observers x
//     source nodes) pairs, the main block's six tables and point family over
//     the grid's distinct folded triples, then the ends' (the reference
//     route `_point_observer_block_dense` the chunked one is gated against);
//   * the contraction `F @ (...).T` as scipy's `csr_matvecs` sums it, the
//     ends' terms in end order, then C1, added into Z's rows.
//
// Python stays the reference: gated through `tests/test_nec4_run_core.py`
// and the fill compare (`scripts/core_split/sinusoidal_fill_compare.py`).
#pragma once

#include "_sinusoidal_front_core.h"

#include <complex>
#include <vector>

namespace mw {
namespace sinusoidal {

// `_crossing_point_rows(geom, seg_view, medium, below, into=G)`: the two
// directions' rows added into `G` ((N, N), C order). Throws FrontError where
// Python raises, nec4::Declined where this copy declines.
void crossing_point_rows(const Front &fr, std::vector<std::complex<double>> &G);

}  // namespace sinusoidal
}  // namespace mw
