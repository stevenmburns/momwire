// momwire#1517 step 4b-2: bs2's crossing serve in C++ (`_crossing_fill`'s
// `cross_complete_block_split` and `self_completions` over the B-spline
// basis, composed into Z as `compute_Z_operator_buried` composes them),
// Python-free. The pieces razor's crossing core has are shared through
// `_crossing_shared_inline.h`; the split's cluster partition, its direct
// groups and the self completions are here.
#pragma once

#include "_bspline_fill_core.h"

#include <complex>
#include <cstdint>
#include <vector>

namespace mw {
namespace bspline {

// The stages a test hook reads back: the main sandwich (`_main_split`, c1
// folded), the cross block after the ends and the corner, and the self
// completions alone; each (n_basis, n_basis) C order.
struct CrossingStages {
    std::vector<cdouble> t_main, t_ab, completions;
};

// `compute_Z_operator_buried`'s crossing branch: on a one-radius deck the
// cross block `t` (above rows x below columns), `Z -= t; Z -= t.T`, then the
// self completions accumulated into Z; at a TWO-RADIUS node (`a_below` > 0,
// antennaknobs plan U5) the two blocks at their radii, `Z -= t_above; Z -=
// t_below.T`, and the two-radius completions, `a_wire` then the smaller
// radius (the axes'). `a_idx` / `b_idx` the above / below segments. Throws FrontNotPorted for what is not ported (a far block the
// split factors by ACA).
void crossing_serve(std::vector<cdouble> &Z, const Geometry &g, const Basis &b, double a_wire,
                    const BuriedMedium &bm, double ground_z, double omega, double mu, double eps,
                    const std::vector<int64_t> &a_idx, const std::vector<int64_t> &b_idx,
                    const nec5::NumpyLinalg &np, CrossingStages *stages = nullptr,
                    double a_above = 0.0, double a_below = 0.0);

}  // namespace bspline
}  // namespace mw
