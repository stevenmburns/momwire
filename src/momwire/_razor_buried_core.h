// razor-2p's BURIED fill in C++ (momwire#1512 step 3): the matrix of a deck
// with wires below a GN 0 / GN 2 ground, the same bits as
// `RazorSolver._assemble_Z_crossing` / `_assemble_Z_below_plane`. No Python.
//
// A built `Front` whose `buried` route is set (`_refuse_buried_geometry`)
// fills medium by medium, in `_assemble_Z_crossing`'s order:
//
//   1. the ABOVE medium on its own sub-geometry (`_medium_geometry`: its
//      segments, the bases it owns, each crossing tent's above wing as a half
//      tent whose other wing is a ghost at sigma = 0, T2 chopped at the node),
//      over the composing ground: `Z_a = D_a - (C2 img_a + Q_a)`;
//   2. `Z = 0; Z[rows_a, rows_a] += Z_a`;
//   3. the BELOW medium, folded window by window:
//      `Z[rows_b, rows_b] += D_b - (A_m img_b + Q_b)`;
//   4. per source radius, the forward cross block (below sources), then the
//      reversed one (above sources): `Z -= t`;
//   5. the node charges into the crossing tents' columns;
//   6. the loading on the full geometry.
//
// A wholly-buried deck is `_assemble_Z_below_plane` on the whole geometry.
//
// This header grows stage by stage (momwire#1512 B2-B5); `BuriedStages`
// reports how far a fill got, for the gates (`scripts/core_split/
// buried_z_compare.py`), which compare each stage's matrix with Python's.
#pragma once

#include "_razor_front_core.h"
#include "_razor_solve_core.h"

#include <memory>
#include <string>
#include <vector>

namespace mw {
namespace razor {

// One medium's own fill (`_medium_geometry`'s (geom, rows, chop)): a Front
// over the sub-geometry, prepared, and each sub-row's basis in the FULL
// geometry.
struct MediumPart {
    std::unique_ptr<Front> front;
    std::vector<int64_t> rows;
};

// `_medium_geometry(geom, side)` and `_assemble_Z_prepare(geom_side,
// chop=chop_side, loading=False)` over a built buried Front. `below` picks
// the side. The above side's front carries the composing ground.
MediumPart medium_part(const Front &f, bool below);

// The stages a buried fill reached, column-major (n_basis, n_basis) each, in
// `buried_z_compare.py`'s names. `reached` lists them in order; the last one
// reached is what the fill stopped at.
struct BuriedStages {
    std::vector<std::string> reached;
    std::vector<std::vector<cd>> z;
};

// Fill `f`'s buried matrix as far as this build serves it, recording each
// stage. Returns "" when the whole matrix (loading included) was filled into
// `out` (column-major, n_basis^2), else the name of the first stage not
// served.
std::string fill_buried(const Front &f, std::vector<cd> &out, BuriedStages *stages,
                        uintptr_t cancel_flag);

}  // namespace razor
}  // namespace mw
