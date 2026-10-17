// The NEC-5 seam's structure half for razor-2p in C++: the engine-neutral
// seam (`_nec5_seam_core.h`: the ladder, the mesh, the ports, the cards)
// with razor-2p's answers to what a basis decides, so a parsed deck comes
// out as the arguments the seam hands `RazorSolver` (`_razor_front_core.h`'s
// `FrontInputs`) and the front half built on them, or the refusal the seam
// prints, with no Python anywhere:
//
//   * `basis()`: "razor-2p", the delta-gap spelling (razor is not in
//     `_CUT_SPELLING`) and a gap on a crossing node (`_crossing_spelling`:
//     razor carries ONE tent across the node, so it has no junction ports);
//   * `check_can_host`: `_check_basis_can_host` on razor's capability row,
//     where only the inert one-segment wire refuses (node gaps and knot feeds
//     are served, and so are contact and buried wires over a medium);
//   * `solver_arguments`: `_solver_for` / `construct_solver` for
//     `RazorSolver`, with the extended kernel by default
//     (`extended_kernel_default_refusal`: a wire below the plane takes the
//     reduced kernel, with an advisory the printout does not carry);
//   * `build`: `RazorSolver(...)` and `compute_port_solution`'s front half
//     (construction, geometry, tables, port columns), whose decline (e.g. the
//     buried fill, momwire#1512) declines the deck as "front:<reason>".
//
//     mw::nec5::SeamResult r = mw::nec5::razor_seam(latin1_text);
//     for (const mw::nec5::Prepared &p : r.prepared)   // built fronts
//         mw::razor::solve_ports(*mw::nec5::razor_part(p).front, ctx, lapack, X, 0);
//
// The Python seam stays the reference; `scripts/core_split/seam_compare.py`
// and `tests/test_nec5_razor_seam_core.py` gate this one against it field for
// field.
#pragma once

#include "_nec5_seam_core.h"
#include "_razor_front_core.h"

#include <memory>
#include <string>

namespace mw {
namespace nec5 {

// razor-2p's part of one preparation.
struct RazorPrepared final : EnginePrepared {
    razor::FrontInputs inputs;  // what the seam passes RazorSolver
    // The C++ front half built on `inputs` (construction, geometry, tables,
    // port columns): what `RazorSolver(...)` and `compute_port_solution`'s
    // front half do with them. Ready for `razor::solve_ports`; null once the
    // run half has released it.
    std::shared_ptr<razor::Front> front;
};

// The razor part of a preparation razor's engine made.
const RazorPrepared &razor_part(const Prepared &p);
RazorPrepared &razor_part(Prepared &p);

class RazorSeamEngine final : public SeamEngine {
  public:
    const SeamBasis &basis() const override;
    void check_can_host(const Deck &d, const Prepared &p, bool has_ground) const override;
    void solver_arguments(const Deck &d, Prepared &p, bool has_ground) const override;
    Built build(Prepared &p) const override;
};

// The whole seam for razor-2p: `seam(latin1_text, RazorSeamEngine(), open_end)`.
SeamResult razor_seam(const std::string &latin1_text, bool open_end = false);

}  // namespace nec5
}  // namespace mw
