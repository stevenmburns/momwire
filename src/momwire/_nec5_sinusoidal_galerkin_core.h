// The NEC-5 seam's structure half and run half for the Sinusoidal-Galerkin
// basis in C++ (momwire#1718 S8), with no Python: the engine-neutral seam
// (`_nec5_seam_core.h`) and run half / writer (`_nec5_printout_core.h`)
// with this basis' answers.
//
//   * `basis()`: "sinusoidal-galerkin", not the cut spelling (`_CUT_SPELLING`
//     is bspline's alone), and a source on a K >= 3 apex or on a crossing
//     node spelled as a NODE GAP (the row serves node gaps and junction
//     ports);
//   * `check_can_host`: `_check_basis_can_host` on SinusoidalGalerkinSolver's
//     row, which refuses none of the four (node gaps, knot feeds, contact over
//     a Sommerfeld ground, buried and crossing wires are all declared);
//   * `solver_arguments`: `_solver_for`'s `construct_solver(...)` call as
//     the constructor's arguments (`nec4::SolverArgs`, the sinusoidal front's
//     input): the pieces, the knot feeds, the radii, the junctions, the
//     loading, the ground, and the dialect's default extended kernel with the
//     two fallbacks this row declares (a radius step at a junction; every
//     wire below the plane);
//   * `build`: the constructor's front half (`sinusoidal::front(args,
//     galerkin)`: the duplicate drop, the geometry, the feed placed at the
//     knot it names with the remainder in `feed_xi`, the basis), and its
//     refusals in Python's words (a ValueError escapes, the extended kernel
//     at a stepped junction is `_prepare`'s refusal);
//   * the run half: `compute_port_solution` (the Galerkin fill and solve,
//     `sinusoidal_galerkin::port_solution`), and the readouts the class
//     inherits from SinusoidalSolver, re-expanded to the caller's wires
//     (`reads_input_wires`, momwire#1042): `currents_at_knots`,
//     `current_slopes`, the walk's sampler `currents_at_knots(coeffs, s)` and
//     `wire_loss_power`.
//
// A node gap (momwire#1718 S9) is `node_gaps=`: a node port on the
// mixed-potential fill, its drive the named member's cut current and its
// readout the exact dual.
//
// DECLINED, never misread: everything the Galerkin fill declines (a wire
// end on the plane before S7, a buried or crossing deck before S11).
#pragma once

#include "_nec5_printout_core.h"
#include "_nec5_seam_core.h"
#include "_razor_solve_core.h"
#include "_sinusoidal_galerkin_fill_core.h"
#include "_sinusoidal_run_core.h"

#include <memory>
#include <string>
#include <vector>

namespace mw {
namespace nec5 {

struct SinusoidalGalerkinPrepared final : EnginePrepared {
    nec4::SolverArgs args;                      // `construct_solver`'s keyword arguments
    std::vector<sinusoidal_galerkin::NodeGap> node_gaps;  // `node_gaps=` (passed when any)
    std::shared_ptr<sinusoidal::Front> front;  // set by `build`; reset on release
};

class SinusoidalGalerkinSeamEngine final : public SeamEngine {
  public:
    const SeamBasis &basis() const override;
    void check_can_host(const Deck &d, const Prepared &p, bool has_ground) const override;
    void solver_arguments(const Deck &d, Prepared &p, bool has_ground) const override;
    Built build(Prepared &p) const override;
};

class SinusoidalGalerkinPrintoutEngine final : public PrintoutEngine {
  public:
    explicit SinusoidalGalerkinPrintoutEngine(const razor::Lapack &lapack) : lapack_(lapack) {}
    const SeamEngine &seam() const override { return seam_; }
    Solved solve(const Prepared &p, const NumpyLinalg &np) override;
    void release(Prepared &p) override;
    std::vector<std::vector<cd>> currents_at_knots(const Prepared &p,
                                                   const std::vector<cd> &coeffs) const override;
    std::vector<std::vector<cd>> current_slopes(
        const Prepared &p, const std::vector<cd> &coeffs, const std::vector<std::vector<cd>> &knots,
        const std::vector<std::vector<double>> &centres_per_piece) const override;
    std::vector<WireWalk> wire_walk(const Prepared &p) const override;
    std::unique_ptr<CurrentSampler> sampler(const Prepared &p, const std::vector<cd> &coeffs,
                                            const std::vector<std::vector<cd>> &knots) const override;
    double wire_loss_power(const Prepared &p, const std::vector<cd> &coeffs, double omega) const override;
    double solver_omega(const Prepared &p) const override;
    double eps0() const override;

  private:
    SinusoidalGalerkinSeamEngine seam_;
    const razor::Lapack &lapack_;
};

SeamResult sinusoidal_galerkin_seam(const std::string &latin1_text, bool open_end = false);

PrintoutResult sinusoidal_galerkin_printout(const std::string &latin1_text, const std::string &basis,
                                            const EngineStamp &stamp, const razor::Lapack &lapack,
                                            const NumpyLinalg &np, std::vector<RunArrays> *arrays = nullptr,
                                            bool open_end = false);

}  // namespace nec5
}  // namespace mw
