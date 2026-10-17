// PROTOTYPE (momwire#1517 step 1, not merged): the NEC-5 seam's run half and
// printout for bspline (bs2) in C++ -- the engine-neutral run half and writer
// (`_nec5_printout_core.h`) with the B-spline basis' answers to what the
// solver is asked:
//
//   * `seam()`: bs2's structure half (`_nec5_bspline_seam_core.h`), whose
//     `build` makes the constructor's front half;
//   * `solve`: `BSplineSolver.compute_port_solution` on the dense free-space
//     route -- `_build_J_blocks`, `_assemble_Z`, `_solve_with_kcl_ports`
//     and `Y = B.T @ X` (`_bspline_fill_core.h`);
//   * the readouts: `currents_at_knots` / `current_slopes` (the spline
//     expansion at the knots or at given arcs), the knot walk over the
//     caller's wires and the spline SAMPLER at the walk's subdivided arcs,
//     and `wire_loss_power` (0: a loaded deck declines in the seam).
//
//     mw::nec5::PrintoutResult r = mw::nec5::bspline_printout(
//         latin1_text, "bspline", stamp, lapack, sym_lapack, numpy);
//
// `numpy.dsyrk` must be set (the assembler's `tangents @ tangents.T`).
#pragma once

#include "_bspline_fill_core.h"
#include "_nec5_bspline_seam_core.h"
#include "_nec5_printout_core.h"
#include "_razor_solve_core.h"

#include <memory>
#include <string>
#include <vector>

namespace mw {
namespace nec5 {

class BSplinePrintoutEngine final : public PrintoutEngine {
  public:
    BSplinePrintoutEngine(const razor::Lapack &lapack, const bspline::SymLapack &sym)
        : lapack_(lapack), sym_(sym) {}
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
    double wire_loss_power(const Prepared &p, const std::vector<cd> &coeffs,
                           double omega) const override;
    double solver_omega(const Prepared &p) const override;
    double eps0() const override;

  private:
    BSplineSeamEngine seam_;
    const razor::Lapack &lapack_;
    const bspline::SymLapack &sym_;
};

PrintoutResult bspline_printout(const std::string &latin1_text, const std::string &basis,
                                const EngineStamp &stamp, const razor::Lapack &lapack,
                                const bspline::SymLapack &sym, const NumpyLinalg &np,
                                std::vector<RunArrays> *arrays = nullptr,
                                bool open_end = false);

}  // namespace nec5
}  // namespace mw
