// The NEC-5 seam's run half and printout for razor-2p in C++: the
// engine-neutral run half and writer (`_nec5_printout_core.h`) with
// razor-2p's answers to what the solver is asked, so deck text comes out as
// the bytes `momwire-nec5-razor-2p` writes, with no Python anywhere:
//
//   * `seam()`: razor-2p's structure half (`_nec5_razor_seam_core.h`);
//   * `solve`: `RazorSolver.compute_port_solution` -- the fill and the LU
//     (`razor::solve_ports`), then its last line, `Y = cols.T @ X`;
//   * the readouts: `currents_at_knots` (`_end_wings` included),
//     `current_slopes`, the knot walk and the sampler (`element_currents`: subdiv 1 for the far field, 8
//     for the near field) and `wire_loss_power`, over the front half
//     (`razor::Front`), a dropped duplicate wire read as its twin
//     (momwire#1042).
//
//     mw::nec5::PrintoutResult r = mw::nec5::razor_printout(
//         latin1_text, "razor-2p", stamp, ctx, lapack, numpy);
//     // r.route: printed (r.printout: the file's bytes) | declined (r.reason)
//
// The Python seam stays the reference: `scripts/core_split/printout_compare.py`
// and `tests/test_razor_core_standalone.py` gate the whole printout byte for
// byte.
#pragma once

#include "_nec5_printout_core.h"
#include "_nec5_razor_seam_core.h"
#include "_razor_solve_core.h"

#include <string>
#include <vector>

namespace mw {
namespace nec5 {

class RazorPrintoutEngine final : public PrintoutEngine {
  public:
    RazorPrintoutEngine(razor::SolveContext &ctx, const razor::Lapack &lapack)
        : ctx_(ctx), lapack_(lapack) {}
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
    RazorSeamEngine seam_;
    razor::SolveContext &ctx_;
    const razor::Lapack &lapack_;
};

// The whole of `momwire-nec5-razor-2p` for one deck file's bytes:
// `printout(latin1_text, basis, stamp, RazorPrintoutEngine(ctx, lapack), np,
// arrays, open_end)`.
PrintoutResult razor_printout(const std::string &latin1_text, const std::string &basis,
                              const EngineStamp &stamp, razor::SolveContext &ctx,
                              const razor::Lapack &lapack, const NumpyLinalg &np,
                              std::vector<RunArrays> *arrays = nullptr, bool open_end = false);

}  // namespace nec5
}  // namespace mw
