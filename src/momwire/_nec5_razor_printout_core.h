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
    bool presolve(const std::vector<Prepared> &prepared, const std::vector<long long> &indices,
                  const NumpyLinalg &np, std::vector<Solved> &out, std::vector<char> &ok) override;

  private:
    RazorSeamEngine seam_;
    razor::SolveContext &ctx_;
    const razor::Lapack &lapack_;
};

// A buried stepping FR's frequencies solved concurrently (`presolve`): up to
// `n` at once, each fill and factor on its own thread (no OpenMP team inside
// it, the caller's BLAS at one thread). 1 (the default) solves them in turn.
// The bytes do not depend on it: the below table's choices are made first,
// in the serial run's order (`razor::prime_below_table`), every cache a
// fill reads holds values that do not depend on who filled them, and the
// printout is written in frequency order afterwards. Not taken (in turn, as
// at 1) with `crossing_table` asked for: its take decision is one
// process-wide flag per fill (`ni_eval`'s `g_crossing_taken`), which two
// concurrent fills would share.
void set_frequency_workers(int n);
int frequency_workers();
// Whether `presolve` takes a sweep with the features this process asked for
// (false with `crossing_table`, above): the executable's policy asks before
// it gives the sweep the serial copy, so a declined sweep keeps the fill's
// own threads.
bool frequency_sweep_allowed();
// The frequencies `presolve` has solved concurrently, a process total (the
// executable's MW_EXE_TIMES gate line: proof the sweep ran).
long long frequencies_presolved();

// The whole of `momwire-nec5-razor-2p` for one deck file's bytes:
// `printout(latin1_text, basis, stamp, RazorPrintoutEngine(ctx, lapack), np,
// arrays, open_end)`.
PrintoutResult razor_printout(const std::string &latin1_text, const std::string &basis,
                              const EngineStamp &stamp, razor::SolveContext &ctx,
                              const razor::Lapack &lapack, const NumpyLinalg &np,
                              std::vector<RunArrays> *arrays = nullptr, bool open_end = false);

}  // namespace nec5
}  // namespace mw
