// The sinusoidal basis' answers to the NEC-4.2 run half (momwire#1607 B4a),
// with no Python in it: `nec4::RunEngine` over B2's front half and B3's fill.
// - `solve`: `compute_port_solution()` (`port_solution`), Y and the
//   coefficients as numpy holds them (scipy's solve returns C order);
// - `currents_at_knots(alpha)`: the knot path (each segment's basis read at
//   its two ends, half weight at an interior knot, `np.add.at` order);
// - `current_slopes(alpha, s)`: `_slope_eval_points` and
//   `_evaluate_basis_slope_at_points`, in free space.
// Every numpy product is written in numpy's operand order: each one here has
// a temporary on its LEFT, so numpy's temporary elision keeps that order.
//
// `printout(text, ...)` is the slot's whole printout for basis
// "sinusoidal" (`_nec4.render(text, basis="sinusoidal")`), or a decline.
#pragma once

#include "_nec4_run_core.h"
#include "_sinusoidal_fill_core.h"

#include <memory>
#include <string>
#include <vector>

namespace mw {
namespace sinusoidal {

class Engine : public nec4::RunEngine {
  public:
    Engine(const nec4::StructureBuild &structure, const FillLinalg &la);
    const nec4::EngineSolve &solve(std::size_t index) override;
    std::vector<std::vector<std::complex<double>>> currents_at_knots(
        std::size_t index, const std::vector<std::complex<double>> &coeffs) override;
    std::vector<std::vector<std::complex<double>>> current_slopes(
        std::size_t index, const std::vector<std::complex<double>> &coeffs,
        const std::vector<std::vector<double>> &s) override;

  private:
    const Front &front_at(std::size_t index);
    const nec4::StructureBuild &structure_;
    const FillLinalg &la_;
    std::vector<std::unique_ptr<Front>> fronts_;
    std::vector<std::unique_ptr<nec4::EngineSolve>> solves_;
};

class Engines : public nec4::EngineFactory {
  public:
    explicit Engines(const FillLinalg &la) : la_(la) {}
    nec4::RunEngine &engine_for(const nec4::StructureBuild &structure) override;

  private:
    const FillLinalg &la_;
    std::vector<std::unique_ptr<Engine>> engines_;
};

// `_nec4.render(text, basis="sinusoidal")`: true with the printout (UTF-8,
// LF), or false with the reason this copy declines.
bool printout(const std::string &latin1_text, const nec5::EngineStamp &stamp, const FillLinalg &la,
              std::string &utf8_printout, std::string &why);
// `_shell._stream_piece(text, basis="sinusoidal", dialect="nec4")`
// (`nec4::render_piece`).
bool printout_piece(const std::string &latin1_text, const nec5::EngineStamp &stamp, const FillLinalg &la,
                    std::string &utf8_printout, std::string &why, bool &go_on);

}  // namespace sinusoidal
}  // namespace mw
