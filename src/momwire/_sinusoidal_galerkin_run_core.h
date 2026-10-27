// The Sinusoidal-Galerkin basis' answers to the NEC-4.2 run half
// (momwire#1718 S3), with no Python in it: `nec4::RunEngine` over the
// sinusoidal front half and the Galerkin fill (`_sinusoidal_galerkin_fill_
// core.h`).
// - `solve`: `compute_port_solution()` (`sinusoidal_galerkin::port_solution`),
//   Y and the coefficients as numpy holds them (`lu_solve` returns F order);
// - every readout is the sinusoidal basis' own, which SinusoidalGalerkinSolver
//   inherits unchanged (`currents_at_knots`, `current_slopes`, the walk's
//   `currents_at_knots(coeffs, s)`, `wire_loss_power`): `sinusoidal::Engine`'s.
// The constructor's refusals are Python's sentences (`nec4::Refusal`): the
// front's ValueErrors in SinusoidalGalerkinSolver's own name, and the
// extended kernel at a stepped-radius junction (NotImplementedError); so are
// the solve's (a non-finite matrix, a singular one).
//
// `printout(text, ...)` is the slot's whole printout for basis
// "sinusoidal-galerkin" (`_nec4.render(text, basis="sinusoidal-galerkin")`),
// or a decline.
#pragma once

#include "_sinusoidal_galerkin_fill_core.h"
#include "_sinusoidal_run_core.h"

#include <memory>
#include <string>
#include <vector>

namespace mw {
namespace sinusoidal_galerkin {

class Engine : public sinusoidal::Engine {
  public:
    Engine(const nec4::StructureBuild &structure, const sinusoidal::FillLinalg &la);
    const nec4::EngineSolve &solve(std::size_t index) override;
};

class Engines : public nec4::EngineFactory {
  public:
    explicit Engines(const sinusoidal::FillLinalg &la) : la_(la) {}
    nec4::RunEngine &engine_for(const nec4::StructureBuild &structure) override;

  private:
    const sinusoidal::FillLinalg &la_;
    std::vector<std::unique_ptr<Engine>> engines_;
};

// `_nec4.render(text, basis="sinusoidal-galerkin")`: true with the printout
// (UTF-8, LF), or false with the reason this copy declines.
bool printout(const std::string &latin1_text, const nec5::EngineStamp &stamp, const sinusoidal::FillLinalg &la,
              std::string &utf8_printout, std::string &why);
// `_shell._stream_piece(text, basis="sinusoidal-galerkin", dialect="nec4")`
// (`nec4::render_piece`).
bool printout_piece(const std::string &latin1_text, const nec5::EngineStamp &stamp,
                    const sinusoidal::FillLinalg &la, std::string &utf8_printout, std::string &why, bool &go_on);

}  // namespace sinusoidal_galerkin
}  // namespace mw
