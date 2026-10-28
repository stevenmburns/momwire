// The B-spline basis (bs2) on the NEC-4.2 slot's run half (momwire#1607
// Phase C1), with no Python in it: `nec4::RunEngine` over bs2's own cores.
// The slot's build (`nec4::build(text, "bspline")`) hands each operating
// point the keyword arguments `construct_solver(BSplineSolver, {"exact_kernel":
// "auto"}, ...)` takes on this dialect: centre-addressed delta gaps, the
// junctions as declared, the extended kernel per the dialect default (in
// part over a buried wire), the loading, the ground. Each point's arguments
// become the NEC-5 seam's `BSplineFrontInputs`, and from there the same
// constructor front half (`BSplineSeamEngine::build`: the duplicate drop,
// the geometry, the KCL rows and knots, the port columns, the exact kernel
// resolved), the same solve (`nec5::bspline_solve_front`: the dense, chunked
// or buried fill with the crossing and transmitted blocks, the corrections,
// the image, the loading, the KCL port solve) and the same readouts
// (`bspline::spline_readout` / `spline_eval_wire`, on the SOLVED wires: the
// run half maps the input wires itself).
//
// Refusals keep `_nec4.solve`'s rule: a ValueError the constructor or the
// fill raises is the slot's refusal in its words; anything this copy does
// not serve (bs2's NotImplementedError among them, whose wording it does not
// promise) is declined by name, never misread.
//
// `printout(text, ...)` is `_nec4.render(text, basis="bspline")`, or a
// decline.
#pragma once

#include "_bspline_fill_core.h"
#include "_nec4_run_core.h"
#include "_nec5_bspline_printout_core.h"
#include "_razor_solve_core.h"

#include <complex>
#include <memory>
#include <string>
#include <vector>

namespace mw {
namespace bspline_nec4 {

// The two OpenBLAS builds Python's bs2 solve reaches: numpy's products
// (`np`) and SciPy's LU and symmetric factor (`lu`, `sym`).
struct Linalg {
    razor::Lapack lu;
    bspline::SymLapack sym;
    nec5::NumpyLinalg np;
};

// One solve's arrays, for the gate (`scripts/core_split/nec4_run_compare.py
// --basis bspline`): the operator the port solve takes, the port columns,
// the coefficients and the port admittances, as numpy holds them.
struct SolveArrays {
    long long n_basis = 0, n_ports = 0;
    std::vector<std::complex<double>> Z;  // (n_basis, n_basis), C order
    std::vector<double> B;                // (n_basis, n_ports), C order
    std::vector<std::complex<double>> X;  // (n_basis, n_ports), C order
    std::vector<std::complex<double>> Y;  // (n_ports, n_ports), C order
};

class Engine : public nec4::RunEngine {
  public:
    Engine(const nec4::StructureBuild &structure, const Linalg &la, std::vector<SolveArrays> *arrays);
    const nec4::EngineSolve &solve(std::size_t index) override;
    std::vector<std::vector<std::complex<double>>> currents_at_knots(
        std::size_t index, const std::vector<std::complex<double>> &coeffs) override;
    std::vector<std::vector<std::complex<double>>> current_slopes(
        std::size_t index, const std::vector<std::complex<double>> &coeffs,
        const std::vector<std::vector<double>> &s) override;
    double wire_loss_power(std::size_t index, const std::vector<std::complex<double>> &coeffs) override;
    std::vector<std::complex<double>> currents_at_arcs(std::size_t index,
                                                       const std::vector<std::complex<double>> &coeffs,
                                                       std::size_t wire, const std::vector<double> &arcs) override;
    double solver_omega(std::size_t index) override;
    double solver_eps0(std::size_t index) override;
    const nec4::WireDedup *dedup(std::size_t index) override;

  private:
    const nec5::BSplineFront &front_at(std::size_t index);
    const nec4::StructureBuild &structure_;
    const Linalg &la_;
    std::vector<SolveArrays> *arrays_;
    std::vector<std::unique_ptr<nec5::Prepared>> preps_;
    std::vector<std::unique_ptr<nec4::EngineSolve>> solves_;
    std::vector<std::unique_ptr<nec4::WireDedup>> dedups_;
};

class Engines : public nec4::EngineFactory {
  public:
    Engines(const Linalg &la, std::vector<SolveArrays> *arrays = nullptr) : la_(la), arrays_(arrays) {}
    nec4::RunEngine &engine_for(const nec4::StructureBuild &structure) override;

  private:
    const Linalg &la_;
    std::vector<SolveArrays> *arrays_;
    std::vector<std::unique_ptr<Engine>> engines_;
};

// `BSplineSolver(...)`'s keyword arguments, from the slot's build, as the
// NEC-5 seam spells them (`BSplineFrontInputs`). Throws nec4::Declined for an
// argument bs2's front does not take here.
nec5::BSplineFrontInputs front_inputs(const nec4::SolverArgs &args);

// `_nec4.render(text, basis="bspline")`: true with the printout (UTF-8, LF),
// or false with the reason this copy declines. `arrays`, when given,
// receives each solve's arrays in solve order.
bool printout(const std::string &latin1_text, const nec5::EngineStamp &stamp, const Linalg &la,
              std::string &utf8_printout, std::string &why, std::vector<SolveArrays> *arrays = nullptr);
// `_shell._stream_piece(text, basis="bspline", dialect="nec4")`
// (`nec4::render_piece`).
bool printout_piece(const std::string &latin1_text, const nec5::EngineStamp &stamp, const Linalg &la,
                    std::string &utf8_printout, std::string &why, bool &go_on);

}  // namespace bspline_nec4
}  // namespace mw
