// The Sinusoidal-Galerkin basis' run-half answers (see
// `_sinusoidal_galerkin_run_core.h`).

#if defined(_MSC_VER) && !defined(__clang__)
#pragma float_control(precise, on)
#pragma fp_contract(off)
#endif

#include "_sinusoidal_galerkin_run_core.h"

#include <chrono>
#include <string>
#include <vector>

namespace mw {
namespace sinusoidal_galerkin {

Engine::Engine(const nec4::StructureBuild &structure, const sinusoidal::FillLinalg &la)
    : sinusoidal::Engine(structure, la) {
    galerkin_ = true;
}

const nec4::EngineSolve &Engine::solve(std::size_t index) {
    if (!solves_.at(index)) {
        const sinusoidal::Front &fr = front_at(index);
        const auto t0 = std::chrono::steady_clock::now();
        PortSolution ps;
        try {
            ps = sinusoidal_galerkin::port_solution(fr, fr.args, la_);  // the solved wires (momwire#1042)
        } catch (const SolverRefusal &e) {
            throw nec4::Refusal(e.message);  // `_nec4.solve`'s Nec4Refusal(str(exc))
        }
        std::unique_ptr<nec4::EngineSolve> es(new nec4::EngineSolve);
        es->fill_seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        es->n_basis = ps.n;
        es->n_ports = ps.n_ports;
        es->y = ps.y;
        es->x = ps.coeffs;  // `lu_solve` returns F order
        es->x_s0 = 1;
        es->x_s1 = ps.n;
        solves_[index] = std::move(es);
    }
    return *solves_[index];
}

nec4::RunEngine &Engines::engine_for(const nec4::StructureBuild &structure) {
    engines_.emplace_back(new Engine(structure, la_));
    return *engines_.back();
}

bool printout(const std::string &text, const nec5::EngineStamp &stamp, const sinusoidal::FillLinalg &la,
              std::string &out, std::string &why) {
    Engines engines(la);
    return nec4::render(text, "sinusoidal-galerkin", stamp, engines, la.np, out, why);
}

bool printout_piece(const std::string &text, const nec5::EngineStamp &stamp, const sinusoidal::FillLinalg &la,
                    std::string &out, std::string &why, bool &go_on) {
    Engines engines(la);
    return nec4::render_piece(text, "sinusoidal-galerkin", stamp, engines, la.np, out, why, go_on);
}

}  // namespace sinusoidal_galerkin
}  // namespace mw
