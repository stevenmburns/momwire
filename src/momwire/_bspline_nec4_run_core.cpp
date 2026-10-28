// The B-spline basis on the NEC-4.2 slot's run half (see
// `_bspline_nec4_run_core.h`).

#if defined(_MSC_VER) && !defined(__clang__)
#pragma float_control(precise, on)
#pragma fp_contract(off)
#endif

#include "_bspline_nec4_run_core.h"

#include "_bspline_front_core.h"
#include "_nec5_bspline_seam_core.h"
#include "_razor_wire_loading_core.h"

#include <chrono>
#include <cmath>
#include <memory>
#include <string>
#include <vector>

namespace mw {
namespace bspline_nec4 {

namespace {

typedef std::complex<double> cd;

}  // namespace

nec5::BSplineFrontInputs front_inputs(const nec4::SolverArgs &a) {
    if (a.has_insulation) throw nec4::Declined("an insulated wire (bs2's front takes no jacket here)");
    if (!a.exact_kernel_auto) throw nec4::Declined("bs2 without exact_kernel='auto'");
    if (a.ground.kind == nec4::Ground::finite_fast) throw nec4::Declined("a reflection-coefficient ground");
    nec5::BSplineFrontInputs in;
    for (std::size_t w = 0; w < a.wires.size(); w++) {
        for (const nec4::Point3 &p : a.wires[w]) in.points.insert(in.points.end(), p.begin(), p.end());
        in.wire_points.push_back(static_cast<int64_t>(a.wires[w].size()));
        in.n_per_edge.emplace_back(a.n_per_edge[w].begin(), a.n_per_edge[w].end());
    }
    for (const nec4::SolverFeed &f : a.feeds) in.feeds.push_back({f.polyline, f.arclength});
    for (const auto &g : a.junctions) {
        std::vector<nec5::JunctionEnd> m;
        for (const nec4::JunctionEnd &e : g) m.push_back({e.polyline, e.start ? nec5::End::start : nec5::End::end});
        in.junctions.push_back(m);
    }
    if (a.uniform_radius) {
        in.radius.assign(1, a.radii.at(0));
    } else {
        in.radius = a.radii;
        in.radius_per_wire = true;
    }
    in.wavelength = a.wavelength;
    in.extended_kernel = a.extended_kernel;
    if (a.has_conductivity) {
        in.has_conductivity = true;
        in.conductivity = a.conductivity;
    }
    if (a.has_distributed) {
        in.has_rlc = true;
        for (std::size_t w = 0; w < a.wires.size(); w++) {
            nec5::BSplineFrontInputs::RLC e;
            if (a.distributed_present.at(w)) {
                const nec4::DistributedRLC &d = a.distributed.at(w);
                if (d.kind != "series" && d.kind != "parallel") throw nec4::Declined("a distributed load kind");
                e.kind = d.kind == "series" ? 1 : 2;
                e.r = d.r;
                e.l = d.l;
                e.c = d.c;
            }
            in.rlc.push_back(e);
        }
    }
    in.has_ground = a.ground.kind != nec4::Ground::none;
    in.ground_z = 0.0;
    if (a.ground.kind == nec4::Ground::finite) {
        in.ground_eps_given = true;
        in.ground_eps_r = a.ground.eps_r;
        in.ground_sigma = a.ground.sigma;
    }
    return in;
}

Engine::Engine(const nec4::StructureBuild &structure, const Linalg &la, std::vector<SolveArrays> *arrays)
    : structure_(structure),
      la_(la),
      arrays_(arrays),
      preps_(structure.points.size()),
      solves_(structure.points.size()),
      dedups_(structure.points.size()) {}

const nec5::BSplineFront &Engine::front_at(std::size_t index) {
    if (!preps_.at(index)) {
        std::unique_ptr<nec5::Prepared> p(new nec5::Prepared);
        auto part = std::make_shared<nec5::BSplinePrepared>();
        part->inputs = front_inputs(structure_.points[index].args);
        p->wavelength = structure_.points[index].args.wavelength;
        p->engine = part;
        const nec5::BSplineSeamEngine engine;
        const nec5::SeamEngine::Built b = engine.build(*p);
        if (b.kind == nec5::SeamEngine::Built::declined) {
            // bs2's NotImplementedError in Python's own sentence (the crossing
            // serve's scope): `_nec4.solve`'s Nec4Refusal(str(exc))
            const std::string prefix = "front:bspline-";
            if (b.error_kind == "NotImplementedError" && b.message.compare(0, prefix.size(), prefix) == 0)
                throw nec4::Refusal(b.message.substr(prefix.size()));
            throw nec4::Declined(b.message);
        }
        // the constructor's ValueError: `_nec4.solve`'s Nec4Refusal(str(exc))
        if (b.kind != nec5::SeamEngine::Built::built) throw nec4::Refusal(b.message);
        preps_[index] = std::move(p);
    }
    const nec5::BSplinePrepared &part = nec5::bspline_part(*preps_[index]);
    if (!part.front) throw nec4::Declined("bspline: no front");
    return *part.front;
}

const nec4::EngineSolve &Engine::solve(std::size_t index) {
    if (!solves_.at(index)) {
        const nec5::BSplineFront &fr = front_at(index);
        const auto t0 = std::chrono::steady_clock::now();
        std::vector<cd> Z;
        std::vector<double> B;
        nec5::Solved s;
        try {
            s = nec5::bspline_solve_front(fr, la_.lu, la_.sym, la_.np, arrays_ ? &Z : nullptr,
                                          arrays_ ? &B : nullptr);
        } catch (const nec5::PrintoutDecline &e) {
            throw nec4::Declined(e.reason);
        } catch (const bspline::FrontNotPorted &e) {
            throw nec4::Declined(std::string("bspline: ") + e.what());
        } catch (const bspline::FrontValueError &e) {
            throw nec4::Refusal(e.what());  // a ValueError the fill raises
        }
        std::unique_ptr<nec4::EngineSolve> es(new nec4::EngineSolve);
        es->fill_seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        es->n_basis = s.n_basis;
        es->n_ports = s.n_ports;
        es->y = s.y.v;
        es->x = s.x;
        if (s.x_c_order) {  // the KCL Schur result, numpy's C order
            es->x_s0 = s.n_ports;
            es->x_s1 = 1;
        } else {
            es->x_s0 = 1;
            es->x_s1 = s.n_basis;
        }
        if (arrays_) {
            SolveArrays a;
            a.n_basis = s.n_basis;
            a.n_ports = s.n_ports;
            a.Z = Z;
            a.B = B;
            a.X.resize(static_cast<std::size_t>(s.n_basis * s.n_ports));
            for (long long i = 0; i < s.n_basis; i++)
                for (long long j = 0; j < s.n_ports; j++)
                    a.X[static_cast<std::size_t>(i * s.n_ports + j)] =
                        s.x[static_cast<std::size_t>(i * es->x_s0 + j * es->x_s1)];
            a.Y = s.y.v;
            arrays_->push_back(std::move(a));
        }
        solves_[index] = std::move(es);
    }
    return *solves_[index];
}

// The readouts on the SOLVED wires (the run half maps the input wires).
std::vector<std::vector<cd>> Engine::currents_at_knots(std::size_t index, const std::vector<cd> &coeffs) {
    const nec5::BSplineFront &fr = front_at(index);
    return bspline::spline_readout(fr.basis, fr.geom, bspline::WireDedup(), coeffs, nullptr, false);
}

std::vector<std::vector<cd>> Engine::current_slopes(std::size_t index, const std::vector<cd> &coeffs,
                                                    const std::vector<std::vector<double>> &s) {
    const nec5::BSplineFront &fr = front_at(index);
    if (s.size() != fr.args.polylines.size()) throw nec4::Declined("a slope request for another wire count");
    for (const auto &row : s)
        for (double v : row)
            if (std::isnan(v)) throw nec4::Declined("a NaN arclength");
    return bspline::spline_readout(fr.basis, fr.geom, bspline::WireDedup(), coeffs, &s, true);
}

std::vector<cd> Engine::currents_at_arcs(std::size_t index, const std::vector<cd> &coeffs, std::size_t wire,
                                         const std::vector<double> &arcs) {
    const nec5::BSplineFront &fr = front_at(index);
    if (wire >= fr.args.polylines.size()) throw nec4::Declined("a sample on a wire the solver does not hold");
    for (double v : arcs)
        if (std::isnan(v)) throw nec4::Declined("a NaN arclength");
    return bspline::spline_eval_wire(fr.basis, fr.geom, coeffs, wire, arcs, false);
}

// `wire_loss_power(coeffs)[0]`: 0.0 when unloaded (no jacket: an insulated
// wire declines in `front_inputs`)
double Engine::wire_loss_power(std::size_t index, const std::vector<cd> &coeffs) {
    const nec5::BSplineFront &fr = front_at(index);
    if (!fr.loading.active()) return 0.0;
    if (!fr.has_gram) throw nec4::Declined("bspline: the loss read before the solve");
    return bspline::wire_loss_power(fr.gram, coeffs, razor::series_impedance_per_wire(fr.omega, fr.radii, fr.loading),
                                    fr.args.polylines.size());
}

double Engine::solver_omega(std::size_t index) { return front_at(index).omega; }

double Engine::solver_eps0(std::size_t index) {
    (void)index;
    return nec5::BSPLINE_EPS;
}

const nec4::WireDedup *Engine::dedup(std::size_t index) {
    const nec5::BSplineFront &fr = front_at(index);
    if (!fr.dedup.active) return nullptr;
    if (!dedups_.at(index)) {
        std::unique_ptr<nec4::WireDedup> d(new nec4::WireDedup);
        d->active = true;
        d->n_input = fr.dedup.n_input;
        d->kept.assign(fr.dedup.kept.begin(), fr.dedup.kept.end());
        for (const auto &r : fr.dedup.removed) d->removed.push_back({r.copy, r.twin, r.reversed});
        dedups_[index] = std::move(d);
    }
    return dedups_[index].get();
}

nec4::RunEngine &Engines::engine_for(const nec4::StructureBuild &structure) {
    engines_.emplace_back(new Engine(structure, la_, arrays_));
    return *engines_.back();
}

bool printout(const std::string &text, const nec5::EngineStamp &stamp, const Linalg &la, std::string &out,
              std::string &why, std::vector<SolveArrays> *arrays) {
    Engines engines(la, arrays);
    return nec4::render(text, "bspline", stamp, engines, la.np, out, why);
}

bool printout_piece(const std::string &text, const nec5::EngineStamp &stamp, const Linalg &la, std::string &out,
                    std::string &why, bool &go_on) {
    Engines engines(la);
    return nec4::render_piece(text, "bspline", stamp, engines, la.np, out, why, go_on);
}

}  // namespace bspline_nec4
}  // namespace mw
