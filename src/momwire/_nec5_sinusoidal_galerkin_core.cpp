// The Sinusoidal-Galerkin basis on the NEC-5 seam (see
// `_nec5_sinusoidal_galerkin_core.h`).
#include "_nec5_sinusoidal_galerkin_core.h"

#if defined(_MSC_VER) && !defined(__clang__)
#pragma float_control(precise, on)
#pragma fp_contract(off)
#endif

#include <algorithm>
#include <cmath>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace mw {
namespace nec5 {

namespace {

const SinusoidalGalerkinPrepared &sg_part(const Prepared &p) {
    const auto *b = dynamic_cast<const SinusoidalGalerkinPrepared *>(p.engine.get());
    if (b == nullptr) throw std::logic_error("a preparation the Sinusoidal-Galerkin engine did not make");
    return *b;
}

SinusoidalGalerkinPrepared &sg_part(Prepared &p) {
    auto *b = dynamic_cast<SinusoidalGalerkinPrepared *>(p.engine.get());
    if (b == nullptr) throw std::logic_error("a preparation the Sinusoidal-Galerkin engine did not make");
    return *b;
}

const sinusoidal::Front &front_of(const Prepared &p) {
    const SinusoidalGalerkinPrepared &part = sg_part(p);
    if (!part.front) throw PrintoutDecline{"sinusoidal-galerkin: no front (released or never built)"};
    return *part.front;
}

// `reads_input_wires("per_wire")`: a readout by SOLVED wire in the caller's
// numbering, zeros (sized like the twin's, or like the caller's own request
// `sizes[r]`) at each dropped copy.
std::vector<std::vector<cd>> to_input(const nec4::WireDedup &dd, const std::vector<std::vector<cd>> &solved,
                                      const std::vector<std::size_t> *sizes) {
    if (!dd.active) return solved;
    std::vector<std::vector<cd>> out(static_cast<std::size_t>(dd.n_input));
    for (std::size_t i = 0; i < dd.kept.size(); i++) out[static_cast<std::size_t>(dd.kept[i])] = solved.at(i);
    for (const auto &d : dd.removed) {
        std::size_t n = 0;
        if (sizes != nullptr) {
            n = sizes->at(static_cast<std::size_t>(d.copy));
        } else {
            std::size_t twin_solved = 0;
            for (std::size_t i = 0; i < dd.kept.size(); i++)
                if (dd.kept[i] == d.twin) twin_solved = i;
            n = solved.at(twin_solved).size();
        }
        out[static_cast<std::size_t>(d.copy)].assign(n, cd(0.0, 0.0));
    }
    return out;
}

// The arc sampler: `currents_at_knots(coeffs, s)` on one solved wire.
class ArcSampler final : public CurrentSampler {
  public:
    ArcSampler(const sinusoidal::Front &fr, const std::vector<cd> &coeffs) : fr_(fr), coeffs_(coeffs) {}
    std::vector<cd> at(std::size_t w, const WireWalk &ww, const std::vector<double> &arcs) const override {
        (void)w;
        return sinusoidal::arc_currents(fr_, coeffs_, static_cast<std::size_t>(ww.solved), arcs);
    }

  private:
    const sinusoidal::Front &fr_;
    const std::vector<cd> coeffs_;
};

}  // namespace

const SeamBasis &SinusoidalGalerkinSeamEngine::basis() const {
    static const SeamBasis row = [] {
        SeamBasis b;
        b.name = "sinusoidal-galerkin";
        b.cut_spelling = false;           // `_CUT_SPELLING = (BSplineSolver,)`
        b.crossing_node_spelling = true;  // node gaps and junction ports
        return b;
    }();
    return row;
}

// SinusoidalGalerkinSolver's row refuses none of `_check_basis_can_host`'s
// four (measured on the row: every cell it reads answers None).
void SinusoidalGalerkinSeamEngine::check_can_host(const Deck &d, const Prepared &p, bool has_ground) const {
    (void)d;
    (void)p;
    (void)has_ground;
}

void SinusoidalGalerkinSeamEngine::solver_arguments(const Deck &d, Prepared &p, bool has_ground) const {
    const SeamMesh &mesh = p.mesh;
    if (mesh.pieces.empty()) throw SeamRaised{"ValueError", "wires must be non-empty"};
    auto part = std::make_shared<SinusoidalGalerkinPrepared>();
    nec4::SolverArgs &a = part->args;
    for (const SeamPiece &piece : mesh.pieces) {
        std::vector<nec4::Point3> pl;
        for (long long i = 0; i < piece.n_points(); i++)
            pl.push_back({piece.points[3 * static_cast<std::size_t>(i)], piece.points[3 * static_cast<std::size_t>(i) + 1],
                          piece.points[3 * static_cast<std::size_t>(i) + 2]});
        a.wires.push_back(pl);
        a.n_per_edge.push_back(piece.edge_elements());
        a.radii.push_back(piece.radius);
    }
    a.uniform_radius = std::all_of(a.radii.begin(), a.radii.end(), [&](double r) { return r == a.radii[0]; });
    for (long long s : mesh.feeds) {
        nec4::SolverFeed f;
        f.polyline = mesh.sites[static_cast<std::size_t>(s)].piece;
        f.arclength = mesh.sites[static_cast<std::size_t>(s)].arclength;
        f.volts = cd(0.0, 0.0);
        a.feeds.push_back(f);
    }
    for (const auto &g : mesh.junctions) {
        std::vector<nec4::JunctionEnd> members;
        for (const JunctionEnd &e : g) {
            nec4::JunctionEnd m;
            m.polyline = e.first;
            m.start = e.second == End::start;
            members.push_back(m);
        }
        a.junctions.push_back(members);
    }
    a.wavelength = p.wavelength;
    // `_wire_loading`: LD 5 per piece (NaN: not this one) and LD 2 / LD 3
    const std::size_t n = mesh.pieces.size();
    a.conductivity.assign(n, std::numeric_limits<double>::quiet_NaN());
    a.insulation_radius.assign(n, std::numeric_limits<double>::quiet_NaN());
    a.insulation_eps_r.assign(n, std::numeric_limits<double>::quiet_NaN());
    a.distributed_present.assign(n, false);
    a.distributed.assign(n, nec4::DistributedRLC());
    for (std::size_t i = 0; i < n; i++) {
        for (const auto &tc : d.wire_conductivity)
            if (tc.first == mesh.pieces[i].tag) {
                a.conductivity[i] = tc.second;
                break;
            }
        for (const auto &tr : d.wire_distributed_rlc)
            if (tr.first == mesh.pieces[i].tag) {
                a.distributed_present[i] = true;
                a.distributed[i].kind = tr.second.kind;
                a.distributed[i].r = tr.second.r;
                a.distributed[i].l = tr.second.l;
                a.distributed[i].c = tr.second.c;
                break;
            }
    }
    a.has_conductivity = std::any_of(a.conductivity.begin(), a.conductivity.end(), [](double x) { return std::isfinite(x); });
    a.has_insulation = false;
    a.has_distributed = std::any_of(a.distributed_present.begin(), a.distributed_present.end(), [](bool b) { return b; });
    // `_ground_kwargs`: a perfect image (GN 1, GD) or the Sommerfeld medium
    if (has_ground) {
        if (d.ground.kind == Ground::sommerfeld) {
            a.ground.kind = nec4::Ground::finite;
            a.ground.eps_r = p.medium_eps_r;
            a.ground.sigma = p.medium_sigma;
        } else {
            a.ground.kind = nec4::Ground::pec;
        }
    }
    // the dialect's default extended kernel (`construct_solver`): off at a
    // radius step at a junction (this row's declared refusal), and when EVERY
    // wire is below the plane (the buried partial's full fallback)
    bool ek = true;
    if (std::any_of(a.radii.begin(), a.radii.end(), [&](double r) { return r != a.radii[0]; }))
        for (const auto &members : a.junctions)
            for (const nec4::JunctionEnd &m : members)
                if (a.radii[static_cast<std::size_t>(m.polyline)] != a.radii[static_cast<std::size_t>(members[0].polyline)])
                    ek = false;
    if (ek && has_ground) {
        long long below = 0;
        for (const SeamPiece &piece : mesh.pieces)
            if (piece.zmin() < 0.0 - piece.touch_tol()) below++;
        if (below > 0 && below == static_cast<long long>(n)) ek = false;
    }
    a.extended_kernel = ek;
    p.engine = std::move(part);
}

SeamEngine::Built SinusoidalGalerkinSeamEngine::build(Prepared &p) const {
    SinusoidalGalerkinPrepared &part = sg_part(p);
    Built b;
    const auto decline = [&](const std::string &why) {
        part.front.reset();
        b.kind = Built::declined;
        b.message = "front:" + why;
        b.unsolved = true;
        return b;
    };
    if (!p.mesh.gaps.empty()) return decline("sinusoidal-galerkin-node-gaps (momwire#1718 S9)");
    try {
        auto fr = std::make_shared<sinusoidal::Front>(sinusoidal::front(part.args, true));
        sinusoidal_galerkin::check_constructor(*fr, fr->args);
        part.front = std::move(fr);
    } catch (const sinusoidal::FrontError &e) {
        b.kind = Built::raised;  // the constructor's ValueError escapes the seam
        b.error_kind = "ValueError";
        b.message = e.message;
        return b;
    } catch (const sinusoidal_galerkin::SolverRefusal &e) {
        if (e.kind == "NotImplementedError") {
            b.kind = Built::refused;  // `_prepare`'s refusal
        } else {
            b.kind = Built::raised;
        }
        b.error_kind = e.kind;
        b.message = e.message;
        return b;
    } catch (const nec4::Declined &e) {
        return decline(std::string("sinusoidal-galerkin: ") + e.what());
    }
    return b;
}

Solved SinusoidalGalerkinPrintoutEngine::solve(const Prepared &p, const NumpyLinalg &np) {
    const sinusoidal::Front &fr = front_of(p);
    if (np.host.zgemm != nullptr) throw PrintoutDecline{"sinusoidal-galerkin: the host BLAS route is not ported"};
    sinusoidal::FillLinalg la;
    la.np = np;
    la.lu = lapack_;
    sinusoidal_galerkin::PortSolution ps;
    try {
        ps = sinusoidal_galerkin::port_solution(fr, fr.args, la);
    } catch (const sinusoidal_galerkin::SolverRefusal &e) {
        throw PrintoutRaise{e.kind, e.message};
    } catch (const nec4::Declined &e) {
        throw PrintoutDecline{std::string("sinusoidal-galerkin: ") + e.what()};
    }
    Solved s;
    s.n_basis = ps.n;
    s.n_ports = ps.n_ports;
    s.x_c_order = false;  // `lu_solve`'s F order
    s.x = ps.coeffs;
    s.y = CMat(s.n_ports, s.n_ports);
    s.y.v = ps.y;
    return s;
}

void SinusoidalGalerkinPrintoutEngine::release(Prepared &p) { sg_part(p).front.reset(); }

std::vector<std::vector<cd>> SinusoidalGalerkinPrintoutEngine::currents_at_knots(const Prepared &p,
                                                                                 const std::vector<cd> &coeffs) const {
    const sinusoidal::Front &fr = front_of(p);
    return to_input(fr.dedup, sinusoidal::knot_currents(fr, coeffs), nullptr);
}

std::vector<std::vector<cd>> SinusoidalGalerkinPrintoutEngine::current_slopes(
    const Prepared &p, const std::vector<cd> &coeffs, const std::vector<std::vector<cd>> &knots,
    const std::vector<std::vector<double>> &centres_per_piece) const {
    (void)knots;
    const sinusoidal::Front &fr = front_of(p);
    const nec4::WireDedup &dd = fr.dedup;
    if (!dd.active) return sinusoidal::slopes(fr, coeffs, centres_per_piece);
    // `dedup.to_solved(s_array)`, then back in the caller's numbering
    std::vector<std::vector<double>> s_solved;
    for (long long w : dd.kept) s_solved.push_back(centres_per_piece.at(static_cast<std::size_t>(w)));
    std::vector<std::size_t> sizes;
    for (const auto &v : centres_per_piece) sizes.push_back(v.size());
    return to_input(dd, sinusoidal::slopes(fr, coeffs, s_solved), &sizes);
}

// `element_currents`' walk: the caller's wires, a dropped copy over its
// twin's polyline and per-edge counts (reversed when it runs the other way).
std::vector<WireWalk> SinusoidalGalerkinPrintoutEngine::wire_walk(const Prepared &p) const {
    const sinusoidal::Front &fr = front_of(p);
    const nec4::WireDedup &dd = fr.dedup;
    const auto of_solved = [&](long long s, WireWalk &ww) {
        const auto &pl = fr.args.wires[static_cast<std::size_t>(s)];
        ww.polyline.clear();
        for (const nec4::Point3 &q : pl) ww.polyline.insert(ww.polyline.end(), q.begin(), q.end());
        ww.npe.assign(fr.args.n_per_edge[static_cast<std::size_t>(s)].begin(),
                      fr.args.n_per_edge[static_cast<std::size_t>(s)].end());
    };
    std::vector<WireWalk> out;
    if (!dd.active) {
        for (std::size_t w = 0; w < fr.args.wires.size(); w++) {
            WireWalk ww;
            ww.solved = static_cast<int64_t>(w);
            of_solved(ww.solved, ww);
            out.push_back(std::move(ww));
        }
        return out;
    }
    const auto solved_of = [&](long long input) -> long long {
        for (std::size_t i = 0; i < dd.kept.size(); i++)
            if (dd.kept[i] == input) return static_cast<long long>(i);
        return -1;
    };
    for (long long w = 0; w < dd.n_input; w++) {
        WireWalk ww;
        ww.solved = solved_of(w);
        if (ww.solved >= 0) {
            of_solved(ww.solved, ww);
        } else {
            for (const auto &d : dd.removed)
                if (d.copy == w) {
                    of_solved(solved_of(d.twin), ww);
                    if (d.reversed) {
                        const std::size_t m = ww.polyline.size() / 3;
                        std::vector<double> r(ww.polyline.size());
                        for (std::size_t i = 0; i < m; i++)
                            for (int c = 0; c < 3; c++) r[3 * i + c] = ww.polyline[3 * (m - 1 - i) + c];
                        ww.polyline = r;
                        std::reverse(ww.npe.begin(), ww.npe.end());
                    }
                }
        }
        out.push_back(std::move(ww));
    }
    return out;
}

std::unique_ptr<CurrentSampler> SinusoidalGalerkinPrintoutEngine::sampler(
    const Prepared &p, const std::vector<cd> &coeffs, const std::vector<std::vector<cd>> &knots) const {
    (void)knots;
    return std::unique_ptr<CurrentSampler>(new ArcSampler(front_of(p), coeffs));
}

// `wire_loss_power(coeffs, omega)[0]`, at the run's omega
double SinusoidalGalerkinPrintoutEngine::wire_loss_power(const Prepared &p, const std::vector<cd> &coeffs,
                                                         double omega) const {
    return sinusoidal::loss_power_at(front_of(p), coeffs, omega);
}

double SinusoidalGalerkinPrintoutEngine::solver_omega(const Prepared &p) const { return front_of(p).omega; }

double SinusoidalGalerkinPrintoutEngine::eps0() const { return 8.8541878188e-12; }

SeamResult sinusoidal_galerkin_seam(const std::string &latin1_text, bool open_end) {
    SinusoidalGalerkinSeamEngine engine;
    return seam(latin1_text, engine, open_end);
}

PrintoutResult sinusoidal_galerkin_printout(const std::string &latin1_text, const std::string &basis,
                                            const EngineStamp &stamp, const razor::Lapack &lapack,
                                            const NumpyLinalg &np, std::vector<RunArrays> *arrays, bool open_end) {
    SinusoidalGalerkinPrintoutEngine engine(lapack);
    return printout(latin1_text, basis, stamp, engine, np, arrays, open_end);
}

}  // namespace nec5
}  // namespace mw
