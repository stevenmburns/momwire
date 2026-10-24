// The NEC-5 seam's structure half for razor-2p: razor's answers to what a
// basis decides -- see `_nec5_razor_seam_core.h`. The ladder, the mesh and
// the ports are the engine-neutral seam's (`_nec5_seam_core.cpp`); each hook
// here names the Python it is a copy of (`eznec/_serve.py` unless said
// otherwise), and its sentences are that Python's, byte for byte.
#include "_nec5_razor_seam_core.h"

// MSVC builds every TU /fp:fast (setup.py); this one replays numpy's and
// Python's elementwise arithmetic one IEEE operation at a time (the
// tolerances), so it asks for precise semantics.
#if defined(_MSC_VER)
#pragma float_control(precise, on)
#pragma fp_contract(off)
#endif

#include <algorithm>
#include <cmath>
#include <limits>
#include <memory>
#include <set>
#include <stdexcept>
#include <utility>
#include <vector>

namespace mw {
namespace nec5 {

const RazorPrepared &razor_part(const Prepared &p) {
    const RazorPrepared *r = dynamic_cast<const RazorPrepared *>(p.engine.get());
    if (r == nullptr) throw std::logic_error("a preparation razor-2p's engine did not make");
    return *r;
}

RazorPrepared &razor_part(Prepared &p) {
    RazorPrepared *r = dynamic_cast<RazorPrepared *>(p.engine.get());
    if (r == nullptr) throw std::logic_error("a preparation razor-2p's engine did not make");
    return *r;
}

const SeamBasis &RazorSeamEngine::basis() const {
    static const SeamBasis row = [] {
        SeamBasis b;
        b.name = "razor-2p";
        b.cut_spelling = false;            // not in `_CUT_SPELLING`
        b.crossing_node_spelling = false;  // node gaps, but no junction ports
        return b;
    }();
    return row;
}

// `_check_basis_can_host` on razor's row: node gaps and knot feeds served,
// contact and buried wires over a medium served; what is left is the inert
// one-segment wire (`_inert_pieces`, razor being `_NATIVE_LOADING`).
void RazorSeamEngine::check_can_host(const Deck &d, const Prepared &p, bool has_ground) const {
    (void)d;
    const SeamMesh &mesh = p.mesh;
    std::set<JunctionEnd> joined;
    for (const auto &g : mesh.junctions)
        for (const JunctionEnd &e : g) joined.insert(e);
    std::vector<long long> inert;
    for (size_t i = 0; i < mesh.pieces.size(); i++) {
        const SeamPiece &piece = mesh.pieces[i];
        if (piece.n_elements() >= 2) continue;
        if (joined.count(JunctionEnd(i, End::start)) || joined.count(JunctionEnd(i, End::end)))
            continue;
        if (has_ground) {
            const double tol = piece.touch_tol();
            if (std::fabs(piece.start()[2] - 0.0) <= tol || std::fabs(piece.end()[2] - 0.0) <= tol)
                continue;
        }
        inert.push_back(static_cast<long long>(i));
    }
    if (!inert.empty())
        throw SeamRefusal{"wire " + mesh.pieces[inert[0]].tag.str() +
                          " is one segment long and meets no other wire at either end, so "
                          "basis " + basis().name + " has no basis function to put on it and it would "
                          "scatter nothing (momwire#608); this deck declares " +
                          std::to_string(inert.size()) + " such GW card" +
                          (inert.size() > 1 ? "s" : "")};
}

// `_solver_for` -> `construct_solver(RazorSolver, ...)`.
void RazorSeamEngine::solver_arguments(const Deck &d, Prepared &p, bool has_ground) const {
    const SeamMesh &mesh = p.mesh;
    // The extended kernel by default: razor declares no refusal for it, and
    // a stepped radius at a junction is a condition it serves. A wire below
    // the plane is served IN PART (momwire#1614, `construct_solver`'s
    // `extended_kernel_partial`): the above-ground medium takes it and the
    // buried wires keep the reduced kernel, with an advisory the printout
    // does not carry -- unless EVERY piece is below (`_wires_below` on the
    // pieces), where it would act on no pair and the default is reduced.
    size_t n_below = 0;
    if (has_ground)
        for (const SeamPiece &piece : mesh.pieces)
            if (piece.zmin() < 0.0 - piece.touch_tol())
                n_below++;
    const bool all_below = !mesh.pieces.empty() && n_below == mesh.pieces.size();
    auto part = std::make_shared<RazorPrepared>();
    razor::FrontInputs &in = part->inputs;
    for (const SeamPiece &piece : mesh.pieces) {
        in.points.insert(in.points.end(), piece.points.begin(), piece.points.end());
        in.wire_points.push_back(piece.n_points());
        razor::EdgeCounts ec;
        ec.kind = razor::EdgeCounts::list;
        for (long long n : piece.edge_elements()) ec.v.push_back(n);
        in.counts.push_back(ec);
    }
    in.counts_given = true;
    // `wire_radius=radii[0] if len(set(radii)) == 1 else radii`
    bool uniform = true;
    for (const SeamPiece &piece : mesh.pieces)
        if (!(piece.radius == mesh.pieces[0].radius)) uniform = false;
    if (mesh.pieces.size() == 1) uniform = true;
    // `RazorSolver(wires=[])` (every GW a phantom, momwire#1139)
    if (mesh.pieces.empty()) throw SeamRaised{"ValueError", "wires must be non-empty"};
    if (uniform) {
        in.radius.assign(1, mesh.pieces[0].radius);
        in.radius_shape.clear();
    } else {
        for (const SeamPiece &piece : mesh.pieces) in.radius.push_back(piece.radius);
        in.radius_shape.assign(1, static_cast<int64_t>(mesh.pieces.size()));
    }
    in.feeds_given = true;
    for (long long s : mesh.feeds) {
        razor::FeedSpec f;
        f.wire = mesh.sites[s].piece;
        f.midpoint = false;
        f.arc = mesh.sites[s].arclength;
        f.volts = cd(0.0, 0.0);
        in.feeds.push_back(f);
    }
    for (long long s : mesh.gaps) {
        razor::NodeGapSpec g;
        g.wire = mesh.sites[s].piece;
        g.end = mesh.sites[s].end == SeamSite::at_start ? razor::End::start : razor::End::end;
        g.volts = cd(0.0, 0.0);
        in.node_gaps.push_back(g);
    }
    // `_solver_for`'s loading: `LD 5` conductivity and `LD 2` / `LD 3`
    // per-metre RLC, one entry per PIECE by its tag (NaN / none where the
    // tag carries none), and no keyword at all where the deck has none.
    if (!d.wire_conductivity.empty()) {
        in.loading.has_conductivity = true;
        for (const SeamPiece &piece : mesh.pieces) {
            double sigma = std::numeric_limits<double>::quiet_NaN();
            for (const auto &tc : d.wire_conductivity)
                if (tc.first == piece.tag) {
                    sigma = tc.second;
                    break;
                }
            in.loading.conductivity.push_back(sigma);
        }
    }
    if (!d.wire_distributed_rlc.empty()) {
        in.loading.has_rlc = true;
        for (const SeamPiece &piece : mesh.pieces) {
            razor::WireRLC e;
            for (const auto &tr : d.wire_distributed_rlc)
                if (tr.first == piece.tag) {
                    e.kind = tr.second.kind == "series" ? razor::WireRLC::series : razor::WireRLC::parallel;
                    e.r = tr.second.r;
                    e.l = tr.second.l;
                    e.c = tr.second.c;
                    break;
                }
            in.loading.rlc.push_back(e);
        }
    }
    in.has_ground = has_ground;
    in.ground_z = 0.0;
    // `_ground_kwargs`: GN 0 / GN 2 add `ground_eps=(medium.eps_r,
    // medium.sigma)` and `ground_model="sommerfeld"`, the composing fill.
    if (d.ground.kind == Ground::sommerfeld) {
        in.ground_eps_given = true;
        in.ground_eps_r = p.medium_eps_r;
        in.ground_sigma = p.medium_sigma;
        in.ground_sommerfeld = true;
    }
    in.wavelength = p.wavelength;
    in.extended_kernel = !all_below;
    in.exact_kernel = razor::ExactKernel::automatic;
    // momwire#1649 (`_solver_for`): the pieces cut at z = 0 are not asked
    for (size_t i = 0; i < mesh.pieces.size(); i++)
        if (mesh.pieces[i].cut_start || mesh.pieces[i].cut_end) in.exact_auto_skip.push_back(static_cast<int64_t>(i));
    p.engine = std::move(part);
}

// `RazorSolver(...)` and `compute_port_solution`'s front half.
SeamEngine::Built RazorSeamEngine::build(Prepared &p) const {
    RazorPrepared &part = razor_part(p);
    part.front = std::make_shared<razor::Front>();
    Built b;
    std::string why;
    try {
        why = part.front->build(part.inputs);
    } catch (const razor::FrontError &e) {
        const bool ni = e.kind == razor::FrontErrorKind::not_implemented;
        b.kind = ni && !e.solve_phase ? Built::refused : Built::raised;
        b.error_kind = ni ? "NotImplementedError" : "ValueError";
        b.message = e.what();
        return b;
    }
    if (!why.empty()) {
        b.kind = Built::declined;
        b.message = "front:" + why;
        b.unsolved = why == "buried-fill";
    }
    return b;
}

SeamResult razor_seam(const std::string &latin1_text, bool open_end) {
    return seam(latin1_text, RazorSeamEngine(), open_end);
}

}  // namespace nec5
}  // namespace mw
