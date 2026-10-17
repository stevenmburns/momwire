// PROTOTYPE (momwire#1517 step 1): the B-spline basis' answers to the seam --
// see `_nec5_bspline_seam_core.h`.
#include "_nec5_bspline_seam_core.h"

#if defined(_MSC_VER)
#pragma float_control(precise, on)
#pragma fp_contract(off)
#endif

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace mw {
namespace nec5 {

const BSplinePrepared &bspline_part(const Prepared &p) {
    const BSplinePrepared *b = dynamic_cast<const BSplinePrepared *>(p.engine.get());
    if (b == nullptr) throw std::logic_error("a preparation bspline's engine did not make");
    return *b;
}

const SeamBasis &BSplineSeamEngine::basis() const {
    static const SeamBasis row = [] {
        SeamBasis b;
        b.name = "bspline";
        b.cut_spelling = true;            // `_CUT_SPELLING = (BSplineSolver,)`
        b.crossing_node_spelling = true;  // node gaps and junction ports
        return b;
    }();
    return row;
}

// BSplineSolver's row refuses none of `_check_basis_can_host`'s four.
void BSplineSeamEngine::check_can_host(const Deck &d, const Prepared &p, bool has_ground) const {
    (void)d;
    (void)p;
    (void)has_ground;
}

void BSplineSeamEngine::solver_arguments(const Deck &d, Prepared &p, bool has_ground) const {
    const SeamMesh &mesh = p.mesh;
    // `RazorSolver(wires=[])`'s twin: `BSplineSolver` refuses an empty census
    // the same way (every GW a phantom, momwire#1139)
    if (mesh.pieces.empty()) throw SeamRaised{"ValueError", "wires must be non-empty"};
    auto part = std::make_shared<BSplinePrepared>();
    BSplineFrontInputs &in = part->inputs;
    for (const SeamPiece &piece : mesh.pieces) {
        in.points.insert(in.points.end(), piece.points, piece.points + 6);
        in.n_per_edge.push_back(piece.n_elements());
    }
    for (long long s : mesh.feeds)
        in.feeds.push_back({mesh.sites[static_cast<size_t>(s)].piece, mesh.sites[static_cast<size_t>(s)].arclength});
    for (long long s : mesh.gaps)
        in.node_gaps.push_back({mesh.sites[static_cast<size_t>(s)].piece,
                                mesh.sites[static_cast<size_t>(s)].end == SeamSite::at_start ? End::start : End::end});
    in.junctions = mesh.junctions;
    bool uniform = true;
    for (const SeamPiece &piece : mesh.pieces)
        if (!(piece.radius == mesh.pieces[0].radius)) uniform = false;
    if (uniform) {
        in.radius.assign(1, mesh.pieces[0].radius);
    } else {
        for (const SeamPiece &piece : mesh.pieces) in.radius.push_back(piece.radius);
        in.radius_per_wire = true;
    }
    in.wavelength = p.wavelength;
    // the extended kernel by default; over a buried wire served in part
    // (`extended_kernel_partial`), and the reduced kernel when EVERY wire is
    // below the plane
    long long below = 0;
    if (has_ground)
        for (const SeamPiece &piece : mesh.pieces)
            if (std::min(piece.points[2], piece.points[5]) <
                0.0 - ground_touch_tol(piece.points, piece.points + 3))
                below++;
    in.extended_kernel = !(below > 0 && below == static_cast<long long>(mesh.pieces.size()));
    if (!d.wire_conductivity.empty()) {
        in.has_conductivity = true;
        for (const SeamPiece &piece : mesh.pieces) {
            double sigma = std::numeric_limits<double>::quiet_NaN();
            for (const auto &tc : d.wire_conductivity)
                if (tc.first == piece.tag) {
                    sigma = tc.second;
                    break;
                }
            in.conductivity.push_back(sigma);
        }
    }
    if (!d.wire_distributed_rlc.empty()) {
        in.has_rlc = true;
        for (const SeamPiece &piece : mesh.pieces) {
            BSplineFrontInputs::RLC e;
            for (const auto &tr : d.wire_distributed_rlc)
                if (tr.first == piece.tag) {
                    e.kind = tr.second.kind == "series" ? 1 : 2;
                    e.r = tr.second.r;
                    e.l = tr.second.l;
                    e.c = tr.second.c;
                    break;
                }
            in.rlc.push_back(e);
        }
    }
    in.has_ground = has_ground;
    in.ground_z = 0.0;
    if (d.ground.kind == Ground::sommerfeld) {
        in.ground_eps_given = true;
        in.ground_eps_r = p.medium_eps_r;
        in.ground_sigma = p.medium_sigma;
    }
    p.engine = std::move(part);
}

SeamEngine::Built BSplineSeamEngine::build(Prepared &p) const {
    (void)p;
    Built b;
    b.kind = Built::declined;
    b.message = "front:bspline-front";
    b.unsolved = true;
    return b;
}

SeamResult bspline_seam(const std::string &latin1_text, bool open_end) {
    return seam(latin1_text, BSplineSeamEngine(), open_end);
}

}  // namespace nec5
}  // namespace mw
