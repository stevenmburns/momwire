// momwire#1517 step 1: the B-spline basis' answers to the seam --
// see `_nec5_bspline_seam_core.h`.
#include "_nec5_bspline_seam_core.h"

#if defined(_MSC_VER)
#pragma float_control(precise, on)
#pragma fp_contract(off)
#endif

#include <algorithm>
#include <cmath>
#include <limits>
#include <memory>
#include <string>
#include <stdexcept>

namespace mw {
namespace nec5 {

const BSplinePrepared &bspline_part(const Prepared &p) {
    const BSplinePrepared *b = dynamic_cast<const BSplinePrepared *>(p.engine.get());
    if (b == nullptr) throw std::logic_error("a preparation bspline's engine did not make");
    return *b;
}

BSplinePrepared &bspline_part(Prepared &p) {
    BSplinePrepared *b = dynamic_cast<BSplinePrepared *>(p.engine.get());
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

namespace {

// The seam's arguments as the constructor's wire-indexed ones.
bspline::WireArgs wire_args(const BSplineFrontInputs &in) {
    bspline::WireArgs wa;
    for (size_t w = 0; w < in.n_per_edge.size(); w++) {
        wa.polylines.emplace_back(in.points.begin() + 6 * w, in.points.begin() + 6 * w + 6);
        wa.n_per_edge.push_back({in.n_per_edge[w]});
    }
    wa.radius = in.radius;
    wa.radius_per_wire = in.radius_per_wire;
    wa.has_conductivity = in.has_conductivity;
    wa.conductivity = in.conductivity;
    wa.has_rlc = in.has_rlc;
    for (const auto &e : in.rlc) wa.rlc.push_back({e.kind, e.r, e.l, e.c});
    for (const auto &f : in.feeds) wa.feeds.push_back({f.wire, f.arc});
    for (const auto &g : in.node_gaps) wa.node_gaps.push_back({g.wire, g.end == End::start});
    wa.junctions_given = !in.junctions.empty();
    for (const auto &g : in.junctions) {
        std::vector<std::pair<int64_t, bool>> mm;
        for (const auto &e : g) mm.push_back({e.first, e.second == End::start});
        wa.junctions.push_back(mm);
    }
    return wa;
}

// `configure_loading`'s fail-fast checks this unit does not spell:
// a conductivity at or below zero raises a ValueError there.
bool loading_refused(const bspline::WireArgs &wa) {
    if (wa.has_conductivity)
        for (double c : wa.conductivity)
            if (std::isfinite(c) && c <= 0.0) return true;
    return false;
}

}  // namespace

SeamEngine::Built BSplineSeamEngine::build(Prepared &p) const {
    BSplinePrepared &part = bspline_part(p);
    const BSplineFrontInputs &in = part.inputs;
    Built b;
    const auto decline = [&](const std::string &why) {
        part.front.reset();
        b.kind = Built::declined;
        b.message = "front:" + why;
        b.unsolved = true;
        return b;
    };
    // step 1's scope (momwire#1517): free space; ground is step 3
    // step 3a: a perfect ground; the Sommerfeld composition is step 3b
    if (in.has_ground && in.ground_eps_given) return decline("bspline-sommerfeld");
    auto fr = std::make_shared<BSplineFront>();
    fr->args = wire_args(in);
    try {
        fr->dedup = bspline::drop_duplicated_wires(fr->args);
        bspline::check_ports(fr->args);
        fr->geom = bspline::build_geometry(fr->args.polylines, fr->args.n_per_edge);
        fr->radii = fr->args.radius;
        if (!fr->args.radius_per_wire) fr->radii.assign(fr->args.polylines.size(), fr->args.radius[0]);
        fr->basis = bspline::build_basis(fr->geom, fr->args.polylines, fr->args.junctions, fr->radii, in.has_ground,
                                         in.ground_z, 2);
        fr->B = bspline::port_columns(fr->args, fr->geom, fr->basis);
    } catch (const bspline::FrontNotPorted &e) {
        return decline(std::string("bspline-") + e.what());
    } catch (const bspline::FrontValueError &e) {
        // Python's ValueError, which escapes to the INTERNAL ERROR frame
        b.kind = Built::raised;
        b.error_kind = "ValueError";
        b.message = e.what();
        return b;
    } catch (const std::invalid_argument &e) {  // `_build_geometry`'s zero-length edge
        b.kind = Built::raised;
        b.error_kind = "ValueError";
        b.message = e.what();
        return b;
    }
    fr->n_ports = static_cast<int64_t>(fr->args.feeds.size() + fr->args.node_gaps.size());
    if (loading_refused(fr->args)) return decline("bspline-loading-refused");
    fr->loading.has_conductivity = fr->args.has_conductivity;
    fr->loading.conductivity = fr->args.conductivity;
    fr->loading.has_rlc = fr->args.has_rlc;
    for (const auto &e : fr->args.rlc) {
        razor::WireRLC r;
        r.kind = static_cast<razor::WireRLC::Kind>(e.kind);
        r.r = e.r;
        r.l = e.l;
        r.c = e.c;
        fr->loading.rlc.push_back(r);
    }
    // `exact_kernel="auto"` (`_resolve_exact_kernel_auto`): on when some
    // segment is shorter than 3 radii (free space serves it)
    fr->exact_kernel = bspline::exact_kernel_auto(fr->geom, fr->radii);
    if (fr->exact_kernel && bspline::exact_kernel_has_cross_job(fr->geom, fr->radii))
        return decline("bspline-exact-kernel-radius-step");
    // `self.c = 1 / np.sqrt(self.eps * self.mu)`, `freq = c / wavelength`,
    // `omega = 2 * np.pi * freq`, `k = omega / c`
    fr->c = 1.0 / std::sqrt(BSPLINE_EPS * BSPLINE_MU);
    const double freq = fr->c / in.wavelength;
    fr->omega = (2.0 * 3.141592653589793) * freq;
    fr->k = fr->omega / fr->c;
    fr->extended_kernel = in.extended_kernel;
    fr->has_ground = in.has_ground;
    fr->ground_z = in.ground_z;
    part.front = std::move(fr);
    return b;
}

SeamResult bspline_seam(const std::string &latin1_text, bool open_end) {
    return seam(latin1_text, BSplineSeamEngine(), open_end);
}

}  // namespace nec5
}  // namespace mw
