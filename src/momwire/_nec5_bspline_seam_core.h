// momwire#1517 step 1: the NEC-5 seam's structure
// half for bspline (bs2) in C++ -- the engine-neutral seam
// (`_nec5_seam_core.h`) with the B-spline basis' answers to what a basis
// decides, so a parsed deck comes out as the keyword arguments the Python seam
// hands `BSplineSolver` (`construct_solver`), with no Python anywhere:
//
//   * `basis()`: "bspline", the CUT spelling (`_CUT_SPELLING` is
//     `(BSplineSolver,)`: a GW is cut at every node a card addresses inside
//     it, and a series EMF at a two-wire node is a node gap), and a node gap
//     on a crossing node (`_crossing_spelling`: node gaps AND junction ports);
//   * `check_can_host`: `_check_basis_can_host` on BSplineSolver's row, which
//     refuses none of the four (node gaps, knot feeds, contact over a medium,
//     buried and crossing wires are all served; the inert one-segment wire is
//     razor's alone, `_NATIVE_LOADING`);
//   * `solver_arguments`: `construct_solver(BSplineSolver, {"exact_kernel":
//     "auto"}, ...)`: junctions declared (not native loading), node gaps
//     where any, the extended kernel by default and served in part over a
//     buried wire (`extended_kernel_partial`), off only when every wire is
//     below the plane;
//   * `build`: the constructor's front half (`_bspline_front_core.h`): the
//     duplicate drop, the port checks, the geometry, the basis and the port
//     columns. STEP 1'S SCOPE: free space, unloaded, the dense moment tensor
//     inside its budget and the exact kernel left off by "auto"; anything
//     else declines (kept for the gates, never solved).
#pragma once

#include "_bspline_fill_core.h"
#include "_bspline_front_core.h"
#include "_nec5_seam_core.h"
#include "_razor_wire_loading_core.h"

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace mw {
namespace nec5 {

// `BSplineSolver(...)`'s keyword arguments as the seam spells them.
struct BSplineFrontInputs {
    std::vector<double> points;       // `wires`: per piece (start, end), 6 doubles
    std::vector<int64_t> n_per_edge;  // `n_per_edge_per_wire`: [[n]] per piece
    struct Feed {
        int64_t wire = 0;
        double arc = 0;
    };
    std::vector<Feed> feeds;  // `feeds`: (wire, arclength, 0j)
    struct Gap {
        int64_t wire = 0;
        End end = End::start;
    };
    std::vector<Gap> node_gaps;  // `node_gaps` (passed only when any)
    std::vector<std::vector<JunctionEnd>> junctions;  // `junctions` (None when empty)
    std::vector<double> radius;  // one value (uniform census) or one per piece
    bool radius_per_wire = false;
    double wavelength = 0;
    bool extended_kernel = false;
    // `wire_conductivity` (NaN: none on that piece) and `distributed_rlc`
    // (kind 0: None, 1 series, 2 parallel), one entry per piece
    bool has_conductivity = false;
    std::vector<double> conductivity;
    struct RLC {
        int kind = 0;
        double r = 0, l = 0, c = 0;
    };
    bool has_rlc = false;
    std::vector<RLC> rlc;
    // `_ground_kwargs`
    bool has_ground = false;
    double ground_z = 0;
    bool ground_eps_given = false;
    double ground_eps_r = 0, ground_sigma = 0;
};

// The constructor's front half for one preparation (`BSplineSeamEngine::
// build`): the wire arguments after the duplicate drop, the geometry, the
// basis and the port columns -- what `compute_port_solution` reads before
// its fill, and what the readouts read after it.
struct BSplineFront {
    bspline::WireArgs args;     // after `drop_duplicated_wires`
    bspline::WireDedup dedup;
    bspline::Geometry geom;
    bspline::Basis basis;
    std::vector<double> radii;  // per solved wire
    std::vector<double> B;      // `_port_columns`' (n_basis, n_ports), row-major
    int64_t n_ports = 0;
    // `self.c`, `self.omega`, `self.k` as the constructor forms them
    double c = 0, omega = 0, k = 0;
    bool extended_kernel = false;
    bool exact_kernel = false;  // "auto", resolved
    // the ground (`ground_z` given): PEC, or the Sommerfeld composition
    bool has_ground = false;
    double ground_z = 0;
    // `configure_loading`'s `wire_conductivity` / `distributed_rlc` per
    // solved wire (`_loading_active` is `loading.active()`), and the loading
    // Gram, built on the first solve (`_cached_loading_gram`)
    razor::WireLoading loading;
    bool has_gram = false;
    bspline::Triplets gram;
};

struct BSplinePrepared final : EnginePrepared {
    BSplineFrontInputs inputs;
    std::shared_ptr<BSplineFront> front;  // set by `build`; reset on release
};

const BSplinePrepared &bspline_part(const Prepared &p);
BSplinePrepared &bspline_part(Prepared &p);

// bs2's constants (`BSplineSolver`'s eps / mu defaults).
constexpr double BSPLINE_EPS = 8.8541878188e-12;
constexpr double BSPLINE_MU = 1.25663706127e-6;

class BSplineSeamEngine final : public SeamEngine {
  public:
    const SeamBasis &basis() const override;
    void check_can_host(const Deck &d, const Prepared &p, bool has_ground) const override;
    void solver_arguments(const Deck &d, Prepared &p, bool has_ground) const override;
    Built build(Prepared &p) const override;
};

SeamResult bspline_seam(const std::string &latin1_text, bool open_end = false);

}  // namespace nec5
}  // namespace mw
