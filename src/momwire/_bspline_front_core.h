// PROTOTYPE (momwire#1517 step 1, not merged): the B-spline solver's FRONT
// half in C++, Python-free, unit by unit. This first unit is the geometry:
// `BSplineSolver._build_geometry`, the wires discretized into segments with
// every array the fill reads, bit for bit numpy's.
#pragma once

#include <cstdint>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace mw {
namespace bspline {

// One wire's part of `_build_geometry`'s `per_wire`.
struct WireGeometry {
    std::vector<double> seg_l, seg_r, tangents;  // (n_total, 3) row-major
    std::vector<double> h_per_seg;               // (n_total,)
    std::vector<int64_t> edge_offsets;           // n_edges + 1
    std::vector<std::vector<double>> edge_arc_edges;  // per edge, n_e + 1
    std::vector<double> arc_at_knot;             // n_total + 1
    int64_t n_total = 0;
};

struct Geometry {
    std::vector<WireGeometry> per_wire;
    std::vector<int64_t> seg_offsets;  // n_wires + 1
    int64_t n_segs_total = 0;
    std::vector<double> h_per_seg, tangents, seg_l, seg_r;  // global, concatenated
};

// The constructor's wire-indexed arguments, before and after
// `_wire_spec.drop_duplicated_wires` (momwire#1042 / #1333).
struct WireArgs {
    std::vector<std::vector<double>> polylines;      // (n_points, 3) each
    std::vector<std::vector<int64_t>> n_per_edge;    // per wire, per edge
    // `wire_radius`: one value, or one per wire (`radius_per_wire`)
    std::vector<double> radius;
    bool radius_per_wire = false;
    // `wire_conductivity` (NaN: none) and `distributed_rlc` (kind 0: None),
    // per wire when given
    bool has_conductivity = false;
    std::vector<double> conductivity;
    struct RLC {
        int kind = 0;
        double r = 0, l = 0, c = 0;
    };
    bool has_rlc = false;
    std::vector<RLC> rlc;
    // the sites' wire indices (`feeds` then `node_gaps`), with their payloads
    struct Feed {
        int64_t wire = 0;
        double arc = 0;
    };
    std::vector<Feed> feeds;
    struct Gap {
        int64_t wire = 0;
        bool at_start = true;
    };
    std::vector<Gap> node_gaps;
    // declared `junctions`: (wire, at_start) members; none -> inferred
    bool junctions_given = false;
    std::vector<std::vector<std::pair<int64_t, bool>>> junctions;
};

// `WireDedup`: the caller's wire count, the kept wires (caller's indices)
// and each dropped copy (copy, kept twin, reversed), sorted by copy.
struct WireDedup {
    bool active = false;
    int64_t n_input = 0;
    std::vector<int64_t> kept;
    struct Dropped {
        int64_t copy = 0, twin = 0;
        bool reversed = false;
    };
    std::vector<Dropped> removed;
};

// Python's ValueError (its message) and NotImplementedError, from the front.
struct FrontValueError : std::runtime_error {
    using std::runtime_error::runtime_error;
};
struct FrontNotPorted : std::runtime_error {
    using std::runtime_error::runtime_error;
};

// `drop_duplicated_wires("BSplineSolver", ...)` with no junction-indexed
// ports: `args` reduced in place, the dedup returned. Refuses as Python does
// (FrontValueError); inferred junctions throw FrontNotPorted.
WireDedup drop_duplicated_wires(WireArgs &args);

// `polylines[w]`: (n_points, 3) row-major; `n_per_edge[w]`: one count per
// edge. Throws std::invalid_argument("wire W edge E has zero length") where
// Python raises that ValueError.
Geometry build_geometry(const std::vector<std::vector<double>> &polylines,
                        const std::vector<std::vector<int64_t>> &n_per_edge);

// `_build_basis_polynomials` (with `_wire_endpoint_status` and
// `_kcl_row_junctions`): per kept basis its (d+1) wing segments and their
// polynomial coefficients, the KCL rows, and each wire's knots.
struct Basis {
    int degree = 2;
    int64_t n_basis = 0;
    std::vector<int64_t> supp_seg;  // (n_basis, d+1), 0 past a truncated wing
    std::vector<double> polys;      // (n_basis, d+1, d+1)
    int64_t n_kcl = 0;
    std::vector<double> kcl_A;      // (n_kcl, n_basis)
    std::vector<std::vector<double>> wire_knots;
    // `_wire_endpoint_status`: per wire, "free" (-1), "ground" (-2) or the
    // junction index, at the start and at the end
    std::vector<int64_t> start_status, end_status;
    // per wire, the global index of its directional basis at the start /
    // the end (-1: none), which a node gap's column names
    std::vector<int64_t> start_dir, end_dir;
    std::vector<bool> grounded_junction;
    // per wire, (basis j of its knot vector, global basis index) per kept
    // basis: `wire_basis_global`'s mapping
    std::vector<std::vector<std::pair<int64_t, int64_t>>> wire_globals;
};

// `ground_z` given or not; `radii` per wire (the conductor's, for the
// stand-off floor). A buried or crossing deck (a wire in the lower medium)
// and a wire below the stand-off floor throw FrontNotPorted; an edge lying in
// the plane throws FrontValueError with Python's sentence.
Basis build_basis(const Geometry &g, const std::vector<std::vector<double>> &polylines,
                  const std::vector<std::vector<std::pair<int64_t, bool>>> &junctions,
                  const std::vector<double> &radii, bool has_ground, double ground_z,
                  int degree = 2);

// The constructor's port spec after the duplicate drop, in its order:
// `feeds` (at least one entry unless node gaps drive), then
// `normalize_junctions` (`check_junction_coincidence`) and
// `normalize_node_gaps`. Throws FrontValueError with Python's sentence;
// a non-coincident junction throws FrontNotPorted.
void check_ports(const WireArgs &args);

// `_node_gap_columns`: (n_basis, n_gaps) row-major, the sigma-signed unit
// indicator of each named member's directional basis.
std::vector<double> node_gap_columns(const WireArgs &args, const Basis &b);

// `_port_columns`' B for the seam's ports, (n_basis, n_feeds + n_gaps)
// row-major, real (numpy's complex128 holds +0 imaginary parts): each gap
// feed's delta-gap source vector (`_build_source_vector`, the point model,
// v_m = Phi_m(s_f) with the wire-end clip), then the node-gap columns. The
// constraint rows are `b.kcl_A` (no junction ports). A feed off its wire's
// knot range throws FrontNotPorted (Python raises there).
std::vector<double> port_columns(const WireArgs &args, const Geometry &g, const Basis &b);

}  // namespace bspline
}  // namespace mw
