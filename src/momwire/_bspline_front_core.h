// PROTOTYPE (momwire#1517 step 1, not merged): the B-spline solver's FRONT
// half in C++, Python-free, unit by unit. This first unit is the geometry:
// `BSplineSolver._build_geometry`, the wires discretized into segments with
// every array the fill reads, bit for bit numpy's.
#pragma once

#include <cstdint>
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

// `polylines[w]`: (n_points, 3) row-major; `n_per_edge[w]`: one count per
// edge. Throws std::invalid_argument("wire W edge E has zero length") where
// Python raises that ValueError.
Geometry build_geometry(const std::vector<std::vector<double>> &polylines,
                        const std::vector<std::vector<int64_t>> &n_per_edge);

}  // namespace bspline
}  // namespace mw
