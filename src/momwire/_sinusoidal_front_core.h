// The sinusoidal basis' FRONT HALF in C++ (momwire#1607 B2), with no Python
// in it. It takes a solver's constructor arguments (`nec4::SolverArgs`, the
// segment-centre engine interface the NEC-4.2 build produces) and gives what
// `SinusoidalSolver` derives from them before any fill:
// - the constructor's constants (c, freq, omega, k, eta from the class's
//   eps and mu);
// - `_wire_spec.drop_duplicated_wires`: a wire listed twice is solved once,
//   or refused in the constructor's words (momwire#1042 / #1333);
// - its junction normalisation (`_wire_spec.normalize_junctions`, inferring
//   coincident ends when none are given, and the coincidence check);
// - `_build_geometry`: segments, the N-/N+ neighbour tables with their sigma
//   signs, the grounded junctions and ground-contact flags, and the feed
//   snap (`_feed_snap.snap`);
// - `_basis_coefs(geom, k)`: the closed-form A, B, C and AC coefficients
//   and the CSR-by-segment view.
//
// Python stays the reference: every array is Python's to the bit, gated by
// `tests/test_sinusoidal_front_core.py` through
// `scripts/core_split/sinusoidal_front_compare.py`. The numpy elementwise
// arithmetic is written out in numpy's own order; numpy's sums are its
// pairwise_sum, and its float64 sin / cos / log are libm's under the
// AVX-512 pins every momwire process sets (`momwire_serve_client.cpu_env`).
//
// DECLINED (`nec4::Declined`), never misread: a buried or plane-crossing
// structure (the complex-wavenumber basis, momwire#1607 B6), a
// reflection-coefficient ground (not served by the NEC-4.2 slot), a malformed
// argument the constructor would refuse.
#pragma once

#include "_nec4_build_core.h"

#include <complex>
#include <cstdint>
#include <string>
#include <vector>

namespace mw {
namespace sinusoidal {

// A refusal the constructor or `_build_geometry` raises (ValueError).
struct FrontError : std::runtime_error {
    std::string message;
    explicit FrontError(const std::string &m) : std::runtime_error(m), message(m) {}
};

struct FeedSite {  // (wire, requested, clamped, centre)
    long long wire = 0;
    double requested = 0, clamped = 0, centre = 0;
};

struct Geometry {  // `_build_geometry`'s dict, in its key order
    std::vector<nec4::Point3> seg_l, seg_r, seg_centers, seg_tangents;
    std::vector<double> seg_h;
    long long n_segs = 0;
    std::vector<long long> nm_basis, nm_seg;
    std::vector<int8_t> nm_sigma;
    std::vector<long long> np_basis, np_seg;
    std::vector<int8_t> np_sigma;
    std::vector<long long> nm_count, np_count;
    std::vector<long long> wire_first, wire_last;
    bool has_feed_seg = false;
    long long feed_seg = 0;
    std::vector<long long> feed_segs;
    std::vector<double> feed_xi;
    std::vector<FeedSite> feed_sites;
    std::vector<bool> ground_minus, crossing_minus, crossing_plus, ground_plus;
    std::vector<long long> grounded_junctions;  // sorted
};

struct SegView {  // `_basis_coefs`' dict
    std::vector<long long> starts, jbasis;
    std::vector<std::complex<double>> A, B, C, AC;
    std::vector<int8_t> sigma;
};

struct Front {
    double c = 0, freq = 0, omega = 0, k = 0, eta = 0;
    std::vector<std::vector<std::pair<long long, bool>>> junctions;  // (wire, start)
    std::vector<double> radius_per_wire;
    bool uniform_radius = true;
    Geometry geometry;
    SegView basis;
    // momwire#1644: the NEC-4 dialect's junction is live on this structure
    // (`SinusoidalSolver._nec4_live`): the basis above carries the junction
    // solve's per-end factors and the end caps, and the fill takes the NEC-4
    // kernel and the caps' fields.
    bool nec4 = false;
    // The constructor's arguments after `_wire_spec.drop_duplicated_wires`
    // (momwire#1042 / #1333): the wires it solves, every wire-indexed
    // argument remapped, and how they map back onto the caller's.
    nec4::SolverArgs args;
    nec4::WireDedup dedup;
};

// Throws FrontError where Python raises, nec4::Declined where this copy
// declines.
// `galerkin`: SinusoidalGalerkinSolver's constructor, which differs here only
// in its name and its coincident-conductor sentence (momwire#1718 S3).
Front front(const nec4::SolverArgs &args, bool galerkin = false);

// The comparison format (`sinusoidal_front_compare.py`), per operating point
// of each built structure of a deck (the NEC-4.2 build's own enumeration).
std::string dump_front(const std::string &latin1_text);

}  // namespace sinusoidal
}  // namespace mw
