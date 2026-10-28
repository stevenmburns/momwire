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
// - the lower medium (momwire#1607 B6): `_wire_media`, the crossing
//   junctions and this family's crossing scope, each refusal in Python's
//   words; a wholly-buried deck's basis at k_m, and a crossing deck's two-k
//   view (`_basis_coefs(geom, np.where(below, k_m, k_p))`: the crossing node
//   an ordinary junction, `_edge_weight` on its edges), each in numpy's
//   complex arithmetic, with the `Medium`.
//
// DECLINED (`nec4::Declined`), never misread: wires on both sides of the
// interface with no crossing (the transmitted block), a reflection-
// coefficient ground (not served by the NEC-4.2 slot), a malformed argument
// the constructor would refuse.
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
    std::string kind = "ValueError";  // or "NotImplementedError" (the crossing scope)
    explicit FrontError(const std::string &m) : std::runtime_error(m), message(m) {}
    FrontError(const std::string &m, const std::string &k) : std::runtime_error(m), message(m), kind(k) {}
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
    // `k_entry` (a complex k's view, momwire#1607 B6): each entry's own k;
    // empty at a real k
    std::vector<std::complex<double>> k_entry;
};

// `_crossing_fill.Medium` (`buried_medium`): the lower medium of a solve,
// CPython's complex arithmetic.
struct Medium {
    std::complex<double> eps_t, eps_m, k_m, c2, a_m;
    double k_p = 0;
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
    // momwire#1607 B6: `_wire_media`, per wire BELOW the interface; a
    // wholly-buried deck (`buried`) is solved in the lower medium
    // (`_operating_medium`): the basis above is `_basis_coefs(geom, k_m)`.
    std::vector<bool> below_wire;
    bool buried = false;
    Medium medium;
    // a crossing deck (momwire#1607 B6b, `_crossing_junction_indices`): its
    // crossing junctions, each an ordinary junction of the point-matched
    // lane; the basis is the two-k view (k_m below, k_p above), `below_seg`
    // its per-segment medium
    bool crossing = false;
    std::vector<long long> crossing_junctions;
    std::vector<bool> below_seg;
    // a mixed deck (momwire#1718 S12): wires on both sides with no crossing
    // junction; the basis is D2's stitch (`_stitch_basis_coefs`: each entry
    // from its segment's medium's view, `k_entry` its k), `below_seg` the
    // per-segment medium as on a crossing deck
    bool mixed = false;
};

// Throws FrontError where Python raises, nec4::Declined where this copy
// declines.
// `galerkin`: SinusoidalGalerkinSolver's constructor, which differs here only
// in its name and its coincident-conductor sentence (momwire#1718 S3).
Front front(const nec4::SolverArgs &args, bool galerkin = false);

// `_basis_coefs(geom, k)` on `fr`'s geometry at another real k (a readout at
// the caller's omega, momwire#1718 S8); declines a NEC-4 junction front.
SegView basis_coefs(const Front &fr, double k);

// A wholly-buried front read at another omega (`_buried_wire_loss_power`'s
// `buried_medium(ground_eps, omega, eps, omega / c)` and `_basis_coefs(geom,
// medium.k_m)`, momwire#1718 S11a): a copy whose k, medium and basis are
// that omega's.
Front buried_at(const Front &fr, double omega);

// The comparison format (`sinusoidal_front_compare.py`), per operating point
// of each built structure of a deck (the NEC-4.2 build's own enumeration).
std::string dump_front(const std::string &latin1_text);

}  // namespace sinusoidal
}  // namespace mw
