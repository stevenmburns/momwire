// The NEC-5 seam's STRUCTURE HALF for razor-2p in C++: a parsed deck
// (`_nec5_deck_core.h`) in, the arguments the seam hands `RazorSolver`
// (`_razor_front_core.h`'s `FrontInputs`) out, or the refusal the seam
// prints, with no Python anywhere. It is the hardened copy of
// `eznec/_serve.py`'s path from `serve` / `serve_runs` to the solver's
// constructor:
//
//   * the scope ladder, `refusal` (`_drive_refusal`, `_load_probe_refusal`,
//     `_geometry_refusal`, the request rungs), every sentence word for word;
//   * `_prepare` up to `compute_port_solution`: `structure_of`, `build_mesh`
//     (one piece per GW, the junctions, one site per addressed node,
//     `_site_for`'s refusals), `_assign_columns`, `_check_one_port_per_drive`,
//     `_check_basis_can_host`, `_medium` / `_ground_kwargs`, and
//     `construct_solver`'s extended-kernel default (which razor always takes
//     on the decks covered here);
//   * `serve_runs`' sharing: a run whose sources sit on the sites of the run
//     prepared before it reuses that preparation.
//
//     mw::nec5::SeamResult r = mw::nec5::razor_seam(latin1_text);
//     // r.route: served (every structure's runs prepared or shared);
//     // refused (the stream stops on that refusal's printout, after the
//     // structures before it); raised; declined
//     for (const mw::nec5::Prepared &p : r.prepared)   // built fronts
//         mw::razor::solve_ports(*p.front, ctx, lapack, X, 0);   // currents
//
// SCOPE (the decks the C++ solve covers): free space, GN 1, or a bare GD
// (which the seam solves over a PERFECT image), no TL / NT, no WIRE loading
// (LD 5 / LD 2 / LD 3; a lumped LD 0 / 1 / 4 is a port site here, its
// impedance the run half's), no phantom wire, finite coordinates, frequency
// and radii, at most `MAX_SEGMENTS` segments in all. Anything else is
// DECLINED by name (`SeamResult::message`), at the point of the ladder where
// the Python seam would first need the part not ported, so a decline never
// hides a refusal the Python seam would have printed earlier, and a refusal
// printed here is one the Python seam prints at the same rung. The Python seam stays the
// reference; `scripts/core_split/seam_compare.py` and
// `tests/test_nec5_razor_seam_core.py` gate this one against it field for
// field.
//
// The front half is built here too, because whether the solver's
// constructor refuses (NotImplementedError: the seam's refusal) or raises
// (anything else: an internal error) is part of the seam's answer; a front
// half that declines (`razor::Front::build`'s reason, e.g. the exact ring
// kernel engaging) declines the deck as "front:<reason>".
//
// A `raised` outcome is an exception the Python seam lets escape (a reader
// IndexError, a constructor ValueError, a NotImplementedError from the
// solve): `_shell.main`'s last line of defence discards whatever the stream
// had printed and writes the whole deck's refusal printout with the reason
// "INTERNAL ERROR IN MOMWIRE ENGINE - <Kind>: <message>"
// (`internal_error_reason`). The executable does the same with the kind and
// message here, which are Python's own (the type name and `str(exc)`).
#pragma once

#include "_nec5_deck_core.h"
#include "_razor_front_core.h"

#include <array>
#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace mw {
namespace nec5 {

// The most segments (all GW cards together) the C++ seam meshes; more
// decline. Far past what a dense solve can hold, so it only bounds the
// node grid.
constexpr long long MAX_SEGMENTS = 10000000;

// `_serve.SPEED_OF_LIGHT_MHZ_M`, `_NODE_EPS`, `_PROBE_VOLTS`,
// `EPSC_CONDUCTIVITY_FACTOR`; a test pins each to its Python twin.
constexpr double SPEED_OF_LIGHT_MHZ_M = 299.8;
constexpr double NODE_EPS = 1e-6;
constexpr double PROBE_VOLTS = 1e-9;
constexpr double EPSC_CONDUCTIVITY_FACTOR = 59.96;

using NodeKey = std::array<long long, 3>;  // `_node_key`

struct SeamPiece {  // `_Piece` (never cut here: one per GW)
    Int tag;
    long long first_node = 0, last_node = 0;
    double points[6] = {0, 0, 0, 0, 0, 0};  // start, end
    double radius = 0;
    long long n_elements() const { return last_node - first_node; }
    double length() const;  // `_vec3.norm3(points[1] - points[0])`
};

struct SeamSite {  // `_Site`
    Node at;
    enum Spelling { gap, node } spelling = gap;
    long long piece = -1;
    enum EndKind { at_start, at_end, interior } end = at_start;
    double sign = 0, arclength = 0;
    bool contact = false;
    long long index = -1, column = -1;
    double weight = 0;
};

using JunctionEnd = std::pair<long long, razor::End>;  // (piece, which end)

struct SeamMesh {  // `_Mesh`
    std::vector<SeamPiece> pieces;
    std::vector<std::vector<JunctionEnd>> junctions;
    std::vector<SeamSite> sites;
    std::vector<long long> feeds, gaps;                     // site indices
    std::vector<std::pair<long long, long long>> element_of;  // (piece, element)
    long long n_columns() const { return static_cast<long long>(feeds.size() + gaps.size()); }
};

struct SeamStructure {  // `Structure`: the STRUCTURE SPECIFICATION counts
    long long node_count = 0, wire_element_count = 0, unknown_count = 0;
    bool ground_plane = false;
};

// One `_prepare`: everything the run half needs from its structure.
struct Prepared {
    razor::FrontInputs inputs;  // what the seam passes RazorSolver
    // The C++ front half built on `inputs` (construction, geometry, tables,
    // port columns): what `RazorSolver(...)` and `compute_port_solution`'s
    // front half do with them. Ready for `razor::solve_ports`.
    std::shared_ptr<razor::Front> front;
    SeamStructure structure;
    SeamMesh mesh;
    double frequency_mhz = 0, wavelength = 0, omega = 0;
    bool has_medium = false;  // GD: the medium the far field reflects off
    double medium_eps_r = 0, medium_sigma = 0;
    double medium_eps_c[2] = {0, 0};
    std::vector<Node> source_sites;  // the drive's addresses, deck order
};

struct RunOutcome {
    enum Kind { prepared, shared, refused, raised, declined } kind = declined;
    std::string message;     // refused: the sentence; raised: str(exc); declined: why
    std::string error_kind;  // raised: the Python exception's type name
    long long prepared_index = -1;  // prepared / shared: into SeamResult::prepared
};

struct StructureOutcome {
    Answer::Kind read = Answer::refused;  // the reader's answer
    std::vector<RunOutcome> runs;         // single: one; multi: one per run, to the stop
};

struct SeamResult {
    enum Route { served, refused, raised, declined } route = declined;
    // refused: the sentence the stopping printout carries; raised: str(exc)
    // and its type name; declined: the reason.
    std::string message, error_kind;
    bool chained = false;
    std::vector<StructureOutcome> structures;
    std::vector<Prepared> prepared;
    // The reader's answers the outcomes above are for (`read_stream`): each
    // structure's deck and its runs' decks, which the run half reads.
    StreamRead stream;
};

// The whole seam, deck text to the solver's arguments: `_shell.render`'s
// nec5 path up to each structure's solve.
SeamResult razor_seam(const std::string &latin1_text);

// One structure's deck (already read) through `_serve.refusal`: "" when in
// scope, else the sentence; `declined` names a decline instead.
std::string scope_refusal(const Deck &deck, std::string &declined);

// `_shell._internal_error_reason`: the last line of defence's sentence.
std::string internal_error_reason(const std::string &error_kind, const std::string &message);

}  // namespace nec5
}  // namespace mw
