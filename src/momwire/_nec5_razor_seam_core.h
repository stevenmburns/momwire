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
// SCOPE (the decks the C++ solve covers): free space, GN 1, a bare GD
// (which the seam solves over a PERFECT image) or GN 0 / GN 2 above the
// plane (the composing Sommerfeld fill, momwire#1479 S3); a buried deck's
// structure half (the plane crossings split at z = 0, momwire#1281, and the
// crossing sites, momwire#1282) is here, its fill declines "front:buried-fill"
// (momwire#1512); no TL / NT, no WIRE loading
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
#include <map>
#include <set>
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

// `_PlaneCrossing` (momwire#1281): where a card crossing z = 0 meets it,
// at a node (`node`) or inside a segment (`segment`, 0-based); -1 = none.
struct PlaneCrossing {
    Int tag;
    double point[3] = {0, 0, 0};
    long long node = -1, segment = -1;
};

// One `_span_table` row: a card, or one half of a crossing card.
struct Span {
    Int tag;
    double a[3] = {0, 0, 0}, b[3] = {0, 0, 0};
    double tol = 0;
};

std::vector<PlaneCrossing> plane_crossings(const Deck &d);
std::vector<Span> span_table(const Deck &d);
std::set<NodeKey> crossing_nodes(const Deck &d);
bool has_buried_wire(const Deck &d);
// `_split_notes` (momwire#1281): one printout line per card split at z = 0.
std::vector<std::string> split_notes(const Deck &d);

// `_SplitElement`: a deck element cut in two at z = 0, read back on the
// pieces holding its two end nodes, its charge at its old centre.
struct SplitElement {
    long long lower = -1, upper = -1, centre_piece = -1;
    double centre_s = 0;
};

struct SeamPiece {  // `_Piece`: a run of one GW (cut only at z = 0)
    Int tag;
    long long first_node = 0, last_node = 0;
    double points[6] = {0, 0, 0, 0, 0, 0};  // start, end
    double radius = 0;
    bool cut_start = false, cut_end = false;  // an end at a z = 0 cut
    long long n_elements() const { return last_node - first_node; }
    double length() const;  // `_vec3.norm3(points[1] - points[0])`
};

struct SeamSite {  // `_Site`
    Node at;
    // "virtual": a node on a PHANTOM wire (`_phantom_tags`, momwire#1139), a
    // deck port with no momwire port under it: piece -1, end "none", no
    // column, no weight.
    enum Spelling { gap, node, virtual_ } spelling = gap;
    long long piece = -1;
    enum EndKind { at_start, at_end, interior, none } end = at_start;
    double sign = 0, arclength = 0;
    bool contact = false;
    bool crossing = false;  // a series source on a crossing node (momwire#1282)
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
    std::map<long long, SplitElement> split_elements;  // by element index
    long long n_columns() const { return static_cast<long long>(feeds.size() + gaps.size()); }
};

// One `TL` / `NT` card resolved onto sites (`_Card`): the six fields in NEC's
// order (`NetworkCard.payload`: a TL's signed Z0, its RESOLVED length and its
// two shunt admittances; an NT's Y11, Y12, Y22), the two sites it attaches
// to, and what its NETWORK DATA row prints (`LineRow` / `NetworkRow`).
struct SeamCard {
    enum Kind { tl, nt } kind = tl;
    double payload[6] = {0, 0, 0, 0, 0, 0};
    long long site_a = -1, site_b = -1;
    Int tag_from, tag_to;
    long long segment_from = 0, segment_to = 0;  // signed: negative for a "-1" node
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
    std::vector<SeamCard> cards;     // `_cards`: lines, then networks, deck order
    std::vector<Int> phantom_tags;   // `_phantom_tags`, deck order
    // Built up to the front, whose fill is not in C++ yet (the buried fill,
    // momwire#1512): kept on a declined route so a gate can compare the
    // preparation, never solved.
    bool unsolved = false;
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
// nec5 path up to each structure's solve. `open_end`: the text is a
// stream's piece (`split_structures`' open_end, momwire#1512).
SeamResult razor_seam(const std::string &latin1_text, bool open_end = false);

// One structure's deck (already read) through `_serve.refusal`: "" when in
// scope, else the sentence; `declined` names a decline instead.
std::string scope_refusal(const Deck &deck, std::string &declined);

// `_shell._internal_error_reason`: the last line of defence's sentence.
std::string internal_error_reason(const std::string &error_kind, const std::string &message);

}  // namespace nec5
}  // namespace mw
