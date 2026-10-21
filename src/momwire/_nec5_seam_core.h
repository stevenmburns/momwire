// The NEC-5 seam's STRUCTURE HALF in C++, for any basis: a parsed deck
// (`_nec5_deck_core.h`) in, the mesh, the ports and the solver's arguments
// out, or the refusal the seam prints, with no Python anywhere. It is the
// hardened copy of `eznec/_serve.py`'s path from `serve` / `serve_runs` to
// the solver's constructor:
//
//   * the scope ladder, `refusal` (`_drive_refusal`, `_load_probe_refusal`,
//     `_geometry_refusal`, the request rungs), every sentence word for word;
//   * `_prepare` up to `compute_port_solution`: `structure_of`, `build_mesh`
//     (one piece per GW, the junctions, one site per addressed node,
//     `_site_for`'s refusals), `_assign_columns`, `_check_one_port_per_drive`,
//     `_medium` / `_ground_kwargs` and `_cards`;
//   * `serve_runs`' sharing: a run whose sources sit on the sites of the run
//     prepared before it reuses that preparation.
//
// What a BASIS decides is asked of a `SeamEngine` (below), as the Python seam
// asks it of the solver class and its capability row: the basis' name in the
// sentences, which spelling of a series EMF the mesh takes (`_CUT_SPELLING`,
// `_crossing_spelling`), `_check_basis_can_host`, and `_solver_for` with the
// constructed solver's front half. razor-2p's engine is
// `_nec5_razor_seam_core.h`; another basis plugs in the same way.
//
//     mw::nec5::SeamResult r = mw::nec5::seam(latin1_text, engine);
//     // r.route: served (every structure's runs prepared or shared);
//     // refused (the stream stops on that refusal's printout, after the
//     // structures before it); raised; declined
//
// SCOPE: free space, GN 1, a bare GD (solved over a PERFECT image) or GN 0 /
// GN 2 above the plane; a buried deck's structure half (the plane crossings
// split at z = 0, momwire#1281, and the crossing sites, momwire#1282); TL /
// NT cards resolved onto sites; WIRE loading (LD 5 / LD 2 / LD 3) and lumped
// loads (LD 0 / 1 / 4, a port site whose impedance is the run half's); a
// phantom wire (momwire#1139); finite coordinates, frequency and radii, at
// most `MAX_SEGMENTS` segments in all. The delta-gap spelling only: a basis
// whose mesh CUTS a wire at an addressed node (`_CUT_SPELLING`) declines
// "mesh:cut-spelling" until that spelling is ported. Anything else is
// DECLINED by name (`SeamResult::message`), at the point of the ladder where
// the Python seam would first need the part not ported, so a decline never
// hides a refusal the Python seam would have printed earlier, and a refusal
// printed here is one the Python seam prints at the same rung. The Python
// seam stays the reference; `scripts/core_split/seam_compare.py` and
// `tests/test_nec5_razor_seam_core.py` gate this one against it field for
// field.
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
// `_ground_spec.ground_touch_tol` of the two-point polyline a -> b: 1e-6 of
// its length (what the seam and the solvers ask "in the plane" with).
double ground_touch_tol(const double *a, const double *b);
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

// A piece's end, as `_Mesh.junctions` names it ("start" / "end"); the same
// values and order as `razor::End`, which the razor engine converts to.
enum class End : int8_t { start = 0, end = 1 };

using JunctionEnd = std::pair<long long, End>;  // (piece, which end)

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

// What an engine built for one preparation: its solver's arguments and the
// constructed solver's front half (razor-2p: `RazorPrepared`,
// `_nec5_razor_seam_core.h`). The seam only holds it.
struct EnginePrepared {
    virtual ~EnginePrepared() = default;
};

// One `_prepare`: everything the run half needs from its structure.
struct Prepared {
    // The engine's part (`SeamEngine::solver_arguments`, `::build`): what
    // `_solver_for` passes the solver and what its constructor and
    // `compute_port_solution`'s front half do with that.
    std::shared_ptr<EnginePrepared> engine;
    SeamStructure structure;
    SeamMesh mesh;
    double frequency_mhz = 0, wavelength = 0, omega = 0;
    bool has_medium = false;  // GD: the medium the far field reflects off
    double medium_eps_r = 0, medium_sigma = 0;
    double medium_eps_c[2] = {0, 0};
    std::vector<Node> source_sites;  // the drive's addresses, deck order
    std::vector<SeamCard> cards;     // `_cards`: lines, then networks, deck order
    std::vector<Int> phantom_tags;   // `_phantom_tags`, deck order
    // Built up to the front, whose fill is not in C++ yet (razor-2p's buried
    // fill, momwire#1512): kept on a declined route so a gate can compare the
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
    // sequence: each entry of `runs`, its run (an index into
    // `Answer::sequence`) and the frequency it solved at
    std::vector<std::pair<std::size_t, double>> steps;
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

// The seam's own answers, thrown by `prepare` and by an engine's hooks and
// caught where the Python seam catches them: a `ServeRefusal` (its
// sentence), a decline (why), and an exception Python lets escape at that
// point (not a ServeRefusal: its type name and `str(exc)`), which `_shell`'s
// INTERNAL ERROR printout reports.
struct SeamRefusal {
    std::string message;
};
struct SeamDeclined {
    std::string reason;
};
struct SeamRaised {
    std::string kind, message;
};

// What the seam asks of a basis that is not a hook: the Python seam reads
// each off the solver class or its capability row.
struct SeamBasis {
    // the name the sentences carry ("razor-2p")
    std::string name;
    // `_CUT_SPELLING`: a GW is cut at every node a card addresses inside it,
    // and a K = 2 junction's series EMF is a node gap (`build_mesh`'s
    // `node_gaps=cut`). Not ported: such a basis declines "mesh:cut-spelling"
    // where the Python seam would cut.
    bool cut_spelling = false;
    // `_crossing_spelling(solver_class) == "node"` (a node port AND junction
    // ports): a series EMF on a crossing node is a node gap, else a gap.
    bool crossing_node_spelling = false;
};

// One `_prepare`'s basis half (`_check_basis_can_host`, `_solver_for` and the
// solver's construction), for one basis. The seam calls them in the Python
// seam's order; each may throw `SeamRefusal`, `SeamDeclined` or `SeamRaised`.
class SeamEngine {
  public:
    virtual ~SeamEngine() = default;
    virtual const SeamBasis &basis() const = 0;
    // `_check_basis_can_host(mesh, _ground_kwargs(deck, medium), basis,
    // solver_class)`, after the medium and before `_cards`.
    virtual void check_can_host(const Deck &d, const Prepared &p, bool has_ground) const = 0;
    // `_solver_for`'s arguments, after `_cards`: sets `p.engine`.
    virtual void solver_arguments(const Deck &d, Prepared &p, bool has_ground) const = 0;
    // The solver's constructor and `compute_port_solution`'s front half on
    // `p.engine`. A NotImplementedError from the CONSTRUCTOR is `_prepare`'s
    // refusal (`refused`); any other exception escapes the seam (`raised`).
    struct Built {
        enum Kind { built, declined, refused, raised } kind = built;
        std::string message;     // declined: why; refused / raised: str(exc)
        std::string error_kind;  // refused / raised: the exception's type name
        // declined: the preparation is kept (`Prepared::unsolved`) so a gate
        // can compare it, although it cannot be solved
        bool unsolved = false;
    };
    virtual Built build(Prepared &p) const = 0;
};

// The whole seam, deck text to the solver's arguments: `_shell.render`'s
// nec5 path up to each structure's solve, for `engine`'s basis. `open_end`:
// the text is a stream's piece (`split_structures`' open_end, momwire#1512).
SeamResult seam(const std::string &latin1_text, const SeamEngine &engine, bool open_end = false);

// One structure's deck (already read) through `_serve.refusal`: "" when in
// scope, else the sentence; `declined` names a decline instead.
std::string scope_refusal(const Deck &deck, std::string &declined);

// `_shell._internal_error_reason`: the last line of defence's sentence.
std::string internal_error_reason(const std::string &error_kind, const std::string &message);

}  // namespace nec5
}  // namespace mw
