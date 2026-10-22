// The NEC-4.2 slot's seam FRONT in C++ (momwire#1607 A2): a deck's text in,
// and out either the runs to solve with the printout's view of the cards,
// or the refusal the slot prints, with no Python anywhere. It is the copy of
// what `eznec/_nec4.py`'s `render` does before the solve:
// - `basis_refusal`;
// - the NX chain split (`deck/_nec5.split_structures`, the NEC-5 reader's,
//   already in C++);
// - `render_structure` / `solve` up to `DeckSolver`, i.e. `parse_deck` (the
//   reader of `_nec4_deck_core.h`, then the portal's card walk: comments,
//   geometry and data cards, one group per execute card with its sources,
//   report card, MP / PT / PQ state and loads), `_further_tables` and
//   `_served_shape`.
//
// Python stays the reference: `scripts/core_split/nec4_seam_compare.py`
// drives the slot's real control flow with the solve stubbed, and
// `tests/test_nec4_seam_core.py` holds the dumps equal deck by deck.
//
// What comes after it (the build: `DeckSolver` and `build_solver`'s mapping
// onto a solver's arguments) is momwire#1607 A2b.
#pragma once

#include "_nec4_deck_core.h"

#include <complex>
#include <string>
#include <utility>
#include <vector>

namespace mw {
namespace nec4 {

struct PortalSource {  // (tag, seg, voltage) as the card wrote it
    Int tag, seg;
    std::complex<double> volts;
};

struct CardFields4 {  // PrintControl / ChargePrintControl: flag, tag, first, last
    Int flag, tag, first, last;
};

struct PortalGroup {  // portal._portal.ExecuteGroup
    std::vector<PortalSource> sources;
    std::vector<double> freqs_mhz;
    bool refilled = true;
    bool has_report = false;
    nec5::Card report;
    bool has_mp = false;
    Int mp_processors, mp_block_size;
    bool has_pt = false;
    CardFields4 pt;
    bool has_pq = false;
    CardFields4 pq;
    bool ek = false;
    bool refilled_partial = false;
    Environment environment;
    std::vector<nec5::Card> loads;

    bool pq_restricted() const {  // ChargePrintControl.restricted
        return has_pq && pq.flag != Int(-1) && (!pq.first.is_zero() || !pq.last.is_zero());
    }
};

struct PortalDeck {  // portal._portal.PortalDeck
    std::vector<std::string> comments;  // UTF-8
    std::vector<nec5::Card> geometry;
    std::vector<nec5::Card> data_cards;
    std::vector<bool> group_ran;    // false: an idle execute card (None)
    std::vector<PortalGroup> groups;  // parallel to group_ran
    std::vector<nec5::Card> loads;
    Ground ground;  // its printout view: free / pec / refl / sommerfeld
    bool has_second_medium = false;
    SecondMedium second_medium;
    bool ground_plane_flag = false;
    bool ground_plane_interpolates = false;
    bool quiet = false;
    bool has_reduced_field = false;
    Int reduced_field;
    Model model;
    Geometry structure;
};

// One structure of the deck through `render_structure`, up to the solve.
struct StructureSeam {
    std::string deck_text;  // latin-1: what is solved
    std::string echo_text;  // latin-1: what the printout echoes
    bool closing = true;    // the last structure carries RUN TIME
    enum Kind { served, refused, raised, declined } kind = served;
    nec5::ErrorKind error_kind = nec5::ErrorKind::deck_error;  // raised
    std::string message;  // UTF-8: refused / raised / declined
    PortalDeck deck;      // served
    std::vector<long long> echo_at;  // -1 for an idle execute card (None)
    // run index -> every (echo number, card) asked of it after it ran
    std::vector<std::pair<long long, std::vector<std::pair<long long, nec5::Card>>>> further;
    std::vector<long long> plan;  // the group indices that run
};

struct Seam {
    std::string basis;
    bool basis_refused = false;  // (never, for a basis this copy serves)
    std::string refusal;         // UTF-8
    bool chained = false;
    // In order, through the first that refuses, raises or declines.
    std::vector<StructureSeam> structures;
};

// `_nec4.render(text, basis=...)` up to the solve. `basis` must be one this
// copy serves ("sinusoidal", "bspline"); any other name is Declined.
Seam seam(const std::string &latin1_text, const std::string &basis);

// The comparison format (`scripts/core_split/nec4_seam_compare.py`).
std::string dump_seam(const std::string &latin1_text, const std::string &basis);

}  // namespace nec4
}  // namespace mw
