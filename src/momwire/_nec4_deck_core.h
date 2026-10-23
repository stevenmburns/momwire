// The nec2 / nec4 deck READER in C++: card text in, the dialect-neutral deck
// model (`deck/model.py`'s `DeckModel`) and the reader's refusals out, with
// no Python anywhere. It is the copy of momwire's own reader for the
// segment-centre dialects -- `deck/_nec2.py` (`_Nec2Parser`, `parse_nec2`),
// `deck/_nec2_geometry.py` (`build_geometry`, the connection passes, the
// `(tag, segment)` resolver) and the NEC-4.2 slot's deltas in
// `deck/_nec4.py` (`_Nec4Parser`, `parse_nec4`) -- the first unit of the
// standalone NEC-4.2-dialect executables (momwire#1607 A1). Python stays the
// reference: every deck the reader accepts is the same model field for field
// (floats bit for bit, ints as Python's exact ints), and every deck it
// refuses is refused with the same exception type and the same message,
// raised at the same card, gated by `tests/test_nec4_deck_core.py` and
// `scripts/core_split/nec4_deck_compare.py`.
//
//     std::string text = ...;   // the deck's bytes, LATIN-1
//     mw::nec4::Model m = mw::nec4::parse(text, mw::nec4::Dialect::nec4);
//                               // throws nec5::ParseError as Python raises,
//                               // or Declined (below)
//
// The card tokenizer, the latin-1 `str` rules and the number spellings are
// the NEC-5 reader's (`_nec5_deck_core.h`), which already follow Python's.
//
// DECLINED. A deck this copy does not read is DECLINED, never misread: a
// card with a non-finite field (Python's object-identity rules make a NaN
// compare equal to itself inside its own tuple, which no value type can
// follow), a geometry that overflows to a non-finite coordinate or length, an
// integer past 64 bits where the reader counts with it, and sizes past what
// this copy materialises (10 million segments, a million wires, frequencies
// or range steps). Python answers some of those and loops for a long time on
// others; either way the comparison counts a decline apart and never
// compares it.
//
// PYTHON VERSION. Two of Python's float rules here are CPython 3.12's:
// `math.dist` (its compensated `vector_norm`) and `sum()` of floats
// (Neumaier summation, new in 3.12). Both are what the release bundles run.
#pragma once

#include "_nec5_deck_core.h"

#include <complex>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace mw {
namespace nec4 {

using nec5::Int;

enum class Dialect { nec2, nec4 };

struct Declined : std::runtime_error {
    explicit Declined(const std::string &why) : std::runtime_error(why) {}
};

// --- the model (`deck/model.py`'s dataclasses, field for field) ------------

struct LoadSpec {  // model.LoadSpec
    std::string kind;  // "series", "parallel", "fixed"
    double r = 0, l = 0, c = 0, x = 0;
    bool operator==(const LoadSpec &o) const {
        return kind == o.kind && r == o.r && l == o.l && c == o.c && x == o.x;
    }
    bool operator!=(const LoadSpec &o) const { return !(*this == o); }
};

struct DistributedRLC {  // _wire_loading.DistributedRLC
    std::string kind;  // "series" (LD 2), "parallel" (LD 3)
    double r = 0, l = 0, c = 0;
    bool operator==(const DistributedRLC &o) const {
        return kind == o.kind && r == o.r && l == o.l && c == o.c;
    }
    bool operator!=(const DistributedRLC &o) const { return !(*this == o); }
};

struct WireMaterial {  // model.WireMaterial (insulation_loss_tangent is None)
    bool has_conductivity = false;
    double conductivity = 0;
    bool has_insulation = false;  // insulation_radius and _eps_r together
    double insulation_radius = 0, insulation_eps_r = 0;
    bool has_distributed = false;
    DistributedRLC distributed;
};

struct DeckWire {  // model.DeckWire: this dialect writes two vertices
    double vertices[2][3] = {{0, 0, 0}, {0, 0, 0}};
    double radius = 0;
    long long edge_elements = 0;  // the one edge's element count
    bool has_material = false;
    WireMaterial material;
};

struct SecondMedium {
    double eps_r = 0, sigma = 0, edge_distance = 0, height = 0;
};

struct Ground {  // None, "pec", ("finite-fast" | "finite", eps_r, sigma)
    enum Kind { none, pec, finite_fast, finite } kind = none;
    double eps_r = 0, sigma = 0;
};

struct Environment {  // ground_z is 0.0 in this dialect
    Ground ground;
    bool has_second_medium = false;
    SecondMedium second_medium;
};

struct Request {  // None, FarFieldRequest, NearFieldRequest
    enum Kind { none, far_field, near_field } kind = none;
    // far_field
    Int mode, n_theta, n_phi;
    double theta0_deg = 0, phi0_deg = 0, d_theta_deg = 0, d_phi_deg = 0, range_m = 0;
    // near_field
    bool magnetic = false;
    Int counts[3];
    double origin[3] = {0, 0, 0}, step[3] = {0, 0, 0};
    Int coordinates;
};

struct PrintControl {
    bool suppressed = false;
    bool has_elements = false;
    std::vector<std::pair<long long, long long>> elements;  // (wire, element)
};

struct ExecuteGroup {
    std::vector<double> frequencies;
    std::vector<std::complex<double>> voltages;
    Request request;
    bool has_print_control = false;
    PrintControl print_control;
    bool extended_kernel = false;
    bool has_multiprocessing = false;
    Int multiprocessing[2];
    bool refilled = true;
    bool refilled_partial = false;
    Environment environment;
    std::vector<long long> loads;  // indices into Model::loads (never None here)
};

struct Feed {  // (wire, arclength, volts -- or amperes for a current feed)
    long long wire = 0;
    double arclength = 0;
    std::complex<double> volts;
};

struct Load {  // (wire, arclength, spec)
    long long wire = 0;
    double arclength = 0;
    LoadSpec spec;
};

struct NetworkCard {
    std::string kind;  // "TL", "NT"
    long long wire_a = 0, wire_b = 0;
    double arclength_a = 0, arclength_b = 0;
    Int address_a[2], address_b[2];  // (tag, segment), as written
    double payload[6] = {0, 0, 0, 0, 0, 0};
    long long first_group = 0;
};

struct Model {  // model.DeckModel, in its field order
    std::vector<DeckWire> wires;
    std::vector<Feed> feeds;
    std::vector<Load> loads;
    Ground ground;
    bool has_second_medium = false;
    SecondMedium second_medium;
    bool ground_plane_flag = false;
    bool ground_plane_interpolates = false;
    std::vector<bool> group_ran;  // false: an execute card that ran nothing (None)
    std::vector<ExecuteGroup> groups;  // parallel to group_ran
    std::vector<std::string> comments;  // UTF-8, each a CM / CE card's raw[2:]
    bool quiet = false;
    bool has_reduced_field = false;
    Int reduced_field;
    std::vector<NetworkCard> networks;
    std::vector<long long> current_feeds;  // nec4's EX 6 ports
    bool extended_kernel_default = false;
};

// `deck/_nec2_geometry.py`'s `Nec2Structure` after `build_geometry(cards)`,
// as plain data (its `cuts` left out: they are the pieces' cut points): the
// flat NEC wires after every transform and the snapping, the pieces they
// shatter into, and the symmetric cell still in force. The NEC-4.2 seam's
// `PortalDeck.structure` (`_nec4_seam_core.h`).
struct GeometryWire {  // FlatWire
    Int tag;
    long long n_seg = 0;
    double p1[3] = {0, 0, 0}, p2[3] = {0, 0, 0};
    double radius = 0;
};
struct GeometryPiece {  // Piece
    long long wire = 0, first_seg = 0, n_seg = 0;
    double p1[3] = {0, 0, 0}, p2[3] = {0, 0, 0};
};
struct Geometry {
    std::vector<GeometryWire> wires;
    std::vector<GeometryPiece> pieces;
    std::vector<std::vector<long long>> pieces_of_wire;
    bool ground_plane_flag = false;
    bool ground_plane_interpolates = false;
    bool has_symmetry = false;  // Symmetry(cell_wires, cell_segments) or None
    long long cell_wires = 0, cell_segments = 0;
};
// `build_geometry(cards)` over a deck's geometry cards (GW GA GH GM GS GX GR GE, in
// order); throws as Python raises, or Declined.
Geometry geometry_of(const std::vector<nec5::Card> &geometry_cards);

// `momwire.deck.parse(text, dialect=...)`: throws nec5::ParseError as Python
// raises (DeckError, ValueError, OverflowError, IndexError), or Declined.
Model parse(const std::string &latin1_text, Dialect dialect);

// The comparison format (`scripts/core_split/deck_compare.py`'s `dump` of
// the DeckModel under the path "parse", or "parse ! <Type> <hex>" for a
// refusal); a decline is "parse ? declined <hex>".
std::string dump_parse(const std::string &latin1_text, Dialect dialect);
std::string dump_model(const Model &model, const std::string &path);

// Python's `repr(x)` of a float and `math.dist` of two 3-points (CPython
// 3.12), for the tests that pin them against Python.
std::string py_repr(double x);
double py_dist(const double a[3], const double b[3]);

// `sum(math.dist(a, b) for a, b in zip(v, v[1:]))` over a GH card's
// `helix_vertices`: the wire table's TOTAL LENGTH OF WIRE IN THE SPIRAL.
double helix_wire_length(const nec5::Card &card);

}  // namespace nec4
}  // namespace mw
