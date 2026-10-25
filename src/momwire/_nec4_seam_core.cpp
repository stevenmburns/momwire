// The NEC-4.2 slot's seam front in C++ (see `_nec4_seam_core.h`). Each
// function names the Python it copies. The order of the checks, and so which
// refusal a deck hears first, is Python's.
#include "_nec4_seam_core.h"

#include <algorithm>
#include <set>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace mw {
namespace nec4 {

using nec5::Card;
using nec5::ErrorKind;
using nec5::ParseError;

namespace {

typedef long long ll;

// `_nec4.Nec4Refusal`: a deck this slot does not print.
struct Refusal : std::runtime_error {
    std::string message;
    explicit Refusal(const std::string &m) : std::runtime_error(m), message(m) {}
};

[[noreturn]] void refuse(const std::string &message) { throw Refusal(message); }

// `_nec4._unmeasured`
[[noreturn]] void unmeasured(const std::string &what) {
    refuse(what + " is not served by this engine's NEC-4.2 printout: no captured NEC-4.2 printout "
                  "shows how it prints");
}

std::string S(const Int &v) { return v.str(); }

bool one_of(const std::string &m, std::initializer_list<const char *> set) {
    for (const char *s : set)
        if (m == s) return true;
    return false;
}

bool is_execute(const std::string &m) { return one_of(m, {"RP", "XQ", "NE", "NH"}); }

// `_printout._post_ge_cards`: every card after GE, re-tokenized from the
// deck's own lines, through EN.
std::vector<Card> post_ge_cards(const std::string &text) {
    std::vector<Card> cards;
    bool seen_ge = false;
    for (const std::string &line : nec5::latin1_splitlines(text)) {
        Card card;
        if (!nec5::read_card(line, card)) continue;
        if (!seen_ge) {
            seen_ge = card.mnemonic == "GE";
            continue;
        }
        cards.push_back(card);
        if (card.mnemonic == "EN") break;
    }
    return cards;
}

// `portal._portal.parse_deck(text, dialect="nec4")`
PortalDeck parse_deck(const std::string &text) {
    PortalDeck d;
    d.model = parse(text, Dialect::nec4);
    const Model &model = d.model;
    std::vector<Card> loads;
    std::vector<PortalSource> sources;
    bool has_mp = false, has_pt = false, has_pq = false;
    Int mp[2];
    CardFields4 pt, pq;
    bool sources_stale = false;
    std::size_t executes = 0;
    for (const std::string &line : nec5::latin1_splitlines(text)) {
        Card card;
        if (!nec5::read_card(line, card)) continue;
        const std::string &m = card.mnemonic;
        if (m == "NX" || m == "EN") break;
        if (m == "CM" || m == "CE") {
            d.comments.push_back(nec5::latin1_to_utf8(card.text()));
            continue;
        }
        if (one_of(m, {"GW", "GA", "GH", "GM", "GS", "GX", "GR", "GE"})) {
            d.geometry.push_back(card);
            continue;
        }
        d.data_cards.push_back(card);
        if (m == "LD") {
            if (card.i(0) == Int(-1)) {
                loads.clear();
            } else {
                loads.push_back(card);
            }
        } else if (m == "MP") {
            has_mp = true;
            mp[0] = card.i(0);
            mp[1] = card.i(1);
        } else if (m == "PT") {
            has_pt = true;
            pt = CardFields4{card.i(0), card.i(1), card.i(2), card.i(3)};
        } else if (m == "PQ") {
            has_pq = true;
            pq = CardFields4{card.i(0), card.i(1), card.i(2), card.i(3)};
        } else if (m == "EX") {
            if (sources_stale) {
                sources.clear();
                sources_stale = false;
            }
            sources.push_back(PortalSource{card.i(1), card.i(2), {card.f(4), card.f(5)}});
        } else if (is_execute(m)) {
            std::size_t k = executes++;
            bool armed = k < model.groups.size() && model.group_ran[k];
            d.group_ran.push_back(armed);
            PortalGroup g;
            if (armed) {
                const ExecuteGroup &a = model.groups[k];
                g.sources = sources;
                g.freqs_mhz = a.frequencies;
                g.refilled = a.refilled;
                if (m != "XQ") {
                    g.has_report = true;
                    g.report = card;
                }
                g.has_mp = has_mp;
                g.mp_processors = mp[0];
                g.mp_block_size = mp[1];
                g.has_pt = has_pt;
                g.pt = pt;
                g.has_pq = has_pq;
                g.pq = pq;
                g.ek = a.extended_kernel;
                g.refilled_partial = a.refilled_partial;
                g.environment = a.environment;
                g.loads = loads;
                sources_stale = true;
            }
            d.groups.push_back(g);
        }
    }
    d.loads = loads;
    d.ground = model.ground;
    d.has_second_medium = model.has_second_medium;
    d.second_medium = model.second_medium;
    d.ground_plane_flag = model.ground_plane_flag;
    d.ground_plane_interpolates = model.ground_plane_interpolates;
    d.quiet = model.quiet;
    d.has_reduced_field = model.has_reduced_field;
    d.reduced_field = model.reduced_field;
    d.structure = geometry_of(d.geometry);
    return d;
}

// `_nec4._further_tables`
void further_tables(const std::string &text, const PortalDeck &deck, StructureSeam &out) {
    std::vector<std::pair<ll, Card>> numbered;
    std::vector<Card> cards = post_ge_cards(text);
    for (std::size_t i = 0; i < cards.size(); i++)
        if (is_execute(cards[i].mnemonic)) numbered.push_back({static_cast<ll>(i + 1), cards[i]});
    if (numbered.size() != deck.group_ran.size())
        refuse(std::to_string(numbered.size()) + " execute cards against " +
               std::to_string(deck.group_ran.size()) +
               " execute runs: this engine cannot place the card echo");
    ll run = -1;
    for (std::size_t index = 0; index < numbered.size(); index++) {
        if (deck.group_ran[index]) {
            run = static_cast<ll>(index);
            out.echo_at.push_back(numbered[index].first);
            continue;
        }
        out.echo_at.push_back(-1);
        const Card &card = numbered[index].second;
        if (one_of(card.mnemonic, {"RP", "NE", "NH"})) {
            auto it = std::find_if(out.further.begin(), out.further.end(),
                                   [&](const auto &kv) { return kv.first == run; });
            if (it == out.further.end()) {
                out.further.push_back({run, {}});
                it = out.further.end() - 1;
            }
            it->second.push_back({numbered[index].first, card});
        }
    }
}

// `_nec4._served_table`
void served_table(const Card &card) {
    if (card.mnemonic != "RP") return;
    if (!card.i(0).is_zero()) unmeasured("RP mode " + S(card.i(0)));
    Int xnda = card.i(3);
    bool served = false;
    for (int x : {1000, 1001, 1002, 1003, 1010, 1011, 1012, 1013}) served = served || xnda == Int(x);
    if (!served) unmeasured("RP with XNDA " + S(xnda));
    if (card.f(8) < 0.0) unmeasured("RP at a negative range");
}

// `_nec4._served_geometry`
void served_geometry(const PortalDeck &deck) {
    std::vector<std::string> seen;
    auto saw = [&](const char *m) { return std::find(seen.begin(), seen.end(), m) != seen.end(); };
    for (const Card &card : deck.geometry) {
        const std::string &m = card.mnemonic;
        if (!one_of(m, {"GW", "GA", "GH", "GS", "GM", "GX", "GE"})) unmeasured("the " + m + " geometry card");
        if (m == "GS" && (!card.i(0).is_zero() || !card.i(1).is_zero()))
            unmeasured("GS scaling a range of tags");
        // several GM cards, and wire cards between them, are measured (p92-p96,
        // momwire#1694); GS stays at most one, and last
        if (m == "GS" && saw("GS")) unmeasured("a second GS card");
        if (one_of(m, {"GW", "GA", "GH"}) && saw("GS")) unmeasured("a " + m + " card after a GS card");
        if (m == "GM" && saw("GS")) unmeasured("a GM card after a GS card");
        if (m == "GX" && saw("GS")) unmeasured("a GX card after a GS card");
        seen.push_back(m);
    }
}

// `_nec4._a_new_frequency`
bool a_new_frequency(const std::vector<Card> &cards, ll previous, ll number) {
    bool fr = false, all = true;
    ll n = static_cast<ll>(cards.size());
    ll lo = std::max<ll>(0, std::min(previous, n)), hi = std::max<ll>(0, std::min(number, n));
    for (ll k = lo; k < hi; k++) {
        const std::string &m = cards[static_cast<std::size_t>(k)].mnemonic;
        fr = fr || m == "FR";
        // `_serve._LATER_SEQUENCE_CARDS`: PT between the runs is print control (momwire#1696)
        all = all && one_of(m, {"EX", "RP", "NE", "NH", "PQ", "XQ", "FR", "PT"});
    }
    return fr && all;
}

// `_nec4._served_shape`
void served_shape(const std::string &text, const PortalDeck &deck, StructureSeam &out) {
    served_geometry(deck);
    for (const Card &card : deck.data_cards) {
        if (!one_of(card.mnemonic, {"GN", "GD", "FR", "EX", "LD", "TL", "NT", "PQ", "PT", "RP", "NE", "NH",
                                    "XQ", "EN", "EK"}))
            unmeasured("the " + card.mnemonic + " card");
        if (card.mnemonic == "GD" && card.i(0) != Int(2)) unmeasured("GD " + S(card.i(0)));
        if (card.mnemonic == "PT" && card.i(0) != Int(-1)) unmeasured("PT " + S(card.i(0)));
    }
    std::vector<ll> runs;
    for (std::size_t i = 0; i < deck.group_ran.size(); i++)
        if (deck.group_ran[i]) runs.push_back(static_cast<ll>(i));
    auto group = [&](ll i) -> const PortalGroup & { return deck.groups[static_cast<std::size_t>(i)]; };
    auto echo = [&](ll i) { return out.echo_at[static_cast<std::size_t>(i)]; };
    if (runs.empty()) unmeasured("a deck with no execute card");
    if (runs.size() > 1) {
        for (std::size_t k = 0; k + 1 < runs.size(); k++)
            if (group(runs[k]).freqs_mhz.size() != 1) unmeasured("several execute cards over a frequency sweep");
        std::vector<Card> cards = post_ge_cards(text);
        for (std::size_t k = 1; k < runs.size(); k++) {
            const PortalGroup &g = group(runs[k]);
            if (!(g.refilled || g.refilled_partial)) continue;
            if (!a_new_frequency(cards, echo(runs[k - 1]), echo(runs[k])))
                unmeasured("a later execute card after a card that refills the matrix");
        }
    }
    std::vector<ll> gd_at;
    {
        std::vector<Card> cards = post_ge_cards(text);
        for (std::size_t i = 0; i < cards.size(); i++)
            if (cards[i].mnemonic == "GD") gd_at.push_back(static_cast<ll>(i + 1));
    }
    for (ll index : runs) {
        const PortalGroup &g = group(index);
        if (g.environment.ground.kind == Ground::finite_fast)
            unmeasured("the GN 0 reflection-coefficient ground");
        if (g.pq_restricted()) unmeasured("a PQ card restricted to a segment range");
        if (g.has_report) served_table(g.report);
        for (const auto &kv : out.further) {
            if (kv.first != index) continue;
            for (const auto &nc : kv.second) {
                const std::string &mn = nc.second.mnemonic;
                if (g.freqs_mhz.size() != 1)
                    unmeasured("a further " + mn + " card after a frequency sweep's execute card");
                for (ll at : gd_at)
                    if (echo(index) < at && at < nc.first)
                        unmeasured("a further " + mn + " card after a GD card read since the run it follows");
                served_table(nc.second);
            }
        }
    }
    out.plan = runs;
}

// `_nec4.render_structure` / `solve`, up to `DeckSolver`.
StructureSeam structure_seam(const std::string &text, const std::string &echo, bool closing) {
    StructureSeam s;
    s.deck_text = text;
    s.echo_text = echo;
    s.closing = closing;
    try {
        try {
            s.deck = parse_deck(text);
        } catch (const ParseError &e) {
            if (e.kind == ErrorKind::deck_error) refuse(e.message);
            throw;
        }
        further_tables(text, s.deck, s);
        served_shape(text, s.deck, s);
        s.kind = StructureSeam::served;
    } catch (const Refusal &e) {
        s.kind = StructureSeam::refused;
        s.message = e.message;
    } catch (const ParseError &e) {
        s.kind = StructureSeam::raised;
        s.error_kind = e.kind;
        s.message = e.message;
    } catch (const Declined &e) {
        s.kind = StructureSeam::declined;
        s.message = e.what();
    }
    return s;
}

// --- the comparison format ---------------------------------------------------

struct Out {
    std::string s;
    void line(const std::string &l) { s += l + "\n"; }
    void cls(const std::string &p, const char *name) { line(p + " : " + name); }
    void seq(const std::string &p, const char *kind, std::size_t n) {
        line(p + " : " + kind + " " + std::to_string(n));
    }
    void none(const std::string &p) { line(p + " = None"); }
    void b(const std::string &p, bool v) { line(p + " = bool " + (v ? "True" : "False")); }
    void i(const std::string &p, const Int &v) { line(p + " = int " + v.str()); }
    void i(const std::string &p, ll v) { line(p + " = int " + std::to_string(v)); }
    void f(const std::string &p, double v) { line(p + " = float " + nec5::dump_bits(v)); }
    void c(const std::string &p, std::complex<double> v) {
        line(p + " = complex " + nec5::dump_bits(v.real()) + " " + nec5::dump_bits(v.imag()));
    }
    void str(const std::string &p, const std::string &utf8) { line(p + " = str " + nec5::dump_hex(utf8)); }
};

std::string at(const std::string &p, std::size_t k) { return p + "[" + std::to_string(k) + "]"; }

void dump_card(Out &o, const std::string &p, const Card &c) {
    o.cls(p, "Card");
    o.str(p + ".mnemonic", c.mnemonic);
    o.seq(p + ".values", "tuple", c.values.size());
    for (std::size_t k = 0; k < c.values.size(); k++) o.f(at(p + ".values", k), c.values[k]);
    o.str(p + ".raw", nec5::latin1_to_utf8(c.raw));
    if (c.has_trailer) {
        o.str(p + ".trailer", nec5::latin1_to_utf8(c.trailer));
    } else {
        o.none(p + ".trailer");
    }
}

void dump_cards(Out &o, const std::string &p, const std::vector<Card> &cards) {
    o.seq(p, "tuple", cards.size());
    for (std::size_t k = 0; k < cards.size(); k++) dump_card(o, at(p, k), cards[k]);
}

void dump_fields4(Out &o, const std::string &p, const char *name, bool has, const CardFields4 &v) {
    if (!has) return o.none(p);
    o.cls(p, name);
    o.i(p + ".flag", v.flag);
    o.i(p + ".tag", v.tag);
    o.i(p + ".first", v.first);
    o.i(p + ".last", v.last);
}

void dump_model_ground(Out &o, const std::string &p, const Ground &g) {
    switch (g.kind) {
        case Ground::none:
            return o.none(p);
        case Ground::pec:
            return o.str(p, "pec");
        default:
            o.seq(p, "tuple", 3);
            o.str(at(p, 0), g.kind == Ground::finite_fast ? "finite-fast" : "finite");
            o.f(at(p, 1), g.eps_r);
            o.f(at(p, 2), g.sigma);
    }
}

void dump_portal(Out &o, const std::string &p, const PortalDeck &d) {
    o.cls(p, "PortalDeck");
    o.seq(p + ".comments", "tuple", d.comments.size());
    for (std::size_t k = 0; k < d.comments.size(); k++) o.str(at(p + ".comments", k), d.comments[k]);
    dump_cards(o, p + ".geometry", d.geometry);
    dump_cards(o, p + ".data_cards", d.data_cards);
    o.seq(p + ".groups", "tuple", d.groups.size());
    for (std::size_t k = 0; k < d.groups.size(); k++) {
        std::string q = at(p + ".groups", k);
        if (!d.group_ran[k]) {
            o.none(q);
            continue;
        }
        const PortalGroup &g = d.groups[k];
        o.cls(q, "ExecuteGroup");
        o.seq(q + ".sources", "tuple", g.sources.size());
        for (std::size_t j = 0; j < g.sources.size(); j++) {
            std::string r = at(q + ".sources", j);
            o.seq(r, "tuple", 3);
            o.i(at(r, 0), g.sources[j].tag);
            o.i(at(r, 1), g.sources[j].seg);
            o.c(at(r, 2), g.sources[j].volts);
        }
        o.seq(q + ".freqs_mhz", "tuple", g.freqs_mhz.size());
        for (std::size_t j = 0; j < g.freqs_mhz.size(); j++) o.f(at(q + ".freqs_mhz", j), g.freqs_mhz[j]);
        o.b(q + ".refilled", g.refilled);
        if (g.has_report) {
            dump_card(o, q + ".report", g.report);
        } else {
            o.none(q + ".report");
        }
        if (g.has_mp) {
            o.cls(q + ".mp", "Multiprocessing");
            o.i(q + ".mp.processors", g.mp_processors);
            o.i(q + ".mp.block_size", g.mp_block_size);
        } else {
            o.none(q + ".mp");
        }
        dump_fields4(o, q + ".pt", "PrintControl", g.has_pt, g.pt);
        dump_fields4(o, q + ".pq", "ChargePrintControl", g.has_pq, g.pq);
        o.b(q + ".ek", g.ek);
        o.b(q + ".refilled_partial", g.refilled_partial);
        std::string e = q + ".environment";
        o.cls(e, "Environment");
        dump_model_ground(o, e + ".ground", g.environment.ground);
        o.f(e + ".ground_z", 0.0);
        if (g.environment.has_second_medium) {
            const SecondMedium &s = g.environment.second_medium;
            o.cls(e + ".second_medium", "SecondMedium");
            o.f(e + ".second_medium.eps_r", s.eps_r);
            o.f(e + ".second_medium.sigma", s.sigma);
            o.f(e + ".second_medium.edge_distance", s.edge_distance);
            o.f(e + ".second_medium.height", s.height);
        } else {
            o.none(e + ".second_medium");
        }
        dump_cards(o, q + ".loads", g.loads);
    }
    dump_cards(o, p + ".loads", d.loads);
    o.cls(p + ".ground", "Ground");
    const char *kind = d.ground.kind == Ground::none          ? "free"
                       : d.ground.kind == Ground::pec         ? "pec"
                       : d.ground.kind == Ground::finite_fast ? "refl"
                                                              : "sommerfeld";
    o.str(p + ".ground.kind", kind);
    bool media = d.ground.kind == Ground::finite_fast || d.ground.kind == Ground::finite;
    o.f(p + ".ground.eps_r", media ? d.ground.eps_r : 0.0);
    o.f(p + ".ground.sigma", media ? d.ground.sigma : 0.0);
    if (d.has_second_medium) {
        o.cls(p + ".second_medium", "SecondMedium");
        o.f(p + ".second_medium.eps_r2", d.second_medium.eps_r);
        o.f(p + ".second_medium.sigma2", d.second_medium.sigma);
        o.f(p + ".second_medium.edge_distance", d.second_medium.edge_distance);
        o.f(p + ".second_medium.height", d.second_medium.height);
    } else {
        o.none(p + ".second_medium");
    }
    o.b(p + ".ground_plane_flag", d.ground_plane_flag);
    o.b(p + ".ground_plane_interpolates", d.ground_plane_interpolates);
    o.b(p + ".quiet", d.quiet);
    if (d.has_reduced_field) {
        o.i(p + ".reduced_field", d.reduced_field);
    } else {
        o.none(p + ".reduced_field");
    }
    o.none(p + ".model");
    o.none(p + ".structure");
}

void dump_point_list(Out &o, const std::string &p, const double *v, const char *kind) {
    o.seq(p, kind, 3);
    for (std::size_t k = 0; k < 3; k++) o.f(at(p, k), v[k]);
}

void dump_structure(Out &o, const std::string &q, const Geometry &g) {
    o.seq(q + ".wires", "list", g.wires.size());
    for (std::size_t k = 0; k < g.wires.size(); k++) {
        const GeometryWire &w = g.wires[k];
        std::string r = at(q + ".wires", k);
        o.cls(r, "FlatWire");
        o.i(r + ".tag", w.tag);
        o.i(r + ".n_seg", w.n_seg);
        dump_point_list(o, r + ".p1", w.p1, "list");
        dump_point_list(o, r + ".p2", w.p2, "list");
        o.f(r + ".radius", w.radius);
    }
    o.seq(q + ".pieces", "list", g.pieces.size());
    for (std::size_t k = 0; k < g.pieces.size(); k++) {
        const GeometryPiece &pc = g.pieces[k];
        std::string r = at(q + ".pieces", k);
        o.cls(r, "Piece");
        o.i(r + ".wire", pc.wire);
        o.i(r + ".first_seg", pc.first_seg);
        o.i(r + ".n_seg", pc.n_seg);
        dump_point_list(o, r + ".p1", pc.p1, "tuple");
        dump_point_list(o, r + ".p2", pc.p2, "tuple");
    }
    o.seq(q + ".pieces_of_wire", "list", g.pieces_of_wire.size());
    for (std::size_t k = 0; k < g.pieces_of_wire.size(); k++) {
        const auto &v = g.pieces_of_wire[k];
        o.seq(at(q + ".pieces_of_wire", k), "list", v.size());
        for (std::size_t j = 0; j < v.size(); j++) o.i(at(at(q + ".pieces_of_wire", k), j), v[j]);
    }
    o.b(q + ".ground_plane_flag", g.ground_plane_flag);
    o.b(q + ".ground_plane_interpolates", g.ground_plane_interpolates);
    if (g.has_symmetry) {
        o.cls(q + ".symmetry", "Symmetry");
        o.i(q + ".symmetry.cell_wires", g.cell_wires);
        o.i(q + ".symmetry.cell_segments", g.cell_segments);
    } else {
        o.none(q + ".symmetry");
    }
}

void dump_structure_seam(Out &o, const std::string &p, const StructureSeam &s) {
    o.str(p + ".deck_text", nec5::latin1_to_utf8(s.deck_text));
    o.str(p + ".echo_text", nec5::latin1_to_utf8(s.echo_text));
    o.b(p + ".closing", s.closing);
    switch (s.kind) {
        case StructureSeam::refused:
            o.line(p + " = refused " + nec5::dump_hex(s.message));
            return;
        case StructureSeam::raised:
            o.line(p + " ! " + nec5::error_kind_name(s.error_kind) + " " + nec5::dump_hex(s.message));
            return;
        case StructureSeam::declined:
            o.line(p + " ? declined " + nec5::dump_hex(s.message));
            return;
        case StructureSeam::served:
            break;
    }
    o.line(p + " = served");
    dump_portal(o, p + ".deck", s.deck);
    dump_structure(o, p + ".structure", s.deck.structure);
    o.seq(p + ".echo_at", "list", s.echo_at.size());
    for (std::size_t k = 0; k < s.echo_at.size(); k++) {
        if (s.echo_at[k] < 0) {
            o.none(at(p + ".echo_at", k));
        } else {
            o.i(at(p + ".echo_at", k), s.echo_at[k]);
        }
    }
    o.line(p + ".further : mapping " + std::to_string(s.further.size()));
    for (std::size_t k = 0; k < s.further.size(); k++) {
        std::string r = p + ".further{" + std::to_string(k) + "}";
        o.i(r + ".key", s.further[k].first);
        const auto &v = s.further[k].second;
        o.seq(r + ".value", "list", v.size());
        for (std::size_t j = 0; j < v.size(); j++) {
            std::string t = at(r + ".value", j);
            o.seq(t, "tuple", 2);
            o.i(at(t, 0), v[j].first);
            dump_card(o, at(t, 1), v[j].second);
        }
    }
    o.seq(p + ".plan", "list", s.plan.size());
    for (std::size_t k = 0; k < s.plan.size(); k++) o.i(at(p + ".plan", k), s.plan[k]);
}

}  // namespace

Seam seam(const std::string &text, const std::string &basis) {
    if (basis != "sinusoidal" && basis != "bspline")
        throw Declined("basis '" + basis + "' is not one this copy of the NEC-4.2 seam serves");
    Seam out;
    out.basis = basis;
    std::vector<nec5::Structure> structures;
    if (!nec5::split_structures(text, structures)) {
        out.chained = false;
        out.structures.push_back(structure_seam(text, text, true));
        return out;
    }
    out.chained = true;
    for (std::size_t k = 0; k < structures.size(); k++) {
        bool last = k + 1 == structures.size() && !structures[k].ends_on_nx();
        StructureSeam s = structure_seam(structures[k].deck_text(), structures[k].echo_text(), last);
        bool go_on = s.kind == StructureSeam::served;
        out.structures.push_back(s);
        if (!go_on) break;
    }
    return out;
}

std::string dump_seam(const std::string &text, const std::string &basis) {
    Out o;
    o.line("seam = basis " + nec5::dump_hex(basis));
    Seam s;
    try {
        s = seam(text, basis);
    } catch (const ParseError &e) {
        o.line(std::string("seam ! ") + nec5::error_kind_name(e.kind) + " " + nec5::dump_hex(e.message));
        return o.s;
    } catch (const Declined &e) {
        o.line(std::string("seam ? declined ") + nec5::dump_hex(e.what()));
        return o.s;
    }
    o.b("seam.chained", s.chained);
    // The Python side announces every structure, then stops after the first
    // that does not serve; so does this.
    std::size_t total = s.structures.size();
    if (s.chained) {
        std::vector<nec5::Structure> all;
        nec5::split_structures(text, all);
        total = all.size();
    }
    o.seq("seam.structures", "list", total);
    for (std::size_t k = 0; k < s.structures.size(); k++)
        dump_structure_seam(o, "seam.structures[" + std::to_string(k) + "]", s.structures[k]);
    return o.s;
}

}  // namespace nec4
}  // namespace mw
