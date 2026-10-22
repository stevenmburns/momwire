// The NEC-5 deck READER in C++: card text in, the parsed deck and the
// reader's refusals out, with no Python anywhere. It is the hardened copy of
// momwire's own reader -- `deck/_cards.py` (`parse_card`), `deck/_nec5.py`
// (`_Nec5Parser`, `parse_nec5`, `parse_nec5_shared`, `split_structures`) and
// the two text-level steps of the EZNEC seam that come before a solve,
// `eznec/_serve.py`'s `split_runs` and `eznec/_shell.py`'s `_answer` /
// `render` up to the call to `serve`. Python stays the reference: every
// deck the reader accepts is the same record field for field (floats bit for
// bit, ints as Python's exact ints), and every deck it refuses is refused
// with the same exception type and the same message, raised at the same
// card, gated by `tests/test_nec5_deck_core.py` and
// `scripts/core_split/deck_compare.py`.
//
//     std::string text = ...;              // the deck file's bytes, LATIN-1
//                                          // (what `_shell.read_deck` decodes)
//     mw::nec5::Deck deck = mw::nec5::parse_nec5(text);   // throws ParseError
//     mw::nec5::StreamRead s = mw::nec5::read_stream(text);   // the seam's
//                                          // front: NX structures, XQ runs,
//                                          // and which refusal it prints
//
// TEXT. Input is latin-1 bytes, one byte per code point, because that is how
// the seam reads a deck. Every string this reader RETURNS (comments, the CE
// text, the source text, messages) is UTF-8: a message can carry characters
// outside latin-1 because Python upper-cases a mnemonic before it names it
// (`µ` becomes U+039C). Python's own `str` rules are reproduced for the 256
// latin-1 code points (`isspace`, `isalpha`, `isdigit`, `upper`,
// `splitlines`), each pinned against Python by a test. No message carries
// Python-specific text (a repr, an interpreter error): the Python reader's
// messages were made plain first, and these are those.
//
// NUMBERS. A field is Python's `float()` of the token after `D`->`E`
// (underscores, `inf`/`nan` spellings and all; correctly rounded, so the
// same bits), and an integer field is `int(round(f))`: round half to even,
// an exact integer however large (`Int`), OverflowError / ValueError (in the
// reader's words, `Card.i`) on an infinite / NaN field exactly where
// Python's is raised.
#pragma once

#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace mw {
namespace nec5 {

// Python's int, as exact as Python's: a field is an integral double of any
// magnitude, and the reader adds and subtracts them (segment boundaries,
// `segment - 1`). Base 1e9 limbs, little-endian; zero is empty and positive.
class Int {
  public:
    Int() = default;
    Int(long long v) { set_i64(v); }  // NOLINT: implicit, like Python's
    static Int from_integral_double(double d);

    bool fits_i64() const;
    long long i64() const;  // throws std::out_of_range when it does not fit
    std::string str() const;
    bool is_zero() const { return mag_.empty(); }
    bool negative() const { return neg_; }

    Int abs() const;
    Int operator+(const Int &o) const;
    Int operator-(const Int &o) const;
    int compare(const Int &o) const;
    bool operator==(const Int &o) const { return compare(o) == 0; }
    bool operator!=(const Int &o) const { return compare(o) != 0; }
    bool operator<(const Int &o) const { return compare(o) < 0; }
    bool operator<=(const Int &o) const { return compare(o) <= 0; }
    bool operator>(const Int &o) const { return compare(o) > 0; }
    bool operator>=(const Int &o) const { return compare(o) >= 0; }

  private:
    void set_i64(long long v);
    void trim();
    bool neg_ = false;
    std::vector<std::uint32_t> mag_;
};

// The exception types the Python reader can raise, by their Python names.
enum class ErrorKind { deck_error, overflow_error, value_error, index_error, serve_refusal };
const char *error_kind_name(ErrorKind k);  // "DeckError", "OverflowError", ...

struct ParseError : std::runtime_error {
    ErrorKind kind;
    // The whole message: what() stops at a NUL, which a refusal quoting a
    // binary file's line carries (the nec4 reader's dump reads this).
    std::string message;
    ParseError(ErrorKind k, const std::string &utf8_message)
        : std::runtime_error(utf8_message), kind(k), message(utf8_message) {}
};

// --- the model (`deck/_nec5.py`'s dataclasses, field for field) ------------

struct Node {  // Nec5Node: a favored wire and a node, never canonicalised
    Int tag;
    Int node;
};

struct Wire {  // Nec5Wire
    Int tag;
    Int segment_count;
    double end1[3] = {0, 0, 0};
    double end2[3] = {0, 0, 0};
    double radius = 0;
};

struct Ground {
    // Nec5FreeSpace | Nec5PerfectGround | Nec5SommerfeldGround |
    // Nec5MininecGround; the fields a kind does not carry stay default.
    enum Kind { free_space, perfect, sommerfeld, mininec } kind = free_space;
    double eps_r = 0, sigma = 0;
    double mu_re = 1, mu_im = 0;
    Int spelling;  // sommerfeld: 0 (GN 0) or 2 (GN 2)
    Int ix;        // mininec: GD's first field
    bool sigma_sets_im_epsc = false;  // mininec
};

struct Source {  // Nec5Source
    Int kind;
    Node at;
    double drive_re = 0, drive_im = 0;
    Int end_code;
    Int printed_location;
};

struct LoadSpec {  // deck.model.LoadSpec
    std::string kind;  // "fixed", "series", "parallel"
    double r = 0, l = 0, c = 0, x = 0;
};

struct Load {  // Nec5Load
    Node at;
    LoadSpec spec;
    Int end_code;
    Int printed_location;
};

struct Conductivity {  // Nec5Conductivity
    Int tag, segment_from, segment_thru;
    double sigma = 0;
};

struct DistributedRLC {  // _wire_loading.DistributedRLC
    std::string kind;  // "series", "parallel"
    double r = 0, l = 0, c = 0;
};

struct DistributedRLCCard {  // Nec5DistributedRLC
    Int tag, segment_from, segment_thru;
    DistributedRLC spec;
};

// One entry of `load_cards` (deck order, every kind): which list, which index.
struct LoadCardRef {
    enum Kind { load, conductivity, distributed_rlc } kind;
    std::size_t index;
};

struct TransmissionLine {  // Nec5TransmissionLine
    Node end_a, end_b;
    double z0 = 0, length_m = 0;
    double shunt[4] = {0, 0, 0, 0};
};

struct Network {  // Nec5Network: three complex entries, (re, im) each
    Node end_a, end_b;
    double y11[2] = {0, 0}, y12[2] = {0, 0}, y22[2] = {0, 0};
};

struct Request {
    // Nec5FarFieldRequest (RP) | Nec5NearFieldRequest (NE / NH) |
    // Nec5ExecuteRequest (XQ); the fields a kind does not carry stay default.
    enum Kind { far_field, near_field, execute } kind = execute;
    // far_field
    Int n_theta, n_phi, xnda;
    double theta0_deg = 0, phi0_deg = 0, d_theta_deg = 0, d_phi_deg = 0, range_m = 0;
    Int mode;
    // near_field
    Int coordinates;
    Int counts[3];
    double origin[3] = {0, 0, 0}, step[3] = {0, 0, 0};
    bool magnetic = false;
    // execute
    Int flag;
};

struct Deck {  // Nec5Deck, in its field order
    std::vector<std::string> comments;  // UTF-8, each a CM card's raw[2:]
    std::string ce_text;                // UTF-8
    std::string source_text;            // UTF-8: the text the deck was read from
    std::vector<Wire> wires;
    Int ge_flag;
    Int ge_second = -1;
    bool geometry_file = false;  // GE's third field nonzero (momwire#1488)
    Ground ground;
    std::vector<Source> sources;
    std::vector<Load> loads;
    std::vector<Conductivity> conductivities;
    std::vector<LoadCardRef> load_cards;
    std::vector<std::pair<Int, double>> wire_conductivity;  // insertion order
    std::vector<DistributedRLCCard> distributed_rlc;
    std::vector<std::pair<Int, DistributedRLC>> wire_distributed_rlc;
    std::vector<TransmissionLine> transmission_lines;
    std::vector<Network> networks;
    bool has_frequency = false;
    double frequency_mhz = 0;
    // A stepping FR's frequencies, in order, the first being frequency_mhz
    // (momwire#1508); empty for a one-frequency deck. `frequencies_too_many`:
    // the count was past what the reader materialises (the list then holds
    // the first frequency only, and the seam declines the deck).
    std::vector<double> frequencies_mhz;
    bool frequencies_too_many = false;
    std::vector<Request> requests;
    bool has_pq = false;
    Int pq;

    const Wire *wire(const Int &tag) const;  // Nec5Deck.wire
};

// --- the reader --------------------------------------------------------------

// `parse_nec5(text)`: one structure's deck, or ParseError.
Deck parse_nec5(const std::string &latin1_text);

// `parse_nec5_shared(texts)`: each text's outcome in place.
struct DeckOutcome {
    bool ok = false;
    Deck deck;
    ErrorKind kind = ErrorKind::deck_error;
    std::string message;  // UTF-8
};
std::vector<DeckOutcome> parse_nec5_shared(const std::vector<std::string> &latin1_texts);

// `split_structures(text, open_end=)`: false is Python's None (no NX before
// EN). `open_end`: the streaming mode's (momwire#1512) -- an NX ending the
// text opens a structure still to come, not an empty one to refuse.
struct Structure {
    std::string body;  // latin-1, each line "\n"-terminated
    bool has_terminator = false;
    std::string terminator;  // latin-1, the NX / EN line stripped
    std::string deck_text() const;
    std::string echo_text() const;
    bool ends_on_nx() const;  // `ends_on_nx(structure)`
};
bool split_structures(const std::string &latin1_text, std::vector<Structure> &out,
                      bool open_end = false);

// `next_structure(text, final=)`: true when a complete NX or EN line ends
// the stream's first structure; `end` is the offset just past it and
// `is_en` says which. `final`: the text is all there will be.
bool next_structure(const std::string &latin1_text, bool final, std::size_t &end, bool &is_en);

// `_serve.split_runs(text)`: false is None; throws ParseError(serve_refusal).
bool split_runs(const std::string &latin1_text, std::vector<std::string> &out);

// `_shell._answer(text, echo_text=...)` up to the solve: the deck it would
// serve (and the per-run decks of a multi-run deck), or the refusal it
// prints (`refused`, the message), or the exception that escapes it
// (`raised`, which `_shell.main`'s last line of defence then reports).
// `stepped`: momwire#1508's stepping FR -- `deck` carries the frequencies,
// and `multi_run` says whether the deck also split into XQ runs (which
// `_stepping_refusal` then refuses).
// One run of a sequence deck (`_serve.SequenceGroup`): its single-run deck
// as parsed (the outcome held, raised where `_answer_sequence` parses it),
// the post-GE card indices it echoes, whether it starts a frequency loop and
// whether it writes its own sources.
struct SequenceGroup {
    DeckOutcome parsed;
    std::vector<std::size_t> cards;
    bool full = false;
    bool sources = false;
};

struct Answer {
    // sequence: a deck read as a program (`_serve.split_sequence`, momwire
    // 0.79.2; served by the C++ route since momwire#1602)
    enum Kind { refused, raised, single, multi, stepped, sequence } kind = refused;
    bool multi_run = false;  // stepped
    ErrorKind error_kind = ErrorKind::deck_error;
    std::string message;  // UTF-8; refused / raised
    Deck deck;            // single / multi; source_text is the echo text
    std::vector<Deck> runs;  // multi
    std::vector<SequenceGroup> groups;  // sequence: its runs, in order
    std::vector<std::size_t> sequence_tail;  // sequence: the cards echoed after the last
};
Answer answer(const std::string &latin1_text, const std::string *latin1_echo_text);

// `_shell.render(text)` for the nec5 dialect, up to the solve: the NX
// structures (if chained) and each one's answer, in order, stopping after
// the first that refuses or raises (`_render_chain`'s break).
struct StreamRead {
    bool chained = false;
    std::vector<Structure> structures;
    std::vector<Answer> answers;
};
StreamRead read_stream(const std::string &latin1_text, bool open_end = false);

// --- the comparison format ---------------------------------------------------
// One line per field, in the reference's own spelling
// (`scripts/core_split/deck_compare.py` writes the Python side): floats as
// their 64 bits in hex, ints in decimal, strings as UTF-8 hex.
std::string dump_parse(const std::string &latin1_text);   // parse_nec5's outcome
std::string dump_stream(const std::string &latin1_text);  // read_stream
std::string dump_deck(const Deck &deck, const std::string &path);

// The latin-1 `str` rules and number spellings, exposed for the tests that
// pin them against Python one code point / token at a time. `op` is one of
// "strip", "split", "splitlines", "upper", "isalpha", "isdigit", "isspace",
// "float", "round", "g", "+g", "card"; the result is the dump-format
// rendering.
std::string text_probe(const std::string &op, const std::string &latin1_arg);

// Python's `format(x, "g")` (printf's `%g` in the C locale, with Python's
// spellings of NaN and the infinities): the `{x:g}` the seam's sentences use.
std::string format_g(double x);

// --- the pieces the printout writer shares ------------------------------------
// (`_nec5_razor_printout_core.cpp` re-reads the deck's card images for its
// echo, as `_printout._post_ge_cards` does, and lays out the comment box.)

// `deck/_cards.Card`: a mnemonic (UTF-8, upper-cased) and its numeric fields.
struct Card {
    std::string mnemonic;
    std::vector<double> values;
    std::string raw;  // latin-1
    bool has_trailer = false;
    std::string trailer;  // latin-1

    double f(std::size_t k) const;  // a field the card did not write reads 0
    Int i(std::size_t k) const;     // rounded half to even; throws ParseError
                                    // (OverflowError / ValueError) where Python raises
    std::string text() const;       // raw[2:]
};

// `parse_card(line)`: false for a blank line; throws ParseError as Python raises.
bool read_card(const std::string &latin1_line, Card &card);

// Python's `str` rules over latin-1 (one byte, one code point).
std::vector<std::string> latin1_splitlines(const std::string &latin1);
std::vector<std::string> latin1_split(const std::string &latin1);
std::string latin1_upper(const std::string &latin1, std::size_t *length = nullptr);  // UTF-8
std::string latin1_to_utf8(const std::string &latin1);
bool latin1_isspace(unsigned char c);

// snprintf(fmt, x) in the C locale, whatever the process's locale is.
std::string format_c_locale(const char *fmt, double x);

// The rest of the reader's toolkit, for the nec2 / nec4 reader
// (`_nec4_deck_core.cpp`), which shares this one's Python text rules:
// Python's `float(token)` (false where it raises), `str.strip()`, and the
// comparison format's spellings (a double's 64 bits, a UTF-8 string's hex).
bool latin1_float(const std::string &latin1_token, double &value);
std::string latin1_strip(const std::string &latin1);
std::string dump_bits(double x);
std::string dump_hex(const std::string &utf8);

}  // namespace nec5
}  // namespace mw
