// The NEC-5 deck reader -- see `_nec5_deck_core.h`. Each function below
// names the Python it is a copy of; the messages are that Python's, byte for
// byte, and the ORDER in which fields are read is too, because an integer
// field is converted when it is read and an infinite or NaN one raises there.
#include "_nec5_deck_core.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <map>

#if defined(_MSC_VER)
// The extension's MSVC flags are /fp:fast, which licenses the compiler to
// assume no NaN; the reader compares NaN fields as Python does.
#pragma float_control(precise, on, push)
#include <locale.h>
#else
#include <locale.h>
#if defined(__APPLE__)
#include <xlocale.h>
#endif
#endif

namespace mw {
namespace nec5 {

// ============================================================================
// Int
// ============================================================================

static const std::uint32_t kBase = 1000000000u;

void Int::set_i64(long long v) {
    mag_.clear();
    neg_ = v < 0;
    // magnitude without overflowing on LLONG_MIN
    unsigned long long m = neg_ ? 0ull - static_cast<unsigned long long>(v)
                                : static_cast<unsigned long long>(v);
    while (m) {
        mag_.push_back(static_cast<std::uint32_t>(m % kBase));
        m /= kBase;
    }
    if (mag_.empty()) neg_ = false;
}

void Int::trim() {
    while (!mag_.empty() && mag_.back() == 0) mag_.pop_back();
    if (mag_.empty()) neg_ = false;
}

Int Int::from_integral_double(double d) {
    if (d > -9.2e18 && d < 9.2e18) return Int(static_cast<long long>(d));
    // |d| >= 2^63: an integer mantissa times a power of two, doubled exactly.
    int e = 0;
    double m = std::frexp(std::fabs(d), &e);  // |d| = m * 2^e, m in [0.5, 1)
    Int out(static_cast<long long>(std::ldexp(m, 53)));
    for (int k = 0; k < e - 53; ++k) out = out + out;
    if (d < 0) out = Int(0) - out;
    return out;
}

bool Int::fits_i64() const {
    static const Int lo(std::numeric_limits<long long>::min());
    static const Int hi(std::numeric_limits<long long>::max());
    return *this >= lo && *this <= hi;
}

long long Int::i64() const {
    if (!fits_i64()) throw std::out_of_range("Int does not fit in 64 bits: " + str());
    unsigned long long m = 0;
    for (std::size_t k = mag_.size(); k-- > 0;) m = m * kBase + mag_[k];
    return neg_ ? static_cast<long long>(0ull - m) : static_cast<long long>(m);
}

std::string Int::str() const {
    if (mag_.empty()) return "0";
    std::string s = neg_ ? "-" : "";
    s += std::to_string(mag_.back());
    char buf[16];
    for (std::size_t k = mag_.size() - 1; k-- > 0;) {
        std::snprintf(buf, sizeof buf, "%09u", static_cast<unsigned>(mag_[k]));
        s += buf;
    }
    return s;
}

Int Int::abs() const {
    Int out = *this;
    out.neg_ = false;
    return out;
}

static int cmp_mag(const std::vector<std::uint32_t> &a, const std::vector<std::uint32_t> &b) {
    if (a.size() != b.size()) return a.size() < b.size() ? -1 : 1;
    for (std::size_t k = a.size(); k-- > 0;)
        if (a[k] != b[k]) return a[k] < b[k] ? -1 : 1;
    return 0;
}

static std::vector<std::uint32_t> add_mag(const std::vector<std::uint32_t> &a,
                                          const std::vector<std::uint32_t> &b) {
    std::vector<std::uint32_t> out;
    std::uint32_t carry = 0;
    for (std::size_t k = 0; k < std::max(a.size(), b.size()) || carry; ++k) {
        std::uint64_t s = carry;
        if (k < a.size()) s += a[k];
        if (k < b.size()) s += b[k];
        out.push_back(static_cast<std::uint32_t>(s % kBase));
        carry = static_cast<std::uint32_t>(s / kBase);
    }
    return out;
}

// |a| >= |b| required
static std::vector<std::uint32_t> sub_mag(const std::vector<std::uint32_t> &a,
                                          const std::vector<std::uint32_t> &b) {
    std::vector<std::uint32_t> out(a);
    std::int64_t borrow = 0;
    for (std::size_t k = 0; k < out.size(); ++k) {
        std::int64_t d = static_cast<std::int64_t>(out[k]) - borrow - (k < b.size() ? b[k] : 0);
        borrow = d < 0;
        if (d < 0) d += kBase;
        out[k] = static_cast<std::uint32_t>(d);
    }
    return out;
}

int Int::compare(const Int &o) const {
    if (neg_ != o.neg_) return neg_ ? -1 : 1;
    int c = cmp_mag(mag_, o.mag_);
    return neg_ ? -c : c;
}

Int Int::operator+(const Int &o) const {
    Int out;
    if (neg_ == o.neg_) {
        out.mag_ = add_mag(mag_, o.mag_);
        out.neg_ = neg_;
    } else if (cmp_mag(mag_, o.mag_) >= 0) {
        out.mag_ = sub_mag(mag_, o.mag_);
        out.neg_ = neg_;
    } else {
        out.mag_ = sub_mag(o.mag_, mag_);
        out.neg_ = o.neg_;
    }
    out.trim();
    return out;
}

Int Int::operator-(const Int &o) const {
    Int neg = o;
    if (!neg.mag_.empty()) neg.neg_ = !neg.neg_;
    return *this + neg;
}

const char *error_kind_name(ErrorKind k) {
    switch (k) {
    case ErrorKind::deck_error: return "DeckError";
    case ErrorKind::overflow_error: return "OverflowError";
    case ErrorKind::value_error: return "ValueError";
    case ErrorKind::index_error: return "IndexError";
    case ErrorKind::serve_refusal: return "ServeRefusal";
    }
    return "?";
}

// ============================================================================
// Python's str rules over latin-1 (one byte = one code point)
// ============================================================================

namespace {

typedef unsigned char uc;

bool py_isspace(uc c) {
    return (c >= 0x09 && c <= 0x0d) || (c >= 0x1c && c <= 0x20) || c == 0x85 || c == 0xa0;
}

bool py_isalpha(uc c) {
    return (c >= 'A' && c <= 'Z') || (c >= 'a' && c <= 'z') || c == 0xaa || c == 0xb5 ||
           c == 0xba || (c >= 0xc0 && c <= 0xd6) || (c >= 0xd8 && c <= 0xf6) || c >= 0xf8;
}

bool py_isdigit(uc c) { return (c >= '0' && c <= '9') || c == 0xb2 || c == 0xb3 || c == 0xb9; }

// str.splitlines' boundaries (\r\n is one)
bool py_linebreak(uc c) {
    return c == '\n' || c == '\r' || c == 0x0b || c == 0x0c || c == 0x1c || c == 0x1d ||
           c == 0x1e || c == 0x85;
}

void put_utf8(std::string &out, std::uint32_t cp) {
    if (cp < 0x80) {
        out += static_cast<char>(cp);
    } else if (cp < 0x800) {
        out += static_cast<char>(0xc0 | (cp >> 6));
        out += static_cast<char>(0x80 | (cp & 0x3f));
    } else {
        out += static_cast<char>(0xe0 | (cp >> 12));
        out += static_cast<char>(0x80 | ((cp >> 6) & 0x3f));
        out += static_cast<char>(0x80 | (cp & 0x3f));
    }
}

std::string utf8(const std::string &latin1) {
    std::string out;
    out.reserve(latin1.size());
    for (char ch : latin1) put_utf8(out, static_cast<uc>(ch));
    return out;
}

// str.upper(), as UTF-8 (it leaves latin-1: µ -> U+039C, ÿ -> U+0178,
// ß -> "SS"), and the length Python sees.
std::string py_upper(const std::string &latin1, std::size_t *length = nullptr) {
    std::string out;
    std::size_t n = 0;
    for (char ch : latin1) {
        uc c = static_cast<uc>(ch);
        if (c >= 'a' && c <= 'z') {
            put_utf8(out, c - 0x20);
        } else if (c == 0xb5) {
            put_utf8(out, 0x039c);
        } else if (c == 0xdf) {
            out += "SS";
            ++n;
        } else if (c == 0xff) {
            put_utf8(out, 0x0178);
        } else if (c >= 0xe0 && c <= 0xfe && c != 0xf7) {
            put_utf8(out, c - 0x20);
        } else {
            put_utf8(out, c);
        }
        ++n;
    }
    if (length) *length = n;
    return out;
}

// The first `n` code points (Python's `s[:n]`).
std::string head(const std::string &s, std::size_t n) { return s.substr(0, std::min(n, s.size())); }

std::string py_strip(const std::string &s) {
    std::size_t a = 0, b = s.size();
    while (a < b && py_isspace(static_cast<uc>(s[a]))) ++a;
    while (b > a && py_isspace(static_cast<uc>(s[b - 1]))) --b;
    return s.substr(a, b - a);
}

std::vector<std::string> py_split(const std::string &s) {
    std::vector<std::string> out;
    std::size_t k = 0;
    while (k < s.size()) {
        while (k < s.size() && py_isspace(static_cast<uc>(s[k]))) ++k;
        if (k >= s.size()) break;
        std::size_t start = k;
        while (k < s.size() && !py_isspace(static_cast<uc>(s[k]))) ++k;
        out.push_back(s.substr(start, k - start));
    }
    return out;
}

std::vector<std::string> py_splitlines(const std::string &s) {
    std::vector<std::string> out;
    std::size_t k = 0, start = 0;
    while (k < s.size()) {
        uc c = static_cast<uc>(s[k]);
        if (py_linebreak(c)) {
            out.push_back(s.substr(start, k - start));
            if (c == '\r' && k + 1 < s.size() && s[k + 1] == '\n') ++k;
            start = ++k;
        } else {
            ++k;
        }
    }
    if (start < s.size()) out.push_back(s.substr(start));
    return out;
}

std::string replace_char(std::string s, char from, char to) {
    std::replace(s.begin(), s.end(), from, to);
    return s;
}

// --- numbers -----------------------------------------------------------------

double strtod_c(const char *s) {
#if defined(_MSC_VER)
    static _locale_t c = _create_locale(LC_NUMERIC, "C");
    return _strtod_l(s, nullptr, c);
#else
    static locale_t c = newlocale(LC_NUMERIC_MASK, "C", (locale_t)0);
    return strtod_l(s, nullptr, c);
#endif
}

// snprintf(fmt, x) in the C locale, whatever the process's locale is.
// Any length: a `%f` of a large double runs to hundreds of digits.
std::string format_c(const char *fmt, double x) {
    std::vector<char> buf(64);
    for (;;) {
#if defined(_MSC_VER)
        static _locale_t c = _create_locale(LC_NUMERIC, "C");
        const int n = _snprintf_l(buf.data(), buf.size(), fmt, c, x);
        if (n >= 0 && static_cast<std::size_t>(n) < buf.size()) return buf.data();
        buf.resize(n >= 0 ? static_cast<std::size_t>(n) + 1 : 2 * buf.size());
#else
        static locale_t c = newlocale(LC_NUMERIC_MASK, "C", (locale_t)0);
        locale_t old = uselocale(c);
        const int n = std::snprintf(buf.data(), buf.size(), fmt, x);
        uselocale(old);
        if (n >= 0 && static_cast<std::size_t>(n) < buf.size()) return buf.data();
        buf.resize(n >= 0 ? static_cast<std::size_t>(n) + 1 : 2 * buf.size());
#endif
    }
}

bool digitpart(const std::string &s, std::size_t &k, std::string &out) {
    // digit (["_"] digit)*
    if (k >= s.size() || !(s[k] >= '0' && s[k] <= '9')) return false;
    out += s[k++];
    while (k < s.size()) {
        if (s[k] >= '0' && s[k] <= '9') {
            out += s[k++];
        } else if (s[k] == '_' && k + 1 < s.size() && s[k + 1] >= '0' && s[k + 1] <= '9') {
            out += s[k + 1];
            k += 2;
        } else {
            break;
        }
    }
    return true;
}

bool iequals(const std::string &a, const char *b) {
    std::size_t n = std::strlen(b);
    if (a.size() != n) return false;
    for (std::size_t k = 0; k < n; ++k) {
        char x = a[k];
        if (x >= 'A' && x <= 'Z') x = static_cast<char>(x + 32);
        if (x != b[k]) return false;
    }
    return true;
}

// Python's float(token) (the token has no whitespace: it came from split()).
bool py_float(const std::string &token, double &value) {
    std::size_t k = 0;
    bool negative = false;
    if (k < token.size() && (token[k] == '+' || token[k] == '-')) negative = token[k++] == '-';
    std::string rest = token.substr(k);
    if (iequals(rest, "inf") || iequals(rest, "infinity")) {
        value = negative ? -std::numeric_limits<double>::infinity()
                         : std::numeric_limits<double>::infinity();
        return true;
    }
    if (iequals(rest, "nan")) {
        value = std::copysign(std::numeric_limits<double>::quiet_NaN(), negative ? -1.0 : 1.0);
        return true;
    }
    std::string clean;
    bool int_part = digitpart(token, k, clean);
    bool frac_part = false;
    if (k < token.size() && token[k] == '.') {
        clean += token[k++];
        frac_part = digitpart(token, k, clean);
    }
    if (!int_part && !frac_part) return false;
    if (k < token.size() && (token[k] == 'e' || token[k] == 'E')) {
        clean += 'e';
        ++k;
        if (k < token.size() && (token[k] == '+' || token[k] == '-')) clean += token[k++];
        if (!digitpart(token, k, clean)) return false;
    }
    if (k != token.size()) return false;
    // The grammar above admitted plain decimal only; strtod in the C locale
    // rounds it correctly (glibc, Apple libc and the UCRT alike), overflow
    // to inf and underflow included.
    double v = strtod_c(clean.c_str());
    value = negative ? -v : v;
    return true;
}

// int(round(x)) for a finite x: half to even, exact.
Int py_round(double x) { return Int::from_integral_double(std::nearbyint(x)); }

// format(x, "g") / format(x, "+g")
std::string py_g(double x, bool plus = false) {
    if (std::isnan(x)) return plus ? "+nan" : "nan";
    if (std::isinf(x)) return x > 0 ? (plus ? "+inf" : "inf") : "-inf";
    std::string s = format_c("%g", x);
    if (plus && !std::signbit(x)) s = "+" + s;
    return s;
}

std::string hex_bits(double x) {
    std::uint64_t u;
    std::memcpy(&u, &x, sizeof u);
    char buf[24];
    std::snprintf(buf, sizeof buf, "%016llx", static_cast<unsigned long long>(u));
    return buf;
}

std::string hex_bytes(const std::string &s) {
    static const char *digits = "0123456789abcdef";
    std::string out;
    out.reserve(2 * s.size());
    for (char ch : s) {
        uc c = static_cast<uc>(ch);
        out += digits[c >> 4];
        out += digits[c & 15];
    }
    return out;
}

[[noreturn]] void deck_error(const std::string &utf8_message) {
    throw ParseError(ErrorKind::deck_error, utf8_message);
}

// ============================================================================
// Card lines (`deck/_cards.py`)
// ============================================================================

const char *kFusedFieldStart = "0123456789.+-";

bool fused_start(char c) { return c != '\0' && std::strchr(kFusedFieldStart, c) != nullptr; }

// `parse_card(line)`: false for a blank line.
bool parse_card(const std::string &line, Card &card) {
    std::string stripped = py_strip(line);
    if (stripped.empty()) return false;
    std::vector<std::string> tokens = py_split(replace_char(stripped, ',', ' '));
    if (tokens.empty())
        throw ParseError(ErrorKind::index_error,
                         "card line has no mnemonic, only separators: " + utf8(stripped));
    const std::string &h = tokens[0];
    if (h.size() > 2 && py_isalpha(static_cast<uc>(h[0])) && py_isalpha(static_cast<uc>(h[1])) &&
        fused_start(h[2])) {
        std::string a = h.substr(0, 2), b = h.substr(2);
        tokens[0] = a;
        tokens.insert(tokens.begin() + 1, b);
    }
    std::size_t length = 0;
    std::string mnemonic = py_upper(tokens[0], &length);
    bool alpha = !tokens[0].empty();
    for (char ch : tokens[0]) alpha = alpha && py_isalpha(static_cast<uc>(ch));
    if (length != 2 || !alpha)
        deck_error("CARD'S MNEMONIC CODE TOO SHORT OR MISSING: " + utf8(stripped));
    std::string raw = line;
    while (!raw.empty() && raw.back() == '\n') raw.pop_back();
    card = Card();
    card.mnemonic = mnemonic;
    card.raw = raw;
    if (mnemonic == "CM" || mnemonic == "CE") return true;
    std::vector<std::string> fields(tokens.begin() + 1, tokens.end());
    double scratch;
    if (mnemonic == "GN" && !fields.empty() && !py_float(replace_char(replace_char(fields.back(), 'D', 'E'), 'd', 'e'), scratch)) {
        card.has_trailer = true;
        card.trailer = fields.back();
        fields.pop_back();
    }
    for (const std::string &token : fields) {
        double v;
        if (!py_float(replace_char(replace_char(token, 'D', 'E'), 'd', 'e'), v))
            deck_error("NON-NUMERICAL CHARACTER IN FIELD: " + utf8(token) + " on " +
                       utf8(stripped));
        card.values.push_back(v);
    }
    return true;
}

// ============================================================================
// The parser (`deck/_nec5.py`'s `_Nec5Parser`)
// ============================================================================

const char *const kVocabulary[] = {"CM", "CE", "GW", "GE", "GN", "GD", "EX", "LD", "TL",
                                   "NT", "FR", "PQ", "RP", "XQ", "NE", "NH", "EN"};

const std::map<std::string, std::string> &refused_by_name() {
    static const std::map<std::string, std::string> table = {
        {"GM", "GM (geometry move) is not part of this engine's nec5 dialect, whose "
               "geometry is GW alone - EZNEC resolves its transforms before writing a deck"},
        {"GX", "GX (symmetry reflection) is not part of this engine's nec5 dialect, "
               "whose geometry is GW alone"},
        {"GR", "GR (cylindrical symmetry) is not part of this engine's nec5 dialect, "
               "whose geometry is GW alone"},
        {"GS", "GS (structure scale) is not part of this engine's nec5 dialect, whose "
               "geometry is GW alone"},
        {"GA", "GA (wire arc) is not part of this engine's nec5 dialect, whose "
               "geometry is GW alone"},
        {"GH", "GH (helix) is not part of this engine's nec5 dialect, whose geometry "
               "is GW alone"},
        {"GC", "GC (tapered wire continuation) is not part of this engine's nec5 "
               "dialect"},
        {"GF", "GF (numerical Green's function) is not part of this engine's nec5 "
               "dialect"},
        {"SM", "SM (multiple-patch surface) is not part of this engine's nec5 dialect, "
               "which models wires only"},
        {"SC", "SC is not a NEC-5 command (it is the NEC-2/NEC-4 patch continuation); "
               "this deck is NEC-2 dialect"},
        {"CP", "CP (coupling request) is not part of this engine's nec5 dialect"},
        {"PL", "PL (plot request) is not part of this engine's nec5 dialect"},
        {"WG", "WG (NGF write request) is not part of this engine's nec5 dialect"},
        {"ZO", "ZO (impedance normalisation) is not part of this engine's nec5 dialect"},
        {"KH", "KH (interaction approximation limit) is not part of this engine's nec5 "
               "dialect"},
        {"EK", "EK (extended thin-wire kernel) is not part of this engine's nec5 "
               "dialect - EZNEC emits no kernel card"},
        {"IS", "IS (insulated sheath) is not part of this engine's nec5 dialect"},
        {"PT", "PT (element-current print control) is not part of this engine's nec5 "
               "dialect; PQ is the only print-control card EZNEC emits"},
        {"MP", "MP (multiprocessing hint) is not part of this engine's nec5 dialect"},
        {"SY", "SY (4nec2 symbolic variables) is not part of this engine's nec5 dialect"},
        {"NX", "NX (next structure) ends one structure and starts the next; "
               "parse_nec5 reads one structure, so split a chained stream with "
               "split_structures first"},
    };
    return table;
}

int min_fields(const std::string &mnemonic) {
    // `_MIN_FIELDS`: a short card reads its missing trailing fields as 0, as
    // the NEC-5 engine does (momwire 0.79.2); GE and GW keep their floors
    static const std::map<std::string, int> table = {{"GE", 1}, {"GW", 9}};
    auto it = table.find(mnemonic);
    return it == table.end() ? -1 : it->second;
}

const char *kSpSphere =
    "SP (sphere) is not part of this engine's nec5 dialect, which models wires only";
const char *kSpPatch =
    "SP in its NEC-2/NEC-4 surface-patch form is not legal NEC-5 syntax "
    "(NEC-5's SP is a sphere); this deck is NEC-2 dialect";

std::vector<std::string> sp_field_tokens(const Card &card) {
    std::vector<std::string> tokens = py_split(replace_char(py_strip(card.raw), ',', ' '));
    if (tokens.empty()) return {};
    const std::string &h = tokens[0];
    if (h.size() > 2 && py_isalpha(static_cast<uc>(h[0])) && py_isalpha(static_cast<uc>(h[1])) &&
        fused_start(h[2])) {
        std::string a = h.substr(0, 2), b = h.substr(2);
        tokens[0] = a;
        tokens.insert(tokens.begin() + 1, b);
    }
    return std::vector<std::string>(tokens.begin() + 1, tokens.end());
}

bool is_int_literal(const std::string &token) {
    std::size_t k = 0;
    while (k < token.size() && (token[k] == '+' || token[k] == '-')) ++k;
    if (k == token.size()) return false;
    for (; k < token.size(); ++k)
        if (!py_isdigit(static_cast<uc>(token[k]))) return false;
    return true;
}

bool sp_is_sphere(const Card &card) {
    std::vector<std::string> tokens = sp_field_tokens(card);
    if (tokens.size() < 8) return false;
    for (int k = 0; k < 4; ++k)
        if (!is_int_literal(tokens[k])) return false;
    if (card.i(1) < Int(1) || card.i(2) < Int(1) || card.f(7) <= 0) return false;
    return true;
}

std::string S(const Int &v) { return v.str(); }

struct Parser {
    std::vector<std::string> comments;  // UTF-8
    bool has_ce = false;
    std::string ce_text;  // UTF-8
    std::vector<Wire> wires;
    std::map<Int, std::size_t> by_tag;
    bool has_ge = false;
    Int ge_flag;
    Int ge_second = -1;
    bool geometry_file = false;
    Ground ground;
    std::vector<Source> sources;
    std::vector<Load> loads;
    std::vector<Conductivity> conductivities;
    std::vector<LoadCardRef> load_cards;
    std::vector<DistributedRLCCard> distributed_rlc;
    std::vector<TransmissionLine> lines;
    std::vector<Network> networks;
    bool has_frequency = false;
    double frequency_mhz = 0;
    std::vector<double> frequencies_mhz;
    bool frequencies_too_many = false;
    std::vector<Request> requests;
    bool has_pq = false;
    Int pq;
    bool saw_en = false;

    const Wire *find(const Int &tag) const {
        auto it = by_tag.find(tag);
        return it == by_tag.end() ? nullptr : &wires[it->second];
    }

    // -- addressing --

    Node address(const Card &card, std::size_t k) {
        const std::string &mn = card.mnemonic;
        Int tag = card.i(k);
        Int written = card.i(k + 1);
        Int node;
        if (written == Int(-1)) {
            node = 0;
        } else if (written.is_zero()) {
            deck_error(mn + " addresses node 0 of tag " + S(tag) +
                       " as a literal 0; this dialect spells node 0 (the wire's end-1 "
                       "boundary) as -1, and a zero connection field is not a node");
        } else if (written.negative()) {
            deck_error(mn + " addresses node " + S(written) + " of tag " + S(tag) +
                       ": -1 is the only negative node spelling in this dialect (it means "
                       "node 0, the wire's end-1 boundary) and no other negative value "
                       "appears in any captured deck");
        } else {
            node = written;
        }
        const Wire *wire = find(tag);
        if (!wire)
            deck_error(mn + " names tag " + S(tag) + ", which no GW card in this deck declares");
        if (node > wire->segment_count)
            deck_error(mn + " addresses node " + S(node) + " of tag " + S(tag) + ", whose " +
                       S(wire->segment_count) + " segments give it nodes 0 to " +
                       S(wire->segment_count) + "; there is no node past the end of a wire");
        return Node{tag, node};
    }

    std::pair<Node, Int> end_coded_address(const Card &card, std::size_t k,
                                           const std::string &field_name,
                                           const std::string &range_note) {
        const std::string &mn = card.mnemonic;
        Int end_code = card.i(k + 2);
        if (end_code.is_zero()) {
            Node at = address(card, k);
            return {at, card.i(k + 1).abs()};
        }
        if (end_code != Int(1) && end_code != Int(2))
            deck_error(mn + " carries " + S(end_code) + " in its " + field_name +
                       " field; this dialect serves 0 (EZNEC's own sign-of-the-"
                       "node-field spelling) and 1 or 2 (antennaknobs' explicit "
                       "end code: " +
                       field_name +
                       " selects the end of the segment "
                       "the middle field names) -- no other value is observed or "
                       "documented" +
                       range_note);
        Int tag = card.i(k);
        Int segment = card.i(k + 1);
        if (segment <= Int(0))
            deck_error(mn + " names segment " + S(segment) + " with an explicit end code (" +
                       S(end_code) +
                       "); the middle field is a positive element number once " + field_name +
                       " selects the end explicitly, and antennaknobs' own writer never emits "
                       "anything else");
        const Wire *wire = find(tag);
        if (!wire)
            deck_error(mn + " names tag " + S(tag) + ", which no GW card in this deck declares");
        if (segment > wire->segment_count)
            deck_error(mn + " addresses segment " + S(segment) + " of tag " + S(tag) +
                       ", which has " + S(wire->segment_count) + " segments");
        Int node = end_code == Int(2) ? segment : segment - Int(1);
        return {Node{tag, node}, segment};
    }

    std::pair<Node, Int> load_address(const Card &card, std::size_t k) {
        return end_coded_address(card, k, "LDTAGT",
                                 ", and a discrete load is one point, never a range "
                                 "(a range is one LD card per element)");
    }

    // -- geometry --

    void gw(const Card &card) {
        Int tag = card.i(0);
        if (find(tag))
            deck_error("GW declares tag " + S(tag) +
                       " a second time; this engine's nec5 dialect gives each wire its own "
                       "tag, because a node address names one wire's segment boundary and a "
                       "tag group has no single wire to count along");
        Int segments = card.i(1);
        if (segments < Int(1))
            deck_error("GW tag " + S(tag) + " asks for " + S(segments) +
                       " segments; a wire needs at least 1");
        Wire w;
        w.tag = tag;
        w.segment_count = segments;
        for (int d = 0; d < 3; ++d) {
            w.end1[d] = card.f(2 + d);
            w.end2[d] = card.f(5 + d);
        }
        w.radius = card.f(8);
        wires.push_back(w);
        by_tag[tag] = wires.size() - 1;
    }

    void ge(const Card &card) {
        ge_flag = card.i(0);
        has_ge = true;
        ge_second = card.i(1);
        // GE's third field (momwire#1488): nonzero asks for a `.GMO` file.
        geometry_file = !card.i(2).is_zero();
    }

    // -- ground --

    void mu(const Card &card, double &re, double &im) {
        re = card.f(6);
        im = card.f(7);
        if (re == 0.0 && im == 0.0) {
            re = 1.0;
            im = 0.0;
        }
        if (re != 1.0 || im != 0.0)
            deck_error(card.mnemonic + " asks for a ground of relative permeability " +
                       py_g(re) + py_g(im, true) +
                       "j; this engine has no magnetic ground and serves mu_r = 1 only "
                       "(EZNEC writes 1.,0. on every ground card it emits)");
    }

    void sommerfeld_file(const Card &card) {
        if (!card.has_trailer || card.trailer == "NOFILE") return;
        deck_error("GN names a Sommerfeld-table file " + utf8(card.trailer) +
                   "; this engine computes its own Sommerfeld solution and serves only "
                   "NOFILE, the name NEC-5 takes to skip writing the table");
    }

    void refuse_file_trailer(const Card &card, const Int &code) {
        if (card.has_trailer)
            deck_error("GN " + S(code) + " carries a trailing token " + utf8(card.trailer) +
                       "; only a finite ground (GN 0 / GN 2) carries the Sommerfeld-table "
                       "file field");
    }

    void gn(const Card &card) {
        Int code = card.i(0);
        if (code == Int(-1)) {
            refuse_file_trailer(card, code);
            ground = Ground();
            return;
        }
        if (code == Int(1)) {
            refuse_file_trailer(card, code);
            ground = Ground();
            ground.kind = Ground::perfect;
            return;
        }
        if (code == Int(0) || code == Int(2)) {
            if (card.values.size() < 5)
                deck_error("GN " + S(code) + " carries " + std::to_string(card.values.size()) +
                           " fields; a finite ground needs its relative permittivity (field 5): "
                           "the NEC-5 engine reads a missing one as 0 and answers NaN");
            sommerfeld_file(card);
            Ground g;
            g.kind = Ground::sommerfeld;
            g.eps_r = card.f(4);
            g.sigma = card.f(5);
            mu(card, g.mu_re, g.mu_im);
            g.spelling = code;
            ground = g;
            return;
        }
        deck_error("GN type " + S(code) +
                   " is not part of this engine's nec5 dialect, which serves GN -1 (free "
                   "space), GN 1 (perfect), and GN 0 / GN 2 (finite ground, Sommerfeld "
                   "solution)");
    }

    void gd(const Card &card) {
        Int ix = card.i(0);
        if (ix == Int(-1)) {
            ground = Ground();
            return;
        }
        double sigma = card.f(5);
        Ground g;
        g.kind = Ground::mininec;
        g.ix = ix;
        g.eps_r = card.f(4);
        g.sigma = sigma;
        mu(card, g.mu_re, g.mu_im);
        g.sigma_sets_im_epsc = sigma < 0.0;
        ground = g;
    }

    // -- excitation, loading, networks --

    void ex(const Card &card) {
        Int kind = card.i(0);
        if (kind != Int(0) && kind != Int(4))
            deck_error("EX type " + S(kind) +
                       " is not part of this engine's nec5 dialect, which serves EX 4 "
                       "(elementary current source) and EX 0 (voltage source) - the two "
                       "EZNEC's source-type setting picks between");
        double re = card.f(4), im = card.f(5);
        if (kind.is_zero() && re == 0.0 && im == 0.0) {
            re = 1.0;
            im = 0.0;
        }
        std::pair<Node, Int> at = end_coded_address(card, 1, "EX's own end-code field", "");
        Source s;
        s.kind = kind;
        s.at = at.first;
        s.drive_re = re;
        s.drive_im = im;
        s.end_code = card.i(3);
        s.printed_location = at.second;
        sources.push_back(s);
    }

    void ld(const Card &card) {
        Int kind = card.i(0);
        if (kind == Int(5)) {
            ld5(card);
            return;
        }
        if (kind == Int(2) || kind == Int(3)) {
            ld23(card);
            return;
        }
        if (kind != Int(0) && kind != Int(1) && kind != Int(4))
            deck_error("LD type " + S(kind) +
                       " is not part of this engine's nec5 dialect, whose loading is LD 0 / "
                       "LD 1 (series / parallel RLC, evaluated to an impedance at the deck's "
                       "own frequency - momwire#1085), LD 2 / LD 3 (per-unit-length RLC - "
                       "momwire#1088), LD 4 (fixed impedance) or LD 5 (wire conductivity)");
        std::pair<Node, Int> at = load_address(card, 1);
        Int end_code = card.i(3);
        LoadSpec spec;
        bool noop;
        if (kind == Int(4)) {
            spec.kind = "fixed";
            spec.r = card.f(4);
            spec.x = card.f(5);
            noop = spec.r == 0.0 && spec.x == 0.0;
        } else {
            spec.kind = kind.is_zero() ? "series" : "parallel";
            spec.r = card.f(4);
            spec.l = card.f(5);
            spec.c = card.f(6);
            noop = spec.r == 0.0 && spec.l == 0.0 && spec.c == 0.0;
        }
        if (noop) return;
        Load load;
        load.at = at.first;
        load.spec = spec;
        load.end_code = end_code;
        load.printed_location = at.second;
        loads.push_back(load);
        load_cards.push_back({LoadCardRef::load, loads.size() - 1});
    }

    void wire_boundaries(std::vector<Int> &starts, std::vector<Int> &ends) const {
        Int at = 1;
        for (const Wire &w : wires) {
            starts.push_back(at);
            at = at + w.segment_count;
            ends.push_back(at - Int(1));
        }
    }

    std::vector<Int> wires_in_absolute_range(const Int &first, const Int &last) const {
        std::vector<Int> starts, ends, out;
        wire_boundaries(starts, ends);
        for (std::size_t k = 0; k < wires.size(); ++k)
            if (first <= starts[k] && ends[k] <= last) out.push_back(wires[k].tag);
        return out;
    }

    std::pair<Int, Int> material_range(const std::string &card, const std::string &what,
                                       const Int &tag, const Int &first, const Int &last) {
        if (tag.is_zero()) {
            Int total = 0;
            for (const Wire &w : wires) total = total + w.segment_count;
            if (first.is_zero() && last.is_zero()) return {Int(1), total};
            std::vector<Int> starts, ends;
            wire_boundaries(starts, ends);
            bool in_starts = std::find(starts.begin(), starts.end(), first) != starts.end();
            bool in_ends = std::find(ends.begin(), ends.end(), last) != ends.end();
            if (in_starts && in_ends && first <= last) return {first, last};
            std::string spans;
            for (std::size_t k = 0; k < starts.size(); ++k) {
                if (k) spans += ", ";
                spans += S(starts[k]) + "-" + S(ends[k]);
            }
            deck_error(card + " addresses tag 0 (whole structure) segments " + S(first) +
                       " to " + S(last) +
                       ", which does not start and end on wire boundaries (this deck's wires "
                       "span " +
                       spans + "); this engine serves the full range 1 to " + S(total) +
                       ", NEC's ordinary ``0,0`` wildcard, or a run of whole wires (EZNEC's "
                       "own spelling stops before its virtual wires, momwire#1096) - a span "
                       "that splits a wire has no captured or written precedent");
        }
        const Wire *wire = find(tag);
        if (!wire)
            deck_error(card + " names tag " + S(tag) + ", which no GW card in this deck declares");
        if (first.is_zero() && last.is_zero()) return {Int(1), wire->segment_count};
        if (first != Int(1) || last != wire->segment_count)
            deck_error(card + " addresses tag " + S(tag) + " segments " + S(first) + " to " +
                       S(last) + "; wire " + S(tag) + " has " + S(wire->segment_count) +
                       " segments and this engine's " + what +
                       " covers whole wires only, the full range 1 to " +
                       S(wire->segment_count) + " or the ``0,0`` wildcard for it");
        return {first, last};
    }

    void ld23(const Card &card) {
        Int kind = card.i(0);
        std::string name = "LD " + S(kind);
        Int tag = card.i(1);
        Int first = card.i(2);
        Int last = card.i(3);
        double r = card.f(4), l = card.f(5), c = card.f(6);
        if (c != 0.0)
            deck_error(name + " asks for a capacitance of " + py_g(c) +
                       " in its per-unit-length RLC, which this engine does not serve: NEC "
                       "scales that field BY the segment length rather than per unit length "
                       "(measured on nec2c and against our licensed materials), so the wire's "
                       "per-metre impedance would carry a 1/(jw C d^2) term that changes when "
                       "the deck is re-segmented - the resistance and inductance fields ARE "
                       "per unit length and are served (momwire#1088)");
        if (r < 0.0 || l < 0.0)
            deck_error(name + " asks for a negative per-unit-length " +
                       (r < 0.0 ? "resistance" : "inductance") + " (" + py_g(r) + ", " +
                       py_g(l) +
                       "); a passive distributed loading is non-negative and this engine "
                       "refuses it at the card rather than let an unobserved value reach the "
                       "solver");
        std::pair<Int, Int> range = material_range(name, "per-unit-length RLC", tag, first, last);
        if (r == 0.0 && l == 0.0) return;
        DistributedRLCCard rec;
        rec.tag = tag;
        rec.segment_from = range.first;
        rec.segment_thru = range.second;
        rec.spec.kind = kind == Int(2) ? "series" : "parallel";
        rec.spec.r = r;
        rec.spec.l = l;
        distributed_rlc.push_back(rec);
        load_cards.push_back({LoadCardRef::distributed_rlc, distributed_rlc.size() - 1});
    }

    void ld5(const Card &card) {
        Int tag = card.i(1);
        Int first = card.i(2);
        Int last = card.i(3);
        double sigma = card.f(4);
        if (sigma <= 0.0)
            deck_error("LD 5 asks for a conductivity of " + py_g(sigma) +
                       " S/m; a material's conductivity must be positive "
                       "(`wire_internal_impedance`, which this value eventually reaches, "
                       "refuses the same way) and this engine refuses it at the card rather "
                       "than let an unobserved value reach the solver");
        double mu = card.f(5);
        if (mu == 0.0) mu = 1.0;
        if (mu != 1.0)
            deck_error("LD 5 asks for a relative permeability of " + py_g(mu) +
                       "; this engine's wire internal-impedance model has no permeability "
                       "parameter (`wire_internal_impedance` hard-codes vacuum permeability) "
                       "and refuses a non-unity value rather than silently modelling it as "
                       "copper");
        std::pair<Int, Int> range =
            material_range("LD 5", "per-wire conductivity", tag, first, last);
        Conductivity rec;
        rec.tag = tag;
        rec.segment_from = range.first;
        rec.segment_thru = range.second;
        rec.sigma = sigma;
        conductivities.push_back(rec);
        load_cards.push_back({LoadCardRef::conductivity, conductivities.size() - 1});
    }

    void tl(const Card &card) {
        Node a = address(card, 0);
        Node b = address(card, 2);
        double z0 = card.f(4);
        if (z0 == 0.0)
            deck_error("TL with a zero characteristic impedance is not a transmission line; "
                       "Z0 must be nonzero, and its SIGN - not its magnitude - is what "
                       "selects a crossed line");
        TransmissionLine t;
        t.end_a = a;
        t.end_b = b;
        t.z0 = z0;
        t.length_m = card.f(5);
        for (int s = 0; s < 4; ++s) t.shunt[s] = card.f(6 + s);
        lines.push_back(t);
    }

    void nt(const Card &card) {
        Network n;
        n.end_a = address(card, 0);
        n.end_b = address(card, 2);
        n.y11[0] = card.f(4);
        n.y11[1] = card.f(5);
        n.y12[0] = card.f(6);
        n.y12[1] = card.f(7);
        n.y22[0] = card.f(8);
        n.y22[1] = card.f(9);
        networks.push_back(n);
    }

    // -- requests --

    // `_fr`: one frequency, or a stepped list of them (momwire#1508): I2
    // frequencies from F, each the last plus dF (I1 0) or times dF (I1 1),
    // accumulated one step at a time (`_stepped_frequencies`).
    void fr(const Card &card) {
        Int n = card.i(1);
        frequency_mhz = card.f(4);
        has_frequency = true;
        frequencies_mhz.clear();
        frequencies_too_many = false;
        if (n <= Int(1)) return;
        Int stepping = card.i(0);
        if (stepping != Int(0) && stepping != Int(1))
            deck_error("FR stepping type " + S(stepping) +
                       " is not one NEC defines; 0 steps the frequency by adding the "
                       "increment and 1 by multiplying by it");
        const double first = card.f(4), step = card.f(5);
        const bool multiplicative = stepping == Int(1);
        // Python builds the whole tuple; past this count it would take
        // gigabytes, and the seam declines such a deck by name instead.
        const long long cap = 1000000;
        if (!n.fits_i64() || n.i64() > cap) {
            frequencies_too_many = true;
            frequencies_mhz.push_back(first);
            return;
        }
        const long long count = n.i64();
        frequencies_mhz.reserve(static_cast<std::size_t>(count));
        frequencies_mhz.push_back(first);
        for (long long k = 1; k < count; ++k) {
            const double last = frequencies_mhz.back();
            frequencies_mhz.push_back(multiplicative ? last * step : last + step);
        }
    }

    static Int at_least_one(const Int &v) { return v < Int(1) ? Int(1) : v; }

    // `_RP_XNDA` (momwire#1503): X = 1, N = 0, D = 0 or 1, any averaging A.
    static bool rp_xnda_served(const Int &xnda) {
        if (!xnda.fits_i64()) return false;
        switch (xnda.i64()) {
        case 1000: case 1001: case 1002: case 1003:
        case 1010: case 1011: case 1012: case 1013: return true;
        default: return false;
        }
    }

    // `_rp_xnda_refusal`: which digit asks for what, and what to set instead.
    static std::string rp_xnda_refusal(const Int &xnda) {
        const std::string served =
            "this engine prints XNDA 1000 to 1003 and 1010 to 1013: vertical and "
            "horizontal gains (X = 1), no normalized-gain table (N = 0), power or "
            "directive gain (D = 0 or 1), with any averaging digit A";
        if (xnda < Int(0) || xnda > Int(9999))
            return "RP 0 with XNDA " + S(xnda) + " is not an XNDA this engine reads; " + served;
        const long long v = xnda.i64();
        const long long x = v / 1000, n = v / 100 % 10, d = v / 10 % 10;
        std::vector<std::string> asks;
        if (x == 0)
            asks.push_back("major / minor axis gains (X = 0)");
        else if (x != 1)
            asks.push_back("output format X = " + std::to_string(x));
        if (n != 0) asks.push_back("a normalized-gain table (N = " + std::to_string(n) + ")");
        if (d != 0 && d != 1) asks.push_back("gain type D = " + std::to_string(d));
        if (asks.empty()) asks.push_back("averaging A = " + std::to_string(v % 10));
        std::string joined;
        for (size_t i = 0; i < asks.size(); ++i) joined += (i ? " and " : "") + asks[i];
        return "RP 0 with XNDA " + S(xnda) + " asks for " + joined +
               ", which this engine does not print; " + served +
               ".  In 4nec2, set Generate > Far Field's Expert settings to X = 1 "
               "(vertical / horizontal) and N = 0 (no normalization)";
    }

    void rp(const Card &card) {
        Int mode = card.i(0);
        if (!mode.is_zero())
            deck_error("RP mode " + S(mode) +
                       " is not part of this engine's nec5 dialect, which serves RP 0 (the "
                       "normal far-field request) alone");
        Int xnda = card.i(3);
        if (!rp_xnda_served(xnda))
            deck_error(rp_xnda_refusal(xnda));
        Request r;
        r.kind = Request::far_field;
        r.mode = mode;
        r.n_theta = at_least_one(card.i(1));
        r.n_phi = at_least_one(card.i(2));
        r.xnda = xnda;
        r.theta0_deg = card.f(4);
        r.phi0_deg = card.f(5);
        r.d_theta_deg = card.f(6);
        r.d_phi_deg = card.f(7);
        r.range_m = card.f(8);
        requests.push_back(r);
    }

    void ne(const Card &card) {
        Int coordinates = card.i(0);
        if (coordinates != Int(0) && coordinates != Int(1))
            deck_error(card.mnemonic + " coordinate system " + S(coordinates) +
                       " is not one this engine reads; the grid is rectangular (0: X, Y, Z "
                       "in metres) or spherical (1: R in metres, then theta from the zenith "
                       "and phi from +x, in degrees)");
        Request r;
        r.kind = Request::near_field;
        r.coordinates = coordinates;
        for (int d = 0; d < 3; ++d) r.counts[d] = at_least_one(card.i(1 + d));
        for (int d = 0; d < 3; ++d) {
            r.origin[d] = card.f(4 + d);
            r.step[d] = card.f(7 + d);
        }
        r.magnetic = card.mnemonic == "NH";
        requests.push_back(r);
    }

    // -- the loop --

    void feed_lines(const std::vector<std::string> &ls, std::size_t from, std::size_t to) {
        if (saw_en) return;
        Card c;
        for (std::size_t k = from; k < to; ++k) {
            if (!parse_card(ls[k], c)) continue;
            if (c.mnemonic == "EN") {
                saw_en = true;
                break;
            }
            card(c);
        }
    }

    void close() {
        if (!saw_en)
            deck_error("deck has no EN card; every deck EZNEC writes terminates with one, so a "
                       "deck that ends without EN is truncated");
    }

    void card(const Card &c) {
        const std::string &mn = c.mnemonic;
        if (mn == "CM") {
            if (has_ce)
                deck_error("CM appears after the CE that closed the comment block; a deck's "
                           "comment cards are its first cards, and the printout echoes them as "
                           "one block");
            comments.push_back(utf8(c.text()));
            return;
        }
        if (mn == "CE") {
            if (has_ce) deck_error("CE appears twice; one CE closes a comment block");
            has_ce = true;
            ce_text = utf8(c.text());
            return;
        }
        if (!has_ce)
            deck_error(mn + " arrives before any CE card; a deck's comment block must be closed "
                            "by CE before its first data card");
        if (mn == "SP") deck_error(sp_is_sphere(c) ? kSpSphere : kSpPatch);
        auto refused = refused_by_name().find(mn);
        if (refused != refused_by_name().end()) deck_error(refused->second);
        bool known = false;
        for (const char *v : kVocabulary) known = known || mn == v;
        if (!known)
            deck_error(mn + " is not a card in this engine's nec5 dialect, whose vocabulary is "
                            "the fifteen mnemonics EZNEC emits: GW, GE, GN, GD, EX, LD, TL, NT, "
                            "FR, PQ, RP, XQ, NE, NH, EN");
        int minimum = min_fields(mn);
        if (minimum >= 0 && c.values.size() < static_cast<std::size_t>(minimum)) {
            std::size_t written = c.values.size();
            deck_error(mn + " carries " + std::to_string(written) + " " +
                       (written == 1 ? "field" : "fields") + " and needs at least " +
                       std::to_string(minimum));
        }
        if (mn == "GW") return gw(c);
        if (mn == "GE") return ge(c);
        if (mn == "GN") return gn(c);
        if (mn == "GD") return gd(c);
        if (mn == "EX") return ex(c);
        if (mn == "LD") return ld(c);
        if (mn == "TL") return tl(c);
        if (mn == "NT") return nt(c);
        if (mn == "FR") return fr(c);
        if (mn == "RP") return rp(c);
        if (mn == "NE" || mn == "NH") return ne(c);
        if (mn == "PQ") {
            pq = c.i(0);
            has_pq = true;
            return;
        }
        // XQ, the vocabulary's remainder (EN never reaches here)
        Request r;
        r.kind = Request::execute;
        r.flag = c.i(0);
        requests.push_back(r);
    }

    // -- the deck --

    Deck deck(const std::string &latin1_source) const {
        if (!has_ge)
            deck_error("deck has no GE card; GE closes the geometry section and carries the "
                       "ground flag, and EZNEC writes one on every deck");
        Deck d;
        d.comments = comments;
        d.ce_text = ce_text;
        d.source_text = utf8(latin1_source);
        d.wires = wires;
        d.ge_flag = ge_flag;
        d.ge_second = ge_second;
        d.geometry_file = geometry_file;
        d.ground = ground;
        d.sources = sources;
        d.loads = loads;
        d.conductivities = conductivities;
        d.load_cards = load_cards;
        for (const Conductivity &c : conductivities) {
            std::vector<Int> tags = c.tag.is_zero()
                                        ? wires_in_absolute_range(c.segment_from, c.segment_thru)
                                        : std::vector<Int>{c.tag};
            for (const Int &t : tags) set_entry(d.wire_conductivity, t, c.sigma);
        }
        d.distributed_rlc = distributed_rlc;
        for (const DistributedRLCCard &c : distributed_rlc) {
            std::vector<Int> tags = c.tag.is_zero()
                                        ? wires_in_absolute_range(c.segment_from, c.segment_thru)
                                        : std::vector<Int>{c.tag};
            for (const Int &t : tags) set_entry(d.wire_distributed_rlc, t, c.spec);
        }
        d.transmission_lines = lines;
        d.networks = networks;
        d.has_frequency = has_frequency;
        d.frequency_mhz = frequency_mhz;
        d.frequencies_mhz = frequencies_mhz;
        d.frequencies_too_many = frequencies_too_many;
        d.requests = requests;
        d.has_pq = has_pq;
        d.pq = pq;
        return d;
    }

    template <class V>
    static void set_entry(std::vector<std::pair<Int, V>> &m, const Int &k, const V &v) {
        for (auto &e : m)
            if (e.first == k) {
                e.second = v;
                return;
            }
        m.emplace_back(k, v);
    }
};

DeckOutcome outcome_of(const ParseError &e) {
    DeckOutcome o;
    o.ok = false;
    o.kind = e.kind;
    o.message = e.what();
    return o;
}

}  // namespace

const Wire *Deck::wire(const Int &tag) const {
    for (const Wire &w : wires)
        if (w.tag == tag) return &w;
    return nullptr;
}

// ============================================================================
// The entry points
// ============================================================================

Deck parse_nec5(const std::string &text) {
    Parser p;
    std::vector<std::string> ls = py_splitlines(text);
    p.feed_lines(ls, 0, ls.size());
    p.close();
    return p.deck(text);
}

std::vector<DeckOutcome> parse_nec5_shared(const std::vector<std::string> &texts) {
    std::vector<DeckOutcome> out;
    if (texts.empty()) return out;
    std::vector<std::vector<std::string>> split;
    for (const std::string &t : texts) split.push_back(py_splitlines(t));
    const std::vector<std::string> &first = split[0];
    std::size_t limit = first.size();
    for (const auto &ls : split) limit = std::min(limit, ls.size());
    std::size_t shared = 0;
    while (shared < limit) {
        bool same = true;
        for (const auto &ls : split) same = same && ls[shared] == first[shared];
        if (!same) break;
        ++shared;
    }
    Parser prefix;
    bool failed = false;
    DeckOutcome failure;
    try {
        prefix.feed_lines(first, 0, shared);
    } catch (const ParseError &e) {
        failed = true;
        failure = outcome_of(e);
    }
    for (std::size_t t = 0; t < texts.size(); ++t) {
        if (failed) {
            out.push_back(failure);
            continue;
        }
        try {
            Parser p = prefix;
            p.feed_lines(split[t], shared, split[t].size());
            p.close();
            DeckOutcome o;
            o.ok = true;
            o.deck = p.deck(texts[t]);
            out.push_back(std::move(o));
        } catch (const ParseError &e) {
            out.push_back(outcome_of(e));
        }
    }
    return out;
}

std::string Structure::deck_text() const { return has_terminator ? body + "EN\n" : body; }

std::string Structure::echo_text() const {
    return has_terminator ? body + terminator + "\n" : body;
}

namespace {

// `_mnemonic(line)`: the first two letters of the first token, upper-cased.
std::string sniff_mnemonic(const std::string &line) {
    std::vector<std::string> tokens = py_split(replace_char(py_strip(line), ',', ' '));
    return tokens.empty() ? std::string() : py_upper(head(tokens[0], 2));
}

}  // namespace

bool Structure::ends_on_nx() const { return has_terminator && sniff_mnemonic(terminator) == "NX"; }

bool split_structures(const std::string &text, std::vector<Structure> &structures, bool open_end) {
    structures.clear();
    std::string body;
    bool broke = false;
    for (const std::string &line : py_splitlines(text)) {
        std::string mn = sniff_mnemonic(line);
        if (mn == "NX" || mn == "EN") {
            Structure s;
            s.body = body;
            s.has_terminator = true;
            s.terminator = py_strip(line);
            structures.push_back(s);
            body.clear();
            if (mn == "EN") {
                broke = true;
                break;
            }
        } else {
            body += line + "\n";
        }
    }
    if (!broke) {
        if (!body.empty() ||
            (!open_end && !structures.empty() && structures.back().has_terminator)) {
            Structure s;
            s.body = body;
            structures.push_back(s);
        }
    }
    bool any_nx = false;
    for (const Structure &s : structures) any_nx = any_nx || s.ends_on_nx();
    if (!any_nx) {
        structures.clear();
        return false;
    }
    return true;
}

bool next_structure(const std::string &text, bool final, std::size_t &end, bool &is_en) {
    // `text.splitlines(keepends=True)`, one line at a time
    std::size_t k = 0, start = 0;
    while (start < text.size()) {
        bool broken = false;
        while (k < text.size()) {
            const uc c = static_cast<uc>(text[k]);
            if (py_linebreak(c)) {
                if (c == '\r' && k + 1 < text.size() && text[k + 1] == '\n') ++k;
                ++k;
                broken = true;
                break;
            }
            ++k;
        }
        if (!broken && !final) return false;
        const std::string mn = sniff_mnemonic(text.substr(start, k - start));
        if (mn == "NX" || mn == "EN") {
            end = k;
            is_en = mn == "EN";
            return true;
        }
        start = k;
    }
    return false;
}

bool split_runs(const std::string &text, std::vector<std::string> &out) {
    out.clear();
    static const char *const run_cards[] = {"EX", "RP", "NE", "NH", "PQ", "XQ"};
    auto is_run_card = [](const std::string &c) {
        for (const char *r : run_cards)
            if (c == r) return true;
        return false;
    };
    auto card_of = [](const std::string &ln) { return py_upper(head(ln, 2)); };
    std::vector<std::string> lines = py_splitlines(text);
    std::size_t ge = lines.size();
    for (std::size_t k = 0; k < lines.size(); ++k)
        if (card_of(lines[k]) == "GE") {
            ge = k;
            break;
        }
    if (ge == lines.size()) return false;
    std::vector<std::vector<std::string>> blocks(1);
    for (std::size_t k = ge + 1; k < lines.size(); ++k) {
        const std::string &ln = lines[k];
        std::string card = card_of(ln);
        if (py_strip(ln).empty() || card == "EN") continue;
        blocks.back().push_back(ln);
        if (card == "XQ") blocks.emplace_back();
    }
    std::vector<std::vector<std::string>> kept;
    for (auto &b : blocks)
        if (!b.empty()) kept.push_back(b);
    std::size_t closed = 0;
    for (const auto &b : kept)
        if (card_of(b.back()) == "XQ") ++closed;
    if (closed <= 1) return false;
    for (std::size_t r = 1; r < kept.size(); ++r)
        for (const std::string &ln : kept[r]) {
            std::string card = card_of(ln);
            if (!is_run_card(card))
                throw ParseError(
                    ErrorKind::serve_refusal,
                    "XQ block " + std::to_string(r + 1) + " carries " + card +
                        ": a multi-run deck is served when its later runs change only their "
                        "sources and requests (EX, RP, NE, NH), which is what SimNEC writes. "
                        "Every run reuses the first block's structure, ground, loads and "
                        "frequency, so a block that changes one of those is refused rather than "
                        "silently ignored");
        }
    std::vector<std::string> common;
    for (const std::string &ln : kept[0])
        if (!is_run_card(card_of(ln))) common.push_back(ln);
    for (const auto &block : kept) {
        std::vector<std::string> parts(lines.begin(), lines.begin() + ge + 1);
        parts.insert(parts.end(), common.begin(), common.end());
        std::string xq;
        bool has_xq = false;
        for (const std::string &ln : block) {
            std::string card = card_of(ln);
            if (is_run_card(card) && card != "XQ") parts.push_back(ln);
        }
        for (const std::string &ln : block)
            if (card_of(ln) == "XQ") {
                xq = ln;
                has_xq = true;
                break;
            }
        parts.push_back(has_xq ? xq : std::string("XQ 0"));
        parts.push_back("EN");
        std::string joined;
        for (std::size_t k = 0; k < parts.size(); ++k) {
            if (k) joined += "\n";
            joined += parts[k];
        }
        out.push_back(joined + "\n");
    }
    return true;
}

// `_shell._stepping_refusal`: why a stepping-FR deck is not served, or ""
// when it is (momwire#1508). `deck_text` is the deck's source text (the echo
// when one was given), latin-1.
std::string stepping_refusal(const std::string &deck_text, bool multi_run) {
    if (multi_run)
        return "a stepping FR is served in a deck with one run; this deck has several "
               "XQ blocks, and the licensed NEC-5 answers every block after the first at "
               "the sweep's last frequency only, which this engine does not reproduce. "
               "Give each run its own deck, or step one frequency per run";
    // `_printout.post_ge_cards`: every card after GE, through EN.
    std::vector<std::string> cards;
    bool seen_ge = false;
    for (const std::string &line : latin1_splitlines(deck_text)) {
        Card card;
        if (!read_card(line, card)) continue;
        if (!seen_ge) {
            seen_ge = card.mnemonic == "GE";
            continue;
        }
        cards.push_back(card.mnemonic);
        if (card.mnemonic == "EN") break;
    }
    std::size_t first = cards.size();
    for (std::size_t k = 0; k < cards.size(); ++k)
        if (cards[k] == "RP" || cards[k] == "XQ") {
            first = k;
            break;
        }
    if (first == cards.size())
        return "a stepping FR needs an XQ or RP after it to run the sweep; without one "
               "the licensed NEC-5 solves no frequency at all (a near-field request "
               "alone does not start the loop), so add XQ after the last request";
    for (std::size_t k = first + 1; k < cards.size(); ++k)
        if (cards[k] != "XQ" && cards[k] != "EN" && cards[k] != "NX")
            return cards[k] +
                   " follows the card that runs the stepping FR's sweep; the licensed "
                   "NEC-5 answers it at the sweep's last frequency only, which this engine "
                   "does not reproduce. Put every request before the first RP or XQ";
    return "";
}

// `_serve.split_sequence`'s reading, up to the solve: 1 when the deck is
// read as a program (a later run carries an FR, the first run steps and
// another follows, or a later run is requests alone), 0 when not, -1 with `why` when a later run changes a
// structural card (`_printout.execution_groups` for the runs).
int sequence_read(const std::string &deck_text, std::string &why) {
    std::vector<Card> cards;
    bool seen_ge = false;
    for (const std::string &line : latin1_splitlines(deck_text)) {
        Card card;
        if (!read_card(line, card)) continue;
        if (!seen_ge) {
            seen_ge = card.mnemonic == "GE";
            continue;
        }
        cards.push_back(card);
        if (card.mnemonic == "EN") break;
    }
    std::vector<std::vector<std::size_t>> groups;
    std::vector<std::size_t> pending;
    bool changed = true;
    for (std::size_t k = 0; k < cards.size(); ++k) {
        const std::string &m = cards[k].mnemonic;
        if (m == "EN" || m == "NX") break;
        pending.push_back(k);
        const bool request = m == "NE" || m == "NH";
        if (m == "RP" || (m == "XQ" && changed) || (request && !changed && !groups.empty())) {
            groups.push_back(pending);
            pending.clear();
            changed = false;
        } else if (m != "XQ" && !request) {
            changed = true;
        }
    }
    if (groups.size() < 2) return 0;
    bool first_steps = false, later_fr = false, any_alone = false, all_alone = true;
    for (std::size_t g = 1; g < groups.size(); ++g) {
        bool alone = true;
        for (std::size_t k : groups[g]) {
            const std::string &m = cards[k].mnemonic;
            alone = alone && (m == "RP" || m == "NE" || m == "NH" || m == "XQ");
        }
        any_alone = any_alone || alone;
        all_alone = all_alone && alone;
    }
    // requests alone count only beside a later run that changes something
    // (momwire#1579's repeated shape); one XQ keeps the 0.79 reading
    const bool requests_alone = any_alone && !all_alone;
    for (std::size_t k : groups[0])
        if (cards[k].mnemonic == "FR" && Int(1) < cards[k].i(1)) first_steps = true;
    for (std::size_t g = 1; g < groups.size(); ++g)
        for (std::size_t k : groups[g])
            if (cards[k].mnemonic == "FR") later_fr = true;
    if (!first_steps && !later_fr && !requests_alone) return 0;
    static const char *kSeq[] = {"EX", "RP", "NE", "NH", "PQ", "XQ", "FR"};
    for (std::size_t g = 1; g < groups.size(); ++g)
        for (std::size_t k : groups[g]) {
            bool ok = false;
            for (const char *c : kSeq) ok = ok || cards[k].mnemonic == c;
            if (!ok) {
                why = "run " + std::to_string(g + 1) + " carries " + cards[k].mnemonic +
                      ": a deck whose later runs change their frequency, sources and requests "
                      "(FR, EX, RP, NE, NH, PQ) is served, each run at the FR in force, as the "
                      "licensed NEC-5 reads it; every run reuses the first run's structure, "
                      "ground and loads, so a run that changes one of those is refused rather "
                      "than silently ignored";
                return -1;
            }
        }
    return 1;
}

Answer answer(const std::string &text, const std::string *echo_text) {
    Answer a;
    std::vector<std::string> runs;
    bool has_runs = false;
    bool split_failed = false;
    ParseError split_failure(ErrorKind::serve_refusal, "");
    try {
        has_runs = split_runs(text, runs);
    } catch (const ParseError &e) {
        split_failed = true;
        split_failure = e;
    }
    std::vector<std::string> texts{text};
    if (has_runs) texts.insert(texts.end(), runs.begin(), runs.end());
    std::vector<DeckOutcome> parsed = parse_nec5_shared(texts);
    auto stop = [&a](ErrorKind kind, const std::string &message, bool refusal) {
        a.kind = refusal ? Answer::refused : Answer::raised;
        a.error_kind = kind;
        a.message = message;
        return a;
    };
    if (!parsed[0].ok)
        return stop(parsed[0].kind, parsed[0].message, parsed[0].kind == ErrorKind::deck_error);
    Deck deck = std::move(parsed[0].deck);
    if (echo_text) deck.source_text = utf8(*echo_text);
    {
        // `_answer`'s sequence reading (momwire 0.79.2), before the split
        std::string why;
        const int seq = sequence_read(text, why);
        if (seq < 0) return stop(ErrorKind::serve_refusal, why, true);
        if (seq > 0) {
            a.kind = Answer::sequence;
            a.deck = std::move(deck);
            return a;
        }
    }
    if (split_failed)
        return stop(split_failure.kind, split_failure.what(),
                    split_failure.kind == ErrorKind::serve_refusal);
    if (!deck.frequencies_mhz.empty()) {
        // `_answer`'s stepping branch (momwire#1508).
        const std::string why = stepping_refusal(echo_text ? *echo_text : text, has_runs);
        if (!why.empty()) return stop(ErrorKind::serve_refusal, why, true);
        a.kind = Answer::stepped;
        a.multi_run = has_runs;
        a.deck = std::move(deck);
        return a;
    }
    if (has_runs) {
        for (std::size_t k = 1; k < parsed.size(); ++k) {
            if (!parsed[k].ok)
                return stop(parsed[k].kind, parsed[k].message,
                            parsed[k].kind == ErrorKind::deck_error ||
                                parsed[k].kind == ErrorKind::serve_refusal);
            a.runs.push_back(std::move(parsed[k].deck));
        }
        a.kind = Answer::multi;
        a.deck = std::move(deck);
        return a;
    }
    a.kind = Answer::single;
    a.deck = std::move(deck);
    return a;
}

StreamRead read_stream(const std::string &text, bool open_end) {
    StreamRead s;
    s.chained = split_structures(text, s.structures, open_end);
    if (!s.chained) {
        s.answers.push_back(answer(text, nullptr));
        return s;
    }
    for (const Structure &st : s.structures) {
        std::string body = st.deck_text(), echo = st.echo_text();
        s.answers.push_back(answer(body, &echo));
        Answer::Kind k = s.answers.back().kind;
        if (k == Answer::refused || k == Answer::raised) break;
    }
    return s;
}

// ============================================================================
// The comparison format (mirrors `scripts/core_split/deck_compare.py`)
// ============================================================================

namespace {

struct Out {
    std::string s;
    void typ(const std::string &path, const std::string &name) { s += path + " : " + name + "\n"; }
    void seq(const std::string &path, const char *kind, std::size_t n) {
        s += path + " : " + kind + " " + std::to_string(n) + "\n";
    }
    void i(const std::string &path, const Int &v) { s += path + " = int " + v.str() + "\n"; }
    void f(const std::string &path, double v) { s += path + " = float " + hex_bits(v) + "\n"; }
    void c(const std::string &path, double re, double im) {
        s += path + " = complex " + hex_bits(re) + " " + hex_bits(im) + "\n";
    }
    void b(const std::string &path, bool v) {
        s += path + " = bool " + (v ? "True" : "False") + "\n";
    }
    void str(const std::string &path, const std::string &utf8_value) {
        s += path + " = str " + hex_bytes(utf8_value) + "\n";
    }
    void none(const std::string &path) { s += path + " = None\n"; }
    void f3(const std::string &path, const double *v, int n) {
        seq(path, "tuple", n);
        for (int k = 0; k < n; ++k) f(path + "[" + std::to_string(k) + "]", v[k]);
    }
    void error(const std::string &path, ErrorKind kind, const std::string &message) {
        s += path + " ! " + error_kind_name(kind) + " " + hex_bytes(message) + "\n";
    }
};

std::string idx(const std::string &path, std::size_t k) {
    return path + "[" + std::to_string(k) + "]";
}

void dump_node(Out &o, const std::string &p, const Node &n) {
    o.typ(p, "Nec5Node");
    o.i(p + ".tag", n.tag);
    o.i(p + ".node", n.node);
}

void dump_rlc(Out &o, const std::string &p, const DistributedRLC &r) {
    o.typ(p, "DistributedRLC");
    o.str(p + ".kind", r.kind);
    o.f(p + ".r", r.r);
    o.f(p + ".l", r.l);
    o.f(p + ".c", r.c);
}

void dump_load(Out &o, const std::string &p, const Load &l) {
    o.typ(p, "Nec5Load");
    dump_node(o, p + ".at", l.at);
    o.typ(p + ".spec", "LoadSpec");
    o.str(p + ".spec.kind", l.spec.kind);
    o.f(p + ".spec.r", l.spec.r);
    o.f(p + ".spec.l", l.spec.l);
    o.f(p + ".spec.c", l.spec.c);
    o.f(p + ".spec.x", l.spec.x);
    o.i(p + ".end_code", l.end_code);
    o.i(p + ".printed_location", l.printed_location);
}

void dump_cond(Out &o, const std::string &p, const Conductivity &c) {
    o.typ(p, "Nec5Conductivity");
    o.i(p + ".tag", c.tag);
    o.i(p + ".segment_from", c.segment_from);
    o.i(p + ".segment_thru", c.segment_thru);
    o.f(p + ".sigma", c.sigma);
}

void dump_drlc(Out &o, const std::string &p, const DistributedRLCCard &c) {
    o.typ(p, "Nec5DistributedRLC");
    o.i(p + ".tag", c.tag);
    o.i(p + ".segment_from", c.segment_from);
    o.i(p + ".segment_thru", c.segment_thru);
    dump_rlc(o, p + ".spec", c.spec);
}

void dump_deck_into(Out &o, const Deck &d, const std::string &p) {
    o.typ(p, "Nec5Deck");
    o.seq(p + ".comments", "tuple", d.comments.size());
    for (std::size_t k = 0; k < d.comments.size(); ++k) o.str(idx(p + ".comments", k), d.comments[k]);
    o.str(p + ".ce_text", d.ce_text);
    o.str(p + ".source_text", d.source_text);
    o.seq(p + ".wires", "tuple", d.wires.size());
    for (std::size_t k = 0; k < d.wires.size(); ++k) {
        std::string q = idx(p + ".wires", k);
        const Wire &w = d.wires[k];
        o.typ(q, "Nec5Wire");
        o.i(q + ".tag", w.tag);
        o.i(q + ".segment_count", w.segment_count);
        o.f3(q + ".end1", w.end1, 3);
        o.f3(q + ".end2", w.end2, 3);
        o.f(q + ".radius", w.radius);
    }
    o.i(p + ".ge_flag", d.ge_flag);
    o.i(p + ".ge_second", d.ge_second);
    o.b(p + ".geometry_file", d.geometry_file);
    {
        std::string q = p + ".ground";
        const Ground &g = d.ground;
        switch (g.kind) {
        case Ground::free_space: o.typ(q, "Nec5FreeSpace"); break;
        case Ground::perfect: o.typ(q, "Nec5PerfectGround"); break;
        case Ground::sommerfeld:
            o.typ(q, "Nec5SommerfeldGround");
            o.f(q + ".eps_r", g.eps_r);
            o.f(q + ".sigma", g.sigma);
            o.c(q + ".mu_r", g.mu_re, g.mu_im);
            o.i(q + ".spelling", g.spelling);
            break;
        case Ground::mininec:
            o.typ(q, "Nec5MininecGround");
            o.i(q + ".ix", g.ix);
            o.f(q + ".eps_r", g.eps_r);
            o.f(q + ".sigma", g.sigma);
            o.c(q + ".mu_r", g.mu_re, g.mu_im);
            o.b(q + ".sigma_sets_im_epsc", g.sigma_sets_im_epsc);
            break;
        }
    }
    o.seq(p + ".sources", "tuple", d.sources.size());
    for (std::size_t k = 0; k < d.sources.size(); ++k) {
        std::string q = idx(p + ".sources", k);
        const Source &s = d.sources[k];
        o.typ(q, "Nec5Source");
        o.i(q + ".kind", s.kind);
        dump_node(o, q + ".at", s.at);
        o.c(q + ".drive", s.drive_re, s.drive_im);
        o.i(q + ".end_code", s.end_code);
        o.i(q + ".printed_location", s.printed_location);
    }
    o.seq(p + ".loads", "tuple", d.loads.size());
    for (std::size_t k = 0; k < d.loads.size(); ++k) dump_load(o, idx(p + ".loads", k), d.loads[k]);
    o.seq(p + ".conductivities", "tuple", d.conductivities.size());
    for (std::size_t k = 0; k < d.conductivities.size(); ++k)
        dump_cond(o, idx(p + ".conductivities", k), d.conductivities[k]);
    o.seq(p + ".load_cards", "tuple", d.load_cards.size());
    for (std::size_t k = 0; k < d.load_cards.size(); ++k) {
        std::string q = idx(p + ".load_cards", k);
        const LoadCardRef &r = d.load_cards[k];
        if (r.kind == LoadCardRef::load) dump_load(o, q, d.loads[r.index]);
        else if (r.kind == LoadCardRef::conductivity) dump_cond(o, q, d.conductivities[r.index]);
        else dump_drlc(o, q, d.distributed_rlc[r.index]);
    }
    o.seq(p + ".wire_conductivity", "mapping", d.wire_conductivity.size());
    for (std::size_t k = 0; k < d.wire_conductivity.size(); ++k) {
        std::string q = p + ".wire_conductivity{" + std::to_string(k) + "}";
        o.i(q + ".key", d.wire_conductivity[k].first);
        o.f(q + ".value", d.wire_conductivity[k].second);
    }
    o.seq(p + ".distributed_rlc", "tuple", d.distributed_rlc.size());
    for (std::size_t k = 0; k < d.distributed_rlc.size(); ++k)
        dump_drlc(o, idx(p + ".distributed_rlc", k), d.distributed_rlc[k]);
    o.seq(p + ".wire_distributed_rlc", "mapping", d.wire_distributed_rlc.size());
    for (std::size_t k = 0; k < d.wire_distributed_rlc.size(); ++k) {
        std::string q = p + ".wire_distributed_rlc{" + std::to_string(k) + "}";
        o.i(q + ".key", d.wire_distributed_rlc[k].first);
        dump_rlc(o, q + ".value", d.wire_distributed_rlc[k].second);
    }
    o.seq(p + ".transmission_lines", "tuple", d.transmission_lines.size());
    for (std::size_t k = 0; k < d.transmission_lines.size(); ++k) {
        std::string q = idx(p + ".transmission_lines", k);
        const TransmissionLine &t = d.transmission_lines[k];
        o.typ(q, "Nec5TransmissionLine");
        dump_node(o, q + ".end_a", t.end_a);
        dump_node(o, q + ".end_b", t.end_b);
        o.f(q + ".z0", t.z0);
        o.f(q + ".length_m", t.length_m);
        o.f3(q + ".shunt", t.shunt, 4);
    }
    o.seq(p + ".networks", "tuple", d.networks.size());
    for (std::size_t k = 0; k < d.networks.size(); ++k) {
        std::string q = idx(p + ".networks", k);
        const Network &n = d.networks[k];
        o.typ(q, "Nec5Network");
        dump_node(o, q + ".end_a", n.end_a);
        dump_node(o, q + ".end_b", n.end_b);
        o.c(q + ".y11", n.y11[0], n.y11[1]);
        o.c(q + ".y12", n.y12[0], n.y12[1]);
        o.c(q + ".y22", n.y22[0], n.y22[1]);
    }
    if (d.has_frequency) o.f(p + ".frequency_mhz", d.frequency_mhz);
    else o.none(p + ".frequency_mhz");
    o.seq(p + ".frequencies_mhz", "tuple", d.frequencies_mhz.size());
    for (std::size_t k = 0; k < d.frequencies_mhz.size(); ++k)
        o.f(idx(p + ".frequencies_mhz", k), d.frequencies_mhz[k]);
    o.seq(p + ".requests", "tuple", d.requests.size());
    for (std::size_t k = 0; k < d.requests.size(); ++k) {
        std::string q = idx(p + ".requests", k);
        const Request &r = d.requests[k];
        if (r.kind == Request::far_field) {
            o.typ(q, "Nec5FarFieldRequest");
            o.i(q + ".n_theta", r.n_theta);
            o.i(q + ".n_phi", r.n_phi);
            o.i(q + ".xnda", r.xnda);
            o.f(q + ".theta0_deg", r.theta0_deg);
            o.f(q + ".phi0_deg", r.phi0_deg);
            o.f(q + ".d_theta_deg", r.d_theta_deg);
            o.f(q + ".d_phi_deg", r.d_phi_deg);
            o.f(q + ".range_m", r.range_m);
            o.i(q + ".mode", r.mode);
        } else if (r.kind == Request::near_field) {
            o.typ(q, "Nec5NearFieldRequest");
            o.i(q + ".coordinates", r.coordinates);
            o.seq(q + ".counts", "tuple", 3);
            for (int c = 0; c < 3; ++c) o.i(idx(q + ".counts", c), r.counts[c]);
            o.f3(q + ".origin", r.origin, 3);
            o.f3(q + ".step", r.step, 3);
            o.b(q + ".magnetic", r.magnetic);
        } else {
            o.typ(q, "Nec5ExecuteRequest");
            o.i(q + ".flag", r.flag);
        }
    }
    if (d.has_pq) o.i(p + ".pq", d.pq);
    else o.none(p + ".pq");
}

}  // namespace

std::string dump_deck(const Deck &deck, const std::string &path) {
    Out o;
    dump_deck_into(o, deck, path);
    return o.s;
}

std::string dump_parse(const std::string &text) {
    Out o;
    try {
        Deck d = parse_nec5(text);
        dump_deck_into(o, d, "parse");
    } catch (const ParseError &e) {
        o.error("parse", e.kind, e.what());
    }
    return o.s;
}

std::string dump_stream(const std::string &text) {
    Out o;
    StreamRead s = read_stream(text);
    if (!s.chained) {
        o.none("stream.structures");
    } else {
        o.seq("stream.structures", "list", s.structures.size());
        for (std::size_t k = 0; k < s.structures.size(); ++k) {
            std::string q = idx("stream.structures", k);
            o.str(q + ".body", utf8(s.structures[k].body));
            if (s.structures[k].has_terminator) o.str(q + ".terminator", utf8(s.structures[k].terminator));
            else o.none(q + ".terminator");
        }
    }
    o.seq("stream.answers", "list", s.answers.size());
    for (std::size_t k = 0; k < s.answers.size(); ++k) {
        std::string q = idx("stream.answers", k);
        const Answer &a = s.answers[k];
        switch (a.kind) {
        case Answer::refused: o.s += q + " = refused " + hex_bytes(a.message) + "\n"; break;
        case Answer::raised: o.error(q, a.error_kind, a.message); break;
        case Answer::single:
            o.s += q + " = single\n";
            dump_deck_into(o, a.deck, q + ".deck");
            break;
        case Answer::stepped:
            o.s += q + " = stepped\n";
            dump_deck_into(o, a.deck, q + ".deck");
            break;
        case Answer::sequence:
            o.s += q + " = sequence\n";
            dump_deck_into(o, a.deck, q + ".deck");
            break;
        case Answer::multi:
            o.s += q + " = multi\n";
            dump_deck_into(o, a.deck, q + ".deck");
            o.seq(q + ".runs", "list", a.runs.size());
            for (std::size_t r = 0; r < a.runs.size(); ++r)
                dump_deck_into(o, a.runs[r], idx(q + ".runs", r));
            break;
        }
    }
    return o.s;
}

std::string text_probe(const std::string &op, const std::string &arg) {
    Out o;
    auto list = [&o](const std::vector<std::string> &v) {
        o.seq("probe", "list", v.size());
        for (std::size_t k = 0; k < v.size(); ++k) o.str(idx("probe", k), utf8(v[k]));
    };
    if (op == "strip") {
        o.str("probe", utf8(py_strip(arg)));
    } else if (op == "split") {
        list(py_split(arg));
    } else if (op == "splitlines") {
        list(py_splitlines(arg));
    } else if (op == "upper") {
        std::size_t n = 0;
        std::string up = py_upper(arg, &n);
        o.str("probe", up);
        o.s += "probe.len = int " + std::to_string(n) + "\n";
    } else if (op == "isalpha" || op == "isdigit" || op == "isspace") {
        bool all = !arg.empty();
        for (char ch : arg) {
            uc c = static_cast<uc>(ch);
            all = all && (op == "isalpha" ? py_isalpha(c)
                          : op == "isdigit" ? py_isdigit(c)
                                            : py_isspace(c));
        }
        o.b("probe", all);
    } else if (op == "float" || op == "round" || op == "g" || op == "+g") {
        double v;
        if (!py_float(replace_char(replace_char(arg, 'D', 'E'), 'd', 'e'), v)) {
            o.none("probe");
        } else if (op == "float") {
            o.f("probe", v);
        } else if (op == "round") {
            if (std::isfinite(v)) o.i("probe", py_round(v));
            else o.s += "probe = nonfinite\n";
        } else {
            o.str("probe", py_g(v, op == "+g"));
        }
    } else if (op == "card") {
        Card c;
        try {
            if (!parse_card(arg, c)) {
                o.none("probe");
            } else {
                o.str("probe.mnemonic", c.mnemonic);
                o.seq("probe.values", "tuple", c.values.size());
                for (std::size_t k = 0; k < c.values.size(); ++k) o.f(idx("probe.values", k), c.values[k]);
                o.str("probe.raw", utf8(c.raw));
                if (c.has_trailer) o.str("probe.trailer", utf8(c.trailer));
                else o.none("probe.trailer");
            }
        } catch (const ParseError &e) {
            o.error("probe", e.kind, e.what());
        }
    } else {
        throw std::invalid_argument("unknown text probe " + op);
    }
    return o.s;
}

std::string format_g(double x) { return py_g(x); }

// --- the pieces the printout writer shares (`_nec5_razor_printout_core.cpp`)

double Card::f(std::size_t k) const { return k < values.size() ? values[k] : 0.0; }

// `Card.i`: an infinite / NaN field raises where Python's does, in its words
Int Card::i(std::size_t k) const {
    double v = f(k);
    if (std::isnan(v))
        throw ParseError(ErrorKind::value_error,
                         mnemonic + " field " + std::to_string(k + 1) +
                             " is NaN; an integer field must be a finite number");
    if (std::isinf(v))
        throw ParseError(ErrorKind::overflow_error,
                         mnemonic + " field " + std::to_string(k + 1) +
                             " is infinite; an integer field must be a finite number");
    return py_round(v);
}

std::string Card::text() const { return raw.size() > 2 ? raw.substr(2) : std::string(); }

bool read_card(const std::string &latin1_line, Card &card) { return parse_card(latin1_line, card); }
std::vector<std::string> latin1_splitlines(const std::string &s) { return py_splitlines(s); }
std::vector<std::string> latin1_split(const std::string &s) { return py_split(s); }
std::string latin1_upper(const std::string &s, std::size_t *length) { return py_upper(s, length); }
std::string latin1_to_utf8(const std::string &s) { return utf8(s); }
bool latin1_isspace(unsigned char c) { return py_isspace(c); }
std::string format_c_locale(const char *fmt, double x) { return format_c(fmt, x); }

}  // namespace nec5
}  // namespace mw

#if defined(_MSC_VER)
#pragma float_control(pop)
#endif
