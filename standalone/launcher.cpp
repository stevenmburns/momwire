// The launcher layer for the no-Python build (launcher.h).
#include "launcher.h"

#include "_nec5_deck_core.h"

#include <array>
#include <cerrno>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <map>
#include <tuple>

#if defined(_WIN32)
#include <io.h>
#else
#include <fcntl.h>
#include <unistd.h>
#endif

namespace mw {
namespace launcher {

namespace {

// One code point of UTF-8 at `i`, or -1 (an invalid byte, consumed alone).
long decode_one(const std::string &s, std::size_t &i) {
    const unsigned char c = static_cast<unsigned char>(s[i]);
    int len = 0;
    long cp = 0;
    if (c < 0x80) {
        i += 1;
        return c;
    } else if (c >= 0xc2 && c <= 0xdf) {
        len = 2;
        cp = c & 0x1f;
    } else if (c >= 0xe0 && c <= 0xef) {
        len = 3;
        cp = c & 0x0f;
    } else if (c >= 0xf0 && c <= 0xf4) {
        len = 4;
        cp = c & 0x07;
    } else {
        i += 1;
        return -1;
    }
    if (i + len > s.size()) {
        i += 1;
        return -1;
    }
    for (int k = 1; k < len; k++) {
        const unsigned char d = static_cast<unsigned char>(s[i + k]);
        if ((d & 0xc0) != 0x80) {
            i += 1;
            return -1;
        }
        cp = (cp << 6) | (d & 0x3f);
    }
    // overlong, surrogate, past U+10FFFF: Python's strict UTF-8 refuses them
    if ((len == 3 && cp < 0x800) || (len == 4 && (cp < 0x10000 || cp > 0x10ffff)) ||
        (cp >= 0xd800 && cp <= 0xdfff)) {
        i += 1;
        return -1;
    }
    i += len;
    return cp;
}

// Python's `str.isspace` over the code points a UTF-8 path can carry.
bool unicode_space(long cp) {
    return (cp >= 0x09 && cp <= 0x0d) || (cp >= 0x1c && cp <= 0x20) || cp == 0x85 ||
           cp == 0xa0 || cp == 0x1680 || (cp >= 0x2000 && cp <= 0x200a) || cp == 0x2028 ||
           cp == 0x2029 || cp == 0x202f || cp == 0x205f || cp == 0x3000;
}

// `repr(str)` for a path decoded with surrogateescape (an invalid byte b is
// U+DC00+b), as OSError's message spells its filename. `str.isprintable`
// is approximated by its categories for the code points a path carries:
// controls, the C1 block, U+00A0 / U+00AD, the spaces and format
// characters, private use and surrogates are escaped; the rest are printed.
std::string py_repr(const std::string &bytes) {
    std::vector<long> cps;
    for (std::size_t i = 0; i < bytes.size();) {
        const std::size_t at = i;
        long cp = decode_one(bytes, i);
        if (cp < 0) cp = 0xdc00 + static_cast<unsigned char>(bytes[at]);
        cps.push_back(cp);
    }
    bool has_single = false, has_double = false;
    for (long cp : cps) {
        has_single = has_single || cp == '\'';
        has_double = has_double || cp == '"';
    }
    const char quote = has_single && !has_double ? '"' : '\'';
    std::string out(1, quote);
    char buf[32];
    for (long cp : cps) {
        if (cp == quote || cp == '\\') {
            out += '\\';
            out += static_cast<char>(cp);
        } else if (cp == '\t') {
            out += "\\t";
        } else if (cp == '\n') {
            out += "\\n";
        } else if (cp == '\r') {
            out += "\\r";
        } else if (cp < 0x20 || (cp >= 0x7f && cp <= 0xa0) || cp == 0xad) {
            std::snprintf(buf, sizeof buf, "\\x%02lx", cp);
            out += buf;
        } else if (cp < 0x7f) {
            out += static_cast<char>(cp);
        } else if (unicode_space(cp) || (cp >= 0x200b && cp <= 0x200f) ||
                   (cp >= 0x202a && cp <= 0x202e) || (cp >= 0x2060 && cp <= 0x206f) ||
                   cp == 0xfeff || (cp >= 0xfff9 && cp <= 0xfffb) ||
                   (cp >= 0xd800 && cp <= 0xdfff) || (cp >= 0xe000 && cp <= 0xf8ff)) {
            if (cp <= 0xff)
                std::snprintf(buf, sizeof buf, "\\x%02lx", cp);
            else if (cp <= 0xffff)
                std::snprintf(buf, sizeof buf, "\\u%04lx", cp);
            else
                std::snprintf(buf, sizeof buf, "\\U%08lx", cp);
            out += buf;
        } else {
            // printable: its UTF-8
            if (cp < 0x800) {
                out += static_cast<char>(0xc0 | (cp >> 6));
                out += static_cast<char>(0x80 | (cp & 0x3f));
            } else if (cp < 0x10000) {
                out += static_cast<char>(0xe0 | (cp >> 12));
                out += static_cast<char>(0x80 | ((cp >> 6) & 0x3f));
                out += static_cast<char>(0x80 | (cp & 0x3f));
            } else {
                out += static_cast<char>(0xf0 | (cp >> 18));
                out += static_cast<char>(0x80 | ((cp >> 12) & 0x3f));
                out += static_cast<char>(0x80 | ((cp >> 6) & 0x3f));
                out += static_cast<char>(0x80 | (cp & 0x3f));
            }
        }
    }
    out += quote;
    return out;
}

std::string fmt(const char *f, double v) {
    char buf[64];
    std::snprintf(buf, sizeof buf, f, v);
    return buf;
}

std::string fmt_i(const char *f, long long v) {
    char buf[64];
    std::snprintf(buf, sizeof buf, f, v);
    return buf;
}

using Key = std::tuple<double, double, double>;

// `_wire_nodes`' keys for one wire's points: numpy's int64 path when every
// rounded coordinate is under 2**62, else `_node_key` point by point, whose
// `round` raises on NaN and infinity in Python's words. A key is the rounded
// value itself, which is the exact integer either path makes.
std::vector<Key> wire_keys(const std::vector<std::array<double, 3>> &points) {
    const double eps = 1e-6;  // `_NODE_EPS`
    std::vector<Key> keys;
    bool fits = true;
    for (const auto &p : points)
        for (double c : p) fits = fits && std::fabs(std::nearbyint(c / eps)) < 4611686018427387904.0;
    for (const auto &p : points) {
        double k[3];
        for (int c = 0; c < 3; c++) {
            const double q = p[c] / eps;
            if (!fits) {
                if (std::isnan(q)) throw PyError{"ValueError", "cannot convert float NaN to integer"};
                if (std::isinf(q))
                    throw PyError{"OverflowError", "cannot convert float infinity to integer"};
            }
            k[c] = std::nearbyint(q) + 0.0;  // -0 and 0 are one key, as ints are
        }
        keys.emplace_back(k[0], k[1], k[2]);
    }
    return keys;
}

// `_gmo.render_gmo`.
std::string render_gmo(const nec5::Deck &deck) {
    std::map<Key, long long> index;
    std::vector<std::array<double, 3>> points;
    struct Element {
        long long first, second;
        double radius;
    };
    std::vector<Element> elements;
    for (const nec5::Wire &w : deck.wires) {
        // the reader holds segment_count >= 1; a count numpy could not
        // allocate is not this build's to answer
        if (!w.segment_count.fits_i64() || w.segment_count.i64() > 100000000LL)
            throw PyError{"", "gmo-segment-count"};
        const long long n = w.segment_count.i64();
        std::vector<std::array<double, 3>> wp(static_cast<std::size_t>(n + 1));
        for (long long k = 0; k <= n; k++) {
            if (k == 0) {
                wp[0] = {w.end1[0], w.end1[1], w.end1[2]};
            } else if (k == n) {
                wp[static_cast<std::size_t>(k)] = {w.end2[0], w.end2[1], w.end2[2]};
            } else {
                const double t = static_cast<double>(k) / static_cast<double>(n);
                for (int c = 0; c < 3; c++)
                    wp[static_cast<std::size_t>(k)][c] = w.end1[c] + (w.end2[c] - w.end1[c]) * t;
            }
        }
        const std::vector<Key> keys = wire_keys(wp);
        std::vector<long long> numbers;
        for (std::size_t j = 0; j < wp.size(); j++) {
            auto it = index.find(keys[j]);
            long long number;
            if (it == index.end()) {
                number = static_cast<long long>(points.size()) + 1;
                index.emplace(keys[j], number);
                points.push_back(wp[j]);
            } else {
                number = it->second;
            }
            numbers.push_back(number);
        }
        for (std::size_t j = 0; j + 1 < numbers.size(); j++)
            elements.push_back({numbers[j], numbers[j + 1], w.radius});
    }
    std::string out = fmt_i("%6lld", static_cast<long long>(points.size())) + "\n";
    for (const auto &p : points)
        out += " " + fmt("%13.5E", p[0]) + fmt("%13.5E", p[1]) + fmt("%13.5E", p[2]) + "\n";
    out += fmt_i("%6lld", static_cast<long long>(elements.size())) + "\n";
    for (const Element &e : elements)
        out += fmt_i("%7lld", e.first) + fmt_i("%7lld", e.second) + fmt("%13.5E", e.radius) + "\n";
    return out;
}

std::string lstrip_latin1(const std::string &s) {
    std::size_t a = 0;
    while (a < s.size() && nec5::latin1_isspace(static_cast<unsigned char>(s[a]))) a++;
    return s.substr(a);
}

PyError reader_error(const nec5::ParseError &e) {
    return PyError{nec5::error_kind_name(e.kind), e.what()};
}

}  // namespace

std::string os_bytes_to_utf8(const std::string &bytes) {
    std::string out;
    for (std::size_t i = 0; i < bytes.size();) {
        const std::size_t at = i;
        const long cp = decode_one(bytes, i);
        if (cp < 0)
            out += "\xef\xbf\xbd";
        else
            out += bytes.substr(at, i - at);
    }
    return out;
}

std::string strip_path_line(const std::string &line) {
    // positions of every code point, to strip whole ones from either end
    std::vector<std::pair<std::size_t, long>> cps;
    for (std::size_t i = 0; i < line.size();) {
        const std::size_t at = i;
        cps.emplace_back(at, decode_one(line, i));
    }
    std::size_t a = 0, b = cps.size();
    while (a < b && unicode_space(cps[a].second)) a++;
    while (b > a && unicode_space(cps[b - 1].second)) b--;
    const std::size_t from = a < cps.size() ? cps[a].first : line.size();
    const std::size_t to = b < cps.size() ? cps[b].first : line.size();
    std::string s = line.substr(from, to - from);
    if (s.size() >= 2 && s.front() == '"' && s.back() == '"') s = s.substr(1, s.size() - 2);
    return s;
}

bool stdin_paths(std::vector<std::string> &out) {
    out.clear();
#if defined(_WIN32)
    if (_isatty(0)) return false;
#else
    if (fcntl(0, F_GETFD) == -1) return false;  // no stdin: `sys.stdin is None`
    if (isatty(0)) return false;
#endif
    while (out.size() < 2) {
        std::string line;
        char c;
        for (;;) {
#if defined(_WIN32)
            const int r = _read(0, &c, 1);
#else
            const ssize_t r = read(0, &c, 1);
#endif
            if (r < 0) {
                if (errno == EINTR) continue;
                out.clear();  // OSError: nothing arrived
                return true;
            }
            if (r == 0) break;
            line += c;
            if (c == '\n') break;
        }
        const std::string path = strip_path_line(line);
        if (path.empty()) break;
        out.push_back(path);
    }
    return true;
}

std::string path_str(const std::string &p) {
    std::string root;
    std::size_t i = 0;
    if (!p.empty() && p[0] == '/') {
        root = (p.size() >= 2 && p[1] == '/' && (p.size() == 2 || p[2] != '/')) ? "//" : "/";
        while (i < p.size() && p[i] == '/') i++;
    }
    std::vector<std::string> parts;
    while (i < p.size()) {
        std::size_t j = p.find('/', i);
        if (j == std::string::npos) j = p.size();
        const std::string part = p.substr(i, j - i);
        if (!part.empty() && part != ".") parts.push_back(part);
        i = j + 1;
    }
    std::string out = root;
    for (std::size_t k = 0; k < parts.size(); k++) out += (k ? "/" : "") + parts[k];
    return out.empty() ? "." : out;
}

std::string close_deck(const std::string &latin1) {
    for (const std::string &line : nec5::latin1_splitlines(latin1)) {
        // `line.strip().replace(",", " ").split()[0][:2].upper() == "EN"`
        std::size_t a = 0;
        while (a < line.size() && (nec5::latin1_isspace(static_cast<unsigned char>(line[a])) ||
                                   line[a] == ','))
            a++;
        // a leading comma survives strip() but is a separator after replace
        if (a + 1 < line.size()) {
            const char c0 = line[a], c1 = line[a + 1];
            if ((c0 == 'E' || c0 == 'e') && (c1 == 'N' || c1 == 'n')) return latin1;
        }
    }
    std::string text = latin1;
    if (!text.empty() && text.back() != '\n' && text.back() != '\r') text += "\n";
    return text + "EN\n";
}

std::string gmo_path(const std::string &printout_path) {
    const std::size_t sep = printout_path.rfind('/');
    const std::size_t dot = printout_path.rfind('.');
    const long long sep_i = sep == std::string::npos ? -1 : static_cast<long long>(sep);
    if (dot != std::string::npos && static_cast<long long>(dot) > sep_i) {
        for (std::size_t f = static_cast<std::size_t>(sep_i + 1); f < dot; f++)
            if (printout_path[f] != '.') return printout_path.substr(0, dot) + ".GMO";
    }
    return printout_path + ".GMO";
}

bool gmo_text(const std::string &latin1, std::string &out) {
    // `requests_gmo`
    bool asks = false;
    for (const std::string &line : nec5::latin1_splitlines(latin1)) {
        const std::string s = lstrip_latin1(line);
        if (s.size() < 2 || (s[0] != 'G' && s[0] != 'g') || (s[1] != 'E' && s[1] != 'e')) continue;
        nec5::Card card;
        try {
            if (!nec5::read_card(line, card)) continue;
        } catch (const nec5::ParseError &e) {
            if (e.kind == nec5::ErrorKind::deck_error || e.kind == nec5::ErrorKind::value_error)
                continue;
            throw reader_error(e);
        }
        if (card.mnemonic == "GE" && card.f(2) != 0.0) {
            asks = true;
            break;
        }
    }
    if (!asks) return false;
    std::vector<nec5::Structure> structures;
    std::vector<std::string> texts;
    if (nec5::split_structures(latin1, structures))
        for (const nec5::Structure &s : structures) texts.push_back(s.deck_text());
    else
        texts.push_back(latin1);
    bool any = false;
    for (const std::string &t : texts) {
        nec5::Deck deck;
        try {
            deck = nec5::parse_nec5(t);
        } catch (const nec5::ParseError &e) {
            if (e.kind == nec5::ErrorKind::deck_error || e.kind == nec5::ErrorKind::value_error ||
                e.kind == nec5::ErrorKind::overflow_error)
                continue;
            throw reader_error(e);
        }
        if (deck.geometry_file) {
            out = render_gmo(deck);
            any = true;
        }
    }
    return any;
}

PyError os_error(int err, const std::string &path) {
    const char *kind = "OSError";
    switch (err) {
    case ENOENT: kind = "FileNotFoundError"; break;
    case EACCES:
    case EPERM: kind = "PermissionError"; break;
    case EISDIR: kind = "IsADirectoryError"; break;
    case ENOTDIR: kind = "NotADirectoryError"; break;
    case EEXIST: kind = "FileExistsError"; break;
    default: break;
    }
    return PyError{kind, "[Errno " + std::to_string(err) + "] " + std::strerror(err) + ": " +
                             py_repr(path)};
}

}  // namespace launcher
}  // namespace mw
