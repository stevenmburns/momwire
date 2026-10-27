// The nec2 / nec4 deck reader in C++ (see `_nec4_deck_core.h`). Each section
// names the Python it copies; the order of the checks, and so which refusal a
// deck hears first, is Python's, card by card.
//
// Arithmetic. Every float expression is Python's, in Python's evaluation
// order, with no contraction (the extension builds -ffp-contract=off, as the
// rest of the tree), so a coordinate, a length or an arclength is the same
// double. `math.dist` and `sum()` are CPython 3.12's algorithms, written out
// below. sin and cos are libm's, each from its own non-inlined call so the
// compiler cannot pair them into a sincos.
//
// MSVC builds the extension /fp:fast, which may fold a compensated sum's
// (a - x) + b to zero. CPython's own MSVC build is /fp:precise, so this TU is
// too: a reader has no fill to speed up.
#if defined(_MSC_VER) && !defined(__clang__)
#pragma float_control(precise, on)
#pragma fp_contract(off)
#endif

// M_PI: see `_accel_common.h`; must stay above the first math header.
#define _USE_MATH_DEFINES

#include "_nec4_deck_core.h"

#include <algorithm>
#include <array>
#include <cerrno>
#include <cfloat>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <limits>
#include <map>
#include <set>
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

// --- limits past which this copy declines ------------------------------------

const ll kMaxSegments = 10000000;   // one wire's, and the structure's
const ll kMaxWires = 1000000;       // after every replication
const ll kMaxCount = 1000000;       // frequencies, address ranges

const double kPi = 3.141592653589793;  // math.pi
const double kDeg = kPi / 180.0;       // _nec2_geometry._DEG
const double kSmin = 1e-3;             // _SMIN
const double kNodeEps = 1e-9;          // _NODE_EPS
const double kDefaultFrequencyMHz = 299.8;
const ll kLdExpandMax = 8;

[[noreturn]] void deck_error(const std::string &message) {
    throw ParseError(ErrorKind::deck_error, message);
}

[[noreturn]] void decline(const std::string &why) { throw Declined(why); }

// --- ints ----------------------------------------------------------------------

ll to_ll(const Int &v, const char *what) {
    if (!v.fits_i64()) decline(std::string("an integer past 64 bits in ") + what);
    return v.i64();
}

// A count this copy loops over: Python's value when it fits, else declined.
ll count_of(const Int &v, ll limit, const char *what) {
    ll n = to_ll(v, what);
    if (n > limit) decline(std::string("too many ") + what);
    return n;
}

Int imax(const Int &a, const Int &b) { return b > a ? b : a; }

std::string S(const Int &v) { return v.str(); }
std::string S(ll v) { return std::to_string(v); }

// Python's floor division and modulo.
ll floordiv(ll a, ll b) {
    ll q = a / b;
    if ((a % b != 0) && ((a < 0) != (b < 0))) --q;
    return q;
}
ll pymod(ll a, ll b) { return a - floordiv(a, b) * b; }

// `int(x)` of a finite double: truncation toward zero, exact.
Int py_int_trunc(double x) { return Int::from_integral_double(std::trunc(x)); }

// --- floats --------------------------------------------------------------------

#if defined(__GNUC__)
#define MW_NOINLINE __attribute__((noinline))
#elif defined(_MSC_VER)
#define MW_NOINLINE __declspec(noinline)
#else
#define MW_NOINLINE
#endif

MW_NOINLINE double libm_sin(double x) { return std::sin(x); }
MW_NOINLINE double libm_cos(double x) { return std::cos(x); }

// CPython 3.12's `vector_norm` (Modules/mathmodule.c), the kernel of
// `math.dist`: lossless scaling, error-free squares and sums, one
// differential correction.
struct DoubleLength {
    double hi, lo;
};
DoubleLength dl_fast_sum(double a, double b) {
    double x = a + b;
    double y = (a - x) + b;
    return {x, y};
}
DoubleLength dl_mul(double x, double y) {
    double z = x * y;
    double zz = std::fma(x, y, -z);
    return {z, zz};
}

double vector_norm(int n, double *vec, double max, bool found_nan) {
    double x, h, scale, csum = 1.0, frac1 = 0.0, frac2 = 0.0;
    DoubleLength pr, sm;
    int max_e;
    if (std::isinf(max)) return max;
    if (found_nan) return std::numeric_limits<double>::quiet_NaN();
    if (max == 0.0 || n <= 1) return max;
    std::frexp(max, &max_e);
    if (max_e < -1023) {
        for (int i = 0; i < n; i++) vec[i] /= DBL_MIN;
        return DBL_MIN * vector_norm(n, vec, max / DBL_MIN, found_nan);
    }
    scale = std::ldexp(1.0, -max_e);
    for (int i = 0; i < n; i++) {
        x = vec[i];
        x *= scale;
        pr = dl_mul(x, x);
        sm = dl_fast_sum(csum, pr.hi);
        csum = sm.hi;
        frac1 += pr.lo;
        frac2 += sm.lo;
    }
    h = std::sqrt(csum - 1.0 + (frac1 + frac2));
    pr = dl_mul(-h, h);
    sm = dl_fast_sum(csum, pr.hi);
    csum = sm.hi;
    frac1 += pr.lo;
    frac2 += sm.lo;
    x = csum - 1.0 + (frac1 + frac2);
    h += x / (2.0 * h);
    return h / scale;
}

double dist3(const double *p, const double *q) {
    double diffs[3];
    double max = 0.0;
    bool found_nan = false;
    for (int i = 0; i < 3; i++) {
        double x = std::fabs(p[i] - q[i]);
        diffs[i] = x;
        found_nan = found_nan || std::isnan(x);
        if (x > max) max = x;
    }
    return vector_norm(3, diffs, max, found_nan);
}

// CPython 3.12's `sum()` of floats from the int start 0: the first item is
// added to the int (0 + x), the rest by Neumaier's compensated summation, and
// the compensation joins at the end when it is nonzero and finite.
double py_sum(const double *v, int n) {
    if (n == 0) return 0.0;  // (the int 0: never reached here)
    double f = 0.0 + v[0];
    double c = 0.0;
    for (int i = 1; i < n; i++) {
        double x = v[i];
        double t = f + x;
        if (std::fabs(f) >= std::fabs(x)) {
            c += (f - t) + x;
        } else {
            c += (x - t) + f;
        }
        f = t;
    }
    if (c != 0.0 && std::isfinite(c)) f += c;
    return f;
}

// `sum(abs(x - y) for x, y in zip(a, b))`: snap_connections' l1.
double l1(const double *a, const double *b) {
    double d[3];
    for (int k = 0; k < 3; k++) d[k] = std::fabs(a[k] - b[k]);
    return py_sum(d, 3);
}

std::string g(double x) { return nec5::format_g(x); }

// --- Python's repr of a str (a GN trailer, a mnemonic), as UTF-8 -------------

bool latin1_printable(unsigned char c) {
    if (c < 0x20 || c == 0x7f) return false;
    if (c < 0x7f) return true;
    if (c <= 0xa0 || c == 0xad) return false;
    return true;
}

void put_utf8(std::string &out, unsigned cp) {
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

std::string repr_latin1(const std::string &s) {
    bool squote = s.find('\'') != std::string::npos;
    bool dquote = s.find('"') != std::string::npos;
    char quote = (squote && !dquote) ? '"' : '\'';
    std::string out(1, quote);
    for (char ch : s) {
        unsigned char c = static_cast<unsigned char>(ch);
        if (c == static_cast<unsigned char>(quote) || c == '\\') {
            out += '\\';
            out += ch;
        } else if (c == '\t') {
            out += "\\t";
        } else if (c == '\n') {
            out += "\\n";
        } else if (c == '\r') {
            out += "\\r";
        } else if (!latin1_printable(c)) {
            char buf[8];
            std::snprintf(buf, sizeof buf, "\\x%02x", c);
            out += buf;
        } else {
            put_utf8(out, c);
        }
    }
    out += quote;
    return out;
}

// repr of an already-UTF-8 mnemonic (two alphabetic code points, which are
// all printable and never a quote).
std::string repr_utf8_alpha(const std::string &s) { return "'" + s + "'"; }

// ============================================================================
// The geometry (`deck/_nec2_geometry.py`)
// ============================================================================

struct FlatWire {
    Int tag;
    ll n_seg = 0;
    double p1[3] = {0, 0, 0}, p2[3] = {0, 0, 0};
    double radius = 0;
    double length() const { return dist3(p1, p2); }
};

struct Piece {
    ll wire = 0, first_seg = 0, n_seg = 0;
    double p1[3] = {0, 0, 0}, p2[3] = {0, 0, 0};
    double length() const { return dist3(p1, p2); }
};

struct Symmetry {
    ll cell_wires = 0, cell_segments = 0;
};

typedef std::pair<ll, ll> Pair;  // (flat wire, 1-based local) or (piece, element)

struct Structure {
    std::vector<FlatWire> wires;
    std::vector<Piece> pieces;
    std::vector<std::vector<ll>> pieces_of_wire;
    bool ground_plane_flag = false;
    bool ground_plane_interpolates = false;
    bool has_symmetry = false;
    Symmetry symmetry;

    ll n_segments() const {
        ll n = 0;
        for (const FlatWire &w : wires) n += w.n_seg;
        return n;
    }

    Pair locate(const Int &tag, const Int &seg_in) const {
        Int seg = imax(seg_in, Int(1));
        ll acc = 0;
        for (std::size_t i = 0; i < wires.size(); i++) {
            const FlatWire &w = wires[i];
            if (!tag.is_zero() && w.tag != tag) continue;
            if (Int(acc + w.n_seg) >= seg) return {static_cast<ll>(i), seg.i64() - acc};
            acc += w.n_seg;
        }
        deck_error("segment " + S(seg) + " is out of range for " +
                   (!tag.is_zero() ? "tag " + S(tag) : std::string("the structure")));
    }

    Pair element_of(ll wire, ll local) const {
        for (ll piece_index : pieces_of_wire[static_cast<std::size_t>(wire)]) {
            const Piece &piece = pieces[static_cast<std::size_t>(piece_index)];
            if (piece.first_seg < local && local <= piece.first_seg + piece.n_seg)
                return {piece_index, local - piece.first_seg - 1};
        }
        throw ParseError(ErrorKind::deck_error, "unreachable: the pieces of a wire cover it");
    }

    Pair element(const Int &tag, const Int &seg) const {
        Pair at = locate(tag, seg);
        return element_of(at.first, at.second);
    }

    std::pair<ll, double> resolve_of(ll wire, ll local) const {
        Pair pe = element_of(wire, local);
        const Piece &piece = pieces[static_cast<std::size_t>(pe.first)];
        double element = static_cast<double>(pe.second);
        return {pe.first, (element + 0.5) * piece.length() / static_cast<double>(piece.n_seg)};
    }

    std::pair<ll, double> resolve(const Int &tag, const Int &seg) const {
        Pair at = locate(tag, seg);
        return resolve_of(at.first, at.second);
    }

    std::vector<Pair> wire_of_segment_range(const Int &tag, Int first, Int last) const {
        if (last < first) std::swap(first, last);
        Int s = imax(first, Int(1));
        std::vector<Pair> out;
        for (; s <= last; s = s + Int(1)) {
            out.push_back(element(tag, s));
            if (static_cast<ll>(out.size()) > kMaxCount) decline("too many segments in a PT range");
        }
        return out;
    }

    std::vector<Pair> ld_segment_range(const Int &tag, const Int &first, const Int &last_in) const {
        std::vector<Pair> out;
        if (tag.is_zero() && first.is_zero()) {
            for (std::size_t i = 0; i < wires.size(); i++)
                for (ll s = 1; s <= wires[i].n_seg; s++) out.push_back({static_cast<ll>(i), s});
            return out;
        }
        if (first.is_zero()) {
            for (std::size_t i = 0; i < wires.size(); i++)
                if (wires[i].tag == tag)
                    for (ll s = 1; s <= wires[i].n_seg; s++) out.push_back({static_cast<ll>(i), s});
            if (out.empty()) deck_error("no wire has tag " + S(tag));
            return out;
        }
        Int last = imax(last_in, first);
        for (Int s = first; s <= last; s = s + Int(1)) {
            out.push_back(locate(tag, s));
            if (static_cast<ll>(out.size()) > kMaxCount) decline("too many segments in an LD / IS range");
        }
        return out;
    }

    // (applied, dropped)
    std::pair<std::vector<Pair>, std::vector<Pair>> under_the_cell_rule(
        const std::vector<Pair> &pairs) const {
        if (!has_symmetry) return {pairs, {}};
        ll cell = symmetry.cell_wires;
        if (cell == 0) decline("a symmetric cell of no wires (the reference divides by zero)");
        ll copies = static_cast<ll>(wires.size()) / cell;
        std::vector<Pair> applied;
        std::set<Pair> seen;
        for (const Pair &p : pairs) {
            if (p.first >= cell) continue;
            for (ll k = 0; k < copies; k++) {
                Pair key{p.first + k * cell, p.second};
                if (seen.insert(key).second) applied.push_back(key);
            }
        }
        std::sort(applied.begin(), applied.end());
        std::vector<Pair> dropped;
        for (const Pair &p : pairs)
            if (!seen.count(p)) dropped.push_back(p);
        return {applied, dropped};
    }
};

void check_wire_count(const std::vector<FlatWire> &wires) {
    if (static_cast<ll>(wires.size()) > kMaxWires) decline("too many wires");
}

void gw(const Card &card, std::vector<FlatWire> &wires) {
    Int tag = card.i(0), n_seg = card.i(1);
    if (n_seg < Int(1)) deck_error("segment count must be >= 1, got " + S(n_seg));
    double radius = card.f(8);
    if (radius <= 0.0)
        deck_error(
            "GW with a non-positive radius announces a tapered wire (GW + GC continuation), "
            "which is not part of this engine's nec2 dialect");
    FlatWire w;
    w.tag = tag;
    w.n_seg = count_of(n_seg, kMaxSegments, "segments on one wire");
    w.p1[0] = card.f(2);
    w.p1[1] = card.f(3);
    w.p1[2] = card.f(4);
    w.p2[0] = card.f(5);
    w.p2[1] = card.f(6);
    w.p2[2] = card.f(7);
    w.radius = radius;
    wires.push_back(w);
    check_wire_count(wires);
}

std::size_t first_wire_with_tag(const std::vector<FlatWire> &wires, const Int &tag) {
    if (tag <= Int(0)) return 0;
    for (std::size_t i = 0; i < wires.size(); i++)
        if (wires[i].tag == tag) return i;
    deck_error("no wire has tag " + S(tag));
}

Int next_tag(const Int &tag, const Int &increment) {
    return tag.is_zero() ? Int(0) : tag + increment;
}

void gm(const Card &card, std::vector<FlatWire> &wires) {
    Int itgi = card.i(0), nrpt = card.i(1);
    double rox = card.f(2) * kDeg, roy = card.f(3) * kDeg, roz = card.f(4) * kDeg;
    double xs = card.f(5), ys = card.f(6), zs = card.f(7);
    Int its = py_int_trunc(card.f(8) + 0.5);
    double sps = libm_sin(rox), cps = libm_cos(rox);
    double sth = libm_sin(roy), cth = libm_cos(roy);
    double sph = libm_sin(roz), cph = libm_cos(roz);
    const double m[3][3] = {
        {cph * cth, cph * sth * sps - sph * cps, cph * sth * cps + sph * sps},
        {sph * cth, sph * sth * sps + cph * cps, sph * sth * cps - cph * sps},
        {-sth, cth * sps, cth * cps},
    };
    auto xf = [&](const double *p, double *out) {
        double q[3];
        q[0] = p[0] * m[0][0] + p[1] * m[0][1] + p[2] * m[0][2] + xs;
        q[1] = p[0] * m[1][0] + p[1] * m[1][1] + p[2] * m[1][2] + ys;
        q[2] = p[0] * m[2][0] + p[1] * m[2][1] + p[2] * m[2][2] + zs;
        out[0] = q[0];
        out[1] = q[1];
        out[2] = q[2];
    };
    std::size_t i1 = first_wire_with_tag(wires, its);
    if (nrpt.is_zero()) {
        for (std::size_t k = i1; k < wires.size(); k++) {
            xf(wires[k].p1, wires[k].p1);
            xf(wires[k].p2, wires[k].p2);
        }
        return;
    }
    std::vector<FlatWire> block(wires.begin() + static_cast<std::ptrdiff_t>(i1), wires.end());
    if (nrpt <= Int(0)) return;  // range(nrpt) is empty
    ll repeats = count_of(nrpt, kMaxWires, "GM repetitions");
    if (!block.empty() && repeats > kMaxWires / static_cast<ll>(block.size())) decline("too many wires");
    for (ll r = 0; r < repeats; r++) {
        std::vector<FlatWire> next;
        next.reserve(block.size());
        for (const FlatWire &w : block) {
            FlatWire c;
            c.tag = next_tag(w.tag, itgi);
            c.n_seg = w.n_seg;
            xf(w.p1, c.p1);
            xf(w.p2, c.p2);
            c.radius = w.radius;
            next.push_back(c);
        }
        block = next;
        wires.insert(wires.end(), block.begin(), block.end());
        check_wire_count(wires);
    }
}

void gs(const Card &card, std::vector<FlatWire> &wires) {
    Int lo = card.i(0), hi = card.i(1);
    double factor = card.f(2);
    if (factor <= 0.0) deck_error("scale factor must be > 0, got " + py_repr(factor));
    bool ranged = lo > Int(0) && hi >= lo;
    for (FlatWire &w : wires) {
        if (ranged && !(lo <= w.tag && w.tag <= hi)) continue;
        for (int k = 0; k < 3; k++) {
            w.p1[k] = w.p1[k] * factor;
            w.p2[k] = w.p2[k] * factor;
        }
        w.radius *= factor;
    }
}

// --- GA and GH (`deck/_nec4.py`, momwire#1592; the nec2 dialect's, #1651) --

// Python's `float ** float` for a finite positive base (`float_pow`): 1 at a
// zero exponent or a base of 1, else libm's pow, and an OverflowError where
// it overflows (an underflow to 0 is no error).
double py_float_pow(double x, double y) {
    if (y == 0.0) return 1.0;
    if (x == 1.0) return 1.0;
    double r = std::pow(x, y);
    if (std::isinf(r))
        throw ParseError(ErrorKind::overflow_error,
                         "(" + std::to_string(ERANGE) + ", '" + std::strerror(ERANGE) + "')");
    return r;
}

// `deck._nec4._chords`: one one-segment wire per pair of vertices, each at
// its own wire radius, all carrying `tag`.
void chords(const Int &tag, const std::vector<std::array<double, 3>> &v,
            const std::vector<double> &radii, std::vector<FlatWire> &wires) {
    for (std::size_t k = 0; k + 1 < v.size(); k++) {
        FlatWire w;
        w.tag = tag;
        w.n_seg = 1;
        for (int c = 0; c < 3; c++) {
            w.p1[c] = v[k][c];
            w.p2[c] = v[k + 1][c];
        }
        w.radius = radii[k];
        wires.push_back(w);
        check_wire_count(wires);
    }
}

// `deck._nec2_geometry.arc_vertices` and `_ga`, both dialects' (momwire#1651):
// they differ only in the widest sweep (`_GA_SWEEP_LIMIT`) and the names.
void ga(const Card &card, std::vector<FlatWire> &wires, Dialect dialect) {
    const bool nec4 = dialect == Dialect::nec4;
    Int n_seg = card.i(1);
    double radius = card.f(2), first = card.f(3), last = card.f(4);
    if (n_seg < Int(1)) deck_error("GA segment count must be >= 1, got " + S(n_seg));
    if (radius <= 0.0) deck_error("GA needs a positive arc radius (F1), got " + g(radius));
    if (first == last)
        deck_error("GA from " + g(first) + " to " + g(last) +
                   " degrees has no length; the arc's two angles (F2, F3) must differ");
    if (std::fabs(last - first) >= (nec4 ? 360.0 : 360.00001))
        deck_error("GA from " + g(first) + " to " + g(last) + " degrees sweeps " + g(std::fabs(last - first)) +
                   " degrees, which " + (nec4 ? "NEC-4.2" : "NEC-2") +
                   " stops on (ARC ANGLE EXCEEDS 360 DEGREES); a closed loop is an arc of less than 360 "
                   "degrees closed by its own end segments");
    if (card.f(5) <= 0.0)
        deck_error("GA needs a positive wire radius (F4), got " + g(card.f(5)) + "; this engine's " +
                   (nec4 ? "nec4" : "nec2") + " dialect has no tapered arc");
    ll n = count_of(n_seg, kMaxWires, "segments on one arc");
    double step = (last - first) * kDeg / static_cast<double>(n);
    double start = first * kDeg;
    std::vector<std::array<double, 3>> v;
    for (ll k = 0; k <= n; k++) {
        double a = start + static_cast<double>(k) * step;
        v.push_back({radius * libm_cos(a), 0.0, radius * libm_sin(a)});
    }
    chords(card.i(0), v, std::vector<double>(static_cast<std::size_t>(n), card.f(5)), wires);
}

// `deck._nec2_geometry.nec2_helix_vertices` and `_gh`: NEC-2's GH, nec2c's
// `helix` (momwire#1651). Not NEC-4.2's card (`helix_points` below).
void gh_nec2(const Card &card, std::vector<FlatWire> &wires) {
    Int n_seg = card.i(1);
    double s = card.f(2), hl = card.f(3), a1 = card.f(4), b1 = card.f(5), a2 = card.f(6), b2 = card.f(7);
    if (n_seg < Int(1)) deck_error("GH segment count must be >= 1, got " + S(n_seg));
    if (s == 0.0)
        deck_error("GH needs a nonzero spacing between turns (F1); this engine's nec2 dialect reads NEC-2's "
                   "GH (turn spacing, total length, four helix radii, wire radius)");
    if (hl == 0.0) deck_error("GH needs a nonzero total length (F2)");
    double lowest = a1;  // Python's `min(a1, b1, a2, b2)`, NaN rules included
    for (double r : {b1, a2, b2})
        if (r < lowest) lowest = r;
    if (lowest < 0.0 || (a1 == 0.0 && a2 == 0.0))
        deck_error("GH needs non-negative helix radii (F3-F6) and a positive one at one end at least, got " +
                   g(a1) + ", " + g(b1) + ", " + g(a2) + " and " + g(b2));
    if (card.f(8) <= 0.0)
        deck_error("GH needs a positive wire radius (F7), got " + g(card.f(8)) +
                   "; this engine's nec2 dialect reads NEC-2's GH (turn spacing, total length, four helix "
                   "radii, wire radius)");
    ll n = count_of(n_seg, kMaxWires, "segments on one helix");
    double zinc = std::fabs(hl) / static_cast<double>(n);
    std::vector<std::array<double, 3>> v;
    double zk = 0.0;
    for (ll k = 0; k <= n; k++) {
        if (k) zk = zk + zinc;
        double angle = 2.0 * kPi * zk / s;
        double x, y;
        if (a2 == a1) {
            x = a1 * libm_cos(angle);
            y = (b1 == 0.0 ? a1 : b1) * libm_sin(angle);
        } else {
            double b_end = b2 == 0.0 ? a2 : b2;
            x = (a1 + (a2 - a1) * zk / std::fabs(hl)) * libm_cos(angle);
            y = (b1 + (b_end - b1) * zk / std::fabs(hl)) * libm_sin(angle);
        }
        if (hl > 0.0)
            v.push_back({x, y, zk});
        else
            v.push_back({y, x, zk});
    }
    chords(card.i(0), v, std::vector<double>(static_cast<std::size_t>(n), card.f(8)), wires);
}

// `deck._nec4.helix_vertices`
std::vector<std::array<double, 3>> helix_points(const Card &card) {
    Int n_seg = card.i(1);
    double turns = card.f(2), zlen = card.f(3), r1 = card.f(4), r2 = card.f(5);
    if (n_seg < Int(1)) deck_error("GH segment count must be >= 1, got " + S(n_seg));
    if (turns == 0.0)
        deck_error(
            "GH with zero turns is not served by this engine's nec4 dialect, which reads "
            "NEC-4.2's GH: F1 is the number of turns, and no NEC-4.2 run shows a helix of none");
    if (r1 <= 0.0 || r2 <= 0.0)
        deck_error("GH needs a positive helix radius at both ends (F3, F4), got " + g(r1) + " and " +
                   g(r2));
    if (card.f(6) <= 0.0 || card.f(7) <= 0.0)
        deck_error("GH needs a positive wire radius at both ends (F5, F6), got " + g(card.f(6)) +
                   " and " + g(card.f(7)) +
                   "; this engine's nec4 dialect reads NEC-4.2's GH (turns, length, two helix "
                   "radii, two wire radii), not NEC-2's");
    ll n = count_of(n_seg, kMaxWires, "segments on one helix");
    double dn = static_cast<double>(n);
    double q = py_float_pow(r2 / r1, 1.0 / dn);
    double step = 2.0 * kPi * turns / dn;
    double qn = py_float_pow(q, dn);
    std::vector<std::array<double, 3>> v;
    for (ll k = 0; k <= n; k++) {
        double dk = static_cast<double>(k);
        double qk = py_float_pow(q, dk);
        double rho = r1 * qk;
        double z = q == 1.0 ? zlen * dk / dn : zlen * (1.0 - qk) / (1.0 - qn);
        double a = dk * step;
        v.push_back({rho * libm_cos(a), rho * libm_sin(a), z});
    }
    return v;
}

// `helix_wire_radii` and `_gh`
void gh(const Card &card, std::vector<FlatWire> &wires) {
    const std::vector<std::array<double, 3>> v = helix_points(card);
    const ll n = static_cast<ll>(v.size()) - 1;
    double w1 = card.f(6), w2 = card.f(7);
    std::vector<double> radii;
    if (n == 1) {
        radii.push_back(w1);
    } else {
        for (ll k = 0; k < n; k++)
            radii.push_back(
                w1 * py_float_pow(w2 / w1, static_cast<double>(k) / static_cast<double>(n - 1)));
    }
    chords(card.i(0), v, radii, wires);
}

// `deck._nec4._catenary` (the NEC-5 reader's copy, which both dialects share)
std::vector<std::array<double, 3>> catenary_points(const Card &card, double *length) {
    std::vector<std::array<double, 3>> v = nec5::catenary_points(card, kMaxWires, length);
    if (v.empty()) decline("too many segments on one catenary");
    return v;
}

// `_cw`
void cw(const Card &card, std::vector<FlatWire> &wires) {
    double length = 0.0;
    const std::vector<std::array<double, 3>> v = catenary_points(card, &length);
    chords(card.i(0), v, std::vector<double>(v.size() - 1, card.f(8)), wires);
}

struct Plane {
    int axis;
    ll flag;
    const char *name;
};

std::array<Plane, 3> planes(const Card &card) {
    ll code = to_ll(card.i(1), "GX's plane code");
    ll ix = pymod(floordiv(code, 100), 10), iy = pymod(floordiv(code, 10), 10), iz = pymod(code, 10);
    return {Plane{2, iz, "Z=0"}, Plane{1, iy, "Y=0"}, Plane{0, ix, "X=0"}};
}

void reflect(std::vector<FlatWire> &wires, int axis, const Int &iti, const char *plane) {
    std::vector<FlatWire> images;
    for (const FlatWire &w : wires) {
        double e1 = w.p1[axis], e2 = w.p2[axis];
        if (std::fabs(e1) + std::fabs(e2) <= 1.0e-5 || e1 * e2 < -1.0e-6)
            deck_error(std::string("a wire lies in or crosses the ") + plane + " symmetry plane");
        FlatWire q = w;
        q.p1[axis] = -e1;
        q.p2[axis] = -e2;
        q.tag = next_tag(w.tag, iti);
        images.push_back(q);
    }
    wires.insert(wires.end(), images.begin(), images.end());
    check_wire_count(wires);
}

void gx(const Card &card, std::vector<FlatWire> &wires) {
    Int iti = card.i(0);
    for (const Plane &p : planes(card)) {
        if (p.flag) {
            reflect(wires, p.axis, iti, p.name);
            iti = iti + iti;
        }
    }
}

void gr(const Card &card, std::vector<FlatWire> &wires) {
    Int itg = card.i(0), nop = card.i(1);
    if (nop < Int(1)) deck_error("structure count must be >= 1, got " + S(nop));
    ll n = count_of(nop, kMaxWires, "GR copies");
    double sam = 2.0 * kPi / static_cast<double>(n);
    double cs = libm_cos(sam), ss = libm_sin(sam);
    std::vector<FlatWire> block = wires;
    if (!block.empty() && n - 1 > kMaxWires / static_cast<ll>(block.size())) decline("too many wires");
    for (ll r = 0; r < n - 1; r++) {
        std::vector<FlatWire> next;
        for (const FlatWire &w : block) {
            FlatWire c;
            c.tag = next_tag(w.tag, itg);
            c.n_seg = w.n_seg;
            c.p1[0] = w.p1[0] * cs - w.p1[1] * ss;
            c.p1[1] = w.p1[0] * ss + w.p1[1] * cs;
            c.p1[2] = w.p1[2];
            c.p2[0] = w.p2[0] * cs - w.p2[1] * ss;
            c.p2[1] = w.p2[0] * ss + w.p2[1] * cs;
            c.p2[2] = w.p2[2];
            c.radius = w.radius;
            next.push_back(c);
        }
        block = next;
        wires.insert(wires.end(), block.begin(), block.end());
        check_wire_count(wires);
    }
}

// `_symmetry_after`
void symmetry_after(const Card &card, const std::vector<FlatWire> &wires, Symmetry before,
                    bool &has, Symmetry &symmetry) {
    const std::string &m = card.mnemonic;
    if (m == "GX") {
        bool any = false;
        for (const Plane &p : planes(card)) any = any || p.flag;
        has = any;
        if (any) symmetry = before;
        return;
    }
    if (m == "GR") {
        has = true;
        symmetry = before;
        return;
    }
    if (m == "GS") return;
    if (m == "GW" || m == "GA" || m == "GH" || m == "CW") {
        has = false;
        return;
    }
    // GM
    if (!card.i(1).is_zero()) {
        has = false;
        return;
    }
    if (first_wire_with_tag(wires, py_int_trunc(card.f(8) + 0.5)) != 0) has = false;
}

// --- connections -------------------------------------------------------------

typedef std::array<ll, 3> Key3;

ll cell_index(double v) {
    double f = std::floor(v);
    if (!(std::fabs(f) < 4.0e18)) decline("a coordinate past the connection grid's range");
    return static_cast<ll>(f);
}

void boundary_point(const double *a, const double *b, ll k, ll n, double *out) {
    double t = static_cast<double>(k) / static_cast<double>(n);
    for (int c = 0; c < 3; c++) out[c] = a[c] + (b[c] - a[c]) * t;
}

void snap_connections(std::vector<FlatWire> &wires) {
    std::vector<double> end_seg_len;
    std::vector<std::array<double, 3>> pts;
    for (const FlatWire &w : wires) {
        double seg = w.length() / static_cast<double>(w.n_seg);
        for (const double *p : {w.p1, w.p2}) {
            end_seg_len.push_back(seg);
            pts.push_back({p[0], p[1], p[2]});
        }
    }
    if (pts.empty()) return;
    double max_tol = kSmin * *std::max_element(end_seg_len.begin(), end_seg_len.end());
    if (max_tol <= 0.0) return;

    auto cell = [&](const double *p) -> Key3 {
        return {cell_index(p[0] / max_tol), cell_index(p[1] / max_tol), cell_index(p[2] / max_tol)};
    };
    auto neighborhood = [&](const std::map<Key3, std::vector<std::size_t>> &grid, const double *p,
                            std::vector<std::size_t> &out) {
        out.clear();
        Key3 c = cell(p);
        for (ll dx = -1; dx <= 1; dx++)
            for (ll dy = -1; dy <= 1; dy++)
                for (ll dz = -1; dz <= 1; dz++) {
                    auto it = grid.find(Key3{c[0] + dx, c[1] + dy, c[2] + dz});
                    if (it != grid.end()) out.insert(out.end(), it->second.begin(), it->second.end());
                }
    };

    // Pass 1: endpoint clusters, union-find under the per-pair tolerance.
    std::map<Key3, std::vector<std::size_t>> grid;
    for (std::size_t k = 0; k < pts.size(); k++) grid[cell(pts[k].data())].push_back(k);
    std::vector<std::size_t> parent(pts.size());
    for (std::size_t k = 0; k < parent.size(); k++) parent[k] = k;
    auto find = [&](std::size_t x) {
        while (parent[x] != x) {
            parent[x] = parent[parent[x]];
            x = parent[x];
        }
        return x;
    };
    std::vector<std::size_t> near;
    for (std::size_t k = 0; k < pts.size(); k++) {
        neighborhood(grid, pts[k].data(), near);
        for (std::size_t j : near) {
            if (j <= k) continue;
            if (l1(pts[k].data(), pts[j].data()) <= kSmin * std::min(end_seg_len[k], end_seg_len[j])) {
                std::size_t rk = find(k);
                parent[find(j)] = rk;
            }
        }
    }
    for (std::size_t k = 0; k < pts.size(); k++) {
        std::size_t root = find(k);
        if (root != k) {
            FlatWire &w = wires[k / 2];
            double *dst = (k % 2 == 0) ? w.p1 : w.p2;
            for (int c = 0; c < 3; c++) dst[c] = pts[root][c];
            pts[k] = pts[root];
        }
    }

    // Pass 2: endpoints onto OTHER wires' interior segment boundaries.
    std::map<Key3, std::vector<std::size_t>> bgrid;
    std::vector<std::pair<std::size_t, std::array<double, 3>>> boundaries;
    for (std::size_t wi = 0; wi < wires.size(); wi++) {
        const FlatWire &w = wires[wi];
        for (ll k = 1; k < w.n_seg; k++) {
            std::array<double, 3> b;
            boundary_point(w.p1, w.p2, k, w.n_seg, b.data());
            bgrid[cell(b.data())].push_back(boundaries.size());
            boundaries.push_back({wi, b});
        }
    }
    for (std::size_t k = 0; k < pts.size(); k++) {
        std::size_t wi = k / 2;
        const double *p = pts[k].data();
        bool have = false;
        double best_d = 0;
        std::array<double, 3> best_b{};
        neighborhood(bgrid, p, near);
        for (std::size_t j : near) {
            std::size_t bw = boundaries[j].first;
            const std::array<double, 3> &b = boundaries[j].second;
            if (bw == wi || (b[0] == p[0] && b[1] == p[1] && b[2] == p[2])) continue;
            double d = l1(p, b.data());
            double seg_b = end_seg_len[2 * bw];
            if (d <= kSmin * std::min(end_seg_len[k], seg_b) && (!have || d < best_d)) {
                have = true;
                best_d = d;
                best_b = b;
            }
        }
        if (have) {
            double *dst = (k % 2 == 0) ? wires[wi].p1 : wires[wi].p2;
            for (int c = 0; c < 3; c++) dst[c] = best_b[c];
        }
    }
}

ll node_key(double v) {
    double r = std::nearbyint(v / kNodeEps);
    if (!(std::fabs(r) < 4.0e18)) decline("a coordinate past the node key's range");
    return static_cast<ll>(r);
}

std::map<std::size_t, std::set<ll>> junction_cuts(const std::vector<FlatWire> &wires) {
    std::map<Key3, std::set<std::size_t>> owners;
    auto key = [](const double *p) { return Key3{node_key(p[0]), node_key(p[1]), node_key(p[2])}; };
    double b[3];
    for (std::size_t i = 0; i < wires.size(); i++) {
        const FlatWire &w = wires[i];
        for (ll k = 0; k <= w.n_seg; k++) {
            boundary_point(w.p1, w.p2, k, w.n_seg, b);
            owners[key(b)].insert(i);
        }
    }
    std::map<std::size_t, std::set<ll>> cuts;
    for (std::size_t i = 0; i < wires.size(); i++) {
        const FlatWire &w = wires[i];
        std::set<ll> shared;
        for (ll k = 1; k < w.n_seg; k++) {
            boundary_point(w.p1, w.p2, k, w.n_seg, b);
            if (owners[key(b)].size() > 1) shared.insert(k);
        }
        if (!shared.empty()) cuts[i] = shared;
    }
    return cuts;
}

void shatter(Structure &s, const std::map<std::size_t, std::set<ll>> &cuts) {
    for (std::size_t i = 0; i < s.wires.size(); i++) {
        const FlatWire &w = s.wires[i];
        std::vector<ll> indices;
        auto it = cuts.find(i);
        if (it == cuts.end()) {
            indices.push_back(static_cast<ll>(s.pieces.size()));
            Piece p;
            p.wire = static_cast<ll>(i);
            p.first_seg = 0;
            p.n_seg = w.n_seg;
            for (int c = 0; c < 3; c++) {
                p.p1[c] = w.p1[c];
                p.p2[c] = w.p2[c];
            }
            s.pieces.push_back(p);
        } else {
            ll prev = 0;
            std::vector<ll> ends(it->second.begin(), it->second.end());
            ends.push_back(w.n_seg);
            for (ll b : ends) {
                indices.push_back(static_cast<ll>(s.pieces.size()));
                Piece p;
                p.wire = static_cast<ll>(i);
                p.first_seg = prev;
                p.n_seg = b - prev;
                boundary_point(w.p1, w.p2, prev, w.n_seg, p.p1);
                boundary_point(w.p1, w.p2, b, w.n_seg, p.p2);
                s.pieces.push_back(p);
                prev = b;
            }
        }
        s.pieces_of_wire.push_back(indices);
    }
}

bool finite3(const double *p) { return std::isfinite(p[0]) && std::isfinite(p[1]) && std::isfinite(p[2]); }

Structure build_geometry(const std::vector<Card> &cards, Dialect dialect) {
    Structure s;
    std::vector<FlatWire> &wires = s.wires;
    for (const Card &card : cards) {
        if (card.mnemonic == "GE") {
            s.ground_plane_flag = !card.i(0).is_zero();
            s.ground_plane_interpolates = card.i(0) > Int(0);
            continue;
        }
        Symmetry before;
        before.cell_wires = static_cast<ll>(wires.size());
        for (const FlatWire &w : wires) before.cell_segments += w.n_seg;
        const std::string &m = card.mnemonic;
        if (m == "GW") {
            gw(card, wires);
        } else if (m == "GM") {
            gm(card, wires);
        } else if (m == "GS") {
            gs(card, wires);
        } else if (m == "GX") {
            gx(card, wires);
        } else if (m == "GA") {
            ga(card, wires, dialect);
        } else if (m == "GH") {
            if (dialect == Dialect::nec4)
                gh(card, wires);
            else
                gh_nec2(card, wires);
        } else if (m == "CW") {
            cw(card, wires);
        } else {
            gr(card, wires);
        }
        symmetry_after(card, wires, before, s.has_symmetry, s.symmetry);
    }
    ll total = 0;
    for (const FlatWire &w : wires) {
        if (!finite3(w.p1) || !finite3(w.p2) || !std::isfinite(w.radius) || !std::isfinite(w.length()))
            decline("a non-finite coordinate, radius or length after the geometry transforms");
        total += w.n_seg;
        if (total > kMaxSegments) decline("too many segments");
    }
    snap_connections(wires);
    std::map<std::size_t, std::set<ll>> cuts = junction_cuts(wires);
    shatter(s, cuts);
    return s;
}

// ============================================================================
// The parser (`deck/_nec2.py`'s `_Nec2Parser`, `deck/_nec4.py`'s deltas)
// ============================================================================

const char *const kGeometryCards[] = {"GW", "GA", "GH", "GM", "GS", "GX", "GR", "GE"};
const char *const kExecuteCards[] = {"XQ", "RP", "NE", "NH"};
const char *const kArmingCards[] = {"EX", "FR", "LD", "GN", "EK", "TL", "NT"};
const char *const kOperatorCards[] = {"GN", "EK", "LD"};
const char *const kNetworkTransparent[] = {"PT", "PQ", "MP"};

template <std::size_t N>
bool in(const std::string &m, const char *const (&set)[N]) {
    for (const char *s : set)
        if (m == s) return true;
    return false;
}

const std::map<std::string, std::string> &refused_by_name() {
    static const std::map<std::string, std::string> table = {
        {"GC", "GC (tapered wire continuation) is not part of this engine's nec2 dialect"},
        {"GF", "GF (numerical Green's function) is not part of this engine's nec2 dialect"},
        {"SY", "SY (4nec2 symbolic variables) is not part of this engine's nec2 dialect"},
        {"SP", "SP (surface patch) is not supported by this engine yet"},
        {"SM", "SM (multiple-patch surface) is not supported by this engine yet"},
        {"SC", "SC (surface patch continuation) is not supported by this engine yet"},
        {"KH", "KH (interaction approximation limit) is not supported by this engine"},
        {"CP", "CP (coupling request) is not supported by this engine"},
        {"PL", "PL (plot request) is not supported by this engine"},
        {"WG", "WG (NGF write request) is not supported by this engine"},
        {"ZO", "ZO (impedance normalisation) is not supported by this engine"},
    };
    return table;
}

const char *const kRefuseNearFieldReflCoef =
    " over the GN 0 reflection-coefficient ground is not supported by this engine (a reflection "
    "coefficient is a far-field construction); GN 2, the Sommerfeld ground, serves the near field";

struct Source {
    Int tag, seg;
    std::complex<double> volts;
};

struct MaterialState {
    bool has_global_conductivity = false;
    double global_conductivity = 0;
    std::map<ll, double> wire_conductivity;
    bool has_global_distributed = false;
    DistributedRLC global_distributed;
    std::map<ll, DistributedRLC> wire_distributed;
    bool operator==(const MaterialState &o) const {
        return has_global_conductivity == o.has_global_conductivity &&
               (!has_global_conductivity || global_conductivity == o.global_conductivity) &&
               wire_conductivity == o.wire_conductivity &&
               has_global_distributed == o.has_global_distributed &&
               (!has_global_distributed || global_distributed == o.global_distributed) &&
               wire_distributed == o.wire_distributed;
    }
    bool operator!=(const MaterialState &o) const { return !(*this == o); }
};

struct Pending {
    std::vector<Source> sources;
    std::vector<double> frequencies;
    Request request;
    bool has_print_control = false;
    PrintControl print_control;
    bool extended_kernel = false;
    bool has_multiprocessing = false;
    Int multiprocessing[2];
    Environment environment;
    bool refilled = false, refilled_partial = false;
    std::vector<ll> loads;
    MaterialState materials;
};

struct LogEntry {
    ll piece = 0;
    double arclength = 0;
    LoadSpec spec;
    bool operator==(const LogEntry &o) const {
        return piece == o.piece && arclength == o.arclength && spec == o.spec;
    }
};

// `_directive(text, keyword)`: true with `value` set, false for None.
bool directive(const std::string &latin1_text, const char *keyword, Int &value) {
    std::vector<std::string> parts = nec5::latin1_split(latin1_text);
    for (std::size_t i = 0; i < parts.size(); i++) {
        if (nec5::latin1_upper(parts[i]) == keyword && i + 1 < parts.size()) {
            double v;
            if (!nec5::latin1_float(parts[i + 1], v)) return false;
            if (std::isnan(v)) return false;  // int(nan) is the ValueError caught
            if (std::isinf(v))
                throw ParseError(ErrorKind::overflow_error, "cannot convert float infinity to integer");
            value = py_int_trunc(v);
            return true;
        }
    }
    return false;
}

class Parser {
  public:
    explicit Parser(Dialect d) : dialect_(d) {
        frequencies_ = {kDefaultFrequencyMHz};
        extended_kernel_ = d == Dialect::nec4;
    }

    void feed(const std::string &text) {
        for (const std::string &line : nec5::latin1_splitlines(text)) {
            std::string stripped = nec5::latin1_strip(line);
            std::replace(stripped.begin(), stripped.end(), ',', ' ');
            std::vector<std::string> head = nec5::latin1_split(stripped);
            if (!head.empty()) {
                auto hit = refused_by_name().find(nec5::latin1_upper(head[0]));
                if (hit != refused_by_name().end()) deck_error(hit->second);
            }
            Card card;
            if (!nec5::read_card(line, card)) continue;
            if (card.mnemonic == "NX" || card.mnemonic == "EN") break;
            // `_Nec2Parser.card`: a malformed GW refuses on the card, in the
            // words every reader uses; any other card's non-finite field is
            // Python's to read (an LD 5 NaN conductivity is served)
            if (card.mnemonic == "GW") {
                const std::string why = nec5::gw_field_refusal(card);
                if (!why.empty()) deck_error(why);
            }
            if (card.mnemonic != "CM" && card.mnemonic != "CE") {
                for (double v : card.values)
                    if (!std::isfinite(v)) decline("a non-finite field on a " + card.mnemonic + " card");
            }
            this->card(card);
        }
        structure();
    }

    Model model();

  private:
    Dialect dialect_;
    std::vector<std::string> comments_;  // UTF-8
    bool quiet_ = false;
    bool has_reduced_field_ = false;
    Int reduced_field_;

    std::vector<Card> geometry_cards_;
    bool structure_built_ = false;
    Structure structure_;

    std::vector<bool> ran_;
    std::vector<Pending> groups_;  // parallel to ran_

    bool armed_ = true;
    ll executed_ = 0;
    bool fresh_fr_ = false;
    bool has_print_control_ = false;
    PrintControl print_control_;
    bool has_multiprocessing_ = false;
    Int multiprocessing_[2];
    bool saw_ex_ = false;
    bool operator_dirty_ = false;

    std::vector<Source> sources_;
    bool sources_stale_ = false;
    std::vector<double> frequencies_;
    Ground ground_;
    bool has_second_medium_ = false;
    SecondMedium second_medium_;
    bool extended_kernel_ = false;

    std::vector<NetworkCard> networks_;
    bool has_destroyer_ = false;
    std::string destroyer_;

    std::vector<LogEntry> load_log_;
    std::vector<ll> live_;
    std::set<Pair> loaded_;
    std::map<Pair, LoadSpec> load_spec_at_;
    MaterialState materials_;
    std::map<ll, std::pair<double, double>> wire_insulation_;  // (radius, eps_r)

    // nec4: (tag, seg) -> the EX type that address was driven with
    std::vector<std::pair<std::pair<Int, Int>, ll>> source_kind_;

    const Structure &structure() {
        if (!structure_built_) {
            structure_ = build_geometry(geometry_cards_, dialect_);
            structure_built_ = true;
        }
        return structure_;
    }

    void geometry(const Card &card) {
        if (dialect_ == Dialect::nec4 && card.mnemonic == "GE") {
            Int i1 = card.i(0);
            if (!(i1 == Int(-1) || i1.is_zero() || i1 == Int(1)))
                deck_error("GE " + S(i1) +
                           " is not a NEC-4.2 ground flag; this engine's nec4 dialect reads GE -1 "
                           "(structure below ground), GE 0 (no ground) and GE 1 (structure above "
                           "ground)");
            Int i2 = card.i(1);
            if (!(i2.is_zero() || i2 == Int(-1)))
                deck_error("GE with a second field of " + S(i2) +
                           " is not served by this engine's nec4 dialect; EZNEC's NEC-4.2 slot writes "
                           "-1 there, and no capture shows what any other value means");
        }
        geometry_cards_.push_back(card);
        structure_built_ = false;
        if (card.mnemonic == "GE") structure();
    }

    // -- GN / GD
    void gn(Card card) {
        if (dialect_ == Dialect::nec4) {
            card.has_trailer = false;
            card.trailer.clear();
            Int code = card.i(0);
            if (code == Int(3)) {
                if (!card.i(1).is_zero())
                    deck_error("GN 3 with a " + S(card.i(1)) +
                               "-wire radial ground screen is not supported by this engine");
                if (!card.values.empty()) card.values[0] = 2.0;
            }
            Int now = card.i(0);
            if ((now.is_zero() || now == Int(2)) && card.f(4) <= 0.0)
                deck_error("GN " + S(code) +
                           " needs a positive relative permittivity (field 5) for a finite ground, "
                           "and this card's reads " +
                           g(card.f(4)));
        }
        if (card.has_trailer)
            deck_error("GN carries a trailing token " + repr_latin1(card.trailer) +
                       "; this engine's nec2 dialect has no Sommerfeld-table file field on GN "
                       "(momwire#1084 adds that field to the nec5 dialect only)");
        Int code = card.i(0);
        Int nradl = card.i(1);
        if ((code.is_zero() || code == Int(2)) && !nradl.is_zero())
            deck_error("GN " + S(code) + " with a " + S(nradl) +
                       "-wire radial ground screen is not supported by this engine");
        if (code == Int(-1)) {
            ground_ = Ground();
        } else if (code == Int(1)) {
            ground_ = Ground();
            ground_.kind = Ground::pec;
        } else if (code.is_zero()) {
            ground_ = Ground();
            ground_.kind = Ground::finite_fast;
            ground_.eps_r = card.f(4);
            ground_.sigma = card.f(5);
        } else if (code == Int(2)) {
            ground_ = Ground();
            ground_.kind = Ground::finite;
            ground_.eps_r = card.f(4);
            ground_.sigma = card.f(5);
        } else {
            deck_error("GN type " + S(code) + " is not supported by this engine");
        }
        if (nradl.is_zero()) {
            has_second_medium_ = true;
            second_medium_ = SecondMedium{card.f(6), card.f(7), card.f(8), card.f(9)};
        }
    }

    void gd(const Card &card) {
        has_second_medium_ = true;
        second_medium_ = SecondMedium{card.f(4), card.f(5), card.f(6), card.f(7)};
    }

    // -- FR / EX
    void fr(const Card &card) {
        Int sweep_type = card.i(0);
        ll n = count_of(imax(card.i(1), Int(1)), kMaxCount, "frequencies");
        double start = card.f(4), step = card.f(5);
        frequencies_.clear();
        for (ll i = 0; i < n; i++) {
            if (sweep_type.is_zero()) {
                frequencies_.push_back(start + static_cast<double>(i) * step);
            } else if (step > 0) {
                frequencies_.push_back(start * py_pow(step, i));
            } else {
                frequencies_.push_back(start);
            }
        }
        fresh_fr_ = true;
    }

    // Python's `float ** int` for a finite positive base: libm's pow, and an
    // OverflowError where it overflows.
    static double py_pow(double x, ll i) {
        if (i == 0) return 1.0;
        if (x == 1.0) return 1.0;
        double r = std::pow(x, static_cast<double>(i));
        if (std::isinf(r))
            // PyErr_SetFromErrno's (errno, strerror): the C library's words,
            // "Numerical result out of range" on glibc, "Result too large" on
            // Apple's libc and the UCRT
            throw ParseError(ErrorKind::overflow_error,
                             "(" + std::to_string(ERANGE) + ", '" + std::strerror(ERANGE) + "')");
        return r;
    }

    void add_source(const Int &tag, const Int &seg, std::complex<double> v) {
        saw_ex_ = true;
        if (sources_stale_) {
            sources_.clear();
            sources_stale_ = false;
        }
        sources_.push_back(Source{tag, seg, v});
    }

    void ex2(const Card &card) {
        Int ex_type = card.i(0);
        if (!ex_type.is_zero())
            deck_error("EX type " + S(ex_type) +
                       " is not a voltage source; this engine drives EX 0 only");
        saw_ex_ = true;
        if (sources_stale_) {
            sources_.clear();
            sources_stale_ = false;
        }
        Int tag = card.i(1), seg = card.i(2);
        std::complex<double> voltage(card.f(4), card.f(5));
        if (voltage.real() == 0 && voltage.imag() == 0) voltage = std::complex<double>(1.0, 0.0);
        sources_.push_back(Source{tag, seg, voltage});
    }

    void ex(const Card &card) {
        if (dialect_ == Dialect::nec2) {
            ex2(card);
            return;
        }
        Int ex_type = card.i(0);
        if (ex_type == Int(4))
            deck_error(
                "EX 4 is not served by this engine's nec4 dialect: NEC-4.2 reads EX 4 as a current "
                "element at the point its first three real fields give, tied to no segment, not as "
                "NEC-5's EX 4 current source on a segment.  For a current source on a segment, "
                "write EX 6 with that tag and segment and the current in amperes in its first two "
                "real fields (EX 0 drives a voltage source)");
        if (!(ex_type.is_zero() || ex_type == Int(6)))
            deck_error("EX type " + S(ex_type) +
                       " is not served by this engine's nec4 dialect, which drives EX 0 (voltage "
                       "source) and EX 6 (current source) only");
        Int tag = card.i(1), seg = card.i(2);
        ll type = ex_type.is_zero() ? 0 : 6;
        ll kind = type;
        bool found = false;
        for (const auto &sk : source_kind_)
            if (sk.first.first == tag && sk.first.second == seg) {
                kind = sk.second;
                found = true;
                break;
            }
        if (!found) source_kind_.push_back({{tag, seg}, type});
        if (kind != type)
            deck_error("EX " + S(ex_type) + " on tag " + S(tag) + " segment " + S(seg) +
                       ", which an earlier EX " + S(kind) +
                       " drove, is not supported by this engine: one segment is one port for the "
                       "whole deck, and a port is either a voltage source or a current source");
        if (type == 0) {
            ex2(card);
            return;
        }
        std::complex<double> amps(card.f(4), card.f(5));
        if (amps.real() == 0 && amps.imag() == 0)
            deck_error("EX 6 on tag " + S(tag) + " segment " + S(seg) +
                       " sets a current of zero, which is an open circuit rather than a source; "
                       "this engine refuses it rather than guess a drive");
        add_source(tag, seg, amps);
    }

    void ek(const Card &card) {
        if (dialect_ == Dialect::nec4) return;  // ignored: NEC-4.2 has no EK switch
        extended_kernel_ = card.i(0) != Int(-1);
    }

    // -- LD
    bool load_spec(ll ldtyp, const Card &card, LoadSpec &spec) {
        if (ldtyp == 0 || ldtyp == 1) {
            double r = card.f(4), l = card.f(5), c = card.f(6);
            if (r == 0.0 && l == 0.0 && c == 0.0) return false;
            spec = LoadSpec{ldtyp == 0 ? "series" : "parallel", r, l, c, 0.0};
            return true;
        }
        double r = card.f(4), x = card.f(5);
        if (r == 0.0 && x == 0.0) return false;
        spec = LoadSpec{"fixed", r, 0.0, 0.0, x};
        return true;
    }

    std::vector<Pair> cell_rule(const std::vector<Pair> &pairs, const LoadSpec *spec) {
        const Structure &st = structure();
        auto ad = st.under_the_cell_rule(pairs);
        if (ad.second.empty()) return ad.first;
        if (spec && restates(ad.second, *spec)) return ad.first;
        ll n = static_cast<ll>(ad.second.size());
        deck_error("LD addresses " + S(n) + " segment" + (n == 1 ? "" : "s") +
                   " outside the GX/GR symmetric cell in force (the cell is segments 1-" +
                   S(st.symmetry.cell_segments) + " of " + S(st.n_segments()) +
                   "), which this engine does not serve (momwire#415): while the symmetry is live "
                   "the matrix is the cell's, so NEC silently drops such a card rather than loading "
                   "the copy — address the cell instead, where a load applies to every copy of it, "
                   "or write the GX/GR out as explicit GW cards");
    }

    bool restates(const std::vector<Pair> &dropped, const LoadSpec &spec) {
        const Structure &st = structure();
        for (const Pair &p : dropped) {
            Pair key = st.element_of(p.first, p.second);
            auto it = load_spec_at_.find(key);
            if (it == load_spec_at_.end() || it->second != spec) return false;
        }
        return true;
    }

    void ld(const Card &card) {
        if (dialect_ == Dialect::nec4) {
            Int t = card.i(0);
            if ((t == Int(2) || t == Int(3)) && card.f(6) != 0.0)
                deck_error("LD " + S(t) + " asks for a capacitance of " + g(card.f(6)) +
                           " in its per-unit-length RLC, which this engine's nec4 dialect does not "
                           "serve: NEC-2 scales that field by the segment length, and no NEC-4.2 run "
                           "shows whether NEC-4.2 does the same");
        }
        Int ldtyp_i = card.i(0);
        if (ldtyp_i == Int(-1)) {
            live_.clear();
            loaded_.clear();
            load_spec_at_.clear();
            materials_ = MaterialState();
            return;
        }
        bool known = false;
        for (int t = 0; t <= 5; t++) known = known || ldtyp_i == Int(t);
        if (!known) deck_error("LD type " + S(ldtyp_i) + " is not supported by this engine");
        ll ldtyp = ldtyp_i.i64();
        Int tag = card.i(1), first = card.i(2), last = card.i(3);
        if (ldtyp == 2 || ldtyp == 3) {
            ld23(ldtyp, tag, first, last, card);
            return;
        }
        if (ldtyp == 5) {
            ld5(tag, first, last, card.f(4), card.f(5));
            return;
        }
        const Structure &st = structure();
        std::vector<Pair> pairs = st.ld_segment_range(tag, first, last);
        if (static_cast<ll>(pairs.size()) > kLdExpandMax)
            deck_error("LD over " + S(static_cast<ll>(pairs.size())) +
                       " segments is not supported by this engine — at most " + S(kLdExpandMax) +
                       " segments expand into per-segment loads");
        LoadSpec spec;
        if (!load_spec(ldtyp, card, spec)) return;
        for (const Pair &p : cell_rule(pairs, &spec)) {
            const Structure &s2 = structure();
            Pair key = s2.element_of(p.first, p.second);
            if (loaded_.count(key))
                deck_error(
                    "LD on a segment that already carries a load is not supported by this engine — a "
                    "second load on one segment is not merged");
            loaded_.insert(key);
            load_spec_at_[key] = spec;
            std::pair<ll, double> ra = s2.resolve_of(p.first, p.second);
            LogEntry entry{key.first, ra.second, spec};
            ll index = -1;
            for (std::size_t k = 0; k < load_log_.size(); k++)
                if (load_log_[k] == entry) {
                    index = static_cast<ll>(k);
                    break;
                }
            if (index < 0) {
                index = static_cast<ll>(load_log_.size());
                load_log_.push_back(entry);
            }
            live_.push_back(index);
        }
    }

    std::map<ll, std::set<ll>> by_wire(const std::vector<Pair> &pairs) {
        std::map<ll, std::set<ll>> out;
        for (const Pair &p : pairs) out[p.first].insert(p.second);
        return out;
    }

    bool whole(const std::map<ll, std::set<ll>> &bw) {
        const Structure &st = structure();
        for (const auto &kv : bw)
            if (static_cast<ll>(kv.second.size()) != st.wires[static_cast<std::size_t>(kv.first)].n_seg)
                return false;
        return true;
    }

    void ld23(ll ldtyp, const Int &tag, const Int &first, const Int &last, const Card &card) {
        double r = card.f(4), l = card.f(5), c = card.f(6);
        if (r < 0.0 || l < 0.0 || c < 0.0) {
            const char *name = r < 0.0 ? "resistance" : l < 0.0 ? "inductance" : "capacitance";
            deck_error("LD " + S(ldtyp) + " asks for a negative per-unit-length " + name + " (" + g(r) +
                       ", " + g(l) + ", " + g(c) +
                       "); a passive distributed loading is non-negative and this engine refuses it "
                       "at the card");
        }
        if (r == 0.0 && l == 0.0 && c == 0.0) return;
        DistributedRLC spec{ldtyp == 2 ? "series" : "parallel", r, l, c};
        if (tag.is_zero() && first.is_zero()) {
            materials_.has_global_distributed = true;
            materials_.global_distributed = spec;
            return;
        }
        std::vector<Pair> pairs = cell_rule(structure().ld_segment_range(tag, first, last), nullptr);
        std::map<ll, std::set<ll>> bw = by_wire(pairs);
        if (!whole(bw))
            deck_error("LD " + S(ldtyp) +
                       " per-unit-length loading on a partial-wire segment range is not supported by "
                       "this engine — a distributed RLC is a wire property and covers whole wires "
                       "only");
        for (const auto &kv : bw) materials_.wire_distributed[kv.first] = spec;
    }

    void ld5(const Int &tag, const Int &first, const Int &last, double sigma, double mu) {
        if (mu != 0.0 && mu != 1.0)
            deck_error("LD 5 asks for a relative permeability of " + g(mu) +
                       "; this engine's wire internal-impedance model has no permeability parameter "
                       "(`wire_internal_impedance` hard-codes vacuum permeability) and refuses a "
                       "non-unity value rather than silently modelling it as copper");
        if (tag.is_zero() && first.is_zero()) {
            materials_.has_global_conductivity = true;
            materials_.global_conductivity = sigma;
            return;
        }
        std::vector<Pair> pairs = cell_rule(structure().ld_segment_range(tag, first, last), nullptr);
        std::map<ll, std::set<ll>> bw = by_wire(pairs);
        if (!whole(bw))
            deck_error(
                "LD 5 conductivity on a partial-wire segment range is not supported by this engine — "
                "per-wire conductivity covers whole wires only");
        for (const auto &kv : bw) materials_.wire_conductivity[kv.first] = sigma;
    }

    // -- IS
    void is_card(const Card &card) {
        if (executed_)
            deck_error(
                "IS after an execute request is not supported by this engine: wire insulation is part "
                "of the structure, so it cannot change between runs");
        Int tag = card.i(1), first = card.i(2), last = card.i(3);
        double eps_r = card.f(4), sigma = card.f(5), radius = card.f(6);
        if (sigma != 0.0)
            deck_error(
                "IS with a conductive sheath (F2 != 0) is not modelled by this engine — the "
                "insulation jacket is a lossless dielectric; set the sheath conductivity to 0");
        if (radius <= 0.0 || eps_r <= 1.0) return;
        const Structure &st = structure();
        if (st.has_symmetry)
            deck_error(
                "IS while a GX/GR symmetric cell is in force is not supported by this engine "
                "(momwire#415): a sheath moves the matrix the way a load does, so the cell rule "
                "applies to it, but NEC-2 has no IS card to measure that rule against — write the "
                "GX/GR out as explicit GW cards");
        std::map<ll, std::set<ll>> bw = by_wire(st.ld_segment_range(tag, first, last));
        if (!whole(bw))
            deck_error(
                "IS: insulation on a partial-wire segment range — per-wire specs cover whole wires "
                "only");
        for (const auto &kv : bw)
            if (radius <= structure().wires[static_cast<std::size_t>(kv.first)].radius)
                deck_error("IS: insulation whose outer radius does not exceed the wire's conductor radius");
        for (const auto &kv : bw) wire_insulation_[kv.first] = {radius, eps_r};
    }

    // -- TL / NT
    void network(const Card &card) {
        if (has_destroyer_)
            deck_error(card.mnemonic + " with an interposed " + destroyer_ +
                       " card between it and an earlier network card is not supported by this engine "
                       "(momwire#456): NEC silently DESTROYS every TL/NT read before such a card, so "
                       "this deck's earlier network cards would vanish with no diagnostic while still "
                       "being echoed in its DATA CARD list as read — keep a deck's TL/NT cards "
                       "contiguous (only PT, PQ and MP may sit between them)");
        Int ends[2][2] = {{card.i(0), card.i(1)}, {card.i(2), card.i(3)}};
        for (auto &end : ends) {
            const Int &tag = end[0], &seg = end[1];
            if (seg < Int(1))
                deck_error(card.mnemonic + " addresses segment " + S(seg) + " of " +
                           (!tag.is_zero() ? "tag " + S(tag) : std::string("the structure")) +
                           ", which is not a segment: NEC halts on this card with CHECK DATA, "
                           "PARAMETER SPECIFYING SEGMENT POSITION IN A GROUP OF EQUAL TAGS MUST NOT BE "
                           "ZERO — a network endpoint names one segment, so its segment number must be "
                           "1 or more");
        }
        if (card.mnemonic == "TL" && card.f(4) == 0.0)
            deck_error(
                "TL with a zero characteristic impedance is not a transmission line — NEC aborts "
                "reading the deck on this card; Z0 must be nonzero, and its SIGN, not its magnitude, "
                "is what selects a crossed line");
        NetworkCard n;
        n.kind = card.mnemonic;
        std::pair<ll, double> a = structure().resolve(ends[0][0], ends[0][1]);
        std::pair<ll, double> b = structure().resolve(ends[1][0], ends[1][1]);
        n.wire_a = a.first;
        n.arclength_a = a.second;
        n.wire_b = b.first;
        n.arclength_b = b.second;
        n.address_a[0] = ends[0][0];
        n.address_a[1] = ends[0][1];
        n.address_b[0] = ends[1][0];
        n.address_b[1] = ends[1][1];
        for (int k = 0; k < 6; k++) n.payload[k] = card.f(static_cast<std::size_t>(4 + k));
        n.first_group = static_cast<ll>(ran_.size());
        networks_.push_back(n);
    }

    // -- requests
    Request far_field(const Card &card) {
        Int mode = card.i(0);
        if (!(mode.is_zero() || mode == Int(2) || mode == Int(3)))
            deck_error("RP mode " + S(mode) + " is not supported by this engine (modes 0, 2, 3 only)");
        if (!mode.is_zero() && ground_.kind != Ground::none) {
            if (!has_second_medium_ || (second_medium_.eps_r == 0.0 && second_medium_.sigma == 0.0))
                deck_error("RP " + S(mode) +
                           " asks for a cliff pattern over a ground with no second medium stated (an "
                           "all-zero EPSR2/SIG2 pair is no medium, not a vacuum one); state the far "
                           "side with a GD card, or use RP 0");
        }
        Request r;
        r.kind = Request::far_field;
        r.mode = mode;
        r.n_theta = imax(card.i(1), Int(1));
        r.n_phi = imax(card.i(2), Int(1));
        r.theta0_deg = card.f(4);
        r.phi0_deg = card.f(5);
        r.d_theta_deg = card.f(6);
        r.d_phi_deg = card.f(7);
        r.range_m = card.f(8);
        return r;
    }

    Int near_coordinates(const Card &card) {
        Int c = card.i(0);
        if (dialect_ == Dialect::nec4) {
            if (!(c.is_zero() || c == Int(1)))
                deck_error(card.mnemonic + " coordinate system " + S(c) +
                           " is not one this engine's nec4 dialect reads; the grid is rectangular (0: "
                           "X, Y, Z in metres) or spherical (1: R in metres, then theta from the "
                           "zenith and phi from +x, in degrees)");
            return c;
        }
        if (!c.is_zero())
            deck_error(card.mnemonic + " coordinate system " + S(c) +
                       " (spherical) is not supported by this engine; rectangular (0) only");
        return Int(0);
    }

    Request near_field(const Card &card) {
        Int coordinates = near_coordinates(card);
        if (ground_.kind == Ground::finite_fast) deck_error(card.mnemonic + kRefuseNearFieldReflCoef);
        Request r;
        r.kind = Request::near_field;
        r.magnetic = card.mnemonic == "NH";
        r.coordinates = coordinates;
        for (int k = 0; k < 3; k++) {
            r.counts[k] = imax(card.i(static_cast<std::size_t>(1 + k)), Int(1));
            r.origin[k] = card.f(static_cast<std::size_t>(4 + k));
            r.step[k] = card.f(static_cast<std::size_t>(7 + k));
        }
        return r;
    }

    void print_control_card(const Card &card) {
        Int flag = card.i(0);
        has_print_control_ = true;
        print_control_ = PrintControl();
        if (flag == Int(-1)) {
            print_control_.suppressed = true;
            return;
        }
        Int tag = card.i(1), first = card.i(2), last = card.i(3);
        if (flag.is_zero() && (!first.is_zero() || !last.is_zero())) {
            print_control_.has_elements = true;
            print_control_.elements = structure().wire_of_segment_range(tag, first, last);
        }
    }

    void multiprocessing_card(const Card &card) {
        for (std::size_t k = 0; k < 2; k++) {
            card.i(k);  // raises where Python's int(round(f)) does
            double f = card.f(k);
            // float(card.i(k)) is round-half-even of f, exact as a double
            if (f != std::nearbyint(f))
                deck_error("MP field " + S(static_cast<ll>(k + 1)) + " must be an integer, not " +
                           py_repr(f));
        }
        has_multiprocessing_ = true;
        multiprocessing_[0] = card.i(0);
        multiprocessing_[1] = card.i(1);
    }

    // -- execution
    void unarmed_execute(const Card &card) {
        if (dialect_ != Dialect::nec4) return;
        if (card.mnemonic == "RP") {
            far_field(card);
        } else if (card.mnemonic == "NE" || card.mnemonic == "NH") {
            near_field(card);
        }
    }

    void execute(const Card &card) {
        if (!armed_ && executed_) {
            unarmed_execute(card);
            ran_.push_back(false);
            groups_.push_back(Pending());
            return;
        }
        Request request;
        if (card.mnemonic == "RP") {
            request = far_field(card);
        } else if (card.mnemonic == "NE" || card.mnemonic == "NH") {
            request = near_field(card);
        }
        bool refilled = fresh_fr_ || !executed_;
        Pending p;
        p.sources = sources_;
        if (fresh_fr_) {
            p.frequencies = frequencies_;
        } else if (!frequencies_.empty()) {
            p.frequencies = {frequencies_.back()};
        }
        p.request = request;
        p.has_print_control = has_print_control_;
        p.print_control = print_control_;
        p.extended_kernel = extended_kernel_;
        p.has_multiprocessing = has_multiprocessing_;
        p.multiprocessing[0] = multiprocessing_[0];
        p.multiprocessing[1] = multiprocessing_[1];
        p.environment.ground = ground_;
        p.environment.has_second_medium = has_second_medium_;
        p.environment.second_medium = second_medium_;
        p.refilled = refilled;
        p.refilled_partial = operator_dirty_ && !refilled;
        p.loads = live_;
        p.materials = materials_;
        ran_.push_back(true);
        groups_.push_back(p);
        executed_ += 1;
        fresh_fr_ = false;
        operator_dirty_ = false;
        armed_ = false;
        sources_stale_ = true;
    }

    void card(const Card &card) {
        const std::string &m = card.mnemonic;
        if (m == "CM" || m == "CE") {
            std::string text = card.text();
            comments_.push_back(nec5::latin1_to_utf8(text));
            Int qq;
            if (directive(text, "QQ", qq) && qq > Int(0)) quiet_ = true;
            Int ff;
            if (directive(text, "FF", ff)) {
                has_reduced_field_ = true;
                reduced_field_ = ff;
            }
            return;
        }
        auto hit = refused_by_name().find(m);
        if (hit != refused_by_name().end()) deck_error(hit->second);
        if (in(m, kArmingCards)) {
            armed_ = true;
            if (in(m, kOperatorCards) && executed_) operator_dirty_ = true;
        }
        if (m == "TL" || m == "NT") {
            network(card);
            return;
        }
        if (!networks_.empty() && !in(m, kNetworkTransparent)) {
            if (!has_destroyer_) {
                has_destroyer_ = true;
                destroyer_ = m;
            }
        }
        if (in(m, kGeometryCards) || (m == "CW" && dialect_ == Dialect::nec4)) {
            geometry(card);
            return;
        }
        if (m == "GN") return gn(card);
        if (m == "GD") return gd(card);
        if (m == "FR") return fr(card);
        if (m == "EX") return ex(card);
        if (m == "LD") return ld(card);
        if (m == "IS") return is_card(card);
        if (m == "EK") return ek(card);
        if (m == "PT") return print_control_card(card);
        if (m == "PQ") return;
        if (m == "MP") return multiprocessing_card(card);
        if (in(m, kExecuteCards)) return execute(card);
        deck_error("unrecognised NEC card " + repr_utf8_alpha(m));
    }

    bool material_for(ll wire, const MaterialState &state, WireMaterial &out) {
        bool has_c = false;
        double conductivity = 0;
        auto ci = state.wire_conductivity.find(wire);
        if (ci != state.wire_conductivity.end()) {
            has_c = true;
            conductivity = ci->second;
        } else if (state.has_global_conductivity) {
            has_c = true;
            conductivity = state.global_conductivity;
        }
        auto ii = wire_insulation_.find(wire);
        bool has_d = false;
        DistributedRLC distributed;
        auto di = state.wire_distributed.find(wire);
        if (di != state.wire_distributed.end()) {
            has_d = true;
            distributed = di->second;
        } else if (state.has_global_distributed) {
            has_d = true;
            distributed = state.global_distributed;
        }
        if (has_d && distributed.c != 0.0) {
            const FlatWire &piece = structure().wires[static_cast<std::size_t>(wire)];
            double d = piece.length() / static_cast<double>(piece.n_seg);
            distributed.c = distributed.c * d * d;
        }
        if (!has_c && ii == wire_insulation_.end() && !has_d) return false;
        out = WireMaterial();
        out.has_conductivity = has_c;
        out.conductivity = conductivity;
        if (ii != wire_insulation_.end()) {
            out.has_insulation = true;
            out.insulation_radius = ii->second.first;
            out.insulation_eps_r = ii->second.second;
        }
        out.has_distributed = has_d;
        out.distributed = distributed;
        return true;
    }
};

bool crossing_junction_at(const std::vector<FlatWire> &wires, const double *point, double tol) {
    bool above = false, below = false;
    for (const FlatWire &w : wires) {
        double own = kSmin * (w.length() / static_cast<double>(w.n_seg));
        double tol_w = own > tol ? own : tol;
        const double *ends[2][2] = {{w.p1, w.p2}, {w.p2, w.p1}};
        for (auto &eo : ends) {
            const double *end = eo[0], *other = eo[1];
            bool all = true;
            for (int k = 0; k < 3; k++) all = all && std::fabs(end[k] - point[k]) <= tol_w;
            if (all) {
                double z = other[2];
                if (z > tol_w) {
                    above = true;
                } else if (z < -tol_w) {
                    below = true;
                }
            }
        }
    }
    return above && below;
}

Model Parser::model() {
    if (!saw_ex_) deck_error("deck has no EX card — nothing drives the structure");
    const Structure &st = structure();

    // _loads_and_materials
    std::vector<ll> used;
    MaterialState materials;
    {
        std::set<ll> u;
        std::vector<const MaterialState *> states;
        bool any = false;
        for (std::size_t k = 0; k < ran_.size(); k++)
            if (ran_[k]) {
                any = true;
                for (ll i : groups_[k].loads) u.insert(i);
                states.push_back(&groups_[k].materials);
            }
        if (!any) {
            u = std::set<ll>(live_.begin(), live_.end());
            states.push_back(&materials_);
        }
        for (std::size_t k = 1; k < states.size(); k++)
            if (*states[k] != *states[0])
                deck_error(
                    "LD 2 / LD 3 / LD 5 wire loading that changes between execute cards is not "
                    "supported by this engine (momwire#1069): a material enters the matrix fill, which "
                    "is shared by every group at one frequency — run each loading as its own deck");
        used.assign(u.begin(), u.end());
        materials = *states[0];
    }
    std::map<ll, ll> place;
    for (std::size_t k = 0; k < used.size(); k++) place[used[k]] = static_cast<ll>(k);

    Model m;
    for (ll index : used) {
        const LogEntry &e = load_log_[static_cast<std::size_t>(index)];
        m.loads.push_back(Load{e.piece, e.arclength, e.spec});
    }
    for (const Piece &piece : st.pieces) {
        DeckWire w;
        for (int c = 0; c < 3; c++) {
            w.vertices[0][c] = piece.p1[c];
            w.vertices[1][c] = piece.p2[c];
        }
        w.radius = st.wires[static_cast<std::size_t>(piece.wire)].radius;
        w.edge_elements = piece.n_seg;
        w.has_material = material_for(piece.wire, materials, w.material);
        if (w.radius <= 0.0)
            throw ParseError(ErrorKind::value_error, "wire radius must be > 0, got " + py_repr(w.radius));
        m.wires.push_back(w);
    }

    // _feeds_and_groups
    std::vector<std::pair<Int, Int>> order;
    for (std::size_t k = 0; k < ran_.size(); k++) {
        if (!ran_[k]) continue;
        for (const Source &s : groups_[k].sources) {
            bool seen = false;
            for (const auto &o : order) seen = seen || (o.first == s.tag && o.second == s.seg);
            if (!seen) {
                order.push_back({s.tag, s.seg});
                std::pair<ll, double> r = st.resolve(s.tag, s.seg);
                m.feeds.push_back(Feed{r.first, r.second, s.volts});
            }
        }
    }
    auto position = [&](const Int &tag, const Int &seg) {
        for (std::size_t k = 0; k < order.size(); k++)
            if (order[k].first == tag && order[k].second == seg) return k;
        return order.size();
    };
    for (std::size_t k = 0; k < ran_.size(); k++) {
        m.group_ran.push_back(ran_[k]);
        ExecuteGroup g;
        if (ran_[k]) {
            const Pending &p = groups_[k];
            g.frequencies = p.frequencies;
            g.voltages.assign(order.size(), std::complex<double>(0.0, 0.0));
            for (const Source &s : p.sources) g.voltages[position(s.tag, s.seg)] = s.volts;
            g.request = p.request;
            g.has_print_control = p.has_print_control;
            g.print_control = p.print_control;
            g.extended_kernel = p.extended_kernel;
            g.has_multiprocessing = p.has_multiprocessing;
            g.multiprocessing[0] = p.multiprocessing[0];
            g.multiprocessing[1] = p.multiprocessing[1];
            g.refilled = p.refilled;
            g.refilled_partial = p.refilled_partial;
            g.environment = p.environment;
            for (ll i : p.loads) g.loads.push_back(place[i]);
            std::sort(g.loads.begin(), g.loads.end());
        }
        m.groups.push_back(g);
    }

    // momwire#489: GE -1 with a wire end standing in the plane under a ground
    if (st.ground_plane_flag && !st.ground_plane_interpolates) {
        bool grounded = false;
        for (std::size_t k = 0; k < ran_.size(); k++)
            if (ran_[k] && groups_[k].environment.ground.kind != Ground::none) grounded = true;
        if (ran_.empty() && ground_.kind != Ground::none) grounded = true;
        if (grounded) {
            for (const FlatWire &w : st.wires) {
                double tol = kSmin * (w.length() / static_cast<double>(w.n_seg));
                for (const double *p : {w.p1, w.p2}) {
                    if (std::fabs(p[2]) <= tol && !crossing_junction_at(st.wires, p, tol))
                        deck_error(
                            "GE -1 declares the ground plane without the ground-contact current "
                            "expansion, and wire " +
                            S(w.tag) + "'s end stands in the plane (z = " + g(p[2]) +
                            "); this engine serves the interpolated (GE 1) contact only — write GE 1, "
                            "or lift the wire clear of the plane");
                }
            }
        }
    }

    m.ground = ground_;
    m.has_second_medium = has_second_medium_;
    m.second_medium = second_medium_;
    m.ground_plane_flag = st.ground_plane_flag;
    m.ground_plane_interpolates = st.ground_plane_interpolates;
    m.comments = comments_;
    m.quiet = quiet_;
    m.has_reduced_field = has_reduced_field_;
    m.reduced_field = reduced_field_;
    m.networks = networks_;
    if (dialect_ == Dialect::nec4) {
        for (std::size_t i = 0; i < order.size(); i++)
            for (const auto &sk : source_kind_)
                if (sk.first.first == order[i].first && sk.first.second == order[i].second &&
                    sk.second == 6)
                    m.current_feeds.push_back(static_cast<ll>(i));
        m.extended_kernel_default = true;
        m.junction_rule = "nec42";
    }
    return m;
}

// ============================================================================
// The comparison format (`deck_compare.dump`)
// ============================================================================

struct Out {
    std::string s;
    void line(const std::string &l) { s += l + "\n"; }
    void cls(const std::string &p, const char *name) { line(p + " : " + name); }
    void tup(const std::string &p, std::size_t n) { line(p + " : tuple " + std::to_string(n)); }
    void none(const std::string &p) { line(p + " = None"); }
    void b(const std::string &p, bool v) { line(p + " = bool " + (v ? "True" : "False")); }
    void i(const std::string &p, const Int &v) { line(p + " = int " + v.str()); }
    void i(const std::string &p, ll v) { line(p + " = int " + std::to_string(v)); }
    void f(const std::string &p, double v) { line(p + " = float " + nec5::dump_bits(v)); }
    void c(const std::string &p, std::complex<double> v) {
        line(p + " = complex " + nec5::dump_bits(v.real()) + " " + nec5::dump_bits(v.imag()));
    }
    void str(const std::string &p, const std::string &utf8) { line(p + " = str " + nec5::dump_hex(utf8)); }
    void point(const std::string &p, const double *v) {
        tup(p, 3);
        for (int k = 0; k < 3; k++) f(p + "[" + std::to_string(k) + "]", v[k]);
    }
};

std::string at(const std::string &p, std::size_t k) { return p + "[" + std::to_string(k) + "]"; }

void dump_ground(Out &o, const std::string &p, const Ground &gr) {
    switch (gr.kind) {
        case Ground::none:
            o.none(p);
            return;
        case Ground::pec:
            o.str(p, "pec");
            return;
        default:
            o.tup(p, 3);
            o.str(at(p, 0), gr.kind == Ground::finite_fast ? "finite-fast" : "finite");
            o.f(at(p, 1), gr.eps_r);
            o.f(at(p, 2), gr.sigma);
    }
}

void dump_second(Out &o, const std::string &p, bool has, const SecondMedium &s) {
    if (!has) return o.none(p);
    o.cls(p, "SecondMedium");
    o.f(p + ".eps_r", s.eps_r);
    o.f(p + ".sigma", s.sigma);
    o.f(p + ".edge_distance", s.edge_distance);
    o.f(p + ".height", s.height);
}

void dump_load_spec(Out &o, const std::string &p, const LoadSpec &s) {
    o.cls(p, "LoadSpec");
    o.str(p + ".kind", s.kind);
    o.f(p + ".r", s.r);
    o.f(p + ".l", s.l);
    o.f(p + ".c", s.c);
    o.f(p + ".x", s.x);
}

void dump_request(Out &o, const std::string &p, const Request &r) {
    if (r.kind == Request::none) return o.none(p);
    if (r.kind == Request::far_field) {
        o.cls(p, "FarFieldRequest");
        o.i(p + ".mode", r.mode);
        o.i(p + ".n_theta", r.n_theta);
        o.i(p + ".n_phi", r.n_phi);
        o.f(p + ".theta0_deg", r.theta0_deg);
        o.f(p + ".phi0_deg", r.phi0_deg);
        o.f(p + ".d_theta_deg", r.d_theta_deg);
        o.f(p + ".d_phi_deg", r.d_phi_deg);
        o.f(p + ".range_m", r.range_m);
        return;
    }
    o.cls(p, "NearFieldRequest");
    o.b(p + ".magnetic", r.magnetic);
    o.tup(p + ".counts", 3);
    for (int k = 0; k < 3; k++) o.i(at(p + ".counts", static_cast<std::size_t>(k)), r.counts[k]);
    o.point(p + ".origin", r.origin);
    o.point(p + ".step", r.step);
    o.i(p + ".coordinates", r.coordinates);
}

}  // namespace

std::string dump_model(const Model &m, const std::string &p) {
    Out o;
    o.cls(p, "DeckModel");
    o.tup(p + ".wires", m.wires.size());
    for (std::size_t k = 0; k < m.wires.size(); k++) {
        const DeckWire &w = m.wires[k];
        std::string q = at(p + ".wires", k);
        o.cls(q, "DeckWire");
        o.tup(q + ".vertices", 2);
        o.point(at(q + ".vertices", 0), w.vertices[0]);
        o.point(at(q + ".vertices", 1), w.vertices[1]);
        o.f(q + ".radius", w.radius);
        o.tup(q + ".edge_elements", 1);
        o.i(at(q + ".edge_elements", 0), w.edge_elements);
        if (!w.has_material) {
            o.none(q + ".material");
        } else {
            const WireMaterial &mt = w.material;
            std::string r = q + ".material";
            o.cls(r, "WireMaterial");
            if (mt.has_conductivity) {
                o.f(r + ".conductivity", mt.conductivity);
            } else {
                o.none(r + ".conductivity");
            }
            if (mt.has_insulation) {
                o.f(r + ".insulation_radius", mt.insulation_radius);
                o.f(r + ".insulation_eps_r", mt.insulation_eps_r);
            } else {
                o.none(r + ".insulation_radius");
                o.none(r + ".insulation_eps_r");
            }
            if (mt.has_distributed) {
                std::string d = r + ".distributed_rlc";
                o.cls(d, "DistributedRLC");
                o.str(d + ".kind", mt.distributed.kind);
                o.f(d + ".r", mt.distributed.r);
                o.f(d + ".l", mt.distributed.l);
                o.f(d + ".c", mt.distributed.c);
            } else {
                o.none(r + ".distributed_rlc");
            }
            o.none(r + ".insulation_loss_tangent");
        }
    }
    o.tup(p + ".feeds", m.feeds.size());
    for (std::size_t k = 0; k < m.feeds.size(); k++) {
        std::string q = at(p + ".feeds", k);
        o.tup(q, 3);
        o.i(at(q, 0), m.feeds[k].wire);
        o.f(at(q, 1), m.feeds[k].arclength);
        o.c(at(q, 2), m.feeds[k].volts);
    }
    o.tup(p + ".node_gaps", 0);
    o.tup(p + ".loads", m.loads.size());
    for (std::size_t k = 0; k < m.loads.size(); k++) {
        std::string q = at(p + ".loads", k);
        o.tup(q, 3);
        o.i(at(q, 0), m.loads[k].wire);
        o.f(at(q, 1), m.loads[k].arclength);
        dump_load_spec(o, at(q, 2), m.loads[k].spec);
    }
    dump_ground(o, p + ".ground", m.ground);
    o.f(p + ".ground_z", 0.0);
    dump_second(o, p + ".second_medium", m.has_second_medium, m.second_medium);
    o.b(p + ".ground_plane_flag", m.ground_plane_flag);
    o.b(p + ".ground_plane_interpolates", m.ground_plane_interpolates);
    o.tup(p + ".groups", m.groups.size());
    for (std::size_t k = 0; k < m.groups.size(); k++) {
        std::string q = at(p + ".groups", k);
        if (!m.group_ran[k]) {
            o.none(q);
            continue;
        }
        const ExecuteGroup &gr = m.groups[k];
        o.cls(q, "ExecuteGroup");
        o.tup(q + ".frequencies", gr.frequencies.size());
        for (std::size_t j = 0; j < gr.frequencies.size(); j++) o.f(at(q + ".frequencies", j), gr.frequencies[j]);
        o.tup(q + ".voltages", gr.voltages.size());
        for (std::size_t j = 0; j < gr.voltages.size(); j++) o.c(at(q + ".voltages", j), gr.voltages[j]);
        dump_request(o, q + ".request", gr.request);
        if (!gr.has_print_control) {
            o.none(q + ".print_control");
        } else {
            std::string r = q + ".print_control";
            o.cls(r, "PrintControl");
            o.b(r + ".suppressed", gr.print_control.suppressed);
            if (!gr.print_control.has_elements) {
                o.none(r + ".elements");
            } else {
                const auto &el = gr.print_control.elements;
                o.tup(r + ".elements", el.size());
                for (std::size_t j = 0; j < el.size(); j++) {
                    o.tup(at(r + ".elements", j), 2);
                    o.i(at(at(r + ".elements", j), 0), el[j].first);
                    o.i(at(at(r + ".elements", j), 1), el[j].second);
                }
            }
        }
        o.b(q + ".extended_kernel", gr.extended_kernel);
        if (!gr.has_multiprocessing) {
            o.none(q + ".multiprocessing");
        } else {
            o.tup(q + ".multiprocessing", 2);
            o.i(at(q + ".multiprocessing", 0), gr.multiprocessing[0]);
            o.i(at(q + ".multiprocessing", 1), gr.multiprocessing[1]);
        }
        o.b(q + ".refilled", gr.refilled);
        o.b(q + ".refilled_partial", gr.refilled_partial);
        std::string e = q + ".environment";
        o.cls(e, "Environment");
        dump_ground(o, e + ".ground", gr.environment.ground);
        o.f(e + ".ground_z", 0.0);
        dump_second(o, e + ".second_medium", gr.environment.has_second_medium, gr.environment.second_medium);
        o.tup(q + ".loads", gr.loads.size());
        for (std::size_t j = 0; j < gr.loads.size(); j++) o.i(at(q + ".loads", j), gr.loads[j]);
    }
    o.tup(p + ".comments", m.comments.size());
    for (std::size_t k = 0; k < m.comments.size(); k++) o.str(at(p + ".comments", k), m.comments[k]);
    o.b(p + ".quiet", m.quiet);
    if (m.has_reduced_field) {
        o.i(p + ".reduced_field", m.reduced_field);
    } else {
        o.none(p + ".reduced_field");
    }
    o.tup(p + ".deferred", 0);
    o.tup(p + ".networks", m.networks.size());
    for (std::size_t k = 0; k < m.networks.size(); k++) {
        const NetworkCard &n = m.networks[k];
        std::string q = at(p + ".networks", k);
        o.cls(q, "NetworkCard");
        o.str(q + ".kind", n.kind);
        o.tup(q + ".end_a", 2);
        o.i(at(q + ".end_a", 0), n.wire_a);
        o.f(at(q + ".end_a", 1), n.arclength_a);
        o.tup(q + ".end_b", 2);
        o.i(at(q + ".end_b", 0), n.wire_b);
        o.f(at(q + ".end_b", 1), n.arclength_b);
        o.tup(q + ".address_a", 2);
        o.i(at(q + ".address_a", 0), n.address_a[0]);
        o.i(at(q + ".address_a", 1), n.address_a[1]);
        o.tup(q + ".address_b", 2);
        o.i(at(q + ".address_b", 0), n.address_b[0]);
        o.i(at(q + ".address_b", 1), n.address_b[1]);
        o.tup(q + ".payload", 6);
        for (std::size_t j = 0; j < 6; j++) o.f(at(q + ".payload", j), n.payload[j]);
        o.i(q + ".first_group", n.first_group);
    }
    o.tup(p + ".current_feeds", m.current_feeds.size());
    for (std::size_t k = 0; k < m.current_feeds.size(); k++) o.i(at(p + ".current_feeds", k), m.current_feeds[k]);
    o.b(p + ".extended_kernel_default", m.extended_kernel_default);
    o.str(p + ".junction_rule", m.junction_rule);
    return o.s;
}

Model parse(const std::string &latin1_text, Dialect dialect) {
    Parser parser(dialect);
    parser.feed(latin1_text);
    return parser.model();
}

std::string dump_parse(const std::string &latin1_text, Dialect dialect) {
    try {
        return dump_model(parse(latin1_text, dialect), "parse");
    } catch (const ParseError &e) {
        return std::string("parse ! ") + nec5::error_kind_name(e.kind) + " " + nec5::dump_hex(e.message) +
               "\n";
    } catch (const Declined &e) {
        return std::string("parse ? declined ") + nec5::dump_hex(e.what()) + "\n";
    }
}

std::string py_repr(double x) {
    if (std::isnan(x)) return "nan";
    if (std::isinf(x)) return x > 0 ? "inf" : "-inf";
    if (x == 0.0) return std::signbit(x) ? "-0.0" : "0.0";
    // The shortest digit string that reads back as x (printf's correctly
    // rounded %.*e, the first precision that round-trips).
    std::string e;
    for (int p = 1; p <= 17; p++) {
        e = nec5::format_c_locale(("%." + std::to_string(p - 1) + "e").c_str(), x);
        double back;
        if (nec5::latin1_float(e, back) && back == x) break;
    }
    bool negative = e[0] == '-';
    std::string body = negative ? e.substr(1) : e;
    std::size_t epos = body.find('e');
    std::string mant = body.substr(0, epos);
    int exp10 = std::stoi(body.substr(epos + 1));
    std::string digits;
    for (char ch : mant)
        if (ch != '.') digits += ch;
    while (digits.size() > 1 && digits.back() == '0') digits.pop_back();
    int decpt = exp10 + 1;
    int nd = static_cast<int>(digits.size());
    std::string out;
    if (decpt <= -4 || decpt > 16) {
        out = digits.substr(0, 1);
        if (nd > 1) out += "." + digits.substr(1);
        char buf[16];
        std::snprintf(buf, sizeof buf, "e%+03d", decpt - 1);
        out += buf;
    } else if (decpt <= 0) {
        out = "0." + std::string(static_cast<std::size_t>(-decpt), '0') + digits;
    } else if (decpt >= nd) {
        out = digits + std::string(static_cast<std::size_t>(decpt - nd), '0') + ".0";
    } else {
        out = digits.substr(0, static_cast<std::size_t>(decpt)) + "." +
              digits.substr(static_cast<std::size_t>(decpt));
    }
    return negative ? "-" + out : out;
}

double py_dist(const double a[3], const double b[3]) { return dist3(a, b); }

double helix_wire_length(const nec5::Card &card) {
    const std::vector<std::array<double, 3>> v = helix_points(card);
    std::vector<double> d;
    d.reserve(v.size());
    for (std::size_t k = 0; k + 1 < v.size(); k++) d.push_back(dist3(v[k].data(), v[k + 1].data()));
    return py_sum(d.data(), static_cast<int>(d.size()));
}

// `deck._nec4.catenary_length` and the wire table's chord total, `sum()` of
// the chords' `math.dist` (momwire#1697).
void catenary_lengths(const nec5::Card &card, double *length, double *chord_length) {
    const std::vector<std::array<double, 3>> v = catenary_points(card, length);
    std::vector<double> d;
    d.reserve(v.size());
    for (std::size_t k = 0; k + 1 < v.size(); k++) d.push_back(dist3(v[k].data(), v[k + 1].data()));
    *chord_length = py_sum(d.data(), static_cast<int>(d.size()));
}

Geometry geometry_of(const std::vector<Card> &cards) {
    Structure st = build_geometry(cards, Dialect::nec4);
    Geometry g;
    for (const FlatWire &w : st.wires) {
        GeometryWire v;
        v.tag = w.tag;
        v.n_seg = w.n_seg;
        for (int c = 0; c < 3; c++) {
            v.p1[c] = w.p1[c];
            v.p2[c] = w.p2[c];
        }
        v.radius = w.radius;
        g.wires.push_back(v);
    }
    for (const Piece &p : st.pieces) {
        GeometryPiece v;
        v.wire = p.wire;
        v.first_seg = p.first_seg;
        v.n_seg = p.n_seg;
        for (int c = 0; c < 3; c++) {
            v.p1[c] = p.p1[c];
            v.p2[c] = p.p2[c];
        }
        g.pieces.push_back(v);
    }
    g.pieces_of_wire = st.pieces_of_wire;
    g.ground_plane_flag = st.ground_plane_flag;
    g.ground_plane_interpolates = st.ground_plane_interpolates;
    g.has_symmetry = st.has_symmetry;
    g.cell_wires = st.symmetry.cell_wires;
    g.cell_segments = st.symmetry.cell_segments;
    return g;
}

}  // namespace nec4
}  // namespace mw
