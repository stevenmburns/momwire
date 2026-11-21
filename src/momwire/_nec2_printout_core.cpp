// The NEC-2 portal's printout in C++ (see `_nec2_printout_core.h`). Each
// function names the Python it copies (`momwire.dropin.nec2c._portal`
// unless it says otherwise); every number is printed as Python's format
// spec prints it, and every value computed here is computed in Python's
// order: CPython's complex arithmetic as `_Py_c_prod` / `_Py_c_quot`,
// numpy's scalar products unfused, its array products fused (`nec5::replay`).

#if defined(_MSC_VER) && !defined(__clang__)
#pragma float_control(precise, on)
#pragma fp_contract(off)
#endif

#include "_nec2_printout_core.h"

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstdio>
#include <limits>
#include <map>
#include <string>
#include <utility>
#include <vector>

namespace mw {
namespace nec2 {

namespace {

typedef long long ll;
typedef std::complex<double> cd;
using nec4::Declined;
using nec5::Card;
using nec5::Int;
namespace rp = nec5::replay;

const double kPi = 3.141592653589793;
const double kEps0 = 8.854187817e-12;  // _far_readout.EPS0
const double kFieldFloor2 = 1.0e-20;   // _far_readout._FIELD_FLOOR2
const double kGainFloor2 = 1.0e-20;    // _far_readout._GAIN_FLOOR2
const double kDustFloor2 = 1.0e-20;    // _PRINTED_DUST_FLOOR2
const double kSmin = 1e-3;             // deck._nec2_geometry._SMIN
const double kRadToDeg = 180.0 / 3.141592653589793;
const double kDegToRad = 3.141592653589793 / 180.0;

[[noreturn]] void decline(const std::string &m) { throw Declined(m); }

ll i64(const Int &v) {
    if (!v.fits_i64()) decline("an integer past 64 bits");
    return v.i64();
}

// --- Python's format specs ----------------------------------------------------

std::string rjust(const std::string &s, std::size_t w) { return s.size() >= w ? s : std::string(w - s.size(), ' ') + s; }
std::string ljust(const std::string &s, std::size_t w) { return s.size() >= w ? s : s + std::string(w - s.size(), ' '); }
// `{n:Wd}`
std::string d(ll v, int w) { return rjust(std::to_string(v), static_cast<std::size_t>(w)); }
std::string d(const Int &v, int w) { return rjust(v.str(), static_cast<std::size_t>(w)); }
// `{x:W.DE}` / `{x:W.Df}` (NaN as Python spells it)
std::string e(double x, int w, int p) { return nec5::py_format_e(x, w, p); }
std::string f(double x, int w, int p) { return nec5::py_format_f(x, w, p); }
// `{x:W.Dg}`: printf's %g in the C locale, Python's NaN and infinities
std::string g(double x, int w, int p) {
    if (std::isnan(x)) return rjust("nan", static_cast<std::size_t>(w));
    if (std::isinf(x)) return rjust(x > 0 ? "inf" : "-inf", static_cast<std::size_t>(w));
    char spec[16];
    std::snprintf(spec, sizeof spec, "%%%d.%dg", w, p);
    return nec5::format_c_locale(spec, x);
}
// `{x:+W.Df}`
std::string f_signed(double x, int w, int p) {
    if (std::isnan(x)) return rjust(std::signbit(x) ? "-nan" : "+nan", static_cast<std::size_t>(w));
    if (std::isinf(x)) return rjust(x > 0 ? "+inf" : "-inf", static_cast<std::size_t>(w));
    char spec[16];
    std::snprintf(spec, sizeof spec, "%%+%d.%df", w, p);
    return nec5::format_c_locale(spec, x);
}

// --- CPython's and numpy's arithmetic --------------------------------------------

// CPython's `math.atan2` (its special cases, then libm) and `math.degrees`
double py_atan2(double y, double x) {
    if (std::isnan(x) || std::isnan(y)) return std::numeric_limits<double>::quiet_NaN();
    if (std::isinf(y)) {
        if (std::isinf(x)) {
            if (std::copysign(1., x) == 1.) return std::copysign(0.25 * kPi, y);
            return std::copysign(0.75 * kPi, y);
        }
        return std::copysign(0.5 * kPi, y);
    }
    if (std::isinf(x) || y == 0.) {
        if (std::copysign(1., x) == 1.) return std::copysign(0., y);
        return std::copysign(kPi, y);
    }
    return std::atan2(y, x);
}
double py_degrees(double x) { return x * kRadToDeg; }
double py_radians(double x) { return x * kDegToRad; }
// `_phase_deg(value)`
double phase_deg(cd v) { return py_degrees(py_atan2(v.imag(), v.real())); }
// `abs(complex)`: libm's hypot
double cabs(cd v) { return std::hypot(v.real(), v.imag()); }

// CPython 3.12's `_Py_c_prod` / `_Py_c_quot` (a zero divisor raises there,
// which every caller here guards)
cd py_mul(cd a, cd b) {
    return cd(a.real() * b.real() - a.imag() * b.imag(), a.real() * b.imag() + a.imag() * b.real());
}
cd py_div(cd a, cd b) {
    const double abs_breal = b.real() < 0 ? -b.real() : b.real();
    const double abs_bimag = b.imag() < 0 ? -b.imag() : b.imag();
    if (abs_breal >= abs_bimag) {
        if (abs_breal == 0.0) decline("a complex division by zero (the reference raises)");
        const double ratio = b.imag() / b.real();
        const double denom = b.real() + b.imag() * ratio;
        return cd((a.real() + a.imag() * ratio) / denom, (a.imag() - a.real() * ratio) / denom);
    }
    if (abs_bimag >= abs_breal) {
        const double ratio = b.real() / b.imag();
        const double denom = b.real() * ratio + b.imag();
        return cd((a.real() * ratio + a.imag()) / denom, (a.imag() * ratio - a.real()) / denom);
    }
    decline("a complex division by NaN");
}
// numpy's scalar complex product (`cdouble_ctype_multiply`): never fused
cd np_smul(cd a, cd b) {
    return cd(a.real() * b.real() - a.imag() * b.imag(), a.real() * b.imag() + a.imag() * b.real());
}

// --- the fixed chrome (verbatim from tests/fixtures/nec_portal/*.out) ---------------

const char *const kCommentsHeader = "                               ---------------- COMMENTS ----------------";
const char *const kCommentIndent = "                              ";
const char *const kStructureHeader = "                               -------- STRUCTURE SPECIFICATION --------";
const char *const kStructureNotes[] = {
    "                                     COORDINATES MUST BE INPUT IN",
    "                                     METERS OR BE SCALED TO METERS",
    "                                     BEFORE STRUCTURE INPUT IS ENDED",
};
const char *const kWireTableHeader[] = {
    "  WIRE                                                                                 SEG FIRST  LAST  TAG",
    "   No:        X1         Y1         Z1         X2         Y2         Z2       RADIUS   No:   SEG   SEG  No:",
};
const char *const kJunctionsHeader[] = {
    "    ---------- MULTIPLE WIRE JUNCTIONS ----------",
    "    JUNCTION  SEGMENTS (- FOR END 1, + FOR END 2)",
};
const char *const kSegmentationHeader = "                               ---------- SEGMENTATION DATA ----------";
const char *const kSegmentationNotes[] = {
    "                                        COORDINATES IN METERS",
    "                            I+ AND I- INDICATE THE SEGMENTS BEFORE AND AFTER I",
};
const char *const kSegmentationTableHeader[] = {
    "   SEG    COORDINATES OF SEGM CENTER     SEGM    ORIENTATION ANGLES    WIRE    CONNECTION DATA   TAG",
    "   No:       X         Y         Z      LENGTH     ALPHA      BETA    RADIUS    I-     I    I+   No:",
};
const char *const kFrequencyHeader = "                               --------- FREQUENCY --------";
const char *const kApproxIntegration[] = {
    "                        APPROXIMATE INTEGRATION EMPLOYED FOR SEGMENTS ",
    "                        THAT ARE MORE THAN 1.000 WAVELENGTHS APART",
};
const char *const kLoadingHeader = "                          ------ STRUCTURE IMPEDANCE LOADING ------";
const char *const kLoadedTwice = "  NOTE, SOME OF THE ABOVE SEGMENTS HAVE BEEN LOADED TWICE - IMPEDANCES ADDED";
const char *const kLoadingNone = "                                 THIS STRUCTURE IS NOT LOADED";
const char *const kLoadingTableHeader[] = {
    "  LOCATION        RESISTANCE  INDUCTANCE  CAPACITANCE     IMPEDANCE (OHMS)   CONDUCTIVITY  CIRCUIT",
    "  ITAG FROM THRU     OHMS       HENRYS      FARADS       REAL     IMAGINARY   MHOS/METER      TYPE",
};
const char *const kEnvironmentHeader = "                            -------- ANTENNA ENVIRONMENT --------";
const char *const kMatrixTimingHeader = "                             ---------- MATRIX TIMING ----------";
const char *const kAipHeader = "                        --------- ANTENNA INPUT PARAMETERS ---------";
const char *const kAipTableHeader[] = {
    "  TAG   SEG       VOLTAGE (VOLTS)         CURRENT (AMPS)         IMPEDANCE (OHMS)        ADMITTANCE (MHOS)     "
    "POWER",
    "  No:   No:     REAL      IMAGINARY     REAL      IMAGINARY     REAL      IMAGINARY    REAL       IMAGINARY   "
    "(WATTS)",
};
const char *const kCurrentsHeader = "                           -------- CURRENTS AND LOCATION --------";
const char *const kCurrentsNote = "                                  DISTANCES IN WAVELENGTHS";
const char *const kCurrentsTableHeader[] = {
    "   SEG  TAG    COORDINATES OF SEGM CENTER     SEGM    ------------- CURRENT (AMPS) -------------",
    "   No:  No:       X         Y         Z      LENGTH     REAL      IMAGINARY    MAGN        PHASE",
};
const char *const kChargeHeader = "                                  ------ CHARGE DENSITIES ------";
const char *const kChargeNote = "                                     DISTANCES IN WAVELENGTHS";
const char *const kChargeTableHeader[] = {
    "   SEG   TAG    COORDINATES OF SEG CENTER     SEG          CHARGE DENSITY (COULOMBS/METER)",
    "   No:   No:     X         Y         Z       LENGTH     REAL      IMAGINARY     MAGN       PHASE",
};
const char *const kPowerHeader = "                               ---------- POWER BUDGET ---------";
const char *const kNetworkDataHeader = "                                            ---------- NETWORK DATA ----------";
const char *const kNetworkTlTableHeader[] = {
    "  -- FROM -  --- TO --      TRANSMISSION LINE        --------- SHUNT ADMITTANCES (MHOS) ---------   LINE",
    "  TAG   SEG  TAG   SEG    IMPEDANCE      LENGTH     ----- END ONE -----      ----- END TWO -----   TYPE",
    "  No:   No:  No:   No:         OHMS      METERS      REAL      IMAGINARY      REAL      IMAGINARY",
};
const char *const kNetworkNtTableHeader[] = {
    "  -- FROM -  --- TO --            -------- ADMITTANCE MATRIX ELEMENTS (MHOS) ---------",
    "  TAG   SEG  TAG   SEG   ----- (ONE,ONE) ------   ----- (ONE,TWO) -----   ----- (TWO,TWO) -------",
    "  No:   No:  No:   No:      REAL      IMAGINARY      REAL     IMAGINARY       REAL      IMAGINARY",
};
const char *const kNetworkExcitationHeader =
    "                          --------- STRUCTURE EXCITATION DATA AT NETWORK CONNECTION POINTS --------";
const char *const kNetworkExcitationTableHeader[] = {
    "  TAG   SEG       VOLTAGE (VOLTS)          CURRENT (AMPS)         IMPEDANCE (OHMS)       ADMITTANCE (MHOS)     "
    "POWER",
    "  No:   No:     REAL      IMAGINARY     REAL      IMAGINARY     REAL      IMAGINARY     REAL      IMAGINARY   "
    "(WATTS)",
};
const char *const kPatternHeader = "                             ---------- RADIATION PATTERNS -----------";
const char *const kFarFieldGroundHeader = "                               ------ FAR FIELD GROUND PARAMETERS ------";
const char *const kPatternTableHeader[] = {
    " ---- ANGLES -----     ----- POWER GAINS -----       ---- POLARIZATION ----   ---- E(THETA) ----    ----- "
    "E(PHI) ------",
    "  THETA      PHI       VERTC    HORIZ    TOTAL       AXIAL      TILT  SENSE   MAGNITUDE    PHASE    MAGNITUDE  "
    "   PHASE",
    " DEGREES   DEGREES        DB       DB       DB       RATIO   DEGREES            VOLTS/M   DEGREES     VOLTS/M  "
    " DEGREES",
};
const char *const kPatternTime = "    Radiation Compute Time 0";
const char *const kNearEHeader = "                             -------- NEAR ELECTRIC FIELDS --------";
const char *const kNearHHeader = "                                   -------- NEAR MAGNETIC FIELDS ---------";
const char *const kNearETableHeader[] = {
    "     ------- LOCATION -------     ------- EX ------    ------- EY ------    ------- EZ ------",
    "      X         Y         Z       MAGNITUDE   PHASE    MAGNITUDE   PHASE    MAGNITUDE   PHASE",
    "    METERS    METERS    METERS     VOLTS/M  DEGREES    VOLTS/M   DEGREES     VOLTS/M  DEGREES",
};
const char *const kNearHTableHeader[] = {
    "     ------- LOCATION -------     ------- HX ------    ------- HY ------    ------- HZ ------",
    "      X         Y         Z       MAGNITUDE   PHASE    MAGNITUDE   PHASE    MAGNITUDE   PHASE",
    "    METERS    METERS    METERS      AMPS/M  DEGREES      AMPS/M  DEGREES      AMPS/M  DEGREES",
};
const char *const kNearFieldTime = "    Near Field Compute Time 0";

template <std::size_t N>
void add(std::vector<std::string> &out, const char *const (&lines)[N]) {
    for (const char *l : lines) out.emplace_back(l);
}

bool is_execute(const std::string &m) { return m == "XQ" || m == "RP" || m == "NE" || m == "NH"; }

// --- row formatting -------------------------------------------------------------

std::string fmt_wire_row(ll n, const double p1[3], const double p2[3], double radius, ll n_seg, ll first,
                         ll last, const Int &tag) {
    return " " + d(n, 5) + " " + f(p1[0], 11, 5) + " " + f(p1[1], 10, 5) + " " + f(p1[2], 10, 5) + " " +
           f(p2[0], 10, 5) + " " + f(p2[1], 10, 5) + " " + f(p2[2], 10, 5) + " " + f(radius, 10, 5) + " " +
           d(n_seg, 5) + " " + d(first, 5) + " " + d(last, 5) + " " + d(tag, 4);
}

// `fmt_aip_row`
std::string fmt_aip_row(ll tag, ll seg, cd v, cd i, cd z, cd y, double p) {
    return " " + d(tag, 4) + " " + d(seg, 5) + " " + e(v.real(), 11, 4) + " " + e(v.imag(), 11, 4) + " " +
           e(i.real(), 11, 4) + " " + e(i.imag(), 11, 4) + " " + e(z.real(), 11, 4) + " " + e(z.imag(), 11, 4) +
           " " + e(y.real(), 11, 4) + " " + e(y.imag(), 11, 4) + " " + e(p, 11, 4);
}

// `fmt_network_row`
std::string fmt_network_row(ll tag_a, ll seg_a, ll tag_b, ll seg_b, const double v[6], const std::string &kind) {
    return " " + d(tag_a, 4) + " " + d(seg_a, 5) + " " + d(tag_b, 4) + " " + d(seg_b, 5) + "  " + e(v[0], 11, 4) +
           " " + e(v[1], 11, 4) + "  " + e(v[2], 11, 4) + " " + e(v[3], 11, 4) + "  " + e(v[4], 11, 4) + " " +
           e(v[5], 11, 4) + " " + rjust(kind, 8);
}

// `_CURRENT_ROW % row` over `fmt_current_rows`' columns
std::string fmt_current_row(ll seg, ll tag, const double c[3], double length, cd value) {
    const double mag = cabs(value);
    const double phase = mag != 0.0 ? py_degrees(py_atan2(value.imag(), value.real())) : 0.0;
    return " " + d(seg, 5) + " " + d(tag, 4) + " " + f(c[0], 9, 4) + " " + f(c[1], 9, 4) + " " + f(c[2], 9, 4) +
           " " + f(length, 9, 5) + " " + e(value.real(), 11, 4) + " " + e(value.imag(), 11, 4) + " " +
           e(mag, 11, 4) + " " + f(phase, 8, 3);
}

// `_loading_cell`
std::string loading_cell(bool present, double value) {
    if (!present || value == 0.0) return std::string(12, ' ');
    return e(value, 12, 4);
}

// --- the structure's tables -------------------------------------------------------

// a segment's end points as numpy writes them: p1 + k * step
void segment_point(const nec4::GeometryWire &w, ll k, double out[3]) {
    for (int c = 0; c < 3; c++) {
        const double step = (w.p2[c] - w.p1[c]) / static_cast<double>(w.n_seg);
        out[c] = w.p1[c] + static_cast<double>(k) * step;
    }
}

// `int(round(float(c) / eps))` per coordinate
std::vector<double> node_key(const double p[3]) {
    std::vector<double> k(3);
    for (int c = 0; c < 3; c++) {
        const double q = p[c] / 1e-9;
        if (!std::isfinite(q)) decline("a non-finite node coordinate");
        double r = std::nearbyint(q);  // round(): half to even
        if (r == 0.0) r = 0.0;
        k[static_cast<std::size_t>(c)] = r;
    }
    return k;
}

// `_segment_end_nodes(wires)`: the nodes in first-seen order
struct EndNodes {
    std::vector<std::vector<ll>> members;
    std::map<std::vector<double>, std::size_t> index;
};

EndNodes end_nodes(const std::vector<nec4::GeometryWire> &wires) {
    EndNodes en;
    ll n = 0;
    for (const nec4::GeometryWire &w : wires)
        for (ll k = 0; k < w.n_seg; k++) {
            n++;
            double a[3], b[3];
            segment_point(w, k, a);
            segment_point(w, k + 1, b);
            const std::pair<const double *, ll> pts[2] = {{a, -n}, {b, n}};
            for (const auto &pt : pts) {
                const std::vector<double> key = node_key(pt.first);
                auto it = en.index.find(key);
                if (it == en.index.end()) {
                    it = en.index.emplace(key, en.members.size()).first;
                    en.members.emplace_back();
                }
                en.members[it->second].push_back(pt.second);
            }
        }
    return en;
}

// `_junction_rows(wires)`
std::vector<std::string> junction_rows(const std::vector<nec4::GeometryWire> &wires) {
    const EndNodes en = end_nodes(wires);
    std::vector<std::string> rows;
    for (const std::vector<ll> &m0 : en.members) {
        if (m0.size() < 3) continue;
        std::vector<ll> m = m0;
        std::stable_sort(m.begin(), m.end(), [](ll x, ll y) { return std::llabs(x) < std::llabs(y); });
        std::string body = d(m[0], 11);
        for (std::size_t k = 1; k < m.size(); k++) body += d(m[k], 5);
        rows.push_back(d(static_cast<ll>(rows.size() + 1), 8) + body);
    }
    return rows;
}

// `_connection_data(wires, interpolates)`: NEC-2's (I-, I+) columns
std::vector<std::pair<ll, ll>> connection_data(const std::vector<nec4::GeometryWire> &wires, bool interpolates) {
    const EndNodes en = end_nodes(wires);
    std::vector<std::pair<ll, ll>> out;
    ll idx = 0;
    for (const nec4::GeometryWire &w : wires) {
        double step[3];
        for (int c = 0; c < 3; c++) step[c] = (w.p2[c] - w.p1[c]) / static_cast<double>(w.n_seg);
        const double slen = std::sqrt((step[0] * step[0] + step[1] * step[1]) + step[2] * step[2]) * kSmin;
        for (ll k = 0; k < w.n_seg; k++) {
            idx++;
            double a[3], b[3];
            segment_point(w, k, a);
            segment_point(w, k + 1, b);
            const std::vector<ll> &here = en.members[en.index.at(node_key(a))];
            const std::vector<ll> &there = en.members[en.index.at(node_key(b))];
            ll i_minus = 0, i_plus = 0;
            for (ll m : here)
                if (std::llabs(m) != idx) {
                    i_minus = m;
                    break;
                }
            for (ll m : there)
                if (std::llabs(m) != idx) {
                    i_plus = -m;
                    break;
                }
            if (interpolates) {
                if (std::fabs(a[2]) <= slen) i_minus = idx;
                if (std::fabs(b[2]) <= slen) i_plus = idx;
            }
            out.push_back({i_minus, i_plus});
        }
    }
    return out;
}

// Python's `a // b` and `a % b` on ints
ll floordiv(ll a, ll b) { return a / b - ((a % b != 0) && ((a < 0) != (b < 0))); }
ll pymod(ll a, ll b) { return ((a % b) + b) % b; }

// `_arc_or_helix_rows(card, wire_no, seg_total)`
void arc_or_helix_rows(const Card &card, ll wire_no, ll seg_total, std::vector<std::string> &rows) {
    const ll n_seg = i64(card.i(1));
    const Int tag = card.i(0);
    const ll first = seg_total + 1, last = seg_total + n_seg;
    if (card.mnemonic == "GA") {
        rows.push_back(" " + d(wire_no, 5) + " ARC RADIUS: " + f(card.f(2), 9, 5) + "  FROM: " + f(card.f(3), 8, 3) +
                       " TO: " + f(card.f(4), 8, 3) + " DEGREES       " + f(card.f(5), 11, 5) + " " + d(n_seg, 5) +
                       " " + d(first, 5) + " " + d(last, 5) + " " + d(tag, 4));
        return;
    }
    const double s = card.f(2), hl = card.f(3), a1 = card.f(4), b1_in = card.f(5), a2 = card.f(6), b2 = card.f(7),
                 rad = card.f(8);
    rows.push_back(" " + d(wire_no, 5) + " HELIX STRUCTURE - SPACING OF TURNS: " + f(s, 8, 3) + " AXIAL LENGTH: " +
                   f(hl, 8, 3) + "  " + f(rad, 8, 3) + " " + d(n_seg, 5) + " " + d(first, 5) + " " + d(last, 5) +
                   " " + d(tag, 4));
    rows.push_back("       RADIUS X1:" + f(a1, 8, 3) + " Y1:" + f(b1_in, 8, 3) + " X2:" + f(a2, 8, 3) +
                   " Y2:" + f(b2, 8, 3) + " ");
    if (a2 != a1) {
        const double den = std::fabs(hl) + (std::fabs(hl) * a1) / (a2 - a1);
        if (den == 0.0) decline("a helix cone the reference divides by zero on");
        const double cone = std::atan(a2 / den);
        rows.push_back("       THE CONE ANGLE OF THE SPIRAL IS " + f(cone, 10, 4));
        return;
    }
    double b1 = b1_in;
    if (b1 == 0.0) b1 = a1;
    double pitch, turn;
    if (a1 == b1) {
        const double hdia = 2.0 * a1;
        if (kPi * hdia == 0.0) decline("a helix pitch the reference divides by zero on");
        pitch = std::atan(s / (kPi * hdia));
        turn = hdia * kPi / std::cos(pitch);
        pitch = 180.0 * pitch / kPi;
    } else {
        const double hmaj = a1 >= b1 ? 2.0 * a1 : 2.0 * b1, hmin = a1 >= b1 ? 2.0 * b1 : 2.0 * a1;
        const double under = (hmaj * hmaj + hmin * hmin) / 2 * hmaj;
        if (!(under >= 0.0)) decline("a helix diameter the reference cannot take the root of");
        const double hdia = std::sqrt(under);
        if (kPi * hdia == 0.0) decline("a helix pitch the reference divides by zero on");
        turn = 2.0 * kPi * hdia;
        pitch = (180.0 / kPi) * std::atan(s / (kPi * hdia));
    }
    rows.push_back("       THE PITCH ANGLE IS: " + f(pitch, 0, 4) + "    THE LENGTH OF WIRE/TURN IS: " + f(turn, 0, 4));
}

// `_structure_rows(deck, solver)`
std::vector<std::string> structure_rows(const nec4::PortalDeck &deck, ll total) {
    std::vector<std::string> rows;
    ll wire_no = 0, seg_total = 0;
    for (const Card &card : deck.geometry) {
        const std::string &m = card.mnemonic;
        if (m == "GW") {
            wire_no++;
            const ll n_seg = i64(card.i(1));
            const double p1[3] = {card.f(2), card.f(3), card.f(4)}, p2[3] = {card.f(5), card.f(6), card.f(7)};
            rows.push_back(fmt_wire_row(wire_no, p1, p2, card.f(8), n_seg, seg_total + 1, seg_total + n_seg, card.i(0)));
            seg_total += n_seg;
        } else if (m == "GA" || m == "GH") {
            wire_no++;
            arc_or_helix_rows(card, wire_no, seg_total, rows);
            seg_total += i64(card.i(1));
        } else if (m == "GX") {
            const ll code = i64(card.i(1));
            const int ix = pymod(floordiv(code, 100), 10) != 0, iy = pymod(floordiv(code, 10), 10) != 0,
                      iz = pymod(code, 10) != 0;
            rows.push_back(std::string("      STRUCTURE REFLECTED ALONG THE AXES ") + (ix ? "X" : "*") + " " +
                           (iy ? "Y" : "*") + " " + (iz ? "Z" : "*") + " - TAGS INCREMENTED BY " + card.i(0).str());
            seg_total *= ll(1) << (ix + iy + iz);
        } else if (m == "GR") {
            const Int nop = card.i(1);
            rows.push_back("  STRUCTURE ROTATED ABOUT Z-AXIS " + nop.str() + " TIMES - LABELS INCREMENTED BY " +
                           card.i(0).str());
            const ll n = i64(nop);
            seg_total *= std::max<ll>(n, 1);
        } else if (m == "GS") {
            rows.push_back("     STRUCTURE SCALED BY FACTOR: " + f(card.f(2), 10, 5));
        } else if (m == "GM") {
            rows.push_back("     THE STRUCTURE HAS BEEN MOVED, MOVE DATA CARD IS:");
            std::string r = " " + d(card.i(0), 5) + " " + d(card.i(1), 5);
            for (int k = 0; k < 7; k++) r += " " + f(card.f(2 + static_cast<std::size_t>(k)), 10, 5);
            rows.push_back(r);
        }
    }
    if (deck.ground_plane_flag) {
        rows.emplace_back("");
        rows.emplace_back("     GROUND PLANE SPECIFIED.");
        if (deck.ground_plane_interpolates) {
            rows.emplace_back("     WHERE WIRE ENDS TOUCH GROUND, CURRENT WILL BE INTERPOLATED TO IMAGE IN GROUND PLANE.");
            rows.emplace_back("");
        }
    }
    rows.emplace_back("");
    rows.push_back("     TOTAL SEGMENTS USED: " + std::to_string(total) + "   SEGMENTS IN A SYMMETRIC CELL: " +
                   std::to_string(total) + "   SYMMETRY FLAG: 0");
    return rows;
}

// `_segmentation_rows(deck, solver)`
std::vector<std::string> segmentation_rows(const nec4::PortalDeck &deck, const nec4::BuildTables &bt) {
    const std::vector<std::pair<ll, ll>> cx = connection_data(deck.structure.wires, deck.ground_plane_interpolates);
    std::vector<std::string> rows;
    for (std::size_t i = 0; i < bt.segments.size(); i++) {
        const nec4::SegmentRow &seg = bt.segments[i];
        const nec4::Point3 &dd = seg.direction;
        const double length = std::sqrt((dd[0] * dd[0] + dd[1] * dd[1]) + dd[2] * dd[2]);
        const double alpha = py_degrees(py_atan2(dd[2], std::hypot(dd[0], dd[1])));
        const double beta = py_degrees(py_atan2(dd[1], dd[0]));
        rows.push_back(" " + d(seg.number, 5) + " " + f(seg.centre[0], 9, 4) + " " + f(seg.centre[1], 9, 4) + " " +
                       f(seg.centre[2], 9, 4) + " " + f(length, 9, 4) + " " + f(alpha, 9, 4) + " " + f(beta, 9, 4) +
                       " " + f(seg.radius, 9, 4) + " " + d(cx.at(i).first, 5) + " " + d(seg.number, 5) + " " +
                       d(cx.at(i).second, 5) + " " + d(seg.tag, 5));
    }
    return rows;
}

// `_loading_rows(cards)`: a PortalError (the run's ERROR frame) on a type
// the table has no column for
std::vector<std::string> loading_rows(const std::vector<Card> &cards) {
    std::vector<std::string> rows;
    for (const Card &card : cards) {
        const Int kind = card.i(0);
        bool present[6] = {false, false, false, false, false, false};
        double cells[6] = {0, 0, 0, 0, 0, 0};
        const char *tail;
        if (kind >= Int(0) && kind <= Int(3)) {
            for (int k = 0; k < 3; k++) {
                present[k] = true;
                cells[k] = card.f(4 + static_cast<std::size_t>(k));
            }
            tail = (kind == Int(1) || kind == Int(3)) ? "   PARALLEL" : "    SERIES ";
        } else if (kind == Int(4)) {
            present[3] = present[4] = true;
            cells[3] = card.f(4);
            cells[4] = card.f(5);
            tail = "   FIXED IMPEDANCE ";
        } else if (kind == Int(5)) {
            present[5] = true;
            cells[5] = card.f(4);
            tail = "     WIRE  ";
        } else {
            throw nec4::Refusal("LD type " + kind.str() + " is not supported by this engine");
        }
        std::string r = " " + d(card.i(1), 5) + " " + d(card.i(2), 4) + " " + d(card.i(3), 4);
        for (int k = 0; k < 6; k++) r += loading_cell(present[k], cells[k]);
        rows.push_back(r + tail);
    }
    return rows;
}

// `_environment_lines(ground, freq_mhz)`
std::vector<std::string> environment_lines(const nec4::Ground &ground, double freq_mhz) {
    const std::string pad(28, ' ');
    if (ground.kind == nec4::Ground::none) return {kEnvironmentHeader, pad + "FREE SPACE"};
    if (ground.kind == nec4::Ground::pec) return {kEnvironmentHeader, pad + "PERFECT GROUND"};
    const double omega = ((2.0 * kPi) * freq_mhz) * 1e6;
    const cd eps_c(ground.eps_r, (-ground.sigma) / (omega * kEps0));
    const bool refl = ground.kind == nec4::Ground::finite_fast;
    std::vector<std::string> lines{kEnvironmentHeader};
    if (!refl) lines.emplace_back("Somnec Computation Time 0");
    lines.push_back(pad + (refl ? "FINITE GROUND - REFLECTION COEFFICIENT APPROXIMATION"
                                : "FINITE GROUND - SOMMERFELD SOLUTION"));
    lines.push_back(pad + "RELATIVE DIELECTRIC CONST:" + f(ground.eps_r, 7, 3));
    lines.push_back(pad + "CONDUCTIVITY:" + e(ground.sigma, 11, 3) + " MHOS/METER");
    lines.push_back(pad + "COMPLEX DIELECTRIC CONSTANT:" + e(eps_c.real(), 12, 4) + e(eps_c.imag(), 11, 4) + "j");
    return lines;
}

// `_gain_db(power_gain)`
double gain_db(double gain) {
    if (gain < kGainFloor2) return -999.99;
    return 10.0 * std::log10(gain);
}

// `_far_readout._polarisation(e_theta, e_phi, floor_scale)`: (axial, tilt, sense)
void polarisation(cd et, cd ep, double floor_scale, double &axial, double &tilt, std::string &sense) {
    const double a = cabs(et), b = cabs(ep);
    const double raw2 = floor_scale * floor_scale;
    if ((a * a) * raw2 <= kFieldFloor2 && (b * b) * raw2 <= kFieldFloor2) {
        axial = tilt = 0.0;
        sense.clear();
        return;
    }
    double delta = phase_deg(ep) - phase_deg(et);
    // `(delta + 180.0) % 360.0 - 180.0`, CPython's float modulo
    double m = std::fmod(delta + 180.0, 360.0);
    if (m != 0.0) {
        if (m < 0.0) m += 360.0;
    } else {
        m = 0.0;
    }
    delta = m - 180.0;
    const double a2 = a * a, b2 = b * b;
    const double squared = (a2 * a2 + b2 * b2) + ((2.0 * a2) * b2) * std::cos(py_radians(2.0 * delta));
    const double root = std::sqrt(0.0 > squared ? 0.0 : squared);
    const double major2 = 0.5 * ((a2 + b2) + root);
    double minor2 = 0.5 * ((a2 + b2) - root);
    if (0.0 > minor2) minor2 = 0.0;
    tilt = 0.5 * py_degrees(py_atan2(((2.0 * a) * b) * std::cos(py_radians(delta)), a2 - b2));
    if (major2 <= 0.0) {
        axial = 0.0;
        sense = "LINEAR";
        return;
    }
    const double ratio = std::sqrt(minor2 / major2);
    const double sin_d = std::sin(py_radians(delta));
    if (ratio < 5.0e-5) {
        axial = 0.0;
        sense = "LINEAR";
        return;
    }
    axial = sin_d >= 0 ? ratio : -ratio;
    sense = sin_d >= 0 ? "LEFT" : "RIGHT";
}

int pattern_ground(const nec4::Ground &gr) {
    if (gr.kind == nec4::Ground::none) return 0;
    if (gr.kind == nec4::Ground::pec) return 1;
    return 2;
}

// `_pattern_lines(card, solver, result, freq_mhz)`
std::vector<std::string> pattern_lines(const Card &card, nec4::PortalSolve &ps, const nec4::PortalRun &r,
                                       const nec4::PortalGroup &group, double freq_mhz) {
    const ll mode = i64(card.i(0));
    const ll n_theta = std::max<ll>(i64(card.i(1)), 1), n_phi = std::max<ll>(i64(card.i(2)), 1);
    if (n_theta > 100000000 / n_phi) decline("a pattern grid past what this build sizes");
    const double theta0 = card.f(4), phi0 = card.f(5), d_theta = card.f(6), d_phi = card.f(7), rng = card.f(8);
    const bool at_range = rng >= kFieldFloor2;
    std::vector<double> thetas(static_cast<std::size_t>(n_theta)), phis(static_cast<std::size_t>(n_phi));
    for (ll i = 0; i < n_theta; i++) thetas[static_cast<std::size_t>(i)] = theta0 + d_theta * static_cast<double>(i);
    for (ll j = 0; j < n_phi; j++) phis[static_cast<std::size_t>(j)] = phi0 + d_phi * static_cast<double>(j);
    const double wavelength = r.wavelength;
    const double k = 2.0 * kPi / wavelength;
    const bool cliff_mode = mode == 2 || mode == 3;
    const nec4::Environment &env = group.environment;
    std::vector<double> mid;
    std::vector<cd> moment;
    ps.far_elements(r, mid, moment);
    // `prop = np.exp(-1j * k * rng) / rng if at_range else complex(1.0)`
    cd prop(1.0, 0.0);
    if (at_range) {
        const cd arg = py_mul(py_mul(cd(-0.0, -1.0), cd(k, 0.0)), cd(rng, 0.0));
#if defined(__GNUC__)
        __complex__ double z;
        __real__ z = arg.real();
        __imag__ z = arg.imag();
        z = __builtin_cexp(z);
        const cd ex(__real__ z, __imag__ z);
#else
        const cd ex = std::exp(arg);
#endif
        prop = rp::np_cdiv(ex, cd(rng, 0.0));
    }
    rp::FarCliff cliff;
    const rp::FarCliff *cliff_p = nullptr;
    if (cliff_mode && env.has_second_medium) {
        cliff.mode = static_cast<int>(mode);
        cliff.edge_distance = env.second_medium.edge_distance;
        cliff.height = env.second_medium.height;
        cliff.eps_r2 = env.second_medium.eps_r;
        cliff.sigma2 = env.second_medium.sigma;
        cliff_p = &cliff;
    }
    std::vector<cd> e_theta, e_phi;
    std::vector<double> g_v, g_h;
    try {
        rp::pattern_fields(thetas, phis, mid, moment, k, pattern_ground(env.ground), env.ground.eps_r,
                           env.ground.sigma, freq_mhz * 1e6, r.p_in, cliff_p, at_range ? &prop : nullptr, e_theta,
                           e_phi, g_v, g_h);
    } catch (const nec5::PrintoutDecline &x) {
        decline(x.reason);
    }
    const double floor_scale = 1.0 / (wavelength * cabs(prop));
    std::vector<std::string> out;
    if (cliff_mode) {
        // `_far_field_ground_lines(mode, second)`
        const std::string pad(31, ' ');
        const nec4::SecondMedium sm = env.has_second_medium ? env.second_medium : nec4::SecondMedium();
        out.emplace_back(kFarFieldGroundHeader);
        out.emplace_back("");
        out.emplace_back("");
        out.push_back(pad + "--- " + (mode == 2 ? "LINEAR" : "CIRCULAR") + " CLIFF ---");
        out.push_back(pad + "EDGE DISTANCE= " + f(sm.edge_distance, 9, 2) + " METERS");
        out.push_back(pad + "       HEIGHT= " + f(sm.height, 9, 2) + " METERS");
        out.push_back(pad + "--- SECOND MEDIUM ---");
        out.push_back(pad + "RELATIVE DIELECTRIC CONST= " + f(sm.eps_r, 10, 3));
        out.push_back(pad + "      GROUND CONDUCTIVITY= " + f(sm.sigma, 10, 3) + " MHOS");
        out.emplace_back("");
        out.emplace_back("");
    }
    out.emplace_back(kPatternHeader);
    if (at_range) {
        out.emplace_back("");
        out.push_back("                             RANGE:" + e(rng, 14, 6) + " METERS");
        out.push_back("                             EXP(-JKR)/R:" + e(1.0 / rng, 13, 5) + " AT PHASE:" +
                      f(py_degrees(py_atan2(prop.imag(), prop.real())), 8, 2) + " DEGREES");
    }
    out.emplace_back("");
    add(out, kPatternTableHeader);
    const double raw2 = floor_scale * floor_scale;
    ll xnda_digit = pymod(i64(card.i(3)), 10);
    const bool rows_printed = xnda_digit != 2;
    for (ll j = 0; j < (rows_printed ? n_phi : 0); j++)
        for (ll i = 0; i < n_theta; i++) {
            const std::size_t q = static_cast<std::size_t>(i * n_phi + j);
            cd et = e_theta[q], ep = e_phi[q];
            if ((et.real() * et.real() + et.imag() * et.imag()) * raw2 <= kFieldFloor2) et = cd(0.0, 0.0);
            if ((ep.real() * ep.real() + ep.imag() * ep.imag()) * raw2 <= kFieldFloor2) ep = cd(0.0, 0.0);
            double axial, tilt;
            std::string sense;
            polarisation(et, ep, floor_scale, axial, tilt, sense);
            out.push_back(f(thetas[static_cast<std::size_t>(i)], 8, 2) + f(phis[static_cast<std::size_t>(j)], 10, 2) +
                          g(gain_db(g_v[q]), 10, 2) + f(gain_db(g_h[q]), 9, 2) + f(gain_db(g_v[q] + g_h[q]), 9, 2) +
                          f(axial, 12, 4) + f(tilt, 10, 2) + " " + ljust(sense, 6) + e(cabs(et), 12, 4) +
                          f(phase_deg(et), 10, 2) + e(cabs(ep), 12, 4) + f(phase_deg(ep), 10, 2));
        }
    out.emplace_back("");
    out.emplace_back("");
    if (xnda_digit != 0) {
        // `average_gain(g_v + g_h, thetas, d_theta, d_phi, n_phi)`, nec2c's rule
        std::vector<double> band(static_cast<std::size_t>(n_theta));
        const auto np_max = [](double x, double y) { return std::isnan(x) ? x : std::isnan(y) ? y : (x >= y ? x : y); };
        const auto np_min = [](double x, double y) { return std::isnan(x) ? x : std::isnan(y) ? y : (x <= y ? x : y); };
        for (ll it = 0; it < n_theta; it++) {
            const std::size_t t = static_cast<std::size_t>(it);
            const double lo = np_max(thetas[t] - 0.5 * d_theta, thetas[0]) * kDegToRad;
            const double hi = np_min(thetas[t] + 0.5 * d_theta, thetas[static_cast<std::size_t>(n_theta - 1)]) * kDegToRad;
            band[t] = std::cos(lo) - std::cos(hi);
        }
        const ll columns = std::max<ll>(n_phi - 1, 1);
        const double step = d_phi != 0.0 ? py_radians(d_phi) : 2.0 * kPi;
        std::vector<double> weighted;
        for (ll it = 0; it < n_theta; it++)
            for (ll ip = 0; ip < columns; ip++) {
                const std::size_t q = static_cast<std::size_t>(it * n_phi + ip);
                weighted.push_back((g_v[q] + g_h[q]) * band[static_cast<std::size_t>(it)]);
            }
        const double total = (0.0 + rp::np_pairwise(weighted.data(), static_cast<int64_t>(weighted.size()))) * step;
        const double solid = ((0.0 + rp::np_pairwise(band.data(), n_theta)) * static_cast<double>(columns)) * step;
        const double average = solid != 0.0 ? total / solid : 0.0;
        out.push_back("  AVERAGE POWER GAIN:" + e(average, 12, 4) + " - SOLID ANGLE USED IN AVERAGING: (" +
                      f_signed(solid / kPi, 7, 4) + ")*PI STERADIANS");
    }
    out.emplace_back(kPatternTime);
    return out;
}

// `_near_field_lines(card, solver, result)`
std::vector<std::string> near_field_lines(const Card &card, nec4::PortalSolve &ps, const nec4::PortalRun &r,
                                          const nec4::PortalGroup &group) {
    const bool magnetic = card.mnemonic == "NH";
    if (group.environment.ground.kind == nec4::Ground::finite_fast)
        throw nec4::Refusal(card.mnemonic + " over the GN 0 reflection-coefficient ground");  // never reached
    const nec4::PrintNearBlock b = ps.near_field(card, r);
    std::vector<std::string> out;
    out.emplace_back(magnetic ? kNearHHeader : kNearEHeader);
    if (magnetic) {
        out.emplace_back("");
        add(out, kNearHTableHeader);
    } else {
        add(out, kNearETableHeader);
    }
    for (const nec4::PrintNearRow &row : b.rows) {
        std::string l = f(row.point[0], 10, 4) + f(row.point[1], 10, 4) + f(row.point[2], 10, 4);
        for (int c = 0; c < 3; c++) l += e(row.magnitudes[c], 13, 4) + f(row.phases_deg[c], 8, 2);
        out.push_back(l);
    }
    out.emplace_back(kNearFieldTime);
    return out;
}

// the rows of `_printed_segments` / `_charge_segments` as a range of
// global numbers [first, last] (everything when `all`)
struct Rows {
    bool all = true;
    ll first = 0, last = 0;
    bool keeps(ll n) const { return all || (first <= n && n <= last); }
};

// `_seg_table(...).lines()` over the rows `keep` leaves standing
void seg_table_lines(const nec4::BuildTables &bt, const Rows &keep, const std::vector<cd> &values,
                     double wavelength, std::vector<std::string> &out) {
    for (const nec4::SegmentRow &seg : bt.segments) {
        if (!keep.keeps(seg.number)) continue;
        const nec4::Point3 &dd = seg.direction;
        const double length = std::sqrt((dd[0] * dd[0] + dd[1] * dd[1]) + dd[2] * dd[2]);
        const double c[3] = {seg.centre[0] / wavelength, seg.centre[1] / wavelength, seg.centre[2] / wavelength};
        out.push_back(fmt_current_row(seg.number, i64(seg.tag), c, length / wavelength,
                                      values.at(static_cast<std::size_t>(seg.number - 1))));
    }
}

// `_network_lines(solver, result, group_index)`
std::vector<std::string> network_lines(nec4::PortalSolve &ps, const nec4::BuildTables &bt, const nec4::PortalRun &r,
                                       ll group_index) {
    std::vector<std::string> out;
    if (r.network_points.empty()) return out;
    const std::vector<const nec4::NetworkCard *> live = ps.live_networks(group_index);
    out.emplace_back(kNetworkDataHeader);
    std::vector<std::string> kinds;
    for (const nec4::NetworkCard *c : live)
        if (std::find(kinds.begin(), kinds.end(), c->kind) == kinds.end()) kinds.push_back(c->kind);
    for (const std::string &kind : kinds) {
        if (kind == "TL")
            add(out, kNetworkTlTableHeader);
        else
            add(out, kNetworkNtTableHeader);
        for (const nec4::NetworkCard *c : live) {
            if (c->kind != kind) continue;
            ll ends[4];
            const Int *addr[2] = {c->address_a, c->address_b};
            for (int en = 0; en < 2; en++) {
                const ll segment = ps.global_segment(addr[en][0], addr[en][1]);
                ends[2 * en] = i64(bt.segments.at(static_cast<std::size_t>(segment - 1)).tag);
                ends[2 * en + 1] = segment;
            }
            double v[6];
            for (int q = 0; q < 6; q++) v[q] = c->payload[q];
            std::string word;
            if (c->kind == "TL") {
                double z0, length;
                bool crossed;
                ps.tl_line(*c, z0, length, crossed);
                v[0] = z0;
                v[1] = length;
                word = crossed ? "CROSSED" : "STRAIGHT";
            }
            out.push_back(fmt_network_row(ends[0], ends[1], ends[2], ends[3], v, word));
        }
    }
    out.emplace_back("");
    out.emplace_back("");
    out.emplace_back(kNetworkExcitationHeader);
    add(out, kNetworkExcitationTableHeader);
    for (const auto &pt : r.network_points) {
        const ll tag = pt.first, segment = pt.second.first, port = pt.second.second;
        const cd volts = r.v_applied.at(static_cast<std::size_t>(port));
        cd current = r.i_port.at(static_cast<std::size_t>(port));
        if (current.real() * current.real() + current.imag() * current.imag() <= kDustFloor2) current = cd(0.0, 0.0);
        const cd z = current != cd(0.0, 0.0) ? py_div(volts, current) : cd(0.0, 0.0);
        const cd y = volts != cd(0.0, 0.0) ? py_div(current, volts) : cd(0.0, 0.0);
        const double p = 0.5 * np_smul(volts, std::conj(current)).real();
        out.push_back(fmt_aip_row(tag, segment, volts, current, z, y, p));
    }
    out.emplace_back("");
    out.emplace_back("");
    return out;
}

// `_run_block(deck, solver, group, freq, group_index)`: `_run_records` and
// `_render_run_block`
std::vector<std::string> run_block(nec4::PortalSolve &ps, const nec4::BuildTables &bt, const nec4::PortalGroup &group,
                                   double freq_mhz, ll group_index) {
    const nec4::PortalRun r = ps.solve_group(group, freq_mhz, group_index);
    const double wavelength = r.wavelength;
    std::vector<std::string> out;
    if (group.refilled) {
        out.emplace_back(kFrequencyHeader);
        out.push_back("                                FREQUENCY : " + e(freq_mhz, 10, 4) + " MHz");
        out.push_back("                                WAVELENGTH: " + e(wavelength, 10, 4) + " Mtr");
        out.emplace_back("");
        add(out, kApproxIntegration);
        if (group.ek) out.emplace_back("                        THE EXTENDED THIN WIRE KERNEL WILL BE USED");
        out.emplace_back("");
        out.emplace_back("");
    }
    if (group.refilled || group.refilled_partial) {
        out.emplace_back(kLoadingHeader);
        const std::vector<std::string> rows = loading_rows(group.loads);
        if (!rows.empty()) {
            add(out, kLoadingTableHeader);
            out.insert(out.end(), rows.begin(), rows.end());
            if (ps.loaded_twice(group_index)) out.emplace_back(kLoadedTwice);
        } else {
            out.emplace_back(kLoadingNone);
        }
        out.emplace_back("");
        out.emplace_back("");
        const std::vector<std::string> env = environment_lines(group.environment.ground, freq_mhz);
        out.insert(out.end(), env.begin(), env.end());
        if (group.has_mp && group.mp_processors != Int(0) && group.mp_processors != Int(1)) {
            out.push_back("MP: multiProcessor " + group.mp_processors.str() + " " + group.mp_block_size.str());
            out.emplace_back("");
        }
        if (!r.end_cap_withdrawn.empty()) {
            // `_nec2_end_cap.withdrawn_notice(reason)`
            // `_nec2_end_cap.WITHDRAWN`: the below-interface refusal names buried wires
            const bool below = r.end_cap_withdrawn.find("buried") != std::string::npos;
            out.push_back(std::string("NEC-2 END CAP NOT APPLIED: ") +
                          (below ? "a buried or ground-crossing wire" : "a junction the NEC-4 rule solves") +
                          " has a free end");
            out.emplace_back("");
        }
        out.emplace_back("");
        out.emplace_back("");
        out.emplace_back(kMatrixTimingHeader);
        const ll fill_ms = static_cast<ll>(std::nearbyint(r.fill_seconds * 1000.0));
        const ll factor_ms = static_cast<ll>(std::nearbyint(r.factor_seconds * 1000.0));
        out.push_back("                               FILL: " + std::to_string(fill_ms) + " msec  FACTOR: " +
                      std::to_string(factor_ms) + " msec");
        out.emplace_back("");
        out.emplace_back("");
    }
    const std::vector<std::string> net = network_lines(ps, bt, r, group_index);
    out.insert(out.end(), net.begin(), net.end());
    out.emplace_back(kAipHeader);
    add(out, kAipTableHeader);
    for (const auto &dv : r.driven) {
        const ll port = dv.first, seg = dv.second.first;
        const cd volts = dv.second.second;
        const cd current = r.i_source.at(static_cast<std::size_t>(port));
        const cd z = current != cd(0.0, 0.0) ? rp::np_cdiv(volts, current) : cd(0.0, 0.0);
        const cd y = volts != cd(0.0, 0.0) ? rp::np_cdiv(current, volts) : cd(0.0, 0.0);
        const double p = 0.5 * np_smul(volts, std::conj(current)).real();
        out.push_back(fmt_aip_row(i64(bt.segments.at(static_cast<std::size_t>(seg - 1)).tag), seg, volts, current, z,
                                  y, p));
    }
    // PT (`PrintControl`): suppressed, or a restricted range
    const bool suppressed =
        group.has_pt && (group.pt.flag == Int(-1) || (group.pt.flag > Int(0) && !group.pt.first.is_zero()));
    if (!suppressed) {
        Rows keep;
        if (group.has_pt && group.pt.flag.is_zero() && !group.pt.first.is_zero()) {
            const Int lo = group.pt.first, hi = group.pt.last.is_zero() ? group.pt.first : group.pt.last;
            keep.all = false;
            keep.first = ps.global_segment(group.pt.tag, lo);
            keep.last = ps.global_segment(group.pt.tag, hi);
        }
        out.emplace_back("");
        out.emplace_back("");
        out.emplace_back(kCurrentsHeader);
        out.emplace_back(kCurrentsNote);
        out.emplace_back("");
        add(out, kCurrentsTableHeader);
        seg_table_lines(bt, keep, r.segment_currents, wavelength, out);
    }
    // PQ (`ChargePrintControl`): off unless a card asked
    if (group.has_pq && group.pq.flag != Int(-1)) {
        Rows keep;
        if (!group.pq.first.is_zero() || !group.pq.last.is_zero()) {
            keep.all = false;
            keep.first = ps.global_segment(group.pq.tag, group.pq.first);
            keep.last = ps.global_segment(group.pq.tag, group.pq.last);
        }
        out.emplace_back("");
        out.emplace_back("");
        out.emplace_back(kChargeHeader);
        out.emplace_back(kChargeNote);
        out.emplace_back("");
        add(out, kChargeTableHeader);
        seg_table_lines(bt, keep, r.segment_charges, wavelength, out);
    }
    const std::string pad(31, ' ');
    out.emplace_back("");
    out.emplace_back("");
    out.emplace_back(kPowerHeader);
    out.push_back(pad + "INPUT POWER   =" + e(r.p_in, 12, 4) + " Watts");
    out.push_back(pad + "RADIATED POWER=" + e(r.p_radiated, 12, 4) + " Watts");
    out.push_back(pad + "STRUCTURE LOSS=" + e(r.p_structure, 12, 4) + " Watts");
    out.push_back(pad + "NETWORK LOSS  =" + e(r.p_network, 12, 4) + " Watts");
    out.push_back(pad + "EFFICIENCY    =" + f(r.efficiency, 8, 2) + " Percent");
    if (group.has_report) {
        const std::vector<std::string> body = group.report.mnemonic == "RP"
                                                  ? pattern_lines(group.report, ps, r, group, freq_mhz)
                                                  : near_field_lines(group.report, ps, r, group);
        out.emplace_back("");
        out.emplace_back("");
        out.insert(out.end(), body.begin(), body.end());
        out.emplace_back("");
    }
    return out;
}

bool deck_error(const nec5::ParseError &x) {
    return x.kind == nec5::ErrorKind::deck_error;
}

}  // namespace

std::vector<std::string> banner_lines(const std::string &suffix) {
    return {"",
            "",
            "",
            "                               __________________________________________",
            "                              |                                          |",
            "                              |  NUMERICAL ELECTROMAGNETICS CODE (nec2c) |",
            "                              |   Translated to 'C' in Double Precision  |",
            "                              |__________________________________________|",
            "",
            "VERSION:nec2c.ae6ty.momwire.9.1" + suffix};
}

std::string fmt_data_card(long long number, const Card &card) {
    std::string ints;
    for (std::size_t k = 0; k < 4; k++) ints += k == 0 ? " " + d(card.i(k), 3) : d(card.i(k), 6);
    std::string reals;
    for (std::size_t k = 0; k < 6; k++) reals += e(card.f(4 + k), 13, 5);
    return "  DATA CARD No:" + d(number, 4) + " " + card.mnemonic + ints + reals;
}

std::string sentinel_card(long long number, const std::string &terminator) {
    Card c;
    c.mnemonic = terminator;
    c.raw = terminator;
    return fmt_data_card(number, c);
}

void append_error(std::vector<std::string> &out, const std::string &reason) {
    out.push_back("ERROR: " + reason);
    out.push_back("ERROR-NEC2C: " + reason);
}

bool render_deck(const std::string &body, const DeckOptions &options, nec4::EngineFactory &engines,
                 const nec5::NumpyLinalg &np, std::vector<std::string> &out, std::vector<std::string> &err,
                 std::string &why) {
    out.assign(3, std::string());
    err.clear();
    why.clear();
    try {
        nec4::StructureSeam ss;
        try {
            ss.deck = nec4::parse_portal_deck(body, nec4::Dialect::nec2);
        } catch (const nec5::ParseError &x) {
            if (!deck_error(x)) decline(std::string("a deck the reference raises ") + nec5::error_kind_name(x.kind) + " on");
            append_error(out, x.message);
            return true;
        }
        const nec4::PortalDeck &deck = ss.deck;
        if (deck.has_reduced_field) err.push_back("reducedField:" + deck.reduced_field.str());
        out.emplace_back(kCommentsHeader);
        for (const std::string &c : deck.comments) out.push_back(kCommentIndent + c);
        out.insert(out.end(), 3, std::string());

        ss.kind = nec4::StructureSeam::served;
        for (std::size_t g = 0; g < deck.group_ran.size(); g++)
            if (deck.group_ran[g]) ss.plan.push_back(static_cast<ll>(g));
        nec4::StructureBuild sb;
        nec4::build_portal_structure(ss, options.basis, sb);
        if (sb.kind == nec4::StructureBuild::refused) {
            append_error(out, sb.message);
            return true;
        }
        if (sb.kind == nec4::StructureBuild::raised) {
            if (sb.error_kind != nec5::ErrorKind::deck_error && sb.error_kind != nec5::ErrorKind::value_error)
                decline(std::string("a build the reference raises ") + nec5::error_kind_name(sb.error_kind) + " on");
            append_error(out, sb.message);
            return true;
        }
        if (sb.kind != nec4::StructureBuild::built) decline(sb.message.empty() ? "a build this copy declines" : sb.message);
        for (nec4::OperatingPoint &op : sb.points) op.args.nec2_end_cap = options.nec2_end_cap;
        nec4::RunEngine &engine = engines.engine_for(sb);
        nec4::PortalSolve ps(ss, sb, engine, np);
        try {
            ps.seed();  // `DeckSolver.__init__` fills the first operating point
        } catch (const nec4::Refusal &x) {
            append_error(out, x.message);
            return true;
        }
        const nec4::BuildTables &bt = sb.tables;
        const ll total = static_cast<ll>(bt.segments.size());

        out.emplace_back(kStructureHeader);
        add(out, kStructureNotes);
        out.emplace_back("");
        add(out, kWireTableHeader);
        const std::vector<std::string> srows = structure_rows(deck, total);
        out.insert(out.end(), srows.begin(), srows.end());
        out.emplace_back("");
        const std::vector<std::string> junctions = junction_rows(deck.structure.wires);
        if (!junctions.empty()) {
            add(out, kJunctionsHeader);
            out.insert(out.end(), junctions.begin(), junctions.end());
            out.emplace_back("");
        }
        out.emplace_back("");
        if (!deck.quiet) {
            out.emplace_back(kSegmentationHeader);
            add(out, kSegmentationNotes);
            out.emplace_back("");
            add(out, kSegmentationTableHeader);
            const std::vector<std::string> rows = segmentation_rows(deck, bt);
            out.insert(out.end(), rows.begin(), rows.end());
            out.emplace_back("");
            out.emplace_back("");
        }
        out.emplace_back("");

        std::size_t group_index = 0;
        ll number = 0;
        for (const Card &card : deck.data_cards) {
            number++;
            out.push_back(fmt_data_card(number, card));
            if (!is_execute(card.mnemonic)) continue;
            if (group_index >= deck.groups.size()) decline("an execute card the walk did not record");
            const std::size_t index = group_index++;
            if (!deck.group_ran[index]) continue;
            const nec4::PortalGroup &group = deck.groups[index];
            out.emplace_back("");
            out.emplace_back("");
            try {
                for (std::size_t i = 0; i < group.freqs_mhz.size(); i++) {
                    if (i) out.insert(out.end(), 2, std::string());
                    const std::vector<std::string> rb =
                        run_block(ps, bt, group, group.freqs_mhz[i], static_cast<ll>(index));
                    out.insert(out.end(), rb.begin(), rb.end());
                }
            } catch (const nec4::Refusal &x) {
                append_error(out, x.message);
            }
            out.insert(out.end(), 3, std::string());
        }
        return true;
    } catch (const Declined &x) {
        why = x.what();
    } catch (const nec5::PrintoutDecline &x) {
        why = x.reason;
    } catch (const nec5::ParseError &x) {
        why = std::string("a deck the reference raises ") + nec5::error_kind_name(x.kind) + " on";
    }
    out.clear();
    err.clear();
    return false;
}

long long largest_solve(const std::string &body) {
    try {
        const nec4::PortalDeck deck = nec4::parse_portal_deck(body, nec4::Dialect::nec2);
        long long n = 0;
        for (const nec4::GeometryWire &w : deck.structure.wires) n += w.n_seg;
        return n;
    } catch (const std::exception &) {
        return -1;
    }
}

}  // namespace nec2
}  // namespace mw
