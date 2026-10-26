// The NEC-4.2 printout writer in C++ (see `_nec4_printout_core.h`). Each
// function names the Python it copies. Every number is formatted the way
// Python's format mini-language formats it (`nec5::py_format_e/f`, which
// spell NaN and the infinities as Python does), and every `str` method used
// on the latin-1 card text (ljust, center, slices) counts code points, which
// for latin-1 are bytes; text is converted to UTF-8 as it is emitted.
#include "_nec4_printout_core.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

namespace mw {
namespace nec4 {

namespace {

typedef long long ll;
typedef std::vector<std::string> Lines;

std::string sp(std::size_t n) { return std::string(n, ' '); }

void add(Lines &out, const Lines &more) { out.insert(out.end(), more.begin(), more.end()); }
void blanks(Lines &out, int n) { out.insert(out.end(), static_cast<std::size_t>(n), std::string()); }

std::string join(const Lines &lines) {
    std::string out;
    for (std::size_t k = 0; k < lines.size(); k++) {
        if (k) out += '\n';
        out += lines[k];
    }
    return out;
}

// format(n, "Nd")
std::string d(ll v, int width) {
    char buf[64];
    std::snprintf(buf, sizeof buf, "%*lld", width, v);
    return buf;
}
std::string d(const Int &v, int width) {
    std::string s = v.str();
    return s.size() >= static_cast<std::size_t>(width) ? s : sp(static_cast<std::size_t>(width) - s.size()) + s;
}
std::string e(double x, int width, int digits) { return nec5::py_format_e(x, width, digits); }
std::string f(double x, int width, int digits) { return nec5::py_format_f(x, width, digits); }

// `_nec4_printout._fixed_or_stars`: width asterisks when the value does not
// fit its fixed field (momwire#1694)
std::string f_or_stars(double x, int width, int digits) {
    std::string text = f(x, width, digits);
    return static_cast<int>(text.size()) > width ? std::string(static_cast<std::size_t>(width), '*') : text;
}

// `_printout._e_fortran`: a three-digit exponent drops its E.
std::string e_fortran(double x, int width, int digits) {
    std::string text = nec5::py_format_e(x, 0, digits);
    std::size_t at = text.find('E');
    if (at != std::string::npos) {
        int exponent = std::atoi(text.c_str() + at + 1);
        if (exponent >= 100 || exponent <= -100) text = text.substr(0, at) + text.substr(at + 1);
    }
    return text.size() >= static_cast<std::size_t>(width) ? text
                                                           : sp(static_cast<std::size_t>(width) - text.size()) + text;
}

// str.ljust / str.rjust / str.center over code points (the text is UTF-8;
// every string padded here is ASCII or latin-1 already widened, so its
// length in code points is counted from the UTF-8).
std::size_t code_points(const std::string &utf8) {
    std::size_t n = 0;
    for (unsigned char c : utf8)
        if ((c & 0xc0) != 0x80) n++;
    return n;
}
std::string ljust(const std::string &s, std::size_t width) {
    std::size_t n = code_points(s);
    return n >= width ? s : s + sp(width - n);
}
std::string rjust(const std::string &s, std::size_t width) {
    std::size_t n = code_points(s);
    return n >= width ? s : sp(width - n) + s;
}
// CPython's pad for str.center: left = marg / 2 + (marg & width & 1).
std::string center(const std::string &s, std::size_t width) {
    std::size_t n = code_points(s);
    if (n >= width) return s;
    std::size_t marg = width - n;
    std::size_t left = marg / 2 + (marg & width & 1);
    return sp(left) + s + sp(marg - left);
}
std::string rstrip(std::string s) {
    while (!s.empty() && s.back() == ' ') s.pop_back();
    return s;
}

// --- the header (`_printout.render_header`, dialect "nec4") ----------------

Lines banner() {
    const std::string indent = sp(32), rule(47, '*');
    const std::string title = ljust("*  NUMERICAL ELECTROMAGNETICS CODE (NEC-4.2)", 46) + "*";
    return {indent + rule, indent + "*" + sp(45) + "*", indent + title, indent + "*" + sp(45) + "*",
            indent + rule, "", "", "", ""};
}

// `comment_cards`: the leading CM run through CE, each card's columns 3-80.
Lines comment_texts(const std::string &latin1) {
    Lines out;
    for (const std::string &line : nec5::latin1_splitlines(latin1)) {
        const std::string mnemonic = nec5::latin1_upper(line.substr(0, std::min<std::size_t>(2, line.size())));
        const std::string text = line.size() > 2 ? line.substr(2, 78) : std::string();
        if (mnemonic == "CM") {
            out.push_back(text);
        } else if (mnemonic == "CE") {
            out.push_back(text);
            break;
        } else {
            break;
        }
    }
    return out;
}

std::string header(const std::string *deck_text, const std::string &basis, const nec5::EngineStamp &stamp) {
    Lines lines = {"1", nec5::latin1_to_utf8(nec5::engine_stamp(basis, stamp)), "", ""};
    add(lines, banner());
    if (deck_text != nullptr) {
        const std::string rule = sp(22) + std::string(58, '*');
        lines.push_back(rule);
        lines.push_back("");
        for (const std::string &text : comment_texts(*deck_text)) {
            std::string padded = text.size() >= 78 ? text : text + sp(78 - text.size());
            lines.push_back(sp(25) + nec5::latin1_to_utf8(padded));
        }
        lines.push_back("");
        lines.push_back(rule);
    }
    blanks(lines, 5);
    return join(lines) + "\n";
}

// --- card echo -----------------------------------------------------------------

std::vector<nec5::Card> post_ge_cards(const std::string &text) {
    std::vector<nec5::Card> cards;
    bool seen_ge = false;
    for (const std::string &line : nec5::latin1_splitlines(text)) {
        nec5::Card card;
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

// `_printout._card_echo`
std::string card_echo(ll index, const nec5::Card &card) {
    std::string ints;
    for (std::size_t k = 0; k < 4; k++) ints += d(card.i(k), k == 0 ? 4 : 5);
    std::string reals;
    for (std::size_t k = 0; k < 6; k++) reals += e(card.f(4 + k), 13, 5);
    std::string line = " ***** INPUT LINE" + d(index, 3) + "  " + card.mnemonic + ints + reals;
    if (card.has_trailer) {
        std::string trailer = card.trailer.size() >= 40 ? card.trailer : card.trailer + sp(40 - card.trailer.size());
        line += "\n" + sp(43) + e(card.f(10), 13, 5) + "  " + nec5::latin1_to_utf8(trailer);
    }
    return line;
}

// `_nec4_printout._echo`
Lines echo(ll index, const nec5::Card &card) {
    std::string line = card_echo(index, card);
    if (card.mnemonic == "EK") return {line, "", " THE EK AND KH COMMANDS HAVE NO EFFECT IN NEC-4", ""};
    return {line};
}

// --- rows ------------------------------------------------------------------------

// The wire table's last four columns (`_nec4_printout._segment_columns`).
std::string segment_columns(const WireTableRow &r) {
    return d(r.segments, 7) + d(r.first, 9) + d(r.last, 6) + d(r.tag, 8);
}

// Python's `f"{s:<width}"`: pad on the right, never cut.
std::string left(const std::string &s, std::size_t width) {
    return s.size() >= width ? s : s + std::string(width - s.size(), ' ');
}

Lines structure_rows(const WireTableRow &r) {
    if (r.kind == WireTableRow::arc) {
        std::string head = d(r.number, 6) + "  ARC RADIUS =" + f(r.arc_radius, 9, 5) + "  FROM" +
                           f(r.angle1_deg, 8, 3) + " TO" + f(r.angle2_deg, 8, 3) + " DEGREES";
        return {left(head, 73) + f(r.radius, 11, 5) + segment_columns(r)};
    }
    if (r.kind == WireTableRow::helix) {
        std::string head = d(r.number, 6) + "      THIS WIRE IS A LOG-SPIRAL OR HELIX";
        return {left(head, 84) + segment_columns(r),
                "          SPIRAL DATA: TURNS=" + f(r.turns, 10, 4) + "  LENGTH=" + e(r.length, 12, 4) +
                    "  H.RAD=" + e(r.helix_radii[0], 12, 4) + e(r.helix_radii[1], 12, 4) +
                    "  W.RAD=" + e(r.wire_radii[0], 12, 4) + e(r.wire_radii[1], 12, 4),
                "          TOTAL LENGTH OF WIRE IN THE SPIRAL =" + e(r.wire_length, 13, 5)};
    }
    if (r.kind == WireTableRow::catenary) {
        WireTableRow w = r;
        w.kind = WireTableRow::wire;
        Lines lines = {"          THE FOLLOWING WIRE IS A CATENARY," + d(r.icat, 5) + e(r.rhm, 12, 5) +
                           e(r.zm, 12, 5),
                       "          Catenary length =" + e(r.length, 12, 5) + "    Total segment length =" +
                           e(r.wire_length, 12, 5)};
        for (const std::string &s : structure_rows(w)) lines.push_back(s);
        return lines;
    }
    if (r.kind == WireTableRow::scale) return {"      STRUCTURE SCALED BY FACTOR" + f(r.factor, 10, 5)};
    if (r.kind == WireTableRow::reflect) {
        std::string axes = std::string(1, r.planes[0] ? 'X' : ' ') + " " + (r.planes[1] ? 'Y' : ' ') + " " +
                           (r.planes[2] ? 'Z' : ' ');
        return {"      STRUCTURE REFLECTED ALONG THE AXES " + axes + ".  TAGS INCREMENTED BY" +
                d(r.tag_increment, 5)};
    }
    if (r.kind == WireTableRow::move) {
        std::string numbers;
        for (double v : r.rotation_deg) numbers += f_or_stars(v, 10, 5);
        for (double v : r.translation) numbers += f_or_stars(v, 10, 5);
        return {"      THE STRUCTURE HAS BEEN MOVED, GM COMMAND DATA IS -",
                "    " + d(r.tag_increment, 5) + d(r.copies, 5) + numbers + d(r.first_tag, 5) + d(1, 5) +
                    d(0, 5) + d(r.move_segments, 5)};
    }
    return {d(r.number, 6) + f(r.end1[0], 11, 5) + f(r.end1[1], 11, 5) + f(r.end1[2], 11, 5) +
            f(r.end2[0], 12, 5) + f(r.end2[1], 11, 5) + f(r.end2[2], 11, 5) + f(r.radius, 11, 5) +
            d(r.segments, 7) + d(r.first, 9) + d(r.last, 6) + d(r.tag, 8)};
}

std::string junction_row(ll number, const std::vector<ll> &members) {
    std::string s = d(number, 6) + d(members[0], 10);
    for (std::size_t k = 1; k < members.size(); k++) s += d(members[k], 5);
    return s;
}

std::string segment_row(const SegmentDataRow &r) {
    return d(r.number, 6) + f(r.centre[0], 10, 5) + f(r.centre[1], 10, 5) + f(r.centre[2], 10, 5) +
           f(r.length, 10, 5) + f(r.alpha_deg, 11, 5) + f(r.beta_deg, 10, 5) + f(r.radius, 10, 5) +
           d(r.i_minus, 6) + d(r.number, 5) + d(r.i_plus, 5) + d(r.tag, 7);
}

// canvas[end - len(text) : end] = list(text), Python's slice rules.
void place(std::string &canvas, const std::string &text, ll end) {
    const ll n = static_cast<ll>(canvas.size());
    ll a = end - static_cast<ll>(text.size()), b = end;
    if (a < 0) a = std::max(0LL, a + n);
    if (b < 0) b = std::max(0LL, b + n);
    a = std::min(a, n);
    b = std::min(b, n);
    if (b < a) b = a;
    canvas.replace(static_cast<std::size_t>(a), static_cast<std::size_t>(b - a), text);
}

// `_printout._load_row` (the shapes NEC-4.2 shares with NEC-5)
std::string nec5_load_row(const PrintLoadRow &row) {
    const std::string &k = row.kind;
    if (k == "SERIES" || k == "PARALLEL" || k == "SERIES (PER METER)" || k == "PARALLEL (PER METER)") {
        bool per_metre = k.size() > 11 && k.compare(k.size() - 11, 11, "(PER METER)") == 0;
        std::string canvas = sp(per_metre ? 122 : 110);
        if (row.tag != 0) place(canvas, std::to_string(row.tag), 8);
        place(canvas, std::to_string(row.node_from), 13);
        place(canvas, std::to_string(row.node_thru), 18);
        if (row.has_resistance && row.resistance != 0.0) place(canvas, e(row.resistance, 11, 4), 34);
        if (row.has_inductance && row.inductance != 0.0) place(canvas, e(row.inductance, 11, 4), 47);
        if (row.has_capacitance && row.capacitance != 0.0) place(canvas, e(row.capacitance, 11, 4), 60);
        if (per_metre) {
            place(canvas, center(k, 20), 122);
        } else {
            place(canvas, center(k, 8), 110);
        }
        return canvas;
    }
    if (row.has_conductivity) {
        std::string canvas = sp(110);
        if (row.tag != 0) place(canvas, std::to_string(row.tag), 8);
        place(canvas, std::to_string(row.node_from), 13);
        place(canvas, std::to_string(row.node_thru), 18);
        place(canvas, e(row.conductivity, 11, 4), 99);
        place(canvas, k, 108);
        return canvas;
    }
    std::string canvas = sp(118);
    place(canvas, std::to_string(row.tag), 8);
    place(canvas, std::to_string(row.node_from), 13);
    place(canvas, std::to_string(row.node_thru), 18);
    place(canvas, e(row.resistance, 11, 4), 73);
    if (row.has_reactance) place(canvas, e(row.reactance, 11, 4), 86);
    place(canvas, ljust(k, 16), 118);
    return canvas;
}

// `_nec4_printout._load_row`
std::string load_row(const PrintLoadRow &row) {
    if (row.kind != "FIXED IMPEDANCE") return nec5_load_row(row);
    std::string canvas = sp(118);
    if (row.tag) place(canvas, std::to_string(row.tag), 8);
    if (row.node_from) place(canvas, std::to_string(row.node_from), 13);
    if (row.node_thru) place(canvas, std::to_string(row.node_thru), 18);
    place(canvas, e(row.resistance, 11, 4), 73);
    if (row.has_reactance) place(canvas, e(row.reactance, 11, 4), 86);
    place(canvas, ljust(row.kind, 16), 118);
    return canvas;
}

// `_nec4_printout._port_row`
std::string port_row(const PrintPortRow &r) {
    const double cells[9] = {r.voltage.real(),   r.voltage.imag(),   r.current.real(),
                             r.current.imag(),   r.impedance.real(), r.impedance.imag(),
                             r.admittance.real(), r.admittance.imag(), r.power};
    std::string s = d(r.tag, 6) + d(r.segment, 6);
    for (double v : cells) s += e(v, 12, 5);
    return s;
}

std::string value_row(const ValueRow &r) {
    return d(r.number, 6) + (r.end ? "E" : " ") + d(r.tag, 4) + f(r.centre[0], 9, 4) + f(r.centre[1], 9, 4) +
           f(r.centre[2], 9, 4) + f(r.length, 9, 5) + e(r.real, 13, 4) + e(r.imag, 12, 4) +
           e(r.magnitude, 12, 4) + f(r.phase_deg, 9, 3);
}

// `_printout._network_row`
std::string network_row(const PrintNetworkRow &r) {
    double cells[6];
    std::string tail;
    if (r.line) {
        const double c[6] = {r.z0, r.length_m, r.shunt_a.real(), r.shunt_a.imag(), r.shunt_b.real(), r.shunt_b.imag()};
        std::copy(c, c + 6, cells);
        tail = std::string("   ") + (r.crossed ? "CROSSED" : "STRAIGHT");
    } else {
        const double c[6] = {r.y11.real(), r.y11.imag(), r.y12.real(), r.y12.imag(), r.y22.real(), r.y22.imag()};
        std::copy(c, c + 6, cells);
    }
    std::string s = d(r.tag_from, 9) + d(r.segment_from, 6) + d(r.tag_to, 6) + d(r.segment_to, 6) + " ";
    for (double v : cells) s += e(v, 14, 4);
    return s + tail;
}

// --- sections --------------------------------------------------------------------

const char *const kStructureHeading = "                                 - - - STRUCTURE SPECIFICATION - - -";

Lines structure(const Nec4Printout &data) {
    Lines lines = {kStructureHeading,
                   "",
                   sp(37) + "COORDINATES MUST BE INPUT IN",
                   sp(37) + "METERS OR BE SCALED TO METERS",
                   sp(37) + "BEFORE STRUCTURE INPUT IS ENDED",
                   "",
                   "",
                   "  WIRE" + sp(79) + "NO. OF    FIRST  LAST     TAG",
                   "  NO.        X1         Y1         Z1          X2         Y2         Z2"
                   "      RADIUS   SEG.     SEG.   SEG.     NO."};
    for (const WireTableRow &w : data.wires) add(lines, structure_rows(w));
    lines.push_back("");
    if (data.ge_flag != 0) add(lines, {"   GROUND PLANE SPECIFIED.", ""});
    if (data.ge_flag > 0)
        add(lines, {"   WHERE WIRE ENDS TOUCH GROUND, CURRENT WILL BE INTERPOLATED TO IMAGE IN GROUND PLANE.", "",
                    ""});
    ll total = data.total_segments;
    ll cell = data.symmetry_flag ? data.cell_segments : total;
    lines.push_back("   TOTAL SEGMENTS USED=" + d(total, 5) + "     NO. SEG. IN A SYMMETRIC CELL=" + d(cell, 5) +
                    "     SYMMETRY FLAG=" + d(data.symmetry_flag, 3));
    if (data.symmetry_flag)
        add(lines, {" STRUCTURE HAS " + std::to_string(data.symmetry_planes) + " PLANES OF SYMMETRY", ""});
    add(lines, {"", "", "         - MULTIPLE WIRE JUNCTIONS -", " JUNCTION    SEGMENTS  (- FOR END 1, + FOR END 2)"});
    if (!data.junctions.empty()) {
        for (std::size_t i = 0; i < data.junctions.size(); i++)
            lines.push_back(junction_row(static_cast<ll>(i + 1), data.junctions[i]));
    } else {
        lines.push_back("  NONE");
    }
    return lines;
}

Lines segmentation(const Nec4Printout &data) {
    Lines lines = {sp(33) + "- - - - SEGMENTATION DATA - - - -",
                   "",
                   sp(40) + "COORDINATES IN METERS",
                   "",
                   sp(25) + "I+ AND I- INDICATE THE SEGMENTS BEFORE AND AFTER I",
                   "",
                   "",
                   "  SEG.   COORDINATES OF SEG. CENTER     SEG.     ORIENTATION ANGLES    WIRE"
                   "    CONNECTION DATA   TAG",
                   "  NO.       X         Y         Z       LENGTH     ALPHA     BETA      RADIUS"
                   "    I-   I    I+    NO."};
    for (const SegmentDataRow &r : data.segments) lines.push_back(segment_row(r));
    return lines;
}

Lines frequency(const PrintPreamble &p) {
    return {sp(33) + "- - - - - - FREQUENCY - - - - - -", "",
            sp(36) + "FREQUENCY=" + e(p.frequency_mhz, 11, 4) + " MHZ",
            sp(36) + "WAVELENGTH=" + e(p.wavelength_m, 11, 4) + " METERS"};
}

// `_printout._environment`, then `_nec4_printout._environment`'s file line.
Lines environment(const PrintPreamble &p) {
    const std::string heading = sp(34) + "- - - ANTENNA ENVIRONMENT - - -";
    Lines lines;
    if (!p.has_ground) {
        lines = {heading, "", rstrip(center(p.environment, 98))};
    } else {
        const std::string m = sp(40);
        lines = {heading,
                 "",
                 m + p.environment,
                 m + "RELATIVE DIELECTRIC CONST.=" + f(p.eps_r, 7, 3),
                 m + "CONDUCTIVITY=" + e(p.sigma, 10, 3) + " MHOS/METER",
                 m + "COMPLEX DIELECTRIC CONSTANT=" + e(p.eps_c.real(), 12, 5) + e(p.eps_c.imag(), 12, 5)};
    }
    if (!p.has_ground_file) return lines;
    const std::string name = ljust(p.ground_file, 40);
    const std::string &env = p.environment;
    if (env.size() >= 3 && env.compare(env.size() - 3, 3, "GN3") == 0) {
        add(lines, {"", " GNDINO3: Unable to open file " + name});
        return lines;
    }
    Lines out(lines.begin(), lines.begin() + 2);
    add(out, {"", " GNDINO: UNABLE TO OPEN FILE " + name, ""});
    out.insert(out.end(), lines.begin() + 2, lines.end());
    return out;
}

const Lines &loading_columns() {
    static const Lines cols = {
        "       LOCATION          RESISTANCE   INDUCTANCE  CAPACITANCE       "
        "IMPEDANCE (OHMS)     CONDUCTIVITY    TYPE",
        "    ITAG FROM THRU          OHMS        HENRYS       FARADS        "
        "REAL      IMAGINARY    MHOS/METER"};
    return cols;
}

Lines loading(const PrintPreamble &p) {
    Lines lines = {sp(31) + "- - - STRUCTURE IMPEDANCE LOADING - - -", ""};
    if (p.loads.empty()) {
        lines.push_back(sp(35) + "THIS STRUCTURE IS NOT LOADED");
        return lines;
    }
    lines.push_back("");
    add(lines, loading_columns());
    for (const PrintLoadRow &r : p.loads) add(lines, {"", load_row(r)});
    return lines;
}

Lines timing(const PrintPreamble &p) {
    return {sp(32) + "- - - MATRIX TIMING - - -", "",
            sp(24) + "FILL=" + f(p.fill_seconds, 9, 3) + " SEC.,  FACTOR=" + f(p.factor_seconds, 9, 3) + " SEC."};
}

const Lines &network_columns() {
    static const Lines cols = {
        "      - FROM -    - TO -                          -  -  ADMITTANCE "
        "MATRIX ELEMENTS (MHOS)  -  -",
        "      HTAG  SEG.   TAG  SEG.             (ONE,ONE)                   "
        "(ONE,TWO)                   (TWO,TWO)",
        "      NO.   NO.   NO.   NO.        REAL          IMAG.         REAL   "
        "       IMAG.         REAL          IMAG."};
    return cols;
}
const Lines &line_columns() {
    static const Lines cols = {
        "      - FROM -    - TO -           TRANSMISSION LINE               "
        "-  -  SHUNT ADMITTANCES (MHOS)  -  -              LINE",
        "      TAG  SEG.   TAG  SEG.      IMPEDANCE      LENGTH            "
        "- END ONE -                 - END TWO -            TYPE",
        "      NO.   NO.   NO.   NO.         OHMS        METERS         REAL   "
        "       IMAG.         REAL          IMAG."};
    return cols;
}

Lines network_data(const PrintRun &run) {
    Lines lines = {sp(44) + "- - - NETWORK DATA - - -", ""};
    std::size_t k = 0, index = 0;
    while (k < run.networks.size()) {
        std::size_t j = k;
        while (j < run.networks.size() && run.networks[j].line == run.networks[k].line) j++;
        if (index++) lines.push_back("");
        add(lines, run.networks[k].line ? line_columns() : network_columns());
        for (std::size_t i = k; i < j; i++) lines.push_back(network_row(run.networks[i]));
        k = j;
    }
    return lines;
}

Lines port_table(const std::string &heading, const std::vector<PrintPortRow> &rows) {
    Lines lines = {heading, "",
                   "   TAG   SEG.    VOLTAGE (VOLTS)         CURRENT (AMPS)         "
                   "IMPEDANCE (OHMS)        ADMITTANCE (MHOS)      POWER",
                   "   NO.   NO.    REAL        IMAG.       REAL        IMAG.       REAL"
                   "        IMAG.       REAL        IMAG.     (WATTS)"};
    for (const PrintPortRow &r : rows) lines.push_back(port_row(r));
    return lines;
}

Lines power_budget(const PrintPowerBudget &p) {
    auto watts = [](const char *label, double v) {
        return sp(43) + ljust(label, 14) + "=" + e(v, 11, 4) + " WATTS";
    };
    Lines lines = {sp(40) + "- - - POWER BUDGET - - -", "", watts("INPUT POWER", p.input_power),
                   watts("RADIATED POWER", p.radiated_power), watts("WIRE LOSS", p.wire_loss)};
    if (p.has_network_loss) lines.push_back(watts("NETWORK LOSS", p.network_loss));
    lines.push_back(sp(43) + ljust("EFFICIENCY", 14) + "=" + f(p.efficiency_percent, 7, 2) + " PERCENT");
    return lines;
}

Lines near_field(const PrintNearBlock &b) {
    const std::string kind = b.magnetic ? "MAGNETIC" : "ELECTRIC";
    const std::string c = b.magnetic ? "H" : "E";
    const std::string unit = b.magnetic ? "AMPS/M" : "VOLTS/M";
    std::string units = "      METERS      METERS      METERS";
    for (int width : {16, 13, 13}) units += rjust(unit, static_cast<std::size_t>(width)) + rjust("DEGREES", 10);
    Lines lines = {sp(38) + "- - - NEAR " + kind + " FIELDS - - -", "",
                   "              -  LOCATION  -                     -  " + c +
                       "X  -               -  " + c + "Y  -               -  " + c + "Z  -",
                   "        X           Y           Z           MAGNITUDE   PHASE"
                   "      MAGNITUDE   PHASE      MAGNITUDE   PHASE",
                   units};
    for (const PrintNearRow &r : b.rows) {
        std::string cells;
        const int widths[3] = {15, 14, 14};
        for (int k = 0; k < 3; k++) cells += e(r.magnitudes[k], widths[k], 4) + f(r.phases_deg[k], 9, 2);
        lines.push_back(e(r.point[0], 14, 4) + e(r.point[1], 12, 4) + e(r.point[2], 12, 4) + cells);
    }
    return lines;
}

std::string pattern_row(const PrintPatternRow &r) {
    // "%8.2f%9.2f%11.2f%8.2f%8.2f%11.5f%9.2f  %-6s%15.5E%9.2f%15.5E%9.2f"
    return f(r.theta_deg, 8, 2) + f(r.phi_deg, 9, 2) + f(r.vert_db, 11, 2) + f(r.hor_db, 8, 2) +
           f(r.total_db, 8, 2) + f(r.axial_ratio, 11, 5) + f(r.tilt_deg, 9, 2) + "  " + ljust(r.sense, 6) +
           e(r.e_theta_magnitude, 15, 5) + f(r.e_theta_phase_deg, 9, 2) + e(r.e_phi_magnitude, 15, 5) +
           f(r.e_phi_phase_deg, 9, 2);
}

Lines pattern(const PrintPatternBlock &b) {
    Lines columns = {"  - - ANGLES - -          - POWER GAINS -        - - - POLARIZATION "
                     "- - -    - - - E(THETA) - - -    - - - E(PHI) - - -",
                     "  THETA     PHI        VERT.   HOR.    TOTAL      AXIAL     TILT   "
                     "SENSE     MAGNITUDE    PHASE      MAGNITUDE    PHASE ",
                     " DEGREES  DEGREES       DB      DB      DB        RATIO     DEG.       "
                     "        VOLTS     DEGREES       VOLTS     DEGREES"};
    if (b.directive) {
        const std::string from = "          - POWER GAINS -        ", to = "       - DIRECTIVE GAINS -       ";
        std::size_t at = columns[0].find(from);
        if (at != std::string::npos) columns[0].replace(at, from.size(), to);
    }
    const std::string heading = sp(48) + "- - - RADIATION PATTERNS - - -";
    Lines lines;
    if (!b.has_range) {
        lines = {heading, ""};
        add(lines, columns);
    } else {
        lines = {heading, sp(54) + "RANGE=" + e(b.range_m, 13, 6) + " METERS",
                 sp(54) + "EXP(-JKR)/R=" + e(1.0 / b.range_m, 12, 5) + " AT PHASE" + f(b.range_phase_deg, 7, 2) +
                     " DEGREES",
                 ""};
        if (b.has_lower)
            lines.push_back(sp(39) + "LOWER MEDIUM - EXP(-JKR)/R=" + e_fortran(b.lower_range_mag, 12, 5) +
                            " AT PHASE" + f(b.lower_range_phase_deg, 7, 2) + " DEGREES");
        add(lines, {"", columns[0], columns[1],
                    " DEGREES  DEGREES       DB      DB      DB        RATIO     DEG.       "
                    "       VOLTS/M    DEGREES      VOLTS/M    DEGREES"});
    }
    for (const PrintPatternRow &r : b.rows) lines.push_back(pattern_row(r));
    if (b.has_average)
        add(lines, {"", "",
                    "   AVERAGE POWER GAIN=" + e(b.average_power_gain, 12, 5) + "       SOLID ANGLE USED IN AVERAGING=(" +
                        f(b.solid_angle_pi, 7, 4) + ")*PI STERADIANS.",
                    "", "   POWER RADIATED ASSUMING RADIATION INTO 4*PI STERADIANS =" + e(b.power_radiated_4pi, 12, 5) +
                            " WATTS"});
    return lines;
}

Lines far_field_ground(const PrintFarFieldGround &g) {
    const std::string pad = sp(40);
    return {sp(31) + "- - - FAR FIELD GROUND PARAMETERS - - -",
            "",
            "",
            pad + g.kind,
            pad + "EDGE DISTANCE=" + f(g.edge_distance, 9, 2) + " METERS",
            pad + "HEIGHT=" + f(g.height, 8, 2) + " METERS",
            pad + "SECOND MEDIUM -",
            pad + "RELATIVE DIELECTRIC CONST.=" + f(g.eps_r2, 7, 3),
            pad + "CONDUCTIVITY=" + e(g.sigma2, 10, 3) + " MHOS"};
}

const int kSectionGap = 3, kPatternGap = 5, kStructureGap = 4;

Lines tables(const PrintRun &run) {
    Lines body;
    for (const PrintNearBlock &n : run.near_fields) {
        add(body, near_field(n));
        blanks(body, kSectionGap);
    }
    if (!run.patterns.empty() && run.has_far_field_ground) {
        add(body, far_field_ground(run.far_field_ground));
        blanks(body, kSectionGap);
    }
    for (const PrintPatternBlock &p : run.patterns) {
        add(body, pattern(p));
        blanks(body, kPatternGap);
    }
    return body;
}

Lines run_lines(const PrintRun &run) {
    Lines body;
    if (run.has_preamble) {
        add(body, frequency(run.preamble));
        blanks(body, kSectionGap);
        add(body, environment(run.preamble));
        blanks(body, kSectionGap);
        add(body, loading(run.preamble));
        blanks(body, kSectionGap);
        add(body, timing(run.preamble));
        blanks(body, kSectionGap);
    }
    if (!run.networks.empty()) {
        add(body, network_data(run));
        blanks(body, kSectionGap);
    }
    if (!run.network_excitation.empty()) {
        add(body, port_table(sp(27) + "- - - STRUCTURE EXCITATION DATA AT NETWORK CONNECTION POINTS - - -",
                             run.network_excitation));
        blanks(body, kSectionGap);
    }
    add(body, port_table(sp(42) + "- - - ANTENNA INPUT PARAMETERS - - -", run.sources));
    blanks(body, kSectionGap);
    if (run.has_currents) {
        add(body, {sp(29) + "- - - CURRENTS AND LOCATION - - -", "",
                   sp(21) + "LENGTHS NORMALIZED BY WAVELENGTH (OR 2.*PI/CABS(K))", "",
                   "  SEG.  TAG    COORD. OF SEG. CENTER     SEG.            - - - CURRENT (AMPS) - - -",
                   "  NO.   NO.     X        Y        Z      LENGTH     REAL        IMAG.       MAG.        PHASE"});
        for (const ValueRow &r : run.currents) body.push_back(value_row(r));
        blanks(body, kSectionGap);
    }
    if (run.has_charges) {
        add(body, {sp(34) + "- - - CHARGE DENSITIES - - -", "",
                   sp(23) + "LENGTHS NORMALIZED TO WAVELENGTH (OR 2.*PI/CABS(K))", "", "",
                   "  SEG.  TAG    COORD. OF SEG. CENTER     SEG.          CHARGE DENSITY (COULOMBS/METER)",
                   "  NO.   NO.     X        Y        Z      LENGTH     REAL        IMAG.       MAG.        PHASE"});
        for (const ValueRow &r : run.charges) body.push_back(value_row(r));
        blanks(body, kSectionGap);
    }
    add(body, power_budget(run.power));
    blanks(body, kSectionGap);
    add(body, tables(run));
    return body;
}

}  // namespace

std::string render_nec4_header(const std::string *deck_text, const std::string &basis,
                               const nec5::EngineStamp &stamp) {
    return header(deck_text, basis, stamp);
}

std::string render_nec4_refusal(const std::string *deck_text, const std::string &reason,
                                const std::string &basis, const nec5::EngineStamp &stamp) {
    return header(deck_text, basis, stamp) + "\n" + " ***** NEC ERROR - " + reason + "\n";
}

std::string render_nec4_printout(const std::string &deck_text, const Nec4Printout &data, const std::string &basis,
                                 const nec5::EngineStamp &stamp, bool closing) {
    std::vector<nec5::Card> cards = post_ge_cards(deck_text);
    bool has_terminator = !cards.empty() && (cards.back().mnemonic == "EN" || cards.back().mnemonic == "NX");
    std::vector<std::pair<ll, const nec5::Card *>> numbered;
    for (std::size_t i = 0; i < cards.size(); i++) {
        if (has_terminator && i + 1 == cards.size()) continue;
        numbered.push_back({static_cast<ll>(i + 1), &cards[i]});
    }
    Lines body;
    add(body, structure(data));
    blanks(body, kStructureGap);
    add(body, segmentation(data));
    blanks(body, kStructureGap);
    // `numbered[echoed : run.echo_through]`, then `echoed = run.echo_through`,
    // with Python's slice clamping (the run's number is stored as given).
    ll echoed = 0;
    auto slice_end = [&](ll v) {  // a slice bound, as Python reads one
        const ll n = static_cast<ll>(numbered.size());
        if (v < 0) v += n;
        return static_cast<std::size_t>(std::max<ll>(0, std::min<ll>(v, n)));
    };
    for (std::size_t index = 0; index < data.runs.size(); index++) {
        const PrintRun &run = data.runs[index];
        if (index) body.push_back("");
        if (run.has_echo_through) {
            for (std::size_t k = slice_end(echoed); k < slice_end(run.echo_through); k++)
                add(body, echo(numbered[k].first, *numbered[k].second));
            echoed = run.echo_through;
            bool opens = index == 0 || (!run.report && run.has_preamble);
            blanks(body, opens ? kStructureGap : kSectionGap);
        }
        add(body, run.report ? tables(run) : run_lines(run));
    }
    body.push_back("");
    for (std::size_t k = slice_end(echoed); k < numbered.size(); k++)
        add(body, echo(numbered[k].first, *numbered[k].second));
    if (has_terminator) body.push_back(card_echo(static_cast<ll>(cards.size()), cards.back()));
    if (closing) add(body, {"", " RUN TIME =" + f(data.run_seconds, 10, 3)});
    return header(&deck_text, basis, stamp) + join(body) + "\n";
}

bool refusal_printout(const std::string &text, const std::string &basis, const nec5::EngineStamp &stamp,
                      std::string &out) {
    Build b;
    try {
        b = build(text, basis);
    } catch (...) {
        return false;
    }
    if (b.structures.empty()) return false;
    const StructureBuild &first = b.structures[0];
    std::string message;
    if (first.kind == StructureBuild::refused) {
        message = first.message;
    } else if (first.kind == StructureBuild::seam) {
        Seam s;
        try {
            s = seam(text, basis);
        } catch (...) {
            return false;
        }
        if (s.structures.empty() || s.structures[0].kind != StructureSeam::refused) return false;
        message = s.structures[0].message;
    } else {
        return false;
    }
    std::string echo_text = text;
    std::vector<nec5::Structure> all;
    if (nec5::split_structures(text, all)) echo_text = all[0].echo_text();
    out = render_nec4_refusal(&echo_text, message, basis, stamp);
    return true;
}

}  // namespace nec4
}  // namespace mw
