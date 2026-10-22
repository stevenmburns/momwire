// The NEC-4.2 printout WRITER in C++ (momwire#1607 A3), with no Python in
// it. It is the copy of `eznec/_nec4_printout.py`'s `render_nec4_printout`
// and the `eznec/_printout.py` sections it shares with the NEC-5 printout
// (header, card echo, FREQUENCY, ANTENNA ENVIRONMENT, loading rows, MATRIX
// TIMING, NETWORK DATA, POWER BUDGET, near fields, RADIATION PATTERNS), plus
// the slot's refusal printout (`_printout.render_refusal(dialect="nec4")`).
//
// A FORMATTER, on the Python module's own rule: it is handed every number in
// a `Nec4Printout` and lays them out; it computes no physics. Filling a
// `Nec4Printout` from a solve is the run half (the sin and bs2 engines'
// units). Python stays the reference: `tests/test_nec4_printout_core.py`
// hands this writer the very `Nec4Printout` Python's own solve built and
// holds the text equal to Python's, byte for byte, and holds the refusal
// printouts the C++ seam and build produce equal to `_nec4.render`'s.
//
// The NEC-5 printout core's writer (`_nec5_printout_core.cpp`) has the same
// sections, typed against NEC-5's run records. This copy is separate so the
// NEC-5 executables compile nothing new; folding the two writers together is
// a later, gated move.
//
// Text out is UTF-8, LF line ends (`nec5::printout_bytes` makes the file).
#pragma once

#include "_nec4_build_core.h"
#include "_nec5_printout_core.h"

#include <complex>
#include <string>
#include <vector>

namespace mw {
namespace nec4 {

typedef std::complex<double> cplx;

struct WireTableRow {  // WireRow | ArcRow | HelixRow | ScaleRow | MoveRow
    enum Kind { wire, arc, helix, scale, move } kind = wire;
    // wire
    long long number = 0, segments = 0, first = 0, last = 0, tag = 0;
    double end1[3] = {0, 0, 0}, end2[3] = {0, 0, 0};
    double radius = 0;
    // arc (GA, momwire#1592): number, radius and the segment columns above too
    double arc_radius = 0, angle1_deg = 0, angle2_deg = 0;
    // helix (GH): number and the segment columns above too
    double turns = 0, length = 0, wire_length = 0;
    double helix_radii[2] = {0, 0}, wire_radii[2] = {0, 0};
    // scale
    double factor = 0;
    // move
    long long tag_increment = 0, copies = 0, first_tag = 0, move_segments = 0;
    double rotation_deg[3] = {0, 0, 0}, translation[3] = {0, 0, 0};
};

struct SegmentDataRow {  // SegmentRow
    long long number = 0, i_minus = 0, i_plus = 0, tag = 0;
    double centre[3] = {0, 0, 0};
    double length = 0, alpha_deg = 0, beta_deg = 0, radius = 0;
};

struct ValueRow {  // SegmentValueRow
    long long number = 0, tag = 0;
    double centre[3] = {0, 0, 0};
    double length = 0, real = 0, imag = 0, magnitude = 0, phase_deg = 0;
    bool end = false;
};

struct PrintLoadRow {  // _printout.LoadRow
    long long tag = 0, node_from = 0, node_thru = 0;
    bool has_resistance = false, has_reactance = false, has_inductance = false,
         has_capacitance = false, has_conductivity = false;
    double resistance = 0, reactance = 0, inductance = 0, capacitance = 0, conductivity = 0;
    std::string kind = "FIXED IMPEDANCE";
};

struct PrintPortRow {  // _printout.PortRow
    long long tag = 0, segment = 0, end_index = 0;
    cplx voltage, current, impedance, admittance;
    double power = 0;
};

struct PrintNetworkRow {  // NetworkRow | LineRow
    bool line = false;
    long long tag_from = 0, segment_from = 0, tag_to = 0, segment_to = 0;
    cplx y11, y12, y22;                 // NetworkRow
    double z0 = 0, length_m = 0;        // LineRow
    cplx shunt_a, shunt_b;
    bool crossed = false;
};

struct PrintPowerBudget {
    double input_power = 0, radiated_power = 0, wire_loss = 0, efficiency_percent = 0;
    bool has_network_loss = false;
    double network_loss = 0;
};

struct PrintNearRow {
    double point[3] = {0, 0, 0}, magnitudes[3] = {0, 0, 0}, phases_deg[3] = {0, 0, 0};
};
struct PrintNearBlock {
    std::vector<PrintNearRow> rows;
    bool magnetic = false;
};

struct PrintPatternRow {
    double theta_deg = 0, phi_deg = 0, vert_db = 0, hor_db = 0, total_db = 0, axial_ratio = 0,
           tilt_deg = 0;
    std::string sense;
    double e_theta_magnitude = 0, e_theta_phase_deg = 0, e_phi_magnitude = 0, e_phi_phase_deg = 0;
};
struct PrintPatternBlock {
    std::vector<PrintPatternRow> rows;
    bool has_average = false;  // average_power_gain, solid_angle_pi, power_radiated_4pi
    double average_power_gain = 0, solid_angle_pi = 0, power_radiated_4pi = 0;
    bool has_range = false;
    double range_m = 0, range_phase_deg = 0;
    bool has_lower = false;
    double lower_range_mag = 0, lower_range_phase_deg = 0;
    bool directive = false;
};

struct PrintFarFieldGround {
    std::string kind;
    double edge_distance = 0, height = 0, eps_r2 = 0, sigma2 = 0;
};

struct PrintPreamble {
    double frequency_mhz = 0, wavelength_m = 0;
    std::string environment;  // UTF-8
    bool has_ground = false;  // GroundMedium
    double eps_r = 0, sigma = 0;
    cplx eps_c;
    std::vector<PrintLoadRow> loads;
    double fill_seconds = 0, factor_seconds = 0;
    bool has_ground_file = false;
    std::string ground_file;  // UTF-8
};

struct PrintRun {  // Nec4Run | Nec4Report
    bool report = false;
    bool has_echo_through = false;
    long long echo_through = 0;
    // Nec4Run only
    bool has_preamble = false;
    PrintPreamble preamble;
    std::vector<PrintNetworkRow> networks;
    std::vector<PrintPortRow> network_excitation, sources;
    bool has_currents = false, has_charges = false;
    std::vector<ValueRow> currents, charges;
    PrintPowerBudget power;
    // both
    std::vector<PrintNearBlock> near_fields;
    bool has_far_field_ground = false;
    PrintFarFieldGround far_field_ground;
    std::vector<PrintPatternBlock> patterns;
};

struct Nec4Printout {
    std::vector<WireTableRow> wires;
    long long ge_flag = 0, total_segments = 0;
    std::vector<std::vector<long long>> junctions;
    std::vector<SegmentDataRow> segments;
    std::vector<PrintRun> runs;
    double run_seconds = 0;
};

// `_printout.render_header(deck_text, basis=..., dialect="nec4")`
std::string render_nec4_header(const std::string *latin1_deck_text, const std::string &basis,
                               const nec5::EngineStamp &stamp);
// `_printout.render_refusal(..., dialect="nec4")`
std::string render_nec4_refusal(const std::string *latin1_deck_text, const std::string &utf8_reason,
                                const std::string &basis, const nec5::EngineStamp &stamp);
// `_nec4_printout.render_nec4_printout(deck_text, data, basis=, closing=)`
std::string render_nec4_printout(const std::string &latin1_deck_text, const Nec4Printout &data,
                                 const std::string &basis, const nec5::EngineStamp &stamp,
                                 bool closing = true);

// The slot's whole printout when the deck refuses before any solve, i.e. its
// first structure refuses at the seam or the build, as `_nec4.render`
// writes it; false when it needs a solve, raises, or the copy declines.
bool refusal_printout(const std::string &latin1_text, const std::string &basis,
                      const nec5::EngineStamp &stamp, std::string &utf8_printout);

}  // namespace nec4
}  // namespace mw
