// The NEC-4.2 slot's BUILD in C++ (momwire#1607 A2b), with no Python in it.
// It takes a served structure from the seam front (`_nec4_seam_core.h`) and
// produces what a solver is constructed from, at every operating point the
// structure's runs ask for. It is the copy of `portal._portal.DeckSolver`'s
// construction and `at`:
// - `deck._solver.prepare_mesh`: the union port set (`_sites`) and
//   `deck._polylines.to_polylines`, which splits wires at the ground plane,
//   cuts port spans, builds the node graph, chains polylines and cycles, and
//   places the ports;
// - `build_solver` / `construct_solver`: the solver's constructor arguments
//   in their call order, the wire-loading and ground kwargs, the extended
//   kernel as the dialect's default with its fallback or partial advisory in
//   the basis's declared words, and the port plan;
// - DeckSolver's NEC-facing tables, and each fill's deck-sign congruence
//   (`_port_signs`).
//
// A `SolverArgs` is the segment-centre engine interface: what a sinusoidal
// or a B-spline engine (momwire#1607 B, C) builds its fill from. Python
// stays the reference: `scripts/core_split/nec4_build_compare.py`, held
// equal by `tests/test_nec4_build_core.py`.
#pragma once

#include "_nec4_seam_core.h"

#include <array>
#include <complex>
#include <string>
#include <utility>
#include <vector>

namespace mw {
namespace nec4 {

typedef std::array<double, 3> Point3;

struct JunctionEnd {  // (polyline, "start" | "end")
    long long polyline = 0;
    bool start = true;
};

struct SolverFeed {  // (polyline, arclength, volts)
    long long polyline = 0;
    double arclength = 0;
    std::complex<double> volts;
};

// The keyword arguments `construct_solver` hands the solver class, in order.
// The Sinusoidal-Galerkin row's declared refusal of the extended kernel at a
// radius-step junction (`_EK_STEPPED_RADIUS_JUNCTION_REFUSAL`): the dialect
// default falls back on it (momwire#1718 S3), and the constructor raises it
// when the kernel is asked for.
extern const char *const kSgEkSteppedJunction;

struct SolverArgs {
    std::vector<std::vector<Point3>> wires;              // wires=
    std::vector<std::vector<long long>> n_per_edge;      // n_per_edge_per_wire=
    std::vector<SolverFeed> feeds;                       // feeds=
    double wavelength = 0;                               // wavelength=
    std::vector<double> radii;                           // wire_radius= (scalar
    bool uniform_radius = true;                          //   when uniform)
    std::vector<std::vector<JunctionEnd>> junctions;     // junctions= (None if empty)
    bool extended_kernel = false;                        // extended_kernel=True, or absent
    bool has_conductivity = false;                       // wire_conductivity=
    std::vector<double> conductivity;                    //   NaN: not this wire
    bool has_insulation = false;                         // insulation_radius= / _eps_r=
    std::vector<double> insulation_radius, insulation_eps_r;
    bool has_distributed = false;                        // distributed_rlc=
    std::vector<bool> distributed_present;
    std::vector<DistributedRLC> distributed;
    Ground ground;                                       // ground_z= / ground_eps= / ground_model=
    bool exact_kernel_auto = false;                      // bspline's exact_kernel='auto'
    bool junction_rule_nec42 = false;                   // junction_rule='nec42' (momwire#1644)
};

// `_wire_spec.WireDedup` (momwire#1042): how a solver's INPUT wires map onto
// the wires it solves once a wire listed twice is dropped. `kept[i]` is the
// input index of solved wire `i`; `removed` one (copy, kept twin, reversed)
// per dropped copy, sorted by copy. Inactive: nothing was dropped.
struct WireDedup {
    bool active = false;
    long long n_input = 0;
    std::vector<long long> kept;
    struct Dropped {
        long long copy = 0, twin = 0;
        bool reversed = false;
    };
    std::vector<Dropped> removed;
};

struct PortSite {
    long long wire = 0;
    double arclength = 0;
    long long feed = -1, load = -1;  // -1: None
    bool has_load_spec = false;
    LoadSpec load_spec;
    bool network = false;
};

struct PortPlan {
    std::vector<PortSite> sites;
    std::vector<long long> feed_ports, load_ports;
    std::vector<bool> voltages_present;  // per model group
    std::vector<std::vector<std::complex<double>>> voltages;
    std::vector<std::pair<long long, long long>> network_ports;
};

struct OperatingPoint {
    double frequency_mhz = 0;
    SolverArgs args;
    PortPlan ports;
    double wavelength = 0;
    long long group = -1;  // -1: None
    bool extended_kernel = false;  // what the solver was built with
    // what the run asked for (`solver.at(freq, extended_kernel=...)`'s key):
    // the dialect default's fallback can build without it (momwire#1718 S3)
    bool extended_kernel_asked = false;
    std::vector<std::string> advisories;
    std::vector<double> signs;
};

struct SegmentRow {  // portal._Segment
    long long number = 0;
    Int tag;
    long long wire = 0, local = 0;
    Point3 centre{}, direction{};
    double radius = 0;
};

struct BuildTables {  // DeckSolver's
    std::vector<std::pair<Int, Int>> ports;  // the union (tag, seg)
    std::vector<SegmentRow> segments;
    long long group = -1;
    double smallest_radius = 0;
    std::vector<long long> feed_index;
    std::vector<std::pair<long long, LoadSpec>> load_ports;
    std::vector<std::pair<long long, LoadSpec>> model_loads;
    std::vector<std::pair<long long, std::vector<long long>>> group_loads;
    std::vector<std::pair<long long, long long>> port_by_segment;
    bool lossy = false;
    std::vector<long long> current_ports;
};

struct StructureBuild {
    enum Kind { seam, built, refused, raised, declined } kind = seam;
    nec5::ErrorKind error_kind = nec5::ErrorKind::value_error;
    std::string message;  // UTF-8
    std::vector<OperatingPoint> points;  // the fills, in the order asked
    BuildTables tables;                  // built
};

struct Build {
    std::string basis;
    bool seam_failed = false;  // the text did not split (never, today)
    std::vector<StructureBuild> structures;  // through the first not built
    std::size_t n_structures = 0;
};

// The seam front, then each served structure's build, for one basis.
Build build(const std::string &latin1_text, const std::string &basis);
std::string dump_build(const std::string &latin1_text, const std::string &basis);

}  // namespace nec4
}  // namespace mw
