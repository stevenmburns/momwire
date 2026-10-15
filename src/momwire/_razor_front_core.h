// razor-2p's FRONT HALF in C++: solver construction (the validations and
// normalisations `RazorSolver.__init__` performs), the geometry
// (`_build_geometry`, junction detection, the wings), the k-independent
// preparation `SolveContext::fill` reads (`_assemble_Z_prepare` on its
// centroid-sharing branch), the bundle refusal and the port columns
// (`_refuse_coincident_segments`, `_port_columns` and the feed snap). No
// Python anywhere, so a C++ caller goes geometry -> currents:
//
//     mw::razor::FrontInputs in;      // the arguments the NEC-5 seam hands
//     ...                             // RazorSolver (wires, counts, radii,
//                                     // feeds, node gaps, ground, lambda, EK)
//     mw::razor::Front front;
//     std::string why = front.build(in);   // "" = covered, else the
//                                          // reason it is not; throws
//                                          // FrontError where Python
//                                          // raises
//     mw::razor::SolveContext ctx;
//     std::vector<cd> X(front.n_basis * front.n_ports);
//     int info = mw::razor::solve_ports(front, ctx, lapack, X.data(), 0);
//
// It is the SAME arithmetic as the Python front half, gated byte for byte
// against it (`scripts/core_split/cxx_solve_corpus.py`,
// `tests/test_razor_cxx_front.py`). No library's rounding is involved: the
// Python reference spells a 3-vector's length in one fixed order on every CPU
// (`_vec3.norm3`, momwire#1484), and its Gauss-Legendre rules are a frozen
// table (`_leggauss_table.py`), which this front half reads through a header
// generated from it (`_leggauss_table.h`, scripts/gen_leggauss_header.py).
//
// SCOPE: the class `_razor_solve_core.h` serves -- free space or a PERFECT
// ground (`ground_z` with no `ground_eps`, which is also what the NEC-5 seam
// builds for a GD card), a real k, razor-2p's two-point path rule
// (`nec5_quadrature=True`), with or without the extended kernel, no loading,
// no exact ring kernel. Anything else `build` declines BY NAME (returns the
// reason) and the caller keeps the Python front half: a wire listed twice
// (momwire#1042's merge), declared `junctions=`, a wire below the plane
// (the shared buried sentences), the exact ring kernel ("auto" engaging, or
// True). Inside the scope every refusal the Python front half raises is
// raised here as a `FrontError` of the same kind with the same message, in
// the same order.
//
// NOT here: the advisories (`warnings.warn` text: the far-mesh class, short
// segments, the gap mesh floor). An ambiguous feed snap (momwire#1262) is
// REPORTED (`Front::ties`), and the caller decides whether to warn -- the
// Python glue asks `_feed_snap.mirror_images`, which this does not port.
#pragma once

#include "_mw_core.h"
#include "_razor_solve_core.h"
#include "_razor_wire_loading_core.h"

#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace mw {
namespace razor {

// The two exception types the Python front half raises. `solve_phase`:
// raised where Python raises it from `compute_port_solution` (the bundle
// refusal, the port columns, the tables) rather than from the constructor,
// which is the difference between a refusal and an internal error on the
// NEC-5 seam (it turns only the CONSTRUCTOR's NotImplementedError into a
// refusal).
enum class FrontErrorKind { value_error, not_implemented };

struct FrontError : std::runtime_error {
    FrontErrorKind kind;
    bool solve_phase = false;
    FrontError(FrontErrorKind k, const std::string &m, bool solve = false)
        : std::runtime_error(m), kind(k), solve_phase(solve) {}
};

enum class End : int8_t { start = 0, end = 1 };
enum class ExactKernel : int8_t { off = 0, on = 1, automatic = 2 };

// One wire's `n_per_edge_per_wire` entry: None (-> `nsegs` on every edge), a
// scalar (that count on every edge) or a per-edge list.
struct EdgeCounts {
    enum Kind : int8_t { none = 0, scalar = 1, list = 2 } kind = none;
    std::vector<int64_t> v;
};

struct FeedSpec {
    int64_t wire = 0;
    bool midpoint = false;  // arclength None: the wire's midpoint
    double arc = 0.0;
    cd volts = cd(1.0, 0.0);
};

struct NodeGapSpec {
    int64_t wire = 0;
    End end = End::start;
    cd volts = cd(0.0, 0.0);
};

// The constructor arguments the NEC-5 seam passes `RazorSolver`
// (`deck._solver.construct_solver`), plus the ones a direct caller may
// default. Wire-indexed values are in the CALLER's wire numbering.
struct FrontInputs {
    // Every wire's anchors, (n_points, 3) row-major, wire after wire, and
    // how many anchors each wire has.
    std::vector<double> points;
    std::vector<int64_t> wire_points;
    // `n_per_edge_per_wire`: absent (every wire `nsegs` per edge) or one
    // entry per wire. A count in the wrong place is refused by name.
    bool counts_given = false;
    std::vector<EdgeCounts> counts;
    int64_t nsegs = 101;
    // `wire_radius` as `np.asarray` reads it: its values and its shape
    // (empty shape = a scalar).
    std::vector<double> radius;
    std::vector<int64_t> radius_shape;
    // `feeds` (given) or the `feed_wire_index` / `feed_arclength` shorthand.
    bool feeds_given = true;
    std::vector<FeedSpec> feeds;
    int64_t feed_wire_index = 0;
    bool feed_midpoint = true;
    double feed_arclength = 0.0;
    std::vector<NodeGapSpec> node_gaps;
    // `wire_conductivity` / `distributed_rlc` (configure_loading's reading);
    // the NEC-5 seam's lumped loads are ports, never here.
    WireLoading loading;
    bool has_ground = false;
    double ground_z = 0.0;
    double wavelength = 22.0;
    bool extended_kernel = false;
    ExactKernel exact_kernel = ExactKernel::off;
    // The three quadrature orders the constructor validates (`n_qp_path`
    // None = derived; razor-2p's path rule never reads it).
    bool n_qp_path_given = false;
    int64_t n_qp_path = 0;
    int64_t n_qp_source = 12;
    int64_t n_qp_sommerfeld = 3;
    // The fill's working-array budget, in complex elements (`_CHUNK_ELEMS`).
    int64_t chunk_elems = 2000000;
};

struct Junction {
    std::vector<std::pair<int64_t, End>> ends;
    std::vector<int64_t> bases;
    bool grounded = false;
};

// A feed that fell exactly between two knots (`_feed_snap.snap`'s tie): the
// smaller arclength was taken. The caller decides whether the deck's own
// symmetry makes the choice harmless (`_feed_snap.mirror_images`).
struct Tie {
    int64_t feed = 0, wire = 0;
    double target = 0.0, lo = 0.0, hi = 0.0, picked = 0.0;
};

// A bool array a `view<const bool, 1>` can read (std::vector<bool> is a
// bitset).
struct BoolArray {
    std::unique_ptr<bool[]> p;
    size_t n = 0;
    void assign(size_t size, bool value) {
        p.reset(new bool[size > 0 ? size : 1]);
        n = size;
        for (size_t i = 0; i < size; i++) p[i] = value;
    }
    bool *data() const { return p.get(); }
};

class Front {
  public:
    // Build everything. Returns "" when this solve is covered, else the name
    // of what the front half does not cover (the caller keeps the Python
    // front half); throws FrontError where the Python front half raises.
    std::string build(const FrontInputs &in);

    // --- construction, normalised (the solver's own numbering) -----------
    int64_t n_wires = 0;
    std::vector<int64_t> pt_off;           // (n_wires + 1) anchor offsets
    std::vector<double> points;            // (n_points, 3)
    std::vector<std::vector<int64_t>> npe; // per wire, per edge
    std::vector<double> radius_per_wire;
    bool uniform_radius = false;
    double radius_value = 0.0;             // the uniform radius
    std::vector<FeedSpec> feeds;
    std::vector<NodeGapSpec> node_gaps;
    bool has_ground = false;
    double ground_z = 0.0;
    bool extended_kernel = false;
    double c = 0.0, k = 0.0, omega = 0.0, c_a_imag = 0.0, c_phi_imag = 0.0;

    // --- geometry (`_build_geometry`'s dict) -----------------------------
    int64_t n_seg = 0, n_basis = 0, n_interior = 0;
    std::vector<int64_t> seg_offsets, basis_offsets;  // (n_wires + 1)
    std::vector<double> seg_p0, seg_t, seg_h;         // (n_seg, 3) / (n_seg)
    std::vector<double> arc_at_knot;  // wire w: [seg_offsets[w] + w, seg_offsets[w+1] + w + 1)
    std::vector<int64_t> wing_seg;    // (n_basis, 2)
    BoolArray wing_rise;              // (n_basis, 2)
    std::vector<double> wing_sigma;   // (n_basis, 2)
    std::vector<Junction> junctions;
    std::vector<int64_t> grounded_bases;

    // --- the prepared tables (`_assemble_Z_prepare`) ---------------------
    std::vector<int64_t> s_a, s_b;
    std::vector<double> h_a, h_b, q_a, q_b;
    std::vector<double> cent;          // (n_seg, 3): the centroid observers
    std::vector<double> tans, wts;     // (2 n_basis, 3), (2 n_basis)
    std::vector<double> td_a, td_b;    // (3, n_basis)
    BoolArray fall_a, fall_b;          // (n_basis)
    std::vector<double> a_col;         // (n_seg): the kernel's radius column
    std::vector<int64_t> group;        // EK labels (n_seg) or empty
    std::vector<double> a_ek;          // (n_seg) or empty
    std::vector<double> b_obs;         // (n_seg) or empty
    std::vector<double> xg, wg;
    bool has_image = false;
    std::vector<double> img_seg_p0, img_seg_t, img_td_a, img_td_b;
    int64_t window_rows = 1;

    // --- ports (`_port_columns`) ----------------------------------------
    int64_t n_ports = 0;
    std::vector<double> cols;  // (n_basis, n_ports), row-major as numpy's
    std::vector<int64_t> feed_basis;
    std::vector<Tie> ties;

    // --- the wire loading (`_loading_stencil` x `loading_for`'s z_seg at
    // the fill's omega): Z(row, col) += val, in this order, after the fill
    // and its image fold and before the LU (`_load_in_place`).
    WireLoading loading;
    std::vector<int64_t> load_rows, load_cols;
    std::vector<cd> load_vals;

    // The fill's arguments, as views into this object (valid while it is).
    SolveInputs solve_inputs(uintptr_t cancel_flag) const;

  private:
    double norm3(const double *v) const;
    void construct(const FrontInputs &in);
    std::string scope(const FrontInputs &in);
    void ground_ends(std::vector<std::pair<int64_t, End>> &touching);
    void detect_junctions(const std::vector<std::pair<int64_t, End>> &grounded_ends,
                          std::vector<Junction> &groups);
    void build_geometry();
    void refuse_coincident_segments() const;
    void prepare();
    void port_columns();
    void loading_stencil();
    std::vector<Junction> groups_;  // `_find_junctions`
};

// fill + LU for every port of a built Front: X (n_basis, n_ports),
// column-major, is written with the solution. Returns 0, or i > 0 when U(i,
// i) is exactly zero (the caller's singular-matrix refusal; X is then the
// right-hand side, unsolved).
int solve_ports(const Front &front, SolveContext &ctx, const Lapack &lp, cd *x,
                uintptr_t cancel_flag);

// `_wire_spec.format_numbers` / `format_shape`: the refusals' portable
// spellings of numbers (`%.12g`, comma-separated, in brackets) and shapes.
std::string format_numbers(const std::vector<double> &values);
std::string format_shape(const std::vector<int64_t> &shape);

// `_quadrature.leggauss(n)` for a tabled order (1..192), from the frozen
// table; false beyond it.
bool leggauss(int64_t n, std::vector<double> &x, std::vector<double> &w);
// RazorSolver.eps / RazorSolver.mu, and the Python front half's tolerances:
// `_junction_rule.JUNCTION_TOL`, `_feed_snap.AMBIGUITY_TOL_FRAC`,
// `_exact_kernel_seam.EXACT_KERNEL_AUTO_H_OVER_A`. A test pins each to its
// Python twin.
constexpr double EPS = 8.8541878188e-12;
constexpr double MU = 1.25663706127e-6;
constexpr double JUNCTION_TOL = 1e-9;
constexpr double AMBIGUITY_TOL_FRAC = 1e-11;
constexpr double EXACT_KERNEL_AUTO_H_OVER_A = 3.0;

}  // namespace razor
}  // namespace mw
