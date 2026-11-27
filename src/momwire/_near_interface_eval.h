// The designed near-interface tables as the crossing fill asks for them
// (`_near_interface.designed_tables` / `radius_tables` / `designed_rows`),
// with no Python (momwire#1512 B4).
//
// What decides the bits is not only the kernel (`_near_interface_core.h`,
// the column twin) but WHICH rows are evaluated fresh and how they group:
// a column's rule is converged for its smallest s = z - z', so a member's
// value depends on its column's membership. This is the Python route's
// contract, step for step:
//
//   * the asked rows' distinct triples in first-appearance order (-0.0 and
//     0.0 one key, the first appearance's floats kept);
//   * the fill's memo (`TripleMemo`, keys `row + 0.0`): hits return the
//     stored floats, the fresh rows are evaluated and inserted in order;
//   * fresh rows the fill's sheet plan names are interpolated from their
//     plane's / height's sheet (`PlaneSheet`, a function of the row alone);
//     the rest go to the twin grouped by (label, exact rho) in first-seen
//     order, members in first-seen order (`_column_blocks`);
//   * the sheet plan (`sheet_plan`, `_census`, `_plane_pays`): decided ONCE
//     per fill from the fill's node pairs.
#pragma once

#include "_near_interface_core.h"

#include <array>
#include <complex>
#include <cstdint>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

namespace ni_eval {

using cd = std::complex<double>;
constexpr int N_KEYS = 6;  // U, V, W, dzW, dzpV, dzpW
using Six = std::array<cd, N_KEYS>;

// A cone sheet's plan entry (`_cone_census`, momwire#1595 option A): the
// surface family z' = zf + beta * u, its fan [blo, bhi], its corner size d
// the observer heights it is tabulated at (sorted, distinct) and its fan
// correction's nodes a cell.
struct Cone {
    double zf, blo, bhi, d;
    std::vector<double> zs;
    int p_fan = 0;  // the fan correction's nodes a cell (0: a fan of one slope)
};

struct SheetPlan {
    std::vector<double> planes;           // sorted z' < 0
    std::vector<double> heights, depths;  // sorted z > 0, each one's reach
    std::vector<Cone> cones;              // in the census's order
    double fold = 0.0;                    // the cones' rows' fold radius
    // momwire#1860 L2 (`height_sheet_recal`): each height's snapped window
    // [lo, hi]; empty, the shipped plan, is `z == h`.
    std::vector<double> lo, hi;
    bool any() const { return !planes.empty() || !heights.empty() || !cones.empty(); }
};

// `sheet_plan(eps_t, k2, specs, wire_radius)`: specs are (above nodes, below
// nodes) pairs, (n, 3) row-major, z relative to the interface.
struct NodePair {
    const double *above;
    size_t n_above;
    const double *below;
    size_t n_below;
};
SheetPlan sheet_plan(double k_p, cd k_m, const std::vector<NodePair> &specs, double wire_radius);

// An open-addressing map from an exact triple's bits (each `x + 0.0`, so
// -0.0 and 0.0 are one key) to a dense index: what `_unique_rows` and the
// memo's dict decide, without a node allocation per key. Lookups are const
// and may run from several threads at once; inserts are single-threaded.
class TripleIndex {
  public:
    static constexpr uint32_t NONE = 0xFFFFFFFFu;
    TripleIndex() { rehash(1024); }
    // The index stored for `row`, or NONE.
    uint32_t find(const double *row) const;
    // The index stored for `row`; when absent, `v` is stored and returned
    // with `inserted` set.
    uint32_t find_or_insert(const double *row, uint32_t v, bool &inserted);
    size_t size() const { return n_; }
    void reserve(size_t n) {
        if (2 * n > slots_.size()) rehash(2 * n);
    }

  private:
    // A slot is 8 bytes (the hash's high half as a tag, the value); the key
    // bits live in keys_[value], read only on a tag match, so a probe walks
    // a table a quarter the size of one holding the keys.
    struct Slot {
        uint32_t tag, v;
    };
    struct Key {
        uint64_t a, b, c;
    };
    static uint64_t mix(uint64_t a, uint64_t b, uint64_t c);
    void rehash(size_t want);
    std::vector<Slot> slots_;
    std::vector<Key> keys_;
    size_t mask_ = 0, n_ = 0;
};

// One fill's memo (`ProductMemo` / `TripleMemo`): exact-triple keyed, with
// the fill's sheet plan and column cache.
class Memo {
  public:
    Memo();
    bool lookup(const double *row, Six &out) const;
    void insert(const double *row, const Six &v);
    // The stored value's index for `row` (NONE: absent, or a NaN row, which
    // never hits), and the value at an index.
    uint32_t find(const double *row) const;
    const Six &at(uint32_t i) const { return vals_[i]; }
    // `designed_index`'s fresh rows: the keys claimed (index = the value's
    // future position) before the values are appended in that order.
    TripleIndex &keys() { return idx_; }
    void append_values(const Six *v, size_t m) { vals_.insert(vals_.end(), v, v + m); }
    // Storage for the next m values, appended zeroed for the caller to write
    // in place: `append_values` without the caller's own copy of them
    // (momwire#1541). `room` more values are reserved beyond them, so that
    // many later appends do not reallocate (and copy) the whole memo.
    Six *grow_values(size_t m, size_t room = 0) {
        const size_t old = vals_.size();
        if (vals_.capacity() < old + m + room) vals_.reserve(old + m + room);
        vals_.resize(old + m);
        return vals_.data() + old;
    }
    SheetPlan plan;
    bool have_plan = false;
    std::unique_ptr<ni_core::ColumnCache> cache;
    size_t size() const { return vals_.size(); }

  private:
    TripleIndex idx_;
    std::vector<Six> vals_;
};

// The evaluator: wavenumbers and the rule constants.
struct Evaluator {
    double k_p = 0;
    cd k_m;
    int n_threads = 0;  // `_table_threads()`; values do not depend on it
    uintptr_t cancel_flag = 0;
};

// `designed_tables` over n asked points (rho already folded, z, z'), with
// optional per-point labels (`group_labels`): out is (n, 6).
void designed_tables(const Evaluator &ev, Memo *memo, const double *rho, const double *z,
                     const double *zp, size_t n, const int64_t *labels, Six *out);

// The same evaluation through a memo, each point answered as an index into
// it (`memo.at(idx[i])`): the values `designed_tables` copies out, without the
// copy. `n_threads` splits only the read-only memo lookups.
void designed_index(const Evaluator &ev, Memo &memo, const double *rho, const double *z,
                    const double *zp, size_t n, const int64_t *labels, uint32_t *idx,
                    int n_threads = 1);

// `_point_radius_tables_folded` (momwire#1653): the point-observer family
// (POINT_KEYS) at m rows (m, 3) of (rho already folded, z, z'), out (m, 4):
// the rows on the plan's planes and heights from the point sheets, the rest
// exact through the point twin, grouped by rho. Memo-free, as the Python
// route is.
void point_tables(const Evaluator &ev, const SheetPlan *plan, const double *rows, size_t m, cd *out);

// `_evaluate_fresh` over distinct rows (m, 3): (m, 6).
void evaluate_fresh(const Evaluator &ev, const double *rows, size_t m, const int64_t *labels,
                    const SheetPlan *plan, ni_core::ColumnCache *cache, Six *out);

// TEST-ONLY (momwire#1595 option A): one cone surface's table as the fills
// build it (`ConeSheet`), grown to u_max: its u cell edges and its values,
// [node][height][key].
void cone_table(double k_p, cd k_m, double d, double zf, double beta, double a,
                const std::vector<double> &zs, int p, double u_max, int n_threads,
                std::vector<double> &u_edges, std::vector<cd> &vals);

// momwire#1860's opt-in features on this C++ route (`momwire._features`).
// Two sets: what the run asked for (`requested_features`, written once by the
// executable from its environment before any deck is read) and what is on
// right now (`active_features`). The Python solvers turn these on only around
// their buried fill -- razor's `_assemble_Z_from_prepared`, bspline's
// `_compute_Z_operator_buried`, under `_near_interface`'s `*_scope`s -- so the
// C++ fills do the same with a `FeatureFill` over theirs, and whatever else
// this route evaluates (a buried near field, say) sees every one off. All
// false is the route before momwire#1860 bit for bit. Requested from the
// start: the default-on ones (`Feature.default`: factored_columns,
// height_sheet_recal), as in Python.
//
// Written by the thread that runs the fill, before any parallel region of
// it starts; the fill's worker threads only read it.
struct Features {
    // momwire#1849: the point-matched sinusoidal fill's complex-k field
    // tensor on the complex Galerkin fill's staged rows
    // (`sinusoidal_galerkin::point_field_tensor_cplx_staged`). Read from
    // `requested_features()` at the sinusoidal fill's two in-medium kernel
    // calls, as `SinusoidalSolver._field_tensor` reads its flag, not under a
    // `FeatureFill` (which would turn on the default-on levers that solver
    // never takes). Not in any executable's reach (`EXE_FEATURES`): reached
    // through the in-process hook (`sinusoidal_nec4_printout(features=)`).
    bool point_cplx_staged = false;
    bool factored_columns = false;    // L1: the twin's grid route (`mw899::Grid`)
    bool height_sheet_recal = false;  // L2: the recalibrated height census
    bool crossing_table = false;      // L5: the crossing rows from one table
    bool crossing_above_rule = false;  // B1: bspline's above x above rule
    bool crossing_sheet_retune = false;  // momwire#1956: the sheets' build weights retuned
    // momwire#1898: bspline's junction rows eliminated before the factor.
    // Not a fill lever: `bspline::solve_ports` reads it from the REQUESTED
    // set at solve time, as `BSplineSolver` reads its flag at construction.
    // No executable requests it (outside EXE_FEATURES); in-process test
    // hooks do.
    bool kcl_nullspace = false;
    // momwire#1454: the exact kernel's point moments as a k-series. Not a
    // near-interface lever either: the razor seam reads it into
    // `FrontInputs::exact_kernel_series`.
    bool exact_kernel_series = false;
    // momwire#1971: razor2p's fill-pair reuse. Not a near-interface lever:
    // carried here because this is where an executable's requested features
    // live; the razor seam reads it into `FrontInputs::pair_reuse`.
    bool razor_pair_reuse = false;
    // razor2p's cross blocks from anchors across a sweep
    // (`_razor_crossing_core.h`, `set_band_plan`); not a near-interface lever
    // either: the crossing fill reads it from the active features.
    bool crossing_anchors = false;
    // momwire#1911: the point-matched sinusoidal fill's pair reuse
    // (`_sin_pair_reuse_core.h`), the one fill-reuse switch's slot. Not a
    // near-interface lever either: the C++ fill reads it from the REQUESTED
    // set, as `SinusoidalSolver` reads its flag at construction. The
    // executables request it with MOMWIRE_X_PAIR_REUSE (`EXE_PARTS` in
    // standalone/momwire_razor2p.cpp); the point-matched fill is its only
    // reader, so on another route it only names the switch on line 2. The
    // in-process hook (`sinusoidal_nec4_printout(features=)`) requests it too.
    bool pair_reuse = false;
    // momwire#1809: `pair_reuse`'s part on bs2, `offset_reuse` (a uniform
    // edge's same-edge regularised moments by offset, inside the same-edge
    // kernel). Not a near-interface lever either: bs2's
    // C++ fill reads it, with `pair_reuse` for the census
    // (`_nec5_bspline_printout_core.cpp`), from the REQUESTED set. The bs2
    // executables request it through the whole (`EXE_PARTS`) or by its own
    // variable.
    bool offset_reuse = false;
    // momwire#1847: the transmitted grid blended from frequency anchors
    // (`somm_trans::anchored_grid_reading`). Not a near-interface lever
    // either: the mixed blocks' grid requests read it, bs2's and the
    // point-matched sinusoidal lane's. Its reach is level 1, so no
    // executable carries it (no EXE_FEATURES row) and none requests it.
    bool transmitted_anchors = false;
};
Features &requested_features();
const Features &active_features();
// Asks for the feature `name` (its `momwire._features` name) in every later
// fill, named on line 2 when `stamped` (false for a default-on feature);
// false, and nothing asked, for a name this route does not carry.
bool request_feature(const std::string &name, bool stamped = true);
// Withdraws `name` (a default-on feature the environment turns off, or one
// outside the executable's reach); false for a name this route does not carry.
bool withdraw_feature(const std::string &name);
// Line 2's feature tail for what was requested stamped, as
// `_features.stamp()` spells it: "+name" each, in registry order, or "";
// followed by " uncertified on this platform" once `stamp_uncertified(true)`
// (a certification build running a feature past its reach, momwire#1911).
std::string requested_stamp();
void stamp_uncertified(bool on);
// Back to the start: the default-on features requested unstamped, nothing
// else, and no " uncertified" tail. An executable re-requests what one deck's
// pragma cards add (`DeckPragmas` in standalone/momwire_razor2p.cpp) from
// here, and what its environment asks after that deck.
void reset_requested();

// One buried fill with the requested features on, as the Python solvers'
// scopes put them around theirs; restores what was on before.
class FeatureFill {
  public:
    FeatureFill();
    ~FeatureFill();
    FeatureFill(const FeatureFill &) = delete;
    FeatureFill &operator=(const FeatureFill &) = delete;

  private:
    Features saved_;
    bool saved_taken_;
};

// The requested features with `names` (registry names) added for one run,
// unstamped, and what was requested before put back when it ends -- a test
// hook's `features=` argument: an in-process run of what an executable's
// environment would ask for, for the features no executable carries
// (momwire#1849's point_cplx_staged, momwire#1898's kcl_nullspace). A name
// this route does not carry throws std::invalid_argument, and nothing is
// then left requested.
//
// Threads. The requested set is process-global (it is what an executable's
// environment sets once), and a hook runs its route with the GIL released,
// so two Python threads can be inside two hooks at once. Unguarded, their
// save / restore pairs need not nest: A saves S and asks a, B saves S+a,
// A restores S under B's run, B restores S+a -- `a` left on for the rest of
// the process, which breaks every later flag-off run in it. So every scope,
// an empty `names` included, holds one process-wide lock for its lifetime:
// two scopes never overlap, and a hook's run sees exactly what it asked for.
//
// Construct it with the GIL RELEASED (inside the hook's
// `py::gil_scoped_release`, or around a short one). A scope waiting on the
// lock while holding the GIL would deadlock against a run inside another
// scope that takes the GIL back, as the numpy host's allocations do.
//
// Not covered: a run OUTSIDE any scope (a fill on another thread that
// constructs no FeatureRequest) still reads whatever a live scope asked
// for. The scopes are test hooks' (no production route constructs one);
// tests/test_kcl_nullspace_cxx_1898.py holds the two-thread case.
class FeatureRequest {
  public:
    explicit FeatureRequest(const std::vector<std::string> &names);
    ~FeatureRequest();
    FeatureRequest(const FeatureRequest &) = delete;
    FeatureRequest &operator=(const FeatureRequest &) = delete;

  private:
    std::unique_lock<std::mutex> lock_;  // first: held before saved_ is read
    Features saved_;
};

// Whether the running fill has taken L5's crossing table (`_CrossingFill.taken`).
bool crossing_table_taken();

// The end terms of a crossing block that Python asks for BEFORE the block's
// main tiles, though this route evaluates them after (see the one use, in
// `_razor_crossing_core.cpp`): around them, the table counts as taken only if
// it was before the main tiles (`taken_before`, read then); afterwards it is
// taken if either the main tiles or the end terms took it.
class CrossingEndsFirst {
  public:
    explicit CrossingEndsFirst(bool taken_before);
    ~CrossingEndsFirst();
    CrossingEndsFirst(const CrossingEndsFirst &) = delete;
    CrossingEndsFirst &operator=(const CrossingEndsFirst &) = delete;

  private:
    bool after_main_;
};

// Python's repr of a float (the refusals' numbers).
std::string py_repr(double x);

}  // namespace ni_eval
