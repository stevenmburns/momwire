// The point-matched sinusoidal fill's `pair_reuse` (momwire#1911) in C++,
// Python-free: `_sin_pair_reuse.py`'s census, plans, class evaluation and
// fused reduction, for the C++ fill (`_sinusoidal_fill_core.cpp`, which the
// nec42-sinusoidal executable and the in-process NEC-4.2 hook run) and the
// extension's bindings (`_accel_sin_pair_reuse.cpp`).
//
// PYTHON IS THE REFERENCE, and the plan is its plan to the byte: the same
// key columns (`_band_keys`, `_ground_band_keys`, in the same order of
// floating-point operations, unfused), the same canonical orientation and
// codes (`_canonical_codes`), the same grouping and numbering. A class is
// numbered where it first occurs in the pairs' flat order: each band groups
// its pairs in flat order and its distinct keys are folded, in band order,
// into one running table (`IntGroups`, `_int_groups_core.h`, the very code
// Python's `_Fold` calls), then renumbered by a stable sort on the
// representative's observer type (direct) or flat index (ground). The type
// ids are `np.unique(axis=0)`'s: rank of each row among the distinct rows,
// lexicographic in signed int64.
//
// THE BUDGET is Python's accounting, not a measurement of this route: what
// `_Fold.held` counts (the class map, the table's `nbytes`, 9 B per class and
// 8 B per near pair kept) plus `BAND_BYTES_PER_PAIR` per band started and not
// folded, bands started by `_census_bands`' rule over the same number of
// workers, the same band height (`_band_rows`) and the plan's fill
// (`fill_nbytes`) held to what is left. So a deck declines here exactly where
// Python's census declines at the same worker count, with the same reason.
// What this route holds per band in flight (its keys and band table, at
// most ~72 B a pair) is under the 136 B Python measured, so the count bounds
// it too.
//
// THE VALUES are the shipped kernel's, called as Python calls it: the
// canonical observer row against `EVAL_CHUNK` classes of one observer radius
// at a time (a source's position in the kernel's batched sincos can move its
// value an ulp, so the pieces are Python's), and the near pairs and the
// ground's representatives in `near_blocks`' calls. The caller supplies the
// kernels (`CanonFn`, `RowsFn`); this file never calls one itself.
//
// THE REDUCTION is `_accel_sin_pair_reuse.cpp`'s fused gather + reduce
// (momwire#1916, #1969), moved here so both routes run one copy.
#pragma once

#include <complex>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <vector>

namespace mw {
namespace sinusoidal {
namespace pair_reuse {

using cd = std::complex<double>;

// `_sin_pair_reuse`'s constants, by name (their comments there carry the
// measurements).
constexpr int KEY_BITS = 48;
constexpr double NEAR_LENGTHS = 4.0;
constexpr int64_t NEAR_ROWS = 16;
constexpr int64_t REP_ROWS = 16;
constexpr int64_t EVAL_CHUNK = 2048;
constexpr int64_t SMALL_DECK_SEGS_FREE = 400;
constexpr int64_t SMALL_DECK_SEGS_GROUND = 200;
constexpr int CODE_BITS = 2;
constexpr int64_t CODED_PAIRS = int64_t(1) << (31 - CODE_BITS);
constexpr int64_t PLAN_BUDGET_BYTES = int64_t(256) << 20;
constexpr int64_t SHARED_PLAN_BYTES = PLAN_BUDGET_BYTES;
constexpr int64_t VALUE_BYTES = 48;
constexpr int64_t BAND_BYTES_PER_PAIR = 136;
constexpr int64_t GROUND_BAND_BYTES_PER_PAIR = 168;
constexpr double BAND_SHARE = 0.25;
constexpr int64_t PLAN_FLOOR_BYTES_PER_CLASS = VALUE_BYTES + 8;

// `over_budget_reason()`, for the budget in force.
std::string over_budget_reason();

// TEST HOOKS, bound for Python's tests only; nothing in the executables sets
// them. The budget in force (<= 0 restores PLAN_BUDGET_BYTES), the small-deck
// thresholds (< 0 restores each constant), and the census's worker count
// (<= 0: the OpenMP team's, `_workers()`'s counterpart). A plan is keyed on
// the budget it was built under, as Python's.
void set_plan_budget(int64_t bytes);
int64_t plan_budget();
void set_small_deck_segs(int64_t free_space, int64_t finite_ground);
int64_t small_deck_segs(bool finite);
void set_workers(int workers);
int workers();

// Per-segment type ids (`reversible_segment_types`): the source type, the
// observer type, and the source type with its tangent reversed, in one id
// space. `src_a` null: the reduced kernel's (half-length only).
struct Types {
    std::vector<int64_t> src, obs, src_rev;
};
Types reversible_segment_types(int64_t n, const double *seg_h, const double *a_obs, const double *src_a,
                               const int8_t *ind1, const int8_t *ind2);
// `reversible_ground_types`: (g_src, g_obs, g_src_rev, g_obs_rev).
struct GroundTypes {
    std::vector<int64_t> src, obs, src_rev, obs_rev;
};
GroundTypes reversible_ground_types(const Types &t, int64_t n, const double *seg_t);

// One `near_blocks` call: observer rows [m0, m1) against sources `ns`, the
// pairs p0..p1 at (r, c) of its block.
struct NearBlock {
    int64_t m0 = 0, m1 = 0, p0 = 0, p1 = 0;
    std::vector<int64_t> ns, r, c;
};
struct Run {  // `Plan.runs`: classes [c0, c1) share observer segment g_obs's radius
    int64_t c0 = 0, c1 = 0, g_obs = 0;
};

// `Plan` (direct / PEC image) or `GroundPlan` (`ground`).
struct Plan {
    bool ground = false;
    int64_t M = 0, N = 0;
    bool coded = false;
    std::vector<int32_t> cls;       // (M, N): class << CODE_BITS | code (coded), else class
    std::vector<int64_t> rep;       // direct: `rep_src`; ground: `rep_flat`
    std::vector<int8_t> rep_code;   // empty when not coded
    std::vector<int64_t> near_flat;
    std::vector<NearBlock> near_blocks, rep_blocks;  // rep_blocks: ground only
    std::vector<double> src_c, src_t;                // direct: the canonical sources (n_cls, 3)
    std::vector<Run> runs;                           // direct only
    int64_t census_peak_bytes = 0;
    int64_t n_classes() const { return static_cast<int64_t>(rep.size()); }
    int64_t n_near() const { return static_cast<int64_t>(near_flat.size()); }
    // `plan_nbytes`: the arrays Python's plan holds, in bytes.
    int64_t plan_nbytes() const;
    // `fill_nbytes`: those and one wavenumber's values.
    int64_t fill_nbytes() const { return plan_nbytes() + VALUE_BYTES * (n_classes() + n_near()); }
};

// What a census reads, views into the caller's arrays. The ground census
// reads `ground_z` and `obs_rev` (`build_ground_plan`).
struct CensusInputs {
    int64_t M = 0, N = 0;
    const double *obs_c = nullptr, *obs_t = nullptr;  // (M, 3)
    const double *src_c = nullptr, *src_t = nullptr;  // (N, 3)
    const double *seg_h = nullptr;                    // (N)
    const double *a_obs = nullptr;                    // (M)
    const int64_t *src_type = nullptr, *obs_type = nullptr;
    const int64_t *src_rev = nullptr, *obs_rev = nullptr;  // null: keyed along the stored tangents
    double ground_z = 0.0;
    int64_t rows = 1;     // the census's swept rows (`census_rows`)
    int64_t budget = -1;  // bytes, < 0 for none
    int workers = 1;
};

// `build_plan` / `build_ground_plan`; null is `OVER_BUDGET`.
std::shared_ptr<const Plan> build_plan(const CensusInputs &in);
std::shared_ptr<const Plan> build_ground_plan(const CensusInputs &in);

// `shared_plan`: the plan for `key` from the process's cache, else `build()`'s,
// kept (null, an over-budget census, is kept too). Least recently used first
// out past SHARED_PLAN_BYTES, the newest kept whatever its size. `built`
// says whether this call ran the census.
std::shared_ptr<const Plan> shared_plan(const std::string &key,
                                        const std::function<std::shared_ptr<const Plan>()> &build,
                                        bool *built = nullptr);
void clear_shared_plans();

// Plans built (censuses run), of those the ones over budget, and plans found
// in the cache: process totals. Built - declined >= 1 is what proves a fill
// reused. `last` is the last fill's outcome: "" (reuse not asked, or the
// fill took a route the reuse never sees), "served", or the reason it
// declined, `pair_reuse_declined`'s words.
struct Counters {
    uint64_t built = 0, declined = 0, found = 0;
};
Counters counters();
void reset_counters();
std::string last_outcome();
void set_last_outcome(const std::string &s);
// The last fill's census per block, as `pair_reuse_census`: (pairs, classes,
// near pairs), block "direct", "image" or "reflected".
struct CensusEntry {
    std::string block;
    int64_t pairs = 0, classes = 0, near = 0;
};
std::vector<CensusEntry> last_census();
void set_last_census(std::vector<CensusEntry> c);

// The kernels the evaluation calls. `rows`: the shipped evaluator in the
// original frame on observer rows [m0, m1) against source segments ns[0..n),
// each out[s] (m1 - m0, n) in C order. `canon`: the shipped kernel for the
// canonical observer (the origin, along +z) at observer segment g_obs's
// radius against n canonical sources (centres, tangents (n, 3)) standing for
// source segments ns, each out[s] (n,). Called concurrently, each call on
// its own slices.
using RowsFn = std::function<void(int64_t m0, int64_t m1, const int64_t *ns, int64_t n, std::vector<cd> *out)>;
using CanonFn = std::function<void(const double *src_c, const double *src_t, const int64_t *ns, int64_t n,
                                   int64_t g_obs, std::vector<cd> *out)>;

// One wavenumber's values: (n_cls, 3) and (n_near, 3), row-major.
struct Values {
    std::vector<cd> cls, near;
};
// `evaluate` (the direct plan's classes, `to_canonical`ed), `evaluate_classes`
// (a ground plan's), `evaluate_near`.
void evaluate(const Plan &p, const CanonFn &canon, Values &out);
void evaluate_classes(const Plan &p, const RowsFn &rows, Values &out);
void evaluate_near(const Plan &p, const RowsFn &rows, Values &out);

// The fused gather + reduction (`reduce_band`): one block's class map over
// the band and where its values come from.
struct Csc {
    const int64_t *ptr;
    const int64_t *ind;
    const double *val;  // interleaved (re, im)
};
struct Block {
    const int32_t *cls = nullptr;  // (rows, n_src), row-major
    const double *rep = nullptr;   // (n_cls, 3) complex, interleaved
    int64_t n_cls = 0;
    const int64_t *near_idx = nullptr;  // (k,) sorted flat r * n_src + n, band-local
    const double *near_val = nullptr;   // (k, 3) complex
    int64_t n_near = 0;
    bool coded = false;   // entries are class << 2 | code (momwire#1969)
    bool neg[4][3] = {};  // shape negated under each code
};
// The coded blocks' sign table (`CODE_SIGNS`) as negations.
void set_code_signs(Block &b);
// out (rows, cols) complex, row stride `row_bytes`, columns contiguous:
// `Φ0@M0 + Φ1@M1 + Φ2@M2`, Φ gathered from `d` less `image` (when given).
// OpenMP over rows (n_threads > 0 sets the team).
void reduce_band(char *out, std::ptrdiff_t row_bytes, std::ptrdiff_t rows, std::ptrdiff_t cols,
                 std::ptrdiff_t n_src, const Block &d, const Block *image, const Csc M[3], int n_threads);
// `scatter`: a band's three Φ gathered from `b` (each out[s] rows * n_src),
// for the dense regime's matmul.
void scatter(const Block &b, std::ptrdiff_t rows, std::ptrdiff_t n_src, cd *out[3]);
// A band's block over plan rows [i0, i1): its class map rows, the values, and
// its near pairs' band-local indices (into `near_idx`, which it fills).
Block band_block(const Plan &p, const Values &v, int64_t i0, int64_t i1, std::vector<int64_t> &near_idx);

}  // namespace pair_reuse
}  // namespace sinusoidal
}  // namespace mw
