// The point-matched sinusoidal fill's `pair_reuse` (momwire#1911): the
// sinusoidal basis's ADAPTER to the common reuse core (`_pair_reuse_core.h`,
// momwire#2063), Python-free, and the one plan both routes run: the Python
// route (`_sin_pair_reuse.py`, through `_accel_sin_pair_reuse.cpp`) and the
// C++ fill (`_sinusoidal_fill_core.cpp`, which the nec42-sinusoidal
// executable and the in-process NEC-4.2 hook run).
//
// THE CORE runs the census (`pair_reuse::census`: exact-equality classes of
// the keys below, numbered `FirstAppearance`, where each first occurs in the
// pairs' flat order whatever the thread count), keeps the plans
// (`pair_reuse::plan_for`, basis "sinusoidal"), counts them, and draws the
// fill report's wire map. The near pairs share one SENTINEL key (`Key::Near`)
// and are listed in the family's `near_flat`; their values are evaluated
// directly and written over the gather.
//
// THIS ADAPTER supplies what is the basis's own:
// - the keys (`DirectKey`, `GroundKey`): the four source-frame scalars, or
//   the reflected term's ground invariants, quantised at KEY_BITS against
//   the pair's singular distance, numpy's operations in their order with no
//   contraction, as the numpy census they replace (momwire#2063 stage 2)
//   computed them; the canonical orientation and its 2-bit code
//   (`canonical`); the near rule, max(|z|, rho) < 4 max(2H, a);
// - the classes renumbered by a stable sort on the representative's
//   observer type (the direct plan), so each radius's classes are one run;
// - THE ALLOWANCE (momwire#2123, the common core's): each plan (the direct
//   block's, the image's or the reflected term's, each its own class map)
//   keeps at most M N / 2 classes, the census stopping once it passes that
//   (`pair_reuse::class_cap`), and its fill (`fill_nbytes`) is held to
//   `pair_reuse::plan_allowance(M N)`, 32 B a pair; the census also stops
//   once the least the finished plan would keep passes that (the .cpp's
//   KEPT_BYTES_PER_CLASS). The census's peak is
//   counted in Python's figures (`pair_reuse::BandBudget`, unbounded): what
//   one running `IntGroups` fold table would hold (its keys and slots, 9 B
//   per class, 8 B per near pair, the class map) plus `BAND_BYTES_PER_PAIR`
//   per band in flight, bands of `band_rows` rows started over `workers`,
//   each cut to BAND_SHARE of the allowance. The census's own tables are
//   `IntGroups` too, one per part.
//
// THE VALUES are the shipped kernel's, called as the fill calls it: the
// canonical observer row against `EVAL_CHUNK` classes of one observer radius
// at a time (a source's position in the kernel's batched sincos can move its
// value an ulp, so the class order and the pieces are part of the plan), and
// the near pairs and the ground's representatives in `near_blocks`' calls.
// The caller supplies the kernels (`CanonFn`, `RowsFn`); this file never
// calls one itself.
//
// THE REDUCTION is the fused gather + reduce (momwire#1916, #1969), one copy
// for both routes, with the orientation's transform, `CODE_SIGNS`.
#pragma once

#include "_pair_reuse_core.h"

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
constexpr int64_t VALUE_BYTES = 48;
constexpr int64_t BAND_BYTES_PER_PAIR = 136;
constexpr int64_t GROUND_BAND_BYTES_PER_PAIR = 168;
constexpr double BAND_SHARE = 0.25;
constexpr int64_t PLAN_FLOOR_BYTES_PER_CLASS = VALUE_BYTES + 8;
static_assert(PLAN_FLOOR_BYTES_PER_CLASS == mw::pair_reuse::PLAN_FLOOR_BYTES_PER_CLASS,
              "the allowance's per-class floor is sinusoidal's values and index");
// The core's cache tag.
const char *const BASIS = "sinusoidal";

// TEST HOOKS, bound for Python's tests only; nothing in the executables sets
// them. The C++ fill's kept-plan allowance in bytes, per plan (<= 0 restores
// the derived one, `pair_reuse::plan_allowance`; read back as -1), the
// small-deck thresholds (< 0 restores each constant), and the census's
// worker count (<= 0: the OpenMP team's, `_workers()`'s counterpart). A plan
// is keyed on the allowance it was built under, as Python's.
void set_plan_allowance(int64_t bytes);
int64_t plan_allowance_override();
void set_small_deck_segs(int64_t free_space, int64_t finite_ground);
int64_t small_deck_segs(bool finite);
void set_workers(int workers);
int workers();
// TEST HOOK (the adversarial gates): a deliberately wrong plan, so a test
// can show the drift gates and `verify` catch it. 0 none; PROBE_NEAR_H the
// near rule in source lengths alone (2H, not max(2H, a)); PROBE_HEIGHT_BLIND
// the reflected key without its height sum; PROBE_FREE_KEY the reflected
// term keyed as the direct block (its 3-D key, uncoded, the types without
// the vertical tangent); PROBE_GROUND_CODES + j the reflected codes mapped
// through `kProbeCodes[j]` (their flips swapped, the source's lost, the
// observer's lost). A plan is keyed on the probe it was built under.
enum : int {
    PROBE_NEAR_H = 1,
    PROBE_HEIGHT_BLIND = 2,
    PROBE_FREE_KEY = 3,
    PROBE_GROUND_CODES = 4,  // + 0, 1, 2
};
void set_probe(int probe);
int probe();

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

// `Plan` (direct / PEC image) or `GroundPlan` (`ground`): the core's family
// (its class map, (M, N) class << CODE_BITS | code when coded, else class;
// the representatives' codes; the near pairs) and what this basis evaluates
// it with.
struct Plan {
    bool ground = false;
    int64_t M = 0, N = 0;
    bool coded = false;
    mw::pair_reuse::FamilyPlan fam;
    std::vector<int64_t> rep;                        // direct: `rep_src`; ground: `rep_flat`
    std::vector<NearBlock> near_blocks, rep_blocks;  // rep_blocks: ground only
    std::vector<double> src_c, src_t;                // direct: the canonical sources (n_cls, 3)
    std::vector<Run> runs;                           // direct only
    int64_t census_peak_bytes = 0;
    const std::vector<int32_t> &cls() const { return fam.map[0]; }
    const std::vector<int64_t> &near_flat() const { return fam.near_flat; }
    int64_t n_classes() const { return static_cast<int64_t>(rep.size()); }
    int64_t n_near() const { return static_cast<int64_t>(fam.near_flat.size()); }
    // `plan_nbytes`: the arrays the plan holds, in Python's terms (the class
    // map, the representatives, their codes when coded, the near pairs and
    // the calls' index arrays, the canonical sources).
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
    int64_t allowance = -1;  // the kept plan's bytes, < 0: pair_reuse::plan_allowance(M N)
    int workers = 1;
    int64_t band_bytes_per_pair = -1;  // < 0: BAND_BYTES_PER_PAIR (ground: GROUND_...)
};

// The allowance a plan of these inputs is held to (`CensusInputs::allowance`).
int64_t allowance_of(const CensusInputs &in);

// `build_plan` / `build_ground_plan`, uncached; null where it declines, and
// `why` (when given) the reason, `pair_reuse_declined`'s words.
std::shared_ptr<const Plan> build_plan(const CensusInputs &in, std::string *why = nullptr);
std::shared_ptr<const Plan> build_ground_plan(const CensusInputs &in, std::string *why = nullptr);

// The plan of these inputs from the process's cache (the core's, keyed on
// every array the census reads, the allowance, the rows and the test hooks
// in force), else built and kept, a declined one too. Null: declined, and
// `why` the reason. `built` says whether this call ran the census. `keep`
// is the core's (`pair_reuse::plan_for`): 2 for a fill's second block.
std::shared_ptr<const Plan> shared_plan(const CensusInputs &in, bool ground, bool *built = nullptr,
                                        std::string *why = nullptr, int keep = 1);
void clear_shared_plans();

// Plans built (censuses run), of those the ones over budget, and plans found
// in the cache: process totals (the core's, for this basis). Built -
// declined >= 1 is what proves a fill reused. `last` is the last C++ fill's
// outcome: "" (reuse not asked, or the fill took a route the reuse never
// sees), "served", or the reason it declined, `pair_reuse_declined`'s words.
mw::pair_reuse::Counters counters();
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

// FOR THE TESTS: every pair's key words (M * N rows of 5, or 6 for the
// reflected term), its code and whether it is near, as the census keys them;
// and `canonical` on given key rows (`ia`, `ib`, `it` the odd columns, `ds`
// / `do_` per source / observer or null).
void pair_keys(const CensusInputs &in, bool ground, std::vector<int64_t> &words, std::vector<int8_t> &codes,
               std::vector<uint8_t> &near);
void canonical_codes(int64_t *rows, int width, int64_t n, int ia, int ib, int it, const uint8_t *near,
                     int64_t n_src, const int64_t *ds, const int64_t *do_, int8_t *codes);

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
