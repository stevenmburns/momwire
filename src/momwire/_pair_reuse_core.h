// The basis-independent half of fill-pair reuse (momwire#2063): the census
// that numbers the classes of a family's pairs from a basis's key, the
// evaluation groups of their representatives, the plan's byte accounting and
// declines, the process's plan cache and its counters, the decode loop of
// the gather and the fill report's wire map. Python-free.
//
// A basis supplies an ADAPTER (razor2p's: `_razor_pair_reuse_core.cpp`):
// its families, its key (what a pair's value reads of the geometry, and the
// orientation code it folds), the evaluation of each class's representative,
// the transform a gathered value takes when its orientation differs from its
// representative's, and its policy constants. The census and the gather are
// templates instantiated in the adapter's TU, so a key's float arithmetic is
// compiled beside the adapter's code; everything compiled in this header's
// own TU (`_pair_reuse_core.cpp`) is integer-only, so no value moves with it
// (momwire#1187).
//
// THE MAP. A family is one class space over `n_blocks` blocks of rows x cols
// pairs (razor's direct moments: the real and the PEC image block of
// centroid observers against source segments). Each pair's entry is
// class << code_bits | code in a non-negative int32, the code being the
// pair's orientation relative to its key's canonical one; a member is its
// class's value transformed by (its code XOR its representative's).
//
// CLASS NUMBERING. The census hashes keys into P parts (P = min(MAX_PARTS,
// OpenMP threads), by hash bits), each part numbering its classes in first
// appearance in row-major order, block by block. `PartOffsets` (razor's)
// keeps the parts' concatenated, so numbering depends on P; the groups' calls
// do not (each group's source union is sorted), so neither does any value
// razor gathers. `FirstAppearance` (sinusoidal's) renumbers them where each
// first occurs in the pairs' flat order, block by block, whatever P: a class's
// representative is its first pair either way, so this is one permutation.
//
// NEAR PAIRS. A key may leave a pair unshared (`Key::Own`: its own class, a
// "single", razor's near pairs) or share it under one SENTINEL key the
// adapter writes (`Key::Near`: a class like any other, its pairs also listed
// in `near_flat`, sinusoidal's near pairs, whose values the adapter computes
// directly and writes over the gather).
#pragma once

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <functional>
#include <map>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace mw {
namespace pair_reuse {

enum class Numbering : uint8_t { PartOffsets, FirstAppearance };

// A family: one class space over n_blocks blocks of (rows x cols) pairs.
struct FamilySpec {
    const char *name = "";
    int key_words = 0;   // the census's W
    int code_bits = 0;   // the orientation code's width in a map entry
    int n_blocks = 1;    // at most 2
    int64_t rows = 0, cols = 0;  // observers x sources
    // Per-k values of one class (razor: two complex, 32 B). With the group
    // entry (cls, at: 8 B) and the representative's code (1 B) it is what a
    // class costs the plan past the maps (`class_bytes`).
    int64_t value_bytes = 0;
    // Whether a pair the key leaves unshared is a NEAR pair (its own class,
    // counted near by the report), or only an uncodeable one (counted as
    // computed).
    bool near_singles = false;
    // TEST HOOK: decline past this many classes (0: the int32 coding's own
    // limit only).
    int64_t class_limit = 0;
    Numbering numbering = Numbering::PartOffsets;
    // Keep each class's representative's flat index (`rep_flat`).
    bool keep_reps = false;
};

inline int64_t class_bytes(const FamilySpec &f) { return f.value_bytes + 2 * 4 + 1; }

// The adapter's census policy.
struct Policy {
    // A census keeping more classes than this fraction of its family's pairs
    // declines, POOR_REASON.
    double max_class_fraction = 1.0;
    // Observer rows per representative evaluation call, at most; 0 builds
    // no groups (an adapter with its own evaluation schedule).
    int rep_rows = 16;
    // Observer rows per census pass (0: `detail::CENSUS_ROWS`). A census
    // hook (`census`) sees the classes after each pass.
    int64_t census_rows = 0;
    // The census's own byte accounting against `room` (razor's). Off, only
    // the hook's and the class coding's limits stop it.
    bool measured = true;
};

// The groups the representatives are evaluated in: observer rows of one
// block against the union of their representatives' sources (ascending).
struct Group {
    uint8_t block = 0;
    std::vector<int32_t> obs, src;  // observer rows, the union of their reps' sources
    std::vector<int32_t> cls, at;   // each class's value at at = row * |src| + col
};

struct FamilyPlan {
    FamilySpec spec;
    std::vector<int32_t> map[2];    // per block, class << code_bits | code
    std::vector<uint8_t> rep_code;  // each class's representative's code
    std::vector<Group> groups;
    // One bit per class, set for a near pair's own class (`near_singles`).
    // Outside `nbytes()` and so outside the budget, which it must not move:
    // 1/8 B per class beside the class_bytes the budget counts.
    std::vector<uint64_t> near_bits;
    // `Key::Near` pairs, ascending flat index b * rows * cols + o * cols + s.
    std::vector<int64_t> near_flat;
    // `keep_reps`: each class's representative's flat index, likewise.
    std::vector<int64_t> rep_flat;
    int64_t pairs = 0, classes = 0, near = 0;
    int64_t rep_evals = 0;  // pairs the representatives' calls evaluate
    // The fill report's split (momwire#2016), by the block of each class's
    // representative: its classes, those that are one near pair, and the
    // pairs its calls evaluate.
    int64_t cls_block[2] = {0, 0}, near_block[2] = {0, 0}, evals_block[2] = {0, 0};
    // Maps, representatives' codes, groups and the two lists.
    size_t nbytes() const;
};

struct Plan {
    std::string basis;        // the cache's tag; outside `nbytes()`
    std::string fingerprint;  // the cache's key, counted in `nbytes()`
    // Why this plan does not serve the deck, or empty: a declined plan has
    // no classes, and the caller fills as without reuse. It is cached like
    // any other, so the census is not re-run per frequency.
    std::string declined;
    std::vector<FamilyPlan> fam;
    double census_seconds = 0.0;
    // What the budget counted: nbytes() plus one wavenumber's class values.
    int64_t budget_bytes = 0;
    // The largest census-plus-plan estimate the budget saw (kept when the
    // plan declines, so a declined deck says how far it got).
    int64_t census_peak_bytes = 0;
    // The adapter's own arrays beside the families (sinusoidal's plan), which
    // the cache holds with the plan and counts as `ext_bytes`.
    std::shared_ptr<const void> ext;
    size_t ext_bytes = 0;
    size_t nbytes() const;
};

// Every family's maps, 4 B per pair per block: known before any census.
int64_t maps_bytes(const Plan &p);
// nbytes() plus one wavenumber's values of every class planned so far.
int64_t held_bytes(const Plan &p);

enum class Outcome : int8_t { Ok, Uncodeable, Poor, OverBudget };
enum class Key : uint8_t { Shared, Own, Near };

const char *const POOR_REASON = "too few shared pairs to repay the plan";
std::string budget_reason(int64_t budget_bytes);
// A plan that does not serve: its families keep their specs and lose their
// classes; its census time and peak are kept.
void decline(Plan &p, const std::string &why);

// Bytes of plans the process keeps between fills, one fill's budget
// (momwire#2026): a long-lived server holds no more than a single fill may use.
constexpr size_t SHARED_PLAN_BYTES = size_t(256) << 20;

// The plan of `basis` keyed `fingerprint`, from the process's cache or built:
// `build` fills a fresh Plan whose basis and fingerprint are already set (the
// fingerprint counts in its budget), and its wall time is the plan's
// census_seconds, a decline's included (momwire#2016). The cache keeps the
// most recent plans of every basis under SHARED_PLAN_BYTES, the newest
// always. `built`, when given, says whether this call ran `build`.
std::shared_ptr<const Plan> plan_for(const std::string &basis, std::string fingerprint,
                                     const std::function<void(Plan &)> &build,
                                     bool *built = nullptr);
// Plans built and found (process totals, per basis, for tests), and of those
// built the ones that came back declined: a declined plan serves nothing, so
// built - declined is what proves a fill reused.
struct Counters {
    uint64_t built = 0, found = 0, declined = 0;
};
Counters counters(const std::string &basis);
void reset_counters(const std::string &basis);
void clear_cache(const std::string &basis);
// TEST HOOK: the cache's bytes in place of SHARED_PLAN_BYTES (< 0 restores
// it; 0 keeps no plan, so every fill builds its own).
void set_cache_bytes(int64_t bytes);

// A family's classes renumbered, class c becoming new_of_old[c] (a
// permutation): its maps (codes kept), representatives' codes, `rep_flat`
// and near bits. A family with groups cannot be renumbered.
void renumber(FamilyPlan &f, const std::vector<int64_t> &new_of_old);

// THE BAND BUDGET (sinusoidal's census accounting, Python's `_census_bands`
// as momwire#2019 wrote it): a census in bands of rows, up to `workers` of
// them in flight, each counted at `band_bytes` until it is folded. A band
// starts only where it fits beside what the census holds under the budget,
// and the census stops once what it holds, or the plan's floor, passes it.
// The census calls `start(i)` before band i (false: not one band fits beside
// what is held, the census stops) and `fold(held, floor)` after it, with what
// the census holds then and the plan's least possible fill (false: past the
// budget). `budget` < 0: none. `peak()` is the most it counted.
class BandBudget {
  public:
    BandBudget(int64_t budget, int64_t band_bytes, int workers, int64_t n_bands, int64_t held);
    bool start(int64_t i);
    bool fold(int64_t held, int64_t floor);
    int64_t peak() const { return peak_; }
    int64_t max_in_flight() const { return max_in_flight_; }

  private:
    bool fits(int64_t k) const { return budget_ < 0 || held_ + band_bytes_ * k <= budget_; }
    int64_t budget_, band_bytes_, n_bands_, held_;
    int workers_;
    int64_t next_ = 0, cur_ = 0, peak_ = 0, max_in_flight_ = 0;
};

// The fill report's wire map (momwire#2016) of one family: for its block b,
// observer wire i and source wire j, out[((b * W + i) * W + j) * 3 + c] is
// INCREMENTED by that block's pairs computed (c = 0, the first pair of a far
// class), reused (1) and near (2). One pass over the class maps, with one bit
// per class of scratch. Integers only.
void wire_map(const FamilyPlan &f, const int32_t *wire_of_row, const int32_t *wire_of_col,
              int64_t n_wires, int64_t *out);

// ---------------------------------------------------------------------------
// The key's quantisation (each adapter's KEY_BITS and singular distance D).

inline uint64_t bits(double x) {
    uint64_t u;
    std::memcpy(&u, &x, sizeof u);
    return u;
}

// The quantum's exponent and its inverse step: 2^(key_bits - ilogb(D)).
inline void quantum(double D, int key_bits, int &e, double &inv) {
    e = std::ilogb(D);
    inv = std::ldexp(1.0, key_bits - e);
}

// A quantised value as a key word; false when it does not fit (the pair is
// then left unshared).
inline bool qword(double x, double inv, uint64_t &w) {
    const double y = x * inv;
    if (!(std::fabs(y) < 4.0e18)) return false;
    w = static_cast<uint64_t>(static_cast<int64_t>(std::llround(y)));
    return true;
}

// Distinct values, numbered in first appearance (a key's exact types).
struct Ids {
    std::map<std::pair<uint64_t, uint64_t>, int64_t> m;
    int64_t id(uint64_t a, uint64_t b = 0) {
        auto it = m.emplace(std::make_pair(a, b), static_cast<int64_t>(m.size()));
        return it.first->second;
    }
};

// ---------------------------------------------------------------------------
// The census and the gather (templates: instantiated in the adapter's TU).

namespace detail {

// Rows of the census computed per pass (its key buffer is rows * cols keys).
constexpr ptrdiff_t CENSUS_ROWS = 64;
constexpr int MAX_PARTS = 32;
// A partition-local class number, in the census's temporary entries: the
// part in the top `part_bits(P)` bits of 32, then the local class, then the
// code. With one part (one thread) the local class has every bit the final
// entry has; with P parts each part's classes have 2^-part_bits of them.
inline int part_bits(int P) {
    int b = 0;
    while ((1 << b) < P) b++;
    return b;
}

inline int n_parts() {
#ifdef _OPENMP
    const int p = omp_get_max_threads();
#else
    const int p = 1;
#endif
    return std::max(1, std::min(MAX_PARTS, p));
}

inline uint64_t mix64(uint64_t x) {
    x ^= x >> 30;
    x *= 0xbf58476d1ce4e5b9ull;
    x ^= x >> 27;
    x *= 0x94d049bb133111ebull;
    x ^= x >> 31;
    return x;
}

template <int W>
uint64_t hash_key(const uint64_t *k) {
    uint64_t h = 0x9e3779b97f4a7c15ull;
    for (int i = 0; i < W; i++) h = mix64(h ^ k[i]);
    return h;
}

// Open addressing over W-word keys; the value is a partition-local class.
template <int W>
struct Table {
    std::vector<uint64_t> keys;
    std::vector<int32_t> vals;
    size_t cap = 0, n = 0;
    void grow() {
        const size_t nc = cap == 0 ? 1024 : cap * 2;
        std::vector<uint64_t> ok;
        std::vector<int32_t> ov;
        ok.swap(keys);
        ov.swap(vals);
        keys.assign(nc * W, 0);
        vals.assign(nc, -1);
        const size_t oc = cap;
        cap = nc;
        for (size_t i = 0; i < oc; i++)
            if (ov[i] >= 0) place(&ok[i * W], hash_key<W>(&ok[i * W]), ov[i]);
    }
    void place(const uint64_t *k, uint64_t h, int32_t v) {
        size_t i = static_cast<size_t>(h) & (cap - 1);
        while (vals[i] >= 0) i = (i + 1) & (cap - 1);
        std::memcpy(&keys[i * W], k, W * sizeof(uint64_t));
        vals[i] = v;
    }
    // The key's class, or `fresh` (then stored) when it is new.
    int32_t find_or_add(const uint64_t *k, uint64_t h, int32_t fresh, bool &added) {
        if (2 * (n + 1) > cap) grow();
        size_t i = static_cast<size_t>(h) & (cap - 1);
        while (vals[i] >= 0) {
            if (std::memcmp(&keys[i * W], k, W * sizeof(uint64_t)) == 0) {
                added = false;
                return vals[i];
            }
            i = (i + 1) & (cap - 1);
        }
        std::memcpy(&keys[i * W], k, W * sizeof(uint64_t));
        vals[i] = fresh;
        n++;
        added = true;
        return fresh;
    }
};

struct Rep {
    uint8_t block = 0;
    int32_t obs = 0, src = 0;
    uint8_t code = 0;
    uint8_t single = 0;  // 1: a pair the key left unshared; 2: a Key::Near class
};

// The finished census into `f`: the representatives' codes, the groups, the
// near bits and the report's split.
void finish(FamilyPlan &f, const std::vector<Rep> &reps, int64_t singles, const Policy &pol);

// `FirstAppearance`: the classes of the concatenated parts renumbered by
// their representatives' flat order (the map and `reps`).
void first_appearance(FamilyPlan &f, std::vector<Rep> &reps);

// No census hook (razor's).
struct NoHook {
    static constexpr bool tail = false;
    bool start(int64_t) { return true; }
    bool done(int64_t, int64_t, int64_t, bool) { return true; }
};

}  // namespace detail

// THE CENSUS of one family. `key(block, o, s, words, code)` writes a pair's
// W-word key and its orientation code, and returns Key::Own for a pair that
// is its own class, Key::Near for one shared under the adapter's sentinel
// key and listed in `near_flat`. Entries go to `f.map[b]`; each class's
// representative is its first pair, row-major per part, numbered by the
// spec's `numbering`. On Ok, `f` is finished (codes, groups, counts).
// Otherwise it declines:
//   Uncodeable: the classes pass the int32 entry, a part's share of it, or
//     the spec's `class_limit`;
//   Poor: they pass `max_class_fraction` of the family's pairs;
//   OverBudget: the census and the plan it is building pass `room` bytes
//     (the budget less what is already held): its hash tables (at 1.5x their
//     capacity, the peak of their last growth, old and new arrays both live),
//     its representatives, its chunk buffers, and `class_bytes` per class for
//     the plan (`pol.measured`); or the hook stopped it.
// Each bound is checked after every chunk of rows, so a doomed plan stops
// within one chunk of it instead of building its whole table. Before each
// chunk the same sum is projected as if every pair in it were a new class
// (each part's table grown to fit), and a chunk whose projection passes
// `room` is not run, so the tables never grow past the budget. `peak` is the
// largest estimate or projection seen.
//
// THE HOOK (an adapter's own accounting, `detail::NoHook` for none):
// `hook.start(i)` before chunk i (false: it is not run, OverBudget), and
// `hook.done(i, classes, near, tail_new)` after it, with the classes and
// near pairs so far and, when `Hook::tail`, whether the class whose first
// pair in the chunk comes last in it was new in the chunk (false: OverBudget).
// `Tab` is the parts' hash table: `find_or_add(key, hash, fresh, added)`, a
// class numbered in insertion order, and `cap` and `n` for the accounting.
template <int W, class Tab = detail::Table<W>, class KeyFn, class Hook>
Outcome census(FamilyPlan &f, const Policy &pol, int64_t room, int64_t &peak, KeyFn key, Hook &hook) {
    using namespace detail;
    const FamilySpec &spec = f.spec;
    if (spec.key_words != W || spec.n_blocks < 1 || spec.n_blocks > 2)
        throw std::invalid_argument("pair_reuse: a family spec disagrees with its census");
    const int n_blocks = spec.n_blocks, code_bits = spec.code_bits;
    const int64_t rows = spec.rows, cols = spec.cols, limit = spec.class_limit;
    const int64_t CLASS_BYTES = class_bytes(spec);
    const int64_t poor = static_cast<int64_t>(
        pol.max_class_fraction * static_cast<double>(static_cast<int64_t>(n_blocks) * rows * cols));
    const int P = n_parts();
    const size_t nn = static_cast<size_t>(rows) * static_cast<size_t>(cols);
    for (int b = 0; b < n_blocks; b++) f.map[b].assign(nn, 0);
    std::vector<Tab> tables(static_cast<size_t>(P));
    std::vector<std::vector<Rep>> reps(static_cast<size_t>(P));
    std::vector<std::vector<int64_t>> nears(static_cast<size_t>(P));
    // Hook::tail: per part, each local class's last chunk seen (+1), and the
    // chunk's last first-seen pair and whether it made its class.
    std::vector<std::vector<int32_t>> stamps(Hook::tail ? static_cast<size_t>(P) : 0);
    std::vector<int64_t> last_i(static_cast<size_t>(P), -1);
    std::vector<uint8_t> last_new(static_cast<size_t>(P), 0);
    const ptrdiff_t R = std::min<ptrdiff_t>(pol.census_rows > 0 ? pol.census_rows : CENSUS_ROWS, rows);
    const size_t chunk = static_cast<size_t>(R) * static_cast<size_t>(cols);
    std::vector<uint64_t> kbuf(chunk * W);
    std::vector<uint64_t> hbuf(chunk);
    std::vector<uint8_t> cbuf(chunk), sbuf(chunk);
    // The final entry is class << code_bits | code in a non-negative int32;
    // the temporary one puts the part above the local class.
    const int pb = part_bits(P);
    const int part_shift = 31 - pb;
    const int64_t cls_max = int64_t(1) << (31 - code_bits);
    const uint32_t local_max = uint32_t(1) << (part_shift - code_bits);
    std::atomic<bool> overflow{false};
    std::vector<int64_t> n_single(static_cast<size_t>(P), 0);
    int64_t ci = 0;  // the chunk's index, over every block
    for (int b = 0; b < n_blocks; b++) {
        int32_t *mp = f.map[b].data();
        for (int64_t o0 = 0; o0 < rows; o0 += R, ci++) {
            const int64_t o1 = std::min<int64_t>(rows, o0 + R);
            const int64_t m = (o1 - o0) * cols;
            if (!hook.start(ci)) return Outcome::OverBudget;
            // Before the chunk: what the census would hold if this chunk's
            // pairs were all new classes, each part taking up to twice its
            // share (the hash spreads them), its table grown to fit and its
            // representatives' vector doubled. Past the room, the chunk is
            // not run: the tables never grow past the budget.
            if (pol.measured) {
                const int64_t share = 2 * ((m + P - 1) / P) + 1;
                int64_t would = static_cast<int64_t>(chunk * (W * 8 + 8 + 2)), so_far = 0;
                for (int t = 0; t < P; t++) {
                    const auto &rp = reps[static_cast<size_t>(t)];
                    const auto &tab = tables[static_cast<size_t>(t)];
                    const int64_t n_t = static_cast<int64_t>(rp.size());
                    so_far += n_t;
                    // at 1.5x its capacity, as after the chunk below
                    size_t cap = tab.cap == 0 ? 1024 : tab.cap;
                    while (2 * static_cast<size_t>(tab.n + share) > cap) cap *= 2;
                    would += static_cast<int64_t>(3 * cap / 2 * (W * 8 + 4));
                    const size_t rcap = std::max(rp.capacity(), static_cast<size_t>(n_t + share));
                    would += static_cast<int64_t>(
                        (rcap > rp.capacity() ? 2 * rcap : rcap) * sizeof(Rep));
                }
                would += (so_far + m) * CLASS_BYTES;
                peak = std::max(peak, would);
                if (would > room) return Outcome::OverBudget;
            }
#pragma omp parallel for schedule(static)
            for (int64_t i = 0; i < m; i++) {
                const int64_t o = o0 + i / cols, s = i % cols;
                uint64_t *w = &kbuf[static_cast<size_t>(i) * W];
                uint8_t code = 0;
                const Key k = key(b, o, s, w, code);
                cbuf[static_cast<size_t>(i)] = code;
                sbuf[static_cast<size_t>(i)] = k == Key::Own ? 1 : (k == Key::Near ? 2 : 0);
                hbuf[static_cast<size_t>(i)] = k != Key::Own ? hash_key<W>(w) : mix64(static_cast<uint64_t>(o));
            }
#pragma omp parallel for schedule(static, 1)
            for (int t = 0; t < P; t++) {
                Tab &tab = tables[static_cast<size_t>(t)];
                std::vector<Rep> &rp = reps[static_cast<size_t>(t)];
                std::vector<int64_t> &nr = nears[static_cast<size_t>(t)];
                int64_t &li = last_i[static_cast<size_t>(t)];
                li = -1;
                for (int64_t i = 0; i < m; i++) {
                    const uint64_t h = hbuf[static_cast<size_t>(i)];
                    if (static_cast<int>((h >> 40) % static_cast<uint64_t>(P)) != t) continue;
                    const int64_t o = o0 + i / cols, s = i % cols;
                    const uint8_t code = cbuf[static_cast<size_t>(i)];
                    const uint8_t sb = sbuf[static_cast<size_t>(i)];
                    int32_t local;
                    bool added = true;
                    if (sb == 1) {
                        local = static_cast<int32_t>(rp.size());
                        rp.push_back(Rep{static_cast<uint8_t>(b), static_cast<int32_t>(o),
                                         static_cast<int32_t>(s), 0, 1});
                        n_single[static_cast<size_t>(t)]++;
                    } else {
                        local = tab.find_or_add(&kbuf[static_cast<size_t>(i) * W], h,
                                                static_cast<int32_t>(rp.size()), added);
                        if (added)
                            rp.push_back(Rep{static_cast<uint8_t>(b), static_cast<int32_t>(o),
                                             static_cast<int32_t>(s), code, static_cast<uint8_t>(sb)});
                        if (sb == 2) nr.push_back((b * rows + o) * cols + s);
                    }
                    if (Hook::tail) {
                        std::vector<int32_t> &st = stamps[static_cast<size_t>(t)];
                        if (added) {
                            st.push_back(static_cast<int32_t>(ci + 1));
                            li = i;
                            last_new[static_cast<size_t>(t)] = 1;
                        } else if (st[static_cast<size_t>(local)] != static_cast<int32_t>(ci + 1)) {
                            st[static_cast<size_t>(local)] = static_cast<int32_t>(ci + 1);
                            li = i;
                            last_new[static_cast<size_t>(t)] = 0;
                        }
                    }
                    if (static_cast<uint32_t>(local) >= local_max ||
                        (limit > 0 && static_cast<int64_t>(local) >= limit)) {
                        overflow.store(true);
                        break;
                    }
                    mp[static_cast<size_t>(o) * static_cast<size_t>(cols) + static_cast<size_t>(s)] =
                        static_cast<int32_t>((static_cast<uint32_t>(t) << part_shift) |
                                             (static_cast<uint32_t>(local) << code_bits) | code);
                }
            }
            if (overflow.load()) return Outcome::Uncodeable;
            int64_t so_far = 0, n_near = 0, bytes = static_cast<int64_t>(chunk * (W * 8 + 8 + 2));
            int64_t tail_i = -1;
            bool tail_new = false;
            for (int t = 0; t < P; t++) {
                const auto &rp = reps[static_cast<size_t>(t)];
                const auto &tab = tables[static_cast<size_t>(t)];
                so_far += static_cast<int64_t>(rp.size());
                n_near += static_cast<int64_t>(nears[static_cast<size_t>(t)].size());
                bytes += static_cast<int64_t>(rp.capacity() * sizeof(Rep));
                bytes += static_cast<int64_t>(3 * tab.cap * (W * 8 + 4) / 2);
                if (last_i[static_cast<size_t>(t)] > tail_i) {
                    tail_i = last_i[static_cast<size_t>(t)];
                    tail_new = last_new[static_cast<size_t>(t)] != 0;
                }
            }
            if (!hook.done(ci, so_far, n_near, tail_new)) return Outcome::OverBudget;
            if (!pol.measured) {
                if (so_far > poor) return Outcome::Poor;
                continue;
            }
            bytes += so_far * CLASS_BYTES;
            peak = std::max(peak, bytes);
            if (so_far > poor) return Outcome::Poor;
            if (bytes > room) return Outcome::OverBudget;
        }
    }
    std::vector<Tab>().swap(tables);
    std::vector<std::vector<int32_t>>().swap(stamps);
    std::vector<int64_t> off(static_cast<size_t>(P + 1), 0);
    for (int t = 0; t < P; t++)
        off[static_cast<size_t>(t + 1)] = off[static_cast<size_t>(t)] + static_cast<int64_t>(reps[static_cast<size_t>(t)].size());
    const int64_t n_cls = off[static_cast<size_t>(P)];
    if (n_cls >= cls_max || (limit > 0 && n_cls > limit)) return Outcome::Uncodeable;
    if (n_cls > poor) return Outcome::Poor;
    const uint32_t code_mask = (uint32_t(1) << code_bits) - 1;
    const uint32_t local_mask = local_max - 1;
    for (int b = 0; b < n_blocks; b++) {
        int32_t *mp = f.map[b].data();
#pragma omp parallel for schedule(static)
        for (int64_t i = 0; i < static_cast<int64_t>(nn); i++) {
            const uint32_t e = static_cast<uint32_t>(mp[i]);
            const uint32_t t = pb == 0 ? 0 : e >> part_shift, local = (e >> code_bits) & local_mask;
            mp[i] = static_cast<int32_t>(((static_cast<uint32_t>(off[t]) + local) << code_bits) | (e & code_mask));
        }
    }
    std::vector<Rep> all;
    all.reserve(static_cast<size_t>(n_cls));
    for (int t = 0; t < P; t++)
        all.insert(all.end(), reps[static_cast<size_t>(t)].begin(), reps[static_cast<size_t>(t)].end());
    std::vector<std::vector<Rep>>().swap(reps);
    f.near_flat.clear();
    for (int t = 0; t < P; t++)
        f.near_flat.insert(f.near_flat.end(), nears[static_cast<size_t>(t)].begin(), nears[static_cast<size_t>(t)].end());
    std::vector<std::vector<int64_t>>().swap(nears);
    std::sort(f.near_flat.begin(), f.near_flat.end());
    if (spec.numbering == Numbering::FirstAppearance) first_appearance(f, all);
    int64_t singles = 0;
    for (int64_t v : n_single) singles += v;
    detail::finish(f, all, singles, pol);
    return Outcome::Ok;
}

template <int W, class Tab = detail::Table<W>, class KeyFn>
Outcome census(FamilyPlan &f, const Policy &pol, int64_t room, int64_t &peak, KeyFn key) {
    detail::NoHook none;
    return census<W, Tab>(f, pol, room, peak, key, none);
}

// THE GATHER's decode: rows `rows` (observer indices) of block `block`
// against every source, `visit(i, s, c, rel)` for row i (of n), source s, its
// class c and rel = its code XOR its representative's (0: the class value as
// it is). Rows run in parallel; `visit` writes the adapter's own layout and
// applies its transform for rel != 0.
template <class Visit>
void gather(const FamilyPlan &f, int block, const int64_t *rows, ptrdiff_t n, Visit visit) {
    if (block < 0 || block >= f.spec.n_blocks || f.map[block].empty())
        throw std::invalid_argument("pair_reuse: no such block");
    const ptrdiff_t cols = static_cast<ptrdiff_t>(f.spec.cols);
    for (ptrdiff_t i = 0; i < n; i++)
        if (rows[i] < 0 || rows[i] >= f.spec.rows)
            throw std::invalid_argument("pair_reuse: observer out of range");
    const int32_t *map = f.map[block].data();
    const uint8_t *rc = f.rep_code.data();
    const int cb = f.spec.code_bits;
    const int32_t mask = (int32_t(1) << cb) - 1;
#pragma omp parallel for schedule(static)
    for (ptrdiff_t i = 0; i < n; i++) {
        const int32_t *row = map + rows[i] * cols;
        for (ptrdiff_t s = 0; s < cols; s++) {
            const int32_t e = row[s];
            const int32_t c = e >> cb;
            visit(i, s, c, (e & mask) ^ rc[c]);
        }
    }
}

}  // namespace pair_reuse
}  // namespace mw
