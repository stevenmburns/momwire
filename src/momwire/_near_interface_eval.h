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
#include <unordered_map>
#include <vector>

namespace ni_eval {

using cd = std::complex<double>;
constexpr int N_KEYS = 6;  // U, V, W, dzW, dzpV, dzpW
using Six = std::array<cd, N_KEYS>;

struct SheetPlan {
    std::vector<double> planes;           // sorted z' < 0
    std::vector<double> heights, depths;  // sorted z > 0, each one's reach
    bool any() const { return !planes.empty() || !heights.empty(); }
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
    int n_threads = 0;  // `_physical_cpu_count()`; values do not depend on it
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

// `_evaluate_fresh` over distinct rows (m, 3): (m, 6).
void evaluate_fresh(const Evaluator &ev, const double *rows, size_t m, const int64_t *labels,
                    const SheetPlan *plan, ni_core::ColumnCache *cache, Six *out);

// Python's repr of a float (the refusals' numbers).
std::string py_repr(double x);

}  // namespace ni_eval
