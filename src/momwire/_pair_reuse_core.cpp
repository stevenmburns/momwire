// The basis-independent half of fill-pair reuse (momwire#2063): see
// `_pair_reuse_core.h`. Its own TU, and integers only: the evaluation groups,
// the byte accounting, the declines, the cache and the wire map. No value a
// fill computes passes through it (momwire#1187).
#include "_pair_reuse_core.h"

#include <chrono>
#include <cstdio>
#include <list>
#include <mutex>

namespace mw {
namespace pair_reuse {

namespace {

using detail::Rep;

// The groups the representatives are evaluated in: their observer rows in
// order, consecutive rows of one block joined while the joint call evaluates
// at most about twice the pairs it needs (plus 256, so small rows still
// join), and at most `rep_rows` rows.
void build_groups(const std::vector<Rep> &reps, int64_t n_rows_block, int64_t n_cols, int n_blocks,
                  int rep_rows, std::vector<Group> &groups, int64_t &evals) {
    const int64_t n_rows = n_blocks * n_rows_block;
    std::vector<int64_t> cnt(static_cast<size_t>(n_rows + 1), 0);
    for (const Rep &r : reps) cnt[static_cast<size_t>(r.block * n_rows_block + r.obs + 1)]++;
    for (int64_t i = 0; i < n_rows; i++) cnt[static_cast<size_t>(i + 1)] += cnt[static_cast<size_t>(i)];
    std::vector<int32_t> order(reps.size());
    {
        std::vector<int64_t> pos(cnt.begin(), cnt.end() - 1);
        for (size_t c = 0; c < reps.size(); c++)
            order[static_cast<size_t>(pos[static_cast<size_t>(reps[c].block * n_rows_block + reps[c].obs)]++)] =
                static_cast<int32_t>(c);
    }
    std::vector<int64_t> mark(static_cast<size_t>(n_cols), -1);
    int64_t stamp = 0;
    evals = 0;
    Group g;
    int64_t g_reps = 0, g_union = 0;
    auto flush = [&]() {
        if (g.obs.empty()) return;
        // The union, in ascending source order, and each class's place in it.
        std::vector<int32_t> src;
        for (int32_t o : g.obs) {
            const int64_t row = g.block * n_rows_block + o;
            for (int64_t q = cnt[static_cast<size_t>(row)]; q < cnt[static_cast<size_t>(row + 1)]; q++)
                src.push_back(reps[static_cast<size_t>(order[static_cast<size_t>(q)])].src);
        }
        std::sort(src.begin(), src.end());
        src.erase(std::unique(src.begin(), src.end()), src.end());
        std::vector<int32_t> col(static_cast<size_t>(n_cols), -1);
        for (size_t u = 0; u < src.size(); u++) col[static_cast<size_t>(src[u])] = static_cast<int32_t>(u);
        g.src = src;
        for (size_t oi = 0; oi < g.obs.size(); oi++) {
            const int64_t row = g.block * n_rows_block + g.obs[oi];
            for (int64_t q = cnt[static_cast<size_t>(row)]; q < cnt[static_cast<size_t>(row + 1)]; q++) {
                const int32_t c = order[static_cast<size_t>(q)];
                g.cls.push_back(c);
                g.at.push_back(static_cast<int32_t>(oi * src.size()) + col[static_cast<size_t>(reps[static_cast<size_t>(c)].src)]);
            }
        }
        evals += static_cast<int64_t>(g.obs.size() * g.src.size());
        groups.push_back(std::move(g));
        g = Group();
        g_reps = g_union = 0;
        stamp++;
    };
    for (int b = 0; b < n_blocks; b++) {
        for (int64_t o = 0; o < n_rows_block; o++) {
            const int64_t row = b * n_rows_block + o;
            const int64_t lo = cnt[static_cast<size_t>(row)], hi = cnt[static_cast<size_t>(row + 1)];
            if (lo == hi) continue;
            if (!g.obs.empty() && g.block != b) flush();
            // The sources this row would add to the union.
            int64_t add = 0;
            for (int64_t q = lo; q < hi; q++) {
                const int32_t s = reps[static_cast<size_t>(order[static_cast<size_t>(q)])].src;
                if (mark[static_cast<size_t>(s)] != stamp) add++;
            }
            const int64_t rows = static_cast<int64_t>(g.obs.size());
            if (rows > 0 && (rows + 1 > rep_rows ||
                             (rows + 1) * (g_union + add) > 2 * (g_reps + (hi - lo)) + 256)) {
                flush();
            }
            g.block = static_cast<uint8_t>(b);
            g.obs.push_back(static_cast<int32_t>(o));
            for (int64_t q = lo; q < hi; q++) {
                const int32_t s = reps[static_cast<size_t>(order[static_cast<size_t>(q)])].src;
                if (mark[static_cast<size_t>(s)] != stamp) {
                    mark[static_cast<size_t>(s)] = stamp;
                    g_union++;
                }
            }
            g_reps += hi - lo;
        }
    }
    flush();
}

struct Entry {
    std::string basis;
    std::shared_ptr<const Plan> plan;
};
std::mutex g_cache_mtx;
std::list<Entry> g_cache;  // most recent first
std::map<std::string, Counters> g_counters;

}  // namespace

namespace detail {

void finish(FamilyPlan &f, const std::vector<Rep> &reps, int64_t singles, const Policy &pol) {
    f.classes = static_cast<int64_t>(reps.size());
    f.pairs = static_cast<int64_t>(f.spec.n_blocks) * f.spec.rows * f.spec.cols;
    f.near = f.spec.near_singles ? singles : 0;
    f.rep_code.resize(reps.size());
    for (size_t c = 0; c < reps.size(); c++) f.rep_code[c] = reps[c].code;
    build_groups(reps, f.spec.rows, f.spec.cols, f.spec.n_blocks, pol.rep_rows, f.groups, f.rep_evals);
    // The fill report's split (momwire#2016): integers read off the census.
    if (f.spec.near_singles) f.near_bits.assign((reps.size() + 63) / 64, 0);
    for (size_t c = 0; c < reps.size(); c++) {
        f.cls_block[reps[c].block]++;
        if (f.spec.near_singles && reps[c].single) {
            f.near_block[reps[c].block]++;
            f.near_bits[c / 64] |= uint64_t(1) << (c % 64);
        }
    }
    for (const Group &G : f.groups)
        f.evals_block[G.block] += static_cast<int64_t>(G.obs.size() * G.src.size());
}

}  // namespace detail

size_t FamilyPlan::nbytes() const {
    size_t b = map[0].size() * 4 + map[1].size() * 4 + rep_code.size();
    for (const Group &g : groups) b += 4 * (g.obs.size() + g.src.size() + g.cls.size() + g.at.size());
    return b;
}

size_t Plan::nbytes() const {
    size_t b = fingerprint.size();
    for (const FamilyPlan &f : fam) b += f.nbytes();
    return b;
}

int64_t maps_bytes(const Plan &p) {
    int64_t b = 0;
    for (const FamilyPlan &f : p.fam) b += 4 * f.spec.rows * f.spec.cols * f.spec.n_blocks;
    return b;
}

int64_t held_bytes(const Plan &p) {
    int64_t b = static_cast<int64_t>(p.nbytes());
    for (const FamilyPlan &f : p.fam) b += f.spec.value_bytes * f.classes;
    return b;
}

std::string budget_reason(int64_t budget_bytes) {
    char buf[96];
    std::snprintf(buf, sizeof buf, "the plan and its census would exceed the %.6g MB budget",
                  static_cast<double>(budget_bytes) / double(1 << 20));
    return buf;
}

void decline(Plan &p, const std::string &why) {
    p.declined = why;
    for (FamilyPlan &f : p.fam) {
        const FamilySpec spec = f.spec;
        f = FamilyPlan();
        f.spec = spec;
    }
    p.budget_bytes = 0;
}

std::shared_ptr<const Plan> plan_for(const std::string &basis, std::string fingerprint,
                                     const std::function<void(Plan &)> &build, bool *built) {
    if (built) *built = false;
    {
        std::lock_guard<std::mutex> lock(g_cache_mtx);
        for (auto it = g_cache.begin(); it != g_cache.end(); ++it)
            if (it->basis == basis && it->plan->fingerprint == fingerprint) {
                Entry e = *it;
                g_cache.erase(it);
                g_cache.push_front(e);
                g_counters[basis].found++;
                return e.plan;
            }
    }
    auto plan = std::make_shared<Plan>();
    plan->basis = basis;
    plan->fingerprint = std::move(fingerprint);
    const auto t0 = std::chrono::steady_clock::now();
    build(*plan);
    plan->census_seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    if (built) *built = true;
    std::shared_ptr<const Plan> p = plan;
    std::lock_guard<std::mutex> lock(g_cache_mtx);
    Counters &c = g_counters[basis];
    c.built++;
    if (!p->declined.empty()) c.declined++;
    g_cache.push_front(Entry{basis, p});
    size_t total = 0;
    for (auto it = g_cache.begin(); it != g_cache.end();) {
        total += it->plan->nbytes();
        if (it != g_cache.begin() && total > SHARED_PLAN_BYTES)
            it = g_cache.erase(it);
        else
            ++it;
    }
    return p;
}

Counters counters(const std::string &basis) {
    std::lock_guard<std::mutex> lock(g_cache_mtx);
    auto it = g_counters.find(basis);
    return it == g_counters.end() ? Counters() : it->second;
}

void clear_cache(const std::string &basis) {
    std::lock_guard<std::mutex> lock(g_cache_mtx);
    for (auto it = g_cache.begin(); it != g_cache.end();)
        if (it->basis == basis)
            it = g_cache.erase(it);
        else
            ++it;
}

void wire_map(const FamilyPlan &f, const int32_t *wire_of_row, const int32_t *wire_of_col,
              int64_t n_wires, int64_t *out) {
    const int64_t rows = f.spec.rows, cols = f.spec.cols, W = n_wires;
    const int cb = f.spec.code_bits;
    const bool near = f.spec.near_singles && !f.near_bits.empty();
    // A class is computed at its first pair in the census's order (block by
    // block, row-major), which is the pair the census made its representative.
    std::vector<uint64_t> seen(static_cast<size_t>((f.classes + 63) / 64), 0);
    for (int b = 0; b < f.spec.n_blocks; b++) {
        if (f.map[b].empty()) continue;
        const int32_t *map = f.map[b].data();
        int64_t *blk = out + b * W * W * 3;
        for (int64_t o = 0; o < rows; o++) {
            int64_t *row = blk + static_cast<int64_t>(wire_of_row[o]) * W * 3;
            for (int64_t s = 0; s < cols; s++) {
                const uint64_t c = static_cast<uint64_t>(map[o * cols + s] >> cb);
                const uint64_t bit = uint64_t(1) << (c % 64);
                int64_t *cell = row + static_cast<int64_t>(wire_of_col[s]) * 3;
                if (near && (f.near_bits[c / 64] & bit)) {
                    cell[2]++;
                } else if (seen[c / 64] & bit) {
                    cell[1]++;
                } else {
                    seen[c / 64] |= bit;
                    cell[0]++;
                }
            }
        }
    }
}

}  // namespace pair_reuse
}  // namespace mw
