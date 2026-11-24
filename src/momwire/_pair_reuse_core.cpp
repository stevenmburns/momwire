// The basis-independent half of fill-pair reuse (momwire#2063): see
// `_pair_reuse_core.h`. Its own TU, and integers only: the evaluation groups,
// the byte accounting, the declines, the cache and the wire map. No value a
// fill computes passes through it (momwire#1187).
#include "_pair_reuse_core.h"

#include <chrono>
#include <type_traits>
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
int64_t g_cache_bytes = -1;  // the test hook's; < 0: SHARED_PLAN_BYTES
std::atomic<int64_t> g_cap_compression{CLASS_CAP_COMPRESSION};

}  // namespace

namespace detail {

void finish(FamilyPlan &f, const std::vector<Rep> &reps, int64_t singles, const Policy &pol) {
    f.classes = static_cast<int64_t>(reps.size());
    f.pairs = static_cast<int64_t>(f.spec.n_blocks) * f.spec.rows * f.spec.cols;
    f.near = (f.spec.near_singles ? singles : 0) + static_cast<int64_t>(f.near_flat.size());
    f.rep_code.resize(reps.size());
    for (size_t c = 0; c < reps.size(); c++) f.rep_code[c] = reps[c].code;
    if (f.spec.keep_reps) {
        f.rep_flat.resize(reps.size());
        for (size_t c = 0; c < reps.size(); c++)
            f.rep_flat[c] = (reps[c].block * f.spec.rows + reps[c].obs) * f.spec.cols + reps[c].src;
    }
    if (pol.rep_rows > 0)
        build_groups(reps, f.spec.rows, f.spec.cols, f.spec.n_blocks, pol.rep_rows, f.groups, f.rep_evals);
    // The fill report's split (momwire#2016): integers read off the census.
    // A near class is a single (`near_singles`) or a Key::Near class.
    const bool near_classes = f.spec.near_singles || !f.near_flat.empty();
    if (near_classes) f.near_bits.assign((reps.size() + 63) / 64, 0);
    for (size_t c = 0; c < reps.size(); c++) {
        f.cls_block[reps[c].block]++;
        if ((f.spec.near_singles && reps[c].single == 1) || reps[c].single == 2) {
            f.near_block[reps[c].block]++;
            f.near_bits[c / 64] |= uint64_t(1) << (c % 64);
        }
    }
    for (const Group &G : f.groups)
        f.evals_block[G.block] += static_cast<int64_t>(G.obs.size() * G.src.size());
}

void first_appearance(FamilyPlan &f, std::vector<Rep> &reps) {
    const int64_t rows = f.spec.rows, cols = f.spec.cols;
    auto flat = [&](const Rep &r) { return (r.block * rows + r.obs) * cols + r.src; };
    std::vector<int64_t> order(reps.size());
    for (size_t c = 0; c < reps.size(); c++) order[c] = static_cast<int64_t>(c);
    // Every class has its own first pair, so the order is strict.
    std::sort(order.begin(), order.end(), [&](int64_t a, int64_t b) {
        return flat(reps[static_cast<size_t>(a)]) < flat(reps[static_cast<size_t>(b)]);
    });
    bool moved = false;
    for (size_t j = 0; j < order.size(); j++) moved = moved || order[j] != static_cast<int64_t>(j);
    if (!moved) return;
    std::vector<int64_t> new_of_old(reps.size());
    for (size_t j = 0; j < order.size(); j++) new_of_old[static_cast<size_t>(order[j])] = static_cast<int64_t>(j);
    renumber(f, new_of_old);
    std::vector<Rep> sorted(reps.size());
    for (size_t j = 0; j < order.size(); j++) sorted[j] = reps[static_cast<size_t>(order[j])];
    reps.swap(sorted);
}

}  // namespace detail

void renumber(FamilyPlan &f, const std::vector<int64_t> &new_of_old) {
    if (!f.groups.empty()) throw std::invalid_argument("pair_reuse: a family with groups cannot be renumbered");
    const int cb = f.spec.code_bits;
    const int32_t mask = (int32_t(1) << cb) - 1;
    for (int b = 0; b < f.spec.n_blocks; b++) {
        int32_t *mp = f.map[b].data();
        const int64_t n = static_cast<int64_t>(f.map[b].size());
#pragma omp parallel for schedule(static)
        for (int64_t i = 0; i < n; i++) {
            const int32_t e = mp[i];
            mp[i] = static_cast<int32_t>((new_of_old[static_cast<size_t>(e >> cb)] << cb) | (e & mask));
        }
    }
    auto permute = [&](auto &v) {
        if (v.empty()) return;
        std::remove_reference_t<decltype(v)> out(v.size());
        for (size_t c = 0; c < v.size(); c++) out[static_cast<size_t>(new_of_old[c])] = v[c];
        v.swap(out);
    };
    permute(f.rep_code);
    permute(f.rep_flat);
    if (!f.near_bits.empty()) {
        std::vector<uint64_t> nb(f.near_bits.size(), 0);
        for (size_t c = 0; c < new_of_old.size(); c++)
            if (f.near_bits[c / 64] & (uint64_t(1) << (c % 64))) {
                const uint64_t d = static_cast<uint64_t>(new_of_old[c]);
                nb[d / 64] |= uint64_t(1) << (d % 64);
            }
        f.near_bits.swap(nb);
    }
}

BandBudget::BandBudget(int64_t budget, int64_t band_bytes, int workers, int64_t n_bands, int64_t held)
    : budget_(budget), band_bytes_(band_bytes), n_bands_(n_bands), held_(held), workers_(std::max(1, workers)) {}

bool BandBudget::start(int64_t i) {
    // The bands in flight are [i, next_): start more while they fit.
    while (next_ < n_bands_ && next_ - i < workers_ && fits(next_ - i + 1)) next_++;
    if (next_ <= i) return false;  // not one band fits beside what is held
    max_in_flight_ = std::max(max_in_flight_, next_ - i);
    peak_ = std::max(peak_, held_ + band_bytes_ * (next_ - i));
    cur_ = i;
    return true;
}

bool BandBudget::fold(int64_t held, int64_t floor) {
    held_ = held;
    peak_ = std::max(peak_, held_ + band_bytes_ * (next_ - cur_ - 1));
    return !(budget_ >= 0 && std::max(peak_, floor) > budget_);
}

size_t FamilyPlan::nbytes() const {
    size_t b = map[0].size() * 4 + map[1].size() * 4 + rep_code.size() + 8 * (near_flat.size() + rep_flat.size());
    for (const Group &g : groups) b += 4 * (g.obs.size() + g.src.size() + g.cls.size() + g.at.size());
    return b;
}

size_t Plan::nbytes() const {
    size_t b = fingerprint.size() + ext_bytes;
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

void set_cap_compression(int64_t c) { g_cap_compression.store(c > 0 ? c : CLASS_CAP_COMPRESSION); }
int64_t cap_compression() { return g_cap_compression.load(); }

std::string cap_reason(int64_t pairs) {
    char buf[128];
    std::snprintf(buf, sizeof buf, "more than %lld classes for %lld pairs (the cap is %lldx compression)",
                  static_cast<long long>(class_cap(pairs)), static_cast<long long>(pairs),
                  static_cast<long long>(cap_compression()));
    return buf;
}

std::string allowance_reason(int64_t allowance_bytes) {
    char buf[96];
    std::snprintf(buf, sizeof buf, "the kept plan would exceed its allowance of %.6g MB",
                  static_cast<double>(allowance_bytes) / double(1 << 20));
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

namespace {

// Drop the least recently used entries past the newest `keep` while the
// cache holds more than `limit` bytes (the caller holds the lock).
void trim(size_t limit, int keep) {
    size_t total = 0;
    int i = 0;
    for (auto it = g_cache.begin(); it != g_cache.end(); i++) {
        total += it->plan->nbytes();
        if (i >= keep && total > limit)
            it = g_cache.erase(it);
        else
            ++it;
    }
}

size_t cache_limit() { return g_cache_bytes < 0 ? SHARED_PLAN_BYTES : static_cast<size_t>(g_cache_bytes); }

}  // namespace

std::shared_ptr<const Plan> plan_for(const std::string &basis, std::string fingerprint,
                                     const std::function<void(Plan &)> &build, bool *built, int keep) {
    if (built) *built = false;
    keep = std::max(1, keep);
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
        // A census never runs beside more than the cap of older plans, the
        // ones this fill has just used (the newest keep - 1) excepted.
        trim(cache_limit(), keep - 1);
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
    if (g_cache_bytes == 0) return p;
    g_cache.push_front(Entry{basis, p});
    trim(cache_limit(), keep);
    return p;
}

Counters counters(const std::string &basis) {
    std::lock_guard<std::mutex> lock(g_cache_mtx);
    auto it = g_counters.find(basis);
    return it == g_counters.end() ? Counters() : it->second;
}

void reset_counters(const std::string &basis) {
    std::lock_guard<std::mutex> lock(g_cache_mtx);
    g_counters.erase(basis);
}

void set_cache_bytes(int64_t bytes) {
    std::lock_guard<std::mutex> lock(g_cache_mtx);
    g_cache_bytes = bytes < 0 ? -1 : bytes;
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
    // near bits are kept for near singles and for a Key::Near class
    const bool near = !f.near_bits.empty();
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
