// razor-2p's fill-pair reuse (momwire#1971): see `_razor_pair_reuse_core.h`.
// Its own TU, so no kernel's codegen moves with it (momwire#1187): it only
// CALLS the shipped kernels (`seg_moments`, `field_proj_batch`,
// `moment_reduce`), across a TU boundary.
#include "_razor_pair_reuse_core.h"

#include "_razor_core.h"
#include "_somm_proj_core.h"

#include <atomic>
#include <chrono>
#include <cstring>
#include <list>
#include <map>
#include <mutex>
#include <utility>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace mw {
namespace razor {
namespace reuse {

namespace {

std::atomic<int> g_probe{PROBE_NONE};

// Rows of the census computed per pass (its key buffer is rows * n_seg keys).
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
std::atomic<int64_t> g_class_limit[2] = {{0}, {0}};

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

inline uint64_t bits(double x) {
    uint64_t u;
    std::memcpy(&u, &x, sizeof u);
    return u;
}

// The quantum's exponent and its inverse step: 2^(KEY_BITS - ilogb(D)).
inline void quantum(double D, int &e, double &inv) {
    e = std::ilogb(D);
    inv = std::ldexp(1.0, KEY_BITS - e);
}

// A quantised value as a key word; false when it does not fit (the pair is
// then left unshared).
inline bool qword(double x, double inv, uint64_t &w) {
    const double y = x * inv;
    if (!(std::fabs(y) < 4.0e18)) return false;
    w = static_cast<uint64_t>(static_cast<int64_t>(std::llround(y)));
    return true;
}

struct Rep {
    uint8_t block = 0;
    int32_t obs = 0, src = 0;
    uint8_t code = 0;
};

int n_parts() {
#ifdef _OPENMP
    const int p = omp_get_max_threads();
#else
    const int p = 1;
#endif
    return std::max(1, std::min(MAX_PARTS, p));
}

// Distinct values, numbered in first appearance.
struct Ids {
    std::map<std::pair<uint64_t, uint64_t>, int64_t> m;
    int64_t id(uint64_t a, uint64_t b = 0) {
        auto it = m.emplace(std::make_pair(a, b), static_cast<int64_t>(m.size()));
        return it.first->second;
    }
};

// The groups the representatives are evaluated in: their observer rows in
// order, consecutive rows of one block joined while the joint call evaluates
// at most about twice the pairs it needs.
void build_groups(const std::vector<Rep> &reps, int64_t n_seg, int n_blocks,
                  std::vector<Group> &groups, int64_t &evals) {
    const int64_t n_rows = n_blocks * n_seg;
    std::vector<int64_t> cnt(static_cast<size_t>(n_rows + 1), 0);
    for (const Rep &r : reps) cnt[static_cast<size_t>(r.block * n_seg + r.obs + 1)]++;
    for (int64_t i = 0; i < n_rows; i++) cnt[static_cast<size_t>(i + 1)] += cnt[static_cast<size_t>(i)];
    std::vector<int32_t> order(reps.size());
    {
        std::vector<int64_t> pos(cnt.begin(), cnt.end() - 1);
        for (size_t c = 0; c < reps.size(); c++)
            order[static_cast<size_t>(pos[static_cast<size_t>(reps[c].block * n_seg + reps[c].obs)]++)] =
                static_cast<int32_t>(c);
    }
    std::vector<int64_t> mark(static_cast<size_t>(n_seg), -1);
    int64_t stamp = 0;
    evals = 0;
    Group g;
    int64_t g_reps = 0, g_union = 0;
    auto flush = [&]() {
        if (g.obs.empty()) return;
        // The union, in ascending source order, and each class's place in it.
        std::vector<int32_t> src;
        for (int32_t o : g.obs) {
            const int64_t row = g.block * n_seg + o;
            for (int64_t q = cnt[static_cast<size_t>(row)]; q < cnt[static_cast<size_t>(row + 1)]; q++)
                src.push_back(reps[static_cast<size_t>(order[static_cast<size_t>(q)])].src);
        }
        std::sort(src.begin(), src.end());
        src.erase(std::unique(src.begin(), src.end()), src.end());
        std::vector<int32_t> col(static_cast<size_t>(n_seg), -1);
        for (size_t u = 0; u < src.size(); u++) col[static_cast<size_t>(src[u])] = static_cast<int32_t>(u);
        g.src = src;
        for (size_t oi = 0; oi < g.obs.size(); oi++) {
            const int64_t row = g.block * n_seg + g.obs[oi];
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
        for (int64_t o = 0; o < n_seg; o++) {
            const int64_t row = b * n_seg + o;
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
            if (rows > 0 && (rows + 1 > REP_ROWS ||
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

// The census of one family (the direct moments, W = 4, or the remainder,
// W = 9): `key(block, o, s, words, code)` writes a pair's key and code and
// returns false for a pair that is its own class. Entries go to `maps[b]` as
// class << code_bits | code; reps are each class's first pair, row-major.
// Returns the number of classes, or -1 when they cannot be coded (past the
// int32 entry, a part's share of it, or the test hook's `class_limit`): the
// plan then declines.
template <int W, class KeyFn>
int64_t census(int64_t n_seg, int n_blocks, int code_bits, int64_t limit, KeyFn key,
               std::vector<int32_t> *maps, std::vector<Rep> &reps_out, int64_t &singles) {
    const int P = n_parts();
    const size_t nn = static_cast<size_t>(n_seg) * static_cast<size_t>(n_seg);
    for (int b = 0; b < n_blocks; b++) maps[b].assign(nn, 0);
    std::vector<Table<W>> tables(static_cast<size_t>(P));
    std::vector<std::vector<Rep>> reps(static_cast<size_t>(P));
    const ptrdiff_t R = std::min<ptrdiff_t>(CENSUS_ROWS, n_seg);
    const size_t chunk = static_cast<size_t>(R) * static_cast<size_t>(n_seg);
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
    for (int b = 0; b < n_blocks; b++) {
        for (int64_t o0 = 0; o0 < n_seg; o0 += R) {
            const int64_t o1 = std::min<int64_t>(n_seg, o0 + R);
            const int64_t m = (o1 - o0) * n_seg;
#pragma omp parallel for schedule(static)
            for (int64_t i = 0; i < m; i++) {
                const int64_t o = o0 + i / n_seg, s = i % n_seg;
                uint64_t *w = &kbuf[static_cast<size_t>(i) * W];
                uint8_t code = 0;
                const bool shared = key(b, o, s, w, code);
                cbuf[static_cast<size_t>(i)] = code;
                sbuf[static_cast<size_t>(i)] = shared ? 0 : 1;
                hbuf[static_cast<size_t>(i)] = shared ? hash_key<W>(w) : mix64(static_cast<uint64_t>(o));
            }
#pragma omp parallel for schedule(static, 1)
            for (int t = 0; t < P; t++) {
                Table<W> &tab = tables[static_cast<size_t>(t)];
                std::vector<Rep> &rp = reps[static_cast<size_t>(t)];
                for (int64_t i = 0; i < m; i++) {
                    const uint64_t h = hbuf[static_cast<size_t>(i)];
                    if (static_cast<int>((h >> 40) % static_cast<uint64_t>(P)) != t) continue;
                    const int64_t o = o0 + i / n_seg, s = i % n_seg;
                    const uint8_t code = cbuf[static_cast<size_t>(i)];
                    int32_t local;
                    if (sbuf[static_cast<size_t>(i)]) {
                        local = static_cast<int32_t>(rp.size());
                        rp.push_back(Rep{static_cast<uint8_t>(b), static_cast<int32_t>(o),
                                         static_cast<int32_t>(s), 0});
                        n_single[static_cast<size_t>(t)]++;
                    } else {
                        bool added = false;
                        local = tab.find_or_add(&kbuf[static_cast<size_t>(i) * W], h,
                                                static_cast<int32_t>(rp.size()), added);
                        if (added)
                            rp.push_back(Rep{static_cast<uint8_t>(b), static_cast<int32_t>(o),
                                             static_cast<int32_t>(s), code});
                    }
                    if (static_cast<uint32_t>(local) >= local_max ||
                        (limit > 0 && static_cast<int64_t>(local) >= limit)) {
                        overflow.store(true);
                        break;
                    }
                    maps[b][static_cast<size_t>(o) * static_cast<size_t>(n_seg) + static_cast<size_t>(s)] =
                        static_cast<int32_t>((static_cast<uint32_t>(t) << part_shift) |
                                             (static_cast<uint32_t>(local) << code_bits) | code);
                }
            }
            if (overflow.load()) return -1;
        }
    }
    std::vector<int64_t> off(static_cast<size_t>(P + 1), 0);
    for (int t = 0; t < P; t++) off[static_cast<size_t>(t + 1)] = off[static_cast<size_t>(t)] + static_cast<int64_t>(reps[static_cast<size_t>(t)].size());
    const int64_t n_cls = off[static_cast<size_t>(P)];
    if (n_cls >= cls_max || (limit > 0 && n_cls > limit)) return -1;
    const uint32_t code_mask = (uint32_t(1) << code_bits) - 1;
    const uint32_t local_mask = local_max - 1;
    for (int b = 0; b < n_blocks; b++) {
        int32_t *mp = maps[b].data();
#pragma omp parallel for schedule(static)
        for (int64_t i = 0; i < static_cast<int64_t>(nn); i++) {
            const uint32_t e = static_cast<uint32_t>(mp[i]);
            const uint32_t t = pb == 0 ? 0 : e >> part_shift, local = (e >> code_bits) & local_mask;
            mp[i] = static_cast<int32_t>(((static_cast<uint32_t>(off[t]) + local) << code_bits) | (e & code_mask));
        }
    }
    reps_out.clear();
    reps_out.reserve(static_cast<size_t>(n_cls));
    for (int t = 0; t < P; t++)
        reps_out.insert(reps_out.end(), reps[static_cast<size_t>(t)].begin(), reps[static_cast<size_t>(t)].end());
    singles = 0;
    for (int64_t v : n_single) singles += v;
    return n_cls;
}

// A plan that does not serve: no classes, its reason, its census time.
std::shared_ptr<const Plan> decline(std::shared_ptr<Plan> &plan, const char *why) {
    plan->declined = why;
    for (auto &m : plan->dmap) std::vector<int32_t>().swap(m);
    std::vector<int32_t>().swap(plan->qmap);
    std::vector<uint8_t>().swap(plan->drep_flip);
    std::vector<uint8_t>().swap(plan->qrep_code);
    std::vector<Group>().swap(plan->dgroups);
    std::vector<Group>().swap(plan->qgroups);
    plan->n_dcls = plan->n_qcls = 0;
    plan->pairs = plan->near_pairs = plan->q_pairs = plan->rep_evals = plan->q_rep_evals = 0;
    return plan;
}

std::shared_ptr<const Plan> build(const Geometry &g, std::string fingerprint) {
    const auto t0 = std::chrono::steady_clock::now();
    auto plan = std::make_shared<Plan>();
    const int64_t n_seg = g.seg_h.shape(0);
    plan->n_seg = n_seg;
    plan->has_image = g.has_image;
    plan->remainder = g.remainder;
    plan->fingerprint = std::move(fingerprint);
    if (n_seg > MAX_SEGS) return decline(plan, "too many segments to plan");
    const int n_blocks = g.has_image ? 2 : 1;
    const bool ek = g.group_i.size() > 0 && g.group_j[0].size() > 0;
    const bool has_b = ek && g.b_obs.size() > 0;
    const int pr = probe();

    // Types: the source's kernel radius, and under the EK its EK radius; the
    // observer's tube.
    Ids a_ids, ae_ids, b_ids;
    std::vector<int64_t> a_id[2], ae_id[2], b_id(static_cast<size_t>(n_seg), 0);
    for (int b = 0; b < n_blocks; b++) {
        a_id[b].resize(static_cast<size_t>(n_seg));
        ae_id[b].resize(static_cast<size_t>(n_seg));
        for (int64_t s = 0; s < n_seg; s++) {
            a_id[b][static_cast<size_t>(s)] = a_ids.id(bits(g.a(s)));
            ae_id[b][static_cast<size_t>(s)] =
                ek ? ae_ids.id(bits(g.a(s)), bits(g.a_ek[b](s))) : 0;
        }
    }
    if (has_b)
        for (int64_t o = 0; o < n_seg; o++) b_id[static_cast<size_t>(o)] = b_ids.id(bits(g.b_obs(o)));
    if (a_ids.m.size() >= (size_t(1) << 25) || ae_ids.m.size() >= (size_t(1) << 25) ||
        b_ids.m.size() >= (size_t(1) << 25))
        return decline(plan, "too many segment types to key");

    auto direct_key = [&](int b, int64_t o, int64_t s, uint64_t *w, uint8_t &code) -> bool {
        // The axis frame, in `seg_moments`' own spelling (momwire#799).
        const double dx = g.cent(o, 0) - g.p0[b](s, 0);
        const double dy = g.cent(o, 1) - g.p0[b](s, 1);
        const double dz = g.cent(o, 2) - g.p0[b](s, 2);
        const double tx = g.t[b](s, 0), ty = g.t[b](s, 1), tz = g.t[b](s, 2);
        const double u_r = dx * tx + dy * ty + dz * tz;
        const double px = dx - u_r * tx, py = dy - u_r * ty, pz = dz - u_r * tz;
        const double perp = px * px + py * py + pz * pz;
        const double h = g.seg_h(s);
        const double v = u_r - 0.5 * h;
        const double av = std::fabs(v);
        const double rho = std::sqrt(perp);
        code = 0;
        // Near in units of the kernel's own scale, the longer of h and the
        // radius: where h < a the values vary over a, not h (`NEAR_LENGTHS`).
        const double near = pr == PROBE_NEAR_H ? h : std::max(h, g.a(s));
        if (!(std::max(av, rho) >= NEAR_LENGTHS * near)) return false;
        const double D = std::max(std::max(av - 0.5 * h, rho), g.a(s));
        if (!(D > 0.0) || !std::isfinite(D)) return false;
        int e;
        double inv;
        quantum(D, e, inv);
        if (!qword(pr == PROBE_NO_FOLD ? v : av, inv, w[0]) || !qword(rho, inv, w[1]) ||
            !qword(h, inv, w[2]))
            return false;
        if (v < 0.0 && w[0] != 0 && pr != PROBE_NO_FLIP && pr != PROBE_NO_FOLD) code = 1;
        const bool elig = ek && g.group_i(o) >= 0 && g.group_i(o) == g.group_j[b](s);
        const uint64_t typ = static_cast<uint64_t>(elig ? ae_id[b][static_cast<size_t>(s)]
                                                        : a_id[b][static_cast<size_t>(s)]);
        const uint64_t bt = elig ? static_cast<uint64_t>(b_id[static_cast<size_t>(o)]) : 0;
        w[3] = static_cast<uint64_t>(e + 1100) | (uint64_t(elig) << 11) | (typ << 12) | (bt << 37);
        if (pr == PROBE_NO_TYPE) w[3] = static_cast<uint64_t>(e + 1100);
        return true;
    };
    std::vector<Rep> reps;
    int64_t singles = 0;
    plan->n_dcls = census<4>(n_seg, n_blocks, 1, class_limit(0), direct_key, plan->dmap, reps, singles);
    if (plan->n_dcls < 0) return decline(plan, "too many classes to code");
    plan->near_pairs = singles;
    plan->pairs = n_blocks * n_seg * n_seg;
    plan->drep_flip.resize(reps.size());
    for (size_t c = 0; c < reps.size(); c++) plan->drep_flip[c] = reps[c].code;
    build_groups(reps, n_seg, n_blocks, plan->dgroups, plan->rep_evals);

    if (g.remainder) {
        const double gz = g.ground_z;
        const double one = std::ldexp(1.0, KEY_BITS);
        auto q_key = [&](int, int64_t o, int64_t s, uint64_t *w, uint8_t &code) -> bool {
            const double zo = g.cent(o, 2) - gz, zs = g.cent(s, 2) - gz;
            const double hs = zo + zs;
            const double dx = g.cent(o, 0) - g.cent(s, 0), dy = g.cent(o, 1) - g.cent(s, 1);
            const double dd2 = dx * dx + dy * dy;
            const double ax = g.t[0](s, 0), ay = g.t[0](s, 1), tsz = g.t[0](s, 2);
            const double bx = g.t[0](o, 0), by = g.t[0](o, 1), toz = g.t[0](o, 2);
            const double h = g.seg_h(s);
            const double D = std::max(std::sqrt(dd2 + hs * hs), h);
            if (!(D > 0.0) || !std::isfinite(D)) return false;
            int e;
            double inv;
            quantum(D, e, inv);
            uint64_t qhs, qdd, qh, qda, qdb, qab, qtsz, qtoz;
            if (!qword(hs, inv, qhs) || !qword(std::sqrt(dd2), inv, qdd) || !qword(h, inv, qh) ||
                !qword(dx * ax + dy * ay, inv, qda) || !qword(dx * bx + dy * by, inv, qdb) ||
                !qword(ax * bx + ay * by, one, qab) || !qword(tsz, one, qtsz) ||
                !qword(toz, one, qtoz))
                return false;
            // The orientation whose odd columns are largest, lexicographically.
            auto neg = [](uint64_t x) { return static_cast<uint64_t>(-static_cast<int64_t>(x)); };
            int64_t best[5] = {0, 0, 0, 0, 0};
            int best_c = -1;
            for (int c = 0; c < 4; c++) {
                const bool sf = c & 1, of = (c & 2) != 0;
                const uint64_t cand[5] = {sf ? neg(qda) : qda, of ? neg(qdb) : qdb,
                                          (sf != of) ? neg(qab) : qab, sf ? neg(qtsz) : qtsz,
                                          of ? neg(qtoz) : qtoz};
                int64_t ci[5];
                for (int j = 0; j < 5; j++) ci[j] = static_cast<int64_t>(cand[j]);
                if (best_c < 0 || std::lexicographical_compare(best, best + 5, ci, ci + 5)) {
                    std::memcpy(best, ci, sizeof best);
                    best_c = c;
                }
            }
            code = pr == PROBE_Q_NO_FLIP ? 0 : static_cast<uint8_t>(best_c);
            w[0] = static_cast<uint64_t>(e + 1100);
            w[1] = pr == PROBE_NO_HEIGHT ? 0 : qhs;
            w[2] = qdd;
            w[3] = qh;
            for (int j = 0; j < 5; j++) w[4 + j] = static_cast<uint64_t>(best[j]);
            return true;
        };
        std::vector<Rep> qreps;
        int64_t qsingles = 0;
        plan->n_qcls = census<9>(n_seg, 1, 2, class_limit(1), q_key, &plan->qmap, qreps, qsingles);
        if (plan->n_qcls < 0) return decline(plan, "too many remainder classes to code");
        plan->q_pairs = n_seg * n_seg;
        plan->qrep_code.resize(qreps.size());
        for (size_t c = 0; c < qreps.size(); c++) plan->qrep_code[c] = qreps[c].code;
        build_groups(qreps, n_seg, 1, plan->qgroups, plan->q_rep_evals);
    }
    plan->census_seconds =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    return plan;
}

template <class T>
void put(std::string &s, const T &v) {
    s.append(reinterpret_cast<const char *>(&v), sizeof v);
}

std::string fingerprint_of(const Geometry &g) {
    std::string s;
    const int64_t n = g.seg_h.shape(0);
    put(s, n);
    put(s, static_cast<int32_t>(KEY_BITS));
    put(s, static_cast<int32_t>(probe()));
    put(s, class_limit(0));
    put(s, class_limit(1));
    put(s, g.has_image);
    put(s, g.remainder);
    put(s, g.ground_z);
    put(s, g.group_i.size());
    put(s, g.b_obs.size());
    for (int64_t i = 0; i < n; i++) {
        for (int c = 0; c < 3; c++) put(s, g.cent(i, c));
        put(s, g.seg_h(i));
        put(s, g.a(i));
        if (g.group_i.size() > 0) put(s, g.group_i(i));
        if (g.b_obs.size() > 0) put(s, g.b_obs(i));
    }
    for (int b = 0; b < (g.has_image ? 2 : 1); b++) {
        put(s, g.group_j[b].size());
        put(s, g.a_ek[b].size());
        for (int64_t i = 0; i < n; i++) {
            for (int c = 0; c < 3; c++) put(s, g.p0[b](i, c));
            for (int c = 0; c < 3; c++) put(s, g.t[b](i, c));
            if (g.group_j[b].size() > 0) put(s, g.group_j[b](i));
            if (g.a_ek[b].size() > 0) put(s, g.a_ek[b](i));
        }
    }
    return s;
}

std::mutex g_cache_mtx;
std::list<std::shared_ptr<const Plan>> g_cache;  // most recent first
CacheCounters g_counters;

void check_geometry(const Geometry &g) {
    const ptrdiff_t n = g.seg_h.shape(0);
    auto bad = [](const char *what) {
        throw std::invalid_argument(std::string("razor pair_reuse: ") + what + " is mis-shaped");
    };
    if (g.cent.shape(0) != n || g.cent.shape(1) != 3) bad("cent");
    if (g.a.shape(0) != n) bad("a");
    if (g.group_i.size() > 0 && g.group_i.shape(0) != n) bad("group_i");
    if (g.b_obs.size() > 0 && g.b_obs.shape(0) != n) bad("b_obs");
    for (int b = 0; b < (g.has_image ? 2 : 1); b++) {
        if (g.p0[b].shape(0) != n || g.p0[b].shape(1) != 3) bad("seg_p0");
        if (g.t[b].shape(0) != n || g.t[b].shape(1) != 3) bad("seg_t");
        if ((g.group_i.size() > 0) != (g.group_j[b].size() > 0)) bad("group_j");
        if (g.group_j[b].size() > 0 && (g.group_j[b].shape(0) != n || g.a_ek[b].shape(0) != n))
            bad("group_j / a_ek");
    }
}

}  // namespace

size_t Plan::nbytes() const {
    size_t b = fingerprint.size() + dmap[0].size() * 4 + dmap[1].size() * 4 + qmap.size() * 4 +
               drep_flip.size() + qrep_code.size();
    for (const auto *gs : {&dgroups, &qgroups})
        for (const Group &g : *gs) b += 4 * (g.obs.size() + g.src.size() + g.cls.size() + g.at.size());
    return b;
}

std::shared_ptr<const Plan> plan_for(const Geometry &g) {
    check_geometry(g);
    std::string fp = fingerprint_of(g);
    {
        std::lock_guard<std::mutex> lock(g_cache_mtx);
        for (auto it = g_cache.begin(); it != g_cache.end(); ++it)
            if ((*it)->fingerprint == fp) {
                std::shared_ptr<const Plan> p = *it;
                g_cache.erase(it);
                g_cache.push_front(p);
                g_counters.found++;
                return p;
            }
    }
    std::shared_ptr<const Plan> p = build(g, std::move(fp));
    std::lock_guard<std::mutex> lock(g_cache_mtx);
    g_counters.built++;
    g_cache.push_front(p);
    size_t total = 0;
    for (auto it = g_cache.begin(); it != g_cache.end();) {
        total += (*it)->nbytes();
        if (it != g_cache.begin() && total > SHARED_PLAN_BYTES)
            it = g_cache.erase(it);
        else
            ++it;
    }
    return p;
}

void set_probe(int p) { g_probe.store(p); }

namespace {
std::atomic<int64_t> g_small_free{SMALL_DECK_SEGS_FREE}, g_small_ground{SMALL_DECK_SEGS_GROUND};
}  // namespace
int64_t small_deck_segs(bool finite) { return finite ? g_small_ground.load() : g_small_free.load(); }
void set_small_deck_segs(int64_t free_space, int64_t finite_ground) {
    g_small_free.store(free_space < 0 ? SMALL_DECK_SEGS_FREE : free_space);
    g_small_ground.store(finite_ground < 0 ? SMALL_DECK_SEGS_GROUND : finite_ground);
}
void set_class_limit(int64_t direct, int64_t remainder) {
    g_class_limit[0].store(direct < 0 ? 0 : direct);
    g_class_limit[1].store(remainder < 0 ? 0 : remainder);
}
int64_t class_limit(int family) { return g_class_limit[family == 0 ? 0 : 1].load(); }
int probe() { return g_probe.load(); }

CacheCounters cache_counters() {
    std::lock_guard<std::mutex> lock(g_cache_mtx);
    return g_counters;
}

void clear_cache() {
    std::lock_guard<std::mutex> lock(g_cache_mtx);
    g_cache.clear();
}

void eval_direct(const Plan &p, const Geometry &g, view<const double, 1> xg,
                 view<const double, 1> wg, double k, uintptr_t cancel_flag, Values &out) {
    out.k = k;
    out.d0.assign(static_cast<size_t>(p.n_dcls), cd(0.0, 0.0));
    out.d1.assign(static_cast<size_t>(p.n_dcls), cd(0.0, 0.0));
    const bool ek = g.group_i.size() > 0 && g.group_j[0].size() > 0;
    const bool has_b = ek && g.b_obs.size() > 0;
    std::vector<double> obs, p0, t, h, a, aek, bo;
    std::vector<int64_t> gi, gj;
    std::vector<cd> m0, m1;
    for (const Group &G : p.dgroups) {
        const int b = G.block;
        const ptrdiff_t no = static_cast<ptrdiff_t>(G.obs.size()), ns = static_cast<ptrdiff_t>(G.src.size());
        obs.resize(static_cast<size_t>(3 * no));
        gi.resize(static_cast<size_t>(ek ? no : 0));
        bo.resize(static_cast<size_t>(has_b ? no : 0));
        for (ptrdiff_t i = 0; i < no; i++) {
            const int64_t o = G.obs[static_cast<size_t>(i)];
            for (int c = 0; c < 3; c++) obs[static_cast<size_t>(3 * i + c)] = g.cent(o, c);
            if (ek) gi[static_cast<size_t>(i)] = g.group_i(o);
            if (has_b) bo[static_cast<size_t>(i)] = g.b_obs(o);
        }
        p0.resize(static_cast<size_t>(3 * ns));
        t.resize(static_cast<size_t>(3 * ns));
        h.resize(static_cast<size_t>(ns));
        a.resize(static_cast<size_t>(ns));
        gj.resize(static_cast<size_t>(ek ? ns : 0));
        aek.resize(static_cast<size_t>(ek ? ns : 0));
        for (ptrdiff_t j = 0; j < ns; j++) {
            const int64_t s = G.src[static_cast<size_t>(j)];
            for (int c = 0; c < 3; c++) {
                p0[static_cast<size_t>(3 * j + c)] = g.p0[b](s, c);
                t[static_cast<size_t>(3 * j + c)] = g.t[b](s, c);
            }
            h[static_cast<size_t>(j)] = g.seg_h(s);
            a[static_cast<size_t>(j)] = g.a(s);
            if (ek) {
                gj[static_cast<size_t>(j)] = g.group_j[b](s);
                aek[static_cast<size_t>(j)] = g.a_ek[b](s);
            }
        }
        m0.resize(static_cast<size_t>(no * ns));
        m1.resize(static_cast<size_t>(no * ns));
        seg_moments(view<const double, 2>::contiguous(obs.data(), {no, 3}),
                    view<const double, 2>::contiguous(p0.data(), {ns, 3}),
                    view<const double, 2>::contiguous(t.data(), {ns, 3}),
                    view<const double, 1>::contiguous(h.data(), {ns}),
                    view<const double, 1>::contiguous(a.data(), {ns}), xg, wg, k, true,
                    view<const int64_t, 1>::contiguous(gi.data(), {static_cast<ptrdiff_t>(gi.size())}),
                    view<const int64_t, 1>::contiguous(gj.data(), {static_cast<ptrdiff_t>(gj.size())}),
                    view<const double, 1>::contiguous(aek.data(), {static_cast<ptrdiff_t>(aek.size())}),
                    cancel_flag,
                    view<const double, 1>::contiguous(bo.data(), {static_cast<ptrdiff_t>(bo.size())}),
                    view<cd, 2>::contiguous(m0.data(), {no, ns}),
                    view<cd, 2>::contiguous(m1.data(), {no, ns}));
        for (size_t q = 0; q < G.cls.size(); q++) {
            out.d0[static_cast<size_t>(G.cls[q])] = m0[static_cast<size_t>(G.at[q])];
            out.d1[static_cast<size_t>(G.cls[q])] = m1[static_cast<size_t>(G.at[q])];
        }
    }
    out.has_d = true;
}

void eval_remainder(const Plan &p, const Geometry &g, const somm_proj::GridView &grid,
                    size_t n_reg, const double *rem_src, const double *rem_t_src,
                    const double *W, ptrdiff_t n_qp, double k, uintptr_t cancel_flag,
                    Values &out) {
    if (!p.remainder) throw std::invalid_argument("razor pair_reuse: the plan has no remainder");
    out.q0.assign(static_cast<size_t>(p.n_qcls), cd(0.0, 0.0));
    out.q1.assign(static_cast<size_t>(p.n_qcls), cd(0.0, 0.0));
    const ptrdiff_t n_seg = static_cast<ptrdiff_t>(p.n_seg), q = n_qp;
    std::vector<double> pts, tans, src, tsrc, Wg;
    std::vector<cd> proj, mom;
    for (size_t gi = 0; gi < p.qgroups.size(); gi++) {
        const Group &G = p.qgroups[gi];
        const ptrdiff_t no = static_cast<ptrdiff_t>(G.obs.size()), ns = static_cast<ptrdiff_t>(G.src.size());
        pts.resize(static_cast<size_t>(3 * no));
        tans.resize(static_cast<size_t>(3 * no));
        for (ptrdiff_t i = 0; i < no; i++) {
            const int64_t o = G.obs[static_cast<size_t>(i)];
            for (int c = 0; c < 3; c++) {
                pts[static_cast<size_t>(3 * i + c)] = g.cent(o, c);
                tans[static_cast<size_t>(3 * i + c)] = g.t[0](o, c);
            }
        }
        src.resize(static_cast<size_t>(3 * ns * q));
        tsrc.resize(static_cast<size_t>(3 * ns * q));
        Wg.resize(static_cast<size_t>(2 * ns * q));
        for (ptrdiff_t j = 0; j < ns; j++) {
            const ptrdiff_t s = G.src[static_cast<size_t>(j)];
            std::memcpy(&src[static_cast<size_t>(3 * j * q)], rem_src + 3 * s * q, sizeof(double) * static_cast<size_t>(3 * q));
            std::memcpy(&tsrc[static_cast<size_t>(3 * j * q)], rem_t_src + 3 * s * q, sizeof(double) * static_cast<size_t>(3 * q));
            for (int pp = 0; pp < 2; pp++)
                std::memcpy(&Wg[static_cast<size_t>((pp * ns + j) * q)], W + (pp * n_seg + s) * q,
                            sizeof(double) * static_cast<size_t>(q));
        }
        proj.resize(static_cast<size_t>(no * ns * q));
        somm_core::field_proj_batch(grid, n_reg, no, ns * q, pts.data(), tans.data(), src.data(),
                                    tsrc.data(), g.ground_z, k, proj.data(), cancel_flag, true);
        mom.resize(static_cast<size_t>(no * ns * 2));
        somm_core::moment_reduce(proj.data(), Wg.data(), 2, ns, q, no, mom.data());
        store_remainder(p, gi, mom.data(), out);
    }
    out.k = k;
    out.has_q = true;
}

void store_remainder(const Plan &p, size_t group, const cd *mom, Values &out) {
    if (group >= p.qgroups.size()) throw std::invalid_argument("razor pair_reuse: no such group");
    if (out.q0.size() != static_cast<size_t>(p.n_qcls)) {
        out.q0.assign(static_cast<size_t>(p.n_qcls), cd(0.0, 0.0));
        out.q1.assign(static_cast<size_t>(p.n_qcls), cd(0.0, 0.0));
    }
    const Group &G = p.qgroups[group];
    for (size_t i = 0; i < G.cls.size(); i++) {
        const size_t at = static_cast<size_t>(G.at[i]);
        out.q0[static_cast<size_t>(G.cls[i])] = mom[2 * at];
        out.q1[static_cast<size_t>(G.cls[i])] = mom[2 * at + 1];
    }
    if (group + 1 == p.qgroups.size()) out.has_q = true;
}

void gather_direct(const Plan &p, int block, const int64_t *need, ptrdiff_t n_need,
                   view<const double, 1> seg_h, const Values &v, cd *m0, cd *m1) {
    if (!v.has_d) throw std::invalid_argument("razor pair_reuse: no direct values");
    if (block < 0 || block > (p.has_image ? 1 : 0))
        throw std::invalid_argument("razor pair_reuse: no such block");
    const ptrdiff_t n_seg = static_cast<ptrdiff_t>(p.n_seg);
    const int32_t *map = p.dmap[block].data();
    const cd *d0 = v.d0.data(), *d1 = v.d1.data();
    const uint8_t *rf = p.drep_flip.data();
    for (ptrdiff_t i = 0; i < n_need; i++)
        if (need[i] < 0 || need[i] >= n_seg) throw std::invalid_argument("razor pair_reuse: observer out of range");
#pragma omp parallel for schedule(static)
    for (ptrdiff_t i = 0; i < n_need; i++) {
        const int32_t *row = map + need[i] * n_seg;
        cd *o0 = m0 + i * n_seg, *o1 = m1 + i * n_seg;
        for (ptrdiff_t s = 0; s < n_seg; s++) {
            const int32_t e = row[s];
            const int32_t c = e >> 1;
            const cd a0 = d0[c], a1 = d1[c];
            o0[s] = a0;
            if ((e & 1) != rf[c]) {
                const double h = seg_h(s);
                o1[s] = cd(h * a0.real() - a1.real(), h * a0.imag() - a1.imag());
            } else {
                o1[s] = a1;
            }
        }
    }
}

void gather_remainder(const Plan &p, const int64_t *seg, const double *sign, ptrdiff_t m,
                      view<const double, 1> seg_h, const Values &v, cd *out) {
    if (!v.has_q) throw std::invalid_argument("razor pair_reuse: no remainder values");
    const ptrdiff_t n_seg = static_cast<ptrdiff_t>(p.n_seg);
    for (ptrdiff_t r = 0; r < m; r++)
        if (seg[r] < 0 || seg[r] >= n_seg) throw std::invalid_argument("razor pair_reuse: observer out of range");
    const int32_t *map = p.qmap.data();
    const cd *q0 = v.q0.data(), *q1 = v.q1.data();
    const uint8_t *rc = p.qrep_code.data();
#pragma omp parallel for schedule(static)
    for (ptrdiff_t r = 0; r < m; r++) {
        const int32_t *row = map + seg[r] * n_seg;
        const bool neg_row = sign[r] < 0.0;
        cd *o = out + 2 * r * n_seg;
        for (ptrdiff_t s = 0; s < n_seg; s++) {
            const int32_t e = row[s];
            const int32_t c = e >> 2;
            const int rel = (e & 3) ^ rc[c];
            double a0r = q0[c].real(), a0i = q0[c].imag(), a1r = q1[c].real(), a1i = q1[c].imag();
            if (rel & 1) {
                // The source reversed: (Q0, Q1) -> (-Q0, Q1 - h Q0).
                const double h = seg_h(s);
                const double n1r = a1r - h * a0r, n1i = a1i - h * a0i;
                a0r = -a0r;
                a0i = -a0i;
                a1r = n1r;
                a1i = n1i;
            }
            if (((rel & 2) != 0) != neg_row) {
                a0r = -a0r;
                a0i = -a0i;
                a1r = -a1r;
                a1i = -a1i;
            }
            o[2 * s] = cd(a0r, a0i);
            o[2 * s + 1] = cd(a1r, a1i);
        }
    }
}

double verify_error(const cd *ref0, const cd *ref1, const cd *got0, const cd *got1,
                    ptrdiff_t n_rows, ptrdiff_t n_seg, ptrdiff_t stride,
                    view<const double, 1> seg_h) {
    double worst = 0.0;
    for (ptrdiff_t r = 0; r < n_rows; r++)
        for (ptrdiff_t s = 0; s < n_seg; s++) {
            const size_t i = static_cast<size_t>((r * n_seg + s) * stride);
            const double h = seg_h(s);
            const double scale = std::max(std::abs(ref0[i]), std::abs(ref1[i]) / h);
            const double err = std::max(std::abs(got0[i] - ref0[i]), std::abs(got1[i] - ref1[i]) / h);
            if (err == 0.0) continue;
            worst = std::max(worst, scale > 0.0 ? err / scale : HUGE_VAL);
        }
    return worst;
}

}  // namespace reuse
}  // namespace razor
}  // namespace mw
