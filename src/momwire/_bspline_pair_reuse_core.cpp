// bs2's fill-pair reuse (momwire#2063 stage 3): see `_bspline_pair_reuse_core.h`.
// bs2's adapter onto the common core (`_pair_reuse_core.h`): its key, its
// policy and the gather's layout. Its own TU, so no kernel's codegen moves
// with it (momwire#1187): it calls no kernel (Python evaluates the
// representatives with the shipped off-edge fill); the core's census and
// gather templates are instantiated here, beside the key.
#include "_bspline_pair_reuse_core.h"

#include <atomic>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <limits>
#include <algorithm>
#include <map>
#include <string>

namespace mw {
namespace bspline {
namespace reuse {

namespace {

using pair_reuse::FamilyPlan;
using pair_reuse::FamilySpec;
using pair_reuse::Ids;
using pair_reuse::Key;
using pair_reuse::Outcome;
using pair_reuse::bits;
using pair_reuse::qword;
using pair_reuse::quantum;

std::atomic<int64_t> g_plan_allowance{0};  // 0: the derived allowance
std::atomic<int64_t> g_class_limit{0};
std::atomic<int64_t> g_small{SMALL_DECK_SEGS};
std::atomic<int> g_length_bits{LENGTH_BITS};
std::atomic<int> g_image_key_probe{0};
std::atomic<int> g_routine_probe{0};
std::atomic<double> g_same_edge_near{NEAR_LENGTHS};
// The routine bit: w[0]'s top bit, which no general key sets (its exponent,
// eligibility and two 25-bit types fill bits 0..62).
constexpr uint64_t ROUTINE_BIT = uint64_t(1) << 63;

constexpr int W = 6;  // key words

// Per row, the [lo, hi) columns of its near same-edge singles.
struct Bands {
    std::vector<int64_t> lo, hi;
};

bool bit(const std::vector<uint64_t> &b, int64_t c) {
    return !b.empty() && (b[static_cast<size_t>(c) / 64] >> (c % 64) & 1);
}
bool near_apart(const FamilyPlan &f, int64_t c) { return bit(f.near_bits, c) && bit(f.apart_bits, c); }

const pair_reuse::Policy POLICY = [] {
    pair_reuse::Policy p;
    p.capped = true;  // `pair_reuse::class_cap`
    p.rep_rows = REP_ROWS;
    return p;
}();

int64_t value_bytes(int degree) {
    const int64_t d1 = degree + 1;
    return d1 * d1 * static_cast<int64_t>(sizeof(cd));
}

FamilySpec direct_spec(int64_t n, int degree, int n_blocks = 1) {
    FamilySpec d;
    d.name = "direct";
    d.key_words = W;
    d.code_bits = 1;
    d.n_blocks = n_blocks;
    d.rows = d.cols = n;
    d.value_bytes = value_bytes(degree);
    d.near_singles = true;
    d.class_limit = g_class_limit.load();
    return d;
}

// A segment length as a key type: on a step of 2^-LENGTH_BITS of its own
// power-of-two floor (0: its exact bits).
std::pair<uint64_t, uint64_t> length_type(double h) {
    const int lb = g_length_bits.load();
    if (lb <= 0 || !(h > 0.0) || !std::isfinite(h)) return {bits(h), 0};
    int e;
    double inv;
    quantum(h, lb, e, inv);
    return {static_cast<uint64_t>(std::llround(h * inv)), static_cast<uint64_t>(e + 2200) | (uint64_t(1) << 32)};
}

void build(const Geometry &g, Plan &plan_ref) {
    Plan *const plan = &plan_ref;
    auto no = [&](const std::string &why) { pair_reuse::decline(*plan, why); };
    const int64_t n = g.n;
    const bool image = g.img_l != nullptr;
    plan->fam.push_back(FamilyPlan());
    plan->fam.back().spec = direct_spec(n, g.degree, image ? 2 : 1);
    if (n > MAX_SEGS) return no("too many segments to plan");
    const int64_t allowance = plan_allowance(*plan, g.degree);
    const int64_t maps_bytes = pair_reuse::maps_bytes(*plan);
    if (maps_bytes > allowance) return no(pair_reuse::allowance_reason(allowance));

    // Per segment: its length as the kernel forms it, its centre, its
    // direction (r - l), and its types; with the image, the same of the
    // mirrored segment, from the end points the image fill hands the kernel.
    const size_t nb = image ? 2 : 1;
    std::vector<double> len(nb * static_cast<size_t>(n)), mid(nb * static_cast<size_t>(3 * n)),
        dir(nb * static_cast<size_t>(3 * n));
    Ids h_ids, a_ids, b_ids, eh_ids;
    std::vector<uint64_t> h_id(nb * static_cast<size_t>(n)), a_id(static_cast<size_t>(n)),
        b_id(static_cast<size_t>(n), 0), eh_id(static_cast<size_t>(n), 0);
    for (size_t blk = 0; blk < nb; blk++)
        for (int64_t i = 0; i < n; i++) {
            const double *l = (blk ? g.img_l : g.seg_l) + 3 * i, *r = (blk ? g.img_r : g.seg_r) + 3 * i;
            const size_t at = blk * static_cast<size_t>(n) + static_cast<size_t>(i);
            const double dx = r[0] - l[0], dy = r[1] - l[1], dz = r[2] - l[2];
            len[at] = std::sqrt(dx * dx + dy * dy + dz * dz);
            for (int c = 0; c < 3; c++) {
                mid[3 * at + static_cast<size_t>(c)] = 0.5 * (l[c] + r[c]);
                dir[3 * at + static_cast<size_t>(c)] = r[c] - l[c];
            }
            const std::pair<uint64_t, uint64_t> ht = length_type(len[at]);
            h_id[at] = static_cast<uint64_t>(h_ids.id(ht.first, ht.second));
            if (blk) continue;
            a_id[static_cast<size_t>(i)] = static_cast<uint64_t>(a_ids.id(bits(g.a[i])));
            eh_id[static_cast<size_t>(i)] = static_cast<uint64_t>(eh_ids.id(bits(g.edge_h[i])));
            if (g.b_src) b_id[static_cast<size_t>(i)] = static_cast<uint64_t>(b_ids.id(bits(g.b_src[i])));
        }
    // TEST HOOK: block 1 keyed on the source's unmirrored direction.
    const bool key_probe = image && g_image_key_probe.load() == 1;
    constexpr size_t TYPE_MAX = size_t(1) << 25;
    if (h_ids.m.size() >= (size_t(1) << 32) || a_ids.m.size() >= TYPE_MAX || b_ids.m.size() >= TYPE_MAX)
        return no("too many segment types to key");
    const bool ek = g.group_i != nullptr && g.group_j != nullptr;
    const std::vector<double> &thr = g.thresholds;
    const bool routine_probe = g_routine_probe.load() == 1;
    const double same_near = g_same_edge_near.load();

    auto key = [&](int blk, int64_t o, int64_t s, uint64_t *w, uint8_t &code) -> Key {
        code = 0;
        // The direct block's same-edge pairs are the same-edge routine's
        // (keyed apart); a grazing edge's own image pairs are their own
        // classes; any other own image pair is keyed like the rest.
        const bool same = g.edge[o] == g.edge[s];
        if (same && blk && g.img_mode[o] == OwnPairs) return Key::Own;
        const bool apart = same && !blk;
        const Key own = apart ? Key::OwnApart : Key::Own;
        if (apart && !routine_probe) {
            // The same-edge routine reads the edge's h and radius, exactly
            // (its static table, by type), the integer offset j - i, and the
            // arcs; the routine bit keeps the class apart. Within
            // `same_near` max(h, a) a pair is its own class.
            const int64_t k = s - o, ak = k < 0 ? -k : k;
            const double he = g.edge_h[o];
            if (!(static_cast<double>(ak) * he >= same_near * std::max(he, g.a[o]))) return Key::OwnApart;
            w[0] = ROUTINE_BIT | (uint64_t(ek) << 12) | (a_id[static_cast<size_t>(o)] << 13);
            w[1] = eh_id[static_cast<size_t>(o)];
            w[2] = static_cast<uint64_t>(ak);
            w[3] = w[4] = w[5] = 0;
            // RECIPROCITY: (s, o) is the transpose of (o, s), one edge being
            // one radius and one EK group.
            code = k < 0 ? 1 : 0;
            return Key::Apart;
        }
        // Block 1's source is the mirrored segment.
        const size_t si = static_cast<size_t>(blk) * static_cast<size_t>(n) + static_cast<size_t>(s);
        const double *mo = &mid[static_cast<size_t>(3 * o)], *ms = &mid[3 * si];
        const double *A = &dir[static_cast<size_t>(3 * o)], *B = &dir[3 * (key_probe ? static_cast<size_t>(s) : si)];
        const double mx = mo[0] - ms[0], my = mo[1] - ms[1], mz = mo[2] - ms[2];
        const double mm = mx * mx + my * my + mz * mz;
        const double dm = std::sqrt(mm);
        const double hi = len[static_cast<size_t>(o)], hj = len[si];
        const double hmax = std::max(hi, hj);
        if (!(dm >= NEAR_LENGTHS * std::max(hmax, g.a[o]))) return own;
        const double ratio = dm / hmax;
        for (double t : thr)
            if (std::fabs(ratio - t) <= TIE_REL * t) return own;
        const double D = std::max(dm - 0.5 * (hi + hj), g.a[o]);
        const double D2 = D * D;
        if (!(D2 > 0.0) || !std::isfinite(D2)) return own;
        int e;
        double inv;
        quantum(D2, KEY_BITS, e, inv);
        const double mA = mx * A[0] + my * A[1] + mz * A[2];
        const double mB = mx * B[0] + my * B[1] + mz * B[2];
        const double AB = A[0] * B[0] + A[1] * B[1] + A[2] * B[2];
        if (!qword(mm, inv, w[2]) || !qword(mA, inv, w[3]) || !qword(mB, inv, w[4]) || !qword(AB, inv, w[5]))
            return own;
        const bool elig = ek && g.group_i[o] >= 0 && g.group_i[o] == g.group_j[s];
        const uint64_t bt = elig && g.b_src ? b_id[static_cast<size_t>(s)] : 0;
        w[0] = static_cast<uint64_t>(e + 2200) | (uint64_t(elig) << 12) | (a_id[static_cast<size_t>(o)] << 13) |
               (bt << 38);
        w[1] = h_id[static_cast<size_t>(o)] | (h_id[si] << 32);
        // RECIPROCITY: the pair read from s is the transpose, J[P, p, s, o]
        // = J[p, P, o, s], when the two rows' radii (and under the extended
        // kernel their tubes and eligibility) agree: its key is this one's
        // with the lengths swapped and m -> -m (exactly -m.B, -m.A). The
        // smaller of the two is the class's; code 1 says this pair is the
        // transpose of its key's form.
        const bool sym = a_id[static_cast<size_t>(o)] == a_id[static_cast<size_t>(s)] &&
                         (!elig || !g.b_src || b_id[static_cast<size_t>(o)] == b_id[static_cast<size_t>(s)]) &&
                         elig == (ek && g.group_i[s] >= 0 && g.group_i[s] == g.group_j[o]);
        if (sym) {
            // The transpose reads this pair's own two segments, source (in
            // block 1 the mirrored one) first. EK eligibility is one label
            // for real and mirrored segments alike (`_ek_axis_labels`), so
            // the direct labels read it in either block.
            const uint64_t t[W] = {w[0],
                                   h_id[si] | (h_id[static_cast<size_t>(o)] << 32),
                                   w[2],
                                   static_cast<uint64_t>(-static_cast<int64_t>(w[4])),
                                   static_cast<uint64_t>(-static_cast<int64_t>(w[3])),
                                   w[5]};
            int64_t a[W], b[W];
            for (int q = 0; q < W; q++) {
                a[q] = static_cast<int64_t>(w[q]);
                b[q] = static_cast<int64_t>(t[q]);
            }
            if (std::lexicographical_compare(b, b + W, a, a + W)) {
                std::memcpy(w, t, sizeof t);
                code = 1;
            }
        }
        return apart ? Key::Apart : Key::Shared;
    };
    FamilyPlan &f = plan->fam[0];
    int64_t peak = 0;
    constexpr int64_t kUnbounded = std::numeric_limits<int64_t>::max();
    const Outcome out = pair_reuse::census<W>(f, POLICY, kUnbounded, peak, key);
    plan->census_peak_bytes = maps_bytes + peak;
    if (out == Outcome::Poor) return no(pair_reuse::cap_reason(n * n));
    if (out != Outcome::Ok) return no("too many classes to code");
    // Each row's band of near same-edge singles (`same_edge_band`): the
    // columns of its own edge whose class is a near single of the routine,
    // on an edge shorter in its segments than its radius (h < a, the
    // hairpin's bridge), where the near block's cancelling sums are what
    // the deck reads; a thinner edge's near band is |j - i| <= 3, its near
    // singles are their class values.
    auto bands = std::make_shared<Bands>();
    bands->lo.assign(static_cast<size_t>(n), 0);
    bands->hi.assign(static_cast<size_t>(n), 0);
    for (int64_t e0 = 0; e0 < n;) {
        int64_t e1 = e0;
        while (e1 < n && g.edge[e1] == g.edge[e0]) e1++;
        for (int64_t o = e0; o < e1 && g.edge_h[e0] < g.a[e0]; o++) {
            int64_t lo = e1, hi = e0;
            for (int64_t s = e0; s < e1; s++)
                if (near_apart(f, f.map[0][static_cast<size_t>(o * n + s)] >> f.spec.code_bits)) {
                    lo = std::min(lo, s);
                    hi = std::max(hi, s + 1);
                }
            if (hi > lo) {
                bands->lo[static_cast<size_t>(o)] = lo;
                bands->hi[static_cast<size_t>(o)] = hi;
            }
        }
        e0 = e1;
    }
    plan->ext_bytes = 16 * static_cast<size_t>(n);
    plan->ext = bands;
    plan->budget_bytes = pair_reuse::held_bytes(*plan);
    if (plan->budget_bytes > allowance) return no(pair_reuse::allowance_reason(allowance));
}

template <class T>
void put(std::string &s, const T &v) {
    s.append(reinterpret_cast<const char *>(&v), sizeof v);
}

std::string fingerprint_of(const Geometry &g) {
    std::string s;
    put(s, g.n);
    put(s, static_cast<int32_t>(KEY_BITS));
    put(s, static_cast<int32_t>(g.degree));
    put(s, g_class_limit.load());
    put(s, g_length_bits.load());
    put(s, g_plan_allowance.load());
    put(s, g_routine_probe.load());
    put(s, g_same_edge_near.load());
    put(s, pair_reuse::cap_key());
    put(s, static_cast<int64_t>(g.thresholds.size()));
    for (double t : g.thresholds) put(s, t);
    put(s, g.group_i != nullptr);
    put(s, g.b_src != nullptr);
    s.append(reinterpret_cast<const char *>(g.seg_l), static_cast<size_t>(3 * g.n) * sizeof(double));
    s.append(reinterpret_cast<const char *>(g.seg_r), static_cast<size_t>(3 * g.n) * sizeof(double));
    s.append(reinterpret_cast<const char *>(g.a), static_cast<size_t>(g.n) * sizeof(double));
    s.append(reinterpret_cast<const char *>(g.edge), static_cast<size_t>(g.n) * sizeof(int32_t));
    s.append(reinterpret_cast<const char *>(g.edge_h), static_cast<size_t>(g.n) * sizeof(double));
    if (g.group_i) {
        s.append(reinterpret_cast<const char *>(g.group_i), static_cast<size_t>(g.n) * sizeof(int64_t));
        s.append(reinterpret_cast<const char *>(g.group_j), static_cast<size_t>(g.n) * sizeof(int64_t));
    }
    if (g.b_src) s.append(reinterpret_cast<const char *>(g.b_src), static_cast<size_t>(g.n) * sizeof(double));
    // A plan without the image block keys as it always did.
    if (g.img_l) {
        put(s, g_image_key_probe.load());
        s.append(reinterpret_cast<const char *>(g.img_l), static_cast<size_t>(3 * g.n) * sizeof(double));
        s.append(reinterpret_cast<const char *>(g.img_r), static_cast<size_t>(3 * g.n) * sizeof(double));
        s.append(reinterpret_cast<const char *>(g.img_mode), static_cast<size_t>(g.n) * sizeof(int8_t));
    }
    return s;
}

}  // namespace

int64_t kept_bytes_per_class(int degree) {
    FamilySpec f;
    f.value_bytes = value_bytes(degree);
    // `class_bytes` (values, group entry, code) and, at most, one observer
    // row and one source in its group's lists (`Group::obs`, `Group::src`).
    return pair_reuse::class_bytes(f) + static_cast<int64_t>(2 * sizeof(int32_t));
}

int64_t plan_allowance(const Plan &p, int degree) {
    const int64_t b = g_plan_allowance.load();
    if (b > 0) return b;
    // Per block, as razor2p's: each block's map and its cap's classes.
    int64_t a = 0;
    for (const FamilyPlan &f : p.fam)
        for (int blk = 0; blk < f.spec.n_blocks; blk++)
            a += pair_reuse::plan_allowance(f.spec.rows * f.spec.cols, kept_bytes_per_class(degree));
    return a;
}

void set_plan_allowance(int64_t bytes) { g_plan_allowance.store(bytes <= 0 ? 0 : bytes); }
void set_class_limit(int64_t classes) { g_class_limit.store(classes < 0 ? 0 : classes); }
void set_small_deck_segs(int64_t segs) { g_small.store(segs < 0 ? SMALL_DECK_SEGS : segs); }
void set_length_bits(int b) { g_length_bits.store(b < 0 ? LENGTH_BITS : b); }
int length_bits() { return g_length_bits.load(); }
int64_t small_deck_segs() { return g_small.load(); }

void set_image_key_probe(int probe) { g_image_key_probe.store(probe); }
void set_routine_probe(int probe) { g_routine_probe.store(probe); }
void set_same_edge_near_lengths(double l) { g_same_edge_near.store(l < 0.0 ? NEAR_LENGTHS : l); }

std::shared_ptr<const Plan> plan_for(const Geometry &g, bool *built) {
    if (g.n <= 0 || !g.seg_l || !g.seg_r || !g.a || !g.edge || !g.edge_h)
        throw std::invalid_argument("bspline pair_reuse: an empty or mis-shaped geometry");
    if ((g.img_l == nullptr) != (g.img_r == nullptr) || (g.img_l == nullptr) != (g.img_mode == nullptr))
        throw std::invalid_argument("bspline pair_reuse: the image's end points and modes go together");
    if ((g.group_i == nullptr) != (g.group_j == nullptr))
        throw std::invalid_argument("bspline pair_reuse: group_i and group_j go together");
    return pair_reuse::plan_for(
        BASIS, fingerprint_of(g), [&g](Plan &p) { build(g, p); }, built);
}

std::shared_ptr<const Plan> plan_of_classes(const int32_t *cls, int64_t n) {
    auto plan = std::make_shared<Plan>();
    plan->basis = BASIS;
    plan->fam.push_back(FamilyPlan());
    FamilyPlan &f = plan->fam[0];
    f.spec = direct_spec(n, 2);
    f.spec.class_limit = 0;
    f.spec.code_bits = 0;
    const int64_t nn = n * n;
    int64_t n_cls = 0;
    for (int64_t i = 0; i < nn; i++) {
        if (cls[i] < 0) throw std::invalid_argument("bspline pair_reuse: a negative class");
        n_cls = std::max<int64_t>(n_cls, int64_t(cls[i]) + 1);
    }
    if (n_cls >= (int64_t(1) << 31)) throw std::invalid_argument("bspline pair_reuse: too many classes");
    // Each class's representative is its first pair, row-major; classes are
    // renumbered in that order so the plan reads as a census's would.
    std::vector<int64_t> first(static_cast<size_t>(n_cls), -1);
    for (int64_t i = 0; i < nn; i++)
        if (first[static_cast<size_t>(cls[i])] < 0) first[static_cast<size_t>(cls[i])] = i;
    std::vector<int32_t> num(static_cast<size_t>(n_cls), -1);
    std::vector<pair_reuse::detail::Rep> reps;
    f.map[0].resize(static_cast<size_t>(nn));
    for (int64_t i = 0; i < nn; i++) {
        int32_t &c = num[static_cast<size_t>(cls[i])];
        if (c < 0) {
            c = static_cast<int32_t>(reps.size());
            pair_reuse::detail::Rep r;
            r.obs = static_cast<int32_t>(i / n);
            r.src = static_cast<int32_t>(i % n);
            reps.push_back(r);
        }
        f.map[0][static_cast<size_t>(i)] = c;
    }
    pair_reuse::detail::finish(f, reps, 0, POLICY);
    return plan;
}

int64_t same_edge_pairs(const Plan &p) { return p.fam.empty() ? 0 : p.fam[0].apart_pairs; }
int64_t same_edge_classes(const Plan &p) { return p.fam.empty() ? 0 : static_cast<int64_t>(p.fam[0].apart_cls.size()); }

bool has_image(const Plan &p) { return !p.fam.empty() && p.fam[0].spec.n_blocks == 2 && !p.fam[0].map[1].empty(); }

int64_t near_pairs(const Plan &p) { return p.fam.empty() ? 0 : p.fam[0].near; }

bool near_same_edge(const Plan &p, int32_t c) { return !p.fam.empty() && near_apart(p.fam[0], c); }

void same_edge_band(const Plan &p, int64_t o, int64_t &lo, int64_t &hi) {
    lo = hi = 0;
    const Bands *b = static_cast<const Bands *>(p.ext.get());
    if (!b || o < 0 || o >= static_cast<int64_t>(b->lo.size())) return;
    lo = b->lo[static_cast<size_t>(o)];
    hi = b->hi[static_cast<size_t>(o)];
}

void near_same_edge_mask(const Plan &p, int64_t r0, int64_t r1, int64_t c0, int64_t c1, uint8_t *out) {
    const FamilyPlan &f = p.fam.at(0);
    const int64_t n = f.spec.cols;
    for (int64_t o = r0; o < r1; o++)
        for (int64_t s = c0; s < c1; s++)
            out[(o - r0) * (c1 - c0) + (s - c0)] =
                near_apart(f, f.map[0][static_cast<size_t>(o * n + s)] >> f.spec.code_bits) ? 1 : 0;
}

void gather(const Plan &p, int64_t r0, int64_t r1, const cd *values, int d1, cd *out) {
    const FamilyPlan &f = p.fam.at(0);
    const int64_t n = f.spec.cols, m = r1 - r0, dd = int64_t(d1) * d1;
    if (r0 < 0 || r1 > f.spec.rows || m <= 0) throw std::invalid_argument("bspline pair_reuse: rows out of range");
    std::vector<int64_t> rows(static_cast<size_t>(m));
    for (int64_t i = 0; i < m; i++) rows[static_cast<size_t>(i)] = r0 + i;
    const int64_t plane = m * n;
    pair_reuse::gather(f, 0, rows.data(), static_cast<ptrdiff_t>(m), [&](ptrdiff_t i, ptrdiff_t s, int32_t c, int rel) {
        const cd *v = values + static_cast<int64_t>(c) * dd;
        cd *o = out + i * n + s;
        if (rel) {
            // The transpose of the class's value (reciprocity): a copy.
            for (int p = 0; p < d1; p++)
                for (int q = 0; q < d1; q++) o[(p * d1 + q) * plane] = v[q * d1 + p];
        } else {
            for (int64_t pq = 0; pq < dd; pq++) o[pq * plane] = v[pq];
        }
    });
}

namespace {
std::atomic<int> g_window_probe{0};
}

void set_window_probe(int probe) { g_window_probe.store(probe); }
int window_probe() { return g_window_probe.load(); }

void check_window(const Window &w) {
    if (!w.plan || !w.values || w.plan->fam.empty())
        throw std::invalid_argument("bspline pair_reuse: a window needs its plan and values");
    const FamilyPlan &f = w.plan->fam[0];
    if (f.map[0].empty()) throw std::invalid_argument("bspline pair_reuse: a declined plan has no window");
    if (w.d1 < 2 || w.d1 > 4) throw std::invalid_argument("bspline pair_reuse: d + 1 must be 2, 3 or 4");
    if (w.own) {
        if (!(w.lo && w.hi && 0 <= w.r0 && w.r0 < w.r1 && w.r1 <= f.spec.rows && w.bw >= 0))
            throw std::invalid_argument("bspline pair_reuse: the own band's rows are out of range");
        for (int64_t i = 0; i < w.r1 - w.r0; i++)
            if (!(0 <= w.lo[i] && w.lo[i] <= w.hi[i] && w.hi[i] - w.lo[i] <= w.bw && w.hi[i] <= f.spec.cols))
                throw std::invalid_argument("bspline pair_reuse: a row's band is out of range");
    }
}

void window_tile(const Window &w, const int64_t *rows, size_t nr, const int64_t *cols, size_t nc, cd *out) {
    const FamilyPlan &f = w.plan->fam[0];
    const int d1 = w.d1;
    const int64_t dd = int64_t(d1) * d1;
    const size_t plane = nr * nc;
    const bool probe = g_window_probe.load(std::memory_order_relaxed) == 1;
    const int64_t bw = w.bw, m = w.r1 - w.r0;
    pair_reuse::gather_pairs(f, 0, rows, static_cast<ptrdiff_t>(nr), cols, static_cast<ptrdiff_t>(nc),
                             [&](ptrdiff_t io, ptrdiff_t is, int32_t c, int rel) {
        cd *o = out + io * static_cast<ptrdiff_t>(nc) + is;
        const int64_t s = cols[is];
        if (w.own && s >= w.lo[rows[io] - w.r0] && s < w.hi[rows[io] - w.r0] && near_apart(f, c)) {
            // A near same-edge single in its row's band: the sweep's own
            // moments of the pair, which the same-edge fixup corrects as
            // without reuse.
            const int64_t i = rows[io] - w.r0, j = s - w.lo[i];
            const cd *v = w.own + i * bw + j;
            for (int64_t pq = 0; pq < dd; pq++) o[pq * plane] = v[pq * m * bw];
            return;
        }
        const cd *v = w.values + static_cast<int64_t>(c) * dd;
        if (rel && !probe) {
            // The transpose of the class's value (reciprocity), as `gather`.
            for (int p = 0; p < d1; p++)
                for (int q = 0; q < d1; q++) o[(p * d1 + q) * plane] = v[q * d1 + p];
        } else {
            for (int64_t pq = 0; pq < dd; pq++) o[pq * plane] = v[pq];
        }
    });
}

void check_image_window(const ImageWindow &w) {
    if (!w.plan || !w.values || !has_image(*w.plan))
        throw std::invalid_argument("bspline pair_reuse: an image window needs a plan with the image block");
    if (w.d1 < 2 || w.d1 > 4) throw std::invalid_argument("bspline pair_reuse: d + 1 must be 2, 3 or 4");
}

void image_tile(const ImageWindow &w, const int64_t *rows, size_t nr, const int64_t *cols, size_t nc, cd *out) {
    const FamilyPlan &f = w.plan->fam[0];
    const int d1 = w.d1;
    const int64_t dd = int64_t(d1) * d1;
    const size_t plane = nr * nc;
    pair_reuse::gather_pairs(f, 1, rows, static_cast<ptrdiff_t>(nr), cols, static_cast<ptrdiff_t>(nc),
                             [&](ptrdiff_t io, ptrdiff_t is, int32_t c, int rel) {
        cd *o = out + io * static_cast<ptrdiff_t>(nc) + is;
        const cd *v = w.values + static_cast<int64_t>(c) * dd;
        if (rel) {
            for (int p = 0; p < d1; p++)
                for (int q = 0; q < d1; q++) o[(p * d1 + q) * plane] = v[q * d1 + p];
        } else {
            for (int64_t pq = 0; pq < dd; pq++) o[pq * plane] = v[pq];
        }
    });
}

namespace {

std::atomic<int> g_remainder_key_probe{0};

constexpr int RW = 9;  // the remainder key's words

FamilySpec remainder_spec(int64_t n, int degree) {
    FamilySpec f;
    f.name = "remainder";
    f.key_words = RW;
    f.code_bits = 1;
    f.n_blocks = 1;
    f.rows = f.cols = n;
    f.value_bytes = value_bytes(degree);
    f.near_singles = true;
    f.keep_reps = true;
    f.class_limit = g_class_limit.load();
    return f;
}

std::string remainder_fingerprint(const RemainderPlanGeometry &g) {
    std::string s("remainder");
    put(s, g.n);
    put(s, static_cast<int32_t>(KEY_BITS));
    put(s, static_cast<int32_t>(g.degree));
    put(s, g_class_limit.load());
    put(s, g_length_bits.load());
    put(s, g_plan_allowance.load());
    put(s, g_remainder_key_probe.load());
    put(s, pair_reuse::cap_key());
    put(s, g.ground_z);
    s.append(reinterpret_cast<const char *>(g.seg_l), static_cast<size_t>(3 * g.n) * sizeof(double));
    s.append(reinterpret_cast<const char *>(g.seg_r), static_cast<size_t>(3 * g.n) * sizeof(double));
    s.append(reinterpret_cast<const char *>(g.h), static_cast<size_t>(g.n) * sizeof(double));
    put(s, g.n_raised);
    if (g.n_raised)
        s.append(reinterpret_cast<const char *>(g.raised), static_cast<size_t>(g.n_raised) * sizeof(int64_t));
    return s;
}

// The remainder's cap counts its unordered pairs, n (n + 1) / 2: the
// kernel's walk already forms each node pair's grid read once for both
// orders (`node_block`), so a plan whose only compression is reciprocity
// would save one projection a node pair and cost a census, its map and a
// class value per unordered pair. Past it the plan declines.
int64_t remainder_cap_pairs(int64_t n) { return n * (n + 1) / 2; }
std::string remainder_cap_reason(int64_t n) {
    // `pair_reuse::cap_reason`'s sentence, of unordered pairs.
    char buf[176];
    const int64_t pairs = remainder_cap_pairs(n);
    std::snprintf(buf, sizeof buf, "more than %lld classes for %lld unordered pairs (the cap is %.4gx compression)",
                  static_cast<long long>(pair_reuse::class_cap(pairs)), static_cast<long long>(pairs),
                  pair_reuse::cap_compression());
    return buf;
}

void build_remainder(const RemainderPlanGeometry &g, Plan &plan) {
    auto no = [&](const std::string &why) { pair_reuse::decline(plan, why); };
    const int64_t n = g.n;
    plan.fam.push_back(FamilyPlan());
    plan.fam.back().spec = remainder_spec(n, g.degree);
    if (n > MAX_SEGS) return no("too many segments to plan");
    const int64_t allowance = plan_allowance(plan, g.degree);
    const int64_t maps_bytes = pair_reuse::maps_bytes(plan);
    if (maps_bytes > allowance) return no(pair_reuse::allowance_reason(allowance));
    // Per segment: its centre, its vector (r - l), the horizontal part's
    // length, and its length's type.
    const size_t un = static_cast<size_t>(n);
    std::vector<double> mid(3 * un), dir(3 * un), hor(un);
    std::vector<uint64_t> h_id(un);
    Ids h_ids;
    for (size_t i = 0; i < un; i++) {
        const double *l = g.seg_l + 3 * i, *r = g.seg_r + 3 * i;
        for (int c = 0; c < 3; c++) {
            mid[3 * i + static_cast<size_t>(c)] = 0.5 * (l[c] + r[c]);
            dir[3 * i + static_cast<size_t>(c)] = r[c] - l[c];
        }
        hor[i] = std::hypot(dir[3 * i], dir[3 * i + 1]);
        const std::pair<uint64_t, uint64_t> ht = length_type(g.h[i]);
        h_id[i] = static_cast<uint64_t>(h_ids.id(ht.first, ht.second));
    }
    if (h_ids.m.size() >= (size_t(1) << 32)) return no("too many segment types to key");
    std::vector<int64_t> raised(g.raised, g.raised + g.n_raised);
    std::sort(raised.begin(), raised.end());
    const int probe = g_remainder_key_probe.load();
    const double gz = g.ground_z;

    auto key = [&](int, int64_t o, int64_t s, uint64_t *w, uint8_t &code) -> Key {
        code = 0;
        const size_t uo = static_cast<size_t>(o), us = static_cast<size_t>(s);
        const double *co = &mid[3 * uo], *cs = &mid[3 * us], *A = &dir[3 * uo], *B = &dir[3 * us];
        const double mx = co[0] - cs[0], my = co[1] - cs[1], mz = co[2] - cs[2];
        const double hmax = std::max(g.h[o], g.h[s]);
        const double mh2 = mx * mx + my * my;
        if (!(std::sqrt(mh2 + mz * mz) >= NEAR_LENGTHS * hmax)) return Key::Own;
        const double S = (co[2] - gz) + (cs[2] - gz);
        const double rho_lb = std::max(std::sqrt(mh2) - 0.5 * (hor[uo] + hor[us]), 0.0);
        const double hh_lb = std::max(S - 0.5 * (std::fabs(A[2]) + std::fabs(B[2])), 0.0);
        const double D = std::hypot(rho_lb, hh_lb);
        if (!(D >= NEAR_LENGTHS * hmax) || !std::isfinite(D)) return Key::Own;
        if (!raised.empty() && std::binary_search(raised.begin(), raised.end(), o * n + s)) return Key::Own;
        const int e = std::ilogb(D);
        const double inv1 = std::ldexp(1.0, KEY_BITS - e), inv2 = std::ldexp(1.0, KEY_BITS - 2 * e);
        double g2[4] = {mh2, mx * A[0] + my * A[1], mx * B[0] + my * B[1], A[0] * B[0] + A[1] * B[1]};
        double g1[3] = {A[2], B[2], probe == 3 ? co[2] - cs[2] : S};
        if (probe == 1) g1[2] = 0.0;
        if (probe == 2) {
            // the tangents as they lie, not in the pair's frame
            g2[1] = A[0] * hmax;
            g2[2] = A[1] * hmax;
            g2[3] = (B[0] + 3.0 * B[1]) * hmax;
        }
        for (int q = 0; q < 4; q++)
            if (!qword(g2[q], inv2, w[2 + q])) return Key::Own;
        for (int q = 0; q < 3; q++)
            if (!qword(g1[q], inv1, w[6 + q])) return Key::Own;
        w[0] = static_cast<uint64_t>(e + 2200);
        w[1] = h_id[uo] | (h_id[us] << 32);
        if (probe == 2) return Key::Shared;  // its words have no reversed form
        // RECIPROCITY: read from s, m -> -m (m.A <- -m.B, m.B <- -m.A),
        // the lengths and A_z, B_z swapped; the probe's DIFFERENCE negates.
        const uint64_t t[RW] = {w[0],
                                h_id[us] | (h_id[uo] << 32),
                                w[2],
                                static_cast<uint64_t>(-static_cast<int64_t>(w[4])),
                                static_cast<uint64_t>(-static_cast<int64_t>(w[3])),
                                w[5],
                                w[7],
                                w[6],
                                probe == 3 ? static_cast<uint64_t>(-static_cast<int64_t>(w[8])) : w[8]};
        int64_t a[RW], b[RW];
        for (int q = 0; q < RW; q++) {
            a[q] = static_cast<int64_t>(w[q]);
            b[q] = static_cast<int64_t>(t[q]);
        }
        if (std::lexicographical_compare(b, b + RW, a, a + RW)) {
            std::memcpy(w, t, sizeof t);
            code = 1;
        }
        return Key::Shared;
    };
    pair_reuse::Policy pol;
    pol.capped = true;
    pol.rep_rows = 0;  // the kernel evaluates the representatives
    pol.cap_pairs = remainder_cap_pairs(n);
    FamilyPlan &f = plan.fam[0];
    int64_t peak = 0;
    const Outcome out = pair_reuse::census<RW>(f, pol, std::numeric_limits<int64_t>::max(), peak, key);
    plan.census_peak_bytes = maps_bytes + peak;
    if (out == Outcome::Poor) return no(remainder_cap_reason(n));
    if (out != Outcome::Ok) return no("too many classes to code");
    plan.budget_bytes = pair_reuse::held_bytes(plan);
    if (plan.budget_bytes > allowance) return no(pair_reuse::allowance_reason(allowance));
}

}  // namespace

void set_remainder_key_probe(int probe) { g_remainder_key_probe.store(probe); }
int remainder_key_probe() { return g_remainder_key_probe.load(); }

std::shared_ptr<const Plan> remainder_plan_for(const RemainderPlanGeometry &g, bool *built) {
    if (g.n <= 0 || !g.seg_l || !g.seg_r || !g.h || (g.n_raised > 0 && !g.raised))
        throw std::invalid_argument("bspline pair_reuse: an empty or mis-shaped remainder geometry");
    for (int64_t t = 0; t < g.n_raised; t++)
        if (g.raised[t] < 0 || g.raised[t] >= g.n * g.n)
            throw std::invalid_argument("bspline pair_reuse: a raised pair out of range");
    // keep 2: the direct plan, just used by the same fill, stays cached
    return pair_reuse::plan_for(
        BASIS, remainder_fingerprint(g), [&g](Plan &p) { build_remainder(g, p); }, built, 2);
}

RemainderClasses remainder_classes(const Plan &p) {
    RemainderClasses c;
    if (p.fam.empty() || std::string(p.fam[0].spec.name) != "remainder" || p.fam[0].map[0].empty()) return c;
    const FamilyPlan &f = p.fam[0];
    c.n = f.spec.rows;
    c.classes = f.classes;
    c.map = f.map[0].data();
    c.rep_flat = f.rep_flat.data();
    c.rep_code = f.rep_code.data();
    c.single = f.near_bits.empty() ? nullptr : f.near_bits.data();
    return c;
}

void remainder_wire_map(const Plan &p, const uint8_t *tied, const int32_t *wire_of_seg, int64_t n_wires,
                        int64_t *out) {
    std::fill(out, out + n_wires * n_wires * 3, int64_t(0));
    const RemainderClasses c = remainder_classes(p);
    if (!c.map) return;
    std::vector<uint8_t> seen(static_cast<size_t>(c.classes), 0);
    for (int64_t o = 0; o < c.n; o++)
        for (int64_t s = 0; s < c.n; s++) {
            const int64_t k = c.map[o * c.n + s] >> 1;
            const bool single = c.single && (c.single[k / 64] >> (k % 64) & 1);
            const bool own = single || (tied && tied[k]) || !seen[static_cast<size_t>(k)];
            seen[static_cast<size_t>(k)] = 1;
            out[(static_cast<int64_t>(wire_of_seg[o]) * n_wires + wire_of_seg[s]) * 3 + (own ? 0 : 1)]++;
        }
}

void wire_map(const Plan &p, const int32_t *wire_of_seg, int64_t n_wires, int64_t *out, int block) {
    const int64_t cell = n_wires * n_wires * 3;
    std::fill(out, out + cell, int64_t(0));
    if (p.fam.empty() || p.fam[0].map[0].empty()) return;
    if (block < 0 || block >= p.fam[0].spec.n_blocks) throw std::invalid_argument("bspline pair_reuse: no such block");
    // Every block in one pass (a class is computed at its first pair, in
    // block order), then the one asked for.
    std::vector<int64_t> all(static_cast<size_t>(cell * p.fam[0].spec.n_blocks), 0);
    pair_reuse::wire_map(p.fam[0], wire_of_seg, wire_of_seg, n_wires, all.data());
    std::copy(all.begin() + block * cell, all.begin() + (block + 1) * cell, out);
}

pair_reuse::Counters cache_counters() { return pair_reuse::counters(BASIS); }

void clear_cache() { pair_reuse::clear_cache(BASIS); }

}  // namespace reuse
}  // namespace bspline
}  // namespace mw
