// bs2's fill-pair reuse (momwire#2063 stage 3): see `_bspline_pair_reuse_core.h`.
// bs2's adapter onto the common core (`_pair_reuse_core.h`): its key, its
// policy and the gather's layout. Its own TU, so no kernel's codegen moves
// with it (momwire#1187): it calls no kernel (Python evaluates the
// representatives with the shipped off-edge fill); the core's census and
// gather templates are instantiated here, beside the key.
#include "_bspline_pair_reuse_core.h"

#include <atomic>
#include <cmath>
#include <cstring>
#include <limits>
#include <map>

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

constexpr int W = 6;  // key words

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

FamilySpec direct_spec(int64_t n, int degree) {
    FamilySpec d;
    d.name = "direct";
    d.key_words = W;
    d.code_bits = 1;
    d.n_blocks = 1;
    d.rows = d.cols = n;
    d.value_bytes = value_bytes(degree);
    d.near_singles = true;
    d.list_near = false;
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
    plan->fam.push_back(FamilyPlan());
    plan->fam.back().spec = direct_spec(n, g.degree);
    if (n > MAX_SEGS) return no("too many segments to plan");
    const int64_t allowance = plan_allowance(*plan, g.degree);
    const int64_t maps_bytes = pair_reuse::maps_bytes(*plan);
    if (maps_bytes > allowance) return no(pair_reuse::allowance_reason(allowance));

    // Per segment: its length as the kernel forms it, its centre, its
    // direction (r - l), and its types.
    std::vector<double> len(static_cast<size_t>(n)), mid(static_cast<size_t>(3 * n)),
        dir(static_cast<size_t>(3 * n));
    Ids h_ids, a_ids, b_ids;
    std::vector<uint64_t> h_id(static_cast<size_t>(n)), a_id(static_cast<size_t>(n)),
        b_id(static_cast<size_t>(n), 0);
    for (int64_t i = 0; i < n; i++) {
        const double *l = g.seg_l + 3 * i, *r = g.seg_r + 3 * i;
        const double dx = r[0] - l[0], dy = r[1] - l[1], dz = r[2] - l[2];
        len[static_cast<size_t>(i)] = std::sqrt(dx * dx + dy * dy + dz * dz);
        for (int c = 0; c < 3; c++) {
            mid[static_cast<size_t>(3 * i + c)] = 0.5 * (l[c] + r[c]);
            dir[static_cast<size_t>(3 * i + c)] = r[c] - l[c];
        }
        const std::pair<uint64_t, uint64_t> ht = length_type(len[static_cast<size_t>(i)]);
        h_id[static_cast<size_t>(i)] = static_cast<uint64_t>(h_ids.id(ht.first, ht.second));
        a_id[static_cast<size_t>(i)] = static_cast<uint64_t>(a_ids.id(bits(g.a[i])));
        if (g.b_src) b_id[static_cast<size_t>(i)] = static_cast<uint64_t>(b_ids.id(bits(g.b_src[i])));
    }
    constexpr size_t TYPE_MAX = size_t(1) << 25;
    if (h_ids.m.size() >= (size_t(1) << 32) || a_ids.m.size() >= TYPE_MAX || b_ids.m.size() >= TYPE_MAX)
        return no("too many segment types to key");
    const bool ek = g.group_i != nullptr && g.group_j != nullptr;
    const std::vector<double> &thr = g.thresholds;

    auto key = [&](int, int64_t o, int64_t s, uint64_t *w, uint8_t &code) -> Key {
        code = 0;
        if (g.edge[o] == g.edge[s]) {
            for (int q = 0; q < W; q++) w[q] = 0;  // no real key has w[0] == 0
            return Key::Near;
        }
        const double *mo = &mid[static_cast<size_t>(3 * o)], *ms = &mid[static_cast<size_t>(3 * s)];
        const double *A = &dir[static_cast<size_t>(3 * o)], *B = &dir[static_cast<size_t>(3 * s)];
        const double mx = mo[0] - ms[0], my = mo[1] - ms[1], mz = mo[2] - ms[2];
        const double mm = mx * mx + my * my + mz * mz;
        const double dm = std::sqrt(mm);
        const double hi = len[static_cast<size_t>(o)], hj = len[static_cast<size_t>(s)];
        const double hmax = std::max(hi, hj);
        if (!(dm >= NEAR_LENGTHS * std::max(hmax, g.a[o]))) return Key::Own;
        const double ratio = dm / hmax;
        for (double t : thr)
            if (std::fabs(ratio - t) <= TIE_REL * t) return Key::Own;
        const double D = std::max(dm - 0.5 * (hi + hj), g.a[o]);
        const double D2 = D * D;
        if (!(D2 > 0.0) || !std::isfinite(D2)) return Key::Own;
        int e;
        double inv;
        quantum(D2, KEY_BITS, e, inv);
        const double mA = mx * A[0] + my * A[1] + mz * A[2];
        const double mB = mx * B[0] + my * B[1] + mz * B[2];
        const double AB = A[0] * B[0] + A[1] * B[1] + A[2] * B[2];
        if (!qword(mm, inv, w[2]) || !qword(mA, inv, w[3]) || !qword(mB, inv, w[4]) || !qword(AB, inv, w[5]))
            return Key::Own;
        const bool elig = ek && g.group_i[o] >= 0 && g.group_i[o] == g.group_j[s];
        const uint64_t bt = elig && g.b_src ? b_id[static_cast<size_t>(s)] : 0;
        w[0] = static_cast<uint64_t>(e + 2200) | (uint64_t(elig) << 12) | (a_id[static_cast<size_t>(o)] << 13) |
               (bt << 38);
        w[1] = h_id[static_cast<size_t>(o)] | (h_id[static_cast<size_t>(s)] << 32);
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
            const uint64_t t[W] = {w[0],
                                   h_id[static_cast<size_t>(s)] | (h_id[static_cast<size_t>(o)] << 32),
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
        return Key::Shared;
    };
    FamilyPlan &f = plan->fam[0];
    int64_t peak = 0;
    constexpr int64_t kUnbounded = std::numeric_limits<int64_t>::max();
    const Outcome out = pair_reuse::census<W>(f, POLICY, kUnbounded, peak, key);
    plan->census_peak_bytes = maps_bytes + peak;
    if (out == Outcome::Poor) return no(pair_reuse::cap_reason(n * n));
    if (out != Outcome::Ok) return no("too many classes to code");
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
    put(s, pair_reuse::cap_key());
    put(s, static_cast<int64_t>(g.thresholds.size()));
    for (double t : g.thresholds) put(s, t);
    put(s, g.group_i != nullptr);
    put(s, g.b_src != nullptr);
    s.append(reinterpret_cast<const char *>(g.seg_l), static_cast<size_t>(3 * g.n) * sizeof(double));
    s.append(reinterpret_cast<const char *>(g.seg_r), static_cast<size_t>(3 * g.n) * sizeof(double));
    s.append(reinterpret_cast<const char *>(g.a), static_cast<size_t>(g.n) * sizeof(double));
    s.append(reinterpret_cast<const char *>(g.edge), static_cast<size_t>(g.n) * sizeof(int32_t));
    if (g.group_i) {
        s.append(reinterpret_cast<const char *>(g.group_i), static_cast<size_t>(g.n) * sizeof(int64_t));
        s.append(reinterpret_cast<const char *>(g.group_j), static_cast<size_t>(g.n) * sizeof(int64_t));
    }
    if (g.b_src) s.append(reinterpret_cast<const char *>(g.b_src), static_cast<size_t>(g.n) * sizeof(double));
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
    int64_t a = 0;
    for (const FamilyPlan &f : p.fam)
        a += pair_reuse::plan_allowance(f.spec.rows * f.spec.cols, kept_bytes_per_class(degree));
    return a;
}

void set_plan_allowance(int64_t bytes) { g_plan_allowance.store(bytes <= 0 ? 0 : bytes); }
void set_class_limit(int64_t classes) { g_class_limit.store(classes < 0 ? 0 : classes); }
void set_small_deck_segs(int64_t segs) { g_small.store(segs < 0 ? SMALL_DECK_SEGS : segs); }
void set_length_bits(int b) { g_length_bits.store(b < 0 ? LENGTH_BITS : b); }
int length_bits() { return g_length_bits.load(); }
int64_t small_deck_segs() { return g_small.load(); }

std::shared_ptr<const Plan> plan_for(const Geometry &g, bool *built) {
    if (g.n <= 0 || !g.seg_l || !g.seg_r || !g.a || !g.edge)
        throw std::invalid_argument("bspline pair_reuse: an empty or mis-shaped geometry");
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

int64_t same_edge_pairs(const Plan &p) { return p.fam.empty() ? 0 : p.fam[0].near_unlisted; }

int64_t near_pairs(const Plan &p) {
    return p.fam.empty() ? 0 : p.fam[0].near - p.fam[0].near_unlisted;
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
    const int64_t n = f.spec.cols;
    if (f.map[0].empty()) throw std::invalid_argument("bspline pair_reuse: a declined plan has no window");
    if (w.d1 < 2 || w.d1 > 4) throw std::invalid_argument("bspline pair_reuse: d + 1 must be 2, 3 or 4");
    if (!(0 <= w.r0 && w.r0 < w.r1 && w.r1 <= n && 0 <= w.o0 && w.o0 < w.o1 && w.o1 <= n &&
          w.o0 <= w.r0 && w.r1 <= w.o1))
        throw std::invalid_argument("bspline pair_reuse: the window's rows must lie on its own edge");
}

void window_tile(const Window &w, const int64_t *rows, size_t nr, const int64_t *cols, size_t nc, cd *out) {
    const FamilyPlan &f = w.plan->fam[0];
    const int d1 = w.d1;
    const int64_t dd = int64_t(d1) * d1, no = w.o1 - w.o0, m = w.r1 - w.r0;
    const size_t plane = nr * nc;
    const bool probe = g_window_probe.load(std::memory_order_relaxed) == 1;
    pair_reuse::gather_pairs(f, 0, rows, static_cast<ptrdiff_t>(nr), cols, static_cast<ptrdiff_t>(nc),
                             [&](ptrdiff_t io, ptrdiff_t is, int32_t c, int rel) {
        cd *o = out + io * static_cast<ptrdiff_t>(nc) + is;
        const int64_t s = cols[is];
        if (s >= w.o0 && s < w.o1) {
            // The own edge: the fill's own moments, or the offset route's zero.
            if (w.own) {
                const cd *v = w.own + (rows[io] - w.r0) * no + (s - w.o0);
                for (int64_t pq = 0; pq < dd; pq++) o[pq * plane] = v[pq * m * no];
            } else {
                for (int64_t pq = 0; pq < dd; pq++) o[pq * plane] = cd(0.0, 0.0);
            }
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

void remainder_blocks(const RemainderGeometry &g, std::vector<int64_t> &ta, std::vector<int64_t> &tb,
                      std::vector<int64_t> &sa, std::vector<int64_t> &sb) {
    ta.clear();
    tb.clear();
    sa.clear();
    sb.clear();
    const int64_t E = g.n_edges, n = g.edge_seg0[E];
    if (E < 2 || g.edge_seg0[0] != 0) return;
    for (int64_t e = 0; e < E; e++)
        if (g.edge_seg0[e + 1] <= g.edge_seg0[e]) throw std::invalid_argument("bspline pair_reuse: an empty edge");
    // Per edge: its end points and an id of its segmentation (each inner
    // boundary's arc over the edge's length, on the key's step, and each
    // segment's radius): two edges of one id and one end-point geometry
    // carry their nodes at the same places.
    std::vector<double> p0(static_cast<size_t>(3 * E)), p1(static_cast<size_t>(3 * E));
    std::vector<uint64_t> pat(static_cast<size_t>(E)), nseg(static_cast<size_t>(E));
    std::map<std::vector<uint64_t>, uint64_t> pats;
    const double one = std::ldexp(1.0, KEY_BITS);
    for (int64_t e = 0; e < E; e++) {
        const int64_t s0 = g.edge_seg0[e], s1 = g.edge_seg0[e + 1];
        nseg[static_cast<size_t>(e)] = static_cast<uint64_t>(s1 - s0);
        for (int c = 0; c < 3; c++) {
            p0[static_cast<size_t>(3 * e + c)] = g.seg_l[3 * s0 + c];
            p1[static_cast<size_t>(3 * e + c)] = g.seg_r[3 * (s1 - 1) + c];
        }
        std::vector<uint64_t> v;
        double L = 0.0;
        for (int c = 0; c < 3; c++) L += (p1[3 * e + c] - p0[3 * e + c]) * (p1[3 * e + c] - p0[3 * e + c]);
        L = std::sqrt(L);
        for (int64_t k = s0; k < s1; k++) {
            double t = 0.0;
            for (int c = 0; c < 3; c++) t += (g.seg_r[3 * k + c] - p0[3 * e + c]) * (g.seg_r[3 * k + c] - p0[3 * e + c]);
            uint64_t w = 0;
            if (!(L > 0.0) || !qword(std::sqrt(t) / L, one, w)) w = ~uint64_t(0);
            v.push_back(w);
            v.push_back(bits(g.a[k]));
        }
        pat[static_cast<size_t>(e)] = pats.emplace(std::move(v), pats.size()).first->second;
    }
    double scale = 0.0;
    for (int64_t i = 0; i < 3 * n; i++) scale = std::max({scale, std::fabs(g.seg_l[i]), std::fabs(g.seg_r[i])});
    if (!(scale > 0.0)) return;
    int e_s;
    double inv;
    quantum(scale, KEY_BITS, e_s, inv);
    auto dist = [](const double *u, const double *v) {
        const double dx = u[0] - v[0], dy = u[1] - v[1], dz = u[2] - v[2];
        return std::sqrt(dx * dx + dy * dy + dz * dz);
    };
    static constexpr int RW = 12;  // static: MSVC (C2131) will not read a local constexpr inside a [&] lambda as a constant
    // The ordered block (x, y): the edges' counts and segmentations, the six
    // distances of (x0, x1, y0, y1) and their four heights above the plane,
    // on a step of 2^-KEY_BITS of the structure's size. Read from (y, x) it
    // is the same block transposed (the kernel's reversal rule): the key is
    // the smaller of the two forms, code 1 the transposed one.
    auto key = [&](int, int64_t x, int64_t y, uint64_t *w, uint8_t &code) -> Key {
        code = 0;
        const double *P[4] = {&p0[static_cast<size_t>(3 * x)], &p1[static_cast<size_t>(3 * x)],
                              &p0[static_cast<size_t>(3 * y)], &p1[static_cast<size_t>(3 * y)]};
        uint64_t d[6], h[4];
        int q = 0;
        for (int i = 0; i < 4; i++)
            for (int j = i + 1; j < 4; j++)
                if (!qword(dist(P[i], P[j]), inv, d[q++])) return Key::Own;
        for (int i = 0; i < 4; i++)
            if (!qword(P[i][2] - g.ground_z, inv, h[i])) return Key::Own;
        const uint64_t self = x == y ? uint64_t(1) << 62 : 0;
        const uint64_t f0[RW] = {self | (nseg[static_cast<size_t>(x)] << 31) | nseg[static_cast<size_t>(y)],
                                 pat[static_cast<size_t>(x)] | (pat[static_cast<size_t>(y)] << 32),
                                 d[0], d[1], d[2], d[3], d[4], d[5], h[0], h[1], h[2], h[3]};
        // (y0, y1, x0, x1): d01 <- d23, d02 <- d02, d03 <- d12, d12 <- d03,
        // d13 <- d13, d23 <- d01.
        const uint64_t f1[RW] = {self | (nseg[static_cast<size_t>(y)] << 31) | nseg[static_cast<size_t>(x)],
                                 pat[static_cast<size_t>(y)] | (pat[static_cast<size_t>(x)] << 32),
                                 d[5], d[1], d[3], d[2], d[4], d[0], h[2], h[3], h[0], h[1]};
        const bool swap = std::lexicographical_compare(f1, f1 + RW, f0, f0 + RW);
        std::memcpy(w, swap ? f1 : f0, sizeof f0);
        code = swap ? 1 : 0;
        return Key::Shared;
    };
    FamilyPlan f;
    f.spec.name = "remainder";
    f.spec.key_words = RW;
    f.spec.code_bits = 1;
    f.spec.n_blocks = 1;
    f.spec.rows = f.spec.cols = E;
    f.spec.value_bytes = 0;
    pair_reuse::Policy pol;
    pol.capped = false;
    pol.rep_rows = 0;
    pol.measured = false;
    f.spec.keep_reps = true;
    int64_t peak = 0;
    if (pair_reuse::census<RW>(f, pol, std::numeric_limits<int64_t>::max(), peak, key) != Outcome::Ok) return;
    // Block (x, y), x <= y, of class c copies its representative (s, t) when
    // their codes agree, else (t, s), and (y, x) copies the transpose of
    // that: a block whose two forms key alike could take either, and the
    // kernel's reversal rule needs the pair. The representative and its
    // transpose are the sources, filled.
    const int32_t *map = f.map[0].data();
    std::vector<int64_t> src(static_cast<size_t>(E * E), -1);
    for (int64_t x = 0; x < E; x++)
        for (int64_t y = x; y < E; y++) {
            const int32_t e = map[x * E + y];
            const int64_t c = e >> 1;
            const int64_t rep = f.rep_flat[static_cast<size_t>(c)];
            int64_t s = rep / E, t = rep % E;
            if ((e & 1) != f.rep_code[static_cast<size_t>(c)]) std::swap(s, t);
            if ((s == x && t == y) || (s == y && t == x)) continue;
            src[static_cast<size_t>(x * E + y)] = s * E + t;
            src[static_cast<size_t>(y * E + x)] = t * E + s;
        }
    for (int64_t q = 0; q < E * E; q++) {
        const int64_t v = src[static_cast<size_t>(q)];
        if (v < 0) continue;
        ta.push_back(q / E);
        tb.push_back(q % E);
        sa.push_back(v / E);
        sb.push_back(v % E);
    }
}

void wire_map(const Plan &p, const int32_t *wire_of_seg, int64_t n_wires, int64_t *out) {
    std::fill(out, out + n_wires * n_wires * 3, int64_t(0));
    if (p.fam.empty() || p.fam[0].map[0].empty()) return;
    pair_reuse::wire_map(p.fam[0], wire_of_seg, wire_of_seg, n_wires, out);
}

pair_reuse::Counters cache_counters() { return pair_reuse::counters(BASIS); }

void clear_cache() { pair_reuse::clear_cache(BASIS); }

}  // namespace reuse
}  // namespace bspline
}  // namespace mw
