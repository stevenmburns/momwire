// razor2p's fill-pair reuse (momwire#1971): see `_razor_pair_reuse_core.h`.
// razor's adapter onto the common core (`_pair_reuse_core.h`, momwire#2063):
// its keys, the evaluation of its representatives, the transforms its gathers
// apply, and its policy. Its own TU, so no kernel's codegen moves with it
// (momwire#1187): it only CALLS the shipped kernels (`seg_moments`,
// `field_proj_batch`, `moment_reduce`), across a TU boundary; the core's
// census and gather templates are instantiated here, beside the keys.
#include "_razor_pair_reuse_core.h"

#include "_fma_inline.h"
#include "_razor_core.h"
#include "_somm_proj_core.h"

#include <algorithm>
#include <atomic>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <utility>

namespace mw {
namespace razor {
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

std::atomic<int> g_probe{PROBE_NONE};
std::atomic<int64_t> g_class_limit[2] = {{0}, {0}};
std::atomic<int64_t> g_plan_allowance{0};  // 0: the derived allowance

// What a class's values cost per k: M0, M1 (or Q0, Q1), two complex.
constexpr int64_t VALUE_BYTES = 2 * 16;

const pair_reuse::Policy POLICY = [] {
    pair_reuse::Policy p;
    p.capped = true;  // `pair_reuse::class_cap`
    p.rep_rows = REP_ROWS;
    return p;
}();

void build(const Geometry &g, Plan &plan_ref) {
    Plan *const plan = &plan_ref;
    auto no = [&](const std::string &why) { pair_reuse::decline(*plan, why); };
    const int64_t n_seg = g.seg_h.shape(0);
    const int n_blocks = g.has_image ? 2 : 1;
    {
        FamilySpec d;
        d.name = "direct";
        d.key_words = 4;
        d.code_bits = 1;
        d.n_blocks = n_blocks;
        d.rows = d.cols = n_seg;
        d.value_bytes = VALUE_BYTES;
        d.near_singles = true;
        d.class_limit = class_limit(0);
        plan->fam.push_back(FamilyPlan());
        plan->fam.back().spec = d;
    }
    if (g.remainder) {
        FamilySpec q;
        q.name = "remainder";
        q.key_words = 9;
        q.code_bits = 2;
        q.n_blocks = 1;
        q.rows = q.cols = n_seg;
        q.value_bytes = VALUE_BYTES;
        q.near_singles = false;
        q.class_limit = class_limit(1);
        plan->fam.push_back(FamilyPlan());
        plan->fam.back().spec = q;
    }
    if (n_seg > MAX_SEGS) return no("too many segments to plan");
    // The allowance (`plan_allowance`, the header's): the maps are known
    // before any census (4 B per pair per block, the remainder's included),
    // so a deck whose maps alone pass it (only under the test hook) declines
    // before allocating them. The census is bounded by the class cap alone
    // (its `room` unbounded); its tables are counted into its peak.
    const int64_t allowance = plan_allowance(*plan);
    const int64_t nn = n_seg * n_seg;
    const int64_t maps_bytes = pair_reuse::maps_bytes(*plan);
    if (maps_bytes > allowance) return no(pair_reuse::allowance_reason(allowance));
    constexpr int64_t kUnbounded = std::numeric_limits<int64_t>::max();
    auto capped = [](const FamilyPlan &f) {
        return pair_reuse::cap_reason(int64_t(f.spec.n_blocks) * f.spec.rows * f.spec.cols);
    };
    const bool ek = g.group_i.size() > 0 && g.group_j[0].size() > 0;
    const bool has_b = ek && g.b_obs.size() > 0;
    const int pr = probe();

    // Types: the source's kernel radius, and under the EK its EK radius; the
    // observer's tube.
    Ids a_ids, ae_ids, b_ids, h_ids;
    std::vector<int64_t> a_id[2], ae_id[2], b_id(static_cast<size_t>(n_seg), 0);
    std::vector<uint64_t> h_id(static_cast<size_t>(n_seg), 0);
    for (int64_t s = 0; s < n_seg; s++)
        h_id[static_cast<size_t>(s)] = static_cast<uint64_t>(h_ids.id(bits(g.seg_h(s))));
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
        b_ids.m.size() >= (size_t(1) << 25) || h_ids.m.size() >= (size_t(1) << 25))
        return no("too many segment types to key");

    auto direct_key = [&](int b, int64_t o, int64_t s, uint64_t *w, uint8_t &code) -> Key {
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
        if (!(std::max(av, rho) >= NEAR_LENGTHS * near)) return Key::Own;
        const double D = std::max(std::max(av - 0.5 * h, rho), g.a(s));
        if (!(D > 0.0) || !std::isfinite(D)) return Key::Own;
        int e;
        double inv;
        quantum(D, KEY_BITS, e, inv);
        if (!qword(pr == PROBE_NO_FOLD ? v : av, inv, w[0]) || !qword(rho, inv, w[1]))
            return Key::Own;
        w[2] = h_id[static_cast<size_t>(s)];
        if (v < 0.0 && w[0] != 0 && pr != PROBE_NO_FLIP && pr != PROBE_NO_FOLD) code = 1;
        const bool elig = ek && g.group_i(o) >= 0 && g.group_i(o) == g.group_j[b](s);
        const uint64_t typ = static_cast<uint64_t>(elig ? ae_id[b][static_cast<size_t>(s)]
                                                        : a_id[b][static_cast<size_t>(s)]);
        const uint64_t bt = elig ? static_cast<uint64_t>(b_id[static_cast<size_t>(o)]) : 0;
        w[3] = static_cast<uint64_t>(e + 1100) | (uint64_t(elig) << 11) | (typ << 12) | (bt << 37);
        if (pr == PROBE_NO_TYPE) w[3] = static_cast<uint64_t>(e + 1100);
        return Key::Shared;
    };
    FamilyPlan &dfam = plan->fam[DIRECT];
    int64_t peak = 0;
    const Outcome d_out = pair_reuse::census<4>(dfam, POLICY, kUnbounded, peak, direct_key);
    plan->census_peak_bytes = maps_bytes + peak;
    if (d_out == Outcome::Poor) return no(capped(dfam));
    if (d_out != Outcome::Ok) return no("too many classes to code");

    if (g.remainder) {
        const double gz = g.ground_z;
        const double one = std::ldexp(1.0, KEY_BITS);
        auto q_key = [&](int, int64_t o, int64_t s, uint64_t *w, uint8_t &code) -> Key {
            const double zo = g.cent(o, 2) - gz, zs = g.cent(s, 2) - gz;
            const double hs = zo + zs;
            const double dx = g.cent(o, 0) - g.cent(s, 0), dy = g.cent(o, 1) - g.cent(s, 1);
            const double dd2 = dx * dx + dy * dy;
            const double ax = g.t[0](s, 0), ay = g.t[0](s, 1), tsz = g.t[0](s, 2);
            const double bx = g.t[0](o, 0), by = g.t[0](o, 1), toz = g.t[0](o, 2);
            const double h = g.seg_h(s);
            const double D = std::max(std::sqrt(dd2 + hs * hs), h);
            if (!(D > 0.0) || !std::isfinite(D)) return Key::Own;
            int e;
            double inv;
            quantum(D, KEY_BITS, e, inv);
            uint64_t qhs, qdd, qda, qdb, qab, qtsz, qtoz;
            if (!qword(hs, inv, qhs) || !qword(std::sqrt(dd2), inv, qdd) ||
                !qword(dx * ax + dy * ay, inv, qda) || !qword(dx * bx + dy * by, inv, qdb) ||
                !qword(ax * bx + ay * by, one, qab) || !qword(tsz, one, qtsz) ||
                !qword(toz, one, qtoz))
                return Key::Own;
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
            w[3] = h_id[static_cast<size_t>(s)];
            for (int j = 0; j < 5; j++) w[4 + j] = static_cast<uint64_t>(best[j]);
            return Key::Shared;
        };
        // The census's peak counts what the direct family holds now (its
        // maps, groups and flags, and its values) and the remainder's maps.
        const int64_t held = pair_reuse::held_bytes(*plan) + 4 * nn;
        int64_t qpeak = 0;
        const Outcome q_out = pair_reuse::census<9>(plan->fam[REMAINDER], POLICY, kUnbounded, qpeak, q_key);
        plan->census_peak_bytes = std::max(plan->census_peak_bytes, held + qpeak);
        if (q_out == Outcome::Poor) return no(capped(plan->fam[REMAINDER]));
        if (q_out != Outcome::Ok) return no("too many remainder classes to code");
    }
    // The exact count, the groups' observer and source lists included.
    plan->budget_bytes = pair_reuse::held_bytes(*plan);
    if (plan->budget_bytes > allowance) return no(pair_reuse::allowance_reason(allowance));
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
    put(s, plan_allowance_override());
    put(s, pair_reuse::cap_compression());
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

std::shared_ptr<const Plan> plan_for(const Geometry &g, bool *built) {
    check_geometry(g);
    return pair_reuse::plan_for(
        BASIS, fingerprint_of(g), [&g](Plan &p) { build(g, p); }, built);
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
void set_plan_allowance(int64_t bytes) { g_plan_allowance.store(bytes <= 0 ? 0 : bytes); }
int64_t plan_allowance_override() { return g_plan_allowance.load(); }
int64_t kept_bytes_per_class(const pair_reuse::FamilySpec &f) {
    // `class_bytes` (values, group entry, code) and, at most, one observer
    // row and one source in its group's lists (`Group::obs`, `Group::src`):
    // a class adds its representative's row and source to its group at most
    // once each.
    return pair_reuse::class_bytes(f) + static_cast<int64_t>(2 * sizeof(int32_t));
}
int64_t plan_allowance(const pair_reuse::Plan &p) {
    const int64_t b = g_plan_allowance.load();
    if (b > 0) return b;
    int64_t a = 0;
    for (const FamilyPlan &f : p.fam)
        for (int blk = 0; blk < f.spec.n_blocks; blk++)
            a += pair_reuse::plan_allowance(f.spec.rows * f.spec.cols, kept_bytes_per_class(f.spec));
    return a;
}
int probe() { return g_probe.load(); }

CacheCounters cache_counters() { return pair_reuse::counters(BASIS); }

void clear_cache() { pair_reuse::clear_cache(BASIS); }

void eval_direct(const Plan &p, const Geometry &g, view<const double, 1> xg,
                 view<const double, 1> wg, double k, uintptr_t cancel_flag, Values &out) {
    out.k = k;
    out.d.assign(static_cast<size_t>(2 * p.fam[DIRECT].classes), cd(0.0, 0.0));
    const bool ek = g.group_i.size() > 0 && g.group_j[0].size() > 0;
    const bool has_b = ek && g.b_obs.size() > 0;
    std::vector<double> obs, p0, t, h, a, aek, bo;
    std::vector<int64_t> gi, gj;
    std::vector<cd> m0, m1;
    for (const Group &G : p.fam[DIRECT].groups) {
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
            out.d[2 * static_cast<size_t>(G.cls[q])] = m0[static_cast<size_t>(G.at[q])];
            out.d[2 * static_cast<size_t>(G.cls[q]) + 1] = m1[static_cast<size_t>(G.at[q])];
        }
    }
    out.has_d = true;
}

void eval_remainder(const Plan &p, const Geometry &g, const somm_proj::GridView &grid,
                    size_t n_reg, const double *rem_src, const double *rem_t_src,
                    const double *W, ptrdiff_t n_qp, double k, uintptr_t cancel_flag,
                    Values &out) {
    if (!has_remainder(p)) throw std::invalid_argument("razor pair_reuse: the plan has no remainder");
    out.q0.assign(static_cast<size_t>(p.fam[REMAINDER].classes), cd(0.0, 0.0));
    out.q1.assign(static_cast<size_t>(p.fam[REMAINDER].classes), cd(0.0, 0.0));
    const ptrdiff_t n_seg = static_cast<ptrdiff_t>(n_segs(p)), q = n_qp;
    std::vector<double> pts, tans, src, tsrc, Wg;
    std::vector<cd> proj, mom;
    for (size_t gi = 0; gi < p.fam[REMAINDER].groups.size(); gi++) {
        const Group &G = p.fam[REMAINDER].groups[gi];
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
    if (!has_remainder(p) || group >= p.fam[REMAINDER].groups.size()) throw std::invalid_argument("razor pair_reuse: no such group");
    if (out.q0.size() != static_cast<size_t>(p.fam[REMAINDER].classes)) {
        out.q0.assign(static_cast<size_t>(p.fam[REMAINDER].classes), cd(0.0, 0.0));
        out.q1.assign(static_cast<size_t>(p.fam[REMAINDER].classes), cd(0.0, 0.0));
    }
    const Group &G = p.fam[REMAINDER].groups[group];
    for (size_t i = 0; i < G.cls.size(); i++) {
        const size_t at = static_cast<size_t>(G.at[i]);
        out.q0[static_cast<size_t>(G.cls[i])] = mom[2 * at];
        out.q1[static_cast<size_t>(G.cls[i])] = mom[2 * at + 1];
    }
    if (group + 1 == p.fam[REMAINDER].groups.size()) out.has_q = true;
}

void gather_direct(const Plan &p, int block, const int64_t *need, ptrdiff_t n_need,
                   view<const double, 1> seg_h, const Values &v, cd *m0, cd *m1) {
    if (!v.has_d) throw std::invalid_argument("razor pair_reuse: no direct values");
    const ptrdiff_t ns = static_cast<ptrdiff_t>(n_segs(p));
    const cd *d = v.d.data();
    pair_reuse::gather(p.fam[DIRECT], block, need, n_need, [&](ptrdiff_t i, ptrdiff_t s, int32_t c, int rel) {
        const cd a0 = d[2 * static_cast<size_t>(c)], a1 = d[2 * static_cast<size_t>(c) + 1];
        m0[i * ns + s] = a0;
        if (rel) {
            // The source reversed: M1 -> h M0 - M1.
            const double h = seg_h(s);
            m1[i * ns + s] = cd(h * a0.real() - a1.real(), h * a0.imag() - a1.imag());
        } else {
            m1[i * ns + s] = a1;
        }
    });
}

void gather_remainder(const Plan &p, const int64_t *seg, const double *sign, ptrdiff_t m,
                      view<const double, 1> seg_h, const Values &v, cd *out) {
    if (!v.has_q) throw std::invalid_argument("razor pair_reuse: no remainder values");
    if (!has_remainder(p)) throw std::invalid_argument("razor pair_reuse: the plan has no remainder");
    const ptrdiff_t ns = static_cast<ptrdiff_t>(n_segs(p));
    const cd *q0 = v.q0.data(), *q1 = v.q1.data();
    pair_reuse::gather(p.fam[REMAINDER], 0, seg, m, [&](ptrdiff_t r, ptrdiff_t s, int32_t c, int rel) {
        const bool neg_row = sign[r] < 0.0;
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
        cd *o = out + 2 * (r * ns + s);
        o[0] = cd(a0r, a0i);
        o[1] = cd(a1r, a1i);
    });
}

// ---------------------------------------------------------------------------
// momwire#2144: a window's entries from the class values, tile by tile.
//
// WHY IT IS THE SAME ARITHMETIC. Under the NEC-5 path rule basis row r's two
// path points are the centroids of its wings sa[r], sb[r], and its T2 reads
// the same two centroids (plus = sb[r]; minus = sa[r], dropped on a grounded
// row). So entry (r, j) reads exactly four pairs, {sa[r], sb[r]} x {sa[j],
// sb[j]}, each one's (M0, M1) as `gather_direct` writes it (the class value,
// M1 -> h M0 - M1 on a reversed source), and then the per-entry expressions
// of `assemble_t1` / `assemble_t1_weighted` (window kind 1) and the fused
// `t2_rows`, spelled here term for term in their order. Every entry is
// independent of every other in all three kernels, and the build does not
// contract (-ffp-contract=off), so neither the tiling nor the loop order
// moves a bit: the reuse-on fill's Z is the three-kernel route's exactly
// (`test_razor_reuse_rows_2144.py` holds that against `verify`, which keeps
// the three kernels).
//
// WHY TILES. The three-kernel route writes an (n_need, n_seg) M0 and M1 per
// window, reads them back for T1, writes T1, and reads M0 and T1 back for
// T2: at razor's window budget (`_CHUNK_ELEMS`, ~470 rows on a 2,000-basis
// deck) each plane is tens of MB, so the assembly ran at memory speed and
// cost more than the reused kernel did (momwire#2144's profile). A tile of
// TILE_ROWS x TILE_COLS entries gathers its own (<= 2 TILE_ROWS) x (<= 2
// TILE_COLS) pairs, about 50 kB, and assembles from cache. Neighbouring
// tiles share their edge pairs, so a pair is gathered about (1 + 1/TILE_ROWS)
// (1 + 1/TILE_COLS) times, against once per window before.
namespace {

constexpr ptrdiff_t TILE_ROWS = 16, TILE_COLS = 64;

// Segment s's index in the tile's list, appended on first sight (mark[s] is
// -1 outside the tile, and is reset by the caller after it).
inline int32_t local_index(std::vector<int32_t> &mark, std::vector<int64_t> &list, int64_t s) {
    int32_t &m = mark[static_cast<size_t>(s)];
    if (m < 0) {
        m = static_cast<int32_t>(list.size());
        list.push_back(s);
    }
    return m;
}

}  // namespace

void assemble_rows(const Plan &p, const Values &v, int block, ptrdiff_t lo, ptrdiff_t hi,
                   const RowTables &t, view<const double, 1> seg_h, view<cd, 2> out) {
    const char *who = "razor pair_reuse assemble_rows";
    if (!v.has_d) throw std::invalid_argument("razor pair_reuse: no direct values");
    const FamilyPlan &f = p.fam[DIRECT];
    if (block < 0 || block >= f.spec.n_blocks || f.map[block].empty())
        throw std::invalid_argument("pair_reuse: no such block");
    const ptrdiff_t n = t.s_a.shape(0), n_seg = static_cast<ptrdiff_t>(n_segs(p));
    if (lo < 0 || hi > n || lo > hi) throw std::invalid_argument(std::string(who) + ": rows out of range");
    if (out.shape(0) != hi - lo || out.shape(1) != n)
        throw std::invalid_argument(std::string(who) + ": out must be (hi - lo, n_basis)");
    if (t.s_b.shape(0) != n || t.h_a.shape(0) != n || t.h_b.shape(0) != n || t.q_a.shape(0) != n ||
        t.q_b.shape(0) != n || t.fall_a.shape(0) != n || t.fall_b.shape(0) != n ||
        t.wts.shape(0) != 2 * n || seg_h.shape(0) != n_seg)
        throw std::invalid_argument(std::string(who) + ": per-basis arrays disagree on n_basis");
    if (t.weighted) {
        if (t.obs_tans.shape(0) != 2 * n || t.obs_tans.shape(1) != 3 || t.src_t.shape(0) != n_seg ||
            t.src_t.shape(1) != 3 || t.sig_a.shape(0) != n || t.sig_b.shape(0) != n)
            throw std::invalid_argument(std::string(who) + ": the weighted tables are mis-shaped");
    } else if (t.tans.shape(0) != 2 * n || t.tans.shape(1) != 3 || t.td_a.shape(0) != 3 ||
               t.td_a.shape(1) != n || t.td_b.shape(0) != 3 || t.td_b.shape(1) != n) {
        throw std::invalid_argument(std::string(who) + ": the tangent tables are mis-shaped");
    }
    if (!(t.c_phi_imag != 0.0)) throw std::invalid_argument(std::string(who) + ": c_Phi must be nonzero");
    for (ptrdiff_t j = 0; j < n; j++)
        if (t.s_a(j) < 0 || t.s_a(j) >= n_seg || t.s_b(j) < 0 || t.s_b(j) >= n_seg)
            throw std::invalid_argument(std::string(who) + ": wing segment index out of range");
    if (lo == hi || n == 0) return;

    // `assemble_t1`'s per-column reciprocals, the same division once per
    // column instead of once per entry.
    std::vector<double> inv_ha(static_cast<size_t>(n)), inv_hb(static_cast<size_t>(n));
    for (ptrdiff_t j = 0; j < n; j++) {
        inv_ha[static_cast<size_t>(j)] = 1.0 / t.h_a(j);
        inv_hb[static_cast<size_t>(j)] = 1.0 / t.h_b(j);
    }
    std::vector<char> grounded(static_cast<size_t>(hi - lo), 0);
    for (ptrdiff_t g = 0; g < t.grounded.shape(0); g++) {
        const int64_t row = t.grounded(g);
        if (row >= lo && row < hi) grounded[static_cast<size_t>(row - lo)] = 1;
    }

    const int32_t *map = f.map[block].data();
    const uint8_t *rc = f.rep_code.data();
    const int cb = f.spec.code_bits;
    const int32_t mask = (int32_t(1) << cb) - 1;
    const cd *d = v.d.data();
    const double a = t.c_a_imag;
    const double scl = 1.0 / t.c_phi_imag;
    const cd c2 = t.c2;
    const bool weighted = t.weighted;
    // `t2_rows`' walk: rows innermost on a window of the column-major Z.
    const bool rows_inner =
        std::llabs(static_cast<long long>(out.st[0])) < std::llabs(static_cast<long long>(out.st[1]));
    const ptrdiff_t n_rt = (hi - lo + TILE_ROWS - 1) / TILE_ROWS;
    const ptrdiff_t n_ct = (n + TILE_COLS - 1) / TILE_COLS;
    const ptrdiff_t n_tiles = n_rt * n_ct;

#pragma omp parallel
    {
        std::vector<int32_t> mark(static_cast<size_t>(n_seg), -1);
        std::vector<int64_t> osegs, ssegs;
        osegs.reserve(2 * TILE_ROWS);
        ssegs.reserve(2 * TILE_COLS);
        int32_t ra[TILE_ROWS], rb[TILE_ROWS], ca[TILE_COLS], cbl[TILE_COLS];
        std::vector<cd> m0, m1, m0w;
#pragma omp for schedule(static)
        for (ptrdiff_t tile = 0; tile < n_tiles; tile++) {
            const ptrdiff_t r0 = lo + (tile / n_ct) * TILE_ROWS, r1 = std::min(hi, r0 + TILE_ROWS);
            const ptrdiff_t j0 = (tile % n_ct) * TILE_COLS, j1 = std::min(n, j0 + TILE_COLS);
            osegs.clear();
            for (ptrdiff_t r = r0; r < r1; r++) {
                ra[r - r0] = local_index(mark, osegs, t.s_a(r));
                rb[r - r0] = local_index(mark, osegs, t.s_b(r));
            }
            for (int64_t s : osegs) mark[static_cast<size_t>(s)] = -1;
            ssegs.clear();
            for (ptrdiff_t j = j0; j < j1; j++) {
                ca[j - j0] = local_index(mark, ssegs, t.s_a(j));
                cbl[j - j0] = local_index(mark, ssegs, t.s_b(j));
            }
            for (int64_t s : ssegs) mark[static_cast<size_t>(s)] = -1;
            const ptrdiff_t no = static_cast<ptrdiff_t>(osegs.size());
            const ptrdiff_t ns = static_cast<ptrdiff_t>(ssegs.size());

            // `gather_direct`'s visit, on the tile's pairs.
            const size_t nt = static_cast<size_t>(no * ns);
            if (m0.size() < nt) {
                m0.resize(nt);
                m1.resize(nt);
                if (weighted) m0w.resize(nt);
            }
            for (ptrdiff_t io = 0; io < no; io++) {
                const int32_t *row = map + osegs[static_cast<size_t>(io)] * n_seg;
                for (ptrdiff_t is = 0; is < ns; is++) {
                    const int64_t s = ssegs[static_cast<size_t>(is)];
                    const int32_t e = row[s];
                    const int32_t c = e >> cb;
                    const cd a0 = d[2 * static_cast<size_t>(c)], a1 = d[2 * static_cast<size_t>(c) + 1];
                    const size_t i = static_cast<size_t>(io * ns + is);
                    m0[i] = a0;
                    if ((e & mask) ^ rc[c]) {
                        // The source reversed: M1 -> h M0 - M1.
                        const double h = seg_h(s);
                        m1[i] = cd(h * a0.real() - a1.real(), h * a0.imag() - a1.imag());
                    } else {
                        m1[i] = a1;
                    }
                    if (weighted) {
                        // The solve core's `M0c *= w_Phi`, w_Phi = C2: numpy's
                        // fused complex product.
                        const double ar = a0.real(), ai = a0.imag();
                        m0w[i] = cd(mw_fma::exact(ar, c2.real(), -(ai * c2.imag())),
                                    mw_fma::exact(ar, c2.imag(), ai * c2.real()));
                    }
                }
            }
            const cd *M0 = m0.data(), *M1 = m1.data(), *MW = weighted ? m0w.data() : m0.data();

            auto entry = [&](ptrdiff_t r, ptrdiff_t j) {
                const ptrdiff_t la = ca[j - j0], lb = cbl[j - j0];
                const ptrdiff_t qrow[2] = {ra[r - r0] * ns, rb[r - r0] * ns};
                const double inv_a = inv_ha[static_cast<size_t>(j)];
                const double inv_b = inv_hb[static_cast<size_t>(j)];
                const bool fall_j_a = t.fall_a(j), fall_j_b = t.fall_b(j);
                double xr, xi;
                if (!weighted) {
                    // `assemble_t1`.
                    const double a0 = t.td_a(0, j), a1 = t.td_a(1, j), a2 = t.td_a(2, j);
                    const double b0 = t.td_b(0, j), b1 = t.td_b(1, j), b2 = t.td_b(2, j);
                    double acc_re = 0.0, acc_im = 0.0;
                    for (ptrdiff_t pp = 0; pp < 2; pp++) {
                        const ptrdiff_t o = 2 * r + pp, q = qrow[pp];
                        std::complex<double> ma = M1[q + la] * inv_a;
                        if (fall_j_a) ma = M0[q + la] - ma;
                        std::complex<double> mb = M1[q + lb] * inv_b;
                        if (fall_j_b) mb = M0[q + lb] - mb;
                        const double dot_a = t.tans(o, 0) * a0 + t.tans(o, 1) * a1 + t.tans(o, 2) * a2;
                        const double dot_b = t.tans(o, 0) * b0 + t.tans(o, 1) * b1 + t.tans(o, 2) * b2;
                        const double wo = t.wts(o);
                        const double gre = dot_a * ma.real() + dot_b * mb.real();
                        const double gim = dot_a * ma.imag() + dot_b * mb.imag();
                        acc_re += wo * gre;
                        acc_im += wo * gim;
                    }
                    xr = acc_re;
                    xi = acc_im;
                } else {
                    // `assemble_t1_weighted` under WINDOW_RULE_CONSTANT_MIRROR:
                    // `razor_a_window` is C2 times the mirrored tangent dot.
                    const int64_t ja = t.s_a(j), jb = t.s_b(j);
                    const double sg_a = t.sig_a(j), sg_b = t.sig_b(j);
                    const double atx = t.src_t(ja, 0), aty = t.src_t(ja, 1), atz = t.src_t(ja, 2);
                    const double btx = t.src_t(jb, 0), bty = t.src_t(jb, 1), btz = t.src_t(jb, 2);
                    std::complex<double> acc(0.0, 0.0);
                    for (ptrdiff_t pp = 0; pp < 2; pp++) {
                        const ptrdiff_t o = 2 * r + pp, q = qrow[pp];
                        std::complex<double> ma = M1[q + la] * inv_a;
                        if (fall_j_a) ma = M0[q + la] - ma;
                        std::complex<double> mb = M1[q + lb] * inv_b;
                        if (fall_j_b) mb = M0[q + lb] - mb;
                        const double tx = t.obs_tans(o, 0), ty = t.obs_tans(o, 1), tz = t.obs_tans(o, 2);
                        const double td_img_a = tx * atx + ty * aty - tz * atz;
                        const double td_img_b = tx * btx + ty * bty - tz * btz;
                        const std::complex<double> wa = (c2 * td_img_a) * sg_a;
                        const std::complex<double> wb = (c2 * td_img_b) * sg_b;
                        acc += t.wts(o) * (wa * ma + wb * mb);
                    }
                    xr = acc.real();
                    xi = acc.imag();
                }
                // The fused `t2_rows`: plus is sb[r]'s row, minus sa[r]'s.
                const cd pa = MW[qrow[1] + la], pb = MW[qrow[1] + lb];
                double dar = pa.real(), dai = pa.imag(), dbr = pb.real(), dbi = pb.imag();
                if (!grounded[static_cast<size_t>(r - lo)]) {
                    const cd ma = MW[qrow[0] + la], mb = MW[qrow[0] + lb];
                    dar = dar - ma.real();
                    dai = dai - ma.imag();
                    dbr = dbr - mb.real();
                    dbi = dbi - mb.imag();
                }
                const double qa = t.q_a(j), qb = t.q_b(j);
                const double t2r = dar * qa + dbr * qb;
                const double t2i = dai * qa + dbi * qb;
                out(r - lo, j) = cd(-(a * xi) - t2i * scl, a * xr + t2r * scl);
            };
            if (rows_inner) {
                for (ptrdiff_t j = j0; j < j1; j++)
                    for (ptrdiff_t r = r0; r < r1; r++) entry(r, j);
            } else {
                for (ptrdiff_t r = r0; r < r1; r++)
                    for (ptrdiff_t j = j0; j < j1; j++) entry(r, j);
            }
        }
    }
}

void wire_map(const Plan &p, const int32_t *wire_of_seg, int64_t n_wires, int64_t *out) {
    const int64_t W = n_wires;
    std::fill(out, out + 3 * W * W * 3, int64_t(0));
    if (n_segs(p) == 0) return;
    pair_reuse::wire_map(p.fam[DIRECT], wire_of_seg, wire_of_seg, n_wires, out);
    if (has_remainder(p))
        pair_reuse::wire_map(p.fam[REMAINDER], wire_of_seg, wire_of_seg, n_wires, out + 2 * W * W * 3);
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
