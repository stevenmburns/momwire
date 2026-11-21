// razor2p's fill-pair reuse (momwire#1971): see `_razor_pair_reuse_core.h`.
// razor's adapter onto the common core (`_pair_reuse_core.h`, momwire#2063):
// its keys, the evaluation of its representatives, the transforms its gathers
// apply, and its policy. Its own TU, so no kernel's codegen moves with it
// (momwire#1187): it only CALLS the shipped kernels (`seg_moments`,
// `field_proj_batch`, `moment_reduce`), across a TU boundary; the core's
// census and gather templates are instantiated here, beside the keys.
#include "_razor_pair_reuse_core.h"

#include "_razor_core.h"
#include "_somm_proj_core.h"

#include <atomic>
#include <cstring>
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
std::atomic<int64_t> g_plan_budget{0};  // 0: PLAN_BUDGET_BYTES

// What a class's values cost per k: M0, M1 (or Q0, Q1), two complex.
constexpr int64_t VALUE_BYTES = 2 * 16;

const pair_reuse::Policy POLICY = [] {
    pair_reuse::Policy p;
    p.max_class_fraction = MAX_CLASS_FRACTION;
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
    // The byte budget (`PLAN_BUDGET_BYTES`): the maps are known before any
    // census (4 B per pair per block, the remainder's included), so a deck
    // whose maps alone pass it declines before allocating them; what is left
    // bounds each census's classes, at `pair_reuse::class_bytes` each.
    const int64_t budget = plan_budget();
    const int64_t nn = n_seg * n_seg;
    const int64_t maps_bytes = pair_reuse::maps_bytes(*plan);
    if (maps_bytes > budget) return no(pair_reuse::budget_reason(budget));
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
    const Outcome d_out = pair_reuse::census<4>(dfam, POLICY, budget - maps_bytes, peak, direct_key);
    plan->census_peak_bytes = maps_bytes + peak;
    if (d_out == Outcome::Poor) return no(pair_reuse::POOR_REASON);
    if (d_out == Outcome::OverBudget) return no(pair_reuse::budget_reason(budget));
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
        // What the direct family holds now (its maps, groups and flags, and
        // its values), plus the remainder's maps, comes out of the budget.
        const int64_t held = pair_reuse::held_bytes(*plan) + 4 * nn;
        int64_t qpeak = 0;
        const Outcome q_out =
            pair_reuse::census<9>(plan->fam[REMAINDER], POLICY, budget - held, qpeak, q_key);
        plan->census_peak_bytes = std::max(plan->census_peak_bytes, held + qpeak);
        if (q_out == Outcome::Poor) return no(pair_reuse::POOR_REASON);
        if (q_out == Outcome::OverBudget) return no(pair_reuse::budget_reason(budget));
        if (q_out != Outcome::Ok) return no("too many remainder classes to code");
    }
    // The exact count, the groups' observer and source lists included.
    plan->budget_bytes = pair_reuse::held_bytes(*plan);
    if (plan->budget_bytes > budget) return no(pair_reuse::budget_reason(budget));
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
    put(s, plan_budget());
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
void set_plan_budget(int64_t bytes) { g_plan_budget.store(bytes <= 0 ? 0 : bytes); }
int64_t plan_budget() {
    const int64_t b = g_plan_budget.load();
    return b > 0 ? b : static_cast<int64_t>(PLAN_BUDGET_BYTES);
}
int probe() { return g_probe.load(); }

CacheCounters cache_counters() { return pair_reuse::counters(BASIS); }

void clear_cache() { pair_reuse::clear_cache(BASIS); }

void eval_direct(const Plan &p, const Geometry &g, view<const double, 1> xg,
                 view<const double, 1> wg, double k, uintptr_t cancel_flag, Values &out) {
    out.k = k;
    out.d0.assign(static_cast<size_t>(p.fam[DIRECT].classes), cd(0.0, 0.0));
    out.d1.assign(static_cast<size_t>(p.fam[DIRECT].classes), cd(0.0, 0.0));
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
    const cd *d0 = v.d0.data(), *d1 = v.d1.data();
    pair_reuse::gather(p.fam[DIRECT], block, need, n_need, [&](ptrdiff_t i, ptrdiff_t s, int32_t c, int rel) {
        const cd a0 = d0[c], a1 = d1[c];
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
