// The sinusoidal fill's `pair_reuse` adapter (see `_sin_pair_reuse_core.h`).
// The keys are numpy's arithmetic written out per element in its order,
// with no contraction (-ffp-contract=off, and the MSVC pragmas below).
#if defined(_MSC_VER) && !defined(__clang__)
#pragma float_control(precise, on)
#pragma fp_contract(off)
#endif

#include "_sin_pair_reuse_core.h"

#include "_int_groups_core.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <mutex>
#include <numeric>
#include <stdexcept>
#include <utility>

#ifdef _OPENMP
#include <omp.h>
#endif
#if defined(__GLIBC__)
#include <malloc.h>
#endif

namespace mw {
namespace sinusoidal {
namespace pair_reuse {

namespace core = mw::pair_reuse;

namespace {

std::mutex g_mu;  // the settings, the last outcome and census
int64_t g_allowance = -1;  // the C++ fill's per plan; < 0: the derived one
int64_t g_small_free = SMALL_DECK_SEGS_FREE;
int64_t g_small_ground = SMALL_DECK_SEGS_GROUND;
int g_workers = 0;
int g_probe = 0;
std::string g_last;
std::vector<CensusEntry> g_census;

constexpr int64_t kNearSentinel = -1;  // below every packed exponent-and-type column
const double kUnit = static_cast<double>(int64_t(1) << KEY_BITS);

// The reflected codes under PROBE_GROUND_CODES + j.
const int8_t kProbeCodes[3][4] = {{0, 2, 1, 3}, {0, 0, 2, 2}, {0, 1, 0, 1}};

// The census's freed pages back before the plan and the fill.
void trim_heap() {
#if defined(__GLIBC__)
    malloc_trim(0);
#endif
}

inline int64_t bits_of(double x) {
    int64_t b;
    std::memcpy(&b, &x, 8);
    return b;
}
inline double double_of(int64_t b) {
    double x;
    std::memcpy(&x, &b, 8);
    return x;
}
// `np.rint(x).astype(np.int64)`
inline int64_t rint64(double x) { return static_cast<int64_t>(std::nearbyint(x)); }

// numpy's float64 maximum / minimum as `np.clip` spells them
// (`_NPY_MAX`, `_NPY_MIN`): the first operand where the compare fails.
inline double clip_max(double a, double b) { return std::isnan(a) ? a : (a > b ? a : b); }
inline double clip_min(double a, double b) { return std::isnan(a) ? a : (a < b ? a : b); }
// `np.maximum`: NaN propagates, otherwise the larger (equal: either, the
// same value up to the sign of a zero, which the operands here never mix).
inline double np_max(double a, double b) {
    if (std::isnan(a)) return a;
    if (std::isnan(b)) return b;
    return a >= b ? a : b;
}

// `np.unique(rows, axis=0, return_inverse=True)`'s inverse: each row's rank
// among the distinct rows, lexicographic in signed int64.
std::vector<int64_t> type_ids(const std::vector<int64_t> &rows, std::size_t width, std::size_t n) {
    std::vector<int64_t> order(n);
    std::iota(order.begin(), order.end(), int64_t(0));
    auto row = [&](int64_t i) { return rows.data() + static_cast<std::size_t>(i) * width; };
    auto less = [&](int64_t a, int64_t b) {
        const int64_t *x = row(a), *y = row(b);
        for (std::size_t c = 0; c < width; ++c)
            if (x[c] != y[c]) return x[c] < y[c];
        return false;
    };
    std::stable_sort(order.begin(), order.end(), less);
    std::vector<int64_t> ids(n);
    int64_t rank = -1;
    for (std::size_t k = 0; k < n; ++k) {
        if (k == 0 || less(order[k - 1], order[k])) ++rank;
        ids[static_cast<std::size_t>(order[k])] = rank;
    }
    return ids;
}

// (exponent, mantissa) of a positive length at KEY_BITS relative bits.
inline void quantise_length(double h, int64_t &e, int64_t &q) {
    int ex = 0;
    const double m = std::frexp(h, &ex);
    e = ex;
    q = rint64(std::ldexp(m, KEY_BITS));
}

// z, rho, td, rdt of one pair, in numpy's order of operations.
inline void scalars(const double *oc, const double *ot, const double *sc, const double *st, double &z,
                    double &rho, double &td, double &rdt) {
    double rvx = oc[0] - sc[0];
    double rvy = oc[1] - sc[1];
    double rvz = oc[2] - sc[2];
    const double tnx = st[0], tny = st[1], tnz = st[2];
    z = (rvx * tnx + rvy * tny) + rvz * tnz;
    rvx -= z * tnx;
    rvy -= z * tny;
    rvz -= z * tnz;
    rho = std::sqrt((rvx * rvx + rvy * rvy) + rvz * rvz);
    const double tmx = ot[0], tmy = ot[1], tmz = ot[2];
    td = (tmx * tnx + tmy * tny) + tmz * tnz;
    rdt = (rvx * tmx + rvy * tmy) + rvz * tmz;
}

// The quantum against `scale`: the exponent column's e and the step's
// inverse, off the bits of a positive normal number (frexp's exponent and
// 2^(KEY_BITS - e)).
inline void quantum(double scale, int64_t &e, double &step_inv) {
    e = (bits_of(scale) >> 52) - 1022;
    step_inv = double_of((1023 + KEY_BITS - e) << 52);
}

// The canonical orientation of one pair: key columns `k` (a = k[ia], b =
// k[ib], t = k[it], k[0] the types), keyed in place, its code returned. `ds`
// / `do_` the type moves of a source / observer reversal (has_*: one is
// given). Of the four orientations the largest key is taken, in the order
// (a, b, t, the types): the sign of a decides the source flip and that of b
// the observer's, and only where one is zero do the rest decide. A near pair
// (its key the sentinel) keeps code 0.
inline int8_t canonical(int64_t *k, int ia, int ib, int it, bool near, bool has_ds, int64_t ds, bool has_do,
                        int64_t do_) {
    int64_t &a = k[ia], &b = k[ib], &t = k[it];
    bool fs = a < 0, fo = b < 0;
    bool za = a == 0, zb = b == 0;
    if (near) za = zb = false;
    if (zb && !za) {
        // only the observer free: the both-odd column, then the observer types
        const int64_t tp = fs ? -t : t;
        bool flip = tp < 0;
        if (has_do) flip = flip || (tp == 0 && do_ > 0);
        fo = flip;
    } else if (za && !zb) {
        const int64_t tp = fo ? -t : t;
        bool flip = tp < 0;
        if (has_ds) flip = flip || (tp == 0 && ds > 0);
        fs = flip;
    } else if (za && zb) {
        // the largest of the four (t, types)
        bool valid = false, best_s = false, best_o = false;
        int64_t best_t = 0, best_e = 0;
        for (int cs = 0; cs < 2; ++cs)
            for (int co = 0; co < 2; ++co) {
                const int64_t tc = cs != co ? -t : t;
                int64_t ec = 0;
                if (cs && has_ds) ec += ds;
                if (co && has_do) ec += do_;
                const bool better = !valid || tc > best_t || (tc == best_t && ec > best_e);
                if (better) {
                    best_t = tc;
                    best_e = ec;
                    best_s = cs != 0;
                    best_o = co != 0;
                }
                valid = true;
            }
        fs = best_s;
        fo = best_o;
    }
    if (fs) a = -a;
    if (fo) b = -b;
    if (fs != fo) t = -t;
    if (has_ds && fs) k[0] += ds;
    if (has_do && fo) k[0] += do_;
    return static_cast<int8_t>((fs ? 1 : 0) | (fo ? 2 : 0));
}

// The near rule's kernel scale: max(2H, a), the source's length or the
// observer's radius, whichever is longer (PROBE_NEAR_H: 2H alone).
inline double near_scale(double H, double a, int probe) {
    return probe == PROBE_NEAR_H ? 2.0 * H : np_max(2.0 * H, a);
}

// The direct key of pair (m, n): the source-frame scalars quantised, the
// near pairs the sentinel; then the canonical orientation (columns e, z,
// rho, rdt, td: z odd under the source's reversal, rdt under the
// observer's, td under either). k[0..5), its code, its nearness.
struct DirectKey {
    const CensusInputs *in;
    int64_t n_obs_types;
    bool coded;
    const int64_t *ds;  // null: None
    int probe;
    inline int8_t operator()(int64_t m, int64_t n, int64_t *k, bool &near) const {
        const CensusInputs &I = *in;
        double z, rho, td, rdt;
        scalars(I.obs_c + 3 * m, I.obs_t + 3 * m, I.src_c + 3 * n, I.src_t + 3 * n, z, rho, td, rdt);
        const double H = 0.5 * I.seg_h[n];
        const double a = I.a_obs[m];
        const double az = std::fabs(z);
        near = np_max(az, rho) < NEAR_LENGTHS * near_scale(H, a, probe);
        // The pair's singular distance, from the observer centre to the
        // source SEGMENT, within sqrt(2), floored at the observer radius.
        double scale = np_max(az - H, rho);
        scale = np_max(scale, a);
        int64_t e;
        double step_inv;
        quantum(scale, e, step_inv);
        if (near) {
            k[0] = kNearSentinel;
            k[1] = k[2] = k[3] = k[4] = 0;
        } else {
            k[1] = rint64(z * step_inv);
            k[2] = rint64(rho * step_inv);
            k[3] = rint64(rdt * step_inv);
            k[4] = rint64(td * kUnit);
            // The exponent and the type share a column: |e| < 2^11 always,
            // and the type id is below N^2 < 2^40.
            k[0] = ((e + (int64_t(1) << 11)) << 40) + (I.src_type[n] * n_obs_types + I.obs_type[m]);
        }
        if (!coded) return 0;
        return canonical(k, 1, 3, 4, near, ds != nullptr, ds ? ds[n] : 0, false, 0);
    }
};

// The reflected key: the height sum, |d|, d.a, d.b, a.b against the pair's
// distance to the IMAGE segment, the near test against it; then the
// canonical orientation (d.a odd under the source's reversal, d.b under the
// observer's, a.b under either). k[0..6).
struct GroundKey {
    const CensusInputs *in;
    int64_t n_obs_types;
    bool coded;
    const int64_t *ds, *do_;
    int probe;
    inline int8_t operator()(int64_t m, int64_t n, int64_t *k, bool &near) const {
        const CensusInputs &I = *in;
        const double *oc = I.obs_c + 3 * m, *ot = I.obs_t + 3 * m;
        const double *sc = I.src_c + 3 * n, *st = I.src_t + 3 * n;
        const double gz = I.ground_z;
        const double dx = oc[0] - sc[0];
        const double dy = oc[1] - sc[1];
        const double hs = (oc[2] - gz) + (sc[2] - gz);
        const double ax = st[0], ay = st[1], azt = st[2];
        const double bx = ot[0], by = ot[1];
        const double dn = std::hypot(dx, dy);
        const double da = dx * ax + dy * ay;
        const double db = dx * bx + dy * by;
        const double ab = bx * ax + by * ay;
        // The image segment's tangent is (a, -t_nz) and the observer sits
        // at (d, hs) from its centre. rho by difference is inaccurate only
        // near the image axis within its span, a near pair.
        const double zi = da - hs * azt;
        double rho = dn * dn;
        rho += hs * hs;
        rho -= zi * zi;
        rho = np_max(rho, 0.0);
        rho = std::sqrt(rho);
        const double H = 0.5 * I.seg_h[n];
        const double a = I.a_obs[m];
        const double az = std::fabs(zi);
        near = np_max(az, rho) < NEAR_LENGTHS * near_scale(H, a, probe);
        double scale = np_max(az - H, rho);
        scale = np_max(scale, a);
        int64_t e;
        double step_inv;
        quantum(scale, e, step_inv);
        if (near) {
            k[0] = kNearSentinel;
            k[1] = k[2] = k[3] = k[4] = k[5] = 0;
        } else {
            k[1] = probe == PROBE_HEIGHT_BLIND ? 0 : rint64(hs * step_inv);
            k[2] = rint64(dn * step_inv);
            k[3] = rint64(da * step_inv);
            k[4] = rint64(db * step_inv);
            k[5] = rint64(ab * kUnit);
            k[0] = ((e + (int64_t(1) << 11)) << 40) + (I.src_type[n] * n_obs_types + I.obs_type[m]);
        }
        if (!coded) return 0;
        const int8_t code =
            canonical(k, 3, 4, 5, near, ds != nullptr, ds ? ds[n] : 0, do_ != nullptr, do_ ? do_[m] : 0);
        if (probe >= PROBE_GROUND_CODES && probe < PROBE_GROUND_CODES + 3)
            return kProbeCodes[probe - PROBE_GROUND_CODES][code];
        return code;
    }
};

// The census's parts' table: an `IntGroups` (the fold table's own layout,
// which the budget's figures count), each new key given the core's `fresh`
// class.
template <int W>
struct GroupTable {
    std::unique_ptr<int_groups::IntGroups> g;
    std::vector<int32_t> local;  // each group's class
    size_t cap = 0, n = 0;       // the core's measured accounting, unused here
    int32_t find_or_add(const uint64_t *k, uint64_t, int32_t fresh, bool &added) {
        if (!g) g.reset(new int_groups::IntGroups(W));
        const int64_t id = g->add_row(k);
        added = id == static_cast<int64_t>(local.size());
        if (added) local.push_back(fresh);
        return local[static_cast<std::size_t>(id)];
    }
};

// What one running `IntGroups` fold table of `width` columns holds after
// its `n`th group, `extra` when it was called again since (its keys in
// blocks of kBlock groups; its slots, 16 doubled while 2 (n_ + 1) passes
// them at a call made with n_ groups).
int64_t fold_table_bytes(int64_t n, bool extra, int64_t width) {
    const int64_t blk = static_cast<int64_t>(int_groups::IntGroups::kBlock);
    const int64_t need = 2 * (n + (extra ? 1 : 0));
    int64_t slots = 16;
    while (slots < need) slots *= 2;
    return (((n + blk - 1) / blk) * blk * width + slots) * 8;
}

// THE CENSUS'S PEAK (the header's): the core's band budget, unbounded, over
// what one fold table holds after each band. The class cap stops it (the
// core's `Policy::capped`), not its bytes: under the allowance a plan at
// the cap is what the allowance holds.
struct FigureHook {
    static constexpr bool tail = true;
    core::BandBudget bands;
    int64_t map_bytes, width;
    bool start(int64_t i) { return bands.start(i); }
    bool done(int64_t, int64_t classes, int64_t near, bool tail_new) {
        // A band whose last new key came after its last new class called
        // the table again with every class in it (`fold_table_bytes`).
        const int64_t held = map_bytes + fold_table_bytes(classes, !tail_new, width) + 9 * classes + 8 * near;
        return bands.fold(held, 0);
    }
};

// `_band_rows`: rows per census band, the swept rows shared out over the
// workers, none so small that its per-band cost shows, and at most
// BAND_SHARE of the plan's allowance over a pool's worth: the bands in
// flight are a transient bounded beside the classes, not stopped by bytes.
int64_t band_rows(int64_t rows, int64_t M, int64_t N, int64_t budget, int64_t per_pair, int workers) {
    const int64_t w = std::max(1, workers);
    rows = std::max<int64_t>(1, rows / w);
    rows = std::min(rows, std::max<int64_t>(16, (M + 4 * w - 1) / (4 * w)));
    if (budget >= 0 && per_pair > 0)
        rows = std::min(rows, std::max<int64_t>(1, static_cast<int64_t>(BAND_SHARE * static_cast<double>(budget)) /
                                                       (w * N * per_pair)));
    return rows;
}

// One family's census (`W` key words) by `key`, into `plan->fam`; false
// where it declines (the class cap, or past the int32 coding), `why` then
// the reason.
template <int W, class Key>
bool census(const Key &key, const CensusInputs &in, bool coded, bool ground, int64_t per_pair, const char *name,
            Plan &plan, std::string &why) {
    const int64_t M = in.M, N = in.N;
    const int workers = std::max(1, in.workers);
    const int64_t rows = band_rows(in.rows, M, N, allowance_of(in, ground, 0), per_pair, workers);
    core::FamilyPlan &f = plan.fam;
    f.spec.name = name;
    f.spec.key_words = W;
    f.spec.code_bits = coded ? CODE_BITS : 0;
    f.spec.n_blocks = 1;
    f.spec.rows = M;
    f.spec.cols = N;
    f.spec.value_bytes = 3 * 16;
    f.spec.numbering = core::Numbering::FirstAppearance;
    f.spec.keep_reps = true;
    core::Policy pol;
    pol.rep_rows = 0;
    pol.census_rows = rows;
    pol.measured = false;
    pol.capped = true;
    const int64_t n_bands = (M + rows - 1) / rows;
    const int64_t map_bytes = 4 * M * N;
    FigureHook hook{core::BandBudget(-1, per_pair * rows * N, workers, n_bands,
                                     map_bytes + fold_table_bytes(0, false, W)),
                    map_bytes, W};
    int64_t unused = 0;
    const core::Outcome out = core::census<W, GroupTable<W>>(
        f, pol, 0, unused,
        [&](int, int64_t o, int64_t s, uint64_t *w, uint8_t &code) {
            int64_t k[W];
            bool near = false;
            code = static_cast<uint8_t>(key(o, s, k, near));
            std::memcpy(w, k, sizeof k);
            return near ? core::Key::Near : core::Key::Shared;
        },
        hook);
    plan.census_peak_bytes = hook.bands.peak();
    if (out == core::Outcome::Poor) {
        why = core::cap_reason(M * N);
        return false;
    }
    if (out != core::Outcome::Ok) {
        why = "too many classes to code";
        return false;
    }
    return true;
}

// The permutation a stable argsort on `key` gives, and whether it moves anything.
std::vector<int64_t> stable_argsort(const std::vector<int64_t> &key, bool &moved) {
    std::vector<int64_t> perm(key.size());
    std::iota(perm.begin(), perm.end(), int64_t(0));
    std::stable_sort(perm.begin(), perm.end(), [&](int64_t a, int64_t b) {
        return key[static_cast<std::size_t>(a)] < key[static_cast<std::size_t>(b)];
    });
    moved = false;
    for (std::size_t i = 0; i < perm.size(); ++i)
        if (perm[i] != static_cast<int64_t>(i)) moved = true;
    return perm;
}

// The direct kernel calls that evaluate the pairs `flat` (sorted flat
// indices): per run of up to `rows` observer rows of one radius, the union
// of their sources, and where each pair sits in that call's block.
std::vector<NearBlock> near_blocks(const std::vector<int64_t> &flat, int64_t N, int64_t M, const double *a_obs,
                                   int64_t rows) {
    std::vector<NearBlock> out;
    if (flat.empty()) return out;
    std::vector<int64_t> edges;
    for (int64_t e = 0; e <= M; e += rows) edges.push_back(e);
    for (int64_t i = 1; i < M; ++i)
        if (a_obs[i] - a_obs[i - 1] != 0.0) edges.push_back(i);
    edges.push_back(M);
    std::sort(edges.begin(), edges.end());
    edges.erase(std::unique(edges.begin(), edges.end()), edges.end());
    const std::size_t K = flat.size();
    std::vector<int64_t> m_all(K), n_all(K);
    for (std::size_t i = 0; i < K; ++i) {
        m_all[i] = flat[i] / N;
        n_all[i] = flat[i] % N;
    }
    for (std::size_t e = 0; e + 1 < edges.size(); ++e) {
        const int64_t m0 = edges[e], m1 = edges[e + 1];
        const int64_t p0 = std::lower_bound(m_all.begin(), m_all.end(), m0) - m_all.begin();
        const int64_t p1 = std::lower_bound(m_all.begin(), m_all.end(), m1) - m_all.begin();
        if (p1 == p0) continue;
        NearBlock b;
        b.m0 = m0;
        b.m1 = m1;
        b.p0 = p0;
        b.p1 = p1;
        b.ns.assign(n_all.begin() + p0, n_all.begin() + p1);
        std::sort(b.ns.begin(), b.ns.end());
        b.ns.erase(std::unique(b.ns.begin(), b.ns.end()), b.ns.end());
        b.r.resize(static_cast<std::size_t>(p1 - p0));
        b.c.resize(static_cast<std::size_t>(p1 - p0));
        for (int64_t p = p0; p < p1; ++p) {
            b.r[static_cast<std::size_t>(p - p0)] = m_all[static_cast<std::size_t>(p)] - m0;
            b.c[static_cast<std::size_t>(p - p0)] =
                std::lower_bound(b.ns.begin(), b.ns.end(), n_all[static_cast<std::size_t>(p)]) - b.ns.begin();
        }
        out.push_back(std::move(b));
    }
    return out;
}

int64_t blocks_nbytes(const std::vector<NearBlock> &bs) {
    int64_t s = 0;
    for (const NearBlock &b : bs) s += 8 * static_cast<int64_t>(b.ns.size() + b.r.size() + b.c.size());
    return s;
}

// The source's centre and tangent placing a class in the canonical observer
// frame (the observer at the origin along +z), from its representative's
// (rho, z, rdt, td). With the source tangent u = (s, 0, td), s = sqrt(1 -
// td^2), the radial vector is alpha p + beta q over p = (-td, 0, s) and q =
// (0, 1, 0): alpha s = rdt and alpha^2 + beta^2 = rho^2.
inline void canonical_source(double rho, double z, double rdt, double td, double *c, double *t) {
    const double s = std::sqrt(np_max(0.0, 1.0 - td * td));
    double alpha = s > 0.0 ? rdt / s : 0.0;
    alpha = clip_min(clip_max(alpha, -rho), rho);
    const double beta = std::sqrt(np_max(0.0, rho * rho - alpha * alpha));
    t[0] = s;
    t[1] = 0.0;
    t[2] = td;
    const double rv0 = -alpha * td + z * s;
    const double rv1 = beta;
    const double rv2 = alpha * s + z * td;
    c[0] = -rv0;
    c[1] = -rv1;
    c[2] = -rv2;
}

const double kCodeSigns[4][3] = {
    {1.0, 1.0, 1.0},    // 0: as stored
    {-1.0, 1.0, -1.0},  // 1: source reversed
    {-1.0, -1.0, -1.0},  // 2: observer reversed
    {1.0, -1.0, 1.0},   // 3: both
};

// `to_canonical`: each class's value its canonical orientation's.
void to_canonical(const Plan &p, std::vector<cd> &vals) {
    if (!p.coded) return;
    for (std::size_t c = 0; c < p.fam.rep_code.size(); ++c)
        for (int s = 0; s < 3; ++s)
            if (kCodeSigns[p.fam.rep_code[c]][s] < 0) vals[3 * c + static_cast<std::size_t>(s)] = -vals[3 * c + static_cast<std::size_t>(s)];
}

// `evaluate_pairs`: (n, 3) values of the pairs `blocks` lay out.
void evaluate_pairs(int64_t n, const std::vector<NearBlock> &blocks, const RowsFn &rows, std::vector<cd> &out) {
    out.assign(static_cast<std::size_t>(3 * n), cd(0.0, 0.0));
    bool failed = false;
    std::string why;
#if defined(_OPENMP)
#pragma omp parallel for schedule(dynamic, 1)
#endif
    for (std::ptrdiff_t q = 0; q < static_cast<std::ptrdiff_t>(blocks.size()); ++q) {
        const NearBlock &b = blocks[static_cast<std::size_t>(q)];
        try {
            std::vector<cd> phis[3];
            const int64_t nn = static_cast<int64_t>(b.ns.size());
            rows(b.m0, b.m1, b.ns.data(), nn, phis);
            for (int64_t i = 0; i < b.p1 - b.p0; ++i) {
                const std::size_t at = static_cast<std::size_t>(b.r[static_cast<std::size_t>(i)] * nn +
                                                                b.c[static_cast<std::size_t>(i)]);
                for (int s = 0; s < 3; ++s)
                    out[static_cast<std::size_t>(3 * (b.p0 + i) + s)] = phis[s][at];
            }
        } catch (const std::exception &e) {
#if defined(_OPENMP)
#pragma omp critical(mw_sin_reuse_eval)
#endif
            {
                failed = true;
                why = e.what();
            }
        }
    }
    if (failed) throw std::runtime_error(why);
}

// The type moves of a source reversal, (src_rev - src_type) * n_obs_types,
// or empty where no reversal moves a type (None).
std::vector<int64_t> source_moves(const CensusInputs &in, int64_t n_obs_types) {
    std::vector<int64_t> ds(static_cast<std::size_t>(in.N));
    bool any = false;
    for (int64_t n = 0; n < in.N; ++n) {
        ds[static_cast<std::size_t>(n)] = (in.src_rev[n] - in.src_type[n]) * n_obs_types;
        any = any || ds[static_cast<std::size_t>(n)] != 0;
    }
    if (!any) ds.clear();
    return ds;
}

int64_t max_type(const int64_t *t, int64_t n) {
    int64_t m = 0;
    for (int64_t i = 0; i < n; ++i) m = std::max(m, t[i] + 1);
    return m;
}

}  // namespace

void set_plan_allowance(int64_t bytes) {
    std::lock_guard<std::mutex> lock(g_mu);
    g_allowance = bytes > 0 ? bytes : -1;
}
int64_t plan_allowance_override() {
    std::lock_guard<std::mutex> lock(g_mu);
    return g_allowance;
}
int64_t allowance_of(const CensusInputs &in, bool ground, int64_t n_near) {
    if (in.allowance >= 0) return in.allowance;
    return core::plan_allowance(in.M * in.N, ground ? GROUND_KEPT_BYTES_PER_CLASS : KEPT_BYTES_PER_CLASS) +
           NEAR_KEPT_BYTES * n_near;
}
void set_small_deck_segs(int64_t free_space, int64_t finite_ground) {
    std::lock_guard<std::mutex> lock(g_mu);
    g_small_free = free_space < 0 ? SMALL_DECK_SEGS_FREE : free_space;
    g_small_ground = finite_ground < 0 ? SMALL_DECK_SEGS_GROUND : finite_ground;
}
int64_t small_deck_segs(bool finite) {
    std::lock_guard<std::mutex> lock(g_mu);
    return finite ? g_small_ground : g_small_free;
}
void set_workers(int w) {
    std::lock_guard<std::mutex> lock(g_mu);
    g_workers = w > 0 ? w : 0;
}
int workers() {
    {
        std::lock_guard<std::mutex> lock(g_mu);
        if (g_workers > 0) return g_workers;
    }
#if defined(_OPENMP)
    return std::max(1, omp_get_max_threads());
#else
    return 1;
#endif
}
void set_probe(int p) {
    std::lock_guard<std::mutex> lock(g_mu);
    g_probe = p;
}
int probe() {
    std::lock_guard<std::mutex> lock(g_mu);
    return g_probe;
}

Types reversible_segment_types(int64_t n, const double *seg_h, const double *a_obs, const double *src_a,
                               const int8_t *ind1, const int8_t *ind2) {
    const std::size_t nz = static_cast<std::size_t>(n);
    Types t;
    // the observer type: the radius, exact
    std::vector<int64_t> arow(nz);
    for (std::size_t i = 0; i < nz; ++i) arow[i] = bits_of(a_obs[i]);
    t.obs = type_ids(arow, 1, nz);
    if (src_a == nullptr) {
        std::vector<int64_t> rows(2 * nz);
        for (std::size_t i = 0; i < nz; ++i) quantise_length(seg_h[i], rows[2 * i], rows[2 * i + 1]);
        t.src = type_ids(rows, 2, nz);
        t.src_rev = t.src;
        return t;
    }
    // both orientations in one id space: the end codes swap
    std::vector<int64_t> rows(5 * 2 * nz);
    for (std::size_t j = 0; j < 2 * nz; ++j) {
        const std::size_t i = j % nz;
        int64_t *r = rows.data() + 5 * j;
        quantise_length(seg_h[i], r[0], r[1]);
        r[2] = bits_of(src_a[i]);
        r[3] = j < nz ? ind1[i] : ind2[i];
        r[4] = j < nz ? ind2[i] : ind1[i];
    }
    const std::vector<int64_t> ids = type_ids(rows, 5, 2 * nz);
    t.src.assign(ids.begin(), ids.begin() + static_cast<std::ptrdiff_t>(nz));
    t.src_rev.assign(ids.begin() + static_cast<std::ptrdiff_t>(nz), ids.end());
    return t;
}

GroundTypes reversible_ground_types(const Types &t, int64_t n, const double *seg_t) {
    const std::size_t nz = static_cast<std::size_t>(n);
    GroundTypes g;
    if (probe() == PROBE_FREE_KEY) {  // the types without the vertical tangent
        g.src = t.src;
        g.obs = t.obs;
        g.src_rev = t.src_rev;
        g.obs_rev = t.obs;
        return g;
    }
    std::vector<int64_t> srows(4 * nz), orows(4 * nz);
    for (std::size_t j = 0; j < 2 * nz; ++j) {
        const std::size_t i = j % nz;
        const double tz = j < nz ? seg_t[3 * i + 2] : -seg_t[3 * i + 2];
        const int64_t q = rint64(tz * kUnit);  // a direction cosine at KEY_BITS absolute bits
        srows[2 * j] = j < nz ? t.src[i] : t.src_rev[i];
        srows[2 * j + 1] = q;
        orows[2 * j] = t.obs[i];
        orows[2 * j + 1] = q;
    }
    const std::vector<int64_t> gs = type_ids(srows, 2, 2 * nz), go = type_ids(orows, 2, 2 * nz);
    g.src.assign(gs.begin(), gs.begin() + static_cast<std::ptrdiff_t>(nz));
    g.src_rev.assign(gs.begin() + static_cast<std::ptrdiff_t>(nz), gs.end());
    g.obs.assign(go.begin(), go.begin() + static_cast<std::ptrdiff_t>(nz));
    g.obs_rev.assign(go.begin() + static_cast<std::ptrdiff_t>(nz), go.end());
    return g;
}

int64_t Plan::plan_nbytes() const {
    int64_t s = 4 * static_cast<int64_t>(cls().size()) + 8 * n_classes() + 8 * n_near() +
                (coded ? static_cast<int64_t>(fam.rep_code.size()) : 0) + blocks_nbytes(near_blocks);
    if (ground) return s + blocks_nbytes(rep_blocks);
    return s + 8 * static_cast<int64_t>(src_c.size() + src_t.size());
}

std::shared_ptr<const Plan> build_plan(const CensusInputs &in, std::string *why) {
    std::string reason;
    auto no = [&](std::string r) -> std::shared_ptr<const Plan> {
        if (why) *why = std::move(r);
        return nullptr;
    };
    const int64_t allowance = allowance_of(in, false, 0);  // the map's and the classes', before the census
    const int64_t M = in.M, N = in.N;
    const int64_t n_obs_types = max_type(in.obs_type, M);
    const bool coded = in.src_rev != nullptr && M * N <= CODED_PAIRS;
    const std::vector<int64_t> ds = coded ? source_moves(in, n_obs_types) : std::vector<int64_t>();
    if (4 * M * N > allowance) return no(core::allowance_reason(allowance));
    auto plan = std::make_shared<Plan>();
    plan->M = M;
    plan->N = N;
    plan->coded = coded;
    const DirectKey key{&in, n_obs_types, coded, ds.empty() ? nullptr : ds.data(), probe()};
    const int64_t per_pair = in.band_bytes_per_pair >= 0 ? in.band_bytes_per_pair : BAND_BYTES_PER_PAIR;
    if (!census<5>(key, in, coded, false, per_pair, "direct", *plan, reason)) return no(reason);
    trim_heap();
    core::FamilyPlan &f = plan->fam;
    // Classes renumbered so those sharing an observer radius are contiguous
    // (stable within a radius): `evaluate` then runs on slices.
    const std::size_t n_cls = f.rep_flat.size();
    std::vector<int64_t> otype(n_cls);
    for (std::size_t c = 0; c < n_cls; ++c) otype[c] = in.obs_type[f.rep_flat[c] / N];
    bool moved = false;
    const std::vector<int64_t> perm = stable_argsort(otype, moved);
    if (moved) {
        std::vector<int64_t> renumber(n_cls);
        for (std::size_t j = 0; j < n_cls; ++j) renumber[static_cast<std::size_t>(perm[j])] = static_cast<int64_t>(j);
        core::renumber(f, renumber);
    }
    std::vector<int64_t> rep_obs(n_cls);
    plan->rep.resize(n_cls);
    for (std::size_t j = 0; j < n_cls; ++j) {
        rep_obs[j] = f.rep_flat[j] / N;
        plan->rep[j] = f.rep_flat[j] % N;
    }
    std::vector<int64_t>().swap(f.rep_flat);
    // Each representative's (rho, z, rdt, td), the bits its key computed,
    // placed in the observer frame.
    plan->src_c.resize(3 * n_cls);
    plan->src_t.resize(3 * n_cls);
    for (std::size_t c = 0; c < n_cls; ++c) {
        const int64_t m = rep_obs[c], n = plan->rep[c];
        double z, rho, td, rdt;
        scalars(in.obs_c + 3 * m, in.obs_t + 3 * m, in.src_c + 3 * n, in.src_t + 3 * n, z, rho, td, rdt);
        canonical_source(rho, z, rdt, td, plan->src_c.data() + 3 * c, plan->src_t.data() + 3 * c);
    }
    for (std::size_t c = 0; c < n_cls;) {
        std::size_t e = c + 1;
        while (e < n_cls && in.obs_type[rep_obs[e]] == in.obs_type[rep_obs[c]]) ++e;
        plan->runs.push_back(Run{static_cast<int64_t>(c), static_cast<int64_t>(e), rep_obs[c]});
        c = e;
    }
    plan->near_blocks = near_blocks(f.near_flat, N, M, in.a_obs, NEAR_ROWS);
    const int64_t held_to = allowance_of(in, false, plan->n_near());
    if (plan->fill_nbytes() > held_to) return no(core::allowance_reason(held_to));
    return plan;
}

std::shared_ptr<const Plan> build_ground_plan(const CensusInputs &in, std::string *why) {
    std::string reason;
    auto no = [&](std::string r) -> std::shared_ptr<const Plan> {
        if (why) *why = std::move(r);
        return nullptr;
    };
    const int64_t allowance = allowance_of(in, true, 0);  // the map's and the classes', before the census
    const int64_t M = in.M, N = in.N;
    const int pb = probe();
    // PROBE_FREE_KEY: the direct block's key, uncoded, standing in.
    const bool free_key = pb == PROBE_FREE_KEY;
    const bool coded = !free_key && in.src_rev != nullptr && in.obs_rev != nullptr && M * N <= CODED_PAIRS;
    int64_t n_obs_types = max_type(in.obs_type, M);
    std::vector<int64_t> ds, dob;
    if (coded) {
        n_obs_types = std::max(n_obs_types, max_type(in.obs_rev, M));
        ds = source_moves(in, n_obs_types);
        dob.resize(static_cast<std::size_t>(M));
        bool any = false;
        for (int64_t m = 0; m < M; ++m) {
            dob[static_cast<std::size_t>(m)] = in.obs_rev[m] - in.obs_type[m];
            any = any || dob[static_cast<std::size_t>(m)] != 0;
        }
        if (!any) dob.clear();
    }
    if (4 * M * N > allowance) return no(core::allowance_reason(allowance));
    auto plan = std::make_shared<Plan>();
    plan->ground = true;
    plan->M = M;
    plan->N = N;
    plan->coded = coded;
    const int64_t per_pair = in.band_bytes_per_pair >= 0 ? in.band_bytes_per_pair : GROUND_BAND_BYTES_PER_PAIR;
    bool ok;
    if (free_key) {
        const DirectKey key{&in, n_obs_types, false, nullptr, 0};
        ok = census<5>(key, in, false, true, per_pair, "reflected", *plan, reason);
    } else {
        const GroundKey key{&in, n_obs_types, coded, ds.empty() ? nullptr : ds.data(),
                            dob.empty() ? nullptr : dob.data(), pb};
        ok = census<6>(key, in, coded, true, per_pair, "reflected", *plan, reason);
    }
    if (!ok) return no(reason);
    trim_heap();
    // Numbered by the representative's flat index: the census's own order.
    core::FamilyPlan &f = plan->fam;
    plan->rep.swap(f.rep_flat);
    plan->rep_blocks = near_blocks(plan->rep, N, M, in.a_obs, REP_ROWS);
    plan->near_blocks = near_blocks(f.near_flat, N, M, in.a_obs, NEAR_ROWS);
    const int64_t held_to = allowance_of(in, true, plan->n_near());
    if (plan->fill_nbytes() > held_to) return no(core::allowance_reason(held_to));
    return plan;
}

namespace {

void append(std::string &key, const void *p, std::size_t n) {
    if (p != nullptr) key.append(static_cast<const char *>(p), n);
    key.push_back('|');
}

// Every input the census reads, and the hooks in force.
std::string plan_key(const CensusInputs &in, bool ground) {
    const std::size_t M = static_cast<std::size_t>(in.M), N = static_cast<std::size_t>(in.N);
    std::string key = ground ? "reflected|" : "direct|";
    const int64_t head[7] = {in.M, in.N, in.rows, allowance_of(in, ground, 0), core::cap_key(),
                             in.band_bytes_per_pair, probe()};
    append(key, head, sizeof head);
    append(key, in.obs_c, 24 * M);
    append(key, in.obs_t, 24 * M);
    append(key, in.src_c, 24 * N);
    append(key, in.src_t, 24 * N);
    append(key, in.seg_h, 8 * N);
    append(key, in.a_obs, 8 * M);
    append(key, in.src_type, 8 * N);
    append(key, in.obs_type, 8 * M);
    append(key, in.src_rev, 8 * N);
    if (ground) {
        append(key, in.obs_rev, 8 * M);
        append(key, &in.ground_z, 8);
    }
    return key;
}

}  // namespace

std::shared_ptr<const Plan> shared_plan(const CensusInputs &in, bool ground, bool *built, std::string *why,
                                        int keep) {
    const std::shared_ptr<const core::Plan> p = core::plan_for(
        BASIS, plan_key(in, ground),
        [&](core::Plan &out) {
            std::string reason;
            std::shared_ptr<const Plan> sp = ground ? build_ground_plan(in, &reason) : build_plan(in, &reason);
            trim_heap();
            if (sp == nullptr) {
                core::decline(out, reason);
                return;
            }
            out.ext_bytes = static_cast<std::size_t>(sp->plan_nbytes());
            out.ext = sp;
        },
        built, keep);
    if (!p->declined.empty()) {
        if (why) *why = p->declined;
        return nullptr;
    }
    return std::shared_ptr<const Plan>(p, static_cast<const Plan *>(p->ext.get()));
}

void clear_shared_plans() { core::clear_cache(BASIS); }

core::Counters counters() { return core::counters(BASIS); }
void reset_counters() {
    core::reset_counters(BASIS);
    std::lock_guard<std::mutex> lock(g_mu);
    g_last.clear();
    g_census.clear();
}
std::string last_outcome() {
    std::lock_guard<std::mutex> lock(g_mu);
    return g_last;
}
void set_last_outcome(const std::string &s) {
    std::lock_guard<std::mutex> lock(g_mu);
    g_last = s;
}
std::vector<CensusEntry> last_census() {
    std::lock_guard<std::mutex> lock(g_mu);
    return g_census;
}
void set_last_census(std::vector<CensusEntry> c) {
    std::lock_guard<std::mutex> lock(g_mu);
    g_census = std::move(c);
}

void pair_keys(const CensusInputs &in, bool ground, std::vector<int64_t> &words, std::vector<int8_t> &codes,
               std::vector<uint8_t> &near) {
    const int64_t M = in.M, N = in.N, W = ground ? 6 : 5;
    int64_t n_obs_types = max_type(in.obs_type, M);
    if (ground && in.obs_rev != nullptr) n_obs_types = std::max(n_obs_types, max_type(in.obs_rev, M));
    std::vector<int64_t> ds, dob;
    if (in.src_rev != nullptr) ds = source_moves(in, n_obs_types);
    if (ground && in.obs_rev != nullptr) {
        dob.resize(static_cast<std::size_t>(M));
        for (int64_t m = 0; m < M; ++m) dob[static_cast<std::size_t>(m)] = in.obs_rev[m] - in.obs_type[m];
    }
    const bool coded = in.src_rev != nullptr && (!ground || in.obs_rev != nullptr);
    words.assign(static_cast<std::size_t>(M * N * W), 0);
    codes.assign(static_cast<std::size_t>(M * N), 0);
    near.assign(static_cast<std::size_t>(M * N), 0);
    const int pb = probe();
    const DirectKey dk{&in, n_obs_types, coded, ds.empty() ? nullptr : ds.data(), pb};
    const GroundKey gk{&in, n_obs_types, coded, ds.empty() ? nullptr : ds.data(),
                       dob.empty() ? nullptr : dob.data(), pb};
    for (int64_t m = 0; m < M; ++m)
        for (int64_t n = 0; n < N; ++n) {
            const std::size_t i = static_cast<std::size_t>(m * N + n);
            bool nr = false;
            codes[i] = ground ? gk(m, n, words.data() + i * W, nr) : dk(m, n, words.data() + i * W, nr);
            near[i] = nr ? 1 : 0;
        }
}

void canonical_codes(int64_t *rows, int width, int64_t n, int ia, int ib, int it, const uint8_t *near,
                     int64_t n_src, const int64_t *ds, const int64_t *do_, int8_t *codes) {
    for (int64_t i = 0; i < n; ++i)
        codes[i] = canonical(rows + i * width, ia, ib, it, near != nullptr && near[i] != 0, ds != nullptr,
                             ds ? ds[i % n_src] : 0, do_ != nullptr, do_ ? do_[i / n_src] : 0);
}

void evaluate(const Plan &p, const CanonFn &canon, Values &out) {
    const int64_t n_cls = p.n_classes();
    out.cls.assign(static_cast<std::size_t>(3 * n_cls), cd(0.0, 0.0));
    struct Piece {
        int64_t c0, c1, g_obs;
    };
    std::vector<Piece> pieces;
    for (const Run &r : p.runs)
        for (int64_t c0 = r.c0; c0 < r.c1; c0 += EVAL_CHUNK) pieces.push_back({c0, std::min(r.c1, c0 + EVAL_CHUNK), r.g_obs});
    bool failed = false;
    std::string why;
#if defined(_OPENMP)
#pragma omp parallel for schedule(dynamic, 1)
#endif
    for (std::ptrdiff_t q = 0; q < static_cast<std::ptrdiff_t>(pieces.size()); ++q) {
        const Piece &pc = pieces[static_cast<std::size_t>(q)];
        try {
            std::vector<cd> phis[3];
            const int64_t n = pc.c1 - pc.c0;
            canon(p.src_c.data() + 3 * pc.c0, p.src_t.data() + 3 * pc.c0, p.rep.data() + pc.c0, n, pc.g_obs, phis);
            for (int64_t i = 0; i < n; ++i)
                for (int s = 0; s < 3; ++s)
                    out.cls[static_cast<std::size_t>(3 * (pc.c0 + i) + s)] = phis[s][static_cast<std::size_t>(i)];
        } catch (const std::exception &e) {
#if defined(_OPENMP)
#pragma omp critical(mw_sin_reuse_eval)
#endif
            {
                failed = true;
                why = e.what();
            }
        }
    }
    if (failed) throw std::runtime_error(why);
    to_canonical(p, out.cls);
}

void evaluate_classes(const Plan &p, const RowsFn &rows, Values &out) {
    evaluate_pairs(p.n_classes(), p.rep_blocks, rows, out.cls);
    to_canonical(p, out.cls);
}

void evaluate_near(const Plan &p, const RowsFn &rows, Values &out) {
    evaluate_pairs(p.n_near(), p.near_blocks, rows, out.near);
}

// --- the fused gather + reduction (momwire#1916, #1969) ---------------------
//
// THE BITS ARE `scatter` + `reduce_phi_band`'s, by construction. The gather
// is a copy: `np.take(rep[s], cls, mode="clip")` is the clamped class's
// value, and the near pairs then overwrite theirs, as `scatter` writes them
// after the take. Under an image block the image is gathered the same way
// and subtracted, `Φ -= Φ_img`, one IEEE subtraction per part as numpy's
// complex `-=`. The reduction is `reduce_phi_band`'s (`_accel_phi_reduce.cpp`):
// per output element a zero plus its column's stored entries of M in stored
// order, one unfused multiply-add each, then `(A + B) + C`. Rows are
// independent, so the thread count moves nothing; LANES rows share each step
// of the walk over M, each its own chain. A coded map's entry is class << 2 |
// code: the gather takes class `entry >> 2` (clamped) and negates each shape
// whose sign under the code is negative, an exact negation.

namespace {

// Observer rows reduced together: every step for LANES rows at once on the
// same M entry, each row's chain exactly as `reduce_phi_band` runs it, so the
// compiler may put the lanes in one vector register; it moves no operation
// within a chain.
constexpr int LANES = 4;

// Row r's three Φ rows into lane `l` of x[0..2], each laid out
// [source n][re, im][lane]: the take, then the near pairs over it.
void gather_row(const Block &b, std::ptrdiff_t r, std::ptrdiff_t n_src, int l, double *x[3]) {
    const int32_t *c = b.cls + r * n_src;
    const int64_t hi = b.n_cls - 1;
    if (b.coded) {
        for (std::ptrdiff_t n = 0; n < n_src; ++n) {
            const int32_t e = c[n];
            int64_t k = e >> 2;  // arithmetic, as numpy's >> on int32
            k = k < 0 ? 0 : (k > hi ? hi : k);  // np.take(mode="clip")
            const bool *ng = b.neg[e & 3];
            const double *v = b.rep + 6 * k;
            const std::ptrdiff_t at = 2 * LANES * n + l;
            x[0][at] = ng[0] ? -v[0] : v[0];
            x[0][at + LANES] = ng[0] ? -v[1] : v[1];
            x[1][at] = ng[1] ? -v[2] : v[2];
            x[1][at + LANES] = ng[1] ? -v[3] : v[3];
            x[2][at] = ng[2] ? -v[4] : v[4];
            x[2][at + LANES] = ng[2] ? -v[5] : v[5];
        }
    } else {
        for (std::ptrdiff_t n = 0; n < n_src; ++n) {
            int64_t k = c[n];
            k = k < 0 ? 0 : (k > hi ? hi : k);  // np.take(mode="clip")
            const double *v = b.rep + 6 * k;
            const std::ptrdiff_t at = 2 * LANES * n + l;
            x[0][at] = v[0];
            x[0][at + LANES] = v[1];
            x[1][at] = v[2];
            x[1][at + LANES] = v[3];
            x[2][at] = v[4];
            x[2][at + LANES] = v[5];
        }
    }
    if (b.n_near == 0) return;
    const int64_t lo = static_cast<int64_t>(r) * n_src, up = lo + n_src;
    const int64_t *e0 = std::lower_bound(b.near_idx, b.near_idx + b.n_near, lo);
    const int64_t *e1 = std::lower_bound(e0, b.near_idx + b.n_near, up);
    for (const int64_t *e = e0; e < e1; ++e) {
        const std::ptrdiff_t n = static_cast<std::ptrdiff_t>(*e - lo);
        const double *v = b.near_val + 6 * (e - b.near_idx);
        const std::ptrdiff_t at = 2 * LANES * n + l;
        x[0][at] = v[0];
        x[0][at + LANES] = v[1];
        x[1][at] = v[2];
        x[1][at + LANES] = v[3];
        x[2][at] = v[4];
        x[2][at + LANES] = v[5];
    }
}

}  // namespace

void set_code_signs(Block &b) {
    for (int c = 0; c < 4; ++c)
        for (int s = 0; s < 3; ++s) b.neg[c][s] = kCodeSigns[c][s] < 0.0;
    b.coded = true;
}

void reduce_band(char *O, std::ptrdiff_t os, std::ptrdiff_t rows, std::ptrdiff_t cols, std::ptrdiff_t n_src,
                 const Block &d, const Block *image, const Csc M[3], int n_threads) {
    int nt = 1;
#ifdef _OPENMP
    nt = n_threads > 0 ? n_threads : omp_get_max_threads();
#else
    (void)n_threads;
#endif
    if (nt < 1) nt = 1;
    const bool has_image = image != nullptr;
    const std::size_t per = static_cast<std::size_t>(2 * LANES * n_src);
    const std::size_t blocks = has_image ? 6 : 3;
    std::vector<double> bufs(static_cast<std::size_t>(nt) * blocks * per);
    const std::ptrdiff_t tiles = (rows + LANES - 1) / LANES;
#if defined(_OPENMP)
#pragma omp parallel for schedule(static) num_threads(nt)
#endif
    for (std::ptrdiff_t tile = 0; tile < tiles; ++tile) {
        int t = 0;
#ifdef _OPENMP
        t = omp_get_thread_num();
#endif
        const std::ptrdiff_t r0 = tile * LANES;
        const int nl = static_cast<int>(std::min<std::ptrdiff_t>(LANES, rows - r0));
        double *base = bufs.data() + static_cast<std::size_t>(t) * blocks * per;
        double *x[3] = {base, base + per, base + 2 * per};
        // A short last tile's idle lanes compute on zeros, never stored.
        if (nl < LANES) std::fill(base, base + blocks * per, 0.0);
        for (int l = 0; l < nl; ++l) gather_row(d, r0 + l, n_src, l, x);
        if (has_image) {
            double *y[3] = {base + 3 * per, base + 4 * per, base + 5 * per};
            for (int l = 0; l < nl; ++l) gather_row(*image, r0 + l, n_src, l, y);
            // `Phi_c -= Phi_i[0]` (and the other two): per part.
            for (int c = 0; c < 3; c++)
                for (std::size_t q = 0; q < per; ++q) x[c][q] = x[c][q] - y[c][q];
        }
        double *o[LANES];
        for (int l = 0; l < LANES; ++l) o[l] = reinterpret_cast<double *>(O + (r0 + std::min(l, nl - 1)) * os);
        for (std::ptrdiff_t j = 0; j < cols; ++j) {
            double sr[3][LANES], si[3][LANES];
            for (int c = 0; c < 3; c++) {
                // reduce_phi_band's chain per lane: a zeroed y, then
                // `y += a * x` per stored entry, unfused.
                double yr[LANES], yi[LANES];
                for (int l = 0; l < LANES; ++l) yr[l] = 0.0, yi[l] = 0.0;
                const Csc &m = M[c];
                for (int64_t e = m.ptr[j]; e < m.ptr[j + 1]; ++e) {
                    const double ar = m.val[2 * e], ai = m.val[2 * e + 1];
                    const double *xr = x[c] + 2 * LANES * m.ind[e];
                    const double *xi = xr + LANES;
                    for (int l = 0; l < LANES; ++l) {
                        const double pr = ar * xr[l] - ai * xi[l];
                        const double pi = ar * xi[l] + ai * xr[l];
                        yr[l] = yr[l] + pr;
                        yi[l] = yi[l] + pi;
                    }
                }
                for (int l = 0; l < LANES; ++l) sr[c][l] = yr[l], si[c][l] = yi[l];
            }
            for (int l = 0; l < nl; ++l) {
                o[l][2 * j] = (sr[0][l] + sr[1][l]) + sr[2][l];
                o[l][2 * j + 1] = (si[0][l] + si[1][l]) + si[2][l];
            }
        }
    }
}

void scatter(const Block &b, std::ptrdiff_t rows, std::ptrdiff_t n_src, cd *out[3]) {
    std::vector<double> buf(static_cast<std::size_t>(3 * 2 * LANES * n_src));
    const std::size_t per = static_cast<std::size_t>(2 * LANES * n_src);
    double *x[3] = {buf.data(), buf.data() + per, buf.data() + 2 * per};
    for (std::ptrdiff_t r = 0; r < rows; ++r) {
        gather_row(b, r, n_src, 0, x);
        for (int s = 0; s < 3; ++s)
            for (std::ptrdiff_t n = 0; n < n_src; ++n)
                out[s][r * n_src + n] = cd(x[s][2 * LANES * n], x[s][2 * LANES * n + LANES]);
    }
}

Block band_block(const Plan &p, const Values &v, int64_t i0, int64_t i1, std::vector<int64_t> &near_idx) {
    Block b;
    b.cls = p.cls().data() + i0 * p.N;
    b.rep = reinterpret_cast<const double *>(v.cls.data());
    b.n_cls = p.n_classes();
    const int64_t lo = i0 * p.N, hi = i1 * p.N;
    const auto e0 = std::lower_bound(p.near_flat().begin(), p.near_flat().end(), lo);
    const auto e1 = std::lower_bound(e0, p.near_flat().end(), hi);
    near_idx.assign(e0, e1);
    for (int64_t &e : near_idx) e -= lo;
    b.near_idx = near_idx.data();
    b.near_val = reinterpret_cast<const double *>(v.near.data()) + 6 * (e0 - p.near_flat().begin());
    b.n_near = static_cast<int64_t>(near_idx.size());
    if (p.coded) set_code_signs(b);
    return b;
}

}  // namespace pair_reuse
}  // namespace sinusoidal
}  // namespace mw
