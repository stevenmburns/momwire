// razor2p's fill-pair reuse (momwire#1971): the census that finds which
// (observer, source segment) pairs of the fill share a value, the evaluation
// of one representative per class, and the gather of each window's moments
// from the class values. Python-free: the solve core (`SolveContext::fill`,
// the executables and the C++ front route) and the Python route (through
// `_accel_razor_solve.cpp`'s binding) run the same plan, values and gather.
//
// WHAT THE MOMENTS READ. Razor's fill evaluates, for every observer point p
// (under the NEC-5 path rule the segment centroids, shared by T1 and T2) and
// every source segment s,
//
//     M0 = int_0^h g dtau,   M1 = int_0^h tau g dtau
//
// (`seg_moments`). Of the pair's geometry the kernel reads exactly: u_r, the
// observer's coordinate along the source axis from the source's start; perp,
// the squared distance from that axis; the source's length h and kernel
// radius a; and, under the extended kernel, whether the pair is eligible
// (group_i[p] == group_j[s] >= 0), the source's EK radius and the observer's
// tube b. Nothing else: the tangents of both segments enter the fill only
// later, in the contraction (`assemble_t1`, `t2_rows`), from the pair's own
// tables. So two pairs agreeing on those quantities have the same M0 and M1,
// whatever rigid motion (reflections included) takes one to the other. The
// PEC image block is the same function of a mirrored source set, and so is
// the composing ground's C2-scaled image (C2 is a constant), so the image
// block takes the same key.
//
// ORIENTATION (#1969's rule, in razor's form). A source's stored tangent is a
// convention: reversed, u_r -> h - u_r, M0 is unchanged and M1 -> h M0 - M1.
// Each pair is keyed with v = u_r - h/2 folded to |v|, and a pair with v < 0
// carries a flip bit; the gather applies M1 -> h M0 - M1 when a member's
// flip differs from its representative's. A pair whose |v| quantises to 0 is
// symmetric and keeps flip 0.
//
// THE KEY quantises |v| and sqrt(perp) with a step of 2^-KEY_BITS times
// the pair's singular distance D = max(|v| - h/2, sqrt(perp), a), rounded down
// to a power of two (so the step is a function of the pair alone); h, a, the
// EK radius and b enter exactly, by type. A member differs from its
// representative by at most half a step in each position, about
// 2^-(KEY_BITS+1) relative to the scale at which the kernel varies. Pairs
// straddling a quantum land in neighbouring classes, which costs compression,
// never accuracy, and two different keys never share a class (exact equality
// of the quantised integers, no hashing of the key into fewer bits).
//
// h is exact, not quantised against D (momwire#1971 review): M0 is
// proportional to h, so its error counts relative to h, and a step of D is
// coarser than h's own by D/h. Far pairs then merged segments whose lengths
// differ by ~1e-11 relative. On a 0.3 m square loop at 137 kHz (440
// segments, one side 4e-11 longer) R moved 8.1e-5, the same under
// translation, while `verify` passed every pair at ~4e-11; with h exact it
// moves 1.7e-9, inside the flag-off fill's own translation noise. A step of
// h itself (2^-40 h) still moved R 3e-6 on the same loop at L/lambda 3e-5,
// where the off fill's noise is 1.4e-9. The cost is sharing across lengths
// that differ by rounding only: 24 radials at random angles from an off-axis
// hub, 40 segments each, keep 728,508 classes of 1e6 pairs against 674,968
// with h on a step of itself.
//
// EXCLUDED: a pair whose observer lies within NEAR_LENGTHS kernel scales of
// the source's centre (max(|v|, sqrt(perp)) < NEAR_LENGTHS max(h, a): the
// self term and its neighbours) is a class of its own, evaluated in its own
// frame, as the fill would. Their values carry the near-singular part the
// basis sums cancel, so any rounding of their geometry other than the
// structure's own is amplified into the answer (the sinusoidal plan's
// measured reason, `_sin_pair_reuse.NEAR_LENGTHS`). The scale is the longer
// of the source's length and its radius, because where h < a the kernel
// varies over a, not h: on a hairpin whose 2 mm bridge has 21 segments of
// 95 um on a 1 mm radius (cond(Z) ~ 7e9), a near rule in h alone shared
// bridge-on-bridge pairs a few h apart whose v differed by coordinate
// rounding, and Zin moved 1.5e-5 with the reduced kernel and exact_kernel
// off, 2.5e-7 with the extended kernel, while every pair passed `verify`
// at 2e-13 (momwire#1971 review). With max(h, a) the same deck moves 7e-11
// and 1e-12 (the reduced kernel's off fill itself moves 2e-8 between the
// deck at the origin and translated 3.3 m). Where h >= a, the usual mesh,
// the rule is the one in h.
//
// THE COMPOSING GROUND'S REMAINDER (momwire#1970's height-aware key). Its
// value for observer segment o (centroid, tangent t_o) and source segment s,
// Q_p = sum_j W_p[s, j] t_o . F(r_o, r_j) . t_s, p = 0, 1, reads the
// remainder grid at the horizontal separation and the SUM of the two heights
// above the plane, and the two tangents; it is invariant only under the
// motions that keep the plane: horizontal translations, rotations about a
// vertical axis and reflections in vertical planes. Its key is the
// configuration's complete invariants under that group: the height sum, |d|
// (d the horizontal separation of the centroids), d.a, d.b, a.b (a, b the
// horizontal parts of t_s, t_o), t_s,z, t_o,z, quantised against the
// distance to the image, D = sqrt(|d|^2 + hsum^2) (floored at h), and h,
// exact by type as in the direct key. Reversing the source tangent negates
// d.a, a.b, t_s,z and takes (Q0, Q1) -> (-Q0, h Q0 - Q1)... with the sign of
// the reversed current, -(h Q0 - Q1); reversing
// the observer's negates d.b, a.b, t_o,z and both values. Each pair is keyed
// in the orientation whose key is the largest of the four, its two flips in
// a code the gather undoes. A path row on the segment is that segment's row
// times its wing sign (the projection is linear in the observer's tangent,
// exactly so in floating point).
//
// MEMORY: one int32 per pair per block (the direct block, the image block,
// the remainder): a quarter of Z each, plus the class tables.
#pragma once

#include "_mw_core.h"
#include "_pair_reuse_core.h"

#include <memory>
#include <string>
#include <vector>

namespace somm_proj {
struct GridView;
}

namespace mw {
namespace razor {
namespace reuse {

// Relative resolution of the key, in bits. Congruent segments placed by
// rotations about an origin carry coordinate rounding of a few 1e-15 m at a
// 15 m scale, which against a 0.1 m pair distance is ~2^-45: at 48 bits most
// such pairs would straddle a quantum. 40 bits merges them and bounds a
// member's argument error at 2^-41 of the pair's own scale.
constexpr int KEY_BITS = 40;
// Pairs nearer than this many source lengths are their own class.
constexpr double NEAR_LENGTHS = 4.0;
// Observer rows per representative evaluation call, at most.
constexpr int REP_ROWS = 16;
// The `verify` mode's per-pair bound: |reused - direct| over the pair's
// largest of |M0|, |M1| / h (the remainder: |Q0|, |Q1| / h).
constexpr double VERIFY_TOL = 1e-9;
// Segments below which the reuse steps aside (both routes: the Python
// route's `_pair_reuse_refusal`, the solve core's `reuse_begin`), the census
// costing about what a small deck's fill saves: in free space and over the
// PEC image, and over a finite ground (the composing Sommerfeld ground, the
// reflection coefficient), whose image term is dearer per pair and repays
// sooner. `verify` (mode 2) ignores them. Measured, Rocketlake, nec5-razor2p
// through the NEC-5 shell (the Python route), wall seconds per deck, on / off,
// cold (one deck, fresh process, the plan built) and warm (the deck streamed
// 6x through one process, the plan found), median of 3 (momwire#1971):
//
//   deck     segs  freqs   8 threads cold / warm   4 threads cold / warm
//   sfd_h1     41   101    1.01 / 0.98             1.02 / 1.03
//   sfd_h3    123   101    1.02 / 1.02             0.94 / 1.16
//   sfd_h5    203   101    1.09 / 1.02             1.03 / 1.03
//   sfd_h9    365   101    1.00 / 0.98             0.98 / 0.98
//   p_m4      425    13    1.19 / 0.96             1.05 / 0.98   (0.25 s)
//   sky_m6    525    13    1.00 / 0.95             1.16 / 1.01   (0.33 s)
//   sfd_h13   529   101    0.86 / 1.01             0.94 / 0.95
//   gsfd_h1    41   101    1.04 / 1.07             1.04 / 1.05   Sommerfeld
//   gsfd_h3   123   101    0.91 / 1.02             0.91 / 0.98   Sommerfeld
//   gsfd_h5   203   101    0.91 / 0.90             0.95 / 0.93   Sommerfeld
//   gsfd_h9   365   101    0.82 / 0.81             0.74 / 0.83   Sommerfeld
//   gsky_m6   525    13    0.70 / 0.76             0.78 / 0.72   Sommerfeld
//
// Free space loses up to 9 % below a few hundred segments and is even to
// about 400; over Sommerfeld only the 41-segment deck loses, 123 is even.
constexpr int64_t SMALL_DECK_SEGS_FREE = 400;
constexpr int64_t SMALL_DECK_SEGS_GROUND = 100;
// The thresholds in force (the constants, unless the TEST HOOK moved them):
// the finite ground's when `finite`.
int64_t small_deck_segs(bool finite);
// TEST HOOK: move the thresholds (a negative value restores its constant), so
// the gates can run the reuse on cheap decks below them.
void set_small_deck_segs(int64_t free_space, int64_t finite_ground);
// THE ALLOWANCE (momwire#2123, the common core's `plan_allowance`): each
// family keeps at most three quarters of its pairs as classes
// (`pair_reuse::class_cap`, 4/3x compression; the direct family's pairs are n_blocks * n_seg^2, the
// remainder's n_seg^2), checked after every chunk of rows, so a deck that
// does not compress stops within a chunk of the cap, "more than C classes
// for P pairs (the cap is 4/3x compression)"; and the kept plan (`Plan::nbytes` and one
// wavenumber's class values, 32 B a class) is held to 4 B per pair per map
// plus `kept_bytes_per_class` (49 B, the plan's own per-class layout:
// `pair_reuse::class_bytes`' 41 and at most one row and one source in its
// group's lists; measured 41.5 B a class on the 10-element Yagi, 42.9 on the
// 155-segment triangle and dipoles) per class the cap allows, 40.75 B per
// pair per map, "the kept plan would exceed its allowance of X MB". The census's own tables (its hash
// tables, representatives and chunk buffers) are a transient freed before
// the fill's Z is allocated: they are counted (`census_peak_bytes`) but not
// bounded, the cap stopping them. One exception: the solve core keeps its Z
// pool between solves of one context (`SolveContext::z_`), so there a
// census for a NEW geometry runs beside the last solve's Z (a plan found in
// the cache runs no census).
//
// The cap replaced a 3/4 class fraction and a fixed 256 MB budget of plan
// and census together (momwire#1971, #2019), which no fraction serving
// sky_m24 (0.34 classes per pair) could hold at the hosted 3000-segment cap.
// Measured then, Rocketlake, the Python route at 8 threads, peak RSS, reuse
// on / off, with no bound:
//
//   deck (segs)                 classes/pairs   no bound
//   75 random wires x 40 (3000) free    0.99    2045 / 296
//   the same over PEC                   0.99    3865 / 328
//   24 random radials x 40 (960) free   0.57     188 / 141
//   the #1971 benchmark decks: fd_h27 0.0095, fd_h51 0.005, gfd_h51 0.003
//     (remainder 0.001), p_m24 0.25, sky_m24 0.34, gsky_m24 0.34 (0.34)
//
// At the cap of 1/2 the random wires decline within a chunk of it, and the
// radial screen at random angles (0.57), the gates' hairpin and lengthened
// loop (between 1/2 and 3/4) decline too; every benchmark deck is served.
//
// Measured under the allowance (momwire#2123, i7-8550U, 8 threads, the
// Python route), the 10-element Yagi (2,154 segments), which declined under
// the 256 MB budget: free space, 0.85 M classes (5.5x), plan 51 MB, census
// counted at 312 MB (363 B a class beyond the map) where the RSS rose 143 MB
// over it (155 B a class), peak RSS 358 MB on / 280 off, |dZin| / |Zin|
// 9e-14; Sommerfeld, 1.68 M + 0.83 M remainder classes (5.5x), plan 152 MB,
// counted 658 MB (252 B a class) where the RSS rose 369 MB (132 B), peak
// RSS 509 / 250, 7e-14. The count is the census's own estimate (its tables
// at 1.5x capacity, vectors at their doubled reserve), so it reads about
// twice the pages the census touches.
// TEST HOOK: the kept plan's allowance in bytes in place of the derived one
// (a value <= 0 restores it), so a test can reach that decline on a small
// deck. A plan is keyed on the allowance it was built under. Bound for
// Python's tests only; nothing in the executables sets it.
void set_plan_allowance(int64_t bytes);
// The allowance in force for a plan of these families' maps: the hook's
// bytes, or the sum of `pair_reuse::plan_allowance` over every map.
int64_t plan_allowance(const pair_reuse::Plan &p);
// What a class of family `f` costs the kept plan, the allowance's per-class
// term.
int64_t kept_bytes_per_class(const pair_reuse::FamilySpec &f);
int64_t plan_allowance_override();

// What the plan reads. Views into the caller's arrays; the plan copies what
// it keeps.
struct Geometry {
    view<const double, 2> cent;      // (n_seg, 3): the centroid observers
    view<const int64_t, 1> group_i;  // (n_seg) EK labels of the observers, or empty
    view<const double, 1> b_obs;     // (n_seg) or empty
    view<const double, 1> seg_h, a;  // (n_seg)
    // The source sets: [0] real, [1] the image (when has_image).
    view<const double, 2> p0[2], t[2];
    view<const int64_t, 1> group_j[2];
    view<const double, 1> a_ek[2];
    bool has_image = false;
    // The composing ground's remainder: planned when set. Its sources are the
    // REAL segments (p0[0], t[0]) and its observers the centroids, along the
    // real segments' own tangents.
    bool remainder = false;
    double ground_z = 0.0;
};

// razor's plan is the core's (`_pair_reuse_core.h`), with families
// fam[DIRECT] (W = 4, one flip bit, blocks direct and, over PEC, image) and,
// over the composing ground, fam[REMAINDER] (W = 9, a 2-bit code, one block),
// both n_seg x n_seg. A declined plan keeps both specs. razor's own decline
// reasons, beside the core's (poor compression, the budget): a deck the
// census cannot code (more classes than an int32 entry holds, more segment
// types than the key's type field, more segments than `MAX_SEGS`).
using Plan = pair_reuse::Plan;
using Group = pair_reuse::Group;
constexpr int DIRECT = 0, REMAINDER = 1;
inline int64_t n_segs(const Plan &p) { return p.fam[DIRECT].spec.rows; }
inline bool has_image(const Plan &p) { return p.fam[DIRECT].spec.n_blocks == 2; }
inline bool has_remainder(const Plan &p) { return p.fam.size() > REMAINDER; }
// The cache's tag for razor's plans.
constexpr const char *BASIS = "razor";

// The per-wavenumber class values (a solve's own; the plan is shared).
struct Values {
    double k = 0.0;
    // Direct, per class c: M0 at d[2c], M1 at d[2c + 1], so a gather's one
    // random read is one 32 B value, not two from separate tables.
    std::vector<cd> d;
    std::vector<cd> q0, q1;  // the remainder, per class
    bool has_d = false, has_q = false;
};

// The plan for `g`, from the process's cache or built and kept. Never null.
// A deck the census cannot code comes back DECLINED (`Plan::declined` says
// why) rather than throwing: the reuse is an optimisation, and a fill it
// cannot serve runs as without it.
// `built`, when given, says whether this call ran the census.
std::shared_ptr<const Plan> plan_for(const Geometry &g, bool *built = nullptr);
// Segments past which a plan declines: an observer-by-source offset of the
// largest deck must fit the int32 the groups store.
constexpr int64_t MAX_SEGS = 46340;
// TEST HOOK: decline the direct census past `direct` classes and the
// remainder's past `remainder` (0 is no limit beyond the int32 coding's), so
// a test can reach the decline without a deck of 2^26 classes. A plan is
// keyed on the limits it was built under. Bound for Python's tests only;
// nothing in the executables sets it.
void set_class_limit(int64_t direct, int64_t remainder);
int64_t class_limit(int family);  // 0 direct, 1 remainder
// Plans built and found (process totals, for tests), and of those built the
// ones that came back declined: a declined plan serves nothing (its fill runs
// as off), so built - declined is what proves a fill reused.
using CacheCounters = pair_reuse::Counters;
CacheCounters cache_counters();
void clear_cache();

// TEST HOOK (the adversarial gates): a deliberately wrong key, so a test can
// show the gates catch it. 0 is the key above; PROBE_NO_HEIGHT drops the
// remainder's height sum, PROBE_NO_FLIP the direct key's orientation (the
// gather then never applies M1 -> h M0 - M1), PROBE_NO_TYPE the direct key's
// radius / EK type, PROBE_Q_NO_FLIP the remainder's orientation code.
// PROBE_NO_FOLD is not wrong, only uncompressed: the direct key keeps the
// sign of v, so a test can count what the orientation fold merges.
// PROBE_NEAR_H is the near rule in source lengths alone, which shares the
// near pairs of a segment shorter than its radius. A plan is
// keyed on the probe it was built under. Bound for Python's tests only;
// nothing in the executables sets it.
enum Probe : int {
    PROBE_NONE = 0,
    PROBE_NO_HEIGHT = 1,
    PROBE_NO_FLIP = 2,
    PROBE_NO_TYPE = 3,
    PROBE_Q_NO_FLIP = 4,
    PROBE_NO_FOLD = 5,
    PROBE_NEAR_H = 6,
};
void set_probe(int probe);
int probe();

// Each class's M0, M1 at real k: the shipped `seg_moments` over each group's
// observer rows against the union of their representatives' sources.
void eval_direct(const Plan &p, const Geometry &g, view<const double, 1> xg,
                 view<const double, 1> wg, double k, uintptr_t cancel_flag, Values &out);

// The remainder's class values through the grid (the C++ routes): each
// group's centroids along their own tangents projected against the union's
// remainder nodes (`field_proj_batch`) and reduced on the moment weights.
// `rem_src` / `rem_t_src` (n_seg * n_qp, 3), W (2, n_seg, n_qp).
void eval_remainder(const Plan &p, const Geometry &g, const somm_proj::GridView &grid,
                    size_t n_reg, const double *rem_src, const double *rem_t_src,
                    const double *W, ptrdiff_t n_qp, double k, uintptr_t cancel_flag,
                    Values &out);
// One group's remainder moments, computed elsewhere (the Python route's own
// producer): mom (|obs|, |src|, 2) C-contiguous.
void store_remainder(const Plan &p, size_t group, const cd *mom, Values &out);

// The window's moments: rows `need` (observer indices) against every source
// of `block`, into C-contiguous (n_need, n_seg) m0 / m1.
void gather_direct(const Plan &p, int block, const int64_t *need, ptrdiff_t n_need,
                   view<const double, 1> seg_h, const Values &v, cd *m0, cd *m1);
// The remainder rows: row r is segment seg[r]'s, times sign[r], (m, n_seg, 2).
void gather_remainder(const Plan &p, const int64_t *seg, const double *sign, ptrdiff_t m,
                      view<const double, 1> seg_h, const Values &v, cd *out);

// What one block's entries read beside the moments (`assemble_rows`): the
// tables `assemble_t1` (or, `weighted`, the constant-mirror window of
// `assemble_t1_weighted`) and the fused `t2_rows` take, indexed by GLOBAL
// basis row and path point (o = 2 r + p: the NEC-5 path rule, the only one
// the plan serves).
struct RowTables {
    view<const int64_t, 1> s_a, s_b;           // (n_basis) wing segments
    view<const double, 1> h_a, h_b, q_a, q_b;  // (n_basis)
    view<const bool, 1> fall_a, fall_b;        // (n_basis)
    view<const double, 1> wts;                 // (2 n_basis) path weights
    view<const int64_t, 1> grounded;           // rows whose T2 drops its A centroid
    // Unweighted (free space, the PEC image): the path tangents (2 n_basis,
    // 3) and the sigma-folded source tangents (3, n_basis).
    view<const double, 2> tans, td_a, td_b;
    // Weighted (the composing ground's image, WINDOW_RULE_CONSTANT_MIRROR):
    // the path tangents (2 n_basis, 3), the segment tangents (n_seg, 3), the
    // wing signs (n_basis) and C2, which is the window's coefficient and the
    // w_Phi that scales T2's M0.
    bool weighted = false;
    view<const double, 2> obs_tans, src_t;
    view<const double, 1> sig_a, sig_b;
    cd c2;
    double c_a_imag = 0.0, c_phi_imag = 0.0;  // c_A, c_Phi, both purely imaginary
};

// momwire#2144: rows [lo, hi) of one block's c_A T1 - T2 / c_Phi from the
// class values, into `out` ((hi - lo, n_basis), any strides): bit for bit
// the entries `gather_direct`, `assemble_t1` (or its weighted twin) and the
// fused `t2_rows` give, without the window's (n_need, n_seg) moment planes
// or its T1. Unchopped rows only (a chopped fill declines the reuse).
void assemble_rows(const Plan &p, const Values &v, int block, ptrdiff_t lo, ptrdiff_t hi,
                   const RowTables &t, view<const double, 1> seg_h, view<cd, 2> out);

// The fill report's wire map (momwire#2016, `pair_reuse::wire_map`): for
// family f (0 direct, 1 image, 2 remainder), observer wire i and source wire
// j, out[((f * W + i) * W + j) * 3 + c] counts that block's pairs computed
// (c = 0, the first pair of a far class), reused (1) and near (2). One pass
// over the class maps, with one bit per class of scratch.
void wire_map(const Plan &p, const int32_t *wire_of_seg, int64_t n_wires, int64_t *out);

// `verify`: the largest per-pair error of `got` against `ref` (two moment
// planes, n rows by n_seg, the second divided by h), relative to the pair's
// own largest. `stride` is the distance between a pair's two values (1 for
// separate planes given as a1/b1, 2 for an interleaved (.., 2) table).
double verify_error(const cd *ref0, const cd *ref1, const cd *got0, const cd *got1,
                    ptrdiff_t n_rows, ptrdiff_t n_seg, ptrdiff_t stride,
                    view<const double, 1> seg_h);

}  // namespace reuse
}  // namespace razor
}  // namespace mw
