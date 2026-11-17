// razor-2p's fill-pair reuse (momwire#1971): the census that finds which
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
// THE KEY quantises |v|, sqrt(perp) and h with a step of 2^-KEY_BITS times
// the pair's singular distance D = max(|v| - h/2, sqrt(perp), a), rounded down
// to a power of two (so the step is a function of the pair alone); a, the EK
// radius and b enter exactly, by type. A member differs from its
// representative by at most half a step in each, about 2^-(KEY_BITS+1)
// relative to the scale at which the kernel varies. Pairs straddling a
// quantum land in neighbouring classes, which costs compression, never
// accuracy, and two different keys never share a class (exact equality of the
// quantised integers, no hashing of the key into fewer bits).
//
// EXCLUDED: a pair whose observer lies within NEAR_LENGTHS source lengths of
// the source's centre (max(|v|, sqrt(perp)) < NEAR_LENGTHS h: the self term
// and its neighbours) is a class of its own, evaluated in its own frame, as
// the fill would. Their values carry the near-singular part the basis sums
// cancel, so any rounding of their geometry other than the structure's own
// is amplified into the answer (the sinusoidal plan's measured reason,
// `_sin_pair_reuse.NEAR_LENGTHS`).
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
// horizontal parts of t_s, t_o), t_s,z, t_o,z, and h, quantised against the
// distance to the image, D = sqrt(|d|^2 + hsum^2) (floored at h). Reversing
// the source tangent negates d.a, a.b, t_s,z and takes (Q0, Q1) -> (-Q0,
// h Q0 - Q1)... with the sign of the reversed current, -(h Q0 - Q1); reversing
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
// Bytes of plans the process keeps between fills.
constexpr size_t SHARED_PLAN_BYTES = size_t(1) << 30;

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

struct Group {
    uint8_t block = 0;
    std::vector<int32_t> obs, src;  // observer rows, the union of their reps' sources
    std::vector<int32_t> cls, at;   // each class's value at at = row * |src| + col
};

struct Plan {
    int64_t n_seg = 0;
    bool has_image = false, remainder = false;
    std::string fingerprint;
    // Direct moments: entry = class << 1 | flip, one map per block.
    std::vector<int32_t> dmap[2];
    int64_t n_dcls = 0;
    std::vector<uint8_t> drep_flip;
    std::vector<Group> dgroups;
    // The remainder: entry = class << 2 | code.
    std::vector<int32_t> qmap;
    int64_t n_qcls = 0;
    std::vector<uint8_t> qrep_code;
    std::vector<Group> qgroups;
    // The census, for tests and benchmarks.
    int64_t pairs = 0, near_pairs = 0, q_pairs = 0;
    int64_t rep_evals = 0, q_rep_evals = 0;  // pairs the representatives' calls evaluate
    double census_seconds = 0.0;
    size_t nbytes() const;
};

// The per-wavenumber class values (a solve's own; the plan is shared).
struct Values {
    double k = 0.0;
    std::vector<cd> d0, d1;  // direct, per class
    std::vector<cd> q0, q1;  // the remainder, per class
    bool has_d = false, has_q = false;
};

// The plan for `g`, from the process's cache or built and kept. Never null;
// throws std::runtime_error when a deck is too large to code (class ids past
// int32).
std::shared_ptr<const Plan> plan_for(const Geometry &g);
// Plans built and found (process totals, for tests).
struct CacheCounters {
    uint64_t built = 0, found = 0;
};
CacheCounters cache_counters();
void clear_cache();

// TEST HOOK (the adversarial gates): a deliberately wrong key, so a test can
// show the gates catch it. 0 is the key above; PROBE_NO_HEIGHT drops the
// remainder's height sum, PROBE_NO_FLIP the direct key's orientation (the
// gather then never applies M1 -> h M0 - M1), PROBE_NO_TYPE the direct key's
// radius / EK type, PROBE_Q_NO_FLIP the remainder's orientation code.
// PROBE_NO_FOLD is not wrong, only uncompressed: the direct key keeps the
// sign of v, so a test can count what the orientation fold merges. A plan is
// keyed on the probe it was built under. Bound for Python's tests only;
// nothing in the executables sets it.
enum Probe : int {
    PROBE_NONE = 0,
    PROBE_NO_HEIGHT = 1,
    PROBE_NO_FLIP = 2,
    PROBE_NO_TYPE = 3,
    PROBE_Q_NO_FLIP = 4,
    PROBE_NO_FOLD = 5,
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
