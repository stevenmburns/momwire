// bs2's fill-pair reuse on the common core (momwire#2063 stage 3): the census
// that finds which (observer segment, source segment) pairs of the Galerkin
// moment fill share their moments, and the gather of a window of the moment
// tensor from one value per class. Python-free; `_accel_bspline_pair_reuse.cpp`
// binds it for the Python route.
//
// WHAT THE MOMENTS READ. The off-edge fill (`seg_seg_full_moments_bspline*`)
// gives, for observer segment i (end points l_i, r_i) and source segment j,
//
//     J[p, P, i, j] = int int u^p u'^P G(R) du du',   R^2 = |x - x'|^2 + a_i^2,
//
// x = l_i + t (r_i - l_i), x' = l_j + t' (r_j - l_j) on each segment's own
// Gauss rule, u, u' the arc lengths. Of the pair's geometry it reads exactly:
// the two lengths h_i, h_j (the moments' polynomial variables and measures),
// the observer's radius a_i, |x - x'|^2, and the quadrature tier (centre
// distance over the longer length, momwire#906) with its per-segment phase
// guard (k h, momwire#920). With m the centre difference and A = r_i - l_i,
// B = r_j - l_j,
//
//     |x - x'|^2 = m.m + 2 s m.A - 2 s' m.B + s^2 A.A + s'^2 B.B - 2 s s' A.B,
//
// s = t - 1/2, s' = t' - 1/2: so (h_i, h_j, m.m, m.A, m.B, A.B) are the
// pair's complete invariants under every rigid motion, reflections included
// (the Gram matrix of A, B, m), and two pairs agreeing on them have the same
// moments. Under the extended kernel the pair also reads whether it is
// eligible (group_i[i] == group_j[j] >= 0) and, when eligible, the source
// tube's radius b_j. Tangents enter only later, in the assembly, from the
// pair's own segments; the basis functions likewise (`supp`, `polys`), so a
// junction or an end basis needs nothing of the plan.
//
// A pair's moments do not depend on the other pairs of its call (the tier and
// phase guard are per pair, momwire#920; the lane path is gated to the bit
// against the pair-at-a-time walk, momwire#1290), so a class value evaluated
// in a group's call is the member's own value up to the key's rounding, and
// a near pair's own class is its fill's bits.
//
// THE KEY quantises m.m, m.A, m.B and A.B on one step, 2^-KEY_BITS of D^2,
// D = max(|m| - (h_i + h_j) / 2, a_i) floored to a power of two: the error a
// member's R^2 carries is then at most ~2^-KEY_BITS of R^2 wherever on the
// pair, whatever the ratio h / D. h_i, h_j enter on a step of their own
// (`LENGTH_BITS`), never of D (razor's finding, momwire#1971: the moments are
// proportional to the lengths, so a length on a step of D errs by D / h
// relative); a_i and b_j enter exactly, by type.
//
// RECIPROCITY (one code bit). Read from the source's side, the pair's moments
// are the transpose, J[P, p, j, i] = J[p, P, i, j], when a_i = a_j (and,
// eligible, b_i = b_j): the same integral with the regularisation radius
// unchanged. Such a pair keys on the smaller of its two forms, the second
// being the first with h_i, h_j swapped and m -> -m; the gather transposes a
// member whose form differs from its representative's. A copy either way.
//
// ORIENTATION is not folded: a reversed segment negates m.A or m.B, so it
// keys apart. Folding would take a (d+1) x (d+1) binomial on each side
// (u -> h - u), arithmetic on every copy; it is left for when its drift is
// measured (on the 10-element Yagi it would halve the classes again).
//
// EXCLUDED.
// - SAME-EDGE pairs are not the off-edge kernel's: the fill overwrites them
//   with the analytic static and regularised split (or the offset table), so
//   they share one SENTINEL class (`Key::Near`, unlisted) the gather writes
//   but nothing reads.
// - NEAR pairs, |m| < NEAR_LENGTHS max(h_i, h_j, a_i) (touching segments at
//   a kink or a junction, and their neighbours), are their own classes, as
//   razor's (`_razor_pair_reuse_core.h`, the hairpin): their values carry the
//   near-singular part the basis sums cancel.
// - TIER TIES, a centre distance within TIE_REL of a ladder threshold, are
//   their own classes: a member there could take the other tier from its
//   representative (`_bspline_fill_core.cpp`'s TIE_REL, the same 1e-9).
#pragma once

#include "_pair_reuse_core.h"

#include <complex>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace mw {
namespace bspline {
namespace reuse {

using cd = std::complex<double>;
using Plan = pair_reuse::Plan;
using Group = pair_reuse::Group;

// The key's relative resolution in bits: razor's, for razor's reason
// (coordinate rounding of congruent segments placed by rotation is ~2^-45 of
// a pair's distance, so 48 bits would split them across quanta).
constexpr int KEY_BITS = 40;
// Pairs nearer than this many kernel scales are their own class.
constexpr double NEAR_LENGTHS = 4.0;
// Centre distances within this of a ladder threshold, relative, are their
// own class.
constexpr double TIE_REL = 1e-9;
// Observer rows per representative evaluation call, at most.
constexpr int REP_ROWS = 16;
// Segments past which a plan declines: an observer-by-source offset of the
// largest deck must fit the int32 the groups store.
constexpr int64_t MAX_SEGS = 46340;
// The lengths' resolution in bits (the key's `length_type`; 0 keys them
// exactly): a step of 2^-44 of the length's power-of-two floor. The segments
// of a uniform edge differ in their last few bits (each length is formed
// from end points rounded at the structure's scale), so exact lengths split
// congruent pairs: on the 10-element Yagi with uniform cores the four
// lengths it has take 21 bit patterns per element, and the plan keeps
// 150,015 classes exact against 91,866 at 44 bits. A member's lengths then
// err by at most 2^-45 relative, far under the 2^-40 of h that moved
// razor2p's 137 kHz loop (`_razor_pair_reuse_core.h`): measured over the
// stage's deck set, free, PEC and Sommerfeld, EK on and off, the worst
// |dZin|/|Zin| is the same at 0 and 44 bits (6.6e-14 / 6.9e-14; a 0.3 m loop
// at 137 kHz 1.4e-8 both, where its off fill moves 5e-7 translated 3.7 m).
constexpr int LENGTH_BITS = 44;
// Segments below which the reuse steps aside, in the core's words ("too few
// segments to repay the plan (N < x)").
constexpr int64_t SMALL_DECK_SEGS = 200;
// The cache's tag for bs2's plans.
constexpr const char *BASIS = "bs2";

// What a class costs the kept plan, from the layout: one wavenumber's
// (d+1)^2 complex moments, its group entry and representative's code
// (`pair_reuse::class_bytes`) and at most one row and one source in its
// group's lists. d = 2: 144 + 9 + 8 = 161 B.
int64_t kept_bytes_per_class(int degree);
// THE ALLOWANCE (momwire#2123): 4 B a pair for the class map plus
// `kept_bytes_per_class` per class the 2x cap allows, `pair_reuse::plan_allowance`
// (84.5 B a pair at d = 2), or the TEST HOOK's bytes.
int64_t plan_allowance(const Plan &p, int degree);
void set_plan_allowance(int64_t bytes);  // TEST HOOK, <= 0 restores
void set_class_limit(int64_t classes);   // TEST HOOK, 0: the coding's only
void set_small_deck_segs(int64_t segs);  // TEST HOOK, < 0 restores
int64_t small_deck_segs();
void set_length_bits(int bits);           // TEST HOOK, < 0 restores
int length_bits();

// What the plan reads; the arrays are C-contiguous and outlive the call.
struct Geometry {
    int64_t n = 0;                      // segments
    const double *seg_l = nullptr;      // (n, 3)
    const double *seg_r = nullptr;      // (n, 3)
    const double *a = nullptr;          // (n): each observer row's radius
    const int32_t *edge = nullptr;      // (n): the same-edge block it belongs to
    const int64_t *group_i = nullptr;   // (n) EK labels, or null (EK off)
    const int64_t *group_j = nullptr;   // (n), with group_i
    const double *b_src = nullptr;      // (n) EK source tube radii, or null (b = a)
    std::vector<double> thresholds;     // the pair-order ladder's ratios
    int degree = 2;
};

// The plan for `g`, from the process's cache or built: one family, "direct",
// n x n, one code bit (reciprocity). Never null; a deck the census does not serve comes
// back DECLINED with the reason. `built` says whether this call ran it.
std::shared_ptr<const Plan> plan_for(const Geometry &g, bool *built = nullptr);

// TEST HOOK (the stage's ExternalClasses sub-gate): a plan whose classes are
// `cls` (n x n, each class's pairs one value; numbered 0..C-1, any order),
// with its groups. Not cached, not capped.
std::shared_ptr<const Plan> plan_of_classes(const int32_t *cls, int64_t n);

// Pairs of the plan in its sentinel class (the same-edge blocks), and the
// near pairs its key left their own classes.
int64_t same_edge_pairs(const Plan &p);
int64_t near_pairs(const Plan &p);

// The moment window of observer rows r0..r1 against every source:
// out[((p * d1 + q) * (r1 - r0) + i) * n + s] = values[c * d1 * d1 + p * d1 + q]
// for pair (r0 + i, s) of class c. Copies only.
void gather(const Plan &p, int64_t r0, int64_t r1, const cd *values, int d1, cd *out);

// THE SOMMERFELD REMAINDER's copy map (3b): its own family, of EDGE pairs,
// because the fused kernel copies whole blocks between edges. The remainder
// reads the plane, so it is invariant only under the motions that keep it:
// block (x, y) is keyed by the edges' segment counts and segmentations (each
// inner boundary's arc over the edge's length, and each segment's radius),
// the six distances of the end points (x0, x1, y0, y1) and their four heights
// above the plane, all on a step of 2^-KEY_BITS of the structure's size. Two
// blocks alike in those are alike under a motion that keeps the plane (their
// horizontal distances are their distances less their height differences),
// node for node. A block keys on the smaller of its two forms, (x, y) and
// (y, x), so the map is closed under reversal, the kernel's rule: (y, x)
// copies (t, s) whenever (x, y) copies (s, t); each class's representative
// and its transpose are its sources, never targets. Own blocks (x, x) are
// keyed alike and copied between congruent edges.
//
// Not the direct classes: a block copies only when every pair of it would,
// and the direct key's rounding of a pair's geometry now and then splits
// congruent pairs across a quantum, which costs a pair only compression but a
// block its copy: matched by direct classes elementwise, the 10-element
// Yagi's remainder computed 1.85 M of its 4.68 M pairs, against 0.58 M by
// edge pairs (the hint route's 0.58 M too).
struct RemainderGeometry {
    const double *seg_l = nullptr, *seg_r = nullptr;  // (n, 3)
    const double *a = nullptr;                        // (n)
    const int64_t *edge_seg0 = nullptr;               // (E + 1): the edges tile the mesh in order
    int64_t n_edges = 0;
    double ground_z = 0.0;
};
// The block copies: targets (ta, tb) and their sources (sa, sb), row-major
// in the target.
void remainder_blocks(const RemainderGeometry &g, std::vector<int64_t> &ta, std::vector<int64_t> &tb,
                      std::vector<int64_t> &sa, std::vector<int64_t> &sb);

// The fill report's wire map (momwire#2016, `pair_reuse::wire_map`): the
// direct family's (W, W, 3) count of computed, reused and near pairs, the
// same-edge pairs counted near.
void wire_map(const Plan &p, const int32_t *wire_of_seg, int64_t n_wires, int64_t *out);

pair_reuse::Counters cache_counters();
void clear_cache();

}  // namespace reuse
}  // namespace bspline
}  // namespace mw
