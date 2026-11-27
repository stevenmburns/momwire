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
// `kept_bytes_per_class` per class the 4/3x cap allows, `pair_reuse::plan_allowance`
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
    // THE IMAGE BLOCK (momwire#2063 step 2a), or null for none: the sources'
    // end points mirrored in the ground plane, (n, 3) each, as the image fill
    // hands them to the kernel, and each segment's own-image mode (`ImageMode`).
    const double *img_l = nullptr, *img_r = nullptr;
    const int8_t *img_mode = nullptr;   // (n), with img_l
};

// THE IMAGE BLOCK. bs2's image moment of observer o against source s is the
// off-edge kernel on o against the MIRRORED segment M(s), on the direct
// fill's ladder and EK labels (`_image_fill_ladder`, `_ek_axis_labels`:
// one label for every segment, real or mirrored, and b_j the source's own
// radius either way). So the pair (o, M(s)) is to the kernel a pair of
// segments like any other, and the direct key's rigid invariants of it are
// its complete ones: block 1 keys (o, M(s)) with the direct key, in the
// direct block's class space (razor2p's n_blocks = 2), and an image pair may
// take a direct pair's value. The value is the plane-blind moments; every
// plane-dependent factor (PEC's -Mt tangent dot and the -1, refl-coef's
// Fresnel dyad, the Sommerfeld image's C2) is the pair's own weight, applied
// at assembly (`image_tile`). A rigid key on WEIGHTED entries would be wrong
// over refl-coef, whose weights read the pair's tilt to the plane.
//
// A segment against its own edge's image is keyed only on a straight edge
// the image fill sweeps like any other (`Keyed`). An offset edge's own image
// block is the offset route's (`OffsetRoute`): the sentinel class, which the
// image tile writes as zero. A grazing edge's (`OwnPairs`) has the near-image
// fixup subtract the sweep's own moments of it (momwire#631), so each pair is
// its own class and its value its own fill's bits.
enum ImageMode : int8_t { Keyed = 0, OffsetRoute = 1, OwnPairs = 2 };
// TEST HOOK (the adversarial gate): 1 keys block 1 on the source's UNMIRRORED
// direction (the image's z-flip dropped from the key only), which no correct
// plan does. Plans are keyed on it.
void set_image_key_probe(int probe);
// Whether `p` has the image block, and the sentinel class (the same-edge
// pairs', pair (0, 0)'s) the image tile writes as zero.
bool has_image(const Plan &p);
int32_t sentinel_class(const Plan &p);
// Image pairs in the sentinel class (the offset edges' own image blocks).
int64_t image_sentinel_pairs(const Plan &p);

// The plan for `g`, from the process's cache or built: one family, "direct",
// n x n, one code bit (reciprocity), over one block, or two with the image. Never null; a deck the census does not serve comes
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

// momwire#2166: the direct family's window as a source for the fused
// assembly (`assemble_Z_bspline_windowed_tiles`, `_bspline_core.h`), so the
// window is never written whole. Observer rows [r0, r1) lie on one edge,
// whose own columns [o0, o1) are `own`, (d1^2, r1 - r0, o1 - o0)
// C-contiguous (the fill's own moments of those pairs), or zero when `own`
// is null (an offset edge, whose own block is added apart). Every other
// source is its class's value, transposed under reciprocity: the window
// `gather` writes with its own-edge columns overwritten, bit for bit. A
// chopped window (rows off one edge) is not one of these; the fills cut
// their windows at the edges.
struct Window {
    const Plan *plan = nullptr;
    const cd *values = nullptr;  // (classes, d1, d1)
    int d1 = 3;
    int64_t r0 = 0, r1 = 0, o0 = 0, o1 = 0;
    const cd *own = nullptr;
};
// Throws unless `w` is one of the above (the tile source cannot: it runs
// inside the assembly's parallel region).
void check_window(const Window &w);
// out[(pq * nr + io) * nc + is] for rows[io] in [r0, r1), cols[is] in [0, n).
void window_tile(const Window &w, const int64_t *rows, size_t nr, const int64_t *cols, size_t nc, cd *out);
// TEST HOOK (the adversarial gate): 1 copies a reciprocal member as its
// class's value untransposed, which no correct fill does, so a test can
// show the fused path is the one that ran and that its gates see it.
void set_window_probe(int probe);
int window_probe();

// The image block's window as a source for the weighted fused assembly
// (`assemble_Z_bspline_weighted_windowed_tiles`): every pair (o, s) is its
// class's block-1 value, transposed under reciprocity, and the sentinel
// class zero. Rows and columns are any segments: no own edge is cut out.
struct ImageWindow {
    const Plan *plan = nullptr;
    const cd *values = nullptr;  // (classes, d1, d1)
    int d1 = 3;
    int32_t zero_class = -1;
};
void check_image_window(const ImageWindow &w);
void image_tile(const ImageWindow &w, const int64_t *rows, size_t nr, const int64_t *cols, size_t nc, cd *out);

// THE SOMMERFELD REMAINDER BY PLANE INVARIANTS (momwire#2063 step 2b): its
// own family and its own plan, of segment pairs. The fused kernel's moments
// of observer segment o against source segment s,
//
//     Jf[p, P] = sum_{qi, rj} W[p, o, qi] t_o . F(x_qi, x'_rj) . t_s W[P, s, rj],
//
// read F through the grid at each node pair's horizontal separation rho and
// height sum hh = (z - z_g) + (z' - z_g), and project it on the two
// tangents in the pair's horizontal frame (`proj_project`: the azimuth of
// each tangent about the separation); W reads only each segment's length h
// (and the rule, one for the whole fill). With m the centre difference, A
// and B the two segments' vectors (r - l) and s, s' the nodes' offsets from
// the centres, a node pair's horizontal separation is m_h + s A_h - s' B_h
// and its height sum S + s A_z + s' B_z, S the centres' height sum. So the
// pair's complete invariants under the motions that keep the plane --
// horizontal translations, rotations about a vertical axis and reflections
// in vertical planes (the dyad is a tensor of the plane's symmetry group,
// and the projection reads the tangents' dots with the separation) -- are
// the horizontal Gram words m_h.m_h, m_h.A_h, m_h.B_h, A_h.B_h, the vertical
// parts A_z, B_z, the height sum S and the two lengths h_o, h_s (|A_h| and
// |B_h| follow from h and A_z). NOT a vertical motion: S moves with it, and
// the grid reads S; nor the plane's own mirror, which the image key uses.
// The value is a smooth function of these words wherever the grid is (the
// Gram words are the O(2) invariants' generators), so a member's error is
// its words' rounding times the value's slope.
//
// THE KEY quantises the Gram words on a step of 2^-KEY_BITS of D^2 and A_z,
// B_z, S on 2^-KEY_BITS of D, D a lower bound on the pair's distance to the
// image, hypot(|m_h| - (|A_h| + |B_h|) / 2, S - (|A_z| + |B_z|) / 2) (each
// part floored at 0) floored to a power of two: the grid's argument r1 =
// hypot(rho, hh) >= D at every node pair, so a member's r1 errs by at most
// ~2^-KEY_BITS of r1. h_o, h_s enter by `length_type`, as in the direct key.
//
// RECIPROCITY (one code bit): the remainder dyad is reciprocal, so the pair
// read from the source's side is the transpose, Jf[P, p](s, o) =
// Jf[p, P](o, s) -- always, the remainder reading no radius. Each pair keys
// on the smaller of its two forms (lengths swapped, m -> -m: m.A <- -m.B,
// m.B <- -m.A, A_z <-> B_z), and the kernel transposes a member whose form
// differs from its representative's: a copy.
//
// OWN CLASSES (`Key::Own`, each pair its own single, filled by the kernel's
// walk as without reuse): pairs nearer than NEAR_LENGTHS max(h_o, h_s),
// centre to centre or D (the same-segment pairs among them), and the pairs
// whose remainder order is raised (momwire#1189's list, `raised`): their
// base-order value is the walk's own, so the later correction, which
// subtracts the pair's own base moments, lands on the bits it always did.
//
// THE CAP counts the unordered pairs, n (n + 1) / 2: the walk already forms
// each node pair's grid read once for both orders, so reciprocity alone is
// no saving here, and a deck with nothing else to share declines.
//
// THE GRID'S REGION BREAKS. The interpolant is continuous across a stencil
// shift inside a region (a 4-point Lagrange rule through the node it
// shifts at) and past the table's edge (the continuation is anchored on the
// edge value), but not in general across the region breaks (r1 at r_break
// and r_near, theta at th_split), where two tables meet at their
// interpolation error: a radial break is a shared node row only where both
// tables' radial steps land on it, and the theta split joins tables of
// different radial steps (measured: a member across the split that took its
// representative's table moved Z 4.7e-10 of its largest entry, 8.8e-17
// under the rule, `test_bs2_remainder_plane_reuse_2063.py`).
// The breaks move with the frequency and the plan does not, so the tie rule
// is the kernel's, per fill: a class whose representative has a node pair
// within `somm_core::remainder_tie_rel()` (1e-9) of a break is left to the
// walk, member by member (`RemainderCounts::tied`). A member's r1 and theta
// sit within ~2^-KEY_BITS of its representative's, far inside 1e-9, so every
// class the kernel serves has each of its members' node pairs in its
// representative's region.
struct RemainderPlanGeometry {
    int64_t n = 0;
    const double *seg_l = nullptr, *seg_r = nullptr;  // (n, 3)
    const double *h = nullptr;                        // (n): the lengths W reads (`h_per_seg`)
    double ground_z = 0.0;
    // The raised pairs, ordered (both orders listed), as flat o * n + s.
    const int64_t *raised = nullptr;
    int64_t n_raised = 0;
    int degree = 2;
};
// The plan for `g`, from the process's cache (keeping the newest two, the
// direct plan beside it) or built: one family, "remainder", n x n, one code
// bit, one block; each class's representative is kept (`rep_flat`). Never
// null; DECLINED with the reason when the census does not serve (the cap,
// the allowance: the direct family's per-class allowance, which its values,
// code and representative fit).
std::shared_ptr<const Plan> remainder_plan_for(const RemainderPlanGeometry &g, bool *built = nullptr);
// TEST HOOK (the adversarial gate): 1 keys without the height sum, 2 on the
// tangents' fixed-frame components (m.m and A_h, B_h as they lie) instead of
// their dots with the separation, 3 on the centres' height DIFFERENCE in
// place of the sum (a vertical shift keys alike). No correct plan does any.
// Plans are keyed on it.
void set_remainder_key_probe(int probe);
int remainder_key_probe();
// The plan's class map as the remainder kernel reads it
// (`somm_core::PairReuse`'s class route).
struct RemainderClasses {
    int64_t n = 0, classes = 0;
    const int32_t *map = nullptr;       // (n * n): class << 1 | code
    const int64_t *rep_flat = nullptr;  // (classes)
    const uint8_t *rep_code = nullptr;  // (classes)
    const uint64_t *single = nullptr;   // (classes + 63) / 64 bits: a pair the key left its own class
};
RemainderClasses remainder_classes(const Plan &p);
// The fill report's wire map of the remainder: (W, W, 3) computed, reused,
// 0, a class's first pair computed, its others reused, and every pair of a
// single or of a class the fill left to the walk (`tied`, (classes) bytes,
// or null) computed.
void remainder_wire_map(const Plan &p, const uint8_t *tied, const int32_t *wire_of_seg, int64_t n_wires,
                        int64_t *out);

// The fill report's wire map (momwire#2016, `pair_reuse::wire_map`): block
// `block`'s (W, W, 3) count of computed, reused and near pairs, the sentinel
// class's pairs counted near. A class is computed at its first pair, block 0
// first, so an image pair of a class a direct pair opened counts reused.
void wire_map(const Plan &p, const int32_t *wire_of_seg, int64_t n_wires, int64_t *out, int block = 0);

pair_reuse::Counters cache_counters();
void clear_cache();

}  // namespace reuse
}  // namespace bspline
}  // namespace mw
