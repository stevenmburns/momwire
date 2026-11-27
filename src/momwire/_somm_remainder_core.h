// The B-spline Sommerfeld remainder block with no Python (momwire#1517 step
// 3): `sommerfeld_remainder_bspline_Q`'s kernel -- the banded rectangular
// route and the symmetric route of the dense obs == src block -- moved
// verbatim from `_accel_somm.cpp`, which keeps only the binding (argument
// checks, the grid view, the route choice). See the block comments in
// `_somm_remainder_core.cpp`.
#pragma once

#include "_accel_somm_proj_inline.h"

#include <complex>
#include <cstddef>
#include <cstdint>

namespace somm_core {

// Pair reuse on the symmetric route (momwire#1809; by class since
// momwire#2063 step 2b, which retired the edge-block copy map):
// `cls_map[si * ns + sj]` is segment pair (si, sj)'s class << 1 | code, in
// the plan's coding (`mw::bspline::reuse::remainder_plan_for`, whose block
// comment is the key). Each class but a single (`single`'s bit set: a pair
// the key left its own) is evaluated once, at its representative
// `rep_flat[c]` (si * ns + sj), by the walk's own per-pair calls, and every
// member's block is that value, transposed where its code differs from
// `rep_code[c]` (reciprocity). A class whose representative has a node pair
// within `remainder_tie_rel()` of one of the grid's region breaks is left to
// the walk member by member (`RemainderCounts::tied`; `tied_out`, when
// given, (n_classes) bytes, says which). The kernel checks the map's shape
// (classes in range, each representative in its own class with its own
// code) and throws std::invalid_argument otherwise. The values are held for
// the call, (n_classes, d1, d1): the plan's allowance counts them. `copied`
// reports the MEMBER segment pairs served (a representative's own read not
// counted).
struct PairReuse {
    const int32_t *cls_map = nullptr;    // (ns * ns)
    std::ptrdiff_t n_classes = 0;
    const int64_t *rep_flat = nullptr;   // (n_classes)
    const uint8_t *rep_code = nullptr;   // (n_classes)
    const uint64_t *single = nullptr;    // ((n_classes + 63) / 64), or null: no singles
    uint8_t *tied_out = nullptr;         // (n_classes), or null
    long long copied = 0;                // out
};

// The class route's tie rule (`PairReuse`): a representative's node pair
// within this of a region break, relative (r1 against r_break and r_near,
// theta against th_split), leaves its class to the walk. 1e-9: a member's
// r1 and theta lie within ~2^-40 of its representative's (the key's step),
// so no member's node pair is in another region than its representative's.
constexpr double REMAINDER_TIE_REL = 1e-9;
// TEST HOOK: the tie rule's width in place of REMAINDER_TIE_REL (< 0
// restores it; 0 turns the rule off). Bound for Python's tests only.
void set_remainder_tie_rel(double rel);
double remainder_tie_rel();

// What the symmetric route did, for the fill report (momwire#2068), counted
// in ORDERED segment-pair entries: `walked` the entries the tile walk formed
// (a segment in two tiles' support halos is walked in both, so this can pass
// ns^2), `copied` those of them served from a class value (a representative's
// own read included, momwire#2073's rule: the walk never forms it again),
// and `cached` the classes evaluated up front (0 when no map was used). The
// kernel evaluated `walked - copied + cached` entries. `tied` the classes the
// tie rule left to the walk and `tied_pairs` their pairs. Tallied per tile
// pair, never inside the per-pair loop.
struct RemainderCounts {
    long long walked = 0;
    long long copied = 0;
    long long cached = 0;
    long long tied = 0;
    long long tied_pairs = 0;
};

// Q (nI, nJ) C order: obs_nodes (nsI, q, 3), obs_tang (nsI, 3), W_obs
// (d1, nsI, q); src likewise; loc_I (nI, d1) into the obs segments with
// pI (nI, d1, d1); loc_J / pJ into the src segments. `symmetric`: the dense
// block's call (obs == src throughout), which takes the symmetric route.
// `max_jf_bytes` 0: the 64 MiB default slab. Cancellation polls
// `cancel_flag` (0 = none) and throws AbortedError after the loops drain.
// `reuse` (symmetric route only; null: none) serves each class's members from
// its value; with no map the kernel is the one it always was, bit for bit.
void remainder_bspline_Q(const somm_proj::GridView &G, double ground_z, double k,
                         const double *obs_nodes, const double *obs_tang, const double *W_obs,
                         std::ptrdiff_t nsI, const double *src_nodes, const double *src_tang,
                         const double *W_src, std::ptrdiff_t nsJ, std::ptrdiff_t q, int d1,
                         const int64_t *loc_I, const double *pI, std::ptrdiff_t nI,
                         const int64_t *loc_J, const double *pJ, std::ptrdiff_t nJ, bool symmetric,
                         size_t max_jf_bytes, std::complex<double> *Q, uintptr_t cancel_flag,
                         PairReuse *reuse = nullptr, RemainderCounts *counts = nullptr);

// Where the symmetric route puts each finished Q[m, n] (momwire#1798), so the
// dense block need not be held beside the matrix it lands in. `fold` 0
// stores Q[m, n] at T[m * rs + n * cs]; -1 / +1 folds it into the entry
// already there (T - Q / T + Q, componentwise, as numpy's `Z -= Q` and
// `M + Q` do), so no n_basis^2 Q is ever live. An entry listed in the
// row-CSR exemption (`ex_ptr` (nb + 1), `ex_col` sorted within a row) is
// stored at side[its CSR position] instead and left out of T: those are the
// entries a later correction adds to (the raised pairs), and Q's own value
// has to take its additions before it is folded, as the dense route adds
// them to Q before `Z -= Q`.
struct RemainderSink {
    std::complex<double> *T = nullptr;
    std::ptrdiff_t rs = 0, cs = 1;
    int fold = 0;
    const int64_t *ex_ptr = nullptr;
    const int64_t *ex_col = nullptr;
    std::complex<double> *side = nullptr;
};

// The dense obs == src block (the symmetric route's call shape: one segment
// set, `loc` = supp_seg) delivered through `sink`. The same entries, formed
// in the same order, as `remainder_bspline_Q(..., symmetric = true, ...)`,
// `reuse` included (momwire#1809; null: none).
void remainder_bspline_Q_sink(const somm_proj::GridView &G, double ground_z, double k,
                              const double *nodes, const double *tang, const double *W,
                              std::ptrdiff_t ns, std::ptrdiff_t q, int d1, const int64_t *loc,
                              const double *polys, std::ptrdiff_t nb, size_t max_jf_bytes,
                              const RemainderSink &sink, uintptr_t cancel_flag,
                              PairReuse *reuse = nullptr, RemainderCounts *counts = nullptr);

}  // namespace somm_core
