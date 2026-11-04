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

// Repeated edge pairs the symmetric route may copy instead of fill
// (momwire#1809). Edges are runs of consecutive segments: edge e is segments
// [edge_seg0[e], edge_seg0[e] + edge_nseg[e]). `src[ta * n_edges + tb]` is the
// ordered edge pair (sa * n_edges + sb) whose moment block the pair (ta, tb)
// copies, segment k of ta standing for segment k of sa, or -1 to fill it.
// The caller vouches that each copy is exact up to rounding: an isometry
// keeping every height carries sa onto ta and sb onto tb (the remainder reads
// only rho and the height sum). The kernel checks only the shape of the map
// -- matching segment counts, every source filled rather than copied, and
// (tb, ta) mapped to (sb, sa) -- and throws std::invalid_argument otherwise.
// The source blocks are filled once into a cache of at most `max_jf_bytes`
// (0: the 64 MiB default); a map whose cache would not fit is not used.
// `copied` reports the segment pairs served from the cache.
struct PairReuse {
    std::ptrdiff_t n_edges = 0;
    const int64_t *edge_seg0 = nullptr;  // (n_edges)
    const int32_t *edge_nseg = nullptr;  // (n_edges)
    const int32_t *seg_edge = nullptr;   // (nsI): the edge of each segment, -1 none
    const int32_t *src = nullptr;        // (n_edges * n_edges)
    long long copied = 0;                // out
};

// Q (nI, nJ) C order: obs_nodes (nsI, q, 3), obs_tang (nsI, 3), W_obs
// (d1, nsI, q); src likewise; loc_I (nI, d1) into the obs segments with
// pI (nI, d1, d1); loc_J / pJ into the src segments. `symmetric`: the dense
// block's call (obs == src throughout), which takes the symmetric route.
// `max_jf_bytes` 0: the 64 MiB default slab. Cancellation polls
// `cancel_flag` (0 = none) and throws AbortedError after the loops drain.
// `reuse` (symmetric route only; null: none) copies repeated edge-pair
// blocks; with no map the kernel is the one it always was, bit for bit.
void remainder_bspline_Q(const somm_proj::GridView &G, double ground_z, double k,
                         const double *obs_nodes, const double *obs_tang, const double *W_obs,
                         std::ptrdiff_t nsI, const double *src_nodes, const double *src_tang,
                         const double *W_src, std::ptrdiff_t nsJ, std::ptrdiff_t q, int d1,
                         const int64_t *loc_I, const double *pI, std::ptrdiff_t nI,
                         const int64_t *loc_J, const double *pJ, std::ptrdiff_t nJ, bool symmetric,
                         size_t max_jf_bytes, std::complex<double> *Q, uintptr_t cancel_flag,
                         PairReuse *reuse = nullptr);

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
                              PairReuse *reuse = nullptr);

}  // namespace somm_core
