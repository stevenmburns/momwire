// razor2p's crossing blocks in C++ (momwire#1512 B4b/c): the cross blocks
// between the two media and the crossing node charges, the same bits as
// `RazorSolver._assemble_Z_crossing`'s steps 4-5 (`_crossing_fill.
// cross_complete_block` / `cross_complete_block_reversed` with razor's path
// axes, `_crossing_node_charges`). No Python.
//
// The Python fill spends most of its lines on memory: product tiles, held
// columns, streamed and fused end loops. Each of those is gated bit-identical
// to the reference spelling it replaces, so this port is the REFERENCE:
//
//   main sandwich   one designed evaluation over the whole (above x below)
//                   node grid -- its distinct triples, whole exact-rho
//                   columns, the fill's sheet plan -- then the six left
//                   products (`left_products_gathered`'s sums) and the
//                   five-term combine (`combine_rows`'), times c1;
//   end terms       the row ends' (V then W) and the column ends' (W then V)
//                   rank-1 updates into E_r / E_c (`_EndTerms`), their tables
//                   from `_end_tables`' spans through the same memo, labelled;
//   fold            `Z[rows, cols] -= t`, one subtraction per entry.
//
// Memory (momwire#1541): the same bits without the reference's whole-size
// scratch. The grid's tables are formed a window of nodes at a time (its
// triples claimed in the whole grid's order, evaluated as one call would),
// each block is held only over the (rows x cols) entries the fold reads, and
// the axes' end values are kept sparse.
#pragma once

#include "_razor_front_core.h"

#include <complex>
#include <string>
#include <vector>

namespace mw {
namespace razor {

// Both cross blocks (per source radius, ascending; below sources first) and
// the node charges into `z` (column-major, f.n_basis^2), in
// `_assemble_Z_crossing`'s order. `tents` are the crossing tents (basis,
// below wing). Throws FrontError (solve phase) where Python raises.
void crossing_blocks(const Front &f, std::vector<cd> &z,
                     const std::vector<std::pair<int64_t, int>> &tents, uintptr_t cancel_flag);

// The opt-in feature `crossing_anchors` (`ni_eval::Features::crossing_anchors`):
// a frequency sweep's cross blocks and node charges computed exactly at a few
// anchor wavenumbers spanning the sweep, and every other sweep point's
// written by barycentric interpolation between them (Chebyshev-Lobatto
// nodes in k, the plan's two ends among them). The pair geometry is the
// same at every frequency; only the kernel's values move, and over a narrow
// band they move smoothly. The above and below blocks stay exact per
// frequency.
//
// The run's frequency plan: the wavelengths every preparation of the deck
// solves at (`Prepared::wavelength`), handed over by the seam
// (`RazorSeamEngine::seam_done`) before any fill; empty clears it. A fill at
// a wavenumber outside the plan, on another geometry than the anchors', or
// under a plan that declines (too few frequencies, too wide a band, a
// structure too large electrically for the anchors to bound, or anchors too
// large to hold in their memory budget) takes the exact
// cross blocks: the same bits as with the feature off. The anchors are built
// once, in a fixed order, under a lock, so concurrent fills of one sweep read
// the same bytes whichever arrives first.
namespace anchors {
struct Size;
}
void set_band_plan(const std::vector<double> &wavelengths, const anchors::Size &size);

// What the anchors did since the plan was last set.
struct AnchorCounters {
    long long anchors_built = 0;  // exact anchor fills
    long long interpolated = 0;   // fills written from the anchors
    long long copied = 0;         // fills that sat on an anchor
    long long declined = 0;       // fills that took the exact blocks while on
    std::string plan;             // "" (no plan), "ok", or why it declines
};
AnchorCounters anchor_counters();

// True when `a` and `b` give the same cross blocks at one wavenumber: the
// geometry, media and ground constants `crossing_blocks` reads.
bool same_crossing_geometry(const Front &a, const Front &b);

// The anchors' arithmetic, one copy for this fill and for the Python
// route's (`razor.py`'s `_crossing_anchors_now`, through the extension), so
// the two write the same bits.
namespace anchors {
// What the error rule reads of the deck: `span`, the largest distance from
// an above segment's end to a below segment's (`span_of`), and the lower
// medium (eps_r, sigma) with the solver's c, for how fast k_m turns with k.
struct Size {
    double span = 0.0;
    double eps_r = 0.0, sigma = 0.0, c = 0.0;
};
double span_of(int64_t n_seg, const double *seg_p0, const double *seg_t, const double *seg_h, double ground_z);
struct Plan {
    std::vector<double> k;  // the plan's distinct wavenumbers, ascending
    bool ok = false;
    std::string why;        // "" (no plan), "ok", or why it declines
    int n_a = 5;
    // The error rule: the half band's phase turn over the span, the
    // Bernstein radius of the origin, and the bound at n_a (`bound`).
    double x = 0.0, rho = 0.0, est = 0.0;
    std::vector<double> anchor_k, bary_w;  // ok: the anchors, k_max first
};
// The a-priori bound on |dZ|/|Z| at n anchors (calibrated; see the .cpp),
// the target a plan's bound must meet (a third of the registry's declared
// drift), and the most anchors a plan takes before it declines.
double bound(int n, double x, double rho);
// Whether n_a anchors over `entries` cross entries of an n-basis fill fit
// the anchors' memory budget (their values plus one n x n scratch).
bool fits(int n_a, int64_t entries, int64_t n);
extern const double BOUND_TARGET;
extern const int NODES_MAX;
// The plan over `wavelengths` (k = 2 pi / wavelength, as `Front::build`) for
// a deck of `size`, with MOMWIRE_CROSSING_ANCHOR_NODES / _BAND read now:
// the fewest anchors (at least 5) whose bound meets the target, or a
// decline.
Plan plan_of(const std::vector<double> &wavelengths, const Size &size);
bool in_plan(const Plan &p, double k);
// The Z entries the cross blocks and node charges write, as column-major
// indices (col * n + row), ascending and distinct: each medium's rows x the
// other's, and the crossing tents' columns.
std::vector<int64_t> entries(int64_t n, const std::vector<int64_t> &rows_a, const std::vector<int64_t> &rows_b,
                             const std::vector<int64_t> &tent_cols);
// z[idx[e]] += the anchors' values (`raw`, n_a x m) at k: the anchor's own
// where k is one (returns 0), else their barycentric interpolant (returns 1).
int apply(cd *z, const int64_t *idx, size_t m, const cd *raw, const Plan &p, double k);
}  // namespace anchors

}  // namespace razor
}  // namespace mw
