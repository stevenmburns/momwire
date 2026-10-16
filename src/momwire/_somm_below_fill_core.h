// The below/below table filled in C++ (momwire#1512 B5): what
// `_sommerfeld_below.SommerfeldGridBelow`, `SommerfeldGridBelowBlended` and
// `get_grid_below` do, with no Python, value for value.
//
//   iv_surfaces    `iv_surfaces_direct_below`: the R1 = 0 limits
//                  (`_limits_r1_zero_below`) and, elsewhere, the six
//                  integrals (`_somm_below_six_inline.h`, the same code the
//                  extension's `below_six_integrals_batch` runs, under the
//                  weighted-averages tail of momwire#1441) over the divided-
//                  out carrier (`divide_out_below`);
//   make_grid      the constructor: the fifteen-region layout, the eager
//                  regions, a seed's rows taken (momwire#1471);
//   fill_region    `_fill_region`: a deferred region, the floor band's
//                  shared columns read from the low band;
//   fill_source    `get_grid_below` as a `Source`: an in-process memo of
//                  exact and blended grids, the anchors of momwire#1471 (the
//                  plan, the sweep test over the seen frequencies, the anchor
//                  grids grown by seed, the blend), and -- in the executable
//                  only -- its own persistent store of exact grids, anchors
//                  and the seen list, beside the masters' (`somm_core`).
//
// Python reads CPython's scalar complex arithmetic and numpy's array
// arithmetic (fused products, `CDOUBLE_divide`, libm's cexp); this file
// spells each operation as the Python line it copies does.
#pragma once

#include "_somm_below_grid_core.h"

#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>

namespace somm_below {

// `_refuse_if_capped`'s ValueError: a node whose tail hit the panel budget.
struct TailCapped : std::runtime_error {
    using std::runtime_error::runtime_error;
};

// `iv_surfaces_direct_below(eps_t, k2, R1, theta, omega=omega, mu=mu)` at
// the grid fill's rtol (1e-9, the contour's 1e-11) over n points: `out` is
// (4, n), the surfaces in `_SURF_KEYS` order.
void iv_surfaces(cd eps_t, double k2, double omega, double mu, const double *r1,
                 const double *theta, size_t n, cd *out, uintptr_t cancel_flag);

// `SommerfeldGridBelow(eps_t, k2, r1_max, omega=, mu=, seed=, spacing=)`;
// `eager` false is `SommerfeldGridBelowBlended`'s lattice (every region
// deferred).
std::unique_ptr<Grid> make_grid(cd eps_t, double k2, double r1_max, double omega, double mu,
                                const double spacing[3], bool eager, const Grid *seed,
                                uintptr_t cancel_flag);

// `SommerfeldGridBelow._fill_region(idx)`, exact.
void fill_region(Grid &g, int idx, uintptr_t cancel_flag);

// `get_grid_below` / `_ensure_for` with the anchors, as the process's Source.
// `anchors` is `anchor_mode()` ("auto", "off", "on"; MOMWIRE_BELOW_ANCHORS
// when empty).
std::shared_ptr<Source> fill_source(const std::string &anchors = "");

// The executable's store (off until configured; the extension and the
// drivers never configure it): exact grids and anchors in the masters'
// directory (`somm_core::configure_disk_cache`), and the seen list.
void configure_below_store(bool on);

}  // namespace somm_below
