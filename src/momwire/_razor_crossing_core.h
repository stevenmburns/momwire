// razor-2p's crossing blocks in C++ (momwire#1512 B4b/c): the cross blocks
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
#pragma once

#include "_razor_front_core.h"

#include <complex>
#include <vector>

namespace mw {
namespace razor {

// Both cross blocks (per source radius, ascending; below sources first) and
// the node charges into `z` (column-major, f.n_basis^2), in
// `_assemble_Z_crossing`'s order. `tents` are the crossing tents (basis,
// below wing). Throws FrontError (solve phase) where Python raises.
void crossing_blocks(const Front &f, std::vector<cd> &z,
                     const std::vector<std::pair<int64_t, int>> &tents, uintptr_t cancel_flag);

}  // namespace razor
}  // namespace mw
