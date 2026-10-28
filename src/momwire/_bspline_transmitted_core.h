// bs2's two transmitted blocks (momwire#1718 S12): `compute_Z_operator_
// buried`'s mixed branch -- a deck with wires on both sides of the interface
// and no crossing node -- as `_field_galerkin_block` fills it over the
// below->above table (`_somm_transmitted_core`). Its own TU so the
// field-form assembler's inline text and the buried fill's other kernels keep
// their codegen in `_bspline_fill_core.cpp`.
#pragma once

#include "_somm_transmitted_core.h"

#include <complex>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

namespace mw {
namespace bspline {

// One class's field-form nodes: points and tangents (n q, 3), the moment
// weights W (nd, n, q), and each global segment's position on the class
// axis (-1 off it).
struct TransmittedSide {
    const std::vector<double> *nodes, *tans, *W;
    const std::vector<int64_t> *pos;
    int64_t n;
};

// `Z -= ` the upward block (above observers, below sources) and then the
// downward one, chunked over observers as `_field_galerkin_block` chunks
// them. Throws `somm_trans::Refused` with the grid eval's refusal name.
void transmitted_blocks(std::vector<std::complex<double>> &Z, const std::vector<double> &polys,
                        const std::vector<int64_t> &supp_seg, int nd, int64_t n_basis, int64_t q,
                        const somm_trans::Grid &grid, double ground_z, double k_p, std::complex<double> k_m,
                        const TransmittedSide &above, const TransmittedSide &below);

}  // namespace bspline
}  // namespace mw
