// The Sinusoidal-Galerkin basis' DIRECT-FIELD FILL in C++ (momwire#1718 S7),
// with no Python in it: `SinusoidalGalerkinSolver._assemble_Z` on the route
// the mixed-potential fill does not serve (`_mp_serves` False), for a
// single-medium deck:
// - `_refuse_coincident_segments` (two segments on one path refuse, in the
//   solver's words);
// - `_test_context` (the uniform test rule's observers and `w_entry`);
// - the free block and the ground's image block (`_tested_contribs`): the
//   fused far fill of the S1 core (`sinusoidal_galerkin_far_fill[_ek]`, with
//   `_ek_far_labels`' coaxial labels under the extended kernel), and the near
//   pairs (`_near_pairs`: the cdist prefilter, then the sampled gap test)
//   re-integrated on the endpoint-graded rule (`_apply_near_correction`):
//   `_field_components_bcast` on the folded cos-1 shape (`_folded_cos_fields`,
//   and `_folded_ek_delta_fields` on the eligible pairs), the plain
//   projection and the test reduction, numpy's arithmetic element by
//   element;
// - the PEC image folded (`fg.mode == "fold"`) or the Sommerfeld image
//   composed (`"compose"`: `c2 * img`, less the tested remainder
//   `_tested_sommerfeld_remainder`, then subtracted);
// - the scatter into G: `_scatter_coef_product`'s dense zgemm below
//   `_DENSE_ASSEMBLY_THRESHOLD`, else `_assemble_G_banded`'s observer bands
//   (`_segment_bands` over `_band_budget_bytes`, `_scatter_band`'s ordered
//   row scatter and its csc products);
// - `_ek_bracket_correction_tested` (momwire#299's end bracket on the nodes
//   `_ek_reduced_ends` marks, symmetrized) and
//   `_contact_charge_correction_tested` (momwire#282 / #292 on a finite
//   ground).
// The caller applies the wire loading (`_apply_loading`), which is the same
// on both routes.
//
// Python stays the reference: G is Python's to the bit, gated by
// `tests/test_sinusoidal_galerkin_fill_core.py` through
// `scripts/core_split/sinusoidal_galerkin_fill_compare.py`.
//
// DECLINED (`nec4::Declined`), never misread: a reflection-coefficient
// ground, a complex k (buried), and anything else the route reaches that
// this copy has no replay of.
#pragma once

#include "_sinusoidal_galerkin_fill_core.h"

#include <complex>
#include <vector>

namespace mw {
namespace sinusoidal_galerkin {

// Whether `_assemble_Z` takes the direct-field fill for this deck on the
// NEC-4.2 slot: a wire end on the ground plane (`_mp_serves` False). The
// junction-port solve (S9) reaches the same fill through
// `assemble_Z_direct(fr, args, view, n_basis, np)`.
bool direct_route(const sinusoidal::Front &fr, const nec4::SolverArgs &args);

// `_assemble_Z(geom, k)` on the direct-field route, before `_apply_loading`:
// G in F order, (n, n) with n = N.
std::vector<std::complex<double>> assemble_Z_direct(const sinusoidal::Front &fr, const nec4::SolverArgs &args,
                                                    const nec5::NumpyLinalg *np);

// The same over a basis view of `n_basis` columns (N plus any port columns
// `_basis_coefs` appends): the view's CSR entries name segments and bases.
std::vector<std::complex<double>> assemble_Z_direct(const sinusoidal::Front &fr, const nec4::SolverArgs &args,
                                                    const sinusoidal::SegView &view, long long n_basis,
                                                    const nec5::NumpyLinalg *np);

}  // namespace sinusoidal_galerkin
}  // namespace mw
