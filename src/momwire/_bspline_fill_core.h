// PROTOTYPE (momwire#1517 step 1, not merged): bs2's free-space fill in C++,
// the orchestration over the Python-free kernels (`_bspline_core.h`):
// `BSplineSolver._build_J_blocks` (off-edge moments with the pair-order
// ladder, the phase-guard / radius-run / source-tube splits and the extended
// kernel; same-edge static + regularised blocks) and `_assemble_Z`.
#pragma once

#include "_bspline_front_core.h"
#include "_nec5_printout_core.h"
#include "_razor_solve_core.h"

#include <complex>
#include <cstdint>
#include <utility>
#include <vector>

namespace mw {
namespace bspline {

using cdouble = std::complex<double>;

struct FillParams {
    double k = 0;             // the solver's real wavenumber
    int degree = 2;
    int64_t n_qp_pair = 8;    // `DEFAULT_N_QP_PAIR` (free space)
    std::vector<std::pair<double, int64_t>> ladder;  // `pair_order_ladder`
    int64_t n_qp_same_edge = 4;
    bool extended_kernel = true;
};

// `_build_J_blocks(geom, k)`: (d+1, d+1, N, N) C order. `radius_per_wire`:
// the conductor radius of each wire.
std::vector<cdouble> build_J_blocks(const Geometry &g, const std::vector<double> &radius_per_wire,
                                    const FillParams &fp);

// numpy's ILP64 `cblas_dsyrk` (`scipy_cblas_dsyrk64_`): `tangents @
// tangents.T` is numpy's syrk case (one operand and its transpose).
using dsyrk_fn = void (*)(int order, int uplo, int trans, int64_t n, int64_t k, double alpha,
                          const double *a, int64_t lda, double beta, double *c, int64_t ldc);

// `_assemble_Z(J, supp_seg, polys, geom)` in free space: the C++ assembler
// with `td_all = tangents @ tangents.T` as numpy forms it (dsyrk on the
// upper triangle, then the triangle copied down). (n_basis, n_basis) C order.
std::vector<cdouble> assemble_Z(const std::vector<cdouble> &J, const Basis &b, const Geometry &g,
                                double omega, double eps, double mu, dsyrk_fn syrk);

// `compute_port_solution`'s tail on a filled Z (C order, n_b x n_b):
// `_port_columns`' B (row-major n_b x n_p, real) and constraint rows
// (row-major n_c x n_b), `_solve_with_kcl_ports` (one `lu_factor` /
// `lu_solve` on [V | A^T], `scipy.linalg.solve(A X, A W)`, `W - X Lam`) and
// `Y = B^T X`. LU on SciPy's LAPACK (`lp`), products on numpy's (`np`).
struct PortSolve {
    int info = 0;            // > 0: an exactly singular U(i, i) (LinAlgError)
    std::vector<cdouble> X;  // (n_b, n_p) C order
    std::vector<cdouble> Y;  // (n_p, n_p) C order
};
PortSolve solve_ports(const std::vector<cdouble> &Z, int64_t n_b, const std::vector<double> &B,
                      int64_t n_p, const std::vector<double> &kcl, int64_t n_c,
                      const razor::Lapack &lp, const nec5::NumpyLinalg &np);

// One SOLVED wire's current (or slope) at arcs `s` (clipped to its knots):
// `_wire_span_polynomials` + `_eval_span_polynomials`.
std::vector<cdouble> spline_eval_wire(const Basis &b, const Geometry &g, const std::vector<cdouble> &coeffs,
                                      size_t w, std::vector<double> s, bool derivative);

// `currents_at_knots(coeffs, s_array)` (`derivative` false) and
// `current_slopes(coeffs, s_array)` (true), per INPUT wire
// (`reads_input_wires("per_wire")`: a dropped copy reads zeros, sized like
// its own `s_array` entry, else like its twin's). `s_array` null: the mesh
// knots. Built on `_wire_span_polynomials` / `_eval_span_polynomials`.
std::vector<std::vector<cdouble>> spline_readout(const Basis &b, const Geometry &g,
                                                 const WireDedup &dd,
                                                 const std::vector<cdouble> &coeffs,
                                                 const std::vector<std::vector<double>> *s_array,
                                                 bool derivative);

// `BSplineSolver.pair_order_ladder` for a free-space deck: the default
// ladder's tiers below `n_qp_pair`, normalised.
std::vector<std::pair<double, int64_t>> default_ladder(int64_t n_qp_pair);

}  // namespace bspline
}  // namespace mw
