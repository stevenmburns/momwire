// momwire#1517 step 1: bs2's free-space fill, solve and readouts in C++,
// the orchestration over the Python-free kernels (`_bspline_core.h`):
// `BSplineSolver._build_J_blocks` (off-edge moments with the pair-order
// ladder, the phase-guard / radius-run / source-tube splits and the extended
// kernel; same-edge static + regularised blocks), `_assemble_Z`, the chunked
// route `_compute_Z_dense_chunked`, `_solve_with_kcl_ports` and the spline
// readouts.
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

// `_compute_Z_dense_chunked(geom, k, supp_seg, polys)` (issue #136, the
// route `_compute_Z_operator` takes when `_dense_tensor_fits_budget` says
// no): observer-row windows of the off-edge moments accumulated straight
// into Z through the windowed assembler, then each edge's same-edge
// correction `(A_st + A_reg) - J` in row windows of its own. Windows cut at
// `_schedule.mb_rows(swept_mem_mb, row_bytes)`, as Python cuts them, so the
// sums are Python's. (n_basis, n_basis) C order.
std::vector<cdouble> compute_Z_chunked(const Geometry &g, const Basis &b,
                                       const std::vector<double> &radius_per_wire, const FillParams &fp,
                                       double omega, double eps, double mu, double swept_mem_mb);

// `_dense_tensor_fits_budget(n_segs)`: one (d+1, d+1, N, N) complex tensor
// within `swept_mem_mb`.
bool dense_tensor_fits(int degree, int64_t n_segs, int64_t swept_mem_mb);

// `_resolve_exact_kernel_auto` on a free-space route: some segment shorter
// than `EXACT_KERNEL_AUTO_H_OVER_A` (3) of its radius.
bool exact_kernel_auto(const Geometry &g, const std::vector<double> &radius_per_wire);

// Whether `coaxial_lines` lists a line carrying a radius step (momwire#1438's
// cross job, which `add_exact_correction` does not serve).
bool exact_kernel_has_cross_job(const Geometry &g, const std::vector<double> &radius_per_wire);

// `_add_exact_kernel_correction(Z, geom, supp_seg, polys, k)` (momwire#1408,
// #1411, #1421): per one-radius coaxial group, observer-row windows of the
// exact ring-kernel moments (`mw::ek::CoaxialRows`) minus the base moments
// the fill put there (`_exact_kernel_base.BaseRows`, its run / diagonal
// dedup replayed so each gathered value is the same representative pair's),
// assembled into Z (C order) by the windowed assembler. A coaxial line with
// a radius step (momwire#1438's cross job, numpy-only in Python) throws
// FrontNotPorted.
void add_exact_correction(std::vector<cdouble> &Z, const Geometry &g, const Basis &b,
                          const std::vector<double> &radius_per_wire, const FillParams &fp, double omega,
                          double eps, double mu, double swept_mem_mb);

// The image term of an above-ground fill (`_accumulate_Z_image_chunked`
// over `PotentialGround.weight_windows()`): observer windows of the
// real-against-mirrored-source moments into the weighted windowed assembler
// with scale -1, then the near-image analytic fixup of every grazing
// horizontal edge (momwire#631). PEC (`compose` false: w_A the mirrored
// tangent dot, w_Phi 1) or the Sommerfeld composition's exact-image half
// (w_A = C2 times it, w_Phi = C2). The tangent dot is numpy's real gemm.
struct ImageGround {
    double ground_z = 0;
    bool compose = false;
    cdouble c2{1.0, 0.0};
};
void accumulate_image(std::vector<cdouble> &Z, const Geometry &g, const Basis &b,
                      const std::vector<double> &radius_per_wire, const FillParams &fp, double omega,
                      double eps, double mu, const ImageGround &gr, double swept_mem_mb,
                      const nec5::NumpyLinalg &np);

// `_remainder_qp_pairs`' O(N) short-circuit (`_quadrature.
// remainder_qp_pairs` at the graded route's c = 4, base `n_qp_sommerfeld`):
// true when it proves no segment pair is raised above the base order. A
// deck it cannot clear takes the per-pair scan and, where pairs are listed,
// the graded pair correction (not ported: such a deck declines).
bool remainder_pairs_cleared(const Geometry &g, double ground_z, int64_t n_qp_sommerfeld);

// `Z -= _Z_sommerfeld_remainder(geom, supp_seg, polys, eps_t)` on a deck
// `remainder_pairs_cleared` clears: the remainder's source nodes and
// moment weights at `n_qp_sommerfeld`, the eager grid
// (`_below_interface.somm_grid`, sized by `max_image_distance`) and the
// fused kernel (`somm_core::remainder_bspline_Q`, the symmetric route).
void subtract_sommerfeld_remainder(std::vector<cdouble> &Z, const Geometry &g, const Basis &b,
                                   double ground_z, cdouble eps_t, double k, double omega, double mu,
                                   int64_t n_qp_sommerfeld);

// numpy's ILP64 `cblas_dsyrk` (`scipy_cblas_dsyrk64_`): `tangents @
// tangents.T` is numpy's syrk case (one operand and its transpose).
using dsyrk_fn = nec5::cblas_dsyrk_fn;

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
    // (n_b, n_p) in Python's layout: C order from the KCL Schur solve,
    // column-major (`lu_solve`'s) when there are no constraint rows
    bool x_c_order = true;
    std::vector<cdouble> X;
    std::vector<cdouble> Y;  // (n_p, n_p) C order
};
// SciPy's symmetric pair (`zsytrf` / `zsytrs`, LP64, the `cython_lapack`
// signatures): what `scipy.linalg.solve` takes for an exactly symmetric
// complex system (its structure detection), with the optimal workspace.
using zsytrf_fn = void (*)(char *uplo, int *n, cdouble *a, int *lda, int *ipiv, cdouble *work, int *lwork,
                           int *info);
using zsytrs_fn = void (*)(char *uplo, int *n, int *nrhs, cdouble *a, int *lda, int *ipiv, cdouble *b,
                           int *ldb, int *info);
struct SymLapack {
    zsytrf_fn sytrf = nullptr;
    zsytrs_fn sytrs = nullptr;
};

PortSolve solve_ports(const std::vector<cdouble> &Z, int64_t n_b, const std::vector<double> &B,
                      int64_t n_p, const std::vector<double> &kcl, int64_t n_c,
                      const razor::Lapack &lp, const nec5::NumpyLinalg &np,
                      const SymLapack *sym = nullptr);

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

// `_overlap_triplets(geom, supp_seg, polys, polys)` (`_loading_gram`): per
// shared segment, in the order Python's dict meets them, the closed-form
// `C @ H @ C.T` on numpy's real BLAS (`np.dgemm` / `dgemv` / `ddot`), COO
// with duplicates, tagged by wire.
struct Triplets {
    std::vector<int64_t> rows, cols, wire;
    std::vector<double> vals;
};
Triplets loading_gram(const Geometry &g, const Basis &b, const nec5::NumpyLinalg &np);

// `_add_wire_scaled(Z, triplets, z_wire)`: `np.add.at(Z, (rows, cols),
// z_wire[wire] * vals)` into a C-ordered (n_basis, n_basis) Z.
void add_wire_scaled(std::vector<cdouble> &Z, int64_t n_basis, const Triplets &t,
                     const std::vector<cdouble> &z_wire);

// `wire_loss_power(coeffs, omega)[0]` without a jacket: ½ Σ Re Z'_w ·
// Re(conj(c_r) c_c) · val, `np.add.at` per wire, then the pairwise sum.
double wire_loss_power(const Triplets &t, const std::vector<cdouble> &coeffs,
                       const std::vector<cdouble> &z_wire, size_t n_wires);

// `BSplineSolver.pair_order_ladder` for a free-space deck: the default
// ladder's tiers below `n_qp_pair`, normalised.
std::vector<std::pair<double, int64_t>> default_ladder(int64_t n_qp_pair);

}  // namespace bspline
}  // namespace mw
