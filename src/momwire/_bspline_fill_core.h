// momwire#1517 step 1: bs2's free-space fill, solve and readouts in C++,
// the orchestration over the Python-free kernels (`_bspline_core.h`):
// `BSplineSolver._build_J_blocks` (off-edge moments with the pair-order
// ladder, the phase-guard / radius-run / source-tube splits and the extended
// kernel; same-edge static + regularised blocks), `_assemble_Z`, the chunked
// route `_compute_Z_dense_chunked`, `_solve_with_kcl_ports` and the spline
// readouts.
#pragma once

#include "_bspline_front_core.h"
#include "_mw_nd.h"
#include "_nec5_printout_core.h"
#include "_razor_solve_core.h"
#include "_somm_grid_core.h"

#include <algorithm>
#include <complex>
#include <cstdint>
#include <memory>
#include <utility>
#include <vector>

namespace mw {
namespace bspline {

using cdouble = std::complex<double>;

// complex128 storage, C order, for the moment tensors: allocated through
// the nd host (large ones on huge pages, `_mw_nd.h`) or a kernel's own
// output adopted as it is, so a tensor the size of J is never copied.
// Copies share the storage.
class CArray {
  public:
    CArray() = default;
    explicit CArray(std::size_t n, bool zero = true) {  // zero-filled unless asked not to be
        nd::array_t<std::complex<double>> a({static_cast<nd::ssize_t>(n)});
        adopt_(a);
        if (zero) std::fill(p_, p_ + n_, cdouble(0.0, 0.0));
    }
    CArray(const cdouble *b, const cdouble *e) : CArray(static_cast<std::size_t>(e - b)) { std::copy(b, e, p_); }
    static CArray adopt(const nd::array_t<std::complex<double>> &a) {
        CArray c;
        c.adopt_(a);
        return c;
    }
    cdouble *data() { return p_; }
    const cdouble *data() const { return p_; }
    std::size_t size() const { return n_; }
    cdouble &operator[](std::size_t i) { return p_[i]; }
    const cdouble &operator[](std::size_t i) const { return p_[i]; }
    cdouble *begin() { return p_; }
    cdouble *end() { return p_ + n_; }
    const cdouble *begin() const { return p_; }
    const cdouble *end() const { return p_ + n_; }

  private:
    void adopt_(const nd::array_t<std::complex<double>> &a) {
        owner_ = a.buffer().owner;
        p_ = static_cast<cdouble *>(a.buffer().ptr);
        n_ = static_cast<std::size_t>(a.size());
    }
    std::shared_ptr<void> owner_;
    cdouble *p_ = nullptr;
    std::size_t n_ = 0;
};

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
CArray build_J_blocks(const Geometry &g, const std::vector<double> &radius_per_wire,
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

// `_resolve_exact_kernel_auto`: some segment shorter than
// `EXACT_KERNEL_AUTO_H_OVER_A` (3) of its radius -- on a deck with buried
// wires (`below_wire` set, one flag per wire) an above-ground one only
// (momwire#1619 stage 1), and none on a wholly buried deck.
bool exact_kernel_auto(const Geometry &g, const std::vector<double> &radius_per_wire,
                       const std::vector<uint8_t> *below_wire = nullptr);

// `_add_exact_kernel_correction(Z, geom, supp_seg, polys, k)` (momwire#1408,
// #1411, #1421): per one-radius coaxial group, observer-row windows of the
// exact ring-kernel moments (`mw::ek::CoaxialRows`) minus the base moments
// the fill put there (`_exact_kernel_base.BaseRows`, its run / diagonal
// dedup replayed so each gathered value is the same representative pair's),
// assembled into Z (C order) by the windowed assembler; then per coaxial
// line with a radius step its cross job (momwire#1438, ported for
// momwire#1615): the two-radius moments (`mw::ek::CoaxialCrossRows`, the
// text Python's `coaxial_cross_rows` runs) minus `BaseRows`' cross mode.
//
// `below_wire` (momwire#1619 stage 1, `above_only=True`): a deck with buried
// wires, Z its whole buried fill. Every job is cut to its above-ground
// segments (`_exact_kernel_jobs(above_only=True)`), a cross job left with
// one radius is dropped, and the base is the above/above block's
// (`_ek_spec_above`: buried segments never extend).
void add_exact_correction(std::vector<cdouble> &Z, const Geometry &g, const Basis &b,
                          const std::vector<double> &radius_per_wire, const FillParams &fp, double omega,
                          double eps, double mu, double swept_mem_mb,
                          const std::vector<uint8_t> *below_wire = nullptr);

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
// true when it proves no segment pair is raised above the base order.
bool remainder_pairs_cleared(const Geometry &g, double ground_z, int64_t n_qp_sommerfeld);

// `Z -= _Z_sommerfeld_remainder(geom, supp_seg, polys, eps_t)`: the
// remainder's source nodes and moment weights at `n_qp_sommerfeld`, the
// eager grid (`_below_interface.somm_grid`, sized by `max_image_distance`),
// the fused kernel (`somm_core::remainder_bspline_Q`, the symmetric route),
// then the raised pairs' graded correction (`_remainder_pair_correction`,
// momwire#1189 / #1201).
void subtract_sommerfeld_remainder(std::vector<cdouble> &Z, const Geometry &g, const Basis &b,
                                   double ground_z, cdouble eps_t, double k, double omega, double mu,
                                   int64_t n_qp_sommerfeld, const nec5::NumpyLinalg &np);

// TEST HOOK: `_remainder_graded.pair_moments(..., I, J, ...)` on `grid`
// (n, d1, d1) and the base-order `_remainder_pair_moments` of each (I[n],
// J[n]) alone.
void debug_pair_moments(const Geometry &g, double gz, double k, const somm_core::Grid &grid, int d1,
                        const std::vector<int64_t> &I, const std::vector<int64_t> &J, int64_t base,
                        std::vector<cdouble> &graded, std::vector<cdouble> &fixed);

void debug_chunk_pair(const Geometry &g, double gz, double k, const somm_core::Grid &grid, int d1, int64_t i,
                      int64_t j, std::vector<int64_t> &o_pair, std::vector<double> &s_o, std::vector<double> &w_o,
                      std::vector<cdouble> &v);

// `_remainder_qp_pairs(seg_l, seg_r, gz)` on the graded route: (I, J, Q)
// of the segment pairs raised above `base`, sorted by I then J.
void remainder_raised_pairs(const Geometry &g, double gz, int64_t base, const nec5::NumpyLinalg &np,
                            std::vector<int64_t> &I, std::vector<int64_t> &J, std::vector<int64_t> &Q);

// numpy's ILP64 `cblas_dsyrk` (`scipy_cblas_dsyrk64_`): `tangents @
// tangents.T` is numpy's syrk case (one operand and its transpose).
using dsyrk_fn = nec5::cblas_dsyrk_fn;

// `_assemble_Z(J, supp_seg, polys, geom)` in free space: the C++ assembler
// with `td_all = tangents @ tangents.T` as numpy forms it (dsyrk on the
// upper triangle, then the triangle copied down). (n_basis, n_basis) C order.
std::vector<cdouble> assemble_Z(const CArray &J, const Basis &b, const Geometry &g,
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

// The lower medium (`_crossing_fill.buried_medium` at the solver's omega):
// eps~, eps0 eps~, k_p, k_m on Im <= 0, C2 and A_m, CPython's arithmetic.
struct BuriedMedium {
    cdouble eps_t, eps_m, k_m, c2, a_m;
    double k_p = 0;
};
BuriedMedium buried_medium(double eps_r, double sigma, double omega, double eps, double k);

// `_accumulate_Z_subset_chunked(Z, geom, k, seg_idx, ...)` (momwire#915):
// every (observer chunk x source run) off-edge window of the sorted segment
// set `seg_idx` against itself or its mirror image, through the complex-eps~
// windowed assemblers -- direct (`weighted` false: the tangent dot formed by
// the assembler) or weighted (w_A = weight t_m . t_image_n, w_Phi = weight)
// -- scaled by `scale`; a direct block's same-edge fixup after. A complex k
// (the lower medium's); no extended kernel. Z (n_basis, n_basis) C order.
void accumulate_subset_chunked(std::vector<cdouble> &Z, const Geometry &g, const Basis &b,
                               const std::vector<double> &radius_per_wire, const FillParams &fp, cdouble k,
                               const std::vector<int64_t> &seg_idx, bool mirror, cdouble eps, double scale,
                               bool weighted, cdouble weight, double omega, double mu, double ground_z,
                               int64_t swept_mem_mb, const nec5::NumpyLinalg &np);

// `compute_Z_operator_buried` (before the loading) on a deck whose every wire
// is below the plane (momwire#1517 step 4a): the plan's below/below extents
// (a capped walk or a grazing refusal throws FrontNotPorted), the direct and
// A_m-weighted image blocks at k_m in the medium, then the below/below
// remainder over `get_grid_below` (the installed `somm_below::source()`),
// field form, each unordered pair projected once. `fp` carries the buried
// pair order (32) and ladder. (n_basis, n_basis) C order.
std::vector<cdouble> fill_buried(const Geometry &g, const Basis &b, const std::vector<double> &radius_per_wire,
                                 const FillParams &fp, double omega, double eps, double mu, double ground_z,
                                 double eps_r, double sigma, int64_t n_qp_sommerfeld, int64_t swept_mem_mb,
                                 const nec5::NumpyLinalg &np, bool stop_before_cross = false,
                                 struct CrossingStages *stages = nullptr);

}  // namespace bspline
}  // namespace mw
