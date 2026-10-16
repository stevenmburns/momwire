// razor-2p's whole k-dependent solve in C++: every row window of the fill
// (seg_moments -> assemble_t1 -> t2_rows, real sources and, over a perfect
// ground, the mirrored ones folded in) written straight into Z, then the LU
// factorisation and the solve for the port columns. No Python anywhere, so a
// standalone program can drive it; `_accel_razor_solve.cpp` binds it.
//
// It is the SAME arithmetic as the Python route, not a second copy of it:
// each window calls the three kernels of `_razor_core.h` that
// `RazorSolver._source_block_rows` calls, on the same arguments, and the LU
// is whatever zgetrf/zgetrs the caller hands in (the binding passes SciPy's
// own, from `scipy.linalg.cython_lapack`, so it is the routine `lu_factor` /
// `lu_solve` reach). What the C++ chain removes is the Python and numpy work
// between the calls, and the per-window allocations: buffers live in the
// `SolveContext` and are reused across windows and across solves.
//
// Scope (the binding's caller, `RazorSolver._cxx_port_solution`, routes only
// these): real k; the NEC-5 path rule (n_path = 2) with T1 sharing the
// centroid moments (`t1_on_centroids`); free space or a PERFECT ground (the
// unweighted image fold, grounded rows included); with or without the
// extended kernel's labels; no loading, no chop, no
// row restriction.
#pragma once

#include "_mw_core.h"

#include <chrono>
#include <functional>

namespace mw {
namespace razor {

// Fortran LAPACK, by pointer: the caller decides which library.
typedef void (*zgetrf_fn)(int *m, int *n, cd *a, int *lda, int *ipiv, int *info);
typedef void (*zgetrs_fn)(char *trans, int *n, int *nrhs, cd *a, int *lda,
                          int *ipiv, cd *b, int *ldb, int *info);

struct Lapack {
    zgetrf_fn getrf = nullptr;
    zgetrs_fn getrs = nullptr;
};

// One source set's arguments: the real segments or their mirror images. The
// observers (centroids) and everything per basis are shared.
struct SourceSet {
    view<const double, 2> seg_p0;   // (n_seg, 3)
    view<const double, 2> seg_t;    // (n_seg, 3)
    view<const int64_t, 1> group_j; // EK source labels (n_seg) or empty
    view<const double, 1> a_ek;     // (n_seg) or empty
    view<const double, 2> td_a;     // (3, n_basis): sigma-folded tangents
    view<const double, 2> td_b;
};

struct SolveInputs {
    // The centroid observers (n_seg, 3), with their EK labels and tubes.
    view<const double, 2> cent;
    view<const int64_t, 1> group_i; // (n_seg) or empty
    view<const double, 1> b_obs;    // (n_seg) or empty
    // The source columns shared by both sets.
    view<const double, 1> seg_h, a, xg, wg;
    // Per basis.
    view<const int64_t, 1> s_a, s_b;
    view<const double, 1> h_a, h_b, q_a, q_b;
    view<const bool, 1> fall_a, fall_b;
    view<const double, 2> tans;  // (n_basis * n_path, 3)
    view<const double, 1> wts;   // (n_basis * n_path)
    size_t n_path = 2;
    view<const int64_t, 1> grounded;  // sorted basis rows, or empty
    double k = 0.0, c_a_imag = 0.0, c_phi_imag = 0.0;
    SourceSet real;
    bool has_image = false;
    SourceSet image;
    ptrdiff_t window_rows = 0;  // basis rows per window (>= 1)
    uintptr_t cancel_flag = 0;
};

// What ran, for tests to read (the "green gate that measures nothing" rule:
// count the branch before believing an agreement).
struct Counters {
    uint64_t solves = 0, windows = 0, image_windows = 0;
    uint64_t seg_moments_calls = 0, assemble_t1_calls = 0, t2_rows_calls = 0;
    uint64_t lu_calls = 0, pool_grows = 0;
    double fill_seconds = 0.0, lu_seconds = 0.0;
};

class SolveContext {
  public:
    Counters counters;

    // Z (n_basis, n_basis) is written column-major into the context's own
    // buffer, which survives until the next fill; `z()` reads it.
    void fill(const SolveInputs &in);
    // The same, with `between(Z)` run once on the free block before the
    // image folds in: razor's `Z = free; Z += exact correction; Z -= image`
    // (`_assemble_Z`). Each element sees the same operations in the same
    // order as `fill` when `between` adds nothing.
    void fill(const SolveInputs &in, const std::function<void(cd *)> &between);

    // Factor the context's Z in place and solve Z X = B for B's columns,
    // overwriting `b` (n_basis, n_rhs, column-major, ld = n_basis) with X.
    // Returns 0, or i > 0 when U(i, i) is exactly zero (the caller raises
    // its singular-matrix error; X is then not written).
    int lu_solve(const Lapack &lp, cd *b, int n_rhs);
    // `np.add.at(Z, (rows, cols), vals)`: unbuffered, in order, onto the
    // filled matrix (column-major), before `lu_solve`.
    void add_at(const int64_t *rows, const int64_t *cols, const cd *vals, size_t n);

    // Free Z and the window pool (a big solve's buffers are not held until
    // the next one); the counters are kept.
    void release();

    const cd *z() const { return z_.data(); }
    ptrdiff_t n_basis() const { return n_; }

  private:
    ptrdiff_t n_ = 0;
    std::vector<cd> z_;
    // The window pool: grown, never shrunk, reused across windows and solves.
    std::vector<cd> m0_, m1_, t1_, img_;
    std::vector<double> obs_;
    std::vector<int64_t> need_, gi_, obs_row_, plus_, minus_, mark_;
    std::vector<double> bo_;
    std::vector<int> ipiv_;
    std::vector<cd> m0k_;  // stays empty: no chopped rows on this route

    template <class T>
    void reserve(std::vector<T> &v, size_t n) {
        if (v.size() < n) {
            v.resize(n);
            counters.pool_grows++;
        }
    }
    void window(const SolveInputs &in, const SourceSet &src, ptrdiff_t lo,
                ptrdiff_t hi, bool image);
};

}  // namespace razor
}  // namespace mw
