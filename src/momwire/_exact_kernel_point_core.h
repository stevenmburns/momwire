// The exact ring kernel's point moments and the point-tested assembly
// (momwire#1433), Python-free, for the standalone path (momwire#1479). The
// arithmetic is `_exact_kernel_point_inline.h`, the same text the
// extension's `ExactKernelPointRows` / `exact_kernel_point_assemble`
// compile, so the two routes compute the same bits.
#pragma once

#include <complex>
#include <cstdint>
#include <memory>
#include <vector>

namespace mw {
namespace ek {

using cd = std::complex<double>;

// `_exact_kernel._point_rules()`: the Gauss-Legendre rule on [0, 1] for
// smooth pieces, and the split rule's nodes, weights and log weights.
struct PointRules {
    std::vector<double> xg, wg, xs, ws, wls;
};

// `_exact_kernel.PointRowsAccel`: one coaxial group's segments (start x0
// along the axis, direction sgn, length h; radius a), the moments
// N_q, q < nq, at observer positions along the axis, with the same pair
// cache (keys first met first, capped at max_cached), quantum q.
class PointRows {
  public:
    PointRows(const double *x0, const double *sgn, const double *h, int64_t n, double a,
              double k, int nq, int64_t max_cached, double q, const PointRules &rules,
              int n_threads);
    ~PointRows();
    PointRows(PointRows &&) noexcept;
    int64_t n() const;
    int64_t evaluated() const;
    // (nq, m, n) C order into out.
    void rows(const double *x_obs, int64_t m, cd *out);

  private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

// `_exact_kernel.PointCrossRowsAccel` (momwire#1438): one coaxial LINE's
// segments with their radii a and radius labels lab; the two-radius moments
// at observers of radii a_obs and labels lab_obs, zero where the labels
// agree, keyed on the quantised (c, h, a_obs, a_src) with quanta q, qa.
class PointCrossRows {
  public:
    PointCrossRows(const double *x0, const double *sgn, const double *h, const double *a,
                   const int64_t *lab, int64_t n, double k, int nq, int64_t max_cached, double q,
                   double qa, const PointRules &rules, int n_threads);
    ~PointCrossRows();
    PointCrossRows(PointCrossRows &&) noexcept;
    int64_t n() const;
    int64_t evaluated() const;
    // (nq, m, n) C order into out, every entry written.
    void rows(const double *x_obs, const double *a_obs, const int64_t *lab_obs, int64_t m,
              cd *out);

  private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

// `exact_kernel_point_assemble` on a column-major Z (leading dimension ld):
// dM0 / dM1 are the window's (r, n) C-order moment differences.
void point_assemble(const cd *dM0, const cd *dM1, int64_t n, const int64_t *row_m,
                    const int64_t *row_o, const cd *row_cA, const cd *row_cP, int64_t R,
                    const int64_t *col_ptr, const int64_t *col_n, int64_t NB,
                    const int64_t *ent_j, const double *ent_t, const uint8_t *ent_rise,
                    const double *ent_q, const double *h, cd *Z, int64_t ld, int n_threads);

}  // namespace ek
}  // namespace mw
