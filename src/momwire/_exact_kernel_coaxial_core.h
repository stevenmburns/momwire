// The exact ring kernel's COAXIAL pair moments (momwire#1410) and the
// observer-row windows over them (`CoaxialRows`, momwire#1411), Python-free:
// the B-spline basis' exact kernel for the standalone path (momwire#1517
// step 0). The arithmetic is `_exact_kernel_coaxial_inline.h` over
// `_exact_kernel_ring_inline.h`, the same text the extension's
// `ExactKernelCoaxialRows` / `exact_kernel_pair_moments` /
// `exact_kernel_ring` compile, so the two routes compute the same bits.
//
// Errors are C++ exceptions (std::runtime_error), with the extension's
// wording.
#pragma once

#include "_exact_kernel_point_core.h"  // PointRules: the same five rules

#include <complex>
#include <cstdint>
#include <memory>

namespace mw {
namespace ek {

// The highest moment count per axis (degree <= 5).
constexpr int COAXIAL_MAX_ND = 6;

// `exact_kernel_ring`: the exact ring kernel K(z) at n points.
void ring_kernel(const double *z, int64_t n, double a, double k, cd *out);

// `exact_kernel_ring_split`: K = A ln|z| + B at n points, every |z| < 6a.
void ring_kernel_split(const double *z, int64_t n, double a, double k, cd *A, cd *B);

// `exact_kernel_pair_moments`: one coaxial pair's (nd, nd) moments, C order,
// any separation; far = -1 classifies as the rows do, 0 forces the near
// route, 1 the far rule.
void coaxial_pair_moments(double hi, double hj, double c, double si, double sj, double a,
                          double k, int nd, const PointRules &rules, int far, cd *out);

// `ExactKernelCoaxialRows`: one coaxial group's segments (start x0 along the
// axis, direction sgn, length h; radius a), the moments of observer rows
// against every segment, with the pair cache (keys first met first, capped at
// max_cached; quantum q; `symmetric` the transpose / mirror dedup).
class CoaxialRows {
  public:
    CoaxialRows(const double *x0, const double *sgn, const double *h, int64_t n, double a,
                double k, int nd, int64_t max_cached, double q, bool symmetric,
                const PointRules &rules, int n_threads);
    ~CoaxialRows();
    CoaxialRows(CoaxialRows &&) noexcept;
    int64_t n() const;
    int64_t evaluated() const;
    int64_t cached() const;
    // Rows [r0, r1) into out, (nd, nd, r1 - r0, n) C order.
    void rows(int64_t r0, int64_t r1, cd *out);

  private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace ek
}  // namespace mw
