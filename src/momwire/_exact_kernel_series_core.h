// The exact ring kernel's point moments as a k-series on geometry-only
// coefficients (momwire#1454, the opt-in feature `exact_kernel_series`),
// Python-free, for the executables' C++ route. The arithmetic is
// `_exact_kernel_series_inline.h`, the same text the extension's
// `ExactKernelPointSeries` compiles, so the two routes compute the same bits.
//
// Reach 1 (`momwire._features`): the C++ route carries it, but no executable
// row asks for it (`EXE_FEATURES`), so the executables still refuse
// MOMWIRE_X_EXACT_KERNEL_SERIES by name. Python reaches this route through
// the front route (`RazorSolver._cxx_front_port_solution`) and the deck seam's
// test hook (`nec5_razor_seam(features=...)`).
#pragma once

#include <complex>
#include <cstdint>
#include <memory>

#include "_exact_kernel_point_core.h"  // PointRules: the same five rules

namespace mw {
namespace ek {

// `_exact_kernel_series.X_MAX`, `X_CENTRED`, `SERIES_TOL`: the term counts'
// bounds (a test pins them to the module's).
constexpr double SERIES_X_MAX = 4.0;
constexpr double SERIES_X_CENTRED = 1.0;
constexpr double SERIES_TOL = 1e-17;

// `_exact_kernel_series.serves(k, group_reach(h, a))`: whether the series
// takes a one-radius group of n segments of lengths h and radius a at k.
bool series_serves(double k, const double *h, int64_t n, double a);

// `_exact_kernel_series.PointSeriesAccel`: one coaxial group's segments
// (start x0 along the axis, direction sgn, length h; radius a), the moments
// N_q, q < nq <= 2, at observer positions along the axis and a wavenumber
// given per call, from coefficients kept per unique pair (keys first met
// first, capped at max_cached; quantum q).
class PointSeries {
  public:
    PointSeries(const double *x0, const double *sgn, const double *h, int64_t n, double a, int nq,
                int64_t max_cached, double q, const PointRules &rules, int n_threads);
    ~PointSeries();
    PointSeries(PointSeries &&) noexcept;
    int64_t n() const;
    int64_t built() const;
    int64_t reused() const;
    int64_t evaluated() const;
    // (nq, m, n) C order into out.
    void rows(const double *x_obs, int64_t m, double k, cd *out);

  private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace ek
}  // namespace mw
