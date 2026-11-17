// See `_exact_kernel_series_core.h`. The TU sees what the extension's series
// TU sees (`_mw_core.h`'s libmvec declarations, `_fma_inline.h`), with the
// numpy-backed buffers replaced by plain vectors and no GIL to release, as
// `_exact_kernel_point_core.cpp` does for the direct rule.
#include "_exact_kernel_series_core.h"

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <vector>

#include "_fma_inline.h"
#include "_mw_core.h"
#ifdef _OPENMP
#include <omp.h>
#endif

#define MW_EK_NO_PYTHON 1
#define MW_EK_RELEASE_GIL

namespace mw {
namespace ek {
namespace series_detail {
#include "_exact_kernel_ring_inline.h"
#include "_exact_kernel_table_inline.h"
#include "_exact_kernel_series_inline.h"
}  // namespace series_detail

bool series_serves(double k, const double *h, int64_t n, double a) {
    // `group_reach`: the largest R a pair with Rc = 0 can see, and the
    // largest |R - Rc| a CENTRED one can.
    double hm = 0.0;
    if (n > 0) hm = *std::max_element(h, h + n);
    const double r_plain = std::hypot(std::max(2.0 * hm, series_detail::T_SWITCH * a + hm), 2.0 * a);
    const double d_centred = 0.5 * hm + a;
    const double ak = std::fabs(k);
    return ak * r_plain <= SERIES_X_MAX && ak * d_centred <= SERIES_X_CENTRED;
}

struct PointSeries::Impl {
    series_detail::PointSeriesCore core;
};

static series_detail::Rules to_rules(const PointRules &r) {
    series_detail::Rules R;
    R.xg = r.xg;
    R.wg = r.wg;
    R.xs = r.xs;
    R.ws = r.ws;
    R.wls = r.wls;
    return R;
}

PointSeries::PointSeries(const double *x0, const double *sgn, const double *h, int64_t n, double a,
                         int nq, int64_t max_cached, double q, const PointRules &rules,
                         int n_threads)
    : impl_(new Impl{series_detail::PointSeriesCore(x0, sgn, h, n, a, nq, max_cached, q,
                                                    to_rules(rules), SERIES_X_MAX,
                                                    SERIES_X_CENTRED, SERIES_TOL, n_threads)}) {}
PointSeries::~PointSeries() = default;
PointSeries::PointSeries(PointSeries &&) noexcept = default;
int64_t PointSeries::n() const { return impl_->core.n(); }
int64_t PointSeries::built() const { return impl_->core.built(); }
int64_t PointSeries::reused() const { return impl_->core.reused(); }
int64_t PointSeries::evaluated() const { return impl_->core.evaluated(); }
void PointSeries::rows(const double *x_obs, int64_t m, double k, cd *out) {
    impl_->core.rows(x_obs, m, std::complex<double>(k, 0.0), out);
}

}  // namespace ek
}  // namespace mw
