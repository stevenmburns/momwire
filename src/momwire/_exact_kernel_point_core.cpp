// See `_exact_kernel_point_core.h`. The TU sees what the extension's point
// TU sees (`_mw_core.h`'s libmvec declarations, `_fma_inline.h`), with the
// numpy-backed buffers replaced by plain vectors and no GIL to release.
#include "_exact_kernel_point_core.h"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <complex>
#include <cstdint>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <tuple>
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
namespace detail {
#include "_exact_kernel_ring_inline.h"
#include "_exact_kernel_table_inline.h"
#include "_exact_kernel_point_inline.h"
#include "_exact_kernel_cross_inline.h"
}  // namespace detail

struct PointRows::Impl {
    detail::PointRowsCore core;
};

static detail::Rules to_rules(const PointRules &r) {
    detail::Rules R;
    R.xg = r.xg;
    R.wg = r.wg;
    R.xs = r.xs;
    R.ws = r.ws;
    R.wls = r.wls;
    return R;
}

PointRows::PointRows(const double *x0, const double *sgn, const double *h, int64_t n, double a,
                     double k, int nq, int64_t max_cached, double q, const PointRules &rules,
                     int n_threads)
    : impl_(new Impl{detail::PointRowsCore(x0, sgn, h, n, a, k, nq, max_cached, q,
                                           to_rules(rules), n_threads)}) {
    if (nq < 1 || nq > detail::MAXND) throw std::runtime_error("PointRows: nq must be in [1, 6]");
    if (!(q > 0.0)) throw std::runtime_error("PointRows: q must be positive");
}
PointRows::~PointRows() = default;
PointRows::PointRows(PointRows &&) noexcept = default;
int64_t PointRows::n() const { return impl_->core.n(); }
int64_t PointRows::evaluated() const { return impl_->core.evaluated(); }
void PointRows::rows(const double *x_obs, int64_t m, cd *out) { impl_->core.rows(x_obs, m, out); }

struct PointCrossRows::Impl {
    detail::PointCrossRowsCore core;
};

PointCrossRows::PointCrossRows(const double *x0, const double *sgn, const double *h,
                               const double *a, const int64_t *lab, int64_t n, double k, int nq,
                               int64_t max_cached, double q, double qa, const PointRules &rules,
                               int n_threads)
    : impl_(new Impl{detail::PointCrossRowsCore(x0, sgn, h, a, lab, n, k, nq, max_cached, q, qa,
                                                to_rules(rules), n_threads)}) {
    if (nq < 1 || nq > detail::MAXND)
        throw std::runtime_error("PointCrossRows: nq must be in [1, 6]");
    if (!(q > 0.0) || !(qa > 0.0))
        throw std::runtime_error("PointCrossRows: q and qa must be positive");
}
PointCrossRows::~PointCrossRows() = default;
PointCrossRows::PointCrossRows(PointCrossRows &&) noexcept = default;
int64_t PointCrossRows::n() const { return impl_->core.n(); }
int64_t PointCrossRows::evaluated() const { return impl_->core.evaluated(); }
void PointCrossRows::rows(const double *x_obs, const double *a_obs, const int64_t *lab_obs,
                          int64_t m, cd *out) {
    impl_->core.rows(x_obs, a_obs, lab_obs, m, out);
}

void point_assemble(const cd *dM0, const cd *dM1, int64_t n, const int64_t *row_m,
                    const int64_t *row_o, const cd *row_cA, const cd *row_cP, int64_t R,
                    const int64_t *col_ptr, const int64_t *col_n, int64_t NB,
                    const int64_t *ent_j, const double *ent_t, const uint8_t *ent_rise,
                    const double *ent_q, const double *h, cd *Z, int64_t ld, int n_threads) {
    detail::point_assemble(dM0, dM1, n, row_m, row_o, row_cA, row_cP, R, col_ptr, col_n, NB, ent_j,
                           ent_t, ent_rise, ent_q, h, Z, ld, detail::resolve_threads(n_threads));
}

}  // namespace ek
}  // namespace mw
