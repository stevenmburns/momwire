// See `_exact_kernel_coaxial_core.h`. The TU sees what the extension's 1410
// TU sees (`_mw_core.h`'s libmvec declarations, `_fma_inline.h`), with the
// numpy-backed buffers replaced by plain vectors and no GIL to release.
#include "_exact_kernel_coaxial_core.h"

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
namespace coaxial_detail {
#include "_exact_kernel_ring_inline.h"
#include "_exact_kernel_coaxial_inline.h"
}  // namespace coaxial_detail

namespace {

coaxial_detail::Rules to_rules(const PointRules &r) {
    coaxial_detail::Rules R;
    R.xg = r.xg;
    R.wg = r.wg;
    R.xs = r.xs;
    R.ws = r.ws;
    R.wls = r.wls;
    return R;
}

}  // namespace

static_assert(COAXIAL_MAX_ND == coaxial_detail::MAXND, "the moment count bound is the kernel's");

void ring_kernel(const double *z, int64_t n, double a, double k, cd *out) {
    for (int64_t i = 0; i < n; ++i) out[i] = coaxial_detail::ring_kernel(z[i], a, k);
}

void ring_kernel_split(const double *z, int64_t n, double a, double k, cd *A, cd *B) {
    for (int64_t i = 0; i < n; ++i) {
        const double zi = std::fabs(z[i]);
        if (zi >= coaxial_detail::T_SWITCH * a)
            throw std::runtime_error("exact_kernel_split serves |z| < 6a only");
        coaxial_detail::near_eval(zi, a, k, true, A[i], B[i]);
    }
}

void coaxial_pair_moments(double hi, double hj, double c, double si, double sj, double a,
                          double k, int nd, const PointRules &rules, int far, cd *out) {
    using namespace coaxial_detail;
    if (nd < 1 || nd > MAXND) throw std::runtime_error("pair moments: nd must be in [1, 6]");
    const Rules R = to_rules(rules);
    Geo g;
    g.hi = hi;
    g.hj = hj;
    g.si = si;
    g.sj = sj;
    g.nd = nd;
    g.a = a;
    g.k = k;
    g.R = &R;
    cd M[NM];
    if (far < 0)
        pair_moments(g, c, M);
    else if (far == 0)
        pair_moments_near(g, c, M);
    else
        pair_moments_far(g, c, M);
    for (int p = 0; p < nd; ++p)
        for (int q = 0; q < nd; ++q) out[p * nd + q] = M[p * MAXND + q];
}

struct CoaxialRows::Impl {
    coaxial_detail::CoaxialRowsCore core;
};

CoaxialRows::CoaxialRows(const double *x0, const double *sgn, const double *h, int64_t n,
                         double a, double k, int nd, int64_t max_cached, double q, bool symmetric,
                         const PointRules &rules, int n_threads)
    : impl_(new Impl{coaxial_detail::CoaxialRowsCore(x0, n, sgn, n, h, n, a, k, nd, max_cached, q,
                                                     symmetric, to_rules(rules), n_threads)}) {}
CoaxialRows::~CoaxialRows() = default;
CoaxialRows::CoaxialRows(CoaxialRows &&) noexcept = default;
int64_t CoaxialRows::n() const { return impl_->core.n(); }
int64_t CoaxialRows::evaluated() const { return impl_->core.evaluated(); }
int64_t CoaxialRows::cached() const { return impl_->core.cached(); }
void CoaxialRows::rows(int64_t r0, int64_t r1, cd *out) { impl_->core.rows(r0, r1, out); }

}  // namespace ek
}  // namespace mw
