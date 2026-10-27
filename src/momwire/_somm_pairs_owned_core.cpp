// See `_somm_pairs_owned_core.h`. The loop `_accel_somm_pairs.cpp` ran, moved
// verbatim (momwire#1718 S5b).
#include "_mw_core.h"
#include "_somm_pairs_owned_core.h"

#ifdef _OPENMP
#include <omp.h>
#endif

namespace somm_core {

void remainder_field_proj_owned(const somm_proj::GridView &G, double ground_z, double k, const double *obs,
                                const double *t_obs, const double *src, const double *t_src, const int64_t *owner,
                                std::ptrdiff_t S, std::complex<double> *out, uintptr_t cancel_flag) {
    MW_CANCEL_SETUP(cancel_flag);
    #pragma omp parallel for schedule(static)
    for (std::ptrdiff_t n = 0; n < S; ++n) {
        MW_CANCEL_POLL();
        const std::ptrdiff_t m = static_cast<std::ptrdiff_t>(owner[n]);
        double ux, uy, th, tz;
        somm_proj::tangent_decomp(t_src[n * 3], t_src[n * 3 + 1], t_src[n * 3 + 2], ux, uy, th, tz);
        out[n] = somm_proj::proj_one(G, ground_z, k, obs[m * 3], obs[m * 3 + 1], obs[m * 3 + 2], t_obs[m * 3],
                                     t_obs[m * 3 + 1], t_obs[m * 3 + 2], src[n * 3], src[n * 3 + 1],
                                     src[n * 3 + 2], ux, uy, th, tz);
    }
    MW_THROW_IF_ABORTED();
}

}  // namespace somm_core
