// The crossing table's build and read (momwire#1860 L5, the opt-in feature
// `crossing_table`). Its own translation unit, so that nothing the stable
// path compiles changes: the column twin is called across the TU boundary,
// and the per-row arithmetic is `_crossing_table_inline.h`.
#include "_crossing_table_core.h"

#include <cmath>
#include <stdexcept>
#include <vector>

#include "_crossing_table_inline.h"
#include "_near_interface_core.h"

namespace ni_core {

using cplx = std::complex<double>;

void crossing_table_build(double k_p, cplx k_m, const double *h, const std::ptrdiff_t *n,
                          double lam_mult, int p, double detour, int n_threads,
                          const double *gx, const double *gw, int ng, uintptr_t cancel_flag,
                          cplx *vals) {
    const std::ptrdiff_t n0 = n[0], n1 = n[1], n2 = n[2];
    if (n0 < 4 || n1 < 4 || n2 < 4) throw std::invalid_argument("crossing table needs 4 nodes an axis");
    const std::ptrdiff_t plane = n1 * n2;
    // One column per rho node; its members are every (z, z') node, z-major,
    // less the R = 0 node of the rho = 0 column.
    std::vector<double> rho(n0), z, zp;
    std::vector<std::ptrdiff_t> off(n0 + 1, 0);
    z.reserve(n0 * plane);
    zp.reserve(n0 * plane);
    for (std::ptrdiff_t i = 0; i < n0; ++i) {
        rho[i] = static_cast<double>(i) * h[0];
        for (std::ptrdiff_t j = 0; j < n1; ++j)
            for (std::ptrdiff_t k = 0; k < n2; ++k) {
                if (i == 0 && j == 0 && k == 0) continue;
                z.push_back(static_cast<double>(j) * h[1]);
                zp.push_back(-static_cast<double>(k) * h[2]);
            }
        off[i + 1] = static_cast<std::ptrdiff_t>(z.size());
    }
    const std::ptrdiff_t m = off[n0];
    std::vector<cplx> f(static_cast<size_t>(m) * 6);
    six_columns(k_p, k_m, rho.data(), off.data(), n0, z.data(), zp.data(), m, lam_mult, p,
                detour, n_threads, gx, gw, ng, cancel_flag, nullptr, f.data(), true, true,
                true);
    for (std::ptrdiff_t q = 0; q < 6; ++q) vals[q] = 0.0;  // the R = 0 node
    for (std::ptrdiff_t i = 0; i < n0; ++i) {
        std::ptrdiff_t r = off[i];
        for (std::ptrdiff_t jk = (i == 0 ? 1 : 0); jk < plane; ++jk, ++r) {
            cplx S[6];
            mw1860::crossing_singular(rho[i], z[r], zp[r], k_p, k_m, S);
            const double R = std::hypot(rho[i], z[r] - zp[r]);
            cplx *v = vals + (i * plane + jk) * 6;
            for (int q = 0; q < 6; ++q) v[q] = R * (f[r * 6 + q] - S[q]);
        }
    }
}

void crossing_table_lookup(double k_p, cplx k_m, const double *h, const std::ptrdiff_t *n,
                           const cplx *vals, const double *rho, const double *z,
                           const double *zp, std::ptrdiff_t m, int n_threads, cplx *out) {
    const std::ptrdiff_t n1 = n[1], n2 = n[2];
#ifdef _OPENMP
#pragma omp parallel for schedule(static) num_threads(n_threads) if (m > 4096)
#endif
    for (std::ptrdiff_t r = 0; r < m; ++r) {
        long i0, j0, k0;
        double wr[4], wz[4], wp[4];
        mw1860::crossing_stencil(rho[r], h[0], static_cast<long>(n[0]), i0, wr);
        mw1860::crossing_stencil(z[r], h[1], static_cast<long>(n1), j0, wz);
        mw1860::crossing_stencil(-zp[r], h[2], static_cast<long>(n2), k0, wp);
        cplx g[6] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
        for (int a = 0; a < 4; ++a)
            for (int b = 0; b < 4; ++b) {
                const double wab = wr[a] * wz[b];
                const cplx *row = vals + (((i0 + a) * n1 + (j0 + b)) * n2 + k0) * 6;
                for (int c = 0; c < 4; ++c) {
                    const double w = wab * wp[c];
                    const cplx *v = row + c * 6;
                    for (int q = 0; q < 6; ++q) g[q] += w * v[q];
                }
            }
        cplx S[6];
        mw1860::crossing_singular(rho[r], z[r], zp[r], k_p, k_m, S);
        const double R = std::hypot(rho[r], z[r] - zp[r]);
        cplx *o = out + r * 6;
        for (int q = 0; q < 6; ++q) o[q] = g[q] / R + S[q];
    }
    (void)n_threads;
}

}  // namespace ni_core
