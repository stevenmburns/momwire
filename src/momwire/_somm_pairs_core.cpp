// See `_somm_pairs_core.h`. Moved verbatim (momwire#1517 step 3).
#include "_mw_core.h"
#include "_somm_pairs_core.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <vector>
#ifdef _OPENMP
#include <omp.h>
#endif

namespace somm_core {

void remainder_graded_inner(const somm_proj::GridView &G, double ground_z, double k, const double *obs,
                            const double *t_obs, const double *a0, const double *dvec, const double *t_src,
                            const double *len, const int64_t *ptr, std::ptrdiff_t M, const double *p0,
                            const double *p1, const double *xg, const double *wg, std::ptrdiff_t NG, int d1,
                            std::complex<double> *out, uintptr_t cancel_flag) {
    using somm_proj::cd;
    const auto ob = [&](std::ptrdiff_t m, int c) { return obs[m * 3 + c]; };
    const auto tob = [&](std::ptrdiff_t m, int c) { return t_obs[m * 3 + c]; };
    const auto A = [&](std::ptrdiff_t m, int c) { return a0[m * 3 + c]; };
    const auto D = [&](std::ptrdiff_t m, int c) { return dvec[m * 3 + c]; };
    const auto ts = [&](std::ptrdiff_t m, int c) { return t_src[m * 3 + c]; };
    const auto Ln = [&](std::ptrdiff_t m) { return len[m]; };
    const auto pt = [&](std::ptrdiff_t m) { return ptr[m]; };
    const auto P0 = [&](std::ptrdiff_t p) { return p0[p]; };
    const auto P1 = [&](std::ptrdiff_t p) { return p1[p]; };
    const auto X = [&](std::ptrdiff_t g) { return xg[g]; };
    const auto Wg = [&](std::ptrdiff_t g) { return wg[g]; };
    const auto om = [&](std::ptrdiff_t m, int P) -> cd & { return out[m * d1 + P]; };
    MW_CANCEL_SETUP(cancel_flag);
    #pragma omp parallel for schedule(dynamic, 64)
    for (std::ptrdiff_t m = 0; m < M; ++m) {
        MW_CANCEL_POLL();
        double ux, uy, th, tz;
        somm_proj::tangent_decomp(ts(m, 0), ts(m, 1), ts(m, 2), ux, uy, th, tz);
        const double ox = ob(m, 0), oy = ob(m, 1), oz = ob(m, 2);
        const double tox = tob(m, 0), toy = tob(m, 1), toz = tob(m, 2);
        const double L = Ln(m);
        std::vector<cd> acc((size_t)d1, cd(0.0, 0.0));
        for (std::ptrdiff_t p = pt(m); p < pt(m + 1); ++p) {
            const double half = 0.5 * (P1(p) - P0(p));
            for (std::ptrdiff_t g = 0; g < NG; ++g) {
                const double t = P0(p) + half * (X(g) + 1.0);
                const double sx = A(m, 0) + t * D(m, 0);
                const double sy = A(m, 1) + t * D(m, 1);
                const double sz = A(m, 2) + t * D(m, 2);
                const cd F = somm_proj::proj_one(G, ground_z, k, ox, oy, oz, tox,
                                                 toy, toz, sx, sy, sz, ux, uy,
                                                 th, tz);
                const double u = L * t;
                cd term = F * (L * half * Wg(g));
                for (int P = 0; P < d1; ++P) {
                    acc[(size_t)P] += term;
                    term *= u;
                }
            }
        }
        for (int P = 0; P < d1; ++P) om(m, P) = acc[(size_t)P];
    }
    MW_THROW_IF_ABORTED();
}

static const double SLACK = 1e-10;  // `_accel_qp_pairs.cpp`'s

void remainder_qp_pairs_scan(const double *obs, std::ptrdiff_t M, int64_t n_node, const double *mir_l,
                             const double *d, const double *dd, const double *lengths, std::ptrdiff_t S,
                             int64_t base, int64_t cap, double c, double edge_tol,
                             std::vector<std::vector<QpHit>> &hits, std::vector<unsigned char> &unsure) {
    using Hit = QpHit;
    const std::ptrdiff_t n_seg = M / n_node;
    const double *O = obs;
    const double *ML = mir_l;
    const double *D = d;
    const double *DD = dd;
    const double *L = lengths;
    const double lo_b = edge_tol != 0.0 ? -edge_tol : 0.0;
    const double hi_b = edge_tol != 0.0 ? 1.0 + edge_tol : 1.0;
    hits.assign(static_cast<size_t>(S), {});
    unsure.assign(static_cast<size_t>(S), 0);
    {
        #pragma omp parallel for schedule(dynamic, 16)
        for (std::ptrdiff_t j = 0; j < S; ++j) {
            const double len = L[j], ddj = DD[j];
            if (len <= 0.0 || !(ddj > 0.0)) continue;  // numpy skips both
            const double mx = ML[3 * j], my = ML[3 * j + 1], mz = ML[3 * j + 2];
            const double dx = D[3 * j], dy = D[3 * j + 1], dz = D[3 * j + 2];
            const double dmax =
                std::max(std::fabs(dx), std::max(std::fabs(dy), std::fabs(dz)));
            std::vector<Hit> &out = hits[j];
            bool bad = false;
            for (std::ptrdiff_t s = 0; s < n_seg; ++s) {
                // numpy's R_min for this observer segment lies in
                // [r_lo, r_hi]: each interior node's dist within +-e of ours.
                const double inf = std::numeric_limits<double>::infinity();
                double r_min = inf, r_lo = inf, r_hi = inf;
                for (int64_t n = 0; n < n_node; ++n) {
                    const double *o = O + 3 * (s * n_node + n);
                    const double ax = o[0] - mx, ay = o[1] - my, az = o[2] - mz;
                    const double pd = ax * dx + ay * dy + az * dz;
                    const double t_raw = pd / ddj;
                    // |numpy's t_raw - ours|: a 3-term dot rounds within a few
                    // eps sum|a_i d_i| whatever its order or fusion.
                    const double mag =
                        (std::fabs(ax * dx) + std::fabs(ay * dy) +
                         std::fabs(az * dz)) / ddj;
                    const double tol = SLACK * (mag + std::fabs(t_raw));
                    if (std::fabs(t_raw - lo_b) <= tol ||
                        std::fabs(t_raw - hi_b) <= tol) {
                        bad = true;
                        break;
                    }
                    if (!(t_raw > lo_b && t_raw < hi_b)) continue;
                    const double t = t_raw < 0.0 ? 0.0 : (t_raw > 1.0 ? 1.0 : t_raw);
                    const double cx = o[0] - (mx + t * dx);
                    const double cy = o[1] - (my + t * dy);
                    const double cz = o[2] - (mz + t * dz);
                    const double dist = std::sqrt(cx * cx + cy * cy + cz * cz);
                    // t can be off by tol, which moves `closest` by tol |d|;
                    // every other float here rounds within SLACK of the
                    // coordinates' scale.
                    const double scale =
                        std::max(std::max(std::fabs(o[0]), std::fabs(o[1])),
                                 std::max(std::fabs(o[2]),
                                          std::max(std::max(std::fabs(mx), std::fabs(my)),
                                                   std::fabs(mz)))) +
                        dmax + dist;
                    const double e = tol * dmax + SLACK * scale;
                    r_min = std::min(r_min, dist);
                    r_lo = std::min(r_lo, dist - e);
                    r_hi = std::min(r_hi, dist + e);
                }
                if (bad) break;
                if (!(r_min < inf)) continue;  // no interior node: base order
                // numpy: ratio = len / R_min (R_min > 0), need = ceil(c ratio),
                // q = clip(need, base, cap); R_min == 0 -> cap. Certain only
                // if every R_min in [r_lo, r_hi] gives the same q.
                if (r_lo <= 0.0) {
                    bad = true;
                    break;
                }
                const double need = std::ceil(c * (len / r_min));
                const double clo = std::ceil(c * (len / r_hi) * (1.0 - SLACK));
                const double chi = std::ceil(c * (len / r_lo) * (1.0 + SLACK));
                const double qn = std::min(std::max(need, (double)base), (double)cap);
                const double ql = std::min(std::max(clo, (double)base), (double)cap);
                const double qh = std::min(std::max(chi, (double)base), (double)cap);
                if (ql != qn || qh != qn) {
                    bad = true;
                    break;
                }
                const int64_t q = static_cast<int64_t>(qn);
                if (q > base) out.push_back(Hit{static_cast<int64_t>(s), q});
            }
            if (bad) {
                unsure[j] = 1;
                out.clear();
            }
        }
    }

}

}  // namespace somm_core
