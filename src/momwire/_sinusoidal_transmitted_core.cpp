// The point-matched lane's transmitted blocks: see
// `_sinusoidal_transmitted_core.h`. numpy's arithmetic per element in its
// order: its array product fused, its einsum's complex inner product plain C
// (`MW_NUMPY_CONTRACTS`), libm's csin / ccos and sin / cos.

#if defined(_MSC_VER) && !defined(__clang__)
#pragma float_control(precise, on)
#pragma fp_contract(off)
#endif

#include "_sinusoidal_transmitted_core.h"

#include "_fma_inline.h"
#include "_libm_complex_inline.h"
#include "_razor_front_core.h"  // `leggauss`

#include <cstddef>
#include <stdexcept>
#include <string>

namespace mw {
namespace sinusoidal {

namespace {

typedef long long ll;
typedef std::complex<double> cd;
using nec4::Point3;

// numpy's array product (fused); a real operand is (x, 0)
inline cd xmul(cd a, cd b) {
    return cd(mw_fma::exact(a.real(), b.real(), -(a.imag() * b.imag())),
              mw_fma::exact(a.real(), b.imag(), a.imag() * b.real()));
}
// numpy's einsum inner product of complex operands (`sum_of_products`), plain
// C: unfused on x86-64, each part contracted on arm64
inline cd e_mul(cd a, cd b) {
#if MW_NUMPY_CONTRACTS
    return cd(mw_fma::exact(a.real(), b.real(), -(a.imag() * b.imag())),
              mw_fma::exact(a.real(), b.imag(), a.imag() * b.real()));
#else
    return cd(a.real() * b.real() - a.imag() * b.imag(), a.real() * b.imag() + a.imag() * b.real());
#endif
}

}  // namespace

std::vector<std::vector<cd>> transmitted_block(const Front &fr, const somm_trans::CrossPlan &plan, ll q, double mu,
                                               bool obs_below, const std::vector<ll> &obs_idx,
                                               const std::vector<ll> &src_idx) {
    const Geometry &g = fr.geometry;
    const Medium &md = fr.medium;
    const double gz = 0.0;
    const std::size_t Q = static_cast<std::size_t>(q), ns = src_idx.size(), no = obs_idx.size();
    // `field_nodes(seg_l[idx], seg_r[idx], tang[idx], h[idx], q)`
    std::vector<double> xg, wg;
    if (!razor::leggauss(q, xg, wg)) throw nec4::Declined("an untabled transmitted field order");
    std::vector<double> nodes(3 * ns * Q), t_src(3 * ns * Q);
    // the three source shapes times the node weight (`shp * w_node`), (3, ns, q)
    std::vector<cd> shp_w(3 * ns * Q);
    const bool cplx = !obs_below;  // `k_src = k_p if obs_below else k_m`
    for (std::size_t n = 0; n < ns; n++) {
        const std::size_t z = static_cast<std::size_t>(src_idx[n]);
        const Point3 &l = g.seg_l[z], &r = g.seg_r[z];
        const double h = g.seg_h[z], half_h = 0.5 * h;
        for (std::size_t j = 0; j < Q; j++) {
            const double tq = 0.5 * (xg[j] + 1.0);
            const std::size_t at = n * Q + j;
            for (int c = 0; c < 3; c++) {
                nodes[3 * at + static_cast<std::size_t>(c)] = l[static_cast<std::size_t>(c)] +
                    tq * (r[static_cast<std::size_t>(c)] - l[static_cast<std::size_t>(c)]);
                t_src[3 * at + static_cast<std::size_t>(c)] = g.seg_tangents[z][static_cast<std::size_t>(c)];
            }
            const double u = h * tq, w = half_h * wg[j];
            const double zloc = u - half_h;
            cd s1, s2;
            if (cplx) {
                const cd arg = xmul(md.k_m, cd(zloc, 0.0));
                s1 = mw_libm_c::csin(arg);
                s2 = mw_libm_c::ccos(arg);
            } else {
                const double arg = md.k_p * zloc;
                s1 = cd(mw_libm_c::rsin(arg), 0.0);
                s2 = cd(mw_libm_c::rcos(arg), 0.0);
            }
            shp_w[0 * ns * Q + at] = xmul(cd(1.0, 0.0), cd(w, 0.0));
            shp_w[1 * ns * Q + at] = xmul(s1, cd(w, 0.0));
            shp_w[2 * ns * Q + at] = xmul(s2, cd(w, 0.0));
        }
    }
    std::vector<double> oc(3 * no), ot(3 * no);
    for (std::size_t m = 0; m < no; m++) {
        const std::size_t z = static_cast<std::size_t>(obs_idx[m]);
        for (int c = 0; c < 3; c++) {
            oc[3 * m + static_cast<std::size_t>(c)] = g.seg_centers[z][static_cast<std::size_t>(c)];
            ot[3 * m + static_cast<std::size_t>(c)] = g.seg_tangents[z][static_cast<std::size_t>(c)];
        }
    }
    const somm_trans::Grid *grid = nullptr;
    try {
        grid = &somm_trans::get_grid(md.eps_t, md.k_p, plan.r_cross_max, plan.zp_min, plan.zp_max, fr.omega, mu,
                                     plan.r_cross_min);
    } catch (const somm_trans::Refused &e) {
        throw nec4::Declined(std::string("a transmitted grid Python refuses (") + e.what() + ")");
    }
    const std::size_t S = ns * Q;
    std::vector<cd> proj(no * S);  // (n_obs, n_src q)
    somm_trans::Extremes ext;
    if (!obs_below) {
        // `transmitted_field_proj_below_to_above(obs_c, obs_t, nodes, t_src, ...)`
        somm_trans::project(*grid, oc.data(), ot.data(), no, nodes.data(), t_src.data(), S, gz, md.k_p, md.k_m,
                            false, proj.data(), ext);
    } else {
        // `transmitted_field_proj_above_to_below`: the role-order table, transposed
        std::vector<cd> role(S * no);
        somm_trans::project(*grid, nodes.data(), t_src.data(), S, oc.data(), ot.data(), no, gz, md.k_p, md.k_m,
                            true, role.data(), ext);
        for (std::size_t a = 0; a < S; a++)
            for (std::size_t m = 0; m < no; m++) proj[m * S + a] = role[a * no + m];
    }
    if (no && S) {
        const std::string why = somm_trans::eval_refusal(*grid, ext);
        if (!why.empty()) throw nec4::Declined("a transmitted query the grid refuses (" + why + ")");
    }
    // `np.einsum("snq,mnq->smn", shp_w, fq)`: each (s, m, n) summed over q in
    // order, from zero
    std::vector<std::vector<cd>> out(3, std::vector<cd>(no * ns));
    for (int s = 0; s < 3; s++)
        for (std::size_t m = 0; m < no; m++)
            for (std::size_t n = 0; n < ns; n++) {
                double ar = 0.0, ai = 0.0;
                for (std::size_t j = 0; j < Q; j++) {
                    const cd p = e_mul(shp_w[static_cast<std::size_t>(s) * S + n * Q + j], proj[m * S + n * Q + j]);
                    ar = ar + p.real();
                    ai = ai + p.imag();
                }
                out[static_cast<std::size_t>(s)][m * ns + n] = cd(ar, ai);
            }
    return out;
}

}  // namespace sinusoidal
}  // namespace mw
