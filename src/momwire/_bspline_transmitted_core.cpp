// bs2's transmitted blocks: see `_bspline_transmitted_core.h`.
#include "_bspline_transmitted_core.h"

#include "_field_galerkin_inline.h"

#include <algorithm>

namespace mw {
namespace bspline {

namespace {

using cdouble = std::complex<double>;

// observers `o` against sources `s`; `up`: the observers are the above class
void cross_block(std::vector<cdouble> &Z, const std::vector<double> &polys, const std::vector<int64_t> &supp_seg,
                 int nd, int64_t n_basis, int64_t q, const somm_trans::Grid &grid, double gz, double k_p,
                 cdouble k_m, bool up, const TransmittedSide &o, const TransmittedSide &s) {
    if (o.n == 0 || s.n == 0) return;
    const size_t Q = static_cast<size_t>(q);
    const int64_t chunk = std::max<int64_t>(1, (int64_t(1) << 19) / std::max<int64_t>(s.n * q * q, 1));
    const size_t S = static_cast<size_t>(s.n) * Q;
    const double *sn = s.nodes->data(), *ts = s.tans->data();
    for (int64_t i0 = 0; i0 < o.n; i0 += chunk) {
        const int64_t i1 = std::min(i0 + chunk, o.n);
        const size_t m = static_cast<size_t>(i1 - i0) * Q;
        const double *op = o.nodes->data() + 3 * static_cast<size_t>(i0) * Q;
        const double *otp = o.tans->data() + 3 * static_cast<size_t>(i0) * Q;
        std::vector<cdouble> proj(m * S);
        somm_trans::Extremes ext;
        if (up) {
            // `transmitted_field_proj_below_to_above`
            somm_trans::project(grid, op, otp, m, sn, ts, S, gz, k_p, k_m, false, proj.data(), ext);
        } else {
            // `transmitted_field_proj_above_to_below`: the role-order table
            // (above = the sources, below = the observers), transposed into
            // (n_obs, n_src)
            std::vector<cdouble> role(S * m);
            somm_trans::project(grid, sn, ts, S, op, otp, m, gz, k_p, k_m, true, role.data(), ext);
            for (size_t a = 0; a < S; a++)
                for (size_t r = 0; r < m; r++) proj[r * S + a] = role[a * m + r];
        }
        const std::string why = somm_trans::eval_refusal(grid, ext);
        if (!why.empty()) throw somm_trans::Refused(why);
        std::vector<double> W_rows(static_cast<size_t>(nd) * m);
        const std::vector<double> &wo = *o.W;
        for (int p = 0; p < nd; p++)
            std::copy(wo.begin() + (static_cast<std::ptrdiff_t>(p) * o.n + i0) * q,
                      wo.begin() + (static_cast<std::ptrdiff_t>(p) * o.n + i1) * q,
                      W_rows.begin() + static_cast<std::ptrdiff_t>(p) * (i1 - i0) * q);
        mw_fg::Args args;
        args.pj = proj.data();
        args.wo = W_rows.data();
        args.ws = s.W->data();
        args.pl = polys.data();
        args.ss = supp_seg.data();
        args.po = o.pos->data();
        args.ps = s.pos->data();
        args.P = nd;
        args.nc = i1 - i0;
        args.q = q;
        args.ns = s.n;
        args.nb = n_basis;
        args.A = nd;
        args.i0 = i0;
        args.q_rows = n_basis;
        args.Qp = Z.data();
        args.sr = n_basis;
        args.sc = 1;
        args.fused = true;
        args.scale = -1.0;
        args.lanes = true;
        mw_fg::assemble(args);
    }
}

}  // namespace

void transmitted_blocks(std::vector<cdouble> &Z, const std::vector<double> &polys, const std::vector<int64_t> &supp_seg,
                        int nd, int64_t n_basis, int64_t q, const somm_trans::Grid &grid, double ground_z,
                        double k_p, cdouble k_m, const TransmittedSide &above, const TransmittedSide &below) {
    cross_block(Z, polys, supp_seg, nd, n_basis, q, grid, ground_z, k_p, k_m, true, above, below);
    cross_block(Z, polys, supp_seg, nd, n_basis, q, grid, ground_z, k_p, k_m, false, below, above);
}

}  // namespace bspline
}  // namespace mw
