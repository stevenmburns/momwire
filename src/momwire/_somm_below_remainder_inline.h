// The below/below remainder's projection over a batch of observers
// (`_sommerfeld_below._remainder_field_proj_below`'s C++ branch): the kernel
// run, the past-cap extremes, the domain refusals, the lazy fill and its
// re-run, the zeroing past the cap. Moved from `_razor_buried_core.cpp`
// (momwire#1517 step 4) so bs2's buried fill projects through the same code;
// `Raise::value(message)` raises Python's ValueError in the caller's terms.
#pragma once

#include "_somm_below_grid_core.h"
#include "_somm_below_proj_core.h"
#include "_somm_grid_core.h"
#include "_somm_proj_core.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <limits>
#include <string>
#include <vector>

namespace somm_below {

// `%.6g`, Python's `:.6g`
inline std::string remainder_g6(double x) {
    char buf[64];
    std::snprintf(buf, sizeof buf, "%.6g", x);
    return buf;
}

// The below remainder of one medium front (razor's `RemainderBelow.field_windows` at
// n_moment = 2, n_qp flat): the grid, the source nodes, the moment weights,
// and the projection wrapper `_remainder_field_proj_below` per window.
template <class Raise>
struct Remainder {
    double gz = 0, k_p = 0;
    cd k_m;
    somm_below::Grid *grid = nullptr;
    somm_proj::GridView view;
    std::vector<double> src, t_src, W, d_src;
    ptrdiff_t n_qp = 0;
    // observers: the front's path points and tangents
    const double *obs = nullptr, *t_obs = nullptr;

    void build_view() {
        double r0[15], dr[15], th0[15], dth[15];
        const cd *vals[15];
        std::ptrdiff_t shape0[15], nr[15], nth[15];
        for (int r = 0; r < 15; r++) {
            const Region &reg = grid->regions[r];
            r0[r] = reg.r0;
            dr[r] = reg.dr;
            th0[r] = reg.th0;
            dth[r] = reg.dth;
            vals[r] = reg.vals.data();
            shape0[r] = 4;
            nr[r] = static_cast<std::ptrdiff_t>(reg.n_r);
            nth[r] = static_cast<std::ptrdiff_t>(reg.n_th);
        }
        view = somm_proj::build_grid_view(grid->r1_max, grid->r_break, grid->th_split,
                                          grid->r_near, r0, dr, th0, dth, 15, vals, shape0, nr, nth);
    }

    // `_served_extremes(obs, src, d_obs, d_src, r1_cap)` over observers
    // [o0, o0 + m): numpy's hypot / sqrt classification, libm's atan2.
    bool served_extremes(const double *pts, ptrdiff_t m, double &r1_hi_o, double &lo_o,
                         double &hi_o) const {
        double lo = std::numeric_limits<double>::infinity(), hi = -lo, r1_hi = -lo;
        const ptrdiff_t S = static_cast<ptrdiff_t>(d_src.size());
        for (ptrdiff_t i = 0; i < m; i++) {
            const double *o = pts + 3 * i;
            const double d_o = gz - o[2];
            for (ptrdiff_t j = 0; j < S; j++) {
                const double rho = std::hypot(o[0] - src[3 * j], o[1] - src[3 * j + 1]);
                const double hh = d_o + d_src[j];
                const double r1 = std::sqrt(rho * rho + hh * hh);
                if (!(r1 <= grid->r1_cap)) continue;
                const double th = std::atan2(hh, rho);
                lo = std::min(lo, th);
                hi = std::max(hi, th);
                r1_hi = std::max(r1_hi, r1);
            }
        }
        if (lo > hi) return false;
        r1_hi_o = r1_hi * (1.0 + 1e-12);
        lo_o = std::max(lo * (1.0 - 1e-12), 0.0);
        hi_o = std::min(hi * (1.0 + 1e-12), 0.5 * M_PI);
        return true;
    }

    // `_zero_past_cap`: numpy's hypot then the root.
    void zero_past_cap(const double *pts, ptrdiff_t m, cd *out) const {
        const ptrdiff_t S = static_cast<ptrdiff_t>(d_src.size());
        for (ptrdiff_t i = 0; i < m; i++) {
            const double *o = pts + 3 * i;
            const double d_o = gz - o[2];
            for (ptrdiff_t j = 0; j < S; j++) {
                const double rho = std::hypot(o[0] - src[3 * j], o[1] - src[3 * j + 1]);
                const double hh = d_o + d_src[j];
                if (::somm_core::below_past_cap(rho, hh, grid->r1_cap)) out[i * S + j] = cd(0.0, 0.0);
            }
        }
    }

    // `SommerfeldGridBelow._refuse_out_of_domain` on (R1, theta) pairs.
    void refuse_out_of_domain(const double *r1, const double *th, int n) const {
        double rmin = r1[0], rmax = r1[0], tmin = th[0], tmax = th[0];
        for (int i = 1; i < n; i++) {
            rmin = std::min(rmin, r1[i]);
            rmax = std::max(rmax, r1[i]);
            tmin = std::min(tmin, th[i]);
            tmax = std::max(tmax, th[i]);
        }
        if (rmin < 0.0) Raise::value("query R1 must be non-negative");
        if (rmax > grid->r1_max * (1.0 + 1e-9)) {
            char buf[128];
            std::snprintf(buf, sizeof buf, "%.3g", grid->r1_max / grid->lam_m);
            Raise::value("below/below grid tabulated to r1_max = " + remainder_g6(grid->r1_max) + " (" + buf +
                        " in-medium wavelengths) but was queried at R1 = " + remainder_g6(rmax) +
                        ": past the tabulation the below/below remainder is NOT a negligible "
                        "tail (it grows relative to the direct term with range), so "
                        "there is no honest clamp \xe2\x80\x94 size the grid for the geometry "
                        "or refuse the geometry");
        }
        if (tmax > 0.5 * M_PI + 1e-12)
            Raise::value("below/below grid theta must lie in [th_min, pi/2]; a theta "
                        "past pi/2 means the geometry is not two points below the "
                        "interface");
        if (tmin < grid->th_min * (1.0 - 1e-9)) {
            char a[64], b[64];
            std::snprintf(a, sizeof a, "%.4g", grid->th_min * (180.0 / M_PI));
            std::snprintf(b, sizeof b, "%.4g", tmin * (180.0 / M_PI));
            Raise::value(std::string("below/below grid tabulated from theta = ") + a +
                        " deg but was queried at " + b +
                        " deg: below the grazing floor the surfaces carry the lateral wave's "
                        "logarithmic structure (measured drift 1.1 to 3.3 of scale "
                        "between 2 and 0.05 deg at R1 = 3.5 in-medium wavelengths), "
                        "and theta = 0 has no node at all -- h = 0 leaves the tail "
                        "without its exponential decay. Raise the pair's depth sum "
                        "relative to its horizontal separation, or refuse the "
                        "geometry");
        }
    }

    // `_remainder_field_proj_below` for m observers (points, tangents), the
    // C++ branch: run, the past-cap extremes, the refusals, the lazy fill,
    // the re-run, the zeroing.
    void project(const double *pts, const double *tans, ptrdiff_t m, cd *out) {
        const ptrdiff_t S = static_cast<ptrdiff_t>(d_src.size());
        double dmin_o = std::numeric_limits<double>::infinity();
        for (ptrdiff_t i = 0; i < m; i++) dmin_o = std::min(dmin_o, gz - pts[3 * i + 2]);
        double dmin_s = std::numeric_limits<double>::infinity();
        for (double d : d_src) dmin_s = std::min(dmin_s, d);
        if (dmin_o <= 0.0 || dmin_s <= 0.0) {
            char buf[64];
            std::snprintf(buf, sizeof buf, "%.17g", gz);
            std::string gzr = buf;
            if (gzr.find_first_of(".eEn") == std::string::npos) gzr += ".0";
            Raise::value("the below/below remainder needs BOTH endpoints strictly below "
                        "ground_z = " + gzr + ": got min observer depth " + remainder_g6(dmin_o) +
                        ", min source depth " + remainder_g6(dmin_s) +
                        " (a non-positive depth is a point on or above the interface "
                        "\xe2\x80\x94 that geometry is momwire#524 phase 3)");
        }
        double mx_r1, mn_th, mx_th;
        auto run = [&]() {
            ::somm_core::below_field_proj_batch(view, 15, grid->th_min, grid->th_band_floor_hi,
                                              grid->th_band_lo_hi, grid->th_band_hi, m, S,
                                              pts, tans, src.data(),
                                              t_src.data(), gz, k_p, k_m, out, mx_r1, mn_th, mx_th,
                                              true, true);
        };
        // A lazy table (momwire#1512). The run reads first, as on an eager
        // table, and its extremes decide the regions exactly as there. A node
        // not yet filled reads NaN and NaN survives every operation of a
        // read, so when no region joined the table and every output that is
        // kept (all of them, or with the pairs past the cap zeroed, the
        // rest) is finite, every node the run read was in: that run is the
        // answer. Otherwise the census of this batch's reads names the nodes,
        // they are filled in the table's regions, and the batch runs again.
        const bool lazy = grid->lazy;
        auto all_finite = [&]() {
            for (ptrdiff_t i = 0; i < m * S; i++)
                if (!std::isfinite(out[i].real()) || !std::isfinite(out[i].imag())) return false;
            return true;
        };
        auto fill_and_run = [&](bool past_cap) {
            std::vector<std::vector<uint8_t>> kept_n(15), past_n(15);
            somm_proj::NodeMarks km, pm;
            km.n_reg = pm.n_reg = 15;
            for (int r = 0; r < 15; r++) {
                const Region &reg = grid->regions[r];
                kept_n[r].assign(static_cast<size_t>(reg.n_r * reg.n_th), 0);
                past_n[r].assign(static_cast<size_t>(reg.n_r * reg.n_th), 0);
                km.m[r] = kept_n[r].data();
                pm.m[r] = past_n[r].data();
                km.nR[r] = pm.nR[r] = static_cast<std::ptrdiff_t>(reg.n_r);
                km.nTh[r] = pm.nTh[r] = static_cast<std::ptrdiff_t>(reg.n_th);
            }
            double a, b, c;
            ::somm_core::below_field_proj_nodes(view, grid->th_min, grid->th_band_floor_hi,
                                              grid->th_band_lo_hi, grid->th_band_hi, m, S, pts,
                                              src.data(), gz, grid->r1_cap, km, pm, a, b, c);
            if (!past_cap)
                for (int r = 0; r < 15; r++)
                    for (size_t q = 0; q < kept_n[r].size(); q++) kept_n[r][q] |= past_n[r][q];
            source()->fill_nodes(*grid, kept_n);
            build_view();
            run();
            if (past_cap) zero_past_cap(pts, m, out);
        };
        std::vector<char> before(15);
        for (int r = 0; r < 15; r++) before[r] = grid->regions[r].filled;
        run();
        if (mx_r1 > grid->r1_cap && mn_th < grid->th_min) {
            double a, b, c;
            if (!served_extremes(pts, m, a, b, c)) {
                std::fill(out, out + m * S, cd(0.0, 0.0));
                return;
            }
            mn_th = b;
            mx_th = c;
        }
        if (m > 0 && S > 0) {
            const double q_r1 = std::min(mx_r1, grid->r1_cap);
            const double rr[2] = {q_r1, q_r1}, tt[2] = {mn_th, mx_th};
            refuse_out_of_domain(rr, tt, 2);
            double fr1 = q_r1, flo = mn_th, fhi = mx_th;
            const bool past_cap = mx_r1 > grid->r1_cap;
            if (past_cap && !served_extremes(pts, m, fr1, flo, fhi)) {
                std::fill(out, out + m * S, cd(0.0, 0.0));
                return;
            }
            source()->ensure_for(*grid, fr1, flo, fhi);
            bool changed = false;
            for (int r = 0; r < 15; r++) changed |= (grid->regions[r].filled != (before[r] != 0));
            if (lazy) {
                if (past_cap) zero_past_cap(pts, m, out);
                // The check mode always takes the census's way, so that it is
                // the census whose completeness the finite check proves.
                if (changed || !all_finite() || ::somm_core::lazy_check()) fill_and_run(past_cap);
            } else {
                if (changed) {
                    build_view();
                    run();
                }
                if (past_cap) zero_past_cap(pts, m, out);
            }
            if (past_cap) {
                if (!all_finite()) {
                    source()->ensure_for(*grid, q_r1, tt[0], tt[1]);
                    if (lazy) {
                        fill_and_run(true);
                    } else {
                        build_view();
                        run();
                        zero_past_cap(pts, m, out);
                    }
                }
            }
        }
    }
};

}  // namespace somm_below
