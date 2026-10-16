// razor-2p's buried fill in C++: see `_razor_buried_core.h`.
#include "_razor_buried_core.h"

#include "_razor_crossing_core.h"

#include "_somm_below_grid_core.h"
#include "_somm_below_fill_core.h"
#include "_somm_below_proj_core.h"
#include "_somm_grid_core.h"
#include "_somm_proj_core.h"

#include <algorithm>
#include <cstdio>
#include <set>
#include <stdexcept>

namespace mw {
namespace razor {

namespace {

// The knot each basis is centred on, read off its side-A wing
// (`_knot_points`): seg_p0[s_a] + (rise ? h : 0) * seg_t[s_a].
void knot_point(const Front &g, int64_t m, double *out) {
    const int64_t s = g.wing_seg[static_cast<size_t>(2 * m)];
    const double along = g.wing_rise.p[2 * m] ? g.seg_h[static_cast<size_t>(s)] : 0.0;
    for (int c = 0; c < 3; c++)
        out[c] = g.seg_p0[static_cast<size_t>(3 * s + c)] + along * g.seg_t[static_cast<size_t>(3 * s + c)];
}

constexpr const char *BELOW_TABLE_MISSING = "below-table";

// `_load_in_place`'s `np.add.at(Z, (rows, cols), vals)`: unbuffered, in the
// stencil's order, onto the column-major matrix.
void add_loading(const Front &f, std::vector<cd> &z) {
    const size_t n = static_cast<size_t>(f.n_basis);
    for (size_t i = 0; i < f.load_rows.size(); i++) {
        cd &e = z[static_cast<size_t>(f.load_rows[i]) + static_cast<size_t>(f.load_cols[i]) * n];
        e = cd(e.real() + f.load_vals[i].real(), e.imag() + f.load_vals[i].imag());
    }
}

// CPython's complex product (`_Py_c_prod`), unfused.
cd py_prod(cd a, cd b) {
    return cd(a.real() * b.real() - a.imag() * b.imag(), a.real() * b.imag() + a.imag() * b.real());
}

[[noreturn]] void raise_value(const std::string &m) {
    throw FrontError(FrontErrorKind::value_error, m, true);
}

std::string py_g6(double x) {
    char buf[64];
    std::snprintf(buf, sizeof buf, "%.6g", x);
    return buf;
}

// The below medium's numbers (`BelowMediumGround` at the fill's omega).
struct BelowMedium {
    cd eps_t, k_m, eps_m, a_m, c_a, c_phi;
};

BelowMedium below_medium(const Front &f) {
    BelowMedium m;
    // `_ground_refl.eps_tilde`: complex(eps_r, -sigma / (omega * eps0))
    m.eps_t = cd(f.ground_eps_r_, (-f.ground_sigma_) / (f.omega * EPS));
    m.k_m = somm_below::k_medium(m.eps_t, f.k);
    // `solver.eps * eps_t`, `(1.0 - eps_t) / (1.0 + eps_t)`, `1j * omega * mu`,
    // `1j * omega * eps_m`: CPython's complex arithmetic, a float promoted.
    m.eps_m = py_prod(cd(EPS, 0.0), m.eps_t);
    m.a_m = somm_core::py_quot(cd(1.0 - m.eps_t.real(), 0.0 - m.eps_t.imag()),
                               cd(1.0 + m.eps_t.real(), 0.0 + m.eps_t.imag()));
    const cd j_omega = py_prod(cd(0.0, 1.0), cd(f.omega, 0.0));
    m.c_a = py_prod(j_omega, cd(MU, 0.0));
    m.c_phi = py_prod(j_omega, m.eps_m);
    return m;
}

// `_potential_ground._r1_max_exact`: max hypot(hypot(dx, dy), d_o + d_s) x 1.001.
double r1_max_exact(const std::vector<double> &obs, const std::vector<double> &ends,
                    const std::vector<double> &d_o, const std::vector<double> &d_s,
                    const std::vector<size_t> *so = nullptr,
                    const std::vector<size_t> *se = nullptr) {
    const size_t no = so ? so->size() : d_o.size(), ne = se ? se->size() : d_s.size();
    double best = -std::numeric_limits<double>::infinity();
    bool nan = false;
    for (size_t a = 0; a < no; a++) {
        const size_t i = so ? (*so)[a] : a;
        for (size_t b = 0; b < ne; b++) {
            const size_t j = se ? (*se)[b] : b;
            const double rho = std::hypot(obs[3 * i] - ends[3 * j], obs[3 * i + 1] - ends[3 * j + 1]);
            const double v = std::hypot(rho, d_o[i] + d_s[j]);
            if (std::isnan(v)) nan = true;
            if (v > best) best = v;
        }
    }
    if (nan) return std::numeric_limits<double>::quiet_NaN();
    return best * 1.001;
}

// `r1_bracket_one_bucket(lo, hi, eps_t, k2)`.
bool r1_bracket_one_bucket(double lo, double hi, cd eps_t, double k2) {
    const double lam_m = somm_below::lambda_m(eps_t, k2);
    const double cap = somm_below::R1_CAP_LAMBDA_M;
    const double x_lo = std::min(lo / lam_m, cap) * (1.0 - 1e-9);
    const double x_hi = std::min(hi / lam_m, cap) * (1.0 + 1e-9);
    if (!(std::isfinite(x_lo) && std::isfinite(x_hi)) || x_lo > x_hi) return false;
    return somm_core::r1_bucket_wl(x_lo) == somm_core::r1_bucket_wl(x_hi);
}

// `_potential_ground._r1_max_below`: the bracket over 64 support directions
// when it proves one bucket (the grid reads the radius only through its
// bucket), else the exact walk. Which value is returned within a bucket
// moves no table, so the support values' rounding is not the numpy one.
double r1_max_below(const std::vector<double> &obs, const std::vector<double> &ends,
                    const std::vector<double> &d_o, const std::vector<double> &d_s, cd eps_t,
                    double k_p) {
    const size_t no = d_o.size(), ne = d_s.size();
    if (no == 0 || ne == 0) return r1_max_exact(obs, ends, d_o, d_s);
    constexpr int K = 64;
    std::vector<size_t> i_o(K), i_e(K);
    double span = -std::numeric_limits<double>::infinity();
    for (int q = 0; q < K; q++) {
        const double th = 2.0 * M_PI * q / K;
        const double ux = std::cos(th), uy = std::sin(th);
        double pmax = -std::numeric_limits<double>::infinity(), emin = std::numeric_limits<double>::infinity();
        for (size_t i = 0; i < no; i++) {
            const double v = obs[3 * i] * ux + obs[3 * i + 1] * uy;
            if (v > pmax) {
                pmax = v;
                i_o[q] = i;
            }
        }
        for (size_t j = 0; j < ne; j++) {
            const double v = ends[3 * j] * ux + ends[3 * j + 1] * uy;
            if (v < emin) {
                emin = v;
                i_e[q] = j;
            }
        }
        span = std::max(span, pmax - emin);
    }
    double scale = 1.0;
    for (size_t i = 0; i < no; i++) scale = std::max({scale, std::fabs(obs[3 * i]), std::fabs(obs[3 * i + 1])});
    for (size_t j = 0; j < ne; j++) scale = std::max({scale, std::fabs(ends[3 * j]), std::fabs(ends[3 * j + 1])});
    const double rho_hi = (std::max(span, 0.0) + 1e-9 * scale) / std::cos(M_PI / K);
    double mo = 0.0, ms = 0.0;
    size_t ao = 0, as = 0;
    for (size_t i = 0; i < no; i++) {
        mo = std::max(mo, std::fabs(d_o[i]));
        if (d_o[i] > d_o[ao]) ao = i;
    }
    for (size_t j = 0; j < ne; j++) {
        ms = std::max(ms, std::fabs(d_s[j]));
        if (d_s[j] > d_s[as]) as = j;
    }
    const double hi = std::hypot(rho_hi, mo + ms) * 1.001 * (1.0 + 1e-9);
    std::set<size_t> so_set(i_o.begin(), i_o.end()), se_set(i_e.begin(), i_e.end());
    so_set.insert(ao);
    se_set.insert(as);
    std::vector<size_t> so(so_set.begin(), so_set.end()), se(se_set.begin(), se_set.end());
    const double lo = r1_max_exact(obs, ends, d_o, d_s, &so, &se);
    if (std::isfinite(hi) && r1_bracket_one_bucket(lo, hi, eps_t, k_p)) return hi;
    return r1_max_exact(obs, ends, d_o, d_s);
}

// The below remainder of one medium front (`RemainderBelow.field_windows` at
// n_moment = 2, n_qp flat): the grid, the source nodes, the moment weights,
// and the projection wrapper `_remainder_field_proj_below` per window.
struct BelowRemainder {
    const Front *f = nullptr;
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
            const somm_below::Region &reg = grid->regions[r];
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
    bool served_extremes(ptrdiff_t o0, ptrdiff_t m, double &r1_hi_o, double &lo_o,
                         double &hi_o) const {
        double lo = std::numeric_limits<double>::infinity(), hi = -lo, r1_hi = -lo;
        const ptrdiff_t S = static_cast<ptrdiff_t>(d_src.size());
        for (ptrdiff_t i = 0; i < m; i++) {
            const double *o = obs + 3 * (o0 + i);
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
    void zero_past_cap(ptrdiff_t o0, ptrdiff_t m, cd *out) const {
        const ptrdiff_t S = static_cast<ptrdiff_t>(d_src.size());
        for (ptrdiff_t i = 0; i < m; i++) {
            const double *o = obs + 3 * (o0 + i);
            const double d_o = gz - o[2];
            for (ptrdiff_t j = 0; j < S; j++) {
                const double rho = std::hypot(o[0] - src[3 * j], o[1] - src[3 * j + 1]);
                const double hh = d_o + d_src[j];
                if (std::sqrt(rho * rho + hh * hh) > grid->r1_cap) out[i * S + j] = cd(0.0, 0.0);
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
        if (rmin < 0.0) raise_value("query R1 must be non-negative");
        if (rmax > grid->r1_max * (1.0 + 1e-9)) {
            char buf[128];
            std::snprintf(buf, sizeof buf, "%.3g", grid->r1_max / grid->lam_m);
            raise_value("below/below grid tabulated to r1_max = " + py_g6(grid->r1_max) + " (" + buf +
                        " in-medium wavelengths) but was queried at R1 = " + py_g6(rmax) +
                        ": past the tabulation the below/below remainder is NOT a negligible "
                        "tail (it grows relative to the direct term with range), so "
                        "there is no honest clamp \xe2\x80\x94 size the grid for the geometry "
                        "or refuse the geometry");
        }
        if (tmax > 0.5 * M_PI + 1e-12)
            raise_value("below/below grid theta must lie in [th_min, pi/2]; a theta "
                        "past pi/2 means the geometry is not two points below the "
                        "interface");
        if (tmin < grid->th_min * (1.0 - 1e-9)) {
            char a[64], b[64];
            std::snprintf(a, sizeof a, "%.4g", grid->th_min * (180.0 / M_PI));
            std::snprintf(b, sizeof b, "%.4g", tmin * (180.0 / M_PI));
            raise_value(std::string("below/below grid tabulated from theta = ") + a +
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

    // `_remainder_field_proj_below` for observers [o0, o0 + m), the C++
    // branch: run, the past-cap extremes, the refusals, the lazy fill, the
    // re-run, the zeroing.
    void project(ptrdiff_t o0, ptrdiff_t m, cd *out) {
        const ptrdiff_t S = static_cast<ptrdiff_t>(d_src.size());
        double dmin_o = std::numeric_limits<double>::infinity();
        for (ptrdiff_t i = 0; i < m; i++) dmin_o = std::min(dmin_o, gz - obs[3 * (o0 + i) + 2]);
        double dmin_s = std::numeric_limits<double>::infinity();
        for (double d : d_src) dmin_s = std::min(dmin_s, d);
        if (dmin_o <= 0.0 || dmin_s <= 0.0) {
            char buf[64];
            std::snprintf(buf, sizeof buf, "%.17g", gz);
            std::string gzr = buf;
            if (gzr.find_first_of(".eEn") == std::string::npos) gzr += ".0";
            raise_value("the below/below remainder needs BOTH endpoints strictly below "
                        "ground_z = " + gzr + ": got min observer depth " + py_g6(dmin_o) +
                        ", min source depth " + py_g6(dmin_s) +
                        " (a non-positive depth is a point on or above the interface "
                        "\xe2\x80\x94 that geometry is momwire#524 phase 3)");
        }
        double mx_r1, mn_th, mx_th;
        auto run = [&]() {
            somm_core::below_field_proj_batch(view, 15, grid->th_min, grid->th_band_floor_hi,
                                              grid->th_band_lo_hi, grid->th_band_hi, m, S,
                                              obs + 3 * o0, t_obs + 3 * o0, src.data(),
                                              t_src.data(), gz, k_p, k_m, out, mx_r1, mn_th, mx_th,
                                              true, true);
        };
        std::vector<char> before(15);
        for (int r = 0; r < 15; r++) before[r] = grid->regions[r].filled;
        run();
        if (mx_r1 > grid->r1_cap && mn_th < grid->th_min) {
            double a, b, c;
            if (!served_extremes(o0, m, a, b, c)) {
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
            if (past_cap && !served_extremes(o0, m, fr1, flo, fhi)) {
                std::fill(out, out + m * S, cd(0.0, 0.0));
                return;
            }
            somm_below::source()->ensure_for(*grid, fr1, flo, fhi);
            bool changed = false;
            for (int r = 0; r < 15; r++) changed |= (grid->regions[r].filled != (before[r] != 0));
            if (changed) {
                build_view();
                run();
            }
            if (past_cap) {
                zero_past_cap(o0, m, out);
                bool finite = true;
                for (ptrdiff_t i = 0; i < m * S && finite; i++)
                    finite = std::isfinite(out[i].real()) && std::isfinite(out[i].imag());
                if (!finite) {
                    somm_below::source()->ensure_for(*grid, q_r1, tt[0], tt[1]);
                    build_view();
                    run();
                    zero_past_cap(o0, m, out);
                }
            }
        }
    }
};

// The path points (each wing's centroid, `_testing_paths`' NEC-5 branch).
void prepare_paths(Front &s) {
    const size_t n = static_cast<size_t>(s.n_basis);
    s.obs_pts.resize(6 * n);
    for (size_t m = 0; m < n; m++)
        for (int c3 = 0; c3 < 3; c3++) {
            s.obs_pts[6 * m + c3] = s.cent[3 * static_cast<size_t>(s.s_a[m]) + c3];
            s.obs_pts[6 * m + 3 + c3] = s.cent[3 * static_cast<size_t>(s.s_b[m]) + c3];
        }
}

int64_t wire_of_seg(const Front &f, int64_t g) {
    int64_t w = 0;
    while (f.seg_offsets[static_cast<size_t>(w) + 1] <= g) w++;
    return w;
}

// `_below_plane_grazing_refusal(geom_b, plan_skip, k_m=k_m)`: "" or the
// sentence. The angles are libm's (Python's plan walk is the C++ kernel's
// libm atan; its rectangular walk is numpy's arctan2, CPU-dispatched), so a
// minimum within 1e-12 of the floor declines rather than decide; the printed
// numbers are 4 significant digits.
std::string grazing_refusal(const Front &s, const std::vector<double> &plan_skip, cd k_m) {
    const double gz = s.ground_z;
    const size_t ns = static_cast<size_t>(s.n_seg);
    std::vector<double> pts;
    pts.reserve(9 * ns);
    for (int part = 0; part < 3; part++)
        for (size_t g = 0; g < ns; g++)
            for (int c = 0; c < 3; c++) {
                const double p0 = s.seg_p0[3 * g + c], h = s.seg_h[g], t = s.seg_t[3 * g + c];
                pts.push_back(part == 0 ? p0 : part == 1 ? p0 + h * t : p0 + 0.5 * h * t);
            }
    if (!plan_skip.empty()) {
        std::vector<double> kept;
        for (size_t i = 0; i < pts.size() / 3; i++) {
            bool keep = true;
            for (size_t q = 0; q < plan_skip.size() / 3; q++) {
                const double d[3] = {pts[3 * i] - plan_skip[3 * q], pts[3 * i + 1] - plan_skip[3 * q + 1],
                                     pts[3 * i + 2] - plan_skip[3 * q + 2]};
                if (!(std::sqrt((d[0] * d[0] + d[1] * d[1]) + d[2] * d[2]) > JUNCTION_TOL)) keep = false;
            }
            if (keep) kept.insert(kept.end(), pts.begin() + 3 * i, pts.begin() + 3 * i + 3);
        }
        pts.swap(kept);
    }
    const size_t np_ = pts.size() / 3;
    std::vector<double> d(np_);
    double dmin = std::numeric_limits<double>::infinity();
    for (size_t i = 0; i < np_; i++) {
        d[i] = gz - pts[3 * i + 2];
        dmin = std::min(dmin, d[i]);
    }
    double depth = 2.0 * dmin;
    const double lam_m = 2.0 * M_PI / std::hypot(k_m.real(), k_m.imag());
    const double r1_cap = (somm_below::R1_CAP_LAMBDA_M * lam_m) * (1.0 + 1e-9);
    const double floor = 0.01666666666666667 * (M_PI / 180.0);
    // The plan's pair extents: the shallowest angle over pairs with rho > 0,
    // capped only when the all-pairs minimum is under the floor with a pair
    // past the cap (`_pair_extents_below`).
    auto plan_extents = [&](bool capped, double &r1_max, double &th) {
        double r1sq_max = 0.0, ratio_min = std::numeric_limits<double>::infinity();
        for (size_t i = 0; i < np_; i++)
            for (size_t j = 0; j < np_; j++) {
                const double dx = pts[3 * i] - pts[3 * j], dy = pts[3 * i + 1] - pts[3 * j + 1];
                const double rho2 = dx * dx + dy * dy;
                const double hh = d[i] + d[j];
                const double r1sq = rho2 + hh * hh;
                r1sq_max = std::max(r1sq_max, r1sq);
                const double rho = std::sqrt(rho2);
                if (!(rho > 0.0)) continue;
                if (capped && !(r1sq <= r1_cap * r1_cap)) continue;
                ratio_min = std::min(ratio_min, hh / rho);
            }
        r1_max = std::sqrt(r1sq_max);
        th = std::atan(ratio_min);
    };
    double r1_max, th_min;
    plan_extents(false, r1_max, th_min);
    if (r1_max > r1_cap && !(th_min >= floor)) plan_extents(true, r1_max, th_min);
    if (depth <= 0.0) th_min = std::min(th_min, std::atan2(depth, 0.0));
    // The pairs the below remainder evaluates: path points x the order
    // n_qp_sommerfeld nodes `seg_p0 + (tq h) t` (`_below_remainder_th_min`).
    std::vector<double> xq, wq;
    if (!leggauss(s.n_qp_sommerfeld, xq, wq)) throw std::runtime_error("grazing order");
    std::vector<double> src;
    for (size_t g = 0; g < ns; g++)
        for (double x : xq) {
            const double tq = 0.5 * (x + 1.0);
            for (int c = 0; c < 3; c++) src.push_back(s.seg_p0[3 * g + c] + (tq * s.seg_h[g]) * s.seg_t[3 * g + c]);
        }
    double best = std::numeric_limits<double>::infinity(), best_hh = 0.0, near = std::numeric_limits<double>::infinity();
    for (size_t i = 0; i < s.obs_pts.size() / 3; i++) {
        const double d_o = gz - s.obs_pts[3 * i + 2];
        for (size_t j = 0; j < src.size() / 3; j++) {
            const double rho = std::hypot(s.obs_pts[3 * i] - src[3 * j], s.obs_pts[3 * i + 1] - src[3 * j + 1]);
            const double hh = d_o + (gz - src[3 * j + 2]);
            double th = std::atan2(hh, rho);
            if (std::sqrt(rho * rho + hh * hh) > r1_cap) th = std::numeric_limits<double>::infinity();
            near = std::min(near, std::fabs(th - floor));
            if (th < best) {
                best = th;
                best_hh = hh;
            }
        }
    }
    if (best < th_min) {
        th_min = best;
        depth = best_hh;
    }
    if (std::fabs(th_min - floor) <= 1e-12 * floor || near <= 1e-12 * floor)
        throw std::runtime_error("grazing-edge");
    if (!(th_min < floor)) return "";
    char th_s[64], depth_s[64];
    std::snprintf(th_s, sizeof th_s, "%.4g", th_min * (180.0 / M_PI));
    std::snprintf(depth_s, sizeof depth_s, "%.4g", depth);
    return std::string("this deck's buried wires reach a below/below pair elevation of theta = ") + th_s +
           " deg among the pairs inside the cap, below the 0.0166667 deg grazing floor the "
           "below/below surfaces are tabulated from (the pair's two depths add to " + depth_s +
           " m, and theta = atan2(depth sum, horizontal separation)). The floor is asked only of "
           "pairs inside the cap \xe2\x80\x94 image distance R1 within 4 in-medium wavelengths, the "
           "pairs whose remainder is evaluated; past it the remainder is served as zero "
           "(momwire#1053, #1187). Below the floor the surfaces carry the lateral wave's "
           "LOGARITHMIC structure \xe2\x80\x94 measured drift 1.1 to 3.3 of scale between 2 and "
           "0.05 deg \xe2\x80\x94 which no uniform lattice resolves, and theta = 0 has no node at "
           "all because h = 0 leaves the tail without its exponential decay. Bury the wires "
           "deeper, shorten them, or wait for the log-spaced grazing band (momwire#553 U2's "
           "recorded follow-up)";
}

// `RemainderBelow.field_windows`' per-solve half on a below medium front:
// the path points (wing centroids) and wing signs the weighted rule and the
// remainder read, the source nodes and moment weights at n_qp_sommerfeld
// (flat: no grazing keying below), the grid radius and the grid.
void prepare_below(Front &s, const BelowMedium &bm, BelowRemainder &rem) {
    const size_t n = static_cast<size_t>(s.n_basis), ns = static_cast<size_t>(s.n_seg);
    s.eps_t = bm.eps_t;
    s.c2 = bm.a_m;  // the image coefficient through the windows: A_m
    s.obs_pts.resize(6 * n);
    s.sig_a.resize(n);
    s.sig_b.resize(n);
    for (size_t m = 0; m < n; m++) {
        for (int c3 = 0; c3 < 3; c3++) {
            s.obs_pts[6 * m + c3] = s.cent[3 * static_cast<size_t>(s.s_a[m]) + c3];
            s.obs_pts[6 * m + 3 + c3] = s.cent[3 * static_cast<size_t>(s.s_b[m]) + c3];
        }
        s.sig_a[m] = s.wing_sigma[2 * m];
        s.sig_b[m] = s.wing_sigma[2 * m + 1];
    }
    std::vector<double> src_r(3 * ns);
    for (size_t g = 0; g < ns; g++)
        for (int c3 = 0; c3 < 3; c3++) src_r[3 * g + c3] = s.seg_p0[3 * g + c3] + s.seg_h[g] * s.seg_t[3 * g + c3];
    // The grid radius over path points x source ENDPOINTS (`_r1_max_below`).
    const double gz = s.ground_z;
    std::vector<double> ends(6 * ns), d_o(2 * n), d_s(2 * ns);
    std::copy(s.seg_p0.begin(), s.seg_p0.end(), ends.begin());
    std::copy(src_r.begin(), src_r.end(), ends.begin() + 3 * ns);
    for (size_t i = 0; i < 2 * n; i++) d_o[i] = gz - s.obs_pts[3 * i + 2];
    for (size_t j = 0; j < 2 * ns; j++) d_s[j] = gz - ends[3 * j + 2];
    const double r1 = r1_max_below(s.obs_pts, ends, d_o, d_s, bm.eps_t, s.k);
    somm_below::Source *src = somm_below::source();
    if (src == nullptr) throw std::runtime_error(BELOW_TABLE_MISSING);
    rem.grid = &src->get(bm.eps_t, s.k, r1, s.omega, MU, somm_below::SPACING_FILL);
    rem.build_view();
    // `leggauss(n_qp)`, the nodes `seg_l + tq * span`, `W = w_node * u ** p`.
    const int64_t nq = s.n_qp_sommerfeld;
    std::vector<double> xq, wq;
    if (!leggauss(nq, xq, wq)) throw std::runtime_error("below remainder order");
    const size_t q = static_cast<size_t>(nq);
    s.n_qp_rem = nq;
    s.rem_src.resize(3 * ns * q);
    s.rem_t_src.resize(3 * ns * q);
    s.rem_W.resize(2 * ns * q);
    for (size_t g = 0; g < ns; g++) {
        double span[3];
        for (int c3 = 0; c3 < 3; c3++) span[c3] = src_r[3 * g + c3] - s.seg_p0[3 * g + c3];
        const double h = std::sqrt((span[0] * span[0] + span[1] * span[1]) + span[2] * span[2]);
        for (size_t j = 0; j < q; j++) {
            const double tq = 0.5 * (xq[j] + 1.0);
            for (int c3 = 0; c3 < 3; c3++) {
                s.rem_src[3 * (g * q + j) + c3] = s.seg_p0[3 * g + c3] + tq * span[c3];
                s.rem_t_src[3 * (g * q + j) + c3] = s.seg_t[3 * g + c3];
            }
            const double u = h * tq, w_node = 0.5 * h * wq[j];
            s.rem_W[g * q + j] = w_node * 1.0;
            s.rem_W[(ns + g) * q + j] = w_node * u;
        }
    }
    s.grid = nullptr;
    s.grid_view.reset();
    s.n_reg = 0;
    rem.f = &s;
    rem.gz = gz;
    rem.k_p = s.k;
    rem.k_m = bm.k_m;
    rem.src = s.rem_src;
    rem.t_src = s.rem_t_src;
    rem.d_src.resize(ns * q);
    for (size_t j = 0; j < ns * q; j++) rem.d_src[j] = gz - rem.src[3 * j + 2];
    rem.obs = s.obs_pts.data();
    rem.t_obs = s.tans.data();
}

// One medium front's block into a fresh context, chop and (below) the lossy
// medium wired in.
const cd *fill_part(const Front &s, SolveContext &ctx, uintptr_t cancel_flag,
                    const BelowMedium *bm, BelowRemainder *rem) {
    SolveInputs in = s.solve_inputs(cancel_flag);
    const ptrdiff_t nc = static_cast<ptrdiff_t>(s.chop_rows.size());
    if (nc) {
        in.chop_rows = view<const int64_t, 1>::contiguous(s.chop_rows.data(), {nc});
        in.chop_keep_a = view<const bool, 1>::contiguous(s.chop_keep_a.data(), {nc});
        in.chop_pts = view<const double, 2>::contiguous(s.chop_pts.data(), {nc, 3});
    }
    if (bm != nullptr) {
        in.k_complex = true;
        in.k_c = bm->k_m;
        in.unfused = true;
        in.c_a = bm->c_a;
        in.c_phi = bm->c_phi;
        in.comp.proj = [rem](ptrdiff_t o0, ptrdiff_t m, cd *out) { rem->project(o0, m, out); };
    }
    ctx.fill(in);
    return ctx.z();
}

}  // namespace

MediumPart medium_part(const Front &f, bool below) {
    MediumPart mp;
    mp.front.reset(new Front());
    Front &s = *mp.front;
    const auto side_of = [&](int64_t w) { return f.media_below[static_cast<size_t>(w)] != 0; };

    // The medium's wires, segments and owned bases, in wire order.
    std::vector<int64_t> seg_i, wires;
    std::vector<int64_t> remap(static_cast<size_t>(f.n_seg), -1);
    for (int64_t w = 0; w < f.n_wires; w++) {
        if (side_of(w) != below) continue;
        wires.push_back(w);
        for (int64_t g = f.seg_offsets[static_cast<size_t>(w)]; g < f.seg_offsets[static_cast<size_t>(w) + 1]; g++) {
            remap[static_cast<size_t>(g)] = static_cast<int64_t>(seg_i.size());
            seg_i.push_back(g);
        }
    }
    std::vector<int64_t> &rows = mp.rows;
    for (int64_t w : wires)
        for (int64_t m = f.basis_offsets[static_cast<size_t>(w)]; m < f.basis_offsets[static_cast<size_t>(w) + 1]; m++)
            rows.push_back(m);
    for (int64_t m = f.basis_offsets.back(); m < f.n_basis; m++)
        for (int j = 0; j < 2; j++)
            if (remap[static_cast<size_t>(f.wing_seg[static_cast<size_t>(2 * m + j)])] >= 0) {
                rows.push_back(m);
                break;
            }

    // The scalar state the fill reads: the solver's, unchanged.
    s.has_ground = true;
    s.ground_z = f.ground_z;
    s.c = f.c;
    s.k = f.k;
    s.omega = f.omega;
    s.c_a_imag = f.c_a_imag;
    s.c_phi_imag = f.c_phi_imag;
    s.extended_kernel = false;
    s.exact = false;
    s.compose = f.compose;
    s.ground_eps_r_ = f.ground_eps_r_;
    s.ground_sigma_ = f.ground_sigma_;
    s.n_qp_sommerfeld = f.n_qp_sommerfeld;
    s.xg = f.xg;
    s.wg = f.wg;
    // The kernel's radius: the scalar on a uniform deck, else each segment
    // its wire's (`seg_a`), which the medium's wires spell by themselves.
    s.n_wires = static_cast<int64_t>(wires.size());
    s.uniform_radius = f.uniform_radius;
    s.radius_value = f.radius_value;
    s.seg_offsets.assign(1, 0);
    for (int64_t w : wires) {
        s.radius_per_wire.push_back(f.radius_per_wire[static_cast<size_t>(w)]);
        s.seg_offsets.push_back(s.seg_offsets.back() + f.seg_offsets[static_cast<size_t>(w) + 1] -
                                f.seg_offsets[static_cast<size_t>(w)]);
    }
    s.n_seg = static_cast<int64_t>(seg_i.size());
    for (int64_t g : seg_i) {
        for (int c = 0; c < 3; c++) {
            s.seg_p0.push_back(f.seg_p0[static_cast<size_t>(3 * g + c)]);
            s.seg_t.push_back(f.seg_t[static_cast<size_t>(3 * g + c)]);
        }
        s.seg_h.push_back(f.seg_h[static_cast<size_t>(g)]);
    }
    // The wings: a wing on another medium's segment becomes a ghost of the
    // live one (its segment and rise) at sigma = 0, and the row is chopped
    // ("A": wing A survives).
    const int64_t n = static_cast<int64_t>(rows.size());
    s.n_basis = n;
    s.wing_seg.assign(static_cast<size_t>(2 * n), 0);
    s.wing_rise.assign(static_cast<size_t>(2 * n), false);
    s.wing_sigma.assign(static_cast<size_t>(2 * n), 0.0);
    std::vector<std::pair<int64_t, bool>> chop;
    for (int64_t i = 0; i < n; i++) {
        const int64_t m = rows[static_cast<size_t>(i)];
        bool side[2];
        for (int j = 0; j < 2; j++) side[j] = remap[static_cast<size_t>(f.wing_seg[static_cast<size_t>(2 * m + j)])] >= 0;
        for (int j = 0; j < 2; j++) {
            const int src = side[j] ? j : 1 - j;
            s.wing_seg[static_cast<size_t>(2 * i + j)] =
                remap[static_cast<size_t>(f.wing_seg[static_cast<size_t>(2 * m + src)])];
            s.wing_rise.p[2 * i + j] = f.wing_rise.p[2 * m + src];
            s.wing_sigma[static_cast<size_t>(2 * i + j)] = side[j] ? f.wing_sigma[static_cast<size_t>(2 * m + j)] : 0.0;
        }
        if (!(side[0] && side[1])) chop.emplace_back(i, side[0]);
    }
    s.grounded_bases.clear();
    s.prepare();
    // `t2_chop`: the chopped rows sorted, which half survives, and each
    // one's knot on the SUB-geometry (`_knot_points(geom_side)`).
    std::sort(chop.begin(), chop.end());
    s.chop_rows.clear();
    s.chop_pts.clear();
    s.chop_keep_a.assign(chop.size(), false);
    for (size_t q = 0; q < chop.size(); q++) {
        s.chop_rows.push_back(chop[q].first);
        s.chop_keep_a.p[q] = chop[q].second;
        double kp[3];
        knot_point(s, chop[q].first, kp);
        s.chop_pts.insert(s.chop_pts.end(), kp, kp + 3);
    }
    // `_schedule.elem_rows(_WEIGHTED_CHUNK_ELEMS, n_path * n_basis)`: the
    // window size moves no bit.
    const int64_t budget = f.chunk_elems / 12;
    s.window_rows = std::max<int64_t>(1, budget / std::max<int64_t>(1, 2 * n));
    s.chunk_elems = f.chunk_elems;
    if (!below && s.compose) {
        const std::string why = s.prepare_compose(s.n_qp_sommerfeld);
        if (!why.empty()) throw std::runtime_error("buried above medium: " + why);
    }
    return mp;
}

static std::string fill_buried_inner(const Front &f, std::vector<cd> &out, BuriedStages *stages,
                                     uintptr_t cancel_flag) {
    if (f.buried == Buried::none) throw std::invalid_argument("fill_buried: not a buried deck");
    const int64_t n = f.n_basis;
    const auto record = [&](const char *name) {
        if (stages == nullptr) return;
        stages->reached.push_back(name);
        stages->z.push_back(out);
    };
    const auto scatter = [&](const MediumPart &p, const cd *zp) {
        // `_ix_accumulate(Z, rows, rows, block)`: one IEEE add per entry.
        const int64_t np_ = p.front->n_basis;
        for (int64_t j = 0; j < np_; j++) {
            const int64_t cj = p.rows[static_cast<size_t>(j)];
            for (int64_t i = 0; i < np_; i++) {
                const int64_t ri = p.rows[static_cast<size_t>(i)];
                cd &e = out[static_cast<size_t>(cj * n + ri)];
                const cd v = zp[static_cast<size_t>(j * np_ + i)];
                e = cd(e.real() + v.real(), e.imag() + v.imag());
            }
        }
    };
    // The below table is B5's; until then a source must be installed (the
    // test bridge) or the fill stops before the below medium.
    if (somm_below::source() == nullptr && f.buried == Buried::below_plane) return BELOW_TABLE_MISSING;
    const BelowMedium bm = below_medium(f);
    if (f.buried == Buried::below_plane) {
        // `_assemble_Z_below_plane` on the whole geometry, then the loading.
        MediumPart b = medium_part(f, true);
        prepare_paths(*b.front);
        const std::string why = grazing_refusal(*b.front, {}, bm.k_m);
        if (!why.empty()) raise_value(why);
        BelowRemainder rem;
        prepare_below(*b.front, bm, rem);
        SolveContext ctx;
        const cd *zb = fill_part(*b.front, ctx, cancel_flag, &bm, &rem);
        out.assign(zb, zb + static_cast<size_t>(n * n));
        if (!f.load_rows.empty()) add_loading(f, out);
        record("final");
        return "";
    }

    // 1-2. The above medium's block, scattered into the zero target.
    out.assign(static_cast<size_t>(n * n), cd(0.0, 0.0));
    {
        MediumPart a = medium_part(f, false);
        SolveContext ctx;
        scatter(a, fill_part(*a.front, ctx, cancel_flag, nullptr, nullptr));
    }
    record("above");
    if (somm_below::source() == nullptr) return BELOW_TABLE_MISSING;
    // 3. The below medium's block, folded (`d_rows -= i_rows; Z += d_rows`).
    {
        MediumPart b = medium_part(f, true);
        prepare_paths(*b.front);
        // The crossing nodes the plan skips: the crossing tents' knots on
        // the FULL geometry (`_knot_points(geom)[tents]`).
        std::vector<double> skip;
        for (int64_t m = 0; m < f.n_basis; m++) {
            const bool a = f.media_below[static_cast<size_t>(wire_of_seg(f, f.wing_seg[static_cast<size_t>(2 * m)]))] != 0;
            const bool b2 = f.media_below[static_cast<size_t>(wire_of_seg(f, f.wing_seg[static_cast<size_t>(2 * m + 1)]))] != 0;
            if (a == b2) continue;
            double kp[3];
            knot_point(f, m, kp);
            skip.insert(skip.end(), kp, kp + 3);
        }
        const std::string why = grazing_refusal(*b.front, skip, bm.k_m);
        if (!why.empty()) raise_value(why);
        BelowRemainder rem;
        prepare_below(*b.front, bm, rem);
        SolveContext ctx;
        scatter(b, fill_part(*b.front, ctx, cancel_flag, &bm, &rem));
    }
    record("below");
    // 4-5. The cross blocks and the node charges.
    std::vector<std::pair<int64_t, int>> tents;
    for (int64_t m = 0; m < n; m++) {
        const bool b0 = f.media_below[static_cast<size_t>(wire_of_seg(f, f.wing_seg[static_cast<size_t>(2 * m)]))] != 0;
        const bool b1 = f.media_below[static_cast<size_t>(wire_of_seg(f, f.wing_seg[static_cast<size_t>(2 * m + 1)]))] != 0;
        if (b0 != b1) tents.emplace_back(m, b0 ? 0 : 1);
    }
    crossing_blocks(f, out, tents, cancel_flag);
    record("cross");
    // 6. The loading, once, on the full geometry.
    if (!f.load_rows.empty()) add_loading(f, out);
    record("final");
    return "";
}

// `fill_buried_inner`, with the below table's `_refuse_if_capped` raised as
// the ValueError Python raises from the fill.
std::string fill_buried(const Front &f, std::vector<cd> &out, BuriedStages *stages,
                        uintptr_t cancel_flag) {
    try {
        return fill_buried_inner(f, out, stages, cancel_flag);
    } catch (const somm_below::TailCapped &e) {
        throw FrontError(FrontErrorKind::value_error, e.what(), true);
    }
}

}  // namespace razor
}  // namespace mw
