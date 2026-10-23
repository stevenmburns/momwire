// The near field across and below a finite ground, Python-free
// (momwire#1653); see the header. Each function is the `_near_buried.py`
// one it names, step for step; the arithmetic is this file's own (the numpy
// spelling is the reference, gated at a tolerance), so the Python route and
// the executables read the same bits by calling the same code.
#define _USE_MATH_DEFINES

#include "_near_buried_core.h"

#include "_near_interface_eval.h"
#include "_somm_below_fill_core.h"
#include "_somm_below_grid_core.h"
#include "_somm_below_remainder_inline.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>

namespace near_buried {

namespace {

constexpr double PI = 3.141592653589793;
constexpr double ETA0 = 376.730313668;
// `_sommerfeld_below._SOMM_BELOW_SPACING_POINT`.
constexpr double SPACING_POINT[3] = {1.0, 1.0, 1.0};
// Pairs per kernel batch: a memory choice (each batch its own memo).
constexpr size_t BATCH_PAIRS = size_t(1) << 18;

struct Raise {
    [[noreturn]] static void value(const std::string &m) { throw std::invalid_argument(m); }
};

cd c1_moment(double omega, double mu) { return cd(0.0, -omega * mu / (4.0 * PI)); }

// The pairs' observer-relative geometry and the kernels at them, (point,
// source) pairs [q0, q1) of an n x S grid: D/rho_eff and the two families.
struct Kern {
    std::vector<double> gx, gy;
    std::vector<ni_eval::Six> six;
    std::vector<cd> pk;  // (m, 4): dRhoV, dzV, dRhoW, dz'V
};

void kernels(const double *pts, const double *src, size_t S, size_t q0, size_t q1, double gz, double a,
             bool above, const ni_eval::Evaluator &ev, const ni_eval::SheetPlan &plan, bool want_six,
             Kern &K) {
    const size_t m = q1 - q0;
    K.gx.resize(m);
    K.gy.resize(m);
    std::vector<double> rows(3 * m), rho_e(m), zz(m), zq(m);
    for (size_t q = 0; q < m; q++) {
        const size_t i = (q0 + q) / S, j = (q0 + q) % S;
        const double *o = pts + 3 * i, *s = src + 3 * j;
        const double dx = o[0] - s[0], dy = o[1] - s[1];
        const double rho = std::hypot(dx, dy);
        const double re = std::hypot(rho, a);
        const double z_o = o[2] - gz, z_s = s[2] - gz;
        const double z = std::max(above ? z_o : z_s, 0.0), zp = std::min(above ? z_s : z_o, 0.0);
        K.gx[q] = dx / re;
        K.gy[q] = dy / re;
        rows[3 * q] = rho_e[q] = re;
        rows[3 * q + 1] = zz[q] = z;
        rows[3 * q + 2] = zq[q] = zp;
    }
    if (want_six) {
        ni_eval::Memo memo;
        memo.plan = plan;
        memo.have_plan = true;
        K.six.resize(m);
        ni_eval::designed_tables(ev, &memo, rho_e.data(), zz.data(), zq.data(), m, nullptr, K.six.data());
    }
    K.pk.resize(4 * m);
    ni_eval::point_tables(ev, &plan, rows.data(), m, K.pk.data());
}

// sum over the elements of the moment part of E / c1: the curl-carrying part
// (and, when `grad` is given, the -p_z grad W gradient), (n, 3) each.
void moment_terms(const double *pts, size_t n, const Sources &s, double gz, double k, double a, bool above,
                  const ni_eval::Evaluator &ev, const ni_eval::SheetPlan &plan, cd *curl, cd *grad) {
    const size_t S = s.n_el();
    std::fill(curl, curl + 3 * n, cd(0.0, 0.0));
    if (grad) std::fill(grad, grad + 3 * n, cd(0.0, 0.0));
    if (S == 0 || n == 0) return;
    const double k2 = k * k;
    const size_t total = n * S;
    Kern K;
    for (size_t q0 = 0; q0 < total; q0 += BATCH_PAIRS) {
        const size_t q1 = std::min(total, q0 + BATCH_PAIRS);
        kernels(pts, s.mid.data(), S, q0, q1, gz, a, above, ev, plan, true, K);
        for (size_t q = q0; q < q1; q++) {
            const size_t i = q / S, j = q % S, r = q - q0;
            const cd *p = &s.moment[3 * j];
            const ni_eval::Six &t = K.six[r];
            const cd U = t[0], V = t[1], dzW = t[3], dzpW = t[5];
            const cd gRhoW = K.pk[4 * r + 2];
            const double gx = K.gx[r], gy = K.gy[r];
            cd z_ng = k2 * V * p[2] + (p[0] * gx + p[1] * gy) * gRhoW;
            // -p_z (D gRhoW + z dzW) is -p_z grad_o W + p_z z (dzpW - dzW)
            // when the observer is the below point (its z-derivative is d/dz').
            if (!above) z_ng += p[2] * (dzpW - dzW);
            curl[3 * i] += U * p[0];
            curl[3 * i + 1] += U * p[1];
            curl[3 * i + 2] += z_ng;
            if (grad) {
                const cd dz_obs = above ? dzW : dzpW;
                grad[3 * i] += -p[2] * gx * gRhoW;
                grad[3 * i + 1] += -p[2] * gy * gRhoW;
                grad[3 * i + 2] += -p[2] * dz_obs;
            }
        }
    }
}

// -sum over the knots of dI grad_o V, (n, 3) over c1.
void charge_term(const double *pts, size_t n, const Sources &s, double gz, double a, bool above,
                 const ni_eval::Evaluator &ev, const ni_eval::SheetPlan &plan, cd *out) {
    const size_t S = s.n_nodes();
    std::fill(out, out + 3 * n, cd(0.0, 0.0));
    if (S == 0 || n == 0) return;
    const size_t total = n * S;
    Kern K;
    for (size_t q0 = 0; q0 < total; q0 += BATCH_PAIRS) {
        const size_t q1 = std::min(total, q0 + BATCH_PAIRS);
        kernels(pts, s.nodes.data(), S, q0, q1, gz, a, above, ev, plan, false, K);
        for (size_t q = q0; q < q1; q++) {
            const size_t i = q / S, j = q % S, r = q - q0;
            const cd qv = -s.delta[j];
            const cd gRhoV = K.pk[4 * r];
            const cd dv = above ? K.pk[4 * r + 1] : K.pk[4 * r + 3];
            out[3 * i] += qv * K.gx[r] * gRhoV;
            out[3 * i + 1] += qv * K.gy[r] * gRhoV;
            out[3 * i + 2] += qv * dv;
        }
    }
}

// `_curl(field, points, step, ground_z, above)`: field(pts, centres, m, out).
template <class F>
void curl(F field, const double *pts, size_t n, double step, double gz, bool above, cd *out) {
    std::vector<char> one(n);
    for (size_t i = 0; i < n; i++) {
        const double h = pts[3 * i + 2] - gz;
        one[i] = above ? (h < step) : (-h < step);
    }
    const double away = above ? 1.0 : -1.0;
    std::vector<double> stacks, centres;
    for (int axis = 0; axis < 3; axis++)
        for (double sign : {1.0, -1.0})
            for (size_t i = 0; i < n; i++) {
                for (int c = 0; c < 3; c++) {
                    double v = pts[3 * i + c] + (c == axis ? sign * step : 0.0);
                    if (axis == 2 && one[i]) v = pts[3 * i + c];
                    stacks.push_back(v);
                    centres.push_back(pts[3 * i + c]);
                }
            }
    std::vector<size_t> side;
    for (size_t i = 0; i < n; i++)
        if (one[i]) side.push_back(i);
    for (double m : {0.0, 1.0, 2.0})
        for (size_t i : side)
            for (int c = 0; c < 3; c++) {
                stacks.push_back(pts[3 * i + c] + (c == 2 ? m * away * step : 0.0));
                centres.push_back(pts[3 * i + c]);
            }
    const size_t total = stacks.size() / 3;
    std::vector<cd> v(3 * total);
    field(stacks.data(), centres.data(), total, v.data());
    const auto at = [&](size_t stack, size_t i, int c) { return v[3 * (stack * n + i) + c]; };
    std::vector<cd> d(9 * n);  // d[axis][i][c]
    for (int a = 0; a < 3; a++)
        for (size_t i = 0; i < n; i++)
            for (int c = 0; c < 3; c++)
                d[(a * n + i) * 3 + c] = (at(2 * a, i, c) - at(2 * a + 1, i, c)) / (2.0 * step);
    const size_t m1 = side.size();
    for (size_t r = 0; r < m1; r++) {
        const size_t i = side[r];
        for (int c = 0; c < 3; c++) {
            const cd w0 = v[3 * (6 * n + r) + c], w1 = v[3 * (6 * n + m1 + r) + c],
                     w2 = v[3 * (6 * n + 2 * m1 + r) + c];
            d[(2 * n + i) * 3 + c] = away * (-3.0 * w0 + 4.0 * w1 - w2) / (2.0 * step);
        }
    }
    for (size_t i = 0; i < n; i++) {
        const auto D = [&](int a, int c) { return d[(a * n + i) * 3 + c]; };
        out[3 * i] = D(1, 2) - D(2, 1);
        out[3 * i + 1] = D(2, 0) - D(0, 2);
        out[3 * i + 2] = D(0, 1) - D(1, 0);
    }
}

// `_plan`: one readout's sheet plan over every (observer, source) row.
ni_eval::SheetPlan plan_for(const double *pts, size_t n, const Sources &s, cd k_m, double gz, double k,
                            double a, bool above, double step, bool magnetic) {
    std::vector<double> obs;
    const auto push = [&](const double *p, double dx, double dy, double dz) {
        const double q[3] = {p[0] + dx, p[1] + dy, p[2] + dz};
        if (above ? q[2] >= gz : q[2] <= gz) {
            obs.push_back(q[0]);
            obs.push_back(q[1]);
            obs.push_back(q[2] - gz);
        }
    };
    for (size_t i = 0; i < n; i++) push(pts + 3 * i, 0, 0, 0);
    if (magnetic)
        for (int ax = 0; ax < 3; ax++)
            for (double m : {-2.0, -1.0, 1.0, 2.0})
                for (size_t i = 0; i < n; i++)
                    push(pts + 3 * i, ax == 0 ? m * step : 0.0, ax == 1 ? m * step : 0.0, ax == 2 ? m * step : 0.0);
    std::vector<double> src(s.mid);
    src.insert(src.end(), s.nodes.begin(), s.nodes.end());
    for (size_t j = 0; j < src.size() / 3; j++) src[3 * j + 2] -= gz;
    const size_t no = obs.size() / 3, ns = src.size() / 3;
    const ni_eval::NodePair pair = above ? ni_eval::NodePair{obs.data(), no, src.data(), ns}
                                         : ni_eval::NodePair{src.data(), ns, obs.data(), no};
    return ni_eval::sheet_plan(k, k_m, {pair}, a);
}

// `medium_element_fields`: `_element_fields` in the lower medium.
void medium_fields(const double *pts, size_t n, const Sources &s, bool image, double gz, double k, cd eps_t,
                   cd k_m, double radius, bool magnetic, cd *out) {
    const size_t S = s.n_el(), Q = s.n_nodes();
    const auto src_pt = [&](const double *p, double o[3]) {
        o[0] = p[0];
        o[1] = p[1];
        o[2] = image ? 2.0 * gz - p[2] : p[2];
    };
    const auto geometry = [&](const double *p, const double *q, double rh[3], double &r, cd &green) {
        double rv[3];
        for (int c = 0; c < 3; c++) rv[c] = p[c] - q[c];
        r = std::sqrt(rv[0] * rv[0] + rv[1] * rv[1] + rv[2] * rv[2] + radius * radius);
        for (int c = 0; c < 3; c++) rh[c] = rv[c] / r;
        green = std::exp(cd(0.0, -1.0) * k_m * r) / (4.0 * PI * r);
    };
    const double flip[3] = {image ? -1.0 : 1.0, image ? -1.0 : 1.0, 1.0};
    for (size_t i = 0; i < n; i++) {
        cd acc[3] = {cd(0, 0), cd(0, 0), cd(0, 0)};
        const double *p = pts + 3 * i;
        if (magnetic) {
            for (size_t j = 0; j < S; j++) {
                double q[3], rh[3], r;
                cd g;
                src_pt(&s.mid[3 * j], q);
                geometry(p, q, rh, r, g);
                cd m[3];
                for (int c = 0; c < 3; c++) m[c] = s.moment[3 * j + c] * flip[c];
                const cd w = (cd(0.0, 1.0) * k_m + 1.0 / r) * g;
                acc[0] += w * (m[1] * rh[2] - m[2] * rh[1]);
                acc[1] += w * (m[2] * rh[0] - m[0] * rh[2]);
                acc[2] += w * (m[0] * rh[1] - m[1] * rh[0]);
            }
        } else {
            const cd pre = cd(0.0, -1.0) * ETA0 * k;
            for (size_t j = 0; j < S; j++) {
                double q[3], rh[3], r;
                cd g;
                src_pt(&s.mid[3 * j], q);
                geometry(p, q, rh, r, g);
                for (int c = 0; c < 3; c++) acc[c] += pre * g * (s.moment[3 * j + c] * flip[c]);
            }
            const cd pre_q = cd(0.0, -1.0) * ETA0 / (k * eps_t);
            for (size_t j = 0; j < Q; j++) {
                double q[3], rh[3], r;
                cd g;
                src_pt(&s.nodes[3 * j], q);
                geometry(p, q, rh, r, g);
                const cd dl = image ? -s.delta[j] : s.delta[j];
                const cd sc = pre_q * (dl * (1.0 + cd(0.0, 1.0) * k_m * r) / r * g);
                for (int c = 0; c < 3; c++) acc[c] += sc * rh[c];
            }
        }
        for (int c = 0; c < 3; c++) out[3 * i + c] = acc[c];
    }
}

double depth_sum_r1(const double *o, const double *s, double gz) {
    const double rho = std::hypot(o[0] - s[0], o[1] - s[1]);
    const double h = (gz - o[2]) + (gz - s[2]);
    return std::sqrt(rho * rho + h * h);
}

// The far route's surfaces combined into a field (`remainder_field_below`
// over a `_DirectBelow`): pairs of one point against (m, p) sources.
void below_direct(const double *pt, const double *m, const cd *p, size_t S, cd eps_t, double gz, double k,
                  cd k_m, double omega, double mu, cd *out) {
    std::vector<double> r1(S), th(S), rho(S), hh(S), dhx(S), dhy(S);
    double r1_max = 0.0;
    for (size_t j = 0; j < S; j++) {
        const double dx = pt[0] - m[3 * j], dy = pt[1] - m[3 * j + 1];
        rho[j] = std::hypot(dx, dy);
        hh[j] = (gz - pt[2]) + (gz - m[3 * j + 2]);
        r1[j] = std::sqrt(rho[j] * rho[j] + hh[j] * hh[j]);
        th[j] = std::atan2(hh[j], rho[j]);
        r1_max = std::max(r1_max, r1[j]);
        dhx[j] = dx;
        dhy[j] = dy;
    }
    const double tiny = 1e-12 * (1.001 * r1_max);
    for (size_t j = 0; j < S; j++) {
        if (rho[j] > tiny) {
            dhx[j] /= rho[j];
            dhy[j] /= rho[j];
        } else {
            dhx[j] = 1.0;
            dhy[j] = 0.0;
        }
    }
    std::vector<cd> surf(4 * S);
    somm_below::iv_surfaces(eps_t, k, omega, mu, r1.data(), th.data(), S, surf.data(), 0);
    cd acc[3] = {cd(0, 0), cd(0, 0), cd(0, 0)};
    for (size_t j = 0; j < S; j++) {
        // `_SURF_KEYS` order: IrhoV, IzV, IrhoH, IphiH
        const cd IrhoV = surf[j], IzV = surf[S + j], IrhoH = surf[2 * S + j], IphiH = surf[3 * S + j];
        const cd g = std::exp(cd(0.0, -1.0) * (k * rho[j] + k_m * hh[j])) / r1[j];
        const cd *q = p + 3 * j;
        const cd cph = q[0] * dhx[j] + q[1] * dhy[j];
        const cd sph = q[0] * dhy[j] - q[1] * dhx[j];
        const cd e_rho = g * (q[2] * IrhoV + cph * IrhoH);
        const cd e_phi = g * sph * IphiH;
        const cd e_z = g * (q[2] * IzV - cph * IrhoV);
        acc[0] += dhx[j] * e_rho - dhy[j] * e_phi;
        acc[1] += dhy[j] * e_rho + dhx[j] * e_phi;
        acc[2] += e_z;
    }
    for (int c = 0; c < 3; c++) out[c] = acc[c];
}

// `_below_remainder`: the grid where every source pair is inside its
// tabulation (decided at the stencil centre with the stencil's reach as
// margin), the direct contour elsewhere.
void below_remainder(const double *pts, const double *centre, size_t n, const Sources &s, cd eps_t, double gz,
                     double k, cd k_m, double omega, double mu, somm_below::Grid *grid, cd *out) {
    const size_t S = s.n_el();
    std::fill(out, out + 3 * n, cd(0.0, 0.0));
    if (S == 0 || n == 0) return;
    double reach = 0.0;
    for (size_t i = 0; i < n; i++) {
        double d2 = 0.0;
        for (int c = 0; c < 3; c++) d2 += (pts[3 * i + c] - centre[3 * i + c]) * (pts[3 * i + c] - centre[3 * i + c]);
        reach = std::max(reach, std::sqrt(d2));
    }
    std::vector<size_t> near_i, far_i;
    for (size_t i = 0; i < n; i++) {
        double mx = 0.0;
        for (size_t j = 0; j < S; j++) mx = std::max(mx, depth_sum_r1(centre + 3 * i, &s.mid[3 * j], gz));
        if (grid == nullptr || mx + 2.0 * reach > grid->r1_max)
            far_i.push_back(i);
        else
            near_i.push_back(i);
    }
    if (!near_i.empty()) {
        somm_below::Remainder<Raise> rem;
        rem.gz = gz;
        rem.k_p = k;
        rem.k_m = k_m;
        rem.grid = grid;
        rem.build_view();
        rem.src = s.mid;
        rem.t_src.resize(3 * S);
        std::vector<cd> w(S);
        for (size_t j = 0; j < S; j++) {
            double t[3], len = 0.0;
            for (int c = 0; c < 3; c++) {
                t[c] = s.end[3 * j + c] - s.start[3 * j + c];
                len += t[c] * t[c];
            }
            len = std::sqrt(len);
            for (int c = 0; c < 3; c++) rem.t_src[3 * j + c] = t[c] / len;
            w[j] = s.moment[3 * j] * rem.t_src[3 * j] + s.moment[3 * j + 1] * rem.t_src[3 * j + 1] +
                   s.moment[3 * j + 2] * rem.t_src[3 * j + 2];
        }
        rem.d_src.resize(S);
        for (size_t j = 0; j < S; j++) rem.d_src[j] = gz - rem.src[3 * j + 2];
        const size_t M = 3 * near_i.size();
        std::vector<double> obs(3 * M), tans(3 * M, 0.0);
        for (size_t r = 0; r < near_i.size(); r++)
            for (int c = 0; c < 3; c++) {
                for (int e = 0; e < 3; e++) obs[3 * (3 * r + c) + e] = pts[3 * near_i[r] + e];
                tans[3 * (3 * r + c) + c] = 1.0;
            }
        std::vector<cd> proj(M * S);
        rem.project(obs.data(), tans.data(), static_cast<std::ptrdiff_t>(M), proj.data());
        for (size_t r = 0; r < near_i.size(); r++)
            for (int c = 0; c < 3; c++) {
                cd acc(0.0, 0.0);
                for (size_t j = 0; j < S; j++) acc += proj[(3 * r + c) * S + j] * w[j];
                out[3 * near_i[r] + c] = acc;
            }
    }
    if (!far_i.empty()) {
        // a mesh segment small on both scales is read as one element
        const double *m = s.mid.data();
        const cd *p = s.moment.data();
        size_t Sf = S;
        if (s.whole && !s.seg_h.empty()) {
            double phase = 0.0;
            for (size_t g = 0; g < s.seg_h.size(); g++)
                phase = std::max(phase, k * s.seg_h[g] + std::abs(k_m) * s.seg_v[g]);
            if (phase <= BELOW_FAR_SEGMENT_PHASE) {
                m = s.seg_mid.data();
                p = s.seg_moment.data();
                Sf = s.seg_h.size();
            }
        }
        for (size_t i : far_i) below_direct(pts + 3 * i, m, p, Sf, eps_t, gz, k, k_m, omega, mu, out + 3 * i);
    }
}

// `_below_grid`: the point-spacing grid sized over `sized` x the mids (the
// pairs within the cap), or nullptr when every pair is past it.
somm_below::Grid *below_grid(const std::vector<double> &sized, const Sources &s, cd eps_t, double gz, double k,
                             cd k_m, double omega, double mu) {
    const double cap = somm_below::R1_CAP_LAMBDA_M * (2.0 * PI / std::abs(k_m));
    double mx = -1.0;
    for (size_t i = 0; i < sized.size() / 3; i++)
        for (size_t j = 0; j < s.n_el(); j++) {
            const double r1 = depth_sum_r1(&sized[3 * i], &s.mid[3 * j], gz);
            if (r1 <= cap) mx = std::max(mx, r1);
        }
    if (mx < 0.0) return nullptr;
    somm_below::Source *src = somm_below::source();
    if (src == nullptr) throw std::runtime_error("below-table: no below/below grid source installed");
    return &src->get(eps_t, k, 1.001 * mx, omega, mu, SPACING_POINT);
}

}  // namespace

void split(const Elements &el, double gz, Sources &above, Sources &below) {
    above = Sources();
    below = Sources();
    const int sub = std::max(el.subdiv, 1);
    size_t e0 = 0;
    for (size_t w = 0; w < el.n_wires; w++) {
        const size_t m = static_cast<size_t>(el.counts[w]);
        bool any = false, all = true;
        for (size_t e = e0; e < e0 + m; e++) {
            const bool b = el.mid[3 * e + 2] < gz;
            any |= b;
            all &= b;
        }
        if (m && any != all)
            throw std::invalid_argument("wire " + std::to_string(w) +
                                        " has elements on both sides of the interface; a crossing card is "
                                        "split in the plane before it is solved");
        Sources &t = (m && any) ? below : above;
        for (size_t e = e0; e < e0 + m; e++) {
            t.mid.insert(t.mid.end(), el.mid + 3 * e, el.mid + 3 * e + 3);
            t.moment.insert(t.moment.end(), el.moment + 3 * e, el.moment + 3 * e + 3);
            t.start.insert(t.start.end(), el.nodes + 3 * (e + w), el.nodes + 3 * (e + w) + 3);
            t.end.insert(t.end.end(), el.nodes + 3 * (e + w + 1), el.nodes + 3 * (e + w + 1) + 3);
        }
        for (size_t q = e0 + w; q < e0 + w + m + 1; q++) {
            t.nodes.insert(t.nodes.end(), el.nodes + 3 * q, el.nodes + 3 * q + 3);
            t.delta.push_back(el.delta[q]);
        }
        e0 += m;
    }
    for (Sources *t : {&above, &below}) {
        const size_t n = t->n_el(), g = n / static_cast<size_t>(sub);
        t->whole = g * static_cast<size_t>(sub) == n;
        t->seg_mid.assign(3 * g, 0.0);
        t->seg_moment.assign(3 * g, cd(0.0, 0.0));
        t->seg_h.resize(g);
        t->seg_v.resize(g);
        for (size_t s = 0; s < g; s++) {
            for (int c = 0; c < 3; c++) {
                double acc = 0.0;
                cd macc(0.0, 0.0);
                for (int e = 0; e < sub; e++) {
                    acc += t->mid[3 * (s * sub + e) + c];
                    macc += t->moment[3 * (s * sub + e) + c];
                }
                t->seg_mid[3 * s + c] = acc / sub;
                t->seg_moment[3 * s + c] = macc;
            }
            double ext[3];
            const double f = static_cast<double>(sub) / std::max(sub - 1, 1);
            for (int c = 0; c < 3; c++)
                ext[c] = std::fabs(t->mid[3 * (s * sub + sub - 1) + c] - t->mid[3 * (s * sub) + c]) * f;
            t->seg_h[s] = std::hypot(ext[0], ext[1]);
            t->seg_v[s] = ext[2];
        }
    }
}

bool crosses(const double *points, size_t n, const Elements &el, double gz) {
    for (size_t e = 0; e < el.n_el; e++)
        if (el.mid[3 * e + 2] < gz) return true;
    for (size_t i = 0; i < n; i++)
        if (points[3 * i + 2] < gz) return true;
    return false;
}

void transmitted(const double *pts, size_t n, const Sources &s, cd eps_t, double gz, double k, double omega,
                 double mu, double radius, bool above, bool magnetic, int n_threads, cd *out) {
    std::fill(out, out + 3 * n, cd(0.0, 0.0));
    if (n == 0 || s.n_el() == 0) return;
    const auto check = [&](const std::vector<double> &src, const char *what) {
        if (src.empty()) return;
        double pmin = std::numeric_limits<double>::infinity(), pmax = -pmin, smin = pmin, smax = -pmin;
        for (size_t i = 0; i < n; i++) {
            pmin = std::min(pmin, pts[3 * i + 2]);
            pmax = std::max(pmax, pts[3 * i + 2]);
        }
        for (size_t j = 0; j < src.size() / 3; j++) {
            smin = std::min(smin, src[3 * j + 2]);
            smax = std::max(smax, src[3 * j + 2]);
        }
        const bool ok = above ? (pmin >= gz && smax <= gz) : (pmax <= gz && smin >= gz);
        if (!ok)
            throw std::invalid_argument(std::string("the transmitted field needs the observers and the ") + what +
                                        " on opposite sides of the interface");
    };
    check(s.mid, "source elements");
    check(s.nodes, "source knots");
    const cd k_m = somm_below::k_medium(eps_t, k);
    const cd c1 = c1_moment(omega, mu);
    const double step = TRANSMITTED_CURL_STEP_WAVELENGTHS * (2.0 * PI / k);
    ni_eval::Evaluator ev;
    ev.k_p = k;
    ev.k_m = k_m;
    ev.n_threads = n_threads;
    const ni_eval::SheetPlan plan = plan_for(pts, n, s, k_m, gz, k, radius, above, step, magnetic);
    if (!magnetic) {
        std::vector<cd> cu(3 * n), gr(3 * n), ch(3 * n);
        moment_terms(pts, n, s, gz, k, radius, above, ev, plan, cu.data(), gr.data());
        charge_term(pts, n, s, gz, radius, above, ev, plan, ch.data());
        for (size_t i = 0; i < 3 * n; i++) out[i] = c1 * (cu[i] + gr[i] + ch[i]);
        return;
    }
    const auto e_ng = [&](const double *q, const double *, size_t m, cd *o) {
        moment_terms(q, m, s, gz, k, radius, above, ev, plan, o, nullptr);
        for (size_t i = 0; i < 3 * m; i++) o[i] *= c1;
    };
    curl(e_ng, pts, n, step, gz, above, out);
    const cd den = cd(0.0, -1.0) * omega * mu;
    for (size_t i = 0; i < 3 * n; i++) out[i] /= den;
}

void below(const double *pts, size_t n, const Sources &s, cd eps_t, double gz, double k, double omega, double mu,
           double radius, bool magnetic, int n_threads, cd *out) {
    (void)n_threads;
    std::fill(out, out + 3 * n, cd(0.0, 0.0));
    if (n == 0 || s.n_el() == 0) return;
    const cd k_m = somm_below::k_medium(eps_t, k);
    const cd a_m = (1.0 - eps_t) / (1.0 + eps_t);
    std::vector<cd> direct(3 * n), image(3 * n), rem(3 * n);
    medium_fields(pts, n, s, false, gz, k, eps_t, k_m, radius, magnetic, direct.data());
    medium_fields(pts, n, s, true, gz, k, eps_t, k_m, radius, magnetic, image.data());
    const double step = BELOW_CURL_STEP_WAVELENGTHS * (2.0 * PI / std::abs(k_m));
    // Sized over every point the curl may read, so all six offsets read ONE
    // interpolant.
    std::vector<double> sized(pts, pts + 3 * n);
    if (magnetic) {
        for (int ax = 0; ax < 3; ax++)
            for (double sg : {1.0, -1.0})
                for (size_t i = 0; i < n; i++)
                    for (int c = 0; c < 3; c++) sized.push_back(pts[3 * i + c] + (c == ax ? sg * step : 0.0));
        for (size_t i = 0; i < n; i++)
            for (int c = 0; c < 3; c++) sized.push_back(pts[3 * i + c] - (c == 2 ? 2.0 * step : 0.0));
    }
    somm_below::Grid *grid = below_grid(sized, s, eps_t, gz, k, k_m, omega, mu);
    const double floor_z = gz - 1e-12 * std::max(1.0, std::fabs(gz)) - 1e-12;
    const auto remainder = [&](const double *q, const double *centre, size_t m, cd *o) {
        // A point a hair under the plane is as deep as the curl lets it be.
        std::vector<double> qq(q, q + 3 * m);
        for (size_t i = 0; i < m; i++) qq[3 * i + 2] = std::min(qq[3 * i + 2], floor_z);
        below_remainder(qq.data(), centre, m, s, eps_t, gz, k, k_m, omega, mu, grid, o);
    };
    if (magnetic) {
        curl(remainder, pts, n, step, gz, false, rem.data());
        const cd den = cd(0.0, -1.0) * omega * mu;
        for (size_t i = 0; i < 3 * n; i++) rem[i] /= den;
    } else {
        remainder(pts, pts, n, rem.data());
    }
    for (size_t i = 0; i < 3 * n; i++) out[i] = direct[i] + (a_m * image[i] + rem[i]);
}

void parts(const double *pts, size_t n, const Elements &el, cd eps_t, double gz, double k, double omega, double mu,
           double radius, bool magnetic, int n_threads, cd *out) {
    Sources above, below_s;
    split(el, gz, above, below_s);
    std::vector<double> pa, pb;
    std::vector<size_t> ia, ib;
    for (size_t i = 0; i < n; i++) {
        (pts[3 * i + 2] >= gz ? pa : pb).insert((pts[3 * i + 2] >= gz ? pa : pb).end(), pts + 3 * i, pts + 3 * i + 3);
        (pts[3 * i + 2] >= gz ? ia : ib).push_back(i);
    }
    std::vector<cd> fa(pa.size()), fb(pb.size()), tb(pb.size());
    transmitted(pa.data(), ia.size(), below_s, eps_t, gz, k, omega, mu, radius, true, magnetic, n_threads, fa.data());
    below(pb.data(), ib.size(), below_s, eps_t, gz, k, omega, mu, radius, magnetic, n_threads, fb.data());
    transmitted(pb.data(), ib.size(), above, eps_t, gz, k, omega, mu, radius, false, magnetic, n_threads, tb.data());
    for (size_t r = 0; r < ia.size(); r++)
        for (int c = 0; c < 3; c++) out[3 * ia[r] + c] = fa[3 * r + c];
    for (size_t r = 0; r < ib.size(); r++)
        for (int c = 0; c < 3; c++) out[3 * ib[r] + c] = fb[3 * r + c] + tb[3 * r + c];
}

}  // namespace near_buried
