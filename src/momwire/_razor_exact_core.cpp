// See `_razor_exact_core.h`. Each step names the Python it replays; the
// arithmetic is that Python's, operation for operation (numpy's elementwise
// complex product is its fused form, `_cmul.cmul` / numpy's AVX2 loop; a
// complex `/` of Python scalars is CPython 3.12's `_Py_c_quot`).
#include "_razor_exact_core.h"
#include "_fma_inline.h"

#include "_exact_kernel_point_core.h"
#include "_razor_core.h"
#include "_razor_front_core.h"

#if defined(_MSC_VER)
#pragma float_control(precise, on)
#pragma fp_contract(off)
#endif

#include <algorithm>
#include <memory>
#include <cmath>
#include <numeric>
#include <stdexcept>

namespace mw {
namespace razor {

namespace {

// numpy's fused elementwise complex product (a real operand is (x, 0)).
inline cd cmul_f(cd a, cd b) {
    return cd(mw_fma::exact(a.real(), b.real(), -(a.imag() * b.imag())),
              mw_fma::exact(a.real(), b.imag(), a.imag() * b.real()));
}

// CPython 3.12's `_Py_c_quot` (Smith's method).
cd py_cquot(cd a, cd b) {
    const double abs_breal = b.real() < 0 ? -b.real() : b.real();
    const double abs_bimag = b.imag() < 0 ? -b.imag() : b.imag();
    if (abs_breal >= abs_bimag) {
        if (abs_breal == 0.0) throw std::domain_error("complex division by zero");
        const double ratio = b.imag() / b.real();
        const double denom = b.real() + b.imag() * ratio;
        return cd((a.real() + a.imag() * ratio) / denom, (a.imag() - a.real() * ratio) / denom);
    }
    const double ratio = b.real() / b.imag();
    const double denom = b.real() * ratio + b.imag();
    return cd((a.real() * ratio + a.imag()) / denom, (a.imag() * ratio - a.real()) / denom);
}

template <class T, int N>
view<T, N> c_view(T *p, std::initializer_list<ptrdiff_t> shape) {
    return view<T, N>::contiguous(p, shape);
}

inline double dot3(const double *a, const double *b) {
    return (a[0] * b[0] + a[1] * b[1]) + a[2] * b[2];
}
inline double norm3(const double *v) { return std::sqrt((v[0] * v[0] + v[1] * v[1]) + v[2] * v[2]); }

}  // namespace

// `_bspline_kernels._ek_axis_groups_coaxial` (tol 1e-6), `_vec3.dot3` order.
std::vector<int64_t> coaxial_labels(const std::vector<double> &l, const std::vector<double> &r,
                                    const std::vector<double> &t, const std::vector<double> &a) {
    const double tol = 1e-6;
    const size_t n = a.size();
    std::vector<double> centers(3 * n);
    for (size_t i = 0; i < 3 * n; i++) centers[i] = 0.5 * (l[i] + r[i]);
    std::vector<int64_t> labels(n, -1);
    std::vector<double> rep_c, rep_t, rep_a;
    for (size_t i = 0; i < n; i++) {
        const double *ti = &t[3 * i];
        int64_t hit = -1;
        for (size_t g = 0; g < rep_a.size(); g++) {
            const bool collinear = std::fabs(dot3(&rep_t[3 * g], ti)) >= 1.0 - tol;
            const bool same_a = std::fabs(a[i] / rep_a[g] - 1.0) <= tol;
            double dvec[3];
            for (int c = 0; c < 3; c++) dvec[c] = centers[3 * i + c] - rep_c[3 * g + c];
            const double axial = dot3(dvec, ti);
            double perp[3];
            for (int c = 0; c < 3; c++) perp[c] = dvec[c] - axial * ti[c];
            const bool coaxial = norm3(perp) <= tol * rep_a[g];
            if (collinear && same_a && coaxial) {
                hit = static_cast<int64_t>(g);
                break;
            }
        }
        if (hit >= 0) {
            labels[i] = hit;
            continue;
        }
        labels[i] = static_cast<int64_t>(rep_a.size());
        rep_c.insert(rep_c.end(), &centers[3 * i], &centers[3 * i] + 3);
        rep_t.insert(rep_t.end(), ti, ti + 3);
        rep_a.push_back(a[i]);
    }
    return labels;
}

// `_exact_kernel._point_rules()`: `_gl01(16)`, `_gl01(20)` and the log-split
// weights `_gl01_log(20)` (numpy's legvander, then each weight a plain sum).
mw::ek::PointRules point_rules() {
    mw::ek::PointRules R;
    auto gl01 = [](int64_t n, std::vector<double> &x, std::vector<double> &w) {
        if (!leggauss(n, x, w)) throw std::logic_error("exact kernel: no tabled rule");
        for (auto &v : x) v = 0.5 * (v + 1.0);
        for (auto &v : w) v = 0.5 * v;
    };
    gl01(16, R.xg, R.wg);
    gl01(20, R.xs, R.ws);
    const int64_t n = 20;
    std::vector<double> lm(n), c(n);
    lm[0] = -1.0;
    for (int64_t m = 1; m < n; m++)
        lm[m] = std::pow(-1.0, static_cast<double>(m + 1)) /
                (static_cast<double>(m) * (static_cast<double>(m) + 1.0));
    for (int64_t m = 0; m < n; m++) c[m] = (2.0 * static_cast<double>(m) + 1.0) * lm[m];
    R.wls.resize(n);
    for (int64_t i = 0; i < n; i++) {
        const double x = 2.0 * R.xs[i] - 1.0;
        std::vector<double> v(n);
        v[0] = x * 0.0 + 1.0;
        if (n > 1) v[1] = x;
        for (int64_t k = 2; k < n; k++)
            v[k] = ((v[k - 1] * x) * static_cast<double>(2 * k - 1) -
                    v[k - 2] * static_cast<double>(k - 1)) /
                   static_cast<double>(k);
        double acc = 0.0;
        for (int64_t m = 0; m < n; m++) acc += v[m] * c[m];
        R.wls[i] = R.ws[i] * acc;
    }
    return R;
}


std::string build_exact_plan(const Front &f, ExactPlan &plan) {
    plan.groups.clear();
    const int64_t n_seg = f.n_seg, n_basis = f.n_basis;
    // `_seg_radius(geom)`: each segment its wire's radius.
    std::vector<double> seg_a(static_cast<size_t>(n_seg));
    for (int64_t w = 0; w < f.n_wires; w++)
        for (int64_t s = f.seg_offsets[w]; s < f.seg_offsets[w + 1]; s++)
            seg_a[s] = f.uniform_radius ? f.radius_value : f.radius_per_wire[w];
    std::vector<double> seg_r(static_cast<size_t>(3 * n_seg));
    for (int64_t s = 0; s < n_seg; s++)
        for (int c = 0; c < 3; c++) seg_r[3 * s + c] = f.seg_p0[3 * s + c] + f.seg_h[s] * f.seg_t[3 * s + c];
    const std::vector<int64_t> labels = coaxial_labels(f.seg_p0, seg_r, f.seg_t, seg_a);
    // The jobs: every one-radius group, then every coaxial LINE carrying
    // more than one radius group (`coaxial_lines`: the one-radius rule with
    // every radius the largest), whose cross-radius pairs take the
    // two-radius moments (momwire#1438).
    struct Job {
        std::vector<int64_t> S, lab;
    };
    std::vector<Job> jobs;
    const int64_t n_groups = labels.empty() ? 0 : *std::max_element(labels.begin(), labels.end()) + 1;
    for (int64_t gl = 0; gl < n_groups; gl++) {
        Job j;
        for (int64_t s = 0; s < n_seg; s++)
            if (labels[s] == gl) j.S.push_back(s);
        jobs.push_back(std::move(j));
    }
    {
        double amax = seg_a.empty() ? 0.0 : seg_a[0];
        for (double v : seg_a) amax = std::max(amax, v);
        const std::vector<int64_t> line =
            coaxial_labels(f.seg_p0, seg_r, f.seg_t, std::vector<double>(seg_a.size(), amax));
        const int64_t nl = line.empty() ? 0 : *std::max_element(line.begin(), line.end()) + 1;
        for (int64_t g = 0; g < nl; g++) {
            Job j;
            bool step = false;
            for (int64_t s = 0; s < n_seg; s++)
                if (line[s] == g) {
                    if (!j.lab.empty() && labels[s] != j.lab[0]) step = true;
                    j.S.push_back(s);
                    j.lab.push_back(labels[s]);
                }
            if (step) jobs.push_back(std::move(j));
        }
    }
    std::vector<bool> grounded(static_cast<size_t>(n_basis), false);
    for (int64_t b : f.grounded_bases) grounded[b] = true;
    // A medium front's chopped rows (momwire#1619, `build_plan`'s `chop`):
    // the ghost wing (sigma 0) is the T2 endpoint moved to the knot, whose
    // observer keeps the fill's kernel. None on any other front.
    std::vector<bool> chopped(static_cast<size_t>(n_basis), false);
    for (int64_t b : f.chop_rows) chopped[static_cast<size_t>(b)] = true;
    // The EK labels (`_ek_labels`) and `obs_rad`, as `build_plan` reads them.
    const bool ek = f.extended_kernel;
    bool one_radius = true;
    for (double v : seg_a)
        if (!(v == seg_a[0])) one_radius = false;
    const int64_t chunk_elems = f.chunk_elems;  // razor's `_CHUNK_ELEMS`
    for (Job &job : jobs) {
        ExactGroup g;
        g.S = job.S;
        const int64_t n = static_cast<int64_t>(g.S.size());
        const int64_t s0 = g.S[0];
        const double *axis = &f.seg_t[3 * s0];
        // `group_frame`
        for (int64_t i = 0; i < n; i++) {
            const int64_t s = g.S[i];
            double d[3], e[3];
            for (int c = 0; c < 3; c++) {
                d[c] = f.seg_p0[3 * s + c] - f.seg_p0[3 * s0 + c];
                e[c] = seg_r[3 * s + c] - f.seg_p0[3 * s + c];
            }
            g.x0.push_back(dot3(d, axis));
            g.sgn.push_back(dot3(&f.seg_t[3 * s], axis) >= 0.0 ? 1.0 : -1.0);
            g.h.push_back(norm3(e));
        }
        std::vector<int64_t> loc(static_cast<size_t>(n_seg), -1);
        for (int64_t i = 0; i < n; i++) loc[g.S[i]] = i;
        // Columns: every live wing on the group, grouped by basis (CSR).
        struct Ent {
            int64_t b, j;
            double t;
            uint8_t rise;
            double q;
        };
        std::vector<Ent> ents;
        for (int w = 0; w < 2; w++)
            for (int64_t b = 0; b < n_basis; b++) {
                const int64_t ss = f.wing_seg[2 * b + w];
                const double sig = f.wing_sigma[2 * b + w];
                if (loc[ss] < 0 || sig == 0.0) continue;
                const int64_t j = loc[ss];
                const bool rise = f.wing_rise.p[2 * b + w];
                ents.push_back({b, j, sig * g.sgn[j], static_cast<uint8_t>(rise ? 1 : 0),
                                (sig * (rise ? 1.0 : -1.0)) / g.h[j]});
            }
        if (ents.empty()) continue;
        std::stable_sort(ents.begin(), ents.end(), [](const Ent &x, const Ent &y) { return x.b < y.b; });
        for (size_t e = 0; e < ents.size(); e++) {
            if (e == 0 || ents[e].b != ents[e - 1].b) {
                g.col_n.push_back(ents[e].b);
                g.col_ptr.push_back(static_cast<int64_t>(e));
            }
            g.ent_j.push_back(ents[e].j);
            g.ent_t.push_back(ents[e].t);
            g.ent_rise.push_back(ents[e].rise);
            g.ent_q.push_back(ents[e].q);
        }
        g.col_ptr.push_back(static_cast<int64_t>(ents.size()));
        // Rows (m, observer), observers the group's centroids (on_cent).
        std::vector<int64_t> r_m, r_o;
        for (int w = 0; w < 2; w++)
            for (int64_t b = 0; b < n_basis; b++) {
                const int64_t ss = f.wing_seg[2 * b + w];
                if (loc[ss] < 0) continue;
                r_m.push_back(b);
                r_o.push_back(loc[ss]);
                g.part_w.push_back(f.wts[2 * b + w]);
                g.part_p.push_back(dot3(&f.tans[6 * b + 3 * w], axis));
                g.part_sign.push_back(w == 0 ? -1.0 : 1.0);
                const bool ghost = chopped[static_cast<size_t>(b)] && f.wing_sigma[2 * b + w] == 0.0;
                g.part_zero.push_back((w == 0 && grounded[b]) || ghost ? 1 : 0);
            }
        for (int64_t i = 0; i < n; i++) {
            const int64_t s = g.S[i];
            double d[3];
            for (int c = 0; c < 3; c++) {
                g.obs_pt.push_back(f.cent[3 * s + c]);
                d[c] = f.cent[3 * s + c] - f.seg_p0[3 * s0 + c];
            }
            g.obs_x.push_back(dot3(d, axis));
        }
        g.by_o.resize(r_o.size());
        std::iota(g.by_o.begin(), g.by_o.end(), int64_t(0));
        std::stable_sort(g.by_o.begin(), g.by_o.end(),
                         [&](int64_t x, int64_t y) { return r_o[x] < r_o[y]; });
        for (int64_t i : g.by_o) {
            g.r_o.push_back(r_o[i]);
            g.r_m.push_back(r_m[i]);
        }
        const int64_t chunk = std::max<int64_t>(1, chunk_elems / std::max<int64_t>(1, 4 * n));
        for (int64_t o0 = 0; o0 < n; o0 += chunk) {
            ExactWindow wd;
            wd.o0 = o0;
            wd.o1 = std::min(n, o0 + chunk);
            wd.e0 = std::lower_bound(g.r_o.begin(), g.r_o.end(), wd.o0) - g.r_o.begin();
            wd.e1 = std::lower_bound(g.r_o.begin(), g.r_o.end(), wd.o1) - g.r_o.begin();
            g.windows.push_back(wd);
        }
        g.a = seg_a[s0];
        if (!job.lab.empty()) {
            // A cross job: per-segment radii and labels, the observers' own
            // (the centroids: the segments themselves).
            g.cross = true;
            g.cross_lab = job.lab;
            for (int64_t s : g.S) g.cross_a.push_back(seg_a[s]);
        }
        for (int64_t s : g.S) {
            for (int c = 0; c < 3; c++) {
                g.sub_p0.push_back(f.seg_p0[3 * s + c]);
                g.sub_t.push_back(f.seg_t[3 * s + c]);
            }
            g.sub_h.push_back(f.seg_h[s]);
            g.sub_a.push_back(f.a_col[s]);  // `_kernel_radius` of the sub-geometry
            if (ek) {
                g.lab_obs.push_back(f.group[s]);
                g.lab_src.push_back(f.group[s]);
                if (!one_radius) g.b_obs.push_back(seg_a[s]);
            }
        }
        g.n = n;
        plan.groups.push_back(std::move(g));
    }
    return "";
}

void add_exact_correction(const Front &f, const ExactPlan &plan, cd *Z, uintptr_t cancel_flag) {
    const cd c_A(0.0, f.c_a_imag), c_Phi(0.0, f.c_phi_imag);
    const mw::ek::PointRules rules = point_rules();
    const int64_t ld = f.n_basis;
    for (const ExactGroup &g : plan.groups) {
        const int64_t n = g.n;
        double hmax = g.h[0];
        for (double v : g.h) hmax = std::max(hmax, v);
        std::unique_ptr<mw::ek::PointRows> source;
        std::unique_ptr<mw::ek::PointCrossRows> cross;
        if (g.cross) {
            double amax = g.cross_a[0];
            for (double v : g.cross_a) amax = std::max(amax, v);
            cross.reset(new mw::ek::PointCrossRows(g.x0.data(), g.sgn.data(), g.h.data(),
                                                   g.cross_a.data(), g.cross_lab.data(), n, f.k,
                                                   2, 4 * n + 1024, 1e-11 * hmax, 1e-11 * amax,
                                                   rules, 0));
        } else {
            source.reset(new mw::ek::PointRows(g.x0.data(), g.sgn.data(), g.h.data(), n, g.a,
                                               f.k, 2, 4 * n + 1024, 1e-11 * hmax, rules, 0));
        }
        // `row_cA` / `row_cP`, then by observer
        const size_t nr = g.part_w.size();
        std::vector<cd> a_pre(nr), p_pre(nr), r_A(nr), r_P(nr);
        for (size_t i = 0; i < nr; i++) {
            a_pre[i] = cmul_f(cmul_f(c_A, cd(g.part_w[i], 0.0)), cd(g.part_p[i], 0.0));
            p_pre[i] = g.part_zero[i] ? cd(0.0, 0.0) : py_cquot(cd(g.part_sign[i], 0.0), c_Phi);
        }
        for (size_t i = 0; i < nr; i++) {
            r_A[i] = a_pre[g.by_o[i]];
            r_P[i] = p_pre[g.by_o[i]];
        }
        for (const ExactWindow &w : g.windows) {
            // `self._checkpoint()`, once per window
            const volatile int32_t *flag = reinterpret_cast<const volatile int32_t *>(cancel_flag);
            if (flag && *flag) throw AbortedError{};
            const int64_t rows = w.o1 - w.o0;
            std::vector<cd> mom(static_cast<size_t>(2 * rows * n));
            if (cross)
                cross->rows(&g.obs_x[w.o0], &g.cross_a[w.o0], &g.cross_lab[w.o0], rows,
                            mom.data());
            else
                source->rows(&g.obs_x[w.o0], rows, mom.data());
            std::vector<cd> m0b(static_cast<size_t>(rows * n)), m1b(static_cast<size_t>(rows * n));
            view<const int64_t, 1> gi, gj;
            view<const double, 1> aek, bo;
            std::vector<int64_t> gi_v;
            std::vector<double> bo_v;
            if (!g.lab_src.empty()) {
                gi_v.assign(g.lab_obs.begin() + w.o0, g.lab_obs.begin() + w.o1);
                gi = c_view<const int64_t, 1>(gi_v.data(), {rows});
                gj = c_view<const int64_t, 1>(g.lab_src.data(), {n});
                aek = c_view<const double, 1>(g.sub_a.data(), {n});
                if (!g.b_obs.empty()) {
                    bo_v.assign(g.b_obs.begin() + w.o0, g.b_obs.begin() + w.o1);
                    bo = c_view<const double, 1>(bo_v.data(), {rows});
                }
            }
            seg_moments(c_view<const double, 2>(&g.obs_pt[3 * w.o0], {rows, 3}),
                        c_view<const double, 2>(g.sub_p0.data(), {n, 3}),
                        c_view<const double, 2>(g.sub_t.data(), {n, 3}),
                        c_view<const double, 1>(g.sub_h.data(), {n}),
                        c_view<const double, 1>(g.sub_a.data(), {n}),
                        c_view<const double, 1>(f.xg.data(), {static_cast<ptrdiff_t>(f.xg.size())}),
                        c_view<const double, 1>(f.wg.data(), {static_cast<ptrdiff_t>(f.wg.size())}),
                        f.k, true, gi, gj, aek, cancel_flag, bo,
                        c_view<cd, 2>(m0b.data(), {rows, n}), c_view<cd, 2>(m1b.data(), {rows, n}));
            if (cross)  // the same-radius pairs are the one-radius groups'
                for (int64_t o = 0; o < rows; o++)
                    for (int64_t j = 0; j < n; j++)
                        if (g.cross_lab[w.o0 + o] == g.cross_lab[j]) {
                            m0b[o * n + j] = cd(0.0, 0.0);
                            m1b[o * n + j] = cd(0.0, 0.0);
                        }
            // `np.subtract(mom[q], M_qb)`
            const size_t pl = static_cast<size_t>(rows * n);
            for (size_t i = 0; i < pl; i++) {
                m0b[i] = cd(mom[i].real() - m0b[i].real(), mom[i].imag() - m0b[i].imag());
                m1b[i] = cd(mom[pl + i].real() - m1b[i].real(), mom[pl + i].imag() - m1b[i].imag());
            }
            const int64_t R = w.e1 - w.e0;
            std::vector<int64_t> ro(static_cast<size_t>(R));
            for (int64_t i = 0; i < R; i++) ro[i] = g.r_o[w.e0 + i] - w.o0;
            mw::ek::point_assemble(m0b.data(), m1b.data(), n, &g.r_m[w.e0], ro.data(),
                                   &r_A[w.e0], &r_P[w.e0], R, g.col_ptr.data(), g.col_n.data(),
                                   static_cast<int64_t>(g.col_n.size()), g.ent_j.data(),
                                   g.ent_t.data(), g.ent_rise.data(), g.ent_q.data(), g.h.data(), Z,
                                   ld, 0);
        }
    }
}

}  // namespace razor
}  // namespace mw
