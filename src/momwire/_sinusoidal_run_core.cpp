// The sinusoidal basis' run-half answers (see `_sinusoidal_run_core.h`).

#if defined(_MSC_VER) && !defined(__clang__)
#pragma float_control(precise, on)
#pragma fp_contract(off)
#endif

#include "_libm_complex_inline.h"
#include "_sinusoidal_run_core.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <string>
#include <vector>

namespace mw {
namespace sinusoidal {

namespace {

typedef long long ll;
typedef std::complex<double> cd;
namespace rp = nec5::replay;

cd cm(cd a, cd b) { return rp::np_cmul(a, b); }
cd cadd(cd a, cd b) { return cd(a.real() + b.real(), a.imag() + b.imag()); }
cd csub(cd a, cd b) { return cd(a.real() - b.real(), a.imag() - b.imag()); }
cd sig(int8_t s) { return cd(static_cast<double>(s), 0.0); }

// `_evaluate_basis_at_points(seg_view, eval_seg, eval_s, alpha)` in free
// space: each eval point's segment entries, `np.add.at` in entry order.
std::vector<cd> basis_at_complex(const Front &fr, const std::vector<ll> &eval_seg, const std::vector<double> &eval_s,
                                 const std::vector<cd> &alpha);
std::vector<cd> slope_at_complex(const Front &fr, const std::vector<ll> &eval_seg, const std::vector<double> &eval_s,
                                 const std::vector<cd> &alpha);

std::vector<cd> basis_at(const Front &fr, const std::vector<ll> &eval_seg, const std::vector<double> &eval_s,
                         const std::vector<cd> &alpha) {
    if (fr.buried) return basis_at_complex(fr, eval_seg, eval_s, alpha);
    const SegView &v = fr.basis;
    const double k = fr.k;
    std::vector<cd> out(eval_seg.size(), cd(0.0, 0.0));
    for (std::size_t e = 0; e < eval_seg.size(); e++) {
        const std::size_t s = static_cast<std::size_t>(eval_seg[e]);
        const double sin_e = std::sin(k * eval_s[e]);
        const double half = std::sin(0.5 * k * eval_s[e]);
        const double cosm1 = -2.0 * half * half;
        for (ll q = v.starts[s]; q < v.starts[s + 1]; q++) {
            const std::size_t g = static_cast<std::size_t>(q);
            const cd inner = cadd(cadd(cm(sig(v.sigma[g]), v.AC[g]), cm(v.B[g], cd(sin_e, 0.0))),
                                  cm(cm(sig(v.sigma[g]), v.C[g]), cd(cosm1, 0.0)));
            out[e] = cadd(out[e], cm(alpha.at(static_cast<std::size_t>(v.jbasis[g])), inner));
        }
    }
    return out;
}

// `_evaluate_basis_slope_at_points(seg_view, eval_seg, eval_s, alpha)`
std::vector<cd> slope_at(const Front &fr, const std::vector<ll> &eval_seg, const std::vector<double> &eval_s,
                         const std::vector<cd> &alpha) {
    if (fr.buried) return slope_at_complex(fr, eval_seg, eval_s, alpha);
    const SegView &v = fr.basis;
    const double k = fr.k;
    std::vector<cd> out(eval_seg.size(), cd(0.0, 0.0));
    for (std::size_t e = 0; e < eval_seg.size(); e++) {
        const std::size_t s = static_cast<std::size_t>(eval_seg[e]);
        const double sin_e = std::sin(k * eval_s[e]);
        const double cos_e = std::cos(k * eval_s[e]);
        for (ll q = v.starts[s]; q < v.starts[s + 1]; q++) {
            const std::size_t g = static_cast<std::size_t>(q);
            const cd ak = cm(alpha.at(static_cast<std::size_t>(v.jbasis[g])), cd(k, 0.0));
            const cd inner = csub(cm(v.B[g], cd(cos_e, 0.0)), cm(cm(sig(v.sigma[g]), v.C[g]), cd(sin_e, 0.0)));
            out[e] = cadd(out[e], cm(ak, inner));
        }
    }
    return out;
}

// The `k_entry` branch of `_evaluate_basis_at_points` (a wholly-buried
// deck's view, every entry at k_m, momwire#1607 B6a): per entry
// `_basis_value(sig * AC, B, sig * C, k_e, s_e)`, times alpha, `np.add.at`
std::vector<cd> basis_at_complex(const Front &fr, const std::vector<ll> &eval_seg, const std::vector<double> &eval_s,
                                 const std::vector<cd> &alpha) {
    const SegView &v = fr.basis;
    const cd k = fr.medium.k_m;
    const cd half_k = cm(cd(0.5, 0.0), k);  // `0.5 * k_e`, an array product
    std::vector<cd> out(eval_seg.size(), cd(0.0, 0.0));
    for (std::size_t e = 0; e < eval_seg.size(); e++) {
        const std::size_t s = static_cast<std::size_t>(eval_seg[e]);
        const cd se(eval_s[e], 0.0);
        const cd half = mw_libm_c::csin(cm(half_k, se));
        const cd sin_e = mw_libm_c::csin(cm(k, se));
        for (ll q = v.starts[s]; q < v.starts[s + 1]; q++) {
            const std::size_t g = static_cast<std::size_t>(q);
            const cd sig_ac = cm(sig(v.sigma[g]), v.AC[g]), sig_c = cm(sig(v.sigma[g]), v.C[g]);
            const cd f = csub(cadd(sig_ac, cm(v.B[g], sin_e)), cm(cm(cd(2.0, 0.0), sig_c), cm(half, half)));
            out[e] = cadd(out[e], cm(alpha.at(static_cast<std::size_t>(v.jbasis[g])), f));
        }
    }
    return out;
}

// `_evaluate_basis_slope_at_points`' `k_entry` branch:
// `k_e * (B * cos(k_e s) - sig * C * sin(k_e s))`
std::vector<cd> slope_at_complex(const Front &fr, const std::vector<ll> &eval_seg, const std::vector<double> &eval_s,
                                 const std::vector<cd> &alpha) {
    const SegView &v = fr.basis;
    const cd k = fr.medium.k_m;
    std::vector<cd> out(eval_seg.size(), cd(0.0, 0.0));
    for (std::size_t e = 0; e < eval_seg.size(); e++) {
        const std::size_t s = static_cast<std::size_t>(eval_seg[e]);
        const cd ks = cm(k, cd(eval_s[e], 0.0));
        const cd cos_e = mw_libm_c::ccos(ks), sin_e = mw_libm_c::csin(ks);
        for (ll q = v.starts[s]; q < v.starts[s + 1]; q++) {
            const std::size_t g = static_cast<std::size_t>(q);
            const cd fd = cm(k, csub(cm(v.B[g], cos_e), cm(cm(sig(v.sigma[g]), v.C[g]), sin_e)));
            out[e] = cadd(out[e], cm(alpha.at(static_cast<std::size_t>(v.jbasis[g])), fd));
        }
    }
    return out;
}

// --- `wire_loss_power(coeffs)`: the conductor's I²R ---------------------------

cd cdivr(cd a, double x) { return nec5::replay::np_cdiv(a, cd(x, 0.0)); }
cd rsubc(double x, cd b) { return cd(x - b.real(), 0.0 - b.imag()); }  // float - complex
cd cneg(cd a) { return cd(-a.real(), -a.imag()); }
double cabs(cd a) { return std::hypot(a.real(), a.imag()); }

// The series' Horner bracket `1 - u2/20 (1 - u2/42 (1 - u2/72 (1 - u2/110)))`
cd bracket(cd u2) {
    const cd w = rsubc(1.0, cdivr(u2, 110.0));
    const cd z = rsubc(1.0, cm(cdivr(u2, 72.0), w));
    const cd y = rsubc(1.0, cm(cdivr(u2, 42.0), z));
    return rsubc(1.0, cm(cdivr(u2, 20.0), y));
}

// `_abs_shape_integrals`' `g_of(c)` at one segment: `h * where(|u| < 0.1,
// series, _sin_minus_arg(u) / safe)`, u = 0.5 c h.
cd g_of(cd c, double h) {
    const cd u = cm(cm(cd(0.5, 0.0), c), cd(h, 0.0));
    const cd u2 = cm(u, u);
    const cd series = cm(cdivr(cneg(u2), 6.0), bracket(u2));
    const cd safe = (u.real() == 0.0 && u.imag() == 0.0) ? cd(1.0, 0.0) : u;
    // `_sin_minus_arg(u)`
    const cd sm_series = cm(cdivr(cneg(cm(u, u2)), 6.0), bracket(u2));
    const cd sm = cabs(u) < 0.1 ? sm_series : csub(std::sin(u), u);
    const cd sel = cabs(u) < 0.1 ? series : nec5::replay::np_cdiv(sm, safe);
    return cm(cd(h, 0.0), sel);
}

// `(-1)**(m + n) / ((2m)! (2n)! (2(m + n) + 1))`, Python's correctly rounded
// int / int, m, n = 1..7
const double kShapeCoef[7][7] = {
    {0x1.999999999999ap-5, -0x1.8618618618618p-9, 0x1.43a2730abee4dp-14, -0x1.2e9e8cd187470p-20, 0x1.6c2d84445b072p-27, -0x1.320e7665ec4e6p-34, 0x1.7bd9af92da276p-42},
    {-0x1.8618618618618p-9, 0x1.948b0fcd6e9e0p-13, -0x1.610e4ef473283p-18, 0x1.556aac001556bp-24, -0x1.a4d3e2cc24ebcp-31, 0x1.681103c334200p-38, -0x1.c527a00d82021p-46},
    {0x1.43a2730abee4dp-14, -0x1.610e4ef473283p-18, 0x1.3ea7d3bbcfa64p-23, -0x1.3b9eea191bb0dp-29, 0x1.8c12b756b9566p-36, -0x1.57a472f92d419p-43, 0x1.b55490b613d13p-51},
    {-0x1.2e9e8cd187470p-20, 0x1.556aac001556bp-24, -0x1.3b9eea191bb0dp-29, 0x1.3e45e59c8bca0p-35, -0x1.9501d0a5ac31dp-42, 0x1.6354b593f01a0p-49, -0x1.c8583df5a55fep-57},
    {0x1.6c2d84445b072p-27, -0x1.a4d3e2cc24ebcp-31, 0x1.8c12b756b9566p-36, -0x1.9501d0a5ac31dp-42, 0x1.04937417278a8p-48, -0x1.cd6a4a0698fd6p-56, 0x1.2a8d03e61c8c1p-63},
    {-0x1.320e7665ec4e6p-34, 0x1.681103c334200p-38, -0x1.57a472f92d419p-43, 0x1.6354b593f01a0p-49, -0x1.cd6a4a0698fd6p-56, 0x1.9ba366586d2dcp-63, -0x1.0c0f223e604fdp-70},
    {0x1.7bd9af92da276p-42, -0x1.c527a00d82021p-46, 0x1.b55490b613d13p-51, -0x1.c8583df5a55fep-57, 0x1.2a8d03e61c8c1p-63, -0x1.0c0f223e604fdp-70, 0x1.5f0bf6222767ap-78},
};

struct ShapeIntegrals {
    cd i_g;
    double i_ss, i_gg, i_cc;
};

// `_abs_shape_integrals(k, h)` at one segment, for a real k
ShapeIntegrals abs_shape_integrals(cd k, double h) {
    ShapeIntegrals r;
    const cd kc = std::conj(k);
    r.i_g = g_of(k, h);
    const double g_d = g_of(csub(k, kc), h).real();
    const double g_s = g_of(cadd(k, kc), h).real();
    r.i_ss = 0.5 * (g_d - g_s);
    r.i_cc = h + 0.5 * (g_d + g_s);
    const cd half_kh = cm(cm(cd(0.5, 0.0), k), cd(h, 0.0));
    const cd z = cm(half_kh, half_kh);  // `(0.5 * k * h) ** 2`: numpy's square is the fused product
    const cd zc = std::conj(z);
    double series = 0.0;
    cd zm(1.0, 0.0);
    for (int m = 1; m < 8; m++) {
        zm = cm(zm, z);
        cd zn(1.0, 0.0);
        for (int n = 1; n < 8; n++) {
            zn = cm(zn, zc);
            series = series + cm(zm, zn).real() * kShapeCoef[m - 1][n - 1];
        }
    }
    const double literal = 0.5 * (g_d + g_s) - 2.0 * r.i_g.real();
    r.i_gg = cabs(half_kh) < 0.25 ? h * series : literal;
    return r;
}

}  // namespace

Engine::Engine(const nec4::StructureBuild &structure, const FillLinalg &la)
    : structure_(structure), la_(la), fronts_(structure.points.size()), solves_(structure.points.size()) {}

const Front &Engine::front_at(std::size_t index) {
    if (!fronts_.at(index)) {
        try {
            fronts_[index].reset(new Front(front(structure_.points[index].args, galerkin_)));
        } catch (const FrontError &e) {
            throw nec4::Refusal(e.message);  // the constructor's ValueError
        }
    }
    return *fronts_[index];
}

const nec4::EngineSolve &Engine::solve(std::size_t index) {
    if (!solves_.at(index)) {
        const Front &fr = front_at(index);
        const auto t0 = std::chrono::steady_clock::now();
        PortSolution ps;
        try {
            ps = port_solution(fr, fr.args, la_);  // the solved wires (momwire#1042)
        } catch (const FrontError &e) {
            throw nec4::Refusal(e.message);  // a ValueError the fill raises (the below remainder's)
        }
        std::unique_ptr<nec4::EngineSolve> es(new nec4::EngineSolve);
        es->fill_seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        es->n_basis = ps.n;
        es->n_ports = ps.n_ports;
        es->y = ps.y;
        es->x = ps.coeffs;  // scipy's solve returns C order
        es->x_s0 = ps.n_ports;
        es->x_s1 = 1;
        solves_[index] = std::move(es);
    }
    return *solves_[index];
}

// `currents_at_knots(alpha)`: the knot path
std::vector<std::vector<cd>> Engine::currents_at_knots(std::size_t index, const std::vector<cd> &alpha) {
    const Front &fr = front_at(index);
    const Geometry &g = fr.geometry;
    std::vector<ll> eval_seg, target;
    std::vector<double> eval_s, weight;
    std::vector<std::size_t> offsets{0};
    for (std::size_t w = 0; w < g.wire_first.size(); w++) {
        const ll first = g.wire_first[w], last = g.wire_last[w], n = last - first + 1;
        const ll base = static_cast<ll>(offsets.back());
        for (int side = 0; side < 2; side++)
            for (ll i = 0; i < n; i++) {
                const double h = g.seg_h[static_cast<std::size_t>(first + i)];
                eval_seg.push_back(first + i);
                eval_s.push_back(side == 0 ? -0.5 * h : 0.5 * h);
                target.push_back(side == 0 ? base + i : base + i + 1);
                const bool edge = side == 0 ? i == 0 : i == n - 1;
                weight.push_back(edge ? 1.0 : 0.5);
            }
        offsets.push_back(static_cast<std::size_t>(base + n + 1));
    }
    const std::vector<cd> evals = basis_at(fr, eval_seg, eval_s, alpha);
    std::vector<cd> flat(offsets.back(), cd(0.0, 0.0));
    for (std::size_t e = 0; e < evals.size(); e++) {
        cd &slot = flat[static_cast<std::size_t>(target[e])];
        slot = cadd(slot, cm(cd(weight[e], 0.0), evals[e]));
    }
    std::vector<std::vector<cd>> out;
    for (std::size_t w = 0; w + 1 < offsets.size(); w++)
        out.emplace_back(flat.begin() + static_cast<std::ptrdiff_t>(offsets[w]),
                         flat.begin() + static_cast<std::ptrdiff_t>(offsets[w + 1]));
    return out;
}

// `currents_at_knots(alpha, s_array)` on one wire: a sample on an interior
// knot (within 1e-12 of the wire's arc, or of 1 m) is the mean of the two
// segments' edge values, any other the containing segment's value
std::vector<cd> Engine::currents_at_arcs(std::size_t index, const std::vector<cd> &alpha, std::size_t w,
                                         const std::vector<double> &sv) {
    const Front &fr = front_at(index);
    const Geometry &g = fr.geometry;
    if (w >= g.wire_first.size()) throw nec4::Declined("a sample on a wire the solver does not hold");
    const ll first = g.wire_first[w], n = g.wire_last[w] - first + 1;
    std::vector<double> arc{0.0};
    double run = 0.0;
    for (ll i = 0; i < n; i++) {
        run = run + g.seg_h[static_cast<std::size_t>(first + i)];
        arc.push_back(run);
    }
    const std::size_t n_s = sv.size();
    if (n_s == 0) return {};
    const double wire_arc = arc.back();
    const double eps = 1e-12 * std::max(wire_arc, 1.0);  // Python's max(wire_arc, 1.0)
    std::vector<ll> nk_seg, k_left, k_right;
    std::vector<double> nk_s;
    std::vector<std::size_t> nk_target, k_target;
    for (std::size_t q = 0; q < n_s; q++) {
        if (std::isnan(sv[q])) throw nec4::Declined("a NaN arclength");
        const double c = std::min(std::max(sv[q], 0.0), wire_arc);  // np.clip
        // searchsorted(arc, c) (side left)
        const ll hit = static_cast<ll>(std::lower_bound(arc.begin(), arc.end(), c) - arc.begin());
        const ll hit_c = std::min(std::max(hit, 0LL), n);
        const bool on_knot = hit > 0 && hit < n && std::fabs(c - arc[static_cast<std::size_t>(hit_c)]) <= eps;
        if (on_knot) {
            k_target.push_back(q);
            k_left.push_back(first + hit - 1);
            k_right.push_back(first + hit);
        } else {
            ll pos = static_cast<ll>(std::upper_bound(arc.begin(), arc.end(), c) - arc.begin()) - 1;
            pos = std::min(std::max(pos, 0LL), n - 1);
            nk_target.push_back(q);
            nk_seg.push_back(first + pos);
            nk_s.push_back((c - arc[static_cast<std::size_t>(pos)]) - 0.5 * g.seg_h[static_cast<std::size_t>(first + pos)]);
        }
    }
    std::vector<ll> all_seg(nk_seg);
    std::vector<double> all_s(nk_s);
    std::vector<std::size_t> all_target(nk_target);
    std::vector<double> all_weight(nk_seg.size(), 1.0);
    for (ll sg : k_left) {
        all_seg.push_back(sg);
        all_s.push_back(+0.5 * g.seg_h[static_cast<std::size_t>(sg)]);
    }
    for (ll sg : k_right) {
        all_seg.push_back(sg);
        all_s.push_back(-0.5 * g.seg_h[static_cast<std::size_t>(sg)]);
    }
    all_target.insert(all_target.end(), k_target.begin(), k_target.end());
    all_target.insert(all_target.end(), k_target.begin(), k_target.end());
    all_weight.insert(all_weight.end(), 2 * k_target.size(), 0.5);
    const std::vector<cd> evals = basis_at(fr, all_seg, all_s, alpha);
    std::vector<cd> out(n_s, cd(0.0, 0.0));
    for (std::size_t e = 0; e < evals.size(); e++) {
        cd &slot = out[all_target[e]];
        slot = cadd(slot, cm(cd(all_weight[e], 0.0), evals[e]));
    }
    return out;
}

double Engine::solver_omega(std::size_t index) { return front_at(index).omega; }
const nec4::WireDedup *Engine::dedup(std::size_t index) {
    const Front &fr = front_at(index);
    return fr.dedup.active ? &fr.dedup : nullptr;
}
double Engine::solver_eps0(std::size_t index) {
    (void)index;
    return 8.8541878188e-12;  // the sinusoidal solver's `eps` (the fill's eps~)
}

// `current_slopes(coeffs, s_array)`
std::vector<std::vector<cd>> Engine::current_slopes(std::size_t index, const std::vector<cd> &alpha,
                                                    const std::vector<std::vector<double>> &s) {
    const Front &fr = front_at(index);
    const Geometry &g = fr.geometry;
    if (s.size() != g.wire_first.size()) throw nec4::Declined("a slope request for another wire count");
    std::vector<ll> segs;
    std::vector<double> xis;
    std::vector<std::size_t> offsets{0};
    for (std::size_t w = 0; w < s.size(); w++) {
        const ll first = g.wire_first[w], n = g.wire_last[w] - first + 1;
        std::vector<double> arc{0.0};
        double run = 0.0;
        for (ll i = 0; i < n; i++) {
            run = run + g.seg_h[static_cast<std::size_t>(first + i)];
            arc.push_back(run);
        }
        for (double sv : s[w]) {
            if (std::isnan(sv)) throw nec4::Declined("a NaN arclength");
            const double c = std::min(std::max(sv, 0.0), arc.back());  // np.clip
            // searchsorted(arc, c, side="right") - 1, clipped to [0, n - 1]
            ll pos = static_cast<ll>(std::upper_bound(arc.begin(), arc.end(), c) - arc.begin()) - 1;
            pos = std::min(std::max(pos, 0LL), n - 1);
            segs.push_back(first + pos);
            xis.push_back((c - arc[static_cast<std::size_t>(pos)]) - 0.5 * g.seg_h[static_cast<std::size_t>(first + pos)]);
        }
        offsets.push_back(segs.size());
    }
    std::vector<std::vector<cd>> out;
    if (segs.empty()) {
        for (std::size_t w = 0; w < s.size(); w++) out.emplace_back();
        return out;
    }
    const std::vector<cd> flat = slope_at(fr, segs, xis, alpha);
    for (std::size_t w = 0; w + 1 < offsets.size(); w++)
        out.emplace_back(flat.begin() + static_cast<std::ptrdiff_t>(offsets[w]),
                         flat.begin() + static_cast<std::ptrdiff_t>(offsets[w + 1]));
    return out;
}

// `SinusoidalSolver.wire_loss_power(coeffs)[0]` (no lower medium, no
// insulation): the I²R of each segment's current, |I|² integrated by
// `_abs_shape_integrals`, summed per wire.
double Engine::wire_loss_power(std::size_t index, const std::vector<cd> &coeffs) {
    const Front &fr = front_at(index);
    const nec4::SolverArgs &args = fr.args;  // the solved wires (momwire#1042)
    std::vector<cd> z_wire;
    if (!wire_impedance_per_wire(fr, args, fr.omega, z_wire)) return 0.0;
    const double k = fr.omega / fr.c;
    if (k != fr.k) throw nec4::Declined("a loss readout at another wavenumber than the fill's");
    // `_buried_wire_loss_power`: the shapes at k_m, the view the fill built
    // (`buried_medium` at omega / c is the front's medium)
    const cd k_shape = fr.buried ? fr.medium.k_m : cd(k, 0.0);
    const Geometry &g = fr.geometry;
    const SegView &v = fr.basis;
    const std::size_t n = static_cast<std::size_t>(g.n_segs);
    std::vector<cd> U(n, cd(0.0, 0.0)), Q(n, cd(0.0, 0.0)), R(n, cd(0.0, 0.0));
    for (std::size_t i = 0; i < n; i++)
        for (ll q = v.starts[i]; q < v.starts[i + 1]; q++) {
            const std::size_t e = static_cast<std::size_t>(q);
            const cd a = coeffs.at(static_cast<std::size_t>(v.jbasis[e]));
            U[i] = cadd(U[i], cm(cm(a, sig(v.sigma[e])), v.AC[e]));
            Q[i] = cadd(Q[i], cm(a, v.B[e]));
            R[i] = cadd(R[i], cm(cm(a, sig(v.sigma[e])), v.C[e]));
        }
    // `U * np.conj(R)`: a fresh complex temporary of 256 KiB or more on the
    // right is elided into `conj(R) * U` (B3's measured rule)
    const bool elide = n >= 16384;
    std::vector<double> per_wire(g.wire_first.size(), 0.0);
    for (std::size_t w = 0; w < g.wire_first.size(); w++)
        for (ll i = g.wire_first[w]; i <= g.wire_last[w]; i++) {
            const std::size_t s = static_cast<std::size_t>(i);
            const double h = g.seg_h[s];
            const ShapeIntegrals si = abs_shape_integrals(k_shape, h);
            const double au = cabs(U[s]), aq = cabs(Q[s]), ar = cabs(R[s]);
            const cd ur = elide ? cm(std::conj(R[s]), U[s]) : cm(U[s], std::conj(R[s]));
            const double cross = 2.0 * cm(ur, std::conj(si.i_g)).real();
            const double i2 = (((au * au) * h + cross) + (aq * aq) * si.i_ss) + (ar * ar) * si.i_gg;
            per_wire[w] = per_wire[w] + (0.5 * z_wire[w].real()) * i2;
        }
    return nec5::replay::np_pairwise(per_wire.data(), static_cast<int64_t>(per_wire.size()));
}

nec4::RunEngine &Engines::engine_for(const nec4::StructureBuild &structure) {
    engines_.emplace_back(new Engine(structure, la_));
    return *engines_.back();
}

bool printout(const std::string &text, const nec5::EngineStamp &stamp, const FillLinalg &la, std::string &out,
              std::string &why) {
    Engines engines(la);
    return nec4::render(text, "sinusoidal", stamp, engines, la.np, out, why);
}

bool printout_piece(const std::string &text, const nec5::EngineStamp &stamp, const FillLinalg &la,
                    std::string &out, std::string &why, bool &go_on) {
    Engines engines(la);
    return nec4::render_piece(text, "sinusoidal", stamp, engines, la.np, out, why, go_on);
}

}  // namespace sinusoidal
}  // namespace mw
