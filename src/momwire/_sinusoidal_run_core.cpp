// The sinusoidal basis' run-half answers (see `_sinusoidal_run_core.h`).

#if defined(_MSC_VER) && !defined(__clang__)
#pragma float_control(precise, on)
#pragma fp_contract(off)
#endif

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
std::vector<cd> basis_at(const Front &fr, const std::vector<ll> &eval_seg, const std::vector<double> &eval_s,
                         const std::vector<cd> &alpha) {
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

}  // namespace

Engine::Engine(const nec4::StructureBuild &structure, const FillLinalg &la)
    : structure_(structure), la_(la), fronts_(structure.points.size()), solves_(structure.points.size()) {}

const Front &Engine::front_at(std::size_t index) {
    if (!fronts_.at(index)) fronts_[index].reset(new Front(front(structure_.points[index].args)));
    return *fronts_[index];
}

const nec4::EngineSolve &Engine::solve(std::size_t index) {
    if (!solves_.at(index)) {
        const Front &fr = front_at(index);
        const auto t0 = std::chrono::steady_clock::now();
        const PortSolution ps = port_solution(fr, structure_.points[index].args, la_);
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

nec4::RunEngine &Engines::engine_for(const nec4::StructureBuild &structure) {
    engines_.emplace_back(new Engine(structure, la_));
    return *engines_.back();
}

bool printout(const std::string &text, const nec5::EngineStamp &stamp, const FillLinalg &la, std::string &out,
              std::string &why) {
    Engines engines(la);
    return nec4::render(text, "sinusoidal", stamp, engines, la.np, out, why);
}

}  // namespace sinusoidal
}  // namespace mw
