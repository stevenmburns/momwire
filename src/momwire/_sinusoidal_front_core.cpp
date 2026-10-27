// The sinusoidal front half in C++ (see `_sinusoidal_front_core.h`). Each
// function names the Python it copies; numpy's vectorised expressions are
// written out per element in numpy's evaluation order, with no contraction.

#if defined(_MSC_VER) && !defined(__clang__)
#pragma float_control(precise, on)
#pragma fp_contract(off)
#endif

#include "_sinusoidal_front_core.h"

#include "_sinusoidal_nec4_core.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <initializer_list>
#include <limits>
#include <map>
#include <numeric>
#include <set>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace mw {
namespace sinusoidal {

namespace {

typedef long long ll;
typedef std::complex<double> cd;
using nec4::Declined;
using nec4::Point3;

const double kEps = 8.8541878188e-12;   // SinusoidalSolver.eps
const double kMu = 1.25663706127e-6;    // SinusoidalSolver.mu
const double kPi = 3.141592653589793;   // np.pi
const double kEulerGamma = 0.5772156649015329;
const double kJunctionTol = 1e-9;       // _junction_rule.JUNCTION_TOL
const double kCoincidenceFloor = 1e-5;  // _wire_spec._JUNCTION_COINCIDENCE_FLOOR
const double kAmbiguityTolFrac = 1e-11; // _feed_snap.AMBIGUITY_TOL_FRAC

[[noreturn]] void fail(const std::string &m) { throw FrontError(m); }
[[noreturn]] void decline(const std::string &m) { throw Declined(m); }

// _vec3.norm3 on one 3-vector
double norm3(double x, double y, double z) { return std::sqrt((x * x + y * y) + z * z); }
double norm3(const Point3 &v) { return norm3(v[0], v[1], v[2]); }
Point3 sub(const Point3 &a, const Point3 &b) { return {a[0] - b[0], a[1] - b[1], a[2] - b[2]}; }

// numpy's pairwise_sum
double pairwise(const double *a, ll n) {
    if (n < 8) {
        double r = 0.0;
        for (ll i = 0; i < n; i++) r += a[i];
        return r;
    }
    if (n <= 128) {
        double r[8];
        for (int j = 0; j < 8; j++) r[j] = a[j];
        ll i = 8;
        for (; i < n - (n % 8); i += 8)
            for (int j = 0; j < 8; j++) r[j] += a[i + j];
        double res = ((r[0] + r[1]) + (r[2] + r[3])) + ((r[4] + r[5]) + (r[6] + r[7]));
        for (; i < n; i++) res += a[i];
        return res;
    }
    ll n2 = n / 2;
    n2 -= n2 % 8;
    return pairwise(a, n2) + pairwise(a + n2, n - n2);
}

// _ground_spec.ground_touch_tol
double ground_touch_tol(const std::vector<Point3> &pl) {
    std::vector<double> l;
    for (std::size_t k = 0; k + 1 < pl.size(); k++) {
        Point3 d = sub(pl[k + 1], pl[k]);
        double sq[3] = {d[0] * d[0], d[1] * d[1], d[2] * d[2]};
        l.push_back(std::sqrt(pairwise(sq, 3)));
    }
    double length = pairwise(l.data(), static_cast<ll>(l.size()));
    return 1e-6 * std::max(length, 1e-30);
}

// np.round(x, 9)
double round9(double x) {
    double y = x * 1e9;
    return std::nearbyint(y) / 1e9;
}

// `_junction_rule.coincident_end_groups`
std::vector<std::vector<std::pair<ll, bool>>> coincident_end_groups(const std::vector<std::vector<Point3>> &pls) {
    std::vector<std::pair<ll, bool>> labels;
    std::vector<Point3> points;
    for (std::size_t i = 0; i < pls.size(); i++) {
        labels.push_back({static_cast<ll>(i), true});
        points.push_back(pls[i].front());
        labels.push_back({static_cast<ll>(i), false});
        points.push_back(pls[i].back());
    }
    const double cell_size = 2.0 * kJunctionTol;
    std::vector<std::array<ll, 3>> cells;
    for (const Point3 &p : points) {
        std::array<ll, 3> c;
        for (int k = 0; k < 3; k++) {
            double q = std::floor(p[k] / cell_size);
            if (!(std::fabs(q) < 281474976710656.0)) decline("a wire end past the junction grid's range");
            c[k] = static_cast<ll>(q);
        }
        cells.push_back(c);
    }
    std::map<std::array<ll, 3>, std::vector<std::size_t>> buckets;
    std::vector<std::size_t> rep;
    for (std::size_t i = 0; i < points.size(); i++) {
        std::vector<std::size_t> candidates;
        for (ll dx = -1; dx <= 1; dx++)
            for (ll dy = -1; dy <= 1; dy++)
                for (ll dz = -1; dz <= 1; dz++) {
                    auto it = buckets.find({cells[i][0] + dx, cells[i][1] + dy, cells[i][2] + dz});
                    if (it != buckets.end()) candidates.insert(candidates.end(), it->second.begin(), it->second.end());
                }
        std::sort(candidates.begin(), candidates.end());
        long long hit = -1;
        for (std::size_t j : candidates)
            if (norm3(sub(points[i], points[j])) <= kJunctionTol) {
                hit = static_cast<long long>(j);
                break;
            }
        if (hit < 0) {
            rep.push_back(i);
            buckets[cells[i]].push_back(i);
        } else {
            rep.push_back(static_cast<std::size_t>(hit));
        }
    }
    std::vector<std::pair<std::size_t, std::vector<std::pair<ll, bool>>>> groups;
    for (std::size_t i = 0; i < labels.size(); i++) {
        auto it = std::find_if(groups.begin(), groups.end(), [&](const auto &g) { return g.first == rep[i]; });
        if (it == groups.end()) {
            groups.push_back({rep[i], {}});
            it = groups.end() - 1;
        }
        it->second.push_back(labels[i]);
    }
    std::vector<std::vector<std::pair<ll, bool>>> out;
    for (auto &g : groups)
        if (g.second.size() >= 2) out.push_back(g.second);
    return out;
}

// `_wire_spec.check_junction_coincidence`
void check_junction_coincidence(const std::vector<std::vector<Point3>> &pls,
                                const std::vector<std::vector<ll>> &npe,
                                const std::vector<std::vector<std::pair<ll, bool>>> &junctions) {
    for (std::size_t j = 0; j < junctions.size(); j++) {
        const auto &group = junctions[j];
        std::vector<Point3> anchors;
        double min_len = std::numeric_limits<double>::infinity();
        for (const auto &m : group) {
            const auto &pl = pls[static_cast<std::size_t>(m.first)];
            const auto &n = npe[static_cast<std::size_t>(m.first)];
            Point3 anchor, edge;
            ll count;
            if (m.second) {
                anchor = pl.front();
                edge = sub(pl[1], pl[0]);
                count = n.front();
            } else {
                anchor = pl.back();
                edge = sub(pl[pl.size() - 1], pl[pl.size() - 2]);
                count = n.back();
            }
            anchors.push_back(anchor);
            min_len = std::min(min_len, norm3(edge) / static_cast<double>(std::max<ll>(count, 1)));
        }
        double tol = std::max(kCoincidenceFloor, 1e-3 * min_len);
        for (std::size_t k = 1; k < group.size(); k++) {
            double d = norm3(sub(anchors[k], anchors[0]));
            if (d > tol) decline("junction members that do not coincide (the refusal's wording is not copied)");
        }
    }
}

// --- the NEC-4 dialect's junction (momwire#1644) ------------------------------

// `_sinusoidal_nec4.solved_nodes`
std::vector<std::size_t> nec4_solved_nodes(const Front &f) {
    const auto &gr = f.geometry.grounded_junctions;
    std::vector<std::size_t> out;
    for (std::size_t j = 0; j < f.junctions.size(); j++) {
        const auto &members = f.junctions[j];
        if (std::find(gr.begin(), gr.end(), static_cast<ll>(j)) != gr.end() || members.size() < 2) continue;
        std::set<double> radii;
        for (const auto &mb : members) radii.insert(f.radius_per_wire[static_cast<std::size_t>(mb.first)]);
        if (members.size() >= 3 || radii.size() > 1) out.push_back(j);
    }
    return out;
}

// `SinusoidalSolver._nec4_end_factors` + `_sinusoidal_nec4.end_cap_x` +
// `_sinusoidal_nec4.basis_coefs_per_end`, numpy's elementwise order.
void nec4_basis(Front &f, const std::vector<double> &a, const std::vector<std::size_t> &nodes) {
    const Geometry &g = f.geometry;
    const double k = f.k;
    const std::size_t n = static_cast<std::size_t>(g.n_segs);
    std::vector<double> base(n), am, ap;
    for (std::size_t i = 0; i < n; i++) {
        const double ka = k * a[i];
        base[i] = 1.0 / (std::log(2.0 / ka) - kEulerGamma);
    }
    am = ap = base;
    for (std::size_t j : nodes) {
        std::vector<std::pair<std::size_t, bool>> segs;
        std::vector<std::array<double, 3>> dirs;
        std::vector<double> radii, lengths;
        for (const auto &mb : f.junctions[j]) {
            const std::size_t w = static_cast<std::size_t>(mb.first);
            const std::size_t sg = static_cast<std::size_t>(mb.second ? g.wire_first[w] : g.wire_last[w]);
            const Point3 &t = g.seg_tangents[sg];
            if (mb.second)
                dirs.push_back({t[0], t[1], t[2]});
            else
                dirs.push_back({-t[0], -t[1], -t[2]});
            segs.push_back({sg, mb.second});
            radii.push_back(f.radius_per_wire[w]);
            lengths.push_back(g.seg_h[sg]);
        }
        std::vector<double> fac;
        try {
            // the first member's LOG-RULE factor (`base`), not one an earlier
            // node already replaced at this segment's other end
            fac = junction_charge_factors(k, dirs, radii, lengths, base[segs[0].first]);
        } catch (const std::domain_error &e) {
            fail(e.what());
        } catch (const std::runtime_error &) {
            continue;  // a singular node keeps the log rule (Python's ZeroDivisionError)
        }
        if (!std::all_of(fac.begin(), fac.end(), [](double v) { return std::isfinite(v); })) continue;
        for (std::size_t q = 0; q < segs.size(); q++) (segs[q].second ? am : ap)[segs[q].first] = fac[q];
    }
    // end_cap_x
    std::vector<double> X(n), kd(n), sin_kd(n), cos_kd(n), s2(n), c2(n), s4(n), recip(n), tanh_(n);
    const double half_k = 0.5 * k;
    for (std::size_t i = 0; i < n; i++) {
        const bool free_m = g.nm_count[i] == 0 && !g.ground_minus[i] && !g.crossing_minus[i];
        const bool free_p = g.np_count[i] == 0 && !g.ground_plus[i] && !g.crossing_plus[i];
        X[i] = (free_m || free_p) ? half_k * a[i] : 0.0;
        kd[i] = k * g.seg_h[i];
        sin_kd[i] = std::sin(kd[i]);
        cos_kd[i] = std::cos(kd[i]);
        s2[i] = std::sin(0.5 * kd[i]);
        c2[i] = std::cos(0.5 * kd[i]);
        s4[i] = std::sin(0.25 * kd[i]);
        const double rs4 = std::sin(0.25 * kd[i]);
        recip[i] = rs4 * rs4 / (std::sin(0.5 * kd[i]) * std::cos(0.5 * kd[i]));
        tanh_[i] = s2[i] / c2[i];
    }
    // the neighbours' factors at the shared node
    std::vector<double> a_nm(g.nm_basis.size()), a_np(g.np_basis.size());
    for (std::size_t e = 0; e < g.nm_basis.size(); e++) {
        const std::size_t sg = static_cast<std::size_t>(g.nm_seg[e]);
        a_nm[e] = g.nm_sigma[e] > 0 ? ap[sg] : am[sg];
    }
    for (std::size_t e = 0; e < g.np_basis.size(); e++) {
        const std::size_t sg = static_cast<std::size_t>(g.np_seg[e]);
        a_np[e] = g.np_sigma[e] > 0 ? am[sg] : ap[sg];
    }
    std::vector<double> Pm(n, 0.0), Pp(n, 0.0);
    for (std::size_t e = 0; e < g.nm_basis.size(); e++)
        Pm[static_cast<std::size_t>(g.nm_basis[e])] += a_nm[e] * tanh_[static_cast<std::size_t>(g.nm_seg[e])];
    for (std::size_t e = 0; e < g.np_basis.size(); e++)
        Pp[static_cast<std::size_t>(g.np_basis[e])] += -(a_np[e] * tanh_[static_cast<std::size_t>(g.np_seg[e])]);
    const bool any_gm = std::find(g.ground_minus.begin(), g.ground_minus.end(), true) != g.ground_minus.end();
    const bool any_gp = std::find(g.ground_plus.begin(), g.ground_plus.end(), true) != g.ground_plus.end();
    if (any_gm)
        for (std::size_t i = 0; i < n; i++) Pm[i] = Pm[i] + (g.ground_minus[i] ? am[i] * tanh_[i] : 0.0);
    if (any_gp)
        for (std::size_t i = 0; i < n; i++) Pp[i] = Pp[i] - (g.ground_plus[i] ? ap[i] * tanh_[i] : 0.0);
    std::vector<bool> has_minus(n), has_plus(n);
    for (std::size_t i = 0; i < n; i++) {
        has_minus[i] = g.nm_count[i] > 0 || g.ground_minus[i] || g.crossing_minus[i];
        has_plus[i] = g.np_count[i] > 0 || g.ground_plus[i] || g.crossing_plus[i];
    }
    std::vector<double> Qm(n), Qp(n), A(n), B(n), C(n), AC(n), sin_safe(n), omc(n), s4sq(n);
    for (std::size_t i = 0; i < n; i++) {
        const double D = (Pm[i] * Pp[i] + am[i] * ap[i]) * sin_kd[i] + (Pm[i] * ap[i] - Pp[i] * am[i]) * cos_kd[i];
        const double D_safe = D != 0 ? D : 1.0;
        sin_safe[i] = sin_kd[i] != 0 ? sin_kd[i] : 1.0;
        omc[i] = 2.0 * s2[i] * s2[i];
        Qm[i] = (ap[i] * omc[i] - Pp[i] * sin_kd[i]) / D_safe;
        Qp[i] = (-am[i] * omc[i] - Pm[i] * sin_kd[i]) / D_safe;
        A[i] = -1.0;
        B[i] = (am[i] * Qm[i] + ap[i] * Qp[i]) * s2[i] / sin_safe[i];
        C[i] = (am[i] * Qm[i] - ap[i] * Qp[i]) * c2[i] / sin_safe[i];
        const double dP = Pm[i] * ap[i] - Pp[i] * am[i];
        s4sq[i] = s4[i] * s4[i];
        AC[i] = (4.0 * am[i] * ap[i] * s2[i] * s4sq[i] - 2.0 * dP * s4sq[i] - 2.0 * s2[i] * c2[i] * Pm[i] * Pp[i] +
                 2.0 * dP * s2[i] * s2[i]) /
                D_safe;
    }
    bool any_branch = false;
    for (std::size_t i = 0; i < n; i++) any_branch = any_branch || !(has_minus[i] && has_plus[i]);
    if (any_branch) {
        for (std::size_t i = 0; i < n; i++) {
            const bool only_plus = has_plus[i] && !has_minus[i];
            const bool only_minus = has_minus[i] && !has_plus[i];
            const bool iso = !has_minus[i] && !has_plus[i];
            const double x = X[i];
            const double denom_x = cos_kd[i] - x * sin_kd[i];
            const double dxs = denom_x != 0 ? denom_x : 1.0;
            const double gap_x = omc[i] + x * sin_kd[i];
            const double cap_s = s2[i] + x * c2[i];
            const double cap_c = c2[i] - x * s2[i];
            const double qpe1_d = (ap[i] + x * Pp[i]) * sin_kd[i] + (ap[i] * x - Pp[i]) * cos_kd[i];
            const double qpe1_s = qpe1_d != 0 ? qpe1_d : 1.0;
            const double Q_plus_e1 = -gap_x / qpe1_s;
            const double B_e1 = (s2[i] + ap[i] * Q_plus_e1 * cap_c) / dxs;
            const double C_e1 = (c2[i] + ap[i] * Q_plus_e1 * cap_s) / dxs;
            const double qme2_d = (am[i] - x * Pm[i]) * sin_kd[i] + (am[i] * x + Pm[i]) * cos_kd[i];
            const double qme2_s = qme2_d != 0 ? qme2_d : 1.0;
            const double Q_minus_e2 = gap_x / qme2_s;
            const double B_e2 = (-s2[i] + am[i] * Q_minus_e2 * cap_c) / dxs;
            const double C_e2 = (c2[i] - am[i] * Q_minus_e2 * cap_s) / dxs;
            const double iso_den = c2[i] - x * s2[i];
            const double iso_s = iso_den != 0 ? iso_den : 1.0;
            const double C_iso = 1.0 / iso_s;
            if (only_plus || iso) Qm[i] = 0.0;
            if (only_minus) Qm[i] = Q_minus_e2;
            if (only_minus || iso) Qp[i] = 0.0;
            if (only_plus) Qp[i] = Q_plus_e1;
            if (only_plus) B[i] = B_e1;
            if (only_minus) B[i] = B_e2;
            if (iso) B[i] = 0.0;
            if (only_plus) C[i] = C_e1;
            if (only_minus) C[i] = C_e2;
            if (iso) C[i] = C_iso;
            const double cos_gap_e = 2.0 * std::sin(0.75 * kd[i]) * s4[i] + x * sin_kd[i];
            const double AC_e1 = (cos_gap_e + ap[i] * Q_plus_e1 * cap_s) / dxs;
            const double AC_e2 = (cos_gap_e - am[i] * Q_minus_e2 * cap_s) / dxs;
            const double AC_iso = (2.0 * s4sq[i] + x * s2[i]) / iso_s;
            if (only_plus) AC[i] = AC_e1;
            if (only_minus) AC[i] = AC_e2;
            if (iso) AC[i] = AC_iso;
        }
    }
    if (any_gm)
        for (std::size_t i = 0; i < n; i++) {
            const double Qg = g.ground_minus[i] ? am[i] * Qm[i] : 0.0;
            A[i] = A[i] + Qg / sin_safe[i];
            B[i] = B[i] - Qg / (2.0 * c2[i]);
            C[i] = C[i] - Qg / (2.0 * s2[i]);
            AC[i] = AC[i] + Qg * recip[i];
        }
    if (any_gp)
        for (std::size_t i = 0; i < n; i++) {
            const double Qg = g.ground_plus[i] ? ap[i] * Qp[i] : 0.0;
            A[i] = A[i] - Qg / sin_safe[i];
            B[i] = B[i] - Qg / (2.0 * c2[i]);
            C[i] = C[i] + Qg / (2.0 * s2[i]);
            AC[i] = AC[i] - Qg * recip[i];
        }
    std::vector<ll> all_seg, all_basis;
    std::vector<double> all_A, all_B, all_C, all_AC;
    std::vector<int8_t> all_sigma;
    for (std::size_t i = 0; i < n; i++) {
        all_seg.push_back(static_cast<ll>(i));
        all_basis.push_back(static_cast<ll>(i));
        all_A.push_back(A[i]);
        all_B.push_back(B[i]);
        all_C.push_back(C[i]);
        all_AC.push_back(AC[i]);
        all_sigma.push_back(1);
    }
    for (std::size_t e = 0; e < g.nm_basis.size(); e++) {
        const std::size_t sg = static_cast<std::size_t>(g.nm_seg[e]);
        const double aq = a_nm[e] * Qm[static_cast<std::size_t>(g.nm_basis[e])];
        all_seg.push_back(g.nm_seg[e]);
        all_basis.push_back(g.nm_basis[e]);
        all_A.push_back(aq / sin_kd[sg]);
        all_B.push_back(aq / (2.0 * c2[sg]));
        all_C.push_back(-aq / (2.0 * s2[sg]));
        all_AC.push_back(aq * recip[sg]);
        all_sigma.push_back(g.nm_sigma[e]);
    }
    for (std::size_t e = 0; e < g.np_basis.size(); e++) {
        const std::size_t sg = static_cast<std::size_t>(g.np_seg[e]);
        const double aq = a_np[e] * Qp[static_cast<std::size_t>(g.np_basis[e])];
        all_seg.push_back(g.np_seg[e]);
        all_basis.push_back(g.np_basis[e]);
        all_A.push_back(-aq / sin_kd[sg]);
        all_B.push_back(aq / (2.0 * c2[sg]));
        all_C.push_back(aq / (2.0 * s2[sg]));
        all_AC.push_back(-aq * recip[sg]);
        all_sigma.push_back(g.np_sigma[e]);
    }
    std::vector<std::size_t> order(all_seg.size());
    std::iota(order.begin(), order.end(), 0);
    std::stable_sort(order.begin(), order.end(), [&](std::size_t x, std::size_t y) { return all_seg[x] < all_seg[y]; });
    SegView &v = f.basis;
    v.starts.assign(n + 1, 0);
    for (std::size_t i = 0; i < n; i++) v.starts[i + 1] = v.starts[i] + 1 + g.nm_count[i] + g.np_count[i];
    for (std::size_t o : order) {
        v.jbasis.push_back(all_basis[o]);
        v.A.push_back(cd(all_A[o], 0.0));
        v.B.push_back(cd(all_B[o], 0.0));
        v.C.push_back(cd(all_C[o], 0.0));
        v.AC.push_back(cd(all_AC[o], 0.0));
        v.sigma.push_back(all_sigma[o]);
    }
    f.nec4 = true;
}


// `_wire_spec.drop_duplicated_wires("SinusoidalSolver", ...)` (momwire#1042 /
// #1333) over the NEC-4.2 build's arguments, which carry no junction-indexed
// ports and no node drive: `out` is `in` with each wire listed twice dropped
// and every wire-indexed argument remapped (`in` itself when nothing is), and
// the dedup says how. Refuses (FrontError) in the constructor's words.
nec4::WireDedup drop_duplicated_wires(const nec4::SolverArgs &in, nec4::SolverArgs &out) {
    static const char *const kFamily = "SinusoidalSolver";
    static const char *const kCoincidentRefusal =
        "this family cannot solve coincident conductors: its answer for them "
        "depends on the CPU's rounding (momwire#1333). Solve the deck with "
        "BSplineSolver, or join both copies to the same wires at each end so "
        "they merge as one";
    nec4::WireDedup dd;
    out = in;
    const auto &pls = in.wires;
    const std::size_t n = pls.size();
    // `find_duplicated_wires`
    bool any = false;
    double scale = 1.0;
    for (const auto &pl : pls)
        for (const Point3 &p : pl) {
            any = true;
            for (double c : p) scale = std::max(scale, std::fabs(c));
        }
    if (!any) return dd;
    const double tol = 1e-12 * scale;
    struct Dup {
        std::size_t copy, first;
        bool rev;
    };
    std::vector<Dup> dups;
    typedef std::vector<std::array<double, 3>> Key;
    std::map<Key, std::size_t> seen;
    for (std::size_t i = 0; i < n; i++) {
        Key fwd, rev;
        for (const Point3 &p : pls[i]) fwd.push_back({round9(p[0]), round9(p[1]), round9(p[2])});
        rev.assign(fwd.rbegin(), fwd.rend());
        const Key &key = std::min(fwd, rev);
        auto it = seen.find(key);
        if (it == seen.end()) {
            seen[key] = i;
            continue;
        }
        const auto &twin = pls[it->second];
        const auto &pl = pls[i];
        if (twin.size() != pl.size()) continue;
        // `np.max(np.abs(twin - pl)) <= tol`: a NaN difference fails it
        bool fwd_ok = true, rev_ok = true;
        double fmax = 0, rmax = 0;
        for (std::size_t v = 0; v < twin.size(); v++)
            for (int c = 0; c < 3; c++) {
                const double df = std::fabs(twin[v][c] - pl[v][c]);
                const double dr = std::fabs(twin[v][c] - pl[pl.size() - 1 - v][c]);
                if (std::isnan(df)) fwd_ok = false;
                if (std::isnan(dr)) rev_ok = false;
                fmax = std::max(fmax, df);
                rmax = std::max(rmax, dr);
            }
        if (fwd_ok && fmax <= tol)
            dups.push_back({i, it->second, false});
        else if (rev_ok && rmax <= tol)
            dups.push_back({i, it->second, true});
    }
    if (dups.empty()) return dd;

    // `_merge_connected_alike(wires, dups, junctions)`
    std::vector<std::vector<std::pair<ll, bool>>> groups;  // (wire, at start)
    if (in.junctions.empty()) {
        groups = coincident_end_groups(pls);
    } else {
        for (const auto &g : in.junctions) {
            std::vector<std::pair<ll, bool>> m;
            for (const auto &e : g) m.push_back({e.polyline, e.start});
            groups.push_back(m);
        }
    }
    std::map<std::pair<ll, bool>, ll> node;
    for (std::size_t j = 0; j < groups.size(); j++)
        for (const auto &m : groups[j]) node[m] = static_cast<ll>(j);
    const auto at = [&](ll w, bool start) {
        auto it = node.find({w, start});
        return it == node.end() ? ll(-1) : it->second;
    };
    // the ends named in the class's FIRST wire's direction
    const auto joined = [&](std::size_t w, bool rev) {
        const ll s = static_cast<ll>(w);
        return std::make_pair(at(s, !rev), at(s, rev));
    };
    std::vector<std::size_t> class_order;
    std::map<std::size_t, std::vector<std::pair<std::size_t, bool>>> classes;
    for (const Dup &d : dups) {
        auto it = classes.find(d.first);
        if (it == classes.end()) {
            class_order.push_back(d.first);
            it = classes.emplace(d.first, std::vector<std::pair<std::size_t, bool>>{{d.first, false}}).first;
        }
        it->second.emplace_back(d.copy, d.rev);
    }
    std::map<std::size_t, std::pair<std::size_t, bool>> removed;
    std::vector<std::size_t> removed_order;
    std::vector<std::pair<std::size_t, std::size_t>> kept_coincident;
    for (std::size_t first : class_order) {
        struct Rep {
            std::size_t w;
            bool rev;
            std::pair<ll, ll> key;
        };
        std::vector<Rep> reps;
        for (const auto &member : classes[first]) {
            const auto key = joined(member.first, member.second);
            const Rep *match = nullptr;
            for (const Rep &r : reps)
                if (r.key == key) {
                    match = &r;
                    break;
                }
            if (match == nullptr) {
                if (!reps.empty()) kept_coincident.push_back({member.first, reps.back().w});
                reps.push_back({member.first, member.second, key});
            } else {
                removed[member.first] = {match->w, member.second != match->rev};
                removed_order.push_back(member.first);
            }
        }
    }
    if (!kept_coincident.empty())
        fail(std::string(kFamily) + ": wires " + std::to_string(kept_coincident[0].second) + " and " +
             std::to_string(kept_coincident[0].first) +
             " coincide, vertex for vertex, but are joined to different wires, so they are two "
             "conductors written on one path rather than a wire listed twice, and " +
             kCoincidentRefusal + " - momwire#1333");
    if (removed.empty()) return dd;
    const auto refuse = [](std::size_t r, std::size_t twin, const std::string &why) {
        fail(std::string(kFamily) + ": wires " + std::to_string(twin) + " and " + std::to_string(r) +
             " are the same wire listed twice, and " + why + ", so they cannot be merged into one conductor. "
             "Delete wire " + std::to_string(r) + ", or make the two copies identical - momwire#1042");
    };
    // `_same` for a float entry (NaN equals NaN)
    const auto same = [](double a, double b) { return (std::isnan(a) && std::isnan(b)) || a == b; };
    for (std::size_t r : removed_order) {  // the dict's insertion order
        const std::size_t twin = removed[r].first;
        const bool rev = removed[r].second;
        std::vector<ll> a = in.n_per_edge[twin], b = in.n_per_edge[r];
        if (rev) std::reverse(b.begin(), b.end());
        if (a != b) refuse(r, twin, "their segment counts differ");
        if (!in.uniform_radius && !same(in.radii.at(r), in.radii.at(twin)))
            refuse(r, twin, "their wire_radius entries differ");
        if (in.has_conductivity && !same(in.conductivity.at(r), in.conductivity.at(twin)))
            refuse(r, twin, "their wire_conductivity entries differ");
        if (in.has_insulation) decline("an insulated wire listed twice");
        if (in.has_distributed) {
            const bool pr = in.distributed_present.at(r), pt = in.distributed_present.at(twin);
            if (pr && pt) {
                const nec4::DistributedRLC &x = in.distributed.at(r), &y = in.distributed.at(twin);
                for (double v : {x.r, x.l, x.c, y.r, y.l, y.c})
                    if (std::isnan(v)) decline("a NaN distributed load on a wire listed twice");
                if (x != y) refuse(r, twin, "their distributed_rlc entries differ");
            } else if (pr != pt) {
                refuse(r, twin, "their distributed_rlc entries differ");
            }
        }
        for (const auto &fd : in.feeds)
            if (fd.polyline == static_cast<ll>(r) || fd.polyline == static_cast<ll>(twin))
                refuse(r, twin, "feeds places a site on wire " + std::to_string(fd.polyline));
    }
    // declared junctions without the dropped members (no junction-indexed
    // ports, so none is refused for it); inferred ones stay inferred
    std::vector<std::vector<nec4::JunctionEnd>> junctions;
    for (const auto &g : in.junctions) {
        std::vector<nec4::JunctionEnd> kept;
        for (const auto &m : g)
            if (!removed.count(static_cast<std::size_t>(m.polyline))) kept.push_back(m);
        if (!kept.empty()) junctions.push_back(kept);
    }
    dd.active = true;
    dd.n_input = static_cast<ll>(n);
    std::vector<ll> new_index(n, -1);
    for (std::size_t w = 0; w < n; w++)
        if (!removed.count(w)) {
            new_index[w] = static_cast<ll>(dd.kept.size());
            dd.kept.push_back(static_cast<ll>(w));
        }
    for (const auto &kv : removed)  // sorted by copy
        dd.removed.push_back({static_cast<ll>(kv.first), static_cast<ll>(kv.second.first), kv.second.second});
    out.wires.clear();
    out.n_per_edge.clear();
    if (!in.uniform_radius) out.radii.clear();
    out.conductivity.clear();
    out.distributed.clear();
    out.distributed_present.clear();
    for (ll w : dd.kept) {
        const std::size_t u = static_cast<std::size_t>(w);
        out.wires.push_back(in.wires[u]);
        out.n_per_edge.push_back(in.n_per_edge[u]);
        if (!in.uniform_radius) out.radii.push_back(in.radii.at(u));
        if (in.has_conductivity) out.conductivity.push_back(in.conductivity.at(u));
        if (in.has_distributed) {
            out.distributed.push_back(in.distributed.at(u));
            out.distributed_present.push_back(in.distributed_present.at(u));
        }
    }
    for (auto &fd : out.feeds) fd.polyline = new_index[static_cast<std::size_t>(fd.polyline)];
    out.junctions.clear();
    for (const auto &g : junctions) {
        std::vector<nec4::JunctionEnd> mm;
        for (const auto &m : g) mm.push_back({new_index[static_cast<std::size_t>(m.polyline)], m.start});
        out.junctions.push_back(mm);
    }
    return dd;
}

}  // namespace

Front front(const nec4::SolverArgs &in) {
    Front f;
    if (in.wires.empty()) fail("wires must be non-empty");
    // momwire#1042: a wire listed twice is solved once; everything below
    // reads the solved wires
    f.dedup = drop_duplicated_wires(in, f.args);
    const nec4::SolverArgs &args = f.args;
    const auto &pls = args.wires;
    if (args.ground.kind == nec4::Ground::finite_fast) decline("a reflection-coefficient ground");
    const bool has_ground = args.ground.kind != nec4::Ground::none;
    const double gz = 0.0;
    if (has_ground)
        for (const auto &pl : pls) {
            double mn = pl[0][2];
            for (const Point3 &p : pl) mn = std::min(mn, p[2]);
            if (mn < gz - ground_touch_tol(pl)) decline("a wire below the ground plane (momwire#1607 B6)");
        }

    // the constructor's constants
    double c = 1 / std::sqrt(kEps * kMu);
    double freq = c / args.wavelength;
    double omega = 2 * kPi * freq;
    f.c = c;
    f.freq = freq;
    f.omega = omega;
    f.k = omega / c;
    f.eta = std::sqrt(kMu / kEps);
    const double k = f.k;

    // normalize_wire_radius
    const std::size_t n_w = pls.size();
    f.radius_per_wire = args.uniform_radius ? std::vector<double>(n_w, args.radii[0]) : args.radii;
    for (double r : f.radius_per_wire)
        if (!(std::isfinite(r) && r > 0.0)) decline("a non-positive or non-finite radius");
    f.uniform_radius = std::all_of(f.radius_per_wire.begin(), f.radius_per_wire.end(),
                                   [&](double r) { return r == f.radius_per_wire[0]; });
    for (std::size_t i = 0; i < n_w; i++)
        if (args.n_per_edge[i].size() + 1 != pls[i].size()) decline("an edge count that does not match the polyline");
    for (const auto &fd : args.feeds)
        if (fd.polyline < 0 || fd.polyline >= static_cast<ll>(n_w)) decline("a feed on no wire");

    // normalize_junctions
    if (args.junctions.empty()) {
        f.junctions = coincident_end_groups(pls);
    } else {
        for (const auto &g : args.junctions) {
            std::vector<std::pair<ll, bool>> m;
            for (const auto &e : g) m.push_back({e.polyline, e.start});
            f.junctions.push_back(m);
        }
    }
    check_junction_coincidence(pls, args.n_per_edge, f.junctions);

    // _build_geometry
    Geometry &g = f.geometry;
    ll running = 0;
    for (std::size_t w = 0; w < n_w; w++) {
        const auto &pl = pls[w];
        ll wire_first = running;
        for (std::size_t e = 0; e + 1 < pl.size(); e++) {
            const Point3 &p0 = pl[e], &p1 = pl[e + 1];
            Point3 vec = sub(p1, p0);
            double edge_len = norm3(vec);
            if (edge_len < 1e-15) fail("wire " + std::to_string(w) + " edge " + std::to_string(e) + " has zero length");
            Point3 tan{vec[0] / edge_len, vec[1] / edge_len, vec[2] / edge_len};
            ll n_e = args.n_per_edge[w][e];
            double h_e = edge_len / static_cast<double>(n_e);
            const double step = 1.0 / static_cast<double>(n_e);  // np.linspace(0, 1, n_e + 1)
            std::vector<double> frac(static_cast<std::size_t>(n_e + 1));
            for (ll i = 0; i <= n_e; i++) frac[static_cast<std::size_t>(i)] = static_cast<double>(i) * step + 0.0;
            frac[static_cast<std::size_t>(n_e)] = 1.0;
            std::vector<Point3> pts;
            for (double t : frac) pts.push_back({p0[0] + t * vec[0], p0[1] + t * vec[1], p0[2] + t * vec[2]});
            for (ll i = 0; i < n_e; i++) {
                const Point3 &l = pts[static_cast<std::size_t>(i)], &r = pts[static_cast<std::size_t>(i + 1)];
                g.seg_l.push_back(l);
                g.seg_r.push_back(r);
                g.seg_centers.push_back({0.5 * (l[0] + r[0]), 0.5 * (l[1] + r[1]), 0.5 * (l[2] + r[2])});
                g.seg_tangents.push_back(tan);
                g.seg_h.push_back(h_e);
            }
            running += n_e;
        }
        g.wire_last.push_back(running - 1);
        g.wire_first.push_back(wire_first);
    }
    const ll n_segs = static_cast<ll>(g.seg_l.size());
    g.n_segs = n_segs;
    for (std::size_t w = 0; w < n_w; w++) {
        ll first = g.wire_first[w], last = g.wire_last[w];
        if (last > first) {
            for (ll i = first + 1; i <= last; i++) {
                g.nm_basis.push_back(i);
                g.nm_seg.push_back(i - 1);
                g.nm_sigma.push_back(1);
            }
            for (ll i = first; i < last; i++) {
                g.np_basis.push_back(i);
                g.np_seg.push_back(i + 1);
                g.np_sigma.push_back(1);
            }
        }
    }
    std::set<ll> grounded;
    if (has_ground) {
        for (std::size_t j = 0; j < f.junctions.size(); j++) {
            const auto &m0 = f.junctions[j][0];
            const auto &pl0 = pls[static_cast<std::size_t>(m0.first)];
            const Point3 &pt = m0.second ? pl0.front() : pl0.back();
            if (std::fabs(pt[2] - gz) <= ground_touch_tol(pl0)) grounded.insert(static_cast<ll>(j));
        }
    }
    for (std::size_t j = 0; j < f.junctions.size(); j++) {
        if (grounded.count(static_cast<ll>(j))) continue;
        std::vector<std::pair<ll, bool>> members;  // (seg, side is L)
        for (const auto &m : f.junctions[j]) {
            if (m.second) {
                members.push_back({g.wire_first[static_cast<std::size_t>(m.first)], true});
            } else {
                members.push_back({g.wire_last[static_cast<std::size_t>(m.first)], false});
            }
        }
        for (std::size_t a = 0; a < members.size(); a++)
            for (std::size_t b = 0; b < members.size(); b++) {
                if (a == b) continue;
                if (members[a].second) {
                    g.nm_basis.push_back(members[a].first);
                    g.nm_seg.push_back(members[b].first);
                    g.nm_sigma.push_back(!members[b].second ? 1 : -1);
                } else {
                    g.np_basis.push_back(members[a].first);
                    g.np_seg.push_back(members[b].first);
                    g.np_sigma.push_back(members[b].second ? 1 : -1);
                }
            }
    }
    // feeds
    for (const auto &fd : args.feeds) {
        ll first = g.wire_first[static_cast<std::size_t>(fd.polyline)];
        ll last = g.wire_last[static_cast<std::size_t>(fd.polyline)];
        std::vector<double> h(g.seg_h.begin() + first, g.seg_h.begin() + last + 1);
        std::vector<double> centers(h.size());
        double cum = 0;
        for (std::size_t i = 0; i < h.size(); i++) {
            cum = i == 0 ? h[0] : cum + h[i];
            centers[i] = cum - 0.5 * h[i];
        }
        double total = pairwise(h.data(), static_cast<ll>(h.size()));
        double arc = fd.arclength;
        arc = std::min(std::max(arc, 0.0), total);  // Python's min / max
        std::size_t pick = 0;
        if (centers.size() > 1) {
            std::vector<double> dist(centers.size());
            for (std::size_t i = 0; i < centers.size(); i++) dist[i] = std::fabs(centers[i] - arc);
            std::vector<std::size_t> order(centers.size());
            std::iota(order.begin(), order.end(), 0);
            std::stable_sort(order.begin(), order.end(), [&](std::size_t x, std::size_t y) { return dist[x] < dist[y]; });
            std::size_t near = order[0], next = order[1];
            double margin = dist[next] - dist[near];
            bool tie = margin <= kAmbiguityTolFrac * total;
            pick = tie ? std::min(near, next) : near;
        }
        g.feed_segs.push_back(first + static_cast<ll>(pick));
        g.feed_xi.push_back(arc - centers[pick]);
        g.feed_sites.push_back({fd.polyline, fd.arclength, arc, centers[pick]});
    }
    g.has_feed_seg = !g.feed_segs.empty();
    if (g.has_feed_seg) g.feed_seg = g.feed_segs[0];
    g.nm_count.assign(static_cast<std::size_t>(n_segs), 0);
    g.np_count.assign(static_cast<std::size_t>(n_segs), 0);
    for (ll b : g.nm_basis) g.nm_count[static_cast<std::size_t>(b)]++;
    for (ll b : g.np_basis) g.np_count[static_cast<std::size_t>(b)]++;
    g.ground_minus.assign(static_cast<std::size_t>(n_segs), false);
    g.ground_plus.assign(static_cast<std::size_t>(n_segs), false);
    g.crossing_minus.assign(static_cast<std::size_t>(n_segs), false);
    g.crossing_plus.assign(static_cast<std::size_t>(n_segs), false);
    if (has_ground) {
        std::set<std::pair<ll, bool>> junctioned;
        for (const auto &jn : f.junctions)
            for (const auto &m : jn) junctioned.insert(m);
        for (std::size_t w = 0; w < n_w; w++) {
            const auto &pl = pls[w];
            double tol = ground_touch_tol(pl);
            bool start_touch = std::fabs(pl.front()[2] - gz) <= tol;
            bool end_touch = std::fabs(pl.back()[2] - gz) <= tol;
            if (start_touch && !junctioned.count({static_cast<ll>(w), true}))
                g.ground_minus[static_cast<std::size_t>(g.wire_first[w])] = true;
            if (end_touch && !junctioned.count({static_cast<ll>(w), false}))
                g.ground_plus[static_cast<std::size_t>(g.wire_last[w])] = true;
        }
        for (ll s = 0; s < n_segs; s++) {
            const std::size_t i = static_cast<std::size_t>(s);
            double lhs = std::fabs(g.seg_l[i][2] - gz) + std::fabs(g.seg_r[i][2] - gz);
            if (lhs <= 2e-6 * std::max(g.seg_h[i], 1e-30))
                fail("segment(s) lying in the ground plane (both endpoints at ground_z) — degenerate over a "
                     "conducting ground");
        }
        for (ll j : grounded)
            for (const auto &m : f.junctions[static_cast<std::size_t>(j)]) {
                if (m.second) {
                    g.ground_minus[static_cast<std::size_t>(g.wire_first[static_cast<std::size_t>(m.first)])] = true;
                } else {
                    g.ground_plus[static_cast<std::size_t>(g.wire_last[static_cast<std::size_t>(m.first)])] = true;
                }
            }
    }
    g.grounded_junctions.assign(grounded.begin(), grounded.end());

    // _basis_coefs(geom, k), a real k
    const std::size_t n = static_cast<std::size_t>(n_segs);
    std::vector<double> a(n);
    if (f.uniform_radius) {
        std::fill(a.begin(), a.end(), f.radius_per_wire[0]);
    } else {
        for (std::size_t w = 0; w < n_w; w++)
            for (ll s = g.wire_first[w]; s <= g.wire_last[w]; s++) a[static_cast<std::size_t>(s)] = f.radius_per_wire[w];
    }
    if (args.junction_rule_nec42) {
        // momwire#1644: `_nec4_live` -- the deck has a node the rule solves
        // (a refl-coef ground or a buried wire is declined above).
        const std::vector<std::size_t> nodes = nec4_solved_nodes(f);
        if (!nodes.empty()) {
            nec4_basis(f, a, nodes);
            return f;
        }
    }
    std::vector<double> a_const(n), kd(n), sin_kd(n), cos_kd(n), s2(n), c2(n), s4(n), recip(n), atom(n);
    for (std::size_t i = 0; i < n; i++) {
        double ka = k * a[i];
        a_const[i] = 1.0 / (std::log(2.0 / ka) - kEulerGamma);
        kd[i] = k * g.seg_h[i];
        sin_kd[i] = std::sin(kd[i]);
        cos_kd[i] = std::cos(kd[i]);
        s2[i] = std::sin(0.5 * kd[i]);
        c2[i] = std::cos(0.5 * kd[i]);
        s4[i] = std::sin(0.25 * kd[i]);
        double rs4 = std::sin(0.25 * kd[i]);
        recip[i] = rs4 * rs4 / (std::sin(0.5 * kd[i]) * std::cos(0.5 * kd[i]));
        atom[i] = (s2[i] / c2[i]) * a_const[i];
    }
    std::vector<double> Pm(n, 0.0), Pp(n, 0.0);
    for (std::size_t e = 0; e < g.nm_basis.size(); e++)
        Pm[static_cast<std::size_t>(g.nm_basis[e])] += atom[static_cast<std::size_t>(g.nm_seg[e])];
    for (std::size_t e = 0; e < g.np_basis.size(); e++)
        Pp[static_cast<std::size_t>(g.np_basis[e])] += -atom[static_cast<std::size_t>(g.np_seg[e])];
    const bool any_gm = std::find(g.ground_minus.begin(), g.ground_minus.end(), true) != g.ground_minus.end();
    const bool any_gp = std::find(g.ground_plus.begin(), g.ground_plus.end(), true) != g.ground_plus.end();
    if (any_gm)
        for (std::size_t i = 0; i < n; i++) Pm[i] = Pm[i] + (g.ground_minus[i] ? atom[i] : 0.0);
    if (any_gp)
        for (std::size_t i = 0; i < n; i++) Pp[i] = Pp[i] - (g.ground_plus[i] ? atom[i] : 0.0);
    std::vector<bool> has_minus(n), has_plus(n);
    for (std::size_t i = 0; i < n; i++) {
        has_minus[i] = g.nm_count[i] > 0 || g.ground_minus[i] || g.crossing_minus[i];
        has_plus[i] = g.np_count[i] > 0 || g.ground_plus[i] || g.crossing_plus[i];
    }
    std::vector<double> Qm(n), Qp(n), A(n), B(n), C(n), AC(n), sin_safe(n), omc(n), s4sq(n);
    for (std::size_t i = 0; i < n; i++) {
        const double ac = a_const[i];
        double D = (Pm[i] * Pp[i] + ac * ac) * sin_kd[i] + (Pm[i] - Pp[i]) * ac * cos_kd[i];
        double D_safe = D != 0 ? D : 1.0;
        sin_safe[i] = sin_kd[i] != 0 ? sin_kd[i] : 1.0;
        omc[i] = 2.0 * s2[i] * s2[i];
        Qm[i] = (ac * omc[i] - Pp[i] * sin_kd[i]) / D_safe;
        Qp[i] = (-ac * omc[i] - Pm[i] * sin_kd[i]) / D_safe;
        A[i] = -1.0;
        B[i] = ac * (Qm[i] + Qp[i]) * s2[i] / sin_safe[i];
        C[i] = ac * (Qm[i] - Qp[i]) * c2[i] / sin_safe[i];
        double dP = Pm[i] - Pp[i];
        s4sq[i] = s4[i] * s4[i];
        AC[i] = (4.0 * ac * ac * s2[i] * s4sq[i] - 2.0 * ac * dP * s4sq[i] - 2.0 * s2[i] * c2[i] * Pm[i] * Pp[i] +
                 2.0 * ac * dP * s2[i] * s2[i]) /
                D_safe;
    }
    bool any_branch = false;
    for (std::size_t i = 0; i < n; i++) any_branch = any_branch || !(has_minus[i] && has_plus[i]);
    if (any_branch) {
        for (std::size_t i = 0; i < n; i++) {
            const bool only_plus = has_plus[i] && !has_minus[i];
            const bool only_minus = has_minus[i] && !has_plus[i];
            const bool iso = !has_minus[i] && !has_plus[i];
            const double ac = a_const[i];
            double dxs = cos_kd[i] != 0 ? cos_kd[i] : 1.0;
            double qpe1_d = ac * sin_kd[i] - Pp[i] * cos_kd[i];
            double qpe1_s = qpe1_d != 0 ? qpe1_d : 1.0;
            double Q_plus_e1 = -omc[i] / qpe1_s;
            double B_e1 = (s2[i] + ac * Q_plus_e1 * c2[i]) / dxs;
            double C_e1 = (c2[i] + ac * Q_plus_e1 * s2[i]) / dxs;
            double qme2_d = ac * sin_kd[i] + Pm[i] * cos_kd[i];
            double qme2_s = qme2_d != 0 ? qme2_d : 1.0;
            double Q_minus_e2 = omc[i] / qme2_s;
            double B_e2 = (-s2[i] + ac * Q_minus_e2 * c2[i]) / dxs;
            double C_e2 = (c2[i] - ac * Q_minus_e2 * s2[i]) / dxs;
            double c2s = c2[i] != 0 ? c2[i] : 1.0;
            double C_iso = 1.0 / c2s;
            if (only_plus || iso) Qm[i] = 0.0;
            if (only_minus) Qm[i] = Q_minus_e2;
            if (only_minus || iso) Qp[i] = 0.0;
            if (only_plus) Qp[i] = Q_plus_e1;
            if (only_plus) B[i] = B_e1;
            if (only_minus) B[i] = B_e2;
            if (iso) B[i] = 0.0;
            if (only_plus) C[i] = C_e1;
            if (only_minus) C[i] = C_e2;
            if (iso) C[i] = C_iso;
            double cos_gap_e = 2.0 * std::sin(0.75 * kd[i]) * s4[i];
            double AC_e1 = (cos_gap_e + ac * Q_plus_e1 * s2[i]) / dxs;
            double AC_e2 = (cos_gap_e - ac * Q_minus_e2 * s2[i]) / dxs;
            double AC_iso = 2.0 * s4sq[i] / c2s;
            if (only_plus) AC[i] = AC_e1;
            if (only_minus) AC[i] = AC_e2;
            if (iso) AC[i] = AC_iso;
        }
    }
    if (any_gm)
        for (std::size_t i = 0; i < n; i++) {
            const double ac = a_const[i];
            double Qg = g.ground_minus[i] ? Qm[i] : 0.0;
            A[i] = A[i] + ac * Qg / sin_safe[i];
            B[i] = B[i] - ac * Qg / (2.0 * c2[i]);
            C[i] = C[i] - ac * Qg / (2.0 * s2[i]);
            AC[i] = AC[i] + ac * Qg * recip[i];
        }
    if (any_gp)
        for (std::size_t i = 0; i < n; i++) {
            const double ac = a_const[i];
            double Qg = g.ground_plus[i] ? Qp[i] : 0.0;
            A[i] = A[i] - ac * Qg / sin_safe[i];
            B[i] = B[i] - ac * Qg / (2.0 * c2[i]);
            C[i] = C[i] + ac * Qg / (2.0 * s2[i]);
            AC[i] = AC[i] - ac * Qg * recip[i];
        }
    std::vector<ll> all_seg, all_basis;
    std::vector<double> all_A, all_B, all_C, all_AC;
    std::vector<int8_t> all_sigma;
    for (std::size_t i = 0; i < n; i++) {
        all_seg.push_back(static_cast<ll>(i));
        all_basis.push_back(static_cast<ll>(i));
        all_A.push_back(A[i]);
        all_B.push_back(B[i]);
        all_C.push_back(C[i]);
        all_AC.push_back(AC[i]);
        all_sigma.push_back(1);
    }
    for (std::size_t e = 0; e < g.nm_basis.size(); e++) {
        const std::size_t sgi = static_cast<std::size_t>(g.nm_seg[e]);
        const double an = a_const[sgi];  // a scalar a_const is the same value everywhere
        const double q = Qm[static_cast<std::size_t>(g.nm_basis[e])];
        all_seg.push_back(g.nm_seg[e]);
        all_basis.push_back(g.nm_basis[e]);
        all_A.push_back(an * q / sin_kd[sgi]);
        all_B.push_back(an * q / (2.0 * c2[sgi]));
        all_C.push_back(-an * q / (2.0 * s2[sgi]));
        all_AC.push_back(an * q * recip[sgi]);
        all_sigma.push_back(g.nm_sigma[e]);
    }
    for (std::size_t e = 0; e < g.np_basis.size(); e++) {
        const std::size_t sgi = static_cast<std::size_t>(g.np_seg[e]);
        const double an = a_const[sgi];
        const double q = Qp[static_cast<std::size_t>(g.np_basis[e])];
        all_seg.push_back(g.np_seg[e]);
        all_basis.push_back(g.np_basis[e]);
        all_A.push_back(-an * q / sin_kd[sgi]);
        all_B.push_back(an * q / (2.0 * c2[sgi]));
        all_C.push_back(an * q / (2.0 * s2[sgi]));
        all_AC.push_back(-an * q * recip[sgi]);
        all_sigma.push_back(g.np_sigma[e]);
    }
    std::vector<std::size_t> order(all_seg.size());
    std::iota(order.begin(), order.end(), 0);
    std::stable_sort(order.begin(), order.end(), [&](std::size_t x, std::size_t y) { return all_seg[x] < all_seg[y]; });
    SegView &v = f.basis;
    v.starts.assign(n + 1, 0);
    for (std::size_t i = 0; i < n; i++) v.starts[i + 1] = v.starts[i] + 1 + g.nm_count[i] + g.np_count[i];
    for (std::size_t o : order) {
        v.jbasis.push_back(all_basis[o]);
        v.A.push_back(cd(all_A[o], 0.0));
        v.B.push_back(cd(all_B[o], 0.0));
        v.C.push_back(cd(all_C[o], 0.0));
        v.AC.push_back(cd(all_AC[o], 0.0));
        v.sigma.push_back(all_sigma[o]);
    }
    return f;
}

// --- the comparison format ---------------------------------------------------

namespace {

struct Out {
    std::string s;
    void line(const std::string &l) { s += l + "\n"; }
    void seq(const std::string &p, const char *kind, std::size_t n) { line(p + " : " + kind + " " + std::to_string(n)); }
    void none(const std::string &p) { line(p + " = None"); }
    void b(const std::string &p, bool v) { line(p + " = bool " + (v ? "True" : "False")); }
    void i(const std::string &p, long long v) { line(p + " = int " + std::to_string(v)); }
    void f(const std::string &p, double v) { line(p + " = float " + nec5::dump_bits(v)); }
    void c(const std::string &p, cd v) {
        line(p + " = complex " + nec5::dump_bits(v.real()) + " " + nec5::dump_bits(v.imag()));
    }
    void str(const std::string &p, const std::string &u) { line(p + " = str " + nec5::dump_hex(u)); }
};

std::string at(const std::string &p, std::size_t k) { return p + "[" + std::to_string(k) + "]"; }

template <class T>
void ints(Out &o, const std::string &p, const std::vector<T> &v) {
    o.seq(p, "list", v.size());
    for (std::size_t k = 0; k < v.size(); k++) o.i(at(p, k), static_cast<long long>(v[k]));
}
void bools(Out &o, const std::string &p, const std::vector<bool> &v) {
    o.seq(p, "list", v.size());
    for (std::size_t k = 0; k < v.size(); k++) o.b(at(p, k), v[k]);
}
void floats(Out &o, const std::string &p, const std::vector<double> &v) {
    o.seq(p, "list", v.size());
    for (std::size_t k = 0; k < v.size(); k++) o.f(at(p, k), v[k]);
}
void cplx(Out &o, const std::string &p, const std::vector<cd> &v) {
    o.seq(p, "list", v.size());
    for (std::size_t k = 0; k < v.size(); k++) o.c(at(p, k), v[k]);
}
void points(Out &o, const std::string &p, const std::vector<Point3> &v) {
    o.seq(p, "list", v.size());
    for (std::size_t k = 0; k < v.size(); k++) {
        o.seq(at(p, k), "list", 3);
        for (std::size_t c = 0; c < 3; c++) o.f(at(at(p, k), c), v[k][c]);
    }
}

void dump_front_into(Out &o, const std::string &p, const Front &fr) {
    o.f(p + ".c", fr.c);
    o.f(p + ".freq", fr.freq);
    o.f(p + ".omega", fr.omega);
    o.f(p + ".k", fr.k);
    o.f(p + ".eta", fr.eta);
    o.seq(p + ".junctions", "list", fr.junctions.size());
    for (std::size_t j = 0; j < fr.junctions.size(); j++) {
        o.seq(at(p + ".junctions", j), "list", fr.junctions[j].size());
        for (std::size_t m = 0; m < fr.junctions[j].size(); m++) {
            std::string q = at(at(p + ".junctions", j), m);
            o.seq(q, "tuple", 2);
            o.i(at(q, 0), fr.junctions[j][m].first);
            o.str(at(q, 1), fr.junctions[j][m].second ? "start" : "end");
        }
    }
    const Geometry &g = fr.geometry;
    std::string q = p + ".geometry";
    o.line(q + " : mapping 25");
    int key = 0;
    auto k = [&](const char *name) {
        std::string r = q + "{" + std::to_string(key++) + "}";
        o.str(r + ".key", name);
        return r + ".value";
    };
    points(o, k("seg_l"), g.seg_l);
    points(o, k("seg_r"), g.seg_r);
    points(o, k("seg_centers"), g.seg_centers);
    points(o, k("seg_tangents"), g.seg_tangents);
    floats(o, k("seg_h"), g.seg_h);
    o.i(k("n_segs"), g.n_segs);
    ints(o, k("nm_basis"), g.nm_basis);
    ints(o, k("nm_seg"), g.nm_seg);
    ints(o, k("nm_sigma"), g.nm_sigma);
    ints(o, k("np_basis"), g.np_basis);
    ints(o, k("np_seg"), g.np_seg);
    ints(o, k("np_sigma"), g.np_sigma);
    ints(o, k("nm_count"), g.nm_count);
    ints(o, k("np_count"), g.np_count);
    ints(o, k("wire_first"), g.wire_first);
    ints(o, k("wire_last"), g.wire_last);
    {
        std::string r = k("feed_seg");
        if (g.has_feed_seg) {
            o.i(r, g.feed_seg);
        } else {
            o.none(r);
        }
    }
    ints(o, k("feed_segs"), g.feed_segs);
    floats(o, k("feed_xi"), g.feed_xi);
    {
        std::string r = k("feed_sites");
        o.seq(r, "list", g.feed_sites.size());
        for (std::size_t s = 0; s < g.feed_sites.size(); s++) {
            std::string t = at(r, s);
            o.seq(t, "tuple", 4);
            o.i(at(t, 0), g.feed_sites[s].wire);
            o.f(at(t, 1), g.feed_sites[s].requested);
            o.f(at(t, 2), g.feed_sites[s].clamped);
            o.f(at(t, 3), g.feed_sites[s].centre);
        }
    }
    bools(o, k("ground_minus"), g.ground_minus);
    bools(o, k("crossing_minus"), g.crossing_minus);
    bools(o, k("crossing_plus"), g.crossing_plus);
    bools(o, k("ground_plus"), g.ground_plus);
    ints(o, k("grounded_junctions"), g.grounded_junctions);
    const SegView &v = fr.basis;
    std::string r = p + ".basis";
    o.line(r + " : mapping 7");
    key = 0;
    auto kb = [&](const char *name) {
        std::string t = r + "{" + std::to_string(key++) + "}";
        o.str(t + ".key", name);
        return t + ".value";
    };
    ints(o, kb("starts"), v.starts);
    ints(o, kb("jbasis"), v.jbasis);
    cplx(o, kb("A"), v.A);
    cplx(o, kb("B"), v.B);
    cplx(o, kb("C"), v.C);
    cplx(o, kb("AC"), v.AC);
    ints(o, kb("sigma"), v.sigma);
}

}  // namespace

std::string dump_front(const std::string &text) {
    Out o;
    nec4::Build b;
    try {
        b = nec4::build(text, "sinusoidal");
    } catch (const Declined &e) {
        return "front ? declined " + nec5::dump_hex(e.what()) + "\n";
    }
    o.seq("front.structures", "list", b.structures.size());
    for (std::size_t s = 0; s < b.structures.size(); s++) {
        const nec4::StructureBuild &sb = b.structures[s];
        std::string p = "front.structures[" + std::to_string(s) + "]";
        if (sb.kind == nec4::StructureBuild::declined) {
            o.line(p + " ? declined " + nec5::dump_hex(sb.message));
            continue;
        }
        o.seq(p + ".points", "list", sb.points.size());
        for (std::size_t j = 0; j < sb.points.size(); j++) {
            std::string q = at(p + ".points", j);
            try {
                dump_front_into(o, q, front(sb.points[j].args));
            } catch (const FrontError &e) {
                o.line(q + " ! ValueError " + nec5::dump_hex(e.message));
            } catch (const Declined &e) {
                o.line(q + " ? declined " + nec5::dump_hex(e.what()));
            }
        }
    }
    return o.s;
}

}  // namespace sinusoidal
}  // namespace mw
