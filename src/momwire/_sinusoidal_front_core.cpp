// The sinusoidal front half in C++ (see `_sinusoidal_front_core.h`). Each
// function names the Python it copies; numpy's vectorised expressions are
// written out per element in numpy's evaluation order, with no contraction.

#if defined(_MSC_VER) && !defined(__clang__)
#pragma float_control(precise, on)
#pragma fp_contract(off)
#endif

#include "_libm_complex_inline.h"
#include "_sinusoidal_front_core.h"

#include "_sinusoidal_nec4_core.h"
#include "_fma_inline.h"
#include "_somm_below_grid_core.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
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
        sin_kd[i] = mw_libm_c::rsin(kd[i]);
        cos_kd[i] = mw_libm_c::rcos(kd[i]);
        s2[i] = mw_libm_c::rsin(0.5 * kd[i]);
        c2[i] = mw_libm_c::rcos(0.5 * kd[i]);
        s4[i] = mw_libm_c::rsin(0.25 * kd[i]);
        const double rs4 = mw_libm_c::rsin(0.25 * kd[i]);
        recip[i] = rs4 * rs4 / (mw_libm_c::rsin(0.5 * kd[i]) * mw_libm_c::rcos(0.5 * kd[i]));
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
            const double cos_gap_e = 2.0 * mw_libm_c::rsin(0.75 * kd[i]) * s4[i] + x * sin_kd[i];
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


// --- the lower medium (momwire#1607 B6) --------------------------------------
//
// `_medium_spec.wire_media` (one label per wire, BELOW the interface or not),
// `_below_interface.crossing_junctions` and this family's crossing scope
// (`_crossing_junction_indices`), each refusal in Python's own sentence: a
// refusal is a printout of its own, the reference's words.

// `sinusoidal._COINCIDENT_CROSSING_MEMBERS_REFUSAL`
const char *const kCoincidentCrossingMembersRefusal =
    "crossing junction {j} joins members whose node-adjacent edges are geometrically COINCIDENT (wi"
    "res {a} and {b} run the same path within one wire radius): this family cannot solve that deck."
    " The thin-wire kernel regularizes a pair at zero separation to the wire radius, which is what "
    "it does for a segment against ITSELF, so coincident members contribute near-identical rows and"
    " columns and the matrix is ill-conditioned rather than wrong in any one entry. Measured in FRE"
    "E SPACE, no ground and no junction at all, on N coincident wires each carrying its own distinc"
    "t arm: max|alpha| runs 2.3 / 41 / 184 for N = 1 / 2 / 4 against BSplineSolver's 2.0e-3 / 4.0e-"
    "3 / 9.0e-3, and over a Sommerfeld ground the same deck reaches 8.4e32. So this is a property o"
    "f the family, not of the crossing serve, and it is refused here because a crossing junction is"
    " where the spelling that provokes it is natural: writing each radial of a buried screen as far"
    "-end -> hub -> RISE gives every radial the same rise. Respell the screen with ONE rise and the"
    " radials joined to it at depth (the buried-hub spelling, which this serve covers and which agr"
    "ees with BSplineSolver to 8.4e-03 on a 12-radial screen), or run each radial straight from its"
    " far end to the node so no two members share a path.";

// `sinusoidal._CROSSING_HUB_REFUSAL`
const char *const kCrossingHubRefusal =
    "SinusoidalSolver serves a crossing node joining exactly two members, one above and one below t"
    "he interface (momwire#1223 stage 2); a buried hub has not been measured on the point-matched l"
    "ane yet. Solve it with SinusoidalGalerkinSolver or BSplineSolver, which serve it";

// `sinusoidal._CROSSING_JUNCTION_REFUSAL`
const char *const kCrossingJunctionRefusal =
    "SinusoidalSolver serves wires wholly below the interface and DETACHED decks (above and buried "
    "wires with no junction in the plane), but not yet a junction joining the two media: the point-"
    "matched crossing node needs its own basis condition (I continuous, I'_above = I'_below/eps_til"
    "de), which is momwire#1220 stage 2. Solve this deck with SinusoidalGalerkinSolver or BSplineSo"
    "lver, which both serve the crossing junction, or leave the buried wires detached";

// `sinusoidal._CROSSING_MULTI_NODE_REFUSAL`
const char *const kCrossingMultiNodeRefusal =
    "SinusoidalSolver serves one crossing node per deck (momwire#1223 stage 2); a deck with several"
    " has not been measured on the point-matched lane yet. Solve it with SinusoidalGalerkinSolver o"
    "r BSplineSolver, which serve it";

// `sinusoidal._CROSSING_OBLIQUE_REFUSAL`
const char *const kCrossingObliqueRefusal =
    "SinusoidalSolver serves a crossing node whose two members' tilts off the interface normal sum "
    "to at most 30 degrees (momwire#1223 stage 2): its imposed node condition, I'_above = I'_below/"
    "eps_tilde, is the charge match for a wire crossing the plane normally, and a node that leans c"
    "onverges to a different slope ratio, leaving an impedance error that grows with the lean and d"
    "oes not refine away. Solve it with SinusoidalGalerkinSolver or BSplineSolver, whose node slope"
    " emerges from the solution";

// `sinusoidal._CROSSING_TWO_RADIUS_REFUSAL`
const char *const kCrossingTwoRadiusRefusal =
    "SinusoidalSolver does not serve a TWO-RADIUS crossing node (one wire radius above the interfac"
    "e and another below, momwire#1630): measured against NEC-5, this lane's impedance responds to "
    "the node's radii with the physics swapped -- the buried wires' radius barely moves it and the "
    "above wire's moves it several times too far -- because its radius-step junction misses NEC eve"
    "n in free space (momwire#1632). Solve it with SinusoidalGalerkinSolver or BSplineSolver, which"
    " serve the two-radius node and track NEC-5's radius response to under 0.5 ohm, or give every w"
    "ire one radius";

// `_medium_spec._REFUSE_BURIED_PEC`
const char *const kRefuseBuriedPec =
    "wire {w} runs below the ground plane (min z = {zmin:.6g} < ground_z = {gz:g}) over a PERFECTLY"
    " CONDUCTING ground, which has no lower medium to put it in: the field inside a perfect conduct"
    "or is identically zero, so a wire there is not buried, it is shorted out. A buried wire is ser"
    "ved only under ground_model='sommerfeld' with a ground_eps, where the half-space below the int"
    "erface is a real medium with a wavenumber k_m = k0*sqrt(eps_tilde) of its own (momwire#553). R"
    "aise the wire to or above the plane, or give the solve a Sommerfeld ground";

// `_medium_spec._REFUSE_CONTACT_WITH_BURIED`
const char *const kRefuseContactWithBuried =
    "wire {cw} stands an END in the ground plane (ground CONTACT) and wire {bw} is buried below it:"
    " that COMBINATION is not served, though each half is. momwire's contact model continues the wi"
    "re's current into the ground as the C2-scaled IMAGE (momwire#151), which is a fiction that wor"
    "ks because no observer is ever inside the ground to look at it - and a buried wire is exactly "
    "such an observer. What it should see is the contact current SPREADING in the lower medium - a "
    "real soil current this deck has NO CONDUCTOR for. momwire#524 phase 2 measured every consisten"
    "t spelling of the missing cross-medium physics against engine prints for exactly these decks -"
    " and those prints have since been withdrawn as a reference (momwire#929). They were captured w"
    "ith the ground card's two fields written in the wrong order (antennaknobs#1025), and this deck"
    " class turns out to have no documented spelling at all: the flag that serves buried wires leav"
    "es a conductor ENDING on the interface without a basis function there, and the flag that bonds"
    " such an end is documented as not usable when wires are buried. So the ~3 ohm (lone) / ~6 ohm "
    "(fan) residual was measured against a combination the engine does not document, and neither it"
    " nor the point-electrode reading drawn from it is evidence about this deck. The refusal itself"
    " is unchanged and rests on momwire's own physics: the contact model has no conductor for the s"
    "preading current, so there is nothing honest to serve here in either convention. Serve it by g"
    "iving the spreading current its conductor: respell the radial - or the whole SCREEN of them - "
    "to RISE to the surface and junction-join the monopole at z = 0; that crossing junction is serv"
    "ed for one above wire over N below wires (momwire#524 phase 2, fan widening), including the sc"
    "reen's buried-hub spelling (one rise, N radials joined at depth). Or raise the above-ground wi"
    "re clear of the interface (an elevated feed over a buried counterpoise is served), or solve th"
    "e buried structure on its own";

// `_medium_spec._REFUSE_CROSSING`
const char *const kRefuseCrossing =
    "wire {w} crosses the ground interface mid-span (polyline z runs {zmin:.6g} to {zmax:.6g} acros"
    "s ground_z = {gz:g}): momwire serves wires wholly at or above the interface, wires strictly be"
    "low it, and current CROSSING it only through a crossing junction (momwire#524 phase 2): split "
    "the wire AT the interface into a below wire whose end stands in the plane and an above wire st"
    "arting there, and declare the junction between them - that deck is served, with continuity of "
    "current and the interface slope condition emerging from the fill itself. A single polyline wit"
    "h points on both sides is not split for you here, because feeds, loads, junctions and per-wire"
    " results are keyed on its wire index and a split would renumber them; the deck seams split a c"
    "ard wire at the plane themselves and keep its addresses (the NEC-2 portal since momwire#667, t"
    "he EZNEC seam since momwire#1281). Alternatively leave the buried part DETACHED (a buried radi"
    "al screen under a base-fed vertical is served that way, momwire#553), or raise the whole wire "
    "clear of the interface";

// `_below_interface.BURIED_EXTENDED_KERNEL_REFUSAL`
const char *const kBuriedExtendedKernelRefusal =
    "extended_kernel=True + a wire below the ground plane is not served: the extended kernel extend"
    "s every pair of the geometry, and momwire#553 measured neither what that means for a pair span"
    "ning two media (the tube expansion's O(a^2) term is written at one wavenumber) nor for the ima"
    "ge of a buried source, which lands in the OTHER medium. Solve the buried deck with extended_ke"
    "rnel=False, which is the default";

// `_below_interface.BURIED_GRAZING_REFUSAL`
const char *const kBuriedGrazingRefusal =
    "this deck's buried wires reach a below/below pair elevation of theta = {th:.4g} deg among the "
    "pairs inside the cap, below the {floor:g} deg grazing floor the below/below surfaces are tabul"
    "ated from (the pair's two depths add to {depth:.4g} m, and theta = atan2(depth sum, horizontal"
    " separation)). The floor is asked only of pairs inside the cap \xe2\x80\x94" " image distance "
    "R1 within 4 in-medium wavelengths, the pairs whose remainder is evaluated; past it the remaind"
    "er is served as zero (momwire#1053, #1187). Below the floor the surfaces carry the lateral wav"
    "e's LOGARITHMIC structure \xe2\x80\x94" " measured drift 1.1 to 3.3 of scale between 2 and 0.0"
    "5 deg \xe2\x80\x94" " which no uniform lattice resolves, and theta = 0 has no node at all beca"
    "use h = 0 leaves the tail without its exponential decay. Bury the wires deeper, shorten them, "
    "or wait for the log-spaced grazing band (momwire#553 U2's recorded follow-up)";


const double kCrossingTiltSumMaxDeg = 30.0;   // sinusoidal._CROSSING_TILT_SUM_MAX_DEG
const double kMinCrossingNodeSeparationM = 1.0;  // _below_interface.MIN_CROSSING_NODE_SEPARATION_M

// Python's `format(x, ".Ng")` (and `"g"`, six digits) for a double
std::string py_g(double x, int prec = 6) {
    char buf[64];
    std::snprintf(buf, sizeof buf, "%.*g", prec, x);
    return buf;
}

// `str.format` over `{name}` / `{name:spec}` fields, each value already
// formatted by its caller
std::string py_fill(const std::string &tmpl, const std::vector<std::pair<std::string, std::string>> &vals) {
    std::string out;
    for (std::size_t i = 0; i < tmpl.size(); i++) {
        if (tmpl[i] != '{') {
            out += tmpl[i];
            continue;
        }
        const std::size_t close = tmpl.find('}', i);
        std::string name = tmpl.substr(i + 1, close - i - 1);
        const std::size_t colon = name.find(':');
        if (colon != std::string::npos) name = name.substr(0, colon);
        bool hit = false;
        for (const auto &kv : vals)
            if (kv.first == name) {
                out += kv.second;
                hit = true;
                break;
            }
        if (!hit) throw std::logic_error("an unfilled refusal field: " + name);
        i = close;
    }
    return out;
}

[[noreturn]] void refuse_impl(const std::string &m) { throw FrontError(m); }
[[noreturn]] void refuse_ni(const std::string &m) { throw FrontError(m, "NotImplementedError"); }

// `_ground_spec.contact_ends(polylines, gz)`: (wire, at start), sorted as
// Python sorts ("end" < "start")
std::vector<std::pair<ll, bool>> contact_ends(const std::vector<std::vector<Point3>> &pls, double gz) {
    std::vector<std::pair<ll, bool>> out;
    for (std::size_t w = 0; w < pls.size(); w++) {
        const double tol = ground_touch_tol(pls[w]);
        if (std::fabs(pls[w].back()[2] - gz) <= tol) out.push_back({static_cast<ll>(w), false});
        if (std::fabs(pls[w].front()[2] - gz) <= tol) out.push_back({static_cast<ll>(w), true});
    }
    return out;
}

// `_medium_spec.grounded_crossing_exemption(polylines, gz, groups)`
std::set<std::pair<ll, bool>> crossing_exemption(const std::vector<std::vector<Point3>> &pls, double gz,
                                                 const std::vector<std::vector<std::pair<ll, bool>>> &groups) {
    std::set<std::pair<ll, bool>> ends;
    for (const auto &members : groups) {
        if (members.size() < 2) continue;
        const auto &pl = pls[static_cast<std::size_t>(members[0].first)];
        const Point3 &pt = members[0].second ? pl.front() : pl.back();
        if (std::fabs(pt[2] - gz) > ground_touch_tol(pl)) continue;
        bool reaches = false;
        for (const auto &m : members) {
            const auto &plm = pls[static_cast<std::size_t>(m.first)];
            double zmax = plm[0][2];
            for (const Point3 &p : plm) zmax = std::max(zmax, p[2]);
            if (zmax > gz + ground_touch_tol(plm)) reaches = true;
        }
        if (!reaches) continue;
        ends.insert(members.begin(), members.end());
    }
    return ends;
}

// `_medium_spec.wire_media(polylines, gz, lower_medium=, pec=, crossing_ends=)`:
// per wire true when BELOW the interface, or Python's ValueError
std::vector<bool> wire_media(const std::vector<std::vector<Point3>> &pls, double gz, bool lower_medium, bool pec,
                             const std::set<std::pair<ll, bool>> &crossing_ends) {
    std::vector<bool> below(pls.size(), false);
    bool any = false;
    for (std::size_t w = 0; w < pls.size(); w++) {
        const auto &pl = pls[w];
        const double tol = ground_touch_tol(pl);
        double zmin = pl[0][2], zmax = pl[0][2];
        for (const Point3 &p : pl) {
            if (std::isnan(p[2])) decline("a NaN height (the reference's min / max are numpy's)");
            zmin = std::min(zmin, p[2]);
            zmax = std::max(zmax, p[2]);
        }
        if (zmin >= gz - tol) continue;
        if (zmax >= gz - tol) {
            // `_is_crossing_below`
            bool ok = !(zmax > gz + tol);
            const bool ts = std::fabs(pl.front()[2] - gz) <= tol, te = std::fabs(pl.back()[2] - gz) <= tol;
            if (!(ts || te)) ok = false;
            for (std::size_t i = 1; i + 1 < pl.size(); i++)
                if (pl[i][2] >= gz - tol) ok = false;
            if (ts && !crossing_ends.count({static_cast<ll>(w), true})) ok = false;
            if (te && !crossing_ends.count({static_cast<ll>(w), false})) ok = false;
            if (!ok)
                refuse_impl(py_fill(kRefuseCrossing, {{"w", std::to_string(w)},
                                                      {"zmin", py_g(zmin)},
                                                      {"zmax", py_g(zmax)},
                                                      {"gz", py_g(gz)}}));
        }
        if (!lower_medium) {
            if (!pec) decline("a buried wire over a reflection-coefficient ground");
            refuse_impl(py_fill(kRefuseBuriedPec, {{"w", std::to_string(w)}, {"zmin", py_g(zmin)}, {"gz", py_g(gz)}}));
        }
        below[w] = true;
        any = true;
    }
    if (any) {
        std::size_t first_below = 0;
        while (!below[first_below]) first_below++;
        for (const auto &c : contact_ends(pls, gz))
            if (!crossing_ends.count(c))
                refuse_impl(py_fill(kRefuseContactWithBuried,
                                    {{"cw", std::to_string(c.first)}, {"bw", std::to_string(first_below)}}));
    }
    return below;
}

// `_below_interface.crossing_junctions(media, groups, grounded, polylines,
// gz, radii, two_radius=True)`
std::vector<ll> crossing_junctions(const std::vector<bool> &below, const std::vector<std::vector<std::pair<ll, bool>>> &groups,
                                   const std::set<ll> &grounded, const std::vector<std::vector<Point3>> &pls, double gz,
                                   const std::vector<double> &radii) {
    if (std::find(below.begin(), below.end(), true) == below.end()) return {};
    std::vector<ll> crossing;
    for (std::size_t j = 0; j < groups.size(); j++) {
        if (!grounded.count(static_cast<ll>(j))) continue;
        bool a = false, b = false;
        for (const auto &m : groups[j]) (below[static_cast<std::size_t>(m.first)] ? b : a) = true;
        if (a && b) crossing.push_back(static_cast<ll>(j));
    }
    std::set<std::pair<ll, bool>> earned;
    for (ll j : crossing) earned.insert(groups[static_cast<std::size_t>(j)].begin(), groups[static_cast<std::size_t>(j)].end());
    std::size_t first_below = 0;
    while (!below[first_below]) first_below++;
    for (const auto &c : contact_ends(pls, gz))
        if (!earned.count(c))
            refuse_impl(py_fill(kRefuseContactWithBuried,
                                {{"cw", std::to_string(c.first)}, {"bw", std::to_string(first_below)}}));
    if (crossing.empty()) return {};
    if (crossing.size() > 1) {
        // `_closest_crossing_nodes`: each node at its first member's end, the
        // pairs in `np.triu_indices` order, numpy's hypot, the first argmin
        std::vector<Point3> pts;
        for (ll j : crossing) {
            const auto &m0 = groups[static_cast<std::size_t>(j)][0];
            const auto &pl = pls[static_cast<std::size_t>(m0.first)];
            pts.push_back(m0.second ? pl.front() : pl.back());
        }
        double best = std::numeric_limits<double>::infinity();
        std::size_t bi = 0, bj = 0;
        bool first = true;
        for (std::size_t i = 0; i < pts.size(); i++)
            for (std::size_t j = i + 1; j < pts.size(); j++) {
                const double sep = std::hypot(pts[i][0] - pts[j][0], pts[i][1] - pts[j][1]);
                if (std::isnan(sep)) decline("a NaN crossing-node separation");
                if (first || sep < best) {
                    best = sep;
                    bi = i;
                    bj = j;
                    first = false;
                }
            }
        if (best < kMinCrossingNodeSeparationM)
            refuse_ni("crossing junctions " + std::to_string(crossing[bi]) + " and " + std::to_string(crossing[bj]) +
                        ", at (" + py_g(pts[bi][0]) + ", " + py_g(pts[bi][1]) + ") and (" + py_g(pts[bj][0]) + ", " +
                        py_g(pts[bj][1]) + "), stand " + py_g(best) +
                        " m apart: the crossing serve completes several crossing nodes per deck from " +
                        py_g(kMinCrossingNodeSeparationM) +
                        " m apart, the closest separation its two-node collapse was gated at (antennaknobs plan "
                        "U9), and closer crossing nodes have no measured completion");
    }
    for (ll j : crossing) {
        ll n_above = 0;
        for (const auto &m : groups[static_cast<std::size_t>(j)])
            if (!below[static_cast<std::size_t>(m.first)]) n_above++;
        if (n_above != 1)
            refuse_ni("crossing junction with more than one above member: the crossing serve joins ONE above "
                        "wire to N below wires at the interface (momwire#524 fan widening); the above-tent x "
                        "above-tent interface corner has no measured convention");
    }
    for (std::size_t j = 0; j < groups.size(); j++) {
        if (std::find(crossing.begin(), crossing.end(), static_cast<ll>(j)) != crossing.end()) continue;
        bool a = false, b = false;
        for (const auto &m : groups[j]) (below[static_cast<std::size_t>(m.first)] ? b : a) = true;
        if (!grounded.count(static_cast<ll>(j)) && !(a && b)) continue;
        refuse_ni("a deck with a crossing junction and an in-plane or media-mixing OTHER junction is not "
                    "served: the complete crossing spelling completes every value-1 end on its axes, and a "
                    "junction standing in the interface sits where the corner and image terms live, which has "
                    "no measured completion (momwire#524 phase 2, momwire#1133)");
    }
    double rmin = radii[0], rmax = radii[0];
    for (double r : radii) {
        rmin = std::min(rmin, r);
        rmax = std::max(rmax, r);
    }
    if (rmax - rmin > 0.0) {
        // `crossing_side_radii(media, radii, crossing_above_member(...))`:
        // the above side is the node's above member; the below side's spread
        ll above_member = -1;
        if (crossing.size() == 1)
            for (const auto &m : groups[static_cast<std::size_t>(crossing[0])])
                if (!below[static_cast<std::size_t>(m.first)]) {
                    above_member = m.first;
                    break;
                }
        for (int side = 0; side < 2; side++) {
            if (side == 0 && above_member >= 0) continue;
            double lo = std::numeric_limits<double>::infinity(), hi = -lo;
            for (std::size_t w = 0; w < below.size(); w++)
                if (below[w] == (side == 1)) {
                    lo = std::min(lo, radii[w]);
                    hi = std::max(hi, radii[w]);
                }
            if (hi - lo > 0.0)
                refuse_ni(std::string("crossing serve with per-wire radii that differ within the ") +
                            (side == 0 ? "above" : "below") +
                            " wires: the two-radius rule serves one radius per side of the interface (the node's "
                            "point tests at the buried radius, antennaknobs plan U5); a spread within one side has "
                            "no measured node radius");
        }
        if (crossing.size() > 1)
            refuse_ni("a TWO-RADIUS crossing deck with " + std::to_string(crossing.size()) +
                        " crossing junctions: the two-radius node (antennaknobs plan U5) and more than one crossing "
                        "node (plan U9) were each measured alone, and a two-radius deck with several nodes has no "
                        "measured completion");
    }
    return crossing;
}

// `SinusoidalSolver._node_edge_tilt_deg(w, end)`: numpy's degrees(arccos)
double node_edge_tilt_deg(const std::vector<Point3> &pl, bool start) {
    const Point3 &node = start ? pl[0] : pl[pl.size() - 1];
    const Point3 &nxt = start ? pl[1] : pl[pl.size() - 2];
    const Point3 d = sub(nxt, node);
    const double cosv = std::fabs(d[2]) / norm3(d);
    return (180.0 / kPi) * std::acos(std::min(1.0, cosv)) * 1.0;
}

// `_crossing_junction_indices`' scope checks: `_refuse_coincident_crossing_
// members`, then `_refuse_crossing_scope` (two radii, several nodes, a hub,
// `_refuse_oblique_crossing`). `galerkin`: SinusoidalGalerkinSolver's, whose
// `_refuse_crossing_scope` refuses nothing (momwire#1718 S11b)
void refuse_crossing_scope(const std::vector<ll> &crossing, const std::vector<std::vector<std::pair<ll, bool>>> &groups,
                           const std::vector<std::vector<Point3>> &pls, const std::vector<double> &radii,
                           bool galerkin) {
    if (crossing.empty()) return;
    for (ll j : crossing) {
        const auto &members = groups[static_cast<std::size_t>(j)];
        std::vector<std::pair<Point3, Point3>> edges;
        for (const auto &m : members) {
            const auto &pl = pls[static_cast<std::size_t>(m.first)];
            edges.push_back(m.second ? std::make_pair(pl[0], pl[1])
                                     : std::make_pair(pl[pl.size() - 1], pl[pl.size() - 2]));
        }
        for (std::size_t i = 0; i < edges.size(); i++)
            for (std::size_t jj = i + 1; jj < edges.size(); jj++) {
                const double a = std::max(radii[static_cast<std::size_t>(members[i].first)],
                                          radii[static_cast<std::size_t>(members[jj].first)]);
                if (norm3(sub(edges[i].first, edges[jj].first)) <= a &&
                    norm3(sub(edges[i].second, edges[jj].second)) <= a)
                    refuse_ni(py_fill(kCoincidentCrossingMembersRefusal, {{"j", std::to_string(j)},
                                                                          {"a", std::to_string(members[i].first)},
                                                                          {"b", std::to_string(members[jj].first)}}));
            }
    }
    if (galerkin) return;
    double rmin = radii[0], rmax = radii[0];
    for (double r : radii) {
        rmin = std::min(rmin, r);
        rmax = std::max(rmax, r);
    }
    if (rmax - rmin > 0.0)
        refuse_ni("crossing junction " + std::to_string(crossing[0]) + " joins wires of different radii: " +
                    kCrossingTwoRadiusRefusal);
    if (crossing.size() > 1)
        refuse_ni(std::to_string(crossing.size()) + " crossing junctions: " + kCrossingMultiNodeRefusal);
    for (ll j : crossing) {
        const std::size_t nm = groups[static_cast<std::size_t>(j)].size();
        if (nm != 2)
            refuse_ni("crossing junction " + std::to_string(j) + " joins " + std::to_string(nm) +
                        " members: " + kCrossingHubRefusal);
    }
    for (ll j : crossing) {
        std::vector<double> tilts;
        double sum = 0.0;
        for (const auto &m : groups[static_cast<std::size_t>(j)]) {
            tilts.push_back(node_edge_tilt_deg(pls[static_cast<std::size_t>(m.first)], m.second));
            sum = sum + tilts.back();
        }
        if (sum > kCrossingTiltSumMaxDeg + 1e-9) {
            std::string each;
            for (std::size_t q = 0; q < tilts.size(); q++) {
                char buf[64];
                std::snprintf(buf, sizeof buf, "%.1f", tilts[q]);
                each += (q ? " + " : "") + std::string(buf);
            }
            char tot[64];
            std::snprintf(tot, sizeof tot, "%.1f", sum);
            refuse_ni("crossing junction " + std::to_string(j) + ": its members' tilts off the interface normal sum to " +
                        tot + " degrees (" + each + "): " + kCrossingObliqueRefusal);
        }
    }
}


// --- numpy's complex arithmetic, for the basis at a complex k ----------------
//
// An array product is numpy's fused one (each part one fused multiply-add,
// `_cmul.cmul`); a product of two 0-d scalars is numpy's scalar math
// (`cdouble_ctype_multiply`); a quotient is `CDOUBLE_divide` either way;
// CPython's own complex scalars take `_Py_c_prod` / `_Py_c_quot`. A real
// operand is promoted to (x, 0) first, as numpy promotes it.
//
// The scalar product and the quotient are numpy's plain C, so their bits are
// numpy's BUILD's (`MW_NUMPY_CONTRACTS`, `_libm_complex_inline.h`): fused on
// arm64, unfused on x86-64. CPython's `_Py_c_prod` / `_Py_c_quot` are unfused
// on both (CPython builds without contraction).
cd xmul(cd a, cd b) {  // array
    return cd(mw_fma::exact(a.real(), b.real(), -(a.imag() * b.imag())),
              mw_fma::exact(a.real(), b.imag(), a.imag() * b.real()));
}
cd pmul(cd a, cd b) {  // CPython 3.12's `_Py_c_prod`
    return cd(a.real() * b.real() - a.imag() * b.imag(), a.real() * b.imag() + a.imag() * b.real());
}
cd smul(cd a, cd b) {  // numpy 0-d scalars
#if MW_NUMPY_CONTRACTS
    return xmul(a, b);  // `a.r * b.r - a.i * b.i` contracted: the fused product
#else
    return pmul(a, b);
#endif
}
cd xdiv(cd a, cd b) {  // `CDOUBLE_divide`
    const double in1r = a.real(), in1i = a.imag(), in2r = b.real(), in2i = b.imag();
    const double in2r_abs = std::fabs(in2r), in2i_abs = std::fabs(in2i);
    if (in2r_abs >= in2i_abs) {
        if (in2r_abs == 0 && in2i_abs == 0) return cd(in1r / in2r_abs, in1i / in2i_abs);
        const double rat = in2i / in2r;
#if MW_NUMPY_CONTRACTS
        const double scl = 1.0 / mw_fma::exact(in2i, rat, in2r);
        return cd(mw_fma::exact(in1i, rat, in1r) * scl, mw_fma::exact(-in1r, rat, in1i) * scl);
#else
        const double scl = 1.0 / (in2r + in2i * rat);
        return cd((in1r + in1i * rat) * scl, (in1i - in1r * rat) * scl);
#endif
    }
    const double rat = in2r / in2i;
#if MW_NUMPY_CONTRACTS
    const double scl = 1.0 / mw_fma::exact(in2r, rat, in2i);
    return cd(mw_fma::exact(in1r, rat, in1i) * scl, mw_fma::exact(in1i, rat, -in1r) * scl);
#else
    const double scl = 1.0 / (in2i + in2r * rat);
    return cd((in1r * rat + in1i) * scl, (in1i * rat - in1r) * scl);
#endif
}
cd pdiv(cd a, cd b) {  // CPython 3.12's `_Py_c_quot`
    const double abs_breal = b.real() < 0 ? -b.real() : b.real();
    const double abs_bimag = b.imag() < 0 ? -b.imag() : b.imag();
    if (abs_breal >= abs_bimag) {
        if (abs_breal == 0.0) decline("a complex division by zero (the reference raises)");
        const double ratio = b.imag() / b.real();
        const double denom = b.real() + b.imag() * ratio;
        return cd((a.real() + a.imag() * ratio) / denom, (a.imag() - a.real() * ratio) / denom);
    }
    const double ratio = b.real() / b.imag();
    const double denom = b.real() * ratio + b.imag();
    return cd((a.real() * ratio + a.imag()) / denom, (a.imag() * ratio - a.real()) / denom);
}
cd xadd(cd a, cd b) { return cd(a.real() + b.real(), a.imag() + b.imag()); }
cd xsub(cd a, cd b) { return cd(a.real() - b.real(), a.imag() - b.imag()); }
cd xneg(cd a) { return cd(-a.real(), -a.imag()); }
cd re(double x) { return cd(x, 0.0); }
bool nonzero(cd z) { return z.real() != 0.0 || z.imag() != 0.0; }
// numpy's complex sin / cos / log: libm's csin / ccos / clog (not libc++'s
// own std:: formulas, which differ by bits on macOS)
cd csin_(cd z) { return mw_libm_c::csin(z); }
cd ccos_(cd z) { return mw_libm_c::ccos(z); }
cd clog_(cd z) { return mw_libm_c::clog(z); }

// `_basis_coefs(geom, k)` at a complex k, in numpy's complex arithmetic op
// for op: a scalar k (a wholly-buried deck's k_m, momwire#1222), or one k
// per segment (a crossing deck's two-k view, momwire#1223: k_m below, k_p
// above, `_edge_weight` on the edges that cross). `a_const` is a 0-d scalar
// only when both k and the radius are (`ka = k * a` CPython's product, the
// log numpy's); every other product is an array product. No ground-junction
// patch: a wire end on the plane beside a buried wire is refused upstream.
void basis_coefs_complex(Front &f, const std::vector<double> &a, const std::vector<cd> &kseg, bool per_seg_k) {
    const Geometry &g = f.geometry;
    const std::size_t n = static_cast<std::size_t>(g.n_segs);
    // a crossing node's C0 end (SinusoidalGalerkinSolver's, momwire#1718
    // S11b) takes the interior branch with no atom: `has_minus` / `has_plus`
    for (std::size_t i = 0; i < n; i++)
        if (g.ground_minus[i] || g.ground_plus[i]) decline("a ground end at a complex k");
    const bool scalar = f.uniform_radius && !per_seg_k;
    auto kat = [&](std::size_t i) { return per_seg_k ? kseg[i] : kseg[0]; };
    std::vector<cd> ac(n);
    if (scalar) {
        const cd ka = pmul(kseg[0], re(a[0]));
        const cd lg = clog_(pdiv(re(2.0), ka));
        const cd v = xdiv(re(1.0), cd(lg.real() - kEulerGamma, lg.imag() - 0.0));
        std::fill(ac.begin(), ac.end(), v);
    } else {
        // `ka = k * a`: an array product (a per-segment k, or radius)
        for (std::size_t i = 0; i < n; i++) {
            const cd ka = xmul(kat(i), re(f.uniform_radius ? a[0] : a[i]));
            const cd lg = clog_(xdiv(re(2.0), ka));
            ac[i] = xdiv(re(1.0), xsub(lg, re(kEulerGamma)));
        }
    }
    // a scalar times a scalar, or an array product
    auto amul = [&](cd x, cd y) { return scalar ? smul(x, y) : xmul(x, y); };
    std::vector<cd> kd(n), sin_kd(n), cos_kd(n), s2(n), c2(n), s4(n), recip(n), atom(n);
    for (std::size_t i = 0; i < n; i++) {
        kd[i] = xmul(kat(i), re(g.seg_h[i]));
        sin_kd[i] = csin_(kd[i]);
        cos_kd[i] = ccos_(kd[i]);
        s2[i] = csin_(xmul(re(0.5), kd[i]));
        c2[i] = ccos_(xmul(re(0.5), kd[i]));
        s4[i] = csin_(xmul(re(0.25), kd[i]));
        // `_recip_sin_gap(kd)`
        const cd r4 = csin_(xmul(re(0.25), kd[i]));
        recip[i] = xdiv(xmul(r4, r4), xmul(csin_(xmul(re(0.5), kd[i])), ccos_(xmul(re(0.5), kd[i]))));
        atom[i] = xmul(xdiv(s2[i], c2[i]), ac[i]);
    }
    std::vector<cd> Pm(n, cd(0.0, 0.0)), Pp(n, cd(0.0, 0.0));
    // the two-k recipe's edge constants (`_edge_weight`): a_i * k_j / k_i
    // across the interface, NEC's a_j within one medium
    std::vector<cd> w_nm, w_np;
    if (per_seg_k) {
        auto weight = [&](ll bi, ll sj) {
            const std::size_t ib = static_cast<std::size_t>(bi), js = static_cast<std::size_t>(sj);
            const cd k_i = kseg[ib], k_j = kseg[js];
            return k_i != k_j ? xdiv(xmul(ac[ib], k_j), k_i) : ac[js];
        };
        for (std::size_t e = 0; e < g.nm_basis.size(); e++) w_nm.push_back(weight(g.nm_basis[e], g.nm_seg[e]));
        for (std::size_t e = 0; e < g.np_basis.size(); e++) w_np.push_back(weight(g.np_basis[e], g.np_seg[e]));
        std::vector<cd> tan_half(n);
        for (std::size_t i = 0; i < n; i++) tan_half[i] = xdiv(s2[i], c2[i]);
        for (std::size_t e = 0; e < g.nm_basis.size(); e++) {
            cd &t = Pm[static_cast<std::size_t>(g.nm_basis[e])];
            t = xadd(t, xmul(w_nm[e], tan_half[static_cast<std::size_t>(g.nm_seg[e])]));
        }
        for (std::size_t e = 0; e < g.np_basis.size(); e++) {
            cd &t = Pp[static_cast<std::size_t>(g.np_basis[e])];
            t = xadd(t, xmul(xneg(w_np[e]), tan_half[static_cast<std::size_t>(g.np_seg[e])]));
        }
    } else {
        for (std::size_t e = 0; e < g.nm_basis.size(); e++) {
            cd &t = Pm[static_cast<std::size_t>(g.nm_basis[e])];
            t = xadd(t, atom[static_cast<std::size_t>(g.nm_seg[e])]);
        }
        for (std::size_t e = 0; e < g.np_basis.size(); e++) {
            cd &t = Pp[static_cast<std::size_t>(g.np_basis[e])];
            t = xadd(t, xneg(atom[static_cast<std::size_t>(g.np_seg[e])]));
        }
    }
    std::vector<bool> has_minus(n), has_plus(n);
    for (std::size_t i = 0; i < n; i++) {
        has_minus[i] = g.nm_count[i] > 0 || g.crossing_minus[i];
        has_plus[i] = g.np_count[i] > 0 || g.crossing_plus[i];
    }
    std::vector<cd> Qm(n), Qp(n), A(n), B(n), C(n), AC(n), sin_safe(n), omc(n), s4sq(n);
    for (std::size_t i = 0; i < n; i++) {
        const cd a_ = ac[i];
        const cd aa = amul(a_, a_);
        const cd D = xadd(xmul(xadd(xmul(Pm[i], Pp[i]), aa), sin_kd[i]),
                          xmul(xmul(xsub(Pm[i], Pp[i]), a_), cos_kd[i]));
        const cd D_safe = nonzero(D) ? D : re(1.0);
        sin_safe[i] = nonzero(sin_kd[i]) ? sin_kd[i] : re(1.0);
        omc[i] = xmul(xmul(re(2.0), s2[i]), s2[i]);
        Qm[i] = xdiv(xsub(xmul(a_, omc[i]), xmul(Pp[i], sin_kd[i])), D_safe);
        Qp[i] = xdiv(xsub(xmul(xneg(a_), omc[i]), xmul(Pm[i], sin_kd[i])), D_safe);
        A[i] = re(-1.0);  // `np.full(n_segs, -1.0)`, cast at the concatenation
        B[i] = xdiv(xmul(xmul(a_, xadd(Qm[i], Qp[i])), s2[i]), sin_safe[i]);
        C[i] = xdiv(xmul(xmul(a_, xsub(Qm[i], Qp[i])), c2[i]), sin_safe[i]);
        const cd dP = xsub(Pm[i], Pp[i]);
        s4sq[i] = xmul(s4[i], s4[i]);
        const cd t1 = xmul(xmul(amul(amul(re(4.0), a_), a_), s2[i]), s4sq[i]);
        const cd t2 = xmul(xmul(amul(re(2.0), a_), dP), s4sq[i]);
        const cd t3 = xmul(xmul(xmul(xmul(re(2.0), s2[i]), c2[i]), Pm[i]), Pp[i]);
        const cd t4 = xmul(xmul(xmul(amul(re(2.0), a_), dP), s2[i]), s2[i]);
        AC[i] = xdiv(xadd(xsub(xsub(t1, t2), t3), t4), D_safe);
    }
    bool any_branch = false;
    for (std::size_t i = 0; i < n; i++) any_branch = any_branch || !(has_minus[i] && has_plus[i]);
    if (any_branch) {
        for (std::size_t i = 0; i < n; i++) {
            const bool only_plus = has_plus[i] && !has_minus[i];
            const bool only_minus = has_minus[i] && !has_plus[i];
            const bool iso = !has_minus[i] && !has_plus[i];
            const cd a_ = ac[i];
            const cd dxs = nonzero(cos_kd[i]) ? cos_kd[i] : re(1.0);
            const cd qpe1_d = xsub(xmul(a_, sin_kd[i]), xmul(Pp[i], cos_kd[i]));
            const cd qpe1_s = nonzero(qpe1_d) ? qpe1_d : re(1.0);
            const cd Q_plus_e1 = xdiv(xneg(omc[i]), qpe1_s);
            const cd B_e1 = xdiv(xadd(s2[i], xmul(xmul(a_, Q_plus_e1), c2[i])), dxs);
            const cd C_e1 = xdiv(xadd(c2[i], xmul(xmul(a_, Q_plus_e1), s2[i])), dxs);
            const cd qme2_d = xadd(xmul(a_, sin_kd[i]), xmul(Pm[i], cos_kd[i]));
            const cd qme2_s = nonzero(qme2_d) ? qme2_d : re(1.0);
            const cd Q_minus_e2 = xdiv(omc[i], qme2_s);
            const cd B_e2 = xdiv(xadd(xneg(s2[i]), xmul(xmul(a_, Q_minus_e2), c2[i])), dxs);
            const cd C_e2 = xdiv(xsub(c2[i], xmul(xmul(a_, Q_minus_e2), s2[i])), dxs);
            const cd c2s = nonzero(c2[i]) ? c2[i] : re(1.0);
            const cd C_iso = xdiv(re(1.0), c2s);
            if (only_plus || iso) Qm[i] = cd(0.0, 0.0);
            if (only_minus) Qm[i] = Q_minus_e2;
            if (only_minus || iso) Qp[i] = cd(0.0, 0.0);
            if (only_plus) Qp[i] = Q_plus_e1;
            if (only_plus) B[i] = B_e1;
            if (only_minus) B[i] = B_e2;
            if (iso) B[i] = cd(0.0, 0.0);
            if (only_plus) C[i] = C_e1;
            if (only_minus) C[i] = C_e2;
            if (iso) C[i] = C_iso;
            const cd cos_gap_e = xmul(xmul(re(2.0), csin_(xmul(re(0.75), kd[i]))), s4[i]);
            const cd AC_e1 = xdiv(xadd(cos_gap_e, xmul(xmul(a_, Q_plus_e1), s2[i])), dxs);
            const cd AC_e2 = xdiv(xsub(cos_gap_e, xmul(xmul(a_, Q_minus_e2), s2[i])), dxs);
            const cd AC_iso = xdiv(xmul(re(2.0), s4sq[i]), c2s);
            if (only_plus) AC[i] = AC_e1;
            if (only_minus) AC[i] = AC_e2;
            if (iso) AC[i] = AC_iso;
        }
    }
    std::vector<ll> all_seg, all_basis;
    std::vector<cd> all_A, all_B, all_C, all_AC;
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
        const cd an = per_seg_k ? w_nm[e] : ac[scalar ? 0 : sgi];
        const cd q = Qm[static_cast<std::size_t>(g.nm_basis[e])];
        all_seg.push_back(g.nm_seg[e]);
        all_basis.push_back(g.nm_basis[e]);
        all_A.push_back(xdiv(xmul(an, q), sin_kd[sgi]));
        all_B.push_back(xdiv(xmul(an, q), xmul(re(2.0), c2[sgi])));
        all_C.push_back(xdiv(xmul(xneg(an), q), xmul(re(2.0), s2[sgi])));
        all_AC.push_back(xmul(xmul(an, q), recip[sgi]));
        all_sigma.push_back(g.nm_sigma[e]);
    }
    for (std::size_t e = 0; e < g.np_basis.size(); e++) {
        const std::size_t sgi = static_cast<std::size_t>(g.np_seg[e]);
        const cd an = per_seg_k ? w_np[e] : ac[scalar ? 0 : sgi];
        const cd q = Qp[static_cast<std::size_t>(g.np_basis[e])];
        all_seg.push_back(g.np_seg[e]);
        all_basis.push_back(g.np_basis[e]);
        all_A.push_back(xdiv(xmul(xneg(an), q), sin_kd[sgi]));
        all_B.push_back(xdiv(xmul(an, q), xmul(re(2.0), c2[sgi])));
        all_C.push_back(xdiv(xmul(an, q), xmul(re(2.0), s2[sgi])));
        all_AC.push_back(xmul(xmul(xneg(an), q), recip[sgi]));
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
        v.A.push_back(all_A[o]);
        v.B.push_back(all_B[o]);
        v.C.push_back(all_C[o]);
        v.AC.push_back(all_AC[o]);
        v.sigma.push_back(all_sigma[o]);
        // `k_entry = k[all_seg[order]]`: each entry at its segment's own k
        v.k_entry.push_back(kat(static_cast<std::size_t>(all_seg[o])));
    }
}

// `_crossing_fill.buried_medium(ground_eps, omega, eps, k)`
Medium buried_medium(double eps_r, double sigma, double omega, double k) {
    Medium m;
    m.eps_t = cd(eps_r, (-sigma) / (omega * kEps));  // `complex(eps_r, -sigma / (omega * eps0))`
    m.k_p = k;
    m.k_m = somm_below::k_medium(m.eps_t, k);
    m.eps_m = pmul(re(kEps), m.eps_t);  // `eps * eps_t`
    m.c2 = pdiv(cd(m.eps_t.real() - 1.0, m.eps_t.imag() - 0.0), cd(m.eps_t.real() + 1.0, m.eps_t.imag() + 0.0));
    m.a_m = pdiv(cd(1.0 - m.eps_t.real(), 0.0 - m.eps_t.imag()), cd(1.0 + m.eps_t.real(), 0.0 + m.eps_t.imag()));
    return m;
}

// `_wire_spec.drop_duplicated_wires("SinusoidalSolver", ...)` (momwire#1042 /
// #1333) over the NEC-4.2 build's arguments, which carry no junction-indexed
// ports and no node drive: `out` is `in` with each wire listed twice dropped
// and every wire-indexed argument remapped (`in` itself when nothing is), and
// the dedup says how. Refuses (FrontError) in the constructor's words.
nec4::WireDedup drop_duplicated_wires(const nec4::SolverArgs &in, nec4::SolverArgs &out, bool galerkin) {
    // `type(self).__name__` and `type(self)._KEPT_COINCIDENT_REFUSAL`
    // (SinusoidalGalerkinSolver's for the Galerkin basis, momwire#1718 S3)
    const char *const kFamily = galerkin ? "SinusoidalGalerkinSolver" : "SinusoidalSolver";
    const char *const kCoincidentRefusal =
        galerkin ? "this family cannot solve coincident conductors: its Galerkin fill "
                   "answers them from rounding (momwire#1333). Solve the deck with "
                   "BSplineSolver, or join both copies to the same wires at each end so "
                   "they merge as one"
                 : "this family cannot solve coincident conductors: its answer for them "
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
    const auto refuse = [kFamily](std::size_t r, std::size_t twin, const std::string &why) {
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

// `_basis_coefs(geom, k)` at a real k, on the geometry and the radii `a`
// per segment (the default rule, not the NEC-4 junction solve): the view.
static SegView basis_coefs_at(const Geometry &g, const std::vector<double> &a, double k) {
    const std::size_t n = static_cast<std::size_t>(g.n_segs);
    SegView view;
    SegView &v = view;
    std::vector<double> a_const(n), kd(n), sin_kd(n), cos_kd(n), s2(n), c2(n), s4(n), recip(n), atom(n);
    for (std::size_t i = 0; i < n; i++) {
        double ka = k * a[i];
        a_const[i] = 1.0 / (std::log(2.0 / ka) - kEulerGamma);
        kd[i] = k * g.seg_h[i];
        sin_kd[i] = mw_libm_c::rsin(kd[i]);
        cos_kd[i] = mw_libm_c::rcos(kd[i]);
        s2[i] = mw_libm_c::rsin(0.5 * kd[i]);
        c2[i] = mw_libm_c::rcos(0.5 * kd[i]);
        s4[i] = mw_libm_c::rsin(0.25 * kd[i]);
        double rs4 = mw_libm_c::rsin(0.25 * kd[i]);
        recip[i] = rs4 * rs4 / (mw_libm_c::rsin(0.5 * kd[i]) * mw_libm_c::rcos(0.5 * kd[i]));
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
            double cos_gap_e = 2.0 * mw_libm_c::rsin(0.75 * kd[i]) * s4[i];
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
    return view;
}

// `_stitch_basis_coefs(geom, below, k_p, k_m)` (SinusoidalGalerkinSolver's
// crossing view, momwire#1718 S11b): `_basis_coefs` at the real k_p and at
// the complex k_m, one CSR topology, each segment's entries from its own
// medium's view, `k_entry = np.where(entry_below, k_m, k_p)`
void stitch_basis_coefs(Front &f, const std::vector<double> &a) {
    const Geometry &g = f.geometry;
    const SegView view_p = basis_coefs_at(g, a, f.k);
    f.basis = SegView();
    basis_coefs_complex(f, a, {f.medium.k_m}, false);
    SegView &v = f.basis;  // view_m
    if (v.starts != view_p.starts || v.jbasis != view_p.jbasis || v.sigma != view_p.sigma)
        throw std::logic_error("the D2 basis stitch: the CSR topology depends on k");
    const std::size_t n = static_cast<std::size_t>(g.n_segs);
    for (std::size_t i = 0; i < n; i++)
        for (ll q = v.starts[i]; q < v.starts[i + 1]; q++) {
            const std::size_t e = static_cast<std::size_t>(q);
            if (f.below_seg[i]) {
                v.k_entry[e] = f.medium.k_m;
                continue;
            }
            v.A[e] = view_p.A[e];
            v.B[e] = view_p.B[e];
            v.C[e] = view_p.C[e];
            v.AC[e] = view_p.AC[e];
            v.k_entry[e] = re(f.medium.k_p);
        }
}

// `_crossing_wing_view(geom, base_view, below, medium)`: one node-wing column
// per crossing-junction member (junctions ascending, members in order), the
// three-term extension at the member segment's own k normalised to unit
// inflow, appended after the N segment bases and stably sorted into its
// segment's entries. A buried member's coefficients in CPython's complex
// arithmetic and numpy's complex scalars (`np.cos` of a Python complex, the
// scalars' products and quotients, `_recip_sin_gap`'s 0-d arrays); an above
// member's in floats.
void crossing_wing_view(Front &f) {
    const Geometry &g = f.geometry;
    std::vector<ll> cj(f.crossing_junctions);
    std::sort(cj.begin(), cj.end());
    const std::size_t N = static_cast<std::size_t>(g.n_segs);
    ll col = static_cast<ll>(N);
    std::vector<ll> segs, bases;
    std::vector<cd> A, B, C, AC, kent;
    std::vector<int8_t> sig;
    for (ll j : cj)
        for (const auto &mem : f.junctions[static_cast<std::size_t>(j)]) {
            const std::size_t w = static_cast<std::size_t>(mem.first);
            const ll m = mem.second ? g.wire_first[w] : g.wire_last[w];
            const int8_t sgn = mem.second ? -1 : 1;
            const double h = g.seg_h[static_cast<std::size_t>(m)];
            if (f.below_seg[static_cast<std::size_t>(m)]) {
                const cd k = f.medium.k_m;
                const cd kd = pmul(k, re(h));
                const cd half = pmul(re(0.5), kd);
                const cd q = xdiv(ccos_(half), csin_(half));
                A.push_back(xdiv(q, csin_(kd)));
                B.push_back(xdiv(q, smul(re(2.0), ccos_(half))));
                C.push_back(xdiv(xneg(q), smul(re(2.0), csin_(half))));
                // `_recip_sin_gap(kd)` over a 0-d array
                const cd s4 = csin_(xmul(re(0.25), kd));
                const cd rs = xdiv(smul(s4, s4), smul(csin_(xmul(re(0.5), kd)), ccos_(xmul(re(0.5), kd))));
                AC.push_back(smul(q, rs));
                kent.push_back(k);
            } else {
                const double k = f.medium.k_p;
                const double kd = k * h;
                const double q = mw_libm_c::rcos(0.5 * kd) / mw_libm_c::rsin(0.5 * kd);
                A.push_back(re(q / mw_libm_c::rsin(kd)));
                B.push_back(re(q / (2.0 * mw_libm_c::rcos(0.5 * kd))));
                C.push_back(re(-q / (2.0 * mw_libm_c::rsin(0.5 * kd))));
                const double s4 = mw_libm_c::rsin(0.25 * kd);
                AC.push_back(re(q * (s4 * s4 / (mw_libm_c::rsin(0.5 * kd) * mw_libm_c::rcos(0.5 * kd)))));
                kent.push_back(re(k));
            }
            segs.push_back(m);
            bases.push_back(col);
            sig.push_back(sgn);
            col++;
        }
    f.n_wings = static_cast<ll>(segs.size());
    if (segs.empty()) return;
    const SegView base = f.basis;
    std::vector<ll> all_seg;
    for (std::size_t i = 0; i < N; i++)
        for (ll q = base.starts[i]; q < base.starts[i + 1]; q++) all_seg.push_back(static_cast<ll>(i));
    const std::size_t nb = all_seg.size();
    all_seg.insert(all_seg.end(), segs.begin(), segs.end());
    std::vector<std::size_t> order(all_seg.size());
    std::iota(order.begin(), order.end(), 0);
    std::stable_sort(order.begin(), order.end(), [&](std::size_t x, std::size_t y) { return all_seg[x] < all_seg[y]; });
    SegView &v = f.basis;
    v = SegView();
    v.starts.assign(N + 1, 0);
    for (ll s : all_seg) v.starts[static_cast<std::size_t>(s) + 1]++;
    for (std::size_t i = 0; i < N; i++) v.starts[i + 1] += v.starts[i];
    for (std::size_t o : order) {
        const bool b = o < nb;
        const std::size_t x = b ? o : o - nb;
        v.jbasis.push_back(b ? base.jbasis[x] : bases[x]);
        v.A.push_back(b ? base.A[x] : A[x]);
        v.B.push_back(b ? base.B[x] : B[x]);
        v.C.push_back(b ? base.C[x] : C[x]);
        v.AC.push_back(b ? base.AC[x] : AC[x]);
        v.sigma.push_back(b ? base.sigma[x] : sig[x]);
        v.k_entry.push_back(b ? base.k_entry[x] : kent[x]);
    }
}

Front front(const nec4::SolverArgs &in, bool galerkin) {
    Front f;
    if (in.wires.empty()) fail("wires must be non-empty");
    // momwire#1042: a wire listed twice is solved once; everything below
    // reads the solved wires
    f.dedup = drop_duplicated_wires(in, f.args, galerkin);
    const nec4::SolverArgs &args = f.args;
    const auto &pls = args.wires;
    if (args.ground.kind == nec4::Ground::finite_fast) decline("a reflection-coefficient ground");
    const bool has_ground = args.ground.kind != nec4::Ground::none;
    const bool lower_medium = args.ground.kind == nec4::Ground::finite;  // the Sommerfeld half-space
    const double gz = 0.0;

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
    // momwire#1607 B6: `_wire_media` and, at a grounded junction over the
    // Sommerfeld ground, `_crossing_junction_indices` with this family's scope
    // (both raise in Python's words; the crossing fill itself is B6b's)
    f.below_wire.assign(n_w, false);
    if (has_ground) {
        const std::set<std::pair<ll, bool>> exempt = crossing_exemption(pls, gz, f.junctions);
        f.below_wire = wire_media(pls, gz, lower_medium, args.ground.kind == nec4::Ground::pec, exempt);
        // the Sinusoidal-Galerkin copy (momwire#1718) fills a wholly buried
        // deck (S11a) and a crossing one (S11b), with its own crossing scope
        // (`_refuse_crossing_scope` refuses nothing there)
        if (lower_medium && !grounded.empty() && !f.junctions.empty()) {
            const std::vector<ll> crossing =
                crossing_junctions(f.below_wire, f.junctions, grounded, pls, gz, f.radius_per_wire);
            refuse_crossing_scope(crossing, f.junctions, pls, f.radius_per_wire, galerkin);
            f.crossing_junctions = crossing;
            f.crossing = !crossing.empty();
        }
        const std::size_t nb = static_cast<std::size_t>(std::count(f.below_wire.begin(), f.below_wire.end(), true));
        // wires on both sides with no crossing junction: the transmitted
        // blocks (momwire#1718 S12), served on the point-matched lane; the
        // Galerkin copy declines them by name
        f.mixed = nb > 0 && nb < n_w && !f.crossing;
        if (galerkin && f.mixed)
            decline("wires on both sides of the interface with no crossing junction (momwire#1718: the "
                    "transmitted blocks, S12)");
        f.buried = nb == n_w;
        if (galerkin && f.crossing) {
            // `_two_radius_crossing` (`two_radius_crossing(media, radii,
            // crossing, groups)`): the node's above member's radius and the
            // buried one (a spread below was refused above)
            double rmin = f.radius_per_wire[0], rmax = f.radius_per_wire[0];
            for (double r : f.radius_per_wire) {
                rmin = std::min(rmin, r);
                rmax = std::max(rmax, r);
            }
            if (rmax - rmin > 0.0) {
                f.two_radius = true;
                ll above_member = -1;
                if (f.crossing_junctions.size() == 1)
                    for (const auto &m : f.junctions[static_cast<std::size_t>(f.crossing_junctions[0])])
                        if (!f.below_wire[static_cast<std::size_t>(m.first)]) {
                            above_member = m.first;
                            break;
                        }
                if (above_member < 0) decline("a two-radius crossing deck without one above member at its node");
                f.a_above = f.radius_per_wire[static_cast<std::size_t>(above_member)];
                for (std::size_t w = 0; w < n_w; w++)
                    if (f.below_wire[w]) f.a_below = f.radius_per_wire[w];
            }
        }
    }
    // the point-matched crossing node is an ordinary junction (`edges_at`)
    auto is_crossing = [&](std::size_t j) {
        return std::find(f.crossing_junctions.begin(), f.crossing_junctions.end(), static_cast<ll>(j)) !=
               f.crossing_junctions.end();
    };
    for (std::size_t j = 0; j < f.junctions.size(); j++) {
        // SinusoidalGalerkinSolver's crossing node is C0 with no condition in
        // the basis (`_crossing_node_in_basis` False): a grounded junction
        if (grounded.count(static_cast<ll>(j)) && (galerkin || !is_crossing(j))) continue;
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
        for (ll j : grounded) {
            // the point-matched crossing node's members are connected; the
            // Galerkin one's take the C0 ends (`crossing_minus` /
            // `crossing_plus`: the interior branch, no atom)
            const bool cross = is_crossing(static_cast<std::size_t>(j));
            if (cross && !galerkin) continue;
            for (const auto &m : f.junctions[static_cast<std::size_t>(j)]) {
                if (m.second) {
                    const std::size_t sg = static_cast<std::size_t>(g.wire_first[static_cast<std::size_t>(m.first)]);
                    (cross ? g.crossing_minus : g.ground_minus)[sg] = true;
                } else {
                    const std::size_t sg = static_cast<std::size_t>(g.wire_last[static_cast<std::size_t>(m.first)]);
                    (cross ? g.crossing_plus : g.ground_plus)[sg] = true;
                }
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
    if (f.buried) {
        // `_operating_medium`: a wholly-buried deck is solved in the lower
        // medium, its basis at k_m (`_readout_view`); `_nec4_live` is false
        // below the interface
        f.medium = buried_medium(args.ground.eps_r, args.ground.sigma, f.omega, f.k);
        basis_coefs_complex(f, a, {f.medium.k_m}, false);
        return f;
    }
    if (f.crossing && galerkin) {
        // SinusoidalGalerkinSolver's crossing view (momwire#1718 S11b):
        // `_stitch_basis_coefs(geom, below, k_p, k_m)`, then the node wings
        // (`_crossing_wing_view`)
        f.medium = buried_medium(args.ground.eps_r, args.ground.sigma, f.omega, f.k);
        f.below_seg.assign(n, false);
        for (std::size_t w = 0; w < n_w; w++)
            for (ll s = g.wire_first[w]; s <= g.wire_last[w]; s++) f.below_seg[static_cast<std::size_t>(s)] = f.below_wire[w];
        stitch_basis_coefs(f, a);
        crossing_wing_view(f);
        return f;
    }
    if (f.mixed) {
        // `_mixed_view` on a mixed deck: `_stitch_basis_coefs(geom, below,
        // k_p, k_m)` -- one CSR topology, each entry's coefficients from its
        // segment's medium's view (`_basis_coefs(geom, k_p)` above,
        // `_basis_coefs(geom, k_m)` below), `k_entry` its k
        f.medium = buried_medium(args.ground.eps_r, args.ground.sigma, f.omega, f.k);
        f.below_seg.assign(n, false);
        for (std::size_t w = 0; w < n_w; w++)
            for (ll s = g.wire_first[w]; s <= g.wire_last[w]; s++) f.below_seg[static_cast<std::size_t>(s)] = f.below_wire[w];
        const SegView view_p = basis_coefs_at(g, a, f.medium.k_p);
        basis_coefs_complex(f, a, {f.medium.k_m}, false);
        const SegView view_m = f.basis;
        if (view_p.starts != view_m.starts || view_p.jbasis != view_m.jbasis || view_p.sigma != view_m.sigma)
            decline("a stitched basis whose topology depends on k (the reference asserts)");
        SegView out = view_p;
        out.k_entry.assign(view_p.A.size(), cd(f.medium.k_p, 0.0));
        for (std::size_t i = 0; i < n; i++) {
            if (!f.below_seg[i]) continue;
            for (ll e = view_p.starts[i]; e < view_p.starts[i + 1]; e++) {
                const std::size_t z = static_cast<std::size_t>(e);
                out.A[z] = view_m.A[z];
                out.B[z] = view_m.B[z];
                out.C[z] = view_m.C[z];
                out.AC[z] = view_m.AC[z];
                out.k_entry[z] = f.medium.k_m;
            }
        }
        f.basis = out;
        return f;
    }
    if (f.crossing) {
        // `_mixed_view` on a crossing deck: `_basis_coefs(geom,
        // np.where(below, k_m, k_p))`, the node condition in the basis
        f.medium = buried_medium(args.ground.eps_r, args.ground.sigma, f.omega, f.k);
        f.below_seg.assign(n, false);
        std::vector<cd> kseg(n);
        for (std::size_t w = 0; w < n_w; w++)
            for (ll s = g.wire_first[w]; s <= g.wire_last[w]; s++) {
                const std::size_t i = static_cast<std::size_t>(s);
                f.below_seg[i] = f.below_wire[w];
                kseg[i] = f.below_wire[w] ? f.medium.k_m : cd(f.medium.k_p, 0.0);
            }
        basis_coefs_complex(f, a, kseg, true);
        return f;
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
    f.basis = basis_coefs_at(g, a, k);
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
    o.line(r + " : mapping " + std::to_string(v.k_entry.empty() ? 7 : 8));
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
    if (!v.k_entry.empty()) cplx(o, kb("k_entry"), v.k_entry);
}

}  // namespace

SegView basis_coefs(const Front &fr, double k) {
    if (fr.nec4) throw nec4::Declined("the NEC-4 junction basis at another k");
    const Geometry &g = fr.geometry;
    std::vector<double> a(static_cast<std::size_t>(g.n_segs));
    if (fr.uniform_radius) {
        std::fill(a.begin(), a.end(), fr.radius_per_wire[0]);
    } else {
        for (std::size_t w = 0; w < g.wire_first.size(); w++)
            for (long long s = g.wire_first[w]; s <= g.wire_last[w]; s++) a[static_cast<std::size_t>(s)] = fr.radius_per_wire[w];
    }
    return basis_coefs_at(g, a, k);
}

Front buried_at(const Front &fr, double omega) {
    if (!fr.buried) throw nec4::Declined("a buried readout on a deck that is not wholly buried");
    Front f = fr;
    f.k = omega / fr.c;
    f.medium = buried_medium(fr.args.ground.eps_r, fr.args.ground.sigma, omega, f.k);
    const Geometry &g = f.geometry;
    std::vector<double> a(static_cast<std::size_t>(g.n_segs));
    if (f.uniform_radius) {
        std::fill(a.begin(), a.end(), f.radius_per_wire[0]);
    } else {
        for (std::size_t w = 0; w < g.wire_first.size(); w++)
            for (long long s = g.wire_first[w]; s <= g.wire_last[w]; s++) a[static_cast<std::size_t>(s)] = f.radius_per_wire[w];
    }
    f.basis = SegView();
    basis_coefs_complex(f, a, {f.medium.k_m}, false);
    return f;
}

Front crossing_at(const Front &fr, double omega) {
    if (!fr.crossing || fr.below_seg.empty())
        throw nec4::Declined("a crossing readout on a deck that is not a Galerkin crossing deck");
    Front f = fr;
    f.k = omega / fr.c;
    f.medium = buried_medium(fr.args.ground.eps_r, fr.args.ground.sigma, omega, f.k);
    const Geometry &g = f.geometry;
    std::vector<double> a(static_cast<std::size_t>(g.n_segs));
    if (f.uniform_radius) {
        std::fill(a.begin(), a.end(), f.radius_per_wire[0]);
    } else {
        for (std::size_t w = 0; w < g.wire_first.size(); w++)
            for (long long s = g.wire_first[w]; s <= g.wire_last[w]; s++) a[static_cast<std::size_t>(s)] = f.radius_per_wire[w];
    }
    f.basis = SegView();
    stitch_basis_coefs(f, a);
    crossing_wing_view(f);
    return f;
}

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
                o.line(q + " ! " + e.kind + " " + nec5::dump_hex(e.message));
            } catch (const Declined &e) {
                o.line(q + " ? declined " + nec5::dump_hex(e.what()));
            }
        }
    }
    return o.s;
}

}  // namespace sinusoidal
}  // namespace mw
