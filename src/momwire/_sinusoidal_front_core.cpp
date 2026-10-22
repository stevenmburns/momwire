// The sinusoidal front half in C++ (see `_sinusoidal_front_core.h`). Each
// function names the Python it copies; numpy's vectorised expressions are
// written out per element in numpy's evaluation order, with no contraction.

#if defined(_MSC_VER) && !defined(__clang__)
#pragma float_control(precise, on)
#pragma fp_contract(off)
#endif

#include "_sinusoidal_front_core.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <map>
#include <numeric>
#include <set>
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

// `_wire_spec.find_duplicated_wires`, as a yes / no
bool has_duplicated_wires(const std::vector<std::vector<Point3>> &pls) {
    double scale = 1.0;
    for (const auto &pl : pls)
        for (const Point3 &p : pl)
            for (double c : p) scale = std::max(scale, std::fabs(c));
    const double tol = 1e-12 * scale;
    typedef std::vector<std::array<double, 3>> Key;
    std::map<Key, std::size_t> seen;
    for (std::size_t i = 0; i < pls.size(); i++) {
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
        if (twin.size() != pls[i].size()) continue;
        double fmax = 0, rmax = 0;
        for (std::size_t v = 0; v < twin.size(); v++)
            for (int c = 0; c < 3; c++) {
                fmax = std::max(fmax, std::fabs(twin[v][c] - pls[i][v][c]));
                rmax = std::max(rmax, std::fabs(twin[v][c] - pls[i][pls[i].size() - 1 - v][c]));
            }
        if (fmax <= tol || rmax <= tol) return true;
    }
    return false;
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

}  // namespace

Front front(const nec4::SolverArgs &args) {
    Front f;
    const auto &pls = args.wires;
    if (pls.empty()) fail("wires must be non-empty");
    if (has_duplicated_wires(pls)) decline("duplicated wires (the solver's merge rule is not copied)");
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
