// razor-2p's front half in C++: see `_razor_front_core.h`.
//
// Every step below names the Python it replays (razor.py unless said
// otherwise) and keeps that Python's floating-point spelling operation for
// operation: numpy's elementwise ufuncs are single IEEE operations, so a C++
// expression with the same operands in the same order (and this build's
// -ffp-contract=off) gives the same bits. The three numpy spellings that are
// NOT single operations are reproduced as numpy computes them:
//
//   a 3-vector's length (`_vec3.norm3`) -> sqrt((x*x + y*y) + z*z);
//   np.add.reduce(x) over a 1-D run -> numpy's pairwise sum (`pairwise_sum`);
//   np.round(x, 9)                  -> rint(x * 1e9) / 1e9.
#include "_razor_front_core.h"
#include "_leggauss_table.h"

// MSVC builds every TU /fp:fast (setup.py), which may fuse and reassociate.
// This TU is numpy's elementwise arithmetic spelled one IEEE operation at a
// time, and it is O(N) geometry, so it asks for precise semantics: the same
// bits as numpy there too, at no measurable cost.
#if defined(_MSC_VER)
#pragma float_control(precise, on)
#pragma fp_contract(off)
#endif

#include <locale.h>
#if defined(__APPLE__)
#include <xlocale.h>
#endif

#include <array>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <numeric>
#include <set>
#include <sstream>
#include <unordered_map>

namespace mw {
namespace razor {

// `_quadrature.leggauss(n)`: the frozen rule's half, mirrored as
// `_quadrature._from_half` mirrors it (negation is exact).
bool leggauss(int64_t n, std::vector<double> &x, std::vector<double> &w) {
    if (n < 1 || n > leggauss_table::N_MAX) return false;
    const int lo = leggauss_table::OFFSET[n - 1], hi = leggauss_table::OFFSET[n];
    const int64_t m = n / 2;
    x.clear();
    w.clear();
    for (int64_t i = 0; i < m; i++) {
        x.push_back(-leggauss_table::NODES[hi - 1 - i]);
        w.push_back(leggauss_table::WEIGHTS[hi - 1 - i]);
    }
    for (int k = lo; k < hi; k++) {
        x.push_back(leggauss_table::NODES[k]);
        w.push_back(leggauss_table::WEIGHTS[k]);
    }
    return true;
}

namespace {

const char *const BUNDLE_REFUSAL =
    "this deck spells a conductor as N geometrically COINCIDENT segments (a "
    "bundle). Razor's tent basis puts one column per segment, so N coincident "
    "segments are N identical columns and the matrix is singular by "
    "construction -- at any mesh, in free space and in soil alike, and "
    "whatever the quadrature. Razor has no bundle rule; BSplineSolver does "
    "(momwire#524 phase 2's fan widening). Respell the bundle as ONE "
    "conductor: a screen whose N radials meet at a buried HUB and rise to the "
    "node on a single rise is the same antenna without the coincidence "
    "(antennaknobs' `buried_radial_vertical` is spelled that way since "
    "antennaknobs#1108; its `bundle` variant is this deck), which "
    "BSplineSolver serves. Or "
    "solve the bundle with BSplineSolver, remembering that a bundle of N "
    "coincident thin wires and one wire of the same radius are two "
    "structures, never two meshes of one";

constexpr double MAX_CELL = 281474976710656.0;  // `_junction_rule._MAX_CELL`, 2**48

[[noreturn]] void value_error(const std::string &m) {
    throw FrontError(FrontErrorKind::value_error, m);
}
[[noreturn]] void not_implemented(const std::string &m) {
    throw FrontError(FrontErrorKind::not_implemented, m);
}

std::string end_name(End e) { return e == End::start ? "start" : "end"; }

// numpy's pairwise summation of a contiguous run (`pairwise_sum_DOUBLE`),
// which is what `np.add.reduce` does to a 1-D float64 array: the run is
// summed from the identity, 0.0 (measured: numpy 2.5, 13,500 random runs).
double pairwise_sum(const double *a, ptrdiff_t n) {
    if (n < 8) {
        double res = 0.0;
        for (ptrdiff_t i = 0; i < n; i++) res += a[i];
        return res;
    }
    if (n <= 128) {
        double r[8];
        for (int j = 0; j < 8; j++) r[j] = a[j];
        ptrdiff_t i = 8;
        for (; i < n - (n % 8); i += 8)
            for (int j = 0; j < 8; j++) r[j] += a[i + j];
        double res = ((r[0] + r[1]) + (r[2] + r[3])) + ((r[4] + r[5]) + (r[6] + r[7]));
        for (; i < n; i++) res += a[i];
        return res;
    }
    ptrdiff_t n2 = n / 2;
    n2 -= n2 % 8;
    return pairwise_sum(a, n2) + pairwise_sum(a + n2, n - n2);
}

// `np.round(x, 9)`: multiply by 10**9, rint, divide.
double round9(double x) { return std::nearbyint(x * 1e9) / 1e9; }

// `_ground_spec.ground_touch_tol` of the `m` anchors at `p`: 1e-6 of the
// polyline's length, `sqrt(add.reduce(d*d, axis=1))` per edge, summed.
double ground_touch_tol(const double *p, ptrdiff_t m) {
    std::vector<double> len(static_cast<size_t>(m > 1 ? m - 1 : 0));
    for (ptrdiff_t e = 0; e + 1 < m; e++) {
        const double dx = p[3 * (e + 1)] - p[3 * e];
        const double dy = p[3 * (e + 1) + 1] - p[3 * e + 1];
        const double dz = p[3 * (e + 1) + 2] - p[3 * e + 2];
        double s = 0.0;
        s += dx * dx;
        s += dy * dy;
        s += dz * dz;
        len[e] = std::sqrt(s);
    }
    const double length = 0.0 + pairwise_sum(len.data(), static_cast<ptrdiff_t>(len.size()));
    return 1e-6 * std::max(length, 1e-30);
}

}  // namespace

// --------------------------------------------------------------------------
// The shared refusals' portable spellings (`_wire_spec.format_numbers` /
// `format_shape`): numbers as printf `%.12g` (Python's `%` operator, which
// spells NaN and the infinities `nan` / `inf` / `-inf`), comma-separated in
// brackets; a shape as its extents joined by `x`.

namespace {

std::string format_12g(double x) {
    if (std::isnan(x)) return "nan";
    if (std::isinf(x)) return x > 0 ? "inf" : "-inf";
    char buf[64];
#if defined(_MSC_VER)
    static _locale_t c = _create_locale(LC_NUMERIC, "C");
    _snprintf_l(buf, sizeof buf, "%.12g", c, x);
#else
    static locale_t c = newlocale(LC_NUMERIC_MASK, "C", (locale_t)0);
    locale_t old = uselocale(c);
    std::snprintf(buf, sizeof buf, "%.12g", x);
    uselocale(old);
#endif
    return buf;
}

}  // namespace

std::string format_numbers(const std::vector<double> &values) {
    std::string s = "[";
    for (size_t i = 0; i < values.size(); i++) {
        if (i) s += ", ";
        s += format_12g(values[i]);
    }
    return s + "]";
}

std::string format_shape(const std::vector<int64_t> &shape) {
    if (shape.empty()) return "scalar";
    std::string s;
    for (size_t i = 0; i < shape.size(); i++) {
        if (i) s += "x";
        s += std::to_string(shape[i]);
    }
    return s;
}

// --------------------------------------------------------------------------

// `_vec3.norm3`: `(x*x + y*y) + z*z`, each operation rounded on its own,
// then a correctly rounded square root -- the same bits on every CPU.
double Front::norm3(const double *v) const {
    return std::sqrt((v[0] * v[0] + v[1] * v[1]) + v[2] * v[2]);
}

// What this front half does not cover, checked on the ARGUMENTS before any
// of them is read the way the constructor reads them: each is a route the
// Python front half takes (or refuses in its own words).
std::string Front::scope(const FrontInputs &in) {
    const size_t n_w = in.wire_points.size();
    size_t total = 0;
    for (int64_t m : in.wire_points) {
        if (m < 0) return "malformed-args";
        total += static_cast<size_t>(m);
    }
    if (in.points.size() != 3 * total) return "malformed-args";
    for (double v : in.points)
        if (!std::isfinite(v)) return "non-finite-geometry";
    // momwire#1042: a wire listed twice is MERGED by `drop_duplicated_wires`
    // (or refused there). Any wire whose nanometre-rounded vertex list
    // repeats an earlier one's, either direction, is that function's
    // candidate; this declines on every candidate, which is a superset of
    // what it merges.
    {
        std::set<std::vector<double>> seen;
        size_t off = 0;
        for (size_t w = 0; w < n_w; w++) {
            const size_t m = static_cast<size_t>(in.wire_points[w]);
            std::vector<double> fwd, rev;
            for (size_t i = 0; i < m; i++)
                for (int c = 0; c < 3; c++) fwd.push_back(round9(in.points[3 * (off + i) + c]) + 0.0);
            for (size_t i = m; i-- > 0;)
                for (int c = 0; c < 3; c++) rev.push_back(round9(in.points[3 * (off + i) + c]) + 0.0);
            const std::vector<double> &key = std::min(fwd, rev);
            if (!seen.insert(key).second) return "duplicated-wire";
            off += m;
        }
    }
    if (in.exact_kernel == ExactKernel::on) return "exact-kernel";
    return "";
}

void Front::construct(const FrontInputs &in) {
    // `RazorSolver.__init__`, in its order, on the arguments this scope
    // keeps (the loading, ground-model and buried arguments are the
    // caller's to have routed elsewhere).
    has_ground = in.has_ground;
    ground_z = in.ground_z;
    if (has_ground && !std::isfinite(ground_z)) value_error("ground_z must be finite");
    if ((in.n_qp_path_given && in.n_qp_path < 1) || in.n_qp_source < 1 ||
        in.n_qp_sommerfeld < 1)
        value_error("n_qp_path, n_qp_source and n_qp_sommerfeld must be >= 1");
    // c, k as the constructor computes them; omega as `compute_port_solution`
    // does (`self.c * self.k`, not the stashed `self.omega`).
    c = 1.0 / std::sqrt(EPS * MU);
    k = 2.0 * M_PI / in.wavelength;
    omega = c * k;
    c_a_imag = omega * MU;
    c_phi_imag = omega * EPS;

    n_wires = static_cast<int64_t>(in.wire_points.size());
    if (n_wires == 0) value_error("wires must be non-empty");
    pt_off.assign(1, 0);
    for (int64_t w = 0; w < n_wires; w++) {
        if (in.wire_points[static_cast<size_t>(w)] < 2)
            value_error("wire " + std::to_string(w) + ": polyline must be (M, 3) with M >= 2");
        pt_off.push_back(pt_off.back() + in.wire_points[static_cast<size_t>(w)]);
    }
    points = in.points;
}

// `_ground_ends`: the wire ends in the plane, and the two contact shapes it
// refuses.
void Front::ground_ends(std::vector<std::pair<int64_t, End>> &touching) {
    touching.clear();
    if (!has_ground) return;
    const double gz = ground_z;
    for (int64_t w = 0; w < n_wires; w++) {
        const double *p = points.data() + 3 * pt_off[w];
        const ptrdiff_t m = pt_off[w + 1] - pt_off[w];
        const double tol = ground_touch_tol(p, m);
        std::vector<char> at(static_cast<size_t>(m));
        for (ptrdiff_t i = 0; i < m; i++) at[i] = std::fabs(p[3 * i + 2] - gz) <= tol;
        for (ptrdiff_t i = 0; i + 1 < m; i++)
            if (at[i] && at[i + 1])
                value_error("wire " + std::to_string(w) +
                            " has an edge lying in the ground plane (both endpoints at "
                            "ground_z) \xe2\x80\x94 degenerate over a conducting ground");
        for (ptrdiff_t i = 1; i + 1 < m; i++)
            if (at[i])
                not_implemented(
                    "wire " + std::to_string(w) +
                    " touches the ground plane at an interior anchor: RazorSolver serves "
                    "ground contact at a wire END only (momwire#398 unit 3). A mid-span "
                    "touchdown needs a second unknown at a knot that already carries a "
                    "tent \xe2\x80\x94 split the wire there, so the contact is two wire ends");
        if (at[0]) touching.emplace_back(w, End::start);
        if (at[m - 1]) touching.emplace_back(w, End::end);
    }
}

// `_detect_junctions` (detected, never declared: declared groups are
// declined): `_junction_rule.grouped` over every wire end, then the groups
// that carry a through-path or touch the plane.
void Front::detect_junctions(const std::vector<std::pair<int64_t, End>> &grounded_ends,
                             std::vector<Junction> &groups) {
    const size_t n_pts = static_cast<size_t>(2 * n_wires);
    std::vector<double> pts(3 * n_pts);
    std::vector<std::pair<int64_t, End>> labels(n_pts);
    for (int64_t w = 0; w < n_wires; w++) {
        const double *first = points.data() + 3 * pt_off[w];
        const double *last = points.data() + 3 * (pt_off[w + 1] - 1);
        for (int c = 0; c < 3; c++) {
            pts[3 * (2 * w) + c] = first[c];
            pts[3 * (2 * w + 1) + c] = last[c];
        }
        labels[2 * w] = {w, End::start};
        labels[2 * w + 1] = {w, End::end};
    }
    const double tol = JUNCTION_TOL;
    auto within = [&](size_t i, size_t j) {
        double d[3] = {pts[3 * i] - pts[3 * j], pts[3 * i + 1] - pts[3 * j + 1],
                       pts[3 * i + 2] - pts[3 * j + 2]};
        return norm3(d) <= tol;
    };
    // `coincident_groups`: the cell grid when it can promise the walk's
    // answer, else the walk.
    std::vector<size_t> rep(n_pts);
    bool grid = true;
    std::vector<int64_t> cells(3 * n_pts);
    for (size_t i = 0; i < 3 * n_pts && grid; i++) {
        const double q = std::floor(pts[i] / (2.0 * tol));
        if (!(std::fabs(q) < MAX_CELL)) grid = false;
        cells[i] = static_cast<int64_t>(q);
    }
    if (grid) {
        struct H {
            size_t operator()(const std::tuple<int64_t, int64_t, int64_t> &t) const {
                uint64_t h = static_cast<uint64_t>(std::get<0>(t)) * 0x9E3779B97F4A7C15ULL;
                h ^= static_cast<uint64_t>(std::get<1>(t)) + 0x7F4A7C159E3779B9ULL + (h << 6) + (h >> 2);
                h ^= static_cast<uint64_t>(std::get<2>(t)) + 0x94D049BB133111EBULL + (h << 6) + (h >> 2);
                return static_cast<size_t>(h);
            }
        };
        std::unordered_map<std::tuple<int64_t, int64_t, int64_t>, std::vector<size_t>, H> buckets;
        for (size_t i = 0; i < n_pts; i++) {
            const int64_t cx = cells[3 * i], cy = cells[3 * i + 1], cz = cells[3 * i + 2];
            std::vector<size_t> cand;
            for (int dx = -1; dx <= 1; dx++)
                for (int dy = -1; dy <= 1; dy++)
                    for (int dz = -1; dz <= 1; dz++) {
                        auto it = buckets.find(std::make_tuple(cx + dx, cy + dy, cz + dz));
                        if (it != buckets.end()) cand.insert(cand.end(), it->second.begin(), it->second.end());
                    }
            std::sort(cand.begin(), cand.end());
            size_t hit = n_pts;
            for (size_t j : cand)
                if (within(i, j)) {
                    hit = j;
                    break;
                }
            if (hit == n_pts) {
                rep[i] = i;
                buckets[std::make_tuple(cx, cy, cz)].push_back(i);
            } else {
                rep[i] = hit;
            }
        }
    } else {
        std::vector<size_t> reps;
        for (size_t i = 0; i < n_pts; i++) {
            size_t hit = n_pts;
            for (size_t j : reps)
                if (within(i, j)) {
                    hit = j;
                    break;
                }
            if (hit == n_pts) {
                rep[i] = i;
                reps.push_back(i);
            } else {
                rep[i] = hit;
            }
        }
    }
    // `grouped`: label groups in creation order (a representative is first
    // seen at its own index).
    std::vector<ptrdiff_t> slot(n_pts, -1);
    std::vector<std::vector<std::pair<int64_t, End>>> out;
    for (size_t i = 0; i < n_pts; i++) {
        if (slot[rep[i]] < 0) {
            slot[rep[i]] = static_cast<ptrdiff_t>(out.size());
            out.emplace_back();
        }
        out[static_cast<size_t>(slot[rep[i]])].push_back(labels[i]);
    }
    groups.clear();
    for (auto &g : out) {
        bool grounded = false;
        for (auto &e : g)
            for (auto &t : grounded_ends)
                if (t == e) grounded = true;
        if (g.size() >= 2 || grounded) {
            Junction j;
            j.ends = g;
            j.grounded = grounded;
            groups.push_back(std::move(j));
        }
    }
}

// `_build_geometry`.
void Front::build_geometry() {
    seg_offsets.assign(1, 0);
    basis_offsets.assign(1, 0);
    seg_p0.clear();
    seg_t.clear();
    seg_h.clear();
    arc_at_knot.clear();
    for (int64_t w = 0; w < n_wires; w++) {
        const double *pl = points.data() + 3 * pt_off[w];
        const int64_t n_edges = pt_off[w + 1] - pt_off[w] - 1;
        const size_t first = seg_h.size();
        for (int64_t e = 0; e < n_edges; e++) {
            const double edge[3] = {pl[3 * (e + 1)] - pl[3 * e], pl[3 * (e + 1) + 1] - pl[3 * e + 1],
                                    pl[3 * (e + 1) + 2] - pl[3 * e + 2]};
            const double edge_len = norm3(edge);
            if (edge_len < 1e-15)
                value_error("wire " + std::to_string(w) + " edge " + std::to_string(e) +
                            " has zero length");
            const double tan[3] = {edge[0] / edge_len, edge[1] / edge_len, edge[2] / edge_len};
            const int64_t n_e = npe[static_cast<size_t>(w)][static_cast<size_t>(e)];
            const double h_e = edge_len / static_cast<double>(n_e);
            for (int64_t j = 0; j < n_e; j++) {
                // `pl[e][None, :] + np.arange(n_e)[:, None] * h_e * tan`
                const double jh = static_cast<double>(j) * h_e;
                for (int c = 0; c < 3; c++) seg_p0.push_back(pl[3 * e + c] + jh * tan[c]);
                for (int c = 0; c < 3; c++) seg_t.push_back(tan[c]);
                seg_h.push_back(h_e);
            }
        }
        const int64_t n_total = static_cast<int64_t>(seg_h.size() - first);
        // `np.concatenate([[0.0], np.cumsum(seg_h)])`
        arc_at_knot.push_back(0.0);
        double run = 0.0;
        for (size_t i = first; i < seg_h.size(); i++) {
            run = (i == first) ? seg_h[i] : run + seg_h[i];
            arc_at_knot.push_back(run);
        }
        seg_offsets.push_back(seg_offsets.back() + n_total);
        basis_offsets.push_back(basis_offsets.back() + n_total - 1);
    }
    n_seg = seg_offsets.back();
    n_interior = basis_offsets.back();

    // Junction tents, junction by junction (`_junction_wings`).
    std::vector<int64_t> j_seg;
    std::vector<char> j_rise;
    std::vector<double> j_sigma;
    junctions.clear();
    grounded_bases.clear();
    auto seg_of = [&](int64_t w, End e) {
        return e == End::end ? seg_offsets[static_cast<size_t>(w) + 1] - 1
                             : seg_offsets[static_cast<size_t>(w)];
    };
    for (const Junction &g : groups_) {
        Junction jn;
        jn.ends = g.ends;
        jn.grounded = g.grounded;
        auto add = [&](int64_t sa, bool ra, double ga, int64_t sb, bool rb, double gb) {
            jn.bases.push_back(n_interior + static_cast<int64_t>(j_seg.size() / 2));
            j_seg.push_back(sa);
            j_seg.push_back(sb);
            j_rise.push_back(ra);
            j_rise.push_back(rb);
            j_sigma.push_back(ga);
            j_sigma.push_back(gb);
        };
        if (g.grounded) {
            for (auto &e : g.ends) {
                const int64_t seg = seg_of(e.first, e.second);
                const bool rise = e.second == End::end;
                add(seg, rise, 0.0, seg, rise, rise ? -1.0 : 1.0);
            }
            grounded_bases.insert(grounded_bases.end(), jn.bases.begin(), jn.bases.end());
        } else {
            const auto &a = g.ends[0];
            for (size_t i = 1; i < g.ends.size(); i++) {
                const auto &b = g.ends[i];
                const bool ra = a.second == End::end, rb = b.second == End::end;
                add(seg_of(a.first, a.second), ra, ra ? 1.0 : -1.0, seg_of(b.first, b.second), rb,
                    rb ? -1.0 : 1.0);
            }
        }
        junctions.push_back(std::move(jn));
    }
    const int64_t n_junction = static_cast<int64_t>(j_seg.size() / 2);
    n_basis = n_interior + n_junction;
    wing_seg.assign(static_cast<size_t>(2 * n_basis), 0);
    wing_rise.assign(static_cast<size_t>(2 * n_basis), false);
    wing_sigma.assign(static_cast<size_t>(2 * n_basis), 0.0);
    int64_t m = 0;
    for (int64_t w = 0; w < n_wires; w++)
        for (int64_t s = seg_offsets[w]; s < seg_offsets[w + 1] - 1; s++, m++) {
            wing_seg[2 * m] = s;
            wing_seg[2 * m + 1] = s + 1;
            wing_rise.p[2 * m] = true;
            wing_rise.p[2 * m + 1] = false;
            wing_sigma[2 * m] = 1.0;
            wing_sigma[2 * m + 1] = 1.0;
        }
    for (size_t i = 0; i < j_seg.size(); i++) {
        wing_seg[2 * n_interior + i] = j_seg[i];
        wing_rise.p[2 * n_interior + i] = j_rise[i] != 0;
        wing_sigma[2 * n_interior + i] = j_sigma[i];
    }
}

// `_refuse_coincident_segments`.
void Front::refuse_coincident_segments() const {
    typedef std::array<double, 3> P;
    std::map<std::pair<P, P>, int64_t> seen;
    for (int64_t i = 0; i < n_seg; i++) {
        P a, b;
        for (int c = 0; c < 3; c++) {
            const double p0 = seg_p0[3 * i + c];
            const double p1 = p0 + seg_h[i] * seg_t[3 * i + c];
            // `+ 0.0` folds -0.0 onto 0.0, as Python's float equality does.
            a[c] = round9(p0) + 0.0;
            b[c] = round9(p1) + 0.0;
        }
        const auto key = (a <= b) ? std::make_pair(a, b) : std::make_pair(b, a);
        auto it = seen.find(key);
        if (it != seen.end())
            value_error("segments " + std::to_string(it->second) + " and " + std::to_string(i) +
                        " run between the same two points: " + BUNDLE_REFUSAL);
        seen.emplace(key, i);
    }
}

// `_port_columns`, through `_feed_basis_indices` / `_snap_to_knot` /
// `_feed_knots` / `_feed_snap.snap`.
void Front::port_columns() {
    feed_basis.clear();
    ties.clear();
    for (size_t i = 0; i < feeds.size(); i++) {
        const int64_t w = feeds[i].wire;
        const double *arcs_w = arc_at_knot.data() + seg_offsets[w] + w;
        const int64_t n_knot = seg_offsets[w + 1] - seg_offsets[w] + 1;
        struct Knot {
            double arc;
            int64_t basis, k;
        };
        std::vector<Knot> knots;
        for (int64_t j = 1; j < n_knot - 1; j++) knots.push_back({arcs_w[j], basis_offsets[w] + j - 1, 1});
        for (const Junction &jn : junctions)
            for (size_t e = 0; e < jn.ends.size(); e++) {
                if (jn.ends[e].first != w) continue;
                const double arc = jn.ends[e].second == End::start ? 0.0 : arcs_w[n_knot - 1];
                if (jn.grounded)
                    knots.push_back({arc, jn.bases[e], 1});
                else
                    knots.push_back({arc, jn.bases[0], static_cast<int64_t>(jn.ends.size())});
            }
        std::stable_sort(knots.begin(), knots.end(),
                         [](const Knot &a, const Knot &b) { return a.arc < b.arc; });
        const double total = arcs_w[n_knot - 1];
        const double target = feeds[i].midpoint ? total / 2.0 : feeds[i].arc;
        if (knots.empty())
            value_error("RazorSolver: no site sites on wire " + std::to_string(w));
        size_t pick = 0;
        if (knots.size() > 1) {
            std::vector<double> dist(knots.size());
            for (size_t j = 0; j < knots.size(); j++) dist[j] = std::fabs(knots[j].arc - target);
            std::vector<size_t> order(knots.size());
            std::iota(order.begin(), order.end(), 0);
            std::stable_sort(order.begin(), order.end(),
                             [&](size_t a, size_t b) { return dist[a] < dist[b]; });
            const size_t near = order[0], next = order[1];
            const double margin = dist[next] - dist[near];
            const bool tie = margin <= AMBIGUITY_TOL_FRAC * total;
            pick = tie ? std::min(near, next) : near;
            if (tie) {
                Tie t;
                t.feed = static_cast<int64_t>(i);
                t.wire = w;
                t.target = target;
                t.lo = knots[std::min(near, next)].arc;
                t.hi = knots[std::max(near, next)].arc;
                t.picked = knots[pick].arc;
                ties.push_back(t);
            }
        }
        if (knots[pick].k >= 3)
            not_implemented("feeds[" + std::to_string(i) + "]: the source snaps to a junction where " +
                            std::to_string(knots[pick].k) +
                            " wire ends meet, and an ARCLENGTH cannot name which of that node's "
                            "branches the gap sits in. Feed an interior knot, or model the source "
                            "on a short bridge wire off the junction.");
        feed_basis.push_back(knots[pick].basis);
    }
    n_ports = static_cast<int64_t>(feed_basis.size() + node_gaps.size());
    cols.assign(static_cast<size_t>(n_basis * n_ports), 0.0);
    for (size_t j = 0; j < feed_basis.size(); j++) cols[feed_basis[j] * n_ports + j] = 1.0;
    if (node_gaps.empty()) return;
    std::map<std::pair<int64_t, End>, std::pair<size_t, size_t>> where;
    for (size_t jn = 0; jn < junctions.size(); jn++)
        for (size_t pos = 0; pos < junctions[jn].ends.size(); pos++)
            where[junctions[jn].ends[pos]] = {jn, pos};
    for (size_t p = 0; p < node_gaps.size(); p++) {
        const auto &loc = where.at({node_gaps[p].wire, node_gaps[p].end});
        const Junction &jn = junctions[loc.first];
        const size_t column = feed_basis.size() + p;
        if (loc.second == 0) {
            for (int64_t b : jn.bases) cols[b * n_ports + column] = -1.0;
        } else {
            cols[jn.bases[loc.second - 1] * n_ports + column] = 1.0;
        }
    }
}

// `_assemble_Z_prepare` on the centroid-sharing branch: what
// `SolveContext::fill` reads.
void Front::prepare() {
    const size_t n = static_cast<size_t>(n_basis);
    s_a.resize(n);
    s_b.resize(n);
    h_a.resize(n);
    h_b.resize(n);
    q_a.resize(n);
    q_b.resize(n);
    for (size_t m = 0; m < n; m++) {
        s_a[m] = wing_seg[2 * m];
        s_b[m] = wing_seg[2 * m + 1];
        h_a[m] = seg_h[s_a[m]];
        h_b[m] = seg_h[s_b[m]];
        // `wing_sigma * np.where(wing_rise, 1.0, -1.0) / h`
        q_a[m] = wing_sigma[2 * m] * (wing_rise.p[2 * m] ? 1.0 : -1.0) / h_a[m];
        q_b[m] = wing_sigma[2 * m + 1] * (wing_rise.p[2 * m + 1] ? 1.0 : -1.0) / h_b[m];
    }
    // The kernel's radius per SOURCE segment (`_kernel_radius`, broadcast by
    // `_FusedMoments`): the scalar on a uniform model, else each segment its
    // wire's.
    const size_t ns = static_cast<size_t>(n_seg);
    a_col.resize(ns);
    for (int64_t w = 0; w < n_wires; w++)
        for (int64_t s = seg_offsets[w]; s < seg_offsets[w + 1]; s++)
            a_col[s] = uniform_radius ? radius_value : radius_per_wire[w];
    group.clear();
    a_ek.clear();
    b_obs.clear();
    if (extended_kernel) {
        // `_ek_axis_groups` labels every segment alike (momwire#1368); the
        // source tube per OBSERVER only when the radii differ.
        group.assign(ns, 0);
        a_ek = a_col;
        if (!uniform_radius) {
            b_obs.resize(ns);
            for (int64_t w = 0; w < n_wires; w++)
                for (int64_t s = seg_offsets[w]; s < seg_offsets[w + 1]; s++)
                    b_obs[s] = radius_per_wire[w];
        }
    }
    // `seg_p0 + 0.5 * seg_h[:, None] * seg_t`
    cent.resize(3 * ns);
    for (size_t s = 0; s < ns; s++) {
        const double half = 0.5 * seg_h[s];
        for (int c = 0; c < 3; c++) cent[3 * s + c] = seg_p0[3 * s + c] + half * seg_t[3 * s + c];
    }
    // `_testing_paths`' NEC-5 branch, flattened as the fill reads it.
    tans.resize(6 * n);
    wts.resize(2 * n);
    td_a.resize(3 * n);
    td_b.resize(3 * n);
    for (size_t m = 0; m < n; m++) {
        const double ga = wing_sigma[2 * m], gb = wing_sigma[2 * m + 1];
        for (int c = 0; c < 3; c++) {
            const double ta = ga * seg_t[3 * s_a[m] + c], tb = gb * seg_t[3 * s_b[m] + c];
            tans[6 * m + c] = ta;
            tans[6 * m + 3 + c] = tb;
            td_a[c * n + m] = ta;
            td_b[c * n + m] = tb;
        }
        wts[2 * m] = 0.5 * h_a[m];
        wts[2 * m + 1] = 0.5 * h_b[m];
    }
    fall_a.assign(n, false);
    fall_b.assign(n, false);
    for (size_t m = 0; m < n; m++) {
        fall_a.p[m] = !wing_rise.p[2 * m];
        fall_b.p[m] = !wing_rise.p[2 * m + 1];
    }
    // The PEC image (`_image_sources` / `_ground_mirror`): origins reflected
    // through z = ground_z, every source tangent z-flipped.
    has_image = has_ground;
    img_seg_p0.clear();
    img_seg_t.clear();
    img_td_a.clear();
    img_td_b.clear();
    if (has_image) {
        img_seg_p0 = seg_p0;
        img_seg_t.resize(3 * ns);
        const double two_gz = 2 * ground_z;
        for (size_t s = 0; s < ns; s++) {
            img_seg_p0[3 * s + 2] = two_gz - seg_p0[3 * s + 2];
            img_seg_t[3 * s] = seg_t[3 * s] * 1.0;
            img_seg_t[3 * s + 1] = seg_t[3 * s + 1] * 1.0;
            img_seg_t[3 * s + 2] = seg_t[3 * s + 2] * -1.0;
        }
        img_td_a.resize(3 * n);
        img_td_b.resize(3 * n);
        const double mirror[3] = {1.0, 1.0, -1.0};
        for (int c = 0; c < 3; c++)
            for (size_t m = 0; m < n; m++) {
                img_td_a[c * n + m] = td_a[c * n + m] * mirror[c];
                img_td_b[c * n + m] = td_b[c * n + m] * mirror[c];
            }
    }
}

std::string Front::build(const FrontInputs &in) {
    const std::string why = scope(in);
    if (!why.empty()) return why;
    construct(in);

    std::vector<std::pair<int64_t, End>> touching;
    ground_ends(touching);

    // `normalize_wire_radius`.
    {
        const bool scalar = in.radius_shape.empty();
        if (scalar) {
            if (in.radius.size() != 1) value_error("wire_radius: malformed");
            radius_per_wire.assign(static_cast<size_t>(n_wires), in.radius[0]);
        } else if (in.radius_shape.size() != 1 || in.radius_shape[0] != n_wires) {
            value_error("wire_radius: expected a scalar or a length-" + std::to_string(n_wires) +
                        " sequence (one entry per wire), got shape " + format_shape(in.radius_shape));
        } else {
            radius_per_wire = in.radius;
        }
        for (double r : radius_per_wire)
            if (!std::isfinite(r) || r <= 0.0)
                value_error("wire_radius entries must be positive and finite, got " +
                            format_numbers(radius_per_wire));
        uniform_radius = true;
        for (double r : radius_per_wire)
            if (r != radius_per_wire[0]) uniform_radius = false;
        radius_value = radius_per_wire[0];
    }

    // `_refuse_buried_geometry` over a perfect ground: every wire at or
    // above the plane (`_medium_spec.wire_media`), else the shared buried
    // sentences, which stay Python's.
    if (has_ground)
        for (int64_t w = 0; w < n_wires; w++) {
            const double *p = points.data() + 3 * pt_off[w];
            const ptrdiff_t m = pt_off[w + 1] - pt_off[w];
            double zmin = p[2];
            for (ptrdiff_t i = 1; i < m; i++) zmin = std::min(zmin, p[3 * i + 2]);
            if (!(zmin >= ground_z - ground_touch_tol(p, m))) return "below-plane";
        }
    extended_kernel = in.extended_kernel;

    // `n_per_edge_per_wire`.
    if (in.counts_given && static_cast<int64_t>(in.counts.size()) != n_wires)
        value_error("n_per_edge_per_wire length " + std::to_string(in.counts.size()) +
                    " != number of wires " + std::to_string(n_wires));
    npe.assign(static_cast<size_t>(n_wires), {});
    for (int64_t w = 0; w < n_wires; w++) {
        const int64_t n_edges = pt_off[w + 1] - pt_off[w] - 1;
        EdgeCounts ec;
        if (in.counts_given) ec = in.counts[static_cast<size_t>(w)];
        std::vector<int64_t> v;
        if (ec.kind == EdgeCounts::none)
            v.assign(static_cast<size_t>(n_edges), in.nsegs);
        else if (ec.kind == EdgeCounts::scalar)
            v.assign(static_cast<size_t>(n_edges), ec.v.empty() ? 0 : ec.v[0]);
        else
            v = ec.v;
        if (static_cast<int64_t>(v.size()) != n_edges)
            value_error("wire " + std::to_string(w) + ": n_per_edge length " +
                        std::to_string(v.size()) + " != number of edges " + std::to_string(n_edges));
        for (int64_t c : v)
            if (c < 1) value_error("wire " + std::to_string(w) + ": every edge needs >= 1 segment");
        npe[static_cast<size_t>(w)] = v;
    }

    // `_find_junctions`, and the one-segment wire junctioned at neither end
    // (momwire#608).
    detect_junctions(touching, groups_);
    {
        std::set<std::pair<int64_t, End>> joined;
        for (auto &g : groups_)
            for (auto &e : g.ends) joined.insert(e);
        for (int64_t w = 0; w < n_wires; w++) {
            int64_t total = 0;
            for (int64_t c : npe[static_cast<size_t>(w)]) total += c;
            if (total > 1) continue;
            if (joined.count({w, End::start}) || joined.count({w, End::end})) continue;
            value_error(
                "wire " + std::to_string(w) +
                ": a one-segment wire junctioned at neither end carries no basis \xe2\x80\x94 it "
                "would hold no current and scatter nothing, and the solve would be identical to "
                "one without it (split it in two, or drop it). A one-segment wire whose end "
                "meets something is fine, whether that is another wire or the ground plane: the "
                "tent such an end carries is the basis an interior segment carries.");
        }
    }

    // `_wire_spec.normalize_node_gaps`, then the grounded-end refusal.
    node_gaps.clear();
    {
        std::map<std::pair<int64_t, End>, size_t> end_to_junction;
        for (size_t j = 0; j < groups_.size(); j++)
            for (auto &e : groups_[j].ends) end_to_junction[e] = j;
        std::set<std::pair<int64_t, End>> seen_members;
        std::set<size_t> seen_junctions;
        for (size_t i = 0; i < in.node_gaps.size(); i++) {
            const NodeGapSpec &g = in.node_gaps[i];
            const std::string at = "node_gaps[" + std::to_string(i) + "]: ";
            if (!(0 <= g.wire && g.wire < n_wires))
                value_error(at + "wire_index " + std::to_string(g.wire) + " out of range [0, " +
                            std::to_string(n_wires) + ")");
            const std::pair<int64_t, End> member(g.wire, g.end);
            auto it = end_to_junction.find(member);
            if (it == end_to_junction.end())
                value_error(at + "the " + end_name(g.end) + " of wire " + std::to_string(g.wire) +
                            " is not a member of any junction group \xe2\x80\x94 a series node gap "
                            "lives at a junction; for a feed inside a wire use feeds=");
            const size_t j = it->second;
            if (groups_[j].ends.size() < 2)
                value_error(at + "junction " + std::to_string(j) +
                            " has a single member \xe2\x80\x94 there is no through-current path to "
                            "be in series with (a lone-end attachment is junction_ports=)");
            if (seen_members.count(member))
                value_error(at + "the " + end_name(g.end) + " of wire " + std::to_string(g.wire) +
                            " is listed twice");
            if (seen_junctions.count(j) && groups_[j].ends.size() == 2)
                value_error(at + "junction " + std::to_string(j) +
                            " already carries a node gap and joins only two wire ends, so a gap "
                            "through the other member is the far side of the same cut \xe2\x80\x94 "
                            "one series gap per two-wire junction (sum the EMFs into one)");
            seen_members.insert(member);
            seen_junctions.insert(j);
            node_gaps.push_back(g);
        }
        std::set<std::pair<int64_t, End>> grounded_members;
        for (auto &g : groups_)
            if (g.grounded)
                for (auto &e : g.ends) grounded_members.insert(e);
        for (size_t i = 0; i < node_gaps.size(); i++)
            if (grounded_members.count({node_gaps[i].wire, node_gaps[i].end}))
                value_error("node_gaps[" + std::to_string(i) + "]: the " +
                            end_name(node_gaps[i].end) + " of wire " +
                            std::to_string(node_gaps[i].wire) + " stands in the ground plane, and a grounded end's tent is already "
                            "the series path between that wire and the plane (momwire#151 "
                            "grounds the node through the image) \xe2\x80\x94 drive it with feeds= "
                            "at that end instead");
    }

    // The ports.
    feeds.clear();
    if (!in.feeds_given) {
        if (!(0 <= in.feed_wire_index && in.feed_wire_index < n_wires))
            value_error("feed_wire_index " + std::to_string(in.feed_wire_index) + " out of range");
        FeedSpec f;
        f.wire = in.feed_wire_index;
        f.midpoint = in.feed_midpoint;
        f.arc = in.feed_arclength;
        f.volts = cd(1.0, 0.0);
        feeds.push_back(f);
    } else {
        for (size_t i = 0; i < in.feeds.size(); i++) {
            if (!(0 <= in.feeds[i].wire && in.feeds[i].wire < n_wires))
                value_error("feeds[" + std::to_string(i) + "]: wire_index " +
                            std::to_string(in.feeds[i].wire) + " out of range [0, " +
                            std::to_string(n_wires) + ")");
            feeds.push_back(in.feeds[i]);
        }
    }
    if (feeds.empty() && node_gaps.empty())
        value_error("no ports: give at least one feeds= gap or one node_gaps= series EMF");

    // The geometry, then "auto"'s resolution on it (`_resolve_exact_kernel_
    // auto`): any segment under three radii engages the exact ring kernel.
    build_geometry();
    exact = false;
    if (in.exact_kernel == ExactKernel::automatic)
        for (int64_t w = 0; w < n_wires && !exact; w++)
            for (int64_t s = seg_offsets[w]; s < seg_offsets[w + 1]; s++)
                if (seg_h[s] < EXACT_KERNEL_AUTO_H_OVER_A * radius_per_wire[w]) {
                    exact = true;
                    break;
                }

    // The solve-time half: the bundle refusal, the ports, the tables. What
    // these raise, Python raises from `compute_port_solution`.
    try {
        refuse_coincident_segments();
        port_columns();
    } catch (const FrontError &e) {
        throw FrontError(e.kind, e.what(), true);
    }
    // The frozen table (`_quadrature.leggauss`); an order beyond it is
    // numpy's eigensolve in Python, which this front half does not replay.
    if (!leggauss(in.n_qp_source, xg, wg)) return "quadrature-order";
    prepare();
    if (in.loading.active()) {
        if (in.loading.has_conductivity) {
            if (static_cast<int64_t>(in.loading.conductivity.size()) != n_wires) return "loading-shape";
            for (double sigma : in.loading.conductivity)
                if (std::isfinite(sigma) && !(sigma > 0.0)) return "loading-values";
        }
        if (in.loading.has_rlc && static_cast<int64_t>(in.loading.rlc.size()) != n_wires)
            return "loading-shape";
        loading = in.loading;
        loading_stencil();
    }
    // `_schedule.elem_rows(_CHUNK_ELEMS, n_path * n_basis)`
    window_rows = std::max<int64_t>(1, in.chunk_elems / std::max<int64_t>(1, 2 * n_basis));
    chunk_elems = in.chunk_elems;
    // The correction's geometry-only tables.
    exact_plan.groups.clear();
    if (exact) {
        const std::string why = build_exact_plan(*this, exact_plan);
        if (!why.empty()) return why;
    }
    return "";
}

SolveInputs Front::solve_inputs(uintptr_t cancel_flag) const {
    SolveInputs s;
    const ptrdiff_t ns = static_cast<ptrdiff_t>(n_seg), n = static_cast<ptrdiff_t>(n_basis);
    s.cent = view<const double, 2>::contiguous(cent.data(), {ns, 3});
    if (!group.empty()) s.group_i = view<const int64_t, 1>::contiguous(group.data(), {ns});
    if (!b_obs.empty()) s.b_obs = view<const double, 1>::contiguous(b_obs.data(), {ns});
    s.seg_h = view<const double, 1>::contiguous(seg_h.data(), {ns});
    s.a = view<const double, 1>::contiguous(a_col.data(), {ns});
    s.xg = view<const double, 1>::contiguous(xg.data(), {static_cast<ptrdiff_t>(xg.size())});
    s.wg = view<const double, 1>::contiguous(wg.data(), {static_cast<ptrdiff_t>(wg.size())});
    s.s_a = view<const int64_t, 1>::contiguous(s_a.data(), {n});
    s.s_b = view<const int64_t, 1>::contiguous(s_b.data(), {n});
    s.h_a = view<const double, 1>::contiguous(h_a.data(), {n});
    s.h_b = view<const double, 1>::contiguous(h_b.data(), {n});
    s.q_a = view<const double, 1>::contiguous(q_a.data(), {n});
    s.q_b = view<const double, 1>::contiguous(q_b.data(), {n});
    s.fall_a = view<const bool, 1>::contiguous(fall_a.data(), {n});
    s.fall_b = view<const bool, 1>::contiguous(fall_b.data(), {n});
    s.tans = view<const double, 2>::contiguous(tans.data(), {2 * n, 3});
    s.wts = view<const double, 1>::contiguous(wts.data(), {2 * n});
    s.n_path = 2;
    if (!grounded_bases.empty())
        s.grounded = view<const int64_t, 1>::contiguous(
            grounded_bases.data(), {static_cast<ptrdiff_t>(grounded_bases.size())});
    s.k = k;
    s.c_a_imag = c_a_imag;
    s.c_phi_imag = c_phi_imag;
    auto source = [&](const std::vector<double> &p0, const std::vector<double> &t,
                      const std::vector<double> &ta, const std::vector<double> &tb) {
        SourceSet src;
        src.seg_p0 = view<const double, 2>::contiguous(p0.data(), {ns, 3});
        src.seg_t = view<const double, 2>::contiguous(t.data(), {ns, 3});
        if (!group.empty()) src.group_j = view<const int64_t, 1>::contiguous(group.data(), {ns});
        if (!a_ek.empty()) src.a_ek = view<const double, 1>::contiguous(a_ek.data(), {ns});
        src.td_a = view<const double, 2>::contiguous(ta.data(), {3, n});
        src.td_b = view<const double, 2>::contiguous(tb.data(), {3, n});
        return src;
    };
    s.real = source(seg_p0, seg_t, td_a, td_b);
    s.has_image = has_image;
    if (has_image) s.image = source(img_seg_p0, img_seg_t, img_td_a, img_td_b);
    s.window_rows = static_cast<ptrdiff_t>(window_rows);
    s.cancel_flag = cancel_flag;
    return s;
}

// `_loading_stencil(geom)` (every ordered pair of wing entries sharing a
// segment, segment-major, each segment's entries in their stable order;
// `(3/8 or 1/8) * h * sigma_l * sigma_r`), times `loading_for`'s z_seg at
// the fill's omega (numpy's complex-by-real product).
void Front::loading_stencil() {
    const std::vector<cd> z_wire = series_impedance_per_wire(omega, radius_per_wire, loading);
    struct Entry {
        int64_t basis, seg;
        bool rise;
        double sigma;
    };
    std::vector<Entry> ent;
    for (int64_t b = 0; b < n_basis; b++)
        for (int w = 0; w < 2; w++) {
            const size_t q = static_cast<size_t>(2 * b + w);
            const double sg = wing_sigma[q];
            if (sg != 0.0) ent.push_back(Entry{b, wing_seg[q], wing_rise.p[q], sg});
        }
    std::stable_sort(ent.begin(), ent.end(), [](const Entry &x, const Entry &y) { return x.seg < y.seg; });
    std::vector<int64_t> wire_of(static_cast<size_t>(n_seg));
    for (int64_t w = 0; w < n_wires; w++)
        for (int64_t s = seg_offsets[w]; s < seg_offsets[w + 1]; s++) wire_of[s] = w;
    size_t i = 0;
    while (i < ent.size()) {
        size_t j = i;
        while (j < ent.size() && ent[j].seg == ent[i].seg) j++;
        const int64_t s = ent[i].seg;
        const cd z = z_wire[static_cast<size_t>(wire_of[s])];
        for (size_t a = i; a < j; a++)
            for (size_t b = i; b < j; b++) {
                double v = (ent[a].rise == ent[b].rise ? 0.375 : 0.125) * seg_h[s];
                v *= ent[a].sigma * ent[b].sigma;
                load_rows.push_back(ent[a].basis);
                load_cols.push_back(ent[b].basis);
                load_vals.push_back(cd(std::fma(z.real(), v, -(z.imag() * 0.0)),
                                       std::fma(z.real(), 0.0, z.imag() * v)));
            }
        i = j;
    }
}

void fill_front(const Front &front, SolveContext &ctx, uintptr_t cancel_flag) {
    if (front.exact)
        ctx.fill(front.solve_inputs(cancel_flag),
                 [&](cd *z) { add_exact_correction(front, front.exact_plan, z, cancel_flag); });
    else
        ctx.fill(front.solve_inputs(cancel_flag));
}

int solve_ports(const Front &front, SolveContext &ctx, const Lapack &lp, cd *x,
                uintptr_t cancel_flag) {
    const int64_t n = front.n_basis, m = front.n_ports;
    // `cols.astype(np.complex128)`, column-major.
    for (int64_t j = 0; j < m; j++)
        for (int64_t i = 0; i < n; i++) x[j * n + i] = cd(front.cols[i * m + j], 0.0);
    fill_front(front, ctx, cancel_flag);
    if (!front.load_vals.empty())
        ctx.add_at(front.load_rows.data(), front.load_cols.data(), front.load_vals.data(),
                   front.load_vals.size());
    return ctx.lu_solve(lp, x, static_cast<int>(m));
}

}  // namespace razor
}  // namespace mw
