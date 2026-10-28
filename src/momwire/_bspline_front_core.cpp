// momwire#1517 step 1: see `_bspline_front_core.h`. Each step is
// numpy's operation as `bspline.py` spells it, one IEEE operation at a time.
#include "_bspline_front_core.h"

#if defined(_MSC_VER)
#pragma float_control(precise, on)
#pragma fp_contract(off)
#endif

#include <algorithm>
#include <array>
#include <cstdint>
#include <cmath>
#include <cstdio>
#include <limits>
#include <map>
#include <set>
#include <stdexcept>
#include <string>

namespace mw {
namespace bspline {

namespace {

// `np.linspace(start, stop, num)` for scalars, num >= 2 (numpy 2.5's form):
// `arange(num) * step + start`, the step-zero branch `(i / div) * delta`,
// and the last entry `stop` exactly.
std::vector<double> linspace(double start, double stop, int64_t num) {
    std::vector<double> y(static_cast<size_t>(num));
    const double div = static_cast<double>(num - 1);
    const double delta = stop - start;
    const double step = delta / div;
    for (int64_t i = 0; i < num; i++) {
        const double fi = static_cast<double>(i);
        y[static_cast<size_t>(i)] = (step == 0.0 ? (fi / div) * delta : fi * step) + start;
    }
    if (num > 1) y[static_cast<size_t>(num - 1)] = stop;
    return y;
}

// `np.round(x, 9)` (rint(x * 1e9) / 1e9), +0.0 so -0.0 keys as 0.0 does
// in Python's tuple comparison.
double round9(double x) { return std::nearbyint(x * 1e9) / 1e9 + 0.0; }

// `_same` for floats (NaN equals NaN) and for RLC entries.
bool same_value(double a, double b) { return (std::isnan(a) && std::isnan(b)) || a == b; }
bool same_rlc(const WireArgs::RLC &a, const WireArgs::RLC &b) {
    if (a.kind == 0 || b.kind == 0) return a.kind == b.kind;
    return a.kind == b.kind && a.r == b.r && a.l == b.l && a.c == b.c;
}

}  // namespace

WireDedup drop_duplicated_wires(WireArgs &args) {
    WireDedup dd;
    const size_t n = args.polylines.size();
    // `find_duplicated_wires`
    double scale = 1.0;
    bool any = false;
    for (const auto &pl : args.polylines)
        for (double v : pl) {
            scale = std::max(scale, std::fabs(v));
            any = true;
        }
    if (!any) return dd;
    const double tol = 1e-12 * scale;
    struct Dup {
        size_t copy, first;
        bool rev;
    };
    std::vector<Dup> dups;
    std::map<std::vector<double>, size_t> seen;
    for (size_t w = 0; w < n; w++) {
        const std::vector<double> &pl = args.polylines[w];
        const size_t m = pl.size() / 3;
        std::vector<double> fwd, rev;
        for (size_t i = 0; i < m; i++)
            for (int c = 0; c < 3; c++) fwd.push_back(round9(pl[3 * i + c]));
        for (size_t i = m; i-- > 0;)
            for (int c = 0; c < 3; c++) rev.push_back(round9(pl[3 * i + c]));
        const std::vector<double> &key = std::min(fwd, rev);
        auto it = seen.find(key);
        if (it == seen.end()) {
            seen.emplace(key, w);
            continue;
        }
        const size_t first = it->second;
        const std::vector<double> &tw = args.polylines[first];
        if (tw.size() != pl.size()) continue;
        double d_fwd = 0.0, d_rev = 0.0;
        for (size_t i = 0; i < m; i++)
            for (int c = 0; c < 3; c++) {
                d_fwd = std::max(d_fwd, std::fabs(tw[3 * i + c] - pl[3 * i + c]));
                d_rev = std::max(d_rev, std::fabs(tw[3 * i + c] - pl[3 * (m - 1 - i) + c]));
            }
        if (d_fwd <= tol)
            dups.push_back({w, first, false});
        else if (d_rev <= tol)
            dups.push_back({w, first, true});
    }
    if (dups.empty()) return dd;
    if (!args.junctions_given)
        throw FrontNotPorted("dedup over inferred junctions");

    // `_merge_connected_alike` over the declared junctions
    std::map<std::pair<int64_t, bool>, int64_t> node;
    for (size_t j = 0; j < args.junctions.size(); j++)
        for (const auto &m : args.junctions[j]) node[m] = static_cast<int64_t>(j);
    const auto at = [&](int64_t w, bool start) {
        auto it = node.find({w, start});
        return it == node.end() ? int64_t(-1) : it->second;
    };
    const auto joined = [&](size_t w, bool rev) {
        const int64_t s = static_cast<int64_t>(w);
        return std::make_pair(at(s, !rev), at(s, rev));  // ("start", "end") read in the first's direction
    };
    std::vector<size_t> class_order;
    std::map<size_t, std::vector<std::pair<size_t, bool>>> classes;
    for (const Dup &d : dups) {
        auto it = classes.find(d.first);
        if (it == classes.end()) {
            class_order.push_back(d.first);
            it = classes.emplace(d.first, std::vector<std::pair<size_t, bool>>{{d.first, false}}).first;
        }
        it->second.emplace_back(d.copy, d.rev);
    }
    std::map<size_t, std::pair<size_t, bool>> removed;
    std::vector<size_t> removed_order;
    for (size_t first : class_order) {
        struct Rep {
            size_t w;
            bool rev;
            std::pair<int64_t, int64_t> key;
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
                reps.push_back({member.first, member.second, key});
            } else {
                removed[member.first] = {match->w, member.second != match->rev};
                removed_order.push_back(member.first);
            }
        }
    }
    if (removed.empty()) return dd;
    const auto refuse = [](size_t r, size_t twin, const std::string &why) {
        throw FrontValueError("BSplineSolver: wires " + std::to_string(twin) + " and " +
                              std::to_string(r) + " are the same wire listed twice, and " + why +
                              ", so they cannot be merged into one conductor. Delete wire " +
                              std::to_string(r) + ", or make the two copies identical - momwire#1042");
    };
    // in `removed`'s insertion order, as Python's dict iterates
    for (size_t r : removed_order) {
        const size_t twin = removed[r].first;
        const bool rev = removed[r].second;
        std::vector<int64_t> a = args.n_per_edge[twin], b = args.n_per_edge[r];
        if (rev) std::reverse(b.begin(), b.end());
        if (a != b) refuse(r, twin, "their segment counts differ");
        if (args.radius_per_wire && !same_value(args.radius[r], args.radius[twin]))
            refuse(r, twin, "their wire_radius entries differ");
        if (args.has_conductivity && !same_value(args.conductivity[r], args.conductivity[twin]))
            refuse(r, twin, "their wire_conductivity entries differ");
        if (args.has_rlc && !same_rlc(args.rlc[r], args.rlc[twin]))
            refuse(r, twin, "their distributed_rlc entries differ");
        for (const auto &f : args.feeds)
            if (f.wire == static_cast<int64_t>(r) || f.wire == static_cast<int64_t>(twin))
                refuse(r, twin, "feeds places a site on wire " + std::to_string(f.wire));
        for (const auto &g : args.node_gaps)
            if (g.wire == static_cast<int64_t>(r) || g.wire == static_cast<int64_t>(twin))
                refuse(r, twin, "node_gaps places a site on wire " + std::to_string(g.wire));
    }
    // the declared junctions without the dropped members (no
    // junction-indexed ports here, so nothing is refused for it)
    std::vector<std::vector<std::pair<int64_t, bool>>> junctions;
    for (const auto &g : args.junctions) {
        std::vector<std::pair<int64_t, bool>> kept;
        for (const auto &m : g)
            if (!removed.count(static_cast<size_t>(m.first))) kept.push_back(m);
        if (!kept.empty()) junctions.push_back(kept);
    }
    dd.active = true;
    dd.n_input = static_cast<int64_t>(n);
    std::vector<int64_t> new_index(n, -1);
    for (size_t w = 0; w < n; w++)
        if (!removed.count(w)) {
            new_index[w] = static_cast<int64_t>(dd.kept.size());
            dd.kept.push_back(static_cast<int64_t>(w));
        }
    for (const auto &kv : removed)  // sorted by copy
        dd.removed.push_back({static_cast<int64_t>(kv.first), static_cast<int64_t>(kv.second.first),
                              kv.second.second});
    WireArgs out = args;
    out.polylines.clear();
    out.n_per_edge.clear();
    if (out.radius_per_wire) out.radius.clear();
    out.conductivity.clear();
    out.rlc.clear();
    for (int64_t w : dd.kept) {
        out.polylines.push_back(args.polylines[static_cast<size_t>(w)]);
        out.n_per_edge.push_back(args.n_per_edge[static_cast<size_t>(w)]);
        if (args.radius_per_wire) out.radius.push_back(args.radius[static_cast<size_t>(w)]);
        if (args.has_conductivity) out.conductivity.push_back(args.conductivity[static_cast<size_t>(w)]);
        if (args.has_rlc) out.rlc.push_back(args.rlc[static_cast<size_t>(w)]);
    }
    for (auto &f : out.feeds) f.wire = new_index[static_cast<size_t>(f.wire)];
    for (auto &g : out.node_gaps) g.wire = new_index[static_cast<size_t>(g.wire)];
    out.junctions.clear();
    for (const auto &g : junctions) {
        std::vector<std::pair<int64_t, bool>> mm;
        for (const auto &m : g) mm.push_back({new_index[static_cast<size_t>(m.first)], m.second});
        out.junctions.push_back(mm);
    }
    args = std::move(out);
    return dd;
}

Geometry build_geometry(const std::vector<std::vector<double>> &polylines,
                        const std::vector<std::vector<int64_t>> &n_per_edge) {
    Geometry g;
    g.seg_offsets.push_back(0);
    for (size_t w = 0; w < polylines.size(); w++) {
        const std::vector<double> &pl = polylines[w];
        const size_t n_pts = pl.size() / 3;
        WireGeometry wg;
        wg.edge_offsets.push_back(0);
        for (size_t e = 0; e + 1 < n_pts; e++) {
            const double *p0 = &pl[3 * e], *p1 = &pl[3 * e + 3];
            const double v[3] = {p1[0] - p0[0], p1[1] - p0[1], p1[2] - p0[2]};
            // `float(norm3(edge_vec))`: math.sqrt((x*x + y*y) + z*z)
            const double edge_len = std::sqrt((v[0] * v[0] + v[1] * v[1]) + v[2] * v[2]);
            if (edge_len < 1e-15)
                throw std::invalid_argument("wire " + std::to_string(w) + " edge " +
                                            std::to_string(e) + " has zero length");
            const double tan[3] = {v[0] / edge_len, v[1] / edge_len, v[2] / edge_len};
            const int64_t n_e = n_per_edge[w][e];
            const double h_e = edge_len / static_cast<double>(n_e);
            const std::vector<double> t = linspace(0.0, 1.0, n_e + 1);
            // `(1 - t)[:, None] * p0 + t[:, None] * p1`
            std::vector<double> pts(static_cast<size_t>(3 * (n_e + 1)));
            for (int64_t i = 0; i <= n_e; i++)
                for (int c = 0; c < 3; c++) {
                    const double ti = t[static_cast<size_t>(i)];
                    pts[static_cast<size_t>(3 * i + c)] = (1.0 - ti) * p0[c] + ti * p1[c];
                }
            for (int64_t i = 0; i < n_e; i++)
                for (int c = 0; c < 3; c++) {
                    wg.seg_l.push_back(pts[static_cast<size_t>(3 * i + c)]);
                    wg.seg_r.push_back(pts[static_cast<size_t>(3 * (i + 1) + c)]);
                    wg.tangents.push_back(tan[c]);
                }
            for (int64_t i = 0; i < n_e; i++) wg.h_per_seg.push_back(h_e);
            wg.edge_arc_edges.push_back(linspace(0.0, edge_len, n_e + 1));
            wg.edge_offsets.push_back(wg.edge_offsets.back() + n_e);
        }
        wg.n_total = static_cast<int64_t>(wg.h_per_seg.size());
        // `np.concatenate([[0.0], np.cumsum(h_per_seg_w)])`
        wg.arc_at_knot.push_back(0.0);
        double run = 0.0;
        for (size_t i = 0; i < wg.h_per_seg.size(); i++) {
            run = i == 0 ? wg.h_per_seg[0] : run + wg.h_per_seg[i];
            wg.arc_at_knot.push_back(run);
        }
        g.seg_offsets.push_back(g.seg_offsets.back() + wg.n_total);
        g.h_per_seg.insert(g.h_per_seg.end(), wg.h_per_seg.begin(), wg.h_per_seg.end());
        g.tangents.insert(g.tangents.end(), wg.tangents.begin(), wg.tangents.end());
        g.seg_l.insert(g.seg_l.end(), wg.seg_l.begin(), wg.seg_l.end());
        g.seg_r.insert(g.seg_r.end(), wg.seg_r.begin(), wg.seg_r.end());
        g.per_wire.push_back(std::move(wg));
    }
    g.n_segs_total = g.seg_offsets.back();
    return g;
}

namespace {

// `_ground_spec.ground_touch_tol(polyline)`: 1e-6 of its length, each edge
// `sqrt(add.reduce(d*d))`, the edges `add.reduce`d (pairwise past 8).
double polyline_touch_tol(const std::vector<double> &pl) {
    const size_t m = pl.size() / 3;
    std::vector<double> len;
    for (size_t e = 0; e + 1 < m; e++) {
        double sq = 0.0;
        for (int c = 0; c < 3; c++) {
            const double dd = pl[3 * (e + 1) + c] - pl[3 * e + c];
            sq += dd * dd;
        }
        len.push_back(std::sqrt(sq));
    }
    if (len.size() >= 8) throw FrontNotPorted("a polyline of 9 or more anchors");
    double length = 0.0;
    for (double l : len) length += l;
    return 1e-6 * std::max(length, 1e-30);
}

// `np.searchsorted(t, x, side="right")`
int64_t search_right(const std::vector<double> &t, double x) {
    return static_cast<int64_t>(std::upper_bound(t.begin(), t.end(), x) - t.begin());
}

}  // namespace

// `_medium_spec.wire_media` over the DECLARED junctions (bs2's
// `_grounded_junction_ends`: `grounded_crossing_exemption`), then
// `_below_interface.crossing_junctions`' census: per wire 1 when BELOW, the
// crossing junctions' indices into `crossing`. Every refusal either raises
// (the crossing mid-span, a buried wire with no lower medium, a ground
// contact beside a buried wire) declines instead: step 4 ports the labels,
// not the sentences.
const double kMinCrossingNodeSeparationM = 1.0;  // _below_interface.MIN_CROSSING_NODE_SEPARATION_M

std::vector<uint8_t> wire_media(const std::vector<std::vector<double>> &polylines,
                                const std::vector<std::vector<std::pair<int64_t, bool>>> &junctions,
                                double gz, bool lower_medium, const std::vector<double> &radii,
                                std::vector<int64_t> &crossing) {
    const size_t n_w = polylines.size();
    std::vector<uint8_t> below(n_w, 0);
    crossing.clear();
    std::vector<double> tol(n_w), zmin(n_w), zmax(n_w);
    bool any_below = false;
    for (size_t w = 0; w < n_w; w++) {
        const std::vector<double> &pl = polylines[w];
        tol[w] = polyline_touch_tol(pl);
        zmin[w] = zmax[w] = pl[2];
        for (size_t i = 1; i < pl.size() / 3; i++) {
            zmin[w] = std::min(zmin[w], pl[3 * i + 2]);
            zmax[w] = std::max(zmax[w], pl[3 * i + 2]);
        }
        if (!(zmin[w] >= gz - tol[w])) any_below = true;
    }
    if (!any_below) return below;
    const auto end_z = [&](int64_t w, bool start) {
        const std::vector<double> &pl = polylines[static_cast<size_t>(w)];
        return start ? pl[2] : pl[pl.size() - 1];
    };
    // `grounded_crossing_exemption`: >= 2 members, the first member's end in
    // the plane, some member reaching above it
    std::set<std::pair<int64_t, bool>> exempt;
    for (const auto &grp : junctions) {
        if (grp.size() < 2) continue;
        const int64_t w0 = grp[0].first;
        if (std::fabs(end_z(w0, grp[0].second) - gz) > tol[static_cast<size_t>(w0)]) continue;
        bool reaches = false;
        for (const auto &m : grp)
            if (zmax[static_cast<size_t>(m.first)] > gz + tol[static_cast<size_t>(m.first)]) reaches = true;
        if (!reaches) continue;
        exempt.insert(grp.begin(), grp.end());
    }
    for (size_t w = 0; w < n_w; w++) {
        const std::vector<double> &pl = polylines[w];
        const size_t m = pl.size() / 3;
        const double t = tol[w];
        if (zmin[w] >= gz - t) continue;
        if (zmax[w] >= gz - t) {
            // `_is_crossing_below`: touching the plane only at exempted
            // anchors, nothing above it
            bool ok = !(zmax[w] > gz + t);
            const bool ts = std::fabs(pl[2] - gz) <= t, te = std::fabs(pl[3 * (m - 1) + 2] - gz) <= t;
            if (!(ts || te)) ok = false;
            for (size_t i = 1; i + 1 < m; i++)
                if (pl[3 * i + 2] >= gz - t) ok = false;
            if (ts && !exempt.count({static_cast<int64_t>(w), true})) ok = false;
            if (te && !exempt.count({static_cast<int64_t>(w), false})) ok = false;
            if (!ok) throw FrontNotPorted("buried-refusal-crossing");  // `crossing_refusal`
        }
        if (!lower_medium) throw FrontNotPorted("buried-refusal-medium");  // `buried_no_medium_refusal`
        below[w] = 1;
    }
    // `_ground_spec.contact_ends`, less the exempted
    std::vector<std::pair<int64_t, bool>> contacts;
    for (size_t w = 0; w < n_w; w++) {
        const std::vector<double> &pl = polylines[w];
        if (std::fabs(pl[pl.size() - 1] - gz) <= tol[w]) contacts.push_back({static_cast<int64_t>(w), false});
        if (std::fabs(pl[2] - gz) <= tol[w]) contacts.push_back({static_cast<int64_t>(w), true});
    }
    for (const auto &c : contacts)
        if (!exempt.count(c)) throw FrontNotPorted("buried-refusal-contact");  // `contact_with_buried_refusal`
    // `crossing_junctions`: the grounded junctions joining the two media,
    // and the contacts they earn
    std::set<std::pair<int64_t, bool>> earned;
    for (size_t j = 0; j < junctions.size(); j++) {
        const auto &m0 = junctions[j][0];
        if (!(std::fabs(end_z(m0.first, m0.second) - gz) <= tol[static_cast<size_t>(m0.first)])) continue;
        bool a = false, b = false;
        for (const auto &m : junctions[j]) (below[static_cast<size_t>(m.first)] ? b : a) = true;
        if (a && b) {
            crossing.push_back(static_cast<int64_t>(j));
            earned.insert(junctions[j].begin(), junctions[j].end());
        }
    }
    for (const auto &c : contacts)
        if (!earned.count(c)) throw FrontNotPorted("buried-refusal-unearned");
    if (crossing.empty()) return below;
    // the crossing serve's scope (`crossing_junctions(..., two_radius=True)`),
    // refused in Python's NotImplementedError sentences (momwire#1607 C1)
    const auto refuse = [](const std::string &m) { throw FrontNotPorted(m, true); };
    const auto g6 = [](double x) {  // Python's `f"{x:.6g}"`
        char buf[64];
        std::snprintf(buf, sizeof buf, "%.6g", x);
        return std::string(buf);
    };
    if (crossing.size() > 1) {
        // `_closest_crossing_nodes`: each node at its first member's end, the
        // pairs in `np.triu_indices` order, numpy's hypot, the first argmin
        std::vector<std::array<double, 2>> pts;
        for (int64_t j : crossing) {
            const auto &m0 = junctions[static_cast<size_t>(j)][0];
            const std::vector<double> &pl = polylines[static_cast<size_t>(m0.first)];
            const size_t at = m0.second ? 0 : pl.size() - 3;
            pts.push_back({pl[at], pl[at + 1]});
        }
        double best = 0.0;
        size_t bi = 0, bj = 0;
        bool first = true;
        for (size_t i = 0; i < pts.size(); i++)
            for (size_t j = i + 1; j < pts.size(); j++) {
                const double sep = std::hypot(pts[i][0] - pts[j][0], pts[i][1] - pts[j][1]);
                if (std::isnan(sep)) throw FrontNotPorted("a NaN crossing-node separation");
                if (first || sep < best) {
                    best = sep;
                    bi = i;
                    bj = j;
                    first = false;
                }
            }
        if (best < kMinCrossingNodeSeparationM)
            refuse("crossing junctions " + std::to_string(crossing[bi]) + " and " + std::to_string(crossing[bj]) +
                   ", at (" + g6(pts[bi][0]) + ", " + g6(pts[bi][1]) + ") and (" + g6(pts[bj][0]) + ", " +
                   g6(pts[bj][1]) + "), stand " + g6(best) +
                   " m apart: the crossing serve completes several crossing nodes per deck from 1 m apart, the "
                   "closest separation its two-node collapse was gated at (antennaknobs plan U9), and closer "
                   "crossing nodes have no measured completion");
    }
    for (int64_t j : crossing) {
        int64_t n_above = 0;
        for (const auto &m : junctions[static_cast<size_t>(j)])
            if (!below[static_cast<size_t>(m.first)]) n_above++;
        if (n_above != 1)
            refuse("crossing junction with more than one above member: the crossing serve joins ONE above wire "
                   "to N below wires at the interface (momwire#524 fan widening); the above-tent x above-tent "
                   "interface corner has no measured convention");
    }
    for (size_t j = 0; j < junctions.size(); j++) {
        if (std::find(crossing.begin(), crossing.end(), static_cast<int64_t>(j)) != crossing.end()) continue;
        const auto &m0 = junctions[j][0];
        const bool grounded = std::fabs(end_z(m0.first, m0.second) - gz) <= tol[static_cast<size_t>(m0.first)];
        bool a = false, b = false;
        for (const auto &m : junctions[j]) (below[static_cast<size_t>(m.first)] ? b : a) = true;
        if (!grounded && !(a && b)) continue;
        refuse("a deck with a crossing junction and an in-plane or media-mixing OTHER junction is not served: "
               "the complete crossing spelling completes every value-1 end on its axes, and a junction standing "
               "in the interface sits where the corner and image terms live, which has no measured completion "
               "(momwire#524 phase 2, momwire#1133)");
    }
    double rmin = radii[0], rmax = radii[0];
    for (double r : radii) {
        rmin = std::min(rmin, r);
        rmax = std::max(rmax, r);
    }
    if (rmax - rmin > 0.0) {
        // `crossing_side_radii(media, radii, crossing_above_member(...))`:
        // the above side is the node's above member; the below side's spread
        int64_t above_member = -1;
        if (crossing.size() == 1)
            for (const auto &m : junctions[static_cast<size_t>(crossing[0])])
                if (!below[static_cast<size_t>(m.first)]) {
                    above_member = m.first;
                    break;
                }
        for (int side = 0; side < 2; side++) {
            if (side == 0 && above_member >= 0) continue;
            double lo = std::numeric_limits<double>::infinity(), hi = -lo;
            for (size_t w = 0; w < below.size(); w++)
                if ((below[w] != 0) == (side == 1)) {
                    lo = std::min(lo, radii[w]);
                    hi = std::max(hi, radii[w]);
                }
            if (hi - lo > 0.0)
                refuse(std::string("crossing serve with per-wire radii that differ within the ") +
                       (side == 0 ? "above" : "below") +
                       " wires: the two-radius rule serves one radius per side of the interface (the node's point "
                       "tests at the buried radius, antennaknobs plan U5); a spread within one side has no measured "
                       "node radius");
        }
        if (crossing.size() > 1)
            refuse("a TWO-RADIUS crossing deck with " + std::to_string(crossing.size()) +
                   " crossing junctions: the two-radius node (antennaknobs plan U5) and more than one crossing node "
                   "(plan U9) were each measured alone, and a two-radius deck with several nodes has no measured "
                   "completion");
    }
    return below;
}

Basis build_basis(const Geometry &g, const std::vector<std::vector<double>> &polylines,
                  const std::vector<std::vector<std::pair<int64_t, bool>>> &junctions,
                  const std::vector<double> &radii, bool has_ground, double ground_z,
                  int degree, bool lower_medium,
                  const std::vector<std::pair<int64_t, bool>> &gap_members) {
    const int d = degree;
    if (d < 1 || d > 3) throw FrontNotPorted("degree");
    Basis b;
    b.degree = d;
    const size_t n_w = polylines.size();
    const int64_t FREE = -1, GROUND = -2;
    b.start_dir.assign(n_w, -1);
    b.end_dir.assign(n_w, -1);
    b.start_status.assign(n_w, FREE);
    b.end_status.assign(n_w, FREE);
    for (size_t j = 0; j < junctions.size(); j++)
        for (const auto &m : junctions[j])
            (m.second ? b.start_status : b.end_status)[static_cast<size_t>(m.first)] =
                static_cast<int64_t>(j);
    std::vector<bool> grounded_junction(junctions.size(), false);
    b.below_wire.assign(n_w, 0);
    if (has_ground) {
        const double gz = ground_z;
        b.below_wire = wire_media(polylines, junctions, gz, lower_medium, radii, b.crossing);
        for (size_t w = 0; w < n_w; w++) {
            // strictly below the interface: no contact to tag, no in-plane
            // edge to diagnose (`_wire_endpoint_status`)
            if (b.below_wire[w]) continue;
            const std::vector<double> &pl = polylines[w];
            const double tol = polyline_touch_tol(pl);
            const size_t m = pl.size() / 3;
            std::vector<bool> z_at(m);
            for (size_t i = 0; i < m; i++) z_at[i] = std::fabs(pl[3 * i + 2] - gz) <= tol;
            for (size_t i = 0; i + 1 < m; i++)
                if (z_at[i] && z_at[i + 1])
                    throw FrontValueError("wire " + std::to_string(w) +
                                          " has an edge lying in the ground plane (both endpoints "
                                          "at ground_z) \xe2\x80\x94 degenerate over a conducting ground");
            // the bare conductor's stand-off floor, 2a (momwire#865)
            const double h_floor = 2.0 * radii[w];
            std::vector<bool> low(m);
            for (size_t i = 0; i < m; i++) {
                const double h = pl[3 * i + 2] - gz;
                low[i] = h > tol && h < h_floor;
            }
            for (size_t i = 0; i + 1 < m; i++)
                if (low[i] && low[i + 1]) throw FrontNotPorted("low-stand-off");
            if (b.start_status[w] == FREE && z_at[0]) b.start_status[w] = GROUND;
            if (b.end_status[w] == FREE && z_at[m - 1]) b.end_status[w] = GROUND;
        }
        // `grounded_junctions`: the first member's end in the plane
        for (size_t j = 0; j < junctions.size(); j++) {
            const auto &m0 = junctions[j][0];
            const std::vector<double> &pl = polylines[static_cast<size_t>(m0.first)];
            const double z = m0.second ? pl[2] : pl[pl.size() - 1];
            if (std::fabs(z - ground_z) <= polyline_touch_tol(pl)) grounded_junction[j] = true;
        }
    }

    // `_V_UNIT_INV[d]`
    static const double V1[4] = {1.0, 0.0, -1.0, 1.0};
    static const double V2[9] = {1.0, 0.0, 0.0, -3.0, 4.0, -1.0, 2.0, -4.0, 2.0};
    static const double V3[16] = {1.0, 0.0, 0.0, 0.0, -5.5, 9.0, -4.5, 1.0,
                                  9.0, -22.5, 18.0, -4.5, -4.5, 13.5, -13.5, 4.5};
    const double *V = d == 1 ? V1 : d == 2 ? V2 : V3;
    const int W = d + 1;
    std::vector<std::vector<std::pair<int64_t, double>>> junction_dirs(junctions.size());
    b.wire_globals.resize(g.per_wire.size());
    int64_t m_global = 0;
    for (size_t w = 0; w < g.per_wire.size(); w++) {
        const WireGeometry &pw = g.per_wire[w];
        const std::vector<double> &arc = pw.arc_at_knot;
        const double wire_arc = arc.back();
        std::vector<double> knots(static_cast<size_t>(d), 0.0);
        knots.insert(knots.end(), arc.begin(), arc.end());
        knots.insert(knots.end(), static_cast<size_t>(d), wire_arc);
        b.wire_knots.push_back(knots);
        const int64_t n_basis_w = static_cast<int64_t>(knots.size()) - d - 1;
        struct Kept {
            int64_t j;
            int kind;  // 0 interior, 1 directional, 2 ground
            int64_t junction;
            bool start;
        };
        std::vector<Kept> kept;
        if (b.start_status[w] == GROUND) kept.push_back({0, 2, -1, true});
        else if (b.start_status[w] >= 0) kept.push_back({0, 1, b.start_status[w], true});
        for (int64_t j = 1; j < n_basis_w - 1; j++) kept.push_back({j, 0, -1, false});
        if (b.end_status[w] == GROUND) kept.push_back({n_basis_w - 1, 2, -1, false});
        else if (b.end_status[w] >= 0) kept.push_back({n_basis_w - 1, 1, b.end_status[w], false});

        const int64_t seg_off = g.seg_offsets[w];
        const int64_t N = pw.n_total;
        // the samples: `arc[:-1, None] + h[:, None] * linspace(0, 1, d + 1)`
        const std::vector<double> unit = linspace(0.0, 1.0, W);
        std::vector<double> x(static_cast<size_t>(N * W));
        for (int64_t s2 = 0; s2 < N; s2++)
            for (int i = 0; i < W; i++)
                x[static_cast<size_t>(s2 * W + i)] =
                    arc[static_cast<size_t>(s2)] + pw.h_per_seg[static_cast<size_t>(s2)] * unit[static_cast<size_t>(i)];
        // `_bspline_basis_rows(x, knots, d)`
        const int64_t n = static_cast<int64_t>(knots.size()) - d - 1;
        const double lo = knots[static_cast<size_t>(d)], hi = knots[static_cast<size_t>(n)];
        for (double v : x)
            if (!std::isfinite(v) || v < lo || v > hi) throw FrontNotPorted("a sample outside the knots");
        std::vector<double> vals(static_cast<size_t>(N * W * W));
        std::vector<int64_t> first(static_cast<size_t>(N * W));
        for (size_t q = 0; q < x.size(); q++) {
            const double xv = x[q];
            int64_t ell = search_right(knots, xv) - 1;
            ell = std::min(std::max<int64_t>(ell, d), n - 1);
            double h[4] = {1.0, 0.0, 0.0, 0.0}, hh[4];
            for (int j = 1; j <= d; j++) {
                for (int c = 0; c < j; c++) hh[c] = h[c];
                h[0] = 0.0;
                for (int nn = 1; nn <= j; nn++) {
                    const double xb = knots[static_cast<size_t>(ell + nn)];
                    const double xa = knots[static_cast<size_t>(ell + nn - j)];
                    const bool same = xb == xa;
                    const double wgt = hh[nn - 1] / (xb - xa);
                    h[nn - 1] = same ? h[nn - 1] : h[nn - 1] + wgt * (xb - xv);
                    h[nn] = same ? 0.0 : wgt * (xv - xa);
                }
            }
            for (int c = 0; c < W; c++) vals[q * W + static_cast<size_t>(c)] = h[c];
            first[q] = ell - d;
        }
        // `_design_matrix_band`, then `einsum("ij,sjk->sik", V, DM)` and the
        // `h ** -p` scaling
        std::vector<double> pps(static_cast<size_t>(N * W * W));  // (s, p, col)
        for (int64_t s2 = 0; s2 < N; s2++) {
            double DM[4][4] = {};
            for (int i = 0; i < W; i++) {
                const int64_t off = first[static_cast<size_t>(s2 * W + i)] - s2;
                if (off < 0 || off > 1) throw FrontNotPorted("design-matrix span offset");
                for (int c = 0; c < W; c++) {
                    const int64_t col = c + off;
                    const double v = vals[static_cast<size_t>((s2 * W + i) * W + c)];
                    if (col == W) {
                        if (v != 0.0) throw FrontNotPorted("design-matrix band");
                    } else {
                        DM[i][col] = v;
                    }
                }
            }
            const double hs = pw.h_per_seg[static_cast<size_t>(s2)];
            for (int i = 0; i < W; i++) {
                const double scale = std::pow(hs, -static_cast<double>(i));
                for (int k = 0; k < W; k++) {
                    double acc = 0.0;
                    for (int j = 0; j < W; j++) acc += V[i * W + j] * DM[j][k];
                    pps[static_cast<size_t>((s2 * W + i) * W + k)] = acc * scale;
                }
            }
        }
        for (const Kept &kb : kept) {
            b.wire_globals[w].push_back({kb.j, m_global});
            const int64_t seg_lo = std::max<int64_t>(0, kb.j - d);
            const int64_t seg_hi = std::min<int64_t>(N, kb.j + 1);
            for (int a = 0; a < W; a++) {
                const int64_t s2 = seg_lo + a;
                const bool on = s2 < seg_hi;
                b.supp_seg.push_back(on ? seg_off + s2 : 0);
                for (int p2 = 0; p2 < W; p2++)
                    b.polys.push_back(on ? pps[static_cast<size_t>((s2 * W + p2) * W + (kb.j - s2))] : 0.0);
            }
            if (kb.kind == 1) {
                junction_dirs[static_cast<size_t>(kb.junction)].push_back({m_global, kb.start ? 1.0 : -1.0});
                (kb.start ? b.start_dir : b.end_dir)[w] = m_global;
            }
            m_global++;
        }
    }
    b.n_basis = m_global;
    b.grounded_junction = grounded_junction;
    // `_node_gap_crossing_junctions`: the grounded junctions a node gap
    // names, with a wire below the interface, that are crossing junctions
    // (momwire#1282): each keeps its KCL row (`_kcl_row_junctions`)
    b.gap_crossing.assign(junctions.size(), false);
    {
        bool any_below = false;
        for (uint8_t v : b.below_wire) any_below = any_below || v != 0;
        // a TWO-RADIUS crossing deck (`_two_radius_crossing`, antennaknobs
        // plan U5) closes every crossing junction with its KCL row
        bool spread = false;
        for (double r : radii) spread = spread || r != radii[0];
        if (has_ground && any_below && spread)
            for (int64_t j : b.crossing)
                if (grounded_junction[static_cast<size_t>(j)]) b.gap_crossing[static_cast<size_t>(j)] = true;
        if (has_ground && any_below && !gap_members.empty())
            for (int64_t j : b.crossing) {
                if (!grounded_junction[static_cast<size_t>(j)]) continue;
                for (const auto &m : junctions[static_cast<size_t>(j)])
                    for (const auto &gm : gap_members)
                        if (m == gm) b.gap_crossing[static_cast<size_t>(j)] = true;
            }
    }
    std::vector<size_t> rows;
    for (size_t j = 0; j < junctions.size(); j++)
        if (!grounded_junction[j] || b.gap_crossing[j]) rows.push_back(j);
    b.n_kcl = static_cast<int64_t>(rows.size());
    b.kcl_A.assign(static_cast<size_t>(b.n_kcl * b.n_basis), 0.0);
    for (size_t r = 0; r < rows.size(); r++)
        for (const auto &ms : junction_dirs[rows[r]])
            b.kcl_A[r * static_cast<size_t>(b.n_basis) + static_cast<size_t>(ms.first)] = ms.second;
    return b;
}

void check_ports(const WireArgs &args) {
    const int64_t n_w = static_cast<int64_t>(args.polylines.size());
    if (args.feeds.empty() && args.node_gaps.empty())
        throw FrontValueError("feeds must contain at least one entry");
    for (size_t i = 0; i < args.feeds.size(); i++)
        if (!(0 <= args.feeds[i].wire && args.feeds[i].wire < n_w))
            throw FrontValueError("feeds[" + std::to_string(i) + "]: wire_index " +
                                  std::to_string(args.feeds[i].wire) + " out of range [0, " +
                                  std::to_string(n_w) + ")");
    // `check_junction_coincidence`: every member within
    // max(floor, 1e-3 of the shortest terminal segment) of the first
    for (const auto &group : args.junctions) {
        std::vector<const double *> anchors;
        double min_seg = 0.0;
        for (size_t q = 0; q < group.size(); q++) {
            const std::vector<double> &pl = args.polylines[static_cast<size_t>(group[q].first)];
            const std::vector<int64_t> &npe = args.n_per_edge[static_cast<size_t>(group[q].first)];
            const size_t m = pl.size() / 3;
            const double *a = group[q].second ? &pl[0] : &pl[3 * (m - 1)];
            const double *o = group[q].second ? &pl[3] : &pl[3 * (m - 2)];
            const double e[3] = {o[0] - a[0], o[1] - a[1], o[2] - a[2]};
            const int64_t count = group[q].second ? npe.front() : npe.back();
            const double len = std::sqrt((e[0] * e[0] + e[1] * e[1]) + e[2] * e[2]) /
                               static_cast<double>(std::max<int64_t>(count, 1));
            min_seg = q == 0 ? len : std::min(min_seg, len);
            anchors.push_back(a);
        }
        // the floor is `_JUNCTION_COINCIDENCE_FLOOR`; a deck the seam meshed
        // shares one node key, far inside any tolerance, so the check is
        // asked conservatively here and anything near it declines
        const double tol = 1e-3 * min_seg;
        for (size_t q = 1; q < anchors.size(); q++) {
            const double d[3] = {anchors[q][0] - anchors[0][0], anchors[q][1] - anchors[0][1],
                                 anchors[q][2] - anchors[0][2]};
            if (std::sqrt((d[0] * d[0] + d[1] * d[1]) + d[2] * d[2]) > tol)
                throw FrontNotPorted("junction-coincidence");
        }
    }
    // `normalize_node_gaps` (no junction ports)
    std::map<std::pair<int64_t, bool>, size_t> end_to_junction;
    for (size_t j = 0; j < args.junctions.size(); j++)
        for (const auto &m : args.junctions[j]) end_to_junction[m] = j;
    std::set<std::pair<int64_t, bool>> seen_members;
    std::set<size_t> seen_junctions;
    for (size_t i = 0; i < args.node_gaps.size(); i++) {
        const auto &gp = args.node_gaps[i];
        const std::string end = gp.at_start ? "start" : "end";
        const std::string at = "node_gaps[" + std::to_string(i) + "]: ";
        if (!(0 <= gp.wire && gp.wire < n_w))
            throw FrontValueError(at + "wire_index " + std::to_string(gp.wire) + " out of range [0, " +
                                  std::to_string(n_w) + ")");
        const std::pair<int64_t, bool> member{gp.wire, gp.at_start};
        auto it = end_to_junction.find(member);
        if (it == end_to_junction.end())
            throw FrontValueError(at + "the " + end + " of wire " + std::to_string(gp.wire) +
                                  " is not a member of any junction group \xe2\x80\x94 a series node gap "
                                  "lives at a junction; for a feed inside a wire use feeds=");
        const size_t j = it->second;
        if (args.junctions[j].size() < 2)
            throw FrontValueError(at + "junction " + std::to_string(j) +
                                  " has a single member \xe2\x80\x94 there is no through-current path "
                                  "to be in series with (a lone-end attachment is junction_ports=; a "
                                  "lone end standing in the ground plane is fed by feeds=(wire, "
                                  "arclength of that end, V), the ground-contact feed, momwire#1120)");
        if (seen_members.count(member))
            throw FrontValueError(at + "the " + end + " of wire " + std::to_string(gp.wire) +
                                  " is listed twice");
        if (seen_junctions.count(j) && args.junctions[j].size() == 2)
            throw FrontValueError(at + "junction " + std::to_string(j) +
                                  " already carries a node gap and joins only two wire ends, so a "
                                  "gap through the other member is the far side of the same cut "
                                  "\xe2\x80\x94 one series gap per two-wire junction (sum the EMFs "
                                  "into one)");
        seen_members.insert(member);
        seen_junctions.insert(j);
    }
}

std::vector<double> node_gap_columns(const WireArgs &args, const Basis &b) {
    const size_t n_g = args.node_gaps.size();
    std::vector<double> cols(static_cast<size_t>(b.n_basis) * n_g, 0.0);
    std::map<std::pair<int64_t, bool>, size_t> end_to_junction;
    for (size_t j = 0; j < args.junctions.size(); j++)
        for (const auto &m : args.junctions[j]) end_to_junction[m] = j;
    for (size_t p = 0; p < n_g; p++) {
        const auto &gp = args.node_gaps[p];
        const size_t j = end_to_junction.at({gp.wire, gp.at_start});
        if (b.grounded_junction[j] && !b.gap_crossing[j])
            throw FrontValueError("node_gaps[" + std::to_string(p) + "]: junction " +
                                  std::to_string(j) +
                                  " is grounded \xe2\x80\x94 a series gap between a wire and the "
                                  "ground stake is not supported (#151 grounds the node through "
                                  "the image instead)");
        const int64_t m = (gp.at_start ? b.start_dir : b.end_dir)[static_cast<size_t>(gp.wire)];
        if (m < 0) throw std::logic_error("a node gap with no directional basis");
        cols[static_cast<size_t>(m) * n_g + p] = gp.at_start ? 1.0 : -1.0;
    }
    return cols;
}

std::vector<double> port_columns(const WireArgs &args, const Geometry &g, const Basis &b) {
    (void)g;
    const int d = b.degree;
    const size_t n_f = args.feeds.size(), n_g = args.node_gaps.size(), n_p = n_f + n_g;
    std::vector<double> B(static_cast<size_t>(b.n_basis) * n_p, 0.0);
    for (size_t f = 0; f < n_f; f++) {
        const int64_t wi = args.feeds[f].wire;
        const std::vector<double> &knots = b.wire_knots[static_cast<size_t>(wi)];
        double s_f = args.feeds[f].arc;
        // the wire-end clip
        const double s_lo = knots[static_cast<size_t>(d)], s_hi = knots[knots.size() - d - 1];
        const double slack = 1e-9 * std::max(s_hi - s_lo, 1.0);
        if (s_lo - slack <= s_f && s_f < s_lo)
            s_f = s_lo;
        else if (s_hi < s_f && s_f <= s_hi + slack)
            s_f = s_hi;
        // `_bspline_design_matrix_dense([s_f], knots, d)[0]`
        const int64_t n = static_cast<int64_t>(knots.size()) - d - 1;
        if (!std::isfinite(s_f) || s_f < knots[static_cast<size_t>(d)] || s_f > knots[static_cast<size_t>(n)])
            throw FrontNotPorted("a feed off its wire's knots");
        int64_t ell = search_right(knots, s_f) - 1;
        ell = std::min(std::max<int64_t>(ell, d), n - 1);
        double h[4] = {1.0, 0.0, 0.0, 0.0}, hh[4];
        for (int j = 1; j <= d; j++) {
            for (int c = 0; c < j; c++) hh[c] = h[c];
            h[0] = 0.0;
            for (int nn = 1; nn <= j; nn++) {
                const double xb = knots[static_cast<size_t>(ell + nn)];
                const double xa = knots[static_cast<size_t>(ell + nn - j)];
                const bool same = xb == xa;
                const double wgt = hh[nn - 1] / (xb - xa);
                h[nn - 1] = same ? h[nn - 1] : h[nn - 1] + wgt * (xb - s_f);
                h[nn] = same ? 0.0 : wgt * (s_f - xa);
            }
        }
        std::vector<double> row(static_cast<size_t>(n), 0.0);
        for (int c = 0; c <= d; c++) row[static_cast<size_t>(ell - d + c)] = h[c];
        for (const auto &jm : b.wire_globals[static_cast<size_t>(wi)])
            B[static_cast<size_t>(jm.second) * n_p + f] = row[static_cast<size_t>(jm.first)];
    }
    const std::vector<double> gc = node_gap_columns(args, b);
    for (int64_t m = 0; m < b.n_basis; m++)
        for (size_t q = 0; q < n_g; q++)
            B[static_cast<size_t>(m) * n_p + n_f + q] = gc[static_cast<size_t>(m) * n_g + q];
    return B;
}

}  // namespace bspline
}  // namespace mw
