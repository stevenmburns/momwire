// PROTOTYPE (momwire#1517 step 1): see `_bspline_front_core.h`. Each step is
// numpy's operation as `bspline.py` spells it, one IEEE operation at a time.
#include "_bspline_front_core.h"

#if defined(_MSC_VER)
#pragma float_control(precise, on)
#pragma fp_contract(off)
#endif

#include <algorithm>
#include <cmath>
#include <map>
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

}  // namespace bspline
}  // namespace mw
