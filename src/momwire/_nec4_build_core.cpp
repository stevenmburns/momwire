// The NEC-4.2 slot's build in C++ (see `_nec4_build_core.h`). Each function
// names the Python it copies, and its float expressions are Python's, in
// Python's order, with no contraction (the tree builds -ffp-contract=off).
// Where Python sums with numpy, the sum is numpy's pairwise_sum (eight
// accumulators past 8 elements, halving past 128, a plain loop from 0.0
// below 8); where it sums a Python list, it is CPython 3.12's `sum()`.

#if defined(_MSC_VER) && !defined(__clang__)
#pragma float_control(precise, on)
#pragma fp_contract(off)
#endif

#include "_nec4_build_core.h"

#include <algorithm>
#include <cmath>
#include <functional>
#include <limits>
#include <map>
#include <set>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace mw {
namespace nec4 {

using nec5::ErrorKind;

namespace {

typedef long long ll;

const double kC = 299792458.0;  // momwire._constants.C_LIGHT
const double kNodeEps = 1e-6;   // deck/_polylines._NODE_EPS
const double kNaN = std::numeric_limits<double>::quiet_NaN();

// A ValueError (or PortalError) the slot's `solve` turns into a refusal.
struct Refused : std::runtime_error {
    std::string message;
    explicit Refused(const std::string &m) : std::runtime_error(m), message(m) {}
};
[[noreturn]] void refuse(const std::string &m) { throw Refused(m); }
[[noreturn]] void decline(const std::string &m) { throw Declined(m); }

// numpy's pairwise_sum over a contiguous float64 run.
double pairwise_sum(const double *a, ll n) {
    if (n < 8) {
        double res = 0.0;
        for (ll i = 0; i < n; i++) res += a[i];
        return res;
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
    return pairwise_sum(a, n2) + pairwise_sum(a + n2, n - n2);
}

// CPython 3.12's sum() of floats from the int 0 (Neumaier).
double py_sum(const std::vector<double> &v) {
    if (v.empty()) return 0.0;
    double f = 0.0 + v[0], c = 0.0;
    for (std::size_t i = 1; i < v.size(); i++) {
        double x = v[i], t = f + x;
        if (std::fabs(f) >= std::fabs(x)) {
            c += (f - t) + x;
        } else {
            c += (x - t) + f;
        }
        f = t;
    }
    if (c != 0.0 && std::isfinite(c)) f += c;
    return f;
}

// Python's float floor division (`_float_div_mod`), for a positive divisor.
double py_floordiv(double vx, double wx) {
    double mod = std::fmod(vx, wx);
    double div = (vx - mod) / wx;
    if (mod != 0.0) {
        if ((wx < 0) != (mod < 0)) div -= 1.0;
    }
    double floordiv;
    if (div != 0.0) {
        floordiv = std::floor(div);
        if (div - floordiv > 0.5) floordiv += 1.0;
    } else {
        floordiv = std::copysign(0.0, vx / wx);
    }
    return floordiv;
}

ll round_key(double v) {
    double r = std::nearbyint(v / kNodeEps);
    if (!(std::fabs(r) < 4.0e18)) decline("a coordinate past the node key's range");
    return static_cast<ll>(r);
}

double dist(const Point3 &a, const Point3 &b) { return py_dist(a.data(), b.data()); }

// numpy's norm3 over one row: sqrt((x*x + y*y) + z*z)
double norm_row(const Point3 &a, const Point3 &b) {
    double x = b[0] - a[0], y = b[1] - a[1], z = b[2] - a[2];
    return std::sqrt((x * x + y * y) + z * z);
}

bool same_material(bool ha, const WireMaterial &a, bool hb, const WireMaterial &b) {
    if (ha != hb) return false;
    if (!ha) return true;
    return a.has_conductivity == b.has_conductivity &&
           (!a.has_conductivity || a.conductivity == b.conductivity) &&
           a.has_insulation == b.has_insulation &&
           (!a.has_insulation ||
            (a.insulation_radius == b.insulation_radius && a.insulation_eps_r == b.insulation_eps_r)) &&
           a.has_distributed == b.has_distributed && (!a.has_distributed || a.distributed == b.distributed);
}

// A model wire as the mesh sees it (after the plane split it may carry
// three vertices).
struct MWire {
    std::vector<Point3> vertices;
    std::vector<ll> edge_elements;
    double radius = 0;
    bool has_material = false;
    WireMaterial material;

    std::vector<double> edge_lengths() const {
        std::vector<double> out;
        for (std::size_t k = 0; k + 1 < vertices.size(); k++) out.push_back(dist(vertices[k], vertices[k + 1]));
        return out;
    }
};

// `_polylines.split_at_plane`
std::vector<MWire> split_at_plane(const Model &model) {
    std::vector<MWire> wires;
    for (const DeckWire &w : model.wires) {
        MWire m;
        m.vertices = {Point3{w.vertices[0][0], w.vertices[0][1], w.vertices[0][2]},
                      Point3{w.vertices[1][0], w.vertices[1][1], w.vertices[1][2]}};
        m.edge_elements = {w.edge_elements};
        m.radius = w.radius;
        m.has_material = w.has_material;
        m.material = w.material;
        wires.push_back(m);
    }
    if (model.ground.kind == Ground::none) return wires;
    const double gz = 0.0;
    for (MWire &w : wires) {
        double tol = 1e-6 * std::max(py_sum(w.edge_lengths()), 1e-9);
        std::vector<Point3> vertices{w.vertices[0]};
        std::vector<ll> elements;
        for (std::size_t e = 0; e + 1 < w.vertices.size(); e++) {
            const Point3 &a = w.vertices[e], &b = w.vertices[e + 1];
            ll n = w.edge_elements[e];
            double za = a[2] - gz, zb = b[2] - gz;
            if ((za < -tol && zb > tol) || (za > tol && zb < -tol)) {
                double t = za / (za - zb);
                Point3 cross{a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t, gz};
                ll n_first = static_cast<ll>(std::nearbyint(static_cast<double>(n) * t));
                n_first = n >= 2 ? std::max<ll>(1, std::min<ll>(n - 1, n_first)) : 1;
                ll n_second = n >= 2 ? n - n_first : 1;
                elements.push_back(n_first);
                vertices.push_back(cross);
                elements.push_back(n_second);
                vertices.push_back(b);
            } else {
                elements.push_back(n);
                vertices.push_back(b);
            }
        }
        w.vertices = vertices;
        w.edge_elements = elements;
    }
    return wires;
}

struct Span {
    Point3 p0{}, p1{};
    ll n = 0;
    ll port = -1;
    double radius = 0;
    bool has_material = false;
    WireMaterial material;
    ll wire = 0, edge = 0;
};

Point3 point(const Point3 &a, const Point3 &b, ll k, ll n) {
    double t = static_cast<double>(k) / static_cast<double>(n);
    return {a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t, a[2] + (b[2] - a[2]) * t};
}

// `_polylines._element_of`
std::pair<ll, ll> element_of(const MWire &w, double arclength) {
    double remaining = arclength;
    std::vector<double> lengths = w.edge_lengths();
    for (std::size_t k = 0; k < lengths.size(); k++) {
        ll n = w.edge_elements[k];
        if (remaining <= lengths[k] || k + 1 == lengths.size()) {
            double step = lengths[k] / static_cast<double>(n);
            ll element = 0;
            if (step > 0.0) {
                double f = py_floordiv(remaining, step);
                if (!(std::fabs(f) < 4.0e18)) decline("a port position past the element index range");
                element = static_cast<ll>(f);
            }
            return {static_cast<ll>(k), std::max<ll>(0, std::min<ll>(n - 1, element))};
        }
        remaining -= lengths[k];
    }
    decline("a wire with no edge");
}

// `_polylines._spans`
std::vector<Span> spans_of(const std::vector<MWire> &wires, const std::vector<std::pair<ll, double>> &ports) {
    std::vector<std::pair<std::array<ll, 3>, ll>> sites;  // insertion order
    for (std::size_t index = 0; index < ports.size(); index++) {
        std::pair<ll, ll> ee = element_of(wires[static_cast<std::size_t>(ports[index].first)], ports[index].second);
        std::array<ll, 3> key{ports[index].first, ee.first, ee.second};
        bool seen = false;
        for (const auto &s : sites) seen = seen || s.first == key;
        if (!seen) sites.push_back({key, static_cast<ll>(index)});
    }
    std::vector<Span> spans;
    for (std::size_t wi = 0; wi < wires.size(); wi++) {
        const MWire &w = wires[wi];
        for (std::size_t edge = 0; edge < w.edge_elements.size(); edge++) {
            ll n = w.edge_elements[edge];
            const Point3 &a = w.vertices[edge], &b = w.vertices[edge + 1];
            std::vector<std::pair<ll, ll>> on_edge;  // (element, port), insertion order
            for (const auto &s : sites)
                if (s.first[0] == static_cast<ll>(wi) && s.first[1] == static_cast<ll>(edge))
                    on_edge.push_back({s.first[2], s.second});
            auto emit = [&](ll first, ll last, ll port) {
                bool whole = first == 0 && last == n;
                Span s;
                s.p0 = whole ? a : point(a, b, first, n);
                s.p1 = whole ? b : point(a, b, last, n);
                s.n = last - first;
                s.port = port;
                s.radius = w.radius;
                s.has_material = w.has_material;
                s.material = w.material;
                s.wire = static_cast<ll>(wi);
                s.edge = static_cast<ll>(edge);
                spans.push_back(s);
            };
            if (on_edge.empty()) {
                emit(0, n, -1);
                continue;
            }
            if (on_edge.size() == 1) {
                ll element = on_edge[0].first;
                if (n % 2 == 1 && element == (n - 1) / 2) {
                    emit(0, n, on_edge[0].second);
                    continue;
                }
            }
            std::set<ll> bset;
            for (const auto &eo : on_edge) {
                bset.insert(eo.first);
                bset.insert(eo.first + 1);
            }
            bset.erase(0);
            bset.erase(n);
            std::vector<ll> bounds(bset.begin(), bset.end());
            bounds.push_back(n);
            ll previous = 0;
            for (ll boundary : bounds) {
                ll count = boundary - previous;
                ll port = -1;
                if (count == 1)
                    for (const auto &eo : on_edge)
                        if (eo.first == boundary - 1) port = eo.second;
                emit(previous, boundary, port);
                previous = boundary;
            }
        }
    }
    return spans;
}

struct Mesh {
    std::vector<std::vector<Point3>> polylines;
    std::vector<std::vector<ll>> edge_elements;
    std::vector<std::vector<JunctionEnd>> junctions;
    std::vector<double> radii;
    std::vector<bool> has_material;
    std::vector<WireMaterial> materials;
    std::vector<std::pair<ll, double>> ports;  // (polyline, arclength)
    std::vector<ll> port_order;
};

// `_polylines.to_polylines`
Mesh to_polylines(const Model &model, const std::vector<std::pair<ll, double>> &ports) {
    std::vector<MWire> wires = split_at_plane(model);
    std::vector<Span> spans = spans_of(wires, ports);
    if (spans.empty()) refuse("the model has no wires");

    std::map<std::array<ll, 3>, ll> node_of;
    std::vector<Point3> nodes;
    auto node_id = [&](const Point3 &p) {
        std::array<ll, 3> key{round_key(p[0]), round_key(p[1]), round_key(p[2])};
        auto it = node_of.find(key);
        if (it != node_of.end()) return it->second;
        ll id = static_cast<ll>(nodes.size());
        node_of[key] = id;
        nodes.push_back(p);
        return id;
    };
    struct Edge {
        ll a, b, span;
    };
    std::vector<Edge> edges;
    for (std::size_t index = 0; index < spans.size(); index++) {
        ll a = node_id(spans[index].p0), b = node_id(spans[index].p1);
        if (a == b)
            refuse("wire " + std::to_string(spans[index].wire) +
                   " has a span whose two ends fall on one node — a conductor shorter than the 1e-06 m "
                   "node tolerance");
        edges.push_back({a, b, static_cast<ll>(index)});
    }
    std::vector<std::vector<std::pair<ll, ll>>> adjacency(nodes.size());  // (other node, edge)
    for (std::size_t e = 0; e < edges.size(); e++) {
        adjacency[static_cast<std::size_t>(edges[e].a)].push_back({edges[e].b, static_cast<ll>(e)});
        adjacency[static_cast<std::size_t>(edges[e].b)].push_back({edges[e].a, static_cast<ll>(e)});
    }
    std::vector<bool> boundary(nodes.size());
    for (std::size_t n = 0; n < nodes.size(); n++) boundary[n] = adjacency[n].size() != 2;
    for (std::size_t n = 0; n < nodes.size(); n++) {
        if (boundary[n]) continue;
        const Span &f = spans[static_cast<std::size_t>(edges[static_cast<std::size_t>(adjacency[n][0].second)].span)];
        const Span &s = spans[static_cast<std::size_t>(edges[static_cast<std::size_t>(adjacency[n][1].second)].span)];
        if (f.radius != s.radius || !same_material(f.has_material, f.material, s.has_material, s.material))
            boundary[n] = true;
    }
    if (model.ground.kind != Ground::none) {
        const double gz = 0.0;
        for (std::size_t n = 0; n < nodes.size(); n++) {
            if (boundary[n] || std::fabs(nodes[n][2] - gz) > kNodeEps) continue;
            double s0 = nodes[static_cast<std::size_t>(adjacency[n][0].first)][2] - gz;
            double s1 = nodes[static_cast<std::size_t>(adjacency[n][1].first)][2] - gz;
            if ((s0 < -kNodeEps && s1 > kNodeEps) || (s0 > kNodeEps && s1 < -kNodeEps)) boundary[n] = true;
        }
    }

    std::vector<bool> walked(edges.size(), false);
    std::vector<std::vector<ll>> paths;
    Mesh mesh;
    std::vector<std::pair<ll, ll>> span_place(spans.size(), {-1, -1});
    std::vector<std::pair<ll, std::vector<JunctionEnd>>> junction_ends;  // insertion order
    std::map<ll, std::size_t> junction_at;
    auto ends_of = [&](ll node) -> std::vector<JunctionEnd> & {
        auto it = junction_at.find(node);
        if (it == junction_at.end()) {
            junction_at[node] = junction_ends.size();
            junction_ends.push_back({node, {}});
            return junction_ends.back().second;
        }
        return junction_ends[it->second].second;
    };
    for (std::size_t n = 0; n < nodes.size(); n++)
        if (boundary[n]) ends_of(static_cast<ll>(n));

    auto walk = [&](ll start, ll first_edge, std::vector<ll> &path_nodes, std::vector<ll> &path_edges) {
        path_nodes = {start};
        path_edges.clear();
        ll previous_edge = -1, current = start, next_edge = first_edge;
        for (;;) {
            walked[static_cast<std::size_t>(next_edge)] = true;
            path_edges.push_back(next_edge);
            const Edge &e = edges[static_cast<std::size_t>(next_edge)];
            current = e.a == current ? e.b : e.a;
            path_nodes.push_back(current);
            if (boundary[static_cast<std::size_t>(current)]) return;
            previous_edge = next_edge;
            ll found = -1;
            for (const auto &ne : adjacency[static_cast<std::size_t>(current)])
                if (ne.second != previous_edge) {
                    found = ne.second;
                    break;
                }
            next_edge = found;
        }
    };
    auto record = [&](const std::vector<ll> &path_nodes, const std::vector<ll> &path_edges) {
        ll index = static_cast<ll>(paths.size());
        paths.push_back(path_nodes);
        std::vector<ll> counts;
        for (ll e : path_edges) counts.push_back(spans[static_cast<std::size_t>(edges[static_cast<std::size_t>(e)].span)].n);
        mesh.edge_elements.push_back(counts);
        const Span &head = spans[static_cast<std::size_t>(edges[static_cast<std::size_t>(path_edges[0])].span)];
        mesh.radii.push_back(head.radius);
        mesh.has_material.push_back(head.has_material);
        mesh.materials.push_back(head.material);
        for (std::size_t position = 0; position < path_edges.size(); position++) {
            ll span = edges[static_cast<std::size_t>(path_edges[position])].span;
            span_place[static_cast<std::size_t>(span)] = {index, static_cast<ll>(position)};
        }
        return index;
    };

    std::vector<ll> pn, pe;
    for (std::size_t start = 0; start < nodes.size(); start++) {
        if (!boundary[start]) continue;
        for (const auto &ne : std::vector<std::pair<ll, ll>>(adjacency[start])) {
            if (walked[static_cast<std::size_t>(ne.second)]) continue;
            walk(static_cast<ll>(start), ne.second, pn, pe);
            ll index = record(pn, pe);
            ends_of(pn.front()).push_back({index, true});
            ends_of(pn.back()).push_back({index, false});
        }
    }

    for (;;) {
        ll seed = -1;
        for (std::size_t e = 0; e < walked.size(); e++)
            if (!walked[e]) {
                seed = static_cast<ll>(e);
                break;
            }
        if (seed < 0) break;
        std::vector<ll> component, stack{seed};
        std::set<ll> seen{seed};
        while (!stack.empty()) {
            ll edge = stack.back();
            stack.pop_back();
            component.push_back(edge);
            walked[static_cast<std::size_t>(edge)] = true;
            for (ll endpoint : {edges[static_cast<std::size_t>(edge)].a, edges[static_cast<std::size_t>(edge)].b})
                for (const auto &ne : adjacency[static_cast<std::size_t>(endpoint)])
                    if (!seen.count(ne.second) && !walked[static_cast<std::size_t>(ne.second)]) {
                        seen.insert(ne.second);
                        stack.push_back(ne.second);
                    }
        }
        ll cut = -1;
        for (ll e : component)
            if (spans[static_cast<std::size_t>(edges[static_cast<std::size_t>(e)].span)].port >= 0) {
                cut = e;
                break;
            }
        if (cut < 0) cut = *std::min_element(component.begin(), component.end());
        ll cut_a = edges[static_cast<std::size_t>(cut)].a, cut_b = edges[static_cast<std::size_t>(cut)].b;
        ll cut_index = record({cut_a, cut_b}, {cut});
        boundary[static_cast<std::size_t>(cut_a)] = true;
        boundary[static_cast<std::size_t>(cut_b)] = true;
        ends_of(cut_a);
        ends_of(cut_b);
        for (ll e : component)
            if (e != cut) walked[static_cast<std::size_t>(e)] = false;
        ll first = -1;
        for (const auto &ne : adjacency[static_cast<std::size_t>(cut_b)])
            if (ne.second != cut && !walked[static_cast<std::size_t>(ne.second)]) {
                first = ne.second;
                break;
            }
        if (first < 0) decline("a cycle with no edge left to walk");
        walk(cut_b, first, pn, pe);
        ll loop_index = record(pn, pe);
        ends_of(cut_a).push_back({cut_index, true});
        ends_of(cut_a).push_back({loop_index, false});
        ends_of(cut_b).push_back({cut_index, false});
        ends_of(cut_b).push_back({loop_index, true});
    }

    for (const auto &path : paths) {
        std::vector<Point3> pl;
        for (ll n : path) pl.push_back(nodes[static_cast<std::size_t>(n)]);
        mesh.polylines.push_back(pl);
    }
    std::vector<std::vector<double>> lengths;
    for (const auto &pl : mesh.polylines) {
        std::vector<double> l;
        for (std::size_t k = 0; k + 1 < pl.size(); k++) l.push_back(norm_row(pl[k], pl[k + 1]));
        lengths.push_back(l);
    }
    for (const auto &je : junction_ends)
        if (je.second.size() >= 2) mesh.junctions.push_back(je.second);
    for (std::size_t s = 0; s < spans.size(); s++) {
        if (spans[s].port < 0) continue;
        ll polyline = span_place[s].first, position = span_place[s].second;
        const std::vector<double> &el = lengths[static_cast<std::size_t>(polyline)];
        double arclength = pairwise_sum(el.data(), position) + 0.5 * el[static_cast<std::size_t>(position)];
        mesh.ports.push_back({polyline, arclength});
        mesh.port_order.push_back(spans[s].port);
    }
    return mesh;
}

// --- deck._solver ----------------------------------------------------------

struct Prepared {
    Mesh mesh;
    PortPlan ports;
};

std::string load_index_message(ll index, ll other) {
    return "load " + std::to_string(index) + " shares its position with load " + std::to_string(other) +
           " — one gap cannot carry two loads";
}

// `_solver._sites` then `prepare_mesh`
Prepared prepare_mesh(const Model &model) {
    std::vector<PortSite> sites;
    std::vector<std::pair<std::pair<ll, double>, ll>> at;  // (wire, arclength) -> site
    auto find = [&](ll w, double a) -> ll {
        for (const auto &e : at)
            if (e.first.first == w && e.first.second == a) return e.second;
        return -1;
    };
    std::vector<ll> feed_ports, load_ports;
    std::vector<std::set<ll>> together;
    {
        bool any = false;
        for (std::size_t g = 0; g < model.groups.size(); g++)
            if (model.group_ran[g]) {
                any = true;
                together.push_back(std::set<ll>(model.groups[g].loads.begin(), model.groups[g].loads.end()));
            }
        if (!any) {
            std::set<ll> every;
            for (std::size_t k = 0; k < model.loads.size(); k++) every.insert(static_cast<ll>(k));
            together.push_back(every);
        }
    }
    for (std::size_t index = 0; index < model.feeds.size(); index++) {
        const Feed &f = model.feeds[index];
        ll existing = find(f.wire, f.arclength);
        if (existing >= 0)
            refuse("feed " + std::to_string(index) + " shares its position with feed " +
                   std::to_string(sites[static_cast<std::size_t>(existing)].feed) +
                   " — one gap cannot carry two sources");
        at.push_back({{f.wire, f.arclength}, static_cast<ll>(sites.size())});
        feed_ports.push_back(static_cast<ll>(sites.size()));
        PortSite s;
        s.wire = f.wire;
        s.arclength = f.arclength;
        s.feed = static_cast<ll>(index);
        sites.push_back(s);
    }
    for (std::size_t index = 0; index < model.loads.size(); index++) {
        const Load &l = model.loads[index];
        ll existing = find(l.wire, l.arclength);
        if (existing < 0) {
            at.push_back({{l.wire, l.arclength}, static_cast<ll>(sites.size())});
            load_ports.push_back(static_cast<ll>(sites.size()));
            PortSite s;
            s.wire = l.wire;
            s.arclength = l.arclength;
            s.load = static_cast<ll>(index);
            s.has_load_spec = true;
            s.load_spec = l.spec;
            sites.push_back(s);
            continue;
        }
        PortSite &site = sites[static_cast<std::size_t>(existing)];
        if (site.load >= 0) {
            for (std::size_t k = 0; k < index; k++) {
                const Load &o = model.loads[k];
                if (!(o.wire == l.wire && o.arclength == l.arclength)) continue;
                for (const auto &live : together)
                    if (live.count(static_cast<ll>(index)) && live.count(static_cast<ll>(k)))
                        refuse(load_index_message(static_cast<ll>(index), site.load));
            }
            load_ports.push_back(existing);
            continue;
        }
        load_ports.push_back(existing);
        site.load = static_cast<ll>(index);
        site.has_load_spec = true;
        site.load_spec = l.spec;
    }
    {
        std::vector<std::pair<ll, double>> endpoints;
        for (const NetworkCard &n : model.networks)
            for (auto end : {std::make_pair(n.wire_a, n.arclength_a), std::make_pair(n.wire_b, n.arclength_b)}) {
                bool seen = false;
                for (const auto &e : endpoints) seen = seen || (e.first == end.first && e.second == end.second);
                if (!seen) endpoints.push_back(end);
            }
        for (const auto &key : endpoints) {
            ll existing = find(key.first, key.second);
            if (existing < 0) {
                at.push_back({key, static_cast<ll>(sites.size())});
                PortSite s;
                s.wire = key.first;
                s.arclength = key.second;
                s.network = true;
                sites.push_back(s);
                continue;
            }
            sites[static_cast<std::size_t>(existing)].network = true;
        }
    }
    std::vector<std::pair<ll, double>> positions;
    for (const PortSite &s : sites) positions.push_back({s.wire, s.arclength});
    Prepared p;
    p.mesh = to_polylines(model, positions);
    if (p.mesh.port_order.size() != sites.size())
        decline("two port sites in one element (Python raises KeyError)");
    std::vector<ll> solver_of_model(sites.size(), -1);
    for (std::size_t i = 0; i < p.mesh.port_order.size(); i++) {
        ll m = p.mesh.port_order[i];
        p.ports.sites.push_back(sites[static_cast<std::size_t>(m)]);
        solver_of_model[static_cast<std::size_t>(m)] = static_cast<ll>(i);
    }
    for (ll f : feed_ports) p.ports.feed_ports.push_back(solver_of_model[static_cast<std::size_t>(f)]);
    for (ll l : load_ports) p.ports.load_ports.push_back(solver_of_model[static_cast<std::size_t>(l)]);
    for (std::size_t g = 0; g < model.groups.size(); g++) {
        p.ports.voltages_present.push_back(model.group_ran[g]);
        std::vector<std::complex<double>> drive;
        if (model.group_ran[g]) {
            drive.assign(p.ports.sites.size(), std::complex<double>(0.0, 0.0));
            const auto &v = model.groups[g].voltages;
            for (std::size_t fi = 0; fi < v.size(); fi++)
                drive[static_cast<std::size_t>(p.ports.feed_ports[fi])] = v[fi];
        }
        p.ports.voltages.push_back(drive);
    }
    auto site_of = [&](ll w, double a) -> ll {
        for (std::size_t i = 0; i < p.ports.sites.size(); i++)
            if (p.ports.sites[i].wire == w && p.ports.sites[i].arclength == a) return static_cast<ll>(i);
        return -1;
    };
    for (const NetworkCard &n : model.networks)
        p.ports.network_ports.push_back({site_of(n.wire_a, n.arclength_a), site_of(n.wire_b, n.arclength_b)});
    return p;
}

// The two bases' declared sentences (`Capabilities.refusal` / `.partials`).
const char *const kSinBuriedEk =
    "extended_kernel=True + a wire below the ground plane is not served: the extended kernel extends "
    "every pair of the geometry, and momwire#553 measured neither what that means for a pair spanning "
    "two media (the tube expansion's O(a^2) term is written at one wavenumber) nor for the image of a "
    "buried source, which lands in the OTHER medium. Solve the buried deck with extended_kernel=False, "
    "which is the default";

double ground_touch_tol(const std::vector<Point3> &pl) {
    std::vector<double> l;
    for (std::size_t k = 0; k + 1 < pl.size(); k++) {
        double x = pl[k + 1][0] - pl[k][0], y = pl[k + 1][1] - pl[k][1], z = pl[k + 1][2] - pl[k][2];
        double sq[3] = {x * x, y * y, z * z};
        l.push_back(std::sqrt(pairwise_sum(sq, 3)));
    }
    double length = pairwise_sum(l.data(), static_cast<ll>(l.size()));
    return 1e-6 * std::max(length, 1e-30);
}

std::vector<ll> wires_below(const std::vector<std::vector<Point3>> &polylines, const Ground &ground) {
    std::vector<ll> out;
    if (ground.kind == Ground::none) return out;
    for (std::size_t i = 0; i < polylines.size(); i++) {
        double mn = polylines[i][0][2];
        for (const Point3 &p : polylines[i]) mn = std::min(mn, p[2]);  // NaN-free here
        if (mn < 0.0 - ground_touch_tol(polylines[i])) out.push_back(static_cast<ll>(i));
    }
    return out;
}

std::string repr_str(const std::string &s) { return "'" + s + "'"; }

// `build_solver` (through `construct_solver`) at one operating point.
OperatingPoint build_point(const Model &model, const Prepared &prep, const std::string &basis, ll group,
                           double frequency_mhz, bool extended_kernel, const Environment &env) {
    if (frequency_mhz <= 0.0) refuse("frequency must be > 0 MHz, got " + py_repr(frequency_mhz));
    OperatingPoint op;
    op.frequency_mhz = frequency_mhz;
    op.ports = prep.ports;
    op.group = group;
    SolverArgs &a = op.args;
    a.wires = prep.mesh.polylines;
    a.n_per_edge = prep.mesh.edge_elements;
    const bool have_drive = group >= 0 && model.group_ran[static_cast<std::size_t>(group)];
    for (std::size_t i = 0; i < prep.mesh.ports.size(); i++) {
        SolverFeed f;
        f.polyline = prep.mesh.ports[i].first;
        f.arclength = prep.mesh.ports[i].second;
        if (have_drive) f.volts = prep.ports.voltages[static_cast<std::size_t>(group)][i];
        a.feeds.push_back(f);
    }
    a.wavelength = kC / (frequency_mhz * 1e6);
    a.radii = prep.mesh.radii;
    a.uniform_radius = std::all_of(a.radii.begin(), a.radii.end(), [&](double r) { return r == a.radii[0]; });
    a.junctions = prep.mesh.junctions;
    a.ground = env.ground;
    a.exact_kernel_auto = basis == "bspline";
    // _wire_loading
    std::size_t n = prep.mesh.materials.size();
    a.conductivity.assign(n, kNaN);
    a.insulation_radius.assign(n, kNaN);
    a.insulation_eps_r.assign(n, kNaN);
    a.distributed_present.assign(n, false);
    a.distributed.assign(n, DistributedRLC());
    for (std::size_t i = 0; i < n; i++) {
        if (!prep.mesh.has_material[i]) continue;
        const WireMaterial &m = prep.mesh.materials[i];
        if (m.has_conductivity) a.conductivity[i] = m.conductivity;
        if (m.has_insulation) {
            a.insulation_radius[i] = m.insulation_radius;
            a.insulation_eps_r[i] = m.insulation_eps_r;
        }
        if (m.has_distributed) {
            a.distributed_present[i] = true;
            a.distributed[i] = m.distributed;
        }
    }
    auto finite_any = [](const std::vector<double> &v) {
        return std::any_of(v.begin(), v.end(), [](double x) { return std::isfinite(x); });
    };
    a.has_conductivity = finite_any(a.conductivity);
    a.has_insulation = finite_any(a.insulation_radius);
    a.has_distributed = std::any_of(a.distributed_present.begin(), a.distributed_present.end(), [](bool b) { return b; });

    // construct_solver: the extended kernel as the dialect's default
    bool ek = extended_kernel;
    const bool fallback_default = model.extended_kernel_default;
    if (ek && fallback_default) {
        const char *reason = nullptr;
        if (basis == "sinusoidal" && !wires_below(a.wires, a.ground).empty()) reason = kSinBuriedEk;
        if (reason) {
            ek = false;
            op.advisories.push_back("basis " + repr_str(basis) +
                                    ": this deck's dialect solves with the extended kernel by default, "
                                    "which this basis cannot serve here, so it solved with the reduced "
                                    "kernel: " +
                                    reason + " (momwire#1326)");
        }
    }
    if (ek && basis == "bspline") {
        std::vector<ll> below = wires_below(a.wires, a.ground);
        if (!below.empty()) {
            if (fallback_default && below.size() == a.wires.size()) {
                ek = false;
                op.advisories.push_back(
                    "basis " + repr_str(basis) +
                    ": this deck's dialect solves with the extended kernel by default, which this basis "
                    "cannot serve here, so it solved with the reduced kernel: every wire of this deck is "
                    "below the ground plane, where the extended kernel is not derived yet (momwire#1170) "
                    "(momwire#1326)");
            } else {
                std::string asked = fallback_default
                                        ? "this deck's dialect solves with the extended kernel by default"
                                        : "this deck asks for the extended kernel";
                op.advisories.push_back(
                    "basis " + repr_str(basis) + ": " + asked +
                    ", and it has wires below the ground plane. The above-ground wires solved with the "
                    "extended kernel; the " +
                    std::to_string(below.size()) + " of " + std::to_string(a.wires.size()) +
                    " wires below the plane kept the reduced kernel, on every pair that touches them, "
                    "because the extended kernel is not derived for the lower medium yet (momwire#1170)");
            }
        }
    }
    a.extended_kernel = ek;
    op.extended_kernel = ek;
    op.wavelength = kC / (frequency_mhz * 1e6);

    // _port_signs over the constructor's arguments
    for (std::size_t index = 0; index < op.ports.sites.size(); index++) {
        const std::vector<Point3> &pl = a.wires[static_cast<std::size_t>(a.feeds[index].polyline)];
        double arclength = a.feeds[index].arclength;
        std::vector<double> lengths;
        for (std::size_t k = 0; k + 1 < pl.size(); k++) lengths.push_back(norm_row(pl[k], pl[k + 1]));
        double cum = 0;
        std::size_t edge = lengths.size();
        for (std::size_t k = 0; k < lengths.size(); k++) {
            cum = k == 0 ? lengths[0] : cum + lengths[k];
            if (cum >= arclength) {
                edge = k;
                break;
            }
        }
        edge = std::min(edge, lengths.size() - 1);
        Point3 walk{pl[edge + 1][0] - pl[edge][0], pl[edge + 1][1] - pl[edge][1], pl[edge + 1][2] - pl[edge][2]};
        const DeckWire &w = model.wires[static_cast<std::size_t>(op.ports.sites[index].wire)];
        double au[3] = {w.vertices[1][0] - w.vertices[0][0], w.vertices[1][1] - w.vertices[0][1],
                        w.vertices[1][2] - w.vertices[0][2]};
        double dot = (walk[0] * au[0] + walk[1] * au[1]) + walk[2] * au[2];
        op.signs.push_back(dot >= 0.0 ? 1.0 : -1.0);
    }
    return op;
}

bool same_ground(const Ground &x, const Ground &y) {
    if (x.kind != y.kind) return false;
    if (x.kind == Ground::finite || x.kind == Ground::finite_fast) return x.eps_r == y.eps_r && x.sigma == y.sigma;
    return true;
}

// `structure.locate` over the geometry view, then `global_segment`.
ll global_segment_of(const Geometry &g, const Int &tag, const Int &seg_in) {
    Int seg = seg_in < Int(1) ? Int(1) : seg_in;
    ll acc = 0;
    for (std::size_t i = 0; i < g.wires.size(); i++) {
        const GeometryWire &w = g.wires[i];
        if (!tag.is_zero() && w.tag != tag) continue;
        if (Int(acc + w.n_seg) >= seg) {
            ll before = 0;
            for (std::size_t j = 0; j < i; j++) before += g.wires[j].n_seg;
            return before + (seg.i64() - acc);
        }
        acc += w.n_seg;
    }
    decline("a feed address the geometry does not hold");
}

// `DeckSolver.__init__`, then `at` per run and frequency of the plan.
void build_structure(const StructureSeam &s, const std::string &basis, StructureBuild &out) {
    const PortalDeck &deck = s.deck;
    const Model &model = deck.model;
    BuildTables &t = out.tables;
    for (std::size_t g = 0; g < deck.groups.size(); g++) {
        if (!deck.group_ran[g]) continue;
        for (const PortalSource &src : deck.groups[g].sources) {
            bool seen = false;
            for (const auto &p : t.ports) seen = seen || (p.first == src.tag && p.second == src.seg);
            if (!seen) t.ports.push_back({src.tag, src.seg});
        }
    }
    if (t.ports.empty()) refuse("deck has no EX card — nothing drives the structure");
    ll number = 0;
    for (std::size_t wi = 0; wi < deck.structure.wires.size(); wi++) {
        const GeometryWire &w = deck.structure.wires[wi];
        double step[3];
        for (int c = 0; c < 3; c++) step[c] = (w.p2[c] - w.p1[c]) / static_cast<double>(w.n_seg);
        for (ll k = 1; k <= w.n_seg; k++) {
            SegmentRow r;
            r.number = ++number;
            r.tag = w.tag;
            r.wire = static_cast<ll>(wi);
            r.local = k;
            double h = static_cast<double>(k) - 0.5;
            for (int c = 0; c < 3; c++) {
                r.centre[c] = w.p1[c] + h * step[c];
                r.direction[c] = step[c];
            }
            r.radius = w.radius;
            t.segments.push_back(r);
        }
    }
    for (std::size_t g = 0; g < model.groups.size(); g++)
        if (model.group_ran[g]) {
            t.group = static_cast<ll>(g);
            break;
        }
    std::size_t armed = deck.groups.size();
    for (std::size_t g = 0; g < deck.groups.size(); g++)
        if (deck.group_ran[g]) {
            armed = g;
            break;
        }
    if (armed == deck.groups.size()) decline("a served structure with no run");
    if (deck.structure.wires.empty()) decline("a structure with no wires");
    t.smallest_radius = deck.structure.wires[0].radius;
    for (const GeometryWire &w : deck.structure.wires)
        if (w.radius < t.smallest_radius) t.smallest_radius = w.radius;

    Prepared prep = prepare_mesh(model);
    const PortalGroup &a = deck.groups[armed];
    struct Key {
        double freq;
        bool ek;
        Ground ground;
    };
    std::vector<Key> keys;
    auto at = [&](double freq, bool ek, const Environment &env) {
        for (const Key &k : keys)
            if (k.freq == freq && k.ek == ek && same_ground(k.ground, env.ground)) return;
        out.points.push_back(build_point(model, prep, basis, t.group, freq, ek, env));
        keys.push_back({freq, ek, env.ground});
    };
    at(a.freqs_mhz.empty() ? 0.0 : a.freqs_mhz[0], a.ek, a.environment);

    const PortPlan &plan = out.points[0].ports;
    t.feed_index = plan.feed_ports;
    for (std::size_t i = 0; i < plan.sites.size(); i++)
        if (plan.sites[i].has_load_spec) t.load_ports.push_back({static_cast<ll>(i), plan.sites[i].load_spec});
    for (std::size_t i = 0; i < plan.load_ports.size() && i < model.loads.size(); i++)
        t.model_loads.push_back({plan.load_ports[i], model.loads[i].spec});
    for (std::size_t g = 0; g < model.groups.size(); g++)
        if (model.group_ran[g]) t.group_loads.push_back({static_cast<ll>(g), model.groups[g].loads});
    for (std::size_t i = 0; i < t.ports.size() && i < t.feed_index.size(); i++) {
        ll segment = global_segment_of(deck.structure, t.ports[i].first, t.ports[i].second);
        bool replaced = false;
        for (auto &kv : t.port_by_segment)
            if (kv.first == segment) {
                kv.second = t.feed_index[i];
                replaced = true;
            }
        if (!replaced) t.port_by_segment.push_back({segment, t.feed_index[i]});
    }
    for (const DeckWire &w : model.wires) t.lossy = t.lossy || w.has_material;
    std::set<ll> current;
    for (ll i : model.current_feeds) current.insert(t.feed_index[static_cast<std::size_t>(i)]);
    t.current_ports.assign(current.begin(), current.end());

    for (ll index : s.plan) {
        const PortalGroup &g = deck.groups[static_cast<std::size_t>(index)];
        for (double freq : g.freqs_mhz) at(freq, g.ek, g.environment);
    }
}

// --- the comparison format ---------------------------------------------------

struct Out {
    std::string s;
    void line(const std::string &l) { s += l + "\n"; }
    void seq(const std::string &p, const char *kind, std::size_t n) {
        line(p + " : " + kind + " " + std::to_string(n));
    }
    void cls(const std::string &p, const char *name) { line(p + " : " + name); }
    void none(const std::string &p) { line(p + " = None"); }
    void b(const std::string &p, bool v) { line(p + " = bool " + (v ? "True" : "False")); }
    void i(const std::string &p, const Int &v) { line(p + " = int " + v.str()); }
    void i(const std::string &p, ll v) { line(p + " = int " + std::to_string(v)); }
    void f(const std::string &p, double v) { line(p + " = float " + nec5::dump_bits(v)); }
    void c(const std::string &p, std::complex<double> v) {
        line(p + " = complex " + nec5::dump_bits(v.real()) + " " + nec5::dump_bits(v.imag()));
    }
    void str(const std::string &p, const std::string &utf8) { line(p + " = str " + nec5::dump_hex(utf8)); }
};

std::string at(const std::string &p, std::size_t k) { return p + "[" + std::to_string(k) + "]"; }

void dump_load_spec(Out &o, const std::string &p, const LoadSpec &s) {
    o.cls(p, "LoadSpec");
    o.str(p + ".kind", s.kind);
    o.f(p + ".r", s.r);
    o.f(p + ".l", s.l);
    o.f(p + ".c", s.c);
    o.f(p + ".x", s.x);
}

void dump_floats(Out &o, const std::string &p, const std::vector<double> &v) {
    o.seq(p, "list", v.size());
    for (std::size_t k = 0; k < v.size(); k++) o.f(at(p, k), v[k]);
}

void dump_args(Out &o, const std::string &p, const SolverArgs &a) {
    // The kwargs in the order construct_solver passes them.
    std::vector<std::pair<std::string, std::function<void(const std::string &)>>> kw;
    kw.push_back({"wires", [&](const std::string &q) {
                      o.seq(q, "list", a.wires.size());
                      for (std::size_t i = 0; i < a.wires.size(); i++) {
                          o.seq(at(q, i), "list", a.wires[i].size());
                          for (std::size_t j = 0; j < a.wires[i].size(); j++) {
                              std::string r = at(at(q, i), j);
                              o.seq(r, "list", 3);
                              for (std::size_t c = 0; c < 3; c++) o.f(at(r, c), a.wires[i][j][c]);
                          }
                      }
                  }});
    kw.push_back({"n_per_edge_per_wire", [&](const std::string &q) {
                      o.seq(q, "list", a.n_per_edge.size());
                      for (std::size_t i = 0; i < a.n_per_edge.size(); i++) {
                          o.seq(at(q, i), "list", a.n_per_edge[i].size());
                          for (std::size_t j = 0; j < a.n_per_edge[i].size(); j++)
                              o.i(at(at(q, i), j), a.n_per_edge[i][j]);
                      }
                  }});
    kw.push_back({"feeds", [&](const std::string &q) {
                      o.seq(q, "list", a.feeds.size());
                      for (std::size_t i = 0; i < a.feeds.size(); i++) {
                          std::string r = at(q, i);
                          o.seq(r, "tuple", 3);
                          o.i(at(r, 0), a.feeds[i].polyline);
                          o.f(at(r, 1), a.feeds[i].arclength);
                          o.c(at(r, 2), a.feeds[i].volts);
                      }
                  }});
    kw.push_back({"wavelength", [&](const std::string &q) { o.f(q, a.wavelength); }});
    kw.push_back({"wire_radius", [&](const std::string &q) {
                      if (a.uniform_radius) {
                          o.f(q, a.radii[0]);
                      } else {
                          dump_floats(o, q, a.radii);
                      }
                  }});
    kw.push_back({"cancel", [&](const std::string &q) { o.none(q); }});
    kw.push_back({"junctions", [&](const std::string &q) {
                      if (a.junctions.empty()) return o.none(q);
                      o.seq(q, "list", a.junctions.size());
                      for (std::size_t i = 0; i < a.junctions.size(); i++) {
                          o.seq(at(q, i), "list", a.junctions[i].size());
                          for (std::size_t j = 0; j < a.junctions[i].size(); j++) {
                              std::string r = at(at(q, i), j);
                              o.seq(r, "tuple", 2);
                              o.i(at(r, 0), a.junctions[i][j].polyline);
                              o.str(at(r, 1), a.junctions[i][j].start ? "start" : "end");
                          }
                      }
                  }});
    if (a.extended_kernel) kw.push_back({"extended_kernel", [&](const std::string &q) { o.b(q, true); }});
    if (a.has_conductivity)
        kw.push_back({"wire_conductivity", [&](const std::string &q) { dump_floats(o, q, a.conductivity); }});
    if (a.has_insulation) {
        kw.push_back({"insulation_radius", [&](const std::string &q) { dump_floats(o, q, a.insulation_radius); }});
        kw.push_back({"insulation_eps_r", [&](const std::string &q) { dump_floats(o, q, a.insulation_eps_r); }});
    }
    if (a.has_distributed)
        kw.push_back({"distributed_rlc", [&](const std::string &q) {
                          o.seq(q, "list", a.distributed.size());
                          for (std::size_t i = 0; i < a.distributed.size(); i++) {
                              std::string r = at(q, i);
                              if (!a.distributed_present[i]) {
                                  o.none(r);
                                  continue;
                              }
                              o.cls(r, "DistributedRLC");
                              o.str(r + ".kind", a.distributed[i].kind);
                              o.f(r + ".r", a.distributed[i].r);
                              o.f(r + ".l", a.distributed[i].l);
                              o.f(r + ".c", a.distributed[i].c);
                          }
                      }});
    if (a.ground.kind == Ground::none) {
        kw.push_back({"ground_z", [&](const std::string &q) { o.none(q); }});
    } else {
        kw.push_back({"ground_z", [&](const std::string &q) { o.f(q, 0.0); }});
        if (a.ground.kind != Ground::pec) {
            kw.push_back({"ground_eps", [&](const std::string &q) {
                              o.seq(q, "tuple", 2);
                              o.f(at(q, 0), a.ground.eps_r);
                              o.f(at(q, 1), a.ground.sigma);
                          }});
            if (a.ground.kind == Ground::finite)
                kw.push_back({"ground_model", [&](const std::string &q) { o.str(q, "sommerfeld"); }});
        }
    }
    if (a.exact_kernel_auto) kw.push_back({"exact_kernel", [&](const std::string &q) { o.str(q, "auto"); }});
    o.line(p + " : mapping " + std::to_string(kw.size()));
    for (std::size_t k = 0; k < kw.size(); k++) {
        std::string r = p + "{" + std::to_string(k) + "}";
        o.str(r + ".key", kw[k].first);
        kw[k].second(r + ".value");
    }
}

void dump_plan(Out &o, const std::string &p, const PortPlan &plan) {
    o.cls(p, "PortPlan");
    o.seq(p + ".sites", "tuple", plan.sites.size());
    for (std::size_t i = 0; i < plan.sites.size(); i++) {
        const PortSite &s = plan.sites[i];
        std::string r = at(p + ".sites", i);
        o.cls(r, "PortSite");
        o.i(r + ".wire", s.wire);
        o.f(r + ".arclength", s.arclength);
        if (s.feed >= 0) o.i(r + ".feed", s.feed); else o.none(r + ".feed");
        if (s.load >= 0) o.i(r + ".load", s.load); else o.none(r + ".load");
        if (s.has_load_spec) dump_load_spec(o, r + ".load_spec", s.load_spec); else o.none(r + ".load_spec");
        o.b(r + ".network", s.network);
    }
    o.seq(p + ".feed_ports", "tuple", plan.feed_ports.size());
    for (std::size_t i = 0; i < plan.feed_ports.size(); i++) o.i(at(p + ".feed_ports", i), plan.feed_ports[i]);
    o.seq(p + ".load_ports", "tuple", plan.load_ports.size());
    for (std::size_t i = 0; i < plan.load_ports.size(); i++) o.i(at(p + ".load_ports", i), plan.load_ports[i]);
    o.seq(p + ".node_gap_ports", "tuple", 0);
    o.seq(p + ".voltages", "tuple", plan.voltages.size());
    for (std::size_t g = 0; g < plan.voltages.size(); g++) {
        std::string r = at(p + ".voltages", g);
        if (!plan.voltages_present[g]) {
            o.none(r);
            continue;
        }
        o.seq(r, "tuple", plan.voltages[g].size());
        for (std::size_t i = 0; i < plan.voltages[g].size(); i++) o.c(at(r, i), plan.voltages[g][i]);
    }
    o.seq(p + ".network_ports", "tuple", plan.network_ports.size());
    for (std::size_t i = 0; i < plan.network_ports.size(); i++) {
        std::string r = at(p + ".network_ports", i);
        o.seq(r, "tuple", 2);
        o.i(at(r, 0), plan.network_ports[i].first);
        o.i(at(r, 1), plan.network_ports[i].second);
    }
}

void dump_point(Out &o, const std::string &p, const OperatingPoint &op) {
    dump_args(o, p + ".kwargs", op.args);
    dump_plan(o, p + ".ports", op.ports);
    o.seq(p + ".site_to_solver_port", "tuple", op.ports.sites.size());
    for (std::size_t i = 0; i < op.ports.sites.size(); i++) o.i(at(p + ".site_to_solver_port", i), static_cast<ll>(i));
    o.f(p + ".frequency_mhz", op.frequency_mhz);
    o.f(p + ".wavelength", op.wavelength);
    if (op.group >= 0) o.i(p + ".group", op.group); else o.none(p + ".group");
    o.b(p + ".extended_kernel", op.extended_kernel);
    o.seq(p + ".advisories", "tuple", op.advisories.size());
    for (std::size_t i = 0; i < op.advisories.size(); i++) o.str(at(p + ".advisories", i), op.advisories[i]);
    dump_floats(o, p + ".signs", op.signs);
}

void dump_spec_pairs(Out &o, const std::string &p, const std::vector<std::pair<ll, LoadSpec>> &v) {
    o.seq(p, "list", v.size());
    for (std::size_t i = 0; i < v.size(); i++) {
        std::string r = at(p, i);
        o.seq(r, "tuple", 2);
        o.i(at(r, 0), v[i].first);
        dump_load_spec(o, at(r, 1), v[i].second);
    }
}

void dump_tables(Out &o, const std::string &p, const BuildTables &t) {
    o.seq(p + ".ports", "list", t.ports.size());
    for (std::size_t i = 0; i < t.ports.size(); i++) {
        std::string r = at(p + ".ports", i);
        o.seq(r, "tuple", 2);
        o.i(at(r, 0), t.ports[i].first);
        o.i(at(r, 1), t.ports[i].second);
    }
    o.seq(p + ".segments", "list", t.segments.size());
    for (std::size_t i = 0; i < t.segments.size(); i++) {
        const SegmentRow &s = t.segments[i];
        std::string r = at(p + ".segments", i);
        o.seq(r, "tuple", 7);
        o.i(at(r, 0), s.number);
        o.i(at(r, 1), s.tag);
        o.i(at(r, 2), s.wire);
        o.i(at(r, 3), s.local);
        std::vector<double> c(s.centre.begin(), s.centre.end()), d(s.direction.begin(), s.direction.end());
        dump_floats(o, at(r, 4), c);
        dump_floats(o, at(r, 5), d);
        o.f(at(r, 6), s.radius);
    }
    if (t.group >= 0) o.i(p + ".group", t.group); else o.none(p + ".group");
    o.f(p + ".smallest_radius", t.smallest_radius);
    o.seq(p + ".feed_index", "list", t.feed_index.size());
    for (std::size_t i = 0; i < t.feed_index.size(); i++) o.i(at(p + ".feed_index", i), t.feed_index[i]);
    dump_spec_pairs(o, p + ".load_ports", t.load_ports);
    dump_spec_pairs(o, p + ".model_loads", t.model_loads);
    o.line(p + ".group_loads : mapping " + std::to_string(t.group_loads.size()));
    for (std::size_t i = 0; i < t.group_loads.size(); i++) {
        std::string r = p + ".group_loads{" + std::to_string(i) + "}";
        o.i(r + ".key", t.group_loads[i].first);
        o.seq(r + ".value", "tuple", t.group_loads[i].second.size());
        for (std::size_t j = 0; j < t.group_loads[i].second.size(); j++)
            o.i(at(r + ".value", j), t.group_loads[i].second[j]);
    }
    o.line(p + ".port_by_segment : mapping " + std::to_string(t.port_by_segment.size()));
    for (std::size_t i = 0; i < t.port_by_segment.size(); i++) {
        std::string r = p + ".port_by_segment{" + std::to_string(i) + "}";
        o.i(r + ".key", t.port_by_segment[i].first);
        o.i(r + ".value", t.port_by_segment[i].second);
    }
    o.b(p + ".lossy", t.lossy);
    o.b(p + ".native_loading", false);
    o.seq(p + ".current_ports", "list", t.current_ports.size());
    for (std::size_t i = 0; i < t.current_ports.size(); i++) o.i(at(p + ".current_ports", i), t.current_ports[i]);
}

}  // namespace

Build build(const std::string &text, const std::string &basis) {
    Build out;
    out.basis = basis;
    Seam s = seam(text, basis);
    out.n_structures = s.chained ? 0 : 1;
    if (s.chained) {
        std::vector<nec5::Structure> all;
        nec5::split_structures(text, all);
        out.n_structures = all.size();
    }
    for (const StructureSeam &st : s.structures) {
        StructureBuild b;
        if (st.kind != StructureSeam::served) {
            b.kind = st.kind == StructureSeam::declined ? StructureBuild::declined : StructureBuild::seam;
            b.message = st.message;
            out.structures.push_back(b);
            break;
        }
        try {
            build_structure(st, basis, b);
            b.kind = StructureBuild::built;
        } catch (const Refused &e) {
            b.kind = StructureBuild::refused;
            b.message = e.message;
        } catch (const nec5::ParseError &e) {
            b.kind = StructureBuild::raised;
            b.error_kind = e.kind;
            b.message = e.message;
        } catch (const Declined &e) {
            b.kind = StructureBuild::declined;
            b.message = e.what();
        }
        bool go_on = b.kind == StructureBuild::built;
        out.structures.push_back(b);
        if (!go_on) break;
    }
    return out;
}

std::string dump_build(const std::string &text, const std::string &basis) {
    Out o;
    o.line("build = basis " + nec5::dump_hex(basis));
    Build b;
    try {
        b = build(text, basis);
    } catch (const Declined &e) {
        o.line(std::string("build ? declined ") + nec5::dump_hex(e.what()));
        return o.s;
    } catch (const nec5::ParseError &e) {
        o.line("build = seam");
        return o.s;
    }
    o.seq("build.structures", "list", b.n_structures);
    for (std::size_t k = 0; k < b.structures.size(); k++) {
        const StructureBuild &sb = b.structures[k];
        std::string p = "build.structures[" + std::to_string(k) + "]";
        if (sb.kind == StructureBuild::seam) {
            o.line(p + " = seam");
            break;
        }
        if (sb.kind == StructureBuild::declined) {
            o.line(p + " ? declined " + nec5::dump_hex(sb.message));
            break;
        }
        o.seq(p + ".points", "list", sb.points.size());
        for (std::size_t j = 0; j < sb.points.size(); j++) dump_point(o, at(p + ".points", j), sb.points[j]);
        if (sb.kind == StructureBuild::built) {
            o.line(p + " = built");
            dump_tables(o, p + ".solver", sb.tables);
        } else if (sb.kind == StructureBuild::refused) {
            o.line(p + " = refused " + nec5::dump_hex(sb.message));
        } else {
            o.line(p + " ! " + nec5::error_kind_name(sb.error_kind) + " " + nec5::dump_hex(sb.message));
        }
    }
    return o.s;
}

}  // namespace nec4
}  // namespace mw
