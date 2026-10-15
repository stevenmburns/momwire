// The NEC-5 seam's structure half for razor-2p -- see
// `_nec5_razor_seam_core.h`. Each function names the Python it is a copy of
// (`eznec/_serve.py` unless said otherwise); its sentences are that Python's,
// byte for byte, and so is the order in which it asks its questions.
#include "_nec5_razor_seam_core.h"

// MSVC builds every TU /fp:fast (setup.py); this one replays numpy's and
// Python's elementwise arithmetic one IEEE operation at a time (the node
// grid, the tolerances, the lengths), so it asks for precise semantics.
#if defined(_MSC_VER)
#pragma float_control(precise, on)
#pragma fp_contract(off)
#endif

#include <algorithm>
#include <cmath>
#include <map>
#include <memory>
#include <set>
#include <unordered_map>

namespace mw {
namespace nec5 {

namespace {

using razor::End;

// --- the sentences (`_serve._REFUSE_*`) --------------------------------------

const char *const REFUSE_NO_EX = "this deck carries no EX card - nothing drives the structure";
const char *const REFUSE_NO_FR =
    "this deck carries no FR card - there is no frequency to solve at";
const char *const REFUSE_NO_REQUEST =
    "this deck asks for nothing: it carries neither an RP nor an XQ card, so there is no run "
    "to report";
const char *const REFUSE_RP_RANGE =
    "RP 0 with a negative range field is not served at this seam: the range form (RFLD > 0, "
    "momwire#1230) is served, and what the engine does with a negative distance has never "
    "been observed";

std::string refuse_mixed_drive_kinds(long long count, long long voltages, long long currents,
                                     const std::string &at, const std::string &why) {
    return "this deck carries " + std::to_string(count) + " EX cards writing BOTH kinds - " +
           std::to_string(voltages) + " EX 0 (a set voltage) and " + std::to_string(currents) +
           " EX 4 (a set current) - and the EX 0 at " + at + " " + why +
           "; a mixed drive is served only where every EX 0 is one of EZNEC's 1e-10 V load "
           "probes at a load's own node (momwire#1110), because a probe that small moves "
           "nothing the printout can show, and a GENUINE mixed drive - a real generator "
           "voltage beside a set current - has no printed row anywhere to be gated against";
}

std::string refuse_duplicate_ex(const std::string &at) {
    return "two EX cards address " + at +
           "; one node is one port, so the second card is a second generator in series "
           "across the same gap - no captured deck writes that, and this seam will not decide "
           "how two set currents share one path";
}

std::string refuse_two_drives_one_port(const std::string &first, const std::string &second) {
    return first + " and " + second +
           " are the two sides of ONE cut and an EX card drives each; the antenna presents a "
           "single port to both, so the two set currents are one boundary condition written "
           "twice - no captured deck writes that, and this seam will not decide which of them "
           "wins";
}

std::string refuse_buried_no_medium(const std::string &tag, double zmin, bool gd) {
    const char *card = gd ? "GD" : "GN 1";
    const char *why =
        gd ? "the MININEC-type ground solves its currents over a PERFECT image and spends its "
             "medium entirely on the far field, so below the plane there is a perfect "
             "conductor here too"
           : "the field inside a perfect conductor is identically zero, so a wire there is not "
             "buried, it is shorted out";
    return "wire " + tag + " runs below the ground plane (min z = " + format_g(zmin) +
           " m) under a " + card + " card, which has no lower medium to put it in: " + why +
           ". A buried wire is served under GN 0 / GN 2, where the half-space below the "
           "interface is a real medium with a wavenumber of its own - write the finite-ground "
           "card, raise the wire to the surface, or model in free space (GN -1)";
}

std::string refuse_in_plane_wire(const std::string &tag) {
    return "wire " + tag +
           " lies in the ground plane (both ends at z = 0) - a horizontal wire IN a conducting "
           "interface is degenerate - raise it above the plane, or stand a wire END on the "
           "plane instead (ground contact is served)";
}

// The seam's own refusals and declines, thrown out of `prepare` and caught
// by `run_one` (the Python seam's `ServeRefusal`, and a decline).
struct Refusal {
    std::string message;
};
struct Declined {
    std::string reason;
};

// `Nec5Node.written` and the "tag,node" spelling every sentence uses.
std::string written(const Node &n) { return n.node.is_zero() ? std::string("-1") : n.node.str(); }
std::string at_of(const Node &n) { return n.tag.str() + "," + written(n); }

bool same_node(const Node &a, const Node &b) { return a.tag == b.tag && a.node == b.node; }
struct NodeLess {
    bool operator()(const Node &a, const Node &b) const {
        if (a.tag != b.tag) return a.tag < b.tag;
        return a.node < b.node;
    }
};

// `_ground_spec.segment_touch_tols` of one two-point polyline (equal, bit
// for bit, to `ground_touch_tol` of it): 1e-6 of its length.
double touch_tol(const double *a, const double *b) {
    const double dx = b[0] - a[0], dy = b[1] - a[1], dz = b[2] - a[2];
    const double length = std::sqrt((dx * dx + dy * dy) + dz * dz);
    return 1e-6 * std::max(length, 1e-30);
}

// `_vec3.norm3(b - a)`.
double norm3_diff(const double *b, const double *a) {
    const double dx = b[0] - a[0], dy = b[1] - a[1], dz = b[2] - a[2];
    return std::sqrt((dx * dx + dy * dy) + dz * dz);
}

bool finite3(const double *p) {
    return std::isfinite(p[0]) && std::isfinite(p[1]) && std::isfinite(p[2]);
}

// --- the ladder (`refusal` and its rungs) ------------------------------------

// `abs(source.drive)`: CPython's complex abs (`_Py_c_abs`), which is libm's
// hypot on finite parts; an infinite part is inf, else a NaN part is NaN.
// Overflow (Python raises OverflowError there) declines.
double complex_abs(double re, double im) {
    if (std::isinf(re) || std::isinf(im)) return std::numeric_limits<double>::infinity();
    if (std::isnan(re) || std::isnan(im)) return std::numeric_limits<double>::quiet_NaN();
    const double h = std::hypot(re, im);
    if (!std::isfinite(h)) throw Declined{"drive-magnitude-overflow"};
    return h;
}

// `_load_probe_refusal`, on a deck with no TL / NT (declined before the
// ladder), so no probe sits at a network connection point.
std::string load_probe_refusal(const Deck &d) {
    std::vector<Node> loaded;
    for (const Load &l : d.loads) loaded.push_back(l.at);
    long long voltages = 0;
    for (const Source &s : d.sources)
        if (s.kind.is_zero()) voltages++;
    const long long count = static_cast<long long>(d.sources.size());
    for (const Source &s : d.sources) {
        if (!s.kind.is_zero()) continue;
        const std::string at = at_of(s.at);
        std::string why;
        const double v = complex_abs(s.drive_re, s.drive_im);
        if (v > PROBE_VOLTS) {
            why = "sets " + format_g(v) + " V, which is a generator rather than a probe";
        } else {
            bool is_loaded = false;
            for (const Node &n : loaded)
                if (same_node(n, s.at)) is_loaded = true;
            if (!is_loaded) why = "sits at a node no LD card loads, so it reads no load current";
        }
        if (!why.empty())
            return refuse_mixed_drive_kinds(count, voltages, count - voltages, at, why);
    }
    return "";
}

// `_drive_refusal`.
std::string drive_refusal(const Deck &d) {
    if (d.sources.size() == 1) return "";
    long long voltages = 0;
    for (const Source &s : d.sources)
        if (s.kind.is_zero()) voltages++;
    if (voltages != 0 && voltages != static_cast<long long>(d.sources.size())) {
        std::string mixed = load_probe_refusal(d);
        if (!mixed.empty()) return mixed;
    }
    std::set<Node, NodeLess> seen;
    for (const Source &s : d.sources) {
        if (seen.count(s.at)) return refuse_duplicate_ex(at_of(s.at));
        seen.insert(s.at);
    }
    return "";
}

// `_geometry_refusal` over GN 1 or a bare GD (free space is exempt, and a
// Sommerfeld ground declines before the ladder). It reads `_span_table`: a
// card crossing z = 0 is its two halves, and its BELOW half is a buried wire
// with no medium under either card -- the first span that refuses names the
// sentence, as the Python's argmax does. The crossing point itself is not
// needed: the above half reaches no lower than z = 0 and no nearer the plane
// than its own end, so it never refuses, and the below half's lowest point is
// the card's own end. Its tolerance is its own, never larger than the card's
// (a half is no longer than its card), so the card's says "below" for both
// unless the end sits within an ulp of the line, which declines.
std::string geometry_refusal(const Deck &d) {
    if (d.ground.kind == Ground::free_space) return "";
    const bool gd = d.ground.kind == Ground::mininec;
    for (const Wire &w : d.wires) {
        const double tol = touch_tol(w.end1, w.end2);
        const double za = w.end1[2], zb = w.end2[2];
        const bool across = (za < -tol && zb > tol) || (za > tol && zb < -tol);
        if (across) {
            const double zlow = std::min(za, zb);
            if (!(zlow < -tol * (1.0 + 1e-9))) throw Declined{"crossing-tolerance"};
            return refuse_buried_no_medium(w.tag.str(), zlow, gd);
        }
        // np.minimum: no NaN reaches here (non-finite geometry declined)
        const double zmin = std::min(za, zb);
        const bool below = zmin < -tol;
        const bool in_plane = !below && std::fabs(za) <= tol && std::fabs(zb) <= tol;
        if (below) return refuse_buried_no_medium(w.tag.str(), zmin, gd);
        if (in_plane) return refuse_in_plane_wire(w.tag.str());
    }
    return "";
}

}  // namespace

std::string scope_refusal(const Deck &d, std::string &declined) {
    declined.clear();
    try {
        // Declined before the ladder: the network cards (TL / NT, the first
        // rung's own subject) and a Sommerfeld ground (GN 0 / GN 2).
        if (!d.transmission_lines.empty() || !d.networks.empty()) throw Declined{"networks"};
        if (d.ground.kind == Ground::sommerfeld) throw Declined{"finite-ground"};
        if (d.sources.empty()) return REFUSE_NO_EX;
        std::string drive = drive_refusal(d);
        if (!drive.empty()) return drive;
        if (!d.has_frequency || d.frequency_mhz <= 0.0) return REFUSE_NO_FR;
        if (!std::isfinite(d.frequency_mhz)) throw Declined{"non-finite-frequency"};
        for (const Wire &w : d.wires)
            if (!finite3(w.end1) || !finite3(w.end2)) throw Declined{"non-finite-geometry"};
        std::string geometry = geometry_refusal(d);
        if (!geometry.empty()) return geometry;
        // `_has_buried_wire` is False here: free space has no plane, and
        // over GN 1 / GD a span below it refused just above.
        for (const Request &r : d.requests) {
            if (r.kind == Request::near_field && d.ground.kind == Ground::mininec)
                throw Declined{"near-field-over-gd"};  // `near_point_refusal`
            if (r.kind == Request::far_field && r.range_m < 0.0) return REFUSE_RP_RANGE;
        }
        if (d.requests.empty()) return REFUSE_NO_REQUEST;
    } catch (const Declined &e) {
        declined = e.reason;
    }
    return "";
}

std::string internal_error_reason(const std::string &error_kind, const std::string &message) {
    return "INTERNAL ERROR IN MOMWIRE ENGINE - " + error_kind + ": " + message;
}

double SeamPiece::length() const { return norm3_diff(points + 3, points); }

namespace {

struct KeyHash {
    size_t operator()(const NodeKey &k) const {
        size_t h = 1469598103934665603ull;
        for (long long v : k) h = (h ^ static_cast<size_t>(v)) * 1099511628211ull;
        return h;
    }
};

// `_node_key`: each coordinate over the grid, rounded half to even.
// `_wire_nodes` takes numpy's int64 path when every |scaled| < 2**62; a
// larger one would be Python's exact int, which declines here.
NodeKey node_key(const double *p) {
    NodeKey k;
    for (int c = 0; c < 3; c++) {
        const double s = std::nearbyint(p[c] / NODE_EPS);
        if (!(std::fabs(s) < 4611686018427387904.0)) throw Declined{"node-key-range"};
        k[c] = static_cast<long long>(s);
    }
    return k;
}

// `_node_grid`'s node k of a wire: `a + (b - a) * (k / n)`, ends authored.
void grid_point(const Wire &w, long long n, long long k, double *out) {
    if (k == 0) {
        std::copy(w.end1, w.end1 + 3, out);
        return;
    }
    if (k == n) {
        std::copy(w.end2, w.end2 + 3, out);
        return;
    }
    const double t = static_cast<double>(k) / static_cast<double>(n);
    for (int c = 0; c < 3; c++) out[c] = w.end1[c] + (w.end2[c] - w.end1[c]) * t;
}

// `math.dist`, to within an ulp or two of CPython's (which is correctly
// rounded or nearly so); `_phantom_tags` declines when a comparison is
// closer than that to call.
double dist(const double *p, const double *q) { return norm3_diff(p, q); }
bool too_close(double a, double b) {
    return std::fabs(a - b) <= 1e-12 * std::max(std::fabs(a), std::fabs(b));
}

// `_phantom_tags`: is any wire EZNEC's parked circuit node? A phantom wire
// is not ported (its sites reach no momwire port), so finding one declines.
void refuse_phantom(const Deck &d, double wavelength) {
    if (d.wires.size() < 2) return;
    const double extent_limit = 0.05 * wavelength;
    for (size_t i = 0; i < d.wires.size(); i++) {
        const Wire &w = d.wires[i];
        const double extent = dist(w.end1, w.end2);
        if (too_close(extent, extent_limit)) throw Declined{"phantom-threshold"};
        if (!(extent < extent_limit)) continue;
        const double floor = std::max(10.0 * wavelength, 100.0 * extent);
        bool near = false, unsure = false;
        for (size_t j = 0; j < d.wires.size() && !near; j++) {
            if (j == i) continue;
            const double *here[2] = {w.end1, w.end2};
            const double *there[2] = {d.wires[j].end1, d.wires[j].end2};
            for (const double *h : here)
                for (const double *t : there) {
                    const double r = dist(h, t);
                    if (too_close(r, floor))
                        unsure = true;
                    else if (!(r > floor))
                        near = true;
                }
        }
        if (near) continue;
        throw Declined{unsure ? "phantom-threshold" : "phantom-wire"};
    }
}

// The pieces' end keys -> the (piece, end) members meeting there, as
// `build_mesh` groups its junctions.
void junctions_of(SeamMesh &mesh) {
    std::map<NodeKey, std::vector<JunctionEnd>> ends_at;
    for (size_t i = 0; i < mesh.pieces.size(); i++) {
        ends_at[node_key(mesh.pieces[i].points)].emplace_back(i, End::start);
        ends_at[node_key(mesh.pieces[i].points + 3)].emplace_back(i, End::end);
    }
    for (auto &kv : ends_at) {
        if (kv.second.size() < 2) continue;
        std::vector<JunctionEnd> members = kv.second;
        // `_canonical_end`: wire, then start before end
        std::sort(members.begin(), members.end());
        mesh.junctions.push_back(members);
    }
}

// `_assign_columns`.
void assign_columns(SeamMesh &mesh) {
    std::map<JunctionEnd, long long> junction_of;
    for (size_t j = 0; j < mesh.junctions.size(); j++)
        for (const JunctionEnd &m : mesh.junctions[j]) junction_of[m] = static_cast<long long>(j);
    std::map<long long, long long> declared;  // junction -> declaring site
    for (SeamSite &site : mesh.sites) {
        long long junction = -1;
        if (!site.contact && site.end != SeamSite::interior) {
            auto it = junction_of.find(
                JunctionEnd(site.piece, site.end == SeamSite::at_start ? End::start : End::end));
            if (it != junction_of.end()) junction = it->second;
        }
        const bool shared = junction >= 0 && mesh.junctions[junction].size() == 2;
        long long first = -1;
        if (shared) {
            auto it = declared.find(junction);
            if (it != declared.end()) first = it->second;
        }
        if (site.spelling == SeamSite::node) {
            site.weight = first < 0 ? site.sign : -site.sign;
        } else if (junction < 0) {
            site.weight = site.sign;
        } else {
            const JunctionEnd self(site.piece,
                                   site.end == SeamSite::at_start ? End::start : End::end);
            const bool side_a = mesh.junctions[junction][0] == self;
            site.weight = side_a ? -site.sign : site.sign;
        }
        if (first >= 0) {
            site.column = mesh.sites[first].column;
            continue;
        }
        if (shared) declared[junction] = site.index;
        std::vector<long long> &ports = site.spelling == SeamSite::gap ? mesh.feeds : mesh.gaps;
        site.column = static_cast<long long>(ports.size());
        ports.push_back(site.index);
    }
    for (SeamSite &site : mesh.sites)
        if (site.spelling == SeamSite::node) site.column += static_cast<long long>(mesh.feeds.size());
}

// `_prepare` from `structure_of` to the constructor call (the ladder has
// already passed). Throws Refusal / Declined.
Prepared prepare(const Deck &d) {
    // `_prepare`'s wire-loading question is razor's to serve; the C++ solve
    // does not carry WIRE loading (LD 5 / LD 2 / LD 3 reach the solver as
    // keyword arguments), so those decline here. A LUMPED load (LD 0 / 1 / 4)
    // never reaches the solver: it is a port at its node, stamped onto the
    // deck's port algebra after the solve (the run half), so it is meshed as
    // a site like a source. Its impedance is the run half's; here only that
    // `LoadSpec.impedance` cannot raise on it (a non-finite value, or an
    // element whose `omega * L` / `omega * C` underflows to a zero divisor).
    if (!d.wire_conductivity.empty() || !d.wire_distributed_rlc.empty())
        throw Declined{"wire-loading"};
    {
        const double omega = 2.0 * 3.141592653589793 * (d.frequency_mhz * 1e6);
        for (const Load &l : d.loads) {
            const LoadSpec &sp = l.spec;
            if (!std::isfinite(sp.r) || !std::isfinite(sp.l) || !std::isfinite(sp.c) ||
                !std::isfinite(sp.x))
                throw Declined{"load-values"};
            if ((sp.l != 0.0 && !(std::fabs(omega * sp.l) > 1e-290)) ||
                (sp.c != 0.0 && !(std::fabs(omega * sp.c) > 1e-290)))
                throw Declined{"load-values"};
        }
    }
    Prepared p;
    const double frequency = d.frequency_mhz;  // `float(deck.frequency_mhz or 0.0)`
    p.frequency_mhz = frequency;
    p.wavelength = SPEED_OF_LIGHT_MHZ_M / frequency;
    p.omega = 2.0 * 3.141592653589793 * frequency * 1e6;

    // --- `structure_of` ----------------------------------------------------
    long long total = 0;
    std::vector<long long> nseg;
    for (const Wire &w : d.wires) {
        if (!w.segment_count.fits_i64()) throw Declined{"segment-count"};
        const long long n = w.segment_count.i64();
        if (n < 1 || n > MAX_SEGMENTS) throw Declined{"segment-count"};
        total += n;
        if (total > MAX_SEGMENTS) throw Declined{"segment-count"};
        nseg.push_back(n);
    }
    for (const Wire &w : d.wires)
        if (!std::isfinite(w.radius)) throw Declined{"non-finite-radius"};
    const bool ground_plane = !d.ge_flag.is_zero();
    std::unordered_map<NodeKey, long long, KeyHash> degree;
    double pt[3];
    for (size_t i = 0; i < d.wires.size(); i++) {
        const long long n = nseg[i];
        for (long long k = 0; k <= n; k++) {
            grid_point(d.wires[i], n, k, pt);
            degree[node_key(pt)] += (k == 0 || k == n) ? 1 : 2;
        }
    }
    if (ground_plane)
        for (const Wire &w : d.wires)
            for (const double *e : {w.end1, w.end2})
                if (std::fabs(e[2]) <= NODE_EPS) degree[node_key(e)] += 1;
    p.structure.ground_plane = ground_plane;
    p.structure.node_count = static_cast<long long>(degree.size());
    p.structure.wire_element_count = total;
    for (const auto &kv : degree) p.structure.unknown_count += std::max(kv.second - 1, 0LL);

    // --- `build_mesh` (razor: the delta-gap spelling, nothing cut) ---------
    refuse_phantom(d, p.wavelength);
    SeamMesh &mesh = p.mesh;
    for (size_t i = 0; i < d.wires.size(); i++) {
        const Wire &w = d.wires[i];
        SeamPiece piece;
        piece.tag = w.tag;
        piece.first_node = 0;
        piece.last_node = nseg[i];
        std::copy(w.end1, w.end1 + 3, piece.points);
        std::copy(w.end2, w.end2 + 3, piece.points + 3);
        piece.radius = w.radius;
        mesh.pieces.push_back(piece);
        for (long long k = 0; k < nseg[i]; k++)
            mesh.element_of.emplace_back(static_cast<long long>(i), k);
    }
    junctions_of(mesh);

    // `_addressed_nodes`: the sources' and the lumped loads' (no TL / NT
    // here), per tag.
    std::map<Int, std::set<long long>> addressed;
    for (const Source &s : d.sources) {
        if (!s.at.node.fits_i64()) throw Declined{"node-range"};
        addressed[s.at.tag].insert(s.at.node.i64());
    }
    for (const Load &l : d.loads) {
        if (!l.at.node.fits_i64()) throw Declined{"node-range"};
        addressed[l.at.tag].insert(l.at.node.i64());
    }
    for (const auto &kv : addressed) {
        const Int &tag = kv.first;
        long long w = -1;  // `Structure.index_of`
        for (size_t i = 0; i < d.wires.size(); i++)
            if (d.wires[i].tag == tag) {
                w = static_cast<long long>(i);
                break;
            }
        for (long long node : kv.second) {
            const Node at{tag, Int(node)};
            SeamSite site;
            site.at = at;
            const long long last = w >= 0 ? nseg[w] : 0;
            if (w >= 0 && 0 < node && node < last) {
                // `_interior_site`
                grid_point(d.wires[w], last, node, pt);
                const long long deg = degree[node_key(pt)];
                if (deg != 2)
                    throw Refusal{at_of(at) + " addresses a node INSIDE wire " + tag.str() +
                                  " where " + std::to_string(deg - 2) +
                                  " more element ends meet; this basis spells a series EMF as "
                                  "a delta gap, which drives one wire's own knot and would "
                                  "leave the others unconnected"};
                site.spelling = SeamSite::gap;
                site.piece = w;
                site.end = SeamSite::interior;
                site.sign = 1.0;
                site.arclength = norm3_diff(pt, mesh.pieces[w].points);
            } else {
                if (w < 0 || !(node == 0 || node == last))
                    throw Refusal{"no wire piece carries node " + std::to_string(node) +
                                  " of tag " + tag.str()};
                const SeamPiece &piece = mesh.pieces[w];
                const bool start = node == 0;
                const double *point = start ? piece.points : piece.points + 3;
                const bool grounded = ground_plane && std::fabs(point[2]) <= NODE_EPS;
                const long long meeting = degree[node_key(point)] - (grounded ? 1 : 0);
                site.piece = w;
                site.end = start ? SeamSite::at_start : SeamSite::at_end;
                site.arclength = start ? 0.0 : piece.length();
                if (meeting >= 2) {
                    if (grounded)
                        throw Refusal{at_of(at) +
                                      " addresses a node where several wires meet IN the "
                                      "ground plane; a series source at a grounded junction is "
                                      "not served at this seam"};
                    site.spelling = meeting >= 3 ? SeamSite::node : SeamSite::gap;
                    site.sign = start ? 1.0 : -1.0;
                } else if (grounded) {
                    site.spelling = SeamSite::gap;
                    site.sign = 1.0;
                    site.contact = true;
                } else {
                    throw Refusal{at_of(at) + " addresses a FREE end of wire " + tag.str() +
                                  " - nothing carries current past a lone conductor end, so "
                                  "there is no path for a source, a load or a network "
                                  "connection to sit in"};
                }
            }
            site.index = static_cast<long long>(mesh.sites.size());
            mesh.sites.push_back(site);
        }
    }
    assign_columns(mesh);

    // --- `_check_one_port_per_drive` ---------------------------------------
    {
        std::map<long long, Node> seen;
        for (const Source &s : d.sources) {
            const SeamSite *site = nullptr;
            for (const SeamSite &x : mesh.sites)
                if (same_node(x.at, s.at)) site = &x;
            if (site == nullptr || site->column < 0) continue;
            auto it = seen.find(site->column);
            if (it != seen.end())
                throw Refusal{refuse_two_drives_one_port(at_of(it->second), at_of(s.at))};
            seen[site->column] = s.at;
        }
    }
    for (const Source &s : d.sources) {
        bool dup = false;
        for (const Node &n : p.source_sites)
            if (same_node(n, s.at)) dup = true;
        if (!dup) p.source_sites.push_back(s.at);
    }

    // --- `_medium`, `_ground_kwargs` ----------------------------------------
    const bool has_ground = d.ground.kind != Ground::free_space;
    if (d.ground.kind == Ground::mininec) {
        p.has_medium = true;
        p.medium_eps_r = d.ground.eps_r;
        if (d.ground.sigma_sets_im_epsc) {
            p.medium_eps_c[0] = d.ground.eps_r;
            p.medium_eps_c[1] = d.ground.sigma;
            p.medium_sigma = -d.ground.sigma / (p.wavelength * EPSC_CONDUCTIVITY_FACTOR);
        } else {
            p.medium_sigma = d.ground.sigma;
            p.medium_eps_c[0] = d.ground.eps_r;
            p.medium_eps_c[1] = -p.medium_sigma * p.wavelength * EPSC_CONDUCTIVITY_FACTOR;
        }
    }

    // --- `_check_basis_can_host` (razor: node gaps and knot feeds served,
    // no `ground_eps`): the inert one-segment wire ------------------------
    {
        std::set<JunctionEnd> joined;
        for (const auto &g : mesh.junctions)
            for (const JunctionEnd &e : g) joined.insert(e);
        std::vector<long long> inert;
        for (size_t i = 0; i < mesh.pieces.size(); i++) {
            const SeamPiece &piece = mesh.pieces[i];
            if (piece.n_elements() >= 2) continue;
            if (joined.count(JunctionEnd(i, End::start)) || joined.count(JunctionEnd(i, End::end)))
                continue;
            if (has_ground) {
                const double tol = touch_tol(piece.points, piece.points + 3);
                if (std::fabs(piece.points[2] - 0.0) <= tol || std::fabs(piece.points[5] - 0.0) <= tol)
                    continue;
            }
            inert.push_back(static_cast<long long>(i));
        }
        if (!inert.empty())
            throw Refusal{"wire " + mesh.pieces[inert[0]].tag.str() +
                          " is one segment long and meets no other wire at either end, so "
                          "basis razor-2p has no basis function to put on it and it would "
                          "scatter nothing (momwire#608); this deck declares " +
                          std::to_string(inert.size()) + " such GW card" +
                          (inert.size() > 1 ? "s" : "")};
    }

    // --- `construct_solver` -------------------------------------------------
    // The extended kernel by default: razor declares no refusal for it, a
    // stepped radius at a junction is a condition it serves, and a wire
    // below the plane (`_wires_below`) refused on the ladder already.
    if (has_ground)
        for (const SeamPiece &piece : mesh.pieces)
            if (std::min(piece.points[2], piece.points[5]) <
                0.0 - touch_tol(piece.points, piece.points + 3))
                throw Declined{"wire-below-plane"};
    razor::FrontInputs &in = p.inputs;
    for (const SeamPiece &piece : mesh.pieces) {
        in.points.insert(in.points.end(), piece.points, piece.points + 6);
        in.wire_points.push_back(2);
        razor::EdgeCounts ec;
        ec.kind = razor::EdgeCounts::list;
        ec.v.push_back(piece.n_elements());
        in.counts.push_back(ec);
    }
    in.counts_given = true;
    // `wire_radius=radii[0] if len(set(radii)) == 1 else radii`
    bool uniform = true;
    for (const SeamPiece &piece : mesh.pieces)
        if (!(piece.radius == mesh.pieces[0].radius)) uniform = false;
    if (mesh.pieces.size() == 1) uniform = true;
    if (mesh.pieces.empty()) throw Declined{"no-wires"};
    if (uniform) {
        in.radius.assign(1, mesh.pieces[0].radius);
        in.radius_shape.clear();
    } else {
        for (const SeamPiece &piece : mesh.pieces) in.radius.push_back(piece.radius);
        in.radius_shape.assign(1, static_cast<int64_t>(mesh.pieces.size()));
    }
    in.feeds_given = true;
    for (long long s : mesh.feeds) {
        razor::FeedSpec f;
        f.wire = mesh.sites[s].piece;
        f.midpoint = false;
        f.arc = mesh.sites[s].arclength;
        f.volts = cd(0.0, 0.0);
        in.feeds.push_back(f);
    }
    for (long long s : mesh.gaps) {
        razor::NodeGapSpec g;
        g.wire = mesh.sites[s].piece;
        g.end = mesh.sites[s].end == SeamSite::at_start ? End::start : End::end;
        g.volts = cd(0.0, 0.0);
        in.node_gaps.push_back(g);
    }
    in.has_ground = has_ground;
    in.ground_z = 0.0;
    in.wavelength = p.wavelength;
    in.extended_kernel = true;
    in.exact_kernel = razor::ExactKernel::automatic;
    return p;
}

// One run through `_prepare` (or `serve_runs`' reuse): its outcome.
RunOutcome run_one(const Deck &d, SeamResult &r, long long *shared_with) {
    RunOutcome o;
    std::string declined;
    std::string why = scope_refusal(d, declined);
    if (!declined.empty()) {
        o.kind = RunOutcome::declined;
        o.message = declined;
        return o;
    }
    if (!why.empty()) {
        o.kind = RunOutcome::refused;
        o.message = why;
        return o;
    }
    if (shared_with != nullptr && *shared_with >= 0) {
        const Prepared &prev = r.prepared[static_cast<size_t>(*shared_with)];
        // `frozenset(source.at for source in deck.sources)` against the
        // prepared run's
        std::set<Node, NodeLess> a(prev.source_sites.begin(), prev.source_sites.end()), b;
        for (const Source &s : d.sources) b.insert(s.at);
        bool same = a.size() == b.size();
        if (same)
            for (auto ia = a.begin(), ib = b.begin(); ia != a.end(); ++ia, ++ib)
                if (!same_node(*ia, *ib)) same = false;
        if (same) {
            o.kind = RunOutcome::shared;
            o.prepared_index = *shared_with;
            return o;
        }
    }
    try {
        Prepared p = prepare(d);
        // `RazorSolver(...)` and `compute_port_solution`'s front half: a
        // NotImplementedError from the CONSTRUCTOR is `_prepare`'s refusal;
        // anything else escapes the seam.
        p.front = std::make_shared<razor::Front>();
        std::string why;
        try {
            why = p.front->build(p.inputs);
        } catch (const razor::FrontError &e) {
            const bool ni = e.kind == razor::FrontErrorKind::not_implemented;
            o.kind = ni && !e.solve_phase ? RunOutcome::refused : RunOutcome::raised;
            o.error_kind = ni ? "NotImplementedError" : "ValueError";
            o.message = e.what();
            return o;
        }
        if (!why.empty()) throw Declined{"front:" + why};
        o.kind = RunOutcome::prepared;
        o.prepared_index = static_cast<long long>(r.prepared.size());
        r.prepared.push_back(std::move(p));
        if (shared_with != nullptr) *shared_with = o.prepared_index;
    } catch (const Refusal &e) {
        o.kind = RunOutcome::refused;
        o.message = e.message;
    } catch (const Declined &e) {
        o.kind = RunOutcome::declined;
        o.message = e.reason;
    }
    return o;
}

}  // namespace

SeamResult razor_seam(const std::string &text) {
    SeamResult r;
    StreamRead s = read_stream(text);
    r.chained = s.chained;
    r.route = SeamResult::served;
    auto stop = [&r](const RunOutcome &o) {
        if (o.kind == RunOutcome::refused) {
            r.route = SeamResult::refused;
        } else if (o.kind == RunOutcome::raised) {
            r.route = SeamResult::raised;
            r.error_kind = o.error_kind;
        } else {
            r.route = SeamResult::declined;
        }
        r.message = o.message;
    };
    for (const Answer &a : s.answers) {
        StructureOutcome so;
        so.read = a.kind;
        bool stopped = false;
        if (a.kind == Answer::refused || a.kind == Answer::raised) {
            RunOutcome o;
            o.kind = a.kind == Answer::refused ? RunOutcome::refused : RunOutcome::raised;
            o.message = a.message;
            o.error_kind = error_kind_name(a.error_kind);
            so.runs.push_back(o);
            stop(o);
            stopped = true;
        } else if (a.kind == Answer::single) {
            RunOutcome o = run_one(a.deck, r, nullptr);
            so.runs.push_back(o);
            if (o.kind != RunOutcome::prepared) {
                stop(o);
                stopped = true;
            }
        } else {
            long long shared = -1;
            for (const Deck &run : a.runs) {
                RunOutcome o = run_one(run, r, &shared);
                so.runs.push_back(o);
                if (o.kind != RunOutcome::prepared && o.kind != RunOutcome::shared) {
                    stop(o);
                    stopped = true;
                    break;
                }
            }
        }
        r.structures.push_back(std::move(so));
        if (stopped) break;
    }
    r.stream = std::move(s);
    return r;
}

}  // namespace nec5
}  // namespace mw
