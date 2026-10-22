// The NEC-4.2 slot's run half in C++ (see `_nec4_run_core.h`). Each function
// names the Python it copies; numpy's arithmetic is written out per element
// in numpy's order (`nec5::replay`'s fused array product and reciprocal-
// scaled quotient), numpy's scalar complex product unfused, CPython's
// complex arithmetic as `_Py_c_prod` / `_Py_c_quot`.

#if defined(_MSC_VER) && !defined(__clang__)
#pragma float_control(precise, on)
#pragma fp_contract(off)
#endif

#include "_nec4_run_core.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <limits>
#include <array>
#include <map>
#include <string>
#include <utility>
#include <vector>

namespace mw {
namespace nec4 {

namespace {

typedef long long ll;
typedef std::complex<double> cd;
namespace rp = nec5::replay;

const double kPi = 3.141592653589793;
const double kCLight = 299792458.0;        // momwire._constants.C_LIGHT
const double kCMhzM = 299.8;               // _nec4._C_MHZ_M
const double kChargeResidualFloor = 1e-8;  // portal._CHARGE_RESIDUAL_FLOOR
const double kSmin = 1e-3;                 // deck._nec2_geometry._SMIN

[[noreturn]] void decline(const std::string &m) { throw Declined(m); }

double seconds_since(std::chrono::steady_clock::time_point t0) {
    return std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
}

// numpy's array arithmetic
cd cm(cd a, cd b) { return rp::np_cmul(a, b); }
cd cdiv(cd a, cd b) { return rp::np_cdiv(a, b); }
cd cadd(cd a, cd b) { return cd(a.real() + b.real(), a.imag() + b.imag()); }
// numpy's SCALAR complex product (`cdouble_ctype_multiply`): never fused
cd scmul(cd a, cd b) {
    return cd(a.real() * b.real() - a.imag() * b.imag(), a.real() * b.imag() + a.imag() * b.real());
}
// CPython 3.12's complex arithmetic
cd py_mul(cd a, cd b) {
    return cd(a.real() * b.real() - a.imag() * b.imag(), a.real() * b.imag() + a.imag() * b.real());
}
cd py_div(cd a, cd b) {
    const double abs_breal = b.real() < 0 ? -b.real() : b.real();
    const double abs_bimag = b.imag() < 0 ? -b.imag() : b.imag();
    if (abs_breal >= abs_bimag) {
        if (abs_breal == 0.0) decline("a complex division by zero (the reference raises)");
        const double ratio = b.imag() / b.real();
        const double denom = b.real() + b.imag() * ratio;
        return cd((a.real() + a.imag() * ratio) / denom, (a.imag() - a.real() * ratio) / denom);
    }
    if (abs_bimag >= abs_breal) {
        const double ratio = b.real() / b.imag();
        const double denom = b.real() * ratio + b.imag();
        return cd((a.real() * ratio + a.imag()) / denom, (a.imag() * ratio - a.real()) / denom);
    }
    decline("a complex division by NaN");
}
cd py_add(cd a, cd b) { return cd(a.real() + b.real(), a.imag() + b.imag()); }

// CPython's `math.atan2` (`m_atan2`: its special cases, then libm) and
// `math.degrees` (`x * (180 / pi)`), as the NEC-5 printout core spells them
double py_atan2(double y, double x) {
    const double pi = 3.141592653589793;
    if (std::isnan(x) || std::isnan(y)) return std::numeric_limits<double>::quiet_NaN();
    if (std::isinf(y)) {
        if (std::isinf(x)) {
            if (std::copysign(1., x) == 1.) return std::copysign(0.25 * pi, y);
            return std::copysign(0.75 * pi, y);
        }
        return std::copysign(0.5 * pi, y);
    }
    if (std::isinf(x) || y == 0.) {
        if (std::copysign(1., x) == 1.) return std::copysign(0., y);
        return std::copysign(pi, y);
    }
    return std::atan2(y, x);
}
double py_degrees(double x) { return (180.0 / 3.141592653589793) * x; }

// `numpy.linalg.solve(a, b)` for a square C-ordered `a` and a vector `b`:
// both copied to Fortran order and one zgesv (numpy's ILP64 LAPACK); with a
// host table, `_lapack.solve_stacked`'s getrf + getrs on the host's library.
// A singular block is numpy's LinAlgError: declined.
std::vector<cd> np_solve(const nec5::NumpyLinalg &np, const nec5::CMat &a, const std::vector<cd> &b) {
    const nec5::np_int n = a.rows, nrhs = 1;
    std::vector<cd> af(static_cast<std::size_t>(n * n));
    for (nec5::np_int j = 0; j < n; j++)
        for (nec5::np_int i = 0; i < n; i++) af[static_cast<std::size_t>(i + j * n)] = a.at(i, j);
    std::vector<cd> x(b);
    if (np.host.zgemm != nullptr && n > 0) {
        if (n > std::numeric_limits<int>::max()) decline("a solve past the LP64 host's int range");
        int ni = static_cast<int>(n), one = 1, info = 0;
        std::vector<int> ipiv(static_cast<std::size_t>(n));
        np.host.getrf(&ni, &ni, af.data(), &ni, ipiv.data(), &info);
        if (info != 0) decline("a singular drive block (the host route raises)");
        char trans = 'N';
        np.host.getrs(&trans, &ni, &one, af.data(), &ni, ipiv.data(), x.data(), &ni, &info);
        if (info != 0) decline("the host's getrs refused an argument");
        return x;
    }
    if (np.zgesv == nullptr) decline("no LAPACK for the drive block");
    std::vector<nec5::np_int> ipiv(static_cast<std::size_t>(n > 0 ? n : 1));
    const nec5::np_int lda = std::max<nec5::np_int>(n, 1), ldb = lda;
    nec5::np_int info = 0;
    np.zgesv(&n, &nrhs, af.data(), &lda, ipiv.data(), x.data(), &ldb, &info);
    if (info != 0) decline("a singular drive block (the reference raises)");
    return x;
}

// `_vec3.norm3` of one 3-vector (math.sqrt over Python floats)
double norm3(const double *v) { return std::sqrt((v[0] * v[0] + v[1] * v[1]) + v[2] * v[2]); }
double norm3(const Point3 &v) { return norm3(v.data()); }

// `value_row`
ValueRow value_row(ll number, ll tag, const double centre[3], double length, cd value, bool end = false) {
    ValueRow r;
    r.number = number;
    r.tag = tag;
    for (int c = 0; c < 3; c++) r.centre[c] = centre[c];
    r.length = length;
    r.real = value.real();
    r.imag = value.imag();
    if (!std::isfinite(value.real()) || !std::isfinite(value.imag())) decline("a non-finite table value");
    r.magnitude = std::hypot(value.real(), value.imag());
    r.phase_deg = r.magnitude != 0.0 ? py_degrees(py_atan2(value.imag(), value.real())) : 0.0;
    r.end = end;
    return r;
}

ll i64(const nec5::Int &v) {
    if (!v.fits_i64()) decline("an integer past 64 bits");
    return v.i64();
}

// --- the structure's tables --------------------------------------------------

struct Tables {
    const PortalDeck *deck = nullptr;
    const BuildTables *bt = nullptr;
    std::vector<ll> wire_first;  // per structure wire, its first segment (0-based)
    std::vector<std::pair<ll, ll>> connections;
    std::vector<std::vector<ll>> junctions;
};

// `_segment_end_nodes(wires)`: the nodes in first-seen order, each its
// signed segment ends (-n at a segment's start, +n at its end).
struct EndNodes {
    std::vector<std::vector<ll>> members;  // in `order`
    std::map<std::vector<double>, std::size_t> index;
};

std::vector<double> node_key(const double p[3]) {
    std::vector<double> k(3);
    for (int c = 0; c < 3; c++) {
        const double q = p[c] / 1e-9;
        if (!std::isfinite(q)) decline("a non-finite node coordinate");
        k[static_cast<std::size_t>(c)] = std::nearbyint(q);  // round(): half to even
        if (k[static_cast<std::size_t>(c)] == 0.0) k[static_cast<std::size_t>(c)] = 0.0;
    }
    return k;
}

// a segment's end points as numpy writes them: p1 + k * step
void segment_point(const GeometryWire &w, ll k, double out[3]) {
    for (int c = 0; c < 3; c++) {
        const double step = (w.p2[c] - w.p1[c]) / static_cast<double>(w.n_seg);
        out[c] = w.p1[c] + static_cast<double>(k) * step;
    }
}

EndNodes end_nodes(const std::vector<GeometryWire> &wires) {
    EndNodes e;
    ll n = 0;
    for (const GeometryWire &w : wires)
        for (ll k = 0; k < w.n_seg; k++) {
            n++;
            double a[3], b[3];
            segment_point(w, k, a);
            segment_point(w, k + 1, b);
            const std::pair<const double *, ll> pts[2] = {{a, -n}, {b, n}};
            for (const auto &pt : pts) {
                const std::vector<double> key = node_key(pt.first);
                auto it = e.index.find(key);
                if (it == e.index.end()) {
                    it = e.index.emplace(key, e.members.size()).first;
                    e.members.emplace_back();
                }
                e.members[it->second].push_back(pt.second);
            }
        }
    return e;
}

std::vector<ll> by_abs(std::vector<ll> v) {
    std::stable_sort(v.begin(), v.end(), [](ll x, ll y) { return std::llabs(x) < std::llabs(y); });
    return v;
}

Tables tables_of(const StructureSeam &ss, const StructureBuild &sb) {
    Tables t;
    t.deck = &ss.deck;
    t.bt = &sb.tables;
    const std::vector<GeometryWire> &wires = ss.deck.structure.wires;
    ll first = 0;
    for (const GeometryWire &w : wires) {
        t.wire_first.push_back(first);
        first += w.n_seg;
    }
    const EndNodes e = end_nodes(wires);
    // `_connection_data(wires, interpolates)`
    const bool interpolates = ss.deck.ground_plane_interpolates;
    ll idx = 0;
    for (const GeometryWire &w : wires) {
        double step[3];
        for (int c = 0; c < 3; c++) step[c] = (w.p2[c] - w.p1[c]) / static_cast<double>(w.n_seg);
        const double slen = norm3(step) * kSmin;
        for (ll k = 0; k < w.n_seg; k++) {
            idx++;
            double a[3], b[3];
            segment_point(w, k, a);
            segment_point(w, k + 1, b);
            const std::vector<ll> &here = e.members[e.index.at(node_key(a))];
            const std::vector<ll> &there = e.members[e.index.at(node_key(b))];
            ll i_minus = 0, i_plus = 0;
            for (ll m : here)
                if (std::llabs(m) != idx) {
                    i_minus = m;
                    break;
                }
            for (ll m : there)
                if (std::llabs(m) != idx) {
                    i_plus = -m;
                    break;
                }
            if (interpolates) {
                if (std::fabs(a[2]) <= slen) i_minus = idx;
                if (std::fabs(b[2]) <= slen) i_plus = idx;
            }
            t.connections.push_back({i_minus, i_plus});
        }
    }
    // `_connections`: NEC-4.2's rule at a junction
    for (const std::vector<ll> &members : e.members) {
        if (members.size() < 3) continue;
        const std::vector<ll> ring = by_abs(members);
        for (std::size_t k = 0; k < ring.size(); k++) {
            const ll member = ring[k];
            const ll following = ring[(k + 1) % ring.size()];
            const std::size_t number = static_cast<std::size_t>(std::llabs(member));
            auto &cx = t.connections[number - 1];
            if (member < 0)
                cx.first = following;
            else
                cx.second = -following;
        }
    }
    for (const std::vector<ll> &members : e.members)
        if (members.size() >= 3) t.junctions.push_back(by_abs(members));
    return t;
}

// `Nec2Structure.locate(tag, seg)` then `DeckSolver.global_segment`
ll global_segment(const std::vector<GeometryWire> &wires, const nec5::Int &tag, const nec5::Int &seg_in) {
    const nec5::Int one(1);
    const nec5::Int seg = seg_in < one ? one : seg_in;
    nec5::Int acc(0);
    ll before = 0;
    for (const GeometryWire &w : wires) {
        if (!tag.is_zero() && w.tag != tag) {
            before += w.n_seg;
            continue;
        }
        if (acc + nec5::Int(w.n_seg) >= seg) return before + i64(seg - acc);
        acc = acc + nec5::Int(w.n_seg);
        before += w.n_seg;
    }
    decline("a source address out of range (the reference raises)");
}

// `_wire_rows(deck)`
std::vector<WireTableRow> wire_rows(const PortalDeck &deck) {
    std::vector<WireTableRow> rows;
    ll first = 1, number = 0;
    for (const nec5::Card &card : deck.geometry) {
        WireTableRow r;
        if (card.mnemonic == "GS") {
            r.kind = WireTableRow::scale;
            r.factor = card.f(2);
            rows.push_back(r);
            continue;
        }
        if (card.mnemonic == "GM") {
            r.kind = WireTableRow::move;
            r.tag_increment = i64(card.i(0));
            r.copies = i64(card.i(1));
            for (int c = 0; c < 3; c++) {
                r.rotation_deg[c] = card.f(2 + c);
                r.translation[c] = card.f(5 + c);
            }
            const double f8 = card.f(8) + 0.5;
            if (!std::isfinite(f8) || std::fabs(f8) >= 9.2e18) decline("a GM first-tag field past 64 bits");
            r.first_tag = static_cast<ll>(std::trunc(f8));
            r.move_segments = first - 1;
            rows.push_back(r);
            continue;
        }
        if (card.mnemonic == "GH") decline("a GH helix row (momwire#1607 B4c)");
        if (card.mnemonic != "GW" && card.mnemonic != "GA") continue;
        number++;
        const ll n_seg = i64(card.i(1));
        const ll last = first + n_seg - 1;
        r.number = number;
        r.segments = n_seg;
        r.first = first;
        r.last = last;
        r.tag = i64(card.i(0));
        if (card.mnemonic == "GA") {
            r.kind = WireTableRow::arc;
            r.arc_radius = card.f(2);
            r.angle1_deg = card.f(3);
            r.angle2_deg = card.f(4);
            r.radius = card.f(5);
        } else {
            r.kind = WireTableRow::wire;
            for (int c = 0; c < 3; c++) {
                r.end1[c] = card.f(2 + c);
                r.end2[c] = card.f(5 + c);
            }
            r.radius = card.f(8);
        }
        rows.push_back(r);
        first += n_seg;
    }
    return rows;
}

// `_load_row(card)`
PrintLoadRow load_row(const nec5::Card &card) {
    PrintLoadRow r;
    const ll kind = i64(card.i(0));
    r.tag = i64(card.i(1));
    r.node_from = i64(card.i(2));
    ll last = i64(card.i(3));
    if (last == 0) last = r.node_from;
    r.node_thru = last;
    auto truthy = [](double x) { return x != 0.0; };  // `x or None` (NaN is truthy)
    if (kind == 4) {
        r.has_resistance = true;
        r.resistance = card.f(4);
        if (card.f(5) != 0.0) {
            r.has_reactance = true;
            r.reactance = card.f(5);
        }
        return r;
    }
    if (kind == 5) {
        r.has_conductivity = true;
        r.conductivity = card.f(4);
        r.kind = "WIRE";
        return r;
    }
    static const char *names[] = {"SERIES", "PARALLEL", "SERIES (PER METER)", "PARALLEL (PER METER)"};
    if (kind < 0 || kind > 3) decline("an LD type this copy does not name");
    if (truthy(card.f(4))) {
        r.has_resistance = true;
        r.resistance = card.f(4);
    }
    if (truthy(card.f(5))) {
        r.has_inductance = true;
        r.inductance = card.f(5);
    }
    if (truthy(card.f(6))) {
        r.has_capacitance = true;
        r.capacitance = card.f(6);
    }
    r.kind = names[kind];
    return r;
}

// `DeckSolver._load_impedances(omega, group_index)`
std::vector<cd> load_impedances(const BuildTables &bt, ll n_ports, double omega, ll group_index) {
    std::vector<cd> z(static_cast<std::size_t>(n_ports), cd(0.0, 0.0));
    const std::vector<ll> *live = nullptr;
    for (const auto &g : bt.group_loads)
        if (g.first == group_index) live = &g.second;
    if (live == nullptr) return z;
    const cd jw = py_mul(cd(0.0, 1.0), cd(omega, 0.0));  // 1j * omega
    for (ll k : *live) {
        const auto &ml = bt.model_loads.at(static_cast<std::size_t>(k));
        if (ml.first < 0) decline("a load with no port");
        const LoadSpec &ld = ml.second;
        cd add;
        if (ld.kind == "fixed") {
            add = cd(ld.r, ld.x);
        } else if (ld.kind == "parallel") {
            cd y(0.0, 0.0);
            if (ld.r != 0.0) y = py_add(y, cd(1.0 / ld.r, 0.0));
            if (ld.l != 0.0) y = py_add(y, py_div(cd(1.0, 0.0), py_mul(jw, cd(ld.l, 0.0))));
            if (ld.c != 0.0) y = py_add(y, py_mul(jw, cd(ld.c, 0.0)));
            add = (y.real() != 0.0 || y.imag() != 0.0) ? py_div(cd(1.0, 0.0), y) : cd(0.0, 0.0);
        } else {
            // `_series_rlc_impedance(r or None, l or None, c or None, omega)`
            cd s(0.0, 0.0);
            if (ld.r != 0.0) s = cd(s.real() + ld.r, s.imag() + 0.0);
            if (ld.l != 0.0) s = py_add(s, py_mul(jw, cd(ld.l, 0.0)));
            if (ld.c != 0.0) s = py_add(s, py_div(cd(1.0, 0.0), py_mul(jw, cd(ld.c, 0.0))));
            add = s;
        }
        cd &slot = z.at(static_cast<std::size_t>(ml.first));
        slot = cd(slot.real() + add.real(), slot.imag() + add.imag());
    }
    return z;
}

// `portal._wire_arc_at_knot` / the knot walk `_element_match` and
// `element_currents` share: per solver polyline, its knots (numpy's
// linspace per edge, the first knot of each later edge dropped).
std::vector<double> knots_of(const std::vector<Point3> &poly, const std::vector<ll> &npe) {
    std::vector<double> knots;
    for (std::size_t e = 0; e < npe.size(); e++) {
        std::vector<double> seg;
        rp::linspace3(poly[e].data(), poly[e + 1].data(), npe[e] + 1, seg);
        knots.insert(knots.end(), seg.begin() + (e == 0 ? 0 : 3), seg.end());
    }
    return knots;
}
std::vector<double> arc_at_knot(const std::vector<double> &knots) {
    const std::size_t n = knots.size() / 3;
    std::vector<double> arc(n, 0.0);
    double run = 0.0;
    for (std::size_t q = 1; q < n; q++) {
        double d[3];
        for (int c = 0; c < 3; c++) d[c] = knots[3 * q + c] - knots[3 * (q - 1) + c];
        run = run + norm3(d);  // np.cumsum: sequential
        arc[q] = run;
    }
    return arc;
}

// The structure's solver polylines, as `RunEngine` numbers them.
struct Walk {
    std::vector<std::vector<double>> knots;  // per polyline
    std::vector<std::vector<double>> mids, dirs;
    std::vector<std::size_t> offset;         // first element of each polyline
    std::size_t n_elements = 0;
};

Walk walk_of(const SolverArgs &args) {
    Walk w;
    for (std::size_t p = 0; p < args.wires.size(); p++) {
        w.knots.push_back(knots_of(args.wires[p], args.n_per_edge[p]));
        const std::vector<double> &k = w.knots.back();
        const std::size_t n = k.size() / 3;
        w.offset.push_back(w.n_elements);
        std::vector<double> mid, dir;
        for (std::size_t q = 0; q + 1 < n; q++)
            for (int c = 0; c < 3; c++) {
                mid.push_back(0.5 * (k[3 * (q + 1) + c] + k[3 * q + c]));
                dir.push_back(k[3 * (q + 1) + c] - k[3 * q + c]);
            }
        w.n_elements += n - 1;
        w.mids.push_back(mid);
        w.dirs.push_back(dir);
    }
    return w;
}

// `DeckSolver._element_match(solver)`: the element nearest each NEC
// segment's centre among those aligned with it. The alignment test reads
// `unit_s @ unit_e.T`, numpy's real matmul: (M, 3) C-ordered by the (3, E)
// transposed view of a C-ordered (E, 3), which numpy hands to
// `cblas_dgemm(RowMajor, NoTrans, Trans, M, E, 3, 1, unit_s, 3, unit_e, 3,
// 0, out, E)` (both operands blasable, neither a vector); a single segment or
// element would take numpy's gemv route, which this copy declines.
std::vector<std::size_t> element_match(const Walk &w, const BuildTables &bt, const nec5::NumpyLinalg &np) {
    std::vector<const double *> mid, dir;
    for (std::size_t p = 0; p < w.mids.size(); p++)
        for (std::size_t q = 0; q < w.mids[p].size() / 3; q++) {
            mid.push_back(&w.mids[p][3 * q]);
            dir.push_back(&w.dirs[p][3 * q]);
        }
    const std::size_t E = mid.size(), M = bt.segments.size();
    if (E < 2 || M < 2) decline("an element match numpy reads by gemv");
    if (np.dgemm == nullptr) decline("no real BLAS for the element match");
    std::vector<double> unit_e(3 * E), unit_s(3 * M);
    for (std::size_t e = 0; e < E; e++) {
        const double n = std::max(norm3(dir[e]), 1e-30);
        for (int c = 0; c < 3; c++) unit_e[3 * e + static_cast<std::size_t>(c)] = dir[e][c] / n;
    }
    for (std::size_t i = 0; i < M; i++) {
        const Point3 &d = bt.segments[i].direction;
        const double n = std::max(norm3(d), 1e-30);
        for (int c = 0; c < 3; c++) unit_s[3 * i + static_cast<std::size_t>(c)] = d[static_cast<std::size_t>(c)] / n;
    }
    std::vector<double> dots(M * E);
    np.dgemm(101, 111, 112, static_cast<nec5::np_int>(M), static_cast<nec5::np_int>(E), 3, 1.0, unit_s.data(), 3,
             unit_e.data(), 3, 0.0, dots.data(), static_cast<nec5::np_int>(E));
    std::vector<std::size_t> nearest;
    for (std::size_t i = 0; i < M; i++) {
        const SegmentRow &s = bt.segments[i];
        double best = std::numeric_limits<double>::infinity(), best_d2 = best;
        std::size_t arg = 0, arg_d2 = 0;
        bool any = false;
        for (std::size_t e = 0; e < E; e++) {
            const double dx = s.centre[0] - mid[e][0], dy = s.centre[1] - mid[e][1], dz = s.centre[2] - mid[e][2];
            const double d2 = (dx * dx + dy * dy) + dz * dz;  // (...)**2 .sum(axis=2)
            if (std::isnan(d2) || std::isnan(dots[i * E + e])) decline("a non-finite element match");
            if (std::fabs(dots[i * E + e]) >= 0.5) {
                any = true;
                if (d2 < best) {
                    best = d2;
                    arg = e;
                }
            }
            if (d2 < best_d2) {
                best_d2 = d2;
                arg_d2 = e;
            }
        }
        nearest.push_back(any ? arg : arg_d2);
    }
    return nearest;
}

struct RunResult {
    std::vector<std::pair<ll, std::pair<ll, cd>>> driven;  // (port, (segment, volts))
    std::vector<cd> v_gap, v_applied, i_port, i_source, coeffs;
    std::vector<cd> segment_currents, segment_charges;
    double p_in = 0, p_structure = 0, p_radiated = 0, efficiency = 0;
    double fill_seconds = 0, wavelength = 0;
    std::size_t point = 0;
};

struct Ctx {
    const StructureSeam *ss = nullptr;
    const StructureBuild *sb = nullptr;
    const Tables *t = nullptr;
    RunEngine *engine = nullptr;
    const nec5::NumpyLinalg *np = nullptr;
    std::map<std::size_t, Walk> walks;
    std::map<std::size_t, std::vector<std::size_t>> matches;
    const Walk &walk(std::size_t point) {
        auto it = walks.find(point);
        if (it == walks.end()) it = walks.emplace(point, walk_of(sb->points[point].args)).first;
        return it->second;
    }
    const std::vector<std::size_t> &match(std::size_t point) {
        auto it = matches.find(point);
        if (it == matches.end()) it = matches.emplace(point, element_match(walk(point), sb->tables, *np)).first;
        return it->second;
    }
};

std::size_t point_for(const StructureBuild &sb, double freq, bool ek, const Environment &env) {
    if (env.ground.kind != Ground::none) decline("a ground (momwire#1607 B5)");
    for (std::size_t i = 0; i < sb.points.size(); i++)
        if (sb.points[i].frequency_mhz == freq && sb.points[i].extended_kernel == ek &&
            sb.points[i].args.ground.kind == Ground::none)
            return i;
    decline("an operating point the build did not make");
}

// `DeckSolver._segment_currents(solver, coeffs)`
std::vector<cd> segment_currents(Ctx &c, std::size_t point, const std::vector<cd> &coeffs) {
    const Walk &w = c.walk(point);
    const std::vector<std::size_t> &nearest = c.match(point);
    const std::vector<std::vector<cd>> knots = c.engine->currents_at_knots(point, coeffs);
    std::vector<cd> vals;  // `_element_values`: 0.5 * (v[1:] + v[:-1])
    std::vector<const double *> dirs;
    for (std::size_t p = 0; p < knots.size(); p++) {
        for (std::size_t q = 0; q + 1 < knots[p].size(); q++)
            vals.push_back(cm(cd(0.5, 0.0), cadd(knots[p][q + 1], knots[p][q])));
        for (std::size_t q = 0; q < w.dirs[p].size() / 3; q++) dirs.push_back(&w.dirs[p][3 * q]);
    }
    if (vals.size() != dirs.size()) decline("an engine's knots that do not match the walk");
    std::vector<cd> out;
    const std::vector<SegmentRow> &segs = c.sb->tables.segments;
    for (std::size_t i = 0; i < segs.size(); i++) {
        const std::size_t j = nearest[i];
        const double *d = dirs[j];
        const Point3 &sd = segs[i].direction;
        // `dot3` of two 3-vectors: Python floats
        const double dot = (d[0] * sd[0] + d[1] * sd[1]) + d[2] * sd[2];
        const double sign = dot >= 0 ? 1.0 : -1.0;
        out.push_back(scmul(cd(sign, 0.0), vals[j]));
    }
    return out;
}

// `DeckSolver._segment_charges(solver, coeffs, omega)`
std::vector<cd> segment_charges(Ctx &c, std::size_t point, const std::vector<cd> &coeffs, double omega) {
    const Walk &w = c.walk(point);
    const std::vector<std::size_t> &nearest = c.match(point);
    std::vector<std::vector<double>> centres;
    for (const std::vector<double> &k : w.knots) {
        const std::vector<double> arc = arc_at_knot(k);
        std::vector<double> ce;
        for (std::size_t q = 0; q + 1 < arc.size(); q++) ce.push_back(0.5 * (arc[q] + arc[q + 1]));
        centres.push_back(ce);
    }
    const std::vector<std::vector<cd>> slopes = c.engine->current_slopes(point, coeffs, centres);
    const cd jw = py_mul(cd(0.0, 1.0), cd(omega, 0.0));
    std::vector<cd> flat;
    for (const auto &s : slopes)
        for (const cd &v : s) flat.push_back(cdiv(cd(-v.real(), -v.imag()), jw));
    std::vector<cd> table;
    for (std::size_t i = 0; i < c.sb->tables.segments.size(); i++) table.push_back(flat.at(nearest[i]));
    double scale = 0.0;  // np.max(np.abs(table))
    bool first = true;
    for (const cd &v : table) {
        const double a = std::hypot(v.real(), v.imag());
        if (std::isnan(a)) decline("a non-finite charge");
        if (first || a > scale) scale = a;
        first = false;
    }
    if (scale > 0.0)
        for (cd &v : table)
            if (std::hypot(v.real(), v.imag()) < kChargeResidualFloor * scale) v = cd(0.0, 0.0);
    return table;
}

// `DeckSolver.solve_group(group, freq, group_index)` with no network
RunResult solve_group(Ctx &c, const PortalGroup &group, double freq, ll group_index) {
    const BuildTables &bt = c.sb->tables;
    const PortalDeck &deck = c.ss->deck;
    if (!deck.model.networks.empty()) decline("a TL / NT network (momwire#1607 B4c)");
    if (bt.lossy) decline("a lossy wire (momwire#1607 B5)");
    RunResult r;
    r.point = point_for(*c.sb, freq, group.ek, group.environment);
    const OperatingPoint &op = c.sb->points[r.point];
    const EngineSolve &es = c.engine->solve(r.point);
    r.fill_seconds = es.fill_seconds;
    r.wavelength = op.wavelength;
    const ll P = es.n_ports, N = es.n_basis;
    if (static_cast<ll>(op.signs.size()) != P) decline("a port-space mismatch (the reference raises)");
    const double omega = 2.0 * kPi * freq * 1e6;
    // Y in the deck's signs: y * signs[:, None] * signs[None, :]
    nec5::CMat y(P, P);
    for (ll i = 0; i < P; i++)
        for (ll j = 0; j < P; j++)
            y.at(i, j) = cm(cm(es.y[static_cast<std::size_t>(i * P + j)], cd(op.signs[static_cast<std::size_t>(i)], 0.0)),
                            cd(op.signs[static_cast<std::size_t>(j)], 0.0));
    // `_driven(group)`
    for (std::size_t k = 0; k < bt.ports.size() && k < bt.feed_index.size(); k++)
        for (const PortalSource &s : group.sources)
            if (s.tag == bt.ports[k].first && s.seg == bt.ports[k].second)
                r.driven.push_back({bt.feed_index[k],
                                    {global_segment(deck.structure.wires, s.tag, s.seg), s.volts}});
    auto is_current = [&](ll port) {
        return std::find(bt.current_ports.begin(), bt.current_ports.end(), port) != bt.current_ports.end();
    };
    bool has_set = false;
    cd set_current;
    if (r.driven.size() == 1 && is_current(r.driven[0].first)) {
        has_set = true;
        set_current = r.driven[0].second.second;
        r.driven[0].second.second = cd(1.0, 0.0);
    }
    std::vector<cd> v_source(static_cast<std::size_t>(P), cd(0.0, 0.0));
    for (const auto &d : r.driven) v_source.at(static_cast<std::size_t>(d.first)) = d.second.second;
    const std::vector<cd> z_load = load_impedances(bt, P, omega, group_index);
    if (r.driven.size() > 1)
        for (const auto &d : r.driven)
            if (is_current(d.first)) decline("phased current sources (momwire#1607 B4c)");
    nec5::CMat system(P, P);
    for (ll i = 0; i < P; i++)
        for (ll j = 0; j < P; j++)
            system.at(i, j) = cadd(cd(i == j ? 1.0 : 0.0, 0.0), cm(z_load[static_cast<std::size_t>(i)], y.at(i, j)));
    r.v_gap = np_solve(*c.np, system, v_source);
    r.i_port.assign(static_cast<std::size_t>(P), cd(0.0, 0.0));
    if (P > 0) {
        const nec5::Op vo{r.v_gap.data(), P, 1, 1, 0};
        rp::np_matmul(*c.np, y.op(), vo, r.i_port.data(), 1, 0);
    }
    r.v_applied = r.v_gap;
    r.i_source = r.i_port;
    if (has_set) {
        const ll port = r.driven[0].first;
        const cd is = r.i_source[static_cast<std::size_t>(port)];
        if (is == cd(0.0, 0.0)) decline("a current source on a port that takes no current (the reference refuses)");
        const cd scale = cdiv(set_current, is);
        for (auto *v : {&r.v_gap, &r.v_applied, &r.i_port, &r.i_source})
            for (cd &x : *v) x = cm(x, scale);
        r.i_source[static_cast<std::size_t>(port)] = set_current;
        r.driven[0].second.second = r.v_applied[static_cast<std::size_t>(port)];
    }
    // coeffs = X @ (signs * v_gap)
    std::vector<cd> sv(static_cast<std::size_t>(P));
    for (ll j = 0; j < P; j++)
        sv[static_cast<std::size_t>(j)] = cm(cd(op.signs[static_cast<std::size_t>(j)], 0.0), r.v_gap[static_cast<std::size_t>(j)]);
    r.coeffs.assign(static_cast<std::size_t>(N), cd(0.0, 0.0));
    if (N > 0) {
        const nec5::Op xo{es.x.data(), N, P, es.x_s0, es.x_s1};
        const nec5::Op vo{sv.data(), P, 1, 1, 0};
        rp::np_matmul(*c.np, xo, vo, r.coeffs.data(), 1, 0);
    }
    r.segment_currents = segment_currents(c, r.point, r.coeffs);
    r.segment_charges = segment_charges(c, r.point, r.coeffs, (2.0 * kPi * kCLight) / op.wavelength);
    // the power budget
    double total = 0.0;
    for (const auto &d : r.driven) {
        const cd i = r.i_source[static_cast<std::size_t>(d.first)];
        total = total + scmul(d.second.second, std::conj(i)).real();
    }
    r.p_in = 0.5 * total;
    std::vector<double> terms;
    for (ll p = 0; p < P; p++) {
        const cd i = r.i_port[static_cast<std::size_t>(p)];
        const double a = std::hypot(i.real(), i.imag());
        terms.push_back(z_load[static_cast<std::size_t>(p)].real() * (a * a));
    }
    const double p_load = 0.5 * rp::np_pairwise(terms.data(), static_cast<int64_t>(terms.size()));
    r.p_structure = (p_load + 0.0) + 0.0;
    r.p_radiated = r.p_in - r.p_structure;
    r.efficiency = r.p_in > 0 ? (100.0 * r.p_radiated) / r.p_in : 0.0;
    return r;
}

// `_charges` + `_charge_rows`
std::vector<ValueRow> charge_rows(Ctx &c, const RunResult &r, double freq, double local) {
    const std::vector<GeometryWire> &wires = c.ss->deck.structure.wires;
    const OperatingPoint &op = c.sb->points[r.point];
    const Walk &w = c.walk(r.point);
    const double omega = 2.0 * kPi * freq * 1e6;
    struct PEnd {
        Point3 start, end;
        double length;
    };
    std::vector<PEnd> pends;
    for (std::size_t p = 0; p < op.args.wires.size(); p++) {
        const std::vector<double> arc = arc_at_knot(w.knots[p]);
        pends.push_back({op.args.wires[p].front(), op.args.wires[p].back(), arc.back()});
    }
    std::vector<std::vector<double>> positions(pends.size());
    std::map<std::pair<std::size_t, bool>, std::pair<std::size_t, std::size_t>> slots;
    for (std::size_t wi = 0; wi < wires.size(); wi++) {
        const ll first = c.t->wire_first[wi], last = first + wires[wi].n_seg - 1;
        for (int side = 0; side < 2; side++) {
            const bool at_start = side == 0;
            const double *point = at_start ? wires[wi].p1 : wires[wi].p2;
            const bool free = at_start ? c.t->connections[static_cast<std::size_t>(first)].first == 0
                                       : c.t->connections[static_cast<std::size_t>(last)].second == 0;
            if (!free) continue;
            const double tol = std::max(norm3(point), 1.0) * 1e-9;
            for (std::size_t p = 0; p < pends.size(); p++) {
                double ds[3], de[3];
                for (int k = 0; k < 3; k++) {
                    ds[k] = pends[p].start[static_cast<std::size_t>(k)] - point[k];
                    de[k] = pends[p].end[static_cast<std::size_t>(k)] - point[k];
                }
                double s;
                if (norm3(ds) <= tol)
                    s = 0.0;
                else if (norm3(de) <= tol)
                    s = pends[p].length;
                else
                    continue;
                slots[{wi, at_start}] = {p, positions[p].size()};
                positions[p].push_back(s);
                break;
            }
        }
    }
    std::map<std::pair<std::size_t, bool>, cd> end_charges;
    if (!slots.empty()) {
        const std::vector<std::vector<cd>> slopes = c.engine->current_slopes(r.point, r.coeffs, positions);
        const cd jw = py_mul(cd(0.0, 1.0), cd(omega, 0.0));
        for (const auto &kv : slots) {
            const cd v = slopes.at(kv.second.first).at(kv.second.second);
            end_charges[kv.first] = cdiv(cd(-v.real(), -v.imag()), jw);
        }
    }
    std::vector<ValueRow> rows;
    const std::vector<SegmentRow> &segs = c.sb->tables.segments;
    for (std::size_t wi = 0; wi < wires.size(); wi++) {
        const ll first = c.t->wire_first[wi];
        const SegmentRow &s0 = segs[static_cast<std::size_t>(first)];
        const SegmentRow &sl = segs[static_cast<std::size_t>(first + wires[wi].n_seg - 1)];
        const double length = norm3(s0.direction);
        auto end_row = [&](const SegmentRow &seg, const double *point, cd q) {
            double ce[3];
            for (int k = 0; k < 3; k++) ce[k] = point[k] / local;
            rows.push_back(value_row(seg.number, i64(seg.tag), ce, length / local, q, true));
        };
        auto it = end_charges.find({wi, true});
        if (it != end_charges.end()) end_row(s0, wires[wi].p1, it->second);
        for (ll k = 0; k < wires[wi].n_seg; k++) {
            const SegmentRow &seg = segs[static_cast<std::size_t>(first + k)];
            double ce[3];
            for (int m = 0; m < 3; m++) ce[m] = seg.centre[static_cast<std::size_t>(m)] / local;
            rows.push_back(value_row(seg.number, i64(seg.tag), ce, length / local,
                                     r.segment_charges[static_cast<std::size_t>(seg.number - 1)]));
        }
        it = end_charges.find({wi, false});
        if (it != end_charges.end()) end_row(sl, wires[wi].p2, it->second);
    }
    return rows;
}

// `_pattern(card, result, freq, gd, medium)` in free space
PrintPatternBlock pattern(Ctx &c, const nec5::Card &card, const RunResult &r, double freq) {
    const Walk &w = c.walk(r.point);
    // `solver.element_currents(coeffs)` (subdiv 1: the knot currents)
    const std::vector<std::vector<cd>> cur = c.engine->currents_at_knots(r.point, r.coeffs);
    std::vector<double> mid;
    std::vector<cd> moment;
    for (std::size_t p = 0; p < cur.size(); p++) {
        const std::vector<double> &k = w.knots[p];
        if (cur[p].size() != k.size() / 3) decline("an engine's knots that do not match the walk");
        for (std::size_t q = 0; q + 1 < cur[p].size(); q++) {
            const cd element = cm(cd(0.5, 0.0), cadd(cur[p][q + 1], cur[p][q]));
            for (int m = 0; m < 3; m++) {
                mid.push_back(0.5 * (k[3 * (q + 1) + m] + k[3 * q + m]));
                moment.push_back(cm(element, cd(k[3 * (q + 1) + m] - k[3 * q + m], 0.0)));
            }
        }
    }
    nec5::Request rq;
    rq.kind = nec5::Request::far_field;
    const nec5::Int one(1);
    rq.n_theta = card.i(1) < one ? one : card.i(1);
    rq.n_phi = card.i(2) < one ? one : card.i(2);
    rq.xnda = card.i(3);
    rq.theta0_deg = card.f(4);
    rq.phi0_deg = card.f(5);
    rq.d_theta_deg = card.f(6);
    rq.d_phi_deg = card.f(7);
    rq.mode = nec5::Int(0);
    rp::PatternBlock b;
    try {
        b = rp::free_space_pattern(rq, mid, moment, freq, r.wavelength, r.p_in, r.p_radiated);
    } catch (const nec5::PrintoutDecline &e) {
        decline(e.reason);
    }
    PrintPatternBlock out;
    for (const rp::PatternRow &x : b.rows) {
        PrintPatternRow row;
        row.theta_deg = x.theta;
        row.phi_deg = x.phi;
        row.vert_db = x.vert;
        row.hor_db = x.hor;
        row.total_db = x.total;
        row.axial_ratio = x.axial;
        row.tilt_deg = x.tilt;
        row.sense = x.sense;
        row.e_theta_magnitude = x.et_mag;
        row.e_theta_phase_deg = x.et_deg;
        row.e_phi_magnitude = x.ep_mag;
        row.e_phi_phase_deg = x.ep_deg;
        out.rows.push_back(row);
    }
    out.has_average = b.has_average;
    out.average_power_gain = b.average_power_gain;
    out.solid_angle_pi = b.solid_angle_pi;
    out.power_radiated_4pi = b.power_radiated_4pi;
    out.directive = b.directive;
    const double range_m = card.f(8);
    if (range_m == 0.0) return out;
    // `_at_range(block, range_m, freq, medium=None, grounded=False)`
    const double k = (2.0 * kPi * freq) / kCMhzM;
    const double rad = -k * range_m;
    const double shift = py_degrees(py_atan2(std::sin(rad), std::cos(rad)));
    for (PrintPatternRow &row : out.rows) {
        row.e_theta_magnitude = row.e_theta_magnitude / range_m;
        row.e_phi_magnitude = row.e_phi_magnitude / range_m;
        row.e_theta_phase_deg = row.e_theta_phase_deg + shift;
        row.e_phi_phase_deg = row.e_phi_phase_deg + shift;
    }
    out.has_range = true;
    out.range_m = range_m;
    out.range_phase_deg = shift;
    return out;
}

// `_tables(card, ...)`: patterns only (an NE / NH declines)
void tables(Ctx &c, bool has_card, const nec5::Card &card, const RunResult &r, double freq,
            std::vector<PrintPatternBlock> &patterns) {
    if (!has_card) return;
    if (card.mnemonic == "RP") {
        patterns.push_back(pattern(c, card, r, freq));
        return;
    }
    decline("an " + card.mnemonic + " near field (momwire#1607 B4c)");
}

// `_run(...)`
PrintRun run(Ctx &c, const PortalGroup &group, ll index, double freq, const RunResult &r, bool has_echo,
             ll echo_through, bool preamble) {
    const PortalDeck &deck = c.ss->deck;
    const double wavelength = kCMhzM / freq;
    const double local = wavelength;
    PrintRun out;
    out.report = false;
    out.has_echo_through = has_echo;
    out.echo_through = echo_through;
    tables(c, group.has_report, group.report, r, freq, out.patterns);
    if (group.has_pq && i64(group.pq.flag) != -1) {
        out.has_charges = true;
        out.charges = charge_rows(c, r, freq, local);
    }
    std::map<ll, ll> gap_current;  // segment -> port
    for (const auto &d : r.driven) gap_current[d.second.first] = d.first;
    out.has_currents = true;
    for (const SegmentRow &seg : c.sb->tables.segments) {
        auto it = gap_current.find(seg.number);
        const cd value = it != gap_current.end() ? r.i_port[static_cast<std::size_t>(it->second)]
                                                 : r.segment_currents[static_cast<std::size_t>(seg.number - 1)];
        double ce[3];
        for (int m = 0; m < 3; m++) ce[m] = seg.centre[static_cast<std::size_t>(m)] / local;
        out.currents.push_back(value_row(seg.number, i64(seg.tag), ce, norm3(seg.direction) / local, value));
    }
    if (preamble) {
        out.has_preamble = true;
        PrintPreamble &p = out.preamble;
        p.frequency_mhz = freq;
        p.wavelength_m = wavelength;
        p.environment = "FREE SPACE";
        p.has_ground = false;
        for (const nec5::Card &ld : group.loads) p.loads.push_back(load_row(ld));
        p.fill_seconds = static_cast<double>(static_cast<ll>(std::nearbyint(r.fill_seconds * 1000.0))) / 1000.0;
        p.factor_seconds = 0.0;
    }
    // `_run_records`' source rows
    for (const auto &d : r.driven) {
        const cd volts = d.second.second;
        const cd current = r.i_source[static_cast<std::size_t>(d.first)];
        PrintPortRow row;
        row.tag = i64(c.sb->tables.segments[static_cast<std::size_t>(d.second.first - 1)].tag);
        row.segment = d.second.first;
        row.end_index = 0;
        row.voltage = volts;
        row.current = current;
        row.impedance = current != cd(0.0, 0.0) ? cdiv(volts, current) : cd(0.0, 0.0);
        row.admittance = volts != cd(0.0, 0.0) ? cdiv(current, volts) : cd(0.0, 0.0);
        row.power = 0.5 * scmul(volts, std::conj(current)).real();
        out.sources.push_back(row);
    }
    out.power.input_power = r.p_in;
    out.power.radiated_power = r.p_radiated;
    out.power.wire_loss = r.p_structure;
    out.power.efficiency_percent = r.efficiency;
    out.power.has_network_loss = false;
    (void)deck;
    (void)index;
    return out;
}

// `solve(text, basis)` for one built structure: its `Nec4Printout`
Nec4Printout solve_structure(Ctx &c) {
    const auto t0 = std::chrono::steady_clock::now();
    const StructureSeam &ss = *c.ss;
    const PortalDeck &deck = ss.deck;
    Nec4Printout data;
    std::size_t count = 0;
    for (ll index : ss.plan) {
        const PortalGroup &group = deck.groups.at(static_cast<std::size_t>(index));
        RunResult last;
        double last_freq = 0;
        for (std::size_t step = 0; step < group.freqs_mhz.size(); step++) {
            const double freq = group.freqs_mhz[step];
            last = solve_group(c, group, freq, index);
            last_freq = freq;
            const ll echo = ss.echo_at.at(static_cast<std::size_t>(index));
            data.runs.push_back(run(c, group, index, freq, last, step == 0 && echo >= 0, echo,
                                    count == 0 || group.refilled));
        }
        for (const auto &f : ss.further)
            if (f.first == index)
                for (const auto &nc : f.second) {
                    PrintRun rep;
                    rep.report = true;
                    rep.has_echo_through = true;
                    rep.echo_through = nc.first;
                    tables(c, true, nc.second, last, last_freq, rep.patterns);
                    data.runs.push_back(rep);
                }
        count++;
    }
    data.wires = wire_rows(deck);
    data.ge_flag = 0;
    for (const nec5::Card &card : deck.geometry)
        if (card.mnemonic == "GE") {
            data.ge_flag = i64(card.i(0));
            break;
        }
    data.total_segments = static_cast<ll>(c.sb->tables.segments.size());
    data.junctions = c.t->junctions;
    for (std::size_t i = 0; i < c.sb->tables.segments.size(); i++) {
        const SegmentRow &seg = c.sb->tables.segments[i];
        SegmentDataRow row;
        row.number = seg.number;
        for (int m = 0; m < 3; m++) row.centre[m] = seg.centre[static_cast<std::size_t>(m)];
        const Point3 &d = seg.direction;
        row.length = norm3(d);
        row.alpha_deg = py_degrees(py_atan2(d[2], std::hypot(d[0], d[1])));
        row.beta_deg = py_degrees(py_atan2(d[1], d[0]));
        row.radius = seg.radius;
        row.i_minus = c.t->connections[i].first;
        row.i_plus = c.t->connections[i].second;
        row.tag = i64(seg.tag);
        data.segments.push_back(row);
    }
    data.run_seconds = seconds_since(t0);
    return data;
}

}  // namespace

bool render(const std::string &text, const std::string &basis, const nec5::EngineStamp &stamp,
            EngineFactory &engines, const nec5::NumpyLinalg &np, std::string &out, std::string &why) {
    out.clear();
    why.clear();
    try {
        const Seam s = seam(text, basis);
        if (s.basis_refused) {
            why = "a refused basis";
            return false;
        }
        const Build b = build(text, basis);
        double seconds = 0.0;
        for (std::size_t k = 0; k < s.structures.size(); k++) {
            const StructureSeam &ss = s.structures[k];
            const std::string &echo = ss.echo_text;
            if (ss.kind == StructureSeam::refused) {
                out += render_nec4_refusal(&echo, ss.message, basis, stamp);
                return true;
            }
            if (ss.kind != StructureSeam::served) {
                why = ss.kind == StructureSeam::declined ? ss.message : "a structure the reference raises on";
                return false;
            }
            if (k >= b.structures.size()) {
                why = "a structure the build did not reach";
                return false;
            }
            const StructureBuild &sb = b.structures[k];
            if (sb.kind == StructureBuild::refused) {
                out += render_nec4_refusal(&echo, sb.message, basis, stamp);
                return true;
            }
            if (sb.kind != StructureBuild::built) {
                why = sb.kind == StructureBuild::declined ? sb.message : "a build the reference raises on";
                return false;
            }
            const Tables t = tables_of(ss, sb);
            Ctx c;
            c.ss = &ss;
            c.sb = &sb;
            c.t = &t;
            c.engine = &engines.engine_for(sb);
            c.np = &np;
            Nec4Printout data = solve_structure(c);
            const double own = data.run_seconds;
            data.run_seconds = own + seconds;
            out += render_nec4_printout(echo, data, basis, stamp, ss.closing);
            seconds += own;
        }
        return true;
    } catch (const Declined &e) {
        why = e.what();
    } catch (const nec5::PrintoutDecline &e) {
        why = e.reason;
    } catch (const std::exception &e) {
        why = std::string("an error this copy does not word: ") + e.what();
    }
    out.clear();
    return false;
}

}  // namespace nec4
}  // namespace mw
