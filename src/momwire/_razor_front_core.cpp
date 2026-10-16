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

#include "_razor_buried_core.h"
#include "_somm_below_grid_core.h"
#include "_leggauss_table.h"
#include "_accel_somm_proj_inline.h"

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
double max_image_distance(const double *seg_l, const double *seg_r, size_t n, double ground_z) {
    // `np.unique(concat(seg_l, seg_r), axis=0)`, then the max over pairs of
    // dx*dx + dy*dy + hz*hz, its root once, times 1.001. The pair value is
    // bit-symmetric and max is exact, so the distinct set (exact duplicates
    // dropped) and the order of the scan move no bit.
    std::vector<std::array<double, 3>> ex;
    ex.reserve(2 * n);
    for (size_t i = 0; i < n; i++) ex.push_back({seg_l[3 * i], seg_l[3 * i + 1], seg_l[3 * i + 2]});
    for (size_t i = 0; i < n; i++) ex.push_back({seg_r[3 * i], seg_r[3 * i + 1], seg_r[3 * i + 2]});
    if (ex.empty()) return 0.0;
    std::sort(ex.begin(), ex.end());
    ex.erase(std::unique(ex.begin(), ex.end()), ex.end());
    double s2 = 0.0;
    const size_t m = ex.size();
    for (size_t i = 0; i < m; i++)
        for (size_t j = i; j < m; j++) {
            const double dx = ex[i][0] - ex[j][0], dy = ex[i][1] - ex[j][1];
            const double hz = (ex[i][2] - ground_z) + (ex[j][2] - ground_z);
            const double v = dx * dx + dy * dy + hz * hz;
            if (v > s2) s2 = v;
        }
    return std::sqrt(s2) * 1.001;
}

int64_t remainder_qp(const double *obs, size_t n_obs, const double *src_l, const double *src_r,
                     size_t n_src, double ground_z, int64_t base, int64_t cap, double c) {
    if (n_src == 0 || n_obs == 0) return base;
    auto norm3 = [](double x, double y, double z) { return std::sqrt((x * x + y * y) + z * z); };
    std::vector<double> mir_l(3 * n_src), d(3 * n_src), dd(n_src), len(n_src);
    for (size_t j = 0; j < n_src; j++) {
        double ml[3], mr[3];
        for (int c3 = 0; c3 < 3; c3++) {
            ml[c3] = src_l[3 * j + c3];
            mr[c3] = src_r[3 * j + c3];
        }
        ml[2] = 2.0 * ground_z - src_l[3 * j + 2];
        mr[2] = 2.0 * ground_z - src_r[3 * j + 2];
        for (int c3 = 0; c3 < 3; c3++) {
            mir_l[3 * j + c3] = ml[c3];
            d[3 * j + c3] = mr[c3] - ml[c3];
        }
        const double *dj = &d[3 * j];
        dd[j] = (dj[0] * dj[0] + dj[1] * dj[1]) + dj[2] * dj[2];
        len[j] = norm3(src_r[3 * j] - src_l[3 * j], src_r[3 * j + 1] - src_l[3 * j + 1],
                       src_r[3 * j + 2] - src_l[3 * j + 2]);
    }
    // The O(N) short-circuit: nothing near the plane returns `base`.
    double h_l = src_l[2] - ground_z, h_r = src_r[2] - ground_z, h_o = obs[2] - ground_z;
    for (size_t j = 1; j < n_src; j++) {
        h_l = std::min(h_l, src_l[3 * j + 2] - ground_z);
        h_r = std::min(h_r, src_r[3 * j + 2] - ground_z);
    }
    for (size_t i = 1; i < n_obs; i++) h_o = std::min(h_o, obs[3 * i + 2] - ground_z);
    const double h_min = std::min(std::min(h_l, h_r), h_o);
    if (h_min > 0.0) {
        double lmax = len[0];
        for (size_t j = 1; j < n_src; j++) lmax = std::max(lmax, len[j]);
        const double bound = lmax / (2.0 * h_min);
        if (static_cast<int64_t>(std::ceil(c * bound)) <= base) return base;
    }
    double worst = 0.0;
    for (size_t j = 0; j < n_src; j++) {
        if (len[j] <= 0.0) continue;
        if (!(dd[j] > 0.0)) continue;  // a degenerate mirror has no broadside
        const double *ml = &mir_l[3 * j], *dj = &d[3 * j];
        bool any = false;
        double r_min = 0.0;
        for (size_t i = 0; i < n_obs; i++) {
            const double *o = &obs[3 * i];
            const double ax = o[0] - ml[0], ay = o[1] - ml[1], az = o[2] - ml[2];
            const double t_raw = (ax * dj[0] + ay * dj[1] + az * dj[2]) / dd[j];
            if (!(t_raw > 0.0 && t_raw < 1.0)) continue;  // broadside only
            const double cx = ml[0] + t_raw * dj[0], cy = ml[1] + t_raw * dj[1],
                         cz = ml[2] + t_raw * dj[2];
            const double r = norm3(o[0] - cx, o[1] - cy, o[2] - cz);
            if (!any || r < r_min) r_min = r;
            any = true;
        }
        if (!any) continue;
        if (r_min <= 0.0) return cap;
        worst = std::max(worst, len[j] / r_min);
    }
    const int64_t need = worst > 0.0 ? static_cast<int64_t>(std::ceil(c * worst)) : 0;
    return std::min(std::max(base, need), cap);
}

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

const char *const BURIED_EXTENDED_KERNEL_REFUSAL =
    "extended_kernel=True + a wire below the ground plane is not served: the "
    "extended kernel extends every pair of the geometry, and momwire#553 "
    "measured neither what that means for a pair spanning two media (the "
    "tube expansion's O(a^2) term is written at one wavenumber) nor for the "
    "image of a buried source, which lands in the OTHER medium. "
    "Solve the buried deck with extended_kernel=False, which is the default";

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
    // momwire#1042: a wire listed twice is MERGED (or refused) by
    // `drop_duplicated_wires` in `build` (`dedup`), before `construct`.
    (void)n_w;
    if (in.exact_kernel == ExactKernel::on) return "exact-kernel";
    if (in.ground_eps_given) {
        // The reflection-coefficient ground's weighted fold is not here; the
        // composing one is, on the shipped grid constants only.
        if (!in.ground_sommerfeld) return "refl-coef-ground";
        for (const char *name : {"MOMWIRE_SOMM_R1_CAP_LAMBDA", "MOMWIRE_SOMM_R1_NEAR_LAMBDA",
                                 "MOMWIRE_SOMM_EPS_IM_BUCKET", "MOMWIRE_SOMM_R1_SWEEP_SLACK"}) {
            const char *v = std::getenv(name);
            if (v != nullptr && v[0] != '\0') return "somm-environment";
        }
    }
    return "";
}

void Front::construct(const FrontInputs &in) {
    // `RazorSolver.__init__`, in its order, on the arguments this scope
    // keeps (the loading, ground-model and buried arguments are the
    // caller's to have routed elsewhere).
    has_ground = in.has_ground;
    ground_z = in.ground_z;
    if (has_ground && !std::isfinite(ground_z)) value_error("ground_z must be finite");
    if (in.ground_eps_given && !has_ground) value_error("ground_eps requires ground_z to be set");
    compose = in.ground_eps_given;
    ground_eps_r_ = in.ground_eps_r;
    ground_sigma_ = in.ground_sigma;
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

namespace {

// `_medium_spec` / `_below_interface` / razor's buried sentences, word for
// word (momwire#1512 B1).
const char *const CROSSING_REFUSAL_TEXT =
    "momwire serves "
    "wires wholly at or above the interface, wires strictly below it, and "
    "current CROSSING it only through a crossing junction (momwire#524 "
    "phase 2): split the wire AT the interface into a below wire whose end "
    "stands in the plane and an above wire starting there, and declare the "
    "junction between them - that deck is served, with continuity of "
    "current and the interface slope condition emerging from the fill "
    "itself. A single polyline with points on both sides is not split for "
    "you here, because feeds, loads, junctions and per-wire results are "
    "keyed on its wire index and a split would renumber them; the deck seams "
    "split a card wire at the plane themselves and keep its addresses (the "
    "NEC-2 portal since momwire#667, the EZNEC seam since momwire#1281). "
    "Alternatively leave the "
    "buried part DETACHED (a buried radial screen under a base-fed "
    "vertical is served that way, momwire#553), or raise the whole wire "
    "clear of the interface";
const char *const BURIED_PEC_REFUSAL_TEXT =
    "over a PERFECTLY CONDUCTING ground, which has no lower medium "
    "to put it in: the field inside a perfect conductor is identically zero, "
    "so a wire there is not buried, it is shorted out. A buried wire is "
    "served only under ground_model='sommerfeld' with a ground_eps, where "
    "the half-space below the interface is a real medium with a wavenumber "
    "k_m = k0*sqrt(eps_tilde) of its own (momwire#553). Raise the wire to or "
    "above the plane, or give the solve a Sommerfeld ground";
const char *const CONTACT_WITH_BURIED_REFUSAL_TEXT =
    "that COMBINATION is not served, though each "
    "half is. momwire's contact model continues the wire's current into the "
    "ground as the C2-scaled IMAGE (momwire#151), which is a fiction that "
    "works because no observer is ever inside the ground to look at it - and "
    "a buried wire is exactly such an observer. What it should see is the "
    "contact current SPREADING in the lower medium - a real soil current "
    "this deck has NO CONDUCTOR for. momwire#524 phase 2 measured every "
    "consistent spelling of the missing cross-medium physics against "
    "engine prints for exactly these decks - and those prints have since "
    "been withdrawn as a reference (momwire#929). They were captured with "
    "the ground card's two fields written in the wrong order (antennaknobs#1025), and this "
    "deck class turns out to have no documented spelling at all: the flag "
    "that serves buried wires leaves a conductor ENDING on the interface "
    "without a basis function there, and the flag that bonds such an end is "
    "documented as not usable when wires are buried. So the ~3 ohm (lone) / "
    "~6 ohm (fan) residual was measured against a combination the engine "
    "does not document, and neither it nor the point-electrode reading "
    "drawn from it is evidence about this deck. The refusal itself is "
    "unchanged and rests on momwire's own physics: the contact model has no "
    "conductor for the spreading current, so there is nothing honest to "
    "serve here in either convention. "
    "Serve it by giving the spreading current its conductor: respell the "
    "radial - or the whole SCREEN of them - to RISE to the surface and "
    "junction-join the monopole at z = 0; that crossing junction is served "
    "for one above wire over N below wires (momwire#524 phase 2, fan "
    "widening), including the screen's buried-hub spelling (one rise, N "
    "radials joined at depth). Or raise the above-ground wire clear of "
    "the interface (an elevated feed over a buried counterpoise is "
    "served), or solve the buried structure on its own";

// Python's `format(x, ".<p>g")` / `"{:g}"` (p = 6): printf's %g in the C
// locale, with Python's spellings of the non-finite values.
std::string py_g(double x, int p = 6) {
    if (std::isnan(x)) return "nan";
    if (std::isinf(x)) return x > 0 ? "inf" : "-inf";
    char fmt[16], buf[64];
    std::snprintf(fmt, sizeof fmt, "%%.%dg", p);
#if defined(_MSC_VER)
    static _locale_t c = _create_locale(LC_NUMERIC, "C");
    _snprintf_l(buf, sizeof buf, fmt, c, x);
#else
    static locale_t c = newlocale(LC_NUMERIC_MASK, "C", (locale_t)0);
    locale_t old = uselocale(c);
    std::snprintf(buf, sizeof buf, fmt, x);
    uselocale(old);
#endif
    return buf;
}

}  // namespace

// `_refuse_buried_geometry`: `_wire_media` (`_medium_spec.wire_media` with
// the crossing exemption `_grounded_junction_ends`), then
// `_crossing_junctions` (`_below_interface.crossing_junctions`,
// two_radius=True) and the route. Every refusal is Python's, in its order.
void Front::refuse_buried_geometry(const std::vector<std::pair<int64_t, End>> &touching) {
    buried = Buried::none;
    media_below.assign(static_cast<size_t>(n_wires), 0);
    crossing.clear();
    if (!has_ground) return;
    const double gz = ground_z;
    const auto pl = [&](int64_t w) { return points.data() + 3 * pt_off[w]; };
    const auto np_ = [&](int64_t w) { return static_cast<ptrdiff_t>(pt_off[w + 1] - pt_off[w]); };
    const auto zmin_of = [&](int64_t w) {
        const double *p = pl(w);
        double z = p[2];
        for (ptrdiff_t i = 1; i < np_(w); i++) z = std::min(z, p[3 * i + 2]);
        return z;
    };
    const auto zmax_of = [&](int64_t w) {
        const double *p = pl(w);
        double z = p[2];
        for (ptrdiff_t i = 1; i < np_(w); i++) z = std::max(z, p[3 * i + 2]);
        return z;
    };
    const auto end_point = [&](int64_t w, End e) {
        return e == End::start ? pl(w) : pl(w) + 3 * (np_(w) - 1);
    };
    // Every wire at or above the plane: the common answer, no labels to read.
    bool any_below = false;
    for (int64_t w = 0; w < n_wires; w++)
        if (!(zmin_of(w) >= gz - ground_touch_tol(pl(w), np_(w)))) any_below = true;
    if (!any_below) return;

    // `_find_junctions` (the detected groups) and the exemption
    // (`_medium_spec.grounded_crossing_exemption`).
    std::vector<Junction> groups;
    detect_junctions(touching, groups);
    std::set<std::pair<int64_t, End>> exempt;
    for (const Junction &g : groups) {
        if (g.ends.size() < 2) continue;
        const int64_t w0 = g.ends[0].first;
        const double *pt = end_point(w0, g.ends[0].second);
        if (std::fabs(pt[2] - gz) > ground_touch_tol(pl(w0), np_(w0))) continue;
        bool reaches = false;
        for (const auto &m : g.ends)
            if (zmax_of(m.first) > gz + ground_touch_tol(pl(m.first), np_(m.first))) reaches = true;
        if (!reaches) continue;
        for (const auto &m : g.ends) exempt.insert(m);
    }

    // `wire_media`'s walk.
    const bool lower_medium = compose;  // ground_eps under "sommerfeld" (scope)
    for (int64_t w = 0; w < n_wires; w++) {
        const double *p = pl(w);
        const ptrdiff_t m = np_(w);
        const double tol = ground_touch_tol(p, m);
        const double zmin = zmin_of(w), zmax = zmax_of(w);
        if (zmin >= gz - tol) continue;
        bool crossing_below = false;
        if (!(zmax > gz + tol)) {
            const bool ts = std::fabs(p[2] - gz) <= tol;
            const bool te = std::fabs(p[3 * (m - 1) + 2] - gz) <= tol;
            if (ts || te) {
                crossing_below = true;
                for (ptrdiff_t i = 1; i + 1 < m; i++)
                    if (!(p[3 * i + 2] < gz - tol)) crossing_below = false;
                if (m > 2) {
                    double imax = p[5];
                    for (ptrdiff_t i = 1; i + 1 < m; i++) imax = std::max(imax, p[3 * i + 2]);
                    if (imax >= gz - tol) crossing_below = false;
                }
                if (ts && !exempt.count({w, End::start})) crossing_below = false;
                if (te && !exempt.count({w, End::end})) crossing_below = false;
            }
        }
        if (zmax >= gz - tol && !crossing_below)
            value_error("wire " + std::to_string(w) + " crosses the ground interface mid-span "
                        "(polyline z runs " + py_g(zmin) + " to " + py_g(zmax) +
                        " across ground_z = " + py_g(gz) + "): " + CROSSING_REFUSAL_TEXT);
        if (!lower_medium)
            value_error("wire " + std::to_string(w) + " runs below the ground plane (min z = " +
                        py_g(zmin) + " < ground_z = " + py_g(gz) + ") " + BURIED_PEC_REFUSAL_TEXT);
        media_below[static_cast<size_t>(w)] = 1;
    }
    int64_t first_below = -1;
    for (int64_t w = 0; w < n_wires; w++)
        if (media_below[static_cast<size_t>(w)]) {
            first_below = w;
            break;
        }
    // `_ground_spec.contact_ends`, sorted as Python sorts ("end" < "start").
    std::vector<std::pair<int64_t, End>> contacts;
    for (int64_t w = 0; w < n_wires; w++) {
        const double *p = pl(w);
        const ptrdiff_t m = np_(w);
        const double tol = ground_touch_tol(p, m);
        if (std::fabs(p[3 * (m - 1) + 2] - gz) <= tol) contacts.emplace_back(w, End::end);
        if (std::fabs(p[2] - gz) <= tol) contacts.emplace_back(w, End::start);
    }
    for (const auto &c : contacts)
        if (!exempt.count(c))
            value_error("wire " + std::to_string(c.first) +
                        " stands an END in the ground plane (ground CONTACT) and wire " +
                        std::to_string(first_below) + " is buried below it: " +
                        CONTACT_WITH_BURIED_REFUSAL_TEXT);

    // `_crossing_junctions`.
    for (size_t j = 0; j < groups.size(); j++) {
        if (!groups[j].grounded) continue;
        bool a = false, b = false;
        for (const auto &m : groups[j].ends) (media_below[static_cast<size_t>(m.first)] ? b : a) = true;
        if (a && b) crossing.push_back(j);
    }
    {
        std::set<std::pair<int64_t, End>> earned;
        for (size_t j : crossing)
            for (const auto &m : groups[j].ends) earned.insert(m);
        for (const auto &c : contacts)
            if (!earned.count(c))
                value_error("wire " + std::to_string(c.first) +
                            " stands an END in the ground plane (ground CONTACT) and wire " +
                            std::to_string(first_below) + " is buried below it: " +
                            CONTACT_WITH_BURIED_REFUSAL_TEXT);
    }
    if (!crossing.empty()) {
        if (crossing.size() > 1) {
            std::vector<const double *> pts;
            for (size_t j : crossing) pts.push_back(end_point(groups[j].ends[0].first, groups[j].ends[0].second));
            double best = 0.0;
            size_t bi = 0, bj = 0;
            bool have = false;
            for (size_t i = 0; i < pts.size(); i++)
                for (size_t q = i + 1; q < pts.size(); q++) {
                    const double sep = std::hypot(pts[i][0] - pts[q][0], pts[i][1] - pts[q][1]);
                    if (!have || sep < best || (std::isnan(sep) && !std::isnan(best))) {
                        if (!have || !std::isnan(best)) {
                            best = sep;
                            bi = i;
                            bj = q;
                            have = true;
                        }
                    }
                }
            if (best < 1.0)
                not_implemented(
                    "crossing junctions " + std::to_string(crossing[bi]) + " and " +
                    std::to_string(crossing[bj]) + ", at (" + py_g(pts[bi][0]) + ", " +
                    py_g(pts[bi][1]) + ") and (" + py_g(pts[bj][0]) + ", " + py_g(pts[bj][1]) +
                    "), stand " + py_g(best) +
                    " m apart: the crossing serve completes several crossing nodes per deck "
                    "from 1 m apart, the closest separation its two-node collapse was gated at "
                    "(antennaknobs plan U9), and closer crossing nodes have no measured completion");
        }
        for (size_t j : crossing) {
            int n_above = 0;
            for (const auto &m : groups[j].ends)
                if (!media_below[static_cast<size_t>(m.first)]) n_above++;
            if (n_above != 1)
                not_implemented(
                    "crossing junction with more than one above member: the "
                    "crossing serve joins ONE above wire to N below wires "
                    "at the interface (momwire#524 fan widening); the "
                    "above-tent x above-tent interface corner has no "
                    "measured convention");
        }
        for (size_t j = 0; j < groups.size(); j++) {
            if (std::find(crossing.begin(), crossing.end(), j) != crossing.end()) continue;
            bool a = false, b = false;
            for (const auto &m : groups[j].ends) (media_below[static_cast<size_t>(m.first)] ? b : a) = true;
            if (!groups[j].grounded && !(a && b)) continue;
            not_implemented(
                "a deck with a crossing junction and an in-plane or "
                "media-mixing OTHER junction is not served: the complete "
                "crossing spelling completes every value-1 end on its axes, "
                "and a junction standing in the interface sits where the "
                "corner and image terms live, which has no measured "
                "completion (momwire#524 phase 2, momwire#1133)");
        }
        double rmax = radius_per_wire[0], rmin = radius_per_wire[0];
        for (double r : radius_per_wire) {
            rmax = std::max(rmax, r);
            rmin = std::min(rmin, r);
        }
        if (rmax - rmin > 0.0) {
            // `crossing_side_radii(media, radii, crossing_above_member(...))`
            int64_t above_member = -1;
            if (crossing.size() == 1)
                for (const auto &m : groups[crossing[0]].ends)
                    if (!media_below[static_cast<size_t>(m.first)]) {
                        above_member = m.first;
                        break;
                    }
            for (int side = 0; side < 2; side++) {
                if (side == 0 && above_member >= 0) continue;
                bool have = false;
                double lo = 0.0, hi = 0.0;
                for (int64_t w = 0; w < n_wires; w++) {
                    if ((media_below[static_cast<size_t>(w)] != 0) != (side == 1)) continue;
                    const double r = radius_per_wire[static_cast<size_t>(w)];
                    if (!have) {
                        lo = hi = r;
                        have = true;
                    } else {
                        lo = std::min(lo, r);
                        hi = std::max(hi, r);
                    }
                }
                if (hi - lo > 0.0)
                    not_implemented(std::string("crossing serve with per-wire radii that differ within the ") +
                                    (side == 0 ? "above" : "below") +
                                    " wires: the two-radius rule serves one radius per side "
                                    "of the interface (the node's point tests at the buried "
                                    "radius, antennaknobs plan U5); a spread within one side has "
                                    "no measured node radius");
            }
            if (crossing.size() > 1)
                not_implemented("a TWO-RADIUS crossing deck with " + std::to_string(crossing.size()) +
                                " crossing junctions: the two-radius node (antennaknobs plan U5) and more "
                                "than one crossing node (plan U9) were each measured alone, and "
                                "a two-radius deck with several nodes has no measured completion");
        }
        buried = Buried::crossing;
        return;
    }
    bool any_above = false;
    for (int64_t w = 0; w < n_wires; w++)
        if (!media_below[static_cast<size_t>(w)]) any_above = true;
    buried = any_above ? Buried::detached : Buried::below_plane;
}

// `_junction_rule.coincident_groups`: each point's representative (the
// first point within JUNCTION_TOL of it, by the cell grid when it can
// promise the walk's answer, else the walk). `pts` is (n_pts, 3).
static std::vector<size_t> coincident_reps(const std::vector<double> &pts, size_t n_pts) {
    const double tol = JUNCTION_TOL;
    auto within = [&](size_t i, size_t j) {
        double d[3] = {pts[3 * i] - pts[3 * j], pts[3 * i + 1] - pts[3 * j + 1],
                       pts[3 * i + 2] - pts[3 * j + 2]};
        return std::sqrt((d[0] * d[0] + d[1] * d[1]) + d[2] * d[2]) <= tol;
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
    return rep;
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
    const std::vector<size_t> rep = coincident_reps(pts, n_pts);
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
    for (size_t gi = 0; gi < groups_.size(); gi++) {
        // A crossing group takes the free-space topology (momwire#813):
        // its grounded flag is demoted here, as `_build_geometry` does.
        Junction g = groups_[gi];
        if (std::find(crossing.begin(), crossing.end(), gi) != crossing.end()) g.grounded = false;
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

int64_t Front::solved_index(int64_t w) const {
    if (!has_dedup) return w;
    for (size_t i = 0; i < dedup_kept.size(); i++)
        if (dedup_kept[i] == w) return static_cast<int64_t>(i);
    return -1;
}

namespace {

// `_wire_spec._same` on the per-wire values the seam passes.
bool same_value(double a, double b) { return (std::isnan(a) && std::isnan(b)) || a == b; }
bool same_rlc(const WireRLC &a, const WireRLC &b) {
    if (a.kind == WireRLC::none || b.kind == WireRLC::none) return a.kind == b.kind;
    return a.kind == b.kind && a.r == b.r && a.l == b.l && a.c == b.c;
}

}  // namespace

// `_wire_spec.drop_duplicated_wires` for `RazorSolver` (inferred junctions,
// no junction-indexed ports): false when no copy is merged (`in` stands);
// else `out` is `in` on the kept wires, every wire index remapped, and the
// map is recorded. Refusals are its ValueErrors.
bool Front::dedup(const FrontInputs &in, FrontInputs &out) {
    const size_t n = in.wire_points.size();
    std::vector<size_t> off(n + 1, 0);
    for (size_t w = 0; w < n; w++) off[w + 1] = off[w] + static_cast<size_t>(in.wire_points[w]);
    // `find_duplicated_wires`: nanometre-rounded vertex keys, either
    // direction, then equal within 1e-12 of the coordinate scale
    double scale = 1.0;
    bool any = false;
    for (size_t w = 0; w < n; w++)
        for (size_t i = 3 * off[w]; i < 3 * off[w + 1]; i++) {
            scale = std::max(scale, std::fabs(in.points[i]));
            any = true;
        }
    if (!any) return false;
    const double tol = 1e-12 * scale;
    struct Dup {
        size_t copy, first;
        bool rev;
    };
    std::vector<Dup> dups;
    std::map<std::vector<double>, size_t> seen;
    for (size_t w = 0; w < n; w++) {
        const size_t m = off[w + 1] - off[w];
        if (m == 0) continue;
        std::vector<double> fwd, rev;
        for (size_t i = 0; i < m; i++)
            for (int c = 0; c < 3; c++) fwd.push_back(round9(in.points[3 * (off[w] + i) + c]) + 0.0);
        for (size_t i = m; i-- > 0;)
            for (int c = 0; c < 3; c++) rev.push_back(round9(in.points[3 * (off[w] + i) + c]) + 0.0);
        const std::vector<double> &key = std::min(fwd, rev);
        auto it = seen.find(key);
        if (it == seen.end()) {
            seen.emplace(key, w);
            continue;
        }
        const size_t first = it->second;
        if (off[first + 1] - off[first] != m) continue;
        double d_fwd = 0.0, d_rev = 0.0;
        for (size_t i = 0; i < m; i++)
            for (int c = 0; c < 3; c++) {
                const double a = in.points[3 * (off[first] + i) + c];
                d_fwd = std::max(d_fwd, std::fabs(a - in.points[3 * (off[w] + i) + c]));
                d_rev = std::max(d_rev, std::fabs(a - in.points[3 * (off[w] + m - 1 - i) + c]));
            }
        if (d_fwd <= tol)
            dups.push_back({w, first, false});
        else if (d_rev <= tol)
            dups.push_back({w, first, true});
    }
    if (dups.empty()) return false;

    // `_merge_connected_alike` over `coincident_end_groups` (inferred)
    std::vector<double> ends(6 * n);
    for (size_t w = 0; w < n; w++) {
        const size_t m = off[w + 1] - off[w];
        for (int c = 0; c < 3; c++) {
            ends[3 * (2 * w) + c] = in.points[3 * off[w] + c];
            ends[3 * (2 * w + 1) + c] = in.points[3 * (off[w] + m - 1) + c];
        }
    }
    const std::vector<size_t> rep = coincident_reps(ends, 2 * n);
    std::vector<int64_t> size_of(2 * n, 0), group_of(2 * n, -1);
    for (size_t i = 0; i < 2 * n; i++) size_of[rep[i]]++;
    {
        int64_t next = 0;
        std::vector<int64_t> id(2 * n, -1);
        for (size_t i = 0; i < 2 * n; i++) {
            if (size_of[rep[i]] < 2) continue;  // lone ends are no group
            if (id[rep[i]] < 0) id[rep[i]] = next++;
            group_of[i] = id[rep[i]];
        }
    }
    // `joined(w, rev)`: ends named in the class's first wire's direction
    const auto joined = [&](size_t w, bool rev) {
        return std::make_pair(group_of[2 * w + (rev ? 1 : 0)], group_of[2 * w + (rev ? 0 : 1)]);
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
    std::vector<std::pair<size_t, std::pair<size_t, bool>>> removed;  // insertion order
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
            if (match == nullptr)
                reps.push_back({member.first, member.second, key});
            else
                removed.push_back({member.first, {match->w, member.second != match->rev}});
        }
    }
    if (removed.empty()) return false;

    const auto refuse = [](size_t r, size_t twin, const std::string &why) {
        value_error("RazorSolver: wires " + std::to_string(twin) + " and " + std::to_string(r) +
                    " are the same wire listed twice, and " + why +
                    ", so they cannot be merged into one conductor. Delete wire " +
                    std::to_string(r) + ", or make the two copies identical - momwire#1042");
    };
    const bool radius_list = !in.radius_shape.empty() && in.radius.size() == n;
    for (const auto &rm : removed) {
        const size_t r = rm.first, twin = rm.second.first;
        const bool rev = rm.second.second;
        if (in.counts_given && in.counts.size() == n) {
            std::vector<int64_t> a = in.counts[twin].v, b = in.counts[r].v;
            if (rev) std::reverse(b.begin(), b.end());
            if (a != b) refuse(r, twin, "their segment counts differ");
        }
        if (radius_list && !same_value(in.radius[r], in.radius[twin]))
            refuse(r, twin, "their wire_radius entries differ");
        if (in.loading.has_conductivity && in.loading.conductivity.size() == n &&
            !same_value(in.loading.conductivity[r], in.loading.conductivity[twin]))
            refuse(r, twin, "their wire_conductivity entries differ");
        if (in.loading.has_rlc && in.loading.rlc.size() == n &&
            !same_rlc(in.loading.rlc[r], in.loading.rlc[twin]))
            refuse(r, twin, "their distributed_rlc entries differ");
        for (const FeedSpec &f : in.feeds)
            if (f.wire == static_cast<int64_t>(r) || f.wire == static_cast<int64_t>(twin))
                refuse(r, twin, "feeds places a site on wire " + std::to_string(f.wire));
        for (const NodeGapSpec &g : in.node_gaps)
            if (g.wire == static_cast<int64_t>(r) || g.wire == static_cast<int64_t>(twin))
                refuse(r, twin, "node_gaps places a site on wire " + std::to_string(g.wire));
    }

    std::vector<bool> dropped(n, false);
    for (const auto &rm : removed) dropped[rm.first] = true;
    std::vector<int64_t> new_index(n, -1);
    out = in;
    out.points.clear();
    out.wire_points.clear();
    out.counts.clear();
    out.radius.clear();
    out.loading.conductivity.clear();
    out.loading.rlc.clear();
    dedup_kept.clear();
    for (size_t w = 0; w < n; w++) {
        if (dropped[w]) continue;
        new_index[w] = static_cast<int64_t>(dedup_kept.size());
        dedup_kept.push_back(static_cast<int64_t>(w));
        out.points.insert(out.points.end(), in.points.begin() + 3 * off[w], in.points.begin() + 3 * off[w + 1]);
        out.wire_points.push_back(in.wire_points[w]);
        if (in.counts_given && in.counts.size() == n) out.counts.push_back(in.counts[w]);
        if (radius_list) out.radius.push_back(in.radius[w]);
        if (in.loading.has_conductivity && in.loading.conductivity.size() == n)
            out.loading.conductivity.push_back(in.loading.conductivity[w]);
        if (in.loading.has_rlc && in.loading.rlc.size() == n) out.loading.rlc.push_back(in.loading.rlc[w]);
    }
    if (!(in.counts_given && in.counts.size() == n)) out.counts = in.counts;
    if (radius_list)
        out.radius_shape.assign(1, static_cast<int64_t>(dedup_kept.size()));
    else
        out.radius = in.radius;
    if (!(in.loading.has_conductivity && in.loading.conductivity.size() == n))
        out.loading.conductivity = in.loading.conductivity;
    if (!(in.loading.has_rlc && in.loading.rlc.size() == n)) out.loading.rlc = in.loading.rlc;
    for (FeedSpec &f : out.feeds)
        if (f.wire >= 0 && f.wire < static_cast<int64_t>(n)) f.wire = new_index[static_cast<size_t>(f.wire)];
    for (NodeGapSpec &g : out.node_gaps)
        if (g.wire >= 0 && g.wire < static_cast<int64_t>(n)) g.wire = new_index[static_cast<size_t>(g.wire)];
    std::sort(removed.begin(), removed.end());
    dedup_removed.clear();
    for (const auto &rm : removed)
        dedup_removed.push_back({static_cast<int64_t>(rm.first), static_cast<int64_t>(rm.second.first),
                                 rm.second.second});
    has_dedup = true;
    n_input_wires = static_cast<int64_t>(n);
    return true;
}

std::string Front::build(const FrontInputs &in_raw) {
    const std::string why = scope(in_raw);
    if (!why.empty()) return why;
    FrontInputs reduced;
    const FrontInputs &in = dedup(in_raw, reduced) ? reduced : in_raw;
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

    // `_refuse_buried_geometry` (momwire#1512 B1): the wire media with the
    // crossing exemption, the crossing serve's scope, and the route.
    refuse_buried_geometry(touching);
    extended_kernel = in.extended_kernel;
    if (extended_kernel && buried != Buried::none)
        not_implemented(std::string("razor's buried fill does not take the extended kernel: ") +
                        BURIED_EXTENDED_KERNEL_REFUSAL);

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
    // `_serves_exact_kernel` is False on every buried route.
    if (in.exact_kernel == ExactKernel::automatic && buried == Buried::none)
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
    n_qp_sommerfeld = in.n_qp_sommerfeld;
    chunk_elems = in.chunk_elems;
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
    // A buried deck fills medium by medium (`_assemble_Z_crossing` /
    // `_assemble_Z_below_plane`, `_razor_buried_core.h`): the full geometry
    // keeps only its loading; each medium's front is built at the fill.
    // Served end to end once the below table has a source (B5's fill, or the
    // gates' bridge); until then the deck declines here.
    if (buried != Buried::none) return somm_below::source() != nullptr ? "" : BURIED_FILL_PENDING;
    if (compose) {
        const std::string why_c = prepare_compose(in.n_qp_sommerfeld);
        if (!why_c.empty()) return why_c;
    }
    // `_schedule.elem_rows(budget, n_path * n_basis)`, the budget
    // `_WEIGHTED_CHUNK_ELEMS` (= `_CHUNK_ELEMS // 12`) under a weighted image.
    const int64_t budget = compose ? in.chunk_elems / 12 : in.chunk_elems;
    window_rows = std::max<int64_t>(1, budget / std::max<int64_t>(1, 2 * n_basis));
    chunk_elems = in.chunk_elems;
    // The correction's geometry-only tables.
    exact_plan.groups.clear();
    if (exact) {
        const std::string why = build_exact_plan(*this, exact_plan);
        if (!why.empty()) return why;
    }
    return "";
}

// The composing ground's per-solve half, at the fill's k and omega:
// `potential_ground_for`'s eps~ and C2, the weighted rule's operands, and
// `Remainder.field_windows`' grid, source nodes and moment weights, each
// spelled as the Python expression it replaces.
std::string Front::prepare_compose(int64_t n_qp_sommerfeld) {
    const size_t n = static_cast<size_t>(n_basis), ns = static_cast<size_t>(n_seg);
    // `complex(float(eps_r), -float(sigma) / (omega * eps0))`
    eps_t = cd(ground_eps_r_, (-ground_sigma_) / (omega * EPS));
    // `(eps_t - 1.0) / (eps_t + 1.0)`
    c2 = somm_core::py_quot(cd(eps_t.real() - 1.0, eps_t.imag() - 0.0),
                            cd(eps_t.real() + 1.0, eps_t.imag() + 0.0));
    // The path points (`_testing_paths`' NEC-5 branch: each wing's
    // centroid) and the wing signs.
    obs_pts.resize(6 * n);
    sig_a.resize(n);
    sig_b.resize(n);
    for (size_t m = 0; m < n; m++) {
        for (int c3 = 0; c3 < 3; c3++) {
            obs_pts[6 * m + c3] = cent[3 * s_a[m] + c3];
            obs_pts[6 * m + 3 + c3] = cent[3 * s_b[m] + c3];
        }
        sig_a[m] = wing_sigma[2 * m];
        sig_b[m] = wing_sigma[2 * m + 1];
    }
    // The remainder's REAL source segments: seg_p0 and
    // `seg_p0 + seg_h[:, None] * seg_t`.
    std::vector<double> src_r(3 * ns);
    for (size_t s = 0; s < ns; s++)
        for (int c3 = 0; c3 < 3; c3++)
            src_r[3 * s + c3] = seg_p0[3 * s + c3] + seg_h[s] * seg_t[3 * s + c3];
    n_qp_rem = remainder_qp(obs_pts.data(), 2 * n, seg_p0.data(), src_r.data(), ns, ground_z,
                            n_qp_sommerfeld, REMAINDER_QP_CAP, REMAINDER_QP_C);
    r1_max = max_image_distance(seg_p0.data(), src_r.data(), ns, ground_z);
    grid = &somm_core::get_grid(eps_t, k, r1_max, omega, MU);
    std::vector<double> xq, wq;
    if (!leggauss(n_qp_rem, xq, wq)) return "remainder-order";
    const size_t q = static_cast<size_t>(n_qp_rem);
    rem_src.resize(3 * ns * q);
    rem_t_src.resize(3 * ns * q);
    rem_W.resize(2 * ns * q);
    for (size_t s = 0; s < ns; s++) {
        double span[3];
        for (int c3 = 0; c3 < 3; c3++) span[c3] = src_r[3 * s + c3] - seg_p0[3 * s + c3];
        const double h = std::sqrt((span[0] * span[0] + span[1] * span[1]) + span[2] * span[2]);
        for (size_t j = 0; j < q; j++) {
            const double tq = 0.5 * (xq[j] + 1.0);
            for (int c3 = 0; c3 < 3; c3++) {
                rem_src[3 * (s * q + j) + c3] = seg_p0[3 * s + c3] + tq * span[c3];
                rem_t_src[3 * (s * q + j) + c3] = seg_t[3 * s + c3];
            }
            const double u = h * tq, w_node = 0.5 * h * wq[j];
            rem_W[s * q + j] = w_node * 1.0;          // u ** 0
            rem_W[(ns + s) * q + j] = w_node * u;     // u ** 1
        }
    }
    // `grid_cpp_args(grid)`: the regions flattened, theta split in radians
    // (`math.radians(30.0)`), and the momwire#1258 far pack.
    const somm_core::Grid &g = *grid;
    n_reg = g.regions.size();
    if (n_reg > somm_proj::MAX_REGIONS) return "grid-layout";
    double r0[somm_proj::MAX_REGIONS], dr[somm_proj::MAX_REGIONS], th0[somm_proj::MAX_REGIONS],
        dth[somm_proj::MAX_REGIONS];
    const cd *vals[somm_proj::MAX_REGIONS];
    std::ptrdiff_t shape0[somm_proj::MAX_REGIONS], nr[somm_proj::MAX_REGIONS],
        nth[somm_proj::MAX_REGIONS];
    for (size_t r = 0; r < n_reg; r++) {
        const somm_core::GridRegion &reg = g.regions[r];
        r0[r] = reg.r0;
        dr[r] = reg.dr;
        th0[r] = reg.th0;
        dth[r] = reg.dth;
        vals[r] = reg.vals.data();
        shape0[r] = 4;
        nr[r] = static_cast<std::ptrdiff_t>(reg.n_r);
        nth[r] = static_cast<std::ptrdiff_t>(reg.n_th);
    }
    grid_view = std::make_shared<somm_proj::GridView>(somm_proj::build_grid_view(
        g.r1_max, g.r_break, 30.0 * (M_PI / 180.0), g.r_near, r0, dr, th0, dth, n_reg, vals,
        shape0, nr, nth));
    const std::vector<cd> far = somm_core::far_pack(g);
    somm_proj::set_far(*grid_view, far.data(), static_cast<std::ptrdiff_t>(far.size()));
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
    s.compose = compose;
    if (compose) {
        Compose &c = s.comp;
        const ptrdiff_t nq = static_cast<ptrdiff_t>(n_qp_rem);
        c.eps_t = eps_t;
        c.c2 = c2;
        c.ground_z = ground_z;
        c.obs_pts = view<const double, 2>::contiguous(obs_pts.data(), {2 * n, 3});
        c.obs_tans = s.tans;
        c.src_c = s.cent;
        c.src_t = s.real.seg_t;
        c.sig_a = view<const double, 1>::contiguous(sig_a.data(), {n});
        c.sig_b = view<const double, 1>::contiguous(sig_b.data(), {n});
        c.grid = grid_view.get();
        c.n_reg = n_reg;
        c.rem_src = view<const double, 2>::contiguous(rem_src.data(), {ns * nq, 3});
        c.rem_t_src = view<const double, 2>::contiguous(rem_t_src.data(), {ns * nq, 3});
        c.W = rem_W.data();
        c.n_qp = nq;
    }
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
    if (front.buried != Buried::none) {
        // The buried fill (`_razor_buried_core.h`), its loading included.
        std::vector<cd> z;
        const std::string why = fill_buried(front, z, nullptr, cancel_flag);
        if (!why.empty()) throw std::runtime_error("the buried fill stopped at " + why);
        ctx.set_z(z.data(), n);
        return ctx.lu_solve(lp, x, static_cast<int>(m));
    }
    fill_front(front, ctx, cancel_flag);
    if (!front.load_vals.empty())
        ctx.add_at(front.load_rows.data(), front.load_cols.data(), front.load_vals.data(),
                   front.load_vals.size());
    return ctx.lu_solve(lp, x, static_cast<int>(m));
}

}  // namespace razor
}  // namespace mw
