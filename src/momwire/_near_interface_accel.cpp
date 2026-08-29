// The C++ twin of `_near_interface.six_point` (momwire#680 U2).
//
// This is a WALK port, not an integrand port (the house twin rule: shared
// walk + limit, never pointwise): the head/mid machinery is the very same
// templated engine `_accelerators.cpp` rides (`_contour_engine_inline.h`,
// momwire#568 U1 — `adaptive_segment`, `head_contour`, the complex J0/J1
// kernels), instantiated here with the crossing serve's six-component
// spectral core, plus a transcription of the ONE structure the engine does
// not carry: the rotated-ray tail lam = lam_top + t e^{+-j pi/4}. Every
// structural rule of the Python walk survives verbatim:
//
//   * head [0, 1.1K] via the first-quadrant half-sine detour (branch points
//     + transmitted pole inside), mid [1.1K, 8K] real-axis adaptive Gauss,
//     tail = rotated rays;
//   * ray panels START at the lam0 scale and double toward the decay scale
//     sqrt(2)/(s+rho) — starting at the decay scale is a measured 44 %
//     silent W-log error at s = 1e-5;
//   * exact-underflow ray panels count as QUIET (high-sigma far pairs
//     underflow the whole tail to exact 0.0 — the tail IS zero, not a
//     stall); two consecutive quiet panels end the ray;
//   * the far-pair kill cap lam <= 60/s, with the head keeping the k_p
//     branch point + transmitted pole inside;
//   * rho = 0 takes ONE up-ray with J0 = 1; rho > 0 splits J0 into Hankel
//     halves, H1 up-ray / H2 down-ray (conjugate ray direction);
//   * derivative bookkeeping: dz <-> -gamma_p, dz' <-> +gamma_m; z' <= 0 so
//     e^{-gamma_m|z'|} = e^{+gamma_m z'}.
//
// The complex-argument Hankel pair comes from the VENDORED scipy/xsf
// (extern/xsf, header-only C++17, scipy.special's own Amos translation —
// see extern/xsf/VENDOR-PIN.md). Its underflow contract is load-bearing:
// Amos underflow returns the exact 0.0 rather than NaN, which is precisely
// the quiet-panel rule above. This is also why the module is compiled at
// C++17 in its OWN extension: `_accelerators` stays at gnu++11, untouched.
//
// The numpy walk (`_near_interface.six_point`) stays production's reference;
// the gates against it are RELATIVE (1e-12), never bit — the transcendental
// libraries and the Gauss dot products differ in the last bits (house rule:
// no cross-build bit equality, momwire#249/#270).
//
// REENTRANCY: the engine header is allocation-free with no mutable static
// (its #568 U1 contract), the spectral core below holds only per-point
// scalars, and xsf's Amos kernels are pure functions — so the batch loop
// runs OpenMP `dynamic` over points with the GIL released, exactly the
// `below_six_integrals_batch` pattern. Domain validation happens BEFORE the
// parallel region (a throw cannot cross an omp boundary); the one in-loop
// failure mode (a ray that never goes quiet) sets a flag that is raised
// afterwards in the Python walk's own words.

#define _USE_MATH_DEFINES

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

#include <algorithm>
#include <cmath>
#include <complex>
#include <stdexcept>

#include "_contour_engine_inline.h"
#include "_branch_cut_inline.h"
#include "xsf/bessel.h"

namespace py = pybind11;

namespace mw680 {
using mw_contour::cd;

// The Python walk's fixed shape constants (`_near_interface`): the ray
// direction e^{+j pi/4}, the panel budget, and the far-pair kill cap.
// Transcribed as constants rather than parameters because they are part of
// the walk's SHAPE — a twin that let them drift would not be a twin.
static const int MW_MAX_RAY_PANELS = 90;
static const double MW_FAR_PAIR_KILL = 60.0;

// The branch cut: one definition for all three call sites, across both
// extensions — the full rationale (the two-sqrt spelling IS the branch
// choice; head detour and rotated rays cross neither vertical cut) lives
// with the definition in _branch_cut_inline.h (#714). Calls in this
// namespace are unqualified; the using-declaration binds them.
using mw_branch::gamma_cut;

// `_near_interface._core`: the six spectral factors x (2 E~ lam), WITHOUT
// the Bessel factor. z' <= 0, so e^{-gamma_m|z'|} = e^{+gamma_m z'}.
// Stacked: 0 U, 1 V, 2 W, 3 dzW (= -gamma_p W~ under the integral),
// 4 dz dz' V (= -gamma_p gamma_m V~), 5 dz'W (= +gamma_m W~).
struct CoreSix {
    double z;
    double zp;
    cd k_p;
    cd k_m;
    cd kp2;  // k_p^2, hoisted
    cd km2;  // k_m^2, hoisted
    void operator()(const cd &lam, cd *out) const {
        const cd g_p = gamma_cut(lam, k_p);
        const cd g_m = gamma_cut(lam, k_m);
        const cd e = 2.0 * std::exp(g_m * zp - g_p * z) * lam;
        const cd u = e / (g_p + g_m);
        const cd v = e / (km2 * g_p + kp2 * g_m);
        const cd w = (g_p - g_m) * v;
        out[0] = u;
        out[1] = v;
        out[2] = w;
        out[3] = -g_p * w;
        out[4] = -(g_p * g_m) * v;
        out[5] = g_m * w;
    }
};

// The head/mid integrand: core(lam) * J0(lam rho), on the engine's own
// complex J0 kernels (gated against scipy.special.jv at #568 U1).
struct J0Core {
    const CoreSix &core;
    double rho;
    J0Core(const CoreSix &c, double r) : core(c), rho(r) {}
    void operator()(const cd &lam, cd *out) const {
        core(lam, out);
        cd b0, b1x;
        mw_contour::bessel_j0_j1x(lam * rho, b0, b1x);
        for (int c = 0; c < 6; ++c) out[c] *= b0;
    }
};

// One rotated ray's integrand over the REAL panel parameter t:
// core(lam0 + t dir) * factor * dir, factor in {1, H1_0(lam rho)/2,
// H2_0(lam rho)/2}. The Hankel halves are xsf's Amos pair at complex
// argument; on the up-ray lam rho gains positive imaginary part and H1
// decays, on the (conjugate-direction) down-ray H2 does — underflow to
// exact 0.0 included, which the quiet rule wants.
enum RayKind { RAY_ONE = 0, RAY_H1 = 1, RAY_H2 = 2 };

struct RayIntegrand {
    const CoreSix &core;
    cd lam0;
    cd dir;
    double rho;
    int kind;
    void operator()(double t, cd *out) const {
        const cd lam = lam0 + t * dir;
        core(lam, out);
        cd fac = dir;
        if (kind == RAY_H1) fac *= 0.5 * xsf::cyl_hankel_1(0.0, lam * rho);
        else if (kind == RAY_H2) fac *= 0.5 * xsf::cyl_hankel_2(0.0, lam * rho);
        for (int c = 0; c < 6; ++c) out[c] *= fac;
    }
};

// `_near_interface._ray_integral`: geometric panels, each adaptive Gauss;
// stops when two consecutive panels contribute < rtol of the running total.
// Panels START at min(0.25 * scale, lam0) and double (the 44 % trap above).
// Returns false when the panel budget is exhausted without going quiet —
// the caller raises in the Python walk's words, outside the omp region.
static bool ray_integral(const RayIntegrand &g, double lam0, double scale,
                         double rtol, int depth, const double *gx,
                         const double *gw, int ng, cd *acc) {
    for (int c = 0; c < 6; ++c) acc[c] = cd(0.0, 0.0);
    double t_lo = 0.0;
    double step = std::min(0.25 * scale, lam0);
    int quiet = 0;
    cd part[6];
    for (int p = 0; p < MW_MAX_RAY_PANELS; ++p) {
        const double t_hi = t_lo + step;
        mw_contour::adaptive_segment<6>(g, t_lo, t_hi, rtol, depth, gx, gw, ng,
                                        0, 0.0, part);
        double ref = 0.0, pmax = 0.0;
        for (int c = 0; c < 6; ++c) {
            acc[c] += part[c];
            ref = std::max(ref, std::abs(acc[c]));
            pmax = std::max(pmax, std::abs(part[c]));
        }
        // ref == 0.0: the whole ray underflows to exact 0 (high-sigma far
        // pairs); consecutive all-zero panels are quiet, the tail IS zero.
        if (ref == 0.0 || pmax < rtol * ref) {
            if (++quiet >= 2) return true;
        } else {
            quiet = 0;
        }
        t_lo = t_hi;
        step *= 2.0;
    }
    return false;
}

// `six_point` for ONE (rho, z, zp): head + mid + tail. Returns false on a
// non-quiet ray (see above); every other pathway matches the Python walk's
// arithmetic term for term.
static bool six_point_one(double k_p, const cd &k_m, double rho, double z,
                          double zp, double rtol, double lam_mult, int depth,
                          double detour, const double *gx, const double *gw,
                          int ng, cd *out) {
    const double s = z - zp;
    const double kk = std::max(k_p, std::abs(k_m));
    double a_head = 1.1 * kk;
    double lam_top = lam_mult * kk;
    // Far-pair kill cap (sigma = 5 class, |k_m| >> k_p): beyond
    // lam ~ 60/s the integrand is e^{-60} of the total — dead range. Cap
    // the extents there, keeping the k_p branch point + transmitted pole
    // (|lam_p| ~ k_p) inside the head. Inactive for every near-interface
    // pair, so nothing the corner probes pinned changes.
    if (s > 0.0 && MW_FAR_PAIR_KILL / s < lam_top) {
        const double lam_kill = MW_FAR_PAIR_KILL / s;
        a_head = std::max(2.2 * k_p, std::min(a_head, lam_kill));
        lam_top = std::max(1.5 * a_head, lam_kill);
    }

    CoreSix core;
    core.z = z;
    core.zp = zp;
    core.k_p = cd(k_p, 0.0);
    core.k_m = k_m;
    core.kp2 = core.k_p * core.k_p;
    core.km2 = k_m * k_m;
    const J0Core f_j0(core, rho);

    cd head[6], mid[6];
    const double marks[2] = {k_p, std::fabs(k_m.real())};
    mw_contour::head_contour<6>(f_j0, a_head, rho, marks, 2, rtol, depth,
                                detour, gx, gw, ng, head);
    const mw_contour::TailWrap<6, J0Core> f_mid(f_j0);
    mw_contour::adaptive_segment<6>(f_mid, a_head, lam_top, rtol, depth, gx,
                                    gw, ng, 0, 0.0, mid);

    const double scale = std::sqrt(2.0) / (s + rho);
    const cd ray_dir = std::exp(cd(0.0, 0.25 * mw_contour::MW_PI));
    cd tail[6];
    if (rho == 0.0) {
        RayIntegrand g{core, cd(lam_top, 0.0), ray_dir, rho, RAY_ONE};
        if (!ray_integral(g, lam_top, scale, rtol, depth, gx, gw, ng, tail))
            return false;
    } else {
        RayIntegrand up{core, cd(lam_top, 0.0), ray_dir, rho, RAY_H1};
        RayIntegrand dn{core, cd(lam_top, 0.0), std::conj(ray_dir), rho,
                        RAY_H2};
        cd dn_acc[6];
        if (!ray_integral(up, lam_top, scale, rtol, depth, gx, gw, ng, tail))
            return false;
        if (!ray_integral(dn, lam_top, scale, rtol, depth, gx, gw, ng, dn_acc))
            return false;
        for (int c = 0; c < 6; ++c) tail[c] += dn_acc[c];
    }
    for (int c = 0; c < 6; ++c) out[c] = head[c] + mid[c] + tail[c];
    return true;
}
}  // namespace mw680

// `six_point` over parallel (rho, z, zp) arrays: the (n, 6) table, OpenMP
// across points with the GIL released. The wavenumbers arrive DERIVED
// (k_p real, k_m complex on the Im <= 0 branch) rather than as eps~, so the
// branch choice stays in `k_medium` where it is written down once; the
// domain raises keep the Python walk's words. The U1 exact-triple memo
// stays in Python — this entry expects the caller to hand it UNIQUE
// triples (that list is also U3's parallel unit).
static py::array_t<std::complex<double>> near_interface_six_batch(
    double k_p, std::complex<double> k_m,
    py::array_t<double, py::array::c_style | py::array::forcecast> rho,
    py::array_t<double, py::array::c_style | py::array::forcecast> z,
    py::array_t<double, py::array::c_style | py::array::forcecast> zp,
    double rtol, double lam_mult, int depth, double detour,
    py::array_t<double, py::array::c_style | py::array::forcecast> gx,
    py::array_t<double, py::array::c_style | py::array::forcecast> gw) {
    auto rb = rho.unchecked<1>();
    auto zb = z.unchecked<1>();
    auto pb = zp.unchecked<1>();
    const py::ssize_t n = rb.shape(0);
    if (zb.shape(0) != n || pb.shape(0) != n)
        throw std::invalid_argument("rho, z and zp must have the same length");
    if (gx.ndim() != 1 || gw.ndim() != 1 || gx.shape(0) != gw.shape(0) ||
        gx.shape(0) < 1)
        throw std::invalid_argument("bad Gauss rule");
    const double *gxp = gx.data();
    const double *gwp = gw.data();
    const int ng = static_cast<int>(gx.shape(0));

    // The Python walk's domain raises, verbatim and BEFORE the parallel
    // region (a throw cannot cross an omp boundary).
    for (py::ssize_t i = 0; i < n; ++i) {
        if (!(zb(i) >= 0.0 && pb(i) <= 0.0))
            throw std::invalid_argument("need z >= 0 >= zp");
        if (rb(i) < 0.0 || (zb(i) - pb(i)) + rb(i) <= 0.0)
            throw std::invalid_argument("need R > 0");
    }

    py::array_t<std::complex<double>> vals({n, py::ssize_t(6)});
    auto vb = vals.mutable_unchecked<2>();
    const mw_contour::cd km(k_m);
    bool ray_failed = false;

    {
        py::gil_scoped_release release;
        // `dynamic`: a corner point costs many more adaptive levels than a
        // mid-range pair, and a fill's unique triples arrive sorted by
        // geometry — static scheduling would hand one thread the hard band.
        #pragma omp parallel for schedule(dynamic)
        for (py::ssize_t i = 0; i < n; ++i) {
            mw_contour::cd out[6];
            if (!mw680::six_point_one(k_p, km, rb(i), zb(i), pb(i), rtol,
                                      lam_mult, depth, detour, gxp, gwp, ng,
                                      out)) {
                #pragma omp atomic write
                ray_failed = true;
                continue;
            }
            for (int c = 0; c < 6; ++c) vb(i, c) = out[c];
        }
    }
    if (ray_failed)
        throw std::runtime_error(
            "rotated tail did not go quiet inside the panel budget");
    return vals;
}

PYBIND11_MODULE(_near_interface_accel, m) {
    m.doc() =
        "momwire#680 U2: the C++ twin of _near_interface.six_point. "
        "Optional; _near_interface falls back to the numpy walk without it.";
    // The capability flag `_near_interface._HAVE_NEAR_INTERFACE_ACCEL`
    // keys on. Its OWN name (not contour_engine_568 / below_fills_568): a
    // .so built at an earlier arc exports those but not this entry, and a
    // shared flag would claim a contract it cannot serve.
    m.attr("near_interface_680") = true;
    m.def("near_interface_six_batch", &near_interface_six_batch,
          py::arg("k_p"), py::arg("k_m"), py::arg("rho"), py::arg("z"),
          py::arg("zp"), py::arg("rtol"), py::arg("lam_mult"),
          py::arg("depth"), py::arg("detour"), py::arg("gx"), py::arg("gw"),
          "six_point over parallel (rho, z, zp) arrays -> (n, 6) complex. "
          "k_p/k_m arrive derived (k_medium keeps the branch choice); the "
          "U1 memo layer in Python hands this UNIQUE triples only.");
}
