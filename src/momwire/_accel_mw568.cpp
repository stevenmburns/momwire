#include "_accel_common.h"
#include "_branch_cut_inline.h"
#include "_accel_somm_proj_inline.h"
// The shared adaptive-contour engine (momwire#568 unit 1): the C++ twin of
// `_sommerfeld_below`'s head + tail machinery, templated on the integrand.
// Header-only, reentrant, allocation-free -- U2/U3 instantiate it with their
// own integrands under OpenMP with the GIL released. Included HERE and not in
// _accel_common.h: this TU is the only consumer, and live engine work
// (momwire#696) should rebuild one TU, not five.
#include "_contour_engine_inline.h"

// mw568 section of the former _accelerators.cpp monolith (momwire#687).
// Code below is byte-identical to the monolith's lines 6273-7497.

// --------------------------------------------------------------------------
// momwire#568 unit 1 -- the shared contour engine's TEST instantiations live
// in register_mw568 below (contour_engine_sommerfeld_identity /
// contour_engine_synth6). They exist so `tests/test_contour_engine_568.py`
// can gate the engine and its complex Bessel pair from Python BEFORE any
// production fill rides on them; nothing in momwire's dispatch calls them.
// U2 (below fills) and U3 (transmitted fills) instantiate
// `mw_contour::run_contour` with their own integrands and never go through
// those entry points. (Banner relocated from the somm TU tail by the #710
// review — the section cut had stranded it above register_somm's PRODUCTION
// bindings.)
// --------------------------------------------------------------------------

namespace mw568 {
using mw_contour::cd;

// (a) The Sommerfeld identity's integrand,
//
//     f(lam) = J0(lam rho) e^{-gamma h} lam / gamma,   gamma = sqrt(lam^2 - k^2)
//
// whose contour integral is exactly e^{-jkR}/R with R = sqrt(rho^2 + h^2).
// Same branch point, same oscillatory tail, same exponential decay as the
// production integrands -- but with an analytic answer to gate against.
//
// The principal sqrt (Re gamma >= 0) is the right branch for the engine's
// FIRST-quadrant head: under e^{+j omega t} the cut runs downward from +k, an
// upward detour never crosses it, and at lam = 0 the signed-zero imaginary
// part of `lam*lam - k*k` lands sqrt on +j k -- the outgoing plane wave.
struct SommIdentity {
    double rho;
    double h;
    cd k;
    void operator()(const cd &lam, cd *out) const {
        const cd g = std::sqrt(lam * lam - k * k);
        cd j0, j1x;
        mw_contour::bessel_j0_j1x(lam * rho, j0, j1x);
        out[0] = j0 * std::exp(-g * h) * lam / g;
    }
};

// (b) A vector-valued (NC = 6) synthetic integrand with a Python twin in the
// test file, so numpy `_run_contour` and this engine can be run on the SAME
// mathematics and compared. Deliberately NOT the production kernel: it has
// the production shape (a gamma-decay factor, both Bessel pieces, components
// spanning several decades so the vector max-norm test is exercised) with a
// pole-free rational weight instead of the Sommerfeld D-pair, so U2 is free
// to write the real integrand however it likes.
struct Synth6 {
    double rho;
    double h;
    double k_p;
    cd k_m;
    void operator()(const cd &lam, cd *out) const {
        const cd gm = std::sqrt(lam * lam - k_m * k_m);
        const cd e = std::exp(-gm * h);
        const cd x = lam * rho;
        cd b0, b1x;
        mw_contour::bessel_j0_j1x(x, b0, b1x);
        const cd w = cd(1.0, 0.0) / (gm + lam + k_p);
        const cd l2 = lam * lam;
        out[0] = w * e * b0 * lam;
        out[1] = w * e * (b1x - b0) * l2;
        out[2] = -(w * e * b1x * x * gm * lam);
        out[3] = -(w * e * b1x * l2);
        out[4] = w * e * b0;
        out[5] = w * e * (b0 + 2.0 * b1x) * lam * gm;
    }
};

// Shared argument checking + raw Gauss-rule pointers.
static void gauss_view(const py::array_t<double, py::array::c_style |
                                                     py::array::forcecast> &gx,
                       const py::array_t<double, py::array::c_style |
                                                     py::array::forcecast> &gw,
                       const double **gxp, const double **gwp, int *ng) {
    if (gx.ndim() != 1 || gw.ndim() != 1)
        throw std::runtime_error("gx / gw must be 1-D");
    if (gx.shape(0) != gw.shape(0))
        throw std::runtime_error("gx and gw must have the same length");
    if (gx.shape(0) < 1) throw std::runtime_error("empty Gauss rule");
    *gxp = gx.data();
    *gwp = gw.data();
    *ng = static_cast<int>(gx.shape(0));
}
}  // namespace mw568

// Complex J0(x) and J1(x)/x, exposed for direct gating against
// scipy.special.jv. Accepts a 1-D complex array, returns (j0, j1x).
static py::tuple bessel_j0_j1x_complex(
    py::array_t<std::complex<double>,
                py::array::c_style | py::array::forcecast> x) {
    if (x.ndim() != 1) throw std::runtime_error("x must be 1-D");
    const py::ssize_t n = x.shape(0);
    py::array_t<std::complex<double>> j0(n), j1x(n);
    const std::complex<double> *xp = x.data();
    std::complex<double> *j0p = j0.mutable_data();
    std::complex<double> *j1p = j1x.mutable_data();
    {
        py::gil_scoped_release release;
        for (py::ssize_t i = 0; i < n; ++i)
            mw_contour::bessel_j0_j1x(xp[i], j0p[i], j1p[i]);
    }
    return py::make_tuple(j0, j1x);
}

// The Sommerfeld-identity contour: NC = 1. `use_k_m` picks which wavenumber
// the INTEGRAND is built on; the head's `a` and its branch-point marks always
// come from the (k_p, k_m) pair, exactly as `_run_contour` computes them.
static py::tuple contour_engine_sommerfeld_identity(
    double rho, double h, double k_p, std::complex<double> k_m, bool use_k_m,
    double rtol, int depth, double detour,
    py::array_t<double, py::array::c_style | py::array::forcecast> gx,
    py::array_t<double, py::array::c_style | py::array::forcecast> gw,
    int max_panels) {
    const double *gxp;
    const double *gwp;
    int ng;
    mw568::gauss_view(gx, gw, &gxp, &gwp, &ng);
    std::complex<double> val;
    mw_contour::ContourHealth hh;
    {
        py::gil_scoped_release release;
        mw568::SommIdentity f;
        f.rho = rho;
        f.h = h;
        f.k = use_k_m ? k_m : std::complex<double>(k_p, 0.0);
        hh = mw_contour::run_contour<1>(f, k_p, k_m, rho, h, rtol, depth,
                                        detour, gxp, gwp, ng, max_panels, &val);
    }
    return py::make_tuple(val, hh.head_panels, hh.tail_panels, hh.converged,
                          hh.accel);
}

// The NC = 6 synthetic twin. Returns (values (6,), head_panels, tail_panels,
// converged, accel) -- the same five things `_run_contour` returns.
static py::tuple contour_engine_synth6(
    double rho, double h, double k_p, std::complex<double> k_m, double rtol,
    int depth, double detour,
    py::array_t<double, py::array::c_style | py::array::forcecast> gx,
    py::array_t<double, py::array::c_style | py::array::forcecast> gw,
    int max_panels) {
    const double *gxp;
    const double *gwp;
    int ng;
    mw568::gauss_view(gx, gw, &gxp, &gwp, &ng);
    py::array_t<std::complex<double>> out(6);
    std::complex<double> *op = out.mutable_data();
    mw_contour::ContourHealth hh;
    {
        py::gil_scoped_release release;
        mw568::Synth6 f;
        f.rho = rho;
        f.h = h;
        f.k_p = k_p;
        f.k_m = k_m;
        hh = mw_contour::run_contour<6>(f, k_p, k_m, rho, h, rtol, depth,
                                        detour, gxp, gwp, ng, max_panels, op);
    }
    return py::make_tuple(out, hh.head_panels, hh.tail_panels, hh.converged,
                          hh.accel);
}


// --------------------------------------------------------------------------
// momwire#568 unit 2 -- the below/below family ON the shared contour engine.
//
// This is the first PRODUCTION rider on `_contour_engine_inline.h`: the six
// lambda-integrands of `_sommerfeld_below._integrand_six_below`, its
// `_six_integrals_below` driver (fine machine plus the optional coarse
// self-convergence twin), and the projected remainder table
// `remainder_field_proj_below`. The numpy spellings stay exactly as they are
// and remain the references; these are twins, gated at tolerance in
// tests/test_below_fills_568.py.
//
// The OpenMP region is the PER-POINT loop of `iv_surfaces_direct_below`: one
// (rho, h) node's contour is completely independent of every other, the
// engine is allocation-free and has no mutable static (U1's reentrancy
// contract), and a grid fill is thousands of nodes whose costs differ by an
// order of magnitude between the steep and grazing bands -- hence `dynamic`
// scheduling rather than `static`.
//
// STACK. U1 measured ~25 kB per `run_contour` at NC = 6 (a ~10 kB
// `wynn_epsilon` frame plus ~0.6 kB per `adaptive_segment` level at depth 16);
// the fine and coarse machines run in sequence, not nested, so that figure is
// the peak per thread and nothing here relies on a large default. Verified by
// running a 64x64 selfconv fill at OMP_NUM_THREADS=8 under OMP_STACKSIZE of
// 128K, 256K and 512K: all three identical to the last bit, none faulted.
//
// MEASURED on an i7-8550U, one contour node against the numpy driver at
// rtol 1e-9 (soil A, 7 MHz, geometries in lambda_m):
//
//     (rho, h) = (0.3, 0.2)      13.8 ms ->  1.59 ms    8.7x
//                (1.0, 0.05)     47.4 ms ->  4.11 ms   11.5x
//                (2.0, 0.5)      18.9 ms ->  1.32 ms   14.3x
//                (0.02, 0.01)    19.7 ms ->  2.96 ms    6.7x
//                (1.0, 0.017)   113.4 ms ->  9.40 ms   12.1x   (deep grazing)
//
// and end to end, `tests/test_sommerfeld_below.py -m slow -n 2` on this box:
// 392.9 s -> 23.0 s (17.1x), MAXRSS 107 MB -> 108 MB.
// --------------------------------------------------------------------------

namespace mw568_below {
using mw_contour::cd;

static const cd MW_BJ(0.0, 1.0);

// The branch cut: one definition for all three call sites, across both
// extensions — the full rationale (the two-sqrt spelling IS the branch
// choice) lives with the definition in _branch_cut_inline.h (#714). The
// using-declaration makes the name a member of this namespace, so the
// qualified `mw568_below::gamma_cut` calls below still resolve.
using mw_branch::gamma_cut;

// `_sommerfeld._d12(lam, k1, k2)` -- NEC eqs 154-155 (the 2s of eqs 141-142
// are inside). The below family calls it SWAPPED, (k1, k2) = (k_p, k_m), so
// the GROUND sits in the decay slot: `_d12` builds its kernels around
// gamma_2 = gamma(k2) as the decay gamma and k1 as the other medium, and that
// swapped reading IS the below/below D-pair (measured at 0.0 relative
// difference against the #524 phase-0 prototype's independent generalized
// form). `g2` is returned because the integrand needs gamma_m itself.
static inline void d12(const cd &lam, const cd &k1, const cd &k2, cd &d1,
                       cd &d2, cd &g2) {
    const cd g1 = gamma_cut(lam, k1);
    g2 = gamma_cut(lam, k2);
    const cd k1s = k1 * k1;
    const cd k2s = k2 * k2;
    d1 = 2.0 / (g1 + g2) - 2.0 * k2s / (g2 * (k1s + k2s));
    d2 = 2.0 / (k1s * g2 + k2s * g1) - 2.0 / (g2 * (k1s + k2s));
}

// `_integrand_six_below`, term for term and in its order:
//
//   0: d2 V/drho^2      [D2 e^{-gamma_m h} lam^3 (J1/x - J0)]
//   1: d2 V/dz^2        [D2 gamma_m^2 e^{-gamma_m h} J0 lam]
//   2: d2 V/drho dz     [-D2 gamma_m e^{-gamma_m h} J1 lam^2]   <- the +/-=- sign
//   3: (1/rho) dV/drho  [-D2 e^{-gamma_m h} (J1/x) lam^3]
//   4: V                [D2 e^{-gamma_m h} J0 lam]
//   5: U                [D1 e^{-gamma_m h} J0 lam]
//
// INDEX 2 IS THE WHOLE OF THE SIGN STORY and the one line of this file that a
// mutation test exists for. h = |z + z'| with z + z' < 0 below the interface,
// so d/dz e^{-gamma_m|z + z'|} = +gamma_m e where the +/-=+ case gets
// -gamma_2 e; with dJ0/drho = -lam J1 the product lands at -D2 gamma_m J1 lam^2,
// the NEGATIVE of `_sommerfeld._integrand_six`'s index 2. Every other
// component is identical between the families once the kernel pair is the
// swapped one -- which is what `test_gu2_1_the_integrand_z_derivative_flips_
// and_nothing_else_does` asserts on the numpy side and what
// `test_g5686_the_index_2_sign_is_load_bearing` asserts on this one.
//
// Bessel form only: there is no Hankel twin here, because there is no fig-14
// contour here.
struct SixBelow {
    double rho;
    double h;
    cd k_p;
    cd k_m;
    void operator()(const cd &lam, cd *out) const {
        cd d1, d2, g_m;
        d12(lam, k_p, k_m, d1, d2, g_m);
        const cd e = std::exp(-g_m * h);
        const cd x = lam * rho;
        cd b0, b1x;
        mw_contour::bessel_j0_j1x(x, b0, b1x);
        const cd l2 = lam * lam;
        const cd l3 = l2 * lam;
        const cd common = d2 * e;
        out[0] = common * (b1x - b0) * l3;
        out[1] = common * g_m * g_m * b0 * lam;
        out[2] = -(common * g_m * (b1x * x) * l2);
        out[3] = -(common * b1x * l3);
        out[4] = common * b0 * lam;
        out[5] = d1 * e * b0 * lam;
    }
};

// What one node of `_six_integrals_below` produces: the six values plus
// everything `Health.note` / `Health.note_selfconv` are handed on the numpy
// side. `selfconv` is -1.0 when the coarse machine was not asked for, so the
// Python layer can tell "not measured" from "measured zero".
struct SixResult {
    cd val[6];
    int head_panels;
    int tail_panels;
    bool converged;
    bool accel;
    double selfconv;
};

// One node: the fine contour, then optionally the coarse self-convergence
// twin (Gauss-16, rtol x100, a shallower detour -- `_six_integrals_below`'s
// `selfconv=True` branch). The componentwise relative spread is the numpy
// spelling exactly: max over components of |fine - coarse| / max(|fine|, 1e-300).
static void six_below_one(double rho, double h, double k_p, const cd &k_m,
                          double rtol_fine, int depth, double detour,
                          const double *gx, const double *gw, int ng,
                          bool selfconv, double rtol_coarse, int depth_coarse,
                          double detour_coarse, const double *gxc,
                          const double *gwc, int ngc, int max_panels,
                          SixResult &r) {
    SixBelow f;
    f.rho = rho;
    f.h = h;
    f.k_p = cd(k_p, 0.0);
    f.k_m = k_m;
    const mw_contour::ContourHealth hh = mw_contour::run_contour<6>(
        f, k_p, k_m, rho, h, rtol_fine, depth, detour, gx, gw, ng, max_panels,
        r.val);
    r.head_panels = hh.head_panels;
    r.tail_panels = hh.tail_panels;
    r.converged = hh.converged;
    r.accel = hh.accel;
    r.selfconv = -1.0;
    if (selfconv) {
        cd coarse[6];
        mw_contour::run_contour<6>(f, k_p, k_m, rho, h, rtol_coarse,
                                   depth_coarse, detour_coarse, gxc, gwc, ngc,
                                   max_panels, coarse);
        double worst = 0.0;
        for (int c = 0; c < 6; ++c) {
            const double scale = std::max(std::abs(r.val[c]), 1e-300);
            const double rel = std::abs(r.val[c] - coarse[c]) / scale;
            if (rel > worst) worst = rel;
        }
        r.selfconv = worst;
    }
}

// The below/below twin of `somm_proj::proj_one` (see the comment on
// `remainder_field_proj_batch_below` for why it is a twin and not a widening).
// Same 4x4 Lagrange stencil, same eqs 143-147 dyad algebra; three things are
// the below family's own:
//
//   * `hh` is the two DEPTHS added, (ground_z - oz) + (ground_z - sz), so a
//     wrong-side endpoint cannot quietly produce a plausible h (the Python
//     layer raises on it before this is ever reached);
//   * `g` is `divide_out_below`'s two-leg blend e^{-j(k_p rho + k_m hh)}/R1,
//     which needs a COMPLEX in-medium wavenumber -- the exact reason
//     `remainder_field_proj_below` could not ride the `double k` kernel;
//   * theta is clamped into [th_min, pi/2], the below grid's own grazing
//     floor, not into [0, pi/2].
//
// The query's (R1, theta) are handed back so the caller can let
// `SommerfeldGridBelow.eval` raise the refusals in its own words; nothing in
// this file transcribes those messages.
static inline cd proj_one_below(const somm_proj::GridView &G, double th_min,
                                double th_band_hi,
                                double ground_z, double k_p, const cd &k_m,
                                double ox, double oy, double oz, double tox,
                                double toy, double toz, double sx, double sy,
                                double sz, double sux, double suy,
                                double sthsrc, double stzsrc, double &r1_out,
                                double &th_out) {
    const double dx = ox - sx;
    const double dy = oy - sy;
    const double rho = std::hypot(dx, dy);
    const double hh = (ground_z - oz) + (ground_z - sz);
    const double r1 = std::sqrt(rho * rho + hh * hh);

    // --- inline SommerfeldGridBelow.eval(r1, theta) ---
    double theta = std::atan2(hh, rho);
    r1_out = r1;
    th_out = theta;
    if (theta < th_min) theta = th_min;
    else if (theta > G.half_pi) theta = G.half_pi;
    const double r1c = r1 > G.r1_max ? G.r1_max : r1;
    // THREE theta bands per R1 zone since momwire#838, ordered (R1 zone) x
    // (theta band) with theta fastest -- the layout SommerfeldGridBelow's
    // constructor builds. NOT the parent's 2-band/3-zone scheme: six regions
    // there means the momwire#159 far zone, which this family does not have
    // (r_near == r1_max), so `th_band_hi` is passed in explicitly rather than
    // inferred from reg_vals.size().
    //
    // STRICT `<` at the band edge, matching `SommerfeldGridBelow._interp`:
    // theta == th_band_hi belongs to the OLD grazing band. The two bands'
    // node at th_band_hi is the same fill, but `th_min + dth*n` need not
    // reproduce it to the last bit.
    const int band = theta < th_band_hi ? 0 : (theta <= G.th_split ? 1 : 2);
    const int reg = (r1c <= G.r_break ? 0 : 3) + band;
    const double fr = (r1c - G.rr0[reg]) / G.rdr[reg];
    const double ft = (theta - G.rth0[reg]) / G.rdth[reg];
    int i0 = (int)std::floor(fr) - 1;
    int j0 = (int)std::floor(ft) - 1;
    if (i0 < 0) i0 = 0; else if (i0 > G.nR[reg] - 4) i0 = (int)G.nR[reg] - 4;
    if (j0 < 0) j0 = 0; else if (j0 > G.nTh[reg] - 4) j0 = (int)G.nTh[reg] - 4;
    double wr[4], wt[4];
    somm_proj::lagrange4(fr - i0, wr);
    somm_proj::lagrange4(ft - j0, wt);
    const cd *V = G.vptr[reg];
    const py::ssize_t nth = G.nTh[reg], nr = G.nR[reg];
    cd surf[4];
    for (int s = 0; s < 4; ++s) {
        const cd *plane = V + (py::ssize_t)s * nr * nth;
        cd acc(0.0, 0.0);
        for (int i = 0; i < 4; ++i) {
            const cd *row = plane + (py::ssize_t)(i0 + i) * nth + j0;
            cd rs = row[0] * wt[0] + row[1] * wt[1] + row[2] * wt[2] +
                    row[3] * wt[3];
            acc += rs * wr[i];
        }
        surf[s] = acc;
    }
    const cd IrhoV = surf[0], IzV = surf[1], IrhoH = surf[2], IphiH = surf[3];

    // --- projection (eqs 143-147), over `divide_out_below`'s g ---
    const cd g = std::exp(-MW_BJ * (cd(k_p * rho, 0.0) + k_m * hh)) / r1;
    const bool safe_r = rho > G.tiny;
    const double inv_rho = safe_r ? 1.0 / rho : 0.0;
    const double dhx = safe_r ? dx * inv_rho : sux;
    const double dhy = safe_r ? dy * inv_rho : suy;
    const double cphi = sux * dhx + suy * dhy;
    const double sphi = sux * dhy - suy * dhx;
    const cd e_rho = g * (stzsrc * IrhoV + sthsrc * cphi * IrhoH);
    const cd e_phi = g * (sthsrc * sphi * IphiH);
    const cd e_z = g * (stzsrc * IzV - sthsrc * cphi * IrhoV);
    return tox * (dhx * e_rho - dhy * e_phi) +
           toy * (dhy * e_rho + dhx * e_phi) + toz * e_z;
}
}  // namespace mw568_below

// `_six_integrals_below` over parallel (rho, h) arrays: the (n, 6) table plus
// everything `Health` records, OpenMP across nodes with the GIL released.
//
// The wavenumbers arrive DERIVED (k_p real, k_m complex on the Im <= 0 branch)
// rather than as eps~, so the branch choice stays in `k_medium` where it is
// written down once; the eps~ == 1 short circuit and the (rho, h) domain
// raise likewise stay in Python, because they are exact and their words are
// part of the contract.
//
// Returns (values (n, 6), tail_panels (n,), head_panels (n,), converged (n,),
// accelerated (n,), selfconv (n,)) -- selfconv is -1.0 where the coarse
// machine was not run.
static py::tuple below_six_integrals_batch(
    double k_p, std::complex<double> k_m,
    py::array_t<double, py::array::c_style | py::array::forcecast> rho,
    py::array_t<double, py::array::c_style | py::array::forcecast> h,
    double rtol_fine, int depth, double detour,
    py::array_t<double, py::array::c_style | py::array::forcecast> gx,
    py::array_t<double, py::array::c_style | py::array::forcecast> gw,
    bool selfconv, double rtol_coarse, int depth_coarse, double detour_coarse,
    py::array_t<double, py::array::c_style | py::array::forcecast> gxc,
    py::array_t<double, py::array::c_style | py::array::forcecast> gwc,
    int max_panels) {
    auto rb = rho.unchecked<1>();
    auto hb = h.unchecked<1>();
    const py::ssize_t n = rb.shape(0);
    if (hb.shape(0) != n)
        throw std::invalid_argument("rho and h must have the same length");
    const double *gxp, *gwp, *gxcp, *gwcp;
    int ng, ngc;
    mw568::gauss_view(gx, gw, &gxp, &gwp, &ng);
    mw568::gauss_view(gxc, gwc, &gxcp, &gwcp, &ngc);

    py::array_t<std::complex<double>> vals({n, py::ssize_t(6)});
    py::array_t<int> tail(n), head(n);
    py::array_t<bool> conv(n), accel(n);
    py::array_t<double> sconv(n);
    auto vb = vals.mutable_unchecked<2>();
    int *tp = tail.mutable_data();
    int *hp = head.mutable_data();
    bool *cp = conv.mutable_data();
    bool *ap = accel.mutable_data();
    double *sp = sconv.mutable_data();
    const mw_contour::cd km(k_m);

    {
        py::gil_scoped_release release;
        // `dynamic`: a grazing node (theta ~ 1 deg) costs an order of
        // magnitude more tail panels than a steep one, and a grid fill's
        // nodes arrive sorted by region -- static scheduling would hand one
        // thread the whole grazing band.
        #pragma omp parallel for schedule(dynamic)
        for (py::ssize_t i = 0; i < n; ++i) {
            mw568_below::SixResult r;
            mw568_below::six_below_one(
                rb(i), hb(i), k_p, km, rtol_fine, depth, detour, gxp, gwp, ng,
                selfconv, rtol_coarse, depth_coarse, detour_coarse, gxcp, gwcp,
                ngc, max_panels, r);
            for (int c = 0; c < 6; ++c) vb(i, c) = r.val[c];
            tp[i] = r.tail_panels;
            hp[i] = r.head_panels;
            cp[i] = r.converged;
            ap[i] = r.accel;
            sp[i] = r.selfconv;
        }
    }
    return py::make_tuple(vals, tail, head, conv, accel, sconv);
}

// `remainder_field_proj_below` in C++: obs/t_obs (M,3), src/t_src (S,3),
// returns ((M,S) complex, max R1, min theta, max theta).
//
// A TWIN of `remainder_field_proj_batch`, not a widening of it, and that was a
// decision rather than an accident. Widening the shipped kernel to carry a
// complex wavenumber would have put a branch (or a second carrier expression)
// inside the +/-=+ family's hottest inner loop -- ~90 % of an above/above
// Sommerfeld solve -- for a family that does not need it, and the two carriers
// are not the same expression even when k_m happens to be real: above/above
// divides out e^{-jk R1}/R1, a function of R1 alone, while the below family
// divides out the two-leg blend e^{-j(k_p rho + k_m hh)}/R1, which depends on
// theta. The +/-=+ path therefore keeps its bytes, literally: not one token of
// `proj_one` moved.
//
// The three refusals (`R1` past the tabulation, theta under the grazing floor,
// theta past pi/2) are NOT transcribed here. The kernel reports the query's
// extremes and the Python layer feeds them straight back to
// `SommerfeldGridBelow.eval`, which raises in its own words -- so there is
// exactly one copy of that prose and no way for the two paths to drift on
// which geometries they serve.
static py::tuple remainder_field_proj_batch_below(
    py::array_t<double, py::array::c_style | py::array::forcecast> obs,
    py::array_t<double, py::array::c_style | py::array::forcecast> t_obs,
    py::array_t<double, py::array::c_style | py::array::forcecast> src,
    py::array_t<double, py::array::c_style | py::array::forcecast> t_src,
    double ground_z, double k_p, std::complex<double> k_m, double th_min,
    double th_band_hi,
    double r1_max, double r_break, double th_split, double r_near,
    py::array_t<double, py::array::c_style | py::array::forcecast> reg_r0,
    py::array_t<double, py::array::c_style | py::array::forcecast> reg_dr,
    py::array_t<double, py::array::c_style | py::array::forcecast> reg_th0,
    py::array_t<double, py::array::c_style | py::array::forcecast> reg_dth,
    std::vector<py::array_t<std::complex<double>,
                            py::array::c_style | py::array::forcecast>> reg_vals) {
    using somm_proj::cd;
    auto ob = obs.unchecked<2>();
    auto tob = t_obs.unchecked<2>();
    auto sb = src.unchecked<2>();
    auto tsb = t_src.unchecked<2>();
    if (ob.shape(1) != 3 || tob.shape(1) != 3 || sb.shape(1) != 3 ||
        tsb.shape(1) != 3)
        throw std::runtime_error("obs/src/tangent arrays must have shape (*, 3)");
    if (ob.shape(0) != tob.shape(0) || sb.shape(0) != tsb.shape(0))
        throw std::runtime_error("points and tangents must have matching length");

    const py::ssize_t M = ob.shape(0);
    const py::ssize_t S = sb.shape(0);
    // The below family is SIX regions exactly (2 R1 zones x 3 theta bands)
    // since momwire#838. A four-region grid here is a stale
    // momwire/_sommerfeld_below.py and would route into unpopulated tables.
    if (reg_vals.size() != 6)
        throw std::runtime_error(
            "the below/below grid is six regions (2 R1 zones x 3 theta bands) "
            "since momwire#838; got a different count, which means a stale "
            "_sommerfeld_below.py");
    somm_proj::GridView G = somm_proj::build_grid_view(
        r1_max, r_break, th_split, r_near, reg_r0.unchecked<1>(),
        reg_dr.unchecked<1>(), reg_th0.unchecked<1>(), reg_dth.unchecked<1>(),
        reg_vals);

    py::array_t<std::complex<double>> out({M, S});
    auto out_m = out.mutable_unchecked<2>();
    const cd km(k_m);

    // Per-observer-row extremes, reduced serially afterwards: a `reduction`
    // clause on min/max is OpenMP 3.1 and this file stays portable to MSVC's
    // classic /openmp.
    std::vector<double> row_r1(M > 0 ? M : 1, 0.0);
    std::vector<double> row_thlo(M > 0 ? M : 1, 0.5 * M_PI);
    std::vector<double> row_thhi(M > 0 ? M : 1, 0.0);

    {
        py::gil_scoped_release release;
        std::vector<double> sx(S), sy(S), sz(S), ux(S), uy(S), thsrc(S), tzsrc(S);
        for (py::ssize_t nn = 0; nn < S; ++nn) {
            sx[nn] = sb(nn, 0);
            sy[nn] = sb(nn, 1);
            sz[nn] = sb(nn, 2);
            somm_proj::tangent_decomp(tsb(nn, 0), tsb(nn, 1), tsb(nn, 2), ux[nn],
                                      uy[nn], thsrc[nn], tzsrc[nn]);
        }

        #pragma omp parallel for schedule(static)
        for (py::ssize_t m = 0; m < M; ++m) {
            const double ox = ob(m, 0), oy = ob(m, 1), oz = ob(m, 2);
            const double tox = tob(m, 0), toy = tob(m, 1), toz = tob(m, 2);
            double rmax = 0.0, tlo = 0.5 * M_PI, thi = 0.0;
            for (py::ssize_t nn = 0; nn < S; ++nn) {
                double r1q, thq;
                out_m(m, nn) = mw568_below::proj_one_below(
                    G, th_min, th_band_hi, ground_z, k_p, km, ox, oy, oz, tox,
                    toy, toz,
                    sx[nn], sy[nn], sz[nn], ux[nn], uy[nn], thsrc[nn], tzsrc[nn],
                    r1q, thq);
                if (r1q > rmax) rmax = r1q;
                if (thq < tlo) tlo = thq;
                if (thq > thi) thi = thq;
            }
            row_r1[m] = rmax;
            row_thlo[m] = tlo;
            row_thhi[m] = thi;
        }
    }

    double mx_r1 = 0.0, mn_th = 0.5 * M_PI, mx_th = 0.0;
    for (py::ssize_t m = 0; m < M; ++m) {
        if (row_r1[m] > mx_r1) mx_r1 = row_r1[m];
        if (row_thlo[m] < mn_th) mn_th = row_thlo[m];
        if (row_thhi[m] > mx_th) mx_th = row_thhi[m];
    }
    return py::make_tuple(out, mx_r1, mn_th, mx_th);
}


// --------------------------------------------------------------------------
// momwire#568 unit 3 -- the TRANSMITTED family on the shared contour engine.
//
// The second production rider on `_contour_engine_inline.h`, and the arc's
// biggest cluster: the six lambda-integrands of
// `_sommerfeld_transmitted._integrand_six_transmitted`, its
// `_six_integrals_transmitted` driver (fine machine plus the optional coarse
// self-convergence twin), the per-point loop of `t_surfaces_direct` -- THIS is
// where the OpenMP region lives -- and the projected pair table behind
// `transmitted_field_proj_below_to_above` / `_above_to_below`. The numpy
// spellings stay exactly as they are and remain the references; these are
// twins, gated at tolerance in tests/test_transmitted_fills_568.py.
//
// FOUR THINGS ARE THIS FAMILY'S OWN and none of them is cosmetic:
//
//   * TWO gammas, not the below family's single swapped `_d12` pair. The
//     transmitted integrand's two exponential legs sit in DIFFERENT media, so
//     `gamma_cut` is called at k_p and at k_m and both survive into the
//     Fresnel denominators k_m^2 gamma_p + k_p^2 gamma_m and gamma_m + gamma_p.
//   * Index 1 is spelled `lam^2`, NEVER `gamma_p^2 + k_p^2`. See the comment
//     on it below; it is a measured eleven-digit cancellation, not a taste.
//   * Index 4 (d2 V_T/drho dz') is why the family has FIVE surfaces. The two
//     legs carry different gamma, so d/dz' is not -d/dz and index 4 is not a
//     multiple of index 0. It must not collapse onto index 0.
//   * `swap` builds the above->below twin (source ABOVE carrying gamma_p,
//     observer BELOW carrying gamma_m). It exists to GATE the reciprocity
//     transpose; the product path serves above->below as a transpose over the
//     same tables and never asks for it.
//
// THE TAIL BUDGET IS A CLIFF, NOT A SOFT LANDING. The transmitted tail must
// reach lam ~ 35/h, panelled on the J0(lam rho) zero lattice pi/rho, so it
// costs ~12 cot(theta_true) panels; `_MAX_TAIL_PANELS_T` is 6000 and the
// grazing floor is SOLVED from that law at 16 panels per cot. When the budget
// runs out `tail_below` falls back to Wynn epsilon, and on this family that
// fallback was measured 4.5e+3 RELATIVE wrong -- three and a half decades,
// silently, with `converged = False` as the only tell. So this block reports
// `converged` per node exactly as the numpy driver does and the Python layer
// tallies it into `Health.nonconvergent`, which the fill gates assert at zero.
// A C++ path that ever returned an unconverged answer unflagged would be the
// worst defect available in this unit.
//
// The cot-theta cost law is KEPT. The lateral-wave asymptotic branch that
// would delete it changes the cost/serve model, and #568 promises answer
// neutrality: each panel gets cheaper, the panel COUNT does not move.
//
// STACK. Same envelope U2 measured: ~25 kB per `run_contour` at NC = 6, the
// fine and coarse machines in sequence rather than nested. Nothing here needs
// a large default; keep `OMP_STACKSIZE` at or above ~256 kB. Verified by
// filling a 1085-node grid at OMP_NUM_THREADS of 1, 4 and 8 under
// OMP_STACKSIZE of 128K, 256K and 512K: all nine runs BIT-IDENTICAL (same
// SHA-256 over the value table) and none faulted. The thread count buys 4.2x
// on top of the per-node figure below (0.63 s -> 0.15 s at 8 threads) wherever
// threads are available; the test suite pins its xdist workers to
// OMP_NUM_THREADS=1, so the slow-lane numbers below are the per-node speedup
// alone with none of that in them.
//
// MEASURED on an i7-8550U at OMP_NUM_THREADS=1 (the suite pins its xdist
// workers there), one contour node against the numpy driver at rtol 1e-9,
// soil A / 7 MHz, |z'| = 0.15 m unless noted:
//
//     (R/lam_p, theta) = (0.01,  45 deg)    15.6 ms ->  0.85 ms   18.4x
//                        (0.5,    5 deg)    31.9 ms ->  1.91 ms   16.7x
//                        (2.0,   30 deg)     5.9 ms ->  0.55 ms   10.7x
//                        (2.0,  0.1 deg)   747.6 ms -> 41.0  ms   18.2x  (grazing)
//                        (2.0, 0.02 deg)  1511.8 ms -> 76.9  ms   19.7x  (past the
//                                                                  panel budget)
//
// The grazing end is the BEST case, which is the useful shape: cost there is
// almost all tail panels and a tail panel is almost all Bessel, where U1's
// real-abscissa kernels are furthest ahead of scipy's complex ones.
//
// End to end, `tests/test_sommerfeld_transmitted.py -m slow -n 2` on this box:
// 446.0 s -> 30.3 s (14.7x), MAXRSS 116 MB -> 116 MB. The residual is now
// dominated by `test_gu3_10_a_truncated_tail_is_not_a_graceful_degradation`
// (14.3 s of the 30.3), which drives `_tail_below` directly at 80,000 panels
// and never reaches this code at all.
// --------------------------------------------------------------------------

namespace mw568_trans {
using mw_contour::cd;

// `_sommerfeld._gamma` is SHARED with U2 rather than re-transcribed
// (`mw568_below::gamma_cut` — since #714 the definition itself lives in
// _branch_cut_inline.h, one copy for both extensions): it is the two-sqrt
// product
// sqrt(-j(lam - k)) sqrt(j(lam + k)), and the spelling IS the branch choice --
// vertical cuts running DOWN from +k and UP from -k, neither of which the
// head's first-quadrant detour crosses. Two copies of that could drift; one
// cannot. This family calls it TWICE per evaluation, at k_p and at k_m,
// because its two exponential legs live in different media.

// `_integrand_six_transmitted`, term for term and in its order:
//
//   0: d2 V_T/drho dz     [a (-lam J1) d_z]
//   1: (d2/dz2 + k_p^2)V_T [a lam^2 J0]
//   2: d2 V_T/drho^2      [a lam^2 (J1/x - J0)]
//   3: (1/rho) dV_T/drho  [-a lam^2 (J1/x)]
//   4: d2 V_T/drho dz'    [a (-lam J1) d_zp]
//   5: U_T                [u J0]
//
// with a = 2 lam e / (k_m^2 gamma_p + k_p^2 gamma_m) the (7f) V_T kernel and
// u = 2 lam e / (gamma_m + gamma_p) the (7g) U_T one, e the two-leg
// exponential. All five derivatives are ANALYTIC under the integral sign
// (d/drho J0 = -lam J1, d2/drho2 via the Bessel ODE identity, d/dz -> x(-g_p),
// d/dz' -> x(+g_m)) -- never a differenced interpolant, which is the
// LLNL-TR-490316 rule the phase-0 comment records.
//
// INDEX 1 IS SPELLED lam^2 AND THAT IS LOAD-BEARING. (d2/dz2 + k^2) is
// gamma^2 + k^2 in either medium, and gamma^2 + k^2 is a difference of two
// O(k^2) numbers whose result is O(lam^2): at the bottom of the contour
// (lam ~ 1e-3, |k_m| ~ 0.6) it cancels away ELEVEN digits. The swapped and
// unswapped spellings of the same quantity were measured 3.5e-11 apart before
// the numpy line said lam^2. Writing `g_p*g_p + k_p*k_p` here would not fail
// loudly; it would quietly cost a grazing gate. `test_g56812_index_1_is_
// lambda_squared_not_gamma_squared_plus_k_squared` is the mutation gate.
//
// INDEX 4 IS THE FIFTH SURFACE. d/dz' multiplies by +gamma_m where d/dz
// multiplies by -gamma_p, and the two are different functions of lam, so
// index 4 is NOT a multiple of index 0 and T_z^H is not -cos(phi) T_rho^V.
// That collapse is a +-=+ identity that does not survive here; the #553 arc
// gated it hardest because the licensed engine departs from the phase-0
// prototype on exactly this component by O(1), with empymod siding with the
// prototype.
struct SixTransmitted {
    double rho;
    double z;
    double zp;
    double k_p;
    cd k_m;
    bool swap;
    void operator()(const cd &lam, cd *out) const {
        const cd kp(k_p, 0.0);
        const cd g_p = mw568_below::gamma_cut(lam, kp);
        const cd g_m = mw568_below::gamma_cut(lam, k_m);
        const double azp = std::fabs(zp);
        cd e, d_z, d_zp;
        if (swap) {
            // Source ABOVE at z' > 0 carrying gamma_p, observer BELOW at
            // z < 0 carrying gamma_m: the same integral with the legs
            // exchanged, which is the reciprocity identity phase 0 measured
            // at 0.0 relative.
            e = 2.0 * std::exp(-g_p * azp - g_m * std::fabs(z)) * lam;
            d_z = g_m;
            d_zp = -g_p;
        } else {
            e = 2.0 * std::exp(-g_m * azp - g_p * z) * lam;
            d_z = -g_p;
            d_zp = g_m;
        }
        const cd zzk = lam * lam;  // NOT g^2 + k^2 -- see the note above
        const double kp2 = k_p * k_p;
        const cd a = e / (k_m * k_m * g_p + kp2 * g_m);  // (7f)
        const cd u = e / (g_m + g_p);                    // (7g)
        const cd x = lam * rho;
        cd b0, b1x;
        mw_contour::bessel_j0_j1x(x, b0, b1x);
        const cd j1 = b1x * x;
        const cd dr = -(lam * j1);  // d/drho J0 = -lam J1
        out[0] = a * dr * d_z;
        out[1] = a * zzk * b0;
        out[2] = a * lam * lam * (b1x - b0);
        out[3] = a * (-(lam * lam * b1x));
        out[4] = a * dr * d_zp;
        out[5] = u * b0;
    }
};

// What one node of `_six_integrals_transmitted` produces: the six values plus
// everything `Health.note` / `Health.note_selfconv` are handed on the numpy
// side. `selfconv` is -1.0 when the coarse machine was not asked for, so the
// Python layer can tell "not measured" from "measured zero".
struct SixResultT {
    cd val[6];
    int head_panels;
    int tail_panels;
    bool converged;
    bool accel;
    double selfconv;
};

// One node: the fine contour, then optionally the coarse self-convergence
// twin (Gauss-16, rtol x100, a shallower detour).
//
// `h_decay = |z'| + |z|` is the TAIL'S DECAY LENGTH, not a coordinate: both
// gammas go to lam at large lam, so e^{-g_m|z'| - g_p z} -> e^{-lam(|z'|+z)}
// exactly as the below family's e^{-g_m h} -> e^{-lam h}. That one argument is
// the whole of the sharing with U2's contour, and `max_panels` is the other
// handle -- this family's grazing rows need 6000 where any below/below
// geometry is done in a few hundred.
static void six_transmitted_one(double rho, double z, double zp, double k_p,
                                const cd &k_m, bool swap, double rtol_fine,
                                int depth, double detour, const double *gx,
                                const double *gw, int ng, bool selfconv,
                                double rtol_coarse, int depth_coarse,
                                double detour_coarse, const double *gxc,
                                const double *gwc, int ngc, int max_panels,
                                SixResultT &r) {
    SixTransmitted f;
    f.rho = rho;
    f.z = z;
    f.zp = zp;
    f.k_p = k_p;
    f.k_m = k_m;
    f.swap = swap;
    const double h_decay = std::fabs(zp) + std::fabs(z);
    const mw_contour::ContourHealth hh = mw_contour::run_contour<6>(
        f, k_p, k_m, rho, h_decay, rtol_fine, depth, detour, gx, gw, ng,
        max_panels, r.val);
    r.head_panels = hh.head_panels;
    r.tail_panels = hh.tail_panels;
    r.converged = hh.converged;
    r.accel = hh.accel;
    r.selfconv = -1.0;
    if (selfconv) {
        cd coarse[6];
        mw_contour::run_contour<6>(f, k_p, k_m, rho, h_decay, rtol_coarse,
                                   depth_coarse, detour_coarse, gxc, gwc, ngc,
                                   max_panels, coarse);
        double worst = 0.0;
        for (int c = 0; c < 6; ++c) {
            const double scale = std::max(std::abs(r.val[c]), 1e-300);
            const double rel = std::abs(r.val[c] - coarse[c]) / scale;
            if (rel > worst) worst = rel;
        }
        r.selfconv = worst;
    }
}

// `TransmittedGrid`'s tabulation, flattened for the inner loop.
//
// NOT `somm_proj::GridView`, and not a widening of it. That view is built for
// the +-=+ / +-=- REMAINDER grids: four surfaces, four-or-six (R1, theta)
// REGIONS selected by (r_break, th_split, r_near), and a LINEAR radial axis
// per region. The transmitted tabulation is one uniform lattice in
// (ln R, theta) -- logarithmic, because the transmitted surface is the whole
// field and carries the source's own 1/R^2 near zone, so there is no finite
// R1 -> 0 limit node the way the remainder families have -- with FIVE surfaces
// and a third axis, the log|z'| ladder. Bending `build_grid_view` around a
// different region model, a different axis law, a different surface count and
// an extra dimension would have made one struct serve two unrelated layouts;
// this is the small honest one.
struct TGridView {
    const cd *vals;  // (n_zp, 5, n_r, n_th), C-contiguous
    py::ssize_t n_zp, n_r, n_th;
    double lnr0, dlnr, th0, dth, lnz0, dlnz;
    double tiny, half_pi;
};

// `TransmittedGrid.eval`'s interpolation, WITHOUT its four refusals: cubic
// Lagrange in ln R and theta, and -- when the ladder has more than one rung --
// cubic in ln|z'| on the divided-out surfaces, which is phase 0's measured
// scheme. The refusals stay in Python; see the comment on
// `transmitted_field_proj_batch`.
static inline void tgrid_eval(const TGridView &G, double R, double theta,
                              double azp, cd *surf) {
    const double fr = (std::log(std::max(R, 1e-300)) - G.lnr0) / G.dlnr;
    int i0 = (int)std::floor(fr) - 1;
    if (i0 < 0) i0 = 0;
    else if (i0 > (int)G.n_r - 4) i0 = (int)G.n_r - 4;
    double thc = theta;
    if (thc < 0.0) thc = 0.0;
    else if (thc > G.half_pi) thc = G.half_pi;
    const double ft = (thc - G.th0) / G.dth;
    int j0 = (int)std::floor(ft) - 1;
    if (j0 < 0) j0 = 0;
    else if (j0 > (int)G.n_th - 4) j0 = (int)G.n_th - 4;
    double wr[4], wt[4];
    somm_proj::lagrange4(fr - i0, wr);
    somm_proj::lagrange4(ft - j0, wt);

    const py::ssize_t nth = G.n_th, nr = G.n_r;
    const py::ssize_t plane = nr * nth;       // one surface at one rung
    const py::ssize_t rung = 5 * plane;       // all five at one rung

    if (G.n_zp == 1) {
        for (int s = 0; s < 5; ++s) {
            const cd *V = G.vals + (py::ssize_t)s * plane;
            cd acc(0.0, 0.0);
            for (int i = 0; i < 4; ++i) {
                const cd *row = V + (py::ssize_t)(i0 + i) * nth + j0;
                const cd rs = row[0] * wt[0] + row[1] * wt[1] + row[2] * wt[2] +
                              row[3] * wt[3];
                acc += rs * wr[i];
            }
            surf[s] = acc;
        }
        return;
    }

    const double fz = (std::log(std::max(azp, 1e-300)) - G.lnz0) / G.dlnz;
    int k0 = (int)std::floor(fz) - 1;
    if (k0 < 0) k0 = 0;
    else if (k0 > (int)G.n_zp - 4) k0 = (int)G.n_zp - 4;
    double wz[4];
    somm_proj::lagrange4(fz - k0, wz);
    for (int s = 0; s < 5; ++s) {
        cd acc(0.0, 0.0);
        for (int k = 0; k < 4; ++k) {
            const cd *V = G.vals + (py::ssize_t)(k0 + k) * rung +
                          (py::ssize_t)s * plane;
            cd sub(0.0, 0.0);
            for (int i = 0; i < 4; ++i) {
                const cd *row = V + (py::ssize_t)(i0 + i) * nth + j0;
                const cd rs = row[0] * wt[0] + row[1] * wt[1] + row[2] * wt[2] +
                              row[3] * wt[3];
                sub += rs * wr[i];
            }
            acc += sub * wz[k];
        }
        surf[s] = acc;
    }
}

// One (above, below) pair: `_crossing_geometry` + `TransmittedGrid.eval` +
// `divide_out_transmitted` + `_combine_transmitted_proj`, with no
// intermediates.
//
// The geometry is ALWAYS built from the above point and the below point, in
// that role order, whichever way the field is travelling: R and theta are the
// ABOVE point's polar coordinates about the below point's ground projection,
// |z'| is the below point's depth, and (dhx, dhy) is the horizontal direction
// from the BELOW point to the ABOVE one in BOTH directions of travel. That
// last one is the single silent sign error available in this family -- the
// surfaces are tabulated with the below point as the source and sin(phi) is
// ODD, so reading d-hat from source-minus-observer would flip T_phi^H on one
// of the two directions only.
//
// `transposed` swaps T_rho^V with T_z^H and NOTHING else. That is the entire
// content of "above->below is the reciprocity transpose": the dyad's 2x2
// horizontal block is already symmetric, so transposing exchanges exactly the
// vertical source's radial row with the horizontal source's vertical row --
// which is the FIFTH surface, and why a four-surface family could not have
// served both directions. E_z's horizontal row therefore reads T_z^H (or
// T_rho^V when transposed) and never -cos(phi) T_rho^V; that collapse is a
// +-=+ identity and it does not hold here.
//
// `to`/`ts` are the OBSERVER's and SOURCE's tangents, so the caller supplies
// them in travel order while the geometry stays in role order. The query's
// (R, theta, |z'|) go back to the caller so `TransmittedGrid.eval` can raise
// the four refusals in its own words; nothing here transcribes that prose.
static inline cd proj_one_transmitted(const TGridView &G, double ground_z,
                                      double k_p, const cd &k_m,
                                      double ax, double ay, double az,
                                      double bx, double by, double bz,
                                      const double *to, const double *ts,
                                      bool transposed, double &r_out,
                                      double &th_out, double &zp_out) {
    const double z = az - ground_z;         // the above point's height, >= 0
    const double depth = ground_z - bz;     // the below point's depth, > 0
    const double dx = ax - bx;
    const double dy = ay - by;
    const double rho = std::hypot(dx, dy);
    const double r_obs = std::hypot(rho, z);
    const double theta = std::atan2(z, rho);
    r_out = r_obs;
    th_out = theta;
    zp_out = depth;

    cd surf[5];
    tgrid_eval(G, r_obs, theta, depth, surf);
    const cd TrhoV = surf[0], TzV = surf[1], TrhoH = surf[2], TphiH = surf[3],
             TzH = surf[4];

    // `divide_out_transmitted`: two legs, because the transmitted ray has two
    // -- |z'| straight up through the ground at k_m, then the rest of the way
    // through the AIR at the real k_p over the true separation R, with the 1/R
    // spherical spreading every point-source field carries.
    const double dz = z + depth;
    const double rr = std::sqrt(rho * rho + dz * dz);
    const cd g = std::exp(-mw568_below::MW_BJ * (k_m * depth + cd(k_p * rr, 0.0))) / rr;

    const bool safe_r = rho > G.tiny;
    const double inv_rho = safe_r ? 1.0 / rho : 0.0;
    const double dhx = safe_r ? dx * inv_rho : 1.0;
    const double dhy = safe_r ? dy * inv_rho : 0.0;
    const double cph = ts[0] * dhx + ts[1] * dhy;
    const double sph = ts[0] * dhy - ts[1] * dhx;
    const double pz = ts[2];

    cd e_rho, e_z;
    if (transposed) {
        e_rho = g * (pz * TzH + cph * TrhoH);
        e_z = g * (pz * TzV + cph * TrhoV);
    } else {
        e_rho = g * (pz * TrhoV + cph * TrhoH);
        e_z = g * (pz * TzV + cph * TzH);
    }
    const cd e_phi = g * (sph * TphiH);
    return to[0] * (dhx * e_rho - dhy * e_phi) +
           to[1] * (dhy * e_rho + dhx * e_phi) + to[2] * e_z;
}
}  // namespace mw568_trans

// `_integrand_six_transmitted` POINTWISE at a list of complex lambda: (n, 6).
//
// Not on any product path -- the integrand is only ever called from inside the
// engine. It exists because the two mutations this unit is most exposed to are
// invisible to an INTEGRATED gate:
//
//   * index 1 spelled `g_p*g_p + k_p*k_p` instead of `lam*lam` moves the
//     integrand by ~1e-5 relative at the bottom of the contour (lam ~ 1e-3,
//     |k_m| ~ 0.6, eleven digits of cancellation) and the integral by ~3.5e-11
//     -- under any parity pin an integrated gate can honestly carry;
//   * index 4 collapsed onto index 0 is a whole surface, but the two agree
//     wherever gamma_m happens to track gamma_p, so a gate has to be READ at
//     lambda where they do not.
//
// Both are named mutation gates in tests/test_transmitted_fills_568.py, and
// both read this entry point at chosen lambda rather than hoping an integral
// notices.
static py::array_t<std::complex<double>> transmitted_integrand_six(
    py::array_t<std::complex<double>,
                py::array::c_style | py::array::forcecast> lam,
    double rho, double z, double zp, double k_p, std::complex<double> k_m,
    bool swap) {
    if (lam.ndim() != 1) throw std::runtime_error("lam must be 1-D");
    const py::ssize_t n = lam.shape(0);
    py::array_t<std::complex<double>> out({n, py::ssize_t(6)});
    auto ob = out.mutable_unchecked<2>();
    const std::complex<double> *lp = lam.data();
    {
        py::gil_scoped_release release;
        mw568_trans::SixTransmitted f;
        f.rho = rho;
        f.z = z;
        f.zp = zp;
        f.k_p = k_p;
        f.k_m = k_m;
        f.swap = swap;
        for (py::ssize_t i = 0; i < n; ++i) {
            mw_contour::cd v[6];
            f(lp[i], v);
            for (int c = 0; c < 6; ++c) ob(i, c) = v[c];
        }
    }
    return out;
}

// `_six_integrals_transmitted` over parallel (rho, z, z') arrays: the (n, 6)
// table plus everything `Health` records, OpenMP across nodes with the GIL
// released. This is `t_surfaces_direct`'s per-point loop -- the fill's hot
// loop and the arc's biggest single cluster.
//
// The wavenumbers arrive DERIVED (k_p real, k_m complex on the Im <= 0 branch)
// rather than as eps~, so the branch choice stays in `k_medium` where it is
// written down once; the side-of-interface validation -- which RAISES BY NAME,
// and whose asymmetry (an observer may sit ON the interface, a SOURCE may not)
// is the geometry rather than sloppiness -- likewise stays in Python, because
// it is exact and its words are contract.
//
// Returns (values (n, 6), tail_panels (n,), head_panels (n,), converged (n,),
// accelerated (n,), selfconv (n,)) -- selfconv is -1.0 where the coarse
// machine was not run. `converged` is the one a caller must not drop: see the
// Wynn note at the top of this block.
static py::tuple transmitted_six_integrals_batch(
    double k_p, std::complex<double> k_m,
    py::array_t<double, py::array::c_style | py::array::forcecast> rho,
    py::array_t<double, py::array::c_style | py::array::forcecast> z,
    py::array_t<double, py::array::c_style | py::array::forcecast> zp,
    bool swap, double rtol_fine, int depth, double detour,
    py::array_t<double, py::array::c_style | py::array::forcecast> gx,
    py::array_t<double, py::array::c_style | py::array::forcecast> gw,
    bool selfconv, double rtol_coarse, int depth_coarse, double detour_coarse,
    py::array_t<double, py::array::c_style | py::array::forcecast> gxc,
    py::array_t<double, py::array::c_style | py::array::forcecast> gwc,
    int max_panels) {
    auto rb = rho.unchecked<1>();
    auto zb = z.unchecked<1>();
    auto pb = zp.unchecked<1>();
    const py::ssize_t n = rb.shape(0);
    if (zb.shape(0) != n || pb.shape(0) != n)
        throw std::invalid_argument("rho, z and zp must have the same length");
    const double *gxp, *gwp, *gxcp, *gwcp;
    int ng, ngc;
    mw568::gauss_view(gx, gw, &gxp, &gwp, &ng);
    mw568::gauss_view(gxc, gwc, &gxcp, &gwcp, &ngc);

    py::array_t<std::complex<double>> vals({n, py::ssize_t(6)});
    py::array_t<int> tail(n), head(n);
    py::array_t<bool> conv(n), accel(n);
    py::array_t<double> sconv(n);
    auto vb = vals.mutable_unchecked<2>();
    int *tp = tail.mutable_data();
    int *hp = head.mutable_data();
    bool *cp = conv.mutable_data();
    bool *ap = accel.mutable_data();
    double *sp = sconv.mutable_data();
    const mw_contour::cd km(k_m);

    {
        py::gil_scoped_release release;
        // `dynamic`, and on this family it matters more than it did on U2's.
        // The tail cost runs as cot(theta_true), so the bottom row of a
        // (ln R, theta) fill costs THOUSANDS of panels per node where the top
        // row costs tens -- a two-decade spread inside one call, arriving
        // sorted by row. Static scheduling would hand one thread the whole
        // grazing band and the fill would take as long as its slowest chunk.
        #pragma omp parallel for schedule(dynamic)
        for (py::ssize_t i = 0; i < n; ++i) {
            mw568_trans::SixResultT r;
            mw568_trans::six_transmitted_one(
                rb(i), zb(i), pb(i), k_p, km, swap, rtol_fine, depth, detour,
                gxp, gwp, ng, selfconv, rtol_coarse, depth_coarse,
                detour_coarse, gxcp, gwcp, ngc, max_panels, r);
            for (int c = 0; c < 6; ++c) vb(i, c) = r.val[c];
            tp[i] = r.tail_panels;
            hp[i] = r.head_panels;
            cp[i] = r.converged;
            ap[i] = r.accel;
            sp[i] = r.selfconv;
        }
    }
    return py::make_tuple(vals, tail, head, conv, accel, sconv);
}

// The projected transmitted pair table in C++: `above`/`t_above` (A, 3),
// `below`/`t_below` (B, 3), returning ((A, B) complex, min R, max R, min
// theta, max theta, min |z'|, max |z'|).
//
// ONE kernel serves BOTH directions of travel. The geometry is in ROLE order
// (above, below) either way -- it has to be, because the surfaces are
// tabulated with the below point as the source and the horizontal direction
// runs from the below point to the above one in both directions -- and
// `transposed` is what turns the upward dyad into the downward one, by
// swapping T_rho^V with T_z^H. The caller transposes the (A, B) table into its
// own (n_obs, n_src) when the observer is the below point. Two entry points
// reading one kernel is exactly the "above->below is the same integral" claim
// expressed in code: if it were two kernels the identity gate would be gating
// two ports against each other rather than a transpose against itself.
//
// `t_above` / `t_below` are the RAW tangents, not decomposed the way
// `somm_proj::tangent_decomp` decomposes the +-=+ family's: this family's
// combination reads t.d-hat and t x d-hat|_z and t_z directly, with no
// magnitude/direction split, because a complex moment has no direction to
// normalize (and the tangent is handed straight through as one).
//
// THE FOUR REFUSALS ARE NOT TRANSCRIBED HERE -- past `r_max`, under `r_min`,
// theta outside the tabulated band, |z'| off the ladder. The kernel reports
// the query's extremes and the Python layer feeds them straight back to
// `TransmittedGrid.eval`, which raises in its own words. One copy of that
// prose, and no way for the two paths to drift on which geometries they serve.
// It matters more here than it did on U2: every one of those four is a REFUSAL
// rather than a clamp precisely because the transmitted surface is the whole
// field and there is no negligible tail to freeze, so a C++ path that quietly
// clamped instead would return a confident wrong number rather than an error.
static py::tuple transmitted_field_proj_batch(
    py::array_t<double, py::array::c_style | py::array::forcecast> above,
    py::array_t<double, py::array::c_style | py::array::forcecast> t_above,
    py::array_t<double, py::array::c_style | py::array::forcecast> below,
    py::array_t<double, py::array::c_style | py::array::forcecast> t_below,
    double ground_z, double k_p, std::complex<double> k_m, bool transposed,
    double lnr0, double dlnr, double th0, double dth, double lnz0, double dlnz,
    double r_max,
    py::array_t<std::complex<double>,
                py::array::c_style | py::array::forcecast> grid_vals) {
    auto ab = above.unchecked<2>();
    auto tab = t_above.unchecked<2>();
    auto bb = below.unchecked<2>();
    auto tbb = t_below.unchecked<2>();
    if (ab.shape(1) != 3 || tab.shape(1) != 3 || bb.shape(1) != 3 ||
        tbb.shape(1) != 3)
        throw std::runtime_error("point/tangent arrays must have shape (*, 3)");
    if (ab.shape(0) != tab.shape(0) || bb.shape(0) != tbb.shape(0))
        throw std::runtime_error("points and tangents must have matching length");
    auto gv = grid_vals.unchecked<4>();
    if (gv.shape(1) != 5)
        throw std::runtime_error(
            "the transmitted grid must have five surfaces per rung");

    mw568_trans::TGridView G;
    G.vals = grid_vals.data();
    G.n_zp = gv.shape(0);
    G.n_r = gv.shape(2);
    G.n_th = gv.shape(3);
    if (G.n_r < 4 || G.n_th < 4 || (G.n_zp != 1 && G.n_zp < 4))
        throw std::runtime_error("the transmitted grid is too small for a cubic");
    G.lnr0 = lnr0;
    G.dlnr = dlnr;
    G.th0 = th0;
    G.dth = dth;
    G.lnz0 = lnz0;
    G.dlnz = dlnz;
    G.tiny = 1e-12 * r_max;
    G.half_pi = 0.5 * M_PI;

    const py::ssize_t A = ab.shape(0);
    const py::ssize_t B = bb.shape(0);
    py::array_t<std::complex<double>> out({A, B});
    auto out_m = out.mutable_unchecked<2>();
    const mw_contour::cd km(k_m);

    // Per-above-row extremes, reduced serially afterwards: a min/max
    // `reduction` clause is OpenMP 3.1 and this file stays portable to MSVC's
    // classic /openmp.
    const py::ssize_t AA = A > 0 ? A : 1;
    std::vector<double> row_rlo(AA, std::numeric_limits<double>::infinity());
    std::vector<double> row_rhi(AA, 0.0);
    std::vector<double> row_thlo(AA, 0.5 * M_PI);
    std::vector<double> row_thhi(AA, 0.0);
    std::vector<double> row_zlo(AA, std::numeric_limits<double>::infinity());
    std::vector<double> row_zhi(AA, 0.0);

    {
        py::gil_scoped_release release;
        #pragma omp parallel for schedule(static)
        for (py::ssize_t a = 0; a < A; ++a) {
            const double ax = ab(a, 0), ay = ab(a, 1), az = ab(a, 2);
            const double ta[3] = {tab(a, 0), tab(a, 1), tab(a, 2)};
            double rlo = std::numeric_limits<double>::infinity(), rhi = 0.0;
            double tlo = 0.5 * M_PI, thi = 0.0;
            double zlo = std::numeric_limits<double>::infinity(), zhi = 0.0;
            for (py::ssize_t b = 0; b < B; ++b) {
                const double tb[3] = {tbb(b, 0), tbb(b, 1), tbb(b, 2)};
                // Travel order: the observer's tangent is the above one going
                // up and the below one coming down; the geometry stays in
                // role order either way.
                const double *to = transposed ? tb : ta;
                const double *ts = transposed ? ta : tb;
                double rq, thq, zq;
                out_m(a, b) = mw568_trans::proj_one_transmitted(
                    G, ground_z, k_p, km, ax, ay, az, bb(b, 0), bb(b, 1),
                    bb(b, 2), to, ts, transposed, rq, thq, zq);
                if (rq < rlo) rlo = rq;
                if (rq > rhi) rhi = rq;
                if (thq < tlo) tlo = thq;
                if (thq > thi) thi = thq;
                if (zq < zlo) zlo = zq;
                if (zq > zhi) zhi = zq;
            }
            row_rlo[a] = rlo;
            row_rhi[a] = rhi;
            row_thlo[a] = tlo;
            row_thhi[a] = thi;
            row_zlo[a] = zlo;
            row_zhi[a] = zhi;
        }
    }

    double mn_r = std::numeric_limits<double>::infinity(), mx_r = 0.0;
    double mn_th = 0.5 * M_PI, mx_th = 0.0;
    double mn_z = std::numeric_limits<double>::infinity(), mx_z = 0.0;
    for (py::ssize_t a = 0; a < A; ++a) {
        if (row_rlo[a] < mn_r) mn_r = row_rlo[a];
        if (row_rhi[a] > mx_r) mx_r = row_rhi[a];
        if (row_thlo[a] < mn_th) mn_th = row_thlo[a];
        if (row_thhi[a] > mx_th) mx_th = row_thhi[a];
        if (row_zlo[a] < mn_z) mn_z = row_zlo[a];
        if (row_zhi[a] > mx_z) mx_z = row_zhi[a];
    }
    if (!std::isfinite(mn_r)) mn_r = 0.0;
    if (!std::isfinite(mn_z)) mn_z = 0.0;
    return py::make_tuple(out, mn_r, mx_r, mn_th, mx_th, mn_z, mx_z);
}



void register_mw568(py::module_ &m) {
    // The three #568 capability flags, set HERE beside the bindings they
    // vouch for (#710 review): `_sommerfeld_below` and
    // `_sommerfeld_transmitted` gate on the flag ALONE, so flag and symbols
    // must live in one TU or an edit here could leave a flag advertising a
    // contract whose symbols are gone.
    //
    // momwire#568 unit 1: the shared contour engine's TEST entry points. The
    // engine itself is header-only (`_contour_engine_inline.h`); these
    // exist so the Python suite can gate it before U2/U3 ride on it.
    m.attr("contour_engine_568") = true;
    // momwire#568 unit 2: the below/below fills on that engine. Its OWN
    // capability flag, deliberately not `contour_engine_568` — a .so built at
    // U1 exports the engine's test entry points and would otherwise claim to
    // carry U2's contract too, handing `_sommerfeld_below` a missing symbol
    // instead of the graceful numpy fallback the guard exists to give.
    m.attr("below_fills_568") = true;
    // momwire#568 unit 3: the transmitted fills on that engine. Its OWN
    // capability flag, deliberately neither `contour_engine_568` nor
    // `below_fills_568` — a .so built at U1 or U2 exports those symbols and
    // would otherwise claim to carry U3's contract too, handing
    // `_sommerfeld_transmitted` a missing symbol instead of the graceful
    // numpy fallback the guard exists to give.
    m.attr("transmitted_fills_568") = true;

    m.def("bessel_j0_j1x_complex", &bessel_j0_j1x_complex,
          "(J0(x), J1(x)/x) at COMPLEX x -- the C++ twin of "
          "_sommerfeld._bessel_j0_j1x, with the same |x| < 1e-6 series switch. "
          "Ascending series below |x| = 8, Miller's normalized downward "
          "recurrence to |x| = 25, Hankel P/Q asymptotics above. Takes a 1-D "
          "complex array, returns two arrays of the same length.",
          py::arg("x"));
    m.def("contour_engine_sommerfeld_identity",
          &contour_engine_sommerfeld_identity,
          "Test instantiation of the shared adaptive-contour engine (NC = 1) "
          "on the Sommerfeld-identity integrand J0(lam rho) e^{-gamma h} "
          "lam/gamma, whose contour integral is exactly e^{-jkR}/R with "
          "R = sqrt(rho^2 + h^2). `use_k_m` selects which wavenumber the "
          "integrand is built on; the head's `a` and branch-point marks always "
          "come from the (k_p, k_m) pair as _run_contour computes them. "
          "Returns (value, head_panels, tail_panels, converged, accel).",
          py::arg("rho"), py::arg("h"), py::arg("k_p"), py::arg("k_m"),
          py::arg("use_k_m"), py::arg("rtol"), py::arg("depth"),
          py::arg("detour"), py::arg("gx"), py::arg("gw"),
          py::arg("max_panels"));
    m.def("contour_engine_synth6", &contour_engine_synth6,
          "Test instantiation of the shared adaptive-contour engine at NC = 6 "
          "on a synthetic vector integrand whose Python twin lives in "
          "tests/test_contour_engine_568.py, so the numpy engine and this one "
          "can be run on the same mathematics and compared. Returns "
          "(values (6,), head_panels, tail_panels, converged, accel) -- the "
          "same five things _sommerfeld_below._run_contour returns.",
          py::arg("rho"), py::arg("h"), py::arg("k_p"), py::arg("k_m"),
          py::arg("rtol"), py::arg("depth"), py::arg("detour"), py::arg("gx"),
          py::arg("gw"), py::arg("max_panels"));
    m.def("below_six_integrals_batch", &below_six_integrals_batch,
          "The six below/below lambda-integrals at each (rho[i], h[i]) — the "
          "C++ twin of _sommerfeld_below._six_integrals_below, on the shared "
          "contour engine, OpenMP across nodes with the GIL released. k_p is "
          "the free-space wavenumber (real) and k_m the in-medium one (complex, "
          "Im <= 0); both arrive derived so the branch choice stays in "
          "`k_medium`. `selfconv` additionally runs the coarse machine and "
          "reports the componentwise relative spread. Returns (values (n, 6), "
          "tail_panels, head_panels, converged, accelerated, selfconv), the "
          "last being -1.0 where the coarse machine was not run.",
          py::arg("k_p"), py::arg("k_m"), py::arg("rho"), py::arg("h"),
          py::arg("rtol_fine"), py::arg("depth"), py::arg("detour"),
          py::arg("gx"), py::arg("gw"), py::arg("selfconv"),
          py::arg("rtol_coarse"), py::arg("depth_coarse"),
          py::arg("detour_coarse"), py::arg("gxc"), py::arg("gwc"),
          py::arg("max_panels"));
    m.def("remainder_field_proj_batch_below", &remainder_field_proj_batch_below,
          "Interpolated + projected below/below remainder table — the complex-"
          "wavenumber TWIN of remainder_field_proj_batch (which carries a "
          "`double k` and a divide-out that depends on R1 alone). Same grid "
          "flattening after (ground_z, k_p, k_m, th_min); hh is the two depths "
          "added and g is divide_out_below's two-leg blend. Returns ((M, S) "
          "complex, max R1, min theta, max theta) — the query extremes so the "
          "caller can let SommerfeldGridBelow.eval raise the refusals in its "
          "own words.",
          py::arg("obs"), py::arg("t_obs"), py::arg("src"), py::arg("t_src"),
          py::arg("ground_z"), py::arg("k_p"), py::arg("k_m"), py::arg("th_min"),
          py::arg("th_band_hi"),
          py::arg("r1_max"), py::arg("r_break"), py::arg("th_split"),
          py::arg("r_near"), py::arg("reg_r0"), py::arg("reg_dr"),
          py::arg("reg_th0"), py::arg("reg_dth"), py::arg("reg_vals"));
    m.def("transmitted_integrand_six", &transmitted_integrand_six,
          "The six transmitted lambda-integrands POINTWISE at complex lam — "
          "the C++ twin of _sommerfeld_transmitted._integrand_six_transmitted, "
          "stacked (n, 6) in that function's order. Not on any product path; "
          "it exists so the two mutations an INTEGRATED gate cannot see — "
          "index 1 spelled gamma^2 + k^2 rather than lam^2, and index 4 "
          "collapsed onto index 0 — can be read at chosen lam.",
          py::arg("lam"), py::arg("rho"), py::arg("z"), py::arg("zp"),
          py::arg("k_p"), py::arg("k_m"), py::arg("swap") = false);
    m.def("transmitted_six_integrals_batch", &transmitted_six_integrals_batch,
          "The six transmitted lambda-integrals at each (rho[i], z[i], zp[i]) "
          "— the C++ twin of _sommerfeld_transmitted._integrand_six_transmitted "
          "under _six_integrals_transmitted's driver, on the shared contour "
          "engine, OpenMP across nodes with the GIL released. k_p is the "
          "free-space wavenumber (real) and k_m the in-medium one (complex, "
          "Im <= 0); both arrive derived so the branch choice stays in "
          "`k_medium`. `swap` builds the above->below twin (the reciprocity "
          "gate, not the product path). The tail's decay length is |zp| + |z|. "
          "`selfconv` additionally runs the coarse machine and reports the "
          "componentwise relative spread. Returns (values (n, 6), tail_panels, "
          "head_panels, converged, accelerated, selfconv), the last being -1.0 "
          "where the coarse machine was not run. `converged` is contract, not "
          "decoration: a truncated transmitted tail was measured 4.5e+3 "
          "relative wrong, so a caller must tally it.",
          py::arg("k_p"), py::arg("k_m"), py::arg("rho"), py::arg("z"),
          py::arg("zp"), py::arg("swap"), py::arg("rtol_fine"), py::arg("depth"),
          py::arg("detour"), py::arg("gx"), py::arg("gw"), py::arg("selfconv"),
          py::arg("rtol_coarse"), py::arg("depth_coarse"),
          py::arg("detour_coarse"), py::arg("gxc"), py::arg("gwc"),
          py::arg("max_panels"));
    m.def("transmitted_field_proj_batch", &transmitted_field_proj_batch,
          "Interpolated + projected transmitted pair table — the C++ twin of "
          "_sommerfeld_transmitted's two projection entry points, ONE kernel "
          "for both directions of travel. The geometry is always in ROLE order "
          "(above, below) and `transposed` swaps T_rho^V with T_z^H, which is "
          "the whole content of the reciprocity transpose; the caller "
          "transposes the (A, B) table into its own (n_obs, n_src) when the "
          "observer is the below point. The grid arrives as one (n_zp, 5, n_r, "
          "n_th) table plus its three axis origins/steps — a single uniform "
          "(ln R, theta) lattice over a log|z'| ladder, not the remainder "
          "families' region model. Returns ((A, B) complex, min R, max R, min "
          "theta, max theta, min |z'|, max |z'|) — the query extremes, so the "
          "caller can let TransmittedGrid.eval raise its four refusals in its "
          "own words.",
          py::arg("above"), py::arg("t_above"), py::arg("below"),
          py::arg("t_below"), py::arg("ground_z"), py::arg("k_p"),
          py::arg("k_m"), py::arg("transposed"), py::arg("lnr0"),
          py::arg("dlnr"), py::arg("th0"), py::arg("dth"), py::arg("lnz0"),
          py::arg("dlnz"), py::arg("r_max"), py::arg("grid_vals"));
}

