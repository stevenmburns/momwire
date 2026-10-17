// The below/below remainder's projection with no Python (momwire#1512 B2):
// the kernel behind `remainder_field_proj_batch_below`, which binds it, moved
// verbatim from `_accel_mw568.cpp` (its five stages, their lanes and the
// blocked loop), so the standalone executable runs the extension's kernel.
// `_mw_core.h` comes first for the TU-wide `omp declare simd` declarations
// (momwire#1194), as in every kernel TU. Not one operation moved: the binding's
// outputs hash identically before and after (scripts/core_split/
// below_proj_hash.py), and the buried fill's stage gate reads it.
#include "_study_timers.h"
#include "_mw_core.h"
#include "_somm_below_proj_core.h"
#include "_accel_somm_proj_inline.h"

#include <cstring>
#include <limits>

#ifdef _OPENMP
#include <omp.h>
#endif

#include "_lanes.h"
#if MW_LANES_SIMD && MW_LANES_FUSED && MW_LANES_PACKED && !defined(MW_STUDY_SSE2_LANES_NI_ONLY)
#define MW568_LANES 1
#else
#define MW568_LANES 0
#endif

namespace mw568_below {
using cd = std::complex<double>;

static const cd MW_BJ(0.0, 1.0);

// acc - x * s for complex acc, x and real s, the product fused into the
// difference (one fused op per part): `proj_one_below`'s differences, in the
// split GCC's contraction made there before momwire#1194.
static inline cd sub_scaled(const cd &acc, const cd &x, double s) {
    return cd(mw_fma::fma(-x.real(), s, acc.real()),
              mw_fma::fma(-x.imag(), s, acc.imag()));
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
// One pair is five stages, each its own function below so that the per-pair
// composition (`proj_one_below`) and the blocked loop of
// `remainder_field_proj_batch_below` run the SAME expressions: geometry,
// stencil address and weights, the stencil read, the divide-out, and the
// projection. momwire#1224 split them out; not one operation moved, and
// tests/test_below_replay_blocked_1224.py holds the two compositions equal
// as uint64.
//
// Explicit fused multiply-adds throughout (momwire#1214, _fma_inline.h). This
// was the body of the kernel's per-pair loop, and each multiply-add GCC
// contracted here before -ffp-contract=off (#1194) is written out fused, in
// the split its contraction made (read from GCC's widening_mul dump of the
// pre-#1194 build: in most sums it fused the SECOND product, and in the
// complex products which half it fused follows the operand order it saw),
// with every sum in its source order. The above-ground twin
// `somm_proj::proj_one` is shared with other TUs and is not touched.

// Stage 1: the pair's offsets, horizontal distance and depth sum.
static inline void below_pair_geometry(double ground_z, double ox, double oy,
                                       double oz, double sx, double sy,
                                       double sz, double &dx, double &dy,
                                       double &rho, double &hh) {
    dx = ox - sx;
    dy = oy - sy;
    rho = std::hypot(dx, dy);
    hh = (ground_z - oz) + (ground_z - sz);
}

static inline double below_pair_r1(double rho, double hh) {
    return std::sqrt(mw_fma::fma(rho, rho, hh * hh));
}

// Where a query reads the tables: the first node of its 4x4 stencil in the
// region's first surface plane, the plane and row strides, and the two
// Lagrange weight vectors.
struct BelowStencil {
    const cd *base;
    std::ptrdiff_t nth;
    std::ptrdiff_t plane;
    double wr[4];
    double wt[4];
};

// Stage 2: inline SommerfeldGridBelow.eval(r1, theta) up to the table read.
// `theta` is the RAW atan2(hh, rho); the clamp is here.
// The address alone (region, axis offsets, first stencil node): the one
// copy, which `below_stencil` reads through and the lazy grid's census
// (`below_field_proj_nodes`) marks from.
static inline int below_address(const somm_proj::GridView &G, double th_min,
                                double th_band_floor_hi, double th_band_lo_hi,
                                double th_band_hi, double r1, double theta,
                                double &fr, double &ft, int &i0, int &j0) {
    if (theta < th_min) theta = th_min;
    else if (theta > G.half_pi) theta = G.half_pi;
    const double r1c = r1 > G.r1_max ? G.r1_max : r1;
    // FIVE theta bands per R1 zone since momwire#1064 (four since #935, three
    // since #838), ordered (R1 zone) x (theta band) with theta fastest -- the layout
    // SommerfeldGridBelow's constructor builds. NOT the parent's 2-band/
    // 3-zone scheme: six regions there means the momwire#159 far zone, which
    // this family does not have (r_near == r1_max).
    //
    // EVERY band edge is passed in explicitly (#935 added the second as its
    // own argument, #1064 the third) rather than derived. Inferring a band count from
    // reg_vals.size() -- or, worse, inferring the edges from the count --
    // would make this a THIRD copy of the layout and the one that silently
    // drifts when a band is next added. The size check at the pybind seam is
    // an assertion about that contract, not the source of it.
    //
    // STRICT `<` at EVERY edge, matching `SommerfeldGridBelow._interp`: the
    // side ABOVE each seam owns the edge, so theta == th_band_floor_hi is the
    // low band's, theta == th_band_lo_hi the mid band's and theta ==
    // th_band_hi the old grazing band's. At the two older seams that side is
    // the coarser one and the node is the same fill on either side, but
    // `th0 + dth*n` need not reproduce it to the last bit. At #1064's seam the
    // floor band's lattice runs PAST its edge (a 4-point stencil needs four
    // nodes), and its top two nodes carry the low band's own values.
    const int band =
        theta < th_band_floor_hi
            ? 0
            : (theta < th_band_lo_hi
                   ? 1
                   : (theta < th_band_hi ? 2 : (theta <= G.th_split ? 3 : 4)));
    // THREE R1 zones since momwire#838 part 2: inner, near, and the far
    // annulus above `r_near` (the old cap), which carries a finer theta
    // lattice of its own. The stride is the BAND count, so it moved with it
    // (to 5 with momwire#1064).
    const int zone = r1c <= G.r_break ? 0 : (r1c <= G.r_near ? 5 : 10);
    const int reg = zone + band;
    fr = (r1c - G.rr0[reg]) / G.rdr[reg];
    ft = (theta - G.rth0[reg]) / G.rdth[reg];
    i0 = (int)std::floor(fr) - 1;
    j0 = (int)std::floor(ft) - 1;
    if (i0 < 0) i0 = 0; else if (i0 > G.nR[reg] - 4) i0 = (int)G.nR[reg] - 4;
    if (j0 < 0) j0 = 0; else if (j0 > G.nTh[reg] - 4) j0 = (int)G.nTh[reg] - 4;
    return reg;
}

static inline void below_stencil(const somm_proj::GridView &G, double th_min,
                                 double th_band_floor_hi, double th_band_lo_hi,
                                 double th_band_hi, double r1, double theta,
                                 BelowStencil &st) {
    double fr, ft;
    int i0, j0;
    const int reg = below_address(G, th_min, th_band_floor_hi, th_band_lo_hi,
                                  th_band_hi, r1, theta, fr, ft, i0, j0);
    somm_proj::lagrange4(fr - i0, st.wr);
    somm_proj::lagrange4(ft - j0, st.wt);
    st.nth = G.nTh[reg];
    st.plane = G.nR[reg] * G.nTh[reg];
    st.base = G.vptr[reg] + (std::ptrdiff_t)i0 * st.nth + j0;
}

// Stage 3: the stencil read, surfaces in the table's order
// (IrhoV, IzV, IrhoH, IphiH).
static inline void below_surfaces(const BelowStencil &st, cd *surf) {
    for (int s = 0; s < 4; ++s) {
        const cd *plane = st.base + (std::ptrdiff_t)s * st.plane;
        cd acc(0.0, 0.0);
        for (int i = 0; i < 4; ++i) {
            const cd *row = plane + (std::ptrdiff_t)i * st.nth;
            // (((row0 w0 + row1 w1) + row2 w2) + row3 w3), as written, each
            // later product fused into the running sum. (Before #1194 GCC's
            // SLP pass packed the first row's re/im halves so that its
            // imaginary half fused row0 instead; that one split is not
            // reproduced.)
            cd rs = mw_fma::mul_add(row[1], st.wt[1], row[0] * st.wt[0]);
            rs = mw_fma::mul_add(row[2], st.wt[2], rs);
            rs = mw_fma::mul_add(row[3], st.wt[3], rs);
            acc = mw_fma::mul_add(rs, st.wr[i], acc);
        }
        surf[s] = acc;
    }
}

// Stage 4: `divide_out_below`'s g. The exponent cd(k_p rho, 0) + k_m hh per
// part: the real part fuses k_m hh; the imaginary part's 0.0 + x is left as
// written (fusing it rounds identically), and so is the product with -j,
// whose "fused" halves only ever multiply by -0.0.
static inline cd below_divide_out(double k_p, const cd &k_m, double rho,
                                  double hh, double r1) {
    const cd arg(mw_fma::fma(k_m.real(), hh, k_p * rho),
                 0.0 + k_m.imag() * hh);
    return std::exp(-MW_BJ * arg) / r1;
}

// `below_divide_out` with the exponential's MODULUS held across pairs of one
// depth sum (momwire queue item 4). The exponent -j (k_p rho + k_m hh) has
// real part 0.0 + Im(k_m) hh, a function of hh alone, and on a buried radial
// screen every pair of one observer row has the same hh. glibc's cexp is
// exp(Re z) * (cos Im z, sin Im z) with one sincos, for a finite z whose real
// part is at most (DBL_MAX_EXP - 1) ln 2 (s_cexp_template.c), and so is this
// -- the real part here is <= 0 -- with exp(Re z) taken from `ex` when hh is
// the previous pair's. The same libm calls on the same arguments: the same
// floats. Used only where `cexp_split_ok()` has checked that claim against
// this process's libm (`below_cexp_split_check`), and only on glibc builds.
#if defined(__GLIBC__)
#define MW568_CEXP_SPLIT 1
static inline cd cexp_split(const cd &z, double ex) {
    double sn, cs;
    if (std::fabs(z.imag()) > std::numeric_limits<double>::min()) {
        sincos(z.imag(), &sn, &cs);
    } else {
        sn = z.imag();
        cs = 1.0;
    }
    return cd(ex * cs, ex * sn);
}

// Whether std::exp(complex) is cexp_split here, bit for bit, over a spread
// of exponents like the divide-out's (real part -40 .. 0, imaginary part
// -60 .. 0, tiny and zero imaginary parts included). Evaluated once.
static bool below_cexp_split_check() {
    uint64_t st = 0x9E3779B97F4A7C15ULL;
    auto u01 = [&st]() {
        st ^= st << 13;
        st ^= st >> 7;
        st ^= st << 17;
        return (double)(st >> 11) * (1.0 / 9007199254740992.0);
    };
    for (int i = 0; i < 4096; ++i) {
        double x = -40.0 * u01() * u01();
        double y = -60.0 * u01();
        if (i % 64 == 0) y = 0.0;
        if (i % 64 == 1) y = -1e-310;
        if (i % 64 == 2) x = 0.0;
        const cd z(x, y);
        const cd a = std::exp(z);
        const cd b = cexp_split(z, std::exp(z.real()));
        if (std::memcmp(&a, &b, sizeof a) != 0) return false;
    }
    return true;
}

static bool cexp_split_ok() {
    static const bool ok = below_cexp_split_check();
    return ok;
}

static inline cd below_divide_out_held(double k_p, const cd &k_m, double rho,
                                       double hh, double r1, double &held_hh,
                                       double &held_ex) {
    const cd arg(mw_fma::fma(k_m.real(), hh, k_p * rho),
                 0.0 + k_m.imag() * hh);
    const cd z = -MW_BJ * arg;
    if (!(hh == held_hh)) {
        held_ex = std::exp(z.real());
        held_hh = hh;
    }
    return cexp_split(z, held_ex) / r1;
}
#else
#define MW568_CEXP_SPLIT 0
#endif

// Stage 5: the projection (eqs 143-147), over g.
static inline cd below_project(const somm_proj::GridView &G, const cd *surf,
                               const cd &g, double rho, double dx, double dy,
                               double tox, double toy, double toz, double sux,
                               double suy, double sthsrc, double stzsrc) {
    const cd IrhoV = surf[0], IzV = surf[1], IrhoH = surf[2], IphiH = surf[3];
    const bool safe_r = rho > G.tiny;
    const double inv_rho = safe_r ? 1.0 / rho : 0.0;
    const double dhx = safe_r ? dx * inv_rho : sux;
    const double dhy = safe_r ? dy * inv_rho : suy;
    const double cphi = mw_fma::fma(sux, dhx, suy * dhy);
    const double sphi = mw_fma::fma(sux, dhy, -(suy * dhx));
    const double sc = sthsrc * cphi;
    // mw_fma::mul(x, y) fuses x.re y.im into the imaginary part, so the
    // operand order below is the contraction's, not a typo: g * S fused
    // S.re g.im in e_rho and e_z and g.re S.im in e_phi.
    using mw_fma::mul;
    const cd e_rho = mul(mw_fma::mul_add(IrhoH, sc, stzsrc * IrhoV), g);
    const cd e_phi = mul(g, sthsrc * sphi * IphiH);
    const cd e_z = mul(sub_scaled(stzsrc * IzV, IrhoV, sc), g);
    const cd r = mw_fma::mul_add(mw_fma::mul_add(e_phi, dhx, dhy * e_rho), toy,
                                 tox * sub_scaled(dhx * e_rho, e_phi, dhy));
    return mw_fma::mul_add(e_z, toz, r);
}

// One pair, the five stages in order. The query's (R1, theta) are handed back
// so the caller can let `SommerfeldGridBelow.eval` raise the refusals in its
// own words; nothing in this file transcribes those messages.
static inline cd proj_one_below(const somm_proj::GridView &G, double th_min,
                                double th_band_floor_hi, double th_band_lo_hi,
                                double th_band_hi,
                                double ground_z, double k_p, const cd &k_m,
                                double ox, double oy, double oz, double tox,
                                double toy, double toz, double sx, double sy,
                                double sz, double sux, double suy,
                                double sthsrc, double stzsrc, double &r1_out,
                                double &th_out) {
    double dx, dy, rho, hh;
    below_pair_geometry(ground_z, ox, oy, oz, sx, sy, sz, dx, dy, rho, hh);
    const double r1 = below_pair_r1(rho, hh);
    const double theta = std::atan2(hh, rho);
    r1_out = r1;
    th_out = theta;
    BelowStencil st;
    below_stencil(G, th_min, th_band_floor_hi, th_band_lo_hi, th_band_hi, r1,
                  theta, st);
    cd surf[4];
    below_surfaces(st, surf);
    const cd g = below_divide_out(k_p, k_m, rho, hh, r1);
    return below_project(G, surf, g, rho, dx, dy, tox, toy, toz, sux, suy,
                         sthsrc, stzsrc);
}

// Sources per block of the kernel's blocked loop. Each stage runs over the
// whole block before the next starts, so a pair's long dependency chain
// (theta -> region -> divisions -> table address -> loads -> fused sums) is
// no longer the loop's critical path: the stages' iterations are independent
// and overlap. Measured on the hub x8 deck's real chunks (i7-4770K, 4
// threads), 37.5 -> 30.4 ns per pair, flat from 16 to 256 (64 here: ~14 kB
// of per-thread scratch, inside L1+L2).
constexpr int BELOW_BLOCK = 64;

#if MW568_LANES
// The blocked loop's stencil, read and projection stages, W pairs to a
// vector (momwire#1290). The blocked loop spent most of a pair's ~120 ns in
// scalar arithmetic: six divisions in the stencil weights, sixteen complex
// fused sums per surface read, and the projection's products, around three
// transcendentals (hypot, atan2, exp) that stay scalar libm calls here as
// they are there.
//
// Every lane does a stage function's own operations on its own pair, in that
// function's order: the same comparisons in the same nesting, the same
// divisions (correctly rounded in any lane), and each `mw_fma` helper's fused
// op as `fmadd`, which is per lane exactly `std::fma`. A negated product is
// `fnmadd`: fma(-a, b, c) and -(a b) + c are one exact product and one
// rounding. Nothing is reassociated and no reduction crosses lanes, so a
// lane's floats are the scalar stage's floats (gated as uint64 against
// `lanes=False` and against the per-pair composition,
// tests/test_below_lanes_1290.py). The stencil's tail and the surface read
// are shared with the above grid (`somm_proj_lanes`, in
// `_accel_somm_proj_inline.h`).
using namespace mw_lanes;

// `below_stencil` for pairs b .. b+W-1: theta and r1 clamped, the band and
// zone selects, then the shared tail -- the region's clamped first stencil
// node (i0, j0) and the two Lagrange weight vectors, into SoA arrays.
static inline void below_stencil_lanes(const somm_proj::GridView &G,
                                       const int *nR32, const int *nTh32,
                                       double th_min, double th_band_floor_hi,
                                       double th_band_lo_hi, double th_band_hi,
                                       const double *r1, const double *th,
                                       int *reg_o, int *i0_o, int *j0_o,
                                       double (*wr)[BELOW_BLOCK],
                                       double (*wt)[BELOW_BLOCK], int b) {
    const vd c_thmin = set1(th_min);
    const vd c_hpi = set1(G.half_pi);
    const vd c_r1max = set1(G.r1_max);
    vd t = loadu(th + b);
    // if (theta < th_min) theta = th_min; else if (theta > half_pi) ...
    const vm t_lo = cmp_lt(t, c_thmin);
    const vm t_hi = cmp_gt(t, c_hpi);
    t = blend(blend(t, c_hpi, t_hi), c_thmin, t_lo);
    vd rc = loadu(r1 + b);
    rc = blend(rc, c_r1max, cmp_gt(rc, c_r1max));
    // The band and zone selects, innermost alternative first so that each
    // outer test overrides it as the scalar ternaries nest. Small integers in
    // double lanes: exact, and the sum converts exactly.
    vd band = blend(set1(4.0), set1(3.0), cmp_le(t, set1(G.th_split)));
    band = blend(band, set1(2.0), cmp_lt(t, set1(th_band_hi)));
    band = blend(band, set1(1.0), cmp_lt(t, set1(th_band_lo_hi)));
    band = blend(band, zero(), cmp_lt(t, set1(th_band_floor_hi)));
    vd zone = blend(set1(10.0), set1(5.0), cmp_le(rc, set1(G.r_near)));
    zone = blend(zone, zero(), cmp_le(rc, set1(G.r_break)));
    somm_proj_lanes::stencil_tail<BELOW_BLOCK>(G, nR32, nTh32, rc, t,
                                               cvtt_i32(add(zone, band)), reg_o,
                                               i0_o, j0_o, wr, wt, b);
}

// `below_project` for pairs b .. b+W-1, every complex helper spelled out per
// part exactly as `_fma_inline.h` defines it; writes the W entries.
static inline void below_project_lanes(
    const somm_proj::GridView &G, const double (*sre)[BELOW_BLOCK],
    const double (*sim)[BELOW_BLOCK], const double *gre, const double *gim,
    const double *rho_, const double *dx_, const double *dy_, double tox,
    double toy, double toz, const double *ux, const double *uy,
    const double *thsrc, const double *tzsrc, int b, cd *out) {
    auto L = [&](const double *p) { return loadu(p + b); };
    const vd rho = L(rho_), dx = L(dx_), dy = L(dy_);
    const vd sux = loadu(ux), suy = loadu(uy);
    const vd sth = loadu(thsrc), stz = loadu(tzsrc);
    const vd gr = L(gre), gi = L(gim);
    const vm safe = cmp_gt(rho, set1(G.tiny));
    const vd inv = blend(zero(), div(set1(1.0), rho), safe);
    const vd dhx = blend(sux, mul(dx, inv), safe);
    const vd dhy = blend(suy, mul(dy, inv), safe);
    const vd cphi = fmadd(sux, dhx, mul(suy, dhy));
    const vd sphi = fmadd(sux, dhy, neg(mul(suy, dhx)));
    const vd sc = mul(sth, cphi);
    const vd s0r = L(sre[0]), s0i = L(sim[0]), s1r = L(sre[1]), s1i = L(sim[1]),
             s2r = L(sre[2]), s2i = L(sim[2]), s3r = L(sre[3]), s3i = L(sim[3]);
    // mw_fma::mul(x, y): (fma(xr, yr, -(xi yi)), fma(xr, yi, xi yr)).
    auto cmul_re = [](vd xr, vd xi, vd yr, vd yi) { return fmadd(xr, yr, neg(mul(xi, yi))); };
    auto cmul_im = [](vd xr, vd xi, vd yr, vd yi) { return fmadd(xr, yi, mul(xi, yr)); };
    // e_rho = mul(mul_add(IrhoH, sc, stzsrc * IrhoV), g)
    const vd ar = fmadd(s2r, sc, mul(stz, s0r));
    const vd ai = fmadd(s2i, sc, mul(stz, s0i));
    const vd er = cmul_re(ar, ai, gr, gi), ei = cmul_im(ar, ai, gr, gi);
    // e_phi = mul(g, sthsrc * sphi * IphiH)
    const vd f = mul(sth, sphi);
    const vd br = mul(f, s3r), bi = mul(f, s3i);
    const vd pr = cmul_re(gr, gi, br, bi), pi = cmul_im(gr, gi, br, bi);
    // e_z = mul(sub_scaled(stzsrc * IzV, IrhoV, sc), g)
    const vd cr = fnmadd(s0r, sc, mul(stz, s1r));
    const vd ci = fnmadd(s0i, sc, mul(stz, s1i));
    const vd zr = cmul_re(cr, ci, gr, gi), zi = cmul_im(cr, ci, gr, gi);
    // r = mul_add(mul_add(e_phi, dhx, dhy * e_rho), toy,
    //             tox * sub_scaled(dhx * e_rho, e_phi, dhy))
    const vd vtox = set1(tox), vtoy = set1(toy), vtoz = set1(toz);
    const vd i1r = fmadd(pr, dhx, mul(dhy, er));
    const vd i1i = fmadd(pi, dhx, mul(dhy, ei));
    const vd i2r = fnmadd(pr, dhy, mul(dhx, er));
    const vd i2i = fnmadd(pi, dhy, mul(dhx, ei));
    const vd rr = fmadd(i1r, vtoy, mul(vtox, i2r));
    const vd ri = fmadd(i1i, vtoy, mul(vtox, i2i));
    // mul_add(e_z, toz, r)
    store_interleaved(reinterpret_cast<double *>(out), fmadd(zr, vtoz, rr),
                      fmadd(zi, vtoz, ri));
}
#endif  // MW568_LANES
}  // namespace mw568_below

namespace somm_core {

bool below_cexp_split_ok() {
#if MW568_CEXP_SPLIT
    return mw568_below::cexp_split_ok();
#else
    return false;
#endif
}

void below_field_proj_batch(const somm_proj::GridView &G, size_t n_reg, double th_min,
                            double th_band_floor_hi, double th_band_lo_hi, double th_band_hi,
                            std::ptrdiff_t M, std::ptrdiff_t S, const double *obs,
                            const double *t_obs, const double *src, const double *t_src,
                            double ground_z, double k_p, std::complex<double> k_m,
                            std::complex<double> *out, double &mx_r1_out, double &mn_th_out,
                            double &mx_th_out, bool blocked, bool lanes) {
    STUDY_T("proj.below_batch");
    using somm_proj::cd;
    using namespace mw568_below;
    const cd km(k_m);
#if MW568_LANES
    int nR32[somm_proj::MAX_REGIONS], nTh32[somm_proj::MAX_REGIONS];
    for (size_t g = 0; g < n_reg; ++g) {
        nR32[g] = static_cast<int>(G.nR[g]);
        nTh32[g] = static_cast<int>(G.nTh[g]);
    }
    const bool use_lanes = blocked && lanes;
#if MW568_CEXP_SPLIT
    const bool split = use_lanes && cexp_split_ok();
#endif
#else
    (void)lanes;
#endif

    // Per-item extremes, reduced serially afterwards: a `reduction` clause on
    // min/max is OpenMP 3.1 and this file stays portable to MSVC's classic
    // /openmp.
    //
    // The work items are (observer row, span of sources), spans fastest,
    // handed out dynamically (momwire#1290). Over rows alone, the bs2
    // remainder's chunks (30 observer rows of up to 15k sources) split
    // 8/8/7/7 over four threads, a fixed 7 % idle. A span is a whole number
    // of blocks starting on a block boundary, and every entry is still one
    // pair's own arithmetic, so where an item runs moves no bit. The
    // extremes are kept per item and reduced in item order below: a max or
    // min with the comparisons the rows used, so the same value whatever the
    // partition (theta is atan2 of a positive depth sum, never a signed zero).
    const std::ptrdiff_t span = 16 * BELOW_BLOCK;
    const std::ptrdiff_t n_span = S > 0 ? (S + span - 1) / span : 1;
    const std::ptrdiff_t n_item = M * n_span;
    std::vector<double> row_r1(n_item > 0 ? n_item : 1, 0.0);
    std::vector<double> row_thlo(n_item > 0 ? n_item : 1, 0.5 * M_PI);
    std::vector<double> row_thhi(n_item > 0 ? n_item : 1, 0.0);

    {
        std::vector<double> sx(S), sy(S), sz(S), ux(S), uy(S), thsrc(S), tzsrc(S);
        for (std::ptrdiff_t nn = 0; nn < S; ++nn) {
            sx[nn] = src[3 * nn];
            sy[nn] = src[3 * nn + 1];
            sz[nn] = src[3 * nn + 2];
            somm_proj::tangent_decomp(t_src[3 * nn], t_src[3 * nn + 1], t_src[3 * nn + 2], ux[nn],
                                      uy[nn], thsrc[nn], tzsrc[nn]);
        }

        #pragma omp parallel for schedule(dynamic)
        for (std::ptrdiff_t item = 0; item < n_item; ++item) {
            const std::ptrdiff_t m = item / n_span;
            const std::ptrdiff_t s_lo = (item % n_span) * span;
            const std::ptrdiff_t s_hi = std::min<std::ptrdiff_t>(S, s_lo + span);
            const double ox = obs[3 * m], oy = obs[3 * m + 1], oz = obs[3 * m + 2];
            const double tox = t_obs[3 * m], toy = t_obs[3 * m + 1], toz = t_obs[3 * m + 2];
            double rmax = 0.0, tlo = 0.5 * M_PI, thi = 0.0;
            if (!blocked) {
                // The per-pair composition: the reference the blocked loop is
                // gated against bit for bit.
                for (std::ptrdiff_t nn = s_lo; nn < s_hi; ++nn) {
                    double r1q, thq;
                    out[m * S + nn] = proj_one_below(
                        G, th_min, th_band_floor_hi, th_band_lo_hi, th_band_hi,
                        ground_z, k_p, km, ox, oy, oz, tox, toy, toz, sx[nn],
                        sy[nn], sz[nn], ux[nn], uy[nn], thsrc[nn], tzsrc[nn],
                        r1q, thq);
                    if (r1q > rmax) rmax = r1q;
                    if (thq < tlo) tlo = thq;
                    if (thq > thi) thi = thq;
                }
            } else {
                // The same five stages, each over a block of sources
                // (momwire#1224). Every pair's operations are proj_one_below's,
                // in its order; only WHEN each runs moves.
                constexpr int B = BELOW_BLOCK;
                double dx[B], dy[B], rho[B], hh[B], r1[B], th[B];
#if MW568_CEXP_SPLIT
                // The held modulus (`below_divide_out_held`): per work item,
                // so no thread reads another's. NaN matches no hh.
                double held_hh = std::numeric_limits<double>::quiet_NaN();
                double held_ex = 0.0;
#endif
                BelowStencil st[B];
                cd surf[B][4];
                cd g[B];
#if MW568_LANES
                int l_reg[B], l_i0[B], l_j0[B];
                double l_wr[4][B], l_wt[4][B], l_sre[4][B], l_sim[4][B];
                double l_gre[B], l_gim[B];
#endif
                for (std::ptrdiff_t n0 = s_lo; n0 < s_hi; n0 += B) {
                    const int nb = (int)std::min<std::ptrdiff_t>(B, s_hi - n0);
                    int b_lo = 0;
                    for (int b = 0; b < nb; ++b)
                        below_pair_geometry(
                            ground_z, ox, oy, oz, sx[n0 + b], sy[n0 + b],
                            sz[n0 + b], dx[b], dy[b], rho[b], hh[b]);
                    for (int b = 0; b < nb; ++b) {
                        r1[b] = below_pair_r1(rho[b], hh[b]);
                        th[b] = std::atan2(hh[b], rho[b]);
                        if (r1[b] > rmax) rmax = r1[b];
                        if (th[b] < tlo) tlo = th[b];
                        if (th[b] > thi) thi = th[b];
                    }
#if MW568_LANES
                    if (use_lanes) {
                        // Whole lane groups through the lanes, the last
                        // nb % W pairs through the scalar stages below.
                        const int nq = nb - nb % mw_lanes::W;
                        for (int b = 0; b < nq; b += mw_lanes::W)
                            below_stencil_lanes(
                                G, nR32, nTh32, th_min, th_band_floor_hi,
                                th_band_lo_hi, th_band_hi, r1, th, l_reg, l_i0,
                                l_j0, l_wr, l_wt, b);
                        for (int b = 0; b < nq; ++b)
                            somm_proj_lanes::surfaces_lanes<B, true>(
                                G, l_reg[b], l_i0[b], l_j0[b], l_wr, l_wt,
                                l_sre, l_sim, b);
#if MW568_CEXP_SPLIT
                        if (split) {
                            for (int b = 0; b < nq; ++b) {
                                const cd gb = below_divide_out_held(
                                    k_p, km, rho[b], hh[b], r1[b], held_hh,
                                    held_ex);
                                l_gre[b] = gb.real();
                                l_gim[b] = gb.imag();
                            }
                        } else
#endif
                        for (int b = 0; b < nq; ++b) {
                            const cd gb = below_divide_out(
                                k_p, km, rho[b], hh[b], r1[b]);
                            l_gre[b] = gb.real();
                            l_gim[b] = gb.imag();
                        }
                        for (int b = 0; b < nq; b += mw_lanes::W) {
                            const std::ptrdiff_t nn = n0 + b;
                            below_project_lanes(
                                G, l_sre, l_sim, l_gre, l_gim, rho, dx, dy,
                                tox, toy, toz, &ux[nn], &uy[nn], &thsrc[nn],
                                &tzsrc[nn], b, &out[m * S + nn]);
                        }
                        b_lo = nq;
                    }
#endif
                    for (int b = b_lo; b < nb; ++b)
                        below_stencil(G, th_min, th_band_floor_hi,
                                                   th_band_lo_hi, th_band_hi,
                                                   r1[b], th[b], st[b]);
                    for (int b = b_lo; b < nb; ++b)
                        below_surfaces(st[b], surf[b]);
                    for (int b = b_lo; b < nb; ++b)
                        g[b] = below_divide_out(k_p, km, rho[b],
                                                             hh[b], r1[b]);
                    for (int b = b_lo; b < nb; ++b) {
                        const std::ptrdiff_t nn = n0 + b;
                        out[m * S + nn] = below_project(
                            G, surf[b], g[b], rho[b], dx[b], dy[b], tox, toy,
                            toz, ux[nn], uy[nn], thsrc[nn], tzsrc[nn]);
                    }
                }
            }
            row_r1[item] = rmax;
            row_thlo[item] = tlo;
            row_thhi[item] = thi;
        }
    }

    double mx_r1 = 0.0, mn_th = 0.5 * M_PI, mx_th = 0.0;
    for (std::ptrdiff_t it = 0; it < n_item; ++it) {
        if (row_r1[it] > mx_r1) mx_r1 = row_r1[it];
        if (row_thlo[it] < mn_th) mn_th = row_thlo[it];
        if (row_thhi[it] > mx_th) mx_th = row_thhi[it];
    }
    mx_r1_out = mx_r1;
    mn_th_out = mn_th;
    mx_th_out = mx_th;
}

void below_field_proj_nodes(const somm_proj::GridView &G, double th_min, double th_band_floor_hi,
                            double th_band_lo_hi, double th_band_hi, std::ptrdiff_t M,
                            std::ptrdiff_t S, const double *obs, const double *src,
                            double ground_z, double r1_cap, somm_proj::NodeMarks &kept,
                            somm_proj::NodeMarks &past, double &mx_r1_out, double &mn_th_out,
                            double &mx_th_out) {
    using namespace mw568_below;
    // Every pair the batch reads, through its own stages: the geometry
    // (`below_pair_geometry`), R1 (`below_pair_r1`), libm's atan2 and the
    // address `below_stencil` reads at (`below_address`; the lanes' stencil
    // is the same address lane by lane). The extremes are the batch's, with
    // its comparisons from its starting values. A pair `_zero_past_cap`
    // zeroes (`below_past_cap`) is marked in `past`, the rest in `kept`.
    double mx_r1 = 0.0, mn_th = 0.5 * M_PI, mx_th = 0.0;
    for (std::ptrdiff_t m = 0; m < M; ++m) {
        const double ox = obs[3 * m], oy = obs[3 * m + 1], oz = obs[3 * m + 2];
        for (std::ptrdiff_t n = 0; n < S; ++n) {
            double dx, dy, rho, hh, fr, ft;
            int i0, j0;
            below_pair_geometry(ground_z, ox, oy, oz, src[3 * n], src[3 * n + 1], src[3 * n + 2],
                                dx, dy, rho, hh);
            const double r1 = below_pair_r1(rho, hh);
            const double th = std::atan2(hh, rho);
            if (r1 > mx_r1) mx_r1 = r1;
            if (th < mn_th) mn_th = th;
            if (th > mx_th) mx_th = th;
            const int reg = below_address(G, th_min, th_band_floor_hi, th_band_lo_hi, th_band_hi,
                                          r1, th, fr, ft, i0, j0);
            (below_past_cap(rho, hh, r1_cap) ? past : kept).mark(reg, i0, j0);
        }
    }
    mx_r1_out = mx_r1;
    mn_th_out = mn_th;
    mx_th_out = mx_th;
}

}  // namespace somm_core
