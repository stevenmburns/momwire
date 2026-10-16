// The above-ground six integrands in lanes (momwire#1441 step 3): `somm::`'s
// integrand_six and its Bessel / Hankel pair, four contour nodes at a time.
//
// Every function here is the lane twin of the scalar function of the same
// name in `_accel_somm.cpp` (and, on the standalone branch, its Python-free
// copy `_somm_kernel_core.cpp`): the same operations in the same order, one
// node per lane, so a lane's result is the scalar walk's result bit for bit
// (`_somm_lanes_inline.h` says how). The data-dependent loops -- the series
// and the asymptotic sums, each truncating where its own terms say -- run
// until the last lane has stopped; a lane that has stopped keeps its values
// through a blend, exactly as the scalar loop's `break` would have left them.
// A group whose nodes straddle the series/asymptotic switch evaluates both
// branches and keeps each lane's own.
//
// `tests/test_somm_lanes_1441.py` holds lanes and walk equal as uint64 over
// recorded contour nodes of both forms.
#pragma once

#include "_somm_lanes_inline.h"

namespace somm_six_lanes {

using namespace mw_sl;

static const double SPI = 3.14159265358979323846;
static const double EULER_GAMMA = 0.57721566490153286061;
static const double SER_EPS2 = 1e-34;
static const double BESSEL_SWITCH2 = 144.0;
static const cd CI(0.0, 1.0);

MW_SL_INLINE C c_one() { return C{set1(1.0), set1(0.0)}; }

// The truncating loops below run every lane until the last one stops, and
// a lane's sums keep accumulating after its own stop. What it returns is
// what it held AT its stop: each loop saves a lane's sums into the outputs
// on the iteration it stops (a blend per stop event, not per iteration),
// and the lanes the term limit ends are saved after the loop. A lane that
// was never active returns the initial values, which nobody reads.

// somm::j01_series on the lanes of `act`.
static inline void j01_series(C z, M act, C &j0o, C &j1o) {
    const C q = cmul(cscale(-0.25, z), z);
    C t0 = c_one(), t1{set1(0.5), set1(0.0)};
    C j0 = t0, j1x = t1;
    j0o = j0;
    j1o = j1x;
    const V eps = set1(SER_EPS2);
    for (int k = 1; k <= 60 && any(act); ++k) {
        t0 = cmul(t0, cdivr(q, set1(double(k * k))));
        t1 = cmul(t1, cdivr(q, set1(double(k * (k + 1)))));
        j0 = cadd(j0, t0);
        j1x = cadd(j1x, t1);
        const M stop = m_and(act, m_and(le(norm2(t0), mul(eps, norm2(j0))),
                                        le(norm2(t1), mul(eps, norm2(j1x)))));
        if (any(stop)) {
            j0o = cblend(j0o, j0, stop);
            j1o = cblend(j1o, j1x, stop);
            act = m_andnot(act, stop);
        }
    }
    if (any(act)) {
        j0o = cblend(j0o, j0, act);
        j1o = cblend(j1o, j1x, act);
    }
}

// somm::jy01_series on the lanes of `act`.
static inline void jy01_series(C z, M act, C &j0o, C &j1o, C &y0, C &y1) {
    const int act0 = bits(act);
    const C q = cmul(cscale(-0.25, z), z);
    C t0 = c_one(), t1{set1(0.5), set1(0.0)};
    C j0 = t0, j1x = t1;
    C s0{set1(0.0), set1(0.0)}, s1 = cscale(2.0, t1);
    C s0o = s0, s1o = s1;
    j0o = j0;
    j1o = j1x;
    double hk = 0.0, hk1 = 1.0;
    const V eps = set1(SER_EPS2);
    const V one = set1(1.0);
    for (int k = 1; k <= 60 && any(act); ++k) {
        t0 = cmul(t0, cdivr(q, set1(double(k * k))));
        t1 = cmul(t1, cdivr(q, set1(double(k * (k + 1)))));
        hk += 1.0 / double(k);
        hk1 += 1.0 / double(k + 1);
        j0 = cadd(j0, t0);
        j1x = cadd(j1x, t1);
        const C term0 = cscale(hk, t0);
        const C term1 = cscale(2.0 * (hk + hk1), t1);
        s0 = csub(s0, term0);
        s1 = cadd(s1, term1);
        const M stop = m_and(
            act, m_and(m_and(le(norm2(t0), mul(eps, norm2(j0))),
                             le(norm2(t1), mul(eps, norm2(j1x)))),
                       m_and(le(norm2(term0), mul(eps, add(norm2(s0), one))),
                             le(norm2(term1), mul(eps, add(norm2(s1), one))))));
        if (any(stop)) {
            j0o = cblend(j0o, j0, stop);
            j1o = cblend(j1o, j1x, stop);
            s0o = cblend(s0o, s0, stop);
            s1o = cblend(s1o, s1, stop);
            act = m_andnot(act, stop);
        }
    }
    if (any(act)) {
        j0o = cblend(j0o, j0, act);
        j1o = cblend(j1o, j1x, act);
        s0o = cblend(s0o, s0, act);
        s1o = cblend(s1o, s1, act);
    }
    const C lg = caddr(clog(cscale(0.5, z), act0), EULER_GAMMA);
    y0 = cscale(2.0 / SPI, cmul_add(lg, j0o, s0o));
    y1 = csub(cscale(2.0 / SPI, csub(cmul(lg, cmul(j1o, z)), cdiv(1.0, z))),
              cmul(cdivr(z, set1(2.0 * SPI)), s1o));
}

// One order of somm::hankel_asym_sums: the term ladder t, the kind-2 sum sm
// and (want_plus) the kind-1 sum sp, each lane stopping at its own optimum.
// A lane whose term grows stops WITHOUT that term (saved before the add); a
// lane whose term is negligible stops WITH it (saved after).
struct AsymOrder {
    C t, sm, sp, smo, spo;
    V prev;
    M live;
};

static inline void asym_init(AsymOrder &o, M act) {
    o.t = o.sm = o.sp = o.smo = o.spo = c_one();
    o.prev = set1(1e300);
    o.live = act;
}

static inline void asym_step(AsymOrder &o, V mult_re, C iz, V sgn, bool want_plus,
                             V eps) {
    const C t = cmul(o.t, cscale(mult_re, iz));
    const V a = norm2(t);
    const M dv = m_and(o.live, ge(a, o.prev));
    if (any(dv)) {
        o.smo = cblend(o.smo, o.sm, dv);
        if (want_plus) o.spo = cblend(o.spo, o.sp, dv);
        o.live = m_andnot(o.live, dv);
    }
    o.t = t;
    o.sm = cadd(o.sm, t);
    if (want_plus) o.sp = cmul_add(t, sgn, o.sp);
    o.prev = a;
    const M stop = m_and(o.live, le(a, mul(eps, norm2(o.sm))));
    if (any(stop)) {
        o.smo = cblend(o.smo, o.sm, stop);
        if (want_plus) o.spo = cblend(o.spo, o.sp, stop);
        o.live = m_andnot(o.live, stop);
    }
}

static inline void asym_finish(AsymOrder &o, bool want_plus) {
    if (any(o.live)) {
        o.smo = cblend(o.smo, o.sm, o.live);
        if (want_plus) o.spo = cblend(o.spo, o.sp, o.live);
    }
}

// somm::hankel_asym_sums on the lanes of `act`.
static inline void hankel_asym_sums(C z, M act, bool want_plus, C &s0m, C &s1m,
                                    C &s0p, C &s1p) {
    const C iz = cdiv(cset1(-CI), z);
    AsymOrder o0, o1;
    asym_init(o0, act);
    asym_init(o1, act);
    const V eps = set1(SER_EPS2);
    for (int k = 0; k < 40 && any(m_or(o0.live, o1.live)); ++k) {
        const double odd = double(2 * k + 1);
        const double odd2 = odd * odd;
        const double den = 8.0 * double(k + 1);
        const V sgn = set1((k & 1) ? 1.0 : -1.0);
        if (any(o0.live)) asym_step(o0, set1(-odd2 / den), iz, sgn, want_plus, eps);
        if (any(o1.live)) asym_step(o1, set1((4.0 - odd2) / den), iz, sgn, want_plus, eps);
    }
    asym_finish(o0, want_plus);
    asym_finish(o1, want_plus);
    s0m = o0.smo;
    s1m = o1.smo;
    s0p = o0.spo;
    s1p = o1.spo;
}

// somm::bessel_j0_j1x.
static inline void bessel_j0_j1x(C x, C &b0, C &b1x) {
    const M ser = le(norm2(x), set1(BESSEL_SWITCH2));
    const int sb = bits(ser);
    C b0s{}, b1s{};
    if (sb) j01_series(x, ser, b0s, b1s);
    if (sb == ALL) {
        b0 = b0s;
        b1x = b1s;
        return;
    }
    const M asy = m_not(ser);
    const int ab = bits(asy);
    C s0m, s1m, s0p, s1p;
    hankel_asym_sums(x, asy, true, s0m, s1m, s0p, s1p);
    const C w = csqrt_libm(cdiv(2.0, cscale(SPI, x)), ab);
    const C e = cexp(cmul(cset1(-CI), csubr(x, 0.25 * SPI)), ab);
    const C p = cmul(w, e), pinv = cdiv(w, e);
    const C b0a = cscale(0.5, cmul_add(pinv, s0p, cmul(p, s0m)));
    const C b1a = cdiv(cmul(cset1(0.5 * CI), csub(cmul(p, s1m), cmul(pinv, s1p))), x);
    b0 = sb ? cblend(b0a, b0s, ser) : b0a;
    b1x = sb ? cblend(b1a, b1s, ser) : b1a;
}

// somm::hankel2_half.
static inline void hankel2_half(C x, C &b0, C &b1x) {
    const M ser = le(norm2(x), set1(BESSEL_SWITCH2));
    const int sb = bits(ser);
    C b0s{}, b1s{};
    if (sb) {
        C j0, j1x, y0, y1;
        jy01_series(x, ser, j0, j1x, y0, y1);
        const C ci = cset1(CI);
        b0s = cscale(0.5, csub(j0, cmul(ci, y0)));
        b1s = cdiv(cscale(0.5, csub(cmul(j1x, x), cmul(ci, y1))), x);
    }
    if (sb == ALL) {
        b0 = b0s;
        b1x = b1s;
        return;
    }
    const M asy = m_not(ser);
    const int ab = bits(asy);
    C s0m, s1m, s0p, s1p;
    hankel_asym_sums(x, asy, false, s0m, s1m, s0p, s1p);
    const C p = cmul(csqrt_libm(cdiv(2.0, cscale(SPI, x)), ab),
                     cexp(cmul(cset1(-CI), csubr(x, 0.25 * SPI)), ab));
    const C b0a = cmul(cscale(0.5, p), s0m);
    const C b1a = cdiv(cmul(cmul(cset1(0.5 * CI), p), s1m), x);
    b0 = sb ? cblend(b0a, b0s, ser) : b0a;
    b1x = sb ? cblend(b1a, b1s, ser) : b1a;
}

// somm::integrand_six at W nodes. k1s, k2s are mw_fma::mul(k, k), formed
// once by the caller exactly as the scalar integrand forms them per node.
static inline void integrand_six(double rho, double h, const cd &k1, const cd &k2,
                                 const cd &k1s, const cd &k2s, bool bessel, C lam,
                                 C out[6]) {
    const C g1 = gamma_cut(lam, k1);
    const C g2 = gamma_cut(lam, k2);
    const C g2ks = cmul(g2, cset1(k1s + k2s));
    const C d1 = csub(cdiv(2.0, cadd(g1, g2)), cdiv(cset1(2.0 * k2s), g2ks));
    const C d2 = csub(cdiv(2.0, cmul_add(cset1(k1s), g2, cmul(cset1(k2s), g1))),
                      cdiv(2.0, g2ks));
    const C e = cexp(cscale(h, cneg(g2)));
    const C x = cscale(rho, lam);
    C b0, b1x;
    if (bessel)
        bessel_j0_j1x(x, b0, b1x);
    else
        hankel2_half(x, b0, b1x);
    const C l2 = cmul(lam, lam);
    const C l3 = cmul(l2, lam);
    const C common = cmul(d2, e);
    const C cg2 = cmul(common, g2);
    out[0] = cmul(cmul(common, csub(b1x, b0)), l3);
    out[1] = cmul(cmul(cmul(cg2, g2), b0), lam);
    out[2] = cmul(cmul(cg2, cmul(b1x, x)), l2);
    out[3] = cmul(cmul(cneg(common), b1x), l3);
    out[4] = cmul(cmul(common, b0), lam);
    out[5] = cmul(cmul(cmul(d1, e), b0), lam);
}

// The NQ-node rule of somm::gauss_segment, its integrand values only:
// f[q * 6 + i] is component i at node q, node q's abscissa
// mw_fma::mul_add(half, gx[q], mid) as the scalar loop forms it. NQ is a
// multiple of W. The weighted sum stays with the caller, in its own order.
template <int NQ>
static inline void gauss_values(double rho, double h, const cd &k1, const cd &k2,
                                bool bessel, const cd &mid, const cd &half,
                                const double *gx, cd *f) {
    static_assert(NQ % W == 0, "the rule is walked W nodes at a time");
    const cd k1s = mw_fma::mul(k1, k1);
    const cd k2s = mw_fma::mul(k2, k2);
    const V hr = set1(half.real()), hi = set1(half.imag());
    const V mr = set1(mid.real()), mi = set1(mid.imag());
    for (int g = 0; g < NQ; g += W) {
        const V x = loadu(gx + g);
        const C lam{fma(hr, x, mr), fma(hi, x, mi)};
        C out[6];
        integrand_six(rho, h, k1, k2, k1s, k2s, bessel, lam, out);
        for (int i = 0; i < 6; ++i) {
            const Buf re = spill(out[i].re), im = spill(out[i].im);
            for (int l = 0; l < W; ++l) f[(g + l) * 6 + i] = cd(re.v[l], im.v[l]);
        }
    }
}

}  // namespace somm_six_lanes
