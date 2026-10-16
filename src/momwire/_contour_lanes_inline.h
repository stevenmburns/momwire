// The contour engine's tail integrands in lanes (momwire#1441 step 3).
//
// The below/below tail (`mw_contour::tail_below` -> `TailWrap<SixBelow>`) is
// 84 % of a buried table fill, and it runs along the REAL lambda axis, so its
// Bessel pair takes `mw_contour::bessel_j0_j1x`'s real path: the templated
// series / Miller / asymptotic kernels at T = double. This header is their
// lane twin and `SixBelow`'s, four tail nodes at a time, each lane the scalar
// walk's operations in its order (`_somm_lanes_inline.h` gives the contract).
//
//   * the regime is chosen per lane (|x|^2 < 64 series, < 625 Miller, else
//     the asymptotic P/Q), and a group that straddles a switch runs each
//     regime it needs on that regime's lanes only;
//   * every truncating loop runs until its last lane stops, a stopped lane
//     keeping its sums through a blend;
//   * Miller starts each lane at its own order M = 2 floor(0.75|x| + 16): the
//     lanes count down together from the largest M and a lane joins when the
//     count reaches its own;
//   * cos, sin and cexp are libm's, per lane (the one thing a lane cannot do
//     itself without changing bits);
//   * a node with |x|^2 < 1e-12 (rho = 0) takes the scalar `bessel_j0_j1x`.
//
// `tests/test_below_lanes_1441.py` holds the lanes and the walk equal as
// uint64, every recorded node and every regime edge.
#pragma once

#include <limits>

#include "_contour_engine_inline.h"
#include "_somm_lanes_inline.h"

namespace mw_contour_lanes {

using namespace mw_sl;

// The truncating loops run every lane until the last one stops; a lane's
// sums keep moving after its own stop, and what it returns is what it held
// AT its stop, saved on that iteration (a blend per stop event rather than
// per iteration; the lanes the term limit ends are saved after the loop).

// mw_contour::mw_bessel_series<double> on the lanes of `act`.
static inline void series(V z, M act, V &j0, V &j1) {
    const V w = mul(set1(-0.25), mul(z, z));
    V t0 = set1(1.0), s0 = set1(1.0);
    V t1 = set1(0.5), s1 = set1(0.5);
    V p0 = set1(1.0), p1 = set1(0.25);
    V s0o = s0, s1o = s1;
    const V tiny = set1(1e-38);
    for (int m = 1; m < 80 && any(act); ++m) {
        const double rm = mw_contour::MW_INV[m];
        t0 = mul(t0, mul(w, set1(rm * rm)));
        t1 = mul(t1, mul(w, set1(rm * mw_contour::MW_INV[m + 1])));
        s0 = add(s0, t0);
        s1 = add(s1, t1);
        const V a0 = mul(t0, t0), a1 = mul(t1, t1);
        p0 = vmax(a0, p0);
        p1 = vmax(a1, p1);
        const M stop =
            m_and(act, m_and(le(a0, mul(tiny, p0)), le(a1, mul(tiny, p1))));
        if (any(stop)) {
            s0o = blend(s0o, s0, stop);
            s1o = blend(s1o, s1, stop);
            act = m_andnot(act, stop);
        }
    }
    if (any(act)) {
        s0o = blend(s0o, s0, act);
        s1o = blend(s1o, s1, act);
    }
    j0 = s0o;
    j1 = mul(s1o, z);
}

// Miller's lanes all count down from the largest order M in the group, a lane
// joining when the count reaches its own M; until then it must hold the
// initial values, so the state moves through blends only while some active
// lane has not joined (n > the smallest active M). Lanes outside `act` are
// not read and run free.
static inline int miller_orders(V az, M act, Buf &mb, int &Mmin) {
    const Buf azb = spill(az);
    const int ab = bits(act);
    int Mmax = 0;
    Mmin = 1 << 30;
    for (int l = 0; l < W; ++l) {
        const int m = (ab & (1 << l)) ? 2 * static_cast<int>(0.75 * azb.v[l] + 16.0) : 0;
        mb.v[l] = static_cast<double>(m);
        if (m > Mmax) Mmax = m;
        if ((ab & (1 << l)) && m < Mmin) Mmin = m;
    }
    return Mmax;
}

// mw_contour::mw_bessel_miller<double> on the lanes of `act`, az = sqrt(nz).
static inline void miller(V z, V az, M act, V &j0, V &j1) {
    Buf mb;
    int Mmin;
    const int Mmax = miller_orders(az, act, mb, Mmin);
    const V Mv = loadu(mb.v);
    const V inv2z = div(set1(2.0), z);
    V fp1 = set1(0.0), fn = set1(1e-280);
    V s = set1(0.0), j0v = set1(0.0), j1v = set1(0.0);
    const V big = set1(1e250), sc = set1(1e-250);
    for (int n = Mmax; n >= 1; --n) {
        const V fm1 = fma(mul(set1(static_cast<double>(n)), inv2z), fn, neg(fp1));
        if (n > Mmin) {
            // The lanes whose own count has reached n.
            const M live = le(set1(static_cast<double>(n)), Mv);
            if ((n & 1) == 0) s = blend(s, add(s, mul(set1(2.0), fn)), live);
            fp1 = blend(fp1, fn, live);
            fn = blend(fn, fm1, live);
        } else {
            if ((n & 1) == 0) s = add(s, mul(set1(2.0), fn));
            fp1 = fn;
            fn = fm1;
        }
        if (n == 2) j1v = fn;
        if (n == 1) j0v = fn;
        const M resc = gt(fabs(fn), big);
        if (any(resc)) {
            // A lane that has not joined holds fn = 1e-280, never past `big`.
            fn = blend(fn, mul(fn, sc), resc);
            fp1 = blend(fp1, mul(fp1, sc), resc);
            s = blend(s, mul(s, sc), resc);
            j1v = blend(j1v, mul(j1v, sc), resc);
            j0v = blend(j0v, mul(j0v, sc), resc);
        }
    }
    s = add(s, j0v);
    const V inv_s = div(set1(1.0), s);
    j0 = mul(j0v, inv_s);
    j1 = mul(j1v, inv_s);
}

// mw_contour::mw_bessel_pq<double> on the lanes of `act`. A lane whose term
// grows stops before adding it; one whose term is negligible, after.
static inline void pq(V z, double mu, M act, V &Po, V &Qo) {
    V P = set1(1.0), Q = set1(0.0);
    Po = P;
    Qo = Q;
    V c = set1(1.0);
    const V iz = div(set1(1.0), z);
    V prev = set1(std::numeric_limits<double>::infinity());
    const V tiny = set1(1e-40);
    for (int k = 1; k <= 40 && any(act); ++k) {
        const double kk = 2.0 * k - 1.0;
        c = mul(c, mul(set1((mu - kk * kk) * 0.125 * mw_contour::MW_INV[k]), iz));
        const V m = mul(c, c);
        // `if (!(m < prev)) break;` -- NaN stops too.
        const M brk = m_andnot(act, lt(m, prev));
        if (any(brk)) {
            Po = blend(Po, P, brk);
            Qo = blend(Qo, Q, brk);
            act = m_andnot(act, brk);
        }
        prev = m;
        if (k & 1) {
            if ((((k - 1) / 2) & 1) == 0) Q = add(Q, c);
            else Q = sub(Q, c);
        } else {
            if (((k / 2) & 1) == 0) P = add(P, c);
            else P = sub(P, c);
        }
        const M stop = m_and(act, lt(m, tiny));
        if (any(stop)) {
            Po = blend(Po, P, stop);
            Qo = blend(Qo, Q, stop);
            act = m_andnot(act, stop);
        }
    }
    if (any(act)) {
        Po = blend(Po, P, act);
        Qo = blend(Qo, Q, act);
    }
}

MW_SL_INLINE V cos_libm(V x, int act) {
    Buf r = spill(x);
    for (int l = 0; l < W; ++l)
        if (act & (1 << l)) r.v[l] = mw_fma::opaque(std::cos(mw_fma::opaque(r.v[l])));
    return loadu(r.v);
}
MW_SL_INLINE V sin_libm(V x, int act) {
    Buf r = spill(x);
    for (int l = 0; l < W; ++l)
        if (act & (1 << l)) r.v[l] = mw_fma::opaque(std::sin(mw_fma::opaque(r.v[l])));
    return loadu(r.v);
}

// mw_contour::mw_bessel_asymptotic<double> on the lanes of `act`.
//
// J1 = false (momwire#1512) is the same J0 without the order-1 P/Q sums and
// J1's own line: nothing J0 reads comes from them, so J0 is unchanged bit for
// bit, and a caller that discards J1 (`mw899::bessel_j0_many`) stops paying
// for it. j1 is left unwritten then.
template <bool J1 = true>
static inline void asymptotic(V z, M act, V &j0, V &j1) {
    const int ab = bits(act);
    V P0, Q0, P1, Q1;
    pq(z, 0.0, act, P0, Q0);
    if (J1) pq(z, 4.0, act, P1, Q1);
    const V cz = cos_libm(z, ab);
    const V sz = sin_libm(z, ab);
    const V pref = sqrt(div(set1(1.0), mul(set1(mw_contour::MW_PI), z)));
    j0 = mul(pref, fma(add(P0, Q0), cz, mul(sub(P0, Q0), sz)));
    if (J1) j1 = mul(pref, fma(add(P1, Q1), sz, mul(sub(Q1, P1), cz)));
}

// mw_contour::bessel_j0_j1x at REAL arguments x = (xr, +0): the tail's.
// `xr` is the real part of x = lam * rho; nx = mw_norm2(x), as the scalar
// forms it from the complex x. Small lanes (see below) return garbage.
template <bool J1 = true>
static inline void bessel_j0_j1x_real(V xr, V nx, V &j0, V &j1x) {
    const M small = lt(nx, set1(1e-12));
    const V ax = fabs(xr);
    const M ser = m_andnot(lt(nx, set1(64.0)), small);
    const M mil = m_andnot(m_andnot(lt(nx, set1(625.0)), small), ser);
    const M asy = m_andnot(m_not(m_or(ser, mil)), small);
    V b0 = set1(0.0), b1 = set1(0.0);
    if (any(ser)) {
        V a, b;
        series(ax, ser, a, b);
        b0 = blend(b0, a, ser);
        b1 = blend(b1, b, ser);
    }
    if (any(mil)) {
        V a, b;
        miller(ax, sqrt(nx), mil, a, b);
        b0 = blend(b0, a, mil);
        b1 = blend(b1, b, mil);
    }
    if (any(asy)) {
        V a, b;
        asymptotic<J1>(ax, asy, a, b);
        b0 = blend(b0, a, asy);
        if (J1) b1 = blend(b1, b, asy);
    }
    j0 = b0;
    if (J1) j1x = div(b1, ax);
    // A lane with |x|^2 < 1e-12 is the scalar's small branch, whose values
    // are complex in general: `six_below_tail` reports it and its caller
    // recomputes that node with the scalar integrand.
}

// ---- the complex path (the head's detour) --------------------------------

// mw_contour::mw_bessel_series<cd> on the lanes of `act`.
static inline void series_c(C z, M act, C &j0, C &j1) {
    const C w = cscale(-0.25, cmul(z, z));
    C t0{set1(1.0), set1(0.0)}, s0 = t0;
    C t1{set1(0.5), set1(0.0)}, s1 = t1;
    C s0o = s0, s1o = s1;
    V p0 = set1(1.0), p1 = set1(0.25);
    const V tiny = set1(1e-38);
    for (int m = 1; m < 80 && any(act); ++m) {
        const double rm = mw_contour::MW_INV[m];
        t0 = cmul(t0, cscale(rm * rm, w));
        t1 = cmul(t1, cscale(rm * mw_contour::MW_INV[m + 1], w));
        s0 = cadd(s0, t0);
        s1 = cadd(s1, t1);
        const V a0 = norm2(t0), a1 = norm2(t1);
        p0 = vmax(a0, p0);
        p1 = vmax(a1, p1);
        const M stop =
            m_and(act, m_and(le(a0, mul(tiny, p0)), le(a1, mul(tiny, p1))));
        if (any(stop)) {
            s0o = cblend(s0o, s0, stop);
            s1o = cblend(s1o, s1, stop);
            act = m_andnot(act, stop);
        }
    }
    if (any(act)) {
        s0o = cblend(s0o, s0, act);
        s1o = cblend(s1o, s1, act);
    }
    j0 = s0o;
    j1 = cmul(s1o, z);
}

// mw_contour::mw_bessel_miller<cd> on the lanes of `act`, az = sqrt(nz).
static inline void miller_c(C z, V az, M act, C &j0, C &j1) {
    Buf mb;
    int Mmin;
    const int Mmax = miller_orders(az, act, mb, Mmin);
    const V Mv = loadu(mb.v);
    const C inv2z = cdiv(2.0, z);
    const C zero{set1(0.0), set1(0.0)};
    C fp1 = zero, fn{set1(1e-280), set1(0.0)};
    C s = zero, j0v = zero, j1v = zero;
    const V big = set1(1e250), sc = set1(1e-250);
    for (int n = Mmax; n >= 1; --n) {
        const C fm1 = cmul_sub(cscale(static_cast<double>(n), inv2z), fn, fp1);
        if (n > Mmin) {
            const M live = le(set1(static_cast<double>(n)), Mv);
            if ((n & 1) == 0) s = cblend(s, cadd(s, cscale(2.0, fn)), live);
            fp1 = cblend(fp1, fn, live);
            fn = cblend(fn, fm1, live);
        } else {
            if ((n & 1) == 0) s = cadd(s, cscale(2.0, fn));
            fp1 = fn;
            fn = fm1;
        }
        if (n == 2) j1v = fn;
        if (n == 1) j0v = fn;
        // mw_mag: std::max(|re|, |im|), i.e. (|re| < |im|) ? |im| : |re|.
        const V ar = fabs(fn.re), ai = fabs(fn.im);
        const V mag = blend(ar, ai, lt(ar, ai));
        const M resc = gt(mag, big);
        if (any(resc)) {
            fn = cblend(fn, cscale(sc, fn), resc);
            fp1 = cblend(fp1, cscale(sc, fp1), resc);
            s = cblend(s, cscale(sc, s), resc);
            j1v = cblend(j1v, cscale(sc, j1v), resc);
            j0v = cblend(j0v, cscale(sc, j0v), resc);
        }
    }
    s = cadd(s, j0v);
    const C inv_s = cdiv(1.0, s);
    j0 = cmul(j0v, inv_s);
    j1 = cmul(j1v, inv_s);
}

// mw_contour::mw_bessel_pq<cd> on the lanes of `act`.
static inline void pq_c(C z, double mu, M act, C &Po, C &Qo) {
    C P{set1(1.0), set1(0.0)}, Q{set1(0.0), set1(0.0)};
    Po = P;
    Qo = Q;
    C c = P;
    const C iz = cdiv(1.0, z);
    V prev = set1(std::numeric_limits<double>::infinity());
    const V tiny = set1(1e-40);
    for (int k = 1; k <= 40 && any(act); ++k) {
        const double kk = 2.0 * k - 1.0;
        c = cmul(c, cscale((mu - kk * kk) * 0.125 * mw_contour::MW_INV[k], iz));
        const V m = norm2(c);
        const M brk = m_andnot(act, lt(m, prev));
        if (any(brk)) {
            Po = cblend(Po, P, brk);
            Qo = cblend(Qo, Q, brk);
            act = m_andnot(act, brk);
        }
        prev = m;
        if (k & 1) {
            if ((((k - 1) / 2) & 1) == 0) Q = cadd(Q, c);
            else Q = csub(Q, c);
        } else {
            if (((k / 2) & 1) == 0) P = cadd(P, c);
            else P = csub(P, c);
        }
        const M stop = m_and(act, lt(m, tiny));
        if (any(stop)) {
            Po = cblend(Po, P, stop);
            Qo = cblend(Qo, Q, stop);
            act = m_andnot(act, stop);
        }
    }
    if (any(act)) {
        Po = cblend(Po, P, act);
        Qo = cblend(Qo, Q, act);
    }
}

struct CcosF {
    cd operator()(const cd &v) const { return std::cos(v); }
};
struct CsinF {
    cd operator()(const cd &v) const { return std::sin(v); }
};

// mw_contour::mw_bessel_asymptotic<cd> on the lanes of `act`.
// J1 = false: `asymptotic`'s note.
template <bool J1 = true>
static inline void asymptotic_c(C z, M act, C &j0, C &j1) {
    const int ab = bits(act);
    C P0, Q0, P1, Q1;
    pq_c(z, 0.0, act, P0, Q0);
    if (J1) pq_c(z, 4.0, act, P1, Q1);
    const C cz = per_lane(z, CcosF(), ab);
    const C sz = per_lane(z, CsinF(), ab);
    const C pref = csqrt_libm(cdiv(1.0, cscale(mw_contour::MW_PI, z)), ab);
    j0 = cmul(pref, cmul_add(cadd(P0, Q0), cz, cmul(csub(P0, Q0), sz)));
    if (J1) j1 = cmul(pref, cmul_add(cadd(P1, Q1), sz, cmul(csub(Q1, P1), cz)));
}

// mw_contour::bessel_j0_j1x at complex x with Im x != 0 and |x|^2 >= 1e-12
// (the caller routes the other lanes to the scalar function).
template <bool J1 = true>
static inline void bessel_j0_j1x_complex(C x, V nx, C &j0, C &j1x) {
    const C z = cblend(x, cneg(x), lt(x.re, set1(0.0)));
    const M ser = lt(nx, set1(64.0));
    const M mil = m_andnot(lt(nx, set1(625.0)), ser);
    const M asy = m_not(m_or(ser, mil));
    C b0{set1(0.0), set1(0.0)}, b1 = b0;
    if (any(ser)) {
        C a, b;
        series_c(z, ser, a, b);
        b0 = cblend(b0, a, ser);
        b1 = cblend(b1, b, ser);
    }
    if (any(mil)) {
        C a, b;
        miller_c(z, sqrt(nx), mil, a, b);
        b0 = cblend(b0, a, mil);
        b1 = cblend(b1, b, mil);
    }
    if (any(asy)) {
        C a, b;
        asymptotic_c<J1>(z, asy, a, b);
        b0 = cblend(b0, a, asy);
        if (J1) b1 = cblend(b1, b, asy);
    }
    j0 = b0;
    if (J1) j1x = cdiv(b1, z);
}

// mw_contour::bessel_j0_j1x at W arbitrary arguments x[0..W): the real path
// when every lane is on the axis, the complex path when none is, and the
// scalar function itself on any lane neither serves (|x| < 1e-6, or a mixed
// group's on-axis lanes).
static inline void bessel_j0_j1x_any(const cd *x, cd *j0, cd *j1x) {
    const C xv = cload(x);
    const V nx = norm2(xv);
    const int axis = bits(eq(xv.im, set1(0.0)));
    int left = bits(lt(nx, set1(1e-12)));
    if (axis == ALL) {
        V a, b;
        bessel_j0_j1x_real(xv.re, nx, a, b);
        const Buf ab = spill(a), bb = spill(b);
        for (int l = 0; l < W; ++l) {
            j0[l] = cd(ab.v[l], 0.0);
            j1x[l] = cd(bb.v[l], 0.0);
        }
    } else {
        left |= axis;
        C a, b;
        bessel_j0_j1x_complex(xv, nx, a, b);
        for (int l = 0; l < W; ++l) {
            j0[l] = lane(a, l);
            j1x[l] = lane(b, l);
        }
    }
    for (int l = 0; l < W; ++l)
        if (left & (1 << l)) mw_contour::bessel_j0_j1x(x[l], j0[l], j1x[l]);
}

// `bessel_j0_j1x_any`'s J0 alone (momwire#1512): the same regimes, the same
// lanes and the same scalar fallback, the order-1 work (`asymptotic`'s J1 =
// false, the J1/x divides) skipped. Every j0 is the one the pair returns.
static inline void bessel_j0_any(const cd *x, cd *j0) {
    const C xv = cload(x);
    const V nx = norm2(xv);
    const int axis = bits(eq(xv.im, set1(0.0)));
    int left = bits(lt(nx, set1(1e-12)));
    if (axis == ALL) {
        V a, b;
        bessel_j0_j1x_real<false>(xv.re, nx, a, b);
        const Buf ab = spill(a);
        for (int l = 0; l < W; ++l) j0[l] = cd(ab.v[l], 0.0);
    } else {
        left |= axis;
        C a, b;
        bessel_j0_j1x_complex<false>(xv, nx, a, b);
        const Buf ar = spill(a.re), ai = spill(a.im);
        for (int l = 0; l < W; ++l) j0[l] = cd(ar.v[l], ai.v[l]);
    }
    for (int l = 0; l < W; ++l)
        if (left & (1 << l)) {
            cd b1x;
            mw_contour::bessel_j0_j1x(x[l], j0[l], b1x);
        }
}

// `mw568_below::SixBelow::operator()` at W nodes lam: the d12 pair, the decay
// exponential and the Bessel pair, then the six products, all exactly as the
// scalar spells them. `real_axis` says every lam is (t, +0), the tail's
// abscissa, whose Bessel pair takes the real path; otherwise every lam is off
// the axis (the head's detour) and takes the complex path. The return value
// has bit l set for a lane the lanes did not serve -- the Bessel pair's
// |x| < 1e-6 branch (rho = 0), or an off-axis lane whose x landed on the axis
// -- which the caller recomputes with the scalar integrand.
static inline int six_below(double rho, double h, const cd &k_p, const cd &k_m,
                            C lam, bool real_axis, C out[6]) {
    // d12(lam, k_p, k_m) -- _accel_mw568.cpp
    const C g1 = gamma_cut(lam, k_p);
    const C g2 = gamma_cut(lam, k_m);
    const cd k1s = mw_fma::mul(k_p, k_p);
    const cd k2s = mw_fma::mul(k_m, k_m);
    const C g2ks = cmul(g2, cset1(k1s + k2s));
    const C d1 = csub(cdiv(2.0, cadd(g1, g2)), cdiv(cset1(2.0 * k2s), g2ks));
    const C d2 = csub(cdiv(2.0, cmul_add(cset1(k1s), g2, cmul(cset1(k2s), g1))),
                      cdiv(2.0, g2ks));
    const C e = cexp(cscale(h, cneg(g2)));
    const C x = cscale(rho, lam);
    const V nx = norm2(x);
    int fallback = bits(lt(nx, set1(1e-12)));
    C b0, b1x;
    if (real_axis) {
        V j0r, j1r;
        bessel_j0_j1x_real(x.re, nx, j0r, j1r);
        b0 = C{j0r, set1(0.0)};
        b1x = C{j1r, set1(0.0)};
    } else {
        fallback |= bits(eq(x.im, set1(0.0)));
        bessel_j0_j1x_complex(x, nx, b0, b1x);
    }
    const C l2 = cmul(lam, lam);
    const C l3 = cmul(l2, lam);
    const C common = cmul(d2, e);
    const C cg = cmul(common, g2);
    out[0] = cmul(cmul(common, csub(b1x, b0)), l3);
    out[1] = cmul(cmul(cmul(cg, g2), b0), lam);
    out[2] = cneg(cmul(cmul(cg, cmul(b1x, x)), l2));
    out[3] = cneg(cmul(cmul(common, b1x), l3));
    out[4] = cmul(cmul(common, b0), lam);
    out[5] = cmul(cmul(cmul(d1, e), b0), lam);
    return fallback;
}

}  // namespace mw_contour_lanes
