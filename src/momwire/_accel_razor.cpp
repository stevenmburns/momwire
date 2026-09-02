#include "_accel_common.h"
#include "_stable_inline.h"

// razor section of the accelerator (momwire#742): the razor-blade
// formulation's segment-moment fill, fused and tiled.
//
// WHAT IS COMPUTED
// ----------------
// For every observation point p and every source segment s,
//
//     M0[p, s] = ∫₀^h dτ g(r_p, r_s(τ)),   M1[p, s] = ∫₀^h τ dτ g(r_p, r_s(τ))
//
// with g the reduced thin-wire kernel exp(−jkR)/(4πR) INCLUDING the 1/4π,
// R = sqrt(|r_p − r_s(τ)|² + a²), and τ the source's local arc from its
// segment start. Both integrals are split the way `_kernel_moments` splits
// them and for the same reason: the static part 1/(4πR) integrates in closed
// form in the segment's axis frame (asinh and sqrt), and what is left,
// (exp(−jkR) − 1)/(4πR), is smooth everywhere the reduced kernel is defined
// and takes plain Gauss–Legendre on the caller's `xg`/`wg`.
//
// WHY IT IS ONE KERNEL AND NOT TWO
// --------------------------------
// The Python pair this replaces is split at the wavenumber:
// `_seg_moments_prepare` builds the k-independent half (the axis frame, the
// closed-form statics, and the (n_obs, n_seg, n_qp) distance table R) and
// `_seg_moments_from_prepared` finishes it at one k. That split is a CACHE of
// R across a swept solve, and it is exactly what makes the fill 52× the size
// of the matrix it builds: R alone is n_obs·n_seg·n_qp doubles and is
// RETAINED for every chunk at once. Fusing the two halves here trades that
// cache for residency — R is formed one scalar at a time in a register, used,
// and dropped — so the kernel's transient is its own output and nothing else.
// A swept solve pays the statics again at every k (measured at ~16 % of a
// swept point by momwire#398's own note in `_assemble_Z_prepare`); at 1601
// segments it buys back 1.7 GB.
//
// LAYOUT AND TILING
// -----------------
// Output is row-major (n_obs, n_seg) complex128 — one row per observer, the
// same array `_seg_moments` has always returned, so nothing downstream moves.
// Entries are INDEPENDENT: there is no reduction across pairs, so the tiling
// here is for residency, not for accumulation. The loop nest is
//
//     for each obs tile (OBS_TILE rows)          <- OpenMP, disjoint rows
//         for each segment tile (SEG_TILE cols)  <- geometry gathered once
//             for p in tile, for s in tile, for q in [0, n_qp)
//
// The segment tile's per-segment scalars (origin, tangent, length, radius)
// and its quadrature tables τ[s][q], w[s][q], τ·w[s][q] are gathered into
// small contiguous buffers once per tile and re-read by all OBS_TILE rows,
// so a fill at any N walks L1-resident source geometry instead of streaming
// an O(N²·n_qp) plane. OBS_TILE·SEG_TILE·n_qp doubles of tile scratch is a
// few tens of KB at the shipped orders; the tile constants below are sized
// for that and nothing about the answer depends on them.
//
// THE EXTENDED KERNEL (momwire#398 D1)
// ------------------------------------
// `group_i` / `group_j` carry the coaxial-equal-radius eligibility labels of
// `_ek_pair_mask`: pair (p, s) extends iff `group_i[p] == group_j[s] >= 0`.
// Eligibility is a property of the PAIR and is therefore resolved once per
// (p, s), outside the quadrature loop — eligible pairs take
// `_static_axis_moments_ek`'s closed form and NEC Eq 89's coaxial factor on
// the remainder, ineligible ones take the reduced forms, and a build with
// empty label arrays never enters the EK arithmetic at all. Both group arrays
// empty means "reduced kernel", which is what the caller spells as `ek=None`.
//
// ARITHMETIC ORDER
// ----------------
// Every expression below is a term-for-term transcription of its numpy
// counterpart in `_kernel_moments` and `_bspline_kernels`, in the same
// multi-step spelling (momwire#205) — including `num / R` as a genuine
// divide rather than a multiply by a reciprocal, and the ρ² − a² grouping in
// the EK statics whose whole purpose is that it is exactly 0.0 on an
// on-axis pair. What is NOT reproduced is the reduction ORDER of numpy's
// `einsum("psq,sq->ps")`, so agreement with the reference path is tight but
// not bitwise; the repo's standard (never pin cross-build bit equality)
// applies, and `tests/test_razor_fill_accel_742.py` measures the deviation
// rather than asserting it away.

// Tile geometry. OBS_TILE rows of output times SEG_TILE columns is the block
// held live while one segment tile's gathered geometry is in L1; SEG_TILE
// also sizes the per-tile quadrature scratch (3 · SEG_TILE · n_qp doubles).
// At the shipped orders (n_qp_source = 12) that is ~36 KB of scratch and
// ~32 KB of touched output per tile.
static constexpr size_t RAZOR_OBS_TILE = 8;
static constexpr size_t RAZOR_SEG_TILE = 128;

// The reduced kernel's closed-form static moments — `_static_axis_moments`,
// scalar. `rho2` already carries the a² of the axis frame.
static inline void razor_statics_reduced(double u_r, double rho2, double h,
                                         double *m0, double *m1) {
    const double u0 = -u_r;
    const double u1 = h - u_r;
    const double r0 = std::sqrt(u0 * u0 + rho2);
    const double r1 = std::sqrt(u1 * u1 + rho2);
    // Both differences cancellation-free (momwire#799), `_stable_inline.h`
    // and its numpy twin term for term. A far observer's u1 and u0 agree to
    // h/|u_r|, which cost the literal asinh difference 1.2e-12 relative on an
    // 801-segment thin dipole and moved the solved Z by 2.6e-11.
    const double mm0 = stable_asinh_diff(u0, u1, rho2, r0, r1);
    *m0 = mm0;
    *m1 = u_r * mm0 + stable_sqrt_diff(u0, u1, r0, r1);
}

// The extended (tubular) kernel's closed-form static moments —
// `_static_axis_moments_ek`, scalar. `perp2 = rho2 − a²` is formed as the
// single subtraction the numpy spelling forms it as: on an eligible pair the
// observer sits on the source's own axis, so ρ = a and this is exactly 0.0,
// which is what keeps the two 1/R terms from cancelling catastrophically.
static inline void razor_statics_ek(double u_r, double rho2, double h,
                                    double a_ek, double *m0, double *m1) {
    const double a2 = a_ek * a_ek;
    const double a4 = a2 * a2;
    const double perp2 = rho2 - a2;
    const double u0 = -u_r;
    const double u1 = h - u_r;
    const double r0 = std::sqrt(u0 * u0 + rho2);
    const double r1 = std::sqrt(u1 * u1 + rho2);
    const double c3 = 0.25 * a4 / rho2;
    const double c1 = 0.5 * a2 * perp2 / (rho2 * rho2);
    // P and Q differenced TERM BY TERM, so the leading asinh and R differences
    // take their stable spellings and the two O(a²) corrections — which carry
    // an explicit a²/R² and lose that much less — stay literal (momwire#799).
    double mm0 = stable_asinh_diff(u0, u1, rho2, r0, r1);
    mm0 = mm0 + c3 * (u1 / (r1 * r1 * r1) - u0 / (r0 * r0 * r0));
    mm0 = mm0 - c1 * (u1 / r1 - u0 / r0);
    double qd = stable_sqrt_diff(u0, u1, r0, r1);
    qd = qd + 0.5 * a2 * (1.0 / r1 - 1.0 / r0);
    qd = qd - 0.25 * a4 * (1.0 / (r1 * r1 * r1) - 1.0 / (r0 * r0 * r0));
    *m0 = mm0;
    *m1 = u_r * mm0 + qd;
}

// Fused segment moments at one wavenumber. Returns (M0, M1); M1 is a (0, 0)
// array when `need_m1` is false, which is how the caller spells "the scalar
// potential only needs M0" without a second entry point.
//
// COMPLEX_K continues this kernel to an in-medium (lossy) wavenumber
// k = k_re + j*k_im with Im k <= 0 (momwire#796) — the same door #778 opened
// on bspline's off-edge kernel, which the block comment above
// `seg_seg_full_moments_bspline_kernel` states in full. The core of it:
//
//     exp(-jkR) = exp(k_im * R) * (cos(k_re*R) - j*sin(k_re*R))
//                 ^^^^^^^^^^^^^ real, monotone in (0, 1] since k_im <= 0
//
// so the trig stays in real lanes and picks up one extra real exp() per point.
// The King & Smith trap #778 records applies here verbatim: substituting |k|
// for a real k is NOT the continuation, and this kernel was exactly that
// spelling while it only ever served real k.
//
// Razor's two branches do not transfer identically, which is the whole of the
// difference from #778:
//
//   - REDUCED pairs integrate (exp(-jkR) - 1)/R. Only the trig is scaled; the
//     `- 1` and the k-free statics are untouched.
//   - EK pairs form C1 = 1 + jkR and C2 = 3*C1 - (kR)^2, which are genuinely
//     complex in kR once k is: jkR contributes -Im(k)*R to the REAL part and
//     (kR)^2 acquires an imaginary part. Those are spelled below term for term
//     and in the same multi-step order as `_ek_factor` / `_ek_reg_extra` in
//     `_bspline_kernels.py`, which are numpy-generic and already correct at
//     complex k. Verified against them to 0.0 relative before this was written.
//
// `if (COMPLEX_K)` rather than `if constexpr`: the build is -std=gnu++11, and
// this is the idiom `if (ek_pair)` already uses here. The branch folds at -O3,
// and the <false> instantiation is textually the pre-#796 loop so the real-k
// output stays bit-identical (proven by rebuild and array_equal, the #762
// protocol, not by reading).
template<bool COMPLEX_K>
static std::pair<py::array_t<std::complex<double>>,
                 py::array_t<std::complex<double>>>
razor_seg_moments_impl(
    py::array_t<double, py::array::c_style | py::array::forcecast> obs,     // (n_obs, 3)
    py::array_t<double, py::array::c_style | py::array::forcecast> seg_p0,  // (n_seg, 3)
    py::array_t<double, py::array::c_style | py::array::forcecast> seg_t,   // (n_seg, 3)
    py::array_t<double, py::array::c_style | py::array::forcecast> seg_h,   // (n_seg,)
    py::array_t<double, py::array::c_style | py::array::forcecast> a,       // (n_seg,)
    py::array_t<double, py::array::c_style | py::array::forcecast> xg,      // (n_qp,)
    py::array_t<double, py::array::c_style | py::array::forcecast> wg,      // (n_qp,)
    double k_re,
    double k_im,
    bool need_m1,
    py::array_t<int64_t, py::array::c_style | py::array::forcecast> group_i, // (n_obs,) or ()
    py::array_t<int64_t, py::array::c_style | py::array::forcecast> group_j, // (n_seg,) or ()
    py::array_t<double, py::array::c_style | py::array::forcecast> a_ek,     // (n_seg,) or ()
    uintptr_t cancel_flag = 0
) {
    const size_t n_obs = (size_t)obs.shape(0);
    const size_t n_seg = (size_t)seg_h.shape(0);
    const size_t n_qp = (size_t)xg.shape(0);

    if (obs.ndim() != 2 || obs.shape(1) != 3)
        throw std::runtime_error("obs must have shape (n_obs, 3)");
    if (seg_p0.ndim() != 2 || seg_p0.shape(1) != 3 ||
        (size_t)seg_p0.shape(0) != n_seg)
        throw std::runtime_error("seg_p0 must have shape (n_seg, 3)");
    if (seg_t.ndim() != 2 || seg_t.shape(1) != 3 ||
        (size_t)seg_t.shape(0) != n_seg)
        throw std::runtime_error("seg_t must have shape (n_seg, 3)");
    if ((size_t)a.shape(0) != n_seg)
        throw std::runtime_error("a must have shape (n_seg,)");
    if ((size_t)wg.shape(0) != n_qp)
        throw std::runtime_error("xg and wg must have the same length");

    // EK is on iff BOTH label arrays are populated — the same "group_i is
    // None or group_j is None means every pair" rule `_ek_pair_mask` states,
    // read the only way an array can spell None.
    const bool ek_on = group_i.size() > 0 && group_j.size() > 0;
    if (ek_on) {
        if ((size_t)group_i.shape(0) != n_obs || (size_t)group_j.shape(0) != n_seg)
            throw std::runtime_error(
                "group_i / group_j must have shapes (n_obs,) / (n_seg,)");
        if ((size_t)a_ek.shape(0) != n_seg)
            throw std::runtime_error("a_ek must have shape (n_seg,) under the EK");
    }

    auto ob = obs.unchecked<2>();
    auto p0 = seg_p0.unchecked<2>();
    auto tg = seg_t.unchecked<2>();
    auto hv = seg_h.unchecked<1>();
    auto av = a.unchecked<1>();
    auto xgv = xg.unchecked<1>();
    auto wgv = wg.unchecked<1>();

    const int64_t *gi = ek_on ? group_i.data() : nullptr;
    const int64_t *gj = ek_on ? group_j.data() : nullptr;
    const double *aek = ek_on ? a_ek.data() : nullptr;

    py::array_t<std::complex<double>> M0({n_obs, n_seg});
    py::array_t<std::complex<double>> M1(
        need_m1 ? std::vector<size_t>{n_obs, n_seg} : std::vector<size_t>{0, 0});
    std::complex<double> *m0out = M0.mutable_data();
    std::complex<double> *m1out = need_m1 ? M1.mutable_data() : nullptr;

    // Phase 0: release the GIL for the heavy compute region below.
    py::gil_scoped_release release;

    const double inv4pi = 1.0 / (4.0 * M_PI);
    const size_t n_obs_tiles = (n_obs + RAZOR_OBS_TILE - 1) / RAZOR_OBS_TILE;

    PYSIM_CANCEL_SETUP(cancel_flag);
#pragma omp parallel
    {
    // Per-segment-tile gathers, allocated ONCE PER THREAD and refilled per
    // tile: the point of the tiling is that these stay hot across the tile's
    // OBS_TILE rows, and an allocation inside the tile loop would put a
    // malloc on the critical path of every tile on every thread (measured:
    // 2.3x thread scaling instead of 3.4x at 1601 segments). Hence the
    // explicit `omp parallel` + `omp for` rather than a combined directive.
    std::vector<double> sp0(RAZOR_SEG_TILE * 3), stan(RAZOR_SEG_TILE * 3);
    std::vector<double> sh(RAZOR_SEG_TILE), sa(RAZOR_SEG_TILE);
    std::vector<double> saek(ek_on ? RAZOR_SEG_TILE : 0);
    std::vector<int64_t> sgj(ek_on ? RAZOR_SEG_TILE : 0);
    std::vector<double> tau(RAZOR_SEG_TILE * n_qp);
    std::vector<double> wq(RAZOR_SEG_TILE * n_qp);
    std::vector<double> twq(RAZOR_SEG_TILE * n_qp);

#pragma omp for schedule(static)
    for (size_t tile = 0; tile < n_obs_tiles; tile++) {
        PYSIM_CANCEL_POLL();
        const size_t p_lo = tile * RAZOR_OBS_TILE;
        const size_t p_hi = std::min(p_lo + RAZOR_OBS_TILE, n_obs);

        for (size_t s_lo = 0; s_lo < n_seg; s_lo += RAZOR_SEG_TILE) {
            const size_t s_hi = std::min(s_lo + RAZOR_SEG_TILE, n_seg);
            const size_t ns = s_hi - s_lo;
            for (size_t js = 0; js < ns; js++) {
                const size_t s = s_lo + js;
                sp0[js * 3 + 0] = p0(s, 0);
                sp0[js * 3 + 1] = p0(s, 1);
                sp0[js * 3 + 2] = p0(s, 2);
                stan[js * 3 + 0] = tg(s, 0);
                stan[js * 3 + 1] = tg(s, 1);
                stan[js * 3 + 2] = tg(s, 2);
                sh[js] = hv(s);
                sa[js] = av(s);
                if (ek_on) {
                    saek[js] = aek[s];
                    sgj[js] = gj[s];
                }
                // tau = (0.5·h)·(1 + xg), wq = (0.5·h)·wg — the association
                // numpy takes in `_seg_moments_from_prepared`, kept so the
                // node positions and weights are the same doubles.
                const double half_h = 0.5 * hv(s);
                for (size_t q = 0; q < n_qp; q++) {
                    const double t = half_h * (1.0 + xgv(q));
                    const double w = half_h * wgv(q);
                    tau[js * n_qp + q] = t;
                    wq[js * n_qp + q] = w;
                    twq[js * n_qp + q] = t * w;
                }
            }

            for (size_t p = p_lo; p < p_hi; p++) {
                const double ox = ob(p, 0), oy = ob(p, 1), oz = ob(p, 2);
                const int64_t gip = ek_on ? gi[p] : -1;
                std::complex<double> *row0 = m0out + p * n_seg;
                std::complex<double> *row1 = need_m1 ? m1out + p * n_seg : nullptr;

                for (size_t js = 0; js < ns; js++) {
                    // Axis frame (`_axis_frame`): the projection of the
                    // observer on this segment's axis, and the squared
                    // perpendicular offset plus a².
                    const double dx = ox - sp0[js * 3 + 0];
                    const double dy = oy - sp0[js * 3 + 1];
                    const double dz = oz - sp0[js * 3 + 2];
                    const double tx = stan[js * 3 + 0];
                    const double ty = stan[js * 3 + 1];
                    const double tz = stan[js * 3 + 2];
                    const double u_r = dx * tx + dy * ty + dz * tz;
                    // |d − (d·t)t|², NOT |d|² − u_r² (momwire#799). The two
                    // are the same number and the second is a cancellation
                    // that is EXACT on a collinear pair, so it returns the
                    // rounding error of |d|² where the answer is 0 — and
                    // whether it returns +0, +2e-14 or −2e-14 depends on
                    // which products the compiler contracts into an FMA.
                    // That is the whole of the 8.2e-13 this kernel and its
                    // numpy twin disagreed by on arm64 while agreeing to
                    // 5e-18 on x86-64. The perpendicular VECTOR is exactly
                    // zero there under any contraction, so both lanes get
                    // 0.0. No clamp: a sum of squares cannot be negative.
                    const double px = dx - u_r * tx;
                    const double py = dy - u_r * ty;
                    const double pz = dz - u_r * tz;
                    const double perp = px * px + py * py + pz * pz;
                    const double aa = sa[js];
                    const double rho2 = perp + aa * aa;

                    const bool ek_pair = ek_on && gip >= 0 && gip == sgj[js];
                    double m0s, m1s;
                    if (ek_pair)
                        razor_statics_ek(u_r, rho2, sh[js], saek[js], &m0s, &m1s);
                    else
                        razor_statics_reduced(u_r, rho2, sh[js], &m0s, &m1s);

                    const double *tq = &tau[js * n_qp];
                    const double *wqq = &wq[js * n_qp];
                    const double *twqq = &twq[js * n_qp];
                    double a0r = 0.0, a0i = 0.0, a1r = 0.0, a1i = 0.0;

                    if (!ek_pair) {
                        // No `omp simd reduction` here, for the same reason as the bspline
                        // kernels (momwire#781): the clause licenses reassociation, so the
                        // reduction tree follows a per-function vectorization choice and
                        // results stop being reproducible across targets. It measured ~4.6%
                        // on razor (vs ~0% on bspline) and was still dropped: momwire#780 is
                        // about to add accelerators here with cross-LANE equality
                        // expectations (nec5 vs Gauss-Legendre sharing one kernel), and that
                        // is exactly the shape this nondeterminism breaks. Revisit if the
                        // 4.6% is ever worth more than reproducibility.
                        for (size_t q = 0; q < n_qp; q++) {
                            const double u = tq[q] - u_r;
                            const double R = std::sqrt(u * u + rho2);
                            const double kr = k_re * R;
                            // exp(−jkR) − 1, then the complex-by-real divide
                            // numpy performs (Smith's algorithm collapses to
                            // a componentwise divide on a real denominator).
                            //
                            // Cancellation-free (momwire#799): the real part
                            // is where the `- 1` bites, and `cos(kr) - 1` is
                            // spelled `-2 sin²(kr/2)`. It costs nothing — the
                            // loop wanted sin(kr) anyway, so cos+sin becomes
                            // sin(kr/2)+sin(kr).
                            double nr, ni;
                            if (COMPLEX_K) {
                                // exp(k_im*R), k_im <= 0: decaying, so no
                                // overflow, and underflow to +0 at large R is
                                // the physical answer. `expm1(a)·cos(y)` is
                                // the decay half of the bracket; the trig half
                                // is the real-k form below, unscaled.
                                stable_expm1_neg_jkR(k_re, k_im, R, &nr, &ni);
                            } else {
                                nr = stable_cos_minus_one(kr);
                                ni = -std::sin(kr);
                            }
                            const double rr = nr / R, ri = ni / R;
                            a0r += rr * wqq[q];
                            a0i += ri * wqq[q];
                            a1r += rr * twqq[q];
                            a1i += ri * twqq[q];
                        }
                    } else {
                        // NEC Eq 89's coaxial factor and its regularising
                        // extra — `_ek_factor` / `_ek_reg_extra`, term for
                        // term and in their multi-step order.
                        const double ae = saek[js];
                        const double a2 = ae * ae;
                        const double a4 = a2 * a2;
                        for (size_t q = 0; q < n_qp; q++) {
                            const double u = tq[q] - u_r;
                            const double R = std::sqrt(u * u + rho2);
                            const double r2 = R * R;
                            const double r4 = r2 * r2;
                            const double kr = k_re * R;
                            const double t1 = 0.25 * a4 / r4;
                            const double t2 = 0.5 * a2 / r2;
                            double facr, faci, exr, exi, br, bi;
                            if (COMPLEX_K) {
                                // kR is complex: jkR = −Im(k)R + j·Re(k)R, so
                                // C1 picks up a REAL term, and (kR)² an
                                // imaginary one. Term for term and in the same
                                // multi-step order as `_ek_factor` /
                                // `_ek_reg_extra`, which are the reference.
                                const double kri = k_im * R;
                                const double kr2r = kr * kr - kri * kri;
                                const double kr2i = 2.0 * kr * kri;
                                // C1 = 1 + jkR, C2 = 3·C1 − (kR)²
                                const double c1r = 1.0 - kri, c1i = kr;
                                const double c2r = 3.0 * c1r - kr2r;
                                const double c2i = 3.0 * c1i - kr2i;
                                // fac = t1·C2 − t2·C1 + 1
                                facr = t1 * c2r - t2 * c1r + 1.0;
                                faci = t1 * c2i - t2 * c1i;
                                // extra = t1·(3jkR − (kR)²) − t2·(jkR)
                                exr = t1 * (-3.0 * kri - kr2r) + t2 * kri;
                                exi = t1 * (3.0 * kr - kr2i) - t2 * kr;
                                // The reduced branch's bracket, verbatim:
                                // the EK remainder IS that object with NEC
                                // Eq 89's factor on it (momwire#799).
                                stable_expm1_neg_jkR(k_re, k_im, R, &br, &bi);
                            } else {
                                const double kr2 = kr * kr;
                                // C1 = 1 + jkR, C2 = 3·C1 − (kR)²
                                const double c1r = 1.0, c1i = kr;
                                const double c2r = 3.0 * c1r - kr2, c2i = 3.0 * c1i;
                                // fac = t1·C2 − t2·C1 + 1
                                facr = t1 * c2r - t2 * c1r + 1.0;
                                faci = t1 * c2i - t2 * c1i;
                                // extra = t1·(3jkR − (kR)²) − t2·(jkR)
                                exr = t1 * (-kr2);
                                exi = t1 * (3.0 * kr) - t2 * kr;
                                br = stable_cos_minus_one(kr);
                                bi = -std::sin(kr);
                            }
                            const double nr = br * facr - bi * faci + exr;
                            const double ni = br * faci + bi * facr + exi;
                            const double rr = nr / R, ri = ni / R;
                            a0r += rr * wqq[q];
                            a0i += ri * wqq[q];
                            a1r += rr * twqq[q];
                            a1i += ri * twqq[q];
                        }
                    }

                    const size_t s = s_lo + js;
                    row0[s] = std::complex<double>((m0s + a0r) * inv4pi,
                                                   a0i * inv4pi);
                    if (need_m1)
                        row1[s] = std::complex<double>((m1s + a1r) * inv4pi,
                                                       a1i * inv4pi);
                }
            }
        }
    }
    }
    PYSIM_THROW_IF_ABORTED();

    return {M0, M1};
}


// The real-k entry — the only one razor's two quadrature lanes reach today.
// Its <false> instantiation is textually the pre-#796 loop, so this call is
// bit-identical to the kernel as it shipped.
static std::pair<py::array_t<std::complex<double>>,
                 py::array_t<std::complex<double>>>
razor_seg_moments(
    py::array_t<double, py::array::c_style | py::array::forcecast> obs,     // (n_obs, 3)
    py::array_t<double, py::array::c_style | py::array::forcecast> seg_p0,  // (n_seg, 3)
    py::array_t<double, py::array::c_style | py::array::forcecast> seg_t,   // (n_seg, 3)
    py::array_t<double, py::array::c_style | py::array::forcecast> seg_h,   // (n_seg,)
    py::array_t<double, py::array::c_style | py::array::forcecast> a,       // (n_seg,)
    py::array_t<double, py::array::c_style | py::array::forcecast> xg,      // (n_qp,)
    py::array_t<double, py::array::c_style | py::array::forcecast> wg,      // (n_qp,)
    double k,
    bool need_m1,
    py::array_t<int64_t, py::array::c_style | py::array::forcecast> group_i, // (n_obs,) or ()
    py::array_t<int64_t, py::array::c_style | py::array::forcecast> group_j, // (n_seg,) or ()
    py::array_t<double, py::array::c_style | py::array::forcecast> a_ek,     // (n_seg,) or ()
    uintptr_t cancel_flag = 0
) {
    return razor_seg_moments_impl<false>(
        obs, seg_p0, seg_t, seg_h, a, xg, wg,
        k, 0.0, need_m1, group_i, group_j, a_ek, cancel_flag);
}

// The in-medium entry (momwire#796). Throws `std::invalid_argument` rather
// than the `std::runtime_error` its #778 bspline twin throws, because #796
// asks for the refusal to reach Python as a ValueError and pybind11 maps this
// one there; the twin predates that ask and still raises RuntimeError.
static std::pair<py::array_t<std::complex<double>>,
                 py::array_t<std::complex<double>>>
razor_seg_moments_cplx(
    py::array_t<double, py::array::c_style | py::array::forcecast> obs,     // (n_obs, 3)
    py::array_t<double, py::array::c_style | py::array::forcecast> seg_p0,  // (n_seg, 3)
    py::array_t<double, py::array::c_style | py::array::forcecast> seg_t,   // (n_seg, 3)
    py::array_t<double, py::array::c_style | py::array::forcecast> seg_h,   // (n_seg,)
    py::array_t<double, py::array::c_style | py::array::forcecast> a,       // (n_seg,)
    py::array_t<double, py::array::c_style | py::array::forcecast> xg,      // (n_qp,)
    py::array_t<double, py::array::c_style | py::array::forcecast> wg,      // (n_qp,)
    std::complex<double> k,
    bool need_m1,
    py::array_t<int64_t, py::array::c_style | py::array::forcecast> group_i, // (n_obs,) or ()
    py::array_t<int64_t, py::array::c_style | py::array::forcecast> group_j, // (n_seg,) or ()
    py::array_t<double, py::array::c_style | py::array::forcecast> a_ek,     // (n_seg,) or ()
    uintptr_t cancel_flag = 0
) {
    if (k.imag() > 0.0) {
        throw std::invalid_argument(
            "razor_seg_moments_cplx: Im k > 0 is the growing exponential "
            "branch; e^{+jwt} requires Im k <= 0 so e^{-jkR} decays. Take the "
            "other branch of k_m = k0*sqrt(eps_tilde); this kernel will not "
            "conjugate for you.");
    }
    return razor_seg_moments_impl<true>(
        obs, seg_p0, seg_t, seg_h, a, xg, wg,
        k.real(), k.imag(), need_m1, group_i, group_j, a_ek, cancel_flag);
}


// ---------------------------------------------------------------------------
// T1 assembly (momwire#780)
//
// The razor fill's other half. `razor_seg_moments` above already returns the
// moments in C++; what followed in Python was the ASSEMBLY -- gather the two
// wings' columns out of (M0, M1), apply the falling-wing correction, contract
// each testing-path point's tangent with the source tangents, weight, and sum
// over the path points. numpy spells that with a (n_obs, n_basis) complex
// intermediate that is then reduced along a reshaped axis; at
// n_path = 2*n_qp_path = 64 on a 200-segment deck that is a 32 MB temporary
// per chunk, and it measured 52-76% of razor's wall time.
//
// ONE KERNEL SERVES BOTH QUADRATURE LANES. `n_path` enters only as the inner
// loop bound -- 2 for `nec5_quadrature` (the two wing-centroid trapezoid
// points) and 2*n_qp_path for the Gauss-Legendre lane. Nothing else differs:
// `_testing_paths` already hands the fill `(pts, tans, wts)` shape-agnostically
// and `_assemble_Z_source_block` contains no branch on the lane. Keeping it
// that way is the point -- the two lanes cannot drift if they are one
// implementation.
//
// Scope: the unweighted integrand, i.e. free space and any FOLDING ground
// (PEC and the reflection-coefficient fold), which is `w_A_fn is None` on the
// Python side. The finite-ground weighted branch keeps its numpy path for now;
// it carries a per-chunk weight table this signature has no room for.
//
// Falling wings: `fall_a`/`fall_b` are per-basis flags, not per-pair, so the
// branch is hoisted out of the observer loop.
static py::array_t<std::complex<double>>
razor_assemble_t1(
    py::array_t<std::complex<double>, py::array::c_style | py::array::forcecast> M0,
    py::array_t<std::complex<double>, py::array::c_style | py::array::forcecast> M1,
    py::array_t<int64_t, py::array::c_style | py::array::forcecast> s_a,
    py::array_t<int64_t, py::array::c_style | py::array::forcecast> s_b,
    py::array_t<double, py::array::c_style | py::array::forcecast> h_a,
    py::array_t<double, py::array::c_style | py::array::forcecast> h_b,
    py::array_t<bool, py::array::c_style | py::array::forcecast> fall_a,
    py::array_t<bool, py::array::c_style | py::array::forcecast> fall_b,
    py::array_t<double, py::array::c_style | py::array::forcecast> t_out,
    py::array_t<double, py::array::c_style | py::array::forcecast> td_a,
    py::array_t<double, py::array::c_style | py::array::forcecast> td_b,
    py::array_t<double, py::array::c_style | py::array::forcecast> wts,
    size_t n_path
) {
    auto m0 = M0.unchecked<2>();
    auto m1 = M1.unchecked<2>();
    auto sa = s_a.unchecked<1>();
    auto sb = s_b.unchecked<1>();
    auto ha = h_a.unchecked<1>();
    auto hb = h_b.unchecked<1>();
    auto fa = fall_a.unchecked<1>();
    auto fb = fall_b.unchecked<1>();
    auto to = t_out.unchecked<2>();
    auto ta = td_a.unchecked<2>();
    auto tb = td_b.unchecked<2>();
    auto w = wts.unchecked<1>();

    const size_t n_obs = static_cast<size_t>(m1.shape(0));
    const size_t n_seg = static_cast<size_t>(m1.shape(1));
    const size_t n_basis = static_cast<size_t>(sa.shape(0));

    if (n_path == 0 || n_obs % n_path != 0) {
        throw std::runtime_error(
            "razor_assemble_t1: n_obs must be a whole number of testing-path "
            "points (n_obs % n_path != 0)");
    }
    if (static_cast<size_t>(to.shape(0)) != n_obs ||
        static_cast<size_t>(w.shape(0)) != n_obs) {
        throw std::runtime_error(
            "razor_assemble_t1: t_out and wts must have one row per observer");
    }
    if (to.shape(1) != 3 || ta.shape(0) != 3 || tb.shape(0) != 3) {
        throw std::runtime_error("razor_assemble_t1: tangents must be 3-vectors");
    }
    if (static_cast<size_t>(ta.shape(1)) != n_basis ||
        static_cast<size_t>(tb.shape(1)) != n_basis ||
        static_cast<size_t>(sb.shape(0)) != n_basis) {
        throw std::runtime_error(
            "razor_assemble_t1: per-basis arrays disagree on n_basis");
    }
    for (size_t j = 0; j < n_basis; j++) {
        if (sa(j) < 0 || static_cast<size_t>(sa(j)) >= n_seg ||
            sb(j) < 0 || static_cast<size_t>(sb(j)) >= n_seg) {
            throw std::runtime_error(
                "razor_assemble_t1: wing segment index out of range");
        }
    }

    const size_t n_rows = n_obs / n_path;
    py::array_t<std::complex<double>> T1({n_rows, n_basis});
    auto out = T1.mutable_unchecked<2>();

    py::gil_scoped_release release;

    PYSIM_OMP_PARALLEL_FOR_COLLAPSE2
    for (size_t r = 0; r < n_rows; r++) {
        for (size_t j = 0; j < n_basis; j++) {
            const size_t ja = static_cast<size_t>(sa(j));
            const size_t jb = static_cast<size_t>(sb(j));
            const double inv_ha = 1.0 / ha(j);
            const double inv_hb = 1.0 / hb(j);
            const bool fall_j_a = fa(j);
            const bool fall_j_b = fb(j);
            const double a0 = ta(0, j), a1 = ta(1, j), a2 = ta(2, j);
            const double b0 = tb(0, j), b1 = tb(1, j), b2 = tb(2, j);

            double acc_re = 0.0, acc_im = 0.0;
            for (size_t p = 0; p < n_path; p++) {
                const size_t o = r * n_path + p;

                // mom_a = M1/h on a rising wing, M0 - M1/h on a falling one.
                std::complex<double> ma = m1(o, ja) * inv_ha;
                if (fall_j_a) ma = m0(o, ja) - ma;
                std::complex<double> mb = m1(o, jb) * inv_hb;
                if (fall_j_b) mb = m0(o, jb) - mb;

                const double dot_a = to(o, 0) * a0 + to(o, 1) * a1 + to(o, 2) * a2;
                const double dot_b = to(o, 0) * b0 + to(o, 1) * b1 + to(o, 2) * b2;
                const double wo = w(o);

                const double gre = dot_a * ma.real() + dot_b * mb.real();
                const double gim = dot_a * ma.imag() + dot_b * mb.imag();
                acc_re += wo * gre;
                acc_im += wo * gim;
            }
            out(r, j) = std::complex<double>(acc_re, acc_im);
        }
    }
    return T1;
}


// `_ground_refl._TINY`. Both guards below are that constant, and it has to be
// the same number: it is what decides whether a ray counts as vertical.
static constexpr double RAZOR_REFL_TINY = 1e-30;

// One pair's A-term weight window, formed in registers and dropped.
//
// This is `specular_ray_tables` -> `specular_pair_tables` -> `fresnel_rho` ->
// `a_term_weights` for a SINGLE (observer, source segment) pair, in that order
// and term for term. The numpy spelling of that chain builds nine
// (n_obs_chunk, n_seg) float64 tables and three complex ones to produce a
// table the assembler reads twice per basis function and throws away; here
// each value is produced where it is consumed. That is the whole of #744 --
// the arithmetic is unchanged, and the multi-step order is preserved
// deliberately so the two paths agree to the bars rather than by luck.
//
// REFL-COEF ONLY, and that is a contract rather than a first cut. The
// `mode == "compose"` (sommerfeld) window is the constant C2 on the same
// mirrored tangent dot, and it would fuse here trivially -- but C2 reaches Z
// THROUGH THE WINDOWS by design (`PotentialGround.image_coefficient`), and a
// consumer that reads the attribute and applies it itself doubles the exact-
// image half. `test_the_consumer_never_applies_the_image_coefficient_itself`
// pins exactly that, with a ground that lies about its coefficient while its
// weights stay honest, and it caught this kernel serving compose. So the
// sommerfeld contraction stays on numpy until C2 can be routed without the
// consumer reading it.
//
// A ground whose weights are NOT the stock Fresnel pair -- the radial
// screen's `standard_fresnel = False` row -- must never reach here either;
// the Python gate refuses it rather than this function guessing.
static inline std::complex<double> razor_a_window(
    double ox, double oy, double oz,     // observer centre
    double tx, double ty, double tz,     // observer tangent
    double sx, double sy, double sz,     // source centre (REAL, unmirrored)
    double ux, double uy, double uz,     // source tangent (REAL)
    double ground_z,
    const std::complex<double> &eps_t
) {
    // t_m . M t_n with M = diag(1, 1, -1) -- `_ground_mirror.mirror_tangents`
    // is a pure z-flip, so the mirror is this one sign and no offset.
    const double td_img = tx * ux + ty * uy - tz * uz;

    // The specular ray: image midpoint (x_n, y_n, 2 z_g - z_n) to r_m.
    const double dx = ox - sx;
    const double dy = oy - sy;
    const double dz = oz + sz - 2.0 * ground_z;

    // `np.hypot(dx, dy)`, then the norm accumulated (dx^2 + dy^2) + dz^2 --
    // the association the numpy tile spells out, so the same rounding.
    const double hyp = std::hypot(dx, dy);
    double r = dx * dx;
    r += dy * dy;
    r += dz * dz;
    r = std::sqrt(r);
    if (r < RAZOR_REFL_TINY) {
        r = RAZOR_REFL_TINY;  // np.maximum(., _TINY)
    }
    const double cos_th = dz / r;

    // p_hat = (-dy, dx, 0)/hyp; a near-vertical ray has no incidence plane and
    // falls back to x_hat. Harmless: rho_v = -rho_h at theta = 0 makes the
    // dyad isotropic there.
    double p_x, p_y;
    if (hyp > RAZOR_REFL_TINY) {
        const double inv_hyp = 1.0 / hyp;
        p_x = -(dy * inv_hyp);
        p_y = dx * inv_hyp;
    } else {
        p_x = 1.0;
        p_y = 0.0;
    }

    const double tm_p = tx * p_x + ty * p_y;
    const double tn_p = ux * p_x + uy * p_y;
    const double P = tm_p * tn_p;

    // `fresnel_rho`. Principal-branch sqrt has Im <= 0 for Im(eps_t) <= 0, the
    // decaying-transmitted-wave branch, matching NEC.
    const double sin2 = 1.0 - cos_th * cos_th;
    const std::complex<double> root = std::sqrt(eps_t - sin2);
    const std::complex<double> ec = eps_t * cos_th;
    const std::complex<double> rho_v = (ec - root) / (ec + root);
    const std::complex<double> rho_h = (cos_th - root) / (cos_th + root);

    // `a_term_weights`.
    return rho_v * (td_img - P) - rho_h * P;
}

// momwire#744: the WEIGHTED twin of `razor_assemble_t1`.
//
// `razor_assemble_t1` serves the unweighted integrand only -- free space and
// the folding grounds -- and says so. The finite-ground weighted branch kept
// its numpy path because it carries a per-pair weight table that signature has
// no room for. This is that branch, with the table formed inside the tile
// instead of passed into it.
//
// WHY THIS IS THE SAME KERNEL AND NOT A DIFFERENT ONE
// ---------------------------------------------------
// The two differ in exactly one factor. Unweighted contracts the observer
// tangent against the source tangent, a REAL dot; weighted replaces that dot
// wholesale with a COMPLEX per-pair window, which is why `td_a` / `td_b`
// (which have sigma folded in) are absent here and `sig_a` / `sig_b` arrive on
// their own. Everything around it -- the two-wing gather out of (M0, M1), the
// falling-wing correction, the path weight, the sum over the path points -- is
// the loop above, term for term.
//
// WHAT IS NEVER FORMED
// --------------------
// The numpy path builds the full (n_obs_chunk, n_seg) window plane and then
// reads two columns of it per basis function. This computes the two columns it
// needs and nothing else. That is ~2 windows per (observer, basis) against
// n_seg per observer, i.e. MORE window arithmetic (each wing segment is
// referenced by about two basis functions, so roughly 2x), traded for the
// plane's memory traffic and for OpenMP. `_WEIGHTED_CHUNK_ELEMS` exists to
// keep that plane's transient in the same class as an unweighted chunk's; it
// still sets the chunking here, and is now a schedule number only.
//
// The numpy path stays as the reference and the fallback -- this is a gate,
// not a replacement.
static py::array_t<std::complex<double>>
razor_assemble_t1_weighted(
    py::array_t<std::complex<double>, py::array::c_style | py::array::forcecast> M0,
    py::array_t<std::complex<double>, py::array::c_style | py::array::forcecast> M1,
    py::array_t<int64_t, py::array::c_style | py::array::forcecast> s_a,
    py::array_t<int64_t, py::array::c_style | py::array::forcecast> s_b,
    py::array_t<double, py::array::c_style | py::array::forcecast> h_a,
    py::array_t<double, py::array::c_style | py::array::forcecast> h_b,
    py::array_t<bool, py::array::c_style | py::array::forcecast> fall_a,
    py::array_t<bool, py::array::c_style | py::array::forcecast> fall_b,
    py::array_t<double, py::array::c_style | py::array::forcecast> sig_a,
    py::array_t<double, py::array::c_style | py::array::forcecast> sig_b,
    py::array_t<double, py::array::c_style | py::array::forcecast> obs_c,
    py::array_t<double, py::array::c_style | py::array::forcecast> obs_t,
    py::array_t<double, py::array::c_style | py::array::forcecast> src_c,
    py::array_t<double, py::array::c_style | py::array::forcecast> src_t,
    py::array_t<double, py::array::c_style | py::array::forcecast> wts,
    size_t n_path,
    std::complex<double> eps_t,
    double ground_z
) {
    auto m0 = M0.unchecked<2>();
    auto m1 = M1.unchecked<2>();
    auto sa = s_a.unchecked<1>();
    auto sb = s_b.unchecked<1>();
    auto ha = h_a.unchecked<1>();
    auto hb = h_b.unchecked<1>();
    auto fa = fall_a.unchecked<1>();
    auto fb = fall_b.unchecked<1>();
    auto ga = sig_a.unchecked<1>();
    auto gb = sig_b.unchecked<1>();
    auto oc = obs_c.unchecked<2>();
    auto ot = obs_t.unchecked<2>();
    auto sc = src_c.unchecked<2>();
    auto st = src_t.unchecked<2>();
    auto w = wts.unchecked<1>();

    const size_t n_obs = static_cast<size_t>(m1.shape(0));
    const size_t n_seg = static_cast<size_t>(m1.shape(1));
    const size_t n_basis = static_cast<size_t>(sa.shape(0));

    if (n_path == 0 || n_obs % n_path != 0) {
        throw std::runtime_error(
            "razor_assemble_t1_weighted: n_obs must be a whole number of "
            "testing-path points (n_obs % n_path != 0)");
    }
    if (static_cast<size_t>(oc.shape(0)) != n_obs ||
        static_cast<size_t>(ot.shape(0)) != n_obs ||
        static_cast<size_t>(w.shape(0)) != n_obs) {
        throw std::runtime_error(
            "razor_assemble_t1_weighted: obs_c, obs_t and wts must have one "
            "row per observer");
    }
    if (oc.shape(1) != 3 || ot.shape(1) != 3 ||
        sc.shape(1) != 3 || st.shape(1) != 3) {
        throw std::runtime_error(
            "razor_assemble_t1_weighted: centres and tangents must be "
            "3-vectors");
    }
    if (static_cast<size_t>(sc.shape(0)) != n_seg ||
        static_cast<size_t>(st.shape(0)) != n_seg) {
        throw std::runtime_error(
            "razor_assemble_t1_weighted: src_c and src_t must have one row per "
            "source segment");
    }
    if (static_cast<size_t>(sb.shape(0)) != n_basis ||
        static_cast<size_t>(ga.shape(0)) != n_basis ||
        static_cast<size_t>(gb.shape(0)) != n_basis) {
        throw std::runtime_error(
            "razor_assemble_t1_weighted: per-basis arrays disagree on n_basis");
    }
    for (size_t j = 0; j < n_basis; j++) {
        if (sa(j) < 0 || static_cast<size_t>(sa(j)) >= n_seg ||
            sb(j) < 0 || static_cast<size_t>(sb(j)) >= n_seg) {
            throw std::runtime_error(
                "razor_assemble_t1_weighted: wing segment index out of range");
        }
    }

    const size_t n_rows = n_obs / n_path;
    py::array_t<std::complex<double>> T1({n_rows, n_basis});
    auto out = T1.mutable_unchecked<2>();

    py::gil_scoped_release release;

    PYSIM_OMP_PARALLEL_FOR_COLLAPSE2
    for (size_t r = 0; r < n_rows; r++) {
        for (size_t j = 0; j < n_basis; j++) {
            const size_t ja = static_cast<size_t>(sa(j));
            const size_t jb = static_cast<size_t>(sb(j));
            const double inv_ha = 1.0 / ha(j);
            const double inv_hb = 1.0 / hb(j);
            const bool fall_j_a = fa(j);
            const bool fall_j_b = fb(j);
            const double sg_a = ga(j);
            const double sg_b = gb(j);

            // The two wing segments' geometry is per-basis, so it is hoisted
            // out of the path-point loop exactly as the tangents are above.
            const double acx = sc(ja, 0), acy = sc(ja, 1), acz = sc(ja, 2);
            const double atx = st(ja, 0), aty = st(ja, 1), atz = st(ja, 2);
            const double bcx = sc(jb, 0), bcy = sc(jb, 1), bcz = sc(jb, 2);
            const double btx = st(jb, 0), bty = st(jb, 1), btz = st(jb, 2);

            std::complex<double> acc(0.0, 0.0);
            for (size_t p = 0; p < n_path; p++) {
                const size_t o = r * n_path + p;

                // mom_a = M1/h on a rising wing, M0 - M1/h on a falling one.
                std::complex<double> ma = m1(o, ja) * inv_ha;
                if (fall_j_a) ma = m0(o, ja) - ma;
                std::complex<double> mb = m1(o, jb) * inv_hb;
                if (fall_j_b) mb = m0(o, jb) - mb;

                const double ox = oc(o, 0), oy = oc(o, 1), oz = oc(o, 2);
                const double tx = ot(o, 0), ty = ot(o, 1), tz = ot(o, 2);

                // `w_A[:, s_a] * sig_a`, one pair at a time.
                const std::complex<double> wa =
                    razor_a_window(ox, oy, oz, tx, ty, tz,
                                   acx, acy, acz, atx, aty, atz,
                                   ground_z, eps_t) * sg_a;
                const std::complex<double> wb =
                    razor_a_window(ox, oy, oz, tx, ty, tz,
                                   bcx, bcy, bcz, btx, bty, btz,
                                   ground_z, eps_t) * sg_b;

                // `integrand = wA_a*mom_a + wA_b*mom_b`, then `*= wts`, then
                // summed over the path -- the numpy order, kept.
                acc += w(o) * (wa * ma + wb * mb);
            }
            out(r, j) = acc;
        }
    }
    return T1;
}


void register_razor(py::module_ &m) {
    // The capability flag lives here, in the TU that defines the symbol it
    // vouches for (#710 review): a build carrying `razor_fill_742` carries
    // this kernel.
    m.attr("razor_fill_742") = true;

    m.def("razor_seg_moments", &razor_seg_moments,
          "Fused, tiled segment-moment fill for the razor-blade formulation "
          "(momwire#742). Returns (M0, M1), each (n_obs, n_seg) complex, with "
          "M0 = int_0^h g dtau and M1 = int_0^h tau g dtau over every source "
          "segment at every observer — the closed-form static moments of "
          "_kernel_moments plus the Gauss-Legendre smooth remainder "
          "(exp(-jkR)-1)/(4 pi R), computed together so the (n_obs, n_seg, "
          "n_qp) distance table the numpy path retains is never formed. M1 is "
          "returned as a (0, 0) array when need_m1 is false. group_i / "
          "group_j are the extended kernel's coaxial-equal-radius labels "
          "(_ek_pair_mask: pair (p, s) extends iff group_i[p] == group_j[s] "
          ">= 0); pass empty arrays for the reduced kernel. a is the axis "
          "frame's regularising radius per SOURCE segment and a_ek the EK "
          "radius, both as (n_seg,) columns.",
          py::arg("obs"), py::arg("seg_p0"), py::arg("seg_t"), py::arg("seg_h"),
          py::arg("a"), py::arg("xg"), py::arg("wg"), py::arg("k"),
          py::arg("need_m1"), py::arg("group_i"), py::arg("group_j"),
          py::arg("a_ek"), py::arg("cancel_flag") = 0);

    // momwire#796, on its OWN flag for the same reason `razor_fill_742` is:
    // a .so built before this lands exports the real-k kernel and not this,
    // and the Python gate must be able to tell the difference rather than
    // assume one symbol implies the other.
    m.attr("razor_cplx_796") = true;
    m.def("razor_seg_moments_cplx", &razor_seg_moments_cplx,
          "The in-medium (complex k) twin of razor_seg_moments (momwire#796). "
          "Identical in shape and meaning; k is a complex wavenumber with "
          "Im k <= 0, and exp(-jkR) is continued as the real scale factor "
          "exp(Im(k) R) on the trig part, so the reduced branch's `- 1` and "
          "the k-free statics are untouched while the EK branch's C1/C2 pick "
          "up their terms in Im(k) R. Raises ValueError on Im k > 0, the "
          "growing-exponential branch this kernel will not evaluate.",
          py::arg("obs"), py::arg("seg_p0"), py::arg("seg_t"), py::arg("seg_h"),
          py::arg("a"), py::arg("xg"), py::arg("wg"), py::arg("k"),
          py::arg("need_m1"), py::arg("group_i"), py::arg("group_j"),
          py::arg("a_ek"), py::arg("cancel_flag") = 0);

    // momwire#780. Declared beside the moments kernel it pairs with.
    m.attr("razor_assemble_780") = true;
    m.def("razor_assemble_t1", &razor_assemble_t1,
          "Fused T1 assembly for the razor-blade formulation (momwire#780). "
          "Gathers both wings out of (M0, M1), applies the falling-wing "
          "correction, contracts each testing-path point's tangent with the "
          "source tangents, weights, and sums over the path points -- without "
          "forming the (n_obs, n_basis) complex intermediate the numpy path "
          "builds and immediately reduces. Returns T1 of shape "
          "(n_obs / n_path, n_basis). `n_path` is the only thing that differs "
          "between the two quadrature lanes (2 under nec5_quadrature, "
          "2*n_qp_path under Gauss-Legendre), so one kernel serves both. "
          "Unweighted integrand only: free space and folding grounds, i.e. "
          "the `w_A_fn is None` branch.",
          py::arg("M0"), py::arg("M1"), py::arg("s_a"), py::arg("s_b"),
          py::arg("h_a"), py::arg("h_b"), py::arg("fall_a"), py::arg("fall_b"),
          py::arg("t_out"), py::arg("td_a"), py::arg("td_b"), py::arg("wts"),
          py::arg("n_path"));

    // momwire#744, on its OWN flag for the same reason every kernel above is:
    // a .so built before this lands exports the unweighted assembler and not
    // this, and the Python gate must be able to tell rather than assume the
    // #780 symbol implies this one.
    m.attr("razor_weighted_744") = true;
    m.def("razor_assemble_t1_weighted", &razor_assemble_t1_weighted,
          "The WEIGHTED twin of razor_assemble_t1 (momwire#744). Same gather, "
          "falling-wing correction, path weighting and path-point sum, with "
          "the real tangent contraction replaced by the complex per-pair "
          "A-term weight window -- formed inside the tile from the observer "
          "and source geometry, so the (n_obs_chunk, n_seg) window plane the "
          "numpy closure materialises is never built. "
          "REFL-COEF ONLY: the specular_pair_tables -> fresnel_rho -> "
          "a_term_weights chain, from eps_t and ground_z. A composing "
          "(sommerfeld) ground is NOT served -- its C2 reaches Z through the "
          "windows by design, and a consumer that applies the attribute "
          "itself doubles the exact-image half. A ground whose weights are "
          "not the stock Fresnel pair (standard_fresnel = False) must be "
          "refused by the caller too. Returns T1 of shape "
          "(n_obs / n_path, n_basis).",
          py::arg("M0"), py::arg("M1"), py::arg("s_a"), py::arg("s_b"),
          py::arg("h_a"), py::arg("h_b"), py::arg("fall_a"), py::arg("fall_b"),
          py::arg("sig_a"), py::arg("sig_b"), py::arg("obs_c"),
          py::arg("obs_t"), py::arg("src_c"), py::arg("src_t"),
          py::arg("wts"), py::arg("n_path"), py::arg("eps_t"),
          py::arg("ground_z"));
}
