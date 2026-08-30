#include "_accel_common.h"

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
    const double rho = std::sqrt(rho2);
    const double u0 = -u_r;
    const double u1 = h - u_r;
    const double mm0 = std::asinh(u1 / rho) - std::asinh(u0 / rho);
    *m0 = mm0;
    *m1 = u_r * mm0 +
          (std::sqrt(u1 * u1 + rho2) - std::sqrt(u0 * u0 + rho2));
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
    const double rho = std::sqrt(rho2);
    const double u0 = -u_r;
    const double u1 = h - u_r;
    const double r0 = std::sqrt(u0 * u0 + rho2);
    const double r1 = std::sqrt(u1 * u1 + rho2);
    const double c3 = 0.25 * a4 / rho2;
    const double c1 = 0.5 * a2 * perp2 / (rho2 * rho2);
    const double p1 = std::asinh(u1 / rho) + c3 * u1 / (r1 * r1 * r1) - c1 * u1 / r1;
    const double p0 = std::asinh(u0 / rho) + c3 * u0 / (r0 * r0 * r0) - c1 * u0 / r0;
    const double mm0 = p1 - p0;
    const double q1 = r1 + 0.5 * a2 / r1 - 0.25 * a4 / (r1 * r1 * r1);
    const double q0 = r0 + 0.5 * a2 / r0 - 0.25 * a4 / (r0 * r0 * r0);
    *m0 = mm0;
    *m1 = u_r * mm0 + (q1 - q0);
}

// Fused segment moments at one wavenumber. Returns (M0, M1); M1 is a (0, 0)
// array when `need_m1` is false, which is how the caller spells "the scalar
// potential only needs M0" without a second entry point.
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
                    const double u_r = dx * stan[js * 3 + 0] +
                                       dy * stan[js * 3 + 1] +
                                       dz * stan[js * 3 + 2];
                    double perp = dx * dx + dy * dy + dz * dz - u_r * u_r;
                    // A truly collinear observer can drive this a few ulps
                    // negative; a² dominates it either way.
                    if (perp < 0.0) perp = 0.0;
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
                        PYSIM_OMP_SIMD(reduction(+ : a0r, a0i, a1r, a1i))
                        for (size_t q = 0; q < n_qp; q++) {
                            const double u = tq[q] - u_r;
                            const double R = std::sqrt(u * u + rho2);
                            const double kr = k * R;
                            // exp(−jkR) − 1, then the complex-by-real divide
                            // numpy performs (Smith's algorithm collapses to
                            // a componentwise divide on a real denominator).
                            const double nr = std::cos(kr) - 1.0;
                            const double ni = -std::sin(kr);
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
                            const double kr = k * R;
                            const double kr2 = kr * kr;
                            const double t1 = 0.25 * a4 / r4;
                            const double t2 = 0.5 * a2 / r2;
                            // C1 = 1 + jkR, C2 = 3·C1 − (kR)²
                            const double c1r = 1.0, c1i = kr;
                            const double c2r = 3.0 * c1r - kr2, c2i = 3.0 * c1i;
                            // fac = t1·C2 − t2·C1 + 1
                            const double facr = t1 * c2r - t2 * c1r + 1.0;
                            const double faci = t1 * c2i - t2 * c1i;
                            // extra = t1·(3jkR − (kR)²) − t2·(jkR)
                            const double exr = t1 * (-kr2);
                            const double exi = t1 * (3.0 * kr) - t2 * kr;
                            const double br = std::cos(kr) - 1.0;
                            const double bi = -std::sin(kr);
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
}
