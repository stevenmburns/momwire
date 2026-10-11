# momwire#1433: the exact ring kernel on razor-2p — derivation

Public mathematics only. The kernel and its log split are momwire#1408's
(`src/momwire/_exact_kernel.py`); nothing here comes from NEC-4.2 or NEC-5
material. NEC-5 appears only as a black-box reference in the ladders.

## 1. What razor's matrix element integrates

The basis and the testing are in `razor.py`'s module docstring.

- **Basis.** One unit tent Λ_n per knot. It has two wings, each linear on one segment.
- **Testing.** Row m is tested along the path P_m, which runs from the centroid of wing A through the knot to the centroid of wing B:

      Z[m,n] = jωμ T1[m,n] − T2[m,n] / (jωε)
      T1[m,n] = ∫_{P_m} t̂·A_n dl,    A_n(r) = ∫ Λ_n(l') g(r, r') t̂(r') dl'
      T2[m,n] = ∫ Λ_n'(l') [g(c_after, r') − g(c_before, r')] dl'

- **The kernel g.** g = e^{−jkR}/(4πR) with R² = |r − r'|² + a², which is the reduced kernel. Under `extended_kernel` it is NEC's series in a²/R² about it.
- **The quadrature.** Under `nec5_quadrature` (razor-2p) the path integral is one node per wing, at the wing's centroid, with weight h_wing/2. Otherwise each half-path takes `n_qp_path` Gauss-Legendre nodes.

Every number the fill uses is one of two moments of g, taken at an observer POINT o against a source segment j of length h_j, with τ' the source's local arc:

    M0[o, j] = ∫_0^{h_j} g(o, τ') dτ',     M1[o, j] = ∫_0^{h_j} τ' g(o, τ') dτ'

- **The A term.** A wing on segment s contributes σ·M1/h to A_n where the tent rises along s (knot at arc h), and σ·(M0 − M1/h) where it falls. It is carried by σ·t̂_s.
- **The charge term.** The charge doublet on a wing is the constant q = σ·(±1/h).
- **Hence:**

      T1[m,n] = Σ_{nodes p of m} w_p (t̂_p · Σ_{wings e of n} σ_e t̂_e mom_e[p])
      T2[m,n] = Σ_{wings e of n} q_e (M0[c_after(m), j_e] − M0[c_before(m), j_e])

  with mom_e = M1/h or M0 − M1/h. A grounded row has no c_before; its plane endpoint drops out.

**There is no 2-D integral anywhere.** Under razor-2p the observers are exactly the segment centroids, for T1 and T2 alike, and on the GL lane they are the path nodes and the centroids. This is the structural difference from bs2. bs2's Galerkin rows integrate the observer over the whole test segment, so its correction needs `pair_moments`' 2-D J_pq = ∬ u^p v^q K(c + v − u), reduced exactly to 1-D by the closed-form weight W_pq. Razor needs only the 1-D moments at a point. Porting bs2's Galerkin moments would integrate something razor's rows never integrate.

## 2. Which kernel a coaxial pair should see

Take a tube current uniform around a straight wire of radius a. Its potential at a point at axial separation z is the ring average of e^{−jkR}/(4πR).

- **Observed on the AXIS.** Every source point on the ring is at R² = z² + a². The ring average is then exactly the reduced kernel. The reduced kernel is not an approximation of a line source: it is the exact potential of the tube, observed on the axis.
- **Observed on the SURFACE,** where the boundary condition belongs. R² = z² + 4a² sin²(φ/2), and the ring average is the exact ring kernel

      K(z) = (1/2π) ∮ e^{−jkR}/(4πR) dφ.

  The extended kernel is a series about the axis value in a²/R². It is outside its validity once R ~ a, which is the self and adjacent pairs at h < a. That is #1422's mechanism.
- **Off-line pairs.** For a pair whose observer lies on the source's own line at the source's radius, the potential on the surface does not depend on φ. Razor's test path, read as a generator of the tube's surface, therefore sees exactly K. Every other pair keeps the fill's kernel: bends, offset parallel wires, radius steps, images, and the Sommerfeld remainder. There the ring average is a true 2-D integral and the current is not ring-uniform (#1408 scope).

So on the coaxial pairs (`_ek_axis_groups_coaxial`: one line, one radius, any separation along it, collinear wires included) the correction is

    ΔM_q[o, j] = ∫_0^{h_j} τ'^q [K(z(o, τ')) − g_fill(o, τ')] dτ',    q = 0, 1,

and Z changes by the fill's own algebra (§1) applied to ΔM. That algebra is linear in (M0, M1), so the result equals the fill evaluated with the corrected moments, to roundoff. g_fill is recomputed by `_seg_moments` with the fill's kernel spec (reduced or EK, with the same labels and radii) at the same observer points, so it cancels what the fill put there.

## 3. The 1-D primitive: point moments

Put the observer at the origin of the common axis and let the source segment run forward from c over length h:

    N_q(c, h) = ∫_0^h v^q K(c + v) dv.

- **Backward segments.** A segment that runs backward (σ = −1) from its start x₀ is the same integral with c = σ(x₀ − x_obs), because K is even. So (c, h) is the whole geometry, and τ' = v is the segment's own coordinate. The moments razor needs are M_q = N_q(σ(x₀ − x_o), h).
- **The singular point.** K is log-singular at v* = −c. That point is INSIDE the segment for razor-2p's self term (the centroid observer), at a segment end on the GL lane's knot-adjacent nodes, and outside otherwise. N_q is finite in every case, since ∫ ln|v − v*| dv converges.
- **Evaluation.** Split [0, h] at v* and integrate each piece outward from it. A piece is ∫_d^{d+ℓ} f(v₀ ± ζ) K(ζ) dζ with f = v^q. It takes the same three rules as `pair_moments`' pieces, with the polynomial in place of the 2-D weight:
  - within 2a of the singular point, K = A ln ζ + B with A and B smooth (#1408's exact split of K(m), E(m) and the recurrence). ∫ f A ln ζ takes product Gauss-Legendre-log weights, exact for polynomials of degree < 20, and ∫ f B plain GL;
  - beyond 2a, geometric grading (ratio 3), 16-point GL per chunk;
  - a piece at least its own length away takes one 16-point GL rule (Bernstein ratio ≥ 3 + √8, error ~1e-24).
- **Accuracy.** Against scipy's adaptive `quad` at epsrel 1e-13, with the singular point named, on seven geometries (observer at the centre, off-centre, at either end, adjacent within 2a, far, and a long segment), the error is ≤ 5.2e-15 relative (`tests/test_exact_kernel_razor_1433.py`).
- **Dedup.** On a uniform run the observer–source offsets repeat, so pairs are keyed on the quantised (c, h). The C++ route also keys on the lesser of c and its mirror −c − h. Under v → h − v (K even), N_q(c, h) = Σ_r C(q, r) h^{q−r} (−1)^r N_r(−c − h, h). So n centroids against n segments cost n kernel evaluations, measured at 3000 for 3000 (the n² remainder is a hash lookup).

## 4. The C++ route, and its tolerance

- **Where it lives.** `_accel_exact_kernel_point.cpp` has its own TU (the #1187 rule). It holds the point moments, the mirror-keyed cache, and the correction's assembly. The assembly is parallel over Z's columns, so no two threads write one entry.
- **Shared headers.** The kernel, its split, the rules and the dedup table are two headers. They were moved verbatim out of the 1410 TU, whose preprocessed token stream is unchanged (verified by `g++ -E -P` diff).
- **Tolerance against the numpy reference.** The 1410 decision says not bit-exact, so the routes are held to a derived bound. The kernel's own accuracy is 3e-12 relative and the two routes share nodes. The gate is therefore 1e-12 of the largest entry. Measured: point rows 1.4e-14, and razor Z (dipole / bend / GL-lane bend) 4e-14, 1.4e-13 and 9.5e-14.

## 5. Mechanism gate

Hand the correction the fill's OWN kernel, reduced or EK, monkeypatched into the numpy route. Z must then not move. On four decks (dipole, 45° bend, collinear junction plus bend, PEC monopole with contact), both kernels and both quadrature lanes, the swap moves Z by ≤ 5.4e-9 relative (the EK bend; the rest ≤ 2e-10). That residual is the fill's own n_qp_source = 12 accuracy, not the correction's. A kernel 1e-6 off moves Z by more than 1e-8.

## 6. What the exact kernel cannot do on razor: the knot gap

Razor's source is a delta gap at a knot, which has zero width. Under the exact kernel a zero-width gap has no limiting reactance: each halving of the segments at it adds 4 ln2·ωε₀a of shunt susceptance (Fikioris & Valagiannopoulos 2005; #1408, #1422 §3).

- **What the kernel does here.** It turns razor-2p's accelerating drift at a fat fed wire (#1422: ×2.3 per √2 refinement) into that logarithmic drift. It cannot remove the drift.
- **What converges.** Conductance, currents away from the feed, and far fields all converge.
- **Where a fixed gap is.** Razor has no end-port feed (`junction_ports=False`). A gap of fixed width is bspline's end-port feed. The advisories say so for razor (`EXACT_KERNEL_FIXED_GAP_FEED = False`).
- **What the ladders below measure.** A gap-free quantity (#1378's broadside current S), the conductance, and the susceptance step per halving against 4 ln2·ωε₀a.

## 7. Results

All runs were on Haswell, from a fresh clone of the branch with a fresh venv. The raw JSON lines are in `results/`. NEC-5 is the licensed binary, run as a black box, and its numbers are antennaknobs' `scratch/dan-tapered-1422/tables.md` and `scratch/1327-endcap/bb/`. bs2 exact numbers are the same tables' `bs2-exact-auto` column.

**Dan's tapered dipole (`taper`), source X in ohms, through the NEC-5 seam:**

| segs/λ | min h/a | NEC-5 | razor-2p | razor-2p exact | bs2 exact | razor-2p − NEC-5 | razor-2p exact − NEC-5 |
|---|---|---|---|---|---|---|---|
| 320 | 4.17 | −0.1460 | −0.1503 | −0.1531 | +0.1008 | −0.0043 | −0.0071 |
| 500 | 2.60 | −0.0379 | −0.0538 | −0.0572 | +0.0751 | −0.0159 | −0.0193 |
| 700 | 1.89 | +0.0061 | −0.0292 | −0.0265 | +0.0298 | −0.0353 | −0.0326 |
| 1000 | 1.33 | +0.0198 | −0.0499 | −0.0207 | −0.0262 | −0.0697 | −0.0405 |
| 1400 | 0.95 | +0.0385 | −0.1287 | −0.0270 | −0.0830 | −0.1672 | −0.0655 |
| 2000 | 0.66 | −0.0026 | −0.3571 | −0.0383 | −0.1444 | −0.3545 | −0.0357 |

**At 2000 segs/λ (Δ/a 0.66), each variant of the family, X − NEC-5 in ohms:**

| variant | razor-2p | razor-2p exact |
|---|---|---|
| taper | −0.355 | −0.036 |
| straight | −0.573 | −0.050 |
| u16 | +0.800 | +0.080 |
| step16to8 | +0.340 | +0.060 |
| step16to1 | −0.435 | −0.314 |

R moves the same way. For u16 the values are NEC-5 90.558, razor-2p 92.124 and razor-2p exact 90.574. For straight they are 92.982, 93.350 and 92.969.

**#1378's gap-free ladder** (5 m dipole, 28 MHz, a = 23.8 mm). S is the broadside short-circuit current; the table gives S − S_ref in µA, where S_ref is the body-of-revolution open tube:

| N | Δ/a | NEC-5 | razor-2p | razor-2p exact | bs2 exact |
|---|---|---|---|---|---|
| 81 | 2.59 | +228+1999j | +236+1986j | +235+1973j | +85+566j |
| 161 | 1.30 | +139+1020j | +129+909j | +143+1025j | +45+281j |
| 321 | 0.65 | +75+501j | +16+83j | +78+514j | +23+133j |
| 641 | 0.33 | +37+238j | −114−671j | +41+251j | +12+59j |
| 1281 | 0.16 | +17+104j | −283−1487j | +21+118j | +6+22j |

**#1408's fat dipole** (L = 0.47λ, d/λ 0.0085, knot/point gap). G = Re(1/Z) in mS, followed by the step per halving:

| n (Δ/a) | razor-2p | razor-2p exact | bs2 exact |
|---|---|---|---|
| 74 (1.49) | 13.249 | 13.266 (−0.275) | 13.052 (−0.087) |
| 150 (0.74) | 13.026 | 13.117 (−0.148) | 13.008 (−0.045) |
| 300 (0.37) | 12.827 (Z 73.0−19.0j) | 13.042 (−0.076) | 12.986 (−0.022) |
| 600 (0.18) | collapse (Z 0.004−0.55j) | 13.003 (−0.039) | 12.975 (−0.011) |

The two exact lanes extrapolate to the same G: 12.965 and 12.964 mS. B per halving climbs toward 4 ln2·ωε₀a = 0.1965 mS on both exact lanes (razor-2p exact +0.090, +0.144; bs2 exact +0.165, +0.182), which is the knot gap's log drift of §6.

## 8. What sin (#1434) and SG (#1435) would reuse

- **sin (point matching).** Field form. The matched E_z at a point is (1/jωε)[k² ∫ f K + ∂_x ∫ f' K], and the second term integrates by parts to end values of f' K plus ∫ f'' K. Over a fat short segment the three-term sinusoid is a polynomial to roundoff at modest degree. So sin needs N_q (q ≤ 5) and K at segment ends, both already in the point primitive (`nq` ≤ 6), plus about the same assembly razor needed.
- **SG (Galerkin).** SG needs bs2's 2-D primitive (`coaxial_rows`, nd ≤ 6) on a polynomial expansion of its basis.
- **Shared by both.** The option, refusals, groups, frames and advisories (`_exact_kernel_seam`).
