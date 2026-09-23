# momwire#1154 — a dielectric jacket on a wire in soil

Labels: **[M]** measured here (script + output named), **[R]** from a
reference or from momwire's own code/docstrings, **[I]** inferred / derived
in this note and not yet measured.

## 1. What momwire serves for a jacket today

[R] `_wire_loading.configure_loading` and `loading_for`: a jacketed wire is a
PAIR (momwire#865, Popovic–Nesic 1984):

- the **kernel radius** is the equivalent radius
  `a'_fs = a (b/a)^((eps_r - 1)/eps_r)` (`equivalent_radius`), written into
  `_radius_per_wire`, and every fill — free space, image, Sommerfeld, and
  the buried families on both trunks — integrates its kernel at that radius;
- a **series inductance** `L'_fs = mu0/2pi (1 - 1/eps_r) ln(b/a)`
  (`insulation_inductance`), evaluated at the conductor radius `a` and added
  per wire through `loading_for`.

Both are medium-independent: neither reads `ground_eps`.

## 2. The quasi-static derivation in an exterior medium eps~

[I] Coated conductor: metal radius `a`, jacket outer radius `b`, jacket
relative permittivity `eps_r`, exterior relative permittivity `eps~`
(complex, `eps~ = eps_r,soil - j sigma/(w eps0)`, i.e.
`_ground_refl.eps_tilde(ground_eps, w, solver.eps)`, the same call the buried
fills make [R]).

Per unit length, referred to a far radius `R` in the exterior:

- **Magnetic.** The jacket is non-magnetic, so the flux linkage is set by the
  METAL: `L'_true = mu0/2pi ln(R/a)`. No permittivity enters. [I]
- **Electric.** The elastance (1/C') of two coaxial dielectric shells in
  series: `S'_true = [ ln(b/a)/eps_r + ln(R/b)/eps~ ] / (2 pi eps0)`. [I]

A bare wire of kernel radius `a_k` in the homogeneous exterior has
`L'_k = mu0/2pi ln(R/a_k)` and `S'_k = ln(R/a_k) / (2 pi eps0 eps~)`. The
corrections the model must add, independent of `R`:

    dL' = L'_true - L'_k = mu0/2pi ln(a_k/a)                         (series)
    dS' = S'_true - S'_k = [ ln(b/a)/eps_r + ln(a_k/b)/eps~ ] / (2 pi eps0)
                                                                     (charge)

Two consistent choices of `a_k`:

1. **`dS' = 0`**: `ln(a_k/a) = (1 - eps~/eps_r) ln(b/a)`, i.e. the in-medium
   equivalent radius `a'(eps~) = a (b/a)^(1 - eps~/eps_r)`, with
   `dL' = mu0/2pi (1 - eps~/eps_r) ln(b/a)`. **This is the issue's formula —
   but only together with the in-medium radius.** With complex `eps~` the
   radius is COMPLEX, and with `|eps~| > eps_r` (every ordinary soil at HF
   against PVC/PE) `a' < a`. No momwire kernel takes a complex radius (the
   C++ kernels are `double a`).
2. **`a_k = a'_fs` (today's kernel, unchanged)**: then
   `dL' = mu0/2pi (1 - 1/eps_r) ln(b/a) = L'_fs` — **today's series term is
   already exactly right in soil** — and the charge correction is

       dS' = ln(b/a) / (2 pi eps0 eps_r) * (1 - 1/eps~)

   which vanishes at `eps~ = 1` and is what momwire is missing.

So the defect is NOT in `insulation_inductance`. It is in the other half of
the pair: `a'_fs` bakes the jacket's CHARGE correction in against a
free-space exterior, and in soil the missing piece is a local elastance.
Changing `L'` to `(1 - eps~/eps_r)` while keeping `a'_fs` (the fix the issue
proposes) would break the magnetic half, which today is exact, and still
leave the charge half wrong: `L'` and `C'` enter the wire's `Z0` differently,
and the error cannot be moved from one to the other by a local constant (it
depends on `ln R`). [I]

### How dS' enters the EFIE

[I] The local elastance adds a scalar potential `phi_x = dS' q`, with
`q = -(1/jw) dI/dl`. Tested exactly the way the kernel's scalar-potential
term is (weak form, boundary terms dropped at junctions and free ends by the
same argument the kernel term uses):

    Z_mn += (1/(j w)) * integral dS'(l) f_m'(l) f_n'(l) dl

— a **derivative Gram**, weighted per segment (each segment is wholly in one
medium; `dS' = 0` on every ABOVE segment). It is k-independent in shape, and
its weight is `w`-dependent through `eps~(w)` and `1/(jw)`, so a swept fill
evaluates it per frequency, like `loading_for`.

Per formulation:

- bspline: `_loading_gram` with the polynomial coefficients differentiated.
- razor: the path-tested charge term — `phi_x` at the path's two ends, one
  per segment (tent derivative is piecewise constant): a stencil of the same
  shape as the scalar-potential end terms, not the loading stencil.
- SinusoidalGalerkinSolver serves buried decks and carries the same pair,
  but serves no buried LOADING at all today — see below.

### What "exterior medium" means

[R] `_medium_spec.wire_media`: a wire is BELOW only when `zmin < gz - tol`;
a wire resting ON the plane is ABOVE (bspline's stand-off floor requires a
jacketed above wire to clear `h >= b`, so its jacket touches but does not
enter the soil). For those, exterior = air and nothing changes. A buried
wire shallower than `b` would put its jacket through the interface — that is
a partly-buried jacket and neither choice above models it. [I]

### Why SinusoidalGalerkinSolver is not touched

[M] probe2: SG's fully-buried fill swaps `self.k` to the complex `k_m` and
its `_apply_loading` calls `loading_for(self, k * self.c)`, so ANY
distributed loading on a fully-buried SG deck raises `TypeError: float()
argument must be ... not 'complex'` inside `series_impedance_per_wire` (the
mixed deck refuses loading by name, `_MIXED_WIRE_LOADING_REFUSAL`). So SG
serves no buried jacket today, right or wrong. That is a separate defect
(an unnamed crash where a named refusal or a real ω belongs), not fixed
here. The point-matched SinusoidalSolver refuses `buried` at construction.

## 3. Measurements

All on this branch, real constructors, 7 MHz. "old" = the charge term
forced off (`buried_jacket_charge -> None`), which is main's free-space pair.

### probe1 — which half of the pair is wrong [M]

`probe1_pair_in_medium.py` / `probe1.out`: bspline, horizontal dipole 0.5 m
deep, LOSSLESS soil eps~ = 4 or 8, jacket a = 1 mm, b = 3 mm, eps_r = 10
(eps~ < eps_r keeps the in-medium L >= 0 so the exact pair is servable as a
bare wire of radius a'(eps~) + `DistributedRLC(l = L(eps~))`). 5 m dipole,
n = 80, eps~ = 4: today − exact = +0.017 +27.6j Ω; the issue's L-only
proposal (a'_fs kernel + L(eps~)) − exact = −0.019 +22.6j Ω; today + the
charge term (monkeypatched prototype) − exact = −0.0007 −0.046j Ω. So the
defect is the charge half, and the L-only change does not fix it.

### probe3 — the shipped term, both trunks [M]

(a) exact pair, 5 m dipole, eps~ = 4 lossless: |fixed − exact| ≤ 0.046 Ω
(bspline), ≤ 0.017 Ω (razor), n = 20/40/80; |old − exact| ≈ 27.2–28.0 Ω,
flat under refinement.

(b) soil A (13, 5 mS/m), PVC-class jacket a = 1 mm, b = 1.8 mm, eps_r 3.5,
5 m dipole 0.5 m deep. **The correction (fixed − old) at the feed is
−9.7 −j46 Ω** on a ~118 +j18 Ω driving point (bspline n = 160; razor
−9.5 −j45). bspline↔razor gap in the jacket shift: fixed 2.51 → 1.49 →
0.93 → 0.61 Ω (n 20..160, ratios 0.59–0.66); old 0.44 → 0.15 Ω. **The old
term converges bspline↔razor too** — both trunks share the same wrong pair —
so trunk agreement is not a red control for it; the exact-pair gate is.

### probe4 — crossing and screen decks [M]

Run before the refusal was deleted, with `_crossing_jacket_refusal`
monkeypatched to None (on the final branch the method no longer exists and
the probe's patch line is inert). (a) exact pair per wire
(a'_fs + L_fs above, a'(eps~) + L(eps~) below), a = 0.25 mm, b = 0.75 mm,
eps_r 10, eps~ = 4 lossless, m = 1/2/4:

| deck | fixed vs exact, bspline / razor | old vs exact |
|---|---|---|
| crossing_deck | 0.002 / 0.026 → 0.002 Ω | 11.6–12.0 Ω |
| hub_deck(4) (screen) | 0.003 / 0.014 → 0.001 Ω | 5.4–5.5 Ω |
| detached_hub | 0.0004 / 0.002 → 0.0001 Ω | 0.68–0.78 Ω |

(b) soil A, PVC-class jacket (b = 0.45 mm, eps_r 3.5, a = 0.25 mm) on every
wire. Correction at the feed (fixed − old, m = 4, bspline): crossing
−19.5 −j46.9 Ω, hub4 −8.9 +j6.9 Ω, detached_hub +8.9 −j6.5 Ω.
bspline↔razor gap in the jacket shift, m = 1/2/4: crossing 0.68 → 0.30 →
0.15, hub4 0.61 → 0.44 → 0.28, detached_hub 1.79 → 0.86 → 0.28 Ω.

So the crossing refusal is retired: the term is right on a crossing deck by
the reference-free bar, and the trunks converge there.

### probe5 — above ground is bit-identical to main [M]

`probe5_ab_main.py`, run once against `git archive origin/main src` (the
same built `.so`, no C++ changed) and once against the branch, each run
printing which `momwire/__init__.py` it imported: free space, PEC,
refl-coef on all four formulations; Sommerfeld-above on bspline, razor, SG;
a jacketed wire resting on the soil at h = b (bspline); a crossing deck
with a jacket on the MAST only; a buried BARE wire with conductivity; plus
swept runs. All `array_equal`. Two CONTROLS, a buried jacket on bspline and
on razor, differ — so the A/B ran two trees.

### swept == single [M]

`tests/test_jacket_in_soil_1154.py`: < 1e-9 Ω on both trunks, and the sweep
differs from the free-space pair's by > 1 Ω. Not bit-exact on bspline's
FIRST point: ~5e-11 Ω, and the bare crossing deck shows the same residue
with no jacket at all, so it predates this term.

## 4. Open

- NEC-5's `IS` card in a buried wire was not probed (optional instrument).
- The quasi-static thin-jacket reading needs |k_soil|·b ≪ 1 and a jacket
  wholly in one medium. A buried wire shallower than its jacket radius, or
  a jacket crossing the interface at a crossing node, is modelled as if the
  jacket were wholly on the wire's side — the same per-segment medium
  labelling the bare fills use. [I]
- SG's buried loading crash (above) deserves its own issue.
- `jacket_elastance` is not a public export; a consumer mirroring the
  jacket into another tool (antennaknobs' NEC writers) cannot express a
  charge-side term in a NEC card anyway.
