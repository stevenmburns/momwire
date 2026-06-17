# Pulse-Basis (d=0) Extension Plan for `BSplinePySim`

Author: smburns47 (planning session 2026-06-16)
Status: **Plan only — to be executed in a later session.**

## Motivation

We have Triangular (tent), B-spline d=1 and d=2, and Sinusoidal solvers.
Adding a degree-0 (piecewise-constant, "pulse") basis gives us the
classical NEC-style basis and one more data point on the convergence
curve. Question this answers: how much does going from constant → linear
→ quadratic basis actually buy us in observed convergence rate on
fan-dipole, hentenna, bowtie, and Y-fixture targets?

**Goal: extend `BSplinePySim` rather than build a standalone
`PulsePySim`.** Most of `bspline.py`'s machinery (geometry build,
junction bookkeeping, KCL Lagrange-multiplier rows, Galerkin reaction
assembly via `_seg_seg_full_moments` / `_seg_seg_static_moments`,
swept-k cache plumbing, smoothed feed source, ground-plane image
reaction) is degree-agnostic. The d=0 case adds two genuine special
cases (basis support, junction directional-basis convention) and one
obsolete code path (XFEM enrichment is dimension-0 at d=0, see
`_xfem_projection_coeffs`).

Once landed, we get d=0/1/2 all under one solver with a uniform
convergence harness.

## Concrete changes

### 1. Drop the `degree >= 1` guard

`bspline.py:205-211`:

```python
if degree < 1:
    raise ValueError(f"degree must be >= 1, got {degree}")
if degree > 2:
    raise NotImplementedError(
        "degree > 2 needs scripts/derive_bspline_static_moments.py "
        "to be re-run with a larger MAX_D"
    )
```

Change to `if degree < 0` / leave `degree > 2` as is. Update the
docstring's `1 ≤ degree ≤ 2` band accordingly.

### 2. Add `_V_UNIT_INV[0]`

`bspline.py:68-71`:

```python
_V_UNIT_INV: dict[int, np.ndarray] = {
    0: np.array([[1.0]]),  # pulse — single value on each segment
    1: np.array([[1.0, 0.0], [-1.0, 1.0]]),
    2: np.array([[1.0, 0.0, 0.0], [-3.0, 4.0, -1.0], [2.0, -4.0, 2.0]]),
}
```

The d=0 basis is `Φ_m(u) = 1` on its supporting segment, `0` elsewhere.
Polynomial monomial expansion is the constant `1`, so the unit
Vandermonde inverse is just `[[1]]`. With `h^p` column scaling at p=0
it stays `[[1]]` — the vectorized `_build_basis_polynomials` path
already handles this with no special case.

### 3. d=0 basis layout in `_build_basis_polynomials`

For d ≥ 1 the basis has `(N + d - 1)` clamped B-splines per wire
(`N` = segment count, plus `d-1` interior overlap plus 2 boundary
bases). The boundary-basis bookkeeping splits cases by free vs
junction endpoint.

For **d=0** the picture collapses:
- Exactly `N` basis functions per wire, one per segment.
- Each basis has support on exactly one segment ("wing count" = 1).
- There are no boundary bases shared with adjacent edges — every basis
  is "interior" in the support sense.
- Free endpoints: nothing special. Junction endpoints: the segment
  touching the junction has its sole basis act as the directional
  basis for KCL.

Recommended implementation: keep the existing code path but special-case
`if self.degree == 0:` early in `_build_basis_polynomials` and return a
streamlined `(supp_seg, polys, kcl_A, wire_knots, wire_basis_global)`
tuple where:

- `supp_seg[m]` = `[seg_idx_of_m]` (length-1 list).
- `polys[m]` = `[[1.0]]` (single coefficient, p=0).
- `wire_basis_global[w]` = `[(seg_local, basis_global) for seg_local in range(N_w)]`.
- `kcl_A` row at junction j picks out the first or last segment's basis
  for each wire-end attached to j, with sign per the outflow convention.

This avoids invoking `scipy.interpolate.BSpline` machinery (which
needs `k >= 0` and clamped knots — does work for k=0 but the boundary
multiplicity / kept-bases logic doesn't apply).

### 4. KCL directional basis at d=0

For d ≥ 1 the directional basis at a junction is the boundary B-spline
that equals 1 at the junction node and decays into the wire. At d=0
there is no "B-spline that equals 1 at the node" — the boundary value
of a pulse is undefined; the natural convention is to read the
**first/last segment's** basis as the directional one. KCL outflow
sign: same convention as d ≥ 1 — `+1` if the wire's start is at the
junction (current flows out into the segment), `-1` if the wire's end
is at the junction.

`_wire_endpoint_status` already returns the per-wire endpoint mapping;
no change there. The diff lives only in how the directional basis is
*identified* — code becomes "for d=0, the boundary basis IS the
adjacent segment's basis" instead of "B_0 / B_{N+d-1} via knot
vector".

### 5. Feed source at d=0

For d ≥ 1 the source vector is `v_m = Φ_m(s_f)` (delta-gap) or a
smoothed cos² bump.

At d=0 the analog is `v_m = 1 if s_f ∈ supp(Φ_m) else 0` — i.e., pick
the segment containing `s_f`. The smoothed-cos² source generalizes
cleanly to d=0 (just integrate the bump's overlap with each segment),
and gives the same basis-limited-vs-source-singularity tradeoff
discussed in `bspline.py:226-232`.

**Decision point for execution: do we always force delta-gap at d=0,
or implement the segment-overlap smoothed source?** Recommend
implementing the smoothed source — without it the d=0 convergence
study at fan-dipole feed-singularity-dominated geometries will be
contaminated by the delta-gap source error and we won't see the basis
order cleanly. The math is a 1-D bump integrated against the
characteristic function of a segment — closed form via the erf-like
antiderivative of cos²/window, or a few Gauss-Legendre nodes.

### 6. XFEM enrichment unavailable at d=0

`_xfem_projection_coeffs(d)` returns `zeros(d + 1)` when `d < 2`. At
d=0 the bubble subspace is empty *and* there is no interior point on
which a t·log(t) shape could live within a pulse basis (the basis is
constant). At d=0 hard-fail in `__init__` if `use_singular_enrichment`
is set:

```python
if use_singular_enrichment and degree == 0:
    raise NotImplementedError(
        "Singular enrichment is incompatible with degree=0 (pulse "
        "basis). Enrichment requires a polynomial basis with at "
        "least linear support."
    )
```

### 7. Static moments file — already covers d=0

`scripts/derive_bspline_static_moments.py` produces `J_{pq}` moments for
`p, q ∈ {0, ..., MAX_D}`. At d=0 we only need `J_{0,0}` (vector-potential
part) and have **no charge term** at all (the charge moments need
`p ≥ 1, q ≥ 1` from the `Z_Φ` formula `p·q · C[m,a,p] C[n,b,q] · J_{p-1,q-1}`,
which is identically 0 when both basis degrees are 0).

Implication: **at d=0 the entire Z_Φ block is zero** — only Z_A matters.
This is also how classical NEC and TWLD work (the scalar-potential
contribution is absorbed into a near-field correction term, not a
separate static moment). The Galerkin assembly path needs to handle a
zero Z_Φ block gracefully — it should already do so since multiplying
through by `p · q` zeros it out when both are 0.

### 8. Assembly accelerator (`_accelerators.assemble_Z_bspline`)

`_BSPLINE_ASSEMBLE_ACCEL_MAX_D = 2`. Extend the accelerator's per-wing
loops to allow `n_coeffs_per_wing = 1` (already supported on the slow
Python path). Either: (a) raise `MAX_D` to 2 explicitly (no change —
d=0 has 1 coeff, which fits inside the d ≤ 2 buffer); or (b) keep the
constant and verify pure-Python path is hit. Likely a no-op other than
flipping the gate to `if 0 <= self.degree <= 2`.

### 9. UI / engine wrapper

`engines/pysim.py` instantiates `BSplinePySim` directly. The
`PysimEngine` constructor takes `solver=BSplinePySim,
solver_kwargs={"degree": 0}` — should work with no engine change. If
the UI exposes degree as a dropdown, add "0 (pulse)" as an option.

### 10. Profile script update

`scripts/profile_compare_engines.py` should pick up `Bs0` alongside
`Bs1`, `Bs2`. One-line add to the `ENGINES` list.

## Test plan

1. **Unit test: basis values at d=0**
   - Construct a single-wire BSplinePySim with degree=0 and N=4 segments.
   - Assert `_build_basis_polynomials` returns 4 bases, each with
     single-segment support and `polys[m] = [[1.0]]`.
   - Assert `kcl_A` is empty when there are no junctions; assert it has
     one ±1 per wire-end when junctions are present.

2. **Convergence test (smooth target)**
   - Take fan-dipole (smooth geometry, smooth feed when smoothed source
     is used) at a single frequency. Sweep N ∈ {21, 41, 81, 161}.
   - Verify Z(N) → Z* with rate ≈ O(1/N) for d=0, O(1/N²) for d=1,
     O(1/N³) for d=2 (basis-limited, post-feed-smoothing).
   - Note: bare delta-gap will *stall* d=1 and d=2 at O(1/N) and mask
     the basis-order gap — must use smoothed feed for this test to be
     meaningful.

3. **Cross-check against TriangularPySim / PyNEC**
   - At a fixed N, compare R(d=0, N) to PyNEC's (pulse-basis-equivalent)
     answer. Order-of-magnitude agreement expected; sub-1% agreement
     after enough N if both implementations are correct on a smooth
     target.

4. **Junction smoke test**
   - Y-fixture (3-wire K=3 junction). At d=0 verify KCL constraint
     holds in the solved current: outflow sum on the three directional
     bases adjacent to the node sums to 0 within solve tolerance.

5. **Swept-k path**
   - `compute_impedance_swept` over a few frequencies at d=0 should
     reuse the geometry / basis cache the same way d=1/d=2 do (since
     the d=0 special case is fully cacheable on the geom signature +
     degree key).

## Status doc to produce after

After execution, generate a status doc summarizing measured d=0
convergence on each target alongside d=1/d=2/Sinusoidal, with one
plot of |Z(N) - Z(N=∞)| vs N per geometry, all four curves overlaid.

## Open questions parked for execution session

1. Does the existing `_seg_seg_full_moments_offedge` kernel return
   correct values when both bases are pulses (`p = q = 0` for both)?
   It should — it's the polynomial-order-agnostic part — but verify
   on a small hand-derived example.

2. Is the `auto_tap_ratio_threshold` / `enrichment_variant` logic
   reachable when d=0 + enrichment is hard-failed? Confirm the guard
   short-circuits before any of that runs.

3. Does the C++ accelerator silently miscompile when given
   `n_coeffs_per_wing = 1`? If so, gate the accel off for d=0 and let
   it fall through to the Python path.

## Out of scope for this plan

- Higher-degree extensions (d ≥ 3) — separate effort needing
  `derive_bspline_static_moments.py` to be re-run with `MAX_D = 3` or
  higher.
- Replacing TriangularPySim with the d=1 BSplinePySim path — needs the
  batched-swept-k path (currently triangular-only) ported into bspline
  first. Tracked separately.
