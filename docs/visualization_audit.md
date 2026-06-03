# Web UI visualization audit

Trace of where the antenna-canvas and far-field rendering diverges from
what the solver actually computed. Done after PR #51 (the orig=1
enrichment sign fix) to be sure nothing else was actively wrong.

## Pipeline

The solve-to-pixel path is:

1. **Model.compute_impedance(...)** in `src/pysim/{triangular,sinusoidal,bspline}.py`
   returns `(Z_drive, coeffs)`. `coeffs` is in the model's native basis
   layout (interior tents for Triangular, three-term per segment for
   Sinusoidal, polynomial + boundary + enrichment + KCL Lagrange for
   B-spline).
2. **Model.currents_at_knots(coeffs)** evaluates the basis sum at every
   mesh knot of every wire, returning `list[ndarray(complex)]` shaped to
   match the geometry's per-wire knot count. Where the basis is non-zero
   between knots but zero at the knots (B-spline enrichment), that
   contribution is dropped here — that's where most of the gaps live.
3. **server.py `_wire_record(knots, currents, label)`** packs into the JSON
   `{knot_positions, knot_currents_re, knot_currents_im, label}` shape
   the frontend expects.
4. **server.py `_compute_directivity_norm`** computes a normalisation
   constant by integrating `|M_perp|²` over the sphere using
   piecewise-linear segment currents.
5. **Frontend `CurrentCanvas`** strokes each segment with
   `lineWidth = (2 + 6·avg(|I_n|, |I_n+1|)/maxI) · s` and
   `strokeStyle = currentColor(avg)`. **`drawArmEnvelope`** draws an
   offset polyline at amplitude `|I[k]|/maxI · envScale` perpendicular
   to each segment, breaking sub-paths at corners > 3°.
6. **Frontend `FarFieldChart`** computes the radiation cut from the
   same piecewise-linear segment-midpoint currents that the server uses.

Where it matters, the basis evaluation in step 2 is the load-bearing
piece — get that right at the knots and the canvas heatmap and the
envelope have something correct to draw. The B-spline orig=1 sign bug
broke step 2 (and the underlying matrix solve) for "end"-orientation
enrichment bases on hentenna-class geometries.

## Findings

Severity legend:

- 🔴 **Material** — visualization shows a value the solver wouldn't agree with
- 🟡 **Approximation** — known piecewise-linear simplification, exact for tent basis only
- ⚪ **Gap** — physics in the solve that isn't surfaced anywhere on screen
- ✓ **Verified correct** — checked, no issue

### 🔴 Material

**None.** After PR #51, every place I checked sampled the basis sum
correctly at every mesh knot and the L-R symmetry test on
hentenna+enrichment passes at machine precision (`< 1e-6` ratio between
mirror knot magnitudes). All three models' impedance and sweep paths
reproduce the underlying compute_impedance call.

### 🟡 Approximation

#### A1. Far-field uses piecewise-linear current between knots — exact only for Triangular

Both server (`_compute_directivity_norm`, `web/server.py:175`) and
frontend (`FarFieldChart`, `web/frontend/src/App.tsx:2305+`) compute the
radiation pattern by treating each segment as a uniform line current
with

```
I_mid = 0.5 · (I[n] + I[n+1])
dr    = knot[n+1] − knot[n]
phase = k · (r̂ · midpoint)
```

For **Triangular** (tent basis), the current is genuinely linear between
adjacent interior knots, so the midpoint average is exact.

For **B-spline d=2**, the current is quadratic on each segment. The
midpoint-average approximation under-represents intra-segment curvature.
Error decays as O(1/N²); at N=21 it's a few-percent shift on pattern
lobes.

For **Sinusoidal** (NEC2 three-term basis), the current is
const + sin + cos. Same midpoint-average issue, similar magnitude.

Fix would be: sample currents at segment quarter-points and use a
piecewise-quadratic far-field integrator for non-tent bases. Doable in
~50 LOC each side.

#### A2. Heatmap stroke + envelope curve interpolate linearly — same caveat

The canvas paints each segment with a `lineWidth` and `strokeStyle`
based on `avg(|I[n]|, |I[n+1]|)` (`web/frontend/src/App.tsx:2895`). The
envelope curve is a polyline between `(knot[i], offset[i])` points
(`drawArmEnvelope`).

For Triangular, the visual gradient between knots is the true current.
For B-spline d=2 and Sinusoidal, the visual hides intra-segment
curvature. At N≥30 the difference is invisible to the eye; at low N it
under-resolves any peak that happens to sit mid-segment (uncommon since
basis design tends to put peaks at knots).

### ⚪ Gap

#### G1. B-spline enrichment basis is dropped from the heatmap and the far-field

`BSplinePySim.currents_at_knots` (`src/pysim/bspline.py:890`) sums only
the polynomial bases. The enrichment basis Φ_sing(u) = (u/h)·log(u/h)
is non-zero mid-segment (peak ~−0.37 at u/h ≈ 1/e) but exactly zero at
both bounding knots — so the knot values are unaffected. The
midpoint-average heatmap stroke and the far-field path do sample
mid-segment current, but they only see the polynomial part there.

Quantified on hentenna+enrichment at N=21: the enrichment contribution
at segment midpoints adjacent to the K=3 junction at D is **~0.4% of
the polynomial value**. Tiny visually.

This is the v1 limitation explicitly noted in the `currents_at_knots`
docstring. A clean fix is straightforward: take an optional `s_array`
parameter for arbitrary sampling and have the heatmap/far-field call it
at segment midpoints with enrichment included. The fix subsumes A1+A2
for B-spline d=2 specifically — once arbitrary-arc sampling exists, the
heatmap can sample at quarter-points and resolve quadratic curvature
too.

#### G2. KCL Lagrange multipliers from the Schur-complement solve are computed but not surfaced

Triangular's and B-spline's `_solve_with_kcl` both return Lagrange
multipliers `λ` from the augmented `[Z A^T; A 0]` system (these enforce
Σ I_outflow = 0 at each junction). The multipliers have a physical
interpretation as the junction-node potential. The caller takes only
the current coefficients and discards `λ`.

Not a visualization mismatch — just discarded state. If junction
debugging ever matters, the values exist in the solve and aren't
exposed.

#### G3. The ground plane itself isn't drawn

When `ground=true`, wires are lifted by `height_m` and the solve uses
image-method PEC (or PyNEC's reflection coefficient). The canvas shows
the wires at their lifted positions but draws no horizontal line at
`z=0` representing the ground. The user has to infer ground location
from the height-above-ground readout in the controls panel.

Cosmetic gap. The math is correct everywhere; the visualization just
doesn't say "this is the ground". A faint horizontal reference line at
z=0 (visible only in side projections where it's not edge-on) would
close this. ~10 LOC of canvas code.

#### G4. PyNEC ground type isn't visually distinguished from PEC ground

PyNEC slots use Sommerfeld-Norton (or fast reflection-coefficient)
ground with εr=10, σ=0.002 S/m. pysim slots use PEC. The far-field path
already accounts for the correct Fresnel reflection per
`result.ground_eps_r` / `result.ground_sigma`, so the radiation lobes
do differ correctly. The heatmap/envelope don't show any indicator of
which ground model is active.

Probably fine — the gear menu labels it explicitly. Worth checking
visually that switching between PyNEC and Triangular at the same
height produces meaningfully different far-field lobes (it should).

#### G5. Sinusoidal / B-spline `compute_impedance_swept` is a fallback loop, not batched

Both models' swept solves rebind `self.k` per frequency and call
single-k `compute_impedance` (`src/pysim/sinusoidal.py:642`,
`src/pysim/bspline.py:859`). Triangular has a true batched assembly.
Means Sinusoidal/B-spline sweeps are ~N× slower per point.

Not a visualization issue, just performance. Documented in each
method's docstring.

### ✓ Verified correct

- **Feed marker position**: `feedWire.knot_positions[feed_knot_index]`
  lands at y=0.00000 on hentenna's feed wire for all three pysim
  models. The geometric center is correctly projected to canvas center
  via `hC = (hMin + hMax) / 2`.
- **Fan-dipole feed wire packaging** (`_fandipole_pack_wires`): the
  synthetic 3-knot record's currents come from
  `sim.currents_at_knots(coeffs)[0]` directly, so junction-directional
  values at T and S flow through naturally.
- **Hentenna and Yagi traversal**: order of the upper polyline
  (B → A → C → D) is L-R symmetric in the data; the visual symmetry now
  holds at machine precision after PR #51.
- **Per-knot heatmap colors and envelope amplitudes**: at each knot
  these ARE the basis evaluation — no accidental scaling or sign flips.
- **Inverted-V apex feed**: `feed_knot_index = n_per_wire` correctly
  points to the midpoint of the 2N+1-knot polyline (the apex).

## Recommended follow-ups, in order of value

1. **G1 + A1 + A2 combined** — extend `currents_at_knots` to take an
   optional `s_array` per wire (or sample at segment quarter-points for
   the heatmap stroke and at finer points for the far-field integrator).
   Would handle B-spline d=2 and Sinusoidal exactly *and* include the
   enrichment shape. Single coherent change spanning model + server +
   frontend.
2. **G3** — draw a thin ground-line at `z=0` when `result.ground === true`.
3. **G4** — quick visual sanity check on PyNEC vs Triangular ground
   patterns at the same height. No code change expected, just
   verification.
4. **G2** — probably no action. Lagrange multipliers are debugging-grade
   info, not user-facing.

None of these are urgent. The orig=1 fix in PR #51 was the only actively
wrong thing this audit surfaced.
