# Sinusoidal basis MoM — design notes

Pulled from the NEC2 Theory Manual (Burke & Poggio, LLNL UCID-18834, 1981 —
`docs/nec2_theory_manual.pdf`). Equation numbers below reference that manual.

The goal of this implementation is **scientific**: not a re-creation of the
NEC2 code, but a from-the-spec implementation we can compare with PyNEC /
nec2c on the hentenna to learn whether the X drift documented in
`NEXT_STEPS.md` item 13 is reproduced by the basis itself or by some other
piece of NEC's machinery.

## Scope

* Free space only (no Sommerfeld ground, no PEC image, no patches/MFIE).
* Thin-wire kernel only (Eq. 68-72 / 75-79); no extended thin-wire kernel.
* Applied-E "delta-gap" source only (Eq. 187); no slope-discontinuity source.
* Wires with arbitrary 3D polylines, junctions at endpoints between wires.
* Same wire radius `a` everywhere (uniform-radius simplification — the
  hentenna is uniform radius).

Out of scope (deliberate): networks, loads, transmission lines, ground plane,
extended thin-wire, magnetic-frill source, ratings of segment ≥ 2a etc.

## The continuous problem (Section II.1)

EFIE on the wire-axis filament (Eq. 7):

    -ŝ · E^I(r) = -jη/(4πk) ∫_L I(s') (k² ŝ·ŝ' - ∂²/∂s∂s') g(r,r') ds'

with `g(r,r') = exp(-jk|r-r'|)/|r-r'|`, `η = √(μ₀/ε₀)`, `k = ω√(μ₀ε₀)`.
We test at segment centers (collocation) on the wire surface (Eq. 18 with
`w_i(r) = δ(r-r_i)`, `r_i` at the surface of segment `i`).

## Basis function shape (Section III.1)

On segment `j` the current is three-term sinusoidal (Eq. 20):

    I_j(s) = A_j + B_j sin k(s-s_j) + C_j cos k(s-s_j),  |s-s_j| < Δ_j/2

with `s_j` = segment center arclength, `Δ_j` = segment length.

The **i-th basis function** `f_i` is the unique current shape whose support
is segment `i` plus every segment connected to segment `i`'s two ends. On
each segment in the support the shape is three-term sinusoidal; coefficients
across segments are fixed by:

1. zero current and zero derivative at outer ends of the support (Eqs. 34, 35);
2. for a free wire end of the segment: zero current via `X_i = 0` (Eq. 23
   reduced); otherwise on a regular wire end (Wu-King): `∂I/∂s = a±·Q±`
   (Eq. 24);
3. Kirchhoff's current law on the central segment's two endpoints (Eqs. 41, 42);
4. amplitude normalization `A_i^0 = -1` (Eq. 49).

After these conditions there is exactly one unknown amplitude per segment,
so the global matrix is `N_segs × N_segs`. Junction continuity / KCL is
**baked into the basis function shapes**, not enforced via Lagrange rows.

### Closed-form per-segment coefficients (interior, N⁻ ≠ 0 and N⁺ ≠ 0)

Quantities used everywhere:

    γ = 0.5772156649…           (Euler-Mascheroni, Eq. 22)
    a_i± = [ln(2/(ka_i)) - γ]⁻¹  (Eq. 25)
    X_i = J_1(ka_i) / J_0(ka_i)  (set to 0 at a free end)

Sums over the segments connected at each end of segment `i`:

    P_i⁻ = Σ_{j ∈ N⁻} [(1 - cos kΔ_j)/sin kΔ_j] · a_j⁺     (Eq. 62)
    P_i⁺ = Σ_{j ∈ N⁺} [(cos kΔ_j - 1)/sin kΔ_j] · a_j⁻     (Eq. 63)

(`N⁻` segments connect to end-1 of segment `i`; `N⁺` connect to end-2.)

End-charge amplitudes:

    D = (P_i⁻P_i⁺ + a_i⁻a_i⁺) sin kΔ_i + (P_i⁻a_i⁺ - P_i⁺a_i⁻) cos kΔ_i
    Q_i⁻ = [a_i⁺(1 - cos kΔ_i) - P_i⁺ sin kΔ_i] / D       (Eq. 52)
    Q_i⁺ = [a_i⁻(cos kΔ_i - 1) - P_i⁻ sin kΔ_i] / D       (Eq. 53)

On segment `i` itself:

    A_i⁰ = -1                                             (Eq. 49)
    B_i⁰ = (a_i⁻Q_i⁻ + a_i⁺Q_i⁺) · sin(kΔ_i/2) / sin kΔ_i  (Eq. 50)
    C_i⁰ = (a_i⁻Q_i⁻ - a_i⁺Q_i⁺) · cos(kΔ_i/2) / sin kΔ_i  (Eq. 51)

On each `j ∈ N⁻` (segment connected at end-1 of segment `i`):

    A_j⁻ = a_j⁺ Q_i⁻ / sin kΔ_j                            (Eq. 43)
    B_j⁻ = a_j⁺ Q_i⁻ / (2 cos(kΔ_j/2))                     (Eq. 44)
    C_j⁻ = -a_j⁺ Q_i⁻ / (2 sin(kΔ_j/2))                    (Eq. 45)

On each `j ∈ N⁺` (segment connected at end-2 of segment `i`):

    A_j⁺ = -a_j⁻ Q_i⁺ / sin kΔ_j                           (Eq. 46)
    B_j⁺ = a_j⁻ Q_i⁺ / (2 cos(kΔ_j/2))                     (Eq. 47)
    C_j⁺ = a_j⁻ Q_i⁺ / (2 sin(kΔ_j/2))                     (Eq. 48)

### Free-end edge segment (N⁻ = 0, N⁺ ≠ 0)

When end-1 of segment `i` is a free wire end:

    A_i⁰ = -1                                              (Eq. 54)
    B_i⁰ = sin(kΔ_i/2) / [cos kΔ_i - X_i sin kΔ_i]
         + a_i⁺ Q_i⁺ · [cos(kΔ_i/2) - X_i sin(kΔ_i/2)] / [cos kΔ_i - X_i sin kΔ_i]  (Eq. 55)
    C_i⁰ = cos(kΔ_i/2) / [cos kΔ_i - X_i sin kΔ_i]
         + a_i⁺ Q_i⁺ · [sin(kΔ_i/2) + X_i cos(kΔ_i/2)] / [cos kΔ_i - X_i sin kΔ_i]  (Eq. 56)
    Q_i⁺ = [cos kΔ_i - 1 - X_i sin kΔ_i] / [(a_i⁺ + X_i P_i⁺) sin kΔ_i + (a_i⁺ X_i - P_i⁺) cos kΔ_i]   (Eq. 57)

The N⁺ ≠ 0 segments use Eqs. 46-48 as before. `X_i = 0` for a free end of a
wire whose end is treated as zero-current; if the end-cap-current treatment
is used, `X_i = J_1(ka)/J_0(ka)` — that is the more physical option, used
by NEC for thick wires; for our hentenna at `ka` ~ 1e-4 the two are
numerically indistinguishable so we use `X_i = 0`.

### Free-end edge segment (N⁻ ≠ 0, N⁺ = 0)

Symmetric (Eqs. 58-61); easy to derive by negating arc-length.

### Isolated segment (N⁻ = N⁺ = 0)

Single segment with both ends free (an isolated short dipole — never present
in our test geometries). Eq. 64 gives the closed-form shape.

## Field of an elementary current segment (Section III.3)

In a local cylindrical frame where the source segment lies on the z-axis
between `z_1` and `z_2`, the field of a current `I_0 · f(z')` on the
segment is (thin-wire kernel, `r_0 = √(ρ² + (z-z')²)`, `G_0 = exp(-jkr_0)/r_0`):

* **Constant** `I_0`:

      E_ρ^f = -I_0/λ · jη/(2k²) · [(1+jkr_0) ρ G_0 / r_0²]_{z_1}^{z_2}                     (Eq. 78)
      E_z^f = -I_0/λ · jη/(2k²) · {[(1+jkr_0)(z-z') G_0 / r_0²]_{z_1}^{z_2} + k² ∫_{z_1}^{z_2} G_0 dz'}   (Eq. 79)

* **Sine** `I_0 sin kz'`:

      E_ρ^f = -I_0/λ · jη/(2k²ρ) · {G_0 · k(z-z') · cos kz'
                                    + [1 - (z-z')²(1+jkr_0)/r_0²] · sin kz'}|_{z_1}^{z_2}  (Eq. 76)
      E_z^f =  I_0/λ · jη/(2k²)  · {G_0 · k · cos kz'
                                    - (1+jkr_0)(z-z')/r_0² · sin kz'}|_{z_1}^{z_2}        (Eq. 77)

* **Cosine** `I_0 cos kz'`: swap sin↔cos and negate the sin-derived term —
  same parenthesized expressions in Eqs. 76, 77 (the manual writes
  `(cos kz' / -sin kz')` as the toggle in the upper/lower bracket).

The sin/cos field expressions are **closed-form, no integral**, because the
operator `(d²/dz'² + k²)` annihilates `sin kz'` and `cos kz'`, killing the
non-trivial `k² ∫ G_0` term that the constant component carries.

For our local frame we re-center on the segment midpoint, so the trig
arguments are `k(s' - s_n)` not `kz'`. The change of origin shifts the
"bracket evaluated at z_1/z_2" pair by a constant offset; the structure is
unchanged.

The radial coordinate `ρ` is the perpendicular distance from the source
segment axis to the observation point on the **surface** of the observation
segment. NEC uses `ρ' = √(ρ² + a_obs²)` where `a_obs` is the observation
segment's radius (cylindrical correction for the offset between observation
segment axis and surface) and projects `E_ρ` onto the observation tangent
direction via the angle correction `ρ/ρ'`.

## Matrix assembly

For each (i, j):

    G_ij = -ŝ_i · E^scat( evaluated at r_i ; current = basis-function j )
         = Σ_{n ∈ supp(j)} ŝ_i · [A_{j,n} E^const_n(r_i) + B_{j,n} E^sin_n(r_i) + C_{j,n} E^cos_n(r_i)]

where `E^x_n(r_i)` is the elementary field at the **center of segment i** on
its surface, of the corresponding shape current on segment `n`, expressed in
global Cartesian after rotating out of the local cylindrical frame of
segment `n`.

This decomposes into a **once-per-(n,i) pair tensor** `Φ[ABCsincos, n, i]` of
shape `(3, N_segs, N_segs)` — the field of unit-amplitude const/sin/cos
current on segment `n` projected onto `ŝ_i` at center of `i` — and a
**once-per-(j,n) coefficient triple** `(A_jn, B_jn, C_jn)` that is non-zero
only on `n ∈ supp(j)`. We compute the tensor once, then form
`G = Φ ⋅ coeff` as a sparse matvec over the support pattern.

The peak segment of basis `j` always contributes (n=j); each adjacent
segment contributes via the N⁻/N⁺ coefficient blocks above.

For multiple wires meeting at a junction, the basis function for segment `i`
adjacent to the junction has N⁻ or N⁺ segments equal to the number of
*other* wires' adjacent segments, plus its own continuation if any. The KCL
sum at the junction (Eqs. 41, 42) is what determines the coefficients on
those neighbouring segments.

## Source vector (Section V.1, Eq. 187)

For a voltage source `V` on segment `m`:

    E_m = V / Δ_m
    E_i = 0  (i ≠ m)

This is the applied-E "constant-field" delta-gap. Sign: positive end of the
source points along `+ŝ_m`.

## Per-wire radius (momwire#147)

`wire_radius` accepts a scalar (every wire) or a length-n_wires sequence
(each wire's own conductor radius). Two conventions, both transcribed from
nec2c/necpp and validated against PyNEC on mixed-radius geometries:

1. **Basis end-condition constants (TBF).** The per-segment constant
   `a_seg = 1/(ln(2/(k·a_seg)) − γ)` uses each segment's OWN radius.
   nec2c's `tbf()` computes `aj` from `bi[jcox]` for every connected
   segment — so the P sums and the N± neighbour coefficient entries take
   the constant at the *neighbour's* segment — and resets `aj = ap =` the
   self constant before the Q/D/B₀/C₀ formulas, so the self-segment
   formulas use only the basis's own radius. At a junction of wires with
   different radii each member contributes its own constant to the P sums.

2. **Field kernel offset (EFLD).** The source current stays a filament on
   the source axis; the boundary condition is enforced on the OBSERVER
   segment's surface: `ρ' = √(ρ² + a_obs²)` with `a_obs` the radius of the
   segment the field is evaluated on (necpp passes
   `ai = segment_radius[i]`, the observer, into `efld`). Self terms use
   the wire's own radius; mutual terms between wires of different radii
   use the observer wire's. The opposite convention (source radius,
   observation on the axis) was tried first and refuted by the oracle:
   on a two-radius dipole the PyNEC delta grew from ~0.8 Ω at N=21 to
   ~11.6 Ω at N=41 — the in-line near-junction pairs are exactly where
   the two conventions diverge. With the observer convention the delta
   is ~0.3 Ω and stable under refinement, inside the single-radius
   fat-wire baseline (~0.44 Ω).

Scalar (and uniform-array) radii keep the historical scalar code paths and
are bit-identical to pre-#147 results. The C++ field-tensor kernels
(`sinusoidal_field_tensor`, `sinusoidal_field_tensor_refl`) take one
scalar radius — the OBSERVER row's — so mixed-radius solves dispatch one
kernel call per contiguous constant-radius run of observer rows and
stitch the results (segments are wire-contiguous, so runs are at most
the wire count; no numpy fallback penalty).

**The BSpline (Galerkin) family** applies the same observer-surface
convention through the a²-regularized moment kernel: each observer ROW of
`_seg_seg_full_moments_offedge` uses its wire's own radius (per-row `a`
argument; C++ served one constant-radius row-run at a time), and same-edge
blocks — always single-wire — use that wire's radius.

**The fast solvers (HMatrixSolver, ArrayBlockSolver)** inherit the same
convention through their block fills: numpy block evaluators pass the
per-observer-row radius slice, same-edge bands use their edge's wire
radius, and the fused C++ off-edge assembler (which regularizes with one
scalar a²) is dispatched one constant-radius observer-basis group per
admissible block. Two consequences are specific to the array solver:
per-segment radii join the element shape signatures and the module-scope
self-block cache keys (translation-identical elements with different
radii must not share a block), and the complex-symmetry coupling shortcut
`Z_ba = Z_ab^T` only fires when both elements of the pair carry one and
the same radius — the observer-row regularization makes the mutual block
(slightly) asymmetric otherwise.

**NEC-2 is not a converged reference at an in-line radius step.** On a
two-radius dipole (arms joined end-to-end, fed away from the step), PyNEC
does not converge under refinement: R drifts ~+2.4 Ω per mesh doubling at
a 10:1 step (146.4 → 153.8 Ω over N=21→161) and ~+0.4 Ω per doubling even
at a mild 2:1 step, with no sign of settling — the classic NEC-2
stepped-radius deficiency (the three-term basis's junction condition
mishandles the charge-distribution jump; the reason stepped-diameter
correction schemes exist in Yagi modeling). momwire's SinusoidalSolver,
which implements NEC's basis, TRACKS PyNEC point-for-point through this
drift (|Δ| = 0.5 → 0.3 Ω, shrinking with N) — that is the parity
criterion for it. The Galerkin BSpline family instead converges cleanly
at the step (134.30 → 134.52 Ω over the same range) and its answer is
basis-degree-independent (d=1 vs d=2 within ~0.1 Ω at N=81), so its
mixed-radius validation rests on (a) cross-degree consistency at the
step, and (b) direct PyNEC parity on mixed-radius JUNCTIONS (fat vertical
+ thin radials: ~0.5 Ω, stable under refinement), where NEC converges.

## Output

After solving `G α = E`, the basis-function amplitudes `α_j` are known.
The current at the center of segment `m` is:

    I(s_m) = Σ_{j : m ∈ supp(j)} α_j · f_{j,m}(s_m)
           = Σ_{j : m ∈ supp(j)} α_j · [A_{j,m} + B_{j,m}·0 + C_{j,m}·1]
           = Σ_{j : m ∈ supp(j)} α_j · (A_{j,m} + C_{j,m})

(at `s = s_m` the local arc-coordinate offset is zero, so `sin = 0`, `cos = 1`.)

The driving-point impedance is `Z = V / I(s_feed_center)`.

## Implementation plan

Three phases:

1. **Straight dipole, free-space, uniform segments.** All segments have the
   same length Δ and radius a, so a_i± and X_i collapse to scalars. The
   formulas above become trivially indexable. Verify by replicating the
   NEC2 dipole values in `docs/convergence_analysis.md` (69.64 - j18.21 Ω at
   13.627 MHz, half-driver 5.291 m, r = 0.5 mm).

2. **Bent single wire (e.g. inverted V).** N⁻=N⁺=1 for every interior
   segment but tangent directions differ across the kink. The basis
   function coefficients are still interior-formula but the cross-segment
   field-evaluation must respect each segment's local frame.

3. **Multi-wire with junctions (hentenna).** N⁻ or N⁺ > 1 at segments
   adjacent to the junction node. The hentenna has three wires meeting at
   the T and S junction nodes — that's K=3 junction multiplicity. The
   basis-function for a segment adjacent to such a junction extends onto
   *every* wire passing through the junction.

The deliverable test is: run the same hentenna geometry as
`NEXT_STEPS.md` item 13 (params_50, 28.47 MHz, r = 0.5 mm, uniform N
segments per non-feed edge) and report the per-N convergence of R + jX.
Compare to PyNEC's tabulated values.

## Plug-in point in pysim

The existing `pysim.TriangularPySim` (in `src/pysim/triangular.py`) is the
default solver and stays the default. The sinusoidal solver will be a peer
class `pysim.SinusoidalPySim` with the same constructor shape (wires,
n_per_edge_per_wire, feed_wire_index, feed_arclength, wavelength,
wire_radius, junctions, ...) and the same primary entry point — a method
that returns the driving-point impedance.

Web-UI integration is deferred — first prove the algorithm on the hentenna
sweep at the script level.
