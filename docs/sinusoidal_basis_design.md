# Sinusoidal basis MoM — design notes

Pulled from the NEC2 Theory Manual (Burke & Poggio, LLNL UCID-18834, 1981 —
`docs/nec2_theory_manual.pdf`). Equation numbers below reference that manual.

The goal of this implementation is **scientific**: not a re-creation of the
NEC2 code, but a from-the-spec implementation we can compare with PyNEC /
nec2c on the hentenna to learn whether the X drift documented in
`NEXT_STEPS.md` item 13 is reproduced by the basis itself or by some other
piece of NEC's machinery.

## Scope

(The v1 scope, kept as a record of what the from-the-spec build set out to
prove. Several bullets have since been lifted by later arcs and are annotated
where a section below carries the design.)

* Free space only (no Sommerfeld ground, no PEC image, no patches/MFIE).
* Thin-wire kernel only (Eq. 68-72 / 75-79). *(Superseded: the extended
  thin-wire kernel is served on both sinusoidal solvers — see "The extended
  thin-wire kernel (Eqs 84–98)" below.)*
* Applied-E "delta-gap" source only (Eq. 187); no slope-discontinuity source.
* Wires with arbitrary 3D polylines, junctions at endpoints between wires.
* Same wire radius `a` everywhere (uniform-radius simplification — the
  hentenna is uniform radius). *(Superseded: per-wire radius, momwire#147,
  below.)*

Out of scope in v1 (deliberate): networks, loads, transmission lines, ground
plane, extended thin-wire, magnetic-frill source, ratings of segment ≥ 2a etc.

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
(`sinusoidal_field_tensor`, `sinusoidal_field_tensor_refl`, and their
extended-thin-wire twins `sinusoidal_field_tensor_ek` /
`sinusoidal_field_tensor_ek_refl`) take one
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

## The extended thin-wire kernel (Eqs 84–98)

Everything above is NEC's *reduced* ("thin-wire") kernel: the source current
is a filament on the wire axis, the observer sits on the wire surface, and
the conductor's girth survives only as Eq. 84's regularization
`R = √((z-z')² + a²)`. The EXTENDED kernel (NEC's EK card; momwire#233 on
the point-matched solver, momwire#246 on the Galerkin one) instead keeps the
current as a uniform tube of surface current at `ρ' = a` and averages the
free-space Green's function over the circumference (Eq. 85). Eqs. 86-88
expand that average in a Taylor series about the axial filament and truncate
at second order — exact to O(a²/R²), and, crucially, reintroducing the
`ρ' ≠ 0` terms the reduced kernel drops. Eq. 89 is the resulting scalar
kernel and Eqs. 90-98 are its z- and ρ-derivatives, the six per-end
quantities the field expressions consume. This is what makes fat conductors
— Δ/a below ~3 — answerable at all.

### Eq. 89 as a factor of the reduced kernel (momwire#249)

The re-derivation both solver families build on (module comment in
`src/momwire/_bspline_kernels.py`): write Eq. 89 as

    G_ek = G_red · fac(R; a, k)
    fac  = 1 + T1·C2 − T2·C1
    C1   = 1 + jkR          C2 = 3·C1 − (kR)²
    T1   = a²ρ²/(4R⁴)       T2 = a²/(2R²)

i.e. the O(a²) truncation of the azimuthal tube average, seen from an
observer a distance ρ off the source axis. momwire extends only COAXIAL
EQUAL-RADIUS pairs (`_ek_axis_groups`, NEC's own thresholds: same line to
|t·t'| ≥ 1 − 1e-6, radii equal to 1e-6 relative), and on those the whole
thing collapses: the observer sits on its own wire's surface on the same
axis, so ρ = a, the tube radius is a, and R = √(ζ² + a²) is *the same
regularized R the reduced kernel already computes*. Eq. 89 becomes a scalar
multiplicative factor of R alone (`_ek_factor`), manifestly symmetric in
i ↔ j and manifestly → 1 as a → 0. NEC's IRA swapped arm is unreachable in
this specialisation (the test `ρ_eval < b` is strict and `ρ_eval = b = a`).

### The point-matched route: per-end substitution (momwire#233/#245)

NEC implements Eqs. 84-98 as `GXX`, which stands in for the reduced-kernel
`GX` at ONE END of ONE SOURCE SEGMENT at a time, and `EKSCX`, which is
`EKSC` with GX swapped for GXX per end plus a correction to the
constant-current term. momwire's collocation solver transcribes EKSCX
directly (`SinusoidalSolver._extended_kernel_fields`; C++ twins
`sinusoidal_field_tensor_ek` / `sinusoidal_field_tensor_ek_refl`), dropping
into the same per-endpoint bracket slots as the reduced Eqs. 76-79 build.

EKSCX's IRA arm — the `RHX < BX` swap of f.3186-3192, where the observation
point falls inside the source conductor and "distance" and "radius" trade
places — is chosen PER PAIR on both backends (momwire#258). Through #245 and
#259 it was one flag for the whole fill (`np.any` over the (M, N) grid), so a
single inside-conductor pair put every pair on the IRA == 1 formula. That is
invisible on a collinear stepped-radius deck, where the arm rewrites only the
ρ-flavoured slots and the ρ-projection factor is identically zero, and worth
7% of Z against nec2c as soon as a skew member joins the deck. The C++ entry
points therefore take no IRA argument at all: they resolve it from the same
per-pair comparison that orders (rh, b), which is the only spelling in which
the two cannot disagree at a knife-edge pair. The coaxial specialisation above
is unaffected — it never reaches the arm.

Gating is NEC's, per source-segment END (`_ek_gating`, the IND1/IND2
codes): a free end extends (IND 1); a two-segment junction whose partner is
collinear and of equal radius extends (IND 0), as does a perpendicular
ground contact, where the image continues the wire straight through;
everything else — a bend, a radius step, a non-perpendicular contact, a
K ≥ 3 junction — keeps the reduced kernel at that end (IND 2). The docstring
at `_ek_gating` maps each code to the nec2-1.2.1.2.f lines it transcribes.

### The Galerkin route: a smooth delta, added (momwire#246)

The Galerkin fill's reduced path is #205's folded closed forms, whose
cancellation discipline is load-bearing; substituting per-end closed forms
into it was rejected outright. Instead the reduced fill is computed exactly
as it always was and `SinusoidalSolver._folded_ek_delta_fields` ADDS a
Gauss-Legendre quadrature of the extended-minus-reduced delta on the
eligible pairs, with the folded source shape evaluated POINTWISE as
−2·sin²(kξ/2). Nothing is subtracted anywhere, so the fold's discipline
never comes up, and an ineligible pair comes back bit-for-bit the reduced
fill's.

**The delta kernel.** Write the reduced kernel as a function of u = R²,
`g(u) = e^{−jkR}/R`, whose u-derivatives are the reverse Bessel polynomials:
`g⁽ⁿ⁾(u) = (−½)ⁿ·e^{−jkR}·Aₙ(jkR)/R^{2n+1}`. Averaging over the source tube
with `R(φ)² = u + a² − 2aρ·cos φ` — the same expansion as Eqs. 86-88, kept
in terms of the moments ⟨R²−u⟩ = a², ⟨(R²−u)²⟩ = a⁴ + 2a²ρ² — gives the
delta this integrates:

    W(ρ, ζ) = a²·g′(u) + a²·ρ²·g″(u)

At ρ = a — which is what eligibility MEANS — W is Eq. 89's `(fac − 1)·G_red`
term for term. Keeping the ρ² rather than substituting a² for it changes
nothing on the pairs served, but it is the difference between a right and a
wrong E_ρ: E_ρ differentiates the kernel in ρ, and Eq. 89's factor form,
being a function of R and a alone, has no honest ρ-derivative to give.
Measured, substituting a² first lands E_ρ at HALF the exact circumferential
average and half of NEC's own EKSCX; W reproduces both. E_z is untouched by
the choice (∂/∂z reaches u only through ζ). The field operators are the
reduced path's own — `E_z[s] = −pref_z·∫ s·(k² + ∂²_z)W dξ`,
`E_ρ[s] = −pref_z·∫ s·∂²_{ρz}W dξ` — sympy-derived and proved against the
shipped reduced closed forms in `scripts/derive_galerkin_ek_delta.py`, so no
sign or prefactor is fitted.

**The quadrature variable, and why it is not ξ.** W is bounded (at ζ = 0 it
tends to ¼·G_red(a)) and analytic along the whole source segment, its
nearest pole — the reduced kernel's own ζ = ±jρ — a full wire radius off the
real axis. But "a wire radius off the axis" governs convergence only once
the path is measured in radii: in ξ the delta is a spike of width ρ inside a
segment of half-length H, so a fixed rule's accuracy is set by ρ/H and
collapses on exactly the pairs a fill cares most about — the self pair and
its neighbours, where the observer sits ON the source segment. Measured, a
plain 16-node ξ rule at the self pair is wrong by 5× the answer at Δ/a = 6
and by 1e6× at Δ/a = 122. The integration therefore runs in the sinh-mapped
variable

    ζ = ρ·sinh t,   R = ρ·cosh t,   t ∈ [asinh((z−H)/ρ), asinh((z+H)/ρ)]

— R in closed form, never through a difference of squares — in which the
spike is O(1) wide whatever ρ/H is, the kernel's poles sit at t = ±jπ/2
independent of ρ, and the interval's half width grows only like ln(2H/ρ),
covered by a composite rule of `n_panels` equal panels of 16 nodes
(`_ek_delta_rule`). Refining means adding panels, not nodes, and one rule
serves every wire thickness.

**Pair rule, not per-end gating.** Eligibility on the Galerkin fill is per
PAIR — coaxial and equal radius (`SinusoidalGalerkinSolver._ek_pairs`, label
scan shared verbatim with `_bspline_kernels._ek_axis_groups`) — and NOT
NEC's per-end IND codes. Transplanted into a Galerkin fill, a per-END
decision depends on which segment is the SOURCE, so G(i, j) would be
extended while G(j, i) was not, and ‖G−Gᵀ‖/‖G‖ — the reciprocity residual
this solver family uses as its error detector — would stop measuring
anything. The pair rule is symmetric by construction, reproduces NEC's
decision on straight wires and on perpendicular ground contacts (via the
mirrored source, scored in ONE joint scan over real ∧ mirrored segments so a
horizontal wire and its offset image are never mislabelled coaxial), and is
strictly MORE conservative at bends, radius steps and K ≥ 3 junctions, where
NEC still extends the cross-arm pairs — worth ~1 % of Z at Δ/a = 2 and O(h)
under refinement (momwire#249 §4.3).

**Near/far tiers.** The panel count is the second tier of the near/far split
the Galerkin test quadrature already runs on: coaxial pairs whose observer
can sit inside the source segment's span are pairs whose segments overlap,
i.e. separation zero, i.e. NEAR pairs — so the split by near-ness IS the
split by whether the spike is inside the integration path. Near pairs take
the dense rule (8 panels, measured converged over Δ/a from 1 to 500); far
pairs are converged on one. This is also why
`SinusoidalGalerkinSolver(extended_kernel=True)` requires
`near_correction=True`: under EK the near path is not a refinement but where
the on-segment pairs are computed at all.

**The floor.** Reduced-plus-delta is a near-cancelling decomposition — the
two kernels agree away from the wire, so the delta's whole-line integral
very nearly vanishes and an on-segment pair carries ~(H/ρ)² of cancellation;
float64 leaves ~ε·(H/ρ)² of the delta's peak behind. The EK shift is itself
O((ρ/H)²), so the two scale against each other and the error that reaches Z
stays ~1e-10 relative out to Δ/a ≈ 500; past Δ/a ≈ 1e4, where the kernel is
a 1e-8 effect anyway, the decomposition is noise-limited. That is a property
of reduced-plus-delta, not of the quadrature, and it bounds how thin a wire
the decomposition can resolve an EK correction for.

**What is served, what refuses.** Free space, the PEC image and the
reflection-coefficient image, plus the graded near-pair correction on each
(the Fresnel dyad is a per-pair weight applied AFTER the field tables, so
the delta rides through it exactly as the reduced field does). The
junction/node lumped-charge blocks stay reduced — their source is a point
charge at a node, which has no tube to average over. The Sommerfeld
remainder under EK refuses (its delta story is a separate validation arc —
momwire#287); `BSplineSolver` serves every ground under EK with the
remainder deliberately reduced on a measured O((a/2h)²) argument
(momwire#269, `bspline.py` class docstring). Measured results — the nec2c
ladder shift, the symmetry ratios, the thin-rung deviation — are §20 of
`docs/sinusoidal-galerkin-instrument-report.md`.

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
