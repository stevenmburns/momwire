# momwire#1149 U2 — the crossing node: derivation and measurements

Labels: **[M]** measured here (log named), **[R]** read from the code,
**[I]** inferred. All runs in this worktree's venv (`momwire.__file__`
asserted into the worktree by every probe), accelerators built with
`make build` (`MOMWIRE_REQUIRE_ACCEL=1`).

## 1. The defect, restated

On a crossing deck razor fills each tent that spans the plane as two HALF
tents, one per medium (`_medium_geometry`: the other wing becomes a ghost at
σ = 0), and assembles four masked terms (`_assemble_Z_crossing`):

    Z[R_a, C_a] += above family (direct − [C₂·image + Q_a])
    Z[R_b, C_b] += below family (direct − [A_m·image + Q_b]) at k_m, ε_m
    Z[R_a, C_b] −= cross block        (path rows A × Galerkin cols, corner off)
    Z[R_b, C_a] −= reversed cross block

Scoping [M]: flat 2-port non-reciprocity (3.2e-2 soil A), vanishes at
ε̃ = 1, grows with contrast, needs an end in the plane, non-radiating
(η·R right, R_in high).

## 2. Where each term puts a half tent's charge [R]

Razor's row m is the path integral of E from centroid(A) through knot m to
centroid(B):

    Z_mn = jωμ·T1 − T2/(jωε),   T2 = Σ_ends sign·Φ̃_n(end)

with Φ̃ in current-derivative units (T2 sums `q = dΛ/dτ` line densities
against segment moments of g = e^{−jkR}/4πR, R = √(d² + a²)).

A half tent of value σ at the node has LINE charge ∫q ds = σ·(±1) on its
live wing and, because its current stops at the node, a POINT charge

    Q_node = −(q_live · h_live) = −σ (wing rising into the knot), +σ (falling)

For the two halves of one tent, `Qa = −Qb` (the whole tent carries none).
So any term may keep both or drop both. Who does what:

| term | node charge | why [R] |
|---|---|---|
| family direct (T2 over `q`) | dropped | T2 sees only the wing doublets; the ghost wing has q = 0 |
| family image (w_Φ = C₂ / A_m) | dropped | same T2, mirrored sources |
| family remainder Q | **KEPT** | `Remainder.field_windows` integrates the field of unit current MOMENTS over the half tent's current; such a field includes every charge the current implies (it is the field of a charge-conserving source) |
| cross block (path rows) | dropped | `Fd_A = 0` on a path axis empties SQ; BT reads only the source LINE charges (`Fd_B·V`); corner off |
| reversed cross block | dropped | the same, transposed roles |

bspline, for contrast, keeps it everywhere [R]: its `self_completions` add
β_dir/β_img `bnd + corner` (the node charge in each family's direct+image),
and its cross blocks carry SQ and the corner. Its remainder carries it
already. That is the complete convention.

## 3. The error, and the correction [R + I]

For an above observer row, the node-charge content of razor's assembly is

    Qa·Φ_rem,a(e)          (from Q_a; nothing from direct/image or cross)

where Φ_rem,a = Φ_exact − Φ_dir+img is the above family's remainder potential
of a unit node charge. For a below row it is Qb·Φ_rem,b(e). The families'
remainders are different functions, so these do not cancel as a continuous
current requires. Both vanish at ε̃ = 1 (no remainder), grow with contrast,
are a fixed unit charge whatever the mesh (flat), and are near-field
potentials (non-radiating): every signature in the scoping.

Remove it (drop convention), per crossing tent n, per row m with T2
endpoints e and razor's signs:

    above rows: ΔZ_mn = −Qa · Σ_e sign_e · [fam_a(e) − c1·V(e; z′ = 0)]
    below rows: ΔZ_mn = −Qb · Σ_e sign_e · [fam_b(e) − c1·V(z = 0; e)]

    fam_a = (1 − C₂)·g_k(R)  / (jωε₀)      (node on the plane: image = node)
    fam_b = (1 − A_m)·g_km(R) / (jωε_m)
    c1·V  = the transmitted V the cross blocks' BT terms read, which is the
            EXACT potential of a charge at z′ = 0 (continuous across the plane)

so `fam − c1·V = −Φ_rem`, and ΔZ = +Q·Φ_rem-at-endpoints undoes Q's share
(T2 enters Z with −1/jωε; signs checked against the cross block's BT and the
family's `Z −= image` fold). The same expression is what the complete
convention adds (family direct+image get +Q·fam, cross blocks get the other
half's −Q·c1V): the conventions differ by terms that cancel exactly between
a family and its cross block.

Checks on the algebra:
- normalization: |c1·V(ε̃ = 1)| / |g/jωε₀| = 1 to 2e-16 at three (R, z, z′)
  [M, probe0, magnitude only]; the phase is carried by the next check;
- ε̃ = 1: C₂ = A_m = 0 and c1·V ≡ fam, so ΔZ ≡ 0 — the ε̃ = 1 ladder rows are
  identical to the digit with and without the term [M, probe1 on/off, probe2];
- quasi-static: (1 − C₂) = (1 − A_m)/ε̃ = 2/(ε̃ + 1), the two families' node
  potentials agree at leading order, so fam − c1V is finite at R = a [R].

Implemented as `RazorSolver._crossing_node_charges` (vectorised over ends ×
tents, V through `_crossing_fill._tables` = `radius_tables`, which matches
`six_point` to ≤1e-13 at the needed points including (a, 0, 0) [M]).

## 4. Measurements

### Reciprocity, the hypothesis test [M, probe1_on/off.log]
crossing_deck(1), probe-2 ladder (node edges FIXED), ports 4.5 / 1.0:

| medium | off x1/x2/x4 | on x1/x2/x4 |
|---|---|---|
| ε̃ = 1 | 2.77e-4 / 7.20e-5 / 1.47e-5 | identical |
| soil A | 3.18e-2 / 3.22e-2 / 3.24e-2 | 6.45e-4 / 1.60e-4 / 4.04e-5 |
| ε_r 13 | 2.63e-2 / 2.64e-2 / 2.64e-2 | 2.88e-4 / 7.35e-5 / 2.39e-5 |
| ε_r 80 | 1.06e-1 / 1.12e-1 / 1.14e-1 | 9.75e-3 / 2.45e-3 / 6.26e-4 |

Implemented through the real constructor reproduces probe 1 to the digit and
the term is called once per fill [M, probe2.log].

### The x16 floor, chased [M]
Probe 2's ladder floors at ~7.6e-6 (soil) / 7.2e-6 (ε_r 13) by x16, and x32
reads 7.600e-6 [probe6]. Ruled out:
- the families: the same deck translated wholly above / wholly below decays
  to 2e-7 / 3.6e-7 at x16 [probe3];
- quadrature: n_qp_sommerfeld 3 → 6 → 12 and crossing axis (q 12 → 20,
  panel 8 → 12) move nothing [probe4]; the Gauss–Legendre path lane floors at
  the same 7.7e-6 [probe6];
- the Sommerfeld grids: 4x finer inner lattices in both families move Z by
  ~1e-3 Ω and the floor not at all [probe5, a temporary uncommitted edit];
- the wire radius: a × 1/4, 1, 4 → floor 6.0e-6, 7.6e-6, 9.7e-6 (weak, not
  ∝ a) [probe7].

Found [probe8]: probe 2's ladder multiplies every edge EXCEPT the two
node-adjacent ones (two 50 mm segments each side at every rung), so the node
region's discretisation error is a constant of that ladder. Refining every
edge:

| medium | x1 → x16 non-reciprocity | per doubling |
|---|---|---|
| ε̃ = 1 | 2.77e-4 → 3.83e-7 | 3.9–6.9x |
| soil A | 6.45e-4 → 2.06e-6 | 4.15 / 4.24 / 4.25 / 4.19 |
| ε_r 13 | 2.88e-4 → 1.51e-6 | 4.27 / 3.63 / 3.52 / 3.50 |
| ε_r 80 | 9.75e-3 → 3.65e-5 | 4.03 / 4.05 / 4.05 / 4.04 |
| 13, σ 0.05 | 1.21e-2 → 4.03e-5 | 4.42 / 4.15 / 4.06 / 4.04 |

### Other decks [M]
- 30° bent deck (above leaning in x, below in y), soil A: 4.74e-4 → 1.14e-4
  → 2.55e-5 → 5.58e-6 (4.16 / 4.46 / 4.58) [probe12].
- three crossing tents at one node (tripod, monopole first): 4.32e-4 →
  1.08e-4 → 2.72e-5 → 6.83e-6 (3.98 / 3.98 / 3.99); with a leg first (one
  crossing tent) identical to the printed digits — the two are one system in
  two bases [probe11].
- two nodes 2 m apart (ports 4.5 / 2.5 so the deck's symmetry is not Y12 =
  Y21): 6.45e-5 → 1.59e-5 → 3.97e-6 (4.05 / 4.01). At 3 m the x4 rung, and
  at 12 m the x1 rung, is refused by bspline's below grazing floor (the
  reference solver of the probe; razor was not run past it) [probe11b].

### Convergence onto bspline, every edge refined [M, probe9/15/16]
| deck | fixed, abs(razor − bs) | unfixed gap |
|---|---|---|
| crossing_deck, feed 4.5, x1..x16 | 6.78, 3.19, 1.68, 0.95, 0.57 Ω | +7.6 … +9.9 R, flat |
| hub_deck(4), feed 4.0, x1..x8 | 7.47, 3.96, 2.23, 1.30 | +19.4 … +20.7 R, flat |
| catalog BRV (4 rad.), x1..x4 | 1.86, 0.81, 0.42 | — |
| hub_deck(N), R gap at x4, N = 1/2/4/8 | −1.13 / −0.51 / −0.47 / −0.41 | +9.9 / +17.4 / +20.5 / +21.7 |

### Power balance, antennaknobs' leg [M, probe10]
catalog buried_radial_vertical through AK's MomwireEngine, soil A:

| N | bspline R, η | fixed dR, η, d(η·R) | unfixed dR, η, d(η·R) |
|---|---|---|---|
| 1 | 170.61, 0.0723 | −0.24, 0.0722, −0.36 % | +2.99, 0.0706, −0.67 % |
| 2 | 110.30, 0.1114 | −0.31, 0.1113, −0.37 % | +3.83, 0.1069, −0.71 % |
| 4 | 78.13, 0.1580 | −0.32, 0.1580, −0.38 % | +5.33, 0.1466, −0.85 % |
| 8 | 59.39, 0.2068 | −0.32, 0.2071, −0.39 % | +7.32, 0.1827, −0.77 % |

η ≤ 1 and monotone in N; η·R within 1 % (0.4 %); R_in moved onto bspline with
η·R essentially unchanged — the removed resistance was the non-radiating one.

### Instrument only: the licensed reference [M, probe13/14]
Black box, verified against our licensed materials, equal mesh, feed 4.5:
node-fixed ladder |d| 0.48 → 0.17 → 0.082 → 0.061 → 0.055 Ω (scoping: +9.99
Ω R at x16); every edge refined 0.48 → 0.12 → 0.013 → 0.025 → 0.036 Ω. The
contrast sweep (x4, node-fixed): dR 0.00 … −0.14 Ω for ε_r 1.1 → 80 and the
lossy cases (scoping: +6.7 / +13.1 / +31.7 at 13 / 30 / 80). The ε̃ = 1 row
(+5.49 Ω) is the reference's own at that ground and is unchanged, bspline
sits at +5.78 there too.

## 5. Scope and what is not done
- Loading on a crossing deck stays the declared U3 refusal.
- Two-radius crossings stay refused by razor's scope call (`two_radius`
  not passed to `crossing_junctions`) [R].
- The fan (coincident rises) stays the bundle refusal (#846) [R].
- The first hypothesis in #1149 (family 1/ε̃ gauge on BELOW rows) was the
  right neighbourhood: it IS a node point charge evaluated in two families.
  The precise statement is that only the remainder carries it. The "next
  suspect", the #831 knot window, is a constant (C₂ / A_m) in compose mode
  and was not needed [R].
