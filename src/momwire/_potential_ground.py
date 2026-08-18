"""Mixed-potential-trunk ground composition: the `PotentialGround` object.

The potential-trunk sibling of `_field_ground.FieldGround`, and the first
unit of the razor-grounds pilot (momwire#398) that
`docs/design/solver-architecture.md` §6 proposes and §0.1 criterion 2 buys.
`FieldGround` serves `Z_mn = ⟨t_m, E[b_n]⟩`; this object serves

    Z_mn = jωμ (t_m·t_n) ⟨f_m, G f_n⟩ + (1/jωε) ⟨f′_m, G f′_n⟩,

which is `bspline.py` / `hmatrix.py` / `array_block.py` / `razor.py`. The
two never merge into one class — §2.2 of the architecture doc is explicit
about why, and the three rows of its table are visible in this file:

* the image is **mirrored geometry into the moment builder**, with the
  image sign **fused into the tangent-dot table** (`ImageGeometry` below),
  not a mirrored source handed to a field evaluator;
* the reflection-coefficient weight is **two scalars per pair**
  `(w_A, w_Φ)` — the Fresnel dyad pre-resolved into the potential
  decomposition — not a projector on `(E_ρ, E_z)`;
* `ground_phi_mode` exists at all. The field form never separates the
  charge term, so it has no analogue; here it is part of the weights'
  physics and rides inside `weight_tables` / `weight_windows`.

What IS shared with the field trunk is exactly the vocabulary the sketch
(`docs/design/field-ground-interface.md`, "what deliberately stays out")
promised would be shared: the FACTORY shape (`potential_ground_for(solver,
geom, k, omega) -> object | None`, with `None` meaning free space) and the
MODE contract (`"fold"` vs `"compose"`). Nothing else, and nothing is
imported from `_field_ground`.

Unit 1 lands the object and the **bspline dense fill's** consumption of
it: `_compute_Z_operator`'s image branch, both of its routes (the chunked
weighted-windowed accumulator and the no-accelerator moment-tensor
fallback), the enrichment fill's ε̃/sommerfeld hoists, and
`_ground_finite_Z`, whose `ground_model != "sommerfeld"` string test is
now a `mode` test. The solver's existing evaluators stay exactly where
they are and this object delegates to them, the same division
`FieldGround` draws.

Unit 2 lands the pilot's actual claim: `razor.py` — which implemented
**zero** ground methods — gains the PEC image by consuming this object and
writing no reflection of its own. It changed one thing here, and the
change is the interesting part of the unit. `ImageGeometry` had been
shaped entirely by its first consumer (mirrored segment ENDPOINTS and an
N² tangent-dot table, because that is what the B-spline moment builder
eats); razor quadratures in each segment's own arc frame and must never
materialise anything N², so it needs a mirrored (origin, tangent) pair and
an O(N) tangent table instead. Neither shape is the mirror — so the class
now exposes the two OPERATIONS (`mirror_positions` / `mirror_tangents`)
and keeps both consumers' shapes as conveniences on top. That is the
generalisation a second consumer is supposed to force, and it is the
cheapest possible one: no new concept, one layer removed.

Unit 4 lands razor's REFLECTION-COEFFICIENT ground, and it forced the
same generalisation one row down the table — on `weight_windows` this
time, and for the same reason. That producer had been shaped by its
first consumer too: "observer rows" meant rows of the geometry's own
square segment × segment table, because the B-spline fill's observers
ARE its segments. Razor's are not. Its A-term observers are the
testing-path quadrature points (each with the path's own tangent) and
its Φ-term observers are the segment centroids its charge term
differences between — two different observer sets, neither of them the
source set, and its `geom` dict does not even carry the keys the old
spelling read. So `weight_windows` now takes an observer set and a
source set as `(centres, tangents)` pairs, with the square
geometry-shaped case as the DEFAULT rather than the definition
(`own_segments`). Same shape of result as unit 2's: one operation
exposed, the existing consumer's call unchanged and bit-identical, no
new concept.

Unit 5 lands razor's SOMMERFELD ground — `mode == "compose"`, the third
row of the §2.2 table and the first COMPOSING ground on this solver —
and it forced the same generalisation a third time, on `Remainder`. That
class had one method, `evaluate(supp_seg, polys)`, which takes a B-spline
basis description and returns a finished `(n_basis, n_basis)` GALERKIN
block; razor's rows are razor-blade path integrals, so the block is the
wrong OPERATION for it rather than the wrong shape of the right one. Two
independent consumers said so before a line was written — unit 4's report
and the `PulseSolver` probe (momwire#416,
`docs/design/pulse-probe.md` §4.1), which was blocked by the identical
signature — and both named the same fix, which is the one taken:
`field_windows(observers, sources)` hands back the remainder FIELD of each
source segment's basis arc moments, projected on the observer tangents, in
observer-row windows, and `evaluate` stays as the Galerkin convenience over
it. `_sommerfeld.remainder_field_proj` was already that primitive one level
down, so once again the generalisation exposes a layer that exists rather
than inventing one.

**What stays out**, deliberately, and named here so its later migration is
a decision rather than a discovery:

* **the schedule** — bands, chunk sizes, budgets, the `Z -=` seams. In
  particular `BSplineSolver._image_weight_row_bytes` still reads
  `ground_eps` / `ground_model`: it prices what a chunk's weight producer
  allocates, which is budget arithmetic, and the field trunk's precedent
  keeps budgets out of the object.
* **the hmatrix / array_block block paths** (`_zblock_image`,
  `_zblock_image_refl`, `_refl_weight_tables`,
  `_zblock_sommerfeld_remainder`). They inherit `BSplineSolver` and keep
  calling the methods this object delegates to, so they are unaffected;
  migrating them is a later unit.
* **`_image_weight_enrich_blocks`** — the enrichment reaction's third
  weight shape (row / col / ee rectangular sub-blocks). Its ε̃ and its
  "is this sommerfeld" question now come from this object; its per-mode
  algebra does not yet.
* **the batched swept fill** (`_swept_batched_z_chunks`). It is PEC-only
  by `_swept_batched_available`, and its weight is a FOURTH spelling —
  the C++ batched assembler forms the tangent dot in-kernel from two
  `(N, 3)` tangent tables (momwire#333), so there is no `(w_A, w_Φ)` pair
  to hand it at all.

Evaluation-order discipline is part of this module's interface for the
same reason it is part of `_field_ground`'s (momwire#392, architecture doc
§3.4): every expression below is the call site's own, moved and not
rewritten, which is what makes the migration bit-identical rather than
merely equivalent.
"""

from __future__ import annotations

import numpy as np

from . import _ground_refl, _sommerfeld
from ._quadrature import leggauss

# M = diag(1, 1, −1) as the row it multiplies a `(..., 3)` array by. One
# array, so `ImageGeometry.mirror_tangents` and `weight_windows`' PEC /
# Sommerfeld closures below are visibly the same reflection.
_MIRROR = np.array([1.0, 1.0, -1.0])

# Default per-source-segment Gauss order for `Remainder.field_windows`. The
# remainder field is smooth on the scale of a segment (it is the ground
# field with its singular C₂-image part already removed), so a low order
# converges: this is `BSplineSolver.n_qp_sommerfeld`'s own default, and a
# consumer that carries its own knob passes `n_qp=` instead of taking it.
_N_QP_REMAINDER = 3


class ImageGeometry:
    """The mirror map, as OPERATIONS first and tables second.

    The reflection through `z = ground_z` is M = diag(1, 1, −1) applied to
    positions about the plane and to direction vectors about the origin,
    and those two are the whole of it:

    `mirror_positions(p)` — any `(..., 3)` array of points, reflected.
    `mirror_tangents(t)` — any `(..., 3)` array of directions, z-flipped.

    Everything else on this class is a CONVENIENCE built out of those two,
    in the shape one consumer happens to want. Unit 1 shipped only the
    conveniences and unit 2 (razor's PEC ground) is what forced the split:
    the B-spline fill wants mirrored segment endpoints and an N² table
    because its moment builder quadratures from endpoints and its kernel
    contracts tangents pairwise; the razor fill wants a mirrored segment
    ORIGIN + TANGENT pair (it quadratures in each segment's own arc frame)
    and a `(3, n_basis)` tangent table, and it must never materialise
    anything N². Two solvers, two shapes, one mirror — so the mirror is
    what the object exposes, and neither consumer is the other's shape.

    The conveniences, both B-spline-shaped and both lazy (a razor `geom`
    dict does not even carry the keys they read, and never touches them):

    `seg_l` / `seg_r` — the segment ENDPOINTS mirrored, which is what the
    B-spline moment builder consumes.

    `tangent_dot()` — the `(N_obs, N_src)` table `t_m · M·t_n`. This is the
    potential trunk's IMAGE SIGN, fused into a table rather than carried as
    a separate factor: the horizontal image current's anti-parallel
    direction and the image charge's sign flip are both absorbed by it plus
    the seams' single global minus (architecture doc §2.2, "image sign
    fused into the `td_all` table"). It is O(N²) and built only when asked.
    Razor gets the same sign from `mirror_tangents` on its own
    `(3, n_basis)` table plus the same single minus, which is the same
    physics at O(N) storage.

    `BSplineSolver._image_positions` / `_image_tangent_dot` still spell the
    same two operations, because hmatrix's block paths call them on their
    own sub-blocks and are not migrated yet; `tests/test_potential_ground.py`
    pins the two spellings bit-equal so they cannot drift apart before that
    migration lands.
    """

    __slots__ = ("_geom", "_ground_z", "_seg_l", "_seg_r")

    def __init__(self, ground_z, geom):
        self._ground_z = ground_z
        self._geom = geom
        self._seg_l = None
        self._seg_r = None

    # --- the operations ------------------------------------------------

    def mirror_positions(self, positions):
        """Reflect `(..., 3)` POINTS through the plane `z = ground_z`."""
        out = positions.copy()
        out[..., 2] = 2 * self._ground_z - out[..., 2]
        return out

    def mirror_tangents(self, tangents):
        """Reflect `(..., 3)` DIRECTIONS: `t → (t_x, t_y, −t_z)`.

        The plane's offset does not enter — a direction is a difference of
        positions, and `2·ground_z` cancels — so this is a pure z-flip and
        `ground_z` is deliberately unread here.
        """
        return tangents * _MIRROR

    # --- the B-spline-shaped conveniences ------------------------------

    @property
    def seg_l(self):
        """Mirrored segment start points, built on first read."""
        if self._seg_l is None:
            self._seg_l = self.mirror_positions(self._geom["seg_l"])
        return self._seg_l

    @property
    def seg_r(self):
        """Mirrored segment end points, built on first read."""
        if self._seg_r is None:
            self._seg_r = self.mirror_positions(self._geom["seg_r"])
        return self._seg_r

    def tangent_dot(self):
        """`t_m · M·t_n`, complex-free (float64) — the PEC mirror table."""
        tangents = self._geom["tangents"]
        return tangents @ self.mirror_tangents(tangents).T


class Remainder:
    """The Sommerfeld remainder operator: the smooth field `Q` that the
    C₂-scaled exact image is composed WITH — as an OPERATION first and a
    Galerkin block second.

    `field_windows(observers, sources)` — the remainder FIELD of each
    source segment's basis ARC MOMENTS, already projected on the observer
    tangents, produced in observer-row windows.

    `evaluate(supp_seg, polys)` — the finished `(n_basis, n_basis)`
    Galerkin block, i.e. that field with the B-spline fill's own testing
    rule already applied. A CONVENIENCE, in the shape of the first
    consumer, exactly as `ImageGeometry`'s `seg_l` / `tangent_dot` (unit 2)
    and `weight_windows`' square `own_segments` default (unit 4) are.

    Unit 5 is what forced the split, and TWO independent consumers named
    the same reason (momwire#398 unit 4's report §6.4; the `PulseSolver`
    probe, momwire#416, `docs/design/pulse-probe.md` §4.1):

    * **razor** tests on PATH INTEGRALS between element centroids, not on
      Galerkin projections. A finished Galerkin block is the wrong
      OPERATION for it at any argument shape — it would have to undo one
      testing rule to apply another. What it needs is the field of each
      source segment's tent moments (∫Λ and ∫τΛ, `n_moment = 2`) at its
      testing-path quadrature points, which is `field_windows` with the
      path points as observers;
    * **pulse** point-matches, so it needs the plain rectangular
      `(n_obs, n_src)` field at one centroid per row — the same call with
      `n_moment = 1`, whose trailing axis it drops.

    And `_sommerfeld.remainder_field_proj(obs, t_obs, src, t_src, gz, k,
    grid)` was ALREADY that primitive one level down: this generalisation
    exposes the layer that exists rather than inventing one, which is the
    same "one layer removed, no new concept" move units 2 and 4 made.

    **Why `evaluate` is not spelled as a composition of `field_windows`,
    when unit 2's conveniences were.** `_Z_sommerfeld_remainder`'s hot path
    is a single fused C++ kernel (`sommerfeld_remainder_bspline_Q`) that
    interpolates, projects, moment-quadratures on BOTH axes and assembles
    into `Q` without ever forming the intermediate this operation returns —
    the same reason `weight_tables` answers `None` for PEC where
    `weight_windows` spells the tables out: there are two kernels, not one
    kernel behind two shapes. So `evaluate` keeps delegating, is
    bit-identical across this generalisation, and
    `tests/test_potential_ground.py` pins the two spellings against each
    other at the quadrature's own agreement instead of at `array_equal`.

    Deliberately NOT the field trunk's prepare/replay pair, and unit 5 did
    not change that either. There, the remainder is evaluated at whatever
    observer set a caller reaches, so the observer-independent state has to
    be hoisted out of the schedule by hand (momwire#357 item 1). Here
    `field_windows` IS that hoist — the grid, the source nodes and the
    moment weights are built once when the producer is made and every
    window replays them — so the split the field trunk spells as two
    methods is spelled here as a closure, which is the same shape
    `weight_windows` already uses on this trunk.

    **The k this object carries is the GROUND's, not the solver's.** Both
    are the same number for `BSplineSolver` (its factory call passes
    `self.k`, and its swept path moves `self.k` itself), but razor sweeps
    by passing k to the factory while `self.k` stays at the constructor's
    value — so a grid built off `solver.k` would silently freeze at one
    wavenumber down a sweep. The Sommerfeld grid is k-dependent through
    and through (its lattice is in wavelengths), which is why unit 5's
    schedule rule is that nothing here may cross a prepare/replay
    boundary.
    """

    __slots__ = ("_eps_tilde", "_geom", "_k", "_omega", "_solver")

    def __init__(self, solver, geom, eps_tilde, k, omega):
        self._solver = solver
        self._geom = geom
        self._eps_tilde = eps_tilde
        self._k = k
        self._omega = omega

    # --- the operation --------------------------------------------------

    def source_segments(self):
        """This geometry's own `(seg_l, seg_r, tangents)` — the default
        `field_windows` takes on its source axis.

        The B-spline-shaped reading of "where the sources are", and lazy
        for the same reason `PotentialGround.own_segments` is: a razor
        `geom` dict carries none of the keys it reads, and razor names its
        own source set instead. Endpoints rather than centres, because the
        source axis is INTEGRATED over here, not sampled at a point.
        """
        geom = self._geom
        return geom["seg_l"], geom["seg_r"], geom["tangents"]

    def field_windows(
        self, observers, sources=None, *, n_moment=1, n_qp=_N_QP_REMAINDER
    ):
        """A producer `moments_fn(i0, i1) -> (i1-i0, n_src, n_moment)` of
        observer-row windows of the projected remainder field's source arc
        moments.

        Entry `[m, n, p]` is

            ∫_{segment n} τ^p · t̂_m · F(r_m, r'(τ)) · t̂_n dτ

        with τ the source's local arc length from its segment start, F the
        smooth remainder field of a unit current MOMENT (theory manual
        eqs 143-147, the ground field minus its C₂-scaled exact image) and
        t̂_m the observer's own tangent. Sign convention unchanged and
        shared with `evaluate`: the caller ADDS this term's contribution to
        the C₂ image and takes one global minus (`C2·img + Q`, then
        `Z -=`), which is the composition `mode == "compose"` names.

        `observers` is a `(points, tangents)` pair of `(n_obs, 3)` arrays —
        the same spelling `weight_windows` takes, and for the same reason:
        an observer is a point plus the direction the row projects on, and
        no consumer's observers have to be its segments. `sources` is a
        `(seg_l, seg_r, tangents)` triple and **defaults to
        `source_segments()`**, the geometry-shaped case.

        `n_moment` is how many arc moments the caller's basis needs on the
        source segment, and it is the whole of what separates the two
        known consumers:

        * `n_moment = 2` — a TENT (razor): `[..., 0]` is ∫Λ-shaped and
          `[..., 1]` the first moment, from which `M1/h` (rising wing) and
          `M0 − M1/h` (falling wing) are the same two combinations razor's
          reduced-kernel moments already take;
        * `n_moment = 1` — a PULSE, or any point-matched consumer: the
          trailing axis is length 1 and `[..., 0]` is the plain rectangular
          `(n_obs, n_src)` field the momwire#416 probe asked for;
        * `n_moment = degree + 1` — the B-spline moments `evaluate`'s own
          fused kernel forms internally.

        Everything observer-independent is hoisted into the producer and
        every window replays it: the interpolation grid, the source
        quadrature nodes and their moment weights. The grid's extent is
        sized from the SOURCE segment endpoints exactly as
        `_Z_sommerfeld_remainder` sizes it (`max_image_distance`, issue
        #331); observers lying on those segments are covered by convexity,
        and anything further out is clamped by the grid's own `r1_max`
        (issue #157) rather than silently extrapolated.
        """
        if sources is None:
            sources = self.source_segments()
        seg_l, seg_r, src_t = sources
        obs_p, obs_t = observers
        solver = self._solver
        gz = solver.ground_z
        k = self._k
        cancel_flag = solver._cancel_flag

        # k-dependent, so it is built HERE — inside the per-solve, per-ω
        # object — and can never be cached across a wavenumber.
        r1_max = _sommerfeld.max_image_distance(seg_l, seg_r, gz)
        grid = _sommerfeld.get_grid(
            self._eps_tilde,
            k,
            r1_max,
            omega=self._omega,
            mu=solver.mu,
            cancel_flag=cancel_flag,
        )

        n_src = seg_l.shape[0]
        xg, wg = leggauss(n_qp)
        tq = 0.5 * (xg + 1.0)
        span = seg_r - seg_l
        nodes = seg_l[:, None, :] + tq[None, :, None] * span[:, None, :]
        h = np.linalg.norm(span, axis=1)
        u_phys = h[:, None] * tq[None, :]  # (n_src, n_qp) physical arc offsets
        w_node = 0.5 * h[:, None] * wg[None, :]
        # Moment weights W[p, j, q] = w_q · u_q^p, the quadrature dual of
        # the source basis's own u^p moments — the same convention
        # `_Z_sommerfeld_remainder` builds for the B-spline polynomials.
        W = w_node[None] * u_phys[None] ** np.arange(n_moment)[:, None, None]
        src = nodes.reshape(n_src * n_qp, 3)
        t_src = np.repeat(src_t, n_qp, axis=0)

        def field_moments(i0, i1):
            proj = _sommerfeld.remainder_field_proj(
                obs_p[i0:i1],
                obs_t[i0:i1],
                src,
                t_src,
                gz,
                k,
                grid,
                cancel_flag=cancel_flag,
            )
            return np.einsum("ojq,pjq->ojp", proj.reshape(i1 - i0, n_src, n_qp), W)

        return field_moments

    # --- the B-spline-shaped convenience --------------------------------

    def evaluate(self, supp_seg, polys):
        """`Q[m, n]` over the whole basis, as `_Z_sommerfeld_remainder`
        returns it — the Galerkin block, i.e. `field_windows`' operation
        with the B-spline fill's own testing rule already applied.

        Sign convention unchanged: the caller ADDS this to the C₂ image and
        takes one global minus (`C2·img + Q`, then `Z -=`), which is the
        composition `mode == "compose"` names.
        """
        return self._solver._Z_sommerfeld_remainder(
            self._geom, supp_seg, polys, self._eps_tilde
        )


class PotentialGround:
    """One solve's ground, in the mixed-potential trunk: which weights,
    which coefficient, which composition.

    Frozen per `(geom, k, ω)` and built by `potential_ground_for`, which is
    the ONE place the `ground_z` / `ground_eps` / `ground_model` /
    `ground_phi_mode` strings are read on this trunk's dense fill. A
    consumer branches on `mode` and applies the members below.

    Attributes:

    `mode` — `"fold"` (PEC, refl-coef, and the radial-wire screen when it
    lands): the image block may go through the seams' single `Z -= …`
    entry by entry, in any order. `"compose"` (sommerfeld):
    `C₂·img + Q` must be associated BEFORE that minus, because
    `free − (c2·img + Q) ≠ (free − c2·img) − Q` in float64. Same contract,
    same words, as `FieldGround.mode` — the one piece of vocabulary the two
    trunks share besides the factory. (The `+ Q` where the field trunk
    writes `− rem` is this trunk's own sign convention, not a difference in
    the contract: `_Z_sommerfeld_remainder` returns `Q` with the EFIE
    contribution `−Q` already folded into the seams' minus.)

    `image_coefficient` — 1.0, or C₂ = (ε̃−1)/(ε̃+1) for sommerfeld.
    **Spelled differently from the field trunk, on purpose.** There, C₂ is
    applied by the consumer to a finished image block, and the operand
    order (`np.multiply(coef, block, out=block)` — coefficient on the LEFT)
    is a measured one-ULP contract. Here it never reaches a block: it
    enters through the constant `(w_A, w_Φ)` tables, which is how
    `_ground_finite_Z` has always spelled it, so the multiply happens once
    per pair inside the assembly kernel instead of once per matrix entry
    afterwards. The attribute is exposed anyway because it is the same
    physical quantity and the same shared vocabulary — a consumer that
    wants to know "how strong is this image" asks it here — but a
    potential-trunk consumer must NOT apply it itself: `weight_tables` and
    `weight_windows` have already folded it in, and applying it twice is
    the obvious bug this sentence exists to prevent.

    `eps_tilde` — the complex relative permittivity ε̃(ω) the weights are
    taken at; `None` over a PEC ground (and the tell for "PEC" throughout
    this class). Hoisted into the factory so a chunked fill pays
    `_ground_refl.eps_tilde` once per fill rather than once per chunk —
    same value either way, it is a pure function of arguments the fill
    holds fixed.

    `phi_mode` — `ground_phi_mode` for a refl-coef ground, `None`
    otherwise. The potential trunk's own knob, with NO field-trunk
    analogue (architecture doc §2.2's tell): the field form never
    separates the charge term, so it cannot express a modelling choice
    about the image charge. It is not a scheduling flag and consumers do
    not read it — it is carried here because it is part of what
    `weight_tables` means, and named on the object so that "which Φ
    weighting" is visibly a ground decision.

    `standard_fresnel` — True when this ground's per-pair weights are the
    stock Fresnel ones. **On the field trunk this gates a backend
    dispatch; here, on the dense fill, it does not, and the difference is
    a finding rather than an oversight.** The bspline dense fill's fused
    kernels (`assemble_Z_bspline_weighted` and its windowed twin) take
    `w_A` / `w_Phi` as complex128 DATA and never compute a reflection
    coefficient, so they already serve ANY coefficient-level ground — the
    radial-wire screen included — with no kernel work and no key. What
    keeps the attribute from being purely vestigial is that this trunk
    DOES have an in-kernel spelling, one layer out from unit 1's scope:
    `bspline_assemble_offedge_block_refl` (and its EK twin) computes ρ_v /
    ρ_h from ε̃ inside the loop and takes `ground_phi_mode` as the two
    coefficients `_ground_refl.phi_mode_coeffs` reduces it to. That kernel
    is hmatrix's / array_block's block path, which this unit does not
    migrate; when it is migrated, `standard_fresnel` is the key its
    dispatch must consult, exactly as `_field_tensor_image_refl`'s does.
    Until then no unit-1 consumer reads it, and the ground-row test is
    what keeps it honest.
    """

    __slots__ = (
        "_geom",
        "_k",
        "_omega",
        "_solver",
        "eps_tilde",
        "image_coefficient",
        "mode",
        "phi_mode",
        "standard_fresnel",
    )

    def __init__(
        self,
        solver,
        geom,
        k,
        omega,
        *,
        mode,
        eps_tilde,
        image_coefficient,
        phi_mode=None,
        standard_fresnel=True,
    ):
        self._solver = solver
        self._geom = geom
        self._k = k
        self._omega = omega
        self.mode = mode
        self.eps_tilde = eps_tilde
        self.image_coefficient = image_coefficient
        self.phi_mode = phi_mode
        self.standard_fresnel = standard_fresnel

    # ------------------------------------------------------------------
    # the mirror map
    # ------------------------------------------------------------------

    def image_geometry(self) -> ImageGeometry:
        """The mirror map: two operations, plus lazy conveniences on top.

        Built fresh on every call and cached nowhere — the object itself is
        two attributes, and everything it can produce is either O(N) or
        materialised only when asked for. EK policy is NOT here:
        `_ek_spec(geom, mirror=True)` stays with the moment builder, the
        same line `FieldGround.image_sources` draws (the ground supplies
        mirrored geometry, never the extended kernel's opinion about it).
        """
        return ImageGeometry(self._solver.ground_z, self._geom)

    # ------------------------------------------------------------------
    # the weight tables — this trunk's (w_A, w_Φ) row
    # ------------------------------------------------------------------

    def weight_tables(self):
        """The whole-geometry `(w_A, w_Φ)` pair, or `None` for PEC.

        `None` means "assemble the image UNWEIGHTED, with the image
        tangent-dot table" — the free/PEC kernel `_assemble_Z(...,
        td_all=…)`, whose Φ term carries no weight at all and whose A term
        takes a real table. That is a genuinely different kernel, not a
        `w_A = td_img, w_Φ = 1` special case of the weighted one, which is
        why `None` is the honest answer rather than a pair of synthesised
        tables.

        Otherwise:

        * refl-coef — the Fresnel pair, through the solver's cached
          k-independent specular prep (`_image_refl_prep`) and its per-ω
          weights (`_image_refl_weights`). The cache is the solver's
          SCHEDULE and stays there, exactly as `_image_refl_prep` is the
          Galerkin field fill's schedule in the sibling module.
        * sommerfeld — the constant C₂ tables that carry the exact-image
          part. This is the `image_coefficient` spelling difference the
          class docstring describes: C₂ multiplies the mirror table here,
          not the assembled block later.

        Beware the asymmetry with `weight_windows` below: the same PEC
        ground answers `None` here and a real `(td_img, ones)` pair there.
        It is not an inconsistency in the ground, it is the two kernels:
        the windowed assembler speaks only weighted, so the chunked route
        has to spell PEC's "no weight" as tables, while the tensor route
        has a kernel that can say it structurally.
        """
        if self.eps_tilde is None:
            return None
        if self.mode == "compose":
            # NEC's decomposition (theory manual eqs 136-147): the exact
            # image scaled by the constant C₂, which absorbs all the
            # singular behaviour, plus the smooth remainder block. ε̃ → ∞:
            # C₂ → 1 and Q → 0, PEC image exactly; ε̃ → 1: both vanish.
            c2 = self.image_coefficient
            td_img = self.image_geometry().tangent_dot()
            w_A = c2 * td_img.astype(np.complex128)
            return w_A, np.full_like(w_A, c2)
        solver = self._solver
        return solver._image_refl_weights(
            solver._image_refl_prep(self._geom), self._omega
        )

    def own_segments(self):
        """This geometry's own `(centres, tangents)` — the default both
        axes of `weight_windows` take.

        The B-spline-shaped reading of "where the weights live", and lazy
        for the same reason `ImageGeometry`'s conveniences are: a razor
        `geom` dict carries none of the keys it reads, and razor never
        calls it (it names its own observer and source sets instead).
        """
        geom = self._geom
        return 0.5 * (geom["seg_l"] + geom["seg_r"]), geom["tangents"]

    def weight_windows(self, observers=None, sources=None):
        """A producer `weights_fn(i0, i1) -> (w_A, w_Φ)` of observer-row
        WINDOWS, each complex128 `(i1-i0, n_src)`.

        The chunked accumulator's contract (momwire#323): the windows are
        produced per chunk and nothing N²-scale in the weights is ever
        allocated, which is what retired the 2× dense-Z residency the
        grounded fill used to carry. Every mode's algebra is row-local, so
        each producer restricts on the observer axis directly rather than
        slicing a global table — the refl-coef one bypasses the
        `_image_refl_prep` cache entirely for that reason
        (`specular_pair_tables` already takes a rectangular observer ×
        source block); the cache stays for `weight_tables` and the
        enrichment path, which do want the whole thing.

        `observers` and `sources` are each a `(centres, tangents)` pair of
        `(n, 3)` arrays. **`sources` defaults to `own_segments()` and
        `observers` defaults to `sources`** — the square, geometry-shaped
        case the B-spline fill takes, in which a window is a band of the
        square table. That default is a CONVENIENCE over the operation,
        exactly the split `ImageGeometry` draws between `mirror_positions`
        / `mirror_tangents` and its `seg_l` / `seg_r` / `tangent_dot`
        conveniences — and unit 4 (razor's refl-coef ground) is what forced
        it, the same way unit 2 forced that one. Razor's A-term observers
        are its testing-path QUADRATURE POINTS, each with the path's own
        tangent there, and its Φ-term observers are the segment centroids
        its charge term differences between; neither is "the geometry's
        segments", and razor's `geom` does not even carry the keys the
        default reads. So the object exposes the operation — Fresnel
        weights between an observer set and a source set — and keeps the
        first consumer's shape as the default, bit for bit (the two
        spellings are pinned equal in `tests/test_potential_ground.py`).

        A weight pair is per (observer, SOURCE SEGMENT): the Fresnel
        coefficients are evaluated once per pair at the specular angle of
        the ray from the source segment's image midpoint to the observer,
        and held constant over the source segment. Both axes' tangents
        enter only through the A-term dyad (`w_Φ` never reads them), which
        is why a consumer that wants the Φ weighting alone may hand the
        same set in twice.

        ε̃ and C₂ are frequency- but not row-dependent and are the
        factory's, so the closures capture them and no chunk recomputes
        them.

        Unlike `weight_tables`, this NEVER returns `None`: see that
        method's last paragraph for why PEC has to spell itself out here.
        """
        if sources is None:
            sources = self.own_segments()
        if observers is None:
            observers = sources
        obs_c, obs_t = observers
        src_c, src_t = sources
        mirror = _MIRROR

        if self.eps_tilde is None:

            def pec_weights(i0, i1):
                # float64 gemm → C-contiguous; astype gives the complex128
                # the assembler wants without a wrapper copy on top.
                w_A = (obs_t[i0:i1] @ (src_t * mirror).T).astype(np.complex128)
                return w_A, np.ones_like(w_A)

            return pec_weights

        eps_t = self.eps_tilde

        if self.mode == "compose":
            c2 = self.image_coefficient

            def sommerfeld_weights(i0, i1):
                # complex scalar × float64 array → complex128 C-contiguous.
                # The smooth remainder is a separate, already-chunked term.
                w_A = c2 * (obs_t[i0:i1] @ (src_t * mirror).T)
                return w_A, np.full(w_A.shape, c2)

            return sommerfeld_weights

        phi_mode = self.phi_mode
        ground_z = self._solver.ground_z

        def refl_weights(i0, i1):
            cos_th, td_img, P = _ground_refl.specular_pair_tables(
                obs_c[i0:i1],
                obs_t[i0:i1],
                ground_z,
                src_centers=src_c,
                src_tangents=src_t,
            )
            rho_v, rho_h = _ground_refl.fresnel_rho(eps_t, cos_th)
            w_A = _ground_refl.a_term_weights(rho_v, rho_h, td_img, P)
            w_Phi = _ground_refl.phi_term_weights(phi_mode, eps_t, rho_v)
            if np.ndim(w_Phi) == 0:
                # "image"/"normal" are pair-independent; "rho_v"/"blend"
                # already come back as per-pair windows.
                w_Phi = np.full(w_A.shape, complex(w_Phi))
            return w_A, w_Phi

        return refl_weights

    # ------------------------------------------------------------------
    # the remainder
    # ------------------------------------------------------------------

    def remainder(self) -> Remainder | None:
        """The remainder operator, or `None` for a ground that has none
        (PEC and refl-coef).

        Present exactly when `mode == "compose"` — the remainder IS what
        the composition is about, so the two are the same fact asked two
        ways, and a consumer may branch on either.
        """
        if self.mode != "compose":
            return None
        return Remainder(self._solver, self._geom, self.eps_tilde, self._k, self._omega)


def potential_ground_for(solver, geom, k, omega) -> PotentialGround | None:
    """The factory: `solver`'s ground as one object, or `None` for free
    space.

    `None` is not a null object and must not become one — the free-space
    fill takes the path it always took, with the ground branch
    structurally absent rather than skipped, so not one float operation
    differs. Same standard as `extended_kernel=False`, and the same
    standard `FieldGround` holds itself to.

    The four grounds momwire ships, and the fifth this interface is
    designed against (architecture doc §6.1: the radial-wire screen must
    land by changing coefficient functions in `_ground_refl`, not
    assembly code in any solver):

    | ground        | mode    | weight_tables | coef  | remainder |
    |---------------|---------|---------------|-------|-----------|
    | free          | *(None — no object)*                        |
    | PEC           | fold    | None          | 1     | None      |
    | refl-coef     | fold    | Fresnel w_A/w_Φ | 1   | None      |
    | sommerfeld    | compose | constant C₂   | C₂(ε̃) | evaluate  |
    | radial screen | fold    | screen-modified ρ_v/ρ_h through the same
                                `a_term_weights` / `phi_term_weights`
                                pair; `standard_fresnel=False`          |

    The screen row is the acceptance test, not a promise about today: it
    lands as a modified reflection-coefficient function one level down and
    a `standard_fresnel=False` row here, with zero edits to `bspline.py`'s
    fill — the dense assembly kernels take the weights as data (see
    `standard_fresnel`), so on THIS trunk's dense route even the fused
    path serves it unchanged.
    """
    if solver.ground_z is None:
        return None
    if solver.ground_eps is None:
        return PotentialGround(
            solver,
            geom,
            k,
            omega,
            mode="fold",
            eps_tilde=None,
            image_coefficient=1.0,
        )
    eps_t = _ground_refl.eps_tilde(solver.ground_eps, omega, solver.eps)
    if solver.ground_model == "sommerfeld":
        return PotentialGround(
            solver,
            geom,
            k,
            omega,
            mode="compose",
            eps_tilde=eps_t,
            image_coefficient=(eps_t - 1.0) / (eps_t + 1.0),
        )
    return PotentialGround(
        solver,
        geom,
        k,
        omega,
        mode="fold",
        eps_tilde=eps_t,
        image_coefficient=1.0,
        phi_mode=solver.ground_phi_mode,
    )
