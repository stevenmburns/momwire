"""Field-trunk ground composition: the `FieldGround` object and its parts.

The migration sketched in `docs/design/field-ground-interface.md`
(momwire#397), which serves criterion 1 of
`docs/design/solver-architecture.md` §0.1 — removing
`SinusoidalGalerkinSolver`'s second complete implementation of the ground
modes. Landed in units:

* **unit 1** — the sketch's **per-pair weight** row: the
  `cos θ → ρ_v/ρ_h → NEC field dyad` chain that
  `SinusoidalSolver._field_tensor_image_refl` and
  `SinusoidalGalerkinSolver._refl_projection` each used to spell out.
  `specular_pair_prep`, `PairTables`, `PairWeights` below.
* **unit 2** — `FieldGround` itself, `field_ground_for`, `Remainder`, and
  the point-matched consumption: `SinusoidalSolver._assemble_Z` reads ONE
  object where it used to read `ground_z` / `ground_eps` / `ground_model`.
* **unit 3** — the Galerkin consumption (`_fold_ground_block`), still to
  come; nothing here is shaped for it beyond what the sketch promises.

The physics is one level down, in `_ground_refl` (ε̃, the Fresnel
coefficients, the specular ray tables) — that layer is where a
coefficient-level ground such as the radial-wire screen will land, which
is why this module sits *on top of* it rather than absorbing it.

What the two spellings shared, and what they did not:

* **shared** — the k-independent per-pair specular geometry (cos θ, p̂, the
  two tangent·p̂ projections) and the per-ω dyad projection

      t_m · D · E = ρ_v·(t_m·E) − (ρ_v + ρ_h)·(t_m·p̂)·(E·p̂),

  applied to the unprojected Eqs 76-79 component tables. Both are here,
  once: `specular_pair_prep` and `PairWeights.project`.
* **not shared, and not moved** — the *schedule*. The point-matched solver
  builds prep per observer band and throws it away (momwire#332 unit B,
  momwire#357 item 2); the Galerkin solver builds it once per geometry and
  caches it, then replays it at whatever (test, source) pairing each block
  reaches. Both schedules are expressed through the same two objects: the
  band-vs-whole choice is `obs_rows`, the pairing choice is `project`'s
  index arguments.

Pairing-agnostic by contract: `project` takes broadcastable `(m_idx,
n_idx)` index arrays exactly as `SinusoidalGalerkinSolver._tested_contribs`
documents for its projector protocol — `(M, 1)` against `(1, N)` for a
tensor build, `(P, 1)` against `(P, 1)` for the near-pair path's
one-to-one pairing — and `None` for "the tables are already at the
caller's pairing", which is the point-matched band build and gathers
nothing. `PairWeights.project` therefore *is* a projector, so the Galerkin
consumer is `...weights(eps_t).project` and nothing else.

Evaluation-order discipline is part of this module's interface, not a
convention it happens to follow (momwire#392, and §3.4 of the architecture
doc): numpy elides a temporary that is the RIGHT operand of an array
product once it passes 256 KB, and the complex128 in-place loop does not
round like the out-of-place one. Every complex product below therefore has
a NAMED right operand, and nothing is reassociated — these are the two
call sites' own expressions, moved but not rewritten, which is what makes
the migration bit-identical rather than merely equivalent.
"""

from __future__ import annotations

import numpy as np

from . import _ground_refl, _sommerfeld


class PairWeights:
    """Per-(test, source)-pair Fresnel dyad weights at one ε̃.

    `rho_v` and `rho_vh` = ρ_v + ρ_h are the two complex tables the dyad
    projection needs (ρ_h never appears alone); `tm_p`, `tn_p`, `px`, `py`
    are the k-independent geometry `specular_pair_prep` built, carried
    along so a consumer holds one object rather than seven arrays.

    All six are at the prep's own shape — a whole (N_obs, N_src) table for
    the Galerkin consumer, one observer band for the point-matched one —
    and `project` gathers from them at whatever pairing it is given.
    """

    __slots__ = ("px", "py", "rho_v", "rho_vh", "tm_p", "tn_p")

    def __init__(self, rho_v, rho_vh, tm_p, tn_p, px, py):
        self.rho_v = rho_v
        self.rho_vh = rho_vh
        self.tm_p = tm_p
        self.tn_p = tn_p
        self.px = px
        self.py = py

    def project(self, cm, m_idx=None, n_idx=None):
        """Apply NEC's Fresnel field dyad to the IMAGE source's field before
        the tangential projection, returning the three tangential tables
        `(Phi_const, Phi_sin, Phi_cos)` at `cm`'s own shape.

        D = ρ_v·(I − p̂p̂) − ρ_h·p̂p̂ with p̂ the horizontal unit normal to the
        plane of incidence of the specular ray (image midpoint → observer
        midpoint), and ρ_v/ρ_h the Fresnel coefficients at that ray's
        incidence angle — per-pair constants, which is NEC's IPERF=0
        approximation. Expanding the projection (t_m has unit norm,
        p̂·p̂ = 1):

            t_m · D · E = ρ_v·(t_m·E) − (ρ_v + ρ_h)·(t_m·p̂)·(E·p̂)

        with both scalars available from the unprojected component tables:
            t_m·E = td·E_z + rho_proj_factor·E_ρ    (the plain projection)
            E·p̂  = (t_n·p̂)·E_z + (ρ̂·p̂)·E_ρ         (t_img·p̂ = t_n·p̂;
                                                     ρ̂ = rho_vec/rho_eval,
                                                     the E_ρ rule's own
                                                     radius-regularized
                                                     denominator, so the
                                                     two E_ρ pickups stay
                                                     mutually consistent)

        PEC limit ε̃ → ∞: ρ_v → +1, ρ_h → −1, so the p̂ correction vanishes
        and this reduces exactly to the plain tangential projection — the
        ε̃ = 1e16 collapse tests ride on that. Near-vertical rays have
        rho_vec → 0, which kills the ρ̂·p̂ term along with the p̂ ambiguity.

        `m_idx` / `n_idx` are broadcastable index arrays naming the (test
        segment, source segment) each entry of `cm` belongs to — the
        projector protocol `_tested_contribs` documents. Pass `None` (both)
        when the tables are ALREADY at the caller's pairing, as the
        point-matched band build's are: that path gathers nothing, which is
        what keeps its per-band residency where momwire#332 unit B put it.
        """
        if m_idx is None:
            rho_v, rho_vh = self.rho_v, self.rho_vh
            tm_p, tn_p = self.tm_p, self.tn_p
            px, py = self.px, self.py
        else:
            # Gathered BEFORE any arithmetic, and bound to names: gathering
            # the sum ρ_v + ρ_h is entry-for-entry the sum of the gathers,
            # and a named operand cannot be elided (momwire#392).
            rho_v = self.rho_v[m_idx, n_idx]
            rho_vh = self.rho_vh[m_idx, n_idx]
            tm_p = self.tm_p[m_idx, n_idx]
            tn_p = self.tn_p[m_idx, n_idx]
            px = self.px[m_idx, n_idx]
            py = self.py[m_idx, n_idx]

        # ρ̂·p̂ from the image-build rho_vec (p̂ is horizontal, so only the
        # x/y components contribute).
        rho_p = (cm["rho_vec"][..., 0] * px + cm["rho_vec"][..., 1] * py) / cm[
            "rho_eval"
        ]
        td = cm["td"]
        rp = cm["rho_proj_factor"]

        def _weighted(Ez, Erho):
            tm_E = td * Ez + rp * Erho  # t_m · E
            E_p = tn_p * Ez + rho_p * Erho  # E · p̂
            return rho_v * tm_E - rho_vh * tm_p * E_p

        return (
            _weighted(cm["Ez_const"], cm["Erho_const"]),
            _weighted(cm["Ez_sin"], cm["Erho_sin"]),
            _weighted(cm["Ez_cos"], cm["Erho_cos"]),
        )


class PairTables:
    """The k-INDEPENDENT half: per-pair specular geometry for one observer
    window against the full source width.

    `cos_th`, `px`, `py` are `_ground_refl.specular_ray_tables`' outputs and
    `tm_p` / `tn_p` the two tangent·p̂ projections, each
    `(N_obs_window, N_src)`. The fused C++ Fresnel kernels take all five
    positionally, which is why they stay plain attributes rather than
    hiding behind accessors.

    Frequency enters only through `weights`, so a swept caller that holds a
    `PairTables` pays the O(N²) geometry once and the ρ tables per ω — the
    prep/replay split the point-matched band path depends on (its prep is
    per band and per fill; nothing here caches on its behalf).
    """

    __slots__ = ("cos_th", "px", "py", "tm_p", "tn_p")

    def __init__(self, cos_th, px, py, tm_p, tn_p):
        self.cos_th = cos_th
        self.px = px
        self.py = py
        self.tm_p = tm_p
        self.tn_p = tn_p

    def weights(self, eps_t) -> PairWeights:
        """Per-ω `PairWeights` at complex relative permittivity `eps_t`.

        ρ_h is summed into ρ_v + ρ_h here and then dropped: it has no other
        consumer, so neither call site has to keep it alive.
        """
        rho_v, rho_h = _ground_refl.fresnel_rho(eps_t, self.cos_th)
        rho_vh = rho_v + rho_h  # → 0 in the PEC limit
        return PairWeights(rho_v, rho_vh, self.tm_p, self.tn_p, self.px, self.py)


def specular_pair_prep(seg_centers, seg_tangents, ground_z, obs_rows=None):
    """Build the k-independent `PairTables` for segments mirrored across
    z = `ground_z`.

    `obs_rows = (i0, i1)` restricts the OBSERVER axis, the same contract as
    `SinusoidalSolver._field_tensor`'s; `None` means the whole geometry.
    Sources are always the full, REAL (unmirrored) centres/tangents —
    `specular_ray_tables` mirrors internally (dz = z_m + z_n − 2·ground_z),
    and p̂ has no z-component while the image tangent flips only z, so
    t_img·p̂ = t_src·p̂ and the real source tangents serve the image-source
    projection unchanged.

    The banded and whole-geometry builds are ONE spelling — the banded one,
    which accumulates `a·px + b·py` through a single shared scratch buffer
    rather than through four temporaries. That is the same association, so
    the same floats, at two `(window, N_src)` temporaries instead of four
    (momwire#357 item 2, which also bought `specular_ray_tables`' own
    elementwise tiling). With `obs_rows=None` the observer arrays ARE the
    source arrays, so the square case reaches `specular_ray_tables` with
    `c is cs` exactly as the separate square spelling did.
    """
    obs_c = seg_centers if obs_rows is None else seg_centers[slice(*obs_rows)]
    obs_t = seg_tangents if obs_rows is None else seg_tangents[slice(*obs_rows)]
    cos_th, px, py = _ground_refl.specular_ray_tables(
        obs_c, ground_z, src_centers=seg_centers
    )
    tm_p = obs_t[:, 0][:, None] * px
    scratch = obs_t[:, 1][:, None] * py
    tm_p += scratch
    tn_p = seg_tangents[:, 0][None, :] * px
    np.multiply(seg_tangents[:, 1][None, :], py, out=scratch)
    tn_p += scratch
    return PairTables(cos_th, px, py, tm_p, tn_p)


class Remainder:
    """The Sommerfeld remainder operator, as the prepare/replay pair the
    sketch names: observer-independent state built once, evaluated at
    whatever observer set a caller reaches.

    A thin handle over `SinusoidalSolver._sommerfeld_remainder_prepare`'s
    dict (momwire#357 item 1) — the *decision* to build it is the ground's,
    the *schedule* it is replayed on stays the caller's, which is why
    `replay` forwards `consume` and `row_group` untouched. Holding the
    prepared state is the whole of this object's residency: O(N) tables
    plus the borrowed module-level grid, alive exactly as long as the
    `FieldGround` that owns it, which is one fill.
    """

    __slots__ = ("_prepared", "_solver")

    def __init__(self, solver, prepared):
        self._solver = solver
        self._prepared = prepared

    def replay(self, obs_centers=None, obs_tangents=None, consume=None, row_group=1):
        """Evaluate the remainder at one observer set, returning the
        `(3, M, N)` tensor — or nothing, when `consume` streams it.

        Arguments and their contracts are
        `_replay_sommerfeld_remainder`'s verbatim: `None` observers mean
        the geometry's own segment centres/tangents (collocation),
        `consume(i0, i1, block)` takes each observer chunk as it exists,
        and `row_group` rounds the chunk boundaries down to a multiple of
        itself so a caller whose reduction groups observers never meets a
        partial group. The point-matched band passes its slice of the
        centres and takes the tensor back; the Galerkin fold will pass its
        test-quadrature points, a consumer, and `row_group = nq`.
        """
        return self._solver._replay_sommerfeld_remainder(
            self._prepared,
            obs_centers=obs_centers,
            obs_tangents=obs_tangents,
            consume=consume,
            row_group=row_group,
        )


class FieldGround:
    """One solve's ground, in the field trunk: which weight, which
    coefficient, which composition.

    Frozen per `(geom, k, ω)` and built by `field_ground_for`, which is the
    ONE place the `ground_z` / `ground_eps` / `ground_model` strings are
    read on this trunk. What a consumer branches on instead is `mode`, and
    what it applies is the three members below. The *schedule* — bands,
    chunk sizes, `subtract_into`, the order the fold reaches entries in —
    is deliberately not here and does not move (sketch, "what stays out").

    Attributes:

    `mode` — `"fold"` (PEC, refl-coef, and the screen when it lands): the
    image block may go through the solver's single global minus entry by
    entry, in any order. `"compose"` (sommerfeld): `coef·img − remainder`
    must be associated BEFORE that minus, because
    `free − (c2·img − rem) ≠ (free − c2·img) + rem` in float64. That
    inequality is the whole reason this field exists; a consumer that
    branches on it cannot get the association wrong for a ground it has
    never heard of.

    `image_coefficient` — 1.0, or C₂ = (ε̃−1)/(ε̃+1) for sommerfeld.
    **Contract, not convention:** consumers MUST apply it as
    `np.multiply(coef, block, out=block)` — coefficient on the LEFT.
    complex128 multiply evaluates the imaginary part as
    `x.re*y.im + x.im*y.re`, so swapping the operands reorders that sum and
    moves the last bit (measured: 1e-17 relative on the assembled Z, i.e.
    one ULP, on every Sommerfeld config). `block *= coef` is the swapped
    order and is wrong here. Same #392-class discipline as this module's
    named right operands: evaluation order is part of the interface's
    value, not an implementation detail behind it.

    `standard_fresnel` — True when this ground's per-pair dyad is the stock
    Fresnel one, so the fused C++ kernels (`sinusoidal_field_tensor_refl`
    and its EK twin), which compute ρ_v/ρ_h in-kernel from ε̃ and cos θ, are
    eligible. That in-kernel spelling is a THIRD copy of the dyad the
    sketch's table did not anticipate, so the amendment stands: the fused
    kernels are an optimization keyed to standard-Fresnel grounds, and a
    coefficient-modified ground — the radial-wire screen, whose ρ_v/ρ_h are
    the earth's in parallel with the screen's surface impedance — sets this
    False and is served by `PairWeights.project` instead, with zero solver
    edits. True for all four grounds momwire ships today.

    `eps_tilde` — the complex relative permittivity ε̃(ω) the weights and
    the fused kernels are taken at; `None` over a PEC ground. Hoisted here
    so a banded fill pays `_ground_refl.eps_tilde` once per fill rather
    than once per band (same value either way — it is a pure function of
    arguments the fill holds fixed).
    """

    __slots__ = (
        "_geom",
        "_k",
        "_omega",
        "_remainders",
        "_solver",
        "_weighted",
        "eps_tilde",
        "image_coefficient",
        "mode",
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
        weighted,
        eps_tilde,
        image_coefficient,
        standard_fresnel=True,
    ):
        self._solver = solver
        self._geom = geom
        self._k = k
        self._omega = omega
        self._weighted = weighted
        self._remainders = {}
        self.mode = mode
        self.eps_tilde = eps_tilde
        self.image_coefficient = image_coefficient
        self.standard_fresnel = standard_fresnel

    def image_sources(self) -> tuple:
        """The mirror map: `(src_c, src_t)` for the image sources, source
        centres mirrored across the plane and source tangents z-flipped.

        Delegates to the solver's `_image_source_centers_tangents`, which
        both sinusoidal solvers already shared — the ground supplies the
        mirrored GEOMETRY and never the extended kernel's policy about it:
        EK eligibility is still scored against these sources by the
        solver's own `mirror=True` plumbing.
        """
        return self._solver._image_source_centers_tangents(self._geom)

    def pair_weights(self, obs_rows=None) -> PairTables | None:
        """The k-independent per-pair specular tables for one observer
        window, or `None` when the image takes the plain tangential
        projection.

        `None` is PEC (no dyad at all) and the sommerfeld IMAGE part (whose
        weighting is the scalar `image_coefficient`, not a dyad) — the
        sketch's five-ground table, rows 2 and 4. Otherwise the
        `PairTables` whose `weights(eps_tilde).project` is the dyad.

        `obs_rows = (i0, i1)` restricts the OBSERVER axis, `_field_tensor`'s
        contract; `None` is the whole geometry. Built FRESH on every call
        and cached nowhere: the point-matched fill asks per band and lets
        each band's tables die with it (momwire#332 unit B, and the memory
        gate `test_sinusoidal_refl_coef_specular_tables_are_not_cached`
        that pins it). A consumer that wants the whole-geometry table
        cached per geometry — the Galerkin one — owns that choice as part
        of its own schedule, exactly as `_image_refl_prep` does today.
        """
        if not self._weighted:
            return None
        return self._solver._image_refl_band(self._geom, obs_rows)

    def image_field(self, obs_rows=None) -> tuple:
        """The image field tensor `(Φ_const, Φ_sin, Φ_cos)` for one
        observer band, already carrying whichever per-pair weight this
        ground says — but NOT `image_coefficient`, which the caller applies
        under the contract above, and not the remainder.

        This is the decision the band loop used to make from strings: a
        dyad-weighted ground goes through the Fresnel build, everything
        else through the plain mirrored-source one. Both are the solver's
        own evaluators, called here rather than chosen there.
        """
        if self._weighted:
            return self._solver._field_tensor_image_refl(
                self._geom, self._k, obs_rows=obs_rows, ground=self
            )
        return self._solver._field_tensor_image(self._geom, self._k, obs_rows=obs_rows)

    def remainder(self, cos_shape="cos") -> Remainder | None:
        """The remainder operator, or `None` for a ground that has none
        (PEC and refl-coef — the sketch's rows 2, 3 and 5).

        Present exactly when `mode == "compose"`: the remainder IS what the
        composition is about. Preparing it here is what a `"compose"`
        consumer does once per fill, in the same scope that fixes k, and
        the state dies with this object.

        `cos_shape` selects the third source shape, `_field_tensor_
        sommerfeld_remainder`'s contract: `"cos"` for the point-matched
        fill, `"cos-1"` for the Galerkin one, which subtracts this block
        from a free-space triple built on the folded shape. One prepared
        state is kept per shape, so asking twice does not rebuild.
        """
        if self.eps_tilde is None or self.mode != "compose":
            return None
        got = self._remainders.get(cos_shape)
        if got is None:
            # The grid-sizing endpoint scan is band- and k-invariant and at
            # O(N²) per call was the dominant cost of the banded fill
            # (momwire#367); hoisted with the prepare it feeds, which is
            # where the fill used to hoist it by hand.
            geom = self._geom
            r1_max = _sommerfeld.max_image_distance(
                geom["seg_l"], geom["seg_r"], self._solver.ground_z
            )
            got = Remainder(
                self._solver,
                self._solver._sommerfeld_remainder_prepare(
                    self._geom,
                    self._k,
                    self.eps_tilde,
                    cos_shape=cos_shape,
                    r1_max=r1_max,
                ),
            )
            self._remainders[cos_shape] = got
        return got


def field_ground_for(solver, geom, k, omega) -> FieldGround | None:
    """The factory: `solver`'s ground as one object, or `None` for free
    space.

    `None` is not a null object and must not become one — the free-space
    fill takes the path it always took, with the ground branch structurally
    absent rather than skipped, so not one float operation differs. Same
    standard as `extended_kernel=False`.

    The four grounds, per the sketch's table (the fifth, the radial-wire
    screen, lands here as a `weighted=True, standard_fresnel=False` row and
    a modified coefficient function one level down in `_ground_refl` —
    zero edits to either sinusoidal file, which is what criterion 1
    accepts on):

    | ground     | mode    | pair_weights | coef  | remainder |
    |------------|---------|--------------|-------|-----------|
    | free       | *(None — no object)*                        |
    | PEC        | fold    | None         | 1     | None      |
    | refl-coef  | fold    | Fresnel dyad | 1     | None      |
    | sommerfeld | compose | None         | C₂(ε̃) | prepare/replay |
    """
    if solver.ground_z is None:
        return None
    if solver.ground_eps is None:
        return FieldGround(
            solver,
            geom,
            k,
            omega,
            mode="fold",
            weighted=False,
            eps_tilde=None,
            image_coefficient=1.0,
        )
    eps_t = _ground_refl.eps_tilde(solver.ground_eps, omega, solver.eps)
    if solver._sommerfeld_ground():
        # NEC's decomposition (theory manual eqs 136-147): the exact image
        # scaled by C₂, which absorbs all the singular behaviour, plus the
        # smooth interpolated remainder. ε̃ → ∞: C₂ → 1, remainder → 0, PEC
        # image exactly; ε̃ → 1: both vanish, free space.
        return FieldGround(
            solver,
            geom,
            k,
            omega,
            mode="compose",
            weighted=False,
            eps_tilde=eps_t,
            image_coefficient=(eps_t - 1.0) / (eps_t + 1.0),
        )
    return FieldGround(
        solver,
        geom,
        k,
        omega,
        mode="fold",
        weighted=True,
        eps_tilde=eps_t,
        image_coefficient=1.0,
    )
