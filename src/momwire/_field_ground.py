"""Field-trunk ground composition: the shared Fresnel pair-weight builder.

Unit 1 of the `FieldGround` migration sketched in
`docs/design/field-ground-interface.md` (momwire#397), which serves
criterion 1 of `docs/design/solver-architecture.md` §0.1 — removing
`SinusoidalGalerkinSolver`'s second complete implementation of the ground
modes. This unit's scope is the sketch's **per-pair weight** row and
nothing else: the `cos θ → ρ_v/ρ_h → NEC field dyad` chain that
`SinusoidalSolver._field_tensor_image_refl` and
`SinusoidalGalerkinSolver._refl_projection` each used to spell out. The
`FieldGround` class itself, the mirror map, the image coefficient and the
Sommerfeld remainder are units 2 and 3 and are deliberately absent here.

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

from . import _ground_refl


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
