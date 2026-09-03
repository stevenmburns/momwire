"""The COMPLETE crossing fill — the interface-crossing junction's blocks
(momwire#524 phase 2).

A deck with a crossing junction (a wholly-below wire whose end stands in
the ground plane, junction-joined there to an above wire) is filled with
the one quadrature-convergent node treatment the phase-2 probes measured:
the complete field-form-equivalent mixed-potential spelling, all by-parts
ends and corners, on all four kernel families, one convention —

  cross:  t_ab = M + SW + SQ + BT + CORNER   (designed kernels, graded
          axes, thin-wire radius folded into every pair distance)
  self:   + β_dir·(bnd + corner)(G_dir) − β_img·(bnd + corner)(G_img)
          per medium family, on the same graded axes

Every "drop" spelling is truncation-regularized — at resolved quadrature
the retained ∬f′f′V's ln(a)-class content diverges with its balancing
end/corner terms deleted — so completeness here is a convergence
requirement, not a preference. Continuity of current through the node and
the AGARD slope condition then EMERGE from the fill's own physics: the
node needs no constraint row and no merged dof (split ≡ merged ≡
V-constrained, measured to the digit on the adjudication decks).

The corner term's sign is STRUCTURAL and ORIENTATION-CARRIED:
−σ_test·σ_src·c1·V(a) against the value-1 tents (σ = −1 at a wire's
start, +1 at its end), calibrated once on the soil-A adjudication deck
and never re-picked per medium (the high-σ ladder pinned its
medium-independence). With the fan widening the below axis carries N
node tents: the corner loop emits one σ-carried term per
(above-tent × below-tent) pair, and the self completion's corner emits
the below×below tent pairs at R = a — all at the ONE crossing node the
scope allows, which is what keeps the single V(a) evaluation honest.

Scope guards this module inherits from the derivation:

  * segments on both axes must be purely horizontal or vertical — the W
    by-parts move uses t̂⊥·∇⊥ = d/dl, exact only there;
  * one wire radius across the deck — the radius rule ρ_eff = √(ρ² + a²)
    is the corner's regularization and a per-pair radius has no pinned
    convention yet.

`bspline.BSplineSolver._compute_Z_operator_buried` is the only caller today;
it hands the fill a `CrossingContext` (below) rather than itself (momwire#801).
"""

from __future__ import annotations

import os
import sys
import warnings
from typing import NamedTuple

import numpy as np
from numpy.polynomial.legendre import leggauss

from . import _aca, _ground_refl, _near_interface, _sommerfeld_below
from ._sommerfeld_transmitted import _c1_moment


class CoarseCrossingNode(UserWarning):
    """A crossing junction's node region is unresolved for #674's class."""


class NodeArm(NamedTuple):
    """One member of a crossing junction, as the advisory reads it.

    `h_resolved` is what the bar gates: the FINEST segment length within
    `NODE_REACH` of the shared point. `h_adjacent` is the segment that
    actually touches the node, carried alongside because it is what a
    reader looks at first and the two can differ a lot (an ungraded feed
    gap in front of a graded chain).
    """

    h_resolved: float
    h_adjacent: float
    wire: int
    end: str
    side: str


# ---------------------------------------------------------------------------
# What the fill takes from a solver, as data (momwire#801).
#
# The fill was written against `BSplineSolver` by address — eight attribute
# reads and five geometry keys — and the G1-3 probe (momwire#651) found that
# only ONE of them is bspline-shaped: `axis_data`'s reading of the basis as
# per-segment polynomials. Everything after the axis dict consumes the dict
# and physical scalars. So the solver is replaced by a `CrossingContext`, and
# any formulation whose basis is piecewise-polynomial on the segments (a
# degree-1 tent is `c0 + c1·u` on each of its two support segments) can
# hand the fill what it needs without the fill knowing the formulation.
#
# What the context does NOT abstract: the fill tests with the basis it
# expands with (axis A's `F`/`Fd` are the test rows), and its by-parts end
# terms and corner are derived for value-1 tents standing at wire ends. A
# formulation that tests differently needs its own test-side axis dict --
# and razor's is `path_test_axis` below (momwire#813), which spells the
# razor-blade path integral in `axis_data`'s language so every block
# function here serves it unchanged. Its rows also want the corner term OFF
# (`corner=False`): the corner is a Galerkin by-parts term and a path-tested
# row has no by-parts to do.


class BasisPolynomials(NamedTuple):
    """The basis as per-segment polynomials in each segment's local arc
    coordinate `u ∈ [0, h]` measured from `seg_l`.

    `supp_seg[m, a]` is the global segment index of basis `m`'s `a`-th
    support segment (rows zero-padded; a slot is live only if its polynomial
    is nonzero — the padding trap `axis_data` guards); `polys[m, a, p]` is
    the coefficient of `u**p`, `p ≤ degree`.
    """

    supp_seg: np.ndarray
    polys: np.ndarray
    degree: int


class AxisGeometry(NamedTuple):
    """The five geometry columns the fill reads, in global segment order."""

    seg_l: np.ndarray  # (n_seg, 3) segment start points
    seg_r: np.ndarray  # (n_seg, 3) segment end points
    h: np.ndarray  # (n_seg,) segment lengths
    tangents: np.ndarray  # (n_seg, 3) unit tangents seg_l → seg_r
    seg_offsets: np.ndarray  # (n_wires + 1,) first global segment per wire


class Medium(NamedTuple):
    """`(eps_t, eps_m, k_p, k_m, c2, a_m)` for a buried solve — what
    `BSplineSolver._buried_medium` returned, as a record with names."""

    eps_t: complex  # the ground's RELATIVE ε̃(ω)
    eps_m: complex  # ε₀·ε̃, what the lower medium's Φ term divides by
    k_p: float  # free-space wavenumber
    k_m: complex  # in-medium wavenumber, Im ≤ 0
    c2: complex  # the ±=+ family's exact-image coefficient
    a_m: complex  # the ±=− family's, `image_coefficient_below`


def buried_medium(ground_eps, omega, eps, k):
    """The `Medium` for a buried solve, from the ground spec and the
    wavenumber alone. `c2` and `a_m` are measured, not derived, and the
    negative of each other — exactly the kind of coincidence a sign scan
    exists to keep honest (`_sommerfeld_below.image_coefficient_below`)."""
    eps_t = _ground_refl.eps_tilde(ground_eps, omega, eps)
    k_p = float(k)
    return Medium(
        eps_t,
        eps * eps_t,
        k_p,
        _sommerfeld_below.k_medium(eps_t, k_p),
        (eps_t - 1.0) / (eps_t + 1.0),
        _sommerfeld_below.image_coefficient_below(eps_t),
    )


class CrossingContext(NamedTuple):
    """Everything the crossing fill reads, and nothing else.

    Built by the caller from its own state — `BSplineSolver._crossing_context`
    and, since momwire#813, `RazorSolver._crossing_context`, which spells its
    tents as `BasisPolynomials` the same way. `a_wire` is the deck's ONE wire
    radius — the scope guard in the module docstring — and `ground_z` the
    interface height.
    """

    basis: BasisPolynomials
    geom: AxisGeometry
    medium: Medium
    ground_z: float
    a_wire: float
    omega: float
    mu: float
    eps: float


# WHAT IS GATED, and why it is not the node-adjacent segment.
#
# The obvious quantity — the length of the segment touching the node —
# does not predict the error, and the two decks this was calibrated on
# say so directly:
#
#   #674 base fan     above arm edges [666.7] mm            7.48 ohm of
#                     never resolves near the node          soil-A mesh move
#
#   antennaknobs      above arm edges [50, 6.25, 18.75,     ~0.04 ohm; an 8x
#   buried-radial     75, 300, 1200, 487] mm — a feed       feed-gap sweep
#   (graded, fixed)   gap in front of a graded chain,       moves the soil-A
#                     resolved to 6.25 mm within 56 mm      answer 0.115 ohm
#
# Node-adjacent h reads those as 667 vs 50 mm — 13x — while their errors
# differ by ~200x. A first-order class cannot do that, so node-adjacent h
# is not the variable. What separates them is whether the arm RESOLVES to
# the recipe scale anywhere near the node, which is exactly what #674's
# grading recipe does and what its two ladders measure:
#
#     uniform node    order 1.0    75.0 mm  0.2269 ohm   (eps~=1,
#                                  37.5 mm  0.1069        |fan - truth|)
#                                  25.0 mm  0.0666
#                                  18.8 mm  0.0475
#     graded node     order ~2.6   25.00 mm 0.0036 ohm
#                                   6.25 mm 0.0001
#                                   1.56 mm 0.0000
#
# At the same 25 mm those regimes differ 18x in error. The regime is the
# dominant variable; the millimetres are secondary. So the gated quantity
# is the finest h within reach of the node, and on that quantity the two
# decks read 667 mm vs 6.25 mm — 107x apart, with the bar comfortably
# inside instead of threaded through a needle.
#
# On real soil the lossy transmitted kernels amplify the class ~30x: the
# 4-rise fan moved 7.48 ohm base -> graded, and antennaknobs' catalog deck
# sat 29 ohm of reactance off the converged answer for months while global
# density sweeps to 4x moved it < 0.2 ohm. The bar matters most exactly
# where it is hardest to see.
#
# 25 mm is #674's own coarsest GRADED rung (0.0036 ohm at eps~=1, ~0.1 ohm
# scaled to soil) and 4x the 6.25 mm converged recipe rung. Above it a
# node region is unresolved for this class; below it the class is at or
# under the measurement floor.
NODE_H_BAR = 0.025

# How far from the shared point counts as "near the node". The recipe's
# own reach: #674's grading spans geometric panels from ~6 mm out to the
# design segment length, ~150 mm on the decks measured here.
#
# The classification is INSENSITIVE to this over a wide range — anything
# from ~60 mm to ~1 m puts both calibration decks on the same sides (the
# base fan's single 667 mm edge overlaps any window, and the antennaknobs
# arm reaches 6.25 mm by 56 mm from the node). That is the point of
# gating the resolved scale rather than the touching segment: the earlier
# spelling of this advisory needed its threshold inside a 50-75 mm window
# and flipped on a 20% move.
NODE_REACH = 0.15

# ABSOLUTE metres, both constants, deliberately: this class is the
# interface corner's own resolution, not a wavelength fraction. Provenance
# is two deck families at the 40 m band with one wire radius each, so a
# deck far from that scale may read these as over- or under-eager. Known
# limitation of the measurements behind them, not of the rule.

# WITHDRAWN (#760). This used to say "the above arm's interface-adjacent mesh
# is the dominant term", on #674 probe2's rise-only 0.2214 ohm against
# mono-only 0.0171. Re-derived across the quadrature axis that study held
# fixed at n_qp_pair=4, the asymmetry is quadrature, not mesh:
#
#   n_qp_pair      mono-only      rise-only
#       4            0.0171         0.2214     <- what the claim was built on
#       8            0.0007         0.0540
#      16            0.0003         0.0082
#      32            0.0002         0.0001     <- no asymmetry left
#
# At converged quadrature both arms sit at ~1e-4 ohm, so there is no dominant
# member to name. The message no longer attributes one.

# What the node mesh is actually worth, per quadrature order, on the soil-A
# fan: |base - fully graded|, against the far-mesh doubling on the same rung.
#
#   n_qp_pair    node grading    far-mesh x2
#       4          10.4101         0.0215
#       8           4.4967         0.0968     <- today's default
#      16           1.5628         0.1182
#      32           0.3025         0.1227
#      64           0.0782         0.1224
#
# Two things follow, and the message says both. Grading the node IS worth
# ~4.5 ohm at the shipped default, so the advice stands. But it is worth that
# because coarse quadrature and a coarse node interact — the mesh term itself
# is 0.08 ohm — so raising n_qp_pair is the other lever, and past q~16 it is
# the bigger one. That lever did not exist when this warning was written: the
# accelerated kernels refused n_qp > 8 until #762 tiled them.
_RAISE_THE_ORDER = (
    "grading is not the only lever, and past n_qp_pair ~ 16 it is not the "
    "bigger one: most of what grading buys at the default order is quadrature "
    "error, not mesh error (#760 — on the soil-A fan, grading moves the answer "
    "4.5 ohm at n_qp_pair=8 but only 0.08 ohm at 64, while doubling the FAR "
    "mesh moves it 0.12 ohm at any order). Raising n_qp_pair runs on the "
    "accelerated path since #762"
)


# Point the warning at the CALLER'S deck, not at a line inside momwire —
# an advisory that reports its own package's internals reads as an
# internal bug. `skip_file_prefixes` (3.12+) walks out of momwire from
# whatever entry point got here (compute_impedance, impedance_sweep,
# far_field), which a fixed `stacklevel` cannot: those paths sit at
# different depths. The 3.10/3.11 fallback is the depth of the
# compute_impedance chain (warn -> _compute_Z_operator_buried ->
# _compute_Z_operator -> compute_impedance -> caller), correct for the
# common path and merely imprecise elsewhere — the message names the
# wire and the mesh either way.
_WARN_TARGET = (
    {"skip_file_prefixes": (os.path.dirname(os.path.abspath(__file__)),)}
    if sys.version_info >= (3, 12)
    else {"stacklevel": 5}
)


def warn_coarse_node(arms):
    """Warn when a crossing junction's node region is unresolved.

    `arms` is an iterable of `NodeArm`. Returns the worst arm by
    `h_resolved` (or None for a deck with no crossing junction), so a
    caller — a test, a diagnostic — can read the number back whether or
    not it crossed the bar.

    ADVISORY ONLY. It never remeshes and never refuses: the deck is the
    deck (banked prints, adjudication culture), and a coarse node is a
    legitimate thing to ASK for — every rung of a convergence ladder but
    the last is one. The knowledge of where and how much is ours; the
    decision is the caller's.
    """
    worst = None
    for a in arms:
        if worst is None or a.h_resolved > worst.h_resolved:
            worst = a
    if worst is None or worst.h_resolved <= NODE_H_BAR:
        return worst
    gap = ""
    if worst.h_adjacent > worst.h_resolved * 1.5:
        gap = (
            f" (its node-adjacent segment is {worst.h_adjacent * 1000:.1f} mm, "
            f"but nothing within {NODE_REACH * 1000:.0f} mm of the node is "
            f"finer than the figure above)"
        )
    warnings.warn(
        f"crossing node: wire {worst.wire}'s {worst.end} is the {worst.side} "
        f"member, and the finest mesh within {NODE_REACH * 1000:.0f} mm of "
        f"the node is {worst.h_resolved * 1000:.1f} mm, above the "
        f"{NODE_H_BAR * 1000:.0f} mm bar{gap}. At the default quadrature "
        f"order this node is worth ~4.5 ohm on the soil-A fan and a global "
        f"density sweep will report it as converged while it is not "
        f"(momwire#674, re-derived in momwire#760). Grade the node: "
        f"geometric panels toward "
        f"the shared point, ~6 mm at the node growing to the design's own "
        f"segment length away from it, spelled as extra VERTICES in the "
        f"wire's polyline with per-edge counts in n_per_edge_per_wire so "
        f"the grading cannot change junction topology. {_RAISE_THE_ORDER}. "
        f"Advisory: nothing is remeshed. See stevenmburns/momwire#696.",
        CoarseCrossingNode,
        **_WARN_TARGET,
    )
    return worst


_TILT_TOL = 1e-12

# How far off `ground_z` a point may sit and still BE the interface.
#
# One number, used by two things that must agree: `axis_data`'s decision that
# a segment touches the plane (and therefore gets graded panels), and
# `_on_plane_side`'s decision that a wrong-side coordinate is the node rather
# than a geometry error. They were separate literals until momwire#852, where
# the second one did not exist at all.
_PLANE_TOL = 1e-12
_CROSS_RTOL = 1e-9
_CORNER_RTOL = 1e-10
_GX8, _GW8 = leggauss(8)

# The #688 admissibility split: far (admissible) segment blocks are evaluated
# on COARSER axes and through the low-rank ACA pass; near / corner-adjacent
# blocks keep the designed direct evaluation unconditionally. The coarse
# knobs are the banked density-ladder combo (far Gauss 6->4, panels G8->G4,
# growth x2->x4: <= 3e-4 ohm movement on the adjudication decks against
# gate envelopes 100x wider). Since #692's DEEPER-deck ladder (0.5 m and
# 1.0 m rungs, base and node-graded meshes, worst soil movement 7e-4 ohm
# with the graded eps1 collapse margins unmoved at the 1e-4 class) the
# NEAR axes carry the same density by default — the _NEAR_* knobs below,
# kept separate from _FAR_* so either side reverts alone. Segments meeting
# at the crossing node have box distance 0 and are inadmissible by
# construction, so every corner-adjacent pair stays dense-direct — and the
# corner V(a) itself routes through six_point, touching none of this.
_ADM_ETA = 1.0
_ACA_TOL = 1e-7
# leaf=3 measured as the knee (2026-08-27 ladder): leaf=4 leaves an extra
# dense ring (~7-10% more evaluations); leaf<=2 pushes coarse axes onto
# close non-touching pairs — at leaf=1 that costs three orders of block
# parity (1e-9 -> 1e-6 relative), and at leaf=2 it moved the g1
# adjudication anchor's fourth printed decimal (138.7670 vs the banked
# 138.7671). leaf=3 keeps every banked print to the digit.
_CLUSTER_LEAF_SEGS = 3
_FAR_Q = 4
_FAR_GROWTH = 4.0
# The near/fine axes' density (#692): same values as _FAR_* today, banked
# by the deeper-deck ladder (scratch/692-study in antennaknobs + the #692
# comment). Reverting the near side alone = q 4->6 here and G4->G8/x4->x2
# below; `_n_qp_buried_field`'s q=6 measurement stays authoritative for
# the buried GRID fills, which never routed through these axes.
_NEAR_Q = 4
_NEAR_GROWTH = 4.0
_NEAR_GX, _NEAR_GW = leggauss(4)
# ACA only where sampling undercuts the full coarse product: rank-r
# sampling costs ~(rows+cols)·(m+n) designed points across the four
# kernels (measured ranks 6-8), so a block must have m·n well past
# that to win. Below the guard, direct coarse evaluation is cheaper
# AND memo-dedups against its mirror blocks.
_ACA_COST_GUARD = 48.0
_GX4, _GW4 = leggauss(4)
_CROSS_KEYS = ("U", "V", "W", "dzW")

# The whole-run dense-direct switch (a timing comparison, a bisect); parity
# tests drive both paths in-process by calling the two entries directly.
_FORCE_DENSE = bool(os.environ.get("MOMWIRE_CROSSING_FORCE_DENSE"))


def _graded_u(h, toward_end, a, growth=2.0, gx=_GX8, gw=_GW8):
    """Quadrature (u, w) on [0, h], log-graded (a-scale, `growth`-factored)
    toward u = h (`toward_end='hi'`) or u = 0 (`'lo'`); Gauss-`len(gx)` per
    panel. The grading is what lets the ln(a)-class end integrals actually
    converge instead of being truncated by segment-scale Gauss."""
    edges = [0.0]
    step = a
    while edges[-1] + step < h:
        edges.append(edges[-1] + step)
        step *= growth
    edges.append(h)
    e = np.array(edges)
    if toward_end == "hi":
        e = h - e[::-1]
    mid = 0.5 * (e[:-1] + e[1:])
    half = 0.5 * (e[1:] - e[:-1])
    u = (mid[:, None] + half[:, None] * gx[None, :]).ravel()
    w = (half[:, None] * gw[None, :]).ravel()
    return u, w


def axis_data(ctx, seg_idx, coarse=False, *, growth=None, panel_order=None, q=None):
    """Everything one axis of the crossing blocks needs: quadrature nodes,
    per-node tangents and weights, per-basis value/derivative samples, and
    the signed wire-end table for the by-parts terms.

    Segments touching the interface get log-graded panels toward it;
    every other segment gets plain Gauss at the crossing fill's own
    density (`_NEAR_Q` since #692 — the buried GRID fills keep
    `_n_qp_buried_field`, which no longer routes through here). The ends
    table keeps only ends where some basis has nonzero value (value-1
    junction/contact tents); free ends carry no basis and drop out.

    `coarse=True` builds the far-block variant of the same axis (the #688
    density knobs — numerically the same densities as near since #692,
    kept a separate variant so either side's knobs revert alone); it
    exists only for admissible blocks and must never be fed to the
    ends/corner terms.

    **The three density knobs are overridable per axis** (momwire#813):
    `growth` and `panel_order` are the graded panels' ratio and Gauss order
    on interface-touching segments, `q` the plain Gauss order everywhere
    else. `None` means the module constant this axis would have used, so a
    caller that passes nothing gets exactly the axis it got before these
    arguments existed — which is what keeps `BSplineSolver`'s crossing fill
    bit-identical.

    They exist because a PATH-tested row that ends AT the node needs a
    finer axis than a Galerkin one does, and the two error plateaus are
    separate: on razor's node row at `crossing_deck(1)`, `panel_order`
    alone takes the residual 5.3e-5 → 2.2e-6 and no further at any
    `growth`, and `q` alone leaves 5.3e-5 untouched — it is
    `panel_order` = 8 AND `q` = 8 together that reach 7.7e-11. Sweeping
    either alone reads as "converged" at the other's plateau, which is how
    that residual came to be recorded as a property of the source Gauss.
    """
    d = ctx.basis.degree
    geom = ctx.geom
    if q is None:
        q = _FAR_Q if coarse else _NEAR_Q
    xg, wg = leggauss(q)
    if growth is None:
        growth = _FAR_GROWTH if coarse else _NEAR_GROWTH
    if panel_order is None:
        gx, gw = (_GX4, _GW4) if coarse else (_NEAR_GX, _NEAR_GW)
    else:
        gx, gw = leggauss(panel_order)
    tq = 0.5 * (xg + 1.0)
    gz = float(ctx.ground_z)
    a_wire = float(ctx.a_wire)
    tol = _PLANE_TOL

    supp_seg, polys = ctx.basis.supp_seg, ctx.basis.polys
    n_basis = polys.shape[0]

    nodes_l, t_l, w_l, u_l, segpos = [], [], [], [], []
    for g in seg_idx:
        sl, sr = geom.seg_l[g], geom.seg_r[g]
        h = geom.h[g]
        tang = geom.tangents[g]
        if abs(tang[2]) > _TILT_TOL and np.hypot(tang[0], tang[1]) > _TILT_TOL:
            raise NotImplementedError(
                "crossing fill: tilted segments need the tilt tables — the "
                "W by-parts move is exact only for purely horizontal or "
                "vertical segments"
            )
        touch_lo = abs(sl[2] - gz) < tol
        touch_hi = abs(sr[2] - gz) < tol
        if touch_lo or touch_hi:
            u, w = _graded_u(h, "lo" if touch_lo else "hi", a_wire, growth, gx, gw)
        else:
            u = h * tq
            w = 0.5 * h * wg
        nodes_l.append(sl[None, :] + (u / h)[:, None] * (sr - sl)[None, :])
        t_l.append(np.repeat(tang[None, :], len(u), axis=0))
        w_l.append(w)
        u_l.append(u)
        segpos.append(np.full(len(u), g, dtype=np.int64))
    nodes = np.concatenate(nodes_l)
    t_node = np.concatenate(t_l)
    w_node = np.concatenate(w_l)
    u_phys = np.concatenate(u_l)
    segof = np.concatenate(segpos)

    F = np.zeros((n_basis, len(u_phys)))
    Fd = np.zeros((n_basis, len(u_phys)))
    for m in range(n_basis):
        for a_ in range(supp_seg.shape[1]):
            # supp_seg rows are zero-padded; a slot is live only if its
            # polynomial is nonzero (the padding trap).
            if not np.any(polys[m, a_] != 0.0):
                continue
            sel = np.nonzero(segof == supp_seg[m, a_])[0]
            if sel.size == 0:
                continue
            u = u_phys[sel]
            for p in range(d + 1):
                c = polys[m, a_, p]
                if c == 0.0:
                    continue
                F[m, sel] += c * u**p
                if p >= 1:
                    Fd[m, sel] += p * c * u ** (p - 1)

    # Signed wire-end table: (point, sign, per-basis value there). σ = −1
    # at a wire's first segment's u = 0 end, +1 at its last segment's
    # u = h end — the by-parts orientation the derivation pinned.
    seg_off = geom.seg_offsets
    ends = []
    on_axis = set(int(g) for g in seg_idx)
    for w in range(len(seg_off) - 1):
        first, last = seg_off[w], seg_off[w + 1] - 1
        if first not in on_axis:
            continue
        for gseg, sign, u_end in ((first, -1.0, 0.0), (last, +1.0, None)):
            hh = geom.h[gseg]
            u = hh if u_end is None else 0.0
            pt = geom.seg_l[gseg] + (u / hh) * (geom.seg_r[gseg] - geom.seg_l[gseg])
            fv = np.zeros(n_basis)
            for m in range(n_basis):
                for a_ in range(supp_seg.shape[1]):
                    if supp_seg[m, a_] == gseg and np.any(polys[m, a_] != 0.0):
                        fv[m] += sum(polys[m, a_, p] * u**p for p in range(d + 1))
            if np.any(fv != 0.0):
                ends.append((pt, sign, fv))
    return dict(
        nodes=nodes,
        t=t_node,
        w=w_node,
        F=F,
        Fd=Fd,
        ends=ends,
        n_basis=n_basis,
        segof=segof,
    )


def path_test_axis(n_basis, rows):
    """An axis dict for PATH-tested rows (momwire#813): razor's razor-blade
    testing spelled in `axis_data`'s own language, so every block function
    here serves it unchanged.

    `rows` is an iterable of ``(m, nodes, t, w, segof, c_before, c_after)``:
    row `m`'s testing-path quadrature points ``(q, 3)``, their flow-direction
    tangents ``(q, 3)``, weights ``(q,)``, the global segment each point lies
    on ``(q,)``, and the path's two endpoints. The dict then reads

      * ``F[m]`` = 1 on row m's own points and 0 elsewhere (the pulse),
      * ``Fd`` = 0 (the pulse has no interior derivative),
      * ``ends`` = ``(c_before, −1, e_m)`` and ``(c_after, +1, e_m)`` — the
        T2 endpoints with razor's signs, ``e_m`` the one-hot row vector,

    which makes the sandwich's ``F_A·t̂`` terms razor's T1 and the BT end term
    razor's T2 = Φ(c_after) − Φ(c_before), and nothing else survives on the
    test side. Measured against razor's own free-space fill at ε̃ = 1
    (momwire#651's probe): interior rows 6.6e-6 relative with the elementwise
    ratio exactly 1, the junction column 7.2e-9, the junction row's
    chopped-at-the-node half 5.3e-5 with `corner=False`.

    A row whose path crosses the plane (the crossing tent's) must be CHOPPED
    at the plane by the caller, one record per half with the node as the
    shared endpoint: the trunk's tables take an observer on one side only.
    """
    nodes, tl, wl, F, segof, ends = [], [], [], [], [], []
    for m, pts, t, w, seg, c_before, c_after in rows:
        pts = np.asarray(pts, dtype=float)
        q = pts.shape[0]
        nodes.append(pts)
        tl.append(np.asarray(t, dtype=float))
        wl.append(np.asarray(w, dtype=float))
        f = np.zeros((n_basis, q))
        f[m] = 1.0
        F.append(f)
        segof.append(np.asarray(seg, dtype=np.int64))
        e = np.zeros(n_basis)
        e[m] = 1.0
        ends.append((np.asarray(c_before, dtype=float), -1.0, e))
        ends.append((np.asarray(c_after, dtype=float), +1.0, e))
    n_pts = sum(x.shape[0] for x in nodes)
    return dict(
        # The split lane rebuilds a COARSE axis with `axis_data`, which can
        # only reconstruct a Galerkin axis from the context's basis — there
        # is no coarse spelling of a testing path. This flag is how
        # `cross_complete_block_split` refuses instead of silently testing
        # its far blocks with the wrong functions (momwire#813).
        path_tested=True,
        nodes=np.concatenate(nodes) if nodes else np.zeros((0, 3)),
        t=np.concatenate(tl) if tl else np.zeros((0, 3)),
        w=np.concatenate(wl) if wl else np.zeros(0),
        F=np.concatenate(F, axis=1) if F else np.zeros((n_basis, 0)),
        Fd=np.zeros((n_basis, n_pts)),
        ends=ends,
        n_basis=n_basis,
        segof=np.concatenate(segof) if segof else np.zeros(0, dtype=np.int64),
    )


def _on_plane_side(zrel, side, what):
    """`z - ground_z` forced onto the side the designed tables require.

    momwire#852. The node's own coordinate is built by segment accumulation,
    so on a uniform rise it lands within an ulp or two of the plane on
    EITHER side depending on the segment count -- 8, 9, 10, 14 and 16
    segments put `hub_deck`'s rise top at +1.04e-17 while 11, 12, 13 and 20
    put it at or below zero. `six_point` requires z >= 0 >= z', so half the
    ladder used to die on a bare `need z >= 0 >= zp` three frames down, with
    a message about an invariant rather than about the deck. Non-monotone in
    the count, because it is a coincidence of floating-point placement and
    not a resolution limit.

    The rule, in three cases:

      * on the required side (or exactly on the plane): passed through
        UNCHANGED, so every deck that solves today keeps its bits;
      * on the wrong side by less than `_PLANE_TOL`: that IS the interface,
        snapped to exactly 0.0 and served;
      * on the wrong side by `_PLANE_TOL` or more: a named refusal, because
        an above axis carrying a genuinely buried end (or the reverse) is a
        geometry error and clamping it would model something else silently.

    The above side already had the middle case, as an unconditional
    `max(..., 0.0)` with no tolerance and no refusal; this makes the two
    sides one rule.
    """
    z = np.asarray(zrel, dtype=float)
    wrong = z < 0.0 if side == "above" else z > 0.0
    if np.any(wrong):
        worst = float(np.max(np.abs(z[wrong])))
        if worst >= _PLANE_TOL:
            raise ValueError(
                f"crossing fill: the {side} axis's {what} sits "
                f"{worst:.6g} on the wrong side of ground_z, past the "
                f"{_PLANE_TOL:g} tolerance that says a point IS the "
                f"interface. The designed tables are derived for "
                f"z >= 0 >= z', so there is no honest side to put this on: "
                f"an {side} member must not cross the plane. Check the "
                f"deck's crossing junction -- exactly one above member and "
                f"the rest below (momwire#852)"
            )
        z = np.where(wrong, 0.0, z)
    return z


def _tables(ctx, eps_t, k_p, rho, z, zp, rtol, memo=None):
    """Designed tables with the deck's wire radius folded in, z relative
    to the interface. `memo` extends the exact-triple dedup across calls
    (one fill = one memo; ε̃, k₂, rtol fixed for its lifetime)."""
    a_wire = float(ctx.a_wire)
    return _near_interface.radius_tables(
        eps_t, k_p, rho, z, zp, a_wire, rtol=rtol, memo=memo
    )


def _main_sandwich(ctx, A, B, eps_t, k_p, c1, gz):
    """The M + SW + SQ sandwich over (above axis A × below axis B), dense.

    Split out of `cross_complete_block` so the REVERSED block (momwire#813)
    can share it: the designed tables accept only z ≥ 0 ≥ z′, so the above
    axis sits in the `z` slot whichever ROLE it plays, and the reversed main
    sandwich is this same product transposed — measured exact to 3e-16 at
    ε̃ = 1 and at soil A, which is what says no kernel swap is needed here
    (`scratch/813-reversed-block/probe4_localise.py`).
    """
    k2sq = k_p * k_p
    dx = A["nodes"][:, 0][:, None] - B["nodes"][:, 0][None, :]
    dy = A["nodes"][:, 1][:, None] - B["nodes"][:, 1][None, :]
    rho = np.hypot(dx, dy)
    z = np.broadcast_to((A["nodes"][:, 2] - gz)[:, None], rho.shape)
    zp = np.broadcast_to((B["nodes"][:, 2] - gz)[None, :], rho.shape)
    tables = _tables(ctx, eps_t, k_p, rho, z, zp, _CROSS_RTOL)
    U, V, W, dzW = tables["U"], tables["V"], tables["W"], tables["dzW"]

    wA, wB = A["w"], B["w"]
    txA, tyA, tzA = A["t"].T
    txB, tyB, tzB = B["t"].T
    FA_w, FB_w = A["F"] * wA, B["F"] * wB
    FdA_w, FdB_w = A["Fd"] * wA, B["Fd"] * wB

    s_u = (FA_w * txA) @ U @ (FB_w * txB).T + (FA_w * tyA) @ U @ (FB_w * tyB).T
    s_zz = (FA_w * tzA) @ (k2sq * V - dzW) @ (FB_w * tzB).T
    s_w1 = (FA_w * tzA) @ W @ FdB_w.T
    s_w2 = FdA_w @ W @ (FB_w * tzB).T
    s_phi = -FdA_w @ V @ FdB_w.T
    return c1 * (s_u + s_zz + s_w1 + s_w2 + s_phi)


def cross_complete_block(ctx, A, B, *, corner=True):
    """t_ab = M + SW + SQ + BT + CORNER over (above axis A × below axis B),
    on designed kernels. Returns the full (n_basis, n_basis) block in the
    subtracting field-block convention (`Z -= t_ab`).

    For GALERKIN rows the opposite block is this one's transpose. For
    PATH-tested rows it is not, and `cross_complete_block_reversed` builds
    it — see there for the one term that separates them."""
    eps_t, _eps_m, k_p, _k_m, _c2, _a_m = ctx.medium
    gz = float(ctx.ground_z)
    c1 = _c1_moment(ctx.omega, ctx.mu)
    t_ab = _main_sandwich(ctx, A, B, eps_t, k_p, c1, gz)
    t_ab += _ends_and_corner(ctx, A, B, eps_t, k_p, c1, gz, corner=corner)
    return t_ab


def _ends_and_corner(ctx, A, B, eps_t, k_p, c1, gz, memo=None, *, corner=True):
    """The by-parts end terms + the designed corner, on the DENSE axes —
    linear in axis size, so the admissibility split never touches them
    (and the corner must never see coarse axes or a low-rank pass; its
    V(a) rides `six_point` at `_CORNER_RTOL`, outside any memo)."""
    t_ab = np.zeros((A["n_basis"], B["n_basis"]), dtype=np.complex128)
    _txA, _tyA, tzA = A["t"].T
    FA_w = A["F"] * A["w"]
    FdA_w = A["Fd"] * A["w"]
    FdB_w = B["Fd"] * B["w"]

    # The by-parts boundary terms — test-side Φ (BT), source-side W and Φ
    # (SW, SQ) — each an end against the other axis's line, radius folded.
    for pt, sign, fv in A["ends"]:
        rho_e = np.hypot(pt[0] - B["nodes"][:, 0], pt[1] - B["nodes"][:, 1])
        te = _tables(
            ctx,
            eps_t,
            k_p,
            rho_e,
            _on_plane_side(np.full_like(rho_e, pt[2] - gz), "above", "end point"),
            _on_plane_side(B["nodes"][:, 2] - gz, "below", "quadrature node"),
            _CROSS_RTOL,
            memo=memo,
        )
        t_ab += c1 * sign * np.outer(fv, FdB_w @ te["V"])
    for pt, sign, fv in B["ends"]:
        rho_e = np.hypot(A["nodes"][:, 0] - pt[0], A["nodes"][:, 1] - pt[1])
        te = _tables(
            ctx,
            eps_t,
            k_p,
            rho_e,
            _on_plane_side(A["nodes"][:, 2] - gz, "above", "quadrature node"),
            _on_plane_side(np.full_like(rho_e, pt[2] - gz), "below", "end point"),
            _CROSS_RTOL,
            memo=memo,
        )
        t_ab += -c1 * sign * np.outer((FA_w * tzA) @ te["W"], fv)
        t_ab += c1 * sign * np.outer(FdA_w @ te["V"], fv)

    # The designed corner: node tents against each other through V at
    # R = a exactly. The sign is STRUCTURAL and orientation-carried:
    # −σ_test·σ_src·c1·V(a), which is +c1·V(a) on the deck class the
    # adjudication calibrated it on (above arm STARTING at the node,
    # σ_a σ_b = −1) and flips with the wires' parametrization — an
    # orientation-blind + wrecks a monopole spelled top-down into the
    # node (measured: 10−1007j on the P3 rise deck, the −1000j
    # truncation-class signature). Never re-pick per MEDIUM. It is the
    # INTERFACE corner, so it applies only to end pairs that BOTH stand
    # in the plane — an end elsewhere (the P3 fan's below-hub junction)
    # carries its by-parts terms above but no corner.
    if not corner:
        # A path-tested row (momwire#813): its in-plane endpoint is a plain
        # potential evaluation at z = 0⁺ (the BT term above), and the corner
        # is a Galerkin by-parts term it never had. Measured on momwire#651's
        # probe: with the corner the razor node row is off by 1.9e5 where
        # razor's own kernel has none; without it, 5e-5 (quadrature).
        return t_ab
    a_wire = float(ctx.a_wire)
    v_corner = None
    for pt_a, sig_a, fv_a in A["ends"]:
        if abs(pt_a[2] - gz) > 1e-12:
            continue
        for pt_b, sig_b, fv_b in B["ends"]:
            if abs(pt_b[2] - gz) > 1e-12:
                continue
            # One V(a) serves every pair only because every in-plane
            # value-1 end stands at the ONE crossing node the scope
            # allows — assert that, don't assume it.
            assert np.hypot(pt_a[0] - pt_b[0], pt_a[1] - pt_b[1]) < 1e-9
            if v_corner is None:
                v_corner = complex(
                    _near_interface.six_point(
                        eps_t, k_p, a_wire, 0.0, 0.0, rtol=_CORNER_RTOL
                    )[1]
                )
            t_ab += (-sig_a * sig_b * c1 * v_corner) * np.outer(fv_a, fv_b)
    return t_ab


# The SW end term's placement in the REVERSED block (momwire#813 step 1) —
# the one thing the eps~ = 1 collapse cannot settle, because W is exactly 0
# in a homogeneous medium and SW is the only end term that carries it.
#
# SETTLED by momwire#813's derivation (b), measured at 5312ca5: on the
# junction tent's below wing at soil A, `s_w1 + SW` reproduces the direct
# current-current form built from the dz′W table to 2.4e-8 with elementwise
# ratio 1.000000, while `s_w1` alone is 29x off. So SW is the by-parts
# REMNANT of the vertical-current coupling, not a source-end charge term —
# and the by-parts that produced it runs along the BELOW axis, against the
# ABOVE axis's t̂z. Both halves of that pairing are fixed by the geometry,
# not by which side is testing.
#
# "by_parts" (default) keeps them paired: SW rides the BELOW axis's ends and
#     contracts the ABOVE axis's t̂z, whichever side tests. The reversed
#     block then reproduces `t_ab.T` EXACTLY — reciprocity comes out of the
#     spelling rather than being assumed, which is the answer to the
#     question this unit was sent to ask.
# "by_role" is the other reading, kept for the record and for the contrast
#     the gates measure: SW rides the SOURCE axis's ends and contracts the
#     TEST axis's t̂z. Bit-identical to "by_parts" at eps~ = 1 (W = 0), and
#     7.938e-04 of the block away at soil A — the number that made this a
#     question before 5312ca5 answered it.
SW_BY_PARTS = "by_parts"
SW_BY_ROLE = "by_role"


def _ends_and_corner_reversed(
    ctx, P, Q, eps_t, k_p, c1, gz, memo=None, *, corner=True, sw_end=SW_BY_PARTS
):
    """`_ends_and_corner` for the REVERSED block: test axis P is BELOW, source
    axis Q is ABOVE. Returns (P n_basis × Q n_basis).

    Two assignments the forward block conflates, because there the test axis
    IS the above axis:

      * which table slot — the above axis goes in `z`, the below in `z′`,
        always (`six_point` raises otherwise);
      * which by-parts term — BT rides the TEST axis's ends, SW + SQ the
        SOURCE axis's.

    Every end term except SW is bit-identical to the forward block's
    transpose in both media under either reading; SW is where they differ and
    `sw_end` says which. The default pairs it with `s_w1` as its by-parts
    partner (momwire#813 derivation (b), measured at 5312ca5), and under that
    pairing the whole reversed block reproduces `t_ab.T` exactly.
    """
    t_ba = np.zeros((P["n_basis"], Q["n_basis"]), dtype=np.complex128)
    _txP, _tyP, tzP = P["t"].T
    _txQ, _tyQ, tzQ = Q["t"].T
    FP_w = P["F"] * P["w"]
    FdP_w = P["Fd"] * P["w"]
    FQ_w = Q["F"] * Q["w"]
    FdQ_w = Q["Fd"] * Q["w"]

    # BT — the test axis's ends. P is below, so its z lands in the z′ slot
    # (clamped at the plane, the mirror of the forward's `max(..., 0.0)`).
    for pt, sign, fv in P["ends"]:
        rho_e = np.hypot(Q["nodes"][:, 0] - pt[0], Q["nodes"][:, 1] - pt[1])
        te = _tables(
            ctx,
            eps_t,
            k_p,
            rho_e,
            Q["nodes"][:, 2] - gz,
            np.full_like(rho_e, min(pt[2] - gz, 0.0)),
            _CROSS_RTOL,
            memo=memo,
        )
        t_ba += c1 * sign * np.outer(fv, FdQ_w @ te["V"])
        if sw_end == SW_BY_PARTS:
            # SW paired with s_w1 by the by-parts that produced it: on the
            # BELOW axis's ends, contracting the ABOVE axis's t̂z (5312ca5).
            t_ba += -c1 * sign * np.outer(fv, (FQ_w * tzQ) @ te["W"])

    # SQ — the source axis's ends (+ SW under the rejected "by_role" reading).
    for pt, sign, fv in Q["ends"]:
        rho_e = np.hypot(P["nodes"][:, 0] - pt[0], P["nodes"][:, 1] - pt[1])
        te = _tables(
            ctx,
            eps_t,
            k_p,
            rho_e,
            np.full_like(rho_e, max(pt[2] - gz, 0.0)),
            P["nodes"][:, 2] - gz,
            _CROSS_RTOL,
            memo=memo,
        )
        if sw_end == SW_BY_ROLE:
            t_ba += -c1 * sign * np.outer((FP_w * tzP) @ te["W"], fv)
        t_ba += c1 * sign * np.outer(FdP_w @ te["V"], fv)

    if not corner:
        return t_ba
    # The corner is symmetric in the two ends' one-hots (−σσ′·c1·V(a)), so
    # it is the forward's transposed and needs no orientation of its own.
    a_wire = float(ctx.a_wire)
    v_corner = None
    for pt_p, sig_p, fv_p in P["ends"]:
        if abs(pt_p[2] - gz) > 1e-12:
            continue
        for pt_q, sig_q, fv_q in Q["ends"]:
            if abs(pt_q[2] - gz) > 1e-12:
                continue
            assert np.hypot(pt_p[0] - pt_q[0], pt_p[1] - pt_q[1]) < 1e-9
            if v_corner is None:
                v_corner = complex(
                    _near_interface.six_point(
                        eps_t, k_p, a_wire, 0.0, 0.0, rtol=_CORNER_RTOL
                    )[1]
                )
            t_ba += (-sig_p * sig_q * c1 * v_corner) * np.outer(fv_p, fv_q)
    return t_ba


def cross_complete_block_reversed(ctx, P, Q, *, corner=True, sw_end=SW_BY_PARTS):
    """The block the other way round: BELOW test rows × ABOVE source columns.

    `cross_complete_block` fills (above rows × below columns). bspline gets
    the opposite block as that one's transpose, which Galerkin reciprocity
    licenses; a PATH-tested fill cannot assume it, because the test
    functional is no longer the basis (momwire#813).

    So this builds it directly. `P` is the below axis and carries the rows
    (razor's paths through `path_test_axis`, or `axis_data` for a Galerkin
    check); `Q` is the above axis and carries the columns. Pass
    `corner=False` for path-tested rows — the corner is a Galerkin by-parts
    term (the momwire#651 probe: 1.9e5 added where razor's truth has none).

    Measured on `crossing_deck(level=1)` at ε̃ = 1, against razor's own
    free-space `Z[below rows, above cols]`: 6.56e-06 relative with the
    elementwise ratio exactly 1 — the same interior class the forward block
    reached (6.6e-6), on both quadrature lanes.

    Reciprocity is then a RESULT rather than an assumption: with Galerkin
    axes on both sides this reproduces `cross_complete_block`'s transpose
    bit for bit, in both media, under the default `sw_end`. The rejected
    `SW_BY_ROLE` spelling agrees at ε̃ = 1 and is 7.94e-4 away at soil A.
    """
    eps_t, _eps_m, k_p, _k_m, _c2, _a_m = ctx.medium
    gz = float(ctx.ground_z)
    c1 = _c1_moment(ctx.omega, ctx.mu)
    t_ba = _main_sandwich(ctx, Q, P, eps_t, k_p, c1, gz).T
    t_ba += _ends_and_corner_reversed(
        ctx, P, Q, eps_t, k_p, c1, gz, corner=corner, sw_end=sw_end
    )
    return t_ba


def _row_weights(ax, ii):
    """The four per-basis row-weight matrices of one axis, restricted to
    the point subset `ii`: (F·w·t̂x, F·w·t̂y, F·w·t̂z, F′·w)."""
    tx, ty, tz = ax["t"][ii].T
    Fw = ax["F"][:, ii] * ax["w"][ii]
    Fdw = ax["Fd"][:, ii] * ax["w"][ii]
    return Fw * tx, Fw * ty, Fw * tz, Fdw


def _sandwich_dense(A, B, iA, iB, K, k2sq):
    """The five-term M+SW+SQ (main) sandwich over dense kernel matrices
    restricted to (iA, iB) — the same term order as the reference fill."""
    P1, P2, P3, P4 = _row_weights(A, iA)
    Q1, Q2, Q3, Q4 = _row_weights(B, iB)
    return (
        P1 @ K["U"] @ Q1.T
        + P2 @ K["U"] @ Q2.T
        + P3 @ (k2sq * K["V"] - K["dzW"]) @ Q3.T
        + P3 @ K["W"] @ Q4.T
        + P4 @ K["W"] @ Q3.T
        - P4 @ K["V"] @ Q4.T
    )


def _axis_segment_tree(geom, seg_idx, leaf):
    """Cluster tree over one axis's SEGMENTS (boxes = segment endpoints).
    Cluster indices are positions into `seg_idx`; returns (tree, seg_idx
    as an array) so callers can map back to global segment ids."""
    idx = np.asarray(seg_idx, dtype=np.int64)
    lo = np.minimum(geom.seg_l[idx], geom.seg_r[idx])
    hi = np.maximum(geom.seg_l[idx], geom.seg_r[idx])
    tree = _aca.build_cluster_tree(np.arange(idx.size), lo, hi, leaf)
    return tree, idx


def _refuse_path_tested(*axes):
    """The #688 split cannot serve a path-tested axis.

    Its far blocks are evaluated on COARSE axes rebuilt by `axis_data` from
    the context's basis. A path-test axis has no such spelling — razor's
    testing paths are not a basis — so a coarse rebuild silently replaces
    the test functions with the Galerkin tents on exactly the far blocks,
    and the block comes back ~20% wrong with nothing raised (measured 2.04e-1
    relative on `crossing_deck(level=1)`, in BOTH directions; Galerkin axes
    on the same deck agree dense-to-split at 1.8e-18).

    momwire#813 half 1 never met this because its gates are all dense. Half
    2's masked assembly would have, so it refuses here rather than there.
    """
    for ax in axes:
        if ax.get("path_tested"):
            raise ValueError(
                "the admissibility split cannot serve a path-tested axis: its "
                "far blocks ride coarse axes rebuilt from the context's basis, "
                "which silently substitutes Galerkin tents for the testing "
                "paths (momwire#813). Use the dense entry point."
            )


def cross_complete_block_split(ctx, a_idx, b_idx, A, B, *, corner=True):
    """`cross_complete_block` through the #688 admissibility split.

    The (above segments × below segments) product is partitioned by the
    standard box rule (`_aca.build_block_tree`, η = 1): inadmissible
    blocks — every pair meeting the crossing node among them, since
    touching boxes have distance 0 — are evaluated dense-direct on the
    dense graded axes, batched into ONE designed-tables call so the
    exact-triple memo and the C++ batch keep their full scope.
    Admissible far blocks are evaluated on the coarse axes and through
    `_aca.aca_partial`, one factorization per kernel {U, V, W, ∂zW},
    sampling designed rows/columns through a shared cache (one designed
    evaluation serves all four kernels at that row/column).

    The by-parts end terms and the corner (−σσ′·c1·V(a)) ride the dense
    axes direct, always — they are linear in axis size and the corner
    routes through neither coarse axes nor the low-rank pass."""
    if _FORCE_DENSE:
        return cross_complete_block(ctx, A, B, corner=corner)
    _refuse_path_tested(A, B)

    eps_t, _eps_m, k_p, _k_m, _c2, _a_m = ctx.medium
    gz = float(ctx.ground_z)
    c1 = _c1_moment(ctx.omega, ctx.mu)
    memo = {}  # one fill = one memo (eps_t, k_p, _CROSS_RTOL fixed here)
    t_ab = _main_split(ctx, a_idx, b_idx, A, B, eps_t, k_p, c1, gz, memo)
    t_ab += _ends_and_corner(ctx, A, B, eps_t, k_p, c1, gz, memo=memo, corner=corner)
    return t_ab


def _main_split(ctx, a_idx, b_idx, A, B, eps_t, k_p, c1, gz, memo):
    """The split fill's main sandwich over (above A × below B) — everything
    `cross_complete_block_split` does except the ends and the corner.

    Split out for the same reason as `_main_sandwich` (momwire#813): the
    reversed block's main part is this product transposed."""
    k2sq = k_p * k_p

    tree_a, seg_a = _axis_segment_tree(ctx.geom, a_idx, _CLUSTER_LEAF_SEGS)
    tree_b, seg_b = _axis_segment_tree(ctx.geom, b_idx, _CLUSTER_LEAF_SEGS)
    far, near = _aca.build_block_tree(tree_a, tree_b, _ADM_ETA)

    t_main = np.zeros((A["n_basis"], B["n_basis"]), dtype=np.complex128)

    if far:
        Ac = axis_data(ctx, a_idx, coarse=True)
        Bc = axis_data(ctx, b_idx, coarse=True)

    # ---- direct blocks: near pairs on the dense axes + small far blocks
    # on the coarse axes (sampling a small block costs more than its full
    # coarse product), ALL in one batched designed evaluation so the
    # exact-triple memo dedups across every block — a symmetric screen's
    # mirrored blocks are the same triples.
    direct, far_aca = [], []
    for cs, ct in near:
        iA = np.flatnonzero(np.isin(A["segof"], seg_a[cs.indices]))
        iB = np.flatnonzero(np.isin(B["segof"], seg_b[ct.indices]))
        if iA.size and iB.size:
            direct.append((A, B, iA, iB))
    for cs, ct in far:
        iA = np.flatnonzero(np.isin(Ac["segof"], seg_a[cs.indices]))
        iB = np.flatnonzero(np.isin(Bc["segof"], seg_b[ct.indices]))
        if iA.size == 0 or iB.size == 0:
            continue
        if iA.size * iB.size > _ACA_COST_GUARD * (iA.size + iB.size):
            far_aca.append((iA, iB))
        else:
            direct.append((Ac, Bc, iA, iB))

    if direct:
        specs, rho_cat, z_cat, zp_cat = [], [], [], []
        for AX, BX, iA, iB in direct:
            pa, pb = AX["nodes"][iA], BX["nodes"][iB]
            rho = np.hypot(
                pa[:, 0][:, None] - pb[:, 0][None, :],
                pa[:, 1][:, None] - pb[:, 1][None, :],
            )
            specs.append((AX, BX, iA, iB, rho.shape))
            rho_cat.append(rho.ravel())
            z_cat.append(np.repeat(pa[:, 2] - gz, iB.size))
            zp_cat.append(np.tile(pb[:, 2] - gz, iA.size))
        tab = _tables(
            ctx,
            eps_t,
            k_p,
            np.concatenate(rho_cat),
            np.concatenate(z_cat),
            np.concatenate(zp_cat),
            _CROSS_RTOL,
            memo=memo,
        )
        off = 0
        for AX, BX, iA, iB, shp in specs:
            nel = shp[0] * shp[1]
            K = {kk: tab[kk][off : off + nel].reshape(shp) for kk in _CROSS_KEYS}
            off += nel
            t_main += _sandwich_dense(AX, BX, iA, iB, K, k2sq)

    # ---- large far blocks: coarse axes, low-rank ACA per kernel. The
    # row/column samples ride the SAME memo — identical matrices in
    # mirrored blocks pick identical pivots, so their samples dedup.
    for iA, iB in far_aca:
        pa, pb = Ac["nodes"][iA], Bc["nodes"][iB]
        zA, zB = pa[:, 2] - gz, pb[:, 2] - gz
        rho = np.hypot(
            pa[:, 0][:, None] - pb[:, 0][None, :],
            pa[:, 1][:, None] - pb[:, 1][None, :],
        )
        m, n = iA.size, iB.size
        rows, cols = {}, {}

        def _row6(i, rho=rho, zA=zA, zB=zB, rows=rows, n=n):
            if i not in rows:
                te = _tables(
                    ctx,
                    eps_t,
                    k_p,
                    rho[i],
                    np.full(n, zA[i]),
                    zB,
                    _CROSS_RTOL,
                    memo=memo,
                )
                rows[i] = np.stack([te[kk] for kk in _CROSS_KEYS])
            return rows[i]

        def _col6(j, rho=rho, zA=zA, zB=zB, cols=cols, m=m):
            if j not in cols:
                te = _tables(
                    ctx,
                    eps_t,
                    k_p,
                    rho[:, j],
                    zA,
                    np.full(m, zB[j]),
                    _CROSS_RTOL,
                    memo=memo,
                )
                cols[j] = np.stack([te[kk] for kk in _CROSS_KEYS])
            return cols[j]

        Kf = {}
        for ki, kk in enumerate(_CROSS_KEYS):
            Kf[kk] = _aca.aca_partial(
                lambda i, ki=ki: _row6(i)[ki],
                lambda j, ki=ki: _col6(j)[ki],
                m,
                n,
                tol=_ACA_TOL,
            )
        P1, P2, P3, P4 = _row_weights(Ac, iA)
        Q1, Q2, Q3, Q4 = _row_weights(Bc, iB)

        def _lr(P, kk, Q, Kf=Kf):
            Uf, Vf = Kf[kk]
            return (P @ Uf) @ (Vf @ Q.T)

        t_main += (
            _lr(P1, "U", Q1)
            + _lr(P2, "U", Q2)
            + k2sq * _lr(P3, "V", Q3)
            - _lr(P3, "dzW", Q3)
            + _lr(P3, "W", Q4)
            + _lr(P4, "W", Q3)
            - _lr(P4, "V", Q4)
        )

    return c1 * t_main


def cross_complete_block_reversed_split(
    ctx, p_idx, q_idx, P, Q, *, corner=True, sw_end=SW_BY_PARTS
):
    """`cross_complete_block_reversed` through the #688 admissibility split.

    `p_idx` are the BELOW axis's segments (test rows), `q_idx` the ABOVE
    axis's (source columns) — the same argument order as the axes. The main
    sandwich is the forward split's, transposed; the ends follow the roles.
    """
    if _FORCE_DENSE:
        return cross_complete_block_reversed(ctx, P, Q, corner=corner, sw_end=sw_end)
    _refuse_path_tested(P, Q)

    eps_t, _eps_m, k_p, _k_m, _c2, _a_m = ctx.medium
    gz = float(ctx.ground_z)
    c1 = _c1_moment(ctx.omega, ctx.mu)
    memo = {}
    t_ba = _main_split(ctx, q_idx, p_idx, Q, P, eps_t, k_p, c1, gz, memo).T
    t_ba += _ends_and_corner_reversed(
        ctx, P, Q, eps_t, k_p, c1, gz, memo=memo, corner=corner, sw_end=sw_end
    )
    return t_ba


def _g_of_r(k, R):
    return np.exp(-1j * k * R) / R


def _bnd_and_corner(ax, k, a_wire, gz, mirror):
    """The same-medium by-parts boundary shape on one axis (β = 1):
    −test-end rows, −source-end columns, +corner — the derivation's
    −,−,+ sign structure. Closed-form kernel G = e^{−jkR}/R at
    R = √(‖Δr‖² + a²), source positions mirrored through the interface
    for the image family. Returns (bnd, corner)."""
    pts = ax["nodes"]
    Fdw = ax["Fd"] * ax["w"]
    n = ax["n_basis"]
    bnd = np.zeros((n, n), dtype=np.complex128)
    corner = np.zeros((n, n), dtype=np.complex128)

    def _mir(p):
        p = np.array(p, dtype=float, copy=True)
        if mirror:
            p[..., 2] = 2.0 * gz - p[..., 2]
        return p

    src = _mir(pts)
    for ptE, sig, fvT in ax["ends"]:  # test ends (observation, unmirrored)
        dr = src - np.asarray(ptE)[None, :]
        R = np.sqrt(a_wire * a_wire + np.einsum("ij,ij->i", dr, dr))
        bnd += -sig * np.outer(fvT, Fdw @ _g_of_r(k, R))
    for ptEp, sigp, fvS in ax["ends"]:  # source ends (mirrored)
        pe = _mir(np.asarray(ptEp))
        dr = pts - pe[None, :]
        R = np.sqrt(a_wire * a_wire + np.einsum("ij,ij->i", dr, dr))
        bnd += -sigp * np.outer(Fdw @ _g_of_r(k, R), fvS)
    for ptE, sig, fvT in ax["ends"]:
        for ptEp, sigp, fvS in ax["ends"]:
            pe = _mir(np.asarray(ptEp))
            dr = np.asarray(ptE) - pe
            R = np.sqrt(a_wire * a_wire + float(dr @ dr))
            corner += sig * sigp * _g_of_r(k, R) * np.outer(fvT, fvS)
    return bnd, corner


def self_completions(ctx, ax_b, ax_a):
    """The self families' missing bnd + corner content, both media, on
    graded axes. Returned as the ADDITIVE Z correction (the fill's
    `Z -= image` convention already folded in: β_dir·(bnd+cor)(G_dir)
    − β_img·(bnd+cor)(G_img) per family)."""
    _eps_t, eps_m, k_p, k_m, c2, a_m = ctx.medium
    gz = float(ctx.ground_z)
    a_wire = float(ctx.a_wire)
    omega, eps0 = ctx.omega, ctx.eps
    total = np.zeros((ax_b["n_basis"],) * 2, dtype=np.complex128)
    for ax, k, wgt, eps in ((ax_b, k_m, a_m, eps_m), (ax_a, k_p, c2, eps0)):
        bnd_dir, cor_dir = _bnd_and_corner(ax, k, a_wire, gz, mirror=False)
        bnd_img, cor_img = _bnd_and_corner(ax, k, a_wire, gz, mirror=True)
        beta_dir = 1.0 / (1j * omega * eps * 4 * np.pi)
        beta_img = wgt / (1j * omega * eps * 4 * np.pi)
        total += beta_dir * (bnd_dir + cor_dir) - beta_img * (bnd_img + cor_img)
    return total
