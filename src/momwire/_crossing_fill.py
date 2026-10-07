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
the below×below tent pairs at R = a. With more than one crossing node
(antennaknobs plan U9) the same loops meet end pairs standing at two
different nodes, and each pair reads V at its own separation,
√(ρ² + a²): the corner is a point-charge pair term, so a second node is a
second V, not a new term. A pair at one node reads V(a) through exactly
the call a one-node deck makes.

Scope guards this module inherits from the derivation:

  * one wire radius across the deck — the radius rule ρ_eff = √(ρ² + a²)
    is the corner's regularization — or, for a caller that opts in
    (BSpline, antennaknobs plan U5), one radius per side of the interface:
    see `cross_complete_blocks_two_radius`.

TILTED SEGMENTS ARE SERVED (momwire#936). They were refused until
2026-09-07 on the reading that "the W by-parts move uses t̂⊥·∇⊥ = d/dl,
exact only" for axis-aligned segments. Measured, that reading does not
survive: THE ASSEMBLY NEVER MAKES THAT SUBSTITUTION.

The decisive evidence is an absence. The substituted form needs the
SOURCE-side derivative kernels `dzpW` / `dzpV` as substitutes for a
transverse derivative — and this module makes no such substitution. What it
contracts is `W` against the other axis's `Fd`, which is the DIRECT
spelling; `dzpW` appears only inside `k²V + ∂z′W`, the ẑẑ dyad's own
coefficient (momwire#956), not as a substituted derivative. Every term of the
main sandwich is written in tangent COMPONENTS — s_u pairs tx/ty through
U, s_zz pairs tz through k²V + ∂z′W (momwire#956: the SOURCE-side
derivative; it was −∂zW, the test-side W term by parts on a vertical
test, until #956 found s_w2 counting that term a second time), and the W
terms carry tz against the other axis's Fd — and Fd is the filament charge
∇·(F t̂) = dF/dl, which is the arclength derivative at ANY orientation. The
`tz` in the by-parts boundary terms is the ẑ-dyad's own coefficient, not an
alignment assumption; the two W end terms (SW on the below ends, TW on the
above ends) are what the by-parts leave, one per axis.

Four measurements, in `scratch/936-study/`:

  * by parts is orientation-EXACT: the direct and by-parted spellings of
    the W term agree to 1e-11..1e-13 at every α, at every node-grading
    rung, with the segment end ON the interface (probe1, probe5). The
    spelling the ASSEMBLY uses is the direct one, so it is exact at any
    orientation; probe1's third column re-implemented the substitution
    the OLD docstring described, and that is the only thing that differs
    with α. A scope note describing a move the code does not make is what
    kept this refused;
  * the non-W structure is orientation-general: the ε̃ = 1 collapse holds
    for a tilted above member, 0.163 against the untilted 0.145 — the
    same node-mesh class, and W ≡ 0 there so the test isolates the rest
    (probe2);
  * end to end against NEC-5 on the OP deck, the lean 0 → 45° sweep adds
    0.70 pp to a residual that is already 4.48 pp at lean 0 (probe9);
  * and restoring the supposedly-dropped −(t⊥·ρ̂)∂W/∂ρ makes the answer
    WORSE, −5.18 % → −11.46 % at 45°, while vanishing identically at
    lean 0 as it must (probe10).

The 0.70 pp lean drift is a KNOWN RESIDUAL, not a fixed defect: it scales
as ~sin²α (1 : 4.3 : 10 at 15/30/45°) where the dropped term would enter
linearly in sin α (1 : 1.9 : 2.7), so its signature is a discretisation
difference between two engines as the geometry leaves axis alignment.

`bspline.BSplineSolver._compute_Z_operator_buried` is the only caller today;
it hands the fill a `CrossingContext` (below) rather than itself (momwire#801).
"""

from __future__ import annotations

import functools
import os
import sys
import warnings
from typing import NamedTuple, Protocol

import numpy as np
import scipy.sparse as _sp
from numpy.polynomial.legendre import leggauss

from . import _accel, _aca, _cancel, _ground_refl, _near_interface, _sommerfeld_below
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
    # momwire#926: what the stand-off floor needs to price this arm's
    # grading. `slope` is |dz/dl| along the node-adjacent edge (0 for a
    # level arm, 1 for a plumb one); `h_floor` is the height that arm's
    # own radius/jacket must clear, or None where the floor cannot apply
    # (a below-side arm never rises through the interface). Defaulted so a
    # caller that does not know them still builds an arm.
    slope: float = 0.0
    h_floor: float | None = None


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
#
# Since momwire#980 (step B) the basis is not read as polynomials either.
# `axis_data` needs three things of it -- value and arc-derivative SAMPLES at
# the quadrature nodes, and the per-basis VALUE at a wire end -- and asks a
# `BasisSampler` for them. `BasisPolynomials` is the piecewise-polynomial
# sampler (bspline, razor's tents) and keeps its bytes: its `samples` IS
# `_basis_samples`. A basis that is not polynomial on a segment -- the NEC
# three-term sinusoid, `SinusoidalGalerkinSolver` -- hands the fill its own
# sampler and every block function downstream is unchanged, because nothing
# downstream of the axis dict knows how F/Fd were produced (the #980 spike:
# the "moment tables" here are `leggauss` weights on POINT kernel tables).


class BasisSampler(Protocol):
    """What `axis_data` reads of a basis (momwire#980 step B).

    `n_basis`
        The number of basis rows on this axis's solver.
    `samples(seg_runs, u_phys)`
        `(F, Fd, seg_rows)`: every basis's value and arc derivative at the
        axis nodes, `(n_basis, n_nodes)` each, zero where a basis has no
        support on the node's segment; and `seg_rows[g]` = the sorted basis
        rows with a live wing on segment `g`, for every `g` in `seg_runs`.
        `seg_runs` is `_segment_runs`' `{segment: (start, count)}` and
        `u_phys[start:start+count]` the local arc `u ∈ [0, h]` from `seg_l`
        of that segment's nodes, in node order.

        SPARSE OR DENSE (momwire#1029 phase 2 / #1109). A basis with local
        support has at most `degree + 1` nonzeros per NODE, so `axis_data`
        carries F/Fd as `scipy.sparse.csr_array` under the keys `F_csr` /
        `Fd_csr`, and a sampler may return that directly —
        `BasisPolynomials` and `SinusoidalBasisSampler` do, built from the
        segment structure and never materialised dense, because the dense
        pair is 4.13 GB of the 8.55 GB peak on the 150-radial screen. A sampler that returns dense arrays
        is converted here and pays only its own dense allocation.
    `end_values(gseg, u)`
        `(n_basis,)`: every basis's value at arc `u` of segment `gseg`, zero
        off support. `axis_data` calls it at `u = 0` of a wire's first
        segment and `u = h` of its last, and keeps the end only if some
        value is nonzero.

    Complex values are allowed (a basis at an in-medium wavenumber): every
    consumer of F/Fd/`fv` downstream is dtype-agnostic.
    """

    n_basis: int

    def samples(self, seg_runs, u_phys): ...

    def end_values(self, gseg, u): ...


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

    # -- the BasisSampler face (momwire#980 step B) ------------------------
    # Delegation, not reimplementation: `samples` is `_basis_samples` and
    # `end_values` is `_end_values`, so the bspline and razor crossing fills
    # produce the same bytes they did before the seam existed.

    @property
    def n_basis(self):
        return int(self.polys.shape[0])

    def samples(self, seg_runs, u_phys):
        return _basis_samples(self.supp_seg, self.polys, seg_runs, u_phys)

    def end_values(self, gseg, u):
        live = np.any(self.polys != 0.0, axis=2)
        return _end_values(self.supp_seg, self.polys, live, gseg, u, self.degree)


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
    tents as `BasisPolynomials` the same way; a non-polynomial basis hands in
    any `BasisSampler` (momwire#980). `a_wire` is the deck's ONE wire
    radius — the scope guard in the module docstring — and `ground_z` the
    interface height.

    `a_above` / `a_below` are set only on a TWO-RADIUS node (every above wire
    at `a_above`, every below wire at `a_below`, the two different;
    antennaknobs plan U5), and `a_wire` is then their minimum, the node
    grading's radius. `None` on every one-radius deck, which keeps the shipped
    spelling.
    """

    basis: BasisSampler
    geom: AxisGeometry
    medium: Medium
    ground_z: float
    a_wire: float
    omega: float
    mu: float
    eps: float
    a_above: float | None = None
    a_below: float | None = None
    # The solver's `CancelToken` (or None), installed as the ambient token by
    # every public entry below (`_cancelable`) so the fill's seams and the
    # near-interface tables can poll it without a token argument.
    cancel: object | None = None


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


# The panel this advisory asks for at the node, in metres. It was a bare
# "~6 mm" in the message text until momwire#926 needed to COMPARE it with
# something.
NODE_TARGET_PANEL = 0.006


def _cancelable(fn):
    """Run a public fill entry with its context's token installed
    (`_cancel.scope`). Every entry takes the `CrossingContext` first."""

    @functools.wraps(fn)
    def run(ctx, *args, **kwargs):
        with _cancel.scope(ctx.cancel):
            return fn(ctx, *args, **kwargs)

    return run


def node_panel_floor(h_floor_m, slope):
    """Shortest node panel the stand-off floor permits on this slope, or None.

    momwire#926: a conductor leaving a node ON the interface at slope `s`
    (rise per unit arclength) sits at height h = s·l, so momwire#865's
    stand-off floor h ≥ h_floor forbids any vertex closer to the node than

        L_min = h_floor / s

    Grading toward the node is precisely the act of putting vertices there,
    so this advisory's instruction walks a sloping deck into that refusal —
    the two rules did not know about each other, which is the whole of #926.
    On the issue's own deck (5 % slope, a = 1 mm, bare) the floor is 2 mm and
    L_min is 40 mm, against the ~6 mm asked for unconditionally before.

    None when the floor cannot bind, and both cases are real rather than
    defensive: a LEVEL arm (s = 0) never approaches the interface by moving
    toward the node, so no panel length is forbidden — that is the flat
    buried radial, and it must keep the old text; and `h_floor` is None for a
    below-side arm, which never rises through the plane at all.

    The floor grows without limit as the slope flattens, which is worth
    knowing before reading the number: a radial rising 0.1 m over 10 m
    cannot have a node panel under 200 mm.
    """
    if h_floor_m is None or not (slope > 0.0) or not (h_floor_m > 0.0):
        return None
    return h_floor_m / slope


# What an unresolved node costs, and the lever past grading, per FILL. The
# default is BSplineSolver's (the text this advisory has always carried);
# razor passes its own (momwire#1149 U2b), because its path-tested node is a
# different quantity: no n_qp_pair, and a node worth a fraction of an ohm
# against its first-order far mesh rather than several ohms of quadrature.
_BSPLINE_NODE_WORTH = (
    "At the default quadrature order this node is worth ~4.5 ohm on the "
    "soil-A fan and a global density sweep will report it as converged while "
    "it is not (momwire#674, re-derived in momwire#760)"
)


def warn_coarse_node(arms, *, worth=None, levers=None):
    """Warn when a crossing junction's node region is unresolved.

    `arms` is an iterable of `NodeArm`. Returns the worst arm by
    `h_resolved` (or None for a deck with no crossing junction), so a
    caller — a test, a diagnostic — can read the number back whether or
    not it crossed the bar.

    `worth` and `levers` are the two sentences that belong to the FILL
    rather than to the geometry: what the unresolved node costs, and what
    to do past grading. None keeps BSplineSolver's text byte for byte;
    `RazorSolver` passes its own (momwire#1149 U2b).

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
    # momwire#926: on a SLOPING arm the stand-off floor forbids the panel
    # this advisory would otherwise ask for, so ask for the shortest one it
    # permits and say why. Silent where the floor cannot bind, which keeps
    # every level and below-side deck's text byte for byte what it was.
    #
    # Chosen across ALL arms, not read off `worst`. Grading a node means
    # grading its arms together, so what binds is the TIGHTEST floor any of
    # them imposes — and that is usually not the worst-meshed arm. On #926's
    # own deck the worst arm is the plumb mast, whose slope of 1 puts L_min
    # at 2 mm; the 40 mm that makes the deck refuse belongs to a 5 % radial
    # the mesh bar does not single out. Reading the floor off `worst` printed
    # the old text on the one deck the issue is about.
    floor_arm, l_min = None, None
    for a in arms:
        cand = node_panel_floor(a.h_floor, a.slope)
        if cand is not None and (l_min is None or cand > l_min):
            floor_arm, l_min = a, cand
    if l_min is not None and l_min > NODE_TARGET_PANEL:
        panel = (
            f"from L_min = {l_min * 1000:.1f} mm at the node (wire "
            f"{floor_arm.wire} leaves the node on a "
            f"{floor_arm.slope * 100:.1f} % slope, so the stand-off floor "
            f"forbids anything shorter: a vertex closer than that to the node "
            f"sits below h = {floor_arm.h_floor * 1000:.2f} mm, that wire's "
            f"floor, and the fill is REFUSED by name; momwire#865/#926) "
            f"growing"
        )
        cannot = (
            f" Note this is coarser than the ~{NODE_TARGET_PANEL * 1000:.0f} "
            f"mm this advisory asks for on a level arm, so on this slope the "
            f"node cannot be graded to the bar at all: raising n_qp_pair is "
            f"the lever that is left."
        )
    else:
        panel = f"~{NODE_TARGET_PANEL * 1000:.0f} mm at the node growing"
        cannot = ""
    warnings.warn(
        f"crossing node: wire {worst.wire}'s {worst.end} is the {worst.side} "
        f"member, and the finest mesh within {NODE_REACH * 1000:.0f} mm of "
        f"the node is {worst.h_resolved * 1000:.1f} mm, above the "
        f"{NODE_H_BAR * 1000:.0f} mm bar{gap}. "
        f"{_BSPLINE_NODE_WORTH if worth is None else worth}. Grade the node: "
        f"geometric panels toward "
        f"the shared point, {panel} to the design's own "
        f"segment length away from it, spelled as extra VERTICES in the "
        f"wire's polyline with per-edge counts in n_per_edge_per_wire so "
        f"the grading cannot change junction topology.{cannot} "
        f"{_RAISE_THE_ORDER if levers is None else levers}. "
        f"Advisory: nothing is remeshed. See stevenmburns/momwire#696.",
        CoarseCrossingNode,
        **_WARN_TARGET,
    )
    return worst


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
_CROSS_KEYS = ("U", "V", "W", "dzpW")
# The whole-axis main sandwich's kernel tables are evaluated in column chunks
# over the BELOW axis's nodes, each chunk sized to this many bytes of live
# per-pair working set. `_MAIN_BYTES_PER_PAIR` is that working set for one
# (above node, below node) pair: the dedup pass's lexsort over the folded
# triples (the (n, 3) float rows twice, four intp arrays and the fold, ~88 B
# before numpy's sort scratch) and, in the product pass, the four complex
# tables (64 B) plus the k²V + ∂z′W operand and its temporary (32 B). 64 MiB
# is ~0.5 M pairs a chunk — hub_deck(16) x8's 3.7 M-pair forward block in 8
# chunks — which keeps the tables out of the fill's peak (they were 0.7 GB
# there, the (n, 6) gather plus its transposed copy) while the per-chunk
# overhead stays a few numpy calls. Chunking never moves a bit
# (`_chunked_tables`), so the budget is a memory choice only. Since #1168 a
# chunk's six left products are contracted before the next chunk arrives
# (`_streamed_sandwich`) rather than assembled whole; they are 96·|rA|/nA
# bytes per pair on top of this (~48 B on a path-tested above axis, where
# |rA| ≈ nA/2), and the cut never moves a bit either.
_MAIN_CHUNK_BYTES = 64 * 2**20
_MAIN_BYTES_PER_PAIR = 128

# TEST-ONLY. False contracts the main sandwich's chunked tables the pre-#1168
# way — the six whole left products assembled, then one contraction — which
# is the in-process reference the streamed path (`_streamed_sandwich`) is
# gated against to the bit.
_MAIN_STREAMED = True
# TEST-ONLY negative control for that gate: False contracts every chunk's
# columns for every row and adds the partial sums, which reassociates the
# running sum of each row whose pattern straddles a chunk boundary.
_STREAMED_WHOLE_ROWS = True

# The whole-run switch that routes the split entries
# (`cross_complete_block_split`, `cross_complete_blocks_two_radius`,
# `cross_complete_block_reversed_split`) onto the WHOLE-AXIS fill — every node
# pair direct through `_main_sandwich`, no admissibility partition, no coarse
# axes, no ACA. For a timing comparison or a bisect; parity tests drive both
# paths in-process by calling the two entries directly.
#
# It was `_FORCE_DENSE` until momwire#1168 U5, from when the whole-axis
# sandwich densified its weights (six full GEMMs). Since 9cd2d22 it contracts
# the sparse weights (`_sandwich_dense`), so "dense" named nothing it still
# does, and the split's near blocks are just as direct. The
# environment variable keeps its LEGACY name, `MOMWIRE_CROSSING_FORCE_DENSE`,
# so existing scripts and bisect notes keep working.
_WHOLE_AXIS_NO_ACA = bool(os.environ.get("MOMWIRE_CROSSING_FORCE_DENSE"))


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


@_cancelable  # its polls read the ambient token; solvers call it directly
def axis_data(
    ctx,
    seg_idx,
    coarse=False,
    *,
    growth=None,
    panel_order=None,
    q=None,
    share_from=None,
    grade_near_plane=False,
):
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

    **`grade_near_plane`** (momwire#1152, razor's cross blocks only) also
    grades a segment that does NOT touch the plane but APPROACHES it: its
    nearer end at distance ``d`` from the plane, its farther end beyond
    ``2d``. The graded panels then run toward the nearer end with the first
    panel ``max(d, a)`` wide — the a-scale floor a touching segment's first
    panel has, lifted to the distance the integrand actually varies on,
    since an observer across the plane sees the source through ``d + d′``
    and never closer than ``d``. The ``2d`` test is what keeps it narrow:
    along a wire that heads away from the plane only the plane-nearest
    segment can pass it (segment ``k`` of a wire starting ``g`` off the
    plane passes only if ``k = 0`` and its length exceeds ``g``), and a
    wire running PARALLEL to the plane never does, since grading toward an
    end cannot help a near-singular point that can sit anywhere along it.
    Off by default, so bspline's and SinusoidalGalerkinSolver's axes are
    unchanged by construction: on a coaxial detached deck with a 0.67 m
    mast segment 10 mm above the plane, razor's reversed cross block
    collapsed at eps~ = 1 only to 6e-6 without it.
    """
    basis = ctx.basis
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

    n_basis = basis.n_basis

    nodes_l, t_l, w_l, u_l, segpos = [], [], [], [], []
    for i_g, g in enumerate(seg_idx):
        if not i_g & 255:
            _cancel.poll()
        sl, sr = geom.seg_l[g], geom.seg_r[g]
        h = geom.h[g]
        tang = geom.tangents[g]
        touch_lo = abs(sl[2] - gz) < tol
        touch_hi = abs(sr[2] - gz) < tol
        d_lo, d_hi = abs(sl[2] - gz), abs(sr[2] - gz)
        if touch_lo or touch_hi:
            u, w = _graded_u(h, "lo" if touch_lo else "hi", a_wire, growth, gx, gw)
        elif grade_near_plane and max(d_lo, d_hi) > 2.0 * min(d_lo, d_hi):
            d = min(d_lo, d_hi)
            u, w = _graded_u(
                h, "lo" if d_lo <= d_hi else "hi", max(d, a_wire), growth, gx, gw
            )
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

    # Every segment's nodes are ONE contiguous run (appended per `g` above),
    # so the axis carries a run table — the thing `_support_rows` and
    # `_main_split`'s block indexing used to rediscover by scanning `segof`
    # or F (momwire#912).
    seg_runs = _segment_runs(segof)
    # momwire#919: the coarse (far-block) axis and the near one are the SAME
    # sampling whenever their density knobs agree — which they do today
    # (`_FAR_Q == _NEAR_Q == 4`, `_FAR_GROWTH == _NEAR_GROWTH == 4.0` since
    # #692) — so the caller can hand the dense axis in and skip rebuilding a
    # second copy of F/Fd. On the 48-radial screen that copy is 445 MB dense
    # and ~1 MB as the CSR it is since momwire#1109; the share still earns its
    # lines as the sampling it skips.
    #
    # Guarded on the SAMPLING, not on the flag: the node count and positions
    # are what make the two identical, and they are exactly what a future
    # divergence of the knobs would change. If they ever differ, this falls
    # through to a real build with no edit here.
    if (
        share_from is not None
        and share_from["F_csr"].shape == (n_basis, u_phys.shape[0])
        and np.array_equal(share_from["nodes"], nodes)
    ):
        F = share_from["F_csr"]
        Fd = share_from["Fd_csr"]
        seg_rows = share_from["seg_rows"]
        # The end table too (momwire#1029 phase 3): it is a function of the
        # basis and the geometry only, never of the sampling, and each end's
        # value row is dense in n -- 19 MB per axis at 150 radials, held for
        # the whole fill, twice.
        shared_ends = share_from["ends"]
    else:
        F, Fd, seg_rows = basis.samples(seg_runs, u_phys)
        F, Fd = _as_csr(F), _as_csr(Fd)
        shared_ends = None

    # Signed wire-end table: (point, sign, per-basis value there). σ = −1
    # at a wire's first segment's u = 0 end, +1 at its last segment's
    # u = h end — the by-parts orientation the derivation pinned.
    seg_off = geom.seg_offsets
    if shared_ends is not None:
        ends = shared_ends
    else:
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
                fv = basis.end_values(gseg, u)
                if np.any(fv != 0.0):
                    ends.append((pt, sign, fv))
    return dict(
        nodes=nodes,
        t=t_node,
        w=w_node,
        # `F_csr` / `Fd_csr`, not `F` / `Fd`: the rename is the point
        # (momwire#1109). Every consumer of the samples had to become an
        # explicit edit when they went sparse, and a key that kept its name
        # would have let one through — `ax["F"] * w` is an elementwise fold on
        # an array and a matmul on a `csr_matrix`, with no error either way.
        F_csr=F,
        Fd_csr=Fd,
        ends=ends,
        n_basis=n_basis,
        segof=segof,
        seg_runs=seg_runs,
        seg_rows=seg_rows,
    )


def _segment_runs(segof):
    """`{segment: (start, count)}` — each segment's contiguous node run on
    an axis. Exact because `axis_data` appends one segment's nodes at a
    time; asserted rather than assumed."""
    uniq, start, counts = np.unique(segof, return_index=True, return_counts=True)
    for g, s0, c in zip(uniq, start, counts):
        if not np.all(segof[s0 : s0 + c] == g):
            raise AssertionError(f"segment {g}'s nodes are not one contiguous run")
    return {int(g): (int(s0), int(c)) for g, s0, c in zip(uniq, start, counts)}


def _basis_samples(supp_seg, polys, seg_runs, u_phys):
    """F / Fd — every basis polynomial and its derivative sampled at the
    axis nodes of its support segments, as CSR — and `seg_rows`, the basis
    rows with a LIVE wing on each segment (momwire#912).

    The same terms in the same order as the per-row loop this replaces:
    `F[m, sel] += c·u^p` for p ascending, one live wing at a time, and the
    running sum below reproduces it because no basis row carries the same
    segment in two live wings (asserted). supp_seg rows are zero-padded,
    and a slot is live only if its polynomial is nonzero (the padding
    trap), so the mask is on the POLYNOMIAL, never on the segment id.

    SPARSE, and built from the STRUCTURE (momwire#1029 phase 2 / #1109), the
    way `_fdw_sparse` (momwire#914) already builds `Fd·w`: the (row, node)
    pairs are the live wings' own rectangles, so there is nothing to scan and
    no dense array to scan it out of. The dense pair was `(n_basis, n_nodes)`
    float64 with at most `degree + 1` nonzeros per COLUMN — 4.13 GB live on
    the 150-radial screen for ~93 k numbers, and the floor under that fill's
    8.55 GB peak. `csr_array`, not `csr_matrix`: the matrix class reads `*`
    as matmul, and the weight folds downstream are elementwise.

    Explicit zeros are KEPT (a live wing whose polynomial samples to 0 at a
    node stays stored), which is what makes `_support_rows`' pattern reading
    an exact superset of the live rows and keeps a NaN-poisoned row in.
    """
    n_basis, _n_wings, n_p = polys.shape
    n_nodes = u_phys.shape[0]
    live = np.any(polys != 0.0, axis=2)
    m_idx, a_idx = np.nonzero(live)
    segs = supp_seg[m_idx, a_idx]
    on = np.array([int(g) in seg_runs for g in segs], dtype=bool)
    m_idx, a_idx, segs = m_idx[on], a_idx[on], segs[on]
    seg_rows: dict[int, np.ndarray] = {}
    if not m_idx.size:
        empty = _sp.csr_array((n_basis, n_nodes), dtype=float)
        return empty, empty.copy(), seg_rows
    pairs = np.stack([m_idx, segs], axis=1)
    if np.unique(pairs, axis=0).shape[0] != pairs.shape[0]:
        raise AssertionError("a basis row carries one segment in two live wings")
    start = np.array([seg_runs[int(g)][0] for g in segs], dtype=np.int64)
    cnt = np.array([seg_runs[int(g)][1] for g in segs], dtype=np.int64)
    row = np.repeat(m_idx, cnt)
    node = np.repeat(start - (np.cumsum(cnt) - cnt), cnt) + np.arange(int(cnt.sum()))
    u = u_phys[node]
    coef = polys[m_idx, a_idx]  # (L, n_p)
    fv = np.zeros(row.size, dtype=float)
    fdv = np.zeros(row.size, dtype=float)
    for p in range(n_p):
        c = np.repeat(coef[:, p], cnt)
        fv += c * u**p
        if p >= 1:
            fdv += (p * c) * u ** (p - 1)
    shape = (n_basis, n_nodes)
    F = _sp.csr_array((fv, (row, node)), shape=shape)
    Fd = _sp.csr_array((fdv, (row, node)), shape=shape)
    order = np.argsort(segs, kind="stable")
    keys, groups = _group_sorted(segs[order], m_idx[order])
    for g, rows in zip(keys, groups):
        seg_rows[int(g)] = np.sort(rows)
    return F, Fd, seg_rows


def _as_csr(M):
    """The axis's F/Fd in the one spelling every consumer below reads."""
    return M if _sp.issparse(M) else _sp.csr_array(np.asarray(M))


def _in_rows(rows, idx):
    """`(sel, pos)` — the positions WITHIN `idx` of its entries that are in
    the sorted `rows`, and where those entries sit in `rows`.

    The whole of the row restriction's bookkeeping (momwire#1029 phase 2):
    `idx[sel]` is what survives and `out[pos]` is where it goes, so a
    restricted block is a gather on one factor and a scatter on the other and
    never a second evaluation of anything.
    """
    idx = np.asarray(idx, dtype=np.int64)
    if rows.size == 0 or idx.size == 0:
        e = np.zeros(0, dtype=np.int64)
        return e, e
    p = np.searchsorted(rows, idx)
    hit = (p < rows.size) & (rows[np.minimum(p, rows.size - 1)] == idx)
    sel = np.flatnonzero(hit)
    return sel, p[sel]


def _scale_cols(M, v):
    """`M * v[None, :]` for a CSR `M` — the column weighting folded into the
    stored values, entry for entry, with no fill-in and no dense broadcast."""
    out = M.copy()
    out.data = out.data * v[out.indices]
    return out


def _group_sorted(keys, values):
    """Split `values` by runs of equal `keys` (keys already sorted)."""
    if keys.size == 0:
        return [], []
    cuts = np.flatnonzero(np.diff(keys)) + 1
    return keys[np.concatenate(([0], cuts))], np.split(values, cuts)


def _end_values(supp_seg, polys, live, gseg, u, d):
    """Per-basis value at arclength `u` of segment `gseg`, summed over the
    live wings that carry it — the wire-end table's entry, in the loop
    form's summation order (p ascending within a wing)."""
    n_basis = polys.shape[0]
    fv = np.zeros(n_basis)
    hit = (supp_seg == gseg) & live
    for a_ in range(supp_seg.shape[1]):
        rows = np.flatnonzero(hit[:, a_])
        if rows.size == 0:
            continue
        acc = np.zeros(rows.size)
        for p in range(d + 1):
            acc = acc + polys[rows, a_, p] * u**p
        fv[rows] += acc
    return fv


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
    nodes, tl, wl, f_rows, f_cols, segof, ends = [], [], [], [], [], [], []
    off = 0
    # Every row's one-hot is a read-only WINDOW of one shared buffer
    # (momwire#1173 design B): `hot[n_basis - m : 2 n_basis - m]` is 1 at m
    # and 0 elsewhere, the same float64 values a fresh `zeros(n_basis)` with
    # a 1 at m holds, so every reader (`flatnonzero`, `fv[nz]`, the corner's
    # outer products, `_end_live_rows`) sees the same array. A vector per row
    # was n_rows · n_basis floats (63 MB at hub_deck(16) x16).
    hot = np.zeros(2 * n_basis + 1)
    hot[n_basis] = 1.0
    hot.setflags(write=False)
    for m, pts, t, w, seg, c_before, c_after in rows:
        pts = np.asarray(pts, dtype=float)
        q = pts.shape[0]
        nodes.append(pts)
        tl.append(np.asarray(t, dtype=float))
        wl.append(np.asarray(w, dtype=float))
        # The pulse as its own nonzeros (momwire#1109): row m carries 1 on its
        # own q points and nothing anywhere else, which is the whole of F.
        f_rows.append(np.full(q, int(m), dtype=np.int64))
        f_cols.append(np.arange(off, off + q, dtype=np.int64))
        off += q
        segof.append(np.asarray(seg, dtype=np.int64))
        e = hot[n_basis - int(m) : 2 * n_basis - int(m)]
        # The endpoints are COPIED (momwire#1173 design B): a caller's view
        # of one row of a whole-geometry array (razor's per-call centroids)
        # would otherwise hold that array for the axis's lifetime — one per
        # row, N² floats at 189 MB on hub_deck(16) x16. The same three floats.
        ends.append((np.array(c_before, dtype=float), -1.0, e))
        ends.append((np.array(c_after, dtype=float), +1.0, e))
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
        F_csr=_sp.csr_array(
            (
                np.ones(n_pts),
                (
                    np.concatenate(f_rows) if f_rows else np.zeros(0, dtype=np.int64),
                    np.concatenate(f_cols) if f_cols else np.zeros(0, dtype=np.int64),
                ),
            ),
            shape=(n_basis, n_pts),
        ),
        Fd_csr=_sp.csr_array((n_basis, n_pts), dtype=float),
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


@_cancelable  # razor's crossing-node charges call it outside any fill entry
def _tables(ctx, eps_t, k_p, rho, z, zp, rtol, memo=None, group_labels=None):
    """Designed tables with the deck's wire radius folded in, z relative
    to the interface. `memo` extends the exact-triple dedup across calls
    (one fill = one memo; ε̃, k₂, rtol fixed for its lifetime);
    `group_labels` lets one call stand in for several (`_end_tables`)."""
    a_wire = float(ctx.a_wire)
    return _near_interface.radius_tables(
        eps_t, k_p, rho, z, zp, a_wire, rtol=rtol, memo=memo, group_labels=group_labels
    )


# The end loops' kernel tables (momwire#1168 U4) are evaluated for a SPAN of
# consecutive ends in one `_tables` call, the span holding at most this many
# (end, node) pairs. One call per end was 2,823 calls at razor hub_deck(16) x8
# for ~5.4 k fresh triples — nearly all memo hits, so the cost was per-call
# overhead. The budget bounds memory: a call's traced peak is ~0.3 kB per pair
# (the stacked inputs and fold, `_unique_rows`' sort arrays, the (n, 6) gather
# and its transposed copy), and at x8 an unbounded span is one call of 3.8 M
# pairs peaking at 864 MiB — the audit's 60-110 MB at 500 k pairs is the same
# rate. 64 k pairs measured 18.3 MiB at its largest call, well under the main
# sandwich's 64 MiB chunk (`_MAIN_CHUNK_BYTES`), for 121 calls at x8 where the
# per-call overhead no longer shows. Spans never move a bit (`_end_tables`),
# so this is a memory choice only.
_END_BATCH_PAIRS = 1 << 16
# TEST-ONLY. False drops the per-end column labels and so reproduces the
# naive batching the #1168 audit measured moving Z; the bit-identity gate's
# negative control flips it to prove the gate can fail.
_END_LABELS = True


def _end_groups(n_ends, n_nodes, memo):
    """`[start, stop)` spans of consecutive ends, each at most
    `_END_BATCH_PAIRS` pairs against an `n_nodes` line (one end at least).
    Without a memo every span is one end: a call per end dedups nothing
    across ends, which one call over several cannot reproduce."""
    per = 1 if memo is None else max(1, _END_BATCH_PAIRS // max(1, n_nodes))
    return [(g0, min(n_ends, g0 + per)) for g0 in range(0, n_ends, per)]


def _end_tables(ctx, eps_t, k_p, ends, n_nodes, memo, args):
    """Yield `(pt, sign, fv, te)` per end of `ends`, in order, with `te` the V
    and W tables `_tables(ctx, eps_t, k_p, *args(pt), _CROSS_RTOL, memo=memo)`
    returns for that end — but served by one call per `_end_groups` span.

    `args(pt)` is the one-end call's (rho, z, zp), broadcast to `n_nodes`.
    The span's rows are stacked end by end, so the flat asked order is the
    per-end calls' asked orders concatenated, and each end's rows carry that
    end's position as its `group_labels` label. `designed_tables` documents
    why that is the per-end sequence to the bit: the same hits, each fresh
    row evaluated in the first asking end's column, the same memo contents.
    A span of one end passes no labels — it IS the per-end call.

    Over a `ProductMemo` holding a main sandwich (momwire#1173 design B) an
    end whose every asked point is a stored product row is served by a
    gather from the product's value block (`_fast_end_rows` says which, on
    the asked floats themselves), and the span's remaining ends go through
    ONE `_tables` call labelled with their span positions. That is the span
    call to the bit:

      * a fast end's points are all memo hits in the span call too (the
        product IS the memo's content for them), each returning the stored
        floats the gather reads;
      * a fast end asks no fresh row, so every fresh row's first asker — its
        label — and its (label, ρ) column are the same in the call without
        the fast ends, the fresh rows keep their relative first-appearance
        order, and the memo receives the same rows with the same values in
        the same order;
      * labels are compared only for equality (`_column_blocks`), so the
        span positions label the slow ends' columns exactly as 0..k-1 did.
    """
    product = getattr(memo, "product", None)
    if product is not None and _PRODUCT_ENDS and product.fast is not None:
        yield from _end_tables_product(ctx, eps_t, k_p, ends, n_nodes, memo, args)
        return
    for g0, g1 in _end_groups(len(ends), n_nodes, memo):
        span = ends[g0:g1]
        cols = [np.broadcast_arrays(*args(pt)) for pt, _sign, _fv in span]
        rho, z, zp = (np.stack([c[i] for c in cols]) for i in range(3))
        del cols
        labels = None
        if len(span) > 1 and _END_LABELS:
            labels = np.broadcast_to(np.arange(len(span))[:, None], rho.shape)
        te = _tables(
            ctx, eps_t, k_p, rho, z, zp, _CROSS_RTOL, memo=memo, group_labels=labels
        )
        for i, (pt, sign, fv) in enumerate(span):
            yield pt, sign, fv, {"V": te["V"][i], "W": te["W"][i]}


def _end_tables_product(
    ctx, eps_t, k_p, ends, n_nodes, memo, args, *, classes=None, fast_te=None
):
    """`_end_tables` over a `ProductMemo` holding a product: see there for
    why the yields are the span call's to the bit. The fast ends' V and W
    are read from the product's value block by `kernels` position: since
    momwire#1173 design C that block is the tiles' (rows, 2) V/W store in
    row order, holding copies of the very floats the evaluation returned.

    `classes` (design C phase 2, `_FusedEnds`) is each end's
    `_fast_end_desc`, decided before the tiles ran, and `fast_te(i)` serves
    end i's `te` in place of the value block (None: the caller has no use
    for it). The spans, which ends are slow and each slow call's rows and
    labels are the same either way — only where a fast end's floats come
    from changes, and they are copies of the same evaluated floats."""
    product = memo.product
    fast = product.fast
    a_wire = float(ctx.a_wire)
    if classes is None:
        # Every end's desc first (`_classify_ends`, batched): deciding one
        # touches nothing the spans below read or write.
        classes, on_node = _classify_ends(fast, a_wire, ends, args)
        _ROUTES["ends_line_on_node"] += int(sum(on_node))
    for g0, g1 in _end_groups(len(ends), n_nodes, memo):
        span = ends[g0:g1]
        got = [None] * len(span)
        slow, cols = [], []
        for i, (pt, _sign, _fv) in enumerate(span):
            desc = classes[g0 + i]
            c = np.broadcast_arrays(*args(pt)) if desc is None else None
            if desc is None:
                slow.append(i)
                cols.append(c)
                continue
            _ROUTES["ends_fast_" + desc[0]] += 1
            if fast_te is not None:
                got[i] = fast_te(g0 + i)
                continue
            vrow = _fast_desc_rows(fast, desc)
            if _PRODUCT_NEG_CONTROL == "row":
                vrow = (vrow + 1) % product.n_rows  # TEST-ONLY: the wrong row
            got[i] = {
                "V": product.values_of(vrow, "V"),
                "W": product.values_of(vrow, "W"),
            }
        _ROUTES["ends_fast"] += len(span) - len(slow)
        _ROUTES["ends_slow"] += len(slow)
        if slow:
            _ROUTES["end_slow_calls"] += 1
            rho, z, zp = (np.stack([c[j] for c in cols]) for j in range(3))
            del cols
            labels = None
            if len(span) > 1 and _END_LABELS:
                labels = np.broadcast_to(np.asarray(slow)[:, None], rho.shape)
            te = _tables(
                ctx, eps_t, k_p, rho, z, zp, _CROSS_RTOL, memo=memo, group_labels=labels
            )
            for j, i in enumerate(slow):
                got[i] = {"V": te["V"][j], "W": te["W"][j]}
            del te
        for i, (pt, sign, fv) in enumerate(span):
            yield pt, sign, fv, got[i]
            got[i] = None


def _fast_end_rows(fast, a_wire, pt, rho, z, zp):
    """The product value row of every point one end asks, or None when the
    end is not wholly a set of product rows (`_fast_end_desc`, counted)."""
    desc = _fast_end_desc(fast, a_wire, pt, rho, z, zp)
    if desc is None:
        return None
    _ROUTES["ends_fast_" + desc[0]] += 1
    return _fast_desc_rows(fast, desc)


def _fast_desc_rows(fast, desc):
    """The product rows a `_fast_end_desc` names, one per asked point: a
    "grouped" end's are group g's z row `zl` against every line key, a
    "line" end's each grouped node's (z, the end's key in its group)."""
    if desc[0] == "grouped":
        _kind, g, zl = desc
        return fast.rowflat[fast.off[g] + zl * fast.nk[g] + fast.kl_rank[g]]
    kg = desc[1]
    grk = fast.grank
    return fast.rowflat[fast.off[grk] + fast.zl_rank * fast.nk[grk] + kg[grk]]


def _fast_desc_rows_batch(fast, descs):
    """`np.stack([_fast_desc_rows(fast, d) for d in descs])`, one gather per
    shape: a run of one shape (every desc of an end loop has the same
    length) is the same index expression with the ends on a new axis."""
    if not descs:
        return np.zeros((0, 0), dtype=np.int64)
    kinds = {d[0] for d in descs}
    if len(kinds) > 1:
        return np.stack([_fast_desc_rows(fast, d) for d in descs])
    if descs[0][0] == "grouped":
        g = np.array([d[1] for d in descs], dtype=np.int64)
        zl = np.array([d[2] for d in descs], dtype=np.int64)
        base = fast.off[g] + zl * fast.nk[g]
        return fast.rowflat[base[:, None] + fast.kl_rank[g]]
    grk = fast.grank
    base = fast.off[grk] + fast.zl_rank * fast.nk[grk]
    KG = np.stack([d[1] for d in descs])
    return fast.rowflat[base[None, :] + KG[:, grk]]


def _fast_end_desc(fast, a_wire, pt, rho, z, zp):
    """Which product rows one end asks — `("grouped", g, zl)` or
    `("line", kg)` (`_fast_desc_rows` lists them) — or None when the end is
    not wholly a set of product rows of the two shapes checked here.

    Checked on the ASKED floats, after `_on_plane_side` (never assumed from
    geometry). Equality is float `==`, the memo's key equality (−0.0 == 0.0,
    NaN never), so a point that passes is a memo hit on exactly the row
    named, and the gather reads the floats the lookup would return. Failing
    proves nothing either way: that end takes the lookup path.

    * "grouped" — the end stands where a grouped node would: its slot value
      is one float, held by group g's z factor; the line slot is the line's
      own z, node by node; and ρ is group g's stored raw line, node by node
      (g found by the end's (x, y); the compare decides). The memo would key
      each point on `radius_fold(ρ)` of that very float, which is the line's
      stored key.
    * "line" — the end stands where a line node would: its slot value is one
      float; the grouped slot is the grouped nodes' own z, node by node; ρ is
      one float within each group; and (radius_fold(ρ_g), that float) is a
      key of group g, for every g.
    """
    gs = fast.grouped_slot
    gv, lv = (z, zp) if gs == "z" else (zp, z)
    n_l, n_g = fast.line_z.size, fast.grouped_z.size
    if gv.size == n_l:
        g = fast.gdict.get((float(pt[0]), float(pt[1])))
        if (
            g is not None
            and np.all(gv == gv[0])
            and np.array_equal(lv, fast.line_z)
            and np.array_equal(rho, _raw_row(fast, g))
        ):
            zl = fast.zmap[g].get(float(gv[0]))
            if zl is not None:
                return ("grouped", g, zl)
    if lv.size == n_g and np.all(lv == lv[0]) and np.array_equal(gv, fast.grouped_z):
        rep = rho[fast.gfirst]
        if np.array_equal(rho, rep[fast.grank]):
            r = _near_interface.radius_fold(rep, a_wire)
            # An end standing ON a line node n asks exactly n's keys: when
            # (r[g], lv[0]) == (line[g, n], line_z[n]) in every group, group
            # g's local key is n's own, `kl_rank[g, n]` -- the dict's answer,
            # since a key is an exact-`==` class. Candidates by the end's
            # (x, y); the compare decides, and a miss takes the search.
            l0 = float(lv[0])
            for n in fast.line_xy.get((float(pt[0]), float(pt[1])), ()):
                if fast.line_z[n] == l0 and np.array_equal(r, fast.line[:, n]):
                    _ROUTES["ends_line_on_node"] += 1
                    return ("line", fast.kl_rank[:, n].copy(), n)
            # Group g's local key of (r[g], lv[0]), every g at once: the
            # global key of the pair, then its place among g's own keys.
            k = fast.keys.ids(r, np.full(r.shape, lv[0]))
            if np.any(k < 0):
                return None
            n_key = fast.keys.n_key
            j = fast.gkey.local(np.arange(r.size, dtype=np.int64) * n_key + k)
            if j is None:
                return None
            return ("line", j)
    return None


# `_classify_ends` decides this many (end, line node) pairs at once. The
# budget bounds the batch's transients (ρ, its compare against the groups'
# representatives, the folded representatives: ~25 B a pair, ~25 MiB here);
# batching never moves a desc (`_classify_ends`), so this is a memory choice
# only.
_END_CLASSIFY_PAIRS = 1 << 20
# TEST-ONLY. False classifies one end at a time through `_fast_end_desc`,
# the reference the batches are gated against (tests/test_end_classify_1224.py).
_END_CLASSIFY_BATCHED = True


def _classify_ends(fast, a_wire, ends, args):
    """`(descs, on_node)`: each end's `_fast_end_desc` over the floats
    `args(pt)` asks, and whether it was found as a "line" end standing on a
    line node (the `ends_line_on_node` count, which the CALLER adds as it
    walks the ends, so an early stop counts what the one-end loop counted).

    The same descs as the one-end loop: `_EndArgs.batch` hands each end the
    floats `args(pt)` does, the per-end tests are that function's on those
    floats, and only their spelling changes -- a test on the line's own
    arrays (the same for every end of the loop) is made once, a test on an
    end's ρ row is a row of one 2-D compare, and the representatives' fold
    is `radius_fold` of the same floats elementwise. The per-end Python
    that is left is the dict lookups and one short compare per candidate
    node. At razor's inverted-L x8 the one-end loop was ~0.3 s of a 3.2 s
    crossing fill, its per-end ρ, broadcasts and whole-line compares."""
    if not ends:
        return [], []
    if not _END_CLASSIFY_BATCHED or not isinstance(args, _EndArgs):
        return _classify_ends_one_by_one(fast, a_wire, ends, args)
    n = args.nodes.shape[0]
    by_nodes = _END_CLASSIFY_BY_NODES and _args_on_plan_nodes(fast, args)
    # By the nodes, no ρ row is formed here: the line shape forms its
    # groups' first nodes' entries for the ends that need them (an end off
    # every line node), and the grouped shape needs none.
    cols = np.zeros(0, np.intp) if by_nodes else None
    per = max(1, _END_CLASSIFY_PAIRS // max(1, n if cols is None else fast.gfirst.size))
    descs, on_node = [], []
    for e0 in range(0, len(ends), per):
        span = ends[e0 : e0 + per]
        pts = np.array([pt for pt, _sign, _fv in span], dtype=float).reshape(-1, 3)
        try:
            rho, end = args.batch(pts, cols)
        except ValueError:
            # A refusal: the one-end loop raises it, at the first end that
            # earns it and in its own words.
            return _classify_ends_one_by_one(fast, a_wire, ends, args)
        d, o = _classify_batch(
            fast, a_wire, pts, rho, end, args, n=n, by_nodes=by_nodes
        )
        descs += d
        on_node += o
    return descs, on_node


# Decide the ρ-row tests of `_classify_batch` from the nodes rather than per
# end (`_args_on_plan_nodes`); False is the per-end compare, the reference.
_END_CLASSIFY_BY_NODES = True


def _args_on_plan_nodes(fast, args):
    """Whether the loop's ρ rows are the plan's by construction, so the two
    per-end ρ compares of `_classify_batch` hold without forming the rows.

    An end's row is `hypot` of its (x, y) minus the loop's nodes' (either
    operand order, which only negates the differences). The "grouped" test
    compares it with group g's raw line, `hypot` of the group's (x, y)
    minus the LINE nodes, for the g whose (x, y) equals the end's under
    `==` (`gdict`); the "line" test compares each node's entry with its
    group's first node's. Equal operands under `==` differ at most in the
    sign of a zero, which a difference carries only as a sign and `hypot`
    drops, so both tests hold whenever the loop's nodes ARE the plan's
    nodes on that side, (x, y) under `==`: the line nodes for the grouped
    shape, and the grouped nodes (each at its group's (x, y)) for the line
    shape. That is checked here once per loop instead of once per end."""
    xy = args.nodes[:, :2]
    n = xy.shape[0]
    if n == fast.line_z.size and not np.array_equal(xy, fast.line_nodes[:, :2]):
        return False
    if n == fast.grouped_z.size and not np.array_equal(xy, fast.gxy[fast.grank]):
        return False
    return True


def _classify_ends_one_by_one(fast, a_wire, ends, args):
    """`_classify_ends` by `_fast_end_desc`, one end at a time (the
    reference), the on-node flags read off the route counter it bumps."""
    descs, on_node = [], []
    for pt, _sign, _fv in ends:
        before = _ROUTES["ends_line_on_node"]
        descs.append(_fast_end_desc(fast, a_wire, pt, *np.broadcast_arrays(*args(pt))))
        on_node.append(_ROUTES["ends_line_on_node"] != before)
        _ROUTES["ends_line_on_node"] = before
    return descs, on_node


def _classify_batch(fast, a_wire, pts, rho, end, args, *, n=None, by_nodes=False):
    """`_fast_end_desc` of the ends `pts`, from `_EndArgs.batch`'s `(rho,
    end)`; see `_classify_ends`. The comments name the one-end test each
    line stands for, with `gv` / `lv` as that function spells them.

    `by_nodes` (`_args_on_plan_nodes` held for this loop): no ρ row is
    handed in, and the ρ tests hold by that check rather than by a compare
    per end."""
    _cancel.poll()
    E = pts.shape[0]
    n = rho.shape[1] if n is None else n
    lz = args.line_z()
    # The end's constant fills the grouped slot (gv) or the line slot (lv).
    end_in_gv = (args.end_side == "above") == (fast.grouped_slot == "z")
    descs, on_node = [None] * E, [False] * E
    todo = np.ones(E, dtype=bool)
    xy = [(float(x), float(y)) for x, y in zip(pts[:, 0].tolist(), pts[:, 1].tolist())]
    if n == fast.line_z.size:  # the "grouped" shape: gv.size == n_l
        if end_in_gv:
            gv_flat = end == end  # np.all(gv == gv[0]) of a constant gv
            lv_line = np.full(E, bool(np.array_equal(lz, fast.line_z)))
        else:
            gv_flat = np.full(E, bool(np.all(lz == lz[0])))
            lv_line = np.all(fast.line_z[None, :] == end[:, None], axis=1)
        raw = {}
        for e in np.flatnonzero(gv_flat & lv_line).tolist():
            g = fast.gdict.get(xy[e])
            if g is None:
                continue
            if not by_nodes:
                if g not in raw:
                    raw[g] = _raw_row(fast, g)
                if not np.array_equal(rho[e], raw[g]):
                    continue
            gv0 = end[e] if end_in_gv else lz[0]
            zl = fast.zmap[g].get(float(gv0))
            if zl is not None:
                descs[e] = ("grouped", g, zl)
                todo[e] = False
        del raw
    if n == fast.grouped_z.size:  # the "line" shape: lv.size == n_g
        if end_in_gv:
            lv_flat = np.full(E, bool(np.all(lz == lz[0])))
            gv_grouped = np.all(fast.grouped_z[None, :] == end[:, None], axis=1)
        else:
            lv_flat = end == end
            gv_grouped = np.full(E, bool(np.array_equal(lz, fast.grouped_z)))
        idx = np.flatnonzero(todo & lv_flat & gv_grouped)
        if idx.size and by_nodes:
            # An end at a line node's (x, y) and z asks that node's own keys:
            # its ρ to each group's (x, y) is the plan's line entry to the bit
            # (`_args_on_plan_nodes`), so the node is found by the dict and
            # the z alone. The rest search the keys from their ρ, formed for
            # them only.
            off_node = []
            for i_e, e in enumerate(idx.tolist()):
                if not i_e & 255:
                    _cancel.poll()
                lv0 = lz[0] if end_in_gv else end[e]
                l0 = float(lv0)
                for nn in fast.line_xy.get(xy[e], ()):
                    if fast.line_z[nn] == l0:
                        descs[e] = ("line", fast.kl_rank[:, nn].copy(), nn)
                        on_node[e] = True
                        break
                else:
                    off_node.append(e)
            if off_node:
                rho_g, _end = args.batch(pts[off_node], fast.gfirst)
                r_all = _near_interface.radius_fold(rho_g, a_wire)
                lv_all = np.array(
                    [lz[0] if end_in_gv else end[e] for e in off_node], dtype=float
                )
                # Every off-node end's keys in one lookup (the same pairs).
                k_all = fast.keys.ids(
                    r_all, np.broadcast_to(lv_all[:, None], r_all.shape)
                )
                for j, e in enumerate(off_node):
                    r = r_all[j]
                    k = k_all[j]
                    if np.any(k < 0):
                        continue
                    n_key = fast.keys.n_key
                    jj = fast.gkey.local(np.arange(r.size, dtype=np.int64) * n_key + k)
                    if jj is not None:
                        descs[e] = ("line", jj)
        elif idx.size:
            rows = rho[idx]
            rep = rows[:, fast.gfirst]
            ok = np.all(rows == rep[:, fast.grank], axis=1)  # rho == rep[grank]
            del rows
            r_all = _near_interface.radius_fold(rep, a_wire)
            for j, e in enumerate(idx.tolist()):
                if not ok[j]:
                    continue
                r = r_all[j]
                lv0 = lz[0] if end_in_gv else end[e]
                l0 = float(lv0)
                for nn in fast.line_xy.get(xy[e], ()):
                    if fast.line_z[nn] == l0 and np.array_equal(r, fast.line[:, nn]):
                        descs[e] = ("line", fast.kl_rank[:, nn].copy(), nn)
                        on_node[e] = True
                        break
                else:
                    k = fast.keys.ids(r, np.full(r.shape, lv0))
                    if np.any(k < 0):
                        continue
                    n_key = fast.keys.n_key
                    jj = fast.gkey.local(np.arange(r.size, dtype=np.int64) * n_key + k)
                    if jj is not None:
                        descs[e] = ("line", jj)
    return descs, on_node


def _direct_coords(specs, gz):
    """ρ, z and z′ for a list of direct `(A, B, iA, iB)` specs — the node
    pairs `iA` of above axis A against `iB` of below axis B, both z relative
    to the plane — shared by `_direct_group` (one group of the split's direct
    blocks) and `_main_sandwich` (the one-spec case: every node of both axes)
    since momwire#1168 U5.

    Returns `(rho, zAs, zBs, shapes)`: ONE flat ρ buffer holding each spec's
    (|iA|, |iB|) grid in C order, spec after spec, and per spec its two z
    columns and its shape. The z columns stay columns — `_direct_group`
    broadcasts them into its flat batch, while `_main_sandwich` broadcasts or
    chunks them without ever materialising the grid.

    A shape unification only: it evaluates nothing and groups nothing. Each
    caller keeps its own `_tables` call grouping, because bspline's memo
    carry-over across groups is not bit-identical if regrouped (momwire#1126).
    """
    shapes = [(iA.size, iB.size) for _A, _B, iA, iB in specs]
    rho = np.empty(int(sum(m * n for m, n in shapes)), dtype=float)
    zAs, zBs = [], []
    off = 0
    for (AX, BX, iA, iB), shp in zip(specs, shapes):
        pa, pb = AX["nodes"][iA], BX["nodes"][iB]
        sl = slice(off, off + shp[0] * shp[1])
        np.hypot(
            pa[:, 0][:, None] - pb[:, 0][None, :],
            pa[:, 1][:, None] - pb[:, 1][None, :],
            out=rho[sl].reshape(shp),
        )
        zAs.append(pa[:, 2] - gz)
        zBs.append(pb[:, 2] - gz)
        off = sl.stop
    return rho, zAs, zBs, shapes


def _plan_sheets(ctx, eps_t, k_p, gz, memo, specs):
    """Decide the fill's plane sheets ONCE (momwire#1173 Design E phase 2):
    `memo.sheet_plan` from the node pairs of `specs` (`(A, B, iA, iB)`, as
    `_direct_coords` takes them), unless the fill already has a plan.

    Called before the fill's first evaluation -- the main sandwich over the
    whole axes, or the split route's direct batch -- so every later call of
    the fill (tiles, chunks, end spans, ACA samples) reads the same plan, and
    how the fill cuts its rows cannot change which rows a sheet serves. No
    memo, no plan: a call without a fill runs exact."""
    if memo is None or memo.sheet_plan is not None:
        return
    pairs = []
    for AX, BX, iA, iB in specs:
        pa = np.array(AX["nodes"][iA], dtype=float)
        pb = np.array(BX["nodes"][iB], dtype=float)
        pa[:, 2] -= gz
        pb[:, 2] -= gz
        pairs.append((pa, pb))
    memo.sheet_plan = _near_interface.sheet_plan(eps_t, k_p, pairs, ctx.a_wire)


def _plan_point_sheets(ctx, eps_t, k_p, gz, memo, obs_pts, src_nodes, observers_above):
    """`_plan_sheets` for a point-observer block (momwire#1224 option A): the
    fill's `SheetPlan` over its (observers × source nodes) pairs, decided
    ONCE before the block's first evaluation and carried on its memo, so the
    six keys (`designed_rows` / `_tables` through the memo) and the point
    family (`point_designed_rows` / `point_radius_tables`, handed
    `memo.sheet_plan`) read the same plan in every chunk and in the ends
    loop. The observers sit in the z slot when they are above, in z′ when
    below -- `sheet_plan` takes (above, below) node pairs."""
    if memo is None or memo.sheet_plan is not None:
        return
    obs = np.array(obs_pts, dtype=float)
    nodes = np.array(src_nodes, dtype=float)
    obs[:, 2] -= gz
    nodes[:, 2] -= gz
    pair = (obs, nodes) if observers_above else (nodes, obs)
    memo.sheet_plan = _near_interface.sheet_plan(eps_t, k_p, [pair], ctx.a_wire)


def _main_sandwich(ctx, A, B, eps_t, k_p, c1, gz, memo=None, support=None, ends=None):
    """The M + SW + SQ sandwich over (above axis A × below axis B), whole-axis.

    Split out of `cross_complete_block` so the REVERSED block (momwire#813)
    can share it: the designed tables accept only z ≥ 0 ≥ z′, so the above
    axis sits in the `z` slot whichever ROLE it plays, and the reversed main
    sandwich is this same product transposed — measured exact to 3e-16 at
    ε̃ = 1 and at soil A, which is what says no kernel swap is needed here
    (`scratch/813-reversed-block/probe4_localise.py`).

    The contraction is `_sandwich_dense`'s — the split route's near-block
    product, here over every node of both axes — so the two trunks share ONE
    spelling of the five terms. This used to densify the four weight matrices
    to (n_basis, n_nodes) and run six full GEMMs: O(n_basis · n_nodes²) and
    mostly zeros, since a path-tested axis carries one nonzero per node and an
    identically-zero `Fd` (two of the six products were all zeros). At razor's
    hub_deck(16) x8 (N = 1402) that was 15 s of GEMM and a 346 MB complex
    temporary per product, for 38 s and 2.6 GB of fill. The sparse form
    visits only the stored weights; see `_sandwich_dense` for why the restricted
    product is the same entries.

    The tables come from the PRODUCT route (`_product_route`, momwire#1173
    design B) when the node geometry factorises and the memo is fresh, and
    from the grid dedup (`_tables` / `_chunked_tables`) otherwise. The grid
    route hands `designed_*` its rows in one call; the product route hands
    the same rows in tiles of whole exact-ρ columns (design C,
    `_ProductTiles`), which is the same columns with the same members and
    so the same values. Both serve each table entry as that value copied,
    so the contraction below cannot tell them apart; the product route's
    chunks are column SETS in tile order, which `_streamed_sandwich`
    contracts row by row exactly as it contracts ranges.
    """
    k2sq = k_p * k_p
    nA, nB = A["nodes"].shape[0], B["nodes"].shape[0]
    iA = np.arange(nA)
    iB = np.arange(nB)
    _plan_sheets(ctx, eps_t, k_p, gz, memo, [(A, B, iA, iB)])
    step = max(1, _MAIN_CHUNK_BYTES // (_MAIN_BYTES_PER_PAIR * max(1, nA)))
    tables = _product_route(ctx, eps_t, k_p, A, B, gz, step, memo, ends=ends)
    if tables is not None:
        if ends is not None and ends.streaming:
            # Streamed (`_FusedEnds`): the contracted rows go to the ends'
            # `sink`, which finishes each unit (`*= c1` included) and folds
            # it; no block is formed here.
            _sandwich_dense(A, B, iA, iB, tables, k2sq, support=support, sink=ends.sink)
            return None
        t = _sandwich_dense(A, B, iA, iB, tables, k2sq, support=support)
        t *= c1
        return t
    # The one-spec case of `_direct_coords` — every node of both axes.
    rho, (zA,), (zB,), _shapes = _direct_coords([(A, B, iA, iB)], gz)
    rho = rho.reshape(nA, nB)
    if nB <= step:
        z = np.broadcast_to(zA[:, None], rho.shape)
        zp = np.broadcast_to(zB[None, :], rho.shape)
        tables = _tables(ctx, eps_t, k_p, rho, z, zp, _CROSS_RTOL, memo=memo)
    else:
        cols = [slice(b0, min(nB, b0 + step)) for b0 in range(0, nB, step)]
        tables = _chunked_tables(ctx, eps_t, k_p, rho, zA, zB, cols, memo)
    # momwire#956 — the exact spelling of the transmitted dyad tested along a
    # wire of ANY orientation (antennaknobs scratch/956-derivation):
    #   E^V  = c1 [ k²V ẑ − ∇W + ∇(−∂z′V) ]      E^Hx = c1 [ U x̂ + ∂xW ẑ + ∇(∂xV) ]
    # By parts on both sides the ẑẑ kernel is k²V + ∂z′W — the W derivative on
    # the SOURCE coordinate — with both W cross terms on full charges and the
    # two W end terms (SW on the source ends, TW on the test ends) that the
    # by-parts leave. The former k²V − ∂zW was the by-parted form of the
    # test-side W term on a VERTICAL test, so s_w2 counted it twice there
    # (the +2 Ω rise residual of #956) and it was wrong on a leaning member.
    # `_sandwich_dense` carries those five terms in this order.
    t = _sandwich_dense(A, B, iA, iB, tables, k2sq, support=support)
    t *= c1
    return t


def _chunked_tables(ctx, eps_t, k_p, rho, zA, zB, cols, memo):
    """`_tables` over the (above × below) grid, served as `(cols, K)` column
    chunks for `_sandwich_dense` — the SAME BITS as one call over the grid.

    Two things could move a bit when the grid is cut, and neither does here:

    * WHICH RULE serves a triple. The column route groups a call's fresh
      triples by exact ρ, and a member's value depends on its column's
      smallest z − z′ (`six_columns`: 7.9e-16 between groupings). Calling
      `_tables` per chunk would split those columns — on a vertical above
      member every chunk carries every ρ. So the evaluation stays ONE call:
      pass 1 dedups each chunk's folded triples, merges them into the grid's
      unique rows in the grid's own first-appearance order (each chunk's
      first occurrences mapped to their flat grid index), and hands that list
      to `designed_tables` once. A list of distinct rows dedups to itself, so
      that call sees the same fresh triples, in the same order and grouping,
      that the whole-grid call saw, and fills `memo` with the same values.
    * The contraction. The chunks are COLUMN slices of the below axis, and a
      sparse @ dense product forms each output column from its own column
      alone, so `P @ K[:, cols]` is those columns of `P @ K` bit for bit; the
      contraction over the below nodes (`L @ Q.T`) runs once, on the whole
      assembled left product, in `_sandwich_dense`.

    Pass 2 then gathers each chunk's tables from the one evaluation by the
    chunk's own dedup inverse — the scatter `designed_tables` does, for four
    kernels instead of six and without its (6, n) transposed copy.

    The memory shape (momwire#1173; at razor hub_deck(16) x16 the pass-1 merge
    was the fill's process peak). Every change below moves integers or copies
    floats, and none reorders or regroups a sum, so none can move a bit:

    * The index arrays are int32 where they fit (`_index_dtype`): each
      chunk's stored inverse (one per grid pair, the largest integer array the
      generator keeps) and the chunk-row -> grid-row map. An index is the same
      integer in either width, and `v[idx]` copies the same element.
    * The merge scatters each chunk's rows straight to their sorted place
      (`dest`, the inverse of the first-appearance sort) and drops the chunk
      as it goes, instead of concatenating them and then gathering the
      concatenation by `order`. `rows[dest[p]] = cat[p]` for every p is
      `rows[k] = cat[order[k]]` for every k, since `dest[order[k]] = k`.
      Likewise `gid[order] = inv` is `gid = inv[dest]`.
    * The merged rows go to `_unique_tri`, which dedups them in place of
      `_unique_rows` re-stacking three column views into a second copy.
    * The one evaluation is `designed_rows`, not `designed_tables`: the list
      is distinct already, so the latter's dedup was the identity and its
      scatter a (6, m) copy of the block (see `designed_rows`). The four
      kernels are read as columns of that block.
    """
    fold = _near_interface.radius_fold
    a_wire = float(ctx.a_wire)
    nB = rho.shape[1]
    idx_t = _index_dtype(rho.size)
    parts, firsts, inverses = [], [], []
    for sl in cols:
        _cancel.poll()
        r = fold(rho[:, sl], a_wire)
        u, inv = _near_interface._unique_rows(
            r,
            np.broadcast_to(zA[:, None], r.shape),
            np.broadcast_to(zB[None, sl], r.shape),
        )
        # `_unique_rows` numbers groups in first-appearance order, so a group
        # first occurs exactly where the running max of the inverse steps up.
        prev = np.maximum.accumulate(np.concatenate(([-1], inv[:-1])))
        i, j = np.divmod(np.flatnonzero(inv > prev), r.shape[1])
        firsts.append(i * nB + sl.start + j)
        parts.append(u)
        inverses.append((inv.astype(idx_t), u.shape[0]))
        del r, u, inv, prev, i, j
    order = np.argsort(np.concatenate(firsts), kind="stable")
    del firsts
    dest = np.empty_like(order)
    dest[order] = np.arange(order.size)
    del order
    rows = np.empty((dest.size, 3), dtype=float)
    off = 0
    for p in range(len(parts)):
        u, parts[p] = parts[p], None
        rows[dest[off : off + u.shape[0]]] = u
        off += u.shape[0]
        del u
    del parts
    uniq, inv_sorted = _near_interface._unique_tri(rows)
    del rows
    gid = inv_sorted.astype(idx_t)[dest]  # chunk-unique row -> grid-unique row
    del dest, inv_sorted
    block = _near_interface.designed_rows(eps_t, k_p, uniq, rtol=_CROSS_RTOL, memo=memo)
    del uniq
    vals = {key: block[:, _near_interface.KEYS.index(key)] for key in _CROSS_KEYS}
    off = 0
    nA = rho.shape[0]
    for sl, (inv, m) in zip(cols, inverses):
        _cancel.poll()
        idx = gid[off : off + m][inv].reshape(nA, sl.stop - sl.start)
        off += m
        yield sl, {key: v[idx] for key, v in vals.items()}


# The product route's switches (momwire#1173 design B). `_PRODUCT_TABLES`
# False sends every main sandwich through the grid dedup and `_PRODUCT_ENDS`
# False every end through the lookup path: the in-process references the
# product route is gated against to the bit. `_PRODUCT_NEG_CONTROL` is
# TEST-ONLY and makes the route WRONG on purpose, so the gate can be shown to
# fail: "transpose" maps a one-group product's rows back to the grid in the
# other factor's order (the product order permuted), "split" evaluates each
# tile's rows as two calls (which cuts its columns, so a column's second part
# is evaluated under the wrong rule witness, s_min — design C's control),
# "held" serves every held row's U and dz′W from its neighbour in the held
# buffer (design C's bookkeeping), "row" serves every fast end one product
# row off, "group0" has the fused vector loop's grouped ends read group 0's
# rows whatever their group (the one-group assumption #1335 lifted), and
# "recycle" frees a held slot one tile early (`_held_slots`).
# `_ROUTES` counts which route ran, so a test can prove the new one did;
# since design C also how many tiles and rows ran, and how much was held.
_PRODUCT_TABLES = True
_PRODUCT_ENDS = True
_PRODUCT_NEG_CONTROL = None
# Caps on the multi-group merge, None for none (the default). They used to
# be 0.25 each: a product was taken only when its candidate triples were at
# most that fraction of the grid, and its groups that fraction of the
# grouped side. Neither guarded a cost the grid route does not pay too: the
# merge's arrays are O(candidates) <= O(grid) and its lines O(groups x line)
# <= O(grid), the same order as the grid route's own rho grid, and the grid
# route then adds the chunk dedup and the hashed memo on top. Measured on
# Skylake, interleaved against the capped route, Z to the bit on every deck
# (momwire#1173): the inverted-L x8 takes the product in both blocks, peak
# RSS 767 -> 471 MB, wall 106.9 -> 106.3 s; antennaknobs' Beverage (two rod
# groups) at nseg 84, 272 -> 212 MB, 13.3 -> 13.0 s. A test may still set a
# fraction to force the grid route.
_PRODUCT_MAX_CAND_FRAC = None
_PRODUCT_MAX_GROUP_FRAC = None
# Groups beyond this build no fast-end structures, and their ends take the
# lookup path, which is exact; None for no cap (the default). It was 64 while
# a line end's key was found through one Python dict per group: at razor's
# inverted-L over 16 radials x8 the reversed block has 769 groups (one per
# node of the top wire's path axis), so its ~2,960 ends took the lookup path,
# 11.5 M asked points re-deduplicated and searched against the 1.1 M-row
# product for ~7 k fresh rows -- 2.7 s of the 4.9 s block on Haswell. The
# structures are now arrays, O(groups x line) like the plan's own `line`
# and `kid` (momwire#1224). TEST-ONLY to set: a cap sends the ends back to
# the lookup path, which is what the bit-identity gate compares against.
_PRODUCT_FAST_MAX_GROUPS = None
# Up to this many groups the fast-end structures keep the plan's raw ρ lines
# (groups x line floats), as they did when 64 was the cap; past it a group's
# row is formed again when an end asks for it (`_raw_row`).
_PRODUCT_FAST_RAW_GROUPS = 64
_ROUTES = dict.fromkeys(
    (
        "main_product",
        "main_product_z",
        "main_product_zp",
        "main_product_groups",
        "main_generic",
        "main_generic_memo",
        "main_generic_nonfinite",
        "main_generic_groups",
        "main_generic_candidates",
        "ends_fast",
        "ends_fast_grouped",
        "ends_fast_line",
        "ends_line_on_node",
        "ends_slow",
        "end_slow_calls",
        "product_rows",
        "tiles",
        "tile_rows",
        "tile_max_rows",
        "tile_held_rows",
        "tile_stores",
        "tile_column_products",
        "tile_blocks_fast",
        "fused_blocks",
        "fused_mode_stream",
        "fused_mode_post",
        "stream_units",
        "stream_finish_calls",
        "stream_held_units",
        "fused_row_ends",
        "fused_unit_tiles",
        "fused_col_te",
        "fused_held_rows",
        "held_slots",
        "fused_declined",
        "fused_declined_hit",
        "fused_declined_mixed",
        "fused_declined_groups",
        "fused_declined_hold",
        "fused_declined_support",
        "fused_declined_stream",
        "stream_chunks",
        "stream_held_cols",
        "point_chunked",
        "point_chunks",
        "point_chunk_max_rows",
        "point_eval_batches",
        "point_eval_max_rows",
        "point_tiles",
        "point_tile_rows",
        "point_tile_pool_max_rows",
        "point_pin_rows",
    ),
    0,
)


def _first_groups(*cols):
    """The distinct rows of equal-length float columns under `!=` (−0.0 with
    0.0; NaN each its own) — `_near_interface._unique_tri`'s grouping rule on
    one or two columns: `(first, rank)`, the index of each group's FIRST
    occurrence with the groups in first-appearance order, and each element's
    group number in that order. The hash kernel's answer when it serves
    (`_near_interface._factorize`, momwire#1224), the same integers."""
    n = cols[0].size
    if n == 0:
        return np.zeros(0, dtype=np.intp), np.zeros(0, dtype=np.intp)
    got = _near_interface._factorize(cols)
    if got is not None:
        return got
    idx = np.lexsort(tuple(reversed(cols)))
    new = np.empty(n, dtype=bool)
    new[0] = True
    step = new[1:]
    step[:] = False
    for c in cols:
        sc = c[idx]
        step |= sc[1:] != sc[:-1]
        del sc
    gid = np.cumsum(new) - 1
    first = idx[new]
    del new
    # `_near_interface._first_appearance`: the groups numbered by their first
    # index without sorting those indices (momwire#1224).
    rank, first_sorted = _near_interface._first_appearance(first, n)
    inv = np.empty(n, dtype=np.intp)
    inv[idx] = rank[gid]
    return first_sorted, inv


def _first_ints(ids):
    """`_first_groups` for a non-negative integer array."""
    if ids.size == 0:
        return np.zeros(0, dtype=np.intp), np.zeros(0, dtype=np.intp)
    got = _near_interface._factorize((ids,), ints=True)
    if got is not None:
        return got
    _u, first, inv = np.unique(ids, return_index=True, return_inverse=True)
    rank, first_sorted = _near_interface._first_appearance(first, ids.size)
    return first_sorted, rank[np.asarray(inv).ravel()]


class _FastEnds(NamedTuple):
    """What `_fast_end_rows` reads, beside the `ProductSet`: the grouped
    side's slot, nodes' z, group and local z rank; the line nodes' z; per
    group its first node, raw ρ line, folded line, (x, y) key and z map;
    the line nodes by (x, y); per group its row-table
    offset / width and the line's local key ranks; and the concatenated
    row tables.

    A line end's key in each group is found by array search rather than a
    per-group dict (momwire#1224): `keys` (the product's `KeyIndex`) names
    the global key of a (ρ_eff, z_line) pair, and `gkey` (`_GroupKeys`) the
    local key of global key k in group g, from the code g·n_key + k. Both are
    exact-`==` classes, the dicts' key equality, so they answer what the
    dicts did; the dicts were ~0.8 M entries per block at razor's inverted-L
    x8 (769 groups), which is why the structures used to stop at 64
    groups.

    Nothing here is O(groups x line) beyond what the plan holds anyway
    (`line`, `kl_rank`): `raw` is kept only up to `_PRODUCT_FAST_RAW_GROUPS`
    groups (as before #1224) and past that each group's row is re-formed
    when an end asks (`_raw_row`), and `gkey` is built on the first line end
    that misses its own node, which no gated deck has. Keeping both was
    +490 MB peak RSS at razor inverted-L x32."""

    grouped_slot: str
    grouped_z: np.ndarray
    grank: np.ndarray
    zl_rank: np.ndarray
    line_z: np.ndarray
    gfirst: np.ndarray
    raw: np.ndarray | None
    gxy: np.ndarray
    line_nodes: np.ndarray
    line: np.ndarray
    line_xy: dict
    gdict: dict
    zmap: list
    keys: _near_interface.KeyIndex
    gkey: "_GroupKeys"
    off: np.ndarray
    nk: np.ndarray
    kl_rank: np.ndarray
    rowflat: np.ndarray


def _raw_row(fast, g):
    """Group g's raw ρ line, `_product_plan`'s `raw[g]`: the stored row, or
    -- past `_PRODUCT_FAST_RAW_GROUPS` groups -- that row formed again by the
    plan's own expression on a (1, line) slice of its operands (the same
    elementwise subtractions and `np.hypot` on the same floats, the inner
    loop running over the line exactly as it did)."""
    if fast.raw is not None:
        return fast.raw[g]
    x0, y0 = fast.gxy[g : g + 1, 0], fast.gxy[g : g + 1, 1]
    L = fast.line_nodes
    if fast.grouped_slot == "z":
        return np.hypot(x0[:, None] - L[None, :, 0], y0[:, None] - L[None, :, 1])[0]
    return np.hypot(L[None, :, 0] - x0[:, None], L[None, :, 1] - y0[:, None])[0]


class _GroupKeys:
    """The local key of global key k in group g, by the code g·n_key + k: a
    sorted code array over every group's keys, BUILT ON FIRST USE (a line end
    off its own node, momwire#1224) -- it is O(sum of the groups' key
    counts), ~13 M codes at razor inverted-L x32, and the on-node route
    answers every end the gated decks have. `local(codes)` is the local key
    of each, or None when any is not a key of its group."""

    def __init__(self, kids, n_key):
        self._kids, self._n_key = kids, int(n_key)
        self._codes = self._loc = None

    def local(self, codes):
        if self._codes is None:
            n_key = np.int64(self._n_key)
            self._codes = _near_interface._SortedCodes(
                np.concatenate(
                    [g * n_key + kj.astype(np.int64) for g, kj in enumerate(self._kids)]
                )
            )
            self._loc = np.concatenate(
                [np.arange(kj.size, dtype=np.intp) for kj in self._kids]
            )
        j = self._codes.ids(codes)
        if np.any(j < 0):
            return None
        return self._loc[j]


def _product_route(ctx, eps_t, k_p, A, B, gz, step, memo, ends=None):
    """The main sandwich's tables by the PRODUCT route (momwire#1173 design
    B), or None to take the grid dedup — which it counts in `_ROUTES`.

    Taken when the memo is a fresh `ProductMemo` (so, as in the grid route,
    every row is fresh), every node coordinate is finite, and one side's
    nodes fall into few exact-(x, y) groups (`_product_plan`). The rows are
    `_chunked_tables`' to the float and the tables served are the same floats
    per grid pair; the memo afterwards holds the same rows with the same V and
    W values. See `_product_plan` for the rows and `_ProductTiles` (design C)
    for why evaluating them in tiles, and serving the tables in the tiles'
    order, moves no bit.

    With one tile and `nB <= step` the answer is the whole table (one
    `_TileTables`, which reads as the dict it was); otherwise a generator of
    `(cols, K_cols)` column sets, which only
    `_streamed_sandwich` (or the assembled reference in `_sandwich_dense`)
    consumes.

    `ends` (a `_FusedEnds`, design C phase 2) is offered the tiles before
    they run: when it attaches, the block's end loops read V and W inside
    the tile pass and no V/W store is kept; otherwise the tiles keep the
    store and the end loops read it afterwards, as in phase 1."""
    if not _PRODUCT_TABLES or not isinstance(memo, _near_interface.ProductMemo):
        return None
    if len(memo):
        # A second sandwich on one memo (the two-radius node) would see hits;
        # the grid route handles that, and nothing here is built for it.
        _ROUTES["main_generic"] += 1
        _ROUTES["main_generic_memo"] += 1
        return None
    plan = _product_plan(ctx, eps_t, k_p, A, B, gz)
    if isinstance(plan, str):
        _ROUTES["main_generic"] += 1
        _ROUTES["main_generic_" + plan] += 1
        return None
    tiles = _ProductTiles(plan, eps_t, k_p, step, sheet_plan=memo.sheet_plan)
    memo.set_product(tiles.product)
    if ends is None or not ends.attach(plan, tiles):
        tiles.keep_values()
    _ROUTES["main_product"] += 1
    _ROUTES["main_product_" + plan.slot] += 1
    _ROUTES["main_product_groups"] = max(
        _ROUTES["main_product_groups"], len(plan.rowtab)
    )
    _ROUTES["product_rows"] += plan.n_rows
    if tiles.n_tiles == 1 and plan.nB <= step:
        ((_cols, K),) = list(tiles.chunks(plan.nB))
        return K
    return tiles.chunks(step)


class _ProductPlan(NamedTuple):
    """`_product_plan`'s answer: the product's rows as factors, nothing
    evaluated (momwire#1173 design C). Row i is the i-th distinct triple in
    the grid's first-appearance order; `rows(ids)` writes any rows' floats
    from the node pair at their first appearance, `chunk_idx(cols)` names
    the row each grid pair (a, cols) asks, and `pair_keys(cols)` its global
    key id (so its exact-ρ column: a key is one ρ_eff)."""

    slot: str
    nA: int
    nB: int
    n_rows: int
    gz_rep: np.ndarray  # one float per grouped-z id
    key_r: np.ndarray  # one ρ_eff per global key id
    key_zl: np.ndarray  # one line z per global key id
    keys: _near_interface.KeyIndex  # the keys' lookup, shared with the product
    rowtab: list  # per group, (z of g, keys of g) -> row id
    zids: list
    kids: list
    kid: np.ndarray  # (groups, line nodes) -> global key id
    grank: np.ndarray
    zfirst: list
    kfirst: list
    nz: np.ndarray
    nk: np.ndarray
    kept_pos: np.ndarray | None  # row -> flat grid position (several groups)
    zA: np.ndarray
    zB: np.ndarray
    line: np.ndarray
    off: np.ndarray
    zl_rank: np.ndarray
    kl_rank: np.ndarray
    rowflat: np.ndarray
    fast: _FastEnds | None

    def pairs(self, ids):
        """(a, b): the node pair at each row's first grid appearance."""
        if self.kept_pos is not None:
            return np.divmod(self.kept_pos[ids], self.nB)
        if self.slot == "z":
            zi, kj = np.divmod(ids, self.nk[0])
            return self.zfirst[0][zi], self.kfirst[0][kj]
        kj, zi = np.divmod(ids, self.nz[0])
        return self.kfirst[0][kj], self.zfirst[0][zi]

    def rows(self, ids):
        """The (|ids|, 3) rows (ρ_eff, z, z′), each written from the node
        pair at its first appearance — the floats the grid computed there."""
        if (
            self.kept_pos is not None
            and _TILE_KERNELS
            and _HAVE_TILE_KERNELS
            and isinstance(self.line, _KeyLine)
            and self.line.kid.dtype == np.int32
        ):
            # The same element copies in one pass (`product_rows`).
            return _accel.acc.product_rows(
                ids,
                self.kept_pos,
                int(self.nB),
                self.slot == "z",
                self.grank,
                self.line.kid,
                self.line.key_r,
                self.zA,
                self.zB,
            )
        a_s, b_s = self.pairs(ids)
        rows = np.empty((a_s.size, 3), dtype=float)
        if self.slot == "z":
            rows[:, 0] = self.line[self.grank[a_s], b_s]
        else:
            rows[:, 0] = self.line[self.grank[b_s], a_s]
        rows[:, 1] = self.zA[a_s]
        rows[:, 2] = self.zB[b_s]
        return rows

    def chunk_idx(self, cols):
        """(nA, |cols|) rows of the grid pairs in below columns `cols`."""
        cols = np.asarray(cols)
        if self.slot == "z":
            g = self.grank
            base = self.off[g] + self.zl_rank * self.nk[g]
            return self.rowflat[base[:, None] + self.kl_rank[g[:, None], cols[None, :]]]
        g = self.grank[cols]
        base = self.off[g] + self.zl_rank[cols] * self.nk[g]
        return self.rowflat[
            base[None, :] + self.kl_rank[g[None, :], np.arange(self.nA)[:, None]]
        ]

    def pair_keys(self, cols):
        """(nA, |cols|) global key ids of the grid pairs in columns `cols`."""
        cols = np.asarray(cols)
        if self.slot == "z":
            return self.kid[self.grank[:, None], cols[None, :]]
        return self.kid[self.grank[cols][None, :], np.arange(self.nA)[:, None]]


class _KeyLine:
    """The plan's folded ρ line `line[g, n]`, read through its keys as
    `key_r[kid[g, n]]` rather than held as a (groups, line) float block: a
    key is one exact-`==` class of positive floats, so every member is its
    representative to the bit."""

    __slots__ = ("key_r", "kid")

    def __init__(self, key_r, kid):
        self.key_r, self.kid = key_r, kid

    @property
    def shape(self):
        return self.kid.shape

    def __getitem__(self, idx):
        return self.key_r[self.kid[idx]]


# The multi-group merge of a product grouped ABOVE numbers its rows by one
# hash pass over the candidates in grid order (`_merge_groups_z`). False is
# the lexsort spelling it replaced, the in-process reference: the same row
# ids, by the argument there.
_PRODUCT_FLAT_MERGE = True
# A many-group plan ranks every group's line keys in one C++ pass
# (`group_first_ranks`); False is `_first_ints` per group, the reference.
_GROUP_RANKS = True
# ...and numbers the merged rows z id by z id (`merge_rows_by_z`); False
# hashes every candidate's code (`_first_ints`), the reference.
_MERGE_BY_Z = True
_HAVE_MERGE_BY_Z_ACCEL = _accel.acc is not None and bool(
    getattr(_accel.acc, "merge_rows_by_z_1290", False)
)
# The plan's key ids by `factorize_line_keys` (momwire#1335); False is
# `_first_groups` over the raveled line and its broadcast z, the reference
# (the same integers, ~1.1 GB more transient at razor's inverted L x32).
_LINE_KEYS_ACCEL = True
_HAVE_LINE_KEYS_ACCEL = _accel.acc is not None and bool(
    getattr(_accel.acc, "factorize_line_keys_1335", False)
)
_HAVE_GROUP_RANKS_ACCEL = _accel.acc is not None and bool(
    getattr(_accel.acc, "group_first_ranks_1290", False)
)
# A many-group product's tiles class its keys by exact ρ with
# `factorize_float_classes` (momwire#1377): `factorize_rows`' groups and
# numbers in int32, with no first-row array and the table at 4/3 of the keys.
# False is `_near_interface._factorize`, the reference (the same integers,
# ~0.18 GB more transient at razor's inverted L x32).
_LEAN_KEY_CLASSES = True
_HAVE_FLOAT_CLASSES_ACCEL = _accel.acc is not None and bool(
    getattr(_accel.acc, "factorize_float_classes_1377", False)
)


def _merge_groups_z(zids, kids, zfirst, kfirst, nz, nk, n_key, nB):
    """`(n_rows, kept_pos, rowflat, rowtab)` of a slot-"z" product of
    several groups: the rows numbered by first grid appearance, each row's
    flat grid position, and the candidates' rows in the plan's group-major
    layout (`rowtab[g]` a view of `rowflat`).

    The candidates of group g are (z id of zfirst[g][i], key of
    kfirst[g][j]) at grid position zfirst[g][i]·nB + kfirst[g][j]. Every
    grouped node is in one group, so the zfirst nodes are distinct, and
    walking them ascending, each with its group's kfirst (ascending), visits
    the candidates in ascending grid position. Numbering the codes z·n_key +
    key by first appearance on that walk (`_first_ints`) is then numbering
    each distinct triple by its smallest position: the lexsort spelling's
    row ids, without a sort over the candidates. `merge_rows_by_z` gives the
    same numbering without hashing the codes at all."""
    nG = len(zids)
    g_of = np.repeat(np.arange(nG), nz)
    zl_of = np.concatenate([np.arange(n) for n in nz.tolist()])
    a_of = np.concatenate(zfirst)
    o = np.argsort(a_of, kind="stable")
    g_s, zl_s, a_s = g_of[o], zl_of[o], a_of[o]
    lens = nk[g_s]
    bstart = np.concatenate(([0], np.cumsum(lens)[:-1])).astype(np.int64)
    n_cand = int(lens.sum())
    koff = np.concatenate(([0], np.cumsum(nk))).astype(np.int64)
    off = np.concatenate(([0], np.cumsum(nz * nk)[:-1])).astype(np.int64)
    if (
        _MERGE_BY_Z
        and _HAVE_MERGE_BY_Z_ACCEL
        and all(kj.dtype == np.int32 for kj in kids)
    ):
        # The same numbering without hashing the codes: two candidates can
        # share one only under one z id (`merge_rows_by_z`).
        zcat = np.concatenate(zids).astype(np.int64)
        zoff = np.concatenate(([0], np.cumsum(nz)[:-1])).astype(np.int64)
        inv, blk, j_first = _accel.acc.merge_rows_by_z(
            zcat[zoff[g_s] + zl_s],
            g_s,
            bstart,
            np.concatenate(kids),
            koff,
            int(n_key),
            int(zcat.max()) + 1,
            cancel_flag=_cancel.ptr(),
        )
        n_rows = int(blk.size)
    else:
        codes = np.empty(n_cand, dtype=np.int64)
        n_key = np.int64(n_key)
        for i, (g, zl) in enumerate(zip(g_s.tolist(), zl_s.tolist())):
            b0 = bstart[i]
            codes[b0 : b0 + lens[i]] = zids[g][zl] * n_key + kids[g]
        first, inv = _first_ints(codes)
        del codes
        n_rows = int(first.size)
        blk = np.searchsorted(bstart, first, side="right") - 1
        j_first = first - bstart[blk]
    rowflat = np.empty(n_cand, dtype=_index_dtype(n_rows))
    for i, (g, zl) in enumerate(zip(g_s.tolist(), zl_s.tolist())):
        if not i & 1023:
            _cancel.poll()
        d0, b0 = off[g] + zl * nk[g], bstart[i]
        rowflat[d0 : d0 + nk[g]] = inv[b0 : b0 + nk[g]]
    del inv
    _cancel.poll()
    # Each row's flat grid position, a span of rows at a time (momwire#1335:
    # the one-shot spelling formed four row-length int64 temporaries, the
    # inverted L x32's merge spike); the same integers.
    kcat = np.concatenate(kfirst)
    kept_pos = np.empty(blk.size, dtype=_index_dtype(int(a_s.max(initial=0)) * nB + nB))
    step = 1 << 20
    for c0 in range(0, blk.size, step):
        _cancel.poll()
        bb = blk[c0 : c0 + step]
        b = kcat[koff[g_s[bb]] + j_first[c0 : c0 + step]]
        kept_pos[c0 : c0 + step] = a_s[bb].astype(np.int64) * nB + b
    del kcat
    rowtab = [
        rowflat[off[g] : off[g] + nz[g] * nk[g]].reshape(nz[g], nk[g])
        for g in range(nG)
    ]
    return n_rows, kept_pos, rowflat, rowtab


def _xy_index(nodes):
    """{(x, y): [node indices, ascending]} of an (n, 3) node array, keyed on
    the floats (−0.0 with 0.0, as a dict keys them)."""
    out = {}
    for n, xy in enumerate(zip(nodes[:, 0].tolist(), nodes[:, 1].tolist())):
        out.setdefault(xy, []).append(n)
    return out


def _product_plan(ctx, eps_t, k_p, A, B, gz):
    """The `_ProductPlan` of the main sandwich over (above axis A × below
    axis B), or the reason it declines ("nonfinite", "groups",
    "candidates"). Nothing is evaluated here since momwire#1173 design C:
    `_ProductTiles` evaluates the rows, tile by tile.

    THE ARGUMENT. The grid route asks, at node pair (a, b), the triple
    (ρ_eff, z, z′) = (fold(hypot(x_a − x_b, y_a − y_b)), z_a, z_b), and hands
    `designed_rows` the distinct triples in the order of their FIRST grid
    appearance under the flat index a·n_B + b, each as the floats at that
    appearance. Group one side's nodes by exact (x, y). Within a group every
    node has the same x and y under `==`, and a difference with an operand
    equal under `==` has the same magnitude (only a zero's sign can differ,
    and hypot ignores signs), so ρ from any node of group g to a node l of
    the other side is ONE float, `line[g, l]`. Group g's triples are then
    exactly {its distinct z} × {the line's distinct (ρ_eff, z_l)}: every
    pair occurs, as the node pair (first node of g with that z, first line
    node with that key). The grid's distinct triples are the union over the
    groups; a triple two groups share is one triple, first appearing at the
    smaller of its candidates' flat positions. So:

      * candidates are numbered by (z id, key id) — both exact-`==` classes,
        and two triples are equal iff both ids are — and deduplicated
        keeping the smallest flat position;
      * the survivors, sorted by that position, are the grid's distinct
        triples in first-appearance order;
      * each row is written from the node pair at that position: z and z′
        straight from the nodes, ρ_eff from the line at that pair — the
        float the grid computed there (`_ProductPlan.rows`).

    With ONE group nothing needs sorting: the grouped nodes' first
    appearances and the line's are both ascending, so the flat positions run
    in the product's own order — z-major when the grouped side is the above
    one (a·n_B + b with a from z), key-major when it is the below one. That
    is also the design-B prototype's order, and it is asserted below; the
    node pair of a row is then its position's divmod, and no position array
    is kept.

    No float is computed here that the grid route did not compute the same
    way (the one fold per line node is `radius_fold` of the grid's own ρ);
    no sum is formed at all.

    Which side is grouped: the one whose lines are cheaper (groups × other
    side's nodes), the above side on a tie. A vertical above member is one
    group (the hub decks); WA7ARK-style decks, whose short buried rod is one
    group under a sloping antenna, group the below side."""
    pa, pb = A["nodes"], B["nodes"]
    nA, nB = pa.shape[0], pb.shape[0]
    if not (np.all(np.isfinite(pa)) and np.all(np.isfinite(pb)) and np.isfinite(gz)):
        return "nonfinite"
    if nA == 0 or nB == 0:
        return "groups"
    fa, ga = _first_groups(pa[:, 0], pa[:, 1])
    fb, gb = _first_groups(pb[:, 0], pb[:, 1])
    if fa.size * nB <= fb.size * nA:
        slot, G, L, gfirst, grank = "z", pa, pb, fa, ga
    else:
        slot, G, L, gfirst, grank = "zp", pb, pa, fb, gb
    nG, nL = gfirst.size, L.shape[0]
    cap = _PRODUCT_MAX_GROUP_FRAC
    if nG > 1 and cap is not None and nG > cap * G.shape[0]:
        return "groups"
    # z relative to the plane, exactly as `_direct_coords` forms it.
    zA = pa[:, 2] - gz
    zB = pb[:, 2] - gz
    gzv, lzv = (zA, zB) if slot == "z" else (zB, zA)
    x0, y0 = G[gfirst, 0], G[gfirst, 1]
    # The grid's operand order is (above − below). Formed a span of groups
    # at a time with a poll between (momwire#1348): elementwise, so the same
    # floats as the one-shot expression, and at x32 the block is a quarter of
    # a second of numpy no cancel could otherwise interrupt.
    raw = np.empty((nG, nL))
    spans = _group_spans(nG, nL)
    for g0, g1 in spans:
        _cancel.poll()
        if slot == "z":
            raw[g0:g1] = np.hypot(
                x0[g0:g1, None] - L[None, :, 0], y0[g0:g1, None] - L[None, :, 1]
            )
        else:
            raw[g0:g1] = np.hypot(
                L[None, :, 0] - x0[g0:g1, None], L[None, :, 1] - y0[g0:g1, None]
            )
    keep_raw = nG <= _PRODUCT_FAST_RAW_GROUPS
    if keep_raw:
        line = _near_interface.radius_fold(raw, float(ctx.a_wire))
    else:
        # `radius_fold` in place: the same ufunc on the same floats, and the
        # (groups, line) raw block is not kept past it (`_raw_row`).
        line = raw
        for g0, g1 in spans:
            _cancel.poll()
            np.hypot(raw[g0:g1], float(ctx.a_wire), out=raw[g0:g1])
        raw = None
    zf, zid = _first_groups(gzv)
    if (
        _LINE_KEYS_ACCEL
        and _HAVE_LINE_KEYS_ACCEL
        and _near_interface._FACTORIZE
        and isinstance(line, np.ndarray)
        and line.flags.c_contiguous
        and nG * nL < 2**31 - 1
    ):
        # The same (first, ids) as the hash kernel's `_first_groups` below,
        # without its transients (`factorize_line_keys`, momwire#1335): the
        # line's z read in place, the ids int32 in the table's own shape,
        # the table grown with the keys rather than sized by the rows.
        kf, kid = _accel.acc.factorize_line_keys(
            line, np.ascontiguousarray(lzv, dtype=float), cancel_flag=_cancel.ptr()
        )
    else:
        kf, kid = _first_groups(line.ravel(), np.broadcast_to(lzv, line.shape).ravel())
    # The (groups, line) index blocks in 32 bits when they fit: they and the
    # row table are the plan's O(groups x line) part (momwire#1224).
    kid = kid.reshape(nG, nL).astype(_index_dtype(kf.size), copy=False)
    key_r = line.ravel()[kf]
    key_zl = lzv[kf % nL]
    # Every line value of key k is key_r[k] to the bit (equal under `==` and
    # positive, as rho_eff >= a > 0), so the plan reads its line through the
    # keys and the folded block goes here.
    del line
    line = _KeyLine(key_r, kid)
    # Per group: members (ascending), local z and key ranks, candidates.
    members = np.argsort(grank, kind="stable")
    bounds = np.concatenate(([0], np.cumsum(np.bincount(grank, minlength=nG))))
    zl_rank = np.empty(G.shape[0], dtype=np.intp)
    kl_rank = np.empty((nG, nL), dtype=_index_dtype(nL))
    zids, kids, zfirst, kfirst, nz, nk = [], [], [], [], [], []
    by_rows = (
        nG > 1
        and _GROUP_RANKS
        and _HAVE_GROUP_RANKS_ACCEL
        and kid.dtype == np.int32
        and kl_rank.dtype == np.int32
    )
    if by_rows:
        # Every group's `_first_ints(kid[g])` in one pass
        # (`group_first_ranks`): the same ranks, first positions and ids.
        kl_rank, nk_g, f_cat, k_cat = _accel.acc.group_first_ranks(
            kid, int(kf.size), cancel_flag=_cancel.ptr()
        )
        koff = np.concatenate(([0], np.cumsum(nk_g))).astype(np.int64)
        # Line positions in the line's own index width (the plan's
        # O(groups x keys) lists).
        f_line = f_cat.astype(_index_dtype(nL), copy=False)
        del f_cat
    for g in range(nG):
        _cancel.poll()
        m_g = members[bounds[g] : bounds[g + 1]]
        if m_g.size == 1:
            f_z = r_z = np.zeros(1, dtype=np.intp)  # `_first_ints` of one id
        else:
            f_z, r_z = _first_ints(zid[m_g])
        zl_rank[m_g] = r_z
        zids.append(zid[m_g[f_z]])
        zfirst.append(m_g[f_z])  # the grouped node where each z first occurs
        nz.append(f_z.size)
        if by_rows:
            kids.append(k_cat[koff[g] : koff[g + 1]])
            kfirst.append(f_line[koff[g] : koff[g + 1]])
            nk.append(int(nk_g[g]))
            continue
        f_k, r_k = _first_ints(kid[g])
        kl_rank[g] = r_k
        kids.append(kid[g, f_k])
        kfirst.append(f_k)  # the line node where each key first occurs
        nk.append(f_k.size)
    nz, nk = np.asarray(nz, dtype=np.intp), np.asarray(nk, dtype=np.intp)
    n_cand = int(np.sum(nz * nk))
    cap = _PRODUCT_MAX_CAND_FRAC
    if nG > 1 and cap is not None and n_cand > cap * nA * nB:
        return "candidates"

    def flat_pos(g):
        """Flat grid positions a·nB + b of group g's (z, key) candidates."""
        if slot == "z":
            return zfirst[g][:, None] * nB + kfirst[g][None, :]
        return kfirst[g][None, :] * nB + zfirst[g][:, None]

    kept_pos = None
    if nG == 1:
        # Ascending already (see the docstring): row = the candidate itself,
        # z-major for the above side, key-major for the below.
        c = np.arange(n_cand, dtype=np.intp)
        if _PRODUCT_NEG_CONTROL == "transpose":
            slot_rows = "zp" if slot == "z" else "z"  # TEST-ONLY: wrong order
        else:
            slot_rows = slot
        cand_row = [
            c.reshape(nz[0], nk[0]) if slot_rows == "z" else c.reshape(nk[0], nz[0]).T
        ]
        pos1 = (flat_pos(0) if slot == "z" else flat_pos(0).T).ravel()
        if pos1.size > 1 and not bool(np.all(pos1[1:] > pos1[:-1])):
            raise AssertionError("one group's candidates are not in grid order")
        del pos1
        n_rows = n_cand
    elif slot == "z" and _PRODUCT_FLAT_MERGE:
        n_rows, kept_pos, rowflat, cand_row = _merge_groups_z(
            zids, kids, zfirst, kfirst, nz, nk, kf.size, nB
        )
    else:
        n_key = kf.size
        codes = np.concatenate(
            [(zids[g][:, None] * n_key + kids[g][None, :]).ravel() for g in range(nG)]
        )
        pos = np.concatenate([flat_pos(g).ravel() for g in range(nG)])
        o = np.lexsort((pos, codes))
        cs = codes[o]
        head = np.empty(cs.size, dtype=bool)
        head[0] = True
        np.not_equal(cs[1:], cs[:-1], out=head[1:])
        run = np.cumsum(head) - 1  # the distinct triple of each sorted candidate
        kept_pos = pos[o][head]  # each triple's first appearance
        rank = np.empty(kept_pos.size, dtype=np.intp)
        rank[np.argsort(kept_pos, kind="stable")] = np.arange(kept_pos.size)
        flat_row = np.empty(cs.size, dtype=np.intp)
        flat_row[o] = rank[run]
        kept_pos = np.sort(kept_pos)
        del codes, pos, o, cs, head, run, rank
        cand_row, off = [], 0
        for g in range(nG):
            cand_row.append(flat_row[off : off + nz[g] * nk[g]].reshape(nz[g], nk[g]))
            off += nz[g] * nk[g]
        del flat_row
        n_rows = kept_pos.size
    # The value block is in ROW order (`_ProductTiles`), so a candidate's
    # value row is its row.
    rowtab = cand_row
    off = np.concatenate(([0], np.cumsum(nz * nk)[:-1])).astype(np.intp)
    if nG == 1 or not (slot == "z" and _PRODUCT_FLAT_MERGE):
        rowflat = np.concatenate([t.ravel() for t in rowtab]).astype(
            _index_dtype(int(n_rows)), copy=False
        )
    keys = _near_interface.KeyIndex(key_r, key_zl)
    cap = _PRODUCT_FAST_MAX_GROUPS
    fast = None
    if cap is None or nG <= cap:
        n_key = kf.size
        fast = _FastEnds(
            grouped_slot=slot,
            grouped_z=gzv,
            grank=grank,
            zl_rank=zl_rank,
            line_z=lzv,
            gfirst=gfirst,
            raw=raw,
            gxy=np.stack([x0, y0], axis=1),
            line_nodes=L,
            line=line,
            line_xy=_xy_index(L),
            gdict={
                (float(x), float(y)): g
                for g, (x, y) in enumerate(zip(x0.tolist(), y0.tolist()))
            },
            zmap=[
                {float(gzv[n]): i for i, n in enumerate(zfirst[g].tolist())}
                for g in range(nG)
            ],
            keys=keys,
            gkey=_GroupKeys(kids, n_key),
            off=off,
            nk=nk,
            kl_rank=kl_rank,
            rowflat=rowflat,
        )
    return _ProductPlan(
        slot=slot,
        nA=nA,
        nB=nB,
        n_rows=int(n_rows),
        gz_rep=gzv[zf],
        key_r=key_r,
        key_zl=key_zl,
        keys=keys,
        rowtab=rowtab,
        zids=zids,
        kids=kids,
        kid=kid,
        grank=grank,
        zfirst=zfirst,
        kfirst=kfirst,
        nz=nz,
        nk=nk,
        kept_pos=kept_pos,
        zA=zA,
        zB=zB,
        line=line,
        off=off,
        zl_rank=zl_rank,
        kl_rank=kl_rank,
        rowflat=rowflat,
        fast=fast,
    )


# momwire#1173 design C, phase 1: the product's rows are evaluated in TILES
# of at most about this many rows (plus one column: a column is never cut),
# each contracted into the main sandwich and dropped, so the (rows, 6) value
# block is never whole. Phase 1's end loops read V and W of every row from a
# (rows, 2) store; since phase 2 they run inside the tiles (`_FusedEnds`) and
# the store is kept only when they cannot (`keep_values`). 2^17 is
# DESIGN-C's measured budget: past about 32 k rows the per-call overhead of
# the kernel stops showing. TEST-ONLY to shrink.
_TILE_ROWS = 1 << 17
# A many-group product (grouped above) cuts its tiles along the LINE instead
# of along ρ (`_column_tiles`). False is the ρ-ordered tiling, the in-process
# reference: both evaluate every exact-ρ class whole in one tile, so the
# tables, and Z, are the same bits either way (`_ProductTiles`).
_COLUMN_TILES = True
# TEST-ONLY False: a many-group tile's rows by the per-group gather and mask
# (`_ProductTiles._tile_rows`), the reference for the one-sort order.
_TILE_ROW_ORDER = True
# `_group_spans`' budget: groups x line entries per pass over `plan.kid`.
_GROUP_SPAN_ELEMS = 1 << 21


def _group_spans(n_groups, n_line):
    """[g0, g1) spans of the plan's groups, each about `_GROUP_SPAN_ELEMS`
    (group, line node) entries: the passes over `kid` that would otherwise
    form a (groups, line) temporary at once."""
    per = max(1, _GROUP_SPAN_ELEMS // max(1, n_line))
    return [(g0, min(n_groups, g0 + per)) for g0 in range(0, n_groups, per)]


def _column_tiles(plan, key_cls, cls_rows):
    """`(tile_of_cls, ready, n_tiles)` for a slot-"z" product of several
    groups: each exact-ρ class's tile, each line node's ready tile.

    The ρ-ordered tiling serves a line node's column once every group's row
    for it has run, and with many groups (one per node of a horizontal
    wire) those rows sit in tiles across the whole ρ range, so nearly every
    column waits for the last tiles and nearly every row is HELD until then
    (4.2 M of 4.4 M rows at razor inverted-L x16: a second value block).

    Here the line is walked in a fixed order and each class is put in the
    tile of the first column that asks any of its keys; the walk is cut
    into tiles of about `_TILE_ROWS` new rows. A column's classes are then
    all in its own tile or an earlier one, so it is ready at its own tile,
    and a row is held only when a column in a later tile asks it too. The
    walk orders the line by the ρ class of the ANCHOR group's key (the
    group with most z values, whose rows outnumber the others' per key),
    so the columns sharing an anchor key are adjacent; a run of them is
    never cut, so the anchor's rows are never held. With mirror-symmetric
    lines (radials at ±θ about a top wire's line) the partner columns share
    the anchor key and with it the rows the mirror makes equal.

    Every class is still whole in one tile, which is the whole of the bit
    argument (`_ProductTiles`): the tiles' order and membership are a
    locality choice."""
    kid = plan.kid
    n_groups, n_line = kid.shape
    anchor = int(np.argmax(plan.nz))
    # The anchor's ρ along the line: its class order is its ρ order (a class
    # is one exact ρ), with equal ρ in line order.
    a_cls = plan.key_r[kid[anchor]]
    order = np.argsort(a_cls, kind="stable")
    # Line positions in the line's own index width: `first` is one per class,
    # ~11.3 M at razor's inverted L x32 (momwire#1377).
    pdt = _index_dtype(n_line + 1)
    pos = np.empty(n_line, dtype=pdt)
    pos[order] = np.arange(n_line, dtype=pdt)
    first = np.full(cls_rows.size, n_line, dtype=pdt)
    for g0, g1 in _group_spans(n_groups, n_line):
        _cancel.poll()
        c = key_cls[kid[g0:g1]]
        np.minimum.at(first, c.ravel(), np.broadcast_to(pos, c.shape).ravel())
        del c
    if np.any(first >= n_line):
        raise AssertionError("a key class no line node asks")
    # `bincount(first, weights=cls_rows)` a span of classes at a time: the
    # weights are whole row counts, so every partial sum is an integer below
    # 2^53 and exact, and the per-node totals are the one call's floats in
    # any grouping, without its class-length intp and float64 copies.
    new_at = np.zeros(n_line)
    for c0 in range(0, first.size, _CLASS_SPAN):
        _cancel.poll()
        new_at += np.bincount(
            first[c0 : c0 + _CLASS_SPAN],
            weights=cls_rows[c0 : c0 + _CLASS_SPAN],
            minlength=n_line,
        )
    start = np.cumsum(new_at) - new_at
    t_pos = (start // max(1, int(_TILE_ROWS))).astype(np.int64)
    a_walk = a_cls[order]
    run = np.cumsum(np.concatenate(([True], a_walk[1:] != a_walk[:-1]))) - 1
    run_head = np.flatnonzero(np.concatenate(([True], a_walk[1:] != a_walk[:-1])))
    t_pos = t_pos[run_head][run]
    _t, t_pos = np.unique(t_pos, return_inverse=True)
    t_pos = np.asarray(t_pos).ravel()
    n_tiles = int(_t.size)
    # Each class's tile in the width the tiles keep (`tile_of_key`), formed
    # straight from the line's: the same integers, no int64 class array.
    t_line = t_pos.astype(_tile_dtype(n_tiles), copy=False)
    return t_line[first], t_pos[pos], n_tiles


def _tile_dtype(n_tiles):
    """The width of a tile number: `_ProductTiles.tile_of_key`'s."""
    return np.int16 if n_tiles < 2**15 else np.int64


# Classes per pass of the tiles' class-length integer work (momwire#1377).
_CLASS_SPAN = 1 << 20


# A many-group product's held store reuses a slot once the row in it has
# been read for the last time (`_ProductTiles._plan_held`, momwire#1335).
# False keeps one slot per late row for the whole tile pass, the reference:
# the slots only ever hold copies, so their numbering moves no bit.
_RECYCLE_HELD = True


def _late_last(n_rows, n_tiles):
    """A per-row LAST reading tile, -1 for a row nobody reads late: the one
    array every late reader marks (`np.maximum.at`), O(rows) in 16 bits
    where the tiles fit -- not a list of (row, tile) pairs, which repeat a
    row once per group that holds it (4 M entries for 1.5 M rows at the
    inverted L's x16)."""
    return np.full(n_rows, -1, dtype=np.int16 if n_tiles < 2**15 else np.int32)


def _mark_late(last, rows, tiles_):
    """`last[rows] = max(last[rows], tiles_)`, element by element (a row
    named twice keeps its later tile)."""
    rows = np.asarray(rows)
    if rows.size:
        vals = np.broadcast_to(np.asarray(tiles_, dtype=last.dtype), rows.shape)
        np.maximum.at(last, rows.ravel(), vals.ravel())


def _held_live_peak(n_tiles, tile_rows, last):
    """The most late rows live in any one tile -- `_held_slots`' slot count
    (interval partitioning reaches it) -- by counts alone: rows written per
    tile, rows last read per tile. For a budget, without assigning slots."""
    born = np.zeros(n_tiles, dtype=np.int64)
    for t in range(n_tiles):
        born[t] = int(np.count_nonzero(last[tile_rows(t)] >= 0))
    lv = last[last >= 0].astype(np.int64)
    if lv.size == 0:
        return 0
    gone = np.bincount(lv, minlength=n_tiles)[:n_tiles]
    live = np.cumsum(born) - np.concatenate(([0], np.cumsum(gone)[:-1]))
    return int(live.max())


def _held_slots(n_tiles, tile_rows, last, hpos=None):
    """The held store's slots, by a walk over the tiles in order; returns
    the slot count and, given `hpos`, writes each late row's slot there.

    `tile_rows(t)` is tile t's rows and `last[row]` the last tile reading a
    row late (-1: none). Tile t first frees the slots of the rows last read
    at tile t - 1, then hands slots to its own late rows: freed ones first,
    then fresh. A row is live from its own tile through its last reading
    tile, and no two live rows share a slot (interval partitioning: the
    count is the most rows live in any tile). A late row must be read
    AFTER its own tile; one that is not is refused (its reader would have
    read it before it was evaluated)."""
    early = 1 if _PRODUCT_NEG_CONTROL == "recycle" else 0
    free = np.zeros(0, dtype=np.int64)
    pending = {}  # last tile -> slot arrays of the rows it frees
    n_slots = 0
    for t in range(n_tiles):
        if not t & 15:
            _cancel.poll()
        # TEST-ONLY "recycle": the slots freed one tile early, at their rows'
        # last reading tile, where a row written there can overwrite them.
        rel = pending.pop(t - 1 + early, None)
        if rel:
            free = np.concatenate([free, *rel])
        ids = tile_rows(t)
        lt = last[ids]
        m = lt >= 0
        if not m.any():
            continue
        new, lt = ids[m], lt[m].astype(np.int64)
        if np.any(lt <= t):
            raise AssertionError("a held row is not read after its own tile")
        k = min(new.size, free.size)
        slots = np.concatenate(
            [free[free.size - k :], n_slots + np.arange(new.size - k, dtype=np.int64)]
        )
        free = free[: free.size - k]
        n_slots += new.size - k
        if hpos is not None:
            hpos[new] = slots
        o = np.argsort(lt, kind="stable")
        lt_s = lt[o]
        cut = np.flatnonzero(np.diff(lt_s)) + 1
        for part, L in zip(np.split(slots[o], cut), lt_s[np.r_[0, cut]].tolist()):
            pending.setdefault(L, []).append(part)
    return n_slots


# `_stable_tile_order`'s chunk of rows (momwire#1335).
_TILE_ORDER_CHUNK = 1 << 20
# `_stable_tile_order` by `stable_tile_order` in C++ (momwire#1377); False
# is the chunked numpy counting sort, the reference (the same integers).
_TILE_ORDER_ACCEL = True
_HAVE_TILE_ORDER_ACCEL = _accel.acc is not None and bool(
    getattr(_accel.acc, "stable_tile_order_1377", False)
)


def _stable_tile_order(t_row, n_tiles, dtype):
    """`(o, b)`: `o = np.argsort(t_row, kind="stable")` in `dtype` and `b`
    the tiles' bounds in it (`searchsorted(t_row[o], arange(n_tiles + 1))`),
    by a counting sort a chunk of rows at a time (momwire#1335). The rows of
    a tile land in its slots in ascending order -- chunks in order, each
    chunk's own stable sort -- which is the stable argsort's answer, without
    its int64 permutation and the narrowing copy of it (~210 MB at razor's
    inverted L x32). `stable_tile_order` (momwire#1377) is the same counting
    sort in one C++ pass, 0.69 -> ~0.1 s there."""
    if (
        _TILE_ORDER_ACCEL
        and _HAVE_TILE_ORDER_ACCEL
        and t_row.dtype == np.int16
        and dtype in (np.int32, np.int64)
    ):
        o, b = _accel.acc.stable_tile_order(
            t_row, int(n_tiles), dtype == np.int64, cancel_flag=_cancel.ptr()
        )
        return o, b
    n = t_row.size
    counts = np.bincount(t_row, minlength=n_tiles).astype(np.int64)
    b = np.zeros(n_tiles + 1, dtype=np.int64)
    np.cumsum(counts, out=b[1:])
    cursor = b[:-1].copy()
    o = np.empty(n, dtype=dtype)
    step = max(1, int(_TILE_ORDER_CHUNK))
    for c0 in range(0, n, step):
        _cancel.poll()
        ts = t_row[c0 : c0 + step]
        oc = np.argsort(ts, kind="stable")
        tsorted = ts[oc]
        cnt = np.bincount(ts, minlength=n_tiles).astype(np.int64)
        gstart = np.cumsum(cnt) - cnt
        dest = cursor[tsorted] + (np.arange(oc.size, dtype=np.int64) - gstart[tsorted])
        o[dest] = oc + c0
        cursor += cnt
    return o, b


class _ProductTiles:
    """The product's rows evaluated a TILE at a time, and the main
    sandwich's table columns served as each becomes complete (momwire#1173
    design C phase 1). Bit-identical to evaluating every row in ONE
    `designed_rows_permuted` call and gathering the tables from that block
    (design B), for three reasons.

    WHOLE COLUMNS. The column route groups one call's rows by exact ρ_eff,
    and a member's value depends on its column only through the column's
    `s_min` (`six_columns`; the C++ twin's `s_min_of` feeds a rule built
    serially per column, and `column_member` reduces each member alone). A
    row's ρ_eff is its key's (`key_r`), so the exact-ρ classes of the keys
    ARE the one call's columns, and each tile here is a union of whole
    classes: every column of the one call is evaluated in exactly one tile,
    with every one of its members, in the same relative order (a tile's rows
    are its row ids ascending, and the one call's member order is row order
    within a column). So each column sees the same ρ, the same members and
    the same `s_min`, builds the same rule, and returns the same six floats
    per member, on every route: the twin (columns independent, members
    independent), the numpy column loop (the same `six_columns` call on the
    same member list) and the point routes (one row at a time). No rule
    witness is needed, because no column is ever cut; the price is that a
    tile can exceed `_TILE_ROWS` by one column. Tiles run in ascending ρ,
    which is only a locality choice: the order of calls moves no bit.

    EXACTLY ONCE. The tiles partition the key set, every row has one key,
    and a row shared by several groups is one row under one key: each row
    is evaluated once (a `done` bitmap refuses a second evaluation and
    checks the last tile left none out), and no row is recomputed — the
    counters `tile_rows` and `product_rows` are equal.

    THE SAME TABLES. A below node's table column holds the values of the
    rows its grid pairs ask (`chunk_idx`). It is served once every one of
    those rows is evaluated — at the tile that completes it (`ready`) — as
    copies of the very floats the one call returned for them: from the
    current tile or, for a row evaluated in an earlier tile and read later,
    from the held store (`hpos`, written once when its tile ran) — U and
    dz′W always, and V and W as well unless the unfused route keeps them in
    its row-ordered store (`keep_values`). Nothing is summed. A fused end
    loop (`_FusedEnds`, phase 2) is handed each tile before its columns are
    served, reads the same copies, and names the rows it reads after their
    own tile (`extra_last`), which the held store keeps beside the
    sandwich's. `_streamed_sandwich` then contracts each
    basis row once, when its whole pattern is present, against its columns
    in ascending order — its own argument, for column SETS; the left
    product of a column is `P @ K[:, j]`, which a sparse @ dense product
    forms from that column alone.

    On a one-group product grouped ABOVE (the hub decks) a below node's
    rows all share its one key, so it is ready at its key's tile and
    nothing is ever held. Grouped below (WA7ARK) a node's rows span every
    key of the line, so it is ready only at the last tile and the U and
    dz′W of every row it reads early are held until then: correct, and
    smaller than design B's block (32 B per held row against 96 per row),
    but not bounded. Bounding that case needs columns cut across
    tiles, which needs the rule witness (and the twin, since the numpy
    column loop's chunking is not member-independent) — not phase 1."""

    def __init__(self, plan, eps_t, k_p, step, sheet_plan=None):
        self.plan = plan
        self.eps_t = eps_t
        self.k_p = k_p
        # The fill's plane-sheet plan (its memo's; the tiles evaluate
        # through `designed_rows_permuted`, which takes no memo).
        self.sheet_plan = sheet_plan
        U = plan.n_rows
        n_key = plan.key_r.size
        # The one call's columns: exact-ρ classes of the keys, ascending.
        # (the plan's `KeyIndex` formed exactly this `np.unique` already)
        column = plan.slot == "z" and len(plan.rowtab) > 1 and _COLUMN_TILES
        lean = (
            column
            and _LEAN_KEY_CLASSES
            and _HAVE_FLOAT_CLASSES_ACCEL
            and _near_interface._FACTORIZE
            and _near_interface._HAVE_FACTORIZE_ACCEL
            and n_key <= _near_interface._FACTORIZE_MAX_ROWS
        )
        if lean:
            # `_factorize`'s groups and numbers (`factorize_float_classes`),
            # in int32 and without its first-row array.
            n_cls, key_cls = _accel.acc.factorize_float_classes(
                np.ascontiguousarray(plan.key_r, dtype=float), cancel_flag=_cancel.ptr()
            )
            n_cls = int(n_cls)
            got = None
        else:
            got = _near_interface._factorize((plan.key_r,)) if column else None
        if got is not None:
            # Column tiles read the classes as labels (their walk orders the
            # line by ρ itself, `_column_tiles`), so they are numbered by one
            # hash pass instead of sorted: the same partition of the keys.
            key_cls = np.asarray(got[1]).ravel()
            n_cls = int(got[0].size)
        elif not lean:
            _r_u, key_cls = plan.keys.take_r_classes()
            n_cls = _r_u.size
        del got
        # A candidate count per key (an upper bound on its rows): at most the
        # grouped nodes, so 32 bits.
        rows_per_key = np.zeros(n_key, dtype=np.int32)
        for g, kj in enumerate(plan.kids):
            if not g & 255:
                _cancel.poll()
            rows_per_key[kj] += plan.nz[g]
        if lean:
            # `bincount(key_cls, weights=rows_per_key)` read in place
            # (`class_sums`): whole counts, so the same floats.
            cls_rows = _accel.acc.class_sums(
                key_cls, rows_per_key, n_cls, cancel_flag=_cancel.ptr()
            )
        else:
            cls_rows = np.bincount(key_cls, weights=rows_per_key, minlength=n_cls)
        del rows_per_key
        col_ready = None
        if column:
            t_of_cls, col_ready, self.n_tiles = _column_tiles(plan, key_cls, cls_rows)
            _ROUTES["tile_column_products"] += 1
        else:
            start = np.cumsum(cls_rows) - cls_rows
            t_of_cls = (start // max(1, int(_TILE_ROWS))).astype(np.int64)
            _t, t_of_cls = np.unique(t_of_cls, return_inverse=True)
            t_of_cls = np.asarray(t_of_cls).ravel()
            self.n_tiles = int(_t.size)
        del cls_rows
        self.tile_of_key = t_of_cls[key_cls].astype(
            _tile_dtype(self.n_tiles), copy=False
        )
        del t_of_cls, key_cls
        # Per group, its local keys sorted by tile -- or, with many groups,
        # every row in tile order (`_tile_rows`).
        self._gkeys = []
        self._by_tile = None
        if len(plan.rowtab) > 1 and self.n_tiles < 2**15 and _TILE_ROW_ORDER:
            t_row = np.empty(U, dtype=np.int16)
            for g, (tab, kj) in enumerate(zip(plan.rowtab, plan.kids)):
                if not g & 255:
                    _cancel.poll()
                t_row[tab] = self.tile_of_key[kj][None, :]
            # A row's tile is its key's, the same in every group holding it.
            # The rows in tile order, ascending within a tile: a stable sort
            # on the tile, by counting (`_stable_tile_order`).
            self._by_tile = _stable_tile_order(t_row, self.n_tiles, _index_dtype(U))
            del t_row
        else:
            for g, kj in enumerate(plan.kids):
                if not g & 255:
                    _cancel.poll()
                tl = self.tile_of_key[kj]
                o = np.argsort(tl, kind="stable")
                b = np.searchsorted(tl[o], np.arange(self.n_tiles + 1))
                self._gkeys.append((o, b))
        # The tile at which each below node's table column is complete.
        if col_ready is not None:
            # `_column_tiles` names it; checked here against what serving
            # needs, no row of the column in a later tile.
            ready, may_hold = col_ready, False
            for g0, g1 in _group_spans(*plan.kid.shape):
                _cancel.poll()
                tk = self.tile_of_key[plan.kid[g0:g1]]
                if np.any(tk > ready[None, :]):
                    raise AssertionError("a column is served before its rows")
                may_hold = may_hold or bool(np.any(tk < ready[None, :]))
                del tk
        elif plan.slot == "z":
            tk = self.tile_of_key[plan.kid]  # (groups, line nodes)
            ready = tk.max(axis=0)
            may_hold = bool(np.any(tk < ready[None, :]))
        else:
            tk = self.tile_of_key[plan.kid]  # (groups, line nodes)
            gmax = tk.max(axis=1)
            ready = gmax[plan.grank]
            may_hold = bool(np.any(tk < gmax[:, None]))
        o = np.argsort(ready, kind="stable")  # ascending node index per tile
        self._ready = (o, np.searchsorted(ready[o], np.arange(self.n_tiles + 1)))
        self._node_ready = ready
        self._may_hold = may_hold
        self._step = step
        # The product's factors, without values until `keep_values` (the
        # unfused route) gives it the V/W store; the fused route
        # (`_FusedEnds`) reads V and W inside the tiles and keeps none.
        self.store = None
        # `_gather`'s index pass in C++ (`product_chunk_index`): the grouped
        # nodes' row-table bases, when the plan's tables are 32-bit.
        self._zbase = None
        if (
            plan.slot == "z"
            and _CHUNK_INDEX_ACCEL
            and _HAVE_CHUNK_INDEX_ACCEL
            and plan.rowflat.dtype == np.int32
            and plan.kl_rank.dtype == np.int32
            and _index_dtype(U) == np.int32
        ):
            g = plan.grank
            self._zbase = (plan.off[g] + plan.zl_rank * plan.nk[g]).astype(np.int64)
        self.product = _near_interface.ProductSet(
            plan.slot,
            None,
            None,
            plan.gz_rep,
            plan.key_r,
            plan.key_zl,
            plan.rowtab,
            plan.zids,
            plan.kids,
            kernels=("V", "W"),
            n_rows=U,
            key_index=plan.keys,
            by_group=len(plan.rowtab) > 1,
        )
        self.product.complete = False
        self.product.fast = plan.fast
        # Set by `_FusedEnds.attach`: the tile listener, and the rows its end
        # loops read after their own tile (held beside the sandwich's).
        self.listener = None
        self.extra_last = None
        self.hpos = None
        self.n_held = 0
        self._mark = None  # `_tile_rows`' row mask, made on first use

    def keep_values(self):
        """The unfused route: a (rows, 2) V/W store in row order, which the
        product answers the end loops' lookups from (phase 1)."""
        # Column-major, as the tiles' stores are (`chunks`): read a kernel's
        # column at a time by row (`_gather`, `ProductSet.values_of`).
        self.store = np.empty((self.plan.n_rows, 2), dtype=np.complex128, order="F")
        self.product.vals = self.store
        _ROUTES["tile_stores"] += 1

    def _plan_held(self):
        """`hpos`: each row read AFTER its own tile a slot in the held store,
        written once when its tile runs — the sandwich's late rows (a below
        node whose table column completes at a later tile) and the fused end
        loops' (`extra_last`).

        A many-group product (momwire#1335) RECYCLES the slots: each late
        row is live from its own tile to the last tile that reads it, and a
        slot is reused once its row's last reader has run (`_held_slots`),
        so the store is the most rows live at once rather than every row
        ever held — the fused end loops of razor's inverted L hold a mast's
        whole column of z rows on every node a straddling unit reads, ~10 M
        rows over a fill at x32, but only one tile boundary's worth at a
        time. The slots hold copies of the evaluated floats either way."""
        plan, U = self.plan, self.plan.n_rows
        if _RECYCLE_HELD and plan.slot == "z" and len(plan.rowtab) > 1:
            return self._plan_held_recycled()
        late_row = None
        if self._may_hold and plan.slot == "z" and len(plan.rowtab) > 1:
            # Per group rather than per grid pair: whether row (g, z, key)
            # is read late does not depend on its z (a column reads every
            # z of the group at the same key), so it is decided on the
            # (groups, line) keys and marked over each late key's z column
            # -- the same set the pair loop below marks, at 1/(group size)
            # of its gathers.
            late_row = np.zeros(U, dtype=bool)
            ready = self._node_ready
            for g0, g1 in _group_spans(*plan.kid.shape):
                _cancel.poll()
                late = self.tile_of_key[plan.kid[g0:g1]] < ready[None, :]
                for g in np.flatnonzero(late.any(axis=1)).tolist():
                    kl = np.unique(plan.kl_rank[g0 + g][late[g]])
                    late_row[plan.rowtab[g0 + g][:, kl]] = True
                del late
        elif self._may_hold:
            late_row = np.zeros(U, dtype=bool)
            ready = self._node_ready
            for c0 in range(0, plan.nB, self._step):
                cols = np.arange(c0, min(plan.nB, c0 + self._step))
                t_pair = self.tile_of_key[plan.pair_keys(cols)]
                late = t_pair < ready[cols][None, :]
                if late.any():
                    late_row[plan.chunk_idx(cols)[late]] = True
        n_sandwich = 0 if late_row is None else int(np.count_nonzero(late_row))
        _ROUTES["tile_held_rows"] = max(_ROUTES["tile_held_rows"], n_sandwich)
        extra_rows = (
            None if self.extra_last is None else np.flatnonzero(self.extra_last >= 0)
        )
        if extra_rows is not None and extra_rows.size:
            if late_row is None:
                late_row = np.zeros(U, dtype=bool)
            late_row[extra_rows] = True
            extra = int(np.count_nonzero(late_row)) - n_sandwich
            _ROUTES["fused_held_rows"] = max(_ROUTES["fused_held_rows"], extra)
        if late_row is None:
            return
        self.n_held = int(np.count_nonzero(late_row))
        _ROUTES["held_slots"] = max(_ROUTES["held_slots"], self.n_held)
        self.hpos = np.full(U, -1, dtype=_index_dtype(U))
        self.hpos[late_row] = np.arange(self.n_held)

    def _plan_held_recycled(self):
        """`_plan_held` for a many-group product, slots recycled: every late
        row's LAST reading tile in one per-row array (`_late_last`), then
        `_held_slots` walks the tiles. The sandwich's: group g's key k is
        read late by the line nodes n with `kl_rank[g, n] == k` whose column
        is ready after k's tile, the last at the latest such `ready[n]`, for
        every z of g. The fused loops' come marked (`extra_last`)."""
        plan, U = self.plan, self.plan.n_rows
        ready = self._node_ready
        last = self.extra_last
        n_extra = 0 if last is None else int(np.count_nonzero(last >= 0))
        sand = None
        if self._may_hold:
            if last is None:
                last = _late_last(U, self.n_tiles)
            sand = np.zeros(U, dtype=bool)
            for g0, g1 in _group_spans(*plan.kid.shape):
                _cancel.poll()
                late = self.tile_of_key[plan.kid[g0:g1]] < ready[None, :]
                for g in np.flatnonzero(late.any(axis=1)).tolist():
                    nodes = np.flatnonzero(late[g])
                    kl, inv = np.unique(
                        plan.kl_rank[g0 + g][nodes], return_inverse=True
                    )
                    lk = np.full(kl.size, -1, dtype=np.int64)
                    np.maximum.at(lk, np.asarray(inv).ravel(), ready[nodes])
                    r = plan.rowtab[g0 + g][:, kl]
                    _mark_late(last, r, lk[None, :])
                    sand[r] = True
                del late
        n_sandwich = 0 if sand is None else int(np.count_nonzero(sand))
        _ROUTES["tile_held_rows"] = max(_ROUTES["tile_held_rows"], n_sandwich)
        if self.extra_last is not None:
            n_fused = (
                n_extra if sand is None else int(np.count_nonzero(~sand & (last >= 0)))
            )
            _ROUTES["fused_held_rows"] = max(_ROUTES["fused_held_rows"], n_fused)
        del sand
        if last is None or not np.any(last >= 0):
            return
        self.hpos = np.full(U, -1, dtype=_index_dtype(U))
        self.n_held = _held_slots(self.n_tiles, self._tile_rows, last, self.hpos)
        _ROUTES["held_slots"] = max(_ROUTES["held_slots"], self.n_held)

    def _tile_rows(self, t):
        """Tile t's rows, ascending: every row of its keys, over the groups."""
        by_tile = getattr(self, "_by_tile", None)
        if by_tile is not None:
            # Rows in tile order, ascending within a tile (the sort is
            # stable): the same ids the gather and mask below find.
            o, b = by_tile
            return o[b[t] : b[t + 1]]
        parts = [
            tab[:, o[b[t] : b[t + 1]]].ravel()
            for tab, (o, b) in zip(self.plan.rowtab, self._gkeys)
        ]
        if len(parts) == 1:
            return np.sort(parts[0])
        # The distinct ids ascending, as `np.unique` gave them, by marking
        # them in a row-length mask (momwire#1224): the ids are row numbers in
        # [0, n_rows), so the mask's nonzeros ARE the sorted distinct set, and
        # the tiles share one mask, cleared after each use. np.unique hashed
        # and sorted them instead -- 1.07 s over razor inverted-L x8's 20
        # tiles on Haswell, against a mask pass of O(n_rows) per tile.
        mark = self._mark
        if mark is None:
            mark = self._mark = np.zeros(self.plan.n_rows, dtype=bool)
        for p in parts:
            mark[p] = True
        ids = np.flatnonzero(mark)
        mark[ids] = False
        return ids

    def _evaluate(self, rows):
        """(vals, pos) of one tile's rows (`pos` None: `vals` is in row
        order): ONE call, or — the TEST-ONLY "split" control — two, which
        cuts the tile's columns."""
        ni = _near_interface
        if _PRODUCT_NEG_CONTROL != "split":
            return ni.designed_rows_permuted(
                self.eps_t, self.k_p, rows, rtol=_CROSS_RTOL, sheet_plan=self.sheet_plan
            )
        h = rows.shape[0] // 2
        v1, p1 = ni.designed_rows_permuted(
            self.eps_t, self.k_p, rows[:h], rtol=_CROSS_RTOL, sheet_plan=self.sheet_plan
        )
        v2, p2 = ni.designed_rows_permuted(
            self.eps_t, self.k_p, rows[h:], rtol=_CROSS_RTOL, sheet_plan=self.sheet_plan
        )
        p1 = np.arange(h) if p1 is None else p1
        p2 = np.arange(rows.shape[0] - h) if p2 is None else p2
        return np.concatenate([v1, v2]), np.concatenate([p1, p2 + h])

    def _gather(self, cols, loc, tb, held):
        """The four tables of below columns `cols`: copies of the evaluated
        floats, from this tile's block `tb` (by `loc`) or else the held store
        (by `hpos`) — U and dz′W always, and V and W too on the fused route;
        the unfused route reads V and W from the row-ordered store."""
        if self._zbase is not None and _PRODUCT_NEG_CONTROL != "held":
            hpos = self.hpos
            li, hp_full, sidx = _accel.acc.product_chunk_index(
                self.plan.rowflat,
                self.plan.kl_rank,
                self.plan.grank,
                self._zbase,
                cols,
                loc,
                _EMPTY_I32_1D if hpos is None else hpos,
                self.store is not None,
                _near_interface._physical_cpu_count(),
            )
            return _TileTables(
                sidx if self.store is not None else None,
                li,
                None,
                None,
                tb,
                held,
                self.store,
                hp_full=hp_full,
            )
        idx = self.plan.chunk_idx(cols)
        li = loc[idx]
        miss = li < 0
        hp = None
        if miss.any():
            hp = None if self.hpos is None else self.hpos[idx[miss]]
            if hp is None or (hp < 0).any():
                raise AssertionError("a ready column reads a row not in hand")
            if _PRODUCT_NEG_CONTROL == "held":
                hp = (hp + 1) % self.n_held  # TEST-ONLY: a neighbour's values
        else:
            miss = None
        return _TileTables(idx, li, miss, hp, tb, held, self.store)

    def chunks(self, step):
        """Yield `(cols, K_cols)`: each tile's newly complete below columns
        (ascending within a tile), at most `step` at a time. Fills the V/W
        store as it goes; the product is `complete` once this is exhausted."""
        plan = self.plan
        ki = {k: _near_interface.KEYS.index(k) for k in ("U", "V", "W", "dzpW")}
        U = plan.n_rows
        if self.store is None and self.listener is None:
            raise AssertionError("tiles need either the V/W store or a listener")
        self._plan_held()
        # The tile block's (and the held store's) columns: U, dz′W — or, on
        # the fused route, U, V, W, dz′W (nothing else holds V and W).
        tb_keys = ("U", "dzpW") if self.store is not None else ("U", "V", "W", "dzpW")
        done = np.zeros(U, dtype=bool)
        loc = np.full(U, -1, dtype=_index_dtype(U))  # a row's place in this tile
        # Both stores COLUMN-major (momwire#1224): every read of them is one
        # kernel's column gathered by row (`_gather`, `_FusedEnds._vw`), which
        # from a contiguous column takes numpy's fast 1-D path -- ~1/3 off
        # the gathers at razor's inverted-L x8 -- and they are only ever
        # copied, so the layout moves no bit.
        held = np.empty((self.n_held, len(tb_keys)), dtype=np.complex128, order="F")
        o_ready, b_ready = self._ready
        kcols = [ki[k] for k in tb_keys]
        hpos_i32 = _EMPTY_I32_1D if self.hpos is None else self.hpos
        for t in range(self.n_tiles):
            _cancel.poll()
            ids = self._tile_rows(t)
            # The tile's stores by `tile_block` (the fused route, 32-bit row
            # ids): the copies below, element for element, in one pass.
            fast = (
                _TILE_KERNELS
                and _HAVE_TILE_KERNELS
                and self.store is None
                and ids.dtype == np.int32
                and loc.dtype == np.int32
                and hpos_i32.dtype == np.int32
            )
            if not fast:
                if done[ids].any():
                    raise AssertionError("a product row was asked to evaluate twice")
                done[ids] = True
            _ROUTES["tiles"] += 1
            _ROUTES["tile_rows"] += ids.size
            _ROUTES["tile_max_rows"] = max(_ROUTES["tile_max_rows"], ids.size)
            rows = plan.rows(ids)
            vals, pos = self._evaluate(rows)
            del rows
            if fast:
                tb, twice = _accel.acc.tile_block(
                    vals,
                    _EMPTY_I64_1D if pos is None else pos,
                    kcols,
                    ids,
                    done,
                    loc,
                    hpos_i32,
                    held,
                )
                if twice:
                    raise AssertionError("a product row was asked to evaluate twice")
                _ROUTES["tile_blocks_fast"] += 1
                del vals, pos
            else:
                if pos is not None:
                    vals = vals[pos]
                del pos
                if self.store is not None:
                    self.store[ids, 0] = vals[:, ki["V"]]
                    self.store[ids, 1] = vals[:, ki["W"]]
                tb = np.empty((ids.size, len(tb_keys)), dtype=np.complex128, order="F")
                for j, k in enumerate(tb_keys):
                    tb[:, j] = vals[:, ki[k]]
                del vals
                if self.hpos is not None:
                    hp = self.hpos[ids]
                    keep = hp >= 0
                    held[hp[keep]] = tb[keep]
                    del hp, keep
                loc[ids] = np.arange(ids.size)
            if self.listener is not None:
                # The fused end loops read this tile's V and W (and the held
                # rows') before the sandwich's columns are served.
                self.listener.tile(t, loc, tb, held, self.hpos)
            J = o_ready[b_ready[t] : b_ready[t + 1]]
            for i0 in range(0, J.size, step):
                _cancel.poll()
                cols = J[i0 : i0 + step]
                # Yielded unbound: the consumer drops the tables once it has
                # formed their left products, and a name held here across the
                # yield would keep them alive through the contraction.
                yield cols, self._gather(cols, loc, tb, held)
            if fast:
                _accel.acc.fill_rows_i32(ids, loc, -1)
            else:
                loc[ids] = -1
            del tb
        if not done.all():
            raise AssertionError("the tiles left a product row unevaluated")
        self.product.complete = True


# The main sandwich's left products straight from the tiles' stores
# (`left_products_gathered`, momwire#1224), when the accelerator carries it.
# False forms them by numpy from the gathered tables: the reference the
# kernel is gated against bit for bit (tests/test_left_gather_1224.py).
_LEFT_GATHER = True
_HAVE_LEFT_GATHER_ACCEL = _accel.acc is not None and bool(
    getattr(_accel.acc, "left_gather_1224", False)
)
# A chunk's table indices in one C++ pass (`product_chunk_index`); False is
# `chunk_idx` and numpy's gathers, the reference (the same integers).
_CHUNK_INDEX_ACCEL = True
_HAVE_CHUNK_INDEX_ACCEL = _accel.acc is not None and bool(
    getattr(_accel.acc, "product_chunk_index_1290", False)
)
# A many-group product's per-tile bookkeeping in C++ (momwire#1377): the
# rows' floats (`product_rows`) and, on the fused route, the tile block, the
# held store, the places and the done marks (`tile_block`). Element copies
# only; False is numpy's gathers and scatters, the reference.
_TILE_KERNELS = True
_HAVE_TILE_KERNELS = _accel.acc is not None and bool(
    getattr(_accel.acc, "product_tiles_1377", False)
)
_EMPTY_I64_1D = np.zeros(0, dtype=np.int64)
_EMPTY_I32_1D = np.zeros(0, dtype=np.int32)


class _TileTables:
    """One chunk's four table columns as `_ProductTiles._gather` serves them,
    held as the indices that name them (momwire#1224): `K[key]` gathers that
    kernel's (nA, |cols|) copy exactly as the dict it replaces did -- from the
    tile block `tb` by `li`, the held store at the misses by `hp`, and V and
    W from the row-ordered `store` when the tiles keep one -- and
    `_left_products` hands the same indices to `left_products_gathered`,
    which reads those floats where they are stored instead of copying them
    out first. The indices are taken when the chunk is served, and every
    store they read is written before and never after, so a late read (a
    caller that lists the chunks first) reads what an early one would."""

    __slots__ = ("idx", "li", "miss", "hp", "tb", "held", "store", "hp_full")
    _KEYS = ("U", "V", "W", "dzpW")

    def __init__(self, idx, li, miss, hp, tb, held, store, hp_full=None):
        # `hp_full` (`product_chunk_index`): the held slot of EVERY entry,
        # -1 where the tile holds it, in place of the misses' `miss` / `hp`.
        self.idx, self.li, self.miss, self.hp = idx, li, miss, hp
        self.tb, self.held, self.store = tb, held, store
        self.hp_full = hp_full

    def keys(self):
        return self._KEYS

    def __iter__(self):
        return iter(self._KEYS)

    def __len__(self):
        return len(self._KEYS)

    def __contains__(self, key):
        return key in self._KEYS

    def __getitem__(self, key):
        if self.store is not None and key in ("V", "W"):
            return self.store[self.idx, 0 if key == "V" else 1]
        tb_keys = ("U", "dzpW") if self.store is not None else self._KEYS
        j = tb_keys.index(key)
        out = self.tb[self.li, j]
        if self.hp_full is not None:
            miss = self.li < 0
            out[miss] = self.held[self.hp_full[miss], j]
        elif self.miss is not None:
            out[self.miss] = self.held[self.hp, j]
        return out

    def as_dict(self):
        return {key: self[key] for key in self._KEYS}

    def left_products(self, Ps, k2sq):
        """`_left_products(Ps, self, k2sq)` by the kernel, or None when it
        cannot serve (no kernel, `_LEFT_GATHER` off, 64-bit tile indices)."""
        if not (_LEFT_GATHER and _HAVE_LEFT_GATHER_ACCEL):
            return None
        if self.li.dtype != np.int32:
            return None
        if any(P.data.dtype != np.float64 for P in Ps):
            # A complex basis's samples (the sinusoidal sampler stores its
            # coefficients complex): the kernel's matrices are real.
            return None
        hp = np.zeros((0, 0), dtype=np.int32)
        if self.hp_full is not None:
            hp = self.hp_full
        elif self.miss is not None:
            hp = np.full(self.li.shape, -1, dtype=np.int32)
            hp[self.miss] = self.hp
        if self.store is None:
            kU, kV, kW, kdz = 0, 1, 2, 3
            sidx = np.zeros((0, 0), dtype=np.int64)
            store = np.zeros((0, 0), dtype=np.complex128)
        else:
            kU, kV, kW, kdz = 0, -1, -1, 1
            sidx, store = self.idx, self.store
        return _accel.acc.left_products_gathered(
            [P.indptr for P in Ps],
            [P.indices for P in Ps],
            [P.data for P in Ps],
            Ps[0].shape[0],
            float(k2sq),
            self.li,
            hp,
            self.tb,
            self.held,
            kU,
            kV,
            kW,
            kdz,
            sidx,
            store,
            _near_interface._physical_cpu_count(),
        )


def _whole_tables(K):
    """Whether `K` is one whole-table answer (a dict of the four kernels, or
    the product route's `_TileTables` of every column) rather than a
    sequence of `(cols, K_cols)` chunks."""
    return isinstance(K, (dict, _TileTables))


def _index_dtype(n):
    """int32 when every index below `n` fits it, else the platform's intp."""
    return np.int32 if n <= np.iinfo(np.int32).max else np.intp


def _block_preamble(ctx):
    """`(eps_t, k_p, gz, c1, memo)` for one cross-block fill: the medium's
    above-side ε̃ and k₂, the plane, the moment constant, and a FRESH
    exact-triple memo (momwire#1017: one fill = one memo, ε̃, k₂ and
    `_CROSS_RTOL` fixed for its lifetime). Every cross-block entry opens with
    these five lines, so they are spelled once (momwire#1168 U5).

    The memo is a `ProductMemo` (momwire#1173 design B): a `TripleMemo` until
    `_product_route` stores a main sandwich's factorised rows in it, and the
    same hit / fresh decisions and stored floats as one after."""
    eps_t, _eps_m, k_p, _k_m, _c2, _a_m = ctx.medium
    gz = float(ctx.ground_z)
    c1 = _c1_moment(ctx.omega, ctx.mu)
    return eps_t, k_p, gz, c1, _near_interface.ProductMemo()


# The six keys `point_observer_block` reads (momwire#1224 stage 3 unit 3):
# not `_CROSS_KEYS` (the by-parts blocks' four) -- `dzW` joins them for the
# forward (observers-above) branch's k3.
_POINT_SIX_KEYS = ("U", "V", "W", "dzW", "dzpW")
# `point_observer_block`'s default route (momwire#1173/#1224 design pattern):
# one dedup + one designed-tables/point-family evaluation over the grid's
# unique triples, gathered and contracted per OBSERVER-ROW chunk. False
# takes `_point_observer_block_dense` instead -- the whole-grid reference
# the chunked route is gated against bit for bit
# (tests/test_point_rows_chunked_1224.py).
_POINT_CHUNKED = True
# TEST-ONLY. Deliberately breaks the chunked route's gather by one observer
# row, so the bit-identity gate can be shown to fail -- `_PRODUCT_NEG_
# CONTROL`'s pattern, narrowed to this route.
_POINT_NEG_CONTROL = False
# One observer-row chunk's pair budget. Per pair, the dense per-pair working
# set `point_observer_block` builds is: the five `_POINT_SIX_KEYS` tables
# (80 B) plus the four POINT_KEYS (64 B), both complex128, plus the real
# g/dx/dy/rho and the complex a_grad_v/k3/k4 and the two U·w products (~10
# more arrays, ~150 B) -- ~300 B/pair live at a chunk's peak. 1<<18 pairs is
# ~75 MiB at that rate, in the same neighbourhood as `_MAIN_CHUNK_BYTES`'s
# 64 MiB main-sandwich chunk. Chunking never moves a bit
# (`_chunked_point_tables`), so this is a memory choice only.
_POINT_CHUNK_PAIRS = 1 << 18
# One evaluation batch's row budget in `_chunked_point_tables` (momwire#1267):
# the grid's unique rows go to `designed_rows` and `point_designed_rows` in
# batches of whole ρ_eff columns (`column_batches`) of about this many rows,
# not in one call. The single call's transient was ~400 B per unique row
# above what it returns (lookup block, fresh copy, the twin's member-order
# block and its row-order copy, the memo's sorted insert): +2.1 GB traced at
# invl x32 (5.47 M rows), the fill's peak. 1<<18 rows is ~100 MiB of it,
# beside `_POINT_CHUNK_PAIRS`' ~75 MiB. Batching never moves a bit
# (`column_batches`), so this is a memory choice only.
_POINT_EVAL_ROWS = 1 << 18
# One evaluation TILE's pair budget in `_chunked_point_tables` (momwire#1224
# perf item 5). The values of the grid's unique rows were held whole -- the
# five six-table keys and the four point keys, 144 B per unique row, ~790 MB
# at invl x32 (5.47 M rows) -- because a ρ_eff column must be evaluated in
# ONE call (`column_batches`) and a column's members can sit in any observer
# chunk. Tiles cut that: the observer chunks are grouped into tiles of about
# this many pairs; a column is evaluated whole at the first tile that reads
# any of its members and its values are freed after the last such tile, so
# only the columns live across a tile boundary are held beyond it. Each
# column still sees one call with all its members in their order, so the
# cut never moves a bit (`_PointTileSchedule`); this is a memory choice.
#
# Sized from the census at invl x32 (15.1 M pairs and ~5.45 M unique rows a
# direction; Haswell, 2026-10-04). Pool high-water mark in rows (the reversed
# direction, the larger): 2.03 M at 1<<18, 2.15 M at 1<<20, 2.48 M at 1<<21,
# 3.66 M at 1<<22 -- the floor is the columns a radial-symmetric grid shares
# across every tile (a mast node's ρ_eff to each radial position), held from
# the first tile that reads them to the last, and no budget cuts below it.
# 1<<20 sits on that floor with 31 tiles where 1<<18 takes 121 for 6 % less;
# process peak 1.3 GB against the untiled 2.02 GB.
_POINT_TILE_PAIRS = 1 << 20
# ...but only a grid past this many pairs is tiled at all; one at or under it
# is a single tile, the untiled route exactly. The schedule is not free --
# a hash grouping of the unique ρ_eff, the ends-key lookup and a pass over
# the pair ids, ~0.12 s per million unique rows (Skylake), only partly repaid
# by the `column_batches` sorts it replaces -- and below this the pool it
# saves is small: at invl x16 (3.8 M pairs a direction) an earlier schedule
# cut ~70 MB for 0.45 s of a 7.3 s solve; at x32 (15.1 M) this one cuts
# ~650 MB for ~1 %. The number is the old refusal's, so every deck stage 2
# served is evaluated exactly as before.
_POINT_TILE_MIN_PAIRS = 4_000_000
# TEST-ONLY. Evaluates each row at its OWN first tile instead of its whole
# column's, so a ρ_eff column read in two tiles is split across calls -- the
# cut the schedule exists to avoid, kept so its gate can be shown to fail
# (tests/test_point_rows_tiles_1224.py).
_POINT_TILE_NEG_CONTROL = False


def _point_pair_grid(P, pts, gz, observers_above):
    """`(dx, dy, rho, z, zp)` over the (observers `P` × `pts`) grid: the
    horizontal offsets, their ρ (unfolded), and the z / z′ slots relative to
    ground (broadcast views), observers in z when above, in z′ when below.
    ONE spelling for `_point_kernels_dense` and for the ends' key set the
    chunked route builds from it (`_point_end_keys`), so the key set holds
    the very triples the ends loop asks."""
    dx = P[:, None, 0] - pts[None, :, 0]
    dy = P[:, None, 1] - pts[None, :, 1]
    rho = np.hypot(dx, dy)
    z_o = np.broadcast_to((P[:, 2] - gz)[:, None], rho.shape)
    z_s = np.broadcast_to((pts[:, 2] - gz)[None, :], rho.shape)
    z, zp = (z_o, z_s) if observers_above else (z_s, z_o)
    return dx, dy, rho, z, zp


def _point_end_keys(P, e_pts, gz, a_wire, observers_above):
    """The ends loop's folded (ρ_eff, z, z′) triples as a `TripleMemo.
    key_set` -- every key `_point_kernels_dense` will look up in the
    block's memo over (observers × end points), folded as `_tables` folds
    them (`radius_tables`). Empty when there are no ends (momwire#1267)."""
    if e_pts is None:
        return _near_interface.TripleMemo()
    _dx, _dy, rho, z, zp = _point_pair_grid(P, e_pts, gz, observers_above)
    rows, _inv = _near_interface._unique_rows(
        _near_interface.radius_fold(rho, a_wire), z, zp
    )
    return _near_interface.TripleMemo.key_set(rows)


def _point_kernels_dense(
    ctx, eps_t, k_p, memo, a_wire, P, A, pts, gz, *, observers_above
):
    """`point_observer_block`'s per-pair kernels over the WHOLE (observers ×
    `pts`) grid, dense: the six tables (through `_tables`, so memo'd) and
    the point family (`point_radius_tables`, memo-free -- it dedups only by
    ρ_eff within the call) plus the tangential chain factor g. Shared by
    the ends loop, on both routes (an end span is a handful of points
    against the whole observer axis, never the fill's memory driver so
    never worth chunking), and by `_point_observer_block_dense`, the
    reference the chunked main route is gated against."""
    pts = np.asarray(pts, dtype=float)
    ax, ay = A[:, 0][:, None], A[:, 1][:, None]
    dx, dy, rho, z, zp = _point_pair_grid(P, pts, gz, observers_above)
    six = _tables(ctx, eps_t, k_p, rho, z, zp, _CROSS_RTOL, memo=memo)
    pk = _near_interface.point_radius_tables(
        eps_t, k_p, rho, z, zp, a_wire, plan=None if memo is None else memo.sheet_plan
    )
    g = (ax * dx + ay * dy) / _near_interface.radius_fold(rho, a_wire)
    return six, pk, g


def _point_grad_v(az, pk, g, *, observers_above):
    dz = pk["gzV"] if observers_above else pk["gzpV"]
    return az * dz + g * pk["gRhoV"]


def _chunked_point_tables(
    ctx,
    eps_t,
    k_p,
    P,
    nodes,
    gz,
    observers_above,
    rows,
    memo,
    keep=None,
    on_ready=None,
):
    """`_tables` and the point family (`point_radius_tables`) together over
    `point_observer_block`'s (observers × source-node) grid, served as
    `(sl, six, point)` OBSERVER-ROW chunks — `_chunked_tables`'s precedent
    (momwire#1173) with the axes swapped (momwire#1224 stage 3 unit 3):
    `point_observer_block` contracts `F @ (...).T`, which forms each
    OBSERVER's output COLUMN from that observer's own ROW of the dense
    factor alone (`_chunked_tables`'s argument for `P @ K[:, cols]`,
    transposed: a row slice of the factor is a column slice of its
    transpose, and a sparse-dense product's column c depends on column c of
    the dense side alone). So ROW chunks of this grid are its bit-identical
    cut, where `_main_sandwich`'s were COLUMN chunks of a below axis.

    Both families key on the SAME folded triple (ρ_eff, z, z′): `radius_
    tables` folds ρ → ρ_eff = hypot(ρ, a) and `point_radius_tables` folds it
    the same way (`radius_fold`, one spelling). One dedup pass over the
    grid's exact triples therefore serves both, exactly as `_chunked_tables`
    serves the four cross keys: pass 1 dedups each chunk's folded triples
    and merges them into the grid's unique rows in the grid's OWN first-
    appearance order; ONE evaluation of that list -- `designed_rows` and
    `point_designed_rows`, in whole-column batches (below); pass 2 gathers each chunk's six AND point
    values from the one evaluation by the chunk's own dedup inverse. See
    `_chunked_tables` for why none of this moves a bit -- the point family's
    per-ρ_eff column grouping depends on its members' distinct s = z − z′
    only through the minimum (`point_designed_rows`), exactly as the six
    family's does, so deduplication cannot change which rule a column picks
    there either.

    The grid is never held whole: each chunk's ρ, z and z′ come from
    `_point_pair_grid` over its own observers `P[sl]` against `nodes`
    (elementwise, so the same floats as the whole grid's rows sl), in pass 1
    here and again in the caller's contraction (momwire#1267).

    The ONE evaluation is made in batches (momwire#1267): the unique rows
    go to `designed_rows` and `point_designed_rows` as `column_batches` of
    whole ρ_eff columns of about `_POINT_EVAL_ROWS` rows each, and the
    values are copied into per-key arrays. A batch holds every row of each
    column it touches, in the list's own order, so each call hands every
    column the members the one call did, in the same order -- the same rule
    and the same bits (`column_batches`). Nothing a batch inserts into
    `memo` can be hit by a later batch (the rows are distinct), so each
    batch's hit / fresh split is the one call's restricted to it. `keep`
    (the ends loop's key set, `_point_end_keys`) limits what the memo
    RETAINS to the rows a later call in this block can ask
    (`designed_rows`); None keeps them all, as the one call did.

    The evaluation is TILED (momwire#1224 perf item 5, `_PointTileSchedule`):
    a column is evaluated at the first tile that reads it and its values are
    held only until the last, so the pool holds the rows live across a tile
    boundary rather than the whole grid's. The columns holding any `keep`
    row are pinned to the FIRST tile, and `on_ready` (when given) is called
    once that tile is evaluated, before the first chunk is yielded: by then
    every main-block row the ends loop can ask is in `memo`, exactly as it is
    after the untiled evaluation, so the caller can run its ends loop there
    and finish each chunk's rows as it contracts them."""
    a_wire = float(ctx.a_wire)
    nA, nB = P.shape[0], nodes.shape[0]
    idx_t = _index_dtype(nA * nB)
    uniq, chunk_ids = _point_grid_rows(
        rows, P, nodes, gz, observers_above, a_wire, idx_t
    )
    plan = None if memo is None else memo.sheet_plan
    pin = None if keep is None or not len(keep) else (keep, uniq)
    sched = _PointTileSchedule(uniq[:, 0], chunk_ids, rows, nB, pin=pin)
    del pin
    six_vals = {
        key: np.empty(sched.pool, dtype=np.complex128) for key in _POINT_SIX_KEYS
    }
    point_vals = {
        key: np.empty(sched.pool, dtype=np.complex128)
        for key in _near_interface.POINT_KEYS
    }
    _ROUTES["point_tiles"] += sched.n_tiles
    _ROUTES["point_tile_pool_max_rows"] = max(
        _ROUTES["point_tile_pool_max_rows"], sched.pool
    )
    c0 = 0
    for t in range(sched.n_tiles):
        batches = sched.take(t)
        for b, sel in enumerate(batches):
            _cancel.poll()
            batches[b] = None
            _ROUTES["point_tile_rows"] += sel.size
            sub = uniq[sel]
            dst = sel if sched.slot is None else sched.slot[sel]
            if sched.is_key is None:
                blk = _near_interface.designed_rows(
                    eps_t, k_p, sub, rtol=_CROSS_RTOL, memo=memo, keep=keep
                )
            else:
                blk = _near_interface.designed_rows(
                    eps_t,
                    k_p,
                    sub,
                    rtol=_CROSS_RTOL,
                    memo=memo,
                    keep_mask=sched.is_key[sel],
                )
            for key, v in six_vals.items():
                v[dst] = blk[:, _near_interface.KEYS.index(key)]
            del blk
            _cancel.poll()
            pv = _near_interface.point_designed_rows(eps_t, k_p, sub, plan=plan)
            for key, v in point_vals.items():
                v[dst] = pv[key]
            _ROUTES["point_eval_batches"] += 1
            _ROUTES["point_eval_max_rows"] = max(
                _ROUTES["point_eval_max_rows"], sel.size
            )
            del pv, sub, sel, dst
        del batches
        if t == sched.n_tiles - 1:
            # Every row is evaluated: free the unique list before the last
            # tile's chunks, as the untiled route freed it before its chunk
            # loop.
            del uniq
        if t == 0 and on_ready is not None:
            on_ready()
        for c in range(c0, sched.chunk_end[t]):
            _cancel.poll()
            sl = rows[c]
            idx, chunk_ids[c] = chunk_ids[c], None
            if sched.slot is not None:
                idx = sched.slot[idx]
            idx = idx.reshape(sl.stop - sl.start, nB)
            if _POINT_NEG_CONTROL:
                idx = np.roll(idx, 1, axis=0)
            yield (
                sl,
                {key: v[idx] for key, v in six_vals.items()},
                {key: v[idx] for key, v in point_vals.items()},
            )
        c0 = sched.chunk_end[t]
        sched.release(t)


class _PointTileSchedule:
    """Which unique rows `_chunked_point_tables` evaluates at each tile, in
    which batches, and where in its value pool each lives (momwire#1224 perf
    item 5).

    The observer chunks `rows` are grouped into consecutive TILES of about
    `_POINT_TILE_PAIRS` pairs (a grid of at most `_POINT_TILE_MIN_PAIRS` is
    one tile). A ρ_eff column (the exact-ρ classes
    `column_batches` cuts by: equal under `==`, so −0.0 with 0.0, and NaNs
    together as `np.unique` takes them) is evaluated at the FIRST tile any
    chunk of which reads one of its members, and its slots are released
    after the LAST such tile.

    Why that cannot move a bit. A member's value depends on the call it is
    evaluated in only through its column's membership in that call (the
    smallest s = z − z′ of the column's fresh, non-sheet members picks the
    rule: `six_columns`, `_point_columns_exact`; sheet rows and memo hits
    are functions of the row alone). The untiled route hands every column
    whole to one `column_batches` batch, members ascending; here every batch
    is likewise whole columns, members ascending -- the same membership in
    the same order (`column_batches` documents the order). The memo: a
    batch's rows are distinct from every other batch's (the unique list is
    distinct), so no batch of this block can hit what another inserted, on
    either route, and every hit is the memo state before the block -- which
    batch order cannot change. Nor can the ends loop, which now runs after
    the first tile rather than after the last: every main row it can ask is
    pinned to that tile and so already in the memo, and what it inserts is
    therefore no main row at all. The gather then reads, per pair, the float
    its row was evaluated to, wherever the pool holds it.

    `pin` = (key set, unique rows): the columns holding a unique row the
    ends loop's key set contains are evaluated at tile 0. The lookup is the
    one the untiled route makes anyway -- `designed_rows`' `keep` asks the
    key set about every fresh row -- moved here and made once (`is_key`),
    and the batches hand its answer to `designed_rows` as `keep_mask`
    instead of asking again.

    `pool` is the largest number of rows live at once (a column is live from
    its first tile to its last), so the pool is allocated once at its exact
    high-water mark, and `slot[u]` is row u's position in it. With one tile
    the schedule is the identity -- every row evaluated at tile 0 into its
    own position, `slot` None, batches straight from `column_batches` --
    exactly the untiled route.

    The columns come from ONE hash grouping of the unique ρ_eff
    (`_near_interface._factorize`), and a tile's batches are cut from each
    column's running size without the per-batch `np.unique` `column_batches`
    would pay. Bookkeeping is int32 row ids and int16 tile numbers (the tile
    sorts are then radix sorts): about 16 B per unique row held across the
    walk (each row's id in its tile's evaluation order and in its tile's
    release list, its column, its slot), ~85 MB at invl x32 against the
    ~790 MB of values the tiles stop holding."""

    def __init__(self, rho, chunk_ids, rows, nB, pin=None):
        rho = np.asarray(rho, dtype=float)
        m = int(rho.size)
        n_chunks = len(rows)
        step = max(1, max((sl.stop - sl.start for sl in rows), default=1))
        per = max(1, _POINT_TILE_PAIRS // max(1, step * nB))
        self.n_tiles = max(1, -(-n_chunks // per))
        n_pairs = (rows[-1].stop if rows else 0) * nB
        if n_pairs <= _POINT_TILE_MIN_PAIRS:
            self.n_tiles, per = 1, max(1, n_chunks)
        self.chunk_end = [min(n_chunks, (t + 1) * per) for t in range(self.n_tiles)]
        self.slot = None
        self.is_key = None
        if self.n_tiles == 1:
            self.pool = m
            self._rho = rho
            return
        i32 = np.int32
        tdt = np.int16 if self.n_tiles < np.iinfo(np.int16).max else i32
        tile_of_chunk = (np.arange(n_chunks) // per).astype(tdt)
        # First tile of each row: the rows are numbered in the grid's first-
        # appearance order and the chunks walk the grid in order, so chunk c
        # first sees exactly the ids [hi[c - 1], hi[c]), hi the running
        # (1 + largest id seen).
        hi = np.maximum.accumulate(
            np.array([int(ids.max()) + 1 if ids.size else 0 for ids in chunk_ids])
        )
        first = np.repeat(tile_of_chunk, np.diff(hi, prepend=0))
        # Last tile of each row: the chunks in order, each overwriting.
        last = np.zeros(m, dtype=tdt)
        for cc, ids in enumerate(chunk_ids):
            last[ids] = tile_of_chunk[cc]
        # The columns, numbered in first-appearance order (so a column's
        # number grows with its first member's row id): the hash kernel's
        # exact-equality classes, which are `np.unique`'s on finite floats
        # (−0.0 with 0.0). `np.unique` itself where the kernel cannot serve,
        # or where a NaN would make the two differ (it groups NaNs together,
        # the kernel leaves each alone) -- the number order then follows ρ,
        # which is as good: only whole columns matter, not their order.
        fz = None if np.isnan(rho).any() else _near_interface._factorize([rho])
        if fz is not None:
            col = np.asarray(fz[1]).astype(i32, copy=False)
            n_col = int(np.asarray(fz[0]).size)
        else:
            _u, col = np.unique(rho, return_inverse=True)
            col = np.asarray(col).ravel().astype(i32)
            n_col = int(_u.size)
            del _u
        del fz
        _cancel.poll()
        if pin is not None:
            self.is_key = pin[0].contains(pin[1])
            first[self.is_key] = 0
            _ROUTES["point_pin_rows"] += int(np.count_nonzero(self.is_key))
        # Widen each row's span to its whole column's.
        span = np.full(n_col, self.n_tiles, dtype=first.dtype)
        np.minimum.at(span, col, first)
        if not _POINT_TILE_NEG_CONTROL:
            first = span[col]
        col_tile = span  # each column's evaluation tile
        span = np.zeros(n_col, dtype=last.dtype)
        np.maximum.at(span, col, last)
        if not _POINT_TILE_NEG_CONTROL:
            last = span[col]
        del span
        # Live rows per tile: +1 from a row's evaluation tile to its last.
        live = np.cumsum(
            np.bincount(first, minlength=self.n_tiles + 1)
            - np.bincount(last.astype(i32) + 1, minlength=self.n_tiles + 1)
        )[: self.n_tiles]
        self.pool = int(live.max()) if m else 0
        # Rows by evaluation tile, ids ascending within each (a stable sort
        # of small ints: numpy's radix sort).
        _cancel.poll()
        self._order, self._eval_bounds = self._by_tile(first)
        del first
        _cancel.poll()
        self._free, ends = self._by_tile(last)
        self._free = [self._free[a:b] for a, b in zip(ends[:-1], ends[1:])]
        del last, ends
        # Batches of whole columns: each tile's columns in number order
        # (one more radix sort, over the columns by their tile), cut every
        # `_POINT_EVAL_ROWS` rows of their running size; a row's batch is
        # its column's.
        self._col = col
        _cancel.poll()
        size = np.bincount(col, minlength=n_col)
        by = np.argsort(col_tile, kind="stable")
        run = np.cumsum(size[by])
        cbounds = np.searchsorted(col_tile[by], np.arange(self.n_tiles + 1))
        base = np.concatenate(([0], run))[cbounds[:-1]]
        tile_base = np.repeat(base, np.diff(cbounds))
        batch = np.empty(n_col, dtype=i32)
        batch[by] = (run - size[by] - tile_base) // max(1, _POINT_EVAL_ROWS)
        self._col_batch = batch
        del size, by, run, cbounds, base, tile_base, col_tile
        self.slot = np.full(m, -1, dtype=i32)
        self._stack = np.arange(self.pool - 1, -1, -1, dtype=i32)
        self._top = self.pool

    def _by_tile(self, tile):
        """`(order, bounds)`: row ids grouped by `tile`, ascending within each
        group, group t at `order[bounds[t]:bounds[t + 1]]`."""
        order = np.argsort(tile, kind="stable").astype(np.int32)
        bounds = np.searchsorted(tile[order], np.arange(self.n_tiles + 1))
        return order, bounds

    def take(self, t):
        """Tile t's evaluation batches -- ascending row-id arrays of whole
        columns, about `_POINT_EVAL_ROWS` rows each -- with slots assigned."""
        if self.slot is None:
            rho, self._rho = self._rho, None
            return _near_interface.column_batches(rho, _POINT_EVAL_ROWS)
        a, b = int(self._eval_bounds[t]), int(self._eval_bounds[t + 1])
        ids = self._order[a:b]
        k = ids.size
        # Slots ascending with the ids: the rows are numbered in first-
        # appearance order, so an observer chunk's rows are mostly near one
        # another, and this keeps them near one another in the pool -- the
        # gathers' locality. (Bits do not depend on where a value sits.)
        self.slot[ids] = np.sort(self._stack[self._top - k : self._top])
        self._top -= k
        if k == 0:
            return []
        bt = self._col_batch[self._col[ids]]
        n_b = int(bt.max()) + 1
        bt = bt.astype(np.int16 if n_b < np.iinfo(np.int16).max else np.int32)
        by = np.argsort(bt, kind="stable")  # ids stay ascending in a batch
        cuts = np.searchsorted(bt[by], np.arange(n_b + 1))
        return [
            ids[by[x:y]].astype(np.intp) for x, y in zip(cuts[:-1], cuts[1:]) if y > x
        ]

    def release(self, t):
        """Return the slots of the rows whose last tile is t."""
        if self.slot is None:
            return
        gone, self._free[t] = self._free[t], None
        k = gone.size
        self._stack[self._top : self._top + k] = self.slot[gone]
        self._top += k
        self.slot[gone] = -1


def _point_grid_rows(rows, P, nodes, gz, observers_above, a_wire, idx_t):
    """`(uniq, chunk_ids)`: the (observers × nodes) grid's distinct folded
    triples in the grid's first-appearance order (each as the floats at its
    first appearance), and per observer-row chunk of `rows` each pair's
    index into them, flat. `_chunked_point_tables`' pass 1.

    By `RowGroups` when the hash kernel serves (momwire#1224): the chunks
    are fed in grid order, so its running first-appearance numbering IS the
    grid's, and the merge below has nothing to do. Otherwise each chunk is
    deduplicated alone and the chunks' rows merged in the grid's order; the
    same classes (exact `!=`, NaN alone) numbered the same way, so the same
    integers and rows either way."""
    fold = _near_interface.radius_fold
    nB = nodes.shape[0]
    groups = _near_interface._row_groups()
    if groups is not None:
        chunk_ids = []
        for sl in rows:
            _dx, _dy, rho, z, zp = _point_pair_grid(P[sl], nodes, gz, observers_above)
            r = fold(rho, a_wire)
            del _dx, _dy, rho
            ids = groups.add(
                [
                    r.ravel(),
                    np.broadcast_to(z, r.shape).ravel(),
                    np.broadcast_to(zp, r.shape).ravel(),
                ],
                cancel_flag=_cancel.ptr(),
            )
            chunk_ids.append(ids.astype(idx_t, copy=False))
            del r, z, zp, ids
        return groups.rows(), chunk_ids
    parts, firsts, inverses = [], [], []
    for sl in rows:
        _dx, _dy, rho, z, zp = _point_pair_grid(P[sl], nodes, gz, observers_above)
        r = fold(rho, a_wire)
        del _dx, _dy, rho
        u, inv = _near_interface._unique_rows(r, z, zp)
        # `_unique_rows` numbers groups in first-appearance order, so a
        # group first occurs exactly where the running max of the inverse
        # steps up (as in `_chunked_tables`).
        prev = np.maximum.accumulate(np.concatenate(([-1], inv[:-1])))
        i, j = np.divmod(np.flatnonzero(inv > prev), nB)
        firsts.append((sl.start + i) * nB + j)
        parts.append(u)
        inverses.append((inv.astype(idx_t), u.shape[0]))
        del r, z, zp, u, inv, prev, i, j
    order = np.argsort(np.concatenate(firsts), kind="stable")
    del firsts
    dest = np.empty_like(order)
    dest[order] = np.arange(order.size)
    del order
    triples = np.empty((dest.size, 3), dtype=float)
    off = 0
    for p in range(len(parts)):
        u, parts[p] = parts[p], None
        triples[dest[off : off + u.shape[0]]] = u
        off += u.shape[0]
        del u
    del parts
    uniq, inv_sorted = _near_interface._unique_tri(triples)
    del triples
    gid = inv_sorted.astype(idx_t)[dest]  # chunk-unique row -> grid-unique row
    del dest, inv_sorted
    chunk_ids, off = [], 0
    for c in range(len(inverses)):
        (inv, m), inverses[c] = inverses[c], None
        chunk_ids.append(gid[off : off + m][inv])
        off += m
        del inv
    return uniq, chunk_ids


@_cancelable
def point_observer_block(
    ctx, obs_pts, obs_t, src, *, observers_above, into=None, into_rows=None
):
    """The cross block at POINT observers (momwire#1223 U4): the transmitted
    field of every basis's part on the source axis `src`, tested as t̂·E at
    each observer — the point-matched lane's crossing rows, one per observer.
    Returns (n_obs, src["n_basis"]), c1-scaled like the other cross blocks.

    Nothing is integrated by parts on the observer side; the source side is,
    exactly as the Galerkin trunk does it, so the source charge and the
    source-end terms are the Galerkin ones. Derived from momwire#956's dyad
    (E^V = c1[k²V ẑ − ∇W − ∇(∂z′V)], E^Hx = c1[U x̂ + ∂xW ẑ + ∇(∂xV)]), for
    an observer tangent a and a source tangent b with source current I(s):

    **Forward** (observers ABOVE, in the z slot; source below): with the
    Galerkin trunk's own conversion of a_z(b_h·∇_h W) to a charge term,

        c1 Σ_n [ U (a_h·b_h) + a_z(k²V + ∂z′W) b_z + a_z W I′
                 − (a·∇W) b_z + (a·∇V) I′ ] + c1 Σ_e σ fv [−a_z W − a·∇V]

    **Reversed** (observers BELOW, in the z′ slot; source above), by dyadic
    reciprocity G(o, s) = G(s, o)ᵀ read off the same kernels:

        c1 Σ_n [ U (a_h·b_h) + a_z k²V b_z − (a_h·Δ̂)∂ρW b_z + a_z W I′
                 + (a·∇V) I′ ] + c1 Σ_e σ fv [−a_z W − a·∇V]

    with ∇ taken at the OBSERVER (∂z in the forward block, ∂z′ in the
    reversed; the horizontal part (a_h·Δh)/ρ_eff·∂ρ, Δh = observer − source,
    at the pair's own ρ_eff), `I′` the source charge weight (`Fd`·w), and the
    end sum over the source axis's signed ends. No corner and no test ends:
    both are Galerkin by-parts terms a point test never has (the razor
    precedent, `corner=False` on path-tested rows).

    `_POINT_CHUNKED` (default True, momwire#1224 stage 3 unit 3): the main
    block's tables come from `_chunked_point_tables` -- one dedup and one
    evaluation of the designed tables and the point family over the grid's
    unique (observers × source nodes) triples, gathered and contracted per
    OBSERVER-ROW chunk, at most `_POINT_CHUNK_PAIRS` pairs live per chunk.
    `_POINT_CHUNKED = False` takes `_point_observer_block_dense`, the whole-
    grid call stage 2 shipped, kept as the bit-identity gate's reference.
    Either way the ends loop stays dense (`_point_kernels_dense`): a span of
    a handful of points was never the block's memory driver.

    Plane sheets (momwire#1224 option A): either route first plans the
    block's `SheetPlan` over its (observers × source nodes) pairs
    (`_plan_point_sheets`, the Galerkin fill's `_plan_sheets` rule), so rows
    on a planned plane or height are interpolated for the six keys (through
    the memo) and for the point family (`PlaneSheet(family="point")`) alike,
    and the rest stay on the exact column twins. GATED at ~1e-11 in Z, not
    bit-identical (tests/test_point_sheet_1224.py);
    `MOMWIRE_NEAR_INTERFACE_SHEET=0` is the exact route.

    `into` / `into_rows` (momwire#1224 perf item 5): the block is ADDED to
    `into[into_rows]` (`into_rows` sorted, one per observer) and `into` is
    returned, with no (n_obs, n_basis) block of its own -- the same one
    addition per element as `into[into_rows] += block`. The chunked route
    finishes each observer chunk's rows (ends, c1) as it contracts them; the
    dense reference builds its block and adds it.
    """
    if _POINT_CHUNKED:
        return _point_observer_block_chunked(
            ctx, obs_pts, obs_t, src, observers_above, into, into_rows
        )
    t = _point_observer_block_dense(ctx, obs_pts, obs_t, src, observers_above)
    if into is None:
        return t
    _add_rows(into, into_rows, 0, t)
    return into


def _add_rows(into, into_rows, r0, block):
    """`into[into_rows[r0 : r0 + len(block)]] += block`, by contiguous runs
    of the target rows: the fancy-indexed spelling would gather those rows
    into a copy first (momwire#1267)."""
    sel = np.asarray(into_rows[r0 : r0 + block.shape[0]])
    if sel.size == 0:
        return
    cut = np.flatnonzero(np.diff(sel) != 1) + 1
    r = 0
    for a, b in zip(np.concatenate(([0], cut)), np.concatenate((cut, [sel.size]))):
        s0 = int(sel[a])
        into[s0 : s0 + (b - a)] += block[r : r + (b - a)]
        r += b - a


def _point_observer_block_dense(ctx, obs_pts, obs_t, src, observers_above):
    """`point_observer_block`'s stage-2 route: one dense call of `_point_
    kernels_dense` over the WHOLE (observers × source nodes) grid, no
    chunking, no gather. Kept for `_POINT_CHUNKED = False` and as the
    bit-identity gate's reference (tests/test_point_rows_chunked_1224.py);
    see `point_observer_block` for the physics and the derivation."""
    eps_t, k_p, gz, c1, memo = _block_preamble(ctx)
    _plan_point_sheets(
        ctx, eps_t, k_p, gz, memo, obs_pts, src["nodes"], observers_above
    )
    k2sq = k_p * k_p
    a_wire = float(ctx.a_wire)
    P = np.asarray(obs_pts, dtype=float)
    A = np.asarray(obs_t, dtype=float)
    ax, ay, az = A[:, 0][:, None], A[:, 1][:, None], A[:, 2][:, None]

    six, pk, g = _point_kernels_dense(
        ctx,
        eps_t,
        k_p,
        memo,
        a_wire,
        P,
        A,
        src["nodes"],
        gz,
        observers_above=observers_above,
    )
    U, V, W = six["U"], six["V"], six["W"]
    a_grad_v = _point_grad_v(az, pk, g, observers_above=observers_above)
    if observers_above:
        a_grad_w = az * six["dzW"] + g * pk["gRhoW"]
        k3 = az * (k2sq * V + six["dzpW"]) - a_grad_w
    else:
        k3 = az * k2sq * V - g * pk["gRhoW"]
    k4 = az * W + a_grad_v

    tx, ty, tz = np.asarray(src["t"], dtype=float).T
    w = np.asarray(src["w"])
    F, Fd = src["F_csr"], src["Fd_csr"]
    # Σ_n K[o, n]·weight_n·F[j, n], as F @ (K·weight)ᵀ: the CSR on the left.
    t = (F @ (ax * U * (w * tx)).T) + (F @ (ay * U * (w * ty)).T)
    t = t + (F @ (k3 * (w * tz)).T) + (Fd @ (k4 * w).T)
    t = np.asarray(t).T

    ends = src["ends"]
    if ends:
        e_pts = np.array([pt for pt, _sign, _fv in ends])
        six_e, pk_e, g_e = _point_kernels_dense(
            ctx,
            eps_t,
            k_p,
            memo,
            a_wire,
            P,
            A,
            e_pts,
            gz,
            observers_above=observers_above,
        )
        e_term = -az * six_e["W"] - _point_grad_v(
            az, pk_e, g_e, observers_above=observers_above
        )  # (n_obs, n_ends)
        for i, (_pt, sign, fv) in enumerate(ends):
            nz = np.flatnonzero(fv)
            t[:, nz] += (sign * e_term[:, i])[:, None] * fv[nz][None, :]
    return c1 * t


def _point_observer_block_chunked(
    ctx, obs_pts, obs_t, src, observers_above, into=None, into_rows=None
):
    """`point_observer_block`'s default route (momwire#1224 stage 3 unit 3):
    the main block through `_chunked_point_tables`, contracted per observer
    chunk (never the whole (observers × source nodes) grid's tables at
    once); the ends loop stays `_point_kernels_dense`, exactly as the dense
    route's, sharing the SAME six-table memo -- so the ends see the same
    hits/misses either route takes, per `_block_preamble`'s "one fill, one
    memo". The block's sheet plan is decided first (`_plan_point_sheets`),
    so both routes serve the same rows from sheets.

    The ends loop runs once the FIRST evaluation tile is in (`on_ready`):
    `_chunked_point_tables` pins every main-block row the ends can ask to
    that tile, so the memo the ends read is the untiled route's (momwire
    #1224 perf item 5). Each chunk's rows are then finished as they are
    contracted -- the ends' terms added in end order, then the c1 scale --
    which is, per element, the sequence the whole block went through
    (`t[:, nz] += ...` per end, then `t *= c1`): the same operations on the
    same floats, so the same bits, without the (n_obs, n_basis) block when
    `into` is given."""
    eps_t, k_p, gz, c1, memo = _block_preamble(ctx)
    _plan_point_sheets(
        ctx, eps_t, k_p, gz, memo, obs_pts, src["nodes"], observers_above
    )
    k2sq = k_p * k_p
    a_wire = float(ctx.a_wire)
    P = np.asarray(obs_pts, dtype=float)
    A = np.asarray(obs_t, dtype=float)
    ax_full, ay_full, az_full = A[:, 0][:, None], A[:, 1][:, None], A[:, 2][:, None]

    nodes = np.asarray(src["nodes"], dtype=float)
    nA, nB = P.shape[0], nodes.shape[0]

    tx, ty, tz = np.asarray(src["t"], dtype=float).T
    w = np.asarray(src["w"])
    F, Fd = src["F_csr"], src["Fd_csr"]
    ends = src["ends"]
    e_pts = np.array([pt for pt, _sign, _fv in ends]) if ends else None
    # Only the ends loop reads the memo after the main block, so the main
    # block's rows it retains are the ends' keys alone (momwire#1267).
    keep = _point_end_keys(P, e_pts, gz, a_wire, observers_above)
    end_terms = []

    def ends_loop():
        if not ends:
            return
        six_e, pk_e, g_e = _point_kernels_dense(
            ctx,
            eps_t,
            k_p,
            memo,
            a_wire,
            P,
            A,
            e_pts,
            gz,
            observers_above=observers_above,
        )
        e_term = -az_full * six_e["W"] - _point_grad_v(
            az_full, pk_e, g_e, observers_above=observers_above
        )  # (n_obs, n_ends)
        for i, (_pt, sign, fv) in enumerate(ends):
            nz = np.flatnonzero(fv)
            end_terms.append((nz, sign * e_term[:, i], fv[nz][None, :]))

    step = max(1, _POINT_CHUNK_PAIRS // max(1, nB))
    row_slices = [slice(r0, min(nA, r0 + step)) for r0 in range(0, nA, step)]
    # Without `into`: the block, filled a chunk of rows at a time. Allocated
    # at the first chunk, after the evaluation's transients are gone.
    t = None
    buf = None
    n_chunks = 0
    for sl, six, pk in _chunked_point_tables(
        ctx,
        eps_t,
        k_p,
        P,
        nodes,
        gz,
        observers_above,
        row_slices,
        memo,
        keep,
        on_ready=ends_loop,
    ):
        _cancel.poll()
        n_chunks += 1
        axc, ayc, azc = ax_full[sl], ay_full[sl], az_full[sl]
        dx, dy, rho, _z, _zp = _point_pair_grid(P[sl], nodes, gz, observers_above)
        g = (axc * dx + ayc * dy) / _near_interface.radius_fold(rho, a_wire)
        del dx, dy, rho, _z, _zp
        U, V, W = six["U"], six["V"], six["W"]
        a_grad_v = _point_grad_v(azc, pk, g, observers_above=observers_above)
        if observers_above:
            a_grad_w = azc * six["dzW"] + g * pk["gRhoW"]
            k3 = azc * (k2sq * V + six["dzpW"]) - a_grad_w
        else:
            k3 = azc * k2sq * V - g * pk["gRhoW"]
        k4 = azc * W + a_grad_v
        tc = (F @ (axc * U * (w * tx)).T) + (F @ (ayc * U * (w * ty)).T)
        tc = tc + (F @ (k3 * (w * tz)).T) + (Fd @ (k4 * w).T)
        # No `del` of this chunk's temporaries here: they are freed when the
        # next chunk rebinds them, AFTER its own are allocated, so the heap
        # top stays in use and glibc never trims it between chunks. Freeing
        # them first (tried) trimmed and re-faulted ~20 MB a chunk: 3x the
        # page faults and +3 % wall at buried x16, all allocator, none
        # arithmetic (with MALLOC_TRIM_THRESHOLD_ pinned the two tie).
        #
        # The chunk's rows, (rows, n_basis), copied into the block or into
        # one chunk buffer reused across chunks, then finished in place:
        # each end's term in end order, then c1 -- the whole block's
        # per-element order.
        if into is not None:
            if buf is None:
                buf = np.empty((src["n_basis"], step), dtype=np.complex128).T
            rc = buf[: sl.stop - sl.start]
        else:
            if t is None:
                t = np.empty((src["n_basis"], nA), dtype=np.complex128).T
            rc = t[sl]
        rc[...] = np.asarray(tc).T
        for nz, se, fvz in end_terms:
            rc[:, nz] += se[sl][:, None] * fvz
        rc *= c1
        if into is not None:
            _add_rows(into, into_rows, sl.start, rc)
    _ROUTES["point_chunked"] += 1
    _ROUTES["point_chunks"] += n_chunks
    _ROUTES["point_chunk_max_rows"] = max(
        _ROUTES["point_chunk_max_rows"], min(step, nA)
    )
    if into is not None:
        return into
    if t is None:
        t = np.zeros((nA, src["n_basis"]), dtype=np.complex128)
    return t


@_cancelable
def cross_complete_block(ctx, A, B, *, corner=True, support=None, into=None):
    """t_ab = M + SW + SQ + BT + CORNER over (above axis A × below axis B),
    on designed kernels. Returns the full (n_basis, n_basis) block in the
    subtracting field-block convention (`Z -= t_ab`).

    For GALERKIN rows the opposite block is this one's transpose. For
    PATH-tested rows it is not, and `cross_complete_block_reversed` builds
    it — see there for the one term that separates them.

    `support=(rows, cols)` (momwire#1173 design B) answers
    `t_ab[np.ix_(rows, cols)]` instead, never allocating the full block —
    for a caller that reads only that (razor's crossing assembly, where the
    full block is 92 % structural zeros). Bit-identical to that slice: see
    `_Support`.

    `into` (momwire#1173 design C phase 2, with `support`) is the matrix the
    caller folds the block into as `into[np.ix_(rows, cols)] -= t_ab`. When
    the block streams (`_FusedEnds`) each finished column is folded there
    as it finishes and None is returned; otherwise the block is returned
    and the caller folds it."""
    # One fill = one memo, exactly as `cross_complete_block_split` does it
    # (momwire#1017). This route built none, so momwire#688's cross-call dedup
    # — the whole reason the parameter exists — never fired for `RazorSolver`,
    # whose crossing serve calls straight in here.
    eps_t, k_p, gz, c1, memo = _block_preamble(ctx)
    sup = _Support.of(support, A["n_basis"], B["n_basis"])
    # The end loops go to the product's tiles when they can (`_FusedEnds`,
    # momwire#1173 design C phase 2) and run after the sandwich otherwise —
    # `_ends_and_corner` with the forward block's arguments either way.
    ends = _FusedEnds(
        ctx,
        A,
        B,
        eps_t,
        k_p,
        c1,
        gz,
        memo,
        row_args=_above_end_args(B, gz),
        col_args=_below_end_args(A, gz),
        corner=corner,
        support=sup,
        units_on="cols",
        into=into,
    )
    t_ab = _main_sandwich(
        ctx, A, B, eps_t, k_p, c1, gz, memo=memo, support=sup, ends=ends
    )
    return ends.finish(t_ab)


class _Support(NamedTuple):
    """The (rows, cols) sub-block a cross-block caller reads (momwire#1173
    design B), with each full index's compact position (−1 off it).

    Why a compact answer is the slice of the full one to the bit: every
    write into the full block is elementwise per entry — an assignment of
    a computed block, an `+=` of one, a scalar `*=` — and each such write
    has its compact twin at `(pos_r[i], pos_c[j])` carrying the same value
    (`_put` gathers the same block entries; the rank-1 updates form their
    outer products elementwise, so restricting a factor restricts the
    product and nothing else). An entry therefore sees the same writes in
    the same order, from the same zero; entries off the support are never
    formed, and the caller never read them."""

    rows: np.ndarray
    cols: np.ndarray
    pos_r: np.ndarray
    pos_c: np.ndarray

    @classmethod
    def of(cls, support, n_r, n_c):
        if support is None:
            return None
        rows, cols = (np.asarray(x, dtype=np.int64) for x in support)
        pos_r, pos_c = _positions(n_r, rows), _positions(n_c, cols)
        return cls(rows, cols, pos_r, pos_c)

    def transposed(self):
        return _Support(self.cols, self.rows, self.pos_c, self.pos_r)

    def zeros(self):
        return np.zeros((self.rows.size, self.cols.size), dtype=np.complex128)


def _put(out, r, c, block, support, *, assign):
    """`out[np.ix_(r, c)] = block` (or `+=`), through `support`'s positions
    when it is given: the entries of `block` whose row and column the support
    keeps, into their compact places."""
    if support is not None:
        pr, pc = support.pos_r[r], support.pos_c[c]
        kr, kc = pr >= 0, pc >= 0
        if not (kr.all() and kc.all()):
            block = block[np.ix_(kr, kc)]
            pr, pc = pr[kr], pc[kc]
        r, c = pr, pc
    ix = np.ix_(r, c)
    if assign:
        out[ix] = block
    else:
        out[ix] += block


def _real_matvec_c(M, v):
    """`M @ v` for a REAL matrix and a COMPLEX vector, without upcasting M.

    momwire#919, and the largest single term at the peak once the weight fold
    and the rank-1 buffer were in. `np.matmul` promotes both operands to a
    common dtype, so a float64 (n_basis, n_nodes) matrix against a complex
    vector materialises a COMPLEX COPY OF THE MATRIX first — 445 MB on the
    48-radial screen, to produce a 43 kB answer. Two real matvecs are the same
    flops and allocate nothing but the result.

    Invisible in the source it replaces, which reads as a matrix-vector
    product and is one; only the dtypes give it away.

    `M` is the axis's CSR since momwire#1109 and the argument is unchanged:
    scipy upcasts a real CSR against a complex vector the same way numpy
    upcasts a real array, and the copy it makes is of the stored values
    rather than of the dense block, which is smaller but still pointless.
    """
    if (
        hasattr(M, "indptr")
        and v.ndim == 1
        and v.dtype == np.complex128
        and _end_matvecs_serve(M, None)
    ):
        # Through `end_matvecs` whenever it serves (momwire#1224), so every
        # end vector of every route -- the tiles', the stores', a slow end's
        # gathered table, the grid route's -- is ONE loop's sum and the
        # routes agree to the bit by construction on any build. (scipy's own
        # product is that sum to the bit only where its build does not
        # contract a*x + acc; arm64 clang's does.) The vector is the store,
        # read in order, unweighted.
        n = v.shape[0]
        return _accel.acc.end_matvecs(
            M.indptr,
            M.indices,
            M.data,
            M.shape[0],
            np.ones(n),
            _EMPTY_I32,
            _EMPTY_I32,
            _EMPTY_C,
            _EMPTY_C,
            0,
            np.arange(n, dtype=np.int64)[None, :],
            v[:, None],
            0,
            1,
        )[0]
    return _real_matvec_c_numpy(M, v)


def _real_matvec_c_numpy(M, v):
    """`_real_matvec_c` by scipy: the reference the kernel is gated against
    (to the bit on x86-64, within its summation-order bound elsewhere)."""
    return (M @ v.real) + 1j * (M @ v.imag)


# The end loops' matvecs read straight from the product's stores
# (`end_matvecs`, momwire#1224) when the accelerator carries it. False forms
# each end's V/W vector and `_real_matvec_c` of it, the reference the kernel
# is gated against bit for bit (tests/test_end_matvecs_1224.py).
_END_MATVECS = True
_HAVE_END_MATVECS_ACCEL = _accel.acc is not None and bool(
    getattr(_accel.acc, "end_matvecs_1224", False)
)
# One `end_matvecs` call's (end, node) pairs: its index arrays and its
# answer are ~16 B and ~16 B per pair of line and of basis; 1 M pairs keeps
# a batch at the main sandwich's chunk scale. Batching never moves a bit
# (each end's vector is its own sum), so this is a memory choice only.
_END_VEC_PAIRS = 1 << 20
_EMPTY_I32 = np.zeros((0, 0), dtype=np.int32)
_EMPTY_I64 = np.zeros((0, 0), dtype=np.int64)
_EMPTY_C = np.zeros((0, 0), dtype=np.complex128)


def _end_matvecs_serve(M, w):
    """Whether `end_matvecs` can serve this matrix and weighting: its
    operands are REAL (a complex basis's samples, the sinusoidal sampler's,
    take the numpy form)."""
    return (
        _END_MATVECS
        and _HAVE_END_MATVECS_ACCEL
        and M.data.dtype == np.float64
        and (w is None or np.asarray(w).dtype == np.float64)
    )


# `_tile_matvecs` hands the rows to `end_matvecs_rows` (momwire#1335); False
# gathers loc / hpos in numpy and calls `end_matvecs`, the reference.
_TILE_MATVECS_ROWS = True
_HAVE_END_MATVECS_ROWS_ACCEL = _accel.acc is not None and bool(
    getattr(_accel.acc, "end_matvecs_rows_1335", False)
)


# The fused ends' rows formed inside `end_matvecs_table` (momwire#1335);
# False forms them in numpy (`_fast_desc_rows_batch`, the grouped gathers)
# and hands them to `_tile_matvecs`, the reference.
_TABLE_MATVECS = True
_HAVE_END_MATVECS_TABLE_ACCEL = _accel.acc is not None and bool(
    getattr(_accel.acc, "end_matvecs_table_1335", False)
)


def _table_serves(M, w, fast, loc, hpos):
    """Whether `end_matvecs_table` serves these operands."""
    return (
        _TABLE_MATVECS
        and _HAVE_END_MATVECS_TABLE_ACCEL
        and _end_matvecs_serve(M, w)
        and loc.dtype == np.int32
        and (hpos is None or hpos.dtype == np.int32)
        and fast.rowflat.dtype == np.int32
        and fast.kl_rank.dtype in (np.int32, np.int64)
    )


def _desc_matvecs(fast, descs, ax, w, w_tz, loc, tb, held, hpos):
    """`(vVs, vWs)`: `_tile_matvecs` of `ax`'s Fd and F against the rows the
    ends `descs` name (`_fast_desc_rows_batch`). "line" descs go to
    `end_matvecs_table` with the rows formed in the kernel: row (e, m) is
    `rowflat[off[grank[m]] + zl_rank[m] * nk[grank[m]] + kg_e[grank[m]]]`,
    the batch spelling's expression, so the same rows and the same sums."""
    Fd, F = ax["Fd_csr"], ax["F_csr"]
    if (
        descs
        and all(d[0] == "line" for d in descs)
        and _table_serves(Fd, w, fast, loc, hpos)
        and _table_serves(F, w_tz, fast, loc, hpos)
    ):
        grk = np.asarray(fast.grank, dtype=np.int64)
        base_j = (fast.off[grk] + fast.zl_rank * fast.nk[grk]).astype(np.int64)
        KG = np.ascontiguousarray(np.stack([d[1] for d in descs]))
        if KG.dtype not in (np.int32, np.int64):
            KG = KG.astype(np.int64)
        rsel = np.arange(len(descs), dtype=np.int64)
        base_e = np.zeros(len(descs), dtype=np.int64)
        out = []
        for M, x, k in ((Fd, w, 1), (F, w_tz, 2)):
            out.append(
                _accel.acc.end_matvecs_table(
                    M.indptr,
                    M.indices,
                    M.data,
                    M.shape[0],
                    x,
                    base_e,
                    base_j,
                    KG,
                    rsel,
                    grk,
                    fast.rowflat,
                    loc,
                    _EMPTY_I32_1D if hpos is None else hpos,
                    tb,
                    held,
                    k,
                    _near_interface._physical_cpu_count(),
                )
            )
        return out[0], out[1]
    R = _fast_desc_rows_batch(fast, descs)
    vVs = _tile_matvecs(Fd, w, R, 1, loc, tb, held, hpos)
    vWs = _tile_matvecs(F, w_tz, R, 2, loc, tb, held, hpos)
    return vVs, vWs


def _tile_matvecs(M, w, R, k, loc, tb, held, hpos):
    """`_real_matvec_c(M, w * X[e])` for each row e of `R` (E product rows
    of n nodes each), X being the tile's kernel column `k` at those rows as
    `_FusedEnds._vw` reads it -- the tile block by `loc`, else the held
    store by `hpos`; (E, M.shape[0]) complex. By `end_matvecs` when it
    serves, else the per-end numpy spelling, the same floats. Where the
    accelerator carries `end_matvecs_rows` (momwire#1335) the rows go to it
    whole and the loc / hpos gather is done per element in C++: the same
    sums, without the two (E, n) index arrays."""
    R = np.asarray(R)
    if (
        _TILE_MATVECS_ROWS
        and _HAVE_END_MATVECS_ROWS_ACCEL
        and _end_matvecs_serve(M, w)
        and loc.dtype == np.int32
        and (hpos is None or hpos.dtype == np.int32)
        and R.dtype in (np.int32, np.int64)
    ):
        try:
            return _accel.acc.end_matvecs_rows(
                M.indptr,
                M.indices,
                M.data,
                M.shape[0],
                w,
                np.ascontiguousarray(R),
                loc,
                _EMPTY_I32_1D if hpos is None else hpos,
                tb,
                held,
                k,
                _near_interface._physical_cpu_count(),
            )
        except RuntimeError as exc:
            if "not in hand" in str(exc):
                raise AssertionError("a fused end reads a row not in hand") from None
            raise
    li = loc[R]
    miss = li < 0
    hp = _EMPTY_I32
    if miss.any():
        hpm = None if hpos is None else hpos[R[miss]]
        if hpm is None or (hpm < 0).any():
            raise AssertionError("a fused end reads a row not in hand")
        hp = np.full(R.shape, -1, dtype=np.int32)
        hp[miss] = hpm
    if (
        _end_matvecs_serve(M, w)
        and li.dtype == np.int32
        and (hp.size == 0 or hpos.dtype == np.int32)
    ):
        return _accel.acc.end_matvecs(
            M.indptr,
            M.indices,
            M.data,
            M.shape[0],
            w,
            li,
            hp,
            tb,
            held,
            k,
            _EMPTY_I64,
            _EMPTY_C,
            -1,
            _near_interface._physical_cpu_count(),
        )
    out = np.empty((R.shape[0], M.shape[0]), dtype=np.complex128)
    for e in range(R.shape[0]):
        X = tb[li[e], k]
        if miss[e].any():
            X[miss[e]] = held[hp[e][miss[e]], k]
        out[e] = _real_matvec_c(M, w * X)
    return out


def _store_matvecs(M, w, R, store, col):
    """`_real_matvec_c(M, w * store[R[e], col])` for each row e of `R`
    (product value rows), (E, M.shape[0]) complex: `ProductSet.values_of`'s
    gather and the matvec, by `end_matvecs` when it serves."""
    R = np.asarray(R, dtype=np.int64)
    if _end_matvecs_serve(M, w):
        return _accel.acc.end_matvecs(
            M.indptr,
            M.indices,
            M.data,
            M.shape[0],
            w,
            _EMPTY_I32,
            _EMPTY_I32,
            _EMPTY_C,
            _EMPTY_C,
            0,
            R,
            store,
            col,
            _near_interface._physical_cpu_count(),
        )
    out = np.empty((R.shape[0], M.shape[0]), dtype=np.complex128)
    for e in range(R.shape[0]):
        out[e] = _real_matvec_c(M, w * store[R[e], col])
    return out


# `_end_vectors`' stand-in for a fast end's tables (`fast_te`): its vectors
# are formed from the store by row, never from a gathered table.
_FAST_TE = object()


def _end_vectors(ctx, eps_t, k_p, ends, ax, w, w_tz, memo, args):
    """Yield `(sign, fv, vV, vW)` per end of `ends`, in order: the two
    matvecs of the end's V and W tables over `ax`'s nodes, vV =
    `_real_matvec_c(ax["Fd_csr"], w * V)` and vW = `_real_matvec_c(ax["F_csr"],
    w_tz * W)`, as `_row_end_terms` / `_col_end_terms` form them.

    The tables are `_end_tables`'. Over a product that keeps its V/W store
    (the unfused route, momwire#1224) the fast ends' vectors are formed from
    the store by row in batches (`_store_matvecs`: the floats
    `_end_tables_product` would have gathered, summed as the matvec sums
    them), and the slow ends' tables come from the same span calls, in the
    same order, through `fast_te`."""
    n_nodes = ax["nodes"].shape[0]
    Fd, F = ax["Fd_csr"], ax["F_csr"]
    product = getattr(memo, "product", None)
    if not (
        product is not None
        and _PRODUCT_ENDS
        and product.fast is not None
        and product.vals is not None
        and _PRODUCT_NEG_CONTROL != "row"
        and _END_MATVECS
        and _HAVE_END_MATVECS_ACCEL
    ):
        for _pt, sign, fv, te in _end_tables(
            ctx, eps_t, k_p, ends, n_nodes, memo, args
        ):
            _cancel.poll()
            yield (
                sign,
                fv,
                _real_matvec_c(Fd, w * te["V"]),
                _real_matvec_c(F, w_tz * te["W"]),
            )
        return
    fast = product.fast
    classes, on_node = _classify_ends(fast, float(ctx.a_wire), ends, args)
    _ROUTES["ends_line_on_node"] += int(sum(on_node))
    per = max(1, _END_VEC_PAIRS // max(1, n_nodes))
    pending = []

    def flush():
        _cancel.poll()
        R = _fast_desc_rows_batch(fast, [classes[i] for i, _s, _f in pending])
        vVs = _store_matvecs(Fd, w, R, product.vals, 0)
        vWs = _store_matvecs(F, w_tz, R, product.vals, 1)
        out = [(s_, f_, vVs[e], vWs[e]) for e, (_i, s_, f_) in enumerate(pending)]
        pending.clear()
        return out

    spans = _end_tables_product(
        ctx,
        eps_t,
        k_p,
        ends,
        n_nodes,
        memo,
        args,
        classes=classes,
        fast_te=lambda _i: _FAST_TE,
    )
    for i, (_pt, sign, fv, te) in enumerate(spans):
        _cancel.poll()
        if te is _FAST_TE:
            pending.append((i, sign, fv))
            if len(pending) >= per:
                yield from flush()
            continue
        if pending:
            yield from flush()
        yield (
            sign,
            fv,
            _real_matvec_c(Fd, w * te["V"]),
            _real_matvec_c(F, w_tz * te["W"]),
        )
    if pending:
        yield from flush()


def _polled_nonzeros(ends):
    """`[np.flatnonzero(fv) for _p, _s, fv in ends]`, polling every 256 ends:
    at x32 the whole list is ~0.15 s."""
    out = []
    for i, (_p, _s, fv) in enumerate(ends):
        if not i & 255:
            _cancel.poll()
        out.append(np.flatnonzero(fv))
    return out


def _vec_batches(items, n):
    """`items` cut into lists of at most `_END_VEC_PAIRS` (item, node) pairs
    against a line of `n` nodes, one item at least."""
    per = max(1, _END_VEC_PAIRS // max(1, int(n)))
    return [items[i : i + per] for i in range(0, len(items), per)]


class _Rank1Buffer:
    """One reusable scratch array for the by-parts rank-1 updates
    (momwire#919).

    THE PEAK IS A MOMENT, NOT A SET OF ARRAYS. Whoever measures this next:
    freeing memory that is live at some *other* phase moves nothing. The
    duplicate coarse F/Fd pair on the 48-radial screen is a genuine 445 MB and
    removing it changed the peak by zero, because the high-water is reached
    here and it simply relocated. Equally, an RSS-by-region trace says where
    the program WAS at the high-water, not what allocated it — it reads like
    attribution and is not. Use `tracemalloc` grouped by traceback, and
    snapshot AT the peak: at 87 % of peak this routine does not even appear,
    at 100 % it is the top term.

    What it replaces: `t_ab[nz] += c1 * sign * np.outer(a, b)` builds the outer
    product, then the sign scaling, then the c1 scaling, then the gather, then
    the sum — five temporaries in flight for one update. Each is SMALL here:
    `fv` really is the "handful of nonzeros" #912 describes, `nz` measuring 1
    or 2 even at 48 radials. This buffer is worth its lines for the 197 calls,
    not for any one of them.

    It is NOT where the 445 MB at the old peak came from — that was
    `_real_matvec_c`'s upcast, on the same source line, and tracemalloc
    attributes a numpy temporary to the Python line that triggered it. Reading
    that number as "the outer products are huge" is the mistake this paragraph
    exists to stop; it cost one wrong hypothesis before the shapes were
    printed.
    """

    __slots__ = ("_buf",)

    def __init__(self):
        self._buf = None

    def take(self, rows, cols):
        need = rows * cols
        if self._buf is None or self._buf.size < need:
            self._buf = np.empty(need, dtype=np.complex128)
        return self._buf[:need].reshape(rows, cols)


def _rank1_add(t_ab, nz, a, b, scale, buf):
    """`t_ab[nz] += scale * outer(a, b)` through one reused buffer."""
    if nz.size == 0:
        return
    out = buf.take(nz.size, b.size)
    np.multiply.outer(a, b, out=out)
    out *= scale
    t_ab[nz] += out


# `_FusedEnds._finish_cols` applies its vector loop's rank-1 updates in one
# `_rank1_adds` (momwire#1335: at razor's inverted L its ~1,500 grouped ends
# times every finish batch were ~110 k `_rank1_add` calls). False is the
# per-term loop, the reference.
_BATCHED_VECTOR_ADDS = True


def _rank1_adds(t_ab, terms):
    """`_rank1_add(t_ab, nz, a, b, scale, ...)` for each `(nz, a, b, scale)`
    of `terms`, in order, as one pass. Each update's entries are formed as
    `_rank1_add` forms them -- the row `a[r] * b` (`multiply.outer`'s
    broadcast of a real factor against a complex row), then `*= scale` -- and
    `np.add.at` adds them unbuffered in term order, so an entry several terms
    write sees the same additions in the same order from the same start."""
    if not terms:
        return
    counts = np.fromiter((t[0].size for t in terms), dtype=np.int64, count=len(terms))
    rows = np.concatenate([t[0] for t in terms])
    a = np.concatenate([t[1] for t in terms])
    B = np.repeat(np.stack([t[2] for t in terms]), counts, axis=0)
    S = np.repeat(np.array([t[3] for t in terms], dtype=np.complex128), counts)
    C = np.multiply(a[:, None], B)
    del B
    C *= S[:, None]
    np.add.at(t_ab, rows, C)


def _rank1_add_cols(t_ab, nz, a, b, scale, buf):
    """`t_ab[:, nz] += scale * outer(a, b)` through one reused buffer."""
    if nz.size == 0:
        return
    out = buf.take(a.size, nz.size)
    np.multiply.outer(a, b, out=out)
    out *= scale
    t_ab[:, nz] += out


# Two in-plane ends closer than this share a crossing node: the tolerance the
# one-node fill asserted. Real crossing nodes stand metres apart
# (`_below_interface.MIN_CROSSING_NODE_SEPARATION_M`).
_SAME_NODE_RHO = 1e-9


def _end_live_rows(ax):
    """The basis rows an axis's END TABLE can touch — the support of every
    by-parts end term the axis contributes (momwire#1029 phase 2 unit C).

    A wire end touches only the basis functions whose support reaches it, so
    the end shape is confined to these rows on one side and these columns on
    the other, exactly as `_bnd_and_corner` (momwire#914) already reads its
    own ends. Measured 317 of 1278 on the N = 113 BLE below-axis, and a far
    smaller fraction on a screen whose radials carry many segments each.
    """
    ends = ax["ends"]
    if not ends:
        return np.zeros(0, dtype=np.int64)
    return np.unique(np.concatenate([np.flatnonzero(e[2]) for e in ends]))


def _positions(n, idx):
    """`pos[i]` = where global index `i` sits in `idx` (-1 off it)."""
    pos = np.full(n, -1, dtype=np.int64)
    pos[idx] = np.arange(idx.size)
    return pos


def _corner_v(cache, eps_t, k_p, a_wire, rho):
    """The corner's V at the regularized end separation, once per distinct ρ.

    A pair at one node (ρ < `_SAME_NODE_RHO`) reads V(a) through the call the
    one-node fill made, so a single-node deck keeps its bytes. A pair across
    two nodes reads V(√(ρ² + a²)) — the transmitted partner of the
    point-charge pairs the self completions already carry at √(‖Δr‖² + a²).
    Measured on the two-node ε̃ = 1 collapse (antennaknobs plan U9 (b), momwire
    scratch/u9-multi-crossing): with it the Z matrix matches the free-space
    two-wire truth to 2.0e-4 Ω at 12 m and at 1 m; without it Z12 is 4.70 Ω
    off, and an orientation-blind sign 9.39 Ω.
    """
    key = 0.0 if rho < _SAME_NODE_RHO else rho
    v = cache.get(key)
    if v is None:
        r_eff = a_wire if key == 0.0 else float(np.hypot(rho, a_wire))
        v = complex(
            _near_interface.six_point(eps_t, k_p, r_eff, 0.0, 0.0, rtol=_CORNER_RTOL)[1]
        )
        cache[key] = v
    return v


def _above_end_args(line, gz):
    """`args(pt)` for `_end_tables`: an ABOVE end point against a BELOW
    line's quadrature nodes — the end in the `z` slot, the line in `z′`.

    Both sides go through `_on_plane_side` (momwire#852), so an end or a node
    on the wrong side of the plane is snapped inside `_PLANE_TOL` and refused
    past it. The forward block's test ends and the reversed block's source
    ends are this one spelling since momwire#1168 U5; the reversed block used
    to clamp with a bare `max(..., 0)` — silently, at any distance — and pass
    the below line's nodes through unchecked."""
    return _EndArgs(line, gz, "above")


class _EndArgs:
    """`args(pt)` of `_end_tables` -- one end's (ρ, z, z′) against a line's
    quadrature nodes, the end in the slot its side takes (`_above_end_args`,
    `_below_end_args`) -- and `batch(pts)`, the same floats for many ends at
    once (momwire#1224), which `_classify_ends` reads.

    `batch` forms ρ with the same elementwise subtractions, in the same
    operand order, and the same `np.hypot`, so row e is `args(pts[e])[0]`
    to the bit; and the end's slot is one value per end, the float every
    entry of `args`' `np.full_like` array holds (`_on_plane_side` is
    elementwise, so it snaps or passes the one value as it does the full
    array). A refusal is left to `args`, which raises it at the first end
    in its own words."""

    __slots__ = ("nodes", "gz", "end_side", "line_z")

    def __init__(self, line, gz, end_side):
        self.nodes = line["nodes"]
        self.gz = gz
        self.end_side = end_side
        self.line_z = _LineZ(
            self.nodes, gz, "below" if end_side == "above" else "above"
        )

    def __call__(self, pt):
        nodes, gz = self.nodes, self.gz
        if self.end_side == "above":
            rho_e = np.hypot(pt[0] - nodes[:, 0], pt[1] - nodes[:, 1])
            return (
                rho_e,
                _on_plane_side(np.full_like(rho_e, pt[2] - gz), "above", "end point"),
                self.line_z(),
            )
        rho_e = np.hypot(nodes[:, 0] - pt[0], nodes[:, 1] - pt[1])
        return (
            rho_e,
            self.line_z(),
            _on_plane_side(np.full_like(rho_e, pt[2] - gz), "below", "end point"),
        )

    def batch(self, pts, cols=None):
        """`(rho, end)` for the (E, 3) end points `pts`: rho (E, n) and the
        end slot's value per end, (E,); see the class. `cols` restricts rho
        to those nodes' columns (the same floats, elementwise)."""
        nodes = self.nodes if cols is None else self.nodes[cols]
        if self.end_side == "above":
            rho = np.hypot(
                pts[:, 0, None] - nodes[None, :, 0], pts[:, 1, None] - nodes[None, :, 1]
            )
        else:
            rho = np.hypot(
                nodes[None, :, 0] - pts[:, 0, None], nodes[None, :, 1] - pts[:, 1, None]
            )
        return rho, _on_plane_side(pts[:, 2] - self.gz, self.end_side, "end point")


class _LineZ:
    """The line nodes' plane-side z of an end-args closure, formed by its
    FIRST call and handed out read-only after that (momwire#1224).

    It is the same `_on_plane_side(nodes[:, 2] - gz, ...)` for every end of
    the loop -- the expression does not read the end -- so the floats are
    those each call formed, and a deck that refuses still refuses at the
    first end, with the same message, and never earlier. Every caller
    copies or only reads it (`np.stack` of the span, `radius_fold` rows,
    `_fast_end_desc`'s compares); read-only makes a writer raise rather
    than poison the later ends. Recomputing it per end was ~40 % of the
    fused ends' classify pass at razor hub_deck(16) x8 (2,823 ends)."""

    __slots__ = ("_nodes", "_gz", "_side", "_z")

    def __init__(self, nodes, gz, side):
        self._nodes, self._gz, self._side, self._z = nodes, gz, side, None

    def __call__(self):
        if self._z is None:
            z = _on_plane_side(
                self._nodes[:, 2] - self._gz, self._side, "quadrature node"
            )
            z.setflags(write=False)
            self._z = z
        return self._z


def _below_end_args(line, gz):
    """`args(pt)` for `_end_tables`: a BELOW end point against an ABOVE line's
    quadrature nodes — the line in the `z` slot, the end in `z′` (the designed
    tables accept only z ≥ 0 ≥ z′, whichever role each side plays). The
    mirror of `_above_end_args`, on the same `_on_plane_side` rule."""
    return _EndArgs(line, gz, "below")


def _ends_and_corner(
    ctx,
    A,
    B,
    eps_t,
    k_p,
    c1,
    gz,
    memo=None,
    *,
    corner=True,
    test_ends=True,
    source_ends=True,
    rows=None,
    out=None,
    support=None,
):
    """The by-parts end terms + the designed corner, on the DENSE axes —
    linear in axis size, so the admissibility split never touches them
    (and the corner must never see coarse axes or a low-rank pass; its
    V(a) rides `six_point` at `_CORNER_RTOL`, outside any memo).

    `test_ends` / `source_ends` select the two end loops — the above axis's
    ends (BT, TW: point tests at the node) and the below axis's (SQ, SW: the
    node's by-parts ends seen from the above line). The corner reads both
    axes' ends either way. Both default on, the one-radius spelling; a
    two-radius node evaluates the loops at different radii
    (`cross_complete_blocks_two_radius`).

    `rows` (momwire#1029 phase 2) is a sorted array of global BASIS rows; the
    answer is then the pair `(t[rows, :], t[:, rows])`, from ONE pass over the
    ends and ONE evaluation of each end's kernel table. Both halves read the
    same length-n vectors the unrestricted spelling builds — an end term is
    rank 1, so a restriction is an index into its two factors and never a
    second contraction.

    `out` (unit C) is the block to ACCUMULATE into — `_main_split`'s, so the
    ends never allocate a second full-size array. Bit-identical to
    `out += _ends_and_corner(...)`: see `_ends_and_corner_rc` for why the
    support scatter preserves each entry's summation order. Without it the
    answer is a fresh block, which is what a direct caller still gets.

    The forward block's ROW axis is the above one, so its row ends are above
    ends (`_above_end_args`) and its column ends below ends."""
    return _ends_and_corner_rc(
        ctx,
        A,
        B,
        eps_t,
        k_p,
        c1,
        gz,
        memo,
        row_args=_above_end_args(B, gz),
        col_args=_below_end_args(A, gz),
        corner=corner,
        row_ends=test_ends,
        col_ends=source_ends,
        rows=rows,
        out=out,
        support=support,
    )


def _ends_and_corner_rc(
    ctx,
    R,
    C,
    eps_t,
    k_p,
    c1,
    gz,
    memo,
    *,
    row_args,
    col_args,
    corner,
    row_ends=True,
    col_ends=True,
    rows=None,
    out=None,
    support=None,
):
    """The end terms and the corner of a cross block whose rows are axis `R`'s
    basis and whose columns are `C`'s, in either orientation (momwire#1168
    U5: the forward and reversed blocks were two copies of this that had
    drifted apart).

    The ROW loop runs `R`'s ends against `C`'s line — the V term through
    `C`'s `Fd`, then the W term through `C`'s `F` against `C`'s t̂z — and the
    COLUMN loop `C`'s ends against `R`'s line, W then V. `row_args` /
    `col_args` build each loop's (ρ, z, z′) and are what carry the
    orientation: the above side always takes the `z` slot
    (`_above_end_args`, `_below_end_args`). In the forward block (R above)
    the row loop is BT + TW and the column loop SW + SQ; in the reversed
    block (R below) the row loop is SW's by-parts partner plus BT and the
    column loop TW + SQ — the same four products either way, which is why
    the reversed block reproduces the forward's transpose (momwire#813,
    momwire#956).

    `support` (a `_Support`) answers only its (rows, cols) sub-block, into an
    `out` of that shape: `E_r` and `E_c` are held on the support's columns
    and rows alone, and every rank-1 update writes the support's part of
    what it wrote before — the same elementwise products in the same order
    per entry (`_Support`).

    The accumulators and their scatter are `_EndTerms`, and one end's terms
    `_row_end_terms` / `_col_end_terms` — shared with the fused route
    (`_FusedEnds`, momwire#1173 design C phase 2), which runs the same
    per-end operations inside the product's tile pass."""
    T = _EndTerms(R, C, rows=rows, support=support)
    wR, wR_tz = _end_weights(R)
    wC, wC_tz = _end_weights(C)
    # The end loops' tables come a span of ends per call (`_end_tables`); the
    # rank-1 updates below still run one end at a time, in the order they did.
    # Each end's two matvecs come from `_end_vectors` (`_row_end_terms` /
    # `_col_end_terms`' vectors, the fast ends' read from the product's store
    # in batches); the rank-1 writes are those functions', end by end.
    for sign, fv, vV, vW in _end_vectors(
        ctx, eps_t, k_p, R["ends"] if row_ends else [], C, wC, wC_tz, memo, row_args
    ):
        nz = np.flatnonzero(fv)
        T.add_rows(nz, fv[nz], vV, c1 * sign)
        T.add_rows(nz, fv[nz], vW, -c1 * sign)
    for sign, fv, vV, vW in _end_vectors(
        ctx, eps_t, k_p, C["ends"] if col_ends else [], R, wR, wR_tz, memo, col_args
    ):
        nz = np.flatnonzero(fv)
        T.add_cols(nz, vW, fv[nz], -c1 * sign)
        T.add_cols(nz, vV, fv[nz], c1 * sign)
    if corner:
        _corner_terms(T, ctx, R, C, eps_t, k_p, c1, gz)
    return T.answer(out)


def _end_weights(ax):
    """`(w, w·t̂z)` of one axis, the two node weightings its end matvecs fold.

    THE NODE WEIGHTS FOLD INTO THE SHORT VECTOR, NOT THE TALL MATRIX
    (momwire#919). `(Fd_C * w_C) @ te["V"]` is `Fd_C @ (w_C * te["V"])`: the
    same contraction over the nodes, but the weighting is applied to a
    length-n_nodes vector instead of an (n_basis, n_nodes) matrix. On the
    48-radial screen that product was 222.6 MB, live for the whole routine
    and second only to the rank-1 updates below. Since momwire#1109 the
    matrix is a CSR and a folded copy would be cheap, and the fold still
    stands: it is one vector multiply either way.

    NOT bit-identical to the old spelling: (Fd*w)·V and Fd·(w*V) round
    differently. Gated at 1e-12 relative. One spelling for both
    orientations since momwire#1168 U5, so the reversed block can no longer
    reassociate one side alone (`test_the_main_sandwich_is_the_forward_
    transposed` pins the two bit-equal, and once broke in CI on exactly that).
    """
    _tx, _ty, tz = ax["t"].T
    w = ax["w"]
    return w, w * tz


def _row_end_terms(T, C, wC, wC_tz, c1, sign, fv, te):
    """One ROW end's two rank-1 terms into `T`, from its V and W tables over
    `C`'s nodes.

    momwire#912: `fv` is a value-1 tent's end value — a handful of nonzeros
    in n_basis — so the rank-1 update lands on those rows only. The same
    products where fv != 0; where it is 0 the full outer added an exact 0.

    The W end on the row axis's ends, contracting the column line's t̂z:
    TW (momwire#956) in the forward block — the test-side W end,
    −σ f_m(E)·∫ f_n t̂z′ W(E,·), left by testing −∇W along the wire, SW's
    partner on the other axis — and SW itself in the reversed block, where
    it stays paired with `s_w1` by the by-parts that produced it
    (momwire#813 derivation (b), 5312ca5)."""
    nz = np.flatnonzero(fv)
    T.add_rows(nz, fv[nz], _real_matvec_c(C["Fd_csr"], wC * te["V"]), c1 * sign)
    T.add_rows(nz, fv[nz], _real_matvec_c(C["F_csr"], wC_tz * te["W"]), -c1 * sign)


def _col_end_terms(T, R, wR, wR_tz, c1, sign, fv, te):
    """One COLUMN end's two rank-1 terms into `T` (W, then V), from its V
    and W tables over `R`'s nodes."""
    nz = np.flatnonzero(fv)
    T.add_cols(nz, _real_matvec_c(R["F_csr"], wR_tz * te["W"]), fv[nz], -c1 * sign)
    T.add_cols(nz, _real_matvec_c(R["Fd_csr"], wR * te["V"]), fv[nz], c1 * sign)


def _corner_terms(T, ctx, R, C, eps_t, k_p, c1, gz):
    """The designed corner into `T`: node tents against each other through V
    at R = a exactly.

    The sign is STRUCTURAL and orientation-carried: −σ_test·σ_src·c1·V(a),
    which is +c1·V(a) on the deck class the adjudication calibrated it on
    (above arm STARTING at the node, σ_a σ_b = −1) and flips with the wires'
    parametrization — an orientation-blind + wrecks a monopole spelled
    top-down into the node (measured: 10−1007j on the P3 rise deck, the
    −1000j truncation-class signature). Never re-pick per MEDIUM. It is the
    INTERFACE corner, so it applies only to end pairs that BOTH stand in the
    plane — an end elsewhere (the P3 fan's below-hub junction) carries its
    by-parts terms above but no corner. Symmetric in the two ends'
    one-hots, so the reversed block's is the forward's transposed and needs
    no orientation of its own.

    A path-tested row (momwire#813) passes `corner=False` and never gets
    here: its in-plane endpoint is a plain potential evaluation at z = 0⁺
    (the BT term), and the corner is a Galerkin by-parts term it never had.
    Measured on momwire#651's probe: with the corner the razor node row is
    off by 1.9e5 where razor's own kernel has none; without it, 5e-5
    (quadrature)."""
    a_wire = float(ctx.a_wire)
    v_at = {}
    for pt_a, sig_a, fv_a in R["ends"]:
        if abs(pt_a[2] - gz) > 1e-12:
            continue
        for pt_b, sig_b, fv_b in C["ends"]:
            if abs(pt_b[2] - gz) > 1e-12:
                continue
            # Every in-plane end pair, at one node or across two
            # (antennaknobs plan U9): V at the pair's own separation.
            rho = float(np.hypot(pt_a[0] - pt_b[0], pt_a[1] - pt_b[1]))
            v_corner = _corner_v(v_at, eps_t, k_p, a_wire, rho)
            nza, nzb = np.flatnonzero(fv_a), np.flatnonzero(fv_b)
            T.add_corner(nza, nzb, fv_a[nza], fv_b[nzb], -sig_a * sig_b * c1 * v_corner)


class _EndTerms:
    """The accumulators of a cross block's end terms and corner, and their
    scatter into the block (`_ends_and_corner_rc`'s, momwire#1029 phase 2
    unit C; a class since momwire#1173 design C phase 2 so the fused route
    can write the same arrays through the same operations).

    THE SHAPE'S OWN SUPPORT, never an (n, n) transient (the momwire#914
    pattern). `E_r` carries every term on the rows R's ends touch — the row
    terms over all columns, and, where they meet, the column terms and the
    corner as well — and `E_c` the column terms on the rows R's ends do NOT
    touch. So each entry accumulates in one place, in the order the loops
    write it, and the scatter at the end adds what the full block would
    have added: BIT-IDENTICAL to `dest += <a full-size ends block>`, which
    is what this replaced. Its 1.06 GB at 150 radials was the third-largest
    term of the fill's peak.

    `rows` (momwire#1029 phase 2) is a sorted array of global BASIS rows; the
    answer is then the pair `(t[rows, :], t[:, rows])`. Both halves read the
    same length-n vectors the unrestricted spelling builds — an end term is
    rank 1, so a restriction is an index into its two factors and never a
    second contraction.

    The streamed route (`_FusedEnds`) replays these same writes on a few
    units at a time, into arrays of its own."""

    def __init__(self, R, C, *, rows=None, support=None):
        nA, nB = R["n_basis"], C["n_basis"]
        sup = support
        if sup is not None and rows is not None:
            raise ValueError("support= and rows= are two answers; ask for one")
        self.nA, self.nB, self.sup, self.rows = nA, nB, sup, rows
        self.buf = _Rank1Buffer()
        self.bufT = _Rank1Buffer()
        if rows is None:
            LA, LB = _end_live_rows(R), _end_live_rows(C)
            self.LA, self.LB = LA, LB
            self.posLA, self.posLB = _positions(nA, LA), _positions(nB, LB)
            if sup is None:
                self.E_r = np.zeros((LA.size, nB), dtype=np.complex128)
                self.E_c = np.zeros((nA, LB.size), dtype=np.complex128)
            else:
                self.E_r = np.zeros((LA.size, sup.cols.size), dtype=np.complex128)
                self.E_c = np.zeros((sup.rows.size, LB.size), dtype=np.complex128)
            self.t_r = self.t_c = None
        else:
            self.t_r = np.zeros((rows.size, nB), dtype=np.complex128)
            self.t_c = np.zeros((nA, rows.size), dtype=np.complex128)

    def add_rows(self, nz, fv_nz, vec, scale):
        """`t[nz, :] += scale * outer(fv_nz, vec)`, into whichever blocks
        this call is answering with. `nz` is an R end's live rows, so it is
        inside `LA` by construction."""
        rows = self.rows
        if rows is None:
            if self.sup is not None:
                vec = vec[self.sup.cols]
            _rank1_add(self.E_r, self.posLA[nz], fv_nz, vec, scale, self.buf)
            return
        sel, pos = _in_rows(rows, nz)
        if sel.size:
            _rank1_add(self.t_r, pos, fv_nz[sel], vec, scale, self.buf)
        _rank1_add(self.t_c, nz, fv_nz, vec[rows], scale, self.buf)

    def add_cols(self, nz, vec, fv_nz, scale):
        """`t[:, nz] += scale * outer(vec, fv_nz)`; `nz` is inside `LB`."""
        rows, sup, bufT = self.rows, self.sup, self.bufT
        if rows is None:
            if sup is None:
                _rank1_add_cols(self.E_r, nz, vec[self.LA], fv_nz, scale, bufT)
                _rank1_add_cols(self.E_c, self.posLB[nz], vec, fv_nz, scale, bufT)
                return
            pc = sup.pos_c[nz]
            keep = pc >= 0
            _rank1_add_cols(self.E_r, pc[keep], vec[self.LA], fv_nz[keep], scale, bufT)
            _rank1_add_cols(self.E_c, self.posLB[nz], vec[sup.rows], fv_nz, scale, bufT)
            return
        _rank1_add_cols(self.t_r, nz, vec[rows], fv_nz, scale, bufT)
        sel, pos = _in_rows(rows, nz)
        if sel.size:
            _rank1_add_cols(self.t_c, pos, vec, fv_nz[sel], scale, bufT)

    def add_corner(self, nza, nzb, fva, fvb, scale):
        rows, sup = self.rows, self.sup
        if rows is None:
            if sup is not None:
                pc = sup.pos_c[nzb]
                keep = pc >= 0
                nzb, fvb = pc[keep], fvb[keep]
            self.E_r[np.ix_(self.posLA[nza], nzb)] += scale * np.outer(fva, fvb)
            return
        sel, pos = _in_rows(rows, nza)
        if sel.size:
            self.t_r[np.ix_(pos, nzb)] += scale * np.outer(fva[sel], fvb)
        selb, posb = _in_rows(rows, nzb)
        if selb.size:
            self.t_c[np.ix_(nza, posb)] += scale * np.outer(fva, fvb[selb])

    def answer(self, out=None):
        """The block (`out` accumulated into, or a fresh one), or the
        `(t_r, t_c)` pair under `rows`."""
        if self.rows is not None:
            return (self.t_r, self.t_c)
        sup, LA, LB, E_r, E_c = self.sup, self.LA, self.LB, self.E_r, self.E_c
        if sup is None:
            dest = (
                out
                if out is not None
                else np.zeros((self.nA, self.nB), dtype=np.complex128)
            )
        else:
            dest = out if out is not None else sup.zeros()
        # `E_r` already carries the LA rows' column terms and corner, so those
        # rows of `E_c` are dropped rather than added twice. Rows first, then
        # columns: an entry in LA x LB then sees its whole term and an exact
        # zero, which is the order the full block's single `+=` had.
        if sup is None:
            if LA.size:
                E_c[LA, :] = 0.0
                dest[LA, :] += E_r
            if LB.size:
                dest[:, LB] += E_c
            return dest
        if LA.size:
            pr = sup.pos_r[LA]
            keep = pr >= 0
            E_c[pr[keep], :] = 0.0
            dest[pr[keep], :] += E_r[keep]
        if LB.size:
            pc = sup.pos_c[LB]
            keep = pc >= 0
            dest[:, pc[keep]] += E_c[:, keep]
        return dest


# momwire#1173 design C phase 2: a cross block's end loops run INSIDE the
# product's tile pass (`_FusedEnds`), so the (rows, 2) V/W store phase 1 kept
# for them (~50 MB per block at razor hub_deck(16) x16) never exists. False
# is the in-process reference (the phase-1 route, store and all).
_FUSED_ENDS = True
# The column width of a streamed unit batch's fold into Z (`_FusedEnds._fold`),
# in bytes of the Z[ix] copy — `razor._IX_SLAB_BYTES`, for the same reason.
_Z_SLAB_BYTES = 8 * 2**20
# The streamed route declines when its holds would outweigh the V/W store it
# replaces — or this many complex entries (8 MiB), when the store is smaller.
_STREAM_MIN_HOLD = 1 << 19
# momwire#1335: the fused route serves a product of SEVERAL groups (razor's
# inverted L: 257 above groups in the forward block at x32, 3073 in the
# reversed), so the V/W store (~557 MB per block there) is never allocated.
# False is the reference it is gated against to the bit: such a product
# declines ("groups") and keeps the store, as before.
_FUSED_MULTI_GROUP = True
# `_FusedEnds._classify` asks the product about its slow ends' rows this
# many rows at a time (momwire#1335); False asks end by end, the reference.
_CLASSIFY_HIT_BATCHED = True
# A many-group product's line end standing on a line node is formed at that
# node's column tile (`_FusedEnds._line_tiles`, momwire#1335); False forms it
# at its own keys' latest tile, its earlier rows marked late, the reference.
_LINE_END_AT_COLUMN = True
# The fused slow-end prepass skips asking the product (its rows were checked
# to miss it at attach); False asks anyway, the reference (momwire#1335).
_PREPASS_MISSES_PROVEN = True
_CLASSIFY_HIT_ROWS = 1 << 20


def _csr_rows(M, rows):
    """`M[rows]` of a CSR, built from its own arrays: each selected row's
    stored entries, in their stored order, with their values."""
    starts = M.indptr[rows]
    lens = M.indptr[rows + 1] - starts
    ip = np.zeros(rows.size + 1, dtype=np.int64)
    np.cumsum(lens, out=ip[1:])
    take = np.repeat(starts - ip[:-1], lens) + np.arange(ip[-1])
    return _sp.csr_array(
        (M.data[take], M.indices[take], ip), shape=(rows.size, M.shape[1])
    )


def _csr_on_cols(M):
    """`(M′, need)`: `M` on the columns it stores, `need` ascending, and
    `M′` holding each row's stored entries in their stored order with column
    j renamed to its position in `need`. A matvec of `M′` against `v[need]`
    is `M @ v` row for row: `csr_matvec` sums a row's stored entries in
    stored order from zero, the renaming is monotone and reads the same
    `v[j]`, and no other entry of `v` is ever read."""
    need = np.unique(M.indices)
    idx = np.searchsorted(need, M.indices)
    return _sp.csr_array((M.data, idx, M.indptr), shape=(M.shape[0], need.size)), need


def _row_max(M, vals):
    """Per row of a CSR, the max of `vals` over its stored columns (−1 for
    an empty row)."""
    out = np.full(M.shape[0], -1, dtype=np.int64)
    live = np.diff(M.indptr) > 0
    if live.any():
        out[live] = np.maximum.reduceat(vals[M.indices], M.indptr[:-1][live])
    return out


class _FusedEnds:
    """A cross block's end loops run INSIDE its product tile pass, and each
    finished column of the block streamed into Z, when they can be; else the
    loops run afterwards as `_ends_and_corner_rc` (momwire#1173 design C
    phase 2). `finish(out)` completes the block either way.

    Phase 1 kept every product row's V and W in a row-ordered store because
    the end loops, which ran after the main sandwich, read them; and the
    block (`t`) and its end accumulators (`E_r`, `E_c`) were whole beside Z
    until razor folded `t` in. Here nothing of the three is whole: the loops
    read V and W from the tile that evaluates them, and the block is
    finished — sandwich, `c1`, end terms, scatter, fold into Z — a UNIT at a
    time as soon as everything that unit reads is in hand. The operations
    are the unfused route's, operand for operand; what moves is only WHEN
    each runs and which array holds its result, and the rest of this
    docstring argues that neither moves a bit.

    Measured at razor hub_deck(16) x16 (Skylake): peak RSS 497 → 417 MB
    with the below fold (`razor._assemble_Z_below_plane(into=)`), wall
    within 4 %.

    UNITS. The main sandwich completes the block along its below axis (the
    product's B side: `_streamed_sandwich` contracts a below basis row when
    its tables are in), so the unit is a below basis function in the
    support: a COLUMN of the forward block (`units_on="cols"`, C below) and
    a ROW of the reversed one (`units_on="rows"`, R below). Every write to
    the block and its accumulators is elementwise on its entry — an
    assignment, `*= c1`, a rank-1 `+=` whose `outer` forms each entry from
    its two factors in their operand order, the answer's scatter — so an
    entry's value depends only on the writes to THAT entry and their order. A unit's entries are therefore formed from
    zero, in a fresh array for the units finishing together, by replaying
    exactly the writes the whole-block route makes to them, in its order —
    each a restriction of the same `_rank1_add` / `_rank1_add_cols` /
    `scale * np.outer` call to the finishing units' rows or columns, which
    `outer` forms entry by entry from the same two factors (numpy's complex
    multiply is not commutative to the bit under FMA, so the factors keep
    their order, and they do):

      sandwich (assigned) · `*= c1` · row-loop terms (end order, V then W)
      · column-loop terms (end order, W then V) · corner · the scatter
      (`E_c` zeroed on `LA`, `+= E_r`, `+= E_c`) · `Z −= t`.

    The one loop whose vectors run over the units (the "vector" loop: the
    row loop when the units are C's, the column loop when they are R's)
    writes EVERY unit; the other ("local") loop writes only the units its
    end touches (`nz`, a handful). Each unit is finished once, when (a) its
    sandwich column is in (`sink`, or at once for a unit no sandwich row
    reaches) and (b) every term of both loops that writes it is computable:
    the static tile `e_tile` below.

    WHERE THE TERMS COME FROM. The slow ends (not product rows by
    `_fast_end_desc`) must all MISS the product — checked on the asked
    floats, as the memo keys them — so their `_tables` calls return the
    same floats and insert the same rows whether the product is complete,
    in progress or valueless (`ProductMemo.lookup` refuses only a HIT); all
    of their spans run first, row loop then column loop, which is the
    memo's order in the unfused route, and each end's two matvecs are held.
    A fast end's tables are copies of the tile's evaluated floats:

      * a "line" end asks one key per grouped node, all in ONE tile, where
        its whole tables are gathered and its matvecs formed exactly as the
        loop forms them (whole `Fd`/`F`, whole `w · V`);
      * a "grouped" end asks group g's z row against every line node. In
        the VECTOR loop its line is the unit axis, and a unit's matvec row
        reads the nodes it stores: it is complete at the latest of their
        key tiles — the same tile for every such end, since they read the
        same keys. There each end's matvec entries for the completed units
        are formed from the unit rows of `Fd`/`F` on the nodes they store
        (`_csr_on_cols`: the same stored entries summed in the same order
        from zero, against the same `w · V` products) and held until the
        units finish; rows a unit reads before its own tile are HELD in the
        tiles' held store. In the LOCAL loop its line is the other axis and
        its tables are gathered node by node over the tiles; its matvecs
        are formed, whole, at the last.

    THE SANDWICH hands each contracted row to `sink` instead of writing it
    into a support block (`_streamed_sandwich`: the same entries), always
    through the streamed contraction — one chunk when the tables came whole,
    which is that routine's own case. The unit's column is kept until it
    finishes, then assigned into zeros and scaled by `c1`, as the whole block
    was.

    SEVERAL GROUPS (momwire#1335). Nothing above needs one group; only the
    tiles at which an end's rows are in hand do. A grouped end reads its
    own group g's keys, so a unit is complete for the vector loop's grouped
    ends at the latest tile any of their groups' keys on its nodes reaches
    (`_max_tile_line`), and each end's rows are its own group's (`vg_g`,
    `vg_zl`). A line end asks one key per group, and those may sit in
    different tiles: it is formed at the latest (`_line_tiles`). Either
    way the rows read before that tile are held (`_late_rows`), so every
    matvec still reads copies of the evaluated floats, summed as before.
    With one group each of these is the expression it replaced. On
    razor's inverted L the line ends stand on line nodes, so their late
    rows are the sandwich's own late rows, already held.

    Declined — the tiles then keep phase 1's store and the loops run after
    them — on a slow end that hits the product, a missing support, or holds
    that would outweigh the store (with several groups the held late rows
    count too); and on a product of several groups when
    `_FUSED_MULTI_GROUP` is off (the reference). A
    block whose ends are all slow (the default lane) takes the "post" mode:
    no store, and the loops after the tiles as in the unfused route.

    Gates: the bits (`gate_z`, `test_fused_ends_1173`), counters that prove
    the fused and streamed routes ran, and a TEST-ONLY control
    (`_PRODUCT_NEG_CONTROL = "fused_order"`: the vector loop replayed last
    end first) that must move Z."""

    def __init__(
        self,
        ctx,
        R,
        C,
        eps_t,
        k_p,
        c1,
        gz,
        memo,
        *,
        row_args,
        col_args,
        corner,
        support,
        units_on,
        into=None,
    ):
        self.ctx, self.R, self.C = ctx, R, C
        self.eps_t, self.k_p, self.c1, self.gz, self.memo = eps_t, k_p, c1, gz, memo
        self.row_args, self.col_args = row_args, col_args
        self.corner, self.support = corner, support
        self.units_on, self.into = units_on, into
        self.attached = False
        self.streaming = False

    # -- deciding ---------------------------------------------------------

    def _classify(self, ends, args, product, fast, a_wire):
        """Each end's `_fast_end_desc`, or the string "hit" for a slow end
        that asks a product row (which this route cannot serve). The descs
        come batched (`_classify_ends`); the walk below is the one-end
        loop's, counting and stopping where it did.

        The slow ends' rows are asked of the product a span of ends at a
        time (`_CLASSIFY_HIT_ROWS` rows, momwire#1335): `value_rows` answers
        each row alone, so one call over several ends' rows is their calls'
        answers, and the walk's count and stop are read off the first end
        that hits. At razor's inverted L x32 the per-end calls through a
        many-group product's `_group_lookup` were ~3.8 s."""
        descs, on_node = _classify_ends(fast, a_wire, ends, args)
        if not _CLASSIFY_HIT_BATCHED:
            out = []
            for (pt, _sign, _fv), desc, on in zip(ends, descs, on_node):
                _ROUTES["ends_line_on_node"] += int(on)
                if desc is None:
                    rows = self._slow_rows(pt, args, a_wire)
                    if (product.value_rows(rows + 0.0) >= 0).any():
                        return "hit"
                out.append(desc)
            return out
        first_hit = None
        span, owners, n_span = [], [], 0

        def ask():
            rows = np.concatenate(span)
            hit = product.value_rows(rows + 0.0) >= 0
            if hit.any():
                return int(np.concatenate(owners)[int(np.argmax(hit))])
            return None

        for e, ((pt, _sign, _fv), desc) in enumerate(zip(ends, descs)):
            if desc is not None:
                continue
            rows = self._slow_rows(pt, args, a_wire)
            span.append(rows)
            owners.append(np.full(rows.shape[0], e, dtype=np.int64))
            n_span += rows.shape[0]
            if n_span >= _CLASSIFY_HIT_ROWS:
                first_hit = ask()
                span, owners, n_span = [], [], 0
                if first_hit is not None:
                    break
        if first_hit is None and span:
            first_hit = ask()
        stop = len(ends) if first_hit is None else first_hit + 1
        _ROUTES["ends_line_on_node"] += int(sum(on_node[:stop]))
        if first_hit is not None:
            return "hit"
        return list(descs)

    def _classify_loops(self, loops, product, fast, a_wire):
        """`_classify` of the row loop then the column loop, their slow ends'
        rows asked of the product together (`_CLASSIFY_HIT_ROWS` rows a
        call): one pass over the keys where each loop paid its own
        (momwire#1335). The walk is the two loops' in turn -- the first end
        that hits stops it, counted, and the column loop is not walked when
        the row loop hit -- so the descs, the decision and the counts are
        `_classify`'s."""
        descs, on_node, ends_of = [], [], []
        for ends, args in loops:
            d, o = _classify_ends(fast, a_wire, ends, args)
            descs.append(d)
            on_node.append(o)
            ends_of.append((ends, args))
        first_hit = None
        span, owners, n_span = [], [], 0

        def ask():
            rows = np.concatenate(span)
            hit = product.value_rows(rows + 0.0) >= 0
            if hit.any():
                return tuple(np.concatenate(owners)[int(np.argmax(hit))].tolist())
            return None

        for li, (ends, args) in enumerate(ends_of):
            for e, ((pt, _sign, _fv), desc) in enumerate(zip(ends, descs[li])):
                if desc is not None:
                    continue
                rows = self._slow_rows(pt, args, a_wire)
                span.append(rows)
                owners.append(
                    np.tile(np.array([[li, e]], dtype=np.int64), (rows.shape[0], 1))
                )
                n_span += rows.shape[0]
                if n_span >= _CLASSIFY_HIT_ROWS:
                    first_hit = ask()
                    span, owners, n_span = [], [], 0
                    if first_hit is not None:
                        break
            if first_hit is not None:
                break
        if first_hit is None and span:
            first_hit = ask()
        for li in range(len(loops)):
            if first_hit is not None and li > first_hit[0]:
                break
            stop = (
                first_hit[1] + 1
                if first_hit is not None and li == first_hit[0]
                else len(descs[li])
            )
            _ROUTES["ends_line_on_node"] += int(sum(on_node[li][:stop]))
        if first_hit is not None:
            return "hit"
        return [list(d) for d in descs]

    @staticmethod
    def _slow_rows(pt, args, a_wire):
        """The (n, 3) rows a slow end asks, as the memo keys them."""
        c = np.broadcast_arrays(*args(pt))
        rows = np.empty((c[0].size, 3), dtype=float)
        rows[:, 0] = _near_interface.radius_fold(c[0], a_wire).ravel()
        rows[:, 1] = c[1].ravel()
        rows[:, 2] = c[2].ravel()
        return rows

    def _decline(self, why):
        _ROUTES["fused_declined"] += 1
        _ROUTES["fused_declined_" + why] += 1
        return False

    def attach(self, plan, tiles):
        """Offer this block's end loops to the tiles; True when they will run
        inside them (the tiles then keep no V/W store)."""
        if (
            not _FUSED_ENDS
            or not _PRODUCT_ENDS
            or plan.fast is None
            or _PRODUCT_NEG_CONTROL == "row"
        ):
            return False
        if len(plan.rowtab) != 1 and not _FUSED_MULTI_GROUP:
            return self._decline("groups")
        R, C, fast = self.R, self.C, plan.fast
        product = tiles.product
        a_wire = float(self.ctx.a_wire)
        if _CLASSIFY_HIT_BATCHED:
            got = self._classify_loops(
                ((R["ends"], self.row_args), (C["ends"], self.col_args)),
                product,
                fast,
                a_wire,
            )
            if got == "hit":
                return self._decline("hit")
            row_cls, col_cls = got
        else:
            row_cls = self._classify(R["ends"], self.row_args, product, fast, a_wire)
            if row_cls == "hit":
                return self._decline("hit")
            col_cls = self._classify(C["ends"], self.col_args, product, fast, a_wire)
            if col_cls == "hit":
                return self._decline("hit")
        self.plan, self.tiles, self.fast = plan, tiles, fast
        self.row_cls, self.col_cls = row_cls, col_cls
        self.wR, self.wR_tz = _end_weights(R)
        self.wC, self.wC_tz = _end_weights(C)
        if all(d is None for d in row_cls + col_cls):
            self.mode = "post"
        else:
            if self.support is None:
                return self._decline("support")
            if not (_MAIN_STREAMED and _STREAMED_WHOLE_ROWS):
                return self._decline("stream")
            if not self._prepare_stream():
                return self._decline("hold")
            self.mode = "stream"
            self.streaming = True
            self._slow_prepass()
            tiles.extra_last = self._extra_last
            self._extra_last = None
        tiles.listener = self
        self.attached = True
        # Every lookup this block's tiles precede is made (the slow ends',
        # above, and the classification's): a key table those built is not
        # read again before `finish`, so it goes now rather than standing
        # through the tile pass (momwire#1335).
        product.keys.drop_index()
        _ROUTES["fused_blocks"] += 1
        _ROUTES["fused_mode_" + self.mode] += 1
        return True

    # -- the streamed route: setup ----------------------------------------

    def _prepare_stream(self):
        """Units, the two loops' roles, what each end needs and when; False
        when the holds would outweigh the store."""
        R, C, sup, plan, tiles = self.R, self.C, self.support, self.plan, self.tiles
        fwd = self.units_on == "cols"
        self.fwd = fwd
        # The unit axis (below) and the other; the vector loop's vectors run
        # over the unit axis: the row loop's over C, the column loop's over R.
        U_ax, O_ax = (C, R) if fwd else (R, C)
        self.units = sup.cols if fwd else sup.rows
        self.posU = sup.pos_c if fwd else sup.pos_r
        self.others = sup.rows if fwd else sup.cols
        self.posO = sup.pos_r if fwd else sup.pos_c
        nU = self.units.size
        self.LA, self.LB = _end_live_rows(R), _end_live_rows(C)
        self.posLA = _positions(R["n_basis"], self.LA)
        self.posLB = _positions(C["n_basis"], self.LB)
        # vector loop: (ends, classes, matrices V/W over the unit axis, their
        # weights); local loop: the same over the other axis.
        row = (R["ends"], self.row_cls, C, self.wC, self.wC_tz)
        col = (C["ends"], self.col_cls, R, self.wR, self.wR_tz)
        self.vec_loop, self.loc_loop = (row, col) if fwd else (col, row)
        self.nz_vec = _polled_nonzeros(self.vec_loop[0])
        self.nz_loc = _polled_nonzeros(self.loc_loop[0])
        n_line = plan.kid.shape[1]
        self._n_line = n_line
        self._tile_lines = {}  # group -> its keys' tiles along the line
        vcls, lcls = self.vec_loop[1], self.loc_loop[1]
        # The vector loop's grouped ends: per-unit completion tiles.
        self.vg = [i for i, d in enumerate(vcls) if d is not None and d[0] == "grouped"]
        e_tile = np.zeros(nU, dtype=np.int64)
        self.unit_tile = None
        if self.vg:
            if U_ax["nodes"].shape[0] != n_line:
                raise AssertionError(
                    "grouped vector ends ask over a line not the units'"
                )
            M = self.vec_loop[2]
            self.MV = _csr_rows(M["Fd_csr"], self.units)
            self.MW = _csr_rows(M["F_csr"], self.units)
            self.vg_g = np.array([vcls[i][1] for i in self.vg], dtype=np.int64)
            self.vg_zl = np.array([vcls[i][2] for i in self.vg], dtype=np.int64)
            # A unit is complete for EVERY grouped end at the latest tile any
            # of their groups' rows on its nodes reaches (momwire#1335: with
            # several groups each end reads its own group's keys); the rows
            # read before it are held (`_late_rows`).
            tile_line = self._max_tile_line(np.unique(self.vg_g))
            self.unit_tile = np.maximum(
                np.maximum(_row_max(self.MV, tile_line), _row_max(self.MW, tile_line)),
                0,
            )
            del tile_line
            np.maximum(e_tile, self.unit_tile, out=e_tile)
            o = np.argsort(self.unit_tile, kind="stable")
            b = np.searchsorted(self.unit_tile[o], np.arange(tiles.n_tiles + 1))
            self._units_of = (o, b)
            self.vg_tile = np.full(nU, -1, dtype=np.int64)
            self.vg_col = np.full(nU, -1, dtype=np.int64)
            self.vg_batches = {}
        # The vector loop's other ends: whole vectors at one tile (or now).
        self.vw = {}  # end -> (vV[units], vW[units])
        self._late = None  # per row, the last tile a fused loop reads it late
        self.v_tile = self._line_tiles(vcls)
        l_line_tile = self._line_tiles(lcls)
        if self.v_tile:
            np.maximum(e_tile, max(self.v_tile.values()), out=e_tile)
        # The local loop's ends: a whole vector over the other axis each, at
        # one tile (line), at the last node's tile (grouped), or now (slow);
        # held until every unit it writes is finished.
        self.lw = {}  # end -> its held vectors
        self.l_tile = {}
        self.l_te = {}  # grouped: [teV, teW, n filled]
        self.l_units = {}  # end -> the unit positions it writes
        self.l_left = {}
        for i, d in enumerate(lcls):
            nz = self.nz_loc[i]
            u = self.posU[nz]
            u = u[u >= 0]
            self.l_units[i] = u
            self.l_left[i] = u.size
            if d is None:
                t = -1
            elif d[0] == "line":
                t = l_line_tile[i]
            else:
                if O_ax["nodes"].shape[0] != n_line:
                    raise AssertionError("grouped local ends ask over a line not O's")
                t = int(self._tile_line_of(d[1]).max())
                nO = O_ax["nodes"].shape[0]
                self.l_te[i] = [
                    np.empty(nO, dtype=np.complex128),
                    np.empty(nO, dtype=np.complex128),
                    0,
                ]
            self.l_tile[i] = t
            if u.size:
                np.maximum.at(e_tile, u, t)
        self.e_tile = e_tile
        # Per tile, the line ends formed there (ascending, as the per-tile
        # scans found them) and the local loop's grouped ends.
        self._v_at, self._l_at = {}, {}
        for i, t in sorted(self.v_tile.items()):
            self._v_at.setdefault(t, []).append(i)
        for i, d in enumerate(lcls):
            if d is not None and d[0] == "line":
                self._l_at.setdefault(self.l_tile[i], []).append(i)
        self._l_grouped = [
            i for i, d in enumerate(lcls) if d is not None and d[0] == "grouped"
        ]
        self._vterms = None
        # unit -> the local ends writing it (each pair once, ends ascending)
        pairs_u = [self.l_units[i] for i in range(len(lcls))]
        pu = np.concatenate(pairs_u) if pairs_u else np.zeros(0, dtype=np.int64)
        pe = np.repeat(np.arange(len(lcls)), [u.size for u in pairs_u])
        o = np.lexsort((pe, pu))
        self._l_by_unit = (
            pe[o],
            np.searchsorted(pu[o], np.arange(nU + 1)),
        )
        # The sandwich's side: a unit no sandwich row reaches is in hand now;
        # the others when the streamed sandwich contracts their row, at the
        # tile that serves the last table column the row reads.
        rB = _support_rows(U_ax, np.arange(U_ax["nodes"].shape[0]))
        self.s_ready = np.ones(nU, dtype=bool)
        pb = self.posU[rB]
        self.s_ready[pb[pb >= 0]] = False
        s_tile = np.zeros(nU, dtype=np.int64)
        node_ready = tiles._node_ready
        for key in ("F_csr", "Fd_csr"):
            np.maximum(
                s_tile,
                _row_max(_csr_rows(U_ax[key], self.units), node_ready),
                out=s_tile,
            )
        # The rows read after their own tile, held beside the sandwich's (U,
        # V, W and dz'W each). With several groups they are counted against
        # the store too, as the upper bound they are (some are the
        # sandwich's own late rows); one group keeps its old budget.
        self._extra_last = self._late_rows()
        held = 0
        if len(plan.rowtab) > 1 and self._extra_last is not None:
            if _RECYCLE_HELD:
                # The held store's slots are recycled (`_plan_held`): what
                # these rows add is the most of them live at once.
                held = 4 * _held_live_peak(
                    tiles.n_tiles, tiles._tile_rows, self._extra_last
                )
            else:
                held = 4 * int(np.count_nonzero(self._extra_last >= 0))
        if self._hold_peak(s_tile, lcls) + held > max(
            2 * plan.n_rows, _STREAM_MIN_HOLD
        ):
            return False
        self.s_cols = {}
        self.e_ready = np.zeros(nU, dtype=bool)
        self.done = np.zeros(nU, dtype=bool)
        self.corner_terms = self._corner_list() if self.corner else []
        # The fold: into Z, or into a block the caller gets back.
        self.res = None if self.into is not None else sup.zeros()
        return True

    def _hold_peak(self, s_tile, lcls):
        """The most complex entries the streamed route holds at once, from
        its static schedule (a unit finishes at the later of its `e_tile`
        and its sandwich tile): the vector loop's whole vectors and grouped
        pieces, the local loop's vectors and gathered tables, and the
        sandwich columns waiting for their units."""
        n_t = self.tiles.n_tiles + 1
        fin = np.maximum(self.e_tile, s_tile)
        d = np.zeros(n_t + 1, dtype=np.int64)

        def hold(lo, hi, n):
            d[max(lo, 0)] += n
            d[max(hi, 0) + 1] -= n

        vcls = self.vec_loop[1]
        n_vwhole = sum(c is None or c[0] == "line" for c in vcls)
        hold(0, n_t - 1, 2 * n_vwhole * self.units.size)
        if self.vg:
            np.add.at(d, self.unit_tile, 2 * len(self.vg))
            np.add.at(d, fin + 1, -2 * len(self.vg))
        np.add.at(d, s_tile, self.others.size)
        np.add.at(d, fin + 1, -self.others.size)
        per_local = 2 * (self.others.size + (self.LA.size if self.fwd else 0))
        for i, c in enumerate(lcls):
            u = self.l_units[i]
            if u.size == 0:
                continue
            hold(self.l_tile[i], int(fin[u].max()), per_local)
            if c is not None and c[0] == "grouped":
                hold(0, self.l_tile[i], 2 * self._n_line)
        return int(np.cumsum(d).max())

    def _corner_list(self):
        """The corner's writes, `(nza, nzb, fva, fvb, scale)` in loop order —
        `_corner_terms`', collected rather than applied."""
        seen = []

        class _Collect:
            def add_corner(self, nza, nzb, fva, fvb, scale):
                seen.append((nza, nzb, fva, fvb, scale))

        _corner_terms(
            _Collect(), self.ctx, self.R, self.C, self.eps_t, self.k_p, self.c1, self.gz
        )
        return seen

    def _late_rows(self):
        """The rows the fused loops read after their own tile, for the tiles'
        held store, as a per-row last reading tile (`_late_last`, -1 for
        none): the vector loop's grouped ends' (a unit completing later reads
        them) and the line ends' (`_line_tiles`, marked already); or None."""
        if self.vg:
            self._late_grouped_rows()
        last, self._late = self._late, None
        if last is None or not np.any(last >= 0):
            return None
        return last

    def _late_marks(self):
        if self._late is None:
            self._late = _late_last(self.plan.n_rows, self.tiles.n_tiles)
        return self._late

    def _late_grouped_rows(self):
        """Mark the vector loop's grouped ends' late rows. Group g's row on
        line node n is read late iff some unit reading n completes after
        that row's tile: `tile_g[n] < maxT[n]`, maxT[n] the latest
        `unit_tile` of a unit whose matvec row stores n -- which is also the
        last tile that reads it. With one group this is the entry-by-entry
        test it replaced (the same nodes); with several it is made per
        group, on the zl values its ends ask."""
        maxT = np.full(self._n_line, -1, dtype=np.int64)
        for M in (self.MV, self.MW):
            t_entry = np.repeat(self.unit_tile, np.diff(M.indptr))
            np.maximum.at(maxT, M.indices, t_entry)
        fast = self.fast
        for g in np.unique(self.vg_g).tolist():
            _cancel.poll()
            tl = self.tiles.tile_of_key[self.plan.kid[g]]
            late_nodes = np.flatnonzero(tl < maxT)
            if late_nodes.size == 0:
                continue
            zls = np.unique(self.vg_zl[self.vg_g == g])
            kn = fast.kl_rank[g][late_nodes]
            tn = maxT[late_nodes]
            # A span of the group's z rows at a time (the mast's ~480 z by
            # its late nodes is ~10 M rows at x32).
            per = max(1, (1 << 20) // max(1, late_nodes.size))
            for z0 in range(0, zls.size, per):
                base = fast.off[g] + zls[z0 : z0 + per] * fast.nk[g]
                r = fast.rowflat[base[:, None] + kn[None, :]]
                _mark_late(self._late_marks(), r, tn[None, :])
                del r

    def _tile_line_of(self, g):
        """Group g's keys' tiles along the line, `tile_of_key[kid[g]]`
        (cached for the groups a local grouped end gathers over)."""
        tl = self._tile_lines.get(g)
        if tl is None:
            tl = self.tiles.tile_of_key[self.plan.kid[g]]
            self._tile_lines[g] = tl
        return tl

    def _max_tile_line(self, groups):
        """Per line node, the latest tile of `groups`' keys there (a span of
        groups at a time: no (groups, line) temporary)."""
        kid, tk = self.plan.kid, self.tiles.tile_of_key
        out = np.zeros(kid.shape[1], dtype=np.int64)
        for g0, g1 in _group_spans(groups.size, kid.shape[1]):
            _cancel.poll()
            np.maximum(out, tk[kid[groups[g0:g1]]].max(axis=0), out=out)
        return out

    def _line_tiles(self, cls):
        """{end: tile} for a loop's "line" ends: the latest tile of the keys
        it asks, one per group (`kids[g][kg[g]]`). Its rows in EARLIER tiles
        are read late, so they are marked for the held store. With
        one group an end's keys are one key, its tile the old
        `tile_of_key[kg[0]]` (the group's local key ids are the global
        ones), and nothing is late."""
        idx = [i for i, d in enumerate(cls) if d is not None and d[0] == "line"]
        if not idx:
            return {}
        plan, tk, fast = self.plan, self.tiles.tile_of_key, self.fast
        if len(plan.kids) == 1:
            return {i: int(tk[plan.kids[0][cls[i][1][0]]]) for i in idx}
        out = {}
        if plan.slot == "z" and _LINE_END_AT_COLUMN:
            # An end standing ON line node n (its desc names n) asks exactly
            # column n's rows: formed when the sandwich serves that column
            # (`_node_ready[n]`, never before its rows' tiles), every row it
            # reads before then is one of the sandwich's own late rows, held
            # to that tile at least -- nothing to mark (momwire#1335).
            ready = self.tiles._node_ready
            rest = []
            for i in idx:
                d = cls[i]
                if len(d) > 2:
                    out[i] = int(ready[d[2]])
                else:
                    rest.append(i)
            idx = rest
            if not idx:
                return out
        # Each (group, local key)'s tile, concatenated by group.
        nk = np.asarray(plan.nk, dtype=np.int64)
        koff = np.zeros(nk.size + 1, dtype=np.int64)
        np.cumsum(nk, out=koff[1:])
        tile_gk = np.empty(int(koff[-1]), dtype=tk.dtype)
        for g, kj in enumerate(plan.kids):
            if not g & 255:
                _cancel.poll()
            tile_gk[koff[g] : koff[g + 1]] = tk[kj]
        for part in _vec_batches(idx, fast.grank.size):
            _cancel.poll()
            KG = np.stack([cls[i][1] for i in part]).astype(np.int64)
            TG = tile_gk[koff[:-1][None, :] + KG]  # (ends, groups)
            del KG
            T = TG.max(axis=1)
            early = TG[:, fast.grank] < T[:, None]  # (ends, grouped nodes)
            del TG
            if early.any():
                R = _fast_desc_rows_batch(fast, [cls[i] for i in part])
                _mark_late(
                    self._late_marks(),
                    R[early],
                    np.broadcast_to(T[:, None], early.shape)[early],
                )
                del R
            del early
            for i, t in zip(part, T.tolist()):
                out[i] = int(t)
        return out

    def _slow_prepass(self):
        """Every slow span, now, in the unfused route's order (the row loop's,
        then the column loop's): each slow end's two matvecs, held."""
        R, C = self.R, self.C
        # `attach` checked every slow end's rows against the product and
        # found no hit (else it declined), so each lookup below would miss
        # the product: the memo skips asking it (`ProductMemo.misses_proven`,
        # momwire#1335) -- the same misses, without a pass over the keys.
        self.memo.misses_proven = _PREPASS_MISSES_PROVEN
        try:
            self._slow_prepass_loops(R, C)
        finally:
            self.memo.misses_proven = False

    def _slow_prepass_loops(self, R, C):
        loops = (
            (R["ends"], C, self.row_args, self.row_cls, self.wC, self.wC_tz, "row"),
            (C["ends"], R, self.col_args, self.col_cls, self.wR, self.wR_tz, "col"),
        )
        for ends, M, args, cls, w, w_tz, which in loops:
            i = 0
            for _pt, _sign, _fv, te in _end_tables_product(
                self.ctx,
                self.eps_t,
                self.k_p,
                ends,
                M["nodes"].shape[0],
                self.memo,
                args,
                classes=cls,
                fast_te=lambda _i: None,
            ):
                if te is not None:
                    vV = _real_matvec_c(M["Fd_csr"], w * te["V"])
                    vW = _real_matvec_c(M["F_csr"], w_tz * te["W"])
                    self._have_vectors(which, i, vV, vW)
                i += 1

    def _have_vectors(self, which, i, vV, vW):
        """End i of loop `which` has its two whole matvecs: keep the entries
        the finish reads."""
        vec_is_row = self.fwd == (which == "row")
        if vec_is_row:
            self.vw[i] = (vV[self.units], vW[self.units])
            return
        if self.fwd:  # a column end: E_r reads vec[LA], E_c vec[support rows]
            self.lw[i] = (vV[self.LA], vW[self.LA], vV[self.others], vW[self.others])
        else:  # a row end: vec over the support's columns
            self.lw[i] = (vV[self.others], vW[self.others])
        if self.l_left[i] == 0:
            self.lw.pop(i)

    # -- the tile pass ----------------------------------------------------

    @staticmethod
    def _vw(rows, loc, tb, held, hpos):
        """V and W of product rows: copies of the evaluated floats, from this
        tile's block or else the held store."""
        li = loc[rows]
        V, W = tb[li, 1], tb[li, 2]
        miss = li < 0
        if miss.any():
            hp = None if hpos is None else hpos[rows[miss]]
            if hp is None or (hp < 0).any():
                raise AssertionError("a fused end reads a row not in hand")
            V[miss] = held[hp, 1]
            W[miss] = held[hp, 2]
        return V, W

    def tile(self, t, loc, tb, held, hpos):
        """Tile t has been evaluated: form what it completes, then finish the
        units it makes ready."""
        if not self.streaming:
            return
        fast = self.fast
        _vends, vcls, M = self.vec_loop[0], self.vec_loop[1], self.vec_loop[2]
        _lends, lcls, N = self.loc_loop[0], self.loc_loop[1], self.loc_loop[2]
        wv, wv_tz = self.vec_loop[3], self.vec_loop[4]
        wl, wl_tz = self.loc_loop[3], self.loc_loop[4]
        which_v = "row" if self.fwd else "col"
        which_l = "col" if self.fwd else "row"
        # Vector loop, line ends at this tile: whole vectors, a batch of ends
        # per `_tile_matvecs` (each end's vectors are its own sums).
        here = self._v_at.get(t, [])
        for part in _vec_batches(here, M["nodes"].shape[0]):
            descs = [vcls[i] for i in part]
            vVs, vWs = _desc_matvecs(fast, descs, M, wv, wv_tz, loc, tb, held, hpos)
            for e, i in enumerate(part):
                self._have_vectors(which_v, i, vVs[e], vWs[e])
                _ROUTES["fused_row_ends"] += 1
        # Vector loop, grouped ends: the units completing here.
        if self.vg:
            o, b = self._units_of
            J = np.sort(o[b[t] : b[t + 1]])
            if J.size:
                _ROUTES["fused_unit_tiles"] += 1
                MV, needV = _csr_on_cols(_csr_rows(self.MV, J))
                MW, needW = _csr_on_cols(_csr_rows(self.MW, J))
                xV, xW = wv[needV], wv_tz[needW]
                VV = np.empty((len(self.vg), J.size), dtype=np.complex128)
                VW = np.empty((len(self.vg), J.size), dtype=np.complex128)
                n_need = max(needV.size, needW.size)
                gp_all = self.vg_g
                if _PRODUCT_NEG_CONTROL == "group0":
                    gp_all = np.zeros_like(gp_all)  # TEST-ONLY: one group's rows
                if _table_serves(MV, xV, fast, loc, hpos) and _table_serves(
                    MW, xW, fast, loc, hpos
                ):
                    # Every end at once: the rows are formed in the kernel
                    # (`end_matvecs_table`), never stored.
                    base = fast.off[gp_all] + self.vg_zl * fast.nk[gp_all]
                    for VX, Mx, need, x, kk in (
                        (VV, MV, needV, xV, 1),
                        (VW, MW, needW, xW, 2),
                    ):
                        VX[:] = _accel.acc.end_matvecs_table(
                            Mx.indptr,
                            Mx.indices,
                            Mx.data,
                            Mx.shape[0],
                            x,
                            base,
                            np.zeros(need.size, dtype=np.int64),
                            fast.kl_rank,
                            gp_all,
                            need,
                            fast.rowflat,
                            loc,
                            _EMPTY_I32_1D if hpos is None else hpos,
                            tb,
                            held,
                            kk,
                            _near_interface._physical_cpu_count(),
                        )
                    _ROUTES["fused_row_ends"] += len(self.vg)
                    parts = []
                else:
                    parts = _vec_batches(list(range(len(self.vg))), n_need)
                for part in parts:
                    # Each end's own group g: its z row zl against the
                    # needed line nodes' local keys in g (one group: g = 0
                    # for every end, the rows they always were).
                    gp = gp_all[part[0] : part[-1] + 1]
                    base = (
                        fast.off[gp] + self.vg_zl[part[0] : part[-1] + 1] * fast.nk[gp]
                    )
                    kV = fast.kl_rank[gp[:, None], needV[None, :]]
                    kW = fast.kl_rank[gp[:, None], needW[None, :]]
                    rows = slice(part[0], part[-1] + 1)
                    VV[rows] = _tile_matvecs(
                        MV,
                        xV,
                        fast.rowflat[base[:, None] + kV],
                        1,
                        loc,
                        tb,
                        held,
                        hpos,
                    )
                    VW[rows] = _tile_matvecs(
                        MW,
                        xW,
                        fast.rowflat[base[:, None] + kW],
                        2,
                        loc,
                        tb,
                        held,
                        hpos,
                    )
                    _ROUTES["fused_row_ends"] += len(part)
                self.vg_batches[t] = [VV, VW, J.size]
                self.vg_tile[J] = t
                self.vg_col[J] = np.arange(J.size)
        # Local loop: line ends at this tile, batched as the vector loop's;
        # grouped ends gather their tables over the tiles and form theirs at
        # the last.
        here = self._l_at.get(t, [])
        for part in _vec_batches(here, N["nodes"].shape[0]):
            descs = [lcls[i] for i in part]
            vVs, vWs = _desc_matvecs(fast, descs, N, wl, wl_tz, loc, tb, held, hpos)
            for e, i in enumerate(part):
                self._have_vectors(which_l, i, vVs[e], vWs[e])
                _ROUTES["fused_col_te"] += 1
        sel_g = {}  # group -> its line nodes whose key is in this tile
        for i in self._l_grouped:
            d = lcls[i]
            buf = self.l_te.get(i)
            if buf is None:
                continue  # formed already, at its last tile
            sel_t = sel_g.get(d[1])
            if sel_t is None:
                sel_t = sel_g[d[1]] = np.flatnonzero(self._tile_line_of(d[1]) == t)
            if sel_t.size:
                rows = _fast_desc_rows(fast, d)[sel_t]
                buf[0][sel_t], buf[1][sel_t] = self._vw(rows, loc, tb, held, hpos)
                buf[2] += sel_t.size
            if self.l_tile[i] != t:
                continue
            if buf[2] != buf[0].size:
                raise AssertionError("a fused local end's tables are incomplete")
            V, W = buf[0], buf[1]
            del self.l_te[i]
            vV = _real_matvec_c(N["Fd_csr"], wl * V)
            vW = _real_matvec_c(N["F_csr"], wl_tz * W)
            self._have_vectors(which_l, i, vV, vW)
            _ROUTES["fused_col_te"] += 1
        # Finish the units this tile makes ready.
        now = self.e_tile == t
        self.e_ready |= now
        self._finish_units(np.flatnonzero(now & self.s_ready & ~self.done))

    def sink(self, rA, b_idx, block):
        """The streamed sandwich's rows `b_idx` of the below axis are
        contracted: `block` holds their entries on the sandwich's rows `rA`
        (`_put`'s block, which this keeps per unit instead of writing it
        into a whole support block: the same entries, assigned)."""
        rA_pos = self.posO[rA]
        kr = rA_pos >= 0
        pu = self.posU[b_idx]
        n_o = self.others.size
        for j in np.flatnonzero(pu >= 0).tolist():
            col = np.zeros(n_o, dtype=np.complex128)
            col[rA_pos[kr]] = block[kr, j]
            self.s_cols[int(pu[j])] = col
        u = pu[pu >= 0]
        self.s_ready[u] = True
        _ROUTES["stream_held_units"] = max(
            _ROUTES["stream_held_units"], len(self.s_cols)
        )
        self._finish_units(u[self.e_ready[u] & ~self.done[u]])

    # -- finishing units --------------------------------------------------

    def _finish_units(self, U):
        """Units U (positions) are complete: form their entries by the
        whole-block route's writes, in its order, and fold them."""
        if U.size == 0:
            return
        U = np.sort(U)
        c1 = self.c1
        fwd = self.fwd
        nU = U.size
        n_o = self.others.size
        # sandwich (assigned into zeros), then `*= c1`
        S = np.zeros((n_o, nU), dtype=np.complex128)
        for k, u in enumerate(U.tolist()):
            col = self.s_cols.pop(u, None)
            if col is not None:
                S[:, k] = col
        S *= c1
        if fwd:
            self._finish_cols(U, S)
        else:
            self._finish_rows(U, S.T.copy())
        self.done[U] = True
        _ROUTES["stream_units"] += nU
        _ROUTES["stream_finish_calls"] += 1

    def _vector_pieces(self, U):
        """Per vector-loop end, its two matvec entries at units U."""
        out = {}
        for i, (vV, vW) in self.vw.items():
            out[i] = (vV[U], vW[U])
        if self.vg:
            VV, VW = self._vg_matrices(U)
            for r, i in enumerate(self.vg):
                out[i] = (VV[r], VW[r])
        return out

    def _vector_terms(self):
        """The forward block's vector-loop rank-1 terms, fixed for the block
        (momwire#1335): in loop order (V then W per end, ends with no live
        row skipped), each term's rows `posLA[nz]`, factor `fv[nz]`, scale
        (`c1 * sign`, `-c1 * sign`) and the row of the per-batch piece
        matrix (`_vector_adds`) its vector is. Built once, so a finish batch
        does no per-end Python."""
        order = list(range(len(self.vec_loop[0])))
        if _PRODUCT_NEG_CONTROL == "fused_order":
            order = order[::-1]  # TEST-ONLY: the ends in the wrong order
        ends = [i for i in order if self.nz_vec[i].size]
        m = len(ends)
        c1 = self.c1
        rows, a, src, sc = [], [], [], []
        for j, i in enumerate(ends):
            nz = self.nz_vec[i]
            _pt, sign, fv = self.vec_loop[0][i]
            r, av, k = self.posLA[nz], fv[nz], nz.size
            rows += [r, r]
            a += [av, av]
            src += [np.full(k, j, dtype=np.int64), np.full(k, m + j, dtype=np.int64)]
            sc += [
                np.full(k, c1 * sign, dtype=np.complex128),
                np.full(k, -c1 * sign, dtype=np.complex128),
            ]
        j_of = {i: j for j, i in enumerate(ends)}
        vg_rows = np.array(
            [r for r, i in enumerate(self.vg) if i in j_of], dtype=np.int64
        )
        vg_j = np.array([j_of[self.vg[r]] for r in vg_rows.tolist()], dtype=np.int64)
        cat = np.concatenate
        empty_f, empty_i = np.zeros(0), np.zeros(0, dtype=np.int64)
        return dict(
            m=m,
            j_of=j_of,
            vg_rows=vg_rows,
            vg_j=vg_j,
            rows=cat(rows) if rows else empty_i,
            a=cat(a) if a else empty_f,
            src=cat(src) if src else empty_i,
            scale=cat(sc) if sc else np.zeros(0, dtype=np.complex128),
        )

    def _vector_adds(self, E_r, U):
        """`_finish_cols`' vector-loop writes for units U: every term's
        `_rank1_add` (the row `a[r] * vec`, then `*= scale`, then added),
        as `_rank1_adds` forms them -- each entry by the same operations on
        the same operands, `np.add.at` in term order -- with the term table
        built once (`_vector_terms`) and the vectors gathered by row."""
        vt = self._vterms
        if vt is None:
            vt = self._vterms = self._vector_terms()
        m = vt["m"]
        if m == 0:
            return
        P = np.empty((2 * m, U.size), dtype=np.complex128)
        for i, (vV, vW) in self.vw.items():
            j = vt["j_of"].get(i)
            if j is not None:
                P[j] = vV[U]
                P[m + j] = vW[U]
        if self.vg:
            VV, VW = self._vg_matrices(U)
            P[vt["vg_j"]] = VV[vt["vg_rows"]]
            P[m + vt["vg_j"]] = VW[vt["vg_rows"]]
        C = np.multiply(vt["a"][:, None], P[vt["src"]])
        del P
        C *= vt["scale"][:, None]
        np.add.at(E_r, vt["rows"], C)

    def _vg_matrices(self, U):
        """The grouped vector ends' entries at units U, (len(vg), |U|) each
        for V and W: `_vector_pieces`' gather from the tiles' batches."""
        VV = np.empty((len(self.vg), U.size), dtype=np.complex128)
        VW = np.empty((len(self.vg), U.size), dtype=np.complex128)
        ts = self.vg_tile[U]
        if (ts < 0).any():
            raise AssertionError("a unit finished before its vector entries")
        for t in np.unique(ts).tolist():
            msk = ts == t
            bt = self.vg_batches[t]
            cols = self.vg_col[U[msk]]
            VV[:, msk] = bt[0][:, cols]
            VW[:, msk] = bt[1][:, cols]
            bt[2] -= int(msk.sum())
            if bt[2] == 0:
                del self.vg_batches[t]
        return VV, VW

    def _release_local(self, U):
        """Drop a local end's vectors once every unit it writes is done."""
        for i in self._local_pairs(U).tolist():
            self.l_left[i] -= 1
            if self.l_left[i] == 0:
                del self.lw[i]

    def _finish_cols(self, U, t_U):
        """The forward block's units: C columns of `E_r`, `E_c` and `t`."""
        sup, c1 = self.support, self.c1
        LA, posLA = self.LA, self.posLA
        E_r = np.zeros((LA.size, U.size), dtype=np.complex128)
        cU = self.units[U]
        inLB = self.posLB[cU] >= 0
        E_c = np.zeros((self.others.size, int(inLB.sum())), dtype=np.complex128)
        kLB = np.full(self.units.size, -1, dtype=np.int64)
        kLB[U[inLB]] = np.arange(E_c.shape[1])
        kU = np.full(self.units.size, -1, dtype=np.int64)
        kU[U] = np.arange(U.size)
        buf = _Rank1Buffer()
        # The row loop (vector): every end in order, V then W.
        if _BATCHED_VECTOR_ADDS:
            self._vector_adds(E_r, U)
        else:
            pieces = self._vector_pieces(U)
            order = range(len(self.vec_loop[0]))
            if _PRODUCT_NEG_CONTROL == "fused_order":
                order = reversed(order)  # TEST-ONLY: the ends in the wrong order
            for i in order:
                nz = self.nz_vec[i]
                if nz.size == 0:
                    continue
                _pt, sign, fv = self.vec_loop[0][i]
                vV, vW = pieces[i]
                _rank1_add(E_r, posLA[nz], fv[nz], vV, c1 * sign, buf)
                _rank1_add(E_r, posLA[nz], fv[nz], vW, -c1 * sign, buf)
        # The column loop (local): its ends touching U in order, W then V —
        # `add_cols`' two writes (E_r on the support's columns, E_c on LB's),
        # on the columns finishing here.
        for i in self._local_touching(U):
            nz = self.nz_loc[i]
            _pt, sign, fv = self.loc_loop[0][i]
            fv_nz = fv[nz]
            pc = sup.pos_c[nz]
            on = pc >= 0
            ku = np.where(on, kU[pc], -1)
            kc = np.where(on, kLB[pc], -1)
            sel, selc = ku >= 0, kc >= 0
            lvV, lvW, svV, svW = self.lw[i]
            for lv, sv, s in ((lvW, svW, -c1 * sign), (lvV, svV, c1 * sign)):
                if sel.any():
                    _rank1_add_cols(E_r, ku[sel], lv, fv_nz[sel], s, buf)
                if selc.any():
                    _rank1_add_cols(E_c, kc[selc], sv, fv_nz[selc], s, buf)
        for nza, nzb, fva, fvb, scale in self.corner_terms:
            pc = sup.pos_c[nzb]
            keep = pc >= 0
            k = kU[pc[keep]]
            sel = k >= 0
            if sel.any():
                E_r[np.ix_(posLA[nza], k[sel])] += scale * np.outer(fva, fvb[keep][sel])
        # The scatter: `E_c` zeroed on the support's LA rows, `+= E_r` there,
        # then `+= E_c` on the LB columns.
        if LA.size:
            pr = sup.pos_r[LA]
            keep = pr >= 0
            E_c[pr[keep], :] = 0.0
            t_U[pr[keep], :] += E_r[keep]
        if E_c.shape[1]:
            t_U[:, inLB] += E_c
        self._fold(self.others, self.units[U], t_U, rows_are_units=False, U=U)
        self._release_local(U)

    def _local_pairs(self, U):
        """The (end, unit) pairs of the local loop on units U."""
        o, b = self._l_by_unit
        return np.concatenate([o[b[u] : b[u + 1]] for u in U.tolist()] or [o[:0]])

    def _local_touching(self, U):
        """The local loop's ends that write any of units U, in loop order;
        each must have its vectors in hand (its tile is in `e_tile`)."""
        out = np.unique(self._local_pairs(U)).tolist()
        for i in out:
            if i not in self.lw:
                raise AssertionError("a unit finished before a local end it reads")
        return out

    def _finish_rows(self, U, t_U):
        """The reversed block's units: R rows of `E_r`, `E_c` and `t`."""
        sup, c1 = self.support, self.c1
        posLA = self.posLA
        rU = self.units[U]  # R basis
        la = posLA[rU] >= 0
        E_r = np.zeros((int(la.sum()), sup.cols.size), dtype=np.complex128)
        kLA = np.full(posLA.size, -1, dtype=np.int64)  # R basis -> E_r row
        kLA[rU[la]] = np.arange(E_r.shape[0])
        E_c = np.zeros((U.size, self.LB.size), dtype=np.complex128)
        buf = _Rank1Buffer()
        # The row loop (local): its ends touching U in order, V then W —
        # `add_rows`' write, on the rows finishing here.
        for i in self._local_touching(U):
            nz = self.nz_loc[i]
            k = kLA[nz]
            sel = k >= 0
            if not sel.any():
                continue
            _pt, sign, fv = self.loc_loop[0][i]
            lvV, lvW = self.lw[i]
            _rank1_add(E_r, k[sel], fv[nz][sel], lvV, c1 * sign, buf)
            _rank1_add(E_r, k[sel], fv[nz][sel], lvW, -c1 * sign, buf)
        # The column loop (vector): every end in order, W then V.
        pieces = self._vector_pieces(U)
        order = range(len(self.vec_loop[0]))
        if _PRODUCT_NEG_CONTROL == "fused_order":
            order = reversed(order)  # TEST-ONLY: the ends in the wrong order
        rows_la = np.flatnonzero(la)
        for i in order:
            nz = self.nz_vec[i]
            if nz.size == 0:
                continue
            _pt, sign, fv = self.vec_loop[0][i]
            fv_nz = fv[nz]
            vV, vW = pieces[i]
            pc = sup.pos_c[nz]
            keep = pc >= 0
            for v, s in ((vW, -c1 * sign), (vV, c1 * sign)):
                _rank1_add_cols(E_r, pc[keep], v[rows_la], fv_nz[keep], s, buf)
                _rank1_add_cols(E_c, self.posLB[nz], v, fv_nz, s, buf)
        for nza, nzb, fva, fvb, scale in self.corner_terms:
            pc = sup.pos_c[nzb]
            keep = pc >= 0
            k = kLA[nza]
            sel = k >= 0
            if sel.any():
                E_r[np.ix_(k[sel], pc[keep])] += scale * np.outer(fva[sel], fvb[keep])
        # The scatter: the LA units' `E_c` rows zeroed and `+= E_r`, then
        # `+= E_c` on the support's LB columns.
        if rows_la.size:
            E_c[rows_la, :] = 0.0
            t_U[rows_la, :] += E_r
        if self.LB.size:
            pc = sup.pos_c[self.LB]
            keep = pc >= 0
            t_U[:, pc[keep]] += E_c[:, keep]
        self._fold(self.units[U], self.others, t_U, rows_are_units=True, U=U)
        self._release_local(U)

    def _fold(self, zr, zc, block, *, rows_are_units, U):
        """`Z[np.ix_(zr, zc)] −= block` (the caller's fold, one subtraction
        per entry, column slabs as `razor._ix_accumulate` takes them), or the
        block's place in the returned support block."""
        if self.into is None:
            if rows_are_units:
                self.res[U, :] = block
            else:
                self.res[:, U] = block
            return
        Z = self.into
        w = max(1, _Z_SLAB_BYTES // (16 * max(1, zr.size)))
        for c0 in range(0, zc.size, w):
            ix = np.ix_(zr, zc[c0 : c0 + w])
            Z[ix] -= block[:, c0 : c0 + w]

    # -- finishing --------------------------------------------------------

    def finish(self, out):
        """The block's end terms and corner — accumulated into `out` (the
        main sandwich's block) and that block returned; or, streamed, every
        unit already folded, and None (`into`) or the block."""
        if not self.attached:
            return _ends_and_corner_rc(
                self.ctx,
                self.R,
                self.C,
                self.eps_t,
                self.k_p,
                self.c1,
                self.gz,
                self.memo,
                row_args=self.row_args,
                col_args=self.col_args,
                corner=self.corner,
                out=out,
                support=self.support,
            )
        if not self.tiles.product.complete:
            raise AssertionError("the block finished before its tiles did")
        # The tiles and this object name each other; a cycle would keep the
        # plan and the tiles alive past the block, until a garbage collection
        # (measured: the forward block's ~45 MB still standing at the
        # reversed block's peak). Cut it here.
        self.tiles.listener = None
        if self.streaming:
            self._finish_units(np.flatnonzero(~self.done))
            if not self.done.all() or self.s_cols or self.lw:
                raise AssertionError("a streamed unit or hold was left over")
            self.tiles = self.plan = self.fast = None
            return None if self.into is not None else self.res
        R, C = self.R, self.C
        T = _EndTerms(R, C, support=self.support)
        ctx, eps_t, k_p, c1, memo = self.ctx, self.eps_t, self.k_p, self.c1, self.memo
        for _pt, sign, fv, te in _end_tables_product(
            ctx,
            eps_t,
            k_p,
            R["ends"],
            C["nodes"].shape[0],
            memo,
            self.row_args,
            classes=self.row_cls,
        ):
            _row_end_terms(T, C, self.wC, self.wC_tz, c1, sign, fv, te)
        for _pt, sign, fv, te in _end_tables_product(
            ctx,
            eps_t,
            k_p,
            C["ends"],
            R["nodes"].shape[0],
            memo,
            self.col_args,
            classes=self.col_cls,
        ):
            _col_end_terms(T, R, self.wR, self.wR_tz, c1, sign, fv, te)
        if self.corner:
            _corner_terms(T, ctx, R, C, eps_t, k_p, c1, self.gz)
        self.tiles = self.plan = self.fast = None
        return T.answer(out)


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
    ctx,
    P,
    Q,
    eps_t,
    k_p,
    c1,
    gz,
    memo=None,
    *,
    corner=True,
    sw_end=SW_BY_PARTS,
    rows=None,
    out=None,
    support=None,
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
    pairing the whole reversed block reproduces `t_ab.T` exactly. Since
    momwire#956 the TW term closes the other half, so both readings are the
    same spelling and `sw_end` selects nothing; it is kept for the API.

    Since momwire#1168 U5 this is `_ends_and_corner_rc` with the rows BELOW:
    the same loops, the same `_on_plane_side` rule on both slots (it used to
    clamp a wrong-side end silently with a bare `min`/`max`, and now refuses
    it past `_PLANE_TOL` exactly as the forward block does), the same #1029
    support scatter instead of a full (n, n) transient with `np.outer`
    corners, and the same `rows=` / `out=` answers. Bit-identical on every
    deck that does not put an end or node on the wrong side of the plane."""
    del sw_end  # both readings are one spelling since momwire#956
    return _ends_and_corner_rc(
        ctx,
        P,
        Q,
        eps_t,
        k_p,
        c1,
        gz,
        memo,
        row_args=_below_end_args(Q, gz),
        col_args=_above_end_args(P, gz),
        corner=corner,
        rows=rows,
        out=out,
        support=support,
    )


@_cancelable
def cross_complete_block_reversed(
    ctx, P, Q, *, corner=True, sw_end=SW_BY_PARTS, support=None, into=None
):
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
    eps_t, k_p, gz, c1, memo = _block_preamble(ctx)  # momwire#1017, as above
    # `support=(rows, cols)` as `cross_complete_block`'s: this block's rows
    # are P's, so the transposed main sandwich takes the support swapped.
    sup = _Support.of(support, P["n_basis"], Q["n_basis"])
    sup_t = None if sup is None else sup.transposed()
    del sw_end  # both readings are one spelling since momwire#956
    # `_ends_and_corner_reversed`'s arguments, fused into the product's tiles
    # when they can be (`_FusedEnds`, momwire#1173 design C phase 2).
    ends = _FusedEnds(
        ctx,
        P,
        Q,
        eps_t,
        k_p,
        c1,
        gz,
        memo,
        row_args=_below_end_args(Q, gz),
        col_args=_above_end_args(P, gz),
        corner=corner,
        support=sup,
        units_on="rows",
        into=into,
    )
    t_ba = _main_sandwich(
        ctx, Q, P, eps_t, k_p, c1, gz, memo=memo, support=sup_t, ends=ends
    )
    # Accumulated in place through the support scatter, bit-identical to
    # `t_ba += <the full ends block>` (`_ends_and_corner_rc`); streamed, the
    # sandwich answers None and every row is already folded (`into`, as in
    # `cross_complete_block`).
    return ends.finish(None if t_ba is None else t_ba.T)


def _row_weights(ax, ii, rows=None):
    """The four per-basis row-weight matrices of one axis, restricted to
    the point subset `ii` — and, when `rows` is given, to those basis rows
    as well: (F·w·t̂x, F·w·t̂y, F·w·t̂z, F′·w), each a CSR sub-block.

    CSR since momwire#1109, entry for entry what the dense gather produced:
    `w` and `t̂` vary by POINT, so every fold here is a column scaling and the
    pattern never changes. The dense form was the allocation the issue named —
    a `(2035, 7992)` float64, 124 MiB per call and 292 calls per fill on the
    150-radial screen, carrying ≤ 3 nonzeros per column. Measured on the
    48-radial screen: 62.7 MB largest and 750 MB of churn, against 0.1 MB and
    12 MB here.
    """
    tx, ty, tz = ax["t"][ii].T
    w = ax["w"][ii]
    F = ax["F_csr"]
    Fd = ax["Fd_csr"]
    if rows is not None:
        F, Fd = F[rows], Fd[rows]
    Fw = _scale_cols(F[:, ii], w)
    Fdw = _scale_cols(Fd[:, ii], w)
    return _scale_cols(Fw, tx), _scale_cols(Fw, ty), _scale_cols(Fw, tz), Fdw


def _support_rows(ax, ii):
    """The basis rows that can be nonzero over the point subset `ii`.

    A B-spline basis function has local support, so over one cluster block's
    handful of points almost every row of the weight matrices is identically
    zero. `w` and `t̂` vary by POINT, not by row, so `F` and `Fd` are the only
    row-varying factors and their union is an exact superset of the live rows
    — never a guess. NaN compares unequal to zero, so a poisoned row stays in
    and still propagates rather than being silently dropped — and the CSR
    fallback below reads the STORED PATTERN, which keeps that true for free
    (`_basis_samples` stores a live wing's entries whatever they sample to,
    so a NaN and an exact zero are both in the pattern; the extra rows an
    exact zero brings in are still an exact restriction, momwire#912's
    argument).
    """
    seg_rows = ax.get("seg_rows")
    if seg_rows is None:
        F = ax["F_csr"][:, ii]
        Fd = ax["Fd_csr"][:, ii]
        return np.flatnonzero((np.diff(F.indptr) > 0) | (np.diff(Fd.indptr) > 0))
    # momwire#912: the same superset from the axis's own segment→rows map —
    # a row is live over `ii` only through a live wing on one of the block's
    # segments, and every such row is in the map. Restricting to extra rows
    # that happen to sample as exact zeros is still an exact restriction.
    segs = np.unique(ax["segof"][ii])
    rows = [seg_rows[int(g)] for g in segs if int(g) in seg_rows]
    if not rows:
        return np.zeros(0, dtype=np.int64)
    return np.unique(np.concatenate(rows))


def _nodes_of(ax, segs):
    """The axis's node indices on `segs`, ascending — `_main_split`'s block
    indexing from the run table instead of a per-block `np.isin` scan over
    every node (momwire#912)."""
    runs = ax["seg_runs"]
    parts = []
    _cancel.poll()
    for g in segs:
        rc = runs.get(int(g))
        if rc is not None:
            parts.append(np.arange(rc[0], rc[0] + rc[1]))
    if not parts:
        return np.zeros(0, dtype=np.int64)
    return np.sort(np.concatenate(parts))


def _left_products(Ps, K, k2sq):
    """The six left products `P_i @ K_x` of the main sandwich, in the
    reference term order (U·x̂, U·ŷ, ẑẑ, the two W cross terms, Φ)."""
    if isinstance(K, _TileTables):
        got = K.left_products(Ps, k2sq)
        if got is not None:
            return got
        K = K.as_dict()
    if _dense_left_serve(Ps, K):
        # The whole-table and grid routes' tables through the same loop as
        # the tiles' (momwire#1224): one sum for every route, so the routes
        # agree to the bit by construction on any build.
        return _accel.acc.left_products_gathered(
            [P.indptr for P in Ps],
            [P.indices for P in Ps],
            [P.data for P in Ps],
            Ps[0].shape[0],
            float(k2sq),
            np.zeros((0, 0), dtype=np.int32),
            np.zeros((0, 0), dtype=np.int32),
            np.zeros((0, 0), dtype=np.complex128),
            np.zeros((0, 0), dtype=np.complex128),
            0,
            1,
            2,
            3,
            np.zeros((0, 0), dtype=np.int64),
            np.zeros((0, 0), dtype=np.complex128),
            _near_interface._physical_cpu_count(),
            [K["U"], K["V"], K["W"], K["dzpW"]],
        )
    P1, P2, P3, P4 = Ps
    return (
        P1 @ K["U"],
        P2 @ K["U"],
        P3 @ (k2sq * K["V"] + K["dzpW"]),
        P3 @ K["W"],
        P4 @ K["W"],
        P4 @ K["V"],
    )


def _dense_left_serve(Ps, K):
    """Whether `left_products_gathered` can serve these whole tables: the
    kernel on, real matrices, complex tables of the matrices' width."""
    if not (_LEFT_GATHER and _HAVE_LEFT_GATHER_ACCEL):
        return False
    if any(P.data.dtype != np.float64 for P in Ps):
        return False
    tabs = [K[key] for key in ("U", "V", "W", "dzpW")]
    return all(
        isinstance(t, np.ndarray)
        and t.dtype == np.complex128
        and t.ndim == 2
        and t.shape == tabs[0].shape
        and t.shape[0] == Ps[0].shape[1]
        for t in tabs
    )


# The five-term combine by `combine_rows` (`_accel_left_gather.cpp`,
# momwire#1290) when the accelerator carries it, for EVERY route, so the
# routes agree to the bit by construction on any build. False is scipy's
# products and numpy's sum, the reference the kernel is gated against
# (tests/test_combine_rows_1290.py): the same bits where neither contracts.
_COMBINE_ACCEL = True
_HAVE_COMBINE_ACCEL = _accel.acc is not None and bool(
    getattr(_accel.acc, "combine_rows_1290", False)
)


_NOT_IN_HAND = np.iinfo(np.int64).max


def _combine_serves(Ls, Qz):
    """Whether `combine_rows` takes these operands: the kernel on, real
    matrices, complex products."""
    return (
        _COMBINE_ACCEL
        and _HAVE_COMBINE_ACCEL
        and all(q.data.dtype == np.float64 for q in Qz)
        and all(L.dtype == np.complex128 and L.ndim == 2 for L in Ls)
    )


def _csr_args(Qs):
    """The four distinct matrices of a (Q1, Q2, Q3, Q4, Q3, Q4) list as
    `combine_rows`' (indptrs, indices, data)."""
    Qs = [_sp.csr_array(q) if not isinstance(q, _sp.csr_array) else q for q in Qs[:4]]
    return (
        [np.asarray(q.indptr, dtype=np.int64) for q in Qs],
        [np.asarray(q.indices, dtype=np.int64) for q in Qs],
        [np.ascontiguousarray(q.data) for q in Qs],
    )


def _combine(Ls, Qz):
    """The five-term contraction of the six left products with the right
    weights (`Qz` in `_sandwich_dense`'s (Q1, Q2, Q3, Q4, Q3, Q4) order), in
    the reference term order."""
    if _combine_serves(Ls, Qz):
        n_rows = Qz[0].shape[0]
        return _accel.acc.combine_rows(
            [np.ascontiguousarray(L) for L in Ls],
            [],
            np.arange(Qz[0].shape[1], dtype=np.int64),
            np.arange(n_rows, dtype=np.int64),
            *_csr_args(Qz),
            _near_interface._physical_cpu_count(),
        )
    return (
        Ls[0] @ Qz[0].T
        + Ls[1] @ Qz[1].T
        + Ls[2] @ Qz[2].T
        + Ls[3] @ Qz[3].T
        + Ls[4] @ Qz[4].T
        - Ls[5] @ Qz[5].T
    )


def _compact(buf, live, size):
    """`buf`'s columns `live`, moved to the front of a (rows, size) buffer
    (`buf` itself when it is that size: the gather copies first)."""
    if buf.shape[1] == size:
        buf[:, : live.size] = buf[:, live]
        return buf
    out = np.empty((buf.shape[0], size), dtype=buf.dtype)
    out[:, : live.size] = buf[:, live]
    return out


def _combine_held(Lc, held, colmap, J, q_args):
    """`_combine` of rows `J` of the right weights (`q_args`, `_csr_args`)
    against left-product columns read where they are held: column c at
    `colmap[c]` of the chunk's `Lc` (>= 0) or at -1 - `colmap[c]` of `held`
    (`combine_rows`)."""
    return _accel.acc.combine_rows(
        [np.ascontiguousarray(x) for x in Lc],
        [] if held is None else held,
        colmap,
        J,
        *q_args,
        _near_interface._physical_cpu_count(),
    )


def _streamed_sandwich(
    Ps, Qs4, K, k2sq, rA, rB, out, *, fresh=False, support=None, sink=None
):
    """`_sandwich_dense` over COLUMN CHUNKS of the tables without ever holding
    the six whole (|rA|, |iB|) left products: `out[rA, rB] += block`, the
    same bits as contracting the assembled products (momwire#1168).

    Those six products were the fill's process peak once the tables were
    chunked (#1169): 6 · |rA| · |iB| complex, 169 MB at razor hub_deck(16)
    x8 and 4x that at x16, alive at once beside the one-call evaluation's
    results. Here each chunk's left products live only until every basis
    row of `B` that reads them has been contracted.

    WHY THE BITS DO NOT MOVE. A right factor is sparse, and `L @ Q.T` is
    computed row by row of `Q` (`csr_matvecs`): output column j is a running
    sum, from zero, over row j's stored entries in stored order. So a column
    depends on the left-product columns row j's pattern touches and on
    nothing else, and it is the same sum whenever those columns are all
    present, whatever else is. A row is therefore contracted exactly once,
    at the first chunk that completes its pattern, against its own columns
    gathered in ascending order (`Q[J][:, cols]` keeps each row's stored
    entries and their order); the five-term combine is elementwise, so its
    order per entry is `_combine`'s. The left-product columns themselves are
    the whole-table ones by `_chunked_tables`' argument.

    What is held between chunks is only the columns some still-pending row
    needs: a row's pattern is a few nodes of one wire, except at a junction,
    whose rows reach the first nodes of every wire they join (16 of 642 rows
    at hub_deck(16) x4 span the whole axis). So the live set is one chunk of
    left products plus those junction slivers, not the whole axis.

    COLUMN SETS (momwire#1173 design C). A chunk's `cols` is a slice or an
    ascending index array, and the chunks may arrive in any order, each
    column exactly once: the tiled product route serves a below column when
    its tile completes it, not in axis order. Nothing above used the order —
    a row is contracted when its pattern is COMPLETE (a per-row count of
    columns still to arrive reaches zero), which for in-order ranges is the
    old test `last < cols.stop` — and its columns are gathered by position
    from wherever they are held, into ascending order, so the same entries
    meet the same `Q` row in the same order.

    `fresh` says `out` is a new zero block, so each entry is ASSIGNED, as the
    whole-block path assigns its block (an accumulate would turn a −0.0 into
    +0.0); otherwise the block accumulates into `out` like `_sandwich_dense`'s.

    `_STREAMED_WHOLE_ROWS = False` is the TEST-ONLY negative control: every
    chunk contracts its own columns for every row and the partial sums are
    added, which reassociates each straddling row's running sum.

    `sink(rA, rB[J], block)` (momwire#1173 design C phase 2) takes each
    contracted block in place of `_put` into `out` (then None): the same
    entries, handed over for the caller to place — `_FusedEnds`, which
    finishes the block a unit at a time. It needs whole rows and a fresh
    block (each entry assigned once).
    """
    if sink is not None and not (fresh and _STREAMED_WHOLE_ROWS):
        raise ValueError("a sink takes whole rows of a fresh block")
    Q1, Q2, Q3, Q4 = Qs4
    Qs = (Q1, Q2, Q3, Q4, Q3, Q4)
    nq, n_cols = Q1.shape
    # Each row's node columns: the union of the four STORED patterns (a stored
    # zero is still a term of the running sum, so it counts).
    p1, p2, p3, p4 = (
        _sp.csr_array((np.ones(q.indices.size), q.indices, q.indptr), shape=q.shape)
        for q in Qs4
    )
    pat = p1 + p2 + p3 + p4
    pat.sort_indices()
    del p1, p2, p3, p4
    # Per row, its pattern columns still to arrive; per column, the pending
    # rows that read it (the pattern holds each column once per row).
    remaining = np.diff(pat.indptr).astype(np.int64)
    readers = np.bincount(pat.indices, minlength=n_cols)
    patT = pat.T.tocsr()
    arrived = np.zeros(n_cols, dtype=bool)
    slot = np.full(n_cols, -1, dtype=np.int64)  # a held/new column's position
    pending = np.ones(nq, dtype=bool)
    held_cols = np.zeros(0, dtype=np.int64)
    held = None
    by_rows = _STREAMED_WHOLE_ROWS and _combine_serves([], Qs4)
    if by_rows:
        q_args = _csr_args(Qs4)
        # A column no chunk or held set names reads as out of range, which
        # `combine_rows` refuses ("a row reads a column not in hand").
        colmap = np.full(n_cols, _NOT_IN_HAND, dtype=np.int64)
        # The held columns live in a buffer they are appended to and that is
        # compacted (or grown) only when full, so a chunk copies the columns
        # it adds (the concatenation below copies the whole held set every
        # chunk). `held_slot` is each held column's place in it.
        held_slot = np.zeros(0, dtype=np.int64)
        n_used = 0
    for cols, Kc in K:
        _cancel.poll()
        c_cols = (
            np.arange(cols.start, cols.stop)
            if isinstance(cols, slice)
            else np.asarray(cols, dtype=np.int64)
        )
        if c_cols.size > 1 and not bool(np.all(c_cols[1:] > c_cols[:-1])):
            raise ValueError("a table chunk's columns must be ascending")
        if arrived[c_cols].any():
            raise ValueError("a table column arrived twice")
        arrived[c_cols] = True
        _ROUTES["stream_chunks"] += 1
        Lc = _left_products(Ps, Kc, k2sq)
        del Kc
        if not _STREAMED_WHOLE_ROWS:
            part = _combine(Lc, [q[:, c_cols] for q in Qs])
            _put(out, rA, rB, part, support, assign=False)
            continue
        touched = patT[c_cols]
        remaining -= np.bincount(touched.indices, minlength=nq)
        del touched
        J = np.flatnonzero(pending & (remaining == 0))
        n_old = held_cols.size
        if not by_rows:
            slot[held_cols] = np.arange(n_old)
            slot[c_cols] = n_old + np.arange(c_cols.size)
        if by_rows:
            # `combine_rows` reads each row's columns where they are held
            # (the chunk's, or the held buffer's slot `colmap` names): the
            # same terms the gathered products and sliced rows below hand
            # `_combine`, in the same order, through the same loop.
            colmap[c_cols] = np.arange(c_cols.size)
            if J.size:
                block = _combine_held(Lc, held, colmap, J, q_args)
        elif J.size:
            need = np.unique(pat[J].indices)
            pos = slot[need]
            if (pos < 0).any():
                raise AssertionError("a pending row's column was not held")
            old = pos < n_old
            Ls = []
            for i in range(6):
                L = np.empty((Lc[i].shape[0], need.size), dtype=np.complex128)
                if held is not None:
                    L[:, old] = held[i][:, pos[old]]
                L[:, ~old] = Lc[i][:, pos[~old] - n_old]
                Ls.append(L)
            block = _combine(Ls, [q[J][:, need] for q in Qs])
            del Ls
        if J.size:
            # each entry written once when `fresh`
            if sink is not None:
                sink(rA, rB[J], block)
            else:
                _put(out, rA, rB[J], block, support, assign=fresh)
            del block
            pending[J] = False
            readers -= np.bincount(pat[J].indices, minlength=n_cols)
        if by_rows:
            # Keep only the columns a still-pending row reads: free the
            # released ones' slots, then move the chunk's kept ones in.
            colmap[c_cols] = _NOT_IN_HAND
            keep_old = readers[held_cols] > 0
            if not keep_old.all():
                colmap[held_cols[~keep_old]] = _NOT_IN_HAND
                held_cols, held_slot = held_cols[keep_old], held_slot[keep_old]
            new = np.flatnonzero(readers[c_cols] > 0)
            if new.size:
                cap = 0 if held is None else held[0].shape[1]
                if n_used + new.size > cap:
                    # Full: move the live columns to the front, into a
                    # buffer twice their need when they would fill half.
                    n_live = held_slot.size
                    size = cap
                    if 2 * (n_live + new.size) > cap:
                        size = max(2 * (n_live + new.size), 64)
                    held = [
                        _compact(b, held_slot, size)
                        if b is not None
                        else np.empty((x.shape[0], size), dtype=np.complex128)
                        for b, x in zip(held or [None] * len(Lc), Lc)
                    ]
                    held_slot = np.arange(n_live, dtype=np.int64)
                    colmap[held_cols] = -1 - held_slot
                    n_used = n_live
                slots = np.arange(n_used, n_used + new.size, dtype=np.int64)
                for b, x in zip(held, Lc):
                    b[:, n_used : n_used + new.size] = x[:, new]
                n_used += new.size
                colmap[c_cols[new]] = -1 - slots
                held_cols = np.concatenate((held_cols, c_cols[new]))
                held_slot = np.concatenate((held_slot, slots))
            _ROUTES["stream_held_cols"] = max(
                _ROUTES["stream_held_cols"], held_cols.size
            )
            del Lc
            continue
        slot[held_cols] = -1
        slot[c_cols] = -1
        # Keep only the columns a still-pending row reads.
        keep_old = readers[held_cols] > 0
        keep_new = readers[c_cols] > 0
        if held is None:
            held = [x[:, keep_new] for x in Lc]
        else:
            held = [
                np.concatenate((h[:, keep_old], x[:, keep_new]), axis=1)
                for h, x in zip(held, Lc)
            ]
        held_cols = np.concatenate((held_cols[keep_old], c_cols[keep_new]))
        _ROUTES["stream_held_cols"] = max(_ROUTES["stream_held_cols"], held_cols.size)
        del Lc
    if _STREAMED_WHOLE_ROWS and pending.any():
        raise ValueError("the table chunks did not cover the below axis")
    return out


def _sandwich_dense(
    A,
    B,
    iA,
    iB,
    K,
    k2sq,
    out=None,
    *,
    rows=None,
    out_cols=None,
    support=None,
    sink=None,
):
    """The five-term M+SW+SQ (main) sandwich over dense kernel matrices
    restricted to (iA, iB) — the same term order as the reference fill.

    Only the basis rows with support on those points take part. The rest give
    identically-zero rows and columns of the block, so skipping them is an
    EXACT restriction rather than an approximation: the contractions over the
    points are kept in full, and the surviving entries are the same products
    summed in the same order. Measured bit-for-bit on the #838 BLE decks and
    FAN_SOIL_A_N2 — but that is not pinned, because BLAS chooses kernels by
    shape and by build, so a residual elsewhere would be reassociation in the
    GEMM and not the algebra. The crossgate tolerances are the gate.

    It earns the indirection: on the 60-radial BLE deck the live rows are 5-7
    of 695, so the naive form spends ~99% of its flops multiplying zeros
    (5.13e10 mults across the fill against 6.4e7 restricted), and the routine
    was 63% of that deck family's wall at 113 radials.

    SPARSE @ DENSE @ SPARSE.T since momwire#1109: the weights carry ≤ degree+1
    nonzeros per point, so the two outer products cost
    O(nnz(P)·|iB| + nnz(Q)·|rA|) instead of O(|rA|·|iA|·|iB| + |rA|·|iB|·|rB|),
    and nothing of the (n_basis, n_nodes) samples is ever materialised. The
    term order is the reference fill's, unchanged; the summation order WITHIN
    a term is not, which is the reassociation the docstring above already
    declines to pin.

    `rows` (momwire#1029 phase 2) is a sorted array of global BASIS rows, and
    `out` / `out_cols` are then the `(|rows|, n)` and `(n, |rows|)` blocks the
    route composes Z from. BOTH HALVES READ THE SAME `K` — the kernel tables
    arrive already evaluated, and the six left products `P_i @ K_x` are formed
    once and then row-gathered for the row half and column-restricted on the
    `Q` side for the column half. That is what makes the pair cost one
    evaluation: the transpose the routing reads is a different slice of the
    same product, not a second fill.

    `support` (a `_Support`, whole-block answers only) is the (rows, cols)
    sub-block as its own array: each entry gets exactly the writes it got in
    the full block (`_put`), and the full block's other entries are never
    held."""
    if support is not None and rows is not None:
        raise ValueError("support= and rows= are two answers; ask for one")
    rA = _support_rows(A, iA)
    rB = _support_rows(B, iB)
    Ps = _row_weights(A, iA, rA)
    Q1, Q2, Q3, Q4 = _row_weights(B, iB, rB)
    if sink is not None:
        # Streamed to a sink (`_FusedEnds`), even when the tables came whole:
        # one chunk of every column is `_streamed_sandwich`'s own case, the
        # same contraction per row (see there), so it is the dict route's
        # block entry for entry.
        chunks = [(slice(0, len(iB)), K)] if _whole_tables(K) else K
        return _streamed_sandwich(
            Ps, (Q1, Q2, Q3, Q4), chunks, k2sq, rA, rB, None, fresh=True, sink=sink
        )
    if not _whole_tables(K) and _MAIN_STREAMED:
        if rows is not None:
            raise ValueError("the streamed main sandwich serves whole blocks only")
        fresh = out is None
        if fresh:
            out = (
                np.zeros((A["n_basis"], B["n_basis"]), dtype=np.complex128)
                if support is None
                else support.zeros()
            )
        return _streamed_sandwich(
            Ps, (Q1, Q2, Q3, Q4), K, k2sq, rA, rB, out, fresh=fresh, support=support
        )
    if _whole_tables(K):
        L = _left_products(Ps, K, k2sq)
    else:
        # COLUMN CHUNKS of the tables (`_main_sandwich`, via `_chunked_tables`),
        # assembled whole: the TEST-ONLY `_MAIN_STREAMED = False` reference the
        # streamed path above is gated against. `K` yields `(cols, K_cols)`
        # over positions of `iB`. A sparse @ dense
        # product builds each output column from that column of the dense
        # factor alone, so the assembled left products are the whole-table
        # ones bit for bit, and the contraction over `iB` below is untouched.
        L = tuple(np.empty((rA.size, len(iB)), dtype=np.complex128) for _ in range(6))
        for cols, Kc in K:
            for dst, part in zip(L, _left_products(Ps, Kc, k2sq)):
                dst[:, cols] = part
    Qs = (Q1, Q2, Q3, Q4, Q3, Q4)
    if rows is not None:
        sel, pos = _in_rows(rows, rA)
        if sel.size:
            out[np.ix_(pos, rB)] += _combine([x[sel] for x in L], Qs)
        selc, posc = _in_rows(rows, rB)
        if selc.size:
            out_cols[np.ix_(rA, posc)] += _combine(L, [q[selc] for q in Qs])
        return out
    block = _combine(L, Qs)
    if out is not None:
        # ACCUMULATE IN PLACE (momwire#914). The caller used to write
        # `t_main += _sandwich_dense(...)`, and this function answered with a
        # full (n_basis, n_basis) array carrying the restricted block and
        # zeros everywhere else. On the 48-radial screen that is 146 blocks x
        # 114.5 MB allocated, zeroed and added — 16.7 GB of traffic — to carry
        # a median of 5 live rows by 28 live columns, an occupancy of 0.002 %.
        #
        # BIT-IDENTICAL, not merely close: the entries this skips were exact
        # zeros in the array being added, and adding an exact zero changes no
        # bits. The products and their order are untouched.
        _put(out, rA, rB, block, support, assign=False)
        return out
    if support is not None:
        full = support.zeros()
        _put(full, rA, rB, block, support, assign=True)
        return full
    full = np.zeros((A["n_basis"], B["n_basis"]), dtype=block.dtype)
    full[np.ix_(rA, rB)] = block
    return full


def _axis_segment_tree(geom, seg_idx, leaf):
    """Cluster tree over one axis's SEGMENTS (boxes = segment endpoints).
    Cluster indices are positions into `seg_idx`; returns (tree, seg_idx
    as an array) so callers can map back to global segment ids."""
    idx = np.asarray(seg_idx, dtype=np.int64)
    lo = np.minimum(geom.seg_l[idx], geom.seg_r[idx])
    hi = np.maximum(geom.seg_l[idx], geom.seg_r[idx])
    _cancel.poll()
    tree = _aca.build_cluster_tree(np.arange(idx.size), lo, hi, leaf)
    _cancel.poll()
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


@_cancelable
def cross_complete_block_split(ctx, a_idx, b_idx, A, B, *, corner=True, rows=None):
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
    routes through neither coarse axes nor the low-rank pass.

    `rows` (momwire#1029 phase 2) is a sorted array of global BASIS rows, and
    the answer is then the PAIR `(t[rows, :], t[:, rows])` — what the buried
    routing needs to write `Z[rows] -= t; Z[rows] -= t.T` without the other
    rows. One evaluation of the kernel tables and one ACA factorisation per
    block serve both halves; `rows=None` is the full block and today's path."""
    if _WHOLE_AXIS_NO_ACA:
        t = cross_complete_block(ctx, A, B, corner=corner)
        # The whole-axis switch answers in the restricted shape too, by slicing:
        # it exists to compare the split against the dense fill, so it must
        # stay drivable from every caller the split has.
        return t if rows is None else (t[rows], t[:, rows])
    _refuse_path_tested(A, B)

    # One fill = one memo (eps_t, k_p, _CROSS_RTOL fixed here).
    eps_t, k_p, gz, c1, memo = _block_preamble(ctx)
    main = _main_split(ctx, a_idx, b_idx, A, B, eps_t, k_p, c1, gz, memo, rows=rows)
    if rows is None:
        _ends_and_corner(
            ctx, A, B, eps_t, k_p, c1, gz, memo=memo, corner=corner, out=main
        )
        return main
    ends = _ends_and_corner(
        ctx, A, B, eps_t, k_p, c1, gz, memo=memo, corner=corner, rows=rows
    )
    main[0][:] += ends[0]
    main[1][:] += ends[1]
    return main


@_cancelable
def cross_complete_blocks_two_radius(ctx, a_idx, b_idx, A, B, *, rows=None):
    """The cross pair at a TWO-RADIUS crossing node: `(t_above, t_below)`,
    composed by the caller as `Z -= t_above; Z -= t_below.T`.

    The rule (antennaknobs plan U5; its derivation and measurements are in
    the momwire scratch record `scratch/u5-mixed-radius/`): every evaluation
    takes the radius of its observation point, and the crossing node is ONE
    observation point.

    * Line tests keep their observer wire's radius. The above rows' main
      sandwich and their source-end terms (the below node's by-parts ends seen
      from the above line) are at `ctx.a_above`; the below rows' at
      `ctx.a_below`.
    * Every point test AT the node — the test-side end terms and the corner,
      in both families' node rows — is at ONE radius, `ctx.a_below`. Point
      tests on two surfaces (the observer-side reading) leave the thin-wire
      potential's jump across the node uncounted: an EMF ∝ ln(a_above/a_below)
      that makes the buried member respond as if it had the above wire's
      radius, measured 10–97 Ω from NEC-5 on a two-radius rod ladder.

    The below rows therefore take every term at `a_below` — the one-radius
    block at that radius, transposed by the caller. Continuity through the
    node is closed by the crossing junction's KCL row
    (`BSplineSolver._kcl_row_junctions`): at two radii the split fill's own
    continuity does not converge under refinement.

    `rows` (momwire#1029 phase 2) answers `(t_above[rows, :], t_below[:, rows])`
    — ONE half of each block, because the caller reads `t_above` by row and
    `t_below` by column, and the two are no longer transposes of each other.
    The below block still costs both halves: `cross_complete_block_split`
    answers with the pair and this discards the row half, which is the price of
    one entry point rather than two. A two-radius deck CAN be rotationally
    symmetric — a mast at one radius over a screen at another is the obvious
    one — so the route is not refused here.
    """
    ctx_above = ctx._replace(a_wire=float(ctx.a_above))
    ctx_below = ctx._replace(a_wire=float(ctx.a_below))
    t_below = cross_complete_block_split(ctx_below, a_idx, b_idx, A, B, rows=rows)
    if rows is not None:
        t_below = t_below[1]

    # Keyed on the folded rho_eff, so one memo per radius stays exact.
    eps_t, k_p, gz, c1, memo = _block_preamble(ctx)
    if _WHOLE_AXIS_NO_ACA:
        t_above = _main_sandwich(ctx_above, A, B, eps_t, k_p, c1, gz, memo=memo)
        if rows is not None:
            t_above = t_above[rows]
    else:
        _refuse_path_tested(A, B)
        t_above = _main_split(
            ctx_above, a_idx, b_idx, A, B, eps_t, k_p, c1, gz, memo, rows=rows
        )
        if rows is not None:
            t_above = t_above[0]
    for ctx_e, memo_e, corner_e, kw in (
        (ctx_above, memo, False, {"test_ends": False}),
        (ctx_below, _near_interface.TripleMemo(), True, {"source_ends": False}),
    ):
        if rows is None:
            _ends_and_corner(
                ctx_e,
                A,
                B,
                eps_t,
                k_p,
                c1,
                gz,
                memo=memo_e,
                corner=corner_e,
                out=t_above,
                **kw,
            )
            continue
        ends = _ends_and_corner(
            ctx_e,
            A,
            B,
            eps_t,
            k_p,
            c1,
            gz,
            memo=memo_e,
            corner=corner_e,
            rows=rows,
            **kw,
        )
        t_above += ends[0]
    return t_above, t_below


def _pair_groups(sizes, *, budget_pairs=600_000):
    """Consecutive `[start, stop)` spans of `sizes` whose pair count stays
    under `budget_pairs`, so `_direct_group`'s one `_tables` call is bounded by
    the budget rather than by how many blocks the partition produced.

    A block bigger than the budget is its own group: splitting one block would
    split a `_sandwich_dense` destination, and block sizes are already bounded
    by `_ACA_COST_GUARD`.

    6e5 is MEASURED, not derived, and it has been measured twice.

    Re-measured in momwire#1029 phase 3 (Skylake, 150-radial route, peak
    RSS by `ru_maxrss`, no tracer), once `self_completions` stopped being the
    ceiling (`_bnd_and_corner(rows=)`, and `_ends_to_nodes` counting its whole
    block):

        budget     peak RSS   cold s
        1.5e6      464.0 MB    7.73
        1.0e6      419.5 MB    7.77
        6.0e5      387.5 MB    7.76
        3.0e5      389.4 MB    7.73

    so the knee moved to 6e5 and costs nothing in time. The phase-2 table
    below is kept for the record: it was taken while the completions phase
    held a higher peak, which is why going finer than 1.5e6 then looked
    WORSE. The original measurement (ntot = 3.89e6 pairs):

        budget        peak RSS   _main_split   _main_split s   total s
        one batch      2217.6       964.5 MB       4.57          28.4
        4.0e6          2217.6       964.5 MB       4.55          28.4
        1.5e6          1789.5       429.1 MB       4.97          29.7
        6.0e5          1855.3       224.3 MB       5.64          31.7

    Two things that table says and arithmetic does not. A budget above ntot is
    a no-op -- one group, the original call, main's numbers exactly -- so this
    costs nothing on any deck that already fits. And going FINER than 1.5e6
    makes the peak WORSE (1855.3 against 1789.5) while costing another 2 s:
    below ~429 MB this phase stops being the ceiling, so the transient buys
    nothing, and more groups grow the memo, which is unbounded memory traded
    for a bounded temporary. The knee is real and it is here.
    """
    out, start, run = [], 0, 0
    for i, n in enumerate(sizes):
        if run and run + n > budget_pairs:
            out.append((start, i))
            start, run = i, 0
        run += n
    if start < len(sizes):
        out.append((start, len(sizes)))
    return out


def _direct_group(ctx, eps_t, k_p, gz, k2sq, memo, direct, t_main, t_cols, rows):
    """One GROUP of direct blocks: fill the coordinate buffers, evaluate the
    tables once for the group, and sandwich each block out of the result.

    Lifted out of `_main_split` unchanged (momwire#1126) so it can run per
    group instead of once over every direct block. `_tables` is where the
    memory is, measured at 150 radials: one call over 3,891,840 pairs, peaking
    at 824 MB to return 356 MB, so ~470 MB is its own internals -- and all of
    it scales with the batch length, which is what makes grouping reach it.
    `_sandwich_dense` is not the problem: 146 calls, 17.5 MB peak each.

    Grouping costs no re-evaluation. `_tables` documents `memo` as extending
    the exact-triple dedup ACROSS CALLS (one fill = one memo), so a group keeps
    every hit the single batch would have had; what it costs is one more
    vectorised call.

    It is NOT bit-identical, which is worth stating because it looks as if it
    should be -- each block's own arithmetic is untouched and only the batching
    boundary moves. The memo is why: a triple first seen in one group is served
    back from the memo in the next, where one batch lets `radius_tables` dedup
    it internally, and the two round differently. Measured on
    `p2_default_path`, single batch against 40k-pair groups: rel_dZ_max
    6.93e-19 and rel_dz_in 1.4e-15 (4 radials) / 5.4e-15 (12), P2_7 green on
    both. rel_dZ_max is the SAME at both radial counts, so it is one
    deterministic rounding difference rather than a drift -- and it sits three
    orders below `_ends_to_nodes`' own 1.85e-13, inside the same contract the
    file already declares ("read to scale, never to the bit").
    """
    # One buffer per column, filled block by block in place (momwire#914):
    # ρ by `_direct_coords`, z and z′ broadcast from its per-spec columns.
    rho_all, zAs, zBs, shapes = _direct_coords(direct, gz)
    z_all = np.empty(rho_all.size, dtype=float)
    zp_all = np.empty(rho_all.size, dtype=float)
    off = 0
    for zA, zB, shp in zip(zAs, zBs, shapes):
        sl = slice(off, off + shp[0] * shp[1])
        z_all[sl].reshape(shp)[:] = zA[:, None]
        zp_all[sl].reshape(shp)[:] = zB[None, :]
        off = sl.stop
    tab = _tables(
        ctx,
        eps_t,
        k_p,
        rho_all,
        z_all,
        zp_all,
        _CROSS_RTOL,
        memo=memo,
    )
    off = 0
    for (AX, BX, iA, iB), shp in zip(direct, shapes):
        nel = shp[0] * shp[1]
        K = {kk: tab[kk][off : off + nel].reshape(shp) for kk in _CROSS_KEYS}
        off += nel
        _sandwich_dense(AX, BX, iA, iB, K, k2sq, out=t_main, rows=rows, out_cols=t_cols)


def _main_split(ctx, a_idx, b_idx, A, B, eps_t, k_p, c1, gz, memo, rows=None):
    """The split fill's main sandwich over (above A × below B) — everything
    `cross_complete_block_split` does except the ends and the corner.

    Split out for the same reason as `_main_sandwich` (momwire#813): the
    reversed block's main part is this product transposed.

    `rows` (momwire#1029 phase 2) answers with `(t[rows, :], t[:, rows])`
    instead of the full block. The BLOCK PARTITION and every kernel evaluation
    are untouched by it — the same cluster trees, the same one batched
    `_tables` call, the same one ACA factorisation per kernel per far block —
    because the restriction lives entirely on the weight side."""
    k2sq = k_p * k_p

    tree_a, seg_a = _axis_segment_tree(ctx.geom, a_idx, _CLUSTER_LEAF_SEGS)
    tree_b, seg_b = _axis_segment_tree(ctx.geom, b_idx, _CLUSTER_LEAF_SEGS)
    far, near = _aca.build_block_tree(tree_a, tree_b, _ADM_ETA)

    nA, nB = A["n_basis"], B["n_basis"]
    if rows is None:
        t_main = np.zeros((nA, nB), dtype=np.complex128)
        t_cols = None
    else:
        t_main = np.zeros((rows.size, nB), dtype=np.complex128)
        t_cols = np.zeros((nA, rows.size), dtype=np.complex128)

    if far:
        # Share the dense axes' sampled F/Fd (momwire#919) — see `axis_data`.
        Ac = axis_data(ctx, a_idx, coarse=True, share_from=A)
        Bc = axis_data(ctx, b_idx, coarse=True, share_from=B)

    # ---- direct blocks: near pairs on the dense axes + small far blocks
    # on the coarse axes (sampling a small block costs more than its full
    # coarse product), ALL in one batched designed evaluation so the
    # exact-triple memo dedups across every block — a symmetric screen's
    # mirrored blocks are the same triples.
    direct, far_aca = [], []
    for cs, ct in near:
        iA = _nodes_of(A, seg_a[cs.indices])
        iB = _nodes_of(B, seg_b[ct.indices])
        if iA.size and iB.size:
            direct.append((A, B, iA, iB))
    for cs, ct in far:
        iA = _nodes_of(Ac, seg_a[cs.indices])
        iB = _nodes_of(Bc, seg_b[ct.indices])
        if iA.size == 0 or iB.size == 0:
            continue
        if iA.size * iB.size > _ACA_COST_GUARD * (iA.size + iB.size):
            far_aca.append((iA, iB))
        else:
            direct.append((Ac, Bc, iA, iB))

    # The fill's plane sheets, from the rows it knows before it evaluates
    # any: the direct batch (the ACA samples are not known until they are
    # asked, and they read the same plan).
    _plan_sheets(ctx, eps_t, k_p, gz, memo, direct)
    if direct:
        # Grouped by a budget on PAIRS, not one global batch (momwire#1126).
        # `_tables` costs ~212 bytes of peak per pair at 150 radials, so the
        # budget is what bounds this phase, not the deck size.
        for g0, g1 in _pair_groups([iA.size * iB.size for _A, _B, iA, iB in direct]):
            _cancel.poll()
            _direct_group(
                ctx,
                eps_t,
                k_p,
                gz,
                k2sq,
                memo,
                direct[g0:g1],
                t_main,
                t_cols,
                rows,
            )

    # ---- large far blocks: coarse axes, low-rank ACA per kernel. The
    # row/column samples ride the SAME memo — identical matrices in
    # mirrored blocks pick identical pivots, so their samples dedup.
    for iA, iB in far_aca:
        _cancel.poll()
        pa, pb = Ac["nodes"][iA], Bc["nodes"][iB]
        zA, zB = pa[:, 2] - gz, pb[:, 2] - gz
        rho = np.hypot(
            pa[:, 0][:, None] - pb[:, 0][None, :],
            pa[:, 1][:, None] - pb[:, 1][None, :],
        )
        m, n = iA.size, iB.size
        # `row_s` / `col_s`, not `rows` / `cols`: `rows` is this function's
        # basis-row restriction since momwire#1029 phase 2, and a sample cache
        # that shadowed it would silently restrict nothing.
        row_s, col_s = {}, {}

        def _row6(i, rho=rho, zA=zA, zB=zB, row_s=row_s, n=n):
            if i not in row_s:
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
                row_s[i] = np.stack([te[kk] for kk in _CROSS_KEYS])
            return row_s[i]

        def _col6(j, rho=rho, zA=zA, zB=zB, col_s=col_s, m=m):
            if j not in col_s:
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
                col_s[j] = np.stack([te[kk] for kk in _CROSS_KEYS])
            return col_s[j]

        Kf = {}
        for ki, kk in enumerate(_CROSS_KEYS):
            Kf[kk] = _aca.aca_partial(
                lambda i, ki=ki: _row6(i)[ki],
                lambda j, ki=ki: _col6(j)[ki],
                m,
                n,
                tol=_ACA_TOL,
            )
        # RESTRICTED ON BOTH SIDES AND SCATTERED (momwire#1109) — the same
        # exact restriction #688's G6 licensed for `_sandwich_dense`, which
        # this branch never took: it built the weights at full basis height,
        # four `(n_basis, |iA|)` and four `(n_basis, |iB|)` dense matrices per
        # far block, 528 MB each on the 150-radial screen for a product whose
        # live rows are single digits. The rows outside the support give an
        # identically-zero block, so adding them was adding exact zeros.
        rA = _support_rows(Ac, iA)
        rB = _support_rows(Bc, iB)
        Pw = _row_weights(Ac, iA, rA)
        Qw = _row_weights(Bc, iB, rB)
        # ONE factorisation per kernel serves BOTH halves of a `rows=` answer
        # (momwire#1029 phase 2): `P @ Uf` and `Vf @ Q.T` are cached, and a
        # restriction is a gather on one of them. Nothing above this line
        # depends on `rows`, which is the property the spy gate asserts.
        PU, VQ = {}, {}

        def _lr(pi, kk, qi, rsel=None, csel=None, Kf=Kf, Pw=Pw, Qw=Qw, PU=PU, VQ=VQ):
            Uf, Vf = Kf[kk]
            if (pi, kk) not in PU:
                PU[(pi, kk)] = Pw[pi] @ Uf
            if (kk, qi) not in VQ:
                VQ[(kk, qi)] = Vf @ Qw[qi].T
            left = PU[(pi, kk)]
            right = VQ[(kk, qi)]
            if rsel is not None:
                left = left[rsel]
            if csel is not None:
                right = right[:, csel]
            return left @ right

        def _terms(rsel=None, csel=None, _lr=_lr, k2sq=k2sq):
            return (
                _lr(0, "U", 0, rsel, csel)
                + _lr(1, "U", 1, rsel, csel)
                + k2sq * _lr(2, "V", 2, rsel, csel)
                + _lr(2, "dzpW", 2, rsel, csel)
                + _lr(2, "W", 3, rsel, csel)
                + _lr(3, "W", 2, rsel, csel)
                - _lr(3, "V", 3, rsel, csel)
            )

        if rows is None:
            t_main[np.ix_(rA, rB)] += _terms()
        else:
            sel, pos = _in_rows(rows, rA)
            if sel.size:
                t_main[np.ix_(pos, rB)] += _terms(rsel=sel)
            selc, posc = _in_rows(rows, rB)
            if selc.size:
                t_cols[np.ix_(rA, posc)] += _terms(csel=selc)

    if rows is None:
        return c1 * t_main
    return c1 * t_main, c1 * t_cols


@_cancelable
def cross_complete_block_reversed_split(
    ctx, p_idx, q_idx, P, Q, *, corner=True, sw_end=SW_BY_PARTS
):
    """`cross_complete_block_reversed` through the #688 admissibility split.

    `p_idx` are the BELOW axis's segments (test rows), `q_idx` the ABOVE
    axis's (source columns) — the same argument order as the axes. The main
    sandwich is the forward split's, transposed; the ends follow the roles.
    """
    if _WHOLE_AXIS_NO_ACA:
        return cross_complete_block_reversed(ctx, P, Q, corner=corner, sw_end=sw_end)
    _refuse_path_tested(P, Q)

    eps_t, k_p, gz, c1, memo = _block_preamble(ctx)
    t_ba = _main_split(ctx, q_idx, p_idx, Q, P, eps_t, k_p, c1, gz, memo).T
    _ends_and_corner_reversed(
        ctx, P, Q, eps_t, k_p, c1, gz, memo=memo, corner=corner, sw_end=sw_end, out=t_ba
    )
    return t_ba


def _g_of_r(k, R):
    return np.exp(-1j * k * R) / R


def _fdw_sparse(ax):
    """`Fd · w` as a CSR, built from the axis's own segment tables and cached
    on the axis (momwire#914).

    `Fd` is block-sparse by construction: on the nodes of segment g only the
    rows `seg_rows[g]` can be nonzero, and `seg_runs[g]` gives that segment's
    node run. The pattern the tables imply is EXACT here, not a superset —
    measured 13,192 index pairs against 13,192 actual nonzeros on the N = 113
    BLE below-axis, i.e. 0.23 % density.

    Built from the STRUCTURE rather than by handing the dense array to
    `csr_matrix`: the dense scan is the cost. On that axis the end-to-node
    product is 71.4 ms dense, 28.5 ms via a CSR scanned out of the dense
    array, and 4.2 ms via this one, all three including construction.

    Since momwire#1109 that structure is `Fd_csr`'s own — `axis_data` builds
    the samples at exactly this pattern — so the loop that rebuilt it here
    from `seg_runs` / `seg_rows` is a column scaling, and the branch for an
    axis with no tables (a path-test axis) falls out with it: its Fd is an
    empty CSR and scaling it is the same answer the dense product gave.
    """
    cached = ax.get("_fdw_csr")
    if cached is not None:
        return cached
    out = _scale_cols(ax["Fd_csr"], ax["w"])
    ax["_fdw_csr"] = out
    return out


_ENDS_TO_NODES_BYTES_PER_PAIR = 112


def _ends_to_nodes(k, a2, obs, src, Fsp, *, budget_mb=48.0):
    """`(Fsp @ G(obs, src).T).T` — the (E, n) end-to-node kernel rows, without
    ever building the (E, P) kernel or the (E, P, 3) difference behind it.

    The difference array is what sets this phase's memory, and it is
    quadratic in the radial count because E (wire ends) and P (nodes) each
    grow linearly with it: 23.3 MB at 48 radials, 92.2 MB at 96, ~228 MB
    extrapolated at 150, against 12.2 MB of terms actually kept (momwire#1126).
    Blocking the NODE axis caps it at `budget_mb` instead, and the full
    complex (E, P) kernel never exists either, because `Fsp @ Ge.T` is a sum
    over nodes and so accumulates block by block.

    That accumulation reassociates the sum over P, so this is NOT bit-identical
    to the one-shot form -- which is the contract `_bnd_and_corner` already
    declares above ("same sums, different order of summation ... read to scale,
    never to the bit"; residual 5.7e-14 against the dense form, gated by the
    crossgate tolerance rather than `array_equal`).

    `budget_mb` buys iterations against footprint: it bounds a block's WHOLE
    working set (`_ENDS_TO_NODES_BYTES_PER_PAIR`), so 48 MB is about twenty
    blocks at 150 radials and five at 48 -- small enough to keep the loop's own
    cost out of a phase that is a few percent of the route's wall clock, large
    enough that each block is still a vectorised call rather than a Python
    inner loop.
    """
    E = obs.shape[0]
    P = src.shape[0]
    n = Fsp.shape[0]
    # Bytes a block holds per (end, node) pair, ALL of them: the (E, block, 3)
    # difference (24), the squared distance and R (8 + 8 + 8 of temporaries),
    # and the complex kernel's -jkR, exp and /R (16 each), plus the transposed
    # copy the sparse product takes (16). Counting only the difference let the
    # "48 MB" budget hold ~200 MB at 150 radials (E = 302, P = 32 432 --
    # measured, momwire#1029 phase 3), the route's whole remaining
    # `self_completions` transient.
    per_node = max(E * _ENDS_TO_NODES_BYTES_PER_PAIR, 1)
    block = max(1, min(P, int(budget_mb * (1 << 20) // per_node)))
    if block >= P:
        # One block is the original expression, so take it unchanged rather
        # than paying a CSC conversion and an accumulator for nothing. This
        # is the small-deck path, and it stays bit-identical there.
        d = src[None, :, :] - obs[:, None, :]
        Ge = _g_of_r(k, np.sqrt(a2 + np.einsum("eij,eij->ei", d, d)))
        return (Fsp @ Ge.T).T
    # CSC once, not per block: slicing COLUMNS of a CSR rebuilds it every
    # time, which would trade this function's memory for the loop's time.
    Fc = Fsp.tocsc()
    Gt = np.zeros((E, n), dtype=np.complex128)
    for p0 in range(0, P, block):
        p1 = min(p0 + block, P)
        d = src[None, p0:p1, :] - obs[:, None, :]
        Ge = _g_of_r(k, np.sqrt(a2 + np.einsum("eij,eij->ei", d, d)))
        Gt += (Fc[:, p0:p1] @ Ge.T).T
    return Gt


def _ends_to_nodes_contracted(k, a2, obs, src, Fsp, right, *, budget_mb=48.0):
    """`Fsp @ (G(obs, src).T @ right)` — `_ends_to_nodes`' rows contracted
    against `right` (E, R) inside the node-block loop, so neither the (E, n)
    kernel rows nor their `(n, E)` product ever exist. `Fsp` may be a row
    subset (the sector route's `rows`). The (n, R) answer is all that is
    kept; R is a handful of live rows on the route (momwire#1029 phase 3).

    Reassociates `_ends_to_nodes` followed by a product with `right`, so it
    is read to scale, never to the bit, the contract `_bnd_and_corner`
    already declares.
    """
    E = obs.shape[0]
    P = src.shape[0]
    per_node = max(E * _ENDS_TO_NODES_BYTES_PER_PAIR, 1)
    block = max(1, min(P, int(budget_mb * (1 << 20) // per_node)))
    Fc = Fsp.tocsc()
    out = np.zeros((Fsp.shape[0], right.shape[1]), dtype=np.complex128)
    for p0 in range(0, P, block):
        p1 = min(p0 + block, P)
        d = src[None, p0:p1, :] - obs[:, None, :]
        Ge = _g_of_r(k, np.sqrt(a2 + np.einsum("eij,eij->ei", d, d)))  # (E, b)
        out += Fc[:, p0:p1] @ (Ge.T @ right)
    return out


def _bnd_and_corner(ax, k, a_wire, gz, mirror, rows=None):
    """The same-medium by-parts boundary shape on one axis (β = 1):
    −test-end rows, −source-end columns, +corner — the derivation's
    −,−,+ sign structure. Closed-form kernel G = e^{−jkR}/R at
    R = √(‖Δr‖² + a²), source positions mirrored through the interface
    for the image family.

    Returns `(live, row_term, col_term, corner)` — the SUPPORT of the shape,
    not two dense (n, n) blocks (momwire#914). `fvT` is the ends' basis
    values, and a wire end touches only the basis functions whose support
    reaches it, so every term here is confined to those rows: the row term to
    `live × all`, the column term to `all × live`, the corner to
    `live × live`. Measured 317 live of 1278 on the N = 113 BLE below-axis.
    The caller scatters them; nothing here is ever materialised at (n, n).

    `rows` (the sector route, momwire#1029) asks only for what that caller
    reads: the row term and the corner on the live rows that are also in
    `rows`, and the column term on `rows`. Returned in that restricted shape,
    so neither full `(L, n)` nor `(n, L)` term is ever formed -- at 150
    radials the two were ~415 MB of transient, this phase's whole peak.

    Same sums, different order of summation — as the stacked form this
    replaces already said, read to scale, never to the bit. The residual
    against the dense form measures 5.7e-14 on that axis; the gate is the
    crossgate tolerance, not `array_equal`.
    """
    pts = ax["nodes"]
    n = ax["n_basis"]

    def _mir(p):
        p = np.array(p, dtype=float, copy=True)
        if mirror:
            p[..., 2] = 2.0 * gz - p[..., 2]
        return p

    ends = ax["ends"]
    empty = np.zeros(0, dtype=np.int64)
    if not ends:
        z = np.zeros((0, n), dtype=np.complex128)
        col0 = np.zeros((n if rows is None else rows.size, 0), dtype=np.complex128)
        return empty, z, col0, np.zeros((0, 0), dtype=np.complex128)
    # Every term below is a sum of outer products over the E wire ends, i.e.
    # a rank-E product. Stacked, the three loops are two matmuls:
    #   bnd    = -(FT^T diag(sig) Gt)  -(Gs^T diag(sig) FS)
    #   corner =   FT^T (sig sig^T o G_ee) FS
    # with FT / FS the (E, n) end-value rows, Gt / Gs the (E, n) end-to-node
    # kernel rows folded through Fd*w, and G_ee the (E, E) end-to-end kernel.
    # The old form built an n x n outer product PER end (and per end pair for
    # the corner): O(E n^2 + E^2 n^2) numpy work, which is O(N^3)..O(N^4) in
    # the radial count and was 60 % of a 60-radial BLE solve (16.7 s of 38.6).
    # Same sums, different order of summation: read to scale, never to the bit.
    ptE = np.array([np.asarray(e[0], dtype=float) for e in ends])  # (E, 3)
    sig = np.array([float(e[1]) for e in ends])  # (E,)
    fvT = np.array([np.asarray(e[2], dtype=np.complex128) for e in ends])  # (E, n)
    # The ends' live basis rows. Every term is confined to them, so the two
    # (n, n) blocks this used to allocate and fill were mostly exact zeros.
    live = np.flatnonzero(np.any(fvT != 0, axis=0))
    sf = sig[:, None] * fvT[:, live]  # (E, L)
    src = _mir(pts)  # (P, 3) source nodes (mirrored for the image family)
    pe = _mir(ptE)  # (E, 3) source ends (mirrored)
    a2 = a_wire * a_wire
    Fsp = _fdw_sparse(ax)  # (n, P) CSR of Fd*w
    d = ptE[:, None, :] - pe[None, :, :]
    Gee = _g_of_r(k, np.sqrt(a2 + np.einsum("eij,eij->ei", d, d)))  # (E, E)
    if rows is None:
        # test ends (unmirrored observation) against mirrored source nodes,
        # and source ends (mirrored) against unmirrored observation nodes.
        Gt = _ends_to_nodes(k, a2, ptE, src, Fsp)  # (E, n)
        Gs = _ends_to_nodes(k, a2, pe, pts, Fsp)  # (E, n)
        row_term = -(sf.T @ Gt)  # (L, n)
        col_term = -(Gs.T @ sf)  # (n, L)
        corner = sf.T @ Gee @ sf  # (L, L)
        return live, row_term, col_term, corner
    # The sector route: the same three terms, contracted inside the node
    # loop, so the (E, n) kernel rows `Gt` / `Gs` are never formed.
    sel, _pos = _in_rows(rows, live)
    sfs = sf[:, sel]  # (E, S): the live rows the route reads
    row_term = -_ends_to_nodes_contracted(k, a2, ptE, src, Fsp, sfs).T  # (S, n)
    col_term = -_ends_to_nodes_contracted(k, a2, pe, pts, Fsp[rows], sf)  # (|rows|, L)
    corner = sfs.T @ Gee @ sf  # (S, L)
    return live, row_term, col_term, corner


class _CompletionScatter:
    """Where `self_completions`' three shapes per family are written.

    Three destinations, one order. Each entry sees its row term, then its
    column term, then its corner — the derivation's own order, and the order
    the full `(n, n)` accumulator had — whichever destination it lands in:

    * `rows` given (the sector route): the `(|rows|, n)` block, and the
      columns outside the request are never read because the routing ADDS
      this term without transposing it;
    * otherwise: the shape's own support (momwire#1029 phase 2 unit C).
      `E_r` carries every term on the rows the axes' ENDS touch, `E_c` the
      column terms on the rows they do not, and the scatter at the end adds
      what a full-size `total` would have added. That array was 1.1 GB at 150
      radials, with its own `total[live, :] +=` temporaries on top — 1.99 GB
      at this phase's peak.
    """

    __slots__ = ("dest", "rows", "L", "posL", "E_r", "E_c")

    def __init__(self, dest, n, rows, live_union):
        self.dest, self.rows = dest, rows
        if rows is not None:
            self.L = self.posL = self.E_r = self.E_c = None
            return
        self.L = live_union
        self.posL = _positions(n, live_union)
        self.E_r = np.zeros((live_union.size, n), dtype=np.complex128)
        self.E_c = np.zeros((n, live_union.size), dtype=np.complex128)

    def add(self, live, beta, row_term, col_term, corner):
        if self.rows is not None:
            # `_bnd_and_corner(rows=)` already restricted all three: the row
            # term and the corner to `sel`, the column term to `rows`.
            sel, pos = _in_rows(self.rows, live)
            if sel.size:
                self.dest[pos, :] += beta * row_term
            self.dest[:, live] += beta * col_term
            if sel.size:
                self.dest[np.ix_(pos, live)] += beta * corner
            return
        pl = self.posL[live]
        self.E_r[pl, :] += beta * row_term
        self.E_r[:, live] += beta * col_term[self.L]
        self.E_c[:, pl] += beta * col_term
        self.E_r[np.ix_(pl, live)] += beta * corner

    def flush(self):
        if self.rows is None and self.L.size:
            # `E_r` already holds the L rows' column terms, so drop them from
            # `E_c` rather than add them twice; rows first, then columns.
            self.E_c[self.L, :] = 0.0
            self.dest[self.L, :] += self.E_r
            self.dest[:, self.L] += self.E_c
        return self.dest


@_cancelable
def self_completions(ctx, ax_b, ax_a, *, rows=None, out=None):
    """The self families' missing bnd + corner content, both media, on
    graded axes. Returned as the ADDITIVE Z correction (the fill's
    `Z -= image` convention already folded in: β_dir·(bnd+cor)(G_dir)
    − β_img·(bnd+cor)(G_img) per family).

    `rows` (momwire#1029 phase 2) answers with `total[rows, :]` alone. There is
    no column half here, and that is not an omission: the routing ADDS this
    term without transposing it, so the columns outside the request are never
    read. `out` (unit C) is the matrix to accumulate into — Z itself — so the
    unrestricted answer needs no full-size array of its own either."""
    _eps_t, eps_m, k_p, k_m, c2, a_m = ctx.medium
    gz = float(ctx.ground_z)
    a_wire = float(ctx.a_wire)
    omega, eps0 = ctx.omega, ctx.eps
    n = ax_b["n_basis"]
    if rows is not None:
        dest = np.zeros((rows.size, n), dtype=np.complex128)
    else:
        dest = out if out is not None else np.zeros((n, n), dtype=np.complex128)
    acc = _CompletionScatter(
        dest,
        n,
        rows,
        np.union1d(_end_live_rows(ax_b), _end_live_rows(ax_a)),
    )
    for ax, k, wgt, eps in ((ax_b, k_m, a_m, eps_m), (ax_a, k_p, c2, eps0)):
        beta_dir = 1.0 / (1j * omega * eps * 4 * np.pi)
        beta_img = wgt / (1j * omega * eps * 4 * np.pi)
        for beta, mirror in ((beta_dir, False), (-beta_img, True)):
            live, row_term, col_term, corner = _bnd_and_corner(
                ax, k, a_wire, gz, mirror=mirror, rows=rows
            )
            if live.size == 0:
                continue
            # Scattered rather than added as two dense (n, n) blocks: the shape
            # is confined to `live` on one side or both (momwire#914). The
            # three writes are disjoint in the sense that matters — each adds
            # its own term, exactly as the dense sum did.
            acc.add(live, beta, row_term, col_term, corner)
    return acc.flush()


@_cancelable
def self_completions_two_radius(ctx, ax_b, ax_a, *, rows=None, out=None):
    """`self_completions` at a TWO-RADIUS crossing node.

    Each family's column terms — its line observers against its node's
    by-parts point charge — keep the family's own radius. Its row terms and
    corner are point tests AT the node, so they take the node's one radius,
    `ctx.a_below` (`cross_complete_blocks_two_radius`). For the below family
    that is its own radius, so it is `self_completions`' spelling at
    `a_below`; only the above family splits.
    """
    _eps_t, eps_m, k_p, k_m, c2, a_m = ctx.medium
    gz = float(ctx.ground_z)
    a_above, a_below = float(ctx.a_above), float(ctx.a_below)
    omega, eps0 = ctx.omega, ctx.eps
    n = ax_b["n_basis"]
    if rows is not None:
        dest = np.zeros((rows.size, n), dtype=np.complex128)
    else:
        dest = out if out is not None else np.zeros((n, n), dtype=np.complex128)
    acc = _CompletionScatter(
        dest,
        n,
        rows,
        np.union1d(_end_live_rows(ax_b), _end_live_rows(ax_a)),
    )
    for ax, k, wgt, eps, a_line in (
        (ax_b, k_m, a_m, eps_m, a_below),
        (ax_a, k_p, c2, eps0, a_above),
    ):
        beta_dir = 1.0 / (1j * omega * eps * 4 * np.pi)
        beta_img = wgt / (1j * omega * eps * 4 * np.pi)
        for beta, mirror in ((beta_dir, False), (-beta_img, True)):
            live, row_term, col_term, corner = _bnd_and_corner(
                ax, k, a_below, gz, mirror=mirror, rows=rows
            )
            if live.size == 0:
                continue
            if a_line != a_below:
                _live, _row, col_term, _corner = _bnd_and_corner(
                    ax, k, a_line, gz, mirror=mirror, rows=rows
                )
            acc.add(live, beta, row_term, col_term, corner)
    return acc.flush()
