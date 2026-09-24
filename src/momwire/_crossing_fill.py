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

import os
import sys
import warnings
from typing import NamedTuple, Protocol

import numpy as np
import scipy.sparse as _sp
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
        `BasisPolynomials` does, built from the segment structure and never
        materialised dense, because the dense pair is 4.13 GB of the 8.55 GB
        peak on the 150-radial screen. A sampler that returns dense arrays
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
    for g in seg_idx:
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
    else:
        F, Fd, seg_rows = basis.samples(seg_runs, u_phys)
        F, Fd = _as_csr(F), _as_csr(Fd)

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


def _end_tables_product(ctx, eps_t, k_p, ends, n_nodes, memo, args):
    """`_end_tables` over a `ProductMemo` holding a product: see there for
    why the yields are the span call's to the bit."""
    product = memo.product
    fast = product.fast
    vals = product.vals
    iv, iw = _near_interface.KEYS.index("V"), _near_interface.KEYS.index("W")
    a_wire = float(ctx.a_wire)
    for g0, g1 in _end_groups(len(ends), n_nodes, memo):
        span = ends[g0:g1]
        got = [None] * len(span)
        slow, cols = [], []
        for i, (pt, _sign, _fv) in enumerate(span):
            c = np.broadcast_arrays(*args(pt))
            vrow = _fast_end_rows(fast, a_wire, pt, *c)
            if vrow is not None and _PRODUCT_NEG_CONTROL == "row":
                vrow = (vrow + 1) % vals.shape[0]  # TEST-ONLY: the wrong row
            if vrow is None:
                slow.append(i)
                cols.append(c)
            else:
                got[i] = {"V": vals[vrow, iv], "W": vals[vrow, iw]}
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
    end is not wholly a set of product rows of the two shapes checked here.

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
    if gv.size == n_l and fast.raw is not None:
        g = fast.gdict.get((float(pt[0]), float(pt[1])))
        if (
            g is not None
            and np.all(gv == gv[0])
            and np.array_equal(lv, fast.line_z)
            and np.array_equal(rho, fast.raw[g])
        ):
            zl = fast.zmap[g].get(float(gv[0]))
            if zl is not None:
                _ROUTES["ends_fast_grouped"] += 1
                return fast.rowflat[fast.off[g] + zl * fast.nk[g] + fast.kl_rank[g]]
    if lv.size == n_g and np.all(lv == lv[0]) and np.array_equal(gv, fast.grouped_z):
        rep = rho[fast.gfirst]
        if np.array_equal(rho, rep[fast.grank]):
            r = _near_interface.radius_fold(rep, a_wire)
            l0 = float(lv[0])
            kg = np.empty(rep.size, dtype=np.intp)
            for g, rg in enumerate(r.tolist()):
                k = fast.kmap[g].get((rg, l0))
                if k is None:
                    return None
                kg[g] = k
            _ROUTES["ends_fast_line"] += 1
            grk = fast.grank
            return fast.rowflat[fast.off[grk] + fast.zl_rank * fast.nk[grk] + kg[grk]]
    return None


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


def _main_sandwich(ctx, A, B, eps_t, k_p, c1, gz, memo=None, support=None):
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
    from the grid dedup (`_tables` / `_chunked_tables`) otherwise; the two
    hand `designed_*` the same rows in the same order and serve the same
    tables in the same chunks, so the contraction below cannot tell them
    apart.
    """
    k2sq = k_p * k_p
    nA, nB = A["nodes"].shape[0], B["nodes"].shape[0]
    iA = np.arange(nA)
    iB = np.arange(nB)
    step = max(1, _MAIN_CHUNK_BYTES // (_MAIN_BYTES_PER_PAIR * max(1, nA)))
    tables = _product_route(ctx, eps_t, k_p, A, B, gz, step, memo)
    if tables is not None:
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
        idx = gid[off : off + m][inv].reshape(nA, sl.stop - sl.start)
        off += m
        yield sl, {key: v[idx] for key, v in vals.items()}


# The product route's switches (momwire#1173 design B). `_PRODUCT_TABLES`
# False sends every main sandwich through the grid dedup and `_PRODUCT_ENDS`
# False every end through the lookup path: the in-process references the
# product route is gated against to the bit. `_PRODUCT_NEG_CONTROL` is
# TEST-ONLY and makes the route WRONG on purpose, so the gate can be shown to
# fail: "transpose" maps a one-group product's rows back to the grid in the
# other factor's order (the product order permuted), "split" evaluates the
# rows as two calls (which changes columns' s_min), "row" serves every fast
# end one product row off. `_ROUTES` counts which route ran, so a test can
# prove the new one did.
_PRODUCT_TABLES = True
_PRODUCT_ENDS = True
_PRODUCT_NEG_CONTROL = None
# A product with several groups pays a merge over its candidate triples; it
# is taken only when those are at most this fraction of the grid, and when
# the groups are few enough that their lines are (`_PRODUCT_MAX_GROUP_FRAC`
# of the grouped side). One group is always taken: its candidates ARE the
# distinct triples, and every array it builds is O(distinct + nodes).
_PRODUCT_MAX_CAND_FRAC = 0.25
_PRODUCT_MAX_GROUP_FRAC = 0.25
# Groups beyond this build no fast-end structures (their raw lines would be
# groups x line floats); their ends take the lookup path, which is exact.
_PRODUCT_FAST_MAX_GROUPS = 64
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
        "ends_slow",
        "end_slow_calls",
    ),
    0,
)


def _first_groups(*cols):
    """The distinct rows of equal-length float columns under `!=` (−0.0 with
    0.0; NaN each its own) — `_near_interface._unique_tri`'s grouping rule on
    one or two columns: `(first, rank)`, the index of each group's FIRST
    occurrence with the groups in first-appearance order, and each element's
    group number in that order."""
    n = cols[0].size
    if n == 0:
        return np.zeros(0, dtype=np.intp), np.zeros(0, dtype=np.intp)
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
    inv = np.empty(n, dtype=np.intp)
    inv[idx] = gid
    order = np.argsort(first, kind="stable")
    rank = np.empty_like(order)
    rank[order] = np.arange(order.size)
    return first[order], rank[inv]


def _first_ints(ids):
    """`_first_groups` for a non-negative integer array."""
    if ids.size == 0:
        return np.zeros(0, dtype=np.intp), np.zeros(0, dtype=np.intp)
    _u, first, inv = np.unique(ids, return_index=True, return_inverse=True)
    order = np.argsort(first, kind="stable")
    rank = np.empty_like(order)
    rank[order] = np.arange(order.size)
    return first[order], rank[np.asarray(inv).ravel()]


class _FastEnds(NamedTuple):
    """What `_fast_end_rows` reads, beside the `ProductSet`: the grouped
    side's slot, nodes' z, group and local z rank; the line nodes' z; per
    group its first node, raw ρ line, (x, y) key, z and key maps, row-table
    offset / width and the line's local key ranks; and the concatenated
    row tables. `raw` and the maps are None past `_PRODUCT_FAST_MAX_GROUPS`
    groups (the product then serves the main sandwich only)."""

    grouped_slot: str
    grouped_z: np.ndarray
    grank: np.ndarray
    zl_rank: np.ndarray
    line_z: np.ndarray
    gfirst: np.ndarray
    raw: np.ndarray | None
    gdict: dict
    zmap: list
    kmap: list
    off: np.ndarray
    nk: np.ndarray
    kl_rank: np.ndarray
    rowflat: np.ndarray


def _product_route(ctx, eps_t, k_p, A, B, gz, step, memo):
    """The main sandwich's tables by the PRODUCT route (momwire#1173 design
    B), or None to take the grid dedup — which it counts in `_ROUTES`.

    Taken when the memo is a fresh `ProductMemo` (so, as in the grid route,
    every row is fresh), every node coordinate is finite, and one side's
    nodes fall into few exact-(x, y) groups (`_product_plan`). The rows it
    hands `designed_rows_permuted`, their order and so every column's
    membership, are `_chunked_tables`' to the float; the tables it serves
    are the same floats in the same `(cols, K)` chunks (or the same dict
    when `nB <= step`); and the memo afterwards holds the same rows with the
    same values. See `_product_plan` for the derivation."""
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
    product, fast, chunk_idx = plan
    product.fast = fast
    memo.set_product(product)
    _ROUTES["main_product"] += 1
    _ROUTES["main_product_" + product.slot] += 1
    _ROUTES["main_product_groups"] = max(
        _ROUTES["main_product_groups"], len(product.rowtab)
    )
    cols_j = [(key, _near_interface.KEYS.index(key)) for key in _CROSS_KEYS]
    vals = product.vals
    nB = B["nodes"].shape[0]

    def chunk(sl):
        idx = chunk_idx(sl)
        return {key: vals[:, j][idx] for key, j in cols_j}

    if nB <= step:
        return chunk(slice(0, nB))

    def gen():
        for b0 in range(0, nB, step):
            sl = slice(b0, min(nB, b0 + step))
            yield sl, chunk(sl)

    return gen()


def _product_plan(ctx, eps_t, k_p, A, B, gz):
    """`(ProductSet, _FastEnds, chunk_idx)` for the main sandwich over (above
    axis A × below axis B), or the reason it declines ("nonfinite", "groups",
    "candidates").

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
        float the grid computed there.

    With ONE group nothing needs sorting: the grouped nodes' first
    appearances and the line's are both ascending, so the flat positions run
    in the product's own order — z-major when the grouped side is the above
    one (a·n_B + b with a from z), key-major when it is the below one. That
    is also the design-B prototype's order, and `_product_plan` asserts
    nothing about it: the rows are built either way from positions.

    No float is computed here that the grid route did not compute the same
    way (the one fold per line node is `radius_fold` of the grid's own ρ);
    no sum is formed at all. The evaluation then sees the same rows in the
    same order, so the same columns with the same members (the column rule
    reads only a column's ρ and its members' s — `designed_tables`), and
    returns the same six floats per row; `designed_rows_permuted` only
    defers the copy into row order, which `rowtab` composes back.

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
    if nG > 1 and nG > _PRODUCT_MAX_GROUP_FRAC * G.shape[0]:
        return "groups"
    # z relative to the plane, exactly as `_direct_coords` forms it.
    zA = pa[:, 2] - gz
    zB = pb[:, 2] - gz
    gzv, lzv = (zA, zB) if slot == "z" else (zB, zA)
    x0, y0 = G[gfirst, 0], G[gfirst, 1]
    # The grid's operand order is (above − below).
    if slot == "z":
        raw = np.hypot(x0[:, None] - L[None, :, 0], y0[:, None] - L[None, :, 1])
    else:
        raw = np.hypot(L[None, :, 0] - x0[:, None], L[None, :, 1] - y0[:, None])
    line = _near_interface.radius_fold(raw, float(ctx.a_wire))
    zf, zid = _first_groups(gzv)
    kf, kid = _first_groups(line.ravel(), np.broadcast_to(lzv, line.shape).ravel())
    kid = kid.reshape(nG, nL)
    # Per group: members (ascending), local z and key ranks, candidates.
    members = np.argsort(grank, kind="stable")
    bounds = np.concatenate(([0], np.cumsum(np.bincount(grank, minlength=nG))))
    zl_rank = np.empty(G.shape[0], dtype=np.intp)
    kl_rank = np.empty((nG, nL), dtype=np.intp)
    zids, kids, zfirst, kfirst, nz, nk = [], [], [], [], [], []
    for g in range(nG):
        m_g = members[bounds[g] : bounds[g + 1]]
        f_z, r_z = _first_ints(zid[m_g])
        f_k, r_k = _first_ints(kid[g])
        zl_rank[m_g] = r_z
        kl_rank[g] = r_k
        zids.append(zid[m_g[f_z]])
        kids.append(kid[g, f_k])
        zfirst.append(m_g[f_z])  # the grouped node where each z first occurs
        kfirst.append(f_k)  # the line node where each key first occurs
        nz.append(f_z.size)
        nk.append(f_k.size)
    nz, nk = np.asarray(nz, dtype=np.intp), np.asarray(nk, dtype=np.intp)
    n_cand = int(np.sum(nz * nk))
    if nG > 1 and n_cand > _PRODUCT_MAX_CAND_FRAC * nA * nB:
        return "candidates"

    def flat_pos(g):
        """Flat grid positions a·nB + b of group g's (z, key) candidates."""
        if slot == "z":
            return zfirst[g][:, None] * nB + kfirst[g][None, :]
        return kfirst[g][None, :] * nB + zfirst[g][:, None]

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
        kept_pos = (flat_pos(0) if slot == "z" else flat_pos(0).T).ravel()
        if kept_pos.size > 1 and not bool(np.all(kept_pos[1:] > kept_pos[:-1])):
            raise AssertionError("one group's candidates are not in grid order")
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
    # The rows, from the node pair at each first appearance.
    a_s, b_s = np.divmod(kept_pos, nB)
    del kept_pos
    rows = np.empty((a_s.size, 3), dtype=float)
    if slot == "z":
        rows[:, 0] = line[grank[a_s], b_s]
    else:
        rows[:, 0] = line[grank[b_s], a_s]
    rows[:, 1] = zA[a_s]
    rows[:, 2] = zB[b_s]
    del a_s, b_s
    if _PRODUCT_NEG_CONTROL == "split":
        h = rows.shape[0] // 2
        v1, p1 = _near_interface.designed_rows_permuted(
            eps_t, k_p, rows[:h], rtol=_CROSS_RTOL
        )
        v2, p2 = _near_interface.designed_rows_permuted(
            eps_t, k_p, rows[h:], rtol=_CROSS_RTOL
        )
        p1 = np.arange(h) if p1 is None else p1
        p2 = np.arange(rows.shape[0] - h) if p2 is None else p2
        vals, row_vrow = np.concatenate([v1, v2]), np.concatenate([p1, p2 + h])
    else:
        vals, row_vrow = _near_interface.designed_rows_permuted(
            eps_t, k_p, rows, rtol=_CROSS_RTOL
        )
        if row_vrow is None:
            row_vrow = np.arange(rows.shape[0], dtype=np.intp)
    del rows
    rowtab = [row_vrow[cr] for cr in cand_row]
    del cand_row
    product = _near_interface.ProductSet(
        slot,
        vals,
        row_vrow,
        gzv[zf],
        line.ravel()[kf],
        np.broadcast_to(lzv, line.shape).ravel()[kf],
        rowtab,
        zids,
        kids,
    )
    off = np.concatenate(([0], np.cumsum(nz * nk)[:-1])).astype(np.intp)
    rowflat = np.concatenate([t.ravel() for t in rowtab])
    small = nG <= _PRODUCT_FAST_MAX_GROUPS
    fast = _FastEnds(
        grouped_slot=slot,
        grouped_z=gzv,
        grank=grank,
        zl_rank=zl_rank,
        line_z=lzv,
        gfirst=gfirst,
        raw=raw if small else None,
        gdict=(
            {
                (float(x), float(y)): g
                for g, (x, y) in enumerate(zip(x0.tolist(), y0.tolist()))
            }
            if small
            else {}
        ),
        zmap=(
            [
                {float(gzv[n]): i for i, n in enumerate(zfirst[g].tolist())}
                for g in range(nG)
            ]
            if small
            else []
        ),
        kmap=(
            [
                {
                    (float(line[g, n]), float(lzv[n])): j
                    for j, n in enumerate(kfirst[g].tolist())
                }
                for g in range(nG)
            ]
            if small
            else []
        ),
        off=off,
        nk=nk,
        kl_rank=kl_rank,
        rowflat=rowflat,
    )
    if not small:
        fast = None

    def chunk_idx(sl):
        """(nA, |sl|) value rows of the grid pairs in columns `sl`."""
        if slot == "z":
            g = grank
            base = off[g] + zl_rank * nk[g]
            return rowflat[
                base[:, None]
                + kl_rank[g[:, None], np.arange(sl.start, sl.stop)[None, :]]
            ]
        g = grank[sl]
        base = off[g] + zl_rank[sl] * nk[g]
        return rowflat[base[None, :] + kl_rank[g[None, :], np.arange(nA)[:, None]]]

    return product, fast, chunk_idx


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


def cross_complete_block(ctx, A, B, *, corner=True, support=None):
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
    `_Support`."""
    # One fill = one memo, exactly as `cross_complete_block_split` does it
    # (momwire#1017). This route built none, so momwire#688's cross-call dedup
    # — the whole reason the parameter exists — never fired for `RazorSolver`,
    # whose crossing serve calls straight in here.
    eps_t, k_p, gz, c1, memo = _block_preamble(ctx)
    sup = _Support.of(support, A["n_basis"], B["n_basis"])
    t_ab = _main_sandwich(ctx, A, B, eps_t, k_p, c1, gz, memo=memo, support=sup)
    _ends_and_corner(
        ctx, A, B, eps_t, k_p, c1, gz, memo=memo, corner=corner, out=t_ab, support=sup
    )
    return t_ab


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
    return (M @ v.real) + 1j * (M @ v.imag)


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
    nodes = line["nodes"]

    def args(pt):
        rho_e = np.hypot(pt[0] - nodes[:, 0], pt[1] - nodes[:, 1])
        return (
            rho_e,
            _on_plane_side(np.full_like(rho_e, pt[2] - gz), "above", "end point"),
            _on_plane_side(nodes[:, 2] - gz, "below", "quadrature node"),
        )

    return args


def _below_end_args(line, gz):
    """`args(pt)` for `_end_tables`: a BELOW end point against an ABOVE line's
    quadrature nodes — the line in the `z` slot, the end in `z′` (the designed
    tables accept only z ≥ 0 ≥ z′, whichever role each side plays). The
    mirror of `_above_end_args`, on the same `_on_plane_side` rule."""
    nodes = line["nodes"]

    def args(pt):
        rho_e = np.hypot(nodes[:, 0] - pt[0], nodes[:, 1] - pt[1])
        return (
            rho_e,
            _on_plane_side(nodes[:, 2] - gz, "above", "quadrature node"),
            _on_plane_side(np.full_like(rho_e, pt[2] - gz), "below", "end point"),
        )

    return args


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
    per entry (`_Support`)."""
    nA, nB = R["n_basis"], C["n_basis"]
    sup = support
    if sup is not None and rows is not None:
        raise ValueError("support= and rows= are two answers; ask for one")
    if rows is None:
        # THE SHAPE'S OWN SUPPORT, never an (n, n) transient (momwire#1029
        # phase 2 unit C, the momwire#914 pattern). `E_r` carries every term
        # on the rows R's ends touch — the row terms over all columns, and,
        # where they meet, the column terms and the corner as well — and `E_c`
        # the column terms on the rows R's ends do NOT touch. So each entry
        # accumulates in one place, in the order the loops below write it, and
        # the scatter at the end adds what the full block would have added:
        # BIT-IDENTICAL to `dest += <a full-size ends block>`, which is what
        # this replaces. Its 1.06 GB at 150 radials was the third-largest term
        # of the fill's peak.
        LA, LB = _end_live_rows(R), _end_live_rows(C)
        posLA, posLB = _positions(nA, LA), _positions(nB, LB)
        if sup is None:
            dest = out if out is not None else np.zeros((nA, nB), dtype=np.complex128)
            E_r = np.zeros((LA.size, nB), dtype=np.complex128)
            E_c = np.zeros((nA, LB.size), dtype=np.complex128)
        else:
            dest = out if out is not None else sup.zeros()
            E_r = np.zeros((LA.size, sup.cols.size), dtype=np.complex128)
            E_c = np.zeros((sup.rows.size, LB.size), dtype=np.complex128)
        t_r = t_c = None
    else:
        dest = None
        t_r = np.zeros((rows.size, nB), dtype=np.complex128)
        t_c = np.zeros((nA, rows.size), dtype=np.complex128)

    def add_rows(nz, fv_nz, vec, scale, buf):
        """`t[nz, :] += scale * outer(fv_nz, vec)`, into whichever blocks
        this call is answering with. `nz` is an R end's live rows, so it is
        inside `LA` by construction."""
        if rows is None:
            if sup is not None:
                vec = vec[sup.cols]
            _rank1_add(E_r, posLA[nz], fv_nz, vec, scale, buf)
            return
        sel, pos = _in_rows(rows, nz)
        if sel.size:
            _rank1_add(t_r, pos, fv_nz[sel], vec, scale, buf)
        _rank1_add(t_c, nz, fv_nz, vec[rows], scale, buf)

    def add_cols(nz, vec, fv_nz, scale, buf):
        """`t[:, nz] += scale * outer(vec, fv_nz)`; `nz` is inside `LB`."""
        if rows is None:
            if sup is None:
                _rank1_add_cols(E_r, nz, vec[LA], fv_nz, scale, buf)
                _rank1_add_cols(E_c, posLB[nz], vec, fv_nz, scale, buf)
                return
            pc = sup.pos_c[nz]
            keep = pc >= 0
            _rank1_add_cols(E_r, pc[keep], vec[LA], fv_nz[keep], scale, buf)
            _rank1_add_cols(E_c, posLB[nz], vec[sup.rows], fv_nz, scale, buf)
            return
        _rank1_add_cols(t_r, nz, vec[rows], fv_nz, scale, buf)
        sel, pos = _in_rows(rows, nz)
        if sel.size:
            _rank1_add_cols(t_c, pos, vec, fv_nz[sel], scale, buf)

    def add_corner(nza, nzb, fva, fvb, scale):
        if rows is None:
            if sup is not None:
                pc = sup.pos_c[nzb]
                keep = pc >= 0
                nzb, fvb = pc[keep], fvb[keep]
            E_r[np.ix_(posLA[nza], nzb)] += scale * np.outer(fva, fvb)
            return
        sel, pos = _in_rows(rows, nza)
        if sel.size:
            t_r[np.ix_(pos, nzb)] += scale * np.outer(fva[sel], fvb)
        selb, posb = _in_rows(rows, nzb)
        if selb.size:
            t_c[np.ix_(nza, posb)] += scale * np.outer(fva, fvb[selb])

    def answer():
        if rows is not None:
            return (t_r, t_c)
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

    _txR, _tyR, tzR = R["t"].T

    # THE NODE WEIGHTS FOLD INTO THE SHORT VECTOR, NOT THE TALL MATRIX
    # (momwire#919). `(Fd_C * w_C) @ te["V"]` is `Fd_C @ (w_C * te["V"])`: the
    # same contraction over the nodes, but the weighting is applied to a
    # length-n_nodes vector instead of an (n_basis, n_nodes) matrix. On the
    # 48-radial screen that product was 222.6 MB, live for the whole routine
    # and second only to the rank-1 updates below. Since momwire#1109 the
    # matrix is a CSR and a folded copy would be cheap, and the fold still
    # stands: it is one vector multiply either way.
    #
    # NOT bit-identical to the old spelling: (Fd*w)·V and Fd·(w*V) round
    # differently. Gated at 1e-12 relative. One spelling for both
    # orientations since momwire#1168 U5, so the reversed block can no longer
    # reassociate one side alone (`test_the_main_sandwich_is_the_forward_
    # transposed` pins the two bit-equal, and once broke in CI on exactly that).
    wR = R["w"]
    wC = C["w"]
    wR_tz = wR * tzR
    _txC, _tyC, tzC = C["t"].T
    wC_tz = wC * tzC
    buf = _Rank1Buffer()
    bufT = _Rank1Buffer()

    # The end loops' tables come a span of ends per call (`_end_tables`); the
    # rank-1 updates below still run one end at a time, in the order they did.
    for pt, sign, fv, te in _end_tables(
        ctx,
        eps_t,
        k_p,
        R["ends"] if row_ends else [],
        C["nodes"].shape[0],
        memo,
        row_args,
    ):
        # momwire#912: `fv` is a value-1 tent's end value — a handful of
        # nonzeros in n_basis — so the rank-1 update lands on those rows only.
        # The same products where fv != 0; where it is 0 the full outer
        # added an exact 0.
        nz = np.flatnonzero(fv)
        add_rows(nz, fv[nz], _real_matvec_c(C["Fd_csr"], wC * te["V"]), c1 * sign, buf)
        # The W end on the row axis's ends, contracting the column line's t̂z:
        # TW (momwire#956) in the forward block — the test-side W end,
        # −σ f_m(E)·∫ f_n t̂z′ W(E,·), left by testing −∇W along the wire,
        # SW's partner on the other axis — and SW itself in the reversed
        # block, where it stays paired with `s_w1` by the by-parts that
        # produced it (momwire#813 derivation (b), 5312ca5).
        add_rows(
            nz, fv[nz], _real_matvec_c(C["F_csr"], wC_tz * te["W"]), -c1 * sign, buf
        )
    for pt, sign, fv, te in _end_tables(
        ctx,
        eps_t,
        k_p,
        C["ends"] if col_ends else [],
        R["nodes"].shape[0],
        memo,
        col_args,
    ):
        nz = np.flatnonzero(fv)
        add_cols(
            nz, _real_matvec_c(R["F_csr"], wR_tz * te["W"]), fv[nz], -c1 * sign, bufT
        )
        add_cols(nz, _real_matvec_c(R["Fd_csr"], wR * te["V"]), fv[nz], c1 * sign, bufT)

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
    # carries its by-parts terms above but no corner. Symmetric in the two
    # ends' one-hots, so the reversed block's is the forward's transposed
    # and needs no orientation of its own.
    if not corner:
        # A path-tested row (momwire#813): its in-plane endpoint is a plain
        # potential evaluation at z = 0⁺ (the BT term above), and the corner
        # is a Galerkin by-parts term it never had. Measured on momwire#651's
        # probe: with the corner the razor node row is off by 1.9e5 where
        # razor's own kernel has none; without it, 5e-5 (quadrature).
        return answer()
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
            add_corner(nza, nzb, fv_a[nza], fv_b[nzb], -sig_a * sig_b * c1 * v_corner)
    return answer()


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


def cross_complete_block_reversed(
    ctx, P, Q, *, corner=True, sw_end=SW_BY_PARTS, support=None
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
    t_ba = _main_sandwich(ctx, Q, P, eps_t, k_p, c1, gz, memo=memo, support=sup_t).T
    # Accumulated in place through the support scatter, bit-identical to
    # `t_ba += <the full ends block>` (`_ends_and_corner_rc`).
    _ends_and_corner_reversed(
        ctx,
        P,
        Q,
        eps_t,
        k_p,
        c1,
        gz,
        memo=memo,
        corner=corner,
        sw_end=sw_end,
        out=t_ba,
        support=sup,
    )
    return t_ba


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
    P1, P2, P3, P4 = Ps
    return (
        P1 @ K["U"],
        P2 @ K["U"],
        P3 @ (k2sq * K["V"] + K["dzpW"]),
        P3 @ K["W"],
        P4 @ K["W"],
        P4 @ K["V"],
    )


def _combine(Ls, Qz):
    """The five-term contraction of the six left products with the right
    weights (`Qz` in `_sandwich_dense`'s (Q1, Q2, Q3, Q4, Q3, Q4) order), in
    the reference term order."""
    return (
        Ls[0] @ Qz[0].T
        + Ls[1] @ Qz[1].T
        + Ls[2] @ Qz[2].T
        + Ls[3] @ Qz[3].T
        + Ls[4] @ Qz[4].T
        - Ls[5] @ Qz[5].T
    )


def _streamed_sandwich(Ps, Qs4, K, k2sq, rA, rB, out, *, fresh=False, support=None):
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

    `fresh` says `out` is a new zero block, so each entry is ASSIGNED, as the
    whole-block path assigns its block (an accumulate would turn a −0.0 into
    +0.0); otherwise the block accumulates into `out` like `_sandwich_dense`'s.

    `_STREAMED_WHOLE_ROWS = False` is the TEST-ONLY negative control: every
    chunk contracts its own columns for every row and the partial sums are
    added, which reassociates each straddling row's running sum.
    """
    Q1, Q2, Q3, Q4 = Qs4
    Qs = (Q1, Q2, Q3, Q4, Q3, Q4)
    nq = Q1.shape[0]
    # Each row's node columns: the union of the four STORED patterns (a stored
    # zero is still a term of the running sum, so it counts).
    p1, p2, p3, p4 = (
        _sp.csr_array((np.ones(q.indices.size), q.indices, q.indptr), shape=q.shape)
        for q in Qs4
    )
    pat = p1 + p2 + p3 + p4
    pat.sort_indices()
    counts = np.diff(pat.indptr)
    last = np.full(nq, -1, dtype=np.int64)
    has = counts > 0
    last[has] = pat.indices[pat.indptr[1:][has] - 1]
    pending = np.ones(nq, dtype=bool)
    held_cols = np.zeros(0, dtype=np.int64)
    held = None
    seen = 0
    for cols, Kc in K:
        if cols.start != seen:
            raise ValueError("the table chunks must arrive in column order")
        seen = cols.stop
        Lc = _left_products(Ps, Kc, k2sq)
        del Kc
        c_cols = np.arange(cols.start, cols.stop)
        if not _STREAMED_WHOLE_ROWS:
            part = _combine(Lc, [q[:, cols] for q in Qs])
            _put(out, rA, rB, part, support, assign=False)
            continue
        J = np.flatnonzero(pending & (last < cols.stop))
        if J.size:
            need = np.unique(pat[J].indices)
            n_old = int(np.searchsorted(need, cols.start))
            pos_old = np.searchsorted(held_cols, need[:n_old])
            if not np.array_equal(held_cols[pos_old], need[:n_old]):
                raise AssertionError("a pending row's column was not held")
            pos_new = need[n_old:] - cols.start
            if held is None:
                Ls = [x[:, pos_new] for x in Lc]
            else:
                Ls = [
                    np.concatenate((h[:, pos_old], x[:, pos_new]), axis=1)
                    for h, x in zip(held, Lc)
                ]
            block = _combine(Ls, [q[J][:, need] for q in Qs])
            del Ls
            # each entry written once when `fresh`
            _put(out, rA, rB[J], block, support, assign=fresh)
            del block
            pending[J] = False
        # Keep only the columns a still-pending row reads.
        keep = np.unique(pat[np.flatnonzero(pending)].indices)
        keep_old = np.isin(held_cols, keep)
        keep_new = np.isin(c_cols, keep)
        if held is None:
            held = [x[:, keep_new] for x in Lc]
        else:
            held = [
                np.concatenate((h[:, keep_old], x[:, keep_new]), axis=1)
                for h, x in zip(held, Lc)
            ]
        held_cols = np.concatenate((held_cols[keep_old], c_cols[keep_new]))
        del Lc
    if _STREAMED_WHOLE_ROWS and pending.any():
        raise ValueError("the table chunks did not cover the below axis")
    return out


def _sandwich_dense(
    A, B, iA, iB, K, k2sq, out=None, *, rows=None, out_cols=None, support=None
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
    if not isinstance(K, dict) and _MAIN_STREAMED:
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
    if isinstance(K, dict):
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


def _pair_groups(sizes, *, budget_pairs=1_500_000):
    """Consecutive `[start, stop)` spans of `sizes` whose pair count stays
    under `budget_pairs`, so `_direct_group`'s one `_tables` call is bounded by
    the budget rather than by how many blocks the partition produced.

    A block bigger than the budget is its own group: splitting one block would
    split a `_sandwich_dense` destination, and block sizes are already bounded
    by `_ACA_COST_GUARD`.

    1.5e6 is MEASURED, not derived. The 150-radial route (ntot = 3.89e6 pairs):

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

    if direct:
        # Grouped by a budget on PAIRS, not one global batch (momwire#1126).
        # `_tables` costs ~212 bytes of peak per pair at 150 radials, so the
        # budget is what bounds this phase, not the deck size.
        for g0, g1 in _pair_groups([iA.size * iB.size for _A, _B, iA, iB in direct]):
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

    `budget_mb` buys iterations against footprint: the (E, block, 3) double
    difference is the term it bounds, so 48 MB is about five blocks at 150
    radials and one at 48 -- small enough to keep the loop's own cost out of a
    phase that is 8.3 % of the route's wall clock, large enough that each
    block is still a vectorised call rather than a Python inner loop.
    """
    E = obs.shape[0]
    P = src.shape[0]
    n = Fsp.shape[0]
    per_node = max(E * 3 * 8, 1)  # bytes of the (E, block, 3) difference
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


def _bnd_and_corner(ax, k, a_wire, gz, mirror):
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
        return empty, z, z.T.copy(), np.zeros((0, 0), dtype=np.complex128)
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
    # test ends (unmirrored observation) against mirrored source nodes, and
    # source ends (mirrored) against unmirrored observation nodes.
    Gt = _ends_to_nodes(k, a2, ptE, src, Fsp)  # (E, n)
    Gs = _ends_to_nodes(k, a2, pe, pts, Fsp)  # (E, n)
    row_term = -(sf.T @ Gt)  # (L, n)
    col_term = -(Gs.T @ sf)  # (n, L)
    d = ptE[:, None, :] - pe[None, :, :]
    Gee = _g_of_r(k, np.sqrt(a2 + np.einsum("eij,eij->ei", d, d)))  # (E, E)
    corner = sf.T @ Gee @ sf  # (L, L)
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
            sel, pos = _in_rows(self.rows, live)
            if sel.size:
                self.dest[pos, :] += beta * row_term[sel]
            self.dest[:, live] += beta * col_term[self.rows]
            if sel.size:
                self.dest[np.ix_(pos, live)] += beta * corner[sel]
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
                ax, k, a_wire, gz, mirror=mirror
            )
            if live.size == 0:
                continue
            # Scattered rather than added as two dense (n, n) blocks: the shape
            # is confined to `live` on one side or both (momwire#914). The
            # three writes are disjoint in the sense that matters — each adds
            # its own term, exactly as the dense sum did.
            acc.add(live, beta, row_term, col_term, corner)
    return acc.flush()


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
                ax, k, a_below, gz, mirror=mirror
            )
            if live.size == 0:
                continue
            if a_line != a_below:
                _live, _row, col_term, _corner = _bnd_and_corner(
                    ax, k, a_line, gz, mirror=mirror
                )
            acc.add(live, beta, row_term, col_term, corner)
    return acc.flush()
