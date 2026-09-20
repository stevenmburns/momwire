"""U5 step (d): the instrument step (c) needed, and the deck it runs on.

Step (c) (`c_within_side_ladder.py`) asked what radius the crossing node's
point test takes when the ABOVE side carries two radii, and could not answer:
the two candidate spellings of `a_above` sat 3e-4 to 2e-3 ohm apart against a
momwire-vs-engine gap of 0.35 ohm, so the winner flipped between rungs. Two
figures of merit say why, and this module measures both:

    S  = |Z(a_above = A) - Z(a_above = B)|      how much the choice moves Z
    D  = |Z_momwire - Z_engine|                 the gap the oracle can see past

On step (c)'s rod S/D was 0.006. This module raises it two ways.

## 1. Geometry: the knob is set by h/a at the node, not by a alone

`a_above` is the observer radius of the ABOVE family's line tests in the
crossing block, so it only matters where the node-adjacent segment length h is
comparable to the wire radius a. Measured here (`d_sd_map.py`): at a fixed
deck S rises about linearly with a, and then by another 30x as h/a falls from
64 to 1.7. Step (c)'s rod sits at h/a = 200. The discriminating decks are fat
at the node and meshed to within about twice the radius of it.

That alone does NOT reach an adjudicating S/D — see the map. D has a floor of
3-10 ohm that no mesh or quadrature setting moves.

## 2. Instrument: a DIFFERENTIAL residual, which removes that floor

The floor is a formulation difference at the node itself, and it is COMMON to
both candidates, so subtract it. Two decks, differing in ONE wire's radius:

    SPREAD   above = [fed stub a_node, radiator a_rad],  below = a_rise
    CONTROL  above = [fed stub a_node, radiator a_node],  below = a_rise

The conductor AT the node is the same wire in both, so the engine's node
treatment is identical in both and its response

    delta_engine = Z_engine(SPREAD) - Z_engine(CONTROL)

is a pure radiator-radius-step response with no node-convention content in it.
momwire must reproduce that response. Define

    E(c) = [Z_mw(c, SPREAD) - Z_engine(SPREAD)]
         - [Z_mw(CONTROL)   - Z_engine(CONTROL)]

for each candidate c. The node gap cancels for any rule that holds `a_above`
fixed across the pair, and does not cancel for any rule that lets `a_above`
depend on the radiator's radius. E is therefore a direct measurement of
whether the far wire belongs in `a_above` at all, and the map reports S/E
beside S/D.

The outcome is not assumed by the construction: if the correct `a_above` were
the radiator's radius, momwire holding it fixed would be wrong on SPREAD and
right on CONTROL, and E would come out near S instead of near zero. Sweeping
`a_above` continuously (`d_verdict_ladder.py --sweep`) shows |E| rising to
several ohm on both sides of its minimum, which is what says E can see.

## The control must take the SAME code path

A deck with one radius everywhere takes `cross_complete_block_split` and drops
the crossing junction's KCL row; a two-radius deck takes
`cross_complete_blocks_two_radius` and keeps it. At EQUAL radii those two
paths do not agree, and by a deck-dependent amount: **0.0152 ohm** on the
soil-A fat rod, **0.119 ohm** on the poor-soil headline rod. The second is
LARGER than that rod's whole differential residual (0.087 ohm), so a control
built from an all-one-radius deck would have carried a bias bigger than the
thing being measured. The control is therefore built with
`a_rise = a_node * (1 - 1e-9)`: a two-radius deck by the scope rule's test,
physically the same rod (1e-9 vs 1e-6 in a_rise moves Z by 5e-5 ohm), and
served by production without any patch. `assert_rung` gates it, by counting
entries into the two-radius fill on the CONTROL as well as on the spread deck.

`crossing_side_radii` is patched, as step (c) patches it, because on a spread
deck there is no side-wide above radius to resolve and production refuses
rather than answering. Everything else is production.

## Two levers that are already spent, measured rather than assumed

* `n_qp_pair`. The buried fill's default is 32. On the headline deck 8 -> 32
  moves Z by 0.139 ohm and 32 -> 128 by 0.0032 ohm, against a D of 7.9 ohm. The
  order is converged at the default; it is not a lever on D here.
* `ctx.a_wire`, which the solver sets to `min(a_above, a_below)`. Poisoning it
  with NaN moves Z by 5.9e-4 ohm on a two-radius crossing deck, so it is
  effectively unread there and S is a pure `a_above` effect, not a mixture of
  the two.

Courtesy: engine results here are from our licensed materials; no internals
are quoted.
"""

from __future__ import annotations

import time
from contextlib import contextmanager
from types import MappingProxyType

import numpy as np

import antennaknobs.engines.momwire as mwe
from antennaknobs.designs.verticals.buried_radial_vertical import Builder
from antennaknobs.engines.momwire import MomwireEngine
from antennaknobs.engines.nec5 import NEC5Engine
from antennaknobs.wire_catalog import Wire, WireSpec, graded_wire
from momwire import _crossing_fill

SOIL_A = ("finite", 13.0, 0.005)
SOIL_POOR = ("finite", 5.0, 0.001)


class StepRod(Builder):
    """Step (c)'s rod with the radius step placed at `z_split`.

        wire 0  rise    hub -> node          a_rise   BELOW
        wire 1  fed     node -> eps_gap      a_node   ABOVE, AT the node
        wire 2  mid     eps_gap -> z_split   a_node   ABOVE          (if any)
        wire 3  far     z_split -> height    a_rad    ABOVE

    `z_split == eps_gap` puts the step at the first vertex above the node,
    which is the live case. `a_rad == a_node` is the CONTROL, wire for wire and
    segment for segment the same mesh.

    `r_node` refines the node panels and the fed wire; `r_far` refines only the
    far mesh. They are separate because the node mesh cannot go below the wire
    radius without leaving the thin-wire regime, while the far mesh can refine
    freely -- and the map shows D is not what the far mesh controls.
    """

    default_params = MappingProxyType(
        {
            **Builder.default_params,
            "r_node": 1,
            "r_far": 1,
            "a_node": 0.00025,
            "a_rad": 0.00025,
            "a_rise": 0.00025,
            "eps_gap": 0.05,
            "z_split": 0.05,
            "rest_h": 0.025,
            "h_node_below": 0.0125,
            "h_node_above": 0.0125,
            "growth": 4.0,
            "n_fed": 3,
        }
    )

    def build_wires(self):
        rn, rf = int(self.r_node), int(self.r_far)
        height = 0.25 * self.design_wavelength * self.length_factor
        eps, zs = float(self.eps_gap), float(self.z_split)
        node, hub = (0.0, 0.0, 0.0), (0.0, 0.0, -self.depth)
        at_node = WireSpec(radius=float(self.a_node))
        radiator = WireSpec(radius=float(self.a_rad))
        rise = WireSpec(radius=float(self.a_rise))
        seg_h = 0.25 * self.design_wavelength / self.nominal_nsegs / rf
        wires = [
            graded_wire(
                hub,
                node,
                toward="p1",
                per_panel=2 * rn,
                rest_h=float(self.rest_h) / rf,
                h_node=float(self.h_node_below),
                growth=float(self.growth),
                spec=rise,
            ),
            Wire(
                node,
                (0.0, 0.0, eps),
                n_seg=max(1, int(self.n_fed) * rn),
                ex=1 + 0j,
                spec=at_node,
            ),
        ]
        if zs > eps * (1.0 + 1e-12):
            wires.append(
                graded_wire(
                    (0.0, 0.0, eps),
                    (0.0, 0.0, zs),
                    toward="p0",
                    per_panel=2 * rn,
                    rest_h=seg_h,
                    h_node=min(float(self.h_node_above), (zs - eps) * 0.4),
                    growth=float(self.growth),
                    spec=at_node,
                )
            )
        wires.append(
            graded_wire(
                (0.0, 0.0, zs),
                (0.0, 0.0, height),
                toward="p0",
                per_panel=2 * rn,
                rest_h=seg_h,
                h_node=min(float(self.h_node_above), (height - zs) * 0.4),
                growth=float(self.growth),
                spec=radiator,
            )
        )
        return wires


@contextmanager
def even_parity():
    """Both engines on ONE mesh and one knot source (b_rod_ladder's patch)."""
    orig = mwe._parity_for_solver
    mwe._parity_for_solver = lambda solver, solver_kwargs: "even"
    try:
        yield
    finally:
        mwe._parity_for_solver = orig


@contextmanager
def a_above_is(a_above, a_below):
    """Name the crossing node's two radii explicitly.

    On a within-side spread there is no side-wide above radius to resolve, so
    production refuses; naming the candidate is the only way to ask the
    question. `crossing_side_radii` is reached twice (the scope check and
    `BSplineSolver._two_radius_crossing`), and this covers both.
    """
    from momwire import _below_interface

    orig = _below_interface.crossing_side_radii
    _below_interface.crossing_side_radii = lambda media, radii: (
        float(a_above),
        float(a_below),
    )
    try:
        yield
    finally:
        _below_interface.crossing_side_radii = orig


@contextmanager
def crossing_calls(counter):
    """Count entries into the two-radius crossing fill."""
    orig = _crossing_fill.cross_complete_blocks_two_radius

    def wrapped(*args, **kwargs):
        counter["two_radius"] = counter.get("two_radius", 0) + 1
        return orig(*args, **kwargs)

    _crossing_fill.cross_complete_blocks_two_radius = wrapped
    try:
        yield counter
    finally:
        _crossing_fill.cross_complete_blocks_two_radius = orig


def build(**kw):
    b = StepRod()
    b.nominal_nsegs = kw.pop("nn", 42)
    b.design_eps_r = kw.pop("design_eps_r", 13.0)
    b.design_sigma = kw.pop("design_sigma", 0.005)
    for k, v in kw.items():
        setattr(b, k, v)
    return b


def control_of(cfg):
    """The CONTROL deck: the radiator's radius set to the node member's."""
    return {**cfg, "a_rad": cfg["a_node"]}


def nec5_segs(b, soil=SOIL_A):
    deck = NEC5Engine(b, ground=soil).deck([b.freq])
    return sum(int(ln.split()[2]) for ln in deck.splitlines() if ln.startswith("GW "))


def z_nec5(b, soil=SOIL_A):
    return complex(NEC5Engine(b, ground=soil).impedance()[0])


def z_momwire(b, a_above=None, a_below=None, soil=SOIL_A, **kw):
    """Patched when `a_above` is named, plain production when it is not."""
    with even_parity():
        if a_above is None:
            return complex(MomwireEngine(b, ground=soil, **kw).impedance()[0])
        with a_above_is(a_above, a_below):
            return complex(MomwireEngine(b, ground=soil, **kw).impedance()[0])


def node_h(b, soil=SOIL_A):
    """(h_above, h_below): the segment lengths touching the crossing node."""
    with even_parity(), a_above_is(float(b.a_node), float(b.a_rise)):
        eng = MomwireEngine(b, ground=soil)
    out = {}
    for w, poly in enumerate(eng._polylines):
        p = np.asarray(poly, dtype=float)
        lens = np.linalg.norm(np.diff(p, axis=0), axis=1)
        edges = eng._edge_segments[w]
        if abs(p[0][2]) < 1e-12:
            out["above" if p[-1][2] > 0 else "below"] = float(lens[0] / edges[0])
        elif abs(p[-1][2]) < 1e-12:
            out["above" if p[0][2] > 0 else "below"] = float(lens[-1] / edges[-1])
    return out.get("above"), out.get("below")


def assert_rung(cfg, soil=SOIL_A):
    """Gate the deck before any number is read off it.

    Both engines on one mesh, the feed on a knot, no extended kernel, the
    two-radius crossing fill entered exactly once -- AND the control taking
    that same path, which is what makes the differential's cancellation valid.
    """
    counts = {}
    b = build(**cfg)
    n5 = nec5_segs(b, soil=soil)
    with (
        even_parity(),
        a_above_is(cfg["a_node"], cfg["a_rise"]),
        crossing_calls(counts),
    ):
        eng = MomwireEngine(b, ground=soil)
        sim, coeffs, _z = eng._solved_excited(eng._wavelength_for(b.freq))
    segs = sum(sum(e) for e in eng._edge_segments)
    if segs != n5:
        raise RuntimeError(f"mesh not matched: momwire {segs} vs engine {n5}")
    w, s_f, _v = eng._feeds[0]
    poly = np.asarray(eng._polylines[w], dtype=float)
    knots = mwe._polyline_knots(poly, eng._edge_segments[w])
    arcs = np.concatenate(
        [[0.0], np.cumsum(np.linalg.norm(np.diff(knots, axis=0), axis=1))]
    )
    if float(np.min(np.abs(arcs - s_f))) >= 1e-12:
        raise RuntimeError("feed does not land on a knot")
    if sim.extended_kernel:
        raise RuntimeError("extended_kernel resolved True")
    if counts.get("two_radius") != 1:
        raise RuntimeError(f"crossing fill entered {counts.get('two_radius')} times")
    ctrl_counts = {}
    cb = build(**control_of(cfg))
    with even_parity(), crossing_calls(ctrl_counts):
        MomwireEngine(cb, ground=soil).impedance()
    if ctrl_counts.get("two_radius") != 1:
        raise RuntimeError(
            "the CONTROL does not take the two-radius path "
            f"({ctrl_counts.get('two_radius')} entries): its a_rise must differ "
            "from a_node, or the paths' 0.0152 ohm offset lands in E"
        )
    ctrl_segs = nec5_segs(cb, soil=soil)
    if ctrl_segs != n5:
        raise RuntimeError(f"control mesh {ctrl_segs} != spread mesh {n5}")
    ha, hb = node_h(b)
    return dict(segs=n5, h_above=ha, h_below=hb, h_over_a=ha / float(cfg["a_node"]))


def measure(cfg, soil=SOIL_A, gate=True, **kw):
    """S, D and the differential residual E for both candidate spellings."""
    t0 = time.time()
    gated = assert_rung(cfg, soil=soil) if gate else {}
    a_node, a_rad, a_rise = cfg["a_node"], cfg["a_rad"], cfg["a_rise"]
    b, cb = build(**cfg), build(**control_of(cfg))
    z5 = z_nec5(b, soil=soil)
    z_node = z_momwire(b, a_node, a_rise, soil=soil, **kw)
    z_far = z_momwire(b, a_rad, a_rise, soil=soil, **kw)
    gap = z_momwire(cb, soil=soil, **kw) - z_nec5(cb, soil=soil)
    s = abs(z_node - z_far)
    d = abs(z_node - z5)
    e_node = abs((z_node - z5) - gap)
    e_far = abs((z_far - z5) - gap)
    return dict(
        S=s,
        D=d,
        D_far=abs(z_far - z5),
        SD=s / d if d else float("inf"),
        gap_ctrl=abs(gap),
        E_node=e_node,
        E_far=e_far,
        SE=s / e_node if e_node else float("inf"),
        z_node=repr(z_node),
        z_far=repr(z_far),
        z_engine=repr(z5),
        secs=time.time() - t0,
        **gated,
    )
