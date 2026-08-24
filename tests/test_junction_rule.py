"""The shared "which wire ends are one node" rule (momwire#590 step 1).

`RazorSolver` spelled this rule and `HarringtonSolver` walked it a second time
by hand, with a comment saying two solvers disagreeing about the same deck's
CONNECTIVITY is a worse failure than either answer being wrong. Now there is
one spelling. These gates pin the properties that made the duplicate worth
keeping in the first place — above all the non-transitivity, which reads like
a bug until you know it is deliberate and is exactly what a well-meaning
rewrite to union-find would destroy.
"""

import numpy as np
import pytest

from momwire._junction_rule import JUNCTION_TOL, coincident_groups, grouped
from momwire.bspline import BSplineSolver
from momwire.harrington import HarringtonSolver
from momwire.razor import RazorSolver
from momwire.sinusoidal import SinusoidalSolver


def _pt(x):
    return np.array([float(x), 0.0, 0.0])


# ----------------------------------------------------------------------
# The rule
# ----------------------------------------------------------------------


def test_the_rule_is_not_transitive():
    """A~B and B~C but A!~C gives {A, B} and a separate {C}.

    Placed at 0, 0.6*tol, 1.2*tol: each neighbour pair is inside tolerance and
    the outer pair is not. First-match-against-representatives makes C its own
    group because it is compared against A (the representative), never against
    B. A union-find over the same pairs would merge all three -- that is the
    substantive difference, and it changes the deck's connectivity.
    """
    pts = [_pt(0.0), _pt(0.6 * JUNCTION_TOL), _pt(1.2 * JUNCTION_TOL)]
    assert coincident_groups(pts) == [0, 0, 2]
    assert grouped(["A", "B", "C"], pts) == [["A", "B"], ["C"]]


def test_representatives_are_the_earliest_member():
    """`rep[i] <= i`, and a representative represents itself. Callers rely on
    both: bucketing by `rep` reproduces group-creation order only because a
    representative `r` is first seen at `i == r`."""
    pts = [_pt(0.0), _pt(5.0), _pt(0.0), _pt(5.0), _pt(9.0)]
    rep = coincident_groups(pts)
    assert rep == [0, 1, 0, 1, 4]
    assert all(r <= i for i, r in enumerate(rep))
    assert all(rep[r] == r for r in rep)


def test_grouping_preserves_creation_order():
    labels = ["a", "b", "c", "d"]
    pts = [_pt(2.0), _pt(0.0), _pt(2.0), _pt(0.0)]
    assert grouped(labels, pts) == [["a", "c"], ["b", "d"]]


def test_the_tolerance_boundary_is_inclusive():
    """`<= tol` joins, strictly greater does not -- the boundary a refusal
    window (harrington's `_NEAR_COINCIDENT_TOL`) is measured against."""
    assert coincident_groups([_pt(0.0), _pt(JUNCTION_TOL)]) == [0, 0]
    assert coincident_groups([_pt(0.0), _pt(1.0000001 * JUNCTION_TOL)]) == [0, 1]


def test_a_lone_point_is_its_own_group():
    assert coincident_groups([_pt(0.0)]) == [0]
    assert coincident_groups([]) == []


# ----------------------------------------------------------------------
# The reason it is shared
# ----------------------------------------------------------------------


def _tee_wires():
    """Three arms off one node -- a K=3 junction both solvers must see."""
    return [
        np.array([[0.0, 0.0, 0.0], [0.4, 0.0, 0.0]]),
        np.array([[0.0, 0.0, 0.0], [-0.2, 0.35, 0.0]]),
        np.array([[0.0, 0.0, 0.0], [-0.2, -0.35, 0.0]]),
    ]


def test_razor_and_harrington_agree_on_connectivity():
    """The property the duplicated copy existed to protect, now structural.

    Razor reports junction GROUPS and harrington merges knot NODE IDS, so the
    two are compared through what they mean: how many distinct nodes the three
    coincident ends collapse to. Both must say one.
    """
    wires = _tee_wires()
    razor = RazorSolver(wires=wires, n_per_edge_per_wire=[[6]] * 3, wavelength=1.0)
    groups = razor._find_junctions()
    at_origin = [
        g
        for g in groups
        if all(
            np.linalg.norm(wires[w][0 if e == "start" else -1]) <= JUNCTION_TOL
            for w, e in g["ends"]
        )
    ]
    assert len(at_origin) == 1
    assert len(at_origin[0]["ends"]) == 3

    # Harrington reaches the same verdict through its own merge: the three
    # arms' first segments must all report the SAME left-hand cell, which is
    # only true if their start knots collapsed to one node.
    harr = HarringtonSolver(wires=wires, n_per_edge_per_wire=[[6]] * 3, wavelength=1.0)
    geom = harr._build_geometry()
    left_node, _right, _pieces, _cop = harr._node_map(geom)
    starts = geom["seg_offsets"][:3]
    assert len(set(left_node[s] for s in starts)) == 1

    # And the claim has teeth: pulled apart, both solvers must instead see
    # three separate nodes. The gap has to clear harrington's
    # `_NEAR_COINCIDENT_TOL` = 1e-6 as well as the grouping tolerance -- ends
    # inside that window are REFUSED rather than disconnected, which is the
    # same defect class momwire#590 is about, already guarded here.
    apart = [w.copy() for w in wires]
    for i, w in enumerate(apart[1:], start=1):
        w[0, 2] = i * 0.01
    razor_apart = RazorSolver(
        wires=apart, n_per_edge_per_wire=[[6]] * 3, wavelength=1.0
    )
    assert razor_apart._find_junctions() == []
    harr_apart = HarringtonSolver(
        wires=apart, n_per_edge_per_wire=[[6]] * 3, wavelength=1.0
    )
    geom_apart = harr_apart._build_geometry()
    left_apart, _r, _p, _c = harr_apart._node_map(geom_apart)
    starts_apart = geom_apart["seg_offsets"][:3]
    assert len(set(left_apart[s] for s in starts_apart)) == 3


def test_a_closed_loop_groups_its_own_two_ends():
    """A wire whose start and end coincide is a junction with itself -- the
    case razor's docstring calls out, and the one an ends-of-different-wires
    reading of the rule would miss."""
    loop = np.array([[0.0, 0.0, 0.0], [0.3, 0.0, 0.0], [0.0, 0.0, 0.0]])
    rep = coincident_groups([loop[0], loop[-1]])
    assert rep == [0, 0]


# ----------------------------------------------------------------------
# The tripwire (momwire#590 step 2)
# ----------------------------------------------------------------------


def _joined():
    """Two wires sharing a node — a junction by anyone's reading."""
    return [
        np.array([[0.0, 0.0, 0.0], [0.25, 0.0, 0.0]]),
        np.array([[0.0, 0.0, 0.0], [0.0, 0.25, 0.0]]),
    ]


@pytest.mark.parametrize("solver", [BSplineSolver, SinusoidalSolver])
def test_touching_wires_are_joined_without_being_told(solver):
    """momwire#590 step 3: the default is inference, not refusal.

    Step 2 refused here, to count who relied on the old silent-disconnect
    behaviour. The answer was one test, and that test was itself a latent bug,
    so the default flipped.
    """
    s = solver(wires=_joined(), n_per_edge_per_wire=[[6], [6]], wavelength=1.0)
    assert len(s.junctions) == 1
    assert sorted(s.junctions[0]) == [(0, "start"), (1, "start")]


@pytest.mark.parametrize("solver", [BSplineSolver, SinusoidalSolver])
def test_inferring_the_junction_gives_the_same_answer_as_declaring_it(solver):
    """The claim the flip rests on, in impedance rather than in bookkeeping.

    If these ever diverge, inference is not reproducing what a caller writing
    the junction out by hand would get, and the default is lying.
    """
    kw = dict(
        wires=_joined(),
        n_per_edge_per_wire=[[8], [8]],
        wavelength=1.0,
        wire_radius=0.001,
        feed_wire_index=0,
        feed_arclength=0.125,
    )
    inferred, _ = solver(**kw).compute_impedance()
    declared, _ = solver(
        **kw, junctions=[[(0, "start"), (1, "start")]]
    ).compute_impedance()
    assert complex(inferred) == complex(declared)


@pytest.mark.parametrize("solver", [BSplineSolver, SinusoidalSolver])
def test_an_empty_list_still_means_deliberately_apart(solver):
    """The escape has to keep working, and it has to keep MEANING something:
    disconnected wires must not silently become joined now that the default
    infers. Their impedances differ, which is the whole point."""
    kw = dict(
        wires=_joined(),
        n_per_edge_per_wire=[[8], [8]],
        wavelength=1.0,
        wire_radius=0.001,
        feed_wire_index=0,
        feed_arclength=0.125,
    )
    apart, _ = solver(**kw, junctions=[]).compute_impedance()
    joined, _ = solver(**kw).compute_impedance()
    assert complex(apart) != complex(joined)


@pytest.mark.parametrize("solver", [BSplineSolver, SinusoidalSolver])
def test_an_explicit_empty_list_is_a_statement_not_a_mistake(solver):
    """`junctions=[]` is the escape and must NOT trip. Omitting the argument
    is an oversight; passing an empty list is a caller saying the wires really
    are meant to be disconnected."""
    s = solver(
        wires=_joined(), n_per_edge_per_wire=[[6], [6]], wavelength=1.0, junctions=[]
    )
    assert s.junctions == []


@pytest.mark.parametrize("solver", [BSplineSolver, SinusoidalSolver])
def test_declaring_the_junction_passes(solver):
    s = solver(
        wires=_joined(),
        n_per_edge_per_wire=[[6], [6]],
        wavelength=1.0,
        junctions=[[(0, "start"), (1, "start")]],
    )
    assert len(s.junctions) == 1


@pytest.mark.parametrize("solver", [BSplineSolver, SinusoidalSolver])
def test_wires_that_do_not_touch_are_unaffected(solver):
    """No coincident ends, no refusal — the tripwire must not tax the common
    case of genuinely separate elements (an array, a Yagi)."""
    apart = [
        np.array([[0.0, 0.0, 0.0], [0.25, 0.0, 0.0]]),
        np.array([[0.0, 1.0, 0.0], [0.25, 1.0, 0.0]]),
    ]
    s = solver(wires=apart, n_per_edge_per_wire=[[6], [6]], wavelength=1.0)
    assert s.junctions == []


def test_a_single_wire_never_trips():
    """One polyline with interior vertices is a bend, not a junction — its
    ends are its own two ends and they do not coincide."""
    bent = [np.array([[-0.25, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.25, 0.0]])]
    s = BSplineSolver(wires=bent, n_per_edge_per_wire=[[6, 6]], wavelength=1.0)
    assert s.junctions == []


def test_a_closed_loop_infers_its_own_self_junction():
    """A loop is one wire whose two ends coincide, so inference has to join a
    wire to ITSELF — the one shape a "junctions connect different wires"
    reading would miss, and loops are common antennas.

    Gated in impedance, not just in the junction list: left open, the same
    geometry is a folded-back dipole with a completely different Z.
    """
    side = 0.25
    loop = np.array(
        [[0, 0, 0], [side, 0, 0], [side, side, 0], [0, side, 0], [0, 0, 0]],
        dtype=float,
    )
    kw = dict(
        wires=[loop],
        n_per_edge_per_wire=[[6, 6, 6, 6]],
        wavelength=1.0,
        wire_radius=0.001,
        feed_wire_index=0,
        feed_arclength=0.125,
    )
    s = BSplineSolver(**kw)
    assert s.junctions == [[(0, "start"), (0, "end")]]

    closed, _ = s.compute_impedance()
    declared, _ = BSplineSolver(
        **kw, junctions=[[(0, "start"), (0, "end")]]
    ).compute_impedance()
    assert complex(closed) == complex(declared)

    open_ended, _ = BSplineSolver(**kw, junctions=[]).compute_impedance()
    assert complex(open_ended) != complex(closed)


# ----------------------------------------------------------------------
# The override reaches razor and harrington (momwire#590 step 3b)
# ----------------------------------------------------------------------


def test_razor_declared_junctions_match_what_it_would_have_inferred():
    """Agreeing with the geometry must be a no-op, byte for byte.

    If declaring the junctions a deck already has changed razor's answer, the
    override would be a trap: callers who pass what the front end computed
    would silently get something other than the inferred solve.
    """
    kw = dict(wires=_joined(), n_per_edge_per_wire=[[6], [6]], wavelength=1.0)
    inferred = RazorSolver(**kw)._find_junctions()
    declared = RazorSolver(
        **kw, junctions=[[(1, "start"), (0, "start")]]
    )._find_junctions()
    # Note the declaration is deliberately written in the WRONG order --
    # canonicalisation is what makes it match.
    assert declared == inferred


@pytest.mark.parametrize("solver", [RazorSolver, HarringtonSolver])
def test_the_override_can_disagree_with_the_geometry(solver):
    """The case that was inexpressible before, and the whole reason the old
    refusals were wrong: two coincident ends the caller wants left APART.

    Razor's refusal called a spec "either redundant or a disagreement with the
    mesh". A deliberate disagreement is a legitimate model, and until step 3b
    neither of these solvers could state one.
    """
    kw = dict(
        wires=_joined(),
        n_per_edge_per_wire=[[8], [8]],
        wavelength=1.0,
        wire_radius=0.001,
        feed_wire_index=0,
        feed_arclength=0.125,
    )
    joined, _ = solver(**kw).compute_impedance()
    apart, _ = solver(**kw, junctions=[]).compute_impedance()
    assert complex(joined) != complex(apart)


def test_every_junction_capable_solver_now_reads_the_same_spec():
    """The point of #590, stated once: one geometry, one `junctions=`, and
    four solvers that agree about connectivity whichever way they are told.

    PulseSolver is absent on purpose -- its basis has no junction unknown to
    constrain, so it is the documented exception rather than an oversight.
    """
    kw = dict(wires=_joined(), n_per_edge_per_wire=[[6], [6]], wavelength=1.0)
    spec = [[(0, "start"), (1, "start")]]
    for solver in (BSplineSolver, SinusoidalSolver, RazorSolver, HarringtonSolver):
        # Accepts the spec...
        solver(**kw, junctions=spec)
        # ...and accepts the escape.
        solver(**kw, junctions=[])
        # ...and infers the same thing when told nothing.
        solver(**kw)


# ----------------------------------------------------------------------
# momwire#522's coincidence guard reaches all four spellings
# ----------------------------------------------------------------------

_FOUR = [BSplineSolver, SinusoidalSolver, RazorSolver, HarringtonSolver]


@pytest.mark.parametrize("solver", _FOUR)
def test_declared_ends_that_do_not_coincide_refuse(solver):
    """`_wire_spec.check_junction_coincidence` (momwire#522, the #518
    postmortem) has guarded bspline and sinusoidal since that issue. Razor and
    harrington only began accepting a spec in momwire#590 step 3b, so they were
    the two spellings it did not cover.

    The failure it exists for is not loud: a wrong wire index welds ends that
    sit nowhere near each other, and the result is a well-posed WRONG model
    that converges cleanly. #518 filed one as a solver bias.
    """
    far = [
        np.array([[0.0, 0.0, 0.0], [0.25, 0.0, 0.0]]),
        np.array([[0.0, 5.0, 0.0], [0.25, 5.0, 0.0]]),
    ]
    with pytest.raises(ValueError, match="do not coincide"):
        solver(
            wires=far,
            n_per_edge_per_wire=[[6], [6]],
            wavelength=1.0,
            wire_radius=0.001,
            junctions=[[(0, "start"), (1, "start")]],
        )


@pytest.mark.parametrize("solver", _FOUR)
def test_the_deck_grids_own_fuzz_is_still_declarable(solver):
    """The guard's tolerance contract, now that four solvers share it.

    `deck/_polylines` quantises endpoints onto a 1e-6 m grid, which can leave
    ~1.8e-6 m of Euclidean fuzz between ends it calls one node.
    `_JUNCTION_COINCIDENCE_FLOOR` is 1e-5 precisely so that can never fire.
    A tighter threshold -- 1e-6 say -- would reject the decks the guard exists
    to protect, so this is gated rather than left to the constant's comment.
    """
    fuzzed = [
        np.array([[0.0, 0.0, 0.0], [0.25, 0.0, 0.0]]),
        np.array([[0.0, 1.8e-6, 0.0], [0.0, 0.25, 0.0]]),
    ]
    s = solver(
        wires=fuzzed,
        n_per_edge_per_wire=[[6], [6]],
        wavelength=1.0,
        wire_radius=0.001,
        junctions=[[(0, "start"), (1, "start")]],
    )
    assert s is not None


@pytest.mark.parametrize("solver", _FOUR)
def test_inference_is_never_put_through_the_guard(solver):
    """What inference finds is coincident to 1e-9 by construction, so a
    geometry that solves with no `junctions=` must keep solving."""
    s = solver(
        wires=_joined(),
        n_per_edge_per_wire=[[6], [6]],
        wavelength=1.0,
        wire_radius=0.001,
    )
    assert s is not None
