"""Every refusal that names another route must name one that works (#604).

This repo states its limits in prose, and a refusal that recommends an
alternative is making a claim about a code path it does not execute:

    "... For the zero-width source use SinusoidalGalerkinSolver(
     feed_model='point'), whose test integral is what makes a delta source
     admissible"

Nothing runs that sentence. When the recommended route grows a constraint,
gets renamed, or is itself refused, the refusal keeps confidently sending
callers at it — and the reader believes it, because a refusal that explains
itself reads as authoritative. momwire#430 was one of these
(`pulse.py`'s Sommerfeld refusal pinned a statement #398 unit 5 made false),
and #586's gates checked exactly this for one case. momwire#604 asks for the
pattern to be generalised; this file is that.

Two halves, and the second is what keeps it honest:

* every recommended route below is EXECUTED, so a broken recommendation is a
  test failure rather than a lie a user discovers;
* the source tree is SCANNED for raised refusals that name another route — a
  solver class, a `--basis`, or a constructor argument the caller is told to
  pass instead — and the roster must cover exactly what the scan finds. Adding
  a new route-naming refusal without a route check fails here — the one-way
  link stated as a failure rather than as a comment, the same shape
  `_portal._BANNER_SUFFIXES` uses to make a missing suffix a KeyError at
  import.

The scan is deliberately narrow — RAISED messages only, not every docstring
that mentions a sibling. Prose claims in docstrings are #604's classes (1)
and (3), and those need reading, not execution.
"""

import ast
import pathlib
import re

import numpy as np
import pytest

from momwire import (
    BSplineSolver,
    HarringtonSolver,
    RazorSolver,
    SinusoidalGalerkinSolver,
)

WL = 20.0
SRC = pathlib.Path(__file__).resolve().parent.parent / "src" / "momwire"


def _tee():
    """Three wires meeting at the origin — a junction to address."""
    return [
        np.array([[0.0, 0.0, 0.0], [3.0, 0.0, 0.0]]),
        np.array([[0.0, 0.0, 0.0], [-3.0, 0.0, 0.0]]),
        np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 3.0]]),
    ]


def _dipole(z0=-2.0, z1=2.0):
    return np.array([[0.0, 0.0, z0], [0.0, 0.0, z1]])


# ---------------------------------------------------------------------------
# The roster: (source file, the refusal's subject, the route it names, run it)
# ---------------------------------------------------------------------------
def _route_bspline_junction_ports():
    BSplineSolver(
        wires=_tee(),
        nsegs=9,
        wavelength=WL,
        wire_radius=1e-3,
        junction_ports=[0],
        feeds=[],
    ).compute_impedance()


def _route_bspline_node_gaps():
    BSplineSolver(
        wires=_tee(),
        nsegs=9,
        wavelength=WL,
        wire_radius=1e-3,
        node_gaps=[(0, "start", 1.0)],
        feeds=[],
    ).compute_impedance()


def _route_galerkin_node_gaps():
    SinusoidalGalerkinSolver(
        wires=_tee(),
        nsegs=9,
        wavelength=WL,
        wire_radius=1e-3,
        node_gaps=[(0, "start", 1.0)],
        feeds=[],
    ).compute_impedance()


def _route_galerkin_point_feed():
    SinusoidalGalerkinSolver(
        wires=[_dipole()],
        nsegs=12,
        wavelength=WL,
        wire_radius=1e-3,
        feed_model="point",
    ).compute_impedance()


def _route_bspline_ground_contact():
    BSplineSolver(
        wires=[_dipole(0.0, 4.0)],
        nsegs=12,
        wavelength=WL,
        wire_radius=1e-3,
        ground_z=0.0,
    ).compute_impedance()


def _route_razor_split_at_touchdown():
    """Razor serves ground contact at a wire END, so a mid-span touchdown is
    served by splitting the wire there — which is what the refusal says."""
    RazorSolver(
        wires=[
            np.array([[-3.0, 0.0, 2.0], [0.0, 0.0, 0.0]]),
            np.array([[0.0, 0.0, 0.0], [3.0, 0.0, 2.0]]),
        ],
        nsegs=9,
        wavelength=WL,
        wire_radius=1e-3,
        ground_z=0.0,
    ).compute_impedance()


def _route_galerkin_ek_with_near_correction():
    SinusoidalGalerkinSolver(
        wires=[_dipole()],
        nsegs=12,
        wavelength=WL,
        wire_radius=1e-3,
        extended_kernel=True,
        near_correction=True,
    ).compute_impedance()


def _route_harrington_junctions_kwarg():
    """Two ends inside the near-coincident window, joined by `junctions=`."""
    gap = 1e-6
    HarringtonSolver(
        wires=[
            np.array([[-3.0, 0.0, 0.0], [0.0, 0.0, 0.0]]),
            np.array([[gap, 0.0, 0.0], [3.0, 0.0, 0.0]]),
        ],
        nsegs=9,
        wavelength=WL,
        wire_radius=1e-3,
        junctions=[[(0, "end"), (1, "start")]],
    ).compute_impedance()


ROSTER = {
    "sinusoidal.py::junction_ports": _route_bspline_junction_ports,
    "sinusoidal.py::node_gaps->bspline": _route_bspline_node_gaps,
    "sinusoidal.py::node_gaps->galerkin": _route_galerkin_node_gaps,
    "sinusoidal.py::feed_model=point": _route_galerkin_point_feed,
    "pulse.py::ground_contact": _route_bspline_ground_contact,
    "razor.py::midspan_touchdown": _route_razor_split_at_touchdown,
    "sinusoidal_galerkin.py::ek_needs_near": _route_galerkin_ek_with_near_correction,
    "harrington.py::near_coincident": _route_harrington_junctions_kwarg,
}


@pytest.mark.parametrize("name", sorted(ROSTER))
def test_the_route_a_refusal_names_actually_works(name):
    """Execute the alternative. A refusal that sends a caller somewhere is
    only as good as the somewhere."""
    ROSTER[name]()


# ---------------------------------------------------------------------------
# The one-way link: the roster must cover what the tree actually raises
# ---------------------------------------------------------------------------
# A route is another solver class, another `--basis`, or another CONSTRUCTOR
# ARGUMENT the caller is told to pass instead. The third matters as much as
# the first: "pass `junctions=` naming them as one node" is the same claim
# about the same kind of untested path.
_NAMES_A_ROUTE = re.compile(
    r"\b[A-Z]\w*Solver\b|--basis\s+[a-z0-9-]+|(?:pass|use)\s+`\w+=`"
)
# Messages that merely name the class they belong to ("RazorSolver got
# unexpected keyword argument(s)") are not recommending anything.
_SELF_NAMING = re.compile(r"got unexpected keyword argument")


def _raised_route_naming_refusals():
    """(file, lineno) of every raise whose message names another route."""
    found = []
    for f in sorted(SRC.rglob("*.py")):
        try:
            tree = ast.parse(f.read_text())
        except SyntaxError:  # pragma: no cover - a broken tree is its own bug
            continue
        for node in ast.walk(tree):
            if not isinstance(node, ast.Raise) or node.exc is None:
                continue
            parts = [
                n.value
                for n in ast.walk(node.exc)
                if isinstance(n, ast.Constant) and isinstance(n.value, str)
            ]
            msg = " ".join(" ".join(parts).split())
            if len(msg) < 20 or _SELF_NAMING.search(msg):
                continue
            if _NAMES_A_ROUTE.search(msg):
                found.append((f.name, node.lineno, msg))
    return found


def test_every_route_naming_refusal_has_a_route_check():
    """A new refusal that recommends a route must bring a check with it.

    This is the half that keeps the roster from rotting: without it the file
    pins whatever was true the day it was written and silently stops covering
    the tree. Counted by FILE rather than by line, so moving a refusal within
    its module does not fail this — what has to stay true is that every module
    which sends callers elsewhere is represented.
    """
    found = _raised_route_naming_refusals()
    assert found, "the scanner found nothing — it has stopped working"

    covered = {k.split("::", 1)[0] for k in ROSTER}
    seen = {f for f, _ln, _msg in found}
    missing = seen - covered
    assert not missing, (
        "these modules raise a refusal naming another route, but no entry in "
        f"ROSTER executes it: {sorted(missing)}. Add one — a refusal that "
        "recommends a route it cannot demonstrate is momwire#604's class (2)."
    )
    stale = covered - seen
    assert not stale, (
        f"ROSTER covers {sorted(stale)}, which no longer raises a "
        "route-naming refusal. Drop the entry, or the file is pinning prose "
        "that has gone."
    )


# ---------------------------------------------------------------------------
# Class (3): universals over the solver roster
# ---------------------------------------------------------------------------
def test_the_port_solution_roster_is_what_the_docstring_says():
    """`_port_solution`'s module docstring used to open "Every solver family
    already builds one right-hand-side column per port ... `PortSolution` is
    what `compute_port_solution()` returns instead."

    Two of the eight exported families did not have the method at all, which
    was momwire#604's class (3) — a universal over a roster that had grown an
    exception, the same shape `PortPlan`'s "Indices into `sites` ARE solver
    port indices" had. It was load-bearing rather than cosmetic: it was
    exactly why the pulse family had no portal surface (momwire#564), so the
    sentence denied the existence of the gap that issue was open about.

    momwire#564 closed it by implementing the method rather than by editing
    the sentence, so `does_not` is EMPTY now and the universal is true. The
    empty set is kept as its own statement rather than deleted with its last
    tenant — the same shape as `test_eznec_basis_choice.RAISED` — because
    "no exported family lacks a portal surface" is the claim, and a family
    arriving without one has to fail here rather than be absorbed.

    Pinned as a roster rather than as prose, so the next family to gain or
    lose the method has to come here and say so.
    """
    import momwire

    serves = {
        "ArrayBlockSolver",
        "BSplineSolver",
        "HarringtonSolver",
        "HMatrixSolver",
        "PulseSolver",
        "RazorSolver",
        "SinusoidalGalerkinSolver",
        "SinusoidalSolver",
    }
    does_not = set()

    exported = {n for n in dir(momwire) if n.endswith("Solver")}
    assert serves | does_not == exported, (
        "the exported solver roster changed; update this gate and "
        "`_port_solution`'s module docstring together"
    )
    for name in sorted(serves):
        assert hasattr(getattr(momwire, name), "compute_port_solution"), name
    for name in sorted(does_not):
        cls = getattr(momwire, name)
        assert not hasattr(cls, "compute_port_solution"), (
            f"{name} grew compute_port_solution — good, but "
            "`_port_solution`'s docstring names it as a family that lacks one"
        )
