"""Segments shorter than two radii warn — momwire#1378 step 1.

Below delta/a = 2 no thin-wire kernel here is inside its validity, and the
answer stops converging as the mesh is refined. `ShortSegments` says so once
per solve, on the mesh the solver will actually solve, and changes nothing
else (Z is bit-identical with or without it). Every check is
construction-only: the advisory is raised before any fill.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest

from momwire._wire_spec import (
    SHORT_SEGMENT_FLOOR,
    ShortSegments,
    short_segment_message,
    short_segment_wires,
)
from momwire.array_block import ArrayBlockSolver
from momwire.bspline import BSplineSolver
from momwire.deck import build_solver, parse
from momwire.harrington import HarringtonSolver
from momwire.hmatrix import HMatrixSolver
from momwire.pulse import PulseSolver
from momwire.razor import RazorSolver
from momwire.sinusoidal import SinusoidalSolver
from momwire.sinusoidal_galerkin import SinusoidalGalerkinSolver

WL = 299792458.0 / 600e6
A = 3.175e-3

# Every constructor family. HMatrix and ArrayBlock subclass BSpline, Harrington
# subclasses Pulse, Sin-Galerkin subclasses Sinusoidal: each is listed because
# the advisory lives in the constructors, and a subclass that skipped its
# parent's tail would read as silent here.
SOLVERS = [
    ("bspline", BSplineSolver, {}),
    ("hmatrix", HMatrixSolver, {}),
    ("arrayblock", ArrayBlockSolver, {}),
    ("razor-2p", RazorSolver, {"nec5_quadrature": True}),
    ("razor-gl", RazorSolver, {"nec5_quadrature": False}),
    ("sinusoidal", SinusoidalSolver, {}),
    ("sin-galerkin", SinusoidalGalerkinSolver, {}),
    ("pulse", PulseSolver, {}),
    ("harrington", HarringtonSolver, {}),
]
ROUTES = pytest.mark.parametrize(
    "cls, kw", [(c, k) for _i, c, k in SOLVERS], ids=[i for i, _c, _k in SOLVERS]
)


def _dipole(n, ratio, x=0.0):
    """A z-directed dipole of `n` equal segments, each `ratio` radii long."""
    half = n * ratio * A / 2
    return np.array([(x, 0.0, -half), (x, 0.0, half)])


def _build(cls, kw, wires, npe, **extra):
    """Construct (not solve) and return the ShortSegments messages raised."""
    feed = (0, float(np.linalg.norm(wires[0][1] - wires[0][0])) / 2, 1 + 0j)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        cls(
            wavelength=WL,
            wire_radius=A,
            wires=wires,
            n_per_edge_per_wire=[[n] if np.isscalar(n) else n for n in npe],
            feeds=[feed],
            **kw,
            **extra,
        )
    return [str(w.message) for w in caught if issubclass(w.category, ShortSegments)]


@ROUTES
def test_just_under_two_radii_warns_naming_the_wire_and_the_ratio(cls, kw):
    (msg,) = _build(cls, kw, [_dipole(10, 1.99)], [10])
    assert "wire 0 has segments 1.99 radii long (shortest)" in msg
    assert "below 2 radii" in msg and "momwire#1378" in msg
    assert msg.startswith(cls.__name__ + ":")


@ROUTES
@pytest.mark.parametrize("ratio", [2.0, 2.01, 5.0])
def test_exactly_two_radii_and_above_is_silent(cls, kw, ratio):
    assert _build(cls, kw, [_dipole(10, ratio)], [10]) == []


def test_the_floor_is_two_and_is_strict():
    assert SHORT_SEGMENT_FLOOR == 2.0
    pl = _dipole(4, 2.0)
    assert short_segment_wires([pl], [[4]], [A]) == []
    assert short_segment_wires([pl], [[4]], [A * 1.01]) != []


def _four_wires(ratios):
    """Parallel dipoles 0.1 m apart, each its own `ratio`; 10 segments."""
    wires = [_dipole(10, r, x=0.1 * i) for i, r in enumerate(ratios)]
    return wires, [10] * len(ratios)


@ROUTES
def test_one_warning_per_solve_with_the_short_wires_named_together(cls, kw):
    # wires 1, 3 and 4 are short; 0 and 2 are fine. The worst (0.5) is wire 3.
    wires, npe = _four_wires([3.0, 1.5, 2.5, 0.5, 1.0])
    msgs = _build(cls, kw, wires, npe)
    assert len(msgs) == 1
    (msg,) = msgs
    assert "wires 3 (0.5 radii), 4 (1 radii), 1 (1.5 radii)" in msg
    assert "worst 0.5 on wire 3" in msg
    assert "wire 0" not in msg and " 2 (" not in msg


def test_a_long_list_is_cut_to_the_first_few_then_and_n_more():
    ratios = [0.1 * (i + 1) for i in range(9)]
    wires, npe = _four_wires(ratios)
    (msg,) = _build(BSplineSolver, {}, wires, npe)
    assert "wires 0 (0.1 radii), 1 (0.2 radii), 2 (0.3 radii), 3 (0.4 radii)" in msg
    assert "and 5 more" in msg
    assert "wire 8" not in msg


@ROUTES
def test_only_the_short_wire_is_named(cls, kw):
    wires, npe = _four_wires([3.0, 1.2])
    (msg,) = _build(cls, kw, wires, npe)
    assert "wire 1 has segments 1.2 radii long" in msg
    assert "wire 0" not in msg


def test_the_worst_edge_of_a_bent_wire_speaks_for_it():
    """Two edges at different densities: the wire's ratio is its shortest
    segment, not its mean."""
    pl = np.array([(0, 0, 0), (0, 0, 6 * A), (0, 0, 6 * A + 8 * A)], dtype=float)
    # edge 1: 6 A over 6 segments = 1 radius each; edge 2: 8 A over 2 = 4 radii
    (msg,) = _build(BSplineSolver, {}, [pl], [[6, 2]])
    assert "wire 0 has segments 1 radii long" in msg


# --- the check reads the FINAL mesh ----------------------------------------


@ROUTES
def test_the_default_segment_count_is_resolved_before_the_check(cls, kw):
    """No `n_per_edge_per_wire` at all: the count comes from `nsegs`, a
    resolution the constructor does, so an input-side check would see
    nothing. 100 segments on a 100-radius wire is 1 radius each."""
    pl = np.array([(0, 0, 0), (0, 0, 100 * A)], dtype=float)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        try:
            cls(
                wavelength=WL,
                wire_radius=A,
                wires=[pl],
                nsegs=100,
                feeds=[(0, 50 * A, 1 + 0j)],
                **kw,
            )
        except TypeError:
            pytest.skip(f"{cls.__name__} takes no nsegs")
    msgs = [w for w in caught if issubclass(w.category, ShortSegments)]
    assert len(msgs) == 1
    assert "1 radii long" in str(msgs[0].message)


_PLANE_SPLIT = """CM a one-element wire straddling the ground plane
CE
GW 1 1 0 0 -0.003 0 0 0.003 2.E-3
GW 2 10 0 0 0.003 0 0 0.063 2.E-3
GE 1
GN 2 0 0 0 13. 0.005
EX 0 2 5 0 1. 0.
FR 0 1 0 0 300.
XQ
NX
"""


@pytest.mark.parametrize("basis", ["bspline", "hmatrix", "arrayblock", "sinusoidal"])
def test_a_plane_split_that_shortens_the_mesh_is_seen(basis):
    """Negative control for 'the check runs on the final mesh'. The deck's
    own segments are all 3 radii (6 mm at a = 2 mm) and the input check would
    be silent; the front end splits wire 1 where it meets the ground plane
    (momwire#667), and each half is 1.5 radii. The only place that can see
    it is the solver's constructor."""
    model = parse(_PLANE_SPLIT)
    for wire in model.wires:  # the input is fine: every segment is 3 radii
        edge = np.asarray(wire.vertices)
        for (p, q), n in zip(zip(edge[:-1], edge[1:]), wire.edge_elements):
            assert np.linalg.norm(q - p) / n / 2e-3 >= SHORT_SEGMENT_FLOOR
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        build_solver(model, basis=basis, frequency_mhz=300.0)
    msgs = [str(w.message) for w in caught if issubclass(w.category, ShortSegments)]
    assert len(msgs) == 1
    assert "1.5 radii" in msgs[0]


def test_the_same_deck_without_the_plane_is_silent():
    free = _PLANE_SPLIT.replace("GE 1\nGN 2 0 0 0 13. 0.005\n", "GE 0\n")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        build_solver(parse(free), basis="bspline", frequency_mhz=300.0)
    assert [w for w in caught if issubclass(w.category, ShortSegments)] == []


# --- the escape hatch -------------------------------------------------------


def test_the_class_is_a_userwarning_and_filterable_by_category():
    assert issubclass(ShortSegments, UserWarning)
    wires, _npe = _four_wires([1.0])
    ctor = dict(
        wavelength=WL,
        wire_radius=A,
        wires=wires,
        n_per_edge_per_wire=[[10]],
        feeds=[(0, 5 * A, 1 + 0j)],
    )
    with warnings.catch_warnings():
        warnings.simplefilter("error", ShortSegments)
        with pytest.raises(ShortSegments, match="momwire#1378"):
            BSplineSolver(**ctor)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        warnings.simplefilter("ignore", ShortSegments)
        BSplineSolver(**ctor)
    assert [w for w in caught if issubclass(w.category, ShortSegments)] == []


def test_the_message_reads_as_one_sentence_for_one_wire():
    msg = short_segment_message("BSplineSolver", [(3, 0.62, 12)])
    assert msg == (
        "BSplineSolver: wire 3 has segments 0.62 radii long (shortest); below 2 "
        "radii the thin-wire kernels are outside their validity and results "
        "stop converging (momwire#1378)."
    )
