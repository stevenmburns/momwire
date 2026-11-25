"""momwire#1338: the sector route reachable from the drop-ins through one
opt-in variable, MOMWIRE_X_ROTATIONAL_SYMMETRY=1.

What is held here:

- the registry entry: python-kind, reach 4 (`dropin-linux-windows`), off by
  default and not offered, honoured by bs2 / bs1 and the point-matched
  sinusoidal solver only;
- the KEYWORD keeps its contract (True refuses a deck the route does not
  serve, False is the dense solve whatever the variable says); only an
  unset keyword reads the variable;
- with the variable on, each drop-in (NEC-5, NEC-4.2, the nec2c portal) runs
  the sector route on a symmetric radial screen -- counted, so a green test
  cannot pass with the route silently off -- and answers within the entry's
  `drift` of the dense solve; line 2 of the NEC-5 and NEC-4.2 printouts
  names it;
- unset or 0, nothing moves: no sector call and the same printout;
- a deck or a solver the route does not serve is DECLINED, not refused:
  solved densely to the bit, the reason on `rotational_symmetry_declined`,
  in the fill report's `sector.declined`, and warned as
  RotationalSymmetryDeclined;
- razor2p, which has no sector route (momwire#1251), runs as with the
  variable off and line 2 does not name it.
"""

from __future__ import annotations

import io
import math
import re
import warnings

import numpy as np
import pytest

from momwire import _features as F
from momwire import _rotational_symmetry as rs
from momwire.bspline import BSplineSolver
from momwire.deck import build_solver, parse
from momwire.dropin.nec2c import _portal
from momwire.dropin.nec5 import _shell
from momwire.hmatrix import HMatrixSolver
from momwire.sinusoidal import SinusoidalSolver
from momwire.sinusoidal_galerkin import SinusoidalGalerkinSolver

ENV = "MOMWIRE_X_ROTATIONAL_SYMMETRY"
FEATURE = F.FEATURES["rotational_symmetry"]
# Wall-clock lines: the only ones two runs of one deck may print differently.
_CLOCK = re.compile(r"msec|SEC\.|TIME|Time", re.I)


@pytest.fixture
def no_features(monkeypatch):
    """Every feature variable (and alias) unset."""
    for f in F.FEATURES.values():
        for var in (f.env, *f.aliases):
            monkeypatch.delenv(var, raising=False)
    return monkeypatch


def screen(dialect, *, n=4, ground="free", h0=3.0, long_radial=False):
    """A vertical on the z axis over `n` radials from its base, fed on its
    second segment (on the axis), written as `dialect` writes a deck:
    ``"nec5"`` / ``"nec4"`` as EZNEC does, ``"nec2"`` for the portal.
    `long_radial` makes radial 0 a metre longer, so the deck is not N
    copies of one sector."""
    eznec = {"nec5": "NEC-5", "nec4": "NEC-4.2"}.get(dialect)
    sep = "," if eznec else " "
    lines = ["CM radial screen"]
    if eznec:
        lines.append(f"CM ! Written by EZNEC/Pro+ v. 7.0 in {eznec} format.")
    lines.append("CE")

    def gw(tag, segs, a, b):
        fields = [tag, segs, *(f"{v:.10g}" for v in (*a, *b)), "0.001"]
        return "GW " + sep.join(str(f) for f in fields)

    lines.append(gw(1, 11, (0.0, 0.0, h0), (0.0, 0.0, h0 + 10.0)))
    for i in range(n):
        phi = 2 * math.pi * i / n
        r = 10.0 + (1.0 if long_radial and i == 0 else 0.0)
        lines.append(
            gw(2 + i, 7, (0.0, 0.0, h0), (r * math.cos(phi), r * math.sin(phi), h0))
        )
    on_ground = 0 if ground == "free" else 1
    gn = {"free": "GN -1", "pec": "GN 1", "sommerfeld": "GN 2,0,0,0,13.,.005"}
    if eznec:
        ex = 4 if dialect == "nec5" else 0
        lines += [f"GE {on_ground},-1", "FR 0,1,0,0,7.", gn[ground]]
        lines += [f"EX {ex},1,2,0,1.414214,0.", "PQ 0"]
        lines += ["RP 0,19,1,1000,0.,0.,5.,0.,0.", "EN"]
    else:
        lines.append(f"GE {on_ground}")
        if ground != "free":
            lines.append(gn[ground].replace(",", " "))
        lines += ["EX 0 1 2 0 1. 0.", "FR 0 1 0 0 7. 0.", "XQ", "EN"]
    return "\n".join(lines) + "\n"


class Probe:
    """What a drop-in run did: every port solution's admittances, whether
    each solving instance held a sector map, and the sector solves called."""

    def __init__(self, monkeypatch):
        self.ys: list[np.ndarray] = []
        self.mapped: list[bool] = []
        self.sector_solves = 0
        for name in ("solve", "solve_ports"):
            original = getattr(rs, name)

            def counted(*a, _f=original, **k):
                self.sector_solves += 1
                return _f(*a, **k)

            monkeypatch.setattr(rs, name, counted)
        for cls in (BSplineSolver, SinusoidalSolver):
            original = cls.compute_port_solution

            def recorded(solver, *a, _f=original, **k):
                out = _f(solver, *a, **k)
                self.ys.append(np.array(out.y))
                self.mapped.append(solver._rotational_map is not None)
                return out

            monkeypatch.setattr(cls, "compute_port_solution", recorded)


def run(monkeypatch, route, basis, deck, flag):
    """`(printout, probe, declined warnings)` for `deck` through drop-in
    `route` (``"nec5"``, ``"nec4"`` or ``"nec2c"``) on `basis`, with the
    variable at `flag` (None: unset)."""
    if flag is None:
        monkeypatch.delenv(ENV, raising=False)
    else:
        monkeypatch.setenv(ENV, flag)
    with monkeypatch.context() as m:
        probe = Probe(m)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            if route == "nec2c":
                out = io.StringIO()
                with _portal.engine_scope():
                    rc = _portal.main(
                        [],
                        stdin=io.StringIO(deck),
                        stdout=out,
                        stderr=io.StringIO(),
                        prog=f"momwire-nec2c-{basis}",
                    )
                assert rc == 0
                text = out.getvalue()
            else:
                text = _shell.render(deck, basis=basis, dialect=route)
    declined = [
        str(w.message)
        for w in caught
        if issubclass(w.category, rs.RotationalSymmetryDeclined)
    ]
    assert "ERROR" not in text, text[-600:]
    return text, probe, declined


def unclocked(text):
    return [s for s in text.splitlines() if not _CLOCK.search(s)]


def rel(a, b):
    return max(float(np.max(np.abs(x - y)) / np.max(np.abs(y))) for x, y in zip(a, b))


# --- the registry ---------------------------------------------------------------


def test_the_registry_entry():
    assert FEATURE.env == ENV
    assert FEATURE.kwarg == "rotational_symmetry"
    assert FEATURE.kind == "python"
    assert FEATURE.reach == F._level("dropin-linux-windows") == 4
    assert FEATURE.solvers == ("bspline", "sinusoidal")
    assert not FEATURE.default and not FEATURE.offer
    assert FEATURE.drift <= F.DEFAULT_DRIFT_MAX
    for platform in ("linux", "win32"):
        assert F.certified("rotational_symmetry", "dropin", platform=platform)
    # Python-kind: the macOS drop-in runs it, stamped uncertified.
    assert not F.certified("rotational_symmetry", "dropin", platform="darwin")
    # No executable level: there is no C++ sector route.
    assert FEATURE.reach < F._level("exe-nec5-bs2-razor2p-windows")


def test_razor2p_does_not_honour_it():
    # momwire#1251: no sector route on razor2p yet, so the variable is not
    # read there and line 2 does not name it.
    assert not F.honours("rotational_symmetry", "razor")
    assert not F.honours("rotational_symmetry", "sinusoidal_galerkin")


# --- the keyword's contract and the variable's ----------------------------------


def _bspline_screen(**kw):
    model = parse(screen("nec2"), dialect="nec2")
    return build_solver(model, basis="bs2", **kw).solver


def test_off_by_default(no_features):
    solver = _bspline_screen()
    assert solver._rotational_map is None
    assert solver.rotational_symmetry is False
    assert solver.rotational_symmetry_declined is None
    assert solver.fill_report()["sector"] is None


def test_the_variable_turns_it_on(no_features):
    no_features.setenv(ENV, "1")
    solver = _bspline_screen()
    assert solver._rotational_map is not None
    assert solver._rotational_map.n_sectors == 4
    assert solver.rotational_symmetry is True


def test_zero_is_off(no_features):
    no_features.setenv(ENV, "0")
    assert _bspline_screen()._rotational_map is None


def test_the_keyword_false_wins_over_the_variable(no_features):
    no_features.setenv(ENV, "1")
    model = parse(screen("nec2"), dialect="nec2")
    solver = build_solver(model, basis="bs2").solver
    assert solver._rotational_map is not None
    wires = [np.array([(0.0, 0.0, 3.0), (0.0, 0.0, 13.0)])]
    off = BSplineSolver(
        wires=wires, n_per_edge_per_wire=[[11]], rotational_symmetry=False
    )
    assert off._rotational_map is None and off.rotational_symmetry_declined is None


def _arm(**kw):
    """One vertical and a single radial: not a screen."""
    wires = [
        np.array([(0.0, 0.0, 3.0), (0.0, 0.0, 13.0)]),
        np.array([(0.0, 0.0, 3.0), (10.0, 0.0, 3.0)]),
    ]
    return dict(
        wires=wires,
        n_per_edge_per_wire=[[11], [7]],
        wire_radius=1e-3,
        junctions=[((0, "start"), (1, "start"))],
        **kw,
    )


def test_the_keyword_true_still_refuses(no_features):
    no_features.setenv(ENV, "1")
    with pytest.raises(rs.RotationalSymmetryRefused, match="N >= 2 sectors"):
        BSplineSolver(**_arm(rotational_symmetry=True))


def test_the_variable_declines_where_the_keyword_refuses(no_features):
    no_features.setenv(ENV, "1")
    with pytest.warns(rs.RotationalSymmetryDeclined, match="N >= 2 sectors"):
        solver = BSplineSolver(**_arm())
    assert solver._rotational_map is None
    assert solver.rotational_symmetry is False
    reason = solver.rotational_symmetry_declined
    assert reason.startswith("rotational symmetry: ") and "N >= 2" in reason
    # The refusal's way out is not repeated: the dense solve was taken.
    assert rs._WAY_OUT not in reason
    solver.compute_impedance()
    sector = solver.fill_report()["sector"]
    assert sector["declined"] == reason
    assert sector["n_sectors"] is None and sector["blocks"] == []


def _screen_kwargs(**kw):
    """A vertical over four radials, as solver keywords."""
    wires = [
        np.array([(0.0, 0.0, 3.0), (0.0, 0.0, 13.0)]),
        *(
            np.array([(0.0, 0.0, 3.0), (10 * math.cos(p), 10 * math.sin(p), 3.0)])
            for p in (0.0, math.pi / 2, math.pi, 3 * math.pi / 2)
        ),
    ]
    return dict(
        wires=wires,
        n_per_edge_per_wire=[[11]] + [[7]] * 4,
        wire_radius=1e-3,
        junctions=[tuple((w, "start") for w in range(5))],
        **kw,
    )


def test_a_solver_without_a_sector_solve_declines(no_features):
    # HMatrixSolver runs BSplineSolver's constructor, but its own
    # solve_strategy axis has no "sector": declined by name, never mapped.
    no_features.setenv(ENV, "1")
    assert BSplineSolver(**_screen_kwargs())._rotational_map is not None
    with pytest.warns(rs.RotationalSymmetryDeclined, match="HMatrixSolver"):
        solver = HMatrixSolver(**_screen_kwargs())
    assert solver._rotational_map is None
    assert "no sector solve" in solver.rotational_symmetry_declined


@pytest.mark.filterwarnings("ignore:exact_kernel=True with a point-gap feed")
def test_exact_kernel_true_declines_under_the_variable(no_features):
    # The keyword refuses exact_kernel=True with the route; the variable,
    # which only asks, declines the route instead and keeps the kernel.
    no_features.setenv(ENV, "1")
    with pytest.warns(rs.RotationalSymmetryDeclined, match="exact_kernel=True"):
        solver = BSplineSolver(**_screen_kwargs(exact_kernel=True))
    assert solver._rotational_map is None and solver.exact_kernel is True


def test_the_galerkin_subclass_ignores_the_variable(no_features):
    no_features.setenv(ENV, "1")
    model = parse(screen("nec2"), dialect="nec2")
    with warnings.catch_warnings():
        warnings.simplefilter("error", rs.RotationalSymmetryDeclined)
        solver = build_solver(model, basis="sg").solver
    assert isinstance(solver, SinusoidalGalerkinSolver)
    assert solver._rotational_map is None
    assert solver.rotational_symmetry_declined is None


# --- the drop-ins ---------------------------------------------------------------

# Each drop-in and basis the route serves there, on a symmetric screen: the
# grounds cover free space, a perfect image and a Sommerfeld half-space.
SERVED = [
    ("nec5", "bs2", "free"),
    ("nec5", "bs2", "pec"),
    ("nec5", "bs1", "free"),
    ("nec4", "bs2", "free"),
    ("nec4", "bs2", "sommerfeld"),
    ("nec2c", "bs2", "pec"),
    ("nec2c", "sinusoidal", "free"),
    ("nec2c", "sinusoidal", "sommerfeld"),
]


@pytest.mark.parametrize(("route", "basis", "ground"), SERVED)
def test_each_dropin_runs_the_sector_route(no_features, route, basis, ground):
    dialect = "nec2" if route == "nec2c" else route
    deck = screen(dialect, ground=ground)
    off_text, off, _ = run(no_features, route, basis, deck, None)
    on_text, on, declined = run(no_features, route, basis, deck, "1")
    # It RAN: every solving instance held the map, and the route's own
    # harmonic solve was called; with it off, never.
    assert on.sector_solves >= 1 and on.mapped and all(on.mapped)
    assert off.sector_solves == 0 and not any(off.mapped)
    assert not declined
    assert len(on.ys) == len(off.ys) >= 1
    assert rel(on.ys, off.ys) <= FEATURE.drift
    lines_off, lines_on = unclocked(off_text), unclocked(on_text)
    assert len(lines_on) == len(lines_off)
    if route == "nec2c":
        # The portal keeps nec2c's layout, which has no line for it.
        assert "rotational_symmetry" not in on_text
    else:
        # The stamp is certified only within the feature's reach: the macOS
        # drop-in (level 5, past reach 4) runs it stamped uncertified.
        level = F.current_level("dropin")
        tail = "" if level <= FEATURE.reach else " uncertified on this platform"
        assert lines_on[1] == lines_off[1] + " +rotational_symmetry" + tail
        assert "+rotational_symmetry" not in off_text


@pytest.mark.parametrize(("route", "basis"), [("nec5", "bs2"), ("nec2c", "sinusoidal")])
def test_zero_and_unset_print_the_same(no_features, route, basis):
    deck = screen("nec2" if route == "nec2c" else route)
    unset_text, unset, _ = run(no_features, route, basis, deck, None)
    zero_text, zero, _ = run(no_features, route, basis, deck, "0")
    assert unclocked(zero_text) == unclocked(unset_text)
    assert zero.sector_solves == unset.sector_solves == 0
    assert all(np.array_equal(a, b) for a, b in zip(zero.ys, unset.ys))


@pytest.mark.parametrize(("route", "basis"), [("nec5", "bs2"), ("nec2c", "sinusoidal")])
def test_a_deck_that_is_not_symmetric_runs_densely(no_features, route, basis):
    deck = screen("nec2" if route == "nec2c" else route, long_radial=True)
    off_text, off, _ = run(no_features, route, basis, deck, None)
    on_text, on, declined = run(no_features, route, basis, deck, "1")
    assert on.sector_solves == 0 and not any(on.mapped)
    # The same answer, to the bit, and the reason said.
    assert all(np.array_equal(a, b) for a, b in zip(on.ys, off.ys)) and on.ys
    assert declined and "rotational symmetry: " in declined[0]
    lines_off, lines_on = unclocked(off_text), unclocked(on_text)
    if route == "nec2c":
        assert lines_on == lines_off
    else:
        assert lines_on[:1] + lines_on[2:] == lines_off[:1] + lines_off[2:]


def test_the_nec42_sinusoidal_slot_declines_by_its_junction_rule(no_features):
    # The NEC-4.2 dialect's sinusoidal junction (its cap columns) is not on
    # the sector route (momwire#1912), so there the variable declines.
    deck = screen("nec4")
    off_text, off, _ = run(no_features, "nec4", "sinusoidal", deck, None)
    on_text, on, declined = run(no_features, "nec4", "sinusoidal", deck, "1")
    assert on.sector_solves == 0
    assert all(np.array_equal(a, b) for a, b in zip(on.ys, off.ys)) and on.ys
    assert declined and "NEC-4 junction rule" in declined[0]


def test_razor2p_runs_as_with_it_off(no_features):
    deck = screen("nec5")
    off_text, _, _ = run(no_features, "nec5", "razor2p", deck, None)
    on_text, on, declined = run(no_features, "nec5", "razor2p", deck, "1")
    assert on.sector_solves == 0 and not declined
    assert unclocked(on_text) == unclocked(off_text)
    assert "+rotational_symmetry" not in on_text.splitlines()[1]


def test_line_2_says_uncertified_on_macos(no_features, monkeypatch):
    no_features.setenv(ENV, "1")
    monkeypatch.setattr(F.sys, "platform", "darwin")
    assert F.stamp("dropin", solver="bspline") == (
        "+rotational_symmetry uncertified on this platform"
    )
    monkeypatch.setattr(F.sys, "platform", "linux")
    assert F.stamp("dropin", solver="bspline") == "+rotational_symmetry"
    assert F.stamp("dropin", solver="razor") == ""
