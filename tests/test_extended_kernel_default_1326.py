"""NEC-4 and NEC-5 decks solve with the extended kernel by default (momwire#1326).

Steve's decision, 2026-10-04.  NEC-5's kernel behaves as EK-on and its dialect
has no ``EK`` card; NEC-4.2 has one thin-wire model and prints that its ``EK``
card has no effect.  So:

* ``nec5`` (the EZNEC NEC-5 slot and SimNEC's NEC-5 engine): on;
* ``nec4`` (the EZNEC NEC-4.2 slot): on, and an ``EK`` card is ignored, with
  the line NEC-4.2 itself prints under it;
* ``nec2``: unchanged, the card decides.

Where a basis or a deck cannot take the kernel, the default falls back to the
reduced one with an ``ExtendedKernelDefault`` advisory instead of refusing,
because the default is ours, not the deck's.

Every gate below proves the kernel at the SOLVER, not at a flag: the
constructed solver's own ``extended_kernel`` is read back.
"""

from __future__ import annotations

import re
import warnings
from pathlib import Path

import pytest

from momwire._wire_spec import GapMeshFloor
from momwire.deck import DeckError, ExtendedKernelDefault, build_solver, parse
from momwire.deck._nec5 import parse_nec5
from momwire.eznec import _serve
from momwire.eznec._nec4 import render as render_nec4

FIXTURES = Path(__file__).parent / "fixtures"

DIPOLE_NEC2 = (
    "CM d\nCE\nGW 1 21 0 0 -2.5 0 0 2.5 0.05\nGE 0\n"
    "{ek}EX 0 1 11 0 1 0\nFR 0 1 0 0 28.0\nXQ\nEN\n"
)


def _built(text: str, dialect: str, basis: str = "bspline"):
    return build_solver(parse(text, dialect=dialect), basis=basis)


# --------------------------------------------------------------------------
# the three dialects


def test_nec2_keeps_the_cards_word():
    assert not _built(DIPOLE_NEC2.format(ek=""), "nec2").solver.extended_kernel
    assert _built(DIPOLE_NEC2.format(ek="EK\n"), "nec2").solver.extended_kernel
    assert not parse(DIPOLE_NEC2.format(ek=""), dialect="nec2").extended_kernel_default


@pytest.mark.parametrize("ek", ["", "EK\n", "EK -1\n", "EK 0\n"])
def test_nec4_solves_extended_whatever_the_card_says(ek):
    model = parse(DIPOLE_NEC2.format(ek=ek), dialect="nec4")
    assert model.extended_kernel_default
    assert all(g.extended_kernel for g in model.groups if g is not None)
    built = build_solver(model)
    assert built.extended_kernel and built.solver.extended_kernel
    assert built.advisories == ()


def test_a_nec4_ek_card_moves_no_number():
    """Ignored means ignored: the deck with the card and without it are one
    solve, bit for bit."""
    with_card = _built(DIPOLE_NEC2.format(ek="EK -1\n"), "nec4")
    without = _built(DIPOLE_NEC2.format(ek=""), "nec4")
    a = with_card.solver.compute_port_solution().y
    b = without.solver.compute_port_solution().y
    assert (a == b).all()


def test_an_explicit_override_still_turns_it_off():
    model = parse(DIPOLE_NEC2.format(ek=""), dialect="nec4")
    assert not build_solver(model, extended_kernel=False).solver.extended_kernel


# --------------------------------------------------------------------------
# the fallback: a default the basis or the deck cannot take


def test_a_basis_without_the_kernel_falls_back_with_an_advisory():
    with pytest.warns(ExtendedKernelDefault, match="reduced kernel"):
        built = _built(DIPOLE_NEC2.format(ek=""), "nec4", basis="pulse")
    assert not built.extended_kernel
    assert len(built.advisories) == 1 and "'pulse'" in built.advisories[0]


def test_a_buried_wire_falls_back_with_an_advisory():
    """0239, EZNEC's buried dipole under GN 3: every NEC2_BASES family that
    serves a buried wire refuses the extended kernel there."""
    (deck,) = (FIXTURES / "eznec_nec42").glob("0239_*.nec")
    text = deck.read_text(encoding="latin-1")
    with pytest.warns(ExtendedKernelDefault, match="below the ground plane"):
        built = _built(text, "nec4")
    assert not built.solver.extended_kernel


def test_a_request_on_nec2_still_refuses():
    """The fallback is for the DEFAULT only: a nec2 deck's ``EK`` card is the
    deck's own request, and a basis that cannot serve it refuses as before."""
    with pytest.raises((NotImplementedError, ValueError)):
        _built(DIPOLE_NEC2.format(ek="EK\n"), "nec2", basis="pulse")


def test_no_advisory_when_the_kernel_is_served():
    with warnings.catch_warnings():
        warnings.simplefilter("error", ExtendedKernelDefault)
        _built(DIPOLE_NEC2.format(ek=""), "nec4", basis="sinusoidal")


# --------------------------------------------------------------------------
# the fine T node (the issue comment): sinusoidal families under EK fail on a
# fine mesh at a 3-wire junction, and #959's advisory is what says so


T_NODE = (
    "CM t\nCE\n"
    # A 1 m stub meeting two 1.5 m arms at (0, 0, 1.1); 2 mm radius.  Each
    # arm starts with three 0.8 mm segments at the node, 0.4 of the radius:
    # inside the 0.1-0.6 a band the issue measured failing.
    "GW 1 9 0 0 0.1 0 0 1.1 0.002\n"
    "GW 2 3 0 0 1.1 0 -0.0024 1.1 0.002\n"
    "GW 3 9 0 -0.0024 1.1 0 -1.5 1.1 0.002\n"
    "GW 4 3 0 0 1.1 0 0.0024 1.1 0.002\n"
    "GW 5 9 0 0.0024 1.1 0 1.5 1.1 0.002\n"
    "GE 0\nEX 0 1 1 0 1 0\nFR 0 1 0 0 28.0\nXQ\nEN\n"
)


@pytest.mark.parametrize("basis", ["sinusoidal", "sinusoidal-galerkin"])
def test_a_fine_t_node_under_the_default_kernel_is_advised(basis):
    """The default turns the extended kernel on for NEC-4 decks, whose twin
    is sinusoidal, and under it those families fail at a finely meshed node
    of three or more wires.  momwire#959's advisory names the junction."""
    with pytest.warns(GapMeshFloor, match="junction"):
        built = _built(T_NODE, "nec4", basis=basis)
        built.solver.compute_port_solution()
    assert built.solver.extended_kernel


# --------------------------------------------------------------------------
# the NEC-5 slot


def _kernel_seen(monkeypatch) -> list[bool]:
    seen: list[bool] = []
    original = _serve._solver_for

    def spy(*args, **kwargs):
        solver = original(*args, **kwargs)
        seen.append(bool(getattr(solver, "extended_kernel", False)))
        return solver

    monkeypatch.setattr(_serve, "_solver_for", spy)
    return seen


def _capture(cid: str) -> str:
    (path,) = (FIXTURES / "eznec" / "decks").glob(f"{cid}_*.nec")
    return path.read_text(encoding="latin-1")


@pytest.mark.parametrize("basis", ["bspline", "razor-2p"])
def test_the_nec5_slot_solves_extended(monkeypatch, basis):
    seen = _kernel_seen(monkeypatch)
    _serve.serve(parse_nec5(_capture("0010")), basis=basis)
    assert seen == [True]


def test_the_nec5_slot_falls_back_on_a_buried_deck(monkeypatch):
    seen = _kernel_seen(monkeypatch)
    text = (
        "CM b\nCE\nGW 1,9,0.,-2.5,-0.1,0.,2.5,-0.1,.001\nGE -1,0\n"
        "GN 0,0,0,0,13.,.005\nFR 0,1,0,0,7.\nEX 4,1,5,0,1.,0.\nXQ\nEN\n"
    )
    with pytest.warns(ExtendedKernelDefault, match="below the ground plane"):
        _serve.serve(parse_nec5(text))
    assert seen == [False]


def test_the_nec5_dialect_still_has_no_ek_card():
    with pytest.raises(DeckError, match="EK"):
        parse_nec5(_capture("0010").replace("EX ", "EK\nEX ", 1))


# --------------------------------------------------------------------------
# the NEC-4.2 slot's printout


def _z(printout: str) -> complex:
    number = re.compile(r"[-+]?(?:\d+\.?\d*|\.\d+)(?:E[-+]\d+)?")
    lines = printout.splitlines()
    start = next(i for i, ln in enumerate(lines) if "ANTENNA INPUT PARAMETERS" in ln)
    row = next(c for ln in lines[start + 1 :] if len(c := number.findall(ln)) == 11)
    return complex(float(row[6]), float(row[7]))


def test_the_nec42_slot_prints_ek_as_nec42_does():
    """NEC-4.2's own layout (the binary given ``EK 0``, 2026-10-04): the
    card's echo, a blank line, its note, a blank line, the next echo."""
    printout = render_nec4(DIPOLE_NEC2.format(ek="EK 0\n"), basis="bspline")
    assert "NEC ERROR" not in printout
    lines = printout.splitlines()
    at = next(i for i, ln in enumerate(lines) if "INPUT LINE" in ln and " EK " in ln)
    assert lines[at + 1 : at + 4] == [
        "",
        " THE EK AND KH COMMANDS HAVE NO EFFECT IN NEC-4",
        "",
    ]
    assert "INPUT LINE" in lines[at + 4]
    assert _z(printout) == _z(render_nec4(DIPOLE_NEC2.format(ek=""), basis="bspline"))


@pytest.mark.parametrize("basis", ["bspline", "sinusoidal"])
def test_the_fat_dipole_lands_nearer_nec42_with_the_default(monkeypatch, basis):
    """p10, a 28 MHz dipole at Delta/a = 1.5: the issue's ladder measured the
    extended kernel 4-5 ohm nearer NEC-4.2 there on both families."""
    (deck,) = (FIXTURES / "eznec_nec42" / "probes").glob("p10_*.nec")
    (out,) = (FIXTURES / "eznec_nec42" / "printouts").glob("p10_*.out")
    target = _z(out.read_text(encoding="latin-1"))
    text = deck.read_text(encoding="latin-1")
    on = abs(_z(render_nec4(text, basis=basis)) - target)
    from momwire.deck import _solver

    monkeypatch.setattr(
        _solver, "extended_kernel_default_refusal", lambda *a, **k: "forced off"
    )
    with pytest.warns(ExtendedKernelDefault):
        off = abs(_z(render_nec4(text, basis=basis)) - target)
    assert on < off - 1.0, (on, off)
