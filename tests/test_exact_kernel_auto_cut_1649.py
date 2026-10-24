"""momwire#1649: a sliver the z = 0 cut leaves does not switch "auto" on.

A card crossing the ground plane is cut there ("cut, never snap"), and a
crossing near a segment boundary leaves a piece just above the plane whose
length is where the plane fell, not the user's mesh. `exact_kernel="auto"`
asks every above-ground segment whether it is under 3 radii, so one sliver
turned the exact ring kernel on for the whole deck. The seam now names the
cut pieces (`exact_kernel_auto_skip`) and "auto" does not ask them; a short
segment the user chose still decides, and an explicit True still reaches
every wire.
"""

from __future__ import annotations

import pytest

from momwire.deck._nec5 import parse_nec5
from momwire.eznec import _serve

TAIL = "GE -1\nFR 0,1,0,0,14.\nGN 2,0,0,0,13.,.005\nEX 0,1,2,0,1.,0.\nXQ\nEN\n"

# One card from z = 1.0 m to -0.98 m in ten 0.198 m segments, a = 5 mm: the
# plane falls 1 cm below node 5, so the cut leaves a 1 cm piece above it
# (2 radii) while every user segment is ~40 radii.
SLIVER = "CM 1649\nCE\nGW 1,10,0.,0.,1.,0.,0.,-0.98,.005\n" + TAIL
# The same, with a 1 cm segment the user wrote on top of it.
USER_SHORT = (
    "CM 1649\nCE\nGW 1,10,0.,0.,1.,0.,0.,-0.98,.005\n"
    "GW 2,1,0.,0.,1.,0.,0.,1.01,.005\n" + TAIL
)

BASES = ["bspline", "razor-2p"]


def _solver(text, basis, monkeypatch):
    built = {}
    orig = _serve.construct_solver

    def capture(cls, kwargs, **rest):
        built["kwargs"] = dict(kwargs)
        built["rest"] = rest
        built["solver"] = orig(cls, kwargs, **rest)
        built["cls"] = cls
        return built["solver"]

    monkeypatch.setattr(_serve, "construct_solver", capture)
    _serve.serve(parse_nec5(text), basis=basis)
    return built


@pytest.mark.parametrize("basis", BASES)
def test_the_cut_sliver_alone_leaves_auto_off(basis, monkeypatch):
    built = _solver(SLIVER, basis, monkeypatch)
    assert built["kwargs"].get("exact_kernel") == "auto"
    assert built["kwargs"].get("exact_kernel_auto_skip"), built["kwargs"]
    assert built["solver"].exact_kernel is False
    # ...and the sliver is what would have switched it on: the same pieces
    # without the skip resolve True
    kwargs = dict(built["kwargs"])
    del kwargs["exact_kernel_auto_skip"]
    monkeypatch.undo()
    again = _serve.construct_solver(built["cls"], kwargs, **built["rest"])
    assert again.exact_kernel is True


@pytest.mark.parametrize("basis", BASES)
def test_a_short_segment_the_user_wrote_still_switches_auto_on(basis, monkeypatch):
    built = _solver(USER_SHORT, basis, monkeypatch)
    assert built["kwargs"].get("exact_kernel_auto_skip")
    assert built["solver"].exact_kernel is True


def test_a_deck_with_no_cut_passes_no_skip(monkeypatch):
    text = "CM 1649\nCE\nGW 1,10,0.,0.,1.,0.,0.,0.,.005\n" + TAIL
    built = _solver(text, "bspline", monkeypatch)
    assert "exact_kernel_auto_skip" not in built["kwargs"]
