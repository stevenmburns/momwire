"""The radius-step advisory names the DECK's wires (momwire#2088).

`RadiusStepJunction` (momwire#1632) used to say "wires 0 and 1", the solver's
own zero-based wire indices, which are the solver's order after the importer's
splits and merges and match no tag in the deck. Through a deck it now names
GW tags; through the Python API (no deck) it keeps the index and says so.

The deck cases go through the real launcher process, so the importer, the
mesh and the constructor are the production ones.
"""

from __future__ import annotations

import subprocess
import sys
import warnings
from pathlib import Path

import numpy as np
import pytest

from momwire.sinusoidal import RadiusStepJunction, SinusoidalSolver

_BIN = Path(sys.executable).parent


def _launcher(name: str) -> str:
    path = _BIN / name
    if not path.exists():
        pytest.skip(f"{name} is not installed beside this interpreter")
    return str(path)


def _run(deck: str, launcher: str = "momwire-nec2c-sinusoidal", advisories=True):
    """``(stdout, stderr)`` of one launcher run over `deck`."""
    env = {"PATH": str(_BIN)}
    if advisories:
        env["MOMWIRE_ADVISORIES"] = "stderr"
    done = subprocess.run(
        [_launcher(launcher), "-", "-"],
        input=deck,
        capture_output=True,
        text=True,
        env=env,
        check=True,
        timeout=300,
    )
    return done.stdout, done.stderr


def _untimed(printout: str) -> str:
    """The printout without its wall-clock FILL/FACTOR line."""
    return "\n".join(ln for ln in printout.splitlines() if "FILL:" not in ln)


def _step_lines(stderr: str) -> list[str]:
    return [ln for ln in stderr.splitlines() if "RadiusStepJunction" in ln]


_TAIL = "GE 0\nEX 0 {tag} {seg} 0 1 0\nFR 0 1 0 0 14 0\nXQ\nEN\n"


def _deck(*wires: str, feed=(1, 10)) -> str:
    return (
        "CM stepped\nCE\n"
        + "\n".join(wires)
        + "\n"
        + _TAIL.format(tag=feed[0], seg=feed[1])
    )


def test_a_deck_names_its_gw_tags():
    """The issue's repro: GW 1 and GW 2, not indices 0 and 1."""
    deck = _deck(
        "GW 1 10 0 0 -3 0 0 0 0.0005",
        "GW 2 10 0 0 0 0 0 3 0.001",
    )
    _out, err = _run(deck)
    (line,) = _step_lines(err)
    assert "wires 1 and 2 (0.5 mm to 1 mm)" in line
    assert "wires 0 and 1" not in line


def test_non_sequential_tags_show_the_tags_not_index_plus_one():
    deck = _deck(
        "GW 7 10 0 0 -3 0 0 0 0.0005",
        "GW 3 10 0 0 0 0 0 3 0.001",
        feed=(7, 10),
    )
    _out, err = _run(deck)
    (line,) = _step_lines(err)
    assert "wires 3 and 7 (0.5 mm to 1 mm)" in line


def test_a_card_the_importer_split_names_that_one_tag():
    """GW 2's end lands mid-span on GW 1, so the importer cuts GW 1 into two
    solver wires: three wires meet at the junction, two tags are named."""
    deck = _deck(
        "GW 5 20 0 0 -3 0 0 3 0.0005",
        "GW 9 10 0 0 0 2 0 0 0.001",
        feed=(5, 5),
    )
    _out, err = _run(deck)
    (line,) = _step_lines(err)
    assert "wires 5 and 9 (0.5 mm to 1 mm)" in line
    assert "wires 0, 1" not in line and "indices" not in line


def test_the_printout_is_untouched_by_the_naming():
    """The advisory rides the advisory channel: stdout is the same bytes
    whether or not it is shown."""
    deck = _deck(
        "GW 1 10 0 0 -3 0 0 0 0.0005",
        "GW 2 10 0 0 0 0 0 3 0.001",
    )
    shown, _ = _run(deck)
    hidden, err = _run(deck, advisories=False)
    assert _untimed(shown) == _untimed(hidden)
    assert "RadiusStepJunction" not in err


def test_a_deck_over_a_ground_names_tags_too():
    """The same naming over a reflection-coefficient ground (GE 1, GN 0),
    where the structure stands clear of the plane."""
    deck = (
        "CM stepped, over a ground\nCE\n"
        "GW 4 10 0 0 0.2 0 0 3 0.0005\n"
        "GW 6 10 0 0 3 0 0 6 0.001\n"
        "GE 1\nGN 0 0 0 0 13 0.005\n"
        "EX 0 4 5 0 1 0\nFR 0 1 0 0 14 0\nXQ\nEN\n"
    )
    _out, err = _run(deck)
    (line,) = _step_lines(err)
    assert "wires 4 and 6" in line


def test_the_nec42_launcher_serves_an_air_step_and_stays_silent():
    """NEC-4's junction solve serves a step in air, so there is nothing to
    advise and nothing to name: the naming did not make it speak."""
    deck = _deck(
        "GW 7 10 0 0 -3 0 0 0 0.0005",
        "GW 3 10 0 0 0 0 0 3 0.001",
        feed=(7, 10),
    )
    _out, err = _run(deck, launcher="momwire-nec42-sinusoidal")
    assert _step_lines(err) == []


def _api_messages(**kw) -> list[str]:
    left = np.array([(-3.0, 0.0, 0.0), (0.0, 0.0, 0.0)])
    right = np.array([(0.0, 0.0, 0.0), (3.0, 0.0, 0.0)])
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        SinusoidalSolver(
            wavelength=299792458.0 / 14e6,
            wire_radius=[0.0005, 0.001],
            wires=[left, right],
            n_per_edge_per_wire=[[20], [20]],
            feeds=[(1, 0.075, 1 + 0j)],
            **kw,
        )
    return [
        str(w.message) for w in caught if issubclass(w.category, RadiusStepJunction)
    ]


def test_the_python_api_keeps_the_index_and_says_it_is_one():
    (msg,) = _api_messages()
    assert "wire indices 0 and 1 (0.5 mm to 1 mm)" in msg


def test_labels_given_to_the_constructor_are_what_it_names():
    (msg,) = _api_messages(wire_labels=[(12,), (40,)])
    assert "wires 12 and 40 (0.5 mm to 1 mm)" in msg
    assert "indices" not in msg


def test_a_wire_without_a_label_sends_the_step_back_to_indices():
    (msg,) = _api_messages(wire_labels=[(12,), None])
    assert "wire indices 0 and 1" in msg
