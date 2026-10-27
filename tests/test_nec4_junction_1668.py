"""The NEC-4 junction-charge solve's readings, held to licensed NEC-4.2
(momwire#1668).

momwire#1644 left two cases of `junction_rule="nec42"` open: a feed beside a
radius step (X drifting to -4 ohm with the mesh) and an equal-radius T
(+3 to +10.6 ohm). Both turned out to be the members' charge RATIO alone:
scaling one member's factor by one number brings every feed position on a
deck to NEC-4.2 at once (to 1e-4 of the factor), so the miss was in the
junction solve, not the kernel, the caps or the source. Three readings of
the theory manual's Section 3.4 solve close it (`_nec4_junction`'s
docstring): n = 3, one common junction length (the members' mean), and the
density read at the node for two members but at each member's own segment
centre for three or more.

`tests/test_nec4_radius_step_1644.py` gates the #1644 ladder, the T and the
Yagis; this module adds the decks that tell the three readings apart (each
fails one of them on its own), and decks built after the readings were
fitted (`H*`). Every deck is fed beside a solved node, at 10 to 40 segments
per wire, where the uniform-wire floor is +0.02 to +0.04 ohm of X: each is
held within 0.1 ohm of X and 0.01 of R. Before #1668 these missed by 1.2 to
19 ohm of X.
"""

from __future__ import annotations

import gzip
import math
import pathlib
import re
import warnings

import pytest

from momwire import _nec4_junction as nj
from momwire.eznec._nec4 import solve as nec4_solve

FIX = pathlib.Path(__file__).resolve().parent / "fixtures" / "nec42_junction_1668"
K14 = 2 * math.pi * 14e6 / 299792458.0
_NUM = re.compile(r"[-+]?\d\.\d+E[-+]\d\d")
DECKS = sorted(p.stem for p in (FIX / "decks").glob("*.nec"))


@pytest.fixture(autouse=True)
def _no_rule_override(monkeypatch):
    monkeypatch.delenv("MOMWIRE_JUNCTION_RULE", raising=False)


def _nec42(name):
    with gzip.open(FIX / "nec42" / f"{name}.out.gz", "rt", errors="replace") as fh:
        lines = fh.read().splitlines()
    for i, line in enumerate(lines):
        if "ANTENNA INPUT PARAMETERS" in line:
            v = _NUM.findall(lines[i + 4])
            return complex(float(v[4]), float(v[5]))
    raise AssertionError(f"{name}: no ANTENNA INPUT PARAMETERS")


def _momwire(name):
    text = (FIX / "decks" / f"{name}.nec").read_text()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return nec4_solve(text, basis="sinusoidal").runs[0].sources[0].impedance


def test_every_deck_has_its_printout():
    assert len(DECKS) == 18
    for name in DECKS:
        assert (FIX / "nec42" / f"{name}.out.gz").exists(), name


@pytest.mark.parametrize("name", DECKS)
def test_a_deck_fed_beside_a_solved_node_tracks_nec42(name):
    d = _momwire(name) - _nec42(name)
    assert abs(d.imag) <= 0.1 and abs(d.real) <= 0.01, d


# --- the readings, on the solve itself ---------------------------------------

_STEP = [(0.0, 0.0, -1.0), (0.0, 0.0, 1.0)]
_TEE = [(-1.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 0.0, 1.0)]


def test_one_length_makes_an_equal_radius_pair_uniform_at_any_mesh():
    """The common length makes a two-member node at one radius symmetric
    whatever its two segments are, so the solve gives the uniform factors
    the rule keeps for it anyway (#1644's F3 could not tell the two apart)."""
    for lengths in ([0.1, 0.05], [0.02, 0.3]):
        f = nj.junction_charge_factors(K14, _STEP, [1e-3, 1e-3], lengths, 0.1)
        assert f == pytest.approx([0.1, 0.1], rel=1e-12)


def test_the_solve_takes_the_mean_length_at_a_step():
    """Swapping which half carries the long segment changes nothing: only
    the mean enters a two-member node."""
    a = nj.junction_densities(K14, _STEP, [1e-3, 5e-3], [0.15, 0.075])
    b = nj.junction_densities(K14, _STEP, [1e-3, 5e-3], [0.075, 0.15])
    c = nj.junction_densities(K14, _STEP, [1e-3, 5e-3], [0.1125, 0.1125])
    assert a[0] / a[1] == pytest.approx(c[0] / c[1], rel=1e-12)
    assert b[0] / b[1] == pytest.approx(c[0] / c[1], rel=1e-12)


def test_three_members_read_each_density_at_its_own_segment_centre():
    """At a T the perpendicular arm's own segment length moves only where
    its density is read: the solve itself sees the mean."""
    d = nj.junction_densities(K14, _TEE, [1e-3] * 3, [0.1, 0.1, 0.05])
    mean = (0.1 + 0.1 + 0.05) / 3
    mat = []
    for i in range(3):
        for s in (0.5 * mean, mean, 2.0 * mean):
            p = [s * c for c in _TEE[i]]
            row = []
            for ell in range(3):
                u = _TEE[ell]
                z = sum(pc * uc for pc, uc in zip(p, u, strict=True))
                x = [pc - z * uc for pc, uc in zip(p, u, strict=True)]
                b = math.sqrt(1e-6 + sum(v * v for v in x))
                tri = nj._triangle(K14, b, z, mean)
                row += [
                    tri,
                    nj._full_triangle(K14, b, z, mean, tri),
                    mean * nj._semi_infinite(K14, b, z),
                ]
            mat.append(row)
    q = nj._solve(mat, [1.0] * 9)
    want = [
        nj._density(q[3 * ell : 3 * ell + 3], mean, 0.5 * length, False)
        for ell, length in enumerate((0.1, 0.1, 0.05))
    ]
    assert d == want


def test_the_tee_ratio_is_the_one_nec42_was_measured_to_use():
    """The 20-segment T of `../nec42_radius_step_1644` (`J_T_n20`): scaling
    the perpendicular arm's factor until Z meets NEC-4.2 gives a
    perpendicular / collinear ratio of 0.9118 at every one of six feed
    positions. #1644's n = 2 node reading gave 0.8919."""
    dens = nj.junction_densities(K14, _TEE, [1e-3] * 3, [0.1] * 3)
    assert dens[2] / dens[0] == pytest.approx(0.9118, abs=5e-4)
