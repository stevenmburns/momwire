"""momwire#1672: the buried remainder near the surface.

A buried wire within a segment of the surface sees its own mirror there at
d + d', and the below/below Sommerfeld remainder carries a spike of that
width. Both bases integrated it at one flat order, so radials leaving a base
at z = 0 and sloping a few centimetres into the soil converged tens of ohms
slowly in X. razor-2p now keys the order to grazing height
(`_remainder_qp`, as above ground), and bs2 raises the listed pairs to their
own order (`_below_pair_correction`), both only as far as the grid's
grazing floor serves.

The decks under ``tests/fixtures/buried_grazing_1672`` are a 5 m vertical
(a = 30 mm) fed at its base over four 5 m radials (a = 2 mm) leaving the
base node, at 14 MHz over GN 2 (eps 13, sigma 0.005): ``shallow`` slopes to
5 cm, ``deep`` to 0.5 m; ``_s40`` / ``_s200`` are 40 and 200 segments per
wavelength. The licensed NEC-5 (nec5cl-simd-d0c1383, a black box,
2026-10-24) reads them as below; momwire main (26785ef6) read razor-2p
60.54 - j23.83 and bs2 65.96 - j17.39 on ``shallow_s40``.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pytest

from momwire import _below_interface
from momwire.bspline import BSplineSolver
from momwire.eznec._shell import render

FIXTURES = Path(__file__).parent / "fixtures" / "buried_grazing_1672"

# licensed NEC-5 d0c1383, the ANTENNA INPUT PARAMETERS row
NEC5 = {
    "shallow_s40": complex(67.534, -7.1501),
    "shallow_s200": complex(68.271, -4.5686),
    "deep_s40": complex(66.227, -6.4838),
}


def _z(text: str) -> complex:
    lines = text.splitlines()
    i = next(k for k, line in enumerate(lines) if "ANTENNA INPUT PARAMETERS" in line)
    for line in lines[i + 1 :]:
        v = re.findall(r"-?\d\.\d+E[+-]\d+", line)
        if len(v) >= 6:
            return complex(float(v[4]), float(v[5]))
    raise AssertionError("no impedance row")


def _deck(name: str) -> str:
    return (FIXTURES / f"{name}.nec").read_text()


@pytest.mark.parametrize("name", sorted(NEC5))
def test_razor_reads_the_shallow_radials_as_nec5_does(name):
    """razor-2p is NEC-5's own basis: with the remainder resolved it lands
    on the licensed engine at a coarse mesh too (main was 7 + j17 ohm off on
    shallow_s40)."""
    z = _z(render(_deck(name), basis="razor-2p"))
    assert abs(z - NEC5[name]) < 0.5, (name, z)


def test_bs2_converges_on_the_shallow_radials():
    """bs2 converges to its own value; what #1672 fixes is how fast. Main
    moved 9.4 ohm in X between 40 and 200 segments per wavelength."""
    z40 = _z(render(_deck("shallow_s40"), basis="bspline"))
    z200 = _z(render(_deck("shallow_s200"), basis="bspline"))
    assert abs(z40 - z200) < 2.5, (z40, z200)


def _radials(depth: float, n: int):
    seg_l, seg_r = [], []
    for x, y in [(5.0, 0.0), (-5.0, 0.0), (0.0, 5.0), (0.0, -5.0)]:
        for k in range(n):
            a, b = k / n, (k + 1) / n
            seg_l.append((a * x, a * y, -a * depth))
            seg_r.append((b * x, b * y, -b * depth))
    return np.array(seg_l), np.array(seg_r)


def test_bs2_lists_the_pairs_near_the_surface_only():
    """The listing rule is the above-ground one (#631/#1189) below the
    plane: radials sloping to 5 cm list pairs, and level radials a metre
    down list none, so their fill is the base order's, bit for bit."""
    solver = BSplineSolver.__new__(BSplineSolver)
    I, _J, Q = solver._below_remainder_pairs(*_radials(0.05, 9), 0.0, 6)
    assert I.size and Q.max() > 6
    sl, sr = _radials(0.0, 9)
    sl[:, 2] -= 1.0
    sr[:, 2] -= 1.0
    I, J, Q = solver._below_remainder_pairs(sl, sr, 0.0, 6)
    assert I.size == J.size == Q.size == 0


def test_a_halved_list_reaches_the_base_fill():
    pairs = (np.array([0, 1]), np.array([1, 0]), np.array([192, 13]))
    I, J, Q = _below_interface._halved(pairs, 6)
    assert I.tolist() == [0] and Q.tolist() == [96]
    assert _below_interface._halved((I, J, np.array([12])), 6) is None


def _shallow_orders(name):
    """The per-segment remainder orders razor fills the deck's buried part
    at (momwire#1688), captured from the fill's own call."""
    from momwire import razor as rz

    seen = []
    orig = rz._remainder_qp_per_source

    def spy(*a, **k):
        out = orig(*a, **k)
        seen.append(np.asarray(out).copy())
        return out

    rz._remainder_qp_per_source = spy
    try:
        render(_deck(name), basis="razor-2p")
    finally:
        rz._remainder_qp_per_source = orig
    return seen


def test_razor_fills_each_buried_segment_at_its_own_order():
    """momwire#1688: one hub segment asking a high order no longer charges
    every segment of the fill. On shallow_s40 each radial's orders fall from
    the hub outward (101 down to 6), so the fill evaluates well under half
    the nodes the single highest order would have cost."""
    from momwire import razor as rz

    seen = _shallow_orders("shallow_s40")
    assert seen, "the per-segment rule never ran"
    q = seen[-1]
    assert len(set(q.tolist())) > 1, q
    assert rz._REMAINDER_QP_CAP >= int(q.max()) > 3
    assert int(q.sum()) < 0.5 * q.size * int(q.max()), q


def test_the_per_source_rule_peaks_at_the_scalar_rule():
    from momwire import _quadrature as qd

    rng = np.random.default_rng(1688)
    for _ in range(200):
        n, m = int(rng.integers(1, 12)), int(rng.integers(1, 30))
        sl = rng.normal(size=(n, 3)) * 2
        sr = sl + rng.normal(size=(n, 3))
        sl[:, 2] -= rng.uniform(0, 0.3, n)
        sr[:, 2] -= rng.uniform(0, 0.3, n)
        obs = rng.normal(size=(m, 3)) * 2
        obs[:, 2] = -np.abs(obs[:, 2]) * 0.1
        scalar = qd.remainder_qp(obs, sl, sr, 0.0, 3, 192, 1.0)
        per = qd.remainder_qp_per_source(obs, sl, sr, 0.0, 3, 192, 1.0)
        assert scalar == int(per.max())


def test_ragged_windows_are_each_segment_at_its_own_order():
    """`RemainderBelow.field_windows` at per-segment orders gives every
    segment the moments the uniform rule gives it at that segment's order,
    bit for bit."""
    from momwire import _potential_ground as pg

    seen = {}
    orig = pg.RemainderBelow.field_windows

    def spy(self, observers, sources=None, **kw):
        seen["args"] = (self, observers, sources, kw)
        return orig(self, observers, sources, **kw)

    pg.RemainderBelow.field_windows = spy
    try:
        render(_deck("shallow_s40"), basis="razor-2p")
    finally:
        pg.RemainderBelow.field_windows = orig
    rem, observers, sources, kw = seen["args"]
    orders = np.asarray(kw["n_qp"])
    assert len(set(orders.tolist())) > 1
    n_obs = min(8, observers[0].shape[0])
    ragged = orig(rem, observers, sources, **kw)(0, n_obs)
    for q in sorted(set(orders.tolist())):
        idx = np.flatnonzero(orders == q)
        sub = tuple(a[idx] for a in sources)
        alone = orig(rem, observers, sub, **{**kw, "n_qp": int(q)})(0, n_obs)
        assert np.array_equal(ragged[:, idx, :], alone), q
