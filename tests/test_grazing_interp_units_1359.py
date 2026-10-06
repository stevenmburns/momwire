"""momwire#1359: the grazing-band interpolation gates, as unit tests.

The four "interpolates to the bar" sweeps -- the floor band (#1064), the low
band (#935), the mid band and the far annulus (#838) -- each cold-filled whole
theta bands on six decks to check a few dozen interpolated values: 27 minutes
of the slow lane for the floor band alone, pinned to one xdist worker. They
test two different things, and each is cheap on its own:

  * THE STENCIL. Given the right node values, the bicubic on the band's
    lattice reaches the bar. Checked by READING the stored nodes of one deck
    (each sweep's measured worst) back into the production grid and
    interpolating at the sweep's own query points. Nothing integrates.
  * THE FILL. The grid's nodes are the right integrals in the right slots.
    Checked by WRITING a few nodes of the same deck, cold, through the
    production `_fill_region` restricted to one R1 row per region, and
    comparing them with the stored values and with the direct evaluation the
    fill made at each slot's own node.

The full six-deck sweeps still run, as `nightly` (scheduled and on release
tags): a soil- or frequency-dependent regression shows there, not here.

The fixtures live in `tests/fixtures/grazing_1359/` with the script that
regenerates them; `grazing_fixture_1359.py` is the machinery both share.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

import grazing_fixture_1359 as gf  # noqa: E402
from momwire import _sommerfeld_below as below  # noqa: E402
from momwire._sommerfeld import _SURF_KEYS  # noqa: E402

SWEEPS = list(gf.SWEEPS)

# The interpolation pin. A stored-node interpolation is a fixed linear
# combination of sixteen stored values with Lagrange weights of magnitude
# under ~1.1, so two platforms can differ only by summation order and the
# weights' own rounding: a few ulp of the node scale, ~1e-15. 1e-12 is a
# thousand times that, and a thousand times under the smallest bar (3.9e-7).
# What it catches that the bar does not: a stencil change too small to break
# the bar, e.g. weights shifted by a fraction of a cell.
INTERP_PIN = 1e-12

# The fill tolerance, against the stored node values. Each node is the four
# surfaces built from six contour integrals, each converged to the fine
# machine's min(rtol, 1e-11). A different ISA (the fixture is written on an
# AVX non-FMA build; CI reads it on AVX2+FMA and on macOS arm64) changes the
# rounding of every panel and can flip an adaptive split near its threshold,
# so agreement is to the integrator's tolerance, not to the bit. The bar is
# the grid's own promise, rtol = 1e-9 of the node scale -- 100x the fine
# machine's tolerance. A wrong integral (a sign, a panel cap, a wrong node)
# moves a node by 1e-6 or far more.
FILL_RTOL = gf.RTOL


def _fixture(sweep):
    if not gf.path(sweep).exists():
        pytest.fail(f"missing fixture {gf.path(sweep)}; run `{gf.REGENERATE}`")
    return gf.load(sweep)


@pytest.mark.parametrize("sweep", SWEEPS)
def test_the_fixture_is_current(sweep):
    """A stale fixture fails here, by name, before it can mislead the gates
    below. Stale = a constant, or a hashed function on the path from a deck to
    an interpolated value, differs from what the fixture was generated with
    (`gf.staleness`); comments, docstrings and re-wraps do not count."""
    problems = gf.staleness(_fixture(sweep))
    assert not problems, (
        f"STALE FIXTURE {gf.path(sweep).name} (generated at "
        f"{_fixture(sweep)['header']['generated_at_commit'][:12]}): "
        + "; ".join(problems)
        + f". Regenerate with `{gf.REGENERATE} {sweep}`; it prints how far the "
        "stored values moved, which is what to review."
    )


@pytest.mark.parametrize("sweep", SWEEPS)
def test_the_stencil_interpolates_the_stored_nodes_to_the_bar(sweep):
    """The sweep's own queries, interpolated from the stored nodes alone.

    The grid holds exactly the nodes the production stencil read when the
    fixture was written, and NaN elsewhere; its fill raises. So a stencil that
    reads a different node returns NaN, a query routed to another region
    fails by name, and the bar is checked against the stored direct reference.
    """
    fx = _fixture(sweep)
    g = gf.loaded_grid(fx)
    got = g.eval(fx["query_r1"], fx["query_th"])
    got = np.stack([np.asarray(got[k], complex) for k in _SURF_KEYS])

    bad = ~np.isfinite(got).all(axis=0)
    assert not bad.any(), (
        f"{int(bad.sum())} of {bad.size} queries read a node the fixture does not "
        "hold (NaN): the stencil no longer reads the nodes it read at generation"
    )
    err = gf.rel_error(sweep, got, fx["ref"])
    s, n = np.unravel_index(np.argmax(err), err.shape)
    assert err[s, n] < gf.bar(sweep), (
        f"{sweep}: {err[s, n]:.3e} at R1 = {float(fx['query_r1'][n])!r}, theta = "
        f"{np.degrees(fx['query_th'][n]):.6g} deg, {_SURF_KEYS[s]} (bar "
        f"{gf.bar(sweep):.1e}; {fx['header']['measured_worst']:.3e} at generation)"
    )
    scale = gf.node_scale(fx["ref"])[None, :]
    pin = np.abs(got - fx["interp"]) / scale
    assert pin.max() <= INTERP_PIN, (
        f"{sweep}: the interpolation moved {pin.max():.3e} of the node scale from "
        f"the stored nodes' own (pin {INTERP_PIN:.0e}) while staying under the bar: "
        "the stencil changed"
    )


@pytest.mark.parametrize("sweep", SWEEPS)
def test_a_cold_fill_of_a_few_nodes_reproduces_them(sweep):
    """One R1 row per region the sweep reads, filled cold by the production
    `_fill_region` restricted to it -- for the floor band that is the joint
    floor+low evaluation, and the copy of the two shared columns.

    Two checks. Every slot holds, bit for bit, the direct evaluation the fill
    made AT THAT SLOT'S NODE (the low band's node, for the floor band's shared
    columns): the right integral in the right place. And every stored node in
    those rows agrees with the fixture to the grid's rtol: the integrals are
    the ones the interpolation gate above was given.
    """
    fx = _fixture(sweep)
    soil, f = fx["header"]["deck"]
    h = below.Health()
    g = gf.lattice_grid(soil, f, health=h)
    stored = {
        (int(r), int(i), int(j)): fx["node_vals"][:, k]
        for k, (r, i, j) in enumerate(
            zip(fx["node_region"], fx["node_i"], fx["node_j"])
        )
    }
    checked = 0
    worst, where = 0.0, None
    for idx, rows in fx["header"]["fill_sample"].items():
        idx = int(idx)
        with gf.spy_direct() as seen:
            filled = gf.fill_rows(g, idx, rows)
        bad = gf.misplaced_slots(g, filled, seen)
        assert not bad, f"{sweep}: misplaced fill slots {bad[:4]}"
        rows_got, sub = filled[idx]
        for a, i in enumerate(rows_got):
            for j in range(sub.shape[2]):
                want = stored.get((idx, int(i), j))
                if want is None:
                    continue
                d = np.abs(sub[:, a, j] - want).max() / np.abs(want).max()
                checked += 1
                if not d <= worst:
                    worst, where = d, (idx, int(i), j)
    assert checked and h.evaluations, (
        f"{sweep}: {checked} stored nodes compared after {h.evaluations} direct "
        "evaluations; the gate measured nothing"
    )
    assert worst <= FILL_RTOL, (
        f"{sweep}: node (region, i, j) = {where} moved {worst:.3e} of its scale from "
        f"the stored value ({checked} nodes compared; tolerance {FILL_RTOL:.0e})"
    )
    assert h.nonconvergent == 0, (
        f"{sweep}: {h.nonconvergent} sampled nodes hit the panel cap"
    )
