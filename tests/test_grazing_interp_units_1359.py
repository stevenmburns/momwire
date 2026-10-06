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
# weights' own rounding: a few ulp of the node scale. MEASURED (momwire#1359
# probe, fixtures written on an AVX non-FMA build): bit-identical on CI's
# AVX2+FMA Linux runner and on macOS arm64. 1e-12 leaves room for a numpy
# that reorders the sum, and is still 1e5 under the smallest bar (3.9e-7).
# What it catches that the bar does not: a stencil change too small to break
# the bar -- weights shifted by 1e-7 of a cell move these values 5e-11 to
# 1.3e-8 and stay under every bar.
INTERP_PIN = 1e-12

# The fill tolerance, against the stored node values. Each node is the four
# surfaces built from six contour integrals, each converged to the fine
# machine's min(rtol, 1e-11); a floor-band node sums ~22,000 tail panels. A
# different ISA rounds every panel differently, so agreement is to the
# integrator's accuracy, not to the bit. MEASURED by recomputing EVERY stored
# node cold (not just the fill sample) against the fixture written on gesher
# (AVX, non-FMA):
#
#                   floor     lo        mid       far
#   CI AVX2+FMA     5.9e-10   5.1e-11   1.6e-11   5.3e-13
#   macOS arm64     1.3e-9    8.0e-11   1.3e-11   8.6e-13
#   numpy path      5.6e-12   6.0e-12   1.0e-12   2.2e-14   (same box, sampled rows)
#
# So the grid's own rtol (1e-9) is NOT a safe cross-platform tolerance for the
# floor band: macOS sits 1.3x over it. 1e-8 is ~8x over the worst measured
# spread. What it must still catch, and does (red runs on #1359): a sign in a
# surface (moves a node 2.0 of its scale), a tail that drops its first panel
# (0.76-4.4), a panel cap the nodes cannot converge under (raises). A change
# that moves nodes by less -- e.g. stopping the tail one quiet panel early,
# 2e-10 -- is inside the integrator's own accuracy; it is caught as a CODE
# change by `test_the_fixture_is_current`, not here.
FILL_RTOL = 1e-8


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
    those rows agrees with the fixture to FILL_RTOL: the integrals are
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
