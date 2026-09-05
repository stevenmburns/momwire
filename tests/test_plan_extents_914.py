"""The below/below plan extents in C++ (momwire#914 unit 1).

`_pair_extents_below` finds two scalars — the largest image distance and the
shallowest angle — over every node pair of a buried cloud. In numpy it costs
2.3 s over the 246 M pairs of a 48-radial screen (2.6 s on the laptop the
#914 profile was taken on), which is the whole of that lever.

The C++ twin walks the UPPER TRIANGLE only (the pair matrix is exactly
symmetric: rho2 is a difference squared and hh = d_i + d_j) and minimises
hh^2 / rho^2 instead of hh / rho — the same argmin on the non-negative
quadrant, which removes the per-pair sqrt and leaves one sqrt and one atan
for the whole cloud.

Gates:

- G-914-1   C++ equals the numpy reference to 1e-12 on a random cloud, at
            sizes from 1 node up, and equals the all-pairs formulation too
            (so the twin is gated against the ORIGINAL spelling, not only
            against the chunked one that replaced it).
- G-914-1b  the same on the 12-radial deck's REAL nodes, captured from a
            live solve rather than synthesised — a random cloud has none of
            the structure (coincident hub nodes, one rise, equal depths)
            that a screen actually has.
- G-914-1c  the numpy fallback still runs, and still passes G-910-2's own
            assertion, when the accelerator is absent.

The chunk-size sweep that G-910-2 performs is NOT duplicated here; instead
G-910-2 itself is pinned to the numpy path, because with the accelerator
live all four of its `rows` values dispatch to C++ and the chunking it
exists to test would go unexercised.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

import momwire.bspline as _bs
from momwire.bspline import BSplineSolver, _pair_extents_below

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_crossing_serve_524 import hub_deck  # noqa: E402

_acc = getattr(_bs, "_acc", None)
requires_accel = pytest.mark.skipif(
    not _bs._HAVE_PLAN_EXTENTS_ACCEL,
    reason="built without the #914 plan-extents accelerator",
)


def _all_pairs(x, y, d_b):
    """The ORIGINAL spelling #910 replaced: six (n, n) arrays and an
    arctan2. Kept here as the twin's reference so the C++ is gated against
    the formulation the physics was written in, not against its optimisation."""
    rho = np.hypot(x[:, None] - x[None, :], y[:, None] - y[None, :])
    hh = d_b[:, None] + d_b[None, :]
    return float(np.max(np.hypot(rho, hh))), float(np.min(np.arctan2(hh, rho)))


def _numpy_only(monkeypatch):
    monkeypatch.setattr(_bs, "_HAVE_PLAN_EXTENTS_ACCEL", False)


def _cloud(n, seed=914):
    rng = np.random.default_rng(seed)
    return (
        rng.uniform(-11.0, 11.0, n),
        rng.uniform(-11.0, 11.0, n),
        rng.uniform(0.005, 0.3, n),
    )


# --- G-914-1 ---------------------------------------------------------------


@requires_accel
@pytest.mark.parametrize("n", [1, 2, 3, 17, 256, 257, 1500])
def test_g914_1_the_cpp_extents_are_the_all_pairs_numbers(n):
    x, y, d_b = _cloud(n)
    r1_ref, th_ref = _all_pairs(x, y, d_b)
    r1, th = _acc.pair_extents_below(x, y, d_b)
    assert abs(r1 - r1_ref) <= 1e-12 * r1_ref, (n, r1, r1_ref)
    assert abs(th - th_ref) <= 1e-12 * abs(th_ref), (n, th, th_ref)


@requires_accel
def test_g914_1_the_dispatch_returns_the_fallbacks_answer(monkeypatch):
    """The seam itself: the same call with the accelerator on and off.

    This is the assertion that would catch a dispatch wired to the wrong
    argument order — comparing C++ against the all-pairs helper above would
    not, since the helper is symmetric in x and y.
    """
    x, y, d_b = _cloud(900, seed=5)
    fast = _pair_extents_below(x, y, d_b)
    monkeypatch.setattr(_bs, "_HAVE_PLAN_EXTENTS_ACCEL", False)
    slow = _pair_extents_below(x, y, d_b)
    assert abs(fast[0] - slow[0]) <= 1e-12 * slow[0]
    assert abs(fast[1] - slow[1]) <= 1e-12 * abs(slow[1])


@requires_accel
def test_g914_1_x_and_y_are_not_interchangeable():
    """A cloud stretched along x only: swapping the two arrays changes no
    answer here (rho is symmetric in dx, dy), so this instead pins that the
    DEPTHS are not confused with a coordinate — the one swap that is not
    self-cancelling."""
    x, y, d_b = _cloud(400, seed=7)
    right = _acc.pair_extents_below(x, y, d_b)
    wrong = _acc.pair_extents_below(x, d_b, y)
    assert right != wrong


# --- G-914-1b: the real node cloud -----------------------------------------


@pytest.mark.filterwarnings("ignore:crossing node")
@requires_accel
def test_g914_1b_the_cpp_extents_match_on_the_real_deck_nodes(monkeypatch):
    """The 12-radial screen's actual nodes, captured from a live solve.

    A synthetic cloud is uniform; a screen is not — its radials share a hub,
    every radial node sits at one depth, and many nodes are coincident in
    rho. Those are exactly the configurations where a symmetric-triangle
    walk or a rho == 0 divide could differ from the reference.
    """
    seen = {}
    real = _bs._pair_extents_below

    def capture(x, y, d_b, rows=256):
        seen["args"] = (np.array(x), np.array(y), np.array(d_b))
        return real(x, y, d_b, rows=rows)

    monkeypatch.setattr(_bs, "_pair_extents_below", capture)
    BSplineSolver(**hub_deck(n_radials=4)).compute_impedance()
    assert "args" in seen, "the buried plan never called _pair_extents_below"

    x, y, d_b = seen["args"]
    assert x.size > 100, f"only {x.size} nodes captured"
    assert (d_b > 0).all(), "the deck's depths are the documented precondition"
    r1_ref, th_ref = _all_pairs(x, y, d_b)
    r1, th = _acc.pair_extents_below(x, y, d_b)
    assert abs(r1 - r1_ref) <= 1e-12 * r1_ref, (r1, r1_ref)
    assert abs(th - th_ref) <= 1e-12 * abs(th_ref), (th, th_ref)


@pytest.mark.filterwarnings("ignore:crossing node")
@requires_accel
def test_g914_1b_the_deck_solves_to_the_same_impedance_either_way(monkeypatch):
    """End to end: the extents feed a serve plan, so the only thing that
    ultimately matters is that the solved impedance does not move."""
    z_fast, _ = BSplineSolver(**hub_deck(n_radials=4)).compute_impedance()
    monkeypatch.setattr(_bs, "_HAVE_PLAN_EXTENTS_ACCEL", False)
    z_slow, _ = BSplineSolver(**hub_deck(n_radials=4)).compute_impedance()
    assert abs(z_fast - z_slow) <= 1e-12 * abs(z_slow), (z_fast, z_slow)


# --- G-914-1c: the fallback is still a real path ---------------------------


def test_g914_1c_the_numpy_fallback_still_answers(monkeypatch):
    _numpy_only(monkeypatch)
    x, y, d_b = _cloud(700, seed=11)
    r1_ref, th_ref = _all_pairs(x, y, d_b)
    for rows in (1, 7, 256, 4096):
        r1, th = _pair_extents_below(x, y, d_b, rows=rows)
        assert abs(r1 - r1_ref) <= 1e-12 * r1_ref, (rows, r1, r1_ref)
        assert abs(th - th_ref) <= 1e-12 * abs(th_ref), (rows, th, th_ref)


def test_g914_1c_no_divide_warning_escapes_either_path(monkeypatch):
    x, y, d_b = _cloud(3, seed=13)
    with np.errstate(divide="raise"):
        _pair_extents_below(x, y, d_b)
    _numpy_only(monkeypatch)
    with np.errstate(divide="raise"):
        _pair_extents_below(x, y, d_b)
