"""The below/below replay kernel's blocked loop (momwire#1224).

`remainder_field_proj_batch_below` runs each of a pair's five stages
(geometry, stencil address, stencil read, divide-out, projection) over a
block of sources before the next, so the stages' iterations overlap instead
of every pair serialising on its own dependency chain. Every pair's
operations are `proj_one_below`'s, in its order, which is still reachable as
`blocked=False`. So the gate is bit equality, not tolerance:

- G-1224-B1  on the real tables and nodes of a live buried solve, the blocked
             table and the per-pair one are the same uint64, and so are the
             three query extremes the refusals are raised from.
- G-1224-B2  the same at source counts on both sides of the block size and
             not a multiple of it, so a short last block and a single-source
             call are both in the frame.
"""

import sys
import warnings
from pathlib import Path

import numpy as np
import pytest

from momwire import BSplineSolver
from momwire import _sommerfeld_below as below

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_field_galerkin_914 import screen_deck  # noqa: E402

pytestmark = [
    pytest.mark.filterwarnings("ignore:crossing node"),
    pytest.mark.skipif(
        not getattr(below._acc, "below_replay_blocked_1224", False),
        reason="the .so predates the blocked below replay: rebuild",
    ),
]


@pytest.fixture(scope="module")
def live_args():
    """Every argument tuple the kernel was handed during one buried solve."""
    seen = []
    real = below._acc.remainder_field_proj_batch_below

    def spy(*a, **kw):
        seen.append(a)
        return real(*a, **kw)

    mp = pytest.MonkeyPatch()
    mp.setattr(below._acc, "remainder_field_proj_batch_below", spy)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            BSplineSolver(**screen_deck(12)).compute_impedance()
    finally:
        mp.undo()
    assert seen, "the solve never reached the below/below replay kernel"
    return real, seen


def _bits(a):
    return np.ascontiguousarray(a).view(np.uint64)


def _both(real, args):
    got = real(*args, blocked=True)
    ref = real(*args, blocked=False)
    return got, ref


def test_g1224_b1_the_blocked_loop_is_the_per_pair_one(live_args):
    real, seen = live_args
    for i, args in enumerate(seen):
        got, ref = _both(real, args)
        assert np.array_equal(_bits(got[0]), _bits(ref[0])), i
        assert got[1:] == ref[1:], (i, got[1:], ref[1:])
        assert np.all(np.isfinite(got[0])), i


@pytest.mark.parametrize("n_src", [1, 7, 63, 64, 65, 130])
def test_g1224_b2_every_block_shape(live_args, n_src):
    real, seen = live_args
    args = list(max(seen, key=lambda a: len(a[2])))
    src, t_src = args[2], args[3]
    assert len(src) >= n_src, (len(src), n_src)
    # an evenly spread subset, so the short block holds far and near pairs
    take = np.linspace(0, len(src) - 1, n_src).astype(int)
    args[2], args[3] = src[take], t_src[take]
    got, ref = _both(real, args)
    assert got[0].shape[1] == n_src
    assert np.array_equal(_bits(got[0]), _bits(ref[0]))
    assert got[1:] == ref[1:]
