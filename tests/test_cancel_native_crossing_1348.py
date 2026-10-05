"""`cancel_flag` reaches the crossing fill's and the near-interface tables'
native calls (momwire#1348 scope).

#1342 gave `_crossing_fill` and `_near_interface` Python poll sites, but no
native call from either module took the flag, so each one ran to completion
once entered. The census below (Skylake, hub/invl decks, longest SINGLE call,
1 and 4 OpenMP threads; antennaknobs scratch has the full table) decided which
ones now take it -- every one whose longest call passed ~0.1 s at x32:

    factorize_rows       razor invl x32  0.81 s     (also factorize_ints)
    merge_rows_by_z      razor invl x32  0.69 s
    group_first_ranks    razor invl x32  0.20 s
    RowIndex.find        razor invl x32  0.19 s
    RowGroups.add        sin invl x32    0.14 s
    near_interface_six_columns   x16 1-thread 0.23 s

The serial hash loops throw at a per-block / per-row poll; the OpenMP ones
drain. What this pins, deterministically (wall time never gates CI):

  * each kernel raises `SolveAborted` on a tripped flag, and an untripped one
    changes no bit of its answer;
  * `RowGroups.add` tripped between two calls does no work item of the second:
    the grouping is exactly what the first call left;
  * from a solve, tripped as the first call starts, the kernel ITSELF raises
    (so the solver's token really is ambient there) and is not called again;
    a control counts the calls with no trip.
"""

from __future__ import annotations

import numpy as np
import pytest

from momwire import (
    BSplineSolver,
    CancelToken,
    RazorSolver,
    SinusoidalGalerkinSolver,
    SinusoidalSolver,
    SolveAborted,
    _accel,
    _cancel,
    _near_interface,
)
from momwire._near_interface import k_medium
from test_crossing_serve_524 import hub_deck, invl_deck

acc = _accel.acc
pytestmark = pytest.mark.skipif(
    acc is None
    or _near_interface._nia is None
    or not hasattr(acc, "merge_rows_by_z_1290"),
    reason="accelerators not built",
)

KP = 2.0 * np.pi / 42.831
EPS_T = 13.0 - 12.84j


def _tripped():
    tok = CancelToken()
    tok.cancel()
    return tok


def _same(a, b):
    if isinstance(a, tuple):
        assert len(a) == len(b)
        for x, y in zip(a, b):
            _same(x, y)
    else:
        assert np.array_equal(np.asarray(a), np.asarray(b))


# --------------------------------------------------------------------------
# Each kernel, called directly.
# --------------------------------------------------------------------------


def _cols(n=5000, seed=1):
    rng = np.random.default_rng(seed)
    return [rng.integers(0, 40, n).astype(float) for _ in range(3)]


_KERNELS = {
    "factorize_rows": lambda flag: acc.factorize_rows(_cols(), cancel_flag=flag),
    "factorize_ints": lambda flag: acc.factorize_ints(
        [c.astype(np.int64) for c in _cols()], cancel_flag=flag
    ),
    "RowIndex.find": lambda flag: acc.RowIndex(_cols()[:2]).find(
        _cols(seed=2)[:2], cancel_flag=flag
    ),
    "group_first_ranks": lambda flag: acc.group_first_ranks(
        np.random.default_rng(3).integers(0, 50, (64, 30)).astype(np.int32),
        50,
        cancel_flag=flag,
    ),
    # Two groups' keys ([0, 1] and [1, 2]) walked as three blocks, two of
    # them under z id 0, so both of the merge's branches run.
    "merge_rows_by_z": lambda flag: acc.merge_rows_by_z(
        np.array([0, 0, 1], dtype=np.int64),
        np.array([0, 1, 0], dtype=np.int64),
        np.array([0, 2, 4], dtype=np.int64),
        np.array([0, 1, 1, 2], dtype=np.int32),
        np.array([0, 2, 4], dtype=np.int64),
        3,
        2,
        cancel_flag=flag,
    ),
}


@pytest.mark.parametrize("name", list(_KERNELS))
def test_a_kernel_raises_on_a_tripped_flag_and_an_untripped_one_moves_no_bit(name):
    run = _KERNELS[name]
    live, tripped = CancelToken(), _tripped()  # held: a flag is an address
    _same(run(0), run(live.ptr))
    with pytest.raises(SolveAborted):
        run(tripped.ptr)


def test_the_column_twin_raises_on_a_tripped_flag_and_an_untripped_one_moves_no_bit():
    k_m = k_medium(EPS_T, KP)
    rho = np.repeat(np.linspace(0.2, 3.0, 12), 20)
    z = np.tile(np.linspace(0.0, 0.5, 20), 12)
    sub = np.stack([rho, z, np.full_like(rho, -0.15)], axis=1)
    ref = _near_interface._column_twin(KP, k_m, sub, _near_interface._LAM_MULT)
    with _cancel.scope(CancelToken()):
        got = _near_interface._column_twin(KP, k_m, sub, _near_interface._LAM_MULT)
    _same(ref, got)
    with _cancel.scope(_tripped()), pytest.raises(SolveAborted):
        _near_interface._column_twin(KP, k_m, sub, _near_interface._LAM_MULT)


def test_row_groups_tripped_between_calls_does_no_work_of_the_second():
    g = acc.RowGroups()
    live, tripped = CancelToken(), _tripped()  # held: a flag is an address
    g.add(_cols(seed=4), cancel_flag=live.ptr)
    before = (len(g), g.rows().copy())
    with pytest.raises(SolveAborted):
        g.add(_cols(seed=5), cancel_flag=tripped.ptr)
    assert len(g) == before[0]
    assert np.array_equal(g.rows(), before[1])


# --------------------------------------------------------------------------
# From a solve: the token is ambient where each kernel runs.
# --------------------------------------------------------------------------

_MAKE = {
    "razor": lambda **kw: RazorSolver(**kw, nec5_quadrature=True),
    "bs2": lambda **kw: BSplineSolver(**kw),
    "sin": lambda **kw: SinusoidalSolver(**kw),
    "sg": lambda **kw: SinusoidalGalerkinSolver(**kw),
}

# (engine, deck, the kernel to intercept): one route per kernel, each a
# kernel that solve is measured to call (the census above).
_ROUTES = [
    ("razor", "invl", "merge_rows_by_z"),
    ("razor", "invl", "group_first_ranks"),
    ("razor", "invl", "factorize_rows"),
    ("razor", "invl", "RowIndex.find"),
    ("razor", "buried", "near_interface_six_columns"),
    ("bs2", "buried", "near_interface_six_columns"),
    ("sg", "invl", "near_interface_six_columns"),
    ("sin", "invl", "RowGroups.add"),
]


def _deck(name):
    return hub_deck(n_radials=4) if name == "buried" else invl_deck(n_radials=4)


def _intercept(monkeypatch, kernel, tok=None):
    """Count calls of `kernel`; with `tok`, trip it as the first call starts,
    so the kernel itself sees the flag. Returns the record."""
    rec = {"calls": 0, "kernel_raised": False}

    def wrap(real):
        def counting(*a, **kw):
            rec["calls"] += 1
            if tok is not None and rec["calls"] == 1:
                tok.cancel()
            try:
                return real(*a, **kw)
            except SolveAborted:
                rec["kernel_raised"] = True
                raise

        return counting

    if kernel == "near_interface_six_columns":
        real_mod = _near_interface._nia
        twin = wrap(real_mod.near_interface_six_columns)

        class Proxy:
            def __getattr__(self, name):
                return getattr(real_mod, name)

            near_interface_six_columns = staticmethod(twin)

        monkeypatch.setattr(_near_interface, "_nia", Proxy())
    elif "." in kernel:
        cls_name, meth = kernel.split(".")
        cls = getattr(acc, cls_name)
        real = getattr(cls, meth)
        counting = wrap(real)
        monkeypatch.setattr(cls, meth, lambda self, *a, **kw: counting(self, *a, **kw))
    else:
        monkeypatch.setattr(acc, kernel, wrap(getattr(acc, kernel)))
    return rec


@pytest.mark.parametrize("engine,deck,kernel", _ROUTES)
def test_tripped_inside_its_first_call_the_kernel_raises_from_a_solve(
    monkeypatch, engine, deck, kernel
):
    tok = CancelToken()
    rec = _intercept(monkeypatch, kernel, tok)
    s = _MAKE[engine](**_deck(deck), cancel=tok)
    with pytest.raises(SolveAborted):
        s.compute_impedance()
    assert rec["calls"] == 1
    assert rec["kernel_raised"], "the kernel ran to completion: no flag reached it"


@pytest.mark.parametrize("engine,deck,kernel", _ROUTES)
def test_not_cancelled_the_solve_reaches_the_kernel(monkeypatch, engine, deck, kernel):
    """The control: the same solve does call the kernel, so the test above
    is not passing on a route that never reaches it."""
    rec = _intercept(monkeypatch, kernel)
    z = _MAKE[engine](**_deck(deck), cancel=CancelToken()).compute_impedance()[0]
    assert rec["calls"] >= 1
    assert np.isfinite(np.atleast_1d(z)[0])
