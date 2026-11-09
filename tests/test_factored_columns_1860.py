"""The near-interface column twin's factored route (momwire#1860 L1, the
opt-in feature `factored_columns`).

A grid-shaped column -- many members over few distinct heights z and depths
z', as the coaxial rod x vertical column of a buried rod under a vertical is
-- takes e^{-gamma_p z} once per distinct z and e^{gamma_m z'} once per
distinct z' and forms each member's exponential as their product
(`mw899::Grid`, `_near_interface_core.cpp`), where the member route takes
one complex exponential per node per member.

What is held:

- off, nothing changes: the feature is off by default, and the binding's
  default is the member route;
- on, a grid-shaped column is a few ulp of its vector scale from the member
  route, and is NOT bit-identical to it (the route ran);
- a column that is not grid-shaped answers bit for bit as the member route,
  flag on or off;
- a member's answer is its own: permuting a column's members (which changes
  each member's group) and changing the thread count (which changes the
  parallel unit) move no bit;
- on, a buried crossing deck (a rod under a vertical) asks the twin for the
  route on both nec5-bspline and nec5-razor-2p, and its Z agrees with off.
"""

from __future__ import annotations

import numpy as np
import pytest

from momwire import _features, _ground_refl
from momwire import _near_interface as ni
from momwire.bspline import BSplineSolver
from momwire.razor import RazorSolver

from test_crossing_serve_524 import crossing_deck

# An experimental feature (`momwire._features`, reach 1).
pytestmark = [
    pytest.mark.reach(1),
    pytest.mark.skipif(
        ni._nia is None or not hasattr(ni._nia, "six_columns_grid_1860"),
        reason="the near-interface accelerator is not built",
    ),
]

C0 = 299792458.0
F = 3.6e6
K_P = 2.0 * np.pi * F / C0
SOIL = _ground_refl.eps_tilde((13.0, 0.005), 2.0 * np.pi * F, 8.8541878128e-12)
K_M = ni.k_medium(complex(SOIL), K_P)


def _columns(columns, *, n_threads=1, **kw):
    rho = np.asarray([c[0] for c in columns], dtype=float)
    z = np.concatenate([np.asarray(c[1], float) for c in columns])
    zp = np.concatenate([np.asarray(c[2], float) for c in columns])
    offsets = np.zeros(len(columns) + 1, dtype=np.intp)
    offsets[1:] = np.cumsum([len(c[1]) for c in columns])
    return np.asarray(
        ni._nia.near_interface_six_columns(
            K_P,
            K_M,
            rho,
            offsets,
            np.ascontiguousarray(z),
            np.ascontiguousarray(zp),
            ni._LAM_MULT,
            ni._COLUMN_P,
            ni._DETOUR,
            n_threads,
            ni._GX,
            ni._GW,
            **kw,
        )
    )


def _coaxial(rho=0.008, nz=24, nzp=9):
    """Every height of a vertical over every depth of a rod below it."""
    zs = np.linspace(0.02, 10.0, nz)
    zps = np.linspace(-3.0, -0.01, nzp)
    z, zp = np.meshgrid(zs, zps, indexing="ij")
    return (rho, z.ravel(), zp.ravel())


def _scale_rel(got, ref):
    """Worst member's error on its own six-kernel vector scale."""
    num = np.abs(got - ref).max(axis=1)
    den = np.abs(ref).max(axis=1)
    return float(np.max(num / den))


def test_off_by_default(monkeypatch):
    monkeypatch.delenv("MOMWIRE_X_FACTORED_COLUMNS", raising=False)
    assert not _features.enabled("factored_columns")
    assert not BSplineSolver(**crossing_deck())._factored_columns
    assert not RazorSolver(**crossing_deck())._factored_columns
    assert "grid" not in ni._columns_lanes_kw()


def test_the_binding_default_is_the_member_route():
    col = [_coaxial()]
    assert np.array_equal(_columns(col), _columns(col, grid=False))


def test_a_grid_column_is_the_member_route_to_a_few_ulp_and_not_its_bits():
    col = [_coaxial(), _coaxial(rho=0.001, nz=40, nzp=6)]
    off = _columns(col)
    on = _columns(col, grid=True)
    assert not np.array_equal(on, off)  # the route ran
    assert _scale_rel(on, off) < 1e-12


def test_a_column_that_is_not_grid_shaped_keeps_the_member_route_bit_for_bit():
    rng = np.random.default_rng(1860)
    n = 200  # all distinct z and z': one value per member, no gain
    col = [(0.3, rng.uniform(0.01, 5.0, n), rng.uniform(-3.0, -0.01, n))]
    small = [_coaxial(nz=6, nzp=6)]  # 36 members: under the minimum
    for c in (col, small):
        assert np.array_equal(_columns(c, grid=True), _columns(c))


def test_a_members_answer_is_its_own_in_any_group_and_any_thread_count():
    rho, z, zp = _coaxial(nz=31, nzp=7)  # 217 members: groups and a remainder
    ref = _columns([(rho, z, zp)], grid=True)
    perm = np.random.default_rng(7).permutation(z.size)
    got = _columns([(rho, z[perm], zp[perm])], grid=True)
    assert np.array_equal(got, ref[perm])
    # Four threads: the column is over one thread's share, so it is split
    # by member (the table filled on every thread) rather than run as one
    # unit.
    assert np.array_equal(_columns([(rho, z, zp)], grid=True, n_threads=4), ref)


def _z(cls, flag):
    s = cls(**crossing_deck(), factored_columns=flag)
    return 1.0 / complex(s.compute_port_solution().y.ravel()[0])


# The solver asks `_features.enabled`, which refuses a native feature outside
# its reach (macOS, sse2), so only this test needs the platform's level.
@pytest.mark.skipif(
    not _features.certified("factored_columns"),
    reason="factored_columns is not certified on this platform",
)
# Slow lane at the 0.89.0 pre-tag check: 5.4 s median on the Linux test job.
@pytest.mark.slow
@pytest.mark.parametrize("cls", [BSplineSolver, RazorSolver])
def test_on_a_crossing_deck_takes_the_route_and_z_agrees(monkeypatch, cls):
    calls = []
    real = ni._nia.near_interface_six_columns

    def counted(*a, **kw):
        calls.append(bool(kw.get("grid", False)))
        return real(*a, **kw)

    monkeypatch.setattr(ni._nia, "near_interface_six_columns", counted)
    z_off = _z(cls, False)
    assert calls and not any(calls)
    calls.clear()
    z_on = _z(cls, True)
    assert calls and all(calls)
    assert abs(z_on - z_off) < 1e-11 * abs(z_off)
