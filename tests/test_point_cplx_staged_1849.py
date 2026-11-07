"""The point-matched sinusoidal fill's complex-k field tensor on the complex
Galerkin fill's staged rows (momwire#1849, the opt-in feature
`point_cplx_staged`).

What is held:

- off, nothing changes: the solver defaults to `sinusoidal_field_tensor_cplx`;
- the staged tensor is the scalar one to a few ulp of scale, all three shapes;
- a row is its observer's alone: any observer subset gives the full call's
  rows bit for bit (the sweep is padded per source tile);
- on, a buried crossing deck takes the staged kernel and its Z agrees with
  the flag off.

Both shape sets: the folded cos k(xi) - 1 the staged rows assemble, and the
literal cosine the solver takes on any mesh with |k|Delta >= 1.5e-3.
"""

from __future__ import annotations

import numpy as np
import pytest

from momwire import _features
from momwire._accel import acc
from momwire.sinusoidal import SinusoidalSolver
from test_crossing_serve_524 import hub_deck

# An experimental feature (`momwire._features`, reach 1).
pytestmark = [
    pytest.mark.reach(1),
    pytest.mark.skipif(
        acc is None or not hasattr(acc, "point_field_tensor_cplx_staged"),
        reason="the accelerator is not built",
    ),
]

K_M = 0.5872 - 0.2387j


def _screen(n_radials=8, n_seg=12, length=6.0, depth=0.15):
    c, t, h = [], [], []
    for r in range(n_radials):
        th = 2 * np.pi * r / n_radials
        u = np.array([np.cos(th), np.sin(th), 0.0])
        edges = np.linspace(0.05, length, n_seg + 1)
        for i in range(n_seg):
            c.append(u * 0.5 * (edges[i] + edges[i + 1]) + [0.0, 0.0, -depth])
            t.append(u)
            h.append(edges[i + 1] - edges[i])
    return np.ascontiguousarray(c), np.ascontiguousarray(t), np.asarray(h)


def _args(rows=slice(None)):
    c, t, h = _screen()
    gx, gw = np.polynomial.legendre.leggauss(8)
    a = np.full(len(c), 0.001)
    eta = 376.730313668 / (K_M / 0.1488)
    return (c[rows], t[rows], a[rows], c, t, h, K_M, eta, gx, gw)


def test_off_by_default(monkeypatch):
    monkeypatch.delenv("MOMWIRE_X_POINT_CPLX_STAGED", raising=False)
    assert not _features.enabled("point_cplx_staged")
    assert not SinusoidalSolver(**hub_deck())._point_cplx_staged


@pytest.mark.parametrize("folded", [True, False])
def test_the_staged_tensor_is_the_scalar_one_to_a_few_ulp_of_scale(folded):
    old = acc.sinusoidal_field_tensor_cplx(*_args(), folded)
    new = acc.point_field_tensor_cplx_staged(*_args(), folded)
    for o, n in zip(old, new, strict=True):
        assert np.max(np.abs(n - o)) <= 4e-15 * np.max(np.abs(o))


@pytest.mark.parametrize("folded", [True, False])
def test_a_row_is_its_observers_alone(folded):
    full = acc.point_field_tensor_cplx_staged(*_args(), folded)
    part = acc.point_field_tensor_cplx_staged(*_args(slice(37, 61)), folded)
    for f, p in zip(full, part, strict=True):
        assert np.array_equal(f[37:61], p)


def _z(flag):
    s = SinusoidalSolver(**hub_deck(), point_cplx_staged=flag)
    return 1.0 / complex(s.compute_port_solution().y.ravel()[0])


# The solver asks `_features.enabled`, which refuses a native feature outside
# its reach (macOS, sse2), so only this test needs the platform's level; the
# kernel tests above call the binding directly.
@pytest.mark.skipif(
    not _features.certified("point_cplx_staged"),
    reason="point_cplx_staged is not certified on this platform",
)
def test_on_a_buried_deck_takes_the_staged_kernel_and_z_agrees(monkeypatch):
    calls = []
    real = acc.point_field_tensor_cplx_staged

    def counted(*a):
        calls.append(len(a[0]))
        return real(*a)

    monkeypatch.setattr(acc, "point_field_tensor_cplx_staged", counted)
    z_off = _z(False)
    assert not calls
    z_on = _z(True)
    assert calls
    assert abs(z_on - z_off) < 1e-11 * abs(z_off)
