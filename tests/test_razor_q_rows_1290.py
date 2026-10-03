"""Razor's composing-ground remainder rows in C++ (momwire#1290,
`razor_q_rows` in `_accel_razor_t2.cpp`) against the numpy spelling.

Each row window's Q took the projected field's two moments through T1's
wing algebra in numpy: four (n_obs, n_basis) complex gathers and as many
temporaries, then a reduction over the path, from the window's gathered
(n_obs, n_seg, 2) moments. The kernel reads each moment where it lies --
the window's distinct observers' rows, through a row map -- and writes each
entry once, in numpy's operation order (its header derives that the bits
are numpy's, signed zeros aside).

Gated here: the kernel against `razor._q_rows_numpy` entry by entry under a
bound derived from the operands (not `==`: the numpy side's bits belong to
numpy's build), whole buried and above-ground fills with the kernel on and
off at the T2 module's route bar, and that the kernel actually ran.
"""

from __future__ import annotations

import numpy as np
import pytest

from momwire import _accel
from momwire import razor as _razor
from momwire.razor import RazorSolver

from test_crossing_serve_524 import hub_deck, invl_deck

pytestmark = pytest.mark.skipif(
    not _razor._HAVE_RAZOR_Q_ACCEL, reason="accelerator without razor_q_rows_1290"
)

EPS = np.finfo(float).eps
BAR_ROUTE = 1e-13


def _operands(rng, n_u, n_seg, n_basis, n_path, n_rows):
    f = rng.standard_normal((n_u, n_seg, 2)) + 1j * rng.standard_normal((n_u, n_seg, 2))
    f *= 10.0 ** rng.uniform(-4, 4, f.shape)
    f[rng.random(f.shape) < 0.02] = 0.0
    n_obs = n_rows * n_path
    obs_row = rng.integers(0, n_u, n_obs)
    s_a = rng.integers(0, n_seg, n_basis)
    s_b = rng.integers(0, n_seg, n_basis)
    h_a = rng.uniform(0.01, 2.0, n_basis)
    h_b = rng.uniform(0.01, 2.0, n_basis)
    fall_a = rng.random(n_basis) < 0.3
    fall_b = rng.random(n_basis) < 0.7
    sig_a = rng.choice([-1.0, 0.0, 1.0], n_basis)
    sig_b = rng.choice([-1.0, 1.0], n_basis)
    w = rng.uniform(0.0, 1.0, n_obs)
    return f, obs_row, s_a, s_b, h_a, h_b, fall_a, fall_b, sig_a, sig_b, w


@pytest.mark.parametrize("n_path", [2, 3])
def test_the_kernel_is_the_numpy_spelling(n_path):
    rng = np.random.default_rng(1290 + n_path)
    f, obs_row, s_a, s_b, h_a, h_b, fa, fb, ga, gb, w = _operands(
        rng, 17, 31, 23, n_path, 11
    )
    got = _accel.acc.razor_q_rows(
        f, obs_row, s_a, s_b, h_a, h_b, fa, fb, ga, gb, w, n_path
    )
    want = _razor._q_rows_numpy(
        f[obs_row],
        s_a,
        s_b,
        h_a,
        h_b,
        np.flatnonzero(fa),
        np.flatnonzero(fb),
        ga,
        gb,
        w,
        n_path,
    )
    assert got.shape == want.shape == (11, 23)
    # Every entry is a fixed short sequence of roundings of the same
    # operands; a build that orders or fuses differently moves it by a few
    # units in the last place of the terms that make it.
    A = np.abs(f[obs_row])

    def wing(s, h, fall):
        m = A[:, s, 1] / h[None, :]
        m[:, fall] += A[:, s[fall], 0]
        return m

    mag = np.abs(ga)[None, :] * wing(s_a, h_a, fa) + np.abs(gb)[None, :] * wing(
        s_b, h_b, fb
    )
    mag = (mag * w[:, None]).reshape(-1, n_path, s_a.size).sum(axis=1)
    bound = 16 * EPS * mag
    assert np.all(np.abs(got.real - want.real) <= bound)
    assert np.all(np.abs(got.imag - want.imag) <= bound)


def test_the_kernel_refuses_an_observer_row_off_the_table():
    rng = np.random.default_rng(3)
    f, obs_row, s_a, s_b, h_a, h_b, fa, fb, ga, gb, w = _operands(rng, 4, 5, 3, 2, 2)
    obs_row[1] = 4
    with pytest.raises(ValueError, match="observer row out of range"):
        _accel.acc.razor_q_rows(f, obs_row, s_a, s_b, h_a, h_b, fa, fb, ga, gb, w, 2)


C0 = 299792458.0
LAM = C0 / 7.0e6


def _above():
    return dict(
        wires=[np.array([(-5.0, 0.0, 6.0), (5.0, 0.0, 6.0)])],
        n_per_edge_per_wire=[[21]],
        feeds=[(0, 5.0, 1 + 0j)],
        wavelength=LAM,
        wire_radius=0.001,
        ground_z=0.0,
        ground_eps=(13.0, 0.005),
        ground_model="sommerfeld",
    )


DECKS = {
    "buried": lambda: hub_deck(n_radials=4),
    "invl": lambda: invl_deck(n_radials=4),
    "above": _above,
}


def _z(deck):
    s = RazorSolver(**DECKS[deck](), nec5_quadrature=True)
    return s._assemble_Z(s._build_geometry(), s.k)


@pytest.mark.parametrize("deck", sorted(DECKS))
def test_the_fill_is_the_numpy_routes(deck, monkeypatch):
    monkeypatch.setattr(_razor, "_HAVE_RAZOR_Q_ACCEL", False)
    ref = _z(deck)
    monkeypatch.setattr(_razor, "_HAVE_RAZOR_Q_ACCEL", True)
    calls = []
    real = _accel.acc.razor_q_rows

    def counted(*a, **k):
        calls.append(1)
        return real(*a, **k)

    monkeypatch.setattr(_accel.acc, "razor_q_rows", counted)
    got = _z(deck)
    assert calls, "the kernel never ran"
    assert np.abs(got - ref).max() <= BAR_ROUTE * np.abs(ref).max()
